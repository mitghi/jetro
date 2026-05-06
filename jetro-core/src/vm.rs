//! Bytecode compiler and stack-machine VM for Jetro expressions.
//!
//! `Compiler` lowers an `Expr` AST to a flat `Arc<[Opcode]>` program and runs
//! peephole passes (`RootChain` fusion, `FilterCount` fusion, `ConstFold`,
//! demand annotation). `VM` owns two caches: a compile cache keyed on the
//! expression string, and a path-resolution cache keyed on document structure.
//! Both caches accumulate over the thread's lifetime via the thread-local in
//! `lib.rs`. The VM is the general scalar fallback; streamable chains are
//! handled by the pipeline IR in `pipeline.rs` / `composed.rs`.

use indexmap::IndexMap;
use smallvec::SmallVec;
use std::{
    collections::hash_map::DefaultHasher,
    collections::{HashMap, VecDeque},
    hash::{Hash, Hasher},
    sync::atomic::{AtomicU64, Ordering},
    sync::Arc,
};

use crate::ast::*;
pub use crate::builtins::BuiltinMethod;
use crate::context::{Env, EvalError};
use crate::data::runtime::call_builtin_method_compiled;
use crate::util::{
    add_vals, cmp_vals_binop, is_truthy, kind_matches, num_op, obj2, val_to_key, val_to_string,
    vals_eq,
};
use crate::data::value::Val;

/// Pop the top of the operand stack, returning a `stack underflow` error if empty.
macro_rules! pop {
    ($stack:expr) => {
        $stack
            .pop()
            .ok_or_else(|| EvalError("stack underflow".into()))?
    };
}
/// Construct an `Err(EvalError(...))` from a format string, mirroring `format!` syntax.
macro_rules! err {
    ($($t:tt)*) => { Err(EvalError(format!($($t)*))) };
}


/// A method call compiled into a `CallMethod` opcode. Lambda/arg bodies are
/// pre-compiled into `sub_progs` exactly once at compile time so the inner
/// loop never re-compiles them. `demand_max_keep` is set by the demand-pass
/// peephole when a `take(n)` follows this call.
#[derive(Debug, Clone)]
pub struct CompiledCall {
    /// Resolved built-in variant; `Unknown` when the name is not a built-in.
    pub method: BuiltinMethod,
    /// Original method name from source, used for error messages and registry lookup.
    pub name: Arc<str>,
    /// Pre-compiled sub-programs for each lambda/expression argument; shared via `Arc`.
    pub sub_progs: Arc<[Arc<Program>]>,
    /// Original un-compiled arguments kept for lambda-param introspection at runtime.
    pub orig_args: Arc<[Arg]>,
    /// When set, `filter`/`map` may stop early after collecting this many results.
    pub demand_max_keep: Option<usize>,
}


/// A field entry inside a `MakeObj` opcode. `Short` is the fast path for
/// `{name}` shorthand — reads from `current` using an inline-cache hint;
/// `KvPath` is the structural fast path for `{key: $.a.b}` chains.
#[derive(Debug, Clone)]
pub enum CompiledObjEntry {
    /// `{name}` shorthand: copies the field from `current`, using `ic` as an
    /// inline-cache slot to remember the last-seen map index.
    Short {
        /// Field name to read from the current object (or variable scope).
        name: Arc<str>,
        /// Inline-cache slot storing the last successful map index + 1 (0 = cold).
        ic: Arc<AtomicU64>,
    },
    /// General `{key: expr}` entry with an optional guard condition.
    Kv {
        /// Output key name.
        key: Arc<str>,
        /// Compiled value expression evaluated against the current environment.
        prog: Arc<Program>,
        /// When true, null values are omitted from the output object.
        optional: bool,
        /// If present, the entry is skipped when the condition evaluates falsy.
        cond: Option<Arc<Program>>,
    },
    /// Structural fast-path for `{key: @.a.b[0]}` — avoids spawning a sub-`exec` call
    /// when the value is a pure chain of field/index steps rooted at `@`.
    KvPath {
        /// Output key name.
        key: Arc<str>,
        /// Ordered field/index steps traversed starting from `current`.
        steps: Arc<[KvStep]>,
        /// When true, null values are omitted from the output object.
        optional: bool,
        /// Per-step inline-cache slots, one per element of `steps`.
        ics: Arc<[AtomicU64]>,
    },
    /// `{(expr): expr}` — both key and value are computed at runtime.
    Dynamic {
        /// Program producing the key; result is coerced to a string.
        key: Arc<Program>,
        /// Program producing the value.
        val: Arc<Program>,
    },
    /// `{...expr}` — shallow-merges all fields from an object value into the output.
    Spread(Arc<Program>),
    /// `{...!expr}` — recursively deep-merges an object value into the output.
    SpreadDeep(Arc<Program>),
}


/// A single traversal step in a `KvPath` entry, representing either a named
/// field access or an integer index into an array.
#[derive(Debug, Clone)]
pub enum KvStep {
    /// Access an object field by name.
    Field(Arc<str>),
    /// Access an array element; negative values count from the end.
    Index(i64),
}


/// A single segment of a compiled format-string (`f"..."`).
/// Segments alternate between literal text and interpolated expressions.
#[derive(Debug, Clone)]
pub enum CompiledFSPart {
    /// A verbatim string fragment that is appended directly to the output buffer.
    Lit(Arc<str>),
    /// An interpolated expression whose result is formatted and inserted at this position.
    Interp {
        /// The compiled sub-program that produces the interpolated value.
        prog: Arc<Program>,
        /// Optional format spec such as `.2f` or `>10`; `None` uses default formatting.
        fmt: Option<FmtSpec>,
    },
}


/// Specifies the destructuring pattern for an object bind step in a pipeline
/// (`... | {a, b, ...rest} -> ...`).
#[derive(Debug, Clone)]
pub struct BindObjSpec {
    /// Named fields that are extracted as individual variables.
    pub fields: Arc<[Arc<str>]>,
    /// If present, remaining fields are collected into this variable as an object.
    pub rest: Option<Arc<str>>,
}


/// A single compiled step inside a `PipelineRun` opcode. Each step either
/// transforms the current pipeline value or captures it into named variables.
#[derive(Debug, Clone)]
pub enum CompiledPipeStep {
    /// Pass the current value through an expression, updating the pipeline value.
    Forward(Arc<Program>),
    /// Bind the current pipeline value to a single named variable.
    BindName(Arc<str>),
    /// Destructure the current object value into named field variables (with optional rest).
    BindObj(Arc<BindObjSpec>),
    /// Destructure the current array value into positional variables by index.
    BindArr(Arc<[Arc<str>]>),
}


/// Compiled specification for a list, set, or generator comprehension
/// (`[expr for vars in iter if cond]`).
#[derive(Debug, Clone)]
pub struct CompSpec {
    /// Expression evaluated for each item to produce the output element.
    pub expr: Arc<Program>,
    /// Variable names bound per iteration; one name for simple loops, two for indexed.
    pub vars: Arc<[Arc<str>]>,
    /// Program whose result is iterated (must yield an array or object).
    pub iter: Arc<Program>,
    /// Optional filter; items for which this evaluates falsy are skipped.
    pub cond: Option<Arc<Program>>,
}

/// Compiled specification for a dictionary comprehension
/// (`{key: val for vars in iter if cond}`).
#[derive(Debug, Clone)]
pub struct DictCompSpec {
    /// Program evaluated to produce each output key; coerced to a string.
    pub key: Arc<Program>,
    /// Program evaluated to produce each output value.
    pub val: Arc<Program>,
    /// Variable names bound per iteration.
    pub vars: Arc<[Arc<str>]>,
    /// Program whose result is iterated (must yield an array or object).
    pub iter: Arc<Program>,
    /// Optional filter; items for which this evaluates falsy are skipped.
    pub cond: Option<Arc<Program>>,
}


/// Single instruction in a compiled `Program`. The VM executes a flat
/// `Arc<[Opcode]>` slice iteratively; no per-opcode stack frames.
#[derive(Debug, Clone)]
pub enum Opcode {
    /// Push the literal `null` value onto the stack.
    PushNull,
    /// Push a boolean literal onto the stack.
    PushBool(bool),
    /// Push a 64-bit integer literal onto the stack.
    PushInt(i64),
    /// Push a 64-bit float literal onto the stack.
    PushFloat(f64),
    /// Push a reference-counted string literal onto the stack.
    PushStr(Arc<str>),

    /// Push the root document value (`$`) onto the stack.
    PushRoot,
    /// Push the current iteration value (`@`) onto the stack.
    PushCurrent,

    /// Pop an object, push the named field (or `null` if absent).
    GetField(Arc<str>),
    /// Pop an array/string, push element at the given index; negative indices count from end.
    GetIndex(i64),
    /// Pop an array, push a sub-slice between the optional start and end indices.
    GetSlice(Option<i64>, Option<i64>),
    /// Pop a container; evaluate the inner program to get a key, then index into the container.
    DynIndex(Arc<Program>),
    /// Like `GetField` but propagates `null` receivers silently instead of erroring.
    OptField(Arc<str>),
    /// Collect all descendants matching the given field name via DFS; result is an array.
    Descendant(Arc<str>),
    /// Collect every scalar and object node in the subtree into an array (DFS pre-order).
    DescendAll,
    /// Filter an array or singleton using the predicate sub-program; `@` is each item.
    InlineFilter(Arc<Program>),
    /// Apply a quantifier to the top-of-stack value (`?` for first, `!` for exactly-one).
    Quantifier(QuantifierKind),

    /// Fused `PushRoot` + one-or-more `GetField` steps; avoids repeated stack traffic.
    /// Results are memoised in `root_chain_cache` keyed by `Arc` pointer identity.
    RootChain(Arc<[Arc<str>]>),

    /// Fused run of consecutive `GetField`/`OptField` steps after a non-root value.
    /// Each step has its own inline-cache slot inside `FieldChainData`.
    FieldChain(Arc<FieldChainData>),

    /// Resolve an identifier: looks up a variable, falls back to a field on `current`.
    LoadIdent(Arc<str>),

    /// Pop two values and push their sum (number or string concatenation).
    Add,
    /// Pop two numbers and push their difference.
    Sub,
    /// Pop two numbers and push their product.
    Mul,
    /// Pop two numbers and push their quotient as a float; errors on divide-by-zero.
    Div,
    /// Pop two numbers and push the remainder.
    Mod,
    /// Pop two values and push `true` if they are equal.
    Eq,
    /// Pop two values and push `true` if they are not equal.
    Neq,
    /// Pop two values and push `true` if the left is strictly less than the right.
    Lt,
    /// Pop two values and push `true` if the left is less than or equal to the right.
    Lte,
    /// Pop two values and push `true` if the left is strictly greater than the right.
    Gt,
    /// Pop two values and push `true` if the left is greater than or equal to the right.
    Gte,
    /// Pop two string values and push `true` if either contains the other (case-insensitive).
    Fuzzy,
    /// Pop a value and push its boolean negation.
    Not,
    /// Pop a number and push its arithmetic negation.
    Neg,

    /// Pop a value and push it cast to the given type.
    CastOp(super::ast::CastType),

    /// Short-circuit AND: pop lhs; if falsy push `false`, else evaluate rhs sub-program.
    AndOp(Arc<Program>),
    /// Short-circuit OR: pop lhs; if truthy push it, else evaluate rhs sub-program.
    OrOp(Arc<Program>),
    /// Null coalescing: pop lhs; if non-null push it, else evaluate rhs sub-program.
    CoalesceOp(Arc<Program>),

    /// Pop receiver and dispatch it with the pre-compiled call descriptor.
    CallMethod(Arc<CompiledCall>),
    /// Like `CallMethod` but silently returns `null` when the receiver is `null`.
    CallOptMethod(Arc<CompiledCall>),

    /// Construct an object literal from the compiled field entries; does not consume the stack.
    MakeObj(Arc<[CompiledObjEntry]>),

    /// Construct an array literal; each element has a compiled program and a spread flag.
    MakeArr(Arc<[(Arc<Program>, bool)]>),

    /// Evaluate a format-string from its compiled parts and push the resulting string.
    FString(Arc<[CompiledFSPart]>),

    /// Pop a value and push a boolean indicating whether it matches the given kind.
    KindCheck {
        /// The kind to test against (e.g. `KindType::Arr`).
        ty: KindType,
        /// When true the boolean result is inverted (`is not`).
        negate: bool,
    },

    /// No-op marker inserted by the pipeline emitter to delimit scope boundaries;
    /// consumed during compilation and stripped from the final program.
    SetCurrent,

    /// Evaluate `base`, then run each `CompiledPipeStep` in sequence, threading
    /// the current value and environment through the pipeline.
    PipelineRun {
        /// The left-hand-side expression whose result seeds the pipeline.
        base: Arc<Program>,
        /// Ordered pipeline steps (forward transforms and bind patterns).
        steps: Arc<[CompiledPipeStep]>,
    },

    /// Pop the initialiser off the stack, bind it to `name`, and evaluate `body`.
    LetExpr {
        /// Variable name introduced for the duration of `body`.
        name: Arc<str>,
        /// Body program evaluated in the extended environment.
        body: Arc<Program>,
    },

    /// Pop the condition, then evaluate either `then_` or `else_` branch.
    IfElse {
        /// Branch evaluated when the condition is truthy.
        then_: Arc<Program>,
        /// Branch evaluated when the condition is falsy.
        else_: Arc<Program>,
    },

    /// Evaluate `body`; on error or null result fall back to `default`.
    TryExpr {
        /// Primary expression to attempt.
        body: Arc<Program>,
        /// Fallback expression evaluated when `body` fails or returns null.
        default: Arc<Program>,
    },
    /// Execute a list comprehension using the given compiled spec.
    ListComp(Arc<CompSpec>),
    /// Execute a dictionary comprehension using the given compiled spec.
    DictComp(Arc<DictCompSpec>),
    /// Execute a set/generator comprehension (deduplication semantics).
    SetComp(Arc<CompSpec>),

    /// Execute a compiled patch expression (`.set`, `.modify`, `.delete`, `.unset`).
    PatchEval(Arc<CompiledPatch>),

    /// Guard that fires when a `DELETE` sentinel reaches execution outside a patch context.
    DeleteMarkErr,
}


/// A compiled, immutable bytecode program. Shared between the compile cache and
/// the path-resolution cache via `Arc`; cloning is O(1).
#[derive(Debug, Clone)]
pub struct Program {
    /// The flat opcode slice executed by the VM.
    pub ops: Arc<[Opcode]>,
    /// The source expression string this program was compiled from; used for cache keys.
    pub source: Arc<str>,
    /// Stable hash of `source` used for fast equality checks.
    pub id: u64,

    /// `true` when every opcode is a pure structural navigation step; allows the
    /// path-resolution cache to memoize results for this program.
    pub is_structural: bool,

    /// Per-opcode inline-cache slots (one `AtomicU64` per opcode); used by `GetField`
    /// and `OptField` to remember the last-seen map index.
    pub ics: Arc<[AtomicU64]>,
}


/// Internal return value from the recursive patch walker indicating whether the
/// target node should be replaced with a new value or removed entirely.
#[derive(Debug)]
enum PatchResult {
    /// The node at this path is replaced with the given value.
    Replace(Val),
    /// The node at this path is deleted from its parent container.
    Delete,
}

/// Resolve a signed index `i` into the range `[0, len]`, clamping out-of-bounds
/// values. Negative indices count backwards from `len` (Python-style).
#[inline]
fn vm_resolve_idx(i: i64, len: i64) -> usize {
    if len == 0 {
        return 0;
    }
    let r = if i < 0 { len + i } else { i };
    if r < 0 {
        0
    } else if r >= len {
        len as usize
    } else {
        r as usize
    }
}


/// A compiled `patch` expression: a root document program plus a list of
/// individual field-mutation operations applied in order.
#[derive(Debug, Clone)]
pub struct CompiledPatch {
    /// Program that yields the base document to patch.
    pub root_prog: Arc<Program>,
    /// Ordered list of compiled field operations applied to the document.
    pub ops: Vec<CompiledPatchOp>,
}

/// A single field-mutation within a `CompiledPatch`: a path, a replacement/delete
/// value, and an optional runtime guard condition.
#[derive(Debug, Clone)]
pub struct CompiledPatchOp {
    /// Sequence of path steps that locate the target node in the document.
    pub path: Vec<CompiledPathStep>,
    /// The value action to perform at the target: replace or delete.
    pub val: CompiledPatchVal,
    /// When present, the operation is skipped unless this program evaluates truthy.
    pub cond: Option<Arc<Program>>,
}

/// The replacement action for a single `CompiledPatchOp`.
#[derive(Debug, Clone)]
pub enum CompiledPatchVal {
    /// Replace the node with the result of evaluating this program; `@` is the old value.
    Replace(Arc<Program>),
    /// Remove the node from its parent (object field or array element).
    Delete,
}

/// A single step in the path portion of a `CompiledPatchOp`.
#[derive(Debug, Clone)]
pub enum CompiledPathStep {
    /// Navigate into an object field by name.
    Field(Arc<str>),
    /// Navigate into an array element by signed index.
    Index(i64),
    /// Navigate to an array element whose index is computed at runtime.
    DynIndex(Arc<Program>),
    /// Apply the operation to every element of an array (`[*]`).
    Wildcard,
    /// Apply the operation to every array element that satisfies the predicate.
    WildcardFilter(Arc<Program>),
    /// Recursively descend and apply the operation wherever the named field exists.
    Descendant(Arc<str>),
}

impl Program {
    /// Construct a new `Program`, computing `is_structural` and allocating fresh inline-cache slots.
    pub fn new(ops: Vec<Opcode>, source: &str) -> Self {
        let id = hash_str(source);
        let is_structural = ops.iter().all(|op| {
            matches!(
                op,
                Opcode::PushRoot
                    | Opcode::PushCurrent
                    | Opcode::GetField(_)
                    | Opcode::GetIndex(_)
                    | Opcode::GetSlice(..)
                    | Opcode::OptField(_)
                    | Opcode::RootChain(_)
                    | Opcode::FieldChain(_)
            )
        });
        let ics = fresh_ics(ops.len());
        Self {
            ops: ops.into(),
            source: source.into(),
            id,
            is_structural,
            ics,
        }
    }
}


/// Cached pointer-path data for a `FieldChain` opcode. Stores the ordered field
/// keys and one inline-cache slot per key for fast map-index lookup.
#[derive(Debug)]
pub struct FieldChainData {
    /// Ordered sequence of field names traversed by this chain.
    pub keys: Arc<[Arc<str>]>,
    /// Per-key inline-cache slots; each stores the last-seen index + 1 (0 = cold).
    pub ics: Box<[AtomicU64]>,
}

impl FieldChainData {
    /// Allocate a new `FieldChainData` with cold (zero) inline-cache slots.
    pub fn new(keys: Arc<[Arc<str>]>) -> Self {
        let n = keys.len();
        let mut ics = Vec::with_capacity(n);
        for _ in 0..n {
            ics.push(AtomicU64::new(0));
        }
        Self {
            keys,
            ics: ics.into_boxed_slice(),
        }
    }
}

/// Deref to the key slice so callers can iterate keys without naming the field.
impl std::ops::Deref for FieldChainData {
    type Target = [Arc<str>];
    #[inline]
    fn deref(&self) -> &[Arc<str>] {
        &self.keys
    }
}


/// Allocate `len` cold inline-cache slots (all zero) and return them as a shared slice.
pub fn fresh_ics(len: usize) -> Arc<[AtomicU64]> {
    let mut v = Vec::with_capacity(len);
    for _ in 0..len {
        v.push(AtomicU64::new(0));
    }
    v.into()
}


/// Return `true` when the `JETRO_DISABLE_OPCODE_FUSION` environment variable is set,
/// suppressing all peephole fusion passes for debugging purposes.
#[inline]
pub(crate) fn disable_opcode_fusion() -> bool {
    std::env::var_os("JETRO_DISABLE_OPCODE_FUSION").is_some()
}

/// Look up `key` in the map using the inline-cache `ic` as a fast-path hint.
/// On a cache miss the map is searched linearly and the cache is updated.
/// Returns `Val::Null` when the key is absent.
fn ic_get_field(m: &Arc<IndexMap<Arc<str>, Val>>, key: &str, ic: &AtomicU64) -> Val {
    let cached = ic.load(Ordering::Relaxed);
    if cached != 0 {
        let slot = (cached - 1) as usize;
        if let Some((k, v)) = m.get_index(slot) {
            if k.as_ref() == key {
                return v.clone();
            }
        }
    }
    if let Some((idx, _, v)) = m.get_full(key) {
        ic.store((idx as u64) + 1, Ordering::Relaxed);
        v.clone()
    } else {
        Val::Null
    }
}


/// Simple binary arithmetic operation used by specialised accumulate/transform paths.
#[derive(Copy, Clone)]
enum AccumOp {
    /// Wrapping integer addition or float addition.
    Add,
    /// Wrapping integer subtraction or float subtraction.
    Sub,
    /// Wrapping integer multiplication or float multiplication.
    Mul,
}


/// Sum a slice of `i64` values using 4-lane manual unrolling to assist auto-vectorisation.
#[inline]
fn simd_sum_i64_slice(a: &[i64]) -> i64 {
    let mut s0: i64 = 0;
    let mut s1: i64 = 0;
    let mut s2: i64 = 0;
    let mut s3: i64 = 0;
    let chunks = a.chunks_exact(4);
    let rem = chunks.remainder();
    for c in chunks {
        s0 = s0.wrapping_add(c[0]);
        s1 = s1.wrapping_add(c[1]);
        s2 = s2.wrapping_add(c[2]);
        s3 = s3.wrapping_add(c[3]);
    }
    let mut tail: i64 = 0;
    for v in rem {
        tail = tail.wrapping_add(*v);
    }
    s0.wrapping_add(s1)
        .wrapping_add(s2)
        .wrapping_add(s3)
        .wrapping_add(tail)
}

/// Sum a slice of `f64` values using 4-lane manual unrolling to assist auto-vectorisation.
#[inline]
fn simd_sum_f64_slice(a: &[f64]) -> f64 {
    let mut s0: f64 = 0.0;
    let mut s1: f64 = 0.0;
    let mut s2: f64 = 0.0;
    let mut s3: f64 = 0.0;
    let chunks = a.chunks_exact(4);
    let rem = chunks.remainder();
    for c in chunks {
        s0 += c[0];
        s1 += c[1];
        s2 += c[2];
        s3 += c[3];
    }
    let mut tail: f64 = 0.0;
    for v in rem {
        tail += *v;
    }
    s0 + s1 + s2 + s3 + tail
}

/// Return the minimum value in an `i64` slice, or `None` when empty.
#[inline]
fn simd_min_i64_slice(a: &[i64]) -> Option<i64> {
    if a.is_empty() {
        return None;
    }
    let mut best = a[0];
    for v in &a[1..] {
        if *v < best {
            best = *v;
        }
    }
    Some(best)
}
/// Return the maximum value in an `i64` slice, or `None` when empty.
#[inline]
fn simd_max_i64_slice(a: &[i64]) -> Option<i64> {
    if a.is_empty() {
        return None;
    }
    let mut best = a[0];
    for v in &a[1..] {
        if *v > best {
            best = *v;
        }
    }
    Some(best)
}
/// Return the minimum value in an `f64` slice, or `None` when empty.
#[inline]
fn simd_min_f64_slice(a: &[f64]) -> Option<f64> {
    if a.is_empty() {
        return None;
    }
    let mut best = a[0];
    for v in &a[1..] {
        if *v < best {
            best = *v;
        }
    }
    Some(best)
}
/// Return the maximum value in an `f64` slice, or `None` when empty.
#[inline]
fn simd_max_f64_slice(a: &[f64]) -> Option<f64> {
    if a.is_empty() {
        return None;
    }
    let mut best = a[0];
    for v in &a[1..] {
        if *v > best {
            best = *v;
        }
    }
    Some(best)
}

/// Sum a heterogeneous `Val` slice, promoting to `Float` as soon as a float element
/// is encountered. Non-numeric elements are silently skipped.
fn agg_sum_typed(a: &[Val]) -> Val {

    let mut i_acc: i64 = 0;
    let mut it = a.iter();
    while let Some(v) = it.next() {
        match v {
            Val::Int(n) => i_acc = i_acc.wrapping_add(*n),
            Val::Float(x) => {
                let mut f_acc = i_acc as f64 + *x;
                for v in it {
                    match v {
                        Val::Int(n) => f_acc += *n as f64,
                        Val::Float(x) => f_acc += *x,
                        _ => {}
                    }
                }
                return Val::Float(f_acc);
            }
            _ => {} 
        }
    }
    Val::Int(i_acc)
}

/// Compute the arithmetic mean of numeric values in a heterogeneous `Val` slice.
/// Returns `Val::Null` when the slice contains no numeric elements.
#[inline]
fn agg_avg_typed(a: &[Val]) -> Val {
    let mut sum: f64 = 0.0;
    let mut n: usize = 0;
    for v in a {
        match v {
            Val::Int(x) => {
                sum += *x as f64;
                n += 1;
            }
            Val::Float(x) => {
                sum += *x;
                n += 1;
            }
            _ => {}
        }
    }
    if n == 0 {
        Val::Null
    } else {
        Val::Float(sum / n as f64)
    }
}

/// Find the minimum or maximum numeric value in a heterogeneous `Val` slice.
/// When `want_max` is `true` the maximum is returned; otherwise the minimum.
/// Promotes to `Float` if any float element is encountered. Returns `Val::Null` when empty.
#[inline]
fn agg_minmax_typed(a: &[Val], want_max: bool) -> Val {
    let mut it = a.iter();
    
    let first = loop {
        match it.next() {
            Some(v) if v.is_number() => break v,
            Some(_) => continue,
            None => return Val::Null,
        }
    };
    match first {
        Val::Int(n0) => {
            let mut best: i64 = *n0;
            
            while let Some(v) = it.next() {
                match v {
                    Val::Int(n) => {
                        let n = *n;
                        if want_max {
                            if n > best {
                                best = n;
                            }
                        } else {
                            if n < best {
                                best = n;
                            }
                        }
                    }
                    Val::Float(x) => {
                        let x = *x;
                        let mut best_f = best as f64;
                        if want_max {
                            if x > best_f {
                                best_f = x;
                            }
                        } else {
                            if x < best_f {
                                best_f = x;
                            }
                        }
                        for v in it {
                            match v {
                                Val::Int(n) => {
                                    let n = *n as f64;
                                    if want_max {
                                        if n > best_f {
                                            best_f = n;
                                        }
                                    } else {
                                        if n < best_f {
                                            best_f = n;
                                        }
                                    }
                                }
                                Val::Float(x) => {
                                    let x = *x;
                                    if want_max {
                                        if x > best_f {
                                            best_f = x;
                                        }
                                    } else {
                                        if x < best_f {
                                            best_f = x;
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                        return Val::Float(best_f);
                    }
                    _ => {}
                }
            }
            Val::Int(best)
        }
        Val::Float(x0) => {
            let mut best_f: f64 = *x0;
            for v in it {
                match v {
                    Val::Int(n) => {
                        let n = *n as f64;
                        if want_max {
                            if n > best_f {
                                best_f = n;
                            }
                        } else {
                            if n < best_f {
                                best_f = n;
                            }
                        }
                    }
                    Val::Float(x) => {
                        let x = *x;
                        if want_max {
                            if x > best_f {
                                best_f = x;
                            }
                        } else {
                            if x < best_f {
                                best_f = x;
                            }
                        }
                    }
                    _ => {}
                }
            }
            Val::Float(best_f)
        }
        _ => Val::Null,
    }
}


use crate::compiler::{Compiler, PassConfig};



/// LRU path-resolution cache keyed by `(doc_hash, JSON-pointer string)`.
/// Avoids re-traversing the document for repeated structural queries.
struct PathCache {
    /// Nested map: outer key is `doc_hash`, inner key is the slash-delimited pointer path.
    docs: HashMap<u64, HashMap<Arc<str>, Val>>,
    /// Insertion-order deque used to identify and evict the least-recently-used entry.
    order: VecDeque<(u64, Arc<str>)>,
    /// Maximum number of cached entries before eviction begins.
    capacity: usize,
}

impl PathCache {
    /// Create a new `PathCache` with the given capacity limit.
    fn new(cap: usize) -> Self {
        Self {
            docs: HashMap::new(),
            order: VecDeque::with_capacity(cap),
            capacity: cap,
        }
    }

    /// Look up a previously cached value by document hash and path pointer.
    /// Returns `None` when the entry is absent or has been evicted.
    #[inline]
    fn get(&self, doc_hash: u64, ptr: &str) -> Option<Val> {
        self.docs.get(&doc_hash)?.get(ptr).cloned()
    }

    /// Insert a path → value mapping for the given document hash, evicting the
    /// oldest entry first when the cache is at capacity.
    fn insert(&mut self, doc_hash: u64, ptr: Arc<str>, val: Val) {
        if self.order.len() >= self.capacity {
            if let Some((old_hash, old_ptr)) = self.order.pop_front() {
                if let Some(inner) = self.docs.get_mut(&old_hash) {
                    inner.remove(old_ptr.as_ref());
                    if inner.is_empty() {
                        self.docs.remove(&old_hash);
                    }
                }
            }
        }
        self.order.push_back((doc_hash, ptr.clone()));
        self.docs
            .entry(doc_hash)
            .or_insert_with(HashMap::new)
            .insert(ptr, val);
    }

    /// Return the current number of cached entries; available in test builds only.
    #[cfg(test)]
    fn len(&self) -> usize {
        self.order.len()
    }
}



/// Stack-machine VM that owns both a compile cache and a path-resolution cache.
/// Instances are long-lived (typically thread-local) so caches warm up across calls.
pub struct VM {
    /// Maps `(pass_config_hash, expression_string)` → compiled `Program`.
    /// Avoids re-compiling the same expression string with the same pass config.
    compile_cache: HashMap<(u64, String), Arc<Program>>,

    /// LRU order for the compile cache; entries are moved to the back on access.
    compile_lru: std::collections::VecDeque<(u64, String)>,
    /// Maximum number of programs kept in the compile cache.
    compile_cap: usize,
    /// Path-resolution cache for structural navigation results.
    path_cache: PathCache,

    /// Per-exec cache of `RootChain` results keyed by `Arc` pointer identity.
    /// Cleared at the start of each `execute` call to stay consistent with the document.
    root_chain_cache: HashMap<usize, Val>,

    /// Hash of the current root document, seeding `path_cache` lookups for this call.
    doc_hash: u64,

    /// One-entry cache pairing a root `Arc` pointer with its computed hash, avoiding
    /// rehashing when the same document object is reused across sequential calls.
    root_hash_cache: Option<(usize, u64)>,

    /// The pass configuration used when compiling new programs in this VM instance.
    config: PassConfig,
}


/// Delegate `Default` to `VM::new` with the standard capacity defaults.
impl Default for VM {
    fn default() -> Self {
        Self::new()
    }
}

impl VM {
    /// Create a `VM` with default capacities (512 compiled programs, 4096 path entries).
    pub fn new() -> Self {
        Self::with_capacity(512, 4096)
    }

    /// Create a `VM` with explicit compile-cache and path-cache capacities.
    pub fn with_capacity(compile_cap: usize, path_cap: usize) -> Self {
        Self {
            compile_cache: HashMap::with_capacity(compile_cap),
            compile_lru: std::collections::VecDeque::with_capacity(compile_cap),
            compile_cap,
            path_cache: PathCache::new(path_cap),
            root_chain_cache: HashMap::new(),
            doc_hash: 0,
            root_hash_cache: None,
            config: PassConfig::default(),
        }
    }

    /// Override the pass configuration; only available in test builds.
    #[cfg(test)]
    pub fn set_pass_config(&mut self, config: PassConfig) {
        self.config = config;
    }

    /// Compile `expr` (or retrieve from cache) and execute it against `doc`; test-only.
    #[cfg(test)]
    pub fn run_str(
        &mut self,
        expr: &str,
        doc: &serde_json::Value,
    ) -> Result<serde_json::Value, EvalError> {
        let prog = self.get_or_compile(expr)?;
        self.execute(&prog, doc)
    }

    /// Execute a pre-compiled `program` against a `serde_json::Value` document; test-only.
    #[cfg(test)]
    pub fn execute(
        &mut self,
        program: &Program,
        doc: &serde_json::Value,
    ) -> Result<serde_json::Value, EvalError> {
        let root = Val::from(doc);
        self.doc_hash = self.compute_or_cache_root_hash(&root);
        
        
        self.root_chain_cache.clear();
        let env = self.make_env(root);
        let result = self.exec(program, &env)?;
        Ok(result.into())
    }

    /// Compute the structural hash of `root`, using the single-entry `root_hash_cache`
    /// to avoid rehashing when the same `Arc`-backed document is reused across calls.
    fn compute_or_cache_root_hash(&mut self, root: &Val) -> u64 {
        let ptr: Option<usize> = match root {
            Val::Obj(m) => Some(Arc::as_ptr(m) as *const () as usize),
            Val::Arr(a) => Some(Arc::as_ptr(a) as *const () as usize),
            Val::IntVec(a) => Some(Arc::as_ptr(a) as *const () as usize),
            Val::FloatVec(a) => Some(Arc::as_ptr(a) as *const () as usize),
            _ => None,
        };
        if let Some(p) = ptr {
            if let Some((cp, h)) = self.root_hash_cache {
                if cp == p {
                    return h;
                }
            }
            let h = hash_val_structure(root);
            self.root_hash_cache = Some((p, h));
            h
        } else {
            hash_val_structure(root)
        }
    }

    /// Execute `program` against the given `Val` root and convert the result to
    /// `serde_json::Value`. Clears the `root_chain_cache` before each run.
    pub fn execute_val(
        &mut self,
        program: &Program,
        root: Val,
    ) -> Result<serde_json::Value, EvalError> {
        Ok(self.execute_val_raw(program, root)?.into())
    }

    /// Execute `program` against the given `Val` root and return the raw `Val` result
    /// without converting to `serde_json::Value`.
    pub fn execute_val_raw(&mut self, program: &Program, root: Val) -> Result<Val, EvalError> {
        self.doc_hash = self.compute_or_cache_root_hash(&root);
        self.root_chain_cache.clear();
        let env = self.make_env(root);
        self.exec(program, &env)
    }

    /// Execute `program` within an already-constructed `Env`, bypassing document-hash
    /// setup. Used by the runtime when the caller manages the environment directly.
    #[inline]
    pub fn exec_in_env(&mut self, program: &Program, env: &Env) -> Result<Val, EvalError> {
        self.exec(program, env)
    }

    /// Return a cached compiled program for `expr`, compiling and caching it if absent.
    /// The cache key includes the `PassConfig` hash so config changes produce distinct entries.
    pub fn get_or_compile(&mut self, expr: &str) -> Result<Arc<Program>, EvalError> {
        let key = (self.config.hash(), expr.to_string());
        if let Some(p) = self.compile_cache.get(&key) {
            let arc = Arc::clone(p);
            self.touch_lru(&key);
            return Ok(arc);
        }
        let prog = Compiler::compile_str_with_config(expr, self.config)?;
        let arc = Arc::new(prog);
        self.insert_compile(key, Arc::clone(&arc));
        Ok(arc)
    }

    
    /// Move `key` to the back of the LRU deque, marking it as most-recently-used.
    fn touch_lru(&mut self, key: &(u64, String)) {
        if let Some(pos) = self.compile_lru.iter().position(|k| k == key) {
            let k = self.compile_lru.remove(pos).unwrap();
            self.compile_lru.push_back(k);
        }
    }

    /// Insert a new compiled program into the cache, evicting least-recently-used
    /// entries when the compile cache is at capacity.
    fn insert_compile(&mut self, key: (u64, String), prog: Arc<Program>) {
        while self.compile_cache.len() >= self.compile_cap && self.compile_cap > 0 {
            if let Some(old) = self.compile_lru.pop_front() {
                self.compile_cache.remove(&old);
            } else {
                break;
            }
        }
        self.compile_lru.push_back(key.clone());
        self.compile_cache.insert(key, prog);
    }

    /// Return `(compile_cache_len, path_cache_len)` for assertion in tests.
    #[cfg(test)]
    pub fn cache_stats(&self) -> (usize, usize) {
        (self.compile_cache.len(), self.path_cache.len())
    }

    /// Construct a fresh `Env` with `root` as both the root and current value.
    fn make_env(&self, root: Val) -> Env {
        Env::new(root)
    }

    /// Core interpreter loop: execute every opcode in `program` against `env`,
    /// returning the single value left on the stack when the program completes.
    pub fn exec(&mut self, program: &Program, env: &Env) -> Result<Val, EvalError> {
        let mut stack: SmallVec<[Val; 16]> = SmallVec::new();
        let ops_slice: &[Opcode] = &program.ops;
        let mut skip_ahead: usize = 0;

        for (op_idx, op) in ops_slice.iter().enumerate() {
            if skip_ahead > 0 {
                skip_ahead -= 1;
                continue;
            }
            match op {
                
                Opcode::PushNull => stack.push(Val::Null),
                Opcode::PushBool(b) => stack.push(Val::Bool(*b)),
                Opcode::PushInt(n) => stack.push(Val::Int(*n)),
                Opcode::PushFloat(f) => stack.push(Val::Float(*f)),
                Opcode::PushStr(s) => stack.push(Val::Str(s.clone())),

                
                Opcode::PushRoot => stack.push(env.root.clone()),
                Opcode::PushCurrent => stack.push(env.current.clone()),

                
                Opcode::GetField(k) => {
                    let v = pop!(stack);
                    let out = match &v {
                        Val::Obj(m) => ic_get_field(m, k.as_ref(), &program.ics[op_idx]),
                        _ => Val::Null,
                    };
                    stack.push(out);
                }
                Opcode::FieldChain(chain) => {
                    let mut cur = pop!(stack);
                    for (i, k) in chain.keys.iter().enumerate() {
                        cur = if let Val::Obj(m) = &cur {
                            ic_get_field(m, k.as_ref(), &chain.ics[i])
                        } else {
                            cur.get_field(k.as_ref())
                        };
                    }
                    stack.push(cur);
                }
                Opcode::GetIndex(i) => {
                    let v = pop!(stack);
                    stack.push(v.get_index(*i));
                }
                Opcode::DynIndex(prog) => {
                    let v = pop!(stack);
                    let key = self.exec(prog, env)?;
                    stack.push(match key {
                        Val::Int(i) => v.get_index(i),
                        Val::Str(s) => v.get_field(s.as_ref()),
                        _ => Val::Null,
                    });
                }
                Opcode::GetSlice(from, to) => {
                    let v = pop!(stack);
                    stack.push(exec_slice(v, *from, *to));
                }
                Opcode::OptField(k) => {
                    let v = pop!(stack);
                    let out = match &v {
                        Val::Null => Val::Null,
                        Val::Obj(m) => ic_get_field(m, k.as_ref(), &program.ics[op_idx]),
                        _ => v.get_field(k.as_ref()),
                    };
                    stack.push(out);
                }
                Opcode::Descendant(k) => {
                    let v = pop!(stack);
                    
                    
                    let from_root = match (&v, &env.root) {
                        (Val::Obj(a), Val::Obj(b)) => Arc::ptr_eq(a, b),
                        (Val::Arr(a), Val::Arr(b)) => Arc::ptr_eq(a, b),
                        _ => matches!((&v, &env.root), (Val::Null, Val::Null)),
                    };
                    
                    
                    if let Some(next) = ops_slice.get(op_idx + 1) {
                        if is_first_selector_op(next) {
                            let hit = find_desc_first(&v, k.as_ref()).unwrap_or(Val::Null);
                            stack.push(hit);
                            skip_ahead = 1;
                            continue;
                        }
                    }
                    let mut found = Vec::new();
                    if from_root {
                        let mut prefix = String::new();
                        let mut cached: Vec<(Arc<str>, Val)> = Vec::new();
                        collect_desc_with_paths(
                            &v,
                            k.as_ref(),
                            &mut prefix,
                            &mut found,
                            &mut cached,
                        );
                        let doc_hash = self.doc_hash;
                        for (ptr, val) in cached {
                            self.path_cache.insert(doc_hash, ptr, val);
                        }
                    } else {
                        collect_desc(&v, k.as_ref(), &mut found);
                    }
                    stack.push(Val::arr(found));
                }
                Opcode::DescendAll => {
                    let v = pop!(stack);
                    let mut found = Vec::new();
                    collect_all(&v, &mut found);
                    stack.push(Val::arr(found));
                }
                Opcode::InlineFilter(pred) => {
                    let val = pop!(stack);
                    let items = match val {
                        Val::Arr(a) => Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone()),
                        other => vec![other],
                    };
                    let mut out = Vec::with_capacity(items.len());
                    let mut scratch = env.clone();
                    for item in items {
                        let prev = scratch.swap_current(item.clone());
                        let keep = is_truthy(&self.exec(pred, &scratch)?);
                        scratch.restore_current(prev);
                        if keep {
                            out.push(item);
                        }
                    }
                    stack.push(Val::arr(out));
                }
                Opcode::Quantifier(kind) => {
                    let val = pop!(stack);
                    stack.push(match kind {
                        QuantifierKind::First => match val {
                            Val::Arr(a) => a.first().cloned().unwrap_or(Val::Null),
                            other => other,
                        },
                        QuantifierKind::One => match val {
                            Val::Arr(a) if a.len() == 1 => a[0].clone(),
                            Val::Arr(a) => {
                                return err!(
                                    "quantifier !: expected exactly one element, got {}",
                                    a.len()
                                )
                            }
                            other => other,
                        },
                    });
                }

                
                Opcode::RootChain(chain) => {
                    
                    
                    let key = Arc::as_ptr(chain) as *const () as usize;
                    if let Some(v) = self.root_chain_cache.get(&key) {
                        stack.push(v.clone());
                        continue;
                    }

                    let doc_hash = self.doc_hash;
                    let mut current = env.root.clone();
                    let mut ptr = String::new();
                    let mut resumed_from_cache = false;
                    for k in chain.iter() {
                        ptr.push('/');
                        ptr.push_str(k.as_ref());
                        
                        
                        if !resumed_from_cache {
                            if let Some(cached) = self.path_cache.get(doc_hash, &ptr) {
                                current = cached;
                                continue;
                            }
                            resumed_from_cache = true;
                        }
                        current = current.get_field(k.as_ref());
                        self.path_cache
                            .insert(doc_hash, Arc::from(ptr.as_str()), current.clone());
                    }

                    self.root_chain_cache.insert(key, current.clone());
                    stack.push(current);
                }
                
                
                Opcode::LoadIdent(name) => {
                    let v = if let Some(v) = env.get_var(name.as_ref()) {
                        v.clone()
                    } else if matches!(
                        &env.current,
                        Val::Arr(_)
                            | Val::IntVec(_)
                            | Val::FloatVec(_)
                            | Val::StrVec(_)
                            | Val::StrSliceVec(_)
                            | Val::Str(_)
                            | Val::StrSlice(_)
                    ) && BuiltinMethod::from_name(name.as_ref()) != BuiltinMethod::Unknown
                    {
                        crate::builtins::eval_builtin_no_args(env.current.clone(), name.as_ref())?
                    } else {
                        env.current.get_field(name.as_ref())
                    };
                    stack.push(v);
                }

                
                Opcode::Add => {
                    let r = pop!(stack);
                    let l = pop!(stack);
                    stack.push(add_vals(l, r)?);
                }
                Opcode::Sub => {
                    let r = pop!(stack);
                    let l = pop!(stack);
                    stack.push(num_op(l, r, |a, b| a - b, |a, b| a - b)?);
                }
                Opcode::Mul => {
                    let r = pop!(stack);
                    let l = pop!(stack);
                    stack.push(num_op(l, r, |a, b| a * b, |a, b| a * b)?);
                }
                Opcode::Div => {
                    let r = pop!(stack);
                    let l = pop!(stack);
                    let b = r.as_f64().unwrap_or(0.0);
                    if b == 0.0 {
                        return err!("division by zero");
                    }
                    stack.push(Val::Float(l.as_f64().unwrap_or(0.0) / b));
                }
                Opcode::Mod => {
                    let r = pop!(stack);
                    let l = pop!(stack);
                    stack.push(num_op(l, r, |a, b| a % b, |a, b| a % b)?);
                }
                Opcode::Eq => {
                    let r = pop!(stack);
                    let l = pop!(stack);
                    stack.push(Val::Bool(vals_eq(&l, &r)));
                }
                Opcode::Neq => {
                    let r = pop!(stack);
                    let l = pop!(stack);
                    stack.push(Val::Bool(!vals_eq(&l, &r)));
                }
                Opcode::Lt => {
                    let r = pop!(stack);
                    let l = pop!(stack);
                    stack.push(Val::Bool(cmp_vals_binop(&l, BinOp::Lt, &r)));
                }
                Opcode::Lte => {
                    let r = pop!(stack);
                    let l = pop!(stack);
                    stack.push(Val::Bool(cmp_vals_binop(&l, BinOp::Lte, &r)));
                }
                Opcode::Gt => {
                    let r = pop!(stack);
                    let l = pop!(stack);
                    stack.push(Val::Bool(cmp_vals_binop(&l, BinOp::Gt, &r)));
                }
                Opcode::Gte => {
                    let r = pop!(stack);
                    let l = pop!(stack);
                    stack.push(Val::Bool(cmp_vals_binop(&l, BinOp::Gte, &r)));
                }
                Opcode::Fuzzy => {
                    let r = pop!(stack);
                    let l = pop!(stack);
                    let ls = match &l {
                        Val::Str(s) => s.to_lowercase(),
                        _ => val_to_string(&l).to_lowercase(),
                    };
                    let rs = match &r {
                        Val::Str(s) => s.to_lowercase(),
                        _ => val_to_string(&r).to_lowercase(),
                    };
                    stack.push(Val::Bool(ls.contains(&rs) || rs.contains(&ls)));
                }
                Opcode::Not => {
                    let v = pop!(stack);
                    stack.push(Val::Bool(!is_truthy(&v)));
                }
                Opcode::Neg => {
                    let v = pop!(stack);
                    stack.push(match v {
                        Val::Int(n) => Val::Int(-n),
                        Val::Float(f) => Val::Float(-f),
                        _ => return err!("unary minus requires a number"),
                    });
                }
                Opcode::CastOp(ty) => {
                    let v = pop!(stack);
                    stack.push(exec_cast(&v, *ty)?);
                }

                
                Opcode::AndOp(rhs) => {
                    let lv = pop!(stack);
                    if !is_truthy(&lv) {
                        stack.push(Val::Bool(false));
                    } else {
                        let rv = self.exec(rhs, env)?;
                        stack.push(Val::Bool(is_truthy(&rv)));
                    }
                }
                Opcode::OrOp(rhs) => {
                    let lv = pop!(stack);
                    if is_truthy(&lv) {
                        stack.push(lv);
                    } else {
                        stack.push(self.exec(rhs, env)?);
                    }
                }
                Opcode::CoalesceOp(rhs) => {
                    let lv = pop!(stack);
                    if !lv.is_null() {
                        stack.push(lv);
                    } else {
                        stack.push(self.exec(rhs, env)?);
                    }
                }
                Opcode::IfElse { then_, else_ } => {
                    let cv = pop!(stack);
                    let branch = if is_truthy(&cv) { then_ } else { else_ };
                    stack.push(self.exec(branch, env)?);
                }
                Opcode::TryExpr { body, default } => {
                    
                    match self.exec(body, env) {
                        Ok(v) if !v.is_null() => stack.push(v),
                        Ok(_) | Err(_) => stack.push(self.exec(default, env)?),
                    }
                }

                
                Opcode::CallMethod(call) => {
                    let recv = pop!(stack);
                    let result = self.exec_call(recv, call, env)?;
                    stack.push(result);
                }
                Opcode::CallOptMethod(call) => {
                    let recv = pop!(stack);
                    if recv.is_null() {
                        stack.push(Val::Null);
                    } else {
                        stack.push(self.exec_call(recv, call, env)?);
                    }
                }

                
                Opcode::MakeObj(entries) => {
                    let entries = Arc::clone(entries);
                    let result = self.exec_make_obj(&entries, env)?;
                    stack.push(result);
                }
                Opcode::MakeArr(progs) => {
                    let progs = Arc::clone(progs);
                    let mut out = Vec::with_capacity(progs.len());
                    for (p, is_spread) in progs.iter() {
                        let v = self.exec(p, env)?;
                        if *is_spread {
                            match v {
                                Val::Arr(a) => {
                                    let items = Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone());
                                    out.extend(items);
                                }
                                Val::IntVec(a) => out.extend(a.iter().map(|n| Val::Int(*n))),
                                Val::FloatVec(a) => out.extend(a.iter().map(|f| Val::Float(*f))),
                                Val::StrVec(a) => out.extend(a.iter().cloned().map(Val::Str)),
                                other => out.push(other),
                            }
                        } else {
                            out.push(v);
                        }
                    }
                    stack.push(Val::arr(out));
                }

                
                Opcode::FString(parts) => {
                    let parts = Arc::clone(parts);
                    let result = self.exec_fstring(&parts, env)?;
                    stack.push(result);
                }

                
                Opcode::KindCheck { ty, negate } => {
                    let v = pop!(stack);
                    let m = kind_matches(&v, *ty);
                    stack.push(Val::Bool(if *negate { !m } else { m }));
                }

                
                Opcode::SetCurrent => {
                    
                    
                }
                Opcode::PipelineRun { base, steps } => {
                    let val = self.exec(base, env)?;
                    let mut local_env = env.clone();
                    let mut cur = val;
                    for step in steps.iter() {
                        match step {
                            CompiledPipeStep::Forward(rhs) => {
                                
                                
                                let prev = local_env.swap_current(cur);
                                cur = self.exec(rhs, &local_env)?;
                                let _ = local_env.swap_current(prev);
                            }
                            CompiledPipeStep::BindName(name) => {
                                local_env = local_env.with_var(name.as_ref(), cur.clone());
                            }
                            CompiledPipeStep::BindObj(spec) => {
                                if let Val::Obj(m) = &cur {
                                    let mut consumed: std::collections::HashSet<&str> =
                                        std::collections::HashSet::new();
                                    for f in spec.fields.iter() {
                                        let v = m.get(f.as_ref()).cloned().unwrap_or(Val::Null);
                                        local_env = local_env.with_var(f.as_ref(), v);
                                        consumed.insert(f.as_ref());
                                    }
                                    if let Some(rest) = &spec.rest {
                                        let mut rest_obj = indexmap::IndexMap::new();
                                        for (k, v) in m.iter() {
                                            if !consumed.contains(k.as_ref()) {
                                                rest_obj.insert(k.clone(), v.clone());
                                            }
                                        }
                                        local_env = local_env
                                            .with_var(rest.as_ref(), Val::Obj(Arc::new(rest_obj)));
                                    }
                                }
                            }
                            CompiledPipeStep::BindArr(names) => {
                                let items: Vec<Val> = match &cur {
                                    Val::Arr(a) => a.iter().cloned().collect(),
                                    Val::IntVec(a) => a.iter().map(|n| Val::Int(*n)).collect(),
                                    Val::FloatVec(a) => a.iter().map(|f| Val::Float(*f)).collect(),
                                    Val::StrVec(a) => a.iter().cloned().map(Val::Str).collect(),
                                    _ => Vec::new(),
                                };
                                for (i, n) in names.iter().enumerate() {
                                    let v = items.get(i).cloned().unwrap_or(Val::Null);
                                    local_env = local_env.with_var(n.as_ref(), v);
                                }
                            }
                        }
                    }
                    stack.push(cur);
                }

                
                Opcode::LetExpr { name, body } => {
                    let init_val = pop!(stack);
                    let body_env = env.with_var(name.as_ref(), init_val);
                    stack.push(self.exec(body, &body_env)?);
                }

                Opcode::ListComp(spec) => {
                    let items = self.exec_iter_vals(&spec.iter, env)?;
                    let mut out = Vec::with_capacity(items.len());
                    
                    
                    if spec.vars.len() == 1 {
                        let vname = spec.vars[0].clone();
                        let mut scratch = env.clone();
                        for item in items {
                            let frame = scratch.push_lam(Some(vname.as_ref()), item);
                            let keep = match &spec.cond {
                                Some(c) => is_truthy(&self.exec(c, &scratch)?),
                                None => true,
                            };
                            if keep {
                                let v = self.exec(&spec.expr, &scratch)?;
                                out.push(v);
                            }
                            scratch.pop_lam(frame);
                        }
                    } else {
                        for item in items {
                            let ie = bind_comp_vars(env, &spec.vars, item);
                            if let Some(cond) = &spec.cond {
                                if !is_truthy(&self.exec(cond, &ie)?) {
                                    continue;
                                }
                            }
                            out.push(self.exec(&spec.expr, &ie)?);
                        }
                    }
                    stack.push(Val::arr(out));
                }

                Opcode::DictComp(spec) => {
                    let items = self.exec_iter_vals(&spec.iter, env)?;
                    let mut map: IndexMap<Arc<str>, Val> = IndexMap::with_capacity(items.len());
                    
                    
                    if spec.vars.len() == 1 {
                        let vname = spec.vars[0].clone();
                        
                        
                        let val_is_ident = matches!(
                            spec.val.ops.as_ref(),
                            [Opcode::LoadIdent(v)] if v.as_ref() == vname.as_ref()
                        );
                        let key_shape = classify_dict_key(&spec.key, vname.as_ref());
                        if spec.cond.is_none() && val_is_ident && key_shape.is_some() {
                            let shape = key_shape.unwrap();
                            for item in items {
                                let k: Arc<str> = match shape {
                                    DictKeyShape::Ident => match &item {
                                        Val::Str(s) => s.clone(),
                                        other => Arc::<str>::from(val_to_key(other)),
                                    },
                                    DictKeyShape::IdentToString => match &item {
                                        Val::Str(s) => s.clone(),
                                        Val::Int(n) => Arc::<str>::from(n.to_string()),
                                        Val::Float(f) => Arc::<str>::from(f.to_string()),
                                        Val::Bool(b) => {
                                            Arc::<str>::from(if *b { "true" } else { "false" })
                                        }
                                        Val::Null => Arc::<str>::from("null"),
                                        other => Arc::<str>::from(val_to_key(other)),
                                    },
                                };
                                map.insert(k, item);
                            }
                            stack.push(Val::obj(map));
                            continue;
                        }
                        let mut scratch = env.clone();
                        for item in items {
                            let frame = scratch.push_lam(Some(vname.as_ref()), item);
                            let keep = match &spec.cond {
                                Some(c) => is_truthy(&self.exec(c, &scratch)?),
                                None => true,
                            };
                            if keep {
                                let k: Arc<str> = match self.exec(&spec.key, &scratch)? {
                                    Val::Str(s) => s,
                                    other => Arc::<str>::from(val_to_key(&other)),
                                };
                                let v = self.exec(&spec.val, &scratch)?;
                                map.insert(k, v);
                            }
                            scratch.pop_lam(frame);
                        }
                    } else {
                        for item in items {
                            let ie = bind_comp_vars(env, &spec.vars, item);
                            if let Some(cond) = &spec.cond {
                                if !is_truthy(&self.exec(cond, &ie)?) {
                                    continue;
                                }
                            }
                            let k: Arc<str> = match self.exec(&spec.key, &ie)? {
                                Val::Str(s) => s,
                                other => Arc::<str>::from(val_to_key(&other)),
                            };
                            let v = self.exec(&spec.val, &ie)?;
                            map.insert(k, v);
                        }
                    }
                    stack.push(Val::obj(map));
                }

                Opcode::SetComp(spec) => {
                    let items = self.exec_iter_vals(&spec.iter, env)?;
                    let mut seen: std::collections::HashSet<String> =
                        std::collections::HashSet::with_capacity(items.len());
                    let mut out = Vec::with_capacity(items.len());
                    if spec.vars.len() == 1 {
                        let vname = spec.vars[0].clone();
                        let mut scratch = env.clone();
                        for item in items {
                            let frame = scratch.push_lam(Some(vname.as_ref()), item);
                            let keep = match &spec.cond {
                                Some(c) => is_truthy(&self.exec(c, &scratch)?),
                                None => true,
                            };
                            if keep {
                                let v = self.exec(&spec.expr, &scratch)?;
                                let k = val_to_key(&v);
                                if seen.insert(k) {
                                    out.push(v);
                                }
                            }
                            scratch.pop_lam(frame);
                        }
                    } else {
                        for item in items {
                            let ie = bind_comp_vars(env, &spec.vars, item);
                            if let Some(cond) = &spec.cond {
                                if !is_truthy(&self.exec(cond, &ie)?) {
                                    continue;
                                }
                            }
                            let v = self.exec(&spec.expr, &ie)?;
                            let k = val_to_key(&v);
                            if seen.insert(k) {
                                out.push(v);
                            }
                        }
                    }
                    stack.push(Val::arr(out));
                }

                
                Opcode::PatchEval(cp) => {
                    let result = self.exec_patch_compiled(cp, env)?;
                    stack.push(result);
                }
                Opcode::DeleteMarkErr => {
                    return err!("DELETE: only valid inside a patch-field value");
                }
            }
        }

        stack
            .pop()
            .ok_or_else(|| EvalError("program produced no value".into()))
    }

    /// Dispatch a `CallMethod` opcode: applies fast numeric/typed specialisations first,
    /// then lambda-aware methods, then the general `call_builtin_method_compiled` fallback.
    fn exec_call(&mut self, recv: Val, call: &CompiledCall, env: &Env) -> Result<Val, EvalError> {
        
        if call.method == BuiltinMethod::Unknown {
            
            
            match call.name.as_ref() {
                "coalesce" | "chain" | "join" | "zip" | "zip_longest" | "product" | "range" => {
                    return crate::data::runtime::eval_global_compiled(self, call, env);
                }
                _ => {}
            }
            return call_builtin_method_compiled(self, recv, call, env);
        }

        if call.method == BuiltinMethod::Count && call.orig_args.is_empty() {
            return Ok(crate::builtins::len_apply(&recv).unwrap_or(Val::Int(0)));
        }

        if matches!(
            call.method,
            BuiltinMethod::Sum | BuiltinMethod::Avg | BuiltinMethod::Min | BuiltinMethod::Max
        ) && !call.orig_args.is_empty()
        {
            let proj = call
                .sub_progs
                .first()
                .ok_or_else(|| EvalError(format!("{}: requires projection", call.name)))?;
            let lam_param: Option<&str> = match call.orig_args.first() {
                Some(Arg::Pos(Expr::Lambda { params, .. })) if !params.is_empty() => {
                    Some(params[0].as_str())
                }
                _ => None,
            };
            let mut scratch = env.clone();
            return crate::builtins::numeric_aggregate_projected_apply(
                &recv,
                call.method,
                |item| self.exec_lam_body_scratch(proj, item, lam_param, &mut scratch),
            );
        }

        
        if call.method.is_lambda_method() {
            return self.exec_lambda_method(recv, call, env);
        }

        
        if call.sub_progs.is_empty() && call.orig_args.is_empty() {
            if let Val::Arr(a) = &recv {
                match call.method {
                    BuiltinMethod::Sum => return Ok(agg_sum_typed(a)),
                    BuiltinMethod::Avg => return Ok(agg_avg_typed(a)),
                    BuiltinMethod::Min => return Ok(agg_minmax_typed(a, false)),
                    BuiltinMethod::Max => return Ok(agg_minmax_typed(a, true)),
                    _ => {}
                }
            }
            
            
            if let Val::IntVec(a) = &recv {
                match call.method {
                    BuiltinMethod::Sum => return Ok(Val::Int(simd_sum_i64_slice(a))),
                    BuiltinMethod::Avg => {
                        if a.is_empty() {
                            return Ok(Val::Null);
                        }
                        return Ok(Val::Float(simd_sum_i64_slice(a) as f64 / a.len() as f64));
                    }
                    BuiltinMethod::Min => {
                        return Ok(simd_min_i64_slice(a).map(Val::Int).unwrap_or(Val::Null));
                    }
                    BuiltinMethod::Max => {
                        return Ok(simd_max_i64_slice(a).map(Val::Int).unwrap_or(Val::Null));
                    }
                    BuiltinMethod::Count | BuiltinMethod::Len => {
                        return Ok(Val::Int(a.len() as i64));
                    }
                    _ => {}
                }
            }
            
            
            if let Val::Arr(a) = &recv {
                let is_all_int = a.iter().all(|v| matches!(v, Val::Int(_)));
                if is_all_int && !a.is_empty() {
                    match call.method {
                        BuiltinMethod::Reverse => {
                            let mut v: Vec<i64> = a
                                .iter()
                                .map(|x| if let Val::Int(n) = x { *n } else { 0 })
                                .collect();
                            v.reverse();
                            return Ok(Val::int_vec(v));
                        }
                        BuiltinMethod::Sort => {
                            let mut v: Vec<i64> = a
                                .iter()
                                .map(|x| if let Val::Int(n) = x { *n } else { 0 })
                                .collect();
                            v.sort_unstable();
                            return Ok(Val::int_vec(v));
                        }
                        BuiltinMethod::Sum => {
                            let s: i64 = a.iter().fold(0i64, |acc, v| {
                                if let Val::Int(n) = v {
                                    acc.wrapping_add(*n)
                                } else {
                                    acc
                                }
                            });
                            return Ok(Val::Int(s));
                        }
                        _ => {}
                    }
                }
            }
            
            if let Val::IntVec(a) = &recv {
                match call.method {
                    BuiltinMethod::Reverse => {
                        let mut v: Vec<i64> =
                            Arc::try_unwrap(a.clone()).unwrap_or_else(|a| (*a).clone());
                        v.reverse();
                        return Ok(Val::int_vec(v));
                    }
                    BuiltinMethod::Sort => {
                        let mut v: Vec<i64> =
                            Arc::try_unwrap(a.clone()).unwrap_or_else(|a| (*a).clone());
                        v.sort_unstable();
                        return Ok(Val::int_vec(v));
                    }
                    _ => {}
                }
            }
            if let Val::FloatVec(a) = &recv {
                match call.method {
                    BuiltinMethod::Sum => return Ok(Val::Float(simd_sum_f64_slice(a))),
                    BuiltinMethod::Avg => {
                        if a.is_empty() {
                            return Ok(Val::Null);
                        }
                        return Ok(Val::Float(simd_sum_f64_slice(a) / a.len() as f64));
                    }
                    BuiltinMethod::Min => {
                        return Ok(simd_min_f64_slice(a).map(Val::Float).unwrap_or(Val::Null))
                    }
                    BuiltinMethod::Max => {
                        return Ok(simd_max_f64_slice(a).map(Val::Float).unwrap_or(Val::Null))
                    }
                    BuiltinMethod::Count | BuiltinMethod::Len => {
                        return Ok(Val::Int(a.len() as i64));
                    }
                    _ => {}
                }
            }
            
            
            if call.method == BuiltinMethod::Flatten {
                if let Val::Arr(a) = &recv {
                    
                    let all_int_inner = a.iter().all(|it| match it {
                        Val::IntVec(_) => true,
                        Val::Arr(inner) => inner.iter().all(|v| matches!(v, Val::Int(_))),
                        Val::Int(_) => true,
                        _ => false,
                    });
                    if all_int_inner {
                        let cap: usize = a
                            .iter()
                            .map(|it| match it {
                                Val::IntVec(inner) => inner.len(),
                                Val::Arr(inner) => inner.len(),
                                _ => 1,
                            })
                            .sum();
                        let mut out: Vec<i64> = Vec::with_capacity(cap);
                        for item in a.iter() {
                            match item {
                                Val::IntVec(inner) => out.extend(inner.iter().copied()),
                                Val::Arr(inner) => {
                                    out.extend(inner.iter().filter_map(|v| v.as_i64()))
                                }
                                Val::Int(n) => out.push(*n),
                                _ => {}
                            }
                        }
                        return Ok(Val::int_vec(out));
                    }
                    let cap: usize = a
                        .iter()
                        .map(|it| match it {
                            Val::Arr(inner) => inner.len(),
                            Val::IntVec(inner) => inner.len(),
                            Val::FloatVec(inner) => inner.len(),
                            Val::StrVec(inner) => inner.len(),
                            Val::StrSliceVec(inner) => inner.len(),
                            _ => 1,
                        })
                        .sum();
                    let mut out = Vec::with_capacity(cap);
                    for item in a.iter() {
                        match item {
                            Val::Arr(inner) => out.extend(inner.iter().cloned()),
                            Val::IntVec(inner) => out.extend(inner.iter().map(|n| Val::Int(*n))),
                            Val::FloatVec(inner) => {
                                out.extend(inner.iter().map(|f| Val::Float(*f)))
                            }
                            Val::StrVec(inner) => {
                                out.extend(inner.iter().map(|s| Val::Str(s.clone())))
                            }
                            Val::StrSliceVec(inner) => {
                                out.extend(inner.iter().map(|s| Val::StrSlice(s.clone())))
                            }
                            other => out.push(other.clone()),
                        }
                    }
                    return Ok(Val::arr(out));
                }
                
                if matches!(
                    &recv,
                    Val::IntVec(_) | Val::FloatVec(_) | Val::StrVec(_) | Val::StrSliceVec(_)
                ) {
                    return Ok(recv);
                }
            }
            
            
            if call.method == BuiltinMethod::ToString {
                let s: Arc<str> = match &recv {
                    Val::Str(s) => return Ok(Val::Str(s.clone())),
                    Val::Int(n) => Arc::from(n.to_string()),
                    Val::Float(f) => Arc::from(f.to_string()),
                    Val::Bool(b) => Arc::from(b.to_string()),
                    Val::Null => Arc::from("null"),
                    other => Arc::from(crate::util::val_to_string(other).as_str()),
                };
                return Ok(Val::Str(s));
            }
            
            
            if call.method == BuiltinMethod::ToJson {
                match &recv {
                    Val::Int(n) => return Ok(Val::Str(Arc::from(n.to_string()))),
                    Val::Float(f) => {
                        let s = if f.is_finite() {
                            f.to_string()
                        } else {
                            "null".to_string()
                        };
                        return Ok(Val::Str(Arc::from(s)));
                    }
                    Val::Bool(b) => {
                        return Ok(Val::Str(Arc::from(if *b { "true" } else { "false" })))
                    }
                    Val::Null => return Ok(Val::Str(Arc::from("null"))),
                    Val::Str(s) => {
                        
                        
                        let src = s.as_ref();
                        let mut needs_escape = false;
                        for &b in src.as_bytes() {
                            if b < 0x20 || b == b'"' || b == b'\\' {
                                needs_escape = true;
                                break;
                            }
                        }
                        if !needs_escape {
                            let mut out = String::with_capacity(src.len() + 2);
                            out.push('"');
                            out.push_str(src);
                            out.push('"');
                            return Ok(Val::Str(Arc::from(out)));
                        }
                        
                    }
                    _ => {}
                }
            }
        }

        if let Some(v) = self.exec_static_builtin_call(&recv, call, env)? {
            return Ok(v);
        }

        
        call_builtin_method_compiled(self, recv, call, env)
    }

    /// Attempt to dispatch a built-in call using a pre-computed `BuiltinCall` descriptor
    /// whose arguments are all static (non-lambda). Returns `Ok(None)` if inapplicable.
    fn exec_static_builtin_call(
        &mut self,
        recv: &Val,
        call: &CompiledCall,
        env: &Env,
    ) -> Result<Option<Val>, EvalError> {
        if call.method == BuiltinMethod::Unknown || call.method.is_lambda_method() {
            return Ok(None);
        }

        if let Some(shared_call) = self.static_shared_builtin_call(call, env)? {
            if let Some(v) = shared_call.try_apply(recv)? {
                return Ok(Some(v));
            }
        }

        Ok(None)
    }

    /// Build a `BuiltinCall` descriptor by evaluating each argument eagerly as a static value.
    /// Returns `None` when the method does not support the static-args protocol.
    fn static_shared_builtin_call(
        &mut self,
        call: &CompiledCall,
        env: &Env,
    ) -> Result<Option<crate::builtins::BuiltinCall>, EvalError> {
        crate::builtins::BuiltinCall::from_static_args(
            call.method,
            call.name.as_ref(),
            call.orig_args.len(),
            |idx| self.static_arg_val(call, env, idx),
            |idx| match call.orig_args.get(idx) {
                Some(Arg::Pos(Expr::Ident(s))) => Some(Arc::from(s.as_str())),
                _ => None,
            },
        )
    }

    /// Evaluate the sub-program at position `idx` in `call` to produce a concrete argument
    /// value. Returns `Ok(None)` when the index is out of range.
    fn static_arg_val(
        &mut self,
        call: &CompiledCall,
        env: &Env,
        idx: usize,
    ) -> Result<Option<Val>, EvalError> {
        match call.sub_progs.get(idx) {
            Some(prog) => self.exec(prog, env).map(Some),
            None => Ok(None),
        }
    }

    /// Execute the class of built-in methods that accept lambda/predicate arguments
    /// (e.g. `filter`, `map`, `sort`, `groupBy`, `accumulate`, …).
    fn exec_lambda_method(
        &mut self,
        recv: Val,
        call: &CompiledCall,
        env: &Env,
    ) -> Result<Val, EvalError> {
        let sub = call.sub_progs.first();
        
        
        let lam_param: Option<&str> = match call.orig_args.first() {
            Some(Arg::Pos(Expr::Lambda { params, .. })) if !params.is_empty() => {
                Some(params[0].as_str())
            }
            _ => None,
        };
        
        
        let mut scratch = env.clone();

        match call.method {
            BuiltinMethod::Filter => {
                let pred = sub.ok_or_else(|| EvalError("filter: requires predicate".into()))?;
                let items = recv
                    .into_vec()
                    .ok_or_else(|| EvalError("filter: expected array".into()))?;
                let out =
                    crate::builtins::filter_apply_bounded(items, call.demand_max_keep, |item| {
                        self.exec_lam_body_scratch(pred, item, lam_param, &mut scratch)
                    })?;
                Ok(Val::arr(out))
            }
            BuiltinMethod::Map => {
                let mapper = sub.ok_or_else(|| EvalError("map: requires mapper".into()))?;
                let items = recv
                    .into_vec()
                    .ok_or_else(|| EvalError("map: expected array".into()))?;
                let out =
                    crate::builtins::map_apply_bounded(items, call.demand_max_keep, |item| {
                        self.exec_lam_body_scratch(mapper, item, lam_param, &mut scratch)
                    })?;
                Ok(Val::arr(out))
            }
            BuiltinMethod::FlatMap => {
                let mapper = sub.ok_or_else(|| EvalError("flatMap: requires mapper".into()))?;
                let items = recv
                    .into_vec()
                    .ok_or_else(|| EvalError("flatMap: expected array".into()))?;
                let out = crate::builtins::flat_map_apply(items, |item| {
                    self.exec_lam_body_scratch(mapper, item, lam_param, &mut scratch)
                })?;
                Ok(Val::arr(out))
            }
            BuiltinMethod::Sort => {
                if call.sub_progs.is_empty() {
                    return crate::builtins::sort_apply(recv);
                }
                if matches!(
                    call.orig_args.first(),
                    Some(Arg::Pos(Expr::Lambda { params, .. }))
                        | Some(Arg::Named(_, Expr::Lambda { params, .. }))
                        if params.len() == 2
                ) {
                    let cmp = call
                        .sub_progs
                        .first()
                        .ok_or_else(|| EvalError("sort: requires comparator".into()))?
                        .clone();
                    return crate::builtins::sort_comparator_apply(recv, |left, right| {
                        self.exec_pair_lam_body(&cmp, left, right, &call.orig_args[0], env)
                    });
                }

                let desc = vec![false; call.sub_progs.len()];
                let progs = call.sub_progs.clone();
                crate::builtins::sort_by_apply(recv, &desc, |item, idx| {
                    let arg = call
                        .orig_args
                        .get(idx)
                        .ok_or_else(|| EvalError("sort: missing key".into()))?;
                    let lam_param = match arg {
                        Arg::Pos(Expr::Lambda { params, .. })
                        | Arg::Named(_, Expr::Lambda { params, .. })
                            if !params.is_empty() =>
                        {
                            Some(params[0].as_str())
                        }
                        _ => None,
                    };
                    self.exec_lam_body_scratch(&progs[idx], item, lam_param, &mut scratch)
                })
            }
            BuiltinMethod::Any => {
                if let Val::Arr(a) = &recv {
                    let pred = sub.ok_or_else(|| EvalError("any: requires predicate".into()))?;
                    for item in a.iter() {
                        if crate::builtins::any_one(item, |v| {
                            self.exec_lam_body_scratch(pred, v, lam_param, &mut scratch)
                        })? {
                            return Ok(Val::Bool(true));
                        }
                    }
                    Ok(Val::Bool(false))
                } else {
                    Ok(Val::Bool(false))
                }
            }
            BuiltinMethod::All => {
                if let Val::Arr(a) = &recv {
                    if a.is_empty() {
                        return Ok(Val::Bool(true));
                    }
                    let pred = sub.ok_or_else(|| EvalError("all: requires predicate".into()))?;
                    for item in a.iter() {
                        if !crate::builtins::all_one(item, |v| {
                            self.exec_lam_body_scratch(pred, v, lam_param, &mut scratch)
                        })? {
                            return Ok(Val::Bool(false));
                        }
                    }
                    Ok(Val::Bool(true))
                } else {
                    Ok(Val::Bool(false))
                }
            }
            BuiltinMethod::Count if !call.sub_progs.is_empty() => {
                if let Val::Arr(a) = &recv {
                    let pred = &call.sub_progs[0];
                    let mut n: i64 = 0;
                    for item in a.iter() {
                        if crate::builtins::filter_one(item, |v| {
                            self.exec_lam_body_scratch(pred, v, lam_param, &mut scratch)
                        })? {
                            n += 1;
                        }
                    }
                    Ok(Val::Int(n))
                } else {
                    Ok(Val::Int(0))
                }
            }
            BuiltinMethod::GroupBy => {
                let key_prog = sub.ok_or_else(|| EvalError("groupBy: requires key".into()))?;
                let items = recv
                    .into_vec()
                    .ok_or_else(|| EvalError("groupBy: expected array".into()))?;
                
                
                if let Some(param) = lam_param {
                    if let [Opcode::LoadIdent(p), Opcode::PushInt(k_lit), Opcode::Mod] =
                        key_prog.ops.as_ref()
                    {
                        if p.as_ref() == param && *k_lit > 0 && *k_lit <= 4096 {
                            let k_lit = *k_lit;
                            let k_u = k_lit as usize;
                            let mut buckets: Vec<Vec<Val>> = vec![Vec::new(); k_u];
                            let mut seen: Vec<bool> = vec![false; k_u];
                            let mut order: Vec<usize> = Vec::new();
                            
                            for item in items {
                                let idx = match &item {
                                    Val::Int(n) => n.rem_euclid(k_lit) as usize,
                                    Val::Float(x) => (x.trunc() as i64).rem_euclid(k_lit) as usize,
                                    _ => return err!("group_by(x % K): non-numeric item"),
                                };
                                if !seen[idx] {
                                    seen[idx] = true;
                                    order.push(idx);
                                }
                                buckets[idx].push(item);
                            }
                            let mut map: IndexMap<Arc<str>, Val> =
                                IndexMap::with_capacity(order.len());
                            for idx in order {
                                let k: Arc<str> = Arc::from(idx.to_string());
                                let bucket = std::mem::take(&mut buckets[idx]);
                                map.insert(k, Val::arr(bucket));
                            }
                            return Ok(Val::obj(map));
                        }
                    }
                }
                
                let map = crate::builtins::group_by_apply(items, |item| {
                    self.exec_lam_body_scratch(key_prog, item, lam_param, &mut scratch)
                })?;
                Ok(Val::obj(map))
            }
            BuiltinMethod::CountBy => {
                let key_prog = sub.ok_or_else(|| EvalError("countBy: requires key".into()))?;
                let items = recv
                    .into_vec()
                    .ok_or_else(|| EvalError("countBy: expected array".into()))?;
                let map = crate::builtins::count_by_apply(items, |item| {
                    self.exec_lam_body_scratch(key_prog, item, lam_param, &mut scratch)
                })?;
                Ok(Val::obj(map))
            }
            BuiltinMethod::IndexBy => {
                let key_prog = sub.ok_or_else(|| EvalError("indexBy: requires key".into()))?;
                let items = recv
                    .into_vec()
                    .ok_or_else(|| EvalError("indexBy: expected array".into()))?;
                let map = crate::builtins::index_by_apply(items, |item| {
                    self.exec_lam_body_scratch(key_prog, item, lam_param, &mut scratch)
                })?;
                Ok(Val::obj(map))
            }
            BuiltinMethod::TakeWhile => {
                let pred = sub.ok_or_else(|| EvalError("takeWhile: requires predicate".into()))?;
                let items = recv
                    .into_vec()
                    .ok_or_else(|| EvalError("takeWhile: expected array".into()))?;
                let out = crate::builtins::take_while_apply(items, |item| {
                    self.exec_lam_body_scratch(pred, item, lam_param, &mut scratch)
                })?;
                Ok(Val::arr(out))
            }
            BuiltinMethod::DropWhile => {
                let pred = sub.ok_or_else(|| EvalError("dropWhile: requires predicate".into()))?;
                let items = recv
                    .into_vec()
                    .ok_or_else(|| EvalError("dropWhile: expected array".into()))?;
                let out = crate::builtins::drop_while_apply(items, |item| {
                    self.exec_lam_body_scratch(pred, item, lam_param, &mut scratch)
                })?;
                Ok(Val::arr(out))
            }
            BuiltinMethod::Accumulate => {
                
                
                let lam_body =
                    sub.ok_or_else(|| EvalError("accumulate: requires lambda".into()))?;
                let (p1, p2) = match call.orig_args.first() {
                    Some(Arg::Pos(Expr::Lambda { params, .. })) if params.len() >= 2 => {
                        (params[0].as_str(), params[1].as_str())
                    }
                    _ => return call_builtin_method_compiled(self, recv, call, env),
                };
                if call
                    .orig_args
                    .iter()
                    .any(|a| matches!(a, Arg::Named(n, _) if n.as_str() == "start"))
                {
                    return call_builtin_method_compiled(self, recv, call, env);
                }
                
                let specialised_binop = match lam_body.ops.as_ref() {
                    [Opcode::LoadIdent(a), Opcode::LoadIdent(b), op]
                        if a.as_ref() == p1 && b.as_ref() == p2 =>
                    {
                        match op {
                            Opcode::Add => Some(AccumOp::Add),
                            Opcode::Sub => Some(AccumOp::Sub),
                            Opcode::Mul => Some(AccumOp::Mul),
                            _ => None,
                        }
                    }
                    _ => None,
                };
                
                if let (Val::IntVec(a), Some(bop)) = (&recv, specialised_binop.as_ref().copied()) {
                    let mut out: Vec<i64> = Vec::with_capacity(a.len());
                    let mut acc: i64 = 0;
                    let mut first = true;
                    for &n in a.iter() {
                        if first {
                            acc = n;
                            first = false;
                        } else {
                            acc = match bop {
                                AccumOp::Add => acc.wrapping_add(n),
                                AccumOp::Sub => acc.wrapping_sub(n),
                                AccumOp::Mul => acc.wrapping_mul(n),
                            };
                        }
                        out.push(acc);
                    }
                    return Ok(Val::int_vec(out));
                }
                if let (Val::FloatVec(a), Some(bop)) = (&recv, specialised_binop.as_ref().copied())
                {
                    let mut out: Vec<f64> = Vec::with_capacity(a.len());
                    let mut acc: f64 = 0.0;
                    let mut first = true;
                    for &n in a.iter() {
                        if first {
                            acc = n;
                            first = false;
                        } else {
                            acc = match bop {
                                AccumOp::Add => acc + n,
                                AccumOp::Sub => acc - n,
                                AccumOp::Mul => acc * n,
                            };
                        }
                        out.push(acc);
                    }
                    return Ok(Val::float_vec(out));
                }
                let items = recv
                    .into_vec()
                    .ok_or_else(|| EvalError("accumulate: expected array".into()))?;
                let mut out = Vec::with_capacity(items.len());
                if let Some(bop) = specialised_binop {
                    
                    
                    if items.iter().all(|v| matches!(v, Val::Int(_))) {
                        let mut acc_out: Vec<i64> = Vec::with_capacity(items.len());
                        let mut acc: i64 = 0;
                        let mut first = true;
                        for item in &items {
                            let n = if let Val::Int(n) = item {
                                *n
                            } else {
                                unreachable!()
                            };
                            if first {
                                acc = n;
                                first = false;
                            } else {
                                acc = match bop {
                                    AccumOp::Add => acc.wrapping_add(n),
                                    AccumOp::Sub => acc.wrapping_sub(n),
                                    AccumOp::Mul => acc.wrapping_mul(n),
                                };
                            }
                            acc_out.push(acc);
                        }
                        return Ok(Val::int_vec(acc_out));
                    }
                    
                    if items.iter().all(|v| matches!(v, Val::Float(_))) {
                        let mut acc_out: Vec<f64> = Vec::with_capacity(items.len());
                        let mut acc: f64 = 0.0;
                        let mut first = true;
                        for item in &items {
                            let n = if let Val::Float(n) = item {
                                *n
                            } else {
                                unreachable!()
                            };
                            if first {
                                acc = n;
                                first = false;
                            } else {
                                acc = match bop {
                                    AccumOp::Add => acc + n,
                                    AccumOp::Sub => acc - n,
                                    AccumOp::Mul => acc * n,
                                };
                            }
                            acc_out.push(acc);
                        }
                        return Ok(Val::float_vec(acc_out));
                    }
                    
                    let mut running: Option<Val> = None;
                    for item in items {
                        let next = match running.take() {
                            Some(acc) => match bop {
                                AccumOp::Add => add_vals(acc, item)?,
                                AccumOp::Sub => num_op(acc, item, |a, b| a - b, |a, b| a - b)?,
                                AccumOp::Mul => num_op(acc, item, |a, b| a * b, |a, b| a * b)?,
                            },
                            None => item,
                        };
                        out.push(next.clone());
                        running = Some(next);
                    }
                    return Ok(Val::arr(out));
                }
                
                let mut running: Option<Val> = None;
                for item in items {
                    let next = if let Some(acc) = running.take() {
                        let f1 = scratch.push_lam(Some(p1), acc);
                        let f2 = scratch.push_lam(Some(p2), item.clone());
                        let r = self.exec(lam_body, &scratch)?;
                        scratch.pop_lam(f2);
                        scratch.pop_lam(f1);
                        r
                    } else {
                        item
                    };
                    out.push(next.clone());
                    running = Some(next);
                }
                Ok(Val::arr(out))
            }
            BuiltinMethod::Partition => {
                let pred = sub.ok_or_else(|| EvalError("partition: requires predicate".into()))?;
                let items = recv
                    .into_vec()
                    .ok_or_else(|| EvalError("partition: expected array".into()))?;
                let (yes, no) = crate::builtins::partition_apply(items, |item| {
                    self.exec_lam_body_scratch(pred, item, lam_param, &mut scratch)
                })?;
                let mut m: IndexMap<Arc<str>, Val> = IndexMap::with_capacity(2);
                m.insert(Arc::from("true"), Val::arr(yes));
                m.insert(Arc::from("false"), Val::arr(no));
                Ok(Val::obj(m))
            }
            BuiltinMethod::TransformKeys => {
                let lam = sub.ok_or_else(|| EvalError("transformKeys: requires lambda".into()))?;
                let map = recv
                    .into_map()
                    .ok_or_else(|| EvalError("transformKeys: expected object".into()))?;
                let out = crate::builtins::transform_keys_apply(map, |k| {
                    self.exec_lam_body_scratch(lam, &Val::Str(k.clone()), lam_param, &mut scratch)
                })?;
                Ok(Val::obj(out))
            }
            BuiltinMethod::TransformValues => {
                let lam =
                    sub.ok_or_else(|| EvalError("transformValues: requires lambda".into()))?;
                
                
                let mut map = recv
                    .into_map()
                    .ok_or_else(|| EvalError("transformValues: expected object".into()))?;
                
                
                let pat = match lam.ops.as_ref() {
                    [Opcode::PushCurrent, Opcode::PushInt(k), op] => match op {
                        Opcode::Add => Some((AccumOp::Add, *k)),
                        Opcode::Sub => Some((AccumOp::Sub, *k)),
                        Opcode::Mul => Some((AccumOp::Mul, *k)),
                        _ => None,
                    },
                    _ => None,
                };
                if let Some((op, k_lit)) = pat {
                    let kf = k_lit as f64;
                    for v in map.values_mut() {
                        match v {
                            Val::Int(n) => {
                                *n = match op {
                                    AccumOp::Add => n.wrapping_add(k_lit),
                                    AccumOp::Sub => n.wrapping_sub(k_lit),
                                    AccumOp::Mul => n.wrapping_mul(k_lit),
                                }
                            }
                            Val::Float(x) => {
                                *x = match op {
                                    AccumOp::Add => *x + kf,
                                    AccumOp::Sub => *x - kf,
                                    AccumOp::Mul => *x * kf,
                                }
                            }
                            _ => {
                                let new =
                                    self.exec_lam_body_scratch(lam, v, lam_param, &mut scratch)?;
                                *v = new;
                            }
                        }
                    }
                    return Ok(Val::obj(map));
                }
                
                
                for v in map.values_mut() {
                    *v = crate::builtins::map_one(v, |item| {
                        self.exec_lam_body_scratch(lam, item, lam_param, &mut scratch)
                    })?;
                }
                Ok(Val::obj(map))
            }
            BuiltinMethod::FilterKeys => {
                let lam = sub.ok_or_else(|| EvalError("filterKeys: requires predicate".into()))?;
                let map = recv
                    .into_map()
                    .ok_or_else(|| EvalError("filterKeys: expected object".into()))?;
                let out = crate::builtins::filter_object_apply(map, |k, _v| {
                    crate::builtins::filter_one(&Val::Str(k.clone()), |item| {
                        self.exec_lam_body_scratch(lam, item, lam_param, &mut scratch)
                    })
                })?;
                Ok(Val::obj(out))
            }
            BuiltinMethod::FilterValues => {
                let lam =
                    sub.ok_or_else(|| EvalError("filterValues: requires predicate".into()))?;
                let map = recv
                    .into_map()
                    .ok_or_else(|| EvalError("filterValues: expected object".into()))?;
                let out = crate::builtins::filter_object_apply(map, |_k, v| {
                    crate::builtins::filter_one(v, |item| {
                        self.exec_lam_body_scratch(lam, item, lam_param, &mut scratch)
                    })
                })?;
                Ok(Val::obj(out))
            }
            BuiltinMethod::Pivot => call_builtin_method_compiled(self, recv, call, env),
            BuiltinMethod::Update => {
                let lam = sub.ok_or_else(|| EvalError("update: requires lambda".into()))?;
                self.exec_lam_body(lam, &recv, lam_param, env)
            }
            _ => call_builtin_method_compiled(self, recv, call, env),
        }
    }

    /// Execute a lambda body `prog` with `item` as the current value, creating a
    /// temporary scratch environment so the caller's `env` is unchanged.
    fn exec_lam_body(
        &mut self,
        prog: &Program,
        item: &Val,
        lam_param: Option<&str>,
        env: &Env,
    ) -> Result<Val, EvalError> {
        let mut scratch = env.clone();
        self.exec_lam_body_scratch(prog, item, lam_param, &mut scratch)
    }

    /// Execute a two-parameter lambda body (e.g. for `sort` comparators), binding
    /// `left` and `right` to the appropriate parameter names extracted from `arg`.
    fn exec_pair_lam_body(
        &mut self,
        prog: &Program,
        left: &Val,
        right: &Val,
        arg: &Arg,
        env: &Env,
    ) -> Result<Val, EvalError> {
        let mut scratch = env.clone();
        match arg {
            Arg::Pos(Expr::Lambda { params, .. }) | Arg::Named(_, Expr::Lambda { params, .. }) => {
                match params.as_slice() {
                    [] => {
                        let frame = scratch.push_lam(None, right.clone());
                        let result = self.exec(prog, &scratch);
                        scratch.pop_lam(frame);
                        result
                    }
                    [param] => {
                        let frame = scratch.push_lam(Some(param), right.clone());
                        let result = self.exec(prog, &scratch);
                        scratch.pop_lam(frame);
                        result
                    }
                    [left_name, right_name, ..] => {
                        let left_frame = scratch.push_lam(Some(left_name), left.clone());
                        let right_frame = scratch.push_lam(Some(right_name), right.clone());
                        let result = self.exec(prog, &scratch);
                        scratch.pop_lam(right_frame);
                        scratch.pop_lam(left_frame);
                        result
                    }
                }
            }
            _ => self.exec_lam_body_scratch(prog, right, None, &mut scratch),
        }
    }

    /// Apply a compiled patch to its root document: evaluate the root, then apply
    /// each operation in order, respecting optional guard conditions.
    fn exec_patch_compiled(&mut self, cp: &CompiledPatch, env: &Env) -> Result<Val, EvalError> {
        let mut doc = self.exec(&cp.root_prog, env)?;
        for op in &cp.ops {
            if let Some(cond) = &op.cond {
                let cenv = env.with_current(doc.clone());
                if !is_truthy(&self.exec(cond, &cenv)?) {
                    continue;
                }
            }
            match self.apply_patch_step_compiled(doc, &op.path, 0, &op.val, env)? {
                PatchResult::Replace(v) => doc = v,
                PatchResult::Delete => doc = Val::Null,
            }
        }
        Ok(doc)
    }

    /// Recursive patch walker: descend to `path[i]` in `v` and apply `val`.
    /// When `i == path.len()` the target has been reached and `val` is applied directly.
    fn apply_patch_step_compiled(
        &mut self,
        v: Val,
        path: &[CompiledPathStep],
        i: usize,
        val: &CompiledPatchVal,
        env: &Env,
    ) -> Result<PatchResult, EvalError> {
        if i == path.len() {
            return Ok(match val {
                CompiledPatchVal::Delete => PatchResult::Delete,
                CompiledPatchVal::Replace(prog) => {
                    let nv = self.exec(prog, &env.with_current(v))?;
                    PatchResult::Replace(nv)
                }
            });
        }
        match &path[i] {
            CompiledPathStep::Field(name) => {
                let mut m = v.into_map().unwrap_or_default();
                let existing = if let Some(slot) = m.get_mut(name.as_ref()) {
                    std::mem::replace(slot, Val::Null)
                } else {
                    Val::Null
                };
                let child = self.apply_patch_step_compiled(existing, path, i + 1, val, env)?;
                match child {
                    PatchResult::Delete => {
                        m.shift_remove(name.as_ref());
                    }
                    PatchResult::Replace(nv) => {
                        m.insert(name.clone(), nv);
                    }
                }
                Ok(PatchResult::Replace(Val::obj(m)))
            }
            CompiledPathStep::Index(idx) => {
                let mut a = v.into_vec().unwrap_or_default();
                let resolved = vm_resolve_idx(*idx, a.len() as i64);
                let existing = if resolved < a.len() {
                    std::mem::replace(&mut a[resolved], Val::Null)
                } else {
                    Val::Null
                };
                let child = self.apply_patch_step_compiled(existing, path, i + 1, val, env)?;
                match child {
                    PatchResult::Delete => {
                        if resolved < a.len() {
                            a.remove(resolved);
                        }
                    }
                    PatchResult::Replace(nv) => {
                        if resolved < a.len() {
                            a[resolved] = nv;
                        }
                    }
                }
                Ok(PatchResult::Replace(Val::arr(a)))
            }
            CompiledPathStep::DynIndex(prog) => {
                let idx_val = self.exec(prog, env)?;
                let idx = idx_val.as_i64().ok_or_else(|| {
                    EvalError(format!(
                        "patch dyn-index: expected integer, got {}",
                        idx_val.type_name()
                    ))
                })?;
                let mut a = v.into_vec().unwrap_or_default();
                let resolved = vm_resolve_idx(idx, a.len() as i64);
                let existing = if resolved < a.len() {
                    std::mem::replace(&mut a[resolved], Val::Null)
                } else {
                    Val::Null
                };
                let child = self.apply_patch_step_compiled(existing, path, i + 1, val, env)?;
                match child {
                    PatchResult::Delete => {
                        if resolved < a.len() {
                            a.remove(resolved);
                        }
                    }
                    PatchResult::Replace(nv) => {
                        if resolved < a.len() {
                            a[resolved] = nv;
                        }
                    }
                }
                Ok(PatchResult::Replace(Val::arr(a)))
            }
            CompiledPathStep::Wildcard => {
                let mut arr = v
                    .into_vec()
                    .ok_or_else(|| EvalError("patch [*]: expected array".into()))?;
                let mut write_idx = 0usize;
                for read_idx in 0..arr.len() {
                    let item = std::mem::replace(&mut arr[read_idx], Val::Null);
                    match self.apply_patch_step_compiled(item, path, i + 1, val, env)? {
                        PatchResult::Delete => {}
                        PatchResult::Replace(nv) => {
                            arr[write_idx] = nv;
                            write_idx += 1;
                        }
                    }
                }
                arr.truncate(write_idx);
                Ok(PatchResult::Replace(Val::arr(arr)))
            }
            CompiledPathStep::WildcardFilter(pred) => {
                let mut arr = v
                    .into_vec()
                    .ok_or_else(|| EvalError("patch [* if]: expected array".into()))?;
                let mut env_mut = env.clone();
                let mut write_idx = 0usize;
                for read_idx in 0..arr.len() {
                    let item = std::mem::replace(&mut arr[read_idx], Val::Null);
                    let frame = env_mut.push_lam(None, item.clone());
                    let include = match self.exec(pred, &env_mut) {
                        Ok(v) => is_truthy(&v),
                        Err(e) => {
                            env_mut.pop_lam(frame);
                            return Err(e);
                        }
                    };
                    env_mut.pop_lam(frame);
                    if include {
                        match self.apply_patch_step_compiled(item, path, i + 1, val, env)? {
                            PatchResult::Delete => {}
                            PatchResult::Replace(nv) => {
                                arr[write_idx] = nv;
                                write_idx += 1;
                            }
                        }
                    } else {
                        arr[write_idx] = item;
                        write_idx += 1;
                    }
                }
                arr.truncate(write_idx);
                Ok(PatchResult::Replace(Val::arr(arr)))
            }
            CompiledPathStep::Descendant(name) => {
                let v2 = self.descend_apply_patch_compiled(v, name, path, i, val, env)?;
                Ok(PatchResult::Replace(v2))
            }
        }
    }

    /// DFS patch application for `Descendant` path steps: recursively apply the
    /// operation wherever `name` appears in the subtree rooted at `v`.
    fn descend_apply_patch_compiled(
        &mut self,
        v: Val,
        name: &Arc<str>,
        path: &[CompiledPathStep],
        i: usize,
        val: &CompiledPatchVal,
        env: &Env,
    ) -> Result<Val, EvalError> {
        match v {
            Val::Obj(m) => {
                let mut map = Arc::try_unwrap(m).unwrap_or_else(|m| (*m).clone());
                let n = map.len();
                for idx in 0..n {
                    let child = if let Some((_, v)) = map.get_index_mut(idx) {
                        std::mem::replace(v, Val::Null)
                    } else {
                        continue;
                    };
                    let replaced =
                        self.descend_apply_patch_compiled(child, name, path, i, val, env)?;
                    if let Some((_, slot)) = map.get_index_mut(idx) {
                        *slot = replaced;
                    }
                }
                if map.contains_key(name.as_ref()) {
                    let existing = map.get(name.as_ref()).cloned().unwrap_or(Val::Null);
                    let r = self.apply_patch_step_compiled(existing, path, i + 1, val, env)?;
                    match r {
                        PatchResult::Delete => {
                            map.shift_remove(name.as_ref());
                        }
                        PatchResult::Replace(nv) => {
                            map.insert(name.clone(), nv);
                        }
                    }
                }
                Ok(Val::obj(map))
            }
            Val::Arr(a) => {
                let mut vec = Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone());
                for slot in vec.iter_mut() {
                    let old = std::mem::replace(slot, Val::Null);
                    *slot = self.descend_apply_patch_compiled(old, name, path, i, val, env)?;
                }
                Ok(Val::arr(vec))
            }
            other => Ok(other),
        }
    }

    /// Push `item` (and optionally bind it to `lam_param`) onto a scratch environment,
    /// execute `prog`, then pop the frame. Reusing `scratch` avoids re-cloning the base env
    /// on every iteration of high-frequency loops like `filter` and `map`.
    fn exec_lam_body_scratch(
        &mut self,
        prog: &Program,
        item: &Val,
        lam_param: Option<&str>,
        scratch: &mut Env,
    ) -> Result<Val, EvalError> {
        let frame = scratch.push_lam(lam_param, item.clone());
        let r = self.exec(prog, scratch);
        scratch.pop_lam(frame);
        r
    }

    /// Build an object `Val` from a slice of compiled field entries, handling all
    /// variants: `Short`, `Kv`, `KvPath`, `Dynamic`, `Spread`, and `SpreadDeep`.
    fn exec_make_obj(&mut self, entries: &[CompiledObjEntry], env: &Env) -> Result<Val, EvalError> {
        let mut map: IndexMap<Arc<str>, Val> = IndexMap::with_capacity(entries.len());
        for entry in entries {
            match entry {
                CompiledObjEntry::Short { name, ic } => {
                    let v = if let Some(v) = env.get_var(name.as_ref()) {
                        v.clone()
                    } else if let Val::Obj(m) = &env.current {
                        ic_get_field(m, name.as_ref(), ic)
                    } else {
                        env.current.get_field(name.as_ref())
                    };
                    if !v.is_null() {
                        map.insert(name.clone(), v);
                    }
                }
                CompiledObjEntry::Kv {
                    key,
                    prog,
                    optional,
                    cond,
                } => {
                    if let Some(c) = cond {
                        if !crate::util::is_truthy(&self.exec(c, env)?) {
                            continue;
                        }
                    }
                    let v = self.exec(prog, env)?;
                    if *optional && v.is_null() {
                        continue;
                    }
                    map.insert(key.clone(), v);
                }
                CompiledObjEntry::KvPath {
                    key,
                    steps,
                    optional,
                    ics,
                } => {
                    let mut v = env.current.clone();
                    for (i, st) in steps.iter().enumerate() {
                        v = match st {
                            KvStep::Field(f) => {
                                if let Val::Obj(m) = &v {
                                    ic_get_field(m, f.as_ref(), &ics[i])
                                } else {
                                    v.get_field(f.as_ref())
                                }
                            }
                            KvStep::Index(i) => v.get_index(*i),
                        };
                        if v.is_null() {
                            break;
                        }
                    }
                    if *optional && v.is_null() {
                        continue;
                    }
                    map.insert(key.clone(), v);
                }
                CompiledObjEntry::Dynamic { key, val } => {
                    let k: Arc<str> = Arc::from(val_to_key(&self.exec(key, env)?).as_str());
                    let v = self.exec(val, env)?;
                    map.insert(k, v);
                }
                CompiledObjEntry::Spread(prog) => {
                    if let Val::Obj(other) = self.exec(prog, env)? {
                        let entries = Arc::try_unwrap(other).unwrap_or_else(|m| (*m).clone());
                        for (k, v) in entries {
                            map.insert(k, v);
                        }
                    }
                }
                CompiledObjEntry::SpreadDeep(prog) => {
                    if let Val::Obj(other) = self.exec(prog, env)? {
                        let base = std::mem::take(&mut map);
                        let merged =
                            crate::util::deep_merge_concat(Val::obj(base), Val::Obj(other));
                        if let Val::Obj(m) = merged {
                            map = Arc::try_unwrap(m).unwrap_or_else(|m| (*m).clone());
                        }
                    }
                }
            }
        }
        Ok(Val::obj(map))
    }

    /// Evaluate a format-string from its compiled parts, applying optional format specs
    /// and fast-path shortcuts for simple `{@}`, `{@.field}`, and `{ident}` patterns.
    fn exec_fstring(&mut self, parts: &[CompiledFSPart], env: &Env) -> Result<Val, EvalError> {
        use std::fmt::Write as _;
        
        let lit_len: usize = parts
            .iter()
            .map(|p| match p {
                CompiledFSPart::Lit(s) => s.len(),
                CompiledFSPart::Interp { .. } => 8,
            })
            .sum();
        let mut out = String::with_capacity(lit_len);
        for part in parts {
            match part {
                CompiledFSPart::Lit(s) => out.push_str(s.as_ref()),
                CompiledFSPart::Interp { prog, fmt } => {
                    
                    
                    let val: Val = match &prog.ops[..] {
                        [Opcode::PushCurrent] => env.current.clone(),
                        [Opcode::PushCurrent, Opcode::GetIndex(n)] => match &env.current {
                            Val::Arr(a) => {
                                let idx = if *n >= 0 {
                                    *n as usize
                                } else {
                                    a.len().saturating_sub((-*n) as usize)
                                };
                                a.get(idx).cloned().unwrap_or(Val::Null)
                            }
                            _ => self.exec(prog, env)?,
                        },
                        [Opcode::PushCurrent, Opcode::GetField(k)] => match &env.current {
                            Val::Obj(m) => m.get(k.as_ref()).cloned().unwrap_or(Val::Null),
                            _ => self.exec(prog, env)?,
                        },
                        [Opcode::LoadIdent(name)] => {
                            env.get_var(name).cloned().unwrap_or(Val::Null)
                        }
                        _ => self.exec(prog, env)?,
                    };
                    match fmt {
                        None => match &val {
                            
                            Val::Str(s) => out.push_str(s.as_ref()),
                            Val::Int(n) => {
                                let _ = write!(out, "{}", n);
                            }
                            Val::Float(f) => {
                                let _ = write!(out, "{}", f);
                            }
                            Val::Bool(b) => {
                                let _ = write!(out, "{}", b);
                            }
                            Val::Null => out.push_str("null"),
                            _ => out.push_str(&val_to_string(&val)),
                        },
                        Some(FmtSpec::Spec(spec)) => {
                            out.push_str(&apply_fmt_spec(&val, spec));
                        }
                        Some(FmtSpec::Pipe(method)) => {
                            let piped = crate::builtins::eval_builtin_no_args(val, method)?;
                            match &piped {
                                Val::Str(s) => out.push_str(s.as_ref()),
                                Val::Int(n) => {
                                    let _ = write!(out, "{}", n);
                                }
                                Val::Float(f) => {
                                    let _ = write!(out, "{}", f);
                                }
                                Val::Bool(b) => {
                                    let _ = write!(out, "{}", b);
                                }
                                Val::Null => out.push_str("null"),
                                _ => out.push_str(&val_to_string(&piped)),
                            }
                        }
                    }
                }
            }
        }
        
        Ok(Val::Str(Arc::<str>::from(out)))
    }

    /// Evaluate `iter_prog` and expand the result into a `Vec<Val>` suitable for iteration.
    /// Objects are converted to `[{key, value}]` pairs; scalars become single-element vecs.
    fn exec_iter_vals(&mut self, iter_prog: &Program, env: &Env) -> Result<Vec<Val>, EvalError> {
        match self.exec(iter_prog, env)? {
            Val::Arr(a) => Ok(Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone())),
            Val::Obj(m) => {
                let entries = Arc::try_unwrap(m).unwrap_or_else(|m| (*m).clone());
                Ok(entries
                    .into_iter()
                    .map(|(k, v)| obj2("key", Val::Str(k), "value", v))
                    .collect())
            }
            other => Ok(vec![other]),
        }
    }
}


/// Return `true` when `op` is a "take the first element" selector (`?` quantifier
/// or a no-arg `.first()` call), used to short-circuit `Descendant` to `find_desc_first`.
fn is_first_selector_op(op: &Opcode) -> bool {
    match op {
        Opcode::Quantifier(QuantifierKind::First) => true,
        Opcode::CallMethod(c) if c.sub_progs.is_empty() && c.method == BuiltinMethod::First => true,
        _ => false,
    }
}

/// Slice an array or typed vector, resolving optional start/end indices with
/// Python-style negative-index semantics and clamping to valid bounds.
fn exec_slice(v: Val, from: Option<i64>, to: Option<i64>) -> Val {
    match v {
        Val::Arr(a) => {
            let len = a.len() as i64;
            let s = resolve_idx(from.unwrap_or(0), len);
            let e = resolve_idx(to.unwrap_or(len), len);
            let items = Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone());
            let s = s.min(items.len());
            let e = e.min(items.len());
            Val::arr(items[s..e].to_vec())
        }
        Val::IntVec(a) => {
            let len = a.len() as i64;
            let s = resolve_idx(from.unwrap_or(0), len);
            let e = resolve_idx(to.unwrap_or(len), len);
            let items = Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone());
            let s = s.min(items.len());
            let e = e.min(items.len());
            Val::int_vec(items[s..e].to_vec())
        }
        Val::FloatVec(a) => {
            let len = a.len() as i64;
            let s = resolve_idx(from.unwrap_or(0), len);
            let e = resolve_idx(to.unwrap_or(len), len);
            let items = Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone());
            let s = s.min(items.len());
            let e = e.min(items.len());
            Val::float_vec(items[s..e].to_vec())
        }
        _ => Val::Null,
    }
}

/// Resolve a signed index into a non-clamping usize offset (used for slice bounds).
fn resolve_idx(i: i64, len: i64) -> usize {
    (if i < 0 { (len + i).max(0) } else { i }) as usize
}

/// Recursively collect every value stored under `name` anywhere in the subtree of `v`,
/// without recording path information (used for non-root `Descendant` traversal).
fn collect_desc(v: &Val, name: &str, out: &mut Vec<Val>) {
    match v {
        Val::Obj(m) => {
            if let Some(v) = m.get(name) {
                out.push(v.clone());
            }
            for v in m.values() {
                collect_desc(v, name, out);
            }
        }
        Val::Arr(a) => {
            for item in a.as_ref() {
                collect_desc(item, name, out);
            }
        }
        _ => {}
    }
}


/// DFS pre-order search returning the first occurrence of `name` in the subtree of `v`.
/// Used to optimise `Descendant` when followed by a `.first()` selector.
fn find_desc_first(v: &Val, name: &str) -> Option<Val> {
    match v {
        Val::Obj(m) => {
            if let Some(v) = m.get(name) {
                return Some(v.clone());
            }
            for child in m.values() {
                if let Some(hit) = find_desc_first(child, name) {
                    return Some(hit);
                }
            }
            None
        }
        Val::Arr(a) => {
            for item in a.as_ref() {
                if let Some(hit) = find_desc_first(item, name) {
                    return Some(hit);
                }
            }
            None
        }
        _ => None,
    }
}

/// Collect every node in the subtree of `v` (DFS pre-order) into `out`.
/// Used by the `DescendAll` opcode to implement `$..**`.
fn collect_all(v: &Val, out: &mut Vec<Val>) {
    match v {
        Val::Obj(m) => {
            out.push(v.clone());
            for child in m.values() {
                collect_all(child, out);
            }
        }
        Val::Arr(a) => {
            for item in a.as_ref() {
                collect_all(item, out);
            }
        }
        other => out.push(other.clone()),
    }
}


/// Like `collect_desc` but also records `(JSON-pointer, value)` pairs in `cached`
/// for bulk insertion into the `PathCache` when traversing from the root document.
fn collect_desc_with_paths(
    v: &Val,
    name: &str,
    prefix: &mut String,
    out: &mut Vec<Val>,
    cached: &mut Vec<(Arc<str>, Val)>,
) {
    match v {
        Val::Obj(m) => {
            if let Some(found) = m.get(name) {
                let mut path = prefix.clone();
                path.push('/');
                path.push_str(name);
                out.push(found.clone());
                cached.push((Arc::from(path.as_str()), found.clone()));
            }
            for (k, child) in m.iter() {
                let prev = prefix.len();
                prefix.push('/');
                prefix.push_str(k.as_ref());
                collect_desc_with_paths(child, name, prefix, out, cached);
                prefix.truncate(prev);
            }
        }
        Val::Arr(a) => {
            for (i, item) in a.iter().enumerate() {
                let prev = prefix.len();
                prefix.push('/');
                let idx = i.to_string();
                prefix.push_str(&idx);
                collect_desc_with_paths(item, name, prefix, out, cached);
                prefix.truncate(prev);
            }
        }
        _ => {}
    }
}

/// Classification of a dict-comp key expression relative to the loop variable,
/// used to select a fast path that avoids a full `exec` call per iteration.
#[derive(Clone, Copy)]
enum DictKeyShape {
    /// Key is `v` (the loop variable itself) — use value directly as the key.
    Ident,
    /// Key is `v.to_string()` — coerce the loop variable to a string.
    IdentToString,
}

/// Detect whether the key program in a dict comprehension has a `DictKeyShape`
/// fast-path pattern relative to `vname`, or return `None` for the general case.
fn classify_dict_key(prog: &Program, vname: &str) -> Option<DictKeyShape> {
    match prog.ops.as_ref() {
        [Opcode::LoadIdent(v)] if v.as_ref() == vname => Some(DictKeyShape::Ident),
        [Opcode::LoadIdent(v), Opcode::CallMethod(call)]
            if v.as_ref() == vname
                && call.method == BuiltinMethod::ToString
                && call.sub_progs.is_empty()
                && call.orig_args.is_empty() =>
        {
            Some(DictKeyShape::IdentToString)
        }
        _ => None,
    }
}

/// Construct an iteration environment for a comprehension by binding `item` to
/// the variable names in `vars`. Single-var uses `{v: item}`, two-var uses the
/// `{index, value}` struct produced by object iteration.
fn bind_comp_vars(env: &Env, vars: &[Arc<str>], item: Val) -> Env {
    match vars {
        [] => env.with_current(item),
        [v] => {
            let mut e = env.with_var(v.as_ref(), item.clone());
            e.current = item;
            e
        }
        [v1, v2, ..] => {
            let idx = item.get("index").cloned().unwrap_or(Val::Null);
            let val = item.get("value").cloned().unwrap_or_else(|| item.clone());
            let mut e = env
                .with_var(v1.as_ref(), idx)
                .with_var(v2.as_ref(), val.clone());
            e.current = val;
            e
        }
    }
}

/// Perform an explicit type cast on `v` as specified by `ty` (`as str`, `as int`,
/// `as float`, `as bool`, `as array`, `as object`, `as null`).
fn exec_cast(v: &Val, ty: super::ast::CastType) -> Result<Val, EvalError> {
    use super::ast::CastType;
    match ty {
        CastType::Str => Ok(Val::Str(Arc::from(
            match v {
                Val::Null => "null".to_string(),
                Val::Bool(b) => b.to_string(),
                Val::Int(n) => n.to_string(),
                Val::Float(f) => f.to_string(),
                Val::Str(s) => s.to_string(),
                other => crate::util::val_to_string(other),
            }
            .as_str(),
        ))),
        CastType::Bool => Ok(Val::Bool(match v {
            Val::Null => false,
            Val::Bool(b) => *b,
            Val::Int(n) => *n != 0,
            Val::Float(f) => *f != 0.0,
            Val::Str(s) => !s.is_empty(),
            Val::StrSlice(r) => !r.is_empty(),
            Val::Arr(a) => !a.is_empty(),
            Val::IntVec(a) => !a.is_empty(),
            Val::FloatVec(a) => !a.is_empty(),
            Val::StrVec(a) => !a.is_empty(),
            Val::StrSliceVec(a) => !a.is_empty(),
            Val::ObjVec(d) => !d.cells.is_empty(),
            Val::Obj(o) => !o.is_empty(),
            Val::ObjSmall(p) => !p.is_empty(),
        })),
        CastType::Number | CastType::Float => match v {
            Val::Int(n) => Ok(Val::Float(*n as f64)),
            Val::Float(_) => Ok(v.clone()),
            Val::Str(s) => s
                .parse::<f64>()
                .map(Val::Float)
                .map_err(|e| EvalError(format!("as float: {}", e))),
            Val::Bool(b) => Ok(Val::Float(if *b { 1.0 } else { 0.0 })),
            Val::Null => Ok(Val::Float(0.0)),
            _ => err!("as float: cannot convert"),
        },
        CastType::Int => match v {
            Val::Int(_) => Ok(v.clone()),
            Val::Float(f) => Ok(Val::Int(*f as i64)),
            Val::Str(s) => s
                .parse::<i64>()
                .map(Val::Int)
                .or_else(|_| s.parse::<f64>().map(|f| Val::Int(f as i64)))
                .map_err(|e| EvalError(format!("as int: {}", e))),
            Val::Bool(b) => Ok(Val::Int(if *b { 1 } else { 0 })),
            Val::Null => Ok(Val::Int(0)),
            _ => err!("as int: cannot convert"),
        },
        CastType::Array => match v {
            Val::Arr(_) => Ok(v.clone()),
            Val::Null => Ok(Val::arr(Vec::new())),
            other => Ok(Val::arr(vec![other.clone()])),
        },
        CastType::Object => match v {
            Val::Obj(_) => Ok(v.clone()),
            _ => err!("as object: cannot convert non-object"),
        },
        CastType::Null => Ok(Val::Null),
    }
}

/// Format `val` according to the mini format spec string `spec` (e.g. `.2f`, `d`,
/// `>10`, `<10`, `^10`, `05`). Falls back to `val_to_string` for unknown specs.
fn apply_fmt_spec(val: &Val, spec: &str) -> String {
    if let Some(rest) = spec.strip_suffix('f') {
        if let Some(prec_str) = rest.strip_prefix('.') {
            if let Ok(prec) = prec_str.parse::<usize>() {
                if let Some(f) = val.as_f64() {
                    return format!("{:.prec$}", f);
                }
            }
        }
    }
    if spec == "d" {
        if let Some(i) = val.as_i64() {
            return format!("{}", i);
        }
    }
    let s = val_to_string(val);
    if let Some(w) = spec.strip_prefix('>').and_then(|s| s.parse::<usize>().ok()) {
        return format!("{:>w$}", s);
    }
    if let Some(w) = spec.strip_prefix('<').and_then(|s| s.parse::<usize>().ok()) {
        return format!("{:<w$}", s);
    }
    if let Some(w) = spec.strip_prefix('^').and_then(|s| s.parse::<usize>().ok()) {
        return format!("{:^w$}", s);
    }
    if let Some(w) = spec.strip_prefix('0').and_then(|s| s.parse::<usize>().ok()) {
        if let Some(i) = val.as_i64() {
            return format!("{:0>w$}", i);
        }
    }
    s
}




/// Compute a fast `u64` hash of a string using `DefaultHasher`.
fn hash_str(s: &str) -> u64 {
    let mut h = DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

/// Compute a structural hash of a `Val` that distinguishes both shape AND leaf values.
/// Two documents with the same shape but different primitive values must produce different
/// hashes so the path-resolution cache does not return stale results across distinct docs.
fn hash_val_structure(v: &Val) -> u64 {
    let mut h = DefaultHasher::new();
    hash_structure_into(v, &mut h, 0);
    h.finish()
}

/// Recursive helper for `hash_val_structure`. Hashing is bounded to depth 8 to
/// avoid O(n) cost on deeply nested documents while still capturing leaf values.
fn hash_structure_into(v: &Val, h: &mut DefaultHasher, depth: usize) {
    if depth > 8 {
        return;
    }
    match v {
        Val::Null => 0u8.hash(h),
        Val::Bool(b) => {
            1u8.hash(h);
            b.hash(h);
        }
        Val::Int(n) => {
            2u8.hash(h);
            n.hash(h);
        }
        Val::Float(f) => {
            3u8.hash(h);
            f.to_bits().hash(h);
        }
        Val::Str(s) => {
            4u8.hash(h);
            s.hash(h);
        }
        Val::StrSlice(r) => {
            4u8.hash(h);
            r.as_str().hash(h);
        }
        Val::Arr(a) => {
            5u8.hash(h);
            a.len().hash(h);
            for item in a.iter() {
                hash_structure_into(item, h, depth + 1);
            }
        }
        Val::IntVec(a) => {
            5u8.hash(h);
            a.len().hash(h);
            for n in a.iter() {
                2u8.hash(h);
                n.hash(h);
            }
        }
        Val::FloatVec(a) => {
            5u8.hash(h);
            a.len().hash(h);
            for f in a.iter() {
                3u8.hash(h);
                f.to_bits().hash(h);
            }
        }
        Val::StrVec(a) => {
            5u8.hash(h);
            a.len().hash(h);
            for s in a.iter() {
                4u8.hash(h);
                s.hash(h);
            }
        }
        Val::StrSliceVec(a) => {
            5u8.hash(h);
            a.len().hash(h);
            for r in a.iter() {
                4u8.hash(h);
                r.as_str().hash(h);
            }
        }
        Val::ObjVec(d) => {
            6u8.hash(h);
            d.nrows().hash(h);
            for k in d.keys.iter() {
                k.hash(h);
            }
        }
        Val::Obj(m) => {
            6u8.hash(h);
            m.len().hash(h);
            for (k, v) in m.iter() {
                k.hash(h);
                hash_structure_into(v, h, depth + 1);
            }
        }
        Val::ObjSmall(p) => {
            6u8.hash(h);
            p.len().hash(h);
            for (k, v) in p.iter() {
                k.hash(h);
                hash_structure_into(v, h, depth + 1);
            }
        }
    }
}
