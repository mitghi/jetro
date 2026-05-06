//! Pure-data opcode and program definitions for the Jetro VM.
//!
//! Holds compiled-program structures (`Opcode`, `Program`, `Compiled*`,
//! `FieldChainData`, comprehension specs, patch ops) and the small helpers
//! that operate only on those structures (`fresh_ics`, `hash_str`,
//! `disable_opcode_fusion`, `Program::new`). Execution and the `VM` struct
//! live in `super::exec`.

use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    sync::atomic::AtomicU64,
    sync::Arc,
};

use crate::parse::ast::*;
pub use crate::builtins::BuiltinMethod;


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
    CastOp(crate::parse::ast::CastType),

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

fn hash_str(s: &str) -> u64 {
    let mut h = DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}
