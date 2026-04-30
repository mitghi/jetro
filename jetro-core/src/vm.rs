//! High-performance bytecode VM for v2 Jetro expressions.
//!
//! # Architecture
//!
//! ```text
//!  String expression
//!        │  parser::parse()
//!        ▼
//!     Expr (AST)
//!        │  planner::plan_query()
//!        ├─ Pipeline IR when the expression shape allows it
//!        │
//!        │  Compiler::compile()
//!        ▼
//!     Program              ← flat Arc<[Opcode]>  (cached: compile_cache)
//!        │  VM::execute()
//!        ▼
//!      Val                 ← result              (structural: resolution_cache)
//! ```
//!
//! # Scalar VM responsibilities
//!
//! 1. **Compile cache** — parse + compile once per unique expression string.
//! 2. **Val type** — `Arc`-wrapped compound nodes; every clone is O(1).
//! 3. **BuiltinMethod enum** — O(1) method dispatch (jump-table vs string hash).
//! 4. **Pre-compiled sub-programs** — lambda/arg bodies compiled to `Arc<Program>`
//!    once at compile time; never re-compiled per call.
//! 5. **Resolution cache** — structural programs (`$.a.b[0]`) cache their
//!    pointer path after the first traversal; subsequent calls skip traversal.
//! 6. **Peephole pass 1 — RootChain** — `PushRoot + GetField+` fused into a
//!    single pointer-resolve opcode.
//! 7. **Pipeline handoff** — streamable chains are planned by `planner.rs`
//!    and executed by `pipeline.rs` / `composed.rs`; the VM remains the
//!    general scalar fallback.
//! 8. **Peephole pass 3 — ConstFold** — arithmetic on adjacent integer literals
//!    folded at compile time.
//! 9. **Stack machine** — iterative `exec()` loop; no per-opcode stack-frame
//!    overhead for simple navigation / arithmetic opcodes.

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
use crate::runtime::call_builtin_method_compiled;
use crate::util::{
    add_vals, cmp_vals_binop, is_truthy, kind_matches, num_op, obj2, val_to_key, val_to_string,
    vals_eq,
};
use crate::value::Val;

macro_rules! pop {
    ($stack:expr) => {
        $stack
            .pop()
            .ok_or_else(|| EvalError("stack underflow".into()))?
    };
}
macro_rules! err {
    ($($t:tt)*) => { Err(EvalError(format!($($t)*))) };
}

// ── Compiled sub-structures ───────────────────────────────────────────────────

/// A compiled method call stored inside `Opcode::CallMethod`.
#[derive(Debug, Clone)]
pub struct CompiledCall {
    pub method: BuiltinMethod,
    pub name: Arc<str>,
    /// Compiled lambda/expression sub-programs (one per arg, in order).
    pub sub_progs: Arc<[Arc<Program>]>,
    /// Original AST args kept for non-lambda dispatch fallback.
    pub orig_args: Arc<[Arg]>,
    /// Demand hint set by `pass_method_demand` peephole pass when this
    /// call is followed by a constant-arg `take(n)`. Filter / Map
    /// handlers read this and dispatch to `*_apply_bounded(items,
    /// Some(n), ...)` so the predicate / mapper stops after `n` keeps.
    /// `None` = no demand; behave as before.
    pub demand_max_keep: Option<usize>,
}

/// A compiled object field for `Opcode::MakeObj`.
#[derive(Debug, Clone)]
pub enum CompiledObjEntry {
    /// `{ name }` / `{ name, … }` shorthand — reads `env.current.name`
    /// (or a bound variable of that name).  `ic` is a per-entry inline
    /// cache hint so that repeated MakeObj calls over objects that share
    /// shape skip the IndexMap key-hash on hit.
    Short {
        name: Arc<str>,
        ic: Arc<AtomicU64>,
    },
    Kv {
        key: Arc<str>,
        prog: Arc<Program>,
        optional: bool,
        cond: Option<Arc<Program>>,
    },
    /// Specialised `Kv` where the value is a pure path from current:
    /// `{ key: @.a.b[0] }` compiles to `KvPath` so `exec_make_obj` can
    /// walk `env.current` through the pre-resolved steps without a
    /// sub-program exec.  `optional=true` mirrors `?` in the source —
    /// the field is omitted when the walk lands on `Null`.
    /// `ics[i]` is an inline-cache slot for `steps[i]` — only used when
    /// the step is `Field`.
    KvPath {
        key: Arc<str>,
        steps: Arc<[KvStep]>,
        optional: bool,
        ics: Arc<[AtomicU64]>,
    },
    Dynamic {
        key: Arc<Program>,
        val: Arc<Program>,
    },
    Spread(Arc<Program>),
    SpreadDeep(Arc<Program>),
}

/// Single step in a pre-resolved `KvPath` projection.
#[derive(Debug, Clone)]
pub enum KvStep {
    Field(Arc<str>),
    Index(i64),
}

/// A compiled f-string interpolation part.
#[derive(Debug, Clone)]
pub enum CompiledFSPart {
    Lit(Arc<str>),
    Interp {
        prog: Arc<Program>,
        fmt: Option<FmtSpec>,
    },
}

/// Compiled bind-object destructure spec.
#[derive(Debug, Clone)]
pub struct BindObjSpec {
    pub fields: Arc<[Arc<str>]>,
    pub rest: Option<Arc<str>>,
}

/// One step inside a `PipelineRun` opcode.  Forward-step evaluates
/// `prog` against an env where `current = pipe_value` and replaces the
/// pipe value with the result.  Bind-step stores the pipe value in the
/// local env's vars (the pipe value is unchanged).
#[derive(Debug, Clone)]
pub enum CompiledPipeStep {
    /// `| <expr>` — env.current = pipe_value; pipe_value = exec(prog).
    Forward(Arc<Program>),
    /// `-> name` — env.set_var(name, pipe_value); pipe_value unchanged.
    BindName(Arc<str>),
    /// `-> { f1, f2, …, …rest }` — destructure object into named vars.
    BindObj(Arc<BindObjSpec>),
    /// `-> [a, b, …]` — destructure array into named vars.
    BindArr(Arc<[Arc<str>]>),
}

/// Compiled comprehension spec.
#[derive(Debug, Clone)]
pub struct CompSpec {
    pub expr: Arc<Program>,
    pub vars: Arc<[Arc<str>]>,
    pub iter: Arc<Program>,
    pub cond: Option<Arc<Program>>,
}

#[derive(Debug, Clone)]
pub struct DictCompSpec {
    pub key: Arc<Program>,
    pub val: Arc<Program>,
    pub vars: Arc<[Arc<str>]>,
    pub iter: Arc<Program>,
    pub cond: Option<Arc<Program>>,
}

// ── Opcode ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum Opcode {
    // ── Literals ─────────────────────────────────────────────────────────────
    PushNull,
    PushBool(bool),
    PushInt(i64),
    PushFloat(f64),
    PushStr(Arc<str>),

    // ── Context ───────────────────────────────────────────────────────────────
    PushRoot,
    PushCurrent,

    // ── Navigation ────────────────────────────────────────────────────────────
    GetField(Arc<str>),
    GetIndex(i64),
    GetSlice(Option<i64>, Option<i64>),
    DynIndex(Arc<Program>),
    OptField(Arc<str>),
    Descendant(Arc<str>),
    DescendAll,
    InlineFilter(Arc<Program>),
    Quantifier(QuantifierKind),

    // ── Peephole fusions ──────────────────────────────────────────────────────
    /// PushRoot + GetField* fused — resolves chain via pointer arithmetic.
    RootChain(Arc<[Arc<str>]>),
    /// GetField(k1) + GetField(k2) + … fused — walks TOS through N fields.
    /// Applies mid-program where `RootChain` does not match (e.g. after a
    /// method call or filter produces an object on stack).
    /// Carries a per-step inline-cache array so that `map(a.b.c)` over a
    /// shape-uniform array of objects hits `get_index(cached_slot)`
    /// instead of re-hashing the key at every iteration.
    FieldChain(Arc<FieldChainData>),
    /// Fused `map(@.replace(lit, lit))` (and `replace_all`) — literal needle
    /// and replacement inlined; skips per-item sub_prog evaluation for arg
    /// strings.  `all=true` means replace every occurrence, else only first.
    /// Fused `map(@.upper().replace(lit, lit))` (and `replace_all`) — scan
    /// bytes once: ASCII-upper + memchr needle scan into a single pre-sized
    /// output String per item. Falls back to non-ASCII path for Unicode.
    /// Fused `map(@.lower().replace(lit, lit))` (and `replace_all`) — same as
    /// above but ASCII-lower.
    /// Fused `map(prefix + @ + suffix)` — per item, allocate exact-size
    /// `Arc<str>` with one uninit slice + copy_nonoverlapping. Either
    /// prefix or suffix may be empty for the 2-operand forms.
    /// Fused `map(@.split(sep).map(len).sum())` — emits IntVec of per-row
    /// sum-of-segment-char-lengths. Uses byte-scan (memchr/memmem) for ASCII
    /// source; falls back to char counting for Unicode.
    /// Fused `map({k1, k2, ..})` — map over an array projecting each object
    /// to a fixed set of `Short`-form fields (bare identifiers). Avoids the
    /// nested `MakeObj` dispatch per row and hoists key `Arc<str>` clones
    /// outside the inner loop. Uses one IC slot per key for shape lookup.
    /// Fused `map(@.split(sep).count())` — byte-scan per row, returns Int;
    /// zero per-row allocations.
    /// Fused `map(@.split(sep).count()).sum()` — scalar Int, no intermediate
    /// `[Int,Int,...]` array. One memchr-backed scan per row, accumulated.
    /// Fused `map(@.split(sep).first())` — first segment only; one Arc per
    /// row instead of N.
    /// Fused `map(@.split(sep).nth(n))` — nth segment; one Arc per row.
    // Map+<aggregate> fusions deleted in Tier 3.

    // ── Field-specialised fusions (Tier 3) ────────────────────────────────────
    // MapFieldSum / Avg / Min / Max migrated to pipeline.rs
    // Sink::NumMap(op, prog).  See memory/project_opcode_migration.md.
    // MapField / MapFieldChain / MapFieldUnique / MapFieldChainUnique /
    // FlatMapChain fused opcodes deleted in Tier 3.  Pipeline IR Stage::Map
    // + BodyKernel::FieldRead / FieldChain / + Stage::FlatMap covers them.

    // ── Filter predicate specialisation deleted in Tier 3 ────────────────────
    // FilterFieldEqLit / FilterFieldCmpLit / FilterCurrentCmpLit /
    // FilterFieldCmpField / FilterFieldCmpFieldCount /
    // FilterFieldEqLitMapField / FilterFieldCmpLitMapField /
    // FilterFieldsAllEqLitCount / FilterFieldsAllCmpLitCount /
    // FilterFieldEqLitCount / FilterFieldCmpLitCount — all replaced by
    // pipeline IR Stage::Filter + composed substrate (auto-index +
    // columnar IntVec/FloatVec/StrVec lanes preserved through Stage
    // dispatch).
    // FilterStrVec* (StartsWith/EndsWith/Contains) + MapStrVec* (Upper/Lower/Trim)
    // + MapNumVecArith + MapNumVecNeg deleted in Tier 3.  Pipeline IR
    // Stage::Filter / Stage::Map + composed substrate dispatch through
    // base CallMethod chain on StrVec/IntVec/FloatVec receivers.

    // ── Ident lookup (var, then current field) ────────────────────────────────
    LoadIdent(Arc<str>),

    // ── Binary / unary ops ────────────────────────────────────────────────────
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    Neq,
    Lt,
    Lte,
    Gt,
    Gte,
    Fuzzy,
    Not,
    Neg,

    // ── Type cast (`as T`) ───────────────────────────────────────────────────
    CastOp(super::ast::CastType),

    // ── Short-circuit ops (embed rhs as sub-program) ──────────────────────────
    AndOp(Arc<Program>),
    OrOp(Arc<Program>),
    CoalesceOp(Arc<Program>),

    // ── Method calls ─────────────────────────────────────────────────────────
    CallMethod(Arc<CompiledCall>),
    CallOptMethod(Arc<CompiledCall>),

    // ── Construction ─────────────────────────────────────────────────────────
    MakeObj(Arc<[CompiledObjEntry]>),
    /// Array literal `[e0, e1, ...]`.  Each entry carries a spread
    /// flag — `true` means the element's program produces an iterable
    /// whose contents are flattened into the result.
    MakeArr(Arc<[(Arc<Program>, bool)]>),

    // ── F-string ─────────────────────────────────────────────────────────────
    FString(Arc<[CompiledFSPart]>),

    // ── Kind check ───────────────────────────────────────────────────────────
    KindCheck {
        ty: KindType,
        negate: bool,
    },

    // ── Pipeline helpers ──────────────────────────────────────────────────────
    /// Pop TOS → env.current, then push it back (pass-through with context update).
    SetCurrent,
    /// TOS → env var by name, TOS remains (for `->` bind).
    BindVar(Arc<str>),
    /// Pop TOS → env var (for `let` init).
    StoreVar(Arc<str>),
    /// Object destructure bind: TOS obj → multiple vars.
    BindObjDestructure(Arc<BindObjSpec>),
    /// Array destructure bind: TOS arr → multiple vars.
    BindArrDestructure(Arc<[Arc<str>]>),
    /// Single-opcode pipeline `base | rhs1 | rhs2 -> n | rhs3 ...` —
    /// runs `base`, threads the value through each step under a local
    /// mutable Env (forward: env.current = val + run rhs; bind: store
    /// in env vars, value unchanged).  Replaces the previous SetCurrent
    /// / BindVar / Bind*Destructure no-op chain in the flat opcode
    /// stream.
    PipelineRun {
        base: Arc<Program>,
        steps: Arc<[CompiledPipeStep]>,
    },

    // ── Complex (recursive sub-programs) ─────────────────────────────────────
    LetExpr {
        name: Arc<str>,
        body: Arc<Program>,
    },
    /// Python-style ternary: TOS is cond; branch into `then_` or `else_`.
    /// Short-circuits — only the taken branch is executed.
    IfElse {
        then_: Arc<Program>,
        else_: Arc<Program>,
    },
    /// `try BODY else DEFAULT` — execute `body`; on `EvalError` or
    /// `Val::Null` result, execute `default`.  Both subprograms are
    /// isolated; only one's value lands on the stack.
    TryExpr {
        body: Arc<Program>,
        default: Arc<Program>,
    },
    ListComp(Arc<CompSpec>),
    DictComp(Arc<DictCompSpec>),
    SetComp(Arc<CompSpec>),

    // ── Resolution cache fast-path ────────────────────────────────────────────
    GetPointer(Arc<str>),

    // ── Patch block ──────────────────────────────────────────────────────────
    /// Patch block — pre-compiled form.  Tree-walker no longer used;
    /// all sub-expressions live as `Arc<Program>` and run via VM.
    PatchEval(Arc<CompiledPatch>),
    /// Sentinel for `DELETE` outside a patch — runtime raises error.
    DeleteMarkErr,
}

// ── Program ───────────────────────────────────────────────────────────────────

/// A compiled, immutable v2 program.  Cheap to clone (`Arc` internals).
#[derive(Debug, Clone)]
pub struct Program {
    pub ops: Arc<[Opcode]>,
    pub source: Arc<str>,
    pub id: u64,
    /// True when the program contains only structural navigation opcodes
    /// (eligible for resolution caching).
    pub is_structural: bool,
    /// Inline caches — one `AtomicU64` slot per opcode.  Populated by
    /// `Opcode::GetField` / `Opcode::OptField` / `Opcode::FieldChain`.
    ///
    /// Encoding: `stored_slot = slot_idx + 1` (0 reserved for "unset").
    /// No Arc-ptr gating — the hit path is `get_index(slot)` + byte-eq
    /// key verify.  That lets a single slot survive across different
    /// `Arc<IndexMap>` instances of the same shape, which is the common
    /// case for repeated queries over distinct docs and for shape-uniform
    /// array iteration reaching the opcode inside a sub-program.
    pub ics: Arc<[AtomicU64]>,
}

// ── Patch runtime helpers ──────────────────────────────────────────

#[derive(Debug)]
enum PatchResult {
    Replace(Val),
    Delete,
}

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

// ── Compiled patch (VM-native) ─────────────────────────────────────

/// Pre-compiled form of `Expr::Patch`.  Every sub-expression is a
/// `vm::Program`; the runtime patch executor uses `self.exec` to
/// evaluate them.
#[derive(Debug, Clone)]
pub struct CompiledPatch {
    pub root_prog: Arc<Program>,
    pub ops: Vec<CompiledPatchOp>,
}

#[derive(Debug, Clone)]
pub struct CompiledPatchOp {
    pub path: Vec<CompiledPathStep>,
    pub val: CompiledPatchVal,
    pub cond: Option<Arc<Program>>,
}

#[derive(Debug, Clone)]
pub enum CompiledPatchVal {
    /// Replace leaf with the value produced by `prog` (with `@` = leaf).
    Replace(Arc<Program>),
    /// Sentinel: delete the leaf entry.
    Delete,
}

#[derive(Debug, Clone)]
pub enum CompiledPathStep {
    Field(Arc<str>),
    Index(i64),
    DynIndex(Arc<Program>),
    Wildcard,
    WildcardFilter(Arc<Program>),
    Descendant(Arc<str>),
}

impl Program {
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
                    | Opcode::GetPointer(_)
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

/// Per-step inline caches for `Opcode::FieldChain`.  One `AtomicU64` slot per
/// key in the chain — same encoding as `Program.ics` (`slot_idx + 1`, 0 unset).
/// Lives inside the opcode rather than the top-level side-table because the
/// chain length is known only at compile time of that specific opcode.
#[derive(Debug)]
pub struct FieldChainData {
    pub keys: Arc<[Arc<str>]>,
    pub ics: Box<[AtomicU64]>,
}

impl FieldChainData {
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
    #[inline]
    pub fn len(&self) -> usize {
        self.keys.len()
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
}

impl std::ops::Deref for FieldChainData {
    type Target = [Arc<str>];
    #[inline]
    fn deref(&self) -> &[Arc<str>] {
        &self.keys
    }
}

/// Build a fresh IC side-table with one zeroed `AtomicU64` per opcode.
/// Kept public so other modules that fabricate `Program` values (schema
/// specialisation, analysis passes) can populate the field.
pub fn fresh_ics(len: usize) -> Arc<[AtomicU64]> {
    let mut v = Vec::with_capacity(len);
    for _ in 0..len {
        v.push(AtomicU64::new(0));
    }
    v.into()
}

/// Look up `key` in `m`, using the IC slot as a speculative hint.
///
/// IC is ptr-independent: slot survives across different `Arc<IndexMap>`
/// instances as long as shape (key ordering) is the same.  Hit path is
/// `get_index(slot) + byte-eq key verify`; miss path is one `get_full`
/// that also refreshes the slot.  Slot is encoded as `idx + 1` so zero
/// stays reserved for "unset".
/// Migration gate for opcode-level fusion.  When the env var
/// `JETRO_DISABLE_OPCODE_FUSION` is set (any value), `Compiler::optimize`
/// passes that emit fused/specialised opcodes become no-ops, leaving
/// the unfused forms in place.  Used to validate that
/// `pipeline.rs` rewrite rules cover every shape an opcode currently
/// fuses.  See `memory/project_opcode_migration.md`.
///
/// Read once per call (env-var lookup is fast but not free); callers
/// invoke this from the per-pass entry, not the per-row inner loop.
#[inline]
fn disable_opcode_fusion() -> bool {
    std::env::var_os("JETRO_DISABLE_OPCODE_FUSION").is_some()
}

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

/// Accumulate lambda pattern tag — selects which fused binop to run.
#[derive(Copy, Clone)]
enum AccumOp {
    Add,
    Sub,
    Mul,
}

// ── Typed-numeric aggregate fast-paths ────────────────────────────────────────
// Direct loops over `&[Val]` for mono-typed or mixed Int/Float arrays.  Used
// by the bare `.sum()/.min()/.max()/.avg()` no-arg method-call fast path in
// `exec_call` — skips fallback dispatch, `into_vec()` clone, and the extra
// `.filter().collect()` that `func_aggregates::collect_nums` performs.
//
// Semantics match `func_aggregates`: non-numeric items are skipped; Int-only
// arrays stay on `i64` (no lossy widening); Float appearance widens once.

/// 4-lane unrolled i64 sum.  LLVM emits SIMD horizontal-reduce for
/// `chunks_exact(4)` over independent accumulators.
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

fn agg_sum_typed(a: &[Val]) -> Val {
    // Tight i64 loop until first Float; then switch to f64 loop.
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
            _ => {} // skip non-numeric
        }
    }
    Val::Int(i_acc)
}

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

#[inline]
fn agg_minmax_typed(a: &[Val], want_max: bool) -> Val {
    let mut it = a.iter();
    // Find first number.
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
            // Mono-Int tight loop; on first Float, promote.
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

/// `&str`-keyed variant of `lookup_field_cached`; ptr-eq shortcut is skipped
/// (caller doesn't hold an `Arc<str>`), so the hit path is byte-eq only.
// ── Variable context (compile-time) ──────────────────────────────────────────

#[derive(Clone, Default)]
struct VarCtx {
    known: SmallVec<[Arc<str>; 4]>,
}

impl VarCtx {
    fn with_var(&self, name: &str) -> Self {
        let mut v = self.clone();
        if !v.known.iter().any(|k| k.as_ref() == name) {
            v.known.push(Arc::from(name));
        }
        v
    }
    fn with_vars(&self, names: &[String]) -> Self {
        let mut v = self.clone();
        for n in names {
            if !v.known.iter().any(|k| k.as_ref() == n.as_str()) {
                v.known.push(Arc::from(n.as_str()));
            }
        }
        v
    }
    fn has(&self, name: &str) -> bool {
        self.known.iter().any(|k| k.as_ref() == name)
    }
}

// ── Compiler ─────────────────────────────────────────────────────────────────

pub struct Compiler;

impl Compiler {
    pub fn compile(expr: &Expr, source: &str) -> Program {
        let mut e = expr.clone();
        Self::reorder_and_operands(&mut e);
        let ctx = VarCtx::default();
        let ops = Self::optimize(Self::emit(&e, &ctx));
        let prog = Program::new(ops, source);
        // Post-pass: canonicalise identical sub-programs.
        let deduped = super::analysis::dedup_subprograms(&prog);
        let ics = fresh_ics(deduped.ops.len());
        Program {
            ops: deduped.ops.clone(),
            source: prog.source,
            id: prog.id,
            is_structural: prog.is_structural,
            ics,
        }
    }

    /// AST rewrite: for each `a and b`, if `b` is more selective than `a`,
    /// swap operands so the cheaper/selective predicate runs first.  Safe
    /// because `and` is commutative on pure, side-effect-free expressions.
    fn reorder_and_operands(expr: &mut Expr) {
        use super::analysis::selectivity_score;
        match expr {
            Expr::BinOp(l, op, r) if *op == BinOp::And => {
                Self::reorder_and_operands(l);
                Self::reorder_and_operands(r);
                if selectivity_score(r) < selectivity_score(l) {
                    std::mem::swap(l, r);
                }
            }
            Expr::BinOp(l, _, r) => {
                Self::reorder_and_operands(l);
                Self::reorder_and_operands(r);
            }
            Expr::UnaryNeg(e) | Expr::Not(e) | Expr::Kind { expr: e, .. } => {
                Self::reorder_and_operands(e)
            }
            Expr::Coalesce(l, r) => {
                Self::reorder_and_operands(l);
                Self::reorder_and_operands(r);
            }
            Expr::Chain(base, steps) => {
                Self::reorder_and_operands(base);
                for s in steps {
                    match s {
                        super::ast::Step::DynIndex(e) | super::ast::Step::InlineFilter(e) => {
                            Self::reorder_and_operands(e)
                        }
                        super::ast::Step::Method(_, args)
                        | super::ast::Step::OptMethod(_, args) => {
                            for a in args {
                                match a {
                                    super::ast::Arg::Pos(e) | super::ast::Arg::Named(_, e) => {
                                        Self::reorder_and_operands(e)
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            Expr::Let { init, body, .. } => {
                Self::reorder_and_operands(init);
                Self::reorder_and_operands(body);
            }
            Expr::Pipeline { base, steps } => {
                Self::reorder_and_operands(base);
                for s in steps {
                    if let super::ast::PipeStep::Forward(e) = s {
                        Self::reorder_and_operands(e);
                    }
                }
            }
            Expr::Object(fields) => {
                for f in fields {
                    match f {
                        super::ast::ObjField::Kv { val, .. } => Self::reorder_and_operands(val),
                        super::ast::ObjField::Dynamic { key, val } => {
                            Self::reorder_and_operands(key);
                            Self::reorder_and_operands(val);
                        }
                        super::ast::ObjField::Spread(e) => Self::reorder_and_operands(e),
                        _ => {}
                    }
                }
            }
            Expr::Array(elems) => {
                for e in elems {
                    match e {
                        super::ast::ArrayElem::Expr(e) | super::ast::ArrayElem::Spread(e) => {
                            Self::reorder_and_operands(e)
                        }
                    }
                }
            }
            Expr::ListComp {
                expr, iter, cond, ..
            }
            | Expr::SetComp {
                expr, iter, cond, ..
            }
            | Expr::GenComp {
                expr, iter, cond, ..
            } => {
                Self::reorder_and_operands(expr);
                Self::reorder_and_operands(iter);
                if let Some(c) = cond {
                    Self::reorder_and_operands(c);
                }
            }
            Expr::DictComp {
                key,
                val,
                iter,
                cond,
                ..
            } => {
                Self::reorder_and_operands(key);
                Self::reorder_and_operands(val);
                Self::reorder_and_operands(iter);
                if let Some(c) = cond {
                    Self::reorder_and_operands(c);
                }
            }
            Expr::Lambda { body, .. } => Self::reorder_and_operands(body),
            Expr::GlobalCall { args, .. } => {
                for a in args {
                    match a {
                        super::ast::Arg::Pos(e) | super::ast::Arg::Named(_, e) => {
                            Self::reorder_and_operands(e)
                        }
                    }
                }
            }
            _ => {}
        }
    }

    pub fn compile_str(input: &str) -> Result<Program, EvalError> {
        let expr = super::parser::parse(input).map_err(|e| EvalError(e.to_string()))?;
        Ok(Self::compile(&expr, input))
    }

    /// Compile with explicit pass configuration.  Cached by callers
    /// under `(config.hash(), expr)`.
    pub fn compile_str_with_config(input: &str, config: PassConfig) -> Result<Program, EvalError> {
        let expr = super::parser::parse(input).map_err(|e| EvalError(e.to_string()))?;
        let mut e = expr.clone();
        if config.reorder_and {
            Self::reorder_and_operands(&mut e);
        }
        let ctx = VarCtx::default();
        let ops = Self::optimize_with(Self::emit(&e, &ctx), config);
        let prog = Program::new(ops, input);
        if config.dedup_subprogs {
            let deduped = super::analysis::dedup_subprograms(&prog);
            let ics = fresh_ics(deduped.ops.len());
            Ok(Program {
                ops: deduped.ops.clone(),
                source: prog.source,
                id: prog.id,
                is_structural: prog.is_structural,
                ics,
            })
        } else {
            Ok(prog)
        }
    }

    // ── Peephole optimizer ────────────────────────────────────────────────────

    fn optimize(ops: Vec<Opcode>) -> Vec<Opcode> {
        Self::optimize_with(ops, PassConfig::default())
    }

    fn optimize_with(ops: Vec<Opcode>, cfg: PassConfig) -> Vec<Opcode> {
        // Migration gate: when JETRO_DISABLE_OPCODE_FUSION is set,
        // skip the fusion-emitting passes so pipeline.rs rewrite
        // rules become the sole fusion mechanism.  Correctness-level
        // passes (const fold / nullness / method-const / kind check /
        // strength reduce / redundant-ops / equi-join — which is a
        // method dispatch fast-path not a chain fusion) keep running.
        // See `memory/project_opcode_migration.md`.
        let no_fusion = disable_opcode_fusion();
        let ops = if cfg.root_chain && !no_fusion {
            Self::pass_root_chain(ops)
        } else {
            ops
        };
        let ops = if cfg.field_chain && !no_fusion {
            Self::pass_field_chain(ops)
        } else {
            ops
        };
        // Tier 3: pass_filter_count / pass_find_quantifier / pass_filter_fusion
        // / pass_string_chain_fusion deleted — composed substrate handles
        // every chain shape via base CallMethod opcodes.
        let ops = if cfg.filter_fusion {
            Self::pass_field_specialise(ops)
        } else {
            ops
        };
        let ops = if !no_fusion {
            Self::pass_list_comp_specialise(ops)
        } else {
            ops
        };
        let ops = if cfg.strength_reduce {
            Self::pass_strength_reduce(ops)
        } else {
            ops
        };
        let ops = if cfg.redundant_ops {
            Self::pass_redundant_ops(ops)
        } else {
            ops
        };
        let ops = if cfg.kind_check_fold {
            Self::pass_kind_check_fold(ops)
        } else {
            ops
        };
        let ops = if cfg.method_const {
            Self::pass_method_const_fold(ops)
        } else {
            ops
        };
        let ops = if cfg.const_fold {
            Self::pass_const_fold(ops)
        } else {
            ops
        };
        let ops = if cfg.nullness {
            Self::pass_nullness_opt_field(ops)
        } else {
            ops
        };
        let ops = if !no_fusion {
            Self::pass_method_demand(ops)
        } else {
            ops
        };
        ops
    }

    /// Demand propagation peephole — when an array-yielding CallMethod
    /// (Filter / Map / FlatMap) is immediately followed by a CallMethod
    /// to `take` with a constant `Int(n)` arg, fold the demand into
    /// the producer's `demand_max_keep` field and drop the take call.
    /// The producer's runtime handler then dispatches to
    /// `*_apply_bounded(items, Some(n), ...)` and stops after `n`
    /// items pass — no full-array materialisation.
    ///
    /// Generic algorithm: parameterised over (method, demand). No
    /// per-shape `FilterTake` / `MapTake` opcode invented; the demand
    /// hint travels in the existing `CompiledCall` struct.
    fn pass_method_demand(ops: Vec<Opcode>) -> Vec<Opcode> {
        // Extract `n` if `call` is `take(Int(n))` with no other args.
        fn take_const(call: &CompiledCall) -> Option<usize> {
            use crate::ast::{Arg, Expr};
            if call.name.as_ref() != "take" {
                return None;
            }
            if call.orig_args.len() != 1 {
                return None;
            }
            match &call.orig_args[0] {
                Arg::Pos(Expr::Int(n)) if *n >= 0 => Some(*n as usize),
                _ => None,
            }
        }
        // Producer must be a per-row 1:1 / filtering call whose handler
        // honours `demand_max_keep`.
        fn is_demand_aware(method: BuiltinMethod) -> bool {
            matches!(method, BuiltinMethod::Filter | BuiltinMethod::Map)
        }
        let mut out = Vec::with_capacity(ops.len());
        let mut i = 0;
        while i < ops.len() {
            if i + 1 < ops.len() {
                if let (Opcode::CallMethod(a), Opcode::CallMethod(b)) = (&ops[i], &ops[i + 1]) {
                    if is_demand_aware(a.method) && a.demand_max_keep.is_none() {
                        if let Some(n) = take_const(b) {
                            // Rewrite `a` with demand; drop `b`.
                            let mut new_call = (**a).clone();
                            new_call.demand_max_keep = Some(n);
                            out.push(Opcode::CallMethod(Arc::new(new_call)));
                            i += 2;
                            continue;
                        }
                    }
                }
            }
            out.push(ops[i].clone());
            i += 1;
        }
        out
    }

    // pass_equi_join_fusion deleted — composed substrate runs base
    // CallMethod(equi_join, [rhs, lk, rk]) via the builtin call path.

    /// Nullness-driven: when the preceding op provably leaves a non-null
    /// receiver on the stack, rewrite `OptField(k)` → `GetField(k)`.
    /// Conservative: only folds when the predecessor is a construction
    /// opcode (MakeObj / MakeArr / PushStr / RootChain / GetField).
    fn pass_nullness_opt_field(ops: Vec<Opcode>) -> Vec<Opcode> {
        let mut out: Vec<Opcode> = Vec::with_capacity(ops.len());
        for op in ops {
            if let Opcode::OptField(k) = &op {
                // Only safe when receiver is provably a non-null object —
                // MakeObj is the only opcode that guarantees this without
                // a schema.  Other cases are handled by schema::specialize.
                let non_null = matches!(out.last(), Some(Opcode::MakeObj(_)));
                if non_null {
                    out.push(Opcode::GetField(k.clone()));
                    continue;
                }
            }
            out.push(op);
        }
        out
    }

    /// Fold built-in methods when receiver is a literal with known length/content:
    ///   PushStr(s) + .len()    → PushInt(utf8 char count)
    ///   PushStr(s) + .upper()  → PushStr(upper)
    ///   PushStr(s) + .lower()  → PushStr(lower)
    ///   PushStr(s) + .trim()   → PushStr(trim)
    ///   MakeArr(n elems) + .len()  → PushInt(n)  (only for non-spread arrays)
    fn pass_method_const_fold(ops: Vec<Opcode>) -> Vec<Opcode> {
        let mut out: Vec<Opcode> = Vec::with_capacity(ops.len());
        for op in ops {
            if let Opcode::CallMethod(c) = &op {
                if c.sub_progs.is_empty() {
                    match (out.last(), c.method) {
                        (Some(Opcode::PushStr(s)), BuiltinMethod::Len) => {
                            let n = s.chars().count() as i64;
                            out.pop();
                            out.push(Opcode::PushInt(n));
                            continue;
                        }
                        (Some(Opcode::PushStr(s)), BuiltinMethod::Upper) => {
                            let u: Arc<str> = Arc::from(s.to_uppercase());
                            out.pop();
                            out.push(Opcode::PushStr(u));
                            continue;
                        }
                        (Some(Opcode::PushStr(s)), BuiltinMethod::Lower) => {
                            let u: Arc<str> = Arc::from(s.to_lowercase());
                            out.pop();
                            out.push(Opcode::PushStr(u));
                            continue;
                        }
                        (Some(Opcode::PushStr(s)), BuiltinMethod::Trim) => {
                            let u: Arc<str> = Arc::from(s.trim());
                            out.pop();
                            out.push(Opcode::PushStr(u));
                            continue;
                        }
                        (Some(Opcode::MakeArr(progs)), BuiltinMethod::Len) => {
                            // Fold to a constant only when no entry is a
                            // spread — spreads expand at runtime so the
                            // length is dynamic.
                            if progs.iter().all(|(_, sp)| !*sp) {
                                let n = progs.len() as i64;
                                out.pop();
                                out.push(Opcode::PushInt(n));
                                continue;
                            }
                        }
                        _ => {}
                    }
                }
            }
            out.push(op);
        }
        out
    }

    /// Fold `KindCheck` when its input type is a literal push:
    ///   PushInt(n)  + KindCheck{number, neg} → PushBool(!neg)
    ///   PushStr(_)  + KindCheck{string, neg} → PushBool(!neg)
    ///   PushNull    + KindCheck{null, neg}   → PushBool(!neg)
    ///   PushBool(_) + KindCheck{bool, neg}   → PushBool(!neg)
    ///   mismatches fold to opposite.
    fn pass_kind_check_fold(ops: Vec<Opcode>) -> Vec<Opcode> {
        use super::analysis::{fold_kind_check, VType};
        let mut out = Vec::with_capacity(ops.len());
        for op in ops {
            if let Opcode::KindCheck { ty, negate } = &op {
                let prev_ty: Option<VType> = match out.last() {
                    Some(Opcode::PushNull) => Some(VType::Null),
                    Some(Opcode::PushBool(_)) => Some(VType::Bool),
                    Some(Opcode::PushInt(_)) => Some(VType::Int),
                    Some(Opcode::PushFloat(_)) => Some(VType::Float),
                    Some(Opcode::PushStr(_)) => Some(VType::Str),
                    Some(Opcode::MakeArr(_)) => Some(VType::Arr),
                    Some(Opcode::MakeObj(_)) => Some(VType::Obj),
                    _ => None,
                };
                if let Some(vt) = prev_ty {
                    if let Some(b) = fold_kind_check(vt, *ty, *negate) {
                        out.pop();
                        out.push(Opcode::PushBool(b));
                        continue;
                    }
                }
            }
            out.push(op);
        }
        out
    }

    /// Lower generic fused opcodes to field-specialised variants when the
    /// sub-program is a trivial `GetField(k)` read. Runs AFTER
    /// pass_find_quantifier / pass_filter_count so those passes see the
    /// generic `CallMethod(Filter)` / `MapSum` forms first.
    ///
    /// Migration gate (`JETRO_DISABLE_OPCODE_FUSION=1`): when set, this
    /// pass becomes a no-op, leaving the unfused opcode in place so
    /// pipeline.rs rewrite rules become the sole fusion mechanism.
    /// Used to validate opcode → rule migration without permanently
    /// deleting code; see `memory/project_opcode_migration.md`.
    fn pass_field_specialise(ops: Vec<Opcode>) -> Vec<Opcode> {
        if disable_opcode_fusion() {
            return ops;
        }
        let mut out2: Vec<Opcode> = Vec::with_capacity(ops.len());
        for op in ops {
            match op {
                // MapSum/Avg/Min/Max specialisations migrated to
                // pipeline.rs `Sink::NumMap(NumOp::*)` rule (see
                // memory/project_opcode_migration.md).  Unfused MapSum
                // / MapAvg / MapMin / MapMax fall through to their
                // generic handlers below for queries pipeline can't
                // lower (sub-programs, non-Root chains).
                // FilterFieldsAllEqLitCount / FilterFieldsAllCmpLitCount
                // (compound-AND filter+count specialisations) migrated to
                // pipeline.rs count reducer with and_chain_prog decoder.
                Opcode::CallMethod(ref b) => {
                    // MapField / MapFieldChain fusion deleted in Tier 3 —
                    // pipeline IR Stage::Map + BodyKernel::FieldRead /
                    // FieldChain covers `map(k)` and `map(a.b.c)`.
                    let _ = b;
                    // GroupByField/CountByField/UniqueByField fusion
                    // deleted in Tier 3 — base CallMethod chain.
                    // Filter field/current cmp-lit fusion deleted in Tier 3 —
                    // pipeline IR Stage::Filter + composed substrate covers
                    // single-field, current-element, and field-vs-field
                    // predicates.  StrVec prefix/suffix/substring fusion stays
                    // (typed-lane SIMD path).
                    // FilterStrVec* / MapStrVec* / MapNumVec* fusion deleted
                    // in Tier 3 — composed substrate runs base CallMethod chain.
                }
                _ => {}
            }
            out2.push(op);
        }
        // Third pass: fold `FilterField* + count()` into FilterField*Count,
        // and collapse chains of `MapField(k1) + MapFlatten(trivial k2) + ...`
        // into a single `FlatMapChain([k1,k2,...])`.
        let mut out3: Vec<Opcode> = Vec::with_capacity(out2.len());
        for op in out2 {
            // FilterFieldEqLit/CmpLit + count() fusion migrated to
            // pipeline.rs count reducer. FilterFieldCmpField + count()
            // fusion deleted in Tier 3 — base Stage::Filter + count reducer.
            // FilterField* + MapField(k) fusion migrated to pipeline.rs
            // Sink::NumFilterMap (and the columnar fast path) — covered
            // for top-level Root-prefix queries.  Sub-program path falls
            // back to the unfused FilterFieldEqLit / FilterFieldCmpLit
            // followed by MapField sequence.
            // MapFlatten-based FlatMapChain fusion deleted with MapFlatten.
            out3.push(op);
        }
        out3
    }

    /// List-comp specialisation deleted in Tier 3 — it emitted the
    /// now-removed Opcode::MapField.  Base Opcode::ListComp handler
    /// covers the same shapes; pipeline IR can hoist the inner
    /// Filter+Map via Stage composition.
    fn pass_list_comp_specialise(ops: Vec<Opcode>) -> Vec<Opcode> {
        ops
    }

    // sort_lam_param helper removed alongside ArgExtreme opcode.

    /// Replace expensive ops with cheaper equivalents:
    ///   sort() + first()    → min()
    ///   sort() + last()     → max()
    ///   sort() + [0]        → min()
    ///   sort() + [-1]       → max()
    ///   reverse() + first() → last()
    ///   reverse() + last()  → first()
    ///   sort_by(k) + first()/last() → ArgExtreme (O(N) scan)
    fn pass_strength_reduce(ops: Vec<Opcode>) -> Vec<Opcode> {
        let mut out: Vec<Opcode> = Vec::with_capacity(ops.len());
        for op in ops {
            // Pattern: [..., prev_method_call, current_op]
            if let Some(Opcode::CallMethod(prev)) = out.last().cloned() {
                let replaced = match (prev.method, &op) {
                    // sort() + [0] → min()
                    (BuiltinMethod::Sort, Opcode::GetIndex(0)) if prev.sub_progs.is_empty() => {
                        Some(make_noarg_call(BuiltinMethod::Min, "min"))
                    }
                    // sort() + [-1] → max()
                    (BuiltinMethod::Sort, Opcode::GetIndex(-1)) if prev.sub_progs.is_empty() => {
                        Some(make_noarg_call(BuiltinMethod::Max, "max"))
                    }
                    // sort() + first() → min()
                    (BuiltinMethod::Sort, Opcode::CallMethod(next))
                        if prev.sub_progs.is_empty() && next.method == BuiltinMethod::First =>
                    {
                        Some(make_noarg_call(BuiltinMethod::Min, "min"))
                    }
                    // sort() + last() → max()
                    (BuiltinMethod::Sort, Opcode::CallMethod(next))
                        if prev.sub_progs.is_empty() && next.method == BuiltinMethod::Last =>
                    {
                        Some(make_noarg_call(BuiltinMethod::Max, "max"))
                    }
                    // sort_by(k) + first() / + last() fusions migrated to
                    // pipeline.rs Sink::MinBy(k) / Sink::MaxBy(k).
                    // Sub-program path: unfused sort_by(k) followed by
                    // First / Last opcodes.
                    // reverse() + first() → last()
                    (BuiltinMethod::Reverse, Opcode::CallMethod(next))
                        if next.method == BuiltinMethod::First =>
                    {
                        Some(make_noarg_call(BuiltinMethod::Last, "last"))
                    }
                    // reverse() + last() → first()
                    (BuiltinMethod::Reverse, Opcode::CallMethod(next))
                        if next.method == BuiltinMethod::Last =>
                    {
                        Some(make_noarg_call(BuiltinMethod::First, "first"))
                    }
                    // sort() + [0:n] / take(n) fusion migrated to
                    // pipeline.rs Sink::TopN { n, asc }.
                    // Cardinality-preserving op + len/count → drop the first op.
                    // sort / reverse preserve length by definition; map is
                    // 1:1 so it also preserves length, and `count` only needs
                    // the input array length.
                    (
                        BuiltinMethod::Sort | BuiltinMethod::Reverse | BuiltinMethod::Map,
                        Opcode::CallMethod(next),
                    ) if next.sub_progs.is_empty()
                        && (next.method == BuiltinMethod::Len
                            || next.method == BuiltinMethod::Count) =>
                    {
                        Some(Opcode::CallMethod(Arc::clone(next)))
                    }
                    // Order-independent aggregate after sort/reverse → drop
                    // the reorder.  sum / avg / min / max only inspect the
                    // multiset of elements, not their order.
                    (BuiltinMethod::Sort | BuiltinMethod::Reverse, Opcode::CallMethod(next))
                        if prev.sub_progs.is_empty()
                            && next.sub_progs.is_empty()
                            && matches!(
                                next.method,
                                BuiltinMethod::Sum
                                    | BuiltinMethod::Avg
                                    | BuiltinMethod::Min
                                    | BuiltinMethod::Max
                            ) =>
                    {
                        Some(Opcode::CallMethod(Arc::clone(next)))
                    }
                    // Idempotent: f(f(x)) == f(x).  `sort(k)` is idempotent
                    // only when both calls use the same key, so we restrict
                    // the no-arg case; `unique()` dedup is always idempotent.
                    (BuiltinMethod::Sort, Opcode::CallMethod(next))
                        if prev.sub_progs.is_empty()
                            && next.method == BuiltinMethod::Sort
                            && next.sub_progs.is_empty() =>
                    {
                        Some(Opcode::CallMethod(Arc::clone(next)))
                    }
                    (BuiltinMethod::Unique, Opcode::CallMethod(next))
                        if next.method == BuiltinMethod::Unique =>
                    {
                        Some(Opcode::CallMethod(Arc::clone(next)))
                    }
                    // UniqueCount removed; pipeline rule
                    // Unique ∘ Count → UniqueCount fuses at lower-time.
                    _ => None,
                };
                if let Some(rep) = replaced {
                    out.pop();
                    out.push(rep);
                    continue;
                }
                // Involution: reverse().reverse() → drop both.
                if prev.method == BuiltinMethod::Reverse && prev.sub_progs.is_empty() {
                    if let Opcode::CallMethod(next) = &op {
                        if next.method == BuiltinMethod::Reverse && next.sub_progs.is_empty() {
                            out.pop();
                            continue;
                        }
                    }
                }
            }
            out.push(op);
        }
        out
    }

    /// Fuse runs of `GetField` not consumed by `pass_root_chain` into a
    /// single `FieldChain`.  Applies mid-program where the object on TOS
    /// came from elsewhere (method return, filter, comprehension).  Singletons
    /// are left as-is — fusion only triggers at length ≥ 2.
    fn pass_field_chain(ops: Vec<Opcode>) -> Vec<Opcode> {
        // Both GetField(k) and OptField(k) devolve to `get_field(k)` which
        // returns Null for non-objects, so OptField can be absorbed into a
        // FieldChain: null propagates through the remaining get_field calls.
        fn field_key(op: &Opcode) -> Option<Arc<str>> {
            match op {
                Opcode::GetField(k) | Opcode::OptField(k) => Some(Arc::clone(k)),
                _ => None,
            }
        }
        let mut out = Vec::with_capacity(ops.len());
        let mut it = ops.into_iter().peekable();
        while let Some(op) = it.next() {
            if let Some(k0) = field_key(&op) {
                if it.peek().and_then(field_key).is_some() {
                    let mut chain: Vec<Arc<str>> = vec![k0];
                    while let Some(k) = it.peek().and_then(field_key) {
                        it.next();
                        chain.push(k);
                    }
                    out.push(Opcode::FieldChain(Arc::new(FieldChainData::new(
                        chain.into(),
                    ))));
                    continue;
                }
                out.push(op);
            } else {
                out.push(op);
            }
        }
        out
    }

    /// Fuse `PushRoot + GetField(k1) + GetField(k2) ...` → `RootChain([k1,k2,...])`.
    fn pass_root_chain(ops: Vec<Opcode>) -> Vec<Opcode> {
        let mut out = Vec::with_capacity(ops.len());
        let mut it = ops.into_iter().peekable();
        while let Some(op) = it.next() {
            if matches!(op, Opcode::PushRoot) {
                let mut chain: Vec<Arc<str>> = Vec::new();
                while let Some(Opcode::GetField(_)) = it.peek() {
                    if let Some(Opcode::GetField(k)) = it.next() {
                        chain.push(k);
                    }
                }
                if chain.is_empty() {
                    out.push(Opcode::PushRoot);
                } else {
                    out.push(Opcode::RootChain(chain.into()));
                }
            } else {
                out.push(op);
            }
        }
        out
    }

    /// Eliminate redundant adjacent method calls:
    ///   reverse() + reverse()         → identity (both dropped)
    ///   unique() + unique()           → unique()
    ///   compact() + compact()         → compact()
    ///   sort() + sort(k)              → sort(k)      (later sort wins on same-array)
    ///   sort(k) + sort(k)             → sort(k)
    ///   Quantifier + Quantifier       → second only  (first wrap is scalar anyway)
    fn pass_redundant_ops(ops: Vec<Opcode>) -> Vec<Opcode> {
        let mut out: Vec<Opcode> = Vec::with_capacity(ops.len());
        for op in ops {
            match (&op, out.last()) {
                // reverse + reverse: drop both
                (Opcode::CallMethod(b), Some(Opcode::CallMethod(a)))
                    if a.method == BuiltinMethod::Reverse && b.method == BuiltinMethod::Reverse =>
                {
                    out.pop();
                    continue;
                }
                // idempotent method pairs: keep second only (drop first)
                (Opcode::CallMethod(b), Some(Opcode::CallMethod(a)))
                    if a.method == b.method
                        && matches!(a.method, BuiltinMethod::Unique | BuiltinMethod::Compact)
                        && a.sub_progs.is_empty()
                        && b.sub_progs.is_empty() =>
                {
                    out.pop();
                    out.push(op);
                    continue;
                }
                // sort + sort(_): later sort wins, drop the first
                (Opcode::CallMethod(b), Some(Opcode::CallMethod(a)))
                    if a.method == BuiltinMethod::Sort && b.method == BuiltinMethod::Sort =>
                {
                    out.pop();
                    out.push(op);
                    continue;
                }
                // Quantifier + Quantifier: second wins (first unwraps scalar,
                // second is no-op on scalar — but keeping second preserves error
                // semantics of `!`). Drop first.
                (Opcode::Quantifier(_), Some(Opcode::Quantifier(_))) => {
                    out.pop();
                    out.push(op);
                    continue;
                }
                // reverse + last → first (strength reduction fallback after sort)
                // already handled in pass_strength_reduce
                // Not + Not: double negation → drop both
                (Opcode::Not, Some(Opcode::Not)) => {
                    out.pop();
                    continue;
                }
                // Neg + Neg: --x → x
                (Opcode::Neg, Some(Opcode::Neg)) => {
                    out.pop();
                    continue;
                }
                _ => {}
            }
            out.push(op);
        }
        out
    }

    /// Constant-fold adjacent integer arithmetic + bool short-circuits.
    fn pass_const_fold(ops: Vec<Opcode>) -> Vec<Opcode> {
        let mut out = Vec::with_capacity(ops.len());
        let mut i = 0;
        while i < ops.len() {
            // 2-op bool short-circuit folds:
            //   PushBool(false) + AndOp(_)  → PushBool(false)
            //   PushBool(true)  + OrOp(_)   → PushBool(true)
            if i + 1 < ops.len() {
                let folded = match (&ops[i], &ops[i + 1]) {
                    (Opcode::PushBool(false), Opcode::AndOp(_)) => Some(Opcode::PushBool(false)),
                    (Opcode::PushBool(true), Opcode::OrOp(_)) => Some(Opcode::PushBool(true)),
                    _ => None,
                };
                if let Some(folded) = folded {
                    out.push(folded);
                    i += 2;
                    continue;
                }
            }
            // 2-op unary folds
            if i + 1 < ops.len() {
                let folded = match (&ops[i], &ops[i + 1]) {
                    (Opcode::PushBool(b), Opcode::Not) => Some(Opcode::PushBool(!b)),
                    (Opcode::PushInt(n), Opcode::Neg) => Some(Opcode::PushInt(-n)),
                    (Opcode::PushFloat(f), Opcode::Neg) => Some(Opcode::PushFloat(-f)),
                    _ => None,
                };
                if let Some(folded) = folded {
                    out.push(folded);
                    i += 2;
                    continue;
                }
            }
            // 3-op arithmetic + comparison folds
            if i + 2 < ops.len() {
                let folded = match (&ops[i], &ops[i + 1], &ops[i + 2]) {
                    (Opcode::PushInt(a), Opcode::PushInt(b), Opcode::Add) => {
                        Some(Opcode::PushInt(a + b))
                    }
                    (Opcode::PushInt(a), Opcode::PushInt(b), Opcode::Sub) => {
                        Some(Opcode::PushInt(a - b))
                    }
                    (Opcode::PushInt(a), Opcode::PushInt(b), Opcode::Mul) => {
                        Some(Opcode::PushInt(a * b))
                    }
                    (Opcode::PushInt(a), Opcode::PushInt(b), Opcode::Mod) if *b != 0 => {
                        Some(Opcode::PushInt(a % b))
                    }
                    (Opcode::PushInt(a), Opcode::PushInt(b), Opcode::Div) if *b != 0 => {
                        Some(Opcode::PushFloat(*a as f64 / *b as f64))
                    }
                    (Opcode::PushFloat(a), Opcode::PushFloat(b), Opcode::Add) => {
                        Some(Opcode::PushFloat(a + b))
                    }
                    (Opcode::PushFloat(a), Opcode::PushFloat(b), Opcode::Sub) => {
                        Some(Opcode::PushFloat(a - b))
                    }
                    (Opcode::PushFloat(a), Opcode::PushFloat(b), Opcode::Mul) => {
                        Some(Opcode::PushFloat(a * b))
                    }
                    (Opcode::PushFloat(a), Opcode::PushFloat(b), Opcode::Div) if *b != 0.0 => {
                        Some(Opcode::PushFloat(a / b))
                    }
                    // Mixed int/float arithmetic
                    (Opcode::PushInt(a), Opcode::PushFloat(b), Opcode::Add) => {
                        Some(Opcode::PushFloat(*a as f64 + b))
                    }
                    (Opcode::PushFloat(a), Opcode::PushInt(b), Opcode::Add) => {
                        Some(Opcode::PushFloat(a + *b as f64))
                    }
                    (Opcode::PushInt(a), Opcode::PushFloat(b), Opcode::Sub) => {
                        Some(Opcode::PushFloat(*a as f64 - b))
                    }
                    (Opcode::PushFloat(a), Opcode::PushInt(b), Opcode::Sub) => {
                        Some(Opcode::PushFloat(a - *b as f64))
                    }
                    (Opcode::PushInt(a), Opcode::PushFloat(b), Opcode::Mul) => {
                        Some(Opcode::PushFloat(*a as f64 * b))
                    }
                    (Opcode::PushFloat(a), Opcode::PushInt(b), Opcode::Mul) => {
                        Some(Opcode::PushFloat(a * *b as f64))
                    }
                    (Opcode::PushInt(a), Opcode::PushFloat(b), Opcode::Div) if *b != 0.0 => {
                        Some(Opcode::PushFloat(*a as f64 / b))
                    }
                    (Opcode::PushFloat(a), Opcode::PushInt(b), Opcode::Div) if *b != 0 => {
                        Some(Opcode::PushFloat(a / *b as f64))
                    }
                    // Mixed int/float comparisons
                    (Opcode::PushInt(a), Opcode::PushFloat(b), Opcode::Lt) => {
                        Some(Opcode::PushBool((*a as f64) < *b))
                    }
                    (Opcode::PushFloat(a), Opcode::PushInt(b), Opcode::Lt) => {
                        Some(Opcode::PushBool(*a < (*b as f64)))
                    }
                    (Opcode::PushInt(a), Opcode::PushFloat(b), Opcode::Gt) => {
                        Some(Opcode::PushBool((*a as f64) > *b))
                    }
                    (Opcode::PushFloat(a), Opcode::PushInt(b), Opcode::Gt) => {
                        Some(Opcode::PushBool(*a > (*b as f64)))
                    }
                    (Opcode::PushInt(a), Opcode::PushFloat(b), Opcode::Lte) => {
                        Some(Opcode::PushBool((*a as f64) <= *b))
                    }
                    (Opcode::PushFloat(a), Opcode::PushInt(b), Opcode::Lte) => {
                        Some(Opcode::PushBool(*a <= (*b as f64)))
                    }
                    (Opcode::PushInt(a), Opcode::PushFloat(b), Opcode::Gte) => {
                        Some(Opcode::PushBool((*a as f64) >= *b))
                    }
                    (Opcode::PushFloat(a), Opcode::PushInt(b), Opcode::Gte) => {
                        Some(Opcode::PushBool(*a >= (*b as f64)))
                    }
                    // Float comparisons (parity with int)
                    (Opcode::PushFloat(a), Opcode::PushFloat(b), Opcode::Lt) => {
                        Some(Opcode::PushBool(a < b))
                    }
                    (Opcode::PushFloat(a), Opcode::PushFloat(b), Opcode::Lte) => {
                        Some(Opcode::PushBool(a <= b))
                    }
                    (Opcode::PushFloat(a), Opcode::PushFloat(b), Opcode::Gt) => {
                        Some(Opcode::PushBool(a > b))
                    }
                    (Opcode::PushFloat(a), Opcode::PushFloat(b), Opcode::Gte) => {
                        Some(Opcode::PushBool(a >= b))
                    }
                    (Opcode::PushInt(a), Opcode::PushInt(b), Opcode::Eq) => {
                        Some(Opcode::PushBool(a == b))
                    }
                    (Opcode::PushInt(a), Opcode::PushInt(b), Opcode::Neq) => {
                        Some(Opcode::PushBool(a != b))
                    }
                    (Opcode::PushInt(a), Opcode::PushInt(b), Opcode::Lt) => {
                        Some(Opcode::PushBool(a < b))
                    }
                    (Opcode::PushInt(a), Opcode::PushInt(b), Opcode::Lte) => {
                        Some(Opcode::PushBool(a <= b))
                    }
                    (Opcode::PushInt(a), Opcode::PushInt(b), Opcode::Gt) => {
                        Some(Opcode::PushBool(a > b))
                    }
                    (Opcode::PushInt(a), Opcode::PushInt(b), Opcode::Gte) => {
                        Some(Opcode::PushBool(a >= b))
                    }
                    (Opcode::PushStr(a), Opcode::PushStr(b), Opcode::Eq) => {
                        Some(Opcode::PushBool(a == b))
                    }
                    (Opcode::PushStr(a), Opcode::PushStr(b), Opcode::Neq) => {
                        Some(Opcode::PushBool(a != b))
                    }
                    (Opcode::PushStr(a), Opcode::PushStr(b), Opcode::Lt) => {
                        Some(Opcode::PushBool(a < b))
                    }
                    (Opcode::PushStr(a), Opcode::PushStr(b), Opcode::Lte) => {
                        Some(Opcode::PushBool(a <= b))
                    }
                    (Opcode::PushStr(a), Opcode::PushStr(b), Opcode::Gt) => {
                        Some(Opcode::PushBool(a > b))
                    }
                    (Opcode::PushStr(a), Opcode::PushStr(b), Opcode::Gte) => {
                        Some(Opcode::PushBool(a >= b))
                    }
                    (Opcode::PushStr(a), Opcode::PushStr(b), Opcode::Add) => {
                        Some(Opcode::PushStr(Arc::<str>::from(format!("{}{}", a, b))))
                    }
                    (Opcode::PushBool(a), Opcode::PushBool(b), Opcode::Eq) => {
                        Some(Opcode::PushBool(a == b))
                    }
                    _ => None,
                };
                if let Some(folded) = folded {
                    out.push(folded);
                    i += 3;
                    continue;
                }
            }
            out.push(ops[i].clone());
            i += 1;
        }
        out
    }

    // ── Main emit ─────────────────────────────────────────────────────────────

    fn emit(expr: &Expr, ctx: &VarCtx) -> Vec<Opcode> {
        let mut ops = Vec::new();
        Self::emit_into(expr, ctx, &mut ops);
        ops
    }

    fn emit_into(expr: &Expr, ctx: &VarCtx, ops: &mut Vec<Opcode>) {
        match expr {
            Expr::Null => ops.push(Opcode::PushNull),
            Expr::Bool(b) => ops.push(Opcode::PushBool(*b)),
            Expr::Int(n) => ops.push(Opcode::PushInt(*n)),
            Expr::Float(f) => ops.push(Opcode::PushFloat(*f)),
            Expr::Str(s) => ops.push(Opcode::PushStr(Arc::from(s.as_str()))),
            Expr::Root => ops.push(Opcode::PushRoot),
            Expr::Current => ops.push(Opcode::PushCurrent),

            Expr::FString(parts) => {
                let compiled: Vec<CompiledFSPart> = parts
                    .iter()
                    .map(|p| match p {
                        FStringPart::Lit(s) => CompiledFSPart::Lit(Arc::from(s.as_str())),
                        FStringPart::Interp { expr, fmt } => CompiledFSPart::Interp {
                            prog: Arc::new(Self::compile_sub(expr, ctx)),
                            fmt: fmt.clone(),
                        },
                    })
                    .collect();
                ops.push(Opcode::FString(compiled.into()));
            }

            Expr::Ident(name) => ops.push(Opcode::LoadIdent(Arc::from(name.as_str()))),

            Expr::Chain(base, steps) => {
                Self::emit_into(base, ctx, ops);
                for step in steps {
                    Self::emit_step(step, ctx, ops);
                }
            }

            Expr::UnaryNeg(e) => {
                Self::emit_into(e, ctx, ops);
                ops.push(Opcode::Neg);
            }
            Expr::Not(e) => {
                Self::emit_into(e, ctx, ops);
                ops.push(Opcode::Not);
            }

            Expr::BinOp(l, op, r) => Self::emit_binop(l, *op, r, ctx, ops),

            Expr::Coalesce(lhs, rhs) => {
                Self::emit_into(lhs, ctx, ops);
                let rhs_prog = Arc::new(Self::compile_sub(rhs, ctx));
                ops.push(Opcode::CoalesceOp(rhs_prog));
            }

            Expr::Kind { expr, ty, negate } => {
                Self::emit_into(expr, ctx, ops);
                ops.push(Opcode::KindCheck {
                    ty: *ty,
                    negate: *negate,
                });
            }

            Expr::Object(fields) => {
                let entries: Vec<CompiledObjEntry> = fields
                    .iter()
                    .map(|f| match f {
                        ObjField::Short(name) => CompiledObjEntry::Short {
                            name: Arc::from(name.as_str()),
                            ic: Arc::new(AtomicU64::new(0)),
                        },
                        ObjField::Kv {
                            key,
                            val,
                            optional,
                            cond,
                        } if cond.is_none() && Self::try_kv_path_steps(val).is_some() => {
                            let steps: Vec<KvStep> = Self::try_kv_path_steps(val).unwrap();
                            let n = steps.len();
                            let mut ics_vec: Vec<AtomicU64> = Vec::with_capacity(n);
                            for _ in 0..n {
                                ics_vec.push(AtomicU64::new(0));
                            }
                            CompiledObjEntry::KvPath {
                                key: Arc::from(key.as_str()),
                                steps: steps.into(),
                                optional: *optional,
                                ics: ics_vec.into(),
                            }
                        }
                        ObjField::Kv {
                            key,
                            val,
                            optional,
                            cond,
                        } => CompiledObjEntry::Kv {
                            key: Arc::from(key.as_str()),
                            prog: Arc::new(Self::compile_sub(val, ctx)),
                            optional: *optional,
                            cond: cond.as_ref().map(|c| Arc::new(Self::compile_sub(c, ctx))),
                        },
                        ObjField::Dynamic { key, val } => CompiledObjEntry::Dynamic {
                            key: Arc::new(Self::compile_sub(key, ctx)),
                            val: Arc::new(Self::compile_sub(val, ctx)),
                        },
                        ObjField::Spread(e) => {
                            CompiledObjEntry::Spread(Arc::new(Self::compile_sub(e, ctx)))
                        }
                        ObjField::SpreadDeep(e) => {
                            CompiledObjEntry::SpreadDeep(Arc::new(Self::compile_sub(e, ctx)))
                        }
                    })
                    .collect();
                ops.push(Opcode::MakeObj(entries.into()));
            }

            Expr::Array(elems) => {
                // Each entry: (sub-program, is_spread).  Spreads execute
                // their program normally; the MakeArr handler flattens
                // an iterable result into the output array.
                let progs: Vec<(Arc<Program>, bool)> = elems
                    .iter()
                    .map(|e| match e {
                        ArrayElem::Expr(ex) => (Arc::new(Self::compile_sub(ex, ctx)), false),
                        ArrayElem::Spread(ex) => (Arc::new(Self::compile_sub(ex, ctx)), true),
                    })
                    .collect();
                ops.push(Opcode::MakeArr(progs.into()));
            }

            Expr::Pipeline { base, steps } => {
                Self::emit_pipeline(base, steps, ctx, ops);
            }

            Expr::ListComp {
                expr,
                vars,
                iter,
                cond,
            } => {
                let inner_ctx = ctx.with_vars(vars);
                ops.push(Opcode::ListComp(Arc::new(CompSpec {
                    expr: Arc::new(Self::compile_sub(expr, &inner_ctx)),
                    vars: vars
                        .iter()
                        .map(|v| Arc::from(v.as_str()))
                        .collect::<Vec<_>>()
                        .into(),
                    iter: Arc::new(Self::compile_sub(iter, ctx)),
                    cond: cond
                        .as_ref()
                        .map(|c| Arc::new(Self::compile_sub(c, &inner_ctx))),
                })));
            }

            Expr::DictComp {
                key,
                val,
                vars,
                iter,
                cond,
            } => {
                let inner_ctx = ctx.with_vars(vars);
                ops.push(Opcode::DictComp(Arc::new(DictCompSpec {
                    key: Arc::new(Self::compile_sub(key, &inner_ctx)),
                    val: Arc::new(Self::compile_sub(val, &inner_ctx)),
                    vars: vars
                        .iter()
                        .map(|v| Arc::from(v.as_str()))
                        .collect::<Vec<_>>()
                        .into(),
                    iter: Arc::new(Self::compile_sub(iter, ctx)),
                    cond: cond
                        .as_ref()
                        .map(|c| Arc::new(Self::compile_sub(c, &inner_ctx))),
                })));
            }

            Expr::SetComp {
                expr,
                vars,
                iter,
                cond,
            }
            | Expr::GenComp {
                expr,
                vars,
                iter,
                cond,
            } => {
                let inner_ctx = ctx.with_vars(vars);
                ops.push(Opcode::SetComp(Arc::new(CompSpec {
                    expr: Arc::new(Self::compile_sub(expr, &inner_ctx)),
                    vars: vars
                        .iter()
                        .map(|v| Arc::from(v.as_str()))
                        .collect::<Vec<_>>()
                        .into(),
                    iter: Arc::new(Self::compile_sub(iter, ctx)),
                    cond: cond
                        .as_ref()
                        .map(|c| Arc::new(Self::compile_sub(c, &inner_ctx))),
                })));
            }

            Expr::Lambda { .. } => {
                // Lambdas as standalone values are errors; they only appear as args
                ops.push(Opcode::PushNull);
            }

            Expr::Let { name, init, body } => {
                // Dead-let: if body never references `name` and init is pure,
                // drop the binding entirely and emit body only.
                if super::analysis::expr_is_pure(init)
                    && !super::analysis::expr_uses_ident(body, name)
                {
                    Self::emit_into(body, ctx, ops);
                } else {
                    Self::emit_into(init, ctx, ops);
                    let body_ctx = ctx.with_var(name);
                    let body_prog = Arc::new(Self::compile_sub(body, &body_ctx));
                    ops.push(Opcode::LetExpr {
                        name: Arc::from(name.as_str()),
                        body: body_prog,
                    });
                }
            }

            Expr::IfElse { cond, then_, else_ } => {
                // Compile-time fold when cond is a literal bool.
                match cond.as_ref() {
                    Expr::Bool(true) => {
                        Self::emit_into(then_, ctx, ops);
                    }
                    Expr::Bool(false) => {
                        Self::emit_into(else_, ctx, ops);
                    }
                    _ => {
                        Self::emit_into(cond, ctx, ops);
                        let then_prog = Arc::new(Self::compile_sub(then_, ctx));
                        let else_prog = Arc::new(Self::compile_sub(else_, ctx));
                        ops.push(Opcode::IfElse {
                            then_: then_prog,
                            else_: else_prog,
                        });
                    }
                }
            }

            Expr::Try { body, default } => {
                // Compile-time fold: if body is a literal non-null constant,
                // emit only the body. If body is null literal, emit only the
                // default. Avoids TryExpr opcode overhead for trivial cases.
                match body.as_ref() {
                    Expr::Null => {
                        Self::emit_into(default, ctx, ops);
                    }
                    Expr::Bool(_) | Expr::Int(_) | Expr::Float(_) | Expr::Str(_) => {
                        Self::emit_into(body, ctx, ops);
                    }
                    _ => {
                        let body_prog = Arc::new(Self::compile_sub(body, ctx));
                        let default_prog = Arc::new(Self::compile_sub(default, ctx));
                        ops.push(Opcode::TryExpr {
                            body: body_prog,
                            default: default_prog,
                        });
                    }
                }
            }

            Expr::GlobalCall { name, args } => {
                // Match runtime `eval_global` semantics: special-case
                // `coalesce/chain/join/zip/zip_longest/product/range` compile
                // through their existing handlers via root receiver; for all
                // OTHER global names (`to_string(x)`, `to_bool(x)`, etc.),
                // first arg is the receiver — equivalent to method-call form.
                // No-args case: current is receiver.
                let is_special = matches!(
                    name.as_str(),
                    "coalesce" | "chain" | "join" | "zip" | "zip_longest" | "product" | "range"
                );
                if !is_special && !args.is_empty() {
                    // <first_arg> CallMethod(name, [args[1..]])
                    let first = match &args[0] {
                        Arg::Pos(e) | Arg::Named(_, e) => e.clone(),
                    };
                    Self::emit_into(&first, ctx, ops);
                    let rest_args: Vec<Arg> = args.iter().skip(1).cloned().collect();
                    let sub_progs: Vec<Arc<Program>> = rest_args
                        .iter()
                        .map(|a| match a {
                            Arg::Pos(e) | Arg::Named(_, e) => Arc::new(Self::compile_sub(e, ctx)),
                        })
                        .collect();
                    let call = Arc::new(CompiledCall {
                        method: BuiltinMethod::from_name(name.as_str()),
                        name: Arc::from(name.as_str()),
                        sub_progs: sub_progs.into(),
                        orig_args: rest_args.into(),
                        demand_max_keep: None,
                    });
                    ops.push(Opcode::CallMethod(call));
                } else {
                    // Special globals (or no-arg) — keep root-receiver path.
                    let sub_progs: Vec<Arc<Program>> = args
                        .iter()
                        .map(|a| match a {
                            Arg::Pos(e) | Arg::Named(_, e) => Arc::new(Self::compile_sub(e, ctx)),
                        })
                        .collect();
                    let call = Arc::new(CompiledCall {
                        method: BuiltinMethod::Unknown,
                        name: Arc::from(name.as_str()),
                        sub_progs: sub_progs.into(),
                        orig_args: args.iter().cloned().collect::<Vec<_>>().into(),
                        demand_max_keep: None,
                    });
                    ops.push(Opcode::PushRoot);
                    ops.push(Opcode::CallMethod(call));
                }
            }

            Expr::Cast { expr, ty } => {
                Self::emit_into(expr, ctx, ops);
                ops.push(Opcode::CastOp(*ty));
            }

            Expr::Patch {
                root,
                ops: patch_ops,
            } => {
                // Patch block — pre-compiled to CompiledPatch.  Every
                // sub-expression (root, op.val, op.cond, dyn-index,
                // wildcard-filter pred) lives as Arc<Program>; runtime
                // patch executor uses self.exec.
                let compiled = Self::compile_patch(root, patch_ops, ctx);
                ops.push(Opcode::PatchEval(Arc::new(compiled)));
            }

            Expr::DeleteMark => {
                // DELETE outside a patch-field value is a static error
                // raised at runtime via `Opcode::DeleteMarkErr` sentinel.
                ops.push(Opcode::DeleteMarkErr);
            }
        }
    }

    fn emit_step(step: &Step, ctx: &VarCtx, ops: &mut Vec<Opcode>) {
        match step {
            Step::Field(name) => ops.push(Opcode::GetField(Arc::from(name.as_str()))),
            Step::OptField(name) => ops.push(Opcode::OptField(Arc::from(name.as_str()))),
            Step::Descendant(n) => ops.push(Opcode::Descendant(Arc::from(n.as_str()))),
            Step::DescendAll => ops.push(Opcode::DescendAll),
            Step::Index(i) => ops.push(Opcode::GetIndex(*i)),
            Step::DynIndex(e) => ops.push(Opcode::DynIndex(Arc::new(Self::compile_sub(e, ctx)))),
            Step::Slice(a, b) => ops.push(Opcode::GetSlice(*a, *b)),
            Step::Method(name, method_args) => {
                let call = Self::compile_call(name, method_args, ctx);
                ops.push(Opcode::CallMethod(Arc::new(call)));
            }
            Step::OptMethod(name, method_args) => {
                let call = Self::compile_call(name, method_args, ctx);
                ops.push(Opcode::CallOptMethod(Arc::new(call)));
            }
            Step::InlineFilter(pred) => {
                ops.push(Opcode::InlineFilter(Arc::new(Self::compile_sub(pred, ctx))));
            }
            Step::Quantifier(k) => ops.push(Opcode::Quantifier(*k)),
        }
    }

    fn compile_call(name: &str, args: &[Arg], ctx: &VarCtx) -> CompiledCall {
        let method = BuiltinMethod::from_name(name);
        let sub_progs: Vec<Arc<Program>> = args
            .iter()
            .map(|a| match a {
                Arg::Pos(e) | Arg::Named(_, e) => Arc::new(Self::compile_lambda_or_expr(e, ctx)),
            })
            .collect();
        CompiledCall {
            method,
            name: Arc::from(name),
            sub_progs: sub_progs.into(),
            orig_args: args.iter().cloned().collect::<Vec<_>>().into(),
            demand_max_keep: None,
        }
    }

    /// Compile an argument expression; for lambdas, the lambda param becomes a
    /// known var in the inner context so `Ident(param)` emits `LoadIdent`.
    /// For single-param lambdas the param is bound to the per-iteration
    /// `current` (push_lam threads both name+current), so substitute the
    /// resulting `LoadIdent(param)` ops with `PushCurrent`.  This lets
    /// trivial_field / trivial_field_chain peepholes recognise the
    /// `b => b.price` shape as equivalent to `map(@.price)`.
    fn compile_lambda_or_expr(expr: &Expr, ctx: &VarCtx) -> Program {
        match expr {
            Expr::Lambda { params, body } => {
                let inner = ctx.with_vars(params);
                let mut p = Self::compile_sub(body, &inner);
                if params.len() == 1 {
                    let name = params[0].as_str();
                    let new_ops: Vec<Opcode> = p
                        .ops
                        .iter()
                        .map(|op| match op {
                            Opcode::LoadIdent(k) if k.as_ref() == name => Opcode::PushCurrent,
                            other => other.clone(),
                        })
                        .collect();
                    p = Program::new(Self::optimize(new_ops), "<lam-body>");
                }
                p
            }
            other => Self::compile_sub(other, ctx),
        }
    }

    fn emit_binop(l: &Expr, op: BinOp, r: &Expr, ctx: &VarCtx, ops: &mut Vec<Opcode>) {
        match op {
            BinOp::And => {
                Self::emit_into(l, ctx, ops);
                let rhs_prog = Arc::new(Self::compile_sub(r, ctx));
                ops.push(Opcode::AndOp(rhs_prog));
            }
            BinOp::Or => {
                Self::emit_into(l, ctx, ops);
                let rhs_prog = Arc::new(Self::compile_sub(r, ctx));
                ops.push(Opcode::OrOp(rhs_prog));
            }
            BinOp::Add => {
                Self::emit_into(l, ctx, ops);
                Self::emit_into(r, ctx, ops);
                ops.push(Opcode::Add);
            }
            BinOp::Sub => {
                Self::emit_into(l, ctx, ops);
                Self::emit_into(r, ctx, ops);
                ops.push(Opcode::Sub);
            }
            BinOp::Mul => {
                Self::emit_into(l, ctx, ops);
                Self::emit_into(r, ctx, ops);
                ops.push(Opcode::Mul);
            }
            BinOp::Div => {
                Self::emit_into(l, ctx, ops);
                Self::emit_into(r, ctx, ops);
                ops.push(Opcode::Div);
            }
            BinOp::Mod => {
                Self::emit_into(l, ctx, ops);
                Self::emit_into(r, ctx, ops);
                ops.push(Opcode::Mod);
            }
            BinOp::Eq => {
                Self::emit_into(l, ctx, ops);
                Self::emit_into(r, ctx, ops);
                ops.push(Opcode::Eq);
            }
            BinOp::Neq => {
                Self::emit_into(l, ctx, ops);
                Self::emit_into(r, ctx, ops);
                ops.push(Opcode::Neq);
            }
            BinOp::Lt => {
                Self::emit_into(l, ctx, ops);
                Self::emit_into(r, ctx, ops);
                ops.push(Opcode::Lt);
            }
            BinOp::Lte => {
                Self::emit_into(l, ctx, ops);
                Self::emit_into(r, ctx, ops);
                ops.push(Opcode::Lte);
            }
            BinOp::Gt => {
                Self::emit_into(l, ctx, ops);
                Self::emit_into(r, ctx, ops);
                ops.push(Opcode::Gt);
            }
            BinOp::Gte => {
                Self::emit_into(l, ctx, ops);
                Self::emit_into(r, ctx, ops);
                ops.push(Opcode::Gte);
            }
            BinOp::Fuzzy => {
                Self::emit_into(l, ctx, ops);
                Self::emit_into(r, ctx, ops);
                ops.push(Opcode::Fuzzy);
            }
        }
    }

    fn emit_pipeline(base: &Expr, steps: &[PipeStep], ctx: &VarCtx, ops: &mut Vec<Opcode>) {
        // Compile base + each step into a fused PipelineRun opcode.
        // Runtime threads the value through a local mutable Env so
        // forward steps see env.current = pipe_value and bind steps
        // store into env vars; downstream steps then resolve the
        // bound names correctly.
        let base_prog = Arc::new(Self::compile_sub(base, ctx));
        let mut cur_ctx = ctx.clone();
        let mut compiled_steps: Vec<CompiledPipeStep> = Vec::with_capacity(steps.len());
        for step in steps {
            match step {
                PipeStep::Forward(rhs) => {
                    // emit_pipe_forward handles the `Ident(method)` and
                    // `Chain(method-base, steps)` shapes specially so a
                    // bare `| len` resolves as a method call on the
                    // pipe value rather than a variable load.  Reuse
                    // that logic by emitting into a fresh Vec then
                    // wrapping as a sub-program.
                    let mut sub_ops: Vec<Opcode> = Vec::new();
                    Self::emit_pipe_forward(rhs, &cur_ctx, &mut sub_ops);
                    // Strip a leading SetCurrent — the PipelineRun
                    // handler already sets env.current before invoking
                    // the sub-program, so the opcode is redundant
                    // here.  All other shapes have no SetCurrent and
                    // need exactly the emitted ops.
                    if let Some(Opcode::SetCurrent) = sub_ops.first() {
                        sub_ops.remove(0);
                    }
                    let prog = Program::new(Self::optimize(sub_ops), "<pipe-fwd>");
                    compiled_steps.push(CompiledPipeStep::Forward(Arc::new(prog)));
                }
                PipeStep::Bind(target) => match target {
                    BindTarget::Name(name) => {
                        compiled_steps.push(CompiledPipeStep::BindName(Arc::from(name.as_str())));
                        cur_ctx = cur_ctx.with_var(name);
                    }
                    BindTarget::Obj { fields, rest } => {
                        let spec = BindObjSpec {
                            fields: fields
                                .iter()
                                .map(|f| Arc::from(f.as_str()))
                                .collect::<Vec<_>>()
                                .into(),
                            rest: rest.as_ref().map(|r| Arc::from(r.as_str())),
                        };
                        compiled_steps.push(CompiledPipeStep::BindObj(Arc::new(spec)));
                        for f in fields {
                            cur_ctx = cur_ctx.with_var(f);
                        }
                        if let Some(r) = rest {
                            cur_ctx = cur_ctx.with_var(r);
                        }
                    }
                    BindTarget::Arr(names) => {
                        let ns: Vec<Arc<str>> =
                            names.iter().map(|n| Arc::from(n.as_str())).collect();
                        compiled_steps.push(CompiledPipeStep::BindArr(ns.into()));
                        for n in names {
                            cur_ctx = cur_ctx.with_var(n);
                        }
                    }
                },
            }
        }
        ops.push(Opcode::PipelineRun {
            base: base_prog,
            steps: compiled_steps.into(),
        });
    }

    fn emit_pipe_forward(rhs: &Expr, ctx: &VarCtx, ops: &mut Vec<Opcode>) {
        match rhs {
            Expr::Ident(name) if !ctx.has(name) => {
                // No-arg method call on the pipe value (env.current).
                // Push current as the receiver so CallMethod's pop yields it.
                let call = CompiledCall {
                    method: BuiltinMethod::from_name(name),
                    name: Arc::from(name.as_str()),
                    sub_progs: Arc::from(&[] as &[Arc<Program>]),
                    orig_args: Arc::from(&[] as &[Arg]),
                    demand_max_keep: None,
                };
                ops.push(Opcode::PushCurrent);
                ops.push(Opcode::CallMethod(Arc::new(call)));
            }
            Expr::Chain(base, steps) if !steps.is_empty() => {
                if let Expr::Ident(name) = base.as_ref() {
                    if !ctx.has(name) {
                        // method(args...) — base is method, steps are chained
                        let call = CompiledCall {
                            method: BuiltinMethod::from_name(name),
                            name: Arc::from(name.as_str()),
                            sub_progs: Arc::from(&[] as &[Arc<Program>]),
                            orig_args: Arc::from(&[] as &[Arg]),
                            demand_max_keep: None,
                        };
                        ops.push(Opcode::PushCurrent);
                        ops.push(Opcode::CallMethod(Arc::new(call)));
                        for step in steps {
                            Self::emit_step(step, ctx, ops);
                        }
                        return;
                    }
                }
                ops.push(Opcode::SetCurrent);
                Self::emit_into(rhs, ctx, ops);
            }
            _ => {
                // Arbitrary expression; set current to pipe input, then eval
                ops.push(Opcode::SetCurrent);
                Self::emit_into(rhs, ctx, ops);
            }
        }
    }

    fn compile_sub(expr: &Expr, ctx: &VarCtx) -> Program {
        let ops = Self::optimize(Self::emit(expr, ctx));
        Program::new(ops, "<sub>")
    }

    /// Compile `Expr::Patch` into VM-native `CompiledPatch`.  Every
    /// sub-expression becomes an `Arc<Program>` so the runtime patch
    /// executor stays inside the VM.
    fn compile_patch(
        root: &Expr,
        patch_ops: &[crate::ast::PatchOp],
        ctx: &VarCtx,
    ) -> CompiledPatch {
        let root_prog = Arc::new(Self::compile_sub(root, ctx));
        let mut ops = Vec::with_capacity(patch_ops.len());
        for po in patch_ops {
            let path: Vec<CompiledPathStep> = po
                .path
                .iter()
                .map(|s| match s {
                    crate::ast::PathStep::Field(n) => {
                        CompiledPathStep::Field(Arc::from(n.as_str()))
                    }
                    crate::ast::PathStep::Index(i) => CompiledPathStep::Index(*i),
                    crate::ast::PathStep::DynIndex(e) => {
                        CompiledPathStep::DynIndex(Arc::new(Self::compile_sub(e, ctx)))
                    }
                    crate::ast::PathStep::Wildcard => CompiledPathStep::Wildcard,
                    crate::ast::PathStep::WildcardFilter(p) => {
                        CompiledPathStep::WildcardFilter(Arc::new(Self::compile_sub(p, ctx)))
                    }
                    crate::ast::PathStep::Descendant(n) => {
                        CompiledPathStep::Descendant(Arc::from(n.as_str()))
                    }
                })
                .collect();
            let val = if matches!(&po.val, Expr::DeleteMark) {
                CompiledPatchVal::Delete
            } else {
                CompiledPatchVal::Replace(Arc::new(Self::compile_sub(&po.val, ctx)))
            };
            let cond = po
                .cond
                .as_ref()
                .map(|c| Arc::new(Self::compile_sub(c, ctx)));
            ops.push(CompiledPatchOp { path, val, cond });
        }
        CompiledPatch { root_prog, ops }
    }

    /// Classify an object-value expression as a pure path on `current`:
    /// a chain of `Field(name)` / `Index(i)` steps rooted at `Expr::Current`.
    /// Returns `None` for anything else — the caller falls back to full
    /// sub-program compilation.
    fn try_kv_path_steps(expr: &Expr) -> Option<Vec<KvStep>> {
        use super::ast::Step;
        let (base, steps) = match expr {
            Expr::Chain(b, s) => (&**b, s.as_slice()),
            _ => return None,
        };
        if !matches!(base, Expr::Current) {
            return None;
        }
        if steps.is_empty() {
            return None;
        }
        let mut out = Vec::with_capacity(steps.len());
        for s in steps {
            match s {
                Step::Field(name) => out.push(KvStep::Field(Arc::from(name.as_str()))),
                Step::Index(i) => out.push(KvStep::Index(*i)),
                _ => return None,
            }
        }
        Some(out)
    }

    // (compile_array_spread / compile_sub_spread removed — MakeArr now
    //  carries an `is_spread` flag per entry, no special program shape
    //  needed.)
    #[allow(dead_code)]
    fn _spread_helpers_removed_marker(_: &Expr, _: &VarCtx) -> Program {
        Program::new(vec![], "<spread>")
    }
}

// ── Path cache ────────────────────────────────────────────────────────────────
//
// Key: (doc_hash, json_pointer) → Val
//
// Doc-scoped (no program_id): any program resolving the same path on the same
// document gets a hit.  Intermediate nodes are cached so a prefix of a longer
// path can be reused without re-traversal.

struct PathCache {
    /// doc_hash → (pointer_string → Val)
    docs: HashMap<u64, HashMap<Arc<str>, Val>>,
    /// FIFO eviction order
    order: VecDeque<(u64, Arc<str>)>,
    capacity: usize,
}

impl PathCache {
    fn new(cap: usize) -> Self {
        Self {
            docs: HashMap::new(),
            order: VecDeque::with_capacity(cap),
            capacity: cap,
        }
    }

    /// O(1) immutable lookup — returns cloned Val (Val::clone is O(1)).
    #[inline]
    fn get(&self, doc_hash: u64, ptr: &str) -> Option<Val> {
        self.docs.get(&doc_hash)?.get(ptr).cloned()
    }

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

    fn len(&self) -> usize {
        self.order.len()
    }
}

// ── VM ────────────────────────────────────────────────────────────────────────

/// High-performance v2 virtual machine.
///
/// Maintains:
/// - **Compile cache** — expression string → `Program` (parse + compile once).
/// - **Path cache** — `(doc_hash, json_pointer)` → `Val`; doc-scoped so any
///   program navigating the same path on the same document shares cached nodes.
///   Intermediate nodes are populated as a side-effect of every traversal,
///   enabling prefix reuse without re-traversal.
///
/// One VM per thread; wrap in `Mutex` for shared use.
/// Toggle each optimiser pass independently.  Default enables every
/// pass.  Disabling a pass invalidates the compile cache for the next
/// compilation by changing `hash()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PassConfig {
    pub root_chain: bool,
    pub field_chain: bool,
    pub filter_count: bool,
    pub filter_fusion: bool,
    pub find_quantifier: bool,
    pub strength_reduce: bool,
    pub redundant_ops: bool,
    pub kind_check_fold: bool,
    pub method_const: bool,
    pub const_fold: bool,
    pub nullness: bool,
    pub reorder_and: bool,
    pub dedup_subprogs: bool,
}

impl Default for PassConfig {
    fn default() -> Self {
        Self {
            root_chain: true,
            field_chain: true,
            filter_count: true,
            filter_fusion: true,
            find_quantifier: true,
            strength_reduce: true,
            redundant_ops: true,
            kind_check_fold: true,
            method_const: true,
            const_fold: true,
            nullness: true,
            reorder_and: true,
            dedup_subprogs: true,
        }
    }
}

impl PassConfig {
    /// Disable every pass — emit raw opcodes.
    pub fn none() -> Self {
        Self {
            root_chain: false,
            field_chain: false,
            filter_count: false,
            filter_fusion: false,
            find_quantifier: false,
            strength_reduce: false,
            redundant_ops: false,
            kind_check_fold: false,
            method_const: false,
            const_fold: false,
            nullness: false,
            reorder_and: false,
            dedup_subprogs: false,
        }
    }

    pub fn hash(&self) -> u64 {
        let mut bits: u64 = 0;
        for (i, b) in [
            self.root_chain,
            self.field_chain,
            self.filter_count,
            self.filter_fusion,
            self.find_quantifier,
            self.strength_reduce,
            self.redundant_ops,
            self.kind_check_fold,
            self.method_const,
            self.const_fold,
            self.nullness,
            self.reorder_and,
            self.dedup_subprogs,
        ]
        .iter()
        .enumerate()
        {
            if *b {
                bits |= 1u64 << i;
            }
        }
        bits
    }
}

pub struct VM {
    /// Cache key = (pass_config_hash, expr_string).  Changing `config`
    /// invalidates prior entries automatically via key divergence.
    compile_cache: HashMap<(u64, String), Arc<Program>>,
    /// LRU ordering for `compile_cache`; front = least recently used.
    /// Entries are moved to back on hit; oldest evicted when over cap.
    compile_lru: std::collections::VecDeque<(u64, String)>,
    compile_cap: usize,
    path_cache: PathCache,
    /// Per-exec RootChain resolution cache.  Key = raw address of the
    /// `chain` Arc slice; value = resolved Val.  Cleared on every top-level
    /// `execute()` call so stale entries never outlive the doc they
    /// reference.  Avoids rebuilding the `/a/b/c` pointer string and
    /// consulting `path_cache` when the same RootChain opcode fires
    /// repeatedly inside a loop.
    root_chain_cache: HashMap<usize, Val>,
    /// Hash of the document currently being executed — set once by `execute()`,
    /// reused by all recursive `exec()` calls within the same top-level call.
    doc_hash: u64,
    /// Cache of root Arc pointer → structural hash.  Lets repeated calls
    /// against the same cached root (e.g. via `Jetro::collect`) skip the
    /// O(doc) structural walk entirely.  Keyed on the inner Arc ptr of
    /// the root `Val::Obj`/`Val::Arr`.
    root_hash_cache: Option<(usize, u64)>,
    /// Optimiser pass toggles.  Default: all on.
    config: PassConfig,
}

// AutoIndexCache + auto_index_key deleted in Tier 3 — they were only used
// by the now-removed FilterFieldEqLit opcode.  Pipeline IR Stage::Filter
// runs predicate without per-VM auto-index state.

impl Default for VM {
    fn default() -> Self {
        Self::new()
    }
}

impl VM {
    pub fn new() -> Self {
        Self::with_capacity(512, 4096)
    }

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

    /// Replace the pass configuration.  The compile cache is not purged,
    /// but future lookups key off the new config hash so old entries
    /// are effectively invalidated for the new regime.
    pub fn set_pass_config(&mut self, config: PassConfig) {
        self.config = config;
    }

    pub fn pass_config(&self) -> PassConfig {
        self.config
    }

    // ── Public entry-points ───────────────────────────────────────────────────

    /// Parse, compile (cached), and execute `expr` against `doc`.
    pub fn run_str(
        &mut self,
        expr: &str,
        doc: &serde_json::Value,
    ) -> Result<serde_json::Value, EvalError> {
        let prog = self.get_or_compile(expr)?;
        self.execute(&prog, doc)
    }

    /// Execute a pre-compiled `Program` against `doc`.
    pub fn execute(
        &mut self,
        program: &Program,
        doc: &serde_json::Value,
    ) -> Result<serde_json::Value, EvalError> {
        let root = Val::from(doc);
        self.doc_hash = self.compute_or_cache_root_hash(&root);
        // Per-exec RootChain cache: entries key off raw Arc addresses that
        // must not outlive this document.  Clear before every run.
        self.root_chain_cache.clear();
        let env = self.make_env(root);
        let result = self.exec(program, &env)?;
        Ok(result.into())
    }

    /// Reuse the cached structural hash if `root`'s inner Arc pointer
    /// matches a prior call; otherwise walk the tree once and cache.
    /// Primitive roots bypass the cache (hashing is already O(1)).
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

    /// Execute against a pre-built `Val` root without raw bytes.  Skips the
    /// `Val::from` conversion only — path cache and doc hash still behave
    /// as in `execute()`.
    pub fn execute_val(
        &mut self,
        program: &Program,
        root: Val,
    ) -> Result<serde_json::Value, EvalError> {
        Ok(self.execute_val_raw(program, root)?.into())
    }

    /// Execute against a pre-built `Val` root and return the raw `Val` —
    /// no `serde_json::Value` materialisation.  Use when the caller will
    /// consume results structurally (further queries, custom walk) and
    /// wants to skip a potentially expensive `Val → Value` conversion.
    pub fn execute_val_raw(&mut self, program: &Program, root: Val) -> Result<Val, EvalError> {
        self.doc_hash = self.compute_or_cache_root_hash(&root);
        self.root_chain_cache.clear();
        let env = self.make_env(root);
        self.exec(program, &env)
    }

    /// Hot-loop variant for pull-based pipelines: skip doc-hash recompute
    /// + root_chain_cache clear + Env construction per call.  Caller
    /// builds the Env once outside the loop and threads it via
    /// `swap_current` per row.  Used by `pipeline::Pipeline::run` and
    /// any per-element evaluator that knows the document hasn't changed.
    #[inline]
    pub fn exec_in_env(&mut self, program: &Program, env: &Env) -> Result<Val, EvalError> {
        self.exec(program, env)
    }

    /// Make an Env for the given root. Public so the pipeline can build one Env per pull loop and rebind
    /// `current` per row instead of per-row Env construction.
    pub fn make_loop_env(&self, root: Val) -> Env {
        self.make_env(root)
    }

    /// Execute a compiled program against a document, first specialising
    // execute_with_schema / execute_with_inferred_schema removed —
    // schema.rs deleted in Tier 3 aggressive sweep (warm-only path).

    /// Get or compile an expression string (compile cache).
    /// Cache key is (pass_config_hash, expr) so that different pass
    /// configurations yield different compiled programs.
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

    /// Move `key` to the back of the LRU queue (most recently used).
    fn touch_lru(&mut self, key: &(u64, String)) {
        if let Some(pos) = self.compile_lru.iter().position(|k| k == key) {
            let k = self.compile_lru.remove(pos).unwrap();
            self.compile_lru.push_back(k);
        }
    }

    /// Insert into compile cache with LRU eviction at `compile_cap`.
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

    /// Cache statistics: `(compile_entries, path_entries)`.
    pub fn cache_stats(&self) -> (usize, usize) {
        (self.compile_cache.len(), self.path_cache.len())
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    fn make_env(&self, root: Val) -> Env {
        Env::new(root)
    }

    // ── Core execution loop ───────────────────────────────────────────────────

    /// Execute `program` in environment `env`, returning the top-of-stack value.
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
                // ── Literals ──────────────────────────────────────────────────
                Opcode::PushNull => stack.push(Val::Null),
                Opcode::PushBool(b) => stack.push(Val::Bool(*b)),
                Opcode::PushInt(n) => stack.push(Val::Int(*n)),
                Opcode::PushFloat(f) => stack.push(Val::Float(*f)),
                Opcode::PushStr(s) => stack.push(Val::Str(s.clone())),

                // ── Context ───────────────────────────────────────────────────
                Opcode::PushRoot => stack.push(env.root.clone()),
                Opcode::PushCurrent => stack.push(env.current.clone()),

                // ── Navigation ────────────────────────────────────────────────
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
                    // (D) When descending from root, track pointer paths and
                    // cache each discovered node for future RootChain lookups.
                    let from_root = match (&v, &env.root) {
                        (Val::Obj(a), Val::Obj(b)) => Arc::ptr_eq(a, b),
                        (Val::Arr(a), Val::Arr(b)) => Arc::ptr_eq(a, b),
                        _ => matches!((&v, &env.root), (Val::Null, Val::Null)),
                    };
                    // Early exit: `$..k.first()` / `$..k!` materialises only
                    // the first self-first DFS hit. Skips pointer-cache
                    // population since single-hit callers do not benefit from
                    // storing siblings.
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

                // ── Peephole fusions ──────────────────────────────────────────
                Opcode::RootChain(chain) => {
                    // Fast path — same RootChain opcode fired earlier in this
                    // execute() call on this same doc.  Arc pointer identity
                    // is stable; cache is cleared per top-level execute().
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
                        // Try resuming from a longer cached prefix before
                        // stepping through get_field.
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
                // MapField / MapFieldChain / MapFieldUnique /
                // MapFieldChainUnique / FlatMapChain handlers deleted in
                // Tier 3.  Pipeline IR runs the base `CallMethod(Map | FlatMap | Unique, [GetField...])`
                // chain via composed `MapField` / `MapFieldChain` /
                // `FlatMapField` / `FlatMapFieldChain` Stages.

                // FilterFieldEqLit / FilterFieldCmpLit / FilterCurrentCmpLit
                // handlers deleted in Tier 3.  Pipeline IR Stage::Filter +
                // BodyKernel::FieldCmpLit / CurrentCmpLit + composed substrate
                // covers these — including auto-index, IntVec/FloatVec/StrVec
                // typed-lane fast paths.
                // FilterStrVec* + MapStrVec* + MapNumVec* handlers deleted in Tier 3.
                // Pipeline IR Stage::Filter / Stage::Map + composed substrate runs
                // base CallMethod chain via the builtin call path on StrVec/IntVec/FloatVec
                // typed-lane receivers.  Loses the autovectoriser-friendly tight
                // loops here in exchange for a single generic execution path.
                // FilterFieldEqLitMapField / FilterFieldCmpLitMapField
                // handlers removed — pipeline.rs Sink::NumFilterMap
                // covers these shapes for top-level Root-prefix queries.
                // Sub-program path executes the unfused
                // FilterFieldEqLit/FilterFieldCmpLit + MapField sequence.
                // FilterFieldCmpField + FilterFieldCmpFieldCount handlers
                // deleted in Tier 3.  Base CallMethod(Filter, [field-vs-field
                // predicate]) chain runs through composed substrate.

                // GroupByField/CountByField/UniqueByField handlers
                // deleted in Tier 3.
                // FilterMapSum / Avg / First / Min / Max handlers
                // removed.  Pipeline.rs Sink::NumFilterMap (sum/avg/min/max)
                // and Sink::FilterFirst cover these shapes for top-level
                // queries; sub-program path executes FilterMap + bare
                // aggregate as two separate ops.
                // FilterLast handler removed — pipeline.rs Sink::FilterLast
                // covers `.filter(p).last()` for top-level queries.
                // EquiJoin handler removed — base CallMethod(equi_join)
                // dispatch covers it via the builtin call path.
                // TopN handler removed — pipeline.rs Sink::TopN covers
                // sort()+take(n) for top-level Root-prefix queries.
                // UniqueCount handler removed — pipeline.rs Sink::UniqueCount
                // covers unique().count() at lower-time.
                // ArgExtreme handler removed — pipeline.rs MinBy/MaxBy
                // covers sort_by(k)+first()/last().
                // MapMap handler removed — pipeline runs two Map stages
                // sequentially, and the bytecode now lowers `map().map()`
                // as two unfused CallMethod(Map) ops.
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

                // ── Operators ─────────────────────────────────────────────────
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

                // ── Short-circuit ops ─────────────────────────────────────────
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
                    // Catch EvalError AND Val::Null; panics propagate.
                    match self.exec(body, env) {
                        Ok(v) if !v.is_null() => stack.push(v),
                        Ok(_) | Err(_) => stack.push(self.exec(default, env)?),
                    }
                }

                // ── Method calls ──────────────────────────────────────────────
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

                // ── Construction ──────────────────────────────────────────────
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

                // ── F-string ──────────────────────────────────────────────────
                Opcode::FString(parts) => {
                    let parts = Arc::clone(parts);
                    let result = self.exec_fstring(&parts, env)?;
                    stack.push(result);
                }

                // ── Kind check ────────────────────────────────────────────────
                Opcode::KindCheck { ty, negate } => {
                    let v = pop!(stack);
                    let m = kind_matches(&v, *ty);
                    stack.push(Val::Bool(if *negate { !m } else { m }));
                }

                // ── Pipeline helpers ──────────────────────────────────────────
                Opcode::SetCurrent => {
                    // This is emitted before an arbitrary pipe-forward expression.
                    // The value flowing through the pipe is now on the stack as TOS.
                    // However we can't mutate env here since env is immutable.
                    // The actual "set current" happens by the caller preparing a new env.
                    // In practice, SetCurrent should not appear in isolation in the
                    // flat opcode stream because pipeline steps that need SetCurrent
                    // are compiled as sub-programs. Skip.
                }
                Opcode::BindVar(name) => {
                    // TOS becomes a named var; TOS remains (pass-through for ->) .
                    // We can't mutate env here. This is handled at the pipeline level.
                    // For now, just keep TOS.
                    let _ = name;
                }
                Opcode::StoreVar(name) => {
                    // Pop and discard (the LetExpr opcode handles binding properly).
                    let _ = name;
                    pop!(stack);
                }
                Opcode::BindObjDestructure(_) | Opcode::BindArrDestructure(_) => {
                    // Pipeline bind destructure — handled at pipeline level.
                }
                Opcode::PipelineRun { base, steps } => {
                    let val = self.exec(base, env)?;
                    let mut local_env = env.clone();
                    let mut cur = val;
                    for step in steps.iter() {
                        match step {
                            CompiledPipeStep::Forward(rhs) => {
                                // Sub-program reads env.current to access
                                // the pipe value.  Method-call shapes
                                // (`| len`, `| upper(2)`) prepend their
                                // own PushCurrent at compile-time so
                                // CallMethod has the receiver on TOS.
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

                // ── Complex recursive ops ─────────────────────────────────────
                Opcode::LetExpr { name, body } => {
                    let init_val = pop!(stack);
                    let body_env = env.with_var(name.as_ref(), init_val);
                    stack.push(self.exec(body, &body_env)?);
                }

                Opcode::ListComp(spec) => {
                    let items = self.exec_iter_vals(&spec.iter, env)?;
                    let mut out = Vec::with_capacity(items.len());
                    // Fast path: single-var `for x in iter` — reuse one
                    // scratch Env via push_lam / pop_lam instead of
                    // cloning per iteration.
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
                    // Single-var fast path: reuse scratch Env via push_lam/pop_lam
                    // instead of cloning per iteration.
                    if spec.vars.len() == 1 {
                        let vname = spec.vars[0].clone();
                        // Hot pattern: `{ f(x): x for x in iter }` with no
                        // condition.  Both key and val depend only on `x`,
                        // which is also `current`, so we can elide the
                        // env rebind and the two `self.exec` calls by
                        // dispatching the common shapes inline.
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

                // ── Path cache lookup ─────────────────────────────────────────
                Opcode::GetPointer(ptr) => {
                    let doc_hash = self.doc_hash;
                    let v = if let Some(cached) = self.path_cache.get(doc_hash, ptr.as_ref()) {
                        cached
                    } else {
                        let v = resolve_pointer(&env.root, ptr.as_ref());
                        self.path_cache.insert(doc_hash, ptr.clone(), v.clone());
                        v
                    };
                    stack.push(v);
                }

                // ── Patch block ───────────────────────────────────────────────
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

    // filter_map_minmax helper removed alongside the FilterMapMin /
    // FilterMapMax opcodes (migrated to pipeline.rs Sink::NumFilterMap).

    // ── Method call dispatch ──────────────────────────────────────────────────

    fn exec_call(&mut self, recv: Val, call: &CompiledCall, env: &Env) -> Result<Val, EvalError> {
        // Global-call opcodes push Root before calling; handle them
        if call.method == BuiltinMethod::Unknown {
            // Global function (coalesce, zip, range, etc.) — known globals
            // route through eval_global; unknown names fall back to method
            // dispatch on the pushed receiver.
            match call.name.as_ref() {
                "coalesce" | "chain" | "join" | "zip" | "zip_longest" | "product" | "range" => {
                    return crate::runtime::eval_global_compiled(self, call, env);
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

        // Lambda methods — VM handles iteration, running sub-programs per item
        if call.method.is_lambda_method() {
            return self.exec_lambda_method(recv, call, env);
        }

        // Typed-numeric aggregate fast-path: bare `.sum()/.min()/.max()/.avg()`
        // on an array.  Skips fallback dispatch + `collect_nums` extra Vec.
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
            // Columnar IntVec — autovec-friendly 4-lane unrolled
            // accumulators get the LLVM vectoriser to emit AVX2 / NEON
            // horizontal-reduce.  Native parity on x86-64-v3 / aarch64.
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
            // Homogeneous-Int `Val::Arr` receivers get routed through
            // columnar reverse/sort — 3x less memory bandwidth than the
            // Vec<Val> path.  Clone cost is identical (O(N) in both cases
            // because the Arr is typically shared); the win is on write.
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
            // Columnar IntVec receiver — reverse/sort in-place on Vec<i64>.
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
            // Bare `.flatten()` (depth=1) — inline depth-1 flatten with exact
            // preallocation.  Skips fallback builtin arg parsing.
            if call.method == BuiltinMethod::Flatten {
                if let Val::Arr(a) = &recv {
                    // Columnar fast-path: all inners Int-only → emit Val::IntVec.
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
                // Columnar receiver itself — already flat; return as-is.
                if matches!(
                    &recv,
                    Val::IntVec(_) | Val::FloatVec(_) | Val::StrVec(_) | Val::StrSliceVec(_)
                ) {
                    return Ok(recv);
                }
            }
            // Scalar `.to_string()` — inline the conversion.  The fallback
            // path allocates a fresh `Val::Str` via `val_to_string`; this
            // does the same work without the dispatch + argslice copy.
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
            // Scalar `.to_json()` — skip the `Val -> serde_json::Value ->
            // String` round-trip for primitives.  Big-ticket users are
            // `.map(@.to_json())` in hot pipelines.
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
                        // JSON-escape — fall through for now; cheap escape
                        // added here to keep the fast-path.  Handles the
                        // common no-escape case with a single scan.
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
                        // Fall through to serde path for escape handling.
                    }
                    _ => {}
                }
            }
        }

        if let Some(v) = self.exec_static_builtin_call(&recv, call, env)? {
            return Ok(v);
        }

        // Value methods — delegate to the builtin dispatcher using the
        // compiled argument programs already stored on the call.
        call_builtin_method_compiled(self, recv, call, env)
    }

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

    fn exec_lambda_method(
        &mut self,
        recv: Val,
        call: &CompiledCall,
        env: &Env,
    ) -> Result<Val, EvalError> {
        let sub = call.sub_progs.first();
        // Hoist the lambda param name out of the per-item loop — otherwise
        // each iteration would re-scan `orig_args` for the Lambda pattern.
        let lam_param: Option<&str> = match call.orig_args.first() {
            Some(Arg::Pos(Expr::Lambda { params, .. })) if !params.is_empty() => {
                Some(params[0].as_str())
            }
            _ => None,
        };
        // Single scratch env per call — reused across every item iteration
        // below via `push_lam` / `pop_lam` instead of a fresh clone per item.
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
                // Pattern specialisation: `lambda x: x % K` where K is a small
                // positive Int.  Skips per-item exec + string-keying.  Uses a
                // dense Vec<Vec<Val>> indexed by (n % K).rem_euclid — collapsed
                // into an IndexMap<Arc<str>, Val> once at the end.
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
                            // All-numeric fast path; error on non-numeric.
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
                // General compiled-bytecode path.
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
                // VM-accelerate the common 2-param form `accumulate(lambda a, x: …)`
                // with no `start:` named arg.  Pattern-specialise for a handful
                // of tight-loop shapes (`a + x`, `a - x`, `a * x`, `max/min`)
                // that the compiled bytecode would otherwise re-dispatch per
                // iteration. Other shapes fall through to a compiled-bytecode
                // VM loop.
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
                // Try pattern specialisation: `LoadIdent(p1), LoadIdent(p2), <BinOp>`.
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
                // Columnar IntVec input → IntVec output (native-parity).
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
                    // Typed i64 tight-loop when every item is Int.  No Val
                    // match, no add_vals dispatch, no per-step Val clone —
                    // native parity.
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
                    // Typed f64 tight-loop when every item is Float.
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
                    // Mixed / non-numeric — inline fold via add_vals/num_op.
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
                // General path: compiled-bytecode VM loop.
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
                // COW: if the receiver's Arc is unique we mutate in place,
                // otherwise deep-clone once and mutate the clone.  Either
                // way, no fresh IndexMap allocation and no key-Arc clone
                // per entry (IndexMap::values_mut preserves slot identity).
                let mut map = recv
                    .into_map()
                    .ok_or_else(|| EvalError("transformValues: expected object".into()))?;
                // Pattern-specialise `[PushCurrent, PushInt(K), <BinOp>]` —
                // the body of `transform_values(@ + K)` / `(@ - K)` / `(@ * K)`.
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
                // General path — mutate in place via values_mut; no new map,
                // no key reinsertion.
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

    /// Convenience wrapper: clones `env` once, runs the prog, discards
    /// the scratch.  Hot loops should use `exec_lam_body_scratch` to
    /// reuse a single scratch env instead.
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

    // ── Patch executor (VM-native) ──────────────────────────────────

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

    /// Scratch-reusing variant: mutates `scratch` in place via
    /// `Env::push_lam` / `Env::pop_lam` instead of cloning per item.
    /// The caller provides (and reuses) the scratch env across loop
    /// iterations.
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

    // ── Object construction ───────────────────────────────────────────────────

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

    // ── F-string ──────────────────────────────────────────────────────────────

    fn exec_fstring(&mut self, parts: &[CompiledFSPart], env: &Env) -> Result<Val, EvalError> {
        use std::fmt::Write as _;
        // Pre-size by summing literal lengths + rough 8 bytes per interp.
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
                    // Fast path: very small interp programs that just extract
                    // a field / index / current — avoid full self.exec() recursion
                    // (stack alloc, ic vec, etc.) per row.
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
                            // Fast paths: avoid val_to_string's temporary String.
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
        // `Arc::<str>::from(String)` transfers the buffer — no realloc.
        Ok(Val::Str(Arc::<str>::from(out)))
    }

    // ── Comprehension helpers ─────────────────────────────────────────────────

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

// ── Free helpers ──────────────────────────────────────────────────────────────

/// Route C — chained-descendant byte chain.
///
/// Entry: `root_key` is the key of the root `Descendant` that triggered the
/// scan.  `tail` is the opcode slice immediately after that `Descendant`.
///
/// Consumes as many tail opcodes as can be handled on byte spans:
///   - `Descendant(k)` — re-scan within each current span.
///   - `Quantifier(First)` — keep first span; `Quantifier(One)` when exactly one.
///   - `InlineFilter(pred)` / `CallMethod(Filter, [pred])` with a canonical
///     equality literal (int/string/bool/null) — bytewise retain.
///
/// Any other opcode terminates the chain; remaining spans are materialised
/// into `Val`s and returned, and the caller resumes normal opcode dispatch.
/// A "first-selector" opcode: bare `.first()` / `Quantifier::First`.
/// When `Descendant(k)` is followed by one of these, the byte scan
/// can stop at the first match per span.
fn is_first_selector_op(op: &Opcode) -> bool {
    match op {
        Opcode::Quantifier(QuantifierKind::First) => true,
        Opcode::CallMethod(c) if c.sub_progs.is_empty() && c.method == BuiltinMethod::First => true,
        _ => false,
    }
}

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

fn resolve_idx(i: i64, len: i64) -> usize {
    (if i < 0 { (len + i).max(0) } else { i }) as usize
}

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

/// Early-exit variant of `collect_desc`: returns the first self-first DFS
/// hit for `name`, matching the order that `collect_desc` would produce.
/// Powers the `$..key.first()` fast path.
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

/// Path-tracking variant — called when descending from root so paths are
/// root-relative and can be cached for future `RootChain` lookups.
/// `prefix` is mutated in-place (push/truncate) to avoid allocations.
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

fn resolve_pointer(root: &Val, ptr: &str) -> Val {
    let mut cur = root.clone();
    for seg in ptr.split('/').filter(|s| !s.is_empty()) {
        cur = cur.get_field(seg);
    }
    cur
}

#[derive(Clone, Copy)]
enum DictKeyShape {
    /// key prog is `[LoadIdent(v)]` — use item directly (stringify non-Str).
    Ident,
    /// key prog is `[LoadIdent(v), CallMethod{ToString, no args}]`.
    IdentToString,
}

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

// ── Opcode helpers ────────────────────────────────────────────────────────────

// WrapVal removed alongside the TopN opcode.

fn make_noarg_call(method: BuiltinMethod, name: &str) -> Opcode {
    Opcode::CallMethod(Arc::new(CompiledCall {
        method,
        name: Arc::from(name),
        sub_progs: Arc::from(&[] as &[Arc<Program>]),
        orig_args: Arc::from(&[] as &[Arg]),
        demand_max_keep: None,
    }))
}

// ── Hash helpers ──────────────────────────────────────────────────────────────

fn hash_str(s: &str) -> u64 {
    let mut h = DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

/// Hash the *structure* (keys + types) of a Val for the resolution cache key.
/// Does NOT hash values, only shape, so structural navigation results are
/// stable across different values in the same-shaped document.
fn hash_val_structure(v: &Val) -> u64 {
    let mut h = DefaultHasher::new();
    hash_structure_into(v, &mut h, 0);
    h.finish()
}

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
