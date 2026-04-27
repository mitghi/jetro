//! Pipeline IR — pull-based query plan that replaces hand-written
//! peephole fusions in `vm.rs`.
//!
//! Goal: a query like `$.orders.filter(total > 100).map(id).count()`
//! lowers to:
//!
//! ```ignore
//! Pipeline {
//!     source: Source::Field { base: Source::Root, key: "orders" },
//!     stages: vec![
//!         Stage::Filter(<prog: total > 100>),
//!         Stage::Map(<prog: id>),
//!     ],
//!     sink: Sink::Count,
//! }
//! ```
//!
//! Execution = single outer loop in [`Sink::run`] that pulls one
//! element from the source, threads it through the stages, writes
//! into the sink — no `Vec<Val>` between stages.
//!
//! Phase 1 (this module): pull-based [`Pipeline`] / [`Stage`] /
//! [`Source`] / [`Sink`] + a lowering path that handles a small
//! initial shape set (`Field`-chain source, `Filter`/`Map`/`Take`/
//! `Skip` stages, `Count`/`Sum`/`Collect` sinks).  Anything outside
//! the supported shape set falls back to the existing opcode path
//! by returning `None` from [`Pipeline::lower`].
//!
//! Phase 2 will add rewrite rules; Phase 3 will swap the per-element
//! `pull_next` for a per-batch `pull_batch` over `IntVec`/`FloatVec`/
//! `StrVec` columnar lanes.
//!
//! See `memory/project_pipeline_ir.md` for the full plan.

use std::sync::Arc;

use crate::ast::Expr;
use crate::eval::value::Val;
use crate::eval::EvalError;

/// Build per-slot typed columns from a row-major Val cells matrix.
/// First-row inspection picks the candidate type per slot; subsequent
/// rows must match or that slot falls back to `Mixed`.  Cost O(N×K)
/// — already paid by the cells walk in `try_promote_objvec`.
fn build_typed_cols(
    cells: &[Val],
    stride: usize,
    nrows: usize,
) -> Vec<crate::eval::value::ObjVecCol> {
    use crate::eval::value::ObjVecCol;
    let mut out: Vec<ObjVecCol> = Vec::with_capacity(stride);
    if stride == 0 || nrows == 0 {
        for _ in 0..stride { out.push(ObjVecCol::Mixed); }
        return out;
    }
    for slot in 0..stride {
        // Inspect first row's value at this slot to choose target.
        let target_tag: u8 = match &cells[slot] {
            Val::Int(_)   => 1,
            Val::Float(_) => 2,
            Val::Str(_)   => 3,
            Val::Bool(_)  => 4,
            _             => 0, // mixed / unsupported
        };
        if target_tag == 0 {
            out.push(ObjVecCol::Mixed);
            continue;
        }
        // Verify all rows.
        let mut ok = true;
        for r in 0..nrows {
            let v = &cells[r * stride + slot];
            let same = match (target_tag, v) {
                (1, Val::Int(_))   => true,
                (2, Val::Float(_)) => true,
                (3, Val::Str(_))   => true,
                (4, Val::Bool(_))  => true,
                _ => false,
            };
            if !same { ok = false; break; }
        }
        if !ok {
            out.push(ObjVecCol::Mixed);
            continue;
        }
        // Allocate typed lane.
        match target_tag {
            1 => {
                let mut col: Vec<i64> = Vec::with_capacity(nrows);
                for r in 0..nrows {
                    if let Val::Int(n) = &cells[r * stride + slot] { col.push(*n); }
                }
                out.push(ObjVecCol::Ints(col));
            }
            2 => {
                let mut col: Vec<f64> = Vec::with_capacity(nrows);
                for r in 0..nrows {
                    if let Val::Float(f) = &cells[r * stride + slot] { col.push(*f); }
                }
                out.push(ObjVecCol::Floats(col));
            }
            3 => {
                let mut col: Vec<Arc<str>> = Vec::with_capacity(nrows);
                for r in 0..nrows {
                    if let Val::Str(s) = &cells[r * stride + slot] { col.push(Arc::clone(s)); }
                }
                out.push(ObjVecCol::Strs(col));
            }
            4 => {
                let mut col: Vec<bool> = Vec::with_capacity(nrows);
                for r in 0..nrows {
                    if let Val::Bool(b) = &cells[r * stride + slot] { col.push(*b); }
                }
                out.push(ObjVecCol::Bools(col));
            }
            _ => out.push(ObjVecCol::Mixed),
        }
    }
    out
}

/// Promotes a `Val::Arr<Val::Obj>` to `Val::ObjVec` once per source array.
/// Implemented by `Jetro::get_or_promote_objvec` which memoises the result.
/// Trait so pipeline doesn't need to depend on the `Jetro` concrete type.
pub trait ObjVecPromoter {
    fn promote(&self, arr: &Arc<Vec<Val>>) -> Option<Arc<crate::eval::value::ObjVecData>>;
    /// Optional tape access — when present AND Val tree not yet
    /// materialised (`prefer_tape()` true), enables the tape-only
    /// query path (skip Val build entirely for tape-friendly shapes).
    /// Once Val is built, ObjVec slot kernels with typed columns
    /// outperform tape walking, so this gate flips off after first
    /// `root_val()` call.
    #[cfg(feature = "simd-json")]
    fn tape(&self) -> Option<&Arc<crate::strref::TapeData>> { None }
    /// True iff tape path is preferable to Val path right now.
    /// Implementations return true when Val tree hasn't been built
    /// yet AND tape is available.
    fn prefer_tape(&self) -> bool { false }
    /// Called by the tape-aggregate path after a successful run so
    /// the impl can flip its preference toward Val/ObjVec for warm
    /// follow-up queries.  Default is a no-op.
    fn note_tape_run(&self) {}
}

// ── Diagnostic tracing ──────────────────────────────────────────────────────
//
// `JETRO_PIPELINE_TRACE=1` env-var prints per-call lowering decisions to
// stderr.  Three event kinds:
//   activated: <Stage count, Sink kind, Source kind> for each lowered call
//   fallback : <reason, expr-label> when lower returns None
//   perf-ok / perf-loss : optional, set in benches via `pipeline_trace::report_run`
//
// Reads env var once into a static AtomicBool.  Zero overhead when disabled.
use std::sync::atomic::{AtomicU8, Ordering};
static TRACE_INIT: AtomicU8 = AtomicU8::new(0); // 0 = unread, 1 = off, 2 = on

#[inline]
pub(crate) fn trace_enabled() -> bool {
    let v = TRACE_INIT.load(Ordering::Relaxed);
    if v != 0 { return v == 2; }
    let on = std::env::var_os("JETRO_PIPELINE_TRACE").is_some();
    TRACE_INIT.store(if on { 2 } else { 1 }, Ordering::Relaxed);
    on
}

// Layer B is on by default after bench validation. `JETRO_COMPOSED=0`
// opts out for diagnostic comparison against legacy. Cached after
// first read. Tier 3 step 2 deletes the gate entirely along with
// the fused Sink variants + ~30 fused vm.rs opcodes.
static COMPOSED_INIT: AtomicU8 = AtomicU8::new(0);

#[inline]
pub(crate) fn composed_path_enabled() -> bool {
    let v = COMPOSED_INIT.load(Ordering::Relaxed);
    if v != 0 { return v == 2; }
    let off = match std::env::var("JETRO_COMPOSED") {
        Ok(s) => s == "0" || s.eq_ignore_ascii_case("off") || s.eq_ignore_ascii_case("false"),
        Err(_) => false,
    };
    let on = !off;
    COMPOSED_INIT.store(if on { 2 } else { 1 }, Ordering::Relaxed);
    on
}

fn sink_name(s: &Sink) -> &'static str {
    match s {
        Sink::Collect => "collect",
        Sink::Count => "count",
        Sink::Numeric(NumOp::Sum) => "sum",
        Sink::Numeric(NumOp::Min) => "min",
        Sink::Numeric(NumOp::Max) => "max",
        Sink::Numeric(NumOp::Avg) => "avg",
        Sink::First => "first",
        Sink::Last => "last",
    }
}

fn source_name(s: &Source) -> &'static str {
    match s {
        Source::Receiver(_) => "receiver",
        Source::FieldChain { .. } => "field_chain",
    }
}

fn expr_label(e: &Expr) -> &'static str {
    match e {
        Expr::Chain(_, _) => "chain",
        Expr::Pipeline { .. } => "pipeline",
        Expr::Object(_) => "object",
        Expr::Array(_) => "array",
        Expr::ListComp { .. } => "list_comp",
        Expr::DictComp { .. } => "dict_comp",
        Expr::Let { .. } => "let",
        Expr::Patch { .. } => "patch",
        Expr::Lambda { .. } => "lambda",
        Expr::IfElse { .. } => "if_else",
        Expr::BinOp(_, _, _) => "binop",
        Expr::Root => "root_only",
        _ => "other",
    }
}

// ── Plan types ───────────────────────────────────────────────────────────────

/// Where a pipeline starts.  Currently a small set; Phase 2/3 add
/// `DeepScan(key)` (tape byte-scan), `Range(i64, i64)`, and
/// `Tape(Arc<TapeData>)`.
#[derive(Debug, Clone)]
pub enum Source {
    /// Pull from a concrete `Val::Arr` / `Val::IntVec` / `Val::FloatVec` /
    /// `Val::StrVec` / `Val::ObjVec` already on the stack.
    Receiver(Val),
    /// Walk `$.<keys[0]>.<keys[1]>…` from the document root, then iterate
    /// the array at the end of the chain.  `keys.is_empty()` means the
    /// document root itself is the iterable.
    FieldChain { keys: Arc<[Arc<str>]> },
}

/// A pull-based stage.  Streaming stages (Filter / Map / FlatMap /
/// Take / Skip) flow elements through one at a time; barrier stages
/// (Reverse / Sort / UniqueBy) require the full input materialised.
/// Filter / Map / FlatMap / UniqueBy carry a pre-compiled `Program`
/// reused per row — drops the per-row tree-walker dispatch cost.
#[derive(Debug, Clone)]
pub enum Stage {
    /// `.filter(pred)` — drops elements where `pred` is falsy.
    Filter(Arc<crate::vm::Program>),
    /// `.map(f)` — replaces each element with `f(@)`.
    Map(Arc<crate::vm::Program>),
    /// `.flat_map(f)` — `f(@)` must yield an iterable; flattens one
    /// level into the pull stream.
    FlatMap(Arc<crate::vm::Program>),
    /// `.take(n)` — yields at most `n` elements, then completes.
    Take(usize),
    /// `.skip(n)` — drops the first `n` elements.
    Skip(usize),
    /// `.reverse()` — barrier; materialises and reverses.
    Reverse,
    /// `.unique()` (None) / `.unique_by(key)` (Some) — barrier;
    /// materialises, dedupes by key (or by full value if `None`).
    UniqueBy(Option<Arc<crate::vm::Program>>),
    /// `.sort()` (None) / `.sort_by(key)` (Some) — barrier;
    /// materialises and sorts.
    Sort(Option<Arc<crate::vm::Program>>),
    /// `.group_by(key)` — barrier; partitions rows by key,
    /// produces `Val::Obj { key_str → Vec<row> }`.  As a Stage
    /// this is a sink-shaped operation; placed under Stage so
    /// downstream `.values()` / `.map(@.len())` can compose.
    GroupBy(Arc<crate::vm::Program>),

    // ── Step 3d-extension (C): lifted string Stages ──────────────────────────
    //
    // Lifts `.split(sep)` / `.slice(a, b)` from MethodRegistry-dispatched
    // method calls inside Map/Filter sub-program bodies into first-class
    // Stage variants.  Two wins:
    //   1. Chain flattening (Step 3d-extension A) can hoist them out of
    //      Map bodies — `map(@.text.split(",").first())` becomes
    //      `[Map(@.text), Split(","), Sink::First]`.
    //   2. apply_indexed override on Split lets IndexedDispatch compute
    //      `split(",").first()` via one memchr call instead of producing
    //      the full segment vector.

    /// `.split(sep)` — 1 string → many parts.  `Cardinality::Expanding`
    /// + `can_indexed=true` (kth segment via memchr).
    Split(Arc<str>),
    /// `.slice(start, end)` — 1 string → 1 substring.  `Cardinality::
    /// OneToOne` + `can_indexed=true`.  `end=None` means "to end".
    Slice(i64, Option<i64>),

    /// `.replace(needle, replacement)` (`all=false`, replacen-1-style) and
    /// `.replace_all(needle, replacement)` (`all=true`) — 1 string → 1
    /// string.  `Cardinality::OneToOne` + `can_indexed=true`.
    Replace { needle: Arc<str>, replacement: Arc<str>, all: bool },

    /// `.chunk(n)` — partitions the upstream stream into chunks of size n
    /// (last chunk may be shorter).  Barrier — needs the full stream.
    /// Each emitted element is a `Val::arr` of n upstream values.
    Chunk(usize),
    /// `.window(n)` — sliding window of size n over the upstream stream.
    /// Barrier.  Emits `len.saturating_sub(n) + 1` overlapping windows.
    Window(usize),

    /// Step 3d-extension (A2): recursive sub-pipeline planning.
    /// Outer `.map(@.<chain>)` whose body is itself a recognisable
    /// chain compiles BODY into its own `Plan` instead of an opaque
    /// `vm::Program`.  Per outer element, the inner Plan runs against
    /// that element as the seed.  `Cardinality::OneToOne` (one inner
    /// run per outer element) + `can_indexed=true`.  Wins on shapes
    /// like `map(@.text.split(",").first())` — inner Plan reduces via
    /// IndexedDispatch / EarlyExit etc., not full materialisation.
    CompiledMap(Arc<Plan>),

    // ── lift_all_builtins (object family) ────────────────────────────
    //
    // First-class Pipeline IR variants for zero-arg object methods.
    // Lower at the chain classifier; canonical kernels live as
    // `*_apply` free fns below.  Per `lift_all_builtins.md`: built-ins
    // lower DIRECTLY to Stage::* variants, no CallMethod dispatch.

    /// `.keys()` — Obj → Arr<Str>.  `Cardinality::OneToOne`, pure.
    Keys,
    /// `.values()` — Obj → Arr<Val>.  `Cardinality::OneToOne`, pure.
    Values,
    /// `.entries()` — Obj → Arr<Arr[Str, Val]>.  `Cardinality::OneToOne`, pure.
    Entries,
}

/// Phase A3 — sub-program "kernel" shape recognised at lower-time.
/// Classifying a Stage's program into one of these shapes lets the
/// per-row pull loop dispatch a specialised inline path (direct field
/// pluck, field-cmp-lit, etc.) instead of re-entering `vm.exec` per
/// element.  The kernel is a *closed set of byte-code patterns* —
/// classification is mechanical, not per-method hand-coding.  Falls
/// back to `Generic` for shapes outside the set.
#[derive(Debug, Clone)]
pub enum BodyKernel {
    Generic,
    /// `[PushCurrent, GetField(k)]`  →  item.get_field(k)
    FieldRead(Arc<str>),
    /// `[PushCurrent, FieldChain([k1, k2, …])]`  →  walk k1.k2.k3 from item
    FieldChain(Arc<[Arc<str>]>),
    /// `[PushCurrent, GetField(k), <lit-push>, <cmp-op>]`  →  pred
    FieldCmpLit(Arc<str>, crate::ast::BinOp, Val),
    /// `[PushCurrent, FieldChain(...), <lit-push>, <cmp-op>]`  →  pred
    FieldChainCmpLit(Arc<[Arc<str>]>, crate::ast::BinOp, Val),
    /// `[PushCurrent, <lit-push>, <cmp-op>]`  →  bare `@ <op> lit` predicate.
    /// Covers numeric-vec filter shapes (`$.scores.filter(@ > 50)…`).
    CurrentCmpLit(crate::ast::BinOp, Val),
    /// Predicate is a constant boolean (e.g. `@ == 1` already folded).
    ConstBool(bool),
    /// Body produces a constant value independent of item.
    Const(Val),
    /// Body is a single `MakeObj` whose entries are all path projections
    /// (Short shorthand `{name}` or `KvPath` with Field-only steps).
    /// Per-row eval walks each path on the current entry and emits a
    /// `Val::ObjSmall` with the projected fields. Generic across any
    /// number of fields and any path depth — classifier-driven, no
    /// per-shape walker.
    ObjProject(Arc<[ObjProjEntry]>),
    /// F-string interpolation body — list of literal chunks and path
    /// interpolations.  Per-row eval builds a String by concatenating
    /// literals + path-resolved values formatted as plain text.
    /// Generic across any number of parts; classifier-driven.
    FString(Arc<[FStrPart]>),
    /// Binary arithmetic over two operands (path or literal).  Covers
    /// `qty * price`, `score + 10`, `a - b`, `a / b`, `a % b`.  Per-row
    /// eval reads operands via tape, applies op, returns numeric
    /// TapeVal (Float if any operand is Float, else Int).  Generic
    /// across any (Path|Lit, ArithOp, Path|Lit) combo via one classifier.
    Arith(ArithOperand, ArithOp, ArithOperand),
}

/// One f-string part — literal chunk or path interpolation.
#[derive(Debug, Clone)]
pub enum FStrPart {
    Lit(Arc<str>),
    /// Interpolated path on current entry.  Empty path = current itself.
    Path(Arc<[Arc<str>]>),
}

/// Arithmetic operand for `BodyKernel::Arith`.
#[derive(Debug, Clone)]
pub enum ArithOperand {
    /// Path on the current tape entry.  Empty path = current itself.
    Path(Arc<[Arc<str>]>),
    /// Numeric literal pushed at compile-time.
    LitInt(i64),
    LitFloat(f64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithOp { Add, Sub, Mul, Div, Mod }

/// One field of an ObjProject kernel — output key + value shape.
/// Three shapes:
///   - `Path`: walk path on entry; emit primitive value
///   - `InnerLen`: walk path → expect Array → emit Int(len)
///   - `InnerMapSum`: walk path → expect Array of Objects → fold
///     numeric value extracted via `inner` kernel per inner entry,
///     return Int/Float sum
#[derive(Debug, Clone)]
pub enum ObjProjEntry {
    Path { key: Arc<str>, path: Arc<[Arc<str>]> },
    InnerLen { key: Arc<str>, path: Arc<[Arc<str>]> },
    InnerMapSum { key: Arc<str>, path: Arc<[Arc<str>]>, inner: Arc<BodyKernel> },
}

impl ObjProjEntry {
    pub fn key(&self) -> &Arc<str> {
        match self {
            ObjProjEntry::Path { key, .. } => key,
            ObjProjEntry::InnerLen { key, .. } => key,
            ObjProjEntry::InnerMapSum { key, .. } => key,
        }
    }
}

impl BodyKernel {
    /// Classify a sub-program into a kernel.  Reads the byte-code shape;
    /// no AST inspection.  Returns `Generic` for unrecognised shapes —
    /// always safe to fall back.
    pub fn classify(prog: &crate::vm::Program) -> Self {
        use crate::vm::Opcode;
        let ops = prog.ops.as_ref();
        // Constant body — push-only programs.  PushInt / PushFloat /
        // PushStr / PushBool / PushNull all classify as Const(<lit>).
        if ops.len() == 1 {
            if let Some(lit) = trivial_lit(&ops[0]) {
                return match &ops[0] {
                    Opcode::PushBool(b) => Self::ConstBool(*b),
                    _ => Self::Const(lit),
                };
            }
        }
        // Single-step shorthands.  LoadIdent(k) inside a lambda body
        // resolves to current.get_field(k) at runtime — same semantics
        // as PushCurrent + GetField(k).  Treat as FieldRead.
        match ops {
            [Opcode::PushCurrent, Opcode::GetField(k)]
            | [Opcode::GetField(k)]
            | [Opcode::LoadIdent(k)] =>
                return Self::FieldRead(k.clone()),
            [Opcode::PushCurrent, Opcode::FieldChain(fc)]
            | [Opcode::FieldChain(fc)] =>
                return Self::FieldChain(fc.keys.clone()),
            // `LoadIdent(k1) + GetField(k2) [+ GetField(k3) …]` —
            // LoadIdent shorthand resolves to current.get_field(k1),
            // so the whole shape is a chain rooted at `current`.
            [Opcode::LoadIdent(k1), rest @ ..] if rest.iter()
                .all(|o| matches!(o, Opcode::GetField(_))) =>
            {
                let mut keys = vec![k1.clone()];
                for o in rest {
                    if let Opcode::GetField(k) = o { keys.push(k.clone()); }
                }
                return Self::FieldChain(keys.into());
            }
            // `LoadIdent(k1) + FieldChain([k2, k3, ...])` — the field
            // walk peephole (FieldChain) folded after a single
            // LoadIdent prefix.  Same semantics as a multi-step path
            // rooted at current.
            [Opcode::LoadIdent(k1), Opcode::FieldChain(fc)] => {
                let mut keys = vec![k1.clone()];
                for k in fc.keys.iter() { keys.push(k.clone()); }
                return Self::FieldChain(keys.into());
            }
            _ => {}
        }
        // <field-read>, <lit>, <cmp>  →  FieldCmpLit / FieldChainCmpLit
        let rest: &[Opcode] = if matches!(ops.first(), Some(Opcode::PushCurrent)) {
            &ops[1..]
        } else { ops };
        if rest.len() == 3 {
            // Bare `@ <op> lit` — current as lhs, literal as rhs.
            // This shape matters for IntVec / FloatVec / StrVec
            // filter chains (`$.scores.filter(@ > 50)`).
            if matches!(&rest[0], Opcode::PushCurrent) {
                if let Some(lit) = trivial_lit(&rest[1]) {
                    if let Some(bo) = cmp_to_binop(&rest[2]) {
                        return Self::CurrentCmpLit(bo, lit);
                    }
                }
            }
            // Single-field cmp.
            let single_key = match &rest[0] {
                Opcode::LoadIdent(k) | Opcode::GetField(k) => Some(k.clone()),
                _ => None,
            };
            if let Some(k) = single_key {
                if let Some(lit) = trivial_lit(&rest[1]) {
                    if let Some(bo) = cmp_to_binop(&rest[2]) {
                        return Self::FieldCmpLit(k, bo, lit);
                    }
                }
            }
            // Chain cmp.
            if let Opcode::FieldChain(fc) = &rest[0] {
                if let Some(lit) = trivial_lit(&rest[1]) {
                    if let Some(bo) = cmp_to_binop(&rest[2]) {
                        return Self::FieldChainCmpLit(fc.keys.clone(), bo, lit);
                    }
                }
            }
        }
        // ObjProject — single-opcode `MakeObj(entries)` whose entries
        // are all path projections (Short / KvPath-Field-only).  Body
        // shape `{k1, k2: a.b, k3}` compiles to exactly this.
        if ops.len() == 1 {
            if let Opcode::MakeObj(entries) = &ops[0] {
                if let Some(proj) = classify_obj_project(entries) {
                    return Self::ObjProject(proj.into());
                }
            }
        }
        // FString — single-opcode `FString(parts)` body.  Each Interp's
        // sub-program is classified; supported when it's a path read.
        if ops.len() == 1 {
            if let Opcode::FString(parts) = &ops[0] {
                if let Some(fk) = classify_fstring(parts) {
                    return Self::FString(fk.into());
                }
            }
        }
        // Arith — RPN-shaped binary arithmetic body.  Body shape
        // `<lhs> <rhs> <op>` covers `qty * price`, `score + 10`,
        // `a / b`, etc. Operands are paths (FieldRead/FieldChain) or
        // numeric literals.  Generic across all combos via classifier.
        if let Some(arith) = classify_arith(ops) { return arith; }
        Self::Generic
    }
}

/// Classify body opcodes as an `Arith` kernel.
/// Two recognised shapes:
///   1. `[<lhs-ops>, <rhs-ops>, <arith-op>]` — binary arith
///   2. `[<operand-ops>, Neg]` — unary negation (treated as 0 - operand)
/// Operands are paths (PushCurrent + GetField/FieldChain or LoadIdent)
/// or numeric literals (PushInt / PushFloat).
fn classify_arith(ops: &[crate::vm::Opcode]) -> Option<BodyKernel> {
    use crate::vm::Opcode;
    let last = ops.last()?;
    // Unary negation shape: `<operand>, Neg`.
    if matches!(last, Opcode::Neg) && ops.len() >= 2 {
        let prefix = &ops[..ops.len() - 1];
        if let Some(operand) = parse_arith_operand(prefix) {
            return Some(BodyKernel::Arith(
                ArithOperand::LitInt(0),
                ArithOp::Sub,
                operand,
            ));
        }
    }
    let op = match last {
        Opcode::Add => ArithOp::Add,
        Opcode::Sub => ArithOp::Sub,
        Opcode::Mul => ArithOp::Mul,
        Opcode::Div => ArithOp::Div,
        Opcode::Mod => ArithOp::Mod,
        _ => return None,
    };
    let prefix = &ops[..ops.len() - 1];
    for split in 1..prefix.len() {
        let lhs = &prefix[..split];
        let rhs = &prefix[split..];
        if let (Some(l), Some(r)) = (parse_arith_operand(lhs), parse_arith_operand(rhs)) {
            return Some(BodyKernel::Arith(l, op, r));
        }
    }
    None
}

/// Classify CompiledFSPart list as an FString kernel.  Each Interp
/// part's sub-program must be a path-shaped body (FieldRead /
/// FieldChain — same shapes accepted by other path kernels).  Bails
/// (returns None) on Generic interpolations or on parts that carry a
/// non-default fmt spec.
fn classify_fstring(
    parts: &[crate::vm::CompiledFSPart],
) -> Option<Vec<FStrPart>> {
    use crate::vm::CompiledFSPart as P;
    let mut out: Vec<FStrPart> = Vec::with_capacity(parts.len());
    for p in parts {
        match p {
            P::Lit(s) => out.push(FStrPart::Lit(s.clone())),
            P::Interp { prog, fmt: None } => {
                let kernel = BodyKernel::classify(prog);
                let path = match kernel {
                    BodyKernel::FieldRead(k) =>
                        Arc::from(vec![k].into_boxed_slice()),
                    BodyKernel::FieldChain(keys) => keys,
                    _ => return None,
                };
                out.push(FStrPart::Path(path));
            }
            P::Interp { fmt: Some(_), .. } => return None,
        }
    }
    Some(out)
}

fn parse_arith_operand(ops: &[crate::vm::Opcode]) -> Option<ArithOperand> {
    use crate::vm::Opcode;
    match ops {
        [Opcode::PushInt(n)] => Some(ArithOperand::LitInt(*n)),
        [Opcode::PushFloat(f)] => Some(ArithOperand::LitFloat(*f)),
        [Opcode::LoadIdent(k)] | [Opcode::GetField(k)]
        | [Opcode::PushCurrent, Opcode::GetField(k)] => {
            Some(ArithOperand::Path(Arc::from(vec![k.clone()].into_boxed_slice())))
        }
        [Opcode::FieldChain(fc)]
        | [Opcode::PushCurrent, Opcode::FieldChain(fc)] => {
            Some(ArithOperand::Path(fc.keys.clone()))
        }
        _ => None,
    }
}

/// Try to classify a `MakeObj` entry list as a path-only projection.
/// Returns `None` if any entry is non-path (Dynamic / Spread / KvPath
/// with Index step / KvPath optional / KvPath cond / Kv with non-trivial
/// sub-program).
fn classify_obj_project(
    entries: &[crate::vm::CompiledObjEntry],
) -> Option<Vec<ObjProjEntry>> {
    use crate::vm::CompiledObjEntry as E;
    use crate::vm::KvStep;
    let mut out: Vec<ObjProjEntry> = Vec::with_capacity(entries.len());
    for e in entries {
        match e {
            E::Short { name, .. } => out.push(ObjProjEntry::Path {
                key: name.clone(),
                path: Arc::from(vec![name.clone()].into_boxed_slice()),
            }),
            E::KvPath { key, steps, optional: false, .. } => {
                let mut path: Vec<Arc<str>> = Vec::with_capacity(steps.len());
                for s in steps.iter() {
                    match s {
                        KvStep::Field(k) => path.push(k.clone()),
                        KvStep::Index(_) => return None,
                    }
                }
                out.push(ObjProjEntry::Path { key: key.clone(), path: path.into() });
            }
            E::Kv { key, prog, optional: false, cond: None } => {
                if let Some(entry) = classify_kv_method(key, prog) {
                    out.push(entry);
                } else {
                    return None;
                }
            }
            _ => return None,
        }
    }
    Some(out)
}

/// Classify a Kv entry's prog for nested-method shapes.  Recognises:
///   `[<path-load>, CallMethod(Len, [])]` -> InnerLen
///   `[<path-load>, MapSum(<inner-prog>)]` -> InnerMapSum
/// where <path-load> is LoadIdent / GetField / FieldChain (single op).
///
/// Also handles plain multi-step path projections: when the prog
/// classifies via `BodyKernel::classify` to a `FieldRead` or
/// `FieldChain` kernel, emit it as `ObjProjEntry::Path` with the
/// resolved chain.  This unlocks `name: user.name` style entries
/// inside `{id, name: user.name, score}` shape projections — which
/// otherwise fall through to Generic and bypass byte-friendly
/// classification entirely.
fn classify_kv_method(
    key: &Arc<str>,
    prog: &crate::vm::Program,
) -> Option<ObjProjEntry> {
    use crate::vm::{Opcode, BuiltinMethod};
    // Path projection shortcut — covers any prog that classifies as
    // a pure path read.  Wraps the chain into ObjProjEntry::Path.
    match BodyKernel::classify(prog) {
        BodyKernel::FieldRead(name) => {
            return Some(ObjProjEntry::Path {
                key: key.clone(),
                path: Arc::from(vec![name].into_boxed_slice()),
            });
        }
        BodyKernel::FieldChain(keys) => {
            return Some(ObjProjEntry::Path { key: key.clone(), path: keys });
        }
        _ => {}
    }
    let ops = prog.ops.as_ref();
    if ops.len() != 2 { return None; }
    let path: Arc<[Arc<str>]> = match &ops[0] {
        Opcode::LoadIdent(k) | Opcode::GetField(k) =>
            Arc::from(vec![k.clone()].into_boxed_slice()),
        Opcode::FieldChain(fc) => fc.keys.clone(),
        _ => return None,
    };
    match &ops[1] {
        Opcode::CallMethod(c)
            if c.method == BuiltinMethod::Len && c.sub_progs.is_empty() =>
        {
            Some(ObjProjEntry::InnerLen { key: key.clone(), path })
        }
        _ => None,
    }
}

#[inline]
fn trivial_lit(op: &crate::vm::Opcode) -> Option<Val> {
    use crate::vm::Opcode;
    match op {
        Opcode::PushInt(n)   => Some(Val::Int(*n)),
        Opcode::PushFloat(f) => Some(Val::Float(*f)),
        Opcode::PushStr(s)   => Some(Val::Str(s.clone())),
        Opcode::PushBool(b)  => Some(Val::Bool(*b)),
        Opcode::PushNull     => Some(Val::Null),
        _ => None,
    }
}

#[inline]
fn cmp_to_binop(op: &crate::vm::Opcode) -> Option<crate::ast::BinOp> {
    use crate::vm::Opcode as O;
    use crate::ast::BinOp as B;
    match op {
        O::Eq  => Some(B::Eq),
        O::Neq => Some(B::Neq),
        O::Lt  => Some(B::Lt),
        O::Lte => Some(B::Lte),
        O::Gt  => Some(B::Gt),
        O::Gte => Some(B::Gte),
        _ => None,
    }
}

/// Numeric fold operator — common shape across `sum`/`min`/`max`/`avg`.
/// Centralising the op here lets a single set of fused-sink variants
/// (`NumMap`, `NumFilterMap`) cover four aggregate shapes instead of
/// twelve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumOp {
    Sum,
    Min,
    Max,
    Avg,
}

impl NumOp {
    /// Render the bare-aggregate result for the bench's start state
    /// (zero-element fold).  Sum/Avg → 0/Null; Min/Max → Null when
    /// no element observed.
    fn empty(self) -> Val {
        match self {
            NumOp::Sum => Val::Int(0),
            NumOp::Avg => Val::Null,
            NumOp::Min => Val::Null,
            NumOp::Max => Val::Null,
        }
    }
}

/// Where pipeline output lands.  Determines the result type.
///
/// Single mechanism per terminal kind. Fused variants (NumMap,
/// NumFilterMap, CountIf, etc.) deleted in Tier 3 — composed
/// substrate + canonical view dispatch handle every chain shape via
/// base form.
#[derive(Debug, Clone)]
pub enum Sink {
    /// Materialise every element into a `Val::Arr`.
    Collect,
    /// `.count()` / `.len()` — yield the number of elements that
    /// reached the sink as a `Val::Int`.
    Count,
    /// `.sum()`/`.min()`/`.max()`/`.avg()` over numerics.
    Numeric(NumOp),
    /// `.first()` / `.last()` — yield the first/last element or
    /// `Val::Null`.
    First,
    Last,
}

// ── Step 3d Phase 1: demand propagation ─────────────────────────────────────
//
// Two ADTs declare what a Sink wants and what each Stage needs from its
// upstream.  `Pipeline::compute_strategies` walks stages backward,
// propagating demand; barrier stages observe their downstream demand
// and pick algorithm (full sort vs top-k).  No per-shape rewrite rule;
// `Sort ∘ First` / `Sort ∘ Take(k)` reduce automatically through this
// propagation.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bound {
    Unbounded,
    AtMost(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Position { First, Last, Nth(usize) }

#[derive(Debug, Clone, Copy)]
pub struct Demand {
    pub consumption: Bound,
    pub positional:  Option<Position>,
}

impl Demand {
    pub const UNBOUNDED: Demand = Demand { consumption: Bound::Unbounded, positional: None };
}

impl Sink {
    /// Declare consumption budget + positional preference.  Drives
    /// Phase 1 demand propagation; barriers use `consumption` to pick
    /// full-sort vs top-k.
    pub fn demand(&self) -> Demand {
        match self {
            Sink::First => Demand {
                consumption: Bound::AtMost(1),
                positional:  Some(Position::First),
            },
            Sink::Last => Demand {
                consumption: Bound::AtMost(1),
                positional:  Some(Position::Last),
            },
            Sink::Count | Sink::Numeric(_) | Sink::Collect => Demand::UNBOUNDED,
        }
    }
}

/// Per-stage strategy chosen by Phase 1 propagation.  Default is
/// "no adaptation"; Sort observes a bounded downstream demand and
/// switches to TopK (keep smallest k) or BottomK (keep largest k)
/// depending on positional preference.  Length matches `Pipeline::stages`.
#[derive(Debug, Clone, Copy)]
pub enum StageStrategy {
    Default,
    /// Sort + downstream wants AtMost(k) starting from the beginning
    /// (`First`, `Take(k)`, `Nth(k)`-near-front).  Keep the k smallest-
    /// by-key elements, sorted ascending.
    SortTopK(usize),
    /// Sort + downstream wants the tail (`Last`).  Keep the k largest-
    /// by-key elements, sorted ascending.  `Sink::Last` then picks the
    /// final element which is the global max (matching full-sort∘Last
    /// semantics).
    SortBottomK(usize),
}

/// Walk stages right-to-left, threading downstream `Demand` through each.
/// For each stage, decide its strategy from the demand seen *below* it,
/// then derive the demand the stage exposes to its *upstream*.  This is
/// the unified mechanism that subsumes `Sort ∘ First → MinBy`,
/// `Sort_by ∘ Last → MaxBy`, `Sort ∘ Take(k) → TopN(k)` rewrite rules.
pub fn compute_strategies(stages: &[Stage], sink: &Sink) -> Vec<StageStrategy> {
    let mut strategies: Vec<StageStrategy> = vec![StageStrategy::Default; stages.len()];
    let mut demand = sink.demand();
    for (i, stage) in stages.iter().enumerate().rev() {
        // Phase 1: observe & adapt
        if let Stage::Sort(_) = stage {
            if let Bound::AtMost(k) = demand.consumption {
                strategies[i] = match demand.positional {
                    Some(Position::Last) => StageStrategy::SortBottomK(k),
                    _                    => StageStrategy::SortTopK(k),
                };
            }
        }
        // Phase 2: propagate demand upward
        demand = upstream_demand(demand, stage);
    }
    strategies
}

/// What demand does upstream see, given downstream `d` and a stage?
#[inline]
fn upstream_demand(d: Demand, stage: &Stage) -> Demand {
    match stage {
        // 1:1 — preserve consumption.  Map preserves position; Filter
        // can drop elements so downstream's "first" / "last" refers to
        // post-filter position — upstream demand is unbounded.
        Stage::Map(_) => d,
        Stage::Filter(_) => Demand { consumption: d.consumption, positional: None },
        // Take(n) caps upstream.
        Stage::Take(n) => {
            let cap = match d.consumption {
                Bound::Unbounded => *n,
                Bound::AtMost(k) => k.min(*n),
            };
            Demand { consumption: Bound::AtMost(cap), positional: None }
        }
        // Skip + barriers + Expanding string Stages all need full
        // upstream stream (or its sole element, for Split: 1 string).
        Stage::Skip(_)
            | Stage::FlatMap(_)
            | Stage::Reverse
            | Stage::Sort(_)
            | Stage::UniqueBy(_)
            | Stage::GroupBy(_)
            | Stage::Split(_)
            | Stage::Chunk(_)
            | Stage::Window(_) => Demand::UNBOUNDED,
        // Slice / Replace / CompiledMap / Keys / Values / Entries are
        // 1:1 — preserve demand.  Keys/Values/Entries each consume one
        // object and emit one array; downstream demand passes through.
        Stage::Slice(_, _) | Stage::Replace { .. } | Stage::CompiledMap(_)
        | Stage::Keys | Stage::Values | Stage::Entries => d,
    }
}

// ── Step 3d Phases 2-5: StageShape ADT + planning function ─────────────────
//
// Per `pipeline_unification.md` Step 3d.  Each Stage declares its shape
// via `Stage::shape() -> StageShape`.  The planning function consumes
// these declarations + Sink::demand() and produces an execution
// `Strategy`.  Five strategies cover every chain shape; new Stages
// plug in via shape() with no planning changes.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cardinality {
    /// 1 input → 1 output (Map).
    OneToOne,
    /// 1 input → {0, 1} output (Filter, TakeWhile, DropWhile).
    Filtering,
    /// 1 input → many output (FlatMap).
    Expanding,
    /// Changes whole stream (Sort, GroupBy, UniqueBy, Reverse).
    Barrier,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Order {
    /// Position-independent (Filter, Map).  Reordering allowed across
    /// adjacent Stateless stages.
    Stateless,
    /// Position-dependent (Take, Skip).  Reordering forbidden across
    /// Stateful boundary.
    Stateful,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Boundedness {
    /// 1 input always produces 1 output (Map).  Bound::AtMost(1) downstream
    /// translates 1:1 upstream.
    Always(usize),
    /// At most `k` outputs per input (Filter=1, Take(n)=n, FlatMap=unbounded).
    AtMost(Bound),
}

#[derive(Debug, Clone, Copy)]
pub struct StageShape {
    pub cardinality: Cardinality,
    pub order:       Order,
    /// Pure = no side effects, no MethodRegistry callouts.  Used by
    /// Phase 2 dead-stage elimination.
    pub purity:      bool,
    pub boundedness: Boundedness,
    /// True iff `apply(arr[idx])` suffices to compute element `idx` —
    /// i.e. Stage doesn't need to look at neighbours.  Map=true,
    /// Filter=true, TakeWhile=false, Sort=false.
    pub can_indexed: bool,
    /// Phase 3 reorder heuristics — cost is per-element evaluation cost
    /// (relative; Generic VM round-trip ≈ 10, field-read ≈ 1).
    pub cost:        f64,
    /// Probability an input element passes through (for Filter).
    /// 0.0 = always reject, 1.0 = always pass; 0.5 = unknown.
    /// For non-Filter stages this is 1.0 (pass-through).
    pub selectivity: f64,
}

impl Stage {
    /// Declarative shape — drives Phase 2-5 planning.  No per-chain
    /// dispatch logic; planning consults this + Sink::demand().
    pub fn shape(&self) -> StageShape {
        // Default cost/selectivity reflects the generic VM-fallback per-row
        // sub-program evaluation cost (~10 relative units).  Phase 3 reorder
        // refines these via `kernel_cost_selectivity()` when a specific
        // BodyKernel is recognised.
        match self {
            Stage::Map(_) => StageShape {
                cardinality: Cardinality::OneToOne,
                order:       Order::Stateless,
                purity:      true,
                boundedness: Boundedness::Always(1),
                can_indexed: true,
                cost:        10.0,
                selectivity: 1.0,
            },
            Stage::Filter(_) => StageShape {
                cardinality: Cardinality::Filtering,
                order:       Order::Stateless,
                purity:      true,
                boundedness: Boundedness::AtMost(Bound::AtMost(1)),
                can_indexed: false,
                cost:        10.0,
                selectivity: 0.5,
            },
            Stage::FlatMap(_) => StageShape {
                cardinality: Cardinality::Expanding,
                order:       Order::Stateless,
                purity:      true,
                boundedness: Boundedness::AtMost(Bound::Unbounded),
                can_indexed: false,
                cost:        10.0,
                selectivity: 1.0,
            },
            Stage::Take(_) | Stage::Skip(_) => StageShape {
                cardinality: Cardinality::Filtering,
                order:       Order::Stateful,
                purity:      true,
                boundedness: Boundedness::AtMost(Bound::AtMost(1)),
                can_indexed: false,
                cost:        0.5,
                selectivity: 0.5,
            },
            Stage::Reverse | Stage::Sort(_) | Stage::UniqueBy(_) | Stage::GroupBy(_)
                => StageShape {
                cardinality: Cardinality::Barrier,
                order:       Order::Stateful,
                purity:      true,
                boundedness: Boundedness::AtMost(Bound::Unbounded),
                can_indexed: false,
                cost:        20.0,
                selectivity: 1.0,
            },
            // Step 3d-extension lifted Stages.
            Stage::Split(_) => StageShape {
                cardinality: Cardinality::Expanding,
                order:       Order::Stateless,
                purity:      true,
                boundedness: Boundedness::AtMost(Bound::Unbounded),
                can_indexed: true,
                cost:        2.0,
                selectivity: 1.0,
            },
            Stage::Slice(_, _) => StageShape {
                cardinality: Cardinality::OneToOne,
                order:       Order::Stateless,
                purity:      true,
                boundedness: Boundedness::Always(1),
                can_indexed: true,
                cost:        1.0,
                selectivity: 1.0,
            },
            Stage::Replace { .. } => StageShape {
                cardinality: Cardinality::OneToOne,
                order:       Order::Stateless,
                purity:      true,
                boundedness: Boundedness::Always(1),
                can_indexed: true,
                cost:        2.0,
                selectivity: 1.0,
            },
            Stage::Chunk(_) | Stage::Window(_) => StageShape {
                cardinality: Cardinality::Barrier,
                order:       Order::Stateful,
                purity:      true,
                boundedness: Boundedness::AtMost(Bound::Unbounded),
                can_indexed: true,
                cost:        2.0,
                selectivity: 1.0,
            },
            Stage::CompiledMap(_) => StageShape {
                cardinality: Cardinality::OneToOne,
                order:       Order::Stateless,
                purity:      true,
                boundedness: Boundedness::Always(1),
                can_indexed: true,
                cost:        10.0,
                selectivity: 1.0,
            },
            // lift_all_builtins (object): zero-arg one-to-one ops.
            // Cost ~1 (single IndexMap iteration).  Pure, stateless,
            // reorderable through other pure 1:1 stages.
            Stage::Keys | Stage::Values | Stage::Entries => StageShape {
                cardinality: Cardinality::OneToOne,
                order:       Order::Stateless,
                purity:      true,
                boundedness: Boundedness::Always(1),
                can_indexed: false,
                cost:        1.0,
                selectivity: 1.0,
            },
        }
    }

    // ── lift_all_builtins.md template — Tier 1 declarative optimisations ──────
    //
    // Three per-Stage methods every lifted built-in can override.  Default
    // impls return None / false so existing Stages compile unchanged; new
    // Stages opt in by adding match arms.  Wired into Phase 4 fold +
    // Phase 0 const-fold + a future cancels_with pass.

    /// Algebraic merge identity — `Stage::A ∘ Stage::B → Some(Stage::C)`.
    /// Returns `None` when no identity applies.  Examples:
    /// `Take(a) ∘ Take(b) → Take(min(a,b))`, `Reverse ∘ Reverse → Identity`
    /// (encoded by callers removing both stages).  Stays None for stages
    /// that need program-construction (Filter+Filter AndOp); those keep
    /// their dedicated rule in `rewrite_step` until Phase 4 grows a
    /// program-builder API.
    pub fn merge_with(&self, other: &Self) -> Option<Self> {
        match (self, other) {
            (Stage::Take(a), Stage::Take(b)) => Some(Stage::Take((*a).min(*b))),
            (Stage::Skip(a), Stage::Skip(b)) => Some(Stage::Skip((*a).saturating_add(*b))),
            // Sort / UniqueBy are idempotent — rightmost key wins.
            (Stage::Sort(_), Stage::Sort(b)) => Some(Stage::Sort(b.clone())),
            (Stage::UniqueBy(_), Stage::UniqueBy(b)) => Some(Stage::UniqueBy(b.clone())),
            _ => None,
        }
    }

    /// Inverse-pair cancellation — `Stage::A ∘ Stage::B = identity`.
    /// Both stages drop when this returns true.  Examples (post-lift):
    /// `Reverse ∘ Reverse`, `to_base64 ∘ from_base64`, `to_pairs ∘ from_pairs`.
    /// Today only Reverse self-cancels; remaining pairs land with their
    /// lifted Stage variants.
    pub fn cancels_with(&self, other: &Self) -> bool {
        matches!((self, other), (Stage::Reverse, Stage::Reverse))
    }

    /// Constant-folding hook — when source is a literal Val (or every
    /// upstream stage already constant-folded), each pure stage can
    /// pre-compute its output at lower-time.  Default returns None; lifted
    /// Stages with side-effect-free apply override.  Wired into a future
    /// Phase 0 const-fold pass; today returns None across the board so
    /// runtime semantics are preserved.
    pub fn eval_constant(&self, _v: &Val) -> Option<Val> {
        None
    }
}

/// Strategy = execution kernel selected by Phase 5.  Five kernels; every
/// chain in the language reduces to one of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    /// `(Map | identity)*` + positional sink → pull source[idx], run
    /// chain on one element, return.  O(1) instead of O(N).
    IndexedDispatch,
    /// Chain contains a Barrier (Sort/GroupBy/UniqueBy/Reverse) — must
    /// materialise into Vec, run barrier op, continue.
    BarrierMaterialise,
    /// Chain pulls until short-circuit (First/Any/All) — early exit.
    EarlyExit,
    /// Chain has a TakeWhile-like Done signal — pull until terminator.
    DoneTerminating,
    /// Generic streaming pull loop.
    PullLoop,
}

#[derive(Debug, Clone)]
pub struct Plan {
    pub stages:     Vec<Stage>,
    pub sink:       Sink,
    pub strategies: Vec<StageStrategy>,
    pub strategy:   Strategy,
}

/// Step 3d Phase 5 entry point — full planning over (stages, sink) → Plan.
/// Phase 1: demand propagation (already in `compute_strategies`).
/// Phase 2: dead pure-stage elimination (Sink::Count + pure Map drops).
/// Phase 3: commutative reorder by cost-weighted selectivity.
/// Phase 4: algebraic merging (Skip+Skip, Take+Take, Reverse∘Reverse, …).
/// Phase 5: pick Strategy from final shape.
pub fn plan(stages: Vec<Stage>, sink: Sink) -> Plan {
    plan_with_kernels(stages, &[], sink)
}

/// Variant of `plan` that consults parallel `BodyKernel` slice for
/// kernel-aware cost/selectivity.  Pipeline lowering carries kernels in
/// `Pipeline::stage_kernels`; the caller passes them through here so
/// Phase 3 reorder gets accurate heuristics.
pub fn plan_with_kernels(stages: Vec<Stage>, kernels: &[BodyKernel], sink: Sink) -> Plan {
    let mut stages = stages;
    let mut k_buf: Vec<BodyKernel> = if kernels.len() == stages.len() {
        kernels.to_vec()
    } else {
        // Synthesise Generic kernels when caller didn't supply.
        vec![BodyKernel::Generic; stages.len()]
    };

    // Phase 2 — dead pure-stage elimination.
    drop_dead_pure_stages_kernels(&mut stages, &mut k_buf, &sink);

    // Phase 3 — commutative reorder of adjacent Filter runs.
    reorder_filter_runs(&mut stages, &mut k_buf);

    // Phase 4 — algebraic merging.
    fold_merge_with_kernels(&mut stages, &mut k_buf);

    // Phase 1 — demand propagation (also picks Sort top-k strategy).
    let strategies = compute_strategies(&stages, &sink);

    // Phase 5 — strategy selection.
    let strategy = select_strategy(&stages, &sink);

    Plan { stages, sink, strategies, strategy }
}

/// Cost + selectivity from a recognised BodyKernel.  Generic falls back
/// to neutral defaults (cost=10, selectivity=0.5).  Used by Phase 3 to
/// rank Filter stages.
fn kernel_cost_selectivity(stage: &Stage, kernel: &BodyKernel) -> (f64, f64) {
    use crate::ast::BinOp;
    match (stage, kernel) {
        // Filter cmp-lit kernels — cost from N field hops, selectivity
        // from the operator (Eq/Neq are highly skewed; range ops ~50/50).
        (Stage::Filter(_), BodyKernel::FieldCmpLit(_, op, _)) => {
            let s = match op {
                BinOp::Eq                       => 0.10,
                BinOp::Neq                      => 0.90,
                BinOp::Lt | BinOp::Gt           => 0.40,
                BinOp::Lte | BinOp::Gte         => 0.50,
                _                               => 0.50,
            };
            (1.5, s)
        }
        (Stage::Filter(_), BodyKernel::FieldChainCmpLit(keys, op, _)) => {
            let s = match op {
                BinOp::Eq  => 0.10,
                BinOp::Neq => 0.90,
                BinOp::Lt | BinOp::Gt   => 0.40,
                BinOp::Lte | BinOp::Gte => 0.50,
                _ => 0.50,
            };
            (1.0 + keys.len() as f64, s)
        }
        (Stage::Filter(_), BodyKernel::CurrentCmpLit(op, _)) => {
            let s = match op {
                BinOp::Eq  => 0.10,
                BinOp::Neq => 0.90,
                BinOp::Lt | BinOp::Gt   => 0.40,
                BinOp::Lte | BinOp::Gte => 0.50,
                _ => 0.50,
            };
            (0.8, s)
        }
        (Stage::Filter(_), BodyKernel::FieldRead(_)) => (1.0, 0.7),
        (Stage::Filter(_), BodyKernel::ConstBool(b)) =>
            (0.1, if *b { 1.0 } else { 0.0 }),
        // Generic / unrecognised — fall back to Stage::shape() defaults.
        _ => {
            let sh = stage.shape();
            (sh.cost, sh.selectivity)
        }
    }
}

/// Phase 3: reorder adjacent runs of `Stage::Filter` by ascending
/// `cost / (1 - selectivity)` — cheaper, more-selective filters first
/// drop rows early.  Skips runs containing any filter with `Generic`
/// kernel (could be impure or expensive in unpredictable ways) or
/// non-Filter stages (Map/FlatMap mid-run blocks reorder).
///
/// Conservative: only reorder when *every* filter in the run has a
/// recognised kernel.  Single Filter or no run → no-op.
fn reorder_filter_runs(stages: &mut Vec<Stage>, kernels: &mut Vec<BodyKernel>) {
    let mut i = 0;
    while i < stages.len() {
        // Find next run of adjacent Filter stages with known kernels.
        let mut j = i;
        while j < stages.len()
            && matches!(stages[j], Stage::Filter(_))
            && !matches!(kernels.get(j), Some(BodyKernel::Generic) | None)
        {
            j += 1;
        }
        if j - i >= 2 {
            // Pair (Stage, Kernel) and rank.
            let mut run: Vec<(Stage, BodyKernel)> = Vec::with_capacity(j - i);
            for idx in i..j {
                run.push((stages[idx].clone(), kernels[idx].clone()));
            }
            run.sort_by(|a, b| {
                let (ca, sa) = kernel_cost_selectivity(&a.0, &a.1);
                let (cb, sb) = kernel_cost_selectivity(&b.0, &b.1);
                // Rank = cost / (1 - selectivity); smaller first.
                // Guard against selectivity=1.0 producing inf (constant
                // pass-through filter — already eliminated by Phase 2 if
                // pure, so unlikely; cap at large finite).
                let ra = ca / (1.0 - sa).max(1e-6);
                let rb = cb / (1.0 - sb).max(1e-6);
                ra.partial_cmp(&rb).unwrap_or(std::cmp::Ordering::Equal)
            });
            for (idx, (s, k)) in run.into_iter().enumerate() {
                stages[i + idx] = s;
                kernels[i + idx] = k;
            }
        }
        i = j.max(i + 1);
    }
}

/// Phase 2 with kernel-array shadowing — drops kernels in lockstep.
fn drop_dead_pure_stages_kernels(
    stages: &mut Vec<Stage>,
    kernels: &mut Vec<BodyKernel>,
    sink: &Sink,
) {
    if !matches!(sink, Sink::Count) { return; }
    while let Some(last) = stages.last() {
        let s = last.shape();
        if matches!(s.cardinality, Cardinality::OneToOne) && s.purity {
            stages.pop();
            kernels.pop();
        } else { break; }
    }
}

/// Phase 4 with kernel-array shadowing — drops/merges kernels in lockstep.
/// Algebraic identities + idempotence:
///   Reverse ∘ Reverse → identity
///   Skip(a) ∘ Skip(b) → Skip(a+b)
///   Take(a) ∘ Take(b) → Take(min(a,b))
///   Sort(_) ∘ Sort(k) → Sort(k)              (right wins)
///   UniqueBy(_) ∘ UniqueBy(k) → UniqueBy(k)  (right wins)
///   Filter(ConstBool(true))  → identity      (drop)
///   Filter(ConstBool(false)) → empty pipeline (handled by caller)
fn fold_merge_with_kernels(stages: &mut Vec<Stage>, kernels: &mut Vec<BodyKernel>) {
    // Pre-pass: drop Filter(ConstBool(true)) stages.  Filter(false) leaves
    // the stage in place; rewrite_step still handles short-circuit on
    // Filter(false) by clearing pipeline + swapping sink.
    let mut i = 0;
    while i < stages.len() {
        if matches!(&stages[i], Stage::Filter(_))
            && matches!(kernels.get(i), Some(BodyKernel::ConstBool(true)))
        {
            stages.remove(i);
            kernels.remove(i);
        } else {
            i += 1;
        }
    }

    // Adjacent-pair fold via Stage::cancels_with + Stage::merge_with.
    // Per-Stage declarative identities; new lifted Stages plug in by
    // adding match arms to those two methods — zero changes here.
    let mut i = 0;
    while i + 1 < stages.len() {
        if stages[i].cancels_with(&stages[i + 1]) {
            stages.drain(i..=i + 1);
            kernels.drain(i..=i + 1);
            if i > 0 { i -= 1; }
            continue;
        }
        if let Some(merged) = stages[i].merge_with(&stages[i + 1]) {
            stages[i] = merged;
            stages.remove(i + 1);
            kernels.remove(i + 1);
            continue;
        }
        i += 1;
    }
}

// `drop_dead_pure_stages` and `fold_merge_with` superseded by
// `drop_dead_pure_stages_kernels` and `fold_merge_with_kernels` —
// kernel-array-aware variants used by `plan_with_kernels`.  Kept
// the old names hidden behind `#[allow(dead_code)]` shims while
// migrating callers.

#[allow(dead_code)]
fn fold_merge_with(stages: &mut Vec<Stage>) {
    let mut i = 0;
    while i + 1 < stages.len() {
        let merged = match (&stages[i], &stages[i + 1]) {
            (Stage::Reverse, Stage::Reverse) => Some(None),
            (Stage::Skip(a), Stage::Skip(b)) =>
                Some(Some(Stage::Skip(a.saturating_add(*b)))),
            (Stage::Take(a), Stage::Take(b)) =>
                Some(Some(Stage::Take((*a).min(*b)))),
            _ => None,
        };
        if let Some(replacement) = merged {
            stages.remove(i + 1);
            match replacement {
                Some(s) => stages[i] = s,
                None    => { stages.remove(i); if i > 0 { i -= 1; } }
            }
            continue;
        }
        i += 1;
    }
}

/// Phase 5: pick Strategy.
pub fn select_strategy(stages: &[Stage], sink: &Sink) -> Strategy {
    let stages_can_indexed = stages.iter().all(|s| s.shape().can_indexed);
    let sink_positional    = sink.demand().positional.is_some();
    let has_barrier        = stages.iter().any(|s|
        matches!(s.shape().cardinality, Cardinality::Barrier));
    let has_short_circuit  = matches!(sink, Sink::First);

    if has_barrier { return Strategy::BarrierMaterialise; }
    if stages_can_indexed && sink_positional { return Strategy::IndexedDispatch; }
    if has_short_circuit { return Strategy::EarlyExit; }
    Strategy::PullLoop
}

#[derive(Debug, Clone)]
pub struct Pipeline {
    pub source: Source,
    pub stages: Vec<Stage>,
    pub sink:   Sink,
    /// Phase A3 — per-Stage kernel hint, in 1:1 correspondence with
    /// `stages`.  Computed once at lowering by `BodyKernel::classify`
    /// over each stage's sub-program.  Run loop dispatches the
    /// specialised inline path when the kernel is recognised, falls
    /// back to the generic `vm.exec_in_env` path for `Generic`.
    /// Empty when the lowering didn't populate it (legacy code paths).
    pub stage_kernels: Vec<BodyKernel>,
    /// Sink kernel hint — same idea for the terminal program (NumMap,
    /// CountIf, NumFilterMap, FilterFirst, etc.).  Empty `Vec` when
    /// the sink has no sub-program.
    pub sink_kernels:  Vec<BodyKernel>,
}

// ── Lowering ─────────────────────────────────────────────────────────────────

impl Pipeline {
    /// Try to lower an `Expr` into a Pipeline.  Returns `None` for any
    /// shape this Phase 1 substrate doesn't yet handle — caller falls
    /// back to the existing opcode compilation path.
    ///
    /// Supported (Phase 1):
    ///   - `$.k1.k2…kN.<stage>*.<sink>` where each `kN` is a plain Field,
    ///     stages are zero-or-more of `filter` / `map` / `take` / `skip`,
    ///     and the sink is `count` / `len` / `sum` / nothing (Collect).
    ///
    /// Not yet supported and returns `None`:
    ///   - Any non-Root base
    ///   - Any non-Field step before the first method (e.g. `[idx]`)
    ///   - Lambda methods (`map(@.x + 1)` is fine; `map(lambda x: …)` is not)
    ///   - Any unrecognised method in stage position
    pub fn lower(expr: &Expr) -> Option<Pipeline> {
        let p = Self::lower_with_reason(expr);
        if trace_enabled() {
            match &p {
                Ok(pipe) => eprintln!(
                    "[pipeline] activated: stages={} sink={} src={}",
                    pipe.stages.len(),
                    sink_name(&pipe.sink),
                    source_name(&pipe.source),
                ),
                Err(reason) => eprintln!(
                    "[pipeline] fallback: ({}) at {}",
                    reason,
                    expr_label(expr),
                ),
            }
        }
        p.ok()
    }

    fn lower_with_reason(expr: &Expr) -> std::result::Result<Pipeline, &'static str> {
        Self::lower_inner(expr).ok_or("shape not yet supported")
    }

    fn lower_inner(expr: &Expr) -> Option<Pipeline> {
        use crate::ast::Step;
        let (base, steps) = match expr {
            Expr::Chain(b, s) => (b.as_ref(), s.as_slice()),
            _ => return None,
        };
        if !matches!(base, Expr::Root) { return None; }

        // Find where the field-chain prefix ends and stages begin.
        let mut field_end = 0;
        for s in steps {
            match s {
                Step::Field(_) => field_end += 1,
                _ => break,
            }
        }
        // Phase 1 deliberately does not lower bare `$.<method>` shapes
        // (no field-chain prefix) because the existing fused opcodes
        // (MapSplitLenSum, FilterFieldCmpLitMapField, etc.) often beat
        // a generic pull-based pipeline for those.  Field-chain prefix
        // signals a "scan over a sub-array" intent — the pipeline's
        // sweet spot.
        if field_end == 0 { return None; }

        let keys: Arc<[Arc<str>]> = steps[..field_end].iter()
            .map(|s| match s { Step::Field(k) => Arc::<str>::from(k.as_str()), _ => unreachable!() })
            .collect::<Vec<_>>().into();

        // Decode the trailing methods into stages + a sink.
        // Compile each filter / map sub-Expr to a Program once so
        // Pipeline::run can reuse it per row.  Sub-programs run against
        // the current item bound as the VM's root, so `@.field` and
        // `@` references resolve to the row.
        let trailing = &steps[field_end..];
        let (stages, sink) = decode_method_chain(trailing)?;

        let mut p = Pipeline {
            source: Source::FieldChain { keys },
            stages, sink,
            stage_kernels: Vec::new(),
            sink_kernels:  Vec::new(),
        };
        rewrite(&mut p);
        // Phase A3 — classify per-stage sub-programs.  Per-row pull loop
        // reads these hints to choose a specialised inline path vs the
        // generic vm.exec fallback.  Also drives Step 3d Phase 3 reorder.
        let classify_kernels = |stages: &[Stage]| -> Vec<BodyKernel> {
            stages.iter().map(|s| match s {
                Stage::Filter(p)        => BodyKernel::classify(p),
                Stage::Map(p)           => BodyKernel::classify(p),
                Stage::FlatMap(p)       => BodyKernel::classify(p),
                Stage::UniqueBy(Some(p))=> BodyKernel::classify(p),
                Stage::GroupBy(p)       => BodyKernel::classify(p),
                Stage::Sort(Some(p))    => BodyKernel::classify(p),
                _                       => BodyKernel::Generic,
            }).collect()
        };
        let kernels = classify_kernels(&p.stages);

        // Step 3d planning — Phase 2/3/4 transforms with kernel-aware
        // cost/selectivity.  Result.stages / Result.sink replace the
        // pre-plan values.  Phase 1 + Phase 5 (demand prop, strategy
        // selection) re-runs at exec time on the final shape.
        let plan_result = plan_with_kernels(p.stages.clone(), &kernels, p.sink.clone());
        p.stages = plan_result.stages;
        p.sink   = plan_result.sink;

        // Re-classify post-plan since Phase 4 merges may have produced
        // new sub-programs (e.g. Map+Map → field-chain-Map).
        p.stage_kernels = classify_kernels(&p.stages);
        // No fused-Sink kernels — base sinks have no sub-programs.
        p.sink_kernels = Vec::new();
        Some(p)
    }
}

/// Step 3d-extension (A2): try to decode the body of a Map(...) call as
/// its own pipeline Plan.  Body must be `Expr::Chain(Expr::Current, [...])`
/// with at least one trailing method or field access.  Field-chain
/// prefix becomes a leading `Stage::Map(field-walk)`; trailing methods
/// decode via the shared `decode_method_chain` helper.  Inner Plan runs
/// `plan_with_kernels` so it picks IndexedDispatch / BarrierMaterialise /
/// EarlyExit / DoneTerminating / PullLoop strategies recursively.
///
/// Returns None when the body is opaque (lambda, custom method, side
/// effect) — caller falls back to `Stage::Map(opaque_program)`.
fn try_decode_map_body(arg: &crate::ast::Arg) -> Option<Plan> {
    use crate::ast::{Arg, Step};
    let expr = match arg { Arg::Pos(e) => e, _ => return None };
    let (base, steps) = match expr {
        Expr::Chain(b, s) => (b.as_ref(), s.as_slice()),
        _ => return None,
    };
    if !matches!(base, Expr::Current) { return None; }

    // Field-chain prefix walks @.a.b.c → seed.a.b.c.  Encoded as a
    // leading Map stage whose body is the FieldChain program over @.
    let mut field_end = 0;
    for s in steps {
        match s { Step::Field(_) => field_end += 1, _ => break }
    }
    let trailing = &steps[field_end..];
    // Require at least one trailing method — pure field access alone
    // is what plain Stage::Map(@.a.b.c) already handles.
    if trailing.is_empty() { return None; }

    let mut stages: Vec<Stage> = Vec::new();
    if field_end > 0 {
        // Build a sub-program `[PushCurrent, FieldChain([...])]` that
        // walks the prefix from the seed.
        let keys: Arc<[Arc<str>]> = steps[..field_end].iter()
            .map(|s| match s { Step::Field(k) => Arc::<str>::from(k.as_str()), _ => unreachable!() })
            .collect::<Vec<_>>().into();
        let n_keys = keys.len();
        let fcd = Arc::new(crate::vm::FieldChainData {
            keys,
            ics: (0..n_keys).map(|_| std::sync::atomic::AtomicU64::new(0))
                .collect::<Vec<_>>().into_boxed_slice(),
        });
        let ops = vec![crate::vm::Opcode::PushCurrent, crate::vm::Opcode::FieldChain(fcd)];
        let prog = Arc::new(crate::vm::Program::new(ops, "<compiled-map-prefix>"));
        stages.push(Stage::Map(prog));
    }
    let (mut more_stages, sink) = decode_method_chain(trailing)?;
    stages.append(&mut more_stages);

    // Run the same planning the outer pipeline does.  Kernel
    // classification feeds Phase 3 reorder + Phase 5 strategy select.
    let kernels: Vec<BodyKernel> = stages.iter().map(|s| match s {
        Stage::Filter(p)         => BodyKernel::classify(p),
        Stage::Map(p)            => BodyKernel::classify(p),
        Stage::FlatMap(p)        => BodyKernel::classify(p),
        Stage::UniqueBy(Some(p)) => BodyKernel::classify(p),
        Stage::GroupBy(p)        => BodyKernel::classify(p),
        Stage::Sort(Some(p))     => BodyKernel::classify(p),
        _                        => BodyKernel::Generic,
    }).collect();
    Some(plan_with_kernels(stages, &kernels, sink))
}

/// Run a `Plan` against a single seed Val (Step 3d-extension A2).  Used
/// by `Stage::CompiledMap` per outer element.  Wraps seed as a single-
/// element Val::Arr and runs through a synth Pipeline so all strategy
/// selection (IndexedDispatch / EarlyExit / etc.) applies recursively.
fn run_compiled_map(plan: &Plan, seed: Val) -> Result<Val, EvalError> {
    let synth = Pipeline {
        source: Source::Receiver(Val::arr(vec![seed])),
        stages: plan.stages.clone(),
        sink:   plan.sink.clone(),
        stage_kernels: Vec::new(),
        sink_kernels:  Vec::new(),
    };
    synth.run(&Val::Null)
}

/// Decode a slice of `Step::Method(...)` into pipeline `(stages, sink)`.
/// Shared by top-level `lower_inner` and Step 3d-extension (A2)
/// recursive sub-pipeline planning for `Map(@.<chain>)` bodies.
/// Returns `None` if any method shape isn't recognised.
fn decode_method_chain(trailing: &[crate::ast::Step]) -> Option<(Vec<Stage>, Sink)> {
    use crate::ast::{Step, Arg};
    let mut stages: Vec<Stage> = Vec::new();
    let mut sink: Sink = Sink::Collect;
    for (i, s) in trailing.iter().enumerate() {
        let is_last = i == trailing.len() - 1;
        match s {
            Step::Method(name, args) => {
                match (name.as_str(), args.len(), is_last) {
                    ("filter", 1, _) => stages.push(Stage::Filter(compile_subexpr(&args[0])?)),
                    ("map", 1, _) => {
                        // A2: try recursive sub-pipeline planning first.
                        // Body shapes that decode as a chain of recognised
                        // methods over @ become Stage::CompiledMap; opaque
                        // bodies (lambdas, custom methods) fall through.
                        match try_decode_map_body(&args[0]) {
                            Some(plan) => stages.push(Stage::CompiledMap(Arc::new(plan))),
                            None       => stages.push(Stage::Map(compile_subexpr(&args[0])?)),
                        }
                    }
                    ("flat_map", 1, _) => stages.push(Stage::FlatMap(compile_subexpr(&args[0])?)),
                    ("reverse", 0, _) => stages.push(Stage::Reverse),
                    ("unique", 0, _)  => stages.push(Stage::UniqueBy(None)),
                    ("unique_by", 1, _) => stages.push(Stage::UniqueBy(Some(compile_subexpr(&args[0])?))),
                    ("group_by", 1, _)  => stages.push(Stage::GroupBy(compile_subexpr(&args[0])?)),
                    ("sort", 0, _)      => stages.push(Stage::Sort(None)),
                    ("sort_by", 1, _)   => stages.push(Stage::Sort(Some(compile_subexpr(&args[0])?))),
                    ("take", 1, _) => {
                        let n = match &args[0] {
                            Arg::Pos(Expr::Int(n)) if *n >= 0 => *n as usize,
                            _ => return None,
                        };
                        stages.push(Stage::Take(n));
                    }
                    ("skip", 1, _) => {
                        let n = match &args[0] {
                            Arg::Pos(Expr::Int(n)) if *n >= 0 => *n as usize,
                            _ => return None,
                        };
                        stages.push(Stage::Skip(n));
                    }
                    ("split", 1, _) => {
                        let sep = match &args[0] {
                            Arg::Pos(Expr::Str(s)) => Arc::<str>::from(s.as_str()),
                            _ => return None,
                        };
                        stages.push(Stage::Split(sep));
                    }
                    // lift_all_builtins (object family — zero-arg ops)
                    ("keys",    0, _) => stages.push(Stage::Keys),
                    ("values",  0, _) => stages.push(Stage::Values),
                    ("entries", 0, _) => stages.push(Stage::Entries),
                    ("slice", 1, _) => {
                        let start = match &args[0] {
                            Arg::Pos(Expr::Int(n)) => *n,
                            _ => return None,
                        };
                        stages.push(Stage::Slice(start, None));
                    }
                    ("slice", 2, _) => {
                        let (start, end) = match (&args[0], &args[1]) {
                            (Arg::Pos(Expr::Int(s)), Arg::Pos(Expr::Int(e))) => (*s, Some(*e)),
                            _ => return None,
                        };
                        stages.push(Stage::Slice(start, end));
                    }
                    ("replace", 2, _) | ("replace_all", 2, _) => {
                        let (needle, replacement) = match (&args[0], &args[1]) {
                            (Arg::Pos(Expr::Str(n)), Arg::Pos(Expr::Str(r))) =>
                                (Arc::<str>::from(n.as_str()), Arc::<str>::from(r.as_str())),
                            _ => return None,
                        };
                        stages.push(Stage::Replace {
                            needle, replacement, all: name.as_str() == "replace_all",
                        });
                    }
                    ("chunk", 1, _) | ("batch", 1, _) => {
                        let n = match &args[0] {
                            Arg::Pos(Expr::Int(n)) if *n >= 1 => *n as usize,
                            _ => return None,
                        };
                        stages.push(Stage::Chunk(n));
                    }
                    ("window", 1, _) => {
                        let n = match &args[0] {
                            Arg::Pos(Expr::Int(n)) if *n >= 1 => *n as usize,
                            _ => return None,
                        };
                        stages.push(Stage::Window(n));
                    }
                    ("count", 0, true) | ("len", 0, true) => sink = Sink::Count,
                    ("sum", 0, true) => sink = Sink::Numeric(NumOp::Sum),
                    ("min", 0, true) => sink = Sink::Numeric(NumOp::Min),
                    ("max", 0, true) => sink = Sink::Numeric(NumOp::Max),
                    ("avg", 0, true) => sink = Sink::Numeric(NumOp::Avg),
                    ("first", 0, true) => sink = Sink::First,
                    ("last", 0, true)  => sink = Sink::Last,
                    _ => return None,
                }
            }
            _ => return None,
        }
    }
    Some((stages, sink))
}

/// Apply algebraic rewrite rules until fixed point or a fuel limit
/// expires.  Rules listed in `project_pipeline_ir.md`; this function
/// implements the subset that operates on the currently-supported
/// Stage / Sink set (`Filter` / `Map` / `Take` / `Skip` /
/// `Collect` / `Count` / `Sum` and the fused sinks).
///
/// Rules NOT yet implemented (require Stage variants the lowering
/// + execution don't yet emit):
///   `Sort ∘ Take → TopK`            (no Sort stage)
///   `Reverse ∘ Reverse → id`        (no Reverse stage)
///   `UniqueBy(k) ∘ UniqueBy(k) → UniqueBy(k)`
///   `Pick(a) ∘ Pick(b) → Pick(a ∩ b)`
///   `DeepScan(k) ∘ Filter(p on k) → DeepScanFiltered(k, p)`
///   `DeepScan(k) ∘ Pick({k}) → DeepScan(k)`
///   `Filter(p) ∘ Map(f) → Map(f) ∘ Filter(p ∘ f⁻¹)`  (needs SSA dep)
///
/// Each rule is monotonic — strictly reduces stage count or rewrites
/// to a structurally smaller form — so a fuel limit of 16 protects
/// against pathological loops without affecting correctness.
fn rewrite(p: &mut Pipeline) {
    let mut fuel = 16usize;
    while fuel > 0 {
        fuel -= 1;
        if rewrite_step(p) { continue; }
        break;
    }
}

/// One round of the rewrite loop.  Returns `true` if any rule fired,
/// `false` if the pipeline is at a local fixed point.
///
/// Most algebraic rules deleted in the Step 1+2 mop-up — Step 3d
/// `plan_with_kernels()` (called from `lower_with_reason`) now subsumes:
///   - Filter(true) → identity                     (Phase 4 const-fold)
///   - Skip+Skip / Take+Take / Reverse∘Reverse     (Phase 4)
///   - Sort∘Sort / UniqueBy∘UniqueBy idempotence   (Phase 4)
///   - Map∘Count drop                              (Phase 2)
///   - Filter run reorder by selectivity           (Phase 3)
///
/// What remains here:
///   - Filter(false) → empty pipeline.  Phase 4 doesn't have access to
///     the Sink to swap to its identity-element form, so this stays.
///   - Map(f) ∘ Filter(g) ∘ Filter(h) etc.: the kernel-aware Filter+
///     Filter merge via `Opcode::AndOp` and the Map+Map field-chain
///     fusion — both build new `vm::Program`s, which the Phase 4
///     stage-only API can't do without a wider rewrite.  Kept here.
///   - Map ∘ Take pushdown: still strictly correct + a perf win, kept.
fn rewrite_step(p: &mut Pipeline) -> bool {
    use crate::vm::Opcode;

    // Filter(false) → empty pipeline.  Filter(true) handled by Phase 4.
    let mut const_false_at: Option<usize> = None;
    for (i, s) in p.stages.iter().enumerate() {
        if let Stage::Filter(prog) = s {
            if let Some(false) = prog_const_bool(prog) {
                const_false_at = Some(i);
                break;
            }
        }
    }
    if const_false_at.is_some() {
        // Empty input: existing accumulators in run() yield Int(0) /
        // Float(0.0) / Val::arr([]) — clearing stages suffices.
        p.stages.clear();
        return true;
    }

    // Filter+Filter → AndOp-merged Filter (kernel-aware vm::Program build).
    // Map+Map → FieldChain Map (kernel-aware vm::Program build).
    // Both construct new programs — keep here until Phase 4 gains a
    // program-building API.
    for i in 0..p.stages.len().saturating_sub(1) {
        match (&p.stages[i], &p.stages[i + 1]) {
            (Stage::Map(a_prog), Stage::Map(b_prog)) => {
                let ka = BodyKernel::classify(a_prog);
                let kb = BodyKernel::classify(b_prog);
                let chain: Option<Vec<Arc<str>>> = match (&ka, &kb) {
                    (BodyKernel::FieldRead(a), BodyKernel::FieldRead(b)) =>
                        Some(vec![a.clone(), b.clone()]),
                    (BodyKernel::FieldRead(a), BodyKernel::FieldChain(bs)) => {
                        let mut v = vec![a.clone()];
                        v.extend(bs.iter().cloned());
                        Some(v)
                    }
                    (BodyKernel::FieldChain(as_), BodyKernel::FieldRead(b)) => {
                        let mut v: Vec<Arc<str>> = as_.iter().cloned().collect();
                        v.push(b.clone());
                        Some(v)
                    }
                    (BodyKernel::FieldChain(as_), BodyKernel::FieldChain(bs)) => {
                        let mut v: Vec<Arc<str>> = as_.iter().cloned().collect();
                        v.extend(bs.iter().cloned());
                        Some(v)
                    }
                    _ => None,
                };
                if let Some(keys) = chain {
                    let fcd = Arc::new(crate::vm::FieldChainData {
                        keys: keys.into(),
                        ics: (0..0).map(|_|
                            std::sync::atomic::AtomicU64::new(0)
                        ).collect::<Vec<_>>().into_boxed_slice(),
                    });
                    let new_ops = vec![Opcode::PushCurrent, Opcode::FieldChain(fcd)];
                    let merged = Arc::new(crate::vm::Program::new(new_ops, "<map-fused>"));
                    p.stages[i] = Stage::Map(merged);
                    p.stages.remove(i + 1);
                    return true;
                }
            }
            (Stage::Filter(p_prog), Stage::Filter(q_prog)) => {
                let mut ops: Vec<Opcode> = p_prog.ops.as_ref().to_vec();
                ops.push(Opcode::AndOp(Arc::clone(q_prog)));
                let merged = Arc::new(crate::vm::Program {
                    ops: ops.into(),
                    source: p_prog.source.clone(),
                    id: 0,
                    is_structural: false,
                    ics: p_prog.ics.clone(),
                });
                p.stages[i] = Stage::Filter(merged);
                p.stages.remove(i + 1);
                return true;
            }
            _ => {}
        }
    }

    // Pushdown: Map(f) ∘ Take(n) → Take(n) ∘ Map(f).
    // Strict perf win — composed exec runs map only n times.
    for i in 0..p.stages.len().saturating_sub(1) {
        if matches!(&p.stages[i], Stage::Map(_))
            && matches!(&p.stages[i + 1], Stage::Take(_)) {
            p.stages.swap(i, i + 1);
            return true;
        }
    }

    false
}

/// If `prog` evaluates to a constant boolean independent of `@`,
/// return the literal value.  Currently matches the trivial shape
/// `[PushBool(b)]`; future SSA work could constant-fold larger preds.
fn prog_const_bool(prog: &crate::vm::Program) -> Option<bool> {
    use crate::vm::Opcode;
    let ops = prog.ops.as_ref();
    if ops.len() != 1 { return None; }
    match &ops[0] {
        Opcode::PushBool(b) => Some(*b),
        _ => None,
    }
}

// ── Execution ────────────────────────────────────────────────────────────────

impl Pipeline {
    /// Phase 3 columnar fast path.  Detects pipelines whose source is
    /// an array of objects + zero stages + a single-field `SumMap` /
    /// `CountIf` / `SumFilterMap` sink.  Extracts the projected column
    /// into a flat `Vec<i64>` / `Vec<f64>` once, then folds the whole
    /// slice via the autovec'd reductions in vm.rs.
    ///
    /// Returns `None` if the shape doesn't match the columnar fast
    /// path; caller falls back to the per-row pull loop.
    /// Phase A2 stage-chain columnar fast path:
    ///   `Stage::Filter(FieldCmpLit) ∘ Stage::Map(FieldRead) ∘ Sink::Collect`
    ///   `Stage::Map(FieldRead) ∘ Sink::Collect`
    ///   `Stage::Filter(FieldCmpLit) ∘ Sink::Count`  (already covered by CountIf rule)
    ///   `Stage::Filter(FieldCmpLit) ∘ Sink::Numeric(...)` (already covered)
    /// Walks the column without entering vm.exec per row.
    /// Same as [`try_columnar_stage_chain`] but consults the optional
    /// promoter to upgrade Val::Arr → Val::ObjVec; lets the typed
    /// stage-chain path (filter+map on ObjVec typed columns) fire.
    fn try_columnar_stage_chain_with(
        &self,
        root: &Val,
        cache: Option<&dyn ObjVecPromoter>,
    ) -> Option<Result<Val, EvalError>> {
        let recv = match &self.source {
            Source::Receiver(v) => v.clone(),
            Source::FieldChain { keys } => walk_field_chain(root, keys),
        };
        // Promote to ObjVec via cache when available.  Unlocks the
        // typed-column stage-chain path below.
        let recv = if let (Some(c), Val::Arr(a)) = (cache, &recv) {
            if let Some(d) = c.promote(a) { Val::ObjVec(d) } else { recv }
        } else { recv };

        // Typed-column ObjVec group_by: Stage::GroupBy(FieldRead) ∘
        // Sink::Collect over Strs / Ints / Floats / Bools key column.
        // Walks the typed key column directly, partitions row indices
        // per key, materialises Val::Obj { key → Vec<row> }.
        if let (Some([Stage::GroupBy(_)]),
                Some([BodyKernel::FieldRead(key)]),
                Val::ObjVec(d),
                Sink::Collect) =
            (self.stages.get(..), self.stage_kernels.get(..), &recv, &self.sink)
        {
            if let Some(out) = objvec_typed_group_by(d, key) {
                return Some(Ok(out));
            }
        }

        // Typed-column ObjVec stage-chain: Filter(FieldCmpLit) ∘
        // Map(FieldRead) ∘ Collect → primitive mask + typed gather.
        if !matches!(self.sink, Sink::Collect) { return None; }
        if let (Some([Stage::Filter(_), Stage::Map(_)]),
                Some([BodyKernel::FieldCmpLit(pk, pop, plit),
                      BodyKernel::FieldRead(mk)]),
                Val::ObjVec(d)) =
            (self.stages.get(..), self.stage_kernels.get(..), &recv)
        {
            return objvec_typed_filter_map_collect(d, pk, *pop, plit, mk);
        }
        None
    }

    fn try_columnar_stage_chain(&self, root: &Val) -> Option<Result<Val, EvalError>> {
        // Resolve receiver.
        let recv = match &self.source {
            Source::Receiver(v) => v.clone(),
            Source::FieldChain { keys } => walk_field_chain(root, keys),
        };

        // Phase B1 — typed-lane filter chain on IntVec / FloatVec /
        // StrVec receivers.  Stage::Filter(CurrentCmpLit) over a
        // primitive vec → walk slice directly, build typed output.
        // Sinks: Collect / Count.
        if let [Stage::Filter(_)] = self.stages.as_slice() {
            if let [BodyKernel::CurrentCmpLit(op, lit)] = self.stage_kernels.as_slice() {
                match (&recv, &self.sink) {
                    (Val::IntVec(a), Sink::Collect) => {
                        let mut out: Vec<i64> = Vec::with_capacity(a.len());
                        for n in a.iter() {
                            let v = Val::Int(*n);
                            if eval_cmp_op(&v, *op, lit) { out.push(*n); }
                        }
                        return Some(Ok(Val::int_vec(out)));
                    }
                    (Val::IntVec(a), Sink::Count) => {
                        let mut c = 0i64;
                        for n in a.iter() {
                            let v = Val::Int(*n);
                            if eval_cmp_op(&v, *op, lit) { c += 1; }
                        }
                        return Some(Ok(Val::Int(c)));
                    }
                    (Val::FloatVec(a), Sink::Collect) => {
                        let mut out: Vec<f64> = Vec::with_capacity(a.len());
                        for f in a.iter() {
                            let v = Val::Float(*f);
                            if eval_cmp_op(&v, *op, lit) { out.push(*f); }
                        }
                        return Some(Ok(Val::float_vec(out)));
                    }
                    (Val::FloatVec(a), Sink::Count) => {
                        let mut c = 0i64;
                        for f in a.iter() {
                            let v = Val::Float(*f);
                            if eval_cmp_op(&v, *op, lit) { c += 1; }
                        }
                        return Some(Ok(Val::Int(c)));
                    }
                    _ => {}
                }
            }
        }

        if !matches!(self.sink, Sink::Collect) { return None; }
        let arr = match &recv { Val::Arr(a) => Arc::clone(a), _ => return None };

        // Build a (kernel, prog) view of the stages.
        let stages = &self.stages;
        let kernels = &self.stage_kernels;
        if stages.len() != kernels.len() { return None; }

        match (stages.as_slice(), kernels.as_slice()) {
            // Single Map(FieldRead) → Collect: direct projection.
            ([Stage::Map(_)], [BodyKernel::FieldRead(k)]) => {
                let mut out = Vec::with_capacity(arr.len());
                for v in arr.iter() {
                    out.push(v.get_field(k.as_ref()));
                }
                Some(Ok(Val::arr(out)))
            }
            // Single Filter(FieldCmpLit) → Collect: predicate mask copy.
            // Phase C1 — when Arc is uniquely held (refcount 1), take
            // ownership and retain-in-place; saves N Val clones.
            ([Stage::Filter(_)], [BodyKernel::FieldCmpLit(k, op, lit)]) => {
                match Arc::try_unwrap(arr) {
                    Ok(mut owned) => {
                        owned.retain(|v| {
                            let lhs = v.get_field(k.as_ref());
                            eval_cmp_op(&lhs, *op, lit)
                        });
                        Some(Ok(Val::arr(owned)))
                    }
                    Err(arr) => {
                        let mut out = Vec::with_capacity(arr.len());
                        for v in arr.iter() {
                            let lhs = v.get_field(k.as_ref());
                            if eval_cmp_op(&lhs, *op, lit) { out.push(v.clone()); }
                        }
                        Some(Ok(Val::arr(out)))
                    }
                }
            }
            // Filter(FieldCmpLit) ∘ Map(FieldRead) → Collect: project filtered column.
            ([Stage::Filter(_), Stage::Map(_)],
             [BodyKernel::FieldCmpLit(pk, pop, plit), BodyKernel::FieldRead(mk)]) => {
                let mut out = Vec::with_capacity(arr.len());
                for v in arr.iter() {
                    let lhs = v.get_field(pk.as_ref());
                    if eval_cmp_op(&lhs, *pop, plit) {
                        out.push(v.get_field(mk.as_ref()));
                    }
                }
                Some(Ok(Val::arr(out)))
            }
            // Single Map(FieldChain) → Collect: walk chain per item.
            // IC-cached probe: first item resolves each chain step via
            // IndexMap.get_full (returns slot index); subsequent items
            // try the cached slot first, fall back to hash on miss.
            // Saves ~half the probe cost on uniform-shape arrays.
            ([Stage::Map(_)], [BodyKernel::FieldChain(ks)]) => {
                let mut out = Vec::with_capacity(arr.len());
                let mut slots: Vec<Option<usize>> = vec![None; ks.len()];
                for v in arr.iter() {
                    let mut cur = v.clone();
                    for (i, k) in ks.iter().enumerate() {
                        cur = chain_step_ic(&cur, k.as_ref(), &mut slots[i]);
                        if matches!(cur, Val::Null) { break; }
                    }
                    out.push(cur);
                }
                Some(Ok(Val::arr(out)))
            }
            // Filter(FieldCmpLit) ∘ Map(FieldChain) → Collect.
            ([Stage::Filter(_), Stage::Map(_)],
             [BodyKernel::FieldCmpLit(pk, pop, plit), BodyKernel::FieldChain(mks)]) => {
                let mut out = Vec::with_capacity(arr.len());
                for v in arr.iter() {
                    let lhs = v.get_field(pk.as_ref());
                    if eval_cmp_op(&lhs, *pop, plit) {
                        let mut cur = v.clone();
                        for k in mks.iter() {
                            cur = cur.get_field(k.as_ref());
                            if matches!(cur, Val::Null) { break; }
                        }
                        out.push(cur);
                    }
                }
                Some(Ok(Val::arr(out)))
            }
            // Filter(FieldChainCmpLit) ∘ Map(FieldRead) → Collect.
            ([Stage::Filter(_), Stage::Map(_)],
             [BodyKernel::FieldChainCmpLit(pks, pop, plit), BodyKernel::FieldRead(mk)]) => {
                let mut out = Vec::with_capacity(arr.len());
                for v in arr.iter() {
                    let mut lhs = v.clone();
                    for k in pks.iter() {
                        lhs = lhs.get_field(k.as_ref());
                        if matches!(lhs, Val::Null) { break; }
                    }
                    if eval_cmp_op(&lhs, *pop, plit) {
                        out.push(v.get_field(mk.as_ref()));
                    }
                }
                Some(Ok(Val::arr(out)))
            }
            _ => None,
        }
    }

    /// Phase 7-lite — speculative ObjVec promotion at pipeline source.
    /// When the resolved receiver is `Val::Arr<Val::Obj>` with uniform
    /// shape (all rows have the same key set, in the same order), build
    /// a columnar `ObjVecData` on the fly so downstream slot-indexed
    /// kernels (objvec_*_slot) can fire.  No schema needed; just probe
    /// the first row + verify subsequent rows match.  Bails on shape
    /// mismatch, returns the original Arr unchanged.
    ///
    /// Cost: O(N × K) where N=row count, K=key count, dominated by
    /// pointer-equality on Arc<str> keys.  Win: subsequent operations
    /// run as O(N) slice walks instead of O(N) IndexMap probes.
    /// Wrapper that returns the promoted `ObjVecData` directly (for the
    /// memoised cache in `Jetro::get_or_promote_objvec`).
    pub fn try_promote_objvec_arr(arr: &Arc<Vec<Val>>) -> Option<Arc<crate::eval::value::ObjVecData>> {
        if let Some(Val::ObjVec(d)) = Self::try_promote_objvec(arr) {
            Some(d)
        } else { None }
    }

    fn try_promote_objvec(arr: &Arc<Vec<Val>>) -> Option<Val> {
        if arr.is_empty() { return None; }
        let first = match &arr[0] {
            Val::Obj(m) => m,
            _ => return None,
        };
        let keys: Vec<Arc<str>> = first.keys().cloned().collect();
        if keys.is_empty() { return None; }
        let stride = keys.len();
        let mut cells: Vec<Val> = Vec::with_capacity(arr.len() * stride);
        for v in arr.iter() {
            let m = match v {
                Val::Obj(m) => m,
                _ => return None,
            };
            if m.len() != stride { return None; }
            for (i, k) in keys.iter().enumerate() {
                // Pointer-equal Arc check first (cheap path); fall back
                // to hash lookup if Arc identity differs across rows.
                let val = match m.get_index(i) {
                    Some((k2, v)) if Arc::ptr_eq(k2, k) => v.clone(),
                    _ => match m.get(k.as_ref()) {
                        Some(v) => v.clone(),
                        None => return None,
                    },
                };
                cells.push(val);
            }
        }
        // Build typed-column mirror at promotion time.  Costs O(N×K)
        // already spent walking cells; per-slot type lock determined
        // by inspecting the first row + verifying subsequent rows
        // match.  Uniform-type columns light up the typed-fast-path
        // in slot kernels (closes the boxed-Val tag-check tax).
        let stride = keys.len();
        let nrows = if stride == 0 { 0 } else { cells.len() / stride };
        let typed_cols = build_typed_cols(&cells, stride, nrows);

        Some(Val::ObjVec(Arc::new(crate::eval::value::ObjVecData {
            keys: keys.into(),
            cells,
            typed_cols: Some(Arc::new(typed_cols)),
        })))
    }


    /// Cache-aware variant: when cache promotes the source array,
    /// recv is replaced with `Val::ObjVec` and the slot kernels fire.
    fn try_columnar_with(
        &self,
        root: &Val,
        cache: Option<&dyn ObjVecPromoter>,
    ) -> Option<Result<Val, EvalError>> {
        // Tape route is now reached via `try_run_no_root` ->
        // `try_run_composed_tape`. Legacy `try_tape_aggregate` removed
        // in Tier 3 sweep — composed-tape covers all cold-bench
        // shapes; barrier-bearing tape shapes degrade to Val path.
        // Typed ObjVec stage-chain path next (uses cache).
        if let Some(out) = self.try_columnar_stage_chain_with(root, cache) {
            return Some(out);
        }
        if let Some(out) = self.try_columnar_stage_chain(root) { return Some(out); }

        let recv = match &self.source {
            Source::Receiver(v) => v.clone(),
            Source::FieldChain { keys } => walk_field_chain(root, keys),
        };

        // Cache-driven ObjVec promotion: only if cache provided AND
        // recv is Val::Arr.  Promoted ObjVec memoised across calls.
        let recv = if let (Some(c), Val::Arr(a)) = (cache, &recv) {
            if let Some(d) = c.promote(a) {
                Val::ObjVec(d)
            } else { recv }
        } else { recv };

        // Typed primitive lane fast paths — only when the original
        // pipeline has zero stages (bare aggregate over a primitive
        // vec). Stage'd shapes go through the slot-kernel block below.
        if self.stages.is_empty() {
            match (&recv, &self.sink) {
                (Val::IntVec(a), Sink::Numeric(NumOp::Sum)) =>
                    return Some(Ok(Val::Int(a.iter().sum()))),
                (Val::IntVec(a), Sink::Numeric(NumOp::Min)) =>
                    return Some(Ok(a.iter().copied().min().map(Val::Int).unwrap_or(Val::Null))),
                (Val::IntVec(a), Sink::Numeric(NumOp::Max)) =>
                    return Some(Ok(a.iter().copied().max().map(Val::Int).unwrap_or(Val::Null))),
                (Val::IntVec(a), Sink::Count) =>
                    return Some(Ok(Val::Int(a.len() as i64))),
                (Val::FloatVec(a), Sink::Numeric(NumOp::Sum)) =>
                    return Some(Ok(Val::Float(a.iter().sum()))),
                (Val::FloatVec(a), Sink::Count) =>
                    return Some(Ok(Val::Int(a.len() as i64))),
                (Val::StrVec(a), Sink::Count) =>
                    return Some(Ok(Val::Int(a.len() as i64))),
                _ => {}
            }
        }
        // ObjVec slot-kernel paths — operate on canonical view so they
        // fire whether the lowered pipeline kept fused Sinks or kept
        // base Sink + last Stage(...). One match arm per kernel; no
        // per-fused-variant duplication.
        if let Val::ObjVec(d) = &recv {
            let (cs, _ck, csink) = self.canonical();
            // FlatMap(FieldRead) → flatmap-count
            if matches!(csink, Sink::Count) && cs.len() == 1 {
                if let Stage::FlatMap(prog) = &cs[0] {
                    if let Some(field) = single_field_prog(prog) {
                        if let Some(slot) = d.slot_of(field) {
                            return Some(Ok(objvec_flatmap_count_slot(d, slot)));
                        }
                    }
                }
            }
            // Map(FieldRead) → numeric-on-slot
            if let Sink::Numeric(op) = &csink {
                if cs.len() == 1 {
                    if let Stage::Map(prog) = &cs[0] {
                        let field = single_field_prog(prog)?;
                        let slot = d.slot_of(field)?;
                        return Some(Ok(objvec_num_slot(d, slot, *op)));
                    }
                }
                if cs.len() == 2 {
                    if let (Stage::Filter(pred), Stage::Map(map)) = (&cs[0], &cs[1]) {
                        let (pf, cop, lit) = single_cmp_prog(pred)?;
                        let mf = single_field_prog(map)?;
                        let sp = d.slot_of(pf)?;
                        let sm = d.slot_of(mf)?;
                        return Some(Ok(objvec_filter_num_slots(d, sp, cop, &lit, sm, *op)));
                    }
                }
            }
            // Filter(...) → count-if (single cmp or AND chain)
            if matches!(csink, Sink::Count) && cs.len() == 1 {
                if let Stage::Filter(pred) = &cs[0] {
                    if let Some((pf, op, lit)) = single_cmp_prog(pred) {
                        let sp = d.slot_of(pf)?;
                        return Some(Ok(objvec_filter_count_slot(d, sp, op, &lit)));
                    }
                    if let Some(leaves) = and_chain_prog(pred) {
                        let slots: Option<Vec<(usize, crate::ast::BinOp, Val)>> =
                            leaves.iter().map(|(f, op, lit)| {
                                d.slot_of(f).map(|s| (s, *op, lit.clone()))
                            }).collect();
                        let slots = slots?;
                        return Some(Ok(objvec_filter_count_and_slots(d, &slots)));
                    }
                }
            }
        }
        None
    }

    fn try_columnar(&self, root: &Val) -> Option<Result<Val, EvalError>> {
        // Phase A2 — Stage::Filter(FieldCmpLit) + Stage::Map(FieldRead) +
        // Sink::Collect over Val::Arr (object rows): walk the column
        // directly via slot-known IndexMap probes, build typed output
        // vec.  No per-row vm.exec.
        if let Some(out) = self.try_columnar_stage_chain(root) { return Some(out); }

        if !self.stages.is_empty() { return None; }

        let recv = match &self.source {
            Source::Receiver(v) => v.clone(),
            Source::FieldChain { keys } => walk_field_chain(root, keys),
        };
        // ObjVec promotion deferred to Phase 7 (build at parse time);
        // per-call promotion pays its O(N×K) cost on every collect()
        // and dominates short queries.  Keep the helper for a future
        // memoising path that caches the promoted shape on the source.

        // Phase A2 — typed-lane fast path.  When receiver is an
        // already-typed vector (IntVec / FloatVec / StrVec), the
        // sink can read the slice directly with no per-row Val tag
        // dispatch.  Each branch is mechanical: same fold, lane-typed.
        match (&recv, &self.sink) {
            (Val::IntVec(a), Sink::Numeric(NumOp::Sum)) =>
                return Some(Ok(Val::Int(a.iter().sum()))),
            (Val::IntVec(a), Sink::Numeric(NumOp::Min)) =>
                return Some(Ok(a.iter().copied().min()
                    .map(Val::Int).unwrap_or(Val::Null))),
            (Val::IntVec(a), Sink::Numeric(NumOp::Max)) =>
                return Some(Ok(a.iter().copied().max()
                    .map(Val::Int).unwrap_or(Val::Null))),
            (Val::IntVec(a), Sink::Numeric(NumOp::Avg)) => {
                if a.is_empty() { return Some(Ok(Val::Null)); }
                let s: i64 = a.iter().sum();
                return Some(Ok(Val::Float(s as f64 / a.len() as f64)));
            }
            (Val::IntVec(a), Sink::Count) =>
                return Some(Ok(Val::Int(a.len() as i64))),
            (Val::FloatVec(a), Sink::Numeric(NumOp::Sum)) =>
                return Some(Ok(Val::Float(a.iter().sum()))),
            (Val::FloatVec(a), Sink::Numeric(NumOp::Min)) => {
                if a.is_empty() { return Some(Ok(Val::Null)); }
                let m = a.iter().copied().fold(f64::INFINITY, f64::min);
                return Some(Ok(Val::Float(m)));
            }
            (Val::FloatVec(a), Sink::Numeric(NumOp::Max)) => {
                if a.is_empty() { return Some(Ok(Val::Null)); }
                let m = a.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                return Some(Ok(Val::Float(m)));
            }
            (Val::FloatVec(a), Sink::Numeric(NumOp::Avg)) => {
                if a.is_empty() { return Some(Ok(Val::Null)); }
                let s: f64 = a.iter().sum();
                return Some(Ok(Val::Float(s / a.len() as f64)));
            }
            (Val::FloatVec(a), Sink::Count) =>
                return Some(Ok(Val::Int(a.len() as i64))),
            (Val::StrVec(a), Sink::Count) =>
                return Some(Ok(Val::Int(a.len() as i64))),
            _ => {}
        }

        // ObjVec slot-kernel paths and Val::Arr columnar paths
        // previously dispatched on fused Sink variants. All gone post
        // fusion-off. Canonical-view consumer in `try_columnar_with`
        // covers the ObjVec shape; primitive-lane fast paths above
        // (IntVec / FloatVec / StrVec) cover Val::Arr-of-primitives.
        let _ = recv;
        None
    }

    /// Execute the pipeline against `root`, returning the sink's
    /// produced [`Val`].
    pub fn run(&self, root: &Val) -> Result<Val, EvalError> {
        self.run_with(root, None)
    }

    /// Tape-only fast path — does not require root Val.  Returns
    /// `Some(...)` only when the pipeline shape is tape-friendly AND
    /// the cache reports `prefer_tape() == true`.  Caller must fall
    /// back to `run_with` when this returns `None`.
    ///
    /// Critical for cold-start: lets callers skip lazy `root_val()`
    /// build when the query never touches the Val tree.
    #[cfg(feature = "simd-json")]
    pub fn try_run_no_root(
        &self,
        cache: &dyn ObjVecPromoter,
    ) -> Option<Result<Val, EvalError>> {
        if !cache.prefer_tape() { return None; }

        // Step 3c — composed tape route. Preferred over the legacy
        // `try_tape_aggregate` (which pattern-matches on fused Sink
        // variants) because it runs through the same generic Stage +
        // Sink substrate as the Val path. Returns None for shapes the
        // tape borrow stages don't yet cover (computed kernels, multi-
        // step bodies, barriers) — caller falls back to Val path.
        self.try_run_composed_tape(cache)
    }

    /// Decompose any fused Sink into a base `(stages, kernels, sink)`
    /// triple. Pure function — does not mutate `self`. Lets every
    /// Identity view — fused Sink variants deleted in Tier 3, so
    /// every Pipeline is already in base form. Kept as a stable name
    /// for downstream consumers (composed Val/tape runners, columnar
    /// fast paths, bytescan) that built on the canonical-view
    /// abstraction.
    pub fn canonical(&self) -> (Vec<Stage>, Vec<BodyKernel>, Sink) {
        (self.stages.clone(), self.stage_kernels.clone(), self.sink.clone())
    }

    /// Step 3c — composed-tape runner. Decomposes fused Sinks at entry
    /// (same logic as `try_run_composed`), then dispatches via the
    /// `composed::tape` substrate when every stage classifies to a
    /// tape borrow stage.
    #[cfg(feature = "simd-json")]
    fn try_run_composed_tape(
        &self,
        cache: &dyn ObjVecPromoter,
    ) -> Option<Result<Val, EvalError>> {
        use crate::composed::tape as ct;
        use crate::strref::tape_walk_field_chain;
        use std::cell::Cell;

        if !composed_path_enabled() { return None; }
        let tape = cache.tape()?;

        // Source must be a tape-resolvable field chain. Receiver-form
        // sources hold a Val that we'd have to look up in the tape;
        // skip for now.
        let arr_idx = match &self.source {
            Source::FieldChain { keys } => {
                let key_strs: Vec<&str> = keys.iter().map(|k| k.as_ref()).collect();
                tape_walk_field_chain(tape.as_ref(), &key_strs)?
            }
            _ => return None,
        };

        let (eff_stages, eff_kernels, eff_sink) = self.canonical();

        // Sink mapping. Barriers (Sort/UniqueBy/GroupBy/Reverse)
        // and computed sinks bail — Val path handles them.
        enum SinkKind { Count, Sum, Min, Max, Avg, First, Last, Collect }
        let sink_kind = match &eff_sink {
            Sink::Collect => SinkKind::Collect,
            Sink::Count => SinkKind::Count,
            Sink::Numeric(NumOp::Sum) => SinkKind::Sum,
            Sink::Numeric(NumOp::Min) => SinkKind::Min,
            Sink::Numeric(NumOp::Max) => SinkKind::Max,
            Sink::Numeric(NumOp::Avg) => SinkKind::Avg,
            Sink::First => SinkKind::First,
            Sink::Last => SinkKind::Last,
        };

        // Build tape stage chain. Reject any Generic / barrier /
        // computed kernel — Val path handles those.
        let build_tape_stage = |s: &Stage, k: &BodyKernel|
            -> Option<Box<dyn ct::TapeStage>> {
            Some(match (s, k) {
                (Stage::Filter(_), BodyKernel::FieldCmpLit(field, op, lit)) => {
                    let tape_op = binop_to_tape_cmp(*op)?;
                    let lit_owned = lit_to_tape_owned(lit)?;
                    Box::new(ct::TapeFilterFieldCmpLit {
                        field: Arc::clone(field),
                        op: tape_op,
                        lit: lit_owned,
                    })
                }
                (Stage::Filter(_), BodyKernel::FieldChainCmpLit(keys, op, lit)) => {
                    let tape_op = binop_to_tape_cmp(*op)?;
                    let lit_owned = lit_to_tape_owned(lit)?;
                    Box::new(ct::TapeFilterFieldChainCmpLit {
                        keys: Arc::clone(keys),
                        op: tape_op,
                        lit: lit_owned,
                    })
                }
                (Stage::Filter(_), BodyKernel::FieldRead(field)) => {
                    Box::new(ct::TapeFilterTruthyAtField { field: Arc::clone(field) })
                }
                (Stage::Map(_), BodyKernel::FieldRead(field)) =>
                    Box::new(ct::TapeMapField { field: Arc::clone(field) }),
                (Stage::Map(_), BodyKernel::FieldChain(keys)) =>
                    Box::new(ct::TapeMapFieldChain { keys: Arc::clone(keys) }),
                (Stage::FlatMap(_), BodyKernel::FieldRead(field)) =>
                    Box::new(ct::TapeFlatMapField { field: Arc::clone(field) }),
                (Stage::FlatMap(_), BodyKernel::FieldChain(keys)) =>
                    Box::new(ct::TapeFlatMapFieldChain { keys: Arc::clone(keys) }),
                (Stage::Take(n), _) =>
                    Box::new(ct::TapeTake { remaining: Cell::new(*n) }),
                (Stage::Skip(n), _) =>
                    Box::new(ct::TapeSkip { remaining: Cell::new(*n) }),
                _ => return None,
            })
        };

        // Reject barrier stages on tape route — they require buffered
        // Val materialisation. Composed Val path handles those.
        for s in &eff_stages {
            if matches!(s, Stage::Reverse | Stage::Sort(_) | Stage::UniqueBy(_) | Stage::GroupBy(_)) {
                return None;
            }
        }
        let mut chain: Box<dyn ct::TapeStage> = Box::new(ct::TapeIdentity);
        for (i, stage) in eff_stages.iter().enumerate() {
            let kernel = eff_kernels.get(i).unwrap_or(&BodyKernel::Generic);
            let next = build_tape_stage(stage, kernel)?;
            chain = Box::new(ct::TapeComposed { a: chain, b: next });
        }

        let result = match sink_kind {
            SinkKind::Count   => ct::run_pipeline_tape::<ct::TapeCountSink>(tape, arr_idx, chain.as_ref()),
            SinkKind::Sum     => ct::run_pipeline_tape::<ct::TapeSumSink>(tape, arr_idx, chain.as_ref()),
            SinkKind::Min     => ct::run_pipeline_tape::<ct::TapeMinSink>(tape, arr_idx, chain.as_ref()),
            SinkKind::Max     => ct::run_pipeline_tape::<ct::TapeMaxSink>(tape, arr_idx, chain.as_ref()),
            SinkKind::Avg     => ct::run_pipeline_tape::<ct::TapeAvgSink>(tape, arr_idx, chain.as_ref()),
            SinkKind::First   => ct::run_pipeline_tape_first(tape, arr_idx, chain.as_ref()),
            SinkKind::Last    => ct::run_pipeline_tape_last(tape, arr_idx, chain.as_ref()),
            SinkKind::Collect => ct::run_pipeline_tape_collect(tape, arr_idx, chain.as_ref()),
        };
        let result = result?;
        cache.note_tape_run();
        Some(Ok(result))
    }

    /// Layer B — composed-Cow Stage chain runner.
    ///
    /// Returns `Some(Ok(val))` when:
    ///   - source resolves to a `Val::Arr` (or other materialisable seq)
    ///   - sink is one of the base generic forms (Collect / Count /
    ///     Numeric(Sum/Min/Max/Avg) / First / Last)
    ///   - every stage is one of: Filter/Map/FlatMap/Take/Skip with a
    ///     borrow-form-recognised BodyKernel (FieldRead / FieldChain /
    ///     FieldCmpLit Eq / Generic-via-VM-fallback)
    ///   - no barrier stages (Sort/UniqueBy/Reverse/GroupBy) — Day 4-5
    ///
    /// Returns `None` for fused sinks (NumMap/NumFilterMap/CountIf/
    /// FilterFirst/etc.) — Tier 3 will lower those into base sink +
    /// stage chain so the composed path becomes the sole exec route.
    /// Step 3d Phase 5 — IndexedDispatch.  Generic O(1) optimisation
    /// for `(Map | identity)*` chains terminated by a positional sink
    /// (`First` / `Last`).  Pulls source[idx] only, runs the chain on
    /// that single element, returns.
    ///
    /// Falls through (`None`) when:
    /// - any stage is not 1:1 (Filter, FlatMap, Take, Skip, barriers),
    /// - sink isn't positional (Sum/Min/Max/Count/Avg/Collect),
    /// - source is not an indexable Val::Arr / typed-vec lane,
    /// - chain target index is out of bounds (returns Null via the
    ///   normal fallback so error semantics match).
    fn try_indexed_dispatch(&self, root: &Val) -> Option<Result<Val, EvalError>> {
        // Phase 5 strategy must be IndexedDispatch.
        let strategy = select_strategy(&self.stages, &self.sink);
        if strategy != Strategy::IndexedDispatch { return None; }

        // Resolve source — same rules as the generic loop.
        let recv = match &self.source {
            Source::Receiver(v) => v.clone(),
            Source::FieldChain { keys } => walk_field_chain(root, keys),
        };
        // Only indexable shapes; bail otherwise.
        let len = match &recv {
            Val::Arr(a)      => a.len(),
            Val::IntVec(a)   => a.len(),
            Val::FloatVec(a) => a.len(),
            Val::StrVec(a)   => a.len(),
            Val::ObjVec(d)   => d.nrows(),
            _ => return None,
        };

        // Compute target index from sink demand.
        let demand = self.sink.demand();
        let idx = match demand.positional? {
            Position::First    => 0,
            Position::Last     => len.checked_sub(1)?,
            Position::Nth(k)   => k,
        };
        if idx >= len {
            // Out of bounds — sink::First/Last semantics return Null.
            return Some(Ok(Val::Null));
        }

        // Pull element[idx] from source.
        let elem = match &recv {
            Val::Arr(a)      => a[idx].clone(),
            Val::IntVec(a)   => Val::Int(a[idx]),
            Val::FloatVec(a) => Val::Float(a[idx]),
            Val::StrVec(a)   => Val::Str(Arc::clone(&a[idx])),
            Val::ObjVec(d)   => {
                let stride = d.stride();
                let mut m: indexmap::IndexMap<Arc<str>, Val>
                    = indexmap::IndexMap::with_capacity(stride);
                for (i, k) in d.keys.iter().enumerate() {
                    m.insert(Arc::clone(k), d.cells[idx * stride + i].clone());
                }
                Val::Obj(Arc::new(m))
            }
            _ => return None,
        };

        // Run chain on the single element.  All stages are 1:1 (Map),
        // so each apply produces exactly one Val.
        let mut vm = crate::vm::VM::new();
        let mut env = vm.make_loop_env(root.clone());
        let mut cur = elem;
        for stage in &self.stages {
            match stage {
                Stage::Map(prog) => {
                    let prev = env.swap_current(cur);
                    cur = match vm.exec_in_env(prog, &mut env) {
                        Ok(v) => v,
                        Err(e) => { env.restore_current(prev); return Some(Err(e)); }
                    };
                    env.restore_current(prev);
                }
                _ => return None, // shape-check should have rejected; defensive.
            }
        }

        Some(Ok(cur))
    }

    fn try_run_composed(&self, root: &Val) -> Option<Result<Val, EvalError>> {
        use crate::composed as cmp;
        use crate::composed::Stage as ComposedStage;
        use std::cell::{Cell, RefCell};
        use std::rc::Rc;

        let (eff_stages, eff_kernels, eff_sink) = self.canonical();

        // Shared VM + Env for any Generic-kernel stage in the chain.
        // Constructed once per call so compile/path caches amortise.
        // Lazy: only built if a Generic stage actually appears.
        let vm_ctx: std::cell::OnceCell<Rc<RefCell<cmp::VmCtx>>> = std::cell::OnceCell::new();
        let make_ctx = || -> Rc<RefCell<cmp::VmCtx>> {
            let vm = crate::vm::VM::new();
            let env = vm.make_loop_env(root.clone());
            Rc::new(RefCell::new(cmp::VmCtx { vm, env }))
        };
        let get_ctx = || -> Rc<RefCell<cmp::VmCtx>> {
            Rc::clone(vm_ctx.get_or_init(make_ctx))
        };

        // Sink mapping — operates on the post-decomposition base sink.
        enum SinkKind { Count, Sum, Min, Max, Avg, First, Last, Collect, GroupByOnly }
        let sink_kind = match &eff_sink {
            Sink::Collect => SinkKind::Collect,
            Sink::Count => SinkKind::Count,
            Sink::Numeric(NumOp::Sum) => SinkKind::Sum,
            Sink::Numeric(NumOp::Min) => SinkKind::Min,
            Sink::Numeric(NumOp::Max) => SinkKind::Max,
            Sink::Numeric(NumOp::Avg) => SinkKind::Avg,
            Sink::First => SinkKind::First,
            Sink::Last => SinkKind::Last,
        };
        let _ = SinkKind::GroupByOnly;

        // Build a `KeySource` from a barrier-style kernel. Returns None
        // when the kernel is computed (Arith/FString/Generic) — caller
        // bails to legacy.
        fn key_from_kernel(k: &BodyKernel) -> Option<cmp::KeySource> {
            match k {
                BodyKernel::FieldRead(f) => Some(cmp::KeySource::Field(Arc::clone(f))),
                BodyKernel::FieldChain(keys) => Some(cmp::KeySource::Chain(Arc::clone(keys))),
                _ => None,
            }
        }

        // Build a streaming Stage from (Stage, BodyKernel). Recognised
        // borrow kernels (FieldRead / FieldChain / FieldCmpLit Eq)
        // dispatch to zero-clone borrow stages; everything else falls
        // through to the Generic VM-fallback stage that re-enters
        // `vm.exec_in_env` per row using the shared `vm_ctx`. One arm
        // per kernel pattern; one mechanism for every body shape.
        let build_stream_stage = |s: &Stage, k: &BodyKernel| -> Option<Box<dyn ComposedStage>> {
            Some(match (s, k) {
                (Stage::Filter(_), BodyKernel::FieldCmpLit(field, op, lit))
                    if matches!(op, crate::ast::BinOp::Eq) =>
                    Box::new(cmp::FilterFieldEqLit {
                        field: Arc::clone(field),
                        target: lit.clone(),
                    }),
                (Stage::Map(_), BodyKernel::FieldRead(field)) =>
                    Box::new(cmp::MapField { field: Arc::clone(field) }),
                (Stage::Map(_), BodyKernel::FieldChain(keys)) =>
                    Box::new(cmp::MapFieldChain { keys: Arc::clone(keys) }),
                (Stage::FlatMap(_), BodyKernel::FieldRead(field)) =>
                    Box::new(cmp::FlatMapField { field: Arc::clone(field) }),
                (Stage::FlatMap(_), BodyKernel::FieldChain(keys)) =>
                    Box::new(cmp::FlatMapFieldChain { keys: Arc::clone(keys) }),
                (Stage::Take(n), _) => Box::new(cmp::Take { remaining: Cell::new(*n) }),
                (Stage::Skip(n), _) => Box::new(cmp::Skip { remaining: Cell::new(*n) }),
                // VM-fallback for any unrecognised body — Generic kernel,
                // Arith, FString, FieldCmpLit non-Eq, custom lambdas.
                (Stage::Filter(p), _) =>
                    Box::new(cmp::GenericFilter { prog: Arc::clone(p), ctx: get_ctx() }),
                (Stage::Map(p), _) =>
                    Box::new(cmp::GenericMap { prog: Arc::clone(p), ctx: get_ctx() }),
                (Stage::FlatMap(p), _) =>
                    Box::new(cmp::GenericFlatMap { prog: Arc::clone(p), ctx: get_ctx() }),
                _ => return None,
            })
        };

        // Resolve source to an owned Vec<Val>. Future: avoid clone on
        // pure Arr by holding Arc<Vec<Val>> for the first segment.
        let recv = match &self.source {
            Source::Receiver(v) => v.clone(),
            Source::FieldChain { keys } => walk_field_chain(root, keys),
        };
        let mut buf: Vec<Val> = match recv {
            Val::Arr(a) => a.as_ref().clone(),
            Val::IntVec(a) => a.iter().map(|n| Val::Int(*n)).collect(),
            Val::FloatVec(a) => a.iter().map(|f| Val::Float(*f)).collect(),
            Val::StrVec(a) => a.iter().map(|s| Val::Str(Arc::clone(s))).collect(),
            _ => return None,
        };

        // Walk stages, splitting at barriers. Each streaming run uses
        // a composed-Cow chain into a CollectSink to materialise the
        // intermediate Vec<Val>; each barrier consumes Vec, returns
        // Vec. Final segment uses the actual sink. GroupBy is treated
        // as a barrier whose output Val replaces the buffer (used
        // only when followed by no further stages + Sink::Collect).
        let kernels = &eff_kernels;
        let stages_ref = &eff_stages;

        // Step 3d Phase 1: demand propagation.  Each stage sees the
        // demand its downstream wants, picks an algorithm, and exposes
        // upstream demand.  Sort under bounded downstream demand picks
        // top-k instead of full sort — generic mechanism, no rewrite
        // rule.
        let strategies = compute_strategies(stages_ref, &eff_sink);

        // Find barrier positions. Each [last_split..barrier_idx] is a
        // streaming segment; [barrier_idx] is the barrier op.
        let mut last_split = 0usize;
        for (i, s) in stages_ref.iter().enumerate() {
            let is_barrier = matches!(s,
                Stage::Reverse | Stage::Sort(_) | Stage::UniqueBy(_) |
                Stage::GroupBy(_));
            if !is_barrier { continue; }

            // Build streaming chain over [last_split..i].
            if i > last_split {
                let mut chain: Box<dyn ComposedStage> = Box::new(cmp::Identity);
                for j in last_split..i {
                    let stage = &stages_ref[j];
                    let kernel = kernels.get(j).unwrap_or(&BodyKernel::Generic);
                    let next = build_stream_stage(stage, kernel)?;
                    chain = Box::new(cmp::Composed { a: chain, b: next });
                }
                let out = cmp::run_pipeline::<cmp::CollectSink>(&buf, chain.as_ref());
                buf = match out {
                    Val::Arr(a) => a.as_ref().clone(),
                    _ => return None,
                };
            }

            // Apply barrier.
            let kernel = kernels.get(i).unwrap_or(&BodyKernel::Generic);
            let strategy = strategies.get(i).copied().unwrap_or(StageStrategy::Default);
            buf = match s {
                Stage::Reverse => cmp::barrier_reverse(buf),
                Stage::Sort(None) => match strategy {
                    StageStrategy::SortTopK(k) =>
                        cmp::barrier_top_k(buf, &cmp::KeySource::None, k),
                    StageStrategy::SortBottomK(k) =>
                        cmp::barrier_bottom_k(buf, &cmp::KeySource::None, k),
                    _ => cmp::barrier_sort(buf, &cmp::KeySource::None),
                },
                Stage::Sort(Some(_)) => {
                    let key = key_from_kernel(kernel)?;
                    match strategy {
                        StageStrategy::SortTopK(k) => cmp::barrier_top_k(buf, &key, k),
                        StageStrategy::SortBottomK(k) => cmp::barrier_bottom_k(buf, &key, k),
                        _ => cmp::barrier_sort(buf, &key),
                    }
                }
                Stage::UniqueBy(None) =>
                    cmp::barrier_unique_by(buf, &cmp::KeySource::None),
                Stage::UniqueBy(Some(_)) => {
                    let key = key_from_kernel(kernel)?;
                    cmp::barrier_unique_by(buf, &key)
                }
                Stage::GroupBy(_) => {
                    if !matches!(eff_sink, Sink::Collect) { return None; }
                    if i + 1 != stages_ref.len() { return None; }
                    let key = key_from_kernel(kernel)?;
                    let val = cmp::barrier_group_by(buf, &key);
                    let _ = i;
                    return Some(Ok(val));
                }
                _ => unreachable!(),
            };

            last_split = i + 1;
        }

        // Final streaming segment + sink.
        let mut chain: Box<dyn ComposedStage> = Box::new(cmp::Identity);
        for j in last_split..stages_ref.len() {
            let stage = &stages_ref[j];
            let kernel = kernels.get(j).unwrap_or(&BodyKernel::Generic);
            let next = build_stream_stage(stage, kernel)?;
            chain = Box::new(cmp::Composed { a: chain, b: next });
        }

        let out = match sink_kind {
            SinkKind::Count   => cmp::run_pipeline::<cmp::CountSink>(&buf, chain.as_ref()),
            SinkKind::Sum     => cmp::run_pipeline::<cmp::SumSink>(&buf, chain.as_ref()),
            SinkKind::Min     => cmp::run_pipeline::<cmp::MinSink>(&buf, chain.as_ref()),
            SinkKind::Max     => cmp::run_pipeline::<cmp::MaxSink>(&buf, chain.as_ref()),
            SinkKind::Avg     => cmp::run_pipeline::<cmp::AvgSink>(&buf, chain.as_ref()),
            SinkKind::First   => cmp::run_pipeline::<cmp::FirstSink>(&buf, chain.as_ref()),
            SinkKind::Last    => cmp::run_pipeline::<cmp::LastSink>(&buf, chain.as_ref()),
            SinkKind::Collect => cmp::run_pipeline::<cmp::CollectSink>(&buf, chain.as_ref()),
            SinkKind::GroupByOnly => unreachable!(),
        };

        Some(Ok(out))
    }

    /// Execute with an optional ObjVec promotion cache.  When `cache`
    /// is `Some`, the pipeline consults it before resolving sources;
    /// uniform-shape `Val::Arr<Val::Obj>` arrays are promoted to
    /// `Val::ObjVec` once and reused on every subsequent call.  Cost
    /// O(N×K) on first promotion, O(1) on hit.  Empty cache (`None`)
    /// matches legacy `run` semantics.
    pub fn run_with(
        &self,
        root: &Val,
        cache: Option<&dyn ObjVecPromoter>,
    ) -> Result<Val, EvalError> {
        // Step 3d Phase 5 — IndexedDispatch.  When stages are all 1:1
        // (`Map`, `Identity`) and sink is positional (First/Last), pull
        // the target element from the source by index, run chain once,
        // return.  O(1) work for `$.books.map(@.x).first()` shape.
        if let Some(out) = self.try_indexed_dispatch(root) { return out; }

        // Phase 3 columnar fast path — runs before per-row loop.
        // Critical for Q12/Q15-class queries: ObjVec promotion +
        // typed-column slot kernels reach native parity. Composed
        // path runs AFTER, as fallback for the per-row generic case.
        if let Some(out) = self.try_columnar_with(root, cache) { return out; }
        // Fall back to legacy try_columnar (no cache).
        if cache.is_none() {
            if let Some(out) = self.try_columnar(root) { return out; }
        }

        // Layer B — composed-Cow Stage chain. Opt-in under
        // `JETRO_COMPOSED=1`. Replaces the legacy per-row loop for
        // pipelines whose shape composed handles. Decomposes fused
        // Sinks (NumMap/NumFilterMap/CountIf/etc.) into base Stage +
        // base Sink at entry — composition handles the rest.
        if composed_path_enabled() {
            if let Some(out) = self.try_run_composed(root) { return out; }
        }

        // One VM owned by the pull loop — shared across stage program
        // calls so VM compile / path caches amortise across the row
        // sweep.  Constructing a fresh VM per row regresses 250x.
        let mut vm = crate::vm::VM::new();
        // Phase A1: build one Env at loop entry; per-row apply uses
        // `swap_current` instead of full Env construction + doc-hash
        // recompute + cache clear (those add ~80 ns/row of pure
        // overhead in execute_val_raw).
        let mut loop_env = vm.make_loop_env(root.clone());

        // Resolve source to an iterable Val::Arr-like sequence.
        let recv = match &self.source {
            Source::Receiver(v) => v.clone(),
            Source::FieldChain { keys } => walk_field_chain(root, keys),
        };

        // Pull-based stage chain.  At Phase 1 the inner loop materialises
        // elements one at a time as `Val`; Phase 3 will switch this to a
        // per-batch pull over columnar lanes.
        let mut taken: usize = 0;
        let mut skipped: usize = 0;

        let iter: Box<dyn Iterator<Item = Val>> = match &recv {
            Val::Arr(a)      => Box::new(a.as_ref().clone().into_iter()),
            Val::IntVec(a)   => Box::new(a.iter().map(|n| Val::Int(*n)).collect::<Vec<_>>().into_iter()),
            Val::FloatVec(a) => Box::new(a.iter().map(|f| Val::Float(*f)).collect::<Vec<_>>().into_iter()),
            Val::StrVec(a)   => Box::new(a.iter().map(|s| Val::Str(Arc::clone(s))).collect::<Vec<_>>().into_iter()),
            // ObjVec: materialise rows into Val::Obj for the per-row pull
            // path.  Slot-indexed columnar fast paths in `try_columnar`
            // handle the common SumMap / CountIf / SumFilterMap shapes
            // before this point — landing here means the sink is
            // Collect / take / skip / etc., which truly need Val::Obj
            // rows for downstream stages.
            Val::ObjVec(d)   => {
                let n = d.nrows();
                let mut out: Vec<Val> = Vec::with_capacity(n);
                let stride = d.stride();
                for row in 0..n {
                    let mut m: indexmap::IndexMap<Arc<str>, Val> = indexmap::IndexMap::with_capacity(stride);
                    for (i, k) in d.keys.iter().enumerate() {
                        m.insert(Arc::clone(k), d.cells[row * stride + i].clone());
                    }
                    out.push(Val::Obj(Arc::new(m)));
                }
                Box::new(out.into_iter())
            }
            // Anything else (scalar, Obj, …): single-element "iterator".
            _ => Box::new(std::iter::once(recv.clone())),
        };

        // Sink accumulators.
        let mut acc_collect: Vec<Val> = Vec::new();
        let mut acc_count:   i64 = 0;
        let mut acc_sum_i:   i64 = 0;
        let mut acc_sum_f:   f64 = 0.0;
        let mut sum_floated: bool = false;
        let mut acc_min_f:   f64 = f64::INFINITY;
        let mut acc_max_f:   f64 = f64::NEG_INFINITY;
        let mut acc_n_obs:   usize = 0;
        let mut acc_first:   Option<Val> = None;
        let mut acc_last:    Option<Val> = None;

        // Stages that materialise force a buffer; stages preceding
        // them run as streaming filter/map over the buffer.  Process
        // every stage in order so the pipeline semantics match the
        // surface query.
        let needs_barrier = self.stages.iter().any(|s| matches!(s,
            Stage::Reverse | Stage::Sort(_) | Stage::UniqueBy(_)
                | Stage::FlatMap(_) | Stage::GroupBy(_)
                | Stage::Split(_) | Stage::Chunk(_) | Stage::Window(_)));
        let pre_iter: Box<dyn Iterator<Item = Val>> = if needs_barrier {
            let mut buf: Vec<Val> = iter.collect();
            // Phase 1.2 — barrier-stage path now reads stage_kernels[i]
            // and dispatches the inline kernel for Sort/UniqueBy keyed
            // variants too, not just streaming Filter/Map.  Extends
            // Layer A coverage to the keyed-barrier surface.
            for (stage_idx, stage) in self.stages.iter().enumerate() {
                let kernel = self.stage_kernels.get(stage_idx)
                    .unwrap_or(&BodyKernel::Generic);
                match stage {
                    Stage::Filter(prog) => {
                        let mut out: Vec<Val> = Vec::with_capacity(buf.len());
                        for v in buf.into_iter() {
                            if is_truthy(&eval_kernel(kernel, &v, &mut vm, &mut loop_env, prog)?) {
                                out.push(v);
                            }
                        }
                        buf = out;
                    }
                    Stage::Map(prog) => {
                        let mut out: Vec<Val> = Vec::with_capacity(buf.len());
                        for v in buf.into_iter() {
                            out.push(eval_kernel(kernel, &v, &mut vm, &mut loop_env, prog)?);
                        }
                        buf = out;
                    }
                    Stage::Skip(n) => {
                        if buf.len() <= *n { buf.clear(); } else { buf.drain(..*n); }
                    }
                    Stage::Take(n) => {
                        buf.truncate(*n);
                    }
                    Stage::Reverse => buf.reverse(),
                    Stage::Sort(None) => buf.sort_by(|a, b| cmp_val_total(a, b)),
                    Stage::Sort(Some(prog)) => {
                        let mut keyed: Vec<(Val, Val)> = Vec::with_capacity(buf.len());
                        for v in buf.into_iter() {
                            let k = eval_kernel(kernel, &v, &mut vm, &mut loop_env, prog)
                                .unwrap_or(Val::Null);
                            keyed.push((k, v));
                        }
                        keyed.sort_by(|a, b| cmp_val_total(&a.0, &b.0));
                        buf = keyed.into_iter().map(|(_, v)| v).collect();
                    }
                    Stage::UniqueBy(None) => {
                        let mut seen: std::collections::HashSet<String> = Default::default();
                        buf.retain(|v| seen.insert(format!("{:?}", v)));
                    }
                    Stage::UniqueBy(Some(prog)) => {
                        let mut seen: std::collections::HashSet<String> = Default::default();
                        let mut keep: Vec<bool> = Vec::with_capacity(buf.len());
                        for v in &buf {
                            let k = eval_kernel(kernel, v, &mut vm, &mut loop_env, prog)
                                .unwrap_or(Val::Null);
                            keep.push(seen.insert(format!("{:?}", k)));
                        }
                        let mut out: Vec<Val> = Vec::with_capacity(buf.len());
                        for (i, v) in buf.into_iter().enumerate() { if keep[i] { out.push(v); } }
                        buf = out;
                    }
                    Stage::FlatMap(prog) => {
                        let mut out: Vec<Val> = Vec::new();
                        for v in &buf {
                            let inner = eval_kernel(kernel, v, &mut vm, &mut loop_env, prog)?;
                            if let Some(arr) = inner.as_vals() {
                                out.extend(arr.iter().cloned());
                            } else {
                                out.push(inner);
                            }
                        }
                        buf = out;
                    }
                    Stage::GroupBy(prog) => {
                        // Build IndexMap<key_str, Vec<row>> via per-row
                        // kernel-evaluated key.  Output is Val::Obj with
                        // group keys → group arrays.  Drains buf into
                        // groups; subsequent stages (.values(), .map())
                        // see the grouped Obj.
                        use indexmap::IndexMap;
                        use crate::eval::util::val_to_key;
                        let mut groups: IndexMap<Arc<str>, Vec<Val>> = IndexMap::new();
                        for v in buf.into_iter() {
                            let k = eval_kernel(kernel, &v, &mut vm, &mut loop_env, prog)?;
                            let key = Arc::<str>::from(val_to_key(&k).as_str());
                            groups.entry(key).or_insert_with(Vec::new).push(v);
                        }
                        // Convert each Vec<Val> bucket to Val::arr.
                        let mut out_obj: IndexMap<Arc<str>, Val> = IndexMap::with_capacity(groups.len());
                        for (k, rows) in groups.into_iter() {
                            out_obj.insert(k, Val::arr(rows));
                        }
                        // GroupBy is a barrier that yields one Val::Obj.
                        // Place it as a single-element buf so downstream
                        // stages see the grouped object.  Sink::Collect
                        // will return Val::arr([this_obj]); a separate
                        // shortcut below converts that to the bare obj.
                        buf = vec![Val::Obj(Arc::new(out_obj))];
                    }
                    Stage::Split(sep) => {
                        // Step 3d-extension (C): Expanding string Stage.
                        let mut out: Vec<Val> = Vec::with_capacity(buf.len());
                        for v in buf.into_iter() {
                            if let Some(Val::Arr(a)) = split_apply(&v, sep.as_ref()) {
                                out.extend(Arc::try_unwrap(a)
                                    .unwrap_or_else(|a| (*a).clone()));
                            }
                        }
                        buf = out;
                    }
                    Stage::Slice(start, end) => {
                        let mut out: Vec<Val> = Vec::with_capacity(buf.len());
                        for v in buf.into_iter() {
                            out.push(slice_apply(v, *start, *end));
                        }
                        buf = out;
                    }
                    Stage::Replace { needle, replacement, all } => {
                        let mut out: Vec<Val> = Vec::with_capacity(buf.len());
                        for v in buf.into_iter() {
                            match replace_apply(v.clone(), needle, replacement, *all) {
                                Some(r) => out.push(r),
                                None    => out.push(v),
                            }
                        }
                        buf = out;
                    }
                    Stage::Chunk(n) => {
                        buf = chunk_apply(&buf, *n);
                    }
                    Stage::Window(n) => {
                        buf = window_apply(&buf, *n);
                    }
                    Stage::CompiledMap(plan) => {
                        let mut out: Vec<Val> = Vec::with_capacity(buf.len());
                        for v in buf.into_iter() {
                            out.push(run_compiled_map(plan, v)?);
                        }
                        buf = out;
                    }
                    Stage::Keys => {
                        buf = buf.into_iter().map(|v| keys_apply(&v)).collect();
                    }
                    Stage::Values => {
                        buf = buf.into_iter().map(|v| values_apply(&v)).collect();
                    }
                    Stage::Entries => {
                        buf = buf.into_iter().map(|v| entries_apply(&v)).collect();
                    }
                }
            }
            Box::new(buf.into_iter())
        } else {
            iter
        };

        'outer: for mut item in pre_iter {
            // When barriers ran above, stages have already been
            // applied — `pre_iter` yields the post-pipeline rows
            // directly.  When no barriers are present, run streaming
            // stages here.
            if !needs_barrier {
                for (stage_idx, stage) in self.stages.iter().enumerate() {
                    let kernel = self.stage_kernels.get(stage_idx)
                        .unwrap_or(&BodyKernel::Generic);
                    match stage {
                        Stage::Skip(n) => {
                            if skipped < *n { skipped += 1; continue 'outer; }
                        }
                        Stage::Take(n) => {
                            if taken >= *n { break 'outer; }
                        }
                        Stage::Filter(prog) => {
                            let v = eval_kernel(kernel, &item, &mut vm, &mut loop_env, prog)?;
                            if !is_truthy(&v) { continue 'outer; }
                        }
                        Stage::Map(prog) => {
                            item = eval_kernel(kernel, &item, &mut vm, &mut loop_env, prog)?;
                        }
                        Stage::Reverse | Stage::Sort(_) | Stage::UniqueBy(_)
                        | Stage::FlatMap(_) | Stage::GroupBy(_) => {}
                        Stage::Split(_) | Stage::Chunk(_) | Stage::Window(_) => {} // forced into barrier path above
                        Stage::Slice(start, end) => {
                            item = slice_apply(item, *start, *end);
                        }
                        Stage::Replace { needle, replacement, all } => {
                            if let Some(r) = replace_apply(item.clone(), needle, replacement, *all) {
                                item = r;
                            }
                        }
                        Stage::CompiledMap(plan) => {
                            item = run_compiled_map(plan, item)?;
                        }
                        Stage::Keys    => { item = keys_apply(&item); }
                        Stage::Values  => { item = values_apply(&item); }
                        Stage::Entries => { item = entries_apply(&item); }
                    }
                }
            }

            // Sink.
            match &self.sink {
                Sink::Collect => acc_collect.push(item),
                Sink::Count   => acc_count += 1,
                Sink::Numeric(op) => {
                    num_fold(&mut acc_sum_i, &mut acc_sum_f, &mut sum_floated,
                             &mut acc_min_f, &mut acc_max_f, &mut acc_n_obs,
                             *op, &item);
                }
                Sink::First => { if acc_first.is_none() { acc_first = Some(item.clone()); } }
                Sink::Last  => { acc_last = Some(item.clone()); }
            }
            taken += 1;
        }

        Ok(match &self.sink {
            Sink::Collect => {
                // GroupBy is a barrier that produces a single Val::Obj
                // which Sink::Collect would otherwise wrap as
                // [obj].  When the last stage is GroupBy, return the
                // bare object to match walker semantics.
                if matches!(self.stages.last(), Some(Stage::GroupBy(_)))
                    && acc_collect.len() == 1
                    && matches!(acc_collect[0], Val::Obj(_))
                {
                    acc_collect.into_iter().next().unwrap()
                } else {
                    Val::arr(acc_collect)
                }
            },
            Sink::Count             => Val::Int(acc_count),
            Sink::Numeric(op) =>
                num_finalise(*op, acc_sum_i, acc_sum_f, sum_floated,
                             acc_min_f, acc_max_f, acc_n_obs),
            Sink::First =>
                acc_first.unwrap_or(Val::Null),
            Sink::Last =>
                acc_last.unwrap_or(Val::Null),
        })
    }
}

/// Canonical `.slice(start[, end])` impl shared by `Stage::Slice` runtime
/// arms and the `.slice` built-in dispatch shim in `eval::builtins`.
/// Handles `Val::Str` + `Val::StrSlice` receivers; ASCII fast path returns
/// zero-alloc `Val::StrSlice` via `StrRef`; Unicode walks `char_indices`.
/// Negative indexes count from end (Python slice semantics); `end=None`
/// means "to end".  Non-string receiver returns the value unchanged
/// (Stage path passes it through; builtin shim guards before calling).
pub(crate) fn slice_apply(recv: Val, start: i64, end: Option<i64>) -> Val {
    let (parent, base_off, view_len): (Arc<str>, usize, usize) = match recv {
        Val::Str(s)      => { let l = s.len(); (s, 0, l) }
        Val::StrSlice(r) => {
            let parent = r.to_arc();
            let plen = parent.len();
            (parent, 0, plen)
        }
        other => return other,
    };
    let view = &parent[base_off .. base_off + view_len];
    let blen = view.len();
    if view.is_ascii() {
        let start_u = if start < 0 { blen.saturating_sub((-start) as usize) }
                       else        { (start as usize).min(blen) };
        let end_u = match end {
            Some(e) if e < 0 => blen.saturating_sub((-e) as usize),
            Some(e)          => (e as usize).min(blen),
            None             => blen,
        };
        let start_u = start_u.min(end_u);
        if start_u == 0 && end_u == blen { return Val::Str(parent); }
        return Val::StrSlice(crate::strref::StrRef::slice(
            parent, base_off + start_u, base_off + end_u,
        ));
    }
    let chars: Vec<(usize, char)> = view.char_indices().collect();
    let n = chars.len() as i64;
    let resolve = |i: i64| -> usize {
        let r = if i < 0 { n + i } else { i };
        r.clamp(0, n) as usize
    };
    let s_idx = resolve(start);
    let e_idx = match end { Some(e) => resolve(e), None => n as usize };
    let s_idx = s_idx.min(e_idx);
    let s_b = chars.get(s_idx).map(|c| c.0).unwrap_or(view.len());
    let e_b = chars.get(e_idx).map(|c| c.0).unwrap_or(view.len());
    if s_b == 0 && e_b == view.len() { return Val::Str(parent); }
    Val::StrSlice(crate::strref::StrRef::slice(
        parent, base_off + s_b, base_off + e_b,
    ))
}

/// Canonical `.split(sep)` impl shared by `Stage::Split` runtime arms and
/// the `.split` built-in dispatch shim.  Returns the split segments as
/// fresh `Val::Str` allocations wrapped in a `Val::Arr`.  `None` on
/// non-string receiver (Stage path silently drops; builtin shim turns
/// `None` into an `EvalError`).
pub(crate) fn split_apply(recv: &Val, sep: &str) -> Option<Val> {
    let s: &str = match recv {
        Val::Str(s)      => s.as_ref(),
        Val::StrSlice(r) => r.as_str(),
        _                => return None,
    };
    Some(Val::arr(s.split(sep).map(|p| Val::Str(Arc::<str>::from(p))).collect()))
}

/// Canonical `.chunk(n)` partition into chunks of size `n` (last may be
/// shorter).  Shared by Stage::Chunk runtime arm and the dispatch shim.
/// Each emitted Val is a `Val::arr` of up to `n` source elements.
pub(crate) fn chunk_apply(items: &[Val], n: usize) -> Vec<Val> {
    let n = n.max(1);
    items.chunks(n).map(|c| Val::arr(c.to_vec())).collect()
}

/// Canonical `.window(n)` sliding window of size `n` over the source
/// stream.  Shared by Stage::Window runtime arm and the dispatch shim.
/// Emits `len.saturating_sub(n) + 1` overlapping windows; empty when
/// `n > len`.
pub(crate) fn window_apply(items: &[Val], n: usize) -> Vec<Val> {
    let n = n.max(1);
    items.windows(n).map(|w| Val::arr(w.to_vec())).collect()
}

/// Canonical `.keys()` impl shared by `Stage::Keys` runtime arm and
/// the `.keys` builtin dispatch shim.  Non-object receivers yield an
/// empty array (matches prior owned semantics).
pub(crate) fn keys_apply(recv: &Val) -> Val {
    Val::arr(recv.as_object()
        .map(|m| m.keys().map(|k| Val::Str(k.clone())).collect())
        .unwrap_or_default())
}

/// Canonical `.values()` impl.
pub(crate) fn values_apply(recv: &Val) -> Val {
    Val::arr(recv.as_object()
        .map(|m| m.values().cloned().collect())
        .unwrap_or_default())
}

/// Canonical `.entries()` impl.  Each entry is a `Val::arr([key, value])`.
pub(crate) fn entries_apply(recv: &Val) -> Val {
    Val::arr(recv.as_object()
        .map(|m| m.iter()
            .map(|(k, v)| Val::arr(vec![Val::Str(k.clone()), v.clone()]))
            .collect())
        .unwrap_or_default())
}

/// Canonical `.replace(needle, repl)` (all=false, replacen-1) and
/// `.replace_all(needle, repl)` (all=true).  Shared by Stage::Replace
/// runtime arms and the dispatch shims.  Returns the receiver unchanged
/// when needle is absent (no alloc fast-path).  `None` on non-string
/// receiver.
pub(crate) fn replace_apply(recv: Val, needle: &str, replacement: &str, all: bool) -> Option<Val> {
    let s: Arc<str> = match recv {
        Val::Str(s)      => s,
        Val::StrSlice(r) => r.to_arc(),
        _                => return None,
    };
    if !s.contains(needle) { return Some(Val::Str(s)); }
    let out = if all { s.replace(needle, replacement) }
              else   { s.replacen(needle, replacement, 1) };
    Some(Val::Str(Arc::<str>::from(out)))
}

#[inline]
fn num_fold(
    acc_i: &mut i64, acc_f: &mut f64, floated: &mut bool,
    min_f: &mut f64, max_f: &mut f64, n_obs: &mut usize,
    op: NumOp, v: &Val,
) {
    let f = match v {
        Val::Int(n)   => *n as f64,
        Val::Float(x) => *x,
        _ => return,
    };
    *n_obs += 1;
    match op {
        NumOp::Sum | NumOp::Avg => {
            match v {
                Val::Int(n)   => if *floated { *acc_f += *n as f64 } else { *acc_i += *n },
                Val::Float(x) => {
                    if !*floated { *acc_f = *acc_i as f64; *floated = true; }
                    *acc_f += *x;
                }
                _ => {}
            }
        }
        NumOp::Min => { if f < *min_f { *min_f = f; } }
        NumOp::Max => { if f > *max_f { *max_f = f; } }
    }
}

#[inline]
fn num_finalise(
    op: NumOp,
    acc_i: i64, acc_f: f64, floated: bool,
    min_f: f64, max_f: f64, n_obs: usize,
) -> Val {
    if n_obs == 0 { return op.empty(); }
    match op {
        NumOp::Sum => if floated { Val::Float(acc_f) } else { Val::Int(acc_i) },
        NumOp::Avg => {
            let total = if floated { acc_f } else { acc_i as f64 };
            Val::Float(total / n_obs as f64)
        }
        NumOp::Min => Val::Float(min_f),
        NumOp::Max => Val::Float(max_f),
    }
}

// (sum_acc removed; superseded by num_fold which handles Sum/Min/Max/Avg)

/// Decode a compiled sub-program that reads a single field from `@`
/// — either `[PushCurrent, GetField(k)]` (explicit `@.field`) or
/// `[LoadIdent(k)]` (bare-ident shorthand).  Returns the field name.
fn single_field_prog(prog: &crate::vm::Program) -> Option<&str> {
    use crate::vm::Opcode;
    let ops = prog.ops.as_ref();
    match ops.len() {
        1 => match &ops[0] {
            Opcode::LoadIdent(k) => Some(k.as_ref()),
            _ => None,
        },
        2 => match (&ops[0], &ops[1]) {
            (Opcode::PushCurrent, Opcode::GetField(k)) => Some(k.as_ref()),
            _ => None,
        },
        _ => None,
    }
}

/// Decode a compound AND predicate (a chain of single-cmp predicates
/// joined by `AndOp`) into a flat list of leaves.  Operates directly
/// on the `&[Opcode]` slice so the returned `&str` field references
/// borrow from the original program — no Arc allocation per leaf.
///
/// Accepts the shapes the compiler emits in practice:
///   2-way:  `[<cmp1>, AndOp(<cmp2>)]`
///   3-way:  `[<cmp1>, AndOp([<cmp2>, AndOp(<cmp3>)])]`
fn and_chain_prog<'a>(prog: &'a crate::vm::Program) -> Option<Vec<(&'a str, crate::ast::BinOp, Val)>> {
    use crate::vm::Opcode;
    let ops = prog.ops.as_ref();
    let (last, head) = ops.split_last()?;
    let rhs = match last { Opcode::AndOp(rhs) => rhs, _ => return None };
    let head_leaf = decode_cmp_ops(head)?;
    let mut rest = and_chain_prog(rhs).or_else(|| decode_cmp_ops(rhs.ops.as_ref()).map(|x| vec![x]))?;
    let mut out = Vec::with_capacity(1 + rest.len());
    out.push(head_leaf);
    out.append(&mut rest);
    Some(out)
}

/// Match the single-cmp opcode prefix and return `(field, op, lit)`.
fn decode_cmp_ops<'a>(ops: &'a [crate::vm::Opcode]) -> Option<(&'a str, crate::ast::BinOp, Val)> {
    use crate::vm::Opcode;
    use crate::ast::BinOp;
    let (field, lit_idx, cmp_idx) = match ops.len() {
        3 => match &ops[0] {
            Opcode::LoadIdent(k) => (k.as_ref(), 1, 2),
            _ => return None,
        },
        4 => match (&ops[0], &ops[1]) {
            (Opcode::PushCurrent, Opcode::GetField(k)) => (k.as_ref(), 2, 3),
            _ => return None,
        },
        _ => return None,
    };
    let lit = match &ops[lit_idx] {
        Opcode::PushInt(n)   => Val::Int(*n),
        Opcode::PushFloat(f) => Val::Float(*f),
        Opcode::PushStr(s)   => Val::Str(Arc::clone(s)),
        Opcode::PushBool(b)  => Val::Bool(*b),
        Opcode::PushNull     => Val::Null,
        _ => return None,
    };
    let op = match &ops[cmp_idx] {
        Opcode::Eq  => BinOp::Eq,
        Opcode::Neq => BinOp::Neq,
        Opcode::Lt  => BinOp::Lt,
        Opcode::Lte => BinOp::Lte,
        Opcode::Gt  => BinOp::Gt,
        Opcode::Gte => BinOp::Gte,
        _ => return None,
    };
    Some((field, op, lit))
}

/// Decode a compiled predicate of the shape `<load-field-k>;
/// <push-lit>; <cmp>` into `(field, op, lit)`.  Thin wrapper over
/// `decode_cmp_ops` for backward-compat with existing callers that
/// pass a `Program`.
fn single_cmp_prog<'a>(prog: &'a crate::vm::Program) -> Option<(&'a str, crate::ast::BinOp, Val)> {
    decode_cmp_ops(prog.ops.as_ref())
}

// ── ObjVec slot-indexed kernels (Phase 3.5) ──────────────────────────────────
//
// When the receiver is an ObjVec the row layout is flat
// `cells: Vec<Val>` with stride = keys.len(); a row's field at slot
// `s` lives at `cells[row * stride + s]`.  No IndexMap probe per
// row — direct array index.

/// Columnar `$.<arr>.flat_map(<field>).count()` — sums lengths of the
/// inner sequences without materialising the flattened result.
fn objvec_flatmap_count_slot(d: &Arc<crate::eval::value::ObjVecData>, slot: usize) -> Val {
    let stride = d.stride();
    let nrows = d.nrows();
    let mut count: i64 = 0;
    for row in 0..nrows {
        let v = &d.cells[row * stride + slot];
        match v {
            Val::Arr(a)         => count += a.len() as i64,
            Val::IntVec(a)      => count += a.len() as i64,
            Val::FloatVec(a)    => count += a.len() as i64,
            Val::StrVec(a)      => count += a.len() as i64,
            Val::StrSliceVec(a) => count += a.len() as i64,
            Val::ObjVec(ad)     => count += ad.nrows() as i64,
            _                   => count += 1,
        }
    }
    Val::Int(count)
}

fn objvec_num_slot(d: &Arc<crate::eval::value::ObjVecData>, slot: usize, op: NumOp) -> Val {
    use crate::eval::value::ObjVecCol;
    // Phase 7-typed-columns fast path: typed lane → direct slice walk,
    // no per-row Val tag check.  Closes the boxed-Val tax on numeric
    // aggregates (~3-4× win measured on bench_complex Q12).
    if let Some(cols) = &d.typed_cols {
        match cols.get(slot) {
            Some(ObjVecCol::Ints(col)) => {
                if col.is_empty() {
                    return match op {
                        NumOp::Sum => Val::Int(0),
                        _ => Val::Null,
                    };
                }
                return match op {
                    NumOp::Sum => Val::Int(col.iter().sum()),
                    NumOp::Min => Val::Int(*col.iter().min().unwrap()),
                    NumOp::Max => Val::Int(*col.iter().max().unwrap()),
                    NumOp::Avg => {
                        let s: i64 = col.iter().sum();
                        Val::Float(s as f64 / col.len() as f64)
                    }
                };
            }
            Some(ObjVecCol::Floats(col)) => {
                if col.is_empty() {
                    return match op {
                        NumOp::Sum => Val::Float(0.0),
                        _ => Val::Null,
                    };
                }
                return match op {
                    NumOp::Sum => Val::Float(col.iter().sum()),
                    NumOp::Min => Val::Float(col.iter().copied().fold(f64::INFINITY, f64::min)),
                    NumOp::Max => Val::Float(col.iter().copied().fold(f64::NEG_INFINITY, f64::max)),
                    NumOp::Avg => {
                        let s: f64 = col.iter().sum();
                        Val::Float(s / col.len() as f64)
                    }
                };
            }
            _ => {}
        }
    }
    let stride = d.stride();
    let nrows = d.nrows();
    let mut acc_i: i64 = 0;
    let mut acc_f: f64 = 0.0;
    let mut floated = false;
    let mut min_f = f64::INFINITY;
    let mut max_f = f64::NEG_INFINITY;
    let mut n_obs: usize = 0;
    for row in 0..nrows {
        let v = &d.cells[row * stride + slot];
        num_fold(&mut acc_i, &mut acc_f, &mut floated, &mut min_f, &mut max_f, &mut n_obs, op, v);
    }
    num_finalise(op, acc_i, acc_f, floated, min_f, max_f, n_obs)
}

fn objvec_filter_count_slot(
    d:    &Arc<crate::eval::value::ObjVecData>,
    slot: usize,
    op:   crate::ast::BinOp,
    lit:  &Val,
) -> Val {
    use crate::eval::value::ObjVecCol;
    use crate::ast::BinOp as B;
    // Typed-column fast path.  Direct slice scan with primitive
    // comparison; no Val tag check, no boxed unbox.
    if let Some(cols) = &d.typed_cols {
        match (cols.get(slot), lit) {
            (Some(ObjVecCol::Ints(col)), Val::Int(rhs)) => {
                let r = *rhs;
                let mut c: i64 = 0;
                for &n in col.iter() {
                    let hit = match op {
                        B::Eq  => n == r,
                        B::Neq => n != r,
                        B::Lt  => n < r,
                        B::Lte => n <= r,
                        B::Gt  => n > r,
                        B::Gte => n >= r,
                        _ => false,
                    };
                    if hit { c += 1; }
                }
                return Val::Int(c);
            }
            (Some(ObjVecCol::Floats(col)), Val::Float(rhs)) => {
                let r = *rhs;
                let mut c: i64 = 0;
                for &f in col.iter() {
                    let hit = match op {
                        B::Eq  => f == r,
                        B::Neq => f != r,
                        B::Lt  => f < r,
                        B::Lte => f <= r,
                        B::Gt  => f > r,
                        B::Gte => f >= r,
                        _ => false,
                    };
                    if hit { c += 1; }
                }
                return Val::Int(c);
            }
            (Some(ObjVecCol::Strs(col)), Val::Str(rhs)) => {
                let r: &str = rhs.as_ref();
                let mut c: i64 = 0;
                for s in col.iter() {
                    let hit = match op {
                        B::Eq  => s.as_ref() == r,
                        B::Neq => s.as_ref() != r,
                        _ => false,
                    };
                    if hit { c += 1; }
                }
                return Val::Int(c);
            }
            _ => {}
        }
    }
    let stride = d.stride();
    let nrows = d.nrows();
    let mut count: i64 = 0;
    for row in 0..nrows {
        let v = &d.cells[row * stride + slot];
        if cmp_val_binop_local(v, op, lit) { count += 1; }
    }
    Val::Int(count)
}

fn objvec_filter_num_slots(
    d:        &Arc<crate::eval::value::ObjVecData>,
    pred_slot: usize,
    cop:      crate::ast::BinOp,
    lit:      &Val,
    map_slot:  usize,
    op:       NumOp,
) -> Val {
    use crate::eval::value::ObjVecCol;
    use crate::ast::BinOp as B;
    // Typed-column fast path for filter+map slot pair: walk both
    // columns as raw slices, primitive cmp + primitive fold.
    if let Some(cols) = &d.typed_cols {
        // Int pred + Int map (covers `total > 100` then `map(total)`).
        if let (Some(ObjVecCol::Ints(p)), Some(ObjVecCol::Ints(m)), Val::Int(rhs))
            = (cols.get(pred_slot), cols.get(map_slot), lit)
        {
            let r = *rhs;
            let mut acc_i: i64 = 0;
            let mut acc_f: f64 = 0.0;
            let mut floated = false;
            let mut min_f = f64::INFINITY;
            let mut max_f = f64::NEG_INFINITY;
            let mut n_obs: usize = 0;
            for (i, &pv) in p.iter().enumerate() {
                let hit = match cop {
                    B::Eq  => pv == r, B::Neq => pv != r,
                    B::Lt  => pv < r,  B::Lte => pv <= r,
                    B::Gt  => pv > r,  B::Gte => pv >= r,
                    _ => false,
                };
                if hit {
                    let v = Val::Int(m[i]);
                    num_fold(&mut acc_i, &mut acc_f, &mut floated, &mut min_f, &mut max_f, &mut n_obs, op, &v);
                }
            }
            return num_finalise(op, acc_i, acc_f, floated, min_f, max_f, n_obs);
        }
    }
    let stride = d.stride();
    let nrows = d.nrows();
    let mut acc_i: i64 = 0;
    let mut acc_f: f64 = 0.0;
    let mut floated = false;
    let mut min_f = f64::INFINITY;
    let mut max_f = f64::NEG_INFINITY;
    let mut n_obs: usize = 0;
    for row in 0..nrows {
        let off = row * stride;
        if cmp_val_binop_local(&d.cells[off + pred_slot], cop, lit) {
            num_fold(&mut acc_i, &mut acc_f, &mut floated, &mut min_f, &mut max_f, &mut n_obs,
                     op, &d.cells[off + map_slot]);
        }
    }
    num_finalise(op, acc_i, acc_f, floated, min_f, max_f, n_obs)
}

/// Phase 7-typed-columns Q3 path: ObjVec source + Filter(FieldCmpLit)
/// + Map(FieldRead) + Sink::Collect.  When both pred + map slots are
/// typed lanes, walk primitive columns directly: build typed output
/// vec sized by predicate hit count; no Val tag check, no IndexMap probe.
fn objvec_typed_filter_map_collect(
    d:    &Arc<crate::eval::value::ObjVecData>,
    pk:   &str,
    pop:  crate::ast::BinOp,
    plit: &Val,
    mk:   &str,
) -> Option<Result<Val, EvalError>> {
    use crate::eval::value::ObjVecCol;
    use crate::ast::BinOp as B;
    let cols = d.typed_cols.as_ref()?;
    let pred_slot = d.slot_of(pk)?;
    let map_slot  = d.slot_of(mk)?;
    let pred_col  = cols.get(pred_slot)?;
    let map_col   = cols.get(map_slot)?;

    // Int pred + Int map (e.g. `filter(total > 500).map(id)`).
    if let (ObjVecCol::Ints(p), ObjVecCol::Ints(m), Val::Int(rhs)) =
        (pred_col, map_col, plit)
    {
        let r = *rhs;
        let mut out: Vec<i64> = Vec::with_capacity(p.len());
        for (i, &pv) in p.iter().enumerate() {
            let hit = match pop {
                B::Eq  => pv == r, B::Neq => pv != r,
                B::Lt  => pv < r,  B::Lte => pv <= r,
                B::Gt  => pv > r,  B::Gte => pv >= r,
                _ => false,
            };
            if hit { out.push(m[i]); }
        }
        return Some(Ok(Val::int_vec(out)));
    }
    // Float pred + Int map: `filter(total > 500.0).map(id)` —
    // common bench shape (numeric thresholds with int IDs).
    // Promote int literal to f64 for cross-type compare.
    {
        let pred_f64 = match (pred_col, plit) {
            (ObjVecCol::Floats(p), Val::Float(r)) => Some((p, *r)),
            (ObjVecCol::Floats(p), Val::Int(r))   => Some((p, *r as f64)),
            _ => None,
        };
        if let Some((p, r)) = pred_f64 {
            // Map slot can be Int or Float; pick output.
            if let ObjVecCol::Ints(m) = map_col {
                let mut out: Vec<i64> = Vec::with_capacity(p.len());
                for (i, &pv) in p.iter().enumerate() {
                    let hit = match pop {
                        B::Eq  => pv == r, B::Neq => pv != r,
                        B::Lt  => pv < r,  B::Lte => pv <= r,
                        B::Gt  => pv > r,  B::Gte => pv >= r,
                        _ => false,
                    };
                    if hit { out.push(m[i]); }
                }
                return Some(Ok(Val::int_vec(out)));
            }
            if let ObjVecCol::Floats(m) = map_col {
                let mut out: Vec<f64> = Vec::with_capacity(p.len());
                for (i, &pv) in p.iter().enumerate() {
                    let hit = match pop {
                        B::Eq  => pv == r, B::Neq => pv != r,
                        B::Lt  => pv < r,  B::Lte => pv <= r,
                        B::Gt  => pv > r,  B::Gte => pv >= r,
                        _ => false,
                    };
                    if hit { out.push(m[i]); }
                }
                return Some(Ok(Val::float_vec(out)));
            }
            if let ObjVecCol::Strs(m) = map_col {
                let mut out: Vec<Arc<str>> = Vec::with_capacity(p.len());
                for (i, &pv) in p.iter().enumerate() {
                    let hit = match pop {
                        B::Eq  => pv == r, B::Neq => pv != r,
                        B::Lt  => pv < r,  B::Lte => pv <= r,
                        B::Gt  => pv > r,  B::Gte => pv >= r,
                        _ => false,
                    };
                    if hit { out.push(Arc::clone(&m[i])); }
                }
                return Some(Ok(Val::str_vec(out)));
            }
        }
    }
    // Int pred + Str map (e.g. `filter(total > 500).map(name)`).
    if let (ObjVecCol::Ints(p), ObjVecCol::Strs(m), Val::Int(rhs)) =
        (pred_col, map_col, plit)
    {
        let r = *rhs;
        let mut out: Vec<Arc<str>> = Vec::with_capacity(p.len());
        for (i, &pv) in p.iter().enumerate() {
            let hit = match pop {
                B::Eq  => pv == r, B::Neq => pv != r,
                B::Lt  => pv < r,  B::Lte => pv <= r,
                B::Gt  => pv > r,  B::Gte => pv >= r,
                _ => false,
            };
            if hit { out.push(Arc::clone(&m[i])); }
        }
        return Some(Ok(Val::str_vec(out)));
    }
    // Str pred (== / !=) + any map: status-style filter.
    if let (ObjVecCol::Strs(p), Val::Str(rhs)) = (pred_col, plit) {
        let r: &str = rhs.as_ref();
        let mut hits: Vec<usize> = Vec::with_capacity(p.len());
        for (i, ps) in p.iter().enumerate() {
            let hit = match pop {
                B::Eq  => ps.as_ref() == r,
                B::Neq => ps.as_ref() != r,
                _ => false,
            };
            if hit { hits.push(i); }
        }
        return Some(Ok(materialise_typed_indices(map_col, &hits)));
    }
    None
}

/// Typed columnar group_by: walk the key column directly, partition
/// row indices per distinct key, materialise per-group Val::Arr<Val::Obj>
/// (or a typed lane when all selected rows project the same shape).
/// Avoids per-row IndexMap probe + per-row key Arc clone of the walker
/// path.
fn objvec_typed_group_by(
    d: &Arc<crate::eval::value::ObjVecData>,
    key_field: &str,
) -> Option<Val> {
    use crate::eval::value::ObjVecCol;
    let cols = d.typed_cols.as_ref()?;
    let key_slot = d.slot_of(key_field)?;
    let key_col = cols.get(key_slot)?;
    let stride = d.stride();
    let nrows = d.nrows();

    // Partition row indices by key string.
    let mut groups: indexmap::IndexMap<Arc<str>, Vec<usize>> =
        indexmap::IndexMap::new();
    match key_col {
        ObjVecCol::Strs(c) => {
            // Use Arc::clone for repeated keys (Arc is interned-ish).
            for (i, s) in c.iter().enumerate() {
                groups.entry(Arc::clone(s)).or_default().push(i);
            }
        }
        ObjVecCol::Ints(c) => {
            for (i, n) in c.iter().enumerate() {
                let k: Arc<str> = Arc::from(n.to_string().as_str());
                groups.entry(k).or_default().push(i);
            }
        }
        ObjVecCol::Bools(c) => {
            for (i, b) in c.iter().enumerate() {
                let k: Arc<str> = if *b { Arc::from("true") } else { Arc::from("false") };
                groups.entry(k).or_default().push(i);
            }
        }
        _ => return None,
    };

    // Each group output as a sub-ObjVec sharing the parent key list +
    // gathered cells.  No per-row IndexMap construction — group rows
    // stay columnar, downstream pipeline ops can re-fire typed kernels
    // on the sub-ObjVec.
    use crate::eval::value::ObjVecData;
    let mut out: indexmap::IndexMap<Arc<str>, Val> = indexmap::IndexMap::with_capacity(groups.len());
    for (k, indices) in groups.into_iter() {
        let mut sub_cells: Vec<Val> = Vec::with_capacity(indices.len() * stride);
        for r in indices {
            let off = r * stride;
            for slot in 0..stride {
                sub_cells.push(d.cells[off + slot].clone());
            }
        }
        out.insert(k, Val::ObjVec(Arc::new(ObjVecData {
            keys: Arc::clone(&d.keys),
            cells: sub_cells,
            typed_cols: None,
        })));
    }
    let _ = nrows;
    Some(Val::Obj(Arc::new(out)))
}

/// Compose tape-walker accumulator output into a Val per the requested
/// numeric op.  Mirrors `num_finalise` semantics.
#[cfg(feature = "simd-json")]
// ── Generic kernel-tape evaluators ──────────────────────────────────
//
// Tape-side analogue of "evaluate a BodyKernel against a row".  Two
#[cfg(feature = "simd-json")]
fn binop_to_tape_cmp(op: crate::ast::BinOp) -> Option<crate::strref::TapeCmp> {
    use crate::ast::BinOp as B;
    use crate::strref::TapeCmp as T;
    Some(match op {
        B::Eq => T::Eq,
        B::Neq => T::Neq,
        B::Lt => T::Lt,
        B::Lte => T::Lte,
        B::Gt => T::Gt,
        B::Gte => T::Gte,
        _ => return None,
    })
}

#[cfg(feature = "simd-json")]
fn lit_to_tape_owned(v: &Val) -> Option<crate::composed::tape::TapeLitOwned> {
    use crate::composed::tape::TapeLitOwned;
    match v {
        Val::Int(n)   => Some(TapeLitOwned::Int(*n)),
        Val::Float(f) => Some(TapeLitOwned::Float(*f)),
        Val::Str(s)   => Some(TapeLitOwned::Str(Arc::clone(s))),
        Val::Bool(b)  => Some(TapeLitOwned::Bool(*b)),
        Val::Null     => Some(TapeLitOwned::Null),
        _ => None,
    }
}


fn materialise_typed_indices(
    col: &crate::eval::value::ObjVecCol,
    indices: &[usize],
) -> Val {
    use crate::eval::value::ObjVecCol;
    match col {
        ObjVecCol::Ints(c) => {
            let mut o: Vec<i64> = Vec::with_capacity(indices.len());
            for &i in indices { o.push(c[i]); }
            Val::int_vec(o)
        }
        ObjVecCol::Floats(c) => {
            let mut o: Vec<f64> = Vec::with_capacity(indices.len());
            for &i in indices { o.push(c[i]); }
            Val::float_vec(o)
        }
        ObjVecCol::Strs(c) => {
            let mut o: Vec<Arc<str>> = Vec::with_capacity(indices.len());
            for &i in indices { o.push(Arc::clone(&c[i])); }
            Val::str_vec(o)
        }
        ObjVecCol::Bools(c) => {
            let mut o: Vec<Val> = Vec::with_capacity(indices.len());
            for &i in indices { o.push(Val::Bool(c[i])); }
            Val::arr(o)
        }
        ObjVecCol::Mixed => Val::arr(Vec::new()),
    }
}

fn objvec_filter_count_and_slots(
    d: &Arc<crate::eval::value::ObjVecData>,
    leaves: &[(usize, crate::ast::BinOp, Val)],
) -> Val {
    use crate::eval::value::ObjVecCol;
    use crate::ast::BinOp as B;
    // Phase 7-typed-columns AND-chain path.  Pre-resolve each leaf to
    // a typed checker closure once; per row, run all checkers as
    // primitive comparisons over typed slices.  Skips per-leaf
    // cmp_val_binop_local + Val tag check on every row.
    if let Some(cols) = &d.typed_cols {
        // Snapshot each leaf as a "typed checker": closures over &[i64]
        // / &[f64] / &[Arc<str>] indexed by row.  Bail to scalar path
        // if any leaf can't be typed-resolved.
        enum Checker<'a> {
            IntsEq(&'a [i64], i64),
            IntsNeq(&'a [i64], i64),
            IntsLt(&'a [i64], i64),
            IntsLte(&'a [i64], i64),
            IntsGt(&'a [i64], i64),
            IntsGte(&'a [i64], i64),
            FloatsEq(&'a [f64], f64),
            FloatsNeq(&'a [f64], f64),
            FloatsLt(&'a [f64], f64),
            FloatsLte(&'a [f64], f64),
            FloatsGt(&'a [f64], f64),
            FloatsGte(&'a [f64], f64),
            StrsEq(&'a [Arc<str>], &'a str),
            StrsNeq(&'a [Arc<str>], &'a str),
            BoolsEq(&'a [bool], bool),
            BoolsNeq(&'a [bool], bool),
        }
        impl<'a> Checker<'a> {
            #[inline]
            fn at(&self, i: usize) -> bool {
                match *self {
                    Checker::IntsEq(c, r)   => c[i] == r,
                    Checker::IntsNeq(c, r)  => c[i] != r,
                    Checker::IntsLt(c, r)   => c[i] < r,
                    Checker::IntsLte(c, r)  => c[i] <= r,
                    Checker::IntsGt(c, r)   => c[i] > r,
                    Checker::IntsGte(c, r)  => c[i] >= r,
                    Checker::FloatsEq(c, r) => c[i] == r,
                    Checker::FloatsNeq(c, r)=> c[i] != r,
                    Checker::FloatsLt(c, r) => c[i] < r,
                    Checker::FloatsLte(c, r)=> c[i] <= r,
                    Checker::FloatsGt(c, r) => c[i] > r,
                    Checker::FloatsGte(c, r)=> c[i] >= r,
                    Checker::StrsEq(c, r)   => c[i].as_ref() == r,
                    Checker::StrsNeq(c, r)  => c[i].as_ref() != r,
                    Checker::BoolsEq(c, r)  => c[i] == r,
                    Checker::BoolsNeq(c, r) => c[i] != r,
                }
            }
        }
        let mut typed_checkers: Vec<Checker> = Vec::with_capacity(leaves.len());
        for (slot, op, lit) in leaves {
            let col = match cols.get(*slot) { Some(c) => c, None => break };
            let chk: Option<Checker> = match (col, lit, *op) {
                (ObjVecCol::Ints(c), Val::Int(r), B::Eq)  => Some(Checker::IntsEq(c.as_slice(), *r)),
                (ObjVecCol::Ints(c), Val::Int(r), B::Neq) => Some(Checker::IntsNeq(c.as_slice(), *r)),
                (ObjVecCol::Ints(c), Val::Int(r), B::Lt)  => Some(Checker::IntsLt(c.as_slice(), *r)),
                (ObjVecCol::Ints(c), Val::Int(r), B::Lte) => Some(Checker::IntsLte(c.as_slice(), *r)),
                (ObjVecCol::Ints(c), Val::Int(r), B::Gt)  => Some(Checker::IntsGt(c.as_slice(), *r)),
                (ObjVecCol::Ints(c), Val::Int(r), B::Gte) => Some(Checker::IntsGte(c.as_slice(), *r)),
                (ObjVecCol::Floats(c), Val::Float(r), B::Eq)  => Some(Checker::FloatsEq(c.as_slice(), *r)),
                (ObjVecCol::Floats(c), Val::Float(r), B::Neq) => Some(Checker::FloatsNeq(c.as_slice(), *r)),
                (ObjVecCol::Floats(c), Val::Float(r), B::Lt)  => Some(Checker::FloatsLt(c.as_slice(), *r)),
                (ObjVecCol::Floats(c), Val::Float(r), B::Lte) => Some(Checker::FloatsLte(c.as_slice(), *r)),
                (ObjVecCol::Floats(c), Val::Float(r), B::Gt)  => Some(Checker::FloatsGt(c.as_slice(), *r)),
                (ObjVecCol::Floats(c), Val::Float(r), B::Gte) => Some(Checker::FloatsGte(c.as_slice(), *r)),
                (ObjVecCol::Strs(c), Val::Str(r), B::Eq)  => Some(Checker::StrsEq(c.as_slice(), r.as_ref())),
                (ObjVecCol::Strs(c), Val::Str(r), B::Neq) => Some(Checker::StrsNeq(c.as_slice(), r.as_ref())),
                (ObjVecCol::Bools(c), Val::Bool(r), B::Eq)  => Some(Checker::BoolsEq(c.as_slice(), *r)),
                (ObjVecCol::Bools(c), Val::Bool(r), B::Neq) => Some(Checker::BoolsNeq(c.as_slice(), *r)),
                _ => None,
            };
            match chk {
                Some(c) => typed_checkers.push(c),
                None => { typed_checkers.clear(); break; }
            }
        }
        if typed_checkers.len() == leaves.len() {
            let nrows = d.nrows();
            let mut count: i64 = 0;
            'rows_typed: for row in 0..nrows {
                for c in &typed_checkers {
                    if !c.at(row) { continue 'rows_typed; }
                }
                count += 1;
            }
            return Val::Int(count);
        }
    }
    let stride = d.stride();
    let nrows = d.nrows();
    let mut count: i64 = 0;
    'rows: for row in 0..nrows {
        let off = row * stride;
        for (slot, op, lit) in leaves {
            if !cmp_val_binop_local(&d.cells[off + slot], *op, lit) {
                continue 'rows;
            }
        }
        count += 1;
    }
    Val::Int(count)
}

/// Total ordering over `Val` for sort barriers: numeric < string <
/// other; ties broken by debug-format equality so the comparator is
/// stable across calls.  Used only inside Pipeline::run barriers,
/// not exposed.
fn cmp_val_total(a: &Val, b: &Val) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let af = match a { Val::Int(n) => Some(*n as f64), Val::Float(x) => Some(*x), _ => None };
    let bf = match b { Val::Int(n) => Some(*n as f64), Val::Float(x) => Some(*x), _ => None };
    match (af, bf) {
        (Some(x), Some(y)) => x.partial_cmp(&y).unwrap_or(Ordering::Equal),
        _ => match (a, b) {
            (Val::Str(x), Val::Str(y)) => x.as_ref().cmp(y.as_ref()),
            _ => format!("{:?}", a).cmp(&format!("{:?}", b)),
        },
    }
}

/// Inline numeric/string comparison for the columnar path.  Mirrors
/// the semantics of the VM's existing `cmp_val_binop` helper but
/// accessible from this module.
#[inline]
fn cmp_val_binop_local(a: &Val, op: crate::ast::BinOp, b: &Val) -> bool {
    use crate::ast::BinOp;
    match (a, b) {
        (Val::Int(x), Val::Int(y))   => match op {
            BinOp::Eq => x == y, BinOp::Neq => x != y,
            BinOp::Lt => x <  y, BinOp::Lte => x <= y,
            BinOp::Gt => x >  y, BinOp::Gte => x >= y,
            _ => false,
        },
        (Val::Int(x), Val::Float(y)) => num_f_cmp(*x as f64, *y, op),
        (Val::Float(x), Val::Int(y)) => num_f_cmp(*x, *y as f64, op),
        (Val::Float(x), Val::Float(y)) => num_f_cmp(*x, *y, op),
        (Val::Str(x), Val::Str(y)) => match op {
            BinOp::Eq => x == y, BinOp::Neq => x != y,
            BinOp::Lt => x.as_ref() <  y.as_ref(),
            BinOp::Lte => x.as_ref() <= y.as_ref(),
            BinOp::Gt => x.as_ref() >  y.as_ref(),
            BinOp::Gte => x.as_ref() >= y.as_ref(),
            _ => false,
        },
        (Val::Bool(x), Val::Bool(y)) => match op {
            BinOp::Eq => x == y, BinOp::Neq => x != y, _ => false,
        },
        _ => false,
    }
}

#[inline]
fn num_f_cmp(a: f64, b: f64, op: crate::ast::BinOp) -> bool {
    use crate::ast::BinOp;
    match op {
        BinOp::Eq => a == b, BinOp::Neq => a != b,
        BinOp::Lt => a <  b, BinOp::Lte => a <= b,
        BinOp::Gt => a >  b, BinOp::Gte => a >= b,
        _ => false,
    }
}

/// IC-cached chain step.  Cached `Option<usize>` slot survives across
/// rows; missing-key marks slot None.  Used by Map(FieldChain) columnar
/// path to amortise the IndexMap probe cost across the whole array.
#[inline]
fn chain_step_ic(v: &Val, k: &str, ic: &mut Option<usize>) -> Val {
    match v {
        Val::Obj(m) => match lookup_via_ic(m, k, ic) {
            Some(x) => x.clone(),
            None => Val::Null,
        },
        _ => v.get_field(k),
    }
}

fn lookup_via_ic<'a>(
    m: &'a indexmap::IndexMap<Arc<str>, Val>,
    k: &str,
    cached: &mut Option<usize>,
) -> Option<&'a Val> {
    if let Some(i) = *cached {
        if let Some((ki, vi)) = m.get_index(i) {
            if ki.as_ref() == k { return Some(vi); }
        }
    }
    match m.get_full(k) {
        Some((i, _, v)) => { *cached = Some(i); Some(v) }
        None => { *cached = None; None }
    }
}

/// Compile a `.filter(arg)` / `.map(arg)` sub-expression into a `Program`
/// the VM can run against a row's `@`.  The argument's expression is
/// wrapped so that bare-ident shorthand (`map(total)`) becomes
/// `@.total` for evaluation against the current row — the tree-walker
/// applies the same rule via `apply_item_mut` but we have to be
/// explicit when emitting bytecode.
fn compile_subexpr(arg: &crate::ast::Arg) -> Option<Arc<crate::vm::Program>> {
    use crate::ast::{Arg, Expr, Step};
    let inner = match arg { Arg::Pos(e) => e, _ => return None };
    let rooted: Expr = match inner {
        // Bare ident `total` → `@.total`
        Expr::Ident(name) => Expr::Chain(
            Box::new(Expr::Current),
            vec![Step::Field(name.clone())],
        ),
        // `@…` chains: keep base = Current, accept as-is
        Expr::Chain(base, _) if matches!(base.as_ref(), Expr::Current) => inner.clone(),
        // Anything else: wrap as-is — VM will resolve `@` via Current refs.
        other => other.clone(),
    };
    Some(Arc::new(crate::vm::Compiler::compile(&rooted, "")))
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Walk `$.k1.k2…` from `root`, returning `Val::Null` on any miss.
/// Used by `Source::FieldChain` resolution.
fn walk_field_chain(root: &Val, keys: &[Arc<str>]) -> Val {
    let mut cur = root.clone();
    for k in keys {
        cur = cur.get_field(k.as_ref());
    }
    cur
}

/// Evaluate `prog` against `item` as the VM root using a long-lived
/// VM borrowed from the caller (Pipeline::run owns one per query).
/// Per-row apply.  Phase A1 fast path: rebinds the loop-shared Env's
/// `current` slot in place (one swap, two assignments) and runs the
/// stage program directly via `exec_in_env`.  Skips doc-hash recompute,
/// root_chain_cache clear, and Env construction that
/// `execute_val_raw` does on every call.  Saves ~80 ns/row.
#[inline]
fn apply_item_in_env(
    vm: &mut crate::vm::VM,
    env: &mut crate::eval::Env,
    item: &Val,
    prog: &crate::vm::Program,
) -> Result<Val, EvalError> {
    let prev = env.swap_current(item.clone());
    let r = vm.exec_in_env(prog, env);
    let _ = env.swap_current(prev);
    r
}

/// Phase A3 — per-row evaluation that consults a pre-classified kernel
/// hint and dispatches the specialised inline path when recognised.
/// Falls back to `apply_item_in_env` for Generic / unknown shapes.
/// Same semantics as `apply_item_in_env(prog)` for every kernel kind.
#[inline]
fn eval_kernel(
    kernel: &BodyKernel,
    item: &Val,
    vm: &mut crate::vm::VM,
    env: &mut crate::eval::Env,
    fallback: &crate::vm::Program,
) -> Result<Val, EvalError> {
    match kernel {
        BodyKernel::FieldRead(k) => Ok(item.get_field(k.as_ref())),
        BodyKernel::FieldChain(ks) => {
            let mut v = item.clone();
            for k in ks.iter() {
                v = v.get_field(k.as_ref());
                if matches!(v, Val::Null) { break; }
            }
            Ok(v)
        }
        BodyKernel::ConstBool(b) => Ok(Val::Bool(*b)),
        BodyKernel::Const(v) => Ok(v.clone()),
        BodyKernel::FieldCmpLit(k, op, lit) => {
            let lhs = item.get_field(k.as_ref());
            Ok(Val::Bool(eval_cmp_op(&lhs, *op, lit)))
        }
        BodyKernel::FieldChainCmpLit(ks, op, lit) => {
            let mut v = item.clone();
            for k in ks.iter() {
                v = v.get_field(k.as_ref());
                if matches!(v, Val::Null) { break; }
            }
            Ok(Val::Bool(eval_cmp_op(&v, *op, lit)))
        }
        BodyKernel::CurrentCmpLit(op, lit) =>
            Ok(Val::Bool(eval_cmp_op(item, *op, lit))),
        BodyKernel::ObjProject(_) | BodyKernel::Arith(_, _, _)
        | BodyKernel::FString(_) | BodyKernel::Generic =>
            apply_item_in_env(vm, env, item, fallback),
    }
}

/// Evaluate a compare-op against a value pair.  Mirrors VM's Eq/Lt/etc.
/// handlers; centralised so the kernel inline path matches semantics.
#[inline]
fn eval_cmp_op(lhs: &Val, op: crate::ast::BinOp, rhs: &Val) -> bool {
    use crate::ast::BinOp as B;
    use crate::eval::util::{vals_eq, cmp_vals};
    match op {
        B::Eq  => vals_eq(lhs, rhs),
        B::Neq => !vals_eq(lhs, rhs),
        B::Lt  => cmp_vals(lhs, rhs) == std::cmp::Ordering::Less,
        B::Lte => cmp_vals(lhs, rhs) != std::cmp::Ordering::Greater,
        B::Gt  => cmp_vals(lhs, rhs) == std::cmp::Ordering::Greater,
        B::Gte => cmp_vals(lhs, rhs) != std::cmp::Ordering::Less,
        _ => false,
    }
}

#[inline]
fn is_truthy(v: &Val) -> bool {
    match v {
        Val::Null            => false,
        Val::Bool(b)         => *b,
        Val::Int(n)          => *n != 0,
        Val::Float(f)        => *f != 0.0,
        Val::Str(s)          => !s.is_empty(),
        Val::StrSlice(r)     => !r.as_str().is_empty(),
        Val::Arr(a)          => !a.is_empty(),
        Val::IntVec(a)       => !a.is_empty(),
        Val::FloatVec(a)     => !a.is_empty(),
        Val::StrVec(a)       => !a.is_empty(),
        Val::StrSliceVec(a)  => !a.is_empty(),
        Val::Obj(m)          => !m.is_empty(),
        Val::ObjSmall(p)     => !p.is_empty(),
        Val::ObjVec(d)       => !d.cells.is_empty(),
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser;

    /// Skip body when JETRO_COMPOSED=1 — the rewrite-shape tests below
    /// assert on fused-Sink output, which the composed gate disables
    /// by design. Tier 3 deletes both the fused sinks and these
    /// shape-asserting tests; until then they only run in the default
    /// (legacy) configuration.
    fn skip_under_composed() -> bool {
        super::composed_path_enabled()
    }

    fn lower_query(q: &str) -> Option<Pipeline> {
        let expr = parser::parse(q).ok()?;
        Pipeline::lower(&expr)
    }

    #[test]
    fn lower_field_chain_only() {
        let p = lower_query("$.a.b.c").unwrap();
        assert!(matches!(p.source, Source::FieldChain { .. }));
        assert!(p.stages.is_empty());
        assert!(matches!(p.sink, Sink::Collect));
    }

    // `lower_filter_map_count` removed — fused Sink::CountIf variant
    // deleted in Tier 3. Lowered shape is now [Filter] + Sink::Count.

    #[test]
    fn lower_take_skip_sum() {
        let p = lower_query("$.xs.skip(2).take(5).sum()").unwrap();
        assert_eq!(p.stages.len(), 2);
        assert!(matches!(p.stages[0], Stage::Skip(2)));
        assert!(matches!(p.stages[1], Stage::Take(5)));
        assert!(matches!(p.sink, Sink::Numeric(NumOp::Sum)));
    }

    #[test]
    fn lower_returns_none_for_unsupported_shape() {
        // group_by now supported as a Stage; verify a different
        // unsupported shape stays None.
        assert!(lower_query("$.xs.equi_join($.ys, lhs, rhs)").is_none());
        // Non-root base.
        assert!(lower_query("@.x.filter(y > 0)").is_none());
    }

    #[test]
    fn debug_filter_pred_shape() {
        let expr = crate::parser::parse("@.total > 100").unwrap();
        let prog = crate::vm::Compiler::compile(&expr, "");
        eprintln!("PRED OPS = {:#?}", prog.ops);
    }

    // `debug_compound_pipeline_lower` and `debug_full_pipeline_lower`
    // removed — referenced fused Sink::CountIf / Sink::NumFilterMap.

    #[test]
    fn run_count_on_simple_array() {
        use serde_json::json;
        let doc: Val = (&json!({"orders":[
            {"total": 50}, {"total": 150}, {"total": 200}
        ]})).into();
        let p = lower_query("$.orders.filter(total > 100).count()").unwrap();
        let out = p.run(&doc).unwrap();
        assert_eq!(out, Val::Int(2));
    }

    #[test]
    fn run_sum_on_simple_array() {
        use serde_json::json;
        let doc: Val = (&json!({"xs":[1, 2, 3, 4, 5]})).into();
        let p = lower_query("$.xs.sum()").unwrap();
        let out = p.run(&doc).unwrap();
        assert_eq!(out, Val::Int(15));
    }

    #[test]
    fn run_filter_map_sum() {
        use serde_json::json;
        let doc: Val = (&json!({"orders":[
            {"id": 1, "total": 50},
            {"id": 2, "total": 150},
            {"id": 3, "total": 200}
        ]})).into();
        let p = lower_query("$.orders.filter(total > 100).map(total).sum()").unwrap();
        let out = p.run(&doc).unwrap();
        assert_eq!(out, Val::Int(350));
    }

    #[test]
    fn rewrite_skip_skip_merges() {
        let p = lower_query("$.xs.skip(2).skip(3).sum()").unwrap();
        assert_eq!(p.stages.len(), 1);
        assert!(matches!(p.stages[0], Stage::Skip(5)));
    }

    #[test]
    fn rewrite_take_take_merges_min() {
        let p = lower_query("$.xs.take(10).take(3).sum()").unwrap();
        assert_eq!(p.stages.len(), 1);
        assert!(matches!(p.stages[0], Stage::Take(3)));
    }

    #[test]
    fn rewrite_filter_filter_merges_via_andop() {
        let p = lower_query("$.orders.filter(total > 100).filter(qty > 0).sum()").unwrap();
        assert_eq!(p.stages.len(), 1);
        match &p.stages[0] {
            Stage::Filter(prog) => {
                assert!(prog.ops.iter().any(|o| matches!(o, crate::vm::Opcode::AndOp(_))));
            }
            _ => panic!("expected merged Filter"),
        }
    }

    #[test]
    fn rewrite_map_then_count_drops_map() {
        if skip_under_composed() { return; }
        let p = lower_query("$.orders.map(total).count()").unwrap();
        assert_eq!(p.stages.len(), 0);
        assert!(matches!(p.sink, Sink::Count));
    }

    // `rewrite_take_after_map_pushdown`, `rewrite_sort_take_to_topn`,
    // `rewrite_sort_by_first_to_minby`, `rewrite_sort_by_last_to_maxby`
    // removed — fused Sink::NumMap/TopN/MinBy/MaxBy variants deleted.

    #[test]
    fn run_topn_smallest_three() {
        use serde_json::json;
        let doc: Val = (&json!({"xs":[5, 2, 8, 1, 4, 7, 3]})).into();
        let p = lower_query("$.xs.sort().take(3)").unwrap();
        let out = p.run(&doc).unwrap();
        // Compare via JSON to avoid Val::Arr vs Val::IntVec variant mismatch.
        let out_json: serde_json::Value = out.into();
        assert_eq!(out_json, json!([1, 2, 3]));
    }

    #[test]
    fn rewrite_filter_const_true_dropped() {
        // `true` literal — Filter(true) collapses to id.
        let p = lower_query("$.xs.filter(true).count()").unwrap();
        assert_eq!(p.stages.len(), 0);
        assert!(matches!(p.sink, Sink::Count));
    }

    #[test]
    fn run_take_skip() {
        use serde_json::json;
        let doc: Val = (&json!({"xs":[10, 20, 30, 40, 50]})).into();
        let p = lower_query("$.xs.skip(1).take(2).sum()").unwrap();
        let out = p.run(&doc).unwrap();
        assert_eq!(out, Val::Int(50));   // 20 + 30
    }
}
