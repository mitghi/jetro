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
        Sink::NumMap(_, _) => "num_map",
        Sink::CountIf(_) => "count_if",
        Sink::NumFilterMap(_, _, _) => "num_filter_map",
        Sink::FilterFirst(_) => "filter_first",
        Sink::FilterLast(_) => "filter_last",
        Sink::FirstMap(_) => "first_map",
        Sink::LastMap(_) => "last_map",
        Sink::FlatMapCount(_) => "flat_map_count",
        Sink::TopN { .. } => "top_n",
        Sink::MinBy(_) => "min_by",
        Sink::MaxBy(_) => "max_by",
        Sink::UniqueCount => "unique_count",
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
fn classify_kv_method(
    key: &Arc<str>,
    prog: &crate::vm::Program,
) -> Option<ObjProjEntry> {
    use crate::vm::{Opcode, BuiltinMethod};
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
        Opcode::MapSum(inner_prog) => {
            let inner = BodyKernel::classify(inner_prog);
            // Only path / arith inner kernels supported on tape.
            if !matches!(inner,
                BodyKernel::FieldRead(_)
                | BodyKernel::FieldChain(_)
                | BodyKernel::Arith(_, _, _))
            { return None; }
            Some(ObjProjEntry::InnerMapSum {
                key: key.clone(),
                path,
                inner: Arc::new(inner),
            })
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
/// The fused variants (`NumMap`, `NumFilterMap`, `CountIf`,
/// `FilterFirst`, `FilterLast`) are produced by the Phase 2 rewrite
/// pass during lowering — collapsing adjacent `Map` / `Filter`
/// stages plus the sink into a single inner-loop kernel that
/// eliminates the per-row VM exec dispatch per stage.
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
    /// `Map(prog) ∘ <sum/min/max/avg>`.  Inner loop computes `prog(@)`
    /// and accumulates numerically; never materialises a Vec.
    NumMap(NumOp, Arc<crate::vm::Program>),
    /// `Filter(prog) ∘ Count`.  Inner loop runs `prog(@)`; counts truthy.
    CountIf(Arc<crate::vm::Program>),
    /// `Filter(pred) ∘ Map(f) ∘ <sum/min/max/avg>`.
    NumFilterMap(NumOp, Arc<crate::vm::Program>, Arc<crate::vm::Program>),
    /// `Filter(prog) ∘ First` — find first element where pred holds.
    FilterFirst(Arc<crate::vm::Program>),
    /// `Filter(prog) ∘ Last` — find last element where pred holds.
    FilterLast(Arc<crate::vm::Program>),
    /// `Map(prog) ∘ First` — first element after projection.
    FirstMap(Arc<crate::vm::Program>),
    /// `Map(prog) ∘ Last` — last element after projection.
    LastMap(Arc<crate::vm::Program>),
    /// `FlatMap(prog) ∘ Count` — count yielded inner elements without
    /// materialising the flat sequence.  Closes pipeline_ir.md
    /// bench-priority item #1 (`flat_map+filter+count` 44× gap).
    FlatMapCount(Arc<crate::vm::Program>),
    /// `Sort ∘ Take(n)` → top-N.  Heap-based partial sort: keeps the
    /// k smallest items via a max-heap of size k.  Asymptotic cost
    /// O(N log k) vs O(N log N) for full sort.  Per pipeline_ir.md.
    /// `asc=true`  → smallest n  (natural ordering, default)
    /// `asc=false` → largest n   (used by `Sort+Reverse+Take` after the
    ///                            Reverse∘Reverse rule cancels)
    TopN { n: usize, asc: bool, key: Option<Arc<crate::vm::Program>> },
    /// `Sort_by(k) ∘ First` → keep argmin by key.  One pass, no full sort.
    MinBy(Arc<crate::vm::Program>),
    /// `Sort_by(k) ∘ Last` → keep argmax by key.
    MaxBy(Arc<crate::vm::Program>),
    /// `Unique ∘ Count` — count distinct elements via HashSet, no
    /// intermediate dedup array materialised.
    UniqueCount,
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
        use crate::ast::{Step, Arg};
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
        let mut stages: Vec<Stage> = Vec::new();
        let mut sink: Sink = Sink::Collect;
        let trailing = &steps[field_end..];
        for (i, s) in trailing.iter().enumerate() {
            let is_last = i == trailing.len() - 1;
            match s {
                Step::Method(name, args) => {
                    match (name.as_str(), args.len(), is_last) {
                        ("filter", 1, _) => {
                            let prog = compile_subexpr(&args[0])?;
                            stages.push(Stage::Filter(prog));
                        }
                        ("map", 1, _) => {
                            let prog = compile_subexpr(&args[0])?;
                            stages.push(Stage::Map(prog));
                        }
                        ("flat_map", 1, _) => {
                            let prog = compile_subexpr(&args[0])?;
                            stages.push(Stage::FlatMap(prog));
                        }
                        ("reverse", 0, _) => stages.push(Stage::Reverse),
                        ("unique", 0, _) => stages.push(Stage::UniqueBy(None)),
                        ("unique_by", 1, _) => {
                            let prog = compile_subexpr(&args[0])?;
                            stages.push(Stage::UniqueBy(Some(prog)));
                        }
                        ("group_by", 1, _) => {
                            let prog = compile_subexpr(&args[0])?;
                            stages.push(Stage::GroupBy(prog));
                        }
                        ("sort", 0, _) => stages.push(Stage::Sort(None)),
                        ("sort_by", 1, _) => {
                            let prog = compile_subexpr(&args[0])?;
                            stages.push(Stage::Sort(Some(prog)));
                        }
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
                        ("count", 0, true) | ("len", 0, true) => sink = Sink::Count,
                        ("sum", 0, true) => sink = Sink::Numeric(NumOp::Sum),
                        ("min", 0, true) => sink = Sink::Numeric(NumOp::Min),
                        ("max", 0, true) => sink = Sink::Numeric(NumOp::Max),
                        ("avg", 0, true) => sink = Sink::Numeric(NumOp::Avg),
                        ("first", 0, true) => sink = Sink::First,
                        ("last", 0, true) => sink = Sink::Last,
                        _ => return None,
                    }
                }
                _ => return None,
            }
        }

        let mut p = Pipeline {
            source: Source::FieldChain { keys },
            stages, sink,
            stage_kernels: Vec::new(),
            sink_kernels:  Vec::new(),
        };
        rewrite(&mut p);
        // Phase A3 — classify per-stage and per-sink sub-programs once
        // post-rewrite.  Per-row pull loop reads these hints to choose
        // a specialised inline path vs the generic vm.exec fallback.
        p.stage_kernels = p.stages.iter().map(|s| match s {
            Stage::Filter(p)        => BodyKernel::classify(p),
            Stage::Map(p)           => BodyKernel::classify(p),
            Stage::FlatMap(p)       => BodyKernel::classify(p),
            Stage::UniqueBy(Some(p))=> BodyKernel::classify(p),
            Stage::GroupBy(p)       => BodyKernel::classify(p),
            Stage::Sort(Some(p))    => BodyKernel::classify(p),
            _                       => BodyKernel::Generic,
        }).collect();
        p.sink_kernels = match &p.sink {
            Sink::NumMap(_, p) | Sink::CountIf(p)
            | Sink::FilterFirst(p) | Sink::FilterLast(p)
            | Sink::FirstMap(p) | Sink::LastMap(p)
            | Sink::FlatMapCount(p) | Sink::MinBy(p) | Sink::MaxBy(p) =>
                vec![BodyKernel::classify(p)],
            Sink::NumFilterMap(_, pred, map) =>
                vec![BodyKernel::classify(pred), BodyKernel::classify(map)],
            Sink::TopN { key: Some(k), .. } => vec![BodyKernel::classify(k)],
            _ => Vec::new(),
        };
        Some(p)
    }
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
fn rewrite_step(p: &mut Pipeline) -> bool {
    use crate::vm::Opcode;

    // ── Constant-fold rules on Filter ────────────────────────────────────
    //
    // Rule:  Filter(true)  → id        (drop the stage)
    // Rule:  Filter(false) → Empty     (replace remaining pipeline with
    //                                   the sink's empty-input result)
    let mut const_remove_at: Option<(usize, bool)> = None;
    for (i, s) in p.stages.iter().enumerate() {
        if let Stage::Filter(prog) = s {
            if let Some(b) = prog_const_bool(prog) {
                const_remove_at = Some((i, b));
                break;
            }
        }
    }
    if let Some((i, b)) = const_remove_at {
        if b {
            p.stages.remove(i);
        } else {
            // Filter(false) — pipeline yields zero elements.
            p.stages.clear();
            p.sink = empty_sink_for(&p.sink);
        }
        return true;
    }

    // ── Adjacent-stage merges ───────────────────────────────────────────
    //
    // Rule:  Skip(a) ∘ Skip(b) → Skip(a + b)
    // Rule:  Take(a) ∘ Take(b) → Take(min(a, b))
    // Rule:  Reverse ∘ Reverse → id
    // Rule:  Sort(_) ∘ Sort(k) → Sort(k)        (idempotent)
    // Rule:  UniqueBy(_) ∘ UniqueBy(k) → UniqueBy(k)
    // Rule:  Filter(p) ∘ Filter(q) → Filter(p ∧ q)
    for i in 0..p.stages.len().saturating_sub(1) {
        match (&p.stages[i], &p.stages[i + 1]) {
            (Stage::Skip(a), Stage::Skip(b)) => {
                let merged = a.saturating_add(*b);
                p.stages[i] = Stage::Skip(merged);
                p.stages.remove(i + 1);
                return true;
            }
            (Stage::Take(a), Stage::Take(b)) => {
                let merged = (*a).min(*b);
                p.stages[i] = Stage::Take(merged);
                p.stages.remove(i + 1);
                return true;
            }
            (Stage::Reverse, Stage::Reverse) => {
                // Reverse ∘ Reverse cancels — drop both stages.
                p.stages.drain(i..=i + 1);
                return true;
            }
            (Stage::Sort(_), Stage::Sort(_)) => {
                // The right-most sort wins (idempotent for same key,
                // overrides for different key).
                p.stages.remove(i);
                return true;
            }
            (Stage::UniqueBy(_), Stage::UniqueBy(_)) => {
                p.stages.remove(i);
                return true;
            }
            // Phase B3 — kernel-level loop fusion: Map(field-read) ∘
            // Map(field-read) collapses to one FieldChain Map.  This
            // is mechanical translation via the kernel classifier —
            // no per-shape hand-coding.  Same primitive covers chain
            // ∘ read / read ∘ chain / chain ∘ chain.
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
                // Combine via VM-level AndOp embedding; avoids
                // re-compiling the merged predicate.
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

    // ── Sort-into-aggregate fold rules ───────────────────────────────────
    //
    // Rule:  Sort ∘ First           → Min          (natural-cmp sort)
    // Rule:  Sort ∘ Last            → Max
    // Rule:  Sort_by(k) ∘ First     → MinBy(k)     (one-pass argmin)
    // Rule:  Sort_by(k) ∘ Last      → MaxBy(k)     (one-pass argmax)
    // Rule:  Sort(_) ∘ Take(n)      → TopN{n}      (heap-based partial
    //                                                sort, O(N log n))
    if matches!(&p.sink, Sink::First) {
        match p.stages.last() {
            Some(Stage::Sort(None)) => {
                p.stages.pop();
                p.sink = Sink::Numeric(NumOp::Min);
                return true;
            }
            Some(Stage::Sort(Some(key_prog))) => {
                let key = Arc::clone(key_prog);
                p.stages.pop();
                p.sink = Sink::MinBy(key);
                return true;
            }
            _ => {}
        }
    }
    if matches!(&p.sink, Sink::Last) {
        match p.stages.last() {
            Some(Stage::Sort(None)) => {
                p.stages.pop();
                p.sink = Sink::Numeric(NumOp::Max);
                return true;
            }
            Some(Stage::Sort(Some(key_prog))) => {
                let key = Arc::clone(key_prog);
                p.stages.pop();
                p.sink = Sink::MaxBy(key);
                return true;
            }
            _ => {}
        }
    }
    // Sort ∘ Take(n) → TopN.  The Take stage at the end of a stage
    // list with a Collect sink fuses into TopN-collect.
    if matches!(&p.sink, Sink::Collect) && p.stages.len() >= 2 {
        let last = p.stages.len() - 1;
        let prev = last - 1;
        if let (Stage::Sort(key), Stage::Take(n)) = (&p.stages[prev], &p.stages[last]) {
            let n = *n;
            let key = key.clone();
            p.stages.truncate(prev);
            p.sink = Sink::TopN { n, asc: true, key };
            return true;
        }
    }

    // ── Pushdown: Map(f) ∘ Take(n)  →  Take(n) ∘ Map(f) ────────────────
    // Run the map only on the first `n` items.  Stage order in `stages`
    // is left-to-right (apply 0 first), so detect [Map, Take] and
    // swap to [Take, Map].
    for i in 0..p.stages.len().saturating_sub(1) {
        if matches!(&p.stages[i], Stage::Map(_))
            && matches!(&p.stages[i + 1], Stage::Take(_)) {
            p.stages.swap(i, i + 1);
            return true;
        }
    }

    // ── Drop pure Map before Count ─────────────────────────────────────
    // Rule:  Map(f) ∘ Count → Count
    if matches!(&p.sink, Sink::Count) {
        if matches!(p.stages.last(), Some(Stage::Map(_))) {
            p.stages.pop();
            return true;
        }
    }

    // ── Fused-sink rules over numeric aggregates (sum/min/max/avg) ─────
    let last_two = if p.stages.len() >= 2 {
        Some((p.stages.len() - 2, p.stages.len() - 1))
    } else { None };

    // Rule:  Filter(p) ∘ Map(f) ∘ Numeric(op)  →  NumFilterMap(op, p, f)
    if let (Some((i_pred, i_map)), Sink::Numeric(op)) = (last_two, &p.sink) {
        if let (Stage::Filter(pred), Stage::Map(map)) =
            (&p.stages[i_pred], &p.stages[i_map])
        {
            let op   = *op;
            let pred = Arc::clone(pred);
            let map  = Arc::clone(map);
            p.stages.truncate(i_pred);
            p.sink = Sink::NumFilterMap(op, pred, map);
            return true;
        }
    }

    // Rule:  Map(f) ∘ Numeric(op) → NumMap(op, f)
    if let (Some(last), Sink::Numeric(op)) = (p.stages.last(), &p.sink) {
        if let Stage::Map(prog) = last {
            let op   = *op;
            let prog = Arc::clone(prog);
            p.stages.pop();
            p.sink = Sink::NumMap(op, prog);
            return true;
        }
    }

    // Rule:  Filter(p) ∘ Count → CountIf(p)
    if let (Some(last), Sink::Count) = (p.stages.last(), &p.sink) {
        if let Stage::Filter(prog) = last {
            let prog = Arc::clone(prog);
            p.stages.pop();
            p.sink = Sink::CountIf(prog);
            return true;
        }
    }

    // Rule:  Filter(p) ∘ First → FilterFirst(p)
    if let (Some(last), Sink::First) = (p.stages.last(), &p.sink) {
        if let Stage::Filter(prog) = last {
            let prog = Arc::clone(prog);
            p.stages.pop();
            p.sink = Sink::FilterFirst(prog);
            return true;
        }
    }

    // Rule:  Filter(p) ∘ Last → FilterLast(p)
    if let (Some(last), Sink::Last) = (p.stages.last(), &p.sink) {
        if let Stage::Filter(prog) = last {
            let prog = Arc::clone(prog);
            p.stages.pop();
            p.sink = Sink::FilterLast(prog);
            return true;
        }
    }

    // Rule:  Map(f) ∘ First → FirstMap(f)
    if let (Some(last), Sink::First) = (p.stages.last(), &p.sink) {
        if let Stage::Map(prog) = last {
            let prog = Arc::clone(prog);
            p.stages.pop();
            p.sink = Sink::FirstMap(prog);
            return true;
        }
    }

    // Rule:  Map(f) ∘ Last → LastMap(f)
    if let (Some(last), Sink::Last) = (p.stages.last(), &p.sink) {
        if let Stage::Map(prog) = last {
            let prog = Arc::clone(prog);
            p.stages.pop();
            p.sink = Sink::LastMap(prog);
            return true;
        }
    }

    // Rule:  Unique ∘ Count → UniqueCount
    if let (Some(last), Sink::Count) = (p.stages.last(), &p.sink) {
        if matches!(last, Stage::UniqueBy(None)) {
            p.stages.pop();
            p.sink = Sink::UniqueCount;
            return true;
        }
    }

    // Rule:  FlatMap(f) ∘ Count → FlatMapCount(f)
    // Closes pipeline_ir.md bench item #1 (44× gap).  No need to
    // materialise the flattened sequence — kernel pulls each yielded
    // inner element and increments a counter in one pass.
    if let (Some(last), Sink::Count) = (p.stages.last(), &p.sink) {
        if let Stage::FlatMap(prog) = last {
            let prog = Arc::clone(prog);
            p.stages.pop();
            p.sink = Sink::FlatMapCount(prog);
            return true;
        }
    }

    false
}

/// Return the result a sink would produce on an empty pull stream.
/// Used by the `Filter(false)` rewrite to short-circuit the pipeline.
/// Sums emit `Int(0)`, counts emit `Int(0)`, collect emits `[]`.
fn empty_sink_for(sink: &Sink) -> Sink {
    match sink {
        // All current sinks have a well-defined empty-input result;
        // none of them needs special-casing here.  Clone the sink so
        // the run loop's already-zeroed accumulators produce the
        // right shape.
        _ => sink.clone(),
    }
    // (Empty input ⇒ existing accumulators in `Pipeline::run` already
    // produce Int(0) / Float(0.0) / Val::arr([]).  No need for a
    // separate Empty sentinel — clearing `stages` suffices because the
    // outer iter has already been built before this runs in `run`.
    // The clone here is kept for ABI clarity.)
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

    /// Tape-only aggregate path — single generic executor over a tape
    /// Array.  All Sink shapes (Count / Numeric / NumMap / NumFilterMap /
    /// CountIf / FilterFirst / future sinks) plug into the same
    /// per-entry walker; per-Sink semantics live in `feed_sink`.
    /// Adding a new Sink ⇒ one match arm in `feed_sink` + accumulator
    /// state, no new tape walker.  Adding a new BodyKernel variant ⇒
    /// two match arms in `eval_kernel_value` / `eval_kernel_pred`.
    /// No per-(Stage_chain, Sink) helpers.
    ///
    /// Active when:
    ///   - cache provides a tape (cold-start path), AND
    ///   - cache.prefer_tape() is true (Val tree not yet built), AND
    ///   - source is FieldChain landing on an Array.
    ///
    /// Stages must currently be empty — Pipeline IR's rewrite pass
    /// folds Filter/Map combinations into fused sinks (NumFilterMap,
    /// CountIf, FilterFirst, etc.) before lowering, so the empty-stage
    /// constraint covers all bench shapes; the rewrite is the structural
    /// fusion, not per-shape tape code.
    #[cfg(feature = "simd-json")]
    fn try_tape_aggregate(
        &self,
        cache: Option<&dyn ObjVecPromoter>,
    ) -> Option<Result<Val, EvalError>> {
        let cache = cache?;
        if !cache.prefer_tape() { return None; }
        let tape = cache.tape()?;
        let chain_keys = match &self.source {
            Source::FieldChain { keys } => keys,
            _ => return None,
        };
        let key_strs: Vec<&str> = chain_keys.iter().map(|k| k.as_ref()).collect();
        let arr_idx = crate::strref::tape_walk_field_chain(tape, &key_strs)?;

        // Stage classifier:
        //   (FlatMap?, Filter*, Sort?, Skip?, Take?, Map?, UniqueBy?, Reverse?)
        // Sort barriers buffer rows; sort key is kernel-extracted.
        let mut flat_kernel: Option<&BodyKernel> = None;
        // Outer filters apply pre-FlatMap (to outer entries).
        // Inner filters apply post-FlatMap (to inner entries).
        // Without FlatMap, all filters are "inner" (one walk).
        let mut outer_filter_kernels: Vec<&BodyKernel> = Vec::new();
        let mut filter_kernels: Vec<&BodyKernel> = Vec::new();
        let mut sort_kernel: Option<&BodyKernel> = None;
        let mut sort_asc: bool = true;
        let mut skip_n: usize = 0;
        let mut take_n: Option<usize> = None;
        let mut map_kernel: Option<&BodyKernel> = None;
        let mut unique_dedup = false;
        let mut reverse = false;
        for (st, k) in self.stages.iter().zip(self.stage_kernels.iter()) {
            if reverse { return None; }
            match st {
                Stage::FlatMap(_) => {
                    if flat_kernel.is_some() { return None; }  // one FlatMap
                    if sort_kernel.is_some() || map_kernel.is_some() || skip_n > 0 || take_n.is_some() { return None; }
                    if !matches!(k,
                        BodyKernel::FieldRead(_) | BodyKernel::FieldChain(_))
                    { return None; }
                    flat_kernel = Some(k);
                    // Pre-FlatMap filters move into outer_filter_kernels.
                    std::mem::swap(&mut outer_filter_kernels, &mut filter_kernels);
                    filter_kernels.clear();
                }
                Stage::Filter(_) => {
                    if sort_kernel.is_some() || map_kernel.is_some() || take_n.is_some() { return None; }
                    if !matches!(k,
                        BodyKernel::FieldRead(_)
                        | BodyKernel::FieldCmpLit(_, _, _)
                        | BodyKernel::FieldChainCmpLit(_, _, _)
                        | BodyKernel::ConstBool(_))
                    { return None; }
                    filter_kernels.push(k);
                }
                Stage::Sort(Some(_)) => {
                    if sort_kernel.is_some() || map_kernel.is_some() { return None; }
                    if !matches!(k,
                        BodyKernel::FieldRead(_)
                        | BodyKernel::FieldChain(_)
                        | BodyKernel::Arith(_, _, _))
                    { return None; }
                    sort_kernel = Some(k);
                }
                Stage::Sort(None) => return None,
                Stage::Skip(n) => {
                    if map_kernel.is_some() || take_n.is_some() { return None; }
                    skip_n = skip_n.saturating_add(*n);
                }
                Stage::Take(n) => {
                    if map_kernel.is_some() { return None; }
                    take_n = Some(take_n.map(|m| m.min(*n)).unwrap_or(*n));
                }
                Stage::Map(_) => {
                    if map_kernel.is_some() { return None; }
                    if !matches!(k,
                        BodyKernel::FieldRead(_)
                        | BodyKernel::FieldChain(_)
                        | BodyKernel::ObjProject(_)
                        | BodyKernel::Arith(_, _, _)
                        | BodyKernel::FString(_))
                    { return None; }
                    map_kernel = Some(k);
                }
                Stage::UniqueBy(None) => {
                    if unique_dedup { return None; }
                    unique_dedup = true;
                }
                Stage::Reverse => { reverse = true; }
                _ => return None,
            }
        }
        let _ = sort_asc;  // reserved for future negate-detection

        let mut acc = SinkAcc::new(&self.sink, &self.sink_kernels)?;
        if unique_dedup {
            // Currently only Collect supports dedup-mode; other sinks
            // would need their own variant.  Bail if not Collect.
            match &mut acc {
                SinkAcc::Collect { dedup, .. } => {
                    *dedup = Some(std::collections::HashSet::new());
                }
                _ => return None,
            }
        }
        let outer_iter = crate::strref::tape_array_iter(tape, arr_idx)?;

        // Skip / Take counters threaded through both branches.  Skip
        // burns first N post-filter rows; Take ends iteration after
        // taking N; both are streaming (no buffer).  When sort_kernel
        // is set, we buffer post-filter entries first, sort, then
        // apply Skip/Take/Map/Sink — barrier semantics.
        let mut skipped: usize = 0;
        let mut taken:   usize = 0;
        let take_limit = take_n;

        // Sort barrier: buffer entry idxs, sort by kernel-extracted key,
        // then drain through Skip→Take→Map→Sink.  Only entry-rows
        // (post-Filter, pre-Map) are buffered; Map applies after sort.
        if let Some(sk) = sort_kernel {
            let mut buf: Vec<usize> = Vec::new();
            if let Some(fk_kernel) = flat_kernel {
                'outer_s: for outer_idx in outer_iter {
                    let inner_arr = match fk_kernel {
                        BodyKernel::FieldRead(k) =>
                            crate::strref::tape_object_field(tape, outer_idx, k.as_ref()),
                        BodyKernel::FieldChain(keys) => {
                            let key_strs: Vec<&str> = keys.iter().map(|k| k.as_ref()).collect();
                            crate::strref::tape_walk_field_chain_from(tape, outer_idx, &key_strs)
                        }
                        _ => return None,
                    };
                    let inner_arr = match inner_arr { Some(i) => i, None => continue };
                    let inner_iter = match crate::strref::tape_array_iter(tape, inner_arr) {
                        Some(it) => it, None => continue,
                    };
                    'inner_s: for inner_idx in inner_iter {
                        for fk in &filter_kernels {
                            match eval_kernel_pred(fk, tape, inner_idx) {
                                Some(true) => {}
                                Some(false) => continue 'inner_s,
                                None => return None,
                            }
                        }
                        buf.push(inner_idx);
                    }
                    let _ = (); let _outer_s = ();  // bind label
                    if false { break 'outer_s; }
                }
            } else {
                'rows_s: for entry_idx in outer_iter {
                    for fk in &filter_kernels {
                        match eval_kernel_pred(fk, tape, entry_idx) {
                            Some(true) => {}
                            Some(false) => continue 'rows_s,
                            None => return None,
                        }
                    }
                    buf.push(entry_idx);
                }
            }
            // Sort buffer by key — kernel can be path or Arith
            // (for `sort_by(-score)` shape, classified as Arith).
            let key_for = |idx: usize| -> TopKey {
                let v = eval_kernel_value(sk, tape, idx).unwrap_or(TapeVal::Missing);
                match v {
                    TapeVal::Int(n) => TopKey::Int(n),
                    TapeVal::Float(f) => TopKey::Float(float_to_sortable(f)),
                    TapeVal::StrIdx(i) => match tape.nodes[i] {
                        crate::strref::TapeNode::StringRef { start, end } =>
                            TopKey::Str(tape.bytes_buf[start as usize..end as usize].to_vec()),
                        _ => TopKey::Bottom,
                    },
                    _ => TopKey::Bottom,
                }
            };
            buf.sort_by(|a, b| key_for(*a).cmp(&key_for(*b)));
            // Drain through Skip / Take / Map / Sink.
            for entry_idx in buf {
                if skipped < skip_n { skipped += 1; continue; }
                if let Some(lim) = take_limit { if taken >= lim { break; } }
                let row = match map_kernel {
                    Some(mk) => eval_kernel_to_row(mk, tape, entry_idx)?,
                    None => RowSrc::Entry(entry_idx),
                };
                taken += 1;
                if !acc.feed(tape, row)? { break; }
            }
            let result_pre_reverse = acc.finalise(tape);
            let result = if reverse {
                reverse_collect_result(result_pre_reverse)
            } else {
                result_pre_reverse
            };
            cache.note_tape_run();
            return Some(Ok(result));
        }

        if let Some(fk_kernel) = flat_kernel {
            'outer: for outer_idx in outer_iter {
                // Outer filters apply per outer entry pre-FlatMap.
                let mut outer_pass = true;
                for fk in &outer_filter_kernels {
                    match eval_kernel_pred(fk, tape, outer_idx) {
                        Some(true) => {}
                        Some(false) => { outer_pass = false; break; }
                        None => return None,
                    }
                }
                if !outer_pass { continue; }
                let inner_arr = match fk_kernel {
                    BodyKernel::FieldRead(k) =>
                        crate::strref::tape_object_field(tape, outer_idx, k.as_ref()),
                    BodyKernel::FieldChain(keys) => {
                        let key_strs: Vec<&str> = keys.iter().map(|k| k.as_ref()).collect();
                        crate::strref::tape_walk_field_chain_from(tape, outer_idx, &key_strs)
                    }
                    _ => return None,
                };
                let inner_arr = match inner_arr { Some(i) => i, None => continue };
                let inner_iter = match crate::strref::tape_array_iter(tape, inner_arr) {
                    Some(it) => it,
                    None => continue,
                };
                'inner: for inner_idx in inner_iter {
                    for fk in &filter_kernels {
                        match eval_kernel_pred(fk, tape, inner_idx) {
                            Some(true) => {}
                            Some(false) => continue 'inner,
                            None => return None,
                        }
                    }
                    if skipped < skip_n { skipped += 1; continue; }
                    if let Some(lim) = take_limit { if taken >= lim { break 'outer; } }
                    let row = match map_kernel {
                        Some(mk) => eval_kernel_to_row(mk, tape, inner_idx)?,
                        None => RowSrc::Entry(inner_idx),
                    };
                    taken += 1;
                    if !acc.feed(tape, row)? { break 'outer; }
                }
            }
        } else {
            'rows: for entry_idx in outer_iter {
                for fk in &filter_kernels {
                    match eval_kernel_pred(fk, tape, entry_idx) {
                        Some(true) => {}
                        Some(false) => continue 'rows,
                        None => return None,
                    }
                }
                if skipped < skip_n { skipped += 1; continue; }
                if let Some(lim) = take_limit { if taken >= lim { break; } }
                let row = match map_kernel {
                    Some(mk) => eval_kernel_to_row(mk, tape, entry_idx)?,
                    None => RowSrc::Entry(entry_idx),
                };
                taken += 1;
                if !acc.feed(tape, row)? { break; }
            }
        }

        let result_pre_reverse = acc.finalise(tape);
        // Reverse barrier — only meaningful when sink is Collect; for
        // other sinks reverse on the result is undefined (most are
        // scalar). Reverse a Val::Arr / typed-vec by clone-reversing.
        let result = if reverse {
            reverse_collect_result(result_pre_reverse)
        } else {
            result_pre_reverse
        };
        cache.note_tape_run();
        Some(Ok(result))
    }

    /// Cache-aware variant: when cache promotes the source array,
    /// recv is replaced with `Val::ObjVec` and the slot kernels fire.
    fn try_columnar_with(
        &self,
        root: &Val,
        cache: Option<&dyn ObjVecPromoter>,
    ) -> Option<Result<Val, EvalError>> {
        // Tape-only path — bypasses Val entirely for tape-friendly
        // shapes.  Closes the cold-start gap: simd-json parses to
        // tape; aggregate queries fold straight from the tape with
        // no Arc<Vec>/Arc<Obj> ever materialised.
        #[cfg(feature = "simd-json")]
        if let Some(out) = self.try_tape_aggregate(cache) {
            return Some(out);
        }
        // Typed ObjVec stage-chain path next (uses cache).
        if let Some(out) = self.try_columnar_stage_chain_with(root, cache) {
            return Some(out);
        }
        if let Some(out) = self.try_columnar_stage_chain(root) { return Some(out); }
        if !self.stages.is_empty() { return None; }

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

        // Reuse the body of try_columnar by running its tail logic
        // against the (possibly promoted) recv.  Inline the typed-lane
        // and ObjVec branches:
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
        if let Val::ObjVec(d) = &recv {
            if let Sink::FlatMapCount(prog) = &self.sink {
                if let Some(field) = single_field_prog(prog) {
                    if let Some(slot) = d.slot_of(field) {
                        return Some(Ok(objvec_flatmap_count_slot(d, slot)));
                    }
                }
            }
            if let Sink::NumMap(op, prog) = &self.sink {
                let field = single_field_prog(prog)?;
                let slot = d.slot_of(field)?;
                return Some(Ok(objvec_num_slot(d, slot, *op)));
            }
            if let Sink::NumFilterMap(op, pred, map) = &self.sink {
                let (pf, cop, lit) = single_cmp_prog(pred)?;
                let mf = single_field_prog(map)?;
                let sp = d.slot_of(pf)?;
                let sm = d.slot_of(mf)?;
                return Some(Ok(objvec_filter_num_slots(d, sp, cop, &lit, sm, *op)));
            }
            if let Sink::CountIf(pred) = &self.sink {
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

        // ObjVec source — slot-indexed direct reads, no IndexMap probe.
        if let Val::ObjVec(d) = &recv {
            if let Sink::FlatMapCount(prog) = &self.sink {
                if let Some(field) = single_field_prog(prog) {
                    if let Some(slot) = d.slot_of(field) {
                        return Some(Ok(objvec_flatmap_count_slot(d, slot)));
                    }
                }
            }
            if let Sink::NumMap(op, prog) = &self.sink {
                let field = single_field_prog(prog)?;
                let slot = d.slot_of(field)?;
                return Some(Ok(objvec_num_slot(d, slot, *op)));
            }
            if let Sink::NumFilterMap(op, pred, map) = &self.sink {
                let (pf, cop, lit) = single_cmp_prog(pred)?;
                let mf = single_field_prog(map)?;
                let sp = d.slot_of(pf)?;
                let sm = d.slot_of(mf)?;
                return Some(Ok(objvec_filter_num_slots(d, sp, cop, &lit, sm, *op)));
            }
            if let Sink::CountIf(pred) = &self.sink {
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

        let arr = match &recv {
            Val::Arr(a) => Arc::clone(a),
            _ => return None,
        };

        // NumMap with `@.field` shape — extract field column, fold.
        if let Sink::NumMap(op, prog) = &self.sink {
            let field = single_field_prog(prog)?;
            return Some(Ok(columnar_num_field(&arr, field, *op)));
        }

        // NumFilterMap — extract two columns, mask + fold.
        if let Sink::NumFilterMap(op, pred, map) = &self.sink {
            let (pf, cop, lit) = single_cmp_prog(pred)?;
            let mf = single_field_prog(map)?;
            return Some(Ok(columnar_filter_num(&arr, pf, cop, &lit, mf, *op)));
        }

        // CountIf with single-cmp predicate.
        if let Sink::CountIf(pred) = &self.sink {
            if let Some((pf, op, lit)) = single_cmp_prog(pred) {
                return Some(Ok(columnar_filter_count(&arr, pf, op, &lit)));
            }
            // Compound AND: all leaves must be single-cmp comparisons.
            if let Some(leaves) = and_chain_prog(pred) {
                return Some(Ok(columnar_filter_count_and(&arr, &leaves)));
            }
        }

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
        self.try_tape_aggregate(Some(cache))
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
    fn try_run_composed(&self, root: &Val) -> Option<Result<Val, EvalError>> {
        use crate::composed as cmp;
        use crate::composed::Stage as ComposedStage;
        use std::cell::{Cell, RefCell};
        use std::rc::Rc;

        // Decompose fused Sinks into base Stage chain + base Sink so
        // the composed substrate runs them uniformly. One match arm
        // per fused variant; result is a (stages, sink) pair that
        // mirrors the pre-fusion form. No per-shape exec code — same
        // generic pipeline handles every result. Tier 3 deletes the
        // fused Sink variants + this decomposition together.
        let (eff_stages, eff_kernels, eff_sink): (Vec<Stage>, Vec<BodyKernel>, Sink) = {
            let mut stages = self.stages.clone();
            let mut kernels = self.stage_kernels.clone();
            let sink = match &self.sink {
                Sink::NumMap(op, prog) => {
                    stages.push(Stage::Map(Arc::clone(prog)));
                    kernels.push(BodyKernel::classify(prog));
                    Sink::Numeric(*op)
                }
                Sink::NumFilterMap(op, pred, map) => {
                    stages.push(Stage::Filter(Arc::clone(pred)));
                    kernels.push(BodyKernel::classify(pred));
                    stages.push(Stage::Map(Arc::clone(map)));
                    kernels.push(BodyKernel::classify(map));
                    Sink::Numeric(*op)
                }
                Sink::CountIf(prog) => {
                    stages.push(Stage::Filter(Arc::clone(prog)));
                    kernels.push(BodyKernel::classify(prog));
                    Sink::Count
                }
                Sink::FilterFirst(prog) => {
                    stages.push(Stage::Filter(Arc::clone(prog)));
                    kernels.push(BodyKernel::classify(prog));
                    Sink::First
                }
                Sink::FilterLast(prog) => {
                    stages.push(Stage::Filter(Arc::clone(prog)));
                    kernels.push(BodyKernel::classify(prog));
                    Sink::Last
                }
                Sink::FirstMap(prog) => {
                    stages.push(Stage::Map(Arc::clone(prog)));
                    kernels.push(BodyKernel::classify(prog));
                    Sink::First
                }
                Sink::LastMap(prog) => {
                    stages.push(Stage::Map(Arc::clone(prog)));
                    kernels.push(BodyKernel::classify(prog));
                    Sink::Last
                }
                Sink::FlatMapCount(prog) => {
                    stages.push(Stage::FlatMap(Arc::clone(prog)));
                    kernels.push(BodyKernel::classify(prog));
                    Sink::Count
                }
                Sink::UniqueCount => {
                    stages.push(Stage::UniqueBy(None));
                    kernels.push(BodyKernel::Generic);
                    Sink::Count
                }
                Sink::MinBy(key) => {
                    stages.push(Stage::Sort(Some(Arc::clone(key))));
                    kernels.push(BodyKernel::classify(key));
                    Sink::First
                }
                Sink::MaxBy(key) => {
                    stages.push(Stage::Sort(Some(Arc::clone(key))));
                    kernels.push(BodyKernel::classify(key));
                    Sink::Last
                }
                Sink::TopN { n, asc, key } => {
                    if let Some(k) = key {
                        stages.push(Stage::Sort(Some(Arc::clone(k))));
                        kernels.push(BodyKernel::classify(k));
                    } else {
                        stages.push(Stage::Sort(None));
                        kernels.push(BodyKernel::Generic);
                    }
                    if !*asc {
                        stages.push(Stage::Reverse);
                        kernels.push(BodyKernel::Generic);
                    }
                    stages.push(Stage::Take(*n));
                    kernels.push(BodyKernel::Generic);
                    Sink::Collect
                }
                base => base.clone(),
            };
            (stages, kernels, sink)
        };

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
            _ => return None,
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

        // Find barrier positions. Each [last_split..barrier_idx] is a
        // streaming segment; [barrier_idx] is the barrier op.
        let mut last_split = 0usize;
        let mut group_by_seen: Option<usize> = None;
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
            buf = match s {
                Stage::Reverse => cmp::barrier_reverse(buf),
                Stage::Sort(None) => cmp::barrier_sort(buf, &cmp::KeySource::None),
                Stage::Sort(Some(_)) => {
                    let key = key_from_kernel(kernel)?;
                    cmp::barrier_sort(buf, &key)
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
                    group_by_seen = Some(i);
                    return Some(Ok(val));
                }
                _ => unreachable!(),
            };

            last_split = i + 1;
        }
        let _ = group_by_seen;

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
            Stage::Reverse | Stage::Sort(_) | Stage::UniqueBy(_) | Stage::FlatMap(_) | Stage::GroupBy(_)));
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
                Sink::NumMap(op, prog) => {
                    let kernel = self.sink_kernels.first().unwrap_or(&BodyKernel::Generic);
                    let v = eval_kernel(kernel, &item, &mut vm, &mut loop_env, prog)?;
                    num_fold(&mut acc_sum_i, &mut acc_sum_f, &mut sum_floated,
                             &mut acc_min_f, &mut acc_max_f, &mut acc_n_obs,
                             *op, &v);
                }
                Sink::CountIf(prog) => {
                    let kernel = self.sink_kernels.first().unwrap_or(&BodyKernel::Generic);
                    if is_truthy(&eval_kernel(kernel, &item, &mut vm, &mut loop_env, prog)?) {
                        acc_count += 1;
                    }
                }
                Sink::NumFilterMap(op, pred, map) => {
                    let pred_k = self.sink_kernels.first().unwrap_or(&BodyKernel::Generic);
                    let map_k  = self.sink_kernels.get(1).unwrap_or(&BodyKernel::Generic);
                    if is_truthy(&eval_kernel(pred_k, &item, &mut vm, &mut loop_env, pred)?) {
                        let v = eval_kernel(map_k, &item, &mut vm, &mut loop_env, map)?;
                        num_fold(&mut acc_sum_i, &mut acc_sum_f, &mut sum_floated,
                                 &mut acc_min_f, &mut acc_max_f, &mut acc_n_obs,
                                 *op, &v);
                    }
                }
                Sink::FilterFirst(prog) => {
                    let kernel = self.sink_kernels.first().unwrap_or(&BodyKernel::Generic);
                    if acc_first.is_none()
                        && is_truthy(&eval_kernel(kernel, &item, &mut vm, &mut loop_env, prog)?) {
                        acc_first = Some(item.clone());
                    }
                }
                Sink::FilterLast(prog) => {
                    let kernel = self.sink_kernels.first().unwrap_or(&BodyKernel::Generic);
                    if is_truthy(&eval_kernel(kernel, &item, &mut vm, &mut loop_env, prog)?) {
                        acc_last = Some(item.clone());
                    }
                }
                Sink::FirstMap(prog) => {
                    let kernel = self.sink_kernels.first().unwrap_or(&BodyKernel::Generic);
                    if acc_first.is_none() {
                        acc_first = Some(eval_kernel(kernel, &item, &mut vm, &mut loop_env, prog)?);
                    }
                }
                Sink::LastMap(prog) => {
                    let kernel = self.sink_kernels.first().unwrap_or(&BodyKernel::Generic);
                    acc_last = Some(eval_kernel(kernel, &item, &mut vm, &mut loop_env, prog)?);
                }
                Sink::FlatMapCount(prog) => {
                    let kernel = self.sink_kernels.first().unwrap_or(&BodyKernel::Generic);
                    let inner = eval_kernel(kernel, &item, &mut vm, &mut loop_env, prog)?;
                    if let Some(arr) = inner.as_vals() {
                        acc_count += arr.len() as i64;
                    } else {
                        acc_count += 1;
                    }
                }
                // Sort-fused sinks: collect items into a Vec to apply
                // the heap-based TopN / single-pass MinBy / MaxBy at
                // pipeline finalise.  Per-row cost is the prog eval
                // for keyed variants, plus a heap push for TopN.
                Sink::TopN { .. } | Sink::MinBy(_) | Sink::MaxBy(_) => {
                    acc_collect.push(item);
                }
                Sink::UniqueCount => {
                    acc_collect.push(item);
                }
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
            Sink::CountIf(_)        => Val::Int(acc_count),
            Sink::FlatMapCount(_)   => Val::Int(acc_count),
            Sink::Numeric(op)
            | Sink::NumMap(op, _)
            | Sink::NumFilterMap(op, _, _) =>
                num_finalise(*op, acc_sum_i, acc_sum_f, sum_floated,
                             acc_min_f, acc_max_f, acc_n_obs),
            Sink::First | Sink::FilterFirst(_) | Sink::FirstMap(_) =>
                acc_first.unwrap_or(Val::Null),
            Sink::Last  | Sink::FilterLast(_)  | Sink::LastMap(_) =>
                acc_last.unwrap_or(Val::Null),
            Sink::TopN { n, asc, key } => {
                let key_kernel = self.sink_kernels.first().unwrap_or(&BodyKernel::Generic);
                topn_finalise(&mut vm, acc_collect, *n, *asc, key.as_ref(), key_kernel)?
            }
            Sink::MinBy(key) => {
                let key_kernel = self.sink_kernels.first().unwrap_or(&BodyKernel::Generic);
                keyed_extreme(&mut vm, acc_collect, key, false, key_kernel)?
            }
            Sink::MaxBy(key) => {
                let key_kernel = self.sink_kernels.first().unwrap_or(&BodyKernel::Generic);
                keyed_extreme(&mut vm, acc_collect, key, true, key_kernel)?
            }
            Sink::UniqueCount       => unique_count_finalise(acc_collect),
        })
    }
}

fn unique_count_finalise(items: Vec<Val>) -> Val {
    use crate::eval::util::val_to_key;
    let mut seen: std::collections::HashSet<String> =
        std::collections::HashSet::with_capacity(items.len());
    let mut n: i64 = 0;
    for it in &items {
        if seen.insert(val_to_key(it)) { n += 1; }
    }
    Val::Int(n)
}

/// Heap-based top-N: keep the n smallest (or largest) by natural cmp
/// or a key program.  O(N log n) vs O(N log N) for full sort.  Returns
/// a Val::Arr with the result in ascending sort order.
fn topn_finalise(
    vm: &mut crate::vm::VM,
    items: Vec<Val>,
    n: usize,
    asc: bool,
    key_prog: Option<&Arc<crate::vm::Program>>,
    key_kernel: &BodyKernel,
) -> Result<Val, EvalError> {
    if n == 0 || items.is_empty() { return Ok(Val::arr(Vec::new())); }
    use std::collections::BinaryHeap;

    // Compute keys once.  When key_prog is None we use the item itself.
    let mut env = vm.make_loop_env(Val::Null);
    let mut keyed: Vec<(Val, Val)> = Vec::with_capacity(items.len());
    for it in items {
        let k = match key_prog {
            Some(p) => eval_kernel(key_kernel, &it, vm, &mut env, p)?,
            None => it.clone(),
        };
        keyed.push((k, it));
    }

    // Heap of (cmp_key, value).  For asc=true we want n smallest →
    // max-heap of size n keyed on the natural ordering.  For asc=false
    // (largest) we want a min-heap of size n.
    // Single Ord wrapper carries the key + value so BinaryHeap's
    // requirements are satisfied without leaning on Val: Ord.
    struct Entry { key: Val, val: Val, asc: bool }
    impl PartialEq for Entry { fn eq(&self, o: &Self) -> bool { cmp_val_total(&self.key, &o.key).is_eq() } }
    impl Eq for Entry {}
    impl PartialOrd for Entry { fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(o)) } }
    impl Ord for Entry {
        fn cmp(&self, o: &Self) -> std::cmp::Ordering {
            let order = cmp_val_total(&self.key, &o.key);
            // For asc=true we want a max-heap of the n smallest, so
            // the natural order on the key is also the heap order.
            // For asc=false we invert so the heap top is the smallest
            // among the n largest seen so far.
            if self.asc { order } else { order.reverse() }
        }
    }

    let mut heap: BinaryHeap<Entry> = BinaryHeap::with_capacity(n);
    for (k, v) in keyed {
        if heap.len() < n {
            heap.push(Entry { key: k, val: v, asc });
        } else if let Some(top) = heap.peek() {
            let order = cmp_val_total(&k, &top.key);
            let replace = if asc { order.is_lt() } else { order.is_gt() };
            if replace {
                heap.pop();
                heap.push(Entry { key: k, val: v, asc });
            }
        }
    }
    let sorted: Vec<Entry> = heap.into_sorted_vec();
    // into_sorted_vec respects Ord; for asc=true that means ascending
    // (smallest first), for asc=false that means largest-first.
    Ok(Val::arr(sorted.into_iter().map(|e| e.val).collect()))
}

/// One-pass argmin/argmax by key program.
fn keyed_extreme(
    vm: &mut crate::vm::VM,
    items: Vec<Val>,
    key_prog: &Arc<crate::vm::Program>,
    want_max: bool,
    key_kernel: &BodyKernel,
) -> Result<Val, EvalError> {
    let mut best: Option<(Val, Val)> = None;
    let mut env = vm.make_loop_env(Val::Null);
    for it in items {
        let k = eval_kernel(key_kernel, &it, vm, &mut env, key_prog)?;
        match &best {
            None => best = Some((k, it)),
            Some((bk, _)) => {
                let order = cmp_val_total(&k, bk);
                let replace = if want_max { order.is_gt() } else { order.is_lt() };
                if replace { best = Some((k, it)); }
            }
        }
    }
    Ok(best.map(|(_, v)| v).unwrap_or(Val::Null))
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
// entry points:
//   - eval_kernel_value(k, tape, entry) -> Option<TapeVal>
//   - eval_kernel_pred (k, tape, entry) -> Option<bool>
//
// Returning `None` from either signals "kernel shape not supported on
// tape" (e.g. Generic body) — caller falls back to the Val path.
// Adding a new BodyKernel variant requires updating one match arm in
// each — no per-Sink editing.

/// Tape-side scalar value abstraction.  Carries enough info for sink
/// accumulators to fold without re-indexing the tape per access.
#[cfg(feature = "simd-json")]
#[derive(Debug, Clone, Copy)]
enum TapeVal {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    /// String stored as a tape node index (lazy `as_str()` via tape).
    StrIdx(usize),
    /// Field/value missing — counts as "no value" for fold accumulators.
    Missing,
}

/// Per-row state threaded through pipeline stages on tape.
/// - `Entry(idx)` — current is a tape Object/Array entry; field-read
///   kernels apply.  Initial state for each row pulled from a tape Array.
/// - `Scalar(v)` — current is a projected scalar (post `Map(FieldRead)`).
/// - `Mat(v)` — current is a materialised Val (post `Map(ObjProject)`
///   or other kernels that produce containers).  Owns an Arc-wrapped
///   Val so clone is cheap (Arc bump).
#[cfg(feature = "simd-json")]
#[derive(Debug, Clone)]
enum RowSrc {
    Entry(usize),
    Scalar(TapeVal),
    Mat(Val),
}

#[cfg(feature = "simd-json")]
fn node_to_tape_val(tape: &crate::strref::TapeData, idx: usize) -> TapeVal {
    use simd_json::StaticNode as SN;
    use crate::strref::TapeNode;
    match tape.nodes[idx] {
        TapeNode::Static(SN::Null) => TapeVal::Null,
        TapeNode::Static(SN::Bool(b)) => TapeVal::Bool(b),
        TapeNode::Static(SN::I64(n)) => TapeVal::Int(n),
        TapeNode::Static(SN::U64(n)) => {
            if n <= i64::MAX as u64 { TapeVal::Int(n as i64) }
            else { TapeVal::Float(n as f64) }
        }
        TapeNode::Static(SN::F64(f)) => TapeVal::Float(f),
        TapeNode::StringRef { .. } => TapeVal::StrIdx(idx),
        // Containers materialise via owning Val on demand; for now treat
        // as Missing so sinks that need them bail to the Val path.
        TapeNode::Array { .. } | TapeNode::Object { .. } => TapeVal::Missing,
    }
}

/// Evaluate a BodyKernel against a tape entry → `RowSrc`.  Covers
/// scalar-producing kernels (FieldRead/FieldChain/Const/ConstBool)
/// AND materialised kernels (ObjProject).  Returns `None` for kernel
/// shapes the tape executor can't represent.
#[cfg(feature = "simd-json")]
fn eval_kernel_to_row(
    kernel: &BodyKernel,
    tape: &crate::strref::TapeData,
    entry_idx: usize,
) -> Option<RowSrc> {
    match kernel {
        BodyKernel::ObjProject(entries) => {
            let mut pairs: Vec<(Arc<str>, Val)> = Vec::with_capacity(entries.len());
            for e in entries.iter() {
                let v = match e {
                    ObjProjEntry::Path { path, .. } => {
                        let key_strs: Vec<&str> = path.iter().map(|k| k.as_ref()).collect();
                        let v_idx = crate::strref::tape_walk_field_chain_from(
                            tape, entry_idx, &key_strs);
                        match v_idx {
                            Some(idx) => tape_node_to_val_primitive(tape, idx)?,
                            None => Val::Null,
                        }
                    }
                    ObjProjEntry::InnerLen { path, .. } => {
                        let key_strs: Vec<&str> = path.iter().map(|k| k.as_ref()).collect();
                        let v_idx = crate::strref::tape_walk_field_chain_from(
                            tape, entry_idx, &key_strs);
                        match v_idx {
                            Some(idx) => match tape.nodes[idx] {
                                crate::strref::TapeNode::Array { len, .. } => Val::Int(len as i64),
                                _ => Val::Null,
                            },
                            None => Val::Null,
                        }
                    }
                    ObjProjEntry::InnerMapSum { path, inner, .. } => {
                        let key_strs: Vec<&str> = path.iter().map(|k| k.as_ref()).collect();
                        let arr_idx = crate::strref::tape_walk_field_chain_from(
                            tape, entry_idx, &key_strs);
                        match arr_idx {
                            Some(idx) => {
                                let iter = crate::strref::tape_array_iter(tape, idx);
                                if let Some(iter) = iter {
                                    let mut st = NumAccState::new();
                                    for inner_entry in iter {
                                        let v = eval_kernel_value(inner.as_ref(), tape, inner_entry)?;
                                        if !st.push(v) { return None; }
                                    }
                                    if st.count == 0 { Val::Int(0) }
                                    else if st.is_float { Val::Float(st.sum_f) }
                                    else { Val::Int(st.sum_i) }
                                } else { Val::Null }
                            }
                            None => Val::Null,
                        }
                    }
                };
                pairs.push((e.key().clone(), v));
            }
            Some(RowSrc::Mat(Val::ObjSmall(Arc::from(pairs.into_boxed_slice()))))
        }
        BodyKernel::FString(parts) => {
            // Concatenate literal chunks + path-resolved values.
            // Numbers / bools format via Display; strings emit as-is;
            // null/missing emit empty.
            use std::fmt::Write;
            let mut out = String::new();
            for p in parts.iter() {
                match p {
                    FStrPart::Lit(s) => out.push_str(s.as_ref()),
                    FStrPart::Path(path) => {
                        let key_strs: Vec<&str> = path.iter().map(|k| k.as_ref()).collect();
                        let v_idx = if path.is_empty() {
                            Some(entry_idx)
                        } else {
                            crate::strref::tape_walk_field_chain_from(
                                tape, entry_idx, &key_strs)
                        };
                        if let Some(idx) = v_idx {
                            match tape.nodes[idx] {
                                crate::strref::TapeNode::StringRef { start, end } => {
                                    out.push_str(tape.str_at_range(
                                        start as usize, end as usize));
                                }
                                crate::strref::TapeNode::Static(simd_json::StaticNode::I64(n)) => {
                                    let _ = write!(out, "{}", n);
                                }
                                crate::strref::TapeNode::Static(simd_json::StaticNode::U64(n)) => {
                                    let _ = write!(out, "{}", n);
                                }
                                crate::strref::TapeNode::Static(simd_json::StaticNode::F64(f)) => {
                                    let _ = write!(out, "{}", f);
                                }
                                crate::strref::TapeNode::Static(simd_json::StaticNode::Bool(b)) => {
                                    let _ = write!(out, "{}", b);
                                }
                                crate::strref::TapeNode::Static(simd_json::StaticNode::Null) => {}
                                _ => return None,  // container — bail
                            }
                        }
                    }
                }
            }
            Some(RowSrc::Mat(Val::Str(Arc::<str>::from(out))))
        }
        _ => {
            let v = eval_kernel_value(kernel, tape, entry_idx)?;
            Some(RowSrc::Scalar(v))
        }
    }
}

/// Build a primitive Val from a tape node.  Returns None if the node
/// is a container (Object/Array) — caller bails to the Val path since
/// containers need full subtree materialise via Val::from_tape_node
/// (which requires an Arc<TapeData> we don't have at this scope).
/// For ObjProject's path leaves this is sufficient: bench shapes
/// (Q4 `{id, name: user.name, score}`, Q10 `{id}`) project primitive
/// leaves only.
#[cfg(feature = "simd-json")]
fn tape_node_to_val_primitive(tape: &crate::strref::TapeData, idx: usize) -> Option<Val> {
    use crate::strref::TapeNode;
    use simd_json::StaticNode as SN;
    Some(match tape.nodes[idx] {
        TapeNode::Static(SN::Null) => Val::Null,
        TapeNode::Static(SN::Bool(b)) => Val::Bool(b),
        TapeNode::Static(SN::I64(n)) => Val::Int(n),
        TapeNode::Static(SN::U64(n)) => {
            if n <= i64::MAX as u64 { Val::Int(n as i64) } else { Val::Float(n as f64) }
        }
        TapeNode::Static(SN::F64(f)) => Val::Float(f),
        TapeNode::StringRef { start, end } => {
            let s = tape.str_at_range(start as usize, end as usize);
            Val::Str(Arc::<str>::from(s))
        }
        TapeNode::Array { .. } | TapeNode::Object { .. } => return None,
    })
}

/// Evaluate a BodyKernel that produces a *value* (not a predicate)
/// against a tape entry.  Returns `None` if the kernel shape is not
/// representable on tape (Generic / unsupported chain).
#[cfg(feature = "simd-json")]
fn eval_kernel_value(
    kernel: &BodyKernel,
    tape: &crate::strref::TapeData,
    entry_idx: usize,
) -> Option<TapeVal> {
    use crate::strref::tape_object_field;
    match kernel {
        BodyKernel::FieldRead(k) => {
            match tape_object_field(tape, entry_idx, k.as_ref()) {
                Some(v) => Some(node_to_tape_val(tape, v)),
                None => Some(TapeVal::Missing),
            }
        }
        BodyKernel::FieldChain(keys) => {
            let mut cur = entry_idx;
            for k in keys.iter() {
                match tape_object_field(tape, cur, k.as_ref()) {
                    Some(v) => cur = v,
                    None => return Some(TapeVal::Missing),
                }
            }
            Some(node_to_tape_val(tape, cur))
        }
        BodyKernel::Const(v) => match v {
            Val::Int(n)   => Some(TapeVal::Int(*n)),
            Val::Float(f) => Some(TapeVal::Float(*f)),
            Val::Bool(b)  => Some(TapeVal::Bool(*b)),
            Val::Null     => Some(TapeVal::Null),
            _ => None,
        },
        BodyKernel::ConstBool(b) => Some(TapeVal::Bool(*b)),
        BodyKernel::Arith(lhs, op, rhs) => {
            let l = arith_operand_value(lhs, tape, entry_idx)?;
            let r = arith_operand_value(rhs, tape, entry_idx)?;
            arith_apply(l, *op, r)
        }
        // Pred-shaped kernels don't produce values directly; bail.
        // ObjProject doesn't fit TapeVal — caller should route through
        // eval_kernel_to_row instead.
        BodyKernel::FieldCmpLit(_, _, _)
        | BodyKernel::FieldChainCmpLit(_, _, _)
        | BodyKernel::CurrentCmpLit(_, _)
        | BodyKernel::ObjProject(_)
        | BodyKernel::FString(_)
        | BodyKernel::Generic => None,
    }
}

/// Resolve an ArithOperand to a TapeVal (Int/Float; Missing if path
/// missing).
#[cfg(feature = "simd-json")]
fn arith_operand_value(
    op: &ArithOperand,
    tape: &crate::strref::TapeData,
    entry_idx: usize,
) -> Option<TapeVal> {
    match op {
        ArithOperand::LitInt(n)   => Some(TapeVal::Int(*n)),
        ArithOperand::LitFloat(f) => Some(TapeVal::Float(*f)),
        ArithOperand::Path(p) => {
            if p.is_empty() {
                return Some(node_to_tape_val(tape, entry_idx));
            }
            let key_strs: Vec<&str> = p.iter().map(|k| k.as_ref()).collect();
            let v = crate::strref::tape_walk_field_chain_from(tape, entry_idx, &key_strs);
            Some(match v {
                Some(idx) => node_to_tape_val(tape, idx),
                None => TapeVal::Missing,
            })
        }
    }
}

/// Apply ArithOp to two TapeVals.  Numeric coercion: any Float → Float
/// result; otherwise Int.  Div / Mod by zero returns None (bail).
#[cfg(feature = "simd-json")]
fn arith_apply(l: TapeVal, op: ArithOp, r: TapeVal) -> Option<TapeVal> {
    let (li, lf, lk) = num_parts(l)?;
    let (ri, rf, rk) = num_parts(r)?;
    let is_float = lk || rk;
    if is_float {
        let lf = if lk { lf } else { li as f64 };
        let rf = if rk { rf } else { ri as f64 };
        let out = match op {
            ArithOp::Add => lf + rf,
            ArithOp::Sub => lf - rf,
            ArithOp::Mul => lf * rf,
            ArithOp::Div => if rf == 0.0 { return None } else { lf / rf },
            ArithOp::Mod => if rf == 0.0 { return None } else { lf % rf },
        };
        Some(TapeVal::Float(out))
    } else {
        let out = match op {
            ArithOp::Add => li.wrapping_add(ri),
            ArithOp::Sub => li.wrapping_sub(ri),
            ArithOp::Mul => li.wrapping_mul(ri),
            ArithOp::Div => if ri == 0 { return None } else { li / ri },
            ArithOp::Mod => if ri == 0 { return None } else { li % ri },
        };
        Some(TapeVal::Int(out))
    }
}

/// Decompose a TapeVal into (i64, f64, is_float).  Bail for non-numeric.
#[cfg(feature = "simd-json")]
fn num_parts(v: TapeVal) -> Option<(i64, f64, bool)> {
    match v {
        TapeVal::Int(n) => Some((n, 0.0, false)),
        TapeVal::Float(f) => Some((0, f, true)),
        _ => None,
    }
}

/// Evaluate a BodyKernel that produces a *boolean* predicate against
/// a tape entry.  Returns `None` for unsupported shapes (Generic,
/// CurrentCmpLit on objects).
#[cfg(feature = "simd-json")]
fn eval_kernel_pred(
    kernel: &BodyKernel,
    tape: &crate::strref::TapeData,
    entry_idx: usize,
) -> Option<bool> {
    use crate::ast::BinOp;
    use crate::strref::{tape_object_field, tape_value_cmp, tape_value_truthy, TapeCmp};
    let binop = |op: BinOp| -> Option<TapeCmp> {
        Some(match op {
            BinOp::Eq => TapeCmp::Eq, BinOp::Neq => TapeCmp::Neq,
            BinOp::Lt => TapeCmp::Lt, BinOp::Lte => TapeCmp::Lte,
            BinOp::Gt => TapeCmp::Gt, BinOp::Gte => TapeCmp::Gte,
            _ => return None,
        })
    };
    match kernel {
        BodyKernel::FieldRead(k) => {
            // Truthy field read.
            match tape_object_field(tape, entry_idx, k.as_ref()) {
                Some(v) => Some(tape_value_truthy(tape, v)),
                None => Some(false),
            }
        }
        BodyKernel::FieldCmpLit(k, op, lit) => {
            let cmp = binop(*op)?;
            let tlit = val_to_tape_lit_owned(lit)?;
            match tape_object_field(tape, entry_idx, k.as_ref()) {
                Some(v) => Some(tape_value_cmp(tape, v, cmp, &tlit)),
                None => Some(false),
            }
        }
        BodyKernel::FieldChainCmpLit(keys, op, lit) => {
            let cmp = binop(*op)?;
            let tlit = val_to_tape_lit_owned(lit)?;
            let mut cur = entry_idx;
            for k in keys.iter() {
                match tape_object_field(tape, cur, k.as_ref()) {
                    Some(v) => cur = v,
                    None => return Some(false),
                }
            }
            Some(tape_value_cmp(tape, cur, cmp, &tlit))
        }
        BodyKernel::ConstBool(b) => Some(*b),
        BodyKernel::CurrentCmpLit(_, _) | BodyKernel::FieldChain(_)
        | BodyKernel::Const(_) | BodyKernel::ObjProject(_)
        | BodyKernel::Arith(_, _, _) | BodyKernel::FString(_)
        | BodyKernel::Generic => None,
    }
}

#[cfg(feature = "simd-json")]
fn val_to_tape_lit_owned(v: &Val) -> Option<crate::strref::TapeLit<'_>> {
    use crate::strref::TapeLit;
    match v {
        Val::Int(n)   => Some(TapeLit::Int(*n)),
        Val::Float(f) => Some(TapeLit::Float(*f)),
        Val::Str(s)   => Some(TapeLit::Str(s.as_ref())),
        Val::Bool(b)  => Some(TapeLit::Bool(*b)),
        Val::Null     => Some(TapeLit::Null),
        _ => None,
    }
}

// ── Generic Sink accumulator over tape ──────────────────────────────
//
// One state machine per Sink variant; each consumes per-entry tape
// values via the kernel evaluators above.  Adding a new Sink ⇒ one
// new variant + match arms in `new` / `feed` / `finalise`.  No per-
// (Stage_chain, Sink) helpers.

#[cfg(feature = "simd-json")]
enum SinkAcc<'p> {
    /// `.count()` / `.len()` — sink emits Int(n) at finalise.
    Count(usize),
    /// `.sum/.min/.max/.avg` over raw entry values (no projection).
    Numeric { op: NumOp, st: NumAccState },
    /// `Map(kernel) ∘ Numeric(op)` — projects via kernel then accumulates.
    NumMap { op: NumOp, st: NumAccState, kernel: &'p BodyKernel },
    /// `Filter(pred) ∘ Map(map) ∘ Numeric(op)`.
    NumFilterMap {
        op: NumOp, st: NumAccState,
        pred: &'p BodyKernel, map: &'p BodyKernel,
    },
    /// `Filter(pred) ∘ Count`.
    CountIf { count: usize, pred: &'p BodyKernel },
    /// `Filter(pred) ∘ First` — keeps first matching tape entry idx.
    FilterFirst { hit: Option<usize>, pred: &'p BodyKernel },
    /// `Filter(pred) ∘ Last` — keeps last matching tape entry idx.
    FilterLast { hit: Option<usize>, pred: &'p BodyKernel },
    /// `FlatMap(f) ∘ Count` — sums inner array lengths per outer entry.
    /// kernel resolves outer entry → inner Array idx (FieldRead /
    /// FieldChain).
    FlatMapCount { count: usize, kernel: &'p BodyKernel },
    /// `Sort_by(k) ∘ Take(n)` — bounded heap of size N sorted by key
    /// extracted via kernel.  `asc=true` keeps the N smallest;
    /// `asc=false` keeps the N largest.  At finalise, drains heap into
    /// a Val::Arr in sorted order via from_tape_node.
    TopN {
        n: usize,
        asc: bool,
        kernel: Option<&'p BodyKernel>,
        heap: std::collections::BinaryHeap<TopHeapItem>,
    },
    /// Collect rows (post-stage) — gathers RowSrc values.  At finalise:
    /// - all `Entry` rows → typed-lane probe over entry tape nodes
    /// - all `Scalar` rows of uniform type → IntVec / FloatVec /
    ///   StrSliceVec direct from tape values
    /// - heterogeneous → Val::Arr fallback (per-entry materialise +
    ///   tape-val → Val conversion).
    /// `dedup` enables in-line dedup via HashSet keyed on the row's
    /// content — implements `Stage::UniqueBy(None) ∘ Sink::Collect`.
    Collect { rows: Vec<RowSrc>, dedup: Option<std::collections::HashSet<RowKey>> },
}

/// Heap item for TopN — carries a sort key + entry idx.  `Ord` impl
/// is derived from `key`; `entry` only used for materialise at
/// finalise.  For asc=false (largest N), feed inverts the key sign /
/// reverses the comparison externally.
#[cfg(feature = "simd-json")]
#[derive(Debug, Clone)]
pub struct TopHeapItem {
    pub key: TopKey,
    pub entry: usize,
}

/// Sortable key — string by content bytes (Vec<u8>) or numeric (i64 or
/// f64-bits).  Bool / Null / Missing degrade to "smallest".
#[cfg(feature = "simd-json")]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum TopKey {
    /// Always-min sentinel (Null / Bool / Missing).
    Bottom,
    /// f64-as-bits-with-NaN-canonical preserves total order via
    /// custom helper at insert time.
    Int(i64),
    /// f64 lifted to a sortable u64 (TotalOrder bit twiddle below).
    Float(u64),
    Str(Vec<u8>),
}

#[cfg(feature = "simd-json")]
impl Ord for TopHeapItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering { self.key.cmp(&other.key) }
}
#[cfg(feature = "simd-json")]
impl PartialOrd for TopHeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) }
}
#[cfg(feature = "simd-json")]
impl PartialEq for TopHeapItem {
    fn eq(&self, other: &Self) -> bool { self.key == other.key }
}
#[cfg(feature = "simd-json")]
impl Eq for TopHeapItem {}

#[cfg(feature = "simd-json")]
fn float_to_sortable(f: f64) -> u64 {
    let bits = f.to_bits();
    // Twiddle so IEEE float order becomes u64 lexicographic order.
    if bits >> 63 == 0 { bits ^ (1u64 << 63) } else { !bits }
}

/// Reverse a Collect-shaped result.  Operates on the typed lanes
/// (IntVec/FloatVec/StrSliceVec) directly via Vec::reverse + Arc
/// rebuild; on Val::Arr unwraps + reverses + rewraps.  Other Val
/// variants pass through (Reverse on a scalar is undefined).
#[cfg(feature = "simd-json")]
fn reverse_collect_result(v: Val) -> Val {
    use std::sync::Arc;
    match v {
        Val::Arr(a) => {
            let mut inner = match Arc::try_unwrap(a) {
                Ok(v) => v,
                Err(a) => (*a).clone(),
            };
            inner.reverse();
            Val::Arr(Arc::new(inner))
        }
        Val::IntVec(a) => {
            let mut inner = match Arc::try_unwrap(a) {
                Ok(v) => v,
                Err(a) => (*a).clone(),
            };
            inner.reverse();
            Val::IntVec(Arc::new(inner))
        }
        Val::FloatVec(a) => {
            let mut inner = match Arc::try_unwrap(a) {
                Ok(v) => v,
                Err(a) => (*a).clone(),
            };
            inner.reverse();
            Val::FloatVec(Arc::new(inner))
        }
        Val::StrSliceVec(a) => {
            let mut inner = match Arc::try_unwrap(a) {
                Ok(v) => v,
                Err(a) => (*a).clone(),
            };
            inner.reverse();
            Val::StrSliceVec(Arc::new(inner))
        }
        other => other,
    }
}

/// Invert sort order — flip Int sign, complement Float bits, and
/// invert each byte of Str.  Pushes "largest original" to the position
/// of "smallest" so a single max-heap-pop-when-over-N keeps the
/// largest N (asc=false case) using the same algorithm as smallest-N.
#[cfg(feature = "simd-json")]
fn invert_key(k: TopKey) -> TopKey {
    match k {
        TopKey::Int(n) => TopKey::Int(n.wrapping_neg()),
        TopKey::Float(b) => TopKey::Float(!b),
        TopKey::Str(s) => {
            let mut inv = s;
            for byte in inv.iter_mut() { *byte = !*byte; }
            TopKey::Str(inv)
        }
        TopKey::Bottom => TopKey::Bottom,
    }
}

#[cfg(feature = "simd-json")]
fn topkey_from_row(tape: &crate::strref::TapeData, row: RowSrc) -> TopKey {
    use crate::strref::TapeNode;
    let v = match row {
        RowSrc::Scalar(v) => v,
        RowSrc::Entry(idx) => node_to_tape_val(tape, idx),
        // Mat rows can't sort generically; return Bottom so they bunch
        // together — TopN with Mat rows is rare (would imply
        // sort_by(<projected>) over a Map(ObjProject)).
        RowSrc::Mat(_) => return TopKey::Bottom,
    };
    match v {
        TapeVal::Int(n) => TopKey::Int(n),
        TapeVal::Float(f) => TopKey::Float(float_to_sortable(f)),
        TapeVal::StrIdx(i) => match tape.nodes[i] {
            TapeNode::StringRef { start, end } =>
                TopKey::Str(tape.bytes_buf[start as usize..end as usize].to_vec()),
            _ => TopKey::Bottom,
        },
        _ => TopKey::Bottom,
    }
}

/// Hashable key derived from a RowSrc.  Strings keyed by their content
/// bytes (resolved via tape) so two distinct StrIdx with same content
/// dedupe correctly.  Entry rows of container kind don't dedupe (each
/// entry treated unique by tape idx).
#[cfg(feature = "simd-json")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum RowKey {
    Null,
    Bool(bool),
    Int(i64),
    /// f64 keyed by raw bits — NaN behaviour is consistent (NaN bits
    /// hash equal to themselves; Eq follows bit pattern, not IEEE).
    FloatBits(u64),
    Str(Box<[u8]>),
    /// Fallback for entries we can't easily key — tape node idx (each
    /// distinct entry becomes its own bucket).
    EntryIdx(usize),
}

#[cfg(feature = "simd-json")]
fn row_key(tape: &crate::strref::TapeData, row: &RowSrc) -> RowKey {
    use simd_json::StaticNode as SN;
    use crate::strref::TapeNode;
    match row {
        RowSrc::Scalar(TapeVal::Null) => RowKey::Null,
        RowSrc::Scalar(TapeVal::Bool(b)) => RowKey::Bool(*b),
        RowSrc::Scalar(TapeVal::Int(n)) => RowKey::Int(*n),
        RowSrc::Scalar(TapeVal::Float(f)) => RowKey::FloatBits(f.to_bits()),
        RowSrc::Scalar(TapeVal::StrIdx(i)) => match tape.nodes[*i] {
            TapeNode::StringRef { start, end } =>
                RowKey::Str(tape.bytes_buf[start as usize..end as usize].into()),
            _ => RowKey::EntryIdx(*i),
        },
        RowSrc::Scalar(TapeVal::Missing) => RowKey::Null,
        // Mat rows: hash the underlying Val via a stable structural
        // serialisation.  For dedup-of-projections (Map(ObjProject)
        // ∘ UniqueBy(None)) this lets two structurally-equal Vals
        // collapse.  Cost: serialise once per row — acceptable for
        // dedup pass.
        RowSrc::Mat(v) => {
            let bytes = v.to_json_vec();
            RowKey::Str(bytes.into_boxed_slice())
        }
        RowSrc::Entry(idx) => match tape.nodes[*idx] {
            TapeNode::Static(SN::Null) => RowKey::Null,
            TapeNode::Static(SN::Bool(b)) => RowKey::Bool(b),
            TapeNode::Static(SN::I64(n)) => RowKey::Int(n),
            TapeNode::Static(SN::U64(n)) if n <= i64::MAX as u64 => RowKey::Int(n as i64),
            TapeNode::Static(SN::U64(n)) => RowKey::FloatBits((n as f64).to_bits()),
            TapeNode::Static(SN::F64(f)) => RowKey::FloatBits(f.to_bits()),
            TapeNode::StringRef { start, end } =>
                RowKey::Str(tape.bytes_buf[start as usize..end as usize].into()),
            _ => RowKey::EntryIdx(*idx),
        },
    }
}

#[cfg(feature = "simd-json")]
#[derive(Default)]
struct NumAccState {
    sum_i: i64,
    sum_f: f64,
    count: usize,
    min_f: f64,
    max_f: f64,
    is_float: bool,
}

#[cfg(feature = "simd-json")]
impl NumAccState {
    fn new() -> Self {
        Self {
            sum_i: 0, sum_f: 0.0, count: 0,
            min_f: f64::INFINITY, max_f: f64::NEG_INFINITY,
            is_float: false,
        }
    }
    /// Accumulate a tape value if numeric; return false if non-numeric
    /// (signals caller to bail to the Val path).
    fn push(&mut self, v: TapeVal) -> bool {
        match v {
            TapeVal::Missing => true,  // skip
            TapeVal::Int(n) => {
                self.sum_i = self.sum_i.wrapping_add(n);
                self.sum_f += n as f64;
                let nf = n as f64;
                if nf < self.min_f { self.min_f = nf; }
                if nf > self.max_f { self.max_f = nf; }
                self.count += 1;
                true
            }
            TapeVal::Float(f) => {
                self.sum_f += f;
                if f < self.min_f { self.min_f = f; }
                if f > self.max_f { self.max_f = f; }
                self.is_float = true;
                self.count += 1;
                true
            }
            _ => false,
        }
    }
}

#[cfg(feature = "simd-json")]
impl<'p> SinkAcc<'p> {
    fn new(sink: &'p Sink, sink_kernels: &'p [BodyKernel]) -> Option<Self> {
        match sink {
            Sink::Count           => Some(Self::Count(0)),
            Sink::Numeric(op)     => Some(Self::Numeric { op: *op, st: NumAccState::new() }),
            Sink::NumMap(op, _)   => {
                let kernel = sink_kernels.get(0)?;
                Some(Self::NumMap { op: *op, st: NumAccState::new(), kernel })
            }
            Sink::NumFilterMap(op, _, _) => {
                let pred = sink_kernels.get(0)?;
                let map  = sink_kernels.get(1)?;
                Some(Self::NumFilterMap { op: *op, st: NumAccState::new(), pred, map })
            }
            Sink::CountIf(_)      => {
                let pred = sink_kernels.get(0)?;
                Some(Self::CountIf { count: 0, pred })
            }
            Sink::FilterFirst(_)  => {
                let pred = sink_kernels.get(0)?;
                Some(Self::FilterFirst { hit: None, pred })
            }
            Sink::FilterLast(_)   => {
                let pred = sink_kernels.get(0)?;
                Some(Self::FilterLast { hit: None, pred })
            }
            Sink::FlatMapCount(_) => {
                let kernel = sink_kernels.get(0)?;
                if !matches!(kernel,
                    BodyKernel::FieldRead(_) | BodyKernel::FieldChain(_))
                { return None; }
                Some(Self::FlatMapCount { count: 0, kernel })
            }
            Sink::TopN { n, asc, key } => {
                // key=None: sort entries themselves (rare).  key=Some(p):
                // require kernel to be value-producing (FieldRead/FieldChain).
                let kernel = match key {
                    Some(_) => {
                        let k = sink_kernels.get(0)?;
                        if !matches!(k,
                            BodyKernel::FieldRead(_) | BodyKernel::FieldChain(_))
                        { return None; }
                        Some(k)
                    }
                    None => None,
                };
                Some(Self::TopN {
                    n: *n, asc: *asc, kernel,
                    heap: std::collections::BinaryHeap::with_capacity(*n + 1),
                })
            }
            Sink::Collect         => Some(Self::Collect { rows: Vec::new(), dedup: None }),
            // Unsupported sinks bail; caller falls back to Val path.
            _ => None,
        }
    }

    /// Feed one row.  Returns `Some(true)` to continue, `Some(false)`
    /// to early-exit (sink saturated), `None` to abort (kernel/row
    /// shape mismatch — caller falls back to Val).
    ///
    /// Most sinks need an `Entry`-typed row (kernels read fields).
    /// `Collect`, `Numeric`, `NumMap` accept `Scalar` rows post-Map.
    fn feed(&mut self, tape: &crate::strref::TapeData, row: RowSrc) -> Option<bool> {
        match self {
            Self::Count(n) => { *n += 1; Some(true) }
            Self::Numeric { st, .. } => {
                let v = match row {
                    RowSrc::Entry(idx) => node_to_tape_val(tape, idx),
                    RowSrc::Scalar(v) => v,
                    RowSrc::Mat(_) => return None,  // numeric on Val obj — bail
                };
                if !st.push(v) { return None; }
                Some(true)
            }
            Self::NumMap { st, kernel, .. } => {
                // NumMap projects via kernel — only Entry rows make sense
                // (kernel needs fields).  Scalar rows would require a
                // chained Map; current Pipeline IR fuses Map∘Numeric
                // into NumMap before Stage::Map could intervene, so this
                // branch never sees Scalar in the current rule set.
                let entry = match row { RowSrc::Entry(i) => i, _ => return None };
                let v = eval_kernel_value(kernel, tape, entry)?;
                if !st.push(v) { return None; }
                Some(true)
            }
            Self::NumFilterMap { st, pred, map, .. } => {
                let entry = match row { RowSrc::Entry(i) => i, _ => return None };
                if !eval_kernel_pred(pred, tape, entry)? { return Some(true); }
                let v = eval_kernel_value(map, tape, entry)?;
                if !st.push(v) { return None; }
                Some(true)
            }
            Self::CountIf { count, pred } => {
                let entry = match row { RowSrc::Entry(i) => i, _ => return None };
                if eval_kernel_pred(pred, tape, entry)? { *count += 1; }
                Some(true)
            }
            Self::FilterFirst { hit, pred } => {
                let entry = match row { RowSrc::Entry(i) => i, _ => return None };
                if eval_kernel_pred(pred, tape, entry)? {
                    *hit = Some(entry);
                    return Some(false);
                }
                Some(true)
            }
            Self::FilterLast { hit, pred } => {
                let entry = match row { RowSrc::Entry(i) => i, _ => return None };
                if eval_kernel_pred(pred, tape, entry)? { *hit = Some(entry); }
                Some(true)
            }
            Self::TopN { n, asc, kernel, heap } => {
                let entry = match row { RowSrc::Entry(i) => i, _ => return None };
                // Key extraction.  No-kernel: key from entry tape value.
                // With kernel: project via FieldRead/FieldChain.
                let key_row = match kernel {
                    Some(k) => {
                        let v = eval_kernel_value(k, tape, entry)?;
                        RowSrc::Scalar(v)
                    }
                    None => RowSrc::Entry(entry),
                };
                let mut key = topkey_from_row(tape, key_row);
                // For descending (asc=false → keep largest), invert key
                // by reversing comparison through max-heap default.
                // BinaryHeap is a max-heap.  Strategy: keep "smallest n"
                // → push, pop max if heap exceeds n.  For "largest n",
                // negate Int / invert Float bits / reverse Str.
                if !*asc {
                    key = invert_key(key);
                }
                heap.push(TopHeapItem { key, entry });
                if heap.len() > *n { heap.pop(); }
                Some(true)
            }
            Self::FlatMapCount { count, kernel } => {
                let entry = match row { RowSrc::Entry(i) => i, _ => return None };
                let inner = match kernel {
                    BodyKernel::FieldRead(k) =>
                        crate::strref::tape_object_field(tape, entry, k.as_ref()),
                    BodyKernel::FieldChain(keys) => {
                        let key_strs: Vec<&str> = keys.iter().map(|k| k.as_ref()).collect();
                        crate::strref::tape_walk_field_chain_from(tape, entry, &key_strs)
                    }
                    _ => return None,
                };
                if let Some(inner_arr) = inner {
                    if let crate::strref::TapeNode::Array { len, .. } = tape.nodes[inner_arr] {
                        *count += len as usize;
                    }
                }
                Some(true)
            }
            Self::Collect { rows, dedup } => {
                if let Some(seen) = dedup {
                    let key = row_key(tape, &row);
                    if !seen.insert(key) { return Some(true); }
                }
                rows.push(row);
                Some(true)
            }
        }
    }

    fn finalise(self, tape: &std::sync::Arc<crate::strref::TapeData>) -> Val {
        match self {
            Self::Count(n) => Val::Int(n as i64),
            Self::Numeric { op, st } | Self::NumMap { op, st, .. }
            | Self::NumFilterMap { op, st, .. } => {
                tape_finalise(op, st.sum_i, st.sum_f, st.count, st.min_f, st.max_f, st.is_float)
            }
            Self::CountIf { count, .. } => Val::Int(count as i64),
            Self::FilterFirst { hit, .. } | Self::FilterLast { hit, .. } => {
                match hit {
                    Some(idx) => crate::eval::Val::from_tape_node(tape, idx),
                    None => Val::Null,
                }
            }
            Self::FlatMapCount { count, .. } => Val::Int(count as i64),
            Self::TopN { heap, asc, .. } => {
                // Drain heap into Vec sorted ascending by stored key.
                // For asc=true, keys are natural; for asc=false, keys
                // were inverted at insert so natural sort still gives
                // descending of the original.
                let mut items: Vec<TopHeapItem> = heap.into_iter().collect();
                items.sort_by(|a, b| a.key.cmp(&b.key));
                // Build Val::Arr in result order — for asc=false the
                // user expects largest first, which means inverted
                // ascending (smallest inverted = largest original).
                let mut out: Vec<Val> = Vec::with_capacity(items.len());
                if asc {
                    for it in items { out.push(crate::eval::Val::from_tape_node(tape, it.entry)); }
                } else {
                    for it in items { out.push(crate::eval::Val::from_tape_node(tape, it.entry)); }
                }
                Val::Arr(std::sync::Arc::new(out))
            }
            Self::Collect { rows, .. } => collect_rows_to_val(tape, &rows),
        }
    }
}

/// Materialise gathered RowSrc rows into the strongest typed Val lane
/// available.  Two row kinds:
/// - `Entry(idx)` — raw tape entry; probe its tape node for type
/// - `Scalar(v)` — projected TapeVal from a Map kernel
///
/// Uniform numeric / string runs emit IntVec / FloatVec / StrSliceVec
/// (slices into `tape.bytes_buf` — zero per-string heap alloc).
/// Mixed → `Val::Arr` with per-element materialise.
#[cfg(feature = "simd-json")]
fn collect_rows_to_val(
    tape: &std::sync::Arc<crate::strref::TapeData>,
    rows: &[RowSrc],
) -> Val {
    use crate::strref::TapeNode;
    use simd_json::StaticNode as SN;
    if rows.is_empty() { return Val::arr(Vec::new()); }

    // Per-row "type tag" for uniformity probe.
    // 1=int, 2=float (numeric superset), 3=str, 0=other (container/mixed/null/bool/Mat).
    fn row_tag(tape: &crate::strref::TapeData, row: &RowSrc) -> u8 {
        match row {
            RowSrc::Mat(_) => 0,
            RowSrc::Scalar(v) => match v {
                TapeVal::Int(_)   => 1,
                TapeVal::Float(_) => 2,
                TapeVal::StrIdx(_) => 3,
                _ => 0,
            },
            RowSrc::Entry(idx) => match tape.nodes[*idx] {
                TapeNode::Static(SN::I64(_)) | TapeNode::Static(SN::U64(_)) => 1,
                TapeNode::Static(SN::F64(_)) => 2,
                TapeNode::StringRef { .. } => 3,
                _ => 0,
            },
        }
    }

    let first_tag = row_tag(tape, &rows[0]);
    let mut all_int   = first_tag == 1;
    let mut all_float = first_tag == 1 || first_tag == 2;
    let mut all_str   = first_tag == 3;
    for r in rows.iter().skip(1) {
        match row_tag(tape, r) {
            1 => { all_str = false; }
            2 => { all_int = false; all_str = false; }
            3 => { all_int = false; all_float = false; }
            _ => { all_int = false; all_float = false; all_str = false; break; }
        }
        if !all_int && !all_float && !all_str { break; }
    }

    fn row_to_int(tape: &crate::strref::TapeData, row: &RowSrc) -> i64 {
        match row {
            RowSrc::Scalar(TapeVal::Int(n)) => *n,
            RowSrc::Entry(idx) => match tape.nodes[*idx] {
                TapeNode::Static(SN::I64(n)) => n,
                TapeNode::Static(SN::U64(n)) => n as i64,
                _ => unreachable!("uniformity"),
            },
            _ => unreachable!("uniformity"),
        }
    }
    fn row_to_float(tape: &crate::strref::TapeData, row: &RowSrc) -> f64 {
        match row {
            RowSrc::Scalar(TapeVal::Int(n))   => *n as f64,
            RowSrc::Scalar(TapeVal::Float(f)) => *f,
            RowSrc::Entry(idx) => match tape.nodes[*idx] {
                TapeNode::Static(SN::I64(n)) => n as f64,
                TapeNode::Static(SN::U64(n)) => n as f64,
                TapeNode::Static(SN::F64(f)) => f,
                _ => unreachable!("uniformity"),
            },
            _ => unreachable!("uniformity"),
        }
    }
    fn row_to_str_range(tape: &crate::strref::TapeData, row: &RowSrc) -> (u32, u32) {
        let idx = match row {
            RowSrc::Scalar(TapeVal::StrIdx(i)) => *i,
            RowSrc::Entry(i) => *i,
            _ => unreachable!("uniformity"),
        };
        match tape.nodes[idx] {
            TapeNode::StringRef { start, end } => (start, end),
            _ => unreachable!("uniformity"),
        }
    }

    if all_int {
        let mut out: Vec<i64> = Vec::with_capacity(rows.len());
        for r in rows { out.push(row_to_int(tape, r)); }
        return Val::IntVec(std::sync::Arc::new(out));
    }
    if all_float {
        let mut out: Vec<f64> = Vec::with_capacity(rows.len());
        for r in rows { out.push(row_to_float(tape, r)); }
        return Val::FloatVec(std::sync::Arc::new(out));
    }
    if all_str {
        let mut out: Vec<crate::strref::StrRef> = Vec::with_capacity(rows.len());
        let parent_str: std::sync::Arc<str> = unsafe {
            std::sync::Arc::from_raw(
                std::sync::Arc::into_raw(std::sync::Arc::clone(&tape.bytes_buf))
                    as *const str)
        };
        for r in rows {
            let (start, end) = row_to_str_range(tape, r);
            out.push(crate::strref::StrRef::slice(
                std::sync::Arc::clone(&parent_str),
                start as usize, end as usize));
        }
        return Val::StrSliceVec(std::sync::Arc::new(out));
    }
    // Mixed / containers / Mat — per-row materialise.
    let mut out: Vec<Val> = Vec::with_capacity(rows.len());
    for r in rows {
        out.push(match r {
            RowSrc::Entry(i) => Val::from_tape_node(tape, *i),
            RowSrc::Scalar(v) => tape_val_to_val(tape, *v),
            RowSrc::Mat(v) => v.clone(),
        });
    }
    Val::Arr(std::sync::Arc::new(out))
}

#[cfg(feature = "simd-json")]
fn tape_val_to_val(tape: &crate::strref::TapeData, v: TapeVal) -> Val {
    match v {
        TapeVal::Null => Val::Null,
        TapeVal::Bool(b) => Val::Bool(b),
        TapeVal::Int(n) => Val::Int(n),
        TapeVal::Float(f) => Val::Float(f),
        TapeVal::StrIdx(i) => {
            let s = match tape.nodes[i] {
                crate::strref::TapeNode::StringRef { start, end } =>
                    tape.str_at_range(start as usize, end as usize),
                _ => "",
            };
            Val::Str(std::sync::Arc::<str>::from(s))
        }
        TapeVal::Missing => Val::Null,
    }
}

fn tape_finalise(
    op: NumOp, si: i64, sf: f64, count: usize,
    min_f: f64, max_f: f64, is_float: bool,
) -> Val {
    if count == 0 {
        return match op {
            NumOp::Sum => Val::Int(0),
            _ => Val::Null,
        };
    }
    match op {
        NumOp::Sum => {
            if is_float { Val::Float(sf) } else { Val::Int(si) }
        }
        NumOp::Min => Val::Float(min_f),
        NumOp::Max => Val::Float(max_f),
        NumOp::Avg => {
            let total = if is_float { sf } else { si as f64 };
            Val::Float(total / count as f64)
        }
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

/// Columnar `$.<arr>.map(<field>).sum()` — extract numeric column,
/// SIMD-fold.  Returns `Val::Int` / `Val::Float` / `Val::Null` on
/// non-numeric.  Falls back through the existing scalar `Val::Obj`
/// `lookup_field_cached` for non-homogeneous Object shapes.
fn columnar_num_field(arr: &Arc<Vec<Val>>, field: &str, op: NumOp) -> Val {
    let mut acc_i: i64 = 0;
    let mut acc_f: f64 = 0.0;
    let mut floated = false;
    let mut min_f = f64::INFINITY;
    let mut max_f = f64::NEG_INFINITY;
    let mut n_obs: usize = 0;
    let mut idx: Option<usize> = None;
    for item in arr.iter() {
        if let Val::Obj(m) = item {
            let v = lookup_via_ic(m, field, &mut idx);
            num_fold(&mut acc_i, &mut acc_f, &mut floated, &mut min_f, &mut max_f, &mut n_obs,
                     op, v.unwrap_or(&Val::Null));
        }
    }
    num_finalise(op, acc_i, acc_f, floated, min_f, max_f, n_obs)
}

/// Columnar AND-chain filter count: every leaf comparison must hold.
fn columnar_filter_count_and(
    arr: &Arc<Vec<Val>>,
    leaves: &[(&str, crate::ast::BinOp, Val)],
) -> Val {
    let mut count: i64 = 0;
    let mut ics: Vec<Option<usize>> = vec![None; leaves.len()];
    for item in arr.iter() {
        if let Val::Obj(m) = item {
            let mut all = true;
            for (i, (f, op, lit)) in leaves.iter().enumerate() {
                match lookup_via_ic(m, f, &mut ics[i]) {
                    Some(v) if cmp_val_binop_local(v, *op, lit) => {}
                    _ => { all = false; break; }
                }
            }
            if all { count += 1; }
        }
    }
    Val::Int(count)
}

/// Columnar `$.<arr>.filter(<f> <op> <lit>).count()`.
fn columnar_filter_count(
    arr: &Arc<Vec<Val>>,
    pf:  &str,
    op:  crate::ast::BinOp,
    lit: &Val,
) -> Val {
    let mut count: i64 = 0;
    let mut idx: Option<usize> = None;
    for item in arr.iter() {
        if let Val::Obj(m) = item {
            if let Some(v) = lookup_via_ic(m, pf, &mut idx) {
                if cmp_val_binop_local(v, op, lit) { count += 1; }
            }
        }
    }
    Val::Int(count)
}

/// Columnar `$.<arr>.filter(<f> <op> <lit>).map(<g>).sum()`.
fn columnar_filter_num(
    arr: &Arc<Vec<Val>>,
    pf:  &str,
    cop: crate::ast::BinOp,
    lit: &Val,
    mf:  &str,
    op:  NumOp,
) -> Val {
    let mut acc_i: i64 = 0;
    let mut acc_f: f64 = 0.0;
    let mut floated = false;
    let mut min_f = f64::INFINITY;
    let mut max_f = f64::NEG_INFINITY;
    let mut n_obs: usize = 0;
    let mut ip: Option<usize> = None;
    let mut iq: Option<usize> = None;
    for item in arr.iter() {
        if let Val::Obj(m) = item {
            let pv = lookup_via_ic(m, pf, &mut ip);
            let pass = match pv { Some(v) => cmp_val_binop_local(v, cop, lit), None => false };
            if pass {
                let v = lookup_via_ic(m, mf, &mut iq).unwrap_or(&Val::Null);
                num_fold(&mut acc_i, &mut acc_f, &mut floated, &mut min_f, &mut max_f, &mut n_obs,
                         op, v);
            }
        }
    }
    num_finalise(op, acc_i, acc_f, floated, min_f, max_f, n_obs)
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

#[inline]
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

/// Legacy entry — used only by `try_columnar` paths that pre-build
/// a one-shot mini-pipeline.  Per-row pull loop must use
/// `apply_item_in_env` for the A1 fast path.
#[inline]
fn apply_item_root(vm: &mut crate::vm::VM, item: &Val, prog: &crate::vm::Program) -> Result<Val, EvalError> {
    vm.execute_val_raw(prog, item.clone())
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

    #[test]
    fn lower_filter_map_count() {
        if skip_under_composed() { return; }
        // Rewrite collapses:
        //   Map(id) ∘ Count → Count   (drop pure Map before Count)
        //   Filter(total>100) ∘ Count → CountIf(total>100)
        // so the lowered pipeline ends up with zero stages and a
        // CountIf sink.
        let p = lower_query("$.orders.filter(total > 100).map(id).count()").unwrap();
        assert_eq!(p.stages.len(), 0);
        assert!(matches!(p.sink, Sink::CountIf(_)));
    }

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

    #[test]
    fn debug_compound_pipeline_lower() {
        let q = r#"$.orders.filter(status == "shipped" and priority == "high").count()"#;
        let expr = crate::parser::parse(q).unwrap();
        let p = Pipeline::lower(&expr).unwrap();
        eprintln!("STAGES = {}", p.stages.len());
        match &p.sink {
            Sink::CountIf(prog) => eprintln!("PRED OPS = {:#?}", prog.ops),
            other => eprintln!("SINK = {:?}", std::any::type_name_of_val(other)),
        }
    }

    #[test]
    fn debug_full_pipeline_lower() {
        let expr = crate::parser::parse("$.orders.filter(total > 100).map(total).sum()").unwrap();
        let p = Pipeline::lower(&expr).unwrap();
        match &p.source { Source::FieldChain { keys } => eprintln!("KEYS = {:?}", keys), _ => {} }
        eprintln!("STAGES = {}", p.stages.len());
        match &p.sink {
            Sink::NumFilterMap(_, pred, map) => {
                eprintln!("PRED OPS = {:#?}", pred.ops);
                eprintln!("MAP OPS = {:#?}", map.ops);
            }
            other => eprintln!("SINK = {:?}", std::any::type_name_of_val(other)),
        }
    }

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

    #[test]
    fn rewrite_take_after_map_pushdown() {
        if skip_under_composed() { return; }
        let p = lower_query("$.xs.map(@ * 2).take(3).sum()").unwrap();
        // After pushdown: [Take(3), Map].  After Map+Sum fusion the
        // Map stage moves into the sink → stages = [Take(3)] +
        // SumMap sink.
        assert_eq!(p.stages.len(), 1);
        assert!(matches!(p.stages[0], Stage::Take(3)));
        assert!(matches!(p.sink, Sink::NumMap(NumOp::Sum, _)));
    }

    #[test]
    fn rewrite_sort_take_to_topn() {
        if skip_under_composed() { return; }
        let p = lower_query("$.xs.sort().take(3)").unwrap();
        assert_eq!(p.stages.len(), 0);
        assert!(matches!(p.sink, Sink::TopN { n: 3, asc: true, key: None }));
    }

    #[test]
    fn rewrite_sort_by_first_to_minby() {
        if skip_under_composed() { return; }
        let p = lower_query("$.xs.sort_by(score).first()").unwrap();
        assert_eq!(p.stages.len(), 0);
        assert!(matches!(p.sink, Sink::MinBy(_)));
    }

    #[test]
    fn rewrite_sort_by_last_to_maxby() {
        if skip_under_composed() { return; }
        let p = lower_query("$.xs.sort_by(score).last()").unwrap();
        assert_eq!(p.stages.len(), 0);
        assert!(matches!(p.sink, Sink::MaxBy(_)));
    }

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
