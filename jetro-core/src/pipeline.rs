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
        Self::Generic
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
        Some(Val::ObjVec(Arc::new(crate::eval::value::ObjVecData {
            keys: keys.into(),
            cells,
        })))
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
        // Phase 3 columnar fast path — runs before per-row loop.
        if let Some(out) = self.try_columnar(root) { return out; }

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
            Stage::Reverse | Stage::Sort(_) | Stage::UniqueBy(_) | Stage::FlatMap(_)));
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
                        Stage::Reverse | Stage::Sort(_) | Stage::UniqueBy(_) | Stage::FlatMap(_) => {}
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
            Sink::Collect           => Val::arr(acc_collect),
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

fn objvec_filter_count_and_slots(
    d: &Arc<crate::eval::value::ObjVecData>,
    leaves: &[(usize, crate::ast::BinOp, Val)],
) -> Val {
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
        BodyKernel::Generic => apply_item_in_env(vm, env, item, fallback),
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
        // Lambda-bodied filter not yet supported.
        assert!(lower_query("$.xs.group_by(status)").is_none());
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
        let p = lower_query("$.orders.map(total).count()").unwrap();
        assert_eq!(p.stages.len(), 0);
        assert!(matches!(p.sink, Sink::Count));
    }

    #[test]
    fn rewrite_take_after_map_pushdown() {
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
        let p = lower_query("$.xs.sort().take(3)").unwrap();
        assert_eq!(p.stages.len(), 0);
        assert!(matches!(p.sink, Sink::TopN { n: 3, asc: true, key: None }));
    }

    #[test]
    fn rewrite_sort_by_first_to_minby() {
        let p = lower_query("$.xs.sort_by(score).first()").unwrap();
        assert_eq!(p.stages.len(), 0);
        assert!(matches!(p.sink, Sink::MinBy(_)));
    }

    #[test]
    fn rewrite_sort_by_last_to_maxby() {
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
