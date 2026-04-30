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
use crate::builtins::{BuiltinMethod, BuiltinNumericReducer, BuiltinViewSink};
use crate::context::EvalError;
use crate::value::Val;

mod canonical;
mod capability;
mod collector;
mod columnar;
mod common;
mod composed_barrier;
mod composed_exec;
mod composed_segment;
mod composed_sink;
mod composed_source;
mod composed_stage;
mod exec;
mod indexed_exec;
mod kernels;
mod legacy_exec;
mod lower;
mod normalize;
mod plan;
mod row_source;
mod sink_accumulator;
pub(crate) use capability::{
    view_capabilities, view_prefix_capabilities, ViewInputMode, ViewMaterialization,
    ViewOutputMode, ViewSinkCapability, ViewStageCapability,
};
pub(crate) use collector::TerminalMapCollector;
pub(crate) use common::{
    apply_item_in_env, bounded_sort_by_key, bounded_sort_by_key_cmp, cmp_val_total, is_truthy,
    num_finalise, num_fold, walk_field_chain,
};
pub use kernels::{eval_cmp_op, eval_kernel, BodyKernel};
pub(crate) use kernels::{eval_view_kernel, CollectLayout, ObjectKernel, ViewKernelValue};
#[cfg(test)]
pub use plan::plan;
pub use plan::{
    compute_strategies, plan_with_exprs, plan_with_kernels, select_strategy, Plan, Position,
    StageStrategy, Strategy,
};
pub(crate) use sink_accumulator::SinkAccumulator;

#[cfg(feature = "simd-json")]
pub(crate) fn run_tape_field_chain(
    body: &PipelineBody,
    tape: &crate::strref::TapeData,
    keys: &[Arc<str>],
) -> Option<Result<Val, EvalError>> {
    legacy_exec::run_tape_field_chain(body, tape, keys)
}

/// Data capabilities supplied by the owning `Jetro` handle to pipeline
/// execution. The pipeline remains independent of `Jetro` itself, but can ask
/// for memoised ObjVec promotion.
pub trait PipelineData {
    fn promote_objvec(&self, arr: &Arc<Vec<Val>>) -> Option<Arc<crate::value::ObjVecData>>;
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
    if v != 0 {
        return v == 2;
    }
    let on = std::env::var_os("JETRO_PIPELINE_TRACE").is_some();
    TRACE_INIT.store(if on { 2 } else { 1 }, Ordering::Relaxed);
    on
}

fn sink_name(s: &Sink) -> &'static str {
    match s {
        Sink::Collect => "collect",
        Sink::Count(_) => "count",
        Sink::Numeric(n) => match n.op {
            NumOp::Sum => "sum",
            NumOp::Min => "min",
            NumOp::Max => "max",
            NumOp::Avg => "avg",
        },
        Sink::First(_) => "first",
        Sink::Last(_) => "last",
        Sink::ApproxCountDistinct => "approx_count_distinct",
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

/// Where a pipeline starts.
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

pub type PipelineBuiltinCall = crate::builtins::BuiltinCall;

/// Canonical internal sort description.
///
/// Surface spellings such as `.sort()`, `.sort(score)`,
/// `.sort(-score)`, and `.sort_by(score)` lower to this single shape.
/// `descending` is part of the sort metadata, so common descending key
/// queries do not need to evaluate unary negation for every input row.
#[derive(Debug, Clone)]
pub struct SortSpec {
    pub key: Option<Arc<crate::vm::Program>>,
    pub descending: bool,
}

impl SortSpec {
    pub fn identity() -> Self {
        Self {
            key: None,
            descending: false,
        }
    }

    pub fn keyed(key: Arc<crate::vm::Program>, descending: bool) -> Self {
        Self {
            key: Some(key),
            descending,
        }
    }
}

/// A pull-based stage.  Streaming stages (Filter / Map / FlatMap /
/// Take / Skip) flow elements through one at a time; barrier stages
/// (Reverse / Sort / UniqueBy) require the full input materialised.
/// Filter / Map / FlatMap / UniqueBy carry a pre-compiled `Program`
/// reused per row — drops per-row parse/compile overhead.
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
    /// `.sort()` / `.sort(key)` / `.sort_by(key)` — barrier; the
    /// planner may choose full sort, bounded top-k, or drop it when the
    /// downstream demand is order-insensitive.
    Sort(SortSpec),
    /// `.group_by(key)` — barrier; partitions rows by key,
    /// produces `Val::Obj { key_str → Vec<row> }`.  As a Stage
    /// this is a sink-shaped operation; placed under Stage so
    /// downstream `.values()` / `.map(@.len())` can compose.
    GroupBy(Arc<crate::vm::Program>),
    /// Pure per-element builtin lowered through `BuiltinMethod`.
    Builtin(PipelineBuiltinCall),

    // ── Step 3d-extension (C): lifted string Stages ──────────────────────────
    //
    // Lifts `.split(sep)` / `.slice(a, b)` from method-call
    // method calls inside Map/Filter sub-program bodies into first-class
    // Stage variants.  Two wins:
    //   1. Chain flattening (Step 3d-extension A) can hoist them out of
    //      Map bodies — `map(@.text.split(",").first())` becomes
    //      `[Map(@.text), Split(","), Sink::First]`.
    //   2. IndexedDispatch can compute `split(",").first()` directly
    //      instead of producing the full segment vector.
    /// `.split(sep)` — 1 string → many parts.  `Cardinality::Expanding`
    /// + `can_indexed=true` (kth segment via memchr).
    Split(Arc<str>),
    /// `.slice(start, end)` — 1 string → 1 substring.  `Cardinality::
    /// OneToOne` + `can_indexed=true`.  `end=None` means "to end".
    Slice(i64, Option<i64>),

    /// `.replace(needle, replacement)` (`all=false`, replacen-1-style) and
    /// `.replace_all(needle, replacement)` (`all=true`) — 1 string → 1
    /// string.  `Cardinality::OneToOne` + `can_indexed=true`.
    Replace {
        needle: Arc<str>,
        replacement: Arc<str>,
        all: bool,
    },

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

    // ── lift_all_builtins (lambda-bearing — Pipeline IR samples) ─────
    //
    // These mirror Filter/Map/FlatMap: carry a pre-compiled
    // `vm::Program` for the lambda body, evaluated per-row via
    // `eval_kernel`.  Streaming Stages (TakeWhile, DropWhile) run
    // inline with state; barrier Stages (IndicesWhere,
    // FindIndex, MaxBy, MinBy) materialise then consume buf once.
    /// `.takewhile(pred)` — emit while pred true; stop on first false.
    TakeWhile(Arc<crate::vm::Program>),
    /// `.dropwhile(pred)` — drop while pred true; emit rest.
    DropWhile(Arc<crate::vm::Program>),
    /// `.indices_where(pred)` — barrier; Arr<Int> of matching indices.
    IndicesWhere(Arc<crate::vm::Program>),
    /// `.find_index(pred)` — barrier; first matching index or Null.
    FindIndex(Arc<crate::vm::Program>),
    /// `.max_by(key)` — barrier; row with max key value.
    MaxBy(Arc<crate::vm::Program>),
    /// `.min_by(key)` — barrier; row with min key value.
    MinBy(Arc<crate::vm::Program>),
    /// `.transform_values(lam)` — per-Obj; new Obj with each value mapped.
    TransformValues(Arc<crate::vm::Program>),
    /// `.transform_keys(lam)` — per-Obj; new Obj with each key mapped.
    TransformKeys(Arc<crate::vm::Program>),
    /// `.filter_values(pred)` — per-Obj; keep entries whose value passes.
    FilterValues(Arc<crate::vm::Program>),
    /// `.filter_keys(pred)` — per-Obj; keep entries whose key passes.
    FilterKeys(Arc<crate::vm::Program>),
    /// `.count_by(key)` — barrier; Obj{key_str → count}.
    CountBy(Arc<crate::vm::Program>),
    /// `.index_by(key)` — barrier; Obj{key_str → row}.
    IndexBy(Arc<crate::vm::Program>),

    /// Algorithmic Category F: `UniqueBy(k) ∘ Sort(k)` merged.  One
    /// traversal: sort buf, then dedup_by adjacent (cache-friendly,
    /// avoids HashSet allocation).  Per
    /// `algorithmic_optimization_cold_only.md` Category F.
    SortedDedup(Option<Arc<crate::vm::Program>>),
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
    pub(crate) fn from_builtin_reducer(reducer: BuiltinNumericReducer) -> Self {
        match reducer {
            BuiltinNumericReducer::Sum => NumOp::Sum,
            BuiltinNumericReducer::Avg => NumOp::Avg,
            BuiltinNumericReducer::Min => NumOp::Min,
            BuiltinNumericReducer::Max => NumOp::Max,
        }
    }

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

/// Numeric aggregate sink. `project` lets lowering represent
/// `map(expr).sum()` as `sum(project=expr)` instead of a separate map stage.
/// This is a general aggregate-input projection, not a per-shape fused
/// builtin: filter/take/skip remain normal stages and all numeric aggregates
/// share the same sink representation.
#[derive(Debug, Clone)]
pub struct NumericSink {
    pub op: NumOp,
    pub project: Option<Arc<crate::vm::Program>>,
}

impl NumericSink {
    pub fn identity(op: NumOp) -> Self {
        Self { op, project: None }
    }

    pub fn projected(op: NumOp, project: Arc<crate::vm::Program>) -> Self {
        Self {
            op,
            project: Some(project),
        }
    }

    pub fn is_identity(&self) -> bool {
        self.project.is_none()
    }

    pub(crate) fn method(&self) -> BuiltinMethod {
        match self.op {
            NumOp::Sum => BuiltinMethod::Sum,
            NumOp::Min => BuiltinMethod::Min,
            NumOp::Max => BuiltinMethod::Max,
            NumOp::Avg => BuiltinMethod::Avg,
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
    Count(BuiltinViewSink),
    /// `.sum()`/`.min()`/`.max()`/`.avg()` over numerics.
    Numeric(NumericSink),
    /// `.first()` / `.last()` — yield the first/last element or
    /// `Val::Null`.
    First(BuiltinViewSink),
    Last(BuiltinViewSink),
    /// Algorithmic Category E: `.approx_count_distinct()` — HLL-12
    /// (~2KB state, ±2% accuracy) returning Int approximate count.
    /// Per `algorithmic_optimization_cold_only.md` Category E (opt-in
    /// approximate sink).
    ApproxCountDistinct,
}
#[derive(Debug, Clone)]
pub struct Pipeline {
    pub source: Source,
    pub stages: Vec<Stage>,
    /// Original AST bodies for expression-bearing stages, aligned with
    /// `stages`. Present for optimizer-only semantic rewrites; execution
    /// still uses compiled programs and kernel hints.
    pub stage_exprs: Vec<Option<Arc<Expr>>>,
    pub sink: Sink,
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
    pub sink_kernels: Vec<BodyKernel>,
}

/// Source-free executable pipeline body.
///
/// Physical plans use this for receiver pipelines whose input value is produced
/// by another physical node at runtime. Keeping the source separate avoids fake
/// placeholder receivers in planned IR while preserving the same executable
/// `Pipeline` representation once the receiver is known.
#[derive(Debug, Clone)]
pub struct PipelineBody {
    pub stages: Vec<Stage>,
    /// Original AST bodies for expression-bearing stages, aligned with
    /// `stages`. Present for optimizer-only semantic rewrites; execution
    /// still uses compiled programs and kernel hints.
    pub stage_exprs: Vec<Option<Arc<Expr>>>,
    pub sink: Sink,
    pub stage_kernels: Vec<BodyKernel>,
    pub sink_kernels: Vec<BodyKernel>,
}

impl PipelineBody {
    #[inline]
    pub fn with_source(self, source: Source) -> Pipeline {
        Pipeline {
            source,
            stages: self.stages,
            stage_exprs: self.stage_exprs,
            sink: self.sink,
            stage_kernels: self.stage_kernels,
            sink_kernels: self.sink_kernels,
        }
    }
}

impl Pipeline {
    #[inline]
    pub fn into_source_body(self) -> (Source, PipelineBody) {
        let body = PipelineBody {
            stages: self.stages,
            stage_exprs: self.stage_exprs,
            sink: self.sink,
            stage_kernels: self.stage_kernels,
            sink_kernels: self.sink_kernels,
        };
        (self.source, body)
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Arg, BinOp, Expr, Step};
    use crate::parser;

    fn lower_query(q: &str) -> Option<Pipeline> {
        let expr = parser::parse(q).ok()?;
        Pipeline::lower(&expr)
    }

    fn only_stage_expr(p: &Pipeline) -> &Expr {
        assert_eq!(p.stage_exprs.len(), p.stages.len());
        p.stage_exprs[0]
            .as_ref()
            .expect("expected optimized stage expression")
            .as_ref()
    }

    fn assert_price_qty_gt_100(expr: &Expr) {
        match expr {
            Expr::BinOp(lhs, BinOp::Gt, rhs) => {
                assert_price_qty_mul(lhs);
                assert!(matches!(rhs.as_ref(), Expr::Int(100)), "{expr:#?}");
            }
            _ => panic!("expected `(price * qty) > 100`, got {expr:#?}"),
        }
    }

    fn assert_price_qty_mul(expr: &Expr) {
        match expr {
            Expr::BinOp(lhs, BinOp::Mul, rhs) => {
                assert!(
                    matches!(lhs.as_ref(), Expr::Ident(name) if name == "price"),
                    "{expr:#?}"
                );
                assert!(
                    matches!(rhs.as_ref(), Expr::Ident(name) if name == "qty"),
                    "{expr:#?}"
                );
            }
            _ => panic!("expected `price * qty`, got {expr:#?}"),
        }
    }

    fn assert_pipeline_matches_vm(query: &str, doc: serde_json::Value) {
        assert_pipeline_matches_vm_query(query, query, doc);
    }

    fn assert_pipeline_matches_vm_query(
        pipeline_query: &str,
        vm_query: &str,
        doc: serde_json::Value,
    ) {
        let pipeline = lower_query(pipeline_query).expect("query should lower to pipeline");
        let root = Val::from(&doc);
        let actual: serde_json::Value = pipeline
            .run(&root)
            .expect("pipeline execution should succeed")
            .into();

        let mut vm = crate::vm::VM::new();
        let expected = vm
            .run_str(vm_query, &doc)
            .expect("VM execution should succeed");

        assert_eq!(
            actual, expected,
            "pipeline diverged from VM for {pipeline_query}"
        );
    }

    #[test]
    fn lower_field_chain_only() {
        let p = lower_query("$.a.b.c").unwrap();
        assert!(matches!(p.source, Source::FieldChain { .. }));
        assert!(p.stages.is_empty());
        assert!(matches!(p.sink, Sink::Collect));
    }

    #[test]
    fn row_source_keeps_objvec_as_streaming_provider() {
        let keys: std::sync::Arc<[std::sync::Arc<str>]> =
            vec![std::sync::Arc::<str>::from("id")].into();
        let data = std::sync::Arc::new(crate::value::ObjVecData {
            keys,
            cells: vec![Val::Int(1), Val::Int(2)],
            typed_cols: None,
        });
        let recv = Val::ObjVec(std::sync::Arc::clone(&data));

        let source = row_source::ValRowSource::from_receiver(&recv);
        assert!(source.is_objvec_streaming());

        let mut iter = source.iter();
        assert!(matches!(iter, row_source::ValRowsIter::ObjVec { .. }));
        assert_eq!(iter.next().unwrap().get_field("id"), Val::Int(1));
        assert_eq!(iter.next().unwrap().get_field("id"), Val::Int(2));
        assert!(iter.next().is_none());
    }

    #[cfg(feature = "simd-json")]
    #[test]
    fn tape_row_source_walks_field_chain_array_lazily() {
        let tape =
            crate::strref::TapeData::parse(br#"{"books":[{"id":1},{"id":2}],"skip":[3]}"#.to_vec())
                .unwrap();
        let keys = vec![std::sync::Arc::<str>::from("books")];

        let source = row_source::TapeRowSource::from_field_chain(&tape, &keys);
        assert!(source.is_array_provider());

        use crate::value_view::ValueView;

        let mut iter = source.iter_views();
        assert!(matches!(iter, row_source::TapeRowsIter::Array { .. }));
        assert_eq!(
            iter.next().unwrap().materialize().get_field("id"),
            Val::Int(1)
        );
        assert_eq!(
            iter.next().unwrap().materialize().get_field("id"),
            Val::Int(2)
        );
        assert!(iter.next().is_none());
    }

    #[test]
    fn receiver_pipeline_start_uses_builtin_metadata() {
        assert!(Pipeline::is_receiver_pipeline_start(&Step::Method(
            "filter".into(),
            vec![Arg::Pos(Expr::Bool(true))]
        )));
        assert!(Pipeline::is_receiver_pipeline_start(&Step::Method(
            "sum".into(),
            Vec::new()
        )));
        assert!(Pipeline::is_receiver_pipeline_start(&Step::Method(
            "first".into(),
            Vec::new()
        )));
        assert!(Pipeline::is_receiver_pipeline_start(&Step::Method(
            "countBy".into(),
            vec![Arg::Pos(Expr::Ident("kind".into()))]
        )));

        assert!(!Pipeline::is_receiver_pipeline_start(&Step::Method(
            "from_json".into(),
            Vec::new()
        )));
    }

    // `lower_filter_map_count` removed — fused Sink::CountIf variant
    // deleted in Tier 3. Lowered shape is now [Filter] + Sink::Count.

    #[test]
    fn lower_take_skip_sum() {
        let p = lower_query("$.xs.skip(2).take(5).sum()").unwrap();
        assert_eq!(p.stages.len(), 2);
        assert!(matches!(p.stages[0], Stage::Skip(2)));
        assert!(matches!(p.stages[1], Stage::Take(5)));
        assert!(matches!(&p.sink, Sink::Numeric(n) if n.op == NumOp::Sum));
    }

    #[test]
    fn lower_pure_builtin_uses_generic_builtin_stage() {
        let p = lower_query("$.names.upper().starts_with(\"A\")").unwrap();
        assert_eq!(p.stages.len(), 2);
        assert!(matches!(
            p.stages[0],
            Stage::Builtin(PipelineBuiltinCall {
                method: crate::builtins::BuiltinMethod::Upper,
                ..
            })
        ));
        assert!(matches!(
            p.stages[1],
            Stage::Builtin(PipelineBuiltinCall {
                method: crate::builtins::BuiltinMethod::StartsWith,
                ..
            })
        ));
        assert!(matches!(p.sink, Sink::Collect));
    }

    #[test]
    fn lower_pipeline_builtin_uses_builtin_owned_allowlist() {
        let p = lower_query("$.items.schema()").unwrap();
        assert_eq!(p.stages.len(), 1);
        assert!(matches!(
            p.stages[0],
            Stage::Builtin(PipelineBuiltinCall {
                method: crate::builtins::BuiltinMethod::Schema,
                ..
            })
        ));
    }

    #[test]
    fn lower_whole_receiver_builtin_not_as_per_element_stage() {
        assert!(lower_query("$.items.compact()").is_none());
        assert!(lower_query("$.items.join(\",\")").is_none());
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
    fn method_chain_scalar_filter_lowers_from_builtin_view_metadata() {
        let p = lower_query("$.people.filter(name.len() == 3).take(1).map(name)").unwrap();

        assert!(matches!(
            &p.stage_kernels[0],
            BodyKernel::CmpLit { lhs, .. }
                if matches!(
                    lhs.as_ref(),
                    BodyKernel::BuiltinCall { receiver, call }
                        if call.spec().view_scalar
                            && matches!(receiver.as_ref(), BodyKernel::FieldRead(k) if k.as_ref() == "name")
                )
        ));
        assert!(p.stage_kernels[0].is_view_native());
    }

    #[test]
    fn method_chain_scalar_filter_lowers_static_arg_view_builtin() {
        let p = lower_query(r#"$.people.filter(name.starts_with("a")).take(1).map(name)"#).unwrap();

        assert!(matches!(
            &p.stage_kernels[0],
            BodyKernel::BuiltinCall { receiver, call }
                if call.spec().view_scalar
                    && call.method == BuiltinMethod::StartsWith
                    && matches!(receiver.as_ref(), BodyKernel::FieldRead(k) if k.as_ref() == "name")
        ));
        assert!(p.stage_kernels[0].is_view_native());
    }

    #[test]
    fn method_chain_scalar_filter_lowers_another_static_arg_view_builtin() {
        let p = lower_query(r#"$.people.filter(name.ends_with("a")).take(1).map(name)"#).unwrap();

        assert!(matches!(
            &p.stage_kernels[0],
            BodyKernel::BuiltinCall { receiver, call }
                if call.spec().view_scalar
                    && call.method == BuiltinMethod::EndsWith
                    && matches!(receiver.as_ref(), BodyKernel::FieldRead(k) if k.as_ref() == "name")
        ));
        assert!(p.stage_kernels[0].is_view_native());
    }

    #[test]
    fn method_chain_scalar_filter_lowers_string_predicate_view_builtin() {
        let p = lower_query(r#"$.people.filter(name.matches("ad")).take(1).map(name)"#).unwrap();

        assert!(matches!(
            &p.stage_kernels[0],
            BodyKernel::BuiltinCall { receiver, call }
                if call.spec().view_scalar
                    && call.method == BuiltinMethod::Matches
                    && matches!(receiver.as_ref(), BodyKernel::FieldRead(k) if k.as_ref() == "name")
        ));
        assert!(p.stage_kernels[0].is_view_native());
    }

    #[test]
    fn method_chain_scalar_filter_lowers_string_index_view_builtin() {
        let p =
            lower_query(r#"$.people.filter(name.index_of("d") >= 1).take(1).map(name)"#).unwrap();

        assert!(matches!(
            &p.stage_kernels[0],
            BodyKernel::CmpLit { lhs, .. }
                if matches!(
                    lhs.as_ref(),
                    BodyKernel::BuiltinCall { receiver, call }
                        if call.spec().view_scalar
                            && call.method == BuiltinMethod::IndexOf
                            && matches!(receiver.as_ref(), BodyKernel::FieldRead(k) if k.as_ref() == "name")
                )
        ));
        assert!(p.stage_kernels[0].is_view_native());
    }

    #[test]
    fn method_chain_scalar_filter_lowers_no_arg_string_view_builtin() {
        let p = lower_query(r#"$.people.filter(code.is_numeric()).take(1).map(code)"#).unwrap();

        assert!(matches!(
            &p.stage_kernels[0],
            BodyKernel::BuiltinCall { receiver, call }
                if call.spec().view_scalar
                    && call.method == BuiltinMethod::IsNumeric
                    && matches!(receiver.as_ref(), BodyKernel::FieldRead(k) if k.as_ref() == "code")
        ));
        assert!(p.stage_kernels[0].is_view_native());
    }

    #[test]
    fn method_chain_scalar_filter_lowers_no_arg_numeric_string_view_builtin() {
        let p = lower_query(r#"$.people.filter(code.byte_len() == 3).take(1).map(code)"#).unwrap();

        assert!(matches!(
            &p.stage_kernels[0],
            BodyKernel::CmpLit { lhs, .. }
                if matches!(
                    lhs.as_ref(),
                    BodyKernel::BuiltinCall { receiver, call }
                        if call.spec().view_scalar
                            && call.method == BuiltinMethod::ByteLen
                            && matches!(receiver.as_ref(), BodyKernel::FieldRead(k) if k.as_ref() == "code")
                )
        ));
        assert!(p.stage_kernels[0].is_view_native());
    }

    #[test]
    fn method_chain_scalar_filter_lowers_numeric_view_builtin() {
        let p = lower_query(r#"$.people.filter(score.abs() > 10).take(1).map(score)"#).unwrap();

        assert!(matches!(
            &p.stage_kernels[0],
            BodyKernel::CmpLit { lhs, .. }
                if matches!(
                    lhs.as_ref(),
                    BodyKernel::BuiltinCall { receiver, call }
                        if call.spec().view_scalar
                            && call.method == BuiltinMethod::Abs
                            && matches!(receiver.as_ref(), BodyKernel::FieldRead(k) if k.as_ref() == "score")
                )
        ));
        assert!(p.stage_kernels[0].is_view_native());
    }

    #[test]
    fn method_chain_scalar_filter_lowers_float_numeric_view_builtin() {
        let p = lower_query(r#"$.people.filter(score.round() == 10).take(1).map(score)"#).unwrap();

        assert!(matches!(
            &p.stage_kernels[0],
            BodyKernel::CmpLit { lhs, .. }
                if matches!(
                    lhs.as_ref(),
                    BodyKernel::BuiltinCall { receiver, call }
                        if call.spec().view_scalar
                            && call.method == BuiltinMethod::Round
                            && matches!(receiver.as_ref(), BodyKernel::FieldRead(k) if k.as_ref() == "score")
                )
        ));
        assert!(p.stage_kernels[0].is_view_native());
    }

    // `debug_compound_pipeline_lower` and `debug_full_pipeline_lower`
    // removed — referenced fused Sink::CountIf / Sink::NumFilterMap.

    #[test]
    fn run_count_on_simple_array() {
        use serde_json::json;
        let doc: Val = (&json!({"orders":[
            {"total": 50}, {"total": 150}, {"total": 200}
        ]}))
            .into();
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
        ]}))
            .into();
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
                assert!(prog
                    .ops
                    .iter()
                    .any(|o| matches!(o, crate::vm::Opcode::AndOp(_))));
            }
            _ => panic!("expected merged Filter"),
        }
    }

    #[test]
    fn rewrite_map_then_count_drops_map() {
        let p = lower_query("$.orders.map(total).count()").unwrap();
        assert_eq!(p.stages.len(), 0);
        assert!(matches!(p.sink, Sink::Count(_)));
    }

    #[test]
    fn demand_optimizer_drops_value_only_work_for_count() {
        let p = lower_query("$.orders.map(total).upper().count()").unwrap();
        assert!(p.stages.is_empty());
        assert!(matches!(p.sink, Sink::Count(_)));
    }

    #[test]
    fn demand_optimizer_pulls_filter_through_map_for_count() {
        let p = lower_query("$.orders.map(total).filter(@ > 10).count()").unwrap();
        assert_eq!(p.stages.len(), 1);
        assert!(matches!(p.stages[0], Stage::Filter(_)));
        assert!(matches!(p.sink, Sink::Count(_)));
    }

    #[test]
    fn demand_optimizer_pulls_filter_through_computed_map_for_count() {
        let p = lower_query("$.orders.map(price * qty).filter(@ > 100).count()").unwrap();
        assert_eq!(p.stages.len(), 1);
        assert!(matches!(p.stages[0], Stage::Filter(_)));
        assert_price_qty_gt_100(only_stage_expr(&p));
        assert!(matches!(p.sink, Sink::Count(_)));
    }

    #[test]
    fn demand_optimizer_pulls_filter_through_method_chain_map_for_count() {
        let p =
            lower_query("$.users.map(name.trim().upper()).filter(@ == \"ADA\").count()").unwrap();
        assert_eq!(p.stages.len(), 1);
        assert!(matches!(p.stages[0], Stage::Filter(_)));
        assert!(matches!(p.sink, Sink::Count(_)));
    }

    #[test]
    fn demand_optimizer_simplifies_object_projection_after_substitution() {
        let p = lower_query("$.orders.map({v: price * qty, id: id}).filter(@.v > 100).count()")
            .unwrap();
        assert_eq!(p.stages.len(), 1);
        assert!(matches!(p.stages[0], Stage::Filter(_)));
        assert_price_qty_gt_100(only_stage_expr(&p));
        assert!(matches!(p.sink, Sink::Count(_)));
    }

    #[test]
    fn demand_optimizer_simplifies_array_projection_after_substitution() {
        let p = lower_query("$.orders.map([price * qty, id]).filter(@[0] > 100).count()").unwrap();
        assert_eq!(p.stages.len(), 1);
        assert!(matches!(p.stages[0], Stage::Filter(_)));
        assert_price_qty_gt_100(only_stage_expr(&p));
        assert!(matches!(p.sink, Sink::Count(_)));
    }

    #[test]
    fn demand_optimizer_pulls_filter_through_map_but_keeps_map_for_collect() {
        let p = lower_query("$.orders.map(total).filter(@ > 10)").unwrap();
        assert_eq!(p.stages.len(), 2);
        assert!(matches!(p.stages[0], Stage::Filter(_)));
        assert!(matches!(p.stages[1], Stage::Map(_)));
        assert!(matches!(p.sink, Sink::Collect));
    }

    #[test]
    fn demand_optimizer_removes_order_only_work_for_numeric_sink() {
        let p = lower_query("$.orders.sort().reverse().map(total).sum()").unwrap();
        assert!(p.stages.is_empty());
        assert!(matches!(&p.sink, Sink::Numeric(n) if n.op == NumOp::Sum && n.project.is_some()));
    }

    #[test]
    fn demand_optimizer_keeps_membership_work_for_numeric_sink() {
        let p = lower_query("$.orders.take(2).map(total).sum()").unwrap();
        assert_eq!(p.stages.len(), 1);
        assert!(matches!(p.stages[0], Stage::Take(2)));
        assert!(matches!(&p.sink, Sink::Numeric(n) if n.op == NumOp::Sum && n.project.is_some()));
    }

    #[test]
    fn demand_optimizer_computed_map_filter_count_matches_vm() {
        use serde_json::json;
        assert_pipeline_matches_vm(
            "$.orders.map(price * qty).filter(@ > 100).count()",
            json!({
                "orders": [
                    {"price": 10, "qty": 3},
                    {"price": 25, "qty": 5},
                    {"price": 8, "qty": 20},
                    {"price": 50, "qty": 1}
                ]
            }),
        );
    }

    #[test]
    fn demand_optimizer_object_projection_filter_count_matches_vm() {
        use serde_json::json;
        assert_pipeline_matches_vm(
            "$.orders.map({v: price * qty, id: id}).filter(@.v > 100).count()",
            json!({
                "orders": [
                    {"id": "a", "price": 10, "qty": 3},
                    {"id": "b", "price": 25, "qty": 5},
                    {"id": "c", "price": 8, "qty": 20},
                    {"id": "d", "price": 50, "qty": 1}
                ]
            }),
        );
    }

    #[test]
    fn demand_optimizer_array_projection_filter_count_matches_vm() {
        use serde_json::json;
        assert_pipeline_matches_vm(
            "$.orders.map([price * qty, id]).filter(@[0] > 100).count()",
            json!({
                "orders": [
                    {"id": "a", "price": 10, "qty": 3},
                    {"id": "b", "price": 25, "qty": 5},
                    {"id": "c", "price": 8, "qty": 20},
                    {"id": "d", "price": 50, "qty": 1}
                ]
            }),
        );
    }

    #[test]
    fn demand_optimizer_order_removal_numeric_sink_matches_vm() {
        use serde_json::json;
        assert_pipeline_matches_vm(
            "$.orders.sort().reverse().map(total).sum()",
            json!({
                "orders": [
                    {"id": 1, "total": 7},
                    {"id": 2, "total": 40},
                    {"id": 3, "total": -2},
                    {"id": 4, "total": 9}
                ]
            }),
        );
    }

    #[test]
    fn demand_optimizer_projected_numeric_sink_with_take_matches_vm() {
        use serde_json::json;
        assert_pipeline_matches_vm_query(
            "$.orders.take(2).map(total).sum()",
            "$.orders.first(2).map(total).sum()",
            json!({
                "orders": [
                    {"id": 1, "total": 7},
                    {"id": 2, "total": 40},
                    {"id": 3, "total": -2},
                    {"id": 4, "total": 9}
                ]
            }),
        );
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
    fn sort_key_spellings_lower_to_canonical_sort_spec() {
        let p = lower_query("$.rows.sort(-score).take(10)").unwrap();
        assert!(matches!(&p.stages[0], Stage::Sort(spec) if spec.key.is_some() && spec.descending));
        assert!(
            matches!(p.stage_kernels[0], BodyKernel::FieldRead(ref k) if k.as_ref() == "score")
        );

        let p = lower_query("$.rows.sort_by(-score).take(10)").unwrap();
        assert!(matches!(&p.stages[0], Stage::Sort(spec) if spec.key.is_some() && spec.descending));
        assert!(
            matches!(p.stage_kernels[0], BodyKernel::FieldRead(ref k) if k.as_ref() == "score")
        );
    }

    #[test]
    fn descending_sort_take_uses_bounded_topk_strategy_and_matches_vm() {
        use serde_json::json;

        let p = lower_query("$.rows.sort_by(-score).take(2)").unwrap();
        let strategies = compute_strategies(&p.stages, &p.sink);
        assert!(matches!(strategies[0], StageStrategy::SortTopK(2)));

        assert_pipeline_matches_vm_query(
            "$.rows.sort_by(-score).take(2)",
            "$.rows.sort(-score).first(2)",
            json!({
                "rows": [
                    {"id": 1, "score": 10},
                    {"id": 2, "score": 30},
                    {"id": 3, "score": 20}
                ]
            }),
        );
    }

    #[test]
    fn first_sink_stops_after_first_passing_filter_row() {
        use serde_json::json;
        let doc: Val = (&json!({
            "data": [
                {"score": 901, "bad": 0},
                {"score": 1, "bad": "not a number"}
            ]
        }))
            .into();
        let p = lower_query("$.data.filter(score > 900 or 1 / 0 > 0).first()").unwrap();
        let out = p.run(&doc).unwrap();
        let out_json: serde_json::Value = out.into();
        assert_eq!(out_json, json!({"score": 901, "bad": 0}));
    }

    #[test]
    fn take_before_filter_caps_upstream_inputs() {
        use serde_json::json;
        let doc: Val = (&json!({
            "data": [
                {"score": 1},
                {"score": 2},
                {"score": 901}
            ]
        }))
            .into();
        let p = lower_query("$.data.take(2).filter(score > 900).first()").unwrap();
        let demand = p.source_demand();
        assert_eq!(demand.chain.pull, crate::chain_ir::PullDemand::AtMost(2));
        let out = p.run(&doc).unwrap();
        assert_eq!(out, Val::Null);
    }

    #[test]
    fn filter_take_collect_stops_after_required_outputs() {
        use serde_json::json;
        let doc: Val = (&json!({
            "data": [
                {"score": 901, "bad": 0},
                {"score": 902, "bad": 0},
                {"score": 1, "bad": "not a number"}
            ]
        }))
            .into();
        let p = lower_query("$.data.filter(score > 900 or 1 / 0 > 0).take(2)").unwrap();
        let demand = p.source_demand();
        assert_eq!(
            demand.chain.pull,
            crate::chain_ir::PullDemand::UntilOutput(2)
        );
        let out = p.run(&doc).unwrap();
        let out_json: serde_json::Value = out.into();
        assert_eq!(
            out_json,
            json!([{"score": 901, "bad": 0}, {"score": 902, "bad": 0}])
        );
    }

    #[test]
    fn last_sink_keeps_full_scan_requirement() {
        use serde_json::json;
        let doc: Val = (&json!({
            "data": [
                {"score": 901},
                {"score": 902}
            ]
        }))
            .into();
        let p = lower_query("$.data.filter(score > 900).last()").unwrap();
        let demand = p.source_demand();
        assert_eq!(demand.chain.pull, crate::chain_ir::PullDemand::All);
        let out = p.run(&doc).unwrap();
        let out_json: serde_json::Value = out.into();
        assert_eq!(out_json, json!({"score": 902}));
    }

    #[test]
    fn rewrite_filter_const_true_dropped() {
        // `true` literal — Filter(true) collapses to id.
        let p = lower_query("$.xs.filter(true).count()").unwrap();
        assert_eq!(p.stages.len(), 0);
        assert!(matches!(p.sink, Sink::Count(_)));
    }

    #[test]
    fn run_take_skip() {
        use serde_json::json;
        let doc: Val = (&json!({"xs":[10, 20, 30, 40, 50]})).into();
        let p = lower_query("$.xs.skip(1).take(2).sum()").unwrap();
        let out = p.run(&doc).unwrap();
        assert_eq!(out, Val::Int(50)); // 20 + 30
    }
}
