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
use crate::value::Val;

mod columnar;
mod common;
mod exec;
mod kernels;
mod lower;
mod plan;
pub(crate) use common::{
    apply_item_in_env, cmp_val_total, is_truthy, num_finalise, num_fold, walk_field_chain,
};
pub use kernels::{eval_cmp_op, eval_kernel, BodyKernel};
#[cfg(test)]
pub use plan::plan;
pub use plan::{
    compute_strategies, plan_with_kernels, select_strategy, Plan, Position, StageStrategy, Strategy,
};

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

// Layer B is on by default after bench validation. `JETRO_COMPOSED=0`
// opts out for diagnostic comparison against legacy. Cached after
// first read. Tier 3 step 2 deletes the gate entirely along with
// the fused Sink variants + ~30 fused vm.rs opcodes.
static COMPOSED_INIT: AtomicU8 = AtomicU8::new(0);

#[inline]
pub(crate) fn composed_path_enabled() -> bool {
    let v = COMPOSED_INIT.load(Ordering::Relaxed);
    if v != 0 {
        return v == 2;
    }
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
    /// `.sort()` (None) / `.sort_by(key)` (Some) — barrier;
    /// materialises and sorts.
    Sort(Option<Arc<crate::vm::Program>>),
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
        if skip_under_composed() {
            return;
        }
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
        assert_eq!(out, Val::Int(50)); // 20 + 30
    }
}
