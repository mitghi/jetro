//! Pull-based pipeline IR for streamable query chains.
//!
//! A query like `$.orders.filter(total > 100).map(id).count()` lowers to a
//! `Pipeline` with a `Source`, a list of `Stage`s, and a `Sink`. Execution is
//! a single outer loop: pull one element from the source, thread it through
//! stages, write into the sink — no intermediate `Vec<Val>` between stages.
//! Queries that fall outside the supported shape set return `None` from
//! `Pipeline::lower` and fall back to the VM opcode path.

use std::sync::Arc;

use crate::ast::Expr;
use crate::builtins::{
    BuiltinCancellation, BuiltinMethod, BuiltinNumericReducer, BuiltinViewStage,
};
use crate::context::{Env, EvalError};
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
mod operator;
mod plan;
mod reducer;
mod row_source;
mod sink_accumulator;
mod stage_factory;
mod stage_flow;
mod val_stage_flow;
pub(crate) use capability::{
    view_capabilities, view_prefix_capabilities, ViewInputMode, ViewMaterialization,
    ViewOutputMode, ViewSinkCapability, ViewStageCapability,
};
pub(crate) use collector::{TerminalCollector, TerminalMapCollector};
pub(crate) use common::{
    apply_item_in_env, bounded_sort_by_key, bounded_sort_by_key_cmp, cmp_val_total, is_truthy,
    num_finalise, num_fold, ordered_by_key_cmp, walk_field_chain, BoundedKeySorter,
    OrderedKeySorter,
};
pub use kernels::{eval_cmp_op, eval_kernel, BodyKernel};
pub(crate) use kernels::{eval_view_kernel, CollectLayout, ObjectKernel, ViewKernelValue};
pub use operator::{ReducerOp, ReducerSpec};
#[cfg(test)]
pub use plan::compute_strategies;
#[cfg(test)]
pub use plan::plan;
pub use plan::{
    compute_strategies_with_kernels, plan_with_exprs, plan_with_kernels, select_strategy, Plan,
    Position, StageStrategy, Strategy,
};
pub(crate) use reducer::ReducerAccumulator;
pub(crate) use sink_accumulator::SinkAccumulator;
pub(crate) use stage_flow::{stage_executor, StageFlow};

#[cfg(feature = "simd-json")]
/// Executes the field-chain traversal of `body` against a borrowed simd-json tape, returning
/// the first matching value or `None` if the shape is not tape-compatible.
pub(crate) fn run_tape_field_chain(
    body: &PipelineBody,
    tape: &crate::strref::TapeData,
    keys: &[Arc<str>],
    base_env: &Env,
) -> Option<Result<Val, EvalError>> {
    legacy_exec::run_tape_field_chain(body, tape, keys, base_env)
}

/// Extension point allowing the host (e.g. `Jetro`) to upgrade a flat `Arc<Vec<Val>>` array
/// into a columnar `ObjVecData` representation for zero-copy row iteration.
pub trait PipelineData {
    /// Attempts to interpret `arr` as a columnar object-vector, returning `None` if the
    /// array layout does not match the expected uniform-key schema.
    fn promote_objvec(&self, arr: &Arc<Vec<Val>>) -> Option<Arc<crate::value::ObjVecData>>;
}

use std::sync::atomic::{AtomicU8, Ordering};
/// Cached initialisation flag for the pipeline trace flag: 0 = uninitialised, 1 = off, 2 = on.
static TRACE_INIT: AtomicU8 = AtomicU8::new(0);

/// Returns `true` when the `JETRO_PIPELINE_TRACE` environment variable is set, caching the
/// result in `TRACE_INIT` so the env lookup happens at most once per process.
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

/// Returns a short human-readable label for `s`, used in pipeline trace output.
fn sink_name(s: &Sink) -> &'static str {
    match s {
        Sink::Collect => "collect",
        Sink::Reducer(spec) => match spec.op {
            ReducerOp::Count => "count",
            ReducerOp::Numeric(NumOp::Sum) => "sum",
            ReducerOp::Numeric(NumOp::Min) => "min",
            ReducerOp::Numeric(NumOp::Max) => "max",
            ReducerOp::Numeric(NumOp::Avg) => "avg",
        },
        Sink::Terminal(BuiltinMethod::First) => "first",
        Sink::Terminal(BuiltinMethod::Last) => "last",
        Sink::Terminal(_) => "terminal",
        Sink::ApproxCountDistinct => "approx_count_distinct",
    }
}

/// Returns a short human-readable label for `s`, used in pipeline trace output.
fn source_name(s: &Source) -> &'static str {
    match s {
        Source::Receiver(_) => "receiver",
        Source::FieldChain { .. } => "field_chain",
    }
}

/// Returns a short label for the top-level `Expr` variant, used in pipeline fallback trace messages.
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

/// The value-producing root of a pipeline — either an already-materialised value or a
/// dot-separated field path that is resolved against the document before the first stage runs.
#[derive(Debug, Clone)]
pub enum Source {
    /// A pre-evaluated value handed directly to the pipeline as its starting collection;
    /// used when the query begins from an existing `Val` rather than a document field path.
    Receiver(Val),

    /// A sequence of field names resolved left-to-right from the document root (`$`),
    /// producing the array (or scalar) that feeds the first pipeline stage.
    FieldChain { keys: Arc<[Arc<str>]> },
}

/// Type alias for a fully-resolved built-in call used as a pipeline stage.
pub type PipelineBuiltinCall = crate::builtins::BuiltinCall;

/// Describes the sort order for a `Stage::Sort` stage, optionally with a key-extraction program.
#[derive(Debug, Clone)]
pub struct SortSpec {
    /// Compiled key-extraction expression, or `None` for natural (value-level) ordering.
    pub key: Option<Arc<crate::vm::Program>>,
    /// When `true` the sort is highest-first; when `false` it is lowest-first.
    pub descending: bool,
}

impl SortSpec {
    /// Creates a `SortSpec` with no key expression and ascending order.
    pub fn identity() -> Self {
        Self {
            key: None,
            descending: false,
        }
    }

    /// Creates a `SortSpec` with the given compiled key program and ordering direction.
    pub fn keyed(key: Arc<crate::vm::Program>, descending: bool) -> Self {
        Self {
            key: Some(key),
            descending,
        }
    }
}

/// A single transformation step in a pull-based pipeline between the source and the sink.
///
/// Each variant carries the compiled predicate / projection program and any metadata needed
/// to select the correct execution path (view-native, VM fallback, etc.).
#[derive(Debug, Clone)]
pub enum Stage {
    /// Retains only elements for which the predicate program yields a truthy value.
    Filter(Arc<crate::vm::Program>, BuiltinViewStage),
    /// Transforms each element by evaluating the projection program against it.
    Map(Arc<crate::vm::Program>, BuiltinViewStage),

    /// Evaluates the program for each element and concatenates the resulting arrays into the stream.
    FlatMap(Arc<crate::vm::Program>, BuiltinViewStage),
    /// Reverses the element order; cancels with an adjacent `Reverse` during plan fusion.
    Reverse(BuiltinCancellation),

    /// Removes duplicates, optionally keyed by a projection program; uses identity equality
    /// when the program is `None`.
    UniqueBy(Option<Arc<crate::vm::Program>>),

    /// Sorts all elements using the `SortSpec`; may be fused into a bounded top-k heap by
    /// the planner when followed by `Take`.
    Sort(SortSpec),

    /// Delegates each element to a pure built-in method call with pre-resolved literal arguments.
    Builtin(PipelineBuiltinCall),

    /// `usize`-argument builtin stage identified by the registry.
    UsizeBuiltin {
        /// Registry method identity for this stage.
        method: BuiltinMethod,
        /// Integer argument accepted by the builtin.
        value: usize,
    },

    /// Single string-argument builtin stage identified by the registry.
    StringBuiltin {
        /// Registry method identity for this stage.
        method: BuiltinMethod,
        /// String argument accepted by the builtin.
        value: Arc<str>,
    },

    /// Two string-argument builtin stage identified by the registry.
    StringPairBuiltin {
        /// Registry method identity for this stage.
        method: BuiltinMethod,
        /// First string argument accepted by the builtin.
        first: Arc<str>,
        /// Second string argument accepted by the builtin.
        second: Arc<str>,
    },

    /// Integer-range builtin stage identified by the registry.
    IntRangeBuiltin {
        /// Registry method identity for this stage.
        method: BuiltinMethod,
        /// Inclusive start index accepted by the builtin.
        start: i64,
        /// Exclusive end index accepted by the builtin, if present.
        end: Option<i64>,
    },

    /// Applies a pre-compiled sub-pipeline `Plan` to each element, replacing it with the result.
    CompiledMap(Arc<Plan>),

    /// Expression-backed builtin stage identified by the registry.
    ExprBuiltin {
        /// Registry method identity for this expression-backed stage.
        method: BuiltinMethod,
        /// Compiled expression body evaluated for each relevant element/key/value.
        body: Arc<crate::vm::Program>,
    },

    /// Removes consecutive duplicates from a pre-sorted stream, optionally keyed by a program.
    SortedDedup(Option<Arc<crate::vm::Program>>),
}

/// The four numeric fold operations supported by the `Reducer` sink.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumOp {
    /// Adds all numeric elements together.
    Sum,
    /// Selects the smallest numeric element.
    Min,
    /// Selects the largest numeric element.
    Max,
    /// Computes the arithmetic mean of all numeric elements.
    Avg,
}

impl NumOp {
    /// Converts a `BuiltinNumericReducer` tag from the builtin registry into the corresponding
    /// `NumOp` variant.
    pub(crate) fn from_builtin_reducer(reducer: BuiltinNumericReducer) -> Self {
        match reducer {
            BuiltinNumericReducer::Sum => NumOp::Sum,
            BuiltinNumericReducer::Avg => NumOp::Avg,
            BuiltinNumericReducer::Min => NumOp::Min,
            BuiltinNumericReducer::Max => NumOp::Max,
        }
    }

    /// Returns the `BuiltinMethod` that corresponds to this numeric operation.
    pub(crate) fn method(self) -> BuiltinMethod {
        match self {
            NumOp::Sum => BuiltinMethod::Sum,
            NumOp::Min => BuiltinMethod::Min,
            NumOp::Max => BuiltinMethod::Max,
            NumOp::Avg => BuiltinMethod::Avg,
        }
    }

    /// Returns the identity / empty-input value for this operation (`0` for Sum, `null` for others).
    fn empty(self) -> Val {
        match self {
            NumOp::Sum => Val::Int(0),
            NumOp::Avg => Val::Null,
            NumOp::Min => Val::Null,
            NumOp::Max => Val::Null,
        }
    }
}

impl Sink {
    /// Returns the `ReducerSpec` if this sink is a `Reducer`, otherwise `None`.
    pub(crate) fn reducer_spec(&self) -> Option<ReducerSpec> {
        match self {
            Sink::Reducer(spec) => Some(spec.clone()),
            _ => None,
        }
    }
}

/// The terminal accumulator of a pipeline — consumes the element stream and produces the final value.
#[derive(Debug, Clone)]
pub enum Sink {
    /// Gathers all passing elements into a `Val::Arr`.
    Collect,

    /// Folds the stream using the given `ReducerSpec` (count, sum, min, max, avg).
    Reducer(ReducerSpec),
    /// Delegates to a built-in method that consumes the stream (e.g. `first`, `last`).
    Terminal(BuiltinMethod),

    /// Computes an approximate count of distinct values using a probabilistic sketch.
    ApproxCountDistinct,
}
/// The complete pipeline IR: source → stages → sink, with pre-classified kernels for each stage.
#[derive(Debug, Clone)]
pub struct Pipeline {
    /// Where element values originate — a document field path or an already-evaluated receiver.
    pub source: Source,
    /// Ordered list of transformation stages applied to each element in turn.
    pub stages: Vec<Stage>,

    /// Preserved AST expressions parallel to `stages`, used by the demand optimiser to
    /// substitute `@` and simplify predicates without re-parsing.
    pub stage_exprs: Vec<Option<Arc<Expr>>>,
    /// How the stream is consumed and a final value produced.
    pub sink: Sink,

    /// Pre-classified kernels parallel to `stages`; avoids VM re-entry for common patterns.
    pub stage_kernels: Vec<BodyKernel>,

    /// Pre-classified kernels for sink sub-programs (predicate / projection inside a reducer).
    pub sink_kernels: Vec<BodyKernel>,
}

/// The source-independent half of a `Pipeline`; can be combined with any `Source` via
/// `with_source` to produce a runnable `Pipeline`.
#[derive(Debug, Clone)]
pub struct PipelineBody {
    /// Ordered transformation stages.
    pub stages: Vec<Stage>,

    /// Preserved AST expressions parallel to `stages` for symbolic optimisation.
    pub stage_exprs: Vec<Option<Arc<Expr>>>,
    /// Terminal accumulator for the pipeline.
    pub sink: Sink,
    /// Pre-classified kernels parallel to `stages`.
    pub stage_kernels: Vec<BodyKernel>,
    /// Pre-classified kernels for sink sub-programs.
    pub sink_kernels: Vec<BodyKernel>,
}

impl PipelineBody {
    /// Attaches `source` to this body, producing a complete executable `Pipeline`.
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
    /// Splits the pipeline into its `Source` and the source-independent `PipelineBody`,
    /// allowing the body to be reused with a different source.
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
            "count_by".into(),
            vec![Arg::Pos(Expr::Ident("kind".into()))]
        )));

        assert!(!Pipeline::is_receiver_pipeline_start(&Step::Method(
            "from_json".into(),
            Vec::new()
        )));
    }

    #[test]
    fn lower_take_skip_sum() {
        let p = lower_query("$.xs.skip(2).take(5).sum()").unwrap();
        assert_eq!(p.stages.len(), 2);
        assert!(matches!(
            p.stages[0],
            Stage::UsizeBuiltin {
                method: BuiltinMethod::Skip,
                value: 2
            }
        ));
        assert!(matches!(
            p.stages[1],
            Stage::UsizeBuiltin {
                method: BuiltinMethod::Take,
                value: 5
            }
        ));
        assert!(
            matches!(&p.sink, Sink::Reducer(spec) if spec.op == ReducerOp::Numeric(NumOp::Sum))
        );
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
        assert!(lower_query("$.xs.equi_join($.ys, lhs, rhs)").is_none());

        assert!(lower_query("@.x.filter(y > 0)").is_none());
    }

    #[test]
    fn debug_filter_pred_shape() {
        let expr = crate::parser::parse("@.total > 100").unwrap();
        let prog = crate::compiler::Compiler::compile(&expr, "");
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
    fn run_terminal_count_predicate() {
        use serde_json::json;
        let doc: Val = (&json!({"orders":[
            {"total": 50}, {"total": 150}, {"total": 200}
        ]}))
            .into();
        let p = lower_query("$.orders.count(total > 100)").unwrap();
        let out = p.run(&doc).unwrap();
        assert_eq!(out, Val::Int(2));
    }

    #[test]
    fn run_terminal_numeric_projection() {
        use serde_json::json;
        let doc: Val = (&json!({"orders":[
            {"total": 50}, {"total": 150}, {"total": 200}
        ]}))
            .into();
        let p = lower_query("$.orders.sum(total)").unwrap();
        let out = p.run(&doc).unwrap();
        assert_eq!(out, Val::Int(400));
    }

    #[test]
    fn rewrite_skip_skip_merges() {
        let p = lower_query("$.xs.skip(2).skip(3).sum()").unwrap();
        assert_eq!(p.stages.len(), 1);
        assert!(matches!(
            p.stages[0],
            Stage::UsizeBuiltin {
                method: BuiltinMethod::Skip,
                value: 5
            }
        ));
    }

    #[test]
    fn rewrite_take_take_merges_min() {
        let p = lower_query("$.xs.take(10).take(3).sum()").unwrap();
        assert_eq!(p.stages.len(), 1);
        assert!(matches!(
            p.stages[0],
            Stage::UsizeBuiltin {
                method: BuiltinMethod::Take,
                value: 3
            }
        ));
    }

    #[test]
    fn rewrite_filter_filter_merges_via_andop() {
        let p = lower_query("$.orders.filter(total > 100).filter(qty > 0).sum()").unwrap();
        assert_eq!(p.stages.len(), 1);
        match &p.stages[0] {
            Stage::Filter(prog, _) => {
                assert!(prog
                    .ops
                    .iter()
                    .any(|o| matches!(o, crate::vm::Opcode::AndOp(_))));
            }
            _ => panic!("expected merged Filter"),
        }
        assert!(
            matches!(&p.stage_kernels[0], BodyKernel::And(predicates) if predicates.len() == 2)
        );
        assert!(p.stage_kernels[0].is_view_native());
    }

    #[test]
    fn predicate_fusion_collapses_long_filter_runs_after_planning() {
        let p = lower_query(
            "$.orders.filter(total > 100).filter(qty > 0).filter(active == true).filter(region == \"eu\").count()",
        )
        .unwrap();

        assert_eq!(p.stages.len(), 1);
        assert!(matches!(&p.stages[0], Stage::Filter(_, _)));
        assert!(
            matches!(&p.stage_kernels[0], BodyKernel::And(predicates) if predicates.len() == 4)
        );
        assert!(p.stage_kernels[0].is_view_native());
    }

    #[test]
    fn rewrite_map_then_count_drops_map() {
        let p = lower_query("$.orders.map(total).count()").unwrap();
        assert_eq!(p.stages.len(), 0);
        assert!(matches!(p.sink, Sink::Reducer(ref spec) if spec.op == ReducerOp::Count));
    }

    #[test]
    fn demand_optimizer_drops_value_only_work_for_count() {
        let p = lower_query("$.orders.map(total).upper().count()").unwrap();
        assert!(p.stages.is_empty());
        assert!(matches!(p.sink, Sink::Reducer(ref spec) if spec.op == ReducerOp::Count));
    }

    #[test]
    fn terminal_count_predicate_lowers_to_reducer_predicate() {
        let p = lower_query("$.orders.count(total > 10)").unwrap();
        assert!(p.stages.is_empty());
        assert!(
            matches!(&p.sink, Sink::Reducer(spec) if spec.op == ReducerOp::Count && spec.predicate.is_some())
        );
    }

    #[test]
    fn terminal_numeric_projection_lowers_to_reducer_projection() {
        let p = lower_query("$.orders.sum(total)").unwrap();
        assert!(p.stages.is_empty());
        assert!(
            matches!(&p.sink, Sink::Reducer(spec) if spec.op == ReducerOp::Numeric(NumOp::Sum) && spec.projection.is_some())
        );
    }

    #[test]
    fn demand_optimizer_pulls_filter_through_map_for_count() {
        let p = lower_query("$.orders.map(total).filter(@ > 10).count()").unwrap();
        assert_eq!(p.stages.len(), 1);
        assert!(matches!(p.stages[0], Stage::Filter(_, _)));
        assert!(matches!(p.sink, Sink::Reducer(ref spec) if spec.op == ReducerOp::Count));
    }

    #[test]
    fn demand_optimizer_pulls_filter_through_computed_map_for_count() {
        let p = lower_query("$.orders.map(price * qty).filter(@ > 100).count()").unwrap();
        assert_eq!(p.stages.len(), 1);
        assert!(matches!(p.stages[0], Stage::Filter(_, _)));
        assert_price_qty_gt_100(only_stage_expr(&p));
        assert!(matches!(p.sink, Sink::Reducer(ref spec) if spec.op == ReducerOp::Count));
    }

    #[test]
    fn demand_optimizer_pulls_filter_through_method_chain_map_for_count() {
        let p =
            lower_query("$.users.map(name.trim().upper()).filter(@ == \"ADA\").count()").unwrap();
        assert_eq!(p.stages.len(), 1);
        assert!(matches!(p.stages[0], Stage::Filter(_, _)));
        assert!(matches!(p.sink, Sink::Reducer(ref spec) if spec.op == ReducerOp::Count));
    }

    #[test]
    fn demand_optimizer_simplifies_object_projection_after_substitution() {
        let p = lower_query("$.orders.map({v: price * qty, id: id}).filter(@.v > 100).count()")
            .unwrap();
        assert_eq!(p.stages.len(), 1);
        assert!(matches!(p.stages[0], Stage::Filter(_, _)));
        assert_price_qty_gt_100(only_stage_expr(&p));
        assert!(matches!(p.sink, Sink::Reducer(ref spec) if spec.op == ReducerOp::Count));
    }

    #[test]
    fn demand_optimizer_simplifies_array_projection_after_substitution() {
        let p = lower_query("$.orders.map([price * qty, id]).filter(@[0] > 100).count()").unwrap();
        assert_eq!(p.stages.len(), 1);
        assert!(matches!(p.stages[0], Stage::Filter(_, _)));
        assert_price_qty_gt_100(only_stage_expr(&p));
        assert!(matches!(p.sink, Sink::Reducer(ref spec) if spec.op == ReducerOp::Count));
    }

    #[test]
    fn demand_optimizer_pulls_filter_through_map_but_keeps_map_for_collect() {
        let p = lower_query("$.orders.map(total).filter(@ > 10)").unwrap();
        assert_eq!(p.stages.len(), 2);
        assert!(matches!(p.stages[0], Stage::Filter(_, _)));
        assert!(matches!(p.stages[1], Stage::Map(_, _)));
        assert!(matches!(p.sink, Sink::Collect));
    }

    #[test]
    fn demand_optimizer_removes_order_only_work_for_numeric_sink() {
        let p = lower_query("$.orders.sort().reverse().map(total).sum()").unwrap();
        assert!(p.stages.is_empty());
        assert!(
            matches!(&p.sink, Sink::Reducer(spec) if spec.op == ReducerOp::Numeric(NumOp::Sum) && spec.projection.is_some())
        );
    }

    #[test]
    fn demand_optimizer_keeps_membership_work_for_numeric_sink() {
        let p = lower_query("$.orders.take(2).map(total).sum()").unwrap();
        assert_eq!(p.stages.len(), 1);
        assert!(matches!(
            p.stages[0],
            Stage::UsizeBuiltin {
                method: BuiltinMethod::Take,
                value: 2
            }
        ));
        assert!(
            matches!(&p.sink, Sink::Reducer(spec) if spec.op == ReducerOp::Numeric(NumOp::Sum) && spec.projection.is_some())
        );
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

    #[test]
    fn run_topn_smallest_three() {
        use serde_json::json;
        let doc: Val = (&json!({"xs":[5, 2, 8, 1, 4, 7, 3]})).into();
        let p = lower_query("$.xs.sort().take(3)").unwrap();
        let out = p.run(&doc).unwrap();

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
    fn sort_take_while_take_uses_prefix_demand_without_key_correlation() {
        let p = lower_query("$.rows.sort_by(-price).take_while(price > 10).take(2)").unwrap();
        let strategies = compute_strategies_with_kernels(&p.stages, &p.stage_kernels, &p.sink);
        assert!(matches!(strategies[0], StageStrategy::SortTopK(2)));

        let p = lower_query("$.rows.sort_by(-score).take_while(price > 10).take(2)").unwrap();
        let strategies = compute_strategies_with_kernels(&p.stages, &p.stage_kernels, &p.sink);
        assert!(matches!(strategies[0], StageStrategy::SortTopK(2)));

        let p = lower_query("$.rows.sort().take_while(price > 10).take(2)").unwrap();
        let strategies = compute_strategies_with_kernels(&p.stages, &p.stage_kernels, &p.sink);
        assert!(matches!(strategies[0], StageStrategy::SortTopK(2)));

        let p = lower_query("$.rows.sort_by(-score).filter(price > 10).take(2)").unwrap();
        let strategies = compute_strategies_with_kernels(&p.stages, &p.stage_kernels, &p.sink);
        assert!(matches!(strategies[0], StageStrategy::SortUntilOutput(2)));
    }

    #[test]
    fn sort_filter_take_uses_lazy_ordered_until_output_and_matches_vm() {
        use serde_json::json;

        let p = lower_query("$.rows.sort_by(-price).filter(test > 10).take(2)").unwrap();
        let strategies = compute_strategies_with_kernels(&p.stages, &p.stage_kernels, &p.sink);
        assert!(matches!(strategies[0], StageStrategy::SortUntilOutput(2)));

        assert_pipeline_matches_vm_query(
            "$.rows.sort_by(-price).filter(test > 10).take(2)",
            "$.rows.sort(-price).filter(test > 10).first(2)",
            json!({
                "rows": [
                    {"id": 1, "price": 100, "test": 0},
                    {"id": 2, "price": 90, "test": 20},
                    {"id": 3, "price": 80, "test": 0},
                    {"id": 4, "price": 70, "test": 30}
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
        assert_eq!(
            demand.chain.pull,
            crate::chain_ir::PullDemand::FirstInput(2)
        );
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
        let p = lower_query("$.xs.filter(true).count()").unwrap();
        assert_eq!(p.stages.len(), 0);
        assert!(matches!(p.sink, Sink::Reducer(ref spec) if spec.op == ReducerOp::Count));
    }

    #[test]
    fn run_take_skip() {
        use serde_json::json;
        let doc: Val = (&json!({"xs":[10, 20, 30, 40, 50]})).into();
        let p = lower_query("$.xs.skip(1).take(2).sum()").unwrap();
        let out = p.run(&doc).unwrap();
        assert_eq!(out, Val::Int(50));
    }
}
