//! Stable numeric IDs for builtins and per-builtin demand-propagation laws.
//!
//! `BuiltinMethod` (the original enum in `builtins.rs`) remains the execution
//! identity used by the VM and pipeline. `BuiltinId` is a compact numeric
//! alias for the same set, stable across refactors, that new planner and
//! analysis code carries without depending on the legacy enum directly.

use std::sync::Arc;

use crate::{
    builtins::{
        builtin::{BarrierCtx, Builtin, StreamCtx},
        defs, BuiltinArgExtremeSink, BuiltinArgs, BuiltinArraySelector, BuiltinCall,
        BuiltinCancellation, BuiltinCardinality, BuiltinCategory, BuiltinColumnarStage,
        BuiltinDemandLaw, BuiltinExprPayload, BuiltinExprStage, BuiltinKeyedReducer,
        BuiltinLogicalShape, BuiltinMembershipSink, BuiltinMethod, BuiltinNullaryStage,
        BuiltinNumericReducer, BuiltinObjectLambda, BuiltinPipelineArity, BuiltinPipelineLowering,
        BuiltinPipelineMaterialization, BuiltinPipelineOrderEffect, BuiltinPipelineShape,
        BuiltinPredicateSink, BuiltinRawJsonScalar, BuiltinRowStreamOp, BuiltinRuntimeHook,
        BuiltinSelectionPosition, BuiltinSinkAccumulator, BuiltinSinkDemand, BuiltinSinkSpec,
        BuiltinSinkValueNeed, BuiltinStageMerge, BuiltinStreamingBoundary, BuiltinStringPairStage,
        BuiltinStructural, BuiltinViewObjectProjection, BuiltinViewScalarOp, BuiltinViewStage,
        BuiltinViewStringExpand, BuiltinViewValueProjection,
    },
    data::{
        context::EvalError,
        value::Val,
        view::{view_matches_value, write_json_view, ValueView},
    },
    exec::pipeline::StageFlow,
    plan::demand::{
        Demand, FieldDemand, FieldPath, FieldSet, PullDemand, SinkResultDemand, ValueNeed,
    },
    util::JsonView,
    vm::Program,
};

/// Compact, stable numeric identity for a builtin. One-to-one with
/// `BuiltinMethod`; used by planner/analysis to avoid re-matching names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct BuiltinId(pub(crate) u16);

/// Optional numeric argument carried alongside a builtin's demand law.
/// `Take(n)` and `Skip(n)` pass their count here so `propagate_demand` can
/// tighten or loosen the upstream `PullDemand` accordingly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BuiltinDemandArg {
    /// No numeric argument; the law is applied unconditionally.
    None,
    /// A specific count (e.g. the `n` in `.take(n)` or `.skip(n)`).
    Usize(usize),
}

/// Concrete composed-pipeline implementation selected by builtin metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BuiltinComposedStage {
    Compact,
    RemoveValue,
}

/// Direct borrowed-view call family for callers that can bypass full
/// materialization but still need to distinguish scalar results from object
/// item enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BuiltinDirectViewCall {
    ScalarValue,
    ObjectItems(BuiltinViewObjectProjection),
}

/// Return the logical planner shape for builtin `id`, if it has one.
#[inline]
pub(crate) fn logical_shape(id: BuiltinId) -> Option<BuiltinLogicalShape> {
    id.method().and_then(|method| method.spec().logical_shape)
}

/// Return source-level `$.rows()` stream behavior for builtin `id`, if it is
/// legal in row-stream position.
#[inline]
pub(crate) fn row_stream_op(id: BuiltinId) -> Option<BuiltinRowStreamOp> {
    id.method().and_then(|method| method.spec().row_stream_op)
}

/// Return whether builtin `id` lifts the current input into a source-level
/// stream boundary such as `$.rows()`.
#[inline]
pub(crate) fn stream_source(id: BuiltinId) -> bool {
    id.method()
        .is_some_and(|method| method.spec().stream_source)
}

/// Return predicate terminal-sink behavior for builtin `id`, if it has one.
#[inline]
pub(crate) fn predicate_sink(id: BuiltinId) -> Option<BuiltinPredicateSink> {
    Some(id.method()?.spec().predicate_sink?)
}

/// Return the value need for a predicate terminal sink.
#[inline]
pub(crate) fn predicate_sink_value_need(sink: BuiltinPredicateSink) -> ValueNeed {
    sink.value_need()
}

/// Return row-stream demand for a predicate terminal sink.
#[inline]
pub(crate) fn predicate_sink_demand(sink: BuiltinPredicateSink) -> Demand {
    Demand {
        pull: PullDemand::All,
        value: predicate_sink_value_need(sink),
        order: false,
    }
}

/// Return scalar-result short-circuit demand for a predicate terminal sink.
#[inline]
pub(crate) fn predicate_sink_result_demand(sink: BuiltinPredicateSink) -> SinkResultDemand {
    sink.result_demand()
}

/// Return membership terminal-sink behavior for builtin `id`, if it has one.
#[inline]
pub(crate) fn membership_sink(id: BuiltinId) -> Option<BuiltinMembershipSink> {
    Some(id.method()?.spec().membership_sink?)
}

/// Return the value need for a membership terminal sink.
#[inline]
pub(crate) fn membership_sink_value_need(sink: BuiltinMembershipSink) -> ValueNeed {
    sink.value_need()
}

/// Return row-stream demand for a membership terminal sink.
#[inline]
pub(crate) fn membership_sink_demand(sink: BuiltinMembershipSink) -> Demand {
    Demand {
        pull: PullDemand::All,
        value: membership_sink_value_need(sink),
        order: false,
    }
}

/// Return scalar-result short-circuit demand for a membership terminal sink.
#[inline]
pub(crate) fn membership_sink_result_demand(sink: BuiltinMembershipSink) -> SinkResultDemand {
    sink.result_demand()
}

/// Return arg-extreme terminal-sink behavior for builtin `id`, if it has one.
#[inline]
pub(crate) fn arg_extreme_sink(id: BuiltinId) -> Option<BuiltinArgExtremeSink> {
    Some(id.method()?.spec().arg_extreme_sink?)
}

/// Return row-stream demand for an arg-extreme terminal sink.
#[inline]
pub(crate) fn arg_extreme_sink_demand(_sink: BuiltinArgExtremeSink) -> Demand {
    Demand {
        pull: PullDemand::All,
        value: ValueNeed::Whole,
        order: true,
    }
}

/// Return the concrete pipeline stage shape for an expression-argument builtin.
#[inline]
pub(crate) fn expr_stage(id: BuiltinId) -> Option<BuiltinExprStage> {
    let method = id.method()?;
    let spec = method.spec();
    if let Some(stage) = spec.expr_stage {
        return Some(stage);
    }
    matches!(
        spec.lowering,
        Some(BuiltinPipelineLowering::ExprArg)
            | Some(BuiltinPipelineLowering::TerminalExprArg { .. })
    )
    .then_some(BuiltinExprStage::ExprBuiltin)
}

/// Return the concrete pipeline stage shape for a nullary pipeline builtin.
#[inline]
pub(crate) fn nullary_stage(id: BuiltinId) -> Option<BuiltinNullaryStage> {
    let method = id.method()?;
    let spec = method.spec();
    if let Some(stage) = spec.nullary_stage {
        return Some(stage);
    }
    (matches!(spec.lowering, Some(BuiltinPipelineLowering::Nullary)) && spec.is_element)
        .then_some(BuiltinNullaryStage::Element)
}

/// Return concrete behavior for a two-string-argument pipeline builtin.
#[inline]
pub(crate) fn string_pair_stage(id: BuiltinId) -> Option<BuiltinStringPairStage> {
    id.method()
        .and_then(|method| method.spec().string_pair_stage)
}

/// Return payload-demand behavior for expression-bearing builtin stages.
#[inline]
pub(crate) fn expr_payload(id: BuiltinId) -> Option<BuiltinExprPayload> {
    let method = id.method()?;
    let spec = method.spec();
    if let Some(payload) = spec.expr_payload {
        return Some(payload);
    }
    if let Some(reducer) = spec.keyed_reducer {
        return Some(reducer.expr_payload());
    }
    match spec.demand_law {
        BuiltinDemandLaw::KeyOnlyReducer => Some(BuiltinExprPayload::KeyOnlyReducer),
        BuiltinDemandLaw::RowKeyedReducer => Some(BuiltinExprPayload::RowKeyedReducer),
        _ => None,
    }
}

/// Return true when an expression-bearing stage can be elided if downstream
/// demand proves its output value is unused.
#[inline]
pub(crate) fn expr_stage_elidable_when_value_unused(id: BuiltinId) -> bool {
    object_lambda(id).is_some()
        && is_pure(id)
        && builtin_cardinality(id) == Some(BuiltinCardinality::OneToOne)
        && effective_pipeline_order_effect(id, false) == BuiltinPipelineOrderEffect::Preserves
}

/// Return true when a builtin stage can be dropped entirely once downstream
/// demand proves its output value is unused. This rule is deliberately metadata
/// only: the executor may adapt concrete stage variants to builtin ids, but it
/// must not re-derive the semantic conditions locally.
#[inline]
pub(crate) fn stage_elidable_when_value_unused(id: BuiltinId) -> bool {
    is_pure(id)
        && builtin_cardinality(id) == Some(BuiltinCardinality::OneToOne)
        && effective_pipeline_order_effect(id, false) == BuiltinPipelineOrderEffect::Preserves
}

/// Return true when a builtin call can be delayed as a view-native projection.
/// This is stricter than generic stage elision: the stage must be safe to move
/// across bounded demand and have a borrowed-view implementation.
#[inline]
pub(crate) fn stage_delayable_view_projection(id: BuiltinId) -> bool {
    stage_elidable_when_value_unused(id) && view_projection(id)
}

/// Return true when a pipeline stage for `id` must inspect row values to
/// decide its output membership, ordering, key, or projected value.
#[inline]
pub(crate) fn pipeline_stage_consumes_value(id: BuiltinId, has_body: bool) -> bool {
    if has_body {
        return true;
    }
    matches!(
        demand_law(id),
        BuiltinDemandLaw::FilterLike
            | BuiltinDemandLaw::TakeWhile
            | BuiltinDemandLaw::DropWhile
            | BuiltinDemandLaw::UniqueLike
            | BuiltinDemandLaw::MapLike
            | BuiltinDemandLaw::PredicateMapLike
            | BuiltinDemandLaw::Slice
            | BuiltinDemandLaw::FlatMapLike
            | BuiltinDemandLaw::NumericReducer
            | BuiltinDemandLaw::KeyOnlyReducer
            | BuiltinDemandLaw::RowKeyedReducer
            | BuiltinDemandLaw::OrderBarrier
    )
}

/// Return true when the stage enforces an input-position window while preserving
/// the relative order of retained rows.
#[inline]
pub(crate) fn pipeline_stage_is_positional(id: BuiltinId) -> bool {
    matches!(
        demand_law(id),
        BuiltinDemandLaw::Take | BuiltinDemandLaw::Skip
    )
}

/// Return true when the stage caps upstream input to a bounded prefix.
#[inline]
pub(crate) fn pipeline_stage_caps_input_prefix(id: BuiltinId) -> bool {
    matches!(demand_law(id), BuiltinDemandLaw::Take)
}

/// Return true when receiver-mode VM execution can consume a downstream output cap.
#[inline]
pub(crate) fn output_cap_receiver(id: BuiltinId) -> bool {
    id.method()
        .is_some_and(|method| method.spec().output_cap_receiver)
}

/// Return true when a stage's work cannot affect a following terminal sink.
#[inline]
pub(crate) fn stage_elidable_before_sink(stage: BuiltinId, sink: BuiltinId) -> bool {
    let Some(sink) = builtin_sink(sink) else {
        return false;
    };
    let demand = sink_demand(sink);
    (demand.value == ValueNeed::CountOnly
        && builtin_cardinality(stage) == Some(BuiltinCardinality::OneToOne))
        || (matches!(demand.pull, PullDemand::All)
            && !demand.order
            && pipeline_stage_is_order_only(stage))
}

/// Return a cheaper terminal method for `stage.position()` when the stage
/// advertises that selection can be expressed directly.
#[inline]
pub(crate) fn terminal_selection_rewrite(
    stage: BuiltinId,
    position: BuiltinSelectionPosition,
) -> Option<BuiltinMethod> {
    let rewrite = stage.method()?.spec().selection_rewrite?;
    match position {
        BuiltinSelectionPosition::First => rewrite.first,
        BuiltinSelectionPosition::Last => rewrite.last,
    }
}

/// Return a cheaper terminal method for indexing the result of `stage`.
#[inline]
pub(crate) fn index_selection_rewrite(stage: BuiltinId, index: i64) -> Option<BuiltinMethod> {
    let rewrite = stage.method()?.spec().selection_rewrite?;
    match index {
        0 => rewrite.index_zero,
        -1 => rewrite.index_minus_one,
        _ => None,
    }
}

/// Return true when the stage only changes row order, not membership or row values.
#[inline]
pub(crate) fn pipeline_stage_is_order_only(id: BuiltinId) -> bool {
    id.method().is_some_and(|method| method.spec().order_only)
}

/// Return the shared runtime hook implementation target for a builtin, if any.
#[inline]
pub(crate) fn runtime_hook(id: BuiltinId) -> Option<BuiltinRuntimeHook> {
    id.method().and_then(|method| method.spec().runtime_hook)
}

/// Return true when a builtin category is meaningful as a collection/pipeline
/// operator in a trailing method chain.
#[inline]
pub(crate) fn pipeline_chain_operator(id: BuiltinId) -> bool {
    matches!(
        builtin_category(id),
        Some(
            BuiltinCategory::StreamingOneToOne
                | BuiltinCategory::StreamingFilter
                | BuiltinCategory::StreamingExpand
                | BuiltinCategory::Reducer
                | BuiltinCategory::Positional
                | BuiltinCategory::Barrier
                | BuiltinCategory::Deep
                | BuiltinCategory::Relational
        )
    )
}

/// Return object-lambda behavior for builtin `id`, if any.
#[inline]
pub(crate) fn object_lambda(id: BuiltinId) -> Option<BuiltinObjectLambda> {
    id.method().and_then(|method| method.spec().object_lambda)
}

/// Return true when a count-like terminal sink accepts a predicate argument.
#[inline]
pub(crate) fn count_sink_accepts_predicate(id: BuiltinId) -> bool {
    id.method()
        .and_then(|method| method.spec().sink)
        .map(|sink| sink.accepts_predicate)
        .unwrap_or(false)
}

/// Return the scalar builtin that is equivalent to applying a plain terminal
/// sink to one already-selected JSON value. Direct byte/tape planners use this
/// to keep sink-to-projection rewrites registry-owned instead of encoding them
/// per backend.
#[inline]
pub(crate) fn direct_scalar_for_plain_sink(id: BuiltinId) -> Option<BuiltinCall> {
    let sink = id.method()?.spec().sink?;
    match sink.accumulator {
        BuiltinSinkAccumulator::Count => {
            Some(BuiltinCall::new(BuiltinMethod::Len, BuiltinArgs::None))
        }
        _ => None,
    }
}

/// Compute the upstream `Demand` that builtin `id` must place on its source
/// given the `downstream` demand from the next stage and optional numeric `arg`.
#[inline]
pub(crate) fn propagate_demand(id: BuiltinId, arg: BuiltinDemandArg, downstream: Demand) -> Demand {
    match id.method() {
        Some(BuiltinMethod::Unknown) | None => return Demand::RESULT,
        Some(_) => {}
    }
    match demand_law(id) {
        BuiltinDemandLaw::Identity => downstream,
        BuiltinDemandLaw::FilterLike => Demand {
            pull: match downstream.pull {
                PullDemand::All => PullDemand::All,
                PullDemand::LastInput(n) => PullDemand::LastInput(n),
                PullDemand::NthInput(_) => PullDemand::All,
                PullDemand::FirstInput(n) | PullDemand::UntilOutput(n) => {
                    PullDemand::UntilOutput(n)
                }
            },
            value: downstream.value.merge(ValueNeed::Predicate),
            order: downstream.order || pull_is_positional(downstream.pull),
        },
        BuiltinDemandLaw::TakeWhile => Demand {
            pull: match downstream.pull {
                PullDemand::All | PullDemand::LastInput(_) | PullDemand::NthInput(_) => {
                    PullDemand::All
                }
                PullDemand::FirstInput(n) | PullDemand::UntilOutput(n) => PullDemand::FirstInput(n),
            },
            value: downstream.value.merge(ValueNeed::Predicate),
            order: downstream.order,
        },
        BuiltinDemandLaw::DropWhile => Demand {
            pull: PullDemand::All,
            value: downstream.value.merge(ValueNeed::Predicate),
            order: true,
        },
        BuiltinDemandLaw::UniqueLike => Demand {
            pull: match downstream.pull {
                PullDemand::All => PullDemand::All,
                PullDemand::LastInput(_) | PullDemand::NthInput(_) => PullDemand::All,
                PullDemand::FirstInput(n) | PullDemand::UntilOutput(n) => {
                    PullDemand::UntilOutput(n)
                }
            },
            value: downstream.value.merge(ValueNeed::Whole),
            order: downstream.order || pull_is_positional(downstream.pull),
        },
        BuiltinDemandLaw::MapLike => Demand {
            value: if downstream.value.requires_payload() {
                ValueNeed::Whole
            } else {
                downstream.value
            },
            ..downstream
        },
        BuiltinDemandLaw::PredicateMapLike => Demand {
            value: if downstream.value.requires_payload() {
                ValueNeed::Predicate
            } else {
                downstream.value
            },
            ..downstream
        },
        BuiltinDemandLaw::Slice => Demand {
            value: if downstream.value.requires_payload() {
                ValueNeed::Whole
            } else {
                downstream.value
            },
            ..downstream
        },
        BuiltinDemandLaw::FlatMapLike => Demand {
            order: downstream.order || pull_is_positional(downstream.pull),
            ..Demand::all(ValueNeed::Whole)
        },
        BuiltinDemandLaw::Take => match arg {
            BuiltinDemandArg::Usize(n) => Demand {
                pull: downstream.pull.cap_inputs(n),
                ..downstream
            },
            BuiltinDemandArg::None => downstream,
        },
        BuiltinDemandLaw::Skip => match arg {
            BuiltinDemandArg::Usize(n) => Demand {
                pull: match downstream.pull {
                    PullDemand::FirstInput(m) => PullDemand::FirstInput(n.saturating_add(m)),
                    PullDemand::NthInput(i) => PullDemand::NthInput(n.saturating_add(i)),
                    PullDemand::All | PullDemand::UntilOutput(_) | PullDemand::LastInput(_) => {
                        PullDemand::All
                    }
                },
                ..downstream
            },
            BuiltinDemandArg::None => downstream,
        },
        BuiltinDemandLaw::Chunk => match arg {
            BuiltinDemandArg::Usize(n) => {
                let width = n.max(1);
                Demand {
                    pull: match downstream.pull {
                        PullDemand::FirstInput(k) | PullDemand::UntilOutput(k) => {
                            PullDemand::FirstInput(width.saturating_mul(k))
                        }
                        PullDemand::NthInput(i) => {
                            PullDemand::FirstInput(width.saturating_mul(i.saturating_add(1)))
                        }
                        PullDemand::All | PullDemand::LastInput(_) => PullDemand::All,
                    },
                    value: downstream.value.merge(ValueNeed::Whole),
                    order: true,
                }
            }
            BuiltinDemandArg::None => Demand::all(ValueNeed::Whole),
        },
        BuiltinDemandLaw::Window => match arg {
            BuiltinDemandArg::Usize(n) => {
                let width = n.max(1);
                Demand {
                    pull: match downstream.pull {
                        PullDemand::FirstInput(k) | PullDemand::UntilOutput(k) => {
                            PullDemand::FirstInput(width.saturating_add(k.saturating_sub(1)))
                        }
                        PullDemand::NthInput(i) => PullDemand::FirstInput(width.saturating_add(i)),
                        PullDemand::All | PullDemand::LastInput(_) => PullDemand::All,
                    },
                    value: downstream.value.merge(ValueNeed::Whole),
                    order: true,
                }
            }
            BuiltinDemandArg::None => Demand::all(ValueNeed::Whole),
        },
        BuiltinDemandLaw::First => Demand::first(downstream.value),
        BuiltinDemandLaw::Last => Demand {
            pull: PullDemand::LastInput(1),
            value: downstream.value,
            order: true,
        },
        BuiltinDemandLaw::Nth => match arg {
            BuiltinDemandArg::Usize(i) => Demand {
                pull: PullDemand::NthInput(i),
                value: downstream.value,
                order: false,
            },
            BuiltinDemandArg::None => Demand::all(ValueNeed::Whole),
        },
        BuiltinDemandLaw::Count => Demand {
            pull: PullDemand::All,
            value: ValueNeed::CountOnly,
            order: false,
        },
        BuiltinDemandLaw::NumericReducer => Demand {
            pull: PullDemand::All,
            value: ValueNeed::Numeric,
            order: false,
        },
        BuiltinDemandLaw::KeyOnlyReducer => Demand {
            pull: PullDemand::All,
            value: ValueNeed::Predicate,
            order: false,
        },
        BuiltinDemandLaw::RowKeyedReducer => Demand {
            pull: PullDemand::All,
            value: ValueNeed::Whole,
            order: false,
        },
        BuiltinDemandLaw::OrderBarrier => Demand {
            pull: PullDemand::All,
            value: downstream.value.merge(ValueNeed::Whole),
            order: true,
        },
        BuiltinDemandLaw::Reverse => Demand {
            pull: match downstream.pull {
                PullDemand::FirstInput(n) | PullDemand::UntilOutput(n) => PullDemand::LastInput(n),
                PullDemand::LastInput(n) => PullDemand::FirstInput(n),
                PullDemand::NthInput(_) => PullDemand::All,
                PullDemand::All => PullDemand::All,
            },
            value: downstream.value,
            order: downstream.order,
        },
    }
}

#[inline(always)]
fn pull_is_positional(pull: PullDemand) -> bool {
    !matches!(pull, PullDemand::All)
}

/// Convert builtin terminal-sink metadata into the shared planner demand model.
#[inline]
pub(crate) fn sink_demand(spec: BuiltinSinkSpec) -> Demand {
    match spec.demand {
        BuiltinSinkDemand::First { value } => Demand::first(sink_value_need(value)),
        BuiltinSinkDemand::Last { value } => Demand {
            pull: PullDemand::LastInput(1),
            value: sink_value_need(value),
            order: true,
        },
        BuiltinSinkDemand::All { value, order } => Demand {
            pull: PullDemand::All,
            value: sink_value_need(value),
            order,
        },
    }
}

#[inline]
fn sink_value_need(value: BuiltinSinkValueNeed) -> ValueNeed {
    match value {
        BuiltinSinkValueNeed::None => ValueNeed::CountOnly,
        BuiltinSinkValueNeed::Whole => ValueNeed::Whole,
        BuiltinSinkValueNeed::Numeric => ValueNeed::Numeric,
    }
}

/// Return `true` if builtin `id` has a non-trivial demand law that can
/// either restrict the amount of input the planner must pull from its source
/// or conservatively widen unsafe downstream pull precision to a full scan.
#[inline]
pub(crate) fn participates_in_demand(id: BuiltinId) -> bool {
    demand_law(id) != BuiltinDemandLaw::Identity || demand_is_conservative_barrier(id)
}

/// Return true when builtin `id` cannot safely preserve downstream pull
/// precision and must be treated as a full-input demand boundary.
#[inline]
pub(crate) fn demand_is_conservative_barrier(id: BuiltinId) -> bool {
    matches!(
        demand_law(id),
        BuiltinDemandLaw::FlatMapLike
            | BuiltinDemandLaw::DropWhile
            | BuiltinDemandLaw::OrderBarrier
    ) || matches!(id.method(), Some(BuiltinMethod::Unknown) | None)
}

/// Return the materialization policy for builtin `id`; defaults to `Streaming`
/// when the builtin has no explicit registry entry.
#[inline]
pub(crate) fn pipeline_materialization(id: BuiltinId) -> BuiltinPipelineMaterialization {
    id.method()
        .map(|m| m.spec().materialization)
        .unwrap_or(BuiltinPipelineMaterialization::Streaming)
}

/// Return the semantic streaming-boundary class for builtin `id`.
/// Unknown builtins are treated as legacy materialized boundaries so callers
/// never accidentally keep unsupported behavior in the borrowed tape domain.
#[inline]
pub(crate) fn streaming_boundary(id: BuiltinId) -> BuiltinStreamingBoundary {
    let Some(method) = id.method() else {
        return BuiltinStreamingBoundary::LegacyMaterialized;
    };
    let spec = method.spec();
    if spec.stream_source {
        return BuiltinStreamingBoundary::SourceStream;
    }
    if !matches!(spec.streaming_boundary, BuiltinStreamingBoundary::RowLocal) {
        return spec.streaming_boundary;
    }
    match spec.materialization {
        BuiltinPipelineMaterialization::Streaming => BuiltinStreamingBoundary::RowLocal,
        BuiltinPipelineMaterialization::ComposedBarrier => match spec.demand_law {
            BuiltinDemandLaw::OrderBarrier | BuiltinDemandLaw::Reverse => {
                BuiltinStreamingBoundary::FullInputOrder
            }
            _ => BuiltinStreamingBoundary::FullInputState,
        },
    }
}

/// Return true when builtin `id` streams row-by-row without buffering.
#[inline]
pub(crate) fn pipeline_streams(id: BuiltinId) -> bool {
    matches!(
        pipeline_materialization(id),
        BuiltinPipelineMaterialization::Streaming
    )
}

/// Return true when builtin `id` buffers through the composed barrier path.
#[inline]
pub(crate) fn pipeline_composed_barrier(id: BuiltinId) -> bool {
    matches!(
        pipeline_materialization(id),
        BuiltinPipelineMaterialization::ComposedBarrier
    )
}

/// Return true when builtin `id` requires the legacy materialized executor.
#[inline]
pub(crate) fn pipeline_legacy_materialized(id: BuiltinId) -> bool {
    matches!(
        streaming_boundary(id),
        BuiltinStreamingBoundary::LegacyMaterialized
    )
}

/// Return the cardinality/cost shape annotation for builtin `id`, used by
/// the pipeline cost estimator during plan selection.
#[inline]
pub(crate) fn pipeline_shape(id: BuiltinId) -> Option<BuiltinPipelineShape> {
    id.method().map(|m| m.spec().pipeline_shape).flatten()
}

/// Return the effective pipeline shape for builtin `id`, using explicit shape
/// metadata when present and otherwise deriving the conservative default from
/// the builtin spec.
#[inline]
pub(crate) fn effective_pipeline_shape(id: BuiltinId) -> Option<BuiltinPipelineShape> {
    let method = id.method()?;
    if let Some(shape) = pipeline_shape(id) {
        return Some(shape);
    }
    let spec = method.spec();
    Some(BuiltinPipelineShape {
        cardinality: spec.cardinality,
        can_indexed: spec.can_indexed,
        cost: spec.cost,
        selectivity: if matches!(spec.category, BuiltinCategory::StreamingFilter) {
            0.5
        } else {
            1.0
        },
    })
}

/// Return the relative builtin execution cost used by heuristic analysis.
#[cfg(test)]
#[inline]
pub(crate) fn heuristic_cost(id: BuiltinId) -> u32 {
    effective_pipeline_shape(id)
        .map(|shape| shape.cost.round().max(1.0) as u32)
        .unwrap_or(8)
}

/// Return the columnar-stage metadata for builtin `id`, if it has one.
#[inline]
pub(crate) fn columnar_stage(id: BuiltinId) -> Option<BuiltinColumnarStage> {
    id.method().and_then(|method| method.spec().columnar_stage)
}

/// Return the stage-merge metadata for builtin `id`, if adjacent stages of the
/// same kind can be merged.
#[inline]
pub(crate) fn stage_merge(id: BuiltinId) -> Option<BuiltinStageMerge> {
    id.method().and_then(|method| method.spec().stage_merge)
}

/// Return the builtin sink metadata for `id`, if the builtin is a terminal sink.
#[inline]
pub(crate) fn builtin_sink(id: BuiltinId) -> Option<BuiltinSinkSpec> {
    id.method().and_then(|method| method.spec().sink)
}

/// Return the sink accumulator behavior for builtin `id`, if the builtin is a terminal sink.
#[inline]
pub(crate) fn sink_accumulator(id: BuiltinId) -> Option<BuiltinSinkAccumulator> {
    Some(builtin_sink(id)?.accumulator)
}

/// Return keyed reducer metadata for builtin `id`, if the builtin groups rows by a key.
#[inline]
pub(crate) fn keyed_reducer(id: BuiltinId) -> Option<BuiltinKeyedReducer> {
    id.method().and_then(|method| method.spec().keyed_reducer)
}

/// Return numeric reducer metadata for builtin `id`, if the builtin reduces rows numerically.
#[inline]
pub(crate) fn numeric_reducer(id: BuiltinId) -> Option<BuiltinNumericReducer> {
    id.method().and_then(|method| method.spec().numeric_reducer)
}

/// Return whether builtin `id` is pure and can participate in pure-stage rewrites.
#[inline]
pub(crate) fn is_pure(id: BuiltinId) -> bool {
    id.method().is_some_and(|method| method.spec().pure)
}

/// Return algebraic cancellation metadata for builtin `id`, if it has one.
#[inline]
pub(crate) fn cancellation(id: BuiltinId) -> Option<BuiltinCancellation> {
    id.method().and_then(|method| method.spec().cancellation)
}

/// Return whether builtin `id` is algebraically idempotent.
#[inline]
pub(crate) fn is_idempotent(id: BuiltinId) -> bool {
    id.method().is_some_and(|method| method.spec().idempotent)
}

/// Return whether builtin `id` accepts a lambda/expression argument at runtime.
#[inline]
pub(crate) fn accepts_lambda_arg(id: BuiltinId) -> bool {
    let Some(method) = id.method() else {
        return false;
    };
    let spec = method.spec();
    spec.expr_stage.is_some()
        || spec.accepts_lambda_arg
        || spec.object_lambda.is_some()
        || spec.keyed_reducer.is_some()
        || spec.arg_extreme_sink.is_some()
        || spec.predicate_sink.is_some()
        || matches!(
            spec.lowering,
            Some(
                BuiltinPipelineLowering::ExprArg
                    | BuiltinPipelineLowering::TerminalExprArg { .. }
                    | BuiltinPipelineLowering::Sort
            )
        )
        || spec.sink.is_some_and(|sink| sink.accepts_predicate)
}

/// Return whether builtin `id` should bypass streaming and run as a direct
/// scalar/object call on the receiver produced by the chain.
#[inline]
pub(crate) fn dispatches_scalar_direct(id: BuiltinId) -> bool {
    id.method()
        .is_some_and(|method| method.spec().dispatches_scalar_direct())
}

/// Return the builtin category for planner classification.
#[inline]
pub(crate) fn builtin_category(id: BuiltinId) -> Option<BuiltinCategory> {
    id.method().map(|m| m.spec().category)
}

/// Return the builtin cardinality for planner classification.
#[inline]
pub(crate) fn builtin_cardinality(id: BuiltinId) -> Option<BuiltinCardinality> {
    id.method().map(|m| m.spec().cardinality)
}

/// Return how builtin `id` affects element ordering in the pipeline, or
/// `None` if the builtin has no registered ordering behaviour.
#[inline]
pub(crate) fn pipeline_order_effect(id: BuiltinId) -> Option<BuiltinPipelineOrderEffect> {
    id.method().map(|m| m.spec().order_effect).flatten()
}

/// Return the view-stage lowering tag for builtin `id`, if the builtin can be
/// represented as a borrowed `ValueView` pipeline stage.
#[inline]
pub(crate) fn view_stage(id: BuiltinId) -> Option<BuiltinViewStage> {
    id.method().and_then(|method| method.spec().view_stage)
}

/// Return the view-native object/path operation for builtin `id`, if any.
#[inline]
pub(crate) fn view_object_projection(id: BuiltinId) -> Option<BuiltinViewObjectProjection> {
    id.method()
        .and_then(|method| method.spec().view_object_projection)
}

/// Return the view-native string expansion operation for builtin `id`, if any.
#[inline]
pub(crate) fn view_string_expand(id: BuiltinId) -> Option<BuiltinViewStringExpand> {
    id.method()
        .and_then(|method| method.spec().view_string_expand)
}

/// Return concrete whole-value borrowed-view projection metadata for builtin `id`.
#[inline]
pub(crate) fn view_value_projection(id: BuiltinId) -> Option<BuiltinViewValueProjection> {
    id.method()
        .and_then(|method| method.spec().view_value_projection)
}

/// Return true when builtin `id` enumerates object keys, values, or entries in
/// a view-native path.
#[inline]
pub(crate) fn view_object_items_projection(id: BuiltinId) -> bool {
    view_object_projection(id).is_some_and(BuiltinViewObjectProjection::is_item_projection)
}

/// Return the object item projection for no-argument builtin calls such as
/// `keys()`, `values()`, and `entries()`.
#[inline]
pub(crate) fn view_object_items_projection_call(
    id: BuiltinId,
    args: &BuiltinArgs,
) -> Option<BuiltinViewObjectProjection> {
    if !matches!(args, BuiltinArgs::None) {
        return None;
    }
    let projection = view_object_projection(id)?;
    projection.is_item_projection().then_some(projection)
}

/// Return true when applying this builtin to an object can be answered from
/// the object's key set alone, without reading or preserving field values.
#[inline]
pub(crate) fn view_projection_needs_only_object_keys(id: BuiltinId, args: &BuiltinArgs) -> bool {
    match (view_scalar_op(id), view_object_projection(id), args) {
        (Some(BuiltinViewScalarOp::Len), _, BuiltinArgs::None) => true,
        (_, Some(BuiltinViewObjectProjection::Keys), BuiltinArgs::None) => true,
        (_, Some(BuiltinViewObjectProjection::HasKey), BuiltinArgs::Str(_)) => true,
        (_, Some(BuiltinViewObjectProjection::Has), BuiltinArgs::Str(_)) => true,
        (_, Some(BuiltinViewObjectProjection::HasAll), BuiltinArgs::StrVec(_)) => true,
        _ => false,
    }
}

/// Return the direct borrowed-view call family for a concrete builtin call.
#[inline]
pub(crate) fn direct_view_call(id: BuiltinId, args: &BuiltinArgs) -> Option<BuiltinDirectViewCall> {
    if view_scalar_value_projection(id) {
        return Some(BuiltinDirectViewCall::ScalarValue);
    }
    view_object_items_projection_call(id, args).map(BuiltinDirectViewCall::ObjectItems)
}

/// Return receiver-local field demand for a view-native object/path builtin
/// call.
#[inline]
pub(crate) fn view_projection_field_demand(
    id: BuiltinId,
    args: &BuiltinArgs,
) -> Option<FieldDemand> {
    match (view_object_projection(id)?, args) {
        (BuiltinViewObjectProjection::HasKey, BuiltinArgs::Str(key)) => {
            Some(FieldDemand::Fields(FieldSet::single(Arc::clone(key))))
        }
        (BuiltinViewObjectProjection::Missing, BuiltinArgs::Str(path)) => {
            path_field_demand(&super::parse_path_segs(path.as_ref()))
        }
        (BuiltinViewObjectProjection::Missing, BuiltinArgs::StrVec(keys)) => {
            field_demand_for_paths(keys)
        }
        (BuiltinViewObjectProjection::Pick, BuiltinArgs::StrVec(keys)) => {
            Some(field_demand_for_keys(keys))
        }
        (
            BuiltinViewObjectProjection::GetPath | BuiltinViewObjectProjection::HasPath,
            BuiltinArgs::Str(path),
        ) => path_field_demand(&super::parse_path_segs(path.as_ref())),
        (
            BuiltinViewObjectProjection::GetPath | BuiltinViewObjectProjection::HasPath,
            BuiltinArgs::Path(path),
        ) => path_field_demand(path),
        _ => None,
    }
}

/// Return field demand for applying a view-native object/path builtin call to
/// a receiver path relative to the current row.
#[inline]
pub(crate) fn view_projection_receiver_field_demand(
    id: BuiltinId,
    args: &BuiltinArgs,
    receiver_path: &[Arc<str>],
) -> Option<FieldDemand> {
    view_projection_field_demand(id, args).map(|demand| prefix_field_demand(receiver_path, demand))
}

/// Return the field-only result path produced by applying a view-native
/// object/path projection to `receiver_path`.
///
/// This is intentionally metadata-only: executors can use it to carry direct
/// path facts through tape, byte, or owned-value lowering without re-matching
/// builtin names.
#[inline]
pub(crate) fn view_projection_result_field_path(
    id: BuiltinId,
    args: &BuiltinArgs,
    receiver_path: &[Arc<str>],
) -> Option<Vec<Arc<str>>> {
    match view_object_projection(id)? {
        BuiltinViewObjectProjection::GetPath => {
            let mut keys = receiver_path.to_vec();
            keys.extend(field_only_path_arg_keys(args)?);
            Some(keys)
        }
        _ => None,
    }
}

fn prefix_field_demand(prefix: &[Arc<str>], demand: FieldDemand) -> FieldDemand {
    if prefix.is_empty() {
        return demand;
    }
    match demand {
        FieldDemand::Fields(fields) => FieldDemand::Fields(fields.prefixed(prefix)),
        other => other,
    }
}

fn field_demand_for_keys(keys: &[Arc<str>]) -> FieldDemand {
    let mut fields = FieldSet::new();
    for key in keys {
        fields.insert(FieldPath::single(Arc::clone(key)));
    }
    FieldDemand::Fields(fields)
}

fn field_demand_for_paths(paths: &[Arc<str>]) -> Option<FieldDemand> {
    let mut demand = FieldDemand::None;
    for path in paths {
        let path_demand =
            path_field_demand(&super::parse_path_segs(path.as_ref())).unwrap_or(FieldDemand::Whole);
        demand = demand.merge(path_demand);
    }
    Some(demand)
}

fn field_only_path_arg_keys(args: &BuiltinArgs) -> Option<Vec<Arc<str>>> {
    let path = match args {
        BuiltinArgs::Str(path) => super::parse_path_segs(path.as_ref()),
        BuiltinArgs::Path(path) => path.to_vec(),
        _ => return None,
    };
    path_field_keys(&path)
}

fn path_field_keys(path: &[crate::builtins::PathSeg]) -> Option<Vec<Arc<str>>> {
    let mut keys = Vec::with_capacity(path.len());
    for segment in path {
        match segment {
            crate::builtins::PathSeg::Field(key) => keys.push(Arc::from(key.as_str())),
            crate::builtins::PathSeg::Index(_) => return None,
        }
    }
    Some(keys)
}

fn path_field_demand(path: &[crate::builtins::PathSeg]) -> Option<FieldDemand> {
    let mut keys: Vec<Arc<str>> = Vec::new();
    for segment in path {
        match segment {
            crate::builtins::PathSeg::Field(key) => keys.push(Arc::from(key.as_str())),
            crate::builtins::PathSeg::Index(_) => break,
        }
    }
    match keys.len() {
        0 => None,
        1 => Some(FieldDemand::Fields(FieldSet::single(keys.remove(0)))),
        _ => Some(FieldDemand::Fields(FieldSet::chain(keys.into()))),
    }
}

/// Return positional array selector behavior for builtin `id`, if any.
#[inline]
pub(crate) fn array_selector(id: BuiltinId) -> Option<BuiltinArraySelector> {
    Some(id.method()?.spec().array_selector?)
}

/// Return terminal select-one position for sinks such as `first` and `last`.
#[inline]
pub(crate) fn terminal_selection_position(id: BuiltinId) -> Option<BuiltinSelectionPosition> {
    match sink_accumulator(id)? {
        BuiltinSinkAccumulator::SelectOne(position) => Some(position),
        _ => None,
    }
}

/// Return true when builtin `id` can be composed into a view-native projection
/// kernel without materialising the receiver row.
#[inline]
pub(crate) fn view_projection(id: BuiltinId) -> bool {
    view_scalar_projection(id)
        || view_value_projection(id).is_some()
        || view_object_projection(id).is_some()
}

/// Return true when applying builtin `id` with `args` to a borrowed view
/// necessarily produces an owned `Val` rather than another borrowed child view.
#[inline]
pub(crate) fn view_projection_returns_owned(id: BuiltinId, _args: &BuiltinArgs) -> bool {
    if view_scalar_projection(id) {
        return true;
    }
    if view_value_projection(id).is_some() {
        return true;
    }
    view_object_projection(id).is_some_and(BuiltinViewObjectProjection::returns_owned)
}

/// Result of a view-native builtin application. Some object/path operations
/// return a borrowed child view, while scalar/object enumeration operations
/// produce an owned `Val`.
pub(crate) enum ViewProjectionResult<V> {
    /// The builtin selected a borrowed child view.
    View(V),
    /// The builtin produced an owned result.
    Owned(Val),
}

/// Apply a builtin registered as a view projection to a borrowed `ValueView`.
pub(crate) fn apply_view_projection<'a, V>(
    id: BuiltinId,
    args: &BuiltinArgs,
    view: V,
) -> Option<ViewProjectionResult<V>>
where
    V: ValueView<'a> + 'a,
{
    if let Some(method) = id.method() {
        if view_scalar_projection(id) {
            if let Some(value) = apply_json_view_scalar_hook(method, args, view.scalar()) {
                return Some(ViewProjectionResult::Owned(value));
            }
            if view_value_projection(id).is_none() && view_object_projection(id).is_none() {
                return Some(ViewProjectionResult::View(view));
            }
        }
    }
    if let Some(projection) = view_value_projection(id) {
        if let Some(result) = apply_view_value_projection(projection, args, view.clone()) {
            return Some(result);
        }
    }
    match (view_object_projection(id), args) {
        (Some(BuiltinViewObjectProjection::Has), BuiltinArgs::Str(key)) => {
            view_has(&view, key.as_ref()).map(|found| ViewProjectionResult::Owned(Val::Bool(found)))
        }
        (Some(BuiltinViewObjectProjection::HasAll), BuiltinArgs::StrVec(keys)) => {
            view_has_all(&view, keys).map(|found| ViewProjectionResult::Owned(Val::Bool(found)))
        }
        (Some(BuiltinViewObjectProjection::HasKey), BuiltinArgs::Str(key)) => Some(
            ViewProjectionResult::Owned(Val::Bool(view.has_key(key.as_ref()).unwrap_or(false))),
        ),
        (Some(BuiltinViewObjectProjection::Missing), BuiltinArgs::Str(key)) => {
            let missing = !view_path_exists(&view, &super::parse_path_segs(key.as_ref()));
            Some(ViewProjectionResult::Owned(Val::Bool(missing)))
        }
        (Some(BuiltinViewObjectProjection::Missing), BuiltinArgs::StrVec(keys)) => {
            let missing = keys
                .iter()
                .filter(|key| !view_path_exists(&view, &super::parse_path_segs(key.as_ref())))
                .map(|key| Val::Str(std::sync::Arc::clone(key)))
                .collect();
            Some(ViewProjectionResult::Owned(Val::arr(missing)))
        }
        (Some(BuiltinViewObjectProjection::GetPath), BuiltinArgs::Str(path)) => {
            let path = super::parse_path_segs(path.as_ref());
            Some(ViewProjectionResult::View(walk_view_path(view, &path)))
        }
        (Some(BuiltinViewObjectProjection::GetPath), BuiltinArgs::Path(path)) => {
            Some(ViewProjectionResult::View(walk_view_path(view, path)))
        }
        (Some(BuiltinViewObjectProjection::HasPath), BuiltinArgs::Str(path)) => {
            Some(ViewProjectionResult::Owned(Val::Bool(view_path_exists(
                &view,
                &super::parse_path_segs(path.as_ref()),
            ))))
        }
        (Some(BuiltinViewObjectProjection::HasPath), BuiltinArgs::Path(path)) => Some(
            ViewProjectionResult::Owned(Val::Bool(view_path_exists(&view, path))),
        ),
        (Some(BuiltinViewObjectProjection::Keys), BuiltinArgs::None) => {
            view.object_keys().map(ViewProjectionResult::Owned)
        }
        (Some(BuiltinViewObjectProjection::Values), BuiltinArgs::None) => {
            view.object_values().map(ViewProjectionResult::Owned)
        }
        (Some(BuiltinViewObjectProjection::Entries), BuiltinArgs::None) => {
            view.object_entries().map(ViewProjectionResult::Owned)
        }
        (Some(BuiltinViewObjectProjection::ToPairs), BuiltinArgs::None) => {
            view.object_pairs().map(ViewProjectionResult::Owned)
        }
        (Some(BuiltinViewObjectProjection::Pick), BuiltinArgs::StrVec(keys)) => {
            view.pick_keys(keys).map(ViewProjectionResult::Owned)
        }
        (Some(BuiltinViewObjectProjection::Omit), BuiltinArgs::StrVec(keys)) => {
            view.omit_keys(keys).map(ViewProjectionResult::Owned)
        }
        _ => apply_json_view_scalar_hook(id.method()?, args, view.scalar())
            .map(ViewProjectionResult::Owned),
    }
}

fn apply_view_value_projection<'a, V>(
    projection: BuiltinViewValueProjection,
    args: &BuiltinArgs,
    view: V,
) -> Option<ViewProjectionResult<V>>
where
    V: ValueView<'a> + 'a,
{
    let mut out = String::new();
    match projection {
        BuiltinViewValueProjection::CamelCase
        | BuiltinViewValueProjection::Capitalize
        | BuiltinViewValueProjection::Dedent
        | BuiltinViewValueProjection::FromBase64
        | BuiltinViewValueProjection::HtmlEscape
        | BuiltinViewValueProjection::HtmlUnescape
        | BuiltinViewValueProjection::KebabCase
        | BuiltinViewValueProjection::PascalCase
        | BuiltinViewValueProjection::SnakeCase
        | BuiltinViewValueProjection::TitleCase
        | BuiltinViewValueProjection::ToBase64
        | BuiltinViewValueProjection::UrlDecode
        | BuiltinViewValueProjection::UrlEncode => {
            if !matches!(args, BuiltinArgs::None) {
                return None;
            }
            return Some(match view.scalar() {
                JsonView::Str(value) => {
                    ViewProjectionResult::Owned(apply_string_codec_projection(projection, value))
                }
                _ => ViewProjectionResult::View(view),
            });
        }
        BuiltinViewValueProjection::Center
        | BuiltinViewValueProjection::PadLeft
        | BuiltinViewValueProjection::PadRight => {
            let BuiltinArgs::Pad { width, fill } = args else {
                return None;
            };
            return Some(match view.scalar() {
                JsonView::Str(value) => {
                    ViewProjectionResult::Owned(pad_view_string(value, *width, *fill, projection))
                }
                _ => ViewProjectionResult::View(view),
            });
        }
        BuiltinViewValueProjection::Replace | BuiltinViewValueProjection::ReplaceAll => {
            let BuiltinArgs::StrPair { first, second } = args else {
                return None;
            };
            return Some(match view.scalar() {
                JsonView::Str(value) => ViewProjectionResult::Owned(replace_view_string(
                    value,
                    first,
                    second,
                    matches!(projection, BuiltinViewValueProjection::ReplaceAll),
                )),
                _ => ViewProjectionResult::View(view),
            });
        }
        BuiltinViewValueProjection::Repeat => {
            let BuiltinArgs::Usize(n) = args else {
                return None;
            };
            return Some(match view.scalar() {
                JsonView::Str(value) => {
                    ViewProjectionResult::Owned(Val::Str(Arc::from(value.repeat(*n))))
                }
                _ => ViewProjectionResult::View(view),
            });
        }
        BuiltinViewValueProjection::ReverseStr => {
            if !matches!(args, BuiltinArgs::None) {
                return None;
            }
            return Some(match view.scalar() {
                JsonView::Str(value) => {
                    ViewProjectionResult::Owned(Val::Str(Arc::from(reverse_view_string(value))))
                }
                _ => ViewProjectionResult::View(view),
            });
        }
        BuiltinViewValueProjection::Slice => {
            let BuiltinArgs::I64Opt { first, second } = args else {
                return None;
            };
            return Some(match view.scalar() {
                JsonView::Str(value) => {
                    ViewProjectionResult::Owned(slice_view_string(value, *first, *second))
                }
                _ => ViewProjectionResult::View(view),
            });
        }
        BuiltinViewValueProjection::StripPrefix | BuiltinViewValueProjection::StripSuffix => {
            let BuiltinArgs::Str(affix) = args else {
                return None;
            };
            return Some(match view.scalar() {
                JsonView::Str(value) => ViewProjectionResult::Owned(strip_view_string_affix(
                    value,
                    affix,
                    matches!(projection, BuiltinViewValueProjection::StripPrefix),
                )),
                _ => ViewProjectionResult::View(view),
            });
        }
        BuiltinViewValueProjection::Indent => {
            let prefix = match args {
                BuiltinArgs::Usize(n) => Arc::<str>::from(" ".repeat(*n)),
                BuiltinArgs::Str(prefix) => prefix.clone(),
                _ => return None,
            };
            return Some(match view.scalar() {
                JsonView::Str(value) => ViewProjectionResult::Owned(Val::Str(Arc::from(
                    crate::builtins::indent_str(value, prefix.as_ref()),
                ))),
                _ => ViewProjectionResult::View(view),
            });
        }
        BuiltinViewValueProjection::Or => {
            let BuiltinArgs::Val(default) = args else {
                return None;
            };
            return Some(match view.scalar() {
                JsonView::Null => ViewProjectionResult::Owned(default.clone()),
                _ => ViewProjectionResult::View(view),
            });
        }
        BuiltinViewValueProjection::Includes => {
            let BuiltinArgs::Val(target) = args else {
                return None;
            };
            return Some(ViewProjectionResult::Owned(Val::Bool(view_includes_value(
                &view, target,
            ))));
        }
        BuiltinViewValueProjection::FromPairs => {
            if !matches!(args, BuiltinArgs::None) {
                return None;
            }
            return view_from_pairs(&view).map(ViewProjectionResult::Owned);
        }
        BuiltinViewValueProjection::Invert => {
            if !matches!(args, BuiltinArgs::None) {
                return None;
            }
            return view_invert(&view).map(ViewProjectionResult::Owned);
        }
        BuiltinViewValueProjection::ToString => {
            if !matches!(args, BuiltinArgs::None) {
                return None;
            }
            write_view_string_value(&view, &mut out)?;
        }
        BuiltinViewValueProjection::ToJson => {
            if !matches!(args, BuiltinArgs::None) {
                return None;
            }
            write_json_view(&view, &mut out)?;
        }
    }
    Some(ViewProjectionResult::Owned(Val::Str(Arc::from(out))))
}

fn view_from_pairs<'a, V>(view: &V) -> Option<Val>
where
    V: ValueView<'a> + 'a,
{
    let len = view.array_len()?;
    let mut out = indexmap::IndexMap::with_capacity(len);
    for item in view.array_iter()? {
        if item.array_len() == Some(2) {
            if let JsonView::Str(key) = item.index(0).scalar() {
                out.insert(Arc::from(key), view_to_owned(item.index(1)));
            }
            continue;
        }

        let key_view = if item.has_key("key").unwrap_or(false) {
            item.field("key")
        } else {
            item.field("k")
        };
        let JsonView::Str(key) = key_view.scalar() else {
            continue;
        };
        let value = if item.has_key("val").unwrap_or(false) {
            view_to_owned(item.field("val"))
        } else if item.has_key("value").unwrap_or(false) {
            view_to_owned(item.field("value"))
        } else if item.has_key("v").unwrap_or(false) {
            view_to_owned(item.field("v"))
        } else {
            Val::Null
        };
        out.insert(Arc::from(key), value);
    }
    Some(Val::obj(out))
}

fn view_invert<'a, V>(view: &V) -> Option<Val>
where
    V: ValueView<'a> + 'a,
{
    let len = view.object_len()?;
    let mut out = indexmap::IndexMap::with_capacity(len);
    for (key, value) in view.object_iter()? {
        out.insert(view_key_from_value(&value), Val::Str(key));
    }
    Some(Val::obj(out))
}

fn view_key_from_value<'a, V>(view: &V) -> Arc<str>
where
    V: ValueView<'a> + 'a,
{
    match view.scalar() {
        JsonView::Str(value) => Arc::from(value),
        JsonView::Int(value) => Arc::from(value.to_string()),
        JsonView::UInt(value) => Arc::from(value.to_string()),
        JsonView::Float(value) => Arc::from(value.to_string()),
        JsonView::Bool(value) => Arc::from(value.to_string()),
        JsonView::Null => Arc::from("null"),
        JsonView::ArrayLen(_) | JsonView::ObjectLen(_) => {
            let mut out = String::new();
            if write_json_view(view, &mut out).is_some() {
                Arc::from(out)
            } else {
                Arc::from("")
            }
        }
    }
}

fn view_to_owned<'a, V>(view: V) -> Val
where
    V: ValueView<'a> + 'a,
{
    crate::data::view::scalar_view_to_owned_val(view.scalar()).unwrap_or_else(|| view.materialize())
}

fn view_includes_value<'a, V>(view: &V, target: &Val) -> bool
where
    V: ValueView<'a> + 'a,
{
    match view.scalar() {
        JsonView::ArrayLen(_) => view
            .array_iter()
            .map(|iter| {
                iter.into_iter()
                    .any(|item| view_matches_value(&item, target))
            })
            .unwrap_or(false),
        JsonView::ObjectLen(_) => target
            .as_str()
            .and_then(|key| view.has_key(key))
            .unwrap_or(false),
        JsonView::Str(value) => target
            .as_str()
            .map(|needle| value.contains(needle))
            .unwrap_or(false),
        _ => false,
    }
}

fn reverse_view_string(value: &str) -> String {
    value.chars().rev().collect()
}

fn apply_string_codec_projection(projection: BuiltinViewValueProjection, value: &str) -> Val {
    match projection {
        BuiltinViewValueProjection::CamelCase => {
            Val::Str(Arc::from(crate::builtins::camel_case_str(value)))
        }
        BuiltinViewValueProjection::Capitalize => {
            Val::Str(Arc::from(crate::builtins::capitalize_str(value)))
        }
        BuiltinViewValueProjection::Dedent => {
            Val::Str(Arc::from(crate::builtins::dedent_str(value)))
        }
        BuiltinViewValueProjection::FromBase64 => match crate::builtins::from_base64_str(value) {
            Ok(decoded) => Val::Str(Arc::from(decoded)),
            Err(()) => Val::Null,
        },
        BuiltinViewValueProjection::HtmlEscape => {
            Val::Str(Arc::from(crate::builtins::html_escape_str(value)))
        }
        BuiltinViewValueProjection::HtmlUnescape => {
            Val::Str(Arc::from(crate::builtins::html_unescape_str(value)))
        }
        BuiltinViewValueProjection::KebabCase => {
            Val::Str(Arc::from(crate::builtins::kebab_case_str(value)))
        }
        BuiltinViewValueProjection::PascalCase => {
            Val::Str(Arc::from(crate::builtins::pascal_case_str(value)))
        }
        BuiltinViewValueProjection::SnakeCase => {
            Val::Str(Arc::from(crate::builtins::snake_case_str(value)))
        }
        BuiltinViewValueProjection::TitleCase => {
            Val::Str(Arc::from(crate::builtins::title_case_str(value)))
        }
        BuiltinViewValueProjection::ToBase64 => {
            Val::Str(Arc::from(crate::builtins::to_base64_str(value)))
        }
        BuiltinViewValueProjection::UrlDecode => {
            Val::Str(Arc::from(crate::builtins::url_decode_str(value)))
        }
        BuiltinViewValueProjection::UrlEncode => {
            Val::Str(Arc::from(crate::builtins::url_encode_str(value)))
        }
        _ => unreachable!("string codec projection expected"),
    }
}

fn pad_view_string(
    value: &str,
    width: usize,
    fill: char,
    projection: BuiltinViewValueProjection,
) -> Val {
    let len = value.chars().count();
    if len >= width {
        return Val::Str(Arc::from(value));
    }
    let total = width - len;
    let (left, right) = match projection {
        BuiltinViewValueProjection::PadLeft => (total, 0),
        BuiltinViewValueProjection::PadRight => (0, total),
        BuiltinViewValueProjection::Center => {
            let left = total / 2;
            (left, total - left)
        }
        _ => unreachable!("non-pad projection passed to pad_view_string"),
    };
    let mut out = String::with_capacity(value.len() + total.saturating_mul(fill.len_utf8()));
    out.extend(std::iter::repeat(fill).take(left));
    out.push_str(value);
    out.extend(std::iter::repeat(fill).take(right));
    Val::Str(Arc::from(out))
}

fn replace_view_string(value: &str, needle: &str, replacement: &str, all: bool) -> Val {
    if !value.contains(needle) {
        return Val::Str(Arc::from(value));
    }
    let out = if all {
        value.replace(needle, replacement)
    } else {
        value.replacen(needle, replacement, 1)
    };
    Val::Str(Arc::from(out))
}

fn strip_view_string_affix(value: &str, affix: &str, prefix: bool) -> Val {
    let stripped = if prefix {
        value.strip_prefix(affix)
    } else {
        value.strip_suffix(affix)
    };
    Val::Str(Arc::from(stripped.unwrap_or(value)))
}

fn slice_view_string(value: &str, start: i64, end: Option<i64>) -> Val {
    if value.is_ascii() {
        let len = value.len();
        let start = resolve_ascii_slice_index(start, len);
        let end = end.map_or(len, |end| resolve_ascii_slice_index(end, len));
        let start = start.min(end);
        return Val::Str(Arc::from(&value[start..end]));
    }

    let chars: Vec<(usize, char)> = value.char_indices().collect();
    let len = chars.len() as i64;
    let resolve = |idx: i64| -> usize {
        let resolved = if idx < 0 { len + idx } else { idx };
        resolved.clamp(0, len) as usize
    };
    let start = resolve(start);
    let end = end.map_or(chars.len(), resolve);
    let start = start.min(end);
    let start_byte = chars.get(start).map(|(idx, _)| *idx).unwrap_or(value.len());
    let end_byte = chars.get(end).map(|(idx, _)| *idx).unwrap_or(value.len());
    Val::Str(Arc::from(&value[start_byte..end_byte]))
}

#[inline]
fn resolve_ascii_slice_index(idx: i64, len: usize) -> usize {
    if idx < 0 {
        len.saturating_sub((-idx) as usize)
    } else {
        (idx as usize).min(len)
    }
}

fn write_view_string_value<'a, V>(view: &V, out: &mut String) -> Option<()>
where
    V: ValueView<'a> + 'a,
{
    match view.scalar() {
        JsonView::Null => out.push_str("null"),
        JsonView::Bool(value) => out.push_str(if value { "true" } else { "false" }),
        JsonView::Int(value) => out.push_str(itoa::Buffer::new().format(value)),
        JsonView::UInt(value) => out.push_str(itoa::Buffer::new().format(value)),
        JsonView::Float(value) => out.push_str(ryu::Buffer::new().format(value)),
        JsonView::Str(value) => out.push_str(value),
        JsonView::ArrayLen(_) | JsonView::ObjectLen(_) => write_json_view(view, out)?,
    }
    Some(())
}

fn walk_view_path<'a, V>(mut cur: V, segs: &[crate::builtins::PathSeg]) -> V
where
    V: ValueView<'a>,
{
    for seg in segs {
        cur = match seg {
            crate::builtins::PathSeg::Field(field) => cur.field(field.as_str()),
            crate::builtins::PathSeg::Index(index) => cur.index(*index),
        };
    }
    cur
}

fn view_path_exists<'a, V>(view: &V, path: &[crate::builtins::PathSeg]) -> bool
where
    V: ValueView<'a>,
{
    !matches!(walk_view_path(view.clone(), path).scalar(), JsonView::Null)
}

fn view_has<'a, V>(view: &V, key: &str) -> Option<bool>
where
    V: ValueView<'a>,
{
    if let Some(found) = view.has_key(key) {
        return Some(found);
    }
    if let JsonView::Str(value) = view.scalar() {
        return Some(value.contains(key));
    }
    if let Some(mut iter) = view.array_iter() {
        return Some(iter.any(|item| scalar_matches_key(item.scalar(), key)));
    }
    None
}

fn view_has_all<'a, V>(view: &V, keys: &[std::sync::Arc<str>]) -> Option<bool>
where
    V: ValueView<'a>,
{
    for key in keys {
        if !view_has(view, key.as_ref())? {
            return Some(false);
        }
    }
    Some(true)
}

#[inline]
fn scalar_matches_key(value: JsonView<'_>, key: &str) -> bool {
    match value {
        JsonView::Str(value) => value == key,
        JsonView::Int(value) => value.to_string() == key,
        JsonView::UInt(value) => value.to_string() == key,
        JsonView::Float(value) => value.to_string() == key,
        JsonView::Bool(true) => key == "true",
        JsonView::Bool(false) => key == "false",
        JsonView::Null => key == "null",
        JsonView::ArrayLen(_) | JsonView::ObjectLen(_) => false,
    }
}

/// Return true when builtin `id` can evaluate directly against a scalar
/// `JsonView` without materialising the receiver.
#[inline]
pub(crate) fn view_scalar_projection(id: BuiltinId) -> bool {
    id.method().is_some_and(|method| method.spec().view_scalar)
}

/// Return concrete borrowed-view scalar dispatch metadata for builtin `id`.
#[inline]
pub(crate) fn view_scalar_op(id: BuiltinId) -> Option<BuiltinViewScalarOp> {
    id.method().and_then(|method| method.spec().view_scalar_op)
}

/// Return true when builtin `id` is a view-native scalar call usable as a
/// direct value projection, excluding object-item enumerators that have their
/// own object projection path.
#[inline]
pub(crate) fn view_scalar_value_projection(id: BuiltinId) -> bool {
    view_scalar_projection(id) && !view_object_items_projection(id)
}

/// Return raw-byte scalar execution support for builtin `id`, if the operation
/// can be served directly from a JSON value slice with the given static args.
#[inline]
pub(crate) fn raw_json_scalar(
    id: BuiltinId,
    args: &crate::builtins::BuiltinArgs,
) -> Option<BuiltinRawJsonScalar> {
    if !matches!(args, crate::builtins::BuiltinArgs::None) {
        return None;
    }
    id.method().and_then(|method| method.spec().raw_json_scalar)
}

/// Return true when a concrete builtin call has a raw-byte scalar
/// implementation. This is narrower than view-scalar support: view scalars can
/// run against borrowed tape values, while raw-byte scalars can write directly
/// from a JSON value slice without parsing/materialising that value.
#[inline]
pub(crate) fn raw_json_scalar_call(id: BuiltinId, args: &BuiltinArgs) -> bool {
    raw_json_scalar(id, args).is_some()
}

/// Return the effective pipeline order behaviour for builtin `id`. Explicit
/// registry metadata wins; optionally, pure one-to-one builtins may be treated
/// as order-preserving by callers that allow this conservative fallback.
#[inline]
pub(crate) fn effective_pipeline_order_effect(
    id: BuiltinId,
    allow_one_to_one_fallback: bool,
) -> BuiltinPipelineOrderEffect {
    let Some(method) = id.method() else {
        return BuiltinPipelineOrderEffect::Blocks;
    };
    let spec = method.spec();
    if let Some(effect) = pipeline_order_effect(id) {
        return effect;
    }
    if allow_one_to_one_fallback && spec.cardinality == BuiltinCardinality::OneToOne {
        BuiltinPipelineOrderEffect::Preserves
    } else {
        BuiltinPipelineOrderEffect::Blocks
    }
}

/// Return true when builtin `id` filters rows without reordering or blocking
/// the surviving stream. This is the shared monotonicity/demand fact for
/// selective stages.
#[inline]
#[cfg(test)]
pub(crate) fn preserves_order_filtering(id: BuiltinId) -> bool {
    builtin_cardinality(id) == Some(BuiltinCardinality::Filtering)
        && !matches!(
            effective_pipeline_order_effect(id, false),
            BuiltinPipelineOrderEffect::Blocks
        )
}

/// Return the pipeline lowering strategy for builtin `id`, indicating which
/// physical stage type and arguments the builtin compiles to.
#[inline]
pub(crate) fn pipeline_lowering(id: BuiltinId) -> Option<BuiltinPipelineLowering> {
    id.method().map(|m| m.spec().lowering).flatten()
}

/// Return the terminal sink target for expression-stage builtins that lower as
/// `stage(expr)` followed by a terminal sink, e.g. `find(pred)` -> `filter(pred).first()`.
#[inline]
pub(crate) fn terminal_expr_target(id: BuiltinId) -> Option<BuiltinId> {
    match pipeline_lowering(id)? {
        BuiltinPipelineLowering::TerminalExprArg { terminal } => {
            Some(BuiltinId::from_method(terminal))
        }
        _ => None,
    }
}

/// Return `true` if builtin `id` can be lowered in pipeline position with
/// `arity` arguments. Terminal sinks are only accepted when `is_last` is true.
#[inline]
pub(crate) fn pipeline_accepts_arity(id: BuiltinId, arity: usize, is_last: bool) -> bool {
    pipeline_arity(id, is_last).is_some_and(|accepted| accepted.accepts(arity))
}

/// Return the canonical accepted pipeline arity for builtin `id`. Terminal
/// sinks are only exposed when `is_last` is true.
#[inline]
pub(crate) fn pipeline_arity(id: BuiltinId, is_last: bool) -> Option<BuiltinPipelineArity> {
    let Some(method) = id.method() else {
        return None;
    };
    match pipeline_lowering(id) {
        Some(BuiltinPipelineLowering::ExprArg)
        | Some(BuiltinPipelineLowering::UsizeArg { .. })
        | Some(BuiltinPipelineLowering::StringArg) => Some(BuiltinPipelineArity::Exact(1)),
        Some(BuiltinPipelineLowering::TerminalExprArg { .. }) => {
            is_last.then_some(BuiltinPipelineArity::Exact(1))
        }
        Some(BuiltinPipelineLowering::Nullary) => Some(BuiltinPipelineArity::Exact(0)),
        Some(BuiltinPipelineLowering::StringPairArg) => Some(BuiltinPipelineArity::Exact(2)),
        Some(BuiltinPipelineLowering::IntRangeArg) => {
            Some(BuiltinPipelineArity::Range { min: 1, max: 2 })
        }
        Some(BuiltinPipelineLowering::Sort) => Some(BuiltinPipelineArity::Range { min: 0, max: 1 }),
        Some(BuiltinPipelineLowering::TerminalSink) => {
            is_last.then(|| terminal_sink_arity(method))?
        }
        Some(BuiltinPipelineLowering::TerminalUsizeSink { .. }) => {
            is_last.then_some(BuiltinPipelineArity::Exact(1))
        }
        None => is_last.then(|| terminal_sink_arity(method))?,
    }
}

#[inline]
fn terminal_sink_arity(method: BuiltinMethod) -> Option<BuiltinPipelineArity> {
    let spec = method.spec();
    if spec.sink.is_none() && matches!(spec.lowering, Some(BuiltinPipelineLowering::TerminalSink)) {
        return Some(BuiltinPipelineArity::Exact(1));
    }
    Some(spec.sink?.pipeline_arity())
}

/// Return `true` if builtin `id` is an element-wise operation that can be
/// applied independently to each item in a vectorised column.
#[inline]
pub(crate) fn pipeline_element(id: BuiltinId) -> bool {
    id.method().map(|m| m.spec().is_element).unwrap_or(false)
}

/// Return `true` if a concrete builtin call can be represented as a generic
/// pipeline `Stage::Builtin` stage.
#[inline]
pub(crate) fn pipeline_builtin_call_stage(id: BuiltinId) -> bool {
    pipeline_element(id) || composed_builtin_stage(id).is_some()
}

/// Return the composed-stage implementation for metadata-backed builtin calls
/// that are not ordinary element-wise stages.
#[inline]
pub(crate) fn composed_builtin_stage(id: BuiltinId) -> Option<BuiltinComposedStage> {
    match view_stage(id)? {
        BuiltinViewStage::Compact => Some(BuiltinComposedStage::Compact),
        BuiltinViewStage::RemoveValue => Some(BuiltinComposedStage::RemoveValue),
        _ => None,
    }
}

/// Return the structural traversal variant for builtin `id` (`DeepFind`,
/// `DeepShape`, `DeepLike`), or `None` for non-structural builtins.
#[inline]
pub(crate) fn structural(id: BuiltinId) -> Option<BuiltinStructural> {
    id.method().map(|m| m.spec().structural).flatten()
}

/// Look up the demand law for `id`, returning `Identity` for any unregistered builtin.
#[inline]
fn demand_law(id: BuiltinId) -> BuiltinDemandLaw {
    id.method()
        .map(|m| m.spec().demand_law)
        .unwrap_or(BuiltinDemandLaw::Identity)
}

/// Dispatch the migrated per-row streaming hook for `method`.
///
/// This is intentionally a static match over concrete builtin definition
/// types: hot executors avoid vtables, while the method-to-hook authority
/// remains in the builtin registry instead of being duplicated by backends.
#[inline]
pub(crate) fn apply_stream_hook_or_else<F>(
    method: BuiltinMethod,
    ctx: &mut StreamCtx<'_, '_>,
    item: Val,
    body: Option<&Program>,
    fallback: F,
) -> Result<StageFlow<Val>, EvalError>
where
    F: FnOnce(Val) -> Result<StageFlow<Val>, EvalError>,
{
    let Some(hook) = runtime_hook(BuiltinId::from_method(method)) else {
        return fallback(item);
    };
    if !hook.has_stream() {
        return fallback(item);
    }
    if matches!(hook, BuiltinRuntimeHook::SharedFilter) {
        return <defs::Filter as Builtin>::apply_stream(ctx, item, body);
    }
    macro_rules! trait_arm {
        ( $( $variant:ident ),* $(,)? ) => {
            match method {
                $(
                    BuiltinMethod::$variant => {
                        <defs::$variant as Builtin>::apply_stream(ctx, item, body)
                    }
                )*
            }
        };
    }
    crate::for_each_builtin!(trait_arm)
}

/// Dispatch the migrated end-of-stream hook for buffered streaming stages.
#[inline]
pub(crate) fn finish_stream_hook(
    method: BuiltinMethod,
    ctx: &mut StreamCtx<'_, '_>,
    body: Option<&Program>,
) -> Result<Vec<Val>, EvalError> {
    let Some(hook) = runtime_hook(BuiltinId::from_method(method)) else {
        return Ok(Vec::new());
    };
    if !hook.has_stream() {
        return Ok(Vec::new());
    }
    macro_rules! trait_arm {
        ( $( $variant:ident ),* $(,)? ) => {
            match method {
                $(
                    BuiltinMethod::$variant => {
                        <defs::$variant as Builtin>::finish_stream(ctx, body)
                    }
                )*
            }
        };
    }
    crate::for_each_builtin!(trait_arm)
}

/// Dispatch migrated scalar hooks for a builtin call.
///
/// Like the stream/barrier hook dispatchers, this is a static generated match:
/// no vtable on the hot path, and the executor-facing call site no longer owns
/// an independent builtin hook table.
#[inline]
pub(crate) fn apply_scalar_hook(
    method: BuiltinMethod,
    args: &BuiltinArgs,
    recv: &Val,
) -> Option<Val> {
    macro_rules! trait_arm {
        ( $( $variant:ident ),* $(,)? ) => {
            match method {
                $(
                    BuiltinMethod::$variant => {
                        if matches!(args, BuiltinArgs::None) {
                            if let Some(value) = <defs::$variant as Builtin>::apply_one(recv) {
                                return Some(value);
                            }
                        }
                        <defs::$variant as Builtin>::apply_args(recv, args)
                    }
                )*
            }
        };
    }
    crate::for_each_builtin!(trait_arm)
}

/// Apply a view-scalar builtin directly to a `JsonView` without materializing
/// the receiver into `Val`.
#[inline]
pub(crate) fn apply_json_view_scalar_hook(
    method: BuiltinMethod,
    args: &BuiltinArgs,
    recv: JsonView<'_>,
) -> Option<Val> {
    if !view_scalar_projection(BuiltinId::from_method(method)) {
        return None;
    }
    match (view_scalar_op(BuiltinId::from_method(method))?, args) {
        (BuiltinViewScalarOp::Len, BuiltinArgs::None) => super::json_view_len(recv).map(Val::Int),
        (BuiltinViewScalarOp::TypeName, BuiltinArgs::None) => Some(Val::Str(std::sync::Arc::from(
            super::json_view_type_name(recv),
        ))),
        (BuiltinViewScalarOp::StringNoArg, BuiltinArgs::None) => {
            let value = super::json_view_str(recv)?;
            super::str_no_arg_scalar_apply(method, value)
        }
        (BuiltinViewScalarOp::ParseInt, BuiltinArgs::None) => {
            let value = super::json_view_str(recv)?;
            Some(super::parse_int_str(value))
        }
        (BuiltinViewScalarOp::ParseInt, BuiltinArgs::Usize(radix)) => {
            let value = super::json_view_str(recv)?;
            let radix = *radix as u32;
            if !(2..=36).contains(&radix) {
                return Some(Val::Null);
            }
            Some(super::parse_int_radix_str(value, radix))
        }
        (BuiltinViewScalarOp::ParseInt, BuiltinArgs::I64(radix)) if *radix > 0 => {
            let value = super::json_view_str(recv)?;
            let radix = *radix as u32;
            if !(2..=36).contains(&radix) {
                return Some(Val::Null);
            }
            Some(super::parse_int_radix_str(value, radix))
        }
        (BuiltinViewScalarOp::NumericNoArg, BuiltinArgs::None) => {
            super::numeric_no_arg_scalar_apply(method, recv)
        }
        (BuiltinViewScalarOp::StringArg, BuiltinArgs::Str(arg)) => {
            let value = super::json_view_str(recv)?;
            super::str_arg_scalar_apply(method, value, arg.as_ref())
        }
        (BuiltinViewScalarOp::StringContainsArg, BuiltinArgs::Val(Val::Str(arg))) => {
            let value = super::json_view_str(recv)?;
            Some(Val::Bool(value.contains(arg.as_ref())))
        }
        (BuiltinViewScalarOp::StringContainsArg, BuiltinArgs::Val(Val::StrSlice(arg))) => {
            let value = super::json_view_str(recv)?;
            Some(Val::Bool(value.contains(arg.as_str())))
        }
        (BuiltinViewScalarOp::StringVecArg, BuiltinArgs::StrVec(args)) => {
            let value = super::json_view_str(recv)?;
            super::str_vec_arg_scalar_apply(method, value, args)
        }
        _ => None,
    }
}

/// Dispatch the migrated materialized-buffer hook for `method`.
///
/// Keep this in lockstep with builtin definitions as methods migrate; executor
/// modules should call this helper instead of carrying local builtin hook
/// tables.
#[inline]
pub(crate) fn apply_barrier_hook(
    method: BuiltinMethod,
    ctx: &mut BarrierCtx<'_>,
    buf: &mut Vec<Val>,
    body: Option<&Program>,
) -> Option<Result<(), EvalError>> {
    let hook = runtime_hook(BuiltinId::from_method(method))?;
    if !hook.has_barrier() {
        return None;
    }
    if matches!(hook, BuiltinRuntimeHook::SharedFilter) {
        return <defs::Filter as Builtin>::apply_barrier(ctx, buf, body);
    }
    macro_rules! trait_arm {
        ( $( $variant:ident ),* $(,)? ) => {
            match method {
                $(
                    BuiltinMethod::$variant => {
                        <defs::$variant as Builtin>::apply_barrier(ctx, buf, body)
                    }
                )*
            }
        };
    }
    crate::for_each_builtin!(trait_arm)
}

impl BuiltinId {
    pub(crate) const ALL: Self = Self(BuiltinMethod::All as u16);
    pub(crate) const ANY: Self = Self(BuiltinMethod::Any as u16);
    pub(crate) const APPROX_COUNT_DISTINCT: Self = Self(BuiltinMethod::ApproxCountDistinct as u16);
    pub(crate) const AVG: Self = Self(BuiltinMethod::Avg as u16);
    pub(crate) const COUNT: Self = Self(BuiltinMethod::Count as u16);
    pub(crate) const COUNT_BY: Self = Self(BuiltinMethod::CountBy as u16);
    pub(crate) const DROP_WHILE: Self = Self(BuiltinMethod::DropWhile as u16);
    pub(crate) const FILTER: Self = Self(BuiltinMethod::Filter as u16);
    pub(crate) const FIND_INDEX: Self = Self(BuiltinMethod::FindIndex as u16);
    pub(crate) const FIND_ONE: Self = Self(BuiltinMethod::FindOne as u16);
    pub(crate) const FLAT_MAP: Self = Self(BuiltinMethod::FlatMap as u16);
    pub(crate) const FIRST: Self = Self(BuiltinMethod::First as u16);
    pub(crate) const GROUP_BY: Self = Self(BuiltinMethod::GroupBy as u16);
    pub(crate) const INCLUDES: Self = Self(BuiltinMethod::Includes as u16);
    pub(crate) const INDEX: Self = Self(BuiltinMethod::Index as u16);
    pub(crate) const INDICES_OF: Self = Self(BuiltinMethod::IndicesOf as u16);
    pub(crate) const INDICES_WHERE: Self = Self(BuiltinMethod::IndicesWhere as u16);
    pub(crate) const INDEX_BY: Self = Self(BuiltinMethod::IndexBy as u16);
    pub(crate) const LAST: Self = Self(BuiltinMethod::Last as u16);
    pub(crate) const MAP: Self = Self(BuiltinMethod::Map as u16);
    pub(crate) const MAX: Self = Self(BuiltinMethod::Max as u16);
    pub(crate) const MAX_BY: Self = Self(BuiltinMethod::MaxBy as u16);
    pub(crate) const MIN: Self = Self(BuiltinMethod::Min as u16);
    pub(crate) const MIN_BY: Self = Self(BuiltinMethod::MinBy as u16);
    #[cfg(test)]
    pub(crate) const NTH: Self = Self(BuiltinMethod::Nth as u16);
    pub(crate) const REVERSE: Self = Self(BuiltinMethod::Reverse as u16);
    pub(crate) const SKIP: Self = Self(BuiltinMethod::Skip as u16);
    pub(crate) const SORT: Self = Self(BuiltinMethod::Sort as u16);
    pub(crate) const SUM: Self = Self(BuiltinMethod::Sum as u16);
    pub(crate) const TAKE: Self = Self(BuiltinMethod::Take as u16);
    pub(crate) const TAKE_WHILE: Self = Self(BuiltinMethod::TakeWhile as u16);
    pub(crate) const UNIQUE: Self = Self(BuiltinMethod::Unique as u16);
    pub(crate) const UNIQUE_BY: Self = Self(BuiltinMethod::UniqueBy as u16);

    /// Construct a `BuiltinId` from a `BuiltinMethod` by casting its discriminant to `u16`.
    #[inline]
    pub(crate) fn from_method(method: BuiltinMethod) -> Self {
        BuiltinId(method as u16)
    }

    /// Resolve this `BuiltinId` back to its `BuiltinMethod`, returning `None`
    /// for IDs that do not correspond to any registered method.
    #[inline]
    pub(crate) fn method(self) -> Option<BuiltinMethod> {
        method_from_id(self)
    }
}

// Trait-driven name lookup replaces the old builtin_registry! macro.

#[inline]
pub(crate) fn method_from_id(id: BuiltinId) -> Option<BuiltinMethod> {
    macro_rules! check {
        ( $( $variant:ident ),* $(,)? ) => {
            $(
                if id.0 == BuiltinMethod::$variant as u16 {
                    return Some(BuiltinMethod::$variant);
                }
            )*
        };
    }
    crate::for_each_builtin!(check);
    None
}

#[inline]
pub(crate) fn by_name(name: &str) -> Option<BuiltinId> {
    macro_rules! check {
        ( $( $variant:ident ),* $(,)? ) => {
            $(
                if name == <crate::builtins::defs::$variant as crate::builtins::builtin::Builtin>::NAME {
                    return Some(BuiltinId(BuiltinMethod::$variant as u16));
                }
                if <crate::builtins::defs::$variant as crate::builtins::builtin::Builtin>::ALIASES
                    .contains(&name)
                {
                    return Some(BuiltinId(BuiltinMethod::$variant as u16));
                }
            )*
        };
    }
    crate::for_each_builtin!(check);
    None
}

/// Return true when a global-call form uses its first argument as the receiver.
/// Non-receiver globals keep every argument as a normal subexpression.
#[inline]
pub(crate) fn global_call_uses_receiver_arg(name: &str, arg_len: usize) -> bool {
    arg_len > 0
        && !matches!(
            name,
            "coalesce" | "chain" | "join" | "zip" | "zip_longest" | "product" | "range"
        )
}

/// Return the builtin method name used by `$..method(...)` deep traversal.
#[inline]
pub(crate) fn deep_method_name(name: &str) -> String {
    match name {
        "find" | "find_all" | "findAll" => "deep_find".to_string(),
        "shape" => "deep_shape".to_string(),
        "like" => "deep_like".to_string(),
        other => format!("deep_{other}"),
    }
}

/// Return the canonical source-level name for builtin `id`.
#[inline]
pub(crate) fn canonical_name(id: BuiltinId) -> Option<&'static str> {
    macro_rules! check {
        ( $( $variant:ident ),* $(,)? ) => {
            $(
                if id.0 == BuiltinMethod::$variant as u16 {
                    return Some(<crate::builtins::defs::$variant as crate::builtins::builtin::Builtin>::NAME);
                }
            )*
        };
    }
    crate::for_each_builtin!(check);
    None
}

/// Return identity entries for all registered builtins: (method, canonical, aliases).
#[cfg(test)]
pub(crate) fn all_method_entries() -> Vec<(BuiltinMethod, &'static str, &'static [&'static str])> {
    macro_rules! collect {
        ( $( $variant:ident ),* $(,)? ) => {
            vec![
                $(
                    (BuiltinMethod::$variant,
                     <crate::builtins::defs::$variant as crate::builtins::builtin::Builtin>::NAME,
                     <crate::builtins::defs::$variant as crate::builtins::builtin::Builtin>::ALIASES),
                )*
            ]
        };
    }
    crate::for_each_builtin!(collect)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::{
        BuiltinPipelineLowering, BuiltinPipelineMaterialization, BuiltinPipelineOrderEffect,
        BuiltinRowStreamArg, BuiltinSelectionPosition, BuiltinSinkAccumulator,
        BuiltinStreamingBoundary,
    };

    #[test]
    fn registry_name_lookup_matches_legacy_lookup() {
        for (method, canonical, aliases) in all_method_entries() {
            assert_eq!(by_name(canonical).and_then(BuiltinId::method), Some(method));
            assert_eq!(
                canonical_name(BuiltinId::from_method(method)),
                Some(canonical)
            );
            for alias in aliases {
                assert_eq!(by_name(alias).and_then(BuiltinId::method), Some(method));
            }
        }
        assert_eq!(by_name("missing_builtin"), None);
    }

    #[test]
    fn registry_classifies_global_receiver_calls() {
        assert!(global_call_uses_receiver_arg("upper", 1));
        assert!(global_call_uses_receiver_arg("unknown_custom", 2));
        assert!(!global_call_uses_receiver_arg("upper", 0));
        for name in [
            "coalesce",
            "chain",
            "join",
            "zip",
            "zip_longest",
            "product",
            "range",
        ] {
            assert!(!global_call_uses_receiver_arg(name, 2), "{name}");
        }
    }

    #[test]
    fn registry_maps_deep_method_names() {
        assert_eq!(deep_method_name("find"), "deep_find");
        assert_eq!(deep_method_name("find_all"), "deep_find");
        assert_eq!(deep_method_name("findAll"), "deep_find");
        assert_eq!(deep_method_name("shape"), "deep_shape");
        assert_eq!(deep_method_name("like"), "deep_like");
        assert_eq!(deep_method_name("parent"), "deep_parent");
    }

    #[test]
    fn registry_does_not_accept_obsolete_camel_case_aliases() {
        for name in [
            "toString",
            "flatMap",
            "groupBy",
            "sortBy",
            "uniqueBy",
            "transformKeys",
            "getPath",
            "isBlank",
            "parseInt",
            "startsWith",
            "replaceAll",
        ] {
            assert_eq!(by_name(name), None);
            assert_eq!(BuiltinMethod::from_name(name), BuiltinMethod::Unknown);
        }

        assert_eq!(BuiltinMethod::from_name("group_by"), BuiltinMethod::GroupBy);
        assert_eq!(BuiltinMethod::from_name("exists"), BuiltinMethod::Any);
        assert_eq!(BuiltinMethod::from_name("distinct"), BuiltinMethod::Unique);
        assert_eq!(
            BuiltinMethod::from_name("distinct_by"),
            BuiltinMethod::UniqueBy
        );
        assert_eq!(BuiltinMethod::from_name("rows"), BuiltinMethod::Rows);
        assert!(BuiltinMethod::Rows.spec().stream_source);
        assert_eq!(BuiltinMethod::from_name("lstrip"), BuiltinMethod::TrimLeft);
    }

    #[test]
    fn registry_names_and_aliases_are_unambiguous() {
        use std::collections::BTreeMap;

        let mut seen = BTreeMap::new();
        for (method, canonical, aliases) in all_method_entries() {
            for name in std::iter::once(canonical).chain(aliases.iter().copied()) {
                if let Some(existing) = seen.insert(name, method) {
                    panic!(
                        "builtin name/alias {name:?} is registered for both {existing:?} and {method:?}"
                    );
                }
                assert_eq!(by_name(name).and_then(BuiltinId::method), Some(method));
                assert_eq!(BuiltinMethod::from_name(name), method);
            }
        }
    }

    #[test]
    fn registry_specs_preserve_basic_metadata_invariants() {
        for (method, _, _) in all_method_entries() {
            let spec = method.spec();
            assert!(
                spec.cost.is_finite() && spec.cost >= 0.0,
                "{method:?} has invalid planner cost {}",
                spec.cost
            );

            let numeric_sink = spec
                .sink
                .is_some_and(BuiltinSinkSpec::requires_numeric_reducer);
            assert_eq!(
                spec.numeric_reducer.is_some(),
                numeric_sink,
                "{method:?} numeric reducer metadata must match numeric sink metadata"
            );

            if let Some(view_stage) = spec.view_stage {
                assert_eq!(
                    spec.cardinality,
                    view_stage.cardinality(),
                    "{method:?} view-stage cardinality must match builtin cardinality"
                );
            }
        }
    }

    #[test]
    fn registry_categories_have_coherent_cardinality_and_execution_surfaces() {
        for (method, _, _) in all_method_entries() {
            let spec = method.spec();
            let expected_cardinality = match spec.category {
                BuiltinCategory::Scalar
                | BuiltinCategory::Object
                | BuiltinCategory::Path
                | BuiltinCategory::Serialization
                | BuiltinCategory::Mutation => Some(BuiltinCardinality::OneToOne),
                BuiltinCategory::StreamingOneToOne => Some(BuiltinCardinality::OneToOne),
                BuiltinCategory::StreamingFilter => Some(BuiltinCardinality::Filtering),
                BuiltinCategory::StreamingExpand | BuiltinCategory::Deep => {
                    Some(BuiltinCardinality::Expanding)
                }
                BuiltinCategory::Reducer => Some(BuiltinCardinality::Reducing),
                BuiltinCategory::Positional => Some(BuiltinCardinality::Bounded),
                BuiltinCategory::Barrier | BuiltinCategory::Relational => {
                    Some(BuiltinCardinality::Barrier)
                }
                BuiltinCategory::Unknown => None,
            };

            if let Some(expected) = expected_cardinality {
                assert_eq!(
                    spec.cardinality, expected,
                    "{method:?} category {:?} must advertise {expected:?} cardinality",
                    spec.category
                );
            }

            assert!(
                !spec.view_scalar || spec.view_native,
                "{method:?} view-scalar execution must also be view-native"
            );
            assert!(
                spec.raw_json_scalar.is_none() || (spec.view_scalar && spec.view_native),
                "{method:?} raw-json scalar execution must have view-scalar fallback metadata"
            );
            assert!(
                spec.view_object_projection.is_none() || spec.view_native,
                "{method:?} object projection execution must be view-native"
            );
            assert!(
                spec.view_value_projection.is_none() || spec.view_native,
                "{method:?} value projection execution must be view-native"
            );
            assert!(
                spec.view_stage.is_none() || spec.view_native,
                "{method:?} view-stage lowering must be view-native"
            );
            assert!(
                spec.structural.is_none() || matches!(spec.category, BuiltinCategory::Deep),
                "{method:?} structural backend hints must stay on deep traversal builtins"
            );
            assert!(
                spec.row_stream_op.is_none() || !spec.stream_source,
                "{method:?} stream sources and row-stream operators are separate planner roles"
            );
        }
    }

    #[test]
    fn registry_runtime_hooks_are_explicit_and_coherent() {
        let mut registered: Vec<_> = all_method_entries()
            .into_iter()
            .filter_map(|(method, _, _)| {
                runtime_hook(BuiltinId::from_method(method)).map(|hook| (method, hook))
            })
            .collect();
        registered.sort_by_key(|(method, _)| *method as u16);
        let mut expected = vec![
            (BuiltinMethod::Filter, BuiltinRuntimeHook::SharedFilter),
            (BuiltinMethod::Find, BuiltinRuntimeHook::SharedFilter),
            (BuiltinMethod::FindAll, BuiltinRuntimeHook::SharedFilter),
            (BuiltinMethod::Compact, BuiltinRuntimeHook::StreamAndBarrier),
            (BuiltinMethod::Map, BuiltinRuntimeHook::StreamAndBarrier),
            (BuiltinMethod::FlatMap, BuiltinRuntimeHook::StreamAndBarrier),
            (BuiltinMethod::Remove, BuiltinRuntimeHook::StreamAndBarrier),
            (BuiltinMethod::Take, BuiltinRuntimeHook::StreamAndBarrier),
            (BuiltinMethod::Skip, BuiltinRuntimeHook::StreamAndBarrier),
            (
                BuiltinMethod::TakeWhile,
                BuiltinRuntimeHook::StreamAndBarrier,
            ),
            (
                BuiltinMethod::DropWhile,
                BuiltinRuntimeHook::StreamAndBarrier,
            ),
            (BuiltinMethod::FindIndex, BuiltinRuntimeHook::Barrier),
            (BuiltinMethod::IndicesWhere, BuiltinRuntimeHook::Barrier),
            (BuiltinMethod::MaxBy, BuiltinRuntimeHook::Barrier),
            (BuiltinMethod::MinBy, BuiltinRuntimeHook::Barrier),
            (BuiltinMethod::Sort, BuiltinRuntimeHook::Barrier),
            (BuiltinMethod::Window, BuiltinRuntimeHook::StreamAndBarrier),
            (BuiltinMethod::Chunk, BuiltinRuntimeHook::StreamAndBarrier),
            (BuiltinMethod::GroupBy, BuiltinRuntimeHook::Barrier),
            (BuiltinMethod::CountBy, BuiltinRuntimeHook::Barrier),
            (BuiltinMethod::IndexBy, BuiltinRuntimeHook::Barrier),
            (BuiltinMethod::Split, BuiltinRuntimeHook::StreamAndBarrier),
            (BuiltinMethod::Unique, BuiltinRuntimeHook::StreamAndBarrier),
            (
                BuiltinMethod::UniqueBy,
                BuiltinRuntimeHook::StreamAndBarrier,
            ),
            (BuiltinMethod::Reverse, BuiltinRuntimeHook::Barrier),
            (
                BuiltinMethod::TransformKeys,
                BuiltinRuntimeHook::StreamAndBarrier,
            ),
            (
                BuiltinMethod::TransformValues,
                BuiltinRuntimeHook::StreamAndBarrier,
            ),
            (
                BuiltinMethod::FilterKeys,
                BuiltinRuntimeHook::StreamAndBarrier,
            ),
            (
                BuiltinMethod::FilterValues,
                BuiltinRuntimeHook::StreamAndBarrier,
            ),
            (BuiltinMethod::Lines, BuiltinRuntimeHook::StreamAndBarrier),
            (BuiltinMethod::Words, BuiltinRuntimeHook::StreamAndBarrier),
            (BuiltinMethod::Chars, BuiltinRuntimeHook::StreamAndBarrier),
            (BuiltinMethod::CharsOf, BuiltinRuntimeHook::StreamAndBarrier),
            (BuiltinMethod::Bytes, BuiltinRuntimeHook::StreamAndBarrier),
        ];
        expected.sort_by_key(|(method, _)| *method as u16);
        assert_eq!(registered, expected);

        for (method, hook) in registered {
            let id = BuiltinId::from_method(method);
            let spec = method.spec();
            if hook.has_stream() {
                assert!(
                    spec.view_stage.is_some()
                        || spec.object_lambda.is_some()
                        || spec.view_string_expand.is_some()
                        || matches!(
                            spec.lowering,
                            Some(
                                BuiltinPipelineLowering::UsizeArg { .. }
                                    | BuiltinPipelineLowering::StringArg
                            )
                        )
                        || matches!(hook, BuiltinRuntimeHook::SharedFilter),
                    "{method:?} stream hook must be tied to stage/object-lambda metadata"
                );
            }
            if hook.has_barrier() {
                assert!(
                    spec.lowering.is_some()
                        || spec.view_stage.is_some()
                        || spec.object_lambda.is_some()
                        || spec.sink.is_some()
                        || spec.arg_extreme_sink.is_some(),
                    "{method:?} barrier hook must be tied to lowering, stage, or sink metadata"
                );
            }
            if matches!(hook, BuiltinRuntimeHook::SharedFilter) {
                assert_eq!(
                    expr_stage(id),
                    Some(BuiltinExprStage::Filter),
                    "{method:?} shared filter hook must expose filter expression metadata"
                );
                assert_eq!(
                    demand_law(id),
                    BuiltinDemandLaw::FilterLike,
                    "{method:?} shared filter hook must use filter-like demand"
                );
            }
        }
    }

    #[test]
    fn registry_runtime_hooks_match_materialization_policy() {
        for (method, _, _) in all_method_entries() {
            let id = BuiltinId::from_method(method);
            let Some(hook) = runtime_hook(id) else {
                continue;
            };

            if hook.has_stream() && !pipeline_streams(id) {
                assert!(
                    hook.has_barrier(),
                    "{method:?} exposes a non-streaming stream hook without a barrier fallback"
                );
            }

            let stage_like_barrier = hook == BuiltinRuntimeHook::Barrier
                && (view_stage(id).is_some()
                    || matches!(
                        pipeline_lowering(id),
                        Some(
                            BuiltinPipelineLowering::ExprArg
                                | BuiltinPipelineLowering::Nullary
                                | BuiltinPipelineLowering::Sort
                                | BuiltinPipelineLowering::UsizeArg { .. }
                        )
                    ));
            if stage_like_barrier {
                assert!(
                    !pipeline_streams(id),
                    "{method:?} exposes a barrier stage hook but is marked streaming"
                );
            }
        }
    }

    #[test]
    fn registry_accessors_match_builtin_specs() {
        for (method, _, _) in all_method_entries() {
            let id = BuiltinId::from_method(method);
            let spec = method.spec();
            assert_eq!(builtin_category(id), Some(spec.category), "{method:?}");
            assert_eq!(
                builtin_cardinality(id),
                Some(spec.cardinality),
                "{method:?}"
            );
            assert_eq!(logical_shape(id), spec.logical_shape, "{method:?}");
            assert_eq!(row_stream_op(id), spec.row_stream_op, "{method:?}");
            let expected_expr_stage = spec.expr_stage.or_else(|| {
                matches!(
                    spec.lowering,
                    Some(
                        BuiltinPipelineLowering::ExprArg
                            | BuiltinPipelineLowering::TerminalExprArg { .. }
                    )
                )
                .then_some(BuiltinExprStage::ExprBuiltin)
            });
            assert_eq!(expr_stage(id), expected_expr_stage, "{method:?}");
            let expected_nullary_stage = spec.nullary_stage.or_else(|| {
                (matches!(spec.lowering, Some(BuiltinPipelineLowering::Nullary)) && spec.is_element)
                    .then_some(BuiltinNullaryStage::Element)
            });
            assert_eq!(nullary_stage(id), expected_nullary_stage, "{method:?}");
            assert_eq!(string_pair_stage(id), spec.string_pair_stage, "{method:?}");
            assert_eq!(object_lambda(id), spec.object_lambda, "{method:?}");
            assert_eq!(
                pipeline_materialization(id),
                spec.materialization,
                "{method:?}"
            );
            assert_eq!(pipeline_shape(id), spec.pipeline_shape, "{method:?}");
            assert_eq!(pipeline_order_effect(id), spec.order_effect, "{method:?}");
            assert_eq!(pipeline_lowering(id), spec.lowering, "{method:?}");
            assert_eq!(columnar_stage(id), spec.columnar_stage, "{method:?}");
            assert_eq!(stage_merge(id), spec.stage_merge, "{method:?}");
            assert_eq!(
                method.spec().selection_rewrite,
                spec.selection_rewrite,
                "{method:?}"
            );
            assert_eq!(view_stage(id), spec.view_stage, "{method:?}");
            assert_eq!(
                view_object_projection(id),
                spec.view_object_projection,
                "{method:?}"
            );
            assert_eq!(array_selector(id), spec.array_selector, "{method:?}");
            assert_eq!(builtin_sink(id), spec.sink, "{method:?}");
            assert_eq!(keyed_reducer(id), spec.keyed_reducer, "{method:?}");
            assert_eq!(numeric_reducer(id), spec.numeric_reducer, "{method:?}");
            assert_eq!(arg_extreme_sink(id), spec.arg_extreme_sink, "{method:?}");
            assert_eq!(predicate_sink(id), spec.predicate_sink, "{method:?}");
            assert_eq!(membership_sink(id), spec.membership_sink, "{method:?}");
            assert_eq!(is_pure(id), spec.pure, "{method:?}");
            assert_eq!(cancellation(id), spec.cancellation, "{method:?}");
            assert_eq!(is_idempotent(id), spec.idempotent, "{method:?}");
            assert_eq!(
                pipeline_stage_is_order_only(id),
                spec.order_only,
                "{method:?}"
            );
            assert_eq!(
                output_cap_receiver(id),
                spec.output_cap_receiver,
                "{method:?}"
            );
            assert_eq!(runtime_hook(id), spec.runtime_hook, "{method:?}");
            assert_eq!(structural(id), spec.structural, "{method:?}");
            assert_eq!(
                raw_json_scalar(id, &BuiltinArgs::None),
                spec.raw_json_scalar,
                "{method:?}"
            );
            let effective_shape =
                effective_pipeline_shape(id).expect("registered builtin should have shape");
            let expected_shape = pipeline_shape(id).unwrap_or(BuiltinPipelineShape {
                cardinality: spec.cardinality,
                can_indexed: spec.can_indexed,
                cost: spec.cost,
                selectivity: if matches!(spec.category, BuiltinCategory::StreamingFilter) {
                    0.5
                } else {
                    1.0
                },
            });
            assert_eq!(effective_shape, expected_shape, "{method:?}");
            assert_eq!(
                dispatches_scalar_direct(id),
                spec.dispatches_scalar_direct(),
                "{method:?}"
            );
            assert_eq!(
                accepts_lambda_arg(id),
                spec.accepts_lambda_arg
                    || spec.expr_stage.is_some()
                    || spec.object_lambda.is_some()
                    || spec.keyed_reducer.is_some()
                    || spec.arg_extreme_sink.is_some()
                    || spec.predicate_sink.is_some()
                    || matches!(
                        spec.lowering,
                        Some(
                            BuiltinPipelineLowering::ExprArg
                                | BuiltinPipelineLowering::TerminalExprArg { .. }
                                | BuiltinPipelineLowering::Sort
                        )
                    )
                    || spec.sink.is_some_and(|sink| sink.accepts_predicate),
                "{method:?}"
            );
        }
    }

    #[test]
    fn registry_scalar_direct_dispatch_contracts_are_safe() {
        for (method, _, _) in all_method_entries() {
            let id = BuiltinId::from_method(method);
            let spec = method.spec();
            if !dispatches_scalar_direct(id) {
                continue;
            }

            assert!(
                matches!(
                    spec.category,
                    BuiltinCategory::Scalar | BuiltinCategory::Object
                ),
                "{method:?} scalar-direct dispatch must stay row-local"
            );
            assert_eq!(
                spec.cardinality,
                BuiltinCardinality::OneToOne,
                "{method:?} scalar-direct dispatch must preserve one receiver value"
            );
            assert!(
                spec.pure,
                "{method:?} scalar-direct dispatch must not bypass an impure stream stage"
            );
            assert!(
                !spec.never_unwrap,
                "{method:?} scalar-direct dispatch must honor never_unwrap"
            );
            assert!(
                !spec.stream_source,
                "{method:?} scalar-direct dispatch must not consume row-stream source syntax"
            );
            assert_eq!(spec.sink, None, "{method:?} must not also be a sink");
            assert_eq!(
                spec.keyed_reducer, None,
                "{method:?} must not also be a keyed reducer"
            );
            assert_eq!(
                spec.numeric_reducer, None,
                "{method:?} must not also be a numeric reducer"
            );
            assert_eq!(
                spec.arg_extreme_sink, None,
                "{method:?} must not also be an arg-extreme sink"
            );
            assert_eq!(
                spec.predicate_sink, None,
                "{method:?} must not also be a predicate sink"
            );
            if spec.membership_sink.is_some() {
                assert!(
                    matches!(spec.category, BuiltinCategory::Scalar | BuiltinCategory::Object),
                    "{method:?} membership sink overlap is only safe for receiver-local scalar/object dispatch"
                );
            }
            assert_eq!(
                spec.array_selector, None,
                "{method:?} must not also be an array selector"
            );
            assert_eq!(
                spec.row_stream_op, None,
                "{method:?} must not also be a row-stream operation"
            );
        }
    }

    #[test]
    fn registry_raw_json_scalar_contracts_are_safe_for_byte_paths() {
        for (method, _, _) in all_method_entries() {
            let id = BuiltinId::from_method(method);
            let spec = method.spec();
            let Some(raw) = spec.raw_json_scalar else {
                assert_eq!(
                    raw_json_scalar(id, &crate::builtins::BuiltinArgs::None),
                    None,
                    "{method:?} without raw metadata must not expose raw byte execution"
                );
                continue;
            };

            assert!(spec.pure, "{method:?} raw byte scalar must be pure");
            assert!(
                spec.view_native && spec.view_scalar,
                "{method:?} raw byte scalar must also be valid on borrowed views"
            );
            assert!(
                !spec.stream_source,
                "{method:?} raw byte scalar must not consume stream-source syntax"
            );
            assert_eq!(
                pipeline_materialization(id),
                BuiltinPipelineMaterialization::Streaming,
                "{method:?} raw byte scalar must not force materialization"
            );
            assert_eq!(
                spec.demand_law,
                raw.demand_law(),
                "{method:?} raw byte scalar demand law must match the raw operation"
            );
            assert_eq!(
                raw_json_scalar(id, &crate::builtins::BuiltinArgs::None),
                Some(raw),
                "{method:?} raw byte scalar lookup must round-trip registry metadata"
            );
            assert!(
                raw_json_scalar_call(id, &crate::builtins::BuiltinArgs::None),
                "{method:?} nullary raw byte scalar call must be enabled"
            );
            assert_eq!(
                raw_json_scalar(id, &crate::builtins::BuiltinArgs::Str(Arc::from("arg"))),
                None,
                "{method:?} raw byte scalar calls with args must fall back"
            );
            assert!(
                !raw_json_scalar_call(id, &crate::builtins::BuiltinArgs::Str(Arc::from("arg"))),
                "{method:?} raw byte scalar call predicate must honor call args"
            );
        }
    }

    #[test]
    fn registry_propagates_core_streaming_demands() {
        let filter = BuiltinId::from_method(BuiltinMethod::Filter);
        let map = BuiltinId::from_method(BuiltinMethod::Map);
        let remove = BuiltinId::from_method(BuiltinMethod::Remove);
        let take = BuiltinId::from_method(BuiltinMethod::Take);
        let count = BuiltinId::from_method(BuiltinMethod::Count);
        let unique = BuiltinId::from_method(BuiltinMethod::Unique);
        let group_by = BuiltinId::from_method(BuiltinMethod::GroupBy);
        let count_by = BuiltinId::from_method(BuiltinMethod::CountBy);
        let index_by = BuiltinId::from_method(BuiltinMethod::IndexBy);
        let approx_distinct = BuiltinId::from_method(BuiltinMethod::ApproxCountDistinct);
        let sort = BuiltinId::from_method(BuiltinMethod::Sort);
        let reverse = BuiltinId::from_method(BuiltinMethod::Reverse);
        let take_while = BuiltinId::from_method(BuiltinMethod::TakeWhile);
        let drop_while = BuiltinId::from_method(BuiltinMethod::DropWhile);
        let slice = BuiltinId::from_method(BuiltinMethod::Slice);
        let chunk = BuiltinId::from_method(BuiltinMethod::Chunk);
        let window = BuiltinId::from_method(BuiltinMethod::Window);
        let first = BuiltinId::from_method(BuiltinMethod::First);
        let last = BuiltinId::from_method(BuiltinMethod::Last);
        let nth = BuiltinId::from_method(BuiltinMethod::Nth);

        let demand = propagate_demand(take, BuiltinDemandArg::Usize(3), Demand::RESULT);
        assert_eq!(demand.pull, PullDemand::FirstInput(3));

        let demand = propagate_demand(filter, BuiltinDemandArg::None, demand);
        assert_eq!(demand.pull, PullDemand::UntilOutput(3));
        assert_eq!(demand.value, ValueNeed::Whole);

        let downstream = Demand {
            pull: PullDemand::LastInput(1),
            value: ValueNeed::Whole,
            order: true,
        };
        let demand = propagate_demand(remove, BuiltinDemandArg::None, downstream);
        assert_eq!(demand.pull, PullDemand::LastInput(1));
        assert_eq!(demand.value, ValueNeed::Whole);
        assert!(demand.order);

        let downstream = Demand {
            pull: PullDemand::NthInput(2),
            value: ValueNeed::Whole,
            order: false,
        };
        let demand = propagate_demand(remove, BuiltinDemandArg::None, downstream);
        assert_eq!(demand.pull, PullDemand::All);
        assert_eq!(demand.value, ValueNeed::Whole);
        assert!(demand.order);

        let downstream = Demand {
            pull: PullDemand::NthInput(4),
            value: ValueNeed::Predicate,
            order: false,
        };
        let demand = propagate_demand(map, BuiltinDemandArg::None, downstream);
        assert_eq!(demand.pull, PullDemand::NthInput(4));
        assert_eq!(demand.value, ValueNeed::Whole);
        assert!(!demand.order);

        let downstream = Demand {
            pull: PullDemand::LastInput(1),
            value: ValueNeed::CountOnly,
            order: true,
        };
        let demand = propagate_demand(map, BuiltinDemandArg::None, downstream);
        assert_eq!(demand.pull, PullDemand::LastInput(1));
        assert_eq!(demand.value, ValueNeed::CountOnly);
        assert!(demand.order);

        let demand = propagate_demand(count, BuiltinDemandArg::None, Demand::RESULT);
        assert_eq!(demand.pull, PullDemand::All);
        assert_eq!(demand.value, ValueNeed::CountOnly);
        assert!(!demand.order);

        let downstream = Demand {
            pull: PullDemand::FirstInput(2),
            value: ValueNeed::Whole,
            order: true,
        };
        let demand = propagate_demand(unique, BuiltinDemandArg::None, downstream);
        assert_eq!(demand.pull, PullDemand::UntilOutput(2));
        assert_eq!(demand.value, ValueNeed::Whole);
        assert!(demand.order);

        let flat_map = BuiltinId::from_method(BuiltinMethod::FlatMap);
        let downstream = Demand {
            pull: PullDemand::FirstInput(1),
            value: ValueNeed::Whole,
            order: false,
        };
        let demand = propagate_demand(flat_map, BuiltinDemandArg::None, downstream);
        assert_eq!(demand.pull, PullDemand::All);
        assert_eq!(demand.value, ValueNeed::Whole);
        assert!(demand.order);

        let demand = propagate_demand(count_by, BuiltinDemandArg::None, Demand::RESULT);
        assert_eq!(demand.pull, PullDemand::All);
        assert_eq!(demand.value, ValueNeed::Predicate);
        assert!(!demand.order);

        for id in [group_by, index_by, approx_distinct] {
            let demand = propagate_demand(id, BuiltinDemandArg::None, Demand::RESULT);
            assert_eq!(demand.pull, PullDemand::All);
            assert_eq!(demand.value, ValueNeed::Whole);
            assert!(!demand.order);
        }

        let downstream = Demand {
            pull: PullDemand::FirstInput(5),
            value: ValueNeed::Predicate,
            order: false,
        };
        let demand = propagate_demand(sort, BuiltinDemandArg::None, downstream);
        assert_eq!(demand.pull, PullDemand::All);
        assert_eq!(demand.value, ValueNeed::Whole);
        assert!(demand.order);

        let downstream = Demand {
            pull: PullDemand::FirstInput(2),
            value: ValueNeed::Whole,
            order: true,
        };
        let demand = propagate_demand(reverse, BuiltinDemandArg::None, downstream);
        assert_eq!(demand.pull, PullDemand::LastInput(2));
        assert_eq!(demand.value, ValueNeed::Whole);
        assert!(demand.order);

        let downstream = Demand {
            pull: PullDemand::LastInput(1),
            value: ValueNeed::Whole,
            order: true,
        };
        let demand = propagate_demand(reverse, BuiltinDemandArg::None, downstream);
        assert_eq!(demand.pull, PullDemand::FirstInput(1));
        assert_eq!(demand.value, ValueNeed::Whole);
        assert!(demand.order);

        let downstream = Demand {
            pull: PullDemand::FirstInput(1),
            value: ValueNeed::Whole,
            order: false,
        };
        let demand = propagate_demand(drop_while, BuiltinDemandArg::None, downstream);
        assert_eq!(demand.pull, PullDemand::All);
        assert_eq!(demand.value, ValueNeed::Whole);
        assert!(demand.order);

        let downstream = Demand {
            pull: PullDemand::LastInput(1),
            value: ValueNeed::Whole,
            order: true,
        };
        let demand = propagate_demand(take_while, BuiltinDemandArg::None, downstream);
        assert_eq!(demand.pull, PullDemand::All);
        assert_eq!(demand.value, ValueNeed::Whole);
        assert!(demand.order);

        let downstream = Demand {
            pull: PullDemand::NthInput(2),
            value: ValueNeed::Whole,
            order: false,
        };
        let demand = propagate_demand(take_while, BuiltinDemandArg::None, downstream);
        assert_eq!(demand.pull, PullDemand::All);
        assert_eq!(demand.value, ValueNeed::Whole);
        assert!(!demand.order);

        let downstream = Demand {
            pull: PullDemand::LastInput(1),
            value: ValueNeed::Predicate,
            order: true,
        };
        let demand = propagate_demand(slice, BuiltinDemandArg::None, downstream);
        assert_eq!(demand.pull, PullDemand::LastInput(1));
        assert_eq!(demand.value, ValueNeed::Whole);
        assert!(demand.order);

        let downstream = Demand {
            pull: PullDemand::FirstInput(3),
            value: ValueNeed::Whole,
            order: true,
        };
        let demand = propagate_demand(chunk, BuiltinDemandArg::Usize(4), downstream);
        assert_eq!(demand.pull, PullDemand::FirstInput(12));
        assert_eq!(demand.value, ValueNeed::Whole);

        let downstream = Demand {
            pull: PullDemand::UntilOutput(3),
            value: ValueNeed::Whole,
            order: true,
        };
        let demand = propagate_demand(window, BuiltinDemandArg::Usize(4), downstream);
        assert_eq!(demand.pull, PullDemand::FirstInput(6));
        assert_eq!(demand.value, ValueNeed::Whole);

        let downstream = Demand {
            pull: PullDemand::All,
            value: ValueNeed::Predicate,
            order: false,
        };
        let demand = propagate_demand(first, BuiltinDemandArg::None, downstream);
        assert_eq!(demand.pull, PullDemand::FirstInput(1));
        assert_eq!(demand.value, ValueNeed::Predicate);
        assert!(!demand.order);

        let demand = propagate_demand(last, BuiltinDemandArg::None, downstream);
        assert_eq!(demand.pull, PullDemand::LastInput(1));
        assert_eq!(demand.value, ValueNeed::Predicate);
        assert!(demand.order);

        let demand = propagate_demand(nth, BuiltinDemandArg::Usize(3), downstream);
        assert_eq!(demand.pull, PullDemand::NthInput(3));
        assert_eq!(demand.value, ValueNeed::Predicate);
        assert!(!demand.order);
    }

    #[test]
    fn registry_converts_sink_demands() {
        let first = sink_demand(builtin_sink(BuiltinId::FIRST).unwrap());
        assert_eq!(first.pull, PullDemand::FirstInput(1));
        assert_eq!(first.value, ValueNeed::Whole);

        let last = sink_demand(builtin_sink(BuiltinId::LAST).unwrap());
        assert_eq!(last.pull, PullDemand::LastInput(1));
        assert_eq!(last.value, ValueNeed::Whole);

        let count = sink_demand(builtin_sink(BuiltinId::COUNT).unwrap());
        assert_eq!(count.pull, PullDemand::All);
        assert_eq!(count.value, ValueNeed::CountOnly);
        assert!(!count.order);
    }

    #[test]
    fn sink_specs_expose_accumulator_value_need() {
        assert_eq!(
            builtin_sink(BuiltinId::COUNT).unwrap().value_need(),
            BuiltinSinkValueNeed::None
        );
        assert_eq!(
            builtin_sink(BuiltinId::from_method(BuiltinMethod::Len))
                .unwrap()
                .value_need(),
            BuiltinSinkValueNeed::None
        );
        assert_eq!(
            builtin_sink(BuiltinId::SUM).unwrap().value_need(),
            BuiltinSinkValueNeed::Numeric
        );
        assert!(builtin_sink(BuiltinId::SUM)
            .unwrap()
            .requires_numeric_reducer());
        assert!(builtin_sink(BuiltinId::SUM)
            .unwrap()
            .accumulator
            .finishes_from_reducer_state());
        assert!(!builtin_sink(BuiltinId::COUNT)
            .unwrap()
            .requires_numeric_reducer());
        assert!(builtin_sink(BuiltinId::COUNT)
            .unwrap()
            .accumulator
            .finishes_from_reducer_state());
        assert_eq!(
            builtin_sink(BuiltinId::FIRST).unwrap().value_need(),
            BuiltinSinkValueNeed::Whole
        );
        assert!(!builtin_sink(BuiltinId::FIRST)
            .unwrap()
            .accumulator
            .finishes_from_reducer_state());
        assert_eq!(
            builtin_sink(BuiltinId::APPROX_COUNT_DISTINCT)
                .unwrap()
                .value_need(),
            BuiltinSinkValueNeed::Whole
        );
    }

    #[test]
    fn registry_sink_demands_match_all_sink_accumulators() {
        for (method, _, _) in all_method_entries() {
            let Some(sink) = method.spec().sink else {
                continue;
            };
            let id = BuiltinId::from_method(method);
            let demand = sink_demand(sink);
            match sink.accumulator {
                BuiltinSinkAccumulator::Count => {
                    assert_eq!(demand_law(id), BuiltinDemandLaw::Count, "{method:?}");
                    assert_eq!(demand.pull, PullDemand::All, "{method:?}");
                    assert_eq!(demand.value, ValueNeed::CountOnly, "{method:?}");
                    assert!(!demand.order, "{method:?}");
                }
                BuiltinSinkAccumulator::Numeric => {
                    assert_eq!(
                        demand_law(id),
                        BuiltinDemandLaw::NumericReducer,
                        "{method:?}"
                    );
                    assert_eq!(demand.pull, PullDemand::All, "{method:?}");
                    assert_eq!(demand.value, ValueNeed::Numeric, "{method:?}");
                    assert!(!demand.order, "{method:?}");
                }
                BuiltinSinkAccumulator::ApproxDistinct => {
                    assert_eq!(
                        demand_law(id),
                        BuiltinDemandLaw::RowKeyedReducer,
                        "{method:?}"
                    );
                    assert_eq!(demand.pull, PullDemand::All, "{method:?}");
                    assert_eq!(demand.value, ValueNeed::Whole, "{method:?}");
                    assert!(!demand.order, "{method:?}");
                }
                BuiltinSinkAccumulator::SelectOne(BuiltinSelectionPosition::First) => {
                    assert!(!BuiltinSelectionPosition::First.wants_last(), "{method:?}");
                    assert_eq!(demand_law(id), BuiltinDemandLaw::First, "{method:?}");
                    assert_eq!(demand.pull, PullDemand::FirstInput(1), "{method:?}");
                    assert_eq!(demand.value, ValueNeed::Whole, "{method:?}");
                    assert!(!demand.order, "{method:?}");
                }
                BuiltinSinkAccumulator::SelectOne(BuiltinSelectionPosition::Last) => {
                    assert!(BuiltinSelectionPosition::Last.wants_last(), "{method:?}");
                    assert_eq!(demand_law(id), BuiltinDemandLaw::Last, "{method:?}");
                    assert_eq!(demand.pull, PullDemand::LastInput(1), "{method:?}");
                    assert_eq!(demand.value, ValueNeed::Whole, "{method:?}");
                    assert!(demand.order, "{method:?}");
                }
            }
        }
    }

    #[test]
    fn registry_numeric_reducers_have_complete_sink_contracts() {
        for (method, _, _) in all_method_entries() {
            let id = BuiltinId::from_method(method);
            let Some(reducer) = numeric_reducer(id) else {
                continue;
            };
            let sink = builtin_sink(id)
                .unwrap_or_else(|| panic!("{method:?} numeric reducer must expose sink metadata"));

            assert_eq!(reducer.method(), method, "{method:?}");
            assert!(
                sink.requires_numeric_reducer(),
                "{method:?} numeric reducer must expose numeric sink metadata"
            );
            assert_eq!(demand_law(id), reducer.demand_law(), "{method:?}");
            assert_eq!(
                reducer.selects_min(),
                matches!(reducer, BuiltinNumericReducer::Min),
                "{method:?}"
            );
            assert_eq!(
                logical_shape(id),
                Some(reducer.logical_shape()),
                "{method:?}"
            );
            assert_eq!(
                pipeline_lowering(id),
                Some(BuiltinPipelineLowering::TerminalSink),
                "{method:?}"
            );
            assert_eq!(
                row_stream_op(id),
                Some(reducer.row_stream_op()),
                "{method:?}"
            );
            assert_eq!(
                row_stream_op(id).map(BuiltinRowStreamOp::arg),
                Some(BuiltinRowStreamArg::None),
                "{method:?}"
            );
        }
    }

    #[test]
    fn registry_predicate_sinks_have_complete_terminal_contracts() {
        for (method, _, _) in all_method_entries() {
            let id = BuiltinId::from_method(method);
            let Some(sink) = predicate_sink(id) else {
                continue;
            };

            assert_eq!(
                pipeline_lowering(id),
                Some(BuiltinPipelineLowering::TerminalSink),
                "{method:?}"
            );
            assert_eq!(demand_law(id), sink.demand_law(), "{method:?}");
            assert_eq!(
                sink_accumulator(id),
                None,
                "{method:?} predicate sink must not also expose accumulator metadata"
            );
            assert!(
                accepts_lambda_arg(id),
                "{method:?} predicate sink must accept a predicate expression"
            );
            match sink {
                BuiltinPredicateSink::FindOne => {
                    assert!(sink.returns_matching_row(), "{method:?}");
                    assert_eq!(
                        predicate_sink_value_need(sink),
                        ValueNeed::Whole,
                        "{method:?}"
                    );
                    assert_eq!(
                        predicate_sink_result_demand(sink),
                        SinkResultDemand::None,
                        "{method:?}"
                    );
                    assert_eq!(
                        sink.view_materialization(),
                        crate::builtins::BuiltinViewMaterialization::SinkFinalRow,
                        "{method:?}"
                    );
                }
                BuiltinPredicateSink::Any
                | BuiltinPredicateSink::All
                | BuiltinPredicateSink::FindIndex
                | BuiltinPredicateSink::IndicesWhere => {
                    assert!(!sink.returns_matching_row(), "{method:?}");
                    assert_eq!(
                        predicate_sink_value_need(sink),
                        ValueNeed::Predicate,
                        "{method:?}"
                    );
                    assert_eq!(
                        sink.view_materialization(),
                        crate::builtins::BuiltinViewMaterialization::Never,
                        "{method:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn registry_membership_sinks_have_complete_terminal_contracts() {
        for (method, _, _) in all_method_entries() {
            let id = BuiltinId::from_method(method);
            let Some(sink) = membership_sink(id) else {
                continue;
            };

            assert_eq!(sink.method(), method, "{method:?}");
            assert_eq!(
                pipeline_lowering(id),
                Some(BuiltinPipelineLowering::TerminalSink),
                "{method:?}"
            );
            assert_eq!(
                pipeline_arity(id, true),
                Some(BuiltinPipelineArity::Exact(1)),
                "{method:?}"
            );
            assert_eq!(demand_law(id), sink.demand_law(), "{method:?}");
            assert_eq!(sink.method(), method, "{method:?}");
            assert_eq!(
                sink_accumulator(id),
                None,
                "{method:?} membership sink must not also expose accumulator metadata"
            );
            assert_eq!(
                row_stream_op(id),
                None,
                "{method:?} membership sink must not silently become a rows() stream op"
            );
            assert!(
                !accepts_lambda_arg(id),
                "{method:?} membership sink expects a target value, not a predicate expression"
            );
            assert_eq!(
                membership_sink_value_need(sink),
                ValueNeed::Whole,
                "{method:?}"
            );
            match sink {
                BuiltinMembershipSink::Includes => {
                    assert!(sink.returns_bool(), "{method:?}");
                    assert_eq!(
                        membership_sink_result_demand(sink),
                        SinkResultDemand::UntilMatch,
                        "{method:?}"
                    );
                }
                BuiltinMembershipSink::Index => {
                    assert!(!sink.returns_bool(), "{method:?}");
                    assert_eq!(
                        membership_sink_result_demand(sink),
                        SinkResultDemand::UntilMatch,
                        "{method:?}"
                    );
                }
                BuiltinMembershipSink::IndicesOf => {
                    assert!(!sink.returns_bool(), "{method:?}");
                    assert_eq!(
                        membership_sink_result_demand(sink),
                        SinkResultDemand::None,
                        "{method:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn registry_arg_extreme_sinks_have_complete_terminal_contracts() {
        for (method, _, _) in all_method_entries() {
            let id = BuiltinId::from_method(method);
            let Some(sink) = arg_extreme_sink(id) else {
                continue;
            };

            assert_eq!(
                pipeline_lowering(id),
                Some(BuiltinPipelineLowering::TerminalSink),
                "{method:?}"
            );
            assert_eq!(
                pipeline_arity(id, true),
                Some(BuiltinPipelineArity::Exact(1)),
                "{method:?}"
            );
            assert_eq!(demand_law(id), sink.demand_law(), "{method:?}");
            assert_eq!(
                sink_accumulator(id),
                None,
                "{method:?} arg-extreme sink must not also expose accumulator metadata"
            );
            assert_eq!(
                row_stream_op(id),
                None,
                "{method:?} arg-extreme sink needs global row ordering and key comparison"
            );
            assert!(
                accepts_lambda_arg(id),
                "{method:?} arg-extreme sink must accept a key expression"
            );
            match sink {
                BuiltinArgExtremeSink::MaxBy => assert!(sink.wants_max(), "{method:?}"),
                BuiltinArgExtremeSink::MinBy => assert!(!sink.wants_max(), "{method:?}"),
            }
        }
    }

    #[test]
    fn registry_drives_direct_scalar_sink_projection() {
        let count = direct_scalar_for_plain_sink(BuiltinId::from_method(BuiltinMethod::Count))
            .expect("count should project as len");
        assert_eq!(count.method, BuiltinMethod::Len);
        assert!(matches!(count.args, BuiltinArgs::None));
        assert_eq!(
            direct_view_call(BuiltinId::from_method(count.method), &count.args),
            Some(BuiltinDirectViewCall::ScalarValue)
        );
        assert!(
            raw_json_scalar_call(BuiltinId::from_method(count.method), &count.args),
            "direct scalar sink projection must be writable from raw NDJSON bytes"
        );

        for method in [
            BuiltinMethod::First,
            BuiltinMethod::Last,
            BuiltinMethod::Sum,
            BuiltinMethod::Avg,
            BuiltinMethod::ApproxCountDistinct,
        ] {
            assert!(
                direct_scalar_for_plain_sink(BuiltinId::from_method(method)).is_none(),
                "{method:?}"
            );
        }
    }

    #[test]
    fn registry_keyed_reducer_contracts_are_exhaustive() {
        let mut actual = Vec::new();
        for (method, name, _) in all_method_entries() {
            let id = BuiltinId::from_method(method);
            if let Some(reducer) = keyed_reducer(id) {
                actual.push((name, reducer));

                let spec = method.spec();
                assert_eq!(spec.category, BuiltinCategory::Reducer, "{method:?}");
                assert_eq!(spec.cardinality, BuiltinCardinality::Reducing, "{method:?}");
                assert_eq!(
                    view_stage(id),
                    Some(BuiltinViewStage::KeyedReduce),
                    "{method:?}"
                );
                assert_eq!(
                    pipeline_materialization(id),
                    BuiltinPipelineMaterialization::ComposedBarrier,
                    "{method:?}"
                );
                assert_eq!(
                    effective_pipeline_shape(id).map(|shape| shape.cardinality),
                    Some(BuiltinCardinality::Reducing),
                    "{method:?} keyed reducers must advertise reducing pipeline shape"
                );
                assert_eq!(
                    effective_pipeline_shape(id).map(|shape| shape.can_indexed),
                    Some(false),
                    "{method:?} keyed reducers must not advertise indexed dispatch"
                );
                assert_eq!(
                    effective_pipeline_order_effect(id, true),
                    BuiltinPipelineOrderEffect::Blocks,
                    "{method:?}"
                );
                assert!(pipeline_composed_barrier(id), "{method:?}");
                assert!(participates_in_demand(id), "{method:?}");
                assert!(
                    matches!(
                        pipeline_lowering(id),
                        Some(BuiltinPipelineLowering::ExprArg)
                            | Some(BuiltinPipelineLowering::TerminalExprArg {
                                terminal: BuiltinMethod::First
                            })
                    ),
                    "{method:?}"
                );

                match reducer {
                    BuiltinKeyedReducer::Count => {
                        assert_eq!(method, BuiltinMethod::CountBy);
                        assert!(!reducer.needs_row_payload());
                        assert_eq!(logical_shape(id), Some(BuiltinLogicalShape::CountBy));
                    }
                    BuiltinKeyedReducer::Group => {
                        assert_eq!(method, BuiltinMethod::GroupBy);
                        assert!(reducer.needs_row_payload());
                        assert_eq!(logical_shape(id), Some(BuiltinLogicalShape::GroupBy));
                    }
                    BuiltinKeyedReducer::Index => {
                        assert_eq!(method, BuiltinMethod::IndexBy);
                        assert!(reducer.needs_row_payload());
                        assert_eq!(logical_shape(id), Some(BuiltinLogicalShape::IndexBy));
                    }
                }
                assert_eq!(demand_law(id), reducer.demand_law(), "{method:?}");
                assert_eq!(expr_payload(id), Some(reducer.expr_payload()), "{method:?}");
            }
        }

        actual.sort_by_key(|(name, _)| *name);
        assert_eq!(
            actual,
            vec![
                ("count_by", BuiltinKeyedReducer::Count),
                ("group_by", BuiltinKeyedReducer::Group),
                ("index_by", BuiltinKeyedReducer::Index),
            ]
        );
    }

    #[test]
    fn registry_order_barrier_contracts_drive_bounded_selection() {
        let sort = BuiltinId::from_method(BuiltinMethod::Sort);
        assert_eq!(demand_law(sort), BuiltinDemandLaw::OrderBarrier);
        assert_eq!(
            pipeline_materialization(sort),
            BuiltinPipelineMaterialization::ComposedBarrier
        );
        assert_eq!(pipeline_lowering(sort), Some(BuiltinPipelineLowering::Sort));
        assert_eq!(logical_shape(sort), Some(BuiltinLogicalShape::Sort));
        assert!(pipeline_stage_is_order_only(sort));
        assert!(pipeline_composed_barrier(sort));
        assert!(is_idempotent(sort));
        assert_eq!(row_stream_op(sort), None);
        assert_eq!(
            terminal_selection_rewrite(sort, BuiltinSelectionPosition::First),
            Some(BuiltinMethod::Min)
        );
        assert_eq!(
            terminal_selection_rewrite(sort, BuiltinSelectionPosition::Last),
            Some(BuiltinMethod::Max)
        );
        assert_eq!(index_selection_rewrite(sort, 0), Some(BuiltinMethod::Min));
        assert_eq!(index_selection_rewrite(sort, -1), Some(BuiltinMethod::Max));
        assert_eq!(index_selection_rewrite(sort, 1), None);
        assert_eq!(
            effective_pipeline_order_effect(sort, true),
            BuiltinPipelineOrderEffect::Blocks
        );
        assert!(stage_elidable_before_sink(
            sort,
            BuiltinId::from_method(BuiltinMethod::Sum)
        ));
        assert!(!stage_elidable_before_sink(
            sort,
            BuiltinId::from_method(BuiltinMethod::First)
        ));
        assert!(!stage_elidable_before_sink(
            sort,
            BuiltinId::from_method(BuiltinMethod::Last)
        ));

        let reverse = BuiltinId::from_method(BuiltinMethod::Reverse);
        assert_eq!(demand_law(reverse), BuiltinDemandLaw::Reverse);
        assert_eq!(
            pipeline_materialization(reverse),
            BuiltinPipelineMaterialization::ComposedBarrier
        );
        assert_eq!(
            pipeline_lowering(reverse),
            Some(BuiltinPipelineLowering::Nullary)
        );
        assert_eq!(logical_shape(reverse), Some(BuiltinLogicalShape::Reverse));
        assert!(pipeline_stage_is_order_only(reverse));
        assert!(pipeline_composed_barrier(reverse));
        assert_eq!(row_stream_op(reverse), Some(BuiltinRowStreamOp::Reverse));
        assert_eq!(
            terminal_selection_rewrite(reverse, BuiltinSelectionPosition::First),
            Some(BuiltinMethod::Last)
        );
        assert_eq!(
            terminal_selection_rewrite(reverse, BuiltinSelectionPosition::Last),
            Some(BuiltinMethod::First)
        );
        assert_eq!(index_selection_rewrite(reverse, 0), None);
    }

    #[test]
    fn registry_unsafe_streaming_stages_keep_conservative_demand_contracts() {
        let prefix_downstream = Demand {
            pull: PullDemand::FirstInput(3),
            value: ValueNeed::CountOnly,
            order: false,
        };
        let last_downstream = Demand {
            pull: PullDemand::LastInput(1),
            value: ValueNeed::CountOnly,
            order: false,
        };

        let take_while = BuiltinId::from_method(BuiltinMethod::TakeWhile);
        assert_eq!(demand_law(take_while), BuiltinDemandLaw::TakeWhile);
        assert_eq!(view_stage(take_while), Some(BuiltinViewStage::TakeWhile));
        assert_eq!(
            pipeline_lowering(take_while),
            Some(BuiltinPipelineLowering::ExprArg)
        );
        assert_eq!(
            pipeline_order_effect(take_while),
            Some(BuiltinPipelineOrderEffect::PredicatePrefix)
        );
        assert!(!demand_is_conservative_barrier(take_while));
        assert_eq!(
            propagate_demand(take_while, BuiltinDemandArg::None, prefix_downstream),
            Demand {
                pull: PullDemand::FirstInput(3),
                value: ValueNeed::Predicate,
                order: false,
            }
        );
        assert_eq!(
            propagate_demand(take_while, BuiltinDemandArg::None, last_downstream),
            Demand {
                pull: PullDemand::All,
                value: ValueNeed::Predicate,
                order: false,
            }
        );

        let drop_while = BuiltinId::from_method(BuiltinMethod::DropWhile);
        assert_eq!(demand_law(drop_while), BuiltinDemandLaw::DropWhile);
        assert_eq!(view_stage(drop_while), Some(BuiltinViewStage::DropWhile));
        assert_eq!(
            pipeline_materialization(drop_while),
            BuiltinPipelineMaterialization::Streaming
        );
        assert_eq!(
            pipeline_order_effect(drop_while),
            Some(BuiltinPipelineOrderEffect::Blocks)
        );
        assert!(demand_is_conservative_barrier(drop_while));
        assert_eq!(
            propagate_demand(drop_while, BuiltinDemandArg::None, prefix_downstream),
            Demand {
                pull: PullDemand::All,
                value: ValueNeed::Predicate,
                order: true,
            }
        );

        let flat_map = BuiltinId::from_method(BuiltinMethod::FlatMap);
        assert_eq!(demand_law(flat_map), BuiltinDemandLaw::FlatMapLike);
        assert_eq!(view_stage(flat_map), Some(BuiltinViewStage::FlatMap));
        assert_eq!(
            pipeline_materialization(flat_map),
            BuiltinPipelineMaterialization::Streaming
        );
        assert!(demand_is_conservative_barrier(flat_map));
        assert_eq!(
            propagate_demand(flat_map, BuiltinDemandArg::None, prefix_downstream),
            Demand {
                pull: PullDemand::All,
                value: ValueNeed::Whole,
                order: true,
            }
        );

        for method in [BuiltinMethod::Unique, BuiltinMethod::UniqueBy] {
            let id = BuiltinId::from_method(method);
            assert_eq!(demand_law(id), BuiltinDemandLaw::UniqueLike, "{method:?}");
            assert_eq!(
                view_stage(id),
                Some(BuiltinViewStage::Distinct),
                "{method:?}"
            );
            assert_eq!(
                pipeline_materialization(id),
                BuiltinPipelineMaterialization::Streaming,
                "{method:?}"
            );
            assert_eq!(
                pipeline_order_effect(id),
                Some(BuiltinPipelineOrderEffect::Preserves),
                "{method:?}"
            );
            assert!(!demand_is_conservative_barrier(id), "{method:?}");
            assert_eq!(
                propagate_demand(id, BuiltinDemandArg::None, prefix_downstream),
                Demand {
                    pull: PullDemand::UntilOutput(3),
                    value: ValueNeed::Whole,
                    order: true,
                },
                "{method:?}"
            );
            assert_eq!(
                propagate_demand(id, BuiltinDemandArg::None, last_downstream),
                Demand {
                    pull: PullDemand::All,
                    value: ValueNeed::Whole,
                    order: true,
                },
                "{method:?}"
            );
        }
    }

    #[test]
    fn registry_logical_shapes_participate_in_demand_model() {
        for (method, _, _) in all_method_entries() {
            let id = BuiltinId::from_method(method);
            if logical_shape(id).is_some() {
                assert!(
                    participates_in_demand(id),
                    "{method:?} has logical pipeline shape but no demand metadata"
                );
            }
        }
    }

    #[test]
    fn registry_pipeline_lowerings_participate_in_demand_model() {
        for (method, _, _) in all_method_entries() {
            let id = BuiltinId::from_method(method);
            if pipeline_lowering(id).is_some() {
                assert!(
                    participates_in_demand(id),
                    "{method:?} has pipeline lowering but no demand metadata"
                );
            }
        }
    }

    #[test]
    fn registry_pipeline_lowerings_have_execution_metadata() {
        for (method, _, _) in all_method_entries() {
            let id = BuiltinId::from_method(method);
            let spec = method.spec();
            match pipeline_lowering(id) {
                Some(BuiltinPipelineLowering::ExprArg) => {
                    assert!(
                        spec.expr_stage.is_some()
                            || spec.object_lambda.is_some()
                            || spec.keyed_reducer.is_some()
                            || spec.arg_extreme_sink.is_some()
                            || spec.structural.is_some()
                            || matches!(
                                spec.logical_shape,
                                Some(
                                    BuiltinLogicalShape::TakeWhile | BuiltinLogicalShape::DropWhile
                                )
                            ),
                        "{method:?} has ExprArg lowering but no expression execution metadata"
                    );
                }
                Some(BuiltinPipelineLowering::TerminalExprArg { terminal }) => {
                    assert!(
                        spec.logical_shape.is_some()
                            || spec.keyed_reducer.is_some()
                            || spec.predicate_sink.is_some(),
                        "{method:?} has TerminalExprArg lowering but no terminal expression metadata"
                    );
                    assert!(
                        terminal_selection_position(BuiltinId::from_method(terminal)).is_some(),
                        "{method:?} TerminalExprArg target {terminal:?} is not a select-one sink"
                    );
                    assert_eq!(
                        terminal_expr_target(id),
                        Some(BuiltinId::from_method(terminal)),
                        "{method:?} terminal expression target accessor must mirror lowering metadata"
                    );
                }
                Some(BuiltinPipelineLowering::TerminalSink) => {
                    assert!(
                        spec.sink.is_some()
                            || spec.predicate_sink.is_some()
                            || spec.membership_sink.is_some()
                            || spec.arg_extreme_sink.is_some(),
                        "{method:?} has TerminalSink lowering but no sink metadata"
                    );
                }
                Some(BuiltinPipelineLowering::TerminalUsizeSink { .. }) => {
                    assert!(
                        spec.array_selector.is_some()
                            || matches!(demand_law(id), BuiltinDemandLaw::Nth),
                        "{method:?} has TerminalUsizeSink lowering but no positional metadata"
                    );
                }
                Some(BuiltinPipelineLowering::Nullary) => {
                    assert!(
                        spec.nullary_stage.is_some()
                            || spec.view_object_projection.is_some()
                            || spec.view_value_projection.is_some()
                            || spec.view_string_expand.is_some()
                            || spec.view_scalar,
                        "{method:?} has Nullary lowering but no nullary/view execution metadata"
                    );
                }
                Some(
                    BuiltinPipelineLowering::UsizeArg { .. }
                    | BuiltinPipelineLowering::StringArg
                    | BuiltinPipelineLowering::StringPairArg
                    | BuiltinPipelineLowering::IntRangeArg
                    | BuiltinPipelineLowering::Sort,
                )
                | None => {}
            }
        }
    }

    #[test]
    fn registry_indexed_pipeline_shapes_are_cardinality_preserving() {
        for (method, _, _) in all_method_entries() {
            let id = BuiltinId::from_method(method);
            if !pipeline_element(id) {
                continue;
            }
            let Some(shape) = effective_pipeline_shape(id) else {
                continue;
            };
            if !shape.can_indexed {
                continue;
            }

            assert_eq!(
                shape.cardinality,
                BuiltinCardinality::OneToOne,
                "{method:?} indexed dispatch requires one-to-one cardinality"
            );
            assert_eq!(
                effective_pipeline_order_effect(id, false),
                BuiltinPipelineOrderEffect::Preserves,
                "{method:?} indexed dispatch requires preserved order"
            );
        }
    }

    #[test]
    fn registry_pipeline_elements_participate_in_demand_model() {
        for (method, _, _) in all_method_entries() {
            let id = BuiltinId::from_method(method);
            if pipeline_element(id) {
                assert!(
                    participates_in_demand(id),
                    "{method:?} is a pipeline element but has no demand metadata"
                );
            }
        }
    }

    #[test]
    fn registry_barrier_lowerings_declare_materialization_boundary() {
        for (method, _, _) in all_method_entries() {
            let id = BuiltinId::from_method(method);
            let spec = method.spec();
            if pipeline_lowering(id).is_some() && spec.cardinality == BuiltinCardinality::Barrier {
                assert_ne!(
                    pipeline_materialization(id),
                    BuiltinPipelineMaterialization::Streaming,
                    "{method:?} lowers as a barrier but inherits streaming materialization"
                );
            }
        }
    }

    #[test]
    fn registry_execution_surfaces_have_explicit_demand_laws() {
        for (method, _, _) in all_method_entries() {
            let id = BuiltinId::from_method(method);
            let spec = method.spec();
            let participates = logical_shape(id).is_some()
                || pipeline_lowering(id).is_some()
                || pipeline_element(id)
                || view_stage(id).is_some()
                || row_stream_op(id).is_some()
                || structural(id).is_some()
                || view_projection(id)
                || spec.sink.is_some()
                || spec.keyed_reducer.is_some()
                || spec.numeric_reducer.is_some()
                || spec.arg_extreme_sink.is_some()
                || spec.predicate_sink.is_some()
                || spec.membership_sink.is_some()
                || spec.array_selector.is_some();

            if participates {
                assert_ne!(
                    demand_law(id),
                    BuiltinDemandLaw::Identity,
                    "{method:?} is exposed to planning/execution surfaces but inherits Identity demand"
                );
            }
        }
    }

    #[test]
    fn registry_order_transform_demands_are_conservative_barriers() {
        let downstream = Demand {
            pull: PullDemand::LastInput(1),
            value: ValueNeed::Whole,
            order: true,
        };

        for method in [
            BuiltinMethod::Lag,
            BuiltinMethod::Lead,
            BuiltinMethod::DiffWindow,
            BuiltinMethod::PctChange,
            BuiltinMethod::CumMax,
            BuiltinMethod::CumMin,
            BuiltinMethod::Zscore,
        ] {
            let id = BuiltinId::from_method(method);
            assert!(demand_is_conservative_barrier(id), "{method:?}");
            let demand = propagate_demand(id, BuiltinDemandArg::None, downstream);
            assert_eq!(demand.pull, PullDemand::All, "{method:?}");
            assert_eq!(demand.value, ValueNeed::Whole, "{method:?}");
            assert!(demand.order, "{method:?}");
        }
    }

    #[test]
    fn registry_relational_barriers_request_all_input() {
        let downstream = Demand {
            pull: PullDemand::LastInput(1),
            value: ValueNeed::Whole,
            order: true,
        };
        let id = BuiltinId::from_method(BuiltinMethod::EquiJoin);

        assert!(demand_is_conservative_barrier(id));
        let demand = propagate_demand(id, BuiltinDemandArg::None, downstream);
        assert_eq!(demand.pull, PullDemand::All);
        assert_eq!(demand.value, ValueNeed::Whole);
        assert!(demand.order);
    }

    #[test]
    fn registry_view_scalar_projection_matches_json_view_dispatch() {
        for (method, _, _) in all_method_entries() {
            assert_eq!(
                method.spec().view_scalar,
                view_scalar_projection(BuiltinId::from_method(method)),
                "{method:?}"
            );
            assert_eq!(
                method.spec().view_scalar_op,
                view_scalar_op(BuiltinId::from_method(method)),
                "{method:?}"
            );
        }
    }

    #[test]
    fn registry_view_scalar_projection_contracts_are_exhaustive() {
        let expected = [
            BuiltinMethod::Abs,
            BuiltinMethod::ByteLen,
            BuiltinMethod::Ceil,
            BuiltinMethod::ContainsAll,
            BuiltinMethod::ContainsAny,
            BuiltinMethod::EndsWith,
            BuiltinMethod::Floor,
            BuiltinMethod::Includes,
            BuiltinMethod::IndexOf,
            BuiltinMethod::IsAlpha,
            BuiltinMethod::IsAscii,
            BuiltinMethod::IsBlank,
            BuiltinMethod::IsNumeric,
            BuiltinMethod::LastIndexOf,
            BuiltinMethod::Len,
            BuiltinMethod::Lower,
            BuiltinMethod::Matches,
            BuiltinMethod::ParseBool,
            BuiltinMethod::ParseFloat,
            BuiltinMethod::ParseInt,
            BuiltinMethod::Round,
            BuiltinMethod::StartsWith,
            BuiltinMethod::ToBool,
            BuiltinMethod::ToNumber,
            BuiltinMethod::Trim,
            BuiltinMethod::TrimLeft,
            BuiltinMethod::TrimRight,
            BuiltinMethod::Type,
            BuiltinMethod::Upper,
        ];

        let registered: Vec<_> = all_method_entries()
            .into_iter()
            .filter_map(|(method, _, _)| {
                view_scalar_projection(BuiltinId::from_method(method)).then_some(method)
            })
            .collect();
        assert_eq!(registered, expected);

        for method in expected {
            let id = BuiltinId::from_method(method);
            let spec = method.spec();
            assert!(spec.view_native, "{method:?}");
            assert!(spec.view_scalar_op.is_some(), "{method:?}");
            if pipeline_element(id) {
                assert_eq!(
                    effective_pipeline_order_effect(id, true),
                    BuiltinPipelineOrderEffect::Preserves,
                    "{method:?}"
                );
            }
            assert!(
                matches!(
                    demand_law(id),
                    BuiltinDemandLaw::MapLike
                        | BuiltinDemandLaw::PredicateMapLike
                        | BuiltinDemandLaw::Count
                ),
                "{method:?}"
            );
            assert!(
                pipeline_element(id)
                    || matches!(method, BuiltinMethod::Len | BuiltinMethod::Includes),
                "{method:?}"
            );
        }

        for (method, op) in [
            (BuiltinMethod::Len, BuiltinViewScalarOp::Len),
            (BuiltinMethod::Type, BuiltinViewScalarOp::TypeName),
            (
                BuiltinMethod::Includes,
                BuiltinViewScalarOp::StringContainsArg,
            ),
            (BuiltinMethod::StartsWith, BuiltinViewScalarOp::StringArg),
            (BuiltinMethod::EndsWith, BuiltinViewScalarOp::StringArg),
            (BuiltinMethod::Matches, BuiltinViewScalarOp::StringArg),
            (BuiltinMethod::IndexOf, BuiltinViewScalarOp::StringArg),
            (BuiltinMethod::LastIndexOf, BuiltinViewScalarOp::StringArg),
            (
                BuiltinMethod::ContainsAny,
                BuiltinViewScalarOp::StringVecArg,
            ),
            (
                BuiltinMethod::ContainsAll,
                BuiltinViewScalarOp::StringVecArg,
            ),
            (BuiltinMethod::Ceil, BuiltinViewScalarOp::NumericNoArg),
            (BuiltinMethod::Floor, BuiltinViewScalarOp::NumericNoArg),
            (BuiltinMethod::Round, BuiltinViewScalarOp::NumericNoArg),
            (BuiltinMethod::Abs, BuiltinViewScalarOp::NumericNoArg),
            (BuiltinMethod::Upper, BuiltinViewScalarOp::StringNoArg),
            (BuiltinMethod::Lower, BuiltinViewScalarOp::StringNoArg),
            (BuiltinMethod::Trim, BuiltinViewScalarOp::StringNoArg),
            (BuiltinMethod::TrimLeft, BuiltinViewScalarOp::StringNoArg),
            (BuiltinMethod::TrimRight, BuiltinViewScalarOp::StringNoArg),
            (BuiltinMethod::ByteLen, BuiltinViewScalarOp::StringNoArg),
            (BuiltinMethod::IsBlank, BuiltinViewScalarOp::StringNoArg),
            (BuiltinMethod::IsNumeric, BuiltinViewScalarOp::StringNoArg),
            (BuiltinMethod::IsAlpha, BuiltinViewScalarOp::StringNoArg),
            (BuiltinMethod::IsAscii, BuiltinViewScalarOp::StringNoArg),
            (BuiltinMethod::ToNumber, BuiltinViewScalarOp::StringNoArg),
            (BuiltinMethod::ToBool, BuiltinViewScalarOp::StringNoArg),
            (BuiltinMethod::ParseInt, BuiltinViewScalarOp::ParseInt),
            (BuiltinMethod::ParseFloat, BuiltinViewScalarOp::StringNoArg),
            (BuiltinMethod::ParseBool, BuiltinViewScalarOp::StringNoArg),
        ] {
            assert_eq!(
                view_scalar_op(BuiltinId::from_method(method)),
                Some(op),
                "{method:?}"
            );
        }
    }

    #[test]
    fn registry_raw_json_scalar_contracts_are_exhaustive() {
        let expected = [
            (BuiltinMethod::Len, BuiltinRawJsonScalar::Len),
            (BuiltinMethod::Lower, BuiltinRawJsonScalar::AsciiLower),
            (BuiltinMethod::Upper, BuiltinRawJsonScalar::AsciiUpper),
        ];

        let registered: Vec<_> = all_method_entries()
            .into_iter()
            .filter_map(|(method, _, _)| {
                raw_json_scalar(BuiltinId::from_method(method), &BuiltinArgs::None)
                    .map(|scalar| (method, scalar))
            })
            .collect();
        assert_eq!(registered, expected);

        for (method, scalar) in expected {
            let id = BuiltinId::from_method(method);
            let spec = method.spec();
            assert_eq!(spec.raw_json_scalar, Some(scalar), "{method:?}");
            assert!(raw_json_scalar_call(id, &BuiltinArgs::None), "{method:?}");
            assert_eq!(demand_law(id), scalar.demand_law(), "{method:?}");
            assert_eq!(
                scalar.writes_string(),
                !matches!(scalar, BuiltinRawJsonScalar::Len),
                "{method:?}"
            );
            assert_eq!(
                scalar.writes_view_len(),
                matches!(scalar, BuiltinRawJsonScalar::Len),
                "{method:?}"
            );
            assert!(view_scalar_projection(id), "{method:?}");
            assert!(spec.view_native, "{method:?}");
            assert_eq!(
                raw_json_scalar(id, &BuiltinArgs::Str(Arc::from("x"))),
                None,
                "{method:?}"
            );
            assert!(
                !raw_json_scalar_call(id, &BuiltinArgs::Str(Arc::from("x"))),
                "{method:?}"
            );
        }

        assert!(!raw_json_scalar_call(
            BuiltinId::from_method(BuiltinMethod::Trim),
            &BuiltinArgs::None
        ));
    }

    #[test]
    fn registry_classifies_direct_view_scalar_value_projection() {
        for (method, _, _) in all_method_entries() {
            let id = BuiltinId::from_method(method);
            assert_eq!(
                view_scalar_value_projection(id),
                view_scalar_projection(id) && !view_object_items_projection(id),
                "{method:?}"
            );
        }

        assert!(view_scalar_value_projection(BuiltinId::from_method(
            BuiltinMethod::Len
        )));
        assert!(!view_scalar_value_projection(BuiltinId::from_method(
            BuiltinMethod::Keys
        )));
        assert!(!view_scalar_value_projection(BuiltinId::from_method(
            BuiltinMethod::Entries
        )));
    }

    #[test]
    fn registry_classifies_direct_view_call_family() {
        assert_eq!(
            direct_view_call(
                BuiltinId::from_method(BuiltinMethod::Len),
                &BuiltinArgs::None
            ),
            Some(BuiltinDirectViewCall::ScalarValue)
        );
        assert_eq!(
            direct_view_call(
                BuiltinId::from_method(BuiltinMethod::Len),
                &BuiltinArgs::Str(Arc::from("x"))
            ),
            Some(BuiltinDirectViewCall::ScalarValue)
        );
        assert_eq!(
            direct_view_call(
                BuiltinId::from_method(BuiltinMethod::Entries),
                &BuiltinArgs::None
            ),
            Some(BuiltinDirectViewCall::ObjectItems(
                BuiltinViewObjectProjection::Entries
            ))
        );
        assert_eq!(
            direct_view_call(
                BuiltinId::from_method(BuiltinMethod::ToPairs),
                &BuiltinArgs::None
            ),
            Some(BuiltinDirectViewCall::ObjectItems(
                BuiltinViewObjectProjection::ToPairs
            ))
        );
        assert_eq!(
            direct_view_call(
                BuiltinId::from_method(BuiltinMethod::Entries),
                &BuiltinArgs::Str(Arc::from("x"))
            ),
            None
        );
    }

    #[test]
    fn registry_direct_view_call_contracts_are_safe() {
        for (method, _, _) in all_method_entries() {
            let id = BuiltinId::from_method(method);
            let spec = method.spec();

            for args in [
                BuiltinArgs::None,
                BuiltinArgs::Str(Arc::from("x")),
                BuiltinArgs::StrVec(vec![Arc::from("x"), Arc::from("y")]),
            ] {
                let Some(call) = direct_view_call(id, &args) else {
                    continue;
                };

                assert!(spec.pure, "{method:?} direct view calls must be pure");
                assert!(
                    spec.view_native,
                    "{method:?} direct view calls must be view-native"
                );
                assert_eq!(
                    pipeline_materialization(id),
                    BuiltinPipelineMaterialization::Streaming,
                    "{method:?} direct view calls must not force materialization"
                );

                match call {
                    BuiltinDirectViewCall::ScalarValue => {
                        assert!(
                            view_scalar_value_projection(id),
                            "{method:?} scalar direct view call must be a scalar value projection"
                        );
                        assert!(
                            view_projection_returns_owned(id, &args),
                            "{method:?} scalar direct view calls must return owned values"
                        );
                        assert!(
                            !view_object_items_projection(id),
                            "{method:?} scalar direct view calls must not enumerate object items"
                        );
                    }
                    BuiltinDirectViewCall::ObjectItems(projection) => {
                        assert!(
                            matches!(args, BuiltinArgs::None),
                            "{method:?} object item direct view calls must be nullary"
                        );
                        assert_eq!(
                            view_object_projection(id),
                            Some(projection),
                            "{method:?} object item direct view call must match object projection metadata"
                        );
                        assert!(
                            projection.is_item_projection(),
                            "{method:?} direct object call must enumerate object items"
                        );
                        assert!(
                            view_projection_returns_owned(id, &args),
                            "{method:?} object item direct view calls must return owned values"
                        );
                        assert!(
                            !view_scalar_value_projection(id),
                            "{method:?} object item direct view calls must not be scalar value projections"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn registry_view_capabilities_are_consistent() {
        for (method, _, _) in all_method_entries() {
            let id = BuiltinId::from_method(method);
            let spec = method.spec();
            assert_eq!(
                view_projection(id),
                spec.view_scalar
                    || spec.view_value_projection.is_some()
                    || spec.view_object_projection.is_some(),
                "{method:?}"
            );
            if spec.view_scalar {
                assert!(
                    spec.view_native,
                    "{method:?} view_scalar must be view_native"
                );
                assert!(
                    spec.view_scalar_op.is_some(),
                    "{method:?} view_scalar must declare a dispatch op"
                );
            }
            if spec.view_scalar_op.is_some() {
                assert!(
                    spec.view_scalar,
                    "{method:?} view_scalar_op implies view_scalar"
                );
                assert!(
                    spec.view_native,
                    "{method:?} view_scalar_op implies view_native"
                );
            }
            if spec.view_object_projection.is_some() {
                assert!(
                    spec.view_native,
                    "{method:?} view_object_projection must be view_native"
                );
            }
            if spec.view_value_projection.is_some() {
                assert!(
                    spec.view_native,
                    "{method:?} view_value_projection must be view_native"
                );
            }
            if spec.raw_json_scalar.is_some() {
                assert!(
                    spec.view_scalar,
                    "{method:?} raw_json_scalar must also be a view scalar"
                );
                assert!(
                    spec.view_native,
                    "{method:?} raw_json_scalar must be view_native"
                );
            }
        }
    }

    #[test]
    fn registry_view_object_projection_demands_are_classified() {
        for (method, _, _) in all_method_entries() {
            let id = BuiltinId::from_method(method);
            let Some(projection) = view_object_projection(id) else {
                continue;
            };
            assert_eq!(demand_law(id), projection.demand_law(), "{method:?}");
        }
    }

    #[test]
    fn registry_identifies_object_key_only_view_projections() {
        assert!(view_projection_needs_only_object_keys(
            BuiltinId::from_method(BuiltinMethod::Len),
            &BuiltinArgs::None
        ));
        assert!(view_projection_needs_only_object_keys(
            BuiltinId::from_method(BuiltinMethod::Keys),
            &BuiltinArgs::None
        ));
        assert!(view_projection_needs_only_object_keys(
            BuiltinId::from_method(BuiltinMethod::HasKey),
            &BuiltinArgs::Str(Arc::from("open"))
        ));
        assert!(view_projection_needs_only_object_keys(
            BuiltinId::from_method(BuiltinMethod::HasAll),
            &BuiltinArgs::StrVec(vec![Arc::from("open"), Arc::from("closed")])
        ));
        assert!(!view_projection_needs_only_object_keys(
            BuiltinId::from_method(BuiltinMethod::Missing),
            &BuiltinArgs::Str(Arc::from("open"))
        ));
        assert!(!view_projection_needs_only_object_keys(
            BuiltinId::from_method(BuiltinMethod::Values),
            &BuiltinArgs::None
        ));
        assert!(!view_projection_needs_only_object_keys(
            BuiltinId::from_method(BuiltinMethod::Pick),
            &BuiltinArgs::StrVec(vec![Arc::from("open")])
        ));
    }

    #[test]
    fn registry_drives_view_projection_result_paths() {
        let receiver = vec![Arc::from("profile")];
        let result = view_projection_result_field_path(
            BuiltinId::from_method(BuiltinMethod::GetPath),
            &BuiltinArgs::Str(Arc::from("author.name")),
            &receiver,
        )
        .expect("get_path result path");
        assert_eq!(
            result.iter().map(|key| key.as_ref()).collect::<Vec<_>>(),
            vec!["profile", "author", "name"]
        );

        assert!(view_projection_result_field_path(
            BuiltinId::from_method(BuiltinMethod::GetPath),
            &BuiltinArgs::Str(Arc::from("items[0].price")),
            &receiver,
        )
        .is_none());
        assert!(view_projection_result_field_path(
            BuiltinId::from_method(BuiltinMethod::HasKey),
            &BuiltinArgs::Str(Arc::from("author")),
            &receiver,
        )
        .is_none());
    }

    #[test]
    fn registry_view_value_projection_contracts_are_exhaustive() {
        let expected = [
            (
                BuiltinMethod::CamelCase,
                BuiltinViewValueProjection::CamelCase,
            ),
            (
                BuiltinMethod::Capitalize,
                BuiltinViewValueProjection::Capitalize,
            ),
            (BuiltinMethod::Center, BuiltinViewValueProjection::Center),
            (BuiltinMethod::Dedent, BuiltinViewValueProjection::Dedent),
            (
                BuiltinMethod::FromBase64,
                BuiltinViewValueProjection::FromBase64,
            ),
            (BuiltinMethod::FromPairs, BuiltinViewValueProjection::FromPairs),
            (
                BuiltinMethod::HtmlEscape,
                BuiltinViewValueProjection::HtmlEscape,
            ),
            (
                BuiltinMethod::HtmlUnescape,
                BuiltinViewValueProjection::HtmlUnescape,
            ),
            (
                BuiltinMethod::Includes,
                BuiltinViewValueProjection::Includes,
            ),
            (BuiltinMethod::Indent, BuiltinViewValueProjection::Indent),
            (BuiltinMethod::Invert, BuiltinViewValueProjection::Invert),
            (
                BuiltinMethod::KebabCase,
                BuiltinViewValueProjection::KebabCase,
            ),
            (BuiltinMethod::Or, BuiltinViewValueProjection::Or),
            (BuiltinMethod::PadLeft, BuiltinViewValueProjection::PadLeft),
            (
                BuiltinMethod::PadRight,
                BuiltinViewValueProjection::PadRight,
            ),
            (
                BuiltinMethod::PascalCase,
                BuiltinViewValueProjection::PascalCase,
            ),
            (BuiltinMethod::Repeat, BuiltinViewValueProjection::Repeat),
            (BuiltinMethod::Replace, BuiltinViewValueProjection::Replace),
            (
                BuiltinMethod::ReplaceAll,
                BuiltinViewValueProjection::ReplaceAll,
            ),
            (
                BuiltinMethod::ReverseStr,
                BuiltinViewValueProjection::ReverseStr,
            ),
            (BuiltinMethod::Slice, BuiltinViewValueProjection::Slice),
            (
                BuiltinMethod::SnakeCase,
                BuiltinViewValueProjection::SnakeCase,
            ),
            (
                BuiltinMethod::StripPrefix,
                BuiltinViewValueProjection::StripPrefix,
            ),
            (
                BuiltinMethod::StripSuffix,
                BuiltinViewValueProjection::StripSuffix,
            ),
            (
                BuiltinMethod::TitleCase,
                BuiltinViewValueProjection::TitleCase,
            ),
            (
                BuiltinMethod::ToBase64,
                BuiltinViewValueProjection::ToBase64,
            ),
            (BuiltinMethod::ToJson, BuiltinViewValueProjection::ToJson),
            (
                BuiltinMethod::ToString,
                BuiltinViewValueProjection::ToString,
            ),
            (
                BuiltinMethod::UrlDecode,
                BuiltinViewValueProjection::UrlDecode,
            ),
            (
                BuiltinMethod::UrlEncode,
                BuiltinViewValueProjection::UrlEncode,
            ),
        ];

        let registered: Vec<_> = all_method_entries()
            .into_iter()
            .filter_map(|(method, _, _)| {
                view_value_projection(BuiltinId::from_method(method))
                    .map(|projection| (method, projection))
            })
            .collect();
        assert_eq!(registered, expected);

        for (method, projection) in expected {
            let id = BuiltinId::from_method(method);
            let spec = method.spec();
            assert!(spec.view_native, "{method:?}");
            if method != BuiltinMethod::Includes {
                assert!(!spec.view_scalar, "{method:?}");
            }
            assert_eq!(view_value_projection(id), Some(projection), "{method:?}");
            assert!(view_projection(id), "{method:?}");
            assert!(
                view_projection_returns_owned(id, &BuiltinArgs::None),
                "{method:?}"
            );
            assert_eq!(demand_law(id), projection.demand_law(), "{method:?}");
            assert_eq!(
                effective_pipeline_order_effect(id, true),
                BuiltinPipelineOrderEffect::Preserves,
                "{method:?}"
            );
        }
    }

    #[test]
    fn registry_view_object_projection_contracts_are_exhaustive() {
        let expected = [
            (BuiltinMethod::Entries, BuiltinViewObjectProjection::Entries),
            (BuiltinMethod::GetPath, BuiltinViewObjectProjection::GetPath),
            (BuiltinMethod::Has, BuiltinViewObjectProjection::Has),
            (BuiltinMethod::HasAll, BuiltinViewObjectProjection::HasAll),
            (BuiltinMethod::HasKey, BuiltinViewObjectProjection::HasKey),
            (BuiltinMethod::HasPath, BuiltinViewObjectProjection::HasPath),
            (BuiltinMethod::Keys, BuiltinViewObjectProjection::Keys),
            (BuiltinMethod::Missing, BuiltinViewObjectProjection::Missing),
            (BuiltinMethod::Omit, BuiltinViewObjectProjection::Omit),
            (BuiltinMethod::Pick, BuiltinViewObjectProjection::Pick),
            (BuiltinMethod::ToPairs, BuiltinViewObjectProjection::ToPairs),
            (BuiltinMethod::Values, BuiltinViewObjectProjection::Values),
        ];

        let registered: Vec<_> = all_method_entries()
            .into_iter()
            .filter_map(|(method, _, _)| {
                view_object_projection(BuiltinId::from_method(method))
                    .map(|projection| (method, projection))
            })
            .collect();
        assert_eq!(registered, expected);

        for (method, projection) in expected {
            let id = BuiltinId::from_method(method);
            let spec = method.spec();
            assert!(spec.view_native, "{method:?}");
            assert_eq!(
                effective_pipeline_order_effect(id, true),
                BuiltinPipelineOrderEffect::Preserves,
                "{method:?}"
            );
            assert_eq!(view_object_projection(id), Some(projection), "{method:?}");
            assert!(view_projection(id), "{method:?}");
            assert_eq!(
                view_object_items_projection(id),
                projection.is_item_projection(),
                "{method:?}"
            );

            match projection {
                BuiltinViewObjectProjection::GetPath => {
                    let args = BuiltinArgs::Str(Arc::from("nested.x"));
                    assert_eq!(
                        view_projection_returns_owned(id, &args),
                        projection.returns_owned(),
                        "{method:?}"
                    );
                    assert!(
                        view_projection_field_demand(id, &args).is_some(),
                        "{method:?}"
                    );
                }
                BuiltinViewObjectProjection::HasPath => {
                    let args = BuiltinArgs::Str(Arc::from("nested.x"));
                    assert_eq!(
                        view_projection_returns_owned(id, &args),
                        projection.returns_owned(),
                        "{method:?}"
                    );
                    assert!(
                        view_projection_field_demand(id, &args).is_some(),
                        "{method:?}"
                    );
                }
                BuiltinViewObjectProjection::HasKey => {
                    let args = BuiltinArgs::Str(Arc::from("isbn"));
                    assert_eq!(
                        view_projection_returns_owned(id, &args),
                        projection.returns_owned(),
                        "{method:?}"
                    );
                    assert!(
                        view_projection_field_demand(id, &args).is_some(),
                        "{method:?}"
                    );
                }
                BuiltinViewObjectProjection::Missing => {
                    let args = BuiltinArgs::StrVec(vec![Arc::from("isbn"), Arc::from("title")]);
                    assert_eq!(
                        view_projection_returns_owned(id, &args),
                        projection.returns_owned(),
                        "{method:?}"
                    );
                    assert!(
                        view_projection_field_demand(id, &args).is_some(),
                        "{method:?}"
                    );
                }
                BuiltinViewObjectProjection::Pick => {
                    let args = BuiltinArgs::StrVec(vec![Arc::from("isbn"), Arc::from("title")]);
                    assert_eq!(
                        view_projection_returns_owned(id, &args),
                        projection.returns_owned(),
                        "{method:?}"
                    );
                    assert!(
                        view_projection_field_demand(id, &args).is_some(),
                        "{method:?}"
                    );
                }
                BuiltinViewObjectProjection::Has
                | BuiltinViewObjectProjection::HasAll
                | BuiltinViewObjectProjection::Keys
                | BuiltinViewObjectProjection::Values
                | BuiltinViewObjectProjection::Entries
                | BuiltinViewObjectProjection::ToPairs
                | BuiltinViewObjectProjection::Omit => {
                    assert_eq!(
                        view_projection_returns_owned(id, &BuiltinArgs::None),
                        projection.returns_owned(),
                        "{method:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn registry_view_object_item_projection_calls_are_nullary_only() {
        for (method, _, _) in all_method_entries() {
            let id = BuiltinId::from_method(method);
            let item_projection =
                view_object_projection(id).filter(|projection| projection.is_item_projection());

            assert_eq!(
                view_object_items_projection_call(id, &BuiltinArgs::None),
                item_projection,
                "{method:?}"
            );
            assert_eq!(
                view_object_items_projection_call(id, &BuiltinArgs::Str(Arc::from("x"))),
                None,
                "{method:?}"
            );

            if item_projection.is_some() {
                assert_eq!(
                    pipeline_lowering(id),
                    Some(BuiltinPipelineLowering::Nullary),
                    "{method:?} object item projection must lower as a nullary call"
                );
                assert!(
                    !view_scalar_value_projection(id),
                    "{method:?} object item projection must not be treated as scalar value projection"
                );
            }
        }
    }

    fn field_paths(demand: Option<FieldDemand>) -> Vec<String> {
        match demand.expect("field demand") {
            FieldDemand::None => Vec::new(),
            FieldDemand::Whole => vec!["*".to_string()],
            FieldDemand::Fields(fields) => fields
                .paths()
                .iter()
                .map(|path| {
                    path.keys()
                        .iter()
                        .map(|key| key.as_ref())
                        .collect::<Vec<_>>()
                        .join(".")
                })
                .collect(),
        }
    }

    #[test]
    fn registry_drives_view_projection_field_demands() {
        assert_eq!(
            field_paths(view_projection_field_demand(
                BuiltinId::from_method(BuiltinMethod::HasKey),
                &BuiltinArgs::Str(Arc::from("isbn"))
            )),
            vec!["isbn"]
        );
        assert_eq!(
            field_paths(view_projection_field_demand(
                BuiltinId::from_method(BuiltinMethod::Missing),
                &BuiltinArgs::StrVec(vec![Arc::from("title"), Arc::from("isbn")])
            )),
            vec!["title", "isbn"]
        );
        assert_eq!(
            field_paths(view_projection_field_demand(
                BuiltinId::from_method(BuiltinMethod::Missing),
                &BuiltinArgs::Str(Arc::from("meta.author.name"))
            )),
            vec!["meta.author.name"]
        );
        assert_eq!(
            field_paths(view_projection_field_demand(
                BuiltinId::from_method(BuiltinMethod::Missing),
                &BuiltinArgs::StrVec(vec![
                    Arc::from("meta.author.name"),
                    Arc::from("items[0].sku")
                ])
            )),
            vec!["meta.author.name", "items"]
        );
        assert_eq!(
            field_paths(view_projection_field_demand(
                BuiltinId::from_method(BuiltinMethod::Pick),
                &BuiltinArgs::StrVec(vec![Arc::from("title"), Arc::from("isbn")])
            )),
            vec!["title", "isbn"]
        );
        assert_eq!(
            field_paths(view_projection_field_demand(
                BuiltinId::from_method(BuiltinMethod::HasPath),
                &BuiltinArgs::Str(Arc::from("user.name"))
            )),
            vec!["user.name"]
        );
        assert_eq!(
            field_paths(view_projection_field_demand(
                BuiltinId::from_method(BuiltinMethod::GetPath),
                &BuiltinArgs::Str(Arc::from("items[0].price"))
            )),
            vec!["items"]
        );
        assert!(view_projection_field_demand(
            BuiltinId::from_method(BuiltinMethod::Has),
            &BuiltinArgs::Str(Arc::from("isbn"))
        )
        .is_none());
        assert_eq!(
            field_paths(view_projection_receiver_field_demand(
                BuiltinId::from_method(BuiltinMethod::HasKey),
                &BuiltinArgs::Str(Arc::from("isbn")),
                &[Arc::from("book")]
            )),
            vec!["book.isbn"]
        );
        assert_eq!(
            field_paths(view_projection_receiver_field_demand(
                BuiltinId::from_method(BuiltinMethod::GetPath),
                &BuiltinArgs::Str(Arc::from("user.name")),
                &[Arc::from("event")]
            )),
            vec!["event.user.name"]
        );
    }

    #[test]
    fn registry_marks_one_to_one_element_demands() {
        let downstream = Demand {
            pull: PullDemand::LastInput(1),
            value: ValueNeed::Whole,
            order: true,
        };

        for method in [
            BuiltinMethod::TransformKeys,
            BuiltinMethod::TransformValues,
            BuiltinMethod::FilterKeys,
            BuiltinMethod::FilterValues,
            BuiltinMethod::GetPath,
            BuiltinMethod::Pick,
            BuiltinMethod::Omit,
            BuiltinMethod::Keys,
            BuiltinMethod::Values,
            BuiltinMethod::Entries,
            BuiltinMethod::ToPairs,
            BuiltinMethod::FromPairs,
            BuiltinMethod::Invert,
            BuiltinMethod::Merge,
            BuiltinMethod::DeepMerge,
            BuiltinMethod::Defaults,
            BuiltinMethod::Rename,
            BuiltinMethod::Pivot,
            BuiltinMethod::Implode,
            BuiltinMethod::FromJson,
            BuiltinMethod::Replace,
            BuiltinMethod::ReplaceAll,
            BuiltinMethod::ToCsv,
            BuiltinMethod::ToTsv,
            BuiltinMethod::SetPath,
            BuiltinMethod::DelPaths,
            BuiltinMethod::FlattenKeys,
            BuiltinMethod::UnflattenKeys,
        ] {
            let demand = propagate_demand(
                BuiltinId::from_method(method),
                BuiltinDemandArg::None,
                downstream,
            );
            assert_eq!(demand.pull, PullDemand::LastInput(1), "{method:?}");
            assert_eq!(demand.value, ValueNeed::Whole, "{method:?}");
            assert!(demand.order, "{method:?}");
        }

        for method in [
            BuiltinMethod::Has,
            BuiltinMethod::HasAll,
            BuiltinMethod::HasKey,
            BuiltinMethod::Missing,
            BuiltinMethod::HasPath,
            BuiltinMethod::IsBlank,
            BuiltinMethod::IsNumeric,
            BuiltinMethod::IsAlpha,
            BuiltinMethod::IsAscii,
            BuiltinMethod::StartsWith,
            BuiltinMethod::EndsWith,
            BuiltinMethod::Matches,
            BuiltinMethod::ContainsAny,
            BuiltinMethod::ContainsAll,
            BuiltinMethod::Includes,
            BuiltinMethod::Index,
            BuiltinMethod::IndicesOf,
        ] {
            let demand = propagate_demand(
                BuiltinId::from_method(method),
                BuiltinDemandArg::None,
                downstream,
            );
            assert_eq!(demand.pull, PullDemand::LastInput(1), "{method:?}");
            assert_eq!(demand.value, ValueNeed::Predicate, "{method:?}");
            assert!(demand.order, "{method:?}");
        }
    }

    #[test]
    fn registry_marks_idempotent_builtins() {
        for method in [
            BuiltinMethod::Upper,
            BuiltinMethod::Lower,
            BuiltinMethod::Trim,
            BuiltinMethod::TrimLeft,
            BuiltinMethod::TrimRight,
            BuiltinMethod::Capitalize,
            BuiltinMethod::TitleCase,
            BuiltinMethod::SnakeCase,
            BuiltinMethod::KebabCase,
            BuiltinMethod::CamelCase,
            BuiltinMethod::PascalCase,
            BuiltinMethod::Dedent,
            BuiltinMethod::Sort,
            BuiltinMethod::Unique,
        ] {
            assert!(is_idempotent(BuiltinId::from_method(method)), "{method:?}");
        }

        for method in [
            BuiltinMethod::Replace,
            BuiltinMethod::Append,
            BuiltinMethod::Reverse,
            BuiltinMethod::ParseInt,
        ] {
            assert!(!is_idempotent(BuiltinId::from_method(method)), "{method:?}");
        }
    }

    #[test]
    fn registry_marks_lambda_arg_builtins() {
        for method in [
            BuiltinMethod::Filter,
            BuiltinMethod::Map,
            BuiltinMethod::FlatMap,
            BuiltinMethod::Sort,
            BuiltinMethod::Any,
            BuiltinMethod::All,
            BuiltinMethod::Count,
            BuiltinMethod::Find,
            BuiltinMethod::FindAll,
            BuiltinMethod::FindFirst,
            BuiltinMethod::FindIndex,
            BuiltinMethod::GroupBy,
            BuiltinMethod::CountBy,
            BuiltinMethod::IndexBy,
            BuiltinMethod::TakeWhile,
            BuiltinMethod::DropWhile,
            BuiltinMethod::MaxBy,
            BuiltinMethod::MinBy,
            BuiltinMethod::Accumulate,
            BuiltinMethod::Fold,
            BuiltinMethod::Partition,
            BuiltinMethod::TransformKeys,
            BuiltinMethod::TransformValues,
            BuiltinMethod::FilterKeys,
            BuiltinMethod::FilterValues,
            BuiltinMethod::Pivot,
            BuiltinMethod::Update,
        ] {
            assert!(
                accepts_lambda_arg(BuiltinId::from_method(method)),
                "{method:?}"
            );
        }

        for method in [
            BuiltinMethod::Upper,
            BuiltinMethod::Take,
            BuiltinMethod::Len,
            BuiltinMethod::HasKey,
        ] {
            assert!(
                !accepts_lambda_arg(BuiltinId::from_method(method)),
                "{method:?}"
            );
        }
    }

    #[test]
    fn registry_marks_conservative_demand_barriers() {
        for method in [
            BuiltinMethod::FlatMap,
            BuiltinMethod::DropWhile,
            BuiltinMethod::Sort,
            BuiltinMethod::DeepFind,
            BuiltinMethod::DeepShape,
            BuiltinMethod::DeepLike,
            BuiltinMethod::EquiJoin,
            BuiltinMethod::Unknown,
        ] {
            assert!(
                demand_is_conservative_barrier(BuiltinId::from_method(method)),
                "{method:?}"
            );
        }

        for method in [
            BuiltinMethod::Map,
            BuiltinMethod::Filter,
            BuiltinMethod::Take,
            BuiltinMethod::Last,
        ] {
            assert!(
                !demand_is_conservative_barrier(BuiltinId::from_method(method)),
                "{method:?}"
            );
        }
    }

    #[test]
    fn registry_preserves_order_for_positional_selective_and_expanding_demands() {
        let positional = Demand {
            pull: PullDemand::FirstInput(1),
            value: ValueNeed::Whole,
            order: false,
        };

        for method in [
            BuiltinMethod::Filter,
            BuiltinMethod::Remove,
            BuiltinMethod::Compact,
            BuiltinMethod::Unique,
            BuiltinMethod::UniqueBy,
            BuiltinMethod::FlatMap,
            BuiltinMethod::Flatten,
            BuiltinMethod::Explode,
        ] {
            let demand = propagate_demand(
                BuiltinId::from_method(method),
                BuiltinDemandArg::None,
                positional,
            );
            assert!(demand.order, "{method:?}");
        }
    }

    #[test]
    fn registry_classifies_order_preserving_filters() {
        for method in [
            BuiltinMethod::Filter,
            BuiltinMethod::Remove,
            BuiltinMethod::Compact,
            BuiltinMethod::TakeWhile,
            BuiltinMethod::DropWhile,
            BuiltinMethod::Unique,
            BuiltinMethod::UniqueBy,
        ] {
            let id = BuiltinId::from_method(method);
            assert_eq!(
                preserves_order_filtering(id),
                builtin_cardinality(id) == Some(BuiltinCardinality::Filtering)
                    && !matches!(
                        effective_pipeline_order_effect(id, false),
                        BuiltinPipelineOrderEffect::Blocks
                    ),
                "{method:?}"
            );
        }

        for method in [
            BuiltinMethod::Map,
            BuiltinMethod::Sort,
            BuiltinMethod::FlatMap,
        ] {
            assert!(
                !preserves_order_filtering(BuiltinId::from_method(method)),
                "{method:?}"
            );
        }
    }

    #[test]
    fn registry_drives_pipeline_execution_policy() {
        assert_eq!(
            pipeline_materialization(BuiltinId::from_method(BuiltinMethod::Sort)),
            BuiltinPipelineMaterialization::ComposedBarrier
        );
        assert_eq!(
            pipeline_materialization(BuiltinId::from_method(BuiltinMethod::Reverse)),
            BuiltinPipelineMaterialization::ComposedBarrier
        );
        for method in [
            BuiltinMethod::GroupBy,
            BuiltinMethod::CountBy,
            BuiltinMethod::IndexBy,
        ] {
            assert_eq!(
                pipeline_materialization(BuiltinId::from_method(method)),
                BuiltinPipelineMaterialization::ComposedBarrier,
                "{method:?}"
            );
        }
        assert_eq!(
            pipeline_materialization(BuiltinId::from_method(BuiltinMethod::Split)),
            BuiltinPipelineMaterialization::Streaming
        );
        assert_eq!(
            pipeline_materialization(BuiltinId::from_method(BuiltinMethod::TakeWhile)),
            BuiltinPipelineMaterialization::Streaming
        );
        assert!(pipeline_streams(BuiltinId::from_method(
            BuiltinMethod::TakeWhile
        )));
        assert!(pipeline_composed_barrier(BuiltinId::from_method(
            BuiltinMethod::Sort
        )));
        assert!(pipeline_streams(BuiltinId::from_method(
            BuiltinMethod::Split
        )));
        assert_eq!(
            streaming_boundary(BuiltinId::from_method(BuiltinMethod::Map)),
            BuiltinStreamingBoundary::RowLocal
        );
        assert_eq!(
            streaming_boundary(BuiltinId::from_method(BuiltinMethod::Rows)),
            BuiltinStreamingBoundary::SourceStream
        );
        assert_eq!(
            streaming_boundary(BuiltinId::from_method(BuiltinMethod::Take)),
            BuiltinStreamingBoundary::BoundedState
        );
        assert_eq!(
            streaming_boundary(BuiltinId::from_method(BuiltinMethod::DropWhile)),
            BuiltinStreamingBoundary::PrefixState
        );
        assert_eq!(
            streaming_boundary(BuiltinId::from_method(BuiltinMethod::Sort)),
            BuiltinStreamingBoundary::FullInputOrder
        );
        assert_eq!(
            streaming_boundary(BuiltinId::from_method(BuiltinMethod::GroupBy)),
            BuiltinStreamingBoundary::FullInputState
        );
        assert_eq!(
            streaming_boundary(BuiltinId::from_method(BuiltinMethod::FlatMap)),
            BuiltinStreamingBoundary::Expanding
        );
        assert_eq!(
            pipeline_shape(BuiltinId::from_method(BuiltinMethod::Split))
                .unwrap()
                .can_indexed,
            false
        );
        assert_eq!(
            pipeline_shape(BuiltinId::from_method(BuiltinMethod::Chunk))
                .unwrap()
                .cost,
            2.0
        );
        assert_eq!(
            pipeline_order_effect(BuiltinId::from_method(BuiltinMethod::Filter)),
            Some(BuiltinPipelineOrderEffect::PredicatePrefix)
        );
        assert_eq!(
            pipeline_order_effect(BuiltinId::from_method(BuiltinMethod::Replace)),
            Some(BuiltinPipelineOrderEffect::Preserves)
        );
        assert_eq!(
            effective_pipeline_order_effect(BuiltinId::from_method(BuiltinMethod::Replace), false),
            BuiltinPipelineOrderEffect::Preserves
        );
        assert_eq!(
            effective_pipeline_order_effect(BuiltinId::from_method(BuiltinMethod::HasKey), true),
            BuiltinPipelineOrderEffect::Preserves
        );
        assert_eq!(
            effective_pipeline_order_effect(BuiltinId::from_method(BuiltinMethod::Count), true),
            BuiltinPipelineOrderEffect::Blocks
        );
        assert!(pipeline_stage_is_positional(BuiltinId::from_method(
            BuiltinMethod::Take
        )));
        assert!(pipeline_stage_is_positional(BuiltinId::from_method(
            BuiltinMethod::Skip
        )));
        assert!(pipeline_stage_caps_input_prefix(BuiltinId::from_method(
            BuiltinMethod::Take
        )));
        assert!(!pipeline_stage_caps_input_prefix(BuiltinId::from_method(
            BuiltinMethod::Skip
        )));
        assert!(stage_elidable_before_sink(
            BuiltinId::from_method(BuiltinMethod::Map),
            BuiltinId::from_method(BuiltinMethod::Count)
        ));
        assert_eq!(
            terminal_selection_rewrite(
                BuiltinId::from_method(BuiltinMethod::Sort),
                BuiltinSelectionPosition::First
            ),
            Some(BuiltinMethod::Min)
        );
        assert_eq!(
            index_selection_rewrite(BuiltinId::from_method(BuiltinMethod::Sort), -1),
            Some(BuiltinMethod::Max)
        );
        assert_eq!(
            terminal_selection_rewrite(
                BuiltinId::from_method(BuiltinMethod::Reverse),
                BuiltinSelectionPosition::Last
            ),
            Some(BuiltinMethod::First)
        );
        assert!(stage_elidable_before_sink(
            BuiltinId::from_method(BuiltinMethod::Sort),
            BuiltinId::from_method(BuiltinMethod::Sum)
        ));
        assert!(!stage_elidable_before_sink(
            BuiltinId::from_method(BuiltinMethod::Sort),
            BuiltinId::from_method(BuiltinMethod::Last)
        ));
        assert!(!stage_elidable_before_sink(
            BuiltinId::from_method(BuiltinMethod::Sort),
            BuiltinId::from_method(BuiltinMethod::First)
        ));
        assert!(!pipeline_stage_is_positional(BuiltinId::from_method(
            BuiltinMethod::Filter
        )));
        assert!(pipeline_stage_is_order_only(BuiltinId::from_method(
            BuiltinMethod::Sort
        )));
        assert!(pipeline_stage_is_order_only(BuiltinId::from_method(
            BuiltinMethod::Reverse
        )));
        assert!(!pipeline_stage_is_order_only(BuiltinId::from_method(
            BuiltinMethod::Append
        )));
        assert!(pipeline_chain_operator(BuiltinId::from_method(
            BuiltinMethod::Map
        )));
        assert!(pipeline_chain_operator(BuiltinId::from_method(
            BuiltinMethod::DeepFind
        )));
        assert!(!pipeline_chain_operator(BuiltinId::from_method(
            BuiltinMethod::Upper
        )));
    }

    #[test]
    fn registry_streaming_boundaries_are_coherent_with_materialization() {
        for (method, _, _) in all_method_entries() {
            let id = BuiltinId::from_method(method);
            let boundary = streaming_boundary(id);
            let materialization = pipeline_materialization(id);
            match boundary {
                BuiltinStreamingBoundary::RowLocal | BuiltinStreamingBoundary::SourceStream => {
                    assert_eq!(
                        materialization,
                        BuiltinPipelineMaterialization::Streaming,
                        "{method:?} row-local/source boundaries must not force materialization"
                    );
                }
                BuiltinStreamingBoundary::BoundedState => {
                    assert!(
                        matches!(
                            demand_law(id),
                            BuiltinDemandLaw::Take
                                | BuiltinDemandLaw::Skip
                                | BuiltinDemandLaw::TakeWhile
                                | BuiltinDemandLaw::Window
                                | BuiltinDemandLaw::Chunk
                                | BuiltinDemandLaw::First
                                | BuiltinDemandLaw::Last
                                | BuiltinDemandLaw::Nth
                        ),
                        "{method:?} bounded-state boundaries must publish bounded demand"
                    );
                }
                BuiltinStreamingBoundary::PrefixState => {
                    assert!(
                        matches!(
                            demand_law(id),
                            BuiltinDemandLaw::TakeWhile | BuiltinDemandLaw::DropWhile
                        ),
                        "{method:?} prefix-state boundaries must publish prefix demand"
                    );
                }
                BuiltinStreamingBoundary::Expanding => {
                    assert_eq!(
                        demand_law(id),
                        BuiltinDemandLaw::FlatMapLike,
                        "{method:?} expanding boundaries must publish flat-map demand"
                    );
                }
                BuiltinStreamingBoundary::FullInputState => {
                    assert!(
                        matches!(
                            demand_law(id),
                            BuiltinDemandLaw::NumericReducer
                                | BuiltinDemandLaw::KeyOnlyReducer
                                | BuiltinDemandLaw::RowKeyedReducer
                                | BuiltinDemandLaw::OrderBarrier
                                | BuiltinDemandLaw::UniqueLike
                        ),
                        "{method:?} full-input state boundary must publish full-input demand"
                    );
                }
                BuiltinStreamingBoundary::FullInputOrder => {
                    assert!(
                        matches!(
                            demand_law(id),
                            BuiltinDemandLaw::OrderBarrier | BuiltinDemandLaw::Reverse
                        ),
                        "{method:?} full-input order boundary must publish ordering demand"
                    );
                }
                BuiltinStreamingBoundary::LegacyMaterialized => {
                    panic!("{method:?} registered builtins must not use legacy materialization");
                }
            }
        }
    }

    #[test]
    fn registry_drives_pipeline_lowering() {
        assert_eq!(
            pipeline_lowering(BuiltinId::from_method(BuiltinMethod::Filter)),
            Some(BuiltinPipelineLowering::ExprArg)
        );
        assert_eq!(
            pipeline_lowering(BuiltinId::from_method(BuiltinMethod::Map)),
            Some(BuiltinPipelineLowering::ExprArg)
        );
        assert_eq!(
            pipeline_lowering(BuiltinId::from_method(BuiltinMethod::FindOne)),
            Some(BuiltinPipelineLowering::TerminalSink)
        );
        assert_eq!(
            pipeline_lowering(BuiltinId::from_method(BuiltinMethod::Take)),
            Some(BuiltinPipelineLowering::UsizeArg { min: 0 })
        );
        assert_eq!(
            pipeline_lowering(BuiltinId::from_method(BuiltinMethod::Sort)),
            Some(BuiltinPipelineLowering::Sort)
        );
        assert_eq!(
            pipeline_lowering(BuiltinId::from_method(BuiltinMethod::Reverse)),
            Some(BuiltinPipelineLowering::Nullary)
        );
        assert_eq!(
            pipeline_lowering(BuiltinId::from_method(BuiltinMethod::Split)),
            Some(BuiltinPipelineLowering::StringArg)
        );
        assert_eq!(
            pipeline_lowering(BuiltinId::from_method(BuiltinMethod::ReplaceAll)),
            Some(BuiltinPipelineLowering::StringPairArg)
        );
        assert_eq!(
            pipeline_lowering(BuiltinId::from_method(BuiltinMethod::Slice)),
            Some(BuiltinPipelineLowering::IntRangeArg)
        );
        assert_eq!(
            pipeline_lowering(BuiltinId::from_method(BuiltinMethod::Count)),
            Some(BuiltinPipelineLowering::TerminalSink)
        );
    }

    #[test]
    fn registry_drives_expression_stage_shapes() {
        assert_eq!(
            expr_stage(BuiltinId::from_method(BuiltinMethod::Filter)),
            Some(BuiltinExprStage::Filter)
        );
        assert_eq!(
            expr_stage(BuiltinId::from_method(BuiltinMethod::Find)),
            Some(BuiltinExprStage::Filter)
        );
        assert_eq!(
            expr_stage(BuiltinId::from_method(BuiltinMethod::Map)),
            Some(BuiltinExprStage::Map)
        );
        assert_eq!(
            expr_stage(BuiltinId::from_method(BuiltinMethod::FlatMap)),
            Some(BuiltinExprStage::FlatMap)
        );
        assert_eq!(
            expr_stage(BuiltinId::from_method(BuiltinMethod::UniqueBy)),
            Some(BuiltinExprStage::UniqueBy)
        );
        assert_eq!(
            expr_stage(BuiltinId::from_method(BuiltinMethod::TransformKeys)),
            Some(BuiltinExprStage::ExprBuiltin)
        );
        assert_eq!(
            expr_stage(BuiltinId::from_method(BuiltinMethod::Take)),
            None
        );
    }

    #[test]
    fn registry_drives_nullary_stage_shapes() {
        assert_eq!(
            nullary_stage(BuiltinId::from_method(BuiltinMethod::Reverse)),
            Some(BuiltinNullaryStage::Reverse)
        );
        assert_eq!(
            nullary_stage(BuiltinId::from_method(BuiltinMethod::Unique)),
            Some(BuiltinNullaryStage::Unique)
        );
        assert_eq!(
            nullary_stage(BuiltinId::from_method(BuiltinMethod::Keys)),
            None
        );
        assert_eq!(
            nullary_stage(BuiltinId::from_method(BuiltinMethod::Take)),
            None
        );
    }

    #[test]
    fn registry_drives_string_pair_stage_shapes() {
        assert_eq!(
            string_pair_stage(BuiltinId::from_method(BuiltinMethod::Replace)),
            Some(BuiltinStringPairStage::Replace { all: false })
        );
        assert_eq!(
            string_pair_stage(BuiltinId::from_method(BuiltinMethod::ReplaceAll)),
            Some(BuiltinStringPairStage::Replace { all: true })
        );
        assert_eq!(
            string_pair_stage(BuiltinId::from_method(BuiltinMethod::Split)),
            None
        );
    }

    #[test]
    fn registry_drives_expression_payload_behavior() {
        assert_eq!(
            expr_payload(BuiltinId::from_method(BuiltinMethod::TakeWhile)),
            Some(BuiltinExprPayload::PredicateScan)
        );
        assert_eq!(
            expr_payload(BuiltinId::from_method(BuiltinMethod::FilterKeys)),
            Some(BuiltinExprPayload::PredicateScan)
        );
        assert_eq!(
            expr_payload(BuiltinId::from_method(BuiltinMethod::TransformValues)),
            Some(BuiltinExprPayload::Projection)
        );
        assert_eq!(
            expr_payload(BuiltinId::from_method(BuiltinMethod::CountBy)),
            Some(BuiltinExprPayload::KeyOnlyReducer)
        );
        assert_eq!(
            expr_payload(BuiltinId::from_method(BuiltinMethod::GroupBy)),
            Some(BuiltinExprPayload::RowKeyedReducer)
        );
        assert_eq!(
            expr_payload(BuiltinId::from_method(BuiltinMethod::Take)),
            None
        );
    }

    #[test]
    fn registry_expression_lowerings_describe_payload_behavior() {
        for (method, _, _) in all_method_entries() {
            let id = BuiltinId::from_method(method);
            let lowering = pipeline_lowering(id);
            let has_expr_lowering = matches!(
                lowering,
                Some(
                    BuiltinPipelineLowering::ExprArg
                        | BuiltinPipelineLowering::TerminalExprArg { .. }
                )
            );
            if !has_expr_lowering {
                continue;
            }

            match expr_stage(id) {
                Some(
                    BuiltinExprStage::Filter
                    | BuiltinExprStage::Map
                    | BuiltinExprStage::FlatMap
                    | BuiltinExprStage::UniqueBy,
                ) => {}
                Some(BuiltinExprStage::ExprBuiltin) | None => assert!(
                    expr_payload(id).is_some(),
                    "{method:?} has expression lowering without payload metadata"
                ),
            }
        }
    }

    #[test]
    fn registry_drives_unused_expression_stage_elision() {
        for method in [
            BuiltinMethod::TransformKeys,
            BuiltinMethod::TransformValues,
            BuiltinMethod::FilterKeys,
            BuiltinMethod::FilterValues,
        ] {
            assert!(
                expr_stage_elidable_when_value_unused(BuiltinId::from_method(method)),
                "{method:?}"
            );
        }
        assert!(!expr_stage_elidable_when_value_unused(
            BuiltinId::from_method(BuiltinMethod::TakeWhile)
        ));
        assert!(!expr_stage_elidable_when_value_unused(
            BuiltinId::from_method(BuiltinMethod::GroupBy)
        ));
    }

    #[test]
    fn registry_drives_unused_builtin_stage_elision() {
        for method in [
            BuiltinMethod::Upper,
            BuiltinMethod::Lower,
            BuiltinMethod::GetPath,
            BuiltinMethod::HasKey,
            BuiltinMethod::Missing,
        ] {
            assert!(
                stage_elidable_when_value_unused(BuiltinId::from_method(method)),
                "{method:?}"
            );
        }

        for method in [
            BuiltinMethod::Filter,
            BuiltinMethod::Sort,
            BuiltinMethod::Take,
            BuiltinMethod::Count,
            BuiltinMethod::Unknown,
        ] {
            assert!(
                !stage_elidable_when_value_unused(BuiltinId::from_method(method)),
                "{method:?}"
            );
        }
    }

    #[test]
    fn registry_drives_delayable_view_projection_classification() {
        for method in [
            BuiltinMethod::Upper,
            BuiltinMethod::GetPath,
            BuiltinMethod::Has,
            BuiltinMethod::HasAll,
            BuiltinMethod::HasKey,
            BuiltinMethod::HasPath,
            BuiltinMethod::Missing,
            BuiltinMethod::Pick,
            BuiltinMethod::Omit,
        ] {
            assert!(
                stage_delayable_view_projection(BuiltinId::from_method(method)),
                "{method:?}"
            );
        }

        for method in [
            BuiltinMethod::Map,
            BuiltinMethod::Filter,
            BuiltinMethod::Sort,
            BuiltinMethod::Rows,
            BuiltinMethod::Unknown,
        ] {
            assert!(
                !stage_delayable_view_projection(BuiltinId::from_method(method)),
                "{method:?}"
            );
        }
    }

    #[test]
    fn view_projection_metadata_is_pipeline_safe() {
        for (method, _, _) in all_method_entries() {
            let id = BuiltinId::from_method(method);
            if !view_projection(id) {
                continue;
            }

            assert!(is_pure(id), "{method:?} view projections must be pure");
            assert_eq!(
                pipeline_materialization(id),
                BuiltinPipelineMaterialization::Streaming,
                "{method:?} view projections must not force materialization"
            );
            if stage_delayable_view_projection(id) {
                assert_eq!(
                    builtin_cardinality(id),
                    Some(BuiltinCardinality::OneToOne),
                    "{method:?} delayable view projections must be one-to-one"
                );
                assert_eq!(
                    effective_pipeline_order_effect(id, true),
                    BuiltinPipelineOrderEffect::Preserves,
                    "{method:?} delayable view projections must preserve row order"
                );
            }
        }
    }

    #[test]
    fn terminal_lowering_metadata_has_sink_semantics() {
        for (method, _, _) in all_method_entries() {
            let id = BuiltinId::from_method(method);
            match pipeline_lowering(id) {
                Some(BuiltinPipelineLowering::TerminalSink) => assert!(
                    builtin_sink(id).is_some()
                        || predicate_sink(id).is_some()
                        || membership_sink(id).is_some()
                        || arg_extreme_sink(id).is_some(),
                    "{method:?} terminal sink lowering must advertise sink behavior"
                ),
                Some(BuiltinPipelineLowering::TerminalUsizeSink { .. }) => assert!(
                    array_selector(id).is_some() || builtin_sink(id).is_some(),
                    "{method:?} terminal usize sink lowering must advertise selection/sink behavior"
                ),
                Some(BuiltinPipelineLowering::TerminalExprArg { .. }) => assert!(
                    expr_stage(id).is_some(),
                    "{method:?} terminal expression lowering must advertise expression stage behavior"
                ),
                _ => {}
            }
        }
    }

    #[test]
    fn registry_drives_object_lambda_classification() {
        for (method, lambda) in [
            (
                BuiltinMethod::TransformKeys,
                BuiltinObjectLambda::TransformKeys,
            ),
            (
                BuiltinMethod::TransformValues,
                BuiltinObjectLambda::TransformValues,
            ),
            (BuiltinMethod::FilterKeys, BuiltinObjectLambda::FilterKeys),
            (
                BuiltinMethod::FilterValues,
                BuiltinObjectLambda::FilterValues,
            ),
        ] {
            let id = BuiltinId::from_method(method);
            assert_eq!(object_lambda(id), Some(lambda), "{method:?}");
            assert_eq!(demand_law(id), lambda.demand_law(), "{method:?}");
            assert_eq!(expr_payload(id), Some(lambda.expr_payload()), "{method:?}");
        }
        assert_eq!(
            object_lambda(BuiltinId::from_method(BuiltinMethod::Map)),
            None
        );
    }

    #[test]
    fn registry_drives_count_predicate_support() {
        assert!(count_sink_accepts_predicate(BuiltinId::from_method(
            BuiltinMethod::Count
        )));
        assert!(!count_sink_accepts_predicate(BuiltinId::from_method(
            BuiltinMethod::ApproxCountDistinct
        )));
        assert!(!count_sink_accepts_predicate(BuiltinId::from_method(
            BuiltinMethod::Sum
        )));
    }

    #[test]
    fn registry_drives_terminal_sink_classification() {
        assert_eq!(
            predicate_sink(BuiltinId::from_method(BuiltinMethod::Any)),
            Some(BuiltinPredicateSink::Any)
        );
        assert_eq!(
            sink_accumulator(BuiltinId::from_method(BuiltinMethod::Any)),
            None
        );
        assert_eq!(
            predicate_sink(BuiltinId::from_method(BuiltinMethod::All)),
            Some(BuiltinPredicateSink::All)
        );
        assert_eq!(
            predicate_sink(BuiltinId::from_method(BuiltinMethod::FindOne)),
            Some(BuiltinPredicateSink::FindOne)
        );
        assert_eq!(
            predicate_sink(BuiltinId::from_method(BuiltinMethod::Count)),
            None
        );
        assert_eq!(
            sink_accumulator(BuiltinId::from_method(BuiltinMethod::Count)),
            Some(BuiltinSinkAccumulator::Count)
        );
        assert_eq!(
            sink_accumulator(BuiltinId::from_method(BuiltinMethod::Sum)),
            Some(BuiltinSinkAccumulator::Numeric)
        );
        assert_eq!(
            sink_accumulator(BuiltinId::from_method(BuiltinMethod::First)),
            Some(BuiltinSinkAccumulator::SelectOne(
                BuiltinSelectionPosition::First
            ))
        );
        assert_eq!(
            predicate_sink_value_need(BuiltinPredicateSink::Any),
            ValueNeed::Predicate
        );
        assert_eq!(
            predicate_sink_value_need(BuiltinPredicateSink::FindOne),
            ValueNeed::Whole
        );
        assert_eq!(
            predicate_sink_demand(BuiltinPredicateSink::Any),
            Demand {
                pull: PullDemand::All,
                value: ValueNeed::Predicate,
                order: false,
            }
        );
        assert_eq!(
            predicate_sink_demand(BuiltinPredicateSink::FindOne),
            Demand {
                pull: PullDemand::All,
                value: ValueNeed::Whole,
                order: false,
            }
        );
        assert_eq!(
            predicate_sink_result_demand(BuiltinPredicateSink::Any),
            SinkResultDemand::UntilMatch
        );
        assert_eq!(
            predicate_sink_result_demand(BuiltinPredicateSink::All),
            SinkResultDemand::UntilFailure
        );
        assert_eq!(
            predicate_sink_result_demand(BuiltinPredicateSink::FindOne),
            SinkResultDemand::None
        );

        assert_eq!(
            membership_sink(BuiltinId::from_method(BuiltinMethod::Includes)),
            Some(BuiltinMembershipSink::Includes)
        );
        assert_eq!(
            membership_sink(BuiltinId::from_method(BuiltinMethod::Index)),
            Some(BuiltinMembershipSink::Index)
        );
        assert_eq!(
            membership_sink(BuiltinId::from_method(BuiltinMethod::IndicesOf)),
            Some(BuiltinMembershipSink::IndicesOf)
        );
        assert_eq!(
            membership_sink(BuiltinId::from_method(BuiltinMethod::Has)),
            None
        );
        assert_eq!(
            membership_sink_value_need(BuiltinMembershipSink::Includes),
            ValueNeed::Whole
        );
        assert_eq!(
            membership_sink_value_need(BuiltinMembershipSink::IndicesOf),
            ValueNeed::Whole
        );
        assert_eq!(
            membership_sink_demand(BuiltinMembershipSink::Includes),
            Demand {
                pull: PullDemand::All,
                value: ValueNeed::Whole,
                order: false,
            }
        );
        assert_eq!(
            membership_sink_result_demand(BuiltinMembershipSink::Includes),
            SinkResultDemand::UntilMatch
        );
        assert_eq!(
            membership_sink_result_demand(BuiltinMembershipSink::Index),
            SinkResultDemand::UntilMatch
        );
        assert_eq!(
            membership_sink_result_demand(BuiltinMembershipSink::IndicesOf),
            SinkResultDemand::None
        );

        assert_eq!(
            arg_extreme_sink(BuiltinId::from_method(BuiltinMethod::MaxBy)),
            Some(BuiltinArgExtremeSink::MaxBy)
        );
        assert_eq!(
            arg_extreme_sink(BuiltinId::from_method(BuiltinMethod::MinBy)),
            Some(BuiltinArgExtremeSink::MinBy)
        );
        assert_eq!(
            arg_extreme_sink_demand(BuiltinArgExtremeSink::MaxBy),
            Demand {
                pull: PullDemand::All,
                value: ValueNeed::Whole,
                order: true,
            }
        );
        assert_eq!(
            arg_extreme_sink(BuiltinId::from_method(BuiltinMethod::Max)),
            None
        );
    }

    #[test]
    fn registry_terminal_sink_specs_report_registered_methods() {
        use crate::exec::pipeline::{MembershipSinkSpec, MembershipSinkTarget, PredicateSinkSpec};
        use crate::vm::Program;
        use std::sync::Arc;

        let program = Arc::new(Program::new(Vec::new(), "<sink-method-test>"));
        for method in [
            BuiltinMethod::Any,
            BuiltinMethod::All,
            BuiltinMethod::FindIndex,
            BuiltinMethod::IndicesWhere,
            BuiltinMethod::FindOne,
        ] {
            let spec = PredicateSinkSpec::from_method(method, Arc::clone(&program), None)
                .unwrap_or_else(|| panic!("{method:?} should be a predicate sink"));
            assert_eq!(spec.method(), method, "{method:?}");
            assert_eq!(
                predicate_sink(BuiltinId::from_method(spec.method())),
                predicate_sink(BuiltinId::from_method(method)),
                "{method:?}"
            );
        }

        for method in [
            BuiltinMethod::Includes,
            BuiltinMethod::Index,
            BuiltinMethod::IndicesOf,
        ] {
            let spec =
                MembershipSinkSpec::from_method(method, MembershipSinkTarget::Literal(Val::Int(1)))
                    .unwrap_or_else(|| panic!("{method:?} should be a membership sink"));
            assert_eq!(spec.method(), method, "{method:?}");
            assert_eq!(
                membership_sink(BuiltinId::from_method(spec.method())),
                membership_sink(BuiltinId::from_method(method)),
                "{method:?}"
            );
        }
    }

    #[test]
    fn registry_drives_logical_shapes() {
        assert_eq!(
            logical_shape(BuiltinId::from_method(BuiltinMethod::Filter)),
            Some(BuiltinLogicalShape::Filter)
        );
        assert_eq!(
            logical_shape(BuiltinId::from_method(BuiltinMethod::Find)),
            Some(BuiltinLogicalShape::FilterThenFirst)
        );
        assert_eq!(
            logical_shape(BuiltinId::from_method(BuiltinMethod::Map)),
            Some(BuiltinLogicalShape::Map)
        );
        assert_eq!(
            logical_shape(BuiltinId::from_method(BuiltinMethod::Sort)),
            Some(BuiltinLogicalShape::Sort)
        );
        assert_eq!(
            logical_shape(BuiltinId::from_method(BuiltinMethod::CountBy)),
            Some(BuiltinLogicalShape::CountBy)
        );
        assert_eq!(
            logical_shape(BuiltinId::from_method(BuiltinMethod::IndexBy)),
            Some(BuiltinLogicalShape::IndexBy)
        );
        assert_eq!(
            logical_shape(BuiltinId::from_method(BuiltinMethod::FromJson)),
            None
        );
    }

    #[test]
    fn registry_drives_row_stream_ops() {
        assert!(stream_source(BuiltinId::from_method(BuiltinMethod::Rows)));
        assert!(!stream_source(BuiltinId::from_method(
            BuiltinMethod::Reverse
        )));
        assert_eq!(
            row_stream_op(BuiltinId::from_method(BuiltinMethod::Rows)),
            None
        );
        assert_eq!(
            row_stream_op(BuiltinId::from_method(BuiltinMethod::Reverse)),
            Some(BuiltinRowStreamOp::Reverse)
        );
        assert_eq!(
            row_stream_op(BuiltinId::from_method(BuiltinMethod::Filter)),
            Some(BuiltinRowStreamOp::Filter)
        );
        assert_eq!(
            row_stream_op(BuiltinId::from_method(BuiltinMethod::Find)),
            Some(BuiltinRowStreamOp::FindFirst)
        );
        assert_eq!(
            row_stream_op(BuiltinId::from_method(BuiltinMethod::FindOne)),
            Some(BuiltinRowStreamOp::FindOne)
        );
        assert_eq!(
            row_stream_op(BuiltinId::from_method(BuiltinMethod::UniqueBy)),
            Some(BuiltinRowStreamOp::DistinctBy)
        );
        assert_eq!(
            row_stream_op(BuiltinId::from_method(BuiltinMethod::Take)),
            Some(BuiltinRowStreamOp::Take)
        );
        assert_eq!(
            row_stream_op(BuiltinId::from_method(BuiltinMethod::First)),
            Some(BuiltinRowStreamOp::First)
        );
        assert_eq!(
            row_stream_op(BuiltinId::from_method(BuiltinMethod::Last)),
            Some(BuiltinRowStreamOp::Last)
        );
        assert_eq!(
            row_stream_op(BuiltinId::from_method(BuiltinMethod::Len)),
            Some(BuiltinRowStreamOp::Count)
        );
        assert_eq!(
            row_stream_op(BuiltinId::from_method(BuiltinMethod::Map)),
            Some(BuiltinRowStreamOp::Map)
        );
        assert_eq!(
            row_stream_op(BuiltinId::from_method(BuiltinMethod::Sort)),
            None
        );
        assert!(BuiltinRowStreamOp::Last.is_terminal());
        assert!(BuiltinRowStreamOp::Count.is_terminal());
        assert!(BuiltinRowStreamOp::Any.is_terminal());
        assert!(BuiltinRowStreamOp::FindOne.is_terminal());
        assert_eq!(
            numeric_reducer(BuiltinId::from_method(BuiltinMethod::Sum)),
            Some(BuiltinNumericReducer::Sum)
        );
        assert_eq!(
            numeric_reducer(BuiltinId::from_method(BuiltinMethod::Avg)),
            Some(BuiltinNumericReducer::Avg)
        );
        assert_eq!(
            numeric_reducer(BuiltinId::from_method(BuiltinMethod::Min)),
            Some(BuiltinNumericReducer::Min)
        );
        assert_eq!(
            numeric_reducer(BuiltinId::from_method(BuiltinMethod::Max)),
            Some(BuiltinNumericReducer::Max)
        );
        assert!(BuiltinRowStreamOp::Sum.is_terminal());
        assert!(BuiltinRowStreamOp::Max.is_terminal());
        assert_eq!(
            BuiltinRowStreamOp::Sum.numeric_reducer(),
            Some(BuiltinNumericReducer::Sum)
        );
        assert_eq!(
            BuiltinRowStreamOp::Avg.numeric_reducer(),
            Some(BuiltinNumericReducer::Avg)
        );
        assert_eq!(
            BuiltinRowStreamOp::Min.numeric_reducer(),
            Some(BuiltinNumericReducer::Min)
        );
        assert_eq!(
            BuiltinRowStreamOp::Max.numeric_reducer(),
            Some(BuiltinNumericReducer::Max)
        );
        assert_eq!(BuiltinRowStreamOp::Count.numeric_reducer(), None);
        assert_eq!(
            BuiltinRowStreamOp::Any.predicate_sink(),
            Some(BuiltinPredicateSink::Any)
        );
        assert_eq!(
            BuiltinRowStreamOp::All.predicate_sink(),
            Some(BuiltinPredicateSink::All)
        );
        assert_eq!(
            BuiltinRowStreamOp::FindOne.predicate_sink(),
            Some(BuiltinPredicateSink::FindOne)
        );
        assert_eq!(BuiltinRowStreamOp::Filter.predicate_sink(), None);
        assert_eq!(
            numeric_reducer(BuiltinId::from_method(BuiltinMethod::Map)),
            None
        );
        assert!(!BuiltinRowStreamOp::FindFirst.is_terminal());
        assert!(!BuiltinRowStreamOp::Map.is_terminal());
    }

    #[test]
    fn registry_stream_sources_are_exhaustive() {
        let registered: Vec<_> = all_method_entries()
            .into_iter()
            .filter_map(|(method, _, _)| {
                stream_source(BuiltinId::from_method(method)).then_some(method)
            })
            .collect();
        assert_eq!(registered, vec![BuiltinMethod::Rows]);

        for method in registered {
            let id = BuiltinId::from_method(method);
            assert_eq!(row_stream_op(id), None, "{method:?}");
            assert!(!accepts_lambda_arg(id), "{method:?}");
            assert_eq!(
                method.spec().cardinality,
                BuiltinCardinality::OneToOne,
                "{method:?}"
            );
        }
    }

    #[test]
    fn registry_row_stream_terminal_ops_have_sink_metadata() {
        for (method, _, _) in all_method_entries() {
            let id = BuiltinId::from_method(method);
            let Some(op) = row_stream_op(id) else {
                continue;
            };

            if numeric_reducer(id).is_some() {
                assert!(
                    matches!(
                        op,
                        BuiltinRowStreamOp::Sum
                            | BuiltinRowStreamOp::Avg
                            | BuiltinRowStreamOp::Min
                            | BuiltinRowStreamOp::Max
                    ),
                    "{method:?} numeric reducer must expose a numeric row-stream op, got {op:?}"
                );
            }

            if op.is_terminal() {
                let spec = method.spec();
                assert!(
                    spec.sink.is_some()
                        || spec.predicate_sink.is_some()
                        || spec.numeric_reducer.is_some(),
                    "{method:?} terminal row-stream op {op:?} has no sink/reducer metadata"
                );
            }
        }
    }

    #[test]
    fn registry_row_stream_ops_match_demand_laws() {
        for (method, _, _) in all_method_entries() {
            let id = BuiltinId::from_method(method);
            let Some(op) = row_stream_op(id) else {
                continue;
            };
            let law = demand_law(id);
            let expected = match op {
                BuiltinRowStreamOp::Reverse => BuiltinDemandLaw::Reverse,
                BuiltinRowStreamOp::Filter | BuiltinRowStreamOp::FindFirst => {
                    BuiltinDemandLaw::FilterLike
                }
                BuiltinRowStreamOp::FindOne | BuiltinRowStreamOp::Any | BuiltinRowStreamOp::All => {
                    BuiltinDemandLaw::PredicateMapLike
                }
                BuiltinRowStreamOp::DistinctBy => BuiltinDemandLaw::UniqueLike,
                BuiltinRowStreamOp::Take => BuiltinDemandLaw::Take,
                BuiltinRowStreamOp::First => BuiltinDemandLaw::First,
                BuiltinRowStreamOp::Last => BuiltinDemandLaw::Last,
                BuiltinRowStreamOp::Count => BuiltinDemandLaw::Count,
                BuiltinRowStreamOp::Sum
                | BuiltinRowStreamOp::Avg
                | BuiltinRowStreamOp::Min
                | BuiltinRowStreamOp::Max => BuiltinDemandLaw::NumericReducer,
                BuiltinRowStreamOp::Map => BuiltinDemandLaw::MapLike,
            };
            assert_eq!(law, expected, "{method:?} row-stream op {op:?}");
        }
    }

    #[test]
    fn registry_row_stream_arg_shapes_match_pipeline_arity() {
        for (method, _, _) in all_method_entries() {
            let id = BuiltinId::from_method(method);
            let Some(op) = row_stream_op(id) else {
                continue;
            };
            let arity = match op.arg() {
                BuiltinRowStreamArg::None => 0,
                BuiltinRowStreamArg::Expr | BuiltinRowStreamArg::Usize => 1,
            };

            assert!(
                pipeline_accepts_arity(id, arity, true),
                "{method:?} row-stream op {op:?} arity {arity} is not accepted by pipeline metadata"
            );
            if op.arg() == BuiltinRowStreamArg::Expr {
                assert!(
                    accepts_lambda_arg(id),
                    "{method:?} row-stream op {op:?} requires an expression but is not lambda-capable"
                );
            }
        }
    }

    #[test]
    fn registry_row_stream_op_terminal_flags_are_exhaustive() {
        for op in [
            BuiltinRowStreamOp::Reverse,
            BuiltinRowStreamOp::Filter,
            BuiltinRowStreamOp::FindFirst,
            BuiltinRowStreamOp::FindOne,
            BuiltinRowStreamOp::DistinctBy,
            BuiltinRowStreamOp::Take,
            BuiltinRowStreamOp::First,
            BuiltinRowStreamOp::Last,
            BuiltinRowStreamOp::Count,
            BuiltinRowStreamOp::Sum,
            BuiltinRowStreamOp::Avg,
            BuiltinRowStreamOp::Min,
            BuiltinRowStreamOp::Max,
            BuiltinRowStreamOp::Any,
            BuiltinRowStreamOp::All,
            BuiltinRowStreamOp::Map,
        ] {
            let expected_arg = match op {
                BuiltinRowStreamOp::Reverse
                | BuiltinRowStreamOp::First
                | BuiltinRowStreamOp::Last
                | BuiltinRowStreamOp::Count
                | BuiltinRowStreamOp::Sum
                | BuiltinRowStreamOp::Avg
                | BuiltinRowStreamOp::Min
                | BuiltinRowStreamOp::Max => BuiltinRowStreamArg::None,
                BuiltinRowStreamOp::Filter
                | BuiltinRowStreamOp::FindFirst
                | BuiltinRowStreamOp::FindOne
                | BuiltinRowStreamOp::DistinctBy
                | BuiltinRowStreamOp::Any
                | BuiltinRowStreamOp::All
                | BuiltinRowStreamOp::Map => BuiltinRowStreamArg::Expr,
                BuiltinRowStreamOp::Take => BuiltinRowStreamArg::Usize,
            };
            assert_eq!(op.arg(), expected_arg, "{op:?}");
            let expected_terminal = matches!(
                op,
                BuiltinRowStreamOp::Last
                    | BuiltinRowStreamOp::Count
                    | BuiltinRowStreamOp::Sum
                    | BuiltinRowStreamOp::Avg
                    | BuiltinRowStreamOp::Min
                    | BuiltinRowStreamOp::Max
                    | BuiltinRowStreamOp::Any
                    | BuiltinRowStreamOp::All
                    | BuiltinRowStreamOp::FindOne
            );
            assert_eq!(op.is_terminal(), expected_terminal, "{op:?}");
            let expected_fixed_retained_limit = match op {
                BuiltinRowStreamOp::First
                | BuiltinRowStreamOp::FindFirst
                | BuiltinRowStreamOp::Last => Some(1),
                _ => None,
            };
            assert_eq!(
                op.fixed_retained_limit(),
                expected_fixed_retained_limit,
                "{op:?}"
            );
            let expected_blocks_parallel = matches!(
                op,
                BuiltinRowStreamOp::DistinctBy | BuiltinRowStreamOp::Last
            );
            assert_eq!(
                op.blocks_parallel_partitioning(),
                expected_blocks_parallel,
                "{op:?}"
            );
            assert_eq!(
                op.is_filter_like(),
                matches!(
                    op,
                    BuiltinRowStreamOp::Filter | BuiltinRowStreamOp::FindFirst
                ),
                "{op:?}"
            );
            assert_eq!(
                op.is_projector(),
                matches!(op, BuiltinRowStreamOp::Map),
                "{op:?}"
            );
            assert_eq!(
                op.is_keyed_state(),
                matches!(op, BuiltinRowStreamOp::DistinctBy),
                "{op:?}"
            );
            assert_eq!(
                op.is_row_selection(),
                matches!(
                    op,
                    BuiltinRowStreamOp::Filter
                        | BuiltinRowStreamOp::FindFirst
                        | BuiltinRowStreamOp::DistinctBy
                ),
                "{op:?}"
            );
            assert_eq!(
                op.preserves_order_before_limit(),
                matches!(
                    op,
                    BuiltinRowStreamOp::Filter
                        | BuiltinRowStreamOp::FindFirst
                        | BuiltinRowStreamOp::Map
                ),
                "{op:?}"
            );
        }
    }

    #[test]
    fn registry_classifies_pipeline_arity_without_method_special_cases() {
        assert_eq!(
            pipeline_arity(BuiltinId::from_method(BuiltinMethod::Filter), false),
            Some(BuiltinPipelineArity::Exact(1))
        );
        assert!(pipeline_accepts_arity(
            BuiltinId::from_method(BuiltinMethod::Filter),
            1,
            false
        ));
        assert!(!pipeline_accepts_arity(
            BuiltinId::from_method(BuiltinMethod::Filter),
            0,
            false
        ));
        assert!(pipeline_accepts_arity(
            BuiltinId::from_method(BuiltinMethod::Sort),
            0,
            false
        ));
        assert_eq!(
            pipeline_arity(BuiltinId::from_method(BuiltinMethod::Sort), false),
            Some(BuiltinPipelineArity::Range { min: 0, max: 1 })
        );
        assert!(pipeline_accepts_arity(
            BuiltinId::from_method(BuiltinMethod::Sort),
            1,
            false
        ));
        assert!(!pipeline_accepts_arity(
            BuiltinId::from_method(BuiltinMethod::Sort),
            2,
            false
        ));
        assert!(pipeline_accepts_arity(
            BuiltinId::from_method(BuiltinMethod::Slice),
            2,
            false
        ));
        assert!(!pipeline_accepts_arity(
            BuiltinId::from_method(BuiltinMethod::Count),
            1,
            false
        ));
        assert_eq!(
            pipeline_arity(BuiltinId::from_method(BuiltinMethod::Count), false),
            None
        );
        assert_eq!(
            pipeline_arity(BuiltinId::from_method(BuiltinMethod::Count), true),
            Some(BuiltinPipelineArity::Range { min: 0, max: 1 })
        );
        assert!(pipeline_accepts_arity(
            BuiltinId::from_method(BuiltinMethod::Count),
            1,
            true
        ));
        assert!(pipeline_accepts_arity(
            BuiltinId::from_method(BuiltinMethod::Sum),
            1,
            true
        ));
        assert!(pipeline_accepts_arity(
            BuiltinId::from_method(BuiltinMethod::First),
            1,
            true
        ));
        assert!(pipeline_accepts_arity(
            BuiltinId::from_method(BuiltinMethod::Last),
            1,
            true
        ));
        assert!(pipeline_accepts_arity(
            BuiltinId::from_method(BuiltinMethod::Includes),
            1,
            true
        ));
        assert!(!pipeline_accepts_arity(
            BuiltinId::from_method(BuiltinMethod::Includes),
            1,
            false
        ));
        assert_eq!(
            pipeline_arity(BuiltinId::from_method(BuiltinMethod::Find), false),
            None
        );
        assert_eq!(
            pipeline_arity(BuiltinId::from_method(BuiltinMethod::Find), true),
            Some(BuiltinPipelineArity::Exact(1))
        );
        assert!(!pipeline_accepts_arity(
            BuiltinId::from_method(BuiltinMethod::Find),
            1,
            false
        ));
        assert!(pipeline_accepts_arity(
            BuiltinId::from_method(BuiltinMethod::Find),
            1,
            true
        ));
    }

    #[test]
    fn sink_specs_own_terminal_pipeline_arity() {
        assert_eq!(
            builtin_sink(BuiltinId::from_method(BuiltinMethod::Count))
                .unwrap()
                .pipeline_arity(),
            BuiltinPipelineArity::Range { min: 0, max: 1 }
        );
        assert_eq!(
            builtin_sink(BuiltinId::from_method(BuiltinMethod::Len))
                .unwrap()
                .pipeline_arity(),
            BuiltinPipelineArity::Exact(0)
        );
        assert_eq!(
            builtin_sink(BuiltinId::from_method(BuiltinMethod::Sum))
                .unwrap()
                .pipeline_arity(),
            BuiltinPipelineArity::Range { min: 0, max: 1 }
        );
        assert_eq!(
            builtin_sink(BuiltinId::from_method(BuiltinMethod::First))
                .unwrap()
                .pipeline_arity(),
            BuiltinPipelineArity::Range { min: 0, max: 1 }
        );
        assert_eq!(
            builtin_sink(BuiltinId::from_method(BuiltinMethod::ApproxCountDistinct))
                .unwrap()
                .pipeline_arity(),
            BuiltinPipelineArity::Exact(0)
        );
    }

    #[test]
    fn registry_drives_pipeline_element_classification() {
        for method in [
            BuiltinMethod::Upper,
            BuiltinMethod::StripPrefix,
            BuiltinMethod::IsNumeric,
            BuiltinMethod::Abs,
            BuiltinMethod::ParseInt,
            // `Has` was previously element-wise but the streaming pipeline
            // wrapped its boolean result into `[true]`. Spec is now whole-
            // input scalar (not element-wise) - no wrap, no element-wise
            // vectorisation. Same for `Keys` / `Values` / `Entries`.
            BuiltinMethod::HasKey,
            BuiltinMethod::HasAll,
            BuiltinMethod::Lines,
            BuiltinMethod::GetPath,
        ] {
            assert!(
                pipeline_element(BuiltinId::from_method(method)),
                "{method:?} should be classified as a pipeline element"
            );
        }

        for method in [
            BuiltinMethod::Has,
            BuiltinMethod::Len,
            BuiltinMethod::FromJson,
            BuiltinMethod::Sort,
            BuiltinMethod::Flatten,
        ] {
            assert!(!pipeline_element(BuiltinId::from_method(method)));
        }

        for method in [
            BuiltinMethod::Upper,
            BuiltinMethod::Compact,
            BuiltinMethod::Remove,
        ] {
            assert!(
                pipeline_builtin_call_stage(BuiltinId::from_method(method)),
                "{method:?} should be accepted as a builtin-call stage"
            );
        }

        assert_eq!(
            composed_builtin_stage(BuiltinId::from_method(BuiltinMethod::Compact)),
            Some(BuiltinComposedStage::Compact)
        );
        assert_eq!(
            composed_builtin_stage(BuiltinId::from_method(BuiltinMethod::Remove)),
            Some(BuiltinComposedStage::RemoveValue)
        );
        assert_eq!(
            composed_builtin_stage(BuiltinId::from_method(BuiltinMethod::Upper)),
            None
        );

        for method in [BuiltinMethod::Has, BuiltinMethod::Len, BuiltinMethod::Sort] {
            assert!(
                !pipeline_builtin_call_stage(BuiltinId::from_method(method)),
                "{method:?} should not be accepted as a builtin-call stage"
            );
        }
    }

    #[test]
    fn registry_drives_view_projection_classification() {
        for method in [
            BuiltinMethod::Has,
            BuiltinMethod::HasAll,
            BuiltinMethod::HasKey,
            BuiltinMethod::Missing,
            BuiltinMethod::GetPath,
            BuiltinMethod::HasPath,
            BuiltinMethod::Keys,
            BuiltinMethod::Values,
            BuiltinMethod::Entries,
            BuiltinMethod::Pick,
            BuiltinMethod::Omit,
        ] {
            let id = BuiltinId::from_method(method);
            assert!(view_object_projection(id).is_some(), "{method:?}");
            assert!(view_projection(id), "{method:?}");
        }

        assert!(view_projection(BuiltinId::from_method(
            BuiltinMethod::Upper
        )));
        assert!(view_scalar_projection(BuiltinId::from_method(
            BuiltinMethod::Upper
        )));
        assert!(view_object_projection(BuiltinId::from_method(BuiltinMethod::Upper)).is_none());
        assert!(!view_projection(BuiltinId::from_method(
            BuiltinMethod::Sort
        )));
        assert!(!view_scalar_projection(BuiltinId::from_method(
            BuiltinMethod::Sort
        )));
    }

    #[test]
    fn registry_drives_raw_json_scalar_ops() {
        use crate::builtins::BuiltinArgs;

        assert_eq!(
            raw_json_scalar(
                BuiltinId::from_method(BuiltinMethod::Len),
                &BuiltinArgs::None
            ),
            Some(BuiltinRawJsonScalar::Len)
        );
        assert_eq!(
            raw_json_scalar(
                BuiltinId::from_method(BuiltinMethod::Upper),
                &BuiltinArgs::None
            ),
            Some(BuiltinRawJsonScalar::AsciiUpper)
        );
        assert_eq!(
            raw_json_scalar(
                BuiltinId::from_method(BuiltinMethod::Lower),
                &BuiltinArgs::None
            ),
            Some(BuiltinRawJsonScalar::AsciiLower)
        );
        assert_eq!(
            raw_json_scalar(
                BuiltinId::from_method(BuiltinMethod::Upper),
                &BuiltinArgs::Str(std::sync::Arc::from("x"))
            ),
            None
        );
        assert_eq!(
            raw_json_scalar(
                BuiltinId::from_method(BuiltinMethod::Sort),
                &BuiltinArgs::None
            ),
            None
        );
    }

    #[test]
    fn registry_drives_view_object_projection_ops() {
        assert_eq!(
            view_object_projection(BuiltinId::from_method(BuiltinMethod::Has)),
            Some(BuiltinViewObjectProjection::Has)
        );
        assert_eq!(
            view_object_projection(BuiltinId::from_method(BuiltinMethod::HasAll)),
            Some(BuiltinViewObjectProjection::HasAll)
        );
        assert_eq!(
            view_object_projection(BuiltinId::from_method(BuiltinMethod::Keys)),
            Some(BuiltinViewObjectProjection::Keys)
        );
        assert!(view_object_items_projection(BuiltinId::from_method(
            BuiltinMethod::Keys
        )));
        assert_eq!(
            view_object_items_projection_call(
                BuiltinId::from_method(BuiltinMethod::Keys),
                &BuiltinArgs::None
            ),
            Some(BuiltinViewObjectProjection::Keys)
        );
        assert_eq!(
            view_object_items_projection_call(
                BuiltinId::from_method(BuiltinMethod::Keys),
                &BuiltinArgs::Str(std::sync::Arc::from("x"))
            ),
            None
        );
        assert!(view_object_items_projection(BuiltinId::from_method(
            BuiltinMethod::Values
        )));
        assert!(view_object_items_projection(BuiltinId::from_method(
            BuiltinMethod::Entries
        )));
        assert_eq!(
            view_object_projection(BuiltinId::from_method(BuiltinMethod::ToPairs)),
            Some(BuiltinViewObjectProjection::ToPairs)
        );
        assert!(view_object_items_projection(BuiltinId::from_method(
            BuiltinMethod::ToPairs
        )));
        assert_eq!(
            view_object_projection(BuiltinId::from_method(BuiltinMethod::Pick)),
            Some(BuiltinViewObjectProjection::Pick)
        );
        assert!(view_projection_returns_owned(
            BuiltinId::from_method(BuiltinMethod::Pick),
            &BuiltinArgs::StrVec(vec![std::sync::Arc::from("a")])
        ));
        assert!(!view_object_items_projection(BuiltinId::from_method(
            BuiltinMethod::Pick
        )));
        assert_eq!(
            view_object_items_projection_call(
                BuiltinId::from_method(BuiltinMethod::Pick),
                &BuiltinArgs::None
            ),
            None
        );
        assert_eq!(
            view_object_projection(BuiltinId::from_method(BuiltinMethod::Upper)),
            None
        );
        for (method, projection) in [
            (
                BuiltinMethod::CamelCase,
                BuiltinViewValueProjection::CamelCase,
            ),
            (
                BuiltinMethod::Capitalize,
                BuiltinViewValueProjection::Capitalize,
            ),
            (BuiltinMethod::Dedent, BuiltinViewValueProjection::Dedent),
            (
                BuiltinMethod::FromBase64,
                BuiltinViewValueProjection::FromBase64,
            ),
            (BuiltinMethod::FromPairs, BuiltinViewValueProjection::FromPairs),
            (
                BuiltinMethod::HtmlEscape,
                BuiltinViewValueProjection::HtmlEscape,
            ),
            (
                BuiltinMethod::HtmlUnescape,
                BuiltinViewValueProjection::HtmlUnescape,
            ),
            (BuiltinMethod::Indent, BuiltinViewValueProjection::Indent),
            (BuiltinMethod::Invert, BuiltinViewValueProjection::Invert),
            (
                BuiltinMethod::KebabCase,
                BuiltinViewValueProjection::KebabCase,
            ),
            (BuiltinMethod::Or, BuiltinViewValueProjection::Or),
            (
                BuiltinMethod::PascalCase,
                BuiltinViewValueProjection::PascalCase,
            ),
            (
                BuiltinMethod::SnakeCase,
                BuiltinViewValueProjection::SnakeCase,
            ),
            (
                BuiltinMethod::TitleCase,
                BuiltinViewValueProjection::TitleCase,
            ),
            (
                BuiltinMethod::ToBase64,
                BuiltinViewValueProjection::ToBase64,
            ),
            (
                BuiltinMethod::UrlDecode,
                BuiltinViewValueProjection::UrlDecode,
            ),
            (
                BuiltinMethod::UrlEncode,
                BuiltinViewValueProjection::UrlEncode,
            ),
        ] {
            assert_eq!(
                view_value_projection(BuiltinId::from_method(method)),
                Some(projection)
            );
        }
        assert_eq!(
            view_value_projection(BuiltinId::from_method(BuiltinMethod::ToString)),
            Some(BuiltinViewValueProjection::ToString)
        );
        assert_eq!(
            view_value_projection(BuiltinId::from_method(BuiltinMethod::Replace)),
            Some(BuiltinViewValueProjection::Replace)
        );
        assert_eq!(
            view_value_projection(BuiltinId::from_method(BuiltinMethod::PadLeft)),
            Some(BuiltinViewValueProjection::PadLeft)
        );
        assert_eq!(
            view_value_projection(BuiltinId::from_method(BuiltinMethod::PadRight)),
            Some(BuiltinViewValueProjection::PadRight)
        );
        assert_eq!(
            view_value_projection(BuiltinId::from_method(BuiltinMethod::Center)),
            Some(BuiltinViewValueProjection::Center)
        );
        assert_eq!(
            view_value_projection(BuiltinId::from_method(BuiltinMethod::Repeat)),
            Some(BuiltinViewValueProjection::Repeat)
        );
        assert_eq!(
            view_value_projection(BuiltinId::from_method(BuiltinMethod::ReplaceAll)),
            Some(BuiltinViewValueProjection::ReplaceAll)
        );
        assert_eq!(
            view_value_projection(BuiltinId::from_method(BuiltinMethod::ReverseStr)),
            Some(BuiltinViewValueProjection::ReverseStr)
        );
        assert_eq!(
            view_value_projection(BuiltinId::from_method(BuiltinMethod::Slice)),
            Some(BuiltinViewValueProjection::Slice)
        );
        assert_eq!(
            view_value_projection(BuiltinId::from_method(BuiltinMethod::StripPrefix)),
            Some(BuiltinViewValueProjection::StripPrefix)
        );
        assert_eq!(
            view_value_projection(BuiltinId::from_method(BuiltinMethod::StripSuffix)),
            Some(BuiltinViewValueProjection::StripSuffix)
        );
        assert_eq!(
            view_value_projection(BuiltinId::from_method(BuiltinMethod::ToJson)),
            Some(BuiltinViewValueProjection::ToJson)
        );
        assert!(view_projection_returns_owned(
            BuiltinId::from_method(BuiltinMethod::Upper),
            &BuiltinArgs::None
        ));
        assert!(!view_projection_returns_owned(
            BuiltinId::from_method(BuiltinMethod::GetPath),
            &BuiltinArgs::Str(std::sync::Arc::from("nested.x"))
        ));
    }

    #[test]
    fn registry_applies_view_projection_ops() {
        use crate::builtins::BuiltinArgs;
        use crate::data::view::{ValView, ValueView};

        fn apply(method: BuiltinMethod, args: BuiltinArgs) -> Val {
            apply_field(method, args, "obj")
        }

        fn apply_field(method: BuiltinMethod, args: BuiltinArgs, field: &str) -> Val {
            let doc = Val::from(&serde_json::json!({
                "obj": {"a": 1, "b": null, "nested": {"x": 7}},
                "arr": ["a", 2, true],
                "text": "a b<&",
                "base64": "aMOp",
                "url": "a%20b%2F",
                "html": "a &lt; b &amp; c",
                "case": "hello WORLD",
                "words": "Hello world_test",
                "indent": "  a\n    b",
                "pairs": [["a", 1], {"key": "b", "value": 2}, {"k": "c"}],
                "invertible": {"a": "one", "b": 2, "c": true, "d": ["x"]}
            }));
            let view = ValView::new(&doc).field(field);
            match apply_view_projection(BuiltinId::from_method(method), &args, view)
                .unwrap_or_else(|| panic!("{method:?} did not apply for args {args:?}"))
            {
                ViewProjectionResult::View(view) => view.materialize(),
                ViewProjectionResult::Owned(value) => value,
            }
        }

        fn assert_view_eq(method: BuiltinMethod, args: BuiltinArgs, expected: serde_json::Value) {
            let actual = apply(method, args);
            let expected = Val::from(&expected);
            assert!(
                crate::util::vals_deep_eq(&actual, &expected),
                "{method:?}: actual {actual:?}, expected {expected:?}"
            );
        }

        assert_eq!(
            apply(
                BuiltinMethod::Has,
                BuiltinArgs::Str(std::sync::Arc::from("a"))
            ),
            Val::Bool(true)
        );
        assert_eq!(
            apply(
                BuiltinMethod::HasAll,
                BuiltinArgs::StrVec(vec![
                    std::sync::Arc::from("a"),
                    std::sync::Arc::from("nested")
                ])
            ),
            Val::Bool(true)
        );
        assert_eq!(
            apply(
                BuiltinMethod::HasKey,
                BuiltinArgs::Str(std::sync::Arc::from("a"))
            ),
            Val::Bool(true)
        );
        assert_eq!(
            apply(
                BuiltinMethod::Missing,
                BuiltinArgs::Str(std::sync::Arc::from("b"))
            ),
            Val::Bool(true)
        );
        assert_eq!(
            apply(
                BuiltinMethod::Missing,
                BuiltinArgs::Str(std::sync::Arc::from("nested.y"))
            ),
            Val::Bool(true)
        );
        assert!(crate::util::vals_deep_eq(
            &apply(
                BuiltinMethod::Missing,
                BuiltinArgs::StrVec(vec![
                    std::sync::Arc::from("a"),
                    std::sync::Arc::from("b"),
                    std::sync::Arc::from("nested.y"),
                ])
            ),
            &Val::arr(vec![
                Val::Str(std::sync::Arc::from("b")),
                Val::Str(std::sync::Arc::from("nested.y")),
            ])
        ));
        assert_eq!(
            apply(
                BuiltinMethod::GetPath,
                BuiltinArgs::Str(std::sync::Arc::from("nested.x"))
            ),
            Val::Int(7)
        );
        assert_eq!(
            apply(
                BuiltinMethod::HasPath,
                BuiltinArgs::Str(std::sync::Arc::from("nested.x"))
            ),
            Val::Bool(true)
        );
        assert_view_eq(
            BuiltinMethod::Keys,
            BuiltinArgs::None,
            serde_json::json!(["a", "b", "nested"]),
        );
        assert_view_eq(
            BuiltinMethod::Values,
            BuiltinArgs::None,
            serde_json::json!([1, null, {"x": 7}]),
        );
        assert_view_eq(
            BuiltinMethod::Entries,
            BuiltinArgs::None,
            serde_json::json!([["a", 1], ["b", null], ["nested", {"x": 7}]]),
        );
        assert_view_eq(
            BuiltinMethod::ToPairs,
            BuiltinArgs::None,
            serde_json::json!([
                {"key": "a", "val": 1},
                {"key": "b", "val": null},
                {"key": "nested", "val": {"x": 7}},
            ]),
        );
        assert!(crate::util::vals_deep_eq(
            &apply_field(BuiltinMethod::FromPairs, BuiltinArgs::None, "pairs"),
            &Val::from(&serde_json::json!({"a": 1, "b": 2, "c": null}))
        ));
        assert!(crate::util::vals_deep_eq(
            &apply_field(BuiltinMethod::Invert, BuiltinArgs::None, "invertible"),
            &Val::from(&serde_json::json!({
                "one": "a",
                "2": "b",
                "true": "c",
                "[\"x\"]": "d"
            }))
        ));
        assert_view_eq(
            BuiltinMethod::Pick,
            BuiltinArgs::StrVec(vec![
                std::sync::Arc::from("nested"),
                std::sync::Arc::from("a"),
            ]),
            serde_json::json!({"nested": {"x": 7}, "a": 1}),
        );
        assert_view_eq(
            BuiltinMethod::Omit,
            BuiltinArgs::StrVec(vec![std::sync::Arc::from("b")]),
            serde_json::json!({"a": 1, "nested": {"x": 7}}),
        );
        assert_eq!(
            apply(BuiltinMethod::ToString, BuiltinArgs::None),
            Val::Str(std::sync::Arc::from(r#"{"a":1,"b":null,"nested":{"x":7}}"#))
        );
        assert_eq!(
            apply(BuiltinMethod::ToJson, BuiltinArgs::None),
            Val::Str(std::sync::Arc::from(r#"{"a":1,"b":null,"nested":{"x":7}}"#))
        );
        assert_eq!(
            apply_field(BuiltinMethod::ToBase64, BuiltinArgs::None, "text"),
            Val::Str(std::sync::Arc::from("YSBiPCY="))
        );
        assert_eq!(
            apply_field(BuiltinMethod::FromBase64, BuiltinArgs::None, "base64"),
            Val::Str(std::sync::Arc::from("hé"))
        );
        assert_eq!(
            apply_field(BuiltinMethod::FromBase64, BuiltinArgs::None, "text"),
            Val::Null
        );
        assert_eq!(
            apply_field(BuiltinMethod::UrlEncode, BuiltinArgs::None, "text"),
            Val::Str(std::sync::Arc::from("a%20b%3C%26"))
        );
        assert_eq!(
            apply_field(BuiltinMethod::UrlDecode, BuiltinArgs::None, "url"),
            Val::Str(std::sync::Arc::from("a b/"))
        );
        assert_eq!(
            apply_field(BuiltinMethod::HtmlEscape, BuiltinArgs::None, "text"),
            Val::Str(std::sync::Arc::from("a b&lt;&amp;"))
        );
        assert_eq!(
            apply_field(BuiltinMethod::HtmlUnescape, BuiltinArgs::None, "html"),
            Val::Str(std::sync::Arc::from("a < b & c"))
        );
        assert_eq!(
            apply_field(BuiltinMethod::Capitalize, BuiltinArgs::None, "case"),
            Val::Str(std::sync::Arc::from("Hello world"))
        );
        assert_eq!(
            apply_field(BuiltinMethod::TitleCase, BuiltinArgs::None, "case"),
            Val::Str(std::sync::Arc::from("Hello World"))
        );
        assert_eq!(
            apply_field(BuiltinMethod::SnakeCase, BuiltinArgs::None, "words"),
            Val::Str(std::sync::Arc::from("hello_world_test"))
        );
        assert_eq!(
            apply_field(BuiltinMethod::KebabCase, BuiltinArgs::None, "words"),
            Val::Str(std::sync::Arc::from("hello-world-test"))
        );
        assert_eq!(
            apply_field(BuiltinMethod::CamelCase, BuiltinArgs::None, "words"),
            Val::Str(std::sync::Arc::from("helloWorldTest"))
        );
        assert_eq!(
            apply_field(BuiltinMethod::PascalCase, BuiltinArgs::None, "words"),
            Val::Str(std::sync::Arc::from("HelloWorldTest"))
        );
        assert_eq!(
            apply_field(BuiltinMethod::Dedent, BuiltinArgs::None, "indent"),
            Val::Str(std::sync::Arc::from("a\n  b"))
        );
        assert_eq!(
            apply_field(BuiltinMethod::Indent, BuiltinArgs::Usize(2), "case"),
            Val::Str(std::sync::Arc::from("  hello WORLD"))
        );
        assert_eq!(
            apply_field(
                BuiltinMethod::Indent,
                BuiltinArgs::Str(std::sync::Arc::from("> ")),
                "indent"
            ),
            Val::Str(std::sync::Arc::from(">   a\n>     b"))
        );
        assert_eq!(
            apply_field(
                BuiltinMethod::Or,
                BuiltinArgs::Val(Val::Str(std::sync::Arc::from("fallback"))),
                "case"
            ),
            Val::Str(std::sync::Arc::from("hello WORLD"))
        );
        assert_eq!(
            apply_field(
                BuiltinMethod::Or,
                BuiltinArgs::Val(Val::Str(std::sync::Arc::from("fallback"))),
                "missing"
            ),
            Val::Str(std::sync::Arc::from("fallback"))
        );
        let sliced_non_string = apply(
            BuiltinMethod::Slice,
            BuiltinArgs::I64Opt {
                first: 1,
                second: Some(4),
            },
        );
        assert_eq!(
            serde_json::Value::from(sliced_non_string),
            serde_json::json!({"a": 1, "b": null, "nested": {"x": 7}})
        );
        let replaced_non_string = apply(
            BuiltinMethod::Replace,
            BuiltinArgs::StrPair {
                first: std::sync::Arc::from("l"),
                second: std::sync::Arc::from("L"),
            },
        );
        assert_eq!(
            serde_json::Value::from(replaced_non_string),
            serde_json::json!({"a": 1, "b": null, "nested": {"x": 7}})
        );
        let stripped_non_string = apply(
            BuiltinMethod::StripPrefix,
            BuiltinArgs::Str(std::sync::Arc::from("pre")),
        );
        assert_eq!(
            serde_json::Value::from(stripped_non_string),
            serde_json::json!({"a": 1, "b": null, "nested": {"x": 7}})
        );
        let repeated_non_string = apply(BuiltinMethod::Repeat, BuiltinArgs::Usize(2));
        assert_eq!(
            serde_json::Value::from(repeated_non_string),
            serde_json::json!({"a": 1, "b": null, "nested": {"x": 7}})
        );
        let padded_non_string = apply(
            BuiltinMethod::PadLeft,
            BuiltinArgs::Pad {
                width: 8,
                fill: '_',
            },
        );
        assert_eq!(
            serde_json::Value::from(padded_non_string),
            serde_json::json!({"a": 1, "b": null, "nested": {"x": 7}})
        );
        let reversed_non_string = apply(BuiltinMethod::ReverseStr, BuiltinArgs::None);
        assert_eq!(
            serde_json::Value::from(reversed_non_string),
            serde_json::json!({"a": 1, "b": null, "nested": {"x": 7}})
        );
        for method in [
            BuiltinMethod::CamelCase,
            BuiltinMethod::Capitalize,
            BuiltinMethod::Dedent,
            BuiltinMethod::FromBase64,
            BuiltinMethod::HtmlEscape,
            BuiltinMethod::HtmlUnescape,
            BuiltinMethod::KebabCase,
            BuiltinMethod::PascalCase,
            BuiltinMethod::SnakeCase,
            BuiltinMethod::TitleCase,
            BuiltinMethod::ToBase64,
            BuiltinMethod::UrlDecode,
            BuiltinMethod::UrlEncode,
        ] {
            let non_string = apply(method, BuiltinArgs::None);
            assert_eq!(
                serde_json::Value::from(non_string),
                serde_json::json!({"a": 1, "b": null, "nested": {"x": 7}}),
                "{method:?}"
            );
        }
        let indented_non_string = apply(BuiltinMethod::Indent, BuiltinArgs::Usize(2));
        assert_eq!(
            serde_json::Value::from(indented_non_string),
            serde_json::json!({"a": 1, "b": null, "nested": {"x": 7}})
        );
    }

    #[test]
    fn registry_drives_array_selector_classification() {
        assert_eq!(
            array_selector(BuiltinArraySelector::First.id()),
            Some(BuiltinArraySelector::First)
        );
        assert_eq!(
            array_selector(BuiltinArraySelector::Last.id()),
            Some(BuiltinArraySelector::Last)
        );
        assert_eq!(
            array_selector(BuiltinArraySelector::Nth.id()),
            Some(BuiltinArraySelector::Nth)
        );
        assert_eq!(
            array_selector(BuiltinId::from_method(BuiltinMethod::Take)),
            None
        );
    }

    #[test]
    fn registry_array_selector_contracts_are_exhaustive() {
        let expected = [
            (
                BuiltinArraySelector::First.id(),
                BuiltinArraySelector::First,
            ),
            (BuiltinArraySelector::Last.id(), BuiltinArraySelector::Last),
            (BuiltinArraySelector::Nth.id(), BuiltinArraySelector::Nth),
        ];
        let mut registered: Vec<_> = all_method_entries()
            .into_iter()
            .filter_map(|(method, _, _)| {
                let id = BuiltinId::from_method(method);
                array_selector(id).map(|selector| (id, selector))
            })
            .collect();
        registered.sort_by_key(|(id, _)| id.0);
        assert_eq!(registered, expected);

        let first_selector = BuiltinArraySelector::First;
        let first = first_selector.id();
        assert_eq!(demand_law(first), first_selector.demand_law());
        assert_eq!(
            propagate_demand(first, BuiltinDemandArg::None, Demand::all(ValueNeed::Whole)).pull,
            PullDemand::FirstInput(1)
        );
        assert_eq!(
            terminal_selection_position(first),
            first_selector.selection_position()
        );
        assert_eq!(
            pipeline_lowering(first),
            Some(first_selector.pipeline_lowering())
        );
        assert_eq!(row_stream_op(first), first_selector.row_stream_op());

        let last_selector = BuiltinArraySelector::Last;
        let last = last_selector.id();
        assert_eq!(demand_law(last), last_selector.demand_law());
        assert_eq!(
            propagate_demand(last, BuiltinDemandArg::None, Demand::all(ValueNeed::Whole)).pull,
            PullDemand::LastInput(1)
        );
        assert_eq!(
            terminal_selection_position(last),
            last_selector.selection_position()
        );
        assert_eq!(
            pipeline_lowering(last),
            Some(last_selector.pipeline_lowering())
        );
        assert_eq!(row_stream_op(last), last_selector.row_stream_op());

        let nth_selector = BuiltinArraySelector::Nth;
        let nth = nth_selector.id();
        assert_eq!(demand_law(nth), nth_selector.demand_law());
        assert_eq!(
            propagate_demand(
                nth,
                BuiltinDemandArg::Usize(2),
                Demand::all(ValueNeed::Whole)
            )
            .pull,
            PullDemand::NthInput(2)
        );
        assert_eq!(
            terminal_selection_position(nth),
            nth_selector.selection_position()
        );
        assert_eq!(
            pipeline_lowering(nth),
            Some(nth_selector.pipeline_lowering())
        );
        assert_eq!(row_stream_op(nth), nth_selector.row_stream_op());
    }

    #[test]
    fn registry_drives_terminal_selection_position() {
        assert_eq!(
            terminal_selection_position(BuiltinId::from_method(BuiltinMethod::First)),
            Some(BuiltinSelectionPosition::First)
        );
        assert_eq!(
            terminal_selection_position(BuiltinId::from_method(BuiltinMethod::Last)),
            Some(BuiltinSelectionPosition::Last)
        );
        assert_eq!(
            terminal_selection_position(BuiltinId::from_method(BuiltinMethod::Count)),
            None
        );
    }

    #[test]
    fn registry_drives_structural_lowering() {
        assert_eq!(
            structural(BuiltinId::from_method(BuiltinMethod::DeepFind)),
            Some(BuiltinStructural::DeepFind)
        );
        assert_eq!(
            structural(BuiltinId::from_method(BuiltinMethod::DeepShape)),
            Some(BuiltinStructural::DeepShape)
        );
        assert_eq!(
            structural(BuiltinId::from_method(BuiltinMethod::DeepLike)),
            Some(BuiltinStructural::DeepLike)
        );
        assert_eq!(
            structural(BuiltinId::from_method(BuiltinMethod::Walk)),
            None
        );
        assert_eq!(
            structural(BuiltinId::from_method(BuiltinMethod::Filter)),
            None
        );
    }

    #[test]
    fn unknown_builtin_demand_is_conservative_barrier() {
        let downstream = Demand {
            pull: PullDemand::FirstInput(7),
            value: ValueNeed::Predicate,
            order: false,
        };
        assert_eq!(
            propagate_demand(
                BuiltinId::from_method(BuiltinMethod::Unknown),
                BuiltinDemandArg::None,
                downstream
            ),
            Demand::RESULT
        );
    }

    #[test]
    fn descriptor_round_trips_method_identity() {
        for (method, _, _) in all_method_entries() {
            let id = BuiltinId::from_method(method);
            assert_eq!(id.method(), Some(method));
            // spec() must not panic for any registered method
            let _ = method.spec();
        }
    }
}
