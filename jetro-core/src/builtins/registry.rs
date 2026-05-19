//! Stable numeric IDs for builtins and per-builtin demand-propagation laws.
//!
//! `BuiltinMethod` (the original enum in `builtins.rs`) remains the execution
//! identity used by the VM and pipeline. `BuiltinId` is a compact numeric
//! alias for the same set, stable across refactors, that new planner and
//! analysis code carries without depending on the legacy enum directly.

use crate::{
    builtins::{
        BuiltinCancellation, BuiltinCardinality, BuiltinCategory, BuiltinColumnarStage,
        BuiltinDemandLaw, BuiltinKeyedReducer, BuiltinMethod, BuiltinNumericReducer,
        BuiltinPipelineLowering,
        BuiltinPipelineMaterialization, BuiltinPipelineOrderEffect, BuiltinPipelineShape,
        BuiltinSinkAccumulator, BuiltinSinkDemand, BuiltinSinkSpec, BuiltinSinkValueNeed,
        BuiltinStageMerge, BuiltinStructural, BuiltinViewStage,
    },
    plan::demand::{Demand, PullDemand, ValueNeed},
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

/// Canonical argument-count contract for pipeline lowering. This keeps
/// receiver-start checks, stage construction, and tests from re-encoding
/// per-builtin arity rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BuiltinPipelineArity {
    /// Accepts exactly N arguments.
    Exact(usize),
    /// Accepts any count in the inclusive range.
    Range { min: usize, max: usize },
}

impl BuiltinPipelineArity {
    #[inline]
    pub(crate) fn accepts(self, arity: usize) -> bool {
        match self {
            Self::Exact(n) => arity == n,
            Self::Range { min, max } => (min..=max).contains(&arity),
        }
    }
}

/// Logical planner shape for pipeline-position builtins.
///
/// This keeps method classification in the builtin registry while allowing
/// `plan::logical` to own construction of `LogicalPlan` nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BuiltinLogicalShape {
    /// `filter(expr)`-style streaming predicate.
    Filter,
    /// `filter(expr)` followed by `first()` when terminal.
    FilterThenFirst,
    /// `map(expr)` one-to-one projection.
    Map,
    /// `flat_map(expr)` expansion.
    FlatMap,
    /// `take(n)` positional prefix.
    Take,
    /// `skip(n)` positional offset.
    Skip,
    /// Nullary terminal first.
    First,
    /// Nullary terminal last.
    Last,
    /// Nullary numeric reducer.
    Sum,
    /// Nullary numeric reducer.
    Avg,
    /// Nullary numeric reducer.
    Min,
    /// Nullary numeric reducer.
    Max,
    /// Nullary count reducer.
    Count,
    /// Nullary reverse barrier.
    Reverse,
    /// Prefix predicate barrier.
    TakeWhile,
    /// Prefix predicate barrier.
    DropWhile,
    /// Sort with optional key.
    Sort,
    /// Nullary unique.
    Unique,
    /// `unique_by(expr)`.
    UniqueBy,
    /// `group_by(expr)`.
    GroupBy,
    /// `count_by(expr)` followed by first when terminal.
    CountBy,
    /// `index_by(expr)` followed by first when terminal.
    IndexBy,
    /// Nullary approximate distinct reducer.
    ApproxCountDistinct,
}

/// Predicate terminal sink behavior for builtins with a predicate argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BuiltinPredicateSink {
    /// Returns true when any row matches the predicate.
    Any,
    /// Returns true when every row matches the predicate.
    All,
    /// Returns the zero-based index of the first matching row, or null.
    FindIndex,
    /// Returns all zero-based indices whose rows match.
    IndicesWhere,
    /// Returns exactly one matching row, erroring on zero or multiple matches.
    FindOne,
}

/// Membership terminal sink behavior for builtins with a target value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BuiltinMembershipSink {
    /// Returns true when any row equals the target.
    Includes,
    /// Returns the zero-based index of the first matching row, or null.
    Index,
    /// Returns all zero-based indices matching the target.
    IndicesOf,
}

/// Arg-extreme terminal sink behavior for builtins with a key expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BuiltinArgExtremeSink {
    /// Keep the row with the largest key.
    MaxBy,
    /// Keep the row with the smallest key.
    MinBy,
}

/// Concrete pipeline stage shape for builtins with one expression argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BuiltinExprStage {
    /// Predicate filter stage.
    Filter,
    /// One-to-one map stage.
    Map,
    /// Expanding flat-map stage.
    FlatMap,
    /// Deduplicate by key stage.
    UniqueBy,
    /// Generic expression-bearing builtin stage.
    ExprBuiltin,
}

/// Concrete pipeline stage shape for nullary stage builtins.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BuiltinNullaryStage {
    /// Reverse stage with cancellation metadata.
    Reverse,
    /// Deduplicate by full row value.
    Unique,
    /// Generic no-argument element builtin.
    Element,
}

/// Return the logical planner shape for builtin `id`, if it has one.
#[inline]
pub(crate) fn logical_shape(id: BuiltinId) -> Option<BuiltinLogicalShape> {
    let method = id.method()?;
    match method {
        BuiltinMethod::Filter | BuiltinMethod::FindAll => Some(BuiltinLogicalShape::Filter),
        BuiltinMethod::Find | BuiltinMethod::FindFirst => matches!(
            pipeline_lowering(id),
            Some(BuiltinPipelineLowering::TerminalExprArg {
                terminal: BuiltinMethod::First,
            })
        )
        .then_some(BuiltinLogicalShape::FilterThenFirst),
        BuiltinMethod::Map => Some(BuiltinLogicalShape::Map),
        BuiltinMethod::FlatMap => Some(BuiltinLogicalShape::FlatMap),
        BuiltinMethod::Take => Some(BuiltinLogicalShape::Take),
        BuiltinMethod::Skip => Some(BuiltinLogicalShape::Skip),
        BuiltinMethod::First => Some(BuiltinLogicalShape::First),
        BuiltinMethod::Last => Some(BuiltinLogicalShape::Last),
        BuiltinMethod::Sum => Some(BuiltinLogicalShape::Sum),
        BuiltinMethod::Avg => Some(BuiltinLogicalShape::Avg),
        BuiltinMethod::Min => Some(BuiltinLogicalShape::Min),
        BuiltinMethod::Max => Some(BuiltinLogicalShape::Max),
        BuiltinMethod::Count => Some(BuiltinLogicalShape::Count),
        BuiltinMethod::Reverse => Some(BuiltinLogicalShape::Reverse),
        BuiltinMethod::TakeWhile => Some(BuiltinLogicalShape::TakeWhile),
        BuiltinMethod::DropWhile => Some(BuiltinLogicalShape::DropWhile),
        BuiltinMethod::Sort => Some(BuiltinLogicalShape::Sort),
        BuiltinMethod::Unique => Some(BuiltinLogicalShape::Unique),
        BuiltinMethod::UniqueBy => Some(BuiltinLogicalShape::UniqueBy),
        BuiltinMethod::GroupBy => Some(BuiltinLogicalShape::GroupBy),
        BuiltinMethod::CountBy => matches!(
            pipeline_lowering(id),
            Some(BuiltinPipelineLowering::TerminalExprArg {
                terminal: BuiltinMethod::First,
            })
        )
        .then_some(BuiltinLogicalShape::CountBy),
        BuiltinMethod::IndexBy => matches!(
            pipeline_lowering(id),
            Some(BuiltinPipelineLowering::TerminalExprArg {
                terminal: BuiltinMethod::First,
            })
        )
        .then_some(BuiltinLogicalShape::IndexBy),
        BuiltinMethod::ApproxCountDistinct => Some(BuiltinLogicalShape::ApproxCountDistinct),
        _ => None,
    }
}

/// Return predicate terminal-sink behavior for builtin `id`, if it has one.
#[inline]
pub(crate) fn predicate_sink(id: BuiltinId) -> Option<BuiltinPredicateSink> {
    match id.method()? {
        BuiltinMethod::Any => Some(BuiltinPredicateSink::Any),
        BuiltinMethod::All => Some(BuiltinPredicateSink::All),
        BuiltinMethod::FindIndex => Some(BuiltinPredicateSink::FindIndex),
        BuiltinMethod::IndicesWhere => Some(BuiltinPredicateSink::IndicesWhere),
        BuiltinMethod::FindOne => Some(BuiltinPredicateSink::FindOne),
        _ => None,
    }
}

/// Return membership terminal-sink behavior for builtin `id`, if it has one.
#[inline]
pub(crate) fn membership_sink(id: BuiltinId) -> Option<BuiltinMembershipSink> {
    match id.method()? {
        BuiltinMethod::Includes => Some(BuiltinMembershipSink::Includes),
        BuiltinMethod::Index => Some(BuiltinMembershipSink::Index),
        BuiltinMethod::IndicesOf => Some(BuiltinMembershipSink::IndicesOf),
        _ => None,
    }
}

/// Return arg-extreme terminal-sink behavior for builtin `id`, if it has one.
#[inline]
pub(crate) fn arg_extreme_sink(id: BuiltinId) -> Option<BuiltinArgExtremeSink> {
    match id.method()? {
        BuiltinMethod::MaxBy => Some(BuiltinArgExtremeSink::MaxBy),
        BuiltinMethod::MinBy => Some(BuiltinArgExtremeSink::MinBy),
        _ => None,
    }
}

/// Return the concrete pipeline stage shape for an expression-argument builtin.
#[inline]
pub(crate) fn expr_stage(id: BuiltinId) -> Option<BuiltinExprStage> {
    match id.method()? {
        BuiltinMethod::Filter
        | BuiltinMethod::Find
        | BuiltinMethod::FindAll
        | BuiltinMethod::FindFirst => Some(BuiltinExprStage::Filter),
        BuiltinMethod::Map => Some(BuiltinExprStage::Map),
        BuiltinMethod::FlatMap => Some(BuiltinExprStage::FlatMap),
        BuiltinMethod::UniqueBy => Some(BuiltinExprStage::UniqueBy),
        method if matches!(
            pipeline_lowering(BuiltinId::from_method(method)),
            Some(BuiltinPipelineLowering::ExprArg)
                | Some(BuiltinPipelineLowering::TerminalExprArg { .. })
        ) =>
        {
            Some(BuiltinExprStage::ExprBuiltin)
        }
        _ => None,
    }
}

/// Return the concrete pipeline stage shape for a nullary pipeline builtin.
#[inline]
pub(crate) fn nullary_stage(id: BuiltinId) -> Option<BuiltinNullaryStage> {
    match id.method()? {
        BuiltinMethod::Reverse => Some(BuiltinNullaryStage::Reverse),
        BuiltinMethod::Unique => Some(BuiltinNullaryStage::Unique),
        method
            if matches!(
                pipeline_lowering(BuiltinId::from_method(method)),
                Some(BuiltinPipelineLowering::Nullary)
            ) && pipeline_element(BuiltinId::from_method(method)) =>
        {
            Some(BuiltinNullaryStage::Element)
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
        BuiltinDemandLaw::First => Demand::first(ValueNeed::Whole),
        BuiltinDemandLaw::Last => Demand {
            pull: PullDemand::LastInput(1),
            value: ValueNeed::Whole,
            order: true,
        },
        BuiltinDemandLaw::Nth => match arg {
            BuiltinDemandArg::Usize(i) => Demand {
                pull: PullDemand::NthInput(i),
                value: ValueNeed::Whole,
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
/// restrict the amount of input the planner must pull from its source.
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
        pipeline_materialization(id),
        BuiltinPipelineMaterialization::LegacyMaterialized
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

/// Return the builtin demand law for `id`.
#[inline]
pub(crate) fn builtin_demand_law(id: BuiltinId) -> BuiltinDemandLaw {
    demand_law(id)
}

/// Return the builtin sink metadata for `id`, if the builtin is a terminal sink.
#[inline]
pub(crate) fn builtin_sink(id: BuiltinId) -> Option<BuiltinSinkSpec> {
    id.method().and_then(|method| method.spec().sink)
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

/// Return true when builtin `id` can be composed into a view-native projection
/// kernel without materialising the receiver row.
#[inline]
pub(crate) fn view_projection(id: BuiltinId) -> bool {
    id.method()
        .is_some_and(|method| method.is_view_projection_method())
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

/// Return the pipeline lowering strategy for builtin `id`, indicating which
/// physical stage type and arguments the builtin compiles to.
#[inline]
pub(crate) fn pipeline_lowering(id: BuiltinId) -> Option<BuiltinPipelineLowering> {
    id.method().map(|m| m.spec().lowering).flatten()
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
        | Some(BuiltinPipelineLowering::TerminalExprArg { .. })
        | Some(BuiltinPipelineLowering::UsizeArg { .. })
        | Some(BuiltinPipelineLowering::StringArg) => Some(BuiltinPipelineArity::Exact(1)),
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
    let Some(sink) = spec.sink else {
        return None;
    };
    Some(match sink.accumulator {
        BuiltinSinkAccumulator::Count => {
            if method == BuiltinMethod::Count {
                BuiltinPipelineArity::Range { min: 0, max: 1 }
            } else {
                BuiltinPipelineArity::Exact(0)
            }
        }
        BuiltinSinkAccumulator::Numeric => BuiltinPipelineArity::Range { min: 0, max: 1 },
        BuiltinSinkAccumulator::SelectOne(_)
            if matches!(method, BuiltinMethod::First | BuiltinMethod::Last) =>
        {
            BuiltinPipelineArity::Range { min: 0, max: 1 }
        }
        BuiltinSinkAccumulator::SelectOne(_) | BuiltinSinkAccumulator::ApproxDistinct => {
            BuiltinPipelineArity::Exact(0)
        }
    })
}

/// Return `true` if builtin `id` is an element-wise operation that can be
/// applied independently to each item in a vectorised column.
#[inline]
pub(crate) fn pipeline_element(id: BuiltinId) -> bool {
    id.method().map(|m| m.spec().is_element).unwrap_or(false)
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

impl BuiltinId {
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

// ── Main registry macro ───────────────────────────────────────────────────────

// ── Trait-driven name lookup (replaces builtin_registry! macro) ───────────────

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
        BuiltinSelectionPosition, BuiltinSinkAccumulator,
    };

    #[test]
    fn registry_name_lookup_matches_legacy_lookup() {
        for (method, canonical, aliases) in all_method_entries() {
            assert_eq!(by_name(canonical).and_then(BuiltinId::method), Some(method));
            for alias in aliases {
                assert_eq!(by_name(alias).and_then(BuiltinId::method), Some(method));
            }
        }
        assert_eq!(by_name("missing_builtin"), None);
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
                .is_some_and(|sink| sink.accumulator == BuiltinSinkAccumulator::Numeric);
            assert_eq!(
                spec.numeric_reducer.is_some(),
                numeric_sink,
                "{method:?} numeric reducer metadata must match numeric sink metadata"
            );
        }
    }

    #[test]
    fn registry_accessors_match_builtin_specs() {
        for (method, _, _) in all_method_entries() {
            let id = BuiltinId::from_method(method);
            let spec = method.spec();
            assert_eq!(builtin_category(id), Some(spec.category), "{method:?}");
            assert_eq!(builtin_cardinality(id), Some(spec.cardinality), "{method:?}");
            assert_eq!(columnar_stage(id), spec.columnar_stage, "{method:?}");
            assert_eq!(stage_merge(id), spec.stage_merge, "{method:?}");
            assert_eq!(builtin_sink(id), spec.sink, "{method:?}");
            assert_eq!(keyed_reducer(id), spec.keyed_reducer, "{method:?}");
            assert_eq!(numeric_reducer(id), spec.numeric_reducer, "{method:?}");
            assert_eq!(is_pure(id), spec.pure, "{method:?}");
            assert_eq!(cancellation(id), spec.cancellation, "{method:?}");
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
    }

    #[test]
    fn registry_converts_sink_demands() {
        let first = sink_demand(BuiltinMethod::First.spec().sink.unwrap());
        assert_eq!(first.pull, PullDemand::FirstInput(1));
        assert_eq!(first.value, ValueNeed::Whole);

        let last = sink_demand(BuiltinMethod::Last.spec().sink.unwrap());
        assert_eq!(last.pull, PullDemand::LastInput(1));
        assert_eq!(last.value, ValueNeed::Whole);

        let count = sink_demand(BuiltinMethod::Count.spec().sink.unwrap());
        assert_eq!(count.pull, PullDemand::All);
        assert_eq!(count.value, ValueNeed::CountOnly);
        assert!(!count.order);
    }

    #[test]
    fn registry_sink_demands_match_all_sink_accumulators() {
        for (method, _, _) in all_method_entries() {
            let Some(sink) = method.spec().sink else {
                continue;
            };
            let demand = sink_demand(sink);
            match sink.accumulator {
                BuiltinSinkAccumulator::Count => {
                    assert_eq!(demand.pull, PullDemand::All, "{method:?}");
                    assert_eq!(demand.value, ValueNeed::CountOnly, "{method:?}");
                    assert!(!demand.order, "{method:?}");
                }
                BuiltinSinkAccumulator::Numeric => {
                    assert_eq!(demand.pull, PullDemand::All, "{method:?}");
                    assert_eq!(demand.value, ValueNeed::Numeric, "{method:?}");
                    assert!(!demand.order, "{method:?}");
                }
                BuiltinSinkAccumulator::ApproxDistinct => {
                    assert_eq!(demand.pull, PullDemand::All, "{method:?}");
                    assert_eq!(demand.value, ValueNeed::Whole, "{method:?}");
                    assert!(!demand.order, "{method:?}");
                }
                BuiltinSinkAccumulator::SelectOne(BuiltinSelectionPosition::First) => {
                    assert_eq!(demand.pull, PullDemand::FirstInput(1), "{method:?}");
                    assert_eq!(demand.value, ValueNeed::Whole, "{method:?}");
                    assert!(!demand.order, "{method:?}");
                }
                BuiltinSinkAccumulator::SelectOne(BuiltinSelectionPosition::Last) => {
                    assert_eq!(demand.pull, PullDemand::LastInput(1), "{method:?}");
                    assert_eq!(demand.value, ValueNeed::Whole, "{method:?}");
                    assert!(demand.order, "{method:?}");
                }
            }
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
    fn registry_marks_conservative_demand_barriers() {
        for method in [
            BuiltinMethod::FlatMap,
            BuiltinMethod::DropWhile,
            BuiltinMethod::Sort,
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
            BuiltinPipelineMaterialization::LegacyMaterialized
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
        assert!(pipeline_legacy_materialized(BuiltinId::from_method(
            BuiltinMethod::Split
        )));
        assert_eq!(
            pipeline_shape(BuiltinId::from_method(BuiltinMethod::Split))
                .unwrap()
                .can_indexed,
            true
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
        assert_eq!(expr_stage(BuiltinId::from_method(BuiltinMethod::Take)), None);
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
        assert_eq!(nullary_stage(BuiltinId::from_method(BuiltinMethod::Keys)), None);
        assert_eq!(nullary_stage(BuiltinId::from_method(BuiltinMethod::Take)), None);
    }

    #[test]
    fn registry_drives_terminal_sink_classification() {
        assert_eq!(
            predicate_sink(BuiltinId::from_method(BuiltinMethod::Any)),
            Some(BuiltinPredicateSink::Any)
        );
        assert_eq!(
            predicate_sink(BuiltinId::from_method(BuiltinMethod::All)),
            Some(BuiltinPredicateSink::All)
        );
        assert_eq!(
            predicate_sink(BuiltinId::from_method(BuiltinMethod::FindOne)),
            Some(BuiltinPredicateSink::FindOne)
        );
        assert_eq!(predicate_sink(BuiltinId::from_method(BuiltinMethod::Count)), None);

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
            arg_extreme_sink(BuiltinId::from_method(BuiltinMethod::MaxBy)),
            Some(BuiltinArgExtremeSink::MaxBy)
        );
        assert_eq!(
            arg_extreme_sink(BuiltinId::from_method(BuiltinMethod::MinBy)),
            Some(BuiltinArgExtremeSink::MinBy)
        );
        assert_eq!(
            arg_extreme_sink(BuiltinId::from_method(BuiltinMethod::Max)),
            None
        );
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
