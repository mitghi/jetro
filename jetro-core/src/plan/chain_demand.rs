//! Demand propagation adapters for planner-facing chain operators.
//!
//! `plan::chain_ir` owns only the operator description. This module maps that
//! representation onto the shared planning demand model.

use crate::{
    builtins::registry::propagate_demand as propagate_builtin_demand,
    plan::{
        chain_ir::{ChainOp, MatchRole},
        demand::{Demand, DemandOperator, PullDemand, ValueNeed},
    },
};
#[cfg(test)]
use crate::builtins::BuiltinCardinality;

/// Describes whether a pipeline slot carries a homogeneous stream, a single
/// scalar result, or an unconstrained mix of values.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueKind {
    /// No constraint on the kind of value in this slot.
    Any,
    /// The slot holds a sequence of values produced by a streaming operator.
    Stream,
    /// The slot holds exactly one scalar value (e.g. the result of `count`).
    Scalar,
}

/// Static specification describing the kind of values a `ChainOp` consumes
/// and produces, along with its cardinality and ordering guarantees.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpSpec {
    /// Kind of values the operator reads from its source.
    pub input: ValueKind,
    /// Kind of values the operator writes to the next stage.
    pub output: ValueKind,
    /// Relationship between input count and output count.
    pub cardinality: BuiltinCardinality,
    /// Whether output elements appear in the same order as their inputs.
    pub preserves_order: bool,
}

impl ChainOp {
    /// Propagate demand using the shared planner demand model.
    pub fn propagate_demand(&self, downstream: Demand) -> Demand {
        <Self as DemandOperator>::propagate_demand(self, downstream)
    }

    /// Derive the static `OpSpec` for this operator by consulting builtin registry metadata.
    #[cfg(test)]
    pub fn spec(&self) -> OpSpec {
        use crate::builtins::{
            registry::{builtin_cardinality, builtin_category, effective_pipeline_order_effect},
            BuiltinCardinality, BuiltinCategory, BuiltinPipelineOrderEffect,
        };

        match self {
            ChainOp::Match { role } => {
                let cardinality = match role {
                    MatchRole::Predicate => BuiltinCardinality::Filtering,
                    MatchRole::Transform => BuiltinCardinality::OneToOne,
                };
                OpSpec {
                    input: ValueKind::Stream,
                    output: ValueKind::Stream,
                    cardinality,
                    preserves_order: true,
                }
            }
            ChainOp::Builtin { id, .. } => {
                let Some(category) = builtin_category(*id) else {
                    return OpSpec {
                        input: ValueKind::Any,
                        output: ValueKind::Any,
                        cardinality: BuiltinCardinality::OneToOne,
                        preserves_order: true,
                    };
                };
                let cardinality =
                    builtin_cardinality(*id).unwrap_or(BuiltinCardinality::OneToOne);
                let input = match category {
                    BuiltinCategory::StreamingOneToOne
                    | BuiltinCategory::StreamingFilter
                    | BuiltinCategory::StreamingExpand
                    | BuiltinCategory::Reducer
                    | BuiltinCategory::Positional
                    | BuiltinCategory::Barrier
                    | BuiltinCategory::Relational => ValueKind::Stream,
                    _ => ValueKind::Any,
                };
                let output = match category {
                    BuiltinCategory::Reducer | BuiltinCategory::Positional => ValueKind::Scalar,
                    BuiltinCategory::StreamingOneToOne
                    | BuiltinCategory::StreamingFilter
                    | BuiltinCategory::StreamingExpand => ValueKind::Stream,
                    _ => ValueKind::Any,
                };
                OpSpec {
                    input,
                    output,
                    cardinality,
                    preserves_order: !matches!(
                        effective_pipeline_order_effect(*id, true),
                        BuiltinPipelineOrderEffect::Blocks
                    ),
                }
            }
        }
    }
}

impl DemandOperator for ChainOp {
    fn propagate_demand(&self, downstream: Demand) -> Demand {
        match self {
            ChainOp::Match { role } => match role {
                // Predicate match drops rows: downstream demand of N outputs
                // requires scanning until N pass the predicate.
                MatchRole::Predicate => Demand {
                    pull: match downstream.pull {
                        PullDemand::FirstInput(n) | PullDemand::UntilOutput(n) => {
                            PullDemand::UntilOutput(n)
                        }
                        PullDemand::LastInput(n) => PullDemand::LastInput(n),
                        PullDemand::NthInput(_) => PullDemand::All,
                        other => other,
                    },
                    value: downstream.value.merge(crate::plan::demand::ValueNeed::Predicate),
                    order: downstream.order || !matches!(downstream.pull, PullDemand::All),
                },
                // Transform match is 1:1 like `map`; positional demand passes
                // through, but observing the transformed value requires the
                // full input value needed to compute that transform.
                MatchRole::Transform => Demand {
                    value: if downstream.value.requires_payload() {
                        ValueNeed::Whole
                    } else {
                        downstream.value
                    },
                    ..downstream
                },
            },
            ChainOp::Builtin { id, demand_arg } => {
                propagate_builtin_demand(*id, *demand_arg, downstream)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        builtins::{BuiltinCardinality, BuiltinMethod},
        plan::demand::{propagate_demands, source_demand, Demand, PullDemand, ValueNeed},
    };

    fn op(method: BuiltinMethod) -> ChainOp {
        ChainOp::builtin(method)
    }

    fn op_usize(method: BuiltinMethod, n: usize) -> ChainOp {
        ChainOp::builtin_usize(method, n)
    }

    fn match_op(role: MatchRole) -> ChainOp {
        ChainOp::match_role(role)
    }

    #[test]
    fn match_predicate_classifies_as_filter() {
        let spec = ChainOp::match_role(MatchRole::Predicate).spec();
        assert_eq!(spec.cardinality, BuiltinCardinality::Filtering);
        assert_eq!(spec.input, ValueKind::Stream);
        assert_eq!(spec.output, ValueKind::Stream);
        assert!(spec.preserves_order);
    }

    #[test]
    fn match_transform_classifies_as_map() {
        let spec = ChainOp::match_role(MatchRole::Transform).spec();
        assert_eq!(spec.cardinality, BuiltinCardinality::OneToOne);
    }

    #[test]
    fn builtin_specs_come_from_registry_metadata() {
        let filter = op(BuiltinMethod::Filter).spec();
        assert_eq!(filter.input, ValueKind::Stream);
        assert_eq!(filter.output, ValueKind::Stream);
        assert_eq!(filter.cardinality, BuiltinCardinality::Filtering);
        assert!(filter.preserves_order);

        let count = op(BuiltinMethod::Count).spec();
        assert_eq!(count.input, ValueKind::Stream);
        assert_eq!(count.output, ValueKind::Scalar);
        assert_eq!(count.cardinality, BuiltinCardinality::Reducing);

        let sort = op(BuiltinMethod::Sort).spec();
        assert_eq!(sort.cardinality, BuiltinCardinality::Barrier);
        assert!(!sort.preserves_order);
    }

    #[test]
    fn chain_adapter_keeps_match_demand_in_plan_layer() {
        let ops = [
            ChainOp::match_role(MatchRole::Predicate),
            ChainOp::builtin(BuiltinMethod::First),
        ];
        let demand = source_demand(&ops, Demand::RESULT);
        assert_eq!(demand.pull, PullDemand::UntilOutput(1));
        assert_eq!(demand.value, ValueNeed::Whole);
    }

    #[test]
    fn match_predicate_first_scans_until_one_output() {
        let ops = [match_op(MatchRole::Predicate), op(BuiltinMethod::First)];
        let demand = source_demand(&ops, Demand::RESULT);
        assert_eq!(demand.pull, PullDemand::UntilOutput(1));
        assert_eq!(demand.value, ValueNeed::Whole);
    }

    #[test]
    fn match_transform_take_caps_upstream() {
        let ops = [
            match_op(MatchRole::Transform),
            op_usize(BuiltinMethod::Take, 3),
        ];
        let demand = source_demand(&ops, Demand::RESULT);
        assert_eq!(demand.pull, PullDemand::FirstInput(3));
    }

    #[test]
    fn match_transform_preserves_map_value_demand() {
        let demand = match_op(MatchRole::Transform).propagate_demand(Demand {
            pull: PullDemand::LastInput(1),
            value: ValueNeed::Predicate,
            order: true,
        });
        assert_eq!(demand.pull, PullDemand::LastInput(1));
        assert_eq!(demand.value, ValueNeed::Whole);
        assert!(demand.order);

        let count_only = match_op(MatchRole::Transform).propagate_demand(Demand {
            pull: PullDemand::All,
            value: ValueNeed::CountOnly,
            order: false,
        });
        assert_eq!(count_only.value, ValueNeed::CountOnly);
    }

    #[test]
    fn match_predicate_take_widens_to_scan() {
        let ops = [
            match_op(MatchRole::Predicate),
            op_usize(BuiltinMethod::Take, 3),
        ];
        let demand = source_demand(&ops, Demand::RESULT);
        assert_eq!(demand.pull, PullDemand::UntilOutput(3));
    }

    #[test]
    fn match_predicate_count_keeps_predicate_value_need() {
        let ops = [match_op(MatchRole::Predicate), op(BuiltinMethod::Count)];
        let demand = source_demand(&ops, Demand::RESULT);
        assert_eq!(demand.pull, PullDemand::All);
        assert_eq!(demand.value, ValueNeed::Predicate);
        assert!(!demand.order);
    }

    #[test]
    fn filter_first_scans_until_one_output() {
        let ops = [op(BuiltinMethod::Filter), op(BuiltinMethod::First)];
        let demand = source_demand(&ops, Demand::RESULT);
        assert_eq!(demand.pull, PullDemand::UntilOutput(1));
        assert_eq!(demand.value, ValueNeed::Whole);
    }

    #[test]
    fn filter_last_requests_reverse_selective_demand() {
        let ops = [op(BuiltinMethod::Filter), op(BuiltinMethod::Last)];
        let demand = source_demand(&ops, Demand::RESULT);
        assert_eq!(demand.pull, PullDemand::LastInput(1));
        assert_eq!(demand.value, ValueNeed::Whole);
        assert!(demand.order);
    }

    #[test]
    fn predicate_nth_requests_ordered_full_scan() {
        let ops = [
            match_op(MatchRole::Predicate),
            op_usize(BuiltinMethod::Nth, 2),
        ];
        let demand = source_demand(&ops, Demand::RESULT);
        assert_eq!(demand.pull, PullDemand::All);
        assert_eq!(demand.value, ValueNeed::Whole);
        assert!(demand.order);
    }

    #[test]
    fn reverse_swaps_first_and_last_input_demand() {
        let ops = [op(BuiltinMethod::Reverse), op(BuiltinMethod::First)];
        let demand = source_demand(&ops, Demand::RESULT);
        assert_eq!(demand.pull, PullDemand::LastInput(1));
        assert_eq!(demand.value, ValueNeed::Whole);

        let ops = [op(BuiltinMethod::Reverse), op(BuiltinMethod::Last)];
        let demand = source_demand(&ops, Demand::RESULT);
        assert_eq!(demand.pull, PullDemand::FirstInput(1));
        assert_eq!(demand.value, ValueNeed::Whole);

        let ops = [op(BuiltinMethod::Reverse), op_usize(BuiltinMethod::Take, 2)];
        let demand = source_demand(&ops, Demand::RESULT);
        assert_eq!(demand.pull, PullDemand::LastInput(2));
        assert_eq!(demand.value, ValueNeed::Whole);
    }

    #[test]
    fn drop_while_is_a_prefix_barrier() {
        let ops = [op(BuiltinMethod::DropWhile), op(BuiltinMethod::First)];
        let demand = source_demand(&ops, Demand::RESULT);
        assert_eq!(demand.pull, PullDemand::All);
        assert_eq!(demand.value, ValueNeed::Whole);
        assert!(demand.order);
    }

    #[test]
    fn map_last_requests_last_input() {
        let ops = [op(BuiltinMethod::Map), op(BuiltinMethod::Last)];
        let demand = source_demand(&ops, Demand::RESULT);
        assert_eq!(demand.pull, PullDemand::LastInput(1));
        assert_eq!(demand.value, ValueNeed::Whole);
    }

    #[test]
    fn map_nth_requests_nth_input() {
        let ops = [op(BuiltinMethod::Map), op_usize(BuiltinMethod::Nth, 2)];
        let demand = source_demand(&ops, Demand::RESULT);
        assert_eq!(demand.pull, PullDemand::NthInput(2));
        assert_eq!(demand.value, ValueNeed::Whole);
    }

    #[test]
    fn scalar_slice_preserves_positional_demand() {
        let ops = [op(BuiltinMethod::Slice), op(BuiltinMethod::Last)];
        let demand = source_demand(&ops, Demand::RESULT);
        assert_eq!(demand.pull, PullDemand::LastInput(1));
        assert_eq!(demand.value, ValueNeed::Whole);
    }

    #[test]
    fn scalar_has_preserves_positional_predicate_demand() {
        let ops = [op(BuiltinMethod::Has), op(BuiltinMethod::Last)];
        let demand = source_demand(&ops, Demand::RESULT);
        assert_eq!(demand.pull, PullDemand::LastInput(1));
        assert_eq!(demand.value, ValueNeed::Predicate);
    }

    #[test]
    fn scalar_has_all_preserves_positional_predicate_demand() {
        let ops = [op(BuiltinMethod::HasAll), op(BuiltinMethod::Last)];
        let demand = source_demand(&ops, Demand::RESULT);
        assert_eq!(demand.pull, PullDemand::LastInput(1));
        assert_eq!(demand.value, ValueNeed::Predicate);
    }

    #[test]
    fn scalar_has_key_preserves_positional_predicate_demand() {
        let ops = [op(BuiltinMethod::HasKey), op(BuiltinMethod::Last)];
        let demand = source_demand(&ops, Demand::RESULT);
        assert_eq!(demand.pull, PullDemand::LastInput(1));
        assert_eq!(demand.value, ValueNeed::Predicate);
    }

    #[test]
    fn scalar_has_path_preserves_positional_predicate_demand() {
        let ops = [op(BuiltinMethod::HasPath), op(BuiltinMethod::Last)];
        let demand = source_demand(&ops, Demand::RESULT);
        assert_eq!(demand.pull, PullDemand::LastInput(1));
        assert_eq!(demand.value, ValueNeed::Predicate);
    }

    #[test]
    fn scalar_missing_preserves_positional_predicate_demand() {
        let ops = [op(BuiltinMethod::Missing), op(BuiltinMethod::Last)];
        let demand = source_demand(&ops, Demand::RESULT);
        assert_eq!(demand.pull, PullDemand::LastInput(1));
        assert_eq!(demand.value, ValueNeed::Predicate);
    }

    #[test]
    fn scalar_predicate_maps_do_not_force_payload_when_counted() {
        for method in [
            BuiltinMethod::Has,
            BuiltinMethod::HasAll,
            BuiltinMethod::HasKey,
            BuiltinMethod::HasPath,
            BuiltinMethod::Missing,
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
        ] {
            let ops = [op(method), op(BuiltinMethod::Count)];
            let demand = source_demand(&ops, Demand::RESULT);
            assert_eq!(demand.pull, PullDemand::All, "{method:?}");
            assert_eq!(demand.value, ValueNeed::CountOnly, "{method:?}");
        }
    }

    #[test]
    fn scalar_view_predicates_request_predicate_payload_for_positional_output() {
        for method in [
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
        ] {
            let ops = [op(method), op(BuiltinMethod::Last)];
            let demand = source_demand(&ops, Demand::RESULT);
            assert_eq!(demand.pull, PullDemand::LastInput(1), "{method:?}");
            assert_eq!(demand.value, ValueNeed::Predicate, "{method:?}");
        }
    }

    #[test]
    fn filter_nth_falls_back_to_all_input() {
        let ops = [op(BuiltinMethod::Filter), op_usize(BuiltinMethod::Nth, 2)];
        let demand = source_demand(&ops, Demand::RESULT);
        assert_eq!(demand.pull, PullDemand::All);
        assert_eq!(demand.value, ValueNeed::Whole);
    }

    #[test]
    fn take_filter_first_caps_upstream_to_take_bound() {
        let ops = [
            op(BuiltinMethod::Map),
            op_usize(BuiltinMethod::Take, 3),
            op(BuiltinMethod::Filter),
            op(BuiltinMethod::First),
        ];
        let steps = propagate_demands(&ops, Demand::RESULT);
        assert_eq!(steps[0].upstream.pull, PullDemand::FirstInput(3));
        assert_eq!(
            source_demand(&ops, Demand::RESULT).pull,
            PullDemand::FirstInput(3)
        );
    }

    #[test]
    fn filter_take_collect_scans_until_take_outputs() {
        let ops = [op(BuiltinMethod::Filter), op_usize(BuiltinMethod::Take, 3)];
        let demand = source_demand(&ops, Demand::RESULT);
        assert_eq!(demand.pull, PullDemand::UntilOutput(3));
    }

    #[test]
    fn compact_and_remove_are_filter_like() {
        let ops = [op(BuiltinMethod::Compact), op(BuiltinMethod::First)];
        let demand = source_demand(&ops, Demand::RESULT);
        assert_eq!(demand.pull, PullDemand::UntilOutput(1));
        assert_eq!(demand.value, ValueNeed::Whole);

        let ops = [op(BuiltinMethod::Remove), op(BuiltinMethod::Last)];
        let demand = source_demand(&ops, Demand::RESULT);
        assert_eq!(demand.pull, PullDemand::LastInput(1));
        assert_eq!(demand.value, ValueNeed::Whole);
        assert!(demand.order);
    }

    #[test]
    fn find_first_is_filter_like_before_first_sink() {
        let ops = [op(BuiltinMethod::FindFirst)];
        let demand = source_demand(&ops, Demand::first(ValueNeed::Whole));
        assert_eq!(demand.pull, PullDemand::UntilOutput(1));
        assert_eq!(demand.value, ValueNeed::Whole);
    }

    #[test]
    fn take_while_take_collect_needs_only_input_prefix() {
        let ops = [
            op(BuiltinMethod::TakeWhile),
            op_usize(BuiltinMethod::Take, 3),
        ];
        let demand = source_demand(&ops, Demand::RESULT);
        assert_eq!(demand.pull, PullDemand::FirstInput(3));
        assert_eq!(demand.value, ValueNeed::Whole);
    }

    #[test]
    fn chunk_and_window_map_bounded_output_to_input_prefix() {
        let ops = [
            op_usize(BuiltinMethod::Chunk, 4),
            op_usize(BuiltinMethod::Take, 3),
        ];
        let demand = source_demand(&ops, Demand::RESULT);
        assert_eq!(demand.pull, PullDemand::FirstInput(12));
        assert_eq!(demand.value, ValueNeed::Whole);

        let ops = [
            op_usize(BuiltinMethod::Window, 4),
            op_usize(BuiltinMethod::Take, 3),
        ];
        let demand = source_demand(&ops, Demand::RESULT);
        assert_eq!(demand.pull, PullDemand::FirstInput(6));
        assert_eq!(demand.value, ValueNeed::Whole);

        let ops = [op_usize(BuiltinMethod::Window, 4), op(BuiltinMethod::Last)];
        let demand = source_demand(&ops, Demand::RESULT);
        assert_eq!(demand.pull, PullDemand::All);
    }

    #[test]
    fn expanding_builtins_are_full_input_barriers_for_positional_sinks() {
        for method in [
            BuiltinMethod::Flatten,
            BuiltinMethod::Explode,
            BuiltinMethod::Split,
            BuiltinMethod::Lines,
            BuiltinMethod::Words,
            BuiltinMethod::Chars,
            BuiltinMethod::CharsOf,
            BuiltinMethod::Bytes,
        ] {
            let ops = [op(method), op(BuiltinMethod::Last)];
            let demand = source_demand(&ops, Demand::RESULT);
            assert_eq!(demand.pull, PullDemand::All, "{method:?}");
            assert_eq!(demand.value, ValueNeed::Whole, "{method:?}");
            assert!(demand.order, "{method:?}");
        }
    }

    #[test]
    fn barrier_builtins_request_full_ordered_input() {
        for method in [
            BuiltinMethod::Append,
            BuiltinMethod::Prepend,
            BuiltinMethod::Diff,
            BuiltinMethod::Intersect,
            BuiltinMethod::Union,
            BuiltinMethod::Join,
            BuiltinMethod::Zip,
            BuiltinMethod::ZipLongest,
            BuiltinMethod::Fold,
        ] {
            let ops = [op(method), op(BuiltinMethod::Last)];
            let demand = source_demand(&ops, Demand::RESULT);
            assert_eq!(demand.pull, PullDemand::All, "{method:?}");
            assert_eq!(demand.value, ValueNeed::Whole, "{method:?}");
            assert!(demand.order, "{method:?}");
        }
    }

    #[test]
    fn count_does_not_need_whole_values() {
        let ops = [op(BuiltinMethod::Map), op(BuiltinMethod::Count)];
        let demand = source_demand(&ops, Demand::RESULT);
        assert_eq!(demand.pull, PullDemand::All);
        assert_eq!(demand.value, ValueNeed::CountOnly);

        let ops = [op(BuiltinMethod::Count)];
        let demand = source_demand(&ops, Demand::RESULT);
        assert_eq!(demand.value, ValueNeed::CountOnly);

        let ops = [op(BuiltinMethod::Map), op(BuiltinMethod::Len)];
        let demand = source_demand(&ops, Demand::RESULT);
        assert_eq!(demand.pull, PullDemand::All);
        assert_eq!(demand.value, ValueNeed::CountOnly);

        let ops = [op(BuiltinMethod::Slice), op(BuiltinMethod::Count)];
        let demand = source_demand(&ops, Demand::RESULT);
        assert_eq!(demand.value, ValueNeed::CountOnly);
    }
}
