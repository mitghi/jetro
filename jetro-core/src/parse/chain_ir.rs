//! Operator cardinality metadata and demand-flow adapters for the chain IR.
//!
//! The shared demand model lives in `plan::demand`; this module adapts the
//! parser-facing chain operator representation to that model.

#![allow(dead_code)]

use crate::{
    builtins::registry::{
        propagate_demand as propagate_builtin_demand, BuiltinDemandArg, BuiltinId,
    },
    builtins::BuiltinMethod,
    plan::demand::{Demand, DemandOperator},
};

#[cfg(test)]
use crate::plan::demand::{propagate_demands, source_demand, PullDemand, ValueNeed};

/// Describes whether a pipeline slot carries a homogeneous stream, a single
/// scalar result, or an unconstrained mix of values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueKind {
    /// No constraint on the kind of value in this slot.
    Any,
    /// The slot holds a sequence of values produced by a streaming operator.
    Stream,
    /// The slot holds exactly one scalar value (e.g. the result of `count`).
    Scalar,
}

/// Output-cardinality classification for a pipeline operator. Used by the
/// demand-propagation pass to decide how many input elements a downstream
/// consumer's limit translates to upstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cardinality {
    /// Each input element produces exactly one output element (e.g. `map`).
    OneToOne,
    /// Some input elements are dropped; outputs ≤ inputs (e.g. `filter`).
    Filtering,
    /// Each input element may produce multiple outputs (e.g. `flat_map`).
    Expanding,
    /// Output count is bounded by a fixed argument, not the input length.
    Bounded,
    /// Collapses all inputs into a single output (e.g. `sum`, `count`).
    Reducing,
    /// Requires the entire input before producing any output (e.g. `sort`).
    Barrier,
}

/// Convert from the `BuiltinCardinality` enum used in `builtins.rs` to the
/// chain IR's `Cardinality`, keeping the two representations in sync.
impl From<crate::builtins::BuiltinCardinality> for Cardinality {
    fn from(value: crate::builtins::BuiltinCardinality) -> Self {
        match value {
            crate::builtins::BuiltinCardinality::OneToOne => Self::OneToOne,
            crate::builtins::BuiltinCardinality::Filtering => Self::Filtering,
            crate::builtins::BuiltinCardinality::Expanding => Self::Expanding,
            crate::builtins::BuiltinCardinality::Bounded => Self::Bounded,
            crate::builtins::BuiltinCardinality::Reducing => Self::Reducing,
            crate::builtins::BuiltinCardinality::Barrier => Self::Barrier,
        }
    }
}

/// A single operator node in the chain IR, carrying identity and demand
/// metadata for each step in a composed pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChainOp {
    /// A registered builtin with a stable `BuiltinId` and optional numeric demand argument.
    Builtin {
        /// Stable numeric ID identifying which builtin this operator represents.
        id: BuiltinId,
        /// Optional count argument used by demand-propagation for `take`/`skip`.
        demand_arg: BuiltinDemandArg,
    },
    /// A `match` expression participating in a streaming chain. Match
    /// behaves as one of three shapes depending on how the surrounding
    /// pipeline uses its result, captured here as `MatchRole`. Demand
    /// propagation treats each role like its closest builtin analogue
    /// (`Filter`, `Map`, `FlatMap`).
    Match {
        /// How the surrounding chain consumes match output.
        role: MatchRole,
    },
}

/// The shape a `match` expression takes when embedded in a streaming
/// pipeline. The role determines its demand-flow behaviour and selects
/// the streaming analogue (filter / map / flat-map) used during planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchRole {
    /// Boolean predicate: arm bodies are `true` / `false`. Drops rows that
    /// do not match (or whose arm body is falsy). Demand law is
    /// `Cardinality::Filtering`.
    Predicate,
    /// Single-value transform: every input row yields exactly one output.
    /// Demand law is `Cardinality::OneToOne`.
    Transform,
    /// Each input row produces zero or more output rows (an arm body
    /// returns an iterable). Demand law is `Cardinality::Expanding`.
    /// Reserved for future pipeline integration; current arm bodies
    /// always produce a single value, so this role is not yet emitted.
    Multi,
}

/// Static specification describing the kind of values a `ChainOp` consumes
/// and produces, along with its cardinality and ordering guarantees.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpSpec {
    /// Kind of values the operator reads from its source.
    pub input: ValueKind,
    /// Kind of values the operator writes to the next stage.
    pub output: ValueKind,
    /// Relationship between input count and output count.
    pub cardinality: Cardinality,
    /// Whether output elements appear in the same order as their inputs.
    pub preserves_order: bool,
}

impl ChainOp {
    /// Construct a `ChainOp::Builtin` from a `BuiltinMethod` with no demand argument.
    pub fn builtin(method: BuiltinMethod) -> Self {
        Self::Builtin {
            id: BuiltinId::from_method(method),
            demand_arg: BuiltinDemandArg::None,
        }
    }

    /// Construct a `ChainOp::Builtin` from a `BuiltinMethod` with a `Usize(n)` demand argument.
    pub fn builtin_usize(method: BuiltinMethod, n: usize) -> Self {
        Self::Builtin {
            id: BuiltinId::from_method(method),
            demand_arg: BuiltinDemandArg::Usize(n),
        }
    }

    /// Construct a `ChainOp::Match` for the given `role`.
    pub fn match_role(role: MatchRole) -> Self {
        Self::Match { role }
    }

    /// Derive the static `OpSpec` for this operator by consulting the builtin
    /// category registry.
    pub fn spec(&self) -> OpSpec {
        match self {
            ChainOp::Match { role } => {
                let cardinality = match role {
                    MatchRole::Predicate => Cardinality::Filtering,
                    MatchRole::Transform => Cardinality::OneToOne,
                    MatchRole::Multi => Cardinality::Expanding,
                };
                OpSpec {
                    input: ValueKind::Stream,
                    output: ValueKind::Stream,
                    cardinality,
                    preserves_order: true,
                }
            }
            ChainOp::Builtin { id, .. } => {
                use crate::builtins::BuiltinCategory as Cat;

                let Some(method) = id.method() else {
                    return OpSpec {
                        input: ValueKind::Any,
                        output: ValueKind::Any,
                        cardinality: Cardinality::OneToOne,
                        preserves_order: true,
                    };
                };
                let spec = method.spec();
                let input = match spec.category {
                    Cat::StreamingOneToOne
                    | Cat::StreamingFilter
                    | Cat::StreamingExpand
                    | Cat::Reducer
                    | Cat::Positional
                    | Cat::Barrier
                    | Cat::Relational => ValueKind::Stream,
                    _ => ValueKind::Any,
                };
                let output = match spec.category {
                    Cat::Reducer | Cat::Positional => ValueKind::Scalar,
                    Cat::StreamingOneToOne | Cat::StreamingFilter | Cat::StreamingExpand => {
                        ValueKind::Stream
                    }
                    _ => ValueKind::Any,
                };
                OpSpec {
                    input,
                    output,
                    cardinality: spec.cardinality.into(),
                    preserves_order: spec.view_native
                        || !matches!(
                            spec.cardinality,
                            crate::builtins::BuiltinCardinality::Barrier
                        ),
                }
            }
        }
    }

    /// Propagate demand using the shared planner demand model.
    pub fn propagate_demand(&self, downstream: Demand) -> Demand {
        <Self as DemandOperator>::propagate_demand(self, downstream)
    }
}

impl DemandOperator for ChainOp {
    fn propagate_demand(&self, downstream: Demand) -> Demand {
        match self {
            ChainOp::Match { role } => match role {
                // Predicate match drops rows: downstream demand of N
                // outputs requires scanning until N pass the predicate,
                // matching `Filter` semantics. The value need passes
                // through unchanged because the body re-emits the
                // matched element (or maps it to a body value when
                // used as a Transform).
                MatchRole::Predicate => Demand {
                    pull: match downstream.pull {
                        crate::plan::demand::PullDemand::FirstInput(n) => {
                            // Filter cannot guarantee N outputs from N
                            // inputs; widen to scan-until-output.
                            crate::plan::demand::PullDemand::UntilOutput(n)
                        }
                        crate::plan::demand::PullDemand::UntilOutput(n) => {
                            crate::plan::demand::PullDemand::UntilOutput(n)
                        }
                        other => other,
                    },
                    value: downstream.value,
                    order: downstream.order,
                },
                // Transform match is 1:1 — demand passes through.
                MatchRole::Transform => downstream,
                // Multi match expands rows; widen `FirstInput(n)` to
                // `All` since one input may produce many outputs.
                MatchRole::Multi => Demand {
                    pull: match downstream.pull {
                        crate::plan::demand::PullDemand::FirstInput(_)
                        | crate::plan::demand::PullDemand::UntilOutput(_) => {
                            crate::plan::demand::PullDemand::All
                        }
                        other => other,
                    },
                    value: downstream.value,
                    order: downstream.order,
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
        let spec = match_op(MatchRole::Predicate).spec();
        assert_eq!(spec.cardinality, Cardinality::Filtering);
        assert_eq!(spec.input, ValueKind::Stream);
        assert_eq!(spec.output, ValueKind::Stream);
        assert!(spec.preserves_order);
    }

    #[test]
    fn match_transform_classifies_as_map() {
        let spec = match_op(MatchRole::Transform).spec();
        assert_eq!(spec.cardinality, Cardinality::OneToOne);
    }

    #[test]
    fn match_multi_classifies_as_flat_map() {
        let spec = match_op(MatchRole::Multi).spec();
        assert_eq!(spec.cardinality, Cardinality::Expanding);
    }

    #[test]
    fn match_predicate_first_scans_until_one_output() {
        // `match (Predicate) | first` ≡ `filter | first`: scan until one
        // output is produced.
        let ops = [match_op(MatchRole::Predicate), op(BuiltinMethod::First)];
        let demand = source_demand(&ops, Demand::RESULT);
        assert_eq!(demand.pull, PullDemand::UntilOutput(1));
        assert_eq!(demand.value, ValueNeed::Whole);
    }

    #[test]
    fn match_transform_take_caps_upstream() {
        // `match (Transform) | take(3)` is 1:1 from match's POV, so the
        // upstream cap propagates exactly.
        let ops = [
            match_op(MatchRole::Transform),
            op_usize(BuiltinMethod::Take, 3),
        ];
        let demand = source_demand(&ops, Demand::RESULT);
        assert_eq!(demand.pull, PullDemand::FirstInput(3));
    }

    #[test]
    fn match_predicate_take_widens_to_scan() {
        // `match (Predicate) | take(3)`: cannot bound input by 3 since
        // some rows are dropped. Widens to `UntilOutput(3)`.
        let ops = [
            match_op(MatchRole::Predicate),
            op_usize(BuiltinMethod::Take, 3),
        ];
        let demand = source_demand(&ops, Demand::RESULT);
        assert_eq!(demand.pull, PullDemand::UntilOutput(3));
    }

    #[test]
    fn filter_first_scans_until_one_output() {
        let ops = [op(BuiltinMethod::Filter), op(BuiltinMethod::First)];
        let demand = source_demand(&ops, Demand::RESULT);
        assert_eq!(demand.pull, PullDemand::UntilOutput(1));
        assert_eq!(demand.value, ValueNeed::Whole);
    }

    #[test]
    fn filter_last_requests_reverse_until_output() {
        let ops = [op(BuiltinMethod::Filter), op(BuiltinMethod::Last)];
        let demand = source_demand(&ops, Demand::RESULT);
        assert_eq!(demand.pull, PullDemand::LastInput(1));
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
    fn count_does_not_need_whole_values() {
        let ops = [op(BuiltinMethod::Map), op(BuiltinMethod::Count)];
        let demand = source_demand(&ops, Demand::RESULT);
        assert_eq!(demand.pull, PullDemand::All);
        assert_eq!(demand.value, ValueNeed::Whole);

        let ops = [op(BuiltinMethod::Count)];
        let demand = source_demand(&ops, Demand::RESULT);
        assert_eq!(demand.value, ValueNeed::None);
    }
}
