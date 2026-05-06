//! Operator cardinality and demand-flow metadata for the chain IR.
//!
//! This module is metadata-first: each operator declares how downstream
//! demand propagates to its input, rather than encoding pairwise rewrites.
//! Executors use `PullDemand` to stop pull loops early without needing a
//! fused operator for every adjacent pair.

#![allow(dead_code)]

use crate::{
    builtins::registry::{propagate_demand as propagate_builtin_demand, BuiltinDemandArg, BuiltinId},
    builtins::BuiltinMethod,
};

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

/// Describes how much of each element's content a pipeline stage actually
/// needs to read, used to skip deserialisation or evaluation work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueNeed {
    /// The stage only needs to know the element exists; payload can be skipped.
    None,
    /// The stage evaluates a predicate and needs enough of the value to test it.
    Predicate,
    /// The stage only needs fields used by a projection.
    Projection,
    /// The full element value is required.
    Whole,
    /// Only the numeric interpretation of the element is needed (e.g. for `sum`).
    Numeric,
}

impl ValueNeed {
    /// Return the stricter of two `ValueNeed` values; `Whole` dominates all others.
    pub(crate) fn merge(self, other: Self) -> Self {
        use ValueNeed::*;
        match (self, other) {
            (Whole, _) | (_, Whole) => Whole,
            (Numeric, _) | (_, Numeric) => Numeric,
            (Projection, _) | (_, Projection) => Projection,
            (Predicate, _) | (_, Predicate) => Predicate,
            (None, None) => None,
        }
    }
}


/// Specifies how many input elements a stage must pull from its source to
/// satisfy a downstream consumer's limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PullDemand {
    /// Pull all available input elements without any limit.
    All,
    /// Pull at most the first `n` input elements regardless of how many outputs they produce.
    FirstInput(usize),
    /// Pull from the end of the input until `n` outputs have been produced.
    LastInput(usize),
    /// Pull the input element at zero-based index `i` when the source can seek to it.
    NthInput(usize),
    /// Pull input until exactly `n` output elements have been produced.
    UntilOutput(usize),
}

impl PullDemand {
    /// Return a `PullDemand` capped to at most `n` input elements,
    /// converting `All` or `UntilOutput` variants to `FirstInput(n)`.
    pub(crate) fn cap_inputs(self, n: usize) -> Self {
        match self {
            PullDemand::All | PullDemand::UntilOutput(_) | PullDemand::LastInput(_) => {
                PullDemand::FirstInput(n)
            }
            PullDemand::FirstInput(m) => PullDemand::FirstInput(m.min(n)),
            PullDemand::NthInput(i) => {
                if i < n {
                    PullDemand::NthInput(i)
                } else {
                    PullDemand::FirstInput(n)
                }
            }
        }
    }
}

/// Combined downstream demand annotation: how much to pull, what payload is
/// needed, and whether the consumer requires stable input ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Demand {
    /// How many upstream elements must be consumed.
    pub pull: PullDemand,
    /// How much of each element's payload is required.
    pub value: ValueNeed,
    /// Whether the consumer depends on elements arriving in their original order.
    pub order: bool,
}

impl Demand {
    /// The terminal demand used at the sink of a pipeline: pull everything,
    /// need whole values, and require ordering.
    pub const RESULT: Demand = Demand {
        pull: PullDemand::All,
        value: ValueNeed::Whole,
        order: true,
    };

    /// Construct a demand that pulls all input with the given value need and
    /// order requirement.
    pub fn all(value: ValueNeed) -> Self {
        Self {
            pull: PullDemand::All,
            value,
            order: true,
        }
    }

    /// Construct a demand that pulls only the first input element with the
    /// given value need, and no ordering requirement.
    pub fn first(value: ValueNeed) -> Self {
        Self {
            pull: PullDemand::FirstInput(1),
            value,
            order: false,
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

    /// Derive the static `OpSpec` for this operator by consulting the builtin
    /// category registry.
    pub fn spec(&self) -> OpSpec {
        match self {
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

    /// Propagate `downstream` demand through this operator, returning the
    /// upstream demand that its source must satisfy.
    pub fn propagate_demand(&self, downstream: Demand) -> Demand {
        match self {
            ChainOp::Builtin { id, demand_arg } => {
                propagate_builtin_demand(*id, *demand_arg, downstream)
            }
        }
    }
}

/// A single annotated step produced by `propagate_demands`, recording an
/// operator alongside the demand it receives and the demand it places upstream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DemandStep {
    /// The operator at this position in the chain.
    pub op: ChainOp,
    /// Demand flowing into this operator from the downstream consumer.
    pub downstream: Demand,
    /// Demand this operator forwards to its upstream source.
    pub upstream: Demand,
}

/// Walk `ops` in reverse and compute each operator's upstream demand given
/// `final_demand` at the sink, returning annotated `DemandStep`s in forward order.
pub fn propagate_demands(ops: &[ChainOp], final_demand: Demand) -> Vec<DemandStep> {
    let mut demand = final_demand;
    let mut out = Vec::with_capacity(ops.len());
    for op in ops.iter().rev() {
        let upstream = op.propagate_demand(demand);
        out.push(DemandStep {
            op: op.clone(),
            downstream: demand,
            upstream,
        });
        demand = upstream;
    }
    out.reverse();
    out
}

/// Fold demand propagation over `ops` from sink to source and return only
/// the final upstream demand without allocating intermediate `DemandStep`s.
pub fn source_demand(ops: &[ChainOp], final_demand: Demand) -> Demand {
    ops.iter()
        .rev()
        .fold(final_demand, |demand, op| op.propagate_demand(demand))
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
        let ops = [
            op(BuiltinMethod::Map),
            op_usize(BuiltinMethod::Nth, 2),
        ];
        let demand = source_demand(&ops, Demand::RESULT);
        assert_eq!(demand.pull, PullDemand::NthInput(2));
        assert_eq!(demand.value, ValueNeed::Whole);
    }

    #[test]
    fn filter_nth_falls_back_to_all_input() {
        let ops = [
            op(BuiltinMethod::Filter),
            op_usize(BuiltinMethod::Nth, 2),
        ];
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
