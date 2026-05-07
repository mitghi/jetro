//! Shared demand model and backward propagation helpers.
//!
//! Demand is a planning concern, not parser syntax: sinks describe how much
//! input and value payload they need, and stage/operator adapters translate
//! that demand backward toward the source.

#![allow(dead_code)]

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

/// Adapter trait implemented by whichever operator representation a planner
/// uses for demand propagation.
pub trait DemandOperator {
    /// Propagate `downstream` demand through this operator, returning the
    /// upstream demand that its source must satisfy.
    fn propagate_demand(&self, downstream: Demand) -> Demand;
}

/// A single annotated step produced by `propagate_demands`, recording an
/// operator alongside the demand it receives and the demand it places upstream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DemandStep<Op> {
    /// The operator at this position in the chain.
    pub op: Op,
    /// Demand flowing into this operator from the downstream consumer.
    pub downstream: Demand,
    /// Demand this operator forwards to its upstream source.
    pub upstream: Demand,
}

/// Walk `ops` in reverse and compute each operator's upstream demand given
/// `final_demand` at the sink, returning annotated `DemandStep`s in forward order.
pub fn propagate_demands<Op>(ops: &[Op], final_demand: Demand) -> Vec<DemandStep<Op>>
where
    Op: DemandOperator + Clone,
{
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
pub fn source_demand<Op>(ops: &[Op], final_demand: Demand) -> Demand
where
    Op: DemandOperator,
{
    ops.iter()
        .rev()
        .fold(final_demand, |demand, op| op.propagate_demand(demand))
}
