//! Canonical chain operator semantics.
//!
//! This module is intentionally metadata-first. It does not encode
//! pairwise rewrites such as `filter + first`; instead each operator
//! declares how downstream demand flows to its input. Executors can use
//! the resulting annotations to stop pull loops without inventing a
//! fused operator for every adjacent pair.

#![allow(dead_code)]

use crate::builtins::BuiltinMethod;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueKind {
    Any,
    Stream,
    Scalar,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cardinality {
    OneToOne,
    Filtering,
    Expanding,
    Bounded,
    Reducing,
    Barrier,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueNeed {
    None,
    Predicate,
    Whole,
    Numeric,
}

impl ValueNeed {
    fn merge(self, other: Self) -> Self {
        use ValueNeed::*;
        match (self, other) {
            (Whole, _) | (_, Whole) => Whole,
            (Numeric, _) | (_, Numeric) => Numeric,
            (Predicate, _) | (_, Predicate) => Predicate,
            (None, None) => None,
        }
    }
}

/// How much of an upstream stream is required.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PullDemand {
    /// Pull every input item.
    All,
    /// Pull at most N input items. This is a hard input bound.
    AtMost(usize),
    /// Pull until N output items have been emitted by the current
    /// operator. For filtering operators this may require scanning more
    /// than N inputs, so it must not be collapsed to `AtMost(N)`.
    UntilOutput(usize),
}

impl PullDemand {
    fn cap_inputs(self, n: usize) -> Self {
        match self {
            PullDemand::All | PullDemand::UntilOutput(_) => PullDemand::AtMost(n),
            PullDemand::AtMost(m) => PullDemand::AtMost(m.min(n)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Demand {
    pub pull: PullDemand,
    pub value: ValueNeed,
    pub order: bool,
}

impl Demand {
    pub const RESULT: Demand = Demand {
        pull: PullDemand::All,
        value: ValueNeed::Whole,
        order: true,
    };

    pub fn all(value: ValueNeed) -> Self {
        Self {
            pull: PullDemand::All,
            value,
            order: true,
        }
    }

    pub fn first(value: ValueNeed) -> Self {
        Self {
            pull: PullDemand::UntilOutput(1),
            value,
            order: false,
        }
    }

    fn with_value(self, value: ValueNeed) -> Self {
        Self {
            value: self.value.merge(value),
            ..self
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChainOp {
    Filter,
    Map,
    FlatMap,
    Take(usize),
    Skip(usize),
    First,
    Last,
    Count,
    Sum,
    Avg,
    Min,
    Max,
    Builtin(BuiltinMethod),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpSpec {
    pub input: ValueKind,
    pub output: ValueKind,
    pub cardinality: Cardinality,
    pub preserves_order: bool,
}

impl ChainOp {
    pub fn spec(&self) -> OpSpec {
        use Cardinality::*;
        use ChainOp::*;
        match self {
            Filter => OpSpec {
                input: ValueKind::Stream,
                output: ValueKind::Stream,
                cardinality: Filtering,
                preserves_order: true,
            },
            Map => OpSpec {
                input: ValueKind::Stream,
                output: ValueKind::Stream,
                cardinality: OneToOne,
                preserves_order: true,
            },
            FlatMap => OpSpec {
                input: ValueKind::Stream,
                output: ValueKind::Stream,
                cardinality: Expanding,
                preserves_order: true,
            },
            Take(_) | Skip(_) => OpSpec {
                input: ValueKind::Stream,
                output: ValueKind::Stream,
                cardinality: Bounded,
                preserves_order: true,
            },
            First | Last | Count | Sum | Avg | Min | Max => OpSpec {
                input: ValueKind::Stream,
                output: ValueKind::Scalar,
                cardinality: Reducing,
                preserves_order: true,
            },
            Builtin(_) => OpSpec {
                input: ValueKind::Any,
                output: ValueKind::Any,
                cardinality: OneToOne,
                preserves_order: true,
            },
        }
    }

    pub fn propagate_demand(&self, downstream: Demand) -> Demand {
        use ChainOp::*;
        use PullDemand::*;
        use ValueNeed::*;
        match self {
            Filter => Demand {
                pull: match downstream.pull {
                    All => All,
                    AtMost(n) | UntilOutput(n) => UntilOutput(n),
                },
                value: downstream.value.merge(Predicate),
                order: downstream.order,
            },
            Map => downstream.with_value(Whole),
            FlatMap => Demand::all(Whole),
            Take(n) => Demand {
                pull: downstream.pull.cap_inputs(*n),
                ..downstream
            },
            Skip(n) => Demand {
                pull: match downstream.pull {
                    AtMost(m) => AtMost(n.saturating_add(m)),
                    All | UntilOutput(_) => All,
                },
                ..downstream
            },
            First => Demand::first(Whole),
            Last => Demand {
                pull: All,
                value: Whole,
                order: true,
            },
            Count => Demand {
                pull: All,
                value: None,
                order: false,
            },
            Sum | Avg | Min | Max => Demand {
                pull: All,
                value: Numeric,
                order: false,
            },
            Builtin(_) => downstream,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DemandStep {
    pub op: ChainOp,
    pub downstream: Demand,
    pub upstream: Demand,
}

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

pub fn source_demand(ops: &[ChainOp], final_demand: Demand) -> Demand {
    ops.iter()
        .rev()
        .fold(final_demand, |demand, op| op.propagate_demand(demand))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_first_scans_until_one_output() {
        let ops = [ChainOp::Filter, ChainOp::First];
        let demand = source_demand(&ops, Demand::RESULT);
        assert_eq!(demand.pull, PullDemand::UntilOutput(1));
        assert_eq!(demand.value, ValueNeed::Whole);
    }

    #[test]
    fn filter_last_requires_all_ordered_input() {
        let ops = [ChainOp::Filter, ChainOp::Last];
        let demand = source_demand(&ops, Demand::RESULT);
        assert_eq!(demand.pull, PullDemand::All);
        assert_eq!(demand.value, ValueNeed::Whole);
        assert!(demand.order);
    }

    #[test]
    fn take_filter_first_caps_upstream_to_take_bound() {
        let ops = [
            ChainOp::Map,
            ChainOp::Take(3),
            ChainOp::Filter,
            ChainOp::First,
        ];
        let steps = propagate_demands(&ops, Demand::RESULT);
        assert_eq!(steps[0].upstream.pull, PullDemand::AtMost(3));
        assert_eq!(
            source_demand(&ops, Demand::RESULT).pull,
            PullDemand::AtMost(3)
        );
    }

    #[test]
    fn filter_take_collect_scans_until_take_outputs() {
        let ops = [ChainOp::Filter, ChainOp::Take(3)];
        let demand = source_demand(&ops, Demand::RESULT);
        assert_eq!(demand.pull, PullDemand::UntilOutput(3));
    }

    #[test]
    fn count_does_not_need_whole_values() {
        let ops = [ChainOp::Map, ChainOp::Count];
        let demand = source_demand(&ops, Demand::RESULT);
        assert_eq!(demand.pull, PullDemand::All);
        assert_eq!(demand.value, ValueNeed::Whole);

        let ops = [ChainOp::Count];
        let demand = source_demand(&ops, Demand::RESULT);
        assert_eq!(demand.value, ValueNeed::None);
    }
}
