//! Canonical chain operator semantics.
//!
//! This module is intentionally metadata-first. It does not encode
//! pairwise rewrites such as `filter + first`; instead each operator
//! declares how downstream demand flows to its input. Executors can use
//! the resulting annotations to stop pull loops without inventing a
//! fused operator for every adjacent pair.

#![allow(dead_code)]

use crate::{
    builtin_registry::{propagate_demand as propagate_builtin_demand, BuiltinDemandArg, BuiltinId},
    builtins::BuiltinMethod,
};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueNeed {
    None,
    Predicate,
    Whole,
    Numeric,
}

impl ValueNeed {
    pub(crate) fn merge(self, other: Self) -> Self {
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
    /// Pull only the first N input items. No later input can affect
    /// the requested downstream result.
    FirstInput(usize),
    /// Pull until N output items have been emitted by the current
    /// operator. For filtering operators this may require scanning more
    /// than N inputs, so it must not be collapsed to `FirstInput(N)`.
    UntilOutput(usize),
}

impl PullDemand {
    pub(crate) fn cap_inputs(self, n: usize) -> Self {
        match self {
            PullDemand::All | PullDemand::UntilOutput(_) => PullDemand::FirstInput(n),
            PullDemand::FirstInput(m) => PullDemand::FirstInput(m.min(n)),
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
            pull: PullDemand::FirstInput(1),
            value,
            order: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChainOp {
    Builtin {
        id: BuiltinId,
        demand_arg: BuiltinDemandArg,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpSpec {
    pub input: ValueKind,
    pub output: ValueKind,
    pub cardinality: Cardinality,
    pub preserves_order: bool,
}

impl ChainOp {
    pub fn builtin(method: BuiltinMethod) -> Self {
        Self::Builtin {
            id: BuiltinId::from_method(method),
            demand_arg: BuiltinDemandArg::None,
        }
    }

    pub fn builtin_usize(method: BuiltinMethod, n: usize) -> Self {
        Self::Builtin {
            id: BuiltinId::from_method(method),
            demand_arg: BuiltinDemandArg::Usize(n),
        }
    }

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

    pub fn propagate_demand(&self, downstream: Demand) -> Demand {
        match self {
            ChainOp::Builtin { id, demand_arg } => {
                propagate_builtin_demand(*id, *demand_arg, downstream)
            }
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
    fn filter_last_requires_all_ordered_input() {
        let ops = [op(BuiltinMethod::Filter), op(BuiltinMethod::Last)];
        let demand = source_demand(&ops, Demand::RESULT);
        assert_eq!(demand.pull, PullDemand::All);
        assert_eq!(demand.value, ValueNeed::Whole);
        assert!(demand.order);
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
