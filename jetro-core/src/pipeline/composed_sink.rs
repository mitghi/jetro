use crate::chain_ir::PullDemand;
use crate::composed as cmp;
use crate::value::Val;

use super::{NumOp, ReducerOp, Sink};

pub(super) fn run(
    sink: &Sink,
    rows: &[Val],
    chain: &dyn cmp::Stage,
    demand: PullDemand,
) -> Option<Val> {
    let out = match sink {
        Sink::Collect => cmp::run_pipeline_with_demand::<cmp::CollectSink>(rows, chain, demand),
        Sink::Reducer(spec) => match spec.op {
            ReducerOp::Count => {
                cmp::run_pipeline_with_demand::<cmp::CountSink>(rows, chain, demand)
            }
            ReducerOp::Numeric(NumOp::Sum) => {
                cmp::run_pipeline_with_demand::<cmp::SumSink>(rows, chain, demand)
            }
            ReducerOp::Numeric(NumOp::Min) => {
                cmp::run_pipeline_with_demand::<cmp::MinSink>(rows, chain, demand)
            }
            ReducerOp::Numeric(NumOp::Max) => {
                cmp::run_pipeline_with_demand::<cmp::MaxSink>(rows, chain, demand)
            }
            ReducerOp::Numeric(NumOp::Avg) => {
                cmp::run_pipeline_with_demand::<cmp::AvgSink>(rows, chain, demand)
            }
        },
        Sink::First(_) => cmp::run_pipeline_with_demand::<cmp::FirstSink>(rows, chain, demand),
        Sink::Last(_) => cmp::run_pipeline_with_demand::<cmp::LastSink>(rows, chain, demand),
        Sink::ApproxCountDistinct => return None,
    };

    Some(out)
}

pub(super) fn run_owned_iter<I>(
    sink: &Sink,
    rows: I,
    chain: &dyn cmp::Stage,
    demand: PullDemand,
) -> Option<Val>
where
    I: IntoIterator<Item = Val>,
{
    let out = match sink {
        Sink::Collect => {
            cmp::run_pipeline_owned_iter_with_demand::<cmp::CollectSink, _>(rows, chain, demand)
        }
        Sink::Reducer(spec) => match spec.op {
            ReducerOp::Count => {
                cmp::run_pipeline_owned_iter_with_demand::<cmp::CountSink, _>(rows, chain, demand)
            }
            ReducerOp::Numeric(NumOp::Sum) => {
                cmp::run_pipeline_owned_iter_with_demand::<cmp::SumSink, _>(rows, chain, demand)
            }
            ReducerOp::Numeric(NumOp::Min) => {
                cmp::run_pipeline_owned_iter_with_demand::<cmp::MinSink, _>(rows, chain, demand)
            }
            ReducerOp::Numeric(NumOp::Max) => {
                cmp::run_pipeline_owned_iter_with_demand::<cmp::MaxSink, _>(rows, chain, demand)
            }
            ReducerOp::Numeric(NumOp::Avg) => {
                cmp::run_pipeline_owned_iter_with_demand::<cmp::AvgSink, _>(rows, chain, demand)
            }
        },
        Sink::First(_) => {
            cmp::run_pipeline_owned_iter_with_demand::<cmp::FirstSink, _>(rows, chain, demand)
        }
        Sink::Last(_) => {
            cmp::run_pipeline_owned_iter_with_demand::<cmp::LastSink, _>(rows, chain, demand)
        }
        Sink::ApproxCountDistinct => return None,
    };

    Some(out)
}
