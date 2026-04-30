use crate::chain_ir::PullDemand;
use crate::composed as cmp;
use crate::value::Val;

use super::{NumOp, Sink};

pub(super) fn run(
    sink: &Sink,
    rows: &[Val],
    chain: &dyn cmp::Stage,
    demand: PullDemand,
) -> Option<Val> {
    let out = match sink {
        Sink::Collect => cmp::run_pipeline_with_demand::<cmp::CollectSink>(rows, chain, demand),
        Sink::Count(_) => cmp::run_pipeline_with_demand::<cmp::CountSink>(rows, chain, demand),
        Sink::Numeric(numeric) if numeric.project.is_some() => return None,
        Sink::Numeric(numeric) => match numeric.op {
            NumOp::Sum => cmp::run_pipeline_with_demand::<cmp::SumSink>(rows, chain, demand),
            NumOp::Min => cmp::run_pipeline_with_demand::<cmp::MinSink>(rows, chain, demand),
            NumOp::Max => cmp::run_pipeline_with_demand::<cmp::MaxSink>(rows, chain, demand),
            NumOp::Avg => cmp::run_pipeline_with_demand::<cmp::AvgSink>(rows, chain, demand),
        },
        Sink::First(_) => cmp::run_pipeline_with_demand::<cmp::FirstSink>(rows, chain, demand),
        Sink::Last(_) => cmp::run_pipeline_with_demand::<cmp::LastSink>(rows, chain, demand),
        Sink::ApproxCountDistinct => return None,
    };

    Some(out)
}
