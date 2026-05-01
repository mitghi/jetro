use crate::builtins::{BuiltinNumericReducer, BuiltinSelectionPosition, BuiltinSinkAccumulator};
use crate::chain_ir::PullDemand;
use crate::composed as cmp;
use crate::value::Val;

use super::Sink;

pub(super) fn run(
    sink: &Sink,
    rows: &[Val],
    chain: &dyn cmp::Stage,
    demand: PullDemand,
) -> Option<Val> {
    let out = match sink {
        Sink::Collect => cmp::run_pipeline_with_demand::<cmp::CollectSink>(rows, chain, demand),
        Sink::Reducer(_) | Sink::Terminal(_) => match sink.builtin_sink_spec()?.accumulator {
            BuiltinSinkAccumulator::Count => {
                cmp::run_pipeline_with_demand::<cmp::CountSink>(rows, chain, demand)
            }
            BuiltinSinkAccumulator::Numeric => match numeric_reducer(sink)? {
                BuiltinNumericReducer::Sum => {
                    cmp::run_pipeline_with_demand::<cmp::SumSink>(rows, chain, demand)
                }
                BuiltinNumericReducer::Min => {
                    cmp::run_pipeline_with_demand::<cmp::MinSink>(rows, chain, demand)
                }
                BuiltinNumericReducer::Max => {
                    cmp::run_pipeline_with_demand::<cmp::MaxSink>(rows, chain, demand)
                }
                BuiltinNumericReducer::Avg => {
                    cmp::run_pipeline_with_demand::<cmp::AvgSink>(rows, chain, demand)
                }
            },
            BuiltinSinkAccumulator::SelectOne(BuiltinSelectionPosition::First) => {
                cmp::run_pipeline_with_demand::<cmp::FirstSink>(rows, chain, demand)
            }
            BuiltinSinkAccumulator::SelectOne(BuiltinSelectionPosition::Last) => {
                cmp::run_pipeline_with_demand::<cmp::LastSink>(rows, chain, demand)
            }
        },
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
        Sink::Reducer(_) | Sink::Terminal(_) => match sink.builtin_sink_spec()?.accumulator {
            BuiltinSinkAccumulator::Count => {
                cmp::run_pipeline_owned_iter_with_demand::<cmp::CountSink, _>(rows, chain, demand)
            }
            BuiltinSinkAccumulator::Numeric => match numeric_reducer(sink)? {
                BuiltinNumericReducer::Sum => {
                    cmp::run_pipeline_owned_iter_with_demand::<cmp::SumSink, _>(rows, chain, demand)
                }
                BuiltinNumericReducer::Min => {
                    cmp::run_pipeline_owned_iter_with_demand::<cmp::MinSink, _>(rows, chain, demand)
                }
                BuiltinNumericReducer::Max => {
                    cmp::run_pipeline_owned_iter_with_demand::<cmp::MaxSink, _>(rows, chain, demand)
                }
                BuiltinNumericReducer::Avg => {
                    cmp::run_pipeline_owned_iter_with_demand::<cmp::AvgSink, _>(rows, chain, demand)
                }
            },
            BuiltinSinkAccumulator::SelectOne(BuiltinSelectionPosition::First) => {
                cmp::run_pipeline_owned_iter_with_demand::<cmp::FirstSink, _>(rows, chain, demand)
            }
            BuiltinSinkAccumulator::SelectOne(BuiltinSelectionPosition::Last) => {
                cmp::run_pipeline_owned_iter_with_demand::<cmp::LastSink, _>(rows, chain, demand)
            }
        },
        Sink::ApproxCountDistinct => return None,
    };

    Some(out)
}

fn numeric_reducer(sink: &Sink) -> Option<BuiltinNumericReducer> {
    sink.reducer_spec()?.method()?.spec().numeric_reducer
}
