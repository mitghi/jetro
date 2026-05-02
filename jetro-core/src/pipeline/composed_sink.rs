use crate::builtins::{BuiltinNumericReducer, BuiltinSelectionPosition, BuiltinSinkAccumulator};
use crate::chain_ir::PullDemand;
use crate::composed as cmp;
use crate::value::Val;

use super::Sink;

macro_rules! run_composed_sink {
    ($runner:ident, $rows:expr, $chain:expr, $demand:expr, $sink:expr) => {
        match $sink.builtin_sink_spec()?.accumulator {
            BuiltinSinkAccumulator::Count => cmp::$runner::<cmp::CountSink>($rows, $chain, $demand),
            BuiltinSinkAccumulator::Numeric => match numeric_reducer($sink)? {
                BuiltinNumericReducer::Sum => cmp::$runner::<cmp::SumSink>($rows, $chain, $demand),
                BuiltinNumericReducer::Min => cmp::$runner::<cmp::MinSink>($rows, $chain, $demand),
                BuiltinNumericReducer::Max => cmp::$runner::<cmp::MaxSink>($rows, $chain, $demand),
                BuiltinNumericReducer::Avg => cmp::$runner::<cmp::AvgSink>($rows, $chain, $demand),
            },
            BuiltinSinkAccumulator::SelectOne(BuiltinSelectionPosition::First) => {
                cmp::$runner::<cmp::FirstSink>($rows, $chain, $demand)
            }
            BuiltinSinkAccumulator::SelectOne(BuiltinSelectionPosition::Last) => {
                cmp::$runner::<cmp::LastSink>($rows, $chain, $demand)
            }
            BuiltinSinkAccumulator::ApproxDistinct => return None,
        }
    };
}

macro_rules! run_composed_owned_sink {
    ($runner:ident, $rows:expr, $chain:expr, $demand:expr, $sink:expr) => {
        match $sink.builtin_sink_spec()?.accumulator {
            BuiltinSinkAccumulator::Count => {
                cmp::$runner::<cmp::CountSink, _>($rows, $chain, $demand)
            }
            BuiltinSinkAccumulator::Numeric => match numeric_reducer($sink)? {
                BuiltinNumericReducer::Sum => {
                    cmp::$runner::<cmp::SumSink, _>($rows, $chain, $demand)
                }
                BuiltinNumericReducer::Min => {
                    cmp::$runner::<cmp::MinSink, _>($rows, $chain, $demand)
                }
                BuiltinNumericReducer::Max => {
                    cmp::$runner::<cmp::MaxSink, _>($rows, $chain, $demand)
                }
                BuiltinNumericReducer::Avg => {
                    cmp::$runner::<cmp::AvgSink, _>($rows, $chain, $demand)
                }
            },
            BuiltinSinkAccumulator::SelectOne(BuiltinSelectionPosition::First) => {
                cmp::$runner::<cmp::FirstSink, _>($rows, $chain, $demand)
            }
            BuiltinSinkAccumulator::SelectOne(BuiltinSelectionPosition::Last) => {
                cmp::$runner::<cmp::LastSink, _>($rows, $chain, $demand)
            }
            BuiltinSinkAccumulator::ApproxDistinct => return None,
        }
    };
}

pub(super) fn run(
    sink: &Sink,
    rows: &[Val],
    chain: &dyn cmp::Stage,
    demand: PullDemand,
) -> Option<Val> {
    let out = match sink {
        Sink::Collect => cmp::run_pipeline_with_demand::<cmp::CollectSink>(rows, chain, demand),
        Sink::Reducer(_) | Sink::Terminal(_) => {
            run_composed_sink!(run_pipeline_with_demand, rows, chain, demand, sink)
        }
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
        Sink::Reducer(_) | Sink::Terminal(_) => run_composed_owned_sink!(
            run_pipeline_owned_iter_with_demand,
            rows,
            chain,
            demand,
            sink
        ),
        Sink::ApproxCountDistinct => return None,
    };

    Some(out)
}

fn numeric_reducer(sink: &Sink) -> Option<BuiltinNumericReducer> {
    sink.reducer_spec()?.method()?.spec().numeric_reducer
}
