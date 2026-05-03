//! Bridges `Sink` IR variants to the generic `composed::Sink` trait via a
//! macro-generated dispatch table. `run_composed_sink!` selects the concrete
//! sink type at lower time so the outer loop is monomorphised per sink kind.

use crate::builtins::{BuiltinNumericReducer, BuiltinSelectionPosition, BuiltinSinkAccumulator};
use crate::chain_ir::PullDemand;
use crate::composed as cmp;
use crate::value::Val;

use super::Sink;

/// Dispatches a borrowed-slice pipeline run to the concrete typed sink for `$sink`.
///
/// Expands to a `cmp::$runner` call monomorphised over the matching `composed::Sink` impl.
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

/// Dispatches an owned-iterator pipeline run to the concrete typed sink for `$sink`.
///
/// Like `run_composed_sink!` but accepts any `IntoIterator<Item = Val>` as the row source.
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

/// Runs `chain` over the borrowed `rows` slice, collecting into the sink determined by `sink`.
///
/// Returns `None` for `ApproxCountDistinct`, which is not supported by the composed path.
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

/// Runs `chain` over an owned iterator `rows`, collecting into the sink determined by `sink`.
///
/// Used after barrier stages that produce a new `Vec<Val>` rather than a borrowed slice.
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

/// Extracts the `BuiltinNumericReducer` variant from a reducer sink for dispatch.
fn numeric_reducer(sink: &Sink) -> Option<BuiltinNumericReducer> {
    sink.reducer_spec()?.method()?.spec().numeric_reducer
}
