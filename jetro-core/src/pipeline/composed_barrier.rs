//! Barrier-stage execution for the composed path.
//! Handles stages that must see all input before producing output (sort,
//! group_by, unique_by) by collecting into `BarrierOutput` then continuing
//! the chain over the resulting rows.

use crate::composed as cmp;
use crate::value::Val;

use super::composed_stage::key_from_kernel;
use super::{BodyKernel, Sink, Stage, StageStrategy};

/// Result of a barrier stage: either a new row list for downstream stages, or a finished value.
pub(super) enum BarrierOutput {
    /// The barrier produced a transformed row set that should continue through the pipeline.
    Rows(Vec<Val>),
    /// The barrier fully consumed its input and produced the final result (e.g. `group_by`).
    Done(Val),
}

/// Executes a barrier stage over the pre-collected `buf`, returning the transformed output.
///
/// Returns `None` when `stage` is not a recognised barrier or its kernel cannot supply a key.
pub(super) fn run(
    stage: &Stage,
    kernel: &BodyKernel,
    strategy: StageStrategy,
    sink: &Sink,
    is_terminal: bool,
    buf: Vec<Val>,
) -> Option<BarrierOutput> {
    let rows = match stage {
        Stage::Reverse(_) => cmp::barrier_reverse(buf),
        Stage::Sort(spec) => {
            let key = match &spec.key {
                None => cmp::KeySource::None,
                Some(_) => key_from_kernel(kernel)?,
            };
            let mut out = match (strategy, spec.descending) {
                (StageStrategy::SortTopK(k), false) | (StageStrategy::SortBottomK(k), true) => {
                    cmp::barrier_top_k(buf, &key, k)
                }
                (StageStrategy::SortTopK(k), true) | (StageStrategy::SortBottomK(k), false) => {
                    cmp::barrier_bottom_k(buf, &key, k)
                }
                (_, false) | (_, true) => cmp::barrier_sort(buf, &key),
            };
            if spec.descending {
                out.reverse();
            }
            out
        }
        Stage::UniqueBy(None) => cmp::barrier_unique_by(buf, &cmp::KeySource::None),
        Stage::UniqueBy(Some(_)) => {
            let key = key_from_kernel(kernel)?;
            cmp::barrier_unique_by(buf, &key)
        }
        Stage::GroupBy(_) => {
            if !matches!(sink, Sink::Collect) || !is_terminal {
                return None;
            }
            let key = key_from_kernel(kernel)?;
            return Some(BarrierOutput::Done(cmp::barrier_group_by(buf, &key)));
        }
        _ => return None,
    };

    Some(BarrierOutput::Rows(rows))
}
