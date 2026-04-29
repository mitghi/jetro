use crate::composed as cmp;
use crate::value::Val;

use super::composed_stage::key_from_kernel;
use super::{BodyKernel, Sink, Stage, StageStrategy};

pub(super) enum BarrierOutput {
    Rows(Vec<Val>),
    Done(Val),
}

pub(super) fn run(
    stage: &Stage,
    kernel: &BodyKernel,
    strategy: StageStrategy,
    sink: &Sink,
    is_terminal: bool,
    buf: Vec<Val>,
) -> Option<BarrierOutput> {
    let rows = match stage {
        Stage::Reverse => cmp::barrier_reverse(buf),
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
