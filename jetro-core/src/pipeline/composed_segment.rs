//! Builds a composed `Stage` chain from a contiguous segment of the pipeline's
//! `stages` list. Used by `composed_exec` and `composed_barrier` to construct
//! the pre-barrier and post-barrier stage chains independently.

use std::ops::Range;

use crate::composed as cmp;
use crate::value::Val;

use super::composed_stage::ComposedStageBuilder;
use super::{BodyKernel, Stage};

/// Builds a composed stage chain for the `stages[range]` slice using `kernels` for specialisation.
///
/// Returns `None` if any stage in the range cannot be lowered to a composed equivalent.
pub(super) fn build_chain(
    stages: &[Stage],
    kernels: &[BodyKernel],
    range: Range<usize>,
    builder: &ComposedStageBuilder<'_>,
) -> Option<Box<dyn cmp::Stage>> {
    let mut chain: Box<dyn cmp::Stage> = Box::new(cmp::Identity);
    for idx in range {
        let stage = &stages[idx];
        let kernel = kernels.get(idx).unwrap_or(&BodyKernel::Generic);
        let next = builder.build(stage, kernel)?;
        chain = Box::new(cmp::Composed { a: chain, b: next });
    }
    Some(chain)
}

/// Runs `chain` over `rows` with a `CollectSink` and unwraps the resulting `Val::Arr`.
///
/// Returns `None` if the pipeline result is not an array (should not happen in normal use).
pub(super) fn collect(rows: &[Val], chain: &dyn cmp::Stage) -> Option<Vec<Val>> {
    match cmp::run_pipeline::<cmp::CollectSink>(rows, chain) {
        Val::Arr(items) => Some(items.as_ref().clone()),
        _ => None,
    }
}
