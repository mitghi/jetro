//! Per-element control-flow enum for the view-pipeline stage loop.
//! Mirrors `StageFlow` from the `Val` pipeline path but parameterised
//! over the `ValueView` type to stay in the borrowed domain.

use std::collections::HashSet;

use crate::{exec::pipeline, data::view::ValueView};

use super::key::ViewKey;

/// Per-element control flow for the view-domain stage loop, parameterised by
/// the concrete `ValueView` type `V` to avoid materialisation.
pub(super) enum ViewStageFlow<V> {
    /// The item passed the stage; carry it forward to the next stage with the
    /// (possibly transformed) view.
    Keep(V),
    /// The item was rejected by the stage (e.g. filter predicate was false);
    /// skip to the next source row.
    Drop,
    /// A limit condition was reached; stop iterating entirely.
    Stop,
}

/// Mutable per-stage state carried across successive rows for stateful view
/// stages like `Take`, `Skip`, `DropWhile`, and `Distinct`.
#[derive(Default)]
pub(super) enum ViewStageState {
    /// Initial state before any row is processed.
    #[default]
    Empty,
    /// A monotonically advancing counter, used by `Skip` and `Take`.
    Counter(usize),
    /// A boolean latch, used by `DropWhile` to track when the prefix ends.
    Flag(bool),
    /// A set of seen keys, used by `Distinct` to filter duplicate rows.
    Keys(HashSet<ViewKey>),
}

impl ViewStageState {
    // Returns a mutable reference to the inner counter, initialising to `0` the first time it is accessed.
    fn counter(&mut self) -> &mut usize {
        if !matches!(self, Self::Counter(_)) {
            *self = Self::Counter(0);
        }
        match self {
            Self::Counter(value) => value,
            _ => unreachable!("counter state was initialized"),
        }
    }

    // Returns a mutable reference to the inner `HashSet<ViewKey>`, initialising to an empty set the first time it is accessed.
    fn keys(&mut self) -> &mut HashSet<ViewKey> {
        if !matches!(self, Self::Keys(_)) {
            *self = Self::Keys(HashSet::new());
        }
        match self {
            Self::Keys(value) => value,
            _ => unreachable!("key state was initialized"),
        }
    }

    // Returns a mutable reference to the inner boolean flag, initialising to `false` the first time it is accessed.
    fn flag(&mut self) -> &mut bool {
        if !matches!(self, Self::Flag(_)) {
            *self = Self::Flag(false);
        }
        match self {
            Self::Flag(value) => value,
            _ => unreachable!("flag state was initialized"),
        }
    }
}

/// Applies a single view-domain stage to `item`, returning the appropriate
/// `ViewStageFlow`. Returns `None` when the stage requires materialisation
/// (e.g. `KeyedReduce` or `FlatMap`) and cannot be handled here.
pub(super) fn apply_stage<'a, V>(
    item: V,
    stage: pipeline::ViewStageCapability,
    op_idx: usize,
    op_state: &mut [ViewStageState],
    stage_kernels: &[pipeline::BodyKernel],
) -> Option<ViewStageFlow<V>>
where
    V: ValueView<'a>,
{
    if !matches!(
        stage.materialization(),
        pipeline::ViewMaterialization::Never
    ) {
        return None;
    }

    match stage {
        pipeline::ViewStageCapability::Skip(n) => {
            debug_assert_eq!(stage.input_mode(), pipeline::ViewInputMode::SkipsViewRead);
            debug_assert_eq!(
                stage.output_mode(),
                pipeline::ViewOutputMode::PreservesInputView
            );
            let seen = op_state.get_mut(op_idx)?.counter();
            if *seen < n {
                *seen += 1;
                Some(ViewStageFlow::Drop)
            } else {
                Some(ViewStageFlow::Keep(item))
            }
        }
        pipeline::ViewStageCapability::Take(n) => {
            debug_assert_eq!(stage.input_mode(), pipeline::ViewInputMode::SkipsViewRead);
            debug_assert_eq!(
                stage.output_mode(),
                pipeline::ViewOutputMode::PreservesInputView
            );
            let seen = op_state.get_mut(op_idx)?.counter();
            if *seen >= n {
                Some(ViewStageFlow::Stop)
            } else {
                *seen += 1;
                Some(ViewStageFlow::Keep(item))
            }
        }
        pipeline::ViewStageCapability::Filter { kernel } => {
            debug_assert_eq!(stage.input_mode(), pipeline::ViewInputMode::ReadsView);
            debug_assert_eq!(
                stage.output_mode(),
                pipeline::ViewOutputMode::PreservesInputView
            );
            let kernel = stage_kernels.get(kernel)?;
            if super::eval_filter_kernel(&item, kernel)? {
                Some(ViewStageFlow::Keep(item))
            } else {
                Some(ViewStageFlow::Drop)
            }
        }
        pipeline::ViewStageCapability::TakeWhile { kernel } => {
            debug_assert_eq!(stage.input_mode(), pipeline::ViewInputMode::ReadsView);
            debug_assert_eq!(
                stage.output_mode(),
                pipeline::ViewOutputMode::PreservesInputView
            );
            let kernel = stage_kernels.get(kernel)?;
            if super::eval_filter_kernel(&item, kernel)? {
                Some(ViewStageFlow::Keep(item))
            } else {
                Some(ViewStageFlow::Stop)
            }
        }
        pipeline::ViewStageCapability::DropWhile { kernel } => {
            debug_assert_eq!(stage.input_mode(), pipeline::ViewInputMode::ReadsView);
            debug_assert_eq!(
                stage.output_mode(),
                pipeline::ViewOutputMode::PreservesInputView
            );
            let done = op_state.get_mut(op_idx)?.flag();
            if *done {
                return Some(ViewStageFlow::Keep(item));
            }
            let kernel = stage_kernels.get(kernel)?;
            if super::eval_filter_kernel(&item, kernel)? {
                Some(ViewStageFlow::Drop)
            } else {
                *done = true;
                Some(ViewStageFlow::Keep(item))
            }
        }
        pipeline::ViewStageCapability::Distinct { kernel } => {
            debug_assert_eq!(stage.input_mode(), pipeline::ViewInputMode::ReadsView);
            debug_assert_eq!(
                stage.output_mode(),
                pipeline::ViewOutputMode::PreservesInputView
            );
            let key = match kernel {
                Some(kernel) => super::eval_view_key(&item, Some(stage_kernels.get(kernel)?))?,
                None => super::eval_view_key(&item, None)?,
            };
            if op_state.get_mut(op_idx)?.keys().insert(key) {
                Some(ViewStageFlow::Keep(item))
            } else {
                Some(ViewStageFlow::Drop)
            }
        }
        pipeline::ViewStageCapability::Map { kernel } => {
            debug_assert_eq!(stage.input_mode(), pipeline::ViewInputMode::ReadsView);
            debug_assert_eq!(
                stage.output_mode(),
                pipeline::ViewOutputMode::BorrowedSubview
            );
            let kernel = stage_kernels.get(kernel)?;
            Some(ViewStageFlow::Keep(super::eval_map_kernel(&item, kernel)?))
        }
        pipeline::ViewStageCapability::KeyedReduce { .. } => None,
        pipeline::ViewStageCapability::FlatMap { .. } => None,
    }
}
