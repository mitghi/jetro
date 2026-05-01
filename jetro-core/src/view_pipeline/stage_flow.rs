use crate::{pipeline, value_view::ValueView};

pub(super) enum ViewStageFlow<V> {
    Keep(V),
    Drop,
    Stop,
}

pub(super) enum ViewFrontierFlow<V> {
    Keep(Vec<V>),
    Stop(Vec<V>),
}

pub(super) fn apply_stage<'a, V>(
    item: V,
    stage: pipeline::ViewStageCapability,
    op_idx: usize,
    op_state: &mut [usize],
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
            if op_state[op_idx] < n {
                op_state[op_idx] += 1;
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
            if op_state[op_idx] >= n {
                Some(ViewStageFlow::Stop)
            } else {
                op_state[op_idx] += 1;
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
        pipeline::ViewStageCapability::Map { kernel } => {
            debug_assert_eq!(stage.input_mode(), pipeline::ViewInputMode::ReadsView);
            debug_assert_eq!(
                stage.output_mode(),
                pipeline::ViewOutputMode::BorrowedSubview
            );
            let kernel = stage_kernels.get(kernel)?;
            Some(ViewStageFlow::Keep(super::eval_map_kernel(&item, kernel)?))
        }
        pipeline::ViewStageCapability::FlatMap { .. } => None,
    }
}

pub(super) fn apply_frontier<'a, V>(
    frontier: Vec<V>,
    stage: pipeline::ViewStageCapability,
    op_idx: usize,
    op_state: &mut [usize],
    stage_kernels: &[pipeline::BodyKernel],
) -> Option<ViewFrontierFlow<V>>
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
        pipeline::ViewStageCapability::FlatMap { kernel } => {
            debug_assert_eq!(stage.input_mode(), pipeline::ViewInputMode::ReadsView);
            debug_assert_eq!(
                stage.output_mode(),
                pipeline::ViewOutputMode::BorrowedSubviews
            );
            let kernel = stage_kernels.get(kernel)?;
            let mut out = Vec::new();
            for item in frontier {
                let iter = super::eval_flat_map_kernel(&item, kernel)?;
                out.extend(iter);
            }
            Some(ViewFrontierFlow::Keep(out))
        }
        _ => {
            let mut out = Vec::with_capacity(frontier.len());
            for item in frontier {
                match apply_stage(item, stage, op_idx, op_state, stage_kernels)? {
                    ViewStageFlow::Keep(next) => out.push(next),
                    ViewStageFlow::Drop => {}
                    ViewStageFlow::Stop => return Some(ViewFrontierFlow::Stop(out)),
                }
            }
            Some(ViewFrontierFlow::Keep(out))
        }
    }
}
