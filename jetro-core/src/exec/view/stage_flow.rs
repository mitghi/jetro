//! Per-element control-flow enum for the view-pipeline stage loop.
//! Mirrors `StageFlow` from the `Val` pipeline path but parameterised
//! over the `ValueView` type to stay in the borrowed domain.

use std::collections::{HashSet, VecDeque};

use crate::{data::value::Val, data::view::ValueView, exec::pipeline};

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
    /// Owned buffered values, used by bounded-state emitting stages.
    Values(Vec<Val>),
    /// Owned sliding values, used by bounded-state window stages.
    Deque(VecDeque<Val>),
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

    pub(super) fn values(&mut self) -> &mut Vec<Val> {
        if !matches!(self, Self::Values(_)) {
            *self = Self::Values(Vec::new());
        }
        match self {
            Self::Values(value) => value,
            _ => unreachable!("values state was initialized"),
        }
    }

    pub(super) fn deque(&mut self) -> &mut VecDeque<Val> {
        if !matches!(self, Self::Deque(_)) {
            *self = Self::Deque(VecDeque::new());
        }
        match self {
            Self::Deque(value) => value,
            _ => unreachable!("deque state was initialized"),
        }
    }

    pub(super) fn next_index(&mut self) -> usize {
        let value = self.counter();
        let out = *value;
        *value = value.saturating_add(1);
        out
    }
}

/// Applies a single view-domain stage to `item`, returning the appropriate
/// `ViewStageFlow`. Returns `None` when the stage requires materialisation
/// or is handled by the recursive view frontier (`FlatMap` expansion).
pub(super) fn apply_stage<'a, V, P, K>(
    item: V,
    stage: pipeline::ViewStageCapability,
    op_idx: usize,
    op_state: &mut [ViewStageState],
    stage_kernels: &[pipeline::BodyKernel],
    vm: &mut crate::vm::VM,
    mut eval_predicate: P,
    mut eval_structural_key: K,
) -> Option<ViewStageFlow<V>>
where
    V: ValueView<'a> + 'a,
    P: FnMut(&V, &pipeline::BodyKernel, &mut crate::vm::VM) -> Option<bool>,
    K: FnMut(&V, Option<&pipeline::BodyKernel>, &mut crate::vm::VM) -> Option<ViewKey>,
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
            if eval_predicate(&item, kernel, vm)? {
                Some(ViewStageFlow::Keep(item))
            } else {
                Some(ViewStageFlow::Drop)
            }
        }
        pipeline::ViewStageCapability::Compact => {
            debug_assert_eq!(stage.input_mode(), pipeline::ViewInputMode::ReadsView);
            debug_assert_eq!(
                stage.output_mode(),
                pipeline::ViewOutputMode::PreservesInputView
            );
            if matches!(item.scalar(), crate::util::JsonView::Null) {
                Some(ViewStageFlow::Drop)
            } else {
                Some(ViewStageFlow::Keep(item))
            }
        }
        pipeline::ViewStageCapability::RemoveValue(ref target) => {
            debug_assert_eq!(stage.input_mode(), pipeline::ViewInputMode::ReadsView);
            debug_assert_eq!(
                stage.output_mode(),
                pipeline::ViewOutputMode::PreservesInputView
            );
            if super::view_matches_value(&item, &target) {
                Some(ViewStageFlow::Drop)
            } else {
                Some(ViewStageFlow::Keep(item))
            }
        }
        pipeline::ViewStageCapability::TakeWhile { kernel } => {
            debug_assert_eq!(stage.input_mode(), pipeline::ViewInputMode::ReadsView);
            debug_assert_eq!(
                stage.output_mode(),
                pipeline::ViewOutputMode::PreservesInputView
            );
            let kernel = stage_kernels.get(kernel)?;
            if eval_predicate(&item, kernel, vm)? {
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
            if eval_predicate(&item, kernel, vm)? {
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
            let key = eval_structural_key(&item, kernel.and_then(|idx| stage_kernels.get(idx)), vm)?;
            if op_state.get_mut(op_idx)?.keys().insert(key) {
                Some(ViewStageFlow::Keep(item))
            } else {
                Some(ViewStageFlow::Drop)
            }
        }
        // Builtin projections and map can emit either a borrowed subview or an
        // owned value, so the frontier handles them before row-local dispatch.
        pipeline::ViewStageCapability::BuiltinProjection { .. } => None,
        pipeline::ViewStageCapability::Map { .. } => None,
        pipeline::ViewStageCapability::KeyedReduce { .. } => None,
        // FlatMap expands one input into many borrowed child views and is
        // handled by `drive_view_item` before row-local stage flow dispatch.
        pipeline::ViewStageCapability::FlatMap { .. } => None,
        pipeline::ViewStageCapability::Flatten { .. } => None,
        pipeline::ViewStageCapability::Explode { .. } => None,
        pipeline::ViewStageCapability::Enumerate => None,
        pipeline::ViewStageCapability::Pairwise => None,
        pipeline::ViewStageCapability::Chunk { .. } => None,
        pipeline::ViewStageCapability::Window { .. } => None,
        pipeline::ViewStageCapability::StringExpand { .. } => None,
    }
}
