//! View-based pipeline execution over borrowed document representations.
//!
//! Runs pipeline plans against a `ValueView` implementation rather than a
//! materialised `Val` tree. Stages that can stay in the borrowed domain do so;
//! only the final collect step, or a stage that requires a `Val` (e.g. a
//! method call), calls `materialize()`. Used by `physical_eval` when the
//! planner selects the `View` backend preference.

use std::cmp::Ordering;
use std::collections::VecDeque;
use std::sync::Arc;

use crate::data::context::{Env, EvalError};
use crate::data::value::Val;
use crate::data::view::{scalar_view_to_owned_val, view_matches_value, ValueView};
use crate::exec::pipeline;
use crate::plan::demand::PullDemand;
use crate::util::JsonView;
use crate::vm::VM;

mod key;
mod reducer_stage;
mod stage_flow;

use key::ViewKey;
use stage_flow::{ViewStageFlow, ViewStageState};

#[derive(Clone)]
enum FrontierRow<V> {
    Borrowed(V),
    Owned(Val),
}

pub(crate) trait FrontierBaseView<'a>: ValueView<'a> + 'a {}

impl<'a> FrontierBaseView<'a> for crate::data::view::ValView<'a> {}
impl<'a> FrontierBaseView<'a> for crate::data::view::TapeView<'a> {}
impl<'a> FrontierBaseView<'a> for crate::data::view::TapeScratchView<'a> {}

impl<'a, V> ValueView<'a> for FrontierRow<V>
where
    V: FrontierBaseView<'a>,
{
    fn scalar(&self) -> JsonView<'_> {
        match self {
            Self::Borrowed(view) => view.scalar(),
            Self::Owned(value) => JsonView::from_val(value),
        }
    }

    fn field(&self, key: &str) -> Self {
        match self {
            Self::Borrowed(view) => Self::Borrowed(view.field(key)),
            Self::Owned(value) => Self::Owned(value.get_field(key)),
        }
    }

    fn has_key(&self, key: &str) -> Option<bool> {
        match self {
            Self::Borrowed(view) => view.has_key(key),
            Self::Owned(Val::Obj(map)) => Some(map.contains_key(key)),
            Self::Owned(Val::ObjSmall(pairs)) => Some(pairs.iter().any(|(k, _)| k.as_ref() == key)),
            Self::Owned(_) => None,
        }
    }

    fn object_keys(&self) -> Option<Val> {
        match self {
            Self::Borrowed(view) => view.object_keys(),
            Self::Owned(value) => crate::data::view::ValView::new(value).object_keys(),
        }
    }

    fn object_values(&self) -> Option<Val> {
        match self {
            Self::Borrowed(view) => view.object_values(),
            Self::Owned(value) => crate::data::view::ValView::new(value).object_values(),
        }
    }

    fn object_entries(&self) -> Option<Val> {
        match self {
            Self::Borrowed(view) => view.object_entries(),
            Self::Owned(value) => crate::data::view::ValView::new(value).object_entries(),
        }
    }

    fn object_pairs(&self) -> Option<Val> {
        match self {
            Self::Borrowed(view) => view.object_pairs(),
            Self::Owned(value) => crate::data::view::ValView::new(value).object_pairs(),
        }
    }

    fn pick_keys(&self, keys: &[Arc<str>]) -> Option<Val> {
        match self {
            Self::Borrowed(view) => view.pick_keys(keys),
            Self::Owned(value) => crate::data::view::ValView::new(value).pick_keys(keys),
        }
    }

    fn omit_keys(&self, keys: &[Arc<str>]) -> Option<Val> {
        match self {
            Self::Borrowed(view) => view.omit_keys(keys),
            Self::Owned(value) => crate::data::view::ValView::new(value).omit_keys(keys),
        }
    }

    fn index(&self, idx: i64) -> Self {
        match self {
            Self::Borrowed(view) => Self::Borrowed(view.index(idx)),
            Self::Owned(value) => Self::Owned(value.get_index(idx)),
        }
    }

    fn array_iter(&self) -> Option<Box<dyn Iterator<Item = Self> + 'a>> {
        match self {
            Self::Borrowed(view) => Some(Box::new(view.array_iter()?.map(Self::Borrowed))),
            Self::Owned(value) => {
                let items = value.as_vals()?.into_owned();
                Some(Box::new(items.into_iter().map(Self::Owned)))
            }
        }
    }

    fn array_iter_rev(&self) -> Option<Box<dyn Iterator<Item = Self> + 'a>> {
        match self {
            Self::Borrowed(view) => Some(Box::new(view.array_iter_rev()?.map(Self::Borrowed))),
            Self::Owned(value) => {
                let mut items = value.as_vals()?.into_owned();
                items.reverse();
                Some(Box::new(items.into_iter().map(Self::Owned)))
            }
        }
    }

    fn object_iter(&self) -> Option<Box<dyn Iterator<Item = (Arc<str>, Self)> + 'a>> {
        match self {
            Self::Borrowed(view) => Some(Box::new(
                view.object_iter()?
                    .map(|(key, value)| (key, Self::Borrowed(value))),
            )),
            Self::Owned(Val::Obj(map)) => {
                let entries = map
                    .iter()
                    .map(|(key, value)| (Arc::clone(key), value.clone()))
                    .collect::<Vec<_>>();
                Some(Box::new(
                    entries
                        .into_iter()
                        .map(|(key, value)| (key, Self::Owned(value))),
                ))
            }
            Self::Owned(Val::ObjSmall(pairs)) => {
                let entries = pairs
                    .iter()
                    .map(|(key, value)| (Arc::clone(key), value.clone()))
                    .collect::<Vec<_>>();
                Some(Box::new(
                    entries
                        .into_iter()
                        .map(|(key, value)| (key, Self::Owned(value))),
                ))
            }
            Self::Owned(_) => None,
        }
    }

    fn materialize(&self) -> Val {
        match self {
            Self::Borrowed(view) => view.materialize(),
            Self::Owned(value) => value.clone(),
        }
    }
}

/// Navigates a field-key sequence on `cur`, calling `ValueView::field` for each
/// key and returning the deepest resolved view. If a step returns a null-like
/// view, traversal continues with that null view.
pub(crate) fn walk_fields<'a, V>(mut cur: V, keys: &[Arc<str>]) -> V
where
    V: ValueView<'a> + 'a,
{
    for key in keys {
        cur = cur.field(key.as_ref());
    }
    cur
}

/// Top-level view-pipeline runner using caller-owned VM state for fallback suffixes
/// and VM-backed sink targets.
pub(crate) fn run_with_env_and_vm<'a, V>(
    source: V,
    body: &pipeline::PipelineBody,
    cache: Option<&dyn pipeline::PipelineData>,
    base_env: &Env,
    vm: &mut VM,
) -> Option<Result<Val, EvalError>>
where
    V: FrontierBaseView<'a>,
{
    if let Some(result) = run_leading_reverse_view(source.clone(), body, Some(base_env), vm) {
        return Some(result);
    }
    if let Some(result) = run_terminal_collect(source.clone(), body, vm) {
        return Some(result);
    }
    if let Some(result) = run_terminal_select_projection(source.clone(), body, vm) {
        return Some(result);
    }
    if let Some(result) = run_arg_extreme_view(source.clone(), body, vm) {
        return Some(result);
    }
    if let Some(result) = run_full_with_env(source.clone(), body, Some(base_env), vm) {
        return Some(result);
    }
    if let Some(result) = run_reverse_prefix_then_view_suffix(source.clone(), body, base_env, vm) {
        return Some(result);
    }
    if let Some(result) =
        run_sorted_dedup_prefix_then_view_suffix(source.clone(), body, base_env, vm)
    {
        return Some(result);
    }
    if let Some(result) = run_reducing_stage_prefix_then_materialized_suffix(
        source.clone(),
        body,
        cache,
        base_env,
        vm,
    ) {
        return Some(result);
    }
    if let Some(result) =
        run_sort_prefix_then_materialized_suffix(source.clone(), body, cache, base_env, vm)
    {
        return Some(result);
    }
    run_prefix_then_materialized_suffix(source, body, cache, base_env, vm)
}

pub(crate) fn run_receiver_nested_body_with_vm<'a, V>(
    source: V,
    body: &pipeline::PipelineBody,
    vm: &mut VM,
) -> Option<Result<Val, EvalError>>
where
    V: FrontierBaseView<'a>,
{
    let env = Env::new(Val::Null);
    run_with_env_and_vm(source, body, None, &env, vm)
}

fn run_leading_reverse_view<'a, V>(
    source: V,
    body: &pipeline::PipelineBody,
    base_env: Option<&Env>,
    vm: &mut VM,
) -> Option<Result<Val, EvalError>>
where
    V: FrontierBaseView<'a>,
{
    let leading_reverses = body
        .stages
        .iter()
        .take_while(|stage| matches!(stage, pipeline::Stage::Reverse(_)))
        .count();
    if leading_reverses == 0 {
        return None;
    }

    let suffix = view_suffix_capabilities(body, leading_reverses)?;
    let source_demand =
        pipeline::Pipeline::segment_pull_demand(&body.stages[leading_reverses..], &body.sink);
    let source_reversed = leading_reverses % 2 == 1;
    let selective_reversed_suffix_last = suffix.sink.requires_full_reverse_scan_for_selective_last(
        source_demand,
        source_reversed,
        &suffix.stages,
    );
    let drive_demand = if selective_reversed_suffix_last {
        PullDemand::All
    } else {
        source_demand
    };
    let sink_source_reversed = source_reversed && !selective_reversed_suffix_last;
    let sink = view_suffix_sink_for_demand(suffix.sink, source_demand, sink_source_reversed);
    let sink = match resolve_view_sink(sink, base_env, vm) {
        Some(Ok(sink)) => sink,
        Some(Err(err)) => return Some(Err(err)),
        None => return None,
    };

    if let Some(result) = direct_sink_result_from_source_len(
        &source,
        &suffix.stages,
        &body.stage_kernels,
        &sink,
        &body.sink_kernels,
        &body.sink,
    ) {
        return Some(Ok(result));
    }

    let mut sink_acc = pipeline::SinkAccumulator::new(&body.sink);
    let result = if source_reversed {
        if let Some(result) = drive_reversed_direct_position(
            &source,
            &suffix.stages,
            &body.stage_kernels,
            drive_demand,
            vm,
            |item, vm| observe_view_sink(item, &sink, &mut sink_acc, &body.sink_kernels, vm),
        ) {
            result
        } else {
            let items = source.array_iter_rev()?;
            drive_view_iter(
                items,
                &suffix.stages,
                &body.stage_kernels,
                drive_demand,
                vm,
                |item, vm| observe_view_sink(item, &sink, &mut sink_acc, &body.sink_kernels, vm),
            )?
        }
    } else {
        drive_view_frontier(
            source,
            pipeline::SourceCapabilities::VIEW_ARRAY,
            &suffix.stages,
            &body.stage_kernels,
            source_demand,
            vm,
            |item, vm| observe_view_sink(item, &sink, &mut sink_acc, &body.sink_kernels, vm),
        )?
    };
    if let Err(err) = result {
        return Some(Err(err));
    }

    Some(sink_acc.finish_result(source_reversed))
}

struct ReverseBarrierPlan {
    prefix: Vec<pipeline::ViewStageCapability>,
    reverse_stage: usize,
}

fn run_reverse_prefix_then_view_suffix<'a, V>(
    source: V,
    body: &pipeline::PipelineBody,
    base_env: &Env,
    vm: &mut VM,
) -> Option<Result<Val, EvalError>>
where
    V: FrontierBaseView<'a>,
{
    let plan = reverse_barrier_plan(body)?;
    let suffix_start = plan.reverse_stage + 1;
    let suffix = view_suffix_capabilities(body, suffix_start)?;
    let source_demand =
        pipeline::Pipeline::segment_pull_demand(&body.stages[suffix_start..], &body.sink);
    let sink = view_suffix_sink_for_demand(suffix.sink.clone(), source_demand, false);
    let sink = match resolve_view_sink(sink, Some(base_env), vm) {
        Some(Ok(sink)) => sink,
        Some(Err(err)) => return Some(Err(err)),
        None => return None,
    };

    if deterministic_prefix_forces_empty(&plan.prefix, &body.stage_kernels) {
        return run_buffered_rows_view_suffix(
            Vec::<FrontierRow<V>>::new(),
            body,
            &suffix,
            sink,
            source_demand,
            vm,
        );
    }
    if source_demand.is_zero() {
        return run_buffered_rows_view_suffix(
            Vec::<FrontierRow<V>>::new(),
            body,
            &suffix,
            sink,
            source_demand,
            vm,
        );
    }

    let mut rows = ReverseRows::new(source_demand);
    if let Err(err) = drive_view_frontier(
        source,
        pipeline::SourceCapabilities::VIEW_ARRAY,
        &plan.prefix,
        &body.stage_kernels,
        PullDemand::All,
        vm,
        |item, _vm| {
            rows.push(item.clone());
            Some(Ok(ViewRowAction::Emit))
        },
    )? {
        return Some(Err(err));
    }
    run_buffered_rows_view_suffix(rows.into_reversed(), body, &suffix, sink, source_demand, vm)
}

struct ReverseRows<V> {
    rows: Vec<FrontierRow<V>>,
    tail: Option<VecDeque<FrontierRow<V>>>,
    limit: usize,
}

impl<V> ReverseRows<V> {
    fn new(demand: PullDemand) -> Self {
        let limit = match demand {
            PullDemand::FirstInput(n) => Some(n),
            PullDemand::NthInput(index) => Some(index.saturating_add(1)),
            _ => None,
        };
        let Some(limit) = limit else {
            return Self {
                rows: Vec::new(),
                tail: None,
                limit: 0,
            };
        };
        Self {
            rows: Vec::new(),
            tail: Some(VecDeque::with_capacity(limit)),
            limit,
        }
    }

    fn push(&mut self, row: FrontierRow<V>) {
        let Some(tail) = self.tail.as_mut() else {
            self.rows.push(row);
            return;
        };
        if self.limit == 0 {
            return;
        }
        if tail.len() == self.limit {
            tail.pop_front();
        }
        tail.push_back(row);
    }

    fn into_reversed(mut self) -> Vec<FrontierRow<V>> {
        if let Some(tail) = self.tail {
            return tail.into_iter().rev().collect();
        }
        self.rows.reverse();
        self.rows
    }
}

fn reverse_barrier_plan(body: &pipeline::PipelineBody) -> Option<ReverseBarrierPlan> {
    let mut prefix = Vec::new();
    for (idx, stage) in body.stages.iter().enumerate() {
        if matches!(stage, pipeline::Stage::Reverse(_)) {
            return Some(ReverseBarrierPlan {
                prefix,
                reverse_stage: idx,
            });
        }
        prefix.push(pipeline::view_never_materializing_stage_capability(
            body, idx,
        )?);
    }
    None
}

fn run_buffered_rows_view_suffix<'a, V>(
    rows: Vec<FrontierRow<V>>,
    body: &pipeline::PipelineBody,
    suffix: &ViewSuffixCapabilities,
    sink: pipeline::ViewSinkCapability,
    source_demand: PullDemand,
    vm: &mut VM,
) -> Option<Result<Val, EvalError>>
where
    V: FrontierBaseView<'a>,
{
    let mut sink_acc = pipeline::SinkAccumulator::new(&body.sink);
    if let Err(err) = drive_frontier_iter(
        rows,
        &suffix.stages,
        &body.stage_kernels,
        source_demand,
        vm,
        |item, vm| observe_view_sink(item, &sink, &mut sink_acc, &body.sink_kernels, vm),
    )? {
        return Some(Err(err));
    }
    Some(sink_acc.finish_result(false))
}

struct SortedDedupBarrierPlan {
    prefix: Vec<pipeline::ViewStageCapability>,
    dedup_stage: usize,
    key_kernel: Option<pipeline::BodyKernel>,
}

fn run_sorted_dedup_prefix_then_view_suffix<'a, V>(
    source: V,
    body: &pipeline::PipelineBody,
    base_env: &Env,
    vm: &mut VM,
) -> Option<Result<Val, EvalError>>
where
    V: FrontierBaseView<'a>,
{
    let plan = sorted_dedup_barrier_plan(body)?;
    let suffix_start = plan.dedup_stage + 1;
    let suffix = view_suffix_capabilities(body, suffix_start)?;
    let source_demand =
        pipeline::Pipeline::segment_pull_demand(&body.stages[suffix_start..], &body.sink);
    let sink = view_suffix_sink_for_demand(suffix.sink.clone(), source_demand, false);
    let sink = match resolve_view_sink(sink, Some(base_env), vm) {
        Some(Ok(sink)) => sink,
        Some(Err(err)) => return Some(Err(err)),
        None => return None,
    };

    if deterministic_prefix_forces_empty(&plan.prefix, &body.stage_kernels) {
        return run_buffered_rows_view_suffix(
            Vec::<FrontierRow<V>>::new(),
            body,
            &suffix,
            sink,
            source_demand,
            vm,
        );
    }

    let mut keyed = Vec::new();
    if let Err(err) = drive_view_frontier(
        source,
        pipeline::SourceCapabilities::VIEW_ARRAY,
        &plan.prefix,
        &body.stage_kernels,
        PullDemand::All,
        vm,
        |item, vm| {
            let key = sorted_dedup_view_key(item, plan.key_kernel.as_ref(), vm)?;
            keyed.push((key, item.clone()));
            Some(Ok(ViewRowAction::Emit))
        },
    )? {
        return Some(Err(err));
    }
    keyed.sort_by(|a, b| pipeline::cmp_val_total(&a.0, &b.0));
    let mut rows = Vec::with_capacity(keyed.len());
    let mut last_key: Option<Val> = None;
    for (key, row) in keyed {
        if last_key
            .as_ref()
            .is_some_and(|last| pipeline::cmp_val_total(last, &key) == Ordering::Equal)
        {
            continue;
        }
        last_key = Some(key);
        rows.push(row);
    }

    run_buffered_rows_view_suffix(rows, body, &suffix, sink, source_demand, vm)
}

fn sorted_dedup_barrier_plan(body: &pipeline::PipelineBody) -> Option<SortedDedupBarrierPlan> {
    let mut prefix = Vec::new();
    for (idx, stage) in body.stages.iter().enumerate() {
        match stage {
            pipeline::Stage::SortedDedup(program) => {
                let key_kernel = match program {
                    Some(_) => {
                        let kernel = body.stage_kernels.get(idx)?;
                        kernel.is_view_native().then(|| kernel.clone())?
                    }
                    None => pipeline::BodyKernel::Current,
                };
                return Some(SortedDedupBarrierPlan {
                    prefix,
                    dedup_stage: idx,
                    key_kernel: Some(key_kernel),
                });
            }
            _ => prefix.push(pipeline::view_never_materializing_stage_capability(
                body, idx,
            )?),
        }
    }
    None
}

fn sorted_dedup_view_key<'a, V>(
    item: &FrontierRow<V>,
    key_kernel: Option<&pipeline::BodyKernel>,
    vm: &mut VM,
) -> Option<Val>
where
    V: FrontierBaseView<'a>,
{
    match key_kernel {
        Some(kernel) => eval_owned_scalar_or_value_kernel_with_vm(item, kernel, vm),
        None => scalar_view_to_owned_val(item.scalar()).or_else(|| Some(item.materialize())),
    }
}

fn drive_reversed_direct_position<'a, V, F>(
    source: &V,
    stages: &[pipeline::ViewStageCapability],
    stage_kernels: &[pipeline::BodyKernel],
    source_demand: PullDemand,
    vm: &mut VM,
    observe: F,
) -> Option<Result<(), EvalError>>
where
    V: FrontierBaseView<'a>,
    F: FnMut(&FrontierRow<V>, &mut VM) -> Option<Result<ViewRowAction, EvalError>>,
{
    if !pipeline::ViewStageCapability::all_preserve_cardinality(stages) {
        return None;
    }
    let JsonView::ArrayLen(len) = source.scalar() else {
        return None;
    };
    let idx = match source_demand {
        PullDemand::FirstInput(1) => pipeline::index_from_end(len, 0)?,
        PullDemand::NthInput(offset) => pipeline::index_from_end(len, offset)?,
        PullDemand::LastInput(1) => {
            if len == 0 {
                return Some(Ok(()));
            }
            0
        }
        _ => return None,
    };
    let items = std::iter::once(source.index(idx as i64));
    drive_view_iter(items, stages, stage_kernels, PullDemand::All, vm, observe)
}

/// Runs the complete pipeline entirely in the view domain when all stages and
/// the sink have a `ViewCapability`. Returns `None` when any stage lacks
/// view support, allowing a less specialised path to take over.
#[cfg(test)]
fn run_full<'a, V>(source: V, body: &pipeline::PipelineBody) -> Option<Result<Val, EvalError>>
where
    V: FrontierBaseView<'a>,
{
    let mut vm = VM::new();
    run_full_with_env(source, body, None, &mut vm)
}

fn run_full_with_env<'a, V>(
    source: V,
    body: &pipeline::PipelineBody,
    base_env: Option<&Env>,
    vm: &mut VM,
) -> Option<Result<Val, EvalError>>
where
    V: FrontierBaseView<'a>,
{
    let capabilities = pipeline::view_capabilities(body)?;
    let mut sink_acc = pipeline::SinkAccumulator::new(&body.sink);
    let source_demand = body.pull_demand();
    let access_stages = pipeline::ViewStageCapability::source_access_stages(
        &capabilities.stages,
        &body.stage_kernels,
    );
    let source_access = pipeline::SourceCapabilities::VIEW_ARRAY
        .choose_view_access(source_demand, access_stages.as_ref());
    let sink = view_suffix_sink_for_demand(
        capabilities.sink,
        source_demand,
        matches!(source_access, pipeline::SourceAccessMode::Reverse { .. }),
    );
    let sink = match resolve_view_sink(sink, base_env, vm) {
        Some(Ok(sink)) => sink,
        Some(Err(err)) => return Some(Err(err)),
        None => return None,
    };
    if let Some(result) = direct_sink_result_from_source_len(
        &source,
        &capabilities.stages,
        &body.stage_kernels,
        &sink,
        &body.sink_kernels,
        &body.sink,
    ) {
        return Some(Ok(result));
    }

    if let Err(err) = drive_view_frontier(
        source,
        pipeline::SourceCapabilities::VIEW_ARRAY,
        &capabilities.stages,
        &body.stage_kernels,
        source_demand,
        vm,
        |item, vm| observe_view_sink(item, &sink, &mut sink_acc, &body.sink_kernels, vm),
    )? {
        return Some(Err(err));
    }

    Some(sink_acc.finish_result(false))
}

fn run_arg_extreme_view<'a, V>(
    source: V,
    body: &pipeline::PipelineBody,
    vm: &mut VM,
) -> Option<Result<Val, EvalError>>
where
    V: FrontierBaseView<'a>,
{
    let capabilities = pipeline::view_capabilities(body)?;
    let (op, key_kernel) = capabilities.sink.arg_extreme_contract()?;
    if deterministic_prefix_forces_empty(&capabilities.stages, &body.stage_kernels) {
        return Some(Ok(Val::Null));
    }

    let mut best_key = None::<Val>;
    let mut best_row = None::<FrontierRow<V>>;

    if let Err(err) = drive_view_frontier(
        source,
        pipeline::SourceCapabilities::VIEW_ARRAY,
        &capabilities.stages,
        &body.stage_kernels,
        body.pull_demand(),
        vm,
        |item, vm| {
            let key = view_arg_extreme_key_with_vm(item, body.sink_kernels.get(key_kernel)?, vm)?;
            let should_take = match best_key.as_ref() {
                None => true,
                Some(existing) => {
                    let ordering = pipeline::cmp_val_total(&key, existing);
                    if op.wants_max() {
                        ordering.is_gt()
                    } else {
                        ordering.is_lt()
                    }
                }
            };
            if should_take {
                best_key = Some(key);
                best_row = Some(item.clone());
            }
            Some(Ok(ViewRowAction::Emit))
        },
    )? {
        return Some(Err(err));
    }

    Some(Ok(best_row
        .map(pipeline::view_kernel_view_to_owned)
        .unwrap_or(Val::Null)))
}

fn direct_sink_result_from_source_len<'a, V>(
    source: &V,
    stages: &[pipeline::ViewStageCapability],
    stage_kernels: &[pipeline::BodyKernel],
    sink: &pipeline::ViewSinkCapability,
    sink_kernels: &[pipeline::BodyKernel],
    body_sink: &pipeline::Sink,
) -> Option<Val>
where
    V: ValueView<'a> + 'a,
{
    let forced_empty = deterministic_prefix_forces_empty(stages, stage_kernels);
    if let Some(count) = if forced_empty {
        Some(0)
    } else if !sink.can_finish_from_known_cardinality(sink_kernels) {
        None
    } else {
        cardinality_after_deterministic_stages(source, stages, stage_kernels)
    } {
        if let Some(result) = sink.result_from_known_cardinality(count, forced_empty, sink_kernels)
        {
            return Some(result);
        }
    }
    forced_empty.then(|| body_sink.empty_stream_result())?
}

fn cardinality_after_deterministic_stages<'a, V>(
    source: &V,
    stages: &[pipeline::ViewStageCapability],
    stage_kernels: &[pipeline::BodyKernel],
) -> Option<usize>
where
    V: ValueView<'a> + 'a,
{
    if !deterministic_cardinality_supported(stages, stage_kernels) {
        return None;
    }
    let JsonView::ArrayLen(count) = source.scalar() else {
        return None;
    };
    pipeline::ViewStageCapability::deterministic_prefix_cardinality_after(
        stages,
        stage_kernels,
        count,
    )
}

fn deterministic_cardinality_supported(
    stages: &[pipeline::ViewStageCapability],
    stage_kernels: &[pipeline::BodyKernel],
) -> bool {
    pipeline::ViewStageCapability::deterministic_prefix_cardinality_after(stages, stage_kernels, 0)
        .is_some()
}

fn deterministic_prefix_forces_empty(
    stages: &[pipeline::ViewStageCapability],
    stage_kernels: &[pipeline::BodyKernel],
) -> bool {
    pipeline::ViewStageCapability::prefix_forces_empty(stages, stage_kernels)
}

fn resolve_view_sink(
    sink: pipeline::ViewSinkCapability,
    base_env: Option<&Env>,
    vm: &mut VM,
) -> Option<Result<pipeline::ViewSinkCapability, EvalError>> {
    if let Some(program) = sink.membership_target_program() {
        let env = base_env?;
        return Some(
            vm.exec_in_env(&program, env)
                .map(|target| sink.with_resolved_membership_target(target)),
        );
    }
    Some(Ok(sink))
}

/// Feeds one view row into the sink accumulator according to `sink`'s capability.
/// Returns `Some(action)` indicating whether to `Emit`, `Skip`, or `Stop`;
/// returns `None` when a kernel lookup fails (signals the view path is unusable).
fn observe_view_sink<'a, V>(
    item: &V,
    sink: &pipeline::ViewSinkCapability,
    sink_acc: &mut pipeline::SinkAccumulator,
    sink_kernels: &[pipeline::BodyKernel],
    vm: &mut VM,
) -> Option<Result<ViewRowAction, EvalError>>
where
    V: ValueView<'a> + 'a,
{
    match sink {
        pipeline::ViewSinkCapability::Collect => {
            debug_assert_eq!(
                sink.materialization(),
                pipeline::ViewMaterialization::SinkOutputRows
            );
            sink_acc.observe_collect(item.materialize());
            Some(Ok(ViewRowAction::Emit))
        }
        pipeline::ViewSinkCapability::Builtin {
            accumulator,
            predicate_kernel,
            project_kernel,
            ..
        } => {
            if !view_sink_predicate_matches(item, *predicate_kernel, sink_kernels, vm)? {
                return Some(Ok(ViewRowAction::Skip));
            }
            let sink_done = sink_acc.observe_builtin_lazy(
                *accumulator,
                || pipeline::view_kernel_view_to_owned(item.clone()),
                || {
                    let kernel = (*project_kernel)?;
                    let kernel = sink_kernels.get(kernel)?;
                    eval_owned_scalar_or_value_kernel_with_vm(item, kernel, vm)
                },
                || Some(eval_view_key_scalar(item)?.object_key().to_string()),
            )?;
            Some(Ok(if sink_done {
                ViewRowAction::Stop
            } else {
                ViewRowAction::Emit
            }))
        }
        pipeline::ViewSinkCapability::Nth { index } => {
            let sink_done = sink_acc
                .observe_nth_lazy(*index, || pipeline::view_kernel_view_to_owned(item.clone()));
            Some(Ok(if sink_done {
                ViewRowAction::Stop
            } else {
                ViewRowAction::Emit
            }))
        }
        pipeline::ViewSinkCapability::Predicate {
            op,
            predicate_kernel,
        } => {
            let kernel = sink_kernels.get(*predicate_kernel)?;
            let matched = eval_filter_kernel_with_vm(item, kernel, vm)?;
            let sink_done = match sink_acc.observe_predicate_lazy(*op, matched, || {
                pipeline::view_kernel_view_to_owned(item.clone())
            }) {
                Ok(done) => done,
                Err(err) => return Some(Err(err)),
            };
            Some(Ok(if sink_done {
                ViewRowAction::Stop
            } else if matched {
                ViewRowAction::Emit
            } else {
                ViewRowAction::Skip
            }))
        }
        pipeline::ViewSinkCapability::Membership { op, target } => {
            let pipeline::ViewMembershipTarget::Literal(target) = target else {
                return None;
            };
            let matched = view_membership_matches(item, target);
            let sink_done = sink_acc.observe_membership_match(*op, matched);
            Some(Ok(if sink_done {
                ViewRowAction::Stop
            } else if matched {
                ViewRowAction::Emit
            } else {
                ViewRowAction::Skip
            }))
        }
        pipeline::ViewSinkCapability::ArgExtreme { op, key_kernel } => {
            let key = view_arg_extreme_key_with_vm(item, sink_kernels.get(*key_kernel)?, vm)?;
            sink_acc.observe_arg_extreme_lazy(op.wants_max(), key, || {
                pipeline::view_kernel_view_to_owned(item.clone())
            });
            Some(Ok(ViewRowAction::Emit))
        }
        pipeline::ViewSinkCapability::SelectMany {
            n,
            from_end,
            source_reversed,
        } => {
            let sink_done =
                sink_acc.observe_select_many_lazy(*n, *from_end, *source_reversed, || {
                    pipeline::view_kernel_view_to_owned(item.clone())
                });
            Some(Ok(if sink_done {
                ViewRowAction::Stop
            } else {
                ViewRowAction::Emit
            }))
        }
    }
}

fn view_membership_matches<'a, V>(item: &V, target: &Val) -> bool
where
    V: ValueView<'a> + 'a,
{
    view_matches_value(item, target)
}

fn view_arg_extreme_key_with_vm<'a, V>(
    item: &V,
    kernel: &pipeline::BodyKernel,
    vm: &mut VM,
) -> Option<Val>
where
    V: ValueView<'a> + 'a,
{
    match pipeline::eval_view_kernel_with_vm(kernel, item, vm)? {
        pipeline::ViewKernelValue::View(view) => Some(pipeline::view_kernel_view_to_owned(view)),
        pipeline::ViewKernelValue::Owned(value) => Some(value),
    }
}

/// Evaluates the sink's optional predicate kernel against `item`. Returns
/// `Some(true)` when there is no predicate, `Some(bool)` for the predicate
/// result, or `None` when the kernel index is out of bounds.
fn view_sink_predicate_matches<'a, V>(
    item: &V,
    predicate_kernel: Option<usize>,
    sink_kernels: &[pipeline::BodyKernel],
    vm: &mut VM,
) -> Option<bool>
where
    V: ValueView<'a> + 'a,
{
    let Some(kernel_idx) = predicate_kernel else {
        return Some(true);
    };
    let kernel = sink_kernels.get(kernel_idx)?;
    eval_filter_kernel_with_vm(item, kernel, vm)
}

/// Runs as many leading stages as possible in the view domain, materialises the
/// resulting boundary rows, then continues execution with the standard pipeline
/// runner on the remaining suffix stages.
fn run_prefix_then_materialized_suffix<'a, V>(
    source: V,
    body: &pipeline::PipelineBody,
    cache: Option<&dyn pipeline::PipelineData>,
    base_env: &Env,
    vm: &mut VM,
) -> Option<Result<Val, EvalError>>
where
    V: FrontierBaseView<'a>,
{
    let prefix = pipeline::view_prefix_capabilities(body)?;
    if prefix.consumed_stages >= body.stages.len()
        && !body.suffix_can_run_with_materialized_source_env(prefix.consumed_stages)
    {
        return None;
    }
    if !body.suffix_can_run_with_materialized_source_env(prefix.consumed_stages) {
        return None;
    }

    if deterministic_prefix_forces_empty(&prefix.stages, &body.stage_kernels) {
        return Some(run_materialized_suffix(
            body,
            prefix.consumed_stages,
            Vec::new(),
            cache,
            base_env,
            vm,
        ));
    }

    let mut boundary_rows = Vec::new();
    let source_demand = body.pull_demand();

    if let Err(err) = drive_view_frontier(
        source,
        pipeline::SourceCapabilities::VIEW_ARRAY,
        &prefix.stages,
        &body.stage_kernels,
        source_demand,
        vm,
        |item, _vm| {
            boundary_rows.push(item.materialize());
            Some(Ok(ViewRowAction::Emit))
        },
    )? {
        return Some(Err(err));
    }

    Some(run_materialized_suffix(
        body,
        prefix.consumed_stages,
        boundary_rows,
        cache,
        base_env,
        vm,
    ))
}

/// Optimised path for pipelines whose suffix is a pure collect sink. Builds a
/// `TerminalCollectPlan` that may fuse trailing projection stages into the
/// collection kernel, avoiding a separate map pass.
fn run_terminal_collect<'a, V>(
    source: V,
    body: &pipeline::PipelineBody,
    vm: &mut VM,
) -> Option<Result<Val, EvalError>>
where
    V: FrontierBaseView<'a>,
{
    let plan = terminal_collect_plan(body)?;
    let mut collector = pipeline::TerminalCollector::new(plan.collect_program.kernel());
    if deterministic_prefix_forces_empty(&plan.prefix, &body.stage_kernels) {
        return Some(Ok(collector.finish()));
    }

    if let Err(err) = drive_view_frontier(
        source,
        pipeline::SourceCapabilities::VIEW_ARRAY,
        &plan.prefix,
        &body.stage_kernels,
        plan.source_demand,
        vm,
        |item, vm| {
            collector.push_view_program_with_vm(item, &plan.collect_program, vm)?;
            Some(Ok(ViewRowAction::Emit))
        },
    )? {
        return Some(Err(err));
    }

    Some(Ok(collector.finish()))
}

/// Optimised path for `map(...).first()` / `map(...).last()` / `map(...).nth(i)`
/// style suffixes where the trailing projection can run only on the selected view row.
fn run_terminal_select_projection<'a, V>(
    source: V,
    body: &pipeline::PipelineBody,
    vm: &mut VM,
) -> Option<Result<Val, EvalError>>
where
    V: FrontierBaseView<'a>,
{
    let (prefix_len, project_kernel) = terminal_projection_run(body, 0)?;
    let position = match &body.sink {
        pipeline::Sink::Nth(_) => TerminalSelectPosition::Nth,
        _ => terminal_select_position(body.sink.select_one_position()?),
    };
    let prefix = terminal_collect_prefix_from(&body.stages[..prefix_len], body, 0)?;
    let source_demand =
        pipeline::Pipeline::segment_pull_demand(&body.stages[..prefix_len], &body.sink);
    if deterministic_prefix_forces_empty(&prefix, &body.stage_kernels) {
        return Some(Ok(Val::Null));
    }
    let mut selected = Val::Null;
    let mut seen = false;
    let mut nth_seen = 0usize;
    let nth_target = match body.sink {
        pipeline::Sink::Nth(_)
            if source_demand.is_nth_input()
                && pipeline::ViewStageCapability::all_preserve_cardinality(&prefix) =>
        {
            Some(0)
        }
        pipeline::Sink::Nth(index) => Some(index),
        _ => None,
    };

    if let Err(err) = drive_view_frontier(
        source,
        pipeline::SourceCapabilities::VIEW_ARRAY,
        &prefix,
        &body.stage_kernels,
        source_demand,
        vm,
        |item, vm| {
            if let Some(target) = nth_target {
                if nth_seen < target {
                    nth_seen += 1;
                    return Some(Ok(ViewRowAction::Emit));
                }
            }
            selected = eval_owned_scalar_or_value_kernel_with_vm(item, &project_kernel, vm)?;
            seen = true;
            Some(Ok(match position {
                TerminalSelectPosition::First | TerminalSelectPosition::Nth => ViewRowAction::Stop,
                TerminalSelectPosition::Last => ViewRowAction::Emit,
            }))
        },
    )? {
        return Some(Err(err));
    }

    Some(Ok(if seen { selected } else { Val::Null }))
}

#[derive(Clone, Copy)]
enum TerminalSelectPosition {
    First,
    Last,
    Nth,
}

fn terminal_select_position(position: pipeline::Position) -> TerminalSelectPosition {
    match position {
        pipeline::Position::First => TerminalSelectPosition::First,
        pipeline::Position::Last => TerminalSelectPosition::Last,
    }
}

/// Action returned by a sink observer after processing one view row.
enum ViewRowAction {
    /// The row did not pass the predicate; do not count it as output.
    Skip,
    /// The row was accepted and counted as an emitted output.
    Emit,
    /// The sink has reached its output limit; stop iterating immediately.
    Stop,
}

/// Control flow returned by the item-level drive helpers.
enum ViewDriveFlow {
    /// Processing of the current item is complete; continue with the next row.
    Continue,
    /// A demand limit was reached; the outer loop should break immediately.
    Stop,
}

/// Iterates over the array rows of `source` and drives each row through
/// `stages`, calling `observe` for rows that reach the end of the stage list.
/// Returns `None` when `source` cannot be iterated as an array.
fn drive_view_frontier<'a, V, F>(
    source: V,
    source_capabilities: pipeline::SourceCapabilities,
    stages: &[pipeline::ViewStageCapability],
    stage_kernels: &[pipeline::BodyKernel],
    source_demand: PullDemand,
    vm: &mut VM,
    observe: F,
) -> Option<Result<(), EvalError>>
where
    V: FrontierBaseView<'a>,
    F: FnMut(&FrontierRow<V>, &mut VM) -> Option<Result<ViewRowAction, EvalError>>,
{
    drive_view_frontier_rows(
        source,
        source_capabilities,
        stages,
        stage_kernels,
        source_demand,
        vm,
        observe,
    )
}

fn drive_view_frontier_rows<'a, V, F>(
    source: V,
    source_capabilities: pipeline::SourceCapabilities,
    stages: &[pipeline::ViewStageCapability],
    stage_kernels: &[pipeline::BodyKernel],
    source_demand: PullDemand,
    vm: &mut VM,
    observe: F,
) -> Option<Result<(), EvalError>>
where
    V: FrontierBaseView<'a>,
    F: FnMut(&FrontierRow<V>, &mut VM) -> Option<Result<ViewRowAction, EvalError>>,
{
    if source_demand.is_zero() {
        return Some(Ok(()));
    }
    let access_stages = pipeline::ViewStageCapability::source_access_stages(stages, stage_kernels);
    let access = source_capabilities.choose_view_access(source_demand, access_stages.as_ref());
    match access {
        pipeline::SourceAccessMode::Reverse { .. } => {
            let items = source.array_iter_rev()?;
            return drive_view_iter(
                items,
                stages,
                stage_kernels,
                access.iterator_demand(source_demand),
                vm,
                observe,
            );
        }
        pipeline::SourceAccessMode::Indexed(_)
        | pipeline::SourceAccessMode::IndexedFromEnd(_)
        | pipeline::SourceAccessMode::IndexedSuffix(_) => {
            let len = match source.scalar() {
                JsonView::ArrayLen(len) => len,
                _ => return None,
            };
            return match access.indexed_access(len)? {
                pipeline::SourceIndexedAccess::Single(idx) => {
                    let items = std::iter::once(source.index(idx as i64));
                    drive_view_iter(items, stages, stage_kernels, PullDemand::All, vm, observe)
                }
                pipeline::SourceIndexedAccess::Range { start, end } => {
                    let indexed_source = source.clone();
                    let items = (start..end).map(move |idx| indexed_source.index(idx as i64));
                    drive_view_iter(items, stages, stage_kernels, PullDemand::All, vm, observe)
                }
                pipeline::SourceIndexedAccess::Empty => Some(Ok(())),
            };
        }
        pipeline::SourceAccessMode::ForwardBounded(inputs) => {
            let items = source.array_iter()?;
            return drive_view_iter(
                items,
                stages,
                stage_kernels,
                PullDemand::FirstInput(inputs),
                vm,
                observe,
            );
        }
        pipeline::SourceAccessMode::Forward | pipeline::SourceAccessMode::MaterializedFallback => {}
    }
    let items = source.array_iter()?;
    let iter_demand = access.iterator_demand(source_demand);
    drive_view_iter(items, stages, stage_kernels, iter_demand, vm, observe)
}

/// Drives an arbitrary `items` iterator through the view-stage frontier, calling
/// `observe` for each row that survives all stages. Respects `source_demand`
/// limits on both inputs consumed and outputs emitted.
fn drive_view_iter<'a, V, I, F>(
    items: I,
    stages: &[pipeline::ViewStageCapability],
    stage_kernels: &[pipeline::BodyKernel],
    source_demand: PullDemand,
    vm: &mut VM,
    observe: F,
) -> Option<Result<(), EvalError>>
where
    V: FrontierBaseView<'a>,
    I: IntoIterator<Item = V>,
    F: FnMut(&FrontierRow<V>, &mut VM) -> Option<Result<ViewRowAction, EvalError>>,
{
    drive_frontier_iter(
        items.into_iter().map(FrontierRow::Borrowed),
        stages,
        stage_kernels,
        source_demand,
        vm,
        observe,
    )
}

fn drive_frontier_iter<'a, V, I, F>(
    items: I,
    stages: &[pipeline::ViewStageCapability],
    stage_kernels: &[pipeline::BodyKernel],
    source_demand: PullDemand,
    vm: &mut VM,
    mut observe: F,
) -> Option<Result<(), EvalError>>
where
    V: FrontierBaseView<'a>,
    I: IntoIterator<Item = FrontierRow<V>>,
    F: FnMut(&FrontierRow<V>, &mut VM) -> Option<Result<ViewRowAction, EvalError>>,
{
    let mut op_state: Vec<ViewStageState> = (0..stages.len())
        .map(|_| ViewStageState::default())
        .collect();
    let mut pulled_inputs = 0usize;
    let mut emitted_outputs = 0usize;

    for row in items {
        if source_demand.input_satisfied_by(pulled_inputs) {
            break;
        }
        pulled_inputs += 1;

        let flow = match drive_view_item(
            row,
            0,
            stages,
            &mut op_state,
            stage_kernels,
            source_demand,
            &mut emitted_outputs,
            vm,
            &mut observe,
        )? {
            Ok(flow) => flow,
            Err(err) => return Some(Err(err)),
        };
        if matches!(flow, ViewDriveFlow::Stop) {
            break;
        }
        if source_demand.output_satisfied_by(emitted_outputs) {
            break;
        }
    }

    Some(Ok(()))
}

fn drive_owned_child<'a, V, F>(
    value: Val,
    stage_idx: usize,
    stages: &[pipeline::ViewStageCapability],
    op_state: &mut [ViewStageState],
    stage_kernels: &[pipeline::BodyKernel],
    source_demand: PullDemand,
    emitted_outputs: &mut usize,
    vm: &mut VM,
    observe: &mut F,
) -> Option<Result<ViewDriveFlow, EvalError>>
where
    V: FrontierBaseView<'a>,
    F: FnMut(&FrontierRow<V>, &mut VM) -> Option<Result<ViewRowAction, EvalError>>,
{
    drive_view_item(
        FrontierRow::Owned(value),
        stage_idx,
        stages,
        op_state,
        stage_kernels,
        source_demand,
        emitted_outputs,
        vm,
        observe,
    )
}

fn drive_owned_children<'a, V, F, I>(
    values: I,
    stage_idx: usize,
    stages: &[pipeline::ViewStageCapability],
    op_state: &mut [ViewStageState],
    stage_kernels: &[pipeline::BodyKernel],
    source_demand: PullDemand,
    emitted_outputs: &mut usize,
    vm: &mut VM,
    observe: &mut F,
) -> Option<Result<ViewDriveFlow, EvalError>>
where
    V: FrontierBaseView<'a>,
    F: FnMut(&FrontierRow<V>, &mut VM) -> Option<Result<ViewRowAction, EvalError>>,
    I: IntoIterator<Item = Val>,
{
    for value in values {
        let flow = match drive_owned_child(
            value,
            stage_idx,
            stages,
            op_state,
            stage_kernels,
            source_demand,
            emitted_outputs,
            vm,
            observe,
        )? {
            Ok(flow) => flow,
            Err(err) => return Some(Err(err)),
        };
        if matches!(flow, ViewDriveFlow::Stop) {
            return Some(Ok(ViewDriveFlow::Stop));
        }
    }
    Some(Ok(ViewDriveFlow::Continue))
}

/// Recursively applies one view stage to `item`, then advances to the next stage.
/// When all stages have been applied it calls `observe`. `FlatMap` stages expand
/// into child views, each of which is recursed independently.
fn drive_view_item<'a, V, F>(
    item: FrontierRow<V>,
    stage_idx: usize,
    stages: &[pipeline::ViewStageCapability],
    op_state: &mut [ViewStageState],
    stage_kernels: &[pipeline::BodyKernel],
    source_demand: PullDemand,
    emitted_outputs: &mut usize,
    vm: &mut VM,
    observe: &mut F,
) -> Option<Result<ViewDriveFlow, EvalError>>
where
    V: FrontierBaseView<'a>,
    F: FnMut(&FrontierRow<V>, &mut VM) -> Option<Result<ViewRowAction, EvalError>>,
{
    let Some(stage) = stages.get(stage_idx).cloned() else {
        return Some(match observe(&item, vm)? {
            Ok(ViewRowAction::Skip) => Ok(ViewDriveFlow::Continue),
            Err(err) => Err(err),
            Ok(ViewRowAction::Emit) => {
                *emitted_outputs += 1;
                Ok(if source_demand.output_satisfied_by(*emitted_outputs) {
                    ViewDriveFlow::Stop
                } else {
                    ViewDriveFlow::Continue
                })
            }
            Ok(ViewRowAction::Stop) => Ok(ViewDriveFlow::Stop),
        });
    };

    if let pipeline::ViewStageCapability::BuiltinProjection { id, args } = stage {
        let next = match crate::builtins::registry::apply_view_projection(id, &args, item)? {
            crate::builtins::registry::ViewProjectionResult::View(view) => view,
            crate::builtins::registry::ViewProjectionResult::Owned(value) => {
                FrontierRow::Owned(value)
            }
        };
        return drive_view_item(
            next,
            stage_idx + 1,
            stages,
            op_state,
            stage_kernels,
            source_demand,
            emitted_outputs,
            vm,
            observe,
        );
    }

    if let pipeline::ViewStageCapability::Map { kernel } = stage {
        debug_assert_eq!(stage.input_mode(), pipeline::ViewInputMode::ReadsView);
        let kernel = stage_kernels.get(kernel)?;
        return drive_view_item(
            eval_frontier_map_kernel(&item, kernel, vm)?,
            stage_idx + 1,
            stages,
            op_state,
            stage_kernels,
            source_demand,
            emitted_outputs,
            vm,
            observe,
        );
    }

    if let pipeline::ViewStageCapability::FlatMap { kernel } = stage {
        debug_assert_eq!(stage.input_mode(), pipeline::ViewInputMode::ReadsView);
        debug_assert_eq!(
            stage.output_mode(),
            pipeline::ViewOutputMode::BorrowedSubviews
        );
        let kernel = stage_kernels.get(kernel)?;
        for child in eval_flat_map_kernel(&item, kernel, vm)? {
            let flow = match drive_view_item(
                child,
                stage_idx + 1,
                stages,
                op_state,
                stage_kernels,
                source_demand,
                emitted_outputs,
                vm,
                observe,
            )? {
                Ok(flow) => flow,
                Err(err) => return Some(Err(err)),
            };
            if matches!(flow, ViewDriveFlow::Stop) {
                return Some(Ok(ViewDriveFlow::Stop));
            }
        }
        return Some(Ok(ViewDriveFlow::Continue));
    }

    if let pipeline::ViewStageCapability::StringExpand { op, arg } = stage {
        let JsonView::Str(value) = item.scalar() else {
            return Some(Ok(ViewDriveFlow::Continue));
        };
        match op {
            crate::builtins::BuiltinViewStringExpand::Split => {
                let sep = arg.as_deref().unwrap_or("");
                return drive_owned_children(
                    value.split(sep).map(|part| Val::Str(Arc::from(part))),
                    stage_idx + 1,
                    stages,
                    op_state,
                    stage_kernels,
                    source_demand,
                    emitted_outputs,
                    vm,
                    observe,
                );
            }
            crate::builtins::BuiltinViewStringExpand::Lines => {
                return drive_owned_children(
                    value.lines().map(|part| Val::Str(Arc::from(part))),
                    stage_idx + 1,
                    stages,
                    op_state,
                    stage_kernels,
                    source_demand,
                    emitted_outputs,
                    vm,
                    observe,
                );
            }
            crate::builtins::BuiltinViewStringExpand::Words => {
                return drive_owned_children(
                    value
                        .split_whitespace()
                        .map(|part| Val::Str(Arc::from(part))),
                    stage_idx + 1,
                    stages,
                    op_state,
                    stage_kernels,
                    source_demand,
                    emitted_outputs,
                    vm,
                    observe,
                );
            }
            crate::builtins::BuiltinViewStringExpand::Chars
            | crate::builtins::BuiltinViewStringExpand::CharsOf => {
                return drive_owned_children(
                    value.chars().map(|ch| {
                        let mut buf = [0u8; 4];
                        Val::Str(Arc::from(ch.encode_utf8(&mut buf)))
                    }),
                    stage_idx + 1,
                    stages,
                    op_state,
                    stage_kernels,
                    source_demand,
                    emitted_outputs,
                    vm,
                    observe,
                );
            }
            crate::builtins::BuiltinViewStringExpand::Bytes => {
                return drive_owned_children(
                    value
                        .as_bytes()
                        .iter()
                        .map(|byte| Val::Int(i64::from(*byte))),
                    stage_idx + 1,
                    stages,
                    op_state,
                    stage_kernels,
                    source_demand,
                    emitted_outputs,
                    vm,
                    observe,
                );
            }
        }
    }

    match apply_view_stage(item, stage, stage_idx, op_state, stage_kernels, vm)? {
        ViewStageFlow::Keep(next) => drive_view_item(
            next,
            stage_idx + 1,
            stages,
            op_state,
            stage_kernels,
            source_demand,
            emitted_outputs,
            vm,
            observe,
        ),
        ViewStageFlow::Drop => Some(Ok(ViewDriveFlow::Continue)),
        ViewStageFlow::Stop => Some(Ok(ViewDriveFlow::Stop)),
    }
}

/// Execution plan for the terminal-collect fast path. Contains the view-domain
/// prefix stages and a potentially composed collection kernel.
struct TerminalCollectPlan {
    /// Stages that run entirely in the view domain before collection.
    prefix: Vec<pipeline::ViewStageCapability>,
    /// The row program used to extract the value of each row for the output array.
    collect_program: pipeline::RowProgram,
    /// Demand constraint derived from the pipeline's suffix stages.
    source_demand: PullDemand,
}

/// Constructs a `TerminalCollectPlan` for the entire pipeline starting from
/// stage 0, or returns `None` when the plan cannot be formed.
fn terminal_collect_plan(body: &pipeline::PipelineBody) -> Option<TerminalCollectPlan> {
    terminal_collect_plan_from(body, 0)
}

/// Constructs a `TerminalCollectPlan` for the pipeline suffix starting at
/// `start`, detecting and fusing trailing projection stages into the collect kernel.
fn terminal_collect_plan_from(
    body: &pipeline::PipelineBody,
    start: usize,
) -> Option<TerminalCollectPlan> {
    if !matches!(
        body.sink.view_capability(&body.sink_kernels)?,
        pipeline::ViewSinkCapability::Collect
    ) {
        return None;
    }

    let suffix_stages = body.stages.get(start..)?;
    let source_demand = pipeline::Pipeline::segment_pull_demand(suffix_stages, &body.sink);
    if let Some((prefix_len, collect_kernel)) = terminal_projection_run(body, start) {
        return Some(TerminalCollectPlan {
            prefix: terminal_collect_prefix_from(&suffix_stages[..prefix_len], body, start)?,
            collect_program: pipeline::RowProgram::from_kernel(collect_kernel)?,
            source_demand,
        });
    }

    Some(TerminalCollectPlan {
        prefix: terminal_collect_prefix_from(suffix_stages, body, start)?,
        collect_program: pipeline::RowProgram::from_kernel(pipeline::BodyKernel::Current)?,
        source_demand,
    })
}

/// Scans trailing stages from the end, collecting view-native projection kernels
/// that can be fused into the terminal collect. Returns the stage index where
/// the projection run ends and the composed kernel, or `None` when no such
/// run exists.
fn terminal_projection_run(
    body: &pipeline::PipelineBody,
    start: usize,
) -> Option<(usize, pipeline::BodyKernel)> {
    let stages = body.stages.get(start..)?;
    let kernels = body.stage_kernels.get(start..)?;
    let projection = pipeline::Pipeline::late_projection_for(stages, kernels)?;
    Some((projection.prefix_len, projection.kernel))
}

/// Converts the given `stages` slice into a vec of `ViewStageCapability` for use
/// as the prefix in a `TerminalCollectPlan`. Returns `None` if any stage has
/// a `ViewMaterialization` other than `Never`.
fn terminal_collect_prefix_from(
    stages: &[pipeline::Stage],
    body: &pipeline::PipelineBody,
    start: usize,
) -> Option<Vec<pipeline::ViewStageCapability>> {
    pipeline::view_never_materializing_stage_range(body, start, start + stages.len())
}

/// Handles pipelines that begin with a keyed-reduce barrier stage
/// (`group_by`, `count_by`, `index_by`). Collects rows in a `ViewStageReducer`
/// while remaining in the view domain, then passes the reduced `Val` to the
/// materialised suffix runner.
fn run_reducing_stage_prefix_then_materialized_suffix<'a, V>(
    source: V,
    body: &pipeline::PipelineBody,
    cache: Option<&dyn pipeline::PipelineData>,
    base_env: &Env,
    vm: &mut VM,
) -> Option<Result<Val, EvalError>>
where
    V: FrontierBaseView<'a>,
{
    let mut plan: reducer_stage::ReducingStagePlan<FrontierRow<V>> = reducer_stage::plan(body)?;
    if !body.suffix_starts_with_direct_view_projection(plan.consumed_stages)
        && !body.suffix_can_run_with_materialized_source_env(plan.consumed_stages)
    {
        return None;
    }
    let source_demand = body.pull_demand();
    if deterministic_prefix_forces_empty(&plan.prefix, &body.stage_kernels) {
        return Some(run_materialized_value_suffix(
            body,
            plan.consumed_stages,
            plan.reducer.finish(),
            cache,
            base_env,
            vm,
        ));
    }

    if let Err(err) = drive_view_frontier(
        source,
        pipeline::SourceCapabilities::VIEW_ARRAY,
        &plan.prefix,
        &body.stage_kernels,
        source_demand,
        vm,
        |item, vm| {
            plan.reducer.observe(item, &body.stage_kernels, vm)?;
            Some(Ok(ViewRowAction::Emit))
        },
    )? {
        return Some(Err(err));
    }

    Some(run_materialized_value_suffix(
        body,
        plan.consumed_stages,
        plan.reducer.finish(),
        cache,
        base_env,
        vm,
    ))
}

/// Handles pipelines with a `Sort` barrier. Runs any preceding view-native
/// stages, accumulates rows into a `BoundedKeySorter` without materialisation,
/// then continues with the sorted rows through either the view or materialised
/// suffix runner.
fn run_sort_prefix_then_materialized_suffix<'a, V>(
    source: V,
    body: &pipeline::PipelineBody,
    cache: Option<&dyn pipeline::PipelineData>,
    base_env: &Env,
    vm: &mut VM,
) -> Option<Result<Val, EvalError>>
where
    V: FrontierBaseView<'a>,
{
    let plan = sort_barrier_plan(body)?;
    let strategies =
        pipeline::compute_strategies_with_kernels(&body.stages, &body.stage_kernels, &body.sink);
    let strategy = strategies
        .get(plan.sort_stage)
        .copied()
        .unwrap_or(pipeline::StageStrategy::Default);
    if matches!(strategy, pipeline::StageStrategy::SortUntilOutput(_)) {
        return run_sort_prefix_then_view_suffix(source, body, &plan, base_env, vm);
    }
    let suffix_start = plan.sort_stage + 1;
    let collect_suffix = terminal_collect_plan_from(body, suffix_start);
    let select_projection_suffix = terminal_projection_run(body, suffix_start).is_some();
    if collect_suffix.is_none()
        && !select_projection_suffix
        && !body.suffix_can_run_with_materialized_source_env(suffix_start)
    {
        return None;
    }
    if deterministic_prefix_forces_empty(&plan.prefix, &body.stage_kernels) {
        return run_sorted_winners_suffix(
            Vec::<FrontierRow<V>>::new(),
            body,
            suffix_start,
            collect_suffix,
            cache,
            base_env,
            vm,
        );
    }

    let mut sorter =
        pipeline::BoundedKeySorter::new(plan.descending, strategy, pipeline::cmp_val_total);
    if let Err(err) = drive_view_frontier(
        source,
        pipeline::SourceCapabilities::VIEW_ARRAY,
        &plan.prefix,
        &body.stage_kernels,
        PullDemand::All,
        vm,
        |item, vm| {
            let key = view_sort_key(item, plan.key_program.as_ref(), vm)?;
            sorter.push_keyed(key, item.clone());
            Some(Ok(ViewRowAction::Emit))
        },
    )? {
        return Some(Err(err));
    }

    run_sorted_winners_suffix(
        sorter.finish(),
        body,
        suffix_start,
        collect_suffix,
        cache,
        base_env,
        vm,
    )
}

fn run_sorted_winners_suffix<'a, V>(
    winners: Vec<FrontierRow<V>>,
    body: &pipeline::PipelineBody,
    suffix_start: usize,
    collect_suffix: Option<TerminalCollectPlan>,
    cache: Option<&dyn pipeline::PipelineData>,
    base_env: &Env,
    vm: &mut VM,
) -> Option<Result<Val, EvalError>>
where
    V: FrontierBaseView<'a>,
{
    if let Some(collect_plan) = collect_suffix {
        return run_sorted_rows_terminal_collect_suffix(
            winners,
            &collect_plan,
            &body.stage_kernels,
            vm,
        );
    }
    if let Some(out) = run_sorted_rows_terminal_select_projection_suffix(
        winners.as_slice(),
        body,
        suffix_start,
        false,
        vm,
    ) {
        return Some(out);
    }
    if let Some(out) =
        run_sorted_rows_view_suffix(winners.as_slice(), body, suffix_start, base_env, vm)
    {
        return Some(out);
    }
    let boundary_rows: Vec<Val> = winners.into_iter().map(|row| row.materialize()).collect();

    Some(run_materialized_suffix(
        body,
        suffix_start,
        boundary_rows,
        cache,
        base_env,
        vm,
    ))
}

/// Feeds a pre-sorted vec of view rows through the terminal-collect plan,
/// applying any remaining prefix stages and the fused projection kernel without
/// a separate materialisation step.
fn run_sorted_rows_terminal_collect_suffix<'a, V>(
    rows: Vec<FrontierRow<V>>,
    plan: &TerminalCollectPlan,
    stage_kernels: &[pipeline::BodyKernel],
    vm: &mut VM,
) -> Option<Result<Val, EvalError>>
where
    V: FrontierBaseView<'a>,
{
    let mut collector = pipeline::TerminalCollector::new(plan.collect_program.kernel());
    if let Err(err) = drive_frontier_iter(
        rows,
        &plan.prefix,
        stage_kernels,
        plan.source_demand,
        vm,
        |item, vm| {
            collector.push_view_program_with_vm(item, &plan.collect_program, vm)?;
            Some(Ok(ViewRowAction::Emit))
        },
    )? {
        return Some(Err(err));
    }

    Some(Ok(collector.finish()))
}

/// Applies a trailing view-native projection only to the selected sorted row
/// for bounded-sort suffixes such as `.sort_by(k).map(f).last()`.
fn run_sorted_rows_terminal_select_projection_suffix<'a, V>(
    rows: &[FrontierRow<V>],
    body: &pipeline::PipelineBody,
    suffix_start: usize,
    source_reversed: bool,
    vm: &mut VM,
) -> Option<Result<Val, EvalError>>
where
    V: FrontierBaseView<'a>,
{
    let (relative_prefix_len, project_kernel) = terminal_projection_run(body, suffix_start)?;
    let prefix_end = suffix_start + relative_prefix_len;
    let prefix =
        terminal_collect_prefix_from(&body.stages[suffix_start..prefix_end], body, suffix_start)?;
    let source_demand =
        pipeline::Pipeline::segment_pull_demand(&body.stages[suffix_start..prefix_end], &body.sink);
    if let pipeline::Sink::SelectMany { from_end, .. } = body.sink {
        let mut selected = Vec::new();
        if let Err(err) = drive_frontier_iter(
            rows.iter().cloned(),
            &prefix,
            &body.stage_kernels,
            source_demand,
            vm,
            |item, vm| {
                selected.push(eval_owned_scalar_or_value_kernel_with_vm(
                    item,
                    &project_kernel,
                    vm,
                )?);
                Some(Ok(ViewRowAction::Emit))
            },
        )? {
            return Some(Err(err));
        }
        if from_end && source_reversed {
            selected.reverse();
        }
        return Some(Ok(Val::Arr(Arc::new(selected))));
    }
    let mut nth_target = None;
    let position = match &body.sink {
        pipeline::Sink::Nth(index) => {
            nth_target = Some(*index);
            TerminalSelectPosition::Nth
        }
        _ => terminal_select_position(body.sink.select_one_position()?),
    };
    let mut selected = Val::Null;
    let mut seen = false;
    let mut selected_index = 0usize;

    if let Err(err) = drive_frontier_iter(
        rows.iter().cloned(),
        &prefix,
        &body.stage_kernels,
        source_demand,
        vm,
        |item, vm| {
            if let Some(target) = nth_target {
                if selected_index < target {
                    selected_index += 1;
                    return Some(Ok(ViewRowAction::Skip));
                }
            }
            selected = eval_owned_scalar_or_value_kernel_with_vm(item, &project_kernel, vm)?;
            seen = true;
            Some(Ok(match position {
                TerminalSelectPosition::First | TerminalSelectPosition::Nth => ViewRowAction::Stop,
                TerminalSelectPosition::Last => ViewRowAction::Emit,
            }))
        },
    )? {
        return Some(Err(err));
    }

    Some(Ok(if seen { selected } else { Val::Null }))
}

/// Feeds pre-sorted view rows through a fully view-native suffix and sink,
/// avoiding the materialized boundary for bounded sort winners.
fn run_sorted_rows_view_suffix<'a, V>(
    rows: &[FrontierRow<V>],
    body: &pipeline::PipelineBody,
    suffix_start: usize,
    base_env: &Env,
    vm: &mut VM,
) -> Option<Result<Val, EvalError>>
where
    V: FrontierBaseView<'a>,
{
    let suffix = view_suffix_capabilities(body, suffix_start)?;
    let source_demand =
        pipeline::Pipeline::segment_pull_demand(&body.stages[suffix_start..], &body.sink);
    let sink = view_suffix_sink_for_demand(suffix.sink, source_demand, false);
    let sink = match resolve_view_sink(sink, Some(base_env), vm) {
        Some(Ok(sink)) => sink,
        Some(Err(err)) => return Some(Err(err)),
        None => return None,
    };
    let mut sink_acc = pipeline::SinkAccumulator::new(&body.sink);

    if let Err(err) = drive_frontier_iter(
        rows.iter().cloned(),
        &suffix.stages,
        &body.stage_kernels,
        source_demand,
        vm,
        |item, vm| observe_view_sink(item, &sink, &mut sink_acc, &body.sink_kernels, vm),
    )? {
        return Some(Err(err));
    }

    Some(sink_acc.finish_result(false))
}

/// Handles `SortUntilOutput` strategy: sorts rows with an `OrderedKeySorter`
/// then drives them through the view-domain suffix to enable lazy top-N pulls
/// that stop as soon as the output demand is met.
fn run_sort_prefix_then_view_suffix<'a, V>(
    source: V,
    body: &pipeline::PipelineBody,
    plan: &SortBarrierPlan,
    base_env: &Env,
    vm: &mut VM,
) -> Option<Result<Val, EvalError>>
where
    V: FrontierBaseView<'a>,
{
    let suffix = view_suffix_capabilities(body, plan.sort_stage + 1)?;
    let source_demand =
        pipeline::Pipeline::segment_pull_demand(&body.stages[plan.sort_stage + 1..], &body.sink);
    let ordered_descending = if source_demand.is_suffix() {
        !plan.descending
    } else {
        plan.descending
    };
    let source_reversed = ordered_descending != plan.descending;
    let sink = view_suffix_sink_for_demand(suffix.sink.clone(), source_demand, source_reversed);
    let sink = match resolve_view_sink(sink, Some(base_env), vm) {
        Some(Ok(sink)) => sink,
        Some(Err(err)) => return Some(Err(err)),
        None => return None,
    };
    if deterministic_prefix_forces_empty(&plan.prefix, &body.stage_kernels) {
        return run_ordered_rows_view_suffix(
            Vec::<FrontierRow<V>>::new(),
            body,
            plan.sort_stage + 1,
            source_reversed,
            &suffix,
            sink,
            source_demand,
            vm,
        );
    }
    let mut sorter = pipeline::OrderedKeySorter::new(ordered_descending, pipeline::cmp_val_total);

    if let Err(err) = drive_view_frontier(
        source,
        pipeline::SourceCapabilities::VIEW_ARRAY,
        &plan.prefix,
        &body.stage_kernels,
        PullDemand::All,
        vm,
        |item, vm| {
            let key = view_sort_key(item, plan.key_program.as_ref(), vm)?;
            sorter.push_keyed(key, item.clone());
            Some(Ok(ViewRowAction::Emit))
        },
    )? {
        return Some(Err(err));
    }

    let ordered: Vec<_> = sorter.finish().collect();
    run_ordered_rows_view_suffix(
        ordered,
        body,
        plan.sort_stage + 1,
        source_reversed,
        &suffix,
        sink,
        source_demand,
        vm,
    )
}

fn run_ordered_rows_view_suffix<'a, V>(
    ordered: Vec<FrontierRow<V>>,
    body: &pipeline::PipelineBody,
    suffix_start: usize,
    source_reversed: bool,
    suffix: &ViewSuffixCapabilities,
    sink: pipeline::ViewSinkCapability,
    source_demand: PullDemand,
    vm: &mut VM,
) -> Option<Result<Val, EvalError>>
where
    V: FrontierBaseView<'a>,
{
    if let Some(out) = run_sorted_rows_terminal_select_projection_suffix(
        ordered.as_slice(),
        body,
        suffix_start,
        source_reversed,
        vm,
    ) {
        return Some(out);
    }
    let mut sink_acc = pipeline::SinkAccumulator::new(&body.sink);

    if let Err(err) = drive_frontier_iter(
        ordered,
        &suffix.stages,
        &body.stage_kernels,
        source_demand,
        vm,
        |item, vm| observe_view_sink(item, &sink, &mut sink_acc, &body.sink_kernels, vm),
    )? {
        return Some(Err(err));
    }

    Some(sink_acc.finish_result(false))
}

fn view_suffix_sink_for_demand(
    sink: pipeline::ViewSinkCapability,
    source_demand: PullDemand,
    source_reversed: bool,
) -> pipeline::ViewSinkCapability {
    sink.for_source_demand(source_demand, source_reversed)
}

/// Plan produced when a `Sort` barrier is detected. Records the view-domain
/// prefix, the index of the sort stage, and the key extraction configuration.
struct SortBarrierPlan {
    /// View-domain stages that precede the sort barrier.
    prefix: Vec<pipeline::ViewStageCapability>,
    /// Index of the `Sort` stage within `body.stages`.
    sort_stage: usize,
    /// Row program for the sort key, or `None` for a natural (identity) sort.
    key_program: Option<pipeline::RowProgram>,
    /// Whether the sort order is descending.
    descending: bool,
}

/// The fully resolved view-domain capabilities for the suffix of a pipeline
/// starting after a barrier stage.
struct ViewSuffixCapabilities {
    /// View-domain stage capabilities for each suffix stage.
    stages: Vec<pipeline::ViewStageCapability>,
    /// View-domain sink capability for the pipeline's terminal sink.
    sink: pipeline::ViewSinkCapability,
}

/// Resolves view-domain capabilities for all stages from `start` to the end of
/// the pipeline plus the sink. Returns `None` if any stage or the sink lacks a
/// view capability.
fn view_suffix_capabilities(
    body: &pipeline::PipelineBody,
    start: usize,
) -> Option<ViewSuffixCapabilities> {
    let mut stages = Vec::with_capacity(body.stages.len().saturating_sub(start));
    for (idx, stage) in body.stages.iter().enumerate().skip(start) {
        stages.push(stage.view_capability(idx, body.stage_kernels.get(idx))?);
    }
    Some(ViewSuffixCapabilities {
        stages,
        sink: body.sink.view_capability(&body.sink_kernels)?,
    })
}

/// Scans `body.stages` for the first `Sort` stage preceded only by view-native
/// `Never`-materialisation stages, building a `SortBarrierPlan`. Returns `None`
/// when no qualifying `Sort` barrier is found.
fn sort_barrier_plan(body: &pipeline::PipelineBody) -> Option<SortBarrierPlan> {
    let mut prefix = Vec::new();
    for (idx, stage) in body.stages.iter().enumerate() {
        match stage {
            pipeline::Stage::Sort(spec) => {
                let key_program = if spec.key.is_some() {
                    let kernel = body.stage_kernels.get(idx)?;
                    kernel
                        .is_view_native()
                        .then(|| pipeline::RowProgram::from_kernel(kernel.clone()))?
                } else {
                    None
                };
                return Some(SortBarrierPlan {
                    prefix,
                    sort_stage: idx,
                    key_program,
                    descending: spec.descending,
                });
            }
            _ => {
                prefix.push(pipeline::view_never_materializing_stage_capability(
                    body, idx,
                )?);
            }
        }
    }
    None
}

/// Runs the suffix of `body` (from `consumed_stages` onward) against a
/// materialised `boundary_rows` array using the standard `Val`-based pipeline runner.
fn run_materialized_suffix(
    body: &pipeline::PipelineBody,
    consumed_stages: usize,
    boundary_rows: Vec<Val>,
    cache: Option<&dyn pipeline::PipelineData>,
    base_env: &Env,
    vm: &mut VM,
) -> Result<Val, EvalError> {
    if consumed_stages >= body.stages.len() && matches!(body.sink, pipeline::Sink::Collect) {
        return Ok(Val::arr(boundary_rows));
    }
    let suffix = suffix_body(body, consumed_stages)
        .with_source(pipeline::Source::Receiver(Val::arr(boundary_rows)));
    let root = Val::Null;
    suffix.run_with_env_and_vm(&root, base_env, cache, vm)
}

/// Runs the suffix of `body` against a single `boundary_value` (e.g. the
/// output of a keyed-reduce barrier). Short-circuits to return the value
/// directly when no suffix stages remain and the sink is `Collect`.
fn run_materialized_value_suffix(
    body: &pipeline::PipelineBody,
    consumed_stages: usize,
    mut boundary_value: Val,
    cache: Option<&dyn pipeline::PipelineData>,
    base_env: &Env,
    vm: &mut VM,
) -> Result<Val, EvalError> {
    let mut consumed_stages = consumed_stages;
    while let Some(pipeline::Stage::Builtin(call)) = body.stages.get(consumed_stages) {
        if !call.is_view_projection() {
            break;
        }
        boundary_value = call
            .try_apply(&boundary_value)?
            .or_else(|| call.apply(&boundary_value))
            .ok_or_else(|| EvalError(format!("{:?}: unsupported projection", call.method)))?;
        consumed_stages += 1;
    }
    if consumed_stages >= body.stages.len() && matches!(body.sink, pipeline::Sink::Collect) {
        return Ok(boundary_value);
    }
    let suffix =
        suffix_body(body, consumed_stages).with_source(pipeline::Source::Receiver(boundary_value));
    let root = Val::Null;
    suffix.run_with_env_and_vm(&root, base_env, cache, vm)
}

/// Applies a single view stage to `item`, returning the control flow decision
/// (`Keep`, `Drop`, or `Stop`). Delegates to `stage_flow::apply_stage`.
fn apply_view_stage<'a, V>(
    item: V,
    stage: pipeline::ViewStageCapability,
    op_idx: usize,
    op_state: &mut [ViewStageState],
    stage_kernels: &[pipeline::BodyKernel],
    vm: &mut VM,
) -> Option<ViewStageFlow<V>>
where
    V: ValueView<'a> + 'a,
{
    stage_flow::apply_stage(item, stage, op_idx, op_state, stage_kernels, vm)
}

/// Slices `body` to produce a new `PipelineBody` starting at `consumed_stages`,
/// preserving the sink and adjusting the stage/kernel slices accordingly.
fn suffix_body(body: &pipeline::PipelineBody, consumed_stages: usize) -> pipeline::PipelineBody {
    let stage_exprs = if body.stage_exprs.len() == body.stages.len() {
        body.stage_exprs[consumed_stages..].to_vec()
    } else {
        Vec::new()
    };
    pipeline::PipelineBody {
        stages: body.stages[consumed_stages..].to_vec(),
        stage_exprs,
        sink: body.sink.clone(),
        stage_kernels: body.stage_kernels[consumed_stages..].to_vec(),
        sink_kernels: body.sink_kernels.clone(),
    }
}

fn eval_filter_kernel_with_vm<'a, V>(
    item: &V,
    kernel: &pipeline::BodyKernel,
    vm: &mut VM,
) -> Option<bool>
where
    V: ValueView<'a> + 'a,
{
    match pipeline::eval_view_kernel_with_vm(kernel, item, vm)? {
        pipeline::ViewKernelValue::View(view) => Some(view.scalar().truthy()),
        pipeline::ViewKernelValue::Owned(value) => Some(crate::util::is_truthy(&value)),
    }
}

fn eval_frontier_map_kernel<'a, V>(
    item: &FrontierRow<V>,
    kernel: &pipeline::BodyKernel,
    vm: &mut VM,
) -> Option<FrontierRow<V>>
where
    V: FrontierBaseView<'a>,
{
    if let pipeline::BodyKernel::NestedPlan(plan) = kernel {
        if let Some(body) = plan.receiver_view_body() {
            let value = match item {
                FrontierRow::Borrowed(view) => {
                    run_receiver_nested_body_with_vm(view.clone(), &body, vm)?.ok()?
                }
                FrontierRow::Owned(value) => plan.run(value.clone()).ok()?,
            };
            return Some(FrontierRow::Owned(value));
        }
    }
    match pipeline::eval_view_kernel_with_vm(kernel, item, vm)? {
        pipeline::ViewKernelValue::View(view) => Some(view),
        pipeline::ViewKernelValue::Owned(value) => Some(FrontierRow::Owned(value)),
    }
}

/// Evaluates `kernel` against `item` expecting an array result, returning an
/// iterator of child views. Returns `None` when the kernel produces an owned
/// `Val` (not array-iterable in the view domain).
fn eval_flat_map_kernel<'a, V>(
    item: &FrontierRow<V>,
    kernel: &pipeline::BodyKernel,
    vm: &mut VM,
) -> Option<Box<dyn Iterator<Item = FrontierRow<V>> + 'a>>
where
    V: FrontierBaseView<'a>,
{
    match pipeline::eval_view_kernel_with_vm(kernel, item, vm)? {
        pipeline::ViewKernelValue::View(view) => view.array_iter(),
        pipeline::ViewKernelValue::Owned(value) => {
            let items = value.as_vals()?.into_owned();
            Some(Box::new(items.into_iter().map(FrontierRow::Owned)))
        }
    }
}

fn eval_owned_scalar_or_value_kernel_with_vm<'a, V>(
    item: &V,
    kernel: &pipeline::BodyKernel,
    vm: &mut VM,
) -> Option<Val>
where
    V: ValueView<'a> + 'a,
{
    match pipeline::eval_view_kernel_with_vm(kernel, item, vm)? {
        pipeline::ViewKernelValue::View(view) => Some(pipeline::view_kernel_view_to_owned(view)),
        pipeline::ViewKernelValue::Owned(value) => Some(value),
    }
}

fn eval_view_key_with_vm<'a, V>(
    item: &V,
    kernel: Option<&pipeline::BodyKernel>,
    vm: &mut VM,
) -> Option<ViewKey>
where
    V: ValueView<'a> + 'a,
{
    match kernel {
        Some(kernel) => match pipeline::eval_view_kernel_with_vm(kernel, item, vm)? {
            pipeline::ViewKernelValue::View(view) => ViewKey::from_value_view(&view),
            pipeline::ViewKernelValue::Owned(value) => Some(ViewKey::from_owned(value)),
        },
        None => eval_view_key_scalar(item),
    }
}

fn eval_structural_view_key_with_vm<'a, V>(
    item: &V,
    kernel: Option<&pipeline::BodyKernel>,
    vm: &mut VM,
) -> Option<ViewKey>
where
    V: ValueView<'a> + 'a,
{
    match kernel {
        Some(kernel) => match pipeline::eval_view_kernel_with_vm(kernel, item, vm)? {
            pipeline::ViewKernelValue::View(view) => ViewKey::from_structural_value_view(&view),
            pipeline::ViewKernelValue::Owned(value) => Some(ViewKey::from_structural_owned(value)),
        },
        None => ViewKey::from_structural_value_view(item),
    }
}

fn eval_view_key_scalar<'a, V>(item: &V) -> Option<ViewKey>
where
    V: ValueView<'a> + 'a,
{
    ViewKey::from_value_view(item)
}

/// Extracts a sort key `Val` from `item`, optionally applying a row program.
/// Falls back to `item.materialize()` when no key program is specified.
fn view_sort_key<'a, V>(item: &V, key: Option<&pipeline::RowProgram>, vm: &mut VM) -> Option<Val>
where
    V: ValueView<'a> + 'a,
{
    match key {
        Some(program) => match program.eval_view_with_vm(item, vm)? {
            pipeline::ViewKernelValue::View(view) => {
                Some(pipeline::view_kernel_view_to_owned(view))
            }
            pipeline::ViewKernelValue::Owned(value) => Some(value),
        },
        None => scalar_view_to_owned_val(item.scalar()).or_else(|| Some(item.materialize())),
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::rc::Rc;
    use std::sync::Arc;

    use indexmap::IndexMap;

    use crate::builtins::registry::view_scalar_projection;
    use crate::builtins::{
        BuiltinMembershipSink, BuiltinPredicateSink, BuiltinSelectionPosition,
        BuiltinSinkAccumulator,
    };
    use crate::compile::compiler::Compiler;
    use crate::data::context::Env;
    use crate::data::value::Val;
    use crate::data::view::{TapeView, ValView, ValueView};
    use crate::exec::pipeline::{
        eval_view_kernel, ArgExtremeSinkSpec, BodyKernel, MembershipSinkSpec, MembershipSinkTarget,
        NestedPlanKernel, NumOp, PipelineBody, Plan, PredicateSinkSpec, ReducerOp, ReducerSpec,
        Sink, Source, SourceCapabilities, Stage, ViewKernelValue, ViewSinkCapability,
        ViewStageCapability,
    };
    use crate::parse::ast::BinOp;
    use crate::plan::demand::PullDemand;
    use crate::util::JsonView;
    use crate::vm::VM;

    #[derive(Clone)]
    struct CountingView {
        rows: Arc<[i64]>,
        idx: Option<usize>,
        scalar_reads: Rc<Cell<usize>>,
        index_reads: Rc<Cell<usize>>,
        array_iter_reads: Rc<Cell<usize>>,
        array_iter_rev_reads: Rc<Cell<usize>>,
        materialize_reads: Rc<Cell<usize>>,
    }

    impl CountingView {
        fn root(rows: &[i64]) -> Self {
            Self {
                rows: rows.iter().copied().collect::<Vec<_>>().into(),
                idx: None,
                scalar_reads: Rc::new(Cell::new(0)),
                index_reads: Rc::new(Cell::new(0)),
                array_iter_reads: Rc::new(Cell::new(0)),
                array_iter_rev_reads: Rc::new(Cell::new(0)),
                materialize_reads: Rc::new(Cell::new(0)),
            }
        }

        fn scalar_reads(&self) -> usize {
            self.scalar_reads.get()
        }

        fn index_reads(&self) -> usize {
            self.index_reads.get()
        }

        fn materialize_reads(&self) -> usize {
            self.materialize_reads.get()
        }

        fn array_iter_reads(&self) -> usize {
            self.array_iter_reads.get()
        }

        fn array_iter_rev_reads(&self) -> usize {
            self.array_iter_rev_reads.get()
        }
    }

    impl<'a> super::FrontierBaseView<'a> for CountingView {}

    impl<'a> ValueView<'a> for CountingView {
        fn scalar(&self) -> JsonView<'_> {
            self.scalar_reads.set(self.scalar_reads.get() + 1);
            if self.idx.is_none() {
                return JsonView::ArrayLen(self.rows.len());
            }
            self.idx
                .and_then(|idx| self.rows.get(idx).copied())
                .map(JsonView::Int)
                .unwrap_or(JsonView::Null)
        }

        fn field(&self, _key: &str) -> Self {
            Self {
                rows: Arc::clone(&self.rows),
                idx: None,
                scalar_reads: Rc::clone(&self.scalar_reads),
                index_reads: Rc::clone(&self.index_reads),
                array_iter_reads: Rc::clone(&self.array_iter_reads),
                array_iter_rev_reads: Rc::clone(&self.array_iter_rev_reads),
                materialize_reads: Rc::clone(&self.materialize_reads),
            }
        }

        fn has_key(&self, _key: &str) -> Option<bool> {
            None
        }

        fn object_keys(&self) -> Option<Val> {
            None
        }

        fn object_values(&self) -> Option<Val> {
            None
        }

        fn object_entries(&self) -> Option<Val> {
            None
        }

        fn object_pairs(&self) -> Option<Val> {
            None
        }

        fn pick_keys(&self, _keys: &[Arc<str>]) -> Option<Val> {
            None
        }

        fn omit_keys(&self, _keys: &[Arc<str>]) -> Option<Val> {
            None
        }

        fn index(&self, idx: i64) -> Self {
            self.index_reads.set(self.index_reads.get() + 1);
            let idx = if idx >= 0 { Some(idx as usize) } else { None };
            Self {
                rows: Arc::clone(&self.rows),
                idx,
                scalar_reads: Rc::clone(&self.scalar_reads),
                index_reads: Rc::clone(&self.index_reads),
                array_iter_reads: Rc::clone(&self.array_iter_reads),
                array_iter_rev_reads: Rc::clone(&self.array_iter_rev_reads),
                materialize_reads: Rc::clone(&self.materialize_reads),
            }
        }

        fn array_iter(&self) -> Option<Box<dyn Iterator<Item = Self> + 'a>> {
            self.array_iter_reads.set(self.array_iter_reads.get() + 1);
            if self.idx.is_some() {
                return None;
            }
            let rows = Arc::clone(&self.rows);
            let scalar_reads = Rc::clone(&self.scalar_reads);
            let index_reads = Rc::clone(&self.index_reads);
            let array_iter_reads = Rc::clone(&self.array_iter_reads);
            let array_iter_rev_reads = Rc::clone(&self.array_iter_rev_reads);
            let materialize_reads = Rc::clone(&self.materialize_reads);
            Some(Box::new((0..rows.len()).map(move |idx| Self {
                rows: Arc::clone(&rows),
                idx: Some(idx),
                scalar_reads: Rc::clone(&scalar_reads),
                index_reads: Rc::clone(&index_reads),
                array_iter_reads: Rc::clone(&array_iter_reads),
                array_iter_rev_reads: Rc::clone(&array_iter_rev_reads),
                materialize_reads: Rc::clone(&materialize_reads),
            })))
        }

        fn array_iter_rev(&self) -> Option<Box<dyn Iterator<Item = Self> + 'a>> {
            self.array_iter_reads.set(self.array_iter_reads.get() + 1);
            self.array_iter_rev_reads
                .set(self.array_iter_rev_reads.get() + 1);
            if self.idx.is_some() {
                return None;
            }
            let rows = Arc::clone(&self.rows);
            let scalar_reads = Rc::clone(&self.scalar_reads);
            let index_reads = Rc::clone(&self.index_reads);
            let array_iter_reads = Rc::clone(&self.array_iter_reads);
            let array_iter_rev_reads = Rc::clone(&self.array_iter_rev_reads);
            let materialize_reads = Rc::clone(&self.materialize_reads);
            Some(Box::new((0..rows.len()).rev().map(move |idx| Self {
                rows: Arc::clone(&rows),
                idx: Some(idx),
                scalar_reads: Rc::clone(&scalar_reads),
                index_reads: Rc::clone(&index_reads),
                array_iter_reads: Rc::clone(&array_iter_reads),
                array_iter_rev_reads: Rc::clone(&array_iter_rev_reads),
                materialize_reads: Rc::clone(&materialize_reads),
            })))
        }

        fn materialize(&self) -> Val {
            self.materialize_reads.set(self.materialize_reads.get() + 1);
            self.idx
                .and_then(|idx| self.rows.get(idx).copied())
                .map(Val::Int)
                .unwrap_or(Val::Null)
        }
    }

    #[derive(Clone)]
    struct CountingNestedView {
        rows: Arc<[Arc<[i64]>]>,
        row_idx: Option<usize>,
        child_idx: Option<usize>,
        scalar_reads: Rc<Cell<usize>>,
        array_iter_reads: Rc<Cell<usize>>,
        materialize_reads: Rc<Cell<usize>>,
    }

    impl CountingNestedView {
        fn root(rows: &[&[i64]]) -> Self {
            Self {
                rows: rows
                    .iter()
                    .map(|row| row.iter().copied().collect::<Vec<_>>().into())
                    .collect::<Vec<_>>()
                    .into(),
                row_idx: None,
                child_idx: None,
                scalar_reads: Rc::new(Cell::new(0)),
                array_iter_reads: Rc::new(Cell::new(0)),
                materialize_reads: Rc::new(Cell::new(0)),
            }
        }

        fn materialize_reads(&self) -> usize {
            self.materialize_reads.get()
        }
    }

    impl<'a> super::FrontierBaseView<'a> for CountingNestedView {}

    impl<'a> ValueView<'a> for CountingNestedView {
        fn scalar(&self) -> JsonView<'_> {
            self.scalar_reads.set(self.scalar_reads.get() + 1);
            match (self.row_idx, self.child_idx) {
                (Some(row_idx), Some(child_idx)) => self
                    .rows
                    .get(row_idx)
                    .and_then(|row| row.get(child_idx))
                    .copied()
                    .map(JsonView::Int)
                    .unwrap_or(JsonView::Null),
                (Some(row_idx), None) => self
                    .rows
                    .get(row_idx)
                    .map(|row| JsonView::ArrayLen(row.len()))
                    .unwrap_or(JsonView::Null),
                (None, None) => JsonView::ArrayLen(self.rows.len()),
                (None, Some(_)) => JsonView::Null,
            }
        }

        fn field(&self, _key: &str) -> Self {
            self.clone()
        }

        fn has_key(&self, _key: &str) -> Option<bool> {
            None
        }

        fn object_keys(&self) -> Option<Val> {
            None
        }

        fn object_values(&self) -> Option<Val> {
            None
        }

        fn object_entries(&self) -> Option<Val> {
            None
        }

        fn object_pairs(&self) -> Option<Val> {
            None
        }

        fn pick_keys(&self, _keys: &[Arc<str>]) -> Option<Val> {
            None
        }

        fn omit_keys(&self, _keys: &[Arc<str>]) -> Option<Val> {
            None
        }

        fn index(&self, idx: i64) -> Self {
            if idx < 0 {
                return self.clone();
            }
            let idx = idx as usize;
            match self.row_idx {
                Some(row_idx) => Self {
                    child_idx: Some(idx),
                    ..self.with_row(row_idx)
                },
                None => self.with_row(idx),
            }
        }

        fn array_iter(&self) -> Option<Box<dyn Iterator<Item = Self> + 'a>> {
            self.array_iter_reads.set(self.array_iter_reads.get() + 1);
            match (self.row_idx, self.child_idx) {
                (None, None) => {
                    let len = self.rows.len();
                    let this = self.clone();
                    Some(Box::new((0..len).map(move |idx| this.with_row(idx))))
                }
                (Some(row_idx), None) => {
                    let len = self.rows.get(row_idx)?.len();
                    let this = self.with_row(row_idx);
                    Some(Box::new((0..len).map(move |idx| Self {
                        child_idx: Some(idx),
                        ..this.clone()
                    })))
                }
                _ => None,
            }
        }

        fn array_iter_rev(&self) -> Option<Box<dyn Iterator<Item = Self> + 'a>> {
            self.array_iter().map(|iter| {
                let mut items: Vec<Self> = iter.collect();
                items.reverse();
                Box::new(items.into_iter()) as Box<dyn Iterator<Item = Self> + 'a>
            })
        }

        fn materialize(&self) -> Val {
            self.materialize_reads.set(self.materialize_reads.get() + 1);
            match (self.row_idx, self.child_idx) {
                (Some(row_idx), Some(child_idx)) => self
                    .rows
                    .get(row_idx)
                    .and_then(|row| row.get(child_idx))
                    .copied()
                    .map(Val::Int)
                    .unwrap_or(Val::Null),
                (Some(row_idx), None) => self
                    .rows
                    .get(row_idx)
                    .map(|row| Val::arr(row.iter().copied().map(Val::Int).collect()))
                    .unwrap_or(Val::Null),
                (None, None) => Val::arr(
                    self.rows
                        .iter()
                        .map(|row| Val::arr(row.iter().copied().map(Val::Int).collect()))
                        .collect(),
                ),
                (None, Some(_)) => Val::Null,
            }
        }
    }

    impl CountingNestedView {
        fn with_row(&self, row_idx: usize) -> Self {
            Self {
                rows: Arc::clone(&self.rows),
                row_idx: Some(row_idx),
                child_idx: None,
                scalar_reads: Rc::clone(&self.scalar_reads),
                array_iter_reads: Rc::clone(&self.array_iter_reads),
                materialize_reads: Rc::clone(&self.materialize_reads),
            }
        }
    }

    #[derive(Clone)]
    struct CountingStringView {
        rows: Arc<[Arc<str>]>,
        idx: Option<usize>,
        materialize_reads: Rc<Cell<usize>>,
    }

    impl CountingStringView {
        fn root(rows: &[&str]) -> Self {
            Self {
                rows: rows
                    .iter()
                    .map(|row| Arc::from(*row))
                    .collect::<Vec<_>>()
                    .into(),
                idx: None,
                materialize_reads: Rc::new(Cell::new(0)),
            }
        }

        fn materialize_reads(&self) -> usize {
            self.materialize_reads.get()
        }
    }

    impl<'a> super::FrontierBaseView<'a> for CountingStringView {}

    impl<'a> ValueView<'a> for CountingStringView {
        fn scalar(&self) -> JsonView<'_> {
            match self.idx {
                Some(idx) => self
                    .rows
                    .get(idx)
                    .map(|row| JsonView::Str(row.as_ref()))
                    .unwrap_or(JsonView::Null),
                None => JsonView::ArrayLen(self.rows.len()),
            }
        }

        fn field(&self, _key: &str) -> Self {
            self.clone()
        }

        fn has_key(&self, _key: &str) -> Option<bool> {
            None
        }

        fn object_keys(&self) -> Option<Val> {
            None
        }

        fn object_values(&self) -> Option<Val> {
            None
        }

        fn object_entries(&self) -> Option<Val> {
            None
        }

        fn object_pairs(&self) -> Option<Val> {
            None
        }

        fn pick_keys(&self, _keys: &[Arc<str>]) -> Option<Val> {
            None
        }

        fn omit_keys(&self, _keys: &[Arc<str>]) -> Option<Val> {
            None
        }

        fn index(&self, idx: i64) -> Self {
            Self {
                rows: Arc::clone(&self.rows),
                idx: (idx >= 0).then_some(idx as usize),
                materialize_reads: Rc::clone(&self.materialize_reads),
            }
        }

        fn array_iter(&self) -> Option<Box<dyn Iterator<Item = Self> + 'a>> {
            if self.idx.is_some() {
                return None;
            }
            let rows = Arc::clone(&self.rows);
            let materialize_reads = Rc::clone(&self.materialize_reads);
            Some(Box::new((0..rows.len()).map(move |idx| Self {
                rows: Arc::clone(&rows),
                idx: Some(idx),
                materialize_reads: Rc::clone(&materialize_reads),
            })))
        }

        fn array_iter_rev(&self) -> Option<Box<dyn Iterator<Item = Self> + 'a>> {
            if self.idx.is_some() {
                return None;
            }
            let rows = Arc::clone(&self.rows);
            let materialize_reads = Rc::clone(&self.materialize_reads);
            Some(Box::new((0..rows.len()).rev().map(move |idx| Self {
                rows: Arc::clone(&rows),
                idx: Some(idx),
                materialize_reads: Rc::clone(&materialize_reads),
            })))
        }

        fn materialize(&self) -> Val {
            self.materialize_reads.set(self.materialize_reads.get() + 1);
            self.idx
                .and_then(|idx| self.rows.get(idx).cloned())
                .map(Val::Str)
                .unwrap_or(Val::Null)
        }
    }

    #[derive(Clone)]
    struct CountingObjectValuesView {
        rows: Arc<[Arc<[i64]>]>,
        idx: Option<usize>,
        materialize_reads: Rc<Cell<usize>>,
        object_value_reads: Rc<Cell<usize>>,
    }

    impl CountingObjectValuesView {
        fn root(rows: &[&[i64]]) -> Self {
            Self {
                rows: rows
                    .iter()
                    .map(|row| Arc::<[i64]>::from(*row))
                    .collect::<Vec<_>>()
                    .into(),
                idx: None,
                materialize_reads: Rc::new(Cell::new(0)),
                object_value_reads: Rc::new(Cell::new(0)),
            }
        }

        fn materialize_reads(&self) -> usize {
            self.materialize_reads.get()
        }

        fn object_value_reads(&self) -> usize {
            self.object_value_reads.get()
        }
    }

    impl<'a> super::FrontierBaseView<'a> for CountingObjectValuesView {}

    impl<'a> ValueView<'a> for CountingObjectValuesView {
        fn scalar(&self) -> JsonView<'_> {
            match self.idx {
                Some(idx) => self
                    .rows
                    .get(idx)
                    .map(|row| JsonView::ObjectLen(row.len()))
                    .unwrap_or(JsonView::Null),
                None => JsonView::ArrayLen(self.rows.len()),
            }
        }

        fn field(&self, _key: &str) -> Self {
            self.clone()
        }

        fn has_key(&self, _key: &str) -> Option<bool> {
            None
        }

        fn object_keys(&self) -> Option<Val> {
            None
        }

        fn object_values(&self) -> Option<Val> {
            self.object_value_reads
                .set(self.object_value_reads.get() + 1);
            let row = self.rows.get(self.idx?)?;
            Some(Val::arr(row.iter().copied().map(Val::Int).collect()))
        }

        fn object_entries(&self) -> Option<Val> {
            None
        }

        fn object_pairs(&self) -> Option<Val> {
            None
        }

        fn pick_keys(&self, _keys: &[Arc<str>]) -> Option<Val> {
            None
        }

        fn omit_keys(&self, _keys: &[Arc<str>]) -> Option<Val> {
            None
        }

        fn index(&self, idx: i64) -> Self {
            Self {
                rows: Arc::clone(&self.rows),
                idx: (idx >= 0).then_some(idx as usize),
                materialize_reads: Rc::clone(&self.materialize_reads),
                object_value_reads: Rc::clone(&self.object_value_reads),
            }
        }

        fn array_iter(&self) -> Option<Box<dyn Iterator<Item = Self> + 'a>> {
            if self.idx.is_some() {
                return None;
            }
            let rows = Arc::clone(&self.rows);
            let materialize_reads = Rc::clone(&self.materialize_reads);
            let object_value_reads = Rc::clone(&self.object_value_reads);
            Some(Box::new((0..rows.len()).map(move |idx| Self {
                rows: Arc::clone(&rows),
                idx: Some(idx),
                materialize_reads: Rc::clone(&materialize_reads),
                object_value_reads: Rc::clone(&object_value_reads),
            })))
        }

        fn array_iter_rev(&self) -> Option<Box<dyn Iterator<Item = Self> + 'a>> {
            if self.idx.is_some() {
                return None;
            }
            let rows = Arc::clone(&self.rows);
            let materialize_reads = Rc::clone(&self.materialize_reads);
            let object_value_reads = Rc::clone(&self.object_value_reads);
            Some(Box::new((0..rows.len()).rev().map(move |idx| Self {
                rows: Arc::clone(&rows),
                idx: Some(idx),
                materialize_reads: Rc::clone(&materialize_reads),
                object_value_reads: Rc::clone(&object_value_reads),
            })))
        }

        fn materialize(&self) -> Val {
            self.materialize_reads.set(self.materialize_reads.get() + 1);
            self.idx
                .and_then(|idx| self.rows.get(idx))
                .map(|row| Val::arr(row.iter().copied().map(Val::Int).collect()))
                .unwrap_or(Val::Null)
        }
    }

    #[derive(Clone)]
    struct CountingKeyedObjectView {
        rows: Arc<[(i64, i64)]>,
        idx: Option<usize>,
        field: Option<KeyedObjectField>,
        scalar_reads: Rc<Cell<usize>>,
        array_iter_reads: Rc<Cell<usize>>,
        materialize_reads: Rc<Cell<usize>>,
    }

    #[derive(Clone, Copy)]
    enum KeyedObjectField {
        Key,
        Value,
    }

    impl CountingKeyedObjectView {
        fn root(rows: &[(i64, i64)]) -> Self {
            Self {
                rows: rows.iter().copied().collect::<Vec<_>>().into(),
                idx: None,
                field: None,
                scalar_reads: Rc::new(Cell::new(0)),
                array_iter_reads: Rc::new(Cell::new(0)),
                materialize_reads: Rc::new(Cell::new(0)),
            }
        }

        fn array_iter_reads(&self) -> usize {
            self.array_iter_reads.get()
        }

        fn materialize_reads(&self) -> usize {
            self.materialize_reads.get()
        }
    }

    impl<'a> super::FrontierBaseView<'a> for CountingKeyedObjectView {}

    impl<'a> ValueView<'a> for CountingKeyedObjectView {
        fn scalar(&self) -> JsonView<'_> {
            self.scalar_reads.set(self.scalar_reads.get() + 1);
            match (self.idx, self.field) {
                (None, _) => JsonView::ArrayLen(self.rows.len()),
                (Some(_), None) => JsonView::ObjectLen(2),
                (Some(idx), Some(KeyedObjectField::Key)) => self
                    .rows
                    .get(idx)
                    .map(|(key, _)| JsonView::Int(*key))
                    .unwrap_or(JsonView::Null),
                (Some(idx), Some(KeyedObjectField::Value)) => self
                    .rows
                    .get(idx)
                    .map(|(_, value)| JsonView::Int(*value))
                    .unwrap_or(JsonView::Null),
            }
        }

        fn field(&self, key: &str) -> Self {
            let field = match key {
                "k" => Some(KeyedObjectField::Key),
                "v" => Some(KeyedObjectField::Value),
                _ => None,
            };
            Self {
                rows: Arc::clone(&self.rows),
                idx: self.idx,
                field,
                scalar_reads: Rc::clone(&self.scalar_reads),
                array_iter_reads: Rc::clone(&self.array_iter_reads),
                materialize_reads: Rc::clone(&self.materialize_reads),
            }
        }

        fn has_key(&self, key: &str) -> Option<bool> {
            Some(matches!(key, "k" | "v") && self.idx.is_some())
        }

        fn object_keys(&self) -> Option<Val> {
            None
        }

        fn object_values(&self) -> Option<Val> {
            None
        }

        fn object_entries(&self) -> Option<Val> {
            None
        }

        fn object_pairs(&self) -> Option<Val> {
            None
        }

        fn pick_keys(&self, _keys: &[Arc<str>]) -> Option<Val> {
            None
        }

        fn omit_keys(&self, _keys: &[Arc<str>]) -> Option<Val> {
            None
        }

        fn index(&self, idx: i64) -> Self {
            Self {
                rows: Arc::clone(&self.rows),
                idx: (idx >= 0).then_some(idx as usize),
                field: None,
                scalar_reads: Rc::clone(&self.scalar_reads),
                array_iter_reads: Rc::clone(&self.array_iter_reads),
                materialize_reads: Rc::clone(&self.materialize_reads),
            }
        }

        fn array_iter(&self) -> Option<Box<dyn Iterator<Item = Self> + 'a>> {
            self.array_iter_reads.set(self.array_iter_reads.get() + 1);
            if self.idx.is_some() {
                return None;
            }
            let rows = Arc::clone(&self.rows);
            let scalar_reads = Rc::clone(&self.scalar_reads);
            let array_iter_reads = Rc::clone(&self.array_iter_reads);
            let materialize_reads = Rc::clone(&self.materialize_reads);
            Some(Box::new((0..rows.len()).map(move |idx| Self {
                rows: Arc::clone(&rows),
                idx: Some(idx),
                field: None,
                scalar_reads: Rc::clone(&scalar_reads),
                array_iter_reads: Rc::clone(&array_iter_reads),
                materialize_reads: Rc::clone(&materialize_reads),
            })))
        }

        fn array_iter_rev(&self) -> Option<Box<dyn Iterator<Item = Self> + 'a>> {
            self.array_iter_reads.set(self.array_iter_reads.get() + 1);
            if self.idx.is_some() {
                return None;
            }
            let rows = Arc::clone(&self.rows);
            let scalar_reads = Rc::clone(&self.scalar_reads);
            let array_iter_reads = Rc::clone(&self.array_iter_reads);
            let materialize_reads = Rc::clone(&self.materialize_reads);
            Some(Box::new((0..rows.len()).rev().map(move |idx| Self {
                rows: Arc::clone(&rows),
                idx: Some(idx),
                field: None,
                scalar_reads: Rc::clone(&scalar_reads),
                array_iter_reads: Rc::clone(&array_iter_reads),
                materialize_reads: Rc::clone(&materialize_reads),
            })))
        }

        fn materialize(&self) -> Val {
            self.materialize_reads.set(self.materialize_reads.get() + 1);
            match (self.idx, self.field) {
                (Some(idx), Some(KeyedObjectField::Key)) => self
                    .rows
                    .get(idx)
                    .map(|(key, _)| Val::Int(*key))
                    .unwrap_or(Val::Null),
                (Some(idx), Some(KeyedObjectField::Value)) => self
                    .rows
                    .get(idx)
                    .map(|(_, value)| Val::Int(*value))
                    .unwrap_or(Val::Null),
                (Some(idx), None) => self
                    .rows
                    .get(idx)
                    .map(|(key, value)| {
                        Val::ObjSmall(Arc::new([
                            (Arc::from("k"), Val::Int(*key)),
                            (Arc::from("v"), Val::Int(*value)),
                        ]))
                    })
                    .unwrap_or(Val::Null),
                (None, _) => Val::Null,
            }
        }
    }

    #[test]
    fn view_frontier_zero_demand_skips_source_access() {
        let source = CountingView::root(&[1, 2, 3]);
        let observed = Rc::new(Cell::new(0usize));
        let observed_in_closure = Rc::clone(&observed);
        let mut vm = VM::new();

        let result = super::drive_view_frontier(
            source.clone(),
            SourceCapabilities::VIEW_ARRAY,
            &[],
            &[],
            PullDemand::FirstInput(0),
            &mut vm,
            move |_, _| {
                observed_in_closure.set(observed_in_closure.get() + 1);
                Some(Ok(super::ViewRowAction::Emit))
            },
        );

        assert!(result.is_some());
        assert_eq!(observed.get(), 0);
        assert_eq!(source.scalar_reads(), 0);
        assert_eq!(source.array_iter_reads(), 0);
        assert_eq!(source.materialize_reads(), 0);
    }

    #[test]
    fn view_frontier_indexed_suffix_preserves_order() {
        let source = CountingView::root(&[1, 2, 3, 4]);
        let observed = Rc::new(std::cell::RefCell::new(Vec::new()));
        let observed_in_closure = Rc::clone(&observed);
        let mut vm = VM::new();

        let result = super::drive_view_frontier(
            source.clone(),
            SourceCapabilities::VIEW_ARRAY,
            &[],
            &[],
            PullDemand::LastInput(2),
            &mut vm,
            move |item, _| {
                observed_in_closure.borrow_mut().push(item.materialize());
                Some(Ok(super::ViewRowAction::Emit))
            },
        );

        assert!(result.is_some());
        assert_eq!(*observed.borrow(), vec![Val::Int(3), Val::Int(4)]);
        assert_eq!(source.scalar_reads(), 1);
        assert_eq!(source.index_reads(), 2);
        assert_eq!(source.array_iter_reads(), 0);
        assert_eq!(source.array_iter_rev_reads(), 0);
    }

    #[test]
    fn nested_plan_kernel_runs_on_view_without_materializing_row() {
        let expr = crate::parse::parser::parse("items.map(@ * 2).sum()").expect("parse");
        let kernel = BodyKernel::classify_expr(&expr);
        let source = CountingView::root(&[1, 2, 3]);
        let row = source.index(0);

        let out = eval_view_kernel(&kernel, &row).and_then(|value| match value {
            ViewKernelValue::Owned(value) => Some(value),
            ViewKernelValue::View(view) => Some(view.materialize()),
        });

        assert_eq!(out, Some(Val::Int(12)));
        assert_eq!(source.materialize_reads(), 0);
    }

    #[test]
    fn view_full_runner_stops_after_until_output_demand_is_met() {
        let source = CountingView::root(&[1, 2, 3]);
        let body = PipelineBody {
            stages: vec![
                Stage::Filter(
                    Arc::new(crate::vm::Program::new(Vec::new(), "")),
                    crate::builtins::BuiltinViewStage::Filter,
                ),
                Stage::UsizeBuiltin {
                    method: crate::builtins::BuiltinMethod::Take,
                    value: 2,
                },
            ],
            stage_exprs: Vec::new(),
            sink: Sink::Collect,
            stage_kernels: vec![
                BodyKernel::CurrentCmpLit(BinOp::Gt, Val::Int(0)),
                BodyKernel::Generic,
            ],
            sink_kernels: Vec::new(),
        };

        let out = super::run_full(source.clone(), &body).unwrap().unwrap();

        let out_json: serde_json::Value = out.into();
        assert_eq!(out_json, serde_json::json!([1, 2]));
        assert_eq!(source.scalar_reads(), 2);
    }

    #[test]
    fn view_full_runner_stops_when_predicate_sink_result_is_decided() {
        let source = CountingView::root(&[1, 2, 3, 4]);
        let body = PipelineBody {
            stages: Vec::new(),
            stage_exprs: Vec::new(),
            sink: Sink::Predicate(PredicateSinkSpec {
                op: BuiltinPredicateSink::Any,
                predicate: Arc::new(crate::vm::Program::new(Vec::new(), "")),
                predicate_expr: None,
            }),
            stage_kernels: Vec::new(),
            sink_kernels: vec![BodyKernel::CurrentCmpLit(BinOp::Gt, Val::Int(2))],
        };

        let out = super::run_full(source.clone(), &body).unwrap().unwrap();

        assert_eq!(out, Val::Bool(true));
        assert_eq!(source.scalar_reads(), 3);
        assert_eq!(source.materialize_reads(), 0);
    }

    #[test]
    fn view_full_runner_stops_when_all_sink_fails() {
        let source = CountingView::root(&[1, 2, 3, 4]);
        let body = PipelineBody {
            stages: Vec::new(),
            stage_exprs: Vec::new(),
            sink: Sink::Predicate(PredicateSinkSpec {
                op: BuiltinPredicateSink::All,
                predicate: Arc::new(crate::vm::Program::new(Vec::new(), "")),
                predicate_expr: None,
            }),
            stage_kernels: Vec::new(),
            sink_kernels: vec![BodyKernel::CurrentCmpLit(BinOp::Lt, Val::Int(3))],
        };

        let out = super::run_full(source.clone(), &body).unwrap().unwrap();

        assert_eq!(out, Val::Bool(false));
        assert_eq!(source.scalar_reads(), 3);
        assert_eq!(source.materialize_reads(), 0);
    }

    #[test]
    fn view_constant_predicate_sink_uses_source_length_without_iterating_rows() {
        let source = CountingView::root(&[1, 2, 3, 4, 5]);
        let body = PipelineBody {
            stages: vec![
                Stage::UsizeBuiltin {
                    method: crate::builtins::BuiltinMethod::Skip,
                    value: 1,
                },
                Stage::UsizeBuiltin {
                    method: crate::builtins::BuiltinMethod::Take,
                    value: 3,
                },
            ],
            stage_exprs: Vec::new(),
            sink: Sink::Predicate(PredicateSinkSpec {
                op: BuiltinPredicateSink::IndicesWhere,
                predicate: Arc::new(crate::vm::Program::new(Vec::new(), "")),
                predicate_expr: None,
            }),
            stage_kernels: vec![BodyKernel::Generic, BodyKernel::Generic],
            sink_kernels: vec![BodyKernel::ConstBool(true)],
        };

        let out = super::run_full(source.clone(), &body).unwrap().unwrap();

        assert_eq!(out, Val::int_vec(vec![0, 1, 2]));
        assert_eq!(source.scalar_reads(), 1);
        assert_eq!(source.array_iter_reads(), 0);
        assert_eq!(source.materialize_reads(), 0);
    }

    #[test]
    fn view_empty_cardinality_sinks_skip_row_and_predicate_evaluation() {
        let predicate_source = CountingView::root(&[1, 2, 3]);
        let predicate_body = PipelineBody {
            stages: vec![Stage::UsizeBuiltin {
                method: crate::builtins::BuiltinMethod::Take,
                value: 0,
            }],
            stage_exprs: Vec::new(),
            sink: Sink::Predicate(PredicateSinkSpec {
                op: BuiltinPredicateSink::Any,
                predicate: Arc::new(crate::vm::Program::new(Vec::new(), "")),
                predicate_expr: None,
            }),
            stage_kernels: vec![BodyKernel::Generic],
            sink_kernels: vec![BodyKernel::CurrentCmpLit(BinOp::Gt, Val::Int(0))],
        };

        let predicate = super::run_full(predicate_source.clone(), &predicate_body)
            .unwrap()
            .unwrap();

        assert_eq!(predicate, Val::Bool(false));
        assert_eq!(predicate_source.scalar_reads(), 0);
        assert_eq!(predicate_source.array_iter_reads(), 0);
        assert_eq!(predicate_source.materialize_reads(), 0);

        let membership_source = CountingView::root(&[1, 2, 3]);
        let membership_body = PipelineBody {
            stages: vec![Stage::UsizeBuiltin {
                method: crate::builtins::BuiltinMethod::Take,
                value: 0,
            }],
            stage_exprs: Vec::new(),
            sink: Sink::Membership(MembershipSinkSpec {
                op: BuiltinMembershipSink::Includes,
                target: MembershipSinkTarget::Literal(Val::Int(1)),
            }),
            stage_kernels: vec![BodyKernel::Generic],
            sink_kernels: Vec::new(),
        };

        let membership = super::run_full(membership_source.clone(), &membership_body)
            .unwrap()
            .unwrap();

        assert_eq!(membership, Val::Bool(false));
        assert_eq!(membership_source.scalar_reads(), 0);
        assert_eq!(membership_source.array_iter_reads(), 0);
        assert_eq!(membership_source.materialize_reads(), 0);

        let compound_membership_source = CountingView::root(&[1, 2, 3]);
        let compound_membership_body = PipelineBody {
            stages: vec![Stage::UsizeBuiltin {
                method: crate::builtins::BuiltinMethod::Take,
                value: 0,
            }],
            stage_exprs: Vec::new(),
            sink: Sink::Membership(MembershipSinkSpec {
                op: BuiltinMembershipSink::IndicesOf,
                target: MembershipSinkTarget::Literal(Val::arr(vec![Val::Int(1)])),
            }),
            stage_kernels: vec![BodyKernel::Generic],
            sink_kernels: Vec::new(),
        };

        let compound_membership = super::run_full(
            compound_membership_source.clone(),
            &compound_membership_body,
        )
        .unwrap()
        .unwrap();

        assert_eq!(
            serde_json::Value::from(compound_membership),
            serde_json::json!([])
        );
        assert_eq!(compound_membership_source.scalar_reads(), 0);
        assert_eq!(compound_membership_source.array_iter_reads(), 0);
        assert_eq!(compound_membership_source.materialize_reads(), 0);
    }

    #[test]
    fn view_constant_predicate_sinks_use_builtin_result_metadata() {
        fn run(op: BuiltinPredicateSink, predicate: BodyKernel) -> (Val, CountingView) {
            let source = CountingView::root(&[1, 2, 3]);
            let body = PipelineBody {
                stages: Vec::new(),
                stage_exprs: Vec::new(),
                sink: Sink::Predicate(PredicateSinkSpec {
                    op,
                    predicate: Arc::new(crate::vm::Program::new(Vec::new(), "")),
                    predicate_expr: None,
                }),
                stage_kernels: Vec::new(),
                sink_kernels: vec![predicate],
            };

            let out = super::run_full(source.clone(), &body).unwrap().unwrap();
            (out, source)
        }

        let (any_false, any_source) = run(BuiltinPredicateSink::Any, BodyKernel::ConstBool(false));
        assert_eq!(any_false, Val::Bool(false));
        assert_eq!(any_source.array_iter_reads(), 0);

        let (all_false, all_source) = run(BuiltinPredicateSink::All, BodyKernel::ConstBool(false));
        assert_eq!(all_false, Val::Bool(false));
        assert_eq!(all_source.array_iter_reads(), 0);

        let (find_index, find_source) =
            run(BuiltinPredicateSink::FindIndex, BodyKernel::ConstBool(true));
        assert_eq!(find_index, Val::Int(0));
        assert_eq!(find_source.array_iter_reads(), 0);

        let (indices, indices_source) = run(
            BuiltinPredicateSink::IndicesWhere,
            BodyKernel::ConstBool(true),
        );
        assert_eq!(indices, Val::int_vec(vec![0, 1, 2]));
        assert_eq!(indices_source.array_iter_reads(), 0);
    }

    #[test]
    fn view_empty_cardinality_select_sinks_skip_row_evaluation() {
        let collect_source = CountingView::root(&[1, 2, 3]);
        let collect_body = PipelineBody {
            stages: vec![Stage::UsizeBuiltin {
                method: crate::builtins::BuiltinMethod::Take,
                value: 0,
            }],
            stage_exprs: Vec::new(),
            sink: Sink::Collect,
            stage_kernels: vec![BodyKernel::Generic],
            sink_kernels: Vec::new(),
        };

        let collect = super::run_full(collect_source.clone(), &collect_body)
            .unwrap()
            .unwrap();
        assert_eq!(serde_json::Value::from(collect), serde_json::json!([]));
        assert_eq!(collect_source.scalar_reads(), 0);
        assert_eq!(collect_source.array_iter_reads(), 0);
        assert_eq!(collect_source.materialize_reads(), 0);

        let nth_source = CountingView::root(&[1, 2, 3]);
        let nth_body = PipelineBody {
            sink: Sink::Nth(1),
            ..collect_body.clone()
        };

        let nth = super::run_full(nth_source.clone(), &nth_body)
            .unwrap()
            .unwrap();
        assert_eq!(nth, Val::Null);
        assert_eq!(nth_source.scalar_reads(), 0);
        assert_eq!(nth_source.array_iter_reads(), 0);
        assert_eq!(nth_source.materialize_reads(), 0);

        let many_source = CountingView::root(&[1, 2, 3]);
        let many_body = PipelineBody {
            sink: Sink::SelectMany {
                n: 2,
                from_end: false,
            },
            ..nth_body
        };

        let many = super::run_full(many_source.clone(), &many_body)
            .unwrap()
            .unwrap();
        assert_eq!(serde_json::Value::from(many), serde_json::json!([]));
        assert_eq!(many_source.scalar_reads(), 0);
        assert_eq!(many_source.array_iter_reads(), 0);
        assert_eq!(many_source.materialize_reads(), 0);

        let extreme_source = CountingView::root(&[1, 2, 3]);
        let extreme_body = PipelineBody {
            sink: Sink::ArgExtreme(ArgExtremeSinkSpec {
                op: crate::builtins::BuiltinArgExtremeSink::MaxBy,
                key: Arc::new(crate::vm::Program::new(Vec::new(), "")),
                key_expr: None,
            }),
            sink_kernels: vec![BodyKernel::Current],
            ..many_body
        };

        let extreme = super::run_full(extreme_source.clone(), &extreme_body)
            .unwrap()
            .unwrap();
        assert_eq!(extreme, Val::Null);
        assert_eq!(extreme_source.scalar_reads(), 0);
        assert_eq!(extreme_source.array_iter_reads(), 0);
        assert_eq!(extreme_source.materialize_reads(), 0);

        let distinct_source = CountingView::root(&[1, 2, 3]);
        let distinct_body = PipelineBody {
            sink: Sink::ApproxCountDistinct,
            sink_kernels: Vec::new(),
            ..extreme_body
        };

        let distinct = super::run_full(distinct_source.clone(), &distinct_body)
            .unwrap()
            .unwrap();
        assert_eq!(distinct, Val::Int(0));
        assert_eq!(distinct_source.scalar_reads(), 0);
        assert_eq!(distinct_source.array_iter_reads(), 0);
        assert_eq!(distinct_source.materialize_reads(), 0);

        let count_source = CountingView::root(&[1, 2, 3]);
        let count_body = PipelineBody {
            sink: Sink::Reducer(ReducerSpec::count()),
            ..distinct_body
        };

        let count = super::run_full(count_source.clone(), &count_body)
            .unwrap()
            .unwrap();
        assert_eq!(count, Val::Int(0));
        assert_eq!(count_source.scalar_reads(), 0);
        assert_eq!(count_source.array_iter_reads(), 0);
        assert_eq!(count_source.materialize_reads(), 0);

        let first_source = CountingView::root(&[1, 2, 3]);
        let first_body = PipelineBody {
            sink: Sink::Terminal(crate::builtins::BuiltinMethod::First),
            ..count_body
        };

        let first = super::run_full(first_source.clone(), &first_body)
            .unwrap()
            .unwrap();
        assert_eq!(first, Val::Null);
        assert_eq!(first_source.scalar_reads(), 0);
        assert_eq!(first_source.array_iter_reads(), 0);
        assert_eq!(first_source.materialize_reads(), 0);
    }

    #[test]
    fn view_empty_cardinality_builtin_selectors_skip_row_evaluation() {
        for sink in [
            Sink::Terminal(crate::builtins::BuiltinMethod::First),
            Sink::Terminal(crate::builtins::BuiltinMethod::Last),
        ] {
            let source = CountingView::root(&[1, 2, 3]);
            let body = PipelineBody {
                stages: vec![Stage::Filter(
                    Arc::new(crate::vm::Program::new(Vec::new(), "")),
                    crate::builtins::BuiltinViewStage::Filter,
                )],
                stage_exprs: Vec::new(),
                sink,
                stage_kernels: vec![BodyKernel::ConstBool(false)],
                sink_kernels: Vec::new(),
            };

            let out = super::run_full(source.clone(), &body).unwrap().unwrap();

            assert_eq!(out, Val::Null);
            assert_eq!(source.scalar_reads(), 0);
            assert_eq!(source.array_iter_reads(), 0);
            assert_eq!(source.materialize_reads(), 0);
        }
    }

    #[test]
    fn view_empty_cardinality_predicate_count_skips_predicate_evaluation() {
        let source = CountingView::root(&[1, 2, 3]);
        let body = PipelineBody {
            stages: vec![Stage::UsizeBuiltin {
                method: crate::builtins::BuiltinMethod::Take,
                value: 0,
            }],
            stage_exprs: Vec::new(),
            sink: Sink::Reducer(ReducerSpec::count_with_predicate(
                Arc::new(crate::vm::Program::new(Vec::new(), "")),
                None,
            )),
            stage_kernels: vec![BodyKernel::Generic],
            sink_kernels: vec![BodyKernel::CurrentCmpLit(BinOp::Gt, Val::Int(0))],
        };

        let out = super::run_full(source.clone(), &body).unwrap().unwrap();

        assert_eq!(out, Val::Int(0));
        assert_eq!(source.scalar_reads(), 0);
        assert_eq!(source.array_iter_reads(), 0);
        assert_eq!(source.materialize_reads(), 0);
    }

    #[test]
    fn view_constant_predicate_count_uses_source_length_without_iterating_rows() {
        let true_source = CountingView::root(&[1, 2, 3, 4, 5]);
        let true_body = PipelineBody {
            stages: vec![
                Stage::UsizeBuiltin {
                    method: crate::builtins::BuiltinMethod::Skip,
                    value: 1,
                },
                Stage::UsizeBuiltin {
                    method: crate::builtins::BuiltinMethod::Take,
                    value: 3,
                },
            ],
            stage_exprs: Vec::new(),
            sink: Sink::Reducer(ReducerSpec::count_with_predicate(
                Arc::new(crate::vm::Program::new(Vec::new(), "")),
                None,
            )),
            stage_kernels: vec![BodyKernel::Generic, BodyKernel::Generic],
            sink_kernels: vec![BodyKernel::ConstBool(true)],
        };

        let true_count = super::run_full(true_source.clone(), &true_body)
            .unwrap()
            .unwrap();

        assert_eq!(true_count, Val::Int(3));
        assert_eq!(true_source.scalar_reads(), 1);
        assert_eq!(true_source.array_iter_reads(), 0);
        assert_eq!(true_source.materialize_reads(), 0);

        let false_source = CountingView::root(&[1, 2, 3, 4, 5]);
        let false_body = PipelineBody {
            sink_kernels: vec![BodyKernel::ConstBool(false)],
            ..true_body
        };

        let false_count = super::run_full(false_source.clone(), &false_body)
            .unwrap()
            .unwrap();

        assert_eq!(false_count, Val::Int(0));
        assert_eq!(false_source.scalar_reads(), 1);
        assert_eq!(false_source.array_iter_reads(), 0);
        assert_eq!(false_source.materialize_reads(), 0);
    }

    #[test]
    fn view_empty_cardinality_numeric_reducers_skip_row_evaluation() {
        for (op, expected) in [
            (NumOp::Sum, Val::Int(0)),
            (NumOp::Avg, Val::Null),
            (NumOp::Min, Val::Null),
            (NumOp::Max, Val::Null),
        ] {
            let source = CountingView::root(&[1, 2, 3]);
            let body = PipelineBody {
                stages: vec![Stage::UsizeBuiltin {
                    method: crate::builtins::BuiltinMethod::Take,
                    value: 0,
                }],
                stage_exprs: Vec::new(),
                sink: Sink::Reducer(ReducerSpec {
                    op: ReducerOp::Numeric(op),
                    predicate: None,
                    projection: None,
                    predicate_expr: None,
                    projection_expr: None,
                }),
                stage_kernels: vec![BodyKernel::Generic],
                sink_kernels: Vec::new(),
            };

            let out = super::run_full(source.clone(), &body).unwrap().unwrap();

            assert_eq!(out, expected, "{op:?}");
            assert_eq!(source.scalar_reads(), 0, "{op:?}");
            assert_eq!(source.array_iter_reads(), 0, "{op:?}");
            assert_eq!(source.materialize_reads(), 0, "{op:?}");
        }
    }

    #[test]
    fn view_constant_cardinality_stages_use_source_length_without_iterating_rows() {
        let count_source = CountingView::root(&[1, 2, 3, 4, 5]);
        let count_body = PipelineBody {
            stages: vec![
                Stage::Filter(
                    Arc::new(crate::vm::Program::new(Vec::new(), "")),
                    crate::builtins::BuiltinViewStage::Filter,
                ),
                Stage::UsizeBuiltin {
                    method: crate::builtins::BuiltinMethod::Take,
                    value: 2,
                },
            ],
            stage_exprs: Vec::new(),
            sink: Sink::Reducer(ReducerSpec::count()),
            stage_kernels: vec![BodyKernel::ConstBool(true), BodyKernel::Generic],
            sink_kernels: Vec::new(),
        };

        let count = super::run_full(count_source.clone(), &count_body)
            .unwrap()
            .unwrap();
        assert_eq!(count, Val::Int(2));
        assert_eq!(count_source.scalar_reads(), 1);
        assert_eq!(count_source.array_iter_reads(), 0);
        assert_eq!(count_source.materialize_reads(), 0);

        let empty_source = CountingView::root(&[1, 2, 3]);
        let empty_body = PipelineBody {
            stages: vec![Stage::Filter(
                Arc::new(crate::vm::Program::new(Vec::new(), "")),
                crate::builtins::BuiltinViewStage::Filter,
            )],
            stage_exprs: Vec::new(),
            sink: Sink::Collect,
            stage_kernels: vec![BodyKernel::ConstBool(false)],
            sink_kernels: Vec::new(),
        };

        let empty = super::run_full(empty_source.clone(), &empty_body)
            .unwrap()
            .unwrap();
        assert_eq!(serde_json::Value::from(empty), serde_json::json!([]));
        assert_eq!(empty_source.scalar_reads(), 0);
        assert_eq!(empty_source.array_iter_reads(), 0);
        assert_eq!(empty_source.materialize_reads(), 0);

        let drop_source = CountingView::root(&[1, 2, 3]);
        let drop_body = PipelineBody {
            stages: vec![Stage::ExprBuiltin {
                method: crate::builtins::BuiltinMethod::DropWhile,
                body: Arc::new(crate::vm::Program::new(Vec::new(), "")),
            }],
            stage_exprs: Vec::new(),
            sink: Sink::Reducer(ReducerSpec {
                op: ReducerOp::Numeric(NumOp::Sum),
                predicate: None,
                projection: None,
                predicate_expr: None,
                projection_expr: None,
            }),
            stage_kernels: vec![BodyKernel::ConstBool(true)],
            sink_kernels: Vec::new(),
        };

        let dropped = super::run_full(drop_source.clone(), &drop_body)
            .unwrap()
            .unwrap();
        assert_eq!(dropped, Val::Int(0));
        assert_eq!(drop_source.scalar_reads(), 0);
        assert_eq!(drop_source.array_iter_reads(), 0);
        assert_eq!(drop_source.materialize_reads(), 0);
    }

    #[test]
    fn view_source_access_ignores_constant_true_predicate_stages() {
        let source = CountingView::root(&[1, 2, 3, 4, 5]);
        let body = PipelineBody {
            stages: vec![Stage::Filter(
                Arc::new(crate::vm::Program::new(Vec::new(), "")),
                crate::builtins::BuiltinViewStage::Filter,
            )],
            stage_exprs: Vec::new(),
            sink: Sink::Terminal(crate::builtins::BuiltinMethod::Last),
            stage_kernels: vec![BodyKernel::ConstBool(true)],
            sink_kernels: Vec::new(),
        };

        let out = super::run_full(source.clone(), &body).unwrap().unwrap();

        assert_eq!(out, Val::Int(5));
        assert_eq!(source.array_iter_reads(), 0);
        assert_eq!(source.materialize_reads(), 0);
    }

    #[test]
    fn view_full_runner_find_one_avoids_materializing_scalar_match() {
        let source = CountingView::root(&[1, 2, 3, 4]);
        let body = PipelineBody {
            stages: Vec::new(),
            stage_exprs: Vec::new(),
            sink: Sink::Predicate(PredicateSinkSpec {
                op: BuiltinPredicateSink::FindOne,
                predicate: Arc::new(crate::vm::Program::new(Vec::new(), "")),
                predicate_expr: None,
            }),
            stage_kernels: Vec::new(),
            sink_kernels: vec![BodyKernel::CurrentCmpLit(BinOp::Eq, Val::Int(3))],
        };

        let out = super::run_full(source.clone(), &body).unwrap().unwrap();

        assert_eq!(out, Val::Int(3));
        assert_eq!(source.scalar_reads(), 5);
        assert_eq!(source.materialize_reads(), 0);
    }

    #[test]
    fn view_full_runner_find_one_preserves_exact_one_errors() {
        let empty_source = CountingView::root(&[1, 2, 3]);
        let empty_body = PipelineBody {
            stages: Vec::new(),
            stage_exprs: Vec::new(),
            sink: Sink::Predicate(PredicateSinkSpec {
                op: BuiltinPredicateSink::FindOne,
                predicate: Arc::new(crate::vm::Program::new(Vec::new(), "")),
                predicate_expr: None,
            }),
            stage_kernels: Vec::new(),
            sink_kernels: vec![BodyKernel::CurrentCmpLit(BinOp::Gt, Val::Int(9))],
        };
        let err = super::run_full(empty_source.clone(), &empty_body)
            .unwrap()
            .unwrap_err();
        assert_eq!(err.0, "find_one: expected exactly one element, got 0");
        assert_eq!(empty_source.materialize_reads(), 0);

        let multi_source = CountingView::root(&[1, 2, 3]);
        let multi_body = PipelineBody {
            sink_kernels: vec![BodyKernel::CurrentCmpLit(BinOp::Gt, Val::Int(1))],
            ..empty_body
        };
        let err = super::run_full(multi_source.clone(), &multi_body)
            .unwrap()
            .unwrap_err();
        assert_eq!(
            err.0,
            "find_one: expected exactly one element, got multiple"
        );
        assert_eq!(multi_source.materialize_reads(), 0);
    }

    #[test]
    fn view_full_runner_stops_when_membership_sink_result_is_decided() {
        let source = CountingView::root(&[1, 2, 3, 4]);
        let body = PipelineBody {
            stages: Vec::new(),
            stage_exprs: Vec::new(),
            sink: Sink::Membership(MembershipSinkSpec {
                op: BuiltinMembershipSink::Includes,
                target: MembershipSinkTarget::Literal(Val::Int(3)),
            }),
            stage_kernels: Vec::new(),
            sink_kernels: Vec::new(),
        };

        let out = super::run_full(source.clone(), &body).unwrap().unwrap();

        assert_eq!(out, Val::Bool(true));
        assert_eq!(source.scalar_reads(), 3);
        assert_eq!(source.materialize_reads(), 0);
    }

    #[test]
    fn view_full_runner_stops_when_index_sink_matches() {
        let source = CountingView::root(&[1, 2, 3, 4]);
        let body = PipelineBody {
            stages: Vec::new(),
            stage_exprs: Vec::new(),
            sink: Sink::Membership(MembershipSinkSpec {
                op: BuiltinMembershipSink::Index,
                target: MembershipSinkTarget::Literal(Val::Int(3)),
            }),
            stage_kernels: Vec::new(),
            sink_kernels: Vec::new(),
        };

        let out = super::run_full(source.clone(), &body).unwrap().unwrap();

        assert_eq!(out, Val::Int(2));
        assert_eq!(source.scalar_reads(), 3);
        assert_eq!(source.materialize_reads(), 0);
    }

    #[test]
    fn view_full_runner_handles_select_many_first_and_last() {
        let first_source = CountingView::root(&[1, 2, 3, 4]);
        let first_body = PipelineBody {
            stages: Vec::new(),
            stage_exprs: Vec::new(),
            sink: Sink::SelectMany {
                n: 2,
                from_end: false,
            },
            stage_kernels: Vec::new(),
            sink_kernels: Vec::new(),
        };

        let first = super::run_full(first_source.clone(), &first_body)
            .unwrap()
            .unwrap();
        let first_json: serde_json::Value = first.into();
        assert_eq!(first_json, serde_json::json!([1, 2]));
        assert_eq!(first_source.materialize_reads(), 0);

        let last_source = CountingView::root(&[1, 2, 3, 4]);
        let last_body = PipelineBody {
            sink: Sink::SelectMany {
                n: 2,
                from_end: true,
            },
            ..first_body
        };

        let last = super::run_full(last_source.clone(), &last_body)
            .unwrap()
            .unwrap();
        let last_json: serde_json::Value = last.into();
        assert_eq!(last_json, serde_json::json!([3, 4]));
        assert_eq!(last_source.materialize_reads(), 0);
    }

    #[test]
    fn view_full_runner_uses_direct_access_for_first_last_and_nth() {
        let first_source = CountingView::root(&[1, 2, 3, 4]);
        let first_body = PipelineBody {
            stages: Vec::new(),
            stage_exprs: Vec::new(),
            sink: Sink::Terminal(crate::builtins::BuiltinMethod::First),
            stage_kernels: Vec::new(),
            sink_kernels: Vec::new(),
        };
        let first = super::run_full(first_source.clone(), &first_body)
            .unwrap()
            .unwrap();
        let first_json: serde_json::Value = first.into();
        assert_eq!(first_json, serde_json::json!(1));
        assert_eq!(first_source.materialize_reads(), 0);
        assert_eq!(first_source.array_iter_reads(), 0);

        let last_source = CountingView::root(&[1, 2, 3, 4]);
        let last_body = PipelineBody {
            sink: Sink::Terminal(crate::builtins::BuiltinMethod::Last),
            ..first_body
        };
        let last = super::run_full(last_source.clone(), &last_body)
            .unwrap()
            .unwrap();
        let last_json: serde_json::Value = last.into();
        assert_eq!(last_json, serde_json::json!(4));
        assert_eq!(last_source.materialize_reads(), 0);
        assert_eq!(last_source.scalar_reads(), 2);
        assert_eq!(last_source.array_iter_reads(), 0);

        let nth_source = CountingView::root(&[1, 2, 3, 4]);
        let nth_body = PipelineBody {
            stages: Vec::new(),
            stage_exprs: Vec::new(),
            sink: Sink::Nth(2),
            stage_kernels: Vec::new(),
            sink_kernels: Vec::new(),
        };
        let nth = super::run_full(nth_source.clone(), &nth_body)
            .unwrap()
            .unwrap();
        let nth_json: serde_json::Value = nth.into();
        assert_eq!(nth_json, serde_json::json!(3));
        assert_eq!(nth_source.materialize_reads(), 0);
        assert_eq!(nth_source.scalar_reads(), 2);
        assert_eq!(nth_source.array_iter_reads(), 0);
    }

    #[test]
    fn view_frontier_zero_demand_does_not_touch_source() {
        let source = CountingView::root(&[1, 2, 3, 4]);
        let mut observed = 0usize;
        let mut vm = VM::new();

        super::drive_view_frontier(
            source.clone(),
            crate::exec::pipeline::SourceCapabilities::VIEW_ARRAY,
            &[],
            &[],
            crate::plan::demand::PullDemand::LastInput(0),
            &mut vm,
            |_, _| {
                observed += 1;
                Some(Ok(super::ViewRowAction::Emit))
            },
        )
        .unwrap()
        .unwrap();

        assert_eq!(observed, 0);
        assert_eq!(source.scalar_reads(), 0);
        assert_eq!(source.array_iter_reads(), 0);
        assert_eq!(source.materialize_reads(), 0);
    }

    #[test]
    fn view_frontier_forward_last_fallback_scans_all_rows() {
        let source = CountingView::root(&[1, 2, 3, 4]);
        let mut observed = 0usize;
        let mut vm = VM::new();
        let caps = crate::exec::pipeline::SourceCapabilities {
            reverse_stream: false,
            indexed_array_child: false,
            ..crate::exec::pipeline::SourceCapabilities::VIEW_ARRAY
        };

        super::drive_view_frontier(
            source.clone(),
            caps,
            &[],
            &[],
            crate::plan::demand::PullDemand::LastInput(1),
            &mut vm,
            |_, _| {
                observed += 1;
                Some(Ok(super::ViewRowAction::Emit))
            },
        )
        .unwrap()
        .unwrap();

        assert_eq!(observed, 4);
        assert_eq!(source.array_iter_reads(), 1);
    }

    #[test]
    fn view_frontier_ignores_overflowing_from_end_offset() {
        assert_eq!(crate::exec::pipeline::index_from_end(4, 0), Some(3));
        assert_eq!(crate::exec::pipeline::index_from_end(4, 3), Some(0));
        assert_eq!(crate::exec::pipeline::index_from_end(4, 4), None);
        assert_eq!(crate::exec::pipeline::index_from_end(4, usize::MAX), None);
    }

    #[test]
    fn view_suffix_sink_marks_reversed_select_many_for_last_input() {
        let sink = ViewSinkCapability::SelectMany {
            n: 2,
            from_end: true,
            source_reversed: false,
        };

        let adjusted = super::view_suffix_sink_for_demand(sink, PullDemand::LastInput(2), true);

        assert!(matches!(
            adjusted,
            ViewSinkCapability::SelectMany {
                n: 2,
                from_end: true,
                source_reversed: true
            }
        ));
    }

    #[test]
    fn view_sink_last_selection_uses_accumulator_metadata() {
        let first = ViewSinkCapability::Builtin {
            accumulator: BuiltinSinkAccumulator::SelectOne(BuiltinSelectionPosition::First),
            predicate_kernel: None,
            project_kernel: None,
            materialization: crate::builtins::BuiltinViewMaterialization::SinkFinalRow,
        };
        let last = ViewSinkCapability::Builtin {
            accumulator: BuiltinSinkAccumulator::SelectOne(BuiltinSelectionPosition::Last),
            predicate_kernel: None,
            project_kernel: None,
            materialization: crate::builtins::BuiltinViewMaterialization::SinkFinalRow,
        };
        let count = ViewSinkCapability::Builtin {
            accumulator: BuiltinSinkAccumulator::Count,
            predicate_kernel: None,
            project_kernel: None,
            materialization: crate::builtins::BuiltinViewMaterialization::Never,
        };

        assert!(!first.selects_from_end());
        assert!(last.selects_from_end());
        assert!(!count.selects_from_end());
    }

    #[test]
    fn view_runner_applies_late_map_only_to_demanded_rows() {
        let map_stage = Stage::Map(
            Arc::new(crate::vm::Program::new(Vec::new(), "")),
            crate::builtins::BuiltinViewStage::Map,
        );

        let first_source = CountingView::root(&[1, 2, 3, 4]);
        let first_body = PipelineBody {
            stages: vec![map_stage.clone()],
            stage_exprs: Vec::new(),
            sink: Sink::Terminal(crate::builtins::BuiltinMethod::First),
            stage_kernels: vec![BodyKernel::Current],
            sink_kernels: Vec::new(),
        };
        let mut vm = VM::new();
        let first =
            super::run_terminal_select_projection(first_source.clone(), &first_body, &mut vm)
                .unwrap()
                .unwrap();
        assert_eq!(first, Val::Int(1));
        assert_eq!(first_source.scalar_reads(), 2);
        assert_eq!(first_source.array_iter_reads(), 0);

        let last_source = CountingView::root(&[1, 2, 3, 4]);
        let last_body = PipelineBody {
            sink: Sink::Terminal(crate::builtins::BuiltinMethod::Last),
            ..first_body
        };
        let last = super::run_terminal_select_projection(last_source.clone(), &last_body, &mut vm)
            .unwrap()
            .unwrap();
        assert_eq!(last, Val::Int(4));
        assert_eq!(last_source.scalar_reads(), 2);
        assert_eq!(last_source.array_iter_reads(), 0);

        let take_source = CountingView::root(&[1, 2, 3, 4]);
        let take_body = PipelineBody {
            stages: vec![
                map_stage,
                Stage::UsizeBuiltin {
                    method: crate::builtins::BuiltinMethod::Take,
                    value: 3,
                },
            ],
            stage_exprs: Vec::new(),
            sink: Sink::Collect,
            stage_kernels: vec![BodyKernel::Current, BodyKernel::Generic],
            sink_kernels: Vec::new(),
        };
        let mut vm = VM::new();
        let take = super::run_terminal_collect(take_source.clone(), &take_body, &mut vm)
            .unwrap()
            .unwrap();
        let take_json: serde_json::Value = take.into();
        assert_eq!(take_json, serde_json::json!([1, 2, 3]));
        assert_eq!(take_source.scalar_reads(), 3);

        let nth_source = CountingView::root(&[1, 2, 3, 4]);
        let nth_body = PipelineBody {
            stages: vec![Stage::Map(
                Arc::new(crate::vm::Program::new(Vec::new(), "")),
                crate::builtins::BuiltinViewStage::Map,
            )],
            stage_exprs: Vec::new(),
            sink: Sink::Nth(2),
            stage_kernels: vec![BodyKernel::Current],
            sink_kernels: Vec::new(),
        };
        let nth = super::run_terminal_select_projection(nth_source.clone(), &nth_body, &mut vm)
            .unwrap()
            .unwrap();
        assert_eq!(nth, Val::Int(3));
        assert_eq!(nth_source.scalar_reads(), 2);
        assert_eq!(nth_source.array_iter_reads(), 0);
    }

    #[test]
    fn view_runner_streams_leading_reverse_without_materializing_source_rows() {
        let source = CountingView::root(&[1, 2, 3, 4]);
        let body = PipelineBody {
            stages: vec![
                Stage::reverse().unwrap(),
                Stage::UsizeBuiltin {
                    method: crate::builtins::BuiltinMethod::Take,
                    value: 2,
                },
            ],
            stage_exprs: Vec::new(),
            sink: Sink::Collect,
            stage_kernels: vec![BodyKernel::Generic, BodyKernel::Generic],
            sink_kernels: Vec::new(),
        };
        let env = Env::new(Val::Null);
        let mut vm = crate::vm::VM::new();

        let out = super::run_with_env_and_vm(source.clone(), &body, None, &env, &mut vm)
            .unwrap()
            .unwrap();

        assert_eq!(serde_json::Value::from(out), serde_json::json!([4, 3]));
        assert_eq!(source.array_iter_reads(), 1);
        assert_eq!(source.materialize_reads(), 2);
    }

    #[test]
    fn view_runner_cancels_even_leading_reverses_for_source_access() {
        let source = CountingView::root(&[1, 2, 3, 4]);
        let body = PipelineBody {
            stages: vec![Stage::reverse().unwrap(), Stage::reverse().unwrap()],
            stage_exprs: Vec::new(),
            sink: Sink::Terminal(crate::builtins::BuiltinMethod::First),
            stage_kernels: vec![BodyKernel::Generic, BodyKernel::Generic],
            sink_kernels: Vec::new(),
        };
        let env = Env::new(Val::Null);
        let mut vm = crate::vm::VM::new();

        let out = super::run_with_env_and_vm(source.clone(), &body, None, &env, &mut vm)
            .unwrap()
            .unwrap();

        assert_eq!(out, Val::Int(1));
        assert_eq!(source.array_iter_reads(), 0);
        assert_eq!(source.materialize_reads(), 0);
    }

    #[test]
    fn view_runner_seeks_leading_reverse_positional_selectors() {
        let first_source = CountingView::root(&[1, 2, 3, 4]);
        let first_body = PipelineBody {
            stages: vec![Stage::reverse().unwrap()],
            stage_exprs: Vec::new(),
            sink: Sink::Terminal(crate::builtins::BuiltinMethod::First),
            stage_kernels: vec![BodyKernel::Generic],
            sink_kernels: Vec::new(),
        };
        let env = Env::new(Val::Null);
        let mut vm = crate::vm::VM::new();

        let first =
            super::run_with_env_and_vm(first_source.clone(), &first_body, None, &env, &mut vm)
                .unwrap()
                .unwrap();

        assert_eq!(first, Val::Int(4));
        assert_eq!(first_source.array_iter_reads(), 0);
        assert_eq!(first_source.materialize_reads(), 0);

        let nth_source = CountingView::root(&[1, 2, 3, 4]);
        let nth_body = PipelineBody {
            sink: Sink::Nth(2),
            ..first_body
        };
        let nth = super::run_with_env_and_vm(nth_source.clone(), &nth_body, None, &env, &mut vm)
            .unwrap()
            .unwrap();

        assert_eq!(nth, Val::Int(2));
        assert_eq!(nth_source.array_iter_reads(), 0);
        assert_eq!(nth_source.materialize_reads(), 0);
    }

    #[test]
    fn view_runner_reverses_filtered_rows_without_materializing_prefix() {
        let source = CountingView::root(&[1, 2, 3, 4]);
        let body = PipelineBody {
            stages: vec![
                Stage::Filter(
                    Arc::new(crate::vm::Program::new(Vec::new(), "")),
                    crate::builtins::BuiltinViewStage::Filter,
                ),
                Stage::reverse().unwrap(),
                Stage::UsizeBuiltin {
                    method: crate::builtins::BuiltinMethod::Take,
                    value: 2,
                },
            ],
            stage_exprs: Vec::new(),
            sink: Sink::Collect,
            stage_kernels: vec![
                BodyKernel::CurrentCmpLit(BinOp::Gt, Val::Int(1)),
                BodyKernel::Generic,
                BodyKernel::Generic,
            ],
            sink_kernels: Vec::new(),
        };
        let env = Env::new(Val::Null);
        let mut vm = crate::vm::VM::new();

        let out = super::run_with_env_and_vm(source.clone(), &body, None, &env, &mut vm)
            .unwrap()
            .unwrap();

        assert_eq!(serde_json::Value::from(out), serde_json::json!([4, 3]));
        assert_eq!(source.array_iter_reads(), 1);
        assert_eq!(source.materialize_reads(), 2);
    }

    #[test]
    fn view_runner_skips_empty_reverse_prefix_without_touching_source() {
        let source = CountingView::root(&[1, 2, 3, 4]);
        let body = PipelineBody {
            stages: vec![
                Stage::UsizeBuiltin {
                    method: crate::builtins::BuiltinMethod::Take,
                    value: 0,
                },
                Stage::reverse().unwrap(),
            ],
            stage_exprs: Vec::new(),
            sink: Sink::Collect,
            stage_kernels: vec![BodyKernel::Generic, BodyKernel::Generic],
            sink_kernels: Vec::new(),
        };
        let env = Env::new(Val::Null);
        let mut vm = crate::vm::VM::new();

        let out = super::run_with_env_and_vm(source.clone(), &body, None, &env, &mut vm)
            .unwrap()
            .unwrap();

        assert_eq!(serde_json::Value::from(out), serde_json::json!([]));
        assert_eq!(source.array_iter_reads(), 0);
        assert_eq!(source.materialize_reads(), 0);
    }

    #[test]
    fn view_runner_skips_zero_demand_reverse_suffix_without_touching_source() {
        let source = CountingView::root(&[1, 2, 3, 4]);
        let body = PipelineBody {
            stages: vec![
                Stage::Filter(
                    Arc::new(crate::vm::Program::new(Vec::new(), "")),
                    crate::builtins::BuiltinViewStage::Filter,
                ),
                Stage::reverse().unwrap(),
                Stage::UsizeBuiltin {
                    method: crate::builtins::BuiltinMethod::Take,
                    value: 0,
                },
            ],
            stage_exprs: Vec::new(),
            sink: Sink::Collect,
            stage_kernels: vec![
                BodyKernel::CurrentCmpLit(BinOp::Gt, Val::Int(1)),
                BodyKernel::Generic,
                BodyKernel::Generic,
            ],
            sink_kernels: Vec::new(),
        };
        let env = Env::new(Val::Null);
        let mut vm = crate::vm::VM::new();

        let out = super::run_with_env_and_vm(source.clone(), &body, None, &env, &mut vm)
            .unwrap()
            .unwrap();

        assert_eq!(serde_json::Value::from(out), serde_json::json!([]));
        assert_eq!(source.array_iter_reads(), 0);
        assert_eq!(source.materialize_reads(), 0);
    }

    #[test]
    fn view_runner_keeps_sorted_dedup_barrier_borrowed_until_suffix() {
        let source = CountingView::root(&[3, 1, 2, 3, 2]);
        let body = PipelineBody {
            stages: vec![
                Stage::Filter(
                    Arc::new(crate::vm::Program::new(Vec::new(), "")),
                    crate::builtins::BuiltinViewStage::Filter,
                ),
                Stage::SortedDedup(None),
                Stage::UsizeBuiltin {
                    method: crate::builtins::BuiltinMethod::Take,
                    value: 2,
                },
            ],
            stage_exprs: Vec::new(),
            sink: Sink::Collect,
            stage_kernels: vec![
                BodyKernel::CurrentCmpLit(BinOp::Gt, Val::Int(1)),
                BodyKernel::Generic,
                BodyKernel::Generic,
            ],
            sink_kernels: Vec::new(),
        };
        let env = Env::new(Val::Null);
        let mut vm = crate::vm::VM::new();

        let out = super::run_with_env_and_vm(source.clone(), &body, None, &env, &mut vm)
            .unwrap()
            .unwrap();

        assert_eq!(serde_json::Value::from(out), serde_json::json!([2, 3]));
        assert_eq!(source.array_iter_reads(), 1);
        assert_eq!(source.materialize_reads(), 2);
    }

    #[test]
    fn view_runner_uses_view_key_for_keyed_sorted_dedup() {
        let source = CountingKeyedObjectView::root(&[(2, 20), (1, 10), (2, 21), (3, 30)]);
        let body = PipelineBody {
            stages: vec![
                Stage::SortedDedup(Some(Arc::new(crate::vm::Program::new(Vec::new(), "")))),
                Stage::UsizeBuiltin {
                    method: crate::builtins::BuiltinMethod::Take,
                    value: 2,
                },
            ],
            stage_exprs: Vec::new(),
            sink: Sink::Collect,
            stage_kernels: vec![BodyKernel::FieldRead(Arc::from("k")), BodyKernel::Generic],
            sink_kernels: Vec::new(),
        };
        let env = Env::new(Val::Null);
        let mut vm = crate::vm::VM::new();

        let out = super::run_with_env_and_vm(source.clone(), &body, None, &env, &mut vm)
            .unwrap()
            .unwrap();

        assert_eq!(
            serde_json::Value::from(out),
            serde_json::json!([{"k": 1, "v": 10}, {"k": 2, "v": 20}])
        );
        assert_eq!(source.array_iter_reads(), 1);
        assert_eq!(source.materialize_reads(), 2);
    }

    #[test]
    fn terminal_collect_skips_empty_prefix_without_iterating_rows() {
        let source = CountingView::root(&[1, 2, 3]);
        let body = PipelineBody {
            stages: vec![Stage::UsizeBuiltin {
                method: crate::builtins::BuiltinMethod::Take,
                value: 0,
            }],
            stage_exprs: Vec::new(),
            sink: Sink::Collect,
            stage_kernels: vec![BodyKernel::Generic],
            sink_kernels: Vec::new(),
        };
        let mut vm = crate::vm::VM::new();

        let out = super::run_terminal_collect(source.clone(), &body, &mut vm)
            .unwrap()
            .unwrap();

        assert_eq!(serde_json::Value::from(out), serde_json::json!([]));
        assert_eq!(source.scalar_reads(), 0);
        assert_eq!(source.array_iter_reads(), 0);
        assert_eq!(source.materialize_reads(), 0);
    }

    #[test]
    fn terminal_select_projection_skips_empty_prefix_without_iterating_rows() {
        let source = CountingView::root(&[1, 2, 3]);
        let body = PipelineBody {
            stages: vec![
                Stage::UsizeBuiltin {
                    method: crate::builtins::BuiltinMethod::Take,
                    value: 0,
                },
                Stage::Map(
                    Arc::new(crate::vm::Program::new(Vec::new(), "")),
                    crate::builtins::BuiltinViewStage::Map,
                ),
            ],
            stage_exprs: Vec::new(),
            sink: Sink::Terminal(crate::builtins::BuiltinMethod::First),
            stage_kernels: vec![BodyKernel::Generic, BodyKernel::Current],
            sink_kernels: Vec::new(),
        };
        let mut vm = crate::vm::VM::new();

        let out = super::run_terminal_select_projection(source.clone(), &body, &mut vm)
            .unwrap()
            .unwrap();

        assert_eq!(out, Val::Null);
        assert_eq!(source.scalar_reads(), 0);
        assert_eq!(source.array_iter_reads(), 0);
        assert_eq!(source.materialize_reads(), 0);
    }

    #[test]
    fn view_full_runner_handles_literal_membership_sinks_without_materializing_scalars() {
        let includes_source = CountingView::root(&[1, 2, 3, 4]);
        let includes_body = PipelineBody {
            stages: Vec::new(),
            stage_exprs: Vec::new(),
            sink: Sink::Membership(MembershipSinkSpec {
                op: BuiltinMembershipSink::Includes,
                target: MembershipSinkTarget::Literal(Val::Int(3)),
            }),
            stage_kernels: Vec::new(),
            sink_kernels: Vec::new(),
        };

        let includes = super::run_full(includes_source.clone(), &includes_body)
            .unwrap()
            .unwrap();
        assert_eq!(includes, Val::Bool(true));
        assert_eq!(includes_source.materialize_reads(), 0);
        assert_eq!(includes_source.scalar_reads(), 3);

        let index_source = CountingView::root(&[1, 2, 3, 4]);
        let index_body = PipelineBody {
            sink: Sink::Membership(MembershipSinkSpec {
                op: BuiltinMembershipSink::Index,
                target: MembershipSinkTarget::Literal(Val::Int(4)),
            }),
            ..includes_body
        };

        let index = super::run_full(index_source.clone(), &index_body)
            .unwrap()
            .unwrap();
        assert_eq!(index, Val::Int(3));
        assert_eq!(index_source.materialize_reads(), 0);
        assert_eq!(index_source.scalar_reads(), 4);

        let indices_source = CountingView::root(&[1, 2, 1, 3]);
        let indices_body = PipelineBody {
            sink: Sink::Membership(MembershipSinkSpec {
                op: BuiltinMembershipSink::IndicesOf,
                target: MembershipSinkTarget::Literal(Val::Int(1)),
            }),
            ..index_body
        };

        let indices = super::run_full(indices_source.clone(), &indices_body)
            .unwrap()
            .unwrap();
        let indices_json: serde_json::Value = indices.into();
        assert_eq!(indices_json, serde_json::json!([0, 2]));
        assert_eq!(indices_source.materialize_reads(), 0);
        assert_eq!(indices_source.scalar_reads(), 4);
    }

    #[test]
    fn view_membership_sinks_match_compound_values_deeply() {
        let tape = crate::data::tape::TapeData::parse(
            br#"[{"id":1,"tags":["a"]},{"id":2,"tags":["b","c"]},["nested",3]]"#.to_vec(),
        )
        .unwrap();
        let rows = TapeView::root(&tape);
        tape.reset_materialized_subtrees();

        let object_body = PipelineBody {
            stages: Vec::new(),
            stage_exprs: Vec::new(),
            sink: Sink::Membership(MembershipSinkSpec {
                op: BuiltinMembershipSink::Includes,
                target: MembershipSinkTarget::Literal(Val::from(&serde_json::json!({
                    "id": 2,
                    "tags": ["b", "c"]
                }))),
            }),
            stage_kernels: Vec::new(),
            sink_kernels: Vec::new(),
        };
        let object_out = super::run_full(rows, &object_body).unwrap().unwrap();
        assert_eq!(object_out, Val::Bool(true));
        assert_eq!(tape.materialized_subtrees(), 0);

        let array_body = PipelineBody {
            sink: Sink::Membership(MembershipSinkSpec {
                op: BuiltinMembershipSink::Index,
                target: MembershipSinkTarget::Literal(Val::from(&serde_json::json!(["nested", 3]))),
            }),
            ..object_body
        };
        let array_out = super::run_full(rows, &array_body).unwrap().unwrap();
        assert_eq!(array_out, Val::Int(2));
        assert_eq!(tape.materialized_subtrees(), 0);
    }

    #[test]
    fn view_full_runner_evaluates_dynamic_membership_targets_once() {
        let mut root = IndexMap::new();
        root.insert(Arc::<str>::from("needle"), Val::Int(3));
        let env = Env::new(Val::obj(root));
        let target = Arc::new(Compiler::compile_str("$.needle").unwrap());

        let includes_source = CountingView::root(&[1, 2, 3, 4]);
        let includes_body = PipelineBody {
            stages: Vec::new(),
            stage_exprs: Vec::new(),
            sink: Sink::Membership(MembershipSinkSpec {
                op: BuiltinMembershipSink::Includes,
                target: MembershipSinkTarget::Program(Arc::clone(&target)),
            }),
            stage_kernels: Vec::new(),
            sink_kernels: Vec::new(),
        };

        let mut vm = crate::vm::VM::new();
        let includes =
            super::run_full_with_env(includes_source.clone(), &includes_body, Some(&env), &mut vm)
                .unwrap()
                .unwrap();
        assert_eq!(includes, Val::Bool(true));
        assert_eq!(includes_source.materialize_reads(), 0);
        assert_eq!(includes_source.scalar_reads(), 3);

        let index_source = CountingView::root(&[1, 2, 3, 4]);
        let index_body = PipelineBody {
            sink: Sink::Membership(MembershipSinkSpec {
                op: BuiltinMembershipSink::Index,
                target: MembershipSinkTarget::Program(Arc::clone(&target)),
            }),
            ..includes_body
        };

        let index =
            super::run_full_with_env(index_source.clone(), &index_body, Some(&env), &mut vm)
                .unwrap()
                .unwrap();
        assert_eq!(index, Val::Int(2));
        assert_eq!(index_source.materialize_reads(), 0);
        assert_eq!(index_source.scalar_reads(), 3);

        let indices_source = CountingView::root(&[3, 1, 3, 4]);
        let indices_body = PipelineBody {
            sink: Sink::Membership(MembershipSinkSpec {
                op: BuiltinMembershipSink::IndicesOf,
                target: MembershipSinkTarget::Program(target),
            }),
            ..index_body
        };

        let indices =
            super::run_full_with_env(indices_source.clone(), &indices_body, Some(&env), &mut vm)
                .unwrap()
                .unwrap();
        let indices_json: serde_json::Value = indices.into();
        assert_eq!(indices_json, serde_json::json!([0, 2]));
        assert_eq!(indices_source.materialize_reads(), 0);
        assert_eq!(indices_source.scalar_reads(), 4);
    }

    #[test]
    fn view_full_runner_handles_arg_extreme_sinks_lazily() {
        let max_source = CountingView::root(&[2, 1, 4, 3]);
        let max_body = PipelineBody {
            stages: Vec::new(),
            stage_exprs: Vec::new(),
            sink: Sink::ArgExtreme(ArgExtremeSinkSpec {
                op: crate::builtins::BuiltinArgExtremeSink::MaxBy,
                key: Arc::new(crate::vm::Program::new(Vec::new(), "")),
                key_expr: None,
            }),
            stage_kernels: Vec::new(),
            sink_kernels: vec![BodyKernel::Current],
        };

        let max = super::run_full(max_source.clone(), &max_body)
            .unwrap()
            .unwrap();
        assert_eq!(max, Val::Int(4));
        assert_eq!(max_source.scalar_reads(), 6);
        assert_eq!(max_source.materialize_reads(), 0);

        let min_source = CountingView::root(&[3, 4, 1, 2]);
        let min_body = PipelineBody {
            sink: Sink::ArgExtreme(ArgExtremeSinkSpec {
                op: crate::builtins::BuiltinArgExtremeSink::MinBy,
                key: Arc::new(crate::vm::Program::new(Vec::new(), "")),
                key_expr: None,
            }),
            ..max_body
        };

        let min = super::run_full(min_source.clone(), &min_body)
            .unwrap()
            .unwrap();
        assert_eq!(min, Val::Int(1));
        assert_eq!(min_source.scalar_reads(), 6);
        assert_eq!(min_source.materialize_reads(), 0);
    }

    #[test]
    fn arg_extreme_view_sink_materializes_only_final_compound_winner() {
        let tape = crate::data::tape::TapeData::parse(
            br#"[{"id":1,"score":1},{"id":2,"score":2},{"id":3,"score":3},{"id":4,"score":4}]"#
                .to_vec(),
        )
        .unwrap();
        tape.reset_materialized_subtrees();
        let body = PipelineBody {
            stages: Vec::new(),
            stage_exprs: Vec::new(),
            sink: Sink::ArgExtreme(ArgExtremeSinkSpec {
                op: crate::builtins::BuiltinArgExtremeSink::MaxBy,
                key: Arc::new(crate::vm::Program::new(Vec::new(), "")),
                key_expr: None,
            }),
            stage_kernels: Vec::new(),
            sink_kernels: vec![BodyKernel::FieldRead(Arc::from("score"))],
        };
        let env = Env::new(Val::Null);
        let mut vm = crate::vm::VM::new();

        let out = super::run_with_env_and_vm(TapeView::root(&tape), &body, None, &env, &mut vm)
            .unwrap()
            .unwrap();

        assert_eq!(
            serde_json::Value::from(out),
            serde_json::json!({"id": 4, "score": 4})
        );
        assert_eq!(tape.materialized_subtrees(), 1);
    }

    #[test]
    fn terminal_collect_plan_accepts_view_native_prefix_and_final_map() {
        let body = PipelineBody {
            stages: vec![
                Stage::Filter(
                    Arc::new(crate::vm::Program::new(Vec::new(), "")),
                    crate::builtins::BuiltinViewStage::Filter,
                ),
                Stage::Map(
                    Arc::new(crate::vm::Program::new(Vec::new(), "")),
                    crate::builtins::BuiltinViewStage::Map,
                ),
            ],
            stage_exprs: Vec::new(),
            sink: Sink::Collect,
            stage_kernels: vec![
                BodyKernel::CurrentCmpLit(BinOp::Gt, Val::Int(1)),
                BodyKernel::Current,
            ],
            sink_kernels: Vec::new(),
        };

        let plan = super::terminal_collect_plan(&body).unwrap();

        assert_eq!(plan.prefix.len(), 1);
        assert!(matches!(plan.prefix[0], ViewStageCapability::Filter { .. }));
        assert!(matches!(plan.collect_program.kernel(), BodyKernel::Current));
    }

    #[test]
    fn terminal_collect_plan_accepts_current_row_collect_without_final_map() {
        let body = PipelineBody {
            stages: vec![
                Stage::Filter(
                    Arc::new(crate::vm::Program::new(Vec::new(), "")),
                    crate::builtins::BuiltinViewStage::Filter,
                ),
                Stage::UsizeBuiltin {
                    method: crate::builtins::BuiltinMethod::Take,
                    value: 1,
                },
            ],
            stage_exprs: Vec::new(),
            sink: Sink::Collect,
            stage_kernels: vec![
                BodyKernel::CurrentCmpLit(BinOp::Gt, Val::Int(1)),
                BodyKernel::Generic,
            ],
            sink_kernels: Vec::new(),
        };

        let plan = super::terminal_collect_plan(&body).unwrap();

        assert_eq!(plan.prefix.len(), 2);
        assert!(matches!(plan.prefix[0], ViewStageCapability::Filter { .. }));
        assert!(matches!(plan.prefix[1], ViewStageCapability::Take(1)));
        assert!(matches!(plan.collect_program.kernel(), BodyKernel::Current));
    }

    #[test]
    fn terminal_collect_plan_composes_trailing_projection_builtins() {
        let call = crate::builtins::BuiltinCall {
            method: crate::builtins::BuiltinMethod::Upper,
            args: crate::builtins::BuiltinArgs::None,
        };
        assert!(view_scalar_projection(call.id()));
        let body = PipelineBody {
            stages: vec![
                Stage::Map(
                    Arc::new(crate::vm::Program::new(Vec::new(), "")),
                    crate::builtins::BuiltinViewStage::Map,
                ),
                Stage::Builtin(call),
            ],
            stage_exprs: Vec::new(),
            sink: Sink::Collect,
            stage_kernels: vec![
                BodyKernel::FieldRead(Arc::from("name")),
                BodyKernel::Generic,
            ],
            sink_kernels: Vec::new(),
        };

        let plan = super::terminal_collect_plan(&body).unwrap();

        assert!(plan.prefix.is_empty());
        assert!(matches!(
            plan.collect_program.kernel(),
            BodyKernel::Compose { .. }
        ));
    }

    #[test]
    fn terminal_collect_plan_composes_trailing_object_key_builtins() {
        let call = crate::builtins::BuiltinCall {
            method: crate::builtins::BuiltinMethod::HasKey,
            args: crate::builtins::BuiltinArgs::Str(Arc::from("isbn")),
        };
        assert!(crate::builtins::registry::view_object_projection(call.id()).is_some());
        let body = PipelineBody {
            stages: vec![Stage::Builtin(call)],
            stage_exprs: Vec::new(),
            sink: Sink::Collect,
            stage_kernels: vec![BodyKernel::Generic],
            sink_kernels: Vec::new(),
        };

        let plan = super::terminal_collect_plan(&body).unwrap();

        assert!(plan.prefix.is_empty());
        assert!(matches!(
            plan.collect_program.kernel(),
            BodyKernel::BuiltinCall { .. }
        ));
    }

    #[test]
    fn terminal_collect_current_row_runner_stops_after_demand_is_met() {
        let source = CountingView::root(&[1, 2, 3]);
        let body = PipelineBody {
            stages: vec![
                Stage::Filter(
                    Arc::new(crate::vm::Program::new(Vec::new(), "")),
                    crate::builtins::BuiltinViewStage::Filter,
                ),
                Stage::UsizeBuiltin {
                    method: crate::builtins::BuiltinMethod::Take,
                    value: 1,
                },
            ],
            stage_exprs: Vec::new(),
            sink: Sink::Collect,
            stage_kernels: vec![
                BodyKernel::CurrentCmpLit(BinOp::Gt, Val::Int(1)),
                BodyKernel::Generic,
            ],
            sink_kernels: Vec::new(),
        };

        let mut vm = VM::new();
        let out = super::run_terminal_collect(source.clone(), &body, &mut vm)
            .unwrap()
            .unwrap();

        let out_json: serde_json::Value = out.into();
        assert_eq!(out_json, serde_json::json!([2]));
        assert_eq!(source.scalar_reads(), 3);
    }

    #[test]
    fn terminal_collect_accepts_flat_map_frontier_prefix() {
        let source = Val::from(&serde_json::json!([
            {"items": [1, 2, 3]},
            {"items": [4]}
        ]));
        let body = PipelineBody {
            stages: vec![
                Stage::FlatMap(
                    Arc::new(crate::vm::Program::new(Vec::new(), "")),
                    crate::builtins::BuiltinViewStage::FlatMap,
                ),
                Stage::UsizeBuiltin {
                    method: crate::builtins::BuiltinMethod::Take,
                    value: 2,
                },
            ],
            stage_exprs: Vec::new(),
            sink: Sink::Collect,
            stage_kernels: vec![
                BodyKernel::FieldRead(Arc::from("items")),
                BodyKernel::Generic,
            ],
            sink_kernels: Vec::new(),
        };

        let plan = super::terminal_collect_plan(&body).unwrap();
        assert_eq!(plan.prefix.len(), 2);
        assert!(matches!(
            plan.prefix[0],
            ViewStageCapability::FlatMap { .. }
        ));

        let mut vm = VM::new();
        let out = super::run_terminal_collect(ValView::new(&source), &body, &mut vm)
            .unwrap()
            .unwrap();

        let out_json: serde_json::Value = out.into();
        assert_eq!(out_json, serde_json::json!([1, 2]));
    }

    #[test]
    fn view_flat_map_count_stays_borrowed_without_materializing_rows() {
        let source = CountingNestedView::root(&[&[1, 2, 3], &[4, 5]]);
        let body = PipelineBody {
            stages: vec![Stage::FlatMap(
                Arc::new(crate::vm::Program::new(Vec::new(), "")),
                crate::builtins::BuiltinViewStage::FlatMap,
            )],
            stage_exprs: Vec::new(),
            sink: Sink::Reducer(crate::exec::pipeline::ReducerSpec::count()),
            stage_kernels: vec![BodyKernel::Current],
            sink_kernels: Vec::new(),
        };

        let out = super::run_full(source.clone(), &body).unwrap().unwrap();

        assert_eq!(out, Val::Int(5));
        assert_eq!(source.materialize_reads(), 0);
    }

    #[test]
    fn view_split_streams_owned_tokens_without_materializing_source_rows() {
        let source = CountingStringView::root(&["a:b", "c:d:e"]);
        let body = PipelineBody {
            stages: vec![
                Stage::StringBuiltin {
                    method: crate::builtins::BuiltinMethod::Split,
                    value: Arc::from(":"),
                },
                Stage::UsizeBuiltin {
                    method: crate::builtins::BuiltinMethod::Take,
                    value: 4,
                },
            ],
            stage_exprs: Vec::new(),
            sink: Sink::Reducer(crate::exec::pipeline::ReducerSpec::count()),
            stage_kernels: vec![BodyKernel::Generic, BodyKernel::Generic],
            sink_kernels: Vec::new(),
        };

        let out = super::run_full(source.clone(), &body).unwrap().unwrap();

        assert_eq!(out, Val::Int(4));
        assert_eq!(source.materialize_reads(), 0);
    }

    #[test]
    fn view_string_expanders_stream_tape_strings_without_materializing_receivers() {
        fn expanded_count(method: crate::builtins::BuiltinMethod) -> Val {
            let tape =
                crate::data::tape::TapeData::parse(br#"["a b","c\nd","xy"]"#.to_vec()).unwrap();
            let body = PipelineBody {
                stages: vec![Stage::Builtin(crate::builtins::BuiltinCall::new(
                    method,
                    crate::builtins::BuiltinArgs::None,
                ))],
                stage_exprs: Vec::new(),
                sink: Sink::Reducer(crate::exec::pipeline::ReducerSpec::count()),
                stage_kernels: vec![BodyKernel::Generic],
                sink_kernels: Vec::new(),
            };

            tape.reset_materialized_subtrees();
            let out = super::run_full(TapeView::root(&tape), &body)
                .unwrap()
                .unwrap();
            assert_eq!(tape.materialized_subtrees(), 0, "{method:?}");
            out
        }

        assert_eq!(
            expanded_count(crate::builtins::BuiltinMethod::Lines),
            Val::Int(4)
        );
        assert_eq!(
            expanded_count(crate::builtins::BuiltinMethod::Words),
            Val::Int(5)
        );
        assert_eq!(
            expanded_count(crate::builtins::BuiltinMethod::Chars),
            Val::Int(8)
        );
        assert_eq!(
            expanded_count(crate::builtins::BuiltinMethod::CharsOf),
            Val::Int(8)
        );
        assert_eq!(
            expanded_count(crate::builtins::BuiltinMethod::Bytes),
            Val::Int(8)
        );
    }

    #[test]
    fn view_map_streams_owned_rows_into_later_stages() {
        let source = CountingView::root(&[1, 2, 3, 4]);
        let body = PipelineBody {
            stages: vec![
                Stage::Map(
                    Arc::new(crate::vm::Program::new(Vec::new(), "")),
                    crate::builtins::BuiltinViewStage::Map,
                ),
                Stage::UsizeBuiltin {
                    method: crate::builtins::BuiltinMethod::Take,
                    value: 2,
                },
            ],
            stage_exprs: Vec::new(),
            sink: Sink::Reducer(crate::exec::pipeline::ReducerSpec::count()),
            stage_kernels: vec![
                BodyKernel::Const(Val::Str(Arc::from("owned"))),
                BodyKernel::Generic,
            ],
            sink_kernels: Vec::new(),
        };

        let out = super::run_full(source.clone(), &body).unwrap().unwrap();

        assert_eq!(out, Val::Int(2));
        assert_eq!(source.materialize_reads(), 0);
    }

    #[test]
    fn view_count_uses_source_length_through_cardinality_preserving_stages() {
        let source = CountingView::root(&[1, 2, 3, 4, 5]);
        let body = PipelineBody {
            stages: vec![
                Stage::Map(
                    Arc::new(crate::vm::Program::new(Vec::new(), "")),
                    crate::builtins::BuiltinViewStage::Map,
                ),
                Stage::UsizeBuiltin {
                    method: crate::builtins::BuiltinMethod::Skip,
                    value: 1,
                },
                Stage::UsizeBuiltin {
                    method: crate::builtins::BuiltinMethod::Take,
                    value: 3,
                },
            ],
            stage_exprs: Vec::new(),
            sink: Sink::Reducer(crate::exec::pipeline::ReducerSpec::count()),
            stage_kernels: vec![
                BodyKernel::Const(Val::Int(1)),
                BodyKernel::Generic,
                BodyKernel::Generic,
            ],
            sink_kernels: Vec::new(),
        };

        let out = super::run_full(source.clone(), &body).unwrap().unwrap();

        assert_eq!(out, Val::Int(3));
        assert_eq!(source.scalar_reads(), 1);
        assert_eq!(source.array_iter_reads(), 0);
        assert_eq!(source.materialize_reads(), 0);
    }

    #[test]
    fn view_builtin_projection_streams_owned_rows_into_later_stages() {
        let source = CountingObjectValuesView::root(&[&[1, 2, 3], &[4, 5]]);
        let body = PipelineBody {
            stages: vec![
                Stage::Builtin(crate::builtins::BuiltinCall::new(
                    crate::builtins::BuiltinMethod::Values,
                    crate::builtins::BuiltinArgs::None,
                )),
                Stage::FlatMap(
                    Arc::new(crate::vm::Program::new(Vec::new(), "")),
                    crate::builtins::BuiltinViewStage::FlatMap,
                ),
                Stage::UsizeBuiltin {
                    method: crate::builtins::BuiltinMethod::Take,
                    value: 4,
                },
            ],
            stage_exprs: Vec::new(),
            sink: Sink::Reducer(crate::exec::pipeline::ReducerSpec::count()),
            stage_kernels: vec![
                BodyKernel::Generic,
                BodyKernel::Current,
                BodyKernel::Generic,
            ],
            sink_kernels: Vec::new(),
        };

        let out = super::run_full(source.clone(), &body).unwrap().unwrap();

        assert_eq!(out, Val::Int(4));
        assert_eq!(source.object_value_reads(), 2);
        assert_eq!(source.materialize_reads(), 0);
    }

    #[test]
    fn view_scalar_builtin_call_runs_without_materializing_receiver() {
        let source = CountingObjectValuesView::root(&[&[1, 2], &[3][..], &[4, 5, 6][..]]);
        let body = PipelineBody {
            stages: vec![Stage::Map(
                Arc::new(crate::vm::Program::new(Vec::new(), "")),
                crate::builtins::BuiltinViewStage::Map,
            )],
            stage_exprs: Vec::new(),
            sink: Sink::Collect,
            stage_kernels: vec![BodyKernel::BuiltinCall {
                receiver: Box::new(BodyKernel::Current),
                call: crate::builtins::BuiltinCall::new(
                    crate::builtins::BuiltinMethod::Len,
                    crate::builtins::BuiltinArgs::None,
                ),
            }],
            sink_kernels: Vec::new(),
        };
        let env = Env::new(Val::Null);
        let mut vm = crate::vm::VM::new();

        let out = super::run_with_env_and_vm(source.clone(), &body, None, &env, &mut vm)
            .unwrap()
            .unwrap();

        assert_eq!(serde_json::Value::from(out), serde_json::json!([2, 1, 3]));
        assert_eq!(source.object_value_reads(), 0);
        assert_eq!(source.materialize_reads(), 0);
    }

    #[test]
    fn view_type_builtin_reads_tape_tags_without_materializing_receivers() {
        let tape =
            crate::data::tape::TapeData::parse(br#"[{"a":1},[1,2],"s",3,true,null]"#.to_vec())
                .unwrap();
        let body = PipelineBody {
            stages: vec![Stage::Map(
                Arc::new(crate::vm::Program::new(Vec::new(), "")),
                crate::builtins::BuiltinViewStage::Map,
            )],
            stage_exprs: Vec::new(),
            sink: Sink::Collect,
            stage_kernels: vec![BodyKernel::BuiltinCall {
                receiver: Box::new(BodyKernel::Current),
                call: crate::builtins::BuiltinCall::new(
                    crate::builtins::BuiltinMethod::Type,
                    crate::builtins::BuiltinArgs::None,
                ),
            }],
            sink_kernels: Vec::new(),
        };

        tape.reset_materialized_subtrees();
        let out = super::run_full(TapeView::root(&tape), &body)
            .unwrap()
            .unwrap();

        assert_eq!(
            serde_json::Value::from(out),
            serde_json::json!(["object", "array", "string", "number", "bool", "null"])
        );
        assert_eq!(tape.materialized_subtrees(), 0);
    }

    #[test]
    fn view_parse_builtins_read_tape_strings_without_materializing_receivers() {
        fn run(method: crate::builtins::BuiltinMethod) -> serde_json::Value {
            let tape =
                crate::data::tape::TapeData::parse(br#"["42","3.5","yes","false","bad"]"#.to_vec())
                    .unwrap();
            let body = PipelineBody {
                stages: vec![Stage::Map(
                    Arc::new(crate::vm::Program::new(Vec::new(), "")),
                    crate::builtins::BuiltinViewStage::Map,
                )],
                stage_exprs: Vec::new(),
                sink: Sink::Collect,
                stage_kernels: vec![BodyKernel::BuiltinCall {
                    receiver: Box::new(BodyKernel::Current),
                    call: crate::builtins::BuiltinCall::new(
                        method,
                        crate::builtins::BuiltinArgs::None,
                    ),
                }],
                sink_kernels: Vec::new(),
            };

            tape.reset_materialized_subtrees();
            let out = super::run_full(TapeView::root(&tape), &body)
                .unwrap()
                .unwrap();
            assert_eq!(tape.materialized_subtrees(), 0, "{method:?}");
            serde_json::Value::from(out)
        }

        assert_eq!(
            run(crate::builtins::BuiltinMethod::ParseInt),
            serde_json::json!([42, null, null, null, null])
        );
        assert_eq!(
            run(crate::builtins::BuiltinMethod::ParseFloat),
            serde_json::json!([42.0, 3.5, null, null, null])
        );
        assert_eq!(
            run(crate::builtins::BuiltinMethod::ParseBool),
            serde_json::json!([null, null, true, false, null])
        );
    }

    #[test]
    fn view_contains_vec_builtins_read_tape_strings_without_materializing_receivers() {
        fn run(method: crate::builtins::BuiltinMethod) -> serde_json::Value {
            let tape =
                crate::data::tape::TapeData::parse(br#"["alpha beta","alpha","gamma"]"#.to_vec())
                    .unwrap();
            let body = PipelineBody {
                stages: vec![Stage::Map(
                    Arc::new(crate::vm::Program::new(Vec::new(), "")),
                    crate::builtins::BuiltinViewStage::Map,
                )],
                stage_exprs: Vec::new(),
                sink: Sink::Collect,
                stage_kernels: vec![BodyKernel::BuiltinCall {
                    receiver: Box::new(BodyKernel::Current),
                    call: crate::builtins::BuiltinCall::new(
                        method,
                        crate::builtins::BuiltinArgs::StrVec(vec![
                            Arc::from("alpha"),
                            Arc::from("beta"),
                        ]),
                    ),
                }],
                sink_kernels: Vec::new(),
            };

            tape.reset_materialized_subtrees();
            let out = super::run_full(TapeView::root(&tape), &body)
                .unwrap()
                .unwrap();
            assert_eq!(tape.materialized_subtrees(), 0, "{method:?}");
            serde_json::Value::from(out)
        }

        assert_eq!(
            run(crate::builtins::BuiltinMethod::ContainsAny),
            serde_json::json!([true, true, false])
        );
        assert_eq!(
            run(crate::builtins::BuiltinMethod::ContainsAll),
            serde_json::json!([true, false, false])
        );
    }

    #[test]
    fn view_or_builtin_defaults_null_without_materializing_non_null_receivers() {
        let tape = crate::data::tape::TapeData::parse(br#"["a",null,"b"]"#.to_vec()).unwrap();
        let body = PipelineBody {
            stages: vec![
                Stage::Builtin(crate::builtins::BuiltinCall::new(
                    crate::builtins::BuiltinMethod::Or,
                    crate::builtins::BuiltinArgs::Val(Val::Str(Arc::from("fallback"))),
                )),
                Stage::Builtin(crate::builtins::BuiltinCall::new(
                    crate::builtins::BuiltinMethod::Type,
                    crate::builtins::BuiltinArgs::None,
                )),
            ],
            stage_exprs: Vec::new(),
            sink: Sink::Collect,
            stage_kernels: vec![BodyKernel::Generic, BodyKernel::Generic],
            sink_kernels: Vec::new(),
        };

        tape.reset_materialized_subtrees();
        let out = super::run_full(TapeView::root(&tape), &body)
            .unwrap()
            .unwrap();

        assert_eq!(
            serde_json::Value::from(out),
            serde_json::json!(["string", "string", "string"])
        );
        assert_eq!(tape.materialized_subtrees(), 0);
    }

    #[test]
    fn view_value_string_builtins_traverse_tape_without_materializing_receivers() {
        let tape =
            crate::data::tape::TapeData::parse(br#"[{"a":1,"b":[true,null]},"plain"]"#.to_vec())
                .unwrap();

        for (method, expected) in [
            (
                crate::builtins::BuiltinMethod::ToString,
                serde_json::json!(["{\"a\":1,\"b\":[true,null]}", "plain"]),
            ),
            (
                crate::builtins::BuiltinMethod::ToJson,
                serde_json::json!(["{\"a\":1,\"b\":[true,null]}", "\"plain\""]),
            ),
        ] {
            let body = PipelineBody {
                stages: vec![Stage::Map(
                    Arc::new(crate::vm::Program::new(Vec::new(), "")),
                    crate::builtins::BuiltinViewStage::Map,
                )],
                stage_exprs: Vec::new(),
                sink: Sink::Collect,
                stage_kernels: vec![BodyKernel::BuiltinCall {
                    receiver: Box::new(BodyKernel::Current),
                    call: crate::builtins::BuiltinCall::new(
                        method,
                        crate::builtins::BuiltinArgs::None,
                    ),
                }],
                sink_kernels: Vec::new(),
            };

            tape.reset_materialized_subtrees();
            let out = super::run_full(TapeView::root(&tape), &body)
                .unwrap()
                .unwrap();

            assert_eq!(serde_json::Value::from(out), expected);
            assert_eq!(tape.materialized_subtrees(), 0);
        }
    }

    #[test]
    fn view_slice_stage_slices_tape_strings_without_materializing_receivers() {
        let tape =
            crate::data::tape::TapeData::parse(br#"["abcdef","\u00e9clair"]"#.to_vec()).unwrap();
        let body = PipelineBody {
            stages: vec![Stage::IntRangeBuiltin {
                method: crate::builtins::BuiltinMethod::Slice,
                start: 1,
                end: Some(4),
            }],
            stage_exprs: Vec::new(),
            sink: Sink::Collect,
            stage_kernels: vec![BodyKernel::Generic],
            sink_kernels: Vec::new(),
        };

        tape.reset_materialized_subtrees();
        let out = super::run_full(TapeView::root(&tape), &body)
            .unwrap()
            .unwrap();

        assert_eq!(
            serde_json::Value::from(out),
            serde_json::json!(["bcd", "cla"])
        );
        assert_eq!(tape.materialized_subtrees(), 0);
    }

    #[test]
    fn view_slice_stage_passes_non_strings_as_borrowed_views() {
        let tape = crate::data::tape::TapeData::parse(br#"[{"k":"v"}]"#.to_vec()).unwrap();
        let body = PipelineBody {
            stages: vec![
                Stage::IntRangeBuiltin {
                    method: crate::builtins::BuiltinMethod::Slice,
                    start: 1,
                    end: Some(4),
                },
                Stage::Builtin(crate::builtins::BuiltinCall::new(
                    crate::builtins::BuiltinMethod::Type,
                    crate::builtins::BuiltinArgs::None,
                )),
            ],
            stage_exprs: Vec::new(),
            sink: Sink::Collect,
            stage_kernels: vec![BodyKernel::Generic, BodyKernel::Generic],
            sink_kernels: Vec::new(),
        };

        tape.reset_materialized_subtrees();
        let out = super::run_full(TapeView::root(&tape), &body)
            .unwrap()
            .unwrap();

        assert_eq!(serde_json::Value::from(out), serde_json::json!(["object"]));
        assert_eq!(tape.materialized_subtrees(), 0);
    }

    #[test]
    fn view_replace_stages_transform_tape_strings_without_materializing_receivers() {
        let tape = crate::data::tape::TapeData::parse(br#"["foo foo","bar"]"#.to_vec()).unwrap();

        for (method, expected) in [
            (
                crate::builtins::BuiltinMethod::Replace,
                serde_json::json!(["xoo foo", "bar"]),
            ),
            (
                crate::builtins::BuiltinMethod::ReplaceAll,
                serde_json::json!(["xoo xoo", "bar"]),
            ),
        ] {
            let body = PipelineBody {
                stages: vec![Stage::StringPairBuiltin {
                    method,
                    first: Arc::from("f"),
                    second: Arc::from("x"),
                }],
                stage_exprs: Vec::new(),
                sink: Sink::Collect,
                stage_kernels: vec![BodyKernel::Generic],
                sink_kernels: Vec::new(),
            };

            tape.reset_materialized_subtrees();
            let out = super::run_full(TapeView::root(&tape), &body)
                .unwrap()
                .unwrap();

            assert_eq!(serde_json::Value::from(out), expected);
            assert_eq!(tape.materialized_subtrees(), 0);
        }
    }

    #[test]
    fn view_replace_stage_passes_non_strings_as_borrowed_views() {
        let tape = crate::data::tape::TapeData::parse(br#"[{"k":"foo"}]"#.to_vec()).unwrap();
        let body = PipelineBody {
            stages: vec![
                Stage::StringPairBuiltin {
                    method: crate::builtins::BuiltinMethod::Replace,
                    first: Arc::from("f"),
                    second: Arc::from("x"),
                },
                Stage::Builtin(crate::builtins::BuiltinCall::new(
                    crate::builtins::BuiltinMethod::Type,
                    crate::builtins::BuiltinArgs::None,
                )),
            ],
            stage_exprs: Vec::new(),
            sink: Sink::Collect,
            stage_kernels: vec![BodyKernel::Generic, BodyKernel::Generic],
            sink_kernels: Vec::new(),
        };

        tape.reset_materialized_subtrees();
        let out = super::run_full(TapeView::root(&tape), &body)
            .unwrap()
            .unwrap();

        assert_eq!(serde_json::Value::from(out), serde_json::json!(["object"]));
        assert_eq!(tape.materialized_subtrees(), 0);
    }

    #[test]
    fn view_strip_affix_builtins_transform_tape_strings_without_materializing_receivers() {
        let tape =
            crate::data::tape::TapeData::parse(br#"["pre_name","name_suf","plain"]"#.to_vec())
                .unwrap();

        for (method, arg, expected) in [
            (
                crate::builtins::BuiltinMethod::StripPrefix,
                Arc::from("pre_"),
                serde_json::json!(["name", "name_suf", "plain"]),
            ),
            (
                crate::builtins::BuiltinMethod::StripSuffix,
                Arc::from("_suf"),
                serde_json::json!(["pre_name", "name", "plain"]),
            ),
        ] {
            let body = PipelineBody {
                stages: vec![Stage::Builtin(crate::builtins::BuiltinCall::new(
                    method,
                    crate::builtins::BuiltinArgs::Str(arg),
                ))],
                stage_exprs: Vec::new(),
                sink: Sink::Collect,
                stage_kernels: vec![BodyKernel::Generic],
                sink_kernels: Vec::new(),
            };

            tape.reset_materialized_subtrees();
            let out = super::run_full(TapeView::root(&tape), &body)
                .unwrap()
                .unwrap();

            assert_eq!(serde_json::Value::from(out), expected);
            assert_eq!(tape.materialized_subtrees(), 0);
        }
    }

    #[test]
    fn view_strip_affix_builtins_pass_non_strings_as_borrowed_views() {
        let tape = crate::data::tape::TapeData::parse(br#"[{"k":"pre_name"}]"#.to_vec()).unwrap();
        let body = PipelineBody {
            stages: vec![
                Stage::Builtin(crate::builtins::BuiltinCall::new(
                    crate::builtins::BuiltinMethod::StripPrefix,
                    crate::builtins::BuiltinArgs::Str(Arc::from("pre_")),
                )),
                Stage::Builtin(crate::builtins::BuiltinCall::new(
                    crate::builtins::BuiltinMethod::Type,
                    crate::builtins::BuiltinArgs::None,
                )),
            ],
            stage_exprs: Vec::new(),
            sink: Sink::Collect,
            stage_kernels: vec![BodyKernel::Generic, BodyKernel::Generic],
            sink_kernels: Vec::new(),
        };

        tape.reset_materialized_subtrees();
        let out = super::run_full(TapeView::root(&tape), &body)
            .unwrap()
            .unwrap();

        assert_eq!(serde_json::Value::from(out), serde_json::json!(["object"]));
        assert_eq!(tape.materialized_subtrees(), 0);
    }

    #[test]
    fn view_repeat_builtin_transforms_tape_strings_without_materializing_receivers() {
        let tape = crate::data::tape::TapeData::parse(br#"["ab",""]"#.to_vec()).unwrap();
        let body = PipelineBody {
            stages: vec![Stage::Builtin(crate::builtins::BuiltinCall::new(
                crate::builtins::BuiltinMethod::Repeat,
                crate::builtins::BuiltinArgs::Usize(3),
            ))],
            stage_exprs: Vec::new(),
            sink: Sink::Collect,
            stage_kernels: vec![BodyKernel::Generic],
            sink_kernels: Vec::new(),
        };

        tape.reset_materialized_subtrees();
        let out = super::run_full(TapeView::root(&tape), &body)
            .unwrap()
            .unwrap();

        assert_eq!(
            serde_json::Value::from(out),
            serde_json::json!(["ababab", ""])
        );
        assert_eq!(tape.materialized_subtrees(), 0);
    }

    #[test]
    fn view_repeat_builtin_passes_non_strings_as_borrowed_views() {
        let tape = crate::data::tape::TapeData::parse(br#"[{"k":"ab"}]"#.to_vec()).unwrap();
        let body = PipelineBody {
            stages: vec![
                Stage::Builtin(crate::builtins::BuiltinCall::new(
                    crate::builtins::BuiltinMethod::Repeat,
                    crate::builtins::BuiltinArgs::Usize(2),
                )),
                Stage::Builtin(crate::builtins::BuiltinCall::new(
                    crate::builtins::BuiltinMethod::Type,
                    crate::builtins::BuiltinArgs::None,
                )),
            ],
            stage_exprs: Vec::new(),
            sink: Sink::Collect,
            stage_kernels: vec![BodyKernel::Generic, BodyKernel::Generic],
            sink_kernels: Vec::new(),
        };

        tape.reset_materialized_subtrees();
        let out = super::run_full(TapeView::root(&tape), &body)
            .unwrap()
            .unwrap();

        assert_eq!(serde_json::Value::from(out), serde_json::json!(["object"]));
        assert_eq!(tape.materialized_subtrees(), 0);
    }

    #[test]
    fn view_pad_builtins_transform_tape_strings_without_materializing_receivers() {
        let tape = crate::data::tape::TapeData::parse(br#"["ab","wide"]"#.to_vec()).unwrap();

        for (method, expected) in [
            (
                crate::builtins::BuiltinMethod::PadLeft,
                serde_json::json!(["__ab", "wide"]),
            ),
            (
                crate::builtins::BuiltinMethod::PadRight,
                serde_json::json!(["ab__", "wide"]),
            ),
            (
                crate::builtins::BuiltinMethod::Center,
                serde_json::json!(["_ab_", "wide"]),
            ),
        ] {
            let body = PipelineBody {
                stages: vec![Stage::Builtin(crate::builtins::BuiltinCall::new(
                    method,
                    crate::builtins::BuiltinArgs::Pad {
                        width: 4,
                        fill: '_',
                    },
                ))],
                stage_exprs: Vec::new(),
                sink: Sink::Collect,
                stage_kernels: vec![BodyKernel::Generic],
                sink_kernels: Vec::new(),
            };

            tape.reset_materialized_subtrees();
            let out = super::run_full(TapeView::root(&tape), &body)
                .unwrap()
                .unwrap();

            assert_eq!(serde_json::Value::from(out), expected);
            assert_eq!(tape.materialized_subtrees(), 0);
        }
    }

    #[test]
    fn view_pad_builtin_passes_non_strings_as_borrowed_views() {
        let tape = crate::data::tape::TapeData::parse(br#"[{"k":"ab"}]"#.to_vec()).unwrap();
        let body = PipelineBody {
            stages: vec![
                Stage::Builtin(crate::builtins::BuiltinCall::new(
                    crate::builtins::BuiltinMethod::PadLeft,
                    crate::builtins::BuiltinArgs::Pad {
                        width: 4,
                        fill: '_',
                    },
                )),
                Stage::Builtin(crate::builtins::BuiltinCall::new(
                    crate::builtins::BuiltinMethod::Type,
                    crate::builtins::BuiltinArgs::None,
                )),
            ],
            stage_exprs: Vec::new(),
            sink: Sink::Collect,
            stage_kernels: vec![BodyKernel::Generic, BodyKernel::Generic],
            sink_kernels: Vec::new(),
        };

        tape.reset_materialized_subtrees();
        let out = super::run_full(TapeView::root(&tape), &body)
            .unwrap()
            .unwrap();

        assert_eq!(serde_json::Value::from(out), serde_json::json!(["object"]));
        assert_eq!(tape.materialized_subtrees(), 0);
    }

    #[test]
    fn view_reverse_str_builtin_transforms_tape_strings_without_materializing_receivers() {
        let tape = crate::data::tape::TapeData::parse(br#"["abc","\u00e9x"]"#.to_vec()).unwrap();
        let body = PipelineBody {
            stages: vec![Stage::Builtin(crate::builtins::BuiltinCall::new(
                crate::builtins::BuiltinMethod::ReverseStr,
                crate::builtins::BuiltinArgs::None,
            ))],
            stage_exprs: Vec::new(),
            sink: Sink::Collect,
            stage_kernels: vec![BodyKernel::Generic],
            sink_kernels: Vec::new(),
        };

        tape.reset_materialized_subtrees();
        let out = super::run_full(TapeView::root(&tape), &body)
            .unwrap()
            .unwrap();

        assert_eq!(
            serde_json::Value::from(out),
            serde_json::json!(["cba", "xé"])
        );
        assert_eq!(tape.materialized_subtrees(), 0);
    }

    #[test]
    fn view_reverse_str_builtin_passes_non_strings_as_borrowed_views() {
        let tape = crate::data::tape::TapeData::parse(br#"[{"k":"abc"}]"#.to_vec()).unwrap();
        let body = PipelineBody {
            stages: vec![
                Stage::Builtin(crate::builtins::BuiltinCall::new(
                    crate::builtins::BuiltinMethod::ReverseStr,
                    crate::builtins::BuiltinArgs::None,
                )),
                Stage::Builtin(crate::builtins::BuiltinCall::new(
                    crate::builtins::BuiltinMethod::Type,
                    crate::builtins::BuiltinArgs::None,
                )),
            ],
            stage_exprs: Vec::new(),
            sink: Sink::Collect,
            stage_kernels: vec![BodyKernel::Generic, BodyKernel::Generic],
            sink_kernels: Vec::new(),
        };

        tape.reset_materialized_subtrees();
        let out = super::run_full(TapeView::root(&tape), &body)
            .unwrap()
            .unwrap();

        assert_eq!(serde_json::Value::from(out), serde_json::json!(["object"]));
        assert_eq!(tape.materialized_subtrees(), 0);
    }

    #[test]
    fn view_string_codec_builtins_transform_tape_strings_without_materializing_receivers() {
        fn run(method: crate::builtins::BuiltinMethod, input: &[u8]) -> serde_json::Value {
            let tape = crate::data::tape::TapeData::parse(input.to_vec()).unwrap();
            let body = PipelineBody {
                stages: vec![Stage::Builtin(crate::builtins::BuiltinCall::new(
                    method,
                    crate::builtins::BuiltinArgs::None,
                ))],
                stage_exprs: Vec::new(),
                sink: Sink::Collect,
                stage_kernels: vec![BodyKernel::Generic],
                sink_kernels: Vec::new(),
            };

            tape.reset_materialized_subtrees();
            let out = super::run_full(TapeView::root(&tape), &body)
                .unwrap()
                .unwrap();
            assert_eq!(tape.materialized_subtrees(), 0, "{method:?}");
            serde_json::Value::from(out)
        }

        assert_eq!(
            run(crate::builtins::BuiltinMethod::ToBase64, br#"["a b<&"]"#),
            serde_json::json!(["YSBiPCY="])
        );
        assert_eq!(
            run(
                crate::builtins::BuiltinMethod::FromBase64,
                br#"["aMOp","bad!"]"#
            ),
            serde_json::json!(["hé", null])
        );
        assert_eq!(
            run(crate::builtins::BuiltinMethod::UrlEncode, br#"["a b<&"]"#),
            serde_json::json!(["a%20b%3C%26"])
        );
        assert_eq!(
            run(
                crate::builtins::BuiltinMethod::UrlDecode,
                br#"["a%20b%2F"]"#
            ),
            serde_json::json!(["a b/"])
        );
        assert_eq!(
            run(crate::builtins::BuiltinMethod::HtmlEscape, br#"["a b<&"]"#),
            serde_json::json!(["a b&lt;&amp;"])
        );
        assert_eq!(
            run(
                crate::builtins::BuiltinMethod::HtmlUnescape,
                br#"["a &lt; b &amp; c"]"#
            ),
            serde_json::json!(["a < b & c"])
        );
    }

    #[test]
    fn view_case_builtins_transform_tape_strings_without_materializing_receivers() {
        fn run(method: crate::builtins::BuiltinMethod, input: &[u8]) -> serde_json::Value {
            let tape = crate::data::tape::TapeData::parse(input.to_vec()).unwrap();
            let body = PipelineBody {
                stages: vec![Stage::Builtin(crate::builtins::BuiltinCall::new(
                    method,
                    crate::builtins::BuiltinArgs::None,
                ))],
                stage_exprs: Vec::new(),
                sink: Sink::Collect,
                stage_kernels: vec![BodyKernel::Generic],
                sink_kernels: Vec::new(),
            };

            tape.reset_materialized_subtrees();
            let out = super::run_full(TapeView::root(&tape), &body)
                .unwrap()
                .unwrap();
            assert_eq!(tape.materialized_subtrees(), 0, "{method:?}");
            serde_json::Value::from(out)
        }

        assert_eq!(
            run(
                crate::builtins::BuiltinMethod::Capitalize,
                br#"["hello WORLD"]"#
            ),
            serde_json::json!(["Hello world"])
        );
        assert_eq!(
            run(
                crate::builtins::BuiltinMethod::TitleCase,
                br#"["hello WORLD"]"#
            ),
            serde_json::json!(["Hello World"])
        );
        assert_eq!(
            run(
                crate::builtins::BuiltinMethod::SnakeCase,
                br#"["Hello world_test"]"#
            ),
            serde_json::json!(["hello_world_test"])
        );
        assert_eq!(
            run(
                crate::builtins::BuiltinMethod::KebabCase,
                br#"["Hello world_test"]"#
            ),
            serde_json::json!(["hello-world-test"])
        );
        assert_eq!(
            run(
                crate::builtins::BuiltinMethod::CamelCase,
                br#"["Hello world_test"]"#
            ),
            serde_json::json!(["helloWorldTest"])
        );
        assert_eq!(
            run(
                crate::builtins::BuiltinMethod::PascalCase,
                br#"["Hello world_test"]"#
            ),
            serde_json::json!(["HelloWorldTest"])
        );
        assert_eq!(
            run(crate::builtins::BuiltinMethod::Dedent, br#"["  a\n    b"]"#),
            serde_json::json!(["a\n  b"])
        );
    }

    #[test]
    fn view_indent_builtin_transforms_tape_strings_without_materializing_receivers() {
        fn run(args: crate::builtins::BuiltinArgs) -> serde_json::Value {
            let tape = crate::data::tape::TapeData::parse(br#"["a\nb",""]"#.to_vec()).unwrap();
            let body = PipelineBody {
                stages: vec![Stage::Builtin(crate::builtins::BuiltinCall::new(
                    crate::builtins::BuiltinMethod::Indent,
                    args,
                ))],
                stage_exprs: Vec::new(),
                sink: Sink::Collect,
                stage_kernels: vec![BodyKernel::Generic],
                sink_kernels: Vec::new(),
            };

            tape.reset_materialized_subtrees();
            let out = super::run_full(TapeView::root(&tape), &body)
                .unwrap()
                .unwrap();
            assert_eq!(tape.materialized_subtrees(), 0);
            serde_json::Value::from(out)
        }

        assert_eq!(
            run(crate::builtins::BuiltinArgs::Usize(2)),
            serde_json::json!(["  a\n  b", ""])
        );
        assert_eq!(
            run(crate::builtins::BuiltinArgs::Str(Arc::from("> "))),
            serde_json::json!(["> a\n> b", ""])
        );
    }

    #[test]
    fn view_flat_map_accepts_owned_array_projection_without_materializing_rows() {
        let source = CountingObjectValuesView::root(&[&[1, 2, 3], &[4, 5]]);
        let body = PipelineBody {
            stages: vec![
                Stage::FlatMap(
                    Arc::new(crate::vm::Program::new(Vec::new(), "")),
                    crate::builtins::BuiltinViewStage::FlatMap,
                ),
                Stage::UsizeBuiltin {
                    method: crate::builtins::BuiltinMethod::Take,
                    value: 4,
                },
            ],
            stage_exprs: Vec::new(),
            sink: Sink::Reducer(crate::exec::pipeline::ReducerSpec::count()),
            stage_kernels: vec![
                BodyKernel::BuiltinCall {
                    receiver: Box::new(BodyKernel::Current),
                    call: crate::builtins::BuiltinCall::new(
                        crate::builtins::BuiltinMethod::Values,
                        crate::builtins::BuiltinArgs::None,
                    ),
                },
                BodyKernel::Generic,
            ],
            sink_kernels: Vec::new(),
        };

        let out = super::run_full(source.clone(), &body).unwrap().unwrap();

        assert_eq!(out, Val::Int(4));
        assert_eq!(source.object_value_reads(), 2);
        assert_eq!(source.materialize_reads(), 0);
    }

    #[test]
    fn nested_receiver_plan_runs_on_tape_view_without_materializing_rows() {
        let tape = crate::data::tape::TapeData::parse(br#"[[1,2,3],[4,5]]"#.to_vec()).unwrap();
        let nested = Plan {
            source: Source::Receiver(Val::Null),
            stages: Vec::new(),
            stage_exprs: Vec::new(),
            sink: Sink::Nth(0),
            stage_kernels: Vec::new(),
            sink_kernels: Vec::new(),
        };
        let body = PipelineBody {
            stages: vec![Stage::Map(
                Arc::new(crate::vm::Program::new(Vec::new(), "")),
                crate::builtins::BuiltinViewStage::Map,
            )],
            stage_exprs: Vec::new(),
            sink: Sink::Collect,
            stage_kernels: vec![BodyKernel::NestedPlan(Arc::new(NestedPlanKernel::new(
                Arc::new(nested),
            )))],
            sink_kernels: Vec::new(),
        };

        tape.reset_materialized_subtrees();
        let out = super::run_full(TapeView::root(&tape), &body)
            .unwrap()
            .unwrap();

        assert_eq!(serde_json::Value::from(out), serde_json::json!([1, 4]));
        assert_eq!(tape.materialized_subtrees(), 0);
    }

    #[test]
    fn view_distinct_stage_feeds_count_sink_without_materializing_rows() {
        let source = CountingView::root(&[7, 8, 7, 9, 8, 7]);
        let body = PipelineBody {
            stages: vec![Stage::UniqueBy(None)],
            stage_exprs: Vec::new(),
            sink: Sink::Reducer(crate::exec::pipeline::ReducerSpec::count()),
            stage_kernels: vec![BodyKernel::Generic],
            sink_kernels: Vec::new(),
        };

        let out = super::run_full(source.clone(), &body).unwrap().unwrap();

        assert_eq!(out, Val::Int(3));
        assert_eq!(source.scalar_reads(), 6);
        assert_eq!(source.materialize_reads(), 0);
    }

    #[test]
    fn view_distinct_stage_keys_compound_tape_rows_without_materializing() {
        let tape = crate::data::tape::TapeData::parse(
            br#"[{"id":1,"tags":["a"]},{"id":1,"tags":["a"]},{"id":2,"tags":["b"]},{"id":1,"tags":["a"]}]"#
                .to_vec(),
        )
        .unwrap();
        let source = TapeView::root(&tape);
        tape.reset_materialized_subtrees();
        let body = PipelineBody {
            stages: vec![Stage::UniqueBy(None)],
            stage_exprs: Vec::new(),
            sink: Sink::Reducer(crate::exec::pipeline::ReducerSpec::count()),
            stage_kernels: vec![BodyKernel::Generic],
            sink_kernels: Vec::new(),
        };

        let out = super::run_full(source, &body).unwrap().unwrap();

        assert_eq!(out, Val::Int(2));
        assert_eq!(tape.materialized_subtrees(), 0);
    }

    #[test]
    fn reducing_count_by_stage_materializes_only_final_boundary_value() {
        let source = CountingView::root(&[1, 2, 1, 3, 2, 1]);
        let body = PipelineBody {
            stages: vec![Stage::ExprBuiltin {
                method: crate::builtins::BuiltinMethod::CountBy,
                body: Arc::new(crate::vm::Program::new(Vec::new(), "")),
            }],
            stage_exprs: Vec::new(),
            sink: Sink::Terminal(crate::builtins::BuiltinMethod::First),
            stage_kernels: vec![BodyKernel::Current],
            sink_kernels: Vec::new(),
        };

        let env = Env::new(Val::Null);
        let mut vm = crate::vm::VM::new();
        let out = super::run_reducing_stage_prefix_then_materialized_suffix(
            source.clone(),
            &body,
            None,
            &env,
            &mut vm,
        )
        .unwrap()
        .unwrap();

        let out_json: serde_json::Value = out.into();
        assert_eq!(out_json, serde_json::json!({"1": 3, "2": 2, "3": 1}));
        assert_eq!(source.scalar_reads(), 6);
        assert_eq!(source.materialize_reads(), 0);
    }

    #[test]
    fn reducing_keyed_stage_skips_empty_prefix_without_iterating_rows() {
        for method in [
            crate::builtins::BuiltinMethod::CountBy,
            crate::builtins::BuiltinMethod::GroupBy,
            crate::builtins::BuiltinMethod::IndexBy,
        ] {
            let source = CountingView::root(&[1, 2, 1, 3]);
            let body = PipelineBody {
                stages: vec![
                    Stage::UsizeBuiltin {
                        method: crate::builtins::BuiltinMethod::Take,
                        value: 0,
                    },
                    Stage::ExprBuiltin {
                        method,
                        body: Arc::new(crate::vm::Program::new(Vec::new(), "")),
                    },
                ],
                stage_exprs: Vec::new(),
                sink: Sink::Terminal(crate::builtins::BuiltinMethod::First),
                stage_kernels: vec![BodyKernel::Generic, BodyKernel::Current],
                sink_kernels: Vec::new(),
            };

            let env = Env::new(Val::Null);
            let mut vm = crate::vm::VM::new();
            let out = super::run_reducing_stage_prefix_then_materialized_suffix(
                source.clone(),
                &body,
                None,
                &env,
                &mut vm,
            )
            .unwrap()
            .unwrap();

            assert_eq!(serde_json::Value::from(out), serde_json::json!({}));
            assert_eq!(source.scalar_reads(), 0, "{method:?}");
            assert_eq!(source.array_iter_reads(), 0, "{method:?}");
            assert_eq!(source.materialize_reads(), 0, "{method:?}");
        }
    }

    #[test]
    fn sort_stage_skips_empty_prefix_without_iterating_rows() {
        let source = CountingView::root(&[3, 1, 2]);
        let body = PipelineBody {
            stages: vec![
                Stage::UsizeBuiltin {
                    method: crate::builtins::BuiltinMethod::Take,
                    value: 0,
                },
                Stage::Sort(crate::exec::pipeline::SortSpec::identity()),
            ],
            stage_exprs: Vec::new(),
            sink: Sink::Collect,
            stage_kernels: vec![BodyKernel::Generic, BodyKernel::Generic],
            sink_kernels: Vec::new(),
        };

        let env = Env::new(Val::Null);
        let mut vm = crate::vm::VM::new();
        let out = super::run_sort_prefix_then_materialized_suffix(
            source.clone(),
            &body,
            None,
            &env,
            &mut vm,
        )
        .unwrap()
        .unwrap();

        assert_eq!(serde_json::Value::from(out), serde_json::json!([]));
        assert_eq!(source.scalar_reads(), 0);
        assert_eq!(source.array_iter_reads(), 0);
        assert_eq!(source.materialize_reads(), 0);
    }

    #[test]
    fn natural_sort_uses_scalar_view_keys_without_materializing_rows() {
        let source = CountingView::root(&[3, 1, 2]);
        let body = PipelineBody {
            stages: vec![Stage::Sort(crate::exec::pipeline::SortSpec::identity())],
            stage_exprs: Vec::new(),
            sink: Sink::Reducer(crate::exec::pipeline::ReducerSpec::count()),
            stage_kernels: vec![BodyKernel::Generic],
            sink_kernels: Vec::new(),
        };

        let env = Env::new(Val::Null);
        let mut vm = crate::vm::VM::new();
        let out = super::run_sort_prefix_then_materialized_suffix(
            source.clone(),
            &body,
            None,
            &env,
            &mut vm,
        )
        .unwrap()
        .unwrap();

        assert_eq!(out, Val::Int(3));
        assert_eq!(source.array_iter_reads(), 1);
        assert_eq!(source.materialize_reads(), 0);
    }

    #[test]
    fn ordered_sort_view_suffix_skips_empty_prefix_without_iterating_rows() {
        let source = CountingView::root(&[3, 1, 2]);
        let body = PipelineBody {
            stages: vec![
                Stage::UsizeBuiltin {
                    method: crate::builtins::BuiltinMethod::Take,
                    value: 0,
                },
                Stage::Sort(crate::exec::pipeline::SortSpec::identity()),
            ],
            stage_exprs: Vec::new(),
            sink: Sink::Collect,
            stage_kernels: vec![BodyKernel::Generic, BodyKernel::Generic],
            sink_kernels: Vec::new(),
        };
        let plan = super::SortBarrierPlan {
            prefix: vec![ViewStageCapability::Take(0)],
            sort_stage: 1,
            key_program: None,
            descending: false,
        };

        let env = Env::new(Val::Null);
        let mut vm = crate::vm::VM::new();
        let out =
            super::run_sort_prefix_then_view_suffix(source.clone(), &body, &plan, &env, &mut vm)
                .unwrap()
                .unwrap();

        assert_eq!(serde_json::Value::from(out), serde_json::json!([]));
        assert_eq!(source.scalar_reads(), 0);
        assert_eq!(source.array_iter_reads(), 0);
        assert_eq!(source.materialize_reads(), 0);
    }

    #[test]
    fn materialized_suffix_skips_empty_prefix_without_iterating_rows() {
        let source = CountingView::root(&[3, 1, 2, 1]);
        let body = PipelineBody {
            stages: vec![
                Stage::UsizeBuiltin {
                    method: crate::builtins::BuiltinMethod::Take,
                    value: 0,
                },
                Stage::SortedDedup(None),
            ],
            stage_exprs: Vec::new(),
            sink: Sink::Collect,
            stage_kernels: vec![BodyKernel::Generic, BodyKernel::Generic],
            sink_kernels: Vec::new(),
        };

        let env = Env::new(Val::Null);
        let mut vm = crate::vm::VM::new();
        let out =
            super::run_prefix_then_materialized_suffix(source.clone(), &body, None, &env, &mut vm)
                .unwrap()
                .unwrap();

        assert_eq!(serde_json::Value::from(out), serde_json::json!([]));
        assert_eq!(source.scalar_reads(), 0);
        assert_eq!(source.array_iter_reads(), 0);
        assert_eq!(source.materialize_reads(), 0);
    }

    #[test]
    fn reducing_index_by_stage_uses_shared_keyed_reducer_path() {
        let source = CountingView::root(&[1, 2, 1, 3]);
        let body = PipelineBody {
            stages: vec![Stage::ExprBuiltin {
                method: crate::builtins::BuiltinMethod::IndexBy,
                body: Arc::new(crate::vm::Program::new(Vec::new(), "")),
            }],
            stage_exprs: Vec::new(),
            sink: Sink::Terminal(crate::builtins::BuiltinMethod::First),
            stage_kernels: vec![BodyKernel::Current],
            sink_kernels: Vec::new(),
        };

        let env = Env::new(Val::Null);
        let mut vm = crate::vm::VM::new();
        let out = super::run_reducing_stage_prefix_then_materialized_suffix(
            source.clone(),
            &body,
            None,
            &env,
            &mut vm,
        )
        .unwrap()
        .unwrap();

        let out_json: serde_json::Value = out.into();
        assert_eq!(out_json, serde_json::json!({"1": 1, "2": 2, "3": 3}));
        assert_eq!(source.scalar_reads(), 7);
        assert_eq!(source.materialize_reads(), 0);
    }

    #[test]
    fn reducing_index_by_keeps_replacements_borrowed_until_finish() {
        let tape = crate::data::tape::TapeData::parse(
            br#"[{"id":1,"score":1},{"id":1,"score":2},{"id":1,"score":3}]"#.to_vec(),
        )
        .unwrap();
        tape.reset_materialized_subtrees();
        let body = PipelineBody {
            stages: vec![Stage::ExprBuiltin {
                method: crate::builtins::BuiltinMethod::IndexBy,
                body: Arc::new(crate::vm::Program::new(Vec::new(), "")),
            }],
            stage_exprs: Vec::new(),
            sink: Sink::Terminal(crate::builtins::BuiltinMethod::First),
            stage_kernels: vec![BodyKernel::FieldRead(Arc::from("id"))],
            sink_kernels: Vec::new(),
        };

        let env = Env::new(Val::Null);
        let mut vm = crate::vm::VM::new();
        let out = super::run_reducing_stage_prefix_then_materialized_suffix(
            TapeView::root(&tape),
            &body,
            None,
            &env,
            &mut vm,
        )
        .unwrap()
        .unwrap();

        assert_eq!(
            serde_json::Value::from(out),
            serde_json::json!({"1": {"id": 1, "score": 3}})
        );
        assert_eq!(tape.materialized_subtrees(), 1);
    }

    #[test]
    fn reducing_group_by_keys_avoids_group_value_materialization() {
        let source = CountingView::root(&[1, 2, 1, 3]);
        let body = PipelineBody {
            stages: vec![
                Stage::ExprBuiltin {
                    method: crate::builtins::BuiltinMethod::GroupBy,
                    body: Arc::new(crate::vm::Program::new(Vec::new(), "")),
                },
                Stage::Builtin(crate::builtins::BuiltinCall::new(
                    crate::builtins::BuiltinMethod::Keys,
                    crate::builtins::BuiltinArgs::None,
                )),
            ],
            stage_exprs: Vec::new(),
            sink: Sink::Collect,
            stage_kernels: vec![BodyKernel::Current, BodyKernel::Generic],
            sink_kernels: Vec::new(),
        };

        let env = Env::new(Val::Null);
        let mut vm = crate::vm::VM::new();
        let out = super::run_reducing_stage_prefix_then_materialized_suffix(
            source.clone(),
            &body,
            None,
            &env,
            &mut vm,
        )
        .unwrap()
        .unwrap();

        let out_json: serde_json::Value = out.into();
        assert_eq!(out_json, serde_json::json!(["1", "2", "3"]));
        assert_eq!(source.scalar_reads(), 4);
        assert_eq!(source.materialize_reads(), 0);
    }

    #[test]
    fn reducing_group_by_key_predicate_avoids_group_value_materialization() {
        let source = CountingView::root(&[1, 2, 1, 3]);
        let body = PipelineBody {
            stages: vec![
                Stage::ExprBuiltin {
                    method: crate::builtins::BuiltinMethod::GroupBy,
                    body: Arc::new(crate::vm::Program::new(Vec::new(), "")),
                },
                Stage::Builtin(crate::builtins::BuiltinCall::new(
                    crate::builtins::BuiltinMethod::HasKey,
                    crate::builtins::BuiltinArgs::Str(Arc::from("2")),
                )),
            ],
            stage_exprs: Vec::new(),
            sink: Sink::Collect,
            stage_kernels: vec![BodyKernel::Current, BodyKernel::Generic],
            sink_kernels: Vec::new(),
        };

        let env = Env::new(Val::Null);
        let mut vm = crate::vm::VM::new();
        let out = super::run_reducing_stage_prefix_then_materialized_suffix(
            source.clone(),
            &body,
            None,
            &env,
            &mut vm,
        )
        .unwrap()
        .unwrap();

        assert_eq!(out, Val::Bool(true));
        assert_eq!(source.scalar_reads(), 4);
        assert_eq!(source.materialize_reads(), 0);
    }

    #[test]
    fn reducing_group_by_len_avoids_group_value_materialization() {
        let source = CountingView::root(&[1, 2, 1, 3]);
        let body = PipelineBody {
            stages: vec![
                Stage::ExprBuiltin {
                    method: crate::builtins::BuiltinMethod::GroupBy,
                    body: Arc::new(crate::vm::Program::new(Vec::new(), "")),
                },
                Stage::Builtin(crate::builtins::BuiltinCall::new(
                    crate::builtins::BuiltinMethod::Len,
                    crate::builtins::BuiltinArgs::None,
                )),
            ],
            stage_exprs: Vec::new(),
            sink: Sink::Collect,
            stage_kernels: vec![BodyKernel::Current, BodyKernel::Generic],
            sink_kernels: Vec::new(),
        };

        let env = Env::new(Val::Null);
        let mut vm = crate::vm::VM::new();
        let out = super::run_reducing_stage_prefix_then_materialized_suffix(
            source.clone(),
            &body,
            None,
            &env,
            &mut vm,
        )
        .unwrap()
        .unwrap();

        assert_eq!(out, Val::Int(3));
        assert_eq!(source.scalar_reads(), 4);
        assert_eq!(source.materialize_reads(), 0);
    }
}
