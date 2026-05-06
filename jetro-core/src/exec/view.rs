//! View-based pipeline execution over borrowed document representations.
//!
//! Runs pipeline plans against a `ValueView` implementation rather than a
//! materialised `Val` tree. Stages that can stay in the borrowed domain do so;
//! only the final collect step, or a stage that requires a `Val` (e.g. a
//! method call), calls `materialize()`. Used by `physical_eval` when the
//! planner selects the `View` backend preference.

use std::sync::Arc;

use crate::parse::chain_ir::PullDemand;
use crate::context::{Env, EvalError};
use crate::exec::pipeline;
use crate::data::value::Val;
use crate::data::view::{scalar_view_to_owned_val, ValueView};

mod key;
mod reducer_stage;
mod stage_flow;

use key::ViewKey;
use stage_flow::{ViewStageFlow, ViewStageState};

/// Navigates a field-key sequence on `cur`, calling `ValueView::field` for each
/// key and returning the deepest resolved view. If a step returns a null-like
/// view, traversal continues with that null view.
pub(crate) fn walk_fields<'a, V>(mut cur: V, keys: &[Arc<str>]) -> V
where
    V: ValueView<'a>,
{
    for key in keys {
        cur = cur.field(key.as_ref());
    }
    cur
}

/// Top-level view-pipeline runner. Tries fast-path sub-runners in priority order:
/// `terminal_collect`, `full`, `reducing_stage_prefix`, `sort_prefix`, then a
/// generic `prefix_then_materialized_suffix` fallback. Returns `None` when no
/// path can handle the pipeline shape, allowing the caller to fall back to `Val`-based execution.
pub(crate) fn run_with_env<'a, V>(
    source: V,
    body: &pipeline::PipelineBody,
    cache: Option<&dyn pipeline::PipelineData>,
    base_env: &Env,
) -> Option<Result<Val, EvalError>>
where
    V: ValueView<'a>,
{
    if let Some(result) = run_terminal_collect(source.clone(), body) {
        return Some(result);
    }
    if let Some(result) = run_full(source.clone(), body) {
        return Some(result);
    }
    if let Some(result) =
        run_reducing_stage_prefix_then_materialized_suffix(source.clone(), body, cache, base_env)
    {
        return Some(result);
    }
    if let Some(result) =
        run_sort_prefix_then_materialized_suffix(source.clone(), body, cache, base_env)
    {
        return Some(result);
    }
    run_prefix_then_materialized_suffix(source, body, cache, base_env)
}

/// Runs the complete pipeline entirely in the view domain when all stages and
/// the sink have a `ViewCapability`. Returns `None` when any stage lacks
/// view support, allowing a less specialised path to take over.
fn run_full<'a, V>(source: V, body: &pipeline::PipelineBody) -> Option<Result<Val, EvalError>>
where
    V: ValueView<'a>,
{
    let capabilities = pipeline::view_capabilities(body)?;
    let mut sink_acc = pipeline::SinkAccumulator::new(&body.sink);
    let source_demand = pipeline::Pipeline::segment_source_demand(&body.stages, &body.sink)
        .chain
        .pull;

    drive_view_frontier(
        source,
        &capabilities.stages,
        &body.stage_kernels,
        source_demand,
        |item| observe_view_sink(item, capabilities.sink, &mut sink_acc, &body.sink_kernels),
    )?;

    Some(Ok(sink_acc.finish(false)))
}

/// Feeds one view row into the sink accumulator according to `sink`'s capability.
/// Returns `Some(action)` indicating whether to `Emit`, `Skip`, or `Stop`;
/// returns `None` when a kernel lookup fails (signals the view path is unusable).
fn observe_view_sink<'a, V>(
    item: &V,
    sink: pipeline::ViewSinkCapability,
    sink_acc: &mut pipeline::SinkAccumulator,
    sink_kernels: &[pipeline::BodyKernel],
) -> Option<ViewRowAction>
where
    V: ValueView<'a>,
{
    match sink {
        pipeline::ViewSinkCapability::Collect => {
            debug_assert_eq!(
                sink.materialization(),
                pipeline::ViewMaterialization::SinkOutputRows
            );
            sink_acc.observe_collect(item.materialize());
            Some(ViewRowAction::Emit)
        }
        pipeline::ViewSinkCapability::Builtin {
            accumulator,
            predicate_kernel,
            project_kernel,
            ..
        } => {
            if !view_sink_predicate_matches(item, predicate_kernel, sink_kernels)? {
                return Some(ViewRowAction::Skip);
            }
            let sink_done = sink_acc.observe_builtin_lazy(
                accumulator,
                || item.materialize(),
                || {
                    let kernel = project_kernel?;
                    let kernel = sink_kernels.get(kernel)?;
                    eval_owned_scalar_or_value_kernel(item, kernel)
                },
                || Some(eval_view_key(item, None)?.object_key().to_string()),
            )?;
            Some(if sink_done {
                ViewRowAction::Stop
            } else {
                ViewRowAction::Emit
            })
        }
    }
}

/// Evaluates the sink's optional predicate kernel against `item`. Returns
/// `Some(true)` when there is no predicate, `Some(bool)` for the predicate
/// result, or `None` when the kernel index is out of bounds.
fn view_sink_predicate_matches<'a, V>(
    item: &V,
    predicate_kernel: Option<usize>,
    sink_kernels: &[pipeline::BodyKernel],
) -> Option<bool>
where
    V: ValueView<'a>,
{
    let Some(kernel_idx) = predicate_kernel else {
        return Some(true);
    };
    let kernel = sink_kernels.get(kernel_idx)?;
    eval_filter_kernel(item, kernel)
}

/// Runs as many leading stages as possible in the view domain, materialises the
/// resulting boundary rows, then continues execution with the standard pipeline
/// runner on the remaining suffix stages.
fn run_prefix_then_materialized_suffix<'a, V>(
    source: V,
    body: &pipeline::PipelineBody,
    cache: Option<&dyn pipeline::PipelineData>,
    base_env: &Env,
) -> Option<Result<Val, EvalError>>
where
    V: ValueView<'a>,
{
    let prefix = pipeline::view_prefix_capabilities(body)?;
    if prefix.consumed_stages >= body.stages.len()
        && !body.suffix_can_run_with_materialized_receiver(prefix.consumed_stages)
    {
        return None;
    }
    if !body.suffix_can_run_with_materialized_receiver(prefix.consumed_stages) {
        return None;
    }

    let mut boundary_rows = Vec::new();
    let source_demand = PullDemand::All;

    drive_view_frontier(
        source,
        &prefix.stages,
        &body.stage_kernels,
        source_demand,
        |item| {
            boundary_rows.push(item.materialize());
            Some(ViewRowAction::Emit)
        },
    )?;

    Some(run_materialized_suffix(
        body,
        prefix.consumed_stages,
        boundary_rows,
        cache,
        base_env,
    ))
}

/// Optimised path for pipelines whose suffix is a pure collect sink. Builds a
/// `TerminalCollectPlan` that may fuse trailing projection stages into the
/// collection kernel, avoiding a separate map pass.
fn run_terminal_collect<'a, V>(
    source: V,
    body: &pipeline::PipelineBody,
) -> Option<Result<Val, EvalError>>
where
    V: ValueView<'a>,
{
    let plan = terminal_collect_plan(body)?;
    let mut collector = pipeline::TerminalCollector::new(&plan.collect_kernel);
    let source_demand = pipeline::Pipeline::segment_source_demand(&body.stages, &body.sink)
        .chain
        .pull;

    drive_view_frontier(
        source,
        &plan.prefix,
        &body.stage_kernels,
        source_demand,
        |item| {
            collector.push_view_row(item, &plan.collect_kernel)?;
            Some(ViewRowAction::Emit)
        },
    )?;

    Some(Ok(collector.finish()))
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
    stages: &[pipeline::ViewStageCapability],
    stage_kernels: &[pipeline::BodyKernel],
    source_demand: PullDemand,
    observe: F,
) -> Option<()>
where
    V: ValueView<'a>,
    F: FnMut(&V) -> Option<ViewRowAction>,
{
    let items = source.array_iter()?;
    drive_view_iter(items, stages, stage_kernels, source_demand, observe)
}

/// Drives an arbitrary `items` iterator through the view-stage frontier, calling
/// `observe` for each row that survives all stages. Respects `source_demand`
/// limits on both inputs consumed and outputs emitted.
fn drive_view_iter<'a, V, I, F>(
    items: I,
    stages: &[pipeline::ViewStageCapability],
    stage_kernels: &[pipeline::BodyKernel],
    source_demand: PullDemand,
    mut observe: F,
) -> Option<()>
where
    V: ValueView<'a>,
    I: IntoIterator<Item = V>,
    F: FnMut(&V) -> Option<ViewRowAction>,
{
    let mut op_state: Vec<ViewStageState> = (0..stages.len())
        .map(|_| ViewStageState::default())
        .collect();
    let mut pulled_inputs = 0usize;
    let mut emitted_outputs = 0usize;

    for row in items {
        if matches!(source_demand, PullDemand::FirstInput(n) if pulled_inputs >= n) {
            break;
        }
        pulled_inputs += 1;

        if matches!(
            drive_view_item(
                row,
                0,
                stages,
                &mut op_state,
                stage_kernels,
                source_demand,
                &mut emitted_outputs,
                &mut observe,
            )?,
            ViewDriveFlow::Stop
        ) {
            break;
        }
    }

    Some(())
}

/// Recursively applies one view stage to `item`, then advances to the next stage.
/// When all stages have been applied it calls `observe`. `FlatMap` stages expand
/// into child views, each of which is recursed independently.
fn drive_view_item<'a, V, F>(
    item: V,
    stage_idx: usize,
    stages: &[pipeline::ViewStageCapability],
    op_state: &mut [ViewStageState],
    stage_kernels: &[pipeline::BodyKernel],
    source_demand: PullDemand,
    emitted_outputs: &mut usize,
    observe: &mut F,
) -> Option<ViewDriveFlow>
where
    V: ValueView<'a>,
    F: FnMut(&V) -> Option<ViewRowAction>,
{
    let Some(stage) = stages.get(stage_idx).copied() else {
        return match observe(&item)? {
            ViewRowAction::Skip => Some(ViewDriveFlow::Continue),
            ViewRowAction::Emit => {
                *emitted_outputs += 1;
                Some(
                    if matches!(source_demand, PullDemand::UntilOutput(n) if *emitted_outputs >= n)
                    {
                        ViewDriveFlow::Stop
                    } else {
                        ViewDriveFlow::Continue
                    },
                )
            }
            ViewRowAction::Stop => Some(ViewDriveFlow::Stop),
        };
    };

    if let pipeline::ViewStageCapability::FlatMap { kernel } = stage {
        debug_assert_eq!(stage.input_mode(), pipeline::ViewInputMode::ReadsView);
        debug_assert_eq!(
            stage.output_mode(),
            pipeline::ViewOutputMode::BorrowedSubviews
        );
        let kernel = stage_kernels.get(kernel)?;
        for child in eval_flat_map_kernel(&item, kernel)? {
            if matches!(
                drive_view_item(
                    child,
                    stage_idx + 1,
                    stages,
                    op_state,
                    stage_kernels,
                    source_demand,
                    emitted_outputs,
                    observe,
                )?,
                ViewDriveFlow::Stop
            ) {
                return Some(ViewDriveFlow::Stop);
            }
        }
        return Some(ViewDriveFlow::Continue);
    }

    match apply_view_stage(item, stage, stage_idx, op_state, stage_kernels)? {
        ViewStageFlow::Keep(next) => drive_view_item(
            next,
            stage_idx + 1,
            stages,
            op_state,
            stage_kernels,
            source_demand,
            emitted_outputs,
            observe,
        ),
        ViewStageFlow::Drop => Some(ViewDriveFlow::Continue),
        ViewStageFlow::Stop => Some(ViewDriveFlow::Stop),
    }
}

/// Execution plan for the terminal-collect fast path. Contains the view-domain
/// prefix stages and a potentially composed collection kernel.
struct TerminalCollectPlan {
    /// Stages that run entirely in the view domain before collection.
    prefix: Vec<pipeline::ViewStageCapability>,
    /// The kernel used to extract the value of each row for the output array.
    collect_kernel: pipeline::BodyKernel,
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
    let source_demand = pipeline::Pipeline::segment_source_demand(suffix_stages, &body.sink)
        .chain
        .pull;
    if let Some((prefix_len, collect_kernel)) = terminal_projection_run(body, start) {
        return Some(TerminalCollectPlan {
            prefix: terminal_collect_prefix_from(&suffix_stages[..prefix_len], body, start)?,
            collect_kernel,
            source_demand,
        });
    }

    Some(TerminalCollectPlan {
        prefix: terminal_collect_prefix_from(suffix_stages, body, start)?,
        collect_kernel: pipeline::BodyKernel::Current,
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
    let suffix_stages = body.stages.get(start..)?;
    let mut idx = suffix_stages.len();
    let mut kernel = pipeline::BodyKernel::Current;
    let mut found = false;

    while idx > 0 {
        let abs_idx = start + idx - 1;
        let Some(stage_kernel) = terminal_projection_stage_kernel(
            &body.stages[abs_idx],
            abs_idx,
            body.stage_kernels.get(abs_idx),
        ) else {
            break;
        };
        kernel = compose_projection_kernels(stage_kernel, kernel);
        found = true;
        idx -= 1;
    }

    found.then_some((idx, kernel))
}

/// Returns the view-native `BodyKernel` for a single trailing stage if it can
/// be fused into the terminal collect kernel. Returns `None` for stages that
/// are not projections or do not have a view-native kernel.
fn terminal_projection_stage_kernel(
    stage: &pipeline::Stage,
    idx: usize,
    kernel: Option<&pipeline::BodyKernel>,
) -> Option<pipeline::BodyKernel> {
    if matches!(
        stage.view_capability(idx, kernel),
        Some(pipeline::ViewStageCapability::Map { .. })
    ) {
        let kernel = kernel?;
        return kernel.is_view_native().then(|| kernel.clone());
    }

    match stage {
        pipeline::Stage::Builtin(call) if call.spec().view_scalar => {
            Some(pipeline::BodyKernel::BuiltinCall {
                receiver: Box::new(pipeline::BodyKernel::Current),
                call: call.clone(),
            })
        }
        _ => None,
    }
}

/// Composes two `BodyKernel` projection steps into a single `Compose` kernel,
/// or returns `first` directly when `then` is the identity `Current` kernel.
fn compose_projection_kernels(
    first: pipeline::BodyKernel,
    then: pipeline::BodyKernel,
) -> pipeline::BodyKernel {
    if matches!(then, pipeline::BodyKernel::Current) {
        return first;
    }
    pipeline::BodyKernel::Compose {
        first: Box::new(first),
        then: Box::new(then),
    }
}

/// Converts the given `stages` slice into a vec of `ViewStageCapability` for use
/// as the prefix in a `TerminalCollectPlan`. Returns `None` if any stage has
/// a `ViewMaterialization` other than `Never`.
fn terminal_collect_prefix_from(
    stages: &[pipeline::Stage],
    body: &pipeline::PipelineBody,
    start: usize,
) -> Option<Vec<pipeline::ViewStageCapability>> {
    let mut prefix = Vec::with_capacity(stages.len());
    for (offset, stage) in stages.iter().enumerate() {
        let idx = start + offset;
        let capability = stage.view_capability(idx, body.stage_kernels.get(idx))?;
        if !matches!(
            capability.materialization(),
            pipeline::ViewMaterialization::Never
        ) {
            return None;
        }
        prefix.push(capability);
    }
    Some(prefix)
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
) -> Option<Result<Val, EvalError>>
where
    V: ValueView<'a>,
{
    let mut plan = reducer_stage::plan(body)?;
    if !body.suffix_can_run_with_materialized_receiver(plan.consumed_stages) {
        return None;
    }
    let source_demand = pipeline::Pipeline::segment_source_demand(&body.stages, &body.sink)
        .chain
        .pull;

    drive_view_frontier(
        source,
        &plan.prefix,
        &body.stage_kernels,
        source_demand,
        |item| {
            plan.reducer.observe(item, &body.stage_kernels)?;
            Some(ViewRowAction::Emit)
        },
    )?;

    Some(run_materialized_value_suffix(
        body,
        plan.consumed_stages,
        plan.reducer.finish(),
        cache,
        base_env,
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
) -> Option<Result<Val, EvalError>>
where
    V: ValueView<'a>,
{
    let plan = sort_barrier_plan(body)?;
    let strategies =
        pipeline::compute_strategies_with_kernels(&body.stages, &body.stage_kernels, &body.sink);
    let strategy = strategies
        .get(plan.sort_stage)
        .copied()
        .unwrap_or(pipeline::StageStrategy::Default);
    if matches!(strategy, pipeline::StageStrategy::SortUntilOutput(_)) {
        return run_sort_prefix_then_view_suffix(source, body, &plan);
    }
    let collect_suffix = terminal_collect_plan_from(body, plan.sort_stage + 1);
    if collect_suffix.is_none()
        && !body.suffix_can_run_with_materialized_receiver(plan.sort_stage + 1)
    {
        return None;
    }

    let mut sorter =
        pipeline::BoundedKeySorter::new(plan.descending, strategy, pipeline::cmp_val_total);
    drive_view_frontier(
        source,
        &plan.prefix,
        &body.stage_kernels,
        PullDemand::All,
        |item| {
            let key = view_sort_key(item, plan.key_kernel, &body.stage_kernels)?;
            sorter.push_keyed(key, item.clone());
            Some(ViewRowAction::Emit)
        },
    )?;

    let winners = sorter.finish();
    if let Some(collect_plan) = collect_suffix {
        return run_sorted_rows_terminal_collect_suffix(
            winners,
            &collect_plan,
            &body.stage_kernels,
        );
    }
    let boundary_rows: Vec<Val> = winners.into_iter().map(|row| row.materialize()).collect();

    Some(run_materialized_suffix(
        body,
        plan.sort_stage + 1,
        boundary_rows,
        cache,
        base_env,
    ))
}

/// Feeds a pre-sorted vec of view rows through the terminal-collect plan,
/// applying any remaining prefix stages and the fused projection kernel without
/// a separate materialisation step.
fn run_sorted_rows_terminal_collect_suffix<'a, V>(
    rows: Vec<V>,
    plan: &TerminalCollectPlan,
    stage_kernels: &[pipeline::BodyKernel],
) -> Option<Result<Val, EvalError>>
where
    V: ValueView<'a>,
{
    let mut collector = pipeline::TerminalCollector::new(&plan.collect_kernel);
    drive_view_iter(
        rows,
        &plan.prefix,
        stage_kernels,
        plan.source_demand,
        |item| {
            collector.push_view_row(item, &plan.collect_kernel)?;
            Some(ViewRowAction::Emit)
        },
    )?;

    Some(Ok(collector.finish()))
}

/// Handles `SortUntilOutput` strategy: sorts rows with an `OrderedKeySorter`
/// then drives them through the view-domain suffix to enable lazy top-N pulls
/// that stop as soon as the output demand is met.
fn run_sort_prefix_then_view_suffix<'a, V>(
    source: V,
    body: &pipeline::PipelineBody,
    plan: &SortBarrierPlan,
) -> Option<Result<Val, EvalError>>
where
    V: ValueView<'a>,
{
    let suffix = view_suffix_capabilities(body, plan.sort_stage + 1)?;
    let mut sorter = pipeline::OrderedKeySorter::new(plan.descending, pipeline::cmp_val_total);

    drive_view_frontier(
        source,
        &plan.prefix,
        &body.stage_kernels,
        PullDemand::All,
        |item| {
            let key = view_sort_key(item, plan.key_kernel, &body.stage_kernels)?;
            sorter.push_keyed(key, item.clone());
            Some(ViewRowAction::Emit)
        },
    )?;

    let mut sink_acc = pipeline::SinkAccumulator::new(&body.sink);
    let source_demand =
        pipeline::Pipeline::segment_source_demand(&body.stages[plan.sort_stage + 1..], &body.sink)
            .chain
            .pull;

    drive_view_iter(
        sorter.finish(),
        &suffix.stages,
        &body.stage_kernels,
        source_demand,
        |item| observe_view_sink(item, suffix.sink, &mut sink_acc, &body.sink_kernels),
    )?;

    Some(Ok(sink_acc.finish(false)))
}

/// Plan produced when a `Sort` barrier is detected. Records the view-domain
/// prefix, the index of the sort stage, and the key extraction configuration.
struct SortBarrierPlan {
    /// View-domain stages that precede the sort barrier.
    prefix: Vec<pipeline::ViewStageCapability>,
    /// Index of the `Sort` stage within `body.stages`.
    sort_stage: usize,
    /// Stage-kernel index for the sort key, or `None` for a natural (identity) sort.
    key_kernel: Option<usize>,
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
                let key_kernel = if spec.key.is_some() {
                    Some(
                        body.stage_kernels
                            .get(idx)?
                            .is_view_native()
                            .then_some(idx)?,
                    )
                } else {
                    None
                };
                return Some(SortBarrierPlan {
                    prefix,
                    sort_stage: idx,
                    key_kernel,
                    descending: spec.descending,
                });
            }
            _ => {
                let capability = stage.view_capability(idx, body.stage_kernels.get(idx))?;
                if !matches!(
                    capability.materialization(),
                    pipeline::ViewMaterialization::Never
                ) {
                    return None;
                }
                prefix.push(capability);
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
) -> Result<Val, EvalError> {
    let suffix = suffix_body(body, consumed_stages)
        .with_source(pipeline::Source::Receiver(Val::arr(boundary_rows)));
    let root = Val::Null;
    suffix.run_with_env(&root, base_env, cache)
}

/// Runs the suffix of `body` against a single `boundary_value` (e.g. the
/// output of a keyed-reduce barrier). Short-circuits to return the value
/// directly when no suffix stages remain and the sink is `Collect`.
fn run_materialized_value_suffix(
    body: &pipeline::PipelineBody,
    consumed_stages: usize,
    boundary_value: Val,
    cache: Option<&dyn pipeline::PipelineData>,
    base_env: &Env,
) -> Result<Val, EvalError> {
    if consumed_stages >= body.stages.len() && matches!(body.sink, pipeline::Sink::Collect) {
        return Ok(boundary_value);
    }
    let suffix =
        suffix_body(body, consumed_stages).with_source(pipeline::Source::Receiver(boundary_value));
    let root = Val::Null;
    suffix.run_with_env(&root, base_env, cache)
}

/// Applies a single view stage to `item`, returning the control flow decision
/// (`Keep`, `Drop`, or `Stop`). Delegates to `stage_flow::apply_stage`.
fn apply_view_stage<'a, V>(
    item: V,
    stage: pipeline::ViewStageCapability,
    op_idx: usize,
    op_state: &mut [ViewStageState],
    stage_kernels: &[pipeline::BodyKernel],
) -> Option<ViewStageFlow<V>>
where
    V: ValueView<'a>,
{
    stage_flow::apply_stage(item, stage, op_idx, op_state, stage_kernels)
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

/// Evaluates `kernel` against `item` as a boolean predicate.
/// Returns `None` when kernel evaluation is not supported in the view domain.
fn eval_filter_kernel<'a, V>(item: &V, kernel: &pipeline::BodyKernel) -> Option<bool>
where
    V: ValueView<'a>,
{
    match pipeline::eval_view_kernel(kernel, item)? {
        pipeline::ViewKernelValue::View(view) => Some(view.scalar().truthy()),
        pipeline::ViewKernelValue::Owned(value) => Some(crate::util::is_truthy(&value)),
    }
}

/// Evaluates `kernel` against `item` as a view-domain projection, returning the
/// result as a `V` subview. Returns `None` when the kernel produces an owned
/// value (requiring materialisation) rather than a borrowed view.
fn eval_map_kernel<'a, V>(item: &V, kernel: &pipeline::BodyKernel) -> Option<V>
where
    V: ValueView<'a>,
{
    match pipeline::eval_view_kernel(kernel, item)? {
        pipeline::ViewKernelValue::View(view) => Some(view),
        pipeline::ViewKernelValue::Owned(_) => None,
    }
}

/// Evaluates `kernel` against `item` expecting an array result, returning an
/// iterator of child views. Returns `None` when the kernel produces an owned
/// `Val` (not array-iterable in the view domain).
fn eval_flat_map_kernel<'a, V>(
    item: &V,
    kernel: &pipeline::BodyKernel,
) -> Option<Box<dyn Iterator<Item = V> + 'a>>
where
    V: ValueView<'a>,
{
    match pipeline::eval_view_kernel(kernel, item)? {
        pipeline::ViewKernelValue::View(view) => view.array_iter(),
        pipeline::ViewKernelValue::Owned(_) => None,
    }
}

/// Evaluates `kernel` against `item` and extracts a scalar `Val`. For view
/// results, attempts a direct scalar conversion before falling back to full
/// materialisation.
fn eval_owned_scalar_or_value_kernel<'a, V>(item: &V, kernel: &pipeline::BodyKernel) -> Option<Val>
where
    V: ValueView<'a>,
{
    match pipeline::eval_view_kernel(kernel, item)? {
        pipeline::ViewKernelValue::View(view) => {
            scalar_view_to_owned_val(view.scalar()).or_else(|| Some(view.materialize()))
        }
        pipeline::ViewKernelValue::Owned(value) => Some(value),
    }
}

/// Extracts a `ViewKey` from `item` using `kernel` when provided, or from the
/// item's own scalar directly. Used for dedup (`distinct`) and group-by keying
/// without materialising the full value.
fn eval_view_key<'a, V>(item: &V, kernel: Option<&pipeline::BodyKernel>) -> Option<ViewKey>
where
    V: ValueView<'a>,
{
    match kernel {
        Some(kernel) => match pipeline::eval_view_kernel(kernel, item)? {
            pipeline::ViewKernelValue::View(view) => ViewKey::from_view(view.scalar())
                .or_else(|| Some(ViewKey::from_owned(view.materialize()))),
            pipeline::ViewKernelValue::Owned(value) => Some(ViewKey::from_owned(value)),
        },
        None => ViewKey::from_view(item.scalar())
            .or_else(|| Some(ViewKey::from_owned(item.materialize()))),
    }
}

/// Extracts a sort key `Val` from `item`, optionally applying `stage_kernels[kernel_idx]`
/// as a projection. Falls back to `item.materialize()` when no kernel is specified.
fn view_sort_key<'a, V>(
    item: &V,
    kernel_idx: Option<usize>,
    stage_kernels: &[pipeline::BodyKernel],
) -> Option<Val>
where
    V: ValueView<'a>,
{
    match kernel_idx {
        Some(idx) => eval_owned_scalar_or_value_kernel(item, stage_kernels.get(idx)?),
        None => Some(item.materialize()),
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::rc::Rc;
    use std::sync::Arc;

    use crate::parse::ast::BinOp;
    use crate::context::Env;
    use crate::exec::pipeline::{BodyKernel, PipelineBody, Sink, Stage, ViewStageCapability};
    use crate::util::JsonView;
    use crate::data::value::Val;
    use crate::data::view::{ValView, ValueView};

    #[derive(Clone)]
    struct CountingView {
        rows: Arc<[i64]>,
        idx: Option<usize>,
        scalar_reads: Rc<Cell<usize>>,
        materialize_reads: Rc<Cell<usize>>,
    }

    impl CountingView {
        fn root(rows: &[i64]) -> Self {
            Self {
                rows: rows.iter().copied().collect::<Vec<_>>().into(),
                idx: None,
                scalar_reads: Rc::new(Cell::new(0)),
                materialize_reads: Rc::new(Cell::new(0)),
            }
        }

        fn scalar_reads(&self) -> usize {
            self.scalar_reads.get()
        }

        fn materialize_reads(&self) -> usize {
            self.materialize_reads.get()
        }
    }

    impl<'a> ValueView<'a> for CountingView {
        fn scalar(&self) -> JsonView<'_> {
            self.scalar_reads.set(self.scalar_reads.get() + 1);
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
                materialize_reads: Rc::clone(&self.materialize_reads),
            }
        }

        fn index(&self, idx: i64) -> Self {
            let idx = if idx >= 0 { Some(idx as usize) } else { None };
            Self {
                rows: Arc::clone(&self.rows),
                idx,
                scalar_reads: Rc::clone(&self.scalar_reads),
                materialize_reads: Rc::clone(&self.materialize_reads),
            }
        }

        fn array_iter(&self) -> Option<Box<dyn Iterator<Item = Self> + 'a>> {
            if self.idx.is_some() {
                return None;
            }
            let rows = Arc::clone(&self.rows);
            let scalar_reads = Rc::clone(&self.scalar_reads);
            let materialize_reads = Rc::clone(&self.materialize_reads);
            Some(Box::new((0..rows.len()).map(move |idx| Self {
                rows: Arc::clone(&rows),
                idx: Some(idx),
                scalar_reads: Rc::clone(&scalar_reads),
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
        assert!(matches!(plan.collect_kernel, BodyKernel::Current));
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
        assert!(matches!(plan.collect_kernel, BodyKernel::Current));
    }

    #[test]
    fn terminal_collect_plan_composes_trailing_projection_builtins() {
        let call = crate::builtins::BuiltinCall {
            method: crate::builtins::BuiltinMethod::Upper,
            args: crate::builtins::BuiltinArgs::None,
        };
        assert!(call.spec().view_scalar);
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
        assert!(matches!(plan.collect_kernel, BodyKernel::Compose { .. }));
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

        let out = super::run_terminal_collect(source.clone(), &body)
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

        let out = super::run_terminal_collect(ValView::new(&source), &body)
            .unwrap()
            .unwrap();

        let out_json: serde_json::Value = out.into();
        assert_eq!(out_json, serde_json::json!([1, 2]));
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
        let out = super::run_reducing_stage_prefix_then_materialized_suffix(
            source.clone(),
            &body,
            None,
            &env,
        )
        .unwrap()
        .unwrap();

        let out_json: serde_json::Value = out.into();
        assert_eq!(out_json, serde_json::json!({"1": 3, "2": 2, "3": 1}));
        assert_eq!(source.scalar_reads(), 6);
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
        let out = super::run_reducing_stage_prefix_then_materialized_suffix(
            source.clone(),
            &body,
            None,
            &env,
        )
        .unwrap()
        .unwrap();

        let out_json: serde_json::Value = out.into();
        assert_eq!(out_json, serde_json::json!({"1": 1, "2": 2, "3": 3}));
        assert_eq!(source.scalar_reads(), 4);
        assert_eq!(source.materialize_reads(), 4);
    }
}
