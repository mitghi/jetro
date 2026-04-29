//! View-backed execution for streamable pipeline bodies.

use std::sync::Arc;

use crate::context::{Env, EvalError};
use crate::pipeline;
use crate::value::Val;
use crate::value_view::ValueView;

pub(crate) fn walk_fields<'a, V>(mut cur: V, keys: &[Arc<str>]) -> V
where
    V: ValueView<'a>,
{
    for key in keys {
        cur = cur.field(key.as_ref());
    }
    cur
}

pub(crate) fn run<'a, V>(
    source: V,
    body: &pipeline::PipelineBody,
    cache: Option<&dyn pipeline::PipelineData>,
) -> Option<Result<Val, EvalError>>
where
    V: ValueView<'a>,
{
    if let Some(result) = run_full(source.clone(), body) {
        return Some(result);
    }
    run_prefix_then_materialized_suffix(source, body, cache)
}

fn run_full<'a, V>(source: V, body: &pipeline::PipelineBody) -> Option<Result<Val, EvalError>>
where
    V: ValueView<'a>,
{
    let capabilities = pipeline::view_capabilities(body)?;
    let items = source.array_items()?;

    let mut acc_collect: Vec<Val> = Vec::new();
    let mut acc_count: i64 = 0;
    let mut acc_sum_i: i64 = 0;
    let mut acc_sum_f: f64 = 0.0;
    let mut sum_floated = false;
    let mut acc_min_f = f64::INFINITY;
    let mut acc_max_f = f64::NEG_INFINITY;
    let mut acc_n_obs = 0usize;
    let mut acc_first: Option<Val> = None;
    let mut acc_last: Option<Val> = None;
    let mut op_state: Vec<usize> = vec![0; capabilities.stages.len()];

    'outer: for row in items {
        let mut item = row;
        for (op_idx, stage) in capabilities.stages.iter().enumerate() {
            if !matches!(
                stage.materialization(),
                pipeline::ViewMaterialization::Never
            ) {
                return None;
            }
            match *stage {
                pipeline::ViewStageCapability::Skip(n) => {
                    debug_assert_eq!(stage.input_mode(), pipeline::ViewInputMode::SkipsViewRead);
                    debug_assert_eq!(
                        stage.output_mode(),
                        pipeline::ViewOutputMode::PreservesInputView
                    );
                    if op_state[op_idx] < n {
                        op_state[op_idx] += 1;
                        continue 'outer;
                    }
                }
                pipeline::ViewStageCapability::Take(n) => {
                    debug_assert_eq!(stage.input_mode(), pipeline::ViewInputMode::SkipsViewRead);
                    debug_assert_eq!(
                        stage.output_mode(),
                        pipeline::ViewOutputMode::PreservesInputView
                    );
                    if op_state[op_idx] >= n {
                        break 'outer;
                    }
                    op_state[op_idx] += 1;
                }
                pipeline::ViewStageCapability::Filter { kernel } => {
                    debug_assert_eq!(stage.input_mode(), pipeline::ViewInputMode::ReadsView);
                    debug_assert_eq!(
                        stage.output_mode(),
                        pipeline::ViewOutputMode::PreservesInputView
                    );
                    let kernel = body.stage_kernels.get(kernel)?;
                    let keep = eval_filter_kernel(&item, kernel)?;
                    if !keep {
                        continue 'outer;
                    }
                }
                pipeline::ViewStageCapability::Map { kernel } => {
                    debug_assert_eq!(stage.input_mode(), pipeline::ViewInputMode::ReadsView);
                    debug_assert_eq!(
                        stage.output_mode(),
                        pipeline::ViewOutputMode::BorrowedSubview
                    );
                    let kernel = body.stage_kernels.get(kernel)?;
                    item = eval_map_kernel(&item, kernel)?;
                }
            }
        }

        match capabilities.sink {
            pipeline::ViewSinkCapability::Collect => {
                debug_assert_eq!(
                    capabilities.sink.materialization(),
                    pipeline::ViewMaterialization::SinkOutputRows
                );
                acc_collect.push(item.materialize());
            }
            pipeline::ViewSinkCapability::Count => {
                debug_assert_eq!(
                    capabilities.sink.materialization(),
                    pipeline::ViewMaterialization::Never
                );
                acc_count += 1;
            }
            pipeline::ViewSinkCapability::Numeric { op, project_kernel } => {
                debug_assert_eq!(
                    capabilities.sink.materialization(),
                    pipeline::ViewMaterialization::SinkNumericInput
                );
                let numeric_item = if let Some(kernel) = project_kernel {
                    let kernel = body.sink_kernels.get(kernel)?;
                    eval_value_kernel(&item, kernel)?
                } else {
                    item.materialize()
                };
                pipeline::num_fold(
                    &mut acc_sum_i,
                    &mut acc_sum_f,
                    &mut sum_floated,
                    &mut acc_min_f,
                    &mut acc_max_f,
                    &mut acc_n_obs,
                    op,
                    &numeric_item,
                );
            }
            pipeline::ViewSinkCapability::First => {
                debug_assert_eq!(
                    capabilities.sink.materialization(),
                    pipeline::ViewMaterialization::SinkFinalRow
                );
                acc_first = Some(item.materialize());
                break 'outer;
            }
            pipeline::ViewSinkCapability::Last => {
                debug_assert_eq!(
                    capabilities.sink.materialization(),
                    pipeline::ViewMaterialization::SinkFinalRow
                );
                acc_last = Some(item.materialize());
            }
        }
    }

    Some(Ok(match capabilities.sink {
        pipeline::ViewSinkCapability::Collect => Val::arr(acc_collect),
        pipeline::ViewSinkCapability::Count => Val::Int(acc_count),
        pipeline::ViewSinkCapability::Numeric { op, .. } => pipeline::num_finalise(
            op,
            acc_sum_i,
            acc_sum_f,
            sum_floated,
            acc_min_f,
            acc_max_f,
            acc_n_obs,
        ),
        pipeline::ViewSinkCapability::First => acc_first.unwrap_or(Val::Null),
        pipeline::ViewSinkCapability::Last => acc_last.unwrap_or(Val::Null),
    }))
}

fn run_prefix_then_materialized_suffix<'a, V>(
    source: V,
    body: &pipeline::PipelineBody,
    cache: Option<&dyn pipeline::PipelineData>,
) -> Option<Result<Val, EvalError>>
where
    V: ValueView<'a>,
{
    let prefix = pipeline::view_prefix_capabilities(body)?;
    if prefix.consumed_stages >= body.stages.len() && !suffix_sink_can_run_without_outer_env(body) {
        return None;
    }
    if !suffix_stages_can_run_without_outer_env(&body.stages[prefix.consumed_stages..])
        || !suffix_sink_can_run_without_outer_env(body)
    {
        return None;
    }

    let items = source.array_items()?;
    let mut boundary_rows = Vec::new();
    let mut op_state: Vec<usize> = vec![0; prefix.stages.len()];

    'outer: for row in items {
        let mut item = row;
        for (op_idx, stage) in prefix.stages.iter().enumerate() {
            match *stage {
                pipeline::ViewStageCapability::Skip(n) => {
                    if op_state[op_idx] < n {
                        op_state[op_idx] += 1;
                        continue 'outer;
                    }
                }
                pipeline::ViewStageCapability::Take(n) => {
                    if op_state[op_idx] >= n {
                        break 'outer;
                    }
                    op_state[op_idx] += 1;
                }
                pipeline::ViewStageCapability::Filter { kernel } => {
                    let kernel = body.stage_kernels.get(kernel)?;
                    let keep = eval_filter_kernel(&item, kernel)?;
                    if !keep {
                        continue 'outer;
                    }
                }
                pipeline::ViewStageCapability::Map { kernel } => {
                    let kernel = body.stage_kernels.get(kernel)?;
                    item = eval_map_kernel(&item, kernel)?;
                }
            }
        }
        boundary_rows.push(item.materialize());
    }

    let suffix = suffix_body(body, prefix.consumed_stages)
        .with_source(pipeline::Source::Receiver(Val::arr(boundary_rows)));
    let root = Val::Null;
    let env = Env::new(Val::Null);
    Some(suffix.run_with_env(&root, &env, cache))
}

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

fn suffix_stages_can_run_without_outer_env(stages: &[pipeline::Stage]) -> bool {
    stages.iter().all(|stage| {
        matches!(
            stage,
            pipeline::Stage::Take(_)
                | pipeline::Stage::Skip(_)
                | pipeline::Stage::Reverse
                | pipeline::Stage::Sort(None)
                | pipeline::Stage::UniqueBy(None)
                | pipeline::Stage::Builtin(_)
                | pipeline::Stage::Split(_)
                | pipeline::Stage::Slice(_, _)
                | pipeline::Stage::Replace { .. }
                | pipeline::Stage::Chunk(_)
                | pipeline::Stage::Window(_)
        )
    })
}

fn suffix_sink_can_run_without_outer_env(body: &pipeline::PipelineBody) -> bool {
    match &body.sink {
        pipeline::Sink::Collect
        | pipeline::Sink::Count
        | pipeline::Sink::First
        | pipeline::Sink::Last
        | pipeline::Sink::ApproxCountDistinct => true,
        pipeline::Sink::Numeric(n) => n.project.is_none(),
    }
}

fn eval_filter_kernel<'a, V>(item: &V, kernel: &pipeline::BodyKernel) -> Option<bool>
where
    V: ValueView<'a>,
{
    match pipeline::eval_view_kernel(kernel, item)? {
        pipeline::ViewKernelValue::View(view) => Some(view.scalar().truthy()),
        pipeline::ViewKernelValue::Owned(value) => Some(crate::util::is_truthy(&value)),
    }
}

fn eval_map_kernel<'a, V>(item: &V, kernel: &pipeline::BodyKernel) -> Option<V>
where
    V: ValueView<'a>,
{
    match pipeline::eval_view_kernel(kernel, item)? {
        pipeline::ViewKernelValue::View(view) => Some(view),
        pipeline::ViewKernelValue::Owned(_) => None,
    }
}

fn eval_value_kernel<'a, V>(item: &V, kernel: &pipeline::BodyKernel) -> Option<Val>
where
    V: ValueView<'a>,
{
    match pipeline::eval_view_kernel(kernel, item)? {
        pipeline::ViewKernelValue::View(view) => Some(view.materialize()),
        pipeline::ViewKernelValue::Owned(value) => Some(value),
    }
}
