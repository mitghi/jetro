//! View-backed execution for streamable pipeline bodies.

use std::sync::Arc;

use crate::context::EvalError;
use crate::pipeline;
use crate::value::Val;
use crate::value_view::ValueView;

pub(crate) fn supports(body: &pipeline::PipelineBody) -> bool {
    pipeline::view_capabilities(body).is_some()
}

pub(crate) fn walk_fields<'a, V>(mut cur: V, keys: &[Arc<str>]) -> V
where
    V: ValueView<'a>,
{
    for key in keys {
        cur = cur.field(key.as_ref());
    }
    cur
}

pub(crate) fn run<'a, V>(source: V, body: &pipeline::PipelineBody) -> Option<Result<Val, EvalError>>
where
    V: ValueView<'a>,
{
    let plan = classify(body)?;
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
    let mut op_state: Vec<usize> = vec![0; plan.ops.len()];

    'outer: for row in items {
        let mut item = row;
        for (op_idx, op) in plan.ops.iter().enumerate() {
            match op {
                ViewOp::Skip(n) => {
                    if op_state[op_idx] < *n {
                        op_state[op_idx] += 1;
                        continue 'outer;
                    }
                }
                ViewOp::Take(n) => {
                    if op_state[op_idx] >= *n {
                        break 'outer;
                    }
                    op_state[op_idx] += 1;
                }
                ViewOp::Filter(kernel) => {
                    let keep = eval_filter_kernel(&item, kernel)?;
                    if !keep {
                        continue 'outer;
                    }
                }
                ViewOp::Map(kernel) => {
                    item = eval_map_kernel(&item, kernel)?;
                }
            }
        }

        match plan.sink {
            ViewSink::Collect => acc_collect.push(item.materialize()),
            ViewSink::Count => acc_count += 1,
            ViewSink::Numeric { op, project } => {
                let numeric_item = if let Some(kernel) = project {
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
            ViewSink::First => {
                acc_first = Some(item.materialize());
                break 'outer;
            }
            ViewSink::Last => {
                acc_last = Some(item.materialize());
            }
        }
    }

    Some(Ok(match plan.sink {
        ViewSink::Collect => Val::arr(acc_collect),
        ViewSink::Count => Val::Int(acc_count),
        ViewSink::Numeric { op, .. } => pipeline::num_finalise(
            op,
            acc_sum_i,
            acc_sum_f,
            sum_floated,
            acc_min_f,
            acc_max_f,
            acc_n_obs,
        ),
        ViewSink::First => acc_first.unwrap_or(Val::Null),
        ViewSink::Last => acc_last.unwrap_or(Val::Null),
    }))
}

struct ViewPlan<'a> {
    ops: Vec<ViewOp<'a>>,
    sink: ViewSink<'a>,
}

fn classify(body: &pipeline::PipelineBody) -> Option<ViewPlan<'_>> {
    let capabilities = pipeline::view_capabilities(body)?;
    Some(ViewPlan {
        ops: view_ops_for_body(body, &capabilities)?,
        sink: view_sink_for_body(body, &capabilities)?,
    })
}

#[derive(Clone, Copy)]
enum ViewOp<'a> {
    Filter(&'a pipeline::BodyKernel),
    Map(&'a pipeline::BodyKernel),
    Take(usize),
    Skip(usize),
}

fn view_ops_for_body<'a>(
    body: &'a pipeline::PipelineBody,
    capabilities: &pipeline::ViewPipelineCapabilities,
) -> Option<Vec<ViewOp<'a>>> {
    let mut ops = Vec::with_capacity(capabilities.stages.len());
    for capability in &capabilities.stages {
        let op = match *capability {
            pipeline::ViewStageCapability::Filter { kernel } => {
                let kernel = body.stage_kernels.get(kernel)?;
                ViewOp::Filter(kernel)
            }
            pipeline::ViewStageCapability::Map { kernel } => {
                let kernel = body.stage_kernels.get(kernel)?;
                ViewOp::Map(kernel)
            }
            pipeline::ViewStageCapability::Take(n) => ViewOp::Take(n),
            pipeline::ViewStageCapability::Skip(n) => ViewOp::Skip(n),
        };
        ops.push(op);
    }
    Some(ops)
}

#[derive(Clone, Copy)]
enum ViewSink<'a> {
    Collect,
    Count,
    Numeric {
        op: pipeline::NumOp,
        project: Option<&'a pipeline::BodyKernel>,
    },
    First,
    Last,
}

fn view_sink_for_body<'a>(
    body: &'a pipeline::PipelineBody,
    capabilities: &pipeline::ViewPipelineCapabilities,
) -> Option<ViewSink<'a>> {
    match capabilities.sink {
        pipeline::ViewSinkCapability::Collect => Some(ViewSink::Collect),
        pipeline::ViewSinkCapability::Count => Some(ViewSink::Count),
        pipeline::ViewSinkCapability::First => Some(ViewSink::First),
        pipeline::ViewSinkCapability::Last => Some(ViewSink::Last),
        pipeline::ViewSinkCapability::Numeric { op, project_kernel } => {
            let project = match project_kernel {
                Some(kernel) => Some(body.sink_kernels.get(kernel)?),
                None => None,
            };
            Some(ViewSink::Numeric { op, project })
        }
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
