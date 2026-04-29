//! View-backed execution for streamable pipeline bodies.

use std::sync::Arc;

use crate::context::EvalError;
use crate::pipeline;
use crate::value::Val;
use crate::value_view::ValueView;

pub(crate) fn supports(body: &pipeline::PipelineBody) -> bool {
    classify(body).is_some()
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
    Some(ViewPlan {
        ops: view_ops_for_body(body)?,
        sink: classify_sink(body)?,
    })
}

#[derive(Clone, Copy)]
enum ViewOp<'a> {
    Filter(&'a pipeline::BodyKernel),
    Map(&'a pipeline::BodyKernel),
    Take(usize),
    Skip(usize),
}

fn view_ops_for_body(body: &pipeline::PipelineBody) -> Option<Vec<ViewOp<'_>>> {
    let mut ops = Vec::with_capacity(body.stages.len());
    for (idx, stage) in body.stages.iter().enumerate() {
        let op = match view_stage_shape(stage)? {
            ViewStageShape::Filter => {
                let kernel = body.stage_kernels.get(idx)?;
                if matches!(kernel, pipeline::BodyKernel::Generic) {
                    return None;
                }
                ViewOp::Filter(kernel)
            }
            ViewStageShape::Map => {
                let kernel = body.stage_kernels.get(idx)?;
                if matches!(kernel, pipeline::BodyKernel::Generic) {
                    return None;
                }
                ViewOp::Map(kernel)
            }
            ViewStageShape::Take(n) => ViewOp::Take(n),
            ViewStageShape::Skip(n) => ViewOp::Skip(n),
        };
        ops.push(op);
    }
    Some(ops)
}

enum ViewStageShape {
    Filter,
    Map,
    Take(usize),
    Skip(usize),
}

fn view_stage_shape(stage: &pipeline::Stage) -> Option<ViewStageShape> {
    match stage {
        pipeline::Stage::Filter(_) => Some(ViewStageShape::Filter),
        pipeline::Stage::Map(_) => Some(ViewStageShape::Map),
        pipeline::Stage::Take(n) => Some(ViewStageShape::Take(*n)),
        pipeline::Stage::Skip(n) => Some(ViewStageShape::Skip(*n)),
        _ => None,
    }
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

fn classify_sink(body: &pipeline::PipelineBody) -> Option<ViewSink<'_>> {
    match &body.sink {
        pipeline::Sink::Collect => Some(ViewSink::Collect),
        pipeline::Sink::Count => Some(ViewSink::Count),
        pipeline::Sink::First => Some(ViewSink::First),
        pipeline::Sink::Last => Some(ViewSink::Last),
        pipeline::Sink::Numeric(n) => {
            let project = if n.project.is_some() {
                let kernel = body.sink_kernels.first()?;
                if matches!(kernel, pipeline::BodyKernel::Generic) {
                    return None;
                }
                Some(kernel)
            } else {
                None
            };
            Some(ViewSink::Numeric { op: n.op, project })
        }
        pipeline::Sink::ApproxCountDistinct => None,
    }
}

fn eval_filter_kernel<'a, V>(item: &V, kernel: &pipeline::BodyKernel) -> Option<bool>
where
    V: ValueView<'a>,
{
    match eval_kernel(item, kernel)? {
        ViewKernelValue::View(view) => Some(view.scalar().truthy()),
        ViewKernelValue::Owned(value) => Some(crate::util::is_truthy(&value)),
    }
}

fn eval_map_kernel<'a, V>(item: &V, kernel: &pipeline::BodyKernel) -> Option<V>
where
    V: ValueView<'a>,
{
    match eval_kernel(item, kernel)? {
        ViewKernelValue::View(view) => Some(view),
        ViewKernelValue::Owned(_) => None,
    }
}

fn eval_value_kernel<'a, V>(item: &V, kernel: &pipeline::BodyKernel) -> Option<Val>
where
    V: ValueView<'a>,
{
    match eval_kernel(item, kernel)? {
        ViewKernelValue::View(view) => Some(view.materialize()),
        ViewKernelValue::Owned(value) => Some(value),
    }
}

enum ViewKernelValue<V> {
    View(V),
    Owned(Val),
}

fn eval_kernel<'a, V>(item: &V, kernel: &pipeline::BodyKernel) -> Option<ViewKernelValue<V>>
where
    V: ValueView<'a>,
{
    match kernel {
        pipeline::BodyKernel::FieldRead(key) => Some(ViewKernelValue::View(item.field(key))),
        pipeline::BodyKernel::FieldChain(keys) => {
            Some(ViewKernelValue::View(walk_fields(item.clone(), keys)))
        }
        pipeline::BodyKernel::ConstBool(value) => Some(ViewKernelValue::Owned(Val::Bool(*value))),
        pipeline::BodyKernel::Const(value) => Some(ViewKernelValue::Owned(value.clone())),
        pipeline::BodyKernel::FieldCmpLit(key, op, lit) => {
            let lhs = item.field(key);
            Some(ViewKernelValue::Owned(Val::Bool(
                crate::util::json_cmp_binop(
                    lhs.scalar(),
                    *op,
                    crate::util::JsonView::from_val(lit),
                ),
            )))
        }
        pipeline::BodyKernel::FieldChainCmpLit(keys, op, lit) => {
            let lhs = walk_fields(item.clone(), keys);
            Some(ViewKernelValue::Owned(Val::Bool(
                crate::util::json_cmp_binop(
                    lhs.scalar(),
                    *op,
                    crate::util::JsonView::from_val(lit),
                ),
            )))
        }
        pipeline::BodyKernel::CurrentCmpLit(op, lit) => Some(ViewKernelValue::Owned(Val::Bool(
            crate::util::json_cmp_binop(item.scalar(), *op, crate::util::JsonView::from_val(lit)),
        ))),
        pipeline::BodyKernel::Generic => None,
    }
}
