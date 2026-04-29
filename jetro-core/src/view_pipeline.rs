//! View-backed execution for streamable pipeline bodies.

use std::sync::Arc;

use crate::context::{Env, EvalError};
use crate::pipeline;
use crate::value::Val;
use crate::value_view::ValueView;
use crate::vm::{Opcode, Program};

pub(crate) fn walk_fields<'a, V>(mut cur: V, keys: &[Arc<str>]) -> V
where
    V: ValueView<'a>,
{
    for key in keys {
        cur = cur.field(key.as_ref());
    }
    cur
}

pub(crate) fn can_run_materialized_receiver(body: &pipeline::PipelineBody) -> bool {
    stages_can_run_with_materialized_receiver(&body.stages)
        && sink_can_run_with_materialized_receiver(&body.sink)
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
    if prefix.consumed_stages >= body.stages.len()
        && !sink_can_run_with_materialized_receiver(&body.sink)
    {
        return None;
    }
    if !stages_can_run_with_materialized_receiver(&body.stages[prefix.consumed_stages..])
        || !sink_can_run_with_materialized_receiver(&body.sink)
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

fn stages_can_run_with_materialized_receiver(stages: &[pipeline::Stage]) -> bool {
    stages
        .iter()
        .all(|stage| stage.can_run_with_receiver_only(program_is_current_only))
}

fn sink_can_run_with_materialized_receiver(sink: &pipeline::Sink) -> bool {
    sink.can_run_with_receiver_only(program_is_current_only)
}

fn program_is_current_only(program: &Program) -> bool {
    program.ops.iter().all(opcode_is_current_only)
}

fn opcode_is_current_only(opcode: &Opcode) -> bool {
    match opcode {
        Opcode::PushRoot | Opcode::RootChain(_) | Opcode::GetPointer(_) => false,
        Opcode::BindVar(_)
        | Opcode::StoreVar(_)
        | Opcode::BindObjDestructure(_)
        | Opcode::BindArrDestructure(_)
        | Opcode::PipelineRun { .. }
        | Opcode::LetExpr { .. }
        | Opcode::ListComp(_)
        | Opcode::DictComp(_)
        | Opcode::SetComp(_)
        | Opcode::PatchEval(_) => false,
        Opcode::DynIndex(prog)
        | Opcode::InlineFilter(prog)
        | Opcode::AndOp(prog)
        | Opcode::OrOp(prog)
        | Opcode::CoalesceOp(prog) => program_is_current_only(prog),
        Opcode::CallMethod(call) | Opcode::CallOptMethod(call) => call
            .sub_progs
            .iter()
            .all(|prog| program_is_current_only(prog)),
        Opcode::IfElse { then_, else_ } => {
            program_is_current_only(then_) && program_is_current_only(else_)
        }
        Opcode::TryExpr { body, default } => {
            program_is_current_only(body) && program_is_current_only(default)
        }
        Opcode::MakeArr(items) => items
            .iter()
            .all(|(prog, _spread)| program_is_current_only(prog)),
        Opcode::FString(parts) => parts.iter().all(|part| match part {
            crate::vm::CompiledFSPart::Lit(_) => true,
            crate::vm::CompiledFSPart::Interp { prog, .. } => program_is_current_only(prog),
        }),
        Opcode::MakeObj(_) => false,
        Opcode::PushNull
        | Opcode::PushBool(_)
        | Opcode::PushInt(_)
        | Opcode::PushFloat(_)
        | Opcode::PushStr(_)
        | Opcode::PushCurrent
        | Opcode::LoadIdent(_)
        | Opcode::GetField(_)
        | Opcode::GetIndex(_)
        | Opcode::GetSlice(_, _)
        | Opcode::OptField(_)
        | Opcode::Descendant(_)
        | Opcode::DescendAll
        | Opcode::Quantifier(_)
        | Opcode::FieldChain(_)
        | Opcode::Add
        | Opcode::Sub
        | Opcode::Mul
        | Opcode::Div
        | Opcode::Mod
        | Opcode::Eq
        | Opcode::Neq
        | Opcode::Lt
        | Opcode::Lte
        | Opcode::Gt
        | Opcode::Gte
        | Opcode::Fuzzy
        | Opcode::Not
        | Opcode::Neg
        | Opcode::CastOp(_)
        | Opcode::KindCheck { .. }
        | Opcode::SetCurrent
        | Opcode::DeleteMarkErr => true,
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
