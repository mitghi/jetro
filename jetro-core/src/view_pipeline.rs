//! View-backed execution for streamable pipeline bodies.

use std::sync::Arc;

use crate::chain_ir::PullDemand;
use crate::context::{Env, EvalError};
use crate::pipeline;
use crate::value::Val;
use crate::value_view::ValueView;
use crate::vm::{CompiledObjEntry, Opcode, Program};

enum ViewStageFlow<V> {
    Keep(V),
    Drop,
    Stop,
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
    if let Some(result) = run_terminal_map_collect(source.clone(), body) {
        return Some(result);
    }
    if let Some(result) = run_sort_prefix_then_materialized_suffix(source.clone(), body, cache) {
        return Some(result);
    }
    run_prefix_then_materialized_suffix(source, body, cache)
}

fn run_full<'a, V>(source: V, body: &pipeline::PipelineBody) -> Option<Result<Val, EvalError>>
where
    V: ValueView<'a>,
{
    let capabilities = pipeline::view_capabilities(body)?;
    let items = source.array_iter()?;

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
    let source_demand = pipeline::Pipeline::segment_source_demand(&body.stages, &body.sink)
        .chain
        .pull;
    let mut pulled_inputs = 0usize;
    let mut emitted_outputs = 0usize;

    'outer: for row in items {
        if matches!(source_demand, PullDemand::AtMost(n) if pulled_inputs >= n) {
            break 'outer;
        }
        pulled_inputs += 1;

        let mut item = row;
        for (op_idx, stage) in capabilities.stages.iter().enumerate() {
            match apply_view_stage(item, *stage, op_idx, &mut op_state, &body.stage_kernels)? {
                ViewStageFlow::Keep(next) => item = next,
                ViewStageFlow::Drop => continue 'outer,
                ViewStageFlow::Stop => break 'outer,
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

        emitted_outputs += 1;
        if matches!(source_demand, PullDemand::UntilOutput(n) if emitted_outputs >= n) {
            break 'outer;
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

    let items = source.array_iter()?;
    let mut boundary_rows = Vec::new();
    let mut op_state: Vec<usize> = vec![0; prefix.stages.len()];
    let source_demand = pipeline::Pipeline::segment_source_demand(&body.stages, &body.sink)
        .chain
        .pull;
    let mut pulled_inputs = 0usize;
    let mut emitted_outputs = 0usize;

    'outer: for row in items {
        if matches!(source_demand, PullDemand::AtMost(n) if pulled_inputs >= n) {
            break 'outer;
        }
        pulled_inputs += 1;

        let mut item = row;
        for (op_idx, stage) in prefix.stages.iter().enumerate() {
            match apply_view_stage(item, *stage, op_idx, &mut op_state, &body.stage_kernels)? {
                ViewStageFlow::Keep(next) => item = next,
                ViewStageFlow::Drop => continue 'outer,
                ViewStageFlow::Stop => break 'outer,
            }
        }
        boundary_rows.push(item.materialize());
        emitted_outputs += 1;
        if matches!(source_demand, PullDemand::UntilOutput(n) if emitted_outputs >= n) {
            break 'outer;
        }
    }

    let suffix = suffix_body(body, prefix.consumed_stages)
        .with_source(pipeline::Source::Receiver(Val::arr(boundary_rows)));
    let root = Val::Null;
    let env = Env::new(Val::Null);
    Some(suffix.run_with_env(&root, &env, cache))
}

fn run_terminal_map_collect<'a, V>(
    source: V,
    body: &pipeline::PipelineBody,
) -> Option<Result<Val, EvalError>>
where
    V: ValueView<'a>,
{
    if !matches!(body.sink, pipeline::Sink::Collect) {
        return None;
    }
    let map_idx = body
        .stages
        .iter()
        .position(|stage| matches!(stage, pipeline::Stage::Map(_)))?;
    if map_idx + 1 != body.stages.len() {
        return None;
    }
    let map_kernel = body.stage_kernels.get(map_idx)?;
    if !map_kernel.is_view_native() {
        return None;
    }
    for (idx, stage) in body.stages[..map_idx].iter().enumerate() {
        stage.view_capability(idx, body.stage_kernels.get(idx))?;
    }
    let items = source.array_iter()?;
    let mut collector = pipeline::TerminalMapCollector::new(map_kernel);
    let mut op_state: Vec<usize> = vec![0; map_idx];

    'outer: for row in items {
        let mut item = row;
        for (op_idx, stage) in body.stages[..map_idx].iter().enumerate() {
            let capability = stage.view_capability(op_idx, body.stage_kernels.get(op_idx))?;
            match apply_view_stage(item, capability, op_idx, &mut op_state, &body.stage_kernels)? {
                ViewStageFlow::Keep(next) => item = next,
                ViewStageFlow::Drop => continue 'outer,
                ViewStageFlow::Stop => break 'outer,
            }
        }
        collector.push_view_row(&item, map_kernel)?;
    }

    Some(Ok(collector.finish()))
}

fn run_sort_prefix_then_materialized_suffix<'a, V>(
    source: V,
    body: &pipeline::PipelineBody,
    cache: Option<&dyn pipeline::PipelineData>,
) -> Option<Result<Val, EvalError>>
where
    V: ValueView<'a>,
{
    let pipeline::Stage::Sort(spec) = body.stages.first()? else {
        return None;
    };
    let strategies = pipeline::compute_strategies(&body.stages, &body.sink);
    let strategy = strategies.first().copied()?;
    if matches!(strategy, pipeline::StageStrategy::Default) {
        return None;
    }
    if !stages_can_run_with_materialized_receiver(&body.stages[1..])
        || !sink_can_run_with_materialized_receiver(&body.sink)
    {
        return None;
    }

    let key_kernel = match spec.key.as_ref() {
        Some(_) => body.stage_kernels.first()?,
        None => return None,
    };
    if !key_kernel.is_view_native() {
        return None;
    }

    let items = source.array_iter()?;
    let winners = match pipeline::bounded_sort_by_key(items, spec.descending, strategy, |row| {
        eval_value_kernel(row, key_kernel)
            .ok_or_else(|| EvalError("view sort: unsupported key".into()))
    }) {
        Ok(winners) => winners,
        Err(err) => return Some(Err(err)),
    };
    let boundary_rows: Vec<Val> = winners.into_iter().map(|row| row.materialize()).collect();

    let suffix =
        suffix_body(body, 1).with_source(pipeline::Source::Receiver(Val::arr(boundary_rows)));
    let root = Val::Null;
    let env = Env::new(Val::Null);
    Some(suffix.run_with_env(&root, &env, cache))
}

fn apply_view_stage<'a, V>(
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
            if eval_filter_kernel(&item, kernel)? {
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
            Some(ViewStageFlow::Keep(eval_map_kernel(&item, kernel)?))
        }
    }
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
        Opcode::MakeObj(entries) => entries.iter().all(obj_entry_is_current_only),
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

fn obj_entry_is_current_only(entry: &CompiledObjEntry) -> bool {
    match entry {
        CompiledObjEntry::Short { .. } | CompiledObjEntry::KvPath { .. } => true,
        CompiledObjEntry::Kv { prog, cond, .. } => {
            program_is_current_only(prog)
                && cond
                    .as_ref()
                    .is_none_or(|cond| program_is_current_only(cond))
        }
        CompiledObjEntry::Dynamic { key, val } => {
            program_is_current_only(key) && program_is_current_only(val)
        }
        CompiledObjEntry::Spread(prog) | CompiledObjEntry::SpreadDeep(prog) => {
            program_is_current_only(prog)
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

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::rc::Rc;
    use std::sync::Arc;

    use crate::ast::BinOp;
    use crate::pipeline::{BodyKernel, PipelineBody, Sink, Stage};
    use crate::util::JsonView;
    use crate::value::Val;
    use crate::value_view::ValueView;

    #[derive(Clone)]
    struct CountingView {
        rows: Arc<[i64]>,
        idx: Option<usize>,
        scalar_reads: Rc<Cell<usize>>,
    }

    impl CountingView {
        fn root(rows: &[i64]) -> Self {
            Self {
                rows: rows.iter().copied().collect::<Vec<_>>().into(),
                idx: None,
                scalar_reads: Rc::new(Cell::new(0)),
            }
        }

        fn scalar_reads(&self) -> usize {
            self.scalar_reads.get()
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
            }
        }

        fn index(&self, idx: i64) -> Self {
            let idx = if idx >= 0 { Some(idx as usize) } else { None };
            Self {
                rows: Arc::clone(&self.rows),
                idx,
                scalar_reads: Rc::clone(&self.scalar_reads),
            }
        }

        fn array_iter(&self) -> Option<Box<dyn Iterator<Item = Self> + 'a>> {
            if self.idx.is_some() {
                return None;
            }
            let rows = Arc::clone(&self.rows);
            let scalar_reads = Rc::clone(&self.scalar_reads);
            Some(Box::new((0..rows.len()).map(move |idx| Self {
                rows: Arc::clone(&rows),
                idx: Some(idx),
                scalar_reads: Rc::clone(&scalar_reads),
            })))
        }

        fn materialize(&self) -> Val {
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
                Stage::Filter(Arc::new(crate::vm::Program::new(Vec::new(), ""))),
                Stage::Take(2),
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
}
