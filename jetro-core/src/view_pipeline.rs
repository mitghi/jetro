//! View-backed execution for streamable pipeline bodies.

use std::sync::Arc;

use crate::chain_ir::PullDemand;
use crate::context::{Env, EvalError};
use crate::pipeline;
use crate::value::Val;
use crate::value_view::{scalar_view_to_owned_val, ValueView};

mod key;
mod reducer_stage;
mod stage_flow;

use key::ViewKey;
use stage_flow::{ViewStageFlow, ViewStageState};

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
    body.can_run_with_materialized_receiver()
}

pub(crate) fn run<'a, V>(
    source: V,
    body: &pipeline::PipelineBody,
    cache: Option<&dyn pipeline::PipelineData>,
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
        run_reducing_stage_prefix_then_materialized_suffix(source.clone(), body, cache)
    {
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
            numeric_op: _,
            ..
        } => {
            if matches!(
                accumulator,
                crate::builtins::BuiltinSinkAccumulator::ApproxDistinct
            ) {
                let key = eval_view_key(item, None)?;
                sink_acc.observe_approx_distinct_key(key.object_key().as_ref());
                return Some(ViewRowAction::Emit);
            }
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
            )?;
            Some(if sink_done {
                ViewRowAction::Stop
            } else {
                ViewRowAction::Emit
            })
        }
    }
}

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
    ))
}

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

enum ViewRowAction {
    Skip,
    Emit,
    Stop,
}

enum ViewDriveFlow {
    Continue,
    Stop,
}

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

struct TerminalCollectPlan {
    prefix: Vec<pipeline::ViewStageCapability>,
    collect_kernel: pipeline::BodyKernel,
}

fn terminal_collect_plan(body: &pipeline::PipelineBody) -> Option<TerminalCollectPlan> {
    if !matches!(
        body.sink.view_capability(&body.sink_kernels)?,
        pipeline::ViewSinkCapability::Collect
    ) {
        return None;
    }

    if let Some((last_stage, prefix_stages)) = body.stages.split_last() {
        let last_idx = prefix_stages.len();
        let capability = last_stage.view_capability(last_idx, body.stage_kernels.get(last_idx))?;
        if let pipeline::ViewStageCapability::Map { kernel } = capability {
            let collect_kernel = body.stage_kernels.get(kernel)?;
            if !collect_kernel.is_view_native() {
                return None;
            }

            return Some(TerminalCollectPlan {
                prefix: terminal_collect_prefix(prefix_stages, body)?,
                collect_kernel: collect_kernel.clone(),
            });
        }
    }

    Some(TerminalCollectPlan {
        prefix: terminal_collect_prefix(&body.stages, body)?,
        collect_kernel: pipeline::BodyKernel::Current,
    })
}

fn terminal_collect_prefix(
    stages: &[pipeline::Stage],
    body: &pipeline::PipelineBody,
) -> Option<Vec<pipeline::ViewStageCapability>> {
    let mut prefix = Vec::with_capacity(stages.len());
    for (idx, stage) in stages.iter().enumerate() {
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

fn run_reducing_stage_prefix_then_materialized_suffix<'a, V>(
    source: V,
    body: &pipeline::PipelineBody,
    cache: Option<&dyn pipeline::PipelineData>,
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

    let boundary_rows = vec![plan.reducer.finish()];
    Some(run_materialized_suffix(
        body,
        plan.consumed_stages,
        boundary_rows,
        cache,
    ))
}

fn run_sort_prefix_then_materialized_suffix<'a, V>(
    source: V,
    body: &pipeline::PipelineBody,
    cache: Option<&dyn pipeline::PipelineData>,
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
    if !body.suffix_can_run_with_materialized_receiver(plan.sort_stage + 1) {
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
    let boundary_rows: Vec<Val> = winners.into_iter().map(|row| row.materialize()).collect();

    Some(run_materialized_suffix(
        body,
        plan.sort_stage + 1,
        boundary_rows,
        cache,
    ))
}

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

struct SortBarrierPlan {
    prefix: Vec<pipeline::ViewStageCapability>,
    sort_stage: usize,
    key_kernel: Option<usize>,
    descending: bool,
}

struct ViewSuffixCapabilities {
    stages: Vec<pipeline::ViewStageCapability>,
    sink: pipeline::ViewSinkCapability,
}

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

fn run_materialized_suffix(
    body: &pipeline::PipelineBody,
    consumed_stages: usize,
    boundary_rows: Vec<Val>,
    cache: Option<&dyn pipeline::PipelineData>,
) -> Result<Val, EvalError> {
    let suffix = suffix_body(body, consumed_stages)
        .with_source(pipeline::Source::Receiver(Val::arr(boundary_rows)));
    let root = Val::Null;
    let env = Env::new(Val::Null);
    suffix.run_with_env(&root, &env, cache)
}

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

    use crate::ast::BinOp;
    use crate::pipeline::{BodyKernel, PipelineBody, Sink, Stage, ViewStageCapability};
    use crate::util::JsonView;
    use crate::value::Val;
    use crate::value_view::{ValView, ValueView};

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
                Stage::Take(
                    2,
                    crate::builtins::BuiltinViewStage::Take,
                    crate::builtins::BuiltinStageMerge::UsizeMin,
                ),
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
                Stage::Take(
                    1,
                    crate::builtins::BuiltinViewStage::Take,
                    crate::builtins::BuiltinStageMerge::UsizeMin,
                ),
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
    fn terminal_collect_current_row_runner_stops_after_demand_is_met() {
        let source = CountingView::root(&[1, 2, 3]);
        let body = PipelineBody {
            stages: vec![
                Stage::Filter(
                    Arc::new(crate::vm::Program::new(Vec::new(), "")),
                    crate::builtins::BuiltinViewStage::Filter,
                ),
                Stage::Take(
                    1,
                    crate::builtins::BuiltinViewStage::Take,
                    crate::builtins::BuiltinStageMerge::UsizeMin,
                ),
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
                Stage::Take(
                    2,
                    crate::builtins::BuiltinViewStage::Take,
                    crate::builtins::BuiltinStageMerge::UsizeMin,
                ),
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
            sink: Sink::Reducer(crate::pipeline::ReducerSpec::count()),
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
            stages: vec![Stage::CountBy(Arc::new(crate::vm::Program::new(
                Vec::new(),
                "",
            )))],
            stage_exprs: Vec::new(),
            sink: Sink::Terminal(crate::builtins::BuiltinMethod::First),
            stage_kernels: vec![BodyKernel::Current],
            sink_kernels: Vec::new(),
        };

        let out =
            super::run_reducing_stage_prefix_then_materialized_suffix(source.clone(), &body, None)
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
            stages: vec![Stage::IndexBy(Arc::new(crate::vm::Program::new(
                Vec::new(),
                "",
            )))],
            stage_exprs: Vec::new(),
            sink: Sink::Terminal(crate::builtins::BuiltinMethod::First),
            stage_kernels: vec![BodyKernel::Current],
            sink_kernels: Vec::new(),
        };

        let out =
            super::run_reducing_stage_prefix_then_materialized_suffix(source.clone(), &body, None)
                .unwrap()
                .unwrap();

        let out_json: serde_json::Value = out.into();
        assert_eq!(out_json, serde_json::json!({"1": 1, "2": 2, "3": 3}));
        assert_eq!(source.scalar_reads(), 4);
        assert_eq!(source.materialize_reads(), 4);
    }
}
