//! View-backed execution for streamable pipeline bodies.

use std::sync::Arc;

use crate::chain_ir::PullDemand;
use crate::context::{Env, EvalError};
use crate::pipeline;
use crate::value::Val;
use crate::value_view::ValueView;

mod stage_flow;

use stage_flow::{ViewFrontierFlow, ViewStageFlow};

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

    let mut sink_acc = pipeline::SinkAccumulator::new(&body.sink);
    let mut op_state: Vec<usize> = vec![0; capabilities.stages.len()];
    let source_demand = pipeline::Pipeline::segment_source_demand(&body.stages, &body.sink)
        .chain
        .pull;
    let mut pulled_inputs = 0usize;
    let mut emitted_outputs = 0usize;

    'outer: for row in items {
        if matches!(source_demand, PullDemand::FirstInput(n) if pulled_inputs >= n) {
            break 'outer;
        }
        pulled_inputs += 1;

        let mut frontier = vec![row];
        let mut stop_after_frontier = false;
        for (op_idx, stage) in capabilities.stages.iter().enumerate() {
            match apply_view_stage_frontier(
                frontier,
                *stage,
                op_idx,
                &mut op_state,
                &body.stage_kernels,
            )? {
                ViewFrontierFlow::Keep(next) if next.is_empty() => continue 'outer,
                ViewFrontierFlow::Keep(next) => frontier = next,
                ViewFrontierFlow::Stop(next) if next.is_empty() => break 'outer,
                ViewFrontierFlow::Stop(next) => {
                    frontier = next;
                    stop_after_frontier = true;
                }
            }
        }

        for item in frontier {
            let sink_done = match capabilities.sink {
                pipeline::ViewSinkCapability::Collect => {
                    debug_assert_eq!(
                        capabilities.sink.materialization(),
                        pipeline::ViewMaterialization::SinkOutputRows
                    );
                    sink_acc.observe_collect(item.materialize());
                    false
                }
                pipeline::ViewSinkCapability::Count { predicate_kernel } => {
                    debug_assert_eq!(
                        capabilities.sink.materialization(),
                        pipeline::ViewMaterialization::Never
                    );
                    if !view_sink_predicate_matches(&item, predicate_kernel, &body.sink_kernels)? {
                        continue;
                    }
                    sink_acc.observe_count();
                    false
                }
                pipeline::ViewSinkCapability::Numeric {
                    op,
                    predicate_kernel,
                    project_kernel,
                } => {
                    let _ = op;
                    debug_assert_eq!(
                        capabilities.sink.materialization(),
                        pipeline::ViewMaterialization::SinkNumericInput
                    );
                    if !view_sink_predicate_matches(&item, predicate_kernel, &body.sink_kernels)? {
                        continue;
                    }
                    let numeric_item = if let Some(kernel) = project_kernel {
                        let kernel = body.sink_kernels.get(kernel)?;
                        eval_value_kernel(&item, kernel)?
                    } else {
                        item.materialize()
                    };
                    sink_acc.observe_numeric(&numeric_item);
                    false
                }
                pipeline::ViewSinkCapability::First => {
                    debug_assert_eq!(
                        capabilities.sink.materialization(),
                        pipeline::ViewMaterialization::SinkFinalRow
                    );
                    sink_acc.observe_first(item.materialize())
                }
                pipeline::ViewSinkCapability::Last => {
                    debug_assert_eq!(
                        capabilities.sink.materialization(),
                        pipeline::ViewMaterialization::SinkFinalRow
                    );
                    sink_acc.observe_last(item.materialize());
                    false
                }
            };

            if sink_done {
                break 'outer;
            }
            emitted_outputs += 1;
            if matches!(source_demand, PullDemand::UntilOutput(n) if emitted_outputs >= n) {
                break 'outer;
            }
        }
        if stop_after_frontier {
            break 'outer;
        }
    }

    Some(Ok(sink_acc.finish(false)))
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

    let items = source.array_iter()?;
    let mut boundary_rows = Vec::new();
    let mut op_state: Vec<usize> = vec![0; prefix.stages.len()];
    let source_demand = pipeline::Pipeline::segment_source_demand(&body.stages, &body.sink)
        .chain
        .pull;
    let mut pulled_inputs = 0usize;
    let mut emitted_outputs = 0usize;

    'outer: for row in items {
        if matches!(source_demand, PullDemand::FirstInput(n) if pulled_inputs >= n) {
            break 'outer;
        }
        pulled_inputs += 1;

        let mut frontier = vec![row];
        let mut stop_after_frontier = false;
        for (op_idx, stage) in prefix.stages.iter().enumerate() {
            match apply_view_stage_frontier(
                frontier,
                *stage,
                op_idx,
                &mut op_state,
                &body.stage_kernels,
            )? {
                ViewFrontierFlow::Keep(next) if next.is_empty() => continue 'outer,
                ViewFrontierFlow::Keep(next) => frontier = next,
                ViewFrontierFlow::Stop(next) if next.is_empty() => break 'outer,
                ViewFrontierFlow::Stop(next) => {
                    frontier = next;
                    stop_after_frontier = true;
                }
            }
        }
        for item in frontier {
            boundary_rows.push(item.materialize());
            emitted_outputs += 1;
            if matches!(source_demand, PullDemand::UntilOutput(n) if emitted_outputs >= n) {
                break 'outer;
            }
        }
        if stop_after_frontier {
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
    let (map_idx, map_kernel, prefix) = terminal_map_collect_plan(body)?;
    let items = source.array_iter()?;
    let mut collector = pipeline::TerminalMapCollector::new(map_kernel);
    let mut op_state: Vec<usize> = vec![0; map_idx];

    'outer: for row in items {
        let mut item = row;
        for (op_idx, capability) in prefix.iter().copied().enumerate() {
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

fn terminal_map_collect_plan(
    body: &pipeline::PipelineBody,
) -> Option<(
    usize,
    &pipeline::BodyKernel,
    Vec<pipeline::ViewStageCapability>,
)> {
    if !matches!(
        body.sink.view_capability(&body.sink_kernels)?,
        pipeline::ViewSinkCapability::Collect
    ) {
        return None;
    }
    let mut prefix = Vec::new();
    let mut terminal_map: Option<(usize, &pipeline::BodyKernel)> = None;

    for (idx, stage) in body.stages.iter().enumerate() {
        let capability = stage.view_capability(idx, body.stage_kernels.get(idx))?;
        if idx + 1 == body.stages.len() {
            let pipeline::ViewStageCapability::Map { kernel } = capability else {
                return None;
            };
            terminal_map = Some((idx, body.stage_kernels.get(kernel)?));
        } else {
            prefix.push(capability);
        }
    }

    let (map_idx, map_kernel) = terminal_map?;
    map_kernel
        .is_view_native()
        .then_some((map_idx, map_kernel, prefix))
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
    let strategies =
        pipeline::compute_strategies_with_kernels(&body.stages, &body.stage_kernels, &body.sink);
    let strategy = strategies.first().copied()?;
    if !matches!(
        strategy,
        pipeline::StageStrategy::SortTopK(_) | pipeline::StageStrategy::SortBottomK(_)
    ) {
        return None;
    }
    if !body.suffix_can_run_with_materialized_receiver(1) {
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
    stage_flow::apply_stage(item, stage, op_idx, op_state, stage_kernels)
}

fn apply_view_stage_frontier<'a, V>(
    frontier: Vec<V>,
    stage: pipeline::ViewStageCapability,
    op_idx: usize,
    op_state: &mut [usize],
    stage_kernels: &[pipeline::BodyKernel],
) -> Option<ViewFrontierFlow<V>>
where
    V: ValueView<'a>,
{
    stage_flow::apply_frontier(frontier, stage, op_idx, op_state, stage_kernels)
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
}
