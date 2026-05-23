//! Legacy per-shape pipeline execution path.
//!
//! Executes pipeline plans that have not been promoted to the composed or
//! columnar paths. Still the hot path for many common shapes; kept separate
//! from `composed_exec` so migration to the composed substrate can proceed
//! incrementally without breaking existing correctness.

use std::borrow::Cow;
use std::sync::Arc;

use crate::{
    data::context::{Env, EvalError},
    data::value::Val,
    vm::VM,
};

use super::nested::PreparedPlan;
use super::row_source;
use super::sink_accumulator::SinkAccumulator;
use super::{
    apply_item_in_env, cmp_val_total, compute_strategies_with_kernels, eval_kernel_with_vm,
    is_truthy, BodyKernel, Pipeline, PipelineBody, Sink, Source, SourceAccessMode, Stage,
    StageFlow, StageStrategy, TerminalMapCollector,
};

use crate::builtins::registry::{
    keyed_reducer, object_lambda as builtin_object_lambda,
    string_pair_stage as builtin_string_pair_stage, BuiltinId,
};
use crate::builtins::{
    replace_apply, slice_apply, split_apply, BuiltinMembershipSink, BuiltinMethod,
    BuiltinObjectLambda, BuiltinStringPairStage,
};
use crate::plan::demand::PullDemand;

/// Runs the pipeline against `root`, materialising barrier stages then streaming the rest.
pub(super) fn run(
    pipeline: &Pipeline,
    root: &Val,
    base_env: &Env,
    vm: &mut VM,
) -> Result<Val, EvalError> {
    let mut loop_env = base_env.clone();

    let recv = row_source::resolve(&pipeline.source, root);

    let source_demand = pipeline.source_demand().chain.pull;
    let mut pulled_inputs: usize = 0;
    let mut emitted_outputs: usize = 0;

    let mut sink_acc = SinkAccumulator::new(&pipeline.sink);
    let membership_target = match &pipeline.sink {
        Sink::Membership(spec) => Some(eval_membership_target(spec, vm, &loop_env)?),
        _ => None,
    };
    if let Sink::Membership(spec) = &pipeline.sink {
        if pipeline.stages.is_empty() && row_source::array_like_rows(&recv).is_none() {
            return Ok(apply_membership_scalar_sink(
                spec,
                membership_target
                    .as_ref()
                    .expect("membership target exists"),
                &recv,
            ));
        }
    }

    let needs_barrier = pipeline
        .stages
        .iter()
        .any(Stage::requires_legacy_materialization);
    if !needs_barrier {
        let planned = planned_stream_for_access(pipeline);
        let out = run_streaming_rows_with_vm(
            planned.pipeline.as_ref(),
            base_env,
            row_source::source_iter_for_access(&recv, planned.access),
            vm,
        )?;
        return Ok(planned.restore(out));
    }

    let pre_iter: LegacyPreIter = {
        let mut buf: Vec<Val> = match source_demand {
            PullDemand::FirstInput(n) => row_source::materialize_source_prefix(&recv, n),
            _ => row_source::materialize_source(&recv),
        };
        let strategies = compute_strategies_with_kernels(
            &pipeline.stages,
            &pipeline.stage_kernels,
            &pipeline.sink,
        );
        for (stage_idx, stage) in pipeline.stages.iter().enumerate() {
            let kernel = pipeline
                .stage_kernels
                .get(stage_idx)
                .unwrap_or(&BodyKernel::Generic);
            let strategy = strategies
                .get(stage_idx)
                .copied()
                .unwrap_or(StageStrategy::Default);
            if let Stage::CompiledMap(plan) = stage {
                let prepared = PreparedPlan::new(plan);
                let mut out: Vec<Val> = Vec::with_capacity(buf.len());
                for v in buf.into_iter() {
                    out.push(prepared.run(v)?);
                }
                buf = out;
                continue;
            }

            if let Some(applied) =
                apply_adapter_materialized(stage, &mut buf, vm, &mut loop_env, kernel, strategy)
            {
                applied?;
                continue;
            }
            unreachable!("descriptor-backed stage was not handled by materialized adapter")
        }
        LegacyPreIter::Owned(buf.into_iter())
    };

    'outer: for item in pre_iter {
        if source_demand.input_satisfied_by(pulled_inputs) {
            break 'outer;
        }
        pulled_inputs += 1;

        let sink_done = match &pipeline.sink {
            Sink::Predicate(_) => {
                observe_predicate_sink_item(pipeline, item, &mut sink_acc, vm, &mut loop_env)?
            }
            Sink::Membership(spec) => sink_acc.observe_membership(
                spec.op,
                &item,
                membership_target
                    .as_ref()
                    .expect("membership target exists"),
            ),
            Sink::ArgExtreme(_) => {
                observe_arg_extreme_sink_item(pipeline, item, &mut sink_acc, vm, &mut loop_env)?
            }
            Sink::Reducer(_) => {
                match observe_reducer_item(pipeline, item, &mut sink_acc, vm, &mut loop_env)? {
                    ReducerItemFlow::Observed => false,
                    ReducerItemFlow::Skipped => continue 'outer,
                }
            }
            _ => sink_acc.push(item),
        };
        if sink_done {
            break 'outer;
        }
        emitted_outputs += 1;
        if source_demand.output_satisfied_by(emitted_outputs) {
            break 'outer;
        }
    }

    // Keyed reducers wrap their output in a single-element array; unwrap it so
    // terminal collection returns the reducer object.
    let unwrap_single_collect_obj = pipeline
        .stages
        .last()
        .and_then(Stage::descriptor)
        .is_some_and(|desc| {
            desc.method
                .is_some_and(|method| keyed_reducer(BuiltinId::from_method(method)).is_some())
        });
    sink_acc.finish_result(unwrap_single_collect_obj)
}

/// Streams a pipeline directly from a `simd-json` tape using caller-owned VM state.
pub(super) fn run_tape_field_chain_with_vm(
    body: &PipelineBody,
    tape: &crate::data::tape::TapeData,
    keys: &[Arc<str>],
    base_env: &Env,
    vm: &mut VM,
) -> Option<Result<Val, EvalError>> {
    if body
        .stages
        .iter()
        .any(Stage::requires_legacy_materialization)
    {
        return None;
    }
    if !body.can_run_with_materialized_receiver() {
        return None;
    }
    let source = row_source::TapeRowSource::from_field_chain(tape, keys);
    if !source.is_array_provider() {
        return None;
    }
    let pipeline = body.clone().with_source(Source::Receiver(Val::Null));
    let planned = planned_stream_for_access(&pipeline);
    let iter = source.iter_materialized_for_access(planned.access);
    Some(
        run_streaming_rows_with_vm(planned.pipeline.as_ref(), base_env, iter, vm)
            .map(|out| planned.restore(out)),
    )
}

#[cfg(test)]
fn run_streaming_rows<I>(pipeline: &Pipeline, base_env: &Env, iter: I) -> Result<Val, EvalError>
where
    I: IntoIterator<Item = Val>,
{
    let mut vm = VM::new();
    run_streaming_rows_with_vm(pipeline, base_env, iter, &mut vm)
}

fn run_streaming_rows_with_vm<I>(
    pipeline: &Pipeline,
    base_env: &Env,
    iter: I,
    vm: &mut VM,
) -> Result<Val, EvalError>
where
    I: IntoIterator<Item = Val>,
{
    let mut loop_env = base_env.clone();
    let source_demand = pipeline.source_demand().chain.pull;
    let late_projection = pipeline
        .can_apply_late_projection_from(0)
        .then(|| pipeline.late_projection.as_ref())
        .flatten()
        .filter(|_| pipeline.sink.supports_late_projection(source_demand));
    let stage_limit = late_projection
        .map(|projection| projection.prefix_len)
        .unwrap_or(pipeline.stages.len());
    let mut pulled_inputs: usize = 0;
    let mut emitted_outputs: usize = 0;
    let mut stage_taken: Vec<usize> = vec![0; pipeline.stages.len()];
    let mut stage_skipped: Vec<usize> = vec![0; pipeline.stages.len()];
    let mut sink_acc = SinkAccumulator::new(&pipeline.sink);
    let membership_target = match &pipeline.sink {
        Sink::Membership(spec) => Some(eval_membership_target(spec, vm, &loop_env)?),
        _ => None,
    };
    if source_demand.is_zero() {
        return sink_acc.finish_result(false);
    }
    let terminal_map_idx = if late_projection.is_none()
        && matches!(pipeline.sink, Sink::Collect)
        && pipeline
            .stages
            .last()
            .is_some_and(Stage::can_use_terminal_map_collector)
    {
        pipeline.stages.len().checked_sub(1)
    } else {
        None
    };
    let terminal_map_kernel = terminal_map_idx.map(|idx| {
        pipeline
            .stage_kernels
            .get(idx)
            .unwrap_or(&BodyKernel::Generic)
    });
    let mut terminal_map_collect = terminal_map_kernel.map(TerminalMapCollector::new);
    let prepared_nested: Vec<Option<PreparedPlan>> = pipeline
        .stages
        .iter()
        .map(|stage| match stage {
            Stage::CompiledMap(plan) => Some(PreparedPlan::new(plan)),
            _ => None,
        })
        .collect();

    'outer: for mut item in iter {
        if source_demand.input_satisfied_by(pulled_inputs) {
            break 'outer;
        }
        if source_demand.skips_before_nth(pulled_inputs) {
            pulled_inputs += 1;
            continue 'outer;
        }
        pulled_inputs += 1;

        for (stage_idx, stage) in pipeline.stages[..stage_limit].iter().enumerate() {
            let kernel = pipeline
                .stage_kernels
                .get(stage_idx)
                .unwrap_or(&BodyKernel::Generic);
            match stage {
                Stage::CompiledMap(_) => {
                    item = prepared_nested[stage_idx]
                        .as_ref()
                        .expect("compiled map stages have prepared nested plans")
                        .run(item)?;
                }
                _ => match super::val_stage_flow::apply_adapter_streaming(
                    stage,
                    stage_idx,
                    item,
                    vm,
                    &mut loop_env,
                    kernel,
                    &mut stage_taken,
                    &mut stage_skipped,
                    terminal_map_idx,
                    &mut terminal_map_collect,
                )? {
                    StageFlow::Continue(next) => item = next,
                    StageFlow::SkipRow => continue 'outer,
                    StageFlow::Stop => break 'outer,
                    StageFlow::TerminalCollected => {
                        emitted_outputs += 1;
                        if source_demand.output_satisfied_by(emitted_outputs) {
                            break 'outer;
                        }
                        continue 'outer;
                    }
                },
            }
        }

        if source_demand.is_nth_input() && matches!(pipeline.sink, Sink::Nth(_)) {
            if let Some(projection) = late_projection {
                return eval_late_projection(&projection.kernel, &item, vm);
            }
            return Ok(item);
        }

        if let Some(projection) = late_projection {
            item = eval_late_projection(&projection.kernel, &item, vm)?;
        }

        let sink_done = match &pipeline.sink {
            Sink::Predicate(_) => {
                observe_predicate_sink_item(pipeline, item, &mut sink_acc, vm, &mut loop_env)?
            }
            Sink::Membership(spec) => sink_acc.observe_membership(
                spec.op,
                &item,
                membership_target
                    .as_ref()
                    .expect("membership target exists"),
            ),
            Sink::ArgExtreme(_) => {
                observe_arg_extreme_sink_item(pipeline, item, &mut sink_acc, vm, &mut loop_env)?
            }
            Sink::Reducer(_) => {
                match observe_reducer_item(pipeline, item, &mut sink_acc, vm, &mut loop_env)? {
                    ReducerItemFlow::Observed => false,
                    ReducerItemFlow::Skipped => continue 'outer,
                }
            }
            _ => sink_acc.push(item),
        };
        if sink_done {
            break 'outer;
        }
        emitted_outputs += 1;
        if source_demand.output_satisfied_by(emitted_outputs) {
            break 'outer;
        }
    }

    if let Some(collector) = terminal_map_collect {
        return Ok(collector.finish());
    }
    sink_acc.finish_result(false)
}

fn restore_reversed_select_many_result(value: Val) -> Val {
    match value {
        Val::Arr(items) => {
            let mut items = Arc::try_unwrap(items).unwrap_or_else(|items| items.as_ref().clone());
            items.reverse();
            Val::arr(items)
        }
        other => other,
    }
}

struct PlannedStream<'a> {
    pipeline: Cow<'a, Pipeline>,
    access: SourceAccessMode,
    restore_reversed_select_many: bool,
}

impl PlannedStream<'_> {
    fn restore(&self, value: Val) -> Val {
        if self.restore_reversed_select_many {
            restore_reversed_select_many_result(value)
        } else {
            value
        }
    }
}

fn planned_stream_for_access(pipeline: &Pipeline) -> PlannedStream<'_> {
    let access = pipeline.source_access();
    if pipeline.source_demand().chain.pull.is_nth_input()
        && matches!(access, SourceAccessMode::Indexed(_))
    {
        if let Some(selected) = pipeline.for_selected_single_row() {
            return PlannedStream {
                pipeline: Cow::Owned(selected),
                access,
                restore_reversed_select_many: false,
            };
        }
    }

    if matches!(access, SourceAccessMode::Reverse { .. }) {
        if let Some(reversed) = pipeline.for_reversed_select_one() {
            return PlannedStream {
                pipeline: Cow::Owned(reversed),
                access,
                restore_reversed_select_many: false,
            };
        }
        if let Some(reversed) = pipeline.for_reversed_select_many() {
            return PlannedStream {
                pipeline: Cow::Owned(reversed),
                access,
                restore_reversed_select_many: true,
            };
        }
        return PlannedStream {
            pipeline: Cow::Borrowed(pipeline),
            access: SourceAccessMode::Forward,
            restore_reversed_select_many: false,
        };
    }

    PlannedStream {
        pipeline: Cow::Borrowed(pipeline),
        access,
        restore_reversed_select_many: false,
    }
}

fn eval_late_projection(
    projection: &BodyKernel,
    item: &Val,
    vm: &mut crate::vm::VM,
) -> Result<Val, EvalError> {
    eval_kernel_with_vm(projection, item, vm, |_, _| {
        Err(EvalError(
            "late projection requires a native body kernel".to_string(),
        ))
    })
}

// barrier stages always produce a Vec<Val>, so only the Owned variant is needed here
enum LegacyPreIter {
    Owned(std::vec::IntoIter<Val>),
}

// returns None for unrecognised stage types so the caller can unreachable!()
fn apply_adapter_materialized(
    stage: &Stage,
    buf: &mut Vec<Val>,
    vm: &mut crate::vm::VM,
    loop_env: &mut Env,
    kernel: &BodyKernel,
    strategy: StageStrategy,
) -> Option<Result<(), EvalError>> {
    // Trait dispatch for migrated barrier methods.
    if let Some(method) = stage.descriptor().and_then(|d| d.method) {
        let body = stage.body_program();
        let mut ctx = crate::builtins::builtin::BarrierCtx {
            vm,
            env: loop_env,
            kernel,
            stage,
            strategy,
        };
        if let Some(r) = crate::builtins::registry::apply_barrier_hook(method, &mut ctx, buf, body)
        {
            return Some(r);
        }
    }
    // Remaining barrier dispatch by Stage variant — all other variants are handled
    // above by Builtin::apply_barrier trait dispatch and never reach this point.
    match stage {
        // Element-wise scalar (Slice, Replace, ReplaceAll, BuiltinCall::apply).
        Stage::Builtin(_) | Stage::IntRangeBuiltin { .. } | Stage::StringPairBuiltin { .. } => {
            let mut out: Vec<Val> = Vec::with_capacity(buf.len());
            for v in std::mem::take(buf) {
                out.push(apply_element_adapter(stage, v));
            }
            *buf = out;
            Some(Ok(()))
        }
        // Expanding scalar (Split).
        Stage::StringBuiltin { .. } => {
            let mut out: Vec<Val> = Vec::with_capacity(buf.len());
            for v in std::mem::take(buf) {
                apply_expanding_adapter(stage, &v, &mut out);
            }
            *buf = out;
            Some(Ok(()))
        }
        // Sorted-dedup barrier — pre-sorted dedup, optionally keyed.
        Stage::SortedDedup(opt_prog) => {
            match opt_prog {
                None => {
                    buf.sort_by(cmp_val_total);
                    buf.dedup_by(|a, b| crate::util::vals_eq(a, b));
                }
                Some(prog) => {
                    let mut keyed: Vec<(Val, Val)> = Vec::with_capacity(buf.len());
                    for v in buf.iter() {
                        let key = match eval_kernel_with_vm(kernel, v, vm, |item, vm| {
                            apply_item_in_env(vm, loop_env, item, prog)
                        }) {
                            Ok(key) => key,
                            Err(err) => return Some(Err(err)),
                        };
                        keyed.push((key, v.clone()));
                    }
                    keyed.sort_by(|a, b| cmp_val_total(&a.0, &b.0));
                    keyed.dedup_by(|a, b| crate::util::vals_eq(&a.0, &b.0));
                    *buf = keyed.into_iter().map(|(_, v)| v).collect();
                }
            }
            Some(Ok(()))
        }
        // All other variants handled above by trait dispatch — unreachable.
        _ => None,
    }
}

/// Applies an element-wise stage (`Slice`, string pair builtins, `Builtin`) to a single `Val` row.
pub(super) fn apply_element_adapter(stage: &Stage, v: Val) -> Val {
    match stage {
        Stage::IntRangeBuiltin {
            method: BuiltinMethod::Slice,
            start,
            end,
        } => slice_apply(v, *start, *end),
        Stage::StringPairBuiltin {
            method,
            first,
            second,
        } => match builtin_string_pair_stage(BuiltinId::from_method(*method)) {
            Some(BuiltinStringPairStage::Replace { all }) => {
                replace_apply(v.clone(), first, second, all).unwrap_or(v)
            }
            None => v,
        },
        Stage::Builtin(call) => call.apply(&v).unwrap_or(v),
        _ => v,
    }
}

fn apply_expanding_adapter(stage: &Stage, v: &Val, out: &mut Vec<Val>) {
    if let Stage::StringBuiltin {
        method: BuiltinMethod::Split,
        value,
    } = stage
    {
        if let Some(Val::Arr(a)) = split_apply(v, value.as_ref()) {
            out.extend(Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone()));
        }
    }
}

impl Iterator for LegacyPreIter {
    type Item = Val;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Owned(iter) => iter.next(),
        }
    }
}

enum ReducerItemFlow {
    Observed,
    Skipped,
}

fn observe_reducer_item(
    pipeline: &Pipeline,
    item: Val,
    sink_acc: &mut SinkAccumulator<'_>,
    vm: &mut crate::vm::VM,
    loop_env: &mut Env,
) -> Result<ReducerItemFlow, EvalError> {
    let Sink::Reducer(spec) = &pipeline.sink else {
        sink_acc.push(item);
        return Ok(ReducerItemFlow::Observed);
    };

    if let Some(predicate) = &spec.predicate {
        let kernel_idx = spec.predicate_kernel_index().expect("predicate exists");
        let kernel = pipeline
            .sink_kernels
            .get(kernel_idx)
            .unwrap_or(&BodyKernel::Generic);
        let keep = eval_kernel_with_vm(kernel, &item, vm, |item, vm| {
            apply_item_in_env(vm, loop_env, item, predicate)
        })?;
        if !crate::util::is_truthy(&keep) {
            return Ok(ReducerItemFlow::Skipped);
        }
    }

    if let Some(project) = &spec.projection {
        let project_kernel_idx = spec.projection_kernel_index().expect("projection exists");
        let kernel = pipeline
            .sink_kernels
            .get(project_kernel_idx)
            .unwrap_or(&BodyKernel::Generic);
        let reducer_item = eval_kernel_with_vm(kernel, &item, vm, |item, vm| {
            apply_item_in_env(vm, loop_env, item, project)
        })?;
        sink_acc.push_projected_numeric(&reducer_item);
    } else {
        sink_acc.push(item);
    }

    Ok(ReducerItemFlow::Observed)
}

fn eval_membership_target(
    spec: &super::MembershipSinkSpec,
    vm: &mut crate::vm::VM,
    env: &Env,
) -> Result<Val, EvalError> {
    match &spec.target {
        super::MembershipSinkTarget::Literal(value) => Ok(value.clone()),
        super::MembershipSinkTarget::Program(program) => vm.exec_in_env(program, env),
    }
}

fn apply_membership_scalar_sink(spec: &super::MembershipSinkSpec, target: &Val, recv: &Val) -> Val {
    match spec.op {
        BuiltinMembershipSink::Includes => crate::builtins::includes_apply(recv, target),
        BuiltinMembershipSink::Index => {
            crate::builtins::index_value_apply(recv, target).unwrap_or(Val::Null)
        }
        BuiltinMembershipSink::IndicesOf => {
            crate::builtins::indices_of_apply(recv, target).unwrap_or(Val::Null)
        }
    }
}

fn observe_predicate_sink_item(
    pipeline: &Pipeline,
    item: Val,
    sink_acc: &mut SinkAccumulator<'_>,
    vm: &mut crate::vm::VM,
    loop_env: &mut Env,
) -> Result<bool, EvalError> {
    let Sink::Predicate(spec) = &pipeline.sink else {
        return Ok(sink_acc.push(item));
    };

    let kernel_idx = spec.predicate_kernel_index();
    let kernel = pipeline
        .sink_kernels
        .get(kernel_idx)
        .unwrap_or(&BodyKernel::Generic);
    let predicate = eval_kernel_with_vm(kernel, &item, vm, |item, vm| {
        apply_item_in_env(vm, loop_env, item, &spec.predicate)
    })?;
    sink_acc.observe_predicate_item(spec.op, crate::util::is_truthy(&predicate), item)
}

fn observe_arg_extreme_sink_item(
    pipeline: &Pipeline,
    item: Val,
    sink_acc: &mut SinkAccumulator<'_>,
    vm: &mut crate::vm::VM,
    loop_env: &mut Env,
) -> Result<bool, EvalError> {
    let Sink::ArgExtreme(spec) = &pipeline.sink else {
        return Ok(sink_acc.push(item));
    };

    let kernel_idx = spec.key_kernel_index();
    let kernel = pipeline
        .sink_kernels
        .get(kernel_idx)
        .unwrap_or(&BodyKernel::Generic);
    let key = eval_kernel_with_vm(kernel, &item, vm, |item, vm| {
        apply_item_in_env(vm, loop_env, item, &spec.key)
    })?;
    sink_acc.observe_arg_extreme(spec.want_max, item, key);
    Ok(false)
}

/// Applies an object-lambda stage (`TransformKeys`, `TransformValues`, `FilterKeys`, `FilterValues`) to `recv`.
pub(crate) fn apply_lambda_obj(
    stage: &Stage,
    recv: &Val,
    vm: &mut crate::vm::VM,
    loop_env: &mut crate::data::context::Env,
    kernel: &BodyKernel,
    prog: &crate::vm::Program,
) -> Result<Val, EvalError> {
    let m = match recv.as_object() {
        Some(m) => m,
        None => return Ok(recv.clone()),
    };
    let mut out: indexmap::IndexMap<std::sync::Arc<str>, Val> =
        indexmap::IndexMap::with_capacity(m.len());
    let operation = match stage {
        Stage::ExprBuiltin { method, .. } => builtin_object_lambda(BuiltinId::from_method(*method)),
        _ => None,
    }
    .expect("apply_lambda_obj called with non-Obj-lambda Stage");

    for (k, v) in m.iter() {
        match operation {
            BuiltinObjectLambda::TransformKeys => {
                let k_val = Val::Str(k.clone());
                let new_k = eval_kernel_with_vm(kernel, &k_val, vm, |item, vm| {
                    apply_item_in_env(vm, loop_env, item, prog)
                })?;
                let new_k_arc = match new_k {
                    Val::Str(s) => s,
                    other => std::sync::Arc::from(crate::util::val_to_string(&other).as_str()),
                };
                out.insert(new_k_arc, v.clone());
            }
            BuiltinObjectLambda::TransformValues => {
                let new_v = eval_kernel_with_vm(kernel, v, vm, |item, vm| {
                    apply_item_in_env(vm, loop_env, item, prog)
                })?;
                out.insert(k.clone(), new_v);
            }
            BuiltinObjectLambda::FilterKeys => {
                let k_val = Val::Str(k.clone());
                if is_truthy(&eval_kernel_with_vm(kernel, &k_val, vm, |item, vm| {
                    apply_item_in_env(vm, loop_env, item, prog)
                })?) {
                    out.insert(k.clone(), v.clone());
                }
            }
            BuiltinObjectLambda::FilterValues => {
                if is_truthy(&eval_kernel_with_vm(kernel, v, vm, |item, vm| {
                    apply_item_in_env(vm, loop_env, item, prog)
                })?) {
                    out.insert(k.clone(), v.clone());
                }
            }
        }
    }
    Ok(Val::obj(out))
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::rc::Rc;
    use std::sync::Arc;

    use crate::data::context::Env;
    use crate::data::value::Val;
    use crate::parse::ast::BinOp;
    use crate::builtins::{BuiltinMembershipSink, BuiltinPredicateSink};

    use super::super::{
        BodyKernel, MembershipSinkSpec, MembershipSinkTarget, PipelineBody, PredicateSinkSpec,
        Sink, Source,
    };

    struct CountingRows {
        next: i64,
        end: i64,
        reads: Rc<Cell<usize>>,
    }

    impl CountingRows {
        fn new(end: i64, reads: Rc<Cell<usize>>) -> Self {
            Self {
                next: 1,
                end,
                reads,
            }
        }
    }

    impl Iterator for CountingRows {
        type Item = Val;

        fn next(&mut self) -> Option<Self::Item> {
            if self.next > self.end {
                return None;
            }
            self.reads.set(self.reads.get() + 1);
            let value = self.next;
            self.next += 1;
            Some(Val::Int(value))
        }
    }

    fn empty_pipeline(sink: Sink, sink_kernels: Vec<BodyKernel>) -> super::Pipeline {
        PipelineBody {
            stages: Vec::new(),
            stage_exprs: Vec::new(),
            sink,
            stage_kernels: Vec::new(),
            sink_kernels,
        }
        .with_source(Source::Receiver(Val::Null))
    }

    #[test]
    fn materialized_streaming_stops_when_any_sink_matches() {
        let reads = Rc::new(Cell::new(0));
        let pipeline = empty_pipeline(
            Sink::Predicate(PredicateSinkSpec {
                op: BuiltinPredicateSink::Any,
                predicate: Arc::new(crate::vm::Program::new(Vec::new(), "")),
                predicate_expr: None,
            }),
            vec![BodyKernel::CurrentCmpLit(BinOp::Gt, Val::Int(2))],
        );
        let env = Env::new(Val::Null);

        let out = super::run_streaming_rows(&pipeline, &env, CountingRows::new(8, reads.clone()))
            .unwrap();

        assert_eq!(out, Val::Bool(true));
        assert_eq!(reads.get(), 3);
    }

    #[test]
    fn materialized_streaming_stops_when_all_sink_fails() {
        let reads = Rc::new(Cell::new(0));
        let pipeline = empty_pipeline(
            Sink::Predicate(PredicateSinkSpec {
                op: BuiltinPredicateSink::All,
                predicate: Arc::new(crate::vm::Program::new(Vec::new(), "")),
                predicate_expr: None,
            }),
            vec![BodyKernel::CurrentCmpLit(BinOp::Lt, Val::Int(3))],
        );
        let env = Env::new(Val::Null);

        let out = super::run_streaming_rows(&pipeline, &env, CountingRows::new(8, reads.clone()))
            .unwrap();

        assert_eq!(out, Val::Bool(false));
        assert_eq!(reads.get(), 3);
    }

    #[test]
    fn materialized_streaming_stops_when_includes_sink_matches() {
        let reads = Rc::new(Cell::new(0));
        let pipeline = empty_pipeline(
            Sink::Membership(MembershipSinkSpec {
                op: BuiltinMembershipSink::Includes,
                target: MembershipSinkTarget::Literal(Val::Int(3)),
            }),
            Vec::new(),
        );
        let env = Env::new(Val::Null);

        let out = super::run_streaming_rows(&pipeline, &env, CountingRows::new(8, reads.clone()))
            .unwrap();

        assert_eq!(out, Val::Bool(true));
        assert_eq!(reads.get(), 3);
    }

    #[test]
    fn materialized_streaming_stops_when_index_sink_matches() {
        let reads = Rc::new(Cell::new(0));
        let pipeline = empty_pipeline(
            Sink::Membership(MembershipSinkSpec {
                op: BuiltinMembershipSink::Index,
                target: MembershipSinkTarget::Literal(Val::Int(3)),
            }),
            Vec::new(),
        );
        let env = Env::new(Val::Null);

        let out = super::run_streaming_rows(&pipeline, &env, CountingRows::new(8, reads.clone()))
            .unwrap();

        assert_eq!(out, Val::Int(2));
        assert_eq!(reads.get(), 3);
    }

    #[test]
    fn tape_row_bridge_applies_indexed_nth_demand_before_materializing() {
        let expr = crate::parse::parser::parse("$.books.map(score + 1).nth(2)").unwrap();
        let pipeline = super::super::Pipeline::lower(&expr).expect("pipeline lower");
        let (_, body) = pipeline.into_source_body();
        let tape = crate::data::tape::TapeData::parse(
            br#"{"books":[{"score":1},{"score":2},{"score":3},{"score":4}]}"#.to_vec(),
        )
        .unwrap();
        tape.reset_materialized_subtrees();
        let mut vm = crate::vm::VM::new();
        let keys = [Arc::<str>::from("books")];

        let out =
            super::run_tape_field_chain_with_vm(&body, &tape, &keys, &Env::new(Val::Null), &mut vm)
                .expect("tape rows path")
                .unwrap();

        assert_eq!(out, Val::Int(4));
        assert_eq!(tape.materialized_subtrees(), 1);
    }

    #[test]
    fn tape_row_bridge_applies_indexed_suffix_before_materializing() {
        let expr = crate::parse::parser::parse("$.books.map(score + 1).last(2)").unwrap();
        let pipeline = super::super::Pipeline::lower(&expr).expect("pipeline lower");
        let (_, body) = pipeline.into_source_body();
        let tape = crate::data::tape::TapeData::parse(
            br#"{"books":[{"score":1},{"score":2},{"score":3},{"score":4}]}"#.to_vec(),
        )
        .unwrap();
        tape.reset_materialized_subtrees();
        let mut vm = crate::vm::VM::new();
        let keys = [Arc::<str>::from("books")];

        let out =
            super::run_tape_field_chain_with_vm(&body, &tape, &keys, &Env::new(Val::Null), &mut vm)
                .expect("tape rows path")
                .unwrap();

        assert_eq!(serde_json::Value::from(out), serde_json::json!([4, 5]));
        assert_eq!(tape.materialized_subtrees(), 2);
    }

    #[test]
    fn tape_row_bridge_scans_reverse_for_selective_last() {
        let expr =
            crate::parse::parser::parse("$.books.filter(active).map(score + 1).last()").unwrap();
        let pipeline = super::super::Pipeline::lower(&expr).expect("pipeline lower");
        let (_, body) = pipeline.into_source_body();
        let tape = crate::data::tape::TapeData::parse(
            br#"{"books":[{"score":1,"active":true},{"score":2,"active":true},{"score":3,"active":false},{"score":4,"active":true}]}"#.to_vec(),
        )
        .unwrap();
        tape.reset_materialized_subtrees();
        let mut vm = crate::vm::VM::new();
        let keys = [Arc::<str>::from("books")];

        let out =
            super::run_tape_field_chain_with_vm(&body, &tape, &keys, &Env::new(Val::Null), &mut vm)
                .expect("tape rows path")
                .unwrap();

        assert_eq!(out, Val::Int(5));
        assert_eq!(tape.materialized_subtrees(), 1);
    }

    #[test]
    fn materialized_reverse_access_preserves_select_many_order() {
        let expr =
            crate::parse::parser::parse("$.books.filter(active).map(score).last(2)").unwrap();
        let pipeline = super::super::Pipeline::lower(&expr).expect("pipeline lower");
        let root_json = serde_json::json!({
            "books": [
                {"score": 1, "active": true},
                {"score": 2, "active": true},
                {"score": 3, "active": false},
                {"score": 4, "active": true}
            ]
        });
        let root = Val::from(&root_json);
        let mut vm = crate::vm::VM::new();

        let out = super::run(&pipeline, &root, &Env::new(root.clone()), &mut vm).unwrap();

        assert_eq!(serde_json::Value::from(out), serde_json::json!([2, 4]));
    }

    #[test]
    fn tape_row_bridge_preserves_reversed_select_many_order() {
        let expr =
            crate::parse::parser::parse("$.books.filter(active).map(score).last(2)").unwrap();
        let pipeline = super::super::Pipeline::lower(&expr).expect("pipeline lower");
        let (_, body) = pipeline.into_source_body();
        let tape = crate::data::tape::TapeData::parse(
            br#"{"books":[{"score":1,"active":true},{"score":2,"active":true},{"score":3,"active":false},{"score":4,"active":true}]}"#.to_vec(),
        )
        .unwrap();
        tape.reset_materialized_subtrees();
        let mut vm = crate::vm::VM::new();
        let keys = [Arc::<str>::from("books")];

        let out =
            super::run_tape_field_chain_with_vm(&body, &tape, &keys, &Env::new(Val::Null), &mut vm)
                .expect("tape rows path")
                .unwrap();

        assert_eq!(serde_json::Value::from(out), serde_json::json!([2, 4]));
        assert_eq!(tape.materialized_subtrees(), 3);
    }
}
