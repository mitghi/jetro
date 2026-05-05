//! Legacy per-shape pipeline execution path.
//!
//! Executes pipeline plans that have not been promoted to the composed or
//! columnar paths. Still the hot path for many common shapes; kept separate
//! from `composed_exec` so migration to the composed substrate can proceed
//! incrementally without breaking existing correctness.

use std::sync::Arc;

use crate::{
    context::{Env, EvalError},
    value::Val,
};

use super::lower::run_compiled_map;
use super::row_source;
use super::sink_accumulator::SinkAccumulator;
use super::{
    apply_item_in_env, cmp_val_total, compute_strategies_with_kernels, eval_kernel, is_truthy,
    stage_executor, BodyKernel, Pipeline, PipelineBody, Sink, Source, Stage, StageFlow,
    StageStrategy, TerminalMapCollector,
};

use crate::builtins::{replace_apply, slice_apply, split_apply, BuiltinMethod, BuiltinPipelineExecutor};
use crate::chain_ir::PullDemand;

/// Runs the pipeline against `root`, materialising barrier stages then streaming the rest.
pub(super) fn run(pipeline: &Pipeline, root: &Val, base_env: &Env) -> Result<Val, EvalError> {
    let mut vm = crate::vm::VM::new();
    let mut loop_env = base_env.clone();

    let recv = row_source::resolve(&pipeline.source, root);

    let source_demand = pipeline.source_demand().chain.pull;
    let mut pulled_inputs: usize = 0;
    let mut emitted_outputs: usize = 0;

    let mut sink_acc = SinkAccumulator::new(&pipeline.sink);

    let needs_barrier = pipeline
        .stages
        .iter()
        .any(Stage::requires_legacy_materialization);
    if !needs_barrier {
        return run_streaming_rows(pipeline, base_env, row_source::source_iter(&recv));
    }

    let pre_iter: LegacyPreIter = {
        let mut buf: Vec<Val> = row_source::materialize_source(&recv);
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
                let mut out: Vec<Val> = Vec::with_capacity(buf.len());
                for v in buf.into_iter() {
                    out.push(run_compiled_map(plan, v)?);
                }
                buf = out;
                continue;
            }

            if let Some(applied) = apply_adapter_materialized(
                stage,
                &mut buf,
                &mut vm,
                &mut loop_env,
                kernel,
                strategy,
            ) {
                applied?;
                continue;
            }
            unreachable!("descriptor-backed stage was not handled by materialized adapter")
        }
        LegacyPreIter::Owned(buf.into_iter())
    };

    'outer: for item in pre_iter {
        if matches!(source_demand, PullDemand::FirstInput(n) if pulled_inputs >= n) {
            break 'outer;
        }
        pulled_inputs += 1;

        let sink_done = match &pipeline.sink {
            Sink::Reducer(_) => {
                match observe_reducer_item(pipeline, item, &mut sink_acc, &mut vm, &mut loop_env)? {
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
        if matches!(source_demand, PullDemand::UntilOutput(n) if emitted_outputs >= n) {
            break 'outer;
        }
    }

    // group_by wraps its output in a single-element array; unwrap it so the caller sees the map
    let unwrap_single_collect_obj = pipeline
        .stages
        .last()
        .and_then(Stage::descriptor)
        .is_some_and(|desc| desc.method == Some(BuiltinMethod::GroupBy));
    Ok(sink_acc.finish(unwrap_single_collect_obj))
}

/// Streams a pipeline directly from a `simd-json` tape; returns `None` when any stage requires materialisation.
#[cfg(feature = "simd-json")]
pub(super) fn run_tape_field_chain(
    body: &PipelineBody,
    tape: &crate::strref::TapeData,
    keys: &[Arc<str>],
    base_env: &Env,
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
    Some(run_streaming_rows(
        &pipeline,
        base_env,
        source.iter_materialized(),
    ))
}

fn run_streaming_rows<I>(pipeline: &Pipeline, base_env: &Env, iter: I) -> Result<Val, EvalError>
where
    I: IntoIterator<Item = Val>,
{
    let mut vm = crate::vm::VM::new();
    let mut loop_env = base_env.clone();
    let source_demand = pipeline.source_demand().chain.pull;
    let mut pulled_inputs: usize = 0;
    let mut emitted_outputs: usize = 0;
    let mut stage_taken: Vec<usize> = vec![0; pipeline.stages.len()];
    let mut stage_skipped: Vec<usize> = vec![0; pipeline.stages.len()];
    let mut sink_acc = SinkAccumulator::new(&pipeline.sink);
    let terminal_map_idx = if matches!(pipeline.sink, Sink::Collect)
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

    'outer: for mut item in iter {
        if matches!(source_demand, PullDemand::FirstInput(n) if pulled_inputs >= n) {
            break 'outer;
        }
        pulled_inputs += 1;

        for (stage_idx, stage) in pipeline.stages.iter().enumerate() {
            let kernel = pipeline
                .stage_kernels
                .get(stage_idx)
                .unwrap_or(&BodyKernel::Generic);
            match stage {
                Stage::CompiledMap(plan) => {
                    item = run_compiled_map(plan, item)?;
                }
                _ => match super::val_stage_flow::apply_adapter_streaming(
                    stage,
                    stage_idx,
                    item,
                    &mut vm,
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
                        if matches!(source_demand, PullDemand::UntilOutput(n) if emitted_outputs >= n)
                        {
                            break 'outer;
                        }
                        continue 'outer;
                    }
                },
            }
        }

        let sink_done = match &pipeline.sink {
            Sink::Reducer(_) => {
                match observe_reducer_item(pipeline, item, &mut sink_acc, &mut vm, &mut loop_env)? {
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
        if matches!(source_demand, PullDemand::UntilOutput(n) if emitted_outputs >= n) {
            break 'outer;
        }
    }

    if let Some(collector) = terminal_map_collect {
        return Ok(collector.finish());
    }
    Ok(sink_acc.finish(false))
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
        let mut ctx = crate::builtins::builtin_def::BarrierCtx {
            vm,
            env: loop_env,
            kernel,
            stage,
            strategy,
        };
        use crate::builtins::{BuiltinMethod as M, builtin_def::Builtin, defs};
        let trait_result = match method {
            M::Reverse => <defs::Reverse as Builtin>::apply_barrier(&mut ctx, buf, body),
            M::Sort => <defs::Sort as Builtin>::apply_barrier(&mut ctx, buf, body),
            M::Window => <defs::Window as Builtin>::apply_barrier(&mut ctx, buf, body),
            M::Chunk => <defs::Chunk as Builtin>::apply_barrier(&mut ctx, buf, body),
            M::GroupBy => <defs::GroupBy as Builtin>::apply_barrier(&mut ctx, buf, body),
            M::CountBy => <defs::CountBy as Builtin>::apply_barrier(&mut ctx, buf, body),
            M::IndexBy => <defs::IndexBy as Builtin>::apply_barrier(&mut ctx, buf, body),
            M::Filter | M::Find | M::FindAll => {
                <defs::Filter as Builtin>::apply_barrier(&mut ctx, buf, body)
            }
            M::Map => <defs::Map as Builtin>::apply_barrier(&mut ctx, buf, body),
            M::FlatMap => <defs::FlatMap as Builtin>::apply_barrier(&mut ctx, buf, body),
            M::Unique => <defs::Unique as Builtin>::apply_barrier(&mut ctx, buf, body),
            M::UniqueBy => <defs::UniqueBy as Builtin>::apply_barrier(&mut ctx, buf, body),
            M::TakeWhile => <defs::TakeWhile as Builtin>::apply_barrier(&mut ctx, buf, body),
            M::DropWhile => <defs::DropWhile as Builtin>::apply_barrier(&mut ctx, buf, body),
            M::Take => <defs::Take as Builtin>::apply_barrier(&mut ctx, buf, body),
            M::Skip => <defs::Skip as Builtin>::apply_barrier(&mut ctx, buf, body),
            M::FindIndex => <defs::FindIndex as Builtin>::apply_barrier(&mut ctx, buf, body),
            M::IndicesWhere => <defs::IndicesWhere as Builtin>::apply_barrier(&mut ctx, buf, body),
            M::MaxBy => <defs::MaxBy as Builtin>::apply_barrier(&mut ctx, buf, body),
            M::MinBy => <defs::MinBy as Builtin>::apply_barrier(&mut ctx, buf, body),
            M::TransformKeys => <defs::TransformKeys as Builtin>::apply_barrier(&mut ctx, buf, body),
            M::TransformValues => <defs::TransformValues as Builtin>::apply_barrier(&mut ctx, buf, body),
            M::FilterKeys => <defs::FilterKeys as Builtin>::apply_barrier(&mut ctx, buf, body),
            M::FilterValues => <defs::FilterValues as Builtin>::apply_barrier(&mut ctx, buf, body),
            _ => None,
        };
        if let Some(r) = trait_result {
            return Some(r);
        }
    }
    // Remaining barrier dispatch: ElementBuiltin (Stage::Builtin/IntRangeBuiltin/StringPairBuiltin),
    // ExpandingBuiltin (Split), and SortedDedup (no trait migration yet).
    // All other variants are handled by trait dispatch above and unreachable here.
    match stage_executor(stage)? {
        BuiltinPipelineExecutor::ElementBuiltin => {
            let mut out: Vec<Val> = Vec::with_capacity(buf.len());
            for v in std::mem::take(buf) {
                out.push(apply_element_adapter(stage, v));
            }
            *buf = out;
            Some(Ok(()))
        }
        BuiltinPipelineExecutor::ExpandingBuiltin => {
            let mut out: Vec<Val> = Vec::with_capacity(buf.len());
            for v in std::mem::take(buf) {
                apply_expanding_adapter(stage, &v, &mut out);
            }
            *buf = out;
            Some(Ok(()))
        }
        BuiltinPipelineExecutor::SortedDedup => {
            match stage.body_program() {
                None => {
                    buf.sort_by(cmp_val_total);
                    buf.dedup_by(|a, b| crate::util::vals_eq(a, b));
                }
                Some(prog) => {
                    let mut keyed: Vec<(Val, Val)> = Vec::with_capacity(buf.len());
                    for v in buf.iter() {
                        let key = match eval_kernel(kernel, v, |item| {
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
        // All other executor variants (RowFilter, RowMap, RowFlatMap, ObjectLambda,
        // Position, Reverse, Sort, UniqueBy, GroupBy, CountBy, IndexBy, IndicesWhere,
        // FindIndex, ArgExtreme, Chunk, Window, PrefixWhile) are handled above by
        // trait dispatch via Builtin::apply_barrier — unreachable here.
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
        } if matches!(*method, BuiltinMethod::Replace | BuiltinMethod::ReplaceAll) => {
            replace_apply(
                v.clone(),
                first,
                second,
                *method == BuiltinMethod::ReplaceAll,
            )
            .unwrap_or(v)
        }
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
        let keep = eval_kernel(kernel, &item, |item| {
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
        let reducer_item = eval_kernel(kernel, &item, |item| {
            apply_item_in_env(vm, loop_env, item, project)
        })?;
        sink_acc.push_projected_numeric(&reducer_item);
    } else {
        sink_acc.push(item);
    }

    Ok(ReducerItemFlow::Observed)
}

/// Applies an object-lambda stage (`TransformKeys`, `TransformValues`, `FilterKeys`, `FilterValues`) to `recv`.
pub(crate) fn apply_lambda_obj(
    stage: &Stage,
    recv: &Val,
    vm: &mut crate::vm::VM,
    loop_env: &mut crate::context::Env,
    kernel: &BodyKernel,
    prog: &crate::vm::Program,
) -> Result<Val, EvalError> {
    let m = match recv.as_object() {
        Some(m) => m,
        None => return Ok(recv.clone()),
    };
    let mut out: indexmap::IndexMap<std::sync::Arc<str>, Val> =
        indexmap::IndexMap::with_capacity(m.len());
    for (k, v) in m.iter() {
        match stage {
            Stage::ExprBuiltin {
                method: BuiltinMethod::TransformKeys,
                ..
            } => {
                let k_val = Val::Str(k.clone());
                let new_k = eval_kernel(kernel, &k_val, |item| {
                    apply_item_in_env(vm, loop_env, item, prog)
                })?;
                let new_k_arc = match new_k {
                    Val::Str(s) => s,
                    other => std::sync::Arc::from(crate::util::val_to_string(&other).as_str()),
                };
                out.insert(new_k_arc, v.clone());
            }
            Stage::ExprBuiltin {
                method: BuiltinMethod::TransformValues,
                ..
            } => {
                let new_v = eval_kernel(kernel, v, |item| {
                    apply_item_in_env(vm, loop_env, item, prog)
                })?;
                out.insert(k.clone(), new_v);
            }
            Stage::ExprBuiltin {
                method: BuiltinMethod::FilterKeys,
                ..
            } => {
                let k_val = Val::Str(k.clone());
                if is_truthy(&eval_kernel(kernel, &k_val, |item| {
                    apply_item_in_env(vm, loop_env, item, prog)
                })?) {
                    out.insert(k.clone(), v.clone());
                }
            }
            Stage::ExprBuiltin {
                method: BuiltinMethod::FilterValues,
                ..
            } => {
                if is_truthy(&eval_kernel(kernel, v, |item| {
                    apply_item_in_env(vm, loop_env, item, prog)
                })?) {
                    out.insert(k.clone(), v.clone());
                }
            }
            _ => unreachable!("apply_lambda_obj called with non-Obj-lambda Stage"),
        }
    }
    Ok(Val::obj(out))
}
