//! Index-aware pipeline execution for `first` / `last` positional sinks.
//! Tracks absolute row position so positional selection can terminate the
//! loop without scanning the entire source.

use crate::{
    data::context::{Env, EvalError},
    data::value::Val,
    plan::demand::PullDemand,
    vm::VM,
};

use super::{
    index_from_end, nested::PreparedPlan, row_source, Pipeline, Position, SourceAccessMode, Stage,
};

/// Executes a positional (`first`/`last`) pipeline by directly indexing the source; returns `None` when the pipeline does not qualify.
pub(super) fn run(
    pipeline: &Pipeline,
    root: &Val,
    base_env: &Env,
    vm: &mut VM,
) -> Option<Result<Val, EvalError>> {
    let recv = row_source::resolve(&pipeline.source, root);
    let len = row_source::row_count(&recv)?;

    let demand = pipeline.source_demand();
    if let super::Sink::SelectMany { n, from_end } = pipeline.sink {
        return run_select_many(pipeline, base_env, vm, &recv, len, n, from_end);
    }

    let idx = match pipeline.source_access() {
        SourceAccessMode::Indexed(idx) => idx,
        SourceAccessMode::IndexedFromEnd(offset) => index_from_end(len, offset)?,
        SourceAccessMode::IndexedSuffix(count) => len.saturating_sub(count),
        SourceAccessMode::ForwardBounded(_) => 0,
        SourceAccessMode::Reverse { .. } => len.checked_sub(1)?,
        SourceAccessMode::Forward | SourceAccessMode::MaterializedFallback => {
            match demand.chain.pull {
                PullDemand::NthInput(idx) => idx,
                PullDemand::FirstInput(_) => 0,
                PullDemand::LastInput(_) => len.checked_sub(1)?,
                _ => match demand.positional? {
                    Position::First => 0,
                    Position::Last => len.checked_sub(1)?,
                },
            }
        }
    };
    if idx >= len {
        return Some(Ok(Val::Null));
    }

    let elem = row_source::row_at(&recv, idx)?;
    let prepared = prepare_nested_stages(pipeline);
    apply_indexed_stages(pipeline, &prepared, base_env, vm, elem)
}

fn run_select_many(
    pipeline: &Pipeline,
    base_env: &Env,
    vm: &mut VM,
    recv: &Val,
    len: usize,
    n: usize,
    from_end: bool,
) -> Option<Result<Val, EvalError>> {
    if n == 0 {
        return Some(Ok(Val::Null));
    }

    let start = if from_end { len.saturating_sub(n) } else { 0 };
    let end = if from_end { len } else { n.min(len) };
    let mut out = Vec::with_capacity(end.saturating_sub(start));
    let prepared = prepare_nested_stages(pipeline);
    for idx in start..end {
        if let Some(elem) = row_source::row_at(recv, idx) {
            match apply_indexed_stages(pipeline, &prepared, base_env, vm, elem)? {
                Ok(value) => out.push(value),
                Err(err) => return Some(Err(err)),
            }
        }
    }

    if n == 1 {
        Some(Ok(out.into_iter().next().unwrap_or(Val::Null)))
    } else {
        Some(Ok(Val::arr(out)))
    }
}

fn prepare_nested_stages(pipeline: &Pipeline) -> Vec<Option<PreparedPlan>> {
    pipeline
        .stages
        .iter()
        .map(|stage| match stage {
            Stage::CompiledMap(plan) => Some(PreparedPlan::new(plan)),
            _ => None,
        })
        .collect()
}

fn apply_indexed_stages(
    pipeline: &Pipeline,
    prepared: &[Option<PreparedPlan>],
    base_env: &Env,
    vm: &mut VM,
    elem: Val,
) -> Option<Result<Val, EvalError>> {
    let mut env = base_env.clone();
    let mut cur = elem;
    for (idx, stage) in pipeline.stages.iter().enumerate() {
        match stage {
            Stage::Map(prog, _) => {
                let prev = env.swap_current(cur);
                cur = match vm.exec_in_env(prog, &mut env) {
                    Ok(v) => v,
                    Err(e) => {
                        env.restore_current(prev);
                        return Some(Err(e));
                    }
                };
                env.restore_current(prev);
            }
            Stage::CompiledMap(plan) => {
                let result = match prepared.get(idx).and_then(Option::as_ref) {
                    Some(prepared) => prepared.run(cur),
                    None => PreparedPlan::new(plan).run(cur),
                };
                cur = match result {
                    Ok(value) => value,
                    Err(err) => return Some(Err(err)),
                };
            }
            Stage::Builtin(call) => {
                cur = call.apply(&cur).unwrap_or(cur);
            }
            _ => return None,
        }
    }

    Some(Ok(cur))
}
