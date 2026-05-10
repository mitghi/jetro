//! Index-aware pipeline execution for `first` / `last` positional sinks.
//! Tracks absolute row position so positional selection can terminate the
//! loop without scanning the entire source.

use crate::{
    data::context::{Env, EvalError},
    data::value::Val,
    plan::demand::PullDemand,
};

use super::{row_source, Pipeline, Position, SourceAccessMode, Stage};

/// Executes a positional (`first`/`last`) pipeline by directly indexing the source; returns `None` when the pipeline does not qualify.
pub(super) fn run(
    pipeline: &Pipeline,
    root: &Val,
    base_env: &Env,
) -> Option<Result<Val, EvalError>> {
    let recv = row_source::resolve(&pipeline.source, root);
    let len = row_source::row_count(&recv)?;

    let demand = pipeline.source_demand();
    if let super::Sink::SelectMany { n, from_end } = pipeline.sink {
        return run_select_many(pipeline, base_env, &recv, len, n, from_end);
    }

    let idx = match pipeline.source_access {
        SourceAccessMode::Indexed(idx) => idx,
        SourceAccessMode::IndexedFromEnd(offset) => len.checked_sub(offset + 1)?,
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
    apply_indexed_stages(pipeline, base_env, elem)
}

fn run_select_many(
    pipeline: &Pipeline,
    base_env: &Env,
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
    for idx in start..end {
        if let Some(elem) = row_source::row_at(recv, idx) {
            match apply_indexed_stages(pipeline, base_env, elem)? {
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

fn apply_indexed_stages(
    pipeline: &Pipeline,
    base_env: &Env,
    elem: Val,
) -> Option<Result<Val, EvalError>> {
    let mut vm = crate::vm::VM::new();
    let mut env = base_env.clone();
    let mut cur = elem;
    for stage in &pipeline.stages {
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
                cur = match super::lower::run_compiled_map(plan, cur) {
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
