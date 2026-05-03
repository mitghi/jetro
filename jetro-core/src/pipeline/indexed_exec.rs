//! Index-aware pipeline execution for `first` / `last` positional sinks.
//! Tracks absolute row position so positional selection can terminate the
//! loop without scanning the entire source.

use crate::{
    context::{Env, EvalError},
    value::Val,
};

use super::{row_source, select_strategy, Pipeline, Position, Stage, Strategy};

/// Executes a positional (`first`/`last`) pipeline by directly indexing the source; returns `None` when the pipeline does not qualify.
pub(super) fn run(
    pipeline: &Pipeline,
    root: &Val,
    base_env: &Env,
) -> Option<Result<Val, EvalError>> {
    let strategy = select_strategy(&pipeline.stages, &pipeline.sink);
    if strategy != Strategy::IndexedDispatch {
        return None;
    }

    let recv = row_source::resolve(&pipeline.source, root);
    let len = row_source::row_count(&recv)?;

    let demand = pipeline.sink.demand();
    let idx = match demand.positional? {
        Position::First => 0,
        Position::Last => len.checked_sub(1)?,
    };
    if idx >= len {
        return Some(Ok(Val::Null));
    }

    let elem = row_source::row_at(&recv, idx)?;

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
            Stage::Builtin(call) => {
                cur = call.apply(&cur).unwrap_or(cur);
            }
            _ => return None,
        }
    }

    Some(Ok(cur))
}
