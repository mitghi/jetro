use std::sync::Arc;

use crate::{
    context::{Env, EvalError},
    value::Val,
};

use super::{select_strategy, walk_field_chain, Pipeline, Position, Source, Stage, Strategy};

/// Generic O(1) optimisation for `(Map | identity)*` chains terminated by a
/// positional sink (`First` / `Last`). Pulls only the target source element,
/// runs the 1:1 chain once, and returns.
pub(super) fn run(
    pipeline: &Pipeline,
    root: &Val,
    base_env: &Env,
) -> Option<Result<Val, EvalError>> {
    let strategy = select_strategy(&pipeline.stages, &pipeline.sink);
    if strategy != Strategy::IndexedDispatch {
        return None;
    }

    let recv = match &pipeline.source {
        Source::Receiver(v) => v.clone(),
        Source::FieldChain { keys } => walk_field_chain(root, keys),
    };

    let len = match &recv {
        Val::Arr(a) => a.len(),
        Val::IntVec(a) => a.len(),
        Val::FloatVec(a) => a.len(),
        Val::StrVec(a) => a.len(),
        Val::ObjVec(d) => d.nrows(),
        _ => return None,
    };

    let demand = pipeline.sink.demand();
    let idx = match demand.positional? {
        Position::First => 0,
        Position::Last => len.checked_sub(1)?,
    };
    if idx >= len {
        return Some(Ok(Val::Null));
    }

    let elem = match &recv {
        Val::Arr(a) => a[idx].clone(),
        Val::IntVec(a) => Val::Int(a[idx]),
        Val::FloatVec(a) => Val::Float(a[idx]),
        Val::StrVec(a) => Val::Str(Arc::clone(&a[idx])),
        Val::ObjVec(d) => {
            let stride = d.stride();
            let mut m: indexmap::IndexMap<Arc<str>, Val> =
                indexmap::IndexMap::with_capacity(stride);
            for (i, k) in d.keys.iter().enumerate() {
                m.insert(Arc::clone(k), d.cells[idx * stride + i].clone());
            }
            Val::Obj(Arc::new(m))
        }
        _ => return None,
    };

    let mut vm = crate::vm::VM::new();
    let mut env = base_env.clone();
    let mut cur = elem;
    for stage in &pipeline.stages {
        match stage {
            Stage::Map(prog) => {
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
