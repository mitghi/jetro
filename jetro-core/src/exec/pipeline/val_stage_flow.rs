//! Val-based stage flow helpers for the legacy execution path.
//! Implements per-element filter, map, and take-while stepping over `Val`
//! items using the builtins-layer primitives.

use crate::{
    data::context::{Env, EvalError},
    data::value::Val,
};

use super::{materialized_exec, BodyKernel, Stage, StageFlow, TerminalMapCollector};

/// Applies `stage` to `item` in the streaming loop; barrier and expanding stages return `Continue`.
pub(super) fn apply_adapter_streaming<'a>(
    stage: &Stage,
    stage_idx: usize,
    item: Val,
    vm: &mut crate::vm::VM,
    loop_env: &mut Env,
    kernel: &BodyKernel,
    stage_taken: &mut [usize],
    stage_skipped: &mut [usize],
    terminal_map_idx: Option<usize>,
    terminal_map_collect: &mut Option<TerminalMapCollector<'a>>,
) -> Result<StageFlow<Val>, EvalError> {
    // Trait dispatch: try Builtin::apply_stream for migrated methods.
    if let Some(method) = stage.descriptor().and_then(|d| d.method) {
        let body = stage.body_program();
        let mut ctx = crate::builtins::builtin::StreamCtx {
            vm,
            env: loop_env,
            kernel,
            stage,
            stage_idx,
            stage_taken,
            stage_skipped,
            terminal_map_idx,
            terminal_map_collect,
        };
        use crate::builtins::{builtin::Builtin, defs, BuiltinMethod as M};
        match method {
            M::Filter | M::Find | M::FindAll => {
                return <defs::Filter as Builtin>::apply_stream(&mut ctx, item, body);
            }
            M::Map => return <defs::Map as Builtin>::apply_stream(&mut ctx, item, body),
            M::TakeWhile => {
                return <defs::TakeWhile as Builtin>::apply_stream(&mut ctx, item, body)
            }
            M::DropWhile => {
                return <defs::DropWhile as Builtin>::apply_stream(&mut ctx, item, body)
            }
            M::Take => return <defs::Take as Builtin>::apply_stream(&mut ctx, item, body),
            M::Skip => return <defs::Skip as Builtin>::apply_stream(&mut ctx, item, body),
            M::TransformKeys => {
                return <defs::TransformKeys as Builtin>::apply_stream(&mut ctx, item, body)
            }
            M::TransformValues => {
                return <defs::TransformValues as Builtin>::apply_stream(&mut ctx, item, body)
            }
            M::FilterKeys => {
                return <defs::FilterKeys as Builtin>::apply_stream(&mut ctx, item, body)
            }
            M::FilterValues => {
                return <defs::FilterValues as Builtin>::apply_stream(&mut ctx, item, body)
            }
            _ => {}
        }
    }
    // ElementBuiltin: element-wise scalar apply via Stage variant match.
    // All other variants pass through (barriers handled by materialised path).
    match stage {
        Stage::Builtin(call) if call.method == crate::builtins::BuiltinMethod::Compact => {
            if matches!(item, Val::Null) {
                Ok(StageFlow::SkipRow)
            } else {
                Ok(StageFlow::Continue(item))
            }
        }
        Stage::Builtin(call) if call.method == crate::builtins::BuiltinMethod::Remove => {
            match &call.args {
                crate::builtins::BuiltinArgs::Val(target)
                    if crate::util::vals_eq(&item, target) =>
                {
                    Ok(StageFlow::SkipRow)
                }
                _ => Ok(StageFlow::Continue(item)),
            }
        }
        Stage::Builtin(_) | Stage::IntRangeBuiltin { .. } | Stage::StringPairBuiltin { .. } => Ok(
            StageFlow::Continue(materialized_exec::apply_element_adapter(stage, item)),
        ),
        _ => Ok(StageFlow::Continue(item)),
    }
}
