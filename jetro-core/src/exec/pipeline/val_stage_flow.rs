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
        return crate::builtins::registry::apply_stream_hook_or_else(
            method,
            &mut ctx,
            item,
            body,
            |item| fallback_streaming(stage, item),
        );
    }
    // ElementBuiltin: element-wise scalar apply via Stage variant match.
    // All other variants pass through (barriers handled by materialised path).
    fallback_streaming(stage, item)
}

fn fallback_streaming(stage: &Stage, item: Val) -> Result<StageFlow<Val>, EvalError> {
    match stage {
        Stage::Builtin(_) | Stage::IntRangeBuiltin { .. } | Stage::StringPairBuiltin { .. } => Ok(
            StageFlow::Continue(materialized_exec::apply_element_adapter(stage, item)),
        ),
        _ => Ok(StageFlow::Continue(item)),
    }
}
