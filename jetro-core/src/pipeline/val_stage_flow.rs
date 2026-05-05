//! Val-based stage flow helpers for the legacy execution path.
//! Implements per-element filter, map, and take-while stepping over `Val`
//! items using the builtins-layer primitives.

use crate::{
    builtins::BuiltinPipelineExecutor,
    context::{Env, EvalError},
    value::Val,
};

use super::{legacy_exec, stage_executor, BodyKernel, Stage, StageFlow, TerminalMapCollector};

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
    // Returns `Some(flow)` when method is migrated; `None` falls through to legacy executor match.
    if let Some(method) = stage.descriptor().and_then(|d| d.method) {
        let body = stage.body_program();
        let mut ctx = crate::builtins::builtin_def::StreamCtx {
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
        // Only dispatch for methods that have overridden `apply_stream`.
        // Migrated set is enumerated explicitly to avoid invoking the default
        // (pass-through) impl on every non-streaming method.
        use crate::builtins::{BuiltinMethod as M, builtin_def::Builtin, defs};
        match method {
            M::Filter | M::Find | M::FindAll => {
                return <defs::Filter as Builtin>::apply_stream(&mut ctx, item, body);
            }
            M::Map => return <defs::Map as Builtin>::apply_stream(&mut ctx, item, body),
            M::TakeWhile => return <defs::TakeWhile as Builtin>::apply_stream(&mut ctx, item, body),
            M::DropWhile => return <defs::DropWhile as Builtin>::apply_stream(&mut ctx, item, body),
            M::Take => return <defs::Take as Builtin>::apply_stream(&mut ctx, item, body),
            M::Skip => return <defs::Skip as Builtin>::apply_stream(&mut ctx, item, body),
            M::TransformKeys => return <defs::TransformKeys as Builtin>::apply_stream(&mut ctx, item, body),
            M::TransformValues => return <defs::TransformValues as Builtin>::apply_stream(&mut ctx, item, body),
            M::FilterKeys => return <defs::FilterKeys as Builtin>::apply_stream(&mut ctx, item, body),
            M::FilterValues => return <defs::FilterValues as Builtin>::apply_stream(&mut ctx, item, body),
            _ => {}
        }
    }
    // Remaining dispatch for non-trait-migrated methods:
    // - ElementBuiltin: Stage::Builtin / IntRangeBuiltin / StringPairBuiltin (apply via element adapter)
    // - All barrier/expanding methods: pass-through (handled by materialised path elsewhere)
    match stage_executor(stage) {
        Some(BuiltinPipelineExecutor::ElementBuiltin) => Ok(StageFlow::Continue(
            legacy_exec::apply_element_adapter(stage, item),
        )),
        // Filter / Map / TakeWhile / DropWhile / Take / Skip / ObjectLambda variants
        // are handled above via Builtin::apply_stream — the executor enum classification
        // is preserved here only as a runtime sanity match for unmigrated paths.
        _ => Ok(StageFlow::Continue(item)),
    }
}
