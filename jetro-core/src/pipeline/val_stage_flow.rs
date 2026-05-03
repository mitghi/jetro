//! Val-based stage flow helpers for the legacy execution path.
//! Implements per-element filter, map, and take-while stepping over `Val`
//! items using the builtins-layer primitives.

use crate::{
    builtins::{filter_one, map_one, take_while_one, BuiltinPipelineExecutor},
    context::{Env, EvalError},
    value::Val,
};

use super::{
    apply_item_in_env, eval_kernel, legacy_exec, stage_executor, BodyKernel, Stage, StageFlow,
    TerminalMapCollector,
};

/// Applies a single `Stage` to `item` in the streaming (legacy) execution loop.
///
/// Dispatches to the appropriate builtin executor or falls back to `StageFlow::Continue` for
/// barrier and expanding stages that are handled outside the per-row loop.
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
    match stage_executor(stage) {
        Some(BuiltinPipelineExecutor::ElementBuiltin) => Ok(StageFlow::Continue(
            legacy_exec::apply_element_adapter(stage, item),
        )),
        Some(BuiltinPipelineExecutor::ObjectLambda) => {
            let prog = legacy_exec::object_lambda_program(stage)
                .expect("object lambda executor must be attached to object lambda stage");
            Ok(StageFlow::Continue(legacy_exec::apply_lambda_obj(
                stage, &item, vm, loop_env, kernel, prog,
            )?))
        }
        Some(BuiltinPipelineExecutor::Position { take }) => {
            let n = match stage.descriptor().and_then(|desc| desc.usize_arg) {
                Some(n) => n,
                None => return Ok(StageFlow::Continue(item)),
            };
            if take {
                if stage_taken[stage_idx] >= n {
                    Ok(StageFlow::Stop)
                } else {
                    stage_taken[stage_idx] += 1;
                    Ok(StageFlow::Continue(item))
                }
            } else if stage_skipped[stage_idx] < n {
                stage_skipped[stage_idx] += 1;
                Ok(StageFlow::SkipRow)
            } else {
                Ok(StageFlow::Continue(item))
            }
        }
        Some(BuiltinPipelineExecutor::RowFilter) => {
            let prog = legacy_exec::row_stage_program(stage)
                .expect("row filter executor must have row program");
            if filter_one(&item, |v| {
                eval_kernel(kernel, v, |item| {
                    apply_item_in_env(vm, loop_env, item, prog)
                })
            })? {
                Ok(StageFlow::Continue(item))
            } else {
                Ok(StageFlow::SkipRow)
            }
        }
        Some(BuiltinPipelineExecutor::RowMap) => {
            let prog = legacy_exec::row_stage_program(stage)
                .expect("row map executor must have row program");
            if Some(stage_idx) == terminal_map_idx {
                terminal_map_collect
                    .as_mut()
                    .expect("terminal map collector")
                    .push_val_row(&item, kernel, |item| {
                        apply_item_in_env(vm, loop_env, item, prog)
                    })?;
                return Ok(StageFlow::TerminalCollected);
            }
            Ok(StageFlow::Continue(map_one(&item, |v| {
                eval_kernel(kernel, v, |item| {
                    apply_item_in_env(vm, loop_env, item, prog)
                })
            })?))
        }
        Some(BuiltinPipelineExecutor::PrefixWhile { take }) => {
            if !take {
                return Ok(StageFlow::Continue(item));
            }
            let prog = legacy_exec::keyed_stage_program(stage)
                .expect("take_while executor must have predicate program");
            if take_while_one(&item, |v| {
                eval_kernel(kernel, v, |item| {
                    apply_item_in_env(vm, loop_env, item, prog)
                })
            })? {
                Ok(StageFlow::Continue(item))
            } else {
                Ok(StageFlow::Stop)
            }
        }
        Some(
            BuiltinPipelineExecutor::ExpandingBuiltin
            | BuiltinPipelineExecutor::RowFlatMap
            | BuiltinPipelineExecutor::Reverse
            | BuiltinPipelineExecutor::Sort
            | BuiltinPipelineExecutor::UniqueBy
            | BuiltinPipelineExecutor::GroupBy
            | BuiltinPipelineExecutor::CountBy
            | BuiltinPipelineExecutor::IndexBy
            | BuiltinPipelineExecutor::FindIndex
            | BuiltinPipelineExecutor::IndicesWhere
            | BuiltinPipelineExecutor::ArgExtreme { .. }
            | BuiltinPipelineExecutor::Chunk
            | BuiltinPipelineExecutor::Window
            | BuiltinPipelineExecutor::SortedDedup,
        )
        | None => Ok(StageFlow::Continue(item)),
    }
}
