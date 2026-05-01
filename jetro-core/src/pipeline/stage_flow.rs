use crate::builtins::BuiltinPipelineExecutor;

use super::Stage;

pub(crate) enum StageFlow<T> {
    Continue(T),
    SkipRow,
    Stop,
    TerminalCollected,
}

pub(crate) fn stage_executor(stage: &Stage) -> Option<BuiltinPipelineExecutor> {
    stage
        .builtin_method_metadata()
        .and_then(|method| method.spec().pipeline_executor)
        .or_else(|| {
            if matches!(stage, Stage::Builtin(_)) {
                Some(BuiltinPipelineExecutor::ElementBuiltin)
            } else if matches!(stage, Stage::SortedDedup(_)) {
                Some(BuiltinPipelineExecutor::SortedDedup)
            } else {
                None
            }
        })
}
