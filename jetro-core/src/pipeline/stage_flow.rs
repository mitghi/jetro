use crate::builtins::BuiltinPipelineExecutor;

use super::Stage;

pub(crate) enum StageFlow<T> {
    Continue(T),
    SkipRow,
    Stop,
    TerminalCollected,
}

pub(crate) fn stage_executor(stage: &Stage) -> Option<BuiltinPipelineExecutor> {
    stage.descriptor()?.executor()
}
