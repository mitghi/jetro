use crate::{
    builtin_registry::{pipeline_executor, BuiltinId},
    builtins::BuiltinPipelineExecutor,
};

use super::Stage;

pub(crate) enum StageFlow<T> {
    Continue(T),
    SkipRow,
    Stop,
    TerminalCollected,
}

pub(crate) fn stage_executor(stage: &Stage) -> Option<BuiltinPipelineExecutor> {
    let desc = stage.descriptor()?;
    desc.executor_override
        .or_else(|| {
            desc.method
                .and_then(|method| pipeline_executor(BuiltinId::from_method(method)))
        })
        .or_else(|| {
            matches!(stage, Stage::Builtin(_)).then_some(BuiltinPipelineExecutor::ElementBuiltin)
        })
}
