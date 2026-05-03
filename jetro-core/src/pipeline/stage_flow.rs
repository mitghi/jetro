//! `StageFlow` control-flow enum for the per-element pipeline loop.
//! Communicates continue, filter-skip, early stop, and terminal-collect
//! signals from a stage to its caller without heap allocation.

use crate::builtins::BuiltinPipelineExecutor;

use super::Stage;

/// Per-element control-flow signal returned by a pipeline stage.
///
/// Propagated up the execution loop so callers can act without heap allocation or panic.
pub(crate) enum StageFlow<T> {
    /// Stage produced a value; pass it to the next stage or sink.
    Continue(T),
    /// Stage filtered out this element; skip to the next input row.
    SkipRow,
    /// Stage signalled early termination; discard remaining input.
    Stop,
    /// A terminal-map stage already wrote the result; no further accumulation needed.
    TerminalCollected,
}

/// Returns the `BuiltinPipelineExecutor` variant that drives `stage`, if any.
///
/// `None` means no builtin executor is registered and the stage falls through to the generic path.
pub(crate) fn stage_executor(stage: &Stage) -> Option<BuiltinPipelineExecutor> {
    stage.descriptor()?.executor()
}
