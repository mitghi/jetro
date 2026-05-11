//! Execution helpers for nested pipeline plans embedded inside pipeline stages.
//!
//! Lowering creates a `Plan`; executors use this module to run that plan against
//! the current row without depending back on the lowering module.

use crate::{
    data::context::EvalError,
    data::value::Val,
};

use super::{PipelineBody, Plan, Source};

/// Runs a nested plan with `seed` as the current row/root.
pub(super) fn run_plan(plan: &Plan, seed: Val) -> Result<Val, EvalError> {
    let source = match &plan.source {
        Source::Receiver(_) => Source::Receiver(seed.clone()),
        source => source.clone(),
    };
    let root = seed;
    let synth = PipelineBody {
        stages: plan.stages.clone(),
        stage_exprs: plan.stage_exprs.clone(),
        sink: plan.sink.clone(),
        stage_kernels: plan.stage_kernels.clone(),
        sink_kernels: plan.sink_kernels.clone(),
    }
    .with_source(source);
    synth.run(&root)
}
