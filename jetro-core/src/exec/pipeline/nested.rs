//! Execution helpers for nested pipeline plans embedded inside pipeline stages.
//!
//! Lowering creates a `Plan`; executors use this module to run that plan against
//! the current row without depending back on the lowering module.

use crate::{data::context::EvalError, data::value::Val};

use super::{Pipeline, PipelineBody, Plan, Source};

/// Prepared nested plan execution metadata. The physical path and demand annotations are stable
/// across rows; receiver-sourced plans only swap the receiver value before running.
#[derive(Debug, Clone)]
pub(super) struct PreparedPlan {
    pipeline: Pipeline,
    receiver_source: bool,
}

impl PreparedPlan {
    pub(super) fn new(plan: &Plan) -> Self {
        let receiver_source = matches!(plan.source, Source::Receiver(_));
        let source = if receiver_source {
            Source::Receiver(Val::Null)
        } else {
            plan.source.clone()
        };
        Self {
            pipeline: body_from_plan(plan).with_source(source),
            receiver_source,
        }
    }

    pub(super) fn run(&self, seed: Val) -> Result<Val, EvalError> {
        if self.receiver_source {
            let mut pipeline = self.pipeline.clone();
            pipeline.source = Source::Receiver(seed.clone());
            return pipeline.run(&seed);
        }
        self.pipeline.run(&seed)
    }
}

fn body_from_plan(plan: &Plan) -> PipelineBody {
    PipelineBody {
        stages: plan.stages.clone(),
        stage_exprs: plan.stage_exprs.clone(),
        sink: plan.sink.clone(),
        stage_kernels: plan.stage_kernels.clone(),
        sink_kernels: plan.sink_kernels.clone(),
    }
}
