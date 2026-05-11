//! Execution helpers for nested pipeline plans embedded inside pipeline stages.
//!
//! Lowering creates a `Plan`; executors use this module to run that plan against
//! the current row without depending back on the lowering module.

use crate::{
    data::context::EvalError,
    data::value::Val,
};

use super::{Pipeline, PipelineBody, Plan, Source};

/// Prepared nested plan execution metadata. Field-chain nested plans are reusable across rows
/// because only the root value changes; receiver plans need a fresh receiver source per row.
#[derive(Clone)]
pub(super) struct PreparedPlan {
    source: Source,
    body: PipelineBody,
    reusable: Option<Pipeline>,
}

impl PreparedPlan {
    pub(super) fn new(plan: &Plan) -> Self {
        let source = plan.source.clone();
        let body = body_from_plan(plan);
        let reusable = match &source {
            Source::Receiver(_) => None,
            source => Some(body.clone().with_source(source.clone())),
        };
        Self {
            source,
            body,
            reusable,
        }
    }

    pub(super) fn run(&self, seed: Val) -> Result<Val, EvalError> {
        if let Some(pipeline) = &self.reusable {
            return pipeline.run(&seed);
        }

        let source = match &self.source {
            Source::Receiver(_) => Source::Receiver(seed.clone()),
            source => source.clone(),
        };
        self.body.clone().with_source(source).run(&seed)
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

/// Runs a nested plan with `seed` as the current row/root.
pub(super) fn run_plan(plan: &Plan, seed: Val) -> Result<Val, EvalError> {
    let source = match &plan.source {
        Source::Receiver(_) => Source::Receiver(seed.clone()),
        source => source.clone(),
    };
    body_from_plan(plan).with_source(source).run(&seed)
}
