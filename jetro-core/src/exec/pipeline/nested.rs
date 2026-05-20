//! Execution helpers for nested pipeline plans embedded inside pipeline stages.
//!
//! Lowering creates a `Plan`; executors use this module to run that plan against
//! the current row without depending back on the lowering module.

use crate::{
    data::context::{Env, EvalError},
    data::value::Val,
    data::view::ValueView,
};

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

/// Runs a nested plan against a borrowed row view when the plan can be evaluated
/// without reading the materialised outer root. Returns `None` when a fallback
/// path would need root/current VM semantics that require ownership.
pub(super) fn run_plan_view<'a, V>(plan: &Plan, seed: &V) -> Option<Result<Val, EvalError>>
where
    V: ValueView<'a>,
{
    let body = body_from_plan(plan);
    if !body.can_run_with_materialized_receiver() {
        return None;
    }
    let source = match &plan.source {
        Source::Receiver(_) => seed.clone(),
        Source::FieldChain { keys } => crate::exec::view::walk_fields(seed.clone(), keys),
    };
    let env = Env::new(Val::Null);
    let mut vm = crate::vm::VM::new();
    crate::exec::view::run_with_env_and_vm(source, &body, None, &env, &mut vm)
}
