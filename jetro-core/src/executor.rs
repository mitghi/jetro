//! Top-level query execution routing.
//!
//! `planner.rs` classifies the expression. This module owns the runtime
//! decision flow for a `Jetro` handle: pipeline first when planned, scalar VM
//! fallback otherwise. Keeping this out of `lib.rs` makes data flow explicit
//! without changing the physical loops in `pipeline.rs` or `vm.rs`.

use serde_json::Value;

use crate::context::EvalError;
use crate::pipeline;
use crate::planner;
use crate::value::Val;
use crate::{with_vm, Jetro, VM};

pub(crate) fn collect_json(j: &Jetro, expr: &str) -> Result<Value, EvalError> {
    let plan = planner::plan_query(expr);

    if let Some(out) = run_pipeline(j, &plan) {
        return out.map(Value::from);
    }

    run_vm_json(j, expr)
}

fn run_pipeline(j: &Jetro, plan: &planner::ExecutionPlan) -> Option<Result<Val, EvalError>> {
    let p = plan.pipeline()?;
    Some(p.run_with(&j.root_val(), Some(j as &dyn pipeline::PipelineData)))
}

fn run_vm_json(j: &Jetro, expr: &str) -> Result<Value, EvalError> {
    with_vm(|cell| match cell.try_borrow_mut() {
        Ok(mut vm) => {
            let prog = vm.get_or_compile(expr)?;
            vm.execute_val(&prog, j.root_val())
        }
        Err(_) => VM::new().run_str(expr, &j.document),
    })
}
