//! Query execution planner.
//!
//! This module is the explicit decision point between scalar VM execution,
//! Pipeline IR execution, and scalar VM fallback. The hot execution loops
//! remain in `vm.rs`, `pipeline.rs`, and `composed.rs`; the planner only
//! classifies a query so callers do not duplicate routing logic.

use crate::parser;
use crate::pipeline::Pipeline;

/// Top-level physical execution choice for a query.
#[derive(Clone)]
pub enum ExecutionPlan {
    /// The expression lowered to Pipeline IR. Pipeline execution may still
    /// choose columnar or composed physical loops internally.
    Pipeline(Pipeline),
    /// Fallback to the scalar bytecode VM.
    Vm,
}

impl ExecutionPlan {
    #[inline]
    pub fn pipeline(&self) -> Option<&Pipeline> {
        match self {
            ExecutionPlan::Pipeline(p) => Some(p),
            ExecutionPlan::Vm => None,
        }
    }
}

/// Parse and classify a query once.
#[inline]
pub fn plan_query(expr: &str) -> ExecutionPlan {
    parser::parse(expr)
        .ok()
        .as_ref()
        .and_then(Pipeline::lower)
        .map(ExecutionPlan::Pipeline)
        .unwrap_or(ExecutionPlan::Vm)
}
