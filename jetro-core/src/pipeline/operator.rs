use std::sync::Arc;

use crate::builtins::BuiltinMethod;
use crate::vm::Program;

use super::NumOp;

#[derive(Debug, Clone)]
pub struct ReducerSpec {
    pub op: ReducerOp,
    pub predicate: Option<Arc<Program>>,
    pub projection: Option<Arc<Program>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReducerOp {
    Count,
    Numeric(NumOp),
}

impl ReducerSpec {
    pub fn count() -> Self {
        Self {
            op: ReducerOp::Count,
            predicate: None,
            projection: None,
        }
    }

    pub fn numeric_op(&self) -> Option<NumOp> {
        match self.op {
            ReducerOp::Numeric(op) => Some(op),
            ReducerOp::Count => None,
        }
    }

    pub(crate) fn sink_programs(&self) -> impl Iterator<Item = &Arc<Program>> {
        self.predicate.iter().chain(self.projection.iter())
    }

    pub(crate) fn predicate_kernel_index(&self) -> Option<usize> {
        self.predicate.as_ref().map(|_| 0)
    }

    pub(crate) fn projection_kernel_index(&self) -> Option<usize> {
        self.projection
            .as_ref()
            .map(|_| usize::from(self.predicate.is_some()))
    }

    pub fn method(&self) -> Option<BuiltinMethod> {
        match self.op {
            ReducerOp::Count => Some(BuiltinMethod::Count),
            ReducerOp::Numeric(op) => Some(op.method()),
        }
    }
}
