//! Operator IR types: `ReducerSpec`, `SortSpec`, `NumOp`, and related enums.
//! Shared across lowering, execution, and the composed substrate.

use std::sync::Arc;

use crate::parse::ast::Expr;
use crate::builtins::BuiltinMethod;
use crate::vm::Program;

use super::NumOp;

/// Specification for a terminal reducer sink (`count`, `sum`, `avg`, `min`, `max`).
#[derive(Debug, Clone)]
pub struct ReducerSpec {
    /// The aggregation operation to perform.
    pub op: ReducerOp,
    /// Optional predicate that gates which rows are counted or aggregated.
    pub predicate: Option<Arc<Program>>,
    /// Optional projection applied to each row before aggregation.
    pub projection: Option<Arc<Program>>,
    /// Source AST for `predicate`, used during IR analysis.
    pub predicate_expr: Option<Arc<Expr>>,
    /// Source AST for `projection`, used during IR analysis.
    pub projection_expr: Option<Arc<Expr>>,
}

/// The kind of reduction a `ReducerSpec` performs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReducerOp {
    /// Counts the number of (predicate-passing) rows.
    Count,
    /// Applies a numeric aggregate (`Sum`, `Avg`, `Min`, `Max`).
    Numeric(NumOp),
}

impl ReducerSpec {
    /// Constructs a plain `Count` reducer with no predicate or projection.
    pub fn count() -> Self {
        Self {
            op: ReducerOp::Count,
            predicate: None,
            projection: None,
            predicate_expr: None,
            projection_expr: None,
        }
    }

    /// Returns the `NumOp` for a `Numeric` reducer, or `None` for `Count`.
    pub fn numeric_op(&self) -> Option<NumOp> {
        match self.op {
            ReducerOp::Numeric(op) => Some(op),
            ReducerOp::Count => None,
        }
    }

    /// Iterates over embedded programs (predicate then projection) for kernel enumeration.
    pub(crate) fn sink_programs(&self) -> impl Iterator<Item = &Arc<Program>> {
        self.predicate.iter().chain(self.projection.iter())
    }

    /// Returns the sink-kernel index for the predicate (`0` when present), or `None`.
    pub(crate) fn predicate_kernel_index(&self) -> Option<usize> {
        self.predicate.as_ref().map(|_| 0)
    }

    /// Returns the sink-kernel index for the projection (`1` when a predicate also exists), or `None`.
    pub(crate) fn projection_kernel_index(&self) -> Option<usize> {
        self.projection
            .as_ref()
            .map(|_| usize::from(self.predicate.is_some()))
    }

    /// Returns the `BuiltinMethod` corresponding to this reducer operation.
    pub fn method(&self) -> Option<BuiltinMethod> {
        match self.op {
            ReducerOp::Count => Some(BuiltinMethod::Count),
            ReducerOp::Numeric(op) => Some(op.method()),
        }
    }
}
