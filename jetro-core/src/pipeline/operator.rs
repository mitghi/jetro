//! Operator IR types: `ReducerSpec`, `SortSpec`, `NumOp`, and related enums.
//! Shared across lowering, execution, and the composed substrate.

use std::sync::Arc;

use crate::ast::Expr;
use crate::builtins::BuiltinMethod;
use crate::vm::Program;

use super::NumOp;

/// Specification for a terminal reducer sink (`count`, `sum`, `avg`, `min`,
/// `max`). Optionally carries a predicate guard and a projection expression
/// that are applied to each row before accumulation.
#[derive(Debug, Clone)]
pub struct ReducerSpec {
    /// The aggregation operation to perform.
    pub op: ReducerOp,
    /// Optional compiled predicate that gates which rows are counted/aggregated.
    pub predicate: Option<Arc<Program>>,
    /// Optional compiled projection applied to each row before aggregation.
    pub projection: Option<Arc<Program>>,
    /// Source AST for `predicate`, used during IR analysis phases.
    pub predicate_expr: Option<Arc<Expr>>,
    /// Source AST for `projection`, used during IR analysis phases.
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

    /// Returns the underlying `NumOp` when the reducer is a `Numeric` variant,
    /// or `None` for a `Count` reducer.
    pub fn numeric_op(&self) -> Option<NumOp> {
        match self.op {
            ReducerOp::Numeric(op) => Some(op),
            ReducerOp::Count => None,
        }
    }

    /// Iterates over the compiled programs embedded in this spec (predicate first,
    /// then projection), used to enumerate all kernel programs for a sink.
    pub(crate) fn sink_programs(&self) -> impl Iterator<Item = &Arc<Program>> {
        self.predicate.iter().chain(self.projection.iter())
    }

    /// Returns the sink-kernel slice index for the predicate program (`0` when
    /// present), or `None` when this spec has no predicate.
    pub(crate) fn predicate_kernel_index(&self) -> Option<usize> {
        self.predicate.as_ref().map(|_| 0)
    }

    /// Returns the sink-kernel slice index for the projection program (`0` when
    /// there is no predicate, `1` when there is), or `None` when no projection exists.
    pub(crate) fn projection_kernel_index(&self) -> Option<usize> {
        self.projection
            .as_ref()
            .map(|_| usize::from(self.predicate.is_some()))
    }

    /// Returns the `BuiltinMethod` that corresponds to this reducer operation,
    /// used when lowering the sink to a builtin call in the physical plan.
    pub fn method(&self) -> Option<BuiltinMethod> {
        match self.op {
            ReducerOp::Count => Some(BuiltinMethod::Count),
            ReducerOp::Numeric(op) => Some(op.method()),
        }
    }
}
