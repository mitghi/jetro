//! Operator IR types: `ReducerSpec`, `SortSpec`, `NumOp`, and related enums.
//! Shared across lowering, execution, and the composed substrate.

use std::sync::Arc;

use crate::builtins::registry::{
    arg_extreme_sink as builtin_arg_extreme_sink,
    arg_extreme_sink_demand as builtin_arg_extreme_sink_demand,
    membership_sink as builtin_membership_sink,
    membership_sink_demand as builtin_membership_sink_demand,
    membership_sink_result_demand as builtin_membership_sink_result_demand,
    numeric_reducer, predicate_sink as builtin_predicate_sink,
    predicate_sink_demand as builtin_predicate_sink_demand,
    predicate_sink_result_demand as builtin_predicate_sink_result_demand,
    BuiltinId,
};
use crate::builtins::{
    BuiltinArgExtremeSink, BuiltinMembershipSink, BuiltinMethod, BuiltinPredicateSink,
};
use crate::parse::ast::Expr;
use crate::plan::demand::{Demand, SinkResultDemand};
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

/// Specification for predicate terminal sinks (`any`, `all`, `find_index`, `find_one`).
#[derive(Debug, Clone)]
pub struct PredicateSinkSpec {
    /// Terminal operation to perform.
    pub op: BuiltinPredicateSink,
    /// Predicate evaluated for each row until the terminal can decide.
    pub predicate: Arc<Program>,
    /// Source AST for `predicate`, used during lexical-env analysis.
    pub predicate_expr: Option<Arc<Expr>>,
}

/// Specification for value-membership terminal sinks (`includes`, `index`, `indices_of`).
#[derive(Debug, Clone)]
pub struct MembershipSinkSpec {
    /// Terminal operation to perform.
    pub op: BuiltinMembershipSink,
    /// Value compared against each row.
    pub target: MembershipSinkTarget,
}

/// Target value source for value-membership terminal sinks.
#[derive(Debug, Clone)]
pub enum MembershipSinkTarget {
    /// Literal known during lowering.
    Literal(crate::data::value::Val),
    /// Expression evaluated once before rows are streamed.
    Program(Arc<Program>),
}

/// Specification for arg-extreme terminal sinks (`max_by`, `min_by`).
#[derive(Debug, Clone)]
pub struct ArgExtremeSinkSpec {
    /// Terminal arg-extreme operation.
    pub op: BuiltinArgExtremeSink,
    /// Key expression evaluated for each row.
    pub key: Arc<Program>,
    /// Source AST for `key`, used during lexical-env analysis.
    pub key_expr: Option<Arc<Expr>>,
}

impl PredicateSinkSpec {
    /// Constructs a predicate terminal sink from a resolved builtin id.
    pub(crate) fn from_id(
        id: BuiltinId,
        predicate: Arc<Program>,
        predicate_expr: Option<Arc<Expr>>,
    ) -> Option<Self> {
        Some(Self {
            op: builtin_predicate_sink(id)?,
            predicate,
            predicate_expr,
        })
    }

    /// Constructs a predicate terminal sink from the builtin method.
    #[cfg(test)]
    pub(crate) fn from_method(
        method: BuiltinMethod,
        predicate: Arc<Program>,
        predicate_expr: Option<Arc<Expr>>,
    ) -> Option<Self> {
        Self::from_id(BuiltinId::from_method(method), predicate, predicate_expr)
    }

    /// Demand placed on the row stream by this terminal predicate sink.
    pub(crate) fn demand(&self) -> Demand {
        builtin_predicate_sink_demand(self.op)
    }

    /// Scalar sink-result demand for accumulator-level short-circuit planning.
    pub(crate) fn sink_result_demand(&self) -> SinkResultDemand {
        builtin_predicate_sink_result_demand(self.op)
    }

    /// Returns true for the terminal sink that returns the matching row itself.
    pub(crate) fn is_find_one(&self) -> bool {
        self.op == BuiltinPredicateSink::FindOne
    }

    /// Returns the builtin method represented by this predicate sink.
    pub(crate) fn method(&self) -> BuiltinMethod {
        self.op.method()
    }

    /// Returns the registry id represented by this predicate sink.
    pub(crate) fn id(&self) -> BuiltinId {
        BuiltinId::from_method(self.method())
    }

    /// Iterates over embedded programs for kernel enumeration.
    pub(crate) fn sink_programs(&self) -> impl Iterator<Item = &Arc<Program>> {
        std::iter::once(&self.predicate)
    }

    /// Returns the sink-kernel index for the predicate.
    pub(crate) fn predicate_kernel_index(&self) -> usize {
        0
    }
}

impl MembershipSinkSpec {
    /// Constructs a membership terminal sink from a resolved builtin id.
    pub(crate) fn from_id(id: BuiltinId, target: MembershipSinkTarget) -> Option<Self> {
        Some(Self {
            op: builtin_membership_sink(id)?,
            target,
        })
    }

    /// Constructs a membership terminal sink from the builtin method.
    #[cfg(test)]
    pub(crate) fn from_method(method: BuiltinMethod, target: MembershipSinkTarget) -> Option<Self> {
        Self::from_id(BuiltinId::from_method(method), target)
    }

    /// Demand placed on the row stream by this terminal membership sink.
    pub(crate) fn demand(&self) -> Demand {
        builtin_membership_sink_demand(self.op)
    }

    /// Scalar sink-result demand for accumulator-level short-circuit planning.
    pub(crate) fn sink_result_demand(&self) -> SinkResultDemand {
        builtin_membership_sink_result_demand(self.op)
    }

    /// Returns true for the boolean membership sink.
    pub(crate) fn is_includes(&self) -> bool {
        self.op == BuiltinMembershipSink::Includes
    }

    /// Returns the builtin method represented by this membership sink.
    pub(crate) fn method(&self) -> BuiltinMethod {
        self.op.method()
    }

    /// Returns the registry id represented by this membership sink.
    pub(crate) fn id(&self) -> BuiltinId {
        BuiltinId::from_method(self.method())
    }

    /// Iterates over embedded programs for kernel enumeration.
    pub(crate) fn sink_programs(&self) -> impl Iterator<Item = &Arc<Program>> {
        match &self.target {
            MembershipSinkTarget::Literal(_) => None,
            MembershipSinkTarget::Program(program) => Some(program),
        }
        .into_iter()
    }
}

impl ArgExtremeSinkSpec {
    /// Constructs an arg-extreme sink from a resolved builtin id.
    pub(crate) fn from_id(
        id: BuiltinId,
        key: Arc<Program>,
        key_expr: Option<Arc<Expr>>,
    ) -> Option<Self> {
        Some(Self {
            op: builtin_arg_extreme_sink(id)?,
            key,
            key_expr,
        })
    }

    /// Constructs an arg-extreme sink from the terminal builtin method.
    #[cfg(test)]
    pub(crate) fn from_method(
        method: BuiltinMethod,
        key: Arc<Program>,
        key_expr: Option<Arc<Expr>>,
    ) -> Option<Self> {
        Self::from_id(BuiltinId::from_method(method), key, key_expr)
    }

    /// Demand placed on the row stream by this terminal arg-extreme sink.
    pub(crate) fn demand(&self) -> Demand {
        builtin_arg_extreme_sink_demand(self.op)
    }

    /// Returns true when this sink keeps the row with the largest projected key.
    pub(crate) fn wants_max(&self) -> bool {
        self.op.wants_max()
    }

    /// Returns the builtin method represented by this arg-extreme sink.
    pub(crate) fn method(&self) -> BuiltinMethod {
        self.op.method()
    }

    /// Returns the registry id represented by this arg-extreme sink.
    pub(crate) fn id(&self) -> BuiltinId {
        BuiltinId::from_method(self.method())
    }

    /// Iterates over embedded programs for kernel enumeration.
    pub(crate) fn sink_programs(&self) -> impl Iterator<Item = &Arc<Program>> {
        std::iter::once(&self.key)
    }

    /// Returns the sink-kernel index for the key projection.
    pub(crate) fn key_kernel_index(&self) -> usize {
        0
    }
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

    /// Constructs a `Count` reducer gated by a predicate expression.
    pub fn count_with_predicate(
        predicate: Arc<Program>,
        predicate_expr: Option<Arc<Expr>>,
    ) -> Self {
        Self {
            op: ReducerOp::Count,
            predicate: Some(predicate),
            projection: None,
            predicate_expr,
            projection_expr: None,
        }
    }

    /// Constructs a numeric reducer from builtin metadata.
    pub fn numeric_id(
        id: BuiltinId,
        projection: Option<Arc<Program>>,
        projection_expr: Option<Arc<Expr>>,
    ) -> Option<Self> {
        Some(Self {
            op: ReducerOp::Numeric(NumOp::from_builtin_reducer(numeric_reducer(id)?)),
            predicate: None,
            projection,
            predicate_expr: None,
            projection_expr,
        })
    }

    /// Constructs a numeric reducer from builtin metadata.
    #[cfg(test)]
    pub fn numeric(
        method: BuiltinMethod,
        projection: Option<Arc<Program>>,
        projection_expr: Option<Arc<Expr>>,
    ) -> Option<Self> {
        Self::numeric_id(BuiltinId::from_method(method), projection, projection_expr)
    }

    /// Returns the `NumOp` for a `Numeric` reducer, or `None` for `Count`.
    pub fn numeric_op(&self) -> Option<NumOp> {
        match self.op {
            ReducerOp::Numeric(op) => Some(op),
            ReducerOp::Count => None,
        }
    }

    /// Returns true for any count reducer, including predicate-bearing count sinks.
    pub(crate) fn is_count(&self) -> bool {
        self.op == ReducerOp::Count
    }

    /// Returns true for a count reducer with no predicate or projection.
    pub(crate) fn is_plain_count(&self) -> bool {
        self.is_count() && self.predicate.is_none() && self.projection.is_none()
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
    #[cfg(test)]
    pub fn method(&self) -> Option<BuiltinMethod> {
        match self.op {
            ReducerOp::Count => Some(BuiltinMethod::Count),
            ReducerOp::Numeric(op) => Some(op.method()),
        }
    }

    /// Returns the registry id corresponding to this reducer operation.
    pub(crate) fn id(&self) -> Option<BuiltinId> {
        match self.op {
            ReducerOp::Count => Some(BuiltinId::COUNT),
            ReducerOp::Numeric(op) => Some(BuiltinId::from_method(op.method())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::demand::{PullDemand, ValueNeed};

    fn empty_program() -> Arc<Program> {
        Arc::new(Program::new(Vec::new(), "<sink-demand-test>"))
    }

    #[test]
    fn terminal_sink_specs_construct_from_methods() {
        let predicate = PredicateSinkSpec::from_method(BuiltinMethod::Any, empty_program(), None)
            .expect("any predicate sink");
        assert_eq!(predicate.op, BuiltinPredicateSink::Any);
        assert!(
            PredicateSinkSpec::from_method(BuiltinMethod::Count, empty_program(), None).is_none()
        );

        let membership = MembershipSinkSpec::from_method(
            BuiltinMethod::IndicesOf,
            MembershipSinkTarget::Literal(crate::data::value::Val::Int(1)),
        )
        .expect("indices_of membership sink");
        assert_eq!(membership.op, BuiltinMembershipSink::IndicesOf);
        assert!(MembershipSinkSpec::from_method(
            BuiltinMethod::Count,
            MembershipSinkTarget::Literal(crate::data::value::Val::Int(1)),
        )
        .is_none());

        let max_by = ArgExtremeSinkSpec::from_method(BuiltinMethod::MaxBy, empty_program(), None)
            .expect("max_by arg-extreme sink");
        assert_eq!(max_by.op, BuiltinArgExtremeSink::MaxBy);
        assert_eq!(max_by.method(), BuiltinMethod::MaxBy);
        assert!(max_by.wants_max());
        let min_by = ArgExtremeSinkSpec::from_method(BuiltinMethod::MinBy, empty_program(), None)
            .expect("min_by arg-extreme sink");
        assert_eq!(min_by.op, BuiltinArgExtremeSink::MinBy);
        assert_eq!(min_by.method(), BuiltinMethod::MinBy);
        assert!(!min_by.wants_max());
        assert!(
            ArgExtremeSinkSpec::from_method(BuiltinMethod::Count, empty_program(), None).is_none()
        );
    }

    #[test]
    fn reducer_spec_classifies_plain_count() {
        assert!(ReducerSpec::count().is_plain_count());
        assert!(!ReducerSpec::count_with_predicate(empty_program(), None).is_plain_count());
        assert!(!ReducerSpec::numeric(BuiltinMethod::Sum, None, None)
            .unwrap()
            .is_plain_count());
    }

    #[test]
    fn predicate_sink_demand_matches_terminal_semantics() {
        let any = PredicateSinkSpec {
            op: BuiltinPredicateSink::Any,
            predicate: empty_program(),
            predicate_expr: None,
        }
        .demand();
        assert_eq!(any.pull, PullDemand::All);
        assert_eq!(any.value, ValueNeed::Predicate);
        assert!(!any.order);
        assert_eq!(
            PredicateSinkSpec {
                op: BuiltinPredicateSink::Any,
                predicate: empty_program(),
                predicate_expr: None,
            }
            .sink_result_demand(),
            SinkResultDemand::UntilMatch
        );
        assert_eq!(
            PredicateSinkSpec {
                op: BuiltinPredicateSink::All,
                predicate: empty_program(),
                predicate_expr: None,
            }
            .sink_result_demand(),
            SinkResultDemand::UntilFailure
        );

        let find_one = PredicateSinkSpec {
            op: BuiltinPredicateSink::FindOne,
            predicate: empty_program(),
            predicate_expr: None,
        }
        .demand();
        assert_eq!(find_one.pull, PullDemand::All);
        assert_eq!(find_one.value, ValueNeed::Whole);
        assert!(!find_one.order);
    }

    #[test]
    fn membership_and_arg_extreme_demands_match_terminal_semantics() {
        let membership = MembershipSinkSpec {
            op: BuiltinMembershipSink::Includes,
            target: MembershipSinkTarget::Literal(crate::data::value::Val::Int(1)),
        }
        .demand();
        assert_eq!(membership.pull, PullDemand::All);
        assert_eq!(membership.value, ValueNeed::Whole);
        assert!(!membership.order);
        assert_eq!(
            MembershipSinkSpec {
                op: BuiltinMembershipSink::Includes,
                target: MembershipSinkTarget::Literal(crate::data::value::Val::Int(1)),
            }
            .sink_result_demand(),
            SinkResultDemand::UntilMatch
        );

        let arg_extreme = ArgExtremeSinkSpec {
            op: BuiltinArgExtremeSink::MaxBy,
            key: empty_program(),
            key_expr: None,
        }
        .demand();
        assert_eq!(arg_extreme.pull, PullDemand::All);
        assert_eq!(arg_extreme.value, ValueNeed::Whole);
        assert!(arg_extreme.order);
    }
}
