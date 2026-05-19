//! Operator IR types: `ReducerSpec`, `SortSpec`, `NumOp`, and related enums.
//! Shared across lowering, execution, and the composed substrate.

use std::sync::Arc;

use crate::builtins::registry::{
    arg_extreme_sink as builtin_arg_extreme_sink, membership_sink as builtin_membership_sink,
    numeric_reducer, predicate_sink as builtin_predicate_sink, BuiltinArgExtremeSink, BuiltinId,
    BuiltinMembershipSink, BuiltinPredicateSink,
};
use crate::builtins::BuiltinMethod;
use crate::parse::ast::Expr;
use crate::plan::demand::{Demand, PullDemand, SinkResultDemand, ValueNeed};
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
    pub op: PredicateSinkOp,
    /// Predicate evaluated for each row until the terminal can decide.
    pub predicate: Arc<Program>,
}

/// Specification for value-membership terminal sinks (`includes`, `index`, `indices_of`).
#[derive(Debug, Clone)]
pub struct MembershipSinkSpec {
    /// Terminal operation to perform.
    pub op: MembershipSinkOp,
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
    /// When true, keeps the row with the largest key; otherwise keeps the smallest key.
    pub want_max: bool,
    /// Key expression evaluated for each row.
    pub key: Arc<Program>,
}

/// Predicate terminal operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredicateSinkOp {
    /// Returns true when any row matches the predicate.
    Any,
    /// Returns true when every row matches the predicate.
    All,
    /// Returns the zero-based index of the first matching row, or null.
    FindIndex,
    /// Returns all zero-based indices whose rows match the predicate.
    IndicesWhere,
    /// Returns exactly one matching row, erroring on zero or multiple matches.
    FindOne,
}

impl PredicateSinkOp {
    #[inline]
    fn from_builtin(kind: BuiltinPredicateSink) -> Self {
        match kind {
            BuiltinPredicateSink::Any => Self::Any,
            BuiltinPredicateSink::All => Self::All,
            BuiltinPredicateSink::FindIndex => Self::FindIndex,
            BuiltinPredicateSink::IndicesWhere => Self::IndicesWhere,
            BuiltinPredicateSink::FindOne => Self::FindOne,
        }
    }
}

/// Value-membership terminal operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MembershipSinkOp {
    /// Returns true when any row equals the target.
    Includes,
    /// Returns the zero-based index of the first matching row, or null.
    Index,
    /// Returns all zero-based indices matching the target.
    IndicesOf,
}

impl MembershipSinkOp {
    #[inline]
    fn from_builtin(kind: BuiltinMembershipSink) -> Self {
        match kind {
            BuiltinMembershipSink::Includes => Self::Includes,
            BuiltinMembershipSink::Index => Self::Index,
            BuiltinMembershipSink::IndicesOf => Self::IndicesOf,
        }
    }
}

impl PredicateSinkSpec {
    /// Constructs a predicate terminal sink from the builtin method.
    pub(crate) fn from_method(method: BuiltinMethod, predicate: Arc<Program>) -> Option<Self> {
        Some(Self {
            op: PredicateSinkOp::from_builtin(builtin_predicate_sink(BuiltinId::from_method(
                method,
            ))?),
            predicate,
        })
    }

    /// Demand placed on the row stream by this terminal predicate sink.
    pub(crate) fn demand(&self) -> Demand {
        // Predicate sinks can short-circuit inside their accumulator, but the shared
        // pull model counts source rows emitted by the stage chain. Treating
        // `any`/`find_index` as `UntilOutput(1)` would stop after one non-matching
        // input row before the predicate sink has produced its scalar result.
        Demand {
            pull: PullDemand::All,
            value: if self.op == PredicateSinkOp::FindOne {
                ValueNeed::Whole
            } else {
                ValueNeed::Predicate
            },
            order: false,
        }
    }

    /// Scalar sink-result demand for accumulator-level short-circuit planning.
    pub(crate) fn sink_result_demand(&self) -> SinkResultDemand {
        match self.op {
            PredicateSinkOp::Any | PredicateSinkOp::FindIndex => SinkResultDemand::UntilMatch,
            PredicateSinkOp::All => SinkResultDemand::UntilFailure,
            PredicateSinkOp::IndicesWhere | PredicateSinkOp::FindOne => SinkResultDemand::None,
        }
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
    /// Constructs a membership terminal sink from the builtin method.
    pub(crate) fn from_method(method: BuiltinMethod, target: MembershipSinkTarget) -> Option<Self> {
        Some(Self {
            op: MembershipSinkOp::from_builtin(builtin_membership_sink(BuiltinId::from_method(
                method,
            ))?),
            target,
        })
    }

    /// Demand placed on the row stream by this terminal membership sink.
    pub(crate) fn demand(&self) -> Demand {
        // `includes` and `index` can stop once their accumulator observes a match,
        // but the pull demand is still over input rows, not over the terminal
        // scalar result. Keep the source conservative and let the sink stop the
        // executor loop when it has enough information.
        Demand {
            pull: PullDemand::All,
            value: ValueNeed::Whole,
            order: false,
        }
    }

    /// Scalar sink-result demand for accumulator-level short-circuit planning.
    pub(crate) fn sink_result_demand(&self) -> SinkResultDemand {
        match self.op {
            MembershipSinkOp::Includes | MembershipSinkOp::Index => SinkResultDemand::UntilMatch,
            MembershipSinkOp::IndicesOf => SinkResultDemand::None,
        }
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
    /// Constructs an arg-extreme sink from the terminal builtin method.
    pub(crate) fn from_method(method: BuiltinMethod, key: Arc<Program>) -> Option<Self> {
        Some(Self {
            want_max: match builtin_arg_extreme_sink(BuiltinId::from_method(method))? {
                BuiltinArgExtremeSink::MaxBy => true,
                BuiltinArgExtremeSink::MinBy => false,
            },
            key,
        })
    }

    /// Demand placed on the row stream by this terminal arg-extreme sink.
    pub(crate) fn demand(&self) -> Demand {
        Demand {
            pull: PullDemand::All,
            value: ValueNeed::Whole,
            order: true,
        }
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
    pub fn numeric(
        method: BuiltinMethod,
        projection: Option<Arc<Program>>,
        projection_expr: Option<Arc<Expr>>,
    ) -> Option<Self> {
        Some(Self {
            op: ReducerOp::Numeric(NumOp::from_builtin_reducer(numeric_reducer(
                BuiltinId::from_method(method),
            )?)),
            predicate: None,
            projection,
            predicate_expr: None,
            projection_expr,
        })
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

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_program() -> Arc<Program> {
        Arc::new(Program::new(Vec::new(), "<sink-demand-test>"))
    }

    #[test]
    fn terminal_sink_specs_construct_from_methods() {
        let predicate = PredicateSinkSpec::from_method(BuiltinMethod::Any, empty_program())
            .expect("any predicate sink");
        assert_eq!(predicate.op, PredicateSinkOp::Any);
        assert!(PredicateSinkSpec::from_method(BuiltinMethod::Count, empty_program()).is_none());

        let membership = MembershipSinkSpec::from_method(
            BuiltinMethod::IndicesOf,
            MembershipSinkTarget::Literal(crate::data::value::Val::Int(1)),
        )
        .expect("indices_of membership sink");
        assert_eq!(membership.op, MembershipSinkOp::IndicesOf);
        assert!(MembershipSinkSpec::from_method(
            BuiltinMethod::Count,
            MembershipSinkTarget::Literal(crate::data::value::Val::Int(1)),
        )
        .is_none());

        let max_by = ArgExtremeSinkSpec::from_method(BuiltinMethod::MaxBy, empty_program())
            .expect("max_by arg-extreme sink");
        assert!(max_by.want_max);
        let min_by = ArgExtremeSinkSpec::from_method(BuiltinMethod::MinBy, empty_program())
            .expect("min_by arg-extreme sink");
        assert!(!min_by.want_max);
        assert!(ArgExtremeSinkSpec::from_method(BuiltinMethod::Count, empty_program()).is_none());
    }

    #[test]
    fn predicate_sink_demand_matches_terminal_semantics() {
        let any = PredicateSinkSpec {
            op: PredicateSinkOp::Any,
            predicate: empty_program(),
        }
        .demand();
        assert_eq!(any.pull, PullDemand::All);
        assert_eq!(any.value, ValueNeed::Predicate);
        assert!(!any.order);
        assert_eq!(
            PredicateSinkSpec {
                op: PredicateSinkOp::Any,
                predicate: empty_program(),
            }
            .sink_result_demand(),
            SinkResultDemand::UntilMatch
        );
        assert_eq!(
            PredicateSinkSpec {
                op: PredicateSinkOp::All,
                predicate: empty_program(),
            }
            .sink_result_demand(),
            SinkResultDemand::UntilFailure
        );

        let find_one = PredicateSinkSpec {
            op: PredicateSinkOp::FindOne,
            predicate: empty_program(),
        }
        .demand();
        assert_eq!(find_one.pull, PullDemand::All);
        assert_eq!(find_one.value, ValueNeed::Whole);
        assert!(!find_one.order);
    }

    #[test]
    fn membership_and_arg_extreme_demands_match_terminal_semantics() {
        let membership = MembershipSinkSpec {
            op: MembershipSinkOp::Includes,
            target: MembershipSinkTarget::Literal(crate::data::value::Val::Int(1)),
        }
        .demand();
        assert_eq!(membership.pull, PullDemand::All);
        assert_eq!(membership.value, ValueNeed::Whole);
        assert!(!membership.order);
        assert_eq!(
            MembershipSinkSpec {
                op: MembershipSinkOp::Includes,
                target: MembershipSinkTarget::Literal(crate::data::value::Val::Int(1)),
            }
            .sink_result_demand(),
            SinkResultDemand::UntilMatch
        );

        let arg_extreme = ArgExtremeSinkSpec {
            want_max: true,
            key: empty_program(),
        }
        .demand();
        assert_eq!(arg_extreme.pull, PullDemand::All);
        assert_eq!(arg_extreme.value, ValueNeed::Whole);
        assert!(arg_extreme.order);
    }
}
