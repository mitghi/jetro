//! Logical plan IR — a tree-shaped, uncompiled representation of a pipeline query.
//!
//! `LogicalPlan` sits between the `Expr` AST (produced by the parser) and the
//! compiled `Pipeline` struct. All stage bodies are `Expr` nodes; compilation to
//! `Arc<Program>` happens only in the lowering pass after optimizer rules have
//! been applied.

use crate::builtins::BuiltinMethod;
use crate::parse::ast::Expr;
use crate::exec::pipeline::{SortSpec, Source};

/// A tree-shaped, uncompiled pipeline plan.
///
/// Each node carries `Expr` bodies; compilation to `Arc<Program>` happens only
/// in the lowering pass after all optimizer rules have been applied.
#[derive(Debug, Clone)]
pub(crate) enum LogicalPlan {
    /// Row source (document root, field path, etc.).
    Source(Source),

    Filter     { input: Box<Self>, predicate: Expr },
    Map        { input: Box<Self>, projection: Expr },
    FlatMap    { input: Box<Self>, expansion: Expr },
    TakeWhile  { input: Box<Self>, predicate: Expr },
    DropWhile  { input: Box<Self>, predicate: Expr },

    Take { input: Box<Self>, n: usize },
    Skip { input: Box<Self>, n: usize },

    Sort   { input: Box<Self>, spec: SortSpec },
    Unique { input: Box<Self>, key: Option<Expr> },
    Reverse { input: Box<Self> },

    GroupBy  { input: Box<Self>, key: Expr },
    CountBy  { input: Box<Self>, key: Expr },
    IndexBy  { input: Box<Self>, key: Expr },

    First  (Box<Self>),
    Last   (Box<Self>),
    Sum    (Box<Self>),
    Avg    (Box<Self>),
    Min    (Box<Self>),
    Max    (Box<Self>),
    Count  (Box<Self>),
    ApproxCountDistinct(Box<Self>),

    /// Any expression the logical planner could not classify; sentinel for the fallback path.
    ScalarExpr,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::pipeline::Source;

    #[test]
    fn logical_nodes_expose_canonical_builtin_methods() {
        assert_eq!(
            LogicalPlan::Filter {
                input: Box::new(LogicalPlan::ScalarExpr),
                predicate: Expr::Bool(true),
            }
            .builtin_method(),
            Some(BuiltinMethod::Filter)
        );
        assert_eq!(
            LogicalPlan::Unique {
                input: Box::new(LogicalPlan::ScalarExpr),
                key: Some(Expr::Current),
            }
            .builtin_method(),
            Some(BuiltinMethod::UniqueBy)
        );
        assert_eq!(
            LogicalPlan::First(Box::new(LogicalPlan::ScalarExpr)).builtin_method(),
            Some(BuiltinMethod::First)
        );
        assert_eq!(
            LogicalPlan::Source(Source::Receiver(crate::data::value::Val::Null))
                .builtin_method(),
            None
        );
    }
}

impl LogicalPlan {
    /// Returns the canonical builtin method represented by this logical node,
    /// if the node maps to a pipeline stage or terminal sink.
    pub(crate) fn builtin_method(&self) -> Option<BuiltinMethod> {
        Some(match self {
            LogicalPlan::Filter { .. } => BuiltinMethod::Filter,
            LogicalPlan::Map { .. } => BuiltinMethod::Map,
            LogicalPlan::FlatMap { .. } => BuiltinMethod::FlatMap,
            LogicalPlan::TakeWhile { .. } => BuiltinMethod::TakeWhile,
            LogicalPlan::DropWhile { .. } => BuiltinMethod::DropWhile,
            LogicalPlan::Take { .. } => BuiltinMethod::Take,
            LogicalPlan::Skip { .. } => BuiltinMethod::Skip,
            LogicalPlan::Sort { .. } => BuiltinMethod::Sort,
            LogicalPlan::Unique { key: None, .. } => BuiltinMethod::Unique,
            LogicalPlan::Unique { key: Some(_), .. } => BuiltinMethod::UniqueBy,
            LogicalPlan::Reverse { .. } => BuiltinMethod::Reverse,
            LogicalPlan::GroupBy { .. } => BuiltinMethod::GroupBy,
            LogicalPlan::CountBy { .. } => BuiltinMethod::CountBy,
            LogicalPlan::IndexBy { .. } => BuiltinMethod::IndexBy,
            LogicalPlan::First(_) => BuiltinMethod::First,
            LogicalPlan::Last(_) => BuiltinMethod::Last,
            LogicalPlan::Sum(_) => BuiltinMethod::Sum,
            LogicalPlan::Avg(_) => BuiltinMethod::Avg,
            LogicalPlan::Min(_) => BuiltinMethod::Min,
            LogicalPlan::Max(_) => BuiltinMethod::Max,
            LogicalPlan::Count(_) => BuiltinMethod::Count,
            LogicalPlan::ApproxCountDistinct(_) => BuiltinMethod::ApproxCountDistinct,
            LogicalPlan::Source(_) | LogicalPlan::ScalarExpr => return None,
        })
    }

    /// Consumes `self` and returns `(input, node_without_input)` for use in rewrites.
    /// Returns `Err(self)` for `Source` and `ScalarExpr`.
    pub(crate) fn take_input(self) -> Result<(Box<LogicalPlan>, LogicalPlan), LogicalPlan> {
        match self {
            LogicalPlan::Source(_) | LogicalPlan::ScalarExpr => Err(self),

            LogicalPlan::Filter { input, predicate } =>
                Ok((input, LogicalPlan::Filter { input: Box::new(LogicalPlan::ScalarExpr), predicate })),
            LogicalPlan::Map { input, projection } =>
                Ok((input, LogicalPlan::Map { input: Box::new(LogicalPlan::ScalarExpr), projection })),
            LogicalPlan::FlatMap { input, expansion } =>
                Ok((input, LogicalPlan::FlatMap { input: Box::new(LogicalPlan::ScalarExpr), expansion })),
            LogicalPlan::TakeWhile { input, predicate } =>
                Ok((input, LogicalPlan::TakeWhile { input: Box::new(LogicalPlan::ScalarExpr), predicate })),
            LogicalPlan::DropWhile { input, predicate } =>
                Ok((input, LogicalPlan::DropWhile { input: Box::new(LogicalPlan::ScalarExpr), predicate })),
            LogicalPlan::Take { input, n } =>
                Ok((input, LogicalPlan::Take { input: Box::new(LogicalPlan::ScalarExpr), n })),
            LogicalPlan::Skip { input, n } =>
                Ok((input, LogicalPlan::Skip { input: Box::new(LogicalPlan::ScalarExpr), n })),
            LogicalPlan::Sort { input, spec } =>
                Ok((input, LogicalPlan::Sort { input: Box::new(LogicalPlan::ScalarExpr), spec })),
            LogicalPlan::Unique { input, key } =>
                Ok((input, LogicalPlan::Unique { input: Box::new(LogicalPlan::ScalarExpr), key })),
            LogicalPlan::Reverse { input } =>
                Ok((input, LogicalPlan::Reverse { input: Box::new(LogicalPlan::ScalarExpr) })),
            LogicalPlan::GroupBy { input, key } =>
                Ok((input, LogicalPlan::GroupBy { input: Box::new(LogicalPlan::ScalarExpr), key })),
            LogicalPlan::CountBy { input, key } =>
                Ok((input, LogicalPlan::CountBy { input: Box::new(LogicalPlan::ScalarExpr), key })),
            LogicalPlan::IndexBy { input, key } =>
                Ok((input, LogicalPlan::IndexBy { input: Box::new(LogicalPlan::ScalarExpr), key })),

            LogicalPlan::First(inner) =>
                Ok((inner, LogicalPlan::First(Box::new(LogicalPlan::ScalarExpr)))),
            LogicalPlan::Last(inner) =>
                Ok((inner, LogicalPlan::Last(Box::new(LogicalPlan::ScalarExpr)))),
            LogicalPlan::Sum(inner) =>
                Ok((inner, LogicalPlan::Sum(Box::new(LogicalPlan::ScalarExpr)))),
            LogicalPlan::Avg(inner) =>
                Ok((inner, LogicalPlan::Avg(Box::new(LogicalPlan::ScalarExpr)))),
            LogicalPlan::Min(inner) =>
                Ok((inner, LogicalPlan::Min(Box::new(LogicalPlan::ScalarExpr)))),
            LogicalPlan::Max(inner) =>
                Ok((inner, LogicalPlan::Max(Box::new(LogicalPlan::ScalarExpr)))),
            LogicalPlan::Count(inner) =>
                Ok((inner, LogicalPlan::Count(Box::new(LogicalPlan::ScalarExpr)))),
            LogicalPlan::ApproxCountDistinct(inner) =>
                Ok((inner, LogicalPlan::ApproxCountDistinct(Box::new(LogicalPlan::ScalarExpr)))),
        }
    }

    /// Replaces the input sub-plan, returning a new node with the same shape.
    /// Panics for `Source` and `ScalarExpr`.
    pub(crate) fn with_input(self, new_input: LogicalPlan) -> LogicalPlan {
        let new_box = Box::new(new_input);
        match self {
            LogicalPlan::Source(_) | LogicalPlan::ScalarExpr =>
                panic!("with_input called on leaf node"),

            LogicalPlan::Filter { predicate, .. } =>
                LogicalPlan::Filter { input: new_box, predicate },
            LogicalPlan::Map { projection, .. } =>
                LogicalPlan::Map { input: new_box, projection },
            LogicalPlan::FlatMap { expansion, .. } =>
                LogicalPlan::FlatMap { input: new_box, expansion },
            LogicalPlan::TakeWhile { predicate, .. } =>
                LogicalPlan::TakeWhile { input: new_box, predicate },
            LogicalPlan::DropWhile { predicate, .. } =>
                LogicalPlan::DropWhile { input: new_box, predicate },
            LogicalPlan::Take { n, .. } =>
                LogicalPlan::Take { input: new_box, n },
            LogicalPlan::Skip { n, .. } =>
                LogicalPlan::Skip { input: new_box, n },
            LogicalPlan::Sort { spec, .. } =>
                LogicalPlan::Sort { input: new_box, spec },
            LogicalPlan::Unique { key, .. } =>
                LogicalPlan::Unique { input: new_box, key },
            LogicalPlan::Reverse { .. } =>
                LogicalPlan::Reverse { input: new_box },
            LogicalPlan::GroupBy { key, .. } =>
                LogicalPlan::GroupBy { input: new_box, key },
            LogicalPlan::CountBy { key, .. } =>
                LogicalPlan::CountBy { input: new_box, key },
            LogicalPlan::IndexBy { key, .. } =>
                LogicalPlan::IndexBy { input: new_box, key },

            LogicalPlan::First(_)   => LogicalPlan::First(new_box),
            LogicalPlan::Last(_)    => LogicalPlan::Last(new_box),
            LogicalPlan::Sum(_)     => LogicalPlan::Sum(new_box),
            LogicalPlan::Avg(_)     => LogicalPlan::Avg(new_box),
            LogicalPlan::Min(_)     => LogicalPlan::Min(new_box),
            LogicalPlan::Max(_)     => LogicalPlan::Max(new_box),
            LogicalPlan::Count(_)   => LogicalPlan::Count(new_box),
            LogicalPlan::ApproxCountDistinct(_) => LogicalPlan::ApproxCountDistinct(new_box),
        }
    }
}
