//! Logical plan IR — a tree-shaped, uncompiled representation of a pipeline query.
//!
//! `LogicalPlan` sits between the `Expr` AST (produced by the parser) and the
//! compiled `Pipeline` struct. All stage bodies are `Expr` nodes; compilation to
//! `Arc<Program>` happens only in the lowering pass after optimizer rules have
//! been applied.

use crate::builtins::registry::BuiltinId;
#[cfg(test)]
use crate::builtins::BuiltinMethod;
use crate::exec::pipeline::{SortSpec, Source};
use crate::parse::ast::Expr;

/// A tree-shaped, uncompiled pipeline plan.
///
/// Each node carries `Expr` bodies; compilation to `Arc<Program>` happens only
/// in the lowering pass after all optimizer rules have been applied.
#[derive(Debug, Clone)]
pub(crate) enum LogicalPlan {
    /// Row source (document root, field path, etc.).
    Source(Source),

    Filter {
        input: Box<Self>,
        predicate: Expr,
    },
    Map {
        input: Box<Self>,
        projection: Expr,
    },
    FlatMap {
        input: Box<Self>,
        expansion: Expr,
    },
    TakeWhile {
        input: Box<Self>,
        predicate: Expr,
    },
    DropWhile {
        input: Box<Self>,
        predicate: Expr,
    },

    Take {
        input: Box<Self>,
        n: usize,
    },
    Skip {
        input: Box<Self>,
        n: usize,
    },

    Sort {
        input: Box<Self>,
        spec: SortSpec,
    },
    Unique {
        input: Box<Self>,
        key: Option<Expr>,
    },
    Reverse {
        input: Box<Self>,
    },

    GroupBy {
        input: Box<Self>,
        key: Expr,
    },
    CountBy {
        input: Box<Self>,
        key: Expr,
    },
    IndexBy {
        input: Box<Self>,
        key: Expr,
    },

    First(Box<Self>),
    Last(Box<Self>),
    Sum(Box<Self>),
    Avg(Box<Self>),
    Min(Box<Self>),
    Max(Box<Self>),
    Count(Box<Self>),
    ApproxCountDistinct(Box<Self>),

    /// Any expression the logical planner could not classify; sentinel for the fallback path.
    ScalarExpr,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::{
        registry::{builtin_sink, logical_shape, pipeline_lowering, BuiltinId},
        BuiltinLogicalShape,
    };
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
            LogicalPlan::Source(Source::Receiver(crate::data::value::Val::Null)).builtin_method(),
            None
        );
    }

    #[test]
    fn logical_nodes_match_registry_shapes_and_execution_metadata() {
        fn assert_node(
            plan: LogicalPlan,
            method: BuiltinMethod,
            shape: BuiltinLogicalShape,
            expects_sink: bool,
        ) {
            let id = BuiltinId::from_method(method);
            assert_eq!(plan.builtin_method(), Some(method), "{method:?}");
            assert_eq!(logical_shape(id), Some(shape), "{method:?}");
            assert!(
                pipeline_lowering(id).is_some(),
                "{method:?} must carry pipeline lowering metadata"
            );
            assert_eq!(
                builtin_sink(id).is_some(),
                expects_sink,
                "{method:?} sink metadata mismatch"
            );
        }

        let input = || Box::new(LogicalPlan::ScalarExpr);
        assert_node(
            LogicalPlan::Filter {
                input: input(),
                predicate: Expr::Bool(true),
            },
            BuiltinMethod::Filter,
            BuiltinLogicalShape::Filter,
            false,
        );
        assert_node(
            LogicalPlan::Map {
                input: input(),
                projection: Expr::Current,
            },
            BuiltinMethod::Map,
            BuiltinLogicalShape::Map,
            false,
        );
        assert_node(
            LogicalPlan::FlatMap {
                input: input(),
                expansion: Expr::Current,
            },
            BuiltinMethod::FlatMap,
            BuiltinLogicalShape::FlatMap,
            false,
        );
        assert_node(
            LogicalPlan::TakeWhile {
                input: input(),
                predicate: Expr::Bool(true),
            },
            BuiltinMethod::TakeWhile,
            BuiltinLogicalShape::TakeWhile,
            false,
        );
        assert_node(
            LogicalPlan::DropWhile {
                input: input(),
                predicate: Expr::Bool(true),
            },
            BuiltinMethod::DropWhile,
            BuiltinLogicalShape::DropWhile,
            false,
        );
        assert_node(
            LogicalPlan::Take {
                input: input(),
                n: 1,
            },
            BuiltinMethod::Take,
            BuiltinLogicalShape::Take,
            false,
        );
        assert_node(
            LogicalPlan::Skip {
                input: input(),
                n: 1,
            },
            BuiltinMethod::Skip,
            BuiltinLogicalShape::Skip,
            false,
        );
        assert_node(
            LogicalPlan::Sort {
                input: input(),
                spec: crate::exec::pipeline::SortSpec::identity(),
            },
            BuiltinMethod::Sort,
            BuiltinLogicalShape::Sort,
            false,
        );
        assert_node(
            LogicalPlan::Unique {
                input: input(),
                key: None,
            },
            BuiltinMethod::Unique,
            BuiltinLogicalShape::Unique,
            false,
        );
        assert_node(
            LogicalPlan::Unique {
                input: input(),
                key: Some(Expr::Current),
            },
            BuiltinMethod::UniqueBy,
            BuiltinLogicalShape::UniqueBy,
            false,
        );
        assert_node(
            LogicalPlan::Reverse { input: input() },
            BuiltinMethod::Reverse,
            BuiltinLogicalShape::Reverse,
            false,
        );
        assert_node(
            LogicalPlan::GroupBy {
                input: input(),
                key: Expr::Current,
            },
            BuiltinMethod::GroupBy,
            BuiltinLogicalShape::GroupBy,
            false,
        );
        assert_node(
            LogicalPlan::CountBy {
                input: input(),
                key: Expr::Current,
            },
            BuiltinMethod::CountBy,
            BuiltinLogicalShape::CountBy,
            false,
        );
        assert_node(
            LogicalPlan::IndexBy {
                input: input(),
                key: Expr::Current,
            },
            BuiltinMethod::IndexBy,
            BuiltinLogicalShape::IndexBy,
            false,
        );
        assert_node(
            LogicalPlan::First(input()),
            BuiltinMethod::First,
            BuiltinLogicalShape::First,
            true,
        );
        assert_node(
            LogicalPlan::Last(input()),
            BuiltinMethod::Last,
            BuiltinLogicalShape::Last,
            true,
        );
        assert_node(
            LogicalPlan::Sum(input()),
            BuiltinMethod::Sum,
            BuiltinLogicalShape::Sum,
            true,
        );
        assert_node(
            LogicalPlan::Avg(input()),
            BuiltinMethod::Avg,
            BuiltinLogicalShape::Avg,
            true,
        );
        assert_node(
            LogicalPlan::Min(input()),
            BuiltinMethod::Min,
            BuiltinLogicalShape::Min,
            true,
        );
        assert_node(
            LogicalPlan::Max(input()),
            BuiltinMethod::Max,
            BuiltinLogicalShape::Max,
            true,
        );
        assert_node(
            LogicalPlan::Count(input()),
            BuiltinMethod::Count,
            BuiltinLogicalShape::Count,
            true,
        );
        assert_node(
            LogicalPlan::ApproxCountDistinct(input()),
            BuiltinMethod::ApproxCountDistinct,
            BuiltinLogicalShape::ApproxCountDistinct,
            true,
        );
    }
}

impl LogicalPlan {
    /// Returns the canonical builtin method represented by this logical node,
    /// if the node maps to a pipeline stage or terminal sink.
    #[cfg(test)]
    pub(crate) fn builtin_method(&self) -> Option<BuiltinMethod> {
        self.builtin_id()?.method()
    }

    /// Returns the canonical builtin id represented by this logical node,
    /// if the node maps to a pipeline stage or terminal sink.
    pub(crate) fn builtin_id(&self) -> Option<BuiltinId> {
        let id = match self {
            LogicalPlan::Filter { .. } => BuiltinId::FILTER,
            LogicalPlan::Map { .. } => BuiltinId::MAP,
            LogicalPlan::FlatMap { .. } => BuiltinId::FLAT_MAP,
            LogicalPlan::TakeWhile { .. } => BuiltinId::TAKE_WHILE,
            LogicalPlan::DropWhile { .. } => BuiltinId::DROP_WHILE,
            LogicalPlan::Take { .. } => BuiltinId::TAKE,
            LogicalPlan::Skip { .. } => BuiltinId::SKIP,
            LogicalPlan::Sort { .. } => BuiltinId::SORT,
            LogicalPlan::Unique { key: None, .. } => BuiltinId::UNIQUE,
            LogicalPlan::Unique { key: Some(_), .. } => BuiltinId::UNIQUE_BY,
            LogicalPlan::Reverse { .. } => BuiltinId::REVERSE,
            LogicalPlan::GroupBy { .. } => BuiltinId::GROUP_BY,
            LogicalPlan::CountBy { .. } => BuiltinId::COUNT_BY,
            LogicalPlan::IndexBy { .. } => BuiltinId::INDEX_BY,
            LogicalPlan::First(_) => BuiltinId::FIRST,
            LogicalPlan::Last(_) => BuiltinId::LAST,
            LogicalPlan::Sum(_) => BuiltinId::SUM,
            LogicalPlan::Avg(_) => BuiltinId::AVG,
            LogicalPlan::Min(_) => BuiltinId::MIN,
            LogicalPlan::Max(_) => BuiltinId::MAX,
            LogicalPlan::Count(_) => BuiltinId::COUNT,
            LogicalPlan::ApproxCountDistinct(_) => BuiltinId::APPROX_COUNT_DISTINCT,
            LogicalPlan::Source(_) | LogicalPlan::ScalarExpr => return None,
        };
        Some(id)
    }

    /// Consumes `self` and returns `(input, node_without_input)` for use in rewrites.
    /// Returns `Err(self)` for `Source` and `ScalarExpr`.
    pub(crate) fn take_input(self) -> Result<(Box<LogicalPlan>, LogicalPlan), LogicalPlan> {
        match self {
            LogicalPlan::Source(_) | LogicalPlan::ScalarExpr => Err(self),

            LogicalPlan::Filter { input, predicate } => Ok((
                input,
                LogicalPlan::Filter {
                    input: Box::new(LogicalPlan::ScalarExpr),
                    predicate,
                },
            )),
            LogicalPlan::Map { input, projection } => Ok((
                input,
                LogicalPlan::Map {
                    input: Box::new(LogicalPlan::ScalarExpr),
                    projection,
                },
            )),
            LogicalPlan::FlatMap { input, expansion } => Ok((
                input,
                LogicalPlan::FlatMap {
                    input: Box::new(LogicalPlan::ScalarExpr),
                    expansion,
                },
            )),
            LogicalPlan::TakeWhile { input, predicate } => Ok((
                input,
                LogicalPlan::TakeWhile {
                    input: Box::new(LogicalPlan::ScalarExpr),
                    predicate,
                },
            )),
            LogicalPlan::DropWhile { input, predicate } => Ok((
                input,
                LogicalPlan::DropWhile {
                    input: Box::new(LogicalPlan::ScalarExpr),
                    predicate,
                },
            )),
            LogicalPlan::Take { input, n } => Ok((
                input,
                LogicalPlan::Take {
                    input: Box::new(LogicalPlan::ScalarExpr),
                    n,
                },
            )),
            LogicalPlan::Skip { input, n } => Ok((
                input,
                LogicalPlan::Skip {
                    input: Box::new(LogicalPlan::ScalarExpr),
                    n,
                },
            )),
            LogicalPlan::Sort { input, spec } => Ok((
                input,
                LogicalPlan::Sort {
                    input: Box::new(LogicalPlan::ScalarExpr),
                    spec,
                },
            )),
            LogicalPlan::Unique { input, key } => Ok((
                input,
                LogicalPlan::Unique {
                    input: Box::new(LogicalPlan::ScalarExpr),
                    key,
                },
            )),
            LogicalPlan::Reverse { input } => Ok((
                input,
                LogicalPlan::Reverse {
                    input: Box::new(LogicalPlan::ScalarExpr),
                },
            )),
            LogicalPlan::GroupBy { input, key } => Ok((
                input,
                LogicalPlan::GroupBy {
                    input: Box::new(LogicalPlan::ScalarExpr),
                    key,
                },
            )),
            LogicalPlan::CountBy { input, key } => Ok((
                input,
                LogicalPlan::CountBy {
                    input: Box::new(LogicalPlan::ScalarExpr),
                    key,
                },
            )),
            LogicalPlan::IndexBy { input, key } => Ok((
                input,
                LogicalPlan::IndexBy {
                    input: Box::new(LogicalPlan::ScalarExpr),
                    key,
                },
            )),

            LogicalPlan::First(inner) => {
                Ok((inner, LogicalPlan::First(Box::new(LogicalPlan::ScalarExpr))))
            }
            LogicalPlan::Last(inner) => {
                Ok((inner, LogicalPlan::Last(Box::new(LogicalPlan::ScalarExpr))))
            }
            LogicalPlan::Sum(inner) => {
                Ok((inner, LogicalPlan::Sum(Box::new(LogicalPlan::ScalarExpr))))
            }
            LogicalPlan::Avg(inner) => {
                Ok((inner, LogicalPlan::Avg(Box::new(LogicalPlan::ScalarExpr))))
            }
            LogicalPlan::Min(inner) => {
                Ok((inner, LogicalPlan::Min(Box::new(LogicalPlan::ScalarExpr))))
            }
            LogicalPlan::Max(inner) => {
                Ok((inner, LogicalPlan::Max(Box::new(LogicalPlan::ScalarExpr))))
            }
            LogicalPlan::Count(inner) => {
                Ok((inner, LogicalPlan::Count(Box::new(LogicalPlan::ScalarExpr))))
            }
            LogicalPlan::ApproxCountDistinct(inner) => Ok((
                inner,
                LogicalPlan::ApproxCountDistinct(Box::new(LogicalPlan::ScalarExpr)),
            )),
        }
    }

    /// Replaces the input sub-plan, returning a new node with the same shape.
    /// Panics for `Source` and `ScalarExpr`.
    pub(crate) fn with_input(self, new_input: LogicalPlan) -> LogicalPlan {
        let new_box = Box::new(new_input);
        match self {
            LogicalPlan::Source(_) | LogicalPlan::ScalarExpr => {
                panic!("with_input called on leaf node")
            }

            LogicalPlan::Filter { predicate, .. } => LogicalPlan::Filter {
                input: new_box,
                predicate,
            },
            LogicalPlan::Map { projection, .. } => LogicalPlan::Map {
                input: new_box,
                projection,
            },
            LogicalPlan::FlatMap { expansion, .. } => LogicalPlan::FlatMap {
                input: new_box,
                expansion,
            },
            LogicalPlan::TakeWhile { predicate, .. } => LogicalPlan::TakeWhile {
                input: new_box,
                predicate,
            },
            LogicalPlan::DropWhile { predicate, .. } => LogicalPlan::DropWhile {
                input: new_box,
                predicate,
            },
            LogicalPlan::Take { n, .. } => LogicalPlan::Take { input: new_box, n },
            LogicalPlan::Skip { n, .. } => LogicalPlan::Skip { input: new_box, n },
            LogicalPlan::Sort { spec, .. } => LogicalPlan::Sort {
                input: new_box,
                spec,
            },
            LogicalPlan::Unique { key, .. } => LogicalPlan::Unique {
                input: new_box,
                key,
            },
            LogicalPlan::Reverse { .. } => LogicalPlan::Reverse { input: new_box },
            LogicalPlan::GroupBy { key, .. } => LogicalPlan::GroupBy {
                input: new_box,
                key,
            },
            LogicalPlan::CountBy { key, .. } => LogicalPlan::CountBy {
                input: new_box,
                key,
            },
            LogicalPlan::IndexBy { key, .. } => LogicalPlan::IndexBy {
                input: new_box,
                key,
            },

            LogicalPlan::First(_) => LogicalPlan::First(new_box),
            LogicalPlan::Last(_) => LogicalPlan::Last(new_box),
            LogicalPlan::Sum(_) => LogicalPlan::Sum(new_box),
            LogicalPlan::Avg(_) => LogicalPlan::Avg(new_box),
            LogicalPlan::Min(_) => LogicalPlan::Min(new_box),
            LogicalPlan::Max(_) => LogicalPlan::Max(new_box),
            LogicalPlan::Count(_) => LogicalPlan::Count(new_box),
            LogicalPlan::ApproxCountDistinct(_) => LogicalPlan::ApproxCountDistinct(new_box),
        }
    }
}
