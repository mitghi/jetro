//! Logical plan IR — a tree-shaped, uncompiled representation of a pipeline query.
//!
//! `LogicalPlan` sits between the `Expr` AST (produced by the parser) and the
//! compiled `Pipeline` struct. All stage bodies are `Expr` nodes; compilation to
//! `Arc<Program>` happens only in the lowering pass after optimizer rules have
//! been applied.

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

    // ── Streaming transforms ──────────────────────────────────────────────
    Filter     { input: Box<Self>, predicate: Expr },
    Map        { input: Box<Self>, projection: Expr },
    FlatMap    { input: Box<Self>, expansion: Expr },
    TakeWhile  { input: Box<Self>, predicate: Expr },
    DropWhile  { input: Box<Self>, predicate: Expr },

    // ── Positional ────────────────────────────────────────────────────────
    Take { input: Box<Self>, n: usize },
    Skip { input: Box<Self>, n: usize },

    // ── Ordering / dedup ──────────────────────────────────────────────────
    Sort   { input: Box<Self>, spec: SortSpec },
    Unique { input: Box<Self>, key: Option<Expr> },
    Reverse { input: Box<Self> },

    // ── Keyed reducers ────────────────────────────────────────────────────
    GroupBy  { input: Box<Self>, key: Expr },
    CountBy  { input: Box<Self>, key: Expr },
    IndexBy  { input: Box<Self>, key: Expr },

    // ── Terminal sinks ────────────────────────────────────────────────────
    First  (Box<Self>),
    Last   (Box<Self>),
    Sum    (Box<Self>),
    Avg    (Box<Self>),
    Min    (Box<Self>),
    Max    (Box<Self>),
    Count  (Box<Self>),
    ApproxCountDistinct(Box<Self>),

    // ── VM fallback ───────────────────────────────────────────────────────
    /// Any expression the logical planner could not classify; executed by VM.
    /// The inner `Expr` is intentionally unused after lowering — `collect()` returns
    /// `None` without reading it, making the variant a sentinel for the fallback path.
    #[allow(dead_code)]
    ScalarExpr(Expr),
}

impl LogicalPlan {
    /// Consumes `self` and returns `(input, node_without_input)` for use in rewrites.
    /// Returns `Err(self)` for `Source` and `ScalarExpr`.
    pub(crate) fn take_input(self) -> Result<(Box<LogicalPlan>, LogicalPlan), LogicalPlan> {
        match self {
            LogicalPlan::Source(_) | LogicalPlan::ScalarExpr(_) => Err(self),

            LogicalPlan::Filter { input, predicate } =>
                Ok((input, LogicalPlan::Filter { input: Box::new(LogicalPlan::ScalarExpr(Expr::Null)), predicate })),
            LogicalPlan::Map { input, projection } =>
                Ok((input, LogicalPlan::Map { input: Box::new(LogicalPlan::ScalarExpr(Expr::Null)), projection })),
            LogicalPlan::FlatMap { input, expansion } =>
                Ok((input, LogicalPlan::FlatMap { input: Box::new(LogicalPlan::ScalarExpr(Expr::Null)), expansion })),
            LogicalPlan::TakeWhile { input, predicate } =>
                Ok((input, LogicalPlan::TakeWhile { input: Box::new(LogicalPlan::ScalarExpr(Expr::Null)), predicate })),
            LogicalPlan::DropWhile { input, predicate } =>
                Ok((input, LogicalPlan::DropWhile { input: Box::new(LogicalPlan::ScalarExpr(Expr::Null)), predicate })),
            LogicalPlan::Take { input, n } =>
                Ok((input, LogicalPlan::Take { input: Box::new(LogicalPlan::ScalarExpr(Expr::Null)), n })),
            LogicalPlan::Skip { input, n } =>
                Ok((input, LogicalPlan::Skip { input: Box::new(LogicalPlan::ScalarExpr(Expr::Null)), n })),
            LogicalPlan::Sort { input, spec } =>
                Ok((input, LogicalPlan::Sort { input: Box::new(LogicalPlan::ScalarExpr(Expr::Null)), spec })),
            LogicalPlan::Unique { input, key } =>
                Ok((input, LogicalPlan::Unique { input: Box::new(LogicalPlan::ScalarExpr(Expr::Null)), key })),
            LogicalPlan::Reverse { input } =>
                Ok((input, LogicalPlan::Reverse { input: Box::new(LogicalPlan::ScalarExpr(Expr::Null)) })),
            LogicalPlan::GroupBy { input, key } =>
                Ok((input, LogicalPlan::GroupBy { input: Box::new(LogicalPlan::ScalarExpr(Expr::Null)), key })),
            LogicalPlan::CountBy { input, key } =>
                Ok((input, LogicalPlan::CountBy { input: Box::new(LogicalPlan::ScalarExpr(Expr::Null)), key })),
            LogicalPlan::IndexBy { input, key } =>
                Ok((input, LogicalPlan::IndexBy { input: Box::new(LogicalPlan::ScalarExpr(Expr::Null)), key })),

            LogicalPlan::First(inner) =>
                Ok((inner, LogicalPlan::First(Box::new(LogicalPlan::ScalarExpr(Expr::Null))))),
            LogicalPlan::Last(inner) =>
                Ok((inner, LogicalPlan::Last(Box::new(LogicalPlan::ScalarExpr(Expr::Null))))),
            LogicalPlan::Sum(inner) =>
                Ok((inner, LogicalPlan::Sum(Box::new(LogicalPlan::ScalarExpr(Expr::Null))))),
            LogicalPlan::Avg(inner) =>
                Ok((inner, LogicalPlan::Avg(Box::new(LogicalPlan::ScalarExpr(Expr::Null))))),
            LogicalPlan::Min(inner) =>
                Ok((inner, LogicalPlan::Min(Box::new(LogicalPlan::ScalarExpr(Expr::Null))))),
            LogicalPlan::Max(inner) =>
                Ok((inner, LogicalPlan::Max(Box::new(LogicalPlan::ScalarExpr(Expr::Null))))),
            LogicalPlan::Count(inner) =>
                Ok((inner, LogicalPlan::Count(Box::new(LogicalPlan::ScalarExpr(Expr::Null))))),
            LogicalPlan::ApproxCountDistinct(inner) =>
                Ok((inner, LogicalPlan::ApproxCountDistinct(Box::new(LogicalPlan::ScalarExpr(Expr::Null))))),
        }
    }

    /// Replaces the input sub-plan, returning a new node with the same shape.
    /// Panics for `Source` and `ScalarExpr`.
    pub(crate) fn with_input(self, new_input: LogicalPlan) -> LogicalPlan {
        let new_box = Box::new(new_input);
        match self {
            LogicalPlan::Source(_) | LogicalPlan::ScalarExpr(_) =>
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
