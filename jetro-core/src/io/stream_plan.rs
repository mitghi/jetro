//! Source-level row stream plan IR.
//!
//! This IR models expressions rooted at `$.rows()` before they are bound to a
//! concrete source implementation such as NDJSON rows or document-array rows.
//! It deliberately contains stream semantics only; byte/tape and materialized
//! execution details live behind source/projector implementations.

use crate::builtins::registry::{
    by_name as builtin_by_name, numeric_reducer, predicate_sink, row_stream_op, row_stream_op_arg,
    row_stream_op_blocks_parallel_partitioning, row_stream_op_is_filter_like,
    row_stream_op_is_projector, row_stream_op_is_row_selection, row_stream_op_is_terminal,
    row_stream_op_preserves_order_before_limit, BuiltinId,
};
use crate::builtins::{
    BuiltinMethod, BuiltinNumericReducer, BuiltinPredicateSink, BuiltinRowStreamArg,
    BuiltinRowStreamOp,
};
use crate::parse::ast::{Arg, Expr, Step};
use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RowStreamSourceKind {
    DocumentRows,
    NdjsonRows,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RowStreamDirection {
    Forward,
    Reverse,
}

impl Default for RowStreamDirection {
    fn default() -> Self {
        Self::Forward
    }
}

#[derive(Clone, Debug)]
pub(crate) struct RowStreamPlan {
    pub source: RowStreamSourceKind,
    pub direction: RowStreamDirection,
    pub stages: Vec<RowStreamStage>,
    pub demand: RowStreamDemand,
}

impl RowStreamPlan {
    pub fn new(source: RowStreamSourceKind) -> Self {
        Self {
            source,
            direction: RowStreamDirection::Forward,
            stages: Vec::new(),
            demand: RowStreamDemand::default(),
        }
    }

    pub(super) fn refresh_demand(&mut self) {
        self.demand = RowStreamDemand::from_plan(self);
    }

    pub(super) fn returns_scalar_value(&self) -> bool {
        self.demand.retained_limit == Some(1) || self.demand.scalar_output
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct RowStreamDemand {
    pub retained_limit: Option<usize>,
    pub scalar_output: bool,
    pub predicate_count: usize,
    pub key_count: usize,
    pub projector_count: usize,
    pub late_projection: bool,
    pub ordered_early_stop: bool,
    pub parallel: RowStreamParallelism,
}

#[derive(Clone, Debug)]
pub(crate) enum RowStreamStage {
    Filter(Expr),
    DistinctBy(Expr),
    Take(usize),
    Map(Expr),
    Last,
    Count,
    Numeric(BuiltinNumericReducer),
    Any(Expr),
    All(Expr),
    FindOne(Expr),
}

impl RowStreamStage {
    fn builtin_method(&self) -> BuiltinMethod {
        match self {
            RowStreamStage::Filter(_) => BuiltinMethod::Filter,
            RowStreamStage::DistinctBy(_) => BuiltinMethod::UniqueBy,
            RowStreamStage::Take(_) => BuiltinMethod::Take,
            RowStreamStage::Map(_) => BuiltinMethod::Map,
            RowStreamStage::Last => BuiltinMethod::Last,
            RowStreamStage::Count => BuiltinMethod::Count,
            RowStreamStage::Numeric(reducer) => reducer.method(),
            RowStreamStage::Any(_) => BuiltinMethod::Any,
            RowStreamStage::All(_) => BuiltinMethod::All,
            RowStreamStage::FindOne(_) => BuiltinMethod::FindOne,
        }
    }

    fn row_stream_op(&self) -> Option<BuiltinRowStreamOp> {
        row_stream_op(BuiltinId::from_method(self.builtin_method()))
    }

    fn scalar_sink(&self) -> bool {
        self.row_stream_op().is_some_and(row_stream_op_is_terminal)
    }

    fn retained_limit(&self) -> Option<usize> {
        match self {
            RowStreamStage::Take(n) => Some(*n),
            RowStreamStage::Last => Some(1),
            _ => None,
        }
    }

    pub(super) fn numeric_reducer(&self) -> Option<BuiltinNumericReducer> {
        match self {
            RowStreamStage::Numeric(reducer) => Some(*reducer),
            _ => None,
        }
    }

    pub(super) fn predicate_sink(&self) -> Option<(BuiltinPredicateSink, &Expr)> {
        let sink = predicate_sink(BuiltinId::from_method(self.builtin_method()))?;
        match self {
            RowStreamStage::Any(expr)
            | RowStreamStage::All(expr)
            | RowStreamStage::FindOne(expr) => Some((sink, expr)),
            _ => None,
        }
    }

    fn blocks_parallel_partitioning(&self) -> bool {
        self.row_stream_op()
            .is_some_and(row_stream_op_blocks_parallel_partitioning)
    }

    fn is_filter_like(&self) -> bool {
        self.row_stream_op()
            .is_some_and(row_stream_op_is_filter_like)
    }

    fn is_projector(&self) -> bool {
        self.row_stream_op().is_some_and(row_stream_op_is_projector)
    }

    fn is_row_selection(&self) -> bool {
        self.row_stream_op()
            .is_some_and(row_stream_op_is_row_selection)
    }

    fn preserves_order_before_limit(&self) -> bool {
        self.row_stream_op()
            .is_some_and(row_stream_op_preserves_order_before_limit)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum RowStreamParallelism {
    #[default]
    Sequential,
    PartitionFilter {
        retained_limit: Option<usize>,
        direction: RowStreamDirection,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RowStreamFileStrategy {
    Sequential,
    Partitioned {
        retained_limit: usize,
    },
    OrderedPartitionSearch {
        direction: RowStreamDirection,
        retained_limit: usize,
    },
}

impl RowStreamPlan {
    pub(crate) fn file_strategy(&self, partition_available: bool) -> RowStreamFileStrategy {
        if partition_available {
            if let Some(retained_limit) = self.ordered_partition_retained_limit() {
                return RowStreamFileStrategy::OrderedPartitionSearch {
                    direction: self.direction,
                    retained_limit,
                };
            }
            if let Some(retained_limit) = self.partition_retained_limit() {
                return RowStreamFileStrategy::Partitioned { retained_limit };
            }
        }
        RowStreamFileStrategy::Sequential
    }

    fn ordered_partition_retained_limit(&self) -> Option<usize> {
        (self.direction == RowStreamDirection::Reverse && self.demand.ordered_early_stop)
            .then_some(self.demand.retained_limit?)
    }

    fn partition_retained_limit(&self) -> Option<usize> {
        if self.direction == RowStreamDirection::Reverse && self.demand.ordered_early_stop {
            return None;
        }
        match self.demand.parallel {
            RowStreamParallelism::PartitionFilter {
                retained_limit: Some(limit),
                ..
            } if limit > 0 => Some(limit),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct RowStreamPlanError {
    message: String,
}

impl RowStreamPlanError {
    pub(super) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for RowStreamPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

#[cfg(test)]
/// Returns true when the expression is rooted at `$.rows()`.
pub(super) fn is_root_rows_expr(expr: &Expr) -> bool {
    root_rows_steps(expr).is_some()
}

/// Lowers a root `$.rows()` expression into the source-level row stream IR.
pub(super) fn lower_root_rows_expr(
    expr: &Expr,
    source: RowStreamSourceKind,
) -> Result<Option<RowStreamPlan>, RowStreamPlanError> {
    let Some(steps) = root_rows_steps(expr) else {
        return Ok(None);
    };

    let mut plan = RowStreamPlan::new(source);
    let mut terminal = None;
    for step in steps {
        let Step::Method(name, args) = step else {
            return Err(RowStreamPlanError::new(format!(
                "unsupported rows() stream step {step:?}"
            )));
        };
        if let Some(terminal) = terminal {
            return Err(RowStreamPlanError::new(format!(
                "rows() stream method {name}() cannot follow terminal method {terminal}()"
            )));
        }
        let Some(id) = builtin_by_name(name) else {
            return Err(RowStreamPlanError::new(format!(
                "unsupported rows() stream method {name}()"
            )));
        };
        let Some(op) = row_stream_op(id) else {
            return Err(RowStreamPlanError::new(format!(
                "unsupported rows() stream method {name}()"
            )));
        };
        let arg = row_stream_op_arg(op);
        let expr_arg = match arg {
            BuiltinRowStreamArg::Expr => Some(single_expr_arg(name, args)?.clone()),
            BuiltinRowStreamArg::Usize | BuiltinRowStreamArg::None => None,
        };
        let usize_arg = match arg {
            BuiltinRowStreamArg::Usize => Some(single_usize_arg(name, args)?),
            BuiltinRowStreamArg::Expr => None,
            BuiltinRowStreamArg::None => {
                require_arity(name, args, 0)?;
                None
            }
        };

        match op {
            BuiltinRowStreamOp::Reverse => {
                plan.direction = match plan.direction {
                    RowStreamDirection::Forward => RowStreamDirection::Reverse,
                    RowStreamDirection::Reverse => RowStreamDirection::Forward,
                };
            }
            BuiltinRowStreamOp::Filter => {
                plan.stages.push(RowStreamStage::Filter(expr_arg.unwrap()));
            }
            BuiltinRowStreamOp::FindFirst => {
                plan.stages.push(RowStreamStage::Filter(expr_arg.unwrap()));
                plan.stages.push(RowStreamStage::Take(1));
            }
            BuiltinRowStreamOp::FindOne => {
                plan.stages.push(RowStreamStage::FindOne(expr_arg.unwrap()));
            }
            BuiltinRowStreamOp::DistinctBy => {
                plan.stages.push(RowStreamStage::DistinctBy(expr_arg.unwrap()));
            }
            BuiltinRowStreamOp::Take => {
                plan.stages.push(RowStreamStage::Take(usize_arg.unwrap()));
            }
            BuiltinRowStreamOp::First => {
                plan.stages.push(RowStreamStage::Take(1));
            }
            BuiltinRowStreamOp::Last => {
                plan.stages.push(RowStreamStage::Last);
            }
            BuiltinRowStreamOp::Count => {
                plan.stages.push(RowStreamStage::Count);
            }
            BuiltinRowStreamOp::Sum
            | BuiltinRowStreamOp::Avg
            | BuiltinRowStreamOp::Min
            | BuiltinRowStreamOp::Max => {
                let reducer = numeric_reducer(id).ok_or_else(|| {
                    RowStreamPlanError::new(format!(
                        "rows() stream method {name}() is missing numeric reducer metadata"
                    ))
                })?;
                plan.stages.push(RowStreamStage::Numeric(reducer));
            }
            BuiltinRowStreamOp::Any => {
                plan.stages.push(RowStreamStage::Any(expr_arg.unwrap()));
            }
            BuiltinRowStreamOp::All => {
                plan.stages.push(RowStreamStage::All(expr_arg.unwrap()));
            }
            BuiltinRowStreamOp::Map => {
                plan.stages.push(RowStreamStage::Map(expr_arg.unwrap()));
            }
        }
        if row_stream_op_is_terminal(op) {
            terminal = Some(name.as_str());
        }
    }

    plan.demand = RowStreamDemand::from_plan(&plan);
    Ok(Some(plan))
}

impl RowStreamDemand {
    fn from_plan(plan: &RowStreamPlan) -> Self {
        let mut demand = RowStreamDemand::default();
        let mut seen_take = None;
        for stage in &plan.stages {
            if stage.is_filter_like() {
                demand.predicate_count += 1;
            }
            if matches!(stage, RowStreamStage::DistinctBy(_)) {
                demand.key_count += 1;
            }
            if let Some(limit) = stage.retained_limit() {
                seen_take.get_or_insert(limit);
            }
            if stage.is_projector() {
                demand.projector_count += 1;
            }
            if stage.scalar_sink() {
                demand.scalar_output = true;
            }
        }
        demand.retained_limit = seen_take;
        demand.late_projection = first_projector_is_after_row_selection(&plan.stages);
        demand.ordered_early_stop =
            demand.retained_limit.is_some() && preserves_source_order_until_limit(&plan.stages);
        demand.parallel = classify_parallelism(plan, demand.retained_limit);
        demand
    }
}

fn classify_parallelism(
    plan: &RowStreamPlan,
    retained_limit: Option<usize>,
) -> RowStreamParallelism {
    let mut saw_filter = false;
    for stage in &plan.stages {
        if stage.blocks_parallel_partitioning() {
            return RowStreamParallelism::Sequential;
        } else if stage.is_filter_like() {
            saw_filter = true;
        } else if stage.is_projector() || stage.retained_limit().is_some() || stage.scalar_sink() {
            // These stages do not prevent partition-filter execution.
        } else {
            return RowStreamParallelism::Sequential;
        }
    }

    if saw_filter {
        RowStreamParallelism::PartitionFilter {
            retained_limit,
            direction: plan.direction,
        }
    } else {
        RowStreamParallelism::Sequential
    }
}

fn first_projector_is_after_row_selection(stages: &[RowStreamStage]) -> bool {
    let Some(map_idx) = stages
        .iter()
        .position(RowStreamStage::is_projector)
    else {
        return false;
    };
    stages[..map_idx].iter().any(|stage| {
        stage.is_row_selection()
            || stage.retained_limit().is_some()
            || stage.scalar_sink()
    })
}

fn preserves_source_order_until_limit(stages: &[RowStreamStage]) -> bool {
    for stage in stages {
        if stage.retained_limit().is_some() {
            return true;
        }
        if !stage.preserves_order_before_limit() {
            return false;
        }
    }
    false
}

pub(super) fn lower_root_rows_query(
    query: &str,
    source: RowStreamSourceKind,
) -> Result<Option<RowStreamPlan>, RowStreamPlanError> {
    if !looks_like_root_rows_query(query) {
        return Ok(None);
    }
    let Ok(expr) = crate::parse::parser::parse(query) else {
        return Ok(None);
    };
    lower_root_rows_expr(&expr, source)
}

#[cfg(test)]
pub(super) fn query_may_contain_rows_stream(query: &str) -> bool {
    parse_rows_stream_candidate_query(query)
        .ok()
        .flatten()
        .is_some()
}

pub(super) fn parse_rows_stream_candidate_query(
    query: &str,
) -> Result<Option<Expr>, RowStreamPlanError> {
    if !query.contains("rows") {
        return Ok(None);
    }
    let expr = crate::parse::parser::parse(query)
        .map_err(|err| RowStreamPlanError::new(err.to_string()))?;
    if expr_contains_root_rows_stream(&expr) {
        Ok(Some(expr))
    } else {
        Ok(None)
    }
}

pub(super) fn looks_like_root_rows_query(query: &str) -> bool {
    let query = query.trim_start();
    query.starts_with("$.rows(") || query.starts_with("$.rows.")
}

fn root_rows_steps(expr: &Expr) -> Option<&[Step]> {
    let Expr::Chain(base, steps) = expr else {
        return None;
    };
    if !matches!(base.as_ref(), Expr::Root) {
        return None;
    }
    let Some((Step::Method(name, args), rest)) = steps.split_first() else {
        return None;
    };
    if !is_root_rows_method(name, args) {
        return None;
    }
    Some(rest)
}

fn expr_contains_root_rows_stream(expr: &Expr) -> bool {
    if root_rows_steps(expr).is_some() {
        return true;
    }

    match expr {
        Expr::FString(parts) => parts.iter().any(|part| match part {
            crate::parse::ast::FStringPart::Lit(_) => false,
            crate::parse::ast::FStringPart::Interp { expr, .. } => {
                expr_contains_root_rows_stream(expr)
            }
        }),
        Expr::Chain(base, steps) => {
            expr_contains_root_rows_stream(base) || steps.iter().any(step_contains_root_rows_stream)
        }
        Expr::BinOp(lhs, _, rhs) | Expr::Coalesce(lhs, rhs) => {
            expr_contains_root_rows_stream(lhs) || expr_contains_root_rows_stream(rhs)
        }
        Expr::UnaryNeg(inner) | Expr::Not(inner) | Expr::Cast { expr: inner, .. } => {
            expr_contains_root_rows_stream(inner)
        }
        Expr::Kind { expr: inner, .. } => expr_contains_root_rows_stream(inner),
        Expr::Object(fields) => fields.iter().any(obj_field_contains_root_rows_stream),
        Expr::Array(elems) => elems.iter().any(array_elem_contains_root_rows_stream),
        Expr::Pipeline { base, steps } => {
            expr_contains_root_rows_stream(base)
                || steps.iter().any(|step| match step {
                    crate::parse::ast::PipeStep::Forward(expr) => {
                        expr_contains_root_rows_stream(expr)
                    }
                    crate::parse::ast::PipeStep::Bind(_) => false,
                })
        }
        Expr::ListComp {
            expr, iter, cond, ..
        }
        | Expr::SetComp {
            expr, iter, cond, ..
        }
        | Expr::GenComp {
            expr, iter, cond, ..
        } => {
            expr_contains_root_rows_stream(expr)
                || expr_contains_root_rows_stream(iter)
                || cond
                    .as_deref()
                    .is_some_and(expr_contains_root_rows_stream)
        }
        Expr::DictComp {
            key,
            val,
            iter,
            cond,
            ..
        } => {
            expr_contains_root_rows_stream(key)
                || expr_contains_root_rows_stream(val)
                || expr_contains_root_rows_stream(iter)
                || cond
                    .as_deref()
                    .is_some_and(expr_contains_root_rows_stream)
        }
        Expr::Lambda { body, .. } => expr_contains_root_rows_stream(body),
        Expr::Let { init, body, .. } => {
            expr_contains_root_rows_stream(init) || expr_contains_root_rows_stream(body)
        }
        Expr::IfElse { cond, then_, else_ } => {
            expr_contains_root_rows_stream(cond)
                || expr_contains_root_rows_stream(then_)
                || expr_contains_root_rows_stream(else_)
        }
        Expr::Try { body, default } => {
            expr_contains_root_rows_stream(body) || expr_contains_root_rows_stream(default)
        }
        Expr::GlobalCall { args, .. } => args.iter().any(arg_contains_root_rows_stream),
        Expr::Patch { root, ops } | Expr::UpdateBatch { root, ops, .. } => {
            expr_contains_root_rows_stream(root)
                || ops.iter().any(|op| {
                    expr_contains_root_rows_stream(&op.val)
                        || op.cond.as_ref().is_some_and(expr_contains_root_rows_stream)
                        || op.path.iter().any(path_step_contains_root_rows_stream)
                })
        }
        Expr::Match { scrutinee, arms } => {
            expr_contains_root_rows_stream(scrutinee)
                || arms.iter().any(|arm| {
                    arm.guard
                        .as_ref()
                        .is_some_and(expr_contains_root_rows_stream)
                        || expr_contains_root_rows_stream(&arm.body)
                })
        }
        Expr::Null
        | Expr::Bool(_)
        | Expr::Int(_)
        | Expr::Float(_)
        | Expr::Str(_)
        | Expr::Root
        | Expr::Current
        | Expr::Ident(_)
        | Expr::DeleteMark => false,
    }
}

fn step_contains_root_rows_stream(step: &Step) -> bool {
    match step {
        Step::DynIndex(expr) | Step::InlineFilter(expr) => expr_contains_root_rows_stream(expr),
        Step::Method(_, args) | Step::OptMethod(_, args) => {
            args.iter().any(arg_contains_root_rows_stream)
        }
        Step::DeepMatch { arms, .. } => arms.iter().any(|arm| {
            arm.guard
                .as_ref()
                .is_some_and(expr_contains_root_rows_stream)
                || expr_contains_root_rows_stream(&arm.body)
        }),
        _ => false,
    }
}

fn arg_contains_root_rows_stream(arg: &Arg) -> bool {
    match arg {
        Arg::Pos(expr) | Arg::Named(_, expr) => expr_contains_root_rows_stream(expr),
    }
}

fn array_elem_contains_root_rows_stream(elem: &crate::parse::ast::ArrayElem) -> bool {
    match elem {
        crate::parse::ast::ArrayElem::Expr(expr)
        | crate::parse::ast::ArrayElem::Spread(expr) => expr_contains_root_rows_stream(expr),
    }
}

fn obj_field_contains_root_rows_stream(field: &crate::parse::ast::ObjField) -> bool {
    match field {
        crate::parse::ast::ObjField::Kv { val, cond, .. } => {
            expr_contains_root_rows_stream(val)
                || cond
                    .as_ref()
                    .is_some_and(expr_contains_root_rows_stream)
        }
        crate::parse::ast::ObjField::Dynamic { key, val } => {
            expr_contains_root_rows_stream(key) || expr_contains_root_rows_stream(val)
        }
        crate::parse::ast::ObjField::Spread(expr)
        | crate::parse::ast::ObjField::SpreadDeep(expr) => expr_contains_root_rows_stream(expr),
        crate::parse::ast::ObjField::Short(_) => false,
    }
}

fn path_step_contains_root_rows_stream(step: &crate::parse::ast::PathStep) -> bool {
    match step {
        crate::parse::ast::PathStep::DynIndex(expr) => expr_contains_root_rows_stream(expr),
        crate::parse::ast::PathStep::WildcardFilter(expr) => {
            expr_contains_root_rows_stream(expr)
        }
        _ => false,
    }
}

/// Returns the number of leading chain steps that make up a root `$.rows()`
/// stream prefix, including the `rows()` method itself.
pub(super) fn root_rows_stream_prefix_len(expr: &Expr) -> Option<usize> {
    let Expr::Chain(base, steps) = expr else {
        return None;
    };
    if !matches!(base.as_ref(), Expr::Root) {
        return None;
    }
    let Some((Step::Method(name, args), _)) = steps.split_first() else {
        return None;
    };
    if !is_root_rows_method(name, args) {
        return None;
    }

    let mut split = 1usize;
    while let Some(Step::Method(name, _)) = steps.get(split) {
        if !is_rows_stream_method(name) {
            break;
        }
        split += 1;
    }
    Some(split)
}

fn is_root_rows_method(name: &str, args: &[Arg]) -> bool {
    builtin_by_name(name) == Some(BuiltinId::from_method(BuiltinMethod::Rows)) && args.is_empty()
}

pub(super) fn is_rows_stream_method(name: &str) -> bool {
    builtin_by_name(name).is_some_and(|id| row_stream_op(id).is_some())
}

fn require_arity(name: &str, args: &[Arg], arity: usize) -> Result<(), RowStreamPlanError> {
    if args.len() == arity {
        Ok(())
    } else {
        Err(RowStreamPlanError::new(format!(
            "rows() stream method {name}() expects {arity} arguments, got {}",
            args.len()
        )))
    }
}

fn single_expr_arg<'a>(name: &str, args: &'a [Arg]) -> Result<&'a Expr, RowStreamPlanError> {
    require_arity(name, args, 1)?;
    match &args[0] {
        Arg::Pos(expr) => Ok(expr),
        Arg::Named(_, _) => Err(RowStreamPlanError::new(format!(
            "rows() stream method {name}() does not accept named arguments"
        ))),
    }
}

fn single_usize_arg(name: &str, args: &[Arg]) -> Result<usize, RowStreamPlanError> {
    let expr = single_expr_arg(name, args)?;
    let Expr::Int(n) = expr else {
        return Err(RowStreamPlanError::new(format!(
            "rows() stream method {name}() expects a literal non-negative integer"
        )));
    };
    usize::try_from(*n).map_err(|_| {
        RowStreamPlanError::new(format!(
            "rows() stream method {name}() expects a literal non-negative integer"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parser::parse;

    #[test]
    fn detects_root_rows_expression() {
        let expr = parse("$.rows().take(2)").unwrap();
        assert!(is_root_rows_expr(&expr));
        let expr = parse("$.items.rows().take(2)").unwrap();
        assert!(!is_root_rows_expr(&expr));
    }

    #[test]
    fn detects_root_rows_stream_prefix_for_wrapped_subqueries() {
        let expr = parse("$.rows().reverse().find($.active).first().id").unwrap();
        assert_eq!(root_rows_stream_prefix_len(&expr), Some(4));

        let expr = parse("$.items.rows().take(2)").unwrap();
        assert_eq!(root_rows_stream_prefix_len(&expr), None);
    }

    #[test]
    fn lowers_rows_stream_chain() {
        let expr = parse("$.rows().reverse().distinct_by($.id).take(10).map($.v)").unwrap();
        let plan = lower_root_rows_expr(&expr, RowStreamSourceKind::NdjsonRows)
            .unwrap()
            .unwrap();
        assert_eq!(plan.source, RowStreamSourceKind::NdjsonRows);
        assert_eq!(plan.direction, RowStreamDirection::Reverse);
        assert_eq!(plan.stages.len(), 3);
        assert_eq!(plan.demand.retained_limit, Some(10));
        assert_eq!(plan.demand.key_count, 1);
        assert_eq!(plan.demand.projector_count, 1);
        assert!(plan.demand.late_projection);
        assert!(matches!(
            plan.demand.parallel,
            RowStreamParallelism::Sequential
        ));
        assert!(matches!(plan.stages[0], RowStreamStage::DistinctBy(_)));
        assert!(matches!(plan.stages[1], RowStreamStage::Take(10)));
        assert!(matches!(plan.stages[2], RowStreamStage::Map(_)));
    }

    #[test]
    fn lowers_rows_find_to_filter_take_one() {
        let expr = parse("$.rows().reverse().find($.name == \"Ada\")").unwrap();
        let plan = lower_root_rows_expr(&expr, RowStreamSourceKind::NdjsonRows)
            .unwrap()
            .unwrap();

        assert_eq!(plan.direction, RowStreamDirection::Reverse);
        assert_eq!(plan.stages.len(), 2);
        assert_eq!(plan.demand.retained_limit, Some(1));
        assert_eq!(plan.demand.predicate_count, 1);
        assert!(matches!(
            plan.demand.parallel,
            RowStreamParallelism::PartitionFilter {
                retained_limit: Some(1),
                direction: RowStreamDirection::Reverse,
            }
        ));
        assert!(matches!(plan.stages[0], RowStreamStage::Filter(_)));
        assert!(matches!(plan.stages[1], RowStreamStage::Take(1)));
    }

    #[test]
    fn lowers_rows_find_all_as_filter_alias() {
        let expr = parse("$.rows().find_all($.active).take(2)").unwrap();
        let plan = lower_root_rows_expr(&expr, RowStreamSourceKind::NdjsonRows)
            .unwrap()
            .unwrap();

        assert_eq!(plan.demand.predicate_count, 1);
        assert_eq!(plan.demand.retained_limit, Some(2));
        assert!(matches!(plan.stages[0], RowStreamStage::Filter(_)));
        assert!(matches!(plan.stages[1], RowStreamStage::Take(2)));
    }

    #[test]
    fn lowers_rows_last_as_scalar_retention_sink() {
        let expr = parse("$.rows().filter($.active).last()").unwrap();
        let plan = lower_root_rows_expr(&expr, RowStreamSourceKind::NdjsonRows)
            .unwrap()
            .unwrap();

        assert_eq!(plan.demand.retained_limit, Some(1));
        assert_eq!(plan.demand.parallel, RowStreamParallelism::Sequential);
        assert!(matches!(plan.stages[0], RowStreamStage::Filter(_)));
        assert!(matches!(plan.stages[1], RowStreamStage::Last));
    }

    #[test]
    fn annotates_stream_demand_without_late_projection_before_selection() {
        let expr = parse("$.rows().map($.v).take(2)").unwrap();
        let plan = lower_root_rows_expr(&expr, RowStreamSourceKind::DocumentRows)
            .unwrap()
            .unwrap();

        assert_eq!(plan.demand.retained_limit, Some(2));
        assert_eq!(plan.demand.projector_count, 1);
        assert!(!plan.demand.late_projection);
        assert_eq!(plan.demand.parallel, RowStreamParallelism::Sequential);
    }

    #[test]
    fn rejects_unsupported_rows_stream_method() {
        let expr = parse("$.rows().sort($.score)").unwrap();
        let err = lower_root_rows_expr(&expr, RowStreamSourceKind::NdjsonRows)
            .unwrap_err()
            .to_string();

        assert_eq!(err, "unsupported rows() stream method sort()");
    }

    #[test]
    fn rejects_rows_stream_wrong_arity_before_execution() {
        let expr = parse("$.rows().reverse(1)").unwrap();
        let err = lower_root_rows_expr(&expr, RowStreamSourceKind::NdjsonRows)
            .unwrap_err()
            .to_string();

        assert_eq!(
            err,
            "rows() stream method reverse() expects 0 arguments, got 1"
        );
    }

    #[test]
    fn rejects_rows_stream_dynamic_take_before_execution() {
        let expr = parse("$.rows().take($.limit)").unwrap();
        let err = lower_root_rows_expr(&expr, RowStreamSourceKind::NdjsonRows)
            .unwrap_err()
            .to_string();

        assert_eq!(
            err,
            "rows() stream method take() expects a literal non-negative integer"
        );
    }

    #[test]
    fn rejects_rows_stream_stage_after_terminal_count() {
        let expr = parse("$.rows().count().map($.id)").unwrap();
        let err = lower_root_rows_expr(&expr, RowStreamSourceKind::NdjsonRows)
            .unwrap_err()
            .to_string();

        assert_eq!(
            err,
            "rows() stream method map() cannot follow terminal method count()"
        );
    }

    #[test]
    fn lowers_rows_stream_sum_sink() {
        let expr = parse("$.rows().filter($.active).map($.price).sum()").unwrap();
        let plan = lower_root_rows_expr(&expr, RowStreamSourceKind::NdjsonRows)
            .unwrap()
            .unwrap();

        assert!(plan.demand.scalar_output);
        assert!(matches!(
            plan.stages.last(),
            Some(RowStreamStage::Numeric(BuiltinNumericReducer::Sum))
        ));
    }

    #[test]
    fn lowers_rows_stream_len_as_count_sink() {
        let expr = parse("$.rows().filter($.active).len()").unwrap();
        let plan = lower_root_rows_expr(&expr, RowStreamSourceKind::NdjsonRows)
            .unwrap()
            .unwrap();

        assert!(plan.demand.scalar_output);
        assert!(matches!(plan.stages.last(), Some(RowStreamStage::Count)));
    }

    #[test]
    fn lowers_rows_stream_find_one_as_exact_one_sink() {
        let expr = parse("$.rows().find_one(active == true)").unwrap();
        let plan = lower_root_rows_expr(&expr, RowStreamSourceKind::NdjsonRows)
            .unwrap()
            .unwrap();

        assert!(plan.demand.scalar_output);
        assert!(matches!(
            plan.stages.last(),
            Some(RowStreamStage::FindOne(_))
        ));
    }

    #[test]
    fn lowers_rows_stream_predicate_sinks() {
        let any = parse("$.rows().any($.active)").unwrap();
        let any_plan = lower_root_rows_expr(&any, RowStreamSourceKind::NdjsonRows)
            .unwrap()
            .unwrap();
        assert!(any_plan.demand.scalar_output);
        assert_eq!(any_plan.demand.predicate_count, 0);
        assert!(matches!(
            any_plan.stages.last(),
            Some(RowStreamStage::Any(_))
        ));

        let all = parse("$.rows().all($.active)").unwrap();
        let all_plan = lower_root_rows_expr(&all, RowStreamSourceKind::NdjsonRows)
            .unwrap()
            .unwrap();
        assert!(all_plan.demand.scalar_output);
        assert!(matches!(
            all_plan.stages.last(),
            Some(RowStreamStage::All(_))
        ));
    }

    #[test]
    fn row_stream_stages_report_predicate_sink_metadata() {
        let expr = parse("$.rows().any($.active)").unwrap();
        let plan = lower_root_rows_expr(&expr, RowStreamSourceKind::NdjsonRows)
            .unwrap()
            .unwrap();
        let (sink, _) = plan
            .stages
            .last()
            .and_then(RowStreamStage::predicate_sink)
            .expect("any sink metadata");
        assert_eq!(sink, BuiltinPredicateSink::Any);

        let expr = parse("$.rows().all($.active)").unwrap();
        let plan = lower_root_rows_expr(&expr, RowStreamSourceKind::NdjsonRows)
            .unwrap()
            .unwrap();
        let (sink, _) = plan
            .stages
            .last()
            .and_then(RowStreamStage::predicate_sink)
            .expect("all sink metadata");
        assert_eq!(sink, BuiltinPredicateSink::All);

        let expr = parse("$.rows().find_one($.active)").unwrap();
        let plan = lower_root_rows_expr(&expr, RowStreamSourceKind::NdjsonRows)
            .unwrap()
            .unwrap();
        let (sink, _) = plan
            .stages
            .last()
            .and_then(RowStreamStage::predicate_sink)
            .expect("find_one sink metadata");
        assert_eq!(sink, BuiltinPredicateSink::FindOne);
    }

    #[test]
    fn scalar_result_classification_comes_from_stream_demand() {
        let take_one = lower_root_rows_query("$.rows().take(1)", RowStreamSourceKind::NdjsonRows)
            .unwrap()
            .unwrap();
        assert!(take_one.returns_scalar_value());

        let count = lower_root_rows_query("$.rows().count()", RowStreamSourceKind::NdjsonRows)
            .unwrap()
            .unwrap();
        assert!(count.returns_scalar_value());

        let take_many = lower_root_rows_query("$.rows().take(2)", RowStreamSourceKind::NdjsonRows)
            .unwrap()
            .unwrap();
        assert!(!take_many.returns_scalar_value());
    }

    #[test]
    fn rejects_non_method_rows_stream_step() {
        let expr = parse("$.rows().name").unwrap();
        let err = lower_root_rows_expr(&expr, RowStreamSourceKind::NdjsonRows)
            .unwrap_err()
            .to_string();

        assert!(err.contains("unsupported rows() stream step"));
    }

    #[test]
    fn root_rows_query_guard_is_specific() {
        assert!(query_may_contain_rows_stream(
            r#"let stream = $.rows() in stream.count()"#
        ));
        assert!(query_may_contain_rows_stream(
            r#"{hit: match $.meta with { _ -> $.rows().find(active) }}"#
        ));
        assert!(!query_may_contain_rows_stream("$.items"));
        assert!(!query_may_contain_rows_stream(r#""$.rows().take(1)""#));
        assert!(looks_like_root_rows_query("$.rows().take(1)"));
        assert!(looks_like_root_rows_query("  $.rows().reverse()"));
        assert!(!looks_like_root_rows_query("$.name"));
        assert!(!looks_like_root_rows_query("$.items.rows().take(1)"));
    }
}
