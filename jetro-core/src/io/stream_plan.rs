//! Source-level row stream plan IR.
//!
//! This IR models expressions rooted at `$.rows()` before they are bound to a
//! concrete source implementation such as NDJSON rows or document-array rows.
//! It deliberately contains stream semantics only; byte/tape and materialized
//! execution details live behind source/projector implementations.

use crate::builtins::BuiltinMethod;
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
    Sum,
    Avg,
    Min,
    Max,
    Any(Expr),
    All(Expr),
}

impl RowStreamStage {
    fn scalar_sink(&self) -> bool {
        matches!(
            self,
            RowStreamStage::Last
                | RowStreamStage::Count
                | RowStreamStage::Sum
                | RowStreamStage::Avg
                | RowStreamStage::Min
                | RowStreamStage::Max
                | RowStreamStage::Any(_)
                | RowStreamStage::All(_)
        )
    }

    fn retained_limit(&self) -> Option<usize> {
        match self {
            RowStreamStage::Take(n) => Some(*n),
            RowStreamStage::Last => Some(1),
            _ => None,
        }
    }

    fn blocks_parallel_partitioning(&self) -> bool {
        matches!(self, RowStreamStage::DistinctBy(_) | RowStreamStage::Last)
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
        let method = BuiltinMethod::from_name(name);
        match method {
            BuiltinMethod::Reverse => {
                require_arity(name, args, 0)?;
                plan.direction = match plan.direction {
                    RowStreamDirection::Forward => RowStreamDirection::Reverse,
                    RowStreamDirection::Reverse => RowStreamDirection::Forward,
                };
            }
            BuiltinMethod::Filter | BuiltinMethod::FindAll => {
                let expr = single_expr_arg(name, args)?.clone();
                plan.stages.push(RowStreamStage::Filter(expr));
            }
            BuiltinMethod::Find | BuiltinMethod::FindFirst | BuiltinMethod::FindOne => {
                let expr = single_expr_arg(name, args)?.clone();
                plan.stages.push(RowStreamStage::Filter(expr));
                plan.stages.push(RowStreamStage::Take(1));
            }
            BuiltinMethod::UniqueBy => {
                let expr = single_expr_arg(name, args)?.clone();
                plan.stages.push(RowStreamStage::DistinctBy(expr));
            }
            BuiltinMethod::Take => {
                let n = single_usize_arg(name, args)?;
                plan.stages.push(RowStreamStage::Take(n));
            }
            BuiltinMethod::First => {
                require_arity(name, args, 0)?;
                plan.stages.push(RowStreamStage::Take(1));
            }
            BuiltinMethod::Last => {
                require_arity(name, args, 0)?;
                plan.stages.push(RowStreamStage::Last);
                terminal = Some(name.as_str());
            }
            BuiltinMethod::Count | BuiltinMethod::Len => {
                require_arity(name, args, 0)?;
                plan.stages.push(RowStreamStage::Count);
                terminal = Some(name.as_str());
            }
            BuiltinMethod::Sum => {
                require_arity(name, args, 0)?;
                plan.stages.push(RowStreamStage::Sum);
                terminal = Some(name.as_str());
            }
            BuiltinMethod::Avg => {
                require_arity(name, args, 0)?;
                plan.stages.push(RowStreamStage::Avg);
                terminal = Some(name.as_str());
            }
            BuiltinMethod::Min => {
                require_arity(name, args, 0)?;
                plan.stages.push(RowStreamStage::Min);
                terminal = Some(name.as_str());
            }
            BuiltinMethod::Max => {
                require_arity(name, args, 0)?;
                plan.stages.push(RowStreamStage::Max);
                terminal = Some(name.as_str());
            }
            BuiltinMethod::Any => {
                let expr = single_expr_arg(name, args)?.clone();
                plan.stages.push(RowStreamStage::Any(expr));
                terminal = Some(name.as_str());
            }
            BuiltinMethod::All => {
                let expr = single_expr_arg(name, args)?.clone();
                plan.stages.push(RowStreamStage::All(expr));
                terminal = Some(name.as_str());
            }
            BuiltinMethod::Map => {
                let expr = single_expr_arg(name, args)?.clone();
                plan.stages.push(RowStreamStage::Map(expr));
            }
            _ => {
                return Err(RowStreamPlanError::new(format!(
                    "unsupported rows() stream method {name}()"
                )));
            }
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
            match stage {
                RowStreamStage::Filter(_) => demand.predicate_count += 1,
                RowStreamStage::DistinctBy(_) => demand.key_count += 1,
                stage if stage.retained_limit().is_some() => {
                    seen_take.get_or_insert(stage.retained_limit().expect("retained limit"));
                }
                RowStreamStage::Map(_) => demand.projector_count += 1,
                stage if stage.scalar_sink() => demand.scalar_output = true,
                _ => {}
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
        match stage {
            RowStreamStage::Filter(_) => saw_filter = true,
            RowStreamStage::Map(_) => {}
            RowStreamStage::Take(_) => {}
            stage if stage.blocks_parallel_partitioning() => {
                return RowStreamParallelism::Sequential;
            }
            stage if stage.scalar_sink() => {}
            _ => {}
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
        .position(|stage| matches!(stage, RowStreamStage::Map(_)))
    else {
        return false;
    };
    stages[..map_idx].iter().any(|stage| {
        matches!(
            stage,
            RowStreamStage::Filter(_) | RowStreamStage::DistinctBy(_)
        ) || stage.retained_limit().is_some()
            || stage.scalar_sink()
    })
}

fn preserves_source_order_until_limit(stages: &[RowStreamStage]) -> bool {
    for stage in stages {
        if stage.retained_limit().is_some() {
            return true;
        }
        match stage {
            RowStreamStage::Filter(_) | RowStreamStage::Map(_) => {}
            _ => return false,
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
    if BuiltinMethod::from_name(name) != BuiltinMethod::Rows || !args.is_empty() {
        return None;
    }
    Some(rest)
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
        assert!(matches!(plan.stages.last(), Some(RowStreamStage::Sum)));
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
    fn rejects_non_method_rows_stream_step() {
        let expr = parse("$.rows().name").unwrap();
        let err = lower_root_rows_expr(&expr, RowStreamSourceKind::NdjsonRows)
            .unwrap_err()
            .to_string();

        assert!(err.contains("unsupported rows() stream step"));
    }

    #[test]
    fn root_rows_query_guard_is_specific() {
        assert!(looks_like_root_rows_query("$.rows().take(1)"));
        assert!(looks_like_root_rows_query("  $.rows().reverse()"));
        assert!(!looks_like_root_rows_query("$.name"));
        assert!(!looks_like_root_rows_query("$.items.rows().take(1)"));
    }
}
