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
pub(super) enum RowStreamSourceKind {
    DocumentRows,
    NdjsonRows,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum RowStreamDirection {
    Forward,
    Reverse,
}

impl Default for RowStreamDirection {
    fn default() -> Self {
        Self::Forward
    }
}

#[derive(Clone, Debug)]
pub(super) struct RowStreamPlan {
    pub source: RowStreamSourceKind,
    pub direction: RowStreamDirection,
    pub stages: Vec<RowStreamStage>,
}

impl RowStreamPlan {
    pub fn new(source: RowStreamSourceKind) -> Self {
        Self {
            source,
            direction: RowStreamDirection::Forward,
            stages: Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub(super) enum RowStreamStage {
    Filter(Expr),
    DistinctBy(Expr),
    Take(usize),
    Map(Expr),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct RowStreamPlanError {
    message: String,
}

impl RowStreamPlanError {
    fn new(message: impl Into<String>) -> Self {
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
    for step in steps {
        let Step::Method(name, args) = step else {
            return Err(RowStreamPlanError::new(format!(
                "unsupported rows() stream step {step:?}"
            )));
        };
        let method = BuiltinMethod::from_name(name);
        match method {
            BuiltinMethod::Reverse => {
                require_arity(name, args, 0)?;
                plan.direction = match plan.direction {
                    RowStreamDirection::Forward => RowStreamDirection::Reverse,
                    RowStreamDirection::Reverse => RowStreamDirection::Forward,
                };
            }
            BuiltinMethod::Filter => {
                let expr = single_expr_arg(name, args)?.clone();
                plan.stages.push(RowStreamStage::Filter(expr));
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

    Ok(Some(plan))
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
        assert!(matches!(plan.stages[0], RowStreamStage::DistinctBy(_)));
        assert!(matches!(plan.stages[1], RowStreamStage::Take(10)));
        assert!(matches!(plan.stages[2], RowStreamStage::Map(_)));
    }
}
