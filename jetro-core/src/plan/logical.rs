//! Logical planner: translates a pipeline-shaped `Expr` into a `LogicalPlan`.
//!
//! Handles the same patterns as `Pipeline::lower()` in `pipeline/lower.rs` but
//! produces uncompiled `LogicalPlan` nodes instead. Expressions that cannot be
//! classified as pipeline stages return `None`, signalling fallback to the
//! existing `Pipeline::lower()` path.

use std::sync::Arc;

use crate::builtins::{
    registry::{logical_shape, pipeline_accepts_arity, BuiltinId, BuiltinLogicalShape},
    BuiltinMethod,
};
use crate::exec::pipeline::{SortSpec, Source};
use crate::ir::logical::LogicalPlan;
use crate::parse::ast::{Arg, Expr, Step};

/// Try to lower a pipeline-shaped `Expr` to a `LogicalPlan`.
/// Returns `None` for expressions that are not pipeline-shaped.
pub(crate) fn try_lower(expr: &Expr) -> Option<LogicalPlan> {
    let (source, steps) = extract_source_and_steps(expr)?;
    let base = LogicalPlan::Source(source);
    apply_steps(base, steps)
}

/// Extracts the leading field chain from a `$.<field>.<field>...` expression
/// and returns the source and remaining steps. Returns `None` if the expression
/// is not rooted at `$` or has no leading field steps.
fn extract_source_and_steps(expr: &Expr) -> Option<(Source, &[Step])> {
    let (base, steps) = match expr {
        Expr::Chain(b, s) => (b.as_ref(), s.as_slice()),
        _ => return None,
    };
    if !matches!(base, Expr::Root) {
        return None;
    }

    let mut field_end = 0;
    for s in steps {
        match s {
            Step::Field(_) => field_end += 1,
            _ => break,
        }
    }
    if field_end == 0 {
        return None;
    }

    let keys: Arc<[Arc<str>]> = steps[..field_end]
        .iter()
        .map(|s| match s {
            Step::Field(k) => Arc::<str>::from(k.as_str()),
            _ => unreachable!(),
        })
        .collect::<Vec<_>>()
        .into();

    Some((Source::FieldChain { keys }, &steps[field_end..]))
}

fn apply_steps(mut plan: LogicalPlan, steps: &[Step]) -> Option<LogicalPlan> {
    for (idx, step) in steps.iter().enumerate() {
        plan = apply_step(plan, step, idx == steps.len() - 1)?;
    }
    Some(plan)
}

fn apply_step(plan: LogicalPlan, step: &Step, is_last: bool) -> Option<LogicalPlan> {
    match step {
        Step::Method(name, args) => apply_method(plan, name.as_str(), args, is_last),
        // Field, OptField, Index, etc. — cannot classify as pipeline stage
        _ => None,
    }
}

fn apply_method(
    input: LogicalPlan,
    name: &str,
    args: &[Arg],
    is_last: bool,
) -> Option<LogicalPlan> {
    let method = BuiltinMethod::from_name(name);
    if method == BuiltinMethod::Unknown {
        return None;
    }
    let id = BuiltinId::from_method(method);
    if !pipeline_accepts_arity(id, args.len(), is_last) {
        return None;
    }

    let plan = match logical_shape(id)? {
        BuiltinLogicalShape::Filter => {
            let pred = single_expr_arg(args)?;
            LogicalPlan::Filter {
                input: Box::new(input),
                predicate: pred.clone(),
            }
        }
        BuiltinLogicalShape::FilterThenFirst => {
            let pred = single_expr_arg(args)?;
            let filtered = LogicalPlan::Filter {
                input: Box::new(input),
                predicate: pred.clone(),
            };
            if is_last {
                LogicalPlan::First(Box::new(filtered))
            } else {
                filtered
            }
        }
        BuiltinLogicalShape::Map => {
            let proj = single_expr_arg(args)?;
            LogicalPlan::Map {
                input: Box::new(input),
                projection: proj.clone(),
            }
        }
        BuiltinLogicalShape::FlatMap => {
            let exp = single_expr_arg(args)?;
            LogicalPlan::FlatMap {
                input: Box::new(input),
                expansion: exp.clone(),
            }
        }
        BuiltinLogicalShape::Take => {
            let n = single_usize_arg(args)?;
            LogicalPlan::Take {
                input: Box::new(input),
                n,
            }
        }
        BuiltinLogicalShape::Skip => {
            let n = single_usize_arg(args)?;
            LogicalPlan::Skip {
                input: Box::new(input),
                n,
            }
        }
        BuiltinLogicalShape::First => {
            if !args.is_empty() {
                return None;
            }
            LogicalPlan::First(Box::new(input))
        }
        BuiltinLogicalShape::Last => {
            if !args.is_empty() {
                return None;
            }
            LogicalPlan::Last(Box::new(input))
        }
        BuiltinLogicalShape::Sum => {
            // sum() with no args — sum with projection arg is handled by Pipeline::lower
            if !args.is_empty() {
                return None;
            }
            LogicalPlan::Sum(Box::new(input))
        }
        BuiltinLogicalShape::Avg => {
            if !args.is_empty() {
                return None;
            }
            LogicalPlan::Avg(Box::new(input))
        }
        BuiltinLogicalShape::Min => {
            if !args.is_empty() {
                return None;
            }
            LogicalPlan::Min(Box::new(input))
        }
        BuiltinLogicalShape::Max => {
            if !args.is_empty() {
                return None;
            }
            LogicalPlan::Max(Box::new(input))
        }
        BuiltinLogicalShape::Count => {
            // count() with no args; count(pred) falls through to Pipeline::lower
            if !args.is_empty() {
                return None;
            }
            LogicalPlan::Count(Box::new(input))
        }
        BuiltinLogicalShape::Reverse => LogicalPlan::Reverse {
            input: Box::new(input),
        },
        BuiltinLogicalShape::TakeWhile => {
            let pred = single_expr_arg(args)?;
            LogicalPlan::TakeWhile {
                input: Box::new(input),
                predicate: pred.clone(),
            }
        }
        BuiltinLogicalShape::DropWhile => {
            let pred = single_expr_arg(args)?;
            LogicalPlan::DropWhile {
                input: Box::new(input),
                predicate: pred.clone(),
            }
        }
        BuiltinLogicalShape::Sort => match args.len() {
            0 => LogicalPlan::Sort {
                input: Box::new(input),
                spec: SortSpec::identity(),
            },
            1 => {
                let (spec, _) = crate::exec::pipeline::compile_sort_spec(&args[0])?;
                LogicalPlan::Sort {
                    input: Box::new(input),
                    spec,
                }
            }
            _ => return None,
        },
        BuiltinLogicalShape::Unique => LogicalPlan::Unique {
            input: Box::new(input),
            key: None,
        },
        BuiltinLogicalShape::UniqueBy => {
            let key = single_expr_arg(args)?;
            LogicalPlan::Unique {
                input: Box::new(input),
                key: Some(key.clone()),
            }
        }
        BuiltinLogicalShape::GroupBy => {
            let key = single_expr_arg(args)?;
            LogicalPlan::GroupBy {
                input: Box::new(input),
                key: key.clone(),
            }
        }
        BuiltinLogicalShape::CountBy => {
            let key = single_expr_arg(args)?;
            let keyed = LogicalPlan::CountBy {
                input: Box::new(input),
                key: key.clone(),
            };
            if is_last {
                LogicalPlan::First(Box::new(keyed))
            } else {
                keyed
            }
        }
        BuiltinLogicalShape::IndexBy => {
            let key = single_expr_arg(args)?;
            let keyed = LogicalPlan::IndexBy {
                input: Box::new(input),
                key: key.clone(),
            };
            if is_last {
                LogicalPlan::First(Box::new(keyed))
            } else {
                keyed
            }
        }
        BuiltinLogicalShape::ApproxCountDistinct => {
            LogicalPlan::ApproxCountDistinct(Box::new(input))
        }
    };
    Some(plan)
}

fn single_expr_arg(args: &[Arg]) -> Option<&Expr> {
    match args {
        [Arg::Pos(e)] => Some(e),
        _ => None,
    }
}

fn single_usize_arg(args: &[Arg]) -> Option<usize> {
    match args {
        [Arg::Pos(Expr::Int(n))] if *n >= 0 => Some(*n as usize),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parser::parse;

    fn lower(query: &str) -> LogicalPlan {
        let expr = parse(query).expect("parse");
        try_lower(&expr).expect("logical lower")
    }

    #[test]
    fn terminal_find_uses_filter_first_shape() {
        let plan = lower("$.xs.find(score > 5)");

        let LogicalPlan::First(inner) = plan else {
            panic!("expected terminal First");
        };
        assert!(matches!(*inner, LogicalPlan::Filter { .. }));
    }

    #[test]
    fn terminal_find_first_uses_filter_first_shape() {
        let plan = lower("$.xs.find_first(score > 5)");

        let LogicalPlan::First(inner) = plan else {
            panic!("expected terminal First");
        };
        assert!(matches!(*inner, LogicalPlan::Filter { .. }));
    }

    #[test]
    fn non_terminal_find_stays_streaming_filter() {
        let plan = lower("$.xs.find(score > 5).map(name)");

        let LogicalPlan::Map { input, .. } = plan else {
            panic!("expected map");
        };
        assert!(matches!(*input, LogicalPlan::Filter { .. }));
    }

    #[test]
    fn terminal_only_sinks_do_not_lower_mid_chain() {
        let expr = parse("$.xs.first().map(name)").expect("parse");
        assert!(try_lower(&expr).is_none());
    }

    #[test]
    fn select_many_first_last_fall_back_to_pipeline_lowerer() {
        let first = parse("$.xs.first(2)").expect("parse");
        assert!(try_lower(&first).is_none());

        let last = parse("$.xs.last(2)").expect("parse");
        assert!(try_lower(&last).is_none());
    }

    #[test]
    fn terminal_keyed_reducers_use_registry_terminal_shape() {
        let count_by = lower("$.xs.count_by(@.kind)");
        let LogicalPlan::First(inner) = count_by else {
            panic!("expected count_by terminal First");
        };
        assert!(matches!(*inner, LogicalPlan::CountBy { .. }));

        let index_by = lower("$.xs.index_by(@.id)");
        let LogicalPlan::First(inner) = index_by else {
            panic!("expected index_by terminal First");
        };
        assert!(matches!(*inner, LogicalPlan::IndexBy { .. }));
    }
}
