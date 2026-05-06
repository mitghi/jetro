//! Logical planner: translates a pipeline-shaped `Expr` into a `LogicalPlan`.
//!
//! Handles the same patterns as `Pipeline::lower()` in `pipeline/lower.rs` but
//! produces uncompiled `LogicalPlan` nodes instead. Expressions that cannot be
//! classified as pipeline stages return `None`, signalling fallback to the
//! existing `Pipeline::lower()` path.

use std::sync::Arc;

use crate::parse::ast::{Arg, Expr, Step};
use crate::builtins::BuiltinMethod;
use crate::logical_plan::LogicalPlan;
use crate::pipeline::{SortSpec, Source};

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
    for step in steps {
        plan = apply_step(plan, step)?;
    }
    Some(plan)
}

fn apply_step(plan: LogicalPlan, step: &Step) -> Option<LogicalPlan> {
    match step {
        Step::Method(name, args) => apply_method(plan, name.as_str(), args),
        // Field, OptField, Index, etc. — cannot classify as pipeline stage
        _ => None,
    }
}

fn apply_method(input: LogicalPlan, name: &str, args: &[Arg]) -> Option<LogicalPlan> {
    // Look up the BuiltinMethod from the name
    let method = BuiltinMethod::from_name(name);
    if method == BuiltinMethod::Unknown {
        return None;
    }

    let plan = match method {
        BuiltinMethod::Filter | BuiltinMethod::Find | BuiltinMethod::FindAll => {
            let pred = single_expr_arg(args)?;
            LogicalPlan::Filter {
                input: Box::new(input),
                predicate: pred.clone(),
            }
        }
        BuiltinMethod::Map => {
            let proj = single_expr_arg(args)?;
            LogicalPlan::Map {
                input: Box::new(input),
                projection: proj.clone(),
            }
        }
        BuiltinMethod::FlatMap => {
            let exp = single_expr_arg(args)?;
            LogicalPlan::FlatMap {
                input: Box::new(input),
                expansion: exp.clone(),
            }
        }
        BuiltinMethod::Take => {
            let n = single_usize_arg(args)?;
            LogicalPlan::Take {
                input: Box::new(input),
                n,
            }
        }
        BuiltinMethod::Skip => {
            let n = single_usize_arg(args)?;
            LogicalPlan::Skip {
                input: Box::new(input),
                n,
            }
        }
        BuiltinMethod::First => {
            // first() takes 0 args in pipeline position
            if !args.is_empty() { return None; }
            LogicalPlan::First(Box::new(input))
        }
        BuiltinMethod::Last => {
            if !args.is_empty() { return None; }
            LogicalPlan::Last(Box::new(input))
        }
        BuiltinMethod::Sum => {
            // sum() with no args — sum with projection arg is handled by Pipeline::lower
            if !args.is_empty() { return None; }
            LogicalPlan::Sum(Box::new(input))
        }
        BuiltinMethod::Avg => {
            if !args.is_empty() { return None; }
            LogicalPlan::Avg(Box::new(input))
        }
        BuiltinMethod::Min => {
            if !args.is_empty() { return None; }
            LogicalPlan::Min(Box::new(input))
        }
        BuiltinMethod::Max => {
            if !args.is_empty() { return None; }
            LogicalPlan::Max(Box::new(input))
        }
        BuiltinMethod::Count => {
            // count() with no args; count(pred) falls through to Pipeline::lower
            if !args.is_empty() { return None; }
            LogicalPlan::Count(Box::new(input))
        }
        BuiltinMethod::Reverse => LogicalPlan::Reverse {
            input: Box::new(input),
        },
        BuiltinMethod::TakeWhile => {
            let pred = single_expr_arg(args)?;
            LogicalPlan::TakeWhile {
                input: Box::new(input),
                predicate: pred.clone(),
            }
        }
        BuiltinMethod::DropWhile => {
            let pred = single_expr_arg(args)?;
            LogicalPlan::DropWhile {
                input: Box::new(input),
                predicate: pred.clone(),
            }
        }
        BuiltinMethod::Sort => {
            match args.len() {
                0 => LogicalPlan::Sort {
                    input: Box::new(input),
                    spec: SortSpec::identity(),
                },
                1 => {
                    // Compile the key arg into a SortSpec.
                    // UnaryNeg wrapping means descending order.
                    let arg_expr = match &args[0] {
                        Arg::Pos(e) => e,
                        _ => return None,
                    };
                    let (key_expr, descending) = match arg_expr {
                        Expr::UnaryNeg(inner) => (inner.as_ref().clone(), true),
                        other => (other.clone(), false),
                    };
                    // Rewrite bare Ident to @.field (body context).
                    let rooted: Expr = match &key_expr {
                        Expr::Ident(name) => Expr::Chain(
                            Box::new(Expr::Current),
                            vec![Step::Field(name.clone())],
                        ),
                        other => other.clone(),
                    };
                    let key_prog = Arc::new(crate::compile::compiler::Compiler::compile(&rooted, ""));
                    LogicalPlan::Sort {
                        input: Box::new(input),
                        spec: SortSpec::keyed(key_prog, descending),
                    }
                }
                _ => return None,
            }
        }
        BuiltinMethod::Unique => LogicalPlan::Unique {
            input: Box::new(input),
            key: None,
        },
        BuiltinMethod::UniqueBy => {
            let key = single_expr_arg(args)?;
            LogicalPlan::Unique {
                input: Box::new(input),
                key: Some(key.clone()),
            }
        }
        BuiltinMethod::GroupBy => {
            let key = single_expr_arg(args)?;
            LogicalPlan::GroupBy {
                input: Box::new(input),
                key: key.clone(),
            }
        }
        BuiltinMethod::CountBy => {
            let key = single_expr_arg(args)?;
            LogicalPlan::CountBy {
                input: Box::new(input),
                key: key.clone(),
            }
        }
        BuiltinMethod::IndexBy => {
            let key = single_expr_arg(args)?;
            LogicalPlan::IndexBy {
                input: Box::new(input),
                key: key.clone(),
            }
        }
        BuiltinMethod::ApproxCountDistinct => {
            LogicalPlan::ApproxCountDistinct(Box::new(input))
        }
        // Not a recognised pipeline operator — fall through to existing path
        _ => return None,
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

