//! Registry-driven stage factory.
//!
//! `lower_method_from_registry` maps a `BuiltinId` and argument list to a
//! concrete `Stage` variant without enumerating every builtin in `lower.rs`.
//! New builtins register their lowering metadata once; this factory picks it up
//! automatically.

use std::sync::Arc;

use crate::ast::Expr;
use crate::builtin_registry::{pipeline_lowering, BuiltinId};
use crate::builtins::{
    BuiltinExprStage, BuiltinMethod, BuiltinNullaryStage, BuiltinPipelineLowering,
    BuiltinSinkAccumulator, BuiltinStringPairStage, BuiltinStringStage, BuiltinUsizeStage,
    BuiltinViewStage,
};

use super::{
    lower::{arg_expr, compile_sort_spec, compile_subexpr, try_decode_map_body},
    NumOp, ReducerOp, ReducerSpec, Sink, Stage,
};

/// Lowers a `BuiltinMethod` call to a concrete `Stage` (or `Sink`) using registry metadata.
///
/// Returns `Some(())` on success or `None` when the method cannot be lowered at this position
/// (e.g. a terminal sink appearing in a non-final position).
pub(super) fn lower_method_from_registry(
    method: BuiltinMethod,
    args: &[crate::ast::Arg],
    is_last: bool,
    stages: &mut Vec<Stage>,
    stage_exprs: &mut Vec<Option<Arc<Expr>>>,
    sink: &mut Sink,
) -> Option<()> {
    let Some(lowering) = pipeline_lowering(BuiltinId::from_method(method)) else {
        if is_last {
            *sink = terminal_sink_for_method(method, args)?;
            return Some(());
        }
        return None;
    };
    match lowering {
        BuiltinPipelineLowering::ExprStage(stage) => {
            if args.len() != 1 {
                return None;
            }
            push_expr_stage(stage, &args[0], stages, stage_exprs)
        }
        BuiltinPipelineLowering::TerminalExprStage { stage, terminal } => {
            if args.len() != 1 {
                return None;
            }
            push_expr_stage(stage, &args[0], stages, stage_exprs)?;
            if is_last {
                set_terminal_sink(terminal, sink)?;
            }
            Some(())
        }
        BuiltinPipelineLowering::NullaryStage(stage) => {
            if !args.is_empty() {
                return None;
            }
            match stage {
                BuiltinNullaryStage::Reverse => {
                    let cancel = method
                        .spec()
                        .cancellation
                        .expect("reverse builtin must define cancellation metadata");
                    stages.push(Stage::Reverse(cancel));
                }
                BuiltinNullaryStage::Unique => stages.push(Stage::UniqueBy(None)),
            }
            stage_exprs.push(None);
            Some(())
        }
        BuiltinPipelineLowering::UsizeStage { stage, min } => {
            if args.len() != 1 {
                return None;
            }
            let n = usize_arg_at_least(&args[0], min)?;
            match stage {
                BuiltinUsizeStage::Take | BuiltinUsizeStage::Skip => {
                    let spec = method.spec();
                    let view_stage = spec.view_stage?;
                    match stage {
                        BuiltinUsizeStage::Take if view_stage == BuiltinViewStage::Take => {
                            stages.push(Stage::Take(n, view_stage, spec.stage_merge?));
                        }
                        BuiltinUsizeStage::Skip if view_stage == BuiltinViewStage::Skip => {
                            stages.push(Stage::Skip(n, view_stage, spec.stage_merge?));
                        }
                        _ => return None,
                    }
                }
                BuiltinUsizeStage::Chunk => stages.push(Stage::Chunk(n)),
                BuiltinUsizeStage::Window => stages.push(Stage::Window(n)),
            }
            stage_exprs.push(None);
            Some(())
        }
        BuiltinPipelineLowering::StringStage(stage) => {
            if args.len() != 1 {
                return None;
            }
            match stage {
                BuiltinStringStage::Split => stages.push(Stage::Split(string_arg(&args[0])?)),
            }
            stage_exprs.push(None);
            Some(())
        }
        BuiltinPipelineLowering::StringPairStage(stage) => {
            if args.len() != 2 {
                return None;
            }
            match stage {
                BuiltinStringPairStage::Replace { all } => stages.push(Stage::Replace {
                    needle: string_arg(&args[0])?,
                    replacement: string_arg(&args[1])?,
                    all,
                }),
            }
            stage_exprs.push(None);
            Some(())
        }
        BuiltinPipelineLowering::Sort => match args {
            [] => {
                stages.push(Stage::Sort(super::SortSpec::identity()));
                stage_exprs.push(None);
                Some(())
            }
            [arg] => {
                let (spec, expr) = compile_sort_spec(arg)?;
                stages.push(Stage::Sort(spec));
                stage_exprs.push(expr);
                Some(())
            }
            _ => None,
        },
        BuiltinPipelineLowering::Slice => match args {
            [arg] => {
                stages.push(Stage::Slice(int_arg(arg)?, None));
                stage_exprs.push(None);
                Some(())
            }
            [start, end] => {
                stages.push(Stage::Slice(int_arg(start)?, Some(int_arg(end)?)));
                stage_exprs.push(None);
                Some(())
            }
            _ => None,
        },
        BuiltinPipelineLowering::TerminalSink if is_last => {
            *sink = terminal_sink_for_method(method, args)?;
            Some(())
        }
        BuiltinPipelineLowering::TerminalSink => None,
    }
}

/// Compiles `arg` into a sub-expression program and appends the corresponding `Stage` variant.
///
/// Returns `None` if `arg` cannot be compiled (e.g. syntax error or unsupported form).
fn push_expr_stage(
    stage: BuiltinExprStage,
    arg: &crate::ast::Arg,
    stages: &mut Vec<Stage>,
    stage_exprs: &mut Vec<Option<Arc<Expr>>>,
) -> Option<()> {
    match stage {
        BuiltinExprStage::Filter => {
            stages.push(Stage::Filter(
                compile_subexpr(arg)?,
                BuiltinViewStage::Filter,
            ));
            stage_exprs.push(arg_expr(arg));
        }
        BuiltinExprStage::Map => match try_decode_map_body(arg) {
            Some(plan) => {
                stages.push(Stage::CompiledMap(Arc::new(plan)));
                stage_exprs.push(arg_expr(arg));
            }
            None => {
                stages.push(Stage::Map(compile_subexpr(arg)?, BuiltinViewStage::Map));
                stage_exprs.push(arg_expr(arg));
            }
        },
        BuiltinExprStage::FlatMap => {
            stages.push(Stage::FlatMap(
                compile_subexpr(arg)?,
                BuiltinViewStage::FlatMap,
            ));
            stage_exprs.push(arg_expr(arg));
        }
        BuiltinExprStage::TakeWhile => {
            stages.push(Stage::TakeWhile(compile_subexpr(arg)?));
            stage_exprs.push(arg_expr(arg));
        }
        BuiltinExprStage::DropWhile => {
            stages.push(Stage::DropWhile(compile_subexpr(arg)?));
            stage_exprs.push(arg_expr(arg));
        }
        BuiltinExprStage::IndicesWhere => {
            stages.push(Stage::IndicesWhere(compile_subexpr(arg)?));
            stage_exprs.push(arg_expr(arg));
        }
        BuiltinExprStage::FindIndex => {
            stages.push(Stage::FindIndex(compile_subexpr(arg)?));
            stage_exprs.push(arg_expr(arg));
        }
        BuiltinExprStage::MaxBy => {
            stages.push(Stage::MaxBy(compile_subexpr(arg)?));
            stage_exprs.push(arg_expr(arg));
        }
        BuiltinExprStage::MinBy => {
            stages.push(Stage::MinBy(compile_subexpr(arg)?));
            stage_exprs.push(arg_expr(arg));
        }
        BuiltinExprStage::UniqueBy => {
            stages.push(Stage::UniqueBy(Some(compile_subexpr(arg)?)));
            stage_exprs.push(arg_expr(arg));
        }
        BuiltinExprStage::GroupBy => {
            stages.push(Stage::GroupBy(compile_subexpr(arg)?));
            stage_exprs.push(arg_expr(arg));
        }
        BuiltinExprStage::CountBy => {
            stages.push(Stage::CountBy(compile_subexpr(arg)?));
            stage_exprs.push(arg_expr(arg));
        }
        BuiltinExprStage::IndexBy => {
            stages.push(Stage::IndexBy(compile_subexpr(arg)?));
            stage_exprs.push(arg_expr(arg));
        }
        BuiltinExprStage::TransformValues => {
            stages.push(Stage::TransformValues(compile_subexpr(arg)?));
            stage_exprs.push(arg_expr(arg));
        }
        BuiltinExprStage::TransformKeys => {
            stages.push(Stage::TransformKeys(compile_subexpr(arg)?));
            stage_exprs.push(arg_expr(arg));
        }
        BuiltinExprStage::FilterValues => {
            stages.push(Stage::FilterValues(compile_subexpr(arg)?));
            stage_exprs.push(arg_expr(arg));
        }
        BuiltinExprStage::FilterKeys => {
            stages.push(Stage::FilterKeys(compile_subexpr(arg)?));
            stage_exprs.push(arg_expr(arg));
        }
    }
    Some(())
}

/// Writes the terminal `Sink` value derived from `method`'s sink spec into `*sink`.
///
/// Returns `None` for numeric reducers that require argument-based configuration.
fn set_terminal_sink(method: BuiltinMethod, sink: &mut Sink) -> Option<()> {
    let spec = method.spec();
    match spec.sink?.accumulator {
        BuiltinSinkAccumulator::SelectOne(_) => *sink = Sink::Terminal(method),
        BuiltinSinkAccumulator::Count => *sink = Sink::Reducer(ReducerSpec::count()),
        BuiltinSinkAccumulator::ApproxDistinct => *sink = Sink::ApproxCountDistinct,
        BuiltinSinkAccumulator::Numeric => return None,
    }
    Some(())
}

/// Extracts a `usize` integer literal from `arg`, enforcing `value >= min`.
///
/// Returns `None` if the argument is not a non-negative integer or is below the minimum.
fn usize_arg_at_least(arg: &crate::ast::Arg, min: usize) -> Option<usize> {
    match arg {
        crate::ast::Arg::Pos(Expr::Int(n)) if *n >= min as i64 => Some(*n as usize),
        _ => None,
    }
}

/// Extracts a signed integer literal from `arg`, accepting any `i64` value.
fn int_arg(arg: &crate::ast::Arg) -> Option<i64> {
    match arg {
        crate::ast::Arg::Pos(Expr::Int(n)) => Some(*n),
        _ => None,
    }
}

/// Extracts a string literal from `arg` and interns it as `Arc<str>`.
fn string_arg(arg: &crate::ast::Arg) -> Option<Arc<str>> {
    match arg {
        crate::ast::Arg::Pos(Expr::Str(s)) => Some(Arc::<str>::from(s.as_str())),
        _ => None,
    }
}

/// Constructs the terminal `Sink` for `method` given its argument list.
///
/// Handles count predicates, numeric reducers with optional projection, and positional selects.
fn terminal_sink_for_method(method: BuiltinMethod, args: &[crate::ast::Arg]) -> Option<Sink> {
    let spec = method.spec();
    match spec.sink?.accumulator {
        BuiltinSinkAccumulator::ApproxDistinct if args.is_empty() => {
            Some(Sink::ApproxCountDistinct)
        }
        BuiltinSinkAccumulator::Count => match args {
            [] => Some(Sink::Reducer(ReducerSpec::count())),
            [arg] if method == BuiltinMethod::Count => Some(Sink::Reducer(ReducerSpec {
                op: ReducerOp::Count,
                predicate: Some(compile_subexpr(arg)?),
                projection: None,
                predicate_expr: arg_expr(arg),
                projection_expr: None,
            })),
            _ => None,
        },
        BuiltinSinkAccumulator::Numeric => Some(Sink::Reducer(ReducerSpec {
            op: ReducerOp::Numeric(NumOp::from_builtin_reducer(spec.numeric_reducer?)),
            predicate: None,
            projection: match args {
                [] => None,
                [arg] => Some(compile_subexpr(arg)?),
                _ => return None,
            },
            predicate_expr: None,
            projection_expr: match args {
                [] => None,
                [arg] => arg_expr(arg),
                _ => return None,
            },
        })),
        BuiltinSinkAccumulator::SelectOne(_) if args.is_empty() => Some(Sink::Terminal(method)),
        _ => None,
    }
}
