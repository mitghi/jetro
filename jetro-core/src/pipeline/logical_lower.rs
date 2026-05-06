//! Lowers an optimized `LogicalPlan` directly into a `Pipeline` by mapping each plan node
//! to the corresponding `Stage` or `Sink` variant without reconstructing an intermediate `Expr`.
//!
//! This replaces the earlier round-trip approach (LogicalPlan → Expr → Pipeline::lower()) and
//! allows the logical optimizer to fire on all queries the planner handles, not just the subset
//! that survive the Expr reconstruction step.

use std::sync::Arc;

use crate::parse::ast::{Expr, Step};
use crate::builtins::{BuiltinMethod, BuiltinViewStage};
use crate::logical_plan::LogicalPlan;
use crate::pipeline::{
    plan_with_exprs, BodyKernel, NumOp, Pipeline, PipelineBody, ReducerOp,
    ReducerSpec, Sink, Source, Stage,
};

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Lower an optimized `LogicalPlan` directly into a compiled `Pipeline`.
///
/// Returns `None` when the plan contains nodes that cannot be expressed as a
/// linear pipeline (e.g. `ScalarExpr`).
pub(crate) fn try_lower(plan: LogicalPlan) -> Option<Pipeline> {
    let (source, stages, stage_exprs, sink) = collect(plan)?;
    let body = build_body(stages, stage_exprs, sink);
    Some(body.with_source(source))
}

// ---------------------------------------------------------------------------
// Plan collection — walks the tree and accumulates source, stages, and sink
// ---------------------------------------------------------------------------

/// Walks `plan` top-down, collecting the source, an ordered list of stages (with parallel
/// expression slots), and the terminal sink.  Returns `None` for plan shapes that cannot be
/// lowered to a linear pipeline.
fn collect(
    plan: LogicalPlan,
) -> Option<(Source, Vec<Stage>, Vec<Option<Arc<Expr>>>, Sink)> {
    match plan {
        // ── Leaf: the row source ───────────────────────────────────────────
        LogicalPlan::Source(source) => Some((source, vec![], vec![], Sink::Collect)),

        // ── Streaming transforms ───────────────────────────────────────────
        LogicalPlan::Filter { input, predicate } => {
            let (source, mut stages, mut exprs, sink) = collect(*input)?;
            let prog = compile_expr_body(&predicate);
            stages.push(Stage::Filter(prog, BuiltinViewStage::Filter));
            exprs.push(Some(Arc::new(predicate)));
            Some((source, stages, exprs, sink))
        }

        LogicalPlan::Map { input, projection } => {
            let (source, mut stages, mut exprs, sink) = collect(*input)?;
            let prog = compile_expr_body(&projection);
            stages.push(Stage::Map(prog, BuiltinViewStage::Map));
            exprs.push(Some(Arc::new(projection)));
            Some((source, stages, exprs, sink))
        }

        LogicalPlan::FlatMap { input, expansion } => {
            let (source, mut stages, mut exprs, sink) = collect(*input)?;
            let prog = compile_expr_body(&expansion);
            stages.push(Stage::FlatMap(prog, BuiltinViewStage::FlatMap));
            exprs.push(Some(Arc::new(expansion)));
            Some((source, stages, exprs, sink))
        }

        LogicalPlan::TakeWhile { input, predicate } => {
            let (source, mut stages, mut exprs, sink) = collect(*input)?;
            let prog = compile_expr_body(&predicate);
            stages.push(Stage::ExprBuiltin {
                method: BuiltinMethod::TakeWhile,
                body: prog,
            });
            exprs.push(Some(Arc::new(predicate)));
            Some((source, stages, exprs, sink))
        }

        LogicalPlan::DropWhile { input, predicate } => {
            let (source, mut stages, mut exprs, sink) = collect(*input)?;
            let prog = compile_expr_body(&predicate);
            stages.push(Stage::ExprBuiltin {
                method: BuiltinMethod::DropWhile,
                body: prog,
            });
            exprs.push(Some(Arc::new(predicate)));
            Some((source, stages, exprs, sink))
        }

        // ── Positional ─────────────────────────────────────────────────────
        LogicalPlan::Take { input, n } => {
            let (source, mut stages, mut exprs, sink) = collect(*input)?;
            stages.push(Stage::UsizeBuiltin {
                method: BuiltinMethod::Take,
                value: n,
            });
            exprs.push(None);
            Some((source, stages, exprs, sink))
        }

        LogicalPlan::Skip { input, n } => {
            let (source, mut stages, mut exprs, sink) = collect(*input)?;
            stages.push(Stage::UsizeBuiltin {
                method: BuiltinMethod::Skip,
                value: n,
            });
            exprs.push(None);
            Some((source, stages, exprs, sink))
        }

        // ── Ordering / dedup ───────────────────────────────────────────────
        LogicalPlan::Sort { input, spec } => {
            let (source, mut stages, mut exprs, sink) = collect(*input)?;
            stages.push(Stage::Sort(spec));
            exprs.push(None);
            Some((source, stages, exprs, sink))
        }

        LogicalPlan::Unique { input, key } => {
            let (source, mut stages, mut exprs, sink) = collect(*input)?;
            match key {
                None => {
                    stages.push(Stage::UniqueBy(None));
                    exprs.push(None);
                }
                Some(key_expr) => {
                    let prog = compile_expr_body(&key_expr);
                    stages.push(Stage::UniqueBy(Some(prog)));
                    exprs.push(Some(Arc::new(key_expr)));
                }
            }
            Some((source, stages, exprs, sink))
        }

        LogicalPlan::Reverse { input } => {
            let (source, mut stages, mut exprs, sink) = collect(*input)?;
            let cancel = BuiltinMethod::Reverse
                .spec()
                .cancellation
                .expect("Reverse must have cancellation metadata");
            stages.push(Stage::Reverse(cancel));
            exprs.push(None);
            Some((source, stages, exprs, sink))
        }

        // ── Keyed reducers ─────────────────────────────────────────────────
        LogicalPlan::GroupBy { input, key } => {
            let (source, mut stages, mut exprs, sink) = collect(*input)?;
            let prog = compile_expr_body(&key);
            stages.push(Stage::ExprBuiltin {
                method: BuiltinMethod::GroupBy,
                body: prog,
            });
            exprs.push(Some(Arc::new(key)));
            Some((source, stages, exprs, sink))
        }

        LogicalPlan::CountBy { input, key } => {
            let (source, mut stages, mut exprs, sink) = collect(*input)?;
            let prog = compile_expr_body(&key);
            stages.push(Stage::ExprBuiltin {
                method: BuiltinMethod::CountBy,
                body: prog,
            });
            exprs.push(Some(Arc::new(key)));
            Some((source, stages, exprs, sink))
        }

        LogicalPlan::IndexBy { input, key } => {
            let (source, mut stages, mut exprs, sink) = collect(*input)?;
            let prog = compile_expr_body(&key);
            stages.push(Stage::ExprBuiltin {
                method: BuiltinMethod::IndexBy,
                body: prog,
            });
            exprs.push(Some(Arc::new(key)));
            Some((source, stages, exprs, sink))
        }

        // ── Terminal sinks — strip the default Collect, install the real one ──
        LogicalPlan::First(inner) => {
            let (source, stages, exprs, _) = collect(*inner)?;
            Some((source, stages, exprs, Sink::Terminal(BuiltinMethod::First)))
        }
        LogicalPlan::Last(inner) => {
            let (source, stages, exprs, _) = collect(*inner)?;
            Some((source, stages, exprs, Sink::Terminal(BuiltinMethod::Last)))
        }
        LogicalPlan::Count(inner) => {
            let (source, stages, exprs, _) = collect(*inner)?;
            Some((source, stages, exprs, Sink::Reducer(ReducerSpec::count())))
        }
        LogicalPlan::Sum(inner) => {
            let (source, stages, exprs, _) = collect(*inner)?;
            Some((source, stages, exprs, Sink::Reducer(ReducerSpec {
                op: ReducerOp::Numeric(NumOp::Sum),
                predicate: None,
                projection: None,
                predicate_expr: None,
                projection_expr: None,
            })))
        }
        LogicalPlan::Avg(inner) => {
            let (source, stages, exprs, _) = collect(*inner)?;
            Some((source, stages, exprs, Sink::Reducer(ReducerSpec {
                op: ReducerOp::Numeric(NumOp::Avg),
                predicate: None,
                projection: None,
                predicate_expr: None,
                projection_expr: None,
            })))
        }
        LogicalPlan::Min(inner) => {
            let (source, stages, exprs, _) = collect(*inner)?;
            Some((source, stages, exprs, Sink::Reducer(ReducerSpec {
                op: ReducerOp::Numeric(NumOp::Min),
                predicate: None,
                projection: None,
                predicate_expr: None,
                projection_expr: None,
            })))
        }
        LogicalPlan::Max(inner) => {
            let (source, stages, exprs, _) = collect(*inner)?;
            Some((source, stages, exprs, Sink::Reducer(ReducerSpec {
                op: ReducerOp::Numeric(NumOp::Max),
                predicate: None,
                projection: None,
                predicate_expr: None,
                projection_expr: None,
            })))
        }
        LogicalPlan::ApproxCountDistinct(inner) => {
            let (source, stages, exprs, _) = collect(*inner)?;
            Some((source, stages, exprs, Sink::ApproxCountDistinct))
        }

        // ── VM fallback — cannot lower to a Pipeline ───────────────────────
        LogicalPlan::ScalarExpr(_) => None,
    }
}

// ---------------------------------------------------------------------------
// Body assembly
// ---------------------------------------------------------------------------

/// Classifies stages into `BodyKernel`s, runs `plan_with_exprs` for filter reordering/fusion,
/// and fills in the kernel vectors required by `PipelineBody`.
fn build_body(
    stages: Vec<Stage>,
    stage_exprs: Vec<Option<Arc<Expr>>>,
    sink: Sink,
) -> PipelineBody {
    let classify_kernels = |stages: &[Stage]| -> Vec<BodyKernel> {
        stages
            .iter()
            .map(|s| {
                s.body_program()
                    .map(BodyKernel::classify)
                    .unwrap_or(BodyKernel::Generic)
            })
            .collect()
    };

    let kernels = classify_kernels(&stages);
    let plan_result = plan_with_exprs(stages, stage_exprs, &kernels, sink);

    let stage_kernels = classify_kernels(&plan_result.stages);
    let sink_kernels = plan_result
        .sink
        .reducer_spec()
        .map(|spec| {
            spec.sink_programs()
                .map(|p| BodyKernel::classify(p))
                .collect()
        })
        .unwrap_or_default();

    PipelineBody {
        stages: plan_result.stages,
        stage_exprs: plan_result.stage_exprs,
        sink: plan_result.sink,
        stage_kernels,
        sink_kernels,
    }
}

// ---------------------------------------------------------------------------
// Expression compilation
// ---------------------------------------------------------------------------

/// Compiles `expr` to a `Program`, rewriting a bare `Ident` to `@.<ident>` so that
/// identifiers in stage body position resolve against the current row, not the document root.
///
/// This replicates the `Ident→@.field` rewrite that `compile_subexpr` performs in `lower.rs`,
/// without requiring access to the `pub(super)` helper there.
fn compile_expr_body(expr: &Expr) -> Arc<crate::vm::Program> {
    let rooted: Expr = match expr {
        Expr::Ident(name) => {
            Expr::Chain(Box::new(Expr::Current), vec![Step::Field(name.clone())])
        }
        other => other.clone(),
    };
    Arc::new(crate::compile::compiler::Compiler::compile(&rooted, ""))
}
