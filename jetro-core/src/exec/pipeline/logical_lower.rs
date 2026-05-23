//! Lowers an optimized `LogicalPlan` directly into a `Pipeline` by mapping each plan node
//! to the corresponding `Stage` or `Sink` variant without reconstructing an intermediate `Expr`.
//!
//! This replaces the earlier round-trip approach (LogicalPlan → Expr → Pipeline::lower()) and
//! allows the logical optimizer to fire on all queries the planner handles, not just the subset
//! that survive the Expr reconstruction step.

use std::sync::Arc;

use crate::builtins::registry::BuiltinId;
use crate::exec::pipeline::{Pipeline, PipelineBody, Sink, Source, Stage};
use crate::ir::logical::LogicalPlan;
use crate::parse::ast::Expr;

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Lower an optimized `LogicalPlan` directly into a compiled `Pipeline`.
///
/// Returns `None` when the plan contains nodes that cannot be expressed as a
/// linear pipeline (e.g. `ScalarExpr`).
pub(crate) fn try_lower(plan: LogicalPlan) -> Option<Pipeline> {
    let (source, stages, stage_exprs, sink) = collect(plan)?;
    let body = PipelineBody::planned(stages, stage_exprs, sink);
    Some(body.with_source(source))
}

// ---------------------------------------------------------------------------
// Plan collection — walks the tree and accumulates source, stages, and sink
// ---------------------------------------------------------------------------

/// Walks `plan` top-down, collecting the source, an ordered list of stages (with parallel
/// expression slots), and the terminal sink.  Returns `None` for plan shapes that cannot be
/// lowered to a linear pipeline.
fn collect(plan: LogicalPlan) -> Option<(Source, Vec<Stage>, Vec<Option<Arc<Expr>>>, Sink)> {
    let id = plan.builtin_id();
    match plan {
        LogicalPlan::Source(source) => Some((source, vec![], vec![], Sink::Collect)),

        LogicalPlan::Filter { input, predicate } => {
            let (source, mut stages, mut exprs, sink) = collect(*input)?;
            let prog = compile_expr_body(&predicate);
            stages.push(Stage::expr_stage_builtin_id(id?, prog)?);
            exprs.push(Some(Arc::new(predicate)));
            Some((source, stages, exprs, sink))
        }

        LogicalPlan::Map { input, projection } => {
            let (source, mut stages, mut exprs, sink) = collect(*input)?;
            let prog = compile_expr_body(&projection);
            stages.push(Stage::expr_stage_builtin_id(id?, prog)?);
            exprs.push(Some(Arc::new(projection)));
            Some((source, stages, exprs, sink))
        }

        LogicalPlan::FlatMap { input, expansion } => {
            let (source, mut stages, mut exprs, sink) = collect(*input)?;
            let prog = compile_expr_body(&expansion);
            stages.push(Stage::expr_stage_builtin_id(id?, prog)?);
            exprs.push(Some(Arc::new(expansion)));
            Some((source, stages, exprs, sink))
        }

        LogicalPlan::TakeWhile { input, predicate } => {
            collect_expr_builtin_stage(*input, id?, predicate)
        }

        LogicalPlan::DropWhile { input, predicate } => {
            collect_expr_builtin_stage(*input, id?, predicate)
        }

        LogicalPlan::Take { input, n } => {
            let (source, mut stages, mut exprs, sink) = collect(*input)?;
            stages.push(Stage::usize_builtin_id(id?, n)?);
            exprs.push(None);
            Some((source, stages, exprs, sink))
        }

        LogicalPlan::Skip { input, n } => {
            let (source, mut stages, mut exprs, sink) = collect(*input)?;
            stages.push(Stage::usize_builtin_id(id?, n)?);
            exprs.push(None);
            Some((source, stages, exprs, sink))
        }

        LogicalPlan::Sort { input, spec } => {
            let (source, mut stages, mut exprs, sink) = collect(*input)?;
            stages.push(Stage::sort_builtin_id(id?, spec)?);
            exprs.push(None);
            Some((source, stages, exprs, sink))
        }

        LogicalPlan::Unique { input, key } => {
            let (source, mut stages, mut exprs, sink) = collect(*input)?;
            match key {
                None => {
                    stages.push(Stage::nullary_builtin_id(id?)?);
                    exprs.push(None);
                }
                Some(key_expr) => {
                    let prog = compile_expr_body(&key_expr);
                    stages.push(Stage::expr_stage_builtin_id(id?, prog)?);
                    exprs.push(Some(Arc::new(key_expr)));
                }
            }
            Some((source, stages, exprs, sink))
        }

        LogicalPlan::Reverse { input } => {
            let (source, mut stages, mut exprs, sink) = collect(*input)?;
            stages.push(Stage::nullary_builtin_id(id?)?);
            exprs.push(None);
            Some((source, stages, exprs, sink))
        }

        LogicalPlan::GroupBy { input, key } => {
            collect_expr_builtin_stage(*input, id?, key)
        }

        LogicalPlan::CountBy { input, key } => {
            collect_expr_builtin_stage(*input, id?, key)
        }

        LogicalPlan::IndexBy { input, key } => {
            collect_expr_builtin_stage(*input, id?, key)
        }

        // Terminal sinks strip the default Collect and install the real sink.
        LogicalPlan::First(inner) => {
            let (source, stages, exprs, _) = collect(*inner)?;
            Some((source, stages, exprs, Sink::terminal_builtin_id(id?)?))
        }
        LogicalPlan::Last(inner) => {
            let (source, stages, exprs, _) = collect(*inner)?;
            Some((source, stages, exprs, Sink::terminal_builtin_id(id?)?))
        }
        LogicalPlan::Count(inner) => {
            let (source, stages, exprs, _) = collect(*inner)?;
            Some((source, stages, exprs, Sink::count_builtin_id(id?)?))
        }
        LogicalPlan::Sum(inner)
        | LogicalPlan::Avg(inner)
        | LogicalPlan::Min(inner)
        | LogicalPlan::Max(inner) => collect_numeric_sink(*inner, id?),
        LogicalPlan::ApproxCountDistinct(inner) => {
            let (source, stages, exprs, _) = collect(*inner)?;
            Some((source, stages, exprs, Sink::approx_distinct_builtin_id(id?)?))
        }

        LogicalPlan::ScalarExpr => None,
    }
}

fn collect_expr_builtin_stage(
    input: LogicalPlan,
    id: BuiltinId,
    body_expr: Expr,
) -> Option<(Source, Vec<Stage>, Vec<Option<Arc<Expr>>>, Sink)> {
    let (source, mut stages, mut exprs, sink) = collect(input)?;
    stages.push(Stage::expr_stage_builtin_id(
        id,
        compile_expr_body(&body_expr),
    )?);
    exprs.push(Some(Arc::new(body_expr)));
    Some((source, stages, exprs, sink))
}

fn collect_numeric_sink(
    inner: LogicalPlan,
    id: BuiltinId,
) -> Option<(Source, Vec<Stage>, Vec<Option<Arc<Expr>>>, Sink)> {
    let (source, stages, exprs, _) = collect(inner)?;
    Some((source, stages, exprs, Sink::numeric_builtin_id(id, None, None)?))
}

// ---------------------------------------------------------------------------
// Expression compilation
// ---------------------------------------------------------------------------

/// Compiles `expr` with the same current-row binding rules as direct pipeline
/// method-chain lowering.
fn compile_expr_body(expr: &Expr) -> Arc<crate::vm::Program> {
    crate::exec::pipeline::compile_pipeline_expr_body(expr)
}
