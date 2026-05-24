//! Pipeline lowering: translates an `Expr` AST into a `Pipeline` IR ready for execution.
//!
//! `Pipeline::lower` is the single entry point; it walks the expression, classifies sources,
//! stages, and sinks, and returns `None` for shapes that cannot be a linear pull chain,
//! signalling fallback to the VM opcode path.
//!
//! This module also contains the registry-driven stage factory helpers (`lower_builtin_from_registry`
//! and friends) that were previously in `stage_factory.rs`.

use std::sync::Arc;

use crate::builtins::registry::{
    by_name, expr_stage, pipeline_accepts_arity, pipeline_chain_operator, pipeline_lowering,
    sink_accumulator as builtin_sink_accumulator, terminal_expr_target, BuiltinId,
};
use crate::builtins::{BuiltinExprStage, BuiltinPipelineLowering, BuiltinSinkAccumulator};
use crate::data::value::Val;
use crate::parse::ast::Expr;
use crate::plan::analysis;

use super::{
    expr_label, plan_with_exprs, sink_name, source_name, trace_enabled, Pipeline, PipelineBody,
    Plan, Sink, SortSpec, Source, Stage,
};

impl Pipeline {
    /// Lowers `expr` into a `Pipeline` IR, returning `None` when the expression shape requires the VM opcode path.
    pub fn lower(expr: &Expr) -> Option<Pipeline> {
        let p = Self::lower_with_reason(expr);
        if trace_enabled() {
            match &p {
                Ok(pipe) => eprintln!(
                    "[pipeline] activated: stages={} sink={} src={}",
                    pipe.stages.len(),
                    sink_name(&pipe.sink),
                    source_name(&pipe.source),
                ),
                Err(reason) => {
                    eprintln!("[pipeline] fallback: ({}) at {}", reason, expr_label(expr),)
                }
            }
        }
        p.ok()
    }

    // Converts `None` from `lower_inner` into `Err(&str)` so the trace path can report the reason.
    fn lower_with_reason(expr: &Expr) -> std::result::Result<Pipeline, &'static str> {
        Self::lower_inner(expr).ok_or("shape not yet supported")
    }

    // Requires `expr` rooted at `$`; extracts the leading field chain and delegates the rest to `lower_from_source`.
    fn lower_inner(expr: &Expr) -> Option<Pipeline> {
        use crate::parse::ast::Step;
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

        Self::lower_from_source(Source::FieldChain { keys }, &steps[field_end..])
    }

    /// Lowers `trailing` steps into a `PipelineBody` and attaches `source`, returning `None` when any step is unclassifiable.
    pub(crate) fn lower_from_source(
        source: Source,
        trailing: &[crate::parse::ast::Step],
    ) -> Option<Pipeline> {
        Some(Self::lower_body_from_steps(trailing)?.with_source(source))
    }

    /// Decodes `trailing` steps into stages and a sink, runs rewrite passes, and classifies body kernels.
    pub(crate) fn lower_body_from_steps(
        trailing: &[crate::parse::ast::Step],
    ) -> Option<PipelineBody> {
        let (stages, stage_exprs, sink) = decode_method_chain(trailing)?;
        Some(PipelineBody::planned(stages, stage_exprs, sink))
    }

    /// Returns `true` when `step` is a method call that can open a receiver-based pipeline without a field-chain prefix.
    pub(crate) fn is_receiver_pipeline_start(step: &crate::parse::ast::Step) -> bool {
        use crate::parse::ast::Step;

        let Step::Method(name, args) = step else {
            return false;
        };
        is_receiver_pipeline_start_method(name.as_str(), args.len())
    }
}

// Returns `true` when `name`/`arity` is a builtin that can open a receiver-based pipeline.
fn is_receiver_pipeline_start_method(name: &str, arity: usize) -> bool {
    by_name(name).is_some_and(|id| pipeline_accepts_arity(id, arity, true))
}

/// Decodes a `map(expr)` argument as a nested pipeline `Plan`, enabling the `CompiledMap` stage optimisation.
pub(super) fn try_decode_map_body(arg: &crate::parse::ast::Arg) -> Option<Plan> {
    use crate::parse::ast::{Arg, Step};
    let expr = match arg {
        Arg::Pos(e) => e,
        _ => return None,
    };
    let capture_param = match expr {
        Expr::Lambda { params, .. } if params.len() == 1 => Some(params[0].as_str()),
        _ => None,
    };
    let (base, steps) = match expr {
        Expr::Chain(b, s) => (b.as_ref(), s.as_slice()),
        _ => return None,
    };

    let mut leading_fields: Vec<Arc<str>> = Vec::new();
    match base {
        Expr::Current => {}
        Expr::Ident(name) => leading_fields.push(Arc::from(name.as_str())),
        _ => return None,
    }

    let mut field_end = 0;
    for s in steps {
        match s {
            Step::Field(_) => field_end += 1,
            _ => break,
        }
    }
    let trailing = &steps[field_end..];
    if trailing.is_empty() {
        return None;
    }
    if analysis::starts_with_direct_view_projection(trailing) {
        return None;
    }
    if !trailing_has_collection_operator(trailing) {
        return None;
    }

    let source = if field_end > 0 || !leading_fields.is_empty() {
        leading_fields.extend(steps[..field_end].iter().map(|s| match s {
            Step::Field(k) => Arc::<str>::from(k.as_str()),
            _ => unreachable!(),
        }));
        Source::FieldChain {
            keys: leading_fields.into(),
        }
    } else {
        Source::Receiver(Val::Null)
    };
    let mut stages: Vec<Stage> = Vec::new();
    let (mut more_stages, more_exprs, sink) = decode_method_chain(trailing)?;
    stages.append(&mut more_stages);
    if capture_param.is_some_and(|name| nested_plan_loads_identifier(&stages, &sink, name)) {
        return None;
    }

    let kernels = Stage::body_kernels(&stages);
    let mut plan = plan_with_exprs(stages, more_exprs, &kernels, sink);
    plan.source = source;
    Some(plan)
}

fn nested_plan_loads_identifier(stages: &[Stage], sink: &Sink, name: &str) -> bool {
    stages
        .iter()
        .filter_map(Stage::body_program)
        .any(|program| program_loads_identifier(program, name))
        || !sink.can_run_with_receiver_only(|program| !program_loads_identifier(program, name))
}

fn program_loads_identifier(program: &crate::vm::Program, name: &str) -> bool {
    crate::vm::program_loads_ident_matching(program, |ident| ident == name)
}

fn trailing_has_collection_operator(trailing: &[crate::parse::ast::Step]) -> bool {
    use crate::parse::ast::Step;
    trailing.iter().any(|step| {
        let Step::Method(name, _) = step else {
            return false;
        };
        by_name(name.as_str()).is_some_and(pipeline_chain_operator)
    })
}

// Classifies each trailing method step as a stage or sink; `None` on any unrecognised step.
fn decode_method_chain(
    trailing: &[crate::parse::ast::Step],
) -> Option<(Vec<Stage>, Vec<Option<Arc<Expr>>>, Sink)> {
    use crate::parse::ast::Step;
    let mut stages: Vec<Stage> = Vec::new();
    let mut stage_exprs: Vec<Option<Arc<Expr>>> = Vec::new();
    let mut sink: Sink = Sink::Collect;
    for (i, s) in trailing.iter().enumerate() {
        let is_last = i == trailing.len() - 1;
        match s {
            Step::Method(name, args) => {
                if let Some(call) =
                    crate::builtins::BuiltinCall::from_pipeline_literal_args(name.as_str(), args)
                {
                    stages.push(Stage::builtin_call(call)?);
                    stage_exprs.push(None);
                    continue;
                }
                let id = by_name(name.as_str())?;
                lower_builtin_from_registry(
                    id,
                    args,
                    is_last,
                    &mut stages,
                    &mut stage_exprs,
                    &mut sink,
                )?;
                if !is_last && stages.last().is_some_and(stage_is_value_reducer_boundary) {
                    return None;
                }
            }
            Step::Slice(start, end, step) => {
                // Pipeline lowering only handles step-1 slices today. Step
                // != 1 falls back to interpreted path (returns `None` to
                // signal "lower me elsewhere").
                if step.unwrap_or(1) != 1 {
                    return None;
                }
                push_path_slice_stages(*start, *end, &mut stages, &mut stage_exprs)?;
            }
            Step::Wildcard => {
                // Wildcard `[*]` is a no-op pass-through in pipeline form;
                // upstream already produced the array element stream.
            }
            _ => return None,
        }
    }
    Some((stages, stage_exprs, sink))
}

fn stage_is_value_reducer_boundary(stage: &Stage) -> bool {
    stage.shape().is_reducing()
}

fn push_path_slice_stages(
    start: Option<i64>,
    end: Option<i64>,
    stages: &mut Vec<Stage>,
    stage_exprs: &mut Vec<Option<Arc<Expr>>>,
) -> Option<()> {
    let start = match start {
        Some(n) if n < 0 => return None,
        Some(n) => n as usize,
        None => 0,
    };
    let end = match end {
        Some(n) if n < 0 => return None,
        Some(n) => Some(n as usize),
        None => None,
    };

    if start > 0 {
        stages.push(Stage::usize_builtin_id(BuiltinId::SKIP, start)?);
        stage_exprs.push(None);
    }

    if let Some(end) = end {
        stages.push(Stage::usize_builtin_id(
            BuiltinId::TAKE,
            end.saturating_sub(start),
        )?);
        stage_exprs.push(None);
    }

    Some(())
}

/// Compiles a positional argument into a VM `Program`, rewriting bare `Ident` nodes into `@.<ident>` field accesses.
pub(super) fn compile_subexpr(arg: &crate::parse::ast::Arg) -> Option<Arc<crate::vm::Program>> {
    use crate::parse::ast::Arg;
    let inner = match arg {
        Arg::Pos(e) => e,
        _ => return None,
    };
    // Single-param lambda: route through `compile_lambda_arg` so the body
    // is substituted and, when nested-lambda refs to the param remain,
    // wrapped in `BindLamCurrent` for correct outer-row binding.
    if matches!(inner, Expr::Lambda { params, .. } if params.len() == 1) {
        return Some(crate::compile::lambda_lower::compile_lambda_arg(inner, ""));
    }
    let rooted = crate::compile::lambda_lower::normalize_pipeline_arg_expr(inner);
    Some(Arc::new(crate::compile::compiler::Compiler::compile(
        &rooted, "",
    )))
}

/// Compiles a pipeline stage body expression using the same current-row
/// binding rules as method-chain lowering.
pub(crate) fn compile_pipeline_expr_body(expr: &Expr) -> Arc<crate::vm::Program> {
    compile_subexpr(&crate::parse::ast::Arg::Pos(expr.clone()))
        .expect("positional pipeline expression must compile")
}

fn compile_raw_arg_expr(arg: &crate::parse::ast::Arg) -> Option<Arc<crate::vm::Program>> {
    use crate::parse::ast::Arg;
    let expr = match arg {
        Arg::Pos(e) => e,
        Arg::Named(_, _) => return None,
    };
    Some(crate::compile::lambda_lower::compile_lambda_arg(expr, ""))
}

/// Compiles a sort-key argument into a `SortSpec`, interpreting `UnaryNeg`-wrapping as descending order.
///
/// Returns `None` for multi-param `Expr::Lambda` arguments — those are
/// 2-arg comparator lambdas (`.sort((a, b) => …)`) which the pipeline IR
/// cannot represent as a single-key `SortSpec`. Bailing out forces the
/// router to fall back to the VM path, where `exec_lambda_method` handles
/// the comparator via `sort_comparator_apply`.
pub(crate) fn compile_sort_spec(
    arg: &crate::parse::ast::Arg,
) -> Option<(SortSpec, Option<Arc<Expr>>)> {
    use crate::parse::ast::{Arg, Expr};
    let expr = match arg {
        Arg::Pos(e) => e,
        _ => return None,
    };
    if matches!(expr, Expr::Lambda { params, .. } if params.len() >= 2) {
        return None;
    }
    let (key_expr, descending) = match expr {
        Expr::UnaryNeg(inner) => (inner.as_ref().clone(), true),
        other => (other.clone(), false),
    };
    let key_arg = Arg::Pos(key_expr.clone());
    Some((
        SortSpec::keyed(compile_subexpr(&key_arg)?, descending),
        Some(Arc::new(expr.clone())),
    ))
}

/// Returns the symbolic row-local expression for a positional pipeline argument.
///
/// This mirrors `compile_subexpr`'s current-row binding so the demand and
/// symbolic planners see the same expression shape that the VM program executes.
pub(super) fn arg_expr(arg: &crate::parse::ast::Arg) -> Option<Arc<Expr>> {
    use crate::parse::ast::Arg;
    match arg {
        Arg::Named(_, _) => None,
        Arg::Pos(e) => Some(Arc::new(
            crate::compile::lambda_lower::normalize_pipeline_arg_expr(e),
        )),
    }
}

fn raw_arg_expr(arg: &crate::parse::ast::Arg) -> Option<Arc<Expr>> {
    use crate::parse::ast::Arg;
    match arg {
        Arg::Named(_, _) => None,
        Arg::Pos(e) => Some(Arc::new(e.clone())),
    }
}

// ---------------------------------------------------------------------------
// Registry-driven stage factory (formerly stage_factory.rs)
// ---------------------------------------------------------------------------

use super::{ArgExtremeSinkSpec, MembershipSinkSpec, MembershipSinkTarget, PredicateSinkSpec};

/// Lowers a resolved builtin id to a concrete `Stage` or `Sink`, returning `None` when the builtin cannot be lowered at this position.
pub(super) fn lower_builtin_from_registry(
    id: BuiltinId,
    args: &[crate::parse::ast::Arg],
    is_last: bool,
    stages: &mut Vec<Stage>,
    stage_exprs: &mut Vec<Option<Arc<Expr>>>,
    sink: &mut Sink,
) -> Option<()> {
    if !pipeline_accepts_arity(id, args.len(), is_last) {
        return None;
    }
    let Some(lowering) = pipeline_lowering(id) else {
        if is_last {
            *sink = terminal_sink_for_id(id, args)?;
            return Some(());
        }
        return None;
    };
    match lowering {
        BuiltinPipelineLowering::ExprArg => {
            if args.len() != 1 {
                return None;
            }
            push_expr_stage(id, &args[0], stages, stage_exprs)
        }
        BuiltinPipelineLowering::TerminalExprArg { .. } => {
            if args.len() != 1 || !is_last {
                return None;
            }
            push_expr_stage(id, &args[0], stages, stage_exprs)?;
            set_terminal_sink(terminal_expr_target(id)?, sink)?;
            Some(())
        }
        BuiltinPipelineLowering::Nullary => {
            if !args.is_empty() {
                return None;
            }
            stages.push(Stage::nullary_builtin_id(id)?);
            stage_exprs.push(None);
            Some(())
        }
        BuiltinPipelineLowering::UsizeArg { min } => {
            if args.len() != 1 {
                return None;
            }
            let n = usize_arg_at_least(&args[0], min)?;
            stages.push(Stage::usize_builtin_id(id, n)?);
            stage_exprs.push(None);
            Some(())
        }
        BuiltinPipelineLowering::StringArg => {
            if args.len() != 1 {
                return None;
            }
            stages.push(Stage::string_builtin_id(id, string_arg(&args[0])?)?);
            stage_exprs.push(None);
            Some(())
        }
        BuiltinPipelineLowering::StringPairArg => {
            if args.len() != 2 {
                return None;
            }
            stages.push(Stage::string_pair_builtin_id(
                id,
                string_arg(&args[0])?,
                string_arg(&args[1])?,
            )?);
            stage_exprs.push(None);
            Some(())
        }
        BuiltinPipelineLowering::Sort => match args {
            [] => {
                stages.push(Stage::sort_builtin_id(id, super::SortSpec::identity())?);
                stage_exprs.push(None);
                Some(())
            }
            [arg] => {
                let (spec, expr) = compile_sort_spec(arg)?;
                stages.push(Stage::sort_builtin_id(id, spec)?);
                stage_exprs.push(expr);
                Some(())
            }
            _ => None,
        },
        BuiltinPipelineLowering::IntRangeArg => match args {
            [arg] => {
                stages.push(Stage::int_range_builtin_id(id, int_arg(arg)?, None)?);
                stage_exprs.push(None);
                Some(())
            }
            [start, end] => {
                stages.push(Stage::int_range_builtin_id(
                    id,
                    int_arg(start)?,
                    Some(int_arg(end)?),
                )?);
                stage_exprs.push(None);
                Some(())
            }
            _ => None,
        },
        BuiltinPipelineLowering::TerminalSink if is_last => {
            *sink = terminal_sink_for_id(id, args)?;
            Some(())
        }
        BuiltinPipelineLowering::TerminalSink => None,
        BuiltinPipelineLowering::TerminalUsizeSink { min } if is_last => {
            if args.len() != 1 {
                return None;
            }
            *sink = Sink::nth_builtin_id(id, usize_arg_at_least(&args[0], min)?)?;
            Some(())
        }
        BuiltinPipelineLowering::TerminalUsizeSink { .. } => None,
    }
}

// Compiles `arg` into a sub-expression program and appends the corresponding `Stage` variant; `None` on compile failure.
fn push_expr_stage(
    id: BuiltinId,
    arg: &crate::parse::ast::Arg,
    stages: &mut Vec<Stage>,
    stage_exprs: &mut Vec<Option<Arc<Expr>>>,
) -> Option<()> {
    match expr_stage(id)? {
        BuiltinExprStage::Map => match try_decode_map_body(arg) {
            Some(plan) => {
                stages.push(Stage::CompiledMap(Arc::new(plan)));
                stage_exprs.push(arg_expr(arg));
            }
            None => push_expr_builtin(id, arg, stages, stage_exprs)?,
        },
        BuiltinExprStage::Filter
        | BuiltinExprStage::FlatMap
        | BuiltinExprStage::UniqueBy
        | BuiltinExprStage::ExprBuiltin => push_expr_builtin(id, arg, stages, stage_exprs)?,
    }
    Some(())
}

fn push_expr_builtin(
    id: BuiltinId,
    arg: &crate::parse::ast::Arg,
    stages: &mut Vec<Stage>,
    stage_exprs: &mut Vec<Option<Arc<Expr>>>,
) -> Option<()> {
    stages.push(Stage::expr_stage_builtin_id(id, compile_subexpr(arg)?)?);
    stage_exprs.push(arg_expr(arg));
    Some(())
}

// Writes the no-arg terminal `Sink` for `id` into `*sink`.
fn set_terminal_sink(id: BuiltinId, sink: &mut Sink) -> Option<()> {
    *sink = terminal_sink_for_id(id, &[])?;
    Some(())
}

// Extracts a `usize` integer literal from `arg` and enforces `value >= min`.
fn usize_arg_at_least(arg: &crate::parse::ast::Arg, min: usize) -> Option<usize> {
    match arg {
        crate::parse::ast::Arg::Pos(Expr::Int(n)) if *n >= min as i64 => Some(*n as usize),
        _ => None,
    }
}

// Extracts a signed integer literal from `arg`.
fn int_arg(arg: &crate::parse::ast::Arg) -> Option<i64> {
    match arg {
        crate::parse::ast::Arg::Pos(Expr::Int(n)) => Some(*n),
        _ => None,
    }
}

// Extracts a string literal from `arg` and interns it as `Arc<str>`.
fn string_arg(arg: &crate::parse::ast::Arg) -> Option<Arc<str>> {
    match arg {
        crate::parse::ast::Arg::Pos(Expr::Str(s)) => Some(Arc::<str>::from(s.as_str())),
        _ => None,
    }
}

// Extracts a literal value from `arg` for value-membership terminal sinks.
fn literal_arg_value(arg: &crate::parse::ast::Arg) -> Option<Val> {
    match arg {
        crate::parse::ast::Arg::Pos(Expr::Null) => Some(Val::Null),
        crate::parse::ast::Arg::Pos(Expr::Bool(b)) => Some(Val::Bool(*b)),
        crate::parse::ast::Arg::Pos(Expr::Int(n)) => Some(Val::Int(*n)),
        crate::parse::ast::Arg::Pos(Expr::Float(n)) => Some(Val::Float(*n)),
        crate::parse::ast::Arg::Pos(Expr::Str(s)) => Some(Val::Str(Arc::from(s.as_str()))),
        _ => None,
    }
}

// Constructs the terminal `Sink` for `id`, handling count predicates, numeric reducers, and positional selects.
fn terminal_sink_for_id(id: BuiltinId, args: &[crate::parse::ast::Arg]) -> Option<Sink> {
    if let Some(sink) = predicate_sink_for_id(id, args) {
        return Some(sink);
    }
    if let Some(sink) = membership_sink_for_id(id, args) {
        return Some(sink);
    }
    if let Some(sink) = arg_extreme_sink_for_id(id, args) {
        return Some(sink);
    }
    match builtin_sink_accumulator(id)? {
        BuiltinSinkAccumulator::ApproxDistinct if args.is_empty() => {
            Sink::approx_distinct_builtin_id(id)
        }
        BuiltinSinkAccumulator::Count => match args {
            [] => Sink::count_builtin_id(id),
            [arg] => Sink::count_predicate_builtin_id(id, compile_subexpr(arg)?, raw_arg_expr(arg)),
            _ => None,
        },
        BuiltinSinkAccumulator::Numeric => {
            let projection = match args {
                [] => None,
                [arg] => Some(compile_subexpr(arg)?),
                _ => return None,
            };
            let projection_expr = match args {
                [] => None,
                [arg] => raw_arg_expr(arg),
                _ => return None,
            };
            Sink::numeric_builtin_id(id, projection, projection_expr)
        }
        BuiltinSinkAccumulator::SelectOne(_) => match args {
            [] => Sink::terminal_builtin_id(id),
            [arg] => Sink::select_many_builtin_id(id, usize_arg_at_least(arg, 1)?),
            _ => None,
        },
        _ => None,
    }
}

fn predicate_sink_for_id(id: BuiltinId, args: &[crate::parse::ast::Arg]) -> Option<Sink> {
    let [arg] = args else {
        return None;
    };
    Some(Sink::Predicate(PredicateSinkSpec::from_id(
        id,
        compile_subexpr(arg)?,
        raw_arg_expr(arg),
    )?))
}

fn membership_sink_for_id(id: BuiltinId, args: &[crate::parse::ast::Arg]) -> Option<Sink> {
    let [arg] = args else {
        return None;
    };
    Some(Sink::Membership(MembershipSinkSpec::from_id(
        id,
        literal_arg_value(arg)
            .map(MembershipSinkTarget::Literal)
            .or_else(|| Some(MembershipSinkTarget::Program(compile_raw_arg_expr(arg)?)))?,
    )?))
}

fn arg_extreme_sink_for_id(id: BuiltinId, args: &[crate::parse::ast::Arg]) -> Option<Sink> {
    let [arg] = args else {
        return None;
    };
    Some(Sink::ArgExtreme(ArgExtremeSinkSpec::from_id(
        id,
        compile_subexpr(arg)?,
        raw_arg_expr(arg),
    )?))
}

#[cfg(test)]
mod tests {
    use super::arg_expr;
    use crate::parse::ast::{Arg, Expr};
    use crate::parse::parser::parse;

    fn normalized_arg(src: &str) -> Expr {
        let expr = parse(src).expect("parse expression");
        arg_expr(&Arg::Pos(expr))
            .expect("positional arg")
            .as_ref()
            .clone()
    }

    fn assert_current_field(expr: Expr, field: &str) {
        let Expr::Chain(base, steps) = expr else {
            panic!("expected current field chain");
        };
        assert!(matches!(base.as_ref(), Expr::Current));
        assert!(matches!(
            steps.as_slice(),
            [crate::parse::ast::Step::Field(name)] if name.as_str() == field
        ));
    }

    #[test]
    fn pipeline_arg_normalization_is_shared_for_symbolic_exprs() {
        assert_current_field(normalized_arg("isbn"), "isbn");
        assert_current_field(normalized_arg("@.isbn"), "isbn");
        assert_current_field(normalized_arg("x => x.isbn"), "isbn");
    }
}
