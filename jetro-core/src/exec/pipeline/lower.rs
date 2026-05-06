//! Pipeline lowering: translates an `Expr` AST into a `Pipeline` IR ready for execution.
//!
//! `Pipeline::lower` is the single entry point; it walks the expression, classifies sources,
//! stages, and sinks, and returns `None` for shapes that cannot be a linear pull chain,
//! signalling fallback to the VM opcode path.
//!
//! This module also contains the registry-driven stage factory helpers (`lower_method_from_registry`
//! and friends) that were previously in `stage_factory.rs`.

use std::sync::Arc;

use crate::parse::ast::Expr;
use crate::builtins::registry::{pipeline_accepts_arity, pipeline_lowering, BuiltinId};
use crate::builtins::{
    BuiltinMethod, BuiltinPipelineLowering, BuiltinSinkAccumulator, BuiltinViewStage,
};
use crate::{data::context::EvalError, data::value::Val};

use super::{
    expr_label, plan_with_exprs, plan_with_kernels, sink_name, source_name, trace_enabled,
    BodyKernel, Pipeline, PipelineBody, Plan, Sink, SortSpec, Source, Stage,
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

        let trailing = &steps[field_end..];
        Self::lower_from_source(Source::FieldChain { keys }, trailing)
    }

    /// Lowers `trailing` steps into a `PipelineBody` and attaches `source`, returning `None` when any step is unclassifiable.
    pub(crate) fn lower_from_source(
        source: Source,
        trailing: &[crate::parse::ast::Step],
    ) -> Option<Pipeline> {
        Some(Self::lower_body_from_steps(trailing)?.with_source(source))
    }

    /// Decodes `trailing` steps into stages and a sink, runs rewrite passes, and classifies body kernels.
    pub(crate) fn lower_body_from_steps(trailing: &[crate::parse::ast::Step]) -> Option<PipelineBody> {
        let (stages, stage_exprs, sink) = decode_method_chain(trailing)?;
        let mut p = PipelineBody {
            stages,
            stage_exprs,
            sink,
            stage_kernels: Vec::new(),
            sink_kernels: Vec::new(),
        };
        rewrite(&mut p);
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
        let kernels = classify_kernels(&p.stages);
        let plan_result = plan_with_exprs(
            p.stages.clone(),
            p.stage_exprs.clone(),
            &kernels,
            p.sink.clone(),
        );
        p.stages = plan_result.stages;
        p.stage_exprs = plan_result.stage_exprs;
        p.sink = plan_result.sink;
        p.stage_kernels = classify_kernels(&p.stages);
        p.sink_kernels = p
            .sink
            .reducer_spec()
            .map(|spec| {
                spec.sink_programs()
                    .map(|p| BodyKernel::classify(p))
                    .collect()
            })
            .unwrap_or_default();
        Some(p)
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
    let method = BuiltinMethod::from_name(name);
    if method == BuiltinMethod::Unknown {
        return false;
    }

    pipeline_accepts_arity(BuiltinId::from_method(method), arity, true)
}

/// Decodes a `map(expr)` argument as a nested pipeline `Plan`, enabling the `CompiledMap` stage optimisation.
pub(super) fn try_decode_map_body(arg: &crate::parse::ast::Arg) -> Option<Plan> {
    use crate::parse::ast::{Arg, Step};
    let expr = match arg {
        Arg::Pos(e) => e,
        _ => return None,
    };
    let (base, steps) = match expr {
        Expr::Chain(b, s) => (b.as_ref(), s.as_slice()),
        _ => return None,
    };
    if !matches!(base, Expr::Current) {
        return None;
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

    let mut stages: Vec<Stage> = Vec::new();
    if field_end > 0 {
        let keys: Arc<[Arc<str>]> = steps[..field_end]
            .iter()
            .map(|s| match s {
                Step::Field(k) => Arc::<str>::from(k.as_str()),
                _ => unreachable!(),
            })
            .collect::<Vec<_>>()
            .into();
        let n_keys = keys.len();
        let fcd = Arc::new(crate::vm::FieldChainData {
            keys,
            ics: (0..n_keys)
                .map(|_| std::sync::atomic::AtomicU64::new(0))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        });
        let ops = vec![
            crate::vm::Opcode::PushCurrent,
            crate::vm::Opcode::FieldChain(fcd),
        ];
        let prog = Arc::new(crate::vm::Program::new(ops, "<compiled-map-prefix>"));
        stages.push(Stage::Map(prog, BuiltinViewStage::Map));
    }
    let (mut more_stages, _more_exprs, sink) = decode_method_chain(trailing)?;
    stages.append(&mut more_stages);

    let kernels: Vec<BodyKernel> = stages
        .iter()
        .map(|s| {
            s.body_program()
                .map(BodyKernel::classify)
                .unwrap_or(BodyKernel::Generic)
        })
        .collect();
    Some(plan_with_kernels(stages, &kernels, sink))
}

/// Wraps `seed` in a single-element receiver pipeline backed by `plan` and runs it.
pub(super) fn run_compiled_map(plan: &Plan, seed: Val) -> Result<Val, EvalError> {
    let synth = Pipeline {
        exec_path: super::select_exec_path(&plan.stages, &plan.sink),
        source: Source::Receiver(Val::arr(vec![seed])),
        stages: plan.stages.clone(),
        stage_exprs: Vec::new(),
        sink: plan.sink.clone(),
        stage_kernels: Vec::new(),
        sink_kernels: Vec::new(),
    };
    synth.run(&Val::Null)
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
                    stages.push(Stage::Builtin(call));
                    stage_exprs.push(None);
                    continue;
                }
                let method = BuiltinMethod::from_name(name.as_str());
                lower_method_from_registry(
                    method,
                    args,
                    is_last,
                    &mut stages,
                    &mut stage_exprs,
                    &mut sink,
                )?;
            }
            _ => return None,
        }
    }
    Some((stages, stage_exprs, sink))
}

// Repeatedly applies `rewrite_step` until fixpoint (at most 16 iterations).
fn rewrite(p: &mut PipelineBody) {
    let mut fuel = 16usize;
    while fuel > 0 {
        fuel -= 1;
        if rewrite_step(p) {
            continue;
        }
        break;
    }
}

// Applies one rewrite pass: const-false short-circuit, adjacent map/filter fusion, map/take commutation.
fn rewrite_step(p: &mut PipelineBody) -> bool {
    use crate::vm::Opcode;

    let mut const_false_at: Option<usize> = None;
    for (i, s) in p.stages.iter().enumerate() {
        if let Stage::Filter(prog, _) = s {
            if let Some(false) = prog_const_bool(prog) {
                const_false_at = Some(i);
                break;
            }
        }
    }
    if const_false_at.is_some() {
        p.stages.clear();
        p.stage_exprs.clear();
        return true;
    }

    for i in 0..p.stages.len().saturating_sub(1) {
        match (&p.stages[i], &p.stages[i + 1]) {
            (Stage::Map(a_prog, _), Stage::Map(b_prog, _)) => {
                let ka = BodyKernel::classify(a_prog);
                let kb = BodyKernel::classify(b_prog);
                let chain: Option<Vec<Arc<str>>> = match (&ka, &kb) {
                    (BodyKernel::FieldRead(a), BodyKernel::FieldRead(b)) => {
                        Some(vec![a.clone(), b.clone()])
                    }
                    (BodyKernel::FieldRead(a), BodyKernel::FieldChain(bs)) => {
                        let mut v = vec![a.clone()];
                        v.extend(bs.iter().cloned());
                        Some(v)
                    }
                    (BodyKernel::FieldChain(as_), BodyKernel::FieldRead(b)) => {
                        let mut v: Vec<Arc<str>> = as_.iter().cloned().collect();
                        v.push(b.clone());
                        Some(v)
                    }
                    (BodyKernel::FieldChain(as_), BodyKernel::FieldChain(bs)) => {
                        let mut v: Vec<Arc<str>> = as_.iter().cloned().collect();
                        v.extend(bs.iter().cloned());
                        Some(v)
                    }
                    _ => None,
                };
                if let Some(keys) = chain {
                    let fcd = Arc::new(crate::vm::FieldChainData {
                        keys: keys.into(),
                        ics: (0..0)
                            .map(|_| std::sync::atomic::AtomicU64::new(0))
                            .collect::<Vec<_>>()
                            .into_boxed_slice(),
                    });
                    let new_ops = vec![Opcode::PushCurrent, Opcode::FieldChain(fcd)];
                    let merged = Arc::new(crate::vm::Program::new(new_ops, "<map-fused>"));
                    p.stages[i] = Stage::Map(merged, BuiltinViewStage::Map);
                    p.stage_exprs[i] = None;
                    p.stages.remove(i + 1);
                    p.stage_exprs.remove(i + 1);
                    return true;
                }
            }
            (Stage::Filter(p_prog, _), Stage::Filter(q_prog, _)) => {
                let mut ops: Vec<Opcode> = p_prog.ops.as_ref().to_vec();
                ops.push(Opcode::AndOp(Arc::clone(q_prog)));
                let merged = Arc::new(crate::vm::Program {
                    ops: ops.into(),
                    source: p_prog.source.clone(),
                    id: 0,
                    is_structural: false,
                    ics: p_prog.ics.clone(),
                });
                p.stages[i] = Stage::Filter(merged, BuiltinViewStage::Filter);
                p.stage_exprs[i] = None;
                p.stages.remove(i + 1);
                p.stage_exprs.remove(i + 1);
                return true;
            }
            _ => {}
        }
    }

    for i in 0..p.stages.len().saturating_sub(1) {
        if matches!(&p.stages[i], Stage::Map(_, _)) && is_take_stage(&p.stages[i + 1]) {
            p.stages.swap(i, i + 1);
            p.stage_exprs.swap(i, i + 1);
            return true;
        }
    }

    false
}

fn is_take_stage(stage: &Stage) -> bool {
    matches!(
        stage,
        Stage::UsizeBuiltin {
            method: BuiltinMethod::Take,
            ..
        }
    )
}

// Returns `Some(b)` when `prog` is a single `PushBool(b)` opcode; detects constant filter stages.
fn prog_const_bool(prog: &crate::vm::Program) -> Option<bool> {
    use crate::vm::Opcode;
    let ops = prog.ops.as_ref();
    if ops.len() != 1 {
        return None;
    }
    match &ops[0] {
        Opcode::PushBool(b) => Some(*b),
        _ => None,
    }
}

/// Compiles a positional argument into a VM `Program`, rewriting bare `Ident` nodes into `@.<ident>` field accesses.
pub(super) fn compile_subexpr(arg: &crate::parse::ast::Arg) -> Option<Arc<crate::vm::Program>> {
    use crate::parse::ast::{Arg, Expr, Step};
    let inner = match arg {
        Arg::Pos(e) => e,
        _ => return None,
    };
    let rooted: Expr = match inner {
        Expr::Ident(name) => Expr::Chain(Box::new(Expr::Current), vec![Step::Field(name.clone())]),
        Expr::Chain(base, _) if matches!(base.as_ref(), Expr::Current) => inner.clone(),
        other => other.clone(),
    };
    Some(Arc::new(crate::compile::compiler::Compiler::compile(&rooted, "")))
}

/// Compiles a sort-key argument into a `SortSpec`, interpreting `UnaryNeg`-wrapping as descending order.
pub(super) fn compile_sort_spec(arg: &crate::parse::ast::Arg) -> Option<(SortSpec, Option<Arc<Expr>>)> {
    use crate::parse::ast::{Arg, Expr};
    let expr = match arg {
        Arg::Pos(e) => e,
        _ => return None,
    };
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

/// Wraps the inner `Expr` of a positional argument as `Arc<Expr>`, returning `None` for named arguments.
pub(super) fn arg_expr(arg: &crate::parse::ast::Arg) -> Option<Arc<Expr>> {
    match arg {
        crate::parse::ast::Arg::Pos(e) => Some(Arc::new(e.clone())),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Registry-driven stage factory (formerly stage_factory.rs)
// ---------------------------------------------------------------------------

use super::{NumOp, ReducerOp, ReducerSpec};

/// Lowers a `BuiltinMethod` call to a concrete `Stage` or `Sink`, returning `None` when the method cannot be lowered at this position.
pub(super) fn lower_method_from_registry(
    method: BuiltinMethod,
    args: &[crate::parse::ast::Arg],
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
        BuiltinPipelineLowering::ExprArg => {
            if args.len() != 1 {
                return None;
            }
            push_expr_stage(method, &args[0], stages, stage_exprs)
        }
        BuiltinPipelineLowering::TerminalExprArg { terminal } => {
            if args.len() != 1 {
                return None;
            }
            push_expr_stage(method, &args[0], stages, stage_exprs)?;
            if is_last {
                set_terminal_sink(terminal, sink)?;
            }
            Some(())
        }
        BuiltinPipelineLowering::Nullary => {
            if !args.is_empty() {
                return None;
            }
            match method {
                BuiltinMethod::Reverse => {
                    let cancel = method
                        .spec()
                        .cancellation
                        .expect("reverse builtin must define cancellation metadata");
                    stages.push(Stage::Reverse(cancel));
                }
                BuiltinMethod::Unique => stages.push(Stage::UniqueBy(None)),
                _ => return None,
            }
            stage_exprs.push(None);
            Some(())
        }
        BuiltinPipelineLowering::UsizeArg { min } => {
            if args.len() != 1 {
                return None;
            }
            let n = usize_arg_at_least(&args[0], min)?;
            stages.push(Stage::UsizeBuiltin { method, value: n });
            stage_exprs.push(None);
            Some(())
        }
        BuiltinPipelineLowering::StringArg => {
            if args.len() != 1 {
                return None;
            }
            stages.push(Stage::StringBuiltin {
                method,
                value: string_arg(&args[0])?,
            });
            stage_exprs.push(None);
            Some(())
        }
        BuiltinPipelineLowering::StringPairArg => {
            if args.len() != 2 {
                return None;
            }
            stages.push(Stage::StringPairBuiltin {
                method,
                first: string_arg(&args[0])?,
                second: string_arg(&args[1])?,
            });
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
        BuiltinPipelineLowering::IntRangeArg => match args {
            [arg] => {
                stages.push(Stage::IntRangeBuiltin {
                    method,
                    start: int_arg(arg)?,
                    end: None,
                });
                stage_exprs.push(None);
                Some(())
            }
            [start, end] => {
                stages.push(Stage::IntRangeBuiltin {
                    method,
                    start: int_arg(start)?,
                    end: Some(int_arg(end)?),
                });
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

// Compiles `arg` into a sub-expression program and appends the corresponding `Stage` variant; `None` on compile failure.
fn push_expr_stage(
    method: BuiltinMethod,
    arg: &crate::parse::ast::Arg,
    stages: &mut Vec<Stage>,
    stage_exprs: &mut Vec<Option<Arc<Expr>>>,
) -> Option<()> {
    match method {
        BuiltinMethod::Filter | BuiltinMethod::Find | BuiltinMethod::FindAll => {
            stages.push(Stage::Filter(
                compile_subexpr(arg)?,
                BuiltinViewStage::Filter,
            ));
            stage_exprs.push(arg_expr(arg));
        }
        BuiltinMethod::Map => match try_decode_map_body(arg) {
            Some(plan) => {
                stages.push(Stage::CompiledMap(Arc::new(plan)));
                stage_exprs.push(arg_expr(arg));
            }
            None => {
                stages.push(Stage::Map(compile_subexpr(arg)?, BuiltinViewStage::Map));
                stage_exprs.push(arg_expr(arg));
            }
        },
        BuiltinMethod::FlatMap => {
            stages.push(Stage::FlatMap(
                compile_subexpr(arg)?,
                BuiltinViewStage::FlatMap,
            ));
            stage_exprs.push(arg_expr(arg));
        }
        BuiltinMethod::UniqueBy => {
            stages.push(Stage::UniqueBy(Some(compile_subexpr(arg)?)));
            stage_exprs.push(arg_expr(arg));
        }
        // Methods that route through the generic ExprBuiltin stage; `method` is preserved
        // verbatim so the runtime executor dispatches on the right semantics.
        BuiltinMethod::TakeWhile
        | BuiltinMethod::DropWhile
        | BuiltinMethod::IndicesWhere
        | BuiltinMethod::FindIndex
        | BuiltinMethod::MaxBy
        | BuiltinMethod::MinBy
        | BuiltinMethod::GroupBy
        | BuiltinMethod::CountBy
        | BuiltinMethod::IndexBy
        | BuiltinMethod::TransformValues
        | BuiltinMethod::TransformKeys
        | BuiltinMethod::FilterValues
        | BuiltinMethod::FilterKeys => {
            push_expr_builtin(method, arg, stages, stage_exprs)?;
        }
        _ => return None,
    }
    Some(())
}

fn push_expr_builtin(
    method: BuiltinMethod,
    arg: &crate::parse::ast::Arg,
    stages: &mut Vec<Stage>,
    stage_exprs: &mut Vec<Option<Arc<Expr>>>,
) -> Option<()> {
    stages.push(Stage::ExprBuiltin {
        method,
        body: compile_subexpr(arg)?,
    });
    stage_exprs.push(arg_expr(arg));
    Some(())
}

// Writes the terminal `Sink` for `method` into `*sink`; returns `None` for numeric reducers needing argument-based config.
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

// Constructs the terminal `Sink` for `method`, handling count predicates, numeric reducers, and positional selects.
fn terminal_sink_for_method(method: BuiltinMethod, args: &[crate::parse::ast::Arg]) -> Option<Sink> {
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
