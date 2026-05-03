//! Lowers an `Expr` AST into a `Pipeline` IR.
//!
//! `Pipeline::lower` is the single entry point. It walks the expression,
//! classifies sources, stages, and sinks, and returns `None` for any shape
//! that cannot be represented as a linear pull chain — signalling fallback
//! to the VM opcode path.

use std::sync::Arc;

use crate::builtin_registry::{pipeline_stage, BuiltinId};
use crate::builtins::{BuiltinMethod, BuiltinPipelineStage, BuiltinViewStage};
use crate::{ast::Expr, context::EvalError, value::Val};

use super::stage_factory::lower_method_from_registry;
use super::{
    expr_label, plan_with_exprs, plan_with_kernels, sink_name, source_name, trace_enabled,
    BodyKernel, Pipeline, PipelineBody, Plan, Sink, SortSpec, Source, Stage,
};

impl Pipeline {
    /// Attempts to lower `expr` into a `Pipeline` IR, returning `None` when the expression
    /// shape is unsupported and the VM opcode path should be used instead.
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

    /// Wraps `lower_inner` and converts `None` into an `Err` with a static reason string,
    /// so the trace machinery can report why lowering was skipped.
    fn lower_with_reason(expr: &Expr) -> std::result::Result<Pipeline, &'static str> {
        Self::lower_inner(expr).ok_or("shape not yet supported")
    }

    /// Core lowering logic: requires `expr` to be `$.<field>*.<method>*` rooted at `$`, extracts
    /// the field-chain source, and delegates trailing steps to `lower_from_source`.
    fn lower_inner(expr: &Expr) -> Option<Pipeline> {
        use crate::ast::Step;
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

    /// Lowers `trailing` steps into a `PipelineBody` and attaches `source` to produce a complete
    /// `Pipeline`; returns `None` when any step cannot be classified.
    pub(crate) fn lower_from_source(
        source: Source,
        trailing: &[crate::ast::Step],
    ) -> Option<Pipeline> {
        Some(Self::lower_body_from_steps(trailing)?.with_source(source))
    }

    /// Decodes `trailing` method steps into stages and a sink, runs the rewrite / planning passes,
    /// and classifies kernels for both stages and sink sub-programs.
    pub(crate) fn lower_body_from_steps(trailing: &[crate::ast::Step]) -> Option<PipelineBody> {
        
        
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

    /// Returns `true` when `step` is a method call that may start a pipeline from a receiver
    /// value (rather than requiring a `$.field` source prefix).
    pub(crate) fn is_receiver_pipeline_start(step: &crate::ast::Step) -> bool {
        use crate::ast::Step;

        let Step::Method(name, args) = step else {
            return false;
        };
        is_receiver_pipeline_start_method(name.as_str(), args.len())
    }
}

/// Returns `true` when a method named `name` with `arity` arguments can serve as the first
/// step of a receiver-based pipeline (i.e. requires no field-chain source prefix).
fn is_receiver_pipeline_start_method(name: &str, arity: usize) -> bool {
    let method = BuiltinMethod::from_name(name);
    if method == BuiltinMethod::Unknown {
        return false;
    }

    match arity {
        0 => {
            let spec = method.spec();
            spec.sink.is_some()
                || pipeline_stage(BuiltinId::from_method(method))
                    == Some(BuiltinPipelineStage::Nullary)
        }
        1 if method == BuiltinMethod::Sort => true,
        1 => pipeline_stage(BuiltinId::from_method(method)) == Some(BuiltinPipelineStage::Unary),
        _ => false,
    }
}


/// Tries to decode a `map(expr)` argument as a nested pipeline `Plan`, enabling the
/// `CompiledMap` stage optimisation for chains like `@.field.filter(…).sum()`.
pub(super) fn try_decode_map_body(arg: &crate::ast::Arg) -> Option<Plan> {
    use crate::ast::{Arg, Step};
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


/// Wraps `seed` in a single-element receiver pipeline backed by `plan` and executes it,
/// used to evaluate a `CompiledMap` stage body against one element at a time.
pub(super) fn run_compiled_map(plan: &Plan, seed: Val) -> Result<Val, EvalError> {
    let synth = Pipeline {
        source: Source::Receiver(Val::arr(vec![seed])),
        stages: plan.stages.clone(),
        stage_exprs: Vec::new(),
        sink: plan.sink.clone(),
        stage_kernels: Vec::new(),
        sink_kernels: Vec::new(),
    };
    synth.run(&Val::Null)
}


/// Iterates `trailing` method steps, classifying each as a builtin stage, a registry-lowered
/// stage, or a sink; returns `None` when any step is a non-method field access or unrecognised.
fn decode_method_chain(
    trailing: &[crate::ast::Step],
) -> Option<(Vec<Stage>, Vec<Option<Arc<Expr>>>, Sink)> {
    use crate::ast::Step;
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


/// Applies `rewrite_step` repeatedly (up to 16 times) until no further rewrites are possible.
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


/// Applies one rewrite to `p` if possible, returning `true` when a rewrite was made so the
/// caller can loop; handles: const-false short-circuit, adjacent map fusion, adjacent filter
/// fusion, and map/take commutation.
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
        if matches!(&p.stages[i], Stage::Map(_, _))
            && matches!(&p.stages[i + 1], Stage::Take(_, _, _))
        {
            p.stages.swap(i, i + 1);
            p.stage_exprs.swap(i, i + 1);
            return true;
        }
    }

    false
}


/// Returns `Some(b)` when `prog` is a single `PushBool(b)` opcode, used to detect
/// constant-true/false filter stages.
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


/// Compiles a positional argument expression into a VM `Program`, rewriting bare `Ident` nodes
/// into `@.<ident>` field accesses so the program runs against the current element.
pub(super) fn compile_subexpr(arg: &crate::ast::Arg) -> Option<Arc<crate::vm::Program>> {
    use crate::ast::{Arg, Expr, Step};
    let inner = match arg {
        Arg::Pos(e) => e,
        _ => return None,
    };
    let rooted: Expr = match inner {
        
        Expr::Ident(name) => Expr::Chain(Box::new(Expr::Current), vec![Step::Field(name.clone())]),
        
        Expr::Chain(base, _) if matches!(base.as_ref(), Expr::Current) => inner.clone(),
        
        other => other.clone(),
    };
    Some(Arc::new(crate::vm::Compiler::compile(&rooted, "")))
}

/// Compiles a sort-key argument (which may be `UnaryNeg`-wrapped for descending order) into a
/// `SortSpec` paired with the preserved `Arc<Expr>` for the demand optimiser.
pub(super) fn compile_sort_spec(arg: &crate::ast::Arg) -> Option<(SortSpec, Option<Arc<Expr>>)> {
    use crate::ast::{Arg, Expr};
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

/// Extracts and wraps the inner `Expr` from a positional argument, or returns `None` for
/// named arguments.
pub(super) fn arg_expr(arg: &crate::ast::Arg) -> Option<Arc<Expr>> {
    match arg {
        crate::ast::Arg::Pos(e) => Some(Arc::new(e.clone())),
        _ => None,
    }
}
