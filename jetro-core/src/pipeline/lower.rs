use std::sync::Arc;

use crate::builtins::{
    BuiltinMethod, BuiltinPipelineSink, BuiltinPipelineStage, BuiltinViewSink, BuiltinViewStage,
};
use crate::{ast::Expr, context::EvalError, value::Val};

use super::{
    expr_label, plan_with_exprs, plan_with_kernels, sink_name, source_name, trace_enabled,
    BodyKernel, NumOp, Pipeline, PipelineBody, Plan, ReducerOp, ReducerSpec, Sink, SortSpec,
    Source, Stage,
};

impl Pipeline {
    /// Try to lower an `Expr` into a Pipeline.  Returns `None` for any
    /// shape this Phase 1 substrate doesn't yet handle — caller falls
    /// back to the existing opcode compilation path.
    ///
    /// Supported (Phase 1):
    ///   - `$.k1.k2…kN.<stage>*.<sink>` where each `kN` is a plain Field,
    ///     stages are zero-or-more of `filter` / `map` / `take` / `skip`,
    ///     and the sink is `count` / `len` / `sum` / nothing (Collect).
    ///
    /// Not yet supported and returns `None`:
    ///   - Any non-Root base
    ///   - Any non-Field step before the first method (e.g. `[idx]`)
    ///   - Lambda methods (`map(@.x + 1)` is fine; `map(lambda x: …)` is not)
    ///   - Any unrecognised method in stage position
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

    fn lower_with_reason(expr: &Expr) -> std::result::Result<Pipeline, &'static str> {
        Self::lower_inner(expr).ok_or("shape not yet supported")
    }

    fn lower_inner(expr: &Expr) -> Option<Pipeline> {
        use crate::ast::Step;
        let (base, steps) = match expr {
            Expr::Chain(b, s) => (b.as_ref(), s.as_slice()),
            _ => return None,
        };
        if !matches!(base, Expr::Root) {
            return None;
        }

        // Find where the field-chain prefix ends and stages begin.
        let mut field_end = 0;
        for s in steps {
            match s {
                Step::Field(_) => field_end += 1,
                _ => break,
            }
        }
        // Phase 1 deliberately does not lower bare `$.<method>` shapes
        // (no field-chain prefix) because the existing fused opcodes
        // (MapSplitLenSum, FilterFieldCmpLitMapField, etc.) often beat
        // a generic pull-based pipeline for those.  Field-chain prefix
        // signals a "scan over a sub-array" intent — the pipeline's
        // sweet spot.
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

    pub(crate) fn lower_from_source(
        source: Source,
        trailing: &[crate::ast::Step],
    ) -> Option<Pipeline> {
        Some(Self::lower_body_from_steps(trailing)?.with_source(source))
    }

    pub(crate) fn lower_body_from_steps(trailing: &[crate::ast::Step]) -> Option<PipelineBody> {
        // Decode the trailing methods into stages + a sink.
        // Compile each filter / map sub-Expr to a Program once so
        // Pipeline::run can reuse it per row.  Sub-programs run against
        // the current item bound as the VM's root, so `@.field` and
        // `@` references resolve to the row.
        let (stages, stage_exprs, sink) = decode_method_chain(trailing)?;
        let mut p = PipelineBody {
            stages,
            stage_exprs,
            sink,
            stage_kernels: Vec::new(),
            sink_kernels: Vec::new(),
        };
        rewrite(&mut p);
        // Phase A3 — classify per-stage sub-programs.  Per-row pull loop
        // reads these hints to choose a specialised inline path vs the
        // generic vm.exec fallback.  Also drives Step 3d Phase 3 reorder.
        let classify_kernels = |stages: &[Stage]| -> Vec<BodyKernel> {
            stages
                .iter()
                .map(|s| match s {
                    Stage::Filter(p, _) => BodyKernel::classify(p),
                    Stage::TakeWhile(p) => BodyKernel::classify(p),
                    Stage::Map(p, _) => BodyKernel::classify(p),
                    Stage::FlatMap(p, _) => BodyKernel::classify(p),
                    Stage::UniqueBy(Some(p)) => BodyKernel::classify(p),
                    Stage::GroupBy(p) => BodyKernel::classify(p),
                    Stage::Sort(spec) => spec
                        .key
                        .as_ref()
                        .map(|p| BodyKernel::classify(p))
                        .unwrap_or(BodyKernel::Generic),
                    _ => BodyKernel::Generic,
                })
                .collect()
        };
        let kernels = classify_kernels(&p.stages);

        // Step 3d planning — Phase 2/3/4 transforms with kernel-aware
        // cost/selectivity.  Result.stages / Result.sink replace the
        // pre-plan values.  Phase 1 + Phase 5 (demand prop, strategy
        // selection) re-runs at exec time on the final shape.
        let plan_result = plan_with_exprs(
            p.stages.clone(),
            p.stage_exprs.clone(),
            &kernels,
            p.sink.clone(),
        );
        p.stages = plan_result.stages;
        p.stage_exprs = plan_result.stage_exprs;
        p.sink = plan_result.sink;

        // Re-classify post-plan since Phase 4 merges may have produced
        // new sub-programs (e.g. Map+Map → field-chain-Map), or demand
        // planning may have absorbed a projection into a numeric sink.
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

    pub(crate) fn is_receiver_pipeline_start(step: &crate::ast::Step) -> bool {
        use crate::ast::Step;

        let Step::Method(name, args) = step else {
            return false;
        };
        is_receiver_pipeline_start_method(name.as_str(), args.len())
    }
}

fn is_receiver_pipeline_start_method(name: &str, arity: usize) -> bool {
    let method = BuiltinMethod::from_name(name);
    if method == BuiltinMethod::Unknown {
        return false;
    }

    let spec = method.spec();
    match arity {
        0 => {
            spec.view_sink.is_some()
                || spec.pipeline_sink.is_some()
                || spec.pipeline_stage == Some(BuiltinPipelineStage::Nullary)
        }
        1 if method == BuiltinMethod::Sort => true,
        1 => spec.pipeline_stage == Some(BuiltinPipelineStage::Unary),
        _ => false,
    }
}

/// Step 3d-extension (A2): try to decode the body of a Map(...) call as
/// its own pipeline Plan.  Body must be `Expr::Chain(Expr::Current, [...])`
/// with at least one trailing method or field access.  Field-chain
/// prefix becomes a leading `Stage::Map(field-walk)`; trailing methods
/// decode via the shared `decode_method_chain` helper.  Inner Plan runs
/// `plan_with_kernels` so it picks IndexedDispatch / BarrierMaterialise /
/// EarlyExit / DoneTerminating / PullLoop strategies recursively.
///
/// Returns None when the body is opaque (lambda or side effect) — caller
/// falls back to `Stage::Map(opaque_program)`.
fn try_decode_map_body(arg: &crate::ast::Arg) -> Option<Plan> {
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

    // Field-chain prefix walks @.a.b.c → seed.a.b.c.  Encoded as a
    // leading Map stage whose body is the FieldChain program over @.
    let mut field_end = 0;
    for s in steps {
        match s {
            Step::Field(_) => field_end += 1,
            _ => break,
        }
    }
    let trailing = &steps[field_end..];
    // Require at least one trailing method — pure field access alone
    // is what plain Stage::Map(@.a.b.c) already handles.
    if trailing.is_empty() {
        return None;
    }

    let mut stages: Vec<Stage> = Vec::new();
    if field_end > 0 {
        // Build a sub-program `[PushCurrent, FieldChain([...])]` that
        // walks the prefix from the seed.
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

    // Run the same planning the outer pipeline does.  Kernel
    // classification feeds Phase 3 reorder + Phase 5 strategy select.
    let kernels: Vec<BodyKernel> = stages
        .iter()
        .map(|s| match s {
            Stage::Filter(p, _) => BodyKernel::classify(p),
            Stage::TakeWhile(p) => BodyKernel::classify(p),
            Stage::Map(p, _) => BodyKernel::classify(p),
            Stage::FlatMap(p, _) => BodyKernel::classify(p),
            Stage::UniqueBy(Some(p)) => BodyKernel::classify(p),
            Stage::GroupBy(p) => BodyKernel::classify(p),
            Stage::Sort(spec) => spec
                .key
                .as_ref()
                .map(|p| BodyKernel::classify(p))
                .unwrap_or(BodyKernel::Generic),
            _ => BodyKernel::Generic,
        })
        .collect();
    Some(plan_with_kernels(stages, &kernels, sink))
}

/// Run a `Plan` against a single seed Val (Step 3d-extension A2).  Used
/// by `Stage::CompiledMap` per outer element.  Wraps seed as a single-
/// element Val::Arr and runs through a synth Pipeline so all strategy
/// selection (IndexedDispatch / EarlyExit / etc.) applies recursively.
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

/// Decode a slice of `Step::Method(...)` into pipeline `(stages, sink)`.
/// Shared by top-level `lower_inner` and Step 3d-extension (A2)
/// recursive sub-pipeline planning for `Map(@.<chain>)` bodies.
/// Returns `None` if any method shape isn't recognised.
fn decode_method_chain(
    trailing: &[crate::ast::Step],
) -> Option<(Vec<Stage>, Vec<Option<Arc<Expr>>>, Sink)> {
    use crate::ast::{Arg, Step};
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
                match (method, args.len(), is_last) {
                    // Lambda-bearing methods — pre-compile body as vm::Program.
                    (BuiltinMethod::TakeWhile, 1, _) => {
                        stages.push(Stage::TakeWhile(compile_subexpr(&args[0])?));
                        stage_exprs.push(arg_expr(&args[0]));
                    }
                    (BuiltinMethod::DropWhile, 1, _) => {
                        stages.push(Stage::DropWhile(compile_subexpr(&args[0])?));
                        stage_exprs.push(arg_expr(&args[0]));
                    }
                    (BuiltinMethod::IndicesWhere, 1, true) => {
                        stages.push(Stage::IndicesWhere(compile_subexpr(&args[0])?));
                        stage_exprs.push(arg_expr(&args[0]));
                        sink = Sink::First(BuiltinViewSink::First);
                    }
                    (BuiltinMethod::FindIndex, 1, true) => {
                        stages.push(Stage::FindIndex(compile_subexpr(&args[0])?));
                        stage_exprs.push(arg_expr(&args[0]));
                        sink = Sink::First(BuiltinViewSink::First);
                    }
                    (BuiltinMethod::MaxBy, 1, true) => {
                        stages.push(Stage::MaxBy(compile_subexpr(&args[0])?));
                        stage_exprs.push(arg_expr(&args[0]));
                        sink = Sink::First(BuiltinViewSink::First);
                    }
                    (BuiltinMethod::MinBy, 1, true) => {
                        stages.push(Stage::MinBy(compile_subexpr(&args[0])?));
                        stage_exprs.push(arg_expr(&args[0]));
                        sink = Sink::First(BuiltinViewSink::First);
                    }
                    (BuiltinMethod::TransformValues, 1, _) => {
                        stages.push(Stage::TransformValues(compile_subexpr(&args[0])?));
                        stage_exprs.push(arg_expr(&args[0]));
                    }
                    (BuiltinMethod::TransformKeys, 1, _) => {
                        stages.push(Stage::TransformKeys(compile_subexpr(&args[0])?));
                        stage_exprs.push(arg_expr(&args[0]));
                    }
                    (BuiltinMethod::FilterValues, 1, _) => {
                        stages.push(Stage::FilterValues(compile_subexpr(&args[0])?));
                        stage_exprs.push(arg_expr(&args[0]));
                    }
                    (BuiltinMethod::FilterKeys, 1, _) => {
                        stages.push(Stage::FilterKeys(compile_subexpr(&args[0])?));
                        stage_exprs.push(arg_expr(&args[0]));
                    }
                    (
                        BuiltinMethod::Filter | BuiltinMethod::Find | BuiltinMethod::FindAll,
                        1,
                        _,
                    ) => {
                        stages.push(Stage::Filter(
                            compile_subexpr(&args[0])?,
                            BuiltinViewStage::Filter,
                        ));
                        stage_exprs.push(arg_expr(&args[0]));
                    }
                    // find_first / find_one: terminal — first match or Null.
                    (BuiltinMethod::FindFirst | BuiltinMethod::FindOne, 1, true) => {
                        stages.push(Stage::Filter(
                            compile_subexpr(&args[0])?,
                            BuiltinViewStage::Filter,
                        ));
                        stage_exprs.push(arg_expr(&args[0]));
                        sink = Sink::First(BuiltinViewSink::First);
                    }
                    (BuiltinMethod::FindFirst | BuiltinMethod::FindOne, 1, false) => {
                        stages.push(Stage::Filter(
                            compile_subexpr(&args[0])?,
                            BuiltinViewStage::Filter,
                        ));
                        stage_exprs.push(arg_expr(&args[0]));
                    }
                    // count_by / index_by: barrier reductions.  Trailing
                    // result is single Obj; force Sink::First when terminal.
                    (BuiltinMethod::CountBy, 1, true) => {
                        stages.push(Stage::CountBy(compile_subexpr(&args[0])?));
                        stage_exprs.push(arg_expr(&args[0]));
                        sink = Sink::First(BuiltinViewSink::First);
                    }
                    (BuiltinMethod::CountBy, 1, false) => {
                        stages.push(Stage::CountBy(compile_subexpr(&args[0])?));
                        stage_exprs.push(arg_expr(&args[0]));
                    }
                    (BuiltinMethod::IndexBy, 1, true) => {
                        stages.push(Stage::IndexBy(compile_subexpr(&args[0])?));
                        stage_exprs.push(arg_expr(&args[0]));
                        sink = Sink::First(BuiltinViewSink::First);
                    }
                    (BuiltinMethod::IndexBy, 1, false) => {
                        stages.push(Stage::IndexBy(compile_subexpr(&args[0])?));
                        stage_exprs.push(arg_expr(&args[0]));
                    }
                    (BuiltinMethod::Map, 1, _) => {
                        // A2: try recursive sub-pipeline planning first.
                        // Body shapes that decode as a chain of recognised
                        // methods over @ become Stage::CompiledMap; opaque
                        // bodies (lambdas or side effects) fall through.
                        match try_decode_map_body(&args[0]) {
                            Some(plan) => {
                                stages.push(Stage::CompiledMap(Arc::new(plan)));
                                stage_exprs.push(arg_expr(&args[0]));
                            }
                            None => {
                                stages.push(Stage::Map(
                                    compile_subexpr(&args[0])?,
                                    BuiltinViewStage::Map,
                                ));
                                stage_exprs.push(arg_expr(&args[0]));
                            }
                        }
                    }
                    (BuiltinMethod::FlatMap, 1, _) => {
                        stages.push(Stage::FlatMap(
                            compile_subexpr(&args[0])?,
                            BuiltinViewStage::FlatMap,
                        ));
                        stage_exprs.push(arg_expr(&args[0]));
                    }
                    (BuiltinMethod::Reverse, 0, _) => {
                        let cancel = method
                            .spec()
                            .cancellation
                            .expect("reverse builtin must define cancellation metadata");
                        stages.push(Stage::Reverse(cancel));
                        stage_exprs.push(None);
                    }
                    (BuiltinMethod::Unique, 0, _) => {
                        stages.push(Stage::UniqueBy(None));
                        stage_exprs.push(None);
                    }
                    (BuiltinMethod::UniqueBy, 1, _) => {
                        stages.push(Stage::UniqueBy(Some(compile_subexpr(&args[0])?)));
                        stage_exprs.push(arg_expr(&args[0]));
                    }
                    (BuiltinMethod::GroupBy, 1, _) => {
                        stages.push(Stage::GroupBy(compile_subexpr(&args[0])?));
                        stage_exprs.push(arg_expr(&args[0]));
                    }
                    (BuiltinMethod::Sort, 0, _) => {
                        stages.push(Stage::Sort(SortSpec::identity()));
                        stage_exprs.push(None);
                    }
                    (BuiltinMethod::Sort, 1, _) => {
                        let (spec, expr) = compile_sort_spec(&args[0])?;
                        stages.push(Stage::Sort(spec));
                        stage_exprs.push(expr);
                    }
                    (BuiltinMethod::Take, 1, _) => {
                        let n = match &args[0] {
                            Arg::Pos(Expr::Int(n)) if *n >= 0 => *n as usize,
                            _ => return None,
                        };
                        let spec = method.spec();
                        let stage = spec.view_stage?;
                        if stage != BuiltinViewStage::Take {
                            return None;
                        }
                        stages.push(Stage::Take(n, stage, spec.stage_merge?));
                        stage_exprs.push(None);
                    }
                    (BuiltinMethod::Skip, 1, _) => {
                        let n = match &args[0] {
                            Arg::Pos(Expr::Int(n)) if *n >= 0 => *n as usize,
                            _ => return None,
                        };
                        let spec = method.spec();
                        let stage = spec.view_stage?;
                        if stage != BuiltinViewStage::Skip {
                            return None;
                        }
                        stages.push(Stage::Skip(n, stage, spec.stage_merge?));
                        stage_exprs.push(None);
                    }
                    (BuiltinMethod::Split, 1, _) => {
                        let sep = match &args[0] {
                            Arg::Pos(Expr::Str(s)) => Arc::<str>::from(s.as_str()),
                            _ => return None,
                        };
                        stages.push(Stage::Split(sep));
                        stage_exprs.push(None);
                    }
                    // Whole-receiver builtins intentionally stay out of
                    // pipeline lowering; `builtins::BuiltinCall` owns the
                    // per-element allowlist used above.
                    (BuiltinMethod::Slice, 1, _) => {
                        let start = match &args[0] {
                            Arg::Pos(Expr::Int(n)) => *n,
                            _ => return None,
                        };
                        stages.push(Stage::Slice(start, None));
                        stage_exprs.push(None);
                    }
                    (BuiltinMethod::Slice, 2, _) => {
                        let (start, end) = match (&args[0], &args[1]) {
                            (Arg::Pos(Expr::Int(s)), Arg::Pos(Expr::Int(e))) => (*s, Some(*e)),
                            _ => return None,
                        };
                        stages.push(Stage::Slice(start, end));
                        stage_exprs.push(None);
                    }
                    (BuiltinMethod::Replace | BuiltinMethod::ReplaceAll, 2, _) => {
                        let (needle, replacement) = match (&args[0], &args[1]) {
                            (Arg::Pos(Expr::Str(n)), Arg::Pos(Expr::Str(r))) => {
                                (Arc::<str>::from(n.as_str()), Arc::<str>::from(r.as_str()))
                            }
                            _ => return None,
                        };
                        stages.push(Stage::Replace {
                            needle,
                            replacement,
                            all: method == BuiltinMethod::ReplaceAll,
                        });
                        stage_exprs.push(None);
                    }
                    (BuiltinMethod::Chunk, 1, _) => {
                        let n = match &args[0] {
                            Arg::Pos(Expr::Int(n)) if *n >= 1 => *n as usize,
                            _ => return None,
                        };
                        stages.push(Stage::Chunk(n));
                        stage_exprs.push(None);
                    }
                    (BuiltinMethod::Window, 1, _) => {
                        let n = match &args[0] {
                            Arg::Pos(Expr::Int(n)) if *n >= 1 => *n as usize,
                            _ => return None,
                        };
                        stages.push(Stage::Window(n));
                        stage_exprs.push(None);
                    }
                    (BuiltinMethod::ApproxCountDistinct, 0, true) => {
                        sink = Sink::ApproxCountDistinct;
                    }
                    (_, _, true) => {
                        sink = terminal_sink_for_method(method, args)?;
                    }
                    _ => return None,
                }
            }
            _ => return None,
        }
    }
    Some((stages, stage_exprs, sink))
}

fn terminal_sink_for_method(method: BuiltinMethod, args: &[crate::ast::Arg]) -> Option<Sink> {
    let spec = method.spec();
    match spec.pipeline_sink {
        Some(BuiltinPipelineSink::ApproxCountDistinct) if args.is_empty() => {
            return Some(Sink::ApproxCountDistinct);
        }
        None => {}
        _ => return None,
    }

    match spec.view_sink? {
        BuiltinViewSink::Count => match args {
            [] => Some(Sink::Reducer(ReducerSpec::count())),
            [arg] if method == BuiltinMethod::Count => Some(Sink::Reducer(ReducerSpec {
                op: ReducerOp::Count,
                predicate: Some(compile_subexpr(arg)?),
                projection: None,
            })),
            _ => None,
        },
        BuiltinViewSink::Numeric => Some(Sink::Reducer(ReducerSpec {
            op: ReducerOp::Numeric(NumOp::from_builtin_reducer(spec.numeric_reducer?)),
            predicate: None,
            projection: match args {
                [] => None,
                [arg] => Some(compile_subexpr(arg)?),
                _ => return None,
            },
        })),
        BuiltinViewSink::First if args.is_empty() => Some(Sink::First(BuiltinViewSink::First)),
        BuiltinViewSink::Last if args.is_empty() => Some(Sink::Last(BuiltinViewSink::Last)),
        _ => None,
    }
}

/// Apply algebraic rewrite rules until fixed point or a fuel limit
/// expires.  Rules listed in `project_pipeline_ir.md`; this function
/// implements the subset that operates on the currently-supported
/// Stage / Sink set (`Filter` / `Map` / `Take` / `Skip` /
/// `Collect` / `Count` / `Sum` and the fused sinks).
///
/// Rules NOT yet implemented (require Stage variants the lowering
/// + execution don't yet emit):
///   `Sort ∘ Take → TopK`            (no Sort stage)
///   `Reverse ∘ Reverse → id`        (no Reverse stage)
///   `UniqueBy(k) ∘ UniqueBy(k) → UniqueBy(k)`
///   `Pick(a) ∘ Pick(b) → Pick(a ∩ b)`
///   `DeepScan(k) ∘ Filter(p on k) → DeepScanFiltered(k, p)`
///   `DeepScan(k) ∘ Pick({k}) → DeepScan(k)`
///   `Filter(p) ∘ Map(f) → Map(f) ∘ Filter(p ∘ f⁻¹)`  (needs SSA dep)
///
/// Each rule is monotonic — strictly reduces stage count or rewrites
/// to a structurally smaller form — so a fuel limit of 16 protects
/// against pathological loops without affecting correctness.
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

/// One round of the rewrite loop.  Returns `true` if any rule fired,
/// `false` if the pipeline is at a local fixed point.
///
/// Most algebraic rules deleted in the Step 1+2 mop-up — Step 3d
/// `plan_with_kernels()` (called from `lower_with_reason`) now subsumes:
///   - Filter(true) → identity                     (Phase 4 const-fold)
///   - Skip+Skip / Take+Take / Reverse∘Reverse     (Phase 4)
///   - Sort∘Sort / UniqueBy∘UniqueBy idempotence   (Phase 4)
///   - Map∘Count drop                              (Phase 2)
///   - Filter run reorder by selectivity           (Phase 3)
///
/// What remains here:
///   - Filter(false) → empty pipeline.  Phase 4 doesn't have access to
///     the Sink to swap to its identity-element form, so this stays.
///   - Map(f) ∘ Filter(g) ∘ Filter(h) etc.: the kernel-aware Filter+
///     Filter merge via `Opcode::AndOp` and the Map+Map field-chain
///     fusion — both build new `vm::Program`s, which the Phase 4
///     stage-only API can't do without a wider rewrite.  Kept here.
///   - Map ∘ Take pushdown: still strictly correct + a perf win, kept.
fn rewrite_step(p: &mut PipelineBody) -> bool {
    use crate::vm::Opcode;

    // Filter(false) → empty pipeline.  Filter(true) handled by Phase 4.
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
        // Empty input: existing accumulators in run() yield Int(0) /
        // Float(0.0) / Val::arr([]) — clearing stages suffices.
        p.stages.clear();
        p.stage_exprs.clear();
        return true;
    }

    // Filter+Filter → AndOp-merged Filter (kernel-aware vm::Program build).
    // Map+Map → FieldChain Map (kernel-aware vm::Program build).
    // Both construct new programs — keep here until Phase 4 gains a
    // program-building API.
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

    // Pushdown: Map(f) ∘ Take(n) → Take(n) ∘ Map(f).
    // Strict perf win — composed exec runs map only n times.
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

/// If `prog` evaluates to a constant boolean independent of `@`,
/// return the literal value.  Currently matches the trivial shape
/// `[PushBool(b)]`; future SSA work could constant-fold larger preds.
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

/// Compile a `.filter(arg)` / `.map(arg)` sub-expression into a `Program`
/// the VM can run against a row's `@`. The argument's expression is
/// wrapped so that bare-ident shorthand (`map(total)`) becomes
/// `@.total` for evaluation against the current row. Runtime adapter
/// helpers apply the same rule for non-pipeline calls, but the pipeline
/// needs an explicit bytecode body here.
fn compile_subexpr(arg: &crate::ast::Arg) -> Option<Arc<crate::vm::Program>> {
    use crate::ast::{Arg, Expr, Step};
    let inner = match arg {
        Arg::Pos(e) => e,
        _ => return None,
    };
    let rooted: Expr = match inner {
        // Bare ident `total` -> `@.total`
        Expr::Ident(name) => Expr::Chain(Box::new(Expr::Current), vec![Step::Field(name.clone())]),
        // `@...` chains: keep base = Current, accept as-is.
        Expr::Chain(base, _) if matches!(base.as_ref(), Expr::Current) => inner.clone(),
        // Anything else: wrap as-is; VM resolves `@` via Current refs.
        other => other.clone(),
    };
    Some(Arc::new(crate::vm::Compiler::compile(&rooted, "")))
}

fn compile_sort_spec(arg: &crate::ast::Arg) -> Option<(SortSpec, Option<Arc<Expr>>)> {
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

fn arg_expr(arg: &crate::ast::Arg) -> Option<Arc<Expr>> {
    match arg {
        crate::ast::Arg::Pos(e) => Some(Arc::new(e.clone())),
        _ => None,
    }
}
