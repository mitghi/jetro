//! Compiler: lowers an `Expr` AST to a flat `Arc<[Opcode]>` `Program`.
//!
//! `Compiler` runs a sequence of peephole passes (`RootChain` fusion,
//! `FilterCount` fusion, `ConstFold`, demand annotation) controlled by
//! `PassConfig`. Split out of `vm.rs` to keep each file focused.

use smallvec::SmallVec;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use crate::parse::ast::*;
use crate::builtins::BuiltinMethod;
use crate::data::context::EvalError;
use crate::vm::{
    Opcode, Program, CompiledCall, CompiledObjEntry, KvStep, CompiledFSPart,
    BindObjSpec, CompiledPipeStep, CompSpec, DictCompSpec,
    CompiledMatch, CompiledPatch, CompiledPatchOp, CompiledPatchVal, CompiledPathStep, MatchOp, MatchSlot,
    fresh_ics, disable_opcode_fusion,
};

/// Classify each pre-compiled sub-program against the `BodyKernel`
/// fast-path catalog so higher-order builtins (`.filter`, `.map`,
/// `.any`, ...) can dispatch through `eval_kernel` at native Rust
/// speed instead of re-entering the VM per element.
pub(crate) fn classify_sub_kernels(
    progs: &[Arc<Program>],
) -> Arc<[crate::exec::pipeline::BodyKernel]> {
    progs
        .iter()
        .map(|p| crate::exec::pipeline::BodyKernel::classify(p))
        .collect::<Vec<_>>()
        .into()
}

/// Compile-time variable scope used by the `Compiler` to decide whether an
/// identifier refers to a bound variable or a built-in/field name.
#[derive(Clone, Default)]
pub(crate) struct VarCtx {
    /// Deduplicated set of names currently in scope; stored inline for small counts.
    known: SmallVec<[Arc<str>; 4]>,
}

impl VarCtx {
    /// Return a new context extended with `name`, deduplicating if already present.
    fn with_var(&self, name: &str) -> Self {
        let mut v = self.clone();
        if !v.known.iter().any(|k| k.as_ref() == name) {
            v.known.push(Arc::from(name));
        }
        v
    }
    /// Return a new context extended with all `names`, deduplicating each.
    fn with_vars(&self, names: &[String]) -> Self {
        let mut v = self.clone();
        for n in names {
            if !v.known.iter().any(|k| k.as_ref() == n.as_str()) {
                v.known.push(Arc::from(n.as_str()));
            }
        }
        v
    }
    /// Return `true` if `name` is currently in scope as a bound variable.
    fn has(&self, name: &str) -> bool {
        self.known.iter().any(|k| k.as_ref() == name)
    }
}


/// Stateless unit struct that compiles an `Expr` AST into a flat `Program`.
/// All methods are associated functions; no instance state is needed.
pub struct Compiler;

impl Compiler {
    /// Compile `expr` with all optimisation passes enabled and sub-program deduplication.
    /// `source` is stored verbatim in the returned `Program` for cache keying.
    pub fn compile(expr: &Expr, source: &str) -> Program {
        // Phase B: fuse contiguous same-root chain-writes into multi-op
        // `Expr::Patch` nodes before emitting bytecode. Phase D's
        // `CompiledPatchTrie::from_ops` then auto-routes the resulting
        // multi-op patches onto the shared-`Arc::make_mut` path.
        let mut e = crate::plan::patch_fusion::fuse_writes(expr.clone());
        Self::reorder_and_operands(&mut e);
        let ctx = VarCtx::default();
        let ops = Self::optimize(Self::emit(&e, &ctx));
        let prog = Program::new(ops, source);
        
        let deduped = crate::plan::analysis::dedup_subprograms(&prog);
        let ics = fresh_ics(deduped.ops.len());
        Program {
            ops: deduped.ops.clone(),
            source: prog.source,
            id: prog.id,
            is_structural: prog.is_structural,
            ics,
        }
    }

    /// Recursively reorder the operands of `&&` expressions so the more selective
    /// (cheaper) operand comes first, enabling short-circuit evaluation to skip work.
    fn reorder_and_operands(expr: &mut Expr) {
        use crate::plan::analysis::selectivity_score;
        match expr {
            Expr::BinOp(l, op, r) if *op == BinOp::And => {
                Self::reorder_and_operands(l);
                Self::reorder_and_operands(r);
                if selectivity_score(r) < selectivity_score(l) {
                    std::mem::swap(l, r);
                }
            }
            Expr::BinOp(l, _, r) => {
                Self::reorder_and_operands(l);
                Self::reorder_and_operands(r);
            }
            Expr::UnaryNeg(e) | Expr::Not(e) | Expr::Kind { expr: e, .. } => {
                Self::reorder_and_operands(e)
            }
            Expr::Coalesce(l, r) => {
                Self::reorder_and_operands(l);
                Self::reorder_and_operands(r);
            }
            Expr::Chain(base, steps) => {
                Self::reorder_and_operands(base);
                for s in steps {
                    match s {
                        crate::parse::ast::Step::DynIndex(e) | crate::parse::ast::Step::InlineFilter(e) => {
                            Self::reorder_and_operands(e)
                        }
                        crate::parse::ast::Step::Method(_, args)
                        | crate::parse::ast::Step::OptMethod(_, args) => {
                            for a in args {
                                match a {
                                    crate::parse::ast::Arg::Pos(e) | crate::parse::ast::Arg::Named(_, e) => {
                                        Self::reorder_and_operands(e)
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            Expr::Let { init, body, .. } => {
                Self::reorder_and_operands(init);
                Self::reorder_and_operands(body);
            }
            Expr::Pipeline { base, steps } => {
                Self::reorder_and_operands(base);
                for s in steps {
                    if let crate::parse::ast::PipeStep::Forward(e) = s {
                        Self::reorder_and_operands(e);
                    }
                }
            }
            Expr::Object(fields) => {
                for f in fields {
                    match f {
                        crate::parse::ast::ObjField::Kv { val, .. } => Self::reorder_and_operands(val),
                        crate::parse::ast::ObjField::Dynamic { key, val } => {
                            Self::reorder_and_operands(key);
                            Self::reorder_and_operands(val);
                        }
                        crate::parse::ast::ObjField::Spread(e) => Self::reorder_and_operands(e),
                        _ => {}
                    }
                }
            }
            Expr::Array(elems) => {
                for e in elems {
                    match e {
                        crate::parse::ast::ArrayElem::Expr(e) | crate::parse::ast::ArrayElem::Spread(e) => {
                            Self::reorder_and_operands(e)
                        }
                    }
                }
            }
            Expr::ListComp {
                expr, iter, cond, ..
            }
            | Expr::SetComp {
                expr, iter, cond, ..
            }
            | Expr::GenComp {
                expr, iter, cond, ..
            } => {
                Self::reorder_and_operands(expr);
                Self::reorder_and_operands(iter);
                if let Some(c) = cond {
                    Self::reorder_and_operands(c);
                }
            }
            Expr::DictComp {
                key,
                val,
                iter,
                cond,
                ..
            } => {
                Self::reorder_and_operands(key);
                Self::reorder_and_operands(val);
                Self::reorder_and_operands(iter);
                if let Some(c) = cond {
                    Self::reorder_and_operands(c);
                }
            }
            Expr::Lambda { body, .. } => Self::reorder_and_operands(body),
            Expr::GlobalCall { args, .. } => {
                for a in args {
                    match a {
                        crate::parse::ast::Arg::Pos(e) | crate::parse::ast::Arg::Named(_, e) => {
                            Self::reorder_and_operands(e)
                        }
                    }
                }
            }
            _ => {}
        }
    }

    /// Parse and compile `input` with all default passes; available in test builds only.
    #[cfg(test)]
    pub fn compile_str(input: &str) -> Result<Program, EvalError> {
        let expr = crate::parse::parser::parse(input).map_err(|e| EvalError(e.to_string()))?;
        Ok(Self::compile(&expr, input))
    }

    /// Parse and compile `input` with the passes controlled by `config`.
    /// Used by `VM::get_or_compile` so pass selection can vary per `VM` instance.
    pub fn compile_str_with_config(input: &str, config: PassConfig) -> Result<Program, EvalError> {
        let expr = crate::parse::parser::parse(input).map_err(|e| EvalError(e.to_string()))?;
        let mut e = expr.clone();
        if config.reorder_and {
            Self::reorder_and_operands(&mut e);
        }
        let ctx = VarCtx::default();
        let ops = Self::optimize_with(Self::emit(&e, &ctx), config);
        let prog = Program::new(ops, input);
        if config.dedup_subprogs {
            let deduped = crate::plan::analysis::dedup_subprograms(&prog);
            let ics = fresh_ics(deduped.ops.len());
            Ok(Program {
                ops: deduped.ops.clone(),
                source: prog.source,
                id: prog.id,
                is_structural: prog.is_structural,
                ics,
            })
        } else {
            Ok(prog)
        }
    }

    /// Run all peephole passes with the default `PassConfig`.
    fn optimize(ops: Vec<Opcode>) -> Vec<Opcode> {
        Self::optimize_with(ops, PassConfig::default())
    }

    /// Run the subset of peephole passes enabled in `cfg`, respecting the
    /// `JETRO_DISABLE_OPCODE_FUSION` environment override.
    fn optimize_with(ops: Vec<Opcode>, cfg: PassConfig) -> Vec<Opcode> {
        use crate::compile::passes as cp;
        let no_fusion = disable_opcode_fusion();
        let ops = if cfg.root_chain && !no_fusion { cp::pass_root_chain(ops) } else { ops };
        let ops = if cfg.field_chain && !no_fusion { cp::pass_field_chain(ops) } else { ops };
        let ops = if cfg.filter_fusion { cp::pass_field_specialise(ops) } else { ops };
        let ops = if !no_fusion { cp::pass_list_comp_specialise(ops) } else { ops };
        let ops = if cfg.strength_reduce { cp::pass_strength_reduce(ops) } else { ops };
        let ops = if cfg.redundant_ops { cp::pass_redundant_ops(ops) } else { ops };
        let ops = if cfg.kind_check_fold { cp::pass_kind_check_fold(ops) } else { ops };
        let ops = if cfg.method_const { cp::pass_method_const_fold(ops) } else { ops };
        let ops = if cfg.const_fold { cp::pass_const_fold(ops) } else { ops };
        let ops = if cfg.nullness { cp::pass_nullness_opt_field(ops) } else { ops };
        let ops = if !no_fusion { cp::pass_method_demand(ops) } else { ops };
        ops
    }

    /// Emit opcodes for `expr` into a fresh vector and return it.
    fn emit(expr: &Expr, ctx: &VarCtx) -> Vec<Opcode> {
        let mut ops = Vec::new();
        Self::emit_into(expr, ctx, &mut ops);
        ops
    }

    /// Recursively emit opcodes for `expr` into `ops`, consulting `ctx` to distinguish
    /// variable references from field/built-in names.
    fn emit_into(expr: &Expr, ctx: &VarCtx, ops: &mut Vec<Opcode>) {
        match expr {
            Expr::Null => ops.push(Opcode::PushNull),
            Expr::Bool(b) => ops.push(Opcode::PushBool(*b)),
            Expr::Int(n) => ops.push(Opcode::PushInt(*n)),
            Expr::Float(f) => ops.push(Opcode::PushFloat(*f)),
            Expr::Str(s) => ops.push(Opcode::PushStr(Arc::from(s.as_str()))),
            Expr::Root => ops.push(Opcode::PushRoot),
            Expr::Current => ops.push(Opcode::PushCurrent),

            Expr::FString(parts) => {
                let compiled: Vec<CompiledFSPart> = parts
                    .iter()
                    .map(|p| match p {
                        FStringPart::Lit(s) => CompiledFSPart::Lit(Arc::from(s.as_str())),
                        FStringPart::Interp { expr, fmt } => CompiledFSPart::Interp {
                            prog: Arc::new(Self::compile_sub(expr, ctx)),
                            fmt: fmt.clone(),
                        },
                    })
                    .collect();
                ops.push(Opcode::FString(compiled.into()));
            }

            Expr::Ident(name) => ops.push(Opcode::LoadIdent(Arc::from(name.as_str()))),

            Expr::Chain(base, steps) => {
                Self::emit_into(base, ctx, ops);
                for step in steps {
                    Self::emit_step(step, ctx, ops);
                }
            }

            Expr::UnaryNeg(e) => {
                Self::emit_into(e, ctx, ops);
                ops.push(Opcode::Neg);
            }
            Expr::Not(e) => {
                Self::emit_into(e, ctx, ops);
                ops.push(Opcode::Not);
            }

            Expr::BinOp(l, op, r) => Self::emit_binop(l, *op, r, ctx, ops),

            Expr::Coalesce(lhs, rhs) => {
                Self::emit_into(lhs, ctx, ops);
                let rhs_prog = Arc::new(Self::compile_sub(rhs, ctx));
                ops.push(Opcode::CoalesceOp(rhs_prog));
            }

            Expr::Kind { expr, ty, negate } => {
                Self::emit_into(expr, ctx, ops);
                ops.push(Opcode::KindCheck {
                    ty: *ty,
                    negate: *negate,
                });
            }

            Expr::Object(fields) => {
                let entries: Vec<CompiledObjEntry> = fields
                    .iter()
                    .map(|f| match f {
                        ObjField::Short(name) => CompiledObjEntry::Short {
                            name: Arc::from(name.as_str()),
                            ic: Arc::new(AtomicU64::new(0)),
                        },
                        ObjField::Kv {
                            key,
                            val,
                            optional,
                            cond,
                        } if cond.is_none() && Self::try_kv_path_steps(val).is_some() => {
                            let steps: Vec<KvStep> = Self::try_kv_path_steps(val).unwrap();
                            let n = steps.len();
                            let mut ics_vec: Vec<AtomicU64> = Vec::with_capacity(n);
                            for _ in 0..n {
                                ics_vec.push(AtomicU64::new(0));
                            }
                            CompiledObjEntry::KvPath {
                                key: Arc::from(key.as_str()),
                                steps: steps.into(),
                                optional: *optional,
                                ics: ics_vec.into(),
                            }
                        }
                        ObjField::Kv {
                            key,
                            val,
                            optional,
                            cond,
                        } => CompiledObjEntry::Kv {
                            key: Arc::from(key.as_str()),
                            prog: Arc::new(Self::compile_sub(val, ctx)),
                            optional: *optional,
                            cond: cond.as_ref().map(|c| Arc::new(Self::compile_sub(c, ctx))),
                        },
                        ObjField::Dynamic { key, val } => CompiledObjEntry::Dynamic {
                            key: Arc::new(Self::compile_sub(key, ctx)),
                            val: Arc::new(Self::compile_sub(val, ctx)),
                        },
                        ObjField::Spread(e) => {
                            CompiledObjEntry::Spread(Arc::new(Self::compile_sub(e, ctx)))
                        }
                        ObjField::SpreadDeep(e) => {
                            CompiledObjEntry::SpreadDeep(Arc::new(Self::compile_sub(e, ctx)))
                        }
                    })
                    .collect();
                ops.push(Opcode::MakeObj(entries.into()));
            }

            Expr::Array(elems) => {
                
                
                let progs: Vec<(Arc<Program>, bool)> = elems
                    .iter()
                    .map(|e| match e {
                        ArrayElem::Expr(ex) => (Arc::new(Self::compile_sub(ex, ctx)), false),
                        ArrayElem::Spread(ex) => (Arc::new(Self::compile_sub(ex, ctx)), true),
                    })
                    .collect();
                ops.push(Opcode::MakeArr(progs.into()));
            }

            Expr::Pipeline { base, steps } => {
                Self::emit_pipeline(base, steps, ctx, ops);
            }

            Expr::ListComp {
                expr,
                vars,
                iter,
                cond,
            } => {
                let inner_ctx = ctx.with_vars(vars);
                ops.push(Opcode::ListComp(Arc::new(CompSpec {
                    expr: Arc::new(Self::compile_sub(expr, &inner_ctx)),
                    vars: vars
                        .iter()
                        .map(|v| Arc::from(v.as_str()))
                        .collect::<Vec<_>>()
                        .into(),
                    iter: Arc::new(Self::compile_sub(iter, ctx)),
                    cond: cond
                        .as_ref()
                        .map(|c| Arc::new(Self::compile_sub(c, &inner_ctx))),
                })));
            }

            Expr::DictComp {
                key,
                val,
                vars,
                iter,
                cond,
            } => {
                let inner_ctx = ctx.with_vars(vars);
                ops.push(Opcode::DictComp(Arc::new(DictCompSpec {
                    key: Arc::new(Self::compile_sub(key, &inner_ctx)),
                    val: Arc::new(Self::compile_sub(val, &inner_ctx)),
                    vars: vars
                        .iter()
                        .map(|v| Arc::from(v.as_str()))
                        .collect::<Vec<_>>()
                        .into(),
                    iter: Arc::new(Self::compile_sub(iter, ctx)),
                    cond: cond
                        .as_ref()
                        .map(|c| Arc::new(Self::compile_sub(c, &inner_ctx))),
                })));
            }

            Expr::SetComp {
                expr,
                vars,
                iter,
                cond,
            }
            | Expr::GenComp {
                expr,
                vars,
                iter,
                cond,
            } => {
                let inner_ctx = ctx.with_vars(vars);
                ops.push(Opcode::SetComp(Arc::new(CompSpec {
                    expr: Arc::new(Self::compile_sub(expr, &inner_ctx)),
                    vars: vars
                        .iter()
                        .map(|v| Arc::from(v.as_str()))
                        .collect::<Vec<_>>()
                        .into(),
                    iter: Arc::new(Self::compile_sub(iter, ctx)),
                    cond: cond
                        .as_ref()
                        .map(|c| Arc::new(Self::compile_sub(c, &inner_ctx))),
                })));
            }

            Expr::Lambda { .. } => {
                
                ops.push(Opcode::PushNull);
            }

            Expr::Let { name, init, body } => {
                
                
                if crate::plan::analysis::expr_is_pure(init)
                    && !crate::plan::analysis::expr_uses_ident(body, name)
                {
                    Self::emit_into(body, ctx, ops);
                } else {
                    Self::emit_into(init, ctx, ops);
                    let body_ctx = ctx.with_var(name);
                    let body_prog = Arc::new(Self::compile_sub(body, &body_ctx));
                    ops.push(Opcode::LetExpr {
                        name: Arc::from(name.as_str()),
                        body: body_prog,
                    });
                }
            }

            Expr::IfElse { cond, then_, else_ } => {
                
                match cond.as_ref() {
                    Expr::Bool(true) => {
                        Self::emit_into(then_, ctx, ops);
                    }
                    Expr::Bool(false) => {
                        Self::emit_into(else_, ctx, ops);
                    }
                    _ => {
                        Self::emit_into(cond, ctx, ops);
                        let then_prog = Arc::new(Self::compile_sub(then_, ctx));
                        let else_prog = Arc::new(Self::compile_sub(else_, ctx));
                        ops.push(Opcode::IfElse {
                            then_: then_prog,
                            else_: else_prog,
                        });
                    }
                }
            }

            Expr::Try { body, default } => {
                
                
                match body.as_ref() {
                    Expr::Null => {
                        Self::emit_into(default, ctx, ops);
                    }
                    Expr::Bool(_) | Expr::Int(_) | Expr::Float(_) | Expr::Str(_) => {
                        Self::emit_into(body, ctx, ops);
                    }
                    _ => {
                        let body_prog = Arc::new(Self::compile_sub(body, ctx));
                        let default_prog = Arc::new(Self::compile_sub(default, ctx));
                        ops.push(Opcode::TryExpr {
                            body: body_prog,
                            default: default_prog,
                        });
                    }
                }
            }

            Expr::GlobalCall { name, args } => {
                
                
                let is_special = matches!(
                    name.as_str(),
                    "coalesce" | "chain" | "join" | "zip" | "zip_longest" | "product" | "range"
                );
                if !is_special && !args.is_empty() {
                    
                    let first = match &args[0] {
                        Arg::Pos(e) | Arg::Named(_, e) => e.clone(),
                    };
                    Self::emit_into(&first, ctx, ops);
                    let rest_args: Vec<Arg> = args.iter().skip(1).cloned().collect();
                    let sub_progs: Vec<Arc<Program>> = rest_args
                        .iter()
                        .map(|a| match a {
                            Arg::Pos(e) | Arg::Named(_, e) => Arc::new(Self::compile_sub(e, ctx)),
                        })
                        .collect();
                    let sub_kernels = classify_sub_kernels(&sub_progs);
                    let call = Arc::new(CompiledCall {
                        method: BuiltinMethod::from_name(name.as_str()),
                        name: Arc::from(name.as_str()),
                        sub_progs: sub_progs.into(),
                        sub_kernels,
                        orig_args: rest_args.into(),
                        demand_max_keep: None,
                    });
                    ops.push(Opcode::CallMethod(call));
                } else {
                    
                    let sub_progs: Vec<Arc<Program>> = args
                        .iter()
                        .map(|a| match a {
                            Arg::Pos(e) | Arg::Named(_, e) => Arc::new(Self::compile_sub(e, ctx)),
                        })
                        .collect();
                    let sub_kernels = classify_sub_kernels(&sub_progs);
                    let call = Arc::new(CompiledCall {
                        method: BuiltinMethod::Unknown,
                        name: Arc::from(name.as_str()),
                        sub_progs: sub_progs.into(),
                        sub_kernels,
                        orig_args: args.iter().cloned().collect::<Vec<_>>().into(),
                        demand_max_keep: None,
                    });
                    ops.push(Opcode::PushRoot);
                    ops.push(Opcode::CallMethod(call));
                }
            }

            Expr::Cast { expr, ty } => {
                Self::emit_into(expr, ctx, ops);
                ops.push(Opcode::CastOp(*ty));
            }

            Expr::Patch {
                root,
                ops: patch_ops,
            } => {
                
                
                let compiled = Self::compile_patch(root, patch_ops, ctx);
                ops.push(Opcode::PatchEval(Arc::new(compiled)));
            }

            Expr::DeleteMark => {


                ops.push(Opcode::DeleteMarkErr);
            }

            Expr::Match { scrutinee, arms } => {
                let compiled = compile_match(scrutinee, arms, ctx);
                ops.push(Opcode::Match(Arc::new(compiled)));
            }
        }
    }

    /// Emit a single chain `Step` as the corresponding opcode(s) into `ops`.
    fn emit_step(step: &Step, ctx: &VarCtx, ops: &mut Vec<Opcode>) {
        match step {
            Step::Field(name) => ops.push(Opcode::GetField(Arc::from(name.as_str()))),
            Step::OptField(name) => ops.push(Opcode::OptField(Arc::from(name.as_str()))),
            Step::Descendant(n) => ops.push(Opcode::Descendant(Arc::from(n.as_str()))),
            Step::DescendAll => ops.push(Opcode::DescendAll),
            Step::Index(i) => ops.push(Opcode::GetIndex(*i)),
            Step::DynIndex(e) => ops.push(Opcode::DynIndex(Arc::new(Self::compile_sub(e, ctx)))),
            Step::Slice(a, b) => ops.push(Opcode::GetSlice(*a, *b)),
            Step::Method(name, method_args) => {
                let call = Self::compile_call(name, method_args, ctx);
                ops.push(Opcode::CallMethod(Arc::new(call)));
            }
            Step::OptMethod(name, method_args) => {
                let call = Self::compile_call(name, method_args, ctx);
                ops.push(Opcode::CallOptMethod(Arc::new(call)));
            }
            Step::InlineFilter(pred) => {
                ops.push(Opcode::InlineFilter(Arc::new(Self::compile_sub(pred, ctx))));
            }
            Step::Quantifier(k) => ops.push(Opcode::Quantifier(*k)),
            Step::DeepMatch { arms, early_stop } => {
                // Both deep-match shapes share their compile path with
                // expression-level `match`; the receiver stack value
                // doubles as the descent root and the per-element
                // scrutinee. `MatchScrutinee::Current` tells the runtime
                // to read the iteration item from `@` rather than
                // re-evaluating a sub-program per descendant. The
                // `early_stop` flag selects between the collect-all and
                // first-match runtime opcodes.
                let scrutinee_marker = Expr::Current;
                let cm = compile_match(&scrutinee_marker, arms, ctx);
                let cm = Arc::new(cm);
                if *early_stop {
                    ops.push(Opcode::DeepMatchFirst(cm));
                } else {
                    ops.push(Opcode::DeepMatchAll(cm));
                }
            }
        }
    }

    /// Build a `CompiledCall` descriptor for a method invocation, pre-compiling
    /// each argument expression into a sub-program.
    fn compile_call(name: &str, args: &[Arg], ctx: &VarCtx) -> CompiledCall {
        let method = BuiltinMethod::from_name(name);
        let sub_progs: Vec<Arc<Program>> = args
            .iter()
            .map(|a| match a {
                Arg::Pos(e) | Arg::Named(_, e) => Arc::new(Self::compile_lambda_or_expr(e, ctx)),
            })
            .collect();
        let sub_kernels = classify_sub_kernels(&sub_progs);
        CompiledCall {
            method,
            name: Arc::from(name),
            sub_progs: sub_progs.into(),
            sub_kernels,
            orig_args: args.iter().cloned().collect::<Vec<_>>().into(),
            demand_max_keep: None,
        }
    }

    /// Compile a method argument that may be a lambda or a plain expression.
    ///
    /// For single-param lambdas, the parameter identifier is substituted with
    /// `Expr::Current` at the AST level — recursively, with shadowing-aware
    /// stop-conditions for nested lambdas, `let`, comprehensions, match-arm
    /// pattern bindings, and pipeline binds — before the body is emitted.
    /// The resulting program is identical to what the equivalent `@`-form
    /// would produce, which means kernel classification (`BodyKernel`) and
    /// every peephole pass apply uniformly. Multi-param lambdas keep their
    /// runtime-binding path: each param resolves through `env.get_var` after
    /// a `push_lam` at call time.
    fn compile_lambda_or_expr(expr: &Expr, ctx: &VarCtx) -> Program {
        match expr {
            Expr::Lambda { params, body } if params.len() == 1 => {
                let name = &params[0];
                let lowered = crate::compile::lambda_lower::substitute_current(
                    (**body).clone(),
                    name.as_str(),
                );
                let inner_ctx = ctx.with_vars(params);
                let prog = Self::compile_sub(&lowered, &inner_ctx);
                // If the substituted body still references the parameter
                // name (only possible from inside a nested lambda whose
                // own `@` is reassigned), wrap the program in a runtime
                // binding opcode so `LoadIdent(name)` resolves to the
                // outer iteration item even when the host pipeline stage
                // advances per row via `swap_current`.
                if crate::plan::analysis::expr_uses_ident(&lowered, name.as_str()) {
                    let body = Arc::new(prog);
                    let ops = vec![Opcode::BindLamCurrent {
                        name: Some(Arc::from(name.as_str())),
                        body,
                    }];
                    Program::new(ops, "<lam-body-bind>")
                } else {
                    prog
                }
            }
            Expr::Lambda { params, body } if params.len() >= 2 => {
                // Multi-param fast path: `apply_compiled_pair` pushes
                // params left-to-right via `push_lam`, so `env.current`
                // ends up bound to the rightmost argument. Substitute
                // that param's identifier with `Expr::Current` at the
                // AST level — the inner params 0..N-1 still resolve
                // through `LoadIdent` / `env.get_var`, but the rightmost
                // (the one read most heavily in comparators and pair
                // reducers per row's right operand) skips the var
                // lookup entirely.
                let last = params.last().expect("multi-param lambda");
                let lowered = crate::compile::lambda_lower::substitute_current(
                    (**body).clone(),
                    last.as_str(),
                );
                let inner_ctx = ctx.with_vars(params);
                Self::compile_sub(&lowered, &inner_ctx)
            }
            Expr::Lambda { params, body } => {
                let inner = ctx.with_vars(params);
                Self::compile_sub(body, &inner)
            }
            other => Self::compile_sub(other, ctx),
        }
    }

    /// Emit the appropriate opcode(s) for a binary operator, using short-circuit
    /// sub-programs for `&&`, `||`, and `??`.
    fn emit_binop(l: &Expr, op: BinOp, r: &Expr, ctx: &VarCtx, ops: &mut Vec<Opcode>) {
        match op {
            BinOp::And => {
                Self::emit_into(l, ctx, ops);
                let rhs_prog = Arc::new(Self::compile_sub(r, ctx));
                ops.push(Opcode::AndOp(rhs_prog));
            }
            BinOp::Or => {
                Self::emit_into(l, ctx, ops);
                let rhs_prog = Arc::new(Self::compile_sub(r, ctx));
                ops.push(Opcode::OrOp(rhs_prog));
            }
            BinOp::Add => {
                Self::emit_into(l, ctx, ops);
                Self::emit_into(r, ctx, ops);
                ops.push(Opcode::Add);
            }
            BinOp::Sub => {
                Self::emit_into(l, ctx, ops);
                Self::emit_into(r, ctx, ops);
                ops.push(Opcode::Sub);
            }
            BinOp::Mul => {
                Self::emit_into(l, ctx, ops);
                Self::emit_into(r, ctx, ops);
                ops.push(Opcode::Mul);
            }
            BinOp::Div => {
                Self::emit_into(l, ctx, ops);
                Self::emit_into(r, ctx, ops);
                ops.push(Opcode::Div);
            }
            BinOp::Mod => {
                Self::emit_into(l, ctx, ops);
                Self::emit_into(r, ctx, ops);
                ops.push(Opcode::Mod);
            }
            BinOp::Eq => {
                Self::emit_into(l, ctx, ops);
                Self::emit_into(r, ctx, ops);
                ops.push(Opcode::Eq);
            }
            BinOp::Neq => {
                Self::emit_into(l, ctx, ops);
                Self::emit_into(r, ctx, ops);
                ops.push(Opcode::Neq);
            }
            BinOp::Lt => {
                Self::emit_into(l, ctx, ops);
                Self::emit_into(r, ctx, ops);
                ops.push(Opcode::Lt);
            }
            BinOp::Lte => {
                Self::emit_into(l, ctx, ops);
                Self::emit_into(r, ctx, ops);
                ops.push(Opcode::Lte);
            }
            BinOp::Gt => {
                Self::emit_into(l, ctx, ops);
                Self::emit_into(r, ctx, ops);
                ops.push(Opcode::Gt);
            }
            BinOp::Gte => {
                Self::emit_into(l, ctx, ops);
                Self::emit_into(r, ctx, ops);
                ops.push(Opcode::Gte);
            }
            BinOp::Fuzzy => {
                Self::emit_into(l, ctx, ops);
                Self::emit_into(r, ctx, ops);
                ops.push(Opcode::Fuzzy);
            }
        }
    }

    /// Emit a `PipelineRun` opcode for a `base | step1 | step2 | …` expression,
    /// compiling each forward and bind step while threading the variable context.
    fn emit_pipeline(base: &Expr, steps: &[PipeStep], ctx: &VarCtx, ops: &mut Vec<Opcode>) {
        
        
        let base_prog = Arc::new(Self::compile_sub(base, ctx));
        let mut cur_ctx = ctx.clone();
        let mut compiled_steps: Vec<CompiledPipeStep> = Vec::with_capacity(steps.len());
        for step in steps {
            match step {
                PipeStep::Forward(rhs) => {
                    
                    
                    let mut sub_ops: Vec<Opcode> = Vec::new();
                    Self::emit_pipe_forward(rhs, &cur_ctx, &mut sub_ops);
                    
                    
                    if let Some(Opcode::SetCurrent) = sub_ops.first() {
                        sub_ops.remove(0);
                    }
                    let prog = Program::new(Self::optimize(sub_ops), "<pipe-fwd>");
                    compiled_steps.push(CompiledPipeStep::Forward(Arc::new(prog)));
                }
                PipeStep::Bind(target) => match target {
                    BindTarget::Name(name) => {
                        compiled_steps.push(CompiledPipeStep::BindName(Arc::from(name.as_str())));
                        cur_ctx = cur_ctx.with_var(name);
                    }
                    BindTarget::Obj { fields, rest } => {
                        let spec = BindObjSpec {
                            fields: fields
                                .iter()
                                .map(|f| Arc::from(f.as_str()))
                                .collect::<Vec<_>>()
                                .into(),
                            rest: rest.as_ref().map(|r| Arc::from(r.as_str())),
                        };
                        compiled_steps.push(CompiledPipeStep::BindObj(Arc::new(spec)));
                        for f in fields {
                            cur_ctx = cur_ctx.with_var(f);
                        }
                        if let Some(r) = rest {
                            cur_ctx = cur_ctx.with_var(r);
                        }
                    }
                    BindTarget::Arr(names) => {
                        let ns: Vec<Arc<str>> =
                            names.iter().map(|n| Arc::from(n.as_str())).collect();
                        compiled_steps.push(CompiledPipeStep::BindArr(ns.into()));
                        for n in names {
                            cur_ctx = cur_ctx.with_var(n);
                        }
                    }
                },
            }
        }
        ops.push(Opcode::PipelineRun {
            base: base_prog,
            steps: compiled_steps.into(),
        });
    }

    /// Emit the right-hand side of a pipe forward step. Bare identifiers and bare
    /// chains rooted at an unbound identifier are treated as zero-arg method calls on
    /// the current value; everything else inserts a `SetCurrent` marker first.
    fn emit_pipe_forward(rhs: &Expr, ctx: &VarCtx, ops: &mut Vec<Opcode>) {
        match rhs {
            Expr::Ident(name) if !ctx.has(name) => {
                
                
                let call = CompiledCall {
                    method: BuiltinMethod::from_name(name),
                    name: Arc::from(name.as_str()),
                    sub_progs: Arc::from(&[] as &[Arc<Program>]),
                    sub_kernels: Arc::from(&[] as &[crate::exec::pipeline::BodyKernel]),
                    orig_args: Arc::from(&[] as &[Arg]),
                    demand_max_keep: None,
                };
                ops.push(Opcode::PushCurrent);
                ops.push(Opcode::CallMethod(Arc::new(call)));
            }
            Expr::Chain(base, steps) if !steps.is_empty() => {
                if let Expr::Ident(name) = base.as_ref() {
                    if !ctx.has(name) {

                        let call = CompiledCall {
                            method: BuiltinMethod::from_name(name),
                            name: Arc::from(name.as_str()),
                            sub_progs: Arc::from(&[] as &[Arc<Program>]),
                            sub_kernels: Arc::from(
                                &[] as &[crate::exec::pipeline::BodyKernel],
                            ),
                            orig_args: Arc::from(&[] as &[Arg]),
                            demand_max_keep: None,
                        };
                        ops.push(Opcode::PushCurrent);
                        ops.push(Opcode::CallMethod(Arc::new(call)));
                        for step in steps {
                            Self::emit_step(step, ctx, ops);
                        }
                        return;
                    }
                }
                ops.push(Opcode::SetCurrent);
                Self::emit_into(rhs, ctx, ops);
            }
            _ => {
                
                ops.push(Opcode::SetCurrent);
                Self::emit_into(rhs, ctx, ops);
            }
        }
    }

    /// Compile a sub-expression (lambda body, arg, or nested expression) with full
    /// optimisation but a generic `"<sub>"` source label.
    fn compile_sub(expr: &Expr, ctx: &VarCtx) -> Program {
        let ops = Self::optimize(Self::emit(expr, ctx));
        Program::new(ops, "<sub>")
    }

    /// Compile a `patch` expression into a `CompiledPatch` by lowering each AST
    /// `PatchOp` to its compiled path steps, value action, and optional condition.
    fn compile_patch(
        root: &Expr,
        patch_ops: &[crate::parse::ast::PatchOp],
        ctx: &VarCtx,
    ) -> CompiledPatch {
        let root_prog = Arc::new(Self::compile_sub(root, ctx));
        let mut ops = Vec::with_capacity(patch_ops.len());
        for po in patch_ops {
            let path: Vec<CompiledPathStep> = po
                .path
                .iter()
                .map(|s| match s {
                    crate::parse::ast::PathStep::Field(n) => {
                        CompiledPathStep::Field(Arc::from(n.as_str()))
                    }
                    crate::parse::ast::PathStep::Index(i) => CompiledPathStep::Index(*i),
                    crate::parse::ast::PathStep::DynIndex(e) => {
                        CompiledPathStep::DynIndex(Arc::new(Self::compile_sub(e, ctx)))
                    }
                    crate::parse::ast::PathStep::Wildcard => CompiledPathStep::Wildcard,
                    crate::parse::ast::PathStep::WildcardFilter(p) => {
                        CompiledPathStep::WildcardFilter(Arc::new(Self::compile_sub(p, ctx)))
                    }
                    crate::parse::ast::PathStep::Descendant(n) => {
                        CompiledPathStep::Descendant(Arc::from(n.as_str()))
                    }
                })
                .collect();
            let val = if matches!(&po.val, Expr::DeleteMark) {
                CompiledPatchVal::Delete
            } else {
                CompiledPatchVal::Replace(Arc::new(Self::compile_sub(&po.val, ctx)))
            };
            let cond = po
                .cond
                .as_ref()
                .map(|c| Arc::new(Self::compile_sub(c, ctx)));
            ops.push(CompiledPatchOp { path, val, cond });
        }
        CompiledPatch {
            root_prog,
            ops,
            trie: std::sync::OnceLock::new(),
        }
    }

    /// Try to lower an `Expr` rooted at `@` into a sequence of `KvStep`s.
    /// Returns `None` if the expression contains anything other than field/index steps.
    fn try_kv_path_steps(expr: &Expr) -> Option<Vec<KvStep>> {
        use crate::parse::ast::Step;
        let (base, steps) = match expr {
            Expr::Chain(b, s) => (&**b, s.as_slice()),
            _ => return None,
        };
        if !matches!(base, Expr::Current) {
            return None;
        }
        if steps.is_empty() {
            return None;
        }
        let mut out = Vec::with_capacity(steps.len());
        for s in steps {
            match s {
                Step::Field(name) => out.push(KvStep::Field(Arc::from(name.as_str()))),
                Step::Index(i) => out.push(KvStep::Index(*i)),
                _ => return None,
            }
        }
        Some(out)
    }
}

/// Per-flag configuration controlling which peephole passes the `Compiler` runs.
/// All flags default to `true`; individual flags can be disabled for testing or profiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PassConfig {
    /// Enable `PushRoot + GetField…` → `RootChain` fusion.
    pub root_chain: bool,
    /// Enable consecutive `GetField`/`OptField` → `FieldChain` fusion.
    pub field_chain: bool,
    /// Enable `FilterCount` fusion (reserved; currently unused at runtime).
    pub filter_count: bool,
    /// Enable field/filter specialisation pass.
    pub filter_fusion: bool,
    /// Enable find-quantifier optimisation pass (reserved for future use).
    pub find_quantifier: bool,
    /// Enable strength-reduction (e.g. `sort()[0]` → `min()`).
    pub strength_reduce: bool,
    /// Enable removal of provably redundant adjacent opcodes.
    pub redundant_ops: bool,
    /// Enable constant folding of `KindCheck` against known-type literals.
    pub kind_check_fold: bool,
    /// Enable constant folding of no-arg method calls on literal operands.
    pub method_const: bool,
    /// Enable general constant folding of arithmetic and comparison operators.
    pub const_fold: bool,
    /// Enable `OptField`-to-`GetField` promotion when the receiver is non-null.
    pub nullness: bool,
    /// Enable reordering of `&&` operands by selectivity.
    pub reorder_and: bool,
    /// Enable sub-program deduplication to share identical `Arc<Program>` instances.
    pub dedup_subprogs: bool,
}

/// All passes enabled — the configuration used in production.
impl Default for PassConfig {
    fn default() -> Self {
        Self {
            root_chain: true,
            field_chain: true,
            filter_count: true,
            filter_fusion: true,
            find_quantifier: true,
            strength_reduce: true,
            redundant_ops: true,
            kind_check_fold: true,
            method_const: true,
            const_fold: true,
            nullness: true,
            reorder_and: true,
            dedup_subprogs: true,
        }
    }
}

impl PassConfig {
    /// Return a `PassConfig` with all passes disabled; useful in tests that
    /// need to inspect unoptimised opcode sequences.
    #[cfg(test)]
    pub fn none() -> Self {
        Self {
            root_chain: false,
            field_chain: false,
            filter_count: false,
            filter_fusion: false,
            find_quantifier: false,
            strength_reduce: false,
            redundant_ops: false,
            kind_check_fold: false,
            method_const: false,
            const_fold: false,
            nullness: false,
            reorder_and: false,
            dedup_subprogs: false,
        }
    }

    /// Encode all pass flags as a single `u64` bitmask; used as part of the compile-cache key
    /// so programs compiled under different configs are stored separately.
    pub fn hash(&self) -> u64 {
        let mut bits: u64 = 0;
        for (i, b) in [
            self.root_chain,
            self.field_chain,
            self.filter_count,
            self.filter_fusion,
            self.find_quantifier,
            self.strength_reduce,
            self.redundant_ops,
            self.kind_check_fold,
            self.method_const,
            self.const_fold,
            self.nullness,
            self.reorder_and,
            self.dedup_subprogs,
        ]
        .iter()
        .enumerate()
        {
            if *b {
                bits |= 1u64 << i;
            }
        }
        bits
    }
}

/// Lower a `match scrutinee with { arms }` AST node into a `CompiledMatch`
/// suitable for the flat decision-machine interpreter in `vm::exec`. Each
/// arm contributes a contiguous block of `MatchOp`s framed by `ResetArm`,
/// pattern tests, optional guard, and a terminating `Body`. Failed tests
/// jump forward to the start of the next arm; the trailing `Fail` op is
/// reached only when no arm matches.
pub(crate) fn compile_match(
    scrutinee: &Expr,
    arms: &[crate::parse::ast::MatchArm],
    ctx: &VarCtx,
) -> CompiledMatch {
    use crate::parse::ast::Pat;

    let mut b = MatchBuilder::default();
    // Reserve a placeholder for each arm-start PC so backward jumps from a
    // miss inside arm N go to arm N+1's `ResetArm`. Final fall-through PC
    // is set to the position of the trailing `Fail` op.
    let mut arm_starts: Vec<u32> = Vec::with_capacity(arms.len());

    // Cross-arm prefix sharing recognises a *contiguous leading run* of
    // arms that agree on shape (all `Pat::Obj` with shared keys, all
    // `Pat::Arr` with same length, or all `Pat::Kind` of the same kind).
    // The shared check is hoisted into a single match-level prelude;
    // arms in the leading run use a compressed codegen that skips the
    // hoisted ops, while later arms (mixed shapes, catch-alls, etc.)
    // fall through to standard codegen. A failed prelude check jumps to
    // the first non-shared arm — or to the global `Fail` op when the
    // entire arm list participates in sharing.
    let (shared_keys, shared_arms_n) = detect_shared_key_prefix(arms);
    let (shared_arr_len, shared_arms_n) = if shared_keys.is_empty() {
        detect_shared_arr_len(arms)
            .map(|(len, n)| (Some(len), n))
            .unwrap_or((None, 0))
    } else {
        (None, shared_arms_n)
    };
    let (shared_kind, shared_arms_n) = if shared_keys.is_empty() && shared_arr_len.is_none() {
        detect_shared_kind(arms)
            .map(|(kind, n)| (Some(kind), n))
            .unwrap_or((None, 0))
    } else {
        (None, shared_arms_n)
    };
    let mut prelude_pending: Vec<u32> = Vec::new();
    if !shared_keys.is_empty() {
        // Object prelude: `ObjCheck` plus one `LoadField` per shared key.
        let p_check = b.emit(MatchOp::ObjCheck {
            slot: 0,
            else_pc: u32::MAX,
        });
        prelude_pending.push(p_check);
        for (i, key) in shared_keys.iter().enumerate() {
            let p_load = b.emit(MatchOp::LoadField {
                src: 0,
                key: Arc::clone(key),
                dst: (i + 1) as u16,
                else_pc: u32::MAX,
            });
            prelude_pending.push(p_load);
        }
    } else if let Some(len) = shared_arr_len {
        // Array prelude: a single `LenCheck` against the agreed length.
        let p_len = b.emit(MatchOp::LenCheck {
            slot: 0,
            len,
            exact: true,
            else_pc: u32::MAX,
        });
        prelude_pending.push(p_len);
    } else if let Some(kind) = shared_kind {
        // Kind prelude: every arm tests the same scalar kind. Hoist the
        // single `KindCheck` so per-arm prologue only emits the binding.
        let p_kind = b.emit(MatchOp::KindCheck {
            slot: 0,
            kind,
            else_pc: u32::MAX,
        });
        prelude_pending.push(p_kind);
    }
    let keep_above: u16 = if !shared_keys.is_empty() {
        (shared_keys.len() + 1) as u16
    } else {
        1
    };

    for (arm_idx, arm) in arms.iter().enumerate() {
        // Open a fresh patch-bucket for this arm so its `else_pc`
        // placeholders are tracked independently of earlier arms.
        b.pending_else.push(Vec::new());
        arm_starts.push(b.next_pc());
        // Per-arm reset preserves the prelude-allocated slots only for
        // arms that are part of the shared run; arms outside the shared
        // run get the default `keep_above = 1` reset so their reads of
        // the shared slots return null and routinely fall through.
        let arm_in_share = arm_idx < shared_arms_n;
        let arm_keep = if arm_in_share { keep_above } else { 1 };
        let reset_idx = b.emit(MatchOp::ResetArm {
            slots: arm_keep,
            keep_above: arm_keep,
        });

        // Slot space: when sharing applies to this arm, slot 1.. already
        // hold the prelude's loaded values, so per-arm allocation begins
        // above the prelude region.
        let mut slot_alloc = SlotAlloc::starting_at(arm_keep);

        if arm_in_share && !shared_keys.is_empty() {
            // Object prefix sharing: every arm is `Pat::Obj` whose first
            // `shared_keys.len()` fields agree on key order. Compile
            // those sub-patterns against prelude-allocated slots; let
            // any extra fields go through the standard load-field path.
            let Pat::Obj { fields, rest: _ } = &arm.pat else {
                unreachable!("shared-prefix invariant: every arm is Pat::Obj");
            };
            for (i, key) in shared_keys.iter().enumerate() {
                debug_assert_eq!(fields[i].0.as_str(), key.as_ref());
                b.compile_pat_for_arm(&fields[i].1, (i + 1) as u16, &mut slot_alloc);
            }
            for (k, sub_pat) in fields.iter().skip(shared_keys.len()) {
                let dst = slot_alloc.alloc();
                b.emit_with_pending_else(MatchOp::LoadField {
                    src: 0,
                    key: Arc::from(k.as_str()),
                    dst,
                    else_pc: u32::MAX,
                });
                b.compile_pat_for_arm(sub_pat, dst, &mut slot_alloc);
            }
        } else if arm_in_share && shared_kind.is_some() {
            // Kind sharing: every arm is `Pat::Kind` testing the same
            // scalar kind. The prelude has already verified the kind, so
            // per-arm codegen only needs to push the optional binding.
            let Pat::Kind { name, kind: _ } = &arm.pat else {
                unreachable!("shared-kind invariant: every arm is Pat::Kind");
            };
            if let Some(n) = name.as_deref() {
                b.emit(MatchOp::Bind {
                    name: Arc::from(n),
                    slot: 0,
                });
            }
        } else if arm_in_share && shared_arr_len.is_some() {
            // Array length sharing: every arm is `Pat::Arr` with the
            // same exact length and no rest binding. The prelude has
            // already verified the length, so per-arm codegen skips
            // `LenCheck` and goes straight to per-element loads.
            let Pat::Arr { elems, rest: _ } = &arm.pat else {
                unreachable!("shared-arr invariant: every arm is Pat::Arr");
            };
            for (i, sub_pat) in elems.iter().enumerate() {
                let dst = slot_alloc.alloc();
                b.emit(MatchOp::LoadIndex {
                    src: 0,
                    idx: i as u32,
                    dst,
                });
                b.compile_pat_for_arm(sub_pat, dst, &mut slot_alloc);
            }
        } else {
            // Standard codegen path: full pattern compile against slot 0.
            b.compile_pat_for_arm(&arm.pat, 0, &mut slot_alloc);
        }

        // Guard (optional).
        if let Some(guard_expr) = arm.guard.as_ref() {
            let prog = Arc::new(Compiler::compile_sub(guard_expr, ctx));
            let prog_idx = b.add_guard(prog);
            b.emit_with_pending_else(MatchOp::Guard {
                prog: prog_idx,
                else_pc: u32::MAX,
            });
        }

        // Body terminator: bindings already pushed; run body program.
        let body_prog = Arc::new(Compiler::compile_sub(&arm.body, ctx));
        let body_idx = b.add_body(body_prog);
        b.emit(MatchOp::Body { prog: body_idx });

        // Patch arm-local slot count into the reset op.
        b.patch_reset_slots(reset_idx, slot_alloc.high_water_mark());
    }

    let fail_pc = b.next_pc();
    b.emit(MatchOp::Fail);

    // Patch the shared-prelude placeholders (object check, key loads,
    // length check, kind check) to the start of the first arm that did
    // not participate in sharing. When every arm participates the miss
    // path is the trailing `Fail` op.
    let prelude_miss_target: u32 = if shared_arms_n < arms.len() {
        arm_starts[shared_arms_n]
    } else {
        fail_pc
    };
    for pc in prelude_pending {
        b.patch_else_pc(pc, prelude_miss_target);
    }

    // Patch all `else_pc` placeholders to next-arm-start (or `fail_pc` for
    // the last arm).
    for (i, miss_pcs) in std::mem::take(&mut b.pending_else).into_iter().enumerate() {
        let target = arm_starts.get(i + 1).copied().unwrap_or(fail_pc);
        for pc in miss_pcs {
            b.patch_else_pc(pc, target);
        }
    }

    // Compute the maximum slot count the runtime ever needs, so the VM
    // can size its slot vector once instead of growing per arm. The walk
    // inspects each `ResetArm.slots` and any explicit `dst` slot — slots
    // are dense within a single match expression.
    let max_slots = compute_max_slots(&b.ops);

    CompiledMatch {
        scrutinee: classify_match_scrutinee(scrutinee, ctx),
        ops: Arc::from(b.ops),
        lits: Arc::from(b.lits),
        guards: Arc::from(b.guards),
        bodies: Arc::from(b.bodies),
        subpats: Arc::from(b.subpats),
        max_slots,
        is_exhaustive: arms.iter().any(|a| {
            a.guard.is_none() && matches!(a.pat, Pat::Wild | Pat::Bind(_))
        }),
        shape_summary: derive_shape_summary(arms),
    }
}

/// Inspect the typed (non-catch-all) arms of a `match` and return a
/// structural summary the deep-match runtime can use to pre-filter
/// candidates via a bitmap index. Returns `None` when arms are
/// heterogeneous or the leading shape cannot be characterised by one
/// of the recognised summaries.
fn derive_shape_summary(
    arms: &[crate::parse::ast::MatchArm],
) -> Option<crate::vm::MatchShapeSummary> {
    use crate::parse::ast::Pat;

    // Strip a trailing catch-all (`_` / unbound bind) from consideration —
    // it does not constrain the candidate set; the typed arms above it
    // determine the summary.
    let typed: &[crate::parse::ast::MatchArm] = match arms.last() {
        Some(a) if matches!(a.pat, Pat::Wild | Pat::Bind(_)) => &arms[..arms.len() - 1],
        _ => arms,
    };
    if typed.is_empty() {
        return None;
    }

    // Object-of-keys summary — every typed arm is a `Pat::Obj`. Collect
    // the union of leading keys.
    if typed.iter().all(|a| matches!(a.pat, Pat::Obj { .. })) {
        let mut keys: Vec<Arc<str>> = Vec::new();
        for arm in typed {
            if let Pat::Obj { fields, .. } = &arm.pat {
                for (k, _) in fields {
                    let arc: Arc<str> = Arc::from(k.as_str());
                    if !keys.iter().any(|existing| existing.as_ref() == k.as_str()) {
                        keys.push(arc);
                    }
                }
            }
        }
        if !keys.is_empty() {
            return Some(crate::vm::MatchShapeSummary::ObjAnyOfKeys(Arc::from(keys)));
        }
    }

    // Kind-only summary — every typed arm is `Pat::Kind` of the same
    // scalar kind.
    if let Pat::Kind {
        kind: first_kind, ..
    } = &typed[0].pat
    {
        if typed.iter().all(|a| match &a.pat {
            Pat::Kind { kind, .. } => kind == first_kind,
            _ => false,
        }) {
            return Some(crate::vm::MatchShapeSummary::KindOnly(*first_kind));
        }
    }

    // Numeric-range summary — every typed arm is a `Pat::Range`. Take
    // the union (min lo, max hi).
    if typed.iter().all(|a| matches!(a.pat, Pat::Range { .. })) {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        let mut inclusive = false;
        for arm in typed {
            if let Pat::Range {
                lo: l,
                hi: h,
                inclusive: inc,
            } = &arm.pat
            {
                if *l < lo {
                    lo = *l;
                }
                if *h > hi {
                    hi = *h;
                }
                if *inc {
                    inclusive = true;
                }
            }
        }
        if lo.is_finite() && hi.is_finite() && lo <= hi {
            return Some(crate::vm::MatchShapeSummary::NumericRange { lo, hi, inclusive });
        }
    }

    None
}

/// Pick the cheapest strategy for evaluating a `match` scrutinee at run
/// time. `match @ with { ... }` and `match $ with { ... }` skip VM
/// re-entry entirely by reading directly from `Env::current` /
/// `Env::root`; everything else compiles a sub-program.
fn classify_match_scrutinee(expr: &Expr, ctx: &VarCtx) -> crate::vm::MatchScrutinee {
    match expr {
        Expr::Current => crate::vm::MatchScrutinee::Current,
        Expr::Root => crate::vm::MatchScrutinee::Root,
        _ => crate::vm::MatchScrutinee::Program(Arc::new(Compiler::compile_sub(expr, ctx))),
    }
}

/// Walk a freshly-emitted op stream and return the largest slot index
/// referenced plus one. Used to size the runtime slot vector once at the
/// top of `exec_match` so prelude ops (which run before any `ResetArm`)
/// can index any slot up to `max_slots - 1`.
fn compute_max_slots(ops: &[MatchOp]) -> u16 {
    let mut hi: u16 = 1;
    for op in ops {
        match op {
            MatchOp::ResetArm { slots, .. } => {
                if *slots > hi {
                    hi = *slots;
                }
            }
            MatchOp::KindCheck { slot, .. }
            | MatchOp::LitEq { slot, .. }
            | MatchOp::RangeCheck { slot, .. }
            | MatchOp::ObjCheck { slot, .. }
            | MatchOp::LenCheck { slot, .. }
            | MatchOp::TestSubPat { slot, .. }
            | MatchOp::Bind { slot, .. } => {
                let n = slot.saturating_add(1);
                if n > hi {
                    hi = n;
                }
            }
            MatchOp::LoadField { src, dst, .. } => {
                let n = src.max(dst).saturating_add(1);
                if n > hi {
                    hi = n;
                }
            }
            MatchOp::LoadIndex { src, dst, .. } | MatchOp::LoadTail { src, dst, .. } => {
                let n = src.max(dst).saturating_add(1);
                if n > hi {
                    hi = n;
                }
            }
            MatchOp::LoadObjRest { src, dst, .. } => {
                let n = src.max(dst).saturating_add(1);
                if n > hi {
                    hi = n;
                }
            }
            MatchOp::Guard { .. }
            | MatchOp::Body { .. }
            | MatchOp::Fail
            | MatchOp::Jump { .. } => {}
        }
    }
    hi
}

/// Detect a contiguous leading run of `Pat::Kind` arms that all test
/// the same scalar kind. Returns `(kind, n)` where `n` is the number of
/// participating arms; arms `arms[n..]` fall through to standard
/// codegen. Like the other shared-prefix detectors, requires at least
/// two participating arms.
fn detect_shared_kind(
    arms: &[crate::parse::ast::MatchArm],
) -> Option<(crate::parse::ast::KindType, usize)> {
    use crate::parse::ast::Pat;

    if arms.len() < 2 {
        return None;
    }
    let Pat::Kind { kind: first, .. } = &arms[0].pat else {
        return None;
    };
    let shared = *first;
    let mut count = 1usize;
    for arm in &arms[1..] {
        match &arm.pat {
            Pat::Kind { kind, .. } if *kind == shared => count += 1,
            _ => break,
        }
    }
    if count < 2 {
        None
    } else {
        Some((shared, count))
    }
}

/// Detect a contiguous leading run of fixed-length `Pat::Arr` arms that
/// all agree on length. Returns `(len, n)` where `n` is the number of
/// participating arms; arms `arms[n..]` fall through to standard
/// codegen. Rest bindings disable participation because they imply a
/// `>=` length test rather than an exact one.
fn detect_shared_arr_len(arms: &[crate::parse::ast::MatchArm]) -> Option<(u32, usize)> {
    use crate::parse::ast::Pat;

    if arms.len() < 2 {
        return None;
    }
    let Pat::Arr {
        elems: first_elems,
        rest: first_rest,
    } = &arms[0].pat
    else {
        return None;
    };
    if first_rest.is_some() {
        return None;
    }
    let shared = first_elems.len() as u32;
    let mut count = 1usize;
    for arm in &arms[1..] {
        match &arm.pat {
            Pat::Arr { elems, rest } if rest.is_none() && elems.len() as u32 == shared => {
                count += 1;
            }
            _ => break,
        }
    }
    if count < 2 {
        None
    } else {
        Some((shared, count))
    }
}

/// Detect the longest leading key sequence shared by a contiguous run
/// of `Pat::Obj` arms at the start of `arms`. The returned tuple is
/// `(shared_keys, n)` where `n` is the number of leading arms that
/// participate in the shared prefix; arms `arms[n..]` fall through to
/// standard codegen. The optimization requires at least two participating
/// arms to be worth its prelude cost.
fn detect_shared_key_prefix(arms: &[crate::parse::ast::MatchArm]) -> (Vec<Arc<str>>, usize) {
    use crate::parse::ast::Pat;

    if arms.len() < 2 {
        return (Vec::new(), 0);
    }

    // Seed the shared prefix with the first arm's full key list.
    let Pat::Obj {
        fields: first_fields,
        ..
    } = &arms[0].pat
    else {
        return (Vec::new(), 0);
    };
    let mut prefix: Vec<&str> = first_fields.iter().map(|(k, _)| k.as_str()).collect();
    if prefix.is_empty() {
        return (Vec::new(), 0);
    }

    // Walk subsequent arms, intersecting their leading key list with
    // the running prefix. Stop at the first arm whose pattern is not
    // `Pat::Obj` or whose first key does not match — every arm that
    // contributes must keep at least one key in the shared prefix.
    let mut count = 1usize;
    for arm in &arms[1..] {
        let Pat::Obj { fields, .. } = &arm.pat else {
            break;
        };
        let common = prefix
            .iter()
            .zip(fields.iter().map(|(k, _)| k.as_str()))
            .take_while(|(a, b)| **a == *b)
            .count();
        if common == 0 {
            break;
        }
        prefix.truncate(common);
        count += 1;
    }

    if count < 2 || prefix.is_empty() {
        return (Vec::new(), 0);
    }

    (prefix.into_iter().map(Arc::from).collect(), count)
}

/// Mutable state used while emitting a `CompiledMatch`. Tracks the op
/// stream, sub-program pools, the literal pool, and the per-arm
/// `else_pc` placeholders that need patching once arm-start PCs are known.
#[derive(Default)]
struct MatchBuilder {
    ops: Vec<MatchOp>,
    lits: Vec<crate::parse::ast::PatLit>,
    guards: Vec<Arc<Program>>,
    bodies: Vec<Arc<Program>>,
    subpats: Vec<crate::parse::ast::Pat>,
    /// One entry per arm; each holds the op indices whose `else_pc` field
    /// must be patched to point to the next arm's start.
    pending_else: Vec<Vec<u32>>,
}

impl MatchBuilder {
    /// PC of the next op to be emitted.
    fn next_pc(&self) -> u32 {
        self.ops.len() as u32
    }

    /// Append an op without registering it for an arm-relative patch.
    fn emit(&mut self, op: MatchOp) -> u32 {
        let idx = self.ops.len() as u32;
        self.ops.push(op);
        idx
    }

    /// Append an op whose `else_pc` field is a sentinel; record its index
    /// in the current arm's patch bucket so it can be resolved to the
    /// next-arm start once the arm has finished emitting.
    fn emit_with_pending_else(&mut self, op: MatchOp) -> u32 {
        let idx = self.emit(op);
        let bucket = self
            .pending_else
            .last_mut()
            .expect("emit_with_pending_else called outside an arm");
        bucket.push(idx);
        idx
    }

    /// Append a literal to the pool, returning its index.
    fn add_lit(&mut self, lit: crate::parse::ast::PatLit) -> u16 {
        let idx = self.lits.len() as u16;
        self.lits.push(lit);
        idx
    }

    /// Append a sub-pattern (e.g. `Or`) to the pool, returning its index.
    fn add_subpat(&mut self, pat: crate::parse::ast::Pat) -> u16 {
        let idx = self.subpats.len() as u16;
        self.subpats.push(pat);
        idx
    }

    /// Append a guard sub-program, returning its index.
    fn add_guard(&mut self, prog: Arc<Program>) -> u16 {
        let idx = self.guards.len() as u16;
        self.guards.push(prog);
        idx
    }

    /// Append a body sub-program, returning its index.
    fn add_body(&mut self, prog: Arc<Program>) -> u16 {
        let idx = self.bodies.len() as u16;
        self.bodies.push(prog);
        idx
    }

    /// Patch the slot-count field of a previously emitted `ResetArm`.
    fn patch_reset_slots(&mut self, reset_idx: u32, slots: u16) {
        if let MatchOp::ResetArm { slots: s, .. } = &mut self.ops[reset_idx as usize] {
            *s = slots;
        }
    }

    /// Patch the `else_pc` field of a previously emitted miss-jumping op.
    fn patch_else_pc(&mut self, idx: u32, target: u32) {
        match &mut self.ops[idx as usize] {
            MatchOp::KindCheck { else_pc, .. }
            | MatchOp::LitEq { else_pc, .. }
            | MatchOp::RangeCheck { else_pc, .. }
            | MatchOp::ObjCheck { else_pc, .. }
            | MatchOp::LoadField { else_pc, .. }
            | MatchOp::LenCheck { else_pc, .. }
            | MatchOp::TestSubPat { else_pc, .. }
            | MatchOp::Guard { else_pc, .. } => *else_pc = target,
            other => panic!("patch_else_pc on op without else_pc: {other:?}"),
        }
    }

    /// Compile a single arm's pattern by emitting the appropriate test /
    /// load / bind ops against `slot`, allocating sub-slots for nested
    /// projections from `slot_alloc`.
    fn compile_pat_for_arm(
        &mut self,
        pat: &crate::parse::ast::Pat,
        slot: MatchSlot,
        slot_alloc: &mut SlotAlloc,
    ) {
        use crate::parse::ast::Pat;
        match pat {
            Pat::Wild => {
                // Always matches; emit nothing.
            }
            Pat::Bind(name) => {
                self.emit(MatchOp::Bind {
                    name: Arc::from(name.as_str()),
                    slot,
                });
            }
            Pat::Lit(lit) => {
                let lit_idx = self.add_lit(lit.clone());
                self.emit_with_pending_else(MatchOp::LitEq {
                    slot,
                    lit: lit_idx,
                    else_pc: u32::MAX,
                });
            }
            Pat::Range { lo, hi, inclusive } => {
                self.emit_with_pending_else(MatchOp::RangeCheck {
                    slot,
                    lo: *lo,
                    hi: *hi,
                    inclusive: *inclusive,
                    else_pc: u32::MAX,
                });
            }
            Pat::Kind { name, kind } => {
                self.emit_with_pending_else(MatchOp::KindCheck {
                    slot,
                    kind: *kind,
                    else_pc: u32::MAX,
                });
                if let Some(n) = name.as_deref() {
                    self.emit(MatchOp::Bind {
                        name: Arc::from(n),
                        slot,
                    });
                }
            }
            Pat::Or(alts) if alts.iter().all(|a| matches!(a, Pat::Lit(_))) => {
                // Or-of-literals cascade: emit one `LitEq` per alt with a
                // forward miss-jump to the next alt's PC, plus an
                // unconditional jump after each successful test that
                // skips past every remaining alternative. The last alt's
                // miss jumps to the arm boundary like any other test.
                //
                // Layout for `a | b | c`:
                //   pc0:  LitEq a, else_pc=pc2
                //   pc1:  Jump exit
                //   pc2:  LitEq b, else_pc=pc4
                //   pc3:  Jump exit
                //   pc4:  LitEq c, else_pc=ARM_MISS
                //   exit: ...
                let mut miss_to_patch: Vec<(u32, usize)> = Vec::new();
                let mut jump_to_patch: Vec<u32> = Vec::new();
                for (i, alt) in alts.iter().enumerate() {
                    let Pat::Lit(lit) = alt else {
                        unreachable!("guarded above")
                    };
                    let lit_idx = self.add_lit(lit.clone());
                    let is_last = i + 1 == alts.len();
                    let cmp_pc = if is_last {
                        // Final alt: miss exits to arm boundary via the
                        // standard pending_else patch path.
                        self.emit_with_pending_else(MatchOp::LitEq {
                            slot,
                            lit: lit_idx,
                            else_pc: u32::MAX,
                        })
                    } else {
                        let pc = self.emit(MatchOp::LitEq {
                            slot,
                            lit: lit_idx,
                            else_pc: u32::MAX,
                        });
                        // Unconditional jump-to-exit on success.
                        let jpc = self.emit(MatchOp::Jump {
                            target_pc: u32::MAX,
                        });
                        miss_to_patch.push((pc, i + 1));
                        jump_to_patch.push(jpc);
                        pc
                    };
                    let _ = cmp_pc;
                }
                let exit_pc = self.next_pc();
                // Patch each non-final miss to the start of the next alt.
                let alt_starts: Vec<u32> = (0..alts.len())
                    .map(|i| {
                        // Each alt occupies 1 op (LitEq) + 1 op (Jump)
                        // except the final which has no Jump. Compute by
                        // recovering from `miss_to_patch` ordering: the
                        // alts were emitted contiguously, and we recorded
                        // each cmp_pc.
                        miss_to_patch
                            .iter()
                            .find(|(_, idx)| *idx == i)
                            .map(|(_pc, _)| {
                                // alt i (i>=1) starts immediately after
                                // alt (i-1)'s Jump op.
                                miss_to_patch[i - 1].0 + 2
                            })
                            .unwrap_or_else(|| miss_to_patch.first().map(|(p, _)| *p).unwrap_or(0))
                    })
                    .collect();
                for (cmp_pc, alt_idx) in &miss_to_patch {
                    let target = alt_starts[*alt_idx];
                    self.patch_else_pc(*cmp_pc, target);
                }
                // Patch every success-jump to the cascade exit.
                for jpc in jump_to_patch {
                    if let MatchOp::Jump { target_pc } = &mut self.ops[jpc as usize] {
                        *target_pc = exit_pc;
                    }
                }
            }
            Pat::Or(_) => {
                // Or-patterns with binders or nested shapes retain their
                // tree shape so the runtime helper can backtrack bindings
                // between alternatives.
                let subpat_idx = self.add_subpat(pat.clone());
                self.emit_with_pending_else(MatchOp::TestSubPat {
                    slot,
                    subpat: subpat_idx,
                    else_pc: u32::MAX,
                });
            }
            Pat::Obj { fields, rest } => {
                self.emit_with_pending_else(MatchOp::ObjCheck {
                    slot,
                    else_pc: u32::MAX,
                });
                for (key, sub_pat) in fields {
                    let dst = slot_alloc.alloc();
                    self.emit_with_pending_else(MatchOp::LoadField {
                        src: slot,
                        key: Arc::from(key.as_str()),
                        dst,
                        else_pc: u32::MAX,
                    });
                    self.compile_pat_for_arm(sub_pat, dst, slot_alloc);
                }
                // A named `...rest` rest binding captures every key not
                // listed in `fields` into a freshly built `Val::Obj`.
                // Anonymous `...` is recorded in the AST but emits no
                // ops because the runtime already admits extra keys.
                if let Some(Some(rest_name)) = rest {
                    let listed_keys: Arc<[Arc<str>]> =
                        fields.iter().map(|(k, _)| Arc::from(k.as_str())).collect();
                    let dst = slot_alloc.alloc();
                    self.emit(MatchOp::LoadObjRest {
                        src: slot,
                        listed_keys,
                        dst,
                    });
                    self.emit(MatchOp::Bind {
                        name: Arc::from(rest_name.as_str()),
                        slot: dst,
                    });
                }
            }
            Pat::Arr { elems, rest } => {
                let prefix = elems.len() as u32;
                let exact = rest.is_none();
                self.emit_with_pending_else(MatchOp::LenCheck {
                    slot,
                    len: prefix,
                    exact,
                    else_pc: u32::MAX,
                });
                for (i, sub_pat) in elems.iter().enumerate() {
                    let dst = slot_alloc.alloc();
                    self.emit(MatchOp::LoadIndex {
                        src: slot,
                        idx: i as u32,
                        dst,
                    });
                    self.compile_pat_for_arm(sub_pat, dst, slot_alloc);
                }
                if let Some(rest_name) = rest.as_ref().and_then(|n| n.as_deref()) {
                    let dst = slot_alloc.alloc();
                    self.emit(MatchOp::LoadTail {
                        src: slot,
                        from: prefix,
                        dst,
                    });
                    self.emit(MatchOp::Bind {
                        name: Arc::from(rest_name),
                        slot: dst,
                    });
                }
            }
        }
    }
}

/// Per-arm slot index allocator. Slot 0 is reserved for the scrutinee, so
/// allocation begins at 1. The high-water mark drives `ResetArm.slots`.
struct SlotAlloc {
    next: u16,
}

impl SlotAlloc {
    /// Create a fresh allocator that will hand out slots starting at index 1.
    #[allow(dead_code)]
    fn new() -> Self {
        Self::starting_at(1)
    }

    /// Create an allocator whose next slot index is `start`. Used when a
    /// match-level prelude has already populated some slots that should
    /// not collide with arm-local sub-projections.
    fn starting_at(start: u16) -> Self {
        Self { next: start.max(1) }
    }

    /// Allocate and return the next slot index.
    fn alloc(&mut self) -> MatchSlot {
        let s = self.next;
        self.next = self
            .next
            .checked_add(1)
            .expect("match arm exceeded u16 slot capacity");
        s
    }

    /// Return the total number of slots required so far (including slot 0).
    fn high_water_mark(&self) -> u16 {
        self.next
    }
}
