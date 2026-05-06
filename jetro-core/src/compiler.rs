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
use crate::context::EvalError;
use crate::vm::{
    Opcode, Program, CompiledCall, CompiledObjEntry, KvStep, CompiledFSPart,
    BindObjSpec, CompiledPipeStep, CompSpec, DictCompSpec,
    CompiledPatch, CompiledPatchOp, CompiledPatchVal, CompiledPathStep,
    fresh_ics, disable_opcode_fusion,
};

/// Compile-time variable scope used by the `Compiler` to decide whether an
/// identifier refers to a bound variable or a built-in/field name.
#[derive(Clone, Default)]
struct VarCtx {
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
        let mut e = expr.clone();
        Self::reorder_and_operands(&mut e);
        let ctx = VarCtx::default();
        let ops = Self::optimize(Self::emit(&e, &ctx));
        let prog = Program::new(ops, source);
        
        let deduped = crate::analysis::dedup_subprograms(&prog);
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
        use crate::analysis::selectivity_score;
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
            let deduped = crate::analysis::dedup_subprograms(&prog);
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
        use crate::compiler_passes as cp;
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
                
                
                if crate::analysis::expr_is_pure(init)
                    && !crate::analysis::expr_uses_ident(body, name)
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
                    let call = Arc::new(CompiledCall {
                        method: BuiltinMethod::from_name(name.as_str()),
                        name: Arc::from(name.as_str()),
                        sub_progs: sub_progs.into(),
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
                    let call = Arc::new(CompiledCall {
                        method: BuiltinMethod::Unknown,
                        name: Arc::from(name.as_str()),
                        sub_progs: sub_progs.into(),
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
        CompiledCall {
            method,
            name: Arc::from(name),
            sub_progs: sub_progs.into(),
            orig_args: args.iter().cloned().collect::<Vec<_>>().into(),
            demand_max_keep: None,
        }
    }

    /// Compile a method argument that may be a lambda or a plain expression.
    /// For single-param lambdas, the parameter identifier is rewritten to `PushCurrent`
    /// so the body can be executed without an extra variable lookup.
    fn compile_lambda_or_expr(expr: &Expr, ctx: &VarCtx) -> Program {
        match expr {
            Expr::Lambda { params, body } => {
                let inner = ctx.with_vars(params);
                let mut p = Self::compile_sub(body, &inner);
                if params.len() == 1 {
                    let name = params[0].as_str();
                    let new_ops: Vec<Opcode> = p
                        .ops
                        .iter()
                        .map(|op| match op {
                            Opcode::LoadIdent(k) if k.as_ref() == name => Opcode::PushCurrent,
                            other => other.clone(),
                        })
                        .collect();
                    p = Program::new(Self::optimize(new_ops), "<lam-body>");
                }
                p
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
        CompiledPatch { root_prog, ops }
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
