//! AST-level lambda body lowering.
//!
//! `substitute_current(body, name)` walks `body` and rewrites every free
//! `Expr::Ident(name)` to `Expr::Current`, respecting all binding-form
//! shadows (nested lambdas, `let`, comprehensions, match-arm pattern
//! bindings, pipeline `as`-binds). The compiler invokes this once per
//! single-param lambda argument so the emitted bytecode is identical to
//! what the equivalent `@`-form expression would produce — no opcode
//! rewriting at the bytecode level, no per-row runtime variable lookup,
//! and uniform participation in kernel classification and peephole passes.

use crate::parse::ast::{
    Arg, ArrayElem, BindTarget, Expr, FStringPart, MatchArm, ObjField, Pat, PatchOp, PathStep,
    PipeStep, Step,
};

/// If `expr` is `Expr::Lambda { params: [name], body }`, return its
/// substituted body (param identifier → `Expr::Current`). Otherwise clone
/// `expr` and return it. Used by every site that calls `Compiler::compile`
/// on an arbitrary `Expr` so a top-level single-param lambda compiles to
/// the same program shape the equivalent `@`-form would.
pub(crate) fn unwrap_single_lambda(expr: &Expr) -> Expr {
    match expr {
        Expr::Lambda { params, body } if params.len() == 1 => {
            substitute_current((**body).clone(), params[0].as_str())
        }
        other => other.clone(),
    }
}

/// Normalise a pipeline/HOF argument into the row-local expression shape used
/// by both VM compilation and symbolic demand analysis.
///
/// Bare identifiers are field reads from the current row (`name` -> `@.name`)
/// and single-parameter lambdas are lowered to their `@`-equivalent body. The
/// caller should still use `compile_lambda_arg` for direct lambda compilation,
/// because it adds the `BindLamCurrent` wrapper needed by nested lambdas that
/// retain references to the outer parameter.
pub(crate) fn normalize_pipeline_arg_expr(expr: &Expr) -> Expr {
    match expr {
        Expr::Lambda { params, body } if params.len() == 1 => {
            substitute_current((**body).clone(), params[0].as_str())
        }
        Expr::Ident(name) => Expr::Chain(
            Box::new(Expr::Current),
            vec![Step::Field(name.clone())],
        ),
        Expr::Chain(base, _) if matches!(base.as_ref(), Expr::Current) => expr.clone(),
        other => other.clone(),
    }
}

/// Inline let-bound lambda values into method-arg references. Implements
/// first-class lambdas as a static macro expansion: `let f = (x => x*2) in
/// $.xs.map(f)` desugars to `$.xs.map(x => x*2)` before any compile pass
/// runs. Pure AST rewrite — no closure runtime, no `Val::Lambda` variant —
/// so the resulting form benefits from the same single-param substitution
/// and `BindLamCurrent` machinery as inline lambdas.
///
/// Resolution rule: an `Expr::Ident(n)` appearing as a positional or named
/// argument to a method or global call is replaced with the most recent
/// active `let n = LAMBDA in ...` binding. `let` shadowing is respected;
/// a non-lambda init for the same name removes the binding.
pub(crate) fn inline_let_bound_lambdas(expr: Expr) -> Expr {
    inline_walk(expr, &mut Vec::new())
}

fn inline_walk(expr: Expr, env: &mut Vec<(String, Expr)>) -> Expr {
    match expr {
        Expr::Null
        | Expr::Bool(_)
        | Expr::Int(_)
        | Expr::Float(_)
        | Expr::Str(_)
        | Expr::Root
        | Expr::Current
        | Expr::Ident(_)
        | Expr::DeleteMark => expr,
        Expr::FString(parts) => Expr::FString(
            parts
                .into_iter()
                .map(|p| match p {
                    FStringPart::Lit(s) => FStringPart::Lit(s),
                    FStringPart::Interp { expr, fmt } => FStringPart::Interp {
                        expr: inline_walk(expr, env),
                        fmt,
                    },
                })
                .collect(),
        ),
        Expr::Chain(base, steps) => Expr::Chain(
            Box::new(inline_walk(*base, env)),
            steps
                .into_iter()
                .map(|s| inline_walk_step(s, env))
                .collect(),
        ),
        Expr::BinOp(l, op, r) => Expr::BinOp(
            Box::new(inline_walk(*l, env)),
            op,
            Box::new(inline_walk(*r, env)),
        ),
        Expr::UnaryNeg(e) => Expr::UnaryNeg(Box::new(inline_walk(*e, env))),
        Expr::Not(e) => Expr::Not(Box::new(inline_walk(*e, env))),
        Expr::Kind { expr, ty, negate } => Expr::Kind {
            expr: Box::new(inline_walk(*expr, env)),
            ty,
            negate,
        },
        Expr::Coalesce(l, r) => Expr::Coalesce(
            Box::new(inline_walk(*l, env)),
            Box::new(inline_walk(*r, env)),
        ),
        Expr::Object(fields) => Expr::Object(
            fields
                .into_iter()
                .map(|f| inline_walk_obj_field(f, env))
                .collect(),
        ),
        Expr::Array(elems) => Expr::Array(
            elems
                .into_iter()
                .map(|e| match e {
                    ArrayElem::Expr(e) => ArrayElem::Expr(inline_walk(e, env)),
                    ArrayElem::Spread(e) => ArrayElem::Spread(inline_walk(e, env)),
                })
                .collect(),
        ),
        Expr::Pipeline { base, steps } => Expr::Pipeline {
            base: Box::new(inline_walk(*base, env)),
            steps: steps
                .into_iter()
                .map(|s| match s {
                    PipeStep::Forward(e) => PipeStep::Forward(inline_walk(e, env)),
                    PipeStep::Bind(b) => PipeStep::Bind(b),
                })
                .collect(),
        },
        Expr::ListComp {
            expr,
            vars,
            iter,
            cond,
        } => Expr::ListComp {
            expr: Box::new(inline_walk(*expr, env)),
            vars,
            iter: Box::new(inline_walk(*iter, env)),
            cond: cond.map(|c| Box::new(inline_walk(*c, env))),
        },
        Expr::DictComp {
            key,
            val,
            vars,
            iter,
            cond,
        } => Expr::DictComp {
            key: Box::new(inline_walk(*key, env)),
            val: Box::new(inline_walk(*val, env)),
            vars,
            iter: Box::new(inline_walk(*iter, env)),
            cond: cond.map(|c| Box::new(inline_walk(*c, env))),
        },
        Expr::SetComp {
            expr,
            vars,
            iter,
            cond,
        } => Expr::SetComp {
            expr: Box::new(inline_walk(*expr, env)),
            vars,
            iter: Box::new(inline_walk(*iter, env)),
            cond: cond.map(|c| Box::new(inline_walk(*c, env))),
        },
        Expr::GenComp {
            expr,
            vars,
            iter,
            cond,
        } => Expr::GenComp {
            expr: Box::new(inline_walk(*expr, env)),
            vars,
            iter: Box::new(inline_walk(*iter, env)),
            cond: cond.map(|c| Box::new(inline_walk(*c, env))),
        },
        Expr::Lambda { params, body } => Expr::Lambda {
            params,
            body: Box::new(inline_walk(*body, env)),
        },
        Expr::Let { name, init, body } => {
            let new_init = inline_walk(*init, env);
            // Bind `name` to a lambda value when:
            //   - `init` is itself an `Expr::Lambda { .. }` (literal), or
            //   - `init` is an `Expr::Ident(prev)` whose binding in scope
            //     resolves to a lambda (let-alias chain).
            let resolved_lambda: Option<Expr> = match &new_init {
                Expr::Lambda { .. } => Some(new_init.clone()),
                Expr::Ident(prev) => env
                    .iter()
                    .rev()
                    .find(|(bound, _)| bound == prev)
                    .map(|(_, lam)| lam.clone())
                    .filter(|e| matches!(e, Expr::Lambda { .. })),
                _ => None,
            };
            match resolved_lambda {
                Some(lam) => env.push((name.clone(), lam)),
                None => {
                    // Even when init is not a Lambda, an outer same-named
                    // binding (if any) is shadowed inside `body`. Push a
                    // sentinel so resolution finds the new binding (a
                    // non-lambda) and stops there rather than reaching past
                    // it.
                    env.push((name.clone(), Expr::Null));
                }
            }
            let new_body = inline_walk(*body, env);
            env.pop();
            Expr::Let {
                name,
                init: Box::new(new_init),
                body: Box::new(new_body),
            }
        }
        Expr::IfElse { cond, then_, else_ } => Expr::IfElse {
            cond: Box::new(inline_walk(*cond, env)),
            then_: Box::new(inline_walk(*then_, env)),
            else_: Box::new(inline_walk(*else_, env)),
        },
        Expr::Try { body, default } => Expr::Try {
            body: Box::new(inline_walk(*body, env)),
            default: Box::new(inline_walk(*default, env)),
        },
        Expr::GlobalCall { name, args } => Expr::GlobalCall {
            name,
            args: args.into_iter().map(|a| inline_walk_arg(a, env)).collect(),
        },
        Expr::Cast { expr, ty } => Expr::Cast {
            expr: Box::new(inline_walk(*expr, env)),
            ty,
        },
        Expr::Patch { root, ops } => Expr::Patch {
            root: Box::new(inline_walk(*root, env)),
            ops: ops
                .into_iter()
                .map(|op| PatchOp {
                    path: op
                        .path
                        .into_iter()
                        .map(|s| match s {
                            PathStep::DynIndex(e) => PathStep::DynIndex(inline_walk(e, env)),
                            PathStep::WildcardFilter(e) => {
                                PathStep::WildcardFilter(Box::new(inline_walk(*e, env)))
                            }
                            other => other,
                        })
                        .collect(),
                    val: inline_walk(op.val, env),
                    cond: op.cond.map(|c| inline_walk(c, env)),
                })
                .collect(),
        },
        Expr::UpdateBatch {
            root,
            selector,
            ops,
        } => Expr::UpdateBatch {
            root: Box::new(inline_walk(*root, env)),
            selector: selector
                .into_iter()
                .map(|s| match s {
                    PathStep::DynIndex(e) => PathStep::DynIndex(inline_walk(e, env)),
                    PathStep::WildcardFilter(e) => {
                        PathStep::WildcardFilter(Box::new(inline_walk(*e, env)))
                    }
                    other => other,
                })
                .collect(),
            ops: ops
                .into_iter()
                .map(|op| PatchOp {
                    path: op
                        .path
                        .into_iter()
                        .map(|s| match s {
                            PathStep::DynIndex(e) => PathStep::DynIndex(inline_walk(e, env)),
                            PathStep::WildcardFilter(e) => {
                                PathStep::WildcardFilter(Box::new(inline_walk(*e, env)))
                            }
                            other => other,
                        })
                        .collect(),
                    val: inline_walk(op.val, env),
                    cond: op.cond.map(|c| inline_walk(c, env)),
                })
                .collect(),
        },
        Expr::Match { scrutinee, arms } => Expr::Match {
            scrutinee: Box::new(inline_walk(*scrutinee, env)),
            arms: arms
                .into_iter()
                .map(|a| MatchArm {
                    pat: a.pat,
                    guard: a.guard.map(|g| inline_walk(g, env)),
                    body: inline_walk(a.body, env),
                })
                .collect(),
        },
    }
}

fn inline_walk_step(step: Step, env: &mut Vec<(String, Expr)>) -> Step {
    match step {
        Step::DynIndex(e) => Step::DynIndex(Box::new(inline_walk(*e, env))),
        Step::InlineFilter(e) => Step::InlineFilter(Box::new(inline_walk(*e, env))),
        Step::Method(n, args) => Step::Method(
            n,
            args.into_iter().map(|a| inline_walk_arg(a, env)).collect(),
        ),
        Step::OptMethod(n, args) => Step::OptMethod(
            n,
            args.into_iter().map(|a| inline_walk_arg(a, env)).collect(),
        ),
        Step::DeepMatch { arms, early_stop } => Step::DeepMatch {
            arms: arms
                .into_iter()
                .map(|a| MatchArm {
                    pat: a.pat,
                    guard: a.guard.map(|g| inline_walk(g, env)),
                    body: inline_walk(a.body, env),
                })
                .collect(),
            early_stop,
        },
        other => other,
    }
}

fn inline_walk_arg(arg: Arg, env: &mut Vec<(String, Expr)>) -> Arg {
    let (name, inner) = match arg {
        Arg::Pos(e) => (None::<String>, e),
        Arg::Named(k, e) => (Some(k), e),
    };
    let resolved = match &inner {
        // Method-arg position is the only place a let-bound lambda is
        // legally invocable as a higher-order function. Resolve only here.
        Expr::Ident(n) => env
            .iter()
            .rev()
            .find(|(bound, _)| bound == n)
            .map(|(_, lambda)| lambda.clone())
            .filter(|e| matches!(e, Expr::Lambda { .. })),
        _ => None,
    };
    let final_expr = match resolved {
        Some(lam) => lam,
        None => inline_walk(inner, env),
    };
    match name {
        None => Arg::Pos(final_expr),
        Some(k) => Arg::Named(k, final_expr),
    }
}

fn inline_walk_obj_field(field: ObjField, env: &mut Vec<(String, Expr)>) -> ObjField {
    match field {
        ObjField::Kv {
            key,
            val,
            optional,
            cond,
        } => ObjField::Kv {
            key,
            val: inline_walk(val, env),
            optional,
            cond: cond.map(|c| inline_walk(c, env)),
        },
        ObjField::Short(s) => ObjField::Short(s),
        ObjField::Dynamic { key, val } => ObjField::Dynamic {
            key: inline_walk(key, env),
            val: inline_walk(val, env),
        },
        ObjField::Spread(e) => ObjField::Spread(inline_walk(e, env)),
        ObjField::SpreadDeep(e) => ObjField::SpreadDeep(inline_walk(e, env)),
    }
}

/// Compile `expr` to a `Program`, transparently lowering single-param
/// `Expr::Lambda` to its substituted body. When the substituted body still
/// references the lambda parameter (only possible from inside a nested
/// lambda whose own `@` is reassigned), wrap the program in a
/// `BindLamCurrent` opcode so `LoadIdent(name)` resolves to the outer row
/// at runtime — even when the host pipeline stage advances per row via
/// `swap_current` rather than `push_lam`. Used by every pipeline-side
/// compile site that previously routed `Compiler::compile` on an
/// `Expr::Lambda` (which would have lowered to a single `PushNull`).
pub(crate) fn compile_lambda_arg(expr: &Expr, source: &str) -> std::sync::Arc<crate::vm::Program> {
    use crate::vm::Opcode;
    use std::sync::Arc;
    if let Expr::Lambda { params, body } = expr {
        if params.len() == 1 {
            let name = params[0].as_str();
            let lowered = substitute_current((**body).clone(), name);
            let body_prog = crate::compile::compiler::Compiler::compile(&lowered, source);
            if crate::plan::analysis::expr_uses_ident(&lowered, name) {
                let body = Arc::new(body_prog);
                let ops = vec![Opcode::BindLamCurrent {
                    name: Some(Arc::from(name)),
                    body,
                }];
                return Arc::new(crate::vm::Program::new(ops, "<lam-body-bind>"));
            }
            return Arc::new(body_prog);
        }
    }
    Arc::new(crate::compile::compiler::Compiler::compile(expr, source))
}

/// Returns `true` when matching `pat` introduces a binding whose name equals
/// `name`. Used to stop substitution descent into match-arm bodies whose
/// patterns shadow the substituted lambda parameter.
fn pat_binds_name(pat: &Pat, name: &str) -> bool {
    match pat {
        Pat::Wild | Pat::Lit(_) | Pat::Range { .. } => false,
        Pat::Bind(n) => n == name,
        Pat::Or(alts) => alts.iter().any(|p| pat_binds_name(p, name)),
        Pat::Obj { fields, rest } => {
            fields.iter().any(|(_, p)| pat_binds_name(p, name))
                || matches!(rest, Some(Some(r)) if r == name)
        }
        Pat::Arr { elems, rest } => {
            elems.iter().any(|p| pat_binds_name(p, name))
                || matches!(rest, Some(Some(r)) if r == name)
        }
        Pat::Kind { name: bind, .. } => matches!(bind, Some(n) if n == name),
    }
}

fn bind_target_shadows(bt: &BindTarget, name: &str) -> bool {
    match bt {
        BindTarget::Name(n) => n == name,
        BindTarget::Obj { fields, rest } => {
            fields.iter().any(|f| f == name) || matches!(rest, Some(r) if r == name)
        }
        BindTarget::Arr(ns) => ns.iter().any(|n| n == name),
    }
}

/// Substitute every free `Expr::Ident(name)` with `Expr::Current` inside
/// `expr`, respecting shadowing introduced by nested lambdas, `let`,
/// comprehensions, match-arm pattern binds, and pipeline binds.
pub(crate) fn substitute_current(expr: Expr, name: &str) -> Expr {
    match expr {
        Expr::Ident(ref n) if n == name => Expr::Current,
        Expr::Null
        | Expr::Bool(_)
        | Expr::Int(_)
        | Expr::Float(_)
        | Expr::Str(_)
        | Expr::Root
        | Expr::Current
        | Expr::Ident(_)
        | Expr::DeleteMark => expr,
        Expr::FString(parts) => Expr::FString(
            parts
                .into_iter()
                .map(|p| match p {
                    FStringPart::Lit(s) => FStringPart::Lit(s),
                    FStringPart::Interp { expr, fmt } => FStringPart::Interp {
                        expr: substitute_current(expr, name),
                        fmt,
                    },
                })
                .collect(),
        ),
        Expr::Chain(base, steps) => Expr::Chain(
            Box::new(substitute_current(*base, name)),
            steps
                .into_iter()
                .map(|s| substitute_step(s, name))
                .collect(),
        ),
        Expr::BinOp(l, op, r) => Expr::BinOp(
            Box::new(substitute_current(*l, name)),
            op,
            Box::new(substitute_current(*r, name)),
        ),
        Expr::UnaryNeg(e) => Expr::UnaryNeg(Box::new(substitute_current(*e, name))),
        Expr::Not(e) => Expr::Not(Box::new(substitute_current(*e, name))),
        Expr::Kind { expr, ty, negate } => Expr::Kind {
            expr: Box::new(substitute_current(*expr, name)),
            ty,
            negate,
        },
        Expr::Coalesce(l, r) => Expr::Coalesce(
            Box::new(substitute_current(*l, name)),
            Box::new(substitute_current(*r, name)),
        ),
        Expr::Object(fields) => Expr::Object(
            fields
                .into_iter()
                .map(|f| substitute_obj_field(f, name))
                .collect(),
        ),
        Expr::Array(elems) => Expr::Array(
            elems
                .into_iter()
                .map(|e| match e {
                    ArrayElem::Expr(e) => ArrayElem::Expr(substitute_current(e, name)),
                    ArrayElem::Spread(e) => ArrayElem::Spread(substitute_current(e, name)),
                })
                .collect(),
        ),
        Expr::Pipeline { base, steps } => {
            let mut shadowed = false;
            let new_base = substitute_current(*base, name);
            let new_steps = steps
                .into_iter()
                .map(|s| match s {
                    PipeStep::Forward(e) => PipeStep::Forward(if shadowed {
                        e
                    } else {
                        substitute_current(e, name)
                    }),
                    PipeStep::Bind(bt) => {
                        if bind_target_shadows(&bt, name) {
                            shadowed = true;
                        }
                        PipeStep::Bind(bt)
                    }
                })
                .collect();
            Expr::Pipeline {
                base: Box::new(new_base),
                steps: new_steps,
            }
        }
        Expr::ListComp {
            expr,
            vars,
            iter,
            cond,
        } => {
            let new_iter = substitute_current(*iter, name);
            if vars.iter().any(|v| v == name) {
                Expr::ListComp {
                    expr,
                    vars,
                    iter: Box::new(new_iter),
                    cond,
                }
            } else {
                Expr::ListComp {
                    expr: Box::new(substitute_current(*expr, name)),
                    vars,
                    iter: Box::new(new_iter),
                    cond: cond.map(|c| Box::new(substitute_current(*c, name))),
                }
            }
        }
        Expr::DictComp {
            key,
            val,
            vars,
            iter,
            cond,
        } => {
            let new_iter = substitute_current(*iter, name);
            if vars.iter().any(|v| v == name) {
                Expr::DictComp {
                    key,
                    val,
                    vars,
                    iter: Box::new(new_iter),
                    cond,
                }
            } else {
                Expr::DictComp {
                    key: Box::new(substitute_current(*key, name)),
                    val: Box::new(substitute_current(*val, name)),
                    vars,
                    iter: Box::new(new_iter),
                    cond: cond.map(|c| Box::new(substitute_current(*c, name))),
                }
            }
        }
        Expr::SetComp {
            expr,
            vars,
            iter,
            cond,
        } => {
            let new_iter = substitute_current(*iter, name);
            if vars.iter().any(|v| v == name) {
                Expr::SetComp {
                    expr,
                    vars,
                    iter: Box::new(new_iter),
                    cond,
                }
            } else {
                Expr::SetComp {
                    expr: Box::new(substitute_current(*expr, name)),
                    vars,
                    iter: Box::new(new_iter),
                    cond: cond.map(|c| Box::new(substitute_current(*c, name))),
                }
            }
        }
        Expr::GenComp {
            expr,
            vars,
            iter,
            cond,
        } => {
            let new_iter = substitute_current(*iter, name);
            if vars.iter().any(|v| v == name) {
                Expr::GenComp {
                    expr,
                    vars,
                    iter: Box::new(new_iter),
                    cond,
                }
            } else {
                Expr::GenComp {
                    expr: Box::new(substitute_current(*expr, name)),
                    vars,
                    iter: Box::new(new_iter),
                    cond: cond.map(|c| Box::new(substitute_current(*c, name))),
                }
            }
        }
        Expr::Lambda { params, body } => {
            // Never descend into a nested `Expr::Lambda` body: that lambda
            // rebinds `@`, so substituting `name` → `Expr::Current` inside
            // would silently change which value `Current` refers to. The
            // inner reference to the outer param remains as `Expr::Ident`
            // and resolves at runtime through the outer call's
            // `env.push_lam(Some(name), item)` binding.
            Expr::Lambda { params, body }
        }
        Expr::Let {
            name: ln,
            init,
            body,
        } => {
            let new_init = substitute_current(*init, name);
            let new_body = if ln == name {
                *body
            } else {
                substitute_current(*body, name)
            };
            Expr::Let {
                name: ln,
                init: Box::new(new_init),
                body: Box::new(new_body),
            }
        }
        Expr::IfElse { cond, then_, else_ } => Expr::IfElse {
            cond: Box::new(substitute_current(*cond, name)),
            then_: Box::new(substitute_current(*then_, name)),
            else_: Box::new(substitute_current(*else_, name)),
        },
        Expr::Try { body, default } => Expr::Try {
            body: Box::new(substitute_current(*body, name)),
            default: Box::new(substitute_current(*default, name)),
        },
        Expr::GlobalCall { name: n, args } => Expr::GlobalCall {
            name: n,
            args: args.into_iter().map(|a| substitute_arg(a, name)).collect(),
        },
        Expr::Cast { expr, ty } => Expr::Cast {
            expr: Box::new(substitute_current(*expr, name)),
            ty,
        },
        Expr::Patch { root, ops } => Expr::Patch {
            root: Box::new(substitute_current(*root, name)),
            ops: ops
                .into_iter()
                .map(|op| substitute_patch_op(op, name))
                .collect(),
        },
        Expr::UpdateBatch {
            root,
            selector,
            ops,
        } => Expr::UpdateBatch {
            root: Box::new(substitute_current(*root, name)),
            selector: selector
                .into_iter()
                .map(|step| substitute_path_step(step, name))
                .collect(),
            ops: ops
                .into_iter()
                .map(|op| substitute_patch_op(op, name))
                .collect(),
        },
        Expr::Match { scrutinee, arms } => Expr::Match {
            scrutinee: Box::new(substitute_current(*scrutinee, name)),
            arms: arms
                .into_iter()
                .map(|a| {
                    if pat_binds_name(&a.pat, name) {
                        a
                    } else {
                        MatchArm {
                            pat: a.pat,
                            guard: a.guard.map(|g| substitute_current(g, name)),
                            body: substitute_current(a.body, name),
                        }
                    }
                })
                .collect(),
        },
    }
}

fn substitute_step(step: Step, name: &str) -> Step {
    match step {
        Step::DynIndex(e) => Step::DynIndex(Box::new(substitute_current(*e, name))),
        Step::Method(n, args) => Step::Method(
            n,
            args.into_iter().map(|a| substitute_arg(a, name)).collect(),
        ),
        Step::OptMethod(n, args) => Step::OptMethod(
            n,
            args.into_iter().map(|a| substitute_arg(a, name)).collect(),
        ),
        Step::InlineFilter(e) => Step::InlineFilter(Box::new(substitute_current(*e, name))),
        Step::DeepMatch { arms, early_stop } => Step::DeepMatch {
            arms: arms
                .into_iter()
                .map(|a| {
                    if pat_binds_name(&a.pat, name) {
                        a
                    } else {
                        MatchArm {
                            pat: a.pat,
                            guard: a.guard.map(|g| substitute_current(g, name)),
                            body: substitute_current(a.body, name),
                        }
                    }
                })
                .collect(),
            early_stop,
        },
        // Field/OptField/Descendant/DescendAll/Index/Slice/Quantifier carry no exprs.
        other => other,
    }
}

fn substitute_arg(arg: Arg, name: &str) -> Arg {
    match arg {
        Arg::Pos(e) => Arg::Pos(substitute_current(e, name)),
        Arg::Named(k, e) => Arg::Named(k, substitute_current(e, name)),
    }
}

fn substitute_obj_field(field: ObjField, name: &str) -> ObjField {
    match field {
        ObjField::Kv {
            key,
            val,
            optional,
            cond,
        } => ObjField::Kv {
            key,
            val: substitute_current(val, name),
            optional,
            cond: cond.map(|c| substitute_current(c, name)),
        },
        ObjField::Short(s) => {
            // `{name}` shorthand desugars to `{name: $.name}` — never touches
            // the lambda parameter, regardless of whether `s == name`.
            ObjField::Short(s)
        }
        ObjField::Dynamic { key, val } => ObjField::Dynamic {
            key: substitute_current(key, name),
            val: substitute_current(val, name),
        },
        ObjField::Spread(e) => ObjField::Spread(substitute_current(e, name)),
        ObjField::SpreadDeep(e) => ObjField::SpreadDeep(substitute_current(e, name)),
    }
}

fn substitute_patch_op(op: PatchOp, name: &str) -> PatchOp {
    PatchOp {
        path: op
            .path
            .into_iter()
            .map(|s| substitute_path_step(s, name))
            .collect(),
        val: substitute_current(op.val, name),
        cond: op.cond.map(|c| substitute_current(c, name)),
    }
}

fn substitute_path_step(step: PathStep, name: &str) -> PathStep {
    match step {
        PathStep::DynIndex(e) => PathStep::DynIndex(substitute_current(e, name)),
        PathStep::WildcardFilter(e) => {
            PathStep::WildcardFilter(Box::new(substitute_current(*e, name)))
        }
        other => other,
    }
}
