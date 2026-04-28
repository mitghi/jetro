//! Interpreter for `physical.rs` expression plans.

use std::sync::Arc;

use crate::ast::BinOp;
use crate::context::{Env, EvalError};
use crate::physical::{
    PhysicalArrayElem, PhysicalChainStep, PhysicalExpr, PhysicalObjField, PhysicalPathStep,
};
use crate::pipeline;
use crate::value::Val;
use crate::{Jetro, VM};

pub(crate) fn run(j: &Jetro, expr: &PhysicalExpr) -> Result<Val, EvalError> {
    let root = j.root_val();
    let mut env = Env::new(root.clone());
    let mut vm = VM::new();
    run_physical_expr(j, expr, &root, &mut env, &mut vm)
}

fn run_physical_expr(
    j: &Jetro,
    expr: &PhysicalExpr,
    root: &Val,
    env: &mut Env,
    vm: &mut VM,
) -> Result<Val, EvalError> {
    match expr {
        PhysicalExpr::Literal(value) => Ok(value.clone()),
        PhysicalExpr::Root => Ok(root.clone()),
        PhysicalExpr::Current => Ok(env.current.clone()),
        PhysicalExpr::Ident(name) => Ok(env
            .get_var(name.as_ref())
            .cloned()
            .unwrap_or_else(|| env.current.get_field(name.as_ref()))),
        PhysicalExpr::Pipeline(pipeline) => {
            pipeline.run_with(root, Some(j as &dyn pipeline::PipelineData))
        }
        PhysicalExpr::RootPath(steps) => Ok(run_root_path(root, steps)),
        PhysicalExpr::Chain { base, steps } => run_physical_chain(j, base, steps, root, env, vm),
        PhysicalExpr::UnaryNeg(inner) => match run_physical_expr(j, inner, root, env, vm)? {
            Val::Int(n) => Ok(Val::Int(-n)),
            Val::Float(f) => Ok(Val::Float(-f)),
            _ => Err(EvalError("unary minus requires a number".into())),
        },
        PhysicalExpr::Not(inner) => {
            let value = run_physical_expr(j, inner, root, env, vm)?;
            Ok(Val::Bool(!crate::util::is_truthy(&value)))
        }
        PhysicalExpr::Binary { lhs, op, rhs } => {
            run_physical_binary(j, lhs, *op, rhs, root, env, vm)
        }
        PhysicalExpr::Kind { expr, ty, negate } => {
            let value = run_physical_expr(j, expr, root, env, vm)?;
            let matched = crate::util::kind_matches(&value, *ty);
            Ok(Val::Bool(if *negate { !matched } else { matched }))
        }
        PhysicalExpr::Coalesce { lhs, rhs } => {
            let lhs = run_physical_expr(j, lhs, root, env, vm)?;
            if lhs.is_null() {
                run_physical_expr(j, rhs, root, env, vm)
            } else {
                Ok(lhs)
            }
        }
        PhysicalExpr::IfElse { cond, then_, else_ } => {
            let cond = run_physical_expr(j, cond, root, env, vm)?;
            if crate::util::is_truthy(&cond) {
                run_physical_expr(j, then_, root, env, vm)
            } else {
                run_physical_expr(j, else_, root, env, vm)
            }
        }
        PhysicalExpr::Try { body, default } => match run_physical_expr(j, body, root, env, vm) {
            Ok(value) if !value.is_null() => Ok(value),
            Ok(_) | Err(_) => run_physical_expr(j, default, root, env, vm),
        },
        PhysicalExpr::Object(fields) => run_physical_object(j, fields, root, env, vm),
        PhysicalExpr::Array(elems) => run_physical_array(j, elems, root, env, vm),
        PhysicalExpr::Let { name, init, body } => {
            let init_val = run_physical_expr(j, init, root, env, vm)?;
            let mut body_env = env.with_var(name.as_ref(), init_val);
            run_physical_expr(j, body, root, &mut body_env, vm)
        }
        PhysicalExpr::Vm(program) => vm.exec_in_env(program, env),
    }
}

fn run_physical_chain(
    j: &Jetro,
    base: &PhysicalExpr,
    steps: &[PhysicalChainStep],
    root: &Val,
    env: &mut Env,
    vm: &mut VM,
) -> Result<Val, EvalError> {
    let mut cur = run_physical_expr(j, base, root, env, vm)?;
    for step in steps {
        cur = match step {
            PhysicalChainStep::Field(key) => cur.get_field(key.as_ref()),
            PhysicalChainStep::Index(idx) => cur.get_index(*idx),
            PhysicalChainStep::DynIndex(expr) => {
                let key = run_physical_expr(j, expr, root, env, vm)?;
                match key {
                    Val::Int(idx) => cur.get_index(idx),
                    Val::Str(key) => cur.get_field(key.as_ref()),
                    Val::StrSlice(key) => cur.get_field(key.as_str()),
                    _ => Val::Null,
                }
            }
            PhysicalChainStep::Method { call, optional } => {
                if *optional && cur.is_null() {
                    Val::Null
                } else {
                    call.try_apply(&cur)?.ok_or_else(|| {
                        EvalError(format!("{:?}: builtin unsupported", call.method))
                    })?
                }
            }
        };
    }
    Ok(cur)
}

fn run_physical_binary(
    j: &Jetro,
    lhs: &PhysicalExpr,
    op: BinOp,
    rhs: &PhysicalExpr,
    root: &Val,
    env: &mut Env,
    vm: &mut VM,
) -> Result<Val, EvalError> {
    if op == BinOp::And {
        let lhs = run_physical_expr(j, lhs, root, env, vm)?;
        if !crate::util::is_truthy(&lhs) {
            return Ok(Val::Bool(false));
        }
        let rhs = run_physical_expr(j, rhs, root, env, vm)?;
        return Ok(Val::Bool(crate::util::is_truthy(&rhs)));
    }
    if op == BinOp::Or {
        let lhs = run_physical_expr(j, lhs, root, env, vm)?;
        if crate::util::is_truthy(&lhs) {
            return Ok(lhs);
        }
        return run_physical_expr(j, rhs, root, env, vm);
    }

    let lhs = run_physical_expr(j, lhs, root, env, vm)?;
    let rhs = run_physical_expr(j, rhs, root, env, vm)?;
    match op {
        BinOp::Add => crate::util::add_vals(lhs, rhs),
        BinOp::Sub => crate::util::num_op(lhs, rhs, |a, b| a - b, |a, b| a - b),
        BinOp::Mul => crate::util::num_op(lhs, rhs, |a, b| a * b, |a, b| a * b),
        BinOp::Div => {
            let denom = rhs.as_f64().unwrap_or(0.0);
            if denom == 0.0 {
                Err(EvalError("division by zero".into()))
            } else {
                Ok(Val::Float(lhs.as_f64().unwrap_or(0.0) / denom))
            }
        }
        BinOp::Mod => crate::util::num_op(lhs, rhs, |a, b| a % b, |a, b| a % b),
        BinOp::Eq => Ok(Val::Bool(crate::util::vals_eq(&lhs, &rhs))),
        BinOp::Neq => Ok(Val::Bool(!crate::util::vals_eq(&lhs, &rhs))),
        BinOp::Lt | BinOp::Lte | BinOp::Gt | BinOp::Gte => {
            Ok(Val::Bool(crate::util::cmp_vals_binop(&lhs, op, &rhs)))
        }
        BinOp::Fuzzy => {
            let lhs = match &lhs {
                Val::Str(s) => s.to_lowercase(),
                _ => crate::util::val_to_string(&lhs).to_lowercase(),
            };
            let rhs = match &rhs {
                Val::Str(s) => s.to_lowercase(),
                _ => crate::util::val_to_string(&rhs).to_lowercase(),
            };
            Ok(Val::Bool(lhs.contains(&rhs) || rhs.contains(&lhs)))
        }
        BinOp::And | BinOp::Or => unreachable!(),
    }
}

fn run_root_path(root: &Val, steps: &[PhysicalPathStep]) -> Val {
    let mut cur = root.clone();
    for step in steps {
        cur = match step {
            PhysicalPathStep::Field(key) => cur.get_field(key.as_ref()),
            PhysicalPathStep::Index(idx) => cur.get_index(*idx),
        };
    }
    cur
}

fn run_physical_object(
    j: &Jetro,
    fields: &[PhysicalObjField],
    root: &Val,
    env: &mut Env,
    vm: &mut VM,
) -> Result<Val, EvalError> {
    let mut map: indexmap::IndexMap<Arc<str>, Val> =
        indexmap::IndexMap::with_capacity(fields.len());
    for field in fields {
        match field {
            PhysicalObjField::Kv {
                key,
                val,
                optional,
                cond,
            } => {
                if let Some(cond) = cond {
                    if !crate::util::is_truthy(&run_physical_expr(j, cond, root, env, vm)?) {
                        continue;
                    }
                }
                let value = run_physical_expr(j, val, root, env, vm)?;
                if *optional && value.is_null() {
                    continue;
                }
                map.insert(Arc::clone(key), value);
            }
            PhysicalObjField::Short(name) => {
                let value = if let Some(value) = env.get_var(name.as_ref()) {
                    value.clone()
                } else {
                    env.current.get_field(name.as_ref())
                };
                if !value.is_null() {
                    map.insert(Arc::clone(name), value);
                }
            }
            PhysicalObjField::Dynamic { key, val } => {
                let key = Arc::from(
                    crate::util::val_to_key(&run_physical_expr(j, key, root, env, vm)?).as_str(),
                );
                let value = run_physical_expr(j, val, root, env, vm)?;
                map.insert(key, value);
            }
            PhysicalObjField::Spread(expr) => {
                if let Val::Obj(other) = run_physical_expr(j, expr, root, env, vm)? {
                    let entries = Arc::try_unwrap(other).unwrap_or_else(|m| (*m).clone());
                    for (key, value) in entries {
                        map.insert(key, value);
                    }
                }
            }
            PhysicalObjField::SpreadDeep(expr) => {
                if let Val::Obj(other) = run_physical_expr(j, expr, root, env, vm)? {
                    let base = std::mem::take(&mut map);
                    let merged = crate::util::deep_merge_concat(Val::obj(base), Val::Obj(other));
                    if let Val::Obj(merged) = merged {
                        map = Arc::try_unwrap(merged).unwrap_or_else(|m| (*m).clone());
                    }
                }
            }
        }
    }
    Ok(Val::obj(map))
}

fn run_physical_array(
    j: &Jetro,
    elems: &[PhysicalArrayElem],
    root: &Val,
    env: &mut Env,
    vm: &mut VM,
) -> Result<Val, EvalError> {
    let mut out = Vec::with_capacity(elems.len());
    for elem in elems {
        match elem {
            PhysicalArrayElem::Expr(expr) => out.push(run_physical_expr(j, expr, root, env, vm)?),
            PhysicalArrayElem::Spread(expr) => match run_physical_expr(j, expr, root, env, vm)? {
                Val::Arr(items) => {
                    let items = Arc::try_unwrap(items).unwrap_or_else(|items| (*items).clone());
                    out.extend(items);
                }
                Val::IntVec(items) => out.extend(items.iter().map(|n| Val::Int(*n))),
                Val::FloatVec(items) => out.extend(items.iter().map(|f| Val::Float(*f))),
                Val::StrVec(items) => out.extend(items.iter().cloned().map(Val::Str)),
                Val::StrSliceVec(items) => out.extend(items.iter().cloned().map(Val::StrSlice)),
                other => out.push(other),
            },
        }
    }
    Ok(Val::arr(out))
}
