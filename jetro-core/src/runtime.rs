use std::sync::Arc;

use crate::ast::{Arg, Expr};
use crate::context::{Env, EvalError};
use crate::value::Val;

pub fn eval_in_env(expr: &Expr, env: &Env) -> Result<Val, EvalError> {
    vm_eval(expr, env)
}

pub fn evaluate(expr: &Expr, root: &serde_json::Value) -> Result<serde_json::Value, EvalError> {
    let prog = crate::vm::Compiler::compile(expr, "<evaluate>");
    let mut vm = crate::vm::VM::new();
    vm.execute(&prog, root)
}

pub fn evaluate_with(
    expr: &Expr,
    root: &serde_json::Value,
) -> Result<serde_json::Value, EvalError> {
    let prog = crate::vm::Compiler::compile(expr, "<evaluate_with>");
    let mut vm = crate::vm::VM::new();
    vm.execute(&prog, root)
}

pub(crate) fn call_builtin_method(
    recv: Val,
    name: &str,
    args: &[Arg],
    env: &Env,
) -> Result<Val, EvalError> {
    crate::builtins::eval_builtin_method(
        recv,
        name,
        args,
        |arg| eval_pos(arg, env),
        |item, arg| {
            let mut scratch = env.clone();
            apply_item_mut(item.clone(), arg, &mut scratch)
        },
        |left, right, arg| {
            let mut scratch = env.clone();
            apply_item2_mut(left.clone(), right.clone(), arg, &mut scratch)
        },
    )
}

pub(crate) fn eval_global(name: &str, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    match name {
        "coalesce" => {
            for arg in args {
                let value = eval_pos(arg, env)?;
                if !value.is_null() {
                    return Ok(value);
                }
            }
            Ok(Val::Null)
        }
        "chain" | "join" => {
            let mut out = Vec::new();
            for arg in args {
                match eval_pos(arg, env)? {
                    Val::Arr(items) => {
                        out.extend(Arc::try_unwrap(items).unwrap_or_else(|items| (*items).clone()));
                    }
                    Val::IntVec(items) => out.extend(items.iter().map(|n| Val::Int(*n))),
                    Val::FloatVec(items) => out.extend(items.iter().map(|f| Val::Float(*f))),
                    value => out.push(value),
                }
            }
            Ok(Val::arr(out))
        }
        "zip" => {
            let arrs: Result<Vec<_>, _> = args.iter().map(|arg| eval_pos(arg, env)).collect();
            Ok(crate::builtins::global_zip_apply(&arrs?))
        }
        "zip_longest" => {
            let fill = args
                .iter()
                .find_map(|arg| match arg {
                    Arg::Named(name, expr) if name == "fill" => vm_eval(expr, env).ok(),
                    _ => None,
                })
                .unwrap_or(Val::Null);
            let arrs: Result<Vec<_>, _> = args
                .iter()
                .filter(|arg| matches!(arg, Arg::Pos(_)))
                .map(|arg| eval_pos(arg, env))
                .collect();
            Ok(crate::builtins::global_zip_longest_apply(&arrs?, &fill))
        }
        "product" => {
            let arrs: Result<Vec<_>, _> = args.iter().map(|arg| eval_pos(arg, env)).collect();
            Ok(crate::builtins::global_product_apply(&arrs?))
        }
        "range" => {
            let mut nums = Vec::with_capacity(args.len());
            for arg in args {
                let n = eval_pos(arg, env)?
                    .as_i64()
                    .ok_or_else(|| EvalError("range: expected integer arg".into()))?;
                nums.push(n);
            }
            crate::builtins::range_apply(&nums)
        }
        other => {
            if let Some(first) = args.first() {
                let recv = eval_pos(first, env)?;
                call_builtin_method(recv, other, args.get(1..).unwrap_or(&[]), env)
            } else {
                call_builtin_method(env.current.clone(), other, &[], env)
            }
        }
    }
}

pub(crate) fn apply_item_mut(item: Val, arg: &Arg, env: &mut Env) -> Result<Val, EvalError> {
    let expr = match arg {
        Arg::Pos(expr) | Arg::Named(_, expr) => expr,
    };
    match expr {
        Expr::Lambda { params, body } => {
            let name = params.first().map(|s| s.as_str());
            let frame = env.push_lam(name, item);
            let result = vm_eval(body, env);
            env.pop_lam(frame);
            result
        }
        _ => {
            let frame = env.push_lam(None, item);
            let result = vm_eval(expr, env);
            env.pop_lam(frame);
            result
        }
    }
}

pub(crate) fn apply_item2_mut(
    first: Val,
    second: Val,
    arg: &Arg,
    env: &mut Env,
) -> Result<Val, EvalError> {
    match arg {
        Arg::Pos(Expr::Lambda { params, body }) | Arg::Named(_, Expr::Lambda { params, body }) => {
            match params.as_slice() {
                [] => {
                    let frame = env.push_lam(None, second);
                    let result = vm_eval(body, env);
                    env.pop_lam(frame);
                    result
                }
                [param] => {
                    let frame = env.push_lam(Some(param), second);
                    let result = vm_eval(body, env);
                    env.pop_lam(frame);
                    result
                }
                [left, right, ..] => {
                    let left_frame = env.push_lam(Some(left), first);
                    let right_frame = env.push_lam(Some(right), second);
                    let result = vm_eval(body, env);
                    env.pop_lam(right_frame);
                    env.pop_lam(left_frame);
                    result
                }
            }
        }
        _ => apply_item_mut(second, arg, env),
    }
}

pub(crate) fn eval_pos(arg: &Arg, env: &Env) -> Result<Val, EvalError> {
    let expr = match arg {
        Arg::Pos(expr) | Arg::Named(_, expr) => expr,
    };
    vm_eval(expr, env)
}

pub(crate) fn vm_eval(expr: &Expr, env: &Env) -> Result<Val, EvalError> {
    use std::sync::atomic::{AtomicU64, Ordering};

    static NONCE: AtomicU64 = AtomicU64::new(0);
    let n = NONCE.fetch_add(1, Ordering::Relaxed);
    let source = format!("<vm_eval#{}>", n);
    let prog = crate::vm::Compiler::compile(expr, &source);
    let mut vm = crate::vm::VM::new();
    vm.exec(&prog, env)
}
