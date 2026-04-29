use std::sync::Arc;

use crate::ast::{Arg, Expr};
use crate::context::{Env, EvalError};
use crate::physical::PipelinePlanSource;
use crate::pipeline;
use crate::value::Val;
use crate::vm::{CompiledCall, VM};

pub(crate) trait PipelineSourceResolver {
    fn resolve_pipeline_source(
        &mut self,
        source: &PipelinePlanSource,
    ) -> Result<ResolvedPipelineSource, EvalError>;
}

pub(crate) enum ResolvedPipelineSource {
    ValFieldChain { keys: Arc<[Arc<str>]> },
    ValReceiver(Val),
}

impl ResolvedPipelineSource {
    pub(crate) fn into_pipeline_source(self) -> pipeline::Source {
        match self {
            Self::ValFieldChain { keys } => pipeline::Source::FieldChain { keys },
            Self::ValReceiver(value) => pipeline::Source::Receiver(value),
        }
    }
}

pub(crate) fn call_builtin_method_compiled(
    vm: &mut VM,
    recv: Val,
    call: &CompiledCall,
    env: &Env,
) -> Result<Val, EvalError> {
    use std::cell::RefCell;

    let vm = RefCell::new(vm);
    crate::builtins::eval_builtin_method(
        recv,
        call.name.as_ref(),
        &call.orig_args,
        |arg| {
            let mut vm = vm.borrow_mut();
            eval_compiled_arg(&mut vm, call, arg, env)
        },
        |item, arg| {
            let mut vm = vm.borrow_mut();
            let mut scratch = env.clone();
            apply_compiled_item(&mut vm, call, item.clone(), arg, &mut scratch)
        },
        |left, right, arg| {
            let mut vm = vm.borrow_mut();
            let mut scratch = env.clone();
            apply_compiled_pair(
                &mut vm,
                call,
                left.clone(),
                right.clone(),
                arg,
                &mut scratch,
            )
        },
    )
}

pub(crate) fn eval_global_compiled(
    vm: &mut VM,
    call: &CompiledCall,
    env: &Env,
) -> Result<Val, EvalError> {
    let name = call.name.as_ref();
    let args = call.orig_args.as_ref();
    match name {
        "coalesce" => {
            for (idx, _) in args.iter().enumerate() {
                let value = eval_compiled_arg_at(vm, call, idx, env)?;
                if !value.is_null() {
                    return Ok(value);
                }
            }
            Ok(Val::Null)
        }
        "chain" | "join" => {
            let mut out = Vec::new();
            for idx in 0..args.len() {
                match eval_compiled_arg_at(vm, call, idx, env)? {
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
            let arrs: Result<Vec<_>, _> = (0..args.len())
                .map(|idx| eval_compiled_arg_at(vm, call, idx, env))
                .collect();
            Ok(crate::builtins::global_zip_apply(&arrs?))
        }
        "zip_longest" => {
            let mut fill = Val::Null;
            let mut arrs = Vec::new();
            for (idx, arg) in args.iter().enumerate() {
                match arg {
                    Arg::Named(name, _) if name == "fill" => {
                        fill = eval_compiled_arg_at(vm, call, idx, env)?;
                    }
                    Arg::Pos(_) => arrs.push(eval_compiled_arg_at(vm, call, idx, env)?),
                    _ => {}
                }
            }
            Ok(crate::builtins::global_zip_longest_apply(&arrs, &fill))
        }
        "product" => {
            let arrs: Result<Vec<_>, _> = (0..args.len())
                .map(|idx| eval_compiled_arg_at(vm, call, idx, env))
                .collect();
            Ok(crate::builtins::global_product_apply(&arrs?))
        }
        "range" => {
            let mut nums = Vec::with_capacity(args.len());
            for idx in 0..args.len() {
                let n = eval_compiled_arg_at(vm, call, idx, env)?
                    .as_i64()
                    .ok_or_else(|| EvalError("range: expected integer arg".into()))?;
                nums.push(n);
            }
            crate::builtins::range_apply(&nums)
        }
        other => if !args.is_empty() {
            let recv = eval_compiled_arg_at(vm, call, 0, env)?;
            call_builtin_method_compiled(vm, recv, call, env)
        } else {
            call_builtin_method_compiled(vm, env.current.clone(), call, env)
        }
        .map_err(|e| EvalError(format!("{}: {}", other, e.0))),
    }
}

fn arg_index(args: &[Arg], needle: &Arg) -> Option<usize> {
    args.iter().position(|arg| std::ptr::eq(arg, needle))
}

fn eval_compiled_arg(
    vm: &mut VM,
    call: &CompiledCall,
    arg: &Arg,
    env: &Env,
) -> Result<Val, EvalError> {
    if let Some(idx) = arg_index(call.orig_args.as_ref(), arg) {
        return eval_compiled_arg_at(vm, call, idx, env);
    }

    // Some builtin adapters construct a static synthetic Arg while decoding
    // richer argument syntax, e.g. `deep_like({k: lit})` evaluates each object
    // field literal independently. Those are not in `CompiledCall::sub_progs`;
    // compile them once for this call rather than failing back to the old
    // tree-walker path.
    let expr = match arg {
        Arg::Pos(expr) | Arg::Named(_, expr) => expr,
    };
    let prog = crate::vm::Compiler::compile(expr, "<synthetic-arg>");
    vm.exec_in_env(&prog, env)
}

fn eval_compiled_arg_at(
    vm: &mut VM,
    call: &CompiledCall,
    idx: usize,
    env: &Env,
) -> Result<Val, EvalError> {
    let prog = call
        .sub_progs
        .get(idx)
        .ok_or_else(|| EvalError(format!("{}: missing compiled argument", call.name)))?;
    vm.exec_in_env(prog, env)
}

fn apply_compiled_item(
    vm: &mut VM,
    call: &CompiledCall,
    item: Val,
    arg: &Arg,
    env: &mut Env,
) -> Result<Val, EvalError> {
    let idx = arg_index(call.orig_args.as_ref(), arg)
        .ok_or_else(|| EvalError(format!("{}: argument lookup failed", call.name)))?;
    let prog = call
        .sub_progs
        .get(idx)
        .ok_or_else(|| EvalError(format!("{}: missing compiled argument", call.name)))?;
    match arg {
        Arg::Pos(Expr::Lambda { params, .. }) | Arg::Named(_, Expr::Lambda { params, .. }) => {
            let name = params.first().map(|s| s.as_str());
            let frame = env.push_lam(name, item);
            let result = vm.exec_in_env(prog, env);
            env.pop_lam(frame);
            result
        }
        _ => {
            let frame = env.push_lam(None, item);
            let result = vm.exec_in_env(prog, env);
            env.pop_lam(frame);
            result
        }
    }
}

fn apply_compiled_pair(
    vm: &mut VM,
    call: &CompiledCall,
    first: Val,
    second: Val,
    arg: &Arg,
    env: &mut Env,
) -> Result<Val, EvalError> {
    let idx = arg_index(call.orig_args.as_ref(), arg)
        .ok_or_else(|| EvalError(format!("{}: argument lookup failed", call.name)))?;
    let prog = call
        .sub_progs
        .get(idx)
        .ok_or_else(|| EvalError(format!("{}: missing compiled argument", call.name)))?;
    match arg {
        Arg::Pos(Expr::Lambda { params, .. }) | Arg::Named(_, Expr::Lambda { params, .. }) => {
            match params.as_slice() {
                [] => {
                    let frame = env.push_lam(None, second);
                    let result = vm.exec_in_env(prog, env);
                    env.pop_lam(frame);
                    result
                }
                [param] => {
                    let frame = env.push_lam(Some(param), second);
                    let result = vm.exec_in_env(prog, env);
                    env.pop_lam(frame);
                    result
                }
                [left, right, ..] => {
                    let left_frame = env.push_lam(Some(left), first);
                    let right_frame = env.push_lam(Some(right), second);
                    let result = vm.exec_in_env(prog, env);
                    env.pop_lam(right_frame);
                    env.pop_lam(left_frame);
                    result
                }
            }
        }
        _ => apply_compiled_item(vm, call, second, arg, env),
    }
}
