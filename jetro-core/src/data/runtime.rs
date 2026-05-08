//! Bridge between the builtin algorithm layer and the VM argument evaluator.
//!
//! The VM compiles lambda/arg bodies into `Arc<Program>` sub-programs stored
//! inside `CompiledCall`. This module provides the closures that re-enter the
//! VM to evaluate those sub-programs per row, then hands those closures to
//! `builtins::eval_builtin_method`. Neither the builtins module nor the VM
//! depends on each other directly — `runtime` is the glue.

use std::sync::Arc;

use crate::parse::ast::{Arg, Expr};
use crate::data::context::{Env, EvalError};
use crate::ir::physical::PipelinePlanSource;
use crate::exec::pipeline::{self, PipelineBody};
use crate::data::value::Val;
use crate::vm::{CompiledCall, VM};

/// Allows `physical_eval`'s `ExecCtx` to hand a pipeline-source resolution
/// request back up to the execution context without importing it directly,
/// breaking the circular dependency between the physical layer and the VM.
pub(crate) trait PipelineSourceResolver {
    /// Resolve a `PipelinePlanSource` into a concrete `ResolvedPipelineSource`
    /// given the pipeline body context, returning an error on failure.
    fn resolve_pipeline_source(
        &mut self,
        source: &PipelinePlanSource,
        body: &PipelineBody,
    ) -> Result<ResolvedPipelineSource, EvalError>;
}

/// The concrete form of a pipeline source after resolution by `ExecCtx`.
/// Either a lazily-traversed field-path key chain or a fully materialised `Val`.
pub(crate) enum ResolvedPipelineSource {
    /// A sequence of field-name keys to follow on each row — avoids cloning
    /// the full document when the source is a simple nested-field path.
    ValFieldChain { keys: Arc<[Arc<str>]> },
    /// A fully materialised value used as the pipeline's input rows.
    ValReceiver(Val),
}

impl ResolvedPipelineSource {
    /// Convert into the `pipeline::Source` variant expected by the pipeline
    /// executor, consuming `self`.
    pub(crate) fn into_pipeline_source(self) -> pipeline::Source {
        match self {
            Self::ValFieldChain { keys } => pipeline::Source::FieldChain { keys },
            Self::ValReceiver(value) => pipeline::Source::Receiver(value),
        }
    }
}

/// Entry point for the VM path: bridges a `CompiledCall` to
/// `builtins::eval_builtin_method` by building three closures that re-enter
/// the VM to evaluate compiled sub-programs for plain args, single-item
/// lambdas, and pair-item lambdas respectively.
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

/// Evaluate a top-level (non-method) global call such as `coalesce`, `chain`,
/// `zip`, `zip_longest`, `product`, or `range`, falling back to
/// `call_builtin_method_compiled` for unrecognised names.
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
        // `now()` — current Unix time in milliseconds. No-arg builtin. Used
        // for timestamp stamping in patches: `last_seen: now()`.
        "now" if args.is_empty() => {
            use std::time::{SystemTime, UNIX_EPOCH};
            let ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis() as i64)
                .unwrap_or(0);
            Ok(crate::data::value::Val::Int(ms))
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

/// Find the position of `needle` inside `args` by pointer identity, returning
/// `None` when the argument was synthesised and is not part of the original slice.
fn arg_index(args: &[Arg], needle: &Arg) -> Option<usize> {
    args.iter().position(|arg| std::ptr::eq(arg, needle))
}

/// Evaluate `arg` against the current environment, preferring the pre-compiled
/// sub-program stored in `call` when the argument is part of the original call
/// and falling back to a fresh `Compiler::compile` for synthetic arguments.
pub(crate) fn eval_compiled_arg(
    vm: &mut VM,
    call: &CompiledCall,
    arg: &Arg,
    env: &Env,
) -> Result<Val, EvalError> {
    if let Some(idx) = arg_index(call.orig_args.as_ref(), arg) {
        return eval_compiled_arg_at(vm, call, idx, env);
    }


    let expr = match arg {
        Arg::Pos(expr) | Arg::Named(_, expr) => expr,
    };
    let prog = crate::compile::compiler::Compiler::compile(expr, "<synthetic-arg>");
    vm.exec_in_env(&prog, env)
}

/// Fetch the nth pre-compiled sub-program from `call` and execute it in `env`,
/// returning an error when the index is out of range.
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

/// Push `item` as the current value (`@`) into `env`, execute the sub-program
/// at `arg`'s position, then restore the previous current value. Handles both
/// named lambda parameters and implicit `@` bindings.
///
/// Higher-order builtins (`.filter`, `.map`, `.any`, `.all`, `.find`, ...)
/// invoke this in a tight per-element loop. When the lambda body has been
/// classified to a non-`Generic` `BodyKernel` at compile time, we dispatch
/// through `eval_kernel` for native-speed Rust evaluation; otherwise the
/// VM runs the program. The kernel form skips the per-iteration VM stack
/// init and opcode-dispatch loop, which dominates simple predicates.
fn apply_compiled_item(
    vm: &mut VM,
    call: &CompiledCall,
    item: Val,
    arg: &Arg,
    env: &mut Env,
) -> Result<Val, EvalError> {
    use crate::exec::pipeline::{eval_kernel, BodyKernel};

    let idx = arg_index(call.orig_args.as_ref(), arg)
        .ok_or_else(|| EvalError(format!("{}: argument lookup failed", call.name)))?;
    let prog = call
        .sub_progs
        .get(idx)
        .ok_or_else(|| EvalError(format!("{}: missing compiled argument", call.name)))?;
    let kernel = call.sub_kernels.get(idx);

    // Fast path: when the kernel is a non-`Generic` shape, evaluate `prog`
    // directly against `item` through native Rust kernels — no VM stack
    // init, no opcode-dispatch loop. The compile-time AST lambda lowering
    // rewrites `r => r.id` to the same opcodes as `@.id`, so single-param
    // named lambdas are kernel-eligible exactly when the equivalent
    // `@`-form is. Multi-param lambdas and bodies wrapped in
    // `BindLamCurrent` (nested-ref case) classify as `Generic`, so the
    // gate alone is sufficient.
    if let Some(k) = kernel {
        if !matches!(k, BodyKernel::Generic) && env.has_no_vars() {
            // `LoadIdent` at the head of a body program dispatches to a
            // no-arg builtin when current is array/string and the ident
            // is a builtin name (e.g. `.map(len)` invokes `len` as a
            // builtin) — a path the kernel form drops. Skip the fast
            // path for those programs.
            let starts_with_load_ident =
                matches!(prog.ops.first(), Some(crate::vm::Opcode::LoadIdent(_)));
            if !starts_with_load_ident {
                return eval_kernel(k, &item, |fallback_item| {
                    let frame = env.push_lam(None, fallback_item.clone());
                    let result = vm.exec_in_env(prog, env);
                    env.pop_lam(frame);
                    result
                });
            }
        }
    }

    match arg {
        Arg::Pos(Expr::Lambda { params, .. }) | Arg::Named(_, Expr::Lambda { params, .. }) => {
            // Single-param lambda body has been AST-substituted to the
            // `@`-form opcode shape — name is unreferenced in `prog`,
            // so push_lam can skip the var insert/remove and only swap
            // `current`. Multi-param lambdas (handled in
            // `apply_compiled_pair`) keep their named bindings.
            let name = if params.len() == 1 {
                None
            } else {
                params.first().map(|s| s.as_str())
            };
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

/// Push a two-value frame (`first`, `second`) for binary lambda arguments
/// such as `reduce(acc, x -> …)`, then execute the sub-program and restore
/// the environment. Degrades to `apply_compiled_item` for non-binary lambdas.
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
                [_param] => {
                    // Single-param lambda body is AST-substituted; param
                    // name is no longer referenced in `prog`.
                    let frame = env.push_lam(None, second);
                    let result = vm.exec_in_env(prog, env);
                    env.pop_lam(frame);
                    result
                }
                [left, _right, ..] => {
                    // Multi-param fast path: rightmost param substituted
                    // to `Current`, only earlier params keep their named
                    // bindings.
                    let left_frame = env.push_lam(Some(left), first);
                    let right_frame = env.push_lam(None, second);
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
