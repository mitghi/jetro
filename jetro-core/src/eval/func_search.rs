//! Tier 1 search / match / collect methods.
//!
//! - `unique_by(lam)` — dedupe an array by a key produced by a lambda.
//! - `collect()`      — wrap scalar into a one-element array; arrays pass through.
//! - `deep_find(p)`   — DFS pre-order; collect every descendant for which `p` is truthy.
//! - `deep_shape({k1, k2})`
//!                    — DFS; collect every object that has *all* listed keys.
//! - `deep_like({k1: lit, ...})`
//!                    — DFS; collect every object whose listed keys equal the literals.
//!
//! `deep_*` walkers take a single object literal argument (`{...}`) so the
//! caller's intent is visible at the call site — no positional lists.

use std::sync::Arc;

use crate::ast::{Arg, Expr, ObjField};
use super::{Env, EvalError, apply_item_mut, eval};
use super::value::Val;
use super::util::{is_truthy, val_to_key, vals_eq};

macro_rules! err {
    ($($t:tt)*) => { Err(EvalError(format!($($t)*))) };
}

// ── unique_by ────────────────────────────────────────────────────────────────

pub fn unique_by(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let keyfn = args.first().ok_or_else(|| EvalError("unique_by: requires key fn".into()))?;
    let items = recv.into_vec().ok_or_else(|| EvalError("unique_by: expected array".into()))?;
    let mut seen = std::collections::HashSet::new();
    let mut out  = Vec::with_capacity(items.len());
    let mut env_mut = env.clone();
    for item in items {
        let k = apply_item_mut(item.clone(), keyfn, &mut env_mut)?;
        if seen.insert(val_to_key(&k)) { out.push(item); }
    }
    Ok(Val::arr(out))
}

// ── walk (Tier A) ────────────────────────────────────────────────────────────
//
// `.walk(fn)` — post-order traversal.  Recurse into children first, rebuild
// the container with the transformed children, then apply `fn` to the
// rebuilt node.  Scalars pass through `fn` directly.
//
// Post-order (bottom-up) is the Clojure-ish default: consumers can assume
// their children are already normalised when they see a composite node.

pub fn walk(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let fn_arg = args.first().ok_or_else(|| EvalError("walk: requires fn".into()))?;
    let mut env_mut = env.clone();
    walk_impl(recv, fn_arg, &mut env_mut, /*pre=*/false)
}

pub fn walk_pre_fn(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let fn_arg = args.first().ok_or_else(|| EvalError("walk_pre: requires fn".into()))?;
    let mut env_mut = env.clone();
    walk_impl(recv, fn_arg, &mut env_mut, /*pre=*/true)
}

fn walk_impl(v: Val, fn_arg: &Arg, env: &mut super::Env, pre: bool) -> Result<Val, EvalError> {
    let transformed = if pre {
        apply_item_mut(v, fn_arg, env)?
    } else {
        v
    };
    let after_children = match transformed {
        Val::Arr(a) => {
            let items = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
            let mut out = Vec::with_capacity(items.len());
            for child in items { out.push(walk_impl(child, fn_arg, env, pre)?); }
            Val::arr(out)
        }
        Val::IntVec(a) => {
            let mut out = Vec::with_capacity(a.len());
            for n in a.iter() { out.push(walk_impl(Val::Int(*n), fn_arg, env, pre)?); }
            Val::arr(out)
        }
        Val::FloatVec(a) => {
            let mut out = Vec::with_capacity(a.len());
            for f in a.iter() { out.push(walk_impl(Val::Float(*f), fn_arg, env, pre)?); }
            Val::arr(out)
        }
        Val::Obj(m) => {
            let items = Arc::try_unwrap(m).unwrap_or_else(|arc| (*arc).clone());
            let mut out = indexmap::IndexMap::with_capacity(items.len());
            for (k, child) in items {
                out.insert(k, walk_impl(child, fn_arg, env, pre)?);
            }
            Val::obj(out)
        }
        other => other,
    };
    if pre {
        Ok(after_children)
    } else {
        apply_item_mut(after_children, fn_arg, env)
    }
}

// ── collect ──────────────────────────────────────────────────────────────────

pub fn collect(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    use crate::composed::{Stage as _, StageOutput, CollectVal};
    let owned: Option<Val> = match CollectVal.apply(&recv) {
        StageOutput::Pass(c) => Some(c.into_owned()),
        _                    => None,
    };
    Ok(owned.unwrap_or(recv))
}

// ── Deep walk helper ─────────────────────────────────────────────────────────

fn walk_pre<F: FnMut(&Val)>(v: &Val, f: &mut F) {
    f(v);
    match v {
        Val::Arr(a) => for c in a.iter() { walk_pre(c, f); },
        Val::Obj(m) => for (_, c) in m.iter() { walk_pre(c, f); },
        _ => {}
    }
}

// ── deep_find ────────────────────────────────────────────────────────────────

pub fn deep_find(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    if args.is_empty() {
        return err!("find: requires at least one predicate");
    }
    // Hot path: every pred in `args` is a bare expression (not Lambda) —
    // share one scratch Env across all node visits, overwrite `current` in
    // place, skip the SmallVec clone inside `with_current`.  For lambdas
    // fall back to `apply_item`.
    let all_bare = args.iter().all(|a| {
        !matches!(a, Arg::Pos(Expr::Lambda { .. }) | Arg::Named(_, Expr::Lambda { .. }))
    });
    let mut out = Vec::new();
    let mut err_cell: Option<EvalError> = None;
    if all_bare {
        let mut scratch = env.clone();
        let exprs: Vec<&Expr> = args.iter().map(|a| match a {
            Arg::Pos(e) | Arg::Named(_, e) => e,
        }).collect();
        walk_pre(&recv, &mut |node| {
            if err_cell.is_some() { return; }
            scratch.current = node.clone();
            for e in &exprs {
                match eval(e, &scratch) {
                    Ok(v)  => if !is_truthy(&v) { return; }
                    Err(err) => { err_cell = Some(err); return; }
                }
            }
            out.push(node.clone());
        });
    } else {
        let mut env_mut = env.clone();
        walk_pre(&recv, &mut |node| {
            if err_cell.is_some() { return; }
            for p in args {
                match apply_item_mut(node.clone(), p, &mut env_mut) {
                    Ok(v)  => if !is_truthy(&v) { return; }
                    Err(e) => { err_cell = Some(e); return; }
                }
            }
            out.push(node.clone());
        });
    }
    if let Some(e) = err_cell { return Err(e); }
    Ok(Val::arr(out))
}

// ── Pattern extraction from object-literal arg ───────────────────────────────

fn pattern_keys_only(arg: &Arg) -> Result<Vec<Arc<str>>, EvalError> {
    let e = match arg { Arg::Pos(e) | Arg::Named(_, e) => e };
    let fields = match e {
        Expr::Object(fs) => fs,
        _ => return err!("shape: expected `{{k1, k2, ...}}` object pattern"),
    };
    if fields.is_empty() { return err!("shape: empty pattern"); }
    let mut keys = Vec::with_capacity(fields.len());
    for f in fields {
        match f {
            ObjField::Short(k)             => keys.push(Arc::from(k.as_str())),
            ObjField::Kv { key, val, .. }  => {
                if !matches!(val, Expr::Ident(n) if n == key) {
                    return err!("shape: pattern fields must be bare identifiers");
                }
                keys.push(Arc::from(key.as_str()));
            }
            _ => return err!("shape: unsupported pattern field"),
        }
    }
    Ok(keys)
}

fn pattern_key_literals(arg: &Arg, env: &Env) -> Result<Vec<(Arc<str>, Val)>, EvalError> {
    let e = match arg { Arg::Pos(e) | Arg::Named(_, e) => e };
    let fields = match e {
        Expr::Object(fs) => fs,
        _ => return err!("like: expected `{{k: lit, ...}}` object pattern"),
    };
    if fields.is_empty() { return err!("like: empty pattern"); }
    let mut out = Vec::with_capacity(fields.len());
    for f in fields {
        match f {
            ObjField::Kv { key, val, .. } => {
                let v = eval(val, env)?;
                out.push((Arc::from(key.as_str()), v));
            }
            ObjField::Short(k) => {
                let v = eval(&Expr::Ident(k.clone()), env)?;
                out.push((Arc::from(k.as_str()), v));
            }
            _ => return err!("like: unsupported pattern field"),
        }
    }
    Ok(out)
}

// ── deep_shape ───────────────────────────────────────────────────────────────

pub fn deep_shape(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let _ = env;
    let arg = args.first().ok_or_else(|| EvalError("shape: requires pattern".into()))?;
    let keys = pattern_keys_only(arg)?;
    let mut out = Vec::new();
    walk_pre(&recv, &mut |node| {
        if let Val::Obj(m) = node {
            if keys.iter().all(|k| m.contains_key(k.as_ref())) { out.push(node.clone()); }
        }
    });
    Ok(Val::arr(out))
}

// ── rec (Tier E, fixpoint) ───────────────────────────────────────────────────
//
// `.rec(step)` — apply `step` with `@` = current value until the result
// stops changing.  Cap at 10_000 iterations to surface non-terminating
// steps instead of hanging.

pub fn rec(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let step = args.first().ok_or_else(|| EvalError("rec: requires step expression".into()))?;
    let mut env_mut = env.clone();
    let mut cur = recv;
    for _ in 0..10_000 {
        let next = apply_item_mut(cur.clone(), step, &mut env_mut)?;
        if vals_eq(&cur, &next) { return Ok(next); }
        cur = next;
    }
    err!("rec: exceeded 10000 iterations without reaching fixpoint")
}

// ── trace_path (Tier E) ──────────────────────────────────────────────────────
//
// DFS pre-order.  For every descendant where `pred` is truthy, emit
// `{path: "$.a.b[0]", value: v}`.  `$` represents the root of the
// received value.

pub fn trace_path(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let pred = args.first().ok_or_else(|| EvalError("trace_path: requires predicate".into()))?;
    let mut env_mut = env.clone();
    let mut out = Vec::new();
    trace_walk(&recv, String::from("$"), pred, &mut env_mut, &mut out)?;
    Ok(Val::arr(out))
}

fn trace_walk(
    v: &Val,
    path: String,
    pred: &Arg,
    env: &mut super::Env,
    out: &mut Vec<Val>,
) -> Result<(), EvalError> {
    let matched = apply_item_mut(v.clone(), pred, env).map(|r| is_truthy(&r)).unwrap_or(false);
    if matched {
        let mut row = indexmap::IndexMap::with_capacity(2);
        row.insert(Arc::from("path"), Val::Str(Arc::from(path.as_str())));
        row.insert(Arc::from("value"), v.clone());
        out.push(Val::obj(row));
    }
    match v {
        Val::Arr(a) => {
            for (i, c) in a.iter().enumerate() {
                trace_walk(c, format!("{}[{}]", path, i), pred, env, out)?;
            }
        }
        Val::IntVec(a) => {
            for (i, n) in a.iter().enumerate() {
                trace_walk(&Val::Int(*n), format!("{}[{}]", path, i), pred, env, out)?;
            }
        }
        Val::FloatVec(a) => {
            for (i, f) in a.iter().enumerate() {
                trace_walk(&Val::Float(*f), format!("{}[{}]", path, i), pred, env, out)?;
            }
        }
        Val::Obj(m) => {
            for (k, c) in m.iter() {
                trace_walk(c, format!("{}.{}", path, k), pred, env, out)?;
            }
        }
        _ => {}
    }
    Ok(())
}

// ── deep_like ────────────────────────────────────────────────────────────────

pub fn deep_like(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let arg  = args.first().ok_or_else(|| EvalError("like: requires pattern".into()))?;
    let pats = pattern_key_literals(arg, env)?;
    let mut out = Vec::new();
    walk_pre(&recv, &mut |node| {
        if let Val::Obj(m) = node {
            let ok = pats.iter().all(|(k, want)| {
                m.get(k.as_ref()).map(|got| vals_eq(got, want)).unwrap_or(false)
            });
            if ok { out.push(node.clone()); }
        }
    });
    Ok(Val::arr(out))
}

