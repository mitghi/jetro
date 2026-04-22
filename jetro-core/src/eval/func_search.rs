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
use super::{Env, EvalError, apply_item, eval};
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
    for item in items {
        let k = apply_item(item.clone(), keyfn, env)?;
        if seen.insert(val_to_key(&k)) { out.push(item); }
    }
    Ok(Val::arr(out))
}

// ── collect ──────────────────────────────────────────────────────────────────

pub fn collect(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    match recv {
        Val::Arr(_) => Ok(recv),
        Val::Null   => Ok(Val::arr(Vec::new())),
        other       => Ok(Val::arr(vec![other])),
    }
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
    let pred = args.first().ok_or_else(|| EvalError("find: requires predicate".into()))?;
    let mut out = Vec::new();
    let mut err_cell: Option<EvalError> = None;
    walk_pre(&recv, &mut |node| {
        if err_cell.is_some() { return; }
        match apply_item(node.clone(), pred, env) {
            Ok(v)  => if is_truthy(&v) { out.push(node.clone()); }
            Err(e) => err_cell = Some(e),
        }
    });
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

