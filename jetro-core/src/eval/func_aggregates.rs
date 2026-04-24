//! Aggregate methods: `sum`, `avg`, `count`, `group_by`, `index_by`,
//! `partition`, `count_by`.
//!
//! These consume an entire array and produce either a scalar or a
//! regrouped structure.  Most take an optional key expression
//! (bare form or lambda) evaluated with each element bound to `@`.
//! Numeric aggregates widen Int → Float when mixed.  Grouping
//! aggregates stringify the computed key via `val_key` so the result
//! can be indexed by an `IndexMap` while preserving insertion order.

use std::sync::Arc;
use indexmap::IndexMap;

use crate::ast::Arg;
use super::{Env, EvalError, apply_item_mut};
use super::value::Val;
use super::util::{is_truthy, val_to_key};

macro_rules! err {
    ($($t:tt)*) => { Err(EvalError(format!($($t)*))) };
}

// ── Numeric aggregates ────────────────────────────────────────────────────────

pub fn sum(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let nums = collect_nums(recv, args, env)?;
    // Single pass: stay on Int until we see a Float, then widen once.
    let mut i_acc: i64 = 0;
    let mut f_acc: f64 = 0.0;
    let mut floated = false;
    for v in &nums {
        match v {
            Val::Int(n)   if !floated => { i_acc += *n; }
            Val::Int(n)              => { f_acc += *n as f64; }
            Val::Float(f) if !floated => { f_acc = i_acc as f64 + *f; floated = true; }
            Val::Float(f)            => { f_acc += *f; }
            _ => {}
        }
    }
    Ok(if floated { Val::Float(f_acc) } else { Val::Int(i_acc) })
}

pub fn avg(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let nums = collect_nums(recv, args, env)?;
    if nums.is_empty() { return Ok(Val::Null); }
    let n = nums.len();
    let mut acc: f64 = 0.0;
    for v in &nums {
        acc += v.as_f64().unwrap_or(0.0);
    }
    Ok(Val::Float(acc / n as f64))
}

pub fn minmax(recv: Val, args: &[Arg], env: &Env, want_max: bool) -> Result<Val, EvalError> {
    let nums = collect_nums(recv, args, env)?;
    let mut iter = nums.into_iter();
    let Some(first) = iter.next() else { return Ok(Val::Null); };
    let mut best_f = first.as_f64().unwrap_or(0.0);
    let mut best = first;
    for v in iter {
        let vf = v.as_f64().unwrap_or(0.0);
        let replace = if want_max { vf > best_f } else { vf < best_f };
        if replace { best_f = vf; best = v; }
    }
    Ok(best)
}

fn collect_nums(recv: Val, args: &[Arg], env: &Env) -> Result<Vec<Val>, EvalError> {
    let items = recv.into_vec().ok_or_else(|| EvalError("expected array for numeric aggregate".into()))?;
    if args.is_empty() {
        Ok(items.into_iter().filter(|v| v.is_number()).collect())
    } else {
        let mut env_mut = env.clone();
        let mut out = Vec::with_capacity(items.len());
        for item in items {
            let v = apply_item_mut(item, &args[0], &mut env_mut)?;
            if v.is_number() { out.push(v); }
        }
        Ok(out)
    }
}

// ── Count / any / all ─────────────────────────────────────────────────────────

pub fn count(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    match recv {
        Val::Arr(a) => {
            if args.is_empty() { return Ok(Val::Int(a.len() as i64)); }
            let items = Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone());
            let Some(pred) = args.first() else { return Ok(Val::Int(0)); };
            let mut env_mut = env.clone();
            let mut n: i64 = 0;
            for item in items {
                if apply_item_mut(item, pred, &mut env_mut).map(|v| is_truthy(&v)).unwrap_or(false) {
                    n += 1;
                }
            }
            Ok(Val::Int(n))
        }
        Val::Str(s)  => Ok(Val::Int(s.chars().count() as i64)),
        Val::Obj(m)  => Ok(Val::Int(m.len() as i64)),
        _ => err!("count: unsupported type"),
    }
}

pub fn any_all(recv: Val, args: &[Arg], env: &Env, want_all: bool) -> Result<Val, EvalError> {
    match recv {
        Val::Arr(a) => {
            if a.is_empty() { return Ok(Val::Bool(want_all)); }
            if args.is_empty() { return Ok(Val::Bool(true)); }
            let pred = &args[0];
            let mut env_mut = env.clone();
            let result = if want_all {
                let mut ok = true;
                for item in a.iter() {
                    if !apply_item_mut(item.clone(), pred, &mut env_mut).map(|v| is_truthy(&v)).unwrap_or(false) {
                        ok = false; break;
                    }
                }
                ok
            } else {
                let mut ok = false;
                for item in a.iter() {
                    if apply_item_mut(item.clone(), pred, &mut env_mut).map(|v| is_truthy(&v)).unwrap_or(false) {
                        ok = true; break;
                    }
                }
                ok
            };
            Ok(Val::Bool(result))
        }
        _ => Ok(Val::Bool(!want_all)),
    }
}

// ── Grouping ──────────────────────────────────────────────────────────────────

/// Compute a grouping key once, reusing the Arc<str> when the key
/// expression already produced a string.
#[inline]
fn group_key_mut(item: &Val, key_arg: &Arg, env: &mut Env) -> Result<Arc<str>, EvalError> {
    Ok(match apply_item_mut(item.clone(), key_arg, env)? {
        Val::Str(s) => s,
        other       => Arc::<str>::from(val_to_key(&other)),
    })
}

pub fn group_by(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let key_arg = args.first().ok_or_else(|| EvalError("groupBy: requires key".into()))?;
    let items = recv.into_vec().ok_or_else(|| EvalError("groupBy: expected array".into()))?;
    let mut env_mut = env.clone();
    let mut map: IndexMap<Arc<str>, Val> = IndexMap::with_capacity(items.len());
    for item in items {
        let k = group_key_mut(&item, key_arg, &mut env_mut)?;
        let bucket = map.entry(k).or_insert_with(|| Val::arr(Vec::new()));
        bucket.as_array_mut().unwrap().push(item);
    }
    Ok(Val::obj(map))
}

pub fn count_by(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let key_arg = args.first().ok_or_else(|| EvalError("countBy: requires key".into()))?;
    let items = recv.into_vec().ok_or_else(|| EvalError("countBy: expected array".into()))?;
    let mut env_mut = env.clone();
    let mut map: IndexMap<Arc<str>, Val> = IndexMap::with_capacity(items.len());
    for item in items {
        let k = group_key_mut(&item, key_arg, &mut env_mut)?;
        let counter = map.entry(k).or_insert(Val::Int(0));
        if let Val::Int(n) = counter { *n += 1; }
    }
    Ok(Val::obj(map))
}

pub fn index_by(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let key_arg = args.first().ok_or_else(|| EvalError("indexBy: requires key".into()))?;
    let items = recv.into_vec().ok_or_else(|| EvalError("indexBy: expected array".into()))?;
    let mut env_mut = env.clone();
    let mut map: IndexMap<Arc<str>, Val> = IndexMap::with_capacity(items.len());
    for item in items {
        let k = group_key_mut(&item, key_arg, &mut env_mut)?;
        map.insert(k, item);
    }
    Ok(Val::obj(map))
}

