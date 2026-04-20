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
use super::{Env, EvalError, eval, apply_item, eval_pos};
use super::value::Val;
use super::util::{is_truthy, val_to_key};

macro_rules! err {
    ($($t:tt)*) => { Err(EvalError(format!($($t)*))) };
}

// ── Numeric aggregates ────────────────────────────────────────────────────────

pub fn sum(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let nums = collect_nums(recv, args, env)?;
    if nums.iter().all(|v| matches!(v, Val::Int(_))) {
        Ok(Val::Int(nums.iter().map(|v| v.as_i64().unwrap_or(0)).sum()))
    } else {
        Ok(Val::Float(nums.iter().map(|v| v.as_f64().unwrap_or(0.0)).sum()))
    }
}

pub fn avg(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let nums = collect_nums(recv, args, env)?;
    if nums.is_empty() { return Ok(Val::Null); }
    Ok(Val::Float(nums.iter().map(|v| v.as_f64().unwrap_or(0.0)).sum::<f64>() / nums.len() as f64))
}

pub fn minmax(recv: Val, args: &[Arg], env: &Env, want_max: bool) -> Result<Val, EvalError> {
    let nums = collect_nums(recv, args, env)?;
    if nums.is_empty() { return Ok(Val::Null); }
    Ok(nums.into_iter().reduce(|a, b| {
        let af = a.as_f64().unwrap_or(0.0);
        let bf = b.as_f64().unwrap_or(0.0);
        if want_max { if bf > af { b } else { a } } else { if bf < af { b } else { a } }
    }).unwrap())
}

fn collect_nums(recv: Val, args: &[Arg], env: &Env) -> Result<Vec<Val>, EvalError> {
    let items = recv.into_vec().ok_or_else(|| EvalError("expected array for numeric aggregate".into()))?;
    if args.is_empty() {
        Ok(items.into_iter().filter(|v| v.is_number()).collect())
    } else {
        let mut out = Vec::new();
        for item in items {
            let v = apply_item(item, &args[0], env)?;
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
            let n = items.into_iter().filter(|item| {
                args.first()
                    .and_then(|pred| apply_item(item.clone(), pred, env).ok())
                    .map(|v| is_truthy(&v))
                    .unwrap_or(false)
            }).count();
            Ok(Val::Int(n as i64))
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
            let result = if want_all {
                a.iter().all(|item| {
                    apply_item(item.clone(), pred, env).map(|v| is_truthy(&v)).unwrap_or(false)
                })
            } else {
                a.iter().any(|item| {
                    apply_item(item.clone(), pred, env).map(|v| is_truthy(&v)).unwrap_or(false)
                })
            };
            Ok(Val::Bool(result))
        }
        _ => Ok(Val::Bool(!want_all)),
    }
}

// ── Grouping ──────────────────────────────────────────────────────────────────

pub fn group_by(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let key_arg = args.first().ok_or_else(|| EvalError("groupBy: requires key".into()))?;
    let items = recv.into_vec().ok_or_else(|| EvalError("groupBy: expected array".into()))?;
    let mut map: IndexMap<Arc<str>, Val> = IndexMap::new();
    for item in items {
        let k: Arc<str> = Arc::from(val_to_key(&apply_item(item.clone(), key_arg, env)?).as_str());
        let bucket = map.entry(k).or_insert_with(|| Val::arr(Vec::new()));
        bucket.as_array_mut().unwrap().push(item);
    }
    Ok(Val::obj(map))
}

pub fn count_by(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let key_arg = args.first().ok_or_else(|| EvalError("countBy: requires key".into()))?;
    let items = recv.into_vec().ok_or_else(|| EvalError("countBy: expected array".into()))?;
    let mut map: IndexMap<Arc<str>, Val> = IndexMap::new();
    for item in items {
        let k: Arc<str> = Arc::from(val_to_key(&apply_item(item.clone(), key_arg, env)?).as_str());
        let counter = map.entry(k).or_insert(Val::Int(0));
        if let Val::Int(n) = counter { *n += 1; }
    }
    Ok(Val::obj(map))
}

pub fn index_by(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let key_arg = args.first().ok_or_else(|| EvalError("indexBy: requires key".into()))?;
    let items = recv.into_vec().ok_or_else(|| EvalError("indexBy: expected array".into()))?;
    let mut map: IndexMap<Arc<str>, Val> = IndexMap::new();
    for item in items {
        let k: Arc<str> = Arc::from(val_to_key(&apply_item(item.clone(), key_arg, env)?).as_str());
        map.insert(k, item);
    }
    Ok(Val::obj(map))
}

// ── Sort helper (exported for func_arrays) ────────────────────────────────────

pub fn sort_key_eval(item: &Val, key_expr: &crate::ast::Expr, env: &Env) -> Result<Val, EvalError> {
    eval(key_expr, &env.with_current(item.clone()))
}
