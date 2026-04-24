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

use crate::ast::{Arg, Expr};
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
        Val::IntVec(a)   => Ok(Val::Int(a.len() as i64)),
        Val::FloatVec(a) => Ok(Val::Int(a.len() as i64)),
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

// ── explode / implode / group_shape (Tier B) ──────────────────────────────────
//
// `explode(field)` — arr-of-obj.  Rows whose `field` is an array produce
// one new row per element, copying the rest of the row.  Rows where the
// field is absent or non-array pass through unchanged.
//
// `implode(field)` — inverse of explode.  Groups rows by every non-`field`
// column; each group collapses to one row whose `field` is the array of
// per-group values.
//
// `group_shape(key, shape)` — `.group_by(key)` then evaluate `shape` with
// `@` bound to each group's items array.  Returns `{key → shape_result}`.

fn field_name(arg: &Arg, who: &str) -> Result<Arc<str>, EvalError> {
    match arg {
        Arg::Pos(Expr::Ident(s)) | Arg::Named(_, Expr::Ident(s)) => Ok(Arc::from(s.as_str())),
        Arg::Pos(Expr::Str(s))   | Arg::Named(_, Expr::Str(s))   => Ok(Arc::from(s.as_str())),
        _ => Err(EvalError(format!("{}: field arg must be identifier or string literal", who))),
    }
}

pub fn explode(recv: Val, args: &[Arg], _env: &Env) -> Result<Val, EvalError> {
    let field = field_name(args.first().ok_or_else(|| EvalError("explode: requires field".into()))?, "explode")?;
    let items = recv.into_vec().ok_or_else(|| EvalError("explode: expected array".into()))?;
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        match item {
            Val::Obj(ref m) => {
                let sub = m.get(field.as_ref()).cloned();
                match sub.as_ref().map(|v| v.is_array()).unwrap_or(false) {
                    true => {
                        let elts = sub.unwrap().into_vec().unwrap();
                        for e in elts {
                            let mut row = (**m).clone();
                            row.insert(Arc::clone(&field), e);
                            out.push(Val::obj(row));
                        }
                    }
                    false => out.push(item),
                }
            }
            other => out.push(other),
        }
    }
    Ok(Val::arr(out))
}

pub fn implode(recv: Val, args: &[Arg], _env: &Env) -> Result<Val, EvalError> {
    let field = field_name(args.first().ok_or_else(|| EvalError("implode: requires field".into()))?, "implode")?;
    let items = recv.into_vec().ok_or_else(|| EvalError("implode: expected array".into()))?;
    let mut groups: IndexMap<Arc<str>, (IndexMap<Arc<str>, Val>, Vec<Val>)> = IndexMap::new();
    for item in items {
        let m = match item {
            Val::Obj(m) => m,
            _ => return Err(EvalError("implode: rows must be objects".into())),
        };
        let mut rest = (*m).clone();
        let val = rest.shift_remove(field.as_ref()).unwrap_or(Val::Null);
        let key_src: IndexMap<Arc<str>, Val> = rest.clone();
        let key = Arc::<str>::from(val_to_key(&Val::obj(key_src)));
        groups.entry(key).or_insert_with(|| (rest, Vec::new())).1.push(val);
    }
    let mut out = Vec::with_capacity(groups.len());
    for (_, (mut rest, vals)) in groups {
        rest.insert(Arc::clone(&field), Val::arr(vals));
        out.push(Val::obj(rest));
    }
    Ok(Val::arr(out))
}

pub fn group_shape(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let key_arg   = args.first().ok_or_else(|| EvalError("group_shape: requires key".into()))?;
    let shape_arg = args.get(1).ok_or_else(|| EvalError("group_shape: requires shape".into()))?;
    let items = recv.into_vec().ok_or_else(|| EvalError("group_shape: expected array".into()))?;
    let mut env_mut = env.clone();
    let mut buckets: IndexMap<Arc<str>, Vec<Val>> = IndexMap::with_capacity(items.len());
    for item in items {
        let k = group_key_mut(&item, key_arg, &mut env_mut)?;
        buckets.entry(k).or_insert_with(Vec::new).push(item);
    }
    let mut out: IndexMap<Arc<str>, Val> = IndexMap::with_capacity(buckets.len());
    for (k, group) in buckets {
        let shaped = apply_item_mut(Val::arr(group), shape_arg, &mut env_mut)?;
        out.insert(k, shaped);
    }
    Ok(Val::obj(out))
}
