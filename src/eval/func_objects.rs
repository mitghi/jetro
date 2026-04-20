//! Object methods: `keys`, `values`, `entries`, `pick`, `omit`,
//! `merge`, `deep_merge`, `rename`, `invert`, `transform_keys`,
//! `transform_values`, `filter_keys`, `filter_values`, `pivot`.
//!
//! All operate on `Val::Obj(Arc<IndexMap<Arc<str>, Val>>)`.  When
//! mutation is needed, the functions `Arc::try_unwrap` the inner map;
//! that is free when the caller holds the last refcount and one
//! `IndexMap::clone()` otherwise.  `IndexMap::shift_remove` is used
//! everywhere (never `remove`) to preserve insertion order — callers
//! depend on that for deterministic serialisation.

use std::sync::Arc;
use indexmap::IndexMap;

use crate::ast::{Arg, Expr};
use super::{Env, EvalError, apply_item, eval_pos, str_arg};
use super::value::Val;
use super::util::{is_truthy, val_to_key, deep_merge, val_key};
use super::func_paths::{parse_path_segs, get_path_impl, PathSeg};

macro_rules! err {
    ($($t:tt)*) => { Err(EvalError(format!($($t)*))) };
}

// ── Field access ──────────────────────────────────────────────────────────────

pub fn keys(recv: Val) -> Result<Val, EvalError> {
    Ok(Val::arr(
        recv.as_object().map(|m| m.keys().map(|k| Val::Str(k.clone())).collect()).unwrap_or_default()
    ))
}

pub fn values(recv: Val) -> Result<Val, EvalError> {
    Ok(Val::arr(recv.as_object().map(|m| m.values().cloned().collect()).unwrap_or_default()))
}

pub fn entries(recv: Val) -> Result<Val, EvalError> {
    Ok(Val::arr(recv.as_object().map(|m| m.iter().map(|(k, v)| {
        Val::arr(vec![Val::Str(k.clone()), v.clone()])
    }).collect()).unwrap_or_default()))
}

// ── Field selection ───────────────────────────────────────────────────────────

pub fn pick(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let map = recv.as_object().ok_or_else(|| EvalError("pick: expected object".into()))?.clone();
    let mut out = IndexMap::new();
    for a in args {
        let key = eval_pos(a, env)?;
        let ks  = match &key { Val::Str(s) => s.to_string(), _ => continue };
        if ks.contains('.') || ks.contains('[') {
            let segs = parse_path_segs(&ks);
            let v    = get_path_impl(&Val::Obj(Arc::new(map.clone())), &segs);
            if let Some(top) = segs.first() {
                let top_key: Arc<str> = match top {
                    PathSeg::Field(f) => Arc::from(f.as_str()),
                    PathSeg::Index(i) => Arc::from(i.to_string().as_str()),
                };
                if !v.is_null() { out.insert(top_key, v); }
            }
        } else if let Some(v) = map.get(ks.as_str()) {
            out.insert(val_key(&ks), v.clone());
        }
    }
    Ok(Val::obj(out))
}

pub fn omit(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let keys: Vec<String> = args.iter()
        .filter_map(|a| eval_pos(a, env).ok())
        .filter_map(|v| if let Val::Str(s) = v { Some(s.to_string()) } else { None })
        .collect();
    let mut map = recv.into_map().ok_or_else(|| EvalError("omit: expected object".into()))?;
    for k in &keys { map.shift_remove(k.as_str()); }
    Ok(Val::obj(map))
}

// ── Merge / defaults ──────────────────────────────────────────────────────────

pub fn merge(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let other = args.first().map(|a| eval_pos(a, env)).transpose()?.unwrap_or(Val::Null);
    match (recv.into_map(), other.into_map()) {
        (Some(mut base), Some(other)) => {
            for (k, v) in other { base.insert(k, v); }
            Ok(Val::obj(base))
        }
        _ => err!("merge: expected two objects"),
    }
}

pub fn deep_merge_method(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let other = args.first().map(|a| eval_pos(a, env)).transpose()?.unwrap_or(Val::Null);
    Ok(deep_merge(recv, other))
}

pub fn defaults(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let other = args.first().map(|a| eval_pos(a, env)).transpose()?.unwrap_or(Val::Null);
    match (recv.into_map(), other.into_map()) {
        (Some(mut base), Some(defs)) => {
            for (k, v) in defs {
                let entry = base.entry(k).or_insert(Val::Null);
                if entry.is_null() { *entry = v; }
            }
            Ok(Val::obj(base))
        }
        _ => err!("defaults: expected two objects"),
    }
}

// ── Rename / invert ───────────────────────────────────────────────────────────

pub fn rename(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let renames = args.first().map(|a| eval_pos(a, env)).transpose()?.unwrap_or(Val::Null);
    match (recv.into_map(), renames.into_map()) {
        (Some(mut obj), Some(renames)) => {
            for (old, new_val) in renames {
                if let Some(v) = obj.shift_remove(old.as_ref()) {
                    let new_key: Arc<str> = if let Val::Str(s) = &new_val {
                        s.clone()
                    } else { old.clone() };
                    obj.insert(new_key, v);
                }
            }
            Ok(Val::obj(obj))
        }
        _ => err!("rename: expected object and rename map"),
    }
}

pub fn invert(recv: Val) -> Result<Val, EvalError> {
    let map = recv.into_map().ok_or_else(|| EvalError("invert: expected object".into()))?;
    let out: IndexMap<Arc<str>, Val> = map.into_iter()
        .map(|(k, v)| (Arc::from(val_to_key(&v).as_str()), Val::Str(k)))
        .collect();
    Ok(Val::obj(out))
}

// ── Transform ─────────────────────────────────────────────────────────────────

pub fn transform_keys(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let lam = args.first().ok_or_else(|| EvalError("transform_keys: requires lambda".into()))?;
    let map = recv.into_map().ok_or_else(|| EvalError("transform_keys: expected object".into()))?;
    let mut out = IndexMap::with_capacity(map.len());
    for (k, v) in map {
        let new_key = Arc::from(val_to_key(&apply_item(Val::Str(k), lam, env)?).as_str());
        out.insert(new_key, v);
    }
    Ok(Val::obj(out))
}

pub fn transform_values(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let lam = args.first().ok_or_else(|| EvalError("transform_values: requires lambda".into()))?;
    let map = recv.into_map().ok_or_else(|| EvalError("transform_values: expected object".into()))?;
    let mut out = IndexMap::with_capacity(map.len());
    for (k, v) in map { out.insert(k, apply_item(v, lam, env)?); }
    Ok(Val::obj(out))
}

pub fn filter_keys(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let lam = args.first().ok_or_else(|| EvalError("filter_keys: requires predicate".into()))?;
    let map = recv.into_map().ok_or_else(|| EvalError("filter_keys: expected object".into()))?;
    let mut out = IndexMap::new();
    for (k, v) in map {
        if is_truthy(&apply_item(Val::Str(k.clone()), lam, env)?) { out.insert(k, v); }
    }
    Ok(Val::obj(out))
}

pub fn filter_values(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let lam = args.first().ok_or_else(|| EvalError("filter_values: requires predicate".into()))?;
    let map = recv.into_map().ok_or_else(|| EvalError("filter_values: expected object".into()))?;
    let mut out = IndexMap::new();
    for (k, v) in map {
        if is_truthy(&apply_item(v.clone(), lam, env)?) { out.insert(k, v); }
    }
    Ok(Val::obj(out))
}

// ── Pairs / pivot ─────────────────────────────────────────────────────────────

pub fn to_pairs(recv: Val) -> Result<Val, EvalError> {
    use super::util::obj2;
    Ok(Val::arr(recv.as_object().map(|m| m.iter().map(|(k, v)| {
        obj2("key", Val::Str(k.clone()), "val", v.clone())
    }).collect()).unwrap_or_default()))
}

pub fn from_pairs(recv: Val) -> Result<Val, EvalError> {
    let items = recv.into_vec().ok_or_else(|| EvalError("from_pairs: expected array".into()))?;
    let mut map = IndexMap::with_capacity(items.len());
    for item in items {
        let k_val = item.get("key").or_else(|| item.get("k")).cloned().unwrap_or(Val::Null);
        let v     = item.get("val").or_else(|| item.get("value")).or_else(|| item.get("v"))
                        .cloned().unwrap_or(Val::Null);
        if let Val::Str(k) = k_val { map.insert(k, v); }
    }
    Ok(Val::obj(map))
}

fn pivot_field(item: &Val, arg: &Arg, env: &Env) -> Result<Val, EvalError> {
    match arg {
        Arg::Pos(Expr::Str(s)) | Arg::Named(_, Expr::Str(s)) => Ok(item.get_field(s.as_str())),
        _ => apply_item(item.clone(), arg, env),
    }
}

pub fn pivot(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let items = recv.into_vec().ok_or_else(|| EvalError("pivot: expected array".into()))?;
    if args.len() >= 3 {
        // 3-arg: pivot(row_field, col_field, val_field) → {row: {col: val}}
        let mut map: IndexMap<Arc<str>, IndexMap<Arc<str>, Val>> = IndexMap::new();
        for item in &items {
            let row = Arc::from(val_to_key(&pivot_field(item, &args[0], env)?).as_str());
            let col = Arc::from(val_to_key(&pivot_field(item, &args[1], env)?).as_str());
            let v   = pivot_field(item, &args[2], env)?;
            map.entry(row).or_insert_with(IndexMap::new).insert(col, v);
        }
        let out: IndexMap<Arc<str>, Val> = map.into_iter()
            .map(|(k, inner)| (k, Val::obj(inner)))
            .collect();
        return Ok(Val::obj(out));
    }
    let key_arg = args.first().ok_or_else(|| EvalError("pivot: requires key arg".into()))?;
    let val_arg = args.get(1).ok_or_else(|| EvalError("pivot: requires value arg".into()))?;
    let mut map = IndexMap::with_capacity(items.len());
    for item in &items {
        let k = Arc::from(val_to_key(&pivot_field(item, key_arg, env)?).as_str());
        let v = pivot_field(item, val_arg, env)?;
        map.insert(k, v);
    }
    Ok(Val::obj(map))
}
