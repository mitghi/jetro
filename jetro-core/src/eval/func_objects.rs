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
use super::{Env, EvalError, apply_item, eval_pos};
use super::value::Val;
use super::util::{is_truthy, val_to_key, deep_merge};
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

/// Resolve one pick argument into `(out_key, src_key_or_path)`.
///
/// Accepted forms:
///   - `name`                → `("name", "name")`    (ident = field name)
///   - `"name"`              → `("name", "name")`    (string literal)
///   - `"a.b"`               → `("a",    "a.b")`     (dotted path, top-key as out key)
///   - `alias: name`         → `("alias","name")`    (named arg, ident rhs)
///   - `alias: "name"`       → `("alias","name")`    (named arg, string rhs)
fn pick_arg(a: &Arg, env: &Env) -> Result<Option<(Arc<str>, String)>, EvalError> {
    match a {
        Arg::Pos(Expr::Ident(s)) => Ok(Some((Arc::from(s.as_str()), s.clone()))),
        Arg::Pos(e) => {
            let v = super::eval(e, env)?;
            match v {
                Val::Str(s) => {
                    let top: Arc<str> = if s.contains('.') || s.contains('[') {
                        match parse_path_segs(&s).first() {
                            Some(PathSeg::Field(f)) => Arc::from(f.as_str()),
                            Some(PathSeg::Index(i)) => Arc::from(i.to_string().as_str()),
                            None => Arc::from(s.as_ref()),
                        }
                    } else { s.clone() };
                    Ok(Some((top, s.to_string())))
                }
                _ => Ok(None),
            }
        }
        Arg::Named(alias, Expr::Ident(src)) => {
            Ok(Some((Arc::from(alias.as_str()), src.clone())))
        }
        Arg::Named(alias, e) => {
            let v = super::eval(e, env)?;
            match v {
                Val::Str(s) => Ok(Some((Arc::from(alias.as_str()), s.to_string()))),
                _ => Ok(None),
            }
        }
    }
}

fn pick_one(obj: &IndexMap<Arc<str>, Val>, resolved: &[(Arc<str>, String)]) -> IndexMap<Arc<str>, Val> {
    let mut out = IndexMap::with_capacity(resolved.len());
    for (out_key, src) in resolved {
        if src.contains('.') || src.contains('[') {
            let segs = parse_path_segs(src);
            let v = get_path_impl(&Val::Obj(Arc::new(obj.clone())), &segs);
            if !v.is_null() { out.insert(out_key.clone(), v); }
        } else if let Some(v) = obj.get(src.as_str()) {
            out.insert(out_key.clone(), v.clone());
        }
    }
    out
}

pub fn pick(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let mut resolved = Vec::with_capacity(args.len());
    for a in args {
        if let Some(p) = pick_arg(a, env)? { resolved.push(p); }
    }
    match recv {
        Val::Obj(m) => Ok(Val::obj(pick_one(&m, &resolved))),
        Val::Arr(a) => {
            let out: Vec<Val> = a.iter().map(|el| match el {
                Val::Obj(m) => Val::obj(pick_one(m, &resolved)),
                other       => other.clone(),
            }).collect();
            Ok(Val::arr(out))
        }
        _ => err!("pick: expected object or array of objects"),
    }
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
        .map(|(k, v)| {
            let nk = match v {
                Val::Str(s) => s,
                other       => Arc::<str>::from(val_to_key(&other)),
            };
            (nk, Val::Str(k))
        })
        .collect();
    Ok(Val::obj(out))
}

// ── Transform ─────────────────────────────────────────────────────────────────

pub fn transform_keys(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let lam = args.first().ok_or_else(|| EvalError("transform_keys: requires lambda".into()))?;
    let map = recv.into_map().ok_or_else(|| EvalError("transform_keys: expected object".into()))?;
    let mut out = IndexMap::with_capacity(map.len());
    for (k, v) in map {
        let new_key: Arc<str> = match apply_item(Val::Str(k), lam, env)? {
            Val::Str(s) => s,
            other       => Arc::<str>::from(val_to_key(&other)),
        };
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
    let mut out = IndexMap::with_capacity(map.len());
    for (k, v) in map {
        if is_truthy(&apply_item(Val::Str(k.clone()), lam, env)?) { out.insert(k, v); }
    }
    Ok(Val::obj(out))
}

pub fn filter_values(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let lam = args.first().ok_or_else(|| EvalError("filter_values: requires predicate".into()))?;
    let map = recv.into_map().ok_or_else(|| EvalError("filter_values: expected object".into()))?;
    let mut out = IndexMap::with_capacity(map.len());
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
    #[inline]
    fn to_arc(v: Val) -> Arc<str> {
        match v { Val::Str(s) => s, other => Arc::<str>::from(val_to_key(&other)) }
    }
    if args.len() >= 3 {
        // 3-arg: pivot(row_field, col_field, val_field) → {row: {col: val}}
        let mut map: IndexMap<Arc<str>, IndexMap<Arc<str>, Val>> = IndexMap::new();
        for item in &items {
            let row = to_arc(pivot_field(item, &args[0], env)?);
            let col = to_arc(pivot_field(item, &args[1], env)?);
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
        let k = to_arc(pivot_field(item, key_arg, env)?);
        let v = pivot_field(item, val_arg, env)?;
        map.insert(k, v);
    }
    Ok(Val::obj(map))
}
