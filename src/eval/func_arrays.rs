//! Array / sequence methods: `filter`, `map`, `flatmap`, `sort`,
//! `reverse`, `take`, `drop`, `takewhile`, `dropwhile`, `unique`,
//! `chunk`, `window`, `zip`, `scan`, set operations, etc.
//!
//! Every function here takes an already-materialised array (`Vec<Val>`)
//! or produces one.  `Val::Arr` is `Arc`-wrapped so most functions
//! `into_vec()` — an `Arc::try_unwrap` + fallback clone — to get
//! mutable ownership; when the caller holds the only reference this
//! is free, otherwise one deep clone.
//!
//! Predicate evaluation uses [`apply_item`], which binds the element
//! to the current scope (`@`) before evaluating the lambda body.  A
//! few two-arg helpers use [`apply_item2`] (e.g. `scan`, which binds
//! accumulator + element).

use std::sync::Arc;
use indexmap::IndexMap;

use crate::ast::{Arg, Expr};
use super::{Env, EvalError, eval, apply_item, apply_item2, eval_pos, first_i64_arg};
use super::value::Val;
use super::util::{is_truthy, val_to_key, val_to_string, flatten_val, zip_arrays, cartesian, cmp_vals, val_key, obj2};

macro_rules! err {
    ($($t:tt)*) => { Err(EvalError(format!($($t)*))) };
}

// ── Filter / map / sort ───────────────────────────────────────────────────────

pub fn filter(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let pred = args.first().ok_or_else(|| EvalError("filter: requires predicate".into()))?;
    let items = recv.into_vec().ok_or_else(|| EvalError("filter: expected array".into()))?;
    let mut out = Vec::new();
    for item in items {
        if is_truthy(&apply_item(item.clone(), pred, env)?) { out.push(item); }
    }
    Ok(Val::arr(out))
}

pub fn map(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let mapper = args.first().ok_or_else(|| EvalError("map: requires mapper".into()))?;
    let items = recv.into_vec().ok_or_else(|| EvalError("map: expected array".into()))?;
    let r: Result<Vec<_>, _> = items.into_iter().map(|item| apply_item(item, mapper, env)).collect();
    Ok(Val::arr(r?))
}

pub fn flat_map(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    Ok(flatten_val(map(recv, args, env)?, 1))
}

pub fn sort(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let mut items = recv.into_vec().ok_or_else(|| EvalError("sort: expected array".into()))?;
    if args.is_empty() {
        items.sort_by(|x, y| cmp_vals(x, y));
        return Ok(Val::arr(items));
    }
    // 2-param lambda comparator: sort(lambda a, b: a.field < b.field)
    if args.len() == 1 {
        if let Arg::Pos(Expr::Lambda { params, body }) | Arg::Named(_, Expr::Lambda { params, body }) = &args[0] {
            if params.len() == 2 {
                let p1 = params[0].clone();
                let p2 = params[1].clone();
                let mut err_cell: Option<EvalError> = None;
                items.sort_by(|x, y| {
                    if err_cell.is_some() { return std::cmp::Ordering::Equal; }
                    let inner = env.with_vars2(&p1, x.clone(), &p2, y.clone());
                    match eval(body, &inner) {
                        Ok(Val::Bool(true)) => std::cmp::Ordering::Less,
                        Ok(_) => std::cmp::Ordering::Greater,
                        Err(e) => { err_cell = Some(e); std::cmp::Ordering::Equal }
                    }
                });
                if let Some(e) = err_cell { return Err(e); }
                return Ok(Val::arr(items));
            }
        }
    }
    let keys: Vec<(&Expr, bool)> = args.iter().filter_map(|arg| match arg {
        Arg::Pos(Expr::UnaryNeg(inner)) => Some((inner.as_ref(), true)),
        Arg::Pos(e)                     => Some((e, false)),
        _                               => None,
    }).collect();

    let mut err_cell: Option<EvalError> = None;
    items.sort_by(|x, y| {
        if err_cell.is_some() { return std::cmp::Ordering::Equal; }
        for (key_expr, desc) in &keys {
            let kx = eval(key_expr, &env.with_current(x.clone()));
            let ky = eval(key_expr, &env.with_current(y.clone()));
            match (kx, ky) {
                (Ok(vx), Ok(vy)) => {
                    let ord = cmp_vals(&vx, &vy);
                    if ord != std::cmp::Ordering::Equal {
                        return if *desc { ord.reverse() } else { ord };
                    }
                }
                (Err(e), _) | (_, Err(e)) => {
                    err_cell = Some(e);
                    return std::cmp::Ordering::Equal;
                }
            }
        }
        std::cmp::Ordering::Equal
    });
    if let Some(e) = err_cell { return Err(e); }
    Ok(Val::arr(items))
}

// ── Dedup / flatten ───────────────────────────────────────────────────────────

pub fn unique(recv: Val) -> Result<Val, EvalError> {
    let items = recv.into_vec().ok_or_else(|| EvalError("unique: expected array".into()))?;
    let mut seen = std::collections::HashSet::new();
    Ok(Val::arr(items.into_iter().filter(|v| seen.insert(val_to_key(v))).collect()))
}

pub fn flatten(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let depth = first_i64_arg(args, env).unwrap_or(1) as usize;
    Ok(flatten_val(recv, depth))
}

pub fn compact(recv: Val) -> Result<Val, EvalError> {
    let items = recv.into_vec().ok_or_else(|| EvalError("compact: expected array".into()))?;
    Ok(Val::arr(items.into_iter().filter(|v| !v.is_null()).collect()))
}

// ── Reorder ───────────────────────────────────────────────────────────────────

pub fn reverse(recv: Val) -> Result<Val, EvalError> {
    match recv {
        Val::Arr(a) => {
            let mut v = Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone());
            v.reverse();
            Ok(Val::arr(v))
        }
        Val::Str(s) => Ok(Val::Str(Arc::from(s.chars().rev().collect::<String>().as_str()))),
        _ => err!("reverse: expected array or string"),
    }
}

// ── Access ────────────────────────────────────────────────────────────────────

pub fn first(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let n = first_i64_arg(args, env)?;
    match recv {
        Val::Arr(a) if n == 1 => Ok(a.first().cloned().unwrap_or(Val::Null)),
        Val::Arr(a)           => Ok(Val::arr(a.iter().take(n as usize).cloned().collect())),
        _                     => Ok(Val::Null),
    }
}

pub fn last(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let n = first_i64_arg(args, env)?;
    match recv {
        Val::Arr(a) if n == 1 => Ok(a.last().cloned().unwrap_or(Val::Null)),
        Val::Arr(a) => {
            let s = a.len().saturating_sub(n as usize);
            Ok(Val::arr(a[s..].to_vec()))
        }
        _ => Ok(Val::Null),
    }
}

pub fn nth(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let n = first_i64_arg(args, env)? as usize;
    Ok(recv.as_array().and_then(|a| a.get(n)).cloned().unwrap_or(Val::Null))
}

// ── Mutation ──────────────────────────────────────────────────────────────────

pub fn append(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let item = args.first().map(|a| eval_pos(a, env)).transpose()?.unwrap_or(Val::Null);
    let mut v = recv.into_vec().ok_or_else(|| EvalError("append: expected array".into()))?;
    v.push(item);
    Ok(Val::arr(v))
}

pub fn prepend(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let item = args.first().map(|a| eval_pos(a, env)).transpose()?.unwrap_or(Val::Null);
    let mut v = recv.into_vec().ok_or_else(|| EvalError("prepend: expected array".into()))?;
    v.insert(0, item);
    Ok(Val::arr(v))
}

pub fn remove(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let pred = args.first().ok_or_else(|| EvalError("remove: requires arg".into()))?;
    let v = recv.into_vec().ok_or_else(|| EvalError("remove: expected array".into()))?;
    let is_lambda = matches!(pred,
        Arg::Pos(Expr::Lambda { .. }) | Arg::Named(_, Expr::Lambda { .. })
    );
    if is_lambda {
        let mut out = Vec::new();
        for item in v {
            if !is_truthy(&apply_item(item.clone(), pred, env)?) { out.push(item); }
        }
        Ok(Val::arr(out))
    } else {
        let item = eval_pos(pred, env)?;
        let key  = val_to_key(&item);
        Ok(Val::arr(v.into_iter().filter(|v| val_to_key(v) != key).collect()))
    }
}

// ── Join ──────────────────────────────────────────────────────────────────────

pub fn join(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let sep = args.first()
        .map(|a| eval_pos(a, env)).transpose()?
        .and_then(|v| if let Val::Str(s) = v { Some(s.to_string()) } else { None })
        .unwrap_or_default();
    let items = recv.into_vec().ok_or_else(|| EvalError("join: expected array".into()))?;
    let parts: Vec<String> = items.iter().map(val_to_string).collect();
    Ok(Val::Str(Arc::from(parts.join(&sep).as_str())))
}

/// Equi-join: `lhs.equi_join(rhs, lhs_key, rhs_key)`.
/// Inner hash-join — builds a hash map on rhs_key values of rhs, probes
/// each lhs element by lhs_key.  Produces a merged object per match
/// (rhs fields override lhs on collision).  Non-object rows are skipped.
pub fn equi_join(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    use std::collections::HashMap;
    let other = args.first().ok_or_else(|| EvalError("equi_join: requires other array".into()))?;
    let other = eval_pos(other, env)?;
    let lhs_key = str_arg_req(args.get(1), env, "equi_join: requires lhs_key")?;
    let rhs_key = str_arg_req(args.get(2), env, "equi_join: requires rhs_key")?;
    let left  = recv.into_vec().ok_or_else(|| EvalError("equi_join: lhs not array".into()))?;
    let right = other.into_vec().ok_or_else(|| EvalError("equi_join: rhs not array".into()))?;
    let mut idx: HashMap<String, Vec<Val>> = HashMap::new();
    for r in right {
        let key = match &r {
            Val::Obj(o) => o.get(rhs_key.as_ref()).map(val_to_key),
            _ => None,
        };
        if let Some(k) = key { idx.entry(k).or_default().push(r); }
    }
    let mut out = Vec::new();
    for l in left {
        let key = match &l {
            Val::Obj(o) => o.get(lhs_key.as_ref()).map(val_to_key),
            _ => None,
        };
        let Some(k) = key else { continue };
        let Some(matches) = idx.get(&k) else { continue };
        for r in matches { out.push(merge_pair(&l, r)); }
    }
    Ok(Val::arr(out))
}

fn str_arg_req(a: Option<&Arg>, env: &Env, msg: &'static str) -> Result<Arc<str>, EvalError> {
    let a = a.ok_or_else(|| EvalError(msg.into()))?;
    match eval_pos(a, env)? {
        Val::Str(s) => Ok(s),
        _ => Err(EvalError(msg.into())),
    }
}

fn merge_pair(l: &Val, r: &Val) -> Val {
    match (l, r) {
        (Val::Obj(lo), Val::Obj(ro)) => {
            let mut m = (**lo).clone();
            for (k, v) in ro.iter() { m.insert(k.clone(), v.clone()); }
            Val::obj(m)
        }
        _ => l.clone(),
    }
}

// ── Set operations ────────────────────────────────────────────────────────────

pub fn diff(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let other = eval_pos(args.first().ok_or_else(|| EvalError("diff: requires arg".into()))?, env)?;
    let a = recv.into_vec().ok_or_else(|| EvalError("diff: expected arrays".into()))?;
    let b = other.into_vec().ok_or_else(|| EvalError("diff: expected arrays".into()))?;
    let b_keys: std::collections::HashSet<String> = b.iter().map(val_to_key).collect();
    Ok(Val::arr(a.into_iter().filter(|v| !b_keys.contains(&val_to_key(v))).collect()))
}

pub fn intersect(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let other = eval_pos(args.first().ok_or_else(|| EvalError("intersect: requires arg".into()))?, env)?;
    let a = recv.into_vec().ok_or_else(|| EvalError("intersect: expected arrays".into()))?;
    let b = other.into_vec().ok_or_else(|| EvalError("intersect: expected arrays".into()))?;
    let b_keys: std::collections::HashSet<String> = b.iter().map(val_to_key).collect();
    Ok(Val::arr(a.into_iter().filter(|v| b_keys.contains(&val_to_key(v))).collect()))
}

pub fn union(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let other = eval_pos(args.first().ok_or_else(|| EvalError("union: requires arg".into()))?, env)?;
    let mut a = recv.into_vec().ok_or_else(|| EvalError("union: expected arrays".into()))?;
    let b     = other.into_vec().ok_or_else(|| EvalError("union: expected arrays".into()))?;
    let a_keys: std::collections::HashSet<String> = a.iter().map(val_to_key).collect();
    for v in b { if !a_keys.contains(&val_to_key(&v)) { a.push(v); } }
    Ok(Val::arr(a))
}

// ── Itertools ─────────────────────────────────────────────────────────────────

pub fn enumerate(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let items = recv.into_vec().ok_or_else(|| EvalError("enumerate: expected array".into()))?;
    if let Some(lam) = args.first() {
        let r: Result<Vec<_>, _> = items.into_iter().enumerate()
            .map(|(i, v)| apply_item2(Val::Int(i as i64), v, lam, env))
            .collect();
        Ok(Val::arr(r?))
    } else {
        Ok(Val::arr(items.into_iter().enumerate()
            .map(|(i, v)| obj2("index", Val::Int(i as i64), "value", v))
            .collect()))
    }
}

pub fn pairwise(recv: Val) -> Result<Val, EvalError> {
    let items = recv.into_vec().ok_or_else(|| EvalError("pairwise: expected array".into()))?;
    Ok(Val::arr(items.windows(2).map(|w| Val::arr(w.to_vec())).collect()))
}

pub fn window(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let n = first_i64_arg(args, env).unwrap_or(2).max(1) as usize;
    let items = recv.into_vec().ok_or_else(|| EvalError("window: expected array".into()))?;
    Ok(Val::arr(items.windows(n).map(|w| Val::arr(w.to_vec())).collect()))
}

pub fn chunk(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let n = first_i64_arg(args, env).unwrap_or(1).max(1) as usize;
    let items = recv.into_vec().ok_or_else(|| EvalError("chunk: expected array".into()))?;
    Ok(Val::arr(items.chunks(n).map(|c| Val::arr(c.to_vec())).collect()))
}

pub fn takewhile(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let pred  = args.first().ok_or_else(|| EvalError("takewhile: requires predicate".into()))?;
    let items = recv.into_vec().ok_or_else(|| EvalError("takewhile: expected array".into()))?;
    let mut out = Vec::new();
    for item in items {
        if is_truthy(&apply_item(item.clone(), pred, env)?) { out.push(item); }
        else { break; }
    }
    Ok(Val::arr(out))
}

pub fn dropwhile(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let pred  = args.first().ok_or_else(|| EvalError("dropwhile: requires predicate".into()))?;
    let items = recv.into_vec().ok_or_else(|| EvalError("dropwhile: expected array".into()))?;
    let mut dropping = true;
    let mut out = Vec::new();
    for item in items {
        if dropping {
            if !is_truthy(&apply_item(item.clone(), pred, env)?) { dropping = false; out.push(item); }
        } else { out.push(item); }
    }
    Ok(Val::arr(out))
}

pub fn accumulate(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let lam   = args.first().ok_or_else(|| EvalError("accumulate: requires lambda".into()))?;
    let start = args.iter().find_map(|a| {
        if let Arg::Named(n, e) = a {
            if n == "start" { eval(e, env).ok() } else { None }
        } else { None }
    });
    let items = recv.into_vec().ok_or_else(|| EvalError("accumulate: expected array".into()))?;
    let mut running: Option<Val> = start;
    let mut out = Vec::new();
    for item in items {
        running = Some(if let Some(acc) = running {
            apply_item2(acc, item, lam, env)?
        } else { item });
        out.push(running.clone().unwrap());
    }
    Ok(Val::arr(out))
}

pub fn partition(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let pred  = args.first().ok_or_else(|| EvalError("partition: requires predicate".into()))?;
    let items = recv.into_vec().ok_or_else(|| EvalError("partition: expected array".into()))?;
    let (mut t, mut f) = (Vec::new(), Vec::new());
    for item in items {
        if is_truthy(&apply_item(item.clone(), pred, env)?) { t.push(item); }
        else { f.push(item); }
    }
    let mut m = IndexMap::with_capacity(2);
    m.insert(val_key("true"),  Val::arr(t));
    m.insert(val_key("false"), Val::arr(f));
    Ok(Val::obj(m))
}

pub fn zip_method(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let other = args.first().map(|a| eval_pos(a, env)).transpose()?.unwrap_or(Val::arr(vec![]));
    zip_arrays(recv, other, false, Val::Null)
}

pub fn zip_longest_method(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let fill = args.iter().find_map(|a| {
        if let Arg::Named(n, e) = a { if n == "fill" { eval(e, env).ok() } else { None } } else { None }
    }).unwrap_or(Val::Null);
    let other = args.iter().find(|a| matches!(a, Arg::Pos(_)))
        .map(|a| eval_pos(a, env)).transpose()?.unwrap_or(Val::arr(vec![]));
    zip_arrays(recv, other, true, fill)
}

// ── Global zip / product ──────────────────────────────────────────────────────

pub fn global_zip(args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let arrs: Result<Vec<_>, _> = args.iter().map(|a| eval_pos(a, env)).collect();
    let arrs = arrs?;
    let len  = arrs.iter().filter_map(|a| a.as_array().map(|a| a.len())).min().unwrap_or(0);
    Ok(Val::arr((0..len).map(|i| {
        Val::arr(arrs.iter().filter_map(|a| a.as_array()?.get(i).cloned()).collect())
    }).collect()))
}

pub fn global_zip_longest(args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let fill = args.iter().find_map(|a| {
        if let Arg::Named(n, e) = a { if n == "fill" { eval(e, env).ok() } else { None } } else { None }
    }).unwrap_or(Val::Null);
    let arrs: Result<Vec<_>, _> = args.iter()
        .filter(|a| matches!(a, Arg::Pos(_)))
        .map(|a| eval_pos(a, env)).collect();
    let arrs = arrs?;
    let len  = arrs.iter().filter_map(|a| a.as_array().map(|a| a.len())).max().unwrap_or(0);
    Ok(Val::arr((0..len).map(|i| {
        Val::arr(arrs.iter().map(|a| {
            a.as_array().and_then(|arr| arr.get(i)).cloned().unwrap_or_else(|| fill.clone())
        }).collect())
    }).collect()))
}

pub fn global_product(args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let arrs: Result<Vec<Vec<Val>>, _> = args.iter()
        .map(|a| eval_pos(a, env).map(|v| v.into_vec().unwrap_or_default()))
        .collect();
    Ok(Val::arr(cartesian(&arrs?).into_iter().map(Val::arr).collect()))
}
