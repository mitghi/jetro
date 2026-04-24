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
use super::{Env, EvalError, eval, apply_item_mut, apply_item2_mut, eval_pos, first_i64_arg};
use super::value::Val;
use super::util::{is_truthy, val_to_key, val_to_string, flatten_val, zip_arrays, cartesian, cmp_vals, val_key, obj2};

macro_rules! err {
    ($($t:tt)*) => { Err(EvalError(format!($($t)*))) };
}

// ── Filter / map / sort ───────────────────────────────────────────────────────

pub fn filter(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let pred = args.first().ok_or_else(|| EvalError("filter: requires predicate".into()))?;
    let items = recv.into_vec().ok_or_else(|| EvalError("filter: expected array".into()))?;
    let mut out = Vec::with_capacity(items.len());
    let bare = match pred {
        Arg::Pos(e) | Arg::Named(_, e) =>
            if !matches!(e, Expr::Lambda { .. }) { Some(e) } else { None },
    };
    if let Some(expr) = bare {
        let mut scratch = env.clone();
        for item in items {
            scratch.current = item.clone();
            if is_truthy(&eval(expr, &scratch)?) { out.push(item); }
        }
    } else {
        let mut env_mut = env.clone();
        for item in items {
            if is_truthy(&apply_item_mut(item.clone(), pred, &mut env_mut)?) { out.push(item); }
        }
    }
    Ok(Val::arr(out))
}

/// `find(p1, p2, ...)` — keep items where *every* predicate is truthy.
/// Single-arg form is equivalent to `filter(p)`.
pub fn find(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    if args.is_empty() {
        return err!("find: requires at least one predicate");
    }
    let items = recv.into_vec().ok_or_else(|| EvalError("find: expected array".into()))?;
    let mut out = Vec::with_capacity(items.len());
    let all_bare = args.iter().all(|a| {
        !matches!(a, Arg::Pos(Expr::Lambda { .. }) | Arg::Named(_, Expr::Lambda { .. }))
    });
    if all_bare {
        let mut scratch = env.clone();
        let exprs: Vec<&Expr> = args.iter().map(|a| match a {
            Arg::Pos(e) | Arg::Named(_, e) => e,
        }).collect();
        'outer: for item in items {
            scratch.current = item.clone();
            for e in &exprs {
                if !is_truthy(&eval(e, &scratch)?) { continue 'outer; }
            }
            out.push(item);
        }
    } else {
        let mut env_mut = env.clone();
        'outer: for item in items {
            for p in args {
                if !is_truthy(&apply_item_mut(item.clone(), p, &mut env_mut)?) { continue 'outer; }
            }
            out.push(item);
        }
    }
    Ok(Val::arr(out))
}

pub fn map(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let mapper = args.first().ok_or_else(|| EvalError("map: requires mapper".into()))?;
    let items = recv.into_vec().ok_or_else(|| EvalError("map: expected array".into()))?;
    let mut env_mut = env.clone();
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        out.push(apply_item_mut(item, mapper, &mut env_mut)?);
    }
    Ok(Val::arr(out))
}

pub fn flat_map(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    Ok(flatten_val(map(recv, args, env)?, 1))
}

pub fn sort(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    // Columnar fast path: numeric arrays sort natively without Val tag dispatch.
    if args.is_empty() {
        match recv {
            Val::IntVec(a) => {
                let mut v = Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone());
                v.sort();
                return Ok(Val::int_vec(v));
            }
            Val::FloatVec(a) => {
                let mut v = Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone());
                v.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
                return Ok(Val::float_vec(v));
            }
            _ => {}
        }
        let mut items = recv.into_vec().ok_or_else(|| EvalError("sort: expected array".into()))?;
        items.sort_by(|x, y| cmp_vals(x, y));
        return Ok(Val::arr(items));
    }
    let mut items = recv.into_vec().ok_or_else(|| EvalError("sort: expected array".into()))?;
    // 2-param lambda comparator: sort(lambda a, b: a.field < b.field)
    if args.len() == 1 {
        if let Arg::Pos(Expr::Lambda { params, body }) | Arg::Named(_, Expr::Lambda { params, body }) = &args[0] {
            if params.len() == 2 {
                let p1 = params[0].clone();
                let p2 = params[1].clone();
                let mut env_mut = env.clone();
                let mut err_cell: Option<EvalError> = None;
                items.sort_by(|x, y| {
                    if err_cell.is_some() { return std::cmp::Ordering::Equal; }
                    let f1 = env_mut.push_lam(Some(&p1), x.clone());
                    let f2 = env_mut.push_lam(Some(&p2), y.clone());
                    let r = eval(body, &env_mut);
                    env_mut.pop_lam(f2);
                    env_mut.pop_lam(f1);
                    match r {
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

    // Schwartzian transform: eval each key expression once per item up
    // front, then sort by the precomputed keys.  With N items the naive
    // sort does O(N log N) comparisons, each evaluating keys twice — so
    // 2·N log N evals.  Precomputing brings this down to N evals plus
    // O(N log N) Val::cmp operations.
    let mut env_mut = env.clone();
    let mut keyed: Vec<(Vec<Val>, Val)> = Vec::with_capacity(items.len());
    for item in items {
        let mut ks = Vec::with_capacity(keys.len());
        let frame = env_mut.push_lam(None, item.clone());
        for (key_expr, _) in &keys {
            match eval(key_expr, &env_mut) {
                Ok(v)  => ks.push(v),
                Err(e) => { env_mut.pop_lam(frame); return Err(e); }
            }
        }
        env_mut.pop_lam(frame);
        keyed.push((ks, item));
    }
    keyed.sort_by(|(xk, _), (yk, _)| {
        for (i, (_, desc)) in keys.iter().enumerate() {
            let ord = cmp_vals(&xk[i], &yk[i]);
            if ord != std::cmp::Ordering::Equal {
                return if *desc { ord.reverse() } else { ord };
            }
        }
        std::cmp::Ordering::Equal
    });
    Ok(Val::arr(keyed.into_iter().map(|(_, v)| v).collect()))
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
        Val::IntVec(a) => {
            let mut v = Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone());
            v.reverse();
            Ok(Val::int_vec(v))
        }
        Val::FloatVec(a) => {
            let mut v = Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone());
            v.reverse();
            Ok(Val::float_vec(v))
        }
        Val::Str(s) => Ok(Val::Str(Arc::<str>::from(s.chars().rev().collect::<String>()))),
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
    let n = first_i64_arg(args, env)?;
    Ok(recv.get_index(n))
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
        let mut env_mut = env.clone();
        let mut out = Vec::new();
        for item in v {
            if !is_truthy(&apply_item_mut(item.clone(), pred, &mut env_mut)?) { out.push(item); }
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
    if items.is_empty() {
        return Ok(Val::Str(Arc::<str>::from("")));
    }
    // Fast path: skip the `Vec<String>` intermediate — write straight into
    // a single preallocated buffer.  Two sub-cases:
    //   - all items already `Val::Str`: use exact capacity + push_str.
    //   - general: `write!` via Display for primitives, fall back to
    //     `val_to_string` only for compound vals.
    if items.iter().all(|v| matches!(v, Val::Str(_))) {
        let total_len: usize = items.iter()
            .map(|v| if let Val::Str(s) = v { s.len() } else { 0 })
            .sum::<usize>()
            + sep.len() * (items.len() - 1);
        let mut out = String::with_capacity(total_len);
        let mut first = true;
        for v in &items {
            if !first { out.push_str(&sep); }
            first = false;
            if let Val::Str(s) = v { out.push_str(s); }
        }
        return Ok(Val::Str(Arc::<str>::from(out)));
    }
    use std::fmt::Write as _;
    let est_cap = items.len() * 8 + sep.len() * items.len();
    let mut out = String::with_capacity(est_cap);
    let mut first = true;
    for v in &items {
        if !first { out.push_str(&sep); }
        first = false;
        match v {
            Val::Str(s)   => out.push_str(s),
            Val::Int(n)   => { let _ = write!(out, "{}", n); }
            Val::Float(f) => { let _ = write!(out, "{}", f); }
            Val::Bool(b)  => out.push_str(if *b { "true" } else { "false" }),
            Val::Null     => out.push_str("null"),
            other         => out.push_str(&val_to_string(other)),
        }
    }
    Ok(Val::Str(Arc::<str>::from(out)))
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
        let mut env_mut = env.clone();
        let mut out = Vec::with_capacity(items.len());
        for (i, v) in items.into_iter().enumerate() {
            out.push(apply_item2_mut(Val::Int(i as i64), v, lam, &mut env_mut)?);
        }
        Ok(Val::arr(out))
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
    let mut env_mut = env.clone();
    let mut out = Vec::new();
    for item in items {
        if is_truthy(&apply_item_mut(item.clone(), pred, &mut env_mut)?) { out.push(item); }
        else { break; }
    }
    Ok(Val::arr(out))
}

pub fn dropwhile(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let pred  = args.first().ok_or_else(|| EvalError("dropwhile: requires predicate".into()))?;
    let items = recv.into_vec().ok_or_else(|| EvalError("dropwhile: expected array".into()))?;
    let mut env_mut = env.clone();
    let mut dropping = true;
    let mut out = Vec::new();
    for item in items {
        if dropping {
            if !is_truthy(&apply_item_mut(item.clone(), pred, &mut env_mut)?) { dropping = false; out.push(item); }
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
    let mut env_mut = env.clone();
    let mut running: Option<Val> = start;
    let mut out = Vec::new();
    for item in items {
        running = Some(if let Some(acc) = running {
            apply_item2_mut(acc, item, lam, &mut env_mut)?
        } else { item });
        out.push(running.clone().unwrap());
    }
    Ok(Val::arr(out))
}

pub fn partition(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let pred  = args.first().ok_or_else(|| EvalError("partition: requires predicate".into()))?;
    let items = recv.into_vec().ok_or_else(|| EvalError("partition: expected array".into()))?;
    let mut env_mut = env.clone();
    let (mut t, mut f) = (Vec::new(), Vec::new());
    for item in items {
        if is_truthy(&apply_item_mut(item.clone(), pred, &mut env_mut)?) { t.push(item); }
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
    let len  = arrs.iter().filter_map(|a| a.arr_len()).min().unwrap_or(0);
    Ok(Val::arr((0..len).map(|i| {
        Val::arr(arrs.iter().map(|a| a.get_index(i as i64)).collect())
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
    let len  = arrs.iter().filter_map(|a| a.arr_len()).max().unwrap_or(0);
    Ok(Val::arr((0..len).map(|i| {
        Val::arr(arrs.iter().map(|a| {
            if (i as usize) < a.arr_len().unwrap_or(0) {
                a.get_index(i as i64)
            } else { fill.clone() }
        }).collect())
    }).collect()))
}

pub fn global_product(args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let arrs: Result<Vec<Vec<Val>>, _> = args.iter()
        .map(|a| eval_pos(a, env).map(|v| v.into_vec().unwrap_or_default()))
        .collect();
    Ok(Val::arr(cartesian(&arrs?).into_iter().map(Val::arr).collect()))
}
