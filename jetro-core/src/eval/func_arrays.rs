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

// `.unique` / `.distinct` LIFTED to composed::UniqueArr; shim composed::shims::unique.

// `.flatten` LIFTED to composed::FlattenDepth; shim composed::shims::flatten.

// `.compact` LIFTED to composed::Compact; shim in composed::shims::compact.

// ── Reorder ───────────────────────────────────────────────────────────────────

// `.reverse` LIFTED to composed::ReverseAny; shim composed::shims::reverse.

// ── Access ────────────────────────────────────────────────────────────────────

// `.first` / `.last` / `.nth` LIFTED to composed::{First, Last, NthAny};
// shims in composed::shims::{first, last, nth}.

// ── Mutation ──────────────────────────────────────────────────────────────────

// `.append` / `.prepend` LIFTED to composed::{Append, Prepend};
// shims in composed::shims::{append, prepend}.

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

// .diff / .intersect / .union LIFTED to composed::{Diff, Intersect,
// Union}; shims in composed::shims::{diff, intersect, union}.

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

// `.pairwise` LIFTED to composed::Pairwise; shim in composed::shims::pairwise.

// `.window` and `.chunk` (alias `.batch`) lifted to
// `pipeline::Stage::Window` / `pipeline::Stage::Chunk`.  Canonical impls
// in `pipeline::{window_apply, chunk_apply}`; dispatch shims in
// `eval/builtins.rs::{window_dispatch, chunk_dispatch}`.

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

// ── Window-style numeric ops ─────────────────────────────────────────────────
//
// Materialise the input array as `Vec<Option<f64>>` (None marks
// missing / non-numeric), apply the window transform, return a
// `Val::FloatVec` (when all positions are present) or `Val::Arr`
// with `Val::Null` for undefined positions (e.g. first `n-1`
// rolling values).

fn to_floats(recv: &Val) -> Result<Vec<Option<f64>>, EvalError> {
    match recv {
        Val::IntVec(a)   => Ok(a.iter().map(|n| Some(*n as f64)).collect()),
        Val::FloatVec(a) => Ok(a.iter().map(|f| Some(*f)).collect()),
        Val::Arr(a)      => Ok(a.iter().map(|v| match v {
            Val::Int(n)   => Some(*n as f64),
            Val::Float(f) => Some(*f),
            _             => None,
        }).collect()),
        _ => Err(EvalError("expected numeric array".into())),
    }
}

fn floats_to_val(out: Vec<Option<f64>>) -> Val {
    if out.iter().all(|v| v.is_some()) {
        Val::float_vec(out.into_iter().map(|v| v.unwrap()).collect())
    } else {
        Val::arr(out.into_iter().map(|v| match v {
            Some(f) => Val::Float(f),
            None    => Val::Null,
        }).collect())
    }
}

pub fn rolling_avg(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let n = first_i64_arg(args, env)? as usize;
    if n == 0 { return err!("rolling_avg: window must be > 0"); }
    let xs = to_floats(&recv)?;
    let mut out: Vec<Option<f64>> = Vec::with_capacity(xs.len());
    let mut sum: f64 = 0.0;
    let mut count: usize = 0;
    for (i, v) in xs.iter().enumerate() {
        if let Some(x) = v { sum += x; count += 1; }
        if i >= n {
            if let Some(old) = xs[i - n] { sum -= old; count -= 1; }
        }
        if i + 1 >= n && count > 0 {
            out.push(Some(sum / count as f64));
        } else {
            out.push(None);
        }
    }
    Ok(floats_to_val(out))
}

pub fn rolling_sum(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let n = first_i64_arg(args, env)? as usize;
    if n == 0 { return err!("rolling_sum: window must be > 0"); }
    let xs = to_floats(&recv)?;
    let mut out: Vec<Option<f64>> = Vec::with_capacity(xs.len());
    let mut sum: f64 = 0.0;
    for (i, v) in xs.iter().enumerate() {
        if let Some(x) = v { sum += x; }
        if i >= n {
            if let Some(old) = xs[i - n] { sum -= old; }
        }
        if i + 1 >= n { out.push(Some(sum)); } else { out.push(None); }
    }
    Ok(floats_to_val(out))
}

pub fn rolling_min(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let n = first_i64_arg(args, env)? as usize;
    if n == 0 { return err!("rolling_min: window must be > 0"); }
    let xs = to_floats(&recv)?;
    let mut out: Vec<Option<f64>> = Vec::with_capacity(xs.len());
    for i in 0..xs.len() {
        if i + 1 < n { out.push(None); continue; }
        let lo = i + 1 - n;
        let m = xs[lo..=i].iter().filter_map(|v| *v)
            .fold(f64::INFINITY, |a, b| a.min(b));
        out.push(if m.is_finite() { Some(m) } else { None });
    }
    Ok(floats_to_val(out))
}

pub fn rolling_max(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let n = first_i64_arg(args, env)? as usize;
    if n == 0 { return err!("rolling_max: window must be > 0"); }
    let xs = to_floats(&recv)?;
    let mut out: Vec<Option<f64>> = Vec::with_capacity(xs.len());
    for i in 0..xs.len() {
        if i + 1 < n { out.push(None); continue; }
        let lo = i + 1 - n;
        let m = xs[lo..=i].iter().filter_map(|v| *v)
            .fold(f64::NEG_INFINITY, |a, b| a.max(b));
        out.push(if m.is_finite() { Some(m) } else { None });
    }
    Ok(floats_to_val(out))
}

pub fn lag(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let n = if args.is_empty() { 1 } else { first_i64_arg(args, env)?.max(0) as usize };
    let xs = to_floats(&recv)?;
    let mut out: Vec<Option<f64>> = Vec::with_capacity(xs.len());
    for i in 0..xs.len() {
        out.push(if i >= n { xs[i - n] } else { None });
    }
    Ok(floats_to_val(out))
}

pub fn lead(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let n = if args.is_empty() { 1 } else { first_i64_arg(args, env)?.max(0) as usize };
    let xs = to_floats(&recv)?;
    let mut out: Vec<Option<f64>> = Vec::with_capacity(xs.len());
    for i in 0..xs.len() {
        let j = i + n;
        out.push(if j < xs.len() { xs[j] } else { None });
    }
    Ok(floats_to_val(out))
}

pub fn diff_window(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    let xs = to_floats(&recv)?;
    let mut out: Vec<Option<f64>> = Vec::with_capacity(xs.len());
    for i in 0..xs.len() {
        out.push(match (i.checked_sub(1).and_then(|j| xs[j]), xs[i]) {
            (Some(p), Some(c)) => Some(c - p),
            _ => None,
        });
    }
    Ok(floats_to_val(out))
}

pub fn pct_change(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    let xs = to_floats(&recv)?;
    let mut out: Vec<Option<f64>> = Vec::with_capacity(xs.len());
    for i in 0..xs.len() {
        out.push(match (i.checked_sub(1).and_then(|j| xs[j]), xs[i]) {
            (Some(p), Some(c)) if p != 0.0 => Some((c - p) / p),
            _ => None,
        });
    }
    Ok(floats_to_val(out))
}

pub fn cummax(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    let xs = to_floats(&recv)?;
    let mut out: Vec<Option<f64>> = Vec::with_capacity(xs.len());
    let mut best: Option<f64> = None;
    for v in xs.iter() {
        match (*v, best) {
            (Some(x), Some(b)) => { best = Some(x.max(b)); out.push(best); }
            (Some(x), None)    => { best = Some(x);        out.push(best); }
            (None, _)          => { out.push(best); }
        }
    }
    Ok(floats_to_val(out))
}

pub fn cummin(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    let xs = to_floats(&recv)?;
    let mut out: Vec<Option<f64>> = Vec::with_capacity(xs.len());
    let mut best: Option<f64> = None;
    for v in xs.iter() {
        match (*v, best) {
            (Some(x), Some(b)) => { best = Some(x.min(b)); out.push(best); }
            (Some(x), None)    => { best = Some(x);        out.push(best); }
            (None, _)          => { out.push(best); }
        }
    }
    Ok(floats_to_val(out))
}

// ── Index lookup family ──────────────────────────────────────────────────────
//
// Returning positions instead of items.

pub fn find_index(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    if args.is_empty() { return err!("find_index: requires a predicate"); }
    let items = recv.into_vec().ok_or_else(|| EvalError("find_index: expected array".into()))?;
    let mut env_mut = env.clone();
    for (i, item) in items.iter().enumerate() {
        let mut all = true;
        for p in args {
            if !is_truthy(&apply_item_mut(item.clone(), p, &mut env_mut)?) {
                all = false; break;
            }
        }
        if all { return Ok(Val::Int(i as i64)); }
    }
    Ok(Val::Null)
}

pub fn index_of_value(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    if args.is_empty() { return err!("index: requires a target value"); }
    let target = eval_pos(&args[0], env)?;
    let items = recv.into_vec().ok_or_else(|| EvalError("index: expected array".into()))?;
    for (i, item) in items.iter().enumerate() {
        if super::util::vals_eq(item, &target) {
            return Ok(Val::Int(i as i64));
        }
    }
    Ok(Val::Null)
}

pub fn indices_where(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    if args.is_empty() { return err!("indices_where: requires a predicate"); }
    let items = recv.into_vec().ok_or_else(|| EvalError("indices_where: expected array".into()))?;
    let mut env_mut = env.clone();
    let mut out: Vec<i64> = Vec::new();
    for (i, item) in items.iter().enumerate() {
        let mut all = true;
        for p in args {
            if !is_truthy(&apply_item_mut(item.clone(), p, &mut env_mut)?) {
                all = false; break;
            }
        }
        if all { out.push(i as i64); }
    }
    Ok(Val::int_vec(out))
}

pub fn indices_of(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    if args.is_empty() { return err!("indices_of: requires a target value"); }
    let target = eval_pos(&args[0], env)?;
    let items = recv.into_vec().ok_or_else(|| EvalError("indices_of: expected array".into()))?;
    let out: Vec<i64> = items.iter().enumerate()
        .filter(|(_, v)| super::util::vals_eq(v, &target))
        .map(|(i, _)| i as i64)
        .collect();
    Ok(Val::int_vec(out))
}

// ── max_by / min_by — single-pass argmax / argmin ────────────────────────────

fn extreme_by(recv: Val, args: &[Arg], env: &Env, want_max: bool, name: &str) -> Result<Val, EvalError> {
    if args.is_empty() { return err!("{}: requires a key expression", name); }
    let items = recv.into_vec().ok_or_else(|| EvalError(format!("{}: expected array", name)))?;
    if items.is_empty() { return Ok(Val::Null); }
    let key_arg = &args[0];
    let mut env_mut = env.clone();
    let mut best_idx: usize = 0;
    let mut best_key: Option<Val> = None;
    for (i, item) in items.iter().enumerate() {
        let k = apply_item_mut(item.clone(), key_arg, &mut env_mut)?;
        let take = match &best_key {
            None => true,
            Some(b) => {
                let ord = cmp_vals(&k, b);
                if want_max { ord == std::cmp::Ordering::Greater }
                else        { ord == std::cmp::Ordering::Less }
            }
        };
        if take { best_idx = i; best_key = Some(k); }
    }
    Ok(items.into_iter().nth(best_idx).unwrap_or(Val::Null))
}

pub fn max_by(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    extreme_by(recv, args, env, true, "max_by")
}

pub fn min_by(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    extreme_by(recv, args, env, false, "min_by")
}

pub fn zscore(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    let xs = to_floats(&recv)?;
    let nums: Vec<f64> = xs.iter().filter_map(|v| *v).collect();
    if nums.is_empty() {
        return Ok(floats_to_val(vec![None; xs.len()]));
    }
    let mean = nums.iter().sum::<f64>() / nums.len() as f64;
    let var  = nums.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / nums.len() as f64;
    let sd   = var.sqrt();
    let mut out: Vec<Option<f64>> = Vec::with_capacity(xs.len());
    for v in xs.iter() {
        out.push(match v {
            Some(x) if sd > 0.0 => Some((x - mean) / sd),
            Some(_)             => Some(0.0),
            None                => None,
        });
    }
    Ok(floats_to_val(out))
}
