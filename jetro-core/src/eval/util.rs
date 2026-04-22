//! Shared helpers used across eval / vm:
//!
//! - `cmp_vals`: total ordering on `Val` (null < bool < num < str <
//!   arr < obj), used by sort, min, max, top-N, distinct.  Unlike
//!   `PartialOrd`, this never returns `None` — it uses a lexicographic
//!   fallback when types are incomparable, which is essential for
//!   heap-based partial sorts.
//! - `add_vals` / `num_op`: numeric widening + arithmetic dispatch
//!   shared by the tree-walker and the VM so both semantics match.
//! - `val_key`: canonical string key for grouping / dedup.
//! - `flatten_val` / `zip_arrays` / `cartesian` / `deep_merge`:
//!   array / object combinators reused by builtins.
//! - `val_str` / `val_to_string`: coercion helpers for string
//!   methods and CSV emission.

use std::sync::Arc;
use indexmap::IndexMap;

use super::value::Val;
use super::EvalError;
use super::super::ast::KindType;

// ── Value predicates ──────────────────────────────────────────────────────────

#[inline]
pub fn is_truthy(v: &Val) -> bool {
    match v {
        Val::Null      => false,
        Val::Bool(b)   => *b,
        Val::Int(n)    => *n != 0,
        Val::Float(f)  => *f != 0.0,
        Val::Str(s)    => !s.is_empty(),
        Val::Arr(a)    => !a.is_empty(),
        Val::Obj(m)    => !m.is_empty(),
    }
}

#[inline]
pub fn kind_matches(v: &Val, ty: KindType) -> bool {
    matches!((v, ty),
        (Val::Null,         KindType::Null)   |
        (Val::Bool(_),      KindType::Bool)   |
        (Val::Int(_),       KindType::Number) |
        (Val::Float(_),     KindType::Number) |
        (Val::Str(_),       KindType::Str)    |
        (Val::Arr(_),       KindType::Array)  |
        (Val::Obj(_),       KindType::Object)
    )
}

#[inline]
pub fn vals_eq(a: &Val, b: &Val) -> bool {
    match (a, b) {
        (Val::Null,     Val::Null)     => true,
        (Val::Bool(x),  Val::Bool(y))  => x == y,
        (Val::Str(x),   Val::Str(y))   => x == y,
        (Val::Int(x),   Val::Int(y))   => x == y,
        (Val::Float(x), Val::Float(y)) => x == y,
        (Val::Int(x),   Val::Float(y)) => (*x as f64) == *y,
        (Val::Float(x), Val::Int(y))   => *x == (*y as f64),
        _ => false,
    }
}

#[inline]
pub fn cmp_vals(a: &Val, b: &Val) -> std::cmp::Ordering {
    match (a, b) {
        (Val::Int(x),   Val::Int(y))   => x.cmp(y),
        (Val::Float(x), Val::Float(y)) => x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal),
        (Val::Int(x),   Val::Float(y)) => (*x as f64).partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal),
        (Val::Float(x), Val::Int(y))   => x.partial_cmp(&(*y as f64)).unwrap_or(std::cmp::Ordering::Equal),
        (Val::Str(x),   Val::Str(y))   => x.cmp(y),
        (Val::Bool(x),  Val::Bool(y))  => x.cmp(y),
        _ => std::cmp::Ordering::Equal,
    }
}

// ── Value conversions ─────────────────────────────────────────────────────────

/// Canonical string key for use in HashSets / dedup (never allocates for Str).
#[inline]
pub fn val_to_key(v: &Val) -> String {
    match v {
        Val::Str(s)    => s.to_string(),
        Val::Int(n)    => n.to_string(),
        Val::Float(f)  => f.to_string(),
        Val::Bool(b)   => b.to_string(),
        Val::Null      => "null".to_string(),
        other          => val_to_string(other),
    }
}

#[inline]
pub fn val_to_string(v: &Val) -> String {
    match v {
        Val::Str(s)    => s.to_string(),
        Val::Int(n)    => n.to_string(),
        Val::Float(f)  => f.to_string(),
        Val::Bool(b)   => b.to_string(),
        Val::Null      => "null".to_string(),
        other          => {
            let sv: serde_json::Value = other.clone().into();
            serde_json::to_string(&sv).unwrap_or_default()
        }
    }
}

// ── Constructors ──────────────────────────────────────────────────────────────

#[inline] pub fn val_int(n: i64)  -> Val { Val::Int(n) }
#[inline] pub fn val_float(f: f64) -> Val { Val::Float(f) }
#[inline] pub fn val_str(s: &str) -> Val { Val::Str(Arc::from(s)) }
#[inline] pub fn val_key(s: &str) -> Arc<str> { Arc::from(s) }

// ── Arithmetic ────────────────────────────────────────────────────────────────

pub fn add_vals(a: Val, b: Val) -> Result<Val, EvalError> {
    match (a, b) {
        (Val::Int(x),   Val::Int(y))   => Ok(Val::Int(x + y)),
        (Val::Float(x), Val::Float(y)) => Ok(Val::Float(x + y)),
        (Val::Int(x),   Val::Float(y)) => Ok(Val::Float(x as f64 + y)),
        (Val::Float(x), Val::Int(y))   => Ok(Val::Float(x + y as f64)),
        (Val::Str(x),   Val::Str(y))   => Ok(Val::Str(Arc::<str>::from(format!("{}{}", x, y)))),
        (Val::Arr(x), Val::Arr(y)) => {
            let mut v = Arc::try_unwrap(x).unwrap_or_else(|a| (*a).clone());
            v.extend_from_slice(&y);
            Ok(Val::arr(v))
        }
        _ => Err(EvalError("+ not supported between these types".into())),
    }
}

pub fn num_op<Fi, Ff>(a: Val, b: Val, fi: Fi, ff: Ff) -> Result<Val, EvalError>
where
    Fi: Fn(i64, i64) -> i64,
    Ff: Fn(f64, f64) -> f64,
{
    match (a, b) {
        (Val::Int(x),   Val::Int(y))   => Ok(Val::Int(fi(x, y))),
        (Val::Float(x), Val::Float(y)) => Ok(Val::Float(ff(x, y))),
        (Val::Int(x),   Val::Float(y)) => Ok(Val::Float(ff(x as f64, y))),
        (Val::Float(x), Val::Int(y))   => Ok(Val::Float(ff(x, y as f64))),
        _ => Err(EvalError("arithmetic on non-numbers".into())),
    }
}

// ── Array helpers ─────────────────────────────────────────────────────────────

pub fn flatten_val(v: Val, depth: usize) -> Val {
    match v {
        Val::Arr(a) if depth > 0 => {
            let mut out = Vec::new();
            let items = Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone());
            for item in items {
                match item {
                    Val::Arr(_) => if let Val::Arr(inner) = flatten_val(item, depth - 1) {
                        out.extend(Arc::try_unwrap(inner).unwrap_or_else(|a| (*a).clone()));
                    },
                    other => out.push(other),
                }
            }
            Val::arr(out)
        }
        other => other,
    }
}

pub fn zip_arrays(a: Val, b: Val, longest: bool, fill: Val) -> Result<Val, EvalError> {
    let av = a.as_array().map(|a| a.to_vec()).unwrap_or_default();
    let bv = b.as_array().map(|b| b.to_vec()).unwrap_or_default();
    let len = if longest { av.len().max(bv.len()) } else { av.len().min(bv.len()) };
    Ok(Val::arr((0..len).map(|i| Val::arr(vec![
        av.get(i).cloned().unwrap_or_else(|| fill.clone()),
        bv.get(i).cloned().unwrap_or_else(|| fill.clone()),
    ])).collect()))
}

pub fn cartesian(arrays: &[Vec<Val>]) -> Vec<Vec<Val>> {
    arrays.iter().fold(vec![vec![]], |acc, arr| {
        acc.into_iter().flat_map(|prefix| {
            arr.iter().map(move |item| {
                let mut row = prefix.clone();
                row.push(item.clone());
                row
            }).collect::<Vec<_>>()
        }).collect()
    })
}

// ── Field existence ───────────────────────────────────────────────────────────

pub fn field_exists_nested(v: &Val, path: &str) -> bool {
    let mut parts = path.splitn(2, '.');
    let first = parts.next().unwrap_or("");
    match (v.get(first), parts.next()) {
        (Some(v), _) if v.is_null() => false,
        (Some(_), None)             => true,
        (Some(child), Some(rest))   => field_exists_nested(child, rest),
        (None, _)                   => false,
    }
}

// ── Deep merge ────────────────────────────────────────────────────────────────

pub fn deep_merge(base: Val, other: Val) -> Val {
    match (base, other) {
        (Val::Obj(bm), Val::Obj(om)) => {
            let mut map = Arc::try_unwrap(bm).unwrap_or_else(|m| (*m).clone());
            for (k, v) in Arc::try_unwrap(om).unwrap_or_else(|m| (*m).clone()) {
                let existing = map.shift_remove(&k);
                map.insert(k, match existing {
                    Some(e) => deep_merge(e, v),
                    None    => v,
                });
            }
            Val::obj(map)
        }
        (_, other) => other,
    }
}

/// Deep-merge where arrays concatenate instead of replace.  Used by the
/// `...**` spread operator.  Objects recurse, arrays concat, scalars rhs-wins.
pub fn deep_merge_concat(base: Val, other: Val) -> Val {
    match (base, other) {
        (Val::Obj(bm), Val::Obj(om)) => {
            let mut map = Arc::try_unwrap(bm).unwrap_or_else(|m| (*m).clone());
            for (k, v) in Arc::try_unwrap(om).unwrap_or_else(|m| (*m).clone()) {
                let existing = map.shift_remove(&k);
                map.insert(k, match existing {
                    Some(e) => deep_merge_concat(e, v),
                    None    => v,
                });
            }
            Val::obj(map)
        }
        (Val::Arr(ba), Val::Arr(oa)) => {
            let mut a = Arc::try_unwrap(ba).unwrap_or_else(|a| (*a).clone());
            for v in Arc::try_unwrap(oa).unwrap_or_else(|a| (*a).clone()) { a.push(v); }
            Val::arr(a)
        }
        (_, other) => other,
    }
}

// ── Object building helpers ───────────────────────────────────────────────────

/// Build a Val::Obj with two string-key entries (common pattern for itertools output).
pub fn obj2(k1: &str, v1: Val, k2: &str, v2: Val) -> Val {
    let mut m = IndexMap::with_capacity(2);
    m.insert(val_key(k1), v1);
    m.insert(val_key(k2), v2);
    Val::obj(m)
}
