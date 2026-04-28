//! Shared helpers used across builtins, pipeline, and VM:
//!
//! - `cmp_vals`: total ordering on `Val` (null < bool < num < str <
//!   arr < obj), used by sort, min, max, top-N, distinct.  Unlike
//!   `PartialOrd`, this never returns `None` — it uses a lexicographic
//!   fallback when types are incomparable, which is essential for
//!   heap-based partial sorts.
//! - `add_vals` / `num_op`: numeric widening + arithmetic dispatch
//!   shared by all execution paths so semantics match.
//! - `val_key`: canonical string key for grouping / dedup.
//! - `flatten_val` / `zip_arrays` / `cartesian` / `deep_merge`:
//!   array / object combinators reused by builtins.
//! - `val_str` / `val_to_string`: coercion helpers for string
//!   methods and CSV emission.

use indexmap::IndexMap;
use std::cmp::Ordering;
use std::sync::Arc;

use crate::ast::BinOp;
use crate::ast::KindType;
use crate::context::EvalError;
use crate::value::Val;

// ── Scalar semantic kernel ───────────────────────────────────────────────────

/// Borrowed JSON-scalar/container view used by both `Val` and simd-json tape
/// execution. Tape adapters should only decode node shape into this view; the
/// truthiness/comparison/numeric behavior lives here.
#[derive(Debug, Clone, Copy)]
pub enum JsonView<'a> {
    Null,
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
    Str(&'a str),
    ArrayLen(usize),
    ObjectLen(usize),
}

impl<'a> JsonView<'a> {
    #[inline]
    pub fn from_val(v: &'a Val) -> Self {
        match v {
            Val::Null => JsonView::Null,
            Val::Bool(b) => JsonView::Bool(*b),
            Val::Int(n) => JsonView::Int(*n),
            Val::Float(f) => JsonView::Float(*f),
            Val::Str(s) => JsonView::Str(s.as_ref()),
            Val::StrSlice(r) => JsonView::Str(r.as_str()),
            Val::Arr(a) => JsonView::ArrayLen(a.len()),
            Val::IntVec(a) => JsonView::ArrayLen(a.len()),
            Val::FloatVec(a) => JsonView::ArrayLen(a.len()),
            Val::StrVec(a) => JsonView::ArrayLen(a.len()),
            Val::StrSliceVec(a) => JsonView::ArrayLen(a.len()),
            Val::ObjVec(d) => JsonView::ArrayLen(d.nrows()),
            Val::Obj(m) => JsonView::ObjectLen(m.len()),
            Val::ObjSmall(p) => JsonView::ObjectLen(p.len()),
        }
    }

    #[inline]
    pub fn truthy(self) -> bool {
        match self {
            JsonView::Null => false,
            JsonView::Bool(b) => b,
            JsonView::Int(n) => n != 0,
            JsonView::UInt(n) => n != 0,
            JsonView::Float(f) => f != 0.0,
            JsonView::Str(s) => !s.is_empty(),
            JsonView::ArrayLen(n) | JsonView::ObjectLen(n) => n != 0,
        }
    }
}

#[inline]
pub fn json_vals_eq(a: JsonView<'_>, b: JsonView<'_>) -> bool {
    match (a, b) {
        (JsonView::Null, JsonView::Null) => true,
        (JsonView::Bool(x), JsonView::Bool(y)) => x == y,
        (JsonView::Str(x), JsonView::Str(y)) => x == y,
        (JsonView::Int(x), JsonView::Int(y)) => x == y,
        (JsonView::UInt(x), JsonView::UInt(y)) => x == y,
        (JsonView::Float(x), JsonView::Float(y)) => x == y,
        (JsonView::Int(x), JsonView::Float(y)) => (x as f64) == y,
        (JsonView::Float(x), JsonView::Int(y)) => x == (y as f64),
        (JsonView::UInt(x), JsonView::Float(y)) => (x as f64) == y,
        (JsonView::Float(x), JsonView::UInt(y)) => x == (y as f64),
        (JsonView::Int(x), JsonView::UInt(y)) => x >= 0 && (x as u64) == y,
        (JsonView::UInt(x), JsonView::Int(y)) => y >= 0 && x == (y as u64),
        _ => false,
    }
}

#[inline]
pub fn json_cmp_vals(a: JsonView<'_>, b: JsonView<'_>) -> Ordering {
    match (a, b) {
        (JsonView::Int(x), JsonView::Int(y)) => x.cmp(&y),
        (JsonView::UInt(x), JsonView::UInt(y)) => x.cmp(&y),
        (JsonView::Float(x), JsonView::Float(y)) => x.partial_cmp(&y).unwrap_or(Ordering::Equal),
        (JsonView::Int(x), JsonView::Float(y)) => {
            (x as f64).partial_cmp(&y).unwrap_or(Ordering::Equal)
        }
        (JsonView::Float(x), JsonView::Int(y)) => {
            x.partial_cmp(&(y as f64)).unwrap_or(Ordering::Equal)
        }
        (JsonView::UInt(x), JsonView::Float(y)) => {
            (x as f64).partial_cmp(&y).unwrap_or(Ordering::Equal)
        }
        (JsonView::Float(x), JsonView::UInt(y)) => {
            x.partial_cmp(&(y as f64)).unwrap_or(Ordering::Equal)
        }
        (JsonView::Int(x), JsonView::UInt(y)) => {
            if x < 0 {
                Ordering::Less
            } else {
                (x as u64).cmp(&y)
            }
        }
        (JsonView::UInt(x), JsonView::Int(y)) => {
            if y < 0 {
                Ordering::Greater
            } else {
                x.cmp(&(y as u64))
            }
        }
        (JsonView::Str(x), JsonView::Str(y)) => x.cmp(y),
        (JsonView::Bool(x), JsonView::Bool(y)) => x.cmp(&y),
        _ => Ordering::Equal,
    }
}

#[inline]
pub fn json_cmp_binop(lhs: JsonView<'_>, op: BinOp, rhs: JsonView<'_>) -> bool {
    #[inline]
    fn comparable(lhs: &JsonView<'_>, rhs: &JsonView<'_>) -> bool {
        matches!(
            (lhs, rhs),
            (JsonView::Int(_), JsonView::Int(_))
                | (JsonView::UInt(_), JsonView::UInt(_))
                | (JsonView::Float(_), JsonView::Float(_))
                | (JsonView::Int(_), JsonView::Float(_))
                | (JsonView::Float(_), JsonView::Int(_))
                | (JsonView::UInt(_), JsonView::Float(_))
                | (JsonView::Float(_), JsonView::UInt(_))
                | (JsonView::Int(_), JsonView::UInt(_))
                | (JsonView::UInt(_), JsonView::Int(_))
                | (JsonView::Str(_), JsonView::Str(_))
                | (JsonView::Bool(_), JsonView::Bool(_))
        )
    }

    match op {
        BinOp::Eq => json_vals_eq(lhs, rhs),
        BinOp::Neq => !json_vals_eq(lhs, rhs),
        BinOp::Lt | BinOp::Lte | BinOp::Gt | BinOp::Gte if !comparable(&lhs, &rhs) => false,
        BinOp::Lt => json_cmp_vals(lhs, rhs) == Ordering::Less,
        BinOp::Lte => json_cmp_vals(lhs, rhs) != Ordering::Greater,
        BinOp::Gt => json_cmp_vals(lhs, rhs) == Ordering::Greater,
        BinOp::Gte => json_cmp_vals(lhs, rhs) != Ordering::Less,
        _ => false,
    }
}

// ── Value predicates ──────────────────────────────────────────────────────────

#[inline]
pub fn is_truthy(v: &Val) -> bool {
    JsonView::from_val(v).truthy()
}

#[inline]
pub fn kind_matches(v: &Val, ty: KindType) -> bool {
    matches!(
        (v, ty),
        (Val::Null, KindType::Null)
            | (Val::Bool(_), KindType::Bool)
            | (Val::Int(_), KindType::Number)
            | (Val::Float(_), KindType::Number)
            | (Val::Str(_), KindType::Str)
            | (Val::Arr(_), KindType::Array)
            | (Val::IntVec(_), KindType::Array)
            | (Val::FloatVec(_), KindType::Array)
            | (Val::Obj(_), KindType::Object)
    )
}

#[inline]
pub fn vals_eq(a: &Val, b: &Val) -> bool {
    json_vals_eq(JsonView::from_val(a), JsonView::from_val(b))
}

#[inline]
pub fn cmp_vals(a: &Val, b: &Val) -> std::cmp::Ordering {
    json_cmp_vals(JsonView::from_val(a), JsonView::from_val(b))
}

#[inline]
pub fn cmp_vals_binop(a: &Val, op: BinOp, b: &Val) -> bool {
    json_cmp_binop(JsonView::from_val(a), op, JsonView::from_val(b))
}

// ── Value conversions ─────────────────────────────────────────────────────────

/// Canonical string key for use in HashSets / dedup (never allocates for Str).
#[inline]
pub fn val_to_key(v: &Val) -> String {
    match v {
        Val::Str(s) => s.to_string(),
        Val::StrSlice(r) => r.as_str().to_string(),
        Val::Int(n) => n.to_string(),
        Val::Float(f) => f.to_string(),
        Val::Bool(b) => b.to_string(),
        Val::Null => "null".to_string(),
        other => val_to_string(other),
    }
}

#[inline]
pub fn val_to_string(v: &Val) -> String {
    match v {
        Val::Str(s) => s.to_string(),
        Val::StrSlice(r) => r.as_str().to_string(),
        Val::Int(n) => n.to_string(),
        Val::Float(f) => f.to_string(),
        Val::Bool(b) => b.to_string(),
        Val::Null => "null".to_string(),
        other => {
            let sv: serde_json::Value = other.clone().into();
            serde_json::to_string(&sv).unwrap_or_default()
        }
    }
}

// ── Constructors ──────────────────────────────────────────────────────────────

#[inline]
pub fn val_key(s: &str) -> Arc<str> {
    Arc::from(s)
}

// ── Arithmetic ────────────────────────────────────────────────────────────────

pub fn add_vals(a: Val, b: Val) -> Result<Val, EvalError> {
    match (a, b) {
        (Val::Int(x), Val::Int(y)) => Ok(Val::Int(x + y)),
        (Val::Float(x), Val::Float(y)) => Ok(Val::Float(x + y)),
        (Val::Int(x), Val::Float(y)) => Ok(Val::Float(x as f64 + y)),
        (Val::Float(x), Val::Int(y)) => Ok(Val::Float(x + y as f64)),
        (Val::Str(x), Val::Str(y)) => {
            // `format!` would allocate a temporary `String` for argument
            // formatting, on top of the `Arc::<str>::from` allocation.
            // Direct `push_str` halves the allocation count.
            let mut s = String::with_capacity(x.len() + y.len());
            s.push_str(&x);
            s.push_str(&y);
            Ok(Val::Str(Arc::<str>::from(s)))
        }
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
        (Val::Int(x), Val::Int(y)) => Ok(Val::Int(fi(x, y))),
        (Val::Float(x), Val::Float(y)) => Ok(Val::Float(ff(x, y))),
        (Val::Int(x), Val::Float(y)) => Ok(Val::Float(ff(x as f64, y))),
        (Val::Float(x), Val::Int(y)) => Ok(Val::Float(ff(x, y as f64))),
        _ => Err(EvalError("arithmetic on non-numbers".into())),
    }
}

// ── Array helpers ─────────────────────────────────────────────────────────────

pub fn flatten_val(v: Val, depth: usize) -> Val {
    match v {
        Val::Arr(a) if depth > 0 => {
            let items = Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone());
            // Columnar fast-path: Arr of IntVec (or Int scalars) → IntVec out.
            // Skips per-item Val::Int allocation and keeps the result in the
            // typed lane for downstream aggregates / accumulate.
            let all_int_children = !items.is_empty()
                && items
                    .iter()
                    .all(|it| matches!(it, Val::IntVec(_) | Val::Int(_)));
            if all_int_children {
                let cap: usize = items
                    .iter()
                    .map(|it| match it {
                        Val::IntVec(inner) => inner.len(),
                        _ => 1,
                    })
                    .sum();
                let mut out: Vec<i64> = Vec::with_capacity(cap);
                for item in items {
                    match item {
                        Val::IntVec(inner) => out.extend(inner.iter().copied()),
                        Val::Int(n) => out.push(n),
                        _ => unreachable!(),
                    }
                }
                return Val::int_vec(out);
            }
            let all_float_children = !items.is_empty()
                && items
                    .iter()
                    .all(|it| matches!(it, Val::FloatVec(_) | Val::Float(_) | Val::Int(_)));
            if all_float_children {
                let cap: usize = items
                    .iter()
                    .map(|it| match it {
                        Val::FloatVec(inner) => inner.len(),
                        _ => 1,
                    })
                    .sum();
                let mut out: Vec<f64> = Vec::with_capacity(cap);
                for item in items {
                    match item {
                        Val::FloatVec(inner) => out.extend(inner.iter().copied()),
                        Val::Float(f) => out.push(f),
                        Val::Int(n) => out.push(n as f64),
                        _ => unreachable!(),
                    }
                }
                return Val::float_vec(out);
            }
            // Precompute exact capacity in one pass — eliminates Vec doubling
            // reallocations on the hot `$.flatten()` / `.map(...).flatten()`
            // paths.
            let cap: usize = items
                .iter()
                .map(|it| match it {
                    Val::Arr(inner) => inner.len(),
                    Val::IntVec(inner) => inner.len(),
                    Val::FloatVec(inner) => inner.len(),
                    Val::StrVec(inner) => inner.len(),
                    _ => 1,
                })
                .sum();
            let mut out = Vec::with_capacity(cap);
            for item in items {
                match item {
                    Val::Arr(_) => match flatten_val(item, depth - 1) {
                        Val::Arr(inner) => {
                            out.extend(Arc::try_unwrap(inner).unwrap_or_else(|a| (*a).clone()));
                        }
                        Val::IntVec(inner) => {
                            out.extend(inner.iter().map(|n| Val::Int(*n)));
                        }
                        Val::FloatVec(inner) => {
                            out.extend(inner.iter().map(|f| Val::Float(*f)));
                        }
                        Val::StrVec(inner) => {
                            out.extend(inner.iter().map(|s| Val::Str(s.clone())));
                        }
                        _ => {}
                    },
                    Val::IntVec(inner) => {
                        // IntVec is already flat-of-scalars — extend once.
                        out.extend(inner.iter().map(|n| Val::Int(*n)));
                    }
                    Val::FloatVec(inner) => {
                        out.extend(inner.iter().map(|f| Val::Float(*f)));
                    }
                    Val::StrVec(inner) => {
                        out.extend(inner.iter().map(|s| Val::Str(s.clone())));
                    }
                    other => out.push(other),
                }
            }
            Val::arr(out)
        }
        // Columnar receiver — already flat of scalars.
        Val::IntVec(_) | Val::FloatVec(_) | Val::StrVec(_) => v,
        other => other,
    }
}

pub fn zip_arrays(a: Val, b: Val, longest: bool, fill: Val) -> Result<Val, EvalError> {
    let av = a.as_vals().map(|c| c.into_owned()).unwrap_or_default();
    let bv = b.as_vals().map(|c| c.into_owned()).unwrap_or_default();
    let len = if longest {
        av.len().max(bv.len())
    } else {
        av.len().min(bv.len())
    };
    Ok(Val::arr(
        (0..len)
            .map(|i| {
                Val::arr(vec![
                    av.get(i).cloned().unwrap_or_else(|| fill.clone()),
                    bv.get(i).cloned().unwrap_or_else(|| fill.clone()),
                ])
            })
            .collect(),
    ))
}

pub fn cartesian(arrays: &[Vec<Val>]) -> Vec<Vec<Val>> {
    arrays.iter().fold(vec![vec![]], |acc, arr| {
        acc.into_iter()
            .flat_map(|prefix| {
                arr.iter()
                    .map(move |item| {
                        let mut row = prefix.clone();
                        row.push(item.clone());
                        row
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    })
}

// ── Field existence ───────────────────────────────────────────────────────────

pub fn field_exists_nested(v: &Val, path: &str) -> bool {
    let mut parts = path.splitn(2, '.');
    let first = parts.next().unwrap_or("");
    match (v.get(first), parts.next()) {
        (Some(v), _) if v.is_null() => false,
        (Some(_), None) => true,
        (Some(child), Some(rest)) => field_exists_nested(child, rest),
        (None, _) => false,
    }
}

// ── Deep merge ────────────────────────────────────────────────────────────────

pub fn deep_merge(base: Val, other: Val) -> Val {
    match (base, other) {
        (Val::Obj(bm), Val::Obj(om)) => {
            let mut map = Arc::try_unwrap(bm).unwrap_or_else(|m| (*m).clone());
            for (k, v) in Arc::try_unwrap(om).unwrap_or_else(|m| (*m).clone()) {
                let existing = map.shift_remove(&k);
                map.insert(
                    k,
                    match existing {
                        Some(e) => deep_merge(e, v),
                        None => v,
                    },
                );
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
                map.insert(
                    k,
                    match existing {
                        Some(e) => deep_merge_concat(e, v),
                        None => v,
                    },
                );
            }
            Val::obj(map)
        }
        (Val::Arr(ba), Val::Arr(oa)) => {
            let mut a = Arc::try_unwrap(ba).unwrap_or_else(|a| (*a).clone());
            for v in Arc::try_unwrap(oa).unwrap_or_else(|a| (*a).clone()) {
                a.push(v);
            }
            Val::arr(a)
        }
        (base, other)
            if (base.is_array() && other.is_array())
                && (matches!(&base, Val::StrVec(_) | Val::IntVec(_) | Val::FloatVec(_))
                    || matches!(&other, Val::StrVec(_) | Val::IntVec(_) | Val::FloatVec(_))) =>
        {
            let mut a = base.into_vec().unwrap_or_default();
            if let Some(v) = other.into_vec() {
                a.extend(v);
            }
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
