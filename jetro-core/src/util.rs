//! Shared utilities: `JsonView` borrowed scalar enum, arithmetic helpers,
//! comparison, truthiness, key extraction, and structural equality used
//! by the VM, builtins, and pipeline backends.

use indexmap::IndexMap;
use std::cmp::Ordering;
use std::sync::Arc;

use crate::data::context::EvalError;
use crate::data::value::Val;
use crate::parse::ast::BinOp;
use crate::parse::ast::KindType;

/// Borrowed scalar view — used by `ValueView` implementations so scalar
/// reads return a stack value without allocating a `Val`.
#[derive(Debug, Clone, Copy)]
pub enum JsonView<'a> {
    /// JSON `null` value.
    Null,
    /// JSON boolean value.
    Bool(bool),
    /// Signed 64-bit integer, used when the JSON number fits in `i64`.
    Int(i64),
    /// Unsigned 64-bit integer, used for large positive integers that don't fit in `i64`.
    UInt(u64),
    /// IEEE-754 double-precision floating point number.
    Float(f64),
    /// Borrowed string slice pointing directly into the source document or `Val` storage.
    Str(&'a str),
    /// The length of an array node, returned instead of materialising the array contents.
    ArrayLen(usize),
    /// The number of key-value pairs in an object node, returned without materialising the map.
    ObjectLen(usize),
}

impl<'a> JsonView<'a> {
    /// Convert a `Val` reference to a `JsonView` without cloning compound nodes;
    /// arrays and objects become length-only variants.
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

    /// Return `true` if this scalar is considered truthy by Jetro semantics
    /// (non-zero numbers, non-empty strings, `true`; `null` and empty are falsy).
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

/// Test structural equality between two `JsonView` scalars, including
/// cross-type numeric comparisons (e.g. `Int(3) == UInt(3)`).
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

/// Return the total ordering between two `JsonView` scalars, promoting
/// mixed integer/float pairs to `f64` for comparison.
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

/// Evaluate a binary comparison operator (`<`, `<=`, `>`, `>=`, `==`, `!=`)
/// between two `JsonView` scalars; incompatible types return `false`.
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

/// Return `true` if `v` is considered truthy according to Jetro semantics.
#[inline]
pub fn is_truthy(v: &Val) -> bool {
    JsonView::from_val(v).truthy()
}

/// Return `true` if `v` belongs to the `KindType` category used by the `type()` builtin.
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

/// Test scalar equality between two `Val` references via `json_vals_eq`.
/// **This is shallow** — compound values (`Val::Obj`, `Val::Arr`, vector
/// variants) compare as `false` even when structurally identical. Use
/// [`vals_deep_eq`] when structural equivalence is required.
#[inline]
pub fn vals_eq(a: &Val, b: &Val) -> bool {
    json_vals_eq(JsonView::from_val(a), JsonView::from_val(b))
}

/// Deep structural equality. Compares scalar values via `vals_eq` and
/// compound values element-wise. `Val::IntVec` / `FloatVec` / `StrVec` /
/// `StrSliceVec` are compared against `Val::Arr` element-wise (so a
/// promoted columnar form is equal to its boxed equivalent), and
/// `Val::ObjSmall` against `Val::Obj`.
///
/// Used by `rec` to detect fixpoint convergence — the previous shallow
/// check returned `false` for two identical `Val::Obj` and made `rec`
/// loop until exhaustion on every object input.
pub fn vals_deep_eq(a: &Val, b: &Val) -> bool {
    // Scalar fast path: pointer equality on Arc-wrapped compound types
    // is a sound shortcut (same Arc → same content).
    match (a, b) {
        (Val::Arr(x), Val::Arr(y)) if Arc::ptr_eq(x, y) => return true,
        (Val::Obj(x), Val::Obj(y)) if Arc::ptr_eq(x, y) => return true,
        (Val::IntVec(x), Val::IntVec(y)) if Arc::ptr_eq(x, y) => return true,
        (Val::FloatVec(x), Val::FloatVec(y)) if Arc::ptr_eq(x, y) => return true,
        _ => {}
    }
    if let (Some(av), Some(bv)) = (a.clone().into_vec(), b.clone().into_vec()) {
        if av.len() != bv.len() {
            return false;
        }
        return av.iter().zip(bv.iter()).all(|(x, y)| vals_deep_eq(x, y));
    }
    if let (Some(a_len), Some(b_len)) = (object_len(a), object_len(b)) {
        if a_len != b_len {
            return false;
        }
        return object_entries_match(a, |key, av| match b.get(key.as_ref()) {
            Some(bv) => vals_deep_eq(av, bv),
            None => false,
        });
    }
    // Mixed types or scalar: fall back to scalar equality.
    vals_eq(a, b)
}

#[inline]
fn object_len(value: &Val) -> Option<usize> {
    match value {
        Val::Obj(map) => Some(map.len()),
        Val::ObjSmall(pairs) => Some(pairs.len()),
        _ => None,
    }
}

fn object_entries_match<F>(value: &Val, mut f: F) -> bool
where
    F: FnMut(&Arc<str>, &Val) -> bool,
{
    match value {
        Val::Obj(map) => {
            for (key, item) in map.iter() {
                if !f(key, item) {
                    return false;
                }
            }
            true
        }
        Val::ObjSmall(pairs) => {
            for (key, item) in pairs.iter() {
                if !f(key, item) {
                    return false;
                }
            }
            true
        }
        _ => false,
    }
}

/// Compare two `Val` references and return their ordering.
#[inline]
pub fn cmp_vals(a: &Val, b: &Val) -> std::cmp::Ordering {
    json_cmp_vals(JsonView::from_val(a), JsonView::from_val(b))
}

/// Evaluate a binary comparison operator between two `Val` references.
#[inline]
pub fn cmp_vals_binop(a: &Val, op: BinOp, b: &Val) -> bool {
    json_cmp_binop(JsonView::from_val(a), op, JsonView::from_val(b))
}

/// Convert a `Val` to a string suitable for use as an object key or map key.
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

/// Convert a `Val` into a canonical key for structural equality buckets.
///
/// Unlike `val_to_key`, this is not object-key coercion: scalar types are
/// tagged, arrays are recursive, and object fields are sorted by key so
/// insertion order does not affect equality-style set operations.
pub fn val_to_structural_key(v: &Val) -> String {
    fn write_key(out: &mut String, v: &Val) {
        match v {
            Val::Null => out.push_str("z:null"),
            Val::Bool(b) => {
                out.push_str("b:");
                out.push_str(if *b { "1" } else { "0" });
            }
            Val::Int(n) => {
                out.push_str("n:");
                out.push_str(&(*n as f64).to_bits().to_string());
            }
            Val::Float(f) => {
                out.push_str("n:");
                let n = if *f == 0.0 { 0.0 } else { *f };
                out.push_str(&n.to_bits().to_string());
            }
            Val::Str(s) => {
                out.push_str("s:");
                out.push_str(&s.len().to_string());
                out.push(':');
                out.push_str(s);
            }
            Val::StrSlice(r) => {
                let s = r.as_str();
                out.push_str("s:");
                out.push_str(&s.len().to_string());
                out.push(':');
                out.push_str(s);
            }
            Val::Arr(a) => {
                out.push_str("a[");
                for item in a.iter() {
                    write_key(out, item);
                    out.push(',');
                }
                out.push(']');
            }
            Val::IntVec(a) => {
                out.push_str("a[");
                for item in a.iter() {
                    out.push_str("n:");
                    out.push_str(&(*item as f64).to_bits().to_string());
                    out.push(',');
                }
                out.push(']');
            }
            Val::FloatVec(a) => {
                out.push_str("a[");
                for item in a.iter() {
                    out.push_str("n:");
                    let n = if *item == 0.0 { 0.0 } else { *item };
                    out.push_str(&n.to_bits().to_string());
                    out.push(',');
                }
                out.push(']');
            }
            Val::StrVec(a) => {
                out.push_str("a[");
                for item in a.iter() {
                    out.push_str("s:");
                    out.push_str(&item.len().to_string());
                    out.push(':');
                    out.push_str(item);
                    out.push(',');
                }
                out.push(']');
            }
            Val::StrSliceVec(a) => {
                out.push_str("a[");
                for item in a.iter() {
                    let s = item.as_str();
                    out.push_str("s:");
                    out.push_str(&s.len().to_string());
                    out.push(':');
                    out.push_str(s);
                    out.push(',');
                }
                out.push(']');
            }
            Val::Obj(m) => {
                out.push_str("o{");
                let mut fields: Vec<(&str, &Val)> =
                    m.iter().map(|(k, v)| (k.as_ref(), v)).collect();
                fields.sort_unstable_by(|a, b| a.0.cmp(b.0));
                for (key, value) in fields {
                    out.push_str(&key.len().to_string());
                    out.push(':');
                    out.push_str(key);
                    out.push('=');
                    write_key(out, value);
                    out.push(',');
                }
                out.push('}');
            }
            Val::ObjSmall(pairs) => {
                out.push_str("o{");
                let mut fields: Vec<(&str, &Val)> =
                    pairs.iter().map(|(k, v)| (k.as_ref(), v)).collect();
                fields.sort_unstable_by(|a, b| a.0.cmp(b.0));
                for (key, value) in fields {
                    out.push_str(&key.len().to_string());
                    out.push(':');
                    out.push_str(key);
                    out.push('=');
                    write_key(out, value);
                    out.push(',');
                }
                out.push('}');
            }
            Val::ObjVec(d) => {
                out.push_str("a[");
                for row in 0..d.nrows() {
                    write_key(out, &d.row_val(row));
                    out.push(',');
                }
                out.push(']');
            }
        }
    }

    let mut out = String::new();
    write_key(&mut out, v);
    out
}

#[derive(Default)]
pub struct StructuralValueSet {
    buckets: std::collections::HashMap<String, Vec<Val>>,
}

impl StructuralValueSet {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buckets: std::collections::HashMap::with_capacity(capacity),
        }
    }

    pub fn insert(&mut self, value: &Val) -> bool {
        let key = val_to_structural_key(value);
        let bucket = self.buckets.entry(key).or_default();
        if bucket.iter().any(|seen| vals_deep_eq(seen, value)) {
            return false;
        }
        bucket.push(value.clone());
        true
    }

    pub fn contains(&self, value: &Val) -> bool {
        self.buckets
            .get(&val_to_structural_key(value))
            .map(|bucket| bucket.iter().any(|seen| vals_deep_eq(seen, value)))
            .unwrap_or(false)
    }
}

/// Render a `Val` as a `String`; compound types are serialised to JSON text.
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

/// Intern a string slice into an `Arc<str>` suitable for use as an `IndexMap` key.
#[inline]
pub fn val_key(s: &str) -> Arc<str> {
    Arc::from(s)
}

/// Borrow a `&str` view of any string-like `Val`. Used to make `+` accept
/// every cross-product of `Val::Str` and `Val::StrSlice` without
/// proliferating match arms.
#[inline]
fn as_str_view(v: &Val) -> Option<&str> {
    match v {
        Val::Str(s) => Some(s.as_ref()),
        Val::StrSlice(s) => Some(s.as_str()),
        _ => None,
    }
}

/// Add two `Val` operands; supports numeric addition, string concatenation
/// (any combination of `Val::Str` and `Val::StrSlice`), and array
/// concatenation. Returns an error for incompatible types.
pub fn add_vals(a: Val, b: Val) -> Result<Val, EvalError> {
    // Numeric: keep tight match on hot `Int`/`Float` pairs first so the
    // string-like path doesn't widen the dispatch fan-out for arithmetic.
    match (&a, &b) {
        (Val::Int(x), Val::Int(y)) => return Ok(Val::Int(x + y)),
        (Val::Float(x), Val::Float(y)) => return Ok(Val::Float(x + y)),
        (Val::Int(x), Val::Float(y)) => return Ok(Val::Float(*x as f64 + *y)),
        (Val::Float(x), Val::Int(y)) => return Ok(Val::Float(*x + *y as f64)),
        _ => {}
    }
    // String concat: any pairing of Str/StrSlice. Borrowed slices avoid
    // an Arc unwrap when the resulting string is a fresh allocation
    // anyway.
    if let (Some(xa), Some(xb)) = (as_str_view(&a), as_str_view(&b)) {
        let mut s = String::with_capacity(xa.len() + xb.len());
        s.push_str(xa);
        s.push_str(xb);
        return Ok(Val::Str(Arc::<str>::from(s)));
    }
    // Array concat.
    if let (Val::Arr(x), Val::Arr(y)) = (&a, &b) {
        let mut v = Arc::try_unwrap(x.clone()).unwrap_or_else(|a| (*a).clone());
        v.extend_from_slice(y);
        return Ok(Val::arr(v));
    }
    Err(EvalError("+ not supported between these types".into()))
}

/// Apply an integer operation `fi` or a float operation `ff` to two `Val` operands,
/// promoting mixed integer/float pairs to `f64`.
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

/// Flatten nested arrays up to `depth` levels; homogeneous element types are
/// collapsed to their optimised columnar representation (`IntVec`, `FloatVec`).
pub fn flatten_val(v: Val, depth: usize) -> Val {
    match v {
        Val::Arr(a) if depth > 0 => {
            let items = Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone());

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

        Val::IntVec(_) | Val::FloatVec(_) | Val::StrVec(_) => v,
        other => other,
    }
}

/// Zip two array-like `Val`s into an array of `[a_i, b_i]` pairs.
/// When `longest` is `true` the shorter side is padded with `fill`; otherwise
/// the result is truncated to the shorter length.
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

/// Compute the Cartesian product of multiple `Val` arrays, returning every
/// combination as a `Vec<Val>` row.
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

/// Return `true` if a dot-separated `path` resolves to a non-null value
/// within `v`, traversing nested objects recursively.
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

/// Recursively merge `other` into `base` for object values; non-object
/// values in `other` overwrite the corresponding entry in `base`.
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

/// Like `deep_merge`, but array-typed values at the same key are concatenated
/// rather than overwritten.
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

/// Construct a two-field object `{k1: v1, k2: v2}` with a pre-sized map.
pub fn obj2(k1: &str, v1: Val, k2: &str, v2: Val) -> Val {
    let mut m = IndexMap::with_capacity(2);
    m.insert(val_key(k1), v1);
    m.insert(val_key(k2), v2);
    Val::obj(m)
}
