use crate::data::value::Val;
use std::sync::Arc;

/// Returns the logical length of an array, object, or string (char count), or `None` for scalars.
#[inline]
pub fn len_apply(recv: &Val) -> Option<Val> {
    let n = match recv {
        Val::Arr(a) => a.len(),
        Val::IntVec(a) => a.len(),
        Val::FloatVec(a) => a.len(),
        Val::StrVec(a) => a.len(),
        Val::StrSliceVec(a) => a.len(),
        Val::Obj(m) => m.len(),
        Val::Str(s) => s.chars().count(),
        Val::StrSlice(r) => r.as_str().chars().count(),
        _ => return None,
    };
    Some(Val::Int(n as i64))
}

/// Removes all `Val::Null` elements from an array.
#[inline]
pub fn compact_apply(recv: &Val) -> Option<Val> {
    let items_cow = recv.as_vals()?;
    let kept: Vec<Val> = items_cow
        .iter()
        .filter(|v| !matches!(v, Val::Null))
        .cloned()
        .collect();
    Some(Val::arr(kept))
}

/// Recursively flattens nested arrays up to `depth` levels deep.
#[inline]
pub fn flatten_depth_apply(recv: &Val, depth: usize) -> Option<Val> {
    if matches!(recv, Val::Arr(_)) {
        Some(crate::util::flatten_val(recv.clone(), depth))
    } else {
        None
    }
}

/// Reverses any sequence type: arrays, typed vectors, and strings (by Unicode codepoints).
#[inline]
pub fn reverse_any_apply(recv: &Val) -> Option<Val> {
    Some(match recv {
        Val::Arr(a) => {
            let mut v: Vec<Val> = a.as_ref().clone();
            v.reverse();
            Val::arr(v)
        }
        Val::IntVec(a) => {
            let mut v: Vec<i64> = a.as_ref().clone();
            v.reverse();
            Val::int_vec(v)
        }
        Val::FloatVec(a) => {
            let mut v: Vec<f64> = a.as_ref().clone();
            v.reverse();
            Val::float_vec(v)
        }
        Val::StrVec(a) => {
            let mut v: Vec<Arc<str>> = a.as_ref().clone();
            v.reverse();
            Val::str_vec(v)
        }
        Val::Str(s) => Val::Str(Arc::<str>::from(s.chars().rev().collect::<String>())),
        Val::StrSlice(r) => Val::Str(Arc::<str>::from(
            r.as_str().chars().rev().collect::<String>(),
        )),
        _ => return None,
    })
}

/// Removes duplicate elements from an array, preserving first-seen order.
#[inline]
pub fn unique_arr_apply(recv: &Val) -> Option<Val> {
    let items_cow = recv.as_vals()?;
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let kept: Vec<Val> = items_cow
        .iter()
        .filter(|v| seen.insert(crate::util::val_to_key(v)))
        .cloned()
        .collect();
    Some(Val::arr(kept))
}

/// Extracts numeric values from any array-like `Val` as `Option<f64>`, preserving nulls as `None`.
fn numeric_options(recv: &Val) -> Option<Vec<Option<f64>>> {
    match recv {
        Val::IntVec(a) => Some(a.iter().map(|n| Some(*n as f64)).collect()),
        Val::FloatVec(a) => Some(a.iter().map(|f| Some(*f)).collect()),
        Val::Arr(a) => Some(
            a.iter()
                .map(|v| match v {
                    Val::Int(n) => Some(*n as f64),
                    Val::Float(f) => Some(*f),
                    _ => None,
                })
                .collect(),
        ),
        _ => None,
    }
}

/// Converts a `Vec<Option<f64>>` back to a `Val`: returns `FloatVec` when all are `Some`, otherwise `Arr` with nulls.
fn numeric_options_to_val(out: Vec<Option<f64>>) -> Val {
    if out.iter().all(|v| v.is_some()) {
        Val::float_vec(out.into_iter().map(|v| v.unwrap()).collect())
    } else {
        Val::arr(
            out.into_iter()
                .map(|v| match v {
                    Some(f) => Val::Float(f),
                    None => Val::Null,
                })
                .collect(),
        )
    }
}

/// Computes a rolling sum over a window of size `n`; positions before the first full window are `Null`.
#[inline]
pub fn rolling_sum_apply(recv: &Val, n: usize) -> Option<Val> {
    if n == 0 {
        return None;
    }
    let xs = numeric_options(recv)?;
    let mut out: Vec<Option<f64>> = Vec::with_capacity(xs.len());
    let mut sum: f64 = 0.0;
    for (i, v) in xs.iter().enumerate() {
        if let Some(x) = v {
            sum += x;
        }
        if i >= n {
            if let Some(old) = xs[i - n] {
                sum -= old;
            }
        }
        if i + 1 >= n {
            out.push(Some(sum));
        } else {
            out.push(None);
        }
    }
    Some(numeric_options_to_val(out))
}

/// Computes a rolling average over a window of size `n`; positions before the first full window are `Null`.
#[inline]
pub fn rolling_avg_apply(recv: &Val, n: usize) -> Option<Val> {
    if n == 0 {
        return None;
    }
    let xs = numeric_options(recv)?;
    let mut out: Vec<Option<f64>> = Vec::with_capacity(xs.len());
    let mut sum: f64 = 0.0;
    let mut count: usize = 0;
    for (i, v) in xs.iter().enumerate() {
        if let Some(x) = v {
            sum += x;
            count += 1;
        }
        if i >= n {
            if let Some(old) = xs[i - n] {
                sum -= old;
                count -= 1;
            }
        }
        if i + 1 >= n && count > 0 {
            out.push(Some(sum / count as f64));
        } else {
            out.push(None);
        }
    }
    Some(numeric_options_to_val(out))
}

/// Computes a rolling minimum over a window of size `n`; positions before the first full window are `Null`.
#[inline]
pub fn rolling_min_apply(recv: &Val, n: usize) -> Option<Val> {
    if n == 0 {
        return None;
    }
    let xs = numeric_options(recv)?;
    let mut out: Vec<Option<f64>> = Vec::with_capacity(xs.len());
    for i in 0..xs.len() {
        if i + 1 < n {
            out.push(None);
            continue;
        }
        let lo = i + 1 - n;
        let m = xs[lo..=i]
            .iter()
            .filter_map(|v| *v)
            .fold(f64::INFINITY, |a, b| a.min(b));
        out.push(if m.is_finite() { Some(m) } else { None });
    }
    Some(numeric_options_to_val(out))
}

/// Computes a rolling maximum over a window of size `n`; positions before the first full window are `Null`.
#[inline]
pub fn rolling_max_apply(recv: &Val, n: usize) -> Option<Val> {
    if n == 0 {
        return None;
    }
    let xs = numeric_options(recv)?;
    let mut out: Vec<Option<f64>> = Vec::with_capacity(xs.len());
    for i in 0..xs.len() {
        if i + 1 < n {
            out.push(None);
            continue;
        }
        let lo = i + 1 - n;
        let m = xs[lo..=i]
            .iter()
            .filter_map(|v| *v)
            .fold(f64::NEG_INFINITY, |a, b| a.max(b));
        out.push(if m.is_finite() { Some(m) } else { None });
    }
    Some(numeric_options_to_val(out))
}

/// Shifts values backward by `n` positions; the first `n` positions are `Null`.
#[inline]
pub fn lag_apply(recv: &Val, n: usize) -> Option<Val> {
    let xs = numeric_options(recv)?;
    let mut out: Vec<Option<f64>> = Vec::with_capacity(xs.len());
    for i in 0..xs.len() {
        out.push(if i >= n { xs[i - n] } else { None });
    }
    Some(numeric_options_to_val(out))
}

/// Shifts values forward by `n` positions; the last `n` positions are `Null`.
#[inline]
pub fn lead_apply(recv: &Val, n: usize) -> Option<Val> {
    let xs = numeric_options(recv)?;
    let mut out: Vec<Option<f64>> = Vec::with_capacity(xs.len());
    for i in 0..xs.len() {
        let j = i + n;
        out.push(if j < xs.len() { xs[j] } else { None });
    }
    Some(numeric_options_to_val(out))
}

/// Returns element-wise first differences (`v[i] - v[i-1]`); the first element is `Null`.
#[inline]
pub fn diff_window_apply(recv: &Val) -> Option<Val> {
    let xs = numeric_options(recv)?;
    let mut out: Vec<Option<f64>> = Vec::with_capacity(xs.len());
    for i in 0..xs.len() {
        out.push(match (i.checked_sub(1).and_then(|j| xs[j]), xs[i]) {
            (Some(p), Some(c)) => Some(c - p),
            _ => None,
        });
    }
    Some(numeric_options_to_val(out))
}

/// Returns element-wise percentage change `(v[i] - v[i-1]) / v[i-1]`; division by zero and the first element yield `Null`.
#[inline]
pub fn pct_change_apply(recv: &Val) -> Option<Val> {
    let xs = numeric_options(recv)?;
    let mut out: Vec<Option<f64>> = Vec::with_capacity(xs.len());
    for i in 0..xs.len() {
        out.push(match (i.checked_sub(1).and_then(|j| xs[j]), xs[i]) {
            (Some(p), Some(c)) if p != 0.0 => Some((c - p) / p),
            _ => None,
        });
    }
    Some(numeric_options_to_val(out))
}

/// Computes a cumulative maximum: each position holds the running max up to that index.
#[inline]
pub fn cummax_apply(recv: &Val) -> Option<Val> {
    let xs = numeric_options(recv)?;
    let mut out: Vec<Option<f64>> = Vec::with_capacity(xs.len());
    let mut best: Option<f64> = None;
    for v in xs.iter() {
        match (*v, best) {
            (Some(x), Some(b)) => {
                best = Some(x.max(b));
                out.push(best);
            }
            (Some(x), None) => {
                best = Some(x);
                out.push(best);
            }
            (None, _) => out.push(best),
        }
    }
    Some(numeric_options_to_val(out))
}

/// Computes a cumulative minimum: each position holds the running min up to that index.
#[inline]
pub fn cummin_apply(recv: &Val) -> Option<Val> {
    let xs = numeric_options(recv)?;
    let mut out: Vec<Option<f64>> = Vec::with_capacity(xs.len());
    let mut best: Option<f64> = None;
    for v in xs.iter() {
        match (*v, best) {
            (Some(x), Some(b)) => {
                best = Some(x.min(b));
                out.push(best);
            }
            (Some(x), None) => {
                best = Some(x);
                out.push(best);
            }
            (None, _) => out.push(best),
        }
    }
    Some(numeric_options_to_val(out))
}

/// Normalises each element to its z-score `(v - mean) / stddev`; returns 0 when stddev is zero, `Null` for non-numeric.
#[inline]
pub fn zscore_apply(recv: &Val) -> Option<Val> {
    let xs = numeric_options(recv)?;
    let nums: Vec<f64> = xs.iter().filter_map(|v| *v).collect();
    if nums.is_empty() {
        return Some(numeric_options_to_val(vec![None; xs.len()]));
    }
    let mean = nums.iter().sum::<f64>() / nums.len() as f64;
    let var = nums.iter().map(|y| (y - mean).powi(2)).sum::<f64>() / nums.len() as f64;
    let sd = var.sqrt();
    let mut out: Vec<Option<f64>> = Vec::with_capacity(xs.len());
    for v in xs.iter() {
        out.push(match v {
            Some(y) if sd > 0.0 => Some((y - mean) / sd),
            Some(_) => Some(0.0),
            None => None,
        });
    }
    Some(numeric_options_to_val(out))
}

/// Returns the first `n` elements of an array; when `n == 1` returns a scalar instead of a single-element array.
#[inline]
pub fn first_apply(recv: &Val, n: i64) -> Option<Val> {
    if let Val::Arr(a) = recv {
        Some(if n == 1 {
            a.first().cloned().unwrap_or(Val::Null)
        } else {
            Val::arr(a.iter().take(n.max(0) as usize).cloned().collect())
        })
    } else {
        Some(Val::Null)
    }
}

/// Returns the last `n` elements of an array; when `n == 1` returns a scalar instead of a single-element array.
#[inline]
pub fn last_apply(recv: &Val, n: i64) -> Option<Val> {
    if let Val::Arr(a) = recv {
        Some(if n == 1 {
            a.last().cloned().unwrap_or(Val::Null)
        } else {
            let s = a.len().saturating_sub(n.max(0) as usize);
            Val::arr(a[s..].to_vec())
        })
    } else {
        Some(Val::Null)
    }
}

/// Returns the element at index `i` (negative indices count from the end); delegates to `Val::get_index`.
#[inline]
pub fn nth_any_apply(recv: &Val, i: i64) -> Option<Val> {
    Some(recv.get_index(i))
}

/// Appends `item` to the end of an array, returning a new array.
#[inline]
pub fn append_apply(recv: &Val, item: &Val) -> Option<Val> {
    let mut v = recv.clone().into_vec()?;
    v.push(item.clone());
    Some(Val::arr(v))
}

/// Inserts `item` at the beginning of an array, returning a new array.
#[inline]
pub fn prepend_apply(recv: &Val, item: &Val) -> Option<Val> {
    let mut v = recv.clone().into_vec()?;
    v.insert(0, item.clone());
    Some(Val::arr(v))
}

/// Removes all elements from an array that are structurally equal to `target`.
#[inline]
pub fn remove_value_apply(recv: &Val, target: &Val) -> Option<Val> {
    use crate::util::val_to_key;
    let items_cow = recv.as_vals()?;
    let key = val_to_key(target);
    let out: Vec<Val> = items_cow
        .iter()
        .filter(|v| val_to_key(v) != key)
        .cloned()
        .collect();
    Some(Val::arr(out))
}

/// Pairs each element with its zero-based index, producing `[{index, value}, …]`.
#[inline]
pub fn enumerate_apply(recv: &Val) -> Option<Val> {
    let items_cow = recv.as_vals()?;
    let out: Vec<Val> = items_cow
        .iter()
        .enumerate()
        .map(|(i, v)| crate::util::obj2("index", Val::Int(i as i64), "value", v.clone()))
        .collect();
    Some(Val::arr(out))
}

/// Joins all array elements into a single string separated by `sep`; non-string elements are coerced.
#[inline]
pub fn join_apply(recv: &Val, sep: &str) -> Option<Val> {
    use crate::util::val_to_string;
    use std::fmt::Write as _;

    let items_cow = recv.as_vals()?;
    let items: &[Val] = items_cow.as_ref();
    if items.is_empty() {
        return Some(Val::Str(Arc::from("")));
    }
    if items.iter().all(|v| matches!(v, Val::Str(_))) {
        let total_len: usize = items
            .iter()
            .map(|v| if let Val::Str(s) = v { s.len() } else { 0 })
            .sum::<usize>()
            + sep.len() * (items.len() - 1);
        let mut out = String::with_capacity(total_len);
        for (idx, v) in items.iter().enumerate() {
            if idx > 0 {
                out.push_str(sep);
            }
            if let Val::Str(s) = v {
                out.push_str(s);
            }
        }
        return Some(Val::Str(Arc::from(out)));
    }

    let mut out = String::with_capacity(items.len() * 8 + sep.len() * items.len());
    for (idx, v) in items.iter().enumerate() {
        if idx > 0 {
            out.push_str(sep);
        }
        match v {
            Val::Str(s) => out.push_str(s),
            Val::Int(n) => {
                let _ = write!(out, "{}", n);
            }
            Val::Float(f) => {
                let _ = write!(out, "{}", f);
            }
            Val::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Val::Null => out.push_str("null"),
            other => out.push_str(&val_to_string(other)),
        }
    }
    Some(Val::Str(Arc::from(out)))
}

/// Returns the zero-based index of the first occurrence of `target`, or `Val::Null` if not found.
#[inline]
pub fn index_value_apply(recv: &Val, target: &Val) -> Option<Val> {
    let items_cow = recv.as_vals()?;
    for (i, item) in items_cow.iter().enumerate() {
        if crate::util::vals_eq(item, target) {
            return Some(Val::Int(i as i64));
        }
    }
    Some(Val::Null)
}

/// Returns all zero-based indices where `target` appears in the array.
#[inline]
pub fn indices_of_apply(recv: &Val, target: &Val) -> Option<Val> {
    let items_cow = recv.as_vals()?;
    let out: Vec<i64> = items_cow
        .iter()
        .enumerate()
        .filter(|(_, v)| crate::util::vals_eq(v, target))
        .map(|(i, _)| i as i64)
        .collect();
    Some(Val::int_vec(out))
}

/// Unnests the array-valued `field` of each row object: each element of the nested array becomes
/// its own row, copying all other fields.
#[inline]
pub fn explode_apply(recv: &Val, field: &str) -> Option<Val> {
    let items_cow = recv.as_vals()?;
    let items: &[Val] = items_cow.as_ref();
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        match item {
            Val::Obj(m) => {
                let sub = m.get(field).cloned();
                if sub.as_ref().map(|v| v.is_array()).unwrap_or(false) {
                    let elts = sub.unwrap().into_vec().unwrap();
                    for e in elts {
                        let mut row = (**m).clone();
                        row.insert(Arc::from(field), e);
                        out.push(Val::obj(row));
                    }
                } else {
                    out.push(item.clone());
                }
            }
            other => out.push(other.clone()),
        }
    }
    Some(Val::arr(out))
}

/// Inverse of `explode`: groups rows by all fields except `field`, collecting the `field` values
/// into an array on each merged row.
#[inline]
pub fn implode_apply(recv: &Val, field: &str) -> Option<Val> {
    use crate::util::val_to_key;
    let items_cow = recv.as_vals()?;
    let items: &[Val] = items_cow.as_ref();
    let mut groups: indexmap::IndexMap<Arc<str>, (indexmap::IndexMap<Arc<str>, Val>, Vec<Val>)> =
        indexmap::IndexMap::new();
    for item in items {
        let m = match item {
            Val::Obj(m) => m,
            _ => return None,
        };
        let mut rest = (**m).clone();
        let val = rest.shift_remove(field).unwrap_or(Val::Null);
        let key_src: indexmap::IndexMap<Arc<str>, Val> = rest.clone();
        let key = Arc::<str>::from(val_to_key(&Val::obj(key_src)));
        groups
            .entry(key)
            .or_insert_with(|| (rest, Vec::new()))
            .1
            .push(val);
    }
    let mut out = Vec::with_capacity(groups.len());
    for (_, (mut rest, vals)) in groups {
        rest.insert(Arc::from(field), Val::arr(vals));
        out.push(Val::obj(rest));
    }
    Some(Val::arr(out))
}

/// Produces all adjacent pairs `[[a,b],[b,c],…]` from an array.
#[inline]
pub fn pairwise_apply(recv: &Val) -> Option<Val> {
    let items_cow = recv.as_vals()?;
    let a = items_cow.as_ref();
    let mut out: Vec<Val> = Vec::with_capacity(a.len().saturating_sub(1));
    for w in a.windows(2) {
        out.push(Val::arr(vec![w[0].clone(), w[1].clone()]));
    }
    Some(Val::arr(out))
}

/// Splits an array into non-overlapping chunks of size `n`; the last chunk may be smaller.
#[inline]
pub fn chunk_arr_apply(recv: &Val, n: usize) -> Option<Val> {
    if n == 0 {
        return None;
    }
    if let Val::Arr(a) = recv {
        let chunks: Vec<Val> = a.chunks(n).map(|c| Val::arr(c.to_vec())).collect();
        Some(Val::arr(chunks))
    } else {
        None
    }
}

/// Produces all overlapping sliding windows of size `n` from an array.
#[inline]
pub fn window_arr_apply(recv: &Val, n: usize) -> Option<Val> {
    if n == 0 {
        return None;
    }
    if let Val::Arr(a) = recv {
        let windows: Vec<Val> = a.windows(n).map(|w| Val::arr(w.to_vec())).collect();
        Some(Val::arr(windows))
    } else {
        None
    }
}

/// Returns elements that appear in both `recv` and `other` (set intersection, order from `recv`).
#[inline]
pub fn intersect_apply(recv: &Val, other: &[Val]) -> Option<Val> {
    if let Val::Arr(a) = recv {
        let other_keys: std::collections::HashSet<String> =
            other.iter().map(crate::util::val_to_key).collect();
        let kept: Vec<Val> = a
            .iter()
            .filter(|v| other_keys.contains(&crate::util::val_to_key(v)))
            .cloned()
            .collect();
        Some(Val::arr(kept))
    } else {
        None
    }
}

/// Returns all elements from `recv` plus elements in `other` not already present (set union).
#[inline]
pub fn union_apply(recv: &Val, other: &[Val]) -> Option<Val> {
    if let Val::Arr(a) = recv {
        let mut out: Vec<Val> = a.as_ref().clone();
        let a_keys: std::collections::HashSet<String> =
            a.iter().map(crate::util::val_to_key).collect();
        for v in other {
            if !a_keys.contains(&crate::util::val_to_key(v)) {
                out.push(v.clone());
            }
        }
        Some(Val::arr(out))
    } else {
        None
    }
}

/// Returns elements from `recv` that do not appear in `other` (set difference).
#[inline]
pub fn diff_apply(recv: &Val, other: &[Val]) -> Option<Val> {
    if let Val::Arr(a) = recv {
        let other_keys: std::collections::HashSet<String> =
            other.iter().map(crate::util::val_to_key).collect();
        let kept: Vec<Val> = a
            .iter()
            .filter(|v| !other_keys.contains(&crate::util::val_to_key(v)))
            .cloned()
            .collect();
        Some(Val::arr(kept))
    } else {
        None
    }
}

/// Converts an array of `[key, value]` pairs or `{key, val}` objects into an object.
#[inline]
pub fn from_pairs_apply(recv: &Val) -> Option<Val> {
    let items = recv.as_vals()?;
    let mut m: indexmap::IndexMap<Arc<str>, Val> = indexmap::IndexMap::with_capacity(items.len());
    for item in items.iter() {
        match item {
            Val::Arr(kv) if kv.len() == 2 => {
                if let Some(k) = kv[0].as_str_ref() {
                    m.insert(Arc::<str>::from(k), kv[1].clone());
                }
            }
            _ => {
                let k_val = item
                    .get("key")
                    .or_else(|| item.get("k"))
                    .cloned()
                    .unwrap_or(Val::Null);
                let v = item
                    .get("val")
                    .or_else(|| item.get("value"))
                    .or_else(|| item.get("v"))
                    .cloned()
                    .unwrap_or(Val::Null);
                if let Val::Str(k) = k_val {
                    m.insert(k, v);
                }
            }
        }
    }
    Some(Val::Obj(Arc::new(m)))
}

/// Swaps keys and values of an object; values are coerced to strings to become new keys.
#[inline]
pub fn invert_apply(recv: &Val) -> Option<Val> {
    let m = recv.as_object()?;
    let mut out: indexmap::IndexMap<Arc<str>, Val> = indexmap::IndexMap::with_capacity(m.len());
    for (k, v) in m.iter() {
        let new_key: Arc<str> = match v {
            Val::Str(s) => s.clone(),
            Val::StrSlice(r) => Arc::<str>::from(r.as_str()),
            other => Arc::<str>::from(crate::util::val_to_key(other).as_str()),
        };
        out.insert(new_key, Val::Str(k.clone()));
    }
    Some(Val::Obj(Arc::new(out)))
}

/// Shallow-merges `other` into `recv`; `other` keys overwrite `recv` keys.
#[inline]
pub fn merge_apply(recv: &Val, other: &Val) -> Option<Val> {
    let base = recv.as_object()?;
    let other = other.as_object()?;
    let mut out: indexmap::IndexMap<Arc<str>, Val> = base.clone();
    for (k, v) in other.iter() {
        out.insert(k.clone(), v.clone());
    }
    Some(Val::Obj(Arc::new(out)))
}

/// Recursively merges `other` into `recv`, combining nested objects rather than replacing them.
#[inline]
pub fn deep_merge_apply(recv: &Val, other: &Val) -> Option<Val> {
    Some(crate::util::deep_merge(recv.clone(), other.clone()))
}

/// Fills missing or null keys of `recv` with values from `other` (non-destructive merge).
#[inline]
pub fn defaults_apply(recv: &Val, other: &Val) -> Option<Val> {
    let base = recv.as_object()?;
    let defs = other.as_object()?;
    let mut out: indexmap::IndexMap<Arc<str>, Val> = base.clone();
    for (k, v) in defs.iter() {
        let entry = out.entry(k.clone()).or_insert(Val::Null);
        if entry.is_null() {
            *entry = v.clone();
        }
    }
    Some(Val::Obj(Arc::new(out)))
}

/// Renames keys in an object according to `renames` (`{old: new, …}`), preserving other keys.
#[inline]
pub fn rename_apply(recv: &Val, renames: &Val) -> Option<Val> {
    let base = recv.as_object()?;
    let renames = renames.as_object()?;
    let mut out: indexmap::IndexMap<Arc<str>, Val> = base.clone();
    for (old, new_val) in renames.iter() {
        if let Some(v) = out.shift_remove(old.as_ref()) {
            let new_key: Arc<str> = new_val
                .as_str_ref()
                .map(Arc::<str>::from)
                .unwrap_or_else(|| old.clone());
            out.insert(new_key, v);
        }
    }
    Some(Val::Obj(Arc::new(out)))
}

