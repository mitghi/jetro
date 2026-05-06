use crate::data::context::EvalError;
use crate::util::{cmp_vals, is_truthy, zip_arrays};
use crate::data::value::Val;
use indexmap::IndexMap;
use std::sync::Arc;

/// Per-row filter primitive: evaluates `eval` on `item` and returns its truthiness.
/// Streaming consumers call this once per row instead of buffering the entire array.
#[inline]
pub fn filter_one<F>(item: &Val, mut eval: F) -> Result<bool, EvalError>
where
    F: FnMut(&Val) -> Result<Val, EvalError>,
{
    Ok(is_truthy(&eval(item)?))
}

/// Buffered filter: applies the predicate to every element and returns all passing items.
/// Barrier consumers call this after collecting the full input.
#[inline]
pub fn filter_apply<F>(items: Vec<Val>, mut eval: F) -> Result<Vec<Val>, EvalError>
where
    F: FnMut(&Val) -> Result<Val, EvalError>,
{
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        if filter_one(&item, &mut eval)? {
            out.push(item);
        }
    }
    Ok(out)
}

/// Bounded filter: like [`filter_apply`] but stops after collecting `max_keep` matching items.
/// Pass `None` for `max_keep` to collect all matches (equivalent to `filter_apply`).
#[inline]
pub fn filter_apply_bounded<F>(
    items: Vec<Val>,
    max_keep: Option<usize>,
    mut eval: F,
) -> Result<Vec<Val>, EvalError>
where
    F: FnMut(&Val) -> Result<Val, EvalError>,
{
    let cap = match max_keep {
        Some(n) => n.min(items.len()),
        None => items.len(),
    };
    let mut out = Vec::with_capacity(cap);
    for item in items {
        if filter_one(&item, &mut eval)? {
            out.push(item);
            if let Some(n) = max_keep {
                if out.len() >= n {
                    break;
                }
            }
        }
    }
    Ok(out)
}

/// Per-row map primitive: evaluates `eval` on `item` and returns the projected value.
#[inline]
pub fn map_one<F>(item: &Val, mut eval: F) -> Result<Val, EvalError>
where
    F: FnMut(&Val) -> Result<Val, EvalError>,
{
    eval(item)
}

/// Buffered map: applies the projection to every element and returns the results.
#[inline]
pub fn map_apply<F>(items: Vec<Val>, mut eval: F) -> Result<Vec<Val>, EvalError>
where
    F: FnMut(&Val) -> Result<Val, EvalError>,
{
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        out.push(map_one(&item, &mut eval)?);
    }
    Ok(out)
}

/// Bounded map: like [`map_apply`] but stops after emitting `max_emit` projected values.
#[inline]
pub fn map_apply_bounded<F>(
    items: Vec<Val>,
    max_emit: Option<usize>,
    mut eval: F,
) -> Result<Vec<Val>, EvalError>
where
    F: FnMut(&Val) -> Result<Val, EvalError>,
{
    let cap = match max_emit {
        Some(n) => n.min(items.len()),
        None => items.len(),
    };
    let mut out = Vec::with_capacity(cap);
    for item in items {
        out.push(map_one(&item, &mut eval)?);
        if let Some(n) = max_emit {
            if out.len() >= n {
                break;
            }
        }
    }
    Ok(out)
}

/// Per-row flat_map primitive: evaluates `eval`, then flattens one level if the result is an array.
/// Returns a `SmallVec` to avoid heap allocation for the common single-element case.
#[inline]
pub fn flat_map_one<F>(item: &Val, mut eval: F) -> Result<smallvec::SmallVec<[Val; 1]>, EvalError>
where
    F: FnMut(&Val) -> Result<Val, EvalError>,
{
    use std::sync::Arc;
    let r = eval(item)?;
    Ok(match r {
        Val::Arr(a) => {
            let v = Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone());
            v.into_iter().collect()
        }
        v => smallvec::smallvec![v],
    })
}

/// Buffered flat_map: maps and flattens every element into a single output vector.
#[inline]
pub fn flat_map_apply<F>(items: Vec<Val>, mut eval: F) -> Result<Vec<Val>, EvalError>
where
    F: FnMut(&Val) -> Result<Val, EvalError>,
{
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        out.extend(flat_map_one(&item, &mut eval)?);
    }
    Ok(out)
}

/// Natural (ascending) sort. Specialises for homogeneous `IntVec` and `FloatVec` arrays
/// before falling back to the generic `cmp_vals` comparator.
#[inline]
pub fn sort_apply(recv: Val) -> Result<Val, EvalError> {
    match recv {
        Val::IntVec(a) => {
            let mut v = Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone());
            v.sort();
            Ok(Val::int_vec(v))
        }
        Val::FloatVec(a) => {
            let mut v = Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone());
            v.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
            Ok(Val::float_vec(v))
        }
        other => {
            let mut items = other
                .into_vec()
                .ok_or_else(|| EvalError("sort: expected array".into()))?;
            items.sort_by(cmp_vals);
            Ok(Val::arr(items))
        }
    }
}

/// Multi-key sort: evaluates one or more key expressions per element, then sorts using
/// the resulting key tuples. Each entry in `desc` controls ascending/descending order
/// for the corresponding key position.
#[inline]
pub fn sort_by_apply<F>(recv: Val, desc: &[bool], mut eval: F) -> Result<Val, EvalError>
where
    F: FnMut(&Val, usize) -> Result<Val, EvalError>,
{
    let items = recv
        .into_vec()
        .ok_or_else(|| EvalError("sort: expected array".into()))?;
    let mut keyed: Vec<(Vec<Val>, Val)> = Vec::with_capacity(items.len());
    for item in items {
        let mut keys = Vec::with_capacity(desc.len());
        for idx in 0..desc.len() {
            keys.push(eval(&item, idx)?);
        }
        keyed.push((keys, item));
    }
    keyed.sort_by(|(xk, _), (yk, _)| {
        for (idx, is_desc) in desc.iter().enumerate() {
            let ord = cmp_vals(&xk[idx], &yk[idx]);
            if ord != std::cmp::Ordering::Equal {
                return if *is_desc { ord.reverse() } else { ord };
            }
        }
        std::cmp::Ordering::Equal
    });
    Ok(Val::arr(keyed.into_iter().map(|(_, v)| v).collect()))
}

/// Sorts an array using a two-argument comparator lambda (returns `true` when left < right).
/// Errors from the comparator are captured and surfaced after the sort completes.
#[inline]
pub fn sort_comparator_apply<F>(recv: Val, mut eval_pair: F) -> Result<Val, EvalError>
where
    F: FnMut(&Val, &Val) -> Result<Val, EvalError>,
{
    let mut items = recv
        .into_vec()
        .ok_or_else(|| EvalError("sort: expected array".into()))?;
    let mut err_cell: Option<EvalError> = None;
    items.sort_by(|x, y| {
        if err_cell.is_some() {
            return std::cmp::Ordering::Equal;
        }
        match eval_pair(x, y) {
            Ok(Val::Bool(true)) => std::cmp::Ordering::Less,
            Ok(_) => std::cmp::Ordering::Greater,
            Err(e) => {
                err_cell = Some(e);
                std::cmp::Ordering::Equal
            }
        }
    });
    if let Some(e) = err_cell {
        Err(e)
    } else {
        Ok(Val::arr(items))
    }
}

/// Removes all elements for which the predicate is truthy (inverse of `filter`).
#[inline]
pub fn remove_predicate_apply<F>(recv: Val, mut eval: F) -> Result<Val, EvalError>
where
    F: FnMut(&Val) -> Result<Val, EvalError>,
{
    let items = recv
        .into_vec()
        .ok_or_else(|| EvalError("remove: expected array".into()))?;
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        if !is_truthy(&eval(&item)?) {
            out.push(item);
        }
    }
    Ok(Val::arr(out))
}

/// Filters an array keeping only elements that satisfy all `pred_count` predicates.
/// Multiple predicates are ANDed together; `eval(item, idx)` evaluates the `idx`-th predicate.
#[inline]
pub fn find_apply<F>(recv: Val, pred_count: usize, mut eval: F) -> Result<Val, EvalError>
where
    F: FnMut(&Val, usize) -> Result<Val, EvalError>,
{
    if pred_count == 0 {
        return Err(EvalError("find: requires at least one predicate".into()));
    }
    let items = recv
        .into_vec()
        .ok_or_else(|| EvalError("find: expected array".into()))?;
    let mut out = Vec::with_capacity(items.len());
    'outer: for item in items {
        for idx in 0..pred_count {
            if !is_truthy(&eval(&item, idx)?) {
                continue 'outer;
            }
        }
        out.push(item);
    }
    Ok(Val::arr(out))
}

/// Deduplicates an array by a key expression, keeping the first occurrence of each distinct key.
#[inline]
pub fn unique_by_apply<F>(recv: Val, mut eval: F) -> Result<Val, EvalError>
where
    F: FnMut(&Val) -> Result<Val, EvalError>,
{
    let items = recv
        .into_vec()
        .ok_or_else(|| EvalError("unique_by: expected array".into()))?;
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        let key = eval(&item)?;
        if seen.insert(crate::util::val_to_key(&key)) {
            out.push(item);
        }
    }
    Ok(Val::arr(out))
}

/// Returns the index of the first element satisfying all predicates, or `Val::Null`.
#[inline]
pub fn find_index_apply<F>(recv: Val, pred_count: usize, mut eval: F) -> Result<Val, EvalError>
where
    F: FnMut(&Val, usize) -> Result<Val, EvalError>,
{
    if pred_count == 0 {
        return Err(EvalError("find_index: requires a predicate".into()));
    }
    let items = recv
        .into_vec()
        .ok_or_else(|| EvalError("find_index: expected array".into()))?;
    'outer: for (idx, item) in items.iter().enumerate() {
        for pred_idx in 0..pred_count {
            if !is_truthy(&eval(item, pred_idx)?) {
                continue 'outer;
            }
        }
        return Ok(Val::Int(idx as i64));
    }
    Ok(Val::Null)
}

/// Returns all indices where every predicate evaluates to truthy; result is `Val::IntVec`.
#[inline]
pub fn indices_where_apply<F>(recv: Val, pred_count: usize, mut eval: F) -> Result<Val, EvalError>
where
    F: FnMut(&Val, usize) -> Result<Val, EvalError>,
{
    if pred_count == 0 {
        return Err(EvalError("indices_where: requires a predicate".into()));
    }
    let items = recv
        .into_vec()
        .ok_or_else(|| EvalError("indices_where: expected array".into()))?;
    let mut out = Vec::new();
    'outer: for (idx, item) in items.iter().enumerate() {
        for pred_idx in 0..pred_count {
            if !is_truthy(&eval(item, pred_idx)?) {
                continue 'outer;
            }
        }
        out.push(idx as i64);
    }
    Ok(Val::int_vec(out))
}

/// Returns the element with the greatest (or smallest) key from the key expression.
/// `want_max = true` for `max_by`, `false` for `min_by`. Returns `Val::Null` for empty arrays.
#[inline]
pub fn extreme_by_apply<F>(recv: Val, want_max: bool, mut eval: F) -> Result<Val, EvalError>
where
    F: FnMut(&Val) -> Result<Val, EvalError>,
{
    let items = recv
        .into_vec()
        .ok_or_else(|| EvalError("max_by/min_by: expected array".into()))?;
    if items.is_empty() {
        return Ok(Val::Null);
    }
    let mut best_idx = 0usize;
    let mut best_key: Option<Val> = None;
    for (idx, item) in items.iter().enumerate() {
        let key = eval(item)?;
        let take = match &best_key {
            None => true,
            Some(best) => {
                let ord = cmp_vals(&key, best);
                if want_max {
                    ord == std::cmp::Ordering::Greater
                } else {
                    ord == std::cmp::Ordering::Less
                }
            }
        };
        if take {
            best_idx = idx;
            best_key = Some(key);
        }
    }
    Ok(items.into_iter().nth(best_idx).unwrap_or(Val::Null))
}

/// Normalises a value to an array: `null` → `[]`, array → identity, scalar → `[scalar]`.
#[inline]
pub fn collect_apply(recv: &Val) -> Val {
    match recv {
        Val::Null => Val::arr(Vec::new()),
        Val::Arr(_) | Val::IntVec(_) | Val::FloatVec(_) | Val::StrVec(_) | Val::StrSliceVec(_) => {
            recv.clone()
        }
        other => Val::arr(vec![other.clone()]),
    }
}

/// Zips two arrays element-wise, stopping at the shorter array.
#[inline]
pub fn zip_apply(recv: Val, other: Val) -> Result<Val, EvalError> {
    zip_arrays(recv, other, false, Val::Null)
}

/// Zips two arrays element-wise, padding the shorter array with `fill`.
#[inline]
pub fn zip_longest_apply(recv: Val, other: Val, fill: Val) -> Result<Val, EvalError> {
    zip_arrays(recv, other, true, fill)
}

/// Zips N arrays element-wise into `[[a0, b0, ...], ...]`, truncating to the shortest.
#[inline]
pub fn global_zip_apply(arrs: &[Val]) -> Val {
    let len = arrs.iter().filter_map(|a| a.arr_len()).min().unwrap_or(0);
    Val::arr(
        (0..len)
            .map(|i| Val::arr(arrs.iter().map(|a| a.get_index(i as i64)).collect()))
            .collect(),
    )
}

/// Zips N arrays element-wise, padding shorter arrays with `fill`.
#[inline]
pub fn global_zip_longest_apply(arrs: &[Val], fill: &Val) -> Val {
    let len = arrs.iter().filter_map(|a| a.arr_len()).max().unwrap_or(0);
    Val::arr(
        (0..len)
            .map(|i| {
                Val::arr(
                    arrs.iter()
                        .map(|a| {
                            if (i as usize) < a.arr_len().unwrap_or(0) {
                                a.get_index(i as i64)
                            } else {
                                fill.clone()
                            }
                        })
                        .collect(),
                )
            })
            .collect(),
    )
}

/// Computes the Cartesian product of N arrays, returning all combinations as `[[...], ...]`.
#[inline]
pub fn global_product_apply(arrs: &[Val]) -> Val {
    let arrays: Vec<Vec<Val>> = arrs
        .iter()
        .map(|v| v.clone().into_vec().unwrap_or_default())
        .collect();
    Val::arr(
        crate::util::cartesian(&arrays)
            .into_iter()
            .map(Val::arr)
            .collect(),
    )
}

/// Generates an integer range. Accepts 1–3 arguments: `(end)`, `(start, end)`, or
/// `(start, end, step)`. Returns an empty array when `step == 0` or the range is empty.
#[inline]
pub fn range_apply(nums: &[i64]) -> Result<Val, EvalError> {
    if nums.is_empty() || nums.len() > 3 {
        return Err(EvalError(format!(
            "range: expected 1..3 args, got {}",
            nums.len()
        )));
    }
    let (from, upto, step) = match nums {
        [n] => (0, *n, 1i64),
        [f, u] => (*f, *u, 1i64),
        [f, u, s] => (*f, *u, *s),
        _ => unreachable!(),
    };
    if step == 0 {
        return Ok(Val::int_vec(Vec::new()));
    }
    let len_hint = if step > 0 && upto > from {
        (((upto - from) + step - 1) / step).max(0) as usize
    } else if step < 0 && upto < from {
        (((from - upto) + (-step) - 1) / (-step)).max(0) as usize
    } else {
        0
    };
    let mut out = Vec::with_capacity(len_hint);
    let mut i = from;
    if step > 0 {
        while i < upto {
            out.push(i);
            i += step;
        }
    } else {
        while i > upto {
            out.push(i);
            i += step;
        }
    }
    Ok(Val::int_vec(out))
}

/// Inner equi-join of two arrays of objects on matching key fields.
/// Builds a hash index over the right-hand array, then iterates the left, merging matches.
#[inline]
pub fn equi_join_apply(
    recv: Val,
    other: Val,
    lhs_key: &str,
    rhs_key: &str,
) -> Result<Val, EvalError> {
    use std::collections::HashMap;

    let left = recv
        .into_vec()
        .ok_or_else(|| EvalError("equi_join: lhs not array".into()))?;
    let right = other
        .into_vec()
        .ok_or_else(|| EvalError("equi_join: rhs not array".into()))?;
    let mut idx: HashMap<String, Vec<Val>> = HashMap::new();
    for r in right {
        let key = match &r {
            Val::Obj(o) => o.get(rhs_key).map(crate::util::val_to_key),
            _ => None,
        };
        if let Some(k) = key {
            idx.entry(k).or_default().push(r);
        }
    }

    let mut out = Vec::new();
    for l in left {
        let key = match &l {
            Val::Obj(o) => o.get(lhs_key).map(crate::util::val_to_key),
            _ => None,
        };
        let Some(k) = key else {
            continue;
        };
        let Some(matches) = idx.get(&k) else {
            continue;
        };
        for r in matches {
            out.push(merge_pair(&l, r));
        }
    }
    Ok(Val::arr(out))
}

/// Shallow-merges two objects (right wins on collision). Used internally by `equi_join`.
fn merge_pair(left: &Val, right: &Val) -> Val {
    match (left, right) {
        (Val::Obj(lo), Val::Obj(ro)) => {
            let mut out = (**lo).clone();
            for (k, v) in ro.iter() {
                out.insert(k.clone(), v.clone());
            }
            Val::obj(out)
        }
        _ => left.clone(),
    }
}

/// Pivots an array of objects into a flat or nested map.
/// Two-arg form: `pivot(key_expr, val_expr)` → `{key: val, ...}`.
/// Three-arg form: `pivot(row_expr, col_expr, val_expr)` → `{row: {col: val, ...}, ...}`.
#[inline]
pub fn pivot_apply<F>(recv: Val, arg_count: usize, mut eval: F) -> Result<Val, EvalError>
where
    F: FnMut(&Val, usize) -> Result<Val, EvalError>,
{
    let items = recv
        .into_vec()
        .ok_or_else(|| EvalError("pivot: expected array".into()))?;

    #[inline]
    fn to_key(v: Val) -> Arc<str> {
        match v {
            Val::Str(s) => s,
            other => Arc::<str>::from(crate::util::val_to_key(&other)),
        }
    }

    if arg_count >= 3 {
        let mut map: IndexMap<Arc<str>, IndexMap<Arc<str>, Val>> = IndexMap::new();
        for item in &items {
            let row = to_key(eval(item, 0)?);
            let col = to_key(eval(item, 1)?);
            let value = eval(item, 2)?;
            map.entry(row).or_default().insert(col, value);
        }
        let out = map
            .into_iter()
            .map(|(k, inner)| (k, Val::obj(inner)))
            .collect();
        return Ok(Val::obj(out));
    }

    if arg_count < 2 {
        return Err(EvalError("pivot: requires key arg and value arg".into()));
    }

    let mut map = IndexMap::with_capacity(items.len());
    for item in &items {
        let key = to_key(eval(item, 0)?);
        let value = eval(item, 1)?;
        map.insert(key, value);
    }
    Ok(Val::obj(map))
}

/// DFS pre-order visitor: calls `f` on every node (parents before children).
fn walk_pre<F: FnMut(&Val)>(value: &Val, f: &mut F) {
    f(value);
    match value {
        Val::Arr(items) => {
            for child in items.iter() {
                walk_pre(child, f);
            }
        }
        Val::Obj(map) => {
            for (_, child) in map.iter() {
                walk_pre(child, f);
            }
        }
        _ => {}
    }
}

/// DFS pre-order search: collects every node in the tree that satisfies all `pred_count` predicates.
/// Visits every descendant including nested arrays and objects.
#[inline]
pub fn deep_find_apply<F>(recv: Val, pred_count: usize, mut eval: F) -> Result<Val, EvalError>
where
    F: FnMut(&Val, usize) -> Result<Val, EvalError>,
{
    if pred_count == 0 {
        return Err(EvalError("find: requires at least one predicate".into()));
    }
    let mut out = Vec::new();
    let mut err_cell: Option<EvalError> = None;
    walk_pre(&recv, &mut |node| {
        if err_cell.is_some() {
            return;
        }
        for idx in 0..pred_count {
            match eval(node, idx) {
                Ok(v) if is_truthy(&v) => {}
                Ok(_) => return,
                Err(e) => {
                    err_cell = Some(e);
                    return;
                }
            }
        }
        out.push(node.clone());
    });
    if let Some(e) = err_cell {
        Err(e)
    } else {
        Ok(Val::arr(out))
    }
}

/// DFS pre-order search: collects every object node that contains all of the given `keys`.
#[inline]
pub fn deep_shape_apply(recv: Val, keys: &[Arc<str>]) -> Result<Val, EvalError> {
    if keys.is_empty() {
        return Err(EvalError("shape: empty pattern".into()));
    }
    let mut out = Vec::new();
    walk_pre(&recv, &mut |node| {
        if let Val::Obj(map) = node {
            if keys.iter().all(|k| map.contains_key(k.as_ref())) {
                out.push(node.clone());
            }
        }
    });
    Ok(Val::arr(out))
}

/// DFS pre-order search: collects every object node whose listed keys equal the given literal values.
#[inline]
pub fn deep_like_apply(recv: Val, pats: &[(Arc<str>, Val)]) -> Result<Val, EvalError> {
    if pats.is_empty() {
        return Err(EvalError("like: empty pattern".into()));
    }
    let mut out = Vec::new();
    walk_pre(&recv, &mut |node| {
        if let Val::Obj(map) = node {
            let ok = pats.iter().all(|(key, want)| {
                map.get(key.as_ref())
                    .map(|got| crate::util::vals_eq(got, want))
                    .unwrap_or(false)
            });
            if ok {
                out.push(node.clone());
            }
        }
    });
    Ok(Val::arr(out))
}

/// Recursive tree transform. When `pre = true` the transform runs top-down (pre-order);
/// when `pre = false` it runs bottom-up (post-order). All array and object children
/// are recursively transformed, then the lambda is applied.
pub fn walk_apply<F>(recv: Val, pre: bool, eval: &mut F) -> Result<Val, EvalError>
where
    F: FnMut(Val) -> Result<Val, EvalError>,
{
    let transformed = if pre { eval(recv)? } else { recv };
    let after_children = match transformed {
        Val::Arr(a) => {
            let items = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
            let mut out = Vec::with_capacity(items.len());
            for child in items {
                out.push(walk_apply(child, pre, eval)?);
            }
            Val::arr(out)
        }
        Val::IntVec(a) => {
            let mut out = Vec::with_capacity(a.len());
            for n in a.iter() {
                out.push(walk_apply(Val::Int(*n), pre, eval)?);
            }
            Val::arr(out)
        }
        Val::FloatVec(a) => {
            let mut out = Vec::with_capacity(a.len());
            for n in a.iter() {
                out.push(walk_apply(Val::Float(*n), pre, eval)?);
            }
            Val::arr(out)
        }
        Val::Obj(m) => {
            let items = Arc::try_unwrap(m).unwrap_or_else(|arc| (*arc).clone());
            let mut out = IndexMap::with_capacity(items.len());
            for (k, child) in items {
                out.insert(k, walk_apply(child, pre, eval)?);
            }
            Val::obj(out)
        }
        other => other,
    };
    if pre {
        Ok(after_children)
    } else {
        eval(after_children)
    }
}

/// Applies `eval` repeatedly until the value reaches a fixpoint (output equals input).
/// Errors if the fixpoint is not reached within 10 000 iterations.
#[inline]
pub fn rec_apply<F>(mut recv: Val, mut eval: F) -> Result<Val, EvalError>
where
    F: FnMut(Val) -> Result<Val, EvalError>,
{
    for _ in 0..10_000 {
        let next = eval(recv.clone())?;
        if crate::util::vals_eq(&recv, &next) {
            return Ok(next);
        }
        recv = next;
    }
    Err(EvalError(
        "rec: exceeded 10000 iterations without reaching fixpoint".into(),
    ))
}

/// Walks the entire value tree and, for every node where the predicate is truthy, emits
/// a `{path: "$...", value: ...}` object. Paths use `$` as the root and `.field` / `[idx]` syntax.
pub fn trace_path_apply<F>(recv: Val, mut eval: F) -> Result<Val, EvalError>
where
    F: FnMut(&Val) -> Result<Val, EvalError>,
{
    fn walk<F>(value: &Val, path: String, eval: &mut F, out: &mut Vec<Val>) -> Result<(), EvalError>
    where
        F: FnMut(&Val) -> Result<Val, EvalError>,
    {
        if is_truthy(&eval(value)?) {
            let mut row = IndexMap::with_capacity(2);
            row.insert(Arc::from("path"), Val::Str(Arc::from(path.as_str())));
            row.insert(Arc::from("value"), value.clone());
            out.push(Val::obj(row));
        }
        match value {
            Val::Arr(items) => {
                for (idx, child) in items.iter().enumerate() {
                    walk(child, format!("{}[{}]", path, idx), eval, out)?;
                }
            }
            Val::IntVec(items) => {
                for (idx, n) in items.iter().enumerate() {
                    walk(&Val::Int(*n), format!("{}[{}]", path, idx), eval, out)?;
                }
            }
            Val::FloatVec(items) => {
                for (idx, n) in items.iter().enumerate() {
                    walk(&Val::Float(*n), format!("{}[{}]", path, idx), eval, out)?;
                }
            }
            Val::Obj(map) => {
                for (key, child) in map.iter() {
                    walk(child, format!("{}.{}", path, key), eval, out)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    let mut out = Vec::new();
    walk(&recv, String::from("$"), &mut eval, &mut out)?;
    Ok(Val::arr(out))
}

/// Evaluates `count` independent expressions against the same receiver and returns
/// the results as an array `[expr0(recv), expr1(recv), ...]`.
#[inline]
pub fn fanout_apply<F>(recv: &Val, count: usize, mut eval: F) -> Result<Val, EvalError>
where
    F: FnMut(&Val, usize) -> Result<Val, EvalError>,
{
    if count == 0 {
        return Err(EvalError("fanout: requires at least one expression".into()));
    }
    let mut out = Vec::with_capacity(count);
    for idx in 0..count {
        out.push(eval(recv, idx)?);
    }
    Ok(Val::arr(out))
}

/// Evaluates named expressions against the receiver and collects results into an object
/// `{name0: expr0(recv), name1: expr1(recv), ...}`.
#[inline]
pub fn zip_shape_apply<F>(recv: &Val, names: &[Arc<str>], mut eval: F) -> Result<Val, EvalError>
where
    F: FnMut(&Val, usize) -> Result<Val, EvalError>,
{
    if names.is_empty() {
        return Err(EvalError("zip_shape: requires at least one field".into()));
    }
    let mut out = IndexMap::with_capacity(names.len());
    for (idx, name) in names.iter().enumerate() {
        out.insert(name.clone(), eval(recv, idx)?);
    }
    Ok(Val::obj(out))
}

/// Groups elements by a key expression (arg 0), then applies a shape expression (arg 1)
/// to each group, returning `{key: shape(group), ...}`.
#[inline]
pub fn group_shape_apply<F>(recv: Val, mut eval: F) -> Result<Val, EvalError>
where
    F: FnMut(Val, usize) -> Result<Val, EvalError>,
{
    let items = recv
        .into_vec()
        .ok_or_else(|| EvalError("group_shape: expected array".into()))?;
    let mut buckets: IndexMap<Arc<str>, Vec<Val>> = IndexMap::with_capacity(items.len());
    for item in items {
        let key = match eval(item.clone(), 0)? {
            Val::Str(s) => s,
            other => Arc::<str>::from(crate::util::val_to_key(&other)),
        };
        buckets.entry(key).or_default().push(item);
    }
    let mut out = IndexMap::with_capacity(buckets.len());
    for (key, group) in buckets {
        out.insert(key, eval(Val::arr(group), 1)?);
    }
    Ok(Val::obj(out))
}

/// Per-row primitive for `take_while`: returns true while the predicate holds.
#[inline]
pub fn take_while_one<F>(item: &Val, eval: F) -> Result<bool, EvalError>
where
    F: FnMut(&Val) -> Result<Val, EvalError>,
{
    filter_one(item, eval)
}

/// Per-row primitive for `any`: returns true when the predicate is truthy.
#[inline]
pub fn any_one<F>(item: &Val, eval: F) -> Result<bool, EvalError>
where
    F: FnMut(&Val) -> Result<Val, EvalError>,
{
    filter_one(item, eval)
}

/// Per-row primitive for `all`: returns true when the predicate is truthy.
#[inline]
pub fn all_one<F>(item: &Val, eval: F) -> Result<bool, EvalError>
where
    F: FnMut(&Val) -> Result<Val, EvalError>,
{
    filter_one(item, eval)
}

/// Buffered `take_while`: keeps the leading elements satisfying the predicate, stops at the first falsy result.
#[inline]
pub fn take_while_apply<F>(items: Vec<Val>, mut eval: F) -> Result<Vec<Val>, EvalError>
where
    F: FnMut(&Val) -> Result<Val, EvalError>,
{
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        if !take_while_one(&item, &mut eval)? {
            break;
        }
        out.push(item);
    }
    Ok(out)
}

/// Buffered `drop_while`: skips leading elements satisfying the predicate, then passes the rest.
#[inline]
pub fn drop_while_apply<F>(items: Vec<Val>, mut eval: F) -> Result<Vec<Val>, EvalError>
where
    F: FnMut(&Val) -> Result<Val, EvalError>,
{
    let mut dropping = true;
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        if dropping {
            if filter_one(&item, &mut eval)? {
                continue;
            }
            dropping = false;
        }
        out.push(item);
    }
    Ok(out)
}

/// Splits elements into two groups: those satisfying the predicate (first) and those that don't (second).
#[inline]
pub fn partition_apply<F>(items: Vec<Val>, mut eval: F) -> Result<(Vec<Val>, Vec<Val>), EvalError>
where
    F: FnMut(&Val) -> Result<Val, EvalError>,
{
    let mut yes = Vec::with_capacity(items.len());
    let mut no = Vec::with_capacity(items.len());
    for item in items {
        if filter_one(&item, &mut eval)? {
            yes.push(item);
        } else {
            no.push(item);
        }
    }
    Ok((yes, no))
}

/// Groups elements by a key expression, returning an `IndexMap<key, [elements]>`.
/// Insertion order of the first occurrence of each key is preserved.
#[inline]
pub fn group_by_apply<F>(
    items: Vec<Val>,
    mut eval: F,
) -> Result<indexmap::IndexMap<std::sync::Arc<str>, Val>, EvalError>
where
    F: FnMut(&Val) -> Result<Val, EvalError>,
{
    use std::sync::Arc;
    let mut map: indexmap::IndexMap<Arc<str>, Val> = indexmap::IndexMap::new();
    for item in items {
        let k: Arc<str> = Arc::from(crate::util::val_to_key(&eval(&item)?).as_str());
        let bucket = map.entry(k).or_insert_with(|| Val::arr(Vec::new()));
        bucket.as_array_mut().unwrap().push(item);
    }
    Ok(map)
}

/// Counts elements per key expression, returning an `IndexMap<key, Int>`.
#[inline]
pub fn count_by_apply<F>(
    items: Vec<Val>,
    mut eval: F,
) -> Result<indexmap::IndexMap<std::sync::Arc<str>, Val>, EvalError>
where
    F: FnMut(&Val) -> Result<Val, EvalError>,
{
    use std::sync::Arc;
    let mut map: indexmap::IndexMap<Arc<str>, Val> = indexmap::IndexMap::new();
    for item in items {
        let k: Arc<str> = Arc::from(crate::util::val_to_key(&eval(&item)?).as_str());
        let counter = map.entry(k).or_insert(Val::Int(0));
        if let Val::Int(n) = counter {
            *n += 1;
        }
    }
    Ok(map)
}

/// Indexes elements by a key expression, returning `IndexMap<key, last_matching_element>`.
/// When two elements share a key, the last one wins.
#[inline]
pub fn index_by_apply<F>(
    items: Vec<Val>,
    mut eval: F,
) -> Result<indexmap::IndexMap<std::sync::Arc<str>, Val>, EvalError>
where
    F: FnMut(&Val) -> Result<Val, EvalError>,
{
    use std::sync::Arc;
    let mut map: indexmap::IndexMap<Arc<str>, Val> = indexmap::IndexMap::new();
    for item in items {
        let k: Arc<str> = Arc::from(crate::util::val_to_key(&eval(&item)?).as_str());
        map.insert(k, item);
    }
    Ok(map)
}

/// Filters an object's entries, keeping only those for which `keep(key, value)` is truthy.
#[inline]
pub fn filter_object_apply<F>(
    map: indexmap::IndexMap<std::sync::Arc<str>, Val>,
    mut keep: F,
) -> Result<indexmap::IndexMap<std::sync::Arc<str>, Val>, EvalError>
where
    F: FnMut(&std::sync::Arc<str>, &Val) -> Result<bool, EvalError>,
{
    let mut out: indexmap::IndexMap<std::sync::Arc<str>, Val> = indexmap::IndexMap::new();
    for (k, v) in map {
        if keep(&k, &v)? {
            out.insert(k, v);
        }
    }
    Ok(out)
}

/// Applies `eval` to every key of the object and rebuilds the map with the new keys.
#[inline]
pub fn transform_keys_apply<F>(
    map: indexmap::IndexMap<std::sync::Arc<str>, Val>,
    mut eval: F,
) -> Result<indexmap::IndexMap<std::sync::Arc<str>, Val>, EvalError>
where
    F: FnMut(&std::sync::Arc<str>) -> Result<Val, EvalError>,
{
    use std::sync::Arc;
    let mut out: indexmap::IndexMap<Arc<str>, Val> = indexmap::IndexMap::with_capacity(map.len());
    for (k, v) in map {
        let new_key: Arc<str> = Arc::from(crate::util::val_to_key(&eval(&k)?).as_str());
        out.insert(new_key, v);
    }
    Ok(out)
}

/// Returns an array of every key in the object, or an empty array for non-objects.
#[inline]
pub fn keys_apply(recv: &Val) -> Val {
    Val::arr(
        recv.as_object()
            .map(|m| m.keys().map(|k| Val::Str(k.clone())).collect())
            .unwrap_or_default(),
    )
}

/// Returns an array of every value in the object, or an empty array for non-objects.
#[inline]
pub fn values_apply(recv: &Val) -> Val {
    Val::arr(
        recv.as_object()
            .map(|m| m.values().cloned().collect())
            .unwrap_or_default(),
    )
}

/// Returns `[[key, value], ...]` pairs for each entry in the object.
#[inline]
pub fn entries_apply(recv: &Val) -> Val {
    Val::arr(
        recv.as_object()
            .map(|m| {
                m.iter()
                    .map(|(k, v)| Val::arr(vec![Val::Str(k.clone()), v.clone()]))
                    .collect()
            })
            .unwrap_or_default(),
    )
}

