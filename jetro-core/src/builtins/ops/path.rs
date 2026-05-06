use crate::util::val_key;
use crate::data::value::Val;
use indexmap::IndexMap;
use std::sync::Arc;

/// A single resolved segment of a dot/bracket path string.
pub(crate) enum PathSeg {
    /// A named object field (`.foo`).
    Field(String),
    /// A numeric array index (`[0]` or `[-1]`).
    Index(i64),
}

/// Describes where to read a value from when executing a `pick` specification.
pub(crate) enum PickSource {
    /// A single top-level field name.
    Field(Arc<str>),
    /// A multi-segment dot/bracket path, pre-parsed into [`PathSeg`]s.
    Path(Vec<PathSeg>),
}

/// One entry in a compiled `pick` call: the output key name and where to read the value from.
pub(crate) struct PickSpec {
    /// Key used in the output object.
    pub out_key: Arc<str>,
    /// Source location (field or path) inside the input object.
    pub source: PickSource,
}

/// Parses a dot/bracket path string (e.g. `"a.b[0].c"`) into a `Vec<PathSeg>`.
pub(crate) fn parse_path_segs(path: &str) -> Vec<PathSeg> {
    let mut segs = Vec::new();
    let mut cur = String::new();
    let mut chars = path.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '.' => {
                if !cur.is_empty() {
                    segs.push(PathSeg::Field(std::mem::take(&mut cur)));
                }
            }
            '[' => {
                if !cur.is_empty() {
                    segs.push(PathSeg::Field(std::mem::take(&mut cur)));
                }
                let mut idx = String::new();
                for c2 in chars.by_ref() {
                    if c2 == ']' {
                        break;
                    }
                    idx.push(c2);
                }
                segs.push(PathSeg::Index(idx.parse().unwrap_or(0)));
            }
            _ => cur.push(c),
        }
    }
    if !cur.is_empty() {
        segs.push(PathSeg::Field(cur));
    }
    segs
}

/// Traverses `val` following `segs`, returning the found value or `Val::Null` when any step is missing.
pub(crate) fn get_path_impl(val: &Val, segs: &[PathSeg]) -> Val {
    if segs.is_empty() {
        return val.clone();
    }
    let next = match &segs[0] {
        PathSeg::Field(f) => val.get(f).cloned().unwrap_or(Val::Null),
        PathSeg::Index(i) => val.get_index(*i),
    };
    get_path_impl(&next, &segs[1..])
}

/// Returns a copy of `val` with the node at `segs` replaced by `new_val`; creates missing intermediate objects.
pub(crate) fn set_path_impl(val: Val, segs: &[PathSeg], new_val: Val) -> Val {
    if segs.is_empty() {
        return new_val;
    }
    match (&segs[0], val) {
        (PathSeg::Field(f), Val::Obj(m)) => {
            let mut map = Arc::try_unwrap(m).unwrap_or_else(|m| (*m).clone());
            let child = map.shift_remove(f.as_str()).unwrap_or(Val::Null);
            map.insert(
                Arc::from(f.as_str()),
                set_path_impl(child, &segs[1..], new_val),
            );
            Val::obj(map)
        }
        (PathSeg::Index(i), Val::Arr(a)) => {
            let mut arr = Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone());
            let idx = resolve_path_idx(*i, arr.len() as i64);
            if idx < arr.len() {
                let child = arr[idx].clone();
                arr[idx] = set_path_impl(child, &segs[1..], new_val);
            }
            Val::arr(arr)
        }
        (PathSeg::Field(f), _) => {
            let mut m = IndexMap::new();
            m.insert(
                Arc::from(f.as_str()),
                set_path_impl(Val::Null, &segs[1..], new_val),
            );
            Val::obj(m)
        }
        (_, v) => v,
    }
}

/// Returns a copy of `val` with the node at `segs` removed; no-ops if the path does not exist.
pub(crate) fn del_path_impl(val: Val, segs: &[PathSeg]) -> Val {
    if segs.is_empty() {
        return Val::Null;
    }
    match (&segs[0], val) {
        (PathSeg::Field(f), Val::Obj(m)) => {
            let mut map = Arc::try_unwrap(m).unwrap_or_else(|m| (*m).clone());
            if segs.len() == 1 {
                map.shift_remove(f.as_str());
            } else if let Some(child) = map.shift_remove(f.as_str()) {
                map.insert(Arc::from(f.as_str()), del_path_impl(child, &segs[1..]));
            }
            Val::obj(map)
        }
        (PathSeg::Index(i), Val::Arr(a)) => {
            let mut arr = Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone());
            let idx = resolve_path_idx(*i, arr.len() as i64);
            if segs.len() == 1 {
                if idx < arr.len() {
                    arr.remove(idx);
                }
            } else if idx < arr.len() {
                let child = arr[idx].clone();
                arr[idx] = del_path_impl(child, &segs[1..]);
            }
            Val::arr(arr)
        }
        (_, v) => v,
    }
}

/// Converts a possibly-negative index into an absolute `usize`, clamped to `[0, len)`.
fn resolve_path_idx(i: i64, len: i64) -> usize {
    (if i < 0 { (len + i).max(0) } else { i }) as usize
}

/// Recursively flattens nested object keys into dot-separated (or `sep`-separated) flat keys,
/// writing results into `out`. Arrays and scalars terminate the recursion.
pub(crate) fn flatten_keys_impl(
    prefix: &str,
    val: &Val,
    sep: &str,
    out: &mut IndexMap<Arc<str>, Val>,
) {
    match val {
        Val::Obj(m) => {
            for (k, v) in m.iter() {
                let full = if prefix.is_empty() {
                    k.to_string()
                } else {
                    format!("{}{}{}", prefix, sep, k)
                };
                flatten_keys_impl(&full, v, sep, out);
            }
        }
        _ => {
            out.insert(Arc::from(prefix), val.clone());
        }
    }
}

/// Reconstructs a nested object from a flat `{sep}-joined-key: value` map.
pub(crate) fn unflatten_keys_impl(m: &IndexMap<Arc<str>, Val>, sep: &str) -> Val {
    let mut root: IndexMap<Arc<str>, Val> = IndexMap::new();
    for (key, val) in m {
        let parts: Vec<&str> = key.split(sep).collect();
        insert_nested(&mut root, &parts, val.clone());
    }
    Val::obj(root)
}

/// Recursively inserts `val` at the nested path `parts` inside `obj`, creating intermediate objects as needed.
fn insert_nested(obj: &mut IndexMap<Arc<str>, Val>, parts: &[&str], val: Val) {
    if parts.is_empty() {
        return;
    }
    if parts.len() == 1 {
        obj.insert(val_key(parts[0]), val);
        return;
    }
    let entry = obj
        .entry(val_key(parts[0]))
        .or_insert_with(|| Val::obj(IndexMap::new()));
    if let Val::Obj(child) = entry {
        insert_nested(Arc::make_mut(child), &parts[1..], val);
    }
}

/// Retrieves the value at a dot/bracket `path` string, returning `Val::Null` for missing nodes.
#[inline]
pub fn get_path_apply(recv: &Val, path: &str) -> Option<Val> {
    let segs = parse_path_segs(path);
    Some(get_path_impl(recv, &segs))
}

/// Returns `Val::Bool(true)` when a value exists (non-null) at the given dot/bracket path.
#[inline]
pub fn has_path_apply(recv: &Val, path: &str) -> Option<Val> {
    let segs = parse_path_segs(path);
    let found = !get_path_impl(recv, &segs).is_null();
    Some(Val::Bool(found))
}

/// Returns `Val::Bool(true)` when the object has a top-level key named `key`.
#[inline]
pub fn has_apply(recv: &Val, key: &str) -> Option<Val> {
    let m = recv.as_object()?;
    Some(Val::Bool(m.contains_key(key)))
}

/// Keeps only the listed `keys` from an object (or each object in an array), dropping all others.
#[inline]
pub fn pick_apply(recv: &Val, keys: &[Arc<str>]) -> Option<Val> {
    use indexmap::IndexMap;

    fn pick_obj(m: &IndexMap<Arc<str>, Val>, keys: &[Arc<str>]) -> Val {
        let mut out = IndexMap::with_capacity(keys.len());
        for key in keys {
            if let Some(v) = m.get(key.as_ref()) {
                out.insert(key.clone(), v.clone());
            }
        }
        Val::obj(out)
    }

    match recv {
        Val::Obj(m) => Some(pick_obj(m, keys)),
        Val::Arr(a) => Some(Val::arr(
            a.iter()
                .filter_map(|v| match v {
                    Val::Obj(m) => Some(pick_obj(m, keys)),
                    _ => None,
                })
                .collect(),
        )),
        _ => None,
    }
}

/// Richer version of `pick_apply` that supports aliasing and deep-path sources via [`PickSpec`].
#[inline]
pub(crate) fn pick_specs_apply(recv: &Val, specs: &[PickSpec]) -> Option<Val> {
    fn pick_obj(m: &IndexMap<Arc<str>, Val>, specs: &[PickSpec]) -> Val {
        let mut out = IndexMap::with_capacity(specs.len());
        let wrapped = Val::Obj(Arc::new(m.clone()));
        for spec in specs {
            match &spec.source {
                PickSource::Field(src) => {
                    if let Some(v) = m.get(src.as_ref()) {
                        out.insert(spec.out_key.clone(), v.clone());
                    }
                }
                PickSource::Path(segs) => {
                    let v = get_path_impl(&wrapped, segs);
                    if !v.is_null() {
                        out.insert(spec.out_key.clone(), v);
                    }
                }
            }
        }
        Val::obj(out)
    }

    match recv {
        Val::Obj(m) => Some(pick_obj(m, specs)),
        Val::Arr(a) => Some(Val::arr(
            a.iter()
                .filter_map(|v| match v {
                    Val::Obj(m) => Some(pick_obj(m, specs)),
                    _ => None,
                })
                .collect(),
        )),
        _ => None,
    }
}

/// Removes the listed `keys` from an object (or each object in an array), keeping all others.
#[inline]
pub fn omit_apply(recv: &Val, keys: &[Arc<str>]) -> Option<Val> {
    fn omit_obj(m: &indexmap::IndexMap<Arc<str>, Val>, keys: &[Arc<str>]) -> Val {
        let mut out = m.clone();
        for key in keys {
            out.shift_remove(key.as_ref());
        }
        Val::obj(out)
    }

    match recv {
        Val::Obj(m) => Some(omit_obj(m, keys)),
        Val::Arr(a) => Some(Val::arr(
            a.iter()
                .filter_map(|v| match v {
                    Val::Obj(m) => Some(omit_obj(m, keys)),
                    _ => None,
                })
                .collect(),
        )),
        _ => None,
    }
}

/// Returns a copy of `recv` with the node at the dot/bracket `path` removed.
#[inline]
pub fn del_path_apply(recv: &Val, path: &str) -> Option<Val> {
    let segs = parse_path_segs(path);
    Some(del_path_impl(recv.clone(), &segs))
}

/// Returns a copy of `recv` with the node at the dot/bracket `path` replaced by `value`.
#[inline]
pub fn set_path_apply(recv: &Val, path: &str, value: &Val) -> Option<Val> {
    let segs = parse_path_segs(path);
    Some(set_path_impl(recv.clone(), &segs, value.clone()))
}

/// Deletes multiple dot/bracket paths from `recv` sequentially, returning the final result.
#[inline]
pub fn del_paths_apply(recv: &Val, paths: &[Arc<str>]) -> Option<Val> {
    let mut out = recv.clone();
    for path in paths {
        let segs = parse_path_segs(path.as_ref());
        out = del_path_impl(out, &segs);
    }
    Some(out)
}

/// Collapses a nested object into a flat object using `sep`-joined key paths (e.g. `"a.b.c": v`).
#[inline]
pub fn flatten_keys_apply(recv: &Val, sep: &str) -> Option<Val> {
    let mut out: indexmap::IndexMap<Arc<str>, Val> = indexmap::IndexMap::new();
    flatten_keys_impl("", recv, sep, &mut out);
    Some(Val::obj(out))
}

/// Reconstructs a nested object from a flat `sep`-delimited key map; inverse of `flatten_keys_apply`.
#[inline]
pub fn unflatten_keys_apply(recv: &Val, sep: &str) -> Option<Val> {
    if let Val::Obj(m) = recv {
        Some(unflatten_keys_impl(m, sep))
    } else {
        None
    }
}

