use indexmap::IndexMap;
use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::ser::{Serialize, SerializeMap, SerializeSeq, Serializer};
use serde_json::{Map, Number};
/// Internal value type for the v2 evaluator.
///
/// Every compound node (Arr, Obj) is `Arc`-wrapped so that `Val::clone()`
/// is a single atomic refcount bump — no deep copies during traversal.
///
/// The API boundary (`evaluate()`) converts `&serde_json::Value → Val`
/// once on entry and `Val → serde_json::Value` once on exit.
use std::borrow::Cow;
use std::sync::Arc;

// ── Core type ─────────────────────────────────────────────────────────────────
//
// `IntVec` / `FloatVec` are columnar variants: mono-typed numeric arrays
// stored as `Vec<i64>` / `Vec<f64>` instead of `Vec<Val>`.  This cuts output
// bandwidth 3x (8B/elem vs 24B Val enum) on producers that emit homogeneous
// numeric arrays (`range()`, `accumulate` typed fast-path, `from_json` of
// all-int seq).  Consumers either handle them directly (typed aggregates,
// serde) or materialize on demand via `as_vals() -> Cow<[Val]>`.

#[derive(Clone, Debug)]
pub enum Val {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(Arc<str>),
    /// Borrowed slice into a parent `Arc<str>` — zero-alloc view.
    /// Produced by slice/split-first/substring to avoid a fresh heap
    /// allocation per row.  Treat identically to `Str` at all semantic
    /// boundaries (display, serialize, compare, hash).
    StrSlice(crate::strref::StrRef),
    Arr(Arc<Vec<Val>>),
    IntVec(Arc<Vec<i64>>),
    FloatVec(Arc<Vec<f64>>),
    StrVec(Arc<Vec<Arc<str>>>),
    /// Columnar lane of borrowed-slice string views.  Each element is a
    /// `StrRef` (parent Arc + byte range); emitted by map-slice / split
    /// fusions to avoid per-row `Val` enum tag + per-row heap allocation.
    /// Serializes directly as JSON array of strings via `ValRef`.
    StrSliceVec(Arc<Vec<crate::strref::StrRef>>),
    Obj(Arc<IndexMap<Arc<str>, Val>>),
    /// Inline small object — flat `(key, value)` pair slice, no hashtable.
    /// Used for per-row `map({k1, k2, ..})` projections and similar hot
    /// loops where the allocating an `Arc<IndexMap>` per row dominates.
    /// Lookup is linear scan (fine for ≤8 entries); insertion-order
    /// preserved.  Promote to `Obj` if growth / key churn demands it.
    ObjSmall(Arc<[(Arc<str>, Val)]>),
    /// Columnar array-of-objects lane — struct-of-arrays with shared key
    /// schema.  Each row is an object with the exact same keys in the same
    /// order; keys stored once in `keys`, row values in `rows[i]`.  Used
    /// by `map({k1, k2, ..})` projections that produce a uniform-shape
    /// array.  Serialize iterates once over rows without per-row Arc
    /// allocation or hashtable reconstruction.
    ObjVec(Arc<ObjVecData>),
}

/// Columnar struct-of-arrays for a uniform-shape array of objects.
///
/// Phase 7 layout: cells are stored row-major in a single flat
/// `Vec<Val>` so that a row's fields live in adjacent memory.  Stride
/// = `keys.len()`; the value at `(row, slot)` is `cells[row * stride + slot]`.
///
/// Compared to the prior `Vec<Vec<Val>>` layout this removes one heap
/// allocation per row and lets the inner loop in slot-aware aggregates
/// stride through cells with a constant offset, which the autovec /
/// prefetch path can exploit on hot kernels.
///
/// Phase 7-typed-columns: alongside `cells` (row-major Val matrix used
/// for compatibility with all existing kernels), `typed_cols` holds an
/// optional per-slot column typed as `Vec<i64>` / `Vec<f64>` / etc.
/// when the slot is uniform-type across all rows.  Slot kernels can
/// read raw `&[i64]` directly — closes the boxed-Val tag-check tax
/// (~3-4× win on numeric aggregates).
#[derive(Debug)]
pub struct ObjVecData {
    pub keys: Arc<[Arc<str>]>,
    pub cells: Vec<Val>,
    /// Typed-column mirror.  `None` if not promoted; `Some(cols)` with
    /// `cols.len() == keys.len()`.  Each column may be `Mixed` (no
    /// type lock) or a typed lane (`Ints` / `Floats` / `Strs` / `Bools`).
    pub typed_cols: Option<Arc<Vec<ObjVecCol>>>,
}

/// Per-slot typed column variants.  Selected at promotion time when
/// every row's value at that slot has the same primitive Val tag.
#[derive(Debug, Clone)]
pub enum ObjVecCol {
    Mixed, // no uniform type; fall back to cells walk
    Ints(Vec<i64>),
    Floats(Vec<f64>),
    Strs(Vec<Arc<str>>),
    Bools(Vec<bool>),
}

/// Build typed-column mirror from a row-major Val cells matrix.
/// Same logic as the pipeline-side helper; lives here so the simd-json
/// promotion path (`from_simd_tape`) can light up typed kernels at
/// cold-parse time without depending on the pipeline crate ordering.
pub fn build_typed_cols_from_cells(cells: &[Val], stride: usize, nrows: usize) -> Vec<ObjVecCol> {
    let mut out: Vec<ObjVecCol> = Vec::with_capacity(stride);
    if stride == 0 || nrows == 0 {
        for _ in 0..stride {
            out.push(ObjVecCol::Mixed);
        }
        return out;
    }
    for slot in 0..stride {
        let target_tag: u8 = match &cells[slot] {
            Val::Int(_) => 1,
            Val::Float(_) => 2,
            Val::Str(_) | Val::StrSlice(_) => 3,
            Val::Bool(_) => 4,
            _ => 0,
        };
        if target_tag == 0 {
            out.push(ObjVecCol::Mixed);
            continue;
        }
        let mut ok = true;
        for r in 0..nrows {
            let v = &cells[r * stride + slot];
            let same = matches!(
                (target_tag, v),
                (1, Val::Int(_))
                    | (2, Val::Float(_))
                    | (3, Val::Str(_))
                    | (3, Val::StrSlice(_))
                    | (4, Val::Bool(_))
            );
            if !same {
                ok = false;
                break;
            }
        }
        if !ok {
            out.push(ObjVecCol::Mixed);
            continue;
        }
        match target_tag {
            1 => {
                let mut col: Vec<i64> = Vec::with_capacity(nrows);
                for r in 0..nrows {
                    if let Val::Int(n) = &cells[r * stride + slot] {
                        col.push(*n);
                    }
                }
                out.push(ObjVecCol::Ints(col));
            }
            2 => {
                let mut col: Vec<f64> = Vec::with_capacity(nrows);
                for r in 0..nrows {
                    if let Val::Float(f) = &cells[r * stride + slot] {
                        col.push(*f);
                    }
                }
                out.push(ObjVecCol::Floats(col));
            }
            3 => {
                let mut col: Vec<Arc<str>> = Vec::with_capacity(nrows);
                for r in 0..nrows {
                    match &cells[r * stride + slot] {
                        Val::Str(s) => col.push(Arc::clone(s)),
                        Val::StrSlice(s) => col.push(s.to_arc()),
                        _ => {}
                    }
                }
                out.push(ObjVecCol::Strs(col));
            }
            4 => {
                let mut col: Vec<bool> = Vec::with_capacity(nrows);
                for r in 0..nrows {
                    if let Val::Bool(b) = &cells[r * stride + slot] {
                        col.push(*b);
                    }
                }
                out.push(ObjVecCol::Bools(col));
            }
            _ => out.push(ObjVecCol::Mixed),
        }
    }
    out
}

impl ObjVecData {
    /// Stride between rows (== number of keys).  Per-instance constant.
    #[inline]
    pub fn stride(&self) -> usize {
        self.keys.len()
    }

    /// Number of rows.  `cells.len() / stride()` (cheap, no division when
    /// stride is a const propagated by the optimiser at the call site).
    #[inline]
    pub fn nrows(&self) -> usize {
        let s = self.stride();
        if s == 0 {
            0
        } else {
            self.cells.len() / s
        }
    }

    /// Slot index for `key`, or `None` if the field isn't in the shape.
    /// Linear scan; called once per opcode, then cached in a closure
    /// over the slot for the per-row inner loop.
    #[inline]
    pub fn slot_of(&self, key: &str) -> Option<usize> {
        self.keys.iter().position(|k| k.as_ref() == key)
    }

    /// Borrow the cell at `(row, slot)`.  Caller asserts `slot < stride()`
    /// and `row < nrows()`; debug-asserted.
    #[inline]
    pub fn cell(&self, row: usize, slot: usize) -> &Val {
        debug_assert!(slot < self.stride());
        debug_assert!(row < self.nrows());
        &self.cells[row * self.stride() + slot]
    }

    /// Iterator over the values in a single column (`slot`).  Steps by
    /// `stride()` through the flat cells vec.  O(N) materialisation
    /// avoided — caller may call `.sum()` / `.min()` directly.
    pub fn column(&self, slot: usize) -> impl Iterator<Item = &Val> {
        let s = self.stride();
        debug_assert!(slot < s);
        self.cells.iter().skip(slot).step_by(s)
    }

    /// Reconstruct row `i` as a borrowed slice over its `stride()`
    /// adjacent cells.  Cheap — no allocation.
    #[inline]
    pub fn row_slice(&self, row: usize) -> &[Val] {
        let s = self.stride();
        let off = row * s;
        &self.cells[off..off + s]
    }
}

impl Val {
    /// Unified borrowed `&str` view that works for both `Val::Str` and
    /// `Val::StrSlice`.  Returns `None` for non-string variants.
    #[inline]
    pub fn as_str_ref(&self) -> Option<&str> {
        match self {
            Val::Str(s) => Some(s.as_ref()),
            Val::StrSlice(r) => Some(r.as_str()),
            _ => None,
        }
    }
}

// ── Constants (avoid heap allocation for common nulls) ────────────────────────

thread_local! {
    static NULL_VAL: Val = Val::Null;
    /// Per-thread key intern cache — shared across `visit_map` / `From<&Value>`
    /// calls within a single deserialize to avoid allocating a fresh
    /// `Arc<str>` for every repeated key across rows of an array.
    /// Scoped keys live until the cache hits its capacity limit; common
    /// short keys like "id", "grp", "status" get reused for the entire
    /// array walk.
    static KEY_INTERN: std::cell::RefCell<std::collections::HashMap<Box<str>, Arc<str>>> =
        std::cell::RefCell::new(std::collections::HashMap::with_capacity(64));
}

/// Intern an object key for the current thread.  Returns a clone of a
/// cached `Arc<str>` when the key has already been seen; otherwise
/// allocates once and caches.  Capped at a soft limit to avoid
/// unbounded growth on pathological docs.
#[inline]
pub fn intern_key(k: &str) -> Arc<str> {
    const CAP: usize = 4096;
    KEY_INTERN.with(|cell| {
        let mut m = cell.borrow_mut();
        if let Some(a) = m.get(k) {
            return Arc::clone(a);
        }
        if m.len() >= CAP {
            // Fall back to a fresh Arc without caching — prevents
            // unbounded growth on docs with wildly varying keys.
            return Arc::<str>::from(k);
        }
        let a: Arc<str> = Arc::<str>::from(k);
        m.insert(k.into(), Arc::clone(&a));
        a
    })
}

// ── Cheap structural operations ───────────────────────────────────────────────

impl Val {
    /// O(1) field lookup — returns a clone of the child (cheap: Arc bump for Arr/Obj, copy for scalars).
    #[inline]
    pub fn get_field(&self, key: &str) -> Val {
        match self {
            Val::Obj(m) => m.get(key).cloned().unwrap_or(Val::Null),
            Val::ObjSmall(pairs) => {
                for (k, v) in pairs.iter() {
                    if k.as_ref() == key {
                        return v.clone();
                    }
                }
                Val::Null
            }
            _ => Val::Null,
        }
    }

    /// O(1) index — returns a clone of the element (cheap: Arc bump for Arr/Obj).
    #[inline]
    pub fn get_index(&self, i: i64) -> Val {
        match self {
            Val::Arr(a) => {
                let idx = if i < 0 {
                    a.len().saturating_sub(i.unsigned_abs() as usize)
                } else {
                    i as usize
                };
                a.get(idx).cloned().unwrap_or(Val::Null)
            }
            Val::IntVec(a) => {
                let idx = if i < 0 {
                    a.len().saturating_sub(i.unsigned_abs() as usize)
                } else {
                    i as usize
                };
                a.get(idx).copied().map(Val::Int).unwrap_or(Val::Null)
            }
            Val::FloatVec(a) => {
                let idx = if i < 0 {
                    a.len().saturating_sub(i.unsigned_abs() as usize)
                } else {
                    i as usize
                };
                a.get(idx).copied().map(Val::Float).unwrap_or(Val::Null)
            }
            Val::StrVec(a) => {
                let idx = if i < 0 {
                    a.len().saturating_sub(i.unsigned_abs() as usize)
                } else {
                    i as usize
                };
                a.get(idx).cloned().map(Val::Str).unwrap_or(Val::Null)
            }
            Val::StrSliceVec(a) => {
                let idx = if i < 0 {
                    a.len().saturating_sub(i.unsigned_abs() as usize)
                } else {
                    i as usize
                };
                a.get(idx).cloned().map(Val::StrSlice).unwrap_or(Val::Null)
            }
            _ => Val::Null,
        }
    }

    #[inline]
    pub fn is_null(&self) -> bool {
        matches!(self, Val::Null)
    }
    #[inline]
    pub fn is_number(&self) -> bool {
        matches!(self, Val::Int(_) | Val::Float(_))
    }
    #[inline]
    pub fn is_string(&self) -> bool {
        matches!(self, Val::Str(_) | Val::StrSlice(_))
    }
    #[inline]
    pub fn is_array(&self) -> bool {
        matches!(
            self,
            Val::Arr(_) | Val::IntVec(_) | Val::FloatVec(_) | Val::StrVec(_) | Val::StrSliceVec(_)
        )
    }

    /// Array length — also works on columnar variants.
    #[inline]
    pub fn arr_len(&self) -> Option<usize> {
        match self {
            Val::Arr(a) => Some(a.len()),
            Val::IntVec(a) => Some(a.len()),
            Val::FloatVec(a) => Some(a.len()),
            Val::StrVec(a) => Some(a.len()),
            Val::StrSliceVec(a) => Some(a.len()),
            Val::ObjVec(d) => Some(d.nrows()),
            _ => None,
        }
    }

    #[inline]
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Val::Int(n) => Some(*n),
            Val::Float(f) => Some(*f as i64),
            _ => None,
        }
    }

    #[inline]
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Val::Float(f) => Some(*f),
            Val::Int(n) => Some(*n as f64),
            _ => None,
        }
    }

    #[inline]
    pub fn as_str(&self) -> Option<&str> {
        self.as_str_ref()
    }

    #[inline]
    pub fn as_array(&self) -> Option<&[Val]> {
        if let Val::Arr(a) = self {
            Some(a)
        } else {
            None
        }
    }

    /// Materialize any array-like (including columnar) as a `Cow<[Val]>`.
    /// Borrowed for `Val::Arr`; owned allocation for `Val::IntVec`/`FloatVec`/`ObjVec`.
    pub fn as_vals(&self) -> Option<Cow<'_, [Val]>> {
        match self {
            Val::Arr(a) => Some(Cow::Borrowed(a.as_slice())),
            Val::IntVec(a) => Some(Cow::Owned(a.iter().map(|n| Val::Int(*n)).collect())),
            Val::FloatVec(a) => Some(Cow::Owned(a.iter().map(|f| Val::Float(*f)).collect())),
            Val::StrVec(a) => Some(Cow::Owned(a.iter().map(|s| Val::Str(s.clone())).collect())),
            Val::ObjVec(d) => {
                let n = d.nrows();
                let mut out: Vec<Val> = Vec::with_capacity(n);
                for row in 0..n {
                    let mut m: IndexMap<Arc<str>, Val> = IndexMap::with_capacity(d.keys.len());
                    for (i, k) in d.keys.iter().enumerate() {
                        m.insert(Arc::clone(k), d.cell(row, i).clone());
                    }
                    out.push(Val::Obj(Arc::new(m)));
                }
                Some(Cow::Owned(out))
            }
            _ => None,
        }
    }

    pub fn as_array_mut(&mut self) -> Option<&mut Vec<Val>> {
        if let Val::Arr(a) = self {
            Some(Arc::make_mut(a))
        } else {
            None
        }
    }

    #[inline]
    pub fn as_object(&self) -> Option<&IndexMap<Arc<str>, Val>> {
        if let Val::Obj(m) = self {
            Some(m)
        } else {
            None
        }
    }

    /// Read-only get (serde_json compat shim).
    pub fn get(&self, key: &str) -> Option<&Val> {
        match self {
            Val::Obj(m) => m.get(key),
            Val::ObjSmall(pairs) => pairs
                .iter()
                .find(|(k, _)| k.as_ref() == key)
                .map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Val::Null => "null",
            Val::Bool(_) => "bool",
            Val::Int(_) | Val::Float(_) => "number",
            Val::Str(_) | Val::StrSlice(_) => "string",
            Val::Arr(_)
            | Val::IntVec(_)
            | Val::FloatVec(_)
            | Val::StrVec(_)
            | Val::StrSliceVec(_)
            | Val::ObjVec(_) => "array",
            Val::Obj(_) | Val::ObjSmall(_) => "object",
        }
    }

    /// Consume self and produce a mutable Vec (clone only if shared).
    /// Columnar variants are materialized into `Vec<Val>`.
    pub fn into_vec(self) -> Option<Vec<Val>> {
        match self {
            Val::Arr(a) => Some(Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone())),
            Val::IntVec(a) => Some(a.iter().map(|n| Val::Int(*n)).collect()),
            Val::FloatVec(a) => Some(a.iter().map(|f| Val::Float(*f)).collect()),
            Val::StrVec(a) => Some(a.iter().map(|s| Val::Str(s.clone())).collect()),
            Val::StrSliceVec(a) => Some(a.iter().map(|s| Val::StrSlice(s.clone())).collect()),
            Val::ObjVec(d) => {
                let stride = d.keys.len();
                let nrows = if stride == 0 {
                    0
                } else {
                    d.cells.len() / stride
                };
                let mut out = Vec::with_capacity(nrows);
                for r in 0..nrows {
                    let mut m: IndexMap<Arc<str>, Val> = IndexMap::with_capacity(stride);
                    for (i, k) in d.keys.iter().enumerate() {
                        m.insert(k.clone(), d.cells[r * stride + i].clone());
                    }
                    out.push(Val::Obj(Arc::new(m)));
                }
                Some(out)
            }
            _ => None,
        }
    }

    /// Consume self and produce a mutable map (clone only if shared).
    pub fn into_map(self) -> Option<IndexMap<Arc<str>, Val>> {
        if let Val::Obj(m) = self {
            Some(Arc::try_unwrap(m).unwrap_or_else(|m| (*m).clone()))
        } else {
            None
        }
    }

    /// Build a Val::Arr from a Vec without an extra allocation.
    #[inline]
    pub fn arr(v: Vec<Val>) -> Self {
        Val::Arr(Arc::new(v))
    }

    /// Build a columnar `Val::IntVec` from a `Vec<i64>`.
    #[inline]
    pub fn int_vec(v: Vec<i64>) -> Self {
        Val::IntVec(Arc::new(v))
    }

    /// Build a columnar `Val::FloatVec` from a `Vec<f64>`.
    #[inline]
    pub fn float_vec(v: Vec<f64>) -> Self {
        Val::FloatVec(Arc::new(v))
    }

    /// Build a columnar `Val::StrVec` from a `Vec<Arc<str>>`.
    #[inline]
    pub fn str_vec(v: Vec<Arc<str>>) -> Self {
        Val::StrVec(Arc::new(v))
    }

    /// Build a Val::Obj from an IndexMap without an extra allocation.
    #[inline]
    pub fn obj(m: IndexMap<Arc<str>, Val>) -> Self {
        Val::Obj(Arc::new(m))
    }

    /// Intern a string key.
    #[inline]
    pub fn key(s: &str) -> Arc<str> {
        Arc::from(s)
    }
}

// ── serde_json::Value ↔ Val conversions ───────────────────────────────────────

impl From<&serde_json::Value> for Val {
    fn from(v: &serde_json::Value) -> Self {
        match v {
            serde_json::Value::Null => Val::Null,
            serde_json::Value::Bool(b) => Val::Bool(*b),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Val::Int(i)
                } else {
                    Val::Float(n.as_f64().unwrap_or(0.0))
                }
            }
            serde_json::Value::String(s) => Val::Str(Arc::from(s.as_str())),
            serde_json::Value::Array(a) => {
                // Columnar fast-path: homogeneous all-int arrays become
                // `Val::IntVec`.  Saves 3x storage for big numeric docs.
                let all_i64 = !a.is_empty()
                    && a.iter().all(|v| {
                        matches!(v,
                    serde_json::Value::Number(n) if n.is_i64())
                    });
                if all_i64 {
                    let out: Vec<i64> = a
                        .iter()
                        .filter_map(|v| {
                            if let serde_json::Value::Number(n) = v {
                                n.as_i64()
                            } else {
                                None
                            }
                        })
                        .collect();
                    return Val::IntVec(Arc::new(out));
                }
                // Homogeneous all-string arrays promote to Val::StrVec for
                // columnar filter/contains fast paths.
                let all_str =
                    !a.is_empty() && a.iter().all(|v| matches!(v, serde_json::Value::String(_)));
                if all_str {
                    let out: Vec<Arc<str>> = a
                        .iter()
                        .filter_map(|v| {
                            if let serde_json::Value::String(s) = v {
                                Some(Arc::from(s.as_str()))
                            } else {
                                None
                            }
                        })
                        .collect();
                    return Val::StrVec(Arc::new(out));
                }
                Val::Arr(Arc::new(a.iter().map(Val::from).collect()))
            }
            serde_json::Value::Object(m) => Val::Obj(Arc::new(
                m.iter()
                    .map(|(k, v)| (intern_key(k.as_str()), Val::from(v)))
                    .collect(),
            )),
        }
    }
}

impl From<Val> for serde_json::Value {
    fn from(v: Val) -> Self {
        match v {
            Val::Null => serde_json::Value::Null,
            Val::Bool(b) => serde_json::Value::Bool(b),
            Val::Int(n) => serde_json::Value::Number(n.into()),
            Val::Float(f) => {
                serde_json::Value::Number(Number::from_f64(f).unwrap_or_else(|| 0.into()))
            }
            Val::Str(s) => serde_json::Value::String(s.to_string()),
            Val::StrSlice(r) => serde_json::Value::String(r.as_str().to_string()),
            Val::Arr(a) => {
                let mut out: Vec<serde_json::Value> = Vec::with_capacity(a.len());
                match Arc::try_unwrap(a) {
                    Ok(vec) => {
                        for item in vec {
                            out.push(item.into());
                        }
                    }
                    Err(a) => {
                        for item in a.iter() {
                            out.push(item.clone().into());
                        }
                    }
                }
                serde_json::Value::Array(out)
            }
            Val::IntVec(a) => {
                let out: Vec<serde_json::Value> = a
                    .iter()
                    .map(|n| serde_json::Value::Number((*n).into()))
                    .collect();
                serde_json::Value::Array(out)
            }
            Val::FloatVec(a) => {
                let out: Vec<serde_json::Value> = a
                    .iter()
                    .map(|f| {
                        serde_json::Value::Number(Number::from_f64(*f).unwrap_or_else(|| 0.into()))
                    })
                    .collect();
                serde_json::Value::Array(out)
            }
            Val::StrVec(a) => {
                let out: Vec<serde_json::Value> = a
                    .iter()
                    .map(|s| serde_json::Value::String(s.to_string()))
                    .collect();
                serde_json::Value::Array(out)
            }
            Val::StrSliceVec(a) => {
                let out: Vec<serde_json::Value> = a
                    .iter()
                    .map(|r| serde_json::Value::String(r.as_str().to_string()))
                    .collect();
                serde_json::Value::Array(out)
            }
            Val::ObjVec(d) => {
                let n = d.nrows();
                let mut out: Vec<serde_json::Value> = Vec::with_capacity(n);
                for row in 0..n {
                    let mut map: Map<String, serde_json::Value> = Map::with_capacity(d.keys.len());
                    for (k, v) in d.keys.iter().zip(d.row_slice(row).iter()) {
                        map.insert(k.to_string(), v.clone().into());
                    }
                    out.push(serde_json::Value::Object(map));
                }
                serde_json::Value::Array(out)
            }
            Val::Obj(m) => {
                let mut map: Map<String, serde_json::Value> = Map::with_capacity(m.len());
                match Arc::try_unwrap(m) {
                    Ok(im) => {
                        for (k, v) in im {
                            map.insert(k.to_string(), v.into());
                        }
                    }
                    Err(m) => {
                        for (k, v) in m.iter() {
                            map.insert(k.to_string(), v.clone().into());
                        }
                    }
                }
                serde_json::Value::Object(map)
            }
            Val::ObjSmall(pairs) => {
                let mut map: Map<String, serde_json::Value> = Map::with_capacity(pairs.len());
                for (k, v) in pairs.iter() {
                    map.insert(k.to_string(), v.clone().into());
                }
                serde_json::Value::Object(map)
            }
        }
    }
}

// ── Lazy serde adapter ────────────────────────────────────────────────────────
//
// `ValRef<'a>` serialises a borrowed `Val` directly to the output byte
// stream, skipping the intermediate `serde_json::Value` tree that
// `From<Val> for serde_json::Value` builds.  For Obj-heavy results this
// elides N key-string clones + the `Map` rebuild per call.
//
// Non-finite floats (NaN/±Inf) are coerced to `0` to match the existing
// `From<Val>` impl, which falls back to `0.into()` when `Number::from_f64`
// returns `None`.

pub struct ValRef<'a>(pub &'a Val);

impl<'a> Serialize for ValRef<'a> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self.0 {
            Val::Null => s.serialize_unit(),
            Val::Bool(b) => s.serialize_bool(*b),
            Val::Int(n) => s.serialize_i64(*n),
            Val::Float(f) => {
                if f.is_finite() {
                    s.serialize_f64(*f)
                } else {
                    s.serialize_i64(0)
                }
            }
            Val::Str(v) => s.serialize_str(v),
            Val::StrSlice(r) => s.serialize_str(r.as_str()),
            Val::Arr(a) => {
                let mut seq = s.serialize_seq(Some(a.len()))?;
                for item in a.iter() {
                    seq.serialize_element(&ValRef(item))?;
                }
                seq.end()
            }
            Val::IntVec(a) => {
                let mut seq = s.serialize_seq(Some(a.len()))?;
                for n in a.iter() {
                    seq.serialize_element(n)?;
                }
                seq.end()
            }
            Val::FloatVec(a) => {
                let mut seq = s.serialize_seq(Some(a.len()))?;
                for f in a.iter() {
                    if f.is_finite() {
                        seq.serialize_element(f)?;
                    } else {
                        seq.serialize_element(&0i64)?;
                    }
                }
                seq.end()
            }
            Val::StrVec(a) => {
                let mut seq = s.serialize_seq(Some(a.len()))?;
                for v in a.iter() {
                    seq.serialize_element(v.as_ref())?;
                }
                seq.end()
            }
            Val::StrSliceVec(a) => {
                let mut seq = s.serialize_seq(Some(a.len()))?;
                for r in a.iter() {
                    seq.serialize_element(r.as_str())?;
                }
                seq.end()
            }
            Val::ObjVec(d) => {
                let n = d.nrows();
                let mut seq = s.serialize_seq(Some(n))?;
                struct RowRef<'a> {
                    keys: &'a [Arc<str>],
                    row: &'a [Val],
                }
                impl<'a> Serialize for RowRef<'a> {
                    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                        let mut m = s.serialize_map(Some(self.keys.len()))?;
                        for (k, v) in self.keys.iter().zip(self.row.iter()) {
                            m.serialize_entry(k.as_ref(), &ValRef(v))?;
                        }
                        m.end()
                    }
                }
                for row_idx in 0..n {
                    seq.serialize_element(&RowRef {
                        keys: &d.keys,
                        row: d.row_slice(row_idx),
                    })?;
                }
                seq.end()
            }
            Val::Obj(m) => {
                let mut map = s.serialize_map(Some(m.len()))?;
                for (k, v) in m.iter() {
                    map.serialize_entry(k.as_ref(), &ValRef(v))?;
                }
                map.end()
            }
            Val::ObjSmall(pairs) => {
                let mut map = s.serialize_map(Some(pairs.len()))?;
                for (k, v) in pairs.iter() {
                    map.serialize_entry(k.as_ref(), &ValRef(v))?;
                }
                map.end()
            }
        }
    }
}

impl Val {
    /// Serialise `self` as JSON bytes without building an intermediate
    /// `serde_json::Value` tree.  Preferred over `serde_json::to_vec(&Value::from(val))`
    /// for Obj-heavy results.
    pub fn to_json_vec(&self) -> Vec<u8> {
        serde_json::to_vec(&ValRef(self)).unwrap_or_default()
    }

    /// Parse JSON text into `Val` directly — one pass, no intermediate
    /// `serde_json::Value` tree.  Preferred over the
    /// `serde_json::from_str -> serde_json::Value -> Val::from` round-trip
    /// for hot `from_json` paths.
    pub fn from_json_str(s: &str) -> serde_json::Result<Val> {
        let mut de = serde_json::Deserializer::from_str(s);
        let v = Val::deserialize(&mut de)?;
        de.end()?;
        Ok(v)
    }

    /// Parse JSON via simd-json (SIMD-accelerated structural scanner).
    /// Requires the `simd-json` feature.  Input bytes are mutated in-place
    /// by the simd-json parser — caller must own a writable buffer.
    ///
    /// Goes through `simd_json::to_borrowed_value` (the lower-level
    /// in-place tree builder, faster than the serde shim per upstream
    /// guidance) and then walks the resulting `BorrowedValue` directly
    /// into `Val`, applying the same columnar-lane heuristics as
    /// `From<&serde_json::Value>` (all-int → `IntVec`, all-str → `StrVec`).
    #[cfg(feature = "simd-json")]
    #[cfg(feature = "simd-json")]
    pub fn from_json_simd(bytes: &mut [u8]) -> Result<Val, String> {
        // Tape path: walk simd-json's flat Vec<Node> directly into Val,
        // skipping the BorrowedValue intermediate tree.  Each Object /
        // Array node carries a `count` field telling us how many tape
        // entries belong to it (for fast skip-ahead in homogeneity probes).
        let tape = simd_json::to_tape(bytes).map_err(|e| e.to_string())?;
        let nodes = tape.0;
        let mut idx = 0usize;
        Ok(Self::from_simd_tape(&nodes, &mut idx))
    }

    /// Number of tape nodes the subtree at `idx` occupies.  Used by
    /// the shape-probe walker.  Matches simd-json's "count" semantics.
    #[cfg(feature = "simd-json")]
    fn node_span(nodes: &[simd_json::Node<'_>], idx: usize) -> usize {
        match nodes[idx] {
            simd_json::Node::Object { count, .. } | simd_json::Node::Array { count, .. } => {
                count + 1
            }
            _ => 1,
        }
    }

    /// Probe an Array of Objects starting at `start` for uniform shape
    /// (same `len`, same key sequence in same order).  Returns the
    /// shared key list as borrowed `&str` slices into the tape's
    /// scratch buffer when promotion is safe; `None` otherwise.
    #[cfg(feature = "simd-json")]
    #[allow(clippy::needless_lifetimes)]
    fn probe_obj_shape_inner<'a>(
        nodes: &'a [simd_json::Node<'a>],
        start: usize,
        n_entries: usize,
        first_len: usize,
    ) -> Option<Vec<&'a str>> {
        use simd_json::Node;
        // Grab the first Object's keys as the reference shape.
        let mut keys: Vec<&'a str> = Vec::with_capacity(first_len);
        if !matches!(nodes[start], Node::Object { len, .. } if len as usize == first_len) {
            return None;
        }
        let mut idx = start + 1;
        for _ in 0..first_len {
            match nodes[idx] {
                Node::String(s) => keys.push(s),
                _ => return None,
            }
            idx += 1;
            // Skip value subtree.
            idx += Self::node_span(nodes, idx);
        }
        let mut entry_start = idx;
        for _ in 1..n_entries {
            // Each subsequent entry: same Object header `len`, same keys.
            match nodes[entry_start] {
                Node::Object { len, .. } if len as usize == first_len => {}
                _ => return None,
            }
            let mut j = entry_start + 1;
            for k in 0..first_len {
                match nodes[j] {
                    Node::String(s) if s == keys[k] => {}
                    _ => return None,
                }
                j += 1;
                j += Self::node_span(nodes, j);
            }
            entry_start = j;
        }
        Some(keys)
    }

    /// Recursive tape materializer. `idx` advances as nodes are consumed.
    /// Honours the same columnar all-i64 / all-string lane fast paths as
    /// `from_simd_borrowed` and `From<&serde_json::Value>`.
    #[cfg(feature = "simd-json")]
    fn from_simd_tape(nodes: &[simd_json::Node<'_>], idx: &mut usize) -> Val {
        use simd_json::Node;
        use simd_json::StaticNode as SN;
        let here = nodes[*idx];
        *idx += 1;
        match here {
            Node::Static(SN::Null) => Val::Null,
            Node::Static(SN::Bool(b)) => Val::Bool(b),
            Node::Static(SN::I64(n)) => Val::Int(n),
            Node::Static(SN::U64(n)) => {
                if n <= i64::MAX as u64 {
                    Val::Int(n as i64)
                } else {
                    Val::Float(n as f64)
                }
            }
            Node::Static(SN::F64(f)) => Val::Float(f),
            Node::String(s) => Val::Str(Arc::<str>::from(s)),
            Node::Array { len, .. } => {
                if len == 0 {
                    return Val::arr(Vec::new());
                }
                let start = *idx;
                // Probe the first element to decide if we can take a
                // columnar lane. Only `Static(I64|U64)` and `String`
                // qualify; if the first child is itself an array/object
                // we won't promote (skip the probe).
                let first = nodes[start];
                let mut try_int =
                    matches!(first, Node::Static(SN::I64(_)) | Node::Static(SN::U64(_)));
                let mut try_str = matches!(first, Node::String(_));
                // Walk siblings to verify homogeneity.  A sibling is
                // identified by stepping over each child's full count.
                let mut probe = start;
                let mut counted = 0usize;
                while counted < len {
                    let n = nodes[probe];
                    match n {
                        Node::Static(SN::I64(_)) | Node::Static(SN::U64(_)) => {
                            try_str = false;
                            probe += 1;
                        }
                        Node::String(_) => {
                            try_int = false;
                            probe += 1;
                        }
                        Node::Static(_) => {
                            try_int = false;
                            try_str = false;
                            probe += 1;
                        }
                        Node::Array { count, .. } | Node::Object { count, .. } => {
                            try_int = false;
                            try_str = false;
                            probe += count + 1;
                        }
                    }
                    counted += 1;
                    if !try_int && !try_str {
                        break;
                    }
                }
                if try_int {
                    let mut out: Vec<i64> = Vec::with_capacity(len);
                    for _ in 0..len {
                        match nodes[*idx] {
                            Node::Static(SN::I64(n)) => out.push(n),
                            Node::Static(SN::U64(n)) if n <= i64::MAX as u64 => out.push(n as i64),
                            _ => unreachable!("homogeneity check passed"),
                        }
                        *idx += 1;
                    }
                    return Val::IntVec(Arc::new(out));
                }
                if try_str {
                    let mut out: Vec<Arc<str>> = Vec::with_capacity(len);
                    for _ in 0..len {
                        if let Node::String(s) = nodes[*idx] {
                            out.push(Arc::<str>::from(s));
                        }
                        *idx += 1;
                    }
                    return Val::StrVec(Arc::new(out));
                }
                // ── Phase 7 foundation (DISABLED): homogeneous-shape Object Array → ObjVec ──
                //
                // The probe + promotion code below is correct but the
                // engine's filter/map/sort/group_by/etc. handlers in
                // runtime.rs and vm.rs only accept `Val::Arr`/`Val::IntVec`/`Val::StrVec`
                // receivers — they do not match `Val::ObjVec`, so promotion
                // here breaks every downstream operation.  Real Phase 7
                // requires migrating ~30-50 match sites to handle ObjVec
                // natively (slot-indexed field reads).  Until that lands,
                // promotion is gated off; the probe path stays in source
                // so the next session can flip it on alongside the
                // handler migration.
                if let Node::Object { len: first_len, .. } = first {
                    if first_len > 0 && first_len <= 64 {
                        let shape_keys =
                            Self::probe_obj_shape_inner(nodes, start, len, first_len as usize);
                        if let Some(keys) = shape_keys {
                            let n_keys = keys.len();
                            let mut cells: Vec<Val> = Vec::with_capacity(len * n_keys);
                            for _ in 0..len {
                                debug_assert!(matches!(nodes[*idx], Node::Object { .. }));
                                *idx += 1;
                                for _ in 0..n_keys {
                                    debug_assert!(matches!(nodes[*idx], Node::String(_)));
                                    *idx += 1;
                                    cells.push(Self::from_simd_tape(nodes, idx));
                                }
                            }
                            let key_arcs: Arc<[Arc<str>]> = keys
                                .iter()
                                .map(|k| intern_key(k))
                                .collect::<Vec<_>>()
                                .into();
                            // Build typed-column mirror at simd-json
                            // promotion time.  Cost paid during cold
                            // parse anyway (cells walk).  Lights up
                            // typed slot kernels for first-call queries.
                            let stride = key_arcs.len();
                            let nrows = if stride == 0 { 0 } else { cells.len() / stride };
                            let typed = build_typed_cols_from_cells(&cells, stride, nrows);
                            return Val::ObjVec(Arc::new(ObjVecData {
                                keys: key_arcs,
                                cells,
                                typed_cols: Some(Arc::new(typed)),
                            }));
                        }
                    }
                }
                let mut out: Vec<Val> = Vec::with_capacity(len);
                for _ in 0..len {
                    out.push(Self::from_simd_tape(nodes, idx));
                }
                Val::Arr(Arc::new(out))
            }
            Node::Object { len, .. } => {
                let mut out: IndexMap<Arc<str>, Val> = IndexMap::with_capacity(len);
                for _ in 0..len {
                    // Object children are key/value pairs: key first,
                    // then value (which may be nested).
                    let key = match nodes[*idx] {
                        Node::String(s) => s,
                        _ => unreachable!("object key must be string"),
                    };
                    *idx += 1;
                    let v = Self::from_simd_tape(nodes, idx);
                    out.insert(intern_key(key), v);
                }
                Val::Obj(Arc::new(out))
            }
        }
    }

    /// Build a Val tree from a parsed `TapeData`.  String values share
    /// `tape.bytes_buf` as their parent buffer — per-string cost is one
    /// `StrRef` (Arc bump + 2 u32s), no fresh heap allocation.  Closes
    /// the cold-path `Arc<str>::from` storm that dominates
    /// `from_json_simd` on docs with many unique string fields.
    ///
    /// Includes ObjVec promotion (homogeneous-shape Object Array →
    /// columnar `Val::ObjVec` with typed_cols) so post-build queries
    /// hit slot kernels.  Object keys still intern via `intern_key`
    /// (Phase B scope: borrow keys too).
    #[cfg(feature = "simd-json")]
    pub fn from_tape_data(tape: &Arc<crate::strref::TapeData>) -> Val {
        let mut idx = 0usize;
        Self::from_tape_walk(tape, &mut idx)
    }

    #[cfg(feature = "simd-json")]
    fn from_tape_walk(tape: &Arc<crate::strref::TapeData>, idx: &mut usize) -> Val {
        use crate::strref::TapeNode;
        use simd_json::StaticNode as SN;
        let here = tape.nodes[*idx];
        *idx += 1;
        match here {
            TapeNode::Static(SN::Null) => Val::Null,
            TapeNode::Static(SN::Bool(b)) => Val::Bool(b),
            TapeNode::Static(SN::I64(n)) => Val::Int(n),
            TapeNode::Static(SN::U64(n)) => {
                if n <= i64::MAX as u64 {
                    Val::Int(n as i64)
                } else {
                    Val::Float(n as f64)
                }
            }
            TapeNode::Static(SN::F64(f)) => Val::Float(f),
            TapeNode::StringRef { start, end } => {
                Val::StrSlice(crate::strref::StrRef::slice_bytes(
                    Arc::clone(&tape.bytes_buf),
                    start as usize,
                    end as usize,
                ))
            }
            TapeNode::Array { len, .. } => {
                let len = len as usize;
                if len == 0 {
                    return Val::arr(Vec::new());
                }
                // Probe homogeneity for IntVec / FloatVec / StrSliceVec lanes.
                let first_idx = *idx;
                let first = tape.nodes[first_idx];
                let mut try_int = matches!(
                    first,
                    TapeNode::Static(SN::I64(_)) | TapeNode::Static(SN::U64(_))
                );
                let mut try_float = matches!(
                    first,
                    TapeNode::Static(SN::I64(_))
                        | TapeNode::Static(SN::U64(_))
                        | TapeNode::Static(SN::F64(_))
                );
                let mut try_str = matches!(first, TapeNode::StringRef { .. });
                let mut probe = first_idx;
                let mut counted = 0usize;
                while counted < len && (try_int || try_float || try_str) {
                    match tape.nodes[probe] {
                        TapeNode::Static(SN::I64(_)) => {
                            try_str = false;
                            probe += 1;
                        }
                        TapeNode::Static(SN::U64(n)) => {
                            if n > i64::MAX as u64 {
                                try_int = false;
                            }
                            try_str = false;
                            probe += 1;
                        }
                        TapeNode::Static(SN::F64(_)) => {
                            try_int = false;
                            try_str = false;
                            probe += 1;
                        }
                        TapeNode::StringRef { .. } => {
                            try_int = false;
                            try_float = false;
                            probe += 1;
                        }
                        TapeNode::Static(_) => {
                            try_int = false;
                            try_float = false;
                            try_str = false;
                            probe += 1;
                        }
                        TapeNode::Array { .. } | TapeNode::Object { .. } => {
                            try_int = false;
                            try_float = false;
                            try_str = false;
                            probe += tape.span(probe);
                        }
                    }
                    counted += 1;
                }
                if try_int {
                    let mut out: Vec<i64> = Vec::with_capacity(len);
                    for _ in 0..len {
                        match tape.nodes[*idx] {
                            TapeNode::Static(SN::I64(n)) => out.push(n),
                            TapeNode::Static(SN::U64(n)) if n <= i64::MAX as u64 => {
                                out.push(n as i64)
                            }
                            _ => unreachable!("homogeneity check"),
                        }
                        *idx += 1;
                    }
                    return Val::IntVec(Arc::new(out));
                }
                if try_float {
                    let mut out: Vec<f64> = Vec::with_capacity(len);
                    for _ in 0..len {
                        match tape.nodes[*idx] {
                            TapeNode::Static(SN::I64(n)) => out.push(n as f64),
                            TapeNode::Static(SN::U64(n)) => out.push(n as f64),
                            TapeNode::Static(SN::F64(f)) => out.push(f),
                            _ => unreachable!("homogeneity check"),
                        }
                        *idx += 1;
                    }
                    return Val::FloatVec(Arc::new(out));
                }
                if try_str {
                    let mut out: Vec<crate::strref::StrRef> = Vec::with_capacity(len);
                    for _ in 0..len {
                        if let TapeNode::StringRef { start, end } = tape.nodes[*idx] {
                            out.push(crate::strref::StrRef::slice_bytes(
                                Arc::clone(&tape.bytes_buf),
                                start as usize,
                                end as usize,
                            ));
                        }
                        *idx += 1;
                    }
                    return Val::StrSliceVec(Arc::new(out));
                }
                // Keep root materialisation semantically conservative:
                // build ordinary Arr<Obj> here. ObjVec promotion remains
                // available through Pipeline's memoised promotion cache,
                // where callers have already selected kernels that handle
                // ObjVec correctly. Eager root-level ObjVec can surprise
                // generic VM/builtin code that expects array buckets.
                let mut out: Vec<Val> = Vec::with_capacity(len);
                for _ in 0..len {
                    out.push(Self::from_tape_walk(tape, idx));
                }
                Val::Arr(Arc::new(out))
            }
            TapeNode::Object { len, .. } => {
                let len = len as usize;
                let mut out: IndexMap<Arc<str>, Val> = IndexMap::with_capacity(len);
                for _ in 0..len {
                    let key = match tape.nodes[*idx] {
                        TapeNode::StringRef { start, end } => {
                            tape.str_at_range(start as usize, end as usize)
                        }
                        _ => unreachable!("object key must be string"),
                    };
                    *idx += 1;
                    let v = Self::from_tape_walk(tape, idx);
                    out.insert(intern_key(key), v);
                }
                Val::Obj(Arc::new(out))
            }
        }
    }

    /// Walk a `simd_json::BorrowedValue` into a `Val`.
    /// Mirrors `From<&serde_json::Value>` with columnar all-int / all-str
    /// fast paths, but skips the serde_json::Value materialisation step.
    #[cfg(feature = "simd-json")]
    #[allow(dead_code)]
    fn from_simd_borrowed(v: &simd_json::BorrowedValue<'_>) -> Val {
        use simd_json::value::borrowed::Value as SV;
        use simd_json::StaticNode as SN;
        match v {
            SV::Static(SN::Null) => Val::Null,
            SV::Static(SN::Bool(b)) => Val::Bool(*b),
            SV::Static(SN::I64(n)) => Val::Int(*n),
            SV::Static(SN::U64(n)) => {
                if *n <= i64::MAX as u64 {
                    Val::Int(*n as i64)
                } else {
                    Val::Float(*n as f64)
                }
            }
            SV::Static(SN::F64(f)) => Val::Float(*f),
            SV::String(s) => Val::Str(Arc::<str>::from(s.as_ref())),
            SV::Array(a) => {
                // All-i64 / all-string columnar fast paths.
                let all_i64 = !a.is_empty()
                    && a.iter()
                        .all(|v| matches!(v, SV::Static(SN::I64(_)) | SV::Static(SN::U64(_))));
                if all_i64 {
                    let mut out: Vec<i64> = Vec::with_capacity(a.len());
                    for v in a.iter() {
                        if let SV::Static(SN::I64(n)) = v {
                            out.push(*n);
                        } else if let SV::Static(SN::U64(n)) = v {
                            if *n <= i64::MAX as u64 {
                                out.push(*n as i64);
                            } else {
                                // Mixed: fall back to mapped Vec<Val> below.
                                return Val::Arr(Arc::new(
                                    a.iter().map(Self::from_simd_borrowed).collect(),
                                ));
                            }
                        }
                    }
                    return Val::IntVec(Arc::new(out));
                }
                let all_str = !a.is_empty() && a.iter().all(|v| matches!(v, SV::String(_)));
                if all_str {
                    let mut out: Vec<Arc<str>> = Vec::with_capacity(a.len());
                    for v in a.iter() {
                        if let SV::String(s) = v {
                            out.push(Arc::<str>::from(s.as_ref()));
                        }
                    }
                    return Val::StrVec(Arc::new(out));
                }
                Val::Arr(Arc::new(a.iter().map(Self::from_simd_borrowed).collect()))
            }
            SV::Object(m) => {
                let mut out: IndexMap<Arc<str>, Val> = IndexMap::with_capacity(m.len());
                for (k, v) in m.iter() {
                    out.insert(intern_key(k.as_ref()), Self::from_simd_borrowed(v));
                }
                Val::Obj(Arc::new(out))
            }
        }
    }
}

// ── Direct Deserialize ────────────────────────────────────────────────────────
//
// Avoids the intermediate `serde_json::Value` tree that `b_from_json` used
// to build.  Preserves order by using IndexMap directly in the visitor.

struct ValVisitor;

impl<'de> Visitor<'de> for ValVisitor {
    type Value = Val;

    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("any JSON value")
    }

    fn visit_unit<E: de::Error>(self) -> Result<Val, E> {
        Ok(Val::Null)
    }
    fn visit_none<E: de::Error>(self) -> Result<Val, E> {
        Ok(Val::Null)
    }
    fn visit_some<D: Deserializer<'de>>(self, d: D) -> Result<Val, D::Error> {
        Val::deserialize(d)
    }
    fn visit_bool<E: de::Error>(self, b: bool) -> Result<Val, E> {
        Ok(Val::Bool(b))
    }
    fn visit_i64<E: de::Error>(self, n: i64) -> Result<Val, E> {
        Ok(Val::Int(n))
    }
    fn visit_u64<E: de::Error>(self, n: u64) -> Result<Val, E> {
        if n <= i64::MAX as u64 {
            Ok(Val::Int(n as i64))
        } else {
            Ok(Val::Float(n as f64))
        }
    }
    fn visit_f64<E: de::Error>(self, f: f64) -> Result<Val, E> {
        Ok(Val::Float(f))
    }
    fn visit_str<E: de::Error>(self, s: &str) -> Result<Val, E> {
        Ok(Val::Str(Arc::from(s)))
    }
    fn visit_string<E: de::Error>(self, s: String) -> Result<Val, E> {
        Ok(Val::Str(Arc::from(s.as_str())))
    }
    fn visit_borrowed_str<E: de::Error>(self, s: &'de str) -> Result<Val, E> {
        Ok(Val::Str(Arc::from(s)))
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut a: A) -> Result<Val, A::Error> {
        // Speculative columnar path: lock the lane on the first element.
        // While every subsequent element matches the lane, push into the
        // typed vec; on first mismatch, migrate to `Vec<Val>`.  Empty arrays
        // default to generic Val::Arr (no lane to commit to).
        enum Lane {
            Unset,
            Int(Vec<i64>),
            Str(Vec<Arc<str>>),
        }
        let cap = a.size_hint().unwrap_or(0);
        let mut lane: Lane = Lane::Unset;
        let mut fallback: Option<Vec<Val>> = None;
        while let Some(item) = a.next_element::<Val>()? {
            if let Some(v) = fallback.as_mut() {
                v.push(item);
                continue;
            }
            match (&mut lane, item) {
                (Lane::Unset, Val::Int(n)) => lane = Lane::Int(vec![n]),
                (Lane::Unset, Val::Str(s)) => lane = Lane::Str(vec![s]),
                (Lane::Unset, other) => {
                    let mut v: Vec<Val> = Vec::with_capacity(cap);
                    v.push(other);
                    fallback = Some(v);
                }
                (Lane::Int(xs), Val::Int(n)) => xs.push(n),
                (Lane::Str(xs), Val::Str(s)) => xs.push(s),
                (lane_ref, other) => {
                    let mut v: Vec<Val> = Vec::with_capacity(cap);
                    match std::mem::replace(lane_ref, Lane::Unset) {
                        Lane::Int(xs) => {
                            for n in xs {
                                v.push(Val::Int(n));
                            }
                        }
                        Lane::Str(xs) => {
                            for s in xs {
                                v.push(Val::Str(s));
                            }
                        }
                        Lane::Unset => {}
                    }
                    v.push(other);
                    fallback = Some(v);
                }
            }
        }
        Ok(match fallback {
            Some(v) => Val::arr(v),
            None => match lane {
                Lane::Int(xs) => Val::IntVec(Arc::new(xs)),
                Lane::Str(xs) => Val::StrVec(Arc::new(xs)),
                Lane::Unset => Val::arr(Vec::new()),
            },
        })
    }
    fn visit_map<A: MapAccess<'de>>(self, mut m: A) -> Result<Val, A::Error> {
        let mut out: IndexMap<Arc<str>, Val> = IndexMap::with_capacity(m.size_hint().unwrap_or(0));
        while let Some((k, v)) = m.next_entry::<String, Val>()? {
            out.insert(intern_key(k.as_str()), v);
        }
        Ok(Val::obj(out))
    }
}

impl<'de> serde::Deserialize<'de> for Val {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Val, D::Error> {
        d.deserialize_any(ValVisitor)
    }
}

#[cfg(test)]
mod valref_tests {
    use super::*;

    fn roundtrip(js: &str) {
        let v: serde_json::Value = serde_json::from_str(js).unwrap();
        let val = Val::from(&v);
        let via_tree = serde_json::to_vec(&serde_json::Value::from(val.clone())).unwrap();
        let via_ref = val.to_json_vec();
        // Both paths must serialise to byte-identical JSON (IndexMap and
        // serde_json::Map both preserve insertion order).
        assert_eq!(via_tree, via_ref, "payload: {js}");
    }

    #[test]
    fn valref_parity_scalars() {
        roundtrip(r#"null"#);
        roundtrip(r#"true"#);
        roundtrip(r#"42"#);
        roundtrip(r#"3.14"#);
        roundtrip(r#""hi""#);
    }

    #[test]
    fn valref_parity_array() {
        roundtrip(r#"[1,2,3,"x",null,true,4.5]"#);
    }

    #[test]
    fn valref_parity_object() {
        roundtrip(r#"{"a":1,"b":"x","c":[1,2],"d":{"nested":true}}"#);
    }

    #[test]
    fn valref_parity_deep() {
        roundtrip(r#"{"orders":[{"id":1,"items":[{"sku":"A","qty":2}]},{"id":2,"items":[]}]}"#);
    }

    #[test]
    fn valref_nan_inf_coerced_to_zero() {
        let val = Val::Float(f64::NAN);
        assert_eq!(val.to_json_vec(), b"0");
        let val = Val::Float(f64::INFINITY);
        assert_eq!(val.to_json_vec(), b"0");
    }

    #[cfg(feature = "simd-json")]
    #[test]
    fn from_tape_data_uses_borrowed_string_slices_and_preserves_json() {
        let js =
            br#"{"title":"Dune","tags":["sci-fi","classic"],"nested":{"name":"Paul"}}"#.to_vec();
        let tape = crate::strref::TapeData::parse(js.clone()).unwrap();
        let val = Val::from_tape_data(&tape);

        assert_eq!(val.to_json_vec(), js);

        let obj = val.as_object().unwrap();
        assert!(matches!(obj.get("title"), Some(Val::StrSlice(_))));
        match obj.get("tags").unwrap() {
            Val::StrSliceVec(items) => {
                assert_eq!(items[0].as_str(), "sci-fi");
                assert_eq!(items[1].as_str(), "classic");
            }
            other => panic!("expected StrSliceVec, got {other:?}"),
        }
        let nested = obj.get("nested").unwrap().as_object().unwrap();
        assert!(matches!(nested.get("name"), Some(Val::StrSlice(_))));
    }

    #[cfg(feature = "simd-json")]
    #[test]
    fn from_tape_data_promotes_float_arrays_and_indexes_str_slice_vec() {
        let js = br#"{"nums":[1,2.5,3],"names":["a","b"]}"#.to_vec();
        let tape = crate::strref::TapeData::parse(js).unwrap();
        let val = Val::from_tape_data(&tape);
        let obj = val.as_object().unwrap();

        match obj.get("nums").unwrap() {
            Val::FloatVec(xs) => assert_eq!(xs.as_slice(), &[1.0, 2.5, 3.0]),
            other => panic!("expected FloatVec, got {other:?}"),
        }
        let names = obj.get("names").unwrap();
        assert!(matches!(names, Val::StrSliceVec(_)));
        assert!(matches!(names.get_index(1), Val::StrSlice(_)));
        assert_eq!(names.get_index(1).as_str_ref(), Some("b"));
    }
}

// ── PartialEq for comparison ──────────────────────────────────────────────────

impl PartialEq for Val {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Val::Null, Val::Null) => true,
            (Val::Bool(a), Val::Bool(b)) => a == b,
            (Val::Str(a), Val::Str(b)) => a == b,
            (Val::Str(a), Val::StrSlice(b)) => a.as_ref() == b.as_str(),
            (Val::StrSlice(a), Val::Str(b)) => a.as_str() == b.as_ref(),
            (Val::StrSlice(a), Val::StrSlice(b)) => a.as_str() == b.as_str(),
            (Val::Int(a), Val::Int(b)) => a == b,
            (Val::Float(a), Val::Float(b)) => a == b,
            (Val::Int(a), Val::Float(b)) => (*a as f64) == *b,
            (Val::Float(a), Val::Int(b)) => *a == (*b as f64),
            (Val::IntVec(a), Val::IntVec(b)) => a == b,
            (Val::FloatVec(a), Val::FloatVec(b)) => a == b,
            (Val::StrVec(a), Val::StrVec(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for Val {}

impl std::hash::Hash for Val {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Val::Null => 0u8.hash(state),
            Val::Bool(b) => {
                1u8.hash(state);
                b.hash(state);
            }
            Val::Int(n) => {
                2u8.hash(state);
                n.hash(state);
            }
            Val::Float(f) => {
                2u8.hash(state);
                f.to_bits().hash(state);
            }
            Val::Str(s) => {
                3u8.hash(state);
                s.hash(state);
            }
            Val::StrSlice(r) => {
                3u8.hash(state);
                r.as_str().hash(state);
            }
            Val::Arr(a) => {
                4u8.hash(state);
                (Arc::as_ptr(a) as usize).hash(state);
            }
            Val::IntVec(a) => {
                4u8.hash(state);
                (Arc::as_ptr(a) as usize).hash(state);
            }
            Val::FloatVec(a) => {
                4u8.hash(state);
                (Arc::as_ptr(a) as usize).hash(state);
            }
            Val::StrVec(a) => {
                4u8.hash(state);
                (Arc::as_ptr(a) as usize).hash(state);
            }
            Val::StrSliceVec(a) => {
                4u8.hash(state);
                (Arc::as_ptr(a) as usize).hash(state);
            }
            Val::ObjVec(d) => {
                5u8.hash(state);
                (Arc::as_ptr(d) as usize).hash(state);
            }
            Val::Obj(m) => {
                5u8.hash(state);
                (Arc::as_ptr(m) as usize).hash(state);
            }
            Val::ObjSmall(p) => {
                5u8.hash(state);
                (p.as_ptr() as usize).hash(state);
            }
        }
    }
}

// ── Display ───────────────────────────────────────────────────────────────────

impl std::fmt::Display for Val {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Val::Null => write!(f, "null"),
            Val::Bool(b) => write!(f, "{}", b),
            Val::Int(n) => write!(f, "{}", n),
            Val::Float(fl) => write!(f, "{}", fl),
            Val::Str(s) => write!(f, "{}", s),
            Val::StrSlice(r) => write!(f, "{}", r.as_str()),
            other => {
                let bytes = serde_json::to_vec(&ValRef(other)).unwrap_or_default();
                f.write_str(std::str::from_utf8(&bytes).unwrap_or(""))
            }
        }
    }
}
