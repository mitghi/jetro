//! Internal value type for the evaluator.
//!
//! `Val` is the single currency type across all execution paths — VM, pipeline,
//! and composed. Compound nodes (`Arr`, `Obj`) are `Arc`-wrapped so every
//! clone is an O(1) refcount bump. Columnar lanes (`IntVec`, `FloatVec`,
//! `StrVec`, `StrSliceVec`, `ObjVec`) reduce per-element overhead on
//! homogeneous arrays: 8 B/element instead of a 24 B `Val` enum tag + pointer.
//! All lanes materialise on demand through `as_vals()` / `into_vals()`.

use indexmap::IndexMap;
use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::ser::{Serialize, SerializeMap, SerializeSeq, Serializer};
use serde_json::{Map, Number};

use std::borrow::Cow;
use std::sync::Arc;

/// Core value type. Every variant is cheaply cloneable; compound variants
/// are O(1) via Arc refcounting. Columnar variants store homogeneous arrays
/// without per-element boxing.
#[derive(Clone, Debug)]
pub enum Val {
    /// JSON null; also used as the sentinel for missing fields and out-of-bounds indexing.
    Null,
    /// JSON boolean value.
    Bool(bool),
    /// 64-bit signed integer; preferred representation for whole-number JSON numbers.
    Int(i64),
    /// 64-bit IEEE-754 float; used when a JSON number is not representable as i64.
    Float(f64),
    /// Heap-allocated interned string; `Arc` makes cloning O(1).
    Str(Arc<str>),
    /// Borrowed string slice into a parent Arc<str>. Zero-alloc view produced
    /// by the simd-json tape path and string-slice builtins.
    StrSlice(crate::tape::StrRef),
    /// Heterogeneous array of `Val` elements; Arc-wrapped for O(1) clone.
    Arr(Arc<Vec<Val>>),
    /// Columnar lane for homogeneous integer arrays; 8 B/element vs 24 B for `Val::Int`.
    IntVec(Arc<Vec<i64>>),
    /// Columnar lane for homogeneous float arrays; avoids per-element enum tag.
    FloatVec(Arc<Vec<f64>>),
    /// Columnar lane for homogeneous Arc<str> arrays; avoids per-element enum tag.
    StrVec(Arc<Vec<Arc<str>>>),
    /// Columnar lane of borrowed-string views; emitted by map-slice fusions.
    StrSliceVec(Arc<Vec<crate::tape::StrRef>>),
    /// JSON object backed by an insertion-ordered hash map; Arc-wrapped for O(1) clone.
    Obj(Arc<IndexMap<Arc<str>, Val>>),
    /// Flat (key, value) pair slice — no hashtable. Hot path for per-row
    /// projections where allocating Arc<IndexMap> per row dominates.
    ObjSmall(Arc<[(Arc<str>, Val)]>),
    /// Struct-of-arrays representation for uniform-shape object arrays; enables zero-tag-check column aggregates.
    ObjVec(Arc<ObjVecData>),
}


/// Struct-of-arrays backing store for `Val::ObjVec`: rows share a single key schema.
/// `cells` is row-major flat: row `r`, column `c` lives at `cells[r * keys.len() + c]`.
#[derive(Debug)]
pub struct ObjVecData {
    /// Shared column names for every row; length equals the stride (number of columns).
    pub keys: Arc<[Arc<str>]>,
    /// Row-major flat cell storage; invariant: `cells.len() == keys.len() * nrows`.
    pub cells: Vec<Val>,
    /// Optional per-column typed lanes built by `build_typed_cols_from_cells`; enables
    /// zero-tag-check aggregates (sum/min/max) over typed columns.
    pub typed_cols: Option<Arc<Vec<ObjVecCol>>>,
}


/// Typed column lane for a single `ObjVecData` column; `Mixed` means the slot contains
/// heterogeneous or non-scalar values and must be accessed through the `cells` flat array.
#[derive(Debug, Clone)]
pub enum ObjVecCol {
    /// Column contains mixed or non-scalar types; fall back to `cells` for per-cell access.
    Mixed,
    /// All values in this column are `i64`; indexed by row directly without tag checks.
    Ints(Vec<i64>),
    /// All values in this column are `f64`; indexed by row directly without tag checks.
    Floats(Vec<f64>),
    /// All values in this column are `Arc<str>`; indexed by row directly without tag checks.
    Strs(Vec<Arc<str>>),
    /// All values in this column are `bool`; indexed by row directly without tag checks.
    Bools(Vec<bool>),
}


/// Inspect the flat `cells` buffer and produce per-column typed lanes for an `ObjVecData`.
/// A column becomes `Mixed` if any row disagrees with the first row's type tag.
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
    /// Return the number of columns (== `keys.len()`); also the flat-buffer row stride.
    #[inline]
    pub fn stride(&self) -> usize {
        self.keys.len()
    }

    /// Return the number of object rows stored in `cells`; derived from `cells.len() / stride`.
    #[inline]
    pub fn nrows(&self) -> usize {
        let s = self.stride();
        if s == 0 {
            0
        } else {
            self.cells.len() / s
        }
    }

    /// Return the column index for `key`, or `None` if the key is not in the schema.
    #[inline]
    pub fn slot_of(&self, key: &str) -> Option<usize> {
        self.keys.iter().position(|k| k.as_ref() == key)
    }

    /// Return a reference to the `Val` at (`row`, `slot`) in the row-major flat buffer.
    #[inline]
    pub fn cell(&self, row: usize, slot: usize) -> &Val {
        debug_assert!(slot < self.stride());
        debug_assert!(row < self.nrows());
        &self.cells[row * self.stride() + slot]
    }

    /// Iterate over every value in column `slot`, stepping across rows via `step_by(stride)`.
    pub fn column(&self, slot: usize) -> impl Iterator<Item = &Val> {
        let s = self.stride();
        debug_assert!(slot < s);
        self.cells.iter().skip(slot).step_by(s)
    }

    /// Return the contiguous slice of `cells` that represents row `row`; length == stride.
    #[inline]
    pub fn row_slice(&self, row: usize) -> &[Val] {
        let s = self.stride();
        let off = row * s;
        &self.cells[off..off + s]
    }

    /// Materialise row `row` as a heap-allocated `Val::Obj`; used at the compatibility boundary.
    #[inline]
    pub fn row_val(&self, row: usize) -> Val {
        let stride = self.stride();
        let mut m: IndexMap<Arc<str>, Val> = IndexMap::with_capacity(stride);
        for (i, k) in self.keys.iter().enumerate() {
            m.insert(Arc::clone(k), self.cell(row, i).clone());
        }
        Val::Obj(Arc::new(m))
    }
}

impl Val {
    /// Return the underlying `&str` for both owned (`Str`) and borrowed (`StrSlice`) string variants.
    /// Returns `None` for all other variants.
    #[inline]
    pub fn as_str_ref(&self) -> Option<&str> {
        match self {
            Val::Str(s) => Some(s.as_ref()),
            Val::StrSlice(r) => Some(r.as_str()),
            _ => None,
        }
    }
}


thread_local! {
    /// Shared sentinel `Val::Null` available without allocation; used by callers that need a `&Val`.
    static NULL_VAL: Val = Val::Null;
    /// Per-thread key-interning cache: maps raw `Box<str>` → shared `Arc<str>` to deduplicate
    /// object key allocations across repeated parses. Capped at 4096 entries to bound memory.
    static KEY_INTERN: std::cell::RefCell<std::collections::HashMap<Box<str>, Arc<str>>> =
        std::cell::RefCell::new(std::collections::HashMap::with_capacity(64));
}


/// Return a shared `Arc<str>` for `k`, reusing a cached copy when possible.
/// Falls back to a fresh allocation once the per-thread cache exceeds 4096 entries.
#[inline]
pub fn intern_key(k: &str) -> Arc<str> {
    const CAP: usize = 4096;
    KEY_INTERN.with(|cell| {
        let mut m = cell.borrow_mut();
        if let Some(a) = m.get(k) {
            return Arc::clone(a);
        }
        if m.len() >= CAP {
            
            
            return Arc::<str>::from(k);
        }
        let a: Arc<str> = Arc::<str>::from(k);
        m.insert(k.into(), Arc::clone(&a));
        a
    })
}


impl Val {
    /// Look up `key` in an `Obj` or `ObjSmall` value; returns `Val::Null` for any other variant or missing key.
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

    /// Index into any array-like variant with Python-style negative indexing; returns `Val::Null`
    /// for out-of-bounds or non-array variants.
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

    /// Return `true` if the value is `Val::Null`.
    #[inline]
    pub fn is_null(&self) -> bool {
        matches!(self, Val::Null)
    }
    /// Return `true` if the value is an `Int` or `Float` scalar.
    #[inline]
    pub fn is_number(&self) -> bool {
        matches!(self, Val::Int(_) | Val::Float(_))
    }
    /// Return `true` for both owned (`Str`) and borrowed (`StrSlice`) string variants.
    #[inline]
    pub fn is_string(&self) -> bool {
        matches!(self, Val::Str(_) | Val::StrSlice(_))
    }
    /// Return `true` for all array-like variants including columnar lanes; does not include `ObjVec`.
    #[inline]
    pub fn is_array(&self) -> bool {
        matches!(
            self,
            Val::Arr(_) | Val::IntVec(_) | Val::FloatVec(_) | Val::StrVec(_) | Val::StrSliceVec(_)
        )
    }

    /// Return the element count for any array-like variant (including `ObjVec`), or `None` for scalars/objects.
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

    /// Coerce `Int` or `Float` to `i64`; truncates floats via `as` cast.
    #[inline]
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Val::Int(n) => Some(*n),
            Val::Float(f) => Some(*f as i64),
            _ => None,
        }
    }

    /// Coerce `Int` or `Float` to `f64`; widens integers losslessly up to 2^53.
    #[inline]
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Val::Float(f) => Some(*f),
            Val::Int(n) => Some(*n as f64),
            _ => None,
        }
    }

    /// Convenience alias for `as_str_ref`; returns `&str` for `Str` and `StrSlice` variants.
    #[inline]
    pub fn as_str(&self) -> Option<&str> {
        self.as_str_ref()
    }

    /// Return a slice reference into the inner `Vec<Val>` only for `Val::Arr`; columnar lanes return `None`.
    #[inline]
    pub fn as_array(&self) -> Option<&[Val]> {
        if let Val::Arr(a) = self {
            Some(a)
        } else {
            None
        }
    }

    /// Alias for `arr_len`; returns element count for all array-like variants including `ObjVec`.
    #[inline]
    pub fn array_len(&self) -> Option<usize> {
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

    /// Materialise any array-like variant as a `Cow<[Val]>`, borrowing `Val::Arr` in place
    /// and heap-allocating a `Vec<Val>` for columnar lanes. Returns `None` for non-array variants.
    pub fn as_vals(&self) -> Option<Cow<'_, [Val]>> {
        match self {
            Val::Arr(a) => Some(Cow::Borrowed(a.as_slice())),
            Val::IntVec(a) => Some(Cow::Owned(a.iter().map(|n| Val::Int(*n)).collect())),
            Val::FloatVec(a) => Some(Cow::Owned(a.iter().map(|f| Val::Float(*f)).collect())),
            Val::StrVec(a) => Some(Cow::Owned(a.iter().map(|s| Val::Str(s.clone())).collect())),
            Val::StrSliceVec(a) => Some(Cow::Owned(
                a.iter().map(|s| Val::StrSlice(s.clone())).collect(),
            )),
            Val::ObjVec(d) => {
                let n = d.nrows();
                let mut out: Vec<Val> = Vec::with_capacity(n);
                for row in 0..n {
                    out.push(d.row_val(row));
                }
                Some(Cow::Owned(out))
            }
            _ => None,
        }
    }

    /// Consume any array-like variant and return an owned `Vec<Val>`, using `Arc::try_unwrap`
    /// to avoid a copy for uniquely-owned `Arr`. Returns `Err(self)` for non-array variants.
    pub fn into_vals(self) -> Result<Vec<Val>, Val> {
        match self {
            Val::Arr(a) => Ok(Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone())),
            Val::IntVec(a) => Ok(a.iter().map(|n| Val::Int(*n)).collect()),
            Val::FloatVec(a) => Ok(a.iter().map(|f| Val::Float(*f)).collect()),
            Val::StrVec(a) => Ok(a.iter().map(|s| Val::Str(Arc::clone(s))).collect()),
            Val::StrSliceVec(a) => Ok(a.iter().map(|s| Val::StrSlice(s.clone())).collect()),
            Val::ObjVec(d) => {
                let n = d.nrows();
                let mut out = Vec::with_capacity(n);
                for row in 0..n {
                    out.push(d.row_val(row));
                }
                Ok(out)
            }
            other => Err(other),
        }
    }

    /// Return a mutable reference to the inner `Vec<Val>` of a `Val::Arr`, triggering COW via
    /// `Arc::make_mut` if the Arc is shared. Returns `None` for all other variants.
    pub fn as_array_mut(&mut self) -> Option<&mut Vec<Val>> {
        if let Val::Arr(a) = self {
            Some(Arc::make_mut(a))
        } else {
            None
        }
    }

    /// Return a reference to the inner `IndexMap` for `Val::Obj` only; `ObjSmall` and `ObjVec` return `None`.
    #[inline]
    pub fn as_object(&self) -> Option<&IndexMap<Arc<str>, Val>> {
        if let Val::Obj(m) = self {
            Some(m)
        } else {
            None
        }
    }

    /// Borrow a field value by key from `Obj` or `ObjSmall`; returns `None` for all other variants.
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

    /// Return a static JSON-type name string for use in error messages and type predicates.
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

    /// Consume any array-like variant and return `Some(Vec<Val>)`; returns `None` for scalars and objects.
    /// Prefer `into_vals` when callers need to distinguish non-array inputs.
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

    /// Consume a `Val::Obj` and return the inner `IndexMap`, using `Arc::try_unwrap` to avoid a copy
    /// when the Arc is uniquely owned. Returns `None` for all other variants.
    pub fn into_map(self) -> Option<IndexMap<Arc<str>, Val>> {
        if let Val::Obj(m) = self {
            Some(Arc::try_unwrap(m).unwrap_or_else(|m| (*m).clone()))
        } else {
            None
        }
    }

    /// Wrap a `Vec<Val>` in an `Arc` and return `Val::Arr`; preferred constructor for heterogeneous arrays.
    #[inline]
    pub fn arr(v: Vec<Val>) -> Self {
        Val::Arr(Arc::new(v))
    }

    /// Wrap a `Vec<i64>` in an `Arc` and return `Val::IntVec`; preferred constructor for integer columnar lanes.
    #[inline]
    pub fn int_vec(v: Vec<i64>) -> Self {
        Val::IntVec(Arc::new(v))
    }

    /// Wrap a `Vec<f64>` in an `Arc` and return `Val::FloatVec`; preferred constructor for float columnar lanes.
    #[inline]
    pub fn float_vec(v: Vec<f64>) -> Self {
        Val::FloatVec(Arc::new(v))
    }

    /// Wrap a `Vec<Arc<str>>` in an `Arc` and return `Val::StrVec`; preferred constructor for string columnar lanes.
    #[inline]
    pub fn str_vec(v: Vec<Arc<str>>) -> Self {
        Val::StrVec(Arc::new(v))
    }

    /// Wrap an `IndexMap` in an `Arc` and return `Val::Obj`; preferred constructor for object values.
    #[inline]
    pub fn obj(m: IndexMap<Arc<str>, Val>) -> Self {
        Val::Obj(Arc::new(m))
    }

    /// Allocate a fresh `Arc<str>` from a `&str` without interning; use `intern_key` for repeated keys.
    #[inline]
    pub fn key(s: &str) -> Arc<str> {
        Arc::from(s)
    }
}


/// Convert a borrowed `serde_json::Value` tree into `Val`, promoting homogeneous arrays
/// to columnar `IntVec` or `StrVec` lanes on the fly.
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

/// Convert a `Val` into a `serde_json::Value` tree, materialising all columnar lanes.
/// Uses `Arc::try_unwrap` to avoid cloning uniquely-owned `Arr` and `Obj` data.
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


/// Lazy serde `Serialize` adapter for `&Val` that avoids allocating an intermediate
/// `serde_json::Value` tree; used by `to_json_vec` and the `Display` impl.
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
    /// Serialise `self` to a compact JSON byte vector via `ValRef`, without building a `serde_json::Value` tree.
    pub fn to_json_vec(&self) -> Vec<u8> {
        serde_json::to_vec(&ValRef(self)).unwrap_or_default()
    }

    /// Parse a JSON string into `Val`, automatically promoting homogeneous arrays to columnar lanes.
    pub fn from_json_str(s: &str) -> serde_json::Result<Val> {
        let mut de = serde_json::Deserializer::from_str(s);
        let v = Val::deserialize(&mut de)?;
        de.end()?;
        Ok(v)
    }

    /// Parse a mutable byte slice in-place using simd-json, then walk the tape to produce `Val`.
    /// The input buffer is mutated by simd-json's in-place unescaping; caller must own the bytes.
    #[cfg(feature = "simd-json")]
    #[cfg(feature = "simd-json")]
    pub fn from_json_simd(bytes: &mut [u8]) -> Result<Val, String> {
        
        
        let tape = simd_json::to_tape(bytes).map_err(|e| e.to_string())?;
        let nodes = tape.0;
        let mut idx = 0usize;
        Ok(Self::from_simd_tape(&nodes, &mut idx))
    }

    /// Return the number of tape slots consumed by the node at `idx`: 1 for scalars/strings,
    /// `count + 1` for arrays and objects (count is the total descendant node count).
    #[cfg(feature = "simd-json")]
    fn node_span(nodes: &[simd_json::Node<'_>], idx: usize) -> usize {
        match nodes[idx] {
            simd_json::Node::Object { count, .. } | simd_json::Node::Array { count, .. } => {
                count + 1
            }
            _ => 1,
        }
    }

    /// Walk the tape starting at `start` and verify that `n_entries` consecutive objects all share
    /// the same `first_len` keys in the same order; returns the shared key slice or `None` on mismatch.
    #[cfg(feature = "simd-json")]
    #[allow(clippy::needless_lifetimes)]
    fn probe_obj_shape_inner<'a>(
        nodes: &'a [simd_json::Node<'a>],
        start: usize,
        n_entries: usize,
        first_len: usize,
    ) -> Option<Vec<&'a str>> {
        use simd_json::Node;
        
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
            
            idx += Self::node_span(nodes, idx);
        }
        let mut entry_start = idx;
        for _ in 1..n_entries {
            
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

    /// Recursively materialise a `Val` from a simd-json tape by advancing `idx`; promotes
    /// homogeneous arrays to columnar lanes and uniform object arrays to `ObjVec`.
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
                
                
                let first = nodes[start];
                let mut try_int =
                    matches!(first, Node::Static(SN::I64(_)) | Node::Static(SN::U64(_)));
                let mut try_str = matches!(first, Node::String(_));
                
                
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

    /// Materialise a `Val` from a `TapeData` (a pre-parsed, Arc-owned simd-json tape), producing
    /// `StrSlice` views into the tape buffer instead of allocating new `Arc<str>` for strings.
    #[cfg(feature = "simd-json")]
    pub fn from_tape_data(tape: &Arc<crate::tape::TapeData>) -> Val {
        let mut idx = 0usize;
        Self::from_tape_walk(tape, &mut idx)
    }

    /// Recursive walk helper for `from_tape_data`; advances `idx` through `TapeNode` entries,
    /// emitting `StrSlice` for string nodes and promoting homogeneous arrays to columnar lanes.
    #[cfg(feature = "simd-json")]
    fn from_tape_walk(tape: &Arc<crate::tape::TapeData>, idx: &mut usize) -> Val {
        use crate::tape::TapeNode;
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
            TapeNode::String(_) => Val::StrSlice(tape.str_ref_at(*idx - 1)),
            TapeNode::Array { len, .. } => {
                if len == 0 {
                    return Val::arr(Vec::new());
                }
                
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
                let mut try_str = matches!(first, TapeNode::String(_));
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
                        TapeNode::String(_) => {
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
                    let mut out: Vec<crate::tape::StrRef> = Vec::with_capacity(len);
                    for _ in 0..len {
                        if let TapeNode::String(_) = tape.nodes[*idx] {
                            out.push(tape.str_ref_at(*idx));
                        }
                        *idx += 1;
                    }
                    return Val::StrSliceVec(Arc::new(out));
                }
                
                
                let mut out: Vec<Val> = Vec::with_capacity(len);
                for _ in 0..len {
                    out.push(Self::from_tape_walk(tape, idx));
                }
                Val::Arr(Arc::new(out))
            }
            TapeNode::Object { len, .. } => {
                let mut out: IndexMap<Arc<str>, Val> = IndexMap::with_capacity(len);
                for _ in 0..len {
                    let key = match tape.nodes[*idx] {
                        TapeNode::String(s) => s,
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

    /// Convert a `simd_json::BorrowedValue` into `Val`, promoting homogeneous integer and string
    /// arrays to columnar lanes. Kept as a fallback for borrowed-value API callers.
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


/// serde `Visitor` implementation that drives `Val`'s `Deserialize`; promotes homogeneous
/// integer and string sequences to `IntVec`/`StrVec` columnar lanes without a two-pass scan.
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

/// Deserialise any JSON token stream into `Val` via `ValVisitor`, routing through `deserialize_any`
/// so both self-describing (JSON) and non-self-describing formats work transparently.
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
        let tape = crate::tape::TapeData::parse(js.clone()).unwrap();
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
        let tape = crate::tape::TapeData::parse(js).unwrap();
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


/// Structural equality for scalar and columnar variants; compound variants (Arr/Obj/ObjVec)
/// are intentionally not deeply compared here to avoid O(n) surprises on large trees.
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

/// Identity-based hashing for compound variants (pointer hash) and value-based for scalars;
/// enables `Val` as a `HashMap` key while keeping compound hashing O(1).
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


/// Human-readable display: scalars and strings emit their raw value; compound variants
/// fall back to compact JSON serialisation via `ValRef` to avoid allocating a tree.
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
