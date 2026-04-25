//! Borrowed-slice string view that shares a parent `Arc<str>`.
//!
//! `StrRef` is a lightweight wrapper around an owning `Arc<str>` plus
//! byte-range offsets.  Cloning it is an atomic Arc bump plus two
//! word-sized copies — no heap allocation — so methods like `slice`,
//! `split.first`, `substring` can return a view into their input
//! without allocating a fresh `Arc<str>` per row.
//!
//! Invariants (enforced at construction):
//! - `start <= end <= parent.len()`.
//! - `parent[start..end]` is valid UTF-8 (the parent is, and we never
//!   split between code units — callers construct slices via
//!   `str::char_indices` or ASCII-only byte offsets).
//!
//! `as_str()` returns the view slice.  `Deref<Target=str>` /
//! `AsRef<str>` allow existing string APIs (len, chars, find, memchr,
//! etc.) to work against a `StrRef` transparently.

use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct StrRef {
    parent: Arc<str>,
    start: u32,
    end: u32,
}

impl StrRef {
    /// Construct a full view of `parent`.  Offsets = [0, parent.len()].
    #[inline]
    pub fn from_arc(parent: Arc<str>) -> Self {
        let end = parent.len() as u32;
        Self { parent, start: 0, end }
    }

    /// Byte-range view into `parent`.  Caller must ensure the range is
    /// a valid UTF-8 boundary (checked in debug via `is_char_boundary`).
    #[inline]
    pub fn slice(parent: Arc<str>, start: usize, end: usize) -> Self {
        debug_assert!(start <= end);
        debug_assert!(end <= parent.len());
        debug_assert!(parent.is_char_boundary(start));
        debug_assert!(parent.is_char_boundary(end));
        Self {
            parent,
            start: start as u32,
            end: end as u32,
        }
    }

    #[inline] pub fn as_str(&self) -> &str {
        &self.parent[self.start as usize .. self.end as usize]
    }

    #[inline] pub fn len(&self) -> usize { (self.end - self.start) as usize }
    #[inline] pub fn is_empty(&self) -> bool { self.end == self.start }

    /// Produce an owning `Arc<str>` — allocates a fresh buffer containing
    /// the view contents.  Use only when an owning Arc is required (e.g.
    /// to insert into an `IndexMap<Arc<str>, Val>` as a key).
    #[inline]
    pub fn to_arc(&self) -> Arc<str> {
        if self.start == 0 && self.end as usize == self.parent.len() {
            Arc::clone(&self.parent)
        } else {
            Arc::<str>::from(self.as_str())
        }
    }
}

impl AsRef<str> for StrRef {
    #[inline] fn as_ref(&self) -> &str { self.as_str() }
}

impl std::ops::Deref for StrRef {
    type Target = str;
    #[inline] fn deref(&self) -> &str { self.as_str() }
}

impl std::fmt::Display for StrRef {
    #[inline] fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl PartialEq for StrRef {
    #[inline] fn eq(&self, other: &Self) -> bool { self.as_str() == other.as_str() }
}
impl Eq for StrRef {}

impl PartialEq<str> for StrRef {
    #[inline] fn eq(&self, other: &str) -> bool { self.as_str() == other }
}
impl PartialEq<&str> for StrRef {
    #[inline] fn eq(&self, other: &&str) -> bool { self.as_str() == *other }
}
impl PartialEq<Arc<str>> for StrRef {
    #[inline] fn eq(&self, other: &Arc<str>) -> bool { self.as_str() == other.as_ref() }
}

impl std::hash::Hash for StrRef {
    #[inline] fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_str().hash(state)
    }
}

impl From<Arc<str>> for StrRef {
    #[inline] fn from(a: Arc<str>) -> Self { Self::from_arc(a) }
}
impl From<&str> for StrRef {
    #[inline] fn from(s: &str) -> Self { Self::from_arc(Arc::<str>::from(s)) }
}
impl From<String> for StrRef {
    #[inline] fn from(s: String) -> Self { Self::from_arc(Arc::<str>::from(s)) }
}

// ── Tape-backed lane (simd-json, opt-in via Jetro::from_simd_lazy) ───────────
//
// Tape-aware VM foundation. The tape stores a parsed JSON document as
// a flat sequence of `TapeNode`s; string nodes carry byte offsets into
// a side `bytes_buf` (escape-decoded once at parse time so subsequent
// reads are pure offset slicing).
//
// Day 1 of the Phase 6 plan (memory/project_tape_aware_vm.md) — provides
// the data structure + constructor.  Day 2 wires opcode handlers
// (execute_tape) that walk the tape directly without materialising Val.

#[cfg(feature = "simd-json")]
#[derive(Debug, Clone, Copy)]
pub enum TapeNode {
    Static(simd_json::StaticNode),
    /// String stored as `[start..end]` byte slice into `TapeData.bytes_buf`.
    StringRef { start: u32, end: u32 },
    /// Object with `len` key/value pairs; `count` total nested nodes
    /// (including children) for fast skip-ahead.
    Object { len: u32, count: u32 },
    /// Array with `len` entries; `count` total nested nodes.
    Array  { len: u32, count: u32 },
}

#[cfg(feature = "simd-json")]
#[derive(Debug)]
pub struct TapeData {
    /// All escape-decoded string contents concatenated; node offsets
    /// reference this single buffer.  Owned by `Arc` so lookup paths
    /// can borrow `&str` slices for the lifetime of the handle.
    pub bytes_buf: Arc<[u8]>,
    /// Flat node sequence; same length as `simd_json::Tape.0`.
    pub nodes:     Vec<TapeNode>,
}

#[cfg(feature = "simd-json")]
impl TapeData {
    /// Parse `bytes` (consumed; simd-json mutates in place) into a
    /// `TapeData`.  Walks `simd_json::to_tape` and copies each string
    /// into a single owned buffer with `(start, end)` offsets so
    /// subsequent reads don't need lifetime gymnastics with the
    /// simd-json scratch region.
    pub fn parse(mut bytes: Vec<u8>) -> Result<Arc<Self>, String> {
        let cap = bytes.len();
        let tape = simd_json::to_tape(&mut bytes).map_err(|e| e.to_string())?;
        // Pre-size the side buffer at the input size — string total
        // is bounded by input size for any escape-free document, and
        // escaped docs typically shrink (\\n -> 1 byte).
        let mut buf: Vec<u8> = Vec::with_capacity(cap);
        let mut nodes: Vec<TapeNode> = Vec::with_capacity(tape.0.len());
        for n in tape.0.iter() {
            nodes.push(match n {
                simd_json::Node::Static(s) => TapeNode::Static(*s),
                simd_json::Node::String(s) => {
                    let start = buf.len();
                    buf.extend_from_slice(s.as_bytes());
                    let end = buf.len();
                    TapeNode::StringRef { start: start as u32, end: end as u32 }
                }
                simd_json::Node::Object { len, count } =>
                    TapeNode::Object { len: *len as u32, count: *count as u32 },
                simd_json::Node::Array  { len, count } =>
                    TapeNode::Array  { len: *len as u32, count: *count as u32 },
            });
        }
        Ok(Arc::new(Self {
            bytes_buf: Arc::from(buf.into_boxed_slice()),
            nodes,
        }))
    }

    /// Borrow the string at node `i` (panics if not a `StringRef`).
    /// SAFETY: simd-json validates UTF-8; the offsets are written
    /// from `&str.as_bytes()` so the slice remains valid UTF-8.
    #[inline]
    pub fn str_at(&self, i: usize) -> &str {
        match self.nodes[i] {
            TapeNode::StringRef { start, end } => unsafe {
                std::str::from_utf8_unchecked(
                    &self.bytes_buf[start as usize .. end as usize])
            },
            _ => unreachable!("str_at: node {} is not a string", i),
        }
    }

    /// Length of the top-level structure (array/object) at the tape
    /// root, or 0 for primitive roots.
    pub fn root_len(&self) -> usize {
        match self.nodes.first() {
            Some(TapeNode::Object { len, .. }) | Some(TapeNode::Array { len, .. }) =>
                *len as usize,
            _ => 0,
        }
    }

    /// Number of nodes the subtree at index `i` occupies (1 for
    /// primitives, `count + 1` for Object/Array).
    #[inline]
    pub fn span(&self, i: usize) -> usize {
        match self.nodes[i] {
            TapeNode::Object { count, .. } | TapeNode::Array { count, .. } =>
                count as usize + 1,
            _ => 1,
        }
    }
}

/// Tape-aware aggregator over `$..key`-style descendant queries.
///
/// Recursively walks the tape, accumulates numeric values found under
/// keys matching `key` at any depth.  Returns `(sum_i64, sum_f64,
/// count, min_f64, max_f64, is_float)` so the caller can project to
/// whichever aggregator the original program asked for.
/// Returns `None` when any value matched by `key` is non-numeric — caller
/// falls back to the Val path so sum/min/max/avg stay correct.  Tuple is
/// `(sum_i, sum_f, count, min_f, max_f, is_float)`.
#[cfg(feature = "simd-json")]
pub fn tape_descend_numeric_fold(
    tape: &TapeData,
    key: &str,
) -> Option<(i64, f64, usize, f64, f64, bool)> {
    let mut acc = NumAcc {
        sum_i: 0, sum_f: 0.0, count: 0,
        min_f: f64::INFINITY, max_f: f64::NEG_INFINITY,
        is_float: false, mixed: false,
    };
    walk(tape, key, 0, &mut acc);
    if acc.mixed { return None; }
    Some((acc.sum_i, acc.sum_f, acc.count, acc.min_f, acc.max_f, acc.is_float))
}

/// Type-agnostic count of every value matched by `key` at any depth.
/// Used for `$..k.count()` / `$..k.len()` where the key may bind
/// strings, objects, or other non-numeric values.
#[cfg(feature = "simd-json")]
pub fn tape_descend_count_any(tape: &TapeData, key: &str) -> usize {
    let mut count = 0usize;
    walk_count(tape, key, 0, &mut count);
    count
}

#[cfg(feature = "simd-json")]
fn walk_count(tape: &TapeData, key: &str, i: usize, count: &mut usize) -> usize {
    match tape.nodes[i] {
        TapeNode::Object { len, .. } => {
            let mut j = i + 1;
            for _ in 0..len {
                let k = tape.str_at(j);
                j += 1;
                if k == key { *count += 1; }
                j = walk_count(tape, key, j, count);
            }
            j
        }
        TapeNode::Array { len, .. } => {
            let mut j = i + 1;
            for _ in 0..len {
                j = walk_count(tape, key, j, count);
            }
            j
        }
        _ => i + 1,
    }
}

#[cfg(feature = "simd-json")]
struct NumAcc {
    sum_i: i64,
    sum_f: f64,
    count: usize,
    min_f: f64,
    max_f: f64,
    is_float: bool,
    mixed: bool,
}

#[cfg(feature = "simd-json")]
fn accumulate(n: TapeNode, acc: &mut NumAcc) {
    use simd_json::StaticNode as SN;
    match n {
        TapeNode::Static(SN::I64(v)) => {
            acc.count += 1;
            if acc.is_float { acc.sum_f += v as f64; } else { acc.sum_i += v; }
            let f = v as f64;
            if f < acc.min_f { acc.min_f = f; }
            if f > acc.max_f { acc.max_f = f; }
        }
        TapeNode::Static(SN::U64(v)) if v <= i64::MAX as u64 => {
            acc.count += 1;
            if acc.is_float { acc.sum_f += v as f64; } else { acc.sum_i += v as i64; }
            let f = v as f64;
            if f < acc.min_f { acc.min_f = f; }
            if f > acc.max_f { acc.max_f = f; }
        }
        TapeNode::Static(SN::F64(v)) => {
            if !acc.is_float { acc.sum_f = acc.sum_i as f64; acc.is_float = true; }
            acc.count += 1;
            acc.sum_f += v;
            if v < acc.min_f { acc.min_f = v; }
            if v > acc.max_f { acc.max_f = v; }
        }
        _ => { acc.mixed = true; }
    }
}

/// Tape-aware extractor: `$..key` returning every numeric value found
/// at any depth under a key matching `key`.  Returns `(ints, floats,
/// is_float)` — caller picks the representation:
///   - is_float=false -> emit Val::IntVec(ints)
///   - is_float=true  -> emit Val::FloatVec(floats) (ints already
///                       promoted to f64 in floats)
/// Returns `None` if any value matching `key` is non-numeric (caller
/// must fall back to the Val path) or if `key` is an Object/Array
/// (descendant aggregates only target leaves here).
#[cfg(feature = "simd-json")]
pub fn tape_descend_collect_numeric(
    tape: &TapeData,
    key: &str,
) -> Option<(Vec<i64>, Vec<f64>, bool)> {
    let mut acc = NumCol {
        ints: Vec::new(),
        floats: Vec::new(),
        is_float: false,
        mixed: false,
    };
    walk_collect(tape, key, 0, &mut acc);
    if acc.mixed { return None; }
    Some((acc.ints, acc.floats, acc.is_float))
}

#[cfg(feature = "simd-json")]
struct NumCol {
    ints:     Vec<i64>,
    floats:   Vec<f64>,
    is_float: bool,
    mixed:    bool,
}

#[cfg(feature = "simd-json")]
fn collect_value(n: TapeNode, acc: &mut NumCol) {
    use simd_json::StaticNode as SN;
    match n {
        TapeNode::Static(SN::I64(v)) => {
            if acc.is_float { acc.floats.push(v as f64); }
            else            { acc.ints.push(v); }
        }
        TapeNode::Static(SN::U64(v)) if v <= i64::MAX as u64 => {
            if acc.is_float { acc.floats.push(v as f64); }
            else            { acc.ints.push(v as i64); }
        }
        TapeNode::Static(SN::F64(v)) => {
            if !acc.is_float {
                acc.is_float = true;
                acc.floats = acc.ints.iter().map(|x| *x as f64).collect();
                acc.ints.clear();
            }
            acc.floats.push(v);
        }
        _ => { acc.mixed = true; }
    }
}

#[cfg(feature = "simd-json")]
fn walk_collect(tape: &TapeData, key: &str, i: usize, acc: &mut NumCol) -> usize {
    match tape.nodes[i] {
        TapeNode::Object { len, .. } => {
            let mut j = i + 1;
            for _ in 0..len {
                let k = tape.str_at(j);
                j += 1;
                if k == key {
                    collect_value(tape.nodes[j], acc);
                }
                j = walk_collect(tape, key, j, acc);
            }
            j
        }
        TapeNode::Array { len, .. } => {
            let mut j = i + 1;
            for _ in 0..len {
                j = walk_collect(tape, key, j, acc);
            }
            j
        }
        _ => i + 1,
    }
}

#[cfg(feature = "simd-json")]
fn walk(tape: &TapeData, key: &str, i: usize, acc: &mut NumAcc) -> usize {
    match tape.nodes[i] {
        TapeNode::Object { len, .. } => {
            let mut j = i + 1;
            for _ in 0..len {
                let k = tape.str_at(j);
                j += 1;
                if k == key {
                    accumulate(tape.nodes[j], acc);
                }
                j = walk(tape, key, j, acc);
            }
            j
        }
        TapeNode::Array { len, .. } => {
            let mut j = i + 1;
            for _ in 0..len {
                j = walk(tape, key, j, acc);
            }
            j
        }
        _ => i + 1,
    }
}

// ── FieldChain tape walker (Phase 6 Day 3) ──────────────────────────────────
//
// Targeted at `$.<arr>.map(<field>).<agg>()` and `$.k1.k2…kN` shapes —
// neither needs Val materialisation when the chain terminates in a
// numeric scalar (or Array of numeric scalars) and the user wants an
// aggregate.

/// Find a top-level field in the Object at node index `i`.  Returns
/// the value's node index, or `None` if `i` is not an Object or `key`
/// is missing.
#[cfg(feature = "simd-json")]
pub fn tape_object_field(tape: &TapeData, i: usize, key: &str) -> Option<usize> {
    if let TapeNode::Object { len, .. } = tape.nodes[i] {
        let mut j = i + 1;
        for _ in 0..len {
            let k = tape.str_at(j);
            j += 1;
            if k == key { return Some(j); }
            j += tape.span(j);
        }
    }
    None
}

/// Walk a sequence of object-field steps from the tape root.  Returns
/// the node index of the final subtree, or `None` if any step misses
/// or transits through a non-object.
#[cfg(feature = "simd-json")]
pub fn tape_walk_field_chain(tape: &TapeData, keys: &[&str]) -> Option<usize> {
    let mut idx = 0usize;
    for k in keys {
        idx = tape_object_field(tape, idx, k)?;
    }
    Some(idx)
}

/// `$.<arr>.map(<field>).<agg>()` numeric fold.
///
/// Walks the Array at `arr_idx`; for each entry expects an Object,
/// finds `field`, accumulates if numeric.  Returns `None` if any entry
/// is not an Object or if any matched value is non-numeric.
#[cfg(feature = "simd-json")]
pub fn tape_array_field_numeric_fold(
    tape: &TapeData,
    arr_idx: usize,
    field: &str,
) -> Option<(i64, f64, usize, f64, f64, bool)> {
    let len = match tape.nodes[arr_idx] {
        TapeNode::Array { len, .. } => len as usize,
        _ => return None,
    };
    let mut acc = NumAcc {
        sum_i: 0, sum_f: 0.0, count: 0,
        min_f: f64::INFINITY, max_f: f64::NEG_INFINITY,
        is_float: false, mixed: false,
    };
    let mut j = arr_idx + 1;
    for _ in 0..len {
        let entry = j;
        if !matches!(tape.nodes[entry], TapeNode::Object { .. }) {
            return None;
        }
        if let Some(v) = tape_object_field(tape, entry, field) {
            accumulate(tape.nodes[v], &mut acc);
            if acc.mixed { return None; }
        }
        j += tape.span(entry);
    }
    Some((acc.sum_i, acc.sum_f, acc.count, acc.min_f, acc.max_f, acc.is_float))
}

/// `$.<arr>.map(<field>)` columnar collect into IntVec/FloatVec.  Same
/// preconditions as the fold variant.
#[cfg(feature = "simd-json")]
pub fn tape_array_field_collect_numeric(
    tape: &TapeData,
    arr_idx: usize,
    field: &str,
) -> Option<(Vec<i64>, Vec<f64>, bool)> {
    let len = match tape.nodes[arr_idx] {
        TapeNode::Array { len, .. } => len as usize,
        _ => return None,
    };
    let mut acc = NumCol { ints: Vec::with_capacity(len), floats: Vec::new(), is_float: false, mixed: false };
    let mut j = arr_idx + 1;
    for _ in 0..len {
        let entry = j;
        if !matches!(tape.nodes[entry], TapeNode::Object { .. }) {
            return None;
        }
        if let Some(v) = tape_object_field(tape, entry, field) {
            collect_value(tape.nodes[v], &mut acc);
            if acc.mixed { return None; }
        }
        j += tape.span(entry);
    }
    Some((acc.ints, acc.floats, acc.is_float))
}
