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
        Self {
            parent,
            start: 0,
            end,
        }
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

    /// Byte-range view into a UTF-8-validated `Arc<[u8]>` parent.
    /// Used by simd-json tape ingestion: tape's `bytes_buf` is one
    /// shared `Arc<[u8]>` for the whole document; per-string `StrRef`s
    /// slice into it with no per-string heap alloc.
    ///
    /// SAFETY: caller must ensure `parent[start..end]` is valid UTF-8.
    /// simd-json validates the entire input as UTF-8 at parse time,
    /// so any range covering a `Node::String(&str)` slice is safe.
    #[inline]
    pub fn slice_bytes(parent: Arc<[u8]>, start: usize, end: usize) -> Self {
        debug_assert!(start <= end);
        debug_assert!(end <= parent.len());
        // Reinterpret the fat pointer: Arc<[u8]> and Arc<str> have
        // identical layout (pointer + length).  UTF-8 validity is the
        // caller's contract.
        let parent_str: Arc<str> = unsafe { Arc::from_raw(Arc::into_raw(parent) as *const str) };
        Self {
            parent: parent_str,
            start: start as u32,
            end: end as u32,
        }
    }

    #[inline]
    pub fn as_str(&self) -> &str {
        &self.parent[self.start as usize..self.end as usize]
    }

    #[inline]
    pub fn len(&self) -> usize {
        (self.end - self.start) as usize
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.end == self.start
    }

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
    #[inline]
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl std::ops::Deref for StrRef {
    type Target = str;
    #[inline]
    fn deref(&self) -> &str {
        self.as_str()
    }
}

impl std::fmt::Display for StrRef {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl PartialEq for StrRef {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_str() == other.as_str()
    }
}
impl Eq for StrRef {}

impl PartialEq<str> for StrRef {
    #[inline]
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}
impl PartialEq<&str> for StrRef {
    #[inline]
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}
impl PartialEq<Arc<str>> for StrRef {
    #[inline]
    fn eq(&self, other: &Arc<str>) -> bool {
        self.as_str() == other.as_ref()
    }
}

impl std::hash::Hash for StrRef {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_str().hash(state)
    }
}

impl From<Arc<str>> for StrRef {
    #[inline]
    fn from(a: Arc<str>) -> Self {
        Self::from_arc(a)
    }
}
impl From<&str> for StrRef {
    #[inline]
    fn from(s: &str) -> Self {
        Self::from_arc(Arc::<str>::from(s))
    }
}
impl From<String> for StrRef {
    #[inline]
    fn from(s: String) -> Self {
        Self::from_arc(Arc::<str>::from(s))
    }
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
    StringRef {
        start: u32,
        end: u32,
    },
    /// Object with `len` key/value pairs; `count` total nested nodes
    /// (including children) for fast skip-ahead.
    Object {
        len: u32,
        count: u32,
    },
    /// Array with `len` entries; `count` total nested nodes.
    Array {
        len: u32,
        count: u32,
    },
}

#[cfg(feature = "simd-json")]
#[derive(Debug)]
pub struct TapeData {
    /// All escape-decoded string contents concatenated; node offsets
    /// reference this single buffer.  Owned by `Arc` so lookup paths
    /// can borrow `&str` slices for the lifetime of the handle.
    pub bytes_buf: Arc<[u8]>,
    /// Flat node sequence; same length as `simd_json::Tape.0`.
    pub nodes: Vec<TapeNode>,
}

#[cfg(feature = "simd-json")]
impl TapeData {
    /// Parse `bytes` (consumed; simd-json mutates in place) into a
    /// `TapeData`.  Walks `simd_json::to_tape` and copies each string
    /// into a single owned buffer with `(start, end)` offsets so
    /// subsequent reads don't need lifetime gymnastics with the
    /// simd-json scratch region.
    pub fn parse(mut bytes: Vec<u8>) -> Result<Arc<Self>, String> {
        // simd-json escape-decodes strings in place inside `bytes`.
        // Each `Node::String(&str)` borrows a slice of that buffer.
        // Capture per-string (offset, len) BEFORE moving `bytes`, then
        // hand `bytes` over as the side buffer — zero string copies.
        // Fallback to `extend_from_slice` only for the (rare) case
        // where simd-json returns a slice outside the input buffer.
        let base = bytes.as_ptr() as usize;
        let bytes_len = bytes.len();
        let limit = base + bytes_len;
        let tape = simd_json::to_tape(&mut bytes).map_err(|e| e.to_string())?;
        let mut nodes: Vec<TapeNode> = Vec::with_capacity(tape.0.len());
        let mut extra_buf: Vec<u8> = Vec::new();
        for n in tape.0.iter() {
            nodes.push(match n {
                simd_json::Node::Static(s) => TapeNode::Static(*s),
                simd_json::Node::String(s) => {
                    let p = s.as_ptr() as usize;
                    if p >= base && p + s.len() <= limit {
                        let off = p - base;
                        TapeNode::StringRef {
                            start: off as u32,
                            end: (off + s.len()) as u32,
                        }
                    } else {
                        let off = bytes_len + extra_buf.len();
                        extra_buf.extend_from_slice(s.as_bytes());
                        TapeNode::StringRef {
                            start: off as u32,
                            end: (off + s.len()) as u32,
                        }
                    }
                }
                simd_json::Node::Object { len, count } => TapeNode::Object {
                    len: *len as u32,
                    count: *count as u32,
                },
                simd_json::Node::Array { len, count } => TapeNode::Array {
                    len: *len as u32,
                    count: *count as u32,
                },
            });
        }
        drop(tape);
        let bytes_buf: Arc<[u8]> = if extra_buf.is_empty() {
            Arc::from(bytes.into_boxed_slice())
        } else {
            let mut combined = bytes;
            combined.extend_from_slice(&extra_buf);
            Arc::from(combined.into_boxed_slice())
        };
        Ok(Arc::new(Self { bytes_buf, nodes }))
    }

    /// Borrow string contents in `bytes_buf` at the given byte range.
    /// SAFETY: range was originally produced from a simd-json
    /// `Node::String(&str)` slice — UTF-8 is validated by the parser.
    #[inline]
    pub fn str_at_range(&self, start: usize, end: usize) -> &str {
        unsafe { std::str::from_utf8_unchecked(&self.bytes_buf[start..end]) }
    }

    /// Borrow the string at node `i` (panics if not a `StringRef`).
    /// SAFETY: simd-json validates UTF-8; the offsets are written
    /// from `&str.as_bytes()` so the slice remains valid UTF-8.
    #[inline]
    pub fn str_at(&self, i: usize) -> &str {
        match self.nodes[i] {
            TapeNode::StringRef { start, end } => unsafe {
                std::str::from_utf8_unchecked(&self.bytes_buf[start as usize..end as usize])
            },
            _ => unreachable!("str_at: node {} is not a string", i),
        }
    }

    /// Length of the top-level structure (array/object) at the tape
    /// root, or 0 for primitive roots.
    pub fn root_len(&self) -> usize {
        match self.nodes.first() {
            Some(TapeNode::Object { len, .. }) | Some(TapeNode::Array { len, .. }) => *len as usize,
            _ => 0,
        }
    }

    /// Number of nodes the subtree at index `i` occupies (1 for
    /// primitives, `count + 1` for Object/Array).
    #[inline]
    pub fn span(&self, i: usize) -> usize {
        match self.nodes[i] {
            TapeNode::Object { count, .. } | TapeNode::Array { count, .. } => count as usize + 1,
            _ => 1,
        }
    }
}

/// Numeric aggregate over an array node on the tape — sum/min/max/avg/count.
/// Walks the array elements directly without building any Val.  Returns
/// (sum_i, sum_f, count, min_f, max_f, is_float, mixed_non_numeric).
/// On `mixed_non_numeric=true` caller bails to the Val path so semantics
/// stay correct on heterogeneous arrays.
#[cfg(feature = "simd-json")]
pub fn tape_array_numeric_fold(
    tape: &TapeData,
    arr_idx: usize,
) -> Option<(i64, f64, usize, f64, f64, bool)> {
    let len = match tape.nodes[arr_idx] {
        TapeNode::Array { len, .. } => len as usize,
        _ => return None,
    };
    let mut acc = NumAcc {
        sum_i: 0,
        sum_f: 0.0,
        count: 0,
        min_f: f64::INFINITY,
        max_f: f64::NEG_INFINITY,
        is_float: false,
        mixed: false,
    };
    let mut j = arr_idx + 1;
    for _ in 0..len {
        match tape.nodes[j] {
            TapeNode::Static(simd_json::StaticNode::I64(n)) => {
                acc.count += 1;
                if acc.is_float {
                    let f = n as f64;
                    acc.sum_f += f;
                    if f < acc.min_f {
                        acc.min_f = f;
                    }
                    if f > acc.max_f {
                        acc.max_f = f;
                    }
                } else {
                    acc.sum_i = acc.sum_i.wrapping_add(n);
                    let f = n as f64;
                    if f < acc.min_f {
                        acc.min_f = f;
                    }
                    if f > acc.max_f {
                        acc.max_f = f;
                    }
                }
            }
            TapeNode::Static(simd_json::StaticNode::U64(u)) => {
                acc.count += 1;
                let n = u as i64;
                if acc.is_float {
                    let f = u as f64;
                    acc.sum_f += f;
                    if f < acc.min_f {
                        acc.min_f = f;
                    }
                    if f > acc.max_f {
                        acc.max_f = f;
                    }
                } else {
                    acc.sum_i = acc.sum_i.wrapping_add(n);
                    let f = u as f64;
                    if f < acc.min_f {
                        acc.min_f = f;
                    }
                    if f > acc.max_f {
                        acc.max_f = f;
                    }
                }
            }
            TapeNode::Static(simd_json::StaticNode::F64(f)) => {
                acc.count += 1;
                if !acc.is_float {
                    acc.sum_f = acc.sum_i as f64;
                    acc.is_float = true;
                }
                acc.sum_f += f;
                if f < acc.min_f {
                    acc.min_f = f;
                }
                if f > acc.max_f {
                    acc.max_f = f;
                }
            }
            _ => return None,
        }
        j += tape.span(j);
    }
    Some((
        acc.sum_i,
        acc.sum_f,
        acc.count,
        acc.min_f,
        acc.max_f,
        acc.is_float,
    ))
}

/// Element count of an array node on the tape — used for `.len()` /
/// `.count()` aggregates.  O(1) (header-encoded).
#[cfg(feature = "simd-json")]
pub fn tape_array_count(tape: &TapeData, arr_idx: usize) -> Option<usize> {
    match tape.nodes[arr_idx] {
        TapeNode::Array { len, .. } => Some(len as usize),
        _ => None,
    }
}

/// Same as `tape_array_field_numeric_fold` but kept under a different
/// name for clarity in the new pipeline tape-aware fast path; thin
/// wrapper that forwards to the existing implementation.
#[cfg(feature = "simd-json")]
pub fn tape_array_project_numeric_fold(
    tape: &TapeData,
    arr_idx: usize,
    field: &str,
) -> Option<(i64, f64, usize, f64, f64, bool)> {
    tape_array_field_numeric_fold(tape, arr_idx, field)
}

#[cfg(feature = "simd-json")]
fn _tape_array_project_numeric_fold_unused(
    tape: &TapeData,
    arr_idx: usize,
    field: &str,
) -> Option<(i64, f64, usize, f64, f64, bool)> {
    let len = match tape.nodes[arr_idx] {
        TapeNode::Array { len, .. } => len as usize,
        _ => return None,
    };
    let mut acc = NumAcc {
        sum_i: 0,
        sum_f: 0.0,
        count: 0,
        min_f: f64::INFINITY,
        max_f: f64::NEG_INFINITY,
        is_float: false,
        mixed: false,
    };
    let mut j = arr_idx + 1;
    for _ in 0..len {
        let elem_idx = j;
        let elem_span = tape.span(elem_idx);
        match tape.nodes[elem_idx] {
            TapeNode::Object { len: olen, .. } => {
                let mut k = elem_idx + 1;
                let mut hit_idx: Option<usize> = None;
                for _ in 0..olen {
                    let kn = tape.str_at(k);
                    k += 1;
                    if kn == field {
                        hit_idx = Some(k);
                        break;
                    }
                    k += tape.span(k);
                }
                if let Some(vi) = hit_idx {
                    match tape.nodes[vi] {
                        TapeNode::Static(simd_json::StaticNode::I64(n)) => {
                            acc.count += 1;
                            if acc.is_float {
                                let f = n as f64;
                                acc.sum_f += f;
                                if f < acc.min_f {
                                    acc.min_f = f;
                                }
                                if f > acc.max_f {
                                    acc.max_f = f;
                                }
                            } else {
                                acc.sum_i = acc.sum_i.wrapping_add(n);
                                let f = n as f64;
                                if f < acc.min_f {
                                    acc.min_f = f;
                                }
                                if f > acc.max_f {
                                    acc.max_f = f;
                                }
                            }
                        }
                        TapeNode::Static(simd_json::StaticNode::U64(u)) => {
                            acc.count += 1;
                            let f = u as f64;
                            if acc.is_float {
                                acc.sum_f += f;
                            } else {
                                acc.sum_i = acc.sum_i.wrapping_add(u as i64);
                            }
                            if f < acc.min_f {
                                acc.min_f = f;
                            }
                            if f > acc.max_f {
                                acc.max_f = f;
                            }
                        }
                        TapeNode::Static(simd_json::StaticNode::F64(f)) => {
                            acc.count += 1;
                            if !acc.is_float {
                                acc.sum_f = acc.sum_i as f64;
                                acc.is_float = true;
                            }
                            acc.sum_f += f;
                            if f < acc.min_f {
                                acc.min_f = f;
                            }
                            if f > acc.max_f {
                                acc.max_f = f;
                            }
                        }
                        _ => return None,
                    }
                }
            }
            _ => return None,
        }
        j += elem_span;
    }
    Some((
        acc.sum_i,
        acc.sum_f,
        acc.count,
        acc.min_f,
        acc.max_f,
        acc.is_float,
    ))
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
        sum_i: 0,
        sum_f: 0.0,
        count: 0,
        min_f: f64::INFINITY,
        max_f: f64::NEG_INFINITY,
        is_float: false,
        mixed: false,
    };
    walk(tape, key, 0, &mut acc);
    if acc.mixed {
        return None;
    }
    Some((
        acc.sum_i,
        acc.sum_f,
        acc.count,
        acc.min_f,
        acc.max_f,
        acc.is_float,
    ))
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
                if k == key {
                    *count += 1;
                }
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
            if acc.is_float {
                acc.sum_f += v as f64;
            } else {
                acc.sum_i += v;
            }
            let f = v as f64;
            if f < acc.min_f {
                acc.min_f = f;
            }
            if f > acc.max_f {
                acc.max_f = f;
            }
        }
        TapeNode::Static(SN::U64(v)) if v <= i64::MAX as u64 => {
            acc.count += 1;
            if acc.is_float {
                acc.sum_f += v as f64;
            } else {
                acc.sum_i += v as i64;
            }
            let f = v as f64;
            if f < acc.min_f {
                acc.min_f = f;
            }
            if f > acc.max_f {
                acc.max_f = f;
            }
        }
        TapeNode::Static(SN::F64(v)) => {
            if !acc.is_float {
                acc.sum_f = acc.sum_i as f64;
                acc.is_float = true;
            }
            acc.count += 1;
            acc.sum_f += v;
            if v < acc.min_f {
                acc.min_f = v;
            }
            if v > acc.max_f {
                acc.max_f = v;
            }
        }
        _ => {
            acc.mixed = true;
        }
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
    if acc.mixed {
        return None;
    }
    Some((acc.ints, acc.floats, acc.is_float))
}

#[cfg(feature = "simd-json")]
struct NumCol {
    ints: Vec<i64>,
    floats: Vec<f64>,
    is_float: bool,
    mixed: bool,
}

#[cfg(feature = "simd-json")]
fn collect_value(n: TapeNode, acc: &mut NumCol) {
    use simd_json::StaticNode as SN;
    match n {
        TapeNode::Static(SN::I64(v)) => {
            if acc.is_float {
                acc.floats.push(v as f64);
            } else {
                acc.ints.push(v);
            }
        }
        TapeNode::Static(SN::U64(v)) if v <= i64::MAX as u64 => {
            if acc.is_float {
                acc.floats.push(v as f64);
            } else {
                acc.ints.push(v as i64);
            }
        }
        TapeNode::Static(SN::F64(v)) => {
            if !acc.is_float {
                acc.is_float = true;
                acc.floats = acc.ints.iter().map(|x| *x as f64).collect();
                acc.ints.clear();
            }
            acc.floats.push(v);
        }
        _ => {
            acc.mixed = true;
        }
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
            if k == key {
                return Some(j);
            }
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

/// Walk a field chain starting at `start` (typically an array entry's
/// Object node).  `keys.is_empty()` returns `Some(start)`.
#[cfg(feature = "simd-json")]
pub fn tape_walk_field_chain_from(tape: &TapeData, start: usize, keys: &[&str]) -> Option<usize> {
    let mut idx = start;
    for k in keys {
        idx = tape_object_field(tape, idx, k)?;
    }
    Some(idx)
}

/// Iterate the entry node indices of a tape Array.  Returns `None` if
/// `arr_idx` is not an Array.  The iterator advances by `tape.span(j)`
/// per element so it walks past nested children correctly.
#[cfg(feature = "simd-json")]
pub fn tape_array_iter(tape: &TapeData, arr_idx: usize) -> Option<TapeArrayIter<'_>> {
    let len = match tape.nodes[arr_idx] {
        TapeNode::Array { len, .. } => len as usize,
        _ => return None,
    };
    Some(TapeArrayIter {
        tape,
        cursor: arr_idx + 1,
        remaining: len,
    })
}

#[cfg(feature = "simd-json")]
pub struct TapeArrayIter<'a> {
    tape: &'a TapeData,
    cursor: usize,
    remaining: usize,
}

#[cfg(feature = "simd-json")]
impl<'a> Iterator for TapeArrayIter<'a> {
    type Item = usize;
    #[inline]
    fn next(&mut self) -> Option<usize> {
        if self.remaining == 0 {
            return None;
        }
        let entry = self.cursor;
        self.cursor += self.tape.span(entry);
        self.remaining -= 1;
        Some(entry)
    }
}

/// JSON truthiness for a tape value at node `i`.  `false` / `null` /
/// `0` / `0.0` / empty string / empty container all read as falsy;
/// everything else truthy.  Mirrors Val-side `is_truthy`.
#[cfg(feature = "simd-json")]
pub fn tape_value_truthy(tape: &TapeData, idx: usize) -> bool {
    tape_json_view(tape, idx).truthy()
}

#[cfg(feature = "simd-json")]
#[inline]
pub fn tape_json_view<'a>(tape: &'a TapeData, idx: usize) -> crate::util::JsonView<'a> {
    use simd_json::StaticNode as SN;
    match tape.nodes[idx] {
        TapeNode::Static(SN::Null) => crate::util::JsonView::Null,
        TapeNode::Static(SN::Bool(b)) => crate::util::JsonView::Bool(b),
        TapeNode::Static(SN::I64(n)) => crate::util::JsonView::Int(n),
        TapeNode::Static(SN::U64(n)) => crate::util::JsonView::UInt(n),
        TapeNode::Static(SN::F64(f)) => crate::util::JsonView::Float(f),
        TapeNode::StringRef { .. } => crate::util::JsonView::Str(tape.str_at(idx)),
        TapeNode::Array { len, .. } => crate::util::JsonView::ArrayLen(len as usize),
        TapeNode::Object { len, .. } => crate::util::JsonView::ObjectLen(len as usize),
    }
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
        sum_i: 0,
        sum_f: 0.0,
        count: 0,
        min_f: f64::INFINITY,
        max_f: f64::NEG_INFINITY,
        is_float: false,
        mixed: false,
    };
    let mut ic: Option<ShapeIC> = None;
    let mut j = arr_idx + 1;
    for _ in 0..len {
        let entry = j;
        if !matches!(tape.nodes[entry], TapeNode::Object { .. }) {
            return None;
        }
        let cached = ic.as_ref().filter(|c| shape_ic_check(tape, entry, c));
        let v: Option<usize> = if let Some(c) = cached {
            if c.rel_offs[0] == u32::MAX {
                None
            } else {
                Some(entry + c.rel_offs[0] as usize)
            }
        } else {
            ic = shape_ic_build(tape, entry, [field, ""]);
            ic.as_ref().and_then(|c| {
                if c.rel_offs[0] == u32::MAX {
                    None
                } else {
                    Some(entry + c.rel_offs[0] as usize)
                }
            })
        };
        if let Some(v) = v {
            accumulate(tape.nodes[v], &mut acc);
            if acc.mixed {
                return None;
            }
        }
        j += tape.span(entry);
    }
    Some((
        acc.sum_i,
        acc.sum_f,
        acc.count,
        acc.min_f,
        acc.max_f,
        acc.is_float,
    ))
}

/// Comparable literal kind for tape-side filter predicates.
#[cfg(feature = "simd-json")]
#[derive(Debug, Clone)]
pub enum TapeLit<'a> {
    Int(i64),
    Float(f64),
    Str(&'a str),
    Bool(bool),
    Null,
}

/// 6-way comparison op encoded as a small enum the tape executor uses.
#[cfg(feature = "simd-json")]
#[derive(Debug, Clone, Copy)]
pub enum TapeCmp {
    Eq,
    Neq,
    Lt,
    Lte,
    Gt,
    Gte,
}

/// Predicate tree for tape-side filters.  Compositions over `Cmp`
/// leaves via boolean conjunction/disjunction.  Negation is folded into
/// the comparison op at classifier time (Eq↔Neq, Lt↔Gte, …) so it does
/// not need its own variant.
#[cfg(feature = "simd-json")]
#[derive(Debug, Clone)]
pub enum TapePred<'a> {
    Cmp {
        field: &'a str,
        op: TapeCmp,
        lit: TapeLit<'a>,
    },
    And(Vec<TapePred<'a>>),
    Or(Vec<TapePred<'a>>),
}

#[cfg(feature = "simd-json")]
impl<'a> TapePred<'a> {
    /// Evaluate the predicate against an Object node `entry`.  Missing
    /// fields are treated as not-matching.
    pub fn eval(&self, tape: &TapeData, entry: usize) -> bool {
        match self {
            TapePred::Cmp { field, op, lit } => match tape_object_field(tape, entry, field) {
                Some(v) => tape_value_cmp(tape, v, *op, lit),
                None => false,
            },
            TapePred::And(xs) => xs.iter().all(|p| p.eval(tape, entry)),
            TapePred::Or(xs) => xs.iter().any(|p| p.eval(tape, entry)),
        }
    }
}

/// Compare a tape value at node `idx` against a literal.  Returns
/// `false` on type mismatch (e.g. Int vs Str).  Numeric comparisons
/// promote i64↔f64.
#[cfg(feature = "simd-json")]
pub fn tape_value_cmp(tape: &TapeData, idx: usize, op: TapeCmp, lit: &TapeLit) -> bool {
    crate::util::json_cmp_binop(
        tape_json_view(tape, idx),
        tape_cmp_to_binop(op),
        tape_lit_view(lit),
    )
}

#[cfg(feature = "simd-json")]
#[inline]
fn tape_lit_view<'a>(lit: &'a TapeLit<'a>) -> crate::util::JsonView<'a> {
    match lit {
        TapeLit::Int(n) => crate::util::JsonView::Int(*n),
        TapeLit::Float(f) => crate::util::JsonView::Float(*f),
        TapeLit::Str(s) => crate::util::JsonView::Str(s),
        TapeLit::Bool(b) => crate::util::JsonView::Bool(*b),
        TapeLit::Null => crate::util::JsonView::Null,
    }
}

#[cfg(feature = "simd-json")]
#[inline]
fn tape_cmp_to_binop(op: TapeCmp) -> crate::ast::BinOp {
    match op {
        TapeCmp::Eq => crate::ast::BinOp::Eq,
        TapeCmp::Neq => crate::ast::BinOp::Neq,
        TapeCmp::Lt => crate::ast::BinOp::Lt,
        TapeCmp::Lte => crate::ast::BinOp::Lte,
        TapeCmp::Gt => crate::ast::BinOp::Gt,
        TapeCmp::Gte => crate::ast::BinOp::Gte,
    }
}

/// `$.<arr>.filter(<field> <op> <lit>).count()` — count Array entries
/// where the projected field satisfies the comparison.  Returns `None`
/// only if `arr_idx` is not an Array.
#[cfg(feature = "simd-json")]
pub fn tape_array_filter_count(
    tape: &TapeData,
    arr_idx: usize,
    field: &str,
    op: TapeCmp,
    lit: &TapeLit,
) -> Option<usize> {
    let len = match tape.nodes[arr_idx] {
        TapeNode::Array { len, .. } => len as usize,
        _ => return None,
    };
    let mut count = 0usize;
    let mut j = arr_idx + 1;
    for _ in 0..len {
        let entry = j;
        if let TapeNode::Object { .. } = tape.nodes[entry] {
            if let Some(v) = tape_object_field(tape, entry, field) {
                if tape_value_cmp(tape, v, op, lit) {
                    count += 1;
                }
            }
        }
        j += tape.span(entry);
    }
    Some(count)
}

/// `$.<arr>.filter(<predicate>).count()` — predicate-only count
/// where the predicate may be any boolean tree of comparisons.
///
/// Single-leaf preds use the 2-slot ShapeIC (low overhead, hot path).
/// Compound preds use ShapeICN (Vec-backed, only worth it when the
/// per-entry name-search cost outweighs the Vec/lookup overhead —
/// i.e. with 2+ leaves AND wide Objects).
#[cfg(feature = "simd-json")]
pub fn tape_array_filter_pred_count(
    tape: &TapeData,
    arr_idx: usize,
    pred: &TapePred,
) -> Option<usize> {
    let len = match tape.nodes[arr_idx] {
        TapeNode::Array { len, .. } => len as usize,
        _ => return None,
    };
    let mut count = 0usize;

    // Single-leaf fast path — same hot-path code used pre-compound.
    if let TapePred::Cmp { field: pf, op, lit } = pred {
        let mut ic: Option<ShapeIC> = None;
        let mut j = arr_idx + 1;
        for _ in 0..len {
            let entry = j;
            if let TapeNode::Object { .. } = tape.nodes[entry] {
                let cached = ic.as_ref().filter(|c| shape_ic_check(tape, entry, c));
                let pred_v: Option<usize> = if let Some(c) = cached {
                    if c.rel_offs[0] == u32::MAX {
                        None
                    } else {
                        Some(entry + c.rel_offs[0] as usize)
                    }
                } else {
                    ic = shape_ic_build(tape, entry, [pf, ""]);
                    ic.as_ref().and_then(|c| {
                        if c.rel_offs[0] == u32::MAX {
                            None
                        } else {
                            Some(entry + c.rel_offs[0] as usize)
                        }
                    })
                };
                let pass = match pred_v {
                    Some(v) => tape_value_cmp(tape, v, *op, lit),
                    None => false,
                };
                if pass {
                    count += 1;
                }
            }
            j += tape.span(entry);
        }
        return Some(count);
    }

    // Compound predicate — name-based per-entry eval.  The per-entry
    // linear key search inside `pred.eval` is already cheap for 4-key
    // Objects (~4 string compares).  An N-field IC adds Vec allocation
    // + closure/position lookup overhead that proves a net loss.
    let mut j = arr_idx + 1;
    for _ in 0..len {
        let entry = j;
        if let TapeNode::Object { .. } = tape.nodes[entry] {
            if pred.eval(tape, entry) {
                count += 1;
            }
        }
        j += tape.span(entry);
    }
    Some(count)
}

/// Filter+map+fold variant accepting an arbitrary `TapePred`.
///
/// Single-leaf preds use the 2-slot ShapeIC; compound preds use
/// ShapeICN.  Specialising on pred shape avoids the Vec/closure
/// overhead of the N-field path on the common single-leaf case.
#[cfg(feature = "simd-json")]
pub fn tape_array_filter_pred_map_numeric_fold(
    tape: &TapeData,
    arr_idx: usize,
    pred: &TapePred,
    map_field: &str,
) -> Option<(i64, f64, usize, f64, f64, bool)> {
    let len = match tape.nodes[arr_idx] {
        TapeNode::Array { len, .. } => len as usize,
        _ => return None,
    };
    let mut acc = NumAcc {
        sum_i: 0,
        sum_f: 0.0,
        count: 0,
        min_f: f64::INFINITY,
        max_f: f64::NEG_INFINITY,
        is_float: false,
        mixed: false,
    };

    if let TapePred::Cmp { field: pf, op, lit } = pred {
        let mut ic: Option<ShapeIC> = None;
        let mut j = arr_idx + 1;
        for _ in 0..len {
            let entry = j;
            if !matches!(tape.nodes[entry], TapeNode::Object { .. }) {
                return None;
            }
            let cached = ic.as_ref().filter(|c| shape_ic_check(tape, entry, c));
            let pred_v: Option<usize> = if let Some(c) = cached {
                if c.rel_offs[0] == u32::MAX {
                    None
                } else {
                    Some(entry + c.rel_offs[0] as usize)
                }
            } else {
                ic = shape_ic_build(tape, entry, [pf, map_field]);
                ic.as_ref().and_then(|c| {
                    if c.rel_offs[0] == u32::MAX {
                        None
                    } else {
                        Some(entry + c.rel_offs[0] as usize)
                    }
                })
            };
            let pass = match pred_v {
                Some(v) => tape_value_cmp(tape, v, *op, lit),
                None => false,
            };
            if pass {
                let mv = ic.as_ref().and_then(|c| {
                    if c.rel_offs[1] == u32::MAX {
                        None
                    } else {
                        Some(entry + c.rel_offs[1] as usize)
                    }
                });
                if let Some(mv) = mv {
                    accumulate(tape.nodes[mv], &mut acc);
                    if acc.mixed {
                        return None;
                    }
                }
            }
            j += tape.span(entry);
        }
        return Some((
            acc.sum_i,
            acc.sum_f,
            acc.count,
            acc.min_f,
            acc.max_f,
            acc.is_float,
        ));
    }

    // Compound pred — name-based per-entry eval (see filter_pred_count).
    let mut j = arr_idx + 1;
    for _ in 0..len {
        let entry = j;
        if !matches!(tape.nodes[entry], TapeNode::Object { .. }) {
            return None;
        }
        if pred.eval(tape, entry) {
            if let Some(mv) = tape_object_field(tape, entry, map_field) {
                accumulate(tape.nodes[mv], &mut acc);
                if acc.mixed {
                    return None;
                }
            }
        }
        j += tape.span(entry);
    }
    Some((
        acc.sum_i,
        acc.sum_f,
        acc.count,
        acc.min_f,
        acc.max_f,
        acc.is_float,
    ))
}

/// Filter+map+collect variant accepting an arbitrary `TapePred`.
/// Same single-leaf vs compound split as the fold variant.
#[cfg(feature = "simd-json")]
pub fn tape_array_filter_pred_map_collect_numeric(
    tape: &TapeData,
    arr_idx: usize,
    pred: &TapePred,
    map_field: &str,
) -> Option<(Vec<i64>, Vec<f64>, bool)> {
    let len = match tape.nodes[arr_idx] {
        TapeNode::Array { len, .. } => len as usize,
        _ => return None,
    };
    let mut acc = NumCol {
        ints: Vec::new(),
        floats: Vec::new(),
        is_float: false,
        mixed: false,
    };

    if let TapePred::Cmp { field: pf, op, lit } = pred {
        let mut ic: Option<ShapeIC> = None;
        let mut j = arr_idx + 1;
        for _ in 0..len {
            let entry = j;
            if !matches!(tape.nodes[entry], TapeNode::Object { .. }) {
                return None;
            }
            let cached = ic.as_ref().filter(|c| shape_ic_check(tape, entry, c));
            let pred_v: Option<usize> = if let Some(c) = cached {
                if c.rel_offs[0] == u32::MAX {
                    None
                } else {
                    Some(entry + c.rel_offs[0] as usize)
                }
            } else {
                ic = shape_ic_build(tape, entry, [pf, map_field]);
                ic.as_ref().and_then(|c| {
                    if c.rel_offs[0] == u32::MAX {
                        None
                    } else {
                        Some(entry + c.rel_offs[0] as usize)
                    }
                })
            };
            let pass = match pred_v {
                Some(v) => tape_value_cmp(tape, v, *op, lit),
                None => false,
            };
            if pass {
                let mv = ic.as_ref().and_then(|c| {
                    if c.rel_offs[1] == u32::MAX {
                        None
                    } else {
                        Some(entry + c.rel_offs[1] as usize)
                    }
                });
                if let Some(mv) = mv {
                    collect_value(tape.nodes[mv], &mut acc);
                    if acc.mixed {
                        return None;
                    }
                }
            }
            j += tape.span(entry);
        }
        return Some((acc.ints, acc.floats, acc.is_float));
    }

    // Compound pred — name-based per-entry eval.
    let mut j = arr_idx + 1;
    for _ in 0..len {
        let entry = j;
        if !matches!(tape.nodes[entry], TapeNode::Object { .. }) {
            return None;
        }
        if pred.eval(tape, entry) {
            if let Some(mv) = tape_object_field(tape, entry, map_field) {
                collect_value(tape.nodes[mv], &mut acc);
                if acc.mixed {
                    return None;
                }
            }
        }
        j += tape.span(entry);
    }
    Some((acc.ints, acc.floats, acc.is_float))
}

/// Tape shape inline-cache.  When an Array's entries are all the same
/// shape (same `len`, same keys at same indices), we can skip per-entry
/// linear key search and read predicate/map field values via cached
/// **relative** offsets from the entry header.  Cache invalidates on
/// the first entry whose `len` differs or whose first-key bytes differ
/// from the cached signature.
///
/// Note: simd-json's tape stores each occurrence of a key as a fresh
/// string in `bytes_buf`, so byte-offset identity is NOT a valid
/// signature — we have to compare key *content* on shape check.  This
/// is one short memcmp per entry, still much cheaper than a full
/// linear key scan when the Object has 4+ keys.
#[cfg(feature = "simd-json")]
struct ShapeIC {
    obj_len: u32,
    first_key_len: u32, // cached length to short-circuit memcmp
    first_key_off: (u32, u32),
    /// Relative tape-node offset from the entry header to the field's
    /// VALUE node (so `entry + rel_off` is the value's index).  `u32::MAX`
    /// = field absent in the cached shape.
    rel_offs: [u32; 2],
}

#[cfg(feature = "simd-json")]
#[inline]
fn shape_ic_check(tape: &TapeData, entry: usize, ic: &ShapeIC) -> bool {
    if let TapeNode::Object { len, .. } = tape.nodes[entry] {
        if len != ic.obj_len {
            return false;
        }
        if let TapeNode::StringRef { start, end } = tape.nodes[entry + 1] {
            let this_len = end - start;
            if this_len != ic.first_key_len {
                return false;
            }
            // Compare bytes: both ranges live in the same bytes_buf.
            let a = &tape.bytes_buf[start as usize..end as usize];
            let b = &tape.bytes_buf[ic.first_key_off.0 as usize..ic.first_key_off.1 as usize];
            return a == b;
        }
    }
    false
}

/// Build a `ShapeIC` for `entry`, recording offsets for the two named
/// fields (or `u32::MAX` if absent).  Empty `""` slot means "field not
/// requested" — leaves rel_off at MAX.
#[cfg(feature = "simd-json")]
#[inline]
fn shape_ic_build(tape: &TapeData, entry: usize, fields: [&str; 2]) -> Option<ShapeIC> {
    let len = match tape.nodes[entry] {
        TapeNode::Object { len, .. } => len,
        _ => return None,
    };
    let (fk_start, fk_end) = match tape.nodes[entry + 1] {
        TapeNode::StringRef { start, end } => (start, end),
        _ => return None,
    };
    let mut rel_offs: [u32; 2] = [u32::MAX, u32::MAX];
    let want = [!fields[0].is_empty(), !fields[1].is_empty()];
    let mut filled = [!want[0], !want[1]];
    let mut j = entry + 1;
    for _ in 0..len {
        if filled[0] && filled[1] {
            j += 1;
            j += tape.span(j);
            continue;
        }
        let k = tape.str_at(j);
        j += 1;
        for (idx, name) in fields.iter().enumerate() {
            if want[idx] && !filled[idx] && k == *name {
                rel_offs[idx] = (j - entry) as u32;
                filled[idx] = true;
            }
        }
        j += tape.span(j);
    }
    Some(ShapeIC {
        obj_len: len,
        first_key_len: fk_end - fk_start,
        first_key_off: (fk_start, fk_end),
        rel_offs,
    })
}

/// `$.<arr>.filter(<pf> <op> <lit>).map(<mf>).<agg>()` — single-pass
/// filter + project + numeric fold.  Returns `None` if any entry is
/// not an Object or the projected `mf` value is non-numeric for an
/// entry that passed the predicate.
#[cfg(feature = "simd-json")]
pub fn tape_array_filter_map_numeric_fold(
    tape: &TapeData,
    arr_idx: usize,
    pred_field: &str,
    op: TapeCmp,
    lit: &TapeLit,
    map_field: &str,
) -> Option<(i64, f64, usize, f64, f64, bool)> {
    let len = match tape.nodes[arr_idx] {
        TapeNode::Array { len, .. } => len as usize,
        _ => return None,
    };
    let mut acc = NumAcc {
        sum_i: 0,
        sum_f: 0.0,
        count: 0,
        min_f: f64::INFINITY,
        max_f: f64::NEG_INFINITY,
        is_float: false,
        mixed: false,
    };
    let mut j = arr_idx + 1;
    for _ in 0..len {
        let entry = j;
        if !matches!(tape.nodes[entry], TapeNode::Object { .. }) {
            return None;
        }
        let pass = match tape_object_field(tape, entry, pred_field) {
            Some(v) => tape_value_cmp(tape, v, op, lit),
            None => false,
        };
        if pass {
            if let Some(mv) = tape_object_field(tape, entry, map_field) {
                accumulate(tape.nodes[mv], &mut acc);
                if acc.mixed {
                    return None;
                }
            }
        }
        j += tape.span(entry);
    }
    Some((
        acc.sum_i,
        acc.sum_f,
        acc.count,
        acc.min_f,
        acc.max_f,
        acc.is_float,
    ))
}

/// `$.<arr>.filter(<pf> <op> <lit>).map(<mf>)` — collect projected
/// numeric values from passing entries into IntVec / FloatVec.
#[cfg(feature = "simd-json")]
pub fn tape_array_filter_map_collect_numeric(
    tape: &TapeData,
    arr_idx: usize,
    pred_field: &str,
    op: TapeCmp,
    lit: &TapeLit,
    map_field: &str,
) -> Option<(Vec<i64>, Vec<f64>, bool)> {
    let len = match tape.nodes[arr_idx] {
        TapeNode::Array { len, .. } => len as usize,
        _ => return None,
    };
    let mut acc = NumCol {
        ints: Vec::new(),
        floats: Vec::new(),
        is_float: false,
        mixed: false,
    };
    let mut j = arr_idx + 1;
    for _ in 0..len {
        let entry = j;
        if !matches!(tape.nodes[entry], TapeNode::Object { .. }) {
            return None;
        }
        let pass = match tape_object_field(tape, entry, pred_field) {
            Some(v) => tape_value_cmp(tape, v, op, lit),
            None => false,
        };
        if pass {
            if let Some(mv) = tape_object_field(tape, entry, map_field) {
                collect_value(tape.nodes[mv], &mut acc);
                if acc.mixed {
                    return None;
                }
            }
        }
        j += tape.span(entry);
    }
    Some((acc.ints, acc.floats, acc.is_float))
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
    let mut acc = NumCol {
        ints: Vec::with_capacity(len),
        floats: Vec::new(),
        is_float: false,
        mixed: false,
    };
    let mut ic: Option<ShapeIC> = None;
    let mut j = arr_idx + 1;
    for _ in 0..len {
        let entry = j;
        if !matches!(tape.nodes[entry], TapeNode::Object { .. }) {
            return None;
        }
        let cached = ic.as_ref().filter(|c| shape_ic_check(tape, entry, c));
        let v: Option<usize> = if let Some(c) = cached {
            if c.rel_offs[0] == u32::MAX {
                None
            } else {
                Some(entry + c.rel_offs[0] as usize)
            }
        } else {
            ic = shape_ic_build(tape, entry, [field, ""]);
            ic.as_ref().and_then(|c| {
                if c.rel_offs[0] == u32::MAX {
                    None
                } else {
                    Some(entry + c.rel_offs[0] as usize)
                }
            })
        };
        if let Some(v) = v {
            collect_value(tape.nodes[v], &mut acc);
            if acc.mixed {
                return None;
            }
        }
        j += tape.span(entry);
    }
    Some((acc.ints, acc.floats, acc.is_float))
}
