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
}
