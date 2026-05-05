//! Zero-copy borrowed string view backed by a shared `Arc<str>`.
//!
//! `StrRef` holds an `Arc<str>` parent plus a `[start, end)` byte range.
//! Cloning bumps the Arc refcount and copies two `u32` offsets — no heap
//! allocation. Used by the simd-json tape path so that string values returned
//! from `Val::StrSlice` / `Val::StrSliceVec` never allocate.

#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Borrowed string slice into a parent `Arc<str>`. See module doc.
#[derive(Clone, Debug)]
pub struct StrRef {
    /// The owning Arc whose refcount keeps the backing memory alive.
    parent: Arc<str>,
    /// Byte offset of the first character of the slice within `parent`.
    start: u32,
    /// Byte offset one past the last character; `start == end` means empty.
    end: u32,
}

impl StrRef {
    /// Wrap an entire `Arc<str>` as a `StrRef` with no sub-slicing.
    #[inline]
    pub fn from_arc(parent: Arc<str>) -> Self {
        let end = parent.len() as u32;
        Self {
            parent,
            start: 0,
            end,
        }
    }

    /// Create a sub-slice `[start, end)` of `parent`; both bounds must be valid UTF-8 boundaries.
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

    /// Create a `StrRef` from a raw byte buffer by transmuting the `Arc<[u8]>` to `Arc<str>`.
    /// The caller must guarantee that `bytes[start..end]` is valid UTF-8.
    #[inline]
    pub fn slice_bytes(parent: Arc<[u8]>, start: usize, end: usize) -> Self {
        debug_assert!(start <= end);
        debug_assert!(end <= parent.len());
        // SAFETY: caller guarantees the byte range is valid UTF-8; the Arc layout
        // is identical between `[u8]` and `str` so the transmute is safe.
        let parent_str: Arc<str> = unsafe { Arc::from_raw(Arc::into_raw(parent) as *const str) };
        Self {
            parent: parent_str,
            start: start as u32,
            end: end as u32,
        }
    }

    /// Return the borrowed `&str` slice without any allocation.
    #[inline]
    pub fn as_str(&self) -> &str {
        &self.parent[self.start as usize..self.end as usize]
    }

    /// Return the byte length of the slice.
    #[inline]
    pub fn len(&self) -> usize {
        (self.end - self.start) as usize
    }

    /// Return `true` when the slice covers zero bytes.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.end == self.start
    }

    /// Return an `Arc<str>` for this slice: re-uses the parent Arc when the
    /// slice covers the entire parent, avoiding an allocation in the common case.
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
    /// Deref to `&str` so `StrRef` can be used anywhere a `&str` is expected.
    #[inline]
    fn deref(&self) -> &str {
        self.as_str()
    }
}

impl std::fmt::Display for StrRef {
    /// Format the slice contents without allocating a new `String`.
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl PartialEq for StrRef {
    /// Equality is content-based, not pointer-based; two slices with the same bytes compare equal.
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_str() == other.as_str()
    }
}
impl Eq for StrRef {}

impl PartialEq<str> for StrRef {
    /// Compare this slice against a plain `str` reference.
    #[inline]
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}
impl PartialEq<&str> for StrRef {
    /// Compare this slice against a `&&str` (common in generic code).
    #[inline]
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}
impl PartialEq<Arc<str>> for StrRef {
    /// Compare this slice against an `Arc<str>` without cloning either side.
    #[inline]
    fn eq(&self, other: &Arc<str>) -> bool {
        self.as_str() == other.as_ref()
    }
}

impl std::hash::Hash for StrRef {
    /// Hash the string content so `StrRef` and `&str` with the same bytes produce the same hash.
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_str().hash(state)
    }
}

impl From<Arc<str>> for StrRef {
    /// Wrap an entire `Arc<str>` as a full-range `StrRef`.
    #[inline]
    fn from(a: Arc<str>) -> Self {
        Self::from_arc(a)
    }
}
impl From<&str> for StrRef {
    /// Allocate a new `Arc<str>` from a borrowed slice and wrap it.
    #[inline]
    fn from(s: &str) -> Self {
        Self::from_arc(Arc::<str>::from(s))
    }
}
impl From<String> for StrRef {
    /// Convert an owned `String` into an `Arc<str>` and wrap it.
    #[inline]
    fn from(s: String) -> Self {
        Self::from_arc(Arc::<str>::from(s))
    }
}


/// Re-export of the simd-json tape node type, fixed to a `'static` lifetime because
/// the backing buffer is owned by `TapeData` and kept alive via `Arc`.
#[cfg(feature = "simd-json")]
pub type TapeNode = simd_json::Node<'static>;

/// Parsed simd-json tape together with the byte buffer and structural-index buffers
/// that must remain alive for the duration of the tape's use.
#[cfg(feature = "simd-json")]
pub struct TapeData {
    /// The raw JSON bytes with simd-json's in-place mutations applied; string
    /// positions in `nodes` index into this buffer.
    pub bytes_buf: Vec<u8>,
    /// Structural-index scratch buffers owned by simd-json; must not be dropped
    /// while `nodes` is in use because the tape borrows into them.
    _buffers: simd_json::Buffers,
    /// The flat tape of parsed JSON nodes; string nodes borrow from `bytes_buf`.
    pub nodes: Vec<TapeNode>,
    /// Counter of how many subtrees were materialised into `Val`; used in tests
    /// to verify lazy-materialisation assumptions.
    #[cfg(test)]
    materialized_subtrees: AtomicUsize,
}

#[cfg(feature = "simd-json")]
impl TapeData {
    /// Parse a JSON byte vector into a `TapeData` wrapped in an `Arc`.
    /// The input buffer is consumed and stored alongside the tape so that
    /// string references remain valid.
    pub fn parse(mut bytes: Vec<u8>) -> Result<Arc<Self>, String> {
        Self::parse_inner(&mut bytes)
            .map_err(|e| e.to_string())
            .map(|(nodes, bytes_buf, buffers)| {
                Arc::new(Self {
                    bytes_buf,
                    _buffers: buffers,
                    nodes,
                    #[cfg(test)]
                    materialized_subtrees: AtomicUsize::new(0),
                })
            })
    }

    /// Internal helper: run simd-json on the mutable byte slice, collect the tape
    /// nodes, and take ownership of both the (now mutated) buffer and the Buffers.
    fn parse_inner(
        bytes: &mut Vec<u8>,
    ) -> Result<(Vec<TapeNode>, Vec<u8>, simd_json::Buffers), simd_json::Error> {
        let mut buffers = simd_json::Buffers::new(bytes.len());
        let tape = simd_json::to_tape_with_buffers(bytes, &mut buffers)?;
        // SAFETY: we extend the lifetime of the tape nodes to `'static` because
        // both `bytes_buf` and `buffers` are stored in the same `TapeData` struct
        // and will not be freed while `nodes` lives.
        let nodes =
            unsafe { std::mem::transmute::<Vec<simd_json::Node<'_>>, Vec<TapeNode>>(tape.0) };
        let bytes_buf = std::mem::take(bytes);
        Ok((nodes, bytes_buf, buffers))
    }

    /// Increment the materialised-subtree counter; called when a tape subtree is
    /// converted to a `Val` tree.  Only compiled in test builds.
    #[cfg(test)]
    #[inline]
    pub(crate) fn observe_materialized_subtree(&self) {
        self.materialized_subtrees.fetch_add(1, Ordering::Relaxed);
    }

    /// Reset the materialised-subtree counter to zero for a fresh test assertion.
    #[cfg(test)]
    #[inline]
    pub(crate) fn reset_materialized_subtrees(&self) {
        self.materialized_subtrees.store(0, Ordering::Relaxed);
    }

    /// Read the current materialised-subtree count for test assertions.
    #[cfg(test)]
    #[inline]
    pub(crate) fn materialized_subtrees(&self) -> usize {
        self.materialized_subtrees.load(Ordering::Relaxed)
    }

    /// Return the UTF-8 string in `bytes_buf` for the byte range `[start, end)`.
    /// The caller must guarantee the range is valid UTF-8.
    #[inline]
    pub fn str_at_range(&self, start: usize, end: usize) -> &str {
        unsafe { std::str::from_utf8_unchecked(&self.bytes_buf[start..end]) }
    }

    /// Return the string value of a `TapeNode::String` at tape index `i`.
    /// Panics (in debug mode) if the node at `i` is not a string.
    #[inline]
    pub fn str_at(&self, i: usize) -> &str {
        match self.nodes[i] {
            TapeNode::String(s) => s,
            _ => unreachable!("str_at: node {} is not a string", i),
        }
    }

    /// Return a zero-copy `StrRef` for the string at tape index `i`.
    #[inline]
    pub fn str_ref_at(&self, i: usize) -> StrRef {
        match self.nodes[i] {
            TapeNode::String(s) => StrRef::from(s),
            _ => unreachable!("str_ref_at: node {} is not a string", i),
        }
    }

    /// Return the number of direct children in the root object or array, or `0` for
    /// other root node types (scalars, empty documents).
    pub fn root_len(&self) -> usize {
        match self.nodes.first() {
            Some(TapeNode::Object { len, .. }) | Some(TapeNode::Array { len, .. }) => *len,
            _ => 0,
        }
    }

    /// Return the number of tape slots occupied by the node at index `i`, including
    /// itself: `1` for scalars and strings, `count + 1` for objects and arrays.
    #[inline]
    pub fn span(&self, i: usize) -> usize {
        match self.nodes[i] {
            TapeNode::Object { count, .. } | TapeNode::Array { count, .. } => count + 1,
            _ => 1,
        }
    }
}
