//! Zero-copy borrowed string view backed by a shared `Arc<str>`.
//!
//! `StrRef` holds an `Arc<str>` parent plus a `[start, end)` byte range.
//! Cloning bumps the Arc refcount and copies two `u32` offsets — no heap
//! allocation. Used by the simd-json tape path so that string values returned
//! from `Val::StrSlice` / `Val::StrSliceVec` never allocate.

use std::cell::{OnceCell, RefCell};
use std::collections::HashMap;
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

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
pub type TapeNode = simd_json::Node<'static>;

const ARRAY_CHILD_INDEX_MIN_LEN: usize = 32;
const OBJECT_FIELD_INDEX_MIN_LEN: usize = 8;

#[derive(Clone, Copy)]
struct ObjectFieldEntry {
    hash: u64,
    key_idx: usize,
    value_idx: usize,
}

/// Parsed simd-json tape together with the byte buffer and structural-index buffers
/// that must remain alive for the duration of the tape's use.
pub struct TapeData {
    /// The raw JSON bytes with simd-json's in-place mutations applied; string
    /// positions in `nodes` index into this buffer.
    pub bytes_buf: Vec<u8>,
    /// Structural-index scratch buffers owned by simd-json; must not be dropped
    /// while `nodes` is in use because the tape borrows into them.
    _buffers: simd_json::Buffers,
    /// The flat tape of parsed JSON nodes; string nodes borrow from `bytes_buf`.
    pub nodes: Vec<TapeNode>,
    /// Immutable direct-child tape starts for arrays large enough to benefit
    /// from positional and reverse access.
    array_child_index: HashMap<usize, Box<[usize]>>,
    /// Lazy direct-field key/value tape slots for objects large enough to
    /// benefit from repeated field lookup.
    object_field_index: OnceLock<HashMap<usize, Box<[ObjectFieldEntry]>>>,
    /// Counter of how many subtrees were materialised into `Val`; used in tests
    /// to verify lazy-materialisation assumptions.
    #[cfg(test)]
    materialized_subtrees: AtomicUsize,
}
impl TapeData {
    /// Parse a JSON byte vector into a `TapeData` wrapped in an `Arc`.
    /// The input buffer is consumed and stored alongside the tape so that
    /// string references remain valid.
    pub fn parse(mut bytes: Vec<u8>) -> Result<Arc<Self>, String> {
        Self::parse_inner(&mut bytes)
            .map_err(|e| e.to_string())
            .map(|(nodes, bytes_buf, buffers)| {
                let array_child_index = build_array_child_index(&nodes);
                Arc::new(Self {
                    bytes_buf,
                    _buffers: buffers,
                    nodes,
                    array_child_index,
                    object_field_index: OnceLock::new(),
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

    /// Return the tape index of child `idx` for an array whose first child is
    /// at `first` and whose direct child count is `len`.
    #[inline]
    pub(crate) fn array_child_start(&self, first: usize, len: usize, idx: usize) -> Option<usize> {
        if idx >= len {
            return None;
        }
        if let Some(children) = self.array_child_index.get(&first) {
            return children.get(idx).copied();
        }
        let mut cur = first;
        for _ in 0..idx {
            cur += self.span(cur);
        }
        Some(cur)
    }

    /// Return all direct array child tape indices for an array whose first
    /// child is at `first` and whose direct child count is `len`.
    pub(crate) fn array_child_starts(&self, first: usize, len: usize) -> Vec<usize> {
        if let Some(children) = self.array_child_index.get(&first) {
            return children.to_vec();
        }
        let mut children = Vec::with_capacity(len);
        let mut cur = first;
        for _ in 0..len {
            children.push(cur);
            cur += self.span(cur);
        }
        children
    }

    /// Borrow precomputed direct array child starts when the parsed tape
    /// already indexed this array.
    #[inline]
    pub(crate) fn array_child_indexed_starts(&self, first: usize) -> Option<&[usize]> {
        self.array_child_index
            .get(&first)
            .map(|children| &**children)
    }

    #[inline]
    pub(crate) fn object_field_value(&self, idx: usize, key: &str) -> Option<usize> {
        let TapeNode::Object { len, .. } = *self.nodes.get(idx)? else {
            return None;
        };
        if len >= OBJECT_FIELD_INDEX_MIN_LEN {
            if let Some(value_idx) =
                indexed_object_field_value(|i| self.str_at(i), self.object_field_index(), idx, key)
            {
                return Some(value_idx);
            }
        }
        let mut cur = idx + 1;
        for _ in 0..len {
            if self.str_at(cur) == key {
                return Some(cur + 1);
            }
            cur += 1;
            cur += self.span(cur);
        }
        None
    }

    #[cfg(test)]
    pub(crate) fn has_array_child_index(&self, first: usize) -> bool {
        self.array_child_index.contains_key(&first)
    }

    #[cfg(test)]
    pub(crate) fn has_object_field_index(&self, object_idx: usize) -> bool {
        self.object_field_index
            .get()
            .is_some_and(|index| index.contains_key(&object_idx))
    }

    #[inline]
    fn object_field_index(&self) -> &HashMap<usize, Box<[ObjectFieldEntry]>> {
        self.object_field_index
            .get_or_init(|| build_object_field_index(&self.nodes))
    }
}

fn build_array_child_index(nodes: &[TapeNode]) -> HashMap<usize, Box<[usize]>> {
    let mut index = HashMap::new();
    rebuild_array_child_index(nodes, &mut index);
    index
}

fn rebuild_array_child_index(nodes: &[TapeNode], index: &mut HashMap<usize, Box<[usize]>>) {
    index.clear();
    for (node_idx, node) in nodes.iter().enumerate() {
        let TapeNode::Array { len, .. } = *node else {
            continue;
        };
        if len < ARRAY_CHILD_INDEX_MIN_LEN {
            continue;
        }
        let first = node_idx + 1;
        let mut children = Vec::with_capacity(len);
        let mut cur = first;
        for _ in 0..len {
            children.push(cur);
            cur += tape_node_span(nodes, cur);
        }
        index.insert(first, children.into_boxed_slice());
    }
}

fn build_object_field_index(nodes: &[TapeNode]) -> HashMap<usize, Box<[ObjectFieldEntry]>> {
    let mut index = HashMap::new();
    rebuild_object_field_index(nodes, &mut index);
    index
}

fn rebuild_object_field_index(
    nodes: &[TapeNode],
    index: &mut HashMap<usize, Box<[ObjectFieldEntry]>>,
) {
    index.clear();
    for (node_idx, node) in nodes.iter().enumerate() {
        let TapeNode::Object { len, .. } = *node else {
            continue;
        };
        if len < OBJECT_FIELD_INDEX_MIN_LEN {
            continue;
        }
        index.insert(
            node_idx,
            build_object_field_entries(nodes, node_idx + 1, len),
        );
    }
}

fn build_object_field_entries(
    nodes: &[TapeNode],
    first: usize,
    len: usize,
) -> Box<[ObjectFieldEntry]> {
    let mut fields = Vec::with_capacity(len);
    let mut cur = first;
    for _ in 0..len {
        let key_idx = cur;
        let value_idx = cur + 1;
        fields.push(ObjectFieldEntry {
            hash: tape_string_hash(nodes, key_idx),
            key_idx,
            value_idx,
        });
        cur = value_idx + tape_node_span(nodes, value_idx);
    }
    fields.into_boxed_slice()
}

#[inline]
fn indexed_object_field_value<'a, F>(
    str_at: F,
    index: &HashMap<usize, Box<[ObjectFieldEntry]>>,
    idx: usize,
    key: &str,
) -> Option<usize>
where
    F: Fn(usize) -> &'a str,
{
    let fields = index.get(&idx)?;
    let needle_hash = str_hash(key);
    fields.iter().find_map(|field| {
        (field.hash == needle_hash && str_at(field.key_idx) == key).then_some(field.value_idx)
    })
}

#[inline]
fn tape_string_hash(nodes: &[TapeNode], idx: usize) -> u64 {
    match nodes[idx] {
        TapeNode::String(value) => str_hash(value),
        _ => 0,
    }
}

#[inline]
fn str_hash(value: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in value.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[inline]
fn tape_node_span(nodes: &[TapeNode], i: usize) -> usize {
    match nodes[i] {
        TapeNode::Object { count, .. } | TapeNode::Array { count, .. } => count + 1,
        _ => 1,
    }
}
pub(crate) struct TapeScratch {
    bytes_buf: Vec<u8>,
    buffers: simd_json::Buffers,
    pub(crate) nodes: Vec<TapeNode>,
    array_child_index: OnceCell<HashMap<usize, Box<[usize]>>>,
    object_field_index: RefCell<HashMap<usize, Box<[ObjectFieldEntry]>>>,
}
impl TapeScratch {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            bytes_buf: Vec::with_capacity(capacity),
            buffers: simd_json::Buffers::new(capacity),
            nodes: Vec::new(),
            array_child_index: OnceCell::new(),
            object_field_index: RefCell::new(HashMap::new()),
        }
    }

    pub(crate) fn parse_slice(&mut self, bytes: &[u8]) -> Result<(), String> {
        self.bytes_buf.clear();
        self.bytes_buf.extend_from_slice(bytes);
        let tape = simd_json::to_tape_with_buffers(&mut self.bytes_buf, &mut self.buffers)
            .map_err(|err| err.to_string())?;
        self.nodes =
            unsafe { std::mem::transmute::<Vec<simd_json::Node<'_>>, Vec<TapeNode>>(tape.0) };
        self.array_child_index = OnceCell::new();
        self.object_field_index.get_mut().clear();
        Ok(())
    }

    #[inline]
    pub(crate) fn str_at(&self, i: usize) -> &str {
        match self.nodes[i] {
            TapeNode::String(s) => s,
            _ => "",
        }
    }

    #[inline]
    pub(crate) fn span(&self, i: usize) -> usize {
        match self.nodes[i] {
            TapeNode::Object { count, .. } | TapeNode::Array { count, .. } => count + 1,
            _ => 1,
        }
    }

    #[inline]
    pub(crate) fn array_child_start(&self, first: usize, len: usize, idx: usize) -> Option<usize> {
        if idx >= len {
            return None;
        }
        if len >= ARRAY_CHILD_INDEX_MIN_LEN {
            if let Some(children) = self.array_child_index().get(&first) {
                return children.get(idx).copied();
            }
        }
        let mut cur = first;
        for _ in 0..idx {
            cur += self.span(cur);
        }
        Some(cur)
    }

    pub(crate) fn array_child_indices(&self, array_idx: usize) -> Option<Vec<usize>> {
        let TapeNode::Array { len, .. } = self.nodes[array_idx] else {
            return None;
        };
        let first = array_idx + 1;
        if len >= ARRAY_CHILD_INDEX_MIN_LEN {
            if let Some(children) = self.array_child_index().get(&first) {
                return Some(children.to_vec());
            }
        }
        let mut children = Vec::with_capacity(len);
        let mut cur = first;
        for _ in 0..len {
            children.push(cur);
            cur += self.span(cur);
        }
        Some(children)
    }

    #[inline]
    pub(crate) fn array_child_indexed_starts(&self, first: usize) -> Option<&[usize]> {
        let array_idx = first.checked_sub(1)?;
        let TapeNode::Array { len, .. } = self.nodes.get(array_idx)? else {
            return None;
        };
        if *len < ARRAY_CHILD_INDEX_MIN_LEN {
            return None;
        }
        self.array_child_index()
            .get(&first)
            .map(|children| &**children)
    }

    #[inline]
    pub(crate) fn object_field_value(&self, idx: usize, key: &str) -> Option<usize> {
        let TapeNode::Object { len, .. } = *self.nodes.get(idx)? else {
            return None;
        };
        if len >= OBJECT_FIELD_INDEX_MIN_LEN {
            if let Some(value_idx) = self.indexed_object_field_value(idx, len, key) {
                return Some(value_idx);
            }
        }
        let mut cur = idx + 1;
        for _ in 0..len {
            if self.str_at(cur) == key {
                return Some(cur + 1);
            }
            cur += 1;
            cur += self.span(cur);
        }
        None
    }

    #[cfg(test)]
    pub(crate) fn has_array_child_index(&self, first: usize) -> bool {
        self.array_child_index
            .get()
            .is_some_and(|index| index.contains_key(&first))
    }

    #[cfg(test)]
    pub(crate) fn has_object_field_index(&self, object_idx: usize) -> bool {
        self.object_field_index.borrow().contains_key(&object_idx)
    }

    #[inline]
    fn array_child_index(&self) -> &HashMap<usize, Box<[usize]>> {
        self.array_child_index
            .get_or_init(|| build_array_child_index(&self.nodes))
    }

    fn indexed_object_field_value(&self, idx: usize, len: usize, key: &str) -> Option<usize> {
        let mut index = self.object_field_index.borrow_mut();
        let fields = index
            .entry(idx)
            .or_insert_with(|| build_object_field_entries(&self.nodes, idx + 1, len));
        let needle_hash = str_hash(key);
        fields.iter().find_map(|field| {
            (field.hash == needle_hash && self.str_at(field.key_idx) == key)
                .then_some(field.value_idx)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{TapeData, TapeNode};

    #[test]
    fn array_child_navigation_returns_direct_child_tape_indices() {
        let tape = TapeData::parse(br#"[{"id":1},[2,3],"x"]"#.to_vec()).unwrap();
        let first = 1;
        let len = tape.root_len();

        let children = tape.array_child_starts(first, len);
        assert!(!tape.has_array_child_index(first));
        assert_eq!(children.len(), 3);
        assert_eq!(tape.array_child_start(first, len, 0), Some(children[0]));
        assert_eq!(tape.array_child_start(first, len, 1), Some(children[1]));
        assert_eq!(tape.array_child_start(first, len, 2), Some(children[2]));
        assert_eq!(tape.array_child_start(first, len, 3), None);

        assert!(matches!(
            tape.nodes[children[0]],
            crate::data::tape::TapeNode::Object { .. }
        ));
        assert!(matches!(
            tape.nodes[children[1]],
            crate::data::tape::TapeNode::Array { .. }
        ));
        assert!(matches!(
            tape.nodes[children[2]],
            crate::data::tape::TapeNode::String(_)
        ));
    }

    #[test]
    fn object_field_value_finds_value_slots() {
        let tape = TapeData::parse(br#"{"book":{"title":"Dune","score":901}}"#.to_vec()).unwrap();
        let book = tape.object_field_value(0, "book").expect("book");
        let title = tape.object_field_value(book, "title").expect("title");
        let score = tape.object_field_value(book, "score").expect("score");

        assert_eq!(tape.str_at(title), "Dune");
        assert!(matches!(tape.nodes[score], TapeNode::Static(_)));
        assert_eq!(tape.object_field_value(book, "missing"), None);
    }

    #[test]
    fn object_field_index_covers_large_objects_without_key_allocation() {
        let fields = (0..12)
            .map(|n| format!(r#""k{n}":{{"value":{n}}}"#))
            .collect::<Vec<_>>()
            .join(",");
        let tape = TapeData::parse(format!("{{{fields}}}").into_bytes()).unwrap();

        assert!(!tape.has_object_field_index(0));
        let k11 = tape.object_field_value(0, "k11").expect("k11");
        assert!(tape.has_object_field_index(0));
        let value = tape.object_field_value(k11, "value").expect("value");

        assert!(matches!(tape.nodes[k11], TapeNode::Object { .. }));
        assert!(matches!(tape.nodes[value], TapeNode::Static(_)));
        assert_eq!(tape.object_field_value(0, "missing"), None);
    }

    #[test]
    fn scratch_object_field_index_is_rebuilt_between_rows() {
        let fields = (0..12)
            .map(|n| format!(r#""k{n}":{n}"#))
            .collect::<Vec<_>>()
            .join(",");
        let mut scratch = super::TapeScratch::with_capacity(fields.len() + 2);
        scratch
            .parse_slice(format!("{{{fields}}}").as_bytes())
            .expect("parse large object");

        assert!(!scratch.has_object_field_index(0));
        assert!(scratch.object_field_value(0, "k11").is_some());
        assert!(scratch.has_object_field_index(0));

        scratch.parse_slice(br#"{"k11":1}"#).expect("parse small");

        assert!(!scratch.has_object_field_index(0));
        assert!(scratch.object_field_value(0, "k11").is_some());
        assert!(!scratch.has_object_field_index(0));
    }

    #[test]
    fn scratch_object_field_index_builds_only_requested_object() {
        let left = (0..12)
            .map(|n| format!(r#""l{n}":{n}"#))
            .collect::<Vec<_>>()
            .join(",");
        let right = (0..12)
            .map(|n| format!(r#""r{n}":{n}"#))
            .collect::<Vec<_>>()
            .join(",");
        let json = format!(r#"{{"left":{{{left}}},"right":{{{right}}}}}"#);
        let mut scratch = super::TapeScratch::with_capacity(json.len());
        scratch.parse_slice(json.as_bytes()).expect("parse");

        let left_idx = scratch.object_field_value(0, "left").expect("left");
        let right_idx = scratch.object_field_value(0, "right").expect("right");
        assert!(!scratch.has_object_field_index(left_idx));
        assert!(!scratch.has_object_field_index(right_idx));

        assert!(scratch.object_field_value(left_idx, "l11").is_some());

        assert!(scratch.has_object_field_index(left_idx));
        assert!(!scratch.has_object_field_index(right_idx));
    }

    #[test]
    fn array_child_index_covers_large_nested_arrays() {
        let nested = (0..40).map(|n| n.to_string()).collect::<Vec<_>>().join(",");
        let bytes = format!("[[{}],{{\"xs\":[3,4,5]}}]", nested).into_bytes();
        let tape = TapeData::parse(bytes).unwrap();
        let root_children = tape.array_child_starts(1, tape.root_len());
        assert!(!tape.has_array_child_index(1));

        let nested_first = root_children[0] + 1;
        assert!(tape.has_array_child_index(nested_first));
        assert_eq!(tape.array_child_start(nested_first, 40, 39), Some(41));
    }

    #[test]
    fn scratch_array_child_index_covers_large_arrays() {
        let values = (0..40).map(|n| n.to_string()).collect::<Vec<_>>().join(",");
        let mut scratch = super::TapeScratch::with_capacity(values.len() + 2);
        scratch
            .parse_slice(format!("[{}]", values).as_bytes())
            .expect("parse");

        assert!(!scratch.has_array_child_index(1));
        assert_eq!(
            scratch
                .array_child_indexed_starts(1)
                .expect("indexed")
                .len(),
            40
        );
        assert!(scratch.has_array_child_index(1));
        assert_eq!(scratch.array_child_start(1, 40, 39), Some(40));
        assert_eq!(scratch.array_child_indices(0).expect("indices").len(), 40);
    }

    #[test]
    fn scratch_array_child_index_is_rebuilt_between_rows() {
        let values = (0..40).map(|n| n.to_string()).collect::<Vec<_>>().join(",");
        let mut scratch = super::TapeScratch::with_capacity(values.len() + 2);
        scratch
            .parse_slice(format!("[{}]", values).as_bytes())
            .expect("parse large");
        assert!(!scratch.has_array_child_index(1));
        assert_eq!(scratch.array_child_start(1, 40, 39), Some(40));
        assert!(scratch.has_array_child_index(1));

        scratch.parse_slice(br#"[1,2,3]"#).expect("parse small");

        assert!(!scratch.has_array_child_index(1));
        assert_eq!(scratch.array_child_start(1, 3, 2), Some(3));
    }
}
