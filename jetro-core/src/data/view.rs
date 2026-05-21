//! Unified borrowed view over `Val` or a simd-json tape node.
//!
//! `ValueView` is the abstract interface that lets `physical_eval` and
//! `view_pipeline` navigate a document without materialising a `Val` tree.
//! Implementations exist for `Val` (in-memory) and for the simd-json tape path.
//! Paths that need a concrete `Val` call `materialize()`; structural navigation
//! stays zero-alloc.

use std::collections::HashSet;
use std::sync::Arc;

use crate::data::value::Val;
use crate::util::JsonView;

trait TapeLike {
    fn nodes(&self) -> &[crate::data::tape::TapeNode];
    fn str_at(&self, i: usize) -> &str;
    fn span(&self, i: usize) -> usize;
    fn materialize_at(&self, idx: &mut usize) -> Val;

    fn array_child_indices(&self, array_idx: usize) -> Option<Vec<usize>> {
        use crate::data::tape::TapeNode;

        let TapeNode::Array { len, .. } = self.nodes()[array_idx] else {
            return None;
        };
        let mut indices = Vec::with_capacity(len);
        let mut cur = array_idx + 1;
        for _ in 0..len {
            indices.push(cur);
            cur += self.span(cur);
        }
        Some(indices)
    }
}

impl TapeLike for crate::data::tape::TapeData {
    #[inline]
    fn nodes(&self) -> &[crate::data::tape::TapeNode] {
        &self.nodes
    }

    #[inline]
    fn str_at(&self, i: usize) -> &str {
        self.str_at(i)
    }

    #[inline]
    fn span(&self, i: usize) -> usize {
        self.span(i)
    }

    #[inline]
    fn materialize_at(&self, idx: &mut usize) -> Val {
        TapeView::materialize_at(self, idx)
    }
}

impl TapeLike for crate::data::tape::TapeScratch {
    #[inline]
    fn nodes(&self) -> &[crate::data::tape::TapeNode] {
        &self.nodes
    }

    #[inline]
    fn str_at(&self, i: usize) -> &str {
        self.str_at(i)
    }

    #[inline]
    fn span(&self, i: usize) -> usize {
        self.span(i)
    }

    #[inline]
    fn materialize_at(&self, idx: &mut usize) -> Val {
        TapeScratchView::materialize_at(self, idx)
    }
}

fn tape_field_idx<T: TapeLike>(tape: &T, idx: usize, key: &str) -> Option<Option<usize>> {
    use crate::data::tape::TapeNode;

    let TapeNode::Object { len, .. } = tape.nodes()[idx] else {
        return None;
    };

    let mut cur = idx + 1;
    for _ in 0..len {
        let current_key = tape.str_at(cur);
        cur += 1;
        if current_key == key {
            return Some(Some(cur));
        }
        cur += tape.span(cur);
    }
    Some(None)
}

#[inline]
fn tape_has_key<T: TapeLike>(tape: &T, idx: usize, key: &str) -> Option<bool> {
    tape_field_idx(tape, idx, key).map(|found| found.is_some())
}

#[inline]
fn key_slice_contains(keys: &[Arc<str>], needle: &str) -> bool {
    keys.iter().any(|key| key.as_ref() == needle)
}

fn tape_object_keys<T: TapeLike>(tape: &T, idx: usize) -> Option<Val> {
    use crate::data::tape::TapeNode;

    let TapeNode::Object { len, .. } = tape.nodes()[idx] else {
        return None;
    };

    let mut out = Vec::with_capacity(len);
    let mut cur = idx + 1;
    for _ in 0..len {
        out.push(Val::Str(Arc::from(tape.str_at(cur))));
        cur += 1;
        cur += tape.span(cur);
    }
    Some(Val::arr(out))
}

fn tape_object_values<T: TapeLike>(tape: &T, idx: usize) -> Option<Val> {
    use crate::data::tape::TapeNode;

    let TapeNode::Object { len, .. } = tape.nodes()[idx] else {
        return None;
    };

    let mut out = Vec::with_capacity(len);
    let mut cur = idx + 1;
    for _ in 0..len {
        cur += 1;
        let mut value_idx = cur;
        out.push(tape.materialize_at(&mut value_idx));
        cur += tape.span(cur);
    }
    Some(Val::arr(out))
}

fn tape_object_entries<T: TapeLike>(tape: &T, idx: usize) -> Option<Val> {
    use crate::data::tape::TapeNode;

    let TapeNode::Object { len, .. } = tape.nodes()[idx] else {
        return None;
    };

    let mut out = Vec::with_capacity(len);
    let mut cur = idx + 1;
    for _ in 0..len {
        let key = Arc::from(tape.str_at(cur));
        cur += 1;
        let mut value_idx = cur;
        out.push(Val::arr(vec![
            Val::Str(key),
            tape.materialize_at(&mut value_idx),
        ]));
        cur += tape.span(cur);
    }
    Some(Val::arr(out))
}

fn tape_pick_keys<T: TapeLike>(tape: &T, idx: usize, keys: &[Arc<str>]) -> Option<Val> {
    use crate::data::tape::TapeNode;

    let TapeNode::Object { len, .. } = tape.nodes()[idx] else {
        return None;
    };

    if keys.len() <= 4 {
        let mut out = indexmap::IndexMap::with_capacity(keys.len());
        let mut cur = idx + 1;
        for _ in 0..len {
            let current_key = tape.str_at(cur);
            cur += 1;
            if key_slice_contains(keys, current_key) {
                let mut value_idx = cur;
                out.insert(
                    crate::data::value::intern_key(current_key),
                    tape.materialize_at(&mut value_idx),
                );
                if out.len() == keys.len() {
                    break;
                }
            }
            cur += tape.span(cur);
        }
        return Some(Val::obj(out));
    }

    let mut out = indexmap::IndexMap::with_capacity(keys.len());
    let mut remaining: HashSet<&str> = keys.iter().map(|key| key.as_ref()).collect();
    let mut cur = idx + 1;
    for _ in 0..len {
        let current_key = tape.str_at(cur);
        cur += 1;
        if remaining.remove(current_key) {
            let mut value_idx = cur;
            out.insert(
                crate::data::value::intern_key(current_key),
                tape.materialize_at(&mut value_idx),
            );
            if remaining.is_empty() {
                break;
            }
        }
        cur += tape.span(cur);
    }
    Some(Val::obj(out))
}

fn tape_omit_keys<T: TapeLike>(tape: &T, idx: usize, keys: &[Arc<str>]) -> Option<Val> {
    use crate::data::tape::TapeNode;

    let TapeNode::Object { len, .. } = tape.nodes()[idx] else {
        return None;
    };

    if keys.len() <= 4 {
        let mut out = indexmap::IndexMap::with_capacity(len.saturating_sub(keys.len()));
        let mut cur = idx + 1;
        for _ in 0..len {
            let current_key = tape.str_at(cur);
            cur += 1;
            if !key_slice_contains(keys, current_key) {
                let mut value_idx = cur;
                out.insert(
                    crate::data::value::intern_key(current_key),
                    tape.materialize_at(&mut value_idx),
                );
            }
            cur += tape.span(cur);
        }
        return Some(Val::obj(out));
    }

    let omitted: HashSet<&str> = keys.iter().map(|key| key.as_ref()).collect();
    let mut out = indexmap::IndexMap::with_capacity(len.saturating_sub(omitted.len()));
    let mut cur = idx + 1;
    for _ in 0..len {
        let current_key = tape.str_at(cur);
        cur += 1;
        if !omitted.contains(current_key) {
            let mut value_idx = cur;
            out.insert(
                crate::data::value::intern_key(current_key),
                tape.materialize_at(&mut value_idx),
            );
        }
        cur += tape.span(cur);
    }
    Some(Val::obj(out))
}

/// Navigation interface shared by all document representations.
/// Implementations exist for `ValView` (in-memory `Val` tree) and `TapeView`
/// (borrowed simd-json tape nodes).
pub(crate) trait ValueView<'a>: Clone {
    /// Return a borrowed scalar view of the current node without allocating.
    fn scalar(&self) -> JsonView<'_>;
    /// Navigate into the named field of an object node, returning `Null` if absent.
    fn field(&self, key: &str) -> Self;
    /// Return whether the current object has `key`, or `None` if the current node is not an object.
    fn has_key(&self, key: &str) -> Option<bool>;
    /// Return the current object's keys without materialising child values.
    fn object_keys(&self) -> Option<Val>;
    /// Return the current object's values, materialising only the values.
    fn object_values(&self) -> Option<Val>;
    /// Return the current object's `[key, value]` entries.
    fn object_entries(&self) -> Option<Val>;
    /// Keep only `keys` from the current object, materialising selected values only.
    fn pick_keys(&self, keys: &[Arc<str>]) -> Option<Val>;
    /// Drop `keys` from the current object.
    fn omit_keys(&self, keys: &[Arc<str>]) -> Option<Val>;
    /// Navigate to the element at `idx` (negative indices count from the end),
    /// returning `Null` if out of bounds.
    fn index(&self, idx: i64) -> Self;
    /// Return an iterator over the elements of an array node, or `None` if the
    /// current node is not an array.
    fn array_iter(&self) -> Option<Box<dyn Iterator<Item = Self> + 'a>>;
    /// Return an iterator over the elements of an array node from the end.
    fn array_iter_rev(&self) -> Option<Box<dyn Iterator<Item = Self> + 'a>>;
    /// Fully materialise the current node as a `Val`, allocating as needed.
    fn materialize(&self) -> Val;
}

/// Convert a `JsonView` scalar to an owned `Val`, returning `None` for the
/// `ArrayLen` and `ObjectLen` pseudo-variants that carry no value payload.
#[inline]
pub(crate) fn scalar_view_to_owned_val(view: JsonView<'_>) -> Option<Val> {
    match view {
        JsonView::Null => Some(Val::Null),
        JsonView::Bool(value) => Some(Val::Bool(value)),
        JsonView::Int(value) => Some(Val::Int(value)),
        JsonView::UInt(value) => Some(if value <= i64::MAX as u64 {
            Val::Int(value as i64)
        } else {
            Val::Float(value as f64)
        }),
        JsonView::Float(value) => Some(Val::Float(value)),
        JsonView::Str(value) => Some(Val::Str(Arc::from(value))),
        JsonView::ArrayLen(_) | JsonView::ObjectLen(_) => None,
    }
}

/// `ValueView` implementation for in-memory `Val` trees.
/// Borrows the source `Val` when possible and falls back to an owned clone
/// only for dynamically computed results such as index out-of-range fallbacks.
#[derive(Clone)]
pub(crate) enum ValView<'a> {
    /// A direct borrow of a `Val` node from the original tree — zero copy.
    Borrowed(&'a Val),
    /// A transiently computed `Val` that has no parent in the original tree.
    Owned(Val),
}

impl<'a> ValView<'a> {
    /// Construct a `ValView` that borrows `value` from the caller.
    #[inline]
    pub(crate) fn new(value: &'a Val) -> Self {
        Self::Borrowed(value)
    }

    /// Return a reference to the underlying `Val` regardless of whether it is
    /// borrowed or owned.
    #[inline]
    fn value(&self) -> &Val {
        match self {
            Self::Borrowed(value) => value,
            Self::Owned(value) => value,
        }
    }
}

impl<'a> ValueView<'a> for ValView<'a> {
    #[inline]
    fn scalar(&self) -> JsonView<'_> {
        JsonView::from_val(self.value())
    }

    #[inline]
    fn field(&self, key: &str) -> Self {
        match self {
            Self::Borrowed(Val::Obj(map)) => map
                .get(key)
                .map(Self::Borrowed)
                .unwrap_or_else(|| Self::Owned(Val::Null)),
            Self::Borrowed(Val::ObjSmall(pairs)) => pairs
                .iter()
                .find_map(|(k, v)| (k.as_ref() == key).then_some(Self::Borrowed(v)))
                .unwrap_or_else(|| Self::Owned(Val::Null)),
            Self::Borrowed(_) => Self::Owned(Val::Null),
            Self::Owned(value) => Self::Owned(value.get_field(key)),
        }
    }

    #[inline]
    fn has_key(&self, key: &str) -> Option<bool> {
        match self.value() {
            Val::Obj(map) => Some(map.contains_key(key)),
            Val::ObjSmall(pairs) => Some(pairs.iter().any(|(k, _)| k.as_ref() == key)),
            _ => None,
        }
    }

    #[inline]
    fn object_keys(&self) -> Option<Val> {
        match self.value() {
            Val::Obj(map) => Some(Val::arr(
                map.keys().cloned().map(Val::Str).collect::<Vec<_>>(),
            )),
            Val::ObjSmall(pairs) => Some(Val::arr(
                pairs
                    .iter()
                    .map(|(key, _)| Val::Str(Arc::clone(key)))
                    .collect::<Vec<_>>(),
            )),
            _ => None,
        }
    }

    #[inline]
    fn object_values(&self) -> Option<Val> {
        match self.value() {
            Val::Obj(map) => Some(Val::arr(map.values().cloned().collect::<Vec<_>>())),
            Val::ObjSmall(pairs) => Some(Val::arr(
                pairs
                    .iter()
                    .map(|(_, value)| value.clone())
                    .collect::<Vec<_>>(),
            )),
            _ => None,
        }
    }

    #[inline]
    fn object_entries(&self) -> Option<Val> {
        match self.value() {
            Val::Obj(map) => Some(Val::arr(
                map.iter()
                    .map(|(key, value)| Val::arr(vec![Val::Str(Arc::clone(key)), value.clone()]))
                    .collect::<Vec<_>>(),
            )),
            Val::ObjSmall(pairs) => Some(Val::arr(
                pairs
                    .iter()
                    .map(|(key, value)| Val::arr(vec![Val::Str(Arc::clone(key)), value.clone()]))
                    .collect::<Vec<_>>(),
            )),
            _ => None,
        }
    }

    #[inline]
    fn pick_keys(&self, keys: &[Arc<str>]) -> Option<Val> {
        match self.value() {
            Val::Obj(map) => {
                let mut out = indexmap::IndexMap::with_capacity(keys.len());
                for key in keys {
                    if let Some(value) = map.get(key.as_ref()) {
                        out.insert(Arc::clone(key), value.clone());
                    }
                }
                Some(Val::obj(out))
            }
            Val::ObjSmall(pairs) => {
                let mut out = indexmap::IndexMap::with_capacity(keys.len());
                for key in keys {
                    if let Some((_, value)) = pairs.iter().find(|(k, _)| k.as_ref() == key.as_ref())
                    {
                        out.insert(Arc::clone(key), value.clone());
                    }
                }
                Some(Val::obj(out))
            }
            _ => None,
        }
    }

    #[inline]
    fn omit_keys(&self, keys: &[Arc<str>]) -> Option<Val> {
        if keys.len() <= 4 {
            return match self.value() {
                Val::Obj(map) => Some(Val::obj(
                    map.iter()
                        .filter(|(key, _)| !key_slice_contains(keys, key.as_ref()))
                        .map(|(key, value)| (Arc::clone(key), value.clone()))
                        .collect(),
                )),
                Val::ObjSmall(pairs) => Some(Val::obj(
                    pairs
                        .iter()
                        .filter(|(key, _)| !key_slice_contains(keys, key.as_ref()))
                        .map(|(key, value)| (Arc::clone(key), value.clone()))
                        .collect(),
                )),
                _ => None,
            };
        }

        let omitted: HashSet<&str> = keys.iter().map(|key| key.as_ref()).collect();
        match self.value() {
            Val::Obj(map) => Some(Val::obj(
                map.iter()
                    .filter(|(key, _)| !omitted.contains(key.as_ref()))
                    .map(|(key, value)| (Arc::clone(key), value.clone()))
                    .collect(),
            )),
            Val::ObjSmall(pairs) => Some(Val::obj(
                pairs
                    .iter()
                    .filter(|(key, _)| !omitted.contains(key.as_ref()))
                    .map(|(key, value)| (Arc::clone(key), value.clone()))
                    .collect(),
            )),
            _ => None,
        }
    }

    #[inline]
    fn index(&self, idx: i64) -> Self {
        match self {
            Self::Borrowed(Val::Arr(items)) => normalize_index(items.len(), idx)
                .and_then(|idx| items.get(idx))
                .map(Self::Borrowed)
                .unwrap_or_else(|| Self::Owned(Val::Null)),
            Self::Borrowed(Val::IntVec(items)) => normalize_index(items.len(), idx)
                .and_then(|idx| items.get(idx).copied())
                .map(Val::Int)
                .map(Self::Owned)
                .unwrap_or_else(|| Self::Owned(Val::Null)),
            Self::Borrowed(Val::FloatVec(items)) => normalize_index(items.len(), idx)
                .and_then(|idx| items.get(idx).copied())
                .map(Val::Float)
                .map(Self::Owned)
                .unwrap_or_else(|| Self::Owned(Val::Null)),
            Self::Borrowed(Val::StrVec(items)) => normalize_index(items.len(), idx)
                .and_then(|idx| items.get(idx).cloned())
                .map(Val::Str)
                .map(Self::Owned)
                .unwrap_or_else(|| Self::Owned(Val::Null)),
            Self::Borrowed(Val::StrSliceVec(items)) => normalize_index(items.len(), idx)
                .and_then(|idx| items.get(idx).cloned())
                .map(Val::StrSlice)
                .map(Self::Owned)
                .unwrap_or_else(|| Self::Owned(Val::Null)),
            Self::Borrowed(_) => Self::Owned(Val::Null),
            Self::Owned(value) => Self::Owned(value.get_index(idx)),
        }
    }

    #[inline]
    fn array_iter(&self) -> Option<Box<dyn Iterator<Item = Self> + 'a>> {
        match self {
            Self::Borrowed(Val::Arr(items)) => Some(Box::new(items.iter().map(Self::Borrowed))),
            Self::Borrowed(Val::IntVec(items)) => Some(Box::new(
                items.iter().copied().map(Val::Int).map(Self::Owned),
            )),
            Self::Borrowed(Val::FloatVec(items)) => Some(Box::new(
                items.iter().copied().map(Val::Float).map(Self::Owned),
            )),
            Self::Borrowed(Val::StrVec(items)) => Some(Box::new(
                items.iter().cloned().map(Val::Str).map(Self::Owned),
            )),
            Self::Borrowed(Val::StrSliceVec(items)) => Some(Box::new(
                items.iter().cloned().map(Val::StrSlice).map(Self::Owned),
            )),
            Self::Borrowed(_) => None,
            Self::Owned(value) => match value {
                Val::Arr(items) => Some(Box::new(
                    Arc::clone(items)
                        .as_ref()
                        .clone()
                        .into_iter()
                        .map(Self::Owned),
                )),
                Val::IntVec(items) => Some(Box::new(
                    Arc::clone(items)
                        .as_ref()
                        .clone()
                        .into_iter()
                        .map(Val::Int)
                        .map(Self::Owned),
                )),
                Val::FloatVec(items) => Some(Box::new(
                    Arc::clone(items)
                        .as_ref()
                        .clone()
                        .into_iter()
                        .map(Val::Float)
                        .map(Self::Owned),
                )),
                Val::StrVec(items) => Some(Box::new(
                    Arc::clone(items)
                        .as_ref()
                        .clone()
                        .into_iter()
                        .map(Val::Str)
                        .map(Self::Owned),
                )),
                Val::StrSliceVec(items) => Some(Box::new(
                    Arc::clone(items)
                        .as_ref()
                        .clone()
                        .into_iter()
                        .map(Val::StrSlice)
                        .map(Self::Owned),
                )),
                _ => None,
            },
        }
    }

    #[inline]
    fn array_iter_rev(&self) -> Option<Box<dyn Iterator<Item = Self> + 'a>> {
        match self {
            Self::Borrowed(Val::Arr(items)) => {
                Some(Box::new(items.iter().rev().map(Self::Borrowed)))
            }
            Self::Borrowed(Val::IntVec(items)) => Some(Box::new(
                items.iter().rev().copied().map(Val::Int).map(Self::Owned),
            )),
            Self::Borrowed(Val::FloatVec(items)) => Some(Box::new(
                items.iter().rev().copied().map(Val::Float).map(Self::Owned),
            )),
            Self::Borrowed(Val::StrVec(items)) => Some(Box::new(
                items.iter().rev().cloned().map(Val::Str).map(Self::Owned),
            )),
            Self::Borrowed(Val::StrSliceVec(items)) => Some(Box::new(
                items
                    .iter()
                    .rev()
                    .cloned()
                    .map(Val::StrSlice)
                    .map(Self::Owned),
            )),
            Self::Borrowed(_) => None,
            Self::Owned(value) => match value {
                Val::Arr(items) => Some(Box::new(
                    Arc::clone(items)
                        .as_ref()
                        .clone()
                        .into_iter()
                        .rev()
                        .map(Self::Owned),
                )),
                Val::IntVec(items) => Some(Box::new(
                    Arc::clone(items)
                        .as_ref()
                        .clone()
                        .into_iter()
                        .rev()
                        .map(Val::Int)
                        .map(Self::Owned),
                )),
                Val::FloatVec(items) => Some(Box::new(
                    Arc::clone(items)
                        .as_ref()
                        .clone()
                        .into_iter()
                        .rev()
                        .map(Val::Float)
                        .map(Self::Owned),
                )),
                Val::StrVec(items) => Some(Box::new(
                    Arc::clone(items)
                        .as_ref()
                        .clone()
                        .into_iter()
                        .rev()
                        .map(Val::Str)
                        .map(Self::Owned),
                )),
                Val::StrSliceVec(items) => Some(Box::new(
                    Arc::clone(items)
                        .as_ref()
                        .clone()
                        .into_iter()
                        .rev()
                        .map(Val::StrSlice)
                        .map(Self::Owned),
                )),
                _ => None,
            },
        }
    }

    #[inline]
    fn materialize(&self) -> Val {
        self.value().clone()
    }
}

/// `ValueView` implementation that navigates a simd-json tape without
/// materialising `Val` nodes until `materialize()` is explicitly called.
#[derive(Clone, Copy)]
pub(crate) enum TapeView<'a> {
    /// A live reference to a tape node at `idx` within the borrowed `TapeData`.
    Node {
        /// The simd-json tape buffer this view points into.
        tape: &'a crate::data::tape::TapeData,
        /// Index of the current node within `tape.nodes`.
        idx: usize,
    },
    /// Sentinel for a field or index that was not found; behaves like `Val::Null`.
    Missing,
}
impl<'a> TapeView<'a> {
    /// Return a `TapeView` pointing at the root node of `tape`, or `Missing`
    /// if the tape is empty (invalid JSON).
    #[inline]
    pub(crate) fn root(tape: &'a crate::data::tape::TapeData) -> Self {
        if tape.nodes.is_empty() {
            Self::Missing
        } else {
            Self::Node { tape, idx: 0 }
        }
    }

    /// Recursively materialise the tape subtree starting at `*idx`, advancing
    /// `idx` past all consumed nodes and returning the resulting `Val`.
    #[inline]
    fn materialize_at(tape: &'a crate::data::tape::TapeData, idx: &mut usize) -> Val {
        use crate::data::tape::TapeNode;
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
                let mut out = Vec::with_capacity(len);
                for _ in 0..len {
                    out.push(Self::materialize_at(tape, idx));
                }
                Val::arr(out)
            }
            TapeNode::Object { len, .. } => {
                let mut out = indexmap::IndexMap::with_capacity(len);
                for _ in 0..len {
                    let key = tape.str_at(*idx);
                    *idx += 1;
                    let value = Self::materialize_at(tape, idx);
                    out.insert(crate::data::value::intern_key(key), value);
                }
                Val::Obj(std::sync::Arc::new(out))
            }
        }
    }
}
impl<'a> ValueView<'a> for TapeView<'a> {
    #[inline]
    fn scalar(&self) -> JsonView<'_> {
        use crate::data::tape::TapeNode;
        use simd_json::StaticNode as SN;

        let Self::Node { tape, idx } = self else {
            return JsonView::Null;
        };
        match tape.nodes[*idx] {
            TapeNode::Static(SN::Null) => JsonView::Null,
            TapeNode::Static(SN::Bool(b)) => JsonView::Bool(b),
            TapeNode::Static(SN::I64(n)) => JsonView::Int(n),
            TapeNode::Static(SN::U64(n)) => JsonView::UInt(n),
            TapeNode::Static(SN::F64(f)) => JsonView::Float(f),
            TapeNode::String(_) => JsonView::Str(tape.str_at(*idx)),
            TapeNode::Array { len, .. } => JsonView::ArrayLen(len),
            TapeNode::Object { len, .. } => JsonView::ObjectLen(len),
        }
    }

    #[inline]
    fn field(&self, key: &str) -> Self {
        let Self::Node { tape, idx } = self else {
            return Self::Missing;
        };
        if let Some(Some(field_idx)) = tape_field_idx(*tape, *idx, key) {
            return Self::Node {
                tape,
                idx: field_idx,
            };
        }
        Self::Missing
    }

    #[inline]
    fn has_key(&self, key: &str) -> Option<bool> {
        let Self::Node { tape, idx } = self else {
            return None;
        };
        tape_has_key(*tape, *idx, key)
    }

    #[inline]
    fn object_keys(&self) -> Option<Val> {
        let Self::Node { tape, idx } = self else {
            return None;
        };
        tape_object_keys(*tape, *idx)
    }

    #[inline]
    fn object_values(&self) -> Option<Val> {
        let Self::Node { tape, idx } = self else {
            return None;
        };
        tape_object_values(*tape, *idx)
    }

    #[inline]
    fn object_entries(&self) -> Option<Val> {
        let Self::Node { tape, idx } = self else {
            return None;
        };
        tape_object_entries(*tape, *idx)
    }

    #[inline]
    fn pick_keys(&self, keys: &[Arc<str>]) -> Option<Val> {
        let Self::Node { tape, idx } = self else {
            return None;
        };
        tape_pick_keys(*tape, *idx, keys)
    }

    #[inline]
    fn omit_keys(&self, keys: &[Arc<str>]) -> Option<Val> {
        let Self::Node { tape, idx } = self else {
            return None;
        };
        tape_omit_keys(*tape, *idx, keys)
    }

    #[inline]
    fn index(&self, idx: i64) -> Self {
        use crate::data::tape::TapeNode;

        let Self::Node { tape, idx: node } = self else {
            return Self::Missing;
        };
        let TapeNode::Array { len, .. } = tape.nodes[*node] else {
            return Self::Missing;
        };
        let Some(target) = normalize_index(len, idx) else {
            return Self::Missing;
        };

        let mut cur = *node + 1;
        for _ in 0..target {
            cur += tape.span(cur);
        }
        Self::Node { tape, idx: cur }
    }

    #[inline]
    fn array_iter(&self) -> Option<Box<dyn Iterator<Item = Self> + 'a>> {
        use crate::data::tape::TapeNode;

        let Self::Node { tape, idx } = self else {
            return None;
        };
        let TapeNode::Array { len, .. } = tape.nodes[*idx] else {
            return None;
        };

        Some(Box::new(TapeArrayIter {
            tape,
            remaining: len,
            cur: *idx + 1,
        }))
    }

    #[inline]
    fn array_iter_rev(&self) -> Option<Box<dyn Iterator<Item = Self> + 'a>> {
        let Self::Node { tape, idx } = self else {
            return None;
        };
        let tape: &'a crate::data::tape::TapeData = *tape;
        let indices = tape.array_child_indices(*idx)?;
        Some(Box::new(indices.into_iter().rev().map(move |child| {
            Self::Node { tape, idx: child }
        })))
    }

    #[inline]
    fn materialize(&self) -> Val {
        match self {
            Self::Node { tape, idx } => {
                #[cfg(test)]
                tape.observe_materialized_subtree();
                let mut idx = *idx;
                Self::materialize_at(tape, &mut idx)
            }
            Self::Missing => Val::Null,
        }
    }
}

/// Iterator that yields `TapeView` nodes for each element of a tape array,
/// advancing through the tape by the span of each node.
struct TapeArrayIter<'a> {
    /// The tape buffer being iterated.
    tape: &'a crate::data::tape::TapeData,
    /// Number of elements still to be yielded.
    remaining: usize,
    /// Current tape position (index of the next element node).
    cur: usize,
}
impl<'a> Iterator for TapeArrayIter<'a> {
    type Item = TapeView<'a>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let idx = self.cur;
        self.remaining -= 1;
        self.cur += self.tape.span(self.cur);
        Some(TapeView::Node {
            tape: self.tape,
            idx,
        })
    }
}
#[derive(Clone, Copy)]
pub(crate) enum TapeScratchView<'a> {
    Node {
        tape: &'a crate::data::tape::TapeScratch,
        idx: usize,
    },
    Missing,
}
impl<'a> TapeScratchView<'a> {
    #[inline]
    fn materialize_at(tape: &'a crate::data::tape::TapeScratch, idx: &mut usize) -> Val {
        use crate::data::tape::TapeNode;
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
            TapeNode::String(_) => {
                Val::StrSlice(crate::data::tape::StrRef::from(tape.str_at(*idx - 1)))
            }
            TapeNode::Array { len, .. } => {
                let mut out = Vec::with_capacity(len);
                for _ in 0..len {
                    out.push(Self::materialize_at(tape, idx));
                }
                Val::arr(out)
            }
            TapeNode::Object { len, .. } => {
                let mut out = indexmap::IndexMap::with_capacity(len);
                for _ in 0..len {
                    let key = tape.str_at(*idx);
                    *idx += 1;
                    let value = Self::materialize_at(tape, idx);
                    out.insert(crate::data::value::intern_key(key), value);
                }
                Val::Obj(std::sync::Arc::new(out))
            }
        }
    }
}
impl<'a> ValueView<'a> for TapeScratchView<'a> {
    #[inline]
    fn scalar(&self) -> JsonView<'_> {
        use crate::data::tape::TapeNode;
        use simd_json::StaticNode as SN;

        let Self::Node { tape, idx } = self else {
            return JsonView::Null;
        };
        match tape.nodes[*idx] {
            TapeNode::Static(SN::Null) => JsonView::Null,
            TapeNode::Static(SN::Bool(b)) => JsonView::Bool(b),
            TapeNode::Static(SN::I64(n)) => JsonView::Int(n),
            TapeNode::Static(SN::U64(n)) => JsonView::UInt(n),
            TapeNode::Static(SN::F64(f)) => JsonView::Float(f),
            TapeNode::String(_) => JsonView::Str(tape.str_at(*idx)),
            TapeNode::Array { len, .. } => JsonView::ArrayLen(len),
            TapeNode::Object { len, .. } => JsonView::ObjectLen(len),
        }
    }

    #[inline]
    fn field(&self, key: &str) -> Self {
        let Self::Node { tape, idx } = self else {
            return Self::Missing;
        };
        if let Some(Some(field_idx)) = tape_field_idx(*tape, *idx, key) {
            return Self::Node {
                tape,
                idx: field_idx,
            };
        }
        Self::Missing
    }

    #[inline]
    fn has_key(&self, key: &str) -> Option<bool> {
        let Self::Node { tape, idx } = self else {
            return None;
        };
        tape_has_key(*tape, *idx, key)
    }

    #[inline]
    fn object_keys(&self) -> Option<Val> {
        let Self::Node { tape, idx } = self else {
            return None;
        };
        tape_object_keys(*tape, *idx)
    }

    #[inline]
    fn object_values(&self) -> Option<Val> {
        let Self::Node { tape, idx } = self else {
            return None;
        };
        tape_object_values(*tape, *idx)
    }

    #[inline]
    fn object_entries(&self) -> Option<Val> {
        let Self::Node { tape, idx } = self else {
            return None;
        };
        tape_object_entries(*tape, *idx)
    }

    #[inline]
    fn pick_keys(&self, keys: &[Arc<str>]) -> Option<Val> {
        let Self::Node { tape, idx } = self else {
            return None;
        };
        tape_pick_keys(*tape, *idx, keys)
    }

    #[inline]
    fn omit_keys(&self, keys: &[Arc<str>]) -> Option<Val> {
        let Self::Node { tape, idx } = self else {
            return None;
        };
        tape_omit_keys(*tape, *idx, keys)
    }

    #[inline]
    fn index(&self, idx: i64) -> Self {
        use crate::data::tape::TapeNode;

        let Self::Node { tape, idx: node } = self else {
            return Self::Missing;
        };
        let TapeNode::Array { len, .. } = tape.nodes[*node] else {
            return Self::Missing;
        };
        let Some(target) = normalize_index(len, idx) else {
            return Self::Missing;
        };
        let mut cur = *node + 1;
        for _ in 0..target {
            cur += tape.span(cur);
        }
        Self::Node { tape, idx: cur }
    }

    #[inline]
    fn array_iter(&self) -> Option<Box<dyn Iterator<Item = Self> + 'a>> {
        use crate::data::tape::TapeNode;

        let Self::Node { tape, idx } = self else {
            return None;
        };
        let TapeNode::Array { len, .. } = tape.nodes[*idx] else {
            return None;
        };
        Some(Box::new(TapeScratchArrayIter {
            tape,
            remaining: len,
            cur: *idx + 1,
        }))
    }

    #[inline]
    fn array_iter_rev(&self) -> Option<Box<dyn Iterator<Item = Self> + 'a>> {
        let Self::Node { tape, idx } = self else {
            return None;
        };
        let tape: &'a crate::data::tape::TapeScratch = *tape;
        let indices = tape.array_child_indices(*idx)?;
        Some(Box::new(indices.into_iter().rev().map(move |child| {
            Self::Node { tape, idx: child }
        })))
    }

    #[inline]
    fn materialize(&self) -> Val {
        match self {
            Self::Node { tape, idx } => {
                let mut idx = *idx;
                Self::materialize_at(tape, &mut idx)
            }
            Self::Missing => Val::Null,
        }
    }
}
struct TapeScratchArrayIter<'a> {
    tape: &'a crate::data::tape::TapeScratch,
    remaining: usize,
    cur: usize,
}
impl<'a> Iterator for TapeScratchArrayIter<'a> {
    type Item = TapeScratchView<'a>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let idx = self.cur;
        self.remaining -= 1;
        self.cur += self.tape.span(self.cur);
        Some(TapeScratchView::Node {
            tape: self.tape,
            idx,
        })
    }
}

#[inline]
fn normalize_index(len: usize, idx: i64) -> Option<usize> {
    let idx = if idx < 0 {
        len.checked_sub(idx.unsigned_abs() as usize)?
    } else {
        idx as usize
    };
    (idx < len).then_some(idx)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::json;

    use super::{scalar_view_to_owned_val, ValView, ValueView};
    use crate::util::{json_cmp_binop, JsonView};
    use crate::{data::value::Val, parse::ast::BinOp};

    #[test]
    fn val_view_reads_nested_fields_without_materializing_parent() {
        let value = Val::from(&json!({
            "book": {"title": "Dune", "score": 901},
            "unused": {"payload": [1, 2, 3]}
        }));
        let root = ValView::new(&value);

        let title = root.field("book").field("title");
        let score = root.field("book").field("score");

        assert!(matches!(title.scalar(), JsonView::Str("Dune")));
        assert!(json_cmp_binop(
            score.scalar(),
            BinOp::Gt,
            JsonView::Int(900)
        ));
    }

    #[test]
    fn val_view_checks_object_keys_without_reading_values() {
        let value = Val::from(&json!({
            "book": {"title": "Dune", "score": 901}
        }));
        let book = ValView::new(&value).field("book");

        assert_eq!(book.has_key("title"), Some(true));
        assert_eq!(book.has_key("missing"), Some(false));
        assert_eq!(book.field("title").has_key("x"), None);
    }

    #[test]
    fn val_view_object_helpers_match_object_semantics() {
        let value = Val::from(&json!({
            "book": {"title": "Dune", "score": 901, "debug": true}
        }));
        let book = ValView::new(&value).field("book");

        assert_eq!(
            serde_json::Value::from(book.object_keys().unwrap()),
            json!(["debug", "score", "title"])
        );
        assert_eq!(
            serde_json::Value::from(book.pick_keys(&[Arc::from("score")]).unwrap()),
            json!({"score": 901})
        );
        assert_eq!(
            serde_json::Value::from(book.omit_keys(&[Arc::from("debug")]).unwrap()),
            json!({"score": 901, "title": "Dune"})
        );
    }

    #[test]
    fn val_view_indexes_columnar_arrays() {
        let nums = Val::IntVec(Arc::new(vec![10, 20, 30]));
        let view = ValView::new(&nums);

        assert!(matches!(view.index(1).scalar(), JsonView::Int(20)));
        assert!(matches!(view.index(-1).scalar(), JsonView::Int(30)));
        assert!(matches!(view.index(99).scalar(), JsonView::Null));
    }

    #[test]
    fn array_reverse_iter_matches_value_order_for_val_and_tape() {
        use super::TapeView;

        let value = Val::from(&json!({"items": [1, {"id": 2}, [3]]}));
        let val_items = ValView::new(&value).field("items");
        let val_rows = val_items
            .array_iter_rev()
            .unwrap()
            .map(|item| serde_json::Value::from(item.materialize()))
            .collect::<Vec<_>>();

        let tape = crate::data::tape::TapeData::parse(br#"{"items":[1,{"id":2},[3]]}"#.to_vec())
            .unwrap();
        let tape_rows = TapeView::root(&tape)
            .field("items")
            .array_iter_rev()
            .unwrap()
            .map(|item| serde_json::Value::from(item.materialize()))
            .collect::<Vec<_>>();

        assert_eq!(val_rows, json!([[3], {"id": 2}, 1]).as_array().unwrap().clone());
        assert_eq!(tape_rows, val_rows);
    }

    #[test]
    fn val_view_materializes_current_view_only() {
        let value = Val::from(&json!({"items": [{"id": 1}, {"id": 2}]}));
        let item = ValView::new(&value).field("items").index(1);

        assert_eq!(
            serde_json::Value::from(item.materialize()),
            json!({"id": 2})
        );
    }

    #[test]
    fn scalar_view_to_owned_val_converts_only_scalars() {
        assert_eq!(scalar_view_to_owned_val(JsonView::Null), Some(Val::Null));
        assert_eq!(
            scalar_view_to_owned_val(JsonView::Str("ada")),
            Some(Val::Str(Arc::from("ada")))
        );
        assert!(scalar_view_to_owned_val(JsonView::ArrayLen(3)).is_none());
        assert!(scalar_view_to_owned_val(JsonView::ObjectLen(2)).is_none());
    }
    #[test]
    fn tape_view_matches_val_view_for_field_index_scalar_reads() {
        use super::TapeView;

        let bytes =
            br#"{"books":[{"title":"low","score":1},{"title":"Dune","score":901}]}"#.to_vec();
        let tape = crate::data::tape::TapeData::parse(bytes).unwrap();
        let val = Val::from_tape_data(&tape);

        let tape_score_view = TapeView::root(&tape).field("books").index(1).field("score");
        let tape_score = tape_score_view.scalar();
        let val_score_view = ValView::new(&val).field("books").index(1).field("score");
        let val_score = val_score_view.scalar();

        assert!(matches!(
            (tape_score, val_score),
            (JsonView::Int(901), JsonView::Int(901))
        ));
        let tape_book = TapeView::root(&tape).field("books").index(1);
        let val_book = ValView::new(&val).field("books").index(1);

        assert_eq!(tape_book.has_key("title"), Some(true));
        assert_eq!(tape_book.has_key("missing"), Some(false));
        assert_eq!(tape_book.has_key("title"), val_book.has_key("title"));

        assert_eq!(
            tape_book.object_keys().map(serde_json::Value::from),
            val_book.object_keys().map(serde_json::Value::from)
        );
        assert_eq!(
            tape_book.object_values().map(serde_json::Value::from),
            val_book.object_values().map(serde_json::Value::from)
        );
        assert_eq!(
            tape_book.object_entries().map(serde_json::Value::from),
            val_book.object_entries().map(serde_json::Value::from)
        );
        assert_eq!(
            tape_book
                .pick_keys(&[Arc::from("score")])
                .map(serde_json::Value::from),
            val_book
                .pick_keys(&[Arc::from("score")])
                .map(serde_json::Value::from)
        );
        assert_eq!(
            tape_book
                .omit_keys(&[Arc::from("title")])
                .map(serde_json::Value::from),
            val_book
                .omit_keys(&[Arc::from("title")])
                .map(serde_json::Value::from)
        );

        let many_keys = [
            Arc::from("missing_a"),
            Arc::from("score"),
            Arc::from("missing_b"),
            Arc::from("title"),
            Arc::from("missing_c"),
        ];
        assert_eq!(
            tape_book.pick_keys(&many_keys).map(serde_json::Value::from),
            val_book.pick_keys(&many_keys).map(serde_json::Value::from)
        );
        assert_eq!(
            tape_book.omit_keys(&many_keys).map(serde_json::Value::from),
            val_book.omit_keys(&many_keys).map(serde_json::Value::from)
        );
    }

    #[test]
    fn tape_scratch_view_object_helpers_match_val_view() {
        use super::{TapeScratchView, TapeView};

        let mut scratch = crate::data::tape::TapeScratch::with_capacity(128);
        scratch
            .parse_slice(br#"{"book":{"title":"Dune","score":901,"debug":true}}"#)
            .unwrap();
        let tape = crate::data::tape::TapeData::parse(
            br#"{"book":{"title":"Dune","score":901,"debug":true}}"#.to_vec(),
        )
        .unwrap();
        let tape_book = TapeScratchView::Node {
            tape: &scratch,
            idx: 0,
        }
        .field("book");
        let stable_tape_book = TapeView::root(&tape).field("book");

        assert_eq!(
            tape_book.object_keys().map(serde_json::Value::from),
            stable_tape_book.object_keys().map(serde_json::Value::from)
        );
        assert_eq!(
            tape_book.object_values().map(serde_json::Value::from),
            stable_tape_book.object_values().map(serde_json::Value::from)
        );
        assert_eq!(
            tape_book.object_entries().map(serde_json::Value::from),
            stable_tape_book.object_entries().map(serde_json::Value::from)
        );
        assert_eq!(
            tape_book
                .pick_keys(&[Arc::from("score")])
                .map(serde_json::Value::from),
            stable_tape_book
                .pick_keys(&[Arc::from("score")])
                .map(serde_json::Value::from)
        );
        assert_eq!(
            tape_book
                .omit_keys(&[Arc::from("debug")])
                .map(serde_json::Value::from),
            stable_tape_book
                .omit_keys(&[Arc::from("debug")])
                .map(serde_json::Value::from)
        );
    }

    #[test]
    fn tape_view_materializes_current_subtree_only() {
        use super::TapeView;

        let tape = crate::data::tape::TapeData::parse(
            br#"{"items":[{"id":1},{"id":2}],"unused":[0]}"#.to_vec(),
        )
        .unwrap();
        let item = TapeView::root(&tape).field("items").index(1);

        assert_eq!(
            serde_json::Value::from(item.materialize()),
            json!({"id": 2})
        );
    }
}
