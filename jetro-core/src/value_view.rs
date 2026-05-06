//! Unified borrowed view over `Val` or a simd-json tape node.
//!
//! `ValueView` is the abstract interface that lets `physical_eval` and
//! `view_pipeline` navigate a document without materialising a `Val` tree.
//! Implementations exist for `Val` (in-memory) and for the tape path
//! (simd-json, behind the `simd-json` feature). Paths that need a concrete
//! `Val` call `materialize()`; structural navigation stays zero-alloc.

use std::sync::Arc;

use crate::util::JsonView;
use crate::value::Val;

/// Navigation interface shared by all document representations.
/// Implementations exist for `ValView` (in-memory `Val` tree) and,
/// behind the `simd-json` feature, for `TapeView` (borrowed tape nodes).
pub(crate) trait ValueView<'a>: Clone {
    /// Return a borrowed scalar view of the current node without allocating.
    fn scalar(&self) -> JsonView<'_>;
    /// Navigate into the named field of an object node, returning `Null` if absent.
    fn field(&self, key: &str) -> Self;
    /// Navigate to the element at `idx` (negative indices count from the end),
    /// returning `Null` if out of bounds.
    fn index(&self, idx: i64) -> Self;
    /// Return an iterator over the elements of an array node, or `None` if the
    /// current node is not an array.
    fn array_iter(&self) -> Option<Box<dyn Iterator<Item = Self> + 'a>>;
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
    fn materialize(&self) -> Val {
        self.value().clone()
    }
}

/// `ValueView` implementation that navigates a simd-json tape without
/// materialising `Val` nodes until `materialize()` is explicitly called.
#[cfg(feature = "simd-json")]
#[derive(Clone, Copy)]
pub(crate) enum TapeView<'a> {
    /// A live reference to a tape node at `idx` within the borrowed `TapeData`.
    Node {
        /// The simd-json tape buffer this view points into.
        tape: &'a crate::tape::TapeData,
        /// Index of the current node within `tape.nodes`.
        idx: usize,
    },
    /// Sentinel for a field or index that was not found; behaves like `Val::Null`.
    Missing,
}

#[cfg(feature = "simd-json")]
impl<'a> TapeView<'a> {
    /// Return a `TapeView` pointing at the root node of `tape`, or `Missing`
    /// if the tape is empty (invalid JSON).
    #[inline]
    pub(crate) fn root(tape: &'a crate::tape::TapeData) -> Self {
        if tape.nodes.is_empty() {
            Self::Missing
        } else {
            Self::Node { tape, idx: 0 }
        }
    }

    /// Recursively materialise the tape subtree starting at `*idx`, advancing
    /// `idx` past all consumed nodes and returning the resulting `Val`.
    #[inline]
    fn materialize_at(tape: &'a crate::tape::TapeData, idx: &mut usize) -> Val {
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
                    out.insert(crate::value::intern_key(key), value);
                }
                Val::Obj(std::sync::Arc::new(out))
            }
        }
    }
}

#[cfg(feature = "simd-json")]
impl<'a> ValueView<'a> for TapeView<'a> {
    #[inline]
    fn scalar(&self) -> JsonView<'_> {
        use crate::tape::TapeNode;
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
        use crate::tape::TapeNode;

        let Self::Node { tape, idx } = self else {
            return Self::Missing;
        };
        let TapeNode::Object { len, .. } = tape.nodes[*idx] else {
            return Self::Missing;
        };

        let mut cur = *idx + 1;
        for _ in 0..len {
            let current_key = tape.str_at(cur);
            cur += 1;
            if current_key == key {
                return Self::Node { tape, idx: cur };
            }
            cur += tape.span(cur);
        }
        Self::Missing
    }

    #[inline]
    fn index(&self, idx: i64) -> Self {
        use crate::tape::TapeNode;

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
        use crate::tape::TapeNode;

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
#[cfg(feature = "simd-json")]
struct TapeArrayIter<'a> {
    /// The tape buffer being iterated.
    tape: &'a crate::tape::TapeData,
    /// Number of elements still to be yielded.
    remaining: usize,
    /// Current tape position (index of the next element node).
    cur: usize,
}

#[cfg(feature = "simd-json")]
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
    use crate::{ast::BinOp, value::Val};

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
    fn val_view_indexes_columnar_arrays() {
        let nums = Val::IntVec(Arc::new(vec![10, 20, 30]));
        let view = ValView::new(&nums);

        assert!(matches!(view.index(1).scalar(), JsonView::Int(20)));
        assert!(matches!(view.index(-1).scalar(), JsonView::Int(30)));
        assert!(matches!(view.index(99).scalar(), JsonView::Null));
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

    #[cfg(feature = "simd-json")]
    #[test]
    fn tape_view_matches_val_view_for_field_index_scalar_reads() {
        use super::TapeView;

        let bytes =
            br#"{"books":[{"title":"low","score":1},{"title":"Dune","score":901}]}"#.to_vec();
        let tape = crate::tape::TapeData::parse(bytes).unwrap();
        let val = Val::from_tape_data(&tape);

        let tape_score_view = TapeView::root(&tape).field("books").index(1).field("score");
        let tape_score = tape_score_view.scalar();
        let val_score_view = ValView::new(&val).field("books").index(1).field("score");
        let val_score = val_score_view.scalar();

        assert!(matches!(
            (tape_score, val_score),
            (JsonView::Int(901), JsonView::Int(901))
        ));
    }

    #[cfg(feature = "simd-json")]
    #[test]
    fn tape_view_materializes_current_subtree_only() {
        use super::TapeView;

        let tape = crate::tape::TapeData::parse(
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
