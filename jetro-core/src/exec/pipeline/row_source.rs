//! Row-level source abstraction for pipeline execution.
//!
//! `Rows` unifies borrowed slice, shared Arc, and ObjVec columnar sources
//! so the execution loop can iterate without knowing which backing store
//! was used. `RowSource` drives one iteration step returning a borrowed or
//! owned Val per row.

use std::borrow::Cow;
use std::sync::Arc;

use crate::data::value::{ObjVecData, Val};

use super::{walk_field_chain, Source, SourceAccessMode};

/// Unified row storage that avoids copying when the source owns or borrows an array slice, while also supporting owned `Vec<Val>`.
pub(super) enum Rows<'a> {
    /// A borrowed slice from a pre-existing array; zero-copy iteration.
    Borrowed(&'a [Val]),
    /// A reference-counted array shared with the source `Val::Arr`.
    Shared(Arc<Vec<Val>>),
    /// Fully owned row buffer, e.g. produced by a barrier stage.
    Owned(Vec<Val>),
}

/// Iterator over `Rows`, producing cloned `Val` items for each backing variant.
pub(super) enum RowsIter<'a> {
    /// Iterates over a borrowed slice, cloning each element on demand.
    Borrowed(std::slice::Iter<'a, Val>),
    /// Iterates over a shared `Arc<Vec<Val>>` by index, cloning each element.
    Shared { rows: Arc<Vec<Val>>, index: usize },
    /// Draining iterator over a fully owned `Vec<Val>`.
    Owned(std::vec::IntoIter<Val>),
}

/// Abstraction over a pipeline row source, handling `ObjVec` columnar data, array-like `Rows`, and single scalars.
pub(super) enum ValRowSource<'a> {
    /// A columnar `ObjVec`; rows are reconstructed as objects on demand.
    ObjVec(Arc<ObjVecData>),
    /// An array-like source backed by a `Rows` variant.
    Rows(Rows<'a>),
    /// A single non-array value treated as a one-element source.
    Single(Val),
}

/// Iterator over a `ValRowSource`, materialising `ObjVec` rows on demand.
pub(super) enum ValRowsIter<'a> {
    /// Delegates to the underlying `RowsIter`.
    Rows(RowsIter<'a>),
    /// Reconstructs each `ObjVec` row as a `Val::Obj` by index.
    ObjVec { data: Arc<ObjVecData>, index: usize },
    /// Single-element iterator for scalar sources.
    Single(std::option::IntoIter<Val>),
}

/// Row source backed directly by a `simd-json` tape, enabling zero-copy streaming without building a `Val` tree.
pub(super) enum TapeRowSource<'a> {
    /// The source tape node is an array; iteration yields each element by span.
    Array {
        tape: &'a crate::data::tape::TapeData,
        // Index of the first array element in the tape.
        first: usize,
        // Number of elements in the array.
        len: usize,
    },
    /// The source tape node is a scalar or object; treated as a single row.
    Single(crate::data::view::TapeView<'a>),
    /// The requested field path did not resolve to any tape node.
    Missing,
}

/// Iterator over tape nodes that yields each array element as a `TapeView` without materialisation.
pub(super) enum TapeRowsIter<'a> {
    /// Advances through array elements by consuming tape spans.
    Array {
        tape: &'a crate::data::tape::TapeData,
        // Elements remaining to be yielded.
        remaining: usize,
        // Current tape index.
        cur: usize,
    },
    /// Advances through precomputed array child tape indices from the end.
    ReverseArray {
        tape: &'a crate::data::tape::TapeData,
        children: std::vec::IntoIter<usize>,
    },
    /// Single-element iterator for a non-array tape node.
    Single(std::option::IntoIter<crate::data::view::TapeView<'a>>),
    /// Iterator for a `Missing` source; always returns `None`.
    Empty,
}

/// Wrapper around `TapeRowsIter` that materialises each `TapeView` into a `Val` on demand.
pub(super) struct TapeMaterializedRowsIter<'a>(TapeRowsIter<'a>);

impl Iterator for ValRowsIter<'_> {
    type Item = Val;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Rows(iter) => iter.next(),
            Self::ObjVec { data, index } => {
                if *index >= data.nrows() {
                    return None;
                }
                let row = objvec_row(data, *index);
                *index += 1;
                Some(row)
            }
            Self::Single(iter) => iter.next(),
        }
    }
}
impl<'a> Iterator for TapeRowsIter<'a> {
    type Item = crate::data::view::TapeView<'a>;

    // Advances the current tape index by the node's span before returning the view.
    fn next(&mut self) -> Option<Self::Item> {
        use crate::data::view::TapeView;

        match self {
            Self::Array {
                tape,
                remaining,
                cur,
            } => {
                if *remaining == 0 {
                    return None;
                }
                let idx = *cur;
                *remaining -= 1;
                *cur += tape.span(idx);
                Some(TapeView::Node { tape, idx })
            }
            Self::ReverseArray { tape, children } => {
                children.next().map(|idx| TapeView::Node { tape, idx })
            }
            Self::Single(iter) => iter.next(),
            Self::Empty => None,
        }
    }
}
impl Iterator for TapeMaterializedRowsIter<'_> {
    type Item = Val;

    fn next(&mut self) -> Option<Self::Item> {
        use crate::data::view::ValueView;

        self.0.next().map(|view| view.materialize())
    }
}

impl Iterator for RowsIter<'_> {
    type Item = Val;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Borrowed(iter) => iter.next().cloned(),
            Self::Shared { rows, index } => {
                let item = rows.get(*index)?.clone();
                *index += 1;
                Some(item)
            }
            Self::Owned(iter) => iter.next(),
        }
    }
}

impl<'a> Rows<'a> {
    /// Returns a slice view of the rows regardless of which backing variant is active.
    pub(super) fn as_slice(&self) -> &[Val] {
        match self {
            Self::Borrowed(rows) => rows,
            Self::Shared(rows) => rows.as_ref(),
            Self::Owned(rows) => rows.as_slice(),
        }
    }

    /// Converts the `Rows` into a `RowsIter` that yields cloned `Val` items.
    pub(super) fn iter_cloned(self) -> RowsIter<'a> {
        match self {
            Self::Borrowed(rows) => RowsIter::Borrowed(rows.iter()),
            Self::Shared(rows) => RowsIter::Shared { rows, index: 0 },
            Self::Owned(rows) => RowsIter::Owned(rows.into_iter()),
        }
    }

    /// Consumes `Rows` and returns an owned `Vec<Val>`, cloning only when the backing store is `Borrowed` or `Shared`.
    pub(super) fn into_vec(self) -> Vec<Val> {
        match self {
            Self::Borrowed(rows) => rows.to_vec(),
            Self::Shared(rows) => rows.as_ref().clone(),
            Self::Owned(rows) => rows,
        }
    }
}

impl<'a> ValRowSource<'a> {
    /// Constructs a `ValRowSource` from `recv`, selecting the most efficient backing for the value's type.
    pub(super) fn from_receiver(recv: &'a Val) -> Self {
        match recv {
            Val::ObjVec(data) => Self::ObjVec(Arc::clone(data)),
            _ => array_like_rows(recv)
                .map(Self::Rows)
                .unwrap_or_else(|| Self::Single(recv.clone())),
        }
    }

    /// Converts this source into a `ValRowsIter` that yields one `Val` per row.
    pub(super) fn iter(self) -> ValRowsIter<'a> {
        match self {
            Self::ObjVec(data) => ValRowsIter::ObjVec { data, index: 0 },
            Self::Rows(rows) => ValRowsIter::Rows(rows.iter_cloned()),
            Self::Single(value) => ValRowsIter::Single(Some(value).into_iter()),
        }
    }

    /// Materialises all rows into an owned `Vec<Val>`, avoiding iteration overhead for `Rows` variants.
    pub(super) fn materialize(self) -> Vec<Val> {
        match self {
            Self::Rows(rows) => rows.into_vec(),
            other => other.iter().collect(),
        }
    }

    /// Returns `true` when this source is an `ObjVec`; used in tests to verify the source is not prematurely materialised.
    #[cfg(test)]
    pub(super) fn is_objvec_streaming(&self) -> bool {
        matches!(self, Self::ObjVec(_))
    }
}
impl<'a> TapeRowSource<'a> {
    /// Walks `keys` through `tape` and returns a `TapeRowSource` rooted at the resolved node, or `Missing` when any key is absent.
    pub(super) fn from_field_chain(
        tape: &'a crate::data::tape::TapeData,
        keys: &[Arc<str>],
    ) -> Self {
        let Some(idx) = tape_walk_field_chain(tape, keys) else {
            return Self::Missing;
        };
        Self::from_tape_index(tape, idx)
    }

    /// Constructs a `TapeRowSource` at tape node `idx`, choosing `Array` for JSON arrays and `Single` otherwise.
    pub(super) fn from_tape_index(tape: &'a crate::data::tape::TapeData, idx: usize) -> Self {
        match tape.nodes.get(idx) {
            Some(crate::data::tape::TapeNode::Array { len, .. }) => Self::Array {
                tape,
                first: idx + 1,
                len: *len,
            },
            Some(_) => Self::Single(crate::data::view::TapeView::Node { tape, idx }),
            None => Self::Missing,
        }
    }

    /// Returns a `TapeRowsIter` that yields each element as a `TapeView` without materialisation.
    pub(super) fn iter_views(self) -> TapeRowsIter<'a> {
        match self {
            Self::Array { tape, first, len } => TapeRowsIter::Array {
                tape,
                remaining: len,
                cur: first,
            },
            Self::Single(view) => TapeRowsIter::Single(Some(view).into_iter()),
            Self::Missing => TapeRowsIter::Empty,
        }
    }

    /// Returns a view iterator constrained by the selected source access mode.
    ///
    /// Direct positional access still walks tape spans to locate the selected
    /// child when an offset table is not available, but it materialises only the
    /// demanded row instead of every preceding sibling.
    pub(super) fn iter_views_for_access(self, access: SourceAccessMode) -> TapeRowsIter<'a> {
        match access {
            SourceAccessMode::Indexed(idx) => self
                .view_at(idx)
                .map(|view| TapeRowsIter::Single(Some(view).into_iter()))
                .unwrap_or(TapeRowsIter::Empty),
            SourceAccessMode::IndexedFromEnd(offset) => self
                .view_from_end(offset)
                .map(|view| TapeRowsIter::Single(Some(view).into_iter()))
                .unwrap_or(TapeRowsIter::Empty),
            SourceAccessMode::ForwardBounded(limit) => match self {
                Self::Array { tape, first, len } => TapeRowsIter::Array {
                    tape,
                    remaining: len.min(limit),
                    cur: first,
                },
                Self::Single(view) if limit > 0 => TapeRowsIter::Single(Some(view).into_iter()),
                Self::Single(_) | Self::Missing => TapeRowsIter::Empty,
            },
            SourceAccessMode::Reverse { .. } => self.iter_views_reversed(),
            SourceAccessMode::Forward | SourceAccessMode::MaterializedFallback => self.iter_views(),
        }
    }

    /// Returns a materialising iterator constrained by the selected source access mode.
    pub(super) fn iter_materialized_for_access(
        self,
        access: SourceAccessMode,
    ) -> TapeMaterializedRowsIter<'a> {
        TapeMaterializedRowsIter(self.iter_views_for_access(access))
    }

    /// Returns `true` when the tape source resolves to an array node, making it a multi-row provider.
    pub(super) fn is_array_provider(&self) -> bool {
        matches!(self, Self::Array { .. })
    }

    fn view_at(&self, idx: usize) -> Option<crate::data::view::TapeView<'a>> {
        use crate::data::view::TapeView;

        match self {
            Self::Array { tape, first, len } => {
                if idx >= *len {
                    return None;
                }
                let mut cur = *first;
                for _ in 0..idx {
                    cur += tape.span(cur);
                }
                Some(TapeView::Node { tape, idx: cur })
            }
            Self::Single(view) => (idx == 0).then_some(*view),
            Self::Missing => None,
        }
    }

    fn view_from_end(&self, offset: usize) -> Option<crate::data::view::TapeView<'a>> {
        let len = match self {
            Self::Array { len, .. } => *len,
            Self::Single(_) => 1,
            Self::Missing => return None,
        };
        let idx = len.checked_sub(offset.checked_add(1)?)?;
        self.view_at(idx)
    }

    fn iter_views_reversed(self) -> TapeRowsIter<'a> {
        match self {
            Self::Array { tape, first, len } => {
                let mut children = Vec::with_capacity(len);
                let mut cur = first;
                for _ in 0..len {
                    children.push(cur);
                    cur += tape.span(cur);
                }
                children.reverse();
                TapeRowsIter::ReverseArray {
                    tape,
                    children: children.into_iter(),
                }
            }
            Self::Single(view) => TapeRowsIter::Single(Some(view).into_iter()),
            Self::Missing => TapeRowsIter::Empty,
        }
    }
}

/// Resolves a `Source` to a `Val`, cloning the embedded receiver or walking the field-chain on `root`.
pub(super) fn resolve(source: &Source, root: &Val) -> Val {
    match source {
        Source::Receiver(v) => v.clone(),
        Source::FieldChain { keys } => walk_field_chain(root, keys),
    }
}

/// Returns `Rows` wrapping the array content of `recv`, or `None` when `recv` is a scalar with no `Cow<[Val]>` representation.
pub(super) fn array_like_rows(recv: &Val) -> Option<Rows<'_>> {
    match recv.as_vals()? {
        Cow::Borrowed(rows) => Some(Rows::Borrowed(rows)),
        Cow::Owned(rows) => Some(Rows::Owned(rows)),
    }
}

/// Constructs a `ValRowsIter` from `recv` for the streaming execution loop.
pub(super) fn source_iter(recv: &Val) -> ValRowsIter<'_> {
    ValRowSource::from_receiver(recv).iter()
}

/// Materialises all rows from `recv` into an owned `Vec<Val>`.
pub(super) fn materialize_source(recv: &Val) -> Vec<Val> {
    ValRowSource::from_receiver(recv).materialize()
}

/// Materialises at most the first `limit` rows from `recv`.
pub(super) fn materialize_source_prefix(recv: &Val, limit: usize) -> Vec<Val> {
    ValRowSource::from_receiver(recv)
        .iter()
        .take(limit)
        .collect()
}

/// Returns the number of rows in `recv`, or `None` when `recv` is a scalar or non-iterable.
pub(super) fn row_count(recv: &Val) -> Option<usize> {
    recv.array_len()
}

/// Returns the row at `idx` from an array-like `recv`, reconstructing an `ObjVec` row as `Val::Obj` when needed.
pub(super) fn row_at(recv: &Val, idx: usize) -> Option<Val> {
    if let Val::ObjVec(rows) = recv {
        return (idx < rows.nrows()).then(|| objvec_row(rows, idx));
    }

    recv.as_vals()?.get(idx).cloned()
}

/// Converts an owned `Val` into `Rows<'static>` when array-like, wrapping `Val::Arr` in `Shared` to avoid copying.
pub(super) fn resolved_array_like_rows(recv: Val) -> Option<Rows<'static>> {
    match recv {
        Val::Arr(items) => Some(Rows::Shared(items)),
        other => other.into_vals().ok().map(Rows::Owned),
    }
}

fn objvec_row(data: &ObjVecData, row: usize) -> Val {
    data.row_val(row)
}

// Returns the tape index of the final node after walking `keys`, or `None` if any key is missing.
fn tape_walk_field_chain(tape: &crate::data::tape::TapeData, keys: &[Arc<str>]) -> Option<usize> {
    let mut cur = 0usize;
    for key in keys {
        cur = tape_field(tape, cur, key.as_ref())?;
    }
    Some(cur)
}

// Scans the tape object at `idx` for `key` and returns the tape index of its value, or `None` when absent.
fn tape_field(tape: &crate::data::tape::TapeData, idx: usize, key: &str) -> Option<usize> {
    let crate::data::tape::TapeNode::Object { len, .. } = *tape.nodes.get(idx)? else {
        return None;
    };
    let mut cur = idx + 1;
    for _ in 0..len {
        if tape.str_at(cur) == key {
            return Some(cur + 1);
        }
        cur += 1;
        cur += tape.span(cur);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::tape::TapeData;
    use serde_json::json;

    fn tape_rows() -> std::sync::Arc<TapeData> {
        TapeData::parse(
            br#"{"books":[{"id":1},{"id":2},{"id":3}],"other":[{"id":99}]}"#.to_vec(),
        )
        .unwrap()
    }

    #[test]
    fn tape_row_source_materializes_only_indexed_from_end_child() {
        let tape = tape_rows();
        tape.reset_materialized_subtrees();
        let keys = [Arc::<str>::from("books")];
        let rows = TapeRowSource::from_field_chain(&tape, &keys);

        let values: Vec<_> = rows
            .iter_materialized_for_access(SourceAccessMode::IndexedFromEnd(0))
            .map(serde_json::Value::from)
            .collect();

        assert_eq!(values, vec![json!({"id": 3})]);
        assert_eq!(tape.materialized_subtrees(), 1);
    }

    #[test]
    fn tape_row_source_prefix_access_bounds_materialization() {
        let tape = tape_rows();
        tape.reset_materialized_subtrees();
        let keys = [Arc::<str>::from("books")];
        let rows = TapeRowSource::from_field_chain(&tape, &keys);

        let values: Vec<_> = rows
            .iter_materialized_for_access(SourceAccessMode::ForwardBounded(2))
            .map(serde_json::Value::from)
            .collect();

        assert_eq!(values, vec![json!({"id": 1}), json!({"id": 2})]);
        assert_eq!(tape.materialized_subtrees(), 2);
    }

    #[test]
    fn tape_row_source_reverse_access_materializes_from_end() {
        let tape = tape_rows();
        tape.reset_materialized_subtrees();
        let keys = [Arc::<str>::from("books")];
        let rows = TapeRowSource::from_field_chain(&tape, &keys);

        let values: Vec<_> = rows
            .iter_materialized_for_access(SourceAccessMode::Reverse { outputs: 1 })
            .take(2)
            .map(serde_json::Value::from)
            .collect();

        assert_eq!(values, vec![json!({"id": 3}), json!({"id": 2})]);
        assert_eq!(tape.materialized_subtrees(), 2);
    }
}
