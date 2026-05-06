//! Row-level source abstraction for pipeline execution.
//!
//! `Rows` unifies borrowed slice, shared Arc, and ObjVec columnar sources
//! so the execution loop can iterate without knowing which backing store
//! was used. `RowSource` drives one iteration step returning a borrowed or
//! owned Val per row.

use std::borrow::Cow;
use std::sync::Arc;

use crate::value::{ObjVecData, Val};

use super::{walk_field_chain, Source};

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
#[cfg(feature = "simd-json")]
pub(super) enum TapeRowSource<'a> {
    /// The source tape node is an array; iteration yields each element by span.
    Array {
        tape: &'a crate::tape::TapeData,
        // Index of the first array element in the tape.
        first: usize,
        // Number of elements in the array.
        len: usize,
    },
    /// The source tape node is a scalar or object; treated as a single row.
    Single(crate::value_view::TapeView<'a>),
    /// The requested field path did not resolve to any tape node.
    Missing,
}

/// Iterator over tape nodes that yields each array element as a `TapeView` without materialisation.
#[cfg(feature = "simd-json")]
pub(super) enum TapeRowsIter<'a> {
    /// Advances through array elements by consuming tape spans.
    Array {
        tape: &'a crate::tape::TapeData,
        // Elements remaining to be yielded.
        remaining: usize,
        // Current tape index.
        cur: usize,
    },
    /// Single-element iterator for a non-array tape node.
    Single(std::option::IntoIter<crate::value_view::TapeView<'a>>),
    /// Iterator for a `Missing` source; always returns `None`.
    Empty,
}

/// Wrapper around `TapeRowsIter` that materialises each `TapeView` into a `Val` on demand.
#[cfg(feature = "simd-json")]
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

#[cfg(feature = "simd-json")]
impl<'a> Iterator for TapeRowsIter<'a> {
    type Item = crate::value_view::TapeView<'a>;

    // Advances the current tape index by the node's span before returning the view.
    fn next(&mut self) -> Option<Self::Item> {
        use crate::value_view::TapeView;

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
            Self::Single(iter) => iter.next(),
            Self::Empty => None,
        }
    }
}

#[cfg(feature = "simd-json")]
impl Iterator for TapeMaterializedRowsIter<'_> {
    type Item = Val;

    fn next(&mut self) -> Option<Self::Item> {
        use crate::value_view::ValueView;

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

#[cfg(feature = "simd-json")]
impl<'a> TapeRowSource<'a> {
    /// Walks `keys` through `tape` and returns a `TapeRowSource` rooted at the resolved node, or `Missing` when any key is absent.
    pub(super) fn from_field_chain(tape: &'a crate::tape::TapeData, keys: &[Arc<str>]) -> Self {
        let Some(idx) = tape_walk_field_chain(tape, keys) else {
            return Self::Missing;
        };
        Self::from_tape_index(tape, idx)
    }

    /// Constructs a `TapeRowSource` at tape node `idx`, choosing `Array` for JSON arrays and `Single` otherwise.
    pub(super) fn from_tape_index(tape: &'a crate::tape::TapeData, idx: usize) -> Self {
        match tape.nodes.get(idx) {
            Some(crate::tape::TapeNode::Array { len, .. }) => Self::Array {
                tape,
                first: idx + 1,
                len: *len,
            },
            Some(_) => Self::Single(crate::value_view::TapeView::Node { tape, idx }),
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

    /// Returns a `TapeMaterializedRowsIter` that materialises each tape element into an owned `Val` as it iterates.
    pub(super) fn iter_materialized(self) -> TapeMaterializedRowsIter<'a> {
        TapeMaterializedRowsIter(self.iter_views())
    }

    /// Returns `true` when the tape source resolves to an array node, making it a multi-row provider.
    pub(super) fn is_array_provider(&self) -> bool {
        matches!(self, Self::Array { .. })
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
#[cfg(feature = "simd-json")]
fn tape_walk_field_chain(tape: &crate::tape::TapeData, keys: &[Arc<str>]) -> Option<usize> {
    let mut cur = 0usize;
    for key in keys {
        cur = tape_field(tape, cur, key.as_ref())?;
    }
    Some(cur)
}

// Scans the tape object at `idx` for `key` and returns the tape index of its value, or `None` when absent.
#[cfg(feature = "simd-json")]
fn tape_field(tape: &crate::tape::TapeData, idx: usize, key: &str) -> Option<usize> {
    let crate::tape::TapeNode::Object { len, .. } = *tape.nodes.get(idx)? else {
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
