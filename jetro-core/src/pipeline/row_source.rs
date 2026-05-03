use std::borrow::Cow;
use std::sync::Arc;

use crate::value::{ObjVecData, Val};

use super::{walk_field_chain, Source};

pub(super) enum Rows<'a> {
    Borrowed(&'a [Val]),
    Shared(Arc<Vec<Val>>),
    Owned(Vec<Val>),
}

pub(super) enum RowsIter<'a> {
    Borrowed(std::slice::Iter<'a, Val>),
    Shared { rows: Arc<Vec<Val>>, index: usize },
    Owned(std::vec::IntoIter<Val>),
}

pub(super) enum ValRowSource<'a> {
    ObjVec(Arc<ObjVecData>),
    Rows(Rows<'a>),
    Single(Val),
}

pub(super) enum ValRowsIter<'a> {
    Rows(RowsIter<'a>),
    ObjVec { data: Arc<ObjVecData>, index: usize },
    Single(std::option::IntoIter<Val>),
}

#[cfg(feature = "simd-json")]
pub(super) enum TapeRowSource<'a> {
    Array {
        tape: &'a crate::strref::TapeData,
        first: usize,
        len: usize,
    },
    Single(crate::value_view::TapeView<'a>),
    Missing,
}

#[cfg(feature = "simd-json")]
pub(super) enum TapeRowsIter<'a> {
    Array {
        tape: &'a crate::strref::TapeData,
        remaining: usize,
        cur: usize,
    },
    Single(std::option::IntoIter<crate::value_view::TapeView<'a>>),
    Empty,
}

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
    pub(super) fn as_slice(&self) -> &[Val] {
        match self {
            Self::Borrowed(rows) => rows,
            Self::Shared(rows) => rows.as_ref(),
            Self::Owned(rows) => rows.as_slice(),
        }
    }

    pub(super) fn iter_cloned(self) -> RowsIter<'a> {
        match self {
            Self::Borrowed(rows) => RowsIter::Borrowed(rows.iter()),
            Self::Shared(rows) => RowsIter::Shared { rows, index: 0 },
            Self::Owned(rows) => RowsIter::Owned(rows.into_iter()),
        }
    }

    pub(super) fn into_vec(self) -> Vec<Val> {
        match self {
            Self::Borrowed(rows) => rows.to_vec(),
            Self::Shared(rows) => rows.as_ref().clone(),
            Self::Owned(rows) => rows,
        }
    }
}

impl<'a> ValRowSource<'a> {
    pub(super) fn from_receiver(recv: &'a Val) -> Self {
        match recv {
            Val::ObjVec(data) => Self::ObjVec(Arc::clone(data)),
            _ => array_like_rows(recv)
                .map(Self::Rows)
                .unwrap_or_else(|| Self::Single(recv.clone())),
        }
    }

    pub(super) fn iter(self) -> ValRowsIter<'a> {
        match self {
            Self::ObjVec(data) => ValRowsIter::ObjVec { data, index: 0 },
            Self::Rows(rows) => ValRowsIter::Rows(rows.iter_cloned()),
            Self::Single(value) => ValRowsIter::Single(Some(value).into_iter()),
        }
    }

    pub(super) fn materialize(self) -> Vec<Val> {
        match self {
            Self::Rows(rows) => rows.into_vec(),
            other => other.iter().collect(),
        }
    }

    #[cfg(test)]
    pub(super) fn is_objvec_streaming(&self) -> bool {
        matches!(self, Self::ObjVec(_))
    }
}

#[cfg(feature = "simd-json")]
impl<'a> TapeRowSource<'a> {
    pub(super) fn from_field_chain(tape: &'a crate::strref::TapeData, keys: &[Arc<str>]) -> Self {
        let Some(idx) = tape_walk_field_chain(tape, keys) else {
            return Self::Missing;
        };
        Self::from_tape_index(tape, idx)
    }

    pub(super) fn from_tape_index(tape: &'a crate::strref::TapeData, idx: usize) -> Self {
        match tape.nodes.get(idx) {
            Some(crate::strref::TapeNode::Array { len, .. }) => Self::Array {
                tape,
                first: idx + 1,
                len: *len,
            },
            Some(_) => Self::Single(crate::value_view::TapeView::Node { tape, idx }),
            None => Self::Missing,
        }
    }

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

    pub(super) fn iter_materialized(self) -> TapeMaterializedRowsIter<'a> {
        TapeMaterializedRowsIter(self.iter_views())
    }

    pub(super) fn is_array_provider(&self) -> bool {
        matches!(self, Self::Array { .. })
    }
}

pub(super) fn resolve(source: &Source, root: &Val) -> Val {
    match source {
        Source::Receiver(v) => v.clone(),
        Source::FieldChain { keys } => walk_field_chain(root, keys),
    }
}

pub(super) fn array_like_rows(recv: &Val) -> Option<Rows<'_>> {
    match recv.as_vals()? {
        Cow::Borrowed(rows) => Some(Rows::Borrowed(rows)),
        Cow::Owned(rows) => Some(Rows::Owned(rows)),
    }
}

pub(super) fn source_iter(recv: &Val) -> ValRowsIter<'_> {
    ValRowSource::from_receiver(recv).iter()
}

pub(super) fn materialize_source(recv: &Val) -> Vec<Val> {
    ValRowSource::from_receiver(recv).materialize()
}

pub(super) fn row_count(recv: &Val) -> Option<usize> {
    recv.array_len()
}

pub(super) fn row_at(recv: &Val, idx: usize) -> Option<Val> {
    if let Val::ObjVec(rows) = recv {
        return (idx < rows.nrows()).then(|| objvec_row(rows, idx));
    }

    recv.as_vals()?.get(idx).cloned()
}

pub(super) fn resolved_array_like_rows(recv: Val) -> Option<Rows<'static>> {
    match recv {
        Val::Arr(items) => Some(Rows::Shared(items)),
        other => other.into_vals().ok().map(Rows::Owned),
    }
}

fn objvec_row(data: &ObjVecData, row: usize) -> Val {
    data.row_val(row)
}

#[cfg(feature = "simd-json")]
fn tape_walk_field_chain(tape: &crate::strref::TapeData, keys: &[Arc<str>]) -> Option<usize> {
    let mut cur = 0usize;
    for key in keys {
        cur = tape_field(tape, cur, key.as_ref())?;
    }
    Some(cur)
}

#[cfg(feature = "simd-json")]
fn tape_field(tape: &crate::strref::TapeData, idx: usize, key: &str) -> Option<usize> {
    let crate::strref::TapeNode::Object { len, .. } = *tape.nodes.get(idx)? else {
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
