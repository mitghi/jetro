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

pub(super) enum SourceIter<'a> {
    Rows(RowsIter<'a>),
    ObjVec { data: Arc<ObjVecData>, index: usize },
    Single(std::option::IntoIter<Val>),
}

impl Iterator for SourceIter<'_> {
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

pub(super) fn resolve(source: &Source, root: &Val) -> Val {
    match source {
        Source::Receiver(v) => v.clone(),
        Source::FieldChain { keys } => walk_field_chain(root, keys),
    }
}

pub(super) fn array_like_rows(recv: &Val) -> Option<Rows<'_>> {
    match recv {
        Val::Arr(items) => Some(Rows::Borrowed(items.as_ref())),
        Val::IntVec(items) => Some(Rows::Owned(items.iter().map(|n| Val::Int(*n)).collect())),
        Val::FloatVec(items) => Some(Rows::Owned(items.iter().map(|f| Val::Float(*f)).collect())),
        Val::StrVec(items) => Some(Rows::Owned(
            items.iter().map(|s| Val::Str(Arc::clone(s))).collect(),
        )),
        Val::StrSliceVec(items) => Some(Rows::Owned(
            items.iter().map(|s| Val::StrSlice(s.clone())).collect(),
        )),
        _ => None,
    }
}

pub(super) fn source_iter(recv: &Val) -> SourceIter<'_> {
    if let Some(rows) = array_like_rows(recv) {
        return SourceIter::Rows(rows.iter_cloned());
    }

    match recv {
        Val::ObjVec(data) => SourceIter::ObjVec {
            data: Arc::clone(data),
            index: 0,
        },
        _ => SourceIter::Single(Some(recv.clone()).into_iter()),
    }
}

pub(super) fn materialize_source(recv: &Val) -> Vec<Val> {
    if let Some(rows) = array_like_rows(recv) {
        rows.into_vec()
    } else {
        source_iter(recv).collect()
    }
}

pub(super) fn row_count(recv: &Val) -> Option<usize> {
    match recv {
        Val::Arr(rows) => Some(rows.len()),
        Val::IntVec(rows) => Some(rows.len()),
        Val::FloatVec(rows) => Some(rows.len()),
        Val::StrVec(rows) => Some(rows.len()),
        Val::StrSliceVec(rows) => Some(rows.len()),
        Val::ObjVec(rows) => Some(rows.nrows()),
        _ => None,
    }
}

pub(super) fn row_at(recv: &Val, idx: usize) -> Option<Val> {
    match recv {
        Val::Arr(rows) => rows.get(idx).cloned(),
        Val::IntVec(rows) => rows.get(idx).map(|n| Val::Int(*n)),
        Val::FloatVec(rows) => rows.get(idx).map(|f| Val::Float(*f)),
        Val::StrVec(rows) => rows.get(idx).map(|s| Val::Str(Arc::clone(s))),
        Val::StrSliceVec(rows) => rows.get(idx).map(|s| Val::StrSlice(s.clone())),
        Val::ObjVec(rows) if idx < rows.nrows() => Some(objvec_row(rows, idx)),
        Val::ObjVec(_) => None,
        _ => None,
    }
}

pub(super) fn resolved_array_like_rows(recv: Val) -> Option<Rows<'static>> {
    match recv {
        Val::Arr(items) => Some(Rows::Shared(items)),
        Val::IntVec(items) => Some(Rows::Owned(items.iter().map(|n| Val::Int(*n)).collect())),
        Val::FloatVec(items) => Some(Rows::Owned(items.iter().map(|f| Val::Float(*f)).collect())),
        Val::StrVec(items) => Some(Rows::Owned(
            items.iter().map(|s| Val::Str(Arc::clone(s))).collect(),
        )),
        Val::StrSliceVec(items) => Some(Rows::Owned(
            items.iter().map(|s| Val::StrSlice(s.clone())).collect(),
        )),
        _ => None,
    }
}

fn objvec_row(data: &ObjVecData, row: usize) -> Val {
    data.row_val(row)
}
