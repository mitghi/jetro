use std::sync::Arc;

use crate::value::Val;

use super::{walk_field_chain, Source};

pub(super) enum Rows<'a> {
    Borrowed(&'a [Val]),
    Shared(Arc<Vec<Val>>),
    Owned(Vec<Val>),
}

impl Rows<'_> {
    pub(super) fn as_slice(&self) -> &[Val] {
        match self {
            Self::Borrowed(rows) => rows,
            Self::Shared(rows) => rows.as_ref(),
            Self::Owned(rows) => rows.as_slice(),
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
        _ => None,
    }
}
