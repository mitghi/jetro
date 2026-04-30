use std::sync::Arc;

use crate::value::Val;

use super::{walk_field_chain, Source};

pub(super) enum Rows<'a> {
    Borrowed(&'a [Val]),
    Owned(Vec<Val>),
}

impl Rows<'_> {
    pub(super) fn into_vec(self) -> Vec<Val> {
        match self {
            Self::Borrowed(rows) => rows.to_vec(),
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
