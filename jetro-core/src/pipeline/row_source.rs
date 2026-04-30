use std::sync::Arc;

use crate::value::Val;

use super::{walk_field_chain, Source};

pub(super) fn resolve(source: &Source, root: &Val) -> Val {
    match source {
        Source::Receiver(v) => v.clone(),
        Source::FieldChain { keys } => walk_field_chain(root, keys),
    }
}

pub(super) fn array_like_rows(recv: &Val) -> Option<Vec<Val>> {
    match recv {
        Val::Arr(items) => Some(items.as_ref().clone()),
        Val::IntVec(items) => Some(items.iter().map(|n| Val::Int(*n)).collect()),
        Val::FloatVec(items) => Some(items.iter().map(|f| Val::Float(*f)).collect()),
        Val::StrVec(items) => Some(items.iter().map(|s| Val::Str(Arc::clone(s))).collect()),
        _ => None,
    }
}
