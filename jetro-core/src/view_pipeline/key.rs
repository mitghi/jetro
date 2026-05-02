use std::sync::Arc;

use crate::{util::JsonView, value::Val};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) enum ViewKey {
    Null,
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(u64),
    Str(Arc<str>),
    Owned(Arc<str>),
}

impl ViewKey {
    pub(super) fn from_view(view: JsonView<'_>) -> Option<Self> {
        match view {
            JsonView::Null => Some(Self::Null),
            JsonView::Bool(value) => Some(Self::Bool(value)),
            JsonView::Int(value) => Some(Self::Int(value)),
            JsonView::UInt(value) => Some(Self::UInt(value)),
            JsonView::Float(value) => Some(Self::Float(value.to_bits())),
            JsonView::Str(value) => Some(Self::Str(Arc::from(value))),
            JsonView::ArrayLen(_) | JsonView::ObjectLen(_) => None,
        }
    }

    pub(super) fn from_owned(value: Val) -> Self {
        match value {
            Val::Null => Self::Null,
            Val::Bool(value) => Self::Bool(value),
            Val::Int(value) => Self::Int(value),
            Val::Float(value) => Self::Float(value.to_bits()),
            Val::Str(value) => Self::Str(value),
            value => Self::Owned(Arc::from(crate::util::val_to_key(&value).as_str())),
        }
    }

    pub(super) fn object_key(self) -> Arc<str> {
        match self {
            Self::Null => Arc::from("null"),
            Self::Bool(value) => Arc::from(if value { "true" } else { "false" }),
            Self::Int(value) => Arc::from(value.to_string().as_str()),
            Self::UInt(value) => Arc::from(value.to_string().as_str()),
            Self::Float(value) => Arc::from(f64::from_bits(value).to_string().as_str()),
            Self::Str(value) | Self::Owned(value) => value,
        }
    }
}
