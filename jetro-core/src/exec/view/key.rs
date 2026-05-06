//! Hashable key type for view-pipeline group-by and dedup operations.
//! `ViewKey` mirrors `Val` scalar variants but implements `Hash` and `Eq`
//! without materialising a full `Val` from the view.

use std::sync::Arc;

use crate::{util::JsonView, data::value::Val};

/// A hashable, equality-comparable key derived from a `ValueView` scalar,
/// used as the hash-map key for `group_by`, `count_by`, `index_by`, and
/// `unique_by` operations in the view pipeline.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) enum ViewKey {
    /// Represents a JSON null value.
    Null,
    /// Represents a JSON boolean value.
    Bool(bool),
    /// Represents a signed 64-bit integer.
    Int(i64),
    /// Represents an unsigned 64-bit integer (from `JsonView::UInt`).
    UInt(u64),
    /// Represents a float stored as its bit pattern to enable `Hash` + `Eq`.
    Float(u64),
    /// A borrowed string key produced from a `JsonView::Str` scalar.
    Str(Arc<str>),
    /// A string key produced by serialising a complex or owned `Val`.
    Owned(Arc<str>),
}

impl ViewKey {
    /// Constructs a `ViewKey` from a `JsonView` scalar. Returns `None` for
    /// `ArrayLen` and `ObjectLen` variants that have no scalar key representation.
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

    /// Constructs a `ViewKey` from a materialised `Val`. Complex values are
    /// serialised to a canonical string representation via `val_to_key`.
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

    /// Converts the key into an `Arc<str>` suitable for use as a JSON object key
    /// (e.g. the group name in a `group_by` result object).
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
