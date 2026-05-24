//! Hashable key type for view-pipeline group-by and dedup operations.
//! `ViewKey` mirrors `Val` scalar variants but implements `Hash` and `Eq`
//! without materialising a full `Val` from the view.

use std::sync::Arc;

use crate::{data::value::Val, data::view::ValueView, util::JsonView};

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

    /// Constructs a key from a borrowed view, serialising compound values by
    /// traversing child views instead of materialising the subtree into `Val`.
    pub(super) fn from_value_view<'a, V>(view: &V) -> Option<Self>
    where
        V: ValueView<'a> + 'a,
    {
        Self::from_view(view.scalar()).or_else(|| {
            let mut out = String::new();
            write_view_key(view, &mut out)?;
            Some(Self::Owned(Arc::from(out)))
        })
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

fn write_view_key<'a, V>(view: &V, out: &mut String) -> Option<()>
where
    V: ValueView<'a> + 'a,
{
    match view.scalar() {
        JsonView::Null => out.push_str("null"),
        JsonView::Bool(value) => out.push_str(if value { "true" } else { "false" }),
        JsonView::Int(value) => out.push_str(&value.to_string()),
        JsonView::UInt(value) => out.push_str(&value.to_string()),
        JsonView::Float(value) => out.push_str(&value.to_string()),
        JsonView::Str(value) => {
            out.push_str(&serde_json::to_string(value).ok()?);
        }
        JsonView::ArrayLen(_) => {
            out.push('[');
            let mut first = true;
            for child in view.array_iter()? {
                if first {
                    first = false;
                } else {
                    out.push(',');
                }
                write_view_key(&child, out)?;
            }
            out.push(']');
        }
        JsonView::ObjectLen(_) => {
            out.push('{');
            let mut first = true;
            for (key, child) in view.object_iter()? {
                if first {
                    first = false;
                } else {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(key.as_ref()).ok()?);
                out.push(':');
                write_view_key(&child, out)?;
            }
            out.push('}');
        }
    }
    Some(())
}
