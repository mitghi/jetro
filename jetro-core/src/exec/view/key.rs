//! Hashable key type for view-pipeline group-by and dedup operations.
//! `ViewKey` mirrors `Val` scalar variants but implements `Hash` and `Eq`
//! without materialising a full `Val` from the view.

use std::sync::Arc;

use crate::{
    data::value::Val,
    data::view::{write_json_view, ValueView},
    util::JsonView,
};

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
            Val::StrSlice(value) => Self::Str(Arc::from(value.as_str())),
            value => Self::Owned(Arc::from(crate::util::val_to_key(&value).as_str())),
        }
    }

    pub(super) fn from_structural_owned(value: Val) -> Self {
        match value {
            Val::Null => Self::Null,
            Val::Bool(value) => Self::Bool(value),
            Val::Int(value) => structural_number_key(value as f64),
            Val::Float(value) => structural_number_key(value),
            Val::Str(value) => Self::Str(value),
            Val::StrSlice(value) => Self::Str(Arc::from(value.as_str())),
            value => Self::Owned(Arc::from(
                crate::util::val_to_structural_key(&value).as_str(),
            )),
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
            write_json_view(view, &mut out)?;
            Some(Self::Owned(Arc::from(out)))
        })
    }

    pub(super) fn from_structural_value_view<'a, V>(view: &V) -> Option<Self>
    where
        V: ValueView<'a> + 'a,
    {
        from_structural_scalar_view(view.scalar()).or_else(|| {
            let mut out = String::new();
            write_structural_view_key(view, &mut out)?;
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

fn from_structural_scalar_view(view: JsonView<'_>) -> Option<ViewKey> {
    match view {
        JsonView::Null => Some(ViewKey::Null),
        JsonView::Bool(value) => Some(ViewKey::Bool(value)),
        JsonView::Int(value) => Some(structural_number_key(value as f64)),
        JsonView::UInt(value) => Some(structural_number_key(value as f64)),
        JsonView::Float(value) => Some(structural_number_key(value)),
        JsonView::Str(value) => Some(ViewKey::Str(Arc::from(value))),
        JsonView::ArrayLen(_) | JsonView::ObjectLen(_) => None,
    }
}

fn structural_number_key(value: f64) -> ViewKey {
    let value = if value == 0.0 { 0.0 } else { value };
    ViewKey::Float(value.to_bits())
}

fn write_structural_view_key<'a, V>(view: &V, out: &mut String) -> Option<()>
where
    V: ValueView<'a> + 'a,
{
    match view.scalar() {
        JsonView::Null => out.push_str("z:null"),
        JsonView::Bool(value) => {
            out.push_str("b:");
            out.push_str(if value { "1" } else { "0" });
        }
        JsonView::Int(value) => {
            out.push_str("n:");
            out.push_str(&(value as f64).to_bits().to_string());
        }
        JsonView::UInt(value) => {
            out.push_str("n:");
            let number = if value <= i64::MAX as u64 {
                value as i64 as f64
            } else {
                value as f64
            };
            out.push_str(&number.to_bits().to_string());
        }
        JsonView::Float(value) => {
            out.push_str("n:");
            let number = if value == 0.0 { 0.0 } else { value };
            out.push_str(&number.to_bits().to_string());
        }
        JsonView::Str(value) => {
            out.push_str("s:");
            out.push_str(&value.len().to_string());
            out.push(':');
            out.push_str(value);
        }
        JsonView::ArrayLen(_) => {
            out.push_str("a[");
            for item in view.array_iter()? {
                write_structural_view_key(&item, out)?;
                out.push(',');
            }
            out.push(']');
        }
        JsonView::ObjectLen(_) => {
            out.push_str("o{");
            let mut fields = view.object_iter()?.collect::<Vec<_>>();
            fields.sort_unstable_by(|a, b| a.0.cmp(&b.0));
            for (key, value) in fields {
                out.push_str(&key.len().to_string());
                out.push(':');
                out.push_str(&key);
                out.push('=');
                write_structural_view_key(&value, out)?;
                out.push(',');
            }
            out.push('}');
        }
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use crate::data::{
        tape::TapeData,
        value::Val,
        view::{TapeView, ValView, ValueView},
    };

    use super::ViewKey;

    #[test]
    fn structural_scalar_numbers_share_key_across_int_and_float_views() {
        let int = Val::Int(1);
        let float = Val::Float(1.0);

        assert_eq!(
            ViewKey::from_structural_owned(int.clone()),
            ViewKey::from_structural_owned(float.clone())
        );
        assert_eq!(
            ViewKey::from_structural_value_view(&ValView::new(&int)),
            ViewKey::from_structural_value_view(&ValView::new(&float))
        );
    }

    #[test]
    fn owned_string_slice_uses_scalar_key_without_stringification() {
        let tape = TapeData::parse(br#"{"name":"ada"}"#.to_vec()).unwrap();
        let name = TapeView::root(&tape).field("name").materialize();

        assert_eq!(
            ViewKey::from_owned(name.clone()),
            ViewKey::from_value_view(&ValView::new(&name)).unwrap()
        );
        assert_eq!(
            ViewKey::from_structural_owned(name.clone()),
            ViewKey::from_structural_value_view(&ValView::new(&name)).unwrap()
        );
    }

    #[test]
    fn structural_compound_tape_keys_are_canonical_without_materializing() {
        let left = TapeData::parse(br#"{"a":1,"b":[1,{"x":true}]}"#.to_vec()).unwrap();
        let right = TapeData::parse(br#"{"b":[1.0,{"x":true}],"a":1.0}"#.to_vec()).unwrap();
        left.reset_materialized_subtrees();
        right.reset_materialized_subtrees();

        assert_eq!(
            ViewKey::from_structural_value_view(&TapeView::root(&left)),
            ViewKey::from_structural_value_view(&TapeView::root(&right))
        );
        assert_eq!(left.materialized_subtrees(), 0);
        assert_eq!(right.materialized_subtrees(), 0);
    }
}
