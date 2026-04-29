use crate::util::JsonView;
use crate::value::Val;

pub(crate) trait ValueView<'a>: Clone {
    fn scalar(&self) -> JsonView<'_>;
    fn field(&self, key: &str) -> Self;
    fn index(&self, idx: i64) -> Self;
    fn materialize(&self) -> Val;
}

#[derive(Clone)]
pub(crate) enum ValView<'a> {
    Borrowed(&'a Val),
    Owned(Val),
}

impl<'a> ValView<'a> {
    #[inline]
    pub(crate) fn new(value: &'a Val) -> Self {
        Self::Borrowed(value)
    }

    #[inline]
    fn value(&self) -> &Val {
        match self {
            Self::Borrowed(value) => value,
            Self::Owned(value) => value,
        }
    }
}

impl<'a> ValueView<'a> for ValView<'a> {
    #[inline]
    fn scalar(&self) -> JsonView<'_> {
        JsonView::from_val(self.value())
    }

    #[inline]
    fn field(&self, key: &str) -> Self {
        match self {
            Self::Borrowed(Val::Obj(map)) => map
                .get(key)
                .map(Self::Borrowed)
                .unwrap_or_else(|| Self::Owned(Val::Null)),
            Self::Borrowed(Val::ObjSmall(pairs)) => pairs
                .iter()
                .find_map(|(k, v)| (k.as_ref() == key).then_some(Self::Borrowed(v)))
                .unwrap_or_else(|| Self::Owned(Val::Null)),
            Self::Borrowed(_) => Self::Owned(Val::Null),
            Self::Owned(value) => Self::Owned(value.get_field(key)),
        }
    }

    #[inline]
    fn index(&self, idx: i64) -> Self {
        match self {
            Self::Borrowed(Val::Arr(items)) => normalize_index(items.len(), idx)
                .and_then(|idx| items.get(idx))
                .map(Self::Borrowed)
                .unwrap_or_else(|| Self::Owned(Val::Null)),
            Self::Borrowed(Val::IntVec(items)) => normalize_index(items.len(), idx)
                .and_then(|idx| items.get(idx).copied())
                .map(Val::Int)
                .map(Self::Owned)
                .unwrap_or_else(|| Self::Owned(Val::Null)),
            Self::Borrowed(Val::FloatVec(items)) => normalize_index(items.len(), idx)
                .and_then(|idx| items.get(idx).copied())
                .map(Val::Float)
                .map(Self::Owned)
                .unwrap_or_else(|| Self::Owned(Val::Null)),
            Self::Borrowed(Val::StrVec(items)) => normalize_index(items.len(), idx)
                .and_then(|idx| items.get(idx).cloned())
                .map(Val::Str)
                .map(Self::Owned)
                .unwrap_or_else(|| Self::Owned(Val::Null)),
            Self::Borrowed(Val::StrSliceVec(items)) => normalize_index(items.len(), idx)
                .and_then(|idx| items.get(idx).cloned())
                .map(Val::StrSlice)
                .map(Self::Owned)
                .unwrap_or_else(|| Self::Owned(Val::Null)),
            Self::Borrowed(_) => Self::Owned(Val::Null),
            Self::Owned(value) => Self::Owned(value.get_index(idx)),
        }
    }

    #[inline]
    fn materialize(&self) -> Val {
        self.value().clone()
    }
}

#[inline]
fn normalize_index(len: usize, idx: i64) -> Option<usize> {
    let idx = if idx < 0 {
        len.checked_sub(idx.unsigned_abs() as usize)?
    } else {
        idx as usize
    };
    (idx < len).then_some(idx)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::json;

    use super::{ValView, ValueView};
    use crate::util::{json_cmp_binop, JsonView};
    use crate::{ast::BinOp, value::Val};

    #[test]
    fn val_view_reads_nested_fields_without_materializing_parent() {
        let value = Val::from(&json!({
            "book": {"title": "Dune", "score": 901},
            "unused": {"payload": [1, 2, 3]}
        }));
        let root = ValView::new(&value);

        let title = root.field("book").field("title");
        let score = root.field("book").field("score");

        assert!(matches!(title.scalar(), JsonView::Str("Dune")));
        assert!(json_cmp_binop(
            score.scalar(),
            BinOp::Gt,
            JsonView::Int(900)
        ));
    }

    #[test]
    fn val_view_indexes_columnar_arrays() {
        let nums = Val::IntVec(Arc::new(vec![10, 20, 30]));
        let view = ValView::new(&nums);

        assert!(matches!(view.index(1).scalar(), JsonView::Int(20)));
        assert!(matches!(view.index(-1).scalar(), JsonView::Int(30)));
        assert!(matches!(view.index(99).scalar(), JsonView::Null));
    }

    #[test]
    fn val_view_materializes_current_view_only() {
        let value = Val::from(&json!({"items": [{"id": 1}, {"id": 2}]}));
        let item = ValView::new(&value).field("items").index(1);

        assert_eq!(
            serde_json::Value::from(item.materialize()),
            json!({"id": 2})
        );
    }
}
