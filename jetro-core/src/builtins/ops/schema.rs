use crate::data::value::Val;
use indexmap::IndexMap;
use std::sync::Arc;

pub(crate) fn schema_of(v: &Val) -> Val {
    match v {
        Val::Null => ty_obj("Null"),
        Val::Bool(_) => ty_obj("Bool"),
        Val::Int(_) => ty_obj("Int"),
        Val::Float(_) => ty_obj("Float"),
        Val::Str(_) | Val::StrSlice(_) => ty_obj("String"),
        Val::IntVec(a) => array_schema(a.len(), ty_obj("Int")),
        Val::FloatVec(a) => array_schema(a.len(), ty_obj("Float")),
        Val::StrVec(a) => array_schema(a.len(), ty_obj("String")),
        Val::StrSliceVec(a) => array_schema(a.len(), ty_obj("String")),
        Val::ObjVec(d) => array_schema(d.nrows(), ty_obj("Object")),
        Val::Arr(a) => {
            let items = if a.is_empty() {
                ty_obj("Unknown")
            } else {
                let mut acc = schema_of(&a[0]);
                for el in a.iter().skip(1) {
                    acc = unify_schema(acc, schema_of(el));
                }
                acc
            };
            array_schema(a.len(), items)
        }
        Val::Obj(m) => schema_object(m.iter().map(|(k, v)| (k.clone(), v))),
        Val::ObjSmall(pairs) => schema_object(pairs.iter().map(|(k, v)| (k.clone(), v))),
    }
}

/// Builds an `Object` schema descriptor from an iterator of `(key, value)` pairs.
fn schema_object<'a>(pairs: impl Iterator<Item = (Arc<str>, &'a Val)>) -> Val {
    let mut required = Vec::new();
    let mut fields = IndexMap::new();
    for (k, child) in pairs {
        let mut field = schema_of(child);
        if matches!(child, Val::Null) {
            field = set_schema_field(field, "nullable", Val::Bool(true));
        } else {
            required.push(Val::Str(k.clone()));
        }
        fields.insert(k, field);
    }
    let mut out: IndexMap<Arc<str>, Val> = IndexMap::with_capacity(3);
    out.insert(Arc::from("type"), Val::Str(Arc::from("Object")));
    out.insert(Arc::from("required"), Val::arr(required));
    out.insert(Arc::from("fields"), Val::obj(fields));
    Val::obj(out)
}

/// Constructs a minimal `{type: name}` schema object.
fn ty_obj(name: &str) -> Val {
    let mut m: IndexMap<Arc<str>, Val> = IndexMap::with_capacity(1);
    m.insert(Arc::from("type"), Val::Str(Arc::from(name)));
    Val::obj(m)
}

/// Constructs an `{type: "Array", len, items}` schema object.
fn array_schema(len: usize, items: Val) -> Val {
    let mut m: IndexMap<Arc<str>, Val> = IndexMap::with_capacity(3);
    m.insert(Arc::from("type"), Val::Str(Arc::from("Array")));
    m.insert(Arc::from("len"), Val::Int(len as i64));
    m.insert(Arc::from("items"), items);
    Val::obj(m)
}

/// Inserts or overwrites a single field in a schema object; returns `obj` unchanged if not an `Obj`.
fn set_schema_field(obj: Val, key: &str, v: Val) -> Val {
    if let Val::Obj(m) = obj {
        let mut m = Arc::try_unwrap(m).unwrap_or_else(|arc| (*arc).clone());
        m.insert(Arc::from(key), v);
        Val::obj(m)
    } else {
        obj
    }
}

/// Extracts the `"type"` string from a schema object, returning `None` for non-schema values.
fn schema_type(v: &Val) -> Option<&str> {
    if let Val::Obj(m) = v {
        if let Some(Val::Str(s)) = m.get("type") {
            return Some(s.as_ref());
        }
    }
    None
}

/// Merges two schema descriptors into one, widening types as needed (same type → recurse, mismatch → `Mixed`).
fn unify_schema(a: Val, b: Val) -> Val {
    match (schema_type(&a), schema_type(&b)) {
        (Some(x), Some(y)) if x == y => match x {
            "Object" => unify_object_schemas(a, b),
            "Array" => unify_array_schemas(a, b),
            _ => mark_nullable_if_either(a, b),
        },
        (Some("Null"), _) => set_schema_field(b, "nullable", Val::Bool(true)),
        (_, Some("Null")) => set_schema_field(a, "nullable", Val::Bool(true)),
        _ => ty_obj("Mixed"),
    }
}

/// Marks schema `a` as nullable if either `a` or `b` is already nullable; otherwise returns `a` unchanged.
fn mark_nullable_if_either(a: Val, b: Val) -> Val {
    if is_schema_nullable(&a) || is_schema_nullable(&b) {
        set_schema_field(a, "nullable", Val::Bool(true))
    } else {
        a
    }
}

/// Returns `true` when the schema object carries `nullable: true`.
fn is_schema_nullable(v: &Val) -> bool {
    matches!(
        v,
        Val::Obj(m) if matches!(m.get("nullable"), Some(Val::Bool(true)))
    )
}

/// Unifies two `Array` schemas: recursively unifies item schemas and sums lengths.
fn unify_array_schemas(a: Val, b: Val) -> Val {
    let items = match (
        extract_schema_field(&a, "items"),
        extract_schema_field(&b, "items"),
    ) {
        (Some(x), Some(y)) => unify_schema(x, y),
        (Some(x), None) => x,
        (None, Some(y)) => y,
        (None, None) => ty_obj("Unknown"),
    };
    let la = extract_schema_int(&a, "len").unwrap_or(0);
    let lb = extract_schema_int(&b, "len").unwrap_or(0);
    array_schema((la + lb) as usize, items)
}

/// Extracts a field from a schema object by key, returning `None` when absent.
fn extract_schema_field(v: &Val, key: &str) -> Option<Val> {
    if let Val::Obj(m) = v {
        m.get(key).cloned()
    } else {
        None
    }
}

/// Extracts an integer field from a schema object; returns `None` when absent or not an integer.
fn extract_schema_int(v: &Val, key: &str) -> Option<i64> {
    if let Some(Val::Int(n)) = extract_schema_field(v, key) {
        Some(n)
    } else {
        None
    }
}

/// Unifies two `Object` schemas: merges field schemas, marks fields present in only one as optional.
fn unify_object_schemas(a: Val, b: Val) -> Val {
    let (Some(Val::Obj(a_fields)), Some(Val::Obj(b_fields))) = (
        extract_schema_field(&a, "fields"),
        extract_schema_field(&b, "fields"),
    ) else {
        return ty_obj("Object");
    };
    let a_map = Arc::try_unwrap(a_fields).unwrap_or_else(|arc| (*arc).clone());
    let b_map = Arc::try_unwrap(b_fields).unwrap_or_else(|arc| (*arc).clone());
    let a_req = extract_required_set(&a);
    let b_req = extract_required_set(&b);

    let mut out_fields: IndexMap<Arc<str>, Val> =
        IndexMap::with_capacity(a_map.len().max(b_map.len()));
    let mut all_keys: Vec<Arc<str>> = Vec::with_capacity(a_map.len() + b_map.len());
    for (k, _) in &a_map {
        all_keys.push(k.clone());
    }
    for (k, _) in &b_map {
        if !a_map.contains_key(k) {
            all_keys.push(k.clone());
        }
    }

    let mut required = Vec::new();
    for k in all_keys {
        let av = a_map.get(&k).cloned();
        let bv = b_map.get(&k).cloned();
        let field = match (av, bv) {
            (Some(x), Some(y)) => unify_schema(x, y),
            (Some(x), None) => set_schema_field(x, "optional", Val::Bool(true)),
            (None, Some(y)) => set_schema_field(y, "optional", Val::Bool(true)),
            _ => ty_obj("Unknown"),
        };
        if a_req.contains(k.as_ref()) && b_req.contains(k.as_ref()) {
            required.push(Val::Str(k.clone()));
        }
        out_fields.insert(k, field);
    }

    let mut out: IndexMap<Arc<str>, Val> = IndexMap::with_capacity(3);
    out.insert(Arc::from("type"), Val::Str(Arc::from("Object")));
    out.insert(Arc::from("required"), Val::arr(required));
    out.insert(Arc::from("fields"), Val::obj(out_fields));
    Val::obj(out)
}

/// Extracts the set of required field names from a schema object's `"required"` array.
fn extract_required_set(v: &Val) -> std::collections::HashSet<String> {
    let mut set = std::collections::HashSet::new();
    if let Some(Val::Arr(a)) = extract_schema_field(v, "required") {
        for el in a.iter() {
            if let Val::Str(k) = el {
                set.insert(k.to_string());
            }
        }
    }
    set
}

/// Public adapter for [`schema_of`]: infers and returns a schema descriptor for any `Val`.
#[inline]
pub fn schema_apply(recv: &Val) -> Option<Val> {
    Some(schema_of(recv))
}

#[cfg(test)]
mod spec_tests {
    use crate::builtins::{
        BuiltinCardinality, BuiltinCategory, BuiltinColumnarStage, BuiltinKeyedReducer,
        BuiltinMethod, BuiltinNumericReducer, BuiltinSelectionPosition, BuiltinSinkAccumulator,
        BuiltinSinkDemand, BuiltinSinkValueNeed, BuiltinStageMerge, BuiltinStructural,
        BuiltinViewInputMode, BuiltinViewOutputMode, BuiltinViewStage,
    };

    #[test]
    fn builtin_specs_describe_execution_shape() {
        let map = BuiltinMethod::Map.spec();
        assert_eq!(map.category, BuiltinCategory::StreamingOneToOne);
        assert_eq!(map.cardinality, BuiltinCardinality::OneToOne);

        let flat_map = BuiltinMethod::FlatMap.spec();
        assert_eq!(flat_map.category, BuiltinCategory::StreamingExpand);
        assert_eq!(flat_map.cardinality, BuiltinCardinality::Expanding);

        let sum = BuiltinMethod::Sum.spec();
        assert_eq!(sum.category, BuiltinCategory::Reducer);
        assert_eq!(sum.cardinality, BuiltinCardinality::Reducing);

        let sort = BuiltinMethod::Sort.spec();
        assert_eq!(sort.category, BuiltinCategory::Barrier);
        assert_eq!(sort.cardinality, BuiltinCardinality::Barrier);
    }

    #[test]
    fn builtin_specs_drive_view_stage_lowering() {
        assert_eq!(
            BuiltinMethod::Filter.spec().view_stage,
            Some(BuiltinViewStage::Filter)
        );
        assert_eq!(
            BuiltinMethod::Map.spec().view_stage,
            Some(BuiltinViewStage::Map)
        );
        assert_eq!(
            BuiltinMethod::FlatMap.spec().view_stage,
            Some(BuiltinViewStage::FlatMap)
        );
        assert_eq!(
            BuiltinMethod::Take.spec().view_stage,
            Some(BuiltinViewStage::Take)
        );
        assert_eq!(
            BuiltinMethod::Take.spec().stage_merge,
            Some(BuiltinStageMerge::UsizeMin)
        );
        assert_eq!(
            BuiltinMethod::Skip.spec().view_stage,
            Some(BuiltinViewStage::Skip)
        );
        assert_eq!(
            BuiltinMethod::Skip.spec().stage_merge,
            Some(BuiltinStageMerge::UsizeSaturatingAdd)
        );

        assert_eq!(BuiltinMethod::Sort.spec().view_stage, None);
        assert_eq!(BuiltinMethod::Upper.spec().view_stage, None);
    }

    #[test]
    fn builtin_specs_drive_structural_lowering() {
        assert_eq!(
            BuiltinMethod::DeepShape.spec().structural,
            Some(BuiltinStructural::DeepShape)
        );
        assert_eq!(
            BuiltinMethod::DeepLike.spec().structural,
            Some(BuiltinStructural::DeepLike)
        );
        assert_eq!(
            BuiltinMethod::DeepFind.spec().structural,
            Some(BuiltinStructural::DeepFind)
        );
    }

    #[test]
    fn builtin_view_stage_metadata_describes_view_flow() {
        assert_eq!(
            BuiltinViewStage::Filter.input_mode(),
            BuiltinViewInputMode::ReadsView
        );
        assert_eq!(
            BuiltinViewStage::Take.input_mode(),
            BuiltinViewInputMode::SkipsViewRead
        );
        assert_eq!(
            BuiltinViewStage::Map.output_mode(),
            BuiltinViewOutputMode::BorrowedSubview
        );
        assert_eq!(
            BuiltinViewStage::FlatMap.output_mode(),
            BuiltinViewOutputMode::BorrowedSubviews
        );
        assert_eq!(
            BuiltinViewStage::Skip.output_mode(),
            BuiltinViewOutputMode::PreservesInputView
        );
    }

    #[test]
    fn builtin_specs_drive_sink_lowering() {
        assert_eq!(
            BuiltinMethod::Count.spec().sink.unwrap().accumulator,
            BuiltinSinkAccumulator::Count
        );
        assert_eq!(
            BuiltinMethod::Len.spec().sink.unwrap().accumulator,
            BuiltinSinkAccumulator::Count
        );
        assert_eq!(
            BuiltinMethod::Sum.spec().sink.unwrap().accumulator,
            BuiltinSinkAccumulator::Numeric
        );
        assert_eq!(
            BuiltinMethod::Sum.spec().numeric_reducer,
            Some(BuiltinNumericReducer::Sum)
        );
        assert_eq!(
            BuiltinMethod::Avg.spec().numeric_reducer,
            Some(BuiltinNumericReducer::Avg)
        );
        assert_eq!(
            BuiltinMethod::Min.spec().numeric_reducer,
            Some(BuiltinNumericReducer::Min)
        );
        assert_eq!(
            BuiltinMethod::Max.spec().numeric_reducer,
            Some(BuiltinNumericReducer::Max)
        );
        assert_eq!(
            BuiltinMethod::First.spec().sink.unwrap().accumulator,
            BuiltinSinkAccumulator::SelectOne(BuiltinSelectionPosition::First)
        );
        assert_eq!(
            BuiltinMethod::First.spec().sink.unwrap().demand,
            BuiltinSinkDemand::First {
                value: BuiltinSinkValueNeed::Whole
            }
        );
        assert_eq!(
            BuiltinMethod::Last.spec().sink.unwrap().accumulator,
            BuiltinSinkAccumulator::SelectOne(BuiltinSelectionPosition::Last)
        );
        assert_eq!(
            BuiltinMethod::Last.spec().sink.unwrap().demand,
            BuiltinSinkDemand::Last {
                value: BuiltinSinkValueNeed::Whole
            }
        );
        assert_eq!(
            BuiltinMethod::Count.spec().sink.unwrap().demand,
            BuiltinSinkDemand::All {
                value: BuiltinSinkValueNeed::None,
                order: false
            }
        );
        assert_eq!(
            BuiltinMethod::Sum.spec().sink.unwrap().demand,
            BuiltinSinkDemand::All {
                value: BuiltinSinkValueNeed::Numeric,
                order: false
            }
        );

        assert!(BuiltinMethod::Sort.spec().sink.is_none());
    }

    #[test]
    fn builtin_specs_drive_columnar_stage_metadata() {
        assert_eq!(
            BuiltinMethod::Filter.spec().columnar_stage,
            Some(BuiltinColumnarStage::Filter)
        );
        assert_eq!(
            BuiltinMethod::Map.spec().columnar_stage,
            Some(BuiltinColumnarStage::Map)
        );
        assert_eq!(
            BuiltinMethod::FlatMap.spec().columnar_stage,
            Some(BuiltinColumnarStage::FlatMap)
        );
        assert_eq!(
            BuiltinMethod::GroupBy.spec().columnar_stage,
            Some(BuiltinColumnarStage::GroupBy)
        );
        assert_eq!(
            BuiltinMethod::CountBy.spec().keyed_reducer,
            Some(BuiltinKeyedReducer::Count)
        );
        assert_eq!(
            BuiltinMethod::IndexBy.spec().keyed_reducer,
            Some(BuiltinKeyedReducer::Index)
        );
        assert_eq!(
            BuiltinMethod::GroupBy.spec().keyed_reducer,
            Some(BuiltinKeyedReducer::Group)
        );
        assert_eq!(BuiltinMethod::Sort.spec().columnar_stage, None);
    }

    #[test]
    fn builtin_specs_drive_view_scalar_kernels() {
        let supported = [
            BuiltinMethod::Len,
            BuiltinMethod::StartsWith,
            BuiltinMethod::EndsWith,
            BuiltinMethod::Matches,
            BuiltinMethod::IndexOf,
            BuiltinMethod::LastIndexOf,
            BuiltinMethod::ByteLen,
            BuiltinMethod::IsBlank,
            BuiltinMethod::IsNumeric,
            BuiltinMethod::IsAlpha,
            BuiltinMethod::IsAscii,
            BuiltinMethod::ToNumber,
            BuiltinMethod::ToBool,
            BuiltinMethod::Ceil,
            BuiltinMethod::Floor,
            BuiltinMethod::Round,
            BuiltinMethod::Abs,
        ];
        for method in supported {
            assert!(method.is_view_scalar_method());
            assert!(method.spec().view_scalar);
        }
        assert!(!BuiltinMethod::Sort.spec().view_scalar);
        assert!(!BuiltinMethod::FromJson.spec().view_scalar);
    }
}
