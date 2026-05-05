use crate::context::EvalError;
use crate::value::Val;
use std::sync::Arc;

pub fn to_csv_apply(recv: &Val) -> Option<Val> {
    Some(Val::Str(Arc::from(
        crate::builtin_helpers::csv_emit(recv, ",").as_str(),
    )))
}

/// Serialises an array of arrays/objects to TSV format (tab-delimited).
#[inline]
pub fn to_tsv_apply(recv: &Val) -> Option<Val> {
    Some(Val::Str(Arc::from(
        crate::builtin_helpers::csv_emit(recv, "\t").as_str(),
    )))
}

/// Converts an object into `[{key, val}, …]`; returns an empty array for non-objects.
#[inline]
pub fn to_pairs_apply(recv: &Val) -> Option<Val> {
    use crate::util::obj2;
    let arr: Vec<Val> = recv
        .as_object()
        .map(|m| {
            m.iter()
                .map(|(k, v)| obj2("key", Val::Str(k.clone()), "val", v.clone()))
                .collect()
        })
        .unwrap_or_default();
    Some(Val::arr(arr))
}

/// Returns the runtime type name of `recv` as a `Val::Str` (e.g. `"Int"`, `"Array"`, `"Object"`).
#[inline]
pub fn type_name_apply(recv: &Val) -> Option<Val> {
    Some(Val::Str(Arc::from(recv.type_name())))
}

/// Coerces any `Val` to its human-readable string representation.
#[inline]
pub fn to_string_apply(recv: &Val) -> Option<Val> {
    Some(Val::Str(Arc::from(
        crate::util::val_to_string(recv).as_str(),
    )))
}

/// Serialises `recv` to a compact JSON string; non-finite floats become `"null"`.
#[inline]
pub fn to_json_apply(recv: &Val) -> Option<Val> {
    let out = match recv {
        Val::Int(n) => n.to_string(),
        Val::Float(f) => {
            if f.is_finite() {
                let v = serde_json::Value::from(*f);
                serde_json::to_string(&v).unwrap_or_default()
            } else {
                "null".to_string()
            }
        }
        Val::Bool(b) => (if *b { "true" } else { "false" }).to_string(),
        Val::Null => "null".to_string(),
        Val::Str(s) => {
            let v = serde_json::Value::String(s.to_string());
            serde_json::to_string(&v).unwrap_or_default()
        }
        other => {
            let sv: serde_json::Value = other.clone().into();
            serde_json::to_string(&sv).unwrap_or_default()
        }
    };
    Some(Val::Str(Arc::from(out)))
}

/// Parses a JSON string into a `Val`; silently returns `None` on parse errors.
#[inline]
pub fn from_json_apply(recv: &Val) -> Option<Val> {
    try_from_json_apply(recv).ok().flatten()
}

/// Fallible variant of [`from_json_apply`]; returns an `EvalError` on invalid JSON.
#[inline]
pub fn try_from_json_apply(recv: &Val) -> Result<Option<Val>, EvalError> {
    #[cfg(feature = "simd-json")]
    {
        let bytes_owned: Vec<u8> = match recv {
            Val::Str(s) => s.as_bytes().to_vec(),
            _ => crate::util::val_to_string(recv).into_bytes(),
        };
        let mut bytes = bytes_owned;
        return Val::from_json_simd(&mut bytes)
            .map(Some)
            .map_err(|e| EvalError(format!("from_json: {}", e)));
    }
    #[cfg(not(feature = "simd-json"))]
    {
        match recv {
            Val::Str(s) => Val::from_json_str(s.as_ref())
                .map(Some)
                .map_err(|e| EvalError(format!("from_json: {}", e))),
            _ => {
                let s = crate::util::val_to_string(recv);
                Val::from_json_str(&s)
                    .map(Some)
                    .map_err(|e| EvalError(format!("from_json: {}", e)))
            }
        }
    }
}

/// Returns `recv` if it is non-null, otherwise returns `default`.
#[inline]
pub fn or_apply(recv: &Val, default: &Val) -> Val {
    if recv.is_null() {
        default.clone()
    } else {
        recv.clone()
    }
}

/// Returns `Val::Bool(true)` when `key` is absent or null at any nesting level inside `recv`.
#[inline]
pub fn missing_apply(recv: &Val, key: &str) -> Val {
    Val::Bool(!crate::util::field_exists_nested(recv, key))
}

/// Membership test: arrays/vectors check element presence, strings check substring, objects check key.
#[inline]
pub fn includes_apply(recv: &Val, item: &Val) -> Val {
    use crate::util::val_to_key;
    let key = val_to_key(item);
    Val::Bool(match recv {
        Val::Arr(a) => a.iter().any(|v| val_to_key(v) == key),
        Val::IntVec(a) => a.iter().any(|n| val_to_key(&Val::Int(*n)) == key),
        Val::FloatVec(a) => a.iter().any(|f| val_to_key(&Val::Float(*f)) == key),
        Val::StrVec(a) => match item.as_str() {
            Some(needle) => a.iter().any(|s| s.as_ref() == needle),
            None => false,
        },
        Val::Str(s) => s.contains(item.as_str().unwrap_or_default()),
        Val::StrSlice(s) => s.as_str().contains(item.as_str().unwrap_or_default()),
        Val::Obj(m) => match item.as_str() {
            Some(k) => m.contains_key(k),
            None => false,
        },
        Val::ObjSmall(p) => match item.as_str() {
            Some(k) => p.iter().any(|(kk, _)| kk.as_ref() == k),
            None => false,
        },
        _ => false,
    })
}

