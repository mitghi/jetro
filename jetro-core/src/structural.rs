use std::collections::HashSet;
use std::sync::Arc;

use jetro_experimental::{StructuralIndex, TokenId, TokenKind};
#[cfg(not(feature = "simd-json"))]
use serde::Deserialize;

use crate::context::EvalError;
use crate::value::Val;

#[derive(Clone)]
pub(crate) enum StructuralPlan {
    DeepShape {
        keys: Arc<[Arc<str>]>,
    },
    DeepLike {
        patterns: Arc<[(Arc<str>, StructuralLiteral)]>,
    },
}

#[derive(Clone)]
pub(crate) enum StructuralLiteral {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(Arc<str>),
}

impl StructuralPlan {
    pub(crate) fn run(&self, idx: &StructuralIndex, bytes: &[u8]) -> Result<Val, EvalError> {
        match self {
            Self::DeepShape { keys } => run_deep_shape(idx, bytes, keys),
            Self::DeepLike { patterns } => run_deep_like(idx, bytes, patterns),
        }
    }
}

fn run_deep_shape(
    idx: &StructuralIndex,
    bytes: &[u8],
    keys: &[Arc<str>],
) -> Result<Val, EvalError> {
    if keys.is_empty() {
        return Err(EvalError("shape: empty pattern".into()));
    }
    let mut out = Vec::new();
    visit_candidate_objects(idx, keys, |object| {
        if keys.iter().all(|key| idx.field_of(object, key).is_some()) {
            out.push(materialize_token(idx, bytes, object)?);
        }
        Ok(())
    })?;
    Ok(Val::arr(out))
}

fn run_deep_like(
    idx: &StructuralIndex,
    bytes: &[u8],
    patterns: &[(Arc<str>, StructuralLiteral)],
) -> Result<Val, EvalError> {
    if patterns.is_empty() {
        return Err(EvalError("like: empty pattern".into()));
    }
    let keys: Vec<Arc<str>> = patterns.iter().map(|(key, _)| Arc::clone(key)).collect();
    let mut out = Vec::new();
    visit_candidate_objects(idx, &keys, |object| {
        for (key, want) in patterns {
            let Some(value) = idx.field_of(object, key) else {
                return Ok(());
            };
            if !literal_matches(idx, bytes, value, want) {
                return Ok(());
            }
        }
        out.push(materialize_token(idx, bytes, object)?);
        Ok(())
    })?;
    Ok(Val::arr(out))
}

fn visit_candidate_objects<F>(
    idx: &StructuralIndex,
    keys: &[Arc<str>],
    mut visit: F,
) -> Result<(), EvalError>
where
    F: FnMut(TokenId) -> Result<(), EvalError>,
{
    let Some(first_key) = keys.first() else {
        return Ok(());
    };
    let mut seen = HashSet::new();
    for key_tok in idx.keys_named(first_key, None) {
        let Some(parent) = idx.parent(key_tok) else {
            continue;
        };
        if idx.kind(parent) != TokenKind::Object || !seen.insert(parent.raw()) {
            continue;
        }
        visit(parent)?;
    }
    Ok(())
}

fn literal_matches(
    idx: &StructuralIndex,
    bytes: &[u8],
    value: TokenId,
    want: &StructuralLiteral,
) -> bool {
    let span = idx.byte_span_in(value, bytes);
    let raw = span.slice(bytes);
    match want {
        StructuralLiteral::Null => raw == b"null",
        StructuralLiteral::Bool(v) => raw == if *v { &b"true"[..] } else { &b"false"[..] },
        StructuralLiteral::Int(v) => {
            let Ok(text) = std::str::from_utf8(raw) else {
                return false;
            };
            text.parse::<i64>().map(|got| got == *v).unwrap_or(false)
        }
        StructuralLiteral::Float(v) => {
            let Ok(text) = std::str::from_utf8(raw) else {
                return false;
            };
            text.parse::<f64>()
                .map(|got| (got - *v).abs() <= f64::EPSILON)
                .unwrap_or(false)
        }
        StructuralLiteral::Str(v) => jetro_experimental::json_string_eq(raw, v.as_bytes()),
    }
}

fn materialize_token(idx: &StructuralIndex, bytes: &[u8], tok: TokenId) -> Result<Val, EvalError> {
    let span = idx.byte_span_in(tok, bytes);
    let raw = span.slice(bytes);
    #[cfg(feature = "simd-json")]
    {
        let mut owned = raw.to_vec();
        return Val::from_json_simd(&mut owned)
            .map_err(|err| EvalError(format!("Invalid JSON subtree: {err}")));
    }
    #[cfg(not(feature = "simd-json"))]
    {
        let mut de = serde_json::Deserializer::from_slice(raw);
        let v = Val::deserialize(&mut de)
            .map_err(|err| EvalError(format!("Invalid JSON subtree: {err}")))?;
        de.end()
            .map_err(|err| EvalError(format!("Invalid JSON subtree: {err}")))?;
        Ok(v)
    }
}
