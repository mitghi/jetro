use std::collections::HashSet;
use std::sync::Arc;

use jetro_experimental::{StructuralIndex, TokenId, TokenKind};
#[cfg(not(feature = "simd-json"))]
use serde::Deserialize;

use crate::ast::{Arg, Expr, ObjField};
use crate::builtins::{BuiltinMethod, BuiltinStructural};
use crate::context::EvalError;
use crate::value::Val;

#[derive(Clone)]
pub(crate) enum StructuralPlan {
    DeepShape {
        anchor: Arc<[StructuralPathStep]>,
        keys: Arc<[Arc<str>]>,
    },
    DeepLike {
        anchor: Arc<[StructuralPathStep]>,
        patterns: Arc<[(Arc<str>, StructuralLiteral)]>,
    },
}

#[derive(Clone)]
pub(crate) enum StructuralPathStep {
    Field(Arc<str>),
    Index(i64),
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
    pub(crate) fn lower_builtin(
        anchor: Arc<[StructuralPathStep]>,
        method: BuiltinMethod,
        args: &[Arg],
    ) -> Option<Self> {
        match method.spec().structural? {
            BuiltinStructural::DeepShape => lower_deep_shape(anchor, args),
            BuiltinStructural::DeepLike => lower_deep_like(anchor, args),
        }
    }

    pub(crate) fn run(&self, idx: &StructuralIndex, bytes: &[u8]) -> Result<Val, EvalError> {
        match self {
            Self::DeepShape { anchor, keys } => run_deep_shape(idx, bytes, anchor, keys),
            Self::DeepLike { anchor, patterns } => run_deep_like(idx, bytes, anchor, patterns),
        }
    }
}

fn lower_deep_shape(anchor: Arc<[StructuralPathStep]>, args: &[Arg]) -> Option<StructuralPlan> {
    let Expr::Object(fields) = arg_expr(args.first()?)? else {
        return None;
    };
    let mut keys = Vec::with_capacity(fields.len());
    for field in fields {
        match field {
            ObjField::Short(key) => keys.push(Arc::from(key.as_str())),
            ObjField::Kv { key, val, .. } if matches!(val, Expr::Ident(name) if name == key) => {
                keys.push(Arc::from(key.as_str()));
            }
            _ => return None,
        }
    }
    if keys.is_empty() {
        return None;
    }
    Some(StructuralPlan::DeepShape {
        anchor,
        keys: Arc::from(keys),
    })
}

fn lower_deep_like(anchor: Arc<[StructuralPathStep]>, args: &[Arg]) -> Option<StructuralPlan> {
    let Expr::Object(fields) = arg_expr(args.first()?)? else {
        return None;
    };
    let mut patterns = Vec::with_capacity(fields.len());
    for field in fields {
        let ObjField::Kv { key, val, .. } = field else {
            return None;
        };
        patterns.push((Arc::from(key.as_str()), lower_literal(val)?));
    }
    if patterns.is_empty() {
        return None;
    }
    Some(StructuralPlan::DeepLike {
        anchor,
        patterns: Arc::from(patterns),
    })
}

fn arg_expr(arg: &Arg) -> Option<&Expr> {
    match arg {
        Arg::Pos(expr) | Arg::Named(_, expr) => Some(expr),
    }
}

fn lower_literal(expr: &Expr) -> Option<StructuralLiteral> {
    match expr {
        Expr::Null => Some(StructuralLiteral::Null),
        Expr::Bool(v) => Some(StructuralLiteral::Bool(*v)),
        Expr::Int(v) => Some(StructuralLiteral::Int(*v)),
        Expr::Float(v) => Some(StructuralLiteral::Float(*v)),
        Expr::Str(v) => Some(StructuralLiteral::Str(Arc::from(v.as_str()))),
        _ => None,
    }
}

fn run_deep_shape(
    idx: &StructuralIndex,
    bytes: &[u8],
    anchor: &[StructuralPathStep],
    keys: &[Arc<str>],
) -> Result<Val, EvalError> {
    if keys.is_empty() {
        return Err(EvalError("shape: empty pattern".into()));
    }
    let Some(anchor) = anchor_token(idx, anchor) else {
        return Ok(Val::arr(Vec::new()));
    };
    let mut out = Vec::new();
    visit_candidate_objects(idx, anchor, keys, |object| {
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
    anchor: &[StructuralPathStep],
    patterns: &[(Arc<str>, StructuralLiteral)],
) -> Result<Val, EvalError> {
    if patterns.is_empty() {
        return Err(EvalError("like: empty pattern".into()));
    }
    let Some(anchor) = anchor_token(idx, anchor) else {
        return Ok(Val::arr(Vec::new()));
    };
    let keys: Vec<Arc<str>> = patterns.iter().map(|(key, _)| Arc::clone(key)).collect();
    let mut out = Vec::new();
    visit_candidate_objects(idx, anchor, &keys, |object| {
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
    anchor: TokenId,
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
    for key_tok in idx.keys_named_in(first_key, anchor) {
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

fn anchor_token(idx: &StructuralIndex, steps: &[StructuralPathStep]) -> Option<TokenId> {
    let mut cur = TokenId::from(0);
    for step in steps {
        cur = match step {
            StructuralPathStep::Field(key) => idx.field_of(cur, key),
            StructuralPathStep::Index(index) => array_child_at(idx, cur, *index),
        }?;
    }
    Some(cur)
}

fn array_child_at(idx: &StructuralIndex, array: TokenId, index: i64) -> Option<TokenId> {
    if idx.kind(array) != TokenKind::Array {
        return None;
    }
    let close = idx
        .close_of(array)
        .map(|tok| tok.raw())
        .unwrap_or_else(|| idx.token_count().saturating_sub(1));
    let children: Vec<TokenId> = idx
        .tokens()
        .skip((array.raw() + 1) as usize)
        .take_while(|tok| tok.raw() < close)
        .filter(|tok| idx.parent(*tok) == Some(array))
        .collect();
    let pos = if index < 0 {
        children.len() as i64 + index
    } else {
        index
    };
    if pos < 0 {
        return None;
    }
    children.get(pos as usize).copied()
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
