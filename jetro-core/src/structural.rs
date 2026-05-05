//! Structural query plans backed by the `jetro-experimental` bitmap index.
//!
//! `StructuralPlan` variants (`DeepFind`, `DeepShape`, `DeepLike`) are emitted
//! by the planner for DFS deep-search operations. At execution time the plan
//! is evaluated against `StructuralIndex` — a pre-built key-bitmap over the
//! raw JSON bytes — so matching never materialises a `Val` tree for unmatched
//! branches. Falls back to a `Val`-walk interpreter when the index is absent.

use std::collections::HashSet;
use std::sync::Arc;

use jetro_experimental::{StructuralIndex, TokenId, TokenKind};
#[cfg(not(feature = "simd-json"))]
use serde::Deserialize;

use crate::ast::{Arg, BinOp, Expr, KindType, ObjField, Step};
use crate::builtin_registry::{self, BuiltinId};
use crate::builtins::{BuiltinMethod, BuiltinStructural};
use crate::context::EvalError;
use crate::value::Val;

/// A compiled structural deep-search plan. Carried inside `PlanNode::Structural`
/// and evaluated by `physical_eval` against a `StructuralIndex`.
#[derive(Clone)]
pub(crate) enum StructuralPlan {
    /// DFS search that returns all descendant objects satisfying every predicate
    /// in `predicates`, starting from the node reached by `anchor`.
    DeepFind {
        /// Path from the document root to the subtree to search.
        anchor: Arc<[StructuralPathStep]>,
        /// Predicates that each candidate object must satisfy to be included.
        predicates: Arc<[StructuralPredicate]>,
    },
    /// Returns all descendant objects that possess every key listed in `keys`,
    /// without checking the values of those keys.
    DeepShape {
        /// Path from the document root to the subtree to search.
        anchor: Arc<[StructuralPathStep]>,
        /// Field names that must all be present on matching objects.
        keys: Arc<[Arc<str>]>,
    },
    /// Returns all descendant objects where each key in `patterns` maps to the
    /// corresponding literal value.
    DeepLike {
        /// Path from the document root to the subtree to search.
        anchor: Arc<[StructuralPathStep]>,
        /// Key/literal pairs that must all match on each candidate object.
        patterns: Arc<[(Arc<str>, StructuralLiteral)]>,
    },
}

/// One step along the anchor path from the document root to the search subtree.
/// Mirrors `PathStep` but restricted to the subset the structural index can resolve.
#[derive(Clone)]
pub(crate) enum StructuralPathStep {
    /// Descend into the named field of an object token.
    Field(Arc<str>),
    /// Descend into the element at the given index of an array token.
    Index(i64),
}

/// A literal value pattern used in `StructuralPlan::DeepLike` and
/// `StructuralPredicate::FieldEqLiteral` to compare against raw JSON bytes
/// without deserialising the full subtree.
#[derive(Clone)]
pub(crate) enum StructuralLiteral {
    /// Match JSON `null`.
    Null,
    /// Match JSON `true` or `false`.
    Bool(bool),
    /// Match a JSON integer by parsed `i64` value.
    Int(i64),
    /// Match a JSON float within `f64::EPSILON`.
    Float(f64),
    /// Match a JSON string using `jetro_experimental::json_string_eq`.
    Str(Arc<str>),
}

/// A composable predicate evaluated against candidate object tokens in the
/// structural index. All matching is done directly on raw JSON byte spans.
#[derive(Clone)]
pub(crate) enum StructuralPredicate {
    /// The candidate token must have kind `Object`.
    KindObject,
    /// The named field of the candidate object must equal the given literal.
    FieldEqLiteral(Arc<str>, StructuralLiteral),
    /// All sub-predicates must hold; flattened from nested `and` expressions.
    And(Arc<[StructuralPredicate]>),
}

impl StructuralPlan {
    /// Attempt to lower a `BuiltinMethod` + args combination into a
    /// `StructuralPlan` starting at `anchor`. Returns `None` when the method is
    /// not structurally lowerable or the arguments are too complex for the index.
    pub(crate) fn lower_builtin(
        anchor: Arc<[StructuralPathStep]>,
        method: BuiltinMethod,
        args: &[Arg],
    ) -> Option<Self> {
        match builtin_registry::structural(BuiltinId::from_method(method))? {
            BuiltinStructural::DeepFind => lower_deep_find(anchor, args),
            BuiltinStructural::DeepShape => lower_deep_shape(anchor, args),
            BuiltinStructural::DeepLike => lower_deep_like(anchor, args),
        }
    }

    /// Execute the plan against `idx` (pre-built structural index) and the raw
    /// `bytes` of the JSON document, returning a `Val::Arr` of matching objects.
    pub(crate) fn run(&self, idx: &StructuralIndex, bytes: &[u8]) -> Result<Val, EvalError> {
        match self {
            Self::DeepFind { anchor, predicates } => run_deep_find(idx, bytes, anchor, predicates),
            Self::DeepShape { anchor, keys } => run_deep_shape(idx, bytes, anchor, keys),
            Self::DeepLike { anchor, patterns } => run_deep_like(idx, bytes, anchor, patterns),
        }
    }
}

/// Attempt to lower `args` into a `StructuralPlan::DeepFind`; returns `None`
/// when no arguments are present or any argument expression is too complex to
/// compile to a `StructuralPredicate`.
fn lower_deep_find(anchor: Arc<[StructuralPathStep]>, args: &[Arg]) -> Option<StructuralPlan> {
    if args.is_empty() {
        return None;
    }
    let mut predicates = Vec::with_capacity(args.len());
    for arg in args {
        predicates.push(lower_predicate(arg_expr(arg)?)?);
    }
    Some(StructuralPlan::DeepFind {
        anchor,
        predicates: Arc::from(predicates),
    })
}

/// Attempt to lower `args` (expected to be a single object expression) into a
/// `StructuralPlan::DeepShape`; returns `None` when the argument is not a
/// simple key list or contains unsupported field patterns.
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

/// Attempt to lower `args` (expected to be a single object expression with
/// literal values) into a `StructuralPlan::DeepLike`; returns `None` when
/// any value is not a lowerable literal or the arg shape is unsupported.
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

/// Extract the `&Expr` inner value from a positional or named `Arg`, returning
/// `None` only when an unsupported variant is encountered (currently never).
fn arg_expr(arg: &Arg) -> Option<&Expr> {
    match arg {
        Arg::Pos(expr) | Arg::Named(_, expr) => Some(expr),
    }
}

/// Attempt to lower a constant `Expr` into a `StructuralLiteral`. Returns `None`
/// for any expression that is not a compile-time constant scalar.
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

/// Attempt to lower a predicate `Expr` into a `StructuralPredicate`. Handles
/// `and` conjunctions (flattened), `field == literal` comparisons (both
/// orderings), and `@ kind is object` type checks. Returns `None` for anything
/// that cannot be expressed in terms of the structural index.
fn lower_predicate(expr: &Expr) -> Option<StructuralPredicate> {
    match expr {
        Expr::BinOp(lhs, BinOp::And, rhs) => {
            let mut parts = Vec::new();
            flatten_and_predicate(lhs, &mut parts)?;
            flatten_and_predicate(rhs, &mut parts)?;
            Some(StructuralPredicate::And(Arc::from(parts)))
        }
        Expr::BinOp(lhs, BinOp::Eq, rhs) => {
            if let (Some(field), Some(lit)) = (field_ref(lhs), lower_literal(rhs)) {
                return Some(StructuralPredicate::FieldEqLiteral(field, lit));
            }
            if let (Some(field), Some(lit)) = (field_ref(rhs), lower_literal(lhs)) {
                return Some(StructuralPredicate::FieldEqLiteral(field, lit));
            }
            None
        }
        Expr::Kind { expr, ty, negate } if !*negate && *ty == KindType::Object => {
            matches!(expr.as_ref(), Expr::Current).then_some(StructuralPredicate::KindObject)
        }
        _ => None,
    }
}

/// Recursively flatten an `And` predicate into `out`, collapsing nested `And`
/// nodes. Returns `None` when `lower_predicate` fails on any sub-expression.
fn flatten_and_predicate(expr: &Expr, out: &mut Vec<StructuralPredicate>) -> Option<()> {
    match lower_predicate(expr)? {
        StructuralPredicate::And(parts) => out.extend(parts.iter().cloned()),
        pred => out.push(pred),
    }
    Some(())
}

/// Return `Some(field_name)` when `expr` is a bare identifier or a
/// `@.field` chain step — the two forms that refer to a field of the current
/// object in a predicate. Returns `None` for anything more complex.
fn field_ref(expr: &Expr) -> Option<Arc<str>> {
    match expr {
        Expr::Ident(name) => Some(Arc::from(name.as_str())),
        Expr::Chain(base, steps) if matches!(base.as_ref(), Expr::Current) && steps.len() == 1 => {
            match &steps[0] {
                Step::Field(name) | Step::OptField(name) => Some(Arc::from(name.as_str())),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Execute a `DeepFind` plan: walk all object tokens under `anchor` and
/// collect those for which every predicate in `predicates` holds, materialising
/// matching subtrees from raw `bytes`.
fn run_deep_find(
    idx: &StructuralIndex,
    bytes: &[u8],
    anchor: &[StructuralPathStep],
    predicates: &[StructuralPredicate],
) -> Result<Val, EvalError> {
    if predicates.is_empty() {
        return Err(EvalError("find: requires at least one predicate".into()));
    }
    let Some(anchor) = anchor_token(idx, anchor) else {
        return Ok(Val::arr(Vec::new()));
    };
    let mut out = Vec::new();
    visit_predicate_candidates(idx, anchor, predicates, |object| {
        if predicates
            .iter()
            .all(|pred| predicate_matches(idx, bytes, object, pred))
        {
            out.push(materialize_token(idx, bytes, object)?);
        }
        Ok(())
    })?;
    Ok(Val::arr(out))
}

/// Execute a `DeepShape` plan: collect every descendant object under `anchor`
/// that has all listed `keys` as present fields, regardless of their values.
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

/// Execute a `DeepLike` plan: collect every descendant object under `anchor`
/// where every `(key, literal)` pattern matches the corresponding field value.
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

/// Find all object tokens under `anchor` that contain at least the first key
/// in `keys`, deduplicated by token id. Uses the index's key-name bitmap to
/// avoid scanning the entire subtree.
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

/// Enumerate candidate object tokens for predicate evaluation. When any
/// predicate references a specific field name, uses `visit_candidate_objects`
/// for a faster targeted scan; otherwise falls back to a full linear walk of
/// all object tokens in the anchor subtree.
fn visit_predicate_candidates<F>(
    idx: &StructuralIndex,
    anchor: TokenId,
    predicates: &[StructuralPredicate],
    mut visit: F,
) -> Result<(), EvalError>
where
    F: FnMut(TokenId) -> Result<(), EvalError>,
{
    if let Some(key) = first_candidate_key(predicates) {
        return visit_candidate_objects(idx, anchor, &[key], visit);
    }

    let close = idx
        .close_of(anchor)
        .map(|tok| tok.raw())
        .unwrap_or_else(|| idx.token_count().saturating_sub(1));
    for tok in idx
        .tokens()
        .skip(anchor.raw() as usize)
        .take_while(|tok| tok.raw() <= close)
    {
        if idx.kind(tok) == TokenKind::Object {
            visit(tok)?;
        }
    }
    Ok(())
}

/// Return the first field name referenced in a `FieldEqLiteral` predicate
/// (or nested inside an `And`) that can serve as a candidate-set seed key for
/// `visit_candidate_objects`. Returns `None` when no field reference is found.
fn first_candidate_key(predicates: &[StructuralPredicate]) -> Option<Arc<str>> {
    for pred in predicates {
        match pred {
            StructuralPredicate::FieldEqLiteral(key, _) => return Some(Arc::clone(key)),
            StructuralPredicate::And(parts) => {
                if let Some(key) = first_candidate_key(parts) {
                    return Some(key);
                }
            }
            StructuralPredicate::KindObject => {}
        }
    }
    None
}

/// Evaluate a single `StructuralPredicate` against a candidate `object` token,
/// comparing field byte spans directly without materialising a `Val`.
fn predicate_matches(
    idx: &StructuralIndex,
    bytes: &[u8],
    object: TokenId,
    pred: &StructuralPredicate,
) -> bool {
    match pred {
        StructuralPredicate::KindObject => idx.kind(object) == TokenKind::Object,
        StructuralPredicate::FieldEqLiteral(key, want) => idx
            .field_of(object, key)
            .map(|value| literal_matches(idx, bytes, value, want))
            .unwrap_or(false),
        StructuralPredicate::And(parts) => parts
            .iter()
            .all(|part| predicate_matches(idx, bytes, object, part)),
    }
}

/// Navigate the `StructuralIndex` from the document root following each step in
/// `steps`, returning the `TokenId` of the final token. Returns `None` when any
/// step fails (field missing, wrong token kind, or out-of-bounds index).
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

/// Return the `TokenId` of the child at position `index` inside an array token,
/// supporting negative indices (Python-style). Returns `None` when `array` is
/// not an array token or the index is out of range.
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

/// Compare a token's raw byte span against a `StructuralLiteral` without
/// deserialising. String comparison delegates to
/// `jetro_experimental::json_string_eq` to handle JSON escape sequences.
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

/// Deserialise the byte span of `tok` into a `Val` using either `simd-json`
/// (when the `simd-json` feature is enabled) or `serde_json`. Returns an error
/// when the span does not contain valid JSON.
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
