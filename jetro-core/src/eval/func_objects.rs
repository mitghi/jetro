//! Object methods: `keys`, `values`, `entries`, `pick`, `omit`,
//! `merge`, `deep_merge`, `rename`, `invert`, `transform_keys`,
//! `transform_values`, `filter_keys`, `filter_values`, `pivot`.
//!
//! All operate on `Val::Obj(Arc<IndexMap<Arc<str>, Val>>)`.  When
//! mutation is needed, the functions `Arc::try_unwrap` the inner map;
//! that is free when the caller holds the last refcount and one
//! `IndexMap::clone()` otherwise.  `IndexMap::shift_remove` is used
//! everywhere (never `remove`) to preserve insertion order — callers
//! depend on that for deterministic serialisation.

use std::sync::Arc;
use indexmap::IndexMap;

use crate::ast::{Arg, Expr};
use super::{Env, EvalError, apply_item, apply_item_mut, eval_pos};
use super::value::Val;
use super::util::{is_truthy, val_to_key, deep_merge};
use super::func_paths::{parse_path_segs, get_path_impl, PathSeg};

macro_rules! err {
    ($($t:tt)*) => { Err(EvalError(format!($($t)*))) };
}

// ── Field access ──────────────────────────────────────────────────────────────

pub fn keys(recv: Val) -> Result<Val, EvalError> {
    Ok(Val::arr(
        recv.as_object().map(|m| m.keys().map(|k| Val::Str(k.clone())).collect()).unwrap_or_default()
    ))
}

pub fn values(recv: Val) -> Result<Val, EvalError> {
    Ok(Val::arr(recv.as_object().map(|m| m.values().cloned().collect()).unwrap_or_default()))
}

pub fn entries(recv: Val) -> Result<Val, EvalError> {
    Ok(Val::arr(recv.as_object().map(|m| m.iter().map(|(k, v)| {
        Val::arr(vec![Val::Str(k.clone()), v.clone()])
    }).collect()).unwrap_or_default()))
}

// ── Field selection ───────────────────────────────────────────────────────────

/// Resolve one pick argument into `(out_key, src_key_or_path)`.
///
/// Accepted forms:
///   - `name`                → `("name", "name")`    (ident = field name)
///   - `"name"`              → `("name", "name")`    (string literal)
///   - `"a.b"`               → `("a",    "a.b")`     (dotted path, top-key as out key)
///   - `alias: name`         → `("alias","name")`    (named arg, ident rhs)
///   - `alias: "name"`       → `("alias","name")`    (named arg, string rhs)
fn pick_arg(a: &Arg, env: &Env) -> Result<Option<(Arc<str>, String)>, EvalError> {
    match a {
        Arg::Pos(Expr::Ident(s)) => Ok(Some((Arc::from(s.as_str()), s.clone()))),
        Arg::Pos(e) => {
            let v = super::eval(e, env)?;
            match v {
                Val::Str(s) => {
                    let top: Arc<str> = if s.contains('.') || s.contains('[') {
                        match parse_path_segs(&s).first() {
                            Some(PathSeg::Field(f)) => Arc::from(f.as_str()),
                            Some(PathSeg::Index(i)) => Arc::from(i.to_string().as_str()),
                            None => Arc::from(s.as_ref()),
                        }
                    } else { s.clone() };
                    Ok(Some((top, s.to_string())))
                }
                _ => Ok(None),
            }
        }
        Arg::Named(alias, Expr::Ident(src)) => {
            Ok(Some((Arc::from(alias.as_str()), src.clone())))
        }
        Arg::Named(alias, e) => {
            let v = super::eval(e, env)?;
            match v {
                Val::Str(s) => Ok(Some((Arc::from(alias.as_str()), s.to_string()))),
                _ => Ok(None),
            }
        }
    }
}

enum PickKind {
    Field,
    Path(Vec<PathSeg>),
}

fn pick_one(
    obj: &IndexMap<Arc<str>, Val>,
    resolved: &[(Arc<str>, String, PickKind)],
    wrapped: &Val,
) -> IndexMap<Arc<str>, Val> {
    let mut out = IndexMap::with_capacity(resolved.len());
    for (out_key, src, kind) in resolved {
        match kind {
            PickKind::Field => {
                if let Some(v) = obj.get(src.as_str()) {
                    out.insert(out_key.clone(), v.clone());
                }
            }
            PickKind::Path(segs) => {
                let v = get_path_impl(wrapped, segs);
                if !v.is_null() { out.insert(out_key.clone(), v); }
            }
        }
    }
    out
}

pub fn pick(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let mut resolved: Vec<(Arc<str>, String, PickKind)> = Vec::with_capacity(args.len());
    for a in args {
        if let Some((k, s)) = pick_arg(a, env)? {
            let kind = if s.contains('.') || s.contains('[') {
                PickKind::Path(parse_path_segs(&s))
            } else {
                PickKind::Field
            };
            resolved.push((k, s, kind));
        }
    }
    let has_path = resolved.iter().any(|(_, _, k)| matches!(k, PickKind::Path(_)));
    match recv {
        Val::Obj(m) => {
            let wrapped = if has_path { Val::Obj(m.clone()) } else { Val::Null };
            Ok(Val::obj(pick_one(&m, &resolved, &wrapped)))
        }
        Val::Arr(a) => {
            let out: Vec<Val> = a.iter().map(|el| match el {
                Val::Obj(m) => {
                    let wrapped = if has_path { Val::Obj(m.clone()) } else { Val::Null };
                    Val::obj(pick_one(m, &resolved, &wrapped))
                }
                other => other.clone(),
            }).collect();
            Ok(Val::arr(out))
        }
        _ => err!("pick: expected object or array of objects"),
    }
}

pub fn omit(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let keys: Vec<String> = args.iter()
        .filter_map(|a| eval_pos(a, env).ok())
        .filter_map(|v| if let Val::Str(s) = v { Some(s.to_string()) } else { None })
        .collect();
    let mut map = recv.into_map().ok_or_else(|| EvalError("omit: expected object".into()))?;
    for k in &keys { map.shift_remove(k.as_str()); }
    Ok(Val::obj(map))
}

// ── Merge / defaults ──────────────────────────────────────────────────────────

pub fn merge(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let other = args.first().map(|a| eval_pos(a, env)).transpose()?.unwrap_or(Val::Null);
    match (recv.into_map(), other.into_map()) {
        (Some(mut base), Some(other)) => {
            for (k, v) in other { base.insert(k, v); }
            Ok(Val::obj(base))
        }
        _ => err!("merge: expected two objects"),
    }
}

pub fn deep_merge_method(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let other = args.first().map(|a| eval_pos(a, env)).transpose()?.unwrap_or(Val::Null);
    Ok(deep_merge(recv, other))
}

pub fn defaults(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let other = args.first().map(|a| eval_pos(a, env)).transpose()?.unwrap_or(Val::Null);
    match (recv.into_map(), other.into_map()) {
        (Some(mut base), Some(defs)) => {
            for (k, v) in defs {
                let entry = base.entry(k).or_insert(Val::Null);
                if entry.is_null() { *entry = v; }
            }
            Ok(Val::obj(base))
        }
        _ => err!("defaults: expected two objects"),
    }
}

// ── Rename / invert ───────────────────────────────────────────────────────────

pub fn rename(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let renames = args.first().map(|a| eval_pos(a, env)).transpose()?.unwrap_or(Val::Null);
    match (recv.into_map(), renames.into_map()) {
        (Some(mut obj), Some(renames)) => {
            for (old, new_val) in renames {
                if let Some(v) = obj.shift_remove(old.as_ref()) {
                    let new_key: Arc<str> = if let Val::Str(s) = &new_val {
                        s.clone()
                    } else { old.clone() };
                    obj.insert(new_key, v);
                }
            }
            Ok(Val::obj(obj))
        }
        _ => err!("rename: expected object and rename map"),
    }
}

pub fn invert(recv: Val) -> Result<Val, EvalError> {
    let map = recv.into_map().ok_or_else(|| EvalError("invert: expected object".into()))?;
    let out: IndexMap<Arc<str>, Val> = map.into_iter()
        .map(|(k, v)| {
            let nk = match v {
                Val::Str(s) => s,
                other       => Arc::<str>::from(val_to_key(&other)),
            };
            (nk, Val::Str(k))
        })
        .collect();
    Ok(Val::obj(out))
}

// ── Transform ─────────────────────────────────────────────────────────────────

pub fn transform_keys(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let lam = args.first().ok_or_else(|| EvalError("transform_keys: requires lambda".into()))?;
    let map = recv.into_map().ok_or_else(|| EvalError("transform_keys: expected object".into()))?;
    let mut env_mut = env.clone();
    let mut out = IndexMap::with_capacity(map.len());
    for (k, v) in map {
        let new_key: Arc<str> = match apply_item_mut(Val::Str(k), lam, &mut env_mut)? {
            Val::Str(s) => s,
            other       => Arc::<str>::from(val_to_key(&other)),
        };
        out.insert(new_key, v);
    }
    Ok(Val::obj(out))
}

pub fn transform_values(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let lam = args.first().ok_or_else(|| EvalError("transform_values: requires lambda".into()))?;
    let map = recv.into_map().ok_or_else(|| EvalError("transform_values: expected object".into()))?;
    let mut env_mut = env.clone();
    let mut out = IndexMap::with_capacity(map.len());
    for (k, v) in map { out.insert(k, apply_item_mut(v, lam, &mut env_mut)?); }
    Ok(Val::obj(out))
}

pub fn filter_keys(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let lam = args.first().ok_or_else(|| EvalError("filter_keys: requires predicate".into()))?;
    let map = recv.into_map().ok_or_else(|| EvalError("filter_keys: expected object".into()))?;
    let mut env_mut = env.clone();
    let mut out = IndexMap::with_capacity(map.len());
    for (k, v) in map {
        if is_truthy(&apply_item_mut(Val::Str(k.clone()), lam, &mut env_mut)?) { out.insert(k, v); }
    }
    Ok(Val::obj(out))
}

pub fn filter_values(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let lam = args.first().ok_or_else(|| EvalError("filter_values: requires predicate".into()))?;
    let map = recv.into_map().ok_or_else(|| EvalError("filter_values: expected object".into()))?;
    let mut env_mut = env.clone();
    let mut out = IndexMap::with_capacity(map.len());
    for (k, v) in map {
        if is_truthy(&apply_item_mut(v.clone(), lam, &mut env_mut)?) { out.insert(k, v); }
    }
    Ok(Val::obj(out))
}

// ── Pairs / pivot ─────────────────────────────────────────────────────────────

pub fn to_pairs(recv: Val) -> Result<Val, EvalError> {
    use super::util::obj2;
    Ok(Val::arr(recv.as_object().map(|m| m.iter().map(|(k, v)| {
        obj2("key", Val::Str(k.clone()), "val", v.clone())
    }).collect()).unwrap_or_default()))
}

pub fn from_pairs(recv: Val) -> Result<Val, EvalError> {
    let items = recv.into_vec().ok_or_else(|| EvalError("from_pairs: expected array".into()))?;
    let mut map = IndexMap::with_capacity(items.len());
    for item in items {
        let k_val = item.get("key").or_else(|| item.get("k")).cloned().unwrap_or(Val::Null);
        let v     = item.get("val").or_else(|| item.get("value")).or_else(|| item.get("v"))
                        .cloned().unwrap_or(Val::Null);
        if let Val::Str(k) = k_val { map.insert(k, v); }
    }
    Ok(Val::obj(map))
}

fn pivot_field(item: &Val, arg: &Arg, env: &Env) -> Result<Val, EvalError> {
    match arg {
        Arg::Pos(Expr::Str(s)) | Arg::Named(_, Expr::Str(s)) => Ok(item.get_field(s.as_str())),
        _ => apply_item(item.clone(), arg, env),
    }
}

pub fn pivot(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let items = recv.into_vec().ok_or_else(|| EvalError("pivot: expected array".into()))?;
    #[inline]
    fn to_arc(v: Val) -> Arc<str> {
        match v { Val::Str(s) => s, other => Arc::<str>::from(val_to_key(&other)) }
    }
    if args.len() >= 3 {
        // 3-arg: pivot(row_field, col_field, val_field) → {row: {col: val}}
        let mut map: IndexMap<Arc<str>, IndexMap<Arc<str>, Val>> = IndexMap::new();
        for item in &items {
            let row = to_arc(pivot_field(item, &args[0], env)?);
            let col = to_arc(pivot_field(item, &args[1], env)?);
            let v   = pivot_field(item, &args[2], env)?;
            map.entry(row).or_insert_with(IndexMap::new).insert(col, v);
        }
        let out: IndexMap<Arc<str>, Val> = map.into_iter()
            .map(|(k, inner)| (k, Val::obj(inner)))
            .collect();
        return Ok(Val::obj(out));
    }
    let key_arg = args.first().ok_or_else(|| EvalError("pivot: requires key arg".into()))?;
    let val_arg = args.get(1).ok_or_else(|| EvalError("pivot: requires value arg".into()))?;
    let mut map = IndexMap::with_capacity(items.len());
    for item in &items {
        let k = to_arc(pivot_field(item, key_arg, env)?);
        let v = pivot_field(item, val_arg, env)?;
        map.insert(k, v);
    }
    Ok(Val::obj(map))
}

// ── fanout / zip_shape (Tier C lite) ─────────────────────────────────────────
//
// `fanout(e1, e2, …)` — evaluate each arg with `@` = recv, return the
// results as an array.  Tuple-on-one-value.
//
// `zip_shape(name=expr, …)` — evaluate each named arg with `@` = recv,
// collect as an ordered object.  Bare idents are shorthand for
// `name=name` (like `.pick` but with evaluated exprs rather than field
// lookups).  Names are *not* visible to later exprs (keeps the hot path
// simple; order is preserved for serialisation only).

pub fn fanout(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    if args.is_empty() { return err!("fanout: requires at least one expression"); }
    let mut env_mut = env.clone();
    let mut out = Vec::with_capacity(args.len());
    for a in args {
        out.push(apply_item_mut(recv.clone(), a, &mut env_mut)?);
    }
    Ok(Val::arr(out))
}

pub fn zip_shape(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    if args.is_empty() { return err!("zip_shape: requires at least one field"); }
    let mut env_mut = env.clone();
    let mut out: IndexMap<Arc<str>, Val> = IndexMap::with_capacity(args.len());
    for a in args {
        let name: Arc<str> = match a {
            Arg::Named(n, _)           => Arc::from(n.as_str()),
            Arg::Pos(Expr::Ident(n))   => Arc::from(n.as_str()),
            _ => return err!("zip_shape: args must be `name = expr` or bare identifier"),
        };
        let v = apply_item_mut(recv.clone(), a, &mut env_mut)?;
        out.insert(name, v);
    }
    Ok(Val::obj(out))
}

// ── schema (Tier A) ───────────────────────────────────────────────────────────
//
// `.schema()` — walk a value and return a schema descriptor.
//
// Result shape:
//   scalar   → { type: "String"|"Int"|"Float"|"Bool"|"Null" }
//   array    → { type: "Array",  len: N, items: <unified-schema-of-items> }
//   object   → { type: "Object", required: [keys...], fields: { k: <schema>... } }
//
// For heterogeneous arrays the item schema is unified via `unify_schema`:
//   - differing scalar types collapse to `type: "Mixed"`
//   - fields present in some but not all objects get `optional: true`
//   - fields sometimes-null get `nullable: true`

pub fn schema(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    Ok(schema_of(&recv))
}

fn schema_of(v: &Val) -> Val {
    match v {
        Val::Null       => ty_obj("Null"),
        Val::Bool(_)    => ty_obj("Bool"),
        Val::Int(_)     => ty_obj("Int"),
        Val::Float(_)   => ty_obj("Float"),
        Val::Str(_)     => ty_obj("String"),
        Val::IntVec(a)  => array_schema(a.len(), ty_obj("Int")),
        Val::FloatVec(a)=> array_schema(a.len(), ty_obj("Float")),
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
        Val::Obj(m) => {
            let mut required: Vec<Val> = Vec::with_capacity(m.len());
            let mut fields: IndexMap<Arc<str>, Val> = IndexMap::with_capacity(m.len());
            for (k, child) in m.iter() {
                let mut field = schema_of(child);
                if matches!(child, Val::Null) {
                    field = set_field(field, "nullable", Val::Bool(true));
                } else {
                    required.push(Val::Str(k.clone()));
                }
                fields.insert(k.clone(), field);
            }
            let mut out: IndexMap<Arc<str>, Val> = IndexMap::with_capacity(3);
            out.insert(Arc::from("type"), Val::Str(Arc::from("Object")));
            out.insert(Arc::from("required"), Val::arr(required));
            out.insert(Arc::from("fields"), Val::obj(fields));
            Val::obj(out)
        }
    }
}

fn ty_obj(name: &str) -> Val {
    let mut m: IndexMap<Arc<str>, Val> = IndexMap::with_capacity(1);
    m.insert(Arc::from("type"), Val::Str(Arc::from(name)));
    Val::obj(m)
}

fn array_schema(len: usize, items: Val) -> Val {
    let mut m: IndexMap<Arc<str>, Val> = IndexMap::with_capacity(3);
    m.insert(Arc::from("type"), Val::Str(Arc::from("Array")));
    m.insert(Arc::from("len"), Val::Int(len as i64));
    m.insert(Arc::from("items"), items);
    Val::obj(m)
}

fn set_field(obj: Val, key: &str, v: Val) -> Val {
    if let Val::Obj(m) = obj {
        let mut m = Arc::try_unwrap(m).unwrap_or_else(|arc| (*arc).clone());
        m.insert(Arc::from(key), v);
        Val::obj(m)
    } else {
        obj
    }
}

fn get_type(v: &Val) -> Option<&str> {
    if let Val::Obj(m) = v {
        if let Some(Val::Str(s)) = m.get("type") { return Some(s.as_ref()); }
    }
    None
}

fn unify_schema(a: Val, b: Val) -> Val {
    let (ta, tb) = (get_type(&a), get_type(&b));
    match (ta, tb) {
        (Some(x), Some(y)) if x == y => {
            match x {
                "Object" => unify_object_schemas(a, b),
                "Array" => unify_array_schemas(a, b),
                _ => mark_nullable_if_either(a, b),
            }
        }
        (Some("Null"), _) => set_field(b, "nullable", Val::Bool(true)),
        (_, Some("Null")) => set_field(a, "nullable", Val::Bool(true)),
        _ => ty_obj("Mixed"),
    }
}

fn mark_nullable_if_either(a: Val, b: Val) -> Val {
    let a_null = is_nullable(&a);
    let b_null = is_nullable(&b);
    if a_null || b_null { set_field(a, "nullable", Val::Bool(true)) } else { a }
}

fn is_nullable(v: &Val) -> bool {
    if let Val::Obj(m) = v {
        matches!(m.get("nullable"), Some(Val::Bool(true)))
    } else { false }
}

fn unify_array_schemas(a: Val, b: Val) -> Val {
    let a_items = extract_field(&a, "items");
    let b_items = extract_field(&b, "items");
    let items = match (a_items, b_items) {
        (Some(x), Some(y)) => unify_schema(x, y),
        (Some(x), None)    => x,
        (None,    Some(y)) => y,
        (None,    None)    => ty_obj("Unknown"),
    };
    let la = extract_int(&a, "len").unwrap_or(0);
    let lb = extract_int(&b, "len").unwrap_or(0);
    array_schema((la + lb) as usize, items)
}

fn extract_field(v: &Val, key: &str) -> Option<Val> {
    if let Val::Obj(m) = v { m.get(key).cloned() } else { None }
}

fn extract_int(v: &Val, key: &str) -> Option<i64> {
    if let Some(Val::Int(n)) = extract_field(v, key) { Some(n) } else { None }
}

fn unify_object_schemas(a: Val, b: Val) -> Val {
    let a_fields = extract_field(&a, "fields");
    let b_fields = extract_field(&b, "fields");
    let (a_map, b_map) = match (a_fields, b_fields) {
        (Some(Val::Obj(x)), Some(Val::Obj(y))) => (
            Arc::try_unwrap(x).unwrap_or_else(|arc| (*arc).clone()),
            Arc::try_unwrap(y).unwrap_or_else(|arc| (*arc).clone()),
        ),
        _ => return ty_obj("Object"),
    };
    let a_req = extract_required_set(&a);
    let b_req = extract_required_set(&b);

    let mut out_fields: IndexMap<Arc<str>, Val> = IndexMap::with_capacity(a_map.len().max(b_map.len()));
    let mut all_keys: Vec<Arc<str>> = Vec::with_capacity(a_map.len() + b_map.len());
    for (k, _) in &a_map { all_keys.push(k.clone()); }
    for (k, _) in &b_map { if !a_map.contains_key(k) { all_keys.push(k.clone()); } }

    let mut required: Vec<Val> = Vec::new();
    for k in all_keys {
        let av = a_map.get(&k).cloned();
        let bv = b_map.get(&k).cloned();
        let field = match (av, bv) {
            (Some(x), Some(y)) => unify_schema(x, y),
            (Some(x), None)    => set_field(x, "optional", Val::Bool(true)),
            (None,    Some(y)) => set_field(y, "optional", Val::Bool(true)),
            _                  => ty_obj("Unknown"),
        };
        let both_required = a_req.contains(k.as_ref()) && b_req.contains(k.as_ref());
        if both_required { required.push(Val::Str(k.clone())); }
        out_fields.insert(k, field);
    }

    let mut out: IndexMap<Arc<str>, Val> = IndexMap::with_capacity(3);
    out.insert(Arc::from("type"), Val::Str(Arc::from("Object")));
    out.insert(Arc::from("required"), Val::arr(required));
    out.insert(Arc::from("fields"), Val::obj(out_fields));
    Val::obj(out)
}

fn extract_required_set(v: &Val) -> std::collections::HashSet<String> {
    let mut s = std::collections::HashSet::new();
    if let Some(Val::Arr(a)) = extract_field(v, "required") {
        for el in a.iter() {
            if let Val::Str(k) = el { s.insert(k.to_string()); }
        }
    }
    s
}
