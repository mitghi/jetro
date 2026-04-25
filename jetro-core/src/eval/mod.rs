//! Tree-walking evaluator — reference semantics for Jetro v2.
//!
//! This module is the *source of truth*: when the optimiser or VM
//! behaviour diverges from it, the VM is wrong.  It is also the
//! smallest path from AST to value and thus the easiest to reason
//! about.  [`vm`](super::vm) is a faster path that caches compiled
//! programs and resolved pointers, but every new feature lands here
//! first.
//!
//! # Data flow
//!
//! ```text
//! evaluate(expr, doc)
//!     │
//!     ▼
//!   apply_expr(&Expr, &Env)
//!     │
//!     ├── literals / operators    (inline)
//!     ├── chain navigation        (apply_chain → apply_step)
//!     ├── method calls            (dispatch_method → func_*/methods.rs)
//!     └── comprehensions / let    (recursive into new Env)
//! ```
//!
//! # `Env`
//!
//! `Env` owns the root doc, the "current" value (`@`), and a
//! `SmallVec<[(Arc<str>, Val); 4]>` of let-bound names plus an
//! `Arc<MethodRegistry>` for user-registered methods.  Scopes are
//! pushed by appending and popped by truncation — lookup is linear
//! but `SmallVec` keeps the first four slots inline, which covers
//! every realistic query.
//!
//! # Registry
//!
//! [`MethodRegistry`] holds user methods behind `Arc<dyn Method>`
//! and is itself `Clone` (via derive) so threading it through
//! recursive calls is free.

use std::sync::Arc;
use indexmap::IndexMap;
use smallvec::SmallVec;

use super::ast::*;

pub mod value;
pub mod util;
pub mod methods;
pub mod builtins;
mod func_strings;
mod func_arrays;
mod func_objects;
mod func_paths;
mod func_aggregates;
mod func_csv;
mod func_search;

pub use value::Val;
pub use methods::{Method, MethodRegistry};
use util::*;

// ── Error ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct EvalError(pub String);

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "eval error: {}", self.0)
    }
}
impl std::error::Error for EvalError {}

macro_rules! err {
    ($($t:tt)*) => { Err(EvalError(format!($($t)*))) };
}

// ── Environment ───────────────────────────────────────────────────────────────
// SmallVec<4> keeps ≤4 bindings on the stack — covers the vast majority of
// queries without a heap allocation.  Linear scan is fine for n ≤ 4.

/// Saved-state token returned by `Env::push_lam` and consumed by
/// `Env::pop_lam`.  Lets hot loops bind a single lambda slot + swap
/// `current` without cloning the whole Env per iteration.
pub struct LamFrame {
    prev_current: Val,
    prev_var:     LamVarPrev,
}

enum LamVarPrev {
    None,
    Pushed,
    Replaced(usize, Val),
}

#[derive(Clone)]
pub struct Env {
    vars:     SmallVec<[(Arc<str>, Val); 4]>,
    pub root:    Val,
    pub current: Val,
    registry: Arc<MethodRegistry>,
    /// Raw JSON bytes the `root` was parsed from, when available.  Enables
    /// SIMD byte-scan fast paths for `$..key` and `$..find(key op lit)`
    /// queries that start at the document root.
    pub(crate) raw_bytes: Option<Arc<[u8]>>,
}

impl Env {
    fn new(root: Val) -> Self {
        Self {
            vars: SmallVec::new(),
            root: root.clone(),
            current: root,
            registry: Arc::new(MethodRegistry::new()),
            raw_bytes: None,
        }
    }

    pub fn new_with_registry(root: Val, registry: Arc<MethodRegistry>) -> Self {
        Self { vars: SmallVec::new(), root: root.clone(), current: root, registry, raw_bytes: None }
    }

    /// Build an `Env` that carries the original JSON source bytes so that
    /// SIMD byte-scan can short-circuit deep-descendant queries.
    pub fn new_with_raw(
        root: Val,
        registry: Arc<MethodRegistry>,
        raw_bytes: Arc<[u8]>,
    ) -> Self {
        Self {
            vars: SmallVec::new(),
            root: root.clone(),
            current: root,
            registry,
            raw_bytes: Some(raw_bytes),
        }
    }

    #[inline]
    pub(super) fn registry_ref(&self) -> &MethodRegistry { &self.registry }

    #[inline]
    pub fn with_current(&self, current: Val) -> Self {
        Self {
            vars: self.vars.clone(),
            root: self.root.clone(),
            current,
            registry: self.registry.clone(),
            raw_bytes: self.raw_bytes.clone(),
        }
    }

    /// In-place swap of `current` — returns previous.  Paired with
    /// `restore_current` this lets hot loops avoid cloning `vars`/`root`
    /// per iteration.
    #[inline]
    pub fn swap_current(&mut self, new: Val) -> Val {
        std::mem::replace(&mut self.current, new)
    }

    #[inline]
    pub fn restore_current(&mut self, old: Val) {
        self.current = old;
    }

    #[inline]
    pub fn get_var(&self, name: &str) -> Option<&Val> {
        self.vars.iter().rev().find(|(k, _)| k.as_ref() == name).map(|(_, v)| v)
    }

    #[inline]
    pub fn has_var(&self, name: &str) -> bool {
        self.vars.iter().any(|(k, _)| k.as_ref() == name)
    }

    pub fn with_var(&self, name: &str, val: Val) -> Self {
        let mut vars = self.vars.clone();
        if let Some(pos) = vars.iter().position(|(k, _)| k.as_ref() == name) {
            vars[pos].1 = val;
        } else {
            vars.push((Arc::from(name), val));
        }
        Self { vars, root: self.root.clone(), current: self.current.clone(), registry: self.registry.clone(), raw_bytes: self.raw_bytes.clone() }
    }

    /// Hot-loop helper: bind `name → val` and swap `current`, returning
    /// the previous state.  If `name` was already bound we remember the
    /// previous value; otherwise we remember that the slot was freshly
    /// pushed so `pop_lam` can truncate it off again.
    #[inline]
    pub fn push_lam(&mut self, name: Option<&str>, val: Val) -> LamFrame {
        let prev_current = std::mem::replace(&mut self.current, val.clone());
        let prev_var = match name {
            None => LamVarPrev::None,
            Some(n) => {
                if let Some(pos) = self.vars.iter().position(|(k, _)| k.as_ref() == n) {
                    let prev = std::mem::replace(&mut self.vars[pos].1, val);
                    LamVarPrev::Replaced(pos, prev)
                } else {
                    self.vars.push((Arc::from(n), val));
                    LamVarPrev::Pushed
                }
            }
        };
        LamFrame { prev_current, prev_var }
    }

    #[inline]
    pub fn pop_lam(&mut self, frame: LamFrame) {
        self.current = frame.prev_current;
        match frame.prev_var {
            LamVarPrev::None => {}
            LamVarPrev::Pushed => { self.vars.pop(); }
            LamVarPrev::Replaced(pos, prev) => { self.vars[pos].1 = prev; }
        }
    }

}

// ── Public entry points ───────────────────────────────────────────────────────

pub fn evaluate(expr: &Expr, root: &serde_json::Value) -> Result<serde_json::Value, EvalError> {
    let val = Val::from(root);
    Ok(eval(expr, &Env::new(val))?.into())
}

pub fn evaluate_with(
    expr: &Expr,
    root: &serde_json::Value,
    registry: Arc<MethodRegistry>,
) -> Result<serde_json::Value, EvalError> {
    let val = Val::from(root);
    Ok(eval(expr, &Env::new_with_registry(val, registry))?.into())
}

/// Evaluate `expr` with the original JSON source bytes retained.  Enables
/// SIMD byte-scan fast paths for `$..key` descent queries.
pub fn evaluate_with_raw(
    expr: &Expr,
    root: &serde_json::Value,
    registry: Arc<MethodRegistry>,
    raw_bytes: Arc<[u8]>,
) -> Result<serde_json::Value, EvalError> {
    let val = Val::from(root);
    Ok(eval(expr, &Env::new_with_raw(val, registry, raw_bytes))?.into())
}

// ── Core evaluator ────────────────────────────────────────────────────────────

pub(super) fn eval(expr: &Expr, env: &Env) -> Result<Val, EvalError> {
    match expr {
        Expr::Null      => Ok(Val::Null),
        Expr::Bool(b)   => Ok(Val::Bool(*b)),
        Expr::Int(n)    => Ok(Val::Int(*n)),
        Expr::Float(f)  => Ok(Val::Float(*f)),
        Expr::Str(s)    => Ok(Val::Str(Arc::from(s.as_str()))),

        Expr::FString(parts) => eval_fstring(parts, env),

        Expr::Root    => Ok(env.root.clone()),
        Expr::Current => Ok(env.current.clone()),

        Expr::Ident(name) => {
            if let Some(v) = env.get_var(name) { return Ok(v.clone()); }
            Ok(env.current.get_field(name))
        }

        Expr::Chain(base, steps) => {
            // SIMD enclosing-object fast path: `$..find(@.k == lit)` with raw
            // bytes — scan locates each object whose `k` field equals `lit`
            // without walking the tree.
            if let (Expr::Root, Some(Step::Method(name, args)), Some(bytes))
                = (&**base, steps.first(), env.raw_bytes.as_ref())
            {
                if name == "deep_find" && !args.is_empty() {
                    if let Some(conjuncts) = canonical_field_eq_literals(args) {
                        let spans = if conjuncts.len() == 1 {
                            super::scan::find_enclosing_objects_eq(
                                bytes, &conjuncts[0].0, &conjuncts[0].1,
                            )
                        } else {
                            super::scan::find_enclosing_objects_eq_multi(
                                bytes, &conjuncts,
                            )
                        };
                        let mut vals: Vec<Val> = Vec::with_capacity(spans.len());
                        for s in &spans {
                            if let Ok(v) = serde_json::from_slice::<serde_json::Value>(
                                &bytes[s.start..s.end]
                            ) {
                                vals.push(Val::from(&v));
                            }
                        }
                        let mut val = Val::arr(vals);
                        for step in &steps[1..] { val = eval_step(val, step, env)?; }
                        return Ok(val);
                    }
                    // Single-conjunct numeric range: `$..find(@.k op num)`
                    // where `op` ∈ `<`, `<=`, `>`, `>=`.  Extends the byte-scan
                    // fast path past pure equality literals.
                    if args.len() == 1 {
                        let e = match &args[0] { Arg::Pos(e) | Arg::Named(_, e) => e };
                        if let Some((field, op, thresh)) = canonical_field_cmp_literal(e) {
                            let spans = super::scan::find_enclosing_objects_cmp(
                                bytes, &field, op, thresh,
                            );
                            let mut vals: Vec<Val> = Vec::with_capacity(spans.len());
                            for s in &spans {
                                if let Ok(v) = serde_json::from_slice::<serde_json::Value>(
                                    &bytes[s.start..s.end]
                                ) {
                                    vals.push(Val::from(&v));
                                }
                            }
                            let mut val = Val::arr(vals);
                            for step in &steps[1..] { val = eval_step(val, step, env)?; }
                            return Ok(val);
                        }
                    }
                    // Mixed multi-conjunct: eq and numeric-range predicates
                    // together, e.g. `$..find(@.status == "shipped", @.total > 500)`.
                    // Pure-eq and single-cmp already handled above; this path
                    // picks up any remaining combination.
                    if let Some(conjuncts) = canonical_field_mixed_predicates(args) {
                        let spans = super::scan::find_enclosing_objects_mixed(
                            bytes, &conjuncts,
                        );
                        let mut vals: Vec<Val> = Vec::with_capacity(spans.len());
                        for s in &spans {
                            if let Ok(v) = serde_json::from_slice::<serde_json::Value>(
                                &bytes[s.start..s.end]
                            ) {
                                vals.push(Val::from(&v));
                            }
                        }
                        let mut val = Val::arr(vals);
                        for step in &steps[1..] { val = eval_step(val, step, env)?; }
                        return Ok(val);
                    }
                }
            }
            // SIMD byte-chain fast path: `$..key<rest>` with raw bytes —
            // chains of descendant/quantifier/filter-eq stay as byte spans
            // and never materialise intermediate Vals.
            if let (Expr::Root, Some(Step::Descendant(name)), Some(bytes))
                = (&**base, steps.first(), env.raw_bytes.as_ref())
            {
                let (mut val, consumed) = byte_chain_eval(bytes, name, steps);
                for step in &steps[consumed..] { val = eval_step(val, step, env)?; }
                return Ok(val);
            }
            let mut val = eval(base, env)?;
            for step in steps { val = eval_step(val, step, env)?; }
            Ok(val)
        }

        Expr::UnaryNeg(e) => match eval(e, env)? {
            Val::Int(n)   => Ok(Val::Int(-n)),
            Val::Float(f) => Ok(Val::Float(-f)),
            _ => err!("unary minus requires a number"),
        },

        Expr::Not(e)  => Ok(Val::Bool(!is_truthy(&eval(e, env)?))),
        Expr::BinOp(l, op, r) => eval_binop(l, *op, r, env),

        Expr::Coalesce(lhs, rhs) => {
            let v = eval(lhs, env)?;
            if !v.is_null() { Ok(v) } else { eval(rhs, env) }
        }

        Expr::Kind { expr, ty, negate } => {
            let v = eval(expr, env)?;
            let m = kind_matches(&v, *ty);
            Ok(Val::Bool(if *negate { !m } else { m }))
        }

        Expr::Object(fields) => eval_object(fields, env),

        Expr::Array(elems) => {
            let mut out = Vec::new();
            for elem in elems {
                match elem {
                    ArrayElem::Expr(e)   => out.push(eval(e, env)?),
                    ArrayElem::Spread(e) => match eval(e, env)? {
                        Val::Arr(a) => {
                            let items = Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone());
                            out.extend(items);
                        }
                        Val::IntVec(a) => out.extend(a.iter().map(|n| Val::Int(*n))),
                        Val::FloatVec(a) => out.extend(a.iter().map(|f| Val::Float(*f))),
                        v => out.push(v),
                    },
                }
            }
            Ok(Val::arr(out))
        }

        Expr::Pipeline { base, steps } => eval_pipeline(base, steps, env),

        Expr::ListComp { expr, vars, iter, cond } => {
            let items = eval_iter(iter, env)?;
            let mut out = Vec::new();
            let mut env_mut = env.clone();
            for item in items {
                let frames = bind_vars_mut(&mut env_mut, vars, item);
                let keep = match cond {
                    Some(c) => match eval(c, &env_mut) {
                        Ok(v)  => is_truthy(&v),
                        Err(e) => { unbind_vars_mut(&mut env_mut, frames); return Err(e); }
                    },
                    None => true,
                };
                if keep {
                    match eval(expr, &env_mut) {
                        Ok(v)  => out.push(v),
                        Err(e) => { unbind_vars_mut(&mut env_mut, frames); return Err(e); }
                    }
                }
                unbind_vars_mut(&mut env_mut, frames);
            }
            Ok(Val::arr(out))
        }

        Expr::DictComp { key, val, vars, iter, cond } => {
            let items = eval_iter(iter, env)?;
            let mut map: IndexMap<Arc<str>, Val> = IndexMap::with_capacity(items.len());
            let mut env_mut = env.clone();
            for item in items {
                let frames = bind_vars_mut(&mut env_mut, vars, item);
                let keep = match cond {
                    Some(c) => match eval(c, &env_mut) {
                        Ok(v)  => is_truthy(&v),
                        Err(e) => { unbind_vars_mut(&mut env_mut, frames); return Err(e); }
                    },
                    None => true,
                };
                if keep {
                    let k: Arc<str> = match eval(key, &env_mut) {
                        Ok(Val::Str(s)) => s,
                        Ok(other)       => Arc::<str>::from(val_to_key(&other)),
                        Err(e) => { unbind_vars_mut(&mut env_mut, frames); return Err(e); }
                    };
                    match eval(val, &env_mut) {
                        Ok(v)  => { map.insert(k, v); }
                        Err(e) => { unbind_vars_mut(&mut env_mut, frames); return Err(e); }
                    }
                }
                unbind_vars_mut(&mut env_mut, frames);
            }
            Ok(Val::obj(map))
        }

        Expr::SetComp { expr, vars, iter, cond } | Expr::GenComp { expr, vars, iter, cond } => {
            let items = eval_iter(iter, env)?;
            let mut seen: std::collections::HashSet<String> =
                std::collections::HashSet::with_capacity(items.len());
            let mut out = Vec::with_capacity(items.len());
            let mut env_mut = env.clone();
            for item in items {
                let frames = bind_vars_mut(&mut env_mut, vars, item);
                let keep = match cond {
                    Some(c) => match eval(c, &env_mut) {
                        Ok(v)  => is_truthy(&v),
                        Err(e) => { unbind_vars_mut(&mut env_mut, frames); return Err(e); }
                    },
                    None => true,
                };
                if keep {
                    match eval(expr, &env_mut) {
                        Ok(v)  => if seen.insert(val_to_key(&v)) { out.push(v); },
                        Err(e) => { unbind_vars_mut(&mut env_mut, frames); return Err(e); }
                    }
                }
                unbind_vars_mut(&mut env_mut, frames);
            }
            Ok(Val::arr(out))
        }

        Expr::Lambda { .. } => err!("lambda cannot be used as standalone value"),

        Expr::Let { name, init, body } => {
            let v = eval(init, env)?;
            eval(body, &env.with_var(name, v))
        }

        Expr::IfElse { cond, then_, else_ } => {
            if is_truthy(&eval(cond, env)?) { eval(then_, env) } else { eval(else_, env) }
        }

        Expr::Try { body, default } => {
            // Catches both EvalError AND Val::Null.  Panics propagate.
            match eval(body, env) {
                Ok(v) if !v.is_null() => Ok(v),
                Ok(_) | Err(_)        => eval(default, env),
            }
        }

        Expr::GlobalCall { name, args } => eval_global(name, args, env),

        Expr::Cast { expr, ty } => {
            let v = eval(expr, env)?;
            cast_val(&v, *ty)
        }

        Expr::Patch { root, ops } => eval_patch(root, ops, env),
        Expr::DeleteMark =>
            err!("DELETE can only appear as a patch-field value"),
    }
}

// ── Patch ─────────────────────────────────────────────────────────────────────

use super::ast::{PatchOp, PathStep};

enum PatchResult { Replace(Val), Delete }

fn eval_patch(root: &Expr, ops: &[PatchOp], env: &Env) -> Result<Val, EvalError> {
    let mut doc = eval(root, env)?;
    for op in ops {
        if let Some(c) = &op.cond {
            let cenv = env.with_current(doc.clone());
            if !is_truthy(&eval(c, &cenv)?) { continue; }
        }
        match apply_patch_step(doc, &op.path, 0, &op.val, env)? {
            PatchResult::Replace(v) => doc = v,
            PatchResult::Delete     => doc = Val::Null,
        }
    }
    Ok(doc)
}

fn apply_patch_step(
    v:        Val,
    path:     &[PathStep],
    i:        usize,
    val_expr: &Expr,
    env:      &Env,
) -> Result<PatchResult, EvalError> {
    if i == path.len() {
        if matches!(val_expr, Expr::DeleteMark) {
            return Ok(PatchResult::Delete);
        }
        let nv = eval(val_expr, &env.with_current(v))?;
        return Ok(PatchResult::Replace(nv));
    }
    match &path[i] {
        PathStep::Field(name) => {
            // Take parent first (clone only if refcount > 1), then mem::replace
            // the child out of its slot so the recursive call owns it with
            // refcount=1 — chains of into_map/into_vec stay alloc-free.
            let mut m = v.into_map().unwrap_or_default();
            let existing = if let Some(slot) = m.get_mut(name.as_str()) {
                std::mem::replace(slot, Val::Null)
            } else { Val::Null };
            let child = apply_patch_step(existing, path, i+1, val_expr, env)?;
            match child {
                PatchResult::Delete => { m.shift_remove(name.as_str()); }
                PatchResult::Replace(nv) => { m.insert(Arc::from(name.as_str()), nv); }
            }
            Ok(PatchResult::Replace(Val::obj(m)))
        }
        PathStep::Index(idx) => {
            let mut a = v.into_vec().unwrap_or_default();
            let resolved = resolve_idx(*idx, a.len() as i64);
            let existing = if resolved < a.len() {
                std::mem::replace(&mut a[resolved], Val::Null)
            } else { Val::Null };
            let child = apply_patch_step(existing, path, i+1, val_expr, env)?;
            match child {
                PatchResult::Delete => {
                    if resolved < a.len() { a.remove(resolved); }
                }
                PatchResult::Replace(nv) => {
                    if resolved < a.len() { a[resolved] = nv; }
                }
            }
            Ok(PatchResult::Replace(Val::arr(a)))
        }
        PathStep::DynIndex(expr) => {
            let idx_val = eval(expr, env)?;
            let idx = idx_val.as_i64().ok_or_else(|| {
                EvalError(format!("patch dyn-index: expected integer, got {}", idx_val.type_name()))
            })?;
            let mut a = v.into_vec().unwrap_or_default();
            let resolved = resolve_idx(idx, a.len() as i64);
            let existing = if resolved < a.len() {
                std::mem::replace(&mut a[resolved], Val::Null)
            } else { Val::Null };
            let child = apply_patch_step(existing, path, i+1, val_expr, env)?;
            match child {
                PatchResult::Delete => {
                    if resolved < a.len() { a.remove(resolved); }
                }
                PatchResult::Replace(nv) => {
                    if resolved < a.len() { a[resolved] = nv; }
                }
            }
            Ok(PatchResult::Replace(Val::arr(a)))
        }
        PathStep::Wildcard => {
            let mut arr = v.into_vec().ok_or_else(|| EvalError("patch [*]: expected array".into()))?;
            // Two-pointer in-place compact: read each slot, mutate/skip,
            // write kept results back at `write_idx`.  Reuses `arr`'s
            // allocation instead of building a fresh Vec.
            let mut write_idx = 0usize;
            for read_idx in 0..arr.len() {
                let item = std::mem::replace(&mut arr[read_idx], Val::Null);
                match apply_patch_step(item, path, i+1, val_expr, env)? {
                    PatchResult::Delete => {}
                    PatchResult::Replace(nv) => {
                        arr[write_idx] = nv;
                        write_idx += 1;
                    }
                }
            }
            arr.truncate(write_idx);
            Ok(PatchResult::Replace(Val::arr(arr)))
        }
        PathStep::WildcardFilter(pred) => {
            let mut arr = v.into_vec().ok_or_else(|| EvalError("patch [* if]: expected array".into()))?;
            let mut env_mut = env.clone();
            let mut write_idx = 0usize;
            for read_idx in 0..arr.len() {
                let item = std::mem::replace(&mut arr[read_idx], Val::Null);
                let frame = env_mut.push_lam(None, item.clone());
                let include = match eval(pred, &env_mut) {
                    Ok(v)  => is_truthy(&v),
                    Err(e) => { env_mut.pop_lam(frame); return Err(e); }
                };
                env_mut.pop_lam(frame);
                if include {
                    match apply_patch_step(item, path, i+1, val_expr, env)? {
                        PatchResult::Delete => {}
                        PatchResult::Replace(nv) => {
                            arr[write_idx] = nv;
                            write_idx += 1;
                        }
                    }
                } else {
                    arr[write_idx] = item;
                    write_idx += 1;
                }
            }
            arr.truncate(write_idx);
            Ok(PatchResult::Replace(Val::arr(arr)))
        }
        PathStep::Descendant(name) => {
            let v = descend_apply_patch(v, name, path, i, val_expr, env)?;
            Ok(PatchResult::Replace(v))
        }
    }
}

/// Descendant patch walker — DFS through the subtree.  At every object that
/// has `name`, apply the remaining path (starting at `i+1`) to the value of
/// `name`.  Children are visited *before* the current level so freshly-written
/// values are not re-walked (avoids runaway rewrites when the new value
/// itself contains `name`).
fn descend_apply_patch(
    v:        Val,
    name:     &str,
    path:     &[PathStep],
    i:        usize,
    val_expr: &Expr,
    env:      &Env,
) -> Result<Val, EvalError> {
    match v {
        Val::Obj(m) => {
            let mut map = Arc::try_unwrap(m).unwrap_or_else(|m| (*m).clone());
            // Recurse into children first, using original values.  In-place
            // slot swap instead of shift_remove+insert (which is O(n) per
            // key → O(n²) per map on wide objects).
            let n = map.len();
            for idx in 0..n {
                let child = if let Some((_, v)) = map.get_index_mut(idx) {
                    std::mem::replace(v, Val::Null)
                } else { continue };
                let replaced = descend_apply_patch(child, name, path, i, val_expr, env)?;
                if let Some((_, slot)) = map.get_index_mut(idx) { *slot = replaced; }
            }
            // Apply at this level.
            if map.contains_key(name) {
                let existing = map.get(name).cloned().unwrap_or(Val::Null);
                let r = apply_patch_step(existing, path, i + 1, val_expr, env)?;
                match r {
                    PatchResult::Delete      => { map.shift_remove(name); }
                    PatchResult::Replace(nv) => { map.insert(Arc::from(name), nv); }
                }
            }
            Ok(Val::obj(map))
        }
        Val::Arr(a) => {
            let mut vec = Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone());
            for slot in vec.iter_mut() {
                let old = std::mem::replace(slot, Val::Null);
                *slot = descend_apply_patch(old, name, path, i, val_expr, env)?;
            }
            Ok(Val::arr(vec))
        }
        other => Ok(other),
    }
}

/// Runtime cast dispatcher — powers the `as` operator.  Semantics mirror
/// the existing `.to_string()` / `.to_bool()` / `.to_number()` methods
/// for overlapping types; `int`/`float` are new targets that narrow
/// numeric values.
fn cast_val(v: &Val, ty: super::ast::CastType) -> Result<Val, EvalError> {
    use super::ast::CastType;
    match ty {
        CastType::Str => Ok(Val::Str(Arc::from(match v {
            Val::Null       => "null".to_string(),
            Val::Bool(b)    => b.to_string(),
            Val::Int(n)     => n.to_string(),
            Val::Float(f)   => f.to_string(),
            Val::Str(s)     => s.to_string(),
            other           => super::eval::util::val_to_string(other),
        }.as_str()))),
        CastType::Bool => Ok(Val::Bool(match v {
            Val::Null       => false,
            Val::Bool(b)    => *b,
            Val::Int(n)     => *n != 0,
            Val::Float(f)   => *f != 0.0,
            Val::Str(s)       => !s.is_empty(),
            Val::StrSlice(r)  => !r.is_empty(),
            Val::Arr(a)       => !a.is_empty(),
            Val::IntVec(a)    => !a.is_empty(),
            Val::FloatVec(a)  => !a.is_empty(),
            Val::StrVec(a)       => !a.is_empty(),
            Val::StrSliceVec(a)  => !a.is_empty(),
            Val::ObjVec(d)       => !d.rows.is_empty(),
            Val::Obj(o)       => !o.is_empty(),
            Val::ObjSmall(p)  => !p.is_empty(),
        })),
        CastType::Number | CastType::Float => match v {
            Val::Int(n)     => Ok(Val::Float(*n as f64)),
            Val::Float(_)   => Ok(v.clone()),
            Val::Str(s)     => s.parse::<f64>().map(Val::Float)
                                .map_err(|e| EvalError(format!("as float: {}", e))),
            Val::Bool(b)    => Ok(Val::Float(if *b { 1.0 } else { 0.0 })),
            Val::Null       => Ok(Val::Float(0.0)),
            _               => err!("as float: cannot convert"),
        },
        CastType::Int => match v {
            Val::Int(_)     => Ok(v.clone()),
            Val::Float(f)   => Ok(Val::Int(*f as i64)),
            Val::Str(s)     => s.parse::<i64>().map(Val::Int)
                                .or_else(|_| s.parse::<f64>().map(|f| Val::Int(f as i64)))
                                .map_err(|e| EvalError(format!("as int: {}", e))),
            Val::Bool(b)    => Ok(Val::Int(if *b { 1 } else { 0 })),
            Val::Null       => Ok(Val::Int(0)),
            _               => err!("as int: cannot convert"),
        },
        CastType::Array => match v {
            Val::Arr(_)     => Ok(v.clone()),
            Val::Null       => Ok(Val::arr(Vec::new())),
            other           => Ok(Val::arr(vec![other.clone()])),
        },
        CastType::Object => match v {
            Val::Obj(_)     => Ok(v.clone()),
            _               => err!("as object: cannot convert non-object"),
        },
        CastType::Null => Ok(Val::Null),
    }
}

// ── Pipeline ──────────────────────────────────────────────────────────────────

fn eval_pipeline(base: &Expr, steps: &[PipeStep], env: &Env) -> Result<Val, EvalError> {
    let mut current = eval(base, env)?;
    let mut env = env.clone();
    for step in steps {
        match step {
            PipeStep::Forward(rhs) => current = eval_pipe(current, rhs, &env)?,
            PipeStep::Bind(target) => env = apply_bind(target, &current, env)?,
        }
    }
    Ok(current)
}

fn apply_bind(target: &BindTarget, val: &Val, env: Env) -> Result<Env, EvalError> {
    match target {
        BindTarget::Name(name) => Ok(env.with_var(name, val.clone())),
        BindTarget::Obj { fields, rest } => {
            let obj = val.as_object()
                .ok_or_else(|| EvalError("bind destructure: expected object".into()))?;
            let mut e = env;
            for f in fields {
                e = e.with_var(f, obj.get(f.as_str()).cloned().unwrap_or(Val::Null));
            }
            if let Some(rest_name) = rest {
                let mut remainder: IndexMap<Arc<str>, Val> = IndexMap::new();
                for (k, v) in obj {
                    if !fields.iter().any(|f| f.as_str() == k.as_ref()) {
                        remainder.insert(k.clone(), v.clone());
                    }
                }
                e = e.with_var(rest_name, Val::obj(remainder));
            }
            Ok(e)
        }
        BindTarget::Arr(names) => {
            let len = val.arr_len()
                .ok_or_else(|| EvalError("bind destructure: expected array".into()))?;
            let mut e = env;
            for (i, name) in names.iter().enumerate() {
                let v = if i < len { val.get_index(i as i64) } else { Val::Null };
                e = e.with_var(name, v);
            }
            Ok(e)
        }
    }
}

fn eval_pipe(left: Val, rhs: &Expr, env: &Env) -> Result<Val, EvalError> {
    match rhs {
        Expr::Ident(name) => {
            if env.has_var(name) {
                eval(rhs, &env.with_current(left))
            } else {
                dispatch_method(left, name, &[], env)
            }
        }
        Expr::Chain(base, steps) => {
            if let Expr::Ident(name) = base.as_ref() {
                if !env.has_var(name) {
                    let mut v = dispatch_method(left, name, &[], env)?;
                    for step in steps { v = eval_step(v, step, env)?; }
                    return Ok(v);
                }
            }
            eval(rhs, &env.with_current(left))
        }
        _ => eval(rhs, &env.with_current(left)),
    }
}

/// Walk as many chain steps as possible on raw byte spans, then materialise.
///
/// Supports: `Descendant`, `Quantifier(First)` with any length, `Quantifier(One)`
/// only when exactly one span remains (otherwise falls through so the materialised
/// path can raise the proper error), `InlineFilter` / `.filter(...)` whose
/// predicate is a canonical equality literal.  All other steps terminate the
/// byte chain; the caller materialises and resumes normal step evaluation.
///
/// Returns `(value, steps_consumed)` where `steps_consumed >= 1` because the
/// entry `Descendant` step is always handled here.
/// A "first-selector" step: `.first()` method call or `Quantifier::First`.
/// When a `Descendant(k)` is immediately followed by one of these, the
/// scan can early-exit on the first match per span.
fn is_first_selector(step: &Step) -> bool {
    match step {
        Step::Quantifier(QuantifierKind::First) => true,
        Step::Method(name, args) if args.is_empty() && name == "first" => true,
        _ => false,
    }
}

fn byte_chain_eval(
    bytes: &[u8],
    root_key: &str,
    steps: &[Step],
) -> (Val, usize) {
    // Early-exit: `$..key.first()` / `$..key!` only needs the first match.
    let first_after_initial = steps.get(1).map(is_first_selector).unwrap_or(false);
    let mut spans: Vec<super::scan::ValueSpan> = if first_after_initial {
        super::scan::find_first_key_value_span(bytes, root_key)
            .into_iter().collect()
    } else {
        super::scan::find_key_value_spans(bytes, root_key)
    };
    let mut scalar = false;
    let mut consumed = 1usize;

    for (idx, step) in steps.iter().enumerate().skip(1) {
        match step {
            Step::Descendant(k) => {
                // Early-exit when the very next step is a first-selector:
                // only find one inner match per outer span, skip the rest.
                let next_first = steps.get(idx + 1)
                    .map(is_first_selector).unwrap_or(false);
                let mut next = Vec::with_capacity(spans.len());
                for s in &spans {
                    let sub = &bytes[s.start..s.end];
                    if next_first {
                        if let Some(s2) = super::scan::find_first_key_value_span(sub, k) {
                            next.push(super::scan::ValueSpan {
                                start: s.start + s2.start,
                                end:   s.start + s2.end,
                            });
                        }
                    } else {
                        for s2 in super::scan::find_key_value_spans(sub, k) {
                            next.push(super::scan::ValueSpan {
                                start: s.start + s2.start,
                                end:   s.start + s2.end,
                            });
                        }
                    }
                }
                spans = next;
                scalar = false;
            }
            Step::Quantifier(QuantifierKind::First) => {
                spans.truncate(1);
                scalar = true;
            }
            Step::Quantifier(QuantifierKind::One) => {
                if spans.len() != 1 { break; }
                scalar = true;
            }
            // `.first()` / `.last()` arrive as method calls (no `!` / `?` syntax).
            Step::Method(name, args) if args.is_empty() && name == "first" => {
                spans.truncate(1);
                scalar = true;
            }
            Step::Method(name, args) if args.is_empty() && name == "last" => {
                if let Some(last) = spans.pop() { spans = vec![last]; }
                scalar = true;
            }
            Step::InlineFilter(pred) => match canonical_eq_literal(pred) {
                Some(lit) => {
                    spans.retain(|s| {
                        s.end - s.start == lit.len()
                            && &bytes[s.start..s.end] == &lit[..]
                    });
                    scalar = false;
                }
                None => break,
            },
            Step::Method(name, args)
                if name == "filter" && args.len() == 1 =>
            {
                let pred = match &args[0] { Arg::Pos(e) | Arg::Named(_, e) => e };
                match canonical_eq_literal(pred) {
                    Some(lit) => {
                        spans.retain(|s| {
                            s.end - s.start == lit.len()
                                && &bytes[s.start..s.end] == &lit[..]
                        });
                        scalar = false;
                    }
                    None => break,
                }
            }
            _ => break,
        }
        consumed += 1;
    }

    // Numeric-fold fast path: trailing `.sum()/.avg()/.min()/.max()/.count()/.len()`
    // with no args — skip Val materialisation, parse numbers inline.
    if !scalar {
        if let Some(Step::Method(name, args)) = steps.get(consumed) {
            if args.is_empty() {
                let tail_ok = steps.len() == consumed + 1;
                if tail_ok {
                    match name.as_str() {
                        "count" | "len" => {
                            return (Val::Int(spans.len() as i64), consumed + 1);
                        }
                        "sum" => {
                            let f = super::scan::fold_nums(bytes, &spans);
                            let v = if f.count == 0 { Val::Int(0) }
                                else if f.is_float { Val::Float(f.float_sum) }
                                else { Val::Int(f.int_sum) };
                            return (v, consumed + 1);
                        }
                        "avg" => {
                            let f = super::scan::fold_nums(bytes, &spans);
                            let v = if f.count == 0 { Val::Null }
                                else { Val::Float(f.float_sum / f.count as f64) };
                            return (v, consumed + 1);
                        }
                        "min" => {
                            let f = super::scan::fold_nums(bytes, &spans);
                            let v = if !f.any { Val::Null }
                                else if f.is_float { Val::Float(f.min_f) }
                                else { Val::Int(f.min_i) };
                            return (v, consumed + 1);
                        }
                        "max" => {
                            let f = super::scan::fold_nums(bytes, &spans);
                            let v = if !f.any { Val::Null }
                                else if f.is_float { Val::Float(f.max_f) }
                                else { Val::Int(f.max_i) };
                            return (v, consumed + 1);
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    let mut materialised: Vec<Val> = Vec::with_capacity(spans.len());
    for s in &spans {
        match serde_json::from_slice::<serde_json::Value>(&bytes[s.start..s.end]) {
            Ok(v) => materialised.push(Val::from(&v)),
            Err(_) => {}
        }
    }

    let out = if scalar {
        materialised.into_iter().next().unwrap_or(Val::Null)
    } else {
        Val::arr(materialised)
    };
    (out, consumed)
}

/// Recognise `@ == lit` / `lit == @` predicates where `lit` is a canonical
/// JSON literal (int, string, bool, null).  Returns the serialised literal
/// bytes that can be fed to `scan::extract_values_eq`.  Float literals are
/// deliberately *not* accepted — `1.0` vs `1` and representation variance
/// make bytewise comparison unsafe.
fn canonical_eq_literal(pred: &Expr) -> Option<Vec<u8>> {
    let (l, r) = match pred {
        Expr::BinOp(l, BinOp::Eq, r) => (&**l, &**r),
        _ => return None,
    };
    let lit = match (l, r) {
        (Expr::Current, lit) => lit,
        (lit, Expr::Current) => lit,
        _ => return None,
    };
    match lit {
        Expr::Int(n)  => Some(n.to_string().into_bytes()),
        Expr::Bool(b) => Some(if *b { b"true".to_vec() } else { b"false".to_vec() }),
        Expr::Null    => Some(b"null".to_vec()),
        Expr::Str(s)  => serde_json::to_vec(&serde_json::Value::String(s.clone())).ok(),
        _ => None,
    }
}

/// Recognise `@.field == lit` / `lit == @.field` predicates. Returns the
/// field name and serialised literal bytes suitable for
/// `scan::find_enclosing_objects_eq`.  Float literals rejected for the same
/// reason as `canonical_eq_literal`.  Only a single leading field step is
/// supported (no nested `@.a.b`).
/// N-conjunct version: every arg must be a canonical `@.k == lit` predicate,
/// else returns None.  Empty input returns None.
pub(crate) fn canonical_field_eq_literals(args: &[Arg]) -> Option<Vec<(String, Vec<u8>)>> {
    if args.is_empty() || args.len() > 64 { return None; }
    let mut out = Vec::with_capacity(args.len());
    for a in args {
        let e = match a { Arg::Pos(e) | Arg::Named(_, e) => e };
        out.push(canonical_field_eq_literal(e)?);
    }
    Some(out)
}

/// Recognise `@.field op num` / `num op @.field` predicates where `op` is
/// one of `<`, `<=`, `>`, `>=` and the literal is a finite JSON number.
/// Returns `(field, ScanCmp, threshold)` normalised so `@.field` is always
/// on the LHS of the comparison (operator flipped when the literal was on
/// the LHS).
pub(crate) fn canonical_field_cmp_literal(
    pred: &Expr,
) -> Option<(String, super::scan::ScanCmp, f64)> {
    use super::scan::ScanCmp;
    let (l, op, r) = match pred {
        Expr::BinOp(l, op @ (BinOp::Lt | BinOp::Lte | BinOp::Gt | BinOp::Gte), r) =>
            (&**l, *op, &**r),
        _ => return None,
    };
    fn as_current_field(e: &Expr) -> Option<String> {
        if let Expr::Chain(base, steps) = e {
            if matches!(**base, Expr::Current) && steps.len() == 1 {
                if let Step::Field(name) = &steps[0] { return Some(name.clone()); }
            }
        }
        None
    }
    fn as_num(e: &Expr) -> Option<f64> {
        match e {
            Expr::Int(n)   => Some(*n as f64),
            Expr::Float(f) => if f.is_finite() { Some(*f) } else { None },
            _ => None,
        }
    }
    let (field, thresh, flip) = match (as_current_field(l), as_num(r)) {
        (Some(f), Some(n)) => (f, n, false),
        _ => match (as_current_field(r), as_num(l)) {
            (Some(f), Some(n)) => (f, n, true),
            _ => return None,
        },
    };
    let scan_op = match (op, flip) {
        (BinOp::Lt,  false) | (BinOp::Gt,  true)  => ScanCmp::Lt,
        (BinOp::Lte, false) | (BinOp::Gte, true)  => ScanCmp::Lte,
        (BinOp::Gt,  false) | (BinOp::Lt,  true)  => ScanCmp::Gt,
        (BinOp::Gte, false) | (BinOp::Lte, true)  => ScanCmp::Gte,
        _ => return None,
    };
    Some((field, scan_op, thresh))
}

/// Canonicalise a predicate list that mixes `@.k == lit` and `@.k op num`
/// conjuncts into a list of `(field, ScanPred)` pairs the mixed byte scan
/// can consume.  Returns `None` if any conjunct fits neither shape.
/// Empty input or more than 64 conjuncts → `None`.
pub(crate) fn canonical_field_mixed_predicates(
    args: &[Arg],
) -> Option<Vec<(String, super::scan::ScanPred)>> {
    use super::scan::ScanPred;
    if args.is_empty() || args.len() > 64 { return None; }
    let mut out = Vec::with_capacity(args.len());
    for a in args {
        let e = match a { Arg::Pos(e) | Arg::Named(_, e) => e };
        if let Some((k, lit)) = canonical_field_eq_literal(e) {
            out.push((k, ScanPred::Eq(lit)));
        } else if let Some((k, op, n)) = canonical_field_cmp_literal(e) {
            out.push((k, ScanPred::Cmp(op, n)));
        } else {
            return None;
        }
    }
    Some(out)
}

pub(crate) fn canonical_field_eq_literal(pred: &Expr) -> Option<(String, Vec<u8>)> {
    let (l, r) = match pred {
        Expr::BinOp(l, BinOp::Eq, r) => (&**l, &**r),
        _ => return None,
    };
    fn as_current_field(e: &Expr) -> Option<String> {
        if let Expr::Chain(base, steps) = e {
            if matches!(**base, Expr::Current) && steps.len() == 1 {
                if let Step::Field(name) = &steps[0] {
                    return Some(name.clone());
                }
            }
        }
        None
    }
    let (field, lit) = if let Some(f) = as_current_field(l) { (f, r) }
                       else if let Some(f) = as_current_field(r) { (f, l) }
                       else { return None };
    let bytes = match lit {
        Expr::Int(n)  => n.to_string().into_bytes(),
        Expr::Bool(b) => if *b { b"true".to_vec() } else { b"false".to_vec() },
        Expr::Null    => b"null".to_vec(),
        Expr::Str(s)  => serde_json::to_vec(&serde_json::Value::String(s.clone())).ok()?,
        _ => return None,
    };
    Some((field, bytes))
}

// ── Step evaluation ───────────────────────────────────────────────────────────

fn eval_step(val: Val, step: &Step, env: &Env) -> Result<Val, EvalError> {
    match step {
        Step::Field(name)    => Ok(val.get_field(name)),
        Step::OptField(name) => {
            if val.is_null() { Ok(Val::Null) } else { Ok(val.get_field(name)) }
        }
        Step::Descendant(name) => {
            let mut found = Vec::new();
            collect_desc(&val, name, &mut found);
            Ok(Val::arr(found))
        }
        Step::DescendAll => {
            let mut found = Vec::new();
            collect_all(&val, &mut found);
            Ok(Val::arr(found))
        }
        Step::InlineFilter(pred) => {
            let items = match val {
                Val::Arr(a) => Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone()),
                other => vec![other],
            };
            let mut out = Vec::new();
            let mut env_mut = env.clone();
            for item in items {
                let frame = env_mut.push_lam(None, item.clone());
                let truthy = match eval(pred, &env_mut) {
                    Ok(v)  => is_truthy(&v),
                    Err(e) => { env_mut.pop_lam(frame); return Err(e); }
                };
                env_mut.pop_lam(frame);
                if truthy { out.push(item); }
            }
            Ok(Val::arr(out))
        }
        Step::Quantifier(kind) => {
            use super::ast::QuantifierKind;
            match kind {
                QuantifierKind::First => {
                    Ok(match val {
                        Val::Arr(a) => a.first().cloned().unwrap_or(Val::Null),
                        other => other,
                    })
                }
                QuantifierKind::One => {
                    match val {
                        Val::Arr(a) if a.len() == 1 => Ok(a[0].clone()),
                        Val::Arr(a) => err!("quantifier !: expected exactly one element, got {}", a.len()),
                        other => Ok(other),
                    }
                }
            }
        }
        Step::Index(i) => Ok(val.get_index(*i)),
        Step::DynIndex(expr) => {
            let key = eval(expr, env)?;
            match key {
                Val::Int(i)  => Ok(val.get_index(i)),
                Val::Str(s)  => Ok(val.get_field(s.as_ref())),
                _ => err!("dynamic index must be a number or string"),
            }
        }
        Step::Slice(from, to) => {
            match val {
                Val::Arr(a) => {
                    let len = a.len() as i64;
                    let s = resolve_idx(from.unwrap_or(0), len);
                    let e = resolve_idx(to.unwrap_or(len), len);
                    let items = Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone());
                    let s = s.min(items.len());
                    let e = e.min(items.len());
                    Ok(Val::arr(items[s..e].to_vec()))
                }
                Val::IntVec(a) => {
                    let len = a.len() as i64;
                    let s = resolve_idx(from.unwrap_or(0), len);
                    let e = resolve_idx(to.unwrap_or(len), len);
                    let items = Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone());
                    let s = s.min(items.len());
                    let e = e.min(items.len());
                    Ok(Val::int_vec(items[s..e].to_vec()))
                }
                Val::FloatVec(a) => {
                    let len = a.len() as i64;
                    let s = resolve_idx(from.unwrap_or(0), len);
                    let e = resolve_idx(to.unwrap_or(len), len);
                    let items = Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone());
                    let s = s.min(items.len());
                    let e = e.min(items.len());
                    Ok(Val::float_vec(items[s..e].to_vec()))
                }
                _ => Ok(Val::Null),
            }
        }
        Step::Method(name, args)    => dispatch_method(val, name, args, env),
        Step::OptMethod(name, args) => {
            if val.is_null() { Ok(Val::Null) } else { dispatch_method(val, name, args, env) }
        }
    }
}

fn resolve_idx(i: i64, len: i64) -> usize {
    (if i < 0 { (len + i).max(0) } else { i }) as usize
}

fn collect_desc(v: &Val, name: &str, out: &mut Vec<Val>) {
    match v {
        Val::Obj(m) => {
            if let Some(v) = m.get(name) { out.push(v.clone()); }
            for v in m.values() { collect_desc(v, name, out); }
        }
        Val::Arr(a) => { for item in a.as_ref() { collect_desc(item, name, out); } }
        _ => {}
    }
}

fn collect_all(v: &Val, out: &mut Vec<Val>) {
    match v {
        Val::Obj(m) => {
            out.push(v.clone());
            for child in m.values() { collect_all(child, out); }
        }
        Val::Arr(a) => {
            for item in a.as_ref() { collect_all(item, out); }
        }
        other => out.push(other.clone()),
    }
}

// ── Method dispatch ───────────────────────────────────────────────────────────

pub(super) fn dispatch_method(recv: Val, name: &str, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    if let Some(f) = builtins::global().get(name) {
        return f(recv, args, env);
    }
    if !env.registry.is_empty() {
        if let Some(method) = env.registry.get(name) {
            let evaluated: Result<Vec<Val>, EvalError> =
                args.iter().map(|a| eval_pos(a, env)).collect();
            return method.call(recv, &evaluated?);
        }
    }
    err!("unknown method '{}'", name)
}

// ── Object construction ───────────────────────────────────────────────────────

fn eval_object(fields: &[ObjField], env: &Env) -> Result<Val, EvalError> {
    let mut map: IndexMap<Arc<str>, Val> = IndexMap::new();
    for field in fields {
        match field {
            ObjField::Short(name) => {
                let v = if let Some(v) = env.get_var(name) { v.clone() }
                        else { env.current.get_field(name) };
                if !v.is_null() { map.insert(Arc::from(name.as_str()), v); }
            }
            ObjField::Kv { key, val, optional, cond } => {
                if let Some(c) = cond {
                    if !is_truthy(&eval(c, env)?) { continue; }
                }
                let v = eval(val, env)?;
                if *optional && v.is_null() { continue; }
                map.insert(Arc::from(key.as_str()), v);
            }
            ObjField::Dynamic { key, val } => {
                let k: Arc<str> = match eval(key, env)? {
                    Val::Str(s) => s,
                    other       => Arc::<str>::from(val_to_key(&other)),
                };
                map.insert(k, eval(val, env)?);
            }
            ObjField::Spread(expr) => {
                if let Val::Obj(other) = eval(expr, env)? {
                    let entries = Arc::try_unwrap(other).unwrap_or_else(|m| (*m).clone());
                    for (k, v) in entries { map.insert(k, v); }
                }
            }
            ObjField::SpreadDeep(expr) => {
                if let Val::Obj(other) = eval(expr, env)? {
                    let base = std::mem::take(&mut map);
                    let merged = deep_merge_concat(Val::obj(base), Val::Obj(other));
                    if let Val::Obj(m) = merged {
                        map = Arc::try_unwrap(m).unwrap_or_else(|m| (*m).clone());
                    }
                }
            }
        }
    }
    Ok(Val::obj(map))
}

// ── F-string ──────────────────────────────────────────────────────────────────

fn eval_fstring(parts: &[FStringPart], env: &Env) -> Result<Val, EvalError> {
    let mut out = String::new();
    for part in parts {
        match part {
            FStringPart::Lit(s) => out.push_str(s),
            FStringPart::Interp { expr, fmt } => {
                let val = eval(expr, env)?;
                let s = match fmt {
                    None                     => val_to_string(&val),
                    Some(FmtSpec::Spec(spec)) => apply_fmt_spec(&val, spec),
                    Some(FmtSpec::Pipe(method)) => {
                        val_to_string(&dispatch_method(val, method, &[], env)?)
                    }
                };
                out.push_str(&s);
            }
        }
    }
    Ok(Val::Str(Arc::from(out.as_str())))
}

fn apply_fmt_spec(val: &Val, spec: &str) -> String {
    if let Some(rest) = spec.strip_suffix('f') {
        if let Some(prec_str) = rest.strip_prefix('.') {
            if let Ok(prec) = prec_str.parse::<usize>() {
                if let Some(f) = val.as_f64() { return format!("{:.prec$}", f); }
            }
        }
    }
    if spec == "d" {
        if let Some(i) = val.as_i64() { return format!("{}", i); }
    }
    let s = val_to_string(val);
    if let Some(w) = spec.strip_prefix('>').and_then(|s| s.parse::<usize>().ok()) { return format!("{:>w$}", s); }
    if let Some(w) = spec.strip_prefix('<').and_then(|s| s.parse::<usize>().ok()) { return format!("{:<w$}", s); }
    if let Some(w) = spec.strip_prefix('^').and_then(|s| s.parse::<usize>().ok()) { return format!("{:^w$}", s); }
    if let Some(w) = spec.strip_prefix('0').and_then(|s| s.parse::<usize>().ok()) {
        if let Some(i) = val.as_i64() { return format!("{:0>w$}", i); }
    }
    s
}

// ── Binary operators ──────────────────────────────────────────────────────────

fn eval_binop(l: &Expr, op: BinOp, r: &Expr, env: &Env) -> Result<Val, EvalError> {
    match op {
        BinOp::And => {
            let lv = eval(l, env)?;
            if !is_truthy(&lv) { return Ok(Val::Bool(false)); }
            Ok(Val::Bool(is_truthy(&eval(r, env)?)))
        }
        BinOp::Or => {
            let lv = eval(l, env)?;
            if is_truthy(&lv) { return Ok(lv); }
            eval(r, env)
        }
        _ => {
            let lv = eval(l, env)?;
            let rv = eval(r, env)?;
            match op {
                BinOp::Add  => add_vals(lv, rv),
                BinOp::Sub  => num_op(lv, rv, |a, b| a - b, |a, b| a - b),
                BinOp::Mul  => num_op(lv, rv, |a, b| a * b, |a, b| a * b),
                BinOp::Div  => {
                    let b = rv.as_f64().unwrap_or(0.0);
                    if b == 0.0 { return err!("division by zero"); }
                    Ok(Val::Float(lv.as_f64().unwrap_or(0.0) / b))
                }
                BinOp::Mod   => num_op(lv, rv, |a, b| a % b, |a, b| a % b),
                BinOp::Eq    => Ok(Val::Bool(vals_eq(&lv, &rv))),
                BinOp::Neq   => Ok(Val::Bool(!vals_eq(&lv, &rv))),
                BinOp::Lt    => Ok(Val::Bool(cmp_vals(&lv, &rv) == std::cmp::Ordering::Less)),
                BinOp::Lte   => Ok(Val::Bool(cmp_vals(&lv, &rv) != std::cmp::Ordering::Greater)),
                BinOp::Gt    => Ok(Val::Bool(cmp_vals(&lv, &rv) == std::cmp::Ordering::Greater)),
                BinOp::Gte   => Ok(Val::Bool(cmp_vals(&lv, &rv) != std::cmp::Ordering::Less)),
                BinOp::Fuzzy => {
                    let ls = match &lv { Val::Str(s) => s.to_lowercase(), _ => val_to_string(&lv).to_lowercase() };
                    let rs = match &rv { Val::Str(s) => s.to_lowercase(), _ => val_to_string(&rv).to_lowercase() };
                    Ok(Val::Bool(ls.contains(&rs) || rs.contains(&ls)))
                }
                BinOp::And | BinOp::Or => unreachable!(),
            }
        }
    }
}

// ── Global functions ──────────────────────────────────────────────────────────

fn eval_global(name: &str, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    match name {
        "coalesce" => {
            for arg in args {
                let v = eval_pos(arg, env)?;
                if !v.is_null() { return Ok(v); }
            }
            Ok(Val::Null)
        }
        "chain" | "join" => {
            let mut out = Vec::new();
            for arg in args {
                match eval_pos(arg, env)? {
                    Val::Arr(a) => {
                        let items = Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone());
                        out.extend(items);
                    }
                    Val::IntVec(a) => out.extend(a.iter().map(|n| Val::Int(*n))),
                    Val::FloatVec(a) => out.extend(a.iter().map(|f| Val::Float(*f))),
                    v => out.push(v),
                }
            }
            Ok(Val::arr(out))
        }
        "zip"         => func_arrays::global_zip(args, env),
        "zip_longest" => func_arrays::global_zip_longest(args, env),
        "product"     => func_arrays::global_product(args, env),
        "range"       => eval_range(args, env),
        other => {
            if let Some(first) = args.first() {
                let recv = eval_pos(first, env)?;
                dispatch_method(recv, other, args.get(1..).unwrap_or(&[]), env)
            } else {
                dispatch_method(env.current.clone(), other, &[], env)
            }
        }
    }
}

// ── range() generator ────────────────────────────────────────────────────────
//
// `range(n)`            -> [0, 1, …, n-1]
// `range(from, upto)`   -> [from, from+1, …, upto-1]
// `range(from, upto, step)` -> arithmetic progression; step may be negative.
//
// Returns an eager `Val::Arr` (jetro is value-oriented, not streaming).
// Int-only; non-numeric args produce a descriptive error.  Empty when
// the step points away from `upto` or when `step == 0`.

fn eval_range(args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let n = args.len();
    if n == 0 || n > 3 {
        return err!("range: expected 1..3 args, got {}", n);
    }
    let mut nums = Vec::with_capacity(n);
    for a in args {
        let v = eval_pos(a, env)?;
        let i = v.as_i64().ok_or_else(|| EvalError("range: expected integer arg".into()))?;
        nums.push(i);
    }
    let (from, upto, step) = match nums.as_slice() {
        [n]          => (0, *n, 1i64),
        [f, u]       => (*f, *u, 1i64),
        [f, u, s]    => (*f, *u, *s),
        _            => unreachable!(),
    };
    if step == 0 { return Ok(Val::int_vec(Vec::new())); }
    let len_hint: usize = if step > 0 && upto > from {
        (((upto - from) + step - 1) / step).max(0) as usize
    } else if step < 0 && upto < from {
        (((from - upto) + (-step) - 1) / (-step)).max(0) as usize
    } else { 0 };
    let mut out: Vec<i64> = Vec::with_capacity(len_hint);
    let mut i = from;
    if step > 0 {
        while i < upto { out.push(i); i += step; }
    } else {
        while i > upto { out.push(i); i += step; }
    }
    Ok(Val::int_vec(out))
}

// ── Apply helpers ─────────────────────────────────────────────────────────────

pub(super) fn apply_item(item: Val, arg: &Arg, env: &Env) -> Result<Val, EvalError> {
    match arg {
        Arg::Pos(expr) | Arg::Named(_, expr) => apply_expr_item(item, expr, env),
    }
}

/// Hot-loop variant of `apply_item` that binds into a caller-supplied
/// mutable Env via `push_lam`/`pop_lam` — zero `SmallVec`/`Arc` clones
/// per call.  Caller clones the Env once before the loop.
#[inline]
pub(super) fn apply_item_mut(item: Val, arg: &Arg, env: &mut Env) -> Result<Val, EvalError> {
    let expr = match arg { Arg::Pos(e) | Arg::Named(_, e) => e };
    match expr {
        Expr::Lambda { params, body } => {
            let name = params.first().map(|s| s.as_str());
            let frame = env.push_lam(name, item);
            let r = eval(body, env);
            env.pop_lam(frame);
            r
        }
        _ => {
            let frame = env.push_lam(None, item);
            let r = eval(expr, env);
            env.pop_lam(frame);
            r
        }
    }
}

/// Hot-loop variant of `apply_item2`.  Binds one or two params (or just
/// `current` for zero-param lambdas) via in-place Env mutation.
#[inline]
pub(super) fn apply_item2_mut(a: Val, b: Val, arg: &Arg, env: &mut Env) -> Result<Val, EvalError> {
    match arg {
        Arg::Pos(Expr::Lambda { params, body }) | Arg::Named(_, Expr::Lambda { params, body }) => {
            match params.as_slice() {
                [] => {
                    let frame = env.push_lam(None, b);
                    let r = eval(body, env);
                    env.pop_lam(frame);
                    r
                }
                [p] => {
                    let frame = env.push_lam(Some(p), b);
                    let r = eval(body, env);
                    env.pop_lam(frame);
                    r
                }
                [p1, p2, ..] => {
                    // Two-param: push p1=a, then p2=b (pop order reversed).
                    let f1 = env.push_lam(Some(p1), a);
                    let f2 = env.push_lam(Some(p2), b);
                    let r = eval(body, env);
                    env.pop_lam(f2);
                    env.pop_lam(f1);
                    r
                }
            }
        }
        _ => apply_item_mut(b, arg, env),
    }
}

fn apply_expr_item(item: Val, expr: &Expr, env: &Env) -> Result<Val, EvalError> {
    match expr {
        Expr::Lambda { params, body } => {
            let inner = if params.is_empty() {
                env.with_current(item)
            } else {
                let mut e = env.with_var(&params[0], item.clone());
                e.current = item;
                e
            };
            eval(body, &inner)
        }
        _ => eval(expr, &env.with_current(item)),
    }
}

pub(super) fn eval_pos(arg: &Arg, env: &Env) -> Result<Val, EvalError> {
    match arg { Arg::Pos(e) | Arg::Named(_, e) => eval(e, env) }
}

pub(super) fn first_i64_arg(args: &[Arg], env: &Env) -> Result<i64, EvalError> {
    args.first()
        .map(|a| eval_pos(a, env)?.as_i64().ok_or_else(|| EvalError("expected integer arg".into())))
        .transpose()
        .map(|v| v.unwrap_or(1))
}

pub(super) fn str_arg(args: &[Arg], idx: usize, env: &Env) -> Result<String, EvalError> {
    args.get(idx)
        .map(|a| eval_pos(a, env))
        .transpose()?
        .map(|v| val_to_string(&v))
        .ok_or_else(|| EvalError(format!("missing string arg at position {}", idx)))
}

// ── Comprehension helpers ─────────────────────────────────────────────────────

fn eval_iter(iter: &Expr, env: &Env) -> Result<Vec<Val>, EvalError> {
    match eval(iter, env)? {
        Val::Arr(a) => Ok(Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone())),
        Val::Obj(m) => {
            let entries = Arc::try_unwrap(m).unwrap_or_else(|m| (*m).clone());
            Ok(entries.into_iter().map(|(k, v)| {
                let mut o: IndexMap<Arc<str>, Val> = IndexMap::new();
                o.insert(Arc::from("key"),   Val::Str(k));
                o.insert(Arc::from("value"), v);
                Val::obj(o)
            }).collect())
        }
        other => Ok(vec![other]),
    }
}

/// In-place comprehension bind — one or two vars pushed + `current`
/// swapped.  Returns a pair of frames (second is `None` for 0/1-var
/// forms) so the caller can unwind with `unbind_vars_mut`.
#[inline]
fn bind_vars_mut(env: &mut Env, vars: &[String], item: Val) -> (LamFrame, Option<LamFrame>) {
    match vars {
        [] => (env.push_lam(None, item), None),
        [v] => (env.push_lam(Some(v), item), None),
        [v1, v2, ..] => {
            let idx = item.get("index").cloned().unwrap_or(Val::Null);
            let val = item.get("value").cloned().unwrap_or_else(|| item.clone());
            let f1 = env.push_lam(Some(v1), idx);
            // Push v2 with val; also sets current to val (matches legacy).
            let f2 = env.push_lam(Some(v2), val);
            (f1, Some(f2))
        }
    }
}

#[inline]
fn unbind_vars_mut(env: &mut Env, frames: (LamFrame, Option<LamFrame>)) {
    if let Some(f2) = frames.1 { env.pop_lam(f2); }
    env.pop_lam(frames.0);
}
