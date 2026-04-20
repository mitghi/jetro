use std::sync::Arc;
use indexmap::IndexMap;
use smallvec::SmallVec;

use super::ast::*;

pub mod value;
pub mod util;
pub mod methods;
pub(super) mod builtins;
mod func_strings;
mod func_arrays;
mod func_objects;
mod func_paths;
mod func_aggregates;
mod func_csv;

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

#[derive(Clone)]
pub struct Env {
    vars:     SmallVec<[(Arc<str>, Val); 4]>,
    pub root:    Val,
    pub current: Val,
    registry: Arc<MethodRegistry>,
}

impl Env {
    fn new(root: Val) -> Self {
        Self {
            vars: SmallVec::new(),
            root: root.clone(),
            current: root,
            registry: Arc::new(MethodRegistry::new()),
        }
    }

    pub fn new_with_registry(root: Val, registry: Arc<MethodRegistry>) -> Self {
        Self { vars: SmallVec::new(), root: root.clone(), current: root, registry }
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
        }
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
        Self { vars, root: self.root.clone(), current: self.current.clone(), registry: self.registry.clone() }
    }

    fn with_vars2(&self, n1: &str, v1: Val, n2: &str, v2: Val) -> Self {
        let mut vars = self.vars.clone();
        if let Some(p) = vars.iter().position(|(k, _)| k.as_ref() == n1) { vars[p].1 = v1; }
        else { vars.push((Arc::from(n1), v1)); }
        if let Some(p) = vars.iter().position(|(k, _)| k.as_ref() == n2) { vars[p].1 = v2; }
        else { vars.push((Arc::from(n2), v2)); }
        Self { vars, root: self.root.clone(), current: self.current.clone(), registry: self.registry.clone() }
    }

    fn extend_vars(&self, extra: impl IntoIterator<Item = (Arc<str>, Val)>) -> Self {
        let mut vars = self.vars.clone();
        for (name, val) in extra {
            if let Some(pos) = vars.iter().position(|(k, _)| *k == name) {
                vars[pos].1 = val;
            } else {
                vars.push((name, val));
            }
        }
        Self { vars, root: self.root.clone(), current: self.current.clone(), registry: self.registry.clone() }
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
            for item in items {
                let ie = bind_vars(env, vars, item);
                if let Some(c) = cond { if !is_truthy(&eval(c, &ie)?) { continue; } }
                out.push(eval(expr, &ie)?);
            }
            Ok(Val::arr(out))
        }

        Expr::DictComp { key, val, vars, iter, cond } => {
            let items = eval_iter(iter, env)?;
            let mut map: IndexMap<Arc<str>, Val> = IndexMap::new();
            for item in items {
                let ie = bind_vars(env, vars, item);
                if let Some(c) = cond { if !is_truthy(&eval(c, &ie)?) { continue; } }
                let k: Arc<str> = Arc::from(val_to_key(&eval(key, &ie)?).as_str());
                map.insert(k, eval(val, &ie)?);
            }
            Ok(Val::obj(map))
        }

        Expr::SetComp { expr, vars, iter, cond } | Expr::GenComp { expr, vars, iter, cond } => {
            let items = eval_iter(iter, env)?;
            let mut seen: Vec<String> = Vec::new();
            let mut out = Vec::new();
            for item in items {
                let ie = bind_vars(env, vars, item);
                if let Some(c) = cond { if !is_truthy(&eval(c, &ie)?) { continue; } }
                let v = eval(expr, &ie)?;
                let k = val_to_key(&v);
                if !seen.contains(&k) { seen.push(k); out.push(v); }
            }
            Ok(Val::arr(out))
        }

        Expr::Lambda { .. } => err!("lambda cannot be used as standalone value"),

        Expr::Let { name, init, body } => {
            let v = eval(init, env)?;
            eval(body, &env.with_var(name, v))
        }

        Expr::GlobalCall { name, args } => eval_global(name, args, env),
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
            let arr = val.as_array()
                .ok_or_else(|| EvalError("bind destructure: expected array".into()))?;
            let mut e = env;
            for (i, name) in names.iter().enumerate() {
                e = e.with_var(name, arr.get(i).cloned().unwrap_or(Val::Null));
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
            for item in items {
                let ie = env.with_current(item.clone());
                if is_truthy(&eval(pred, &ie)?) { out.push(item); }
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
            if let Val::Arr(a) = val {
                let len = a.len() as i64;
                let s = resolve_idx(from.unwrap_or(0), len);
                let e = resolve_idx(to.unwrap_or(len), len);
                let items = Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone());
                let s = s.min(items.len());
                let e = e.min(items.len());
                Ok(Val::arr(items[s..e].to_vec()))
            } else { Ok(Val::Null) }
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
            ObjField::Kv { key, val, optional } => {
                let v = eval(val, env)?;
                if *optional && v.is_null() { continue; }
                map.insert(Arc::from(key.as_str()), v);
            }
            ObjField::Dynamic { key, val } => {
                let k: Arc<str> = Arc::from(val_to_key(&eval(key, env)?).as_str());
                map.insert(k, eval(val, env)?);
            }
            ObjField::Spread(expr) => {
                if let Val::Obj(other) = eval(expr, env)? {
                    let entries = Arc::try_unwrap(other).unwrap_or_else(|m| (*m).clone());
                    for (k, v) in entries { map.insert(k, v); }
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
                    v => out.push(v),
                }
            }
            Ok(Val::arr(out))
        }
        "zip"         => func_arrays::global_zip(args, env),
        "zip_longest" => func_arrays::global_zip_longest(args, env),
        "product"     => func_arrays::global_product(args, env),
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

// ── Apply helpers ─────────────────────────────────────────────────────────────

pub(super) fn apply_item(item: Val, arg: &Arg, env: &Env) -> Result<Val, EvalError> {
    match arg {
        Arg::Pos(expr) | Arg::Named(_, expr) => apply_expr_item(item, expr, env),
    }
}

pub(super) fn apply_item2(a: Val, b: Val, arg: &Arg, env: &Env) -> Result<Val, EvalError> {
    match arg {
        Arg::Pos(Expr::Lambda { params, body }) | Arg::Named(_, Expr::Lambda { params, body }) => {
            let inner = match params.as_slice() {
                []           => env.with_current(b),
                [p]          => env.with_var(p, b),
                [p1, p2, ..] => env.with_vars2(p1, a, p2, b),
            };
            eval(body, &inner)
        }
        _ => apply_item(b, arg, env),
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

fn bind_vars(env: &Env, vars: &[String], item: Val) -> Env {
    match vars {
        [] => env.with_current(item),
        [v] => { let mut e = env.with_var(v, item.clone()); e.current = item; e }
        [v1, v2, ..] => {
            let idx = item.get("index").cloned().unwrap_or(Val::Null);
            let val = item.get("value").cloned().unwrap_or_else(|| item.clone());
            let mut e = env.with_vars2(v1, idx, v2, val.clone());
            e.current = val;
            e
        }
    }
}
