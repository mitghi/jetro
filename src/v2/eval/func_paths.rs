use std::sync::Arc;
use indexmap::IndexMap;

use crate::v2::ast::Arg;
use super::{Env, EvalError, eval_pos, str_arg};
use super::value::Val;
use super::util::{val_to_string, val_key};

macro_rules! err {
    ($($t:tt)*) => { Err(EvalError(format!($($t)*))) };
}

// ── Path segment ──────────────────────────────────────────────────────────────

pub enum PathSeg { Field(String), Index(i64) }

pub fn parse_path_segs(path: &str) -> Vec<PathSeg> {
    let mut segs = Vec::new();
    let mut cur  = String::new();
    let mut chars = path.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '.' => {
                if !cur.is_empty() { segs.push(PathSeg::Field(std::mem::take(&mut cur))); }
            }
            '[' => {
                if !cur.is_empty() { segs.push(PathSeg::Field(std::mem::take(&mut cur))); }
                let mut idx = String::new();
                for c2 in chars.by_ref() { if c2 == ']' { break; } idx.push(c2); }
                segs.push(PathSeg::Index(idx.parse().unwrap_or(0)));
            }
            _ => cur.push(c),
        }
    }
    if !cur.is_empty() { segs.push(PathSeg::Field(cur)); }
    segs
}

// ── Path get / set / del ──────────────────────────────────────────────────────

pub fn get_path_impl(val: &Val, segs: &[PathSeg]) -> Val {
    if segs.is_empty() { return val.clone(); }  // O(1) clone (Arc bump for compound types)
    let next: Val = match &segs[0] {
        PathSeg::Field(f) => val.get(f).cloned().unwrap_or(Val::Null),
        PathSeg::Index(i) => val.get_index(*i),
    };
    get_path_impl(&next, &segs[1..])
}

pub fn set_path_impl(val: Val, segs: &[PathSeg], new_val: Val) -> Val {
    if segs.is_empty() { return new_val; }
    match (&segs[0], val) {
        (PathSeg::Field(f), Val::Obj(m)) => {
            let mut map = Arc::try_unwrap(m).unwrap_or_else(|m| (*m).clone());
            let child   = map.shift_remove(f.as_str()).unwrap_or(Val::Null);
            map.insert(Arc::from(f.as_str()), set_path_impl(child, &segs[1..], new_val));
            Val::obj(map)
        }
        (PathSeg::Index(i), Val::Arr(a)) => {
            let mut arr = Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone());
            let idx     = resolve_idx(*i, arr.len() as i64);
            if idx < arr.len() {
                let child = arr[idx].clone();
                arr[idx] = set_path_impl(child, &segs[1..], new_val);
            }
            Val::arr(arr)
        }
        (PathSeg::Field(f), _) => {
            let mut m = IndexMap::new();
            m.insert(Arc::from(f.as_str()), set_path_impl(Val::Null, &segs[1..], new_val));
            Val::obj(m)
        }
        (_, v) => v,
    }
}

pub fn del_path_impl(val: Val, segs: &[PathSeg]) -> Val {
    if segs.is_empty() { return Val::Null; }
    match (&segs[0], val) {
        (PathSeg::Field(f), Val::Obj(m)) => {
            let mut map = Arc::try_unwrap(m).unwrap_or_else(|m| (*m).clone());
            if segs.len() == 1 {
                map.shift_remove(f.as_str());
            } else if let Some(child) = map.shift_remove(f.as_str()) {
                map.insert(Arc::from(f.as_str()), del_path_impl(child, &segs[1..]));
            }
            Val::obj(map)
        }
        (PathSeg::Index(i), Val::Arr(a)) => {
            let mut arr = Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone());
            let idx     = resolve_idx(*i, arr.len() as i64);
            if segs.len() == 1 {
                if idx < arr.len() { arr.remove(idx); }
            } else if idx < arr.len() {
                let child = arr[idx].clone();
                arr[idx] = del_path_impl(child, &segs[1..]);
            }
            Val::arr(arr)
        }
        (_, v) => v,
    }
}

fn resolve_idx(i: i64, len: i64) -> usize {
    (if i < 0 { (len + i).max(0) } else { i }) as usize
}

// ── Flatten / unflatten ───────────────────────────────────────────────────────

pub fn flatten_keys_impl(prefix: &str, val: &Val, sep: &str, out: &mut IndexMap<Arc<str>, Val>) {
    match val {
        Val::Obj(m) => {
            for (k, v) in m.iter() {
                let full = if prefix.is_empty() {
                    k.to_string()
                } else {
                    format!("{}{}{}", prefix, sep, k)
                };
                flatten_keys_impl(&full, v, sep, out);
            }
        }
        _ => { out.insert(Arc::from(prefix), val.clone()); }
    }
}

pub fn unflatten_keys_impl(m: &IndexMap<Arc<str>, Val>, sep: &str) -> Val {
    let mut root: IndexMap<Arc<str>, Val> = IndexMap::new();
    for (key, val) in m {
        let parts: Vec<&str> = key.split(sep).collect();
        insert_nested(&mut root, &parts, val.clone());
    }
    Val::obj(root)
}

fn insert_nested(obj: &mut IndexMap<Arc<str>, Val>, parts: &[&str], val: Val) {
    if parts.is_empty() { return; }
    if parts.len() == 1 { obj.insert(val_key(parts[0]), val); return; }
    let entry = obj.entry(val_key(parts[0])).or_insert_with(|| Val::obj(IndexMap::new()));
    if let Val::Obj(child) = entry {
        insert_nested(Arc::make_mut(child), &parts[1..], val);
    }
}

// ── Method entry points ───────────────────────────────────────────────────────

pub fn get_path(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let path = str_arg(args, 0, env)?;
    Ok(get_path_impl(&recv, &parse_path_segs(&path)))
}

pub fn set_path(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let path    = str_arg(args, 0, env)?;
    let new_val = args.get(1).map(|a| eval_pos(a, env)).transpose()?.unwrap_or(Val::Null);
    Ok(set_path_impl(recv, &parse_path_segs(&path), new_val))
}

pub fn del_path(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let path = str_arg(args, 0, env)?;
    Ok(del_path_impl(recv, &parse_path_segs(&path)))
}

pub fn del_paths(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let mut val = recv;
    for a in args {
        if let Ok(path) = eval_pos(a, env).map(|v| val_to_string(&v)) {
            val = del_path_impl(val, &parse_path_segs(&path));
        }
    }
    Ok(val)
}

pub fn has_path(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let path = str_arg(args, 0, env)?;
    Ok(Val::Bool(!get_path_impl(&recv, &parse_path_segs(&path)).is_null()))
}

pub fn flatten_keys(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let sep = args.first()
        .and_then(|a| eval_pos(a, env).ok())
        .and_then(|v| if let Val::Str(s) = v { Some(s.to_string()) } else { None })
        .unwrap_or_else(|| ".".to_string());
    let mut out = IndexMap::new();
    flatten_keys_impl("", &recv, &sep, &mut out);
    Ok(Val::obj(out))
}

pub fn unflatten_keys(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let sep = args.first()
        .and_then(|a| eval_pos(a, env).ok())
        .and_then(|v| if let Val::Str(s) = v { Some(s.to_string()) } else { None })
        .unwrap_or_else(|| ".".to_string());
    match &recv {
        Val::Obj(m) => Ok(unflatten_keys_impl(m, &sep)),
        _ => err!("unflatten_keys: expected object"),
    }
}
