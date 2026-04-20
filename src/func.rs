//! Module func provides abstraction for jetro functions.

use crate::context::{Context, Error, Filter, Func, FuncArg, MapBody, ObjKey, Path};
use serde_json::Value;
use std::{
    cell::RefCell,
    collections::{BTreeMap, VecDeque},
    rc::Rc,
};

use super::context::MapAST;

// ── BuiltinFn ─────────────────────────────────────────────────────────────────

/// Compile-time identity for every built-in function.
///
/// Resolved once at compile time from the function name string and stored
/// directly in `Opcode::CallBuiltin`, eliminating all `BTreeMap` lookups,
/// vtable dispatches, and `Box<dyn Callable>` indirections at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BuiltinFn {
    Reverse, Formats, Sum, Len, Head, Tail, All, Map, Keys, Values, Min, Max, Zip,
    Avg, Add, Sub, Mul, Div, Abs, Round, Floor, Ceil,
    Any, Not,
    Last, Nth, Flatten, FlatMap, Chunk, Unique, Distinct, SortBy, JoinStr, Compact, Count,
    GroupBy, CountBy, IndexBy, Tally,
    Merge, Omit, Select, Rename, Set, Coalesce,
    Join, Lookup,
    Find, FilterBy, Get, Resolve, Deref, Pluck,
}

impl BuiltinFn {
    #[inline]
    pub(crate) fn from_name(name: &str) -> Option<Self> {
        match name {
            "reverse"   => Some(Self::Reverse),
            "formats"   => Some(Self::Formats),
            "sum"       => Some(Self::Sum),
            "len"       => Some(Self::Len),
            "head"      => Some(Self::Head),
            "tail"      => Some(Self::Tail),
            "all"       => Some(Self::All),
            "map"       => Some(Self::Map),
            "keys"      => Some(Self::Keys),
            "values"    => Some(Self::Values),
            "min"       => Some(Self::Min),
            "max"       => Some(Self::Max),
            "zip"       => Some(Self::Zip),
            "avg"       => Some(Self::Avg),
            "add"       => Some(Self::Add),
            "sub"       => Some(Self::Sub),
            "mul"       => Some(Self::Mul),
            "div"       => Some(Self::Div),
            "abs"       => Some(Self::Abs),
            "round"     => Some(Self::Round),
            "floor"     => Some(Self::Floor),
            "ceil"      => Some(Self::Ceil),
            "any"       => Some(Self::Any),
            "not"       => Some(Self::Not),
            "last"      => Some(Self::Last),
            "nth"       => Some(Self::Nth),
            "flatten"   => Some(Self::Flatten),
            "flat_map"  => Some(Self::FlatMap),
            "chunk"     => Some(Self::Chunk),
            "unique"    => Some(Self::Unique),
            "distinct"  => Some(Self::Distinct),
            "sort_by"   => Some(Self::SortBy),
            "join_str"  => Some(Self::JoinStr),
            "compact"   => Some(Self::Compact),
            "count"     => Some(Self::Count),
            "group_by"  => Some(Self::GroupBy),
            "count_by"  => Some(Self::CountBy),
            "index_by"  => Some(Self::IndexBy),
            "tally"     => Some(Self::Tally),
            "merge"     => Some(Self::Merge),
            "omit"      => Some(Self::Omit),
            "select"    => Some(Self::Select),
            "rename"    => Some(Self::Rename),
            "set"       => Some(Self::Set),
            "coalesce"  => Some(Self::Coalesce),
            "join"      => Some(Self::Join),
            "lookup"    => Some(Self::Lookup),
            "find"      => Some(Self::Find),
            "filter_by" => Some(Self::FilterBy),
            "get"       => Some(Self::Get),
            "resolve"   => Some(Self::Resolve),
            "deref"     => Some(Self::Deref),
            "pluck"     => Some(Self::Pluck),
            _ => None,
        }
    }
}

pub trait Callable {
    fn call(
        &mut self,
        func: &Func,
        value: &Value,
        ctx: Option<&mut Context<'_>>,
    ) -> Result<Value, Error>;
}

pub(crate) trait Registry: Callable {}

pub struct FuncRegistry {
    map: BTreeMap<String, Box<dyn Callable>>,
}

impl FuncRegistry {
    pub fn new() -> Self {
        Self {
            map: BTreeMap::new(),
        }
    }

    #[inline(always)]
    pub fn register<S: Into<String>>(&mut self, name: S, func: Box<dyn Callable>) {
        self.map.insert(name.into(), func);
    }

    #[inline(always)]
    pub fn get<S: Into<String>>(&mut self, name: S) -> Option<&mut Box<dyn Callable>> {
        self.map.get_mut(&name.into())
    }
}

impl Callable for FuncRegistry {
    fn call(
        &mut self,
        func: &Func,
        value: &Value,
        ctx: Option<&mut Context<'_>>,
    ) -> Result<Value, Error> {
        let target = if let Some(result) = self.get(&func.name) {
            result
        } else {
            return Err(Error::FuncEval(format!(
                "function '{}' not found",
                func.name
            )));
        };
        target.call(func, value, ctx)
    }
}

impl Registry for FuncRegistry {}

// ═══════════════════════════════════════════════════════════════════════════════
// Existing functions (preserved as-is)
// ═══════════════════════════════════════════════════════════════════════════════

struct Reverse;
impl Callable for Reverse {
    fn call(&mut self, _: &Func, value: &Value, _ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        match value {
            Value::Array(ref array) => {
                let mut v = array.clone();
                v.reverse();
                Ok(Value::Array(v))
            }
            _ => Err(Error::FuncEval("expected json value of type array".into())),
        }
    }
}

struct Formats;
impl Callable for Formats {
    fn call(&mut self, func: &Func, value: &Value, _ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        if func.args.len() < 2 {
            return Err(Error::FuncEval("deficit number of arguments".into()));
        }
        let format = match func.args.get(0).unwrap() {
            FuncArg::Key(k) => k,
            _ => return Err(Error::FuncEval("invalid type, expected string".into())),
        };
        if func.alias.is_none() {
            return Err(Error::FuncEval("expected alias to have some value".into()));
        }
        let mut args: Vec<String> = vec![];
        for v in func.args[1..].iter() {
            match v {
                FuncArg::Key(k) => args.push(k.clone()),
                _ => return Err(Error::FuncEval("invalid type, expected string".into())),
            };
        }
        let formater: Box<dyn crate::fmt::KeyFormater> = crate::fmt::default_formater();
        match formater.eval(
            &crate::context::FormatOp::FormatString {
                format: format.to_string(),
                arguments: args,
                alias: func.alias.clone().unwrap(),
            },
            value,
        ) {
            Some(output) => Ok(output),
            _ => Err(Error::FuncEval("format failed".into())),
        }
    }
}

struct SumFn;
impl Callable for SumFn {
    fn call(&mut self, _: &Func, value: &Value, ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        match value {
            Value::Object(_) => Err(Error::FuncEval("sum on objects is not implemented".into())),
            Value::Array(ref array) => {
                let mut sum = ctx.unwrap().reduce_stack_to_sum();
                for v in array {
                    if v.is_number() { sum.add(v); }
                }
                Ok(Value::Number(serde_json::Number::from(sum)))
            }
            _ => {
                let mut sum = ctx.unwrap().reduce_stack_to_sum();
                if value.is_number() { sum.add(value); }
                Ok(Value::Number(serde_json::Number::from(sum)))
            }
        }
    }
}

struct LenFn;
impl Callable for LenFn {
    fn call(&mut self, _: &Func, value: &Value, ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        match value {
            Value::Object(ref obj) => Ok(Value::Number(serde_json::Number::from(obj.len()))),
            Value::Array(ref array) => Ok(Value::Number(serde_json::Number::from(array.len()))),
            _ => {
                let count: i64 = ctx.unwrap().reduce_stack_to_num_count() + 1;
                Ok(Value::Number(serde_json::Number::from(count)))
            }
        }
    }
}

struct Head;
impl Callable for Head {
    fn call(&mut self, _: &Func, value: &Value, _ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        match value {
            Value::Array(ref array) => {
                if array.is_empty() { Ok(value.clone()) } else { Ok(array[0].clone()) }
            }
            _ => Err(Error::FuncEval("expected array".into())),
        }
    }
}

struct Tail;
impl Callable for Tail {
    fn call(&mut self, _: &Func, value: &Value, _ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        match value {
            Value::Array(ref array) => match array.len() {
                0 | 1 => Ok(Value::Array(vec![])),
                _ => Ok(Value::Array(array[1..].to_vec())),
            },
            _ => Err(Error::FuncEval("expected array".into())),
        }
    }
}

struct AllOnBoolean;
impl Callable for AllOnBoolean {
    fn call(&mut self, _: &Func, value: &Value, ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        match value {
            Value::Bool(ref v) => {
                let all = ctx.unwrap().reduce_stack_to_all_truth();
                Ok(Value::Bool(all & (*v == true)))
            }
            Value::Array(ref array) => {
                let mut all = ctx.unwrap().reduce_stack_to_all_truth();
                for v in array {
                    if v.is_boolean() { all = all & (v.as_bool().unwrap() == true); }
                }
                Ok(Value::Bool(all))
            }
            _ => Err(Error::FuncEval("input is not reducable".into())),
        }
    }
}

// ── MapFn — enhanced to support ObjConstruct and ArrConstruct args ─────────────

struct MapFn;
impl MapFn {
    fn eval_subpath(&mut self, value: &Value, subpath: &[Filter]) -> Result<Vec<Value>, Error> {
        let mut output: Vec<Value> = vec![];
        match value {
            Value::Array(ref array) => {
                for item in array {
                    let result = Path::collect_with_filter(item.clone(), subpath);
                    if result.0.is_empty() {
                        return Err(Error::FuncEval(
                            "map statement does not evaluate to anything".into(),
                        ));
                    }
                    output.push(result.0[0].clone());
                }
            }
            _ => {}
        };
        Ok(output)
    }

    fn eval_obj_construct(
        value: &Value,
        fields: &[(ObjKey, Vec<Filter>)],
    ) -> Result<Value, Error> {
        match value {
            Value::Array(ref array) => {
                let mut out: Vec<Value> = Vec::with_capacity(array.len());
                for item in array {
                    let mut map = serde_json::Map::new();
                    for (obj_key, val_filters) in fields {
                        let key = match obj_key {
                            ObjKey::Static(s) => s.clone(),
                            ObjKey::Dynamic(kf) => {
                                let r = Path::collect_with_filter(item.clone(), kf);
                                match r.0.into_iter().next() {
                                    Some(Value::String(s)) => s,
                                    Some(other) => {
                                        let raw = other.to_string();
                                        raw.trim_matches('"').to_string()
                                    }
                                    None => continue,
                                }
                            }
                        };
                        let r = Path::collect_with_filter(item.clone(), val_filters);
                        if let Some(v) = r.0.into_iter().next() {
                            map.insert(key, v);
                        }
                    }
                    out.push(Value::Object(map));
                }
                Ok(Value::Array(out))
            }
            _ => Err(Error::FuncEval("map: expected array".into())),
        }
    }

    fn eval_arr_construct(value: &Value, elems: &[Vec<Filter>]) -> Result<Value, Error> {
        match value {
            Value::Array(ref array) => {
                let mut out: Vec<Value> = Vec::with_capacity(array.len());
                for item in array {
                    let mut row: Vec<Value> = Vec::with_capacity(elems.len());
                    for ef in elems {
                        let r = Path::collect_with_filter(item.clone(), ef);
                        if let Some(v) = r.0.into_iter().next() {
                            row.push(v);
                        }
                    }
                    out.push(Value::Array(row));
                }
                Ok(Value::Array(out))
            }
            _ => Err(Error::FuncEval("map: expected array".into())),
        }
    }
}

impl Callable for MapFn {
    fn call(&mut self, func: &Func, value: &Value, _ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        match func.args.get(0) {
            Some(FuncArg::MapStmt(MapAST { arg: _, ref body })) => match body {
                MapBody::Method { .. } => {
                    Err(Error::FuncEval("WIP: method map not implemented".into()))
                }
                MapBody::Subpath(ref subpath) => {
                    let output = self.eval_subpath(value, subpath.as_slice())?;
                    Ok(Value::Array(output))
                }
                _ => Err(Error::FuncEval("expected method call on path".into())),
            },
            Some(FuncArg::ObjConstruct(ref fields)) => {
                MapFn::eval_obj_construct(value, fields)
            }
            Some(FuncArg::ArrConstruct(ref elems)) => {
                MapFn::eval_arr_construct(value, elems)
            }
            Some(FuncArg::SubExpr(ref filters)) => {
                // Allow #map(>/field) as shorthand
                let output = self.eval_subpath(value, filters.as_slice())?;
                Ok(Value::Array(output))
            }
            _ => Err(Error::FuncEval("expected first argument to be map statement".into())),
        }
    }
}

struct Keys;
impl Callable for Keys {
    fn call(&mut self, func: &Func, value: &Value, _ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        if func.args.len() > 0 {
            return Err(Error::FuncEval("keys takes no argument".into()));
        }
        match value {
            Value::Object(ref obj) => {
                let keys: Vec<Value> = obj.keys().map(|v| Value::String(v.to_string())).collect();
                Ok(Value::Array(keys))
            }
            _ => Err(Error::FuncEval("keys can be exec only on objects".into())),
        }
    }
}

struct Values;
impl Callable for Values {
    fn call(&mut self, func: &Func, value: &Value, _ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        if func.args.len() > 0 {
            return Err(Error::FuncEval("values takes no argument".into()));
        }
        match value {
            Value::Object(ref obj) => {
                let values: Vec<Value> = obj.values().cloned().collect();
                Ok(Value::Array(values))
            }
            _ => Err(Error::FuncEval("values can be exec only on objects".into())),
        }
    }
}

struct Min;
impl Callable for Min {
    fn call(&mut self, _: &Func, value: &Value, _ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        match value {
            Value::Array(ref array) => {
                let mut nums: Vec<f64> = array.iter().filter(|v| v.is_number()).map(|v| v.as_f64().unwrap()).collect();
                nums.sort_by(|a, b| a.partial_cmp(b).unwrap());
                match nums.first() {
                    Some(&n) => Ok(Value::Number(serde_json::Number::from_f64(n).unwrap())),
                    _ => Ok(Value::Array(vec![])),
                }
            }
            _ => Err(Error::FuncEval("expected value to be array".into())),
        }
    }
}

struct Max;
impl Callable for Max {
    fn call(&mut self, _: &Func, value: &Value, _ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        match value {
            Value::Array(ref array) => {
                let mut nums: Vec<f64> = array.iter().filter(|v| v.is_number()).map(|v| v.as_f64().unwrap()).collect();
                nums.sort_by(|a, b| b.partial_cmp(a).unwrap());
                match nums.first() {
                    Some(&n) => Ok(Value::Number(serde_json::Number::from_f64(n).unwrap())),
                    _ => Ok(Value::Array(vec![])),
                }
            }
            _ => Err(Error::FuncEval("expected value to be array".into())),
        }
    }
}

struct Zip;
impl Callable for Zip {
    fn call(&mut self, _: &Func, value: &Value, _ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        match value {
            Value::Object(ref obj) => {
                let mut stack: VecDeque<Value> = VecDeque::new();
                for (i, (k, vs)) in obj.iter().enumerate() {
                    if let Value::Array(ref array) = vs {
                        for (j, v) in array.iter().enumerate() {
                            if i == 0 {
                                let mut item = serde_json::Map::<String, Value>::new();
                                item.insert(k.clone(), v.clone());
                                stack.push_back(Value::Object(item));
                            } else {
                                let entry = stack.remove(j);
                                if let Some(Value::Object(mut obj)) = entry {
                                    obj.insert(k.clone(), v.clone());
                                    stack.insert(j, Value::Object(obj));
                                }
                            }
                        }
                    }
                }
                Ok(Value::Array(Vec::from(stack)))
            }
            _ => Err(Error::Eval("nothing to zip".into())),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ── Math ──────────────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════════

/// Extract the first numeric arg from a Func, returning f64.
fn first_num_arg(func: &Func) -> Result<f64, Error> {
    match func.args.first() {
        Some(FuncArg::Number(n)) => Ok(*n),
        Some(FuncArg::Key(s)) => s.parse::<f64>().map_err(|_| Error::FuncEval(format!("expected number, got '{}'", s))),
        _ => Err(Error::FuncEval("expected numeric argument".into())),
    }
}

fn value_as_f64(v: &Value) -> Option<f64> {
    if let Some(n) = v.as_f64() { Some(n) } else { None }
}

struct Avg;
impl Callable for Avg {
    fn call(&mut self, _: &Func, value: &Value, _ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        match value {
            Value::Array(ref array) => {
                let nums: Vec<f64> = array.iter().filter_map(value_as_f64).collect();
                if nums.is_empty() {
                    return Ok(Value::Null);
                }
                let avg = nums.iter().sum::<f64>() / nums.len() as f64;
                Ok(Value::Number(serde_json::Number::from_f64(avg).unwrap()))
            }
            _ if value.is_number() => Ok(value.clone()),
            _ => Err(Error::FuncEval("avg: expected array of numbers".into())),
        }
    }
}

struct AddFn;
impl Callable for AddFn {
    fn call(&mut self, func: &Func, value: &Value, _ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        let n = first_num_arg(func)?;
        match value_as_f64(value) {
            Some(v) => Ok(Value::Number(serde_json::Number::from_f64(v + n).unwrap())),
            _ => Err(Error::FuncEval("add: expected number".into())),
        }
    }
}

struct SubFn;
impl Callable for SubFn {
    fn call(&mut self, func: &Func, value: &Value, _ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        let n = first_num_arg(func)?;
        match value_as_f64(value) {
            Some(v) => Ok(Value::Number(serde_json::Number::from_f64(v - n).unwrap())),
            _ => Err(Error::FuncEval("sub: expected number".into())),
        }
    }
}

struct MulFn;
impl Callable for MulFn {
    fn call(&mut self, func: &Func, value: &Value, _ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        let n = first_num_arg(func)?;
        match value_as_f64(value) {
            Some(v) => Ok(Value::Number(serde_json::Number::from_f64(v * n).unwrap())),
            _ => Err(Error::FuncEval("mul: expected number".into())),
        }
    }
}

struct DivFn;
impl Callable for DivFn {
    fn call(&mut self, func: &Func, value: &Value, _ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        let n = first_num_arg(func)?;
        if n == 0.0 {
            return Err(Error::FuncEval("div: division by zero".into()));
        }
        match value_as_f64(value) {
            Some(v) => Ok(Value::Number(serde_json::Number::from_f64(v / n).unwrap())),
            _ => Err(Error::FuncEval("div: expected number".into())),
        }
    }
}

struct AbsFn;
impl Callable for AbsFn {
    fn call(&mut self, _: &Func, value: &Value, _ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        match value_as_f64(value) {
            Some(v) => Ok(Value::Number(serde_json::Number::from_f64(v.abs()).unwrap())),
            _ => Err(Error::FuncEval("abs: expected number".into())),
        }
    }
}

struct RoundFn;
impl Callable for RoundFn {
    fn call(&mut self, _: &Func, value: &Value, _ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        match value_as_f64(value) {
            Some(v) => Ok(Value::Number(serde_json::Number::from_f64(v.round()).unwrap())),
            _ => Err(Error::FuncEval("round: expected number".into())),
        }
    }
}

struct FloorFn;
impl Callable for FloorFn {
    fn call(&mut self, _: &Func, value: &Value, _ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        match value_as_f64(value) {
            Some(v) => Ok(Value::Number(serde_json::Number::from_f64(v.floor()).unwrap())),
            _ => Err(Error::FuncEval("floor: expected number".into())),
        }
    }
}

struct CeilFn;
impl Callable for CeilFn {
    fn call(&mut self, _: &Func, value: &Value, _ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        match value_as_f64(value) {
            Some(v) => Ok(Value::Number(serde_json::Number::from_f64(v.ceil()).unwrap())),
            _ => Err(Error::FuncEval("ceil: expected number".into())),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ── Boolean ───────────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════════

struct AnyFn;
impl Callable for AnyFn {
    fn call(&mut self, _: &Func, value: &Value, _ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        match value {
            Value::Array(ref array) => {
                let any = array.iter().any(|v| v.as_bool().unwrap_or(false));
                Ok(Value::Bool(any))
            }
            Value::Bool(b) => Ok(Value::Bool(*b)),
            _ => Err(Error::FuncEval("any: expected array of booleans".into())),
        }
    }
}

struct NotFn;
impl Callable for NotFn {
    fn call(&mut self, _: &Func, value: &Value, _ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        match value {
            Value::Bool(b) => Ok(Value::Bool(!b)),
            _ => Err(Error::FuncEval("not: expected boolean".into())),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ── Array transforms ──────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════════

struct LastFn;
impl Callable for LastFn {
    fn call(&mut self, _: &Func, value: &Value, _ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        match value {
            Value::Array(ref array) => {
                match array.last() {
                    Some(v) => Ok(v.clone()),
                    None => Ok(Value::Array(vec![])),
                }
            }
            _ => Err(Error::FuncEval("last: expected array".into())),
        }
    }
}

struct NthFn;
impl Callable for NthFn {
    fn call(&mut self, func: &Func, value: &Value, _ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        let idx = first_num_arg(func)? as usize;
        match value {
            Value::Array(ref array) => {
                match array.get(idx) {
                    Some(v) => Ok(v.clone()),
                    None => Ok(Value::Null),
                }
            }
            _ => Err(Error::FuncEval("nth: expected array".into())),
        }
    }
}

struct FlattenFn;
impl Callable for FlattenFn {
    fn call(&mut self, _: &Func, value: &Value, _ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        match value {
            Value::Array(ref array) => {
                let mut out: Vec<Value> = Vec::new();
                for item in array {
                    match item {
                        Value::Array(ref inner) => out.extend(inner.clone()),
                        other => out.push(other.clone()),
                    }
                }
                Ok(Value::Array(out))
            }
            _ => Err(Error::FuncEval("flatten: expected array".into())),
        }
    }
}

struct FlatMapFn;
impl FlatMapFn {
    fn map_and_flatten(value: &Value, subpath: &[Filter]) -> Result<Value, Error> {
        match value {
            Value::Array(ref array) => {
                let mut out: Vec<Value> = Vec::new();
                for item in array {
                    let r = Path::collect_with_filter(item.clone(), subpath);
                    for v in r.0 {
                        match v {
                            Value::Array(inner) => out.extend(inner),
                            other => out.push(other),
                        }
                    }
                }
                Ok(Value::Array(out))
            }
            _ => Err(Error::FuncEval("flat_map: expected array".into())),
        }
    }
}

impl Callable for FlatMapFn {
    fn call(&mut self, func: &Func, value: &Value, _ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        match func.args.first() {
            Some(FuncArg::MapStmt(ref ast)) => {
                if let super::context::MapBody::Subpath(ref sp) = ast.body {
                    FlatMapFn::map_and_flatten(value, sp)
                } else {
                    Err(Error::FuncEval("flat_map: expected subpath body".into()))
                }
            }
            Some(FuncArg::SubExpr(ref filters)) => FlatMapFn::map_and_flatten(value, filters),
            _ => Err(Error::FuncEval("flat_map: expected map statement or subexpr".into())),
        }
    }
}

struct ChunkFn;
impl Callable for ChunkFn {
    fn call(&mut self, func: &Func, value: &Value, _ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        let size = first_num_arg(func)? as usize;
        if size == 0 {
            return Err(Error::FuncEval("chunk: size must be > 0".into()));
        }
        match value {
            Value::Array(ref array) => {
                let chunks: Vec<Value> = array.chunks(size).map(|c| Value::Array(c.to_vec())).collect();
                Ok(Value::Array(chunks))
            }
            _ => Err(Error::FuncEval("chunk: expected array".into())),
        }
    }
}

struct UniqueFn;
impl Callable for UniqueFn {
    fn call(&mut self, _: &Func, value: &Value, _ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        match value {
            Value::Array(ref array) => {
                let mut seen: Vec<String> = Vec::new();
                let mut out: Vec<Value> = Vec::new();
                for v in array {
                    let key = v.to_string();
                    if !seen.contains(&key) {
                        seen.push(key);
                        out.push(v.clone());
                    }
                }
                Ok(Value::Array(out))
            }
            _ => Err(Error::FuncEval("unique: expected array".into())),
        }
    }
}

struct DistinctFn;
impl Callable for DistinctFn {
    fn call(&mut self, func: &Func, value: &Value, _ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        let key = match func.args.first() {
            Some(FuncArg::Key(k)) => k.clone(),
            _ => return Err(Error::FuncEval("distinct: expected key argument".into())),
        };
        match value {
            Value::Array(ref array) => {
                let mut seen: Vec<String> = Vec::new();
                let mut out: Vec<Value> = Vec::new();
                for v in array {
                    if let Some(field) = v.get(&key) {
                        let k = field.to_string();
                        if !seen.contains(&k) {
                            seen.push(k);
                            out.push(v.clone());
                        }
                    }
                }
                Ok(Value::Array(out))
            }
            _ => Err(Error::FuncEval("distinct: expected array".into())),
        }
    }
}

struct SortByFn;
impl Callable for SortByFn {
    fn call(&mut self, func: &Func, value: &Value, _ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        let key = match func.args.first() {
            Some(FuncArg::Key(k)) => k.clone(),
            _ => return Err(Error::FuncEval("sort_by: expected key argument".into())),
        };
        let descending = match func.args.get(1) {
            Some(FuncArg::Key(dir)) => dir == "desc",
            _ => false,
        };
        match value {
            Value::Array(ref array) => {
                let mut v = array.clone();
                v.sort_by(|a, b| {
                    let av = a.get(&key).and_then(value_as_f64);
                    let bv = b.get(&key).and_then(value_as_f64);
                    let order = match (av, bv) {
                        (Some(an), Some(bn)) => an.partial_cmp(&bn).unwrap_or(std::cmp::Ordering::Equal),
                        _ => a.get(&key).and_then(Value::as_str).cmp(&b.get(&key).and_then(Value::as_str)),
                    };
                    if descending { order.reverse() } else { order }
                });
                Ok(Value::Array(v))
            }
            _ => Err(Error::FuncEval("sort_by: expected array".into())),
        }
    }
}

struct JoinStrFn;
impl Callable for JoinStrFn {
    fn call(&mut self, func: &Func, value: &Value, _ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        let sep = match func.args.first() {
            Some(FuncArg::Key(s)) => s.clone(),
            _ => String::new(),
        };
        match value {
            Value::Array(ref array) => {
                let parts: Vec<String> = array.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect();
                Ok(Value::String(parts.join(&sep)))
            }
            _ => Err(Error::FuncEval("join_str: expected array".into())),
        }
    }
}

struct CompactFn;
impl Callable for CompactFn {
    fn call(&mut self, _: &Func, value: &Value, _ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        match value {
            Value::Array(ref array) => {
                let out: Vec<Value> = array.iter().filter(|v| !v.is_null()).cloned().collect();
                Ok(Value::Array(out))
            }
            _ => Err(Error::FuncEval("compact: expected array".into())),
        }
    }
}

/// #count — alias for #len
struct CountFn;
impl Callable for CountFn {
    fn call(&mut self, func: &Func, value: &Value, ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        LenFn.call(func, value, ctx)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ── Grouping / indexing ───────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════════

fn first_key_arg(func: &Func) -> Result<String, Error> {
    match func.args.first() {
        Some(FuncArg::Key(k)) => Ok(k.clone()),
        _ => Err(Error::FuncEval("expected string key argument".into())),
    }
}

struct GroupByFn;
impl Callable for GroupByFn {
    fn call(&mut self, func: &Func, value: &Value, _ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        let key = first_key_arg(func)?;
        match value {
            Value::Array(ref array) => {
                let mut map: serde_json::Map<String, Value> = serde_json::Map::new();
                for item in array {
                    if let Some(field) = item.get(&key) {
                        let group_key = match field {
                            Value::String(s) => s.clone(),
                            other => other.to_string().trim_matches('"').to_string(),
                        };
                        let entry = map.entry(group_key).or_insert_with(|| Value::Array(vec![]));
                        if let Value::Array(ref mut arr) = entry {
                            arr.push(item.clone());
                        }
                    }
                }
                Ok(Value::Object(map))
            }
            _ => Err(Error::FuncEval("group_by: expected array".into())),
        }
    }
}

struct CountByFn;
impl Callable for CountByFn {
    fn call(&mut self, func: &Func, value: &Value, _ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        let key = first_key_arg(func)?;
        match value {
            Value::Array(ref array) => {
                let mut map: serde_json::Map<String, Value> = serde_json::Map::new();
                for item in array {
                    if let Some(field) = item.get(&key) {
                        let group_key = match field {
                            Value::String(s) => s.clone(),
                            other => other.to_string().trim_matches('"').to_string(),
                        };
                        let entry = map.entry(group_key).or_insert(Value::Number(serde_json::Number::from(0i64)));
                        let prev = entry.as_i64().unwrap_or(0);
                        *entry = Value::Number(serde_json::Number::from(prev + 1));
                    }
                }
                Ok(Value::Object(map))
            }
            _ => Err(Error::FuncEval("count_by: expected array".into())),
        }
    }
}

struct IndexByFn;
impl Callable for IndexByFn {
    fn call(&mut self, func: &Func, value: &Value, _ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        let key = first_key_arg(func)?;
        match value {
            Value::Array(ref array) => {
                let mut map: serde_json::Map<String, Value> = serde_json::Map::new();
                for item in array {
                    if let Some(field) = item.get(&key) {
                        let k = match field {
                            Value::String(s) => s.clone(),
                            other => other.to_string().trim_matches('"').to_string(),
                        };
                        map.insert(k, item.clone());
                    }
                }
                Ok(Value::Object(map))
            }
            _ => Err(Error::FuncEval("index_by: expected array".into())),
        }
    }
}

struct TallyFn;
impl Callable for TallyFn {
    fn call(&mut self, _: &Func, value: &Value, _ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        match value {
            Value::Array(ref array) => {
                let mut map: serde_json::Map<String, Value> = serde_json::Map::new();
                for item in array {
                    let k = match item {
                        Value::String(s) => s.clone(),
                        other => other.to_string().trim_matches('"').to_string(),
                    };
                    let entry = map.entry(k).or_insert(Value::Number(serde_json::Number::from(0i64)));
                    let prev = entry.as_i64().unwrap_or(0);
                    *entry = Value::Number(serde_json::Number::from(prev + 1));
                }
                Ok(Value::Object(map))
            }
            _ => Err(Error::FuncEval("tally: expected array".into())),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ── Object manipulation ───────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════════

struct MergeFn;
impl Callable for MergeFn {
    fn call(&mut self, _: &Func, value: &Value, _ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        match value {
            Value::Array(ref array) => {
                let mut out: serde_json::Map<String, Value> = serde_json::Map::new();
                for item in array {
                    if let Value::Object(ref obj) = item {
                        for (k, v) in obj {
                            out.insert(k.clone(), v.clone());
                        }
                    }
                }
                Ok(Value::Object(out))
            }
            Value::Object(_) => Ok(value.clone()),
            _ => Err(Error::FuncEval("merge: expected array of objects".into())),
        }
    }
}

struct OmitFn;
impl Callable for OmitFn {
    fn call(&mut self, func: &Func, value: &Value, _ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        let keys: Vec<String> = func.args.iter().filter_map(|a| {
            if let FuncArg::Key(k) = a { Some(k.clone()) } else { None }
        }).collect();
        let omit_keys = |obj: &serde_json::Map<String, Value>| -> Value {
            let filtered: serde_json::Map<_, _> = obj.iter()
                .filter(|(k, _)| !keys.contains(k))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            Value::Object(filtered)
        };
        match value {
            Value::Object(ref obj) => Ok(omit_keys(obj)),
            Value::Array(ref array) => {
                let out: Vec<Value> = array.iter().filter_map(|item| {
                    if let Value::Object(ref obj) = item { Some(omit_keys(obj)) } else { None }
                }).collect();
                Ok(Value::Array(out))
            }
            _ => Err(Error::FuncEval("omit: expected object or array".into())),
        }
    }
}

struct SelectFn;
impl Callable for SelectFn {
    fn call(&mut self, func: &Func, value: &Value, _ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        let keys: Vec<String> = func.args.iter().filter_map(|a| {
            if let FuncArg::Key(k) = a { Some(k.clone()) } else { None }
        }).collect();
        let select_keys = |obj: &serde_json::Map<String, Value>| -> Value {
            let filtered: serde_json::Map<_, _> = obj.iter()
                .filter(|(k, _)| keys.contains(k))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            Value::Object(filtered)
        };
        match value {
            Value::Object(ref obj) => Ok(select_keys(obj)),
            Value::Array(ref array) => {
                let out: Vec<Value> = array.iter().filter_map(|item| {
                    if let Value::Object(ref obj) = item { Some(select_keys(obj)) } else { None }
                }).collect();
                Ok(Value::Array(out))
            }
            _ => Err(Error::FuncEval("select: expected object or array".into())),
        }
    }
}

struct RenameFn;
impl Callable for RenameFn {
    fn call(&mut self, func: &Func, value: &Value, _ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        let (from, to) = match (func.args.get(0), func.args.get(1)) {
            (Some(FuncArg::Key(f)), Some(FuncArg::Key(t))) => (f.clone(), t.clone()),
            _ => return Err(Error::FuncEval("rename: expected two key arguments (from, to)".into())),
        };
        let rename = |obj: &serde_json::Map<String, Value>| -> Value {
            let renamed: serde_json::Map<_, _> = obj.iter().map(|(k, v)| {
                if k == &from { (to.clone(), v.clone()) } else { (k.clone(), v.clone()) }
            }).collect();
            Value::Object(renamed)
        };
        match value {
            Value::Object(ref obj) => Ok(rename(obj)),
            Value::Array(ref array) => {
                let out: Vec<Value> = array.iter().filter_map(|item| {
                    if let Value::Object(ref obj) = item { Some(rename(obj)) } else { None }
                }).collect();
                Ok(Value::Array(out))
            }
            _ => Err(Error::FuncEval("rename: expected object or array".into())),
        }
    }
}

struct SetFn;
impl Callable for SetFn {
    fn call(&mut self, func: &Func, value: &Value, _ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        let (key, val) = match (func.args.get(0), func.args.get(1)) {
            (Some(FuncArg::Key(k)), Some(FuncArg::Key(v))) => (k.clone(), Value::String(v.clone())),
            (Some(FuncArg::Key(k)), Some(FuncArg::Number(n))) => (k.clone(), Value::Number(serde_json::Number::from_f64(*n).unwrap())),
            _ => return Err(Error::FuncEval("set: expected key and value arguments".into())),
        };
        let insert = |obj: &serde_json::Map<String, Value>| -> Value {
            let mut m = obj.clone();
            m.insert(key.clone(), val.clone());
            Value::Object(m)
        };
        match value {
            Value::Object(ref obj) => Ok(insert(obj)),
            Value::Array(ref array) => {
                let out: Vec<Value> = array.iter().filter_map(|item| {
                    if let Value::Object(ref obj) = item { Some(insert(obj)) } else { None }
                }).collect();
                Ok(Value::Array(out))
            }
            _ => Err(Error::FuncEval("set: expected object or array".into())),
        }
    }
}

struct CoalesceFn;
impl Callable for CoalesceFn {
    fn call(&mut self, func: &Func, value: &Value, _ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        let is_empty = match value {
            Value::Null => true,
            Value::Array(ref a) => a.is_empty(),
            Value::String(ref s) => s.is_empty(),
            _ => false,
        };
        if !is_empty {
            return Ok(value.clone());
        }
        match func.args.first() {
            Some(FuncArg::Key(k)) => Ok(Value::String(k.clone())),
            Some(FuncArg::Number(n)) => Ok(Value::Number(serde_json::Number::from_f64(*n).unwrap())),
            _ => Ok(Value::Null),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ── Join operations ───────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════════

/// Resolve the sub-expression argument against the context root.
fn resolve_subexpr(func: &Func, arg_idx: usize, ctx: &Context<'_>) -> Result<Vec<Value>, Error> {
    match func.args.get(arg_idx) {
        Some(FuncArg::SubExpr(ref filters)) => {
            let r = Path::collect_with_filter(ctx.root.clone(), filters);
            Ok(r.0)
        }
        _ => Err(Error::FuncEval(format!("expected sub-expression at arg {}", arg_idx))),
    }
}

fn resolve_subexpr_to_array(func: &Func, arg_idx: usize, ctx: &Context<'_>) -> Result<Vec<Value>, Error> {
    let vals = resolve_subexpr(func, arg_idx, ctx)?;
    match vals.into_iter().next() {
        Some(Value::Array(arr)) => Ok(arr),
        Some(other) => Ok(vec![other]),
        None => Ok(vec![]),
    }
}

/// #join(>/path/to/other, 'left_key', 'right_key')
/// Inner join: for each item in left array, find all items in right where left[lk] == right[rk].
/// Result: array of {"left": item, "right": matched_item} objects.
struct JoinFn;
impl Callable for JoinFn {
    fn call(&mut self, func: &Func, value: &Value, ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        let ctx = ctx.ok_or_else(|| Error::FuncEval("join: requires context".into()))?;
        let left_key = match func.args.get(1) {
            Some(FuncArg::Key(k)) => k.clone(),
            _ => return Err(Error::FuncEval("join: expected left key at arg 1".into())),
        };
        let right_key = match func.args.get(2) {
            Some(FuncArg::Key(k)) => k.clone(),
            _ => return Err(Error::FuncEval("join: expected right key at arg 2".into())),
        };
        let right = resolve_subexpr_to_array(func, 0, ctx)?;
        let left_arr = match value {
            Value::Array(a) => a.clone(),
            _ => return Err(Error::FuncEval("join: expected array on left side".into())),
        };
        let mut out: Vec<Value> = Vec::new();
        for left_item in &left_arr {
            let lv = left_item.get(&left_key);
            for right_item in &right {
                let rv = right_item.get(&right_key);
                if lv == rv {
                    let mut merged = serde_json::Map::new();
                    if let Value::Object(ref lo) = left_item { for (k, v) in lo { merged.insert(k.clone(), v.clone()); } }
                    if let Value::Object(ref ro) = right_item {
                        for (k, v) in ro {
                            if !merged.contains_key(k) { merged.insert(k.clone(), v.clone()); }
                        }
                    }
                    out.push(Value::Object(merged));
                }
            }
        }
        Ok(Value::Array(out))
    }
}

/// #lookup(>/path/to/other, 'left_key', 'right_key')
/// For each item in the left array, find the FIRST match in the right collection and
/// merge it. Returns array of merged objects (null-patched where no match found).
struct LookupFn;
impl Callable for LookupFn {
    fn call(&mut self, func: &Func, value: &Value, ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        let ctx = ctx.ok_or_else(|| Error::FuncEval("lookup: requires context".into()))?;
        let left_key = match func.args.get(1) {
            Some(FuncArg::Key(k)) => k.clone(),
            _ => return Err(Error::FuncEval("lookup: expected left key at arg 1".into())),
        };
        let right_key = match func.args.get(2) {
            Some(FuncArg::Key(k)) => k.clone(),
            _ => return Err(Error::FuncEval("lookup: expected right key at arg 2".into())),
        };
        let right = resolve_subexpr_to_array(func, 0, ctx)?;
        let left_arr = match value {
            Value::Array(a) => a.clone(),
            other => vec![other.clone()],
        };
        let mut out: Vec<Value> = Vec::new();
        for left_item in &left_arr {
            let lv = left_item.get(&left_key);
            let matched = right.iter().find(|ri| ri.get(&right_key) == lv);
            let mut merged = serde_json::Map::new();
            if let Value::Object(ref lo) = left_item { for (k, v) in lo { merged.insert(k.clone(), v.clone()); } }
            if let Some(Value::Object(ref ro)) = matched {
                for (k, v) in ro {
                    if !merged.contains_key(k) { merged.insert(k.clone(), v.clone()); }
                }
            }
            out.push(Value::Object(merged));
        }
        Ok(Value::Array(out))
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ── Field-resolution functions ────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════════

/// #find(condition)
/// Returns the first element in the array that satisfies the given filter condition.
/// Example: >/users/#find('age' > 18)
struct FindFn;
impl Callable for FindFn {
    fn call(&mut self, func: &Func, value: &Value, _ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        let ast = match func.args.get(0) {
            Some(FuncArg::FilterExpr(ast)) => ast.clone(),
            _ => return Err(Error::FuncEval("find: expected filter condition argument".into())),
        };
        let arr = match value {
            Value::Array(a) => a,
            _ => return Err(Error::FuncEval("find: expected array input".into())),
        };
        match arr.iter().find(|item| ast.eval(item)) {
            Some(v) => Ok(v.clone()),
            None => Ok(Value::Null),
        }
    }
}

/// #filter_by(condition)
/// Returns all elements in the array that satisfy the given filter condition.
/// Example: >/users/#filter_by('age' > 18)
struct FilterByFn;
impl Callable for FilterByFn {
    fn call(&mut self, func: &Func, value: &Value, _ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        let ast = match func.args.get(0) {
            Some(FuncArg::FilterExpr(ast)) => ast.clone(),
            _ => return Err(Error::FuncEval("filter_by: expected filter condition argument".into())),
        };
        let arr = match value {
            Value::Array(a) => a,
            _ => return Err(Error::FuncEval("filter_by: expected array input".into())),
        };
        Ok(Value::Array(arr.iter().filter(|item| ast.eval(item)).cloned().collect()))
    }
}

/// #get('key')
/// Gets a single field from the current object by key. Useful in pipelines.
/// Example: >/user/#get('name')
struct GetFn;
impl Callable for GetFn {
    fn call(&mut self, func: &Func, value: &Value, _ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        let key = match func.args.get(0) {
            Some(FuncArg::Key(k)) => k.clone(),
            _ => return Err(Error::FuncEval("get: expected string key argument".into())),
        };
        match value {
            Value::Object(ref obj) => match obj.get(&key) {
                Some(v) => Ok(v.clone()),
                None => Ok(Value::Null),
            },
            _ => Err(Error::FuncEval("get: expected object input".into())),
        }
    }
}

/// #resolve('ref_field', >/target [, 'match_field'])
///
/// For each object in the input array, look up the value of `ref_field`, find the
/// first item in `target` where `target.match_field == item.ref_field`, and merge the
/// found object into the current item under key `ref_field` (replacing the scalar).
///
/// If `match_field` is omitted it defaults to `ref_field`.
///
/// Example (resolve order.customer_id against customers array on 'id'):
///   >/orders/#resolve('customer_id', >/customers, 'id')
struct ResolveFn;
impl Callable for ResolveFn {
    fn call(&mut self, func: &Func, value: &Value, ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        let ctx = ctx.ok_or_else(|| Error::FuncEval("resolve: requires context".into()))?;
        let ref_field = match func.args.get(0) {
            Some(FuncArg::Key(k)) => k.clone(),
            _ => return Err(Error::FuncEval("resolve: expected ref_field string at arg 0".into())),
        };
        let target = resolve_subexpr_to_array(func, 1, ctx)?;
        let match_field = match func.args.get(2) {
            Some(FuncArg::Key(k)) => k.clone(),
            _ => ref_field.clone(),
        };

        let process = |item: &Value| -> Value {
            let ref_val = match item.get(&ref_field) {
                Some(v) => v,
                None => return item.clone(),
            };
            let matched = target.iter().find(|t| t.get(&match_field) == Some(ref_val));
            match matched {
                None => item.clone(),
                Some(found) => {
                    let mut merged = serde_json::Map::new();
                    if let Value::Object(ref o) = item { for (k, v) in o { merged.insert(k.clone(), v.clone()); } }
                    // Replace the scalar ref_field with the resolved object
                    merged.insert(ref_field.clone(), found.clone());
                    Value::Object(merged)
                }
            }
        };

        match value {
            Value::Array(ref arr) => Ok(Value::Array(arr.iter().map(process).collect())),
            single => Ok(process(single)),
        }
    }
}

/// #deref(>/target [, 'match_field'])
///
/// The current value IS the reference key. Find the first item in `target` where
/// `target.match_field == current_value`. If `match_field` is omitted and the target
/// contains objects, the first object with any field equal to the current value is
/// returned. If the target is an array of scalars, returns the first equal element.
///
/// Example (current item is an ID string, resolve to full user object):
///   >/order/customer_id/#deref(>/customers, 'id')
struct DerefFn;
impl Callable for DerefFn {
    fn call(&mut self, func: &Func, value: &Value, ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        let ctx = ctx.ok_or_else(|| Error::FuncEval("deref: requires context".into()))?;
        let target = resolve_subexpr_to_array(func, 0, ctx)?;
        let match_field = match func.args.get(1) {
            Some(FuncArg::Key(k)) => Some(k.clone()),
            _ => None,
        };
        let result = match match_field {
            Some(field) => target.into_iter().find(|t| t.get(&field) == Some(value)),
            None => {
                // No field specified: for objects scan all values, for scalars direct eq
                target.into_iter().find(|t| match t {
                    Value::Object(ref o) => o.values().any(|v| v == value),
                    other => other == value,
                })
            }
        };
        Ok(result.unwrap_or(Value::Null))
    }
}

/// #pluck('field')
/// Extract a single field from every object in an array (like map + get).
/// Example: >/users/#pluck('name')  →  ["Alice", "Bob", ...]
struct PluckFn;
impl Callable for PluckFn {
    fn call(&mut self, func: &Func, value: &Value, _ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
        let key = match func.args.get(0) {
            Some(FuncArg::Key(k)) => k.clone(),
            _ => return Err(Error::FuncEval("pluck: expected string key argument".into())),
        };
        match value {
            Value::Array(ref arr) => {
                let out: Vec<Value> = arr.iter()
                    .filter_map(|item| item.get(&key).cloned())
                    .collect();
                Ok(Value::Array(out))
            }
            Value::Object(ref obj) => match obj.get(&key) {
                Some(v) => Ok(v.clone()),
                None => Ok(Value::Null),
            },
            _ => Err(Error::FuncEval("pluck: expected array or object input".into())),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ── Registry default ──────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════════

impl Default for FuncRegistry {
    fn default() -> Self {
        let mut r = FuncRegistry::new();

        // ── Existing ──────────────────────────────────────────────────────────
        r.register("reverse",  Box::new(Reverse));
        r.register("formats",  Box::new(Formats));
        r.register("sum",      Box::new(SumFn));
        r.register("len",      Box::new(LenFn));
        r.register("head",     Box::new(Head));
        r.register("tail",     Box::new(Tail));
        r.register("all",      Box::new(AllOnBoolean));
        r.register("map",      Box::new(MapFn));
        r.register("keys",     Box::new(Keys));
        r.register("values",   Box::new(Values));
        r.register("min",      Box::new(Min));
        r.register("max",      Box::new(Max));
        r.register("zip",      Box::new(Zip));

        // ── Math ──────────────────────────────────────────────────────────────
        r.register("avg",      Box::new(Avg));
        r.register("add",      Box::new(AddFn));
        r.register("sub",      Box::new(SubFn));
        r.register("mul",      Box::new(MulFn));
        r.register("div",      Box::new(DivFn));
        r.register("abs",      Box::new(AbsFn));
        r.register("round",    Box::new(RoundFn));
        r.register("floor",    Box::new(FloorFn));
        r.register("ceil",     Box::new(CeilFn));

        // ── Boolean ───────────────────────────────────────────────────────────
        r.register("any",      Box::new(AnyFn));
        r.register("not",      Box::new(NotFn));

        // ── Array transforms ──────────────────────────────────────────────────
        r.register("last",     Box::new(LastFn));
        r.register("nth",      Box::new(NthFn));
        r.register("flatten",  Box::new(FlattenFn));
        r.register("flat_map", Box::new(FlatMapFn));
        r.register("chunk",    Box::new(ChunkFn));
        r.register("unique",   Box::new(UniqueFn));
        r.register("distinct", Box::new(DistinctFn));
        r.register("sort_by",  Box::new(SortByFn));
        r.register("join_str", Box::new(JoinStrFn));
        r.register("compact",  Box::new(CompactFn));
        r.register("count",    Box::new(CountFn));

        // ── Grouping / indexing ───────────────────────────────────────────────
        r.register("group_by", Box::new(GroupByFn));
        r.register("count_by", Box::new(CountByFn));
        r.register("index_by", Box::new(IndexByFn));
        r.register("tally",    Box::new(TallyFn));

        // ── Object manipulation ───────────────────────────────────────────────
        r.register("merge",    Box::new(MergeFn));
        r.register("omit",     Box::new(OmitFn));
        r.register("select",   Box::new(SelectFn));
        r.register("rename",   Box::new(RenameFn));
        r.register("set",      Box::new(SetFn));
        r.register("coalesce", Box::new(CoalesceFn));

        // ── Join operations ───────────────────────────────────────────────────
        r.register("join",      Box::new(JoinFn));
        r.register("lookup",    Box::new(LookupFn));

        // ── Field resolution ──────────────────────────────────────────────────
        r.register("find",      Box::new(FindFn));
        r.register("filter_by", Box::new(FilterByFn));
        r.register("get",       Box::new(GetFn));
        r.register("resolve",   Box::new(ResolveFn));
        r.register("deref",     Box::new(DerefFn));
        r.register("pluck",     Box::new(PluckFn));

        r
    }
}

pub(crate) fn default_registry() -> Rc<RefCell<impl Registry>> {
    let output: FuncRegistry = Default::default();
    Rc::new(RefCell::new(output))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod test {
    use crate::context::FuncArg;
    use super::*;

    #[test]
    fn insert_and_fetch() {
        const NAME: &'static str = "foo";
        let mut m = FuncRegistry::new();
        m.register(NAME, Box::new(Reverse));
        assert_eq!(m.map.len(), 1);
        let v = m.get(NAME);
        assert_eq!(v.is_none(), false);
        let d = v.unwrap().as_mut();
        let serde_value = Value::Array(vec![Value::Bool(false), Value::Bool(true)]);
        let mut func = Func::new();
        func.args.push(FuncArg::Key("test".to_owned()));
        match d.call(&func, &serde_value, None) {
            Ok(result) => match result {
                Value::Array(ref array) => {
                    assert_eq!(*array, vec![Value::Bool(true), Value::Bool(false)]);
                }
                _ => panic!("invalid types"),
            },
            Err(err) => panic!("{}", err),
        };
    }

    #[test]
    fn get_head() {
        const NAME: &'static str = "head";
        let mut m = FuncRegistry::new();
        m.register("head", Box::new(Head));
        let v = m.get(NAME).unwrap().as_mut();
        let serde_value = Value::Array(vec![
            Value::String("foo".to_string()),
            Value::String("bar".to_string()),
        ]);
        let func = Func::new();
        match v.call(&func, &serde_value, None) {
            Ok(result) => match result {
                Value::String(ref output) => {
                    assert_eq!(*output, "foo".to_string());
                }
                _ => panic!("invalid types {:?}", result),
            },
            Err(err) => panic!("{}", err),
        }
    }

    #[test]
    fn test_avg() {
        let data = Value::Array(vec![
            Value::Number(serde_json::Number::from(10i64)),
            Value::Number(serde_json::Number::from(20i64)),
            Value::Number(serde_json::Number::from(30i64)),
        ]);
        let mut avg = Avg;
        let func = Func::new();
        let result = avg.call(&func, &data, None).unwrap();
        assert_eq!(result.as_f64(), Some(20.0));
    }

    #[test]
    fn test_group_by() {
        let data = serde_json::json!([
            {"type": "a", "v": 1},
            {"type": "b", "v": 2},
            {"type": "a", "v": 3},
        ]);
        let mut f = GroupByFn;
        let mut func = Func::new();
        func.args.push(FuncArg::Key("type".into()));
        let result = f.call(&func, &data, None).unwrap();
        let obj = result.as_object().unwrap();
        assert_eq!(obj["a"].as_array().unwrap().len(), 2);
        assert_eq!(obj["b"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn test_sort_by() {
        let data = serde_json::json!([
            {"name": "c", "score": 3},
            {"name": "a", "score": 1},
            {"name": "b", "score": 2},
        ]);
        let mut f = SortByFn;
        let mut func = Func::new();
        func.args.push(FuncArg::Key("score".into()));
        let result = f.call(&func, &data, None).unwrap();
        let arr = result.as_array().unwrap();
        assert_eq!(arr[0]["name"], "a");
        assert_eq!(arr[1]["name"], "b");
        assert_eq!(arr[2]["name"], "c");
    }

    #[test]
    fn test_flatten() {
        let data = serde_json::json!([[1, 2], [3, 4], [5]]);
        let mut f = FlattenFn;
        let func = Func::new();
        let result = f.call(&func, &data, None).unwrap();
        assert_eq!(result, serde_json::json!([1, 2, 3, 4, 5]));
    }

    #[test]
    fn test_merge() {
        let data = serde_json::json!([{"a": 1}, {"b": 2}, {"c": 3}]);
        let mut f = MergeFn;
        let func = Func::new();
        let result = f.call(&func, &data, None).unwrap();
        let obj = result.as_object().unwrap();
        assert_eq!(obj.len(), 3);
    }

    #[test]
    fn test_omit() {
        let data = serde_json::json!({"a": 1, "b": 2, "c": 3});
        let mut f = OmitFn;
        let mut func = Func::new();
        func.args.push(FuncArg::Key("b".into()));
        let result = f.call(&func, &data, None).unwrap();
        let obj = result.as_object().unwrap();
        assert!(!obj.contains_key("b"));
        assert!(obj.contains_key("a"));
        assert!(obj.contains_key("c"));
    }

    #[test]
    fn test_unique() {
        let data = serde_json::json!([1, 2, 2, 3, 1]);
        let mut f = UniqueFn;
        let func = Func::new();
        let result = f.call(&func, &data, None).unwrap();
        assert_eq!(result.as_array().unwrap().len(), 3);
    }

    #[test]
    fn test_tally() {
        let data = serde_json::json!(["a", "b", "a", "c", "b", "a"]);
        let mut f = TallyFn;
        let func = Func::new();
        let result = f.call(&func, &data, None).unwrap();
        let obj = result.as_object().unwrap();
        assert_eq!(obj["a"].as_i64(), Some(3));
        assert_eq!(obj["b"].as_i64(), Some(2));
        assert_eq!(obj["c"].as_i64(), Some(1));
    }

    #[test]
    fn test_chunk() {
        let data = serde_json::json!([1, 2, 3, 4, 5]);
        let mut f = ChunkFn;
        let mut func = Func::new();
        func.args.push(FuncArg::Number(2.0));
        let result = f.call(&func, &data, None).unwrap();
        let arr = result.as_array().unwrap();
        assert_eq!(arr.len(), 3);
        assert_eq!(arr[0].as_array().unwrap().len(), 2);
        assert_eq!(arr[2].as_array().unwrap().len(), 1);
    }

    #[test]
    fn test_get() {
        let data = serde_json::json!({"name": "Alice", "age": 30});
        let mut f = GetFn;
        let mut func = Func::new();
        func.args.push(FuncArg::Key("name".into()));
        let result = f.call(&func, &data, None).unwrap();
        assert_eq!(result, serde_json::json!("Alice"));
    }

    #[test]
    fn test_pluck() {
        let data = serde_json::json!([
            {"name": "Alice", "age": 30},
            {"name": "Bob",   "age": 25},
            {"name": "Carol", "age": 28},
        ]);
        let mut f = PluckFn;
        let mut func = Func::new();
        func.args.push(FuncArg::Key("name".into()));
        let result = f.call(&func, &data, None).unwrap();
        assert_eq!(result, serde_json::json!(["Alice", "Bob", "Carol"]));
    }

    #[test]
    fn test_find() {
        use crate::context::{FilterAST, FilterInner, FilterInnerRighthand, FilterOp};
        let data = serde_json::json!([
            {"name": "Alice", "age": 30},
            {"name": "Bob",   "age": 17},
            {"name": "Carol", "age": 22},
        ]);
        // find first item where age == 17
        let cond = FilterInner::Cond {
            left: "age".into(),
            op: FilterOp::Eq,
            right: FilterInnerRighthand::Number(17),
        };
        let ast = FilterAST::new(cond);
        let mut f = FindFn;
        let mut func = Func::new();
        func.args.push(FuncArg::FilterExpr(ast));
        let result = f.call(&func, &data, None).unwrap();
        assert_eq!(result["name"], "Bob");
    }

    #[test]
    fn test_filter_by() {
        use crate::context::{FilterAST, FilterInner, FilterInnerRighthand, FilterOp};
        let data = serde_json::json!([
            {"name": "Alice", "score": 80},
            {"name": "Bob",   "score": 45},
            {"name": "Carol", "score": 90},
        ]);
        // filter_by score >= 80
        let cond = FilterInner::Cond {
            left: "score".into(),
            op: FilterOp::Gq,
            right: FilterInnerRighthand::Number(80),
        };
        let ast = FilterAST::new(cond);
        let mut f = FilterByFn;
        let mut func = Func::new();
        func.args.push(FuncArg::FilterExpr(ast));
        let result = f.call(&func, &data, None).unwrap();
        let arr = result.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["name"], "Alice");
        assert_eq!(arr[1]["name"], "Carol");
    }
}
