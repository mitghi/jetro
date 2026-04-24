//! Static builtin registry — zero vtable overhead via raw function pointers.
//!
//! All builtins are `fn(Val, &[Arg], &Env) -> Result<Val, EvalError>`.
//! The registry is initialized once on first use via `OnceLock`.

use std::collections::HashMap;
use std::sync::OnceLock;
use std::sync::Arc;

use crate::ast::Arg;

use super::{Env, EvalError, eval_pos, apply_item};
use super::value::Val;
use super::util::{val_to_string, val_str, field_exists_nested};
use super::{func_strings, func_arrays, func_objects, func_paths, func_aggregates, func_csv, func_search};

macro_rules! err {
    ($($t:tt)*) => { Err(EvalError(format!($($t)*))) };
}

// ── Type alias ────────────────────────────────────────────────────────────────

pub type BuiltinFn = fn(Val, &[Arg], &Env) -> Result<Val, EvalError>;

// ── Registry ──────────────────────────────────────────────────────────────────

pub struct BuiltinRegistry {
    table: HashMap<&'static str, BuiltinFn>,
}

impl BuiltinRegistry {
    #[inline]
    pub fn get(&self, name: &str) -> Option<BuiltinFn> {
        self.table.get(name).copied()
    }

    /// Iterate all registered builtin names (includes snake_case and camelCase aliases).
    pub fn names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.table.keys().copied()
    }
}

/// Convenience — all builtin method names as a sorted `Vec<&'static str>`.
pub fn all_names() -> Vec<&'static str> {
    let mut v: Vec<&'static str> = global().names().collect();
    v.sort_unstable();
    v
}

static BUILTINS: OnceLock<BuiltinRegistry> = OnceLock::new();

pub fn global() -> &'static BuiltinRegistry {
    BUILTINS.get_or_init(build)
}

// ── Build ─────────────────────────────────────────────────────────────────────

fn build() -> BuiltinRegistry {
    let mut t: HashMap<&'static str, BuiltinFn> = HashMap::with_capacity(128);

    // Basics
    t.insert("len",        b_len);
    t.insert("type",       b_type);
    t.insert("to_string",  b_to_string);
    t.insert("toString",   b_to_string);
    t.insert("to_json",    b_to_json);
    t.insert("toJson",     b_to_json);
    t.insert("from_json",  b_from_json);
    t.insert("fromJson",   b_from_json);

    // Object
    t.insert("keys",        b_keys);
    t.insert("values",      b_values);
    t.insert("entries",     b_entries);
    t.insert("to_pairs",    b_to_pairs);
    t.insert("toPairs",     b_to_pairs);
    t.insert("from_pairs",  b_from_pairs);
    t.insert("fromPairs",   b_from_pairs);
    t.insert("invert",      b_invert);
    t.insert("pick",        func_objects::pick);
    t.insert("omit",        func_objects::omit);
    t.insert("merge",       func_objects::merge);
    t.insert("deep_merge",  func_objects::deep_merge_method);
    t.insert("deepMerge",   func_objects::deep_merge_method);
    t.insert("defaults",    func_objects::defaults);
    t.insert("rename",      func_objects::rename);
    t.insert("transform_keys",   func_objects::transform_keys);
    t.insert("transformKeys",    func_objects::transform_keys);
    t.insert("transform_values", func_objects::transform_values);
    t.insert("transformValues",  func_objects::transform_values);
    t.insert("filter_keys",   func_objects::filter_keys);
    t.insert("filterKeys",    func_objects::filter_keys);
    t.insert("filter_values", func_objects::filter_values);
    t.insert("filterValues",  func_objects::filter_values);
    t.insert("pivot",       func_objects::pivot);

    // Arrays — full signature
    t.insert("filter",    func_arrays::filter);
    t.insert("find",      func_arrays::find);        // Tier 1 (shallow, multi-pred AND)
    t.insert("find_all",  func_arrays::find);        // Tier 1 (shallow, multi-pred AND)
    t.insert("findAll",   func_arrays::find);
    t.insert("map",       func_arrays::map);
    t.insert("flatMap",   func_arrays::flat_map);
    t.insert("flat_map",  func_arrays::flat_map);
    t.insert("sort",      func_arrays::sort);
    t.insert("flatten",   func_arrays::flatten);
    t.insert("join",      func_arrays::join);
    t.insert("equi_join", func_arrays::equi_join);
    t.insert("equiJoin",  func_arrays::equi_join);
    t.insert("first",     func_arrays::first);
    t.insert("last",      func_arrays::last);
    t.insert("nth",       func_arrays::nth);
    t.insert("append",    func_arrays::append);
    t.insert("prepend",   func_arrays::prepend);
    t.insert("remove",    func_arrays::remove);
    t.insert("diff",      func_arrays::diff);
    t.insert("intersect", func_arrays::intersect);
    t.insert("union",     func_arrays::union);
    t.insert("enumerate", func_arrays::enumerate);
    t.insert("window",    func_arrays::window);
    t.insert("chunk",     func_arrays::chunk);
    t.insert("batch",     func_arrays::chunk);
    t.insert("takewhile", func_arrays::takewhile);
    t.insert("take_while",func_arrays::takewhile);
    t.insert("dropwhile", func_arrays::dropwhile);
    t.insert("drop_while",func_arrays::dropwhile);
    t.insert("accumulate",func_arrays::accumulate);
    t.insert("partition", func_arrays::partition);
    t.insert("zip",       func_arrays::zip_method);
    t.insert("zip_longest",  func_arrays::zip_longest_method);
    t.insert("zipLongest",   func_arrays::zip_longest_method);

    // Arrays — zero-extra-arg wrappers
    t.insert("reverse",  b_reverse);
    t.insert("unique",   b_unique);
    t.insert("distinct", b_unique);
    t.insert("compact",  b_compact);
    t.insert("pairwise", b_pairwise);

    // Tier 1 search / match / collect
    t.insert("unique_by", func_search::unique_by);
    t.insert("uniqueBy",  func_search::unique_by);
    t.insert("collect",   func_search::collect);
    t.insert("deep_find", func_search::deep_find);
    t.insert("deepFind",  func_search::deep_find);
    t.insert("deep_shape", func_search::deep_shape);
    t.insert("deepShape",  func_search::deep_shape);
    t.insert("deep_like",  func_search::deep_like);
    t.insert("deepLike",   func_search::deep_like);

    // Aggregates — full signature
    t.insert("sum",      func_aggregates::sum);
    t.insert("avg",      func_aggregates::avg);
    t.insert("count",    func_aggregates::count);
    t.insert("groupBy",  func_aggregates::group_by);
    t.insert("group_by", func_aggregates::group_by);
    t.insert("countBy",  func_aggregates::count_by);
    t.insert("count_by", func_aggregates::count_by);
    t.insert("indexBy",  func_aggregates::index_by);
    t.insert("index_by", func_aggregates::index_by);

    // Aggregates — bool-flag wrappers
    t.insert("min", b_min);
    t.insert("max", b_max);
    t.insert("any", b_any);
    t.insert("all", b_all);

    // Numeric scalar ops
    t.insert("ceil",  b_ceil);
    t.insert("floor", b_floor);
    t.insert("round", b_round);
    t.insert("abs",   b_abs);

    // Paths
    t.insert("get_path",       func_paths::get_path);
    t.insert("getPath",        func_paths::get_path);
    t.insert("set_path",       func_paths::set_path);
    t.insert("setPath",        func_paths::set_path);
    t.insert("del_path",       func_paths::del_path);
    t.insert("delPath",        func_paths::del_path);
    t.insert("del_paths",      func_paths::del_paths);
    t.insert("delPaths",       func_paths::del_paths);
    t.insert("has_path",       func_paths::has_path);
    t.insert("hasPath",        func_paths::has_path);
    t.insert("flatten_keys",   func_paths::flatten_keys);
    t.insert("flattenKeys",    func_paths::flatten_keys);
    t.insert("unflatten_keys", func_paths::unflatten_keys);
    t.insert("unflattenKeys",  func_paths::unflatten_keys);

    // CSV
    t.insert("to_csv", b_to_csv);
    t.insert("toCsv",  b_to_csv);
    t.insert("to_tsv", b_to_tsv);
    t.insert("toTsv",  b_to_tsv);

    // Null safety / existence
    t.insert("or",       b_or);
    t.insert("has",      b_has);
    t.insert("missing",  b_missing);
    t.insert("includes", b_includes);
    t.insert("contains", b_includes);

    // Update / set
    t.insert("set",    b_set);
    t.insert("update", b_update);

    // String methods
    t.insert("upper",          func_strings::upper);
    t.insert("lower",          func_strings::lower);
    t.insert("capitalize",     func_strings::capitalize);
    t.insert("title_case",     func_strings::title_case);
    t.insert("titleCase",      func_strings::title_case);
    t.insert("trim",           func_strings::trim);
    t.insert("trim_left",      func_strings::trim_left);
    t.insert("trimLeft",       func_strings::trim_left);
    t.insert("trim_right",     func_strings::trim_right);
    t.insert("trimRight",      func_strings::trim_right);
    t.insert("lines",          func_strings::lines);
    t.insert("words",          func_strings::words);
    t.insert("chars",          func_strings::chars);
    t.insert("to_number",      func_strings::to_number);
    t.insert("toNumber",       func_strings::to_number);
    t.insert("to_bool",        func_strings::to_bool);
    t.insert("toBool",         func_strings::to_bool);
    t.insert("to_base64",      func_strings::to_base64);
    t.insert("toBase64",       func_strings::to_base64);
    t.insert("from_base64",    func_strings::from_base64);
    t.insert("fromBase64",     func_strings::from_base64);
    t.insert("url_encode",     func_strings::url_encode);
    t.insert("urlEncode",      func_strings::url_encode);
    t.insert("url_decode",     func_strings::url_decode);
    t.insert("urlDecode",      func_strings::url_decode);
    t.insert("html_escape",    func_strings::html_escape);
    t.insert("htmlEscape",     func_strings::html_escape);
    t.insert("html_unescape",  func_strings::html_unescape);
    t.insert("htmlUnescape",   func_strings::html_unescape);
    t.insert("repeat",         func_strings::repeat);
    t.insert("pad_left",       func_strings::pad_left);
    t.insert("padLeft",        func_strings::pad_left);
    t.insert("pad_right",      func_strings::pad_right);
    t.insert("padRight",       func_strings::pad_right);
    t.insert("starts_with",    func_strings::starts_with);
    t.insert("startsWith",     func_strings::starts_with);
    t.insert("ends_with",      func_strings::ends_with);
    t.insert("endsWith",       func_strings::ends_with);
    t.insert("index_of",       func_strings::index_of);
    t.insert("indexOf",        func_strings::index_of);
    t.insert("last_index_of",  func_strings::last_index_of);
    t.insert("lastIndexOf",    func_strings::last_index_of);
    t.insert("replace",        func_strings::replace);
    t.insert("replace_all",    func_strings::replace_all);
    t.insert("replaceAll",     func_strings::replace_all);
    t.insert("strip_prefix",   func_strings::strip_prefix);
    t.insert("stripPrefix",    func_strings::strip_prefix);
    t.insert("strip_suffix",   func_strings::strip_suffix);
    t.insert("stripSuffix",    func_strings::strip_suffix);
    t.insert("slice",          func_strings::str_slice);
    t.insert("split",          func_strings::split);
    t.insert("indent",         func_strings::indent);
    t.insert("dedent",         func_strings::dedent);
    t.insert("matches",        func_strings::str_matches);
    t.insert("scan",           func_strings::scan);

    BuiltinRegistry { table: t }
}

// ── Wrapper functions ─────────────────────────────────────────────────────────

fn b_len(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    Ok(Val::Int(match &recv {
        Val::Arr(a) => a.len() as i64,
        Val::Obj(m) => m.len() as i64,
        Val::Str(s) => s.chars().count() as i64,
        _ => return err!("len: unsupported type"),
    }))
}

fn b_type(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    Ok(Val::Str(Arc::from(recv.type_name())))
}

fn b_to_string(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    Ok(val_str(&val_to_string(&recv)))
}

fn b_to_json(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    let sv: serde_json::Value = recv.into();
    Ok(val_str(&serde_json::to_string(&sv).unwrap_or_default()))
}

fn b_from_json(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    // Direct one-pass parse via `Val::deserialize` — no intermediate
    // `serde_json::Value` tree.  For `Val::Str` receivers we skip the
    // buffer deep-clone too.
    match &recv {
        Val::Str(s) => Val::from_json_str(s.as_ref())
            .map_err(|e| EvalError(format!("from_json: {}", e))),
        _ => {
            let s = val_to_string(&recv);
            Val::from_json_str(&s)
                .map_err(|e| EvalError(format!("from_json: {}", e)))
        }
    }
}

fn b_keys(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    func_objects::keys(recv)
}

fn b_values(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    func_objects::values(recv)
}

fn b_entries(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    func_objects::entries(recv)
}

fn b_to_pairs(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    func_objects::to_pairs(recv)
}

fn b_from_pairs(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    func_objects::from_pairs(recv)
}

fn b_invert(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    func_objects::invert(recv)
}

fn b_reverse(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    func_arrays::reverse(recv)
}

fn b_unique(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    func_arrays::unique(recv)
}

fn b_compact(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    func_arrays::compact(recv)
}

fn b_pairwise(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    func_arrays::pairwise(recv)
}

fn b_min(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    func_aggregates::minmax(recv, args, env, false)
}

fn b_max(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    func_aggregates::minmax(recv, args, env, true)
}

fn b_any(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    func_aggregates::any_all(recv, args, env, false)
}

fn b_all(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    func_aggregates::any_all(recv, args, env, true)
}

fn b_ceil(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    match recv {
        Val::Int(n)   => Ok(Val::Int(n)),
        Val::Float(f) => Ok(Val::Int(f.ceil() as i64)),
        _ => err!("ceil: expected number"),
    }
}

fn b_floor(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    match recv {
        Val::Int(n)   => Ok(Val::Int(n)),
        Val::Float(f) => Ok(Val::Int(f.floor() as i64)),
        _ => err!("floor: expected number"),
    }
}

fn b_round(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    match recv {
        Val::Int(n)   => Ok(Val::Int(n)),
        // Banker's rounding is the IEEE default; jq uses `round()` which
        // ties-away-from-zero.  Rust's `f64::round` matches jq here.
        Val::Float(f) => Ok(Val::Int(f.round() as i64)),
        _ => err!("round: expected number"),
    }
}

fn b_abs(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    match recv {
        Val::Int(n)   => Ok(Val::Int(n.wrapping_abs())),
        Val::Float(f) => Ok(Val::Float(f.abs())),
        _ => err!("abs: expected number"),
    }
}

fn b_to_csv(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    Ok(val_str(&func_csv::to_csv(&recv)))
}

fn b_to_tsv(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    Ok(val_str(&func_csv::to_tsv(&recv)))
}

fn b_or(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let default = args.first().map(|a| eval_pos(a, env)).transpose()?.unwrap_or(Val::Null);
    if recv.is_null() { Ok(default) } else { Ok(recv) }
}

fn b_has(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let result = args.iter().all(|a| {
        eval_pos(a, env).ok()
            .and_then(|v| v.as_str().map(|s| s.to_string()))
            .map(|key| field_exists_nested(&recv, &key))
            .unwrap_or(false)
    });
    Ok(Val::Bool(result))
}

fn b_missing(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let key = args.first().map(|a| eval_pos(a, env)).transpose()?
        .and_then(|v| v.as_str().map(|s| s.to_string()))
        .unwrap_or_default();
    Ok(Val::Bool(!field_exists_nested(&recv, &key)))
}

fn b_includes(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let item = args.first().map(|a| eval_pos(a, env)).transpose()?.unwrap_or(Val::Null);
    use super::util::val_to_key;
    let key = val_to_key(&item);
    Ok(Val::Bool(match &recv {
        Val::Arr(a) => a.iter().any(|v| val_to_key(v) == key),
        Val::Str(s) => s.contains(item.as_str().unwrap_or_default()),
        _ => false,
    }))
}

fn b_set(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let _ = recv;
    Ok(args.first().map(|a| eval_pos(a, env)).transpose()?.unwrap_or(Val::Null))
}

fn b_update(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let lam = args.first().ok_or_else(|| EvalError("update: requires lambda".into()))?;
    apply_item(recv, lam, env)
}
