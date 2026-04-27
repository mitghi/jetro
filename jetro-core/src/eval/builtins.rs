//! Static builtin registry — zero vtable overhead via raw function pointers.
//!
//! All builtins are `fn(Val, &[Arg], &Env) -> Result<Val, EvalError>`.
//! The registry is initialized once on first use via `OnceLock`.

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::ast::Arg;

use super::util::{field_exists_nested, val_str, val_to_string};
use super::value::Val;
use super::{apply_item, eval_pos, first_i64_arg, str_arg, Env, EvalError};
use super::{func_aggregates, func_arrays, func_csv, func_objects, func_paths, func_search};

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

    /// Iterate all registered builtin names (snake_case canonical form).
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
    t.insert("len", b_len);
    t.insert("type", b_type);
    t.insert("to_string", b_to_string);
    t.insert("to_json", b_to_json);
    t.insert("from_json", b_from_json);

    // Object
    t.insert("keys", keys_dispatch);
    t.insert("values", values_dispatch);
    t.insert("entries", entries_dispatch);
    t.insert("to_pairs", b_to_pairs);
    t.insert("from_pairs", b_from_pairs);
    t.insert("invert", crate::composed::shims::invert);
    t.insert("pick", func_objects::pick);
    t.insert("omit", func_objects::omit);
    t.insert("merge", crate::composed::shims::merge);
    t.insert("deep_merge", crate::composed::shims::deep_merge);
    t.insert("defaults", crate::composed::shims::defaults);
    t.insert("rename", crate::composed::shims::rename);
    t.insert("transform_keys", func_objects::transform_keys);
    t.insert("transform_values", func_objects::transform_values);
    t.insert("filter_keys", func_objects::filter_keys);
    t.insert("filter_values", func_objects::filter_values);
    t.insert("pivot", func_objects::pivot);

    // Arrays — full signature
    t.insert("filter", func_arrays::filter);
    t.insert("find", func_arrays::find); // Tier 1 (shallow, multi-pred AND)
    t.insert("find_all", func_arrays::find); // Tier 1 (shallow, multi-pred AND)
    t.insert("map", func_arrays::map);
    t.insert("flat_map", func_arrays::flat_map);
    t.insert("sort", func_arrays::sort);
    t.insert("flatten", crate::composed::shims::flatten);
    t.insert("join", func_arrays::join);
    t.insert("equi_join", func_arrays::equi_join);
    t.insert("first", crate::composed::shims::first);
    t.insert("last", crate::composed::shims::last);
    t.insert("nth", crate::composed::shims::nth);
    t.insert("append", crate::composed::shims::append);
    t.insert("prepend", crate::composed::shims::prepend);
    t.insert("remove", func_arrays::remove);
    t.insert("diff", crate::composed::shims::diff);
    t.insert("intersect", crate::composed::shims::intersect);
    t.insert("union", crate::composed::shims::union);
    t.insert("enumerate", func_arrays::enumerate);
    // .window / .chunk / .batch lifted to pipeline::Stage::Window /
    // Stage::Chunk (batch alias).
    t.insert("window", window_dispatch);
    t.insert("chunk", chunk_dispatch);
    t.insert("batch", chunk_dispatch);
    t.insert("takewhile", func_arrays::takewhile);
    t.insert("take_while", func_arrays::takewhile);
    t.insert("dropwhile", func_arrays::dropwhile);
    t.insert("drop_while", func_arrays::dropwhile);
    t.insert("accumulate", func_arrays::accumulate);
    t.insert("partition", func_arrays::partition);
    t.insert("zip", func_arrays::zip_method);
    t.insert("zip_longest", func_arrays::zip_longest_method);

    // Arrays — zero-extra-arg wrappers (lifted to composed Stages).
    t.insert("reverse", crate::composed::shims::reverse);
    t.insert("unique", crate::composed::shims::unique);
    t.insert("distinct", crate::composed::shims::unique);
    t.insert("compact", crate::composed::shims::compact);
    t.insert("pairwise", crate::composed::shims::pairwise);

    // Tier 1 search / match / collect
    t.insert("unique_by", func_search::unique_by);
    t.insert("collect", func_search::collect);
    t.insert("deep_find", func_search::deep_find);
    t.insert("deep_shape", func_search::deep_shape);
    t.insert("deep_like", func_search::deep_like);
    t.insert("walk", func_search::walk);
    t.insert("walk_pre", func_search::walk_pre_fn);
    t.insert("schema", func_objects::schema);
    t.insert("rec", func_search::rec);
    t.insert("trace_path", func_search::trace_path);
    t.insert("fanout", func_objects::fanout);
    t.insert("zip_shape", func_objects::zip_shape);

    // Aggregates — full signature
    t.insert("sum", func_aggregates::sum);
    t.insert("avg", func_aggregates::avg);
    t.insert("count", func_aggregates::count);
    t.insert("group_by", func_aggregates::group_by);
    t.insert("count_by", func_aggregates::count_by);
    t.insert("index_by", func_aggregates::index_by);
    t.insert("explode", func_aggregates::explode);
    t.insert("implode", func_aggregates::implode);
    t.insert("group_shape", func_aggregates::group_shape);

    // Aggregates — bool-flag wrappers
    t.insert("min", b_min);
    t.insert("max", b_max);
    t.insert("any", b_any);
    t.insert("exists", b_any); // alias — reads natural in queries
    t.insert("all", b_all);
    t.insert("every", b_all); // alias for all

    // Numeric scalar ops
    t.insert("ceil", b_ceil);
    t.insert("floor", b_floor);
    t.insert("round", b_round);
    t.insert("abs", b_abs);

    // Paths
    t.insert("get_path", func_paths::get_path);
    t.insert("set_path", func_paths::set_path);
    t.insert("del_path", func_paths::del_path);
    t.insert("del_paths", func_paths::del_paths);
    t.insert("has_path", func_paths::has_path);
    t.insert("flatten_keys", func_paths::flatten_keys);
    t.insert("unflatten_keys", func_paths::unflatten_keys);

    // CSV
    t.insert("to_csv", b_to_csv);
    t.insert("to_tsv", b_to_tsv);

    // Null safety / existence
    t.insert("or", b_or);
    t.insert("has", b_has);
    t.insert("missing", b_missing);
    t.insert("includes", b_includes);
    t.insert("contains", b_includes);

    // Update / set
    t.insert("set", b_set);
    t.insert("update", b_update);

    // String methods — every body lives in `composed.rs` as a
    // first-class Stage; dispatch shims live in `composed::shims`.
    // (`eval/func_strings.rs` was deleted.)
    t.insert("upper", crate::composed::shims::upper);
    t.insert("lower", crate::composed::shims::lower);
    t.insert("capitalize", crate::composed::shims::capitalize);
    t.insert("title_case", crate::composed::shims::title_case);
    t.insert("trim", crate::composed::shims::trim);
    t.insert("trim_left", crate::composed::shims::trim_left);
    t.insert("trim_right", crate::composed::shims::trim_right);
    t.insert("lines", crate::composed::shims::lines);
    t.insert("words", crate::composed::shims::words);
    t.insert("chars", crate::composed::shims::chars);
    t.insert("to_number", crate::composed::shims::to_number);
    t.insert("to_bool", crate::composed::shims::to_bool);
    t.insert("to_base64", crate::composed::shims::to_base64);
    t.insert("from_base64", crate::composed::shims::from_base64);
    t.insert("url_encode", crate::composed::shims::url_encode);
    t.insert("url_decode", crate::composed::shims::url_decode);
    t.insert("html_escape", crate::composed::shims::html_escape);
    t.insert("html_unescape", crate::composed::shims::html_unescape);
    t.insert("repeat", crate::composed::shims::repeat);
    t.insert("pad_left", crate::composed::shims::pad_left);
    t.insert("pad_right", crate::composed::shims::pad_right);
    t.insert("starts_with", crate::composed::shims::starts_with);
    t.insert("ends_with", crate::composed::shims::ends_with);
    t.insert("index_of", crate::composed::shims::index_of);
    t.insert("last_index_of", crate::composed::shims::last_index_of);
    // .replace / .replace_all lifted to pipeline::Stage::Replace.
    t.insert("replace", replace_dispatch);
    t.insert("replace_all", replace_all_dispatch);
    t.insert("strip_prefix", crate::composed::shims::strip_prefix);
    t.insert("strip_suffix", crate::composed::shims::strip_suffix);
    // .slice / .split lifted to pipeline::Stage::Slice / Stage::Split.
    t.insert("slice", slice_dispatch);
    t.insert("split", split_dispatch);
    t.insert("indent", crate::composed::shims::indent);
    t.insert("dedent", crate::composed::shims::dedent);
    t.insert("matches", crate::composed::shims::str_matches);
    t.insert("scan", crate::composed::shims::scan);

    // Case-conversion family — lifted Stages.
    t.insert("snake_case", crate::composed::shims::snake_case);
    t.insert("kebab_case", crate::composed::shims::kebab_case);
    t.insert("camel_case", crate::composed::shims::camel_case);
    t.insert("pascal_case", crate::composed::shims::pascal_case);

    // Padding / repetition / reversal.
    t.insert("center", crate::composed::shims::center);
    t.insert("repeat_str", crate::composed::shims::repeat_str);
    t.insert("reverse_str", crate::composed::shims::reverse_str);

    // Char / byte introspection.
    t.insert("chars_of", crate::composed::shims::chars_of);
    t.insert("bytes", crate::composed::shims::bytes_of);
    t.insert("byte_len", crate::composed::shims::byte_len);

    // Predicates / parsers.
    t.insert("is_blank", crate::composed::shims::is_blank);
    t.insert("is_numeric", crate::composed::shims::is_numeric);
    t.insert("is_alpha", crate::composed::shims::is_alpha);
    t.insert("is_ascii", crate::composed::shims::is_ascii);
    t.insert("parse_int", crate::composed::shims::parse_int);
    t.insert("parse_float", crate::composed::shims::parse_float);
    t.insert("parse_bool", crate::composed::shims::parse_bool);

    // Substring set predicates.
    t.insert("contains_any", crate::composed::shims::contains_any);
    t.insert("contains_all", crate::composed::shims::contains_all);

    // Index lookup family.
    t.insert("find_index", func_arrays::find_index);
    t.insert("index", func_arrays::index_of_value);
    t.insert("indices_where", func_arrays::indices_where);
    t.insert("indices_of", func_arrays::indices_of);

    // Argmax / argmin by key.
    t.insert("max_by", func_arrays::max_by);
    t.insert("min_by", func_arrays::min_by);

    // Window-style numeric ops.
    t.insert("rolling_avg", func_arrays::rolling_avg);
    t.insert("rolling_sum", func_arrays::rolling_sum);
    t.insert("rolling_min", func_arrays::rolling_min);
    t.insert("rolling_max", func_arrays::rolling_max);
    t.insert("lag", func_arrays::lag);
    t.insert("lead", func_arrays::lead);
    t.insert("diff_window", func_arrays::diff_window);
    t.insert("pct_change", func_arrays::pct_change);
    t.insert("cummax", func_arrays::cummax);
    t.insert("cummin", func_arrays::cummin);
    t.insert("zscore", func_arrays::zscore);

    // Regex family.
    t.insert("re_match", crate::composed::shims::re_match);
    t.insert("match_first", crate::composed::shims::re_match_first);
    t.insert("match_all", crate::composed::shims::re_match_all);
    t.insert("captures", crate::composed::shims::re_captures);
    t.insert("captures_all", crate::composed::shims::re_captures_all);
    t.insert("replace_re", crate::composed::shims::re_replace);
    t.insert("replace_all_re", crate::composed::shims::re_replace_all);
    t.insert("split_re", crate::composed::shims::re_split);

    BuiltinRegistry { table: t }
}

// ── Wrapper functions ─────────────────────────────────────────────────────────

fn b_len(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    use crate::composed::{Len, Stage as _, StageOutput};
    match Len.apply(&recv) {
        StageOutput::Pass(c) => Ok(c.into_owned()),
        _ => err!("len: unsupported type"),
    }
}

fn b_type(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    use crate::composed::{Stage as _, StageOutput, TypeName};
    match TypeName.apply(&recv) {
        StageOutput::Pass(c) => Ok(c.into_owned()),
        _ => err!("type: unsupported"),
    }
}

fn b_to_string(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    use crate::composed::{Stage as _, StageOutput};
    match crate::composed::ToString.apply(&recv) {
        StageOutput::Pass(c) => Ok(c.into_owned()),
        _ => err!("to_string: unsupported"),
    }
}

fn b_to_json(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    use crate::composed::{Stage as _, StageOutput};
    match crate::composed::ToJson.apply(&recv) {
        StageOutput::Pass(c) => Ok(c.into_owned()),
        _ => err!("to_json: unsupported"),
    }
}

fn b_from_json(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    // Direct one-pass parse via `Val::deserialize` — no intermediate
    // `serde_json::Value` tree.  For `Val::Str` receivers we skip the
    // buffer deep-clone too.  With the `simd-json` feature enabled,
    // route through simd-json's SIMD structural scanner instead; the
    // str receiver path copies once into a mutable buffer (required by
    // simd-json), still faster than serde_json on docs over ~4KB.
    #[cfg(feature = "simd-json")]
    {
        let bytes_owned: Vec<u8> = match &recv {
            Val::Str(s) => s.as_bytes().to_vec(),
            _ => val_to_string(&recv).into_bytes(),
        };
        let mut bytes = bytes_owned;
        return Val::from_json_simd(&mut bytes).map_err(|e| EvalError(format!("from_json: {}", e)));
    }
    #[cfg(not(feature = "simd-json"))]
    match &recv {
        Val::Str(s) => {
            Val::from_json_str(s.as_ref()).map_err(|e| EvalError(format!("from_json: {}", e)))
        }
        _ => {
            let s = val_to_string(&recv);
            Val::from_json_str(&s).map_err(|e| EvalError(format!("from_json: {}", e)))
        }
    }
}

fn b_to_pairs(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    func_objects::to_pairs(recv)
}

fn b_from_pairs(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    func_objects::from_pairs(recv)
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
        Val::Int(n) => Ok(Val::Int(n)),
        Val::Float(f) => Ok(Val::Int(f.ceil() as i64)),
        _ => err!("ceil: expected number"),
    }
}

fn b_floor(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    match recv {
        Val::Int(n) => Ok(Val::Int(n)),
        Val::Float(f) => Ok(Val::Int(f.floor() as i64)),
        _ => err!("floor: expected number"),
    }
}

fn b_round(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    match recv {
        Val::Int(n) => Ok(Val::Int(n)),
        // Banker's rounding is the IEEE default; jq uses `round()` which
        // ties-away-from-zero.  Rust's `f64::round` matches jq here.
        Val::Float(f) => Ok(Val::Int(f.round() as i64)),
        _ => err!("round: expected number"),
    }
}

fn b_abs(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    match recv {
        Val::Int(n) => Ok(Val::Int(n.wrapping_abs())),
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
    let default = args
        .first()
        .map(|a| eval_pos(a, env))
        .transpose()?
        .unwrap_or(Val::Null);
    if recv.is_null() {
        Ok(default)
    } else {
        Ok(recv)
    }
}

fn b_has(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let result = args.iter().all(|a| {
        eval_pos(a, env)
            .ok()
            .and_then(|v| v.as_str().map(|s| s.to_string()))
            .map(|key| field_exists_nested(&recv, &key))
            .unwrap_or(false)
    });
    Ok(Val::Bool(result))
}

// ── Pipeline-Stage dispatch shims (lift_all_builtins.md template) ─────────────
//
// Each shim parses positional args + delegates to the canonical Stage::apply
// helper in `crate::pipeline`. Stage's runtime path and the `dispatch_method`
// path share one implementation. Once a method is lifted, its eval/func_*.rs
// body deletes — only the shim and the pipeline helper remain.

fn split_dispatch(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let sep = str_arg(args, 0, env)?;
    crate::builtins::split_apply(&recv, sep.as_str())
        .ok_or_else(|| EvalError("split: expected string".into()))
}

fn slice_dispatch(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    if !matches!(recv, Val::Str(_) | Val::StrSlice(_)) {
        return err!("slice: expected string");
    }
    let start = first_i64_arg(args, env).unwrap_or(0);
    let end = args
        .get(1)
        .and_then(|a| eval_pos(a, env).ok())
        .and_then(|v| v.as_i64());
    Ok(crate::builtins::slice_apply(recv, start, end))
}

fn replace_dispatch(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let needle = str_arg(args, 0, env)?;
    let replacement = str_arg(args, 1, env)?;
    crate::builtins::replace_apply(recv, needle.as_str(), replacement.as_str(), false)
        .ok_or_else(|| EvalError("replace: expected string".into()))
}

fn replace_all_dispatch(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let needle = str_arg(args, 0, env)?;
    let replacement = str_arg(args, 1, env)?;
    crate::builtins::replace_apply(recv, needle.as_str(), replacement.as_str(), true)
        .ok_or_else(|| EvalError("replace_all: expected string".into()))
}

fn chunk_dispatch(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let n = first_i64_arg(args, env).unwrap_or(1).max(1) as usize;
    let items = recv
        .into_vec()
        .ok_or_else(|| EvalError("chunk: expected array".into()))?;
    Ok(Val::arr(crate::builtins::chunk_apply(&items, n)))
}

fn window_dispatch(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let n = first_i64_arg(args, env).unwrap_or(2).max(1) as usize;
    let items = recv
        .into_vec()
        .ok_or_else(|| EvalError("window: expected array".into()))?;
    Ok(Val::arr(crate::builtins::window_apply(&items, n)))
}

// lift_all_builtins (object family) — fallback dispatch for non-pipeline
// contexts (e.g., method-call inside a Map sub-program body).  Shares
// the canonical kernel with `Stage::Keys` / `Stage::Values` / `Stage::Entries`.
fn keys_dispatch(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    Ok(crate::builtins::keys_apply(&recv))
}

fn values_dispatch(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    Ok(crate::builtins::values_apply(&recv))
}

fn entries_dispatch(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    Ok(crate::builtins::entries_apply(&recv))
}

fn b_missing(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let key = args
        .first()
        .map(|a| eval_pos(a, env))
        .transpose()?
        .and_then(|v| v.as_str().map(|s| s.to_string()))
        .unwrap_or_default();
    Ok(Val::Bool(!field_exists_nested(&recv, &key)))
}

fn b_includes(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let item = args
        .first()
        .map(|a| eval_pos(a, env))
        .transpose()?
        .unwrap_or(Val::Null);
    use super::util::val_to_key;
    let key = val_to_key(&item);
    Ok(Val::Bool(match &recv {
        Val::Arr(a) => a.iter().any(|v| val_to_key(v) == key),
        Val::IntVec(a) => a.iter().any(|n| val_to_key(&Val::Int(*n)) == key),
        Val::FloatVec(a) => a.iter().any(|f| val_to_key(&Val::Float(*f)) == key),
        Val::StrVec(a) => match item.as_str() {
            Some(needle) => a.iter().any(|s| s.as_ref() == needle),
            None => false,
        },
        Val::Str(s) => s.contains(item.as_str().unwrap_or_default()),
        // Obj receiver: treat the arg as a key.  Lets `'name' in $.user`
        // and `$.user has 'name'` resolve to "object has this key".
        Val::Obj(m) => match item.as_str() {
            Some(k) => m.contains_key(k),
            None => false,
        },
        Val::ObjSmall(p) => match item.as_str() {
            Some(k) => p.iter().any(|(kk, _)| kk.as_ref() == k),
            None => false,
        },
        _ => false,
    }))
}

fn b_set(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let _ = recv;
    Ok(args
        .first()
        .map(|a| eval_pos(a, env))
        .transpose()?
        .unwrap_or(Val::Null))
}

fn b_update(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    let lam = args
        .first()
        .ok_or_else(|| EvalError("update: requires lambda".into()))?;
    apply_item(recv, lam, env)
}
