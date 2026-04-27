//! Lifted built-in functions — single home for `lift_all_builtins.md` Stage bodies.
//!
//! This module is the migration target for built-in operator bodies
//! moving out of `eval/func_*.rs` and previously interspersed in
//! `composed.rs`.  Each lifted built-in becomes a `Stage`-trait struct
//! with `apply<'a>(&Val) -> StageOutput<'a>` body living here as the
//! canonical implementation.  `composed.rs` retains only the
//! Stage/Sink trait substrate, the `Composed<A,B>` monoid, the generic
//! `run_pipeline`, VM-driven Generic* stages, and barrier ops.
//!
//! Visibility: `composed.rs` re-exports everything from this module
//! (`pub use crate::functions::*;`) so existing `crate::composed::Upper`
//! / `crate::composed::shims::*` paths keep resolving.  Future lifts
//! land here directly without going through composed.rs.

use std::borrow::Cow;
use smallvec::SmallVec;

use crate::eval::value::Val;
use crate::composed::{Stage, StageOutput};

// ── Lifted built-in Stages (lift_all_builtins.md workstream) ─────
//
// First-class Stage variants for builtin string/array/object/etc.
// methods.  Per `lift_all_builtins.md`: ~120 builtins lift to
// declarative Stages so the planner sees them as first-class
// operations (chain flattening, demand prop, dead-stage elim,
// commutative reorder, merge_with, cancels_with, eval_constant,
// column pruning).  Adding one builtin = one struct + 2 trait impls
// + (optional) optimisation hooks.
//
// Today only Upper is lifted as POC.  Future commits add the rest
// family-by-family per the lift_all_builtins.md effort estimate.

/// Run a single Stage and unwrap its single-Val result, or return
/// None when the Stage filtered/done.  Used by Method-dispatch shims
/// (`composed::shims::*`) that delegate to lifted Stages: the
/// function body lives in the Stage, the dispatch shim is thin.
#[inline]
pub fn run_single<S: Stage>(stage: &S, recv: &Val) -> Option<Val> {
    match stage.apply(recv) {
        StageOutput::Pass(c) => Some(c.into_owned()),
        _ => None,
    }
}

/// Method-dispatch shims for lifted built-ins.
///
/// Per `lift_all_builtins.md`: the `eval::builtins` registration
/// table was previously the home of `Opcode::CallMethod` dispatch
/// entry points (signature `fn(Val, &[Arg], &Env) -> Result<Val,
/// EvalError>`).  As built-in bodies migrate into first-class
/// `Stage` impls in this module, the dispatch entry points move
/// here too.  Each shim does:
///   1. validate receiver type (Val::Str / Val::Arr / etc.)
///   2. parse args (where applicable)
///   3. delegate to the Stage's `apply` via `run_single`
///   4. emit type-mismatch error if the Stage filters
///
/// `eval::func_*` then shrinks to ONLY the not-yet-lifted shims.
/// Eventually `eval::func_*.rs` files delete entirely once every
/// builtin lifts.
pub mod shims {
    use super::*;
    use crate::eval::{Env, EvalError};
    use crate::ast::{Arg, Expr};

    macro_rules! err { ($($t:tt)*) => { Err(EvalError(format!($($t)*))) }; }

    /// VM-based runtime arg evaluator.  Replaces `crate::eval::eval`
    /// (tree-walker) — compiles the Expr to a VM Program and executes
    /// via a thread-local VM.  Used inside Stage shims to evaluate
    /// runtime arguments without re-entering tree-walker code.
    ///
    /// Per-call compile cost is acceptable: shim arg eval is typically
    /// once-per-method-call (not per-row).  For hot per-row paths,
    /// callers should pre-compile and use VM directly.
    pub(crate) fn vm_eval(expr: &Expr, env: &Env) -> Result<Val, EvalError> {
        use std::cell::RefCell;
        thread_local! {
            static VM_CELL: RefCell<crate::vm::VM> = RefCell::new(crate::vm::VM::new());
        }
        let prog = crate::vm::Compiler::compile(expr, "<runtime-arg>");
        VM_CELL.with(|vm_cell| {
            let mut vm = vm_cell.borrow_mut();
            vm.exec(&prog, env)
        })
    }

    macro_rules! delegate_str_stage {
        ($shim:ident, $stage:expr, $err:expr) => {
            pub fn $shim(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
                if !matches!(recv, Val::Str(_)) { return err!($err); }
                run_single(&$stage, &recv).ok_or_else(|| EvalError($err.into()))
            }
        };
    }

    delegate_str_stage!(upper,        Upper,        "upper: expected string");
    delegate_str_stage!(lower,        Lower,        "lower: expected string");
    delegate_str_stage!(capitalize,   Capitalize,   "capitalize: expected string");
    delegate_str_stage!(title_case,   TitleCase,    "title_case: expected string");
    delegate_str_stage!(trim,         Trim,         "trim: expected string");
    delegate_str_stage!(trim_left,    TrimLeft,     "trim_left: expected string");
    delegate_str_stage!(trim_right,   TrimRight,    "trim_right: expected string");
    delegate_str_stage!(lines,        Lines,        "lines: expected string");
    delegate_str_stage!(words,        Words,        "words: expected string");
    delegate_str_stage!(chars,        Chars,        "chars: expected string");
    delegate_str_stage!(to_number,    ToNumber,     "to_number: expected string");
    delegate_str_stage!(to_base64,    ToBase64,     "to_base64: expected string");
    delegate_str_stage!(url_encode,   UrlEncode,    "url_encode: expected string");
    delegate_str_stage!(url_decode,   UrlDecode,    "url_decode: expected string");
    delegate_str_stage!(html_escape,  HtmlEscape,   "html_escape: expected string");
    delegate_str_stage!(html_unescape,HtmlUnescape, "html_unescape: expected string");
    delegate_str_stage!(dedent,       Dedent,       "dedent: expected string");
    delegate_str_stage!(snake_case,   SnakeCase,    "snake_case: expected string");
    delegate_str_stage!(kebab_case,   KebabCase,    "kebab_case: expected string");
    delegate_str_stage!(camel_case,   CamelCase,    "camel_case: expected string");
    delegate_str_stage!(pascal_case,  PascalCase,   "pascal_case: expected string");
    delegate_str_stage!(reverse_str,  ReverseStr,   "reverse: expected string");

    pub fn scan(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
        if !matches!(recv, Val::Str(_)) { return err!("scan: expected string"); }
        let pat = first_str_arg(args, env, "scan")?;
        run_single(&Scan::new(pat), &recv)
            .ok_or_else(|| EvalError("scan: stage filtered".into()))
    }

    /// `to_bool` errors on non-recognised inputs (cannot use generic
    /// macro).
    pub fn to_bool(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
        if let Val::Str(s) = recv {
            match s.as_ref() {
                "true"  => Ok(Val::Bool(true)),
                "false" => Ok(Val::Bool(false)),
                _       => err!("to_bool: not a boolean: {}", s),
            }
        } else { err!("to_bool: expected string") }
    }

    fn first_str_arg(args: &[Arg], env: &Env, name: &str) -> Result<std::sync::Arc<str>, EvalError> {
        let a = args.first().ok_or_else(|| EvalError(format!("{}: missing argument", name)))?;
        let v = match a {
            Arg::Pos(Expr::Ident(s)) => return Ok(std::sync::Arc::from(s.as_str())),
            Arg::Pos(e) | Arg::Named(_, e) => vm_eval(e, env)?,
        };
        match v {
            Val::Str(s) => Ok(s),
            _ => err!("{}: expected string argument", name),
        }
    }

    fn first_i64_arg(args: &[Arg], env: &Env, name: &str) -> Result<i64, EvalError> {
        let a = args.first().ok_or_else(|| EvalError(format!("{}: missing argument", name)))?;
        let v = match a {
            Arg::Pos(e) | Arg::Named(_, e) => vm_eval(e, env)?,
        };
        match v {
            Val::Int(n)   => Ok(n),
            Val::Float(f) => Ok(f as i64),
            _ => err!("{}: expected number argument", name),
        }
    }

    fn second_char_arg(args: &[Arg], env: &Env) -> char {
        args.get(1)
            .and_then(|a| match a {
                Arg::Pos(e) | Arg::Named(_, e) => vm_eval(e, env).ok(),
            })
            .and_then(|v| if let Val::Str(s) = v { s.chars().next() } else { None })
            .unwrap_or(' ')
    }

    pub fn starts_with(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
        if !matches!(recv, Val::Str(_)) { return err!("starts_with: expected string"); }
        let prefix = first_str_arg(args, env, "starts_with")?;
        run_single(&StartsWith::new(prefix), &recv)
            .ok_or_else(|| EvalError("starts_with: stage filtered".into()))
    }

    pub fn ends_with(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
        if !matches!(recv, Val::Str(_)) { return err!("ends_with: expected string"); }
        let suffix = first_str_arg(args, env, "ends_with")?;
        run_single(&EndsWith::new(suffix), &recv)
            .ok_or_else(|| EvalError("ends_with: stage filtered".into()))
    }

    pub fn index_of(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
        if !matches!(recv, Val::Str(_)) { return err!("index_of: expected string"); }
        let needle = first_str_arg(args, env, "index_of")?;
        run_single(&IndexOf::new(needle), &recv)
            .ok_or_else(|| EvalError("index_of: stage filtered".into()))
    }

    pub fn last_index_of(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
        if !matches!(recv, Val::Str(_)) { return err!("last_index_of: expected string"); }
        let needle = first_str_arg(args, env, "last_index_of")?;
        run_single(&LastIndexOf::new(needle), &recv)
            .ok_or_else(|| EvalError("last_index_of: stage filtered".into()))
    }

    pub fn strip_prefix(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
        if !matches!(recv, Val::Str(_)) { return err!("strip_prefix: expected string"); }
        let pat = first_str_arg(args, env, "strip_prefix")?;
        run_single(&StripPrefix::new(pat), &recv)
            .ok_or_else(|| EvalError("strip_prefix: stage filtered".into()))
    }

    pub fn strip_suffix(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
        if !matches!(recv, Val::Str(_)) { return err!("strip_suffix: expected string"); }
        let pat = first_str_arg(args, env, "strip_suffix")?;
        run_single(&StripSuffix::new(pat), &recv)
            .ok_or_else(|| EvalError("strip_suffix: stage filtered".into()))
    }

    pub fn repeat(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
        if !matches!(recv, Val::Str(_)) { return err!("repeat: expected string"); }
        let n = first_i64_arg(args, env, "repeat").unwrap_or(1).max(0) as usize;
        run_single(&Repeat::new(n), &recv)
            .ok_or_else(|| EvalError("repeat: stage filtered".into()))
    }

    pub fn pad_left(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
        if !matches!(recv, Val::Str(_)) { return err!("pad_left: expected string"); }
        let width = first_i64_arg(args, env, "pad_left").unwrap_or(0).max(0) as usize;
        let fill = second_char_arg(args, env);
        run_single(&PadLeft::new(width, fill), &recv)
            .ok_or_else(|| EvalError("pad_left: stage filtered".into()))
    }

    pub fn pad_right(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
        if !matches!(recv, Val::Str(_)) { return err!("pad_right: expected string"); }
        let width = first_i64_arg(args, env, "pad_right").unwrap_or(0).max(0) as usize;
        let fill = second_char_arg(args, env);
        run_single(&PadRight::new(width, fill), &recv)
            .ok_or_else(|| EvalError("pad_right: stage filtered".into()))
    }

    pub fn indent(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
        if !matches!(recv, Val::Str(_)) { return err!("indent: expected string"); }
        let n = first_i64_arg(args, env, "indent").unwrap_or(2).max(0) as usize;
        run_single(&Indent::new(n), &recv)
            .ok_or_else(|| EvalError("indent: stage filtered".into()))
    }

    pub fn str_matches(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
        if !matches!(recv, Val::Str(_)) { return err!("matches: expected string"); }
        let pat = first_str_arg(args, env, "matches")?;
        run_single(&StrMatches::new(pat), &recv)
            .ok_or_else(|| EvalError("matches: stage filtered".into()))
    }

    // ── Newly-lifted shims (composed bodies) ─────────────────────────

    delegate_str_stage!(chars_of,    CharsOf,    "chars: expected string");
    delegate_str_stage!(bytes_of,    BytesOf,    "bytes: expected string");
    delegate_str_stage!(byte_len,    ByteLen,    "byte_len: expected string");
    delegate_str_stage!(is_blank,    IsBlank,    "is_blank: expected string");
    delegate_str_stage!(is_numeric,  IsNumeric,  "is_numeric: expected string");
    delegate_str_stage!(is_alpha,    IsAlpha,    "is_alpha: expected string");
    delegate_str_stage!(is_ascii,    IsAscii,    "is_ascii: expected string");
    delegate_str_stage!(parse_int,   ParseInt,   "parse_int: expected string");
    delegate_str_stage!(parse_float, ParseFloat, "parse_float: expected string");
    delegate_str_stage!(parse_bool,  ParseBool,  "parse_bool: expected string");

    /// `.from_base64()` — explicit error semantics (errs on bad input).
    pub fn from_base64(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
        if let Val::Str(s) = recv {
            match super::from_base64_eager(&s) {
                Ok(decoded) => Ok(Val::Str(std::sync::Arc::from(decoded.as_str()))),
                Err(e)      => err!("from_base64: {}", e),
            }
        } else { err!("from_base64: expected string") }
    }

    fn pad_arg_char(args: &[Arg], idx: usize, env: &Env) -> Result<char, EvalError> {
        match args.get(idx) {
            None => Ok(' '),
            Some(a) => {
                let v = match a {
                    Arg::Pos(e) | Arg::Named(_, e) => vm_eval(e, env)?,
                };
                match v {
                    Val::Str(s) if s.chars().count() == 1 => Ok(s.chars().next().unwrap()),
                    _ => err!("pad: filler must be a single-char string"),
                }
            }
        }
    }

    pub fn center(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
        if !matches!(recv, Val::Str(_)) { return err!("center: expected string"); }
        let width = first_i64_arg(args, env, "center").unwrap_or(0).max(0) as usize;
        let fill = pad_arg_char(args, 1, env)?;
        run_single(&Center::new(width, fill), &recv)
            .ok_or_else(|| EvalError("center: stage filtered".into()))
    }

    pub fn repeat_str(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
        if !matches!(recv, Val::Str(_)) { return err!("repeat: expected string"); }
        let n = first_i64_arg(args, env, "repeat").unwrap_or(0).max(0) as usize;
        run_single(&Repeat::new(n), &recv)
            .ok_or_else(|| EvalError("repeat: stage filtered".into()))
    }

    fn collect_str_arg_list(args: &[Arg], env: &Env, who: &str) -> Result<Vec<std::sync::Arc<str>>, EvalError> {
        if args.len() == 1 {
            let v = match &args[0] {
                Arg::Pos(e) | Arg::Named(_, e) => vm_eval(e, env)?,
            };
            if let Val::Arr(a) = v {
                let mut out = Vec::with_capacity(a.len());
                for item in a.iter() {
                    if let Val::Str(s) = item { out.push(s.clone()); }
                    else { return err!("{}: array elements must be strings", who); }
                }
                return Ok(out);
            }
            // re-evaluate not needed; if it wasn't an array fall through to N-arg branch
            // by re-entering with the single arg.
            if let Val::Str(s) = match &args[0] {
                Arg::Pos(e) | Arg::Named(_, e) => vm_eval(e, env)?,
            } {
                return Ok(vec![s]);
            }
            return err!("{}: arg must be string or array of strings", who);
        }
        let mut out = Vec::with_capacity(args.len());
        for a in args {
            let v = match a {
                Arg::Pos(e) | Arg::Named(_, e) => vm_eval(e, env)?,
            };
            if let Val::Str(s) = v { out.push(s); }
            else { return err!("{}: arg must be string", who); }
        }
        Ok(out)
    }

    pub fn contains_any(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
        if !matches!(recv, Val::Str(_)) { return err!("contains_any: expected string"); }
        let needles = collect_str_arg_list(args, env, "contains_any")?;
        run_single(&ContainsAny::new(needles), &recv)
            .ok_or_else(|| EvalError("contains_any: stage filtered".into()))
    }

    pub fn contains_all(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
        if !matches!(recv, Val::Str(_)) { return err!("contains_all: expected string"); }
        let needles = collect_str_arg_list(args, env, "contains_all")?;
        run_single(&ContainsAll::new(needles), &recv)
            .ok_or_else(|| EvalError("contains_all: stage filtered".into()))
    }

    // ── Regex shims ──────────────────────────────────────────────────

    macro_rules! delegate_re_pat {
        ($shim:ident, $stage:ident, $name:literal) => {
            pub fn $shim(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
                if !matches!(recv, Val::Str(_)) { return err!("{}: expected string", $name); }
                let pat = first_str_arg(args, env, $name)?;
                // Eager compile so a bad pattern errs (consistent with prior shim).
                if let Err(e) = super::compile_regex(pat.as_ref()) {
                    return err!("{}: {}", $name, e);
                }
                run_single(&$stage::new(pat), &recv)
                    .ok_or_else(|| EvalError(format!("{}: stage filtered", $name)))
            }
        };
    }

    delegate_re_pat!(re_match,         ReMatch,       "match");
    delegate_re_pat!(re_match_first,   ReMatchFirst,  "match_first");
    delegate_re_pat!(re_match_all,     ReMatchAll,    "match_all");
    delegate_re_pat!(re_captures,      ReCaptures,    "captures");
    delegate_re_pat!(re_captures_all,  ReCapturesAll, "captures_all");
    delegate_re_pat!(re_split,         ReSplit,       "split_re");

    pub fn re_replace(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
        if !matches!(recv, Val::Str(_)) { return err!("replace_re: expected string"); }
        let pat = first_str_arg(args, env, "replace_re")?;
        let with = {
            let a = args.get(1).ok_or_else(|| EvalError("replace_re: missing replacement".into()))?;
            match a {
                Arg::Pos(Expr::Ident(s)) => std::sync::Arc::from(s.as_str()),
                Arg::Pos(e) | Arg::Named(_, e) => match vm_eval(e, env)? {
                    Val::Str(s) => s,
                    _ => return err!("replace_re: replacement must be string"),
                },
            }
        };
        if let Err(e) = super::compile_regex(pat.as_ref()) {
            return err!("replace_re: {}", e);
        }
        run_single(&ReReplace::new(pat, with), &recv)
            .ok_or_else(|| EvalError("replace_re: stage filtered".into()))
    }

    pub fn re_replace_all(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
        if !matches!(recv, Val::Str(_)) { return err!("replace_all_re: expected string"); }
        let pat = first_str_arg(args, env, "replace_all_re")?;
        let with = {
            let a = args.get(1).ok_or_else(|| EvalError("replace_all_re: missing replacement".into()))?;
            match a {
                Arg::Pos(Expr::Ident(s)) => std::sync::Arc::from(s.as_str()),
                Arg::Pos(e) | Arg::Named(_, e) => match vm_eval(e, env)? {
                    Val::Str(s) => s,
                    _ => return err!("replace_all_re: replacement must be string"),
                },
            }
        };
        if let Err(e) = super::compile_regex(pat.as_ref()) {
            return err!("replace_all_re: {}", e);
        }
        run_single(&ReReplaceAll::new(pat, with), &recv)
            .ok_or_else(|| EvalError("replace_all_re: stage filtered".into()))
    }

    // ── Array Stages (lifted bodies in composed.rs) ──────────────────

    /// Coerce typed vec receivers (IntVec/FloatVec/StrVec) into Val::Arr
    /// so Arr-only Stages apply.  Owned Val::Arr passes through via Arc clone.
    fn coerce_arr(recv: Val, who: &str) -> Result<Val, EvalError> {
        match recv {
            Val::Arr(_) => Ok(recv),
            other => other.into_vec()
                .map(Val::arr)
                .ok_or_else(|| EvalError(format!("{}: expected array", who))),
        }
    }

    pub fn compact(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
        let recv = coerce_arr(recv, "compact")?;
        run_single(&Compact, &recv)
            .ok_or_else(|| EvalError("compact: stage filtered".into()))
    }

    pub fn pairwise(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
        let recv = coerce_arr(recv, "pairwise")?;
        run_single(&Pairwise, &recv)
            .ok_or_else(|| EvalError("pairwise: stage filtered".into()))
    }

    fn vec_arg(args: &[Arg], env: &Env, who: &str) -> Result<Vec<Val>, EvalError> {
        let a = args.first().ok_or_else(|| EvalError(format!("{}: requires arg", who)))?;
        let v = match a {
            Arg::Pos(e) | Arg::Named(_, e) => vm_eval(e, env)?,
        };
        v.into_vec().ok_or_else(|| EvalError(format!("{}: expected array arg", who)))
    }

    pub fn diff(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
        let recv = coerce_arr(recv, "diff")?;
        let other = vec_arg(args, env, "diff")?;
        run_single(&Diff::new(other), &recv)
            .ok_or_else(|| EvalError("diff: stage filtered".into()))
    }

    pub fn intersect(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
        let recv = coerce_arr(recv, "intersect")?;
        let other = vec_arg(args, env, "intersect")?;
        run_single(&Intersect::new(other), &recv)
            .ok_or_else(|| EvalError("intersect: stage filtered".into()))
    }

    pub fn union(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
        let recv = coerce_arr(recv, "union")?;
        let other = vec_arg(args, env, "union")?;
        run_single(&Union::new(other), &recv)
            .ok_or_else(|| EvalError("union: stage filtered".into()))
    }

    pub fn flatten(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
        let recv = coerce_arr(recv, "flatten")?;
        let depth = first_i64_arg(args, env, "flatten").unwrap_or(1).max(0) as usize;
        run_single(&FlattenDepth::new(depth), &recv)
            .ok_or_else(|| EvalError("flatten: stage filtered".into()))
    }

    /// `.reverse()` — works on Arr / IntVec / FloatVec / StrVec / Str.
    /// No coerce_arr — Stage handles every supported variant directly.
    pub fn reverse(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
        run_single(&ReverseAny, &recv)
            .ok_or_else(|| EvalError("reverse: expected array or string".into()))
    }

    pub fn unique(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
        let recv = coerce_arr(recv, "unique")?;
        run_single(&UniqueArr, &recv)
            .ok_or_else(|| EvalError("unique: stage filtered".into()))
    }

    pub fn first(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
        // Default n=1; non-array receivers return Null (matches prior shim).
        let n = first_i64_arg(args, env, "first").unwrap_or(1);
        match recv {
            Val::Arr(_) => run_single(&First::new(n), &recv)
                .ok_or_else(|| EvalError("first: stage filtered".into())),
            other => match other.into_vec() {
                Some(v) => run_single(&First::new(n), &Val::arr(v))
                    .ok_or_else(|| EvalError("first: stage filtered".into())),
                None => Ok(Val::Null),
            },
        }
    }

    pub fn last(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
        let n = first_i64_arg(args, env, "last").unwrap_or(1);
        match recv {
            Val::Arr(_) => run_single(&Last::new(n), &recv)
                .ok_or_else(|| EvalError("last: stage filtered".into())),
            other => match other.into_vec() {
                Some(v) => run_single(&Last::new(n), &Val::arr(v))
                    .ok_or_else(|| EvalError("last: stage filtered".into())),
                None => Ok(Val::Null),
            },
        }
    }

    pub fn nth(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
        let i = first_i64_arg(args, env, "nth")?;
        run_single(&NthAny::new(i), &recv)
            .ok_or_else(|| EvalError("nth: stage filtered".into()))
    }

    pub fn append(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
        let recv = coerce_arr(recv, "append")?;
        let item = match args.first() {
            Some(a) => match a {
                Arg::Pos(e) | Arg::Named(_, e) => vm_eval(e, env)?,
            },
            None => Val::Null,
        };
        run_single(&Append::new(item), &recv)
            .ok_or_else(|| EvalError("append: stage filtered".into()))
    }

    pub fn prepend(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
        let recv = coerce_arr(recv, "prepend")?;
        let item = match args.first() {
            Some(a) => match a {
                Arg::Pos(e) | Arg::Named(_, e) => vm_eval(e, env)?,
            },
            None => Val::Null,
        };
        run_single(&Prepend::new(item), &recv)
            .ok_or_else(|| EvalError("prepend: stage filtered".into()))
    }

    // ── Object Stages ────────────────────────────────────────────────
    //
    // `keys` / `values` / `entries` LIFTED to native Pipeline IR
    // `Stage::Keys` / `Stage::Values` / `Stage::Entries`.  Canonical
    // kernels in `pipeline::{keys_apply, values_apply, entries_apply}`.
    // Method-call fallback in `eval::builtins::{keys,values,entries}_dispatch`.

    pub fn invert(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
        run_single(&Invert, &recv)
            .ok_or_else(|| EvalError("invert: expected object".into()))
    }

    fn first_val_arg(args: &[Arg], env: &Env) -> Result<Val, EvalError> {
        match args.first() {
            Some(a) => match a {
                Arg::Pos(e) | Arg::Named(_, e) => vm_eval(e, env),
            },
            None => Ok(Val::Null),
        }
    }

    pub fn merge(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
        let other = first_val_arg(args, env)?;
        run_single(&Merge::new(other), &recv)
            .ok_or_else(|| EvalError("merge: expected two objects".into()))
    }

    pub fn deep_merge(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
        let other = first_val_arg(args, env)?;
        run_single(&DeepMerge::new(other), &recv)
            .ok_or_else(|| EvalError("deep_merge: stage filtered".into()))
    }

    pub fn defaults(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
        let other = first_val_arg(args, env)?;
        run_single(&Defaults::new(other), &recv)
            .ok_or_else(|| EvalError("defaults: expected two objects".into()))
    }

    pub fn rename(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
        let renames = first_val_arg(args, env)?;
        run_single(&Rename::new(renames), &recv)
            .ok_or_else(|| EvalError("rename: expected object and rename map".into()))
    }
}

// Helper macro — generates per-builtin owned Stage impl with the
// same shape: input must be Val::Str, apply transform, return
// Pass(Owned(Str)) or Filtered.  Per-string-builtin work shrinks
// to one macro invocation + the transform body.
macro_rules! lifted_str_stage {
    ($name:ident, $transform:expr) => {
        pub struct $name;
        impl $name { pub fn new() -> Self { Self } }
        impl Default for $name { fn default() -> Self { Self } }

        impl Stage for $name {
            fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
                if let Val::Str(s) = x {
                    let f: fn(&str) -> String = $transform;
                    let out = f(s.as_ref());
                    return StageOutput::Pass(Cow::Owned(
                        Val::Str(std::sync::Arc::from(out))));
                }
                StageOutput::Filtered
            }
        }
    };
}

// `.upper()` — ASCII fast path; full Unicode fallback.
lifted_str_stage!(Upper, |s| {
    if s.is_ascii() {
        let mut buf = s.to_owned();
        buf.make_ascii_uppercase();
        buf
    } else {
        s.to_uppercase()
    }
});

// `.lower()` — ASCII fast path; full Unicode fallback.
lifted_str_stage!(Lower, |s| {
    if s.is_ascii() {
        let mut buf = s.to_owned();
        buf.make_ascii_lowercase();
        buf
    } else {
        s.to_lowercase()
    }
});

// `.trim()` — strip whitespace from both ends.
lifted_str_stage!(Trim, |s| s.trim().to_owned());

// `.trim_left()` / `.trim_start()` — strip leading whitespace.
lifted_str_stage!(TrimLeft, |s| s.trim_start().to_owned());

// `.trim_right()` / `.trim_end()` — strip trailing whitespace.
lifted_str_stage!(TrimRight, |s| s.trim_end().to_owned());

// `.capitalize()` — uppercase first char, lowercase rest (per Unicode).
lifted_str_stage!(Capitalize, |s| {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    if let Some(first) = chars.next() {
        for c in first.to_uppercase() { out.push(c); }
        out.push_str(&chars.as_str().to_lowercase());
    }
    out
});

// `.title_case()` — uppercase first char of each word, lowercase rest.
lifted_str_stage!(TitleCase, |s| {
    let mut out = String::with_capacity(s.len());
    let mut at_start = true;
    for c in s.chars() {
        if c.is_whitespace() {
            out.push(c);
            at_start = true;
        } else if at_start {
            for u in c.to_uppercase() { out.push(u); }
            at_start = false;
        } else {
            for l in c.to_lowercase() { out.push(l); }
        }
    }
    out
});

// `.html_escape()` — replace HTML special chars with entities.
lifted_str_stage!(HtmlEscape, |s| {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '&' => out.push_str("&amp;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    out
});

// Generalised macro variant: transform takes &str, returns Val
// directly (allows lines() → Arr, to_number() → Int/Float/Null,
// to_bool() → Bool, base64 → Str, etc.).
macro_rules! lifted_str_to_val {
    ($name:ident, $transform:expr) => {
        pub struct $name;
        impl $name { pub fn new() -> Self { Self } }
        impl Default for $name { fn default() -> Self { Self } }

        impl Stage for $name {
            fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
                if let Val::Str(s) = x {
                    let f: fn(&str) -> Val = $transform;
                    let out = f(s.as_ref());
                    return StageOutput::Pass(Cow::Owned(out));
                }
                StageOutput::Filtered
            }
        }
    };
}

// `.url_encode()` — percent-encode per RFC 3986 unreserved set.
lifted_str_stage!(UrlEncode, |s| {
    let mut out = String::with_capacity(s.len());
    for b in s.as_bytes() {
        let b = *b;
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9'
                | b'-' | b'_' | b'.' | b'~' => out.push(b as char),
            _ => {
                use std::fmt::Write;
                let _ = write!(out, "%{:02X}", b);
            }
        }
    }
    out
});

// `.url_decode()` — percent-decode + plus-to-space.
lifted_str_stage!(UrlDecode, |s| {
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let h1 = char::from(bytes[i + 1]).to_digit(16);
            let h2 = char::from(bytes[i + 2]).to_digit(16);
            if let (Some(h1), Some(h2)) = (h1, h2) {
                out.push((h1 * 16 + h2) as u8);
                i += 3;
                continue;
            }
        } else if bytes[i] == b'+' {
            out.push(b' ');
            i += 1;
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
});

// `.html_unescape()` — reverse the 5 entity replacements.
lifted_str_stage!(HtmlUnescape, |s| {
    s.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
     .replace("&quot;", "\"").replace("&#39;", "'")
});

// ── Val-returning string Stages (lines / words / chars / casts) ───

// `.lines()` — split into Val::Arr of Val::Str.
lifted_str_to_val!(Lines, |s| {
    Val::arr(s.lines().map(|l| Val::Str(std::sync::Arc::from(l))).collect())
});

// `.words()` — whitespace split into Val::Arr of Val::Str.
lifted_str_to_val!(Words, |s| {
    Val::arr(s.split_whitespace()
        .map(|w| Val::Str(std::sync::Arc::from(w)))
        .collect())
});

// `.chars()` — codepoint split into Val::Arr of Val::Str.
lifted_str_to_val!(Chars, |s| {
    Val::arr(s.chars()
        .map(|c| Val::Str(std::sync::Arc::from(c.to_string())))
        .collect())
});

// `.to_number()` — parse i64 or f64; null on parse failure.
lifted_str_to_val!(ToNumber, |s| {
    if let Ok(i) = s.parse::<i64>() { return Val::Int(i); }
    if let Ok(f) = s.parse::<f64>() { return Val::Float(f); }
    Val::Null
});

// `.to_bool()` — recognised "true"/"false" → Val::Bool, else Null.
lifted_str_to_val!(ToBool, |s| {
    match s {
        "true"  => Val::Bool(true),
        "false" => Val::Bool(false),
        _       => Val::Null,
    }
});

// `.to_base64()` — RFC 4648 base64 encoding.
lifted_str_stage!(ToBase64, |s| { base64_encode(s.as_bytes()) });

// `.from_base64()` — decode; non-UTF-8 bytes are lossy-converted.
// Returns Val::Null on decode failure (matches owned semantics
// for chain composition; owned form errs).
lifted_str_to_val!(FromBase64, |s| {
    match base64_decode(s) {
        Ok(bytes) => Val::Str(std::sync::Arc::from(
            String::from_utf8_lossy(&bytes).as_ref())),
        Err(_) => Val::Null,
    }
});

// ── Single-arg string Stages ───────────────────────────────────────

/// `.starts_with(prefix)` — returns Val::Bool.
pub struct StartsWith { pub prefix: std::sync::Arc<str> }
impl StartsWith {
    pub fn new(prefix: std::sync::Arc<str>) -> Self { Self { prefix } }
}
impl Stage for StartsWith {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        if let Val::Str(s) = x {
            return StageOutput::Pass(Cow::Owned(
                Val::Bool(s.starts_with(self.prefix.as_ref()))));
        }
        StageOutput::Filtered
    }
}

/// `.ends_with(suffix)` — returns Val::Bool.
pub struct EndsWith { pub suffix: std::sync::Arc<str> }
impl EndsWith {
    pub fn new(suffix: std::sync::Arc<str>) -> Self { Self { suffix } }
}
impl Stage for EndsWith {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        if let Val::Str(s) = x {
            return StageOutput::Pass(Cow::Owned(
                Val::Bool(s.ends_with(self.suffix.as_ref()))));
        }
        StageOutput::Filtered
    }
}

/// `.contains(needle)` — returns Val::Bool.
pub struct Contains { pub needle: std::sync::Arc<str> }
impl Contains {
    pub fn new(needle: std::sync::Arc<str>) -> Self { Self { needle } }
}
impl Stage for Contains {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        if let Val::Str(s) = x {
            return StageOutput::Pass(Cow::Owned(
                Val::Bool(s.contains(self.needle.as_ref()))));
        }
        StageOutput::Filtered
    }
}

/// `.repeat(n)` — repeat string n times.
pub struct Repeat { pub n: usize }
impl Repeat { pub fn new(n: usize) -> Self { Self { n } } }
impl Stage for Repeat {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        if let Val::Str(s) = x {
            return StageOutput::Pass(Cow::Owned(
                Val::Str(std::sync::Arc::from(s.repeat(self.n)))));
        }
        StageOutput::Filtered
    }
}

/// `.split(sep)` — returns Val::Arr of Val::Str.
pub struct Split { pub sep: std::sync::Arc<str> }
impl Split { pub fn new(sep: std::sync::Arc<str>) -> Self { Self { sep } } }
impl Stage for Split {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        if let Val::Str(s) = x {
            let items: Vec<Val> = s.split(self.sep.as_ref())
                .map(|p| Val::Str(std::sync::Arc::from(p)))
                .collect();
            return StageOutput::Pass(Cow::Owned(Val::arr(items)));
        }
        StageOutput::Filtered
    }
}

/// `.replace(needle, with)` — single substitution per match.
pub struct Replace {
    pub needle: std::sync::Arc<str>,
    pub with:   std::sync::Arc<str>,
}
impl Replace {
    pub fn new(needle: std::sync::Arc<str>, with: std::sync::Arc<str>) -> Self {
        Self { needle, with }
    }
}
impl Stage for Replace {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        if let Val::Str(s) = x {
            let out = s.replace(self.needle.as_ref(), self.with.as_ref());
            return StageOutput::Pass(Cow::Owned(
                Val::Str(std::sync::Arc::from(out))));
        }
        StageOutput::Filtered
    }
}

/// `.strip_prefix(prefix)` — strip if present, else return original.
pub struct StripPrefix { pub prefix: std::sync::Arc<str> }
impl StripPrefix {
    pub fn new(prefix: std::sync::Arc<str>) -> Self { Self { prefix } }
}
impl Stage for StripPrefix {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        if let Val::Str(s) = x {
            let out = s.strip_prefix(self.prefix.as_ref())
                .map(std::sync::Arc::<str>::from)
                .unwrap_or_else(|| s.clone());
            return StageOutput::Pass(Cow::Owned(Val::Str(out)));
        }
        StageOutput::Filtered
    }
}

/// `.strip_suffix(suffix)` — strip if present, else return original.
pub struct StripSuffix { pub suffix: std::sync::Arc<str> }
impl StripSuffix {
    pub fn new(suffix: std::sync::Arc<str>) -> Self { Self { suffix } }
}
impl Stage for StripSuffix {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        if let Val::Str(s) = x {
            let out = s.strip_suffix(self.suffix.as_ref())
                .map(std::sync::Arc::<str>::from)
                .unwrap_or_else(|| s.clone());
            return StageOutput::Pass(Cow::Owned(Val::Str(out)));
        }
        StageOutput::Filtered
    }
}

/// `.pad_left(width, fill)` — left-pad to width with fill char.
pub struct PadLeft { pub width: usize, pub fill: char }
impl PadLeft {
    pub fn new(width: usize, fill: char) -> Self { Self { width, fill } }
}
impl Stage for PadLeft {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        if let Val::Str(s) = x {
            let n = s.chars().count();
            if n >= self.width {
                return StageOutput::Pass(Cow::Borrowed(x));
            }
            let pad: String = std::iter::repeat(self.fill)
                .take(self.width - n).collect();
            let out = pad + s.as_ref();
            return StageOutput::Pass(Cow::Owned(
                Val::Str(std::sync::Arc::from(out))));
        }
        StageOutput::Filtered
    }
}

/// `.pad_right(width, fill)` — right-pad to width with fill char.
pub struct PadRight { pub width: usize, pub fill: char }
impl PadRight {
    pub fn new(width: usize, fill: char) -> Self { Self { width, fill } }
}
impl Stage for PadRight {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        if let Val::Str(s) = x {
            let n = s.chars().count();
            if n >= self.width {
                return StageOutput::Pass(Cow::Borrowed(x));
            }
            let pad: String = std::iter::repeat(self.fill)
                .take(self.width - n).collect();
            let out = s.to_string() + &pad;
            return StageOutput::Pass(Cow::Owned(
                Val::Str(std::sync::Arc::from(out))));
        }
        StageOutput::Filtered
    }
}

/// `.indent(n)` — prepend n spaces to each line.
pub struct Indent { pub n: usize }
impl Indent { pub fn new(n: usize) -> Self { Self { n } } }
impl Stage for Indent {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        if let Val::Str(s) = x {
            let prefix: String = std::iter::repeat(' ').take(self.n).collect();
            let out = s.lines()
                .map(|l| format!("{}{}", prefix, l))
                .collect::<Vec<_>>()
                .join("\n");
            return StageOutput::Pass(Cow::Owned(
                Val::Str(std::sync::Arc::from(out))));
        }
        StageOutput::Filtered
    }
}

// `.dedent()` — strip common leading whitespace from all non-empty lines.
lifted_str_stage!(Dedent, |s| {
    let min_indent = s.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.len() - l.trim_start().len())
        .min().unwrap_or(0);
    s.lines()
        .map(|l| if l.len() >= min_indent { &l[min_indent..] } else { l })
        .collect::<Vec<_>>()
        .join("\n")
});

/// `.str_matches(needle)` — alias for contains; returns Val::Bool.
pub struct StrMatches { pub needle: std::sync::Arc<str> }
impl StrMatches {
    pub fn new(needle: std::sync::Arc<str>) -> Self { Self { needle } }
}
impl Stage for StrMatches {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        if let Val::Str(s) = x {
            return StageOutput::Pass(Cow::Owned(
                Val::Bool(s.contains(self.needle.as_ref()))));
        }
        StageOutput::Filtered
    }
}

/// `.index_of(needle)` — char-position of first match; -1 on miss.
pub struct IndexOf { pub needle: std::sync::Arc<str> }
impl IndexOf {
    pub fn new(needle: std::sync::Arc<str>) -> Self { Self { needle } }
}
impl Stage for IndexOf {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        if let Val::Str(s) = x {
            let r = match s.find(self.needle.as_ref()) {
                Some(i) => Val::Int(s[..i].chars().count() as i64),
                None    => Val::Int(-1),
            };
            return StageOutput::Pass(Cow::Owned(r));
        }
        StageOutput::Filtered
    }
}

// ── Array / iterable Stages (lift_all_builtins arrays family) ─────
//
// Operate on Val::Arr inputs (columnar lanes IntVec/FloatVec/StrVec/
// StrSliceVec defer to future per-stage lane-aware impls — for now
// they Filter, matching the conservative semantics).  Each Stage
// has owned composed::Stage impl + unified::Stage<BVal> placeholder.

/// `.len()` / `.count()` — number of elements; works on Arr only
/// in this lift (columnar lane support deferred).
pub struct Len;
impl Len { pub fn new() -> Self { Self } }
impl Default for Len { fn default() -> Self { Self } }
impl Stage for Len {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        let n = match x {
            Val::Arr(a)         => a.len(),
            Val::IntVec(a)      => a.len(),
            Val::FloatVec(a)    => a.len(),
            Val::StrVec(a)      => a.len(),
            Val::StrSliceVec(a) => a.len(),
            Val::Obj(m)         => m.len(),
            Val::Str(s)         => s.chars().count(),
            _ => return StageOutput::Filtered,
        };
        StageOutput::Pass(Cow::Owned(Val::Int(n as i64)))
    }
}

/// `.compact()` — drop Null entries from an Arr.
pub struct Compact;
impl Compact { pub fn new() -> Self { Self } }
impl Default for Compact { fn default() -> Self { Self } }
impl Stage for Compact {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        let items_cow = match x.as_vals() { Some(c) => c, None => return StageOutput::Filtered };
        let kept: Vec<Val> = items_cow.iter()
            .filter(|v| !matches!(v, Val::Null))
            .cloned()
            .collect();
        StageOutput::Pass(Cow::Owned(Val::arr(kept)))
    }
}

/// `.flatten()` — one level of array flattening.
pub struct FlattenOne;
impl FlattenOne { pub fn new() -> Self { Self } }
impl Default for FlattenOne { fn default() -> Self { Self } }
impl Stage for FlattenOne {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        if let Val::Arr(outer) = x {
            let mut out: Vec<Val> = Vec::new();
            for v in outer.iter() {
                match v {
                    Val::Arr(inner) => out.extend(inner.iter().cloned()),
                    other => out.push(other.clone()),
                }
            }
            return StageOutput::Pass(Cow::Owned(Val::arr(out)));
        }
        StageOutput::Filtered
    }
}

/// `.flatten(depth)` — recursively flatten up to `depth` levels of
/// nested arrays.  Delegates to `eval::util::flatten_val` for the
/// columnar fast path (Arr<IntVec> → IntVec etc.).
pub struct FlattenDepth { pub depth: usize }
impl FlattenDepth { pub fn new(depth: usize) -> Self { Self { depth } } }
impl Stage for FlattenDepth {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        if matches!(x, Val::Arr(_)) {
            return StageOutput::Pass(Cow::Owned(
                crate::eval::util::flatten_val(x.clone(), self.depth)));
        }
        StageOutput::Filtered
    }
}

/// `.reverse()` — reverse Arr / IntVec / FloatVec / StrVec / Str.
pub struct ReverseAny;
impl ReverseAny { pub fn new() -> Self { Self } }
impl Default for ReverseAny { fn default() -> Self { Self } }
impl Stage for ReverseAny {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        let out = match x {
            Val::Arr(a) => {
                let mut v: Vec<Val> = a.as_ref().clone();
                v.reverse();
                Val::arr(v)
            }
            Val::IntVec(a) => {
                let mut v: Vec<i64> = a.as_ref().clone();
                v.reverse();
                Val::int_vec(v)
            }
            Val::FloatVec(a) => {
                let mut v: Vec<f64> = a.as_ref().clone();
                v.reverse();
                Val::float_vec(v)
            }
            Val::StrVec(a) => {
                let mut v: Vec<std::sync::Arc<str>> = a.as_ref().clone();
                v.reverse();
                Val::str_vec(v)
            }
            Val::Str(s) => Val::Str(std::sync::Arc::<str>::from(
                s.chars().rev().collect::<String>())),
            _ => return StageOutput::Filtered,
        };
        StageOutput::Pass(Cow::Owned(out))
    }
}

/// `.unique()` / `.distinct()` — dedup by canonical key.
pub struct UniqueArr;
impl UniqueArr { pub fn new() -> Self { Self } }
impl Default for UniqueArr { fn default() -> Self { Self } }
impl Stage for UniqueArr {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        if let Val::Arr(a) = x {
            let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
            let kept: Vec<Val> = a.iter()
                .filter(|v| seen.insert(crate::eval::util::val_to_key(v)))
                .cloned()
                .collect();
            return StageOutput::Pass(Cow::Owned(Val::arr(kept)));
        }
        StageOutput::Filtered
    }
}

/// `.first(n)` — n==1 returns scalar (or Null), else Arr of first n.
pub struct First { pub n: i64 }
impl First { pub fn new(n: i64) -> Self { Self { n } } }
impl Stage for First {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        if let Val::Arr(a) = x {
            let out = if self.n == 1 {
                a.first().cloned().unwrap_or(Val::Null)
            } else {
                Val::arr(a.iter().take(self.n.max(0) as usize).cloned().collect())
            };
            return StageOutput::Pass(Cow::Owned(out));
        }
        StageOutput::Pass(Cow::Owned(Val::Null))
    }
}

/// `.last(n)` — n==1 returns scalar (or Null), else Arr of last n.
pub struct Last { pub n: i64 }
impl Last { pub fn new(n: i64) -> Self { Self { n } } }
impl Stage for Last {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        if let Val::Arr(a) = x {
            let out = if self.n == 1 {
                a.last().cloned().unwrap_or(Val::Null)
            } else {
                let s = a.len().saturating_sub(self.n.max(0) as usize);
                Val::arr(a[s..].to_vec())
            };
            return StageOutput::Pass(Cow::Owned(out));
        }
        StageOutput::Pass(Cow::Owned(Val::Null))
    }
}

/// `.nth(i)` — index lookup; supports Arr/IntVec/FloatVec/StrVec/ObjVec.
/// Negative `i` indexes from the end.
pub struct NthAny { pub i: i64 }
impl NthAny { pub fn new(i: i64) -> Self { Self { i } } }
impl Stage for NthAny {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        StageOutput::Pass(Cow::Owned(x.get_index(self.i)))
    }
}

/// `.append(item)` — push item onto end.  Coerces typed vecs via
/// `into_vec`.  Stage holds the item as owned Val.
pub struct Append { pub item: Val }
impl Append { pub fn new(item: Val) -> Self { Self { item } } }
impl Stage for Append {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        let mut v = match x.clone().into_vec() {
            Some(v) => v,
            None => return StageOutput::Filtered,
        };
        v.push(self.item.clone());
        StageOutput::Pass(Cow::Owned(Val::arr(v)))
    }
}

/// `.prepend(item)` — insert item at front.
pub struct Prepend { pub item: Val }
impl Prepend { pub fn new(item: Val) -> Self { Self { item } } }
impl Stage for Prepend {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        let mut v = match x.clone().into_vec() {
            Some(v) => v,
            None => return StageOutput::Filtered,
        };
        v.insert(0, self.item.clone());
        StageOutput::Pass(Cow::Owned(Val::arr(v)))
    }
}

/// `.enumerate()` — Arr → Arr of [index, item] pairs (as Val::Arr).
pub struct Enumerate;
impl Enumerate { pub fn new() -> Self { Self } }
impl Default for Enumerate { fn default() -> Self { Self } }
impl Stage for Enumerate {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        if let Val::Arr(a) = x {
            let pairs: Vec<Val> = a.iter().enumerate()
                .map(|(i, v)| Val::arr(vec![Val::Int(i as i64), v.clone()]))
                .collect();
            return StageOutput::Pass(Cow::Owned(Val::arr(pairs)));
        }
        StageOutput::Filtered
    }
}

/// `.pairwise()` — Arr → Arr of [arr[i], arr[i+1]] adjacent pairs.
pub struct Pairwise;
impl Pairwise { pub fn new() -> Self { Self } }
impl Default for Pairwise { fn default() -> Self { Self } }
impl Stage for Pairwise {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        let items_cow = match x.as_vals() { Some(c) => c, None => return StageOutput::Filtered };
        let a = items_cow.as_ref();
        let mut out: Vec<Val> = Vec::with_capacity(a.len().saturating_sub(1));
        for w in a.windows(2) {
            out.push(Val::arr(vec![w[0].clone(), w[1].clone()]));
        }
        StageOutput::Pass(Cow::Owned(Val::arr(out)))
    }
}

/// `.chunk(n)` — split into Arr of length-n Arrs (last may be shorter).
pub struct Chunk { pub n: usize }
impl Chunk { pub fn new(n: usize) -> Self { Self { n } } }
impl Stage for Chunk {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        if self.n == 0 { return StageOutput::Filtered; }
        if let Val::Arr(a) = x {
            let chunks: Vec<Val> = a.chunks(self.n)
                .map(|c| Val::arr(c.to_vec()))
                .collect();
            return StageOutput::Pass(Cow::Owned(Val::arr(chunks)));
        }
        StageOutput::Filtered
    }
}

/// `.window(n)` — Arr → Arr of length-n sliding windows.
pub struct Window { pub n: usize }
impl Window { pub fn new(n: usize) -> Self { Self { n } } }
impl Stage for Window {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        if self.n == 0 { return StageOutput::Filtered; }
        if let Val::Arr(a) = x {
            let windows: Vec<Val> = a.windows(self.n)
                .map(|w| Val::arr(w.to_vec()))
                .collect();
            return StageOutput::Pass(Cow::Owned(Val::arr(windows)));
        }
        StageOutput::Filtered
    }
}

// ── Object Stages (lift_all_builtins objects family) ─────────────

/// `.keys()` — Obj → Arr<Str>.
pub struct Keys;
impl Keys { pub fn new() -> Self { Self } }
impl Default for Keys { fn default() -> Self { Self } }
impl Stage for Keys {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        let m = match x.as_object() { Some(m) => m, None => return StageOutput::Filtered };
        let items: Vec<Val> = m.keys().map(|k| Val::Str(k.clone())).collect();
        StageOutput::Pass(Cow::Owned(Val::arr(items)))
    }
}

/// `.values()` — Obj → Arr.
pub struct Values;
impl Values { pub fn new() -> Self { Self } }
impl Default for Values { fn default() -> Self { Self } }
impl Stage for Values {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        let m = match x.as_object() { Some(m) => m, None => return StageOutput::Filtered };
        let items: Vec<Val> = m.values().cloned().collect();
        StageOutput::Pass(Cow::Owned(Val::arr(items)))
    }
}

/// `.entries()` / `.to_pairs()` — Obj → Arr<[Str, Val]>.
pub struct Entries;
impl Entries { pub fn new() -> Self { Self } }
impl Default for Entries { fn default() -> Self { Self } }
impl Stage for Entries {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        let m = match x.as_object() { Some(m) => m, None => return StageOutput::Filtered };
        let pairs: Vec<Val> = m.iter().map(|(k, v)| {
            Val::arr(vec![Val::Str(k.clone()), v.clone()])
        }).collect();
        StageOutput::Pass(Cow::Owned(Val::arr(pairs)))
    }
}

/// `.from_pairs()` — Arr<[Str, Val]> → Obj.
pub struct FromPairs;
impl FromPairs { pub fn new() -> Self { Self } }
impl Default for FromPairs { fn default() -> Self { Self } }
impl Stage for FromPairs {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        if let Val::Arr(pairs) = x {
            let mut m: indexmap::IndexMap<std::sync::Arc<str>, Val> =
                indexmap::IndexMap::with_capacity(pairs.len());
            for p in pairs.iter() {
                if let Val::Arr(kv) = p {
                    if kv.len() == 2 {
                        if let Val::Str(k) = &kv[0] {
                            m.insert(k.clone(), kv[1].clone());
                        }
                    }
                }
            }
            return StageOutput::Pass(Cow::Owned(Val::Obj(std::sync::Arc::new(m))));
        }
        StageOutput::Filtered
    }
}

/// `.invert()` — Obj{k → v} → Obj{v_str → k} (values stringified).
pub struct Invert;
impl Invert { pub fn new() -> Self { Self } }
impl Default for Invert { fn default() -> Self { Self } }
impl Stage for Invert {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        let m = match x.as_object() { Some(m) => m, None => return StageOutput::Filtered };
        let mut out: indexmap::IndexMap<std::sync::Arc<str>, Val> =
            indexmap::IndexMap::with_capacity(m.len());
        for (k, v) in m.iter() {
            let new_key: std::sync::Arc<str> = match v {
                Val::Str(s) => s.clone(),
                other       => std::sync::Arc::<str>::from(
                    crate::eval::util::val_to_key(other).as_str()),
            };
            out.insert(new_key, Val::Str(k.clone()));
        }
        StageOutput::Pass(Cow::Owned(Val::Obj(std::sync::Arc::new(out))))
    }
}

/// `.merge(other)` — shallow merge; keys in `other` override receiver.
pub struct Merge { pub other: Val }
impl Merge { pub fn new(other: Val) -> Self { Self { other } } }
impl Stage for Merge {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        let base = match x.as_object() { Some(m) => m, None => return StageOutput::Filtered };
        let other = match self.other.as_object() { Some(m) => m, None => return StageOutput::Filtered };
        let mut out: indexmap::IndexMap<std::sync::Arc<str>, Val> = base.clone();
        for (k, v) in other.iter() { out.insert(k.clone(), v.clone()); }
        StageOutput::Pass(Cow::Owned(Val::Obj(std::sync::Arc::new(out))))
    }
}

/// `.deep_merge(other)` — recursive merge; nested objects merge by key.
pub struct DeepMerge { pub other: Val }
impl DeepMerge { pub fn new(other: Val) -> Self { Self { other } } }
impl Stage for DeepMerge {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        StageOutput::Pass(Cow::Owned(
            crate::eval::util::deep_merge(x.clone(), self.other.clone())))
    }
}

/// `.defaults(other)` — fill null/missing keys from `other`.
pub struct Defaults { pub other: Val }
impl Defaults { pub fn new(other: Val) -> Self { Self { other } } }
impl Stage for Defaults {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        let base = match x.as_object() { Some(m) => m, None => return StageOutput::Filtered };
        let defs = match self.other.as_object() { Some(m) => m, None => return StageOutput::Filtered };
        let mut out: indexmap::IndexMap<std::sync::Arc<str>, Val> = base.clone();
        for (k, v) in defs.iter() {
            let entry = out.entry(k.clone()).or_insert(Val::Null);
            if entry.is_null() { *entry = v.clone(); }
        }
        StageOutput::Pass(Cow::Owned(Val::Obj(std::sync::Arc::new(out))))
    }
}

/// `.rename({old: new, ...})` — rename keys per mapping object.
pub struct Rename { pub renames: Val }
impl Rename { pub fn new(renames: Val) -> Self { Self { renames } } }
impl Stage for Rename {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        let base = match x.as_object() { Some(m) => m, None => return StageOutput::Filtered };
        let renames = match self.renames.as_object() { Some(m) => m, None => return StageOutput::Filtered };
        let mut out: indexmap::IndexMap<std::sync::Arc<str>, Val> = base.clone();
        for (old, new_val) in renames.iter() {
            if let Some(v) = out.shift_remove(old.as_ref()) {
                let new_key: std::sync::Arc<str> = if let Val::Str(s) = new_val {
                    s.clone()
                } else { old.clone() };
                out.insert(new_key, v);
            }
        }
        StageOutput::Pass(Cow::Owned(Val::Obj(std::sync::Arc::new(out))))
    }
}

// ── Set Stages (lift_all_builtins sets family) ─────────────────────
//
// Take a `other: Vec<Val>` set parameter and produce a new Arr.
// Reuses `eval::util::val_to_key` for hashable canonical-string
// repr (matches owned semantics).

/// `.intersect(other)` — keep elements present in both arrays.
pub struct Intersect { pub other: Vec<Val> }
impl Intersect { pub fn new(other: Vec<Val>) -> Self { Self { other } } }
impl Stage for Intersect {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        if let Val::Arr(a) = x {
            let other_keys: std::collections::HashSet<String> =
                self.other.iter().map(crate::eval::util::val_to_key).collect();
            let kept: Vec<Val> = a.iter()
                .filter(|v| other_keys.contains(&crate::eval::util::val_to_key(v)))
                .cloned()
                .collect();
            return StageOutput::Pass(Cow::Owned(Val::arr(kept)));
        }
        StageOutput::Filtered
    }
}

/// `.union(other)` — combine, preserve order, dedup.
pub struct Union { pub other: Vec<Val> }
impl Union { pub fn new(other: Vec<Val>) -> Self { Self { other } } }
impl Stage for Union {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        if let Val::Arr(a) = x {
            let mut out: Vec<Val> = a.as_ref().clone();
            let a_keys: std::collections::HashSet<String> =
                a.iter().map(crate::eval::util::val_to_key).collect();
            for v in &self.other {
                if !a_keys.contains(&crate::eval::util::val_to_key(v)) {
                    out.push(v.clone());
                }
            }
            return StageOutput::Pass(Cow::Owned(Val::arr(out)));
        }
        StageOutput::Filtered
    }
}

/// `.diff(other)` — keep elements present in self but not in other.
pub struct Diff { pub other: Vec<Val> }
impl Diff { pub fn new(other: Vec<Val>) -> Self { Self { other } } }
impl Stage for Diff {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        if let Val::Arr(a) = x {
            let other_keys: std::collections::HashSet<String> =
                self.other.iter().map(crate::eval::util::val_to_key).collect();
            let kept: Vec<Val> = a.iter()
                .filter(|v| !other_keys.contains(&crate::eval::util::val_to_key(v)))
                .cloned()
                .collect();
            return StageOutput::Pass(Cow::Owned(Val::arr(kept)));
        }
        StageOutput::Filtered
    }
}

// ── Path Stages (lift_all_builtins paths family) ───────────────────

/// `.get_path(path)` — read leaf at dotted/bracket path.
pub struct GetPath { pub path: std::sync::Arc<str> }
impl GetPath {
    pub fn new(path: std::sync::Arc<str>) -> Self { Self { path } }
}
impl Stage for GetPath {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        let segs = crate::eval::func_paths::parse_path_segs(self.path.as_ref());
        let v = crate::eval::func_paths::get_path_impl(x, &segs);
        StageOutput::Pass(Cow::Owned(v))
    }
}

/// `.has_path(path)` — does the path resolve to a non-null value?
pub struct HasPath { pub path: std::sync::Arc<str> }
impl HasPath {
    pub fn new(path: std::sync::Arc<str>) -> Self { Self { path } }
}
impl Stage for HasPath {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        let segs = crate::eval::func_paths::parse_path_segs(self.path.as_ref());
        let found = !crate::eval::func_paths::get_path_impl(x, &segs).is_null();
        StageOutput::Pass(Cow::Owned(Val::Bool(found)))
    }
}

/// `.has(key)` — does Obj contain `key`? Val::Bool.
pub struct Has { pub key: std::sync::Arc<str> }
impl Has { pub fn new(key: std::sync::Arc<str>) -> Self { Self { key } } }
impl Stage for Has {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        let m = match x.as_object() { Some(m) => m, None => return StageOutput::Filtered };
        let found = m.contains_key(self.key.as_ref());
        StageOutput::Pass(Cow::Owned(Val::Bool(found)))
    }
}

/// `.pick([keys])` — narrow Obj to selected keys (as-is, no rename).
pub struct Pick { pub keys: Vec<std::sync::Arc<str>> }
impl Pick {
    pub fn new(keys: Vec<std::sync::Arc<str>>) -> Self { Self { keys } }
}
impl Stage for Pick {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        let m = match x.as_object() { Some(m) => m, None => return StageOutput::Filtered };
        let mut out: indexmap::IndexMap<std::sync::Arc<str>, Val> =
            indexmap::IndexMap::with_capacity(self.keys.len());
        for k in &self.keys {
            if let Some(v) = m.get(k.as_ref()) {
                out.insert(k.clone(), v.clone());
            }
        }
        StageOutput::Pass(Cow::Owned(Val::Obj(std::sync::Arc::new(out))))
    }
}

/// `.omit([keys])` — drop selected keys from Obj.
pub struct Omit { pub keys: Vec<std::sync::Arc<str>> }
impl Omit {
    pub fn new(keys: Vec<std::sync::Arc<str>>) -> Self { Self { keys } }
}
impl Stage for Omit {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        let m = match x.as_object() { Some(m) => m, None => return StageOutput::Filtered };
        let drop: std::collections::HashSet<&str> =
            self.keys.iter().map(|k| k.as_ref()).collect();
        let mut out: indexmap::IndexMap<std::sync::Arc<str>, Val> =
            indexmap::IndexMap::with_capacity(m.len());
        for (k, v) in m.iter() {
            if !drop.contains(k.as_ref()) {
                out.insert(k.clone(), v.clone());
            }
        }
        StageOutput::Pass(Cow::Owned(Val::Obj(std::sync::Arc::new(out))))
    }
}

/// `.nth(i)` — index into Arr; supports negative indexing.
pub struct Nth { pub i: i64 }
impl Nth { pub fn new(i: i64) -> Self { Self { i } } }
impl Stage for Nth {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        if let Val::Arr(a) = x {
            let len = a.len();
            let idx = if self.i < 0 {
                len.saturating_sub(self.i.unsigned_abs() as usize)
            } else { self.i as usize };
            return StageOutput::Pass(Cow::Owned(
                a.get(idx).cloned().unwrap_or(Val::Null)));
        }
        StageOutput::Filtered
    }
}

// ── Case-conversion Stages ─────────────────────────────────────────

/// Split into lowercased word tokens.  Tokens break on `[-_ \t]`,
/// lower→upper transitions, and non-alphanumeric boundaries.  Used
/// by SnakeCase / KebabCase / CamelCase / PascalCase.
pub(crate) fn split_words_lower(s: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut prev_lower = false;
    for c in s.chars() {
        let is_sep = c == '_' || c == '-' || c.is_whitespace();
        if is_sep {
            if !cur.is_empty() { out.push(std::mem::take(&mut cur)); }
            prev_lower = false;
            continue;
        }
        if c.is_uppercase() && prev_lower {
            out.push(std::mem::take(&mut cur));
        }
        for d in c.to_lowercase() { cur.push(d); }
        prev_lower = c.is_lowercase();
    }
    if !cur.is_empty() { out.push(cur); }
    out
}

pub(crate) fn upper_first_into(p: &str, out: &mut String) {
    let mut chars = p.chars();
    if let Some(f) = chars.next() {
        for u in f.to_uppercase() { out.push(u); }
        out.push_str(chars.as_str());
    }
}

lifted_str_stage!(SnakeCase,  |s| split_words_lower(s).join("_"));
lifted_str_stage!(KebabCase,  |s| split_words_lower(s).join("-"));
lifted_str_stage!(CamelCase,  |s| {
    let parts = split_words_lower(s);
    let mut out = String::with_capacity(s.len());
    for (i, p) in parts.iter().enumerate() {
        if i == 0 { out.push_str(p); }
        else { upper_first_into(p, &mut out); }
    }
    out
});
lifted_str_stage!(PascalCase, |s| {
    let parts = split_words_lower(s);
    let mut out = String::with_capacity(s.len());
    for p in parts.iter() { upper_first_into(p, &mut out); }
    out
});

// `.reverse()` on a string — char-reversed (Unicode-aware).
lifted_str_stage!(ReverseStr, |s| s.chars().rev().collect::<String>());

/// `.scan(pat)` — collect all occurrences of `pat` in `s`.
pub struct Scan { pub pat: std::sync::Arc<str> }
impl Scan { pub fn new(pat: std::sync::Arc<str>) -> Self { Self { pat } } }
impl Stage for Scan {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        if let Val::Str(s) = x {
            let pat = self.pat.as_ref();
            let mut out: Vec<Val> = Vec::new();
            if !pat.is_empty() {
                let mut start = 0usize;
                while let Some(pos) = s[start..].find(pat) {
                    out.push(Val::Str(std::sync::Arc::from(pat)));
                    start += pos + pat.len();
                }
            }
            return StageOutput::Pass(Cow::Owned(Val::arr(out)));
        }
        StageOutput::Filtered
    }
}

/// ── Base64 helpers (moved from eval::func_strings) ─────────────────

pub fn base64_encode(bytes: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((bytes.len() + 2) / 3 * 4);
    let mut i = 0;
    while i < bytes.len() {
        let b0 = bytes[i] as u32;
        let b1 = if i + 1 < bytes.len() { bytes[i + 1] as u32 } else { 0 };
        let b2 = if i + 2 < bytes.len() { bytes[i + 2] as u32 } else { 0 };
        let n  = (b0 << 16) | (b1 << 8) | b2;
        out.push(CHARS[((n >> 18) & 0x3f) as usize] as char);
        out.push(CHARS[((n >> 12) & 0x3f) as usize] as char);
        out.push(if i + 1 < bytes.len() { CHARS[((n >> 6) & 0x3f) as usize] as char } else { '=' });
        out.push(if i + 2 < bytes.len() { CHARS[(n & 0x3f) as usize] as char } else { '=' });
        i += 3;
    }
    out
}

pub fn base64_decode(s: &str) -> Result<Vec<u8>, String> {
    const DECODE: [i8; 128] = {
        let mut t = [-1i8; 128];
        let chars = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut i = 0usize;
        while i < chars.len() { t[chars[i] as usize] = i as i8; i += 1; }
        t
    };
    let s = s.trim();
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    let mut i = 0;
    while i + 3 < bytes.len() {
        let dc = |c: u8| -> Result<u32, String> {
            if c == b'=' { return Ok(0); }
            if c as usize >= 128 || DECODE[c as usize] < 0 {
                return Err(format!("invalid base64 char: {}", c as char));
            }
            Ok(DECODE[c as usize] as u32)
        };
        let b0 = dc(bytes[i])?;
        let b1 = dc(bytes[i + 1])?;
        let b2 = dc(bytes[i + 2])?;
        let b3 = dc(bytes[i + 3])?;
        let n  = (b0 << 18) | (b1 << 12) | (b2 << 6) | b3;
        out.push(((n >> 16) & 0xff) as u8);
        if bytes[i + 2] != b'=' { out.push(((n >> 8) & 0xff) as u8); }
        if bytes[i + 3] != b'=' { out.push((n & 0xff) as u8); }
        i += 4;
    }
    Ok(out)
}

// ── Padding / repetition Stages ────────────────────────────────────

/// `.center(width, fill)` — center-pad to width with fill char.
pub struct Center { pub width: usize, pub fill: char }
impl Center { pub fn new(width: usize, fill: char) -> Self { Self { width, fill } } }
impl Stage for Center {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        if let Val::Str(s) = x {
            let cur = s.chars().count();
            if cur >= self.width { return StageOutput::Pass(Cow::Borrowed(x)); }
            let total = self.width - cur;
            let left = total / 2;
            let right = total - left;
            let mut out = String::with_capacity(s.len() + total);
            for _ in 0..left { out.push(self.fill); }
            out.push_str(s.as_ref());
            for _ in 0..right { out.push(self.fill); }
            return StageOutput::Pass(Cow::Owned(Val::Str(std::sync::Arc::from(out))));
        }
        StageOutput::Filtered
    }
}

/// `.repeat_str(n)` — alias of `.repeat(n)`; same as `Repeat`.
pub use Repeat as RepeatStr;

// ── Char / byte introspection Stages ───────────────────────────────

// `.chars_of()` — Str → Arr<Str> (one Str per Unicode char).
lifted_str_to_val!(CharsOf, |s| {
    let mut out: Vec<Val> = Vec::new();
    let mut tmp = [0u8; 4];
    for c in s.chars() {
        let utf8 = c.encode_utf8(&mut tmp);
        out.push(Val::Str(std::sync::Arc::from(utf8.as_ref())));
    }
    Val::arr(out)
});

// `.bytes()` — Str → IntVec of byte values.
lifted_str_to_val!(BytesOf, |s| {
    let v: Vec<i64> = s.as_bytes().iter().map(|&b| b as i64).collect();
    Val::int_vec(v)
});

// `.byte_len()` — Str → Int byte count.
lifted_str_to_val!(ByteLen, |s| Val::Int(s.len() as i64));

// ── Predicates / parsers ───────────────────────────────────────────

lifted_str_to_val!(IsBlank,   |s| Val::Bool(s.chars().all(|c| c.is_whitespace())));
lifted_str_to_val!(IsNumeric, |s| Val::Bool(!s.is_empty() && s.chars().all(|c| c.is_ascii_digit())));
lifted_str_to_val!(IsAlpha,   |s| Val::Bool(!s.is_empty() && s.chars().all(|c| c.is_alphabetic())));
lifted_str_to_val!(IsAscii,   |s| Val::Bool(s.is_ascii()));

lifted_str_to_val!(ParseInt,   |s| s.trim().parse::<i64>().map(Val::Int).unwrap_or(Val::Null));
lifted_str_to_val!(ParseFloat, |s| s.trim().parse::<f64>().map(Val::Float).unwrap_or(Val::Null));
lifted_str_to_val!(ParseBool,  |s| match s.trim().to_ascii_lowercase().as_str() {
    "true" | "yes" | "1" | "on"  => Val::Bool(true),
    "false" | "no" | "0" | "off" => Val::Bool(false),
    _ => Val::Null,
});

// ── Lifted: zero-arg eval/func_* migrations ─────────────────────────
//
// Per `lift_all_builtins.md`: bodies stay single-source in their
// existing `eval/func_*.rs` files; this module adds the `Stage` trait
// wrapper so the planner sees them as first-class.  No body
// duplication — apply just delegates.

// CSV/TSV body — single source of truth here in composed.rs.
// `eval/func_csv.rs` is now a thin re-export shim.

#[inline]
fn csv_cell(v: &Val, sep: &str) -> String {
    use crate::eval::util::val_to_string;
    match v {
        Val::Str(s) if s.contains(sep) || s.contains('"') || s.contains('\n') => {
            format!("\"{}\"", s.replace('"', "\"\""))
        }
        Val::Str(s) => s.to_string(),
        other       => val_to_string(other),
    }
}

pub(crate) fn csv_emit(val: &Val, sep: &str) -> String {
    match val {
        Val::Arr(rows) => rows.iter().map(|row| match row {
            Val::Arr(cells) => cells.iter().map(|c| csv_cell(c, sep)).collect::<Vec<_>>().join(sep),
            Val::Obj(m)     => m.values().map(|c| csv_cell(c, sep)).collect::<Vec<_>>().join(sep),
            v               => csv_cell(v, sep),
        }).collect::<Vec<_>>().join("\n"),
        Val::Obj(m) => m.values().map(|c| csv_cell(c, sep)).collect::<Vec<_>>().join(sep),
        v => csv_cell(v, sep),
    }
}

/// `.to_csv()` — Val → Str (CSV emission).
pub struct ToCsv;
impl ToCsv { pub fn new() -> Self { Self } }
impl Default for ToCsv { fn default() -> Self { Self } }
impl Stage for ToCsv {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        let s = csv_emit(x, ",");
        StageOutput::Pass(Cow::Owned(Val::Str(std::sync::Arc::from(s))))
    }
}

/// `.to_tsv()` — Val → Str (TSV emission).
pub struct ToTsv;
impl ToTsv { pub fn new() -> Self { Self } }
impl Default for ToTsv { fn default() -> Self { Self } }
impl Stage for ToTsv {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        let s = csv_emit(x, "\t");
        StageOutput::Pass(Cow::Owned(Val::Str(std::sync::Arc::from(s))))
    }
}

/// `.to_pairs()` — Obj → Arr<{key, val}>.  Body lives here.
pub struct ToPairs;
impl ToPairs { pub fn new() -> Self { Self } }
impl Default for ToPairs { fn default() -> Self { Self } }
impl Stage for ToPairs {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        use crate::eval::util::obj2;
        let arr: Vec<Val> = x.as_object().map(|m| m.iter().map(|(k, v)| {
            obj2("key", Val::Str(k.clone()), "val", v.clone())
        }).collect()).unwrap_or_default();
        StageOutput::Pass(Cow::Owned(Val::arr(arr)))
    }
}

// (FromPairs already defined above as positional `[k,v]` pair Stage;
//  the `from_pairs` surface name keeps legacy `b_from_pairs` dispatch
//  for the named-obj `{key, val}` format until the two semantics are
//  unified.  No duplicate Stage variant added here.)

// ── Numeric-array windowed Stages — bodies live here ─────────────
//
// Helpers `to_floats` / `floats_to_val` re-exported pub(crate) from
// `eval::func_arrays` (still consumed by rolling_avg / lag / lead /
// zscore there).  Single source of truth for the windowed kernels
// is in this module; eval/func_arrays.rs becomes a thin shim.

macro_rules! lifted_num_arr_stage {
    ($name:ident, $body:expr) => {
        pub struct $name;
        impl $name { pub fn new() -> Self { Self } }
        impl Default for $name { fn default() -> Self { Self } }
        impl Stage for $name {
            fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
                use crate::eval::func_arrays::{to_floats, floats_to_val};
                let xs = match to_floats(x) { Ok(v) => v, Err(_) => return StageOutput::Filtered };
                let f: fn(&[Option<f64>]) -> Vec<Option<f64>> = $body;
                StageOutput::Pass(Cow::Owned(floats_to_val(f(&xs))))
            }
        }
    };
}

lifted_num_arr_stage!(CumMax, |xs| {
    let mut out: Vec<Option<f64>> = Vec::with_capacity(xs.len());
    let mut best: Option<f64> = None;
    for v in xs.iter() {
        match (*v, best) {
            (Some(x), Some(b)) => { best = Some(x.max(b)); out.push(best); }
            (Some(x), None)    => { best = Some(x);        out.push(best); }
            (None, _)          => { out.push(best); }
        }
    }
    out
});

lifted_num_arr_stage!(CumMin, |xs| {
    let mut out: Vec<Option<f64>> = Vec::with_capacity(xs.len());
    let mut best: Option<f64> = None;
    for v in xs.iter() {
        match (*v, best) {
            (Some(x), Some(b)) => { best = Some(x.min(b)); out.push(best); }
            (Some(x), None)    => { best = Some(x);        out.push(best); }
            (None, _)          => { out.push(best); }
        }
    }
    out
});

lifted_num_arr_stage!(DiffWindow, |xs| {
    let mut out: Vec<Option<f64>> = Vec::with_capacity(xs.len());
    for i in 0..xs.len() {
        out.push(match (i.checked_sub(1).and_then(|j| xs[j]), xs[i]) {
            (Some(p), Some(c)) => Some(c - p),
            _ => None,
        });
    }
    out
});

lifted_num_arr_stage!(PctChange, |xs| {
    let mut out: Vec<Option<f64>> = Vec::with_capacity(xs.len());
    for i in 0..xs.len() {
        out.push(match (i.checked_sub(1).and_then(|j| xs[j]), xs[i]) {
            (Some(p), Some(c)) if p != 0.0 => Some((c - p) / p),
            _ => None,
        });
    }
    out
});

// ── Numeric-array windowed Stages w/ usize arg (rolling, lag, lead) ──
//
// Bodies migrated from `eval/func_arrays.rs::{rolling_*, lag, lead, zscore}`.
// Single source of truth here.  Helpers `to_floats` / `floats_to_val`
// stay in `eval/func_arrays.rs` (still used by other not-yet-lifted
// funcs there) — re-imported below.

macro_rules! lifted_sized_num_stage {
    ($name:ident, $body:expr) => {
        pub struct $name { pub n: usize }
        impl $name { pub fn new(n: usize) -> Self { Self { n } } }
        impl Stage for $name {
            fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
                use crate::eval::func_arrays::{to_floats, floats_to_val};
                if self.n == 0 { return StageOutput::Filtered; }
                let xs = match to_floats(x) { Ok(v) => v, Err(_) => return StageOutput::Filtered };
                let f: fn(&[Option<f64>], usize) -> Vec<Option<f64>> = $body;
                StageOutput::Pass(Cow::Owned(floats_to_val(f(&xs, self.n))))
            }
        }
    };
}

lifted_sized_num_stage!(RollingSum, |xs, n| {
    let mut out: Vec<Option<f64>> = Vec::with_capacity(xs.len());
    let mut sum: f64 = 0.0;
    for (i, v) in xs.iter().enumerate() {
        if let Some(x) = v { sum += x; }
        if i >= n { if let Some(old) = xs[i - n] { sum -= old; } }
        if i + 1 >= n { out.push(Some(sum)); } else { out.push(None); }
    }
    out
});

lifted_sized_num_stage!(RollingAvg, |xs, n| {
    let mut out: Vec<Option<f64>> = Vec::with_capacity(xs.len());
    let mut sum: f64 = 0.0;
    let mut count: usize = 0;
    for (i, v) in xs.iter().enumerate() {
        if let Some(x) = v { sum += x; count += 1; }
        if i >= n { if let Some(old) = xs[i - n] { sum -= old; count -= 1; } }
        if i + 1 >= n && count > 0 {
            out.push(Some(sum / count as f64));
        } else {
            out.push(None);
        }
    }
    out
});

lifted_sized_num_stage!(RollingMin, |xs, n| {
    let mut out: Vec<Option<f64>> = Vec::with_capacity(xs.len());
    for i in 0..xs.len() {
        if i + 1 < n { out.push(None); continue; }
        let lo = i + 1 - n;
        let m = xs[lo..=i].iter().filter_map(|v| *v)
            .fold(f64::INFINITY, |a, b| a.min(b));
        out.push(if m.is_finite() { Some(m) } else { None });
    }
    out
});

lifted_sized_num_stage!(RollingMax, |xs, n| {
    let mut out: Vec<Option<f64>> = Vec::with_capacity(xs.len());
    for i in 0..xs.len() {
        if i + 1 < n { out.push(None); continue; }
        let lo = i + 1 - n;
        let m = xs[lo..=i].iter().filter_map(|v| *v)
            .fold(f64::NEG_INFINITY, |a, b| a.max(b));
        out.push(if m.is_finite() { Some(m) } else { None });
    }
    out
});

// Lag/Lead reuse the same macro shape but n=0 is valid (just identity)
// so they need a tiny variant.

pub struct Lag { pub n: usize }
impl Lag { pub fn new(n: usize) -> Self { Self { n } } }
impl Stage for Lag {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        use crate::eval::func_arrays::{to_floats, floats_to_val};
        let xs = match to_floats(x) { Ok(v) => v, Err(_) => return StageOutput::Filtered };
        let n = self.n;
        let mut out: Vec<Option<f64>> = Vec::with_capacity(xs.len());
        for i in 0..xs.len() {
            out.push(if i >= n { xs[i - n] } else { None });
        }
        StageOutput::Pass(Cow::Owned(floats_to_val(out)))
    }
}

pub struct Lead { pub n: usize }
impl Lead { pub fn new(n: usize) -> Self { Self { n } } }
impl Stage for Lead {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        use crate::eval::func_arrays::{to_floats, floats_to_val};
        let xs = match to_floats(x) { Ok(v) => v, Err(_) => return StageOutput::Filtered };
        let n = self.n;
        let mut out: Vec<Option<f64>> = Vec::with_capacity(xs.len());
        for i in 0..xs.len() {
            let j = i + n;
            out.push(if j < xs.len() { xs[j] } else { None });
        }
        StageOutput::Pass(Cow::Owned(floats_to_val(out)))
    }
}

/// `.zscore()` — Arr<num> → Arr<num> standardised.  Zero-arg.
pub struct ZScore;
impl ZScore { pub fn new() -> Self { Self } }
impl Default for ZScore { fn default() -> Self { Self } }
impl Stage for ZScore {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        use crate::eval::func_arrays::{to_floats, floats_to_val};
        let xs = match to_floats(x) { Ok(v) => v, Err(_) => return StageOutput::Filtered };
        let nums: Vec<f64> = xs.iter().filter_map(|v| *v).collect();
        if nums.is_empty() {
            return StageOutput::Pass(Cow::Owned(floats_to_val(vec![None; xs.len()])));
        }
        let mean = nums.iter().sum::<f64>() / nums.len() as f64;
        let var  = nums.iter().map(|y| (y - mean).powi(2)).sum::<f64>() / nums.len() as f64;
        let sd   = var.sqrt();
        let mut out: Vec<Option<f64>> = Vec::with_capacity(xs.len());
        for v in xs.iter() {
            out.push(match v {
                Some(y) if sd > 0.0 => Some((y - mean) / sd),
                Some(_)             => Some(0.0),
                None                => None,
            });
        }
        StageOutput::Pass(Cow::Owned(floats_to_val(out)))
    }
}

// ── Array literal-arg / aggregate Stages — body migrations ────────

/// `.enumerate()` (zero-arg) — Arr → Arr<{index, value}>.
pub struct EnumerateZ;
impl EnumerateZ { pub fn new() -> Self { Self } }
impl Default for EnumerateZ { fn default() -> Self { Self } }
impl Stage for EnumerateZ {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        use crate::eval::util::obj2;
        let items_cow = match x.as_vals() { Some(c) => c, None => return StageOutput::Filtered };
        let out: Vec<Val> = items_cow.iter().enumerate()
            .map(|(i, v)| obj2("index", Val::Int(i as i64), "value", v.clone()))
            .collect();
        StageOutput::Pass(Cow::Owned(Val::arr(out)))
    }
}

/// `.join(sep)` — Arr<Val> → Str.
pub struct Join { pub sep: std::sync::Arc<str> }
impl Join { pub fn new(sep: std::sync::Arc<str>) -> Self { Self { sep } } }
impl Stage for Join {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        use crate::eval::util::val_to_string;
        use std::fmt::Write as _;
        let items_cow = match x.as_vals() { Some(c) => c, None => return StageOutput::Filtered };
        let items: &[Val] = items_cow.as_ref();
        if items.is_empty() {
            return StageOutput::Pass(Cow::Owned(Val::Str(std::sync::Arc::from(""))));
        }
        let sep = self.sep.as_ref();
        if items.iter().all(|v| matches!(v, Val::Str(_))) {
            let total_len: usize = items.iter()
                .map(|v| if let Val::Str(s) = v { s.len() } else { 0 })
                .sum::<usize>()
                + sep.len() * (items.len() - 1);
            let mut out = String::with_capacity(total_len);
            let mut first = true;
            for v in items {
                if !first { out.push_str(sep); }
                first = false;
                if let Val::Str(s) = v { out.push_str(s); }
            }
            return StageOutput::Pass(Cow::Owned(Val::Str(std::sync::Arc::from(out))));
        }
        let est_cap = items.len() * 8 + sep.len() * items.len();
        let mut out = String::with_capacity(est_cap);
        let mut first = true;
        for v in items {
            if !first { out.push_str(sep); }
            first = false;
            match v {
                Val::Str(s)   => out.push_str(s),
                Val::Int(n)   => { let _ = write!(out, "{}", n); }
                Val::Float(f) => { let _ = write!(out, "{}", f); }
                Val::Bool(b)  => out.push_str(if *b { "true" } else { "false" }),
                Val::Null     => out.push_str("null"),
                other         => out.push_str(&val_to_string(other)),
            }
        }
        StageOutput::Pass(Cow::Owned(Val::Str(std::sync::Arc::from(out))))
    }
}

/// `.index_of_value(target)` — first index of literal target, else Null.
pub struct IndexOfValue { pub target: Val }
impl IndexOfValue { pub fn new(target: Val) -> Self { Self { target } } }
impl Stage for IndexOfValue {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        let items_cow = match x.as_vals() { Some(c) => c, None => return StageOutput::Filtered };
        for (i, item) in items_cow.iter().enumerate() {
            if crate::eval::util::vals_eq(item, &self.target) {
                return StageOutput::Pass(Cow::Owned(Val::Int(i as i64)));
            }
        }
        StageOutput::Pass(Cow::Owned(Val::Null))
    }
}

/// `.indices_of(target)` — all indices of literal target.
pub struct IndicesOf { pub target: Val }
impl IndicesOf { pub fn new(target: Val) -> Self { Self { target } } }
impl Stage for IndicesOf {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        let items_cow = match x.as_vals() { Some(c) => c, None => return StageOutput::Filtered };
        let out: Vec<i64> = items_cow.iter().enumerate()
            .filter(|(_, v)| crate::eval::util::vals_eq(v, &self.target))
            .map(|(i, _)| i as i64)
            .collect();
        StageOutput::Pass(Cow::Owned(Val::int_vec(out)))
    }
}

/// `.explode(field)` — Arr<Obj> → Arr<Obj> with array-valued `field`
/// expanded one row per element.  Body migrated from
/// `func_aggregates::explode`.
pub struct Explode { pub field: std::sync::Arc<str> }
impl Explode { pub fn new(field: std::sync::Arc<str>) -> Self { Self { field } } }
impl Stage for Explode {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        let items_cow = match x.as_vals() { Some(c) => c, None => return StageOutput::Filtered };
        let items: &[Val] = items_cow.as_ref();
        let field = self.field.as_ref();
        let mut out = Vec::with_capacity(items.len());
        for item in items {
            match item {
                Val::Obj(m) => {
                    let sub = m.get(field).cloned();
                    match sub.as_ref().map(|v| v.is_array()).unwrap_or(false) {
                        true => {
                            let elts = sub.unwrap().into_vec().unwrap();
                            for e in elts {
                                let mut row = (**m).clone();
                                row.insert(self.field.clone(), e);
                                out.push(Val::obj(row));
                            }
                        }
                        false => out.push(item.clone()),
                    }
                }
                other => out.push(other.clone()),
            }
        }
        StageOutput::Pass(Cow::Owned(Val::arr(out)))
    }
}

/// `.implode(field)` — inverse of explode: groups rows by all-but-`field`,
/// concatenating `field` values into an array.  Body migrated from
/// `func_aggregates::implode`.
pub struct Implode { pub field: std::sync::Arc<str> }
impl Implode { pub fn new(field: std::sync::Arc<str>) -> Self { Self { field } } }
impl Stage for Implode {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        use crate::eval::util::val_to_key;
        let items_cow = match x.as_vals() { Some(c) => c, None => return StageOutput::Filtered };
        let items: &[Val] = items_cow.as_ref();
        let field = self.field.as_ref();
        let mut groups: indexmap::IndexMap<std::sync::Arc<str>,
            (indexmap::IndexMap<std::sync::Arc<str>, Val>, Vec<Val>)> = indexmap::IndexMap::new();
        for item in items {
            let m = match item {
                Val::Obj(m) => m,
                _ => return StageOutput::Filtered,
            };
            let mut rest = (**m).clone();
            let val = rest.shift_remove(field).unwrap_or(Val::Null);
            let key_src: indexmap::IndexMap<std::sync::Arc<str>, Val> = rest.clone();
            let key = std::sync::Arc::<str>::from(val_to_key(&Val::obj(key_src)));
            groups.entry(key).or_insert_with(|| (rest, Vec::new())).1.push(val);
        }
        let mut out = Vec::with_capacity(groups.len());
        for (_, (mut rest, vals)) in groups {
            rest.insert(self.field.clone(), Val::arr(vals));
            out.push(Val::obj(rest));
        }
        StageOutput::Pass(Cow::Owned(Val::arr(out)))
    }
}

// ── More eval/func_* migrations ──────────────────────────────────

/// `.collect()` — Val → Arr.  Identity for arrays, [] for Null,
/// [val] for scalars.  Body migrated from `func_search::collect`.
pub struct CollectVal;
impl CollectVal { pub fn new() -> Self { Self } }
impl Default for CollectVal { fn default() -> Self { Self } }
impl Stage for CollectVal {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        match x {
            Val::Arr(_) | Val::IntVec(_) | Val::FloatVec(_) | Val::StrVec(_) | Val::StrSliceVec(_) => {
                StageOutput::Pass(Cow::Borrowed(x))
            }
            Val::Null => StageOutput::Pass(Cow::Owned(Val::arr(Vec::new()))),
            other     => StageOutput::Pass(Cow::Owned(Val::arr(vec![other.clone()]))),
        }
    }
}

/// `.remove(target)` — Arr → Arr without elements `vals_eq` to literal.
pub struct RemoveVal { pub target: Val }
impl RemoveVal { pub fn new(target: Val) -> Self { Self { target } } }
impl Stage for RemoveVal {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        use crate::eval::util::val_to_key;
        let items_cow = match x.as_vals() { Some(c) => c, None => return StageOutput::Filtered };
        let key = val_to_key(&self.target);
        let out: Vec<Val> = items_cow.iter()
            .filter(|v| val_to_key(v) != key)
            .cloned()
            .collect();
        StageOutput::Pass(Cow::Owned(Val::arr(out)))
    }
}

/// `.set_path(path, value)` — set leaf at path, COW spine clone.
/// Body via `func_paths::set_path_impl`.
pub struct SetPath { pub path: std::sync::Arc<str>, pub value: Val }
impl SetPath {
    pub fn new(path: std::sync::Arc<str>, value: Val) -> Self { Self { path, value } }
}
impl Stage for SetPath {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        let segs = crate::eval::func_paths::parse_path_segs(self.path.as_ref());
        let v = crate::eval::func_paths::set_path_impl(x.clone(), &segs, self.value.clone());
        StageOutput::Pass(Cow::Owned(v))
    }
}

/// `.del_paths(p1, p2, ...)` — multi-path delete (var-arg literal paths).
pub struct DelPaths { pub paths: Vec<std::sync::Arc<str>> }
impl DelPaths {
    pub fn new(paths: Vec<std::sync::Arc<str>>) -> Self { Self { paths } }
}
impl Stage for DelPaths {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        let mut v = x.clone();
        for p in &self.paths {
            let segs = crate::eval::func_paths::parse_path_segs(p.as_ref());
            v = crate::eval::func_paths::del_path_impl(v, &segs);
        }
        StageOutput::Pass(Cow::Owned(v))
    }
}

/// `.omit(k1, k2, ...)` — Obj → Obj minus listed keys.
/// Body migrated from `func_objects::omit`.
pub struct OmitKeys { pub keys: Vec<std::sync::Arc<str>> }
impl OmitKeys {
    pub fn new(keys: Vec<std::sync::Arc<str>>) -> Self { Self { keys } }
}
impl Stage for OmitKeys {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        let m = match x.as_object() { Some(m) => m, None => return StageOutput::Filtered };
        let mut out = m.clone();
        for k in &self.keys {
            out.shift_remove(k.as_ref());
        }
        StageOutput::Pass(Cow::Owned(Val::obj(out)))
    }
}

// ── Scalar Val→Str transforms (eval/builtins.rs migrations) ───────

/// `.type()` — Val → Str (type name).
pub struct TypeName;
impl TypeName { pub fn new() -> Self { Self } }
impl Default for TypeName { fn default() -> Self { Self } }
impl Stage for TypeName {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        StageOutput::Pass(Cow::Owned(Val::Str(std::sync::Arc::from(x.type_name()))))
    }
}

/// `.to_string()` — Val → Str (display form).
pub struct ToString;
impl ToString { pub fn new() -> Self { Self } }
impl Default for ToString { fn default() -> Self { Self } }
impl Stage for ToString {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        let s = crate::eval::util::val_to_string(x);
        StageOutput::Pass(Cow::Owned(Val::Str(std::sync::Arc::from(s))))
    }
}

/// `.to_json()` — Val → Str (JSON encoding).  Inline fast paths for
/// primitives (avoid serde_json::Value detour) — match prior
/// `b_to_json` behaviour.
pub struct ToJson;
impl ToJson { pub fn new() -> Self { Self } }
impl Default for ToJson { fn default() -> Self { Self } }
impl Stage for ToJson {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        let out = match x {
            Val::Int(n)  => n.to_string(),
            Val::Float(f) => {
                if f.is_finite() {
                    let v = serde_json::Value::from(*f);
                    serde_json::to_string(&v).unwrap_or_default()
                } else {
                    "null".to_string()
                }
            }
            Val::Bool(b) => (if *b { "true" } else { "false" }).to_string(),
            Val::Null    => "null".to_string(),
            Val::Str(s)  => {
                let v = serde_json::Value::String(s.to_string());
                serde_json::to_string(&v).unwrap_or_default()
            }
            other => {
                let sv: serde_json::Value = other.clone().into();
                serde_json::to_string(&sv).unwrap_or_default()
            }
        };
        StageOutput::Pass(Cow::Owned(Val::Str(std::sync::Arc::from(out))))
    }
}

// ── Path Stages — body via func_paths::*_impl helpers ─────────────

/// `.del_path(path)` — remove value at dotted path.  Body via
/// `func_paths::del_path_impl`.
pub struct DelPath { pub path: std::sync::Arc<str> }
impl DelPath { pub fn new(path: std::sync::Arc<str>) -> Self { Self { path } } }
impl Stage for DelPath {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        let segs = crate::eval::func_paths::parse_path_segs(self.path.as_ref());
        let v = crate::eval::func_paths::del_path_impl(x.clone(), &segs);
        StageOutput::Pass(Cow::Owned(v))
    }
}

/// `.flatten_keys(sep)` — Obj → flat-Obj with `sep`-joined keys.
pub struct FlattenKeys { pub sep: std::sync::Arc<str> }
impl FlattenKeys {
    pub fn new(sep: std::sync::Arc<str>) -> Self { Self { sep } }
}
impl Stage for FlattenKeys {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        let mut out: indexmap::IndexMap<std::sync::Arc<str>, Val> = indexmap::IndexMap::new();
        crate::eval::func_paths::flatten_keys_impl("", x, self.sep.as_ref(), &mut out);
        StageOutput::Pass(Cow::Owned(Val::obj(out)))
    }
}

/// `.unflatten_keys(sep)` — flat-Obj → nested Obj.
pub struct UnflattenKeys { pub sep: std::sync::Arc<str> }
impl UnflattenKeys {
    pub fn new(sep: std::sync::Arc<str>) -> Self { Self { sep } }
}
impl Stage for UnflattenKeys {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        match x {
            Val::Obj(m) => {
                let v = crate::eval::func_paths::unflatten_keys_impl(m, self.sep.as_ref());
                StageOutput::Pass(Cow::Owned(v))
            }
            _ => StageOutput::Filtered,
        }
    }
}

/// `.schema()` — Val → schema-Obj describing types/required fields/array shape.
pub struct Schema;
impl Schema { pub fn new() -> Self { Self } }
impl Default for Schema { fn default() -> Self { Self } }
impl Stage for Schema {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        StageOutput::Pass(Cow::Owned(crate::eval::func_objects::schema_of(x)))
    }
}

// ── Substring set predicates ───────────────────────────────────────

/// `.contains_any([needles])` — true if any needle appears in the
/// receiver.
pub struct ContainsAny { pub needles: Vec<std::sync::Arc<str>> }
impl ContainsAny {
    pub fn new(needles: Vec<std::sync::Arc<str>>) -> Self { Self { needles } }
}
impl Stage for ContainsAny {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        if let Val::Str(s) = x {
            let any = self.needles.iter().any(|n| s.contains(n.as_ref()));
            return StageOutput::Pass(Cow::Owned(Val::Bool(any)));
        }
        StageOutput::Filtered
    }
}

/// `.contains_all([needles])` — true if every needle appears.
pub struct ContainsAll { pub needles: Vec<std::sync::Arc<str>> }
impl ContainsAll {
    pub fn new(needles: Vec<std::sync::Arc<str>>) -> Self { Self { needles } }
}
impl Stage for ContainsAll {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        if let Val::Str(s) = x {
            let all = self.needles.iter().all(|n| s.contains(n.as_ref()));
            return StageOutput::Pass(Cow::Owned(Val::Bool(all)));
        }
        StageOutput::Filtered
    }
}

// ── FromBase64 (with explicit error semantics) ─────────────────────

/// `.from_base64()` — decode; returns Err on invalid input (cannot
/// use the generic Stage filter→err shim because Stage::Filtered
/// doesn't carry an error message).
pub fn from_base64_eager(s: &str) -> Result<String, String> {
    base64_decode(s).map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
}

// ── Regex Stages ───────────────────────────────────────────────────
//
// Compile cache (thread-local).  Each unique pattern compiles once
// per thread; subsequent uses reuse the same `Arc<Regex>`.
thread_local! {
    static REGEX_CACHE: std::cell::RefCell<std::collections::HashMap<String, std::sync::Arc<regex::Regex>>>
        = std::cell::RefCell::new(std::collections::HashMap::with_capacity(32));
}

pub(crate) fn compile_regex(pat: &str) -> Result<std::sync::Arc<regex::Regex>, String> {
    REGEX_CACHE.with(|cell| {
        let mut m = cell.borrow_mut();
        if let Some(r) = m.get(pat) { return Ok(std::sync::Arc::clone(r)); }
        let re = regex::Regex::new(pat).map_err(|e| format!("regex compile: {}", e))?;
        let arc = std::sync::Arc::new(re);
        m.insert(pat.to_string(), std::sync::Arc::clone(&arc));
        Ok(arc)
    })
}

/// `.match(pat)` — Bool, regex match anywhere.
pub struct ReMatch { pub pat: std::sync::Arc<str> }
impl ReMatch { pub fn new(pat: std::sync::Arc<str>) -> Self { Self { pat } } }
impl Stage for ReMatch {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        if let Val::Str(s) = x {
            let re = match compile_regex(self.pat.as_ref()) {
                Ok(r) => r,
                Err(_) => return StageOutput::Filtered,
            };
            return StageOutput::Pass(Cow::Owned(Val::Bool(re.is_match(s.as_ref()))));
        }
        StageOutput::Filtered
    }
}

/// `.match_first(pat)` — Str of first match or Null.
pub struct ReMatchFirst { pub pat: std::sync::Arc<str> }
impl ReMatchFirst { pub fn new(pat: std::sync::Arc<str>) -> Self { Self { pat } } }
impl Stage for ReMatchFirst {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        if let Val::Str(s) = x {
            let re = match compile_regex(self.pat.as_ref()) {
                Ok(r) => r,
                Err(_) => return StageOutput::Filtered,
            };
            let v = re.find(s.as_ref())
                .map(|m| Val::Str(std::sync::Arc::from(m.as_str())))
                .unwrap_or(Val::Null);
            return StageOutput::Pass(Cow::Owned(v));
        }
        StageOutput::Filtered
    }
}

/// `.match_all(pat)` — Arr<Str> of all matches.
pub struct ReMatchAll { pub pat: std::sync::Arc<str> }
impl ReMatchAll { pub fn new(pat: std::sync::Arc<str>) -> Self { Self { pat } } }
impl Stage for ReMatchAll {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        if let Val::Str(s) = x {
            let re = match compile_regex(self.pat.as_ref()) {
                Ok(r) => r,
                Err(_) => return StageOutput::Filtered,
            };
            let out: Vec<std::sync::Arc<str>> = re.find_iter(s.as_ref())
                .map(|m| std::sync::Arc::<str>::from(m.as_str())).collect();
            return StageOutput::Pass(Cow::Owned(Val::str_vec(out)));
        }
        StageOutput::Filtered
    }
}

/// `.captures(pat)` — Arr of capture groups (group 0 first); Null on no match.
pub struct ReCaptures { pub pat: std::sync::Arc<str> }
impl ReCaptures { pub fn new(pat: std::sync::Arc<str>) -> Self { Self { pat } } }
impl Stage for ReCaptures {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        if let Val::Str(s) = x {
            let re = match compile_regex(self.pat.as_ref()) {
                Ok(r) => r,
                Err(_) => return StageOutput::Filtered,
            };
            let v = match re.captures(s.as_ref()) {
                Some(c) => {
                    let mut out: Vec<Val> = Vec::with_capacity(c.len());
                    for i in 0..c.len() {
                        out.push(c.get(i)
                            .map(|m| Val::Str(std::sync::Arc::from(m.as_str())))
                            .unwrap_or(Val::Null));
                    }
                    Val::arr(out)
                }
                None => Val::Null,
            };
            return StageOutput::Pass(Cow::Owned(v));
        }
        StageOutput::Filtered
    }
}

/// `.captures_all(pat)` — Arr<Arr> of capture groups for every match.
pub struct ReCapturesAll { pub pat: std::sync::Arc<str> }
impl ReCapturesAll { pub fn new(pat: std::sync::Arc<str>) -> Self { Self { pat } } }
impl Stage for ReCapturesAll {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        if let Val::Str(s) = x {
            let re = match compile_regex(self.pat.as_ref()) {
                Ok(r) => r,
                Err(_) => return StageOutput::Filtered,
            };
            let mut all: Vec<Val> = Vec::new();
            for c in re.captures_iter(s.as_ref()) {
                let mut row: Vec<Val> = Vec::with_capacity(c.len());
                for i in 0..c.len() {
                    row.push(c.get(i)
                        .map(|m| Val::Str(std::sync::Arc::from(m.as_str())))
                        .unwrap_or(Val::Null));
                }
                all.push(Val::arr(row));
            }
            return StageOutput::Pass(Cow::Owned(Val::arr(all)));
        }
        StageOutput::Filtered
    }
}

/// `.replace_re(pat, with)` — single regex replacement.
pub struct ReReplace { pub pat: std::sync::Arc<str>, pub with: std::sync::Arc<str> }
impl ReReplace {
    pub fn new(pat: std::sync::Arc<str>, with: std::sync::Arc<str>) -> Self { Self { pat, with } }
}
impl Stage for ReReplace {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        if let Val::Str(s) = x {
            let re = match compile_regex(self.pat.as_ref()) {
                Ok(r) => r,
                Err(_) => return StageOutput::Filtered,
            };
            let out = re.replace(s.as_ref(), self.with.as_ref());
            return StageOutput::Pass(Cow::Owned(
                Val::Str(std::sync::Arc::from(out.as_ref()))));
        }
        StageOutput::Filtered
    }
}

/// `.replace_all_re(pat, with)` — all regex replacements.
pub struct ReReplaceAll { pub pat: std::sync::Arc<str>, pub with: std::sync::Arc<str> }
impl ReReplaceAll {
    pub fn new(pat: std::sync::Arc<str>, with: std::sync::Arc<str>) -> Self { Self { pat, with } }
}
impl Stage for ReReplaceAll {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        if let Val::Str(s) = x {
            let re = match compile_regex(self.pat.as_ref()) {
                Ok(r) => r,
                Err(_) => return StageOutput::Filtered,
            };
            let out = re.replace_all(s.as_ref(), self.with.as_ref());
            return StageOutput::Pass(Cow::Owned(
                Val::Str(std::sync::Arc::from(out.as_ref()))));
        }
        StageOutput::Filtered
    }
}

/// `.split_re(pat)` — regex split.
pub struct ReSplit { pub pat: std::sync::Arc<str> }
impl ReSplit { pub fn new(pat: std::sync::Arc<str>) -> Self { Self { pat } } }
impl Stage for ReSplit {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        if let Val::Str(s) = x {
            let re = match compile_regex(self.pat.as_ref()) {
                Ok(r) => r,
                Err(_) => return StageOutput::Filtered,
            };
            let out: Vec<std::sync::Arc<str>> = re.split(s.as_ref())
                .map(std::sync::Arc::<str>::from).collect();
            return StageOutput::Pass(Cow::Owned(Val::str_vec(out)));
        }
        StageOutput::Filtered
    }
}

/// `.last_index_of(needle)` — char-position of last match; -1 on miss.
pub struct LastIndexOf { pub needle: std::sync::Arc<str> }
impl LastIndexOf {
    pub fn new(needle: std::sync::Arc<str>) -> Self { Self { needle } }
}
impl Stage for LastIndexOf {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        if let Val::Str(s) = x {
            let r = match s.rfind(self.needle.as_ref()) {
                Some(i) => Val::Int(s[..i].chars().count() as i64),
                None    => Val::Int(-1),
            };
            return StageOutput::Pass(Cow::Owned(r));
        }
        StageOutput::Filtered
    }
}

/// Closure-based `.filter(pred)` — for the borrow runner where the
/// predicate is built from a kernel at lowering time (FieldCmpLit etc.
/// → owned literal compare).  composed.rs's `GenericFilter` uses VM
/// dispatch and only impls owned `Stage`; this `Filter` is closure-
/// based and impls both `unified::Stage<R>` (any substrate) and the
/// owned `Stage` (R = `&Val` adapter).
pub struct Filter<R, F: Fn(&R) -> bool> {
    pub pred: F,
    _marker: std::marker::PhantomData<fn(R)>,
}
impl<R, F: Fn(&R) -> bool> Filter<R, F> {
    pub fn new(pred: F) -> Self {
        Self { pred, _marker: std::marker::PhantomData }
    }
}
/// Monoidal composition. `Composed { a, b }.apply(x) = b.apply(a.apply(x))`
/// with proper handling of Filtered / Many / Done propagation.
///
/// Trait bounds live on the impl blocks rather than the struct so the
/// SAME `Composed<A, B>` can chain stages under either the owned
/// `Stage` trait or the substrate-generic `unified::Stage<R>` trait.
pub struct Composed<A, B> {
    pub a: A,
    pub b: B,
}

impl<A, B> Composed<A, B> {
    pub fn new(a: A, b: B) -> Self { Self { a, b } }
}

impl<A: Stage, B: Stage> Stage for Composed<A, B> {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        match self.a.apply(x) {
            StageOutput::Pass(Cow::Borrowed(v)) => self.b.apply(v),
            StageOutput::Pass(Cow::Owned(v)) => {
                // `b` may borrow from v — but v dies at scope end. Force
                // owned result so lifetime is independent of `v`. The
                // computed-then-borrow case is rare; common case is
                // borrow-borrow which is zero-cost above.
                let owned = match self.b.apply(&v) {
                    StageOutput::Pass(c) => Cow::Owned(c.into_owned()),
                    StageOutput::Filtered => return StageOutput::Filtered,
                    StageOutput::Many(items) => {
                        let mut out: SmallVec<[Cow<'a, Val>; 4]> = SmallVec::new();
                        for it in items {
                            out.push(Cow::Owned(it.into_owned()));
                        }
                        return StageOutput::Many(out);
                    }
                    StageOutput::Done => return StageOutput::Done,
                };
                StageOutput::Pass(owned)
            }
            StageOutput::Filtered => StageOutput::Filtered,
            StageOutput::Many(items) => {
                let mut out: SmallVec<[Cow<'a, Val>; 4]> = SmallVec::new();
                for it in items {
                    match it {
                        Cow::Borrowed(v) => match self.b.apply(v) {
                            StageOutput::Pass(c) => out.push(c),
                            StageOutput::Filtered => continue,
                            StageOutput::Many(more) => out.extend(more),
                            StageOutput::Done => {
                                return if out.is_empty() {
                                    StageOutput::Done
                                } else {
                                    StageOutput::Many(out)
                                };
                            }
                        },
                        Cow::Owned(v) => {
                            // `v` is owned and dies at end of this arm; any
                            // borrow returned by `b.apply(&v)` must be
                            // promoted to owned so `out` outlives `v`.
                            match self.b.apply(&v) {
                                StageOutput::Pass(c) => out.push(Cow::Owned(c.into_owned())),
                                StageOutput::Filtered => continue,
                                StageOutput::Many(more) => {
                                    for m in more {
                                        out.push(Cow::Owned(m.into_owned()));
                                    }
                                }
                                StageOutput::Done => {
                                    return if out.is_empty() {
                                        StageOutput::Done
                                    } else {
                                        StageOutput::Many(out)
                                    };
                                }
                            }
                        }
                    }
                }
                if out.is_empty() {
                    StageOutput::Filtered
                } else if out.len() == 1 {
                    StageOutput::Pass(out.into_iter().next().unwrap())
                } else {
                    StageOutput::Many(out)
                }
            }
            StageOutput::Done => StageOutput::Done,
        }
    }
}
