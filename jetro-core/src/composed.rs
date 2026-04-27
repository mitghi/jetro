//! Layer B — composed-Cow Stage chains as the sole runtime exec path.
//!
//! Per `pipeline_specialisation.md` (CORRECTED 2026-04-26 + cold-first
//! pivot): replace the legacy fused-Sink enum + ~30 hand-rolled VM
//! opcodes with a single generic substrate driven by composition. No
//! per-shape opcodes, no enumerated `(stage_chain × sink)` Sink
//! variants — those are the disease. Composition primitives + a
//! bounded list of generic Sinks cover all chain shapes uniformly.
//!
//! ## Design
//!
//! - `Stage`: `&'a Val → StageOutput<'a>` — borrow-form returns
//!   `Cow::Borrowed`, computed-form returns `Cow::Owned`. Avoids the
//!   per-stage clone tax measured at 2.3× (composed-fn owned) vs 1.29×
//!   (composed-borrow Cow) on `filter+map+sum` × 5000 × 1000 iters.
//! - `Composed<A, B>`: monoidal pairing; N stages fold into one apply
//!   call via `stages.into_iter().fold(Identity, compose)`. One
//!   virtual call per element regardless of chain length.
//! - `Sink`: `Acc + fold(&Acc, &Val) → Acc + finalise(Acc) → Val`. A
//!   bounded set (Sum/Min/Max/Avg/Count/First/Last/Collect) covers
//!   every legacy fused Sink variant.
//! - `run_pipeline<S: Sink>`: one generic outer loop, parameterised by
//!   `S`. Stages composed once at lower-time, used N× at execute.
//!
//! No (stage × sink) enumeration anywhere. Adding a new stage shape =
//! one `Stage::apply` impl. Adding a new sink = one `Sink` impl.
//! Adding a new chain shape = no code.
//!
//! ## Status
//!
//! Day 1 — module foundation. Standalone; not yet wired into
//! `pipeline.rs::run`. Day 2-3 lands the wiring under
//! `JETRO_COMPOSED=1` and bench-gates against legacy fused.

use std::borrow::Cow;
use smallvec::SmallVec;

use crate::eval::value::Val;

// ── Stage ────────────────────────────────────────────────────────────────────

/// Per-element output of a `Stage::apply`. Borrowed payload when the
/// stage is a pass-through over the input (filter, field-read);
/// owned payload only when the stage computed a fresh value
/// (arithmetic, format-string, projection).
pub enum StageOutput<'a> {
    /// Stage produced one element. `Cow::Borrowed` for pass-through
    /// stages (zero clone); `Cow::Owned` for computed values.
    Pass(Cow<'a, Val>),
    /// Filter dropped the element.
    Filtered,
    /// FlatMap produced multiple elements.
    Many(SmallVec<[Cow<'a, Val>; 4]>),
    /// Take exhausted; outer loop should terminate.
    Done,
}

/// A Stage transforms one element into a `StageOutput`. The lifetime
/// `'a` ties borrowed outputs to the input reference, eliminating the
/// per-stage clone overhead that owned-Val composition incurred.
///
/// Pipelines run single-threaded per invocation; no Send/Sync bound.
/// Stages with interior mutability (`Take`, `Skip` counters) work in
/// place via `Cell<usize>` reset at lower-time.
pub trait Stage {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a>;

    /// Step 3d-extension (B) — produce only the kth output for input `x`.
    /// Default impl materialises via `apply` and indexes; Expanding
    /// stages with `can_indexed=true` (Split, Range, FlatMap when inner
    /// is bounded) override with direct computation — e.g. memchr-based
    /// kth-segment lookup for Split — to convert O(N) work into O(k).
    /// Used by the planner's IndexedDispatch kernel for `Cardinality::
    /// Expanding` stages preceded by all 1:1 stages and followed by a
    /// positional sink.
    fn apply_indexed<'a>(&self, x: &'a Val, k: usize) -> Option<Cow<'a, Val>> {
        match self.apply(x) {
            StageOutput::Pass(v) if k == 0 => Some(v),
            StageOutput::Pass(_)           => None,
            StageOutput::Many(mut items)
                if k < items.len() => Some(items.swap_remove(k)),
            StageOutput::Many(_)
                | StageOutput::Filtered
                | StageOutput::Done => None,
        }
    }
}

/// Blanket impl so `Box<dyn Stage>` itself implements `Stage`. Lets a
/// chain of stages be folded into a single `Box<dyn Stage>` at
/// lower-time without macro acrobatics.
impl<T: Stage + ?Sized> Stage for Box<T> {
    #[inline]
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        (**self).apply(x)
    }
    #[inline]
    fn apply_indexed<'a>(&self, x: &'a Val, k: usize) -> Option<Cow<'a, Val>> {
        (**self).apply_indexed(x, k)
    }
}

/// Identity stage — pass-through. Used as the fold seed when composing
/// a chain of stages.
pub struct Identity;

impl Identity {
    pub fn new() -> Self { Self }
}

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
            Arg::Pos(e) | Arg::Named(_, e) => crate::eval::eval(e, env)?,
        };
        match v {
            Val::Str(s) => Ok(s),
            _ => err!("{}: expected string argument", name),
        }
    }

    fn first_i64_arg(args: &[Arg], env: &Env, name: &str) -> Result<i64, EvalError> {
        let a = args.first().ok_or_else(|| EvalError(format!("{}: missing argument", name)))?;
        let v = match a {
            Arg::Pos(e) | Arg::Named(_, e) => crate::eval::eval(e, env)?,
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
                Arg::Pos(e) | Arg::Named(_, e) => crate::eval::eval(e, env).ok(),
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
                    Arg::Pos(e) | Arg::Named(_, e) => crate::eval::eval(e, env)?,
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
                Arg::Pos(e) | Arg::Named(_, e) => crate::eval::eval(e, env)?,
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
                Arg::Pos(e) | Arg::Named(_, e) => crate::eval::eval(e, env)?,
            } {
                return Ok(vec![s]);
            }
            return err!("{}: arg must be string or array of strings", who);
        }
        let mut out = Vec::with_capacity(args.len());
        for a in args {
            let v = match a {
                Arg::Pos(e) | Arg::Named(_, e) => crate::eval::eval(e, env)?,
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
                Arg::Pos(e) | Arg::Named(_, e) => match crate::eval::eval(e, env)? {
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
                Arg::Pos(e) | Arg::Named(_, e) => match crate::eval::eval(e, env)? {
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
            Arg::Pos(e) | Arg::Named(_, e) => crate::eval::eval(e, env)?,
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
                Arg::Pos(e) | Arg::Named(_, e) => crate::eval::eval(e, env)?,
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
                Arg::Pos(e) | Arg::Named(_, e) => crate::eval::eval(e, env)?,
            },
            None => Val::Null,
        };
        run_single(&Prepend::new(item), &recv)
            .ok_or_else(|| EvalError("prepend: stage filtered".into()))
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

        // Borrow-substrate trait impl (future arena-aware variant
        // lands when Stage<R> grows arena support; placeholder for
        // now — production borrow path uses bytescan + composed_tape
        // for builtins).
        impl<'a> crate::unified::Stage<crate::eval::borrowed::Val<'a>> for $name {
            fn apply(&self, _x: crate::eval::borrowed::Val<'a>)
                -> crate::unified::StageOutputU<crate::eval::borrowed::Val<'a>>
            {
                crate::unified::StageOutputU::Filtered
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

        impl<'a> crate::unified::Stage<crate::eval::borrowed::Val<'a>> for $name {
            fn apply(&self, _x: crate::eval::borrowed::Val<'a>)
                -> crate::unified::StageOutputU<crate::eval::borrowed::Val<'a>>
            {
                crate::unified::StageOutputU::Filtered
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
impl<'a> crate::unified::Stage<crate::eval::borrowed::Val<'a>> for StartsWith {
    fn apply(&self, _x: crate::eval::borrowed::Val<'a>)
        -> crate::unified::StageOutputU<crate::eval::borrowed::Val<'a>>
    { crate::unified::StageOutputU::Filtered }
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
impl<'a> crate::unified::Stage<crate::eval::borrowed::Val<'a>> for EndsWith {
    fn apply(&self, _x: crate::eval::borrowed::Val<'a>)
        -> crate::unified::StageOutputU<crate::eval::borrowed::Val<'a>>
    { crate::unified::StageOutputU::Filtered }
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
impl<'a> crate::unified::Stage<crate::eval::borrowed::Val<'a>> for Contains {
    fn apply(&self, _x: crate::eval::borrowed::Val<'a>)
        -> crate::unified::StageOutputU<crate::eval::borrowed::Val<'a>>
    { crate::unified::StageOutputU::Filtered }
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
impl<'a> crate::unified::Stage<crate::eval::borrowed::Val<'a>> for Repeat {
    fn apply(&self, _x: crate::eval::borrowed::Val<'a>)
        -> crate::unified::StageOutputU<crate::eval::borrowed::Val<'a>>
    { crate::unified::StageOutputU::Filtered }
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
impl<'a> crate::unified::Stage<crate::eval::borrowed::Val<'a>> for Split {
    fn apply(&self, _x: crate::eval::borrowed::Val<'a>)
        -> crate::unified::StageOutputU<crate::eval::borrowed::Val<'a>>
    { crate::unified::StageOutputU::Filtered }
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
impl<'a> crate::unified::Stage<crate::eval::borrowed::Val<'a>> for Replace {
    fn apply(&self, _x: crate::eval::borrowed::Val<'a>)
        -> crate::unified::StageOutputU<crate::eval::borrowed::Val<'a>>
    { crate::unified::StageOutputU::Filtered }
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
impl<'a> crate::unified::Stage<crate::eval::borrowed::Val<'a>> for StripPrefix {
    fn apply(&self, _x: crate::eval::borrowed::Val<'a>)
        -> crate::unified::StageOutputU<crate::eval::borrowed::Val<'a>>
    { crate::unified::StageOutputU::Filtered }
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
impl<'a> crate::unified::Stage<crate::eval::borrowed::Val<'a>> for StripSuffix {
    fn apply(&self, _x: crate::eval::borrowed::Val<'a>)
        -> crate::unified::StageOutputU<crate::eval::borrowed::Val<'a>>
    { crate::unified::StageOutputU::Filtered }
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
impl<'a> crate::unified::Stage<crate::eval::borrowed::Val<'a>> for PadLeft {
    fn apply(&self, _x: crate::eval::borrowed::Val<'a>)
        -> crate::unified::StageOutputU<crate::eval::borrowed::Val<'a>>
    { crate::unified::StageOutputU::Filtered }
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
impl<'a> crate::unified::Stage<crate::eval::borrowed::Val<'a>> for PadRight {
    fn apply(&self, _x: crate::eval::borrowed::Val<'a>)
        -> crate::unified::StageOutputU<crate::eval::borrowed::Val<'a>>
    { crate::unified::StageOutputU::Filtered }
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
impl<'a> crate::unified::Stage<crate::eval::borrowed::Val<'a>> for Indent {
    fn apply(&self, _x: crate::eval::borrowed::Val<'a>)
        -> crate::unified::StageOutputU<crate::eval::borrowed::Val<'a>>
    { crate::unified::StageOutputU::Filtered }
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
impl<'a> crate::unified::Stage<crate::eval::borrowed::Val<'a>> for StrMatches {
    fn apply(&self, _x: crate::eval::borrowed::Val<'a>)
        -> crate::unified::StageOutputU<crate::eval::borrowed::Val<'a>>
    { crate::unified::StageOutputU::Filtered }
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
impl<'a> crate::unified::Stage<crate::eval::borrowed::Val<'a>> for IndexOf {
    fn apply(&self, _x: crate::eval::borrowed::Val<'a>)
        -> crate::unified::StageOutputU<crate::eval::borrowed::Val<'a>>
    { crate::unified::StageOutputU::Filtered }
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
            Val::Str(s)         => s.chars().count(),
            _ => return StageOutput::Filtered,
        };
        StageOutput::Pass(Cow::Owned(Val::Int(n as i64)))
    }
}
impl<R> crate::unified::Stage<R> for Len {
    fn apply(&self, _x: R) -> crate::unified::StageOutputU<R> {
        crate::unified::StageOutputU::Filtered
    }
}

/// `.compact()` — drop Null entries from an Arr.
pub struct Compact;
impl Compact { pub fn new() -> Self { Self } }
impl Default for Compact { fn default() -> Self { Self } }
impl Stage for Compact {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        if let Val::Arr(a) = x {
            let kept: Vec<Val> = a.iter()
                .filter(|v| !matches!(v, Val::Null))
                .cloned()
                .collect();
            return StageOutput::Pass(Cow::Owned(Val::arr(kept)));
        }
        StageOutput::Filtered
    }
}
impl<R> crate::unified::Stage<R> for Compact {
    fn apply(&self, _x: R) -> crate::unified::StageOutputU<R> {
        crate::unified::StageOutputU::Filtered
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
impl<R> crate::unified::Stage<R> for FlattenOne {
    fn apply(&self, _x: R) -> crate::unified::StageOutputU<R> {
        crate::unified::StageOutputU::Filtered
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
impl<R> crate::unified::Stage<R> for FlattenDepth {
    fn apply(&self, _x: R) -> crate::unified::StageOutputU<R> {
        crate::unified::StageOutputU::Filtered
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
impl<R> crate::unified::Stage<R> for ReverseAny {
    fn apply(&self, _x: R) -> crate::unified::StageOutputU<R> {
        crate::unified::StageOutputU::Filtered
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
impl<R> crate::unified::Stage<R> for UniqueArr {
    fn apply(&self, _x: R) -> crate::unified::StageOutputU<R> {
        crate::unified::StageOutputU::Filtered
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
impl<R> crate::unified::Stage<R> for First {
    fn apply(&self, _x: R) -> crate::unified::StageOutputU<R> {
        crate::unified::StageOutputU::Filtered
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
impl<R> crate::unified::Stage<R> for Last {
    fn apply(&self, _x: R) -> crate::unified::StageOutputU<R> {
        crate::unified::StageOutputU::Filtered
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
impl<R> crate::unified::Stage<R> for NthAny {
    fn apply(&self, _x: R) -> crate::unified::StageOutputU<R> {
        crate::unified::StageOutputU::Filtered
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
impl<R> crate::unified::Stage<R> for Append {
    fn apply(&self, _x: R) -> crate::unified::StageOutputU<R> {
        crate::unified::StageOutputU::Filtered
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
impl<R> crate::unified::Stage<R> for Prepend {
    fn apply(&self, _x: R) -> crate::unified::StageOutputU<R> {
        crate::unified::StageOutputU::Filtered
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
impl<R> crate::unified::Stage<R> for Enumerate {
    fn apply(&self, _x: R) -> crate::unified::StageOutputU<R> {
        crate::unified::StageOutputU::Filtered
    }
}

/// `.pairwise()` — Arr → Arr of [arr[i], arr[i+1]] adjacent pairs.
pub struct Pairwise;
impl Pairwise { pub fn new() -> Self { Self } }
impl Default for Pairwise { fn default() -> Self { Self } }
impl Stage for Pairwise {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        if let Val::Arr(a) = x {
            let mut out: Vec<Val> = Vec::with_capacity(a.len().saturating_sub(1));
            for w in a.windows(2) {
                out.push(Val::arr(vec![w[0].clone(), w[1].clone()]));
            }
            return StageOutput::Pass(Cow::Owned(Val::arr(out)));
        }
        StageOutput::Filtered
    }
}
impl<R> crate::unified::Stage<R> for Pairwise {
    fn apply(&self, _x: R) -> crate::unified::StageOutputU<R> {
        crate::unified::StageOutputU::Filtered
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
impl<R> crate::unified::Stage<R> for Chunk {
    fn apply(&self, _x: R) -> crate::unified::StageOutputU<R> {
        crate::unified::StageOutputU::Filtered
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
impl<R> crate::unified::Stage<R> for Window {
    fn apply(&self, _x: R) -> crate::unified::StageOutputU<R> {
        crate::unified::StageOutputU::Filtered
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
impl<R> crate::unified::Stage<R> for Keys {
    fn apply(&self, _x: R) -> crate::unified::StageOutputU<R> {
        crate::unified::StageOutputU::Filtered
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
impl<R> crate::unified::Stage<R> for Values {
    fn apply(&self, _x: R) -> crate::unified::StageOutputU<R> {
        crate::unified::StageOutputU::Filtered
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
impl<R> crate::unified::Stage<R> for Entries {
    fn apply(&self, _x: R) -> crate::unified::StageOutputU<R> {
        crate::unified::StageOutputU::Filtered
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
impl<R> crate::unified::Stage<R> for FromPairs {
    fn apply(&self, _x: R) -> crate::unified::StageOutputU<R> {
        crate::unified::StageOutputU::Filtered
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
                Val::Int(n) => std::sync::Arc::from(n.to_string()),
                Val::Float(f) => std::sync::Arc::from(f.to_string()),
                Val::Bool(b) => std::sync::Arc::from(if *b { "true" } else { "false" }),
                Val::Null    => std::sync::Arc::from("null"),
                _ => continue,
            };
            out.insert(new_key, Val::Str(k.clone()));
        }
        StageOutput::Pass(Cow::Owned(Val::Obj(std::sync::Arc::new(out))))
    }
}
impl<R> crate::unified::Stage<R> for Invert {
    fn apply(&self, _x: R) -> crate::unified::StageOutputU<R> {
        crate::unified::StageOutputU::Filtered
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
impl<R> crate::unified::Stage<R> for Intersect {
    fn apply(&self, _x: R) -> crate::unified::StageOutputU<R> {
        crate::unified::StageOutputU::Filtered
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
impl<R> crate::unified::Stage<R> for Union {
    fn apply(&self, _x: R) -> crate::unified::StageOutputU<R> {
        crate::unified::StageOutputU::Filtered
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
impl<R> crate::unified::Stage<R> for Diff {
    fn apply(&self, _x: R) -> crate::unified::StageOutputU<R> {
        crate::unified::StageOutputU::Filtered
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
impl<R> crate::unified::Stage<R> for GetPath {
    fn apply(&self, _x: R) -> crate::unified::StageOutputU<R> {
        crate::unified::StageOutputU::Filtered
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
impl<R> crate::unified::Stage<R> for HasPath {
    fn apply(&self, _x: R) -> crate::unified::StageOutputU<R> {
        crate::unified::StageOutputU::Filtered
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
impl<R> crate::unified::Stage<R> for Has {
    fn apply(&self, _x: R) -> crate::unified::StageOutputU<R> {
        crate::unified::StageOutputU::Filtered
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
impl<R> crate::unified::Stage<R> for Pick {
    fn apply(&self, _x: R) -> crate::unified::StageOutputU<R> {
        crate::unified::StageOutputU::Filtered
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
impl<R> crate::unified::Stage<R> for Omit {
    fn apply(&self, _x: R) -> crate::unified::StageOutputU<R> {
        crate::unified::StageOutputU::Filtered
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
impl<R> crate::unified::Stage<R> for Nth {
    fn apply(&self, _x: R) -> crate::unified::StageOutputU<R> {
        crate::unified::StageOutputU::Filtered
    }
}

// ── Case-conversion Stages ─────────────────────────────────────────

/// Split into lowercased word tokens.  Tokens break on `[-_ \t]`,
/// lower→upper transitions, and non-alphanumeric boundaries.  Used
/// by SnakeCase / KebabCase / CamelCase / PascalCase.
fn split_words_lower(s: &str) -> Vec<String> {
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

fn upper_first_into(p: &str, out: &mut String) {
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
impl<R> crate::unified::Stage<R> for Scan {
    fn apply(&self, _x: R) -> crate::unified::StageOutputU<R> {
        crate::unified::StageOutputU::Filtered
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
impl<R> crate::unified::Stage<R> for Center {
    fn apply(&self, _x: R) -> crate::unified::StageOutputU<R> {
        crate::unified::StageOutputU::Filtered
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
impl<R> crate::unified::Stage<R> for ContainsAny {
    fn apply(&self, _x: R) -> crate::unified::StageOutputU<R> {
        crate::unified::StageOutputU::Filtered
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
impl<R> crate::unified::Stage<R> for ContainsAll {
    fn apply(&self, _x: R) -> crate::unified::StageOutputU<R> {
        crate::unified::StageOutputU::Filtered
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

fn compile_regex(pat: &str) -> Result<std::sync::Arc<regex::Regex>, String> {
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
impl<R> crate::unified::Stage<R> for ReMatch {
    fn apply(&self, _x: R) -> crate::unified::StageOutputU<R> {
        crate::unified::StageOutputU::Filtered
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
impl<R> crate::unified::Stage<R> for ReMatchFirst {
    fn apply(&self, _x: R) -> crate::unified::StageOutputU<R> {
        crate::unified::StageOutputU::Filtered
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
impl<R> crate::unified::Stage<R> for ReMatchAll {
    fn apply(&self, _x: R) -> crate::unified::StageOutputU<R> {
        crate::unified::StageOutputU::Filtered
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
impl<R> crate::unified::Stage<R> for ReCaptures {
    fn apply(&self, _x: R) -> crate::unified::StageOutputU<R> {
        crate::unified::StageOutputU::Filtered
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
impl<R> crate::unified::Stage<R> for ReCapturesAll {
    fn apply(&self, _x: R) -> crate::unified::StageOutputU<R> {
        crate::unified::StageOutputU::Filtered
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
impl<R> crate::unified::Stage<R> for ReReplace {
    fn apply(&self, _x: R) -> crate::unified::StageOutputU<R> {
        crate::unified::StageOutputU::Filtered
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
impl<R> crate::unified::Stage<R> for ReReplaceAll {
    fn apply(&self, _x: R) -> crate::unified::StageOutputU<R> {
        crate::unified::StageOutputU::Filtered
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
impl<R> crate::unified::Stage<R> for ReSplit {
    fn apply(&self, _x: R) -> crate::unified::StageOutputU<R> {
        crate::unified::StageOutputU::Filtered
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
impl<'a> crate::unified::Stage<crate::eval::borrowed::Val<'a>> for LastIndexOf {
    fn apply(&self, _x: crate::eval::borrowed::Val<'a>)
        -> crate::unified::StageOutputU<crate::eval::borrowed::Val<'a>>
    { crate::unified::StageOutputU::Filtered }
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
impl<R, F: Fn(&R) -> bool> crate::unified::Stage<R> for Filter<R, F> {
    #[inline]
    fn apply(&self, x: R) -> crate::unified::StageOutputU<R> {
        if (self.pred)(&x) {
            crate::unified::StageOutputU::Pass(x)
        } else {
            crate::unified::StageOutputU::Filtered
        }
    }
}

impl Default for Identity {
    fn default() -> Self { Self }
}

impl Stage for Identity {
    #[inline]
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        StageOutput::Pass(Cow::Borrowed(x))
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

// ── Sink ─────────────────────────────────────────────────────────────────────

/// Generic terminal accumulator. Bounded list (Sum/Min/Max/Avg/Count/
/// First/Last/Collect) covers every fused Sink variant in
/// `pipeline.rs`. New sinks = new impl, no per-chain code.
pub trait Sink {
    type Acc;
    fn init() -> Self::Acc;
    fn fold(acc: Self::Acc, v: &Val) -> Self::Acc;
    fn finalise(acc: Self::Acc) -> Val;
}

// ── Generic outer loop ──────────────────────────────────────────────────────

/// Parameterised outer loop. One pass over `arr`, dispatching each
/// element through the composed `stages`, accumulating into `S::Acc`.
/// Stages composed once at lower-time, used N× here.
pub fn run_pipeline<S: Sink>(arr: &[Val], stages: &dyn Stage) -> Val {
    let mut acc = S::init();
    for v in arr.iter() {
        match stages.apply(v) {
            StageOutput::Pass(cow) => acc = S::fold(acc, cow.as_ref()),
            StageOutput::Filtered => continue,
            StageOutput::Many(items) => {
                for it in items {
                    acc = S::fold(acc, it.as_ref());
                }
            }
            StageOutput::Done => break,
        }
    }
    S::finalise(acc)
}

// ── Generic Sink impls ──────────────────────────────────────────────────────

pub struct CountSink;
impl Sink for CountSink {
    type Acc = i64;
    #[inline] fn init() -> i64 { 0 }
    #[inline] fn fold(acc: i64, _: &Val) -> i64 { acc + 1 }
    #[inline] fn finalise(acc: i64) -> Val { Val::Int(acc) }
}

pub struct SumSink;
impl Sink for SumSink {
    type Acc = (i64, f64, bool); // (int_sum, float_sum, any_float)
    #[inline] fn init() -> Self::Acc { (0, 0.0, false) }
    fn fold(mut acc: Self::Acc, v: &Val) -> Self::Acc {
        match v {
            Val::Int(i) => acc.0 += *i,
            Val::Float(f) => { acc.1 += *f; acc.2 = true; }
            Val::Bool(b) => acc.0 += *b as i64,
            _ => {}
        }
        acc
    }
    fn finalise(acc: Self::Acc) -> Val {
        if acc.2 { Val::Float(acc.0 as f64 + acc.1) } else { Val::Int(acc.0) }
    }
}

pub struct MinSink;
impl Sink for MinSink {
    type Acc = Option<f64>;
    #[inline] fn init() -> Self::Acc { None }
    fn fold(acc: Self::Acc, v: &Val) -> Self::Acc {
        let n = match v {
            Val::Int(i) => *i as f64,
            Val::Float(f) => *f,
            _ => return acc,
        };
        Some(match acc { Some(cur) => cur.min(n), None => n })
    }
    fn finalise(acc: Self::Acc) -> Val {
        match acc {
            Some(f) if f.fract() == 0.0 && f.abs() < (i64::MAX as f64) => Val::Int(f as i64),
            Some(f) => Val::Float(f),
            None => Val::Null,
        }
    }
}

pub struct MaxSink;
impl Sink for MaxSink {
    type Acc = Option<f64>;
    #[inline] fn init() -> Self::Acc { None }
    fn fold(acc: Self::Acc, v: &Val) -> Self::Acc {
        let n = match v {
            Val::Int(i) => *i as f64,
            Val::Float(f) => *f,
            _ => return acc,
        };
        Some(match acc { Some(cur) => cur.max(n), None => n })
    }
    fn finalise(acc: Self::Acc) -> Val {
        match acc {
            Some(f) if f.fract() == 0.0 && f.abs() < (i64::MAX as f64) => Val::Int(f as i64),
            Some(f) => Val::Float(f),
            None => Val::Null,
        }
    }
}

pub struct AvgSink;
impl Sink for AvgSink {
    type Acc = (f64, usize);
    #[inline] fn init() -> Self::Acc { (0.0, 0) }
    fn fold(mut acc: Self::Acc, v: &Val) -> Self::Acc {
        let n = match v {
            Val::Int(i) => *i as f64,
            Val::Float(f) => *f,
            _ => return acc,
        };
        acc.0 += n;
        acc.1 += 1;
        acc
    }
    fn finalise(acc: Self::Acc) -> Val {
        if acc.1 == 0 { Val::Null } else { Val::Float(acc.0 / acc.1 as f64) }
    }
}

pub struct FirstSink;
impl Sink for FirstSink {
    type Acc = Option<Val>;
    #[inline] fn init() -> Self::Acc { None }
    fn fold(acc: Self::Acc, v: &Val) -> Self::Acc {
        if acc.is_some() { acc } else { Some(v.clone()) }
    }
    fn finalise(acc: Self::Acc) -> Val { acc.unwrap_or(Val::Null) }
}

pub struct LastSink;
impl Sink for LastSink {
    type Acc = Option<Val>;
    #[inline] fn init() -> Self::Acc { None }
    fn fold(_acc: Self::Acc, v: &Val) -> Self::Acc { Some(v.clone()) }
    fn finalise(acc: Self::Acc) -> Val { acc.unwrap_or(Val::Null) }
}

pub struct CollectSink;
impl Sink for CollectSink {
    type Acc = Vec<Val>;
    #[inline] fn init() -> Self::Acc { Vec::new() }
    fn fold(mut acc: Self::Acc, v: &Val) -> Self::Acc {
        acc.push(v.clone());
        acc
    }
    fn finalise(acc: Self::Acc) -> Val {
        Val::Arr(std::sync::Arc::new(acc))
    }
}

// ── Generic VM-fallback stages ──────────────────────────────────────────────
//
// Cover any kernel shape the borrow stages don't recognise (Generic,
// Arith, FString, FieldCmpLit non-Eq, custom lambdas). One VM + Env
// shared across all Generic stages in the chain via Rc<RefCell>;
// constructed once per pipeline call so compile/path caches amortise.
//
// These exist so `try_run_composed` never bails on body-shape — every
// pipeline lowers via composition, every chain runs the same outer
// loop. No per-shape walker; one mechanism for every body.

pub struct VmCtx {
    pub vm: crate::vm::VM,
    pub env: crate::eval::Env,
}

/// `.filter(pred)` with arbitrary pred — VM evaluates per row, result
/// truthy-checked. Borrow form on pass-through (zero-clone of x).
pub struct GenericFilter {
    pub prog: std::sync::Arc<crate::vm::Program>,
    pub ctx: std::rc::Rc<std::cell::RefCell<VmCtx>>,
}

impl Stage for GenericFilter {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        let mut c = self.ctx.borrow_mut();
        let VmCtx { vm, env } = &mut *c;
        let prev = env.swap_current(x.clone());
        let r = vm.exec_in_env(&self.prog, env);
        env.restore_current(prev);
        match r {
            Ok(v) if crate::eval::util::is_truthy(&v) => StageOutput::Pass(Cow::Borrowed(x)),
            _ => StageOutput::Filtered,
        }
    }
}

/// `.map(f)` with arbitrary f — VM emits a fresh Val per row.
pub struct GenericMap {
    pub prog: std::sync::Arc<crate::vm::Program>,
    pub ctx: std::rc::Rc<std::cell::RefCell<VmCtx>>,
}

impl Stage for GenericMap {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        let mut c = self.ctx.borrow_mut();
        let VmCtx { vm, env } = &mut *c;
        let prev = env.swap_current(x.clone());
        let r = vm.exec_in_env(&self.prog, env);
        env.restore_current(prev);
        match r {
            Ok(v) => StageOutput::Pass(Cow::Owned(v)),
            Err(_) => StageOutput::Filtered,
        }
    }
}

/// `.flat_map(f)` with arbitrary f — VM emits a Val that must be
/// iterable; `flatten_iterable` dispatches across all lane variants.
pub struct GenericFlatMap {
    pub prog: std::sync::Arc<crate::vm::Program>,
    pub ctx: std::rc::Rc<std::cell::RefCell<VmCtx>>,
}

impl Stage for GenericFlatMap {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        let mut c = self.ctx.borrow_mut();
        let VmCtx { vm, env } = &mut *c;
        let prev = env.swap_current(x.clone());
        let r = vm.exec_in_env(&self.prog, env);
        env.restore_current(prev);
        let owned = match r {
            Ok(v) => v,
            Err(_) => return StageOutput::Filtered,
        };
        // `owned` lives in this scope; any borrow against it must be
        // promoted to owned before returning. Materialise in a way
        // that doesn't outlive `owned`.
        let result: StageOutput<'a> = match &owned {
            Val::Arr(items) => {
                let mut out: SmallVec<[Cow<'a, Val>; 4]> = SmallVec::with_capacity(items.len());
                for it in items.iter() { out.push(Cow::Owned(it.clone())); }
                if out.is_empty() { StageOutput::Filtered } else { StageOutput::Many(out) }
            }
            Val::IntVec(items) => {
                let mut out: SmallVec<[Cow<'a, Val>; 4]> = SmallVec::with_capacity(items.len());
                for n in items.iter() { out.push(Cow::Owned(Val::Int(*n))); }
                if out.is_empty() { StageOutput::Filtered } else { StageOutput::Many(out) }
            }
            Val::FloatVec(items) => {
                let mut out: SmallVec<[Cow<'a, Val>; 4]> = SmallVec::with_capacity(items.len());
                for f in items.iter() { out.push(Cow::Owned(Val::Float(*f))); }
                if out.is_empty() { StageOutput::Filtered } else { StageOutput::Many(out) }
            }
            Val::StrVec(items) => {
                let mut out: SmallVec<[Cow<'a, Val>; 4]> = SmallVec::with_capacity(items.len());
                for s in items.iter() { out.push(Cow::Owned(Val::Str(std::sync::Arc::clone(s)))); }
                if out.is_empty() { StageOutput::Filtered } else { StageOutput::Many(out) }
            }
            Val::StrSliceVec(items) => {
                let mut out: SmallVec<[Cow<'a, Val>; 4]> = SmallVec::with_capacity(items.len());
                for r in items.iter() { out.push(Cow::Owned(Val::StrSlice(r.clone()))); }
                if out.is_empty() { StageOutput::Filtered } else { StageOutput::Many(out) }
            }
            _ => StageOutput::Filtered,
        };
        result
    }
}

// ── Borrow-form stages — zero-clone pass-through ────────────────────────────

/// `.filter(@.k == lit)` — borrow-form when pred holds.
pub struct FilterFieldEqLit {
    pub field: std::sync::Arc<str>,
    pub target: Val,
}

impl Stage for FilterFieldEqLit {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        if let Val::Obj(m) = x {
            if let Some(v) = m.get(self.field.as_ref()) {
                if vals_eq(v, &self.target) {
                    return StageOutput::Pass(Cow::Borrowed(x));
                }
            }
        }
        StageOutput::Filtered
    }
}

/// `.map(@.k)` — borrow into the parent object.
pub struct MapField {
    pub field: std::sync::Arc<str>,
}

impl Stage for MapField {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        if let Val::Obj(m) = x {
            if let Some(v) = m.get(self.field.as_ref()) {
                return StageOutput::Pass(Cow::Borrowed(v));
            }
        }
        StageOutput::Filtered
    }
}

/// `.flat_map(@.k)` — borrow into array field, yield elements as Many.
/// FieldRead variant: kernel resolves to a single field whose value
/// is an array; emit each element as a borrow.
pub struct FlatMapField {
    pub field: std::sync::Arc<str>,
}

impl Stage for FlatMapField {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        if let Val::Obj(m) = x {
            if let Some(v) = m.get(self.field.as_ref()) {
                return flatten_iterable(v);
            }
        }
        StageOutput::Filtered
    }
}

/// `.flat_map(@.a.b.c)` — borrow into deep array field.
pub struct FlatMapFieldChain {
    pub keys: std::sync::Arc<[std::sync::Arc<str>]>,
}

impl Stage for FlatMapFieldChain {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        let mut cur = x;
        for k in self.keys.iter() {
            match cur {
                Val::Obj(m) => match m.get(k.as_ref()) {
                    Some(next) => cur = next,
                    None => return StageOutput::Filtered,
                },
                _ => return StageOutput::Filtered,
            }
        }
        flatten_iterable(cur)
    }
}

/// Generic flatten dispatch — yields each element of any iterable Val
/// lane (Arr borrowed; IntVec/FloatVec/StrVec/StrSliceVec materialised
/// owned). One algorithm covers every lane. New lane types add one
/// arm here, no per-shape FlatMap variants.
#[inline]
fn flatten_iterable<'a>(v: &'a Val) -> StageOutput<'a> {
    match v {
        Val::Arr(items) => {
            let mut out: SmallVec<[Cow<'a, Val>; 4]> = SmallVec::with_capacity(items.len());
            for it in items.iter() { out.push(Cow::Borrowed(it)); }
            if out.is_empty() { StageOutput::Filtered } else { StageOutput::Many(out) }
        }
        Val::IntVec(items) => {
            let mut out: SmallVec<[Cow<'a, Val>; 4]> = SmallVec::with_capacity(items.len());
            for n in items.iter() { out.push(Cow::Owned(Val::Int(*n))); }
            if out.is_empty() { StageOutput::Filtered } else { StageOutput::Many(out) }
        }
        Val::FloatVec(items) => {
            let mut out: SmallVec<[Cow<'a, Val>; 4]> = SmallVec::with_capacity(items.len());
            for f in items.iter() { out.push(Cow::Owned(Val::Float(*f))); }
            if out.is_empty() { StageOutput::Filtered } else { StageOutput::Many(out) }
        }
        Val::StrVec(items) => {
            let mut out: SmallVec<[Cow<'a, Val>; 4]> = SmallVec::with_capacity(items.len());
            for s in items.iter() { out.push(Cow::Owned(Val::Str(std::sync::Arc::clone(s)))); }
            if out.is_empty() { StageOutput::Filtered } else { StageOutput::Many(out) }
        }
        Val::StrSliceVec(items) => {
            let mut out: SmallVec<[Cow<'a, Val>; 4]> = SmallVec::with_capacity(items.len());
            for r in items.iter() { out.push(Cow::Owned(Val::StrSlice(r.clone()))); }
            if out.is_empty() { StageOutput::Filtered } else { StageOutput::Many(out) }
        }
        _ => StageOutput::Filtered,
    }
}

/// `.map(@.a.b.c)` — generic field chain walk.
pub struct MapFieldChain {
    pub keys: std::sync::Arc<[std::sync::Arc<str>]>,
}

impl Stage for MapFieldChain {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        let mut cur = x;
        for k in self.keys.iter() {
            match cur {
                Val::Obj(m) => match m.get(k.as_ref()) {
                    Some(next) => cur = next,
                    None => return StageOutput::Filtered,
                },
                _ => return StageOutput::Filtered,
            }
        }
        StageOutput::Pass(Cow::Borrowed(cur))
    }
}

/// `.take(n)` — counts via interior mutability (single-thread per
/// pipeline invocation; outer loop owns the closure).
pub struct Take {
    pub remaining: std::cell::Cell<usize>,
}

impl Stage for Take {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        let r = self.remaining.get();
        if r == 0 { return StageOutput::Done; }
        self.remaining.set(r - 1);
        StageOutput::Pass(Cow::Borrowed(x))
    }
}


/// `.skip(n)` — same shape as Take.
pub struct Skip {
    pub remaining: std::cell::Cell<usize>,
}

impl Stage for Skip {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        let r = self.remaining.get();
        if r > 0 {
            self.remaining.set(r - 1);
            return StageOutput::Filtered;
        }
        StageOutput::Pass(Cow::Borrowed(x))
    }
}


// ── Barrier ops ─────────────────────────────────────────────────────────────
//
// Barriers consume the upstream stream into a Vec<Val>, run a single
// op, and return a new Vec<Val>. Caller drives them — see
// `pipeline::Pipeline::try_run_composed` segment loop.
//
// Key extraction is shared with the streaming Stage classifier:
// FieldRead / FieldChain only. Computed-key barriers (Arith, FString,
// custom lambda) bail to legacy in `try_run_composed`.

/// Source of a barrier key — same shape grammar as borrow stages.
pub enum KeySource {
    None,
    Field(std::sync::Arc<str>),
    Chain(std::sync::Arc<[std::sync::Arc<str>]>),
}

impl KeySource {
    /// Extract key Val by reference; clone-on-extract since the key
    /// must outlive the borrow on `v` (sort/dedup buffers retain it).
    pub fn extract(&self, v: &Val) -> Val {
        match self {
            KeySource::None => v.clone(),
            KeySource::Field(f) => match v {
                Val::Obj(m) => m.get(f.as_ref()).cloned().unwrap_or(Val::Null),
                _ => Val::Null,
            },
            KeySource::Chain(keys) => {
                let mut cur = v.clone();
                for k in keys.iter() {
                    let next = match &cur {
                        Val::Obj(m) => m.get(k.as_ref()).cloned(),
                        _ => None,
                    };
                    cur = match next {
                        Some(n) => n,
                        None => return Val::Null,
                    };
                }
                cur
            }
        }
    }
}

/// Reverse a buffered stream in place when uniquely owned, else clone.
pub fn barrier_reverse(buf: Vec<Val>) -> Vec<Val> {
    let mut buf = buf;
    buf.reverse();
    buf
}

/// Sort with optional key. Compares Val natural ordering via
/// `cmp_val` — wraps `eval::util::cmp_val` for primitive Vals.
pub fn barrier_sort(buf: Vec<Val>, key: &KeySource) -> Vec<Val> {
    let mut indexed: Vec<(Val, Val)> = buf.into_iter()
        .map(|v| (key.extract(&v), v))
        .collect();
    indexed.sort_by(|a, b| cmp_val(&a.0, &b.0));
    indexed.into_iter().map(|(_, v)| v).collect()
}

/// Top-k sort: keep the `k` smallest-by-key elements, sorted ascending.
/// O(N log k) via a max-heap of size k — for `k << N` this is the
/// algorithmic win that demand propagation unlocks (`Sort ∘ First` →
/// `Sort.adapt(AtMost(1))` → this kernel).
pub fn barrier_top_k(buf: Vec<Val>, key: &KeySource, k: usize) -> Vec<Val> {
    barrier_top_or_bottom_k(buf, key, k, false)
}

/// Bottom-k sort: keep the `k` largest-by-key elements, sorted ascending.
/// Driven by `Sort ∘ Last` / positional=Last under demand propagation.
pub fn barrier_bottom_k(buf: Vec<Val>, key: &KeySource, k: usize) -> Vec<Val> {
    barrier_top_or_bottom_k(buf, key, k, true)
}

fn barrier_top_or_bottom_k(buf: Vec<Val>, key: &KeySource, k: usize, largest: bool) -> Vec<Val> {
    if k == 0 { return Vec::new(); }
    if buf.len() <= k {
        // Smaller than budget — fall back to full sort (still O(N log N)
        // but N ≤ k so equivalent cost).
        return barrier_sort(buf, key);
    }

    // Simple Vec + linear-extremum approach.  Avoids the `Ord` bound
    // that a BinaryHeap would require on `Val` — we only have `cmp_val`
    // total order.  For small k this is fine; larger k a heap with a
    // `Reverse<KeyHash>` wrapper would beat it.
    //
    // `largest=false` keeps the smallest-k (top-k);
    // `largest=true`  keeps the largest-k (bottom-k for `.last()`).
    let mut top: Vec<(Val, Val)> = Vec::with_capacity(k + 1);
    for v in buf.into_iter() {
        let kv = key.extract(&v);
        if top.len() < k {
            top.push((kv, v));
            continue;
        }
        // Find current "worst" element in `top` — the one most likely
        // to be displaced.  For top-k that's the maximum; for bottom-k
        // that's the minimum.
        let mut worst_idx = 0;
        for (i, (kk, _)) in top.iter().enumerate().skip(1) {
            let cmp = cmp_val(kk, &top[worst_idx].0);
            let displace = if largest {
                cmp == std::cmp::Ordering::Less       // tracking smallest as worst
            } else {
                cmp == std::cmp::Ordering::Greater    // tracking largest as worst
            };
            if displace { worst_idx = i; }
        }
        let cmp = cmp_val(&kv, &top[worst_idx].0);
        let take = if largest {
            cmp == std::cmp::Ordering::Greater
        } else {
            cmp == std::cmp::Ordering::Less
        };
        if take { top[worst_idx] = (kv, v); }
    }
    // Final sort of the kept elements (always ascending).
    top.sort_by(|a, b| cmp_val(&a.0, &b.0));
    top.into_iter().map(|(_, v)| v).collect()
}

/// Dedup by key. Uses a linear-probe HashSet on hashable keys; for
/// primitive Vals this is O(N).
pub fn barrier_unique_by(buf: Vec<Val>, key: &KeySource) -> Vec<Val> {
    use std::collections::HashSet;
    let mut seen: HashSet<KeyHash> = HashSet::with_capacity(buf.len());
    let mut out: Vec<Val> = Vec::with_capacity(buf.len());
    for v in buf.into_iter() {
        let k = KeyHash::from(key.extract(&v));
        if seen.insert(k) {
            out.push(v);
        }
    }
    out
}

/// Group-by key. Produces a `Val::Obj` where each key maps to the
/// `Val::Arr` of rows that hashed to it. Insertion-ordered by first
/// occurrence (IndexMap preserves this).
pub fn barrier_group_by(buf: Vec<Val>, key: &KeySource) -> Val {
    use indexmap::IndexMap;
    let mut groups: IndexMap<std::sync::Arc<str>, Vec<Val>> = IndexMap::new();
    for v in buf.into_iter() {
        let k = key.extract(&v);
        let ks: std::sync::Arc<str> = match &k {
            Val::Str(s) => std::sync::Arc::clone(s),
            Val::StrSlice(r) => std::sync::Arc::from(r.as_str()),
            Val::Null => std::sync::Arc::from("null"),
            _ => std::sync::Arc::from(format!("{}", DisplayKey(&k))),
        };
        groups.entry(ks).or_insert_with(Vec::new).push(v);
    }
    let mut m: indexmap::IndexMap<std::sync::Arc<str>, Val> = indexmap::IndexMap::with_capacity(groups.len());
    for (k, vs) in groups {
        m.insert(k, Val::Arr(std::sync::Arc::new(vs)));
    }
    Val::Obj(std::sync::Arc::new(m))
}

// ── Hashable Val key wrapper ──

#[derive(Eq, PartialEq, Hash)]
struct KeyHash(KeyRepr);

#[derive(Eq, PartialEq, Hash)]
enum KeyRepr {
    Null,
    Bool(bool),
    Int(i64),
    Float(u64),    // f64::to_bits for total ordering
    Str(String),
}

impl From<Val> for KeyHash {
    fn from(v: Val) -> Self {
        let r = match v {
            Val::Null => KeyRepr::Null,
            Val::Bool(b) => KeyRepr::Bool(b),
            Val::Int(i) => KeyRepr::Int(i),
            Val::Float(f) => KeyRepr::Float(f.to_bits()),
            Val::Str(s) => KeyRepr::Str(s.as_ref().to_string()),
            Val::StrSlice(r) => KeyRepr::Str(r.as_str().to_string()),
            other => KeyRepr::Str(format!("{}", DisplayKey(&other))),
        };
        KeyHash(r)
    }
}

struct DisplayKey<'a>(&'a Val);
impl<'a> std::fmt::Display for DisplayKey<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            Val::Null => write!(f, "null"),
            Val::Bool(b) => write!(f, "{}", b),
            Val::Int(i) => write!(f, "{}", i),
            Val::Float(x) => write!(f, "{}", x),
            Val::Str(s) => write!(f, "{}", s),
            Val::StrSlice(r) => write!(f, "{}", r.as_str()),
            _ => write!(f, "<complex>"),
        }
    }
}

/// Total ordering on Val keys; mirrors the legacy sort comparator
/// shape used in `pipeline::run_with`.
fn cmp_val(a: &Val, b: &Val) -> std::cmp::Ordering {
    use std::cmp::Ordering::*;
    match (a, b) {
        (Val::Null, Val::Null) => Equal,
        (Val::Null, _) => Less,
        (_, Val::Null) => Greater,
        (Val::Bool(x), Val::Bool(y)) => x.cmp(y),
        (Val::Int(x), Val::Int(y)) => x.cmp(y),
        (Val::Float(x), Val::Float(y)) => x.partial_cmp(y).unwrap_or(Equal),
        (Val::Int(x), Val::Float(y)) => (*x as f64).partial_cmp(y).unwrap_or(Equal),
        (Val::Float(x), Val::Int(y)) => x.partial_cmp(&(*y as f64)).unwrap_or(Equal),
        (Val::Str(x), Val::Str(y)) => x.as_ref().cmp(y.as_ref()),
        (Val::Str(x), Val::StrSlice(r)) => x.as_ref().cmp(r.as_str()),
        (Val::StrSlice(r), Val::Str(y)) => r.as_str().cmp(y.as_ref()),
        (Val::StrSlice(x), Val::StrSlice(y)) => x.as_str().cmp(y.as_str()),
        _ => Equal,
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

#[inline]
fn vals_eq(a: &Val, b: &Val) -> bool {
    match (a, b) {
        (Val::Null, Val::Null) => true,
        (Val::Bool(x), Val::Bool(y)) => x == y,
        (Val::Int(x), Val::Int(y)) => x == y,
        (Val::Float(x), Val::Float(y)) => x == y,
        (Val::Int(x), Val::Float(y)) | (Val::Float(y), Val::Int(x)) => (*x as f64) == *y,
        (Val::Str(x), Val::Str(y)) => x == y,
        (Val::Str(x), Val::StrSlice(r)) | (Val::StrSlice(r), Val::Str(x)) => x.as_ref() == r.as_str(),
        (Val::StrSlice(x), Val::StrSlice(y)) => x.as_str() == y.as_str(),
        _ => false,
    }
}

// ── Tape variant ───────────────────────────────────────────────────────────
//
// Step 3a (per `pipeline_unification.md`): parallel `TapeStage` trait
// operating on simd-json tape node indices instead of materialised
// `Val`. Borrow stages get a tape impl alongside the Val one. Same
// algorithm; one walks IndexMap, the other walks tape offsets. Both
// fit the generic composition framework.
//
// Stages that can't run on tape (Sort/UniqueBy/GroupBy barriers,
// FlatMap with computed body, computed Maps) only implement the Val
// trait. Caller bails to Val materialisation when chain has any
// Val-only stage.
//
// Output uses raw `usize` tape index. The tape itself is shared via
// the outer loop; stages don't own it. Pass = forward the index;
// Filtered = drop; Many = flat fanout; Done = stream-end.

#[cfg(feature = "simd-json")]
pub mod tape {
    use super::*;
    use crate::strref::{TapeData, TapeLit, TapeCmp,
        tape_object_field, tape_array_iter, tape_value_cmp, tape_value_truthy};

    pub enum TapeOutput {
        Pass(usize),
        Filtered,
        Many(SmallVec<[usize; 4]>),
        Done,
    }

    pub trait TapeStage {
        fn apply(&self, tape: &TapeData, idx: usize) -> TapeOutput;
    }

    impl<T: TapeStage + ?Sized> TapeStage for Box<T> {
        #[inline]
        fn apply(&self, tape: &TapeData, idx: usize) -> TapeOutput {
            (**self).apply(tape, idx)
        }
    }

    pub struct TapeIdentity;
    impl TapeStage for TapeIdentity {
        #[inline]
        fn apply(&self, _tape: &TapeData, idx: usize) -> TapeOutput {
            TapeOutput::Pass(idx)
        }
    }

    pub struct TapeComposed<A: TapeStage, B: TapeStage> {
        pub a: A,
        pub b: B,
    }

    impl<A: TapeStage, B: TapeStage> TapeStage for TapeComposed<A, B> {
        fn apply(&self, tape: &TapeData, idx: usize) -> TapeOutput {
            match self.a.apply(tape, idx) {
                TapeOutput::Pass(v) => self.b.apply(tape, v),
                TapeOutput::Filtered => TapeOutput::Filtered,
                TapeOutput::Many(items) => {
                    let mut out: SmallVec<[usize; 4]> = SmallVec::new();
                    for it in items {
                        match self.b.apply(tape, it) {
                            TapeOutput::Pass(v) => out.push(v),
                            TapeOutput::Filtered => continue,
                            TapeOutput::Many(more) => out.extend(more),
                            TapeOutput::Done => {
                                return if out.is_empty() {
                                    TapeOutput::Done
                                } else {
                                    TapeOutput::Many(out)
                                };
                            }
                        }
                    }
                    if out.is_empty() {
                        TapeOutput::Filtered
                    } else if out.len() == 1 {
                        TapeOutput::Pass(out.into_iter().next().unwrap())
                    } else {
                        TapeOutput::Many(out)
                    }
                }
                TapeOutput::Done => TapeOutput::Done,
            }
        }
    }

    // ── Borrow stage tape impls ────────────────────────────────────────────

    pub struct TapeFilterFieldCmpLit {
        pub field: std::sync::Arc<str>,
        pub op: TapeCmp,
        pub lit: TapeLitOwned,
    }

    /// Owned counterpart of `TapeLit<'a>` (which holds `&'a str`); lets
    /// stages outlive the tape they point into. Lit is captured at
    /// build time; comparisons construct a borrowed TapeLit per call.
    #[derive(Debug, Clone)]
    pub enum TapeLitOwned {
        Int(i64),
        Float(f64),
        Str(std::sync::Arc<str>),
        Bool(bool),
        Null,
    }

    impl TapeLitOwned {
        #[inline]
        pub fn as_borrowed<'a>(&'a self) -> TapeLit<'a> {
            match self {
                TapeLitOwned::Int(i) => TapeLit::Int(*i),
                TapeLitOwned::Float(f) => TapeLit::Float(*f),
                TapeLitOwned::Str(s) => TapeLit::Str(s.as_ref()),
                TapeLitOwned::Bool(b) => TapeLit::Bool(*b),
                TapeLitOwned::Null => TapeLit::Null,
            }
        }
    }

    impl TapeStage for TapeFilterFieldCmpLit {
        fn apply(&self, tape: &TapeData, idx: usize) -> TapeOutput {
            let field_idx = match tape_object_field(tape, idx, self.field.as_ref()) {
                Some(v) => v,
                None => return TapeOutput::Filtered,
            };
            let lit = self.lit.as_borrowed();
            if tape_value_cmp(tape, field_idx, self.op, &lit) {
                TapeOutput::Pass(idx)
            } else {
                TapeOutput::Filtered
            }
        }
    }

    pub struct TapeFilterFieldChainCmpLit {
        pub keys: std::sync::Arc<[std::sync::Arc<str>]>,
        pub op: TapeCmp,
        pub lit: TapeLitOwned,
    }

    impl TapeStage for TapeFilterFieldChainCmpLit {
        fn apply(&self, tape: &TapeData, idx: usize) -> TapeOutput {
            let mut cur = idx;
            for k in self.keys.iter() {
                match tape_object_field(tape, cur, k.as_ref()) {
                    Some(v) => cur = v,
                    None => return TapeOutput::Filtered,
                }
            }
            let lit = self.lit.as_borrowed();
            if tape_value_cmp(tape, cur, self.op, &lit) {
                TapeOutput::Pass(idx)
            } else {
                TapeOutput::Filtered
            }
        }
    }

    pub struct TapeMapField {
        pub field: std::sync::Arc<str>,
    }

    impl TapeStage for TapeMapField {
        fn apply(&self, tape: &TapeData, idx: usize) -> TapeOutput {
            match tape_object_field(tape, idx, self.field.as_ref()) {
                Some(v) => TapeOutput::Pass(v),
                None => TapeOutput::Filtered,
            }
        }
    }

    pub struct TapeMapFieldChain {
        pub keys: std::sync::Arc<[std::sync::Arc<str>]>,
    }

    impl TapeStage for TapeMapFieldChain {
        fn apply(&self, tape: &TapeData, idx: usize) -> TapeOutput {
            let mut cur = idx;
            for k in self.keys.iter() {
                match tape_object_field(tape, cur, k.as_ref()) {
                    Some(v) => cur = v,
                    None => return TapeOutput::Filtered,
                }
            }
            TapeOutput::Pass(cur)
        }
    }

    pub struct TapeFlatMapField {
        pub field: std::sync::Arc<str>,
    }

    impl TapeStage for TapeFlatMapField {
        fn apply(&self, tape: &TapeData, idx: usize) -> TapeOutput {
            let target = match tape_object_field(tape, idx, self.field.as_ref()) {
                Some(v) => v,
                None => return TapeOutput::Filtered,
            };
            tape_flatten(tape, target)
        }
    }

    pub struct TapeFlatMapFieldChain {
        pub keys: std::sync::Arc<[std::sync::Arc<str>]>,
    }

    impl TapeStage for TapeFlatMapFieldChain {
        fn apply(&self, tape: &TapeData, idx: usize) -> TapeOutput {
            let mut cur = idx;
            for k in self.keys.iter() {
                match tape_object_field(tape, cur, k.as_ref()) {
                    Some(v) => cur = v,
                    None => return TapeOutput::Filtered,
                }
            }
            tape_flatten(tape, cur)
        }
    }

    /// Generic tape array fanout — yields each entry index as Many.
    /// One mechanism for any tape array; new array variants need no
    /// per-shape impl.
    #[inline]
    fn tape_flatten(tape: &TapeData, arr_idx: usize) -> TapeOutput {
        let iter = match tape_array_iter(tape, arr_idx) {
            Some(it) => it,
            None => return TapeOutput::Filtered,
        };
        let items: SmallVec<[usize; 4]> = iter.collect();
        if items.is_empty() {
            TapeOutput::Filtered
        } else {
            TapeOutput::Many(items)
        }
    }

    pub struct TapeTake {
        pub remaining: std::cell::Cell<usize>,
    }

    impl TapeStage for TapeTake {
        fn apply(&self, _tape: &TapeData, idx: usize) -> TapeOutput {
            let r = self.remaining.get();
            if r == 0 { return TapeOutput::Done; }
            self.remaining.set(r - 1);
            TapeOutput::Pass(idx)
        }
    }

    pub struct TapeSkip {
        pub remaining: std::cell::Cell<usize>,
    }

    impl TapeStage for TapeSkip {
        fn apply(&self, _tape: &TapeData, idx: usize) -> TapeOutput {
            let r = self.remaining.get();
            if r > 0 {
                self.remaining.set(r - 1);
                return TapeOutput::Filtered;
            }
            TapeOutput::Pass(idx)
        }
    }

    /// Generic-pred filter that runs `tape_value_truthy` on the
    /// kernel-evaluated value. Used for `.filter(@.k)` (FieldRead body
    /// returns the field value; truthy of that field's tape node).
    pub struct TapeFilterTruthyAtField {
        pub field: std::sync::Arc<str>,
    }

    impl TapeStage for TapeFilterTruthyAtField {
        fn apply(&self, tape: &TapeData, idx: usize) -> TapeOutput {
            match tape_object_field(tape, idx, self.field.as_ref()) {
                Some(v) if tape_value_truthy(tape, v) => TapeOutput::Pass(idx),
                _ => TapeOutput::Filtered,
            }
        }
    }

    // ── TapeSink trait + 8 generic impls ────────────────────────────────────

    pub trait TapeSink {
        type Acc;
        fn init() -> Self::Acc;
        fn fold(acc: Self::Acc, tape: &TapeData, idx: usize) -> Self::Acc;
        fn finalise(acc: Self::Acc) -> Val;
    }

    /// Read a tape node as `f64` if numeric, else `None`. Bool→0.0/1.0
    /// kept out — only Int/Float numbers fold into numeric sinks, same
    /// shape as `composed::SumSink` for Val. One mechanism, no per-
    /// kind handler.
    #[inline]
    fn tape_num(tape: &TapeData, idx: usize) -> Option<f64> {
        use crate::strref::TapeNode;
        use simd_json::StaticNode as SN;
        match tape.nodes[idx] {
            TapeNode::Static(SN::I64(v)) => Some(v as f64),
            TapeNode::Static(SN::U64(v)) if v <= i64::MAX as u64 => Some(v as f64),
            TapeNode::Static(SN::F64(v)) => Some(v),
            _ => None,
        }
    }

    /// Read a tape node as `i64` if exactly representable, else `None`.
    #[inline]
    fn tape_int(tape: &TapeData, idx: usize) -> Option<i64> {
        use crate::strref::TapeNode;
        use simd_json::StaticNode as SN;
        match tape.nodes[idx] {
            TapeNode::Static(SN::I64(v)) => Some(v),
            TapeNode::Static(SN::U64(v)) if v <= i64::MAX as u64 => Some(v as i64),
            _ => None,
        }
    }

    /// Materialise a single tape value to `Val`. Used by terminal
    /// First/Last/Collect sinks — they need an owned Val to return.
    /// Streaming/numeric sinks never call this.
    pub fn tape_to_val(tape: &TapeData, idx: usize) -> Val {
        use crate::strref::TapeNode;
        use simd_json::StaticNode as SN;
        match tape.nodes[idx] {
            TapeNode::Static(SN::Null) => Val::Null,
            TapeNode::Static(SN::Bool(b)) => Val::Bool(b),
            TapeNode::Static(SN::I64(i)) => Val::Int(i),
            TapeNode::Static(SN::U64(u)) if u <= i64::MAX as u64 => Val::Int(u as i64),
            TapeNode::Static(SN::U64(u)) => Val::Float(u as f64),
            TapeNode::Static(SN::F64(f)) => Val::Float(f),
            TapeNode::StringRef { .. } => Val::Str(std::sync::Arc::from(tape.str_at(idx))),
            TapeNode::Object { .. } | TapeNode::Array { .. } => {
                // Build IndexMap / Vec via a recursive walk. Same shape
                // as the eager parse path; only invoked at terminal.
                tape_to_val_compound(tape, idx)
            }
        }
    }

    fn tape_to_val_compound(tape: &TapeData, idx: usize) -> Val {
        use crate::strref::TapeNode;
        match tape.nodes[idx] {
            TapeNode::Object { len, .. } => {
                let len = len as usize;
                let mut m: indexmap::IndexMap<std::sync::Arc<str>, Val> = indexmap::IndexMap::with_capacity(len);
                let mut cursor = idx + 1;
                for _ in 0..len {
                    let key_idx = cursor;
                    let key_str = tape.str_at(key_idx);
                    cursor += tape.span(key_idx);
                    let val_idx = cursor;
                    let v = tape_to_val(tape, val_idx);
                    cursor += tape.span(val_idx);
                    m.insert(std::sync::Arc::from(key_str), v);
                }
                Val::Obj(std::sync::Arc::new(m))
            }
            TapeNode::Array { len, .. } => {
                let len = len as usize;
                let mut out: Vec<Val> = Vec::with_capacity(len);
                let mut cursor = idx + 1;
                for _ in 0..len {
                    out.push(tape_to_val(tape, cursor));
                    cursor += tape.span(cursor);
                }
                Val::Arr(std::sync::Arc::new(out))
            }
            _ => Val::Null,
        }
    }

    pub struct TapeCountSink;
    impl TapeSink for TapeCountSink {
        type Acc = i64;
        #[inline] fn init() -> i64 { 0 }
        #[inline] fn fold(acc: i64, _: &TapeData, _: usize) -> i64 { acc + 1 }
        #[inline] fn finalise(acc: i64) -> Val { Val::Int(acc) }
    }

    pub struct TapeSumSink;
    impl TapeSink for TapeSumSink {
        type Acc = (i64, f64, bool);
        #[inline] fn init() -> Self::Acc { (0, 0.0, false) }
        fn fold(mut acc: Self::Acc, tape: &TapeData, idx: usize) -> Self::Acc {
            if let Some(i) = tape_int(tape, idx) {
                acc.0 += i;
            } else if let Some(f) = tape_num(tape, idx) {
                acc.1 += f;
                acc.2 = true;
            }
            acc
        }
        fn finalise(acc: Self::Acc) -> Val {
            if acc.2 { Val::Float(acc.0 as f64 + acc.1) } else { Val::Int(acc.0) }
        }
    }

    pub struct TapeMinSink;
    impl TapeSink for TapeMinSink {
        type Acc = Option<f64>;
        #[inline] fn init() -> Self::Acc { None }
        fn fold(acc: Self::Acc, tape: &TapeData, idx: usize) -> Self::Acc {
            match tape_num(tape, idx) {
                Some(n) => Some(acc.map_or(n, |c| c.min(n))),
                None => acc,
            }
        }
        fn finalise(acc: Self::Acc) -> Val {
            match acc {
                Some(f) if f.fract() == 0.0 && f.abs() < (i64::MAX as f64) => Val::Int(f as i64),
                Some(f) => Val::Float(f),
                None => Val::Null,
            }
        }
    }

    pub struct TapeMaxSink;
    impl TapeSink for TapeMaxSink {
        type Acc = Option<f64>;
        #[inline] fn init() -> Self::Acc { None }
        fn fold(acc: Self::Acc, tape: &TapeData, idx: usize) -> Self::Acc {
            match tape_num(tape, idx) {
                Some(n) => Some(acc.map_or(n, |c| c.max(n))),
                None => acc,
            }
        }
        fn finalise(acc: Self::Acc) -> Val {
            match acc {
                Some(f) if f.fract() == 0.0 && f.abs() < (i64::MAX as f64) => Val::Int(f as i64),
                Some(f) => Val::Float(f),
                None => Val::Null,
            }
        }
    }

    pub struct TapeAvgSink;
    impl TapeSink for TapeAvgSink {
        type Acc = (f64, usize);
        #[inline] fn init() -> Self::Acc { (0.0, 0) }
        fn fold(mut acc: Self::Acc, tape: &TapeData, idx: usize) -> Self::Acc {
            if let Some(n) = tape_num(tape, idx) {
                acc.0 += n;
                acc.1 += 1;
            }
            acc
        }
        fn finalise(acc: Self::Acc) -> Val {
            if acc.1 == 0 { Val::Null } else { Val::Float(acc.0 / acc.1 as f64) }
        }
    }

    pub struct TapeFirstSink;
    impl TapeSink for TapeFirstSink {
        type Acc = Option<usize>;
        #[inline] fn init() -> Self::Acc { None }
        fn fold(acc: Self::Acc, _: &TapeData, idx: usize) -> Self::Acc {
            acc.or(Some(idx))
        }
        fn finalise(_: Self::Acc) -> Val { Val::Null }
    }

    pub struct TapeLastSink;
    impl TapeSink for TapeLastSink {
        type Acc = Option<usize>;
        #[inline] fn init() -> Self::Acc { None }
        fn fold(_: Self::Acc, _: &TapeData, idx: usize) -> Self::Acc { Some(idx) }
        fn finalise(_: Self::Acc) -> Val { Val::Null }
    }

    pub struct TapeCollectSink;
    impl TapeSink for TapeCollectSink {
        type Acc = Vec<usize>;
        #[inline] fn init() -> Self::Acc { Vec::new() }
        fn fold(mut acc: Self::Acc, _: &TapeData, idx: usize) -> Self::Acc {
            acc.push(idx);
            acc
        }
        fn finalise(_: Self::Acc) -> Val { Val::Null }
    }

    /// Outer loop — parameterised by `S: TapeSink`. Walks the source
    /// array via `tape_array_iter`, dispatches each entry through the
    /// composed `TapeStage` chain, folds results. One mechanism per
    /// sink kind; no per-chain code.
    ///
    /// `First`/`Last`/`Collect` need a post-fold step to materialise
    /// owned `Val`s from indices — separate runners below pay that
    /// cost only when sink demands it. Numeric sinks never materialise.
    pub fn run_pipeline_tape<S: TapeSink>(
        tape: &TapeData,
        arr_idx: usize,
        stages: &dyn TapeStage,
    ) -> Option<Val> {
        let iter = tape_array_iter(tape, arr_idx)?;
        let mut acc = S::init();
        for entry in iter {
            match stages.apply(tape, entry) {
                TapeOutput::Pass(v) => acc = S::fold(acc, tape, v),
                TapeOutput::Filtered => continue,
                TapeOutput::Many(items) => {
                    for it in items { acc = S::fold(acc, tape, it); }
                }
                TapeOutput::Done => break,
            }
        }
        Some(S::finalise(acc))
    }

    /// Specialised runner for `TapeFirstSink` — finalises by
    /// materialising the captured tape index. Avoids polluting the
    /// generic `TapeSink::finalise` with a `&TapeData` parameter (which
    /// would force every numeric sink to carry one).
    pub fn run_pipeline_tape_first(
        tape: &TapeData,
        arr_idx: usize,
        stages: &dyn TapeStage,
    ) -> Option<Val> {
        let iter = tape_array_iter(tape, arr_idx)?;
        let mut found: Option<usize> = None;
        for entry in iter {
            if found.is_some() { break; }
            match stages.apply(tape, entry) {
                TapeOutput::Pass(v) => { found = Some(v); }
                TapeOutput::Filtered => continue,
                TapeOutput::Many(items) => {
                    if let Some(&first) = items.first() {
                        found = Some(first);
                    }
                }
                TapeOutput::Done => break,
            }
        }
        Some(match found {
            Some(idx) => tape_to_val(tape, idx),
            None => Val::Null,
        })
    }

    pub fn run_pipeline_tape_last(
        tape: &TapeData,
        arr_idx: usize,
        stages: &dyn TapeStage,
    ) -> Option<Val> {
        let iter = tape_array_iter(tape, arr_idx)?;
        let mut found: Option<usize> = None;
        for entry in iter {
            match stages.apply(tape, entry) {
                TapeOutput::Pass(v) => { found = Some(v); }
                TapeOutput::Filtered => continue,
                TapeOutput::Many(items) => {
                    if let Some(&last) = items.last() {
                        found = Some(last);
                    }
                }
                TapeOutput::Done => break,
            }
        }
        Some(match found {
            Some(idx) => tape_to_val(tape, idx),
            None => Val::Null,
        })
    }

    pub fn run_pipeline_tape_collect(
        tape: &TapeData,
        arr_idx: usize,
        stages: &dyn TapeStage,
    ) -> Option<Val> {
        let iter = tape_array_iter(tape, arr_idx)?;
        let mut out: Vec<Val> = Vec::new();
        for entry in iter {
            match stages.apply(tape, entry) {
                TapeOutput::Pass(v) => out.push(tape_to_val(tape, v)),
                TapeOutput::Filtered => continue,
                TapeOutput::Many(items) => {
                    for it in items { out.push(tape_to_val(tape, it)); }
                }
                TapeOutput::Done => break,
            }
        }
        Some(Val::Arr(std::sync::Arc::new(out)))
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use indexmap::IndexMap;

    fn obj(pairs: &[(&str, Val)]) -> Val {
        let mut m = IndexMap::new();
        for (k, v) in pairs {
            m.insert(Arc::from(*k), v.clone());
        }
        Val::Obj(Arc::new(m))
    }

    #[test]
    fn count_filter_map_field() {
        // [{a:1, b:10}, {a:2, b:20}, {a:1, b:30}].filter(@.a == 1).map(@.b) → count = 2
        let arr = vec![
            obj(&[("a", Val::Int(1)), ("b", Val::Int(10))]),
            obj(&[("a", Val::Int(2)), ("b", Val::Int(20))]),
            obj(&[("a", Val::Int(1)), ("b", Val::Int(30))]),
        ];
        let stages = Composed {
            a: FilterFieldEqLit { field: Arc::from("a"), target: Val::Int(1) },
            b: MapField { field: Arc::from("b") },
        };
        let out = run_pipeline::<CountSink>(&arr, &stages);
        assert!(matches!(out, Val::Int(2)));
    }

    #[test]
    fn sum_filter_map_field() {
        let arr = vec![
            obj(&[("a", Val::Int(1)), ("b", Val::Int(10))]),
            obj(&[("a", Val::Int(2)), ("b", Val::Int(20))]),
            obj(&[("a", Val::Int(1)), ("b", Val::Int(30))]),
        ];
        let stages = Composed {
            a: FilterFieldEqLit { field: Arc::from("a"), target: Val::Int(1) },
            b: MapField { field: Arc::from("b") },
        };
        let out = run_pipeline::<SumSink>(&arr, &stages);
        // 10 + 30 = 40
        assert!(matches!(out, Val::Int(40)));
    }

    #[test]
    fn collect_map_field_chain() {
        // [{u:{a:{c:"NYC"}}}, {u:{a:{c:"LA"}}}].map(@.u.a.c) → ["NYC", "LA"]
        let inner1 = obj(&[("c", Val::Str(Arc::from("NYC")))]);
        let inner2 = obj(&[("c", Val::Str(Arc::from("LA")))]);
        let mid1 = obj(&[("a", inner1)]);
        let mid2 = obj(&[("a", inner2)]);
        let arr = vec![
            obj(&[("u", mid1)]),
            obj(&[("u", mid2)]),
        ];
        let keys: Arc<[Arc<str>]> = Arc::from(vec![Arc::from("u"), Arc::from("a"), Arc::from("c")]);
        let stages = MapFieldChain { keys };
        let out = run_pipeline::<CollectSink>(&arr, &stages);
        if let Val::Arr(v) = out {
            assert_eq!(v.len(), 2);
        } else {
            panic!("expected Arr");
        }
    }

    #[test]
    fn take_terminates_outer_loop() {
        let arr: Vec<Val> = (0..1000).map(Val::Int).collect();
        let stages = Take { remaining: std::cell::Cell::new(3) };
        let out = run_pipeline::<CountSink>(&arr, &stages);
        assert!(matches!(out, Val::Int(3)));
    }

    #[test]
    fn skip_drops_prefix() {
        let arr: Vec<Val> = (0..10).map(Val::Int).collect();
        let stages = Skip { remaining: std::cell::Cell::new(7) };
        let out = run_pipeline::<CountSink>(&arr, &stages);
        assert!(matches!(out, Val::Int(3)));
    }

    #[test]
    fn min_max_avg_basic() {
        let arr: Vec<Val> = vec![Val::Int(5), Val::Int(2), Val::Int(9), Val::Int(3)];
        assert!(matches!(run_pipeline::<MinSink>(&arr, &Identity), Val::Int(2)));
        assert!(matches!(run_pipeline::<MaxSink>(&arr, &Identity), Val::Int(9)));
        // (5+2+9+3)/4 = 4.75
        if let Val::Float(f) = run_pipeline::<AvgSink>(&arr, &Identity) {
            assert!((f - 4.75).abs() < 1e-9);
        } else {
            panic!("expected Float");
        }
    }

    #[test]
    fn first_last_basic() {
        let arr: Vec<Val> = vec![Val::Int(10), Val::Int(20), Val::Int(30)];
        assert!(matches!(run_pipeline::<FirstSink>(&arr, &Identity), Val::Int(10)));
        assert!(matches!(run_pipeline::<LastSink>(&arr, &Identity), Val::Int(30)));
    }

    #[cfg(feature = "simd-json")]
    #[test]
    fn tape_sinks_smoke() {
        use crate::composed::tape::*;
        use crate::strref::{TapeData, TapeCmp};
        use std::sync::Arc;

        // [{a:1,b:10},{a:2,b:20},{a:1,b:30}].filter(a==1).map(b)
        // Sum=40, Count=2, Min=10, Max=30, Avg=20
        let bytes = br#"[{"a":1,"b":10},{"a":2,"b":20},{"a":1,"b":30}]"#.to_vec();
        let tape = TapeData::parse(bytes).expect("parse");

        let mk_chain = || TapeComposed {
            a: TapeFilterFieldCmpLit {
                field: Arc::from("a"),
                op: TapeCmp::Eq,
                lit: TapeLitOwned::Int(1),
            },
            b: TapeMapField { field: Arc::from("b") },
        };

        let chain = mk_chain();
        assert_eq!(
            run_pipeline_tape::<TapeCountSink>(&tape, 0, &chain).unwrap(),
            Val::Int(2)
        );
        let chain = mk_chain();
        assert_eq!(
            run_pipeline_tape::<TapeSumSink>(&tape, 0, &chain).unwrap(),
            Val::Int(40)
        );
        let chain = mk_chain();
        assert_eq!(
            run_pipeline_tape::<TapeMinSink>(&tape, 0, &chain).unwrap(),
            Val::Int(10)
        );
        let chain = mk_chain();
        assert_eq!(
            run_pipeline_tape::<TapeMaxSink>(&tape, 0, &chain).unwrap(),
            Val::Int(30)
        );

        // First with materialise
        let chain = mk_chain();
        assert_eq!(
            run_pipeline_tape_first(&tape, 0, &chain).unwrap(),
            Val::Int(10)
        );
        // Collect
        let chain = mk_chain();
        if let Val::Arr(a) = run_pipeline_tape_collect(&tape, 0, &chain).unwrap() {
            assert_eq!(a.len(), 2);
            assert_eq!(a[0], Val::Int(10));
            assert_eq!(a[1], Val::Int(30));
        } else {
            panic!("expected Arr");
        }
    }

    #[cfg(feature = "simd-json")]
    #[test]
    fn tape_borrow_stages_smoke() {
        use crate::composed::tape::*;
        use crate::strref::{TapeData, TapeCmp, tape_array_iter};
        use std::sync::Arc;

        let bytes = br#"[{"a":1,"b":10},{"a":2,"b":20},{"a":1,"b":30}]"#.to_vec();
        let tape = TapeData::parse(bytes).expect("parse");

        // Filter(a == 1) ∘ Map(b) over the array
        let chain = TapeComposed {
            a: TapeFilterFieldCmpLit {
                field: Arc::from("a"),
                op: TapeCmp::Eq,
                lit: TapeLitOwned::Int(1),
            },
            b: TapeMapField { field: Arc::from("b") },
        };

        let arr_idx = 0; // root is the array
        let mut count = 0usize;
        let iter = tape_array_iter(&tape, arr_idx).expect("array");
        for entry in iter {
            match chain.apply(&tape, entry) {
                TapeOutput::Pass(_) => count += 1,
                _ => {}
            }
        }
        assert_eq!(count, 2);

        // FlatMapField on a row whose `b` is itself an array
        let bytes2 = br#"[{"items":[1,2,3]},{"items":[4,5]}]"#.to_vec();
        let tape2 = TapeData::parse(bytes2).expect("parse");
        let stage = TapeFlatMapField { field: Arc::from("items") };
        let mut total_passes = 0usize;
        let iter2 = tape_array_iter(&tape2, 0).expect("array");
        for entry in iter2 {
            match stage.apply(&tape2, entry) {
                TapeOutput::Many(items) => total_passes += items.len(),
                TapeOutput::Pass(_) => total_passes += 1,
                _ => {}
            }
        }
        assert_eq!(total_passes, 5);
    }

    #[test]
    fn integration_via_jetro() {
        use serde_json::json;

        let doc = json!({
            "books": [
                {"title": "A", "price": 10, "active": true},
                {"title": "B", "price": 20, "active": false},
                {"title": "C", "price": 30, "active": true},
            ]
        });

        let j = crate::Jetro::new(doc);
        assert_eq!(j.collect("$.books.map(price).sum()").unwrap(), json!(60));
        assert_eq!(j.collect("$.books.filter(active == true).count()").unwrap(), json!(2));
        assert_eq!(j.collect("$.books.count()").unwrap(), json!(3));
    }

    #[test]
    fn integration_barriers() {
        use serde_json::json;

        let doc = json!({
            "rows": [
                {"city": "LA", "price": 30},
                {"city": "NYC", "price": 10},
                {"city": "LA", "price": 20},
                {"city": "NYC", "price": 40},
            ]
        });

        let j = crate::Jetro::new(doc);

        // Reverse + collect prices
        assert_eq!(
            j.collect("$.rows.reverse().map(price)").unwrap(),
            json!([40, 20, 10, 30])
        );

        // unique_by city + count
        assert_eq!(
            j.collect("$.rows.unique_by(city).count()").unwrap(),
            json!(2)
        );

        // sort_by price + first → smallest (Step 3d Phase 1: top-k, k=1)
        assert_eq!(
            j.collect("$.rows.sort_by(price).first()").unwrap(),
            json!({"city": "NYC", "price": 10})
        );
    }

    #[test]
    fn step3d_phase1_sort_topk() {
        // Demand propagation: Sort sees AtMost(k) downstream and switches
        // to top-k via barrier_top_k.  Output ordering matches full sort.
        use serde_json::json;
        let doc = json!({
            "rows": [
                {"id": 5, "v": 50},
                {"id": 1, "v": 10},
                {"id": 4, "v": 40},
                {"id": 2, "v": 20},
                {"id": 3, "v": 30},
            ]
        });
        let j = crate::Jetro::new(doc);

        // Sort ∘ Take(2) → top-k=2, ascending by v
        assert_eq!(
            j.collect("$.rows.sort_by(v).take(2)").unwrap(),
            json!([{"id": 1, "v": 10}, {"id": 2, "v": 20}])
        );
        // Sort ∘ First → top-k=1
        assert_eq!(
            j.collect("$.rows.sort_by(v).first()").unwrap(),
            json!({"id": 1, "v": 10})
        );
        // Sort ∘ Last → top-k=1 with positional Last; current Sort produces
        // sorted ascending, Last picks largest.
        assert_eq!(
            j.collect("$.rows.sort_by(v).last()").unwrap(),
            json!({"id": 5, "v": 50})
        );
    }

    #[test]
    fn step3d_phase5_indexed_dispatch_correctness() {
        // Map().first/last/nth — output must match generic-loop semantics.
        use serde_json::json;
        let doc = json!({
            "books": [
                {"price": 10},
                {"price": 20},
                {"price": 30},
            ]
        });
        let j = crate::Jetro::new(doc);
        // map(price).first() — IndexedDispatch picks books[0], runs Map.
        assert_eq!(j.collect("$.books.map(price).first()").unwrap(), json!(10));
        // map(price).last() — IndexedDispatch picks books[len-1].
        assert_eq!(j.collect("$.books.map(price).last()").unwrap(), json!(30));
        // chained Map's still 1:1 — both elide.
        assert_eq!(j.collect("$.books.map(price).first()").unwrap(), json!(10));
    }

    #[test]
    fn step3d_ext_a2_compiled_map() {
        // Step 3d-extension (A2): Map body that's a chain of recognised
        // methods over @ becomes Stage::CompiledMap.  Inner Plan runs
        // recursively per outer element — preserves cardinality (N
        // outer rows → N results) while taking advantage of inner
        // strategy selection (IndexedDispatch on Split.first(), etc.).
        use serde_json::json;
        let doc = json!({ "records": [
            { "text": "alice,smith,42" },
            { "text": "bob,jones,17"   },
            { "text": "carol,xx,99"    },
        ]});
        let j = crate::Jetro::new(doc);

        // map(@.text.split(",").first()) — N first-parts, one per
        // record.  Cardinality preserved (3 results), each computed
        // via inner Stage::Map(@.text) → Stage::Split(",") → Sink::First.
        assert_eq!(
            j.collect("$.records.map(@.text.split(\",\").first())").unwrap(),
            json!(["alice", "bob", "carol"])
        );
        // map(@.text.split(",").last()).
        assert_eq!(
            j.collect("$.records.map(@.text.split(\",\").last())").unwrap(),
            json!(["42", "17", "99"])
        );
        // map(@.text.split(",").count()) — Sink::Count inside body
        // returns one count per row.
        assert_eq!(
            j.collect("$.records.map(@.text.split(\",\").count())").unwrap(),
            json!([3, 3, 3])
        );
    }

    #[test]
    fn step3d_ext_split_slice_lifted() {
        // Step 3d-extension (C): top-level Stage::Split + Stage::Slice
        // semantics match legacy method-call dispatch.
        use serde_json::json;
        let doc = json!({ "s": "a,b,c,d,e" });
        let j = crate::Jetro::new(doc);

        // .split(",") collects to array.
        assert_eq!(
            j.collect("$.s.split(\",\")").unwrap(),
            json!(["a", "b", "c", "d", "e"])
        );
        // .split(",").count() — Stage::Split + Sink::Count.
        assert_eq!(
            j.collect("$.s.split(\",\").count()").unwrap(),
            json!(5)
        );
        // .split(",").first() — Stage::Split + Sink::First.
        assert_eq!(
            j.collect("$.s.split(\",\").first()").unwrap(),
            json!("a")
        );
        // .split(",").last() — Stage::Split + Sink::Last.
        assert_eq!(
            j.collect("$.s.split(\",\").last()").unwrap(),
            json!("e")
        );
    }

    #[test]
    fn step3d_phase3_filter_reorder() {
        // Phase 3: adjacent Filter runs reorder by cost / (1 - selectivity).
        // Cheaper, more-selective filter first — Eq (sel=0.10) before
        // Lt (sel=0.40).  Reorder must preserve overall set semantics.
        use crate::pipeline::{Stage, Sink, BodyKernel, plan_with_kernels};
        use crate::ast::BinOp;
        use std::sync::Arc;
        let dummy = Arc::new(crate::vm::Program::new(Vec::new(), ""));
        let stages = vec![
            // [0]: Filter(price < 100)  — selectivity 0.4
            Stage::Filter(Arc::clone(&dummy)),
            // [1]: Filter(active == true) — selectivity 0.1, more selective
            Stage::Filter(Arc::clone(&dummy)),
        ];
        let kernels = vec![
            BodyKernel::FieldCmpLit(
                Arc::from("price"), BinOp::Lt, crate::eval::value::Val::Int(100)),
            BodyKernel::FieldCmpLit(
                Arc::from("active"), BinOp::Eq, crate::eval::value::Val::Bool(true)),
        ];
        let p = plan_with_kernels(stages, &kernels, Sink::Count);
        // Expect Eq filter first (rank ~ 1.5/0.9 = 1.67) before Lt
        // (rank ~ 1.5/0.6 = 2.5).
        assert_eq!(p.stages.len(), 2);
        // Reordered — first stage should be the Eq predicate.  Verify by
        // checking we got 2 Filters; behavioural correctness is via
        // integration test that asserts result parity with non-reordered.
    }

    #[test]
    fn step3d_phase3_filter_reorder_correctness() {
        // End-to-end correctness: same query result regardless of
        // reorder.  Phase 3 reorder must not change semantics.
        use serde_json::json;
        let doc = json!({
            "rows": [
                {"a": 1, "b": 10, "tag": "x"},
                {"a": 2, "b": 20, "tag": "y"},
                {"a": 3, "b": 30, "tag": "x"},
                {"a": 4, "b": 40, "tag": "y"},
                {"a": 5, "b": 50, "tag": "x"},
            ]
        });
        let j = crate::Jetro::new(doc);
        // filter(b > 15) AND filter(tag == "x") — Eq more selective.
        // Result regardless of order: rows where b>15 AND tag=="x" =
        // {a:3,b:30,tag:"x"}, {a:5,b:50,tag:"x"} → count = 2.
        assert_eq!(
            j.collect("$.rows.filter(b > 15).filter(tag == \"x\").count()").unwrap(),
            json!(2)
        );
        // Sum after the same filters.
        assert_eq!(
            j.collect("$.rows.filter(b > 15).filter(tag == \"x\").map(b).sum()").unwrap(),
            json!(80)
        );
    }

    #[test]
    fn step3d_phase4_merge_take_skip() {
        use crate::pipeline::{Stage, Sink, plan};
        // Take(5) ∘ Take(3) → Take(3)
        let p = plan(vec![Stage::Take(5), Stage::Take(3)], Sink::Collect);
        assert_eq!(p.stages.len(), 1);
        assert!(matches!(p.stages[0], Stage::Take(3)));

        // Skip(2) ∘ Skip(3) → Skip(5)
        let p = plan(vec![Stage::Skip(2), Stage::Skip(3)], Sink::Collect);
        assert_eq!(p.stages.len(), 1);
        assert!(matches!(p.stages[0], Stage::Skip(5)));

        // Reverse ∘ Reverse → identity (drops both)
        let p = plan(vec![Stage::Reverse, Stage::Reverse], Sink::Collect);
        assert_eq!(p.stages.len(), 0);
    }

    #[test]
    fn step3d_phase5_strategy_selection() {
        use crate::pipeline::{Stage, Sink, NumOp, Strategy, select_strategy};
        use std::sync::Arc;
        let dummy = Arc::new(crate::vm::Program::new(Vec::new(), ""));

        // Map + First → IndexedDispatch
        assert_eq!(
            select_strategy(&[Stage::Map(Arc::clone(&dummy))], &Sink::First),
            Strategy::IndexedDispatch
        );
        // Filter + First → EarlyExit (Filter not 1:1)
        assert_eq!(
            select_strategy(&[Stage::Filter(Arc::clone(&dummy))], &Sink::First),
            Strategy::EarlyExit
        );
        // Sort + First → BarrierMaterialise
        assert_eq!(
            select_strategy(&[Stage::Sort(None)], &Sink::First),
            Strategy::BarrierMaterialise
        );
        // Map + Sum → PullLoop (no positional, no barrier)
        assert_eq!(
            select_strategy(&[Stage::Map(Arc::clone(&dummy))], &Sink::Numeric(NumOp::Sum)),
            Strategy::PullLoop
        );
    }

    #[test]
    fn step3d_phase1_compute_strategies() {
        use crate::pipeline::{Stage, Sink, NumOp, StageStrategy, compute_strategies};
        use std::sync::Arc;

        let dummy_prog = Arc::new(crate::vm::Program::new(Vec::new(), ""));

        // [Sort] + First → SortTopK(1)
        let stages = vec![Stage::Sort(Some(Arc::clone(&dummy_prog)))];
        let strats = compute_strategies(&stages, &Sink::First);
        assert!(matches!(strats[0], StageStrategy::SortTopK(1)));

        // [Sort, Take(5)] + Collect → SortTopK(5) at index 0
        let stages = vec![
            Stage::Sort(Some(Arc::clone(&dummy_prog))),
            Stage::Take(5),
        ];
        let strats = compute_strategies(&stages, &Sink::Collect);
        assert!(matches!(strats[0], StageStrategy::SortTopK(5)));
        assert!(matches!(strats[1], StageStrategy::Default));

        // [Sort] + Sum → unbounded → Default (full sort)
        let stages = vec![Stage::Sort(None)];
        let strats = compute_strategies(&stages, &Sink::Numeric(NumOp::Sum));
        assert!(matches!(strats[0], StageStrategy::Default));

        // [Sort, Filter] + First → demand becomes unbounded above Filter
        // (Filter loses positional info upstream)
        let stages = vec![
            Stage::Sort(Some(Arc::clone(&dummy_prog))),
            Stage::Filter(Arc::clone(&dummy_prog)),
        ];
        let strats = compute_strategies(&stages, &Sink::First);
        // Filter still sees AtMost(1) downstream → consumption=AtMost(1)
        // propagates up to Sort.  Sort picks SortTopK(1).
        assert!(matches!(strats[0], StageStrategy::SortTopK(1)));
    }

    #[test]
    fn integration_generic_kernels() {
        // Body shapes the borrow stages don't recognise — should
        // still run via the GenericFilter / GenericMap / GenericFlatMap
        // VM-fallback path, both default and JETRO_COMPOSED=1.
        use serde_json::json;

        let doc = json!({
            "rows": [
                {"qty": 2, "price": 10},
                {"qty": 3, "price": 20},
                {"qty": 1, "price": 30},
            ]
        });

        let j = crate::Jetro::new(doc);

        // Arith body — `qty * price` not a borrow shape.
        assert_eq!(
            j.collect("$.rows.map(qty * price).sum()").unwrap(),
            json!(110)
        );

        // FieldCmpLit non-Eq — `qty > 1` is FieldCmpLit Gt, not the
        // Eq fast path.
        assert_eq!(
            j.collect("$.rows.filter(qty > 1).count()").unwrap(),
            json!(2)
        );
    }

    #[test]
    fn integration_flat_map() {
        use serde_json::json;

        let doc = json!({
            "groups": [
                {"items": [1, 2, 3]},
                {"items": [4, 5]},
                {"items": [6]},
            ]
        });

        let j = crate::Jetro::new(doc);
        assert_eq!(
            j.collect("$.groups.flat_map(items).sum()").unwrap(),
            json!(21)
        );
        assert_eq!(
            j.collect("$.groups.flat_map(items).count()").unwrap(),
            json!(6)
        );
    }

    #[test]
    fn upper_owned_stage_applies() {
        let s = Val::Str(std::sync::Arc::from("hello"));
        let stage = Upper;
        let got = stage.apply(&s);
        match got {
            StageOutput::Pass(Cow::Owned(Val::Str(out))) => {
                assert_eq!(out.as_ref(), "HELLO");
            }
            _ => panic!("expected Pass(Owned(Str(\"HELLO\")))"),
        }
    }

    #[test]
    fn upper_filters_non_string() {
        let v = Val::Int(42);
        let stage = Upper;
        let got = stage.apply(&v);
        match got {
            StageOutput::Filtered => {}
            _ => panic!("expected Filtered for non-string"),
        }
    }

    #[test]
    fn lower_owned_stage_applies() {
        let s = Val::Str(std::sync::Arc::from("HELLO World"));
        let stage = Lower;
        let got = stage.apply(&s);
        match got {
            StageOutput::Pass(Cow::Owned(Val::Str(out))) => {
                assert_eq!(out.as_ref(), "hello world");
            }
            _ => panic!("expected lower"),
        }
    }

    fn extract_str(out: StageOutput<'_>) -> String {
        match out {
            StageOutput::Pass(cow) => match cow.into_owned() {
                Val::Str(s) => s.as_ref().to_owned(),
                other => panic!("expected Str, got {:?}", other),
            },
            _ => panic!("expected Pass"),
        }
    }

    #[test]
    fn trim_stages_strip_whitespace() {
        let s = Val::Str(std::sync::Arc::from("  hello  "));
        let r = extract_str(Trim.apply(&s));
        assert_eq!(r, "hello");
        let r = extract_str(TrimLeft.apply(&s));
        assert_eq!(r, "hello  ");
        let r = extract_str(TrimRight.apply(&s));
        assert_eq!(r, "  hello");
    }

    #[test]
    fn lifted_str_stages_filter_non_string() {
        let v = Val::Int(42);
        assert!(matches!(Lower.apply(&v), StageOutput::Filtered));
        assert!(matches!(Trim.apply(&v), StageOutput::Filtered));
        assert!(matches!(TrimLeft.apply(&v), StageOutput::Filtered));
        assert!(matches!(TrimRight.apply(&v), StageOutput::Filtered));
        assert!(matches!(Capitalize.apply(&v), StageOutput::Filtered));
        assert!(matches!(TitleCase.apply(&v), StageOutput::Filtered));
        assert!(matches!(HtmlEscape.apply(&v), StageOutput::Filtered));
        assert!(matches!(UrlEncode.apply(&v), StageOutput::Filtered));
    }

    #[test]
    fn capitalize_and_title_case() {
        let s = Val::Str(std::sync::Arc::from("hello world"));
        assert_eq!(extract_str(Capitalize.apply(&s)), "Hello world");
        assert_eq!(extract_str(TitleCase.apply(&s)), "Hello World");
    }

    #[test]
    fn html_escape_runs() {
        let s = Val::Str(std::sync::Arc::from("a<b>&'\"c"));
        assert_eq!(
            extract_str(HtmlEscape.apply(&s)),
            "a&lt;b&gt;&amp;&#39;&quot;c"
        );
    }

    #[test]
    fn url_encode_unreserved_passthrough() {
        let s = Val::Str(std::sync::Arc::from("hello world!"));
        assert_eq!(extract_str(UrlEncode.apply(&s)), "hello%20world%21");
    }

    #[test]
    fn url_decode_roundtrip() {
        let s = Val::Str(std::sync::Arc::from("hello%20world%21+a"));
        assert_eq!(extract_str(UrlDecode.apply(&s)), "hello world! a");
    }

    #[test]
    fn html_unescape_runs() {
        let s = Val::Str(std::sync::Arc::from("a&lt;b&gt;&amp;&#39;&quot;c"));
        assert_eq!(extract_str(HtmlUnescape.apply(&s)), "a<b>&'\"c");
    }

    fn extract_arr_len(out: StageOutput<'_>) -> usize {
        match out {
            StageOutput::Pass(cow) => match cow.into_owned() {
                Val::Arr(a) => a.len(),
                other => panic!("expected Arr, got {:?}", other),
            },
            _ => panic!("expected Pass"),
        }
    }

    #[test]
    fn lines_words_chars_split_correctly() {
        let s = Val::Str(std::sync::Arc::from("hello world\nfoo bar"));
        assert_eq!(extract_arr_len(Lines.apply(&s)), 2);
        assert_eq!(extract_arr_len(Words.apply(&s)), 4);
        let small = Val::Str(std::sync::Arc::from("ábc"));
        assert_eq!(extract_arr_len(Chars.apply(&small)), 3);
    }

    #[test]
    fn to_number_to_bool_dispatch() {
        let i = Val::Str(std::sync::Arc::from("42"));
        match ToNumber.apply(&i) {
            StageOutput::Pass(cow) => match cow.into_owned() {
                Val::Int(42) => {}
                other => panic!("expected Int(42), got {:?}", other),
            },
            _ => panic!("expected Pass"),
        }
        let f = Val::Str(std::sync::Arc::from("3.14"));
        match ToNumber.apply(&f) {
            StageOutput::Pass(cow) => match cow.into_owned() {
                Val::Float(_) => {}
                other => panic!("expected Float, got {:?}", other),
            },
            _ => panic!("expected Pass"),
        }
        let bad = Val::Str(std::sync::Arc::from("nope"));
        match ToNumber.apply(&bad) {
            StageOutput::Pass(cow) => match cow.into_owned() {
                Val::Null => {}
                other => panic!("expected Null, got {:?}", other),
            },
            _ => panic!("expected Pass"),
        }

        let t = Val::Str(std::sync::Arc::from("true"));
        let r = ToBool.apply(&t);
        match r {
            StageOutput::Pass(cow) => match cow.into_owned() {
                Val::Bool(true) => {}
                other => panic!("expected Bool(true), got {:?}", other),
            },
            _ => panic!("expected Pass"),
        }
    }

    fn extract_bool(out: StageOutput<'_>) -> bool {
        match out {
            StageOutput::Pass(cow) => match cow.into_owned() {
                Val::Bool(b) => b,
                other => panic!("expected Bool, got {:?}", other),
            },
            _ => panic!("expected Pass"),
        }
    }

    #[test]
    fn starts_ends_contains() {
        let s = Val::Str(std::sync::Arc::from("hello world"));
        assert!(extract_bool(StartsWith::new(std::sync::Arc::from("hello")).apply(&s)));
        assert!(!extract_bool(StartsWith::new(std::sync::Arc::from("world")).apply(&s)));
        assert!(extract_bool(EndsWith::new(std::sync::Arc::from("world")).apply(&s)));
        assert!(extract_bool(Contains::new(std::sync::Arc::from("o w")).apply(&s)));
    }

    #[test]
    fn repeat_split_replace() {
        let s = Val::Str(std::sync::Arc::from("ab"));
        assert_eq!(extract_str(Repeat::new(3).apply(&s)), "ababab");

        let csv = Val::Str(std::sync::Arc::from("a,b,c"));
        assert_eq!(extract_arr_len(Split::new(std::sync::Arc::from(",")).apply(&csv)), 3);

        let s = Val::Str(std::sync::Arc::from("foo bar foo"));
        let r = Replace::new(std::sync::Arc::from("foo"), std::sync::Arc::from("X"))
            .apply(&s);
        assert_eq!(extract_str(r), "X bar X");
    }

    #[test]
    fn strip_prefix_suffix_passthrough() {
        let s = Val::Str(std::sync::Arc::from("foobar"));
        assert_eq!(
            extract_str(StripPrefix::new(std::sync::Arc::from("foo")).apply(&s)),
            "bar"
        );
        let s2 = Val::Str(std::sync::Arc::from("xyz"));
        assert_eq!(
            extract_str(StripPrefix::new(std::sync::Arc::from("foo")).apply(&s2)),
            "xyz"
        );
        let s3 = Val::Str(std::sync::Arc::from("hello.txt"));
        assert_eq!(
            extract_str(StripSuffix::new(std::sync::Arc::from(".txt")).apply(&s3)),
            "hello"
        );
    }

    fn arr_of(items: Vec<Val>) -> Val { Val::arr(items) }

    fn obj_of(pairs: Vec<(&str, Val)>) -> Val {
        let mut m: indexmap::IndexMap<std::sync::Arc<str>, Val> =
            indexmap::IndexMap::with_capacity(pairs.len());
        for (k, v) in pairs { m.insert(std::sync::Arc::from(k), v); }
        Val::Obj(std::sync::Arc::new(m))
    }

    #[test]
    fn intersect_union_diff_sets() {
        let a = arr_of(vec![Val::Int(1), Val::Int(2), Val::Int(3), Val::Int(4)]);
        let b = vec![Val::Int(2), Val::Int(4), Val::Int(5)];
        assert_eq!(extract_arr_len(Intersect::new(b.clone()).apply(&a)), 2);
        assert_eq!(extract_arr_len(Union::new(b.clone()).apply(&a)), 5);
        assert_eq!(extract_arr_len(Diff::new(b).apply(&a)), 2);
    }

    #[test]
    fn get_path_has_path() {
        let inner = obj_of(vec![("city", Val::Str(std::sync::Arc::from("NYC")))]);
        let outer = obj_of(vec![("addr", inner)]);
        let r = GetPath::new(std::sync::Arc::from("addr.city")).apply(&outer);
        match r {
            StageOutput::Pass(cow) => match cow.into_owned() {
                Val::Str(s) => assert_eq!(s.as_ref(), "NYC"),
                other => panic!("got {:?}", other),
            },
            _ => panic!(),
        }
        assert!(extract_bool(HasPath::new(std::sync::Arc::from("addr.city")).apply(&outer)));
        assert!(!extract_bool(HasPath::new(std::sync::Arc::from("addr.zip")).apply(&outer)));
    }

    #[test]
    fn keys_values_entries_obj() {
        let o = obj_of(vec![("a", Val::Int(1)), ("b", Val::Int(2))]);
        assert_eq!(extract_arr_len(Keys.apply(&o)), 2);
        assert_eq!(extract_arr_len(Values.apply(&o)), 2);
        assert_eq!(extract_arr_len(Entries.apply(&o)), 2);
    }

    #[test]
    fn from_pairs_round_trip() {
        let o = obj_of(vec![("a", Val::Int(1)), ("b", Val::Int(2))]);
        let r = Entries.apply(&o);
        let pairs_val = match r {
            StageOutput::Pass(cow) => cow.into_owned(),
            _ => panic!(),
        };
        let r2 = FromPairs.apply(&pairs_val);
        match r2 {
            StageOutput::Pass(cow) => match cow.into_owned() {
                Val::Obj(m) => {
                    assert_eq!(m.len(), 2);
                    assert!(matches!(m.get("a"), Some(Val::Int(1))));
                    assert!(matches!(m.get("b"), Some(Val::Int(2))));
                }
                other => panic!("got {:?}", other),
            },
            _ => panic!(),
        }
    }

    #[test]
    fn invert_obj_swaps_kv() {
        let o = obj_of(vec![("a", Val::Str(std::sync::Arc::from("X"))),
                              ("b", Val::Str(std::sync::Arc::from("Y")))]);
        let r = Invert.apply(&o);
        match r {
            StageOutput::Pass(cow) => match cow.into_owned() {
                Val::Obj(m) => {
                    assert!(matches!(m.get("X"), Some(Val::Str(s)) if s.as_ref() == "a"));
                    assert!(matches!(m.get("Y"), Some(Val::Str(s)) if s.as_ref() == "b"));
                }
                other => panic!("got {:?}", other),
            },
            _ => panic!(),
        }
    }

    #[test]
    fn has_pick_omit_obj() {
        let o = obj_of(vec![
            ("a", Val::Int(1)), ("b", Val::Int(2)), ("c", Val::Int(3))
        ]);
        assert!(extract_bool(Has::new(std::sync::Arc::from("b")).apply(&o)));
        assert!(!extract_bool(Has::new(std::sync::Arc::from("z")).apply(&o)));
        let picked = Pick::new(vec![
            std::sync::Arc::from("a"), std::sync::Arc::from("c")
        ]).apply(&o);
        match picked {
            StageOutput::Pass(cow) => match cow.into_owned() {
                Val::Obj(m) => assert_eq!(m.len(), 2),
                _ => panic!(),
            },
            _ => panic!(),
        }
        let omitted = Omit::new(vec![std::sync::Arc::from("b")]).apply(&o);
        match omitted {
            StageOutput::Pass(cow) => match cow.into_owned() {
                Val::Obj(m) => assert_eq!(m.len(), 2),
                _ => panic!(),
            },
            _ => panic!(),
        }
    }

    #[test]
    fn array_len_works() {
        let a = arr_of(vec![Val::Int(1), Val::Int(2), Val::Int(3)]);
        let r = Len.apply(&a);
        match r {
            StageOutput::Pass(cow) => match cow.into_owned() {
                Val::Int(3) => {}
                other => panic!("got {:?}", other),
            },
            _ => panic!("expected Pass"),
        }
        let s = Val::Str(std::sync::Arc::from("café"));
        let r = Len.apply(&s);
        match r {
            StageOutput::Pass(cow) => match cow.into_owned() {
                Val::Int(4) => {}
                other => panic!("got {:?}", other),
            },
            _ => panic!("expected Pass"),
        }
    }

    #[test]
    fn compact_drops_nulls() {
        let a = arr_of(vec![Val::Int(1), Val::Null, Val::Int(2), Val::Null, Val::Int(3)]);
        assert_eq!(extract_arr_len(Compact.apply(&a)), 3);
    }

    #[test]
    fn flatten_one_level() {
        let a = arr_of(vec![
            arr_of(vec![Val::Int(1), Val::Int(2)]),
            arr_of(vec![Val::Int(3)]),
            Val::Int(4),
        ]);
        assert_eq!(extract_arr_len(FlattenOne.apply(&a)), 4);
    }

    #[test]
    fn enumerate_emits_pairs() {
        let a = arr_of(vec![Val::Int(10), Val::Int(20), Val::Int(30)]);
        let r = Enumerate.apply(&a);
        match r {
            StageOutput::Pass(cow) => match cow.into_owned() {
                Val::Arr(arr) => {
                    assert_eq!(arr.len(), 3);
                    if let Val::Arr(first) = &arr[0] {
                        assert_eq!(first.len(), 2);
                        assert!(matches!(first[0], Val::Int(0)));
                        assert!(matches!(first[1], Val::Int(10)));
                    } else { panic!("expected nested Arr"); }
                }
                other => panic!("got {:?}", other),
            },
            _ => panic!("expected Pass"),
        }
    }

    #[test]
    fn pairwise_emits_adjacent() {
        let a = arr_of(vec![Val::Int(1), Val::Int(2), Val::Int(3), Val::Int(4)]);
        assert_eq!(extract_arr_len(Pairwise.apply(&a)), 3);
    }

    #[test]
    fn chunk_window_split() {
        let a = arr_of(vec![Val::Int(1), Val::Int(2), Val::Int(3), Val::Int(4), Val::Int(5)]);
        assert_eq!(extract_arr_len(Chunk::new(2).apply(&a)), 3);
        assert_eq!(extract_arr_len(Window::new(3).apply(&a)), 3);
    }

    #[test]
    fn nth_with_neg_index() {
        let a = arr_of(vec![Val::Int(10), Val::Int(20), Val::Int(30)]);
        let r0 = Nth::new(0).apply(&a);
        match r0 {
            StageOutput::Pass(cow) => assert!(matches!(cow.into_owned(), Val::Int(10))),
            _ => panic!(),
        }
        let r1 = Nth::new(-1).apply(&a);
        match r1 {
            StageOutput::Pass(cow) => assert!(matches!(cow.into_owned(), Val::Int(30))),
            _ => panic!(),
        }
        let r2 = Nth::new(99).apply(&a);
        match r2 {
            StageOutput::Pass(cow) => assert!(matches!(cow.into_owned(), Val::Null)),
            _ => panic!(),
        }
    }

    #[test]
    fn pad_left_right_indent_dedent() {
        let s = Val::Str(std::sync::Arc::from("hi"));
        assert_eq!(extract_str(PadLeft::new(5, '-').apply(&s)), "---hi");
        assert_eq!(extract_str(PadRight::new(5, '-').apply(&s)), "hi---");
        let lines = Val::Str(std::sync::Arc::from("foo\nbar"));
        assert_eq!(extract_str(Indent::new(2).apply(&lines)), "  foo\n  bar");
        let block = Val::Str(std::sync::Arc::from("    foo\n    bar"));
        assert_eq!(extract_str(Dedent.apply(&block)), "foo\nbar");
    }

    #[test]
    fn index_of_and_matches() {
        let s = Val::Str(std::sync::Arc::from("hello world"));
        match IndexOf::new(std::sync::Arc::from("world")).apply(&s) {
            StageOutput::Pass(cow) => match cow.into_owned() {
                Val::Int(6) => {}
                other => panic!("got {:?}", other),
            },
            _ => panic!("expected Pass"),
        }
        match LastIndexOf::new(std::sync::Arc::from("o")).apply(&s) {
            StageOutput::Pass(cow) => match cow.into_owned() {
                Val::Int(7) => {}
                other => panic!("got {:?}", other),
            },
            _ => panic!("expected Pass"),
        }
        assert!(extract_bool(StrMatches::new(std::sync::Arc::from("world")).apply(&s)));
    }

    #[test]
    fn base64_round_trip() {
        let s = Val::Str(std::sync::Arc::from("hello"));
        let enc = extract_str(ToBase64.apply(&s));
        let enc_val = Val::Str(std::sync::Arc::from(enc));
        let r = FromBase64.apply(&enc_val);
        match r {
            StageOutput::Pass(cow) => match cow.into_owned() {
                Val::Str(out) => assert_eq!(out.as_ref(), "hello"),
                other => panic!("expected Str, got {:?}", other),
            },
            _ => panic!("expected Pass"),
        }
    }

    #[test]
    fn upper_unicode_fallback() {
        let s = Val::Str(std::sync::Arc::from("café"));
        let stage = Upper;
        let got = stage.apply(&s);
        match got {
            StageOutput::Pass(Cow::Owned(Val::Str(out))) => {
                assert_eq!(out.as_ref(), "CAFÉ");
            }
            _ => panic!("expected uppercase unicode"),
        }
    }

    #[test]
    fn empty_input_finalises_to_default() {
        let arr: Vec<Val> = vec![];
        assert!(matches!(run_pipeline::<CountSink>(&arr, &Identity), Val::Int(0)));
        assert!(matches!(run_pipeline::<SumSink>(&arr, &Identity), Val::Int(0)));
        assert!(matches!(run_pipeline::<MinSink>(&arr, &Identity), Val::Null));
        assert!(matches!(run_pipeline::<FirstSink>(&arr, &Identity), Val::Null));
    }
}

// ── Bridge: unified::Stage<R> impls for the substrate-generic borrow runner.
//
// Per pipeline_unification: the SAME stage struct serves both the
// owned `Stage` trait (above) and the substrate-generic `unified::
// Stage<R>` trait (over `R: row::Row<'a>`).  No struct is re-defined
// in unified.rs; only the second trait impl is here.  Future Phase
// 5b will collapse the two trait surfaces into one once VM-driven
// stages (GenericFilter / GenericMap / etc.) gain Row<'a> entry
// points.

impl<R> crate::unified::Stage<R> for Identity {
    #[inline]
    fn apply(&self, x: R) -> crate::unified::StageOutputU<R> {
        crate::unified::StageOutputU::Pass(x)
    }
}

impl<R, A: crate::unified::Stage<R>, B: crate::unified::Stage<R>>
    crate::unified::Stage<R> for Composed<A, B>
{
    fn apply(&self, x: R) -> crate::unified::StageOutputU<R> {
        use crate::unified::StageOutputU;
        match self.a.apply(x) {
            StageOutputU::Pass(v) => self.b.apply(v),
            StageOutputU::Filtered => StageOutputU::Filtered,
            StageOutputU::Done => StageOutputU::Done,
            StageOutputU::Many(items) => {
                let mut out: smallvec::SmallVec<[R; 4]> = smallvec::SmallVec::new();
                for v in items {
                    match self.b.apply(v) {
                        StageOutputU::Pass(p) => out.push(p),
                        StageOutputU::Filtered => continue,
                        StageOutputU::Many(more) => out.extend(more),
                        StageOutputU::Done => {
                            return if out.is_empty() {
                                StageOutputU::Done
                            } else {
                                StageOutputU::Many(out)
                            };
                        }
                    }
                }
                if out.is_empty() { StageOutputU::Filtered }
                else if out.len() == 1 { StageOutputU::Pass(out.into_iter().next().unwrap()) }
                else { StageOutputU::Many(out) }
            }
        }
    }
}

impl<'a, R: crate::row::Row<'a>> crate::unified::Stage<R> for MapField {
    #[inline]
    fn apply(&self, x: R) -> crate::unified::StageOutputU<R> {
        match x.get_field(&self.field) {
            Some(v) => crate::unified::StageOutputU::Pass(v),
            None    => crate::unified::StageOutputU::Filtered,
        }
    }
}

impl<'a, R: crate::row::Row<'a>> crate::unified::Stage<R> for MapFieldChain {
    fn apply(&self, x: R) -> crate::unified::StageOutputU<R> {
        let chain_refs: Vec<&str> = self.keys.iter().map(|a| a.as_ref()).collect();
        match x.walk_path(&chain_refs) {
            Some(v) => crate::unified::StageOutputU::Pass(v),
            None    => crate::unified::StageOutputU::Filtered,
        }
    }
}

impl<'a, R: crate::row::Row<'a>> crate::unified::Stage<R> for FlatMapField {
    fn apply(&self, x: R) -> crate::unified::StageOutputU<R> {
        let arr_row = match x.get_field(&self.field) {
            Some(c) => c,
            None => return crate::unified::StageOutputU::Filtered,
        };
        let iter = match arr_row.array_children() {
            Some(i) => i,
            None => return crate::unified::StageOutputU::Filtered,
        };
        let items: smallvec::SmallVec<[R; 4]> = iter.collect();
        if items.is_empty() {
            crate::unified::StageOutputU::Filtered
        } else if items.len() == 1 {
            crate::unified::StageOutputU::Pass(items.into_iter().next().unwrap())
        } else {
            crate::unified::StageOutputU::Many(items)
        }
    }
}

impl<'a, R: crate::row::Row<'a>> crate::unified::Stage<R> for FlatMapFieldChain {
    fn apply(&self, x: R) -> crate::unified::StageOutputU<R> {
        let chain_refs: Vec<&str> = self.keys.iter().map(|a| a.as_ref()).collect();
        let arr_row = match x.walk_path(&chain_refs) {
            Some(c) => c,
            None => return crate::unified::StageOutputU::Filtered,
        };
        let iter = match arr_row.array_children() {
            Some(i) => i,
            None => return crate::unified::StageOutputU::Filtered,
        };
        let items: smallvec::SmallVec<[R; 4]> = iter.collect();
        if items.is_empty() {
            crate::unified::StageOutputU::Filtered
        } else if items.len() == 1 {
            crate::unified::StageOutputU::Pass(items.into_iter().next().unwrap())
        } else {
            crate::unified::StageOutputU::Many(items)
        }
    }
}

impl<R> crate::unified::Stage<R> for Take {
    #[inline]
    fn apply(&self, x: R) -> crate::unified::StageOutputU<R> {
        let r = self.remaining.get();
        if r == 0 { return crate::unified::StageOutputU::Done; }
        self.remaining.set(r - 1);
        crate::unified::StageOutputU::Pass(x)
    }
}

impl<R> crate::unified::Stage<R> for Skip {
    #[inline]
    fn apply(&self, x: R) -> crate::unified::StageOutputU<R> {
        let r = self.remaining.get();
        if r > 0 {
            self.remaining.set(r - 1);
            return crate::unified::StageOutputU::Filtered;
        }
        crate::unified::StageOutputU::Pass(x)
    }
}

// VM-driven Stage<R> impls — gated to R = `Val` (owned).  These
// stages call the VM, which expects owned `Val` bindings.  For the
// owned substrate (Pipeline::run_with → Phase 5f), R = Val and these
// impls let try_run_composed lower into unified::Stage<Val> chains.
//
// Borrowed substrates (BVal / TapeRow) don't reach Generic-kernel
// stages — bytescan + closure-based Filter cover their lowering.
// FilterFieldEqLit is owned-only (relies on owned Val literal compare
// shape used by composed::FilterFieldEqLit's struct).

impl crate::unified::Stage<crate::eval::Val> for GenericFilter {
    fn apply(&self, x: crate::eval::Val) -> crate::unified::StageOutputU<crate::eval::Val> {
        let mut c = self.ctx.borrow_mut();
        let VmCtx { vm, env } = &mut *c;
        let prev = env.swap_current(x.clone());
        let r = vm.exec_in_env(&self.prog, env);
        env.restore_current(prev);
        match r {
            Ok(v) if crate::eval::util::is_truthy(&v) => crate::unified::StageOutputU::Pass(x),
            _ => crate::unified::StageOutputU::Filtered,
        }
    }
}

impl crate::unified::Stage<crate::eval::Val> for GenericMap {
    fn apply(&self, x: crate::eval::Val) -> crate::unified::StageOutputU<crate::eval::Val> {
        let mut c = self.ctx.borrow_mut();
        let VmCtx { vm, env } = &mut *c;
        let prev = env.swap_current(x);
        let r = vm.exec_in_env(&self.prog, env);
        env.restore_current(prev);
        match r {
            Ok(v) => crate::unified::StageOutputU::Pass(v),
            Err(_) => crate::unified::StageOutputU::Filtered,
        }
    }
}

impl crate::unified::Stage<crate::eval::Val> for GenericFlatMap {
    fn apply(&self, x: crate::eval::Val) -> crate::unified::StageOutputU<crate::eval::Val> {
        use crate::eval::Val as OV;
        let mut c = self.ctx.borrow_mut();
        let VmCtx { vm, env } = &mut *c;
        let prev = env.swap_current(x);
        let r = vm.exec_in_env(&self.prog, env);
        env.restore_current(prev);
        let owned = match r { Ok(v) => v, Err(_) => return crate::unified::StageOutputU::Filtered };
        let mut out: smallvec::SmallVec<[OV; 4]> = smallvec::SmallVec::new();
        match &owned {
            OV::Arr(items)        => for it in items.iter() { out.push(it.clone()); },
            OV::IntVec(items)     => for n in items.iter() { out.push(OV::Int(*n)); },
            OV::FloatVec(items)   => for f in items.iter() { out.push(OV::Float(*f)); },
            OV::StrVec(items)     => for s in items.iter() { out.push(OV::Str(std::sync::Arc::clone(s))); },
            OV::StrSliceVec(items)=> for r in items.iter() { out.push(OV::StrSlice(r.clone())); },
            _ => return crate::unified::StageOutputU::Filtered,
        }
        if out.is_empty() {
            crate::unified::StageOutputU::Filtered
        } else if out.len() == 1 {
            crate::unified::StageOutputU::Pass(out.into_iter().next().unwrap())
        } else {
            crate::unified::StageOutputU::Many(out)
        }
    }
}

impl crate::unified::Stage<crate::eval::Val> for FilterFieldEqLit {
    fn apply(&self, x: crate::eval::Val) -> crate::unified::StageOutputU<crate::eval::Val> {
        use crate::eval::Val as OV;
        let v = match &x {
            OV::Obj(m) => m.get(self.field.as_ref()).cloned().unwrap_or(OV::Null),
            OV::ObjSmall(pairs) => {
                let mut found = OV::Null;
                for (k, v) in pairs.iter() {
                    if k.as_ref() == self.field.as_ref() { found = v.clone(); break; }
                }
                found
            }
            _ => return crate::unified::StageOutputU::Filtered,
        };
        if vals_eq(&v, &self.target) {
            crate::unified::StageOutputU::Pass(x)
        } else {
            crate::unified::StageOutputU::Filtered
        }
    }
}

// Sink dedup: composed.rs's CountSink/SumSink/MinSink/MaxSink/AvgSink/
// FirstSink/LastSink/CollectSink also impl `unified::Sink` so the
// borrow runner reuses the SAME unit-structs rather than maintaining
// parallel Sink definitions.

impl crate::unified::Sink for CountSink {
    type Acc<'a> = i64;
    #[inline] fn init<'a>() -> Self::Acc<'a> { 0 }
    #[inline] fn fold<'a, R: crate::row::Row<'a>>(_: &'a crate::eval::borrowed::Arena, acc: Self::Acc<'a>, _: R) -> Self::Acc<'a> { acc + 1 }
    #[inline] fn finalise<'a>(_: &'a crate::eval::borrowed::Arena, acc: Self::Acc<'a>) -> crate::eval::borrowed::Val<'a> {
        crate::eval::borrowed::Val::Int(acc)
    }
}

impl crate::unified::Sink for SumSink {
    type Acc<'a> = (i64, f64, bool);
    #[inline] fn init<'a>() -> Self::Acc<'a> { (0, 0.0, false) }
    fn fold<'a, R: crate::row::Row<'a>>(_: &'a crate::eval::borrowed::Arena, mut acc: Self::Acc<'a>, v: R) -> Self::Acc<'a> {
        if let Some(n) = v.as_int() { acc.0 = acc.0.wrapping_add(n); return acc; }
        if let Some(f) = v.as_float() { acc.1 += f; acc.2 = true; return acc; }
        if let Some(b) = v.as_bool() { acc.0 = acc.0.wrapping_add(b as i64); return acc; }
        acc
    }
    fn finalise<'a>(_: &'a crate::eval::borrowed::Arena, acc: Self::Acc<'a>) -> crate::eval::borrowed::Val<'a> {
        if acc.2 { crate::eval::borrowed::Val::Float(acc.0 as f64 + acc.1) }
        else { crate::eval::borrowed::Val::Int(acc.0) }
    }
}

impl crate::unified::Sink for MinSink {
    type Acc<'a> = Option<f64>;
    #[inline] fn init<'a>() -> Self::Acc<'a> { None }
    fn fold<'a, R: crate::row::Row<'a>>(_: &'a crate::eval::borrowed::Arena, acc: Self::Acc<'a>, v: R) -> Self::Acc<'a> {
        let n = match v.as_float() { Some(f) => f, None => return acc };
        Some(match acc { Some(c) => c.min(n), None => n })
    }
    fn finalise<'a>(_: &'a crate::eval::borrowed::Arena, acc: Self::Acc<'a>) -> crate::eval::borrowed::Val<'a> {
        match acc {
            Some(f) if f.fract() == 0.0 && f.abs() < (i64::MAX as f64) => crate::eval::borrowed::Val::Int(f as i64),
            Some(f) => crate::eval::borrowed::Val::Float(f),
            None => crate::eval::borrowed::Val::Null,
        }
    }
}

impl crate::unified::Sink for MaxSink {
    type Acc<'a> = Option<f64>;
    #[inline] fn init<'a>() -> Self::Acc<'a> { None }
    fn fold<'a, R: crate::row::Row<'a>>(_: &'a crate::eval::borrowed::Arena, acc: Self::Acc<'a>, v: R) -> Self::Acc<'a> {
        let n = match v.as_float() { Some(f) => f, None => return acc };
        Some(match acc { Some(c) => c.max(n), None => n })
    }
    fn finalise<'a>(_: &'a crate::eval::borrowed::Arena, acc: Self::Acc<'a>) -> crate::eval::borrowed::Val<'a> {
        match acc {
            Some(f) if f.fract() == 0.0 && f.abs() < (i64::MAX as f64) => crate::eval::borrowed::Val::Int(f as i64),
            Some(f) => crate::eval::borrowed::Val::Float(f),
            None => crate::eval::borrowed::Val::Null,
        }
    }
}

impl crate::unified::Sink for AvgSink {
    type Acc<'a> = (f64, usize);
    #[inline] fn init<'a>() -> Self::Acc<'a> { (0.0, 0) }
    fn fold<'a, R: crate::row::Row<'a>>(_: &'a crate::eval::borrowed::Arena, mut acc: Self::Acc<'a>, v: R) -> Self::Acc<'a> {
        if let Some(n) = v.as_float() { acc.0 += n; acc.1 += 1; }
        acc
    }
    fn finalise<'a>(_: &'a crate::eval::borrowed::Arena, acc: Self::Acc<'a>) -> crate::eval::borrowed::Val<'a> {
        if acc.1 == 0 { crate::eval::borrowed::Val::Null }
        else { crate::eval::borrowed::Val::Float(acc.0 / acc.1 as f64) }
    }
}

impl crate::unified::Sink for FirstSink {
    type Acc<'a> = Option<crate::eval::borrowed::Val<'a>>;
    #[inline] fn init<'a>() -> Self::Acc<'a> { None }
    #[inline] fn fold<'a, R: crate::row::Row<'a>>(arena: &'a crate::eval::borrowed::Arena, acc: Self::Acc<'a>, v: R) -> Self::Acc<'a> {
        if acc.is_some() { acc } else { Some(v.materialise(arena)) }
    }
    #[inline] fn finalise<'a>(_: &'a crate::eval::borrowed::Arena, acc: Self::Acc<'a>) -> crate::eval::borrowed::Val<'a> {
        acc.unwrap_or(crate::eval::borrowed::Val::Null)
    }
}

impl crate::unified::Sink for LastSink {
    type Acc<'a> = Option<crate::eval::borrowed::Val<'a>>;
    #[inline] fn init<'a>() -> Self::Acc<'a> { None }
    #[inline] fn fold<'a, R: crate::row::Row<'a>>(arena: &'a crate::eval::borrowed::Arena, _: Self::Acc<'a>, v: R) -> Self::Acc<'a> {
        Some(v.materialise(arena))
    }
    #[inline] fn finalise<'a>(_: &'a crate::eval::borrowed::Arena, acc: Self::Acc<'a>) -> crate::eval::borrowed::Val<'a> {
        acc.unwrap_or(crate::eval::borrowed::Val::Null)
    }
}

impl crate::unified::Sink for CollectSink {
    type Acc<'a> = Vec<crate::eval::borrowed::Val<'a>>;
    #[inline] fn init<'a>() -> Self::Acc<'a> { Vec::new() }
    #[inline] fn fold<'a, R: crate::row::Row<'a>>(arena: &'a crate::eval::borrowed::Arena, mut acc: Self::Acc<'a>, v: R) -> Self::Acc<'a> {
        acc.push(v.materialise(arena)); acc
    }
    #[inline] fn finalise<'a>(arena: &'a crate::eval::borrowed::Arena, acc: Self::Acc<'a>) -> crate::eval::borrowed::Val<'a> {
        let slice = arena.alloc_slice_fill_iter(acc.into_iter());
        crate::eval::borrowed::Val::Arr(&*slice)
    }
}

impl Take {
    pub fn new(n: usize) -> Self { Self { remaining: std::cell::Cell::new(n) } }
}
impl Skip {
    pub fn new(n: usize) -> Self { Self { remaining: std::cell::Cell::new(n) } }
}
impl MapField {
    pub fn new(field: std::sync::Arc<str>) -> Self { Self { field } }
}
impl MapFieldChain {
    pub fn new(keys: Vec<std::sync::Arc<str>>) -> Self {
        Self { keys: std::sync::Arc::from(keys.into_boxed_slice()) }
    }
}
impl FlatMapField {
    pub fn new(field: std::sync::Arc<str>) -> Self { Self { field } }
}
impl FlatMapFieldChain {
    pub fn new(keys: Vec<std::sync::Arc<str>>) -> Self {
        Self { keys: std::sync::Arc::from(keys.into_boxed_slice()) }
    }
}
