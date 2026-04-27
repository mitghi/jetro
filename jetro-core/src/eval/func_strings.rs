//! String methods: case conversion, trim/pad, split/join,
//! slicing, search, URL / HTML / base64 encoding, templating.
//!
//! All operate on UTF-8 `Arc<str>`.  Indices are char-based, not
//! byte-based — `"á".chars().count() == 1` even though the byte
//! length is 2.  Slice/pad arithmetic therefore goes via
//! `str::chars()` rather than raw byte ranges, which is slower but
//! matches user expectations for JSON data.

use std::sync::Arc;
use crate::ast::Arg;
use super::{Env, EvalError, eval_pos, first_i64_arg, str_arg};
use super::value::Val;
use super::util::val_str;

macro_rules! err {
    ($($t:tt)*) => { Err(EvalError(format!($($t)*))) };
}

// ── Standard-signature string methods ─────────────────────────────────────────
// All follow fn(Val, &[Arg], &Env) -> Result<Val, EvalError>.
//
// Bodies LIFTED to first-class Stages in `composed.rs` per
// `lift_all_builtins.md`.  These shims dispatch Method-call
// invocations to the same Stage kernel — single source of truth.

macro_rules! delegate_str_stage {
    ($shim:ident, $stage:expr, $err:expr) => {
        pub fn $shim(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
            if !matches!(recv, Val::Str(_)) { return err!($err); }
            crate::composed::run_single(&$stage, &recv).ok_or_else(|| EvalError($err.into()))
        }
    };
}

delegate_str_stage!(upper,        crate::composed::Upper,        "upper: expected string");
delegate_str_stage!(lower,        crate::composed::Lower,        "lower: expected string");
delegate_str_stage!(capitalize,   crate::composed::Capitalize,   "capitalize: expected string");
delegate_str_stage!(title_case,   crate::composed::TitleCase,    "title_case: expected string");
delegate_str_stage!(trim,         crate::composed::Trim,         "trim: expected string");
delegate_str_stage!(trim_left,    crate::composed::TrimLeft,     "trim_left: expected string");
delegate_str_stage!(trim_right,   crate::composed::TrimRight,    "trim_right: expected string");
delegate_str_stage!(lines,        crate::composed::Lines,        "lines: expected string");
delegate_str_stage!(words,        crate::composed::Words,        "words: expected string");
delegate_str_stage!(chars,        crate::composed::Chars,        "chars: expected string");
delegate_str_stage!(to_number,    crate::composed::ToNumber,     "to_number: expected string");

pub fn to_bool(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        Ok(Val::Bool(matches!(s.to_lowercase().as_str(), "true" | "1" | "yes" | "on")))
    } else { err!("to_bool: expected string") }
}

delegate_str_stage!(to_base64,    crate::composed::ToBase64,     "to_base64: expected string");
delegate_str_stage!(url_encode,   crate::composed::UrlEncode,    "url_encode: expected string");
delegate_str_stage!(url_decode,   crate::composed::UrlDecode,    "url_decode: expected string");
delegate_str_stage!(html_escape,  crate::composed::HtmlEscape,   "html_escape: expected string");
delegate_str_stage!(html_unescape,crate::composed::HtmlUnescape, "html_unescape: expected string");

// from_base64 has explicit error-on-bad-input semantics; cannot use
// the generic delegate macro (which only collapses Filter → err).
pub fn from_base64(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        match base64_decode(&s) {
            Ok(bytes) => Ok(val_str(&String::from_utf8_lossy(&bytes))),
            Err(e)    => err!("from_base64: {}", e),
        }
    } else { err!("from_base64: expected string") }
}

pub fn repeat(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        let n = first_i64_arg(args, env).unwrap_or(1).max(0) as usize;
        Ok(val_str(&s.repeat(n)))
    } else { err!("repeat: expected string") }
}

pub fn pad_left(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        let width = first_i64_arg(args, env).unwrap_or(0) as usize;
        let fill  = fill_char(args, 1, env);
        let n = s.chars().count();
        if n >= width { return Ok(Val::Str(s)); }
        let pad: String = std::iter::repeat(fill).take(width - n).collect();
        Ok(val_str(&(pad + &s)))
    } else { err!("pad_left: expected string") }
}

pub fn pad_right(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        let width = first_i64_arg(args, env).unwrap_or(0) as usize;
        let fill  = fill_char(args, 1, env);
        let n = s.chars().count();
        if n >= width { return Ok(Val::Str(s)); }
        let pad: String = std::iter::repeat(fill).take(width - n).collect();
        Ok(val_str(&(s.to_string() + &pad)))
    } else { err!("pad_right: expected string") }
}

pub fn starts_with(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        Ok(Val::Bool(s.starts_with(str_arg(args, 0, env)?.as_str())))
    } else { err!("starts_with: expected string") }
}

pub fn ends_with(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        Ok(Val::Bool(s.ends_with(str_arg(args, 0, env)?.as_str())))
    } else { err!("ends_with: expected string") }
}

pub fn index_of(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        let pat = str_arg(args, 0, env)?;
        Ok(match s.find(pat.as_str()) {
            Some(i) => Val::Int(s[..i].chars().count() as i64),
            None    => Val::Int(-1),
        })
    } else { err!("index_of: expected string") }
}

pub fn last_index_of(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        let pat = str_arg(args, 0, env)?;
        Ok(match s.rfind(pat.as_str()) {
            Some(i) => Val::Int(s[..i].chars().count() as i64),
            None    => Val::Int(-1),
        })
    } else { err!("last_index_of: expected string") }
}

// `.replace` and `.replace_all` lifted to `pipeline::Stage::Replace`.
// Canonical impl in `pipeline::replace_apply`; dispatch shims in
// `eval/builtins.rs::{replace_dispatch, replace_all_dispatch}`.

pub fn strip_prefix(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        let pat = str_arg(args, 0, env)?;
        Ok(Val::Str(Arc::from(s.strip_prefix(pat.as_str()).unwrap_or(&s))))
    } else { err!("strip_prefix: expected string") }
}

pub fn strip_suffix(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        let pat = str_arg(args, 0, env)?;
        Ok(Val::Str(Arc::from(s.strip_suffix(pat.as_str()).unwrap_or(&s))))
    } else { err!("strip_suffix: expected string") }
}

// `.slice` and `.split` lifted to `pipeline::Stage::Slice` / `Stage::Split`
// in v2.4 (Step 3d-extension C). Canonical impls live in `pipeline.rs`
// (`slice_apply` / `split_apply`); dispatch shims in `eval/builtins.rs`
// (`slice_dispatch` / `split_dispatch`) parse positional args and delegate.

pub fn indent(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        let n = first_i64_arg(args, env).unwrap_or(2) as usize;
        let prefix: String = std::iter::repeat(' ').take(n).collect();
        Ok(val_str(&s.lines()
            .map(|l| format!("{}{}", prefix, l))
            .collect::<Vec<_>>().join("\n")))
    } else { err!("indent: expected string") }
}

pub fn dedent(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        let min_indent = s.lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.len() - l.trim_start().len())
            .min().unwrap_or(0);
        Ok(val_str(&s.lines()
            .map(|l| if l.len() >= min_indent { &l[min_indent..] } else { l })
            .collect::<Vec<_>>().join("\n")))
    } else { err!("dedent: expected string") }
}

pub fn str_matches(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        let pat = str_arg(args, 0, env)?;
        Ok(Val::Bool(s.contains(pat.as_str())))
    } else { err!("matches: expected string") }
}

pub fn scan(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        let pat = str_arg(args, 0, env)?;
        Ok(Val::arr(scan_raw(&s, &pat).into_iter().map(|p| val_str(&p)).collect()))
    } else { err!("scan: expected string") }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn fill_char(args: &[Arg], idx: usize, env: &Env) -> char {
    args.get(idx)
        .and_then(|a| eval_pos(a, env).ok())
        .and_then(|v| if let Val::Str(s) = v { s.chars().next() } else { None })
        .unwrap_or(' ')
}

// `capitalize_str`, `title_case_raw`, `url_encode_raw`,
// `url_decode_raw`, `html_escape_raw`, `html_unescape_raw` deleted —
// bodies migrated into composed.rs Stage impls (Capitalize / TitleCase
// / UrlEncode / UrlDecode / HtmlEscape / HtmlUnescape).
//
// `scan_raw` retained — used by str_matches.

fn scan_raw(s: &str, pat: &str) -> Vec<String> {
    if pat.is_empty() { return vec![]; }
    let mut out = Vec::new();
    let mut start = 0;
    while let Some(pos) = s[start..].find(pat) {
        out.push(pat.to_string());
        start += pos + pat.len();
    }
    out
}

// ── Base64 ────────────────────────────────────────────────────────────────────

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

// ── Case-conversion family ────────────────────────────────────────────────────

/// `"helloWorld" -> "hello_world"`. Splits on existing case
/// transitions and `[-_ ]` separators; joins with `_`.
pub fn snake_case(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        return Ok(val_str(&split_words_lower(&s).join("_")));
    }
    err!("snake_case: expected string")
}

pub fn kebab_case(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        return Ok(val_str(&split_words_lower(&s).join("-")));
    }
    err!("kebab_case: expected string")
}

pub fn camel_case(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        let parts = split_words_lower(&s);
        let mut out = String::with_capacity(s.len());
        for (i, p) in parts.iter().enumerate() {
            if i == 0 { out.push_str(p); }
            else { upper_first_into(p, &mut out); }
        }
        return Ok(val_str(&out));
    }
    err!("camel_case: expected string")
}

pub fn pascal_case(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        let parts = split_words_lower(&s);
        let mut out = String::with_capacity(s.len());
        for p in parts.iter() { upper_first_into(p, &mut out); }
        return Ok(val_str(&out));
    }
    err!("pascal_case: expected string")
}

/// Split a string into lowercased word tokens.  Tokens break on
/// `[-_ \t]`, on lower→upper transitions (`helloWorld`), and on
/// non-alphanumeric boundaries.
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

fn upper_first_into(s: &str, out: &mut String) {
    let mut chars = s.chars();
    if let Some(c) = chars.next() {
        for u in c.to_uppercase() { out.push(u); }
        for c in chars { out.push(c); }
    }
}

// ── Padding / repetition ──────────────────────────────────────────────────────

fn pad_arg_char(args: &[Arg], idx: usize, env: &Env) -> Result<char, EvalError> {
    match args.get(idx) {
        None => Ok(' '),
        Some(a) => {
            let v = eval_pos(a, env)?;
            match v {
                Val::Str(s) if s.chars().count() == 1 => Ok(s.chars().next().unwrap()),
                _ => err!("pad: filler must be a single-char string"),
            }
        }
    }
}

pub fn center(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        let n = first_i64_arg(args, env).unwrap_or(0).max(0) as usize;
        let c = pad_arg_char(args, 1, env)?;
        let cur = s.chars().count();
        if cur >= n { return Ok(Val::Str(s)); }
        let total = n - cur;
        let left = total / 2;
        let right = total - left;
        let mut out = String::with_capacity(s.len() + total);
        for _ in 0..left { out.push(c); }
        out.push_str(&s);
        for _ in 0..right { out.push(c); }
        return Ok(val_str(&out));
    }
    err!("center: expected string")
}

pub fn repeat_str(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        let n = first_i64_arg(args, env).unwrap_or(0).max(0) as usize;
        return Ok(val_str(&s.repeat(n)));
    }
    err!("repeat: expected string")
}

pub fn reverse_str(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        return Ok(val_str(&s.chars().rev().collect::<String>()));
    }
    err!("reverse: expected string")
}

// ── Char / byte introspection ─────────────────────────────────────────────────

pub fn chars_of(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        let out: Vec<Val> = s.chars().map(|c| {
            let mut tmp = [0u8; 4];
            Val::Str(Arc::<str>::from(c.encode_utf8(&mut tmp).to_string().as_str()))
        }).collect();
        return Ok(Val::arr(out));
    }
    err!("chars: expected string")
}

pub fn bytes_of(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        let out: Vec<i64> = s.as_bytes().iter().map(|&b| b as i64).collect();
        return Ok(Val::int_vec(out));
    }
    err!("bytes: expected string")
}

pub fn byte_len(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv { return Ok(Val::Int(s.len() as i64)); }
    err!("byte_len: expected string")
}

// ── Predicates / parsing ──────────────────────────────────────────────────────

pub fn is_blank(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        return Ok(Val::Bool(s.chars().all(|c| c.is_whitespace())));
    }
    err!("is_blank: expected string")
}

pub fn is_numeric(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        return Ok(Val::Bool(!s.is_empty() && s.chars().all(|c| c.is_ascii_digit())));
    }
    err!("is_numeric: expected string")
}

pub fn is_alpha(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        return Ok(Val::Bool(!s.is_empty() && s.chars().all(|c| c.is_alphabetic())));
    }
    err!("is_alpha: expected string")
}

pub fn is_ascii(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        return Ok(Val::Bool(s.is_ascii()));
    }
    err!("is_ascii: expected string")
}

pub fn parse_int(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        return Ok(s.trim().parse::<i64>().map(Val::Int).unwrap_or(Val::Null));
    }
    err!("parse_int: expected string")
}

pub fn parse_float(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        return Ok(s.trim().parse::<f64>().map(Val::Float).unwrap_or(Val::Null));
    }
    err!("parse_float: expected string")
}

pub fn parse_bool(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        return Ok(match s.trim().to_ascii_lowercase().as_str() {
            "true" | "yes" | "1" | "on"  => Val::Bool(true),
            "false" | "no" | "0" | "off" => Val::Bool(false),
            _ => Val::Null,
        });
    }
    err!("parse_bool: expected string")
}

// ── Substring set predicates ─────────────────────────────────────────────────

pub fn contains_any(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        let needles = collect_str_arg_list(args, env)?;
        for n in &needles {
            if s.contains(n.as_ref()) { return Ok(Val::Bool(true)); }
        }
        return Ok(Val::Bool(false));
    }
    err!("contains_any: expected string")
}

pub fn contains_all(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        let needles = collect_str_arg_list(args, env)?;
        for n in &needles {
            if !s.contains(n.as_ref()) { return Ok(Val::Bool(false)); }
        }
        return Ok(Val::Bool(true));
    }
    err!("contains_all: expected string")
}

fn collect_str_arg_list(args: &[Arg], env: &Env) -> Result<Vec<Arc<str>>, EvalError> {
    // Accept either a single Val::Arr arg of strings, or N positional
    // string args.
    if args.len() == 1 {
        if let Val::Arr(a) = eval_pos(&args[0], env)? {
            let mut out = Vec::with_capacity(a.len());
            for item in a.iter() {
                if let Val::Str(s) = item { out.push(s.clone()); }
                else { return err!("contains_*: array elements must be strings"); }
            }
            return Ok(out);
        }
    }
    let mut out = Vec::with_capacity(args.len());
    for a in args {
        if let Val::Str(s) = eval_pos(a, env)? { out.push(s); }
        else { return err!("contains_*: arg must be string"); }
    }
    Ok(out)
}

// ── Regex family ──────────────────────────────────────────────────────────────
//
// Pattern compile cache (thread-local).  Each unique pattern string
// hits the compile path once per thread; subsequent uses reuse the
// same `Arc<Regex>`.

thread_local! {
    static REGEX_CACHE: std::cell::RefCell<std::collections::HashMap<String, std::sync::Arc<regex::Regex>>>
        = std::cell::RefCell::new(std::collections::HashMap::with_capacity(32));
}

fn compile_regex(pat: &str) -> Result<std::sync::Arc<regex::Regex>, EvalError> {
    REGEX_CACHE.with(|cell| {
        let mut m = cell.borrow_mut();
        if let Some(r) = m.get(pat) { return Ok(std::sync::Arc::clone(r)); }
        let re = regex::Regex::new(pat).map_err(|e| EvalError(format!("regex compile: {}", e)))?;
        let arc = std::sync::Arc::new(re);
        m.insert(pat.to_string(), std::sync::Arc::clone(&arc));
        Ok(arc)
    })
}

pub fn re_match(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        let pat = str_arg(args, 0, env)?;
        let re = compile_regex(pat.as_str())?;
        return Ok(Val::Bool(re.is_match(s.as_ref())));
    }
    err!("match: expected string")
}

pub fn re_match_first(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        let pat = str_arg(args, 0, env)?;
        let re = compile_regex(pat.as_str())?;
        return Ok(re.find(s.as_ref())
            .map(|m| Val::Str(Arc::<str>::from(m.as_str())))
            .unwrap_or(Val::Null));
    }
    err!("match_first: expected string")
}

pub fn re_match_all(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        let pat = str_arg(args, 0, env)?;
        let re = compile_regex(pat.as_str())?;
        let out: Vec<Arc<str>> = re.find_iter(s.as_ref())
            .map(|m| Arc::<str>::from(m.as_str())).collect();
        return Ok(Val::str_vec(out));
    }
    err!("match_all: expected string")
}

pub fn re_captures(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        let pat = str_arg(args, 0, env)?;
        let re = compile_regex(pat.as_str())?;
        return Ok(match re.captures(s.as_ref()) {
            Some(c) => {
                let mut out: Vec<Val> = Vec::with_capacity(c.len());
                for i in 0..c.len() {
                    out.push(c.get(i)
                        .map(|m| Val::Str(Arc::<str>::from(m.as_str())))
                        .unwrap_or(Val::Null));
                }
                Val::arr(out)
            }
            None => Val::Null,
        });
    }
    err!("captures: expected string")
}

pub fn re_captures_all(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        let pat = str_arg(args, 0, env)?;
        let re = compile_regex(pat.as_str())?;
        let mut all: Vec<Val> = Vec::new();
        for c in re.captures_iter(s.as_ref()) {
            let mut row: Vec<Val> = Vec::with_capacity(c.len());
            for i in 0..c.len() {
                row.push(c.get(i)
                    .map(|m| Val::Str(Arc::<str>::from(m.as_str())))
                    .unwrap_or(Val::Null));
            }
            all.push(Val::arr(row));
        }
        return Ok(Val::arr(all));
    }
    err!("captures_all: expected string")
}

pub fn re_replace(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        let pat = str_arg(args, 0, env)?;
        let with = str_arg(args, 1, env)?;
        let re = compile_regex(pat.as_str())?;
        let out = re.replace(s.as_ref(), with.as_str());
        return Ok(val_str(&out));
    }
    err!("replace_re: expected string")
}

pub fn re_replace_all(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        let pat = str_arg(args, 0, env)?;
        let with = str_arg(args, 1, env)?;
        let re = compile_regex(pat.as_str())?;
        let out = re.replace_all(s.as_ref(), with.as_str());
        return Ok(val_str(&out));
    }
    err!("replace_all_re: expected string")
}

pub fn re_split(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        let pat = str_arg(args, 0, env)?;
        let re = compile_regex(pat.as_str())?;
        let out: Vec<Arc<str>> = re.split(s.as_ref()).map(Arc::<str>::from).collect();
        return Ok(Val::str_vec(out));
    }
    err!("split_re: expected string")
}
