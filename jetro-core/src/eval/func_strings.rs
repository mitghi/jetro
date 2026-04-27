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

// ── Lifted built-ins (bodies live in `composed.rs`) ──────────────────────────
//
// All standard-signature string methods (upper/lower/trim/.../pad_*/
// index_of/...) were lifted to first-class Stages in `composed.rs`.
// Dispatch shims live in `composed::shims::*` and are registered
// directly by `eval::builtins.rs` — `eval::func_strings::*` no
// longer holds those entry points.
//
// What remains in this file:
//   - `from_base64`           — explicit error-on-bad-input semantics
//                               (cannot use the generic Stage filter →
//                               err shim macro)
//   - `scan`                  — not yet lifted
//   - case-conversion (snake/kebab/camel/pascal/screaming/title) —
//                               not yet lifted
//   - `base64_encode/decode`  — public helper used by composed::ToBase64
//                               and explicit from_base64 shim
//
// Everything else delete-able once the remaining shims migrate.

pub fn from_base64(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        match base64_decode(&s) {
            Ok(bytes) => Ok(val_str(&String::from_utf8_lossy(&bytes))),
            Err(e)    => err!("from_base64: {}", e),
        }
    } else { err!("from_base64: expected string") }
}

// `scan` LIFTED to composed::Scan; shim in composed::shims::scan.

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

// Case-conversion family LIFTED to composed::{SnakeCase, KebabCase,
// CamelCase, PascalCase}.  Helpers `split_words_lower` /
// `upper_first_into` moved to composed.rs alongside the Stages.

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

// `reverse_str` LIFTED to composed::ReverseStr; shim in
// composed::shims::reverse_str.

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
