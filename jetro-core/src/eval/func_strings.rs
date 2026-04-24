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

pub fn upper(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        if s.is_ascii() {
            let mut buf: String = s.as_ref().to_owned();
            buf.make_ascii_uppercase();
            return Ok(Val::Str(Arc::<str>::from(buf)));
        }
        Ok(Val::Str(Arc::<str>::from(s.to_uppercase())))
    } else { err!("upper: expected string") }
}

pub fn lower(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        if s.is_ascii() {
            let mut buf: String = s.as_ref().to_owned();
            buf.make_ascii_lowercase();
            return Ok(Val::Str(Arc::<str>::from(buf)));
        }
        Ok(Val::Str(Arc::<str>::from(s.to_lowercase())))
    } else { err!("lower: expected string") }
}

pub fn capitalize(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv { Ok(val_str(&capitalize_str(&s))) }
    else { err!("capitalize: expected string") }
}

pub fn title_case(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv { Ok(val_str(&title_case_raw(&s))) }
    else { err!("title_case: expected string") }
}

pub fn trim(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        let t = s.trim();
        if t.len() == s.len() { return Ok(Val::Str(s)); }
        Ok(Val::Str(Arc::from(t)))
    } else { err!("trim: expected string") }
}

pub fn trim_left(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        let t = s.trim_start();
        if t.len() == s.len() { return Ok(Val::Str(s)); }
        Ok(Val::Str(Arc::from(t)))
    } else { err!("trim_left: expected string") }
}

pub fn trim_right(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        let t = s.trim_end();
        if t.len() == s.len() { return Ok(Val::Str(s)); }
        Ok(Val::Str(Arc::from(t)))
    } else { err!("trim_right: expected string") }
}

pub fn lines(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv { Ok(Val::arr(s.lines().map(val_str).collect())) }
    else { err!("lines: expected string") }
}

pub fn words(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv { Ok(Val::arr(s.split_whitespace().map(val_str).collect())) }
    else { err!("words: expected string") }
}

pub fn chars(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        Ok(Val::arr(s.chars().map(|c| val_str(&c.to_string())).collect()))
    } else { err!("chars: expected string") }
}

pub fn to_number(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        if let Ok(i) = s.parse::<i64>()  { return Ok(Val::Int(i)); }
        if let Ok(f) = s.parse::<f64>()  { return Ok(Val::Float(f)); }
        Ok(Val::Null)
    } else { err!("to_number: expected string") }
}

pub fn to_bool(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        Ok(Val::Bool(matches!(s.to_lowercase().as_str(), "true" | "1" | "yes" | "on")))
    } else { err!("to_bool: expected string") }
}

pub fn to_base64(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv { Ok(val_str(&base64_encode(s.as_bytes()))) }
    else { err!("to_base64: expected string") }
}

pub fn from_base64(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        match base64_decode(&s) {
            Ok(bytes) => Ok(val_str(&String::from_utf8_lossy(&bytes))),
            Err(e)    => err!("from_base64: {}", e),
        }
    } else { err!("from_base64: expected string") }
}

pub fn url_encode(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv { Ok(val_str(&url_encode_raw(&s))) }
    else { err!("url_encode: expected string") }
}

pub fn url_decode(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv { Ok(val_str(&url_decode_raw(&s))) }
    else { err!("url_decode: expected string") }
}

pub fn html_escape(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv { Ok(val_str(&html_escape_raw(&s))) }
    else { err!("html_escape: expected string") }
}

pub fn html_unescape(recv: Val, _: &[Arg], _: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv { Ok(val_str(&html_unescape_raw(&s))) }
    else { err!("html_unescape: expected string") }
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

pub fn replace(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        let from = str_arg(args, 0, env)?;
        let to   = str_arg(args, 1, env)?;
        // Short-circuit: needle absent -> return receiver unchanged (no alloc).
        if !s.contains(from.as_str()) { return Ok(Val::Str(s)); }
        Ok(Val::Str(Arc::<str>::from(s.replacen(from.as_str(), to.as_str(), 1))))
    } else { err!("replace: expected string") }
}

pub fn replace_all(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        let from = str_arg(args, 0, env)?;
        let to   = str_arg(args, 1, env)?;
        if !s.contains(from.as_str()) { return Ok(Val::Str(s)); }
        Ok(Val::Str(Arc::<str>::from(s.replace(from.as_str(), to.as_str()))))
    } else { err!("replace_all: expected string") }
}

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

pub fn str_slice(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        let start = first_i64_arg(args, env).unwrap_or(0) as usize;
        let chars: Vec<char> = s.chars().collect();
        let end = args.get(1)
            .and_then(|a| eval_pos(a, env).ok())
            .and_then(|v| v.as_i64())
            .map(|n| n as usize)
            .unwrap_or(chars.len())
            .min(chars.len());
        let start = start.min(end);
        Ok(val_str(&chars[start..end].iter().collect::<String>()))
    } else { err!("slice: expected string") }
}

pub fn split(recv: Val, args: &[Arg], env: &Env) -> Result<Val, EvalError> {
    if let Val::Str(s) = recv {
        let sep = str_arg(args, 0, env)?;
        Ok(Val::arr(s.split(sep.as_str()).map(val_str).collect()))
    } else { err!("split: expected string") }
}

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

fn capitalize_str(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None    => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + &c.as_str().to_lowercase(),
    }
}

fn title_case_raw(s: &str) -> String {
    s.split_whitespace().map(capitalize_str).collect::<Vec<_>>().join(" ")
}

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

fn url_encode_raw(s: &str) -> String {
    s.bytes().flat_map(|b| {
        if b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b'.' || b == b'~' {
            vec![b as char]
        } else {
            format!("%{:02X}", b).chars().collect()
        }
    }).collect()
}

fn url_decode_raw(s: &str) -> String {
    let mut out = Vec::new();
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(h1), Some(h2)) = (
                char::from(bytes[i + 1]).to_digit(16),
                char::from(bytes[i + 2]).to_digit(16),
            ) {
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
}

fn html_escape_raw(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
     .replace('"', "&quot;").replace('\'', "&#39;")
}

fn html_unescape_raw(s: &str) -> String {
    s.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
     .replace("&quot;", "\"").replace("&#39;", "'")
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
