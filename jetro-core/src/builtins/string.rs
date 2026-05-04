use crate::context::EvalError;
use crate::value::Val;
use super::BuiltinMethod;
use std::sync::Arc;

/// Returns a substring by character indices, supporting negative indexing.
/// Returns a zero-copy `StrSlice` view when the input is ASCII; allocates otherwise.
pub fn slice_apply(recv: Val, start: i64, end: Option<i64>) -> Val {
    let (parent, base_off, view_len): (Arc<str>, usize, usize) = match recv {
        Val::Str(s) => {
            let l = s.len();
            (s, 0, l)
        }
        Val::StrSlice(r) => {
            let parent = r.to_arc();
            let plen = parent.len();
            (parent, 0, plen)
        }
        other => return other,
    };
    let view = &parent[base_off..base_off + view_len];
    let blen = view.len();
    if view.is_ascii() {
        let start_u = if start < 0 {
            blen.saturating_sub((-start) as usize)
        } else {
            (start as usize).min(blen)
        };
        let end_u = match end {
            Some(e) if e < 0 => blen.saturating_sub((-e) as usize),
            Some(e) => (e as usize).min(blen),
            None => blen,
        };
        let start_u = start_u.min(end_u);
        if start_u == 0 && end_u == blen {
            return Val::Str(parent);
        }
        return Val::StrSlice(crate::strref::StrRef::slice(
            parent,
            base_off + start_u,
            base_off + end_u,
        ));
    }
    let chars: Vec<(usize, char)> = view.char_indices().collect();
    let n = chars.len() as i64;
    let resolve = |i: i64| -> usize {
        let r = if i < 0 { n + i } else { i };
        r.clamp(0, n) as usize
    };
    let s_idx = resolve(start);
    let e_idx = match end {
        Some(e) => resolve(e),
        None => n as usize,
    };
    let s_idx = s_idx.min(e_idx);
    let s_b = chars.get(s_idx).map(|c| c.0).unwrap_or(view.len());
    let e_b = chars.get(e_idx).map(|c| c.0).unwrap_or(view.len());
    if s_b == 0 && e_b == view.len() {
        return Val::Str(parent);
    }
    Val::StrSlice(crate::strref::StrRef::slice(
        parent,
        base_off + s_b,
        base_off + e_b,
    ))
}

/// Splits a string on `sep` and returns the parts as an array of strings.
#[inline]
pub fn split_apply(recv: &Val, sep: &str) -> Option<Val> {
    let s: &str = match recv {
        Val::Str(s) => s.as_ref(),
        Val::StrSlice(r) => r.as_str(),
        _ => return None,
    };
    Some(Val::arr(
        s.split(sep)
            .map(|p| Val::Str(Arc::<str>::from(p)))
            .collect(),
    ))
}

/// Splits a slice into non-overlapping chunks of size `n` (last chunk may be smaller).
#[inline]
pub fn chunk_apply(items: &[Val], n: usize) -> Vec<Val> {
    let n = n.max(1);
    items.chunks(n).map(|c| Val::arr(c.to_vec())).collect()
}

/// Produces all contiguous windows of size `n` from a slice of values.
#[inline]
pub fn window_apply(items: &[Val], n: usize) -> Vec<Val> {
    let n = n.max(1);
    items.windows(n).map(|w| Val::arr(w.to_vec())).collect()
}

/// Replaces `needle` with `replacement` in a string. When `all` is true replaces every
/// occurrence; otherwise only the first. Returns the original value unchanged if `needle` is absent.
#[inline]
pub fn replace_apply(recv: Val, needle: &str, replacement: &str, all: bool) -> Option<Val> {
    let s: Arc<str> = match recv {
        Val::Str(s) => s,
        Val::StrSlice(r) => r.to_arc(),
        _ => return None,
    };
    if !s.contains(needle) {
        return Some(Val::Str(s));
    }
    let out = if all {
        s.replace(needle, replacement)
    } else {
        s.replacen(needle, replacement, 1)
    };
    Some(Val::Str(Arc::<str>::from(out)))
}

/// Applies a `&str → String` transform to the string inside `recv`, wrapping the result in `Val::Str`.
#[inline]
fn map_str_owned(recv: &Val, f: impl FnOnce(&str) -> String) -> Option<Val> {
    let s = recv.as_str_ref()?;
    Some(Val::Str(Arc::<str>::from(f(s).as_str())))
}

/// Converts the string to all-uppercase (ASCII fast path).
#[inline]
pub fn upper_apply(recv: &Val) -> Option<Val> {
    map_str_owned(recv, |s| {
        if s.is_ascii() {
            let mut buf = s.to_owned();
            buf.make_ascii_uppercase();
            buf
        } else {
            s.to_uppercase()
        }
    })
}

/// Converts the string to all-lowercase (ASCII fast path).
#[inline]
pub fn lower_apply(recv: &Val) -> Option<Val> {
    map_str_owned(recv, |s| {
        if s.is_ascii() {
            let mut buf = s.to_owned();
            buf.make_ascii_lowercase();
            buf
        } else {
            s.to_lowercase()
        }
    })
}

/// Strips leading and trailing whitespace from a string.
#[inline]
pub fn trim_apply(recv: &Val) -> Option<Val> {
    map_str_owned(recv, |s| s.trim().to_owned())
}

/// Strips leading whitespace from a string.
#[inline]
pub fn trim_left_apply(recv: &Val) -> Option<Val> {
    map_str_owned(recv, |s| s.trim_start().to_owned())
}

/// Strips trailing whitespace from a string.
#[inline]
pub fn trim_right_apply(recv: &Val) -> Option<Val> {
    map_str_owned(recv, |s| s.trim_end().to_owned())
}

/// Uppercases the first character and lowercases the rest of a string.
#[inline]
pub fn capitalize_apply(recv: &Val) -> Option<Val> {
    map_str_owned(recv, |s| {
        let mut out = String::with_capacity(s.len());
        let mut chars = s.chars();
        if let Some(first) = chars.next() {
            for c in first.to_uppercase() {
                out.push(c);
            }
            out.push_str(&chars.as_str().to_lowercase());
        }
        out
    })
}

/// Capitalises the first letter of each whitespace-delimited word.
#[inline]
pub fn title_case_apply(recv: &Val) -> Option<Val> {
    map_str_owned(recv, |s| {
        let mut out = String::with_capacity(s.len());
        let mut at_start = true;
        for c in s.chars() {
            if c.is_whitespace() {
                out.push(c);
                at_start = true;
            } else if at_start {
                for u in c.to_uppercase() {
                    out.push(u);
                }
                at_start = false;
            } else {
                for l in c.to_lowercase() {
                    out.push(l);
                }
            }
        }
        out
    })
}

/// Escapes `<`, `>`, `&`, `"`, and `'` to their HTML entity equivalents.
#[inline]
pub fn html_escape_apply(recv: &Val) -> Option<Val> {
    map_str_owned(recv, |s| {
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
    })
}

/// Converts HTML entities (`&lt;`, `&gt;`, `&amp;`, `&quot;`, `&#39;`) back to their characters.
#[inline]
pub fn html_unescape_apply(recv: &Val) -> Option<Val> {
    map_str_owned(recv, |s| {
        s.replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", "\"")
            .replace("&#39;", "'")
    })
}

/// Percent-encodes a string using RFC 3986 unreserved characters (`A-Z a-z 0-9 - _ . ~`).
#[inline]
pub fn url_encode_apply(recv: &Val) -> Option<Val> {
    map_str_owned(recv, |s| {
        let mut out = String::with_capacity(s.len());
        for b in s.as_bytes() {
            let b = *b;
            match b {
                b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                    out.push(b as char)
                }
                _ => {
                    use std::fmt::Write;
                    let _ = write!(out, "%{:02X}", b);
                }
            }
        }
        out
    })
}

/// Decodes a percent-encoded URL string, also converting `+` to space.
#[inline]
pub fn url_decode_apply(recv: &Val) -> Option<Val> {
    map_str_owned(recv, |s| {
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
    })
}

/// Encodes a string's bytes as standard (non-padded) Base64.
#[inline]
pub fn to_base64_apply(recv: &Val) -> Option<Val> {
    map_str_owned(recv, |s| {
        crate::builtin_helpers::base64_encode(s.as_bytes())
    })
}

/// Removes the common leading whitespace prefix from every non-blank line.
#[inline]
pub fn dedent_apply(recv: &Val) -> Option<Val> {
    map_str_owned(recv, |s| {
        let min_indent = s
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.len() - l.trim_start().len())
            .min()
            .unwrap_or(0);
        s.lines()
            .map(|l| {
                if l.len() >= min_indent {
                    &l[min_indent..]
                } else {
                    l
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    })
}

/// Converts a string to `snake_case` by splitting on word boundaries and joining with `_`.
#[inline]
pub fn snake_case_apply(recv: &Val) -> Option<Val> {
    map_str_owned(recv, |s| {
        crate::builtin_helpers::split_words_lower(s).join("_")
    })
}

/// Converts a string to `kebab-case` by splitting on word boundaries and joining with `-`.
#[inline]
pub fn kebab_case_apply(recv: &Val) -> Option<Val> {
    map_str_owned(recv, |s| {
        crate::builtin_helpers::split_words_lower(s).join("-")
    })
}

/// Converts a string to `camelCase` (first word lowercase, subsequent words title-cased).
#[inline]
pub fn camel_case_apply(recv: &Val) -> Option<Val> {
    map_str_owned(recv, |s| {
        let parts = crate::builtin_helpers::split_words_lower(s);
        let mut out = String::with_capacity(s.len());
        for (i, p) in parts.iter().enumerate() {
            if i == 0 {
                out.push_str(p);
            } else {
                crate::builtin_helpers::upper_first_into(p, &mut out);
            }
        }
        out
    })
}

/// Converts a string to `PascalCase` (every word title-cased, no separator).
#[inline]
pub fn pascal_case_apply(recv: &Val) -> Option<Val> {
    map_str_owned(recv, |s| {
        let parts = crate::builtin_helpers::split_words_lower(s);
        let mut out = String::with_capacity(s.len());
        for p in parts.iter() {
            crate::builtin_helpers::upper_first_into(p, &mut out);
        }
        out
    })
}

/// Reverses the Unicode codepoints of a string.
#[inline]
pub fn reverse_str_apply(recv: &Val) -> Option<Val> {
    map_str_owned(recv, |s| s.chars().rev().collect::<String>())
}

/// Applies a `&str → Val` transform to the string inside `recv`.
#[inline]
fn map_str_val(recv: &Val, f: impl FnOnce(&str) -> Val) -> Option<Val> {
    Some(f(recv.as_str_ref()?))
}

/// Splits a string on newlines and returns each line as a `Val::Str`.
#[inline]
pub fn lines_apply(recv: &Val) -> Option<Val> {
    map_str_val(recv, |s| {
        Val::arr(s.lines().map(|l| Val::Str(Arc::from(l))).collect())
    })
}

/// Splits a string on whitespace and returns each token as a `Val::Str`.
#[inline]
pub fn words_apply(recv: &Val) -> Option<Val> {
    map_str_val(recv, |s| {
        Val::arr(
            s.split_whitespace()
                .map(|w| Val::Str(Arc::from(w)))
                .collect(),
        )
    })
}

/// Returns each Unicode character as a single-char `Val::Str`.
#[inline]
pub fn chars_apply(recv: &Val) -> Option<Val> {
    map_str_val(recv, |s| {
        Val::arr(
            s.chars()
                .map(|c| Val::Str(Arc::from(c.to_string())))
                .collect(),
        )
    })
}

/// Returns each Unicode code point re-encoded as a UTF-8 `Val::Str` (same as `chars` for BMP).
#[inline]
pub fn chars_of_apply(recv: &Val) -> Option<Val> {
    map_str_val(recv, |s| {
        let mut out: Vec<Val> = Vec::new();
        let mut tmp = [0u8; 4];
        for c in s.chars() {
            let utf8 = c.encode_utf8(&mut tmp);
            out.push(Val::Str(Arc::from(utf8.as_ref())));
        }
        Val::arr(out)
    })
}

/// Returns each byte of the string's UTF-8 encoding as a `Val::Int`.
#[inline]
pub fn bytes_of_apply(recv: &Val) -> Option<Val> {
    map_str_val(recv, |s| {
        let v: Vec<i64> = s.as_bytes().iter().map(|&b| b as i64).collect();
        Val::int_vec(v)
    })
}

/// Returns the ceiling (round-up) of a numeric value as `Val::Int`.
#[inline]
pub fn ceil_apply(recv: &Val) -> Option<Val> {
    match recv {
        Val::Int(n) => Some(Val::Int(*n)),
        Val::Float(f) => Some(Val::Int(f.ceil() as i64)),
        _ => None,
    }
}

/// Like [`ceil_apply`] but returns an error for non-numeric receivers.
#[inline]
pub fn try_ceil_apply(recv: &Val) -> Result<Option<Val>, EvalError> {
    ceil_apply(recv)
        .map(Some)
        .ok_or_else(|| EvalError("ceil: expected number".into()))
}

/// Returns the floor (round-down) of a numeric value as `Val::Int`.
#[inline]
pub fn floor_apply(recv: &Val) -> Option<Val> {
    match recv {
        Val::Int(n) => Some(Val::Int(*n)),
        Val::Float(f) => Some(Val::Int(f.floor() as i64)),
        _ => None,
    }
}

/// Like [`floor_apply`] but returns an error for non-numeric receivers.
#[inline]
pub fn try_floor_apply(recv: &Val) -> Result<Option<Val>, EvalError> {
    floor_apply(recv)
        .map(Some)
        .ok_or_else(|| EvalError("floor: expected number".into()))
}

/// Rounds a numeric value to the nearest integer.
#[inline]
pub fn round_apply(recv: &Val) -> Option<Val> {
    match recv {
        Val::Int(n) => Some(Val::Int(*n)),
        Val::Float(f) => Some(Val::Int(f.round() as i64)),
        _ => None,
    }
}

/// Like [`round_apply`] but returns an error for non-numeric receivers.
#[inline]
pub fn try_round_apply(recv: &Val) -> Result<Option<Val>, EvalError> {
    round_apply(recv)
        .map(Some)
        .ok_or_else(|| EvalError("round: expected number".into()))
}

/// Returns the absolute value of an integer or float.
#[inline]
pub fn abs_apply(recv: &Val) -> Option<Val> {
    match recv {
        Val::Int(n) => Some(Val::Int(n.wrapping_abs())),
        Val::Float(f) => Some(Val::Float(f.abs())),
        _ => None,
    }
}

/// Like [`abs_apply`] but returns an error for non-numeric receivers.
#[inline]
pub fn try_abs_apply(recv: &Val) -> Result<Option<Val>, EvalError> {
    abs_apply(recv)
        .map(Some)
        .ok_or_else(|| EvalError("abs: expected number".into()))
}

/// Parses the string as a base-10 `i64`; returns `Val::Null` on failure.
#[inline]
pub fn parse_int_apply(recv: &Val) -> Option<Val> {
    map_str_val(recv, |s| {
        s.trim().parse::<i64>().map(Val::Int).unwrap_or(Val::Null)
    })
}

/// Parses the string as an `f64`; returns `Val::Null` on failure.
#[inline]
pub fn parse_float_apply(recv: &Val) -> Option<Val> {
    map_str_val(recv, |s| {
        s.trim().parse::<f64>().map(Val::Float).unwrap_or(Val::Null)
    })
}

/// Parses common truthy/falsy string representations to `Val::Bool`; returns `Val::Null` otherwise.
/// Recognises `true/yes/1/on` and `false/no/0/off` (case-insensitive).
#[inline]
pub fn parse_bool_apply(recv: &Val) -> Option<Val> {
    map_str_val(recv, |s| match s.trim().to_ascii_lowercase().as_str() {
        "true" | "yes" | "1" | "on" => Val::Bool(true),
        "false" | "no" | "0" | "off" => Val::Bool(false),
        _ => Val::Null,
    })
}

/// Decodes a Base64 string to its UTF-8 representation; returns `Val::Null` for invalid input.
#[inline]
pub fn from_base64_apply(recv: &Val) -> Option<Val> {
    map_str_val(recv, |s| match crate::builtin_helpers::base64_decode(s) {
        Ok(bytes) => Val::Str(Arc::from(String::from_utf8_lossy(&bytes).as_ref())),
        Err(_) => Val::Null,
    })
}

/// Returns the string repeated `n` times.
#[inline]
pub fn repeat_apply(recv: &Val, n: usize) -> Option<Val> {
    Some(Val::Str(Arc::from(recv.as_str_ref()?.repeat(n))))
}

/// Removes `prefix` from the beginning of the string if present; returns the original otherwise.
#[inline]
pub fn strip_prefix_apply(recv: &Val, prefix: &str) -> Option<Val> {
    let s = recv.as_str_ref()?;
    Some(match s.strip_prefix(prefix) {
        Some(stripped) => Val::Str(Arc::<str>::from(stripped)),
        None => recv.clone(),
    })
}

/// Removes `suffix` from the end of the string if present; returns the original otherwise.
#[inline]
pub fn strip_suffix_apply(recv: &Val, suffix: &str) -> Option<Val> {
    let s = recv.as_str_ref()?;
    Some(match s.strip_suffix(suffix) {
        Some(stripped) => Val::Str(Arc::<str>::from(stripped)),
        None => recv.clone(),
    })
}

/// Left-pads the string to `width` characters with `fill`; returns the original when already wide enough.
#[inline]
pub fn pad_left_apply(recv: &Val, width: usize, fill: char) -> Option<Val> {
    let s = recv.as_str_ref()?;
    let n = s.chars().count();
    if n >= width {
        return Some(recv.clone());
    }
    let pad: String = std::iter::repeat(fill).take(width - n).collect();
    Some(Val::Str(Arc::from(pad + s)))
}

/// Right-pads the string to `width` characters with `fill`; returns the original when already wide enough.
#[inline]
pub fn pad_right_apply(recv: &Val, width: usize, fill: char) -> Option<Val> {
    let s = recv.as_str_ref()?;
    let n = s.chars().count();
    if n >= width {
        return Some(recv.clone());
    }
    let pad: String = std::iter::repeat(fill).take(width - n).collect();
    Some(Val::Str(Arc::from(s.to_string() + &pad)))
}

/// Centers the string within `width` characters by padding both sides with `fill`.
#[inline]
pub fn center_apply(recv: &Val, width: usize, fill: char) -> Option<Val> {
    let s = recv.as_str_ref()?;
    let cur = s.chars().count();
    if cur >= width {
        return Some(recv.clone());
    }
    let total = width - cur;
    let left = total / 2;
    let right = total - left;
    let mut out = String::with_capacity(s.len() + total);
    for _ in 0..left {
        out.push(fill);
    }
    out.push_str(s);
    for _ in 0..right {
        out.push(fill);
    }
    Some(Val::Str(Arc::from(out)))
}

/// Prepends `n` spaces to each line of the string.
#[inline]
pub fn indent_apply(recv: &Val, n: usize) -> Option<Val> {
    let s = recv.as_str_ref()?;
    let prefix: String = std::iter::repeat(' ').take(n).collect();
    let out = s
        .lines()
        .map(|l| format!("{}{}", prefix, l))
        .collect::<Vec<_>>()
        .join("\n");
    Some(Val::Str(Arc::from(out)))
}

/// Finds every non-overlapping occurrence of `pat` and returns an array of the matched strings.
#[inline]
pub fn scan_apply(recv: &Val, pat: &str) -> Option<Val> {
    let s = recv.as_str_ref()?;
    let mut out: Vec<Val> = Vec::new();
    if !pat.is_empty() {
        let mut start = 0usize;
        while let Some(pos) = s[start..].find(pat) {
            out.push(Val::Str(Arc::from(pat)));
            start += pos + pat.len();
        }
    }
    Some(Val::arr(out))
}

/// Applies a numeric aggregate (`sum`, `avg`, `min`, `max`) to an array or typed numeric vector.
/// Returns `Val::Null` when the receiver is not an array-like type.
#[inline]
pub fn numeric_aggregate_apply(recv: &Val, method: BuiltinMethod) -> Val {
    match recv {
        Val::IntVec(a) => return numeric_aggregate_i64(a, method),
        Val::FloatVec(a) => return numeric_aggregate_f64(a, method),
        Val::Arr(a) => numeric_aggregate_values(a, method),
        _ => Val::Null,
    }
}

/// Numeric aggregate with a projection: evaluates `eval` on each element first,
/// then aggregates all numeric results. Non-numeric projected values are silently skipped.
#[inline]
pub fn numeric_aggregate_projected_apply<F>(
    recv: &Val,
    method: BuiltinMethod,
    mut eval: F,
) -> Result<Val, EvalError>
where
    F: FnMut(&Val) -> Result<Val, EvalError>,
{
    let items = recv
        .as_vals()
        .ok_or_else(|| EvalError("expected array for numeric aggregate".into()))?;

    let mut vals = Vec::with_capacity(items.len());
    for item in items.iter() {
        let v = eval(item)?;
        if v.is_number() {
            vals.push(v);
        }
    }
    Ok(numeric_aggregate_values(&vals, method))
}

/// Numeric aggregate specialised for homogeneous `i64` slices.
#[inline]
fn numeric_aggregate_i64(a: &[i64], method: BuiltinMethod) -> Val {
    match method {
        BuiltinMethod::Sum => Val::Int(a.iter().fold(0i64, |acc, n| acc.wrapping_add(*n))),
        BuiltinMethod::Avg => {
            if a.is_empty() {
                Val::Null
            } else {
                let s = a.iter().fold(0i64, |acc, n| acc.wrapping_add(*n));
                Val::Float(s as f64 / a.len() as f64)
            }
        }
        BuiltinMethod::Min => a.iter().min().copied().map(Val::Int).unwrap_or(Val::Null),
        BuiltinMethod::Max => a.iter().max().copied().map(Val::Int).unwrap_or(Val::Null),
        _ => Val::Null,
    }
}

/// Numeric aggregate specialised for homogeneous `f64` slices.
#[inline]
fn numeric_aggregate_f64(a: &[f64], method: BuiltinMethod) -> Val {
    match method {
        BuiltinMethod::Sum => Val::Float(a.iter().sum()),
        BuiltinMethod::Avg => {
            if a.is_empty() {
                Val::Null
            } else {
                Val::Float(a.iter().sum::<f64>() / a.len() as f64)
            }
        }
        BuiltinMethod::Min => a
            .iter()
            .copied()
            .reduce(f64::min)
            .map(Val::Float)
            .unwrap_or(Val::Null),
        BuiltinMethod::Max => a
            .iter()
            .copied()
            .reduce(f64::max)
            .map(Val::Float)
            .unwrap_or(Val::Null),
        _ => Val::Null,
    }
}

/// Numeric aggregate for heterogeneous `Val` slices; skips non-numeric elements.
#[inline]
fn numeric_aggregate_values(a: &[Val], method: BuiltinMethod) -> Val {
    match method {
        BuiltinMethod::Sum => {
            let mut i_acc: i64 = 0;
            let mut f_acc: f64 = 0.0;
            let mut floated = false;
            for v in a {
                match v {
                    Val::Int(n) if !floated => i_acc = i_acc.wrapping_add(*n),
                    Val::Int(n) => f_acc += *n as f64,
                    Val::Float(f) if !floated => {
                        f_acc = i_acc as f64 + *f;
                        floated = true;
                    }
                    Val::Float(f) => f_acc += *f,
                    _ => {}
                }
            }
            if floated {
                Val::Float(f_acc)
            } else {
                Val::Int(i_acc)
            }
        }
        BuiltinMethod::Avg => {
            let mut sum = 0.0;
            let mut n = 0usize;
            for v in a {
                match v {
                    Val::Int(i) => {
                        sum += *i as f64;
                        n += 1;
                    }
                    Val::Float(f) => {
                        sum += *f;
                        n += 1;
                    }
                    _ => {}
                }
            }
            if n == 0 {
                Val::Null
            } else {
                Val::Float(sum / n as f64)
            }
        }
        BuiltinMethod::Min | BuiltinMethod::Max => {
            let want_max = method == BuiltinMethod::Max;
            let mut best: Option<Val> = None;
            let mut best_f = 0.0;
            for v in a {
                if !v.is_number() {
                    continue;
                }
                let vf = v.as_f64().unwrap_or(0.0);
                let replace = match best {
                    None => true,
                    Some(_) if want_max => vf > best_f,
                    Some(_) => vf < best_f,
                };
                if replace {
                    best_f = vf;
                    best = Some(v.clone());
                }
            }
            best.unwrap_or(Val::Null)
        }
        _ => Val::Null,
    }
}

