//! String-manipulation helpers shared by builtin implementations.
//! Utility functions for case conversion, base64, and string formatting
//! that would be noise inside the large `builtins.rs` dispatch table.

use crate::value::Val;

/// Split a camelCase, snake_case, or kebab-case string into lowercase words.
/// Used by case-conversion builtins such as `snake_case` and `camel_case`.
pub(crate) fn split_words_lower(s: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut prev_lower = false;
    for c in s.chars() {
        let is_sep = c == '_' || c == '-' || c.is_whitespace();
        if is_sep {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            prev_lower = false;
            continue;
        }
        if c.is_uppercase() && prev_lower {
            out.push(std::mem::take(&mut cur));
        }
        for d in c.to_lowercase() {
            cur.push(d);
        }
        prev_lower = c.is_lowercase();
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Append `p` to `out` with its first Unicode character uppercased.
/// Used by `PascalCase` and `TitleCase` builtins.
pub(crate) fn upper_first_into(p: &str, out: &mut String) {
    let mut chars = p.chars();
    if let Some(f) = chars.next() {
        for u in f.to_uppercase() {
            out.push(u);
        }
        out.push_str(chars.as_str());
    }
}

/// Encode a byte slice to a standard Base64 string with `=` padding.
pub(crate) fn base64_encode(bytes: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((bytes.len() + 2) / 3 * 4);
    let mut i = 0;
    while i < bytes.len() {
        let b0 = bytes[i] as u32;
        let b1 = if i + 1 < bytes.len() {
            bytes[i + 1] as u32
        } else {
            0
        };
        let b2 = if i + 2 < bytes.len() {
            bytes[i + 2] as u32
        } else {
            0
        };
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(CHARS[((n >> 18) & 0x3f) as usize] as char);
        out.push(CHARS[((n >> 12) & 0x3f) as usize] as char);
        out.push(if i + 1 < bytes.len() {
            CHARS[((n >> 6) & 0x3f) as usize] as char
        } else {
            '='
        });
        out.push(if i + 2 < bytes.len() {
            CHARS[(n & 0x3f) as usize] as char
        } else {
            '='
        });
        i += 3;
    }
    out
}

/// Decode a Base64 string (with optional `=` padding) to raw bytes,
/// returning an error message if any character is invalid.
pub(crate) fn base64_decode(s: &str) -> Result<Vec<u8>, String> {
    const DECODE: [i8; 128] = {
        let mut t = [-1i8; 128];
        let chars = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut i = 0usize;
        while i < chars.len() {
            t[chars[i] as usize] = i as i8;
            i += 1;
        }
        t
    };
    let s = s.trim();
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    let mut i = 0;
    while i + 3 < bytes.len() {
        let dc = |c: u8| -> Result<u32, String> {
            if c == b'=' {
                return Ok(0);
            }
            if c as usize >= 128 || DECODE[c as usize] < 0 {
                return Err(format!("invalid base64 char: {}", c as char));
            }
            Ok(DECODE[c as usize] as u32)
        };
        let b0 = dc(bytes[i])?;
        let b1 = dc(bytes[i + 1])?;
        let b2 = dc(bytes[i + 2])?;
        let b3 = dc(bytes[i + 3])?;
        let n = (b0 << 18) | (b1 << 12) | (b2 << 6) | b3;
        out.push(((n >> 16) & 0xff) as u8);
        if bytes[i + 2] != b'=' {
            out.push(((n >> 8) & 0xff) as u8);
        }
        if bytes[i + 3] != b'=' {
            out.push((n & 0xff) as u8);
        }
        i += 4;
    }
    Ok(out)
}

/// Format a single `Val` as a CSV cell, quoting and escaping if the value
/// contains the separator, double quotes, or newlines.
#[inline]
fn csv_cell(v: &Val, sep: &str) -> String {
    use crate::util::val_to_string;
    match v {
        Val::Str(s) if s.contains(sep) || s.contains('"') || s.contains('\n') => {
            format!("\"{}\"", s.replace('"', "\"\""))
        }
        Val::Str(s) => s.to_string(),
        other => val_to_string(other),
    }
}

/// Serialize a `Val` to CSV text using `sep` as the field separator.
/// Arrays of arrays become multi-row CSV; arrays of objects emit object values.
pub(crate) fn csv_emit(val: &Val, sep: &str) -> String {
    match val {
        Val::Arr(rows) => rows
            .iter()
            .map(|row| match row {
                Val::Arr(cells) => cells
                    .iter()
                    .map(|c| csv_cell(c, sep))
                    .collect::<Vec<_>>()
                    .join(sep),
                Val::Obj(m) => m
                    .values()
                    .map(|c| csv_cell(c, sep))
                    .collect::<Vec<_>>()
                    .join(sep),
                v => csv_cell(v, sep),
            })
            .collect::<Vec<_>>()
            .join("\n"),
        Val::Obj(m) => m
            .values()
            .map(|c| csv_cell(c, sep))
            .collect::<Vec<_>>()
            .join(sep),
        v => csv_cell(v, sep),
    }
}

// Per-thread cache that maps pattern strings to compiled `Regex` objects,
// avoiding repeated compilation for the same pattern across builtin calls.
thread_local! {
    static REGEX_CACHE: std::cell::RefCell<std::collections::HashMap<String, std::sync::Arc<regex::Regex>>>
        = std::cell::RefCell::new(std::collections::HashMap::with_capacity(32));
}

/// Compile `pat` into a `Regex`, returning a cached `Arc` clone on subsequent
/// calls with the same pattern. Returns an error string if the pattern is invalid.
pub(crate) fn compile_regex(pat: &str) -> Result<std::sync::Arc<regex::Regex>, String> {
    REGEX_CACHE.with(|cell| {
        let mut m = cell.borrow_mut();
        if let Some(r) = m.get(pat) {
            return Ok(std::sync::Arc::clone(r));
        }
        let re = regex::Regex::new(pat).map_err(|e| format!("regex compile: {}", e))?;
        let arc = std::sync::Arc::new(re);
        m.insert(pat.to_string(), std::sync::Arc::clone(&arc));
        Ok(arc)
    })
}
