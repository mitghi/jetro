//! SIMD byte-scan over raw JSON bytes.
//!
//! Locates `"key":` occurrences in a JSON document without first parsing
//! the document into a tree.  `memchr` (AVX2-internal when available)
//! jumps byte-by-byte to the next structurally relevant character — a
//! `"` outside strings, a `"` or `\\` inside strings — so the scanner
//! traverses the document at near-memory-bandwidth speed.
//!
//! ## When to use
//!
//! For `$..key` (all descendants by name) or `$..find(@.key op lit)`
//! shapes over a **large** JSON document where the caller retained the
//! raw bytes (see `Jetro::from_bytes`).  Skips the tree walk entirely —
//! scan cost is bounded by byte length, not node count.
//!
//! ## When not to use
//!
//! - Document already parsed; raw bytes discarded — fall back to the
//!   tree walker in `eval/mod.rs::collect_desc`.
//! - Document is tiny (< a few KB): `serde_json` per-hit parse cost
//!   overtakes the scan win.
//!
//! ## Correctness
//!
//! The scanner respects JSON string-literal escape rules: an unescaped
//! `"` toggles `in_string`, and a `\\` inside a string skips the next
//! byte.  A needle match must begin at a `"` encountered while
//! `in_string` is *false* — exactly where a JSON object-key literal
//! can legally appear.  Hits inside string values (`"comment":"the
//! \"test\" case"`) are therefore rejected.

use memchr::{memchr, memchr2};
use serde_json::Value;

/// Scan raw JSON `bytes` for every `"key":` occurrence that starts at a
/// structural position (i.e. not inside a string literal).  Returns the
/// byte offset of each matching opening `"` in document order.
pub fn find_key_positions(bytes: &[u8], key: &str) -> Vec<usize> {
    let needle = {
        let mut s = String::with_capacity(key.len() + 3);
        s.push('"');
        s.push_str(key);
        s.push_str("\":");
        s
    };
    let needle_b = needle.as_bytes();
    if needle_b.len() > bytes.len() { return Vec::new(); }

    let mut out = Vec::new();
    let mut i = 0usize;
    let mut in_string = false;
    let mut escape = false;

    while i < bytes.len() {
        if escape {
            escape = false;
            i += 1;
            continue;
        }
        if in_string {
            // SIMD jump to next `\\` or `"` (the only state-changing bytes
            // inside a string literal).
            let rest = &bytes[i..];
            match memchr2(b'\\', b'"', rest) {
                Some(off) => {
                    i += off;
                    match bytes[i] {
                        b'\\' => { escape = true;     i += 1; }
                        b'"'  => { in_string = false; i += 1; }
                        _     => unreachable!(),
                    }
                }
                None => break,
            }
        } else {
            // SIMD jump to next `"` — only positions where a key literal
            // can start (or where a top-level string value can open).
            let rest = &bytes[i..];
            match memchr(b'"', rest) {
                Some(off) => {
                    let q = i + off;
                    if q + needle_b.len() <= bytes.len()
                        && &bytes[q..q + needle_b.len()] == needle_b {
                        out.push(q);
                        // Past `"key":` we remain outside any string,
                        // pointing at the start of the value.
                        i = q + needle_b.len();
                    } else {
                        in_string = true;
                        i = q + 1;
                    }
                }
                None => break,
            }
        }
    }
    out
}

/// Extract every value paired with `key` at any depth.  Uses
/// `find_key_positions` to locate each `"key":` site and then parses the
/// single value that follows via a streaming `serde_json::Deserializer`
/// (stops at the end of the first value — not the whole document).
///
/// Parse failures are silently skipped; they should never arise on
/// valid JSON input, and we refuse to panic on malformed payloads.
pub fn extract_values(bytes: &[u8], key: &str) -> Vec<Value> {
    let positions = find_key_positions(bytes, key);
    let mut out = Vec::with_capacity(positions.len());
    let prefix_len = key.len() + 3; // `"key":`
    for pos in positions {
        let mut start = pos + prefix_len;
        while start < bytes.len()
            && matches!(bytes[start], b' ' | b'\t' | b'\n' | b'\r') {
            start += 1;
        }
        if start >= bytes.len() { continue; }
        let mut stream = serde_json::Deserializer::from_slice(&bytes[start..])
            .into_iter::<Value>();
        if let Some(Ok(v)) = stream.next() {
            out.push(v);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_top_level_key() {
        let doc = br#"{"test": 42, "other": 7}"#;
        let pos = find_key_positions(doc, "test");
        assert_eq!(pos, vec![1]);
    }

    #[test]
    fn finds_nested_keys() {
        let doc = br#"{"a":{"test":1},"b":[{"test":2},{"test":3}]}"#;
        let pos = find_key_positions(doc, "test");
        assert_eq!(pos.len(), 3);
    }

    #[test]
    fn ignores_hits_inside_string_values() {
        let doc = br#"{"comment":"the \"test\": lie","test":99}"#;
        let vals = extract_values(doc, "test");
        assert_eq!(vals, vec![serde_json::json!(99)]);
    }

    #[test]
    fn does_not_match_longer_key_suffix() {
        let doc = br#"{"nottest":1,"test":2}"#;
        let vals = extract_values(doc, "test");
        assert_eq!(vals, vec![serde_json::json!(2)]);
    }

    #[test]
    fn handles_escaped_backslash_then_quote() {
        // Backslash escapes itself: `"c:\\"` closes normally.  The
        // subsequent `"test":` must then be recognised.
        let doc = br#"{"path":"c:\\","test":"ok"}"#;
        let vals = extract_values(doc, "test");
        assert_eq!(vals, vec![serde_json::json!("ok")]);
    }

    #[test]
    fn empty_on_missing_key() {
        let doc = br#"{"a":1,"b":2}"#;
        assert!(find_key_positions(doc, "zzz").is_empty());
        assert!(extract_values(doc, "zzz").is_empty());
    }

    #[test]
    fn extracts_nested_object_value() {
        let doc = br#"{"test":{"nested":[1,2,3]}}"#;
        let vals = extract_values(doc, "test");
        assert_eq!(vals, vec![serde_json::json!({"nested":[1,2,3]})]);
    }

    #[test]
    fn extracts_all_nested_hits_in_order() {
        let doc = br#"{"a":{"test":1},"b":[{"test":2},{"test":3}]}"#;
        let vals = extract_values(doc, "test");
        assert_eq!(vals, vec![
            serde_json::json!(1),
            serde_json::json!(2),
            serde_json::json!(3),
        ]);
    }
}
