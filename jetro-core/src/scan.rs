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
    if needle_b.len() > bytes.len() {
        return Vec::new();
    }

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
                        b'\\' => {
                            escape = true;
                            i += 1;
                        }
                        b'"' => {
                            in_string = false;
                            i += 1;
                        }
                        _ => unreachable!(),
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
                        && &bytes[q..q + needle_b.len()] == needle_b
                    {
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
        while start < bytes.len() && matches!(bytes[start], b' ' | b'\t' | b'\n' | b'\r') {
            start += 1;
        }
        if start >= bytes.len() {
            continue;
        }
        let mut stream = serde_json::Deserializer::from_slice(&bytes[start..]).into_iter::<Value>();
        if let Some(Ok(v)) = stream.next() {
            out.push(v);
        }
    }
    out
}

/// Span of a single JSON value in `bytes`: start offset inclusive,
/// end offset exclusive.  Produced by `find_key_value_spans`; the caller
/// may compare raw bytes against a literal without allocating a `Value`.
#[derive(Debug, Clone, Copy)]
pub struct ValueSpan {
    pub start: usize,
    pub end: usize,
}

/// Early-exit variant of `find_key_value_spans` — returns the first
/// span paired with `key` encountered in document order, or `None` if
/// the key does not appear.  Powers the `Descendant(k) + .first()`
/// fast path: walks only as far as needed to find one match, rather
/// than scanning the entire byte buffer.
pub fn find_first_key_value_span(bytes: &[u8], key: &str) -> Option<ValueSpan> {
    let needle = {
        let mut s = String::with_capacity(key.len() + 3);
        s.push('"');
        s.push_str(key);
        s.push_str("\":");
        s
    };
    let needle_b = needle.as_bytes();
    if needle_b.len() > bytes.len() {
        return None;
    }
    let prefix_len = needle_b.len();

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
            let rest = &bytes[i..];
            match memchr2(b'\\', b'"', rest) {
                Some(off) => {
                    i += off;
                    match bytes[i] {
                        b'\\' => {
                            escape = true;
                            i += 1;
                        }
                        b'"' => {
                            in_string = false;
                            i += 1;
                        }
                        _ => unreachable!(),
                    }
                }
                None => return None,
            }
        } else {
            let rest = &bytes[i..];
            match memchr(b'"', rest) {
                Some(off) => {
                    let q = i + off;
                    if q + prefix_len <= bytes.len() && &bytes[q..q + prefix_len] == needle_b {
                        let mut start = q + prefix_len;
                        while start < bytes.len()
                            && matches!(bytes[start], b' ' | b'\t' | b'\n' | b'\r')
                        {
                            start += 1;
                        }
                        if start >= bytes.len() {
                            return None;
                        }
                        return value_end(bytes, start).map(|end| ValueSpan { start, end });
                    } else {
                        in_string = true;
                        i = q + 1;
                    }
                }
                None => return None,
            }
        }
    }
    None
}

/// Locate the byte span of every value paired with `key`.  Skips
/// whitespace between `:` and the value and then walks the value to its
/// end — strings obey escape rules, containers track nesting depth,
/// scalars run until the next structural terminator.
pub fn find_key_value_spans(bytes: &[u8], key: &str) -> Vec<ValueSpan> {
    let positions = find_key_positions(bytes, key);
    let prefix_len = key.len() + 3;
    let mut out = Vec::with_capacity(positions.len());
    for pos in positions {
        let mut start = pos + prefix_len;
        while start < bytes.len() && matches!(bytes[start], b' ' | b'\t' | b'\n' | b'\r') {
            start += 1;
        }
        if start >= bytes.len() {
            continue;
        }
        if let Some(end) = value_end(bytes, start) {
            out.push(ValueSpan { start, end });
        }
    }
    out
}

/// Extract the parsed `Value` for every `key` site whose raw bytes
/// equal `lit`.  Matches by bytewise equality on the span — safe for
/// JSON primitives (strings, numbers, bools, null) which serialise
/// canonically, not for objects/arrays.  Non-matching sites are
/// skipped without paying the `serde_json` parse cost.
pub fn extract_values_eq(bytes: &[u8], key: &str, lit: &[u8]) -> Vec<Value> {
    let mut out = Vec::new();
    for span in find_key_value_spans(bytes, key) {
        if span.end - span.start == lit.len() && &bytes[span.start..span.end] == lit {
            if let Ok(v) = serde_json::from_slice::<Value>(&bytes[span.start..span.end]) {
                out.push(v);
            }
        }
    }
    out
}

/// Extract every value for `key` **whose raw bytes equal `lit`** after
/// trimming leading whitespace.  `lit` is expected to be pre-serialised
/// JSON (e.g. `br#""action""#`, `b"42"`).  Bytewise comparison is safe
/// for JSON primitives with canonical serialisation; it is *not* correct
/// for objects/arrays where key order or whitespace may differ.
///
/// Skips non-matching sites entirely — no `Value` allocation.
pub fn count_key_value_eq(bytes: &[u8], key: &str, lit: &[u8]) -> usize {
    let mut n = 0;
    for span in find_key_value_spans(bytes, key) {
        if span.end - span.start == lit.len() && &bytes[span.start..span.end] == lit {
            n += 1;
        }
    }
    n
}

/// Locate the byte span of every **enclosing object** whose `key` field
/// equals the canonical-serialised literal `lit`.  Powers the SIMD fast
/// path for `$..find(@.key == lit)`.
///
/// Implementation: single forward pass.  An explicit stack tracks the
/// start position of every currently-open `{`.  When `"key":` is encountered
/// at the top object and the following value bytes equal `lit`, the top
/// frame is flagged.  On matching `}`, flagged frames emit a `ValueSpan`
/// covering the object.  Output is sorted by `start` so order matches the
/// DFS pre-order of the tree walker.
///
/// Bytewise literal comparison is safe for JSON primitives (int, string,
/// bool, null) because they serialise canonically.  It is **not** correct
/// for floats (`1.0` / `1` representation variance) or structured values
/// (object key order / whitespace) — callers must reject those literals.
pub fn find_enclosing_objects_eq(bytes: &[u8], key: &str, lit: &[u8]) -> Vec<ValueSpan> {
    let needle = {
        let mut s = String::with_capacity(key.len() + 3);
        s.push('"');
        s.push_str(key);
        s.push_str("\":");
        s
    };
    let needle_b = needle.as_bytes();
    let mut out = Vec::new();
    let mut stack: Vec<(usize, bool)> = Vec::new();
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
            let rest = &bytes[i..];
            match memchr2(b'\\', b'"', rest) {
                Some(off) => {
                    i += off;
                    match bytes[i] {
                        b'\\' => {
                            escape = true;
                            i += 1;
                        }
                        b'"' => {
                            in_string = false;
                            i += 1;
                        }
                        _ => unreachable!(),
                    }
                }
                None => break,
            }
            continue;
        }
        match bytes[i] {
            b'{' => {
                stack.push((i, false));
                i += 1;
            }
            b'}' => {
                if let Some((start, matched)) = stack.pop() {
                    if matched {
                        out.push(ValueSpan { start, end: i + 1 });
                    }
                }
                i += 1;
            }
            b'"' => {
                if i + needle_b.len() <= bytes.len() && &bytes[i..i + needle_b.len()] == needle_b {
                    let mut vs = i + needle_b.len();
                    while vs < bytes.len() && matches!(bytes[vs], b' ' | b'\t' | b'\n' | b'\r') {
                        vs += 1;
                    }
                    if let Some(ve) = value_end(bytes, vs) {
                        if ve - vs == lit.len() && &bytes[vs..ve] == lit {
                            if let Some(top) = stack.last_mut() {
                                top.1 = true;
                            }
                        }
                        i = ve;
                    } else {
                        i = vs;
                    }
                } else {
                    in_string = true;
                    i += 1;
                }
            }
            _ => i += 1,
        }
    }

    out.sort_by_key(|s| s.start);
    out
}

/// Like `find_enclosing_objects_eq` but accepts N `(key, lit)` conjuncts.
/// An object is emitted iff it *directly* contains every listed key with
/// the matching canonical literal value.  Each frame carries a bitmask of
/// which conjuncts have matched so far (max 64 conjuncts).
pub fn find_enclosing_objects_eq_multi(
    bytes: &[u8],
    conjuncts: &[(String, Vec<u8>)],
) -> Vec<ValueSpan> {
    assert!(conjuncts.len() <= 64, "at most 64 conjuncts supported");
    if conjuncts.is_empty() {
        return Vec::new();
    }

    // Pre-build each needle as `"<key>":` so we compare against a
    // contiguous slice at the current cursor.
    let needles: Vec<Vec<u8>> = conjuncts
        .iter()
        .map(|(k, _)| {
            let mut s = Vec::with_capacity(k.len() + 3);
            s.push(b'"');
            s.extend_from_slice(k.as_bytes());
            s.extend_from_slice(b"\":");
            s
        })
        .collect();
    let full_mask: u64 = if conjuncts.len() == 64 {
        u64::MAX
    } else {
        (1u64 << conjuncts.len()) - 1
    };

    let mut out = Vec::new();
    let mut stack: Vec<(usize, u64)> = Vec::new();
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
            let rest = &bytes[i..];
            match memchr2(b'\\', b'"', rest) {
                Some(off) => {
                    i += off;
                    match bytes[i] {
                        b'\\' => {
                            escape = true;
                            i += 1;
                        }
                        b'"' => {
                            in_string = false;
                            i += 1;
                        }
                        _ => unreachable!(),
                    }
                }
                None => break,
            }
            continue;
        }
        match bytes[i] {
            b'{' => {
                stack.push((i, 0u64));
                i += 1;
            }
            b'}' => {
                if let Some((start, mask)) = stack.pop() {
                    if mask == full_mask {
                        out.push(ValueSpan { start, end: i + 1 });
                    }
                }
                i += 1;
            }
            b'"' => {
                let mut matched_idx: Option<usize> = None;
                for (idx, nb) in needles.iter().enumerate() {
                    if i + nb.len() <= bytes.len() && &bytes[i..i + nb.len()] == &nb[..] {
                        matched_idx = Some(idx);
                        break;
                    }
                }
                if let Some(idx) = matched_idx {
                    let nb = &needles[idx];
                    let mut vs = i + nb.len();
                    while vs < bytes.len() && matches!(bytes[vs], b' ' | b'\t' | b'\n' | b'\r') {
                        vs += 1;
                    }
                    if let Some(ve) = value_end(bytes, vs) {
                        let lit = &conjuncts[idx].1;
                        if ve - vs == lit.len() && &bytes[vs..ve] == &lit[..] {
                            if let Some(top) = stack.last_mut() {
                                top.1 |= 1u64 << idx;
                            }
                        }
                        i = ve;
                    } else {
                        i = vs;
                    }
                } else {
                    in_string = true;
                    i += 1;
                }
            }
            _ => i += 1,
        }
    }

    out.sort_by_key(|s| s.start);
    out
}

/// Comparison operator for numeric-range byte scans.  Mirrors the subset of
/// `ast::BinOp` that makes sense against a canonical JSON number literal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanCmp {
    Lt,
    Lte,
    Gt,
    Gte,
}

impl ScanCmp {
    #[inline]
    fn holds(self, lhs: f64, rhs: f64) -> bool {
        match self {
            ScanCmp::Lt => lhs < rhs,
            ScanCmp::Lte => lhs <= rhs,
            ScanCmp::Gt => lhs > rhs,
            ScanCmp::Gte => lhs >= rhs,
        }
    }
}

/// A single predicate against the value paired with a key inside an
/// enclosing object.  Drives `find_enclosing_objects_mixed`.
#[derive(Debug, Clone)]
pub enum ScanPred {
    /// Bytewise equality against a canonical JSON literal (int/string/
    /// bool/null — same shape as `find_enclosing_objects_eq_multi`).
    Eq(Vec<u8>),
    /// Numeric comparison: value parsed as f64 then `op` applied vs
    /// `threshold`.  Non-numeric values do not match.
    Cmp(ScanCmp, f64),
}

/// Locate the byte span of every enclosing object whose `key` field is a
/// JSON number satisfying `op threshold`.  Powers the fast path for
/// `$..find(@.key op num)` where `op` ∈ `<`, `<=`, `>`, `>=`.
///
/// Matches the shape of `find_enclosing_objects_eq` — single forward pass,
/// stack of opened `{` frames, flag-on-match, emit-on-close, output sorted
/// by start offset so order mirrors the tree walker's DFS pre-order.
///
/// The value is byte-parsed via `parse_num_span`.  Non-numeric values and
/// malformed numbers are skipped (the conjunct simply doesn't fire).
pub fn find_enclosing_objects_cmp(
    bytes: &[u8],
    key: &str,
    op: ScanCmp,
    threshold: f64,
) -> Vec<ValueSpan> {
    let needle = {
        let mut s = String::with_capacity(key.len() + 3);
        s.push('"');
        s.push_str(key);
        s.push_str("\":");
        s
    };
    let needle_b = needle.as_bytes();
    let mut out = Vec::new();
    let mut stack: Vec<(usize, bool)> = Vec::new();
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
            let rest = &bytes[i..];
            match memchr2(b'\\', b'"', rest) {
                Some(off) => {
                    i += off;
                    match bytes[i] {
                        b'\\' => {
                            escape = true;
                            i += 1;
                        }
                        b'"' => {
                            in_string = false;
                            i += 1;
                        }
                        _ => unreachable!(),
                    }
                }
                None => break,
            }
            continue;
        }
        match bytes[i] {
            b'{' => {
                stack.push((i, false));
                i += 1;
            }
            b'}' => {
                if let Some((start, matched)) = stack.pop() {
                    if matched {
                        out.push(ValueSpan { start, end: i + 1 });
                    }
                }
                i += 1;
            }
            b'"' => {
                if i + needle_b.len() <= bytes.len() && &bytes[i..i + needle_b.len()] == needle_b {
                    let mut vs = i + needle_b.len();
                    while vs < bytes.len() && matches!(bytes[vs], b' ' | b'\t' | b'\n' | b'\r') {
                        vs += 1;
                    }
                    if let Some(ve) = value_end(bytes, vs) {
                        if let Some((_, as_f, _)) = parse_num_span(&bytes[vs..ve]) {
                            if op.holds(as_f, threshold) {
                                if let Some(top) = stack.last_mut() {
                                    top.1 = true;
                                }
                            }
                        }
                        i = ve;
                    } else {
                        i = vs;
                    }
                } else {
                    in_string = true;
                    i += 1;
                }
            }
            _ => i += 1,
        }
    }

    out.sort_by_key(|s| s.start);
    out
}

/// Extract the span of the *direct* child named `key` inside an object
/// whose bytes span is `obj_bytes[0] == b'{'`.  Depth-aware: matches
/// only keys at the top level of the object, not keys nested inside
/// arrays or sub-objects.  Returned span is relative to `obj_bytes`.
pub fn find_direct_field(obj_bytes: &[u8], key: &str) -> Option<ValueSpan> {
    if obj_bytes.is_empty() || obj_bytes[0] != b'{' {
        return None;
    }
    let needle = {
        let mut s = String::with_capacity(key.len() + 3);
        s.push('"');
        s.push_str(key);
        s.push_str("\":");
        s
    };
    let needle_b = needle.as_bytes();
    let mut depth: usize = 0;
    let mut i = 0usize;
    let mut in_string = false;
    let mut escape = false;
    while i < obj_bytes.len() {
        if escape {
            escape = false;
            i += 1;
            continue;
        }
        if in_string {
            match memchr2(b'\\', b'"', &obj_bytes[i..]) {
                Some(off) => {
                    i += off;
                    match obj_bytes[i] {
                        b'\\' => {
                            escape = true;
                            i += 1;
                        }
                        b'"' => {
                            in_string = false;
                            i += 1;
                        }
                        _ => unreachable!(),
                    }
                }
                None => return None,
            }
            continue;
        }
        match obj_bytes[i] {
            b'{' | b'[' => {
                depth += 1;
                i += 1;
            }
            b'}' | b']' => {
                if depth == 0 {
                    return None;
                }
                depth -= 1;
                i += 1;
            }
            b'"' => {
                if depth == 1
                    && i + needle_b.len() <= obj_bytes.len()
                    && &obj_bytes[i..i + needle_b.len()] == needle_b
                {
                    let mut vs = i + needle_b.len();
                    while vs < obj_bytes.len()
                        && matches!(obj_bytes[vs], b' ' | b'\t' | b'\n' | b'\r')
                    {
                        vs += 1;
                    }
                    return value_end(obj_bytes, vs).map(|end| ValueSpan { start: vs, end });
                }
                in_string = true;
                i += 1;
            }
            _ => i += 1,
        }
    }
    None
}

/// Mixed multi-conjunct scan: each conjunct is `(key, ScanPred)` and an
/// enclosing object is emitted iff every conjunct matches on the same
/// `{...}` frame.  Generalises `find_enclosing_objects_eq_multi` to allow
/// equality literals and numeric-range comparisons in the same query.
/// Frames carry a bitmask of satisfied conjuncts (max 64).
pub fn find_enclosing_objects_mixed(
    bytes: &[u8],
    conjuncts: &[(String, ScanPred)],
) -> Vec<ValueSpan> {
    assert!(conjuncts.len() <= 64, "at most 64 conjuncts supported");
    if conjuncts.is_empty() {
        return Vec::new();
    }

    let needles: Vec<Vec<u8>> = conjuncts
        .iter()
        .map(|(k, _)| {
            let mut s = Vec::with_capacity(k.len() + 3);
            s.push(b'"');
            s.extend_from_slice(k.as_bytes());
            s.extend_from_slice(b"\":");
            s
        })
        .collect();
    let full_mask: u64 = if conjuncts.len() == 64 {
        u64::MAX
    } else {
        (1u64 << conjuncts.len()) - 1
    };

    let mut out = Vec::new();
    let mut stack: Vec<(usize, u64)> = Vec::new();
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
            let rest = &bytes[i..];
            match memchr2(b'\\', b'"', rest) {
                Some(off) => {
                    i += off;
                    match bytes[i] {
                        b'\\' => {
                            escape = true;
                            i += 1;
                        }
                        b'"' => {
                            in_string = false;
                            i += 1;
                        }
                        _ => unreachable!(),
                    }
                }
                None => break,
            }
            continue;
        }
        match bytes[i] {
            b'{' => {
                stack.push((i, 0u64));
                i += 1;
            }
            b'}' => {
                if let Some((start, mask)) = stack.pop() {
                    if mask == full_mask {
                        out.push(ValueSpan { start, end: i + 1 });
                    }
                }
                i += 1;
            }
            b'"' => {
                let mut matched_idx: Option<usize> = None;
                for (idx, nb) in needles.iter().enumerate() {
                    if i + nb.len() <= bytes.len() && &bytes[i..i + nb.len()] == &nb[..] {
                        matched_idx = Some(idx);
                        break;
                    }
                }
                if let Some(idx) = matched_idx {
                    let nb = &needles[idx];
                    let mut vs = i + nb.len();
                    while vs < bytes.len() && matches!(bytes[vs], b' ' | b'\t' | b'\n' | b'\r') {
                        vs += 1;
                    }
                    if let Some(ve) = value_end(bytes, vs) {
                        let fires = match &conjuncts[idx].1 {
                            ScanPred::Eq(lit) => ve - vs == lit.len() && &bytes[vs..ve] == &lit[..],
                            ScanPred::Cmp(op, thresh) => parse_num_span(&bytes[vs..ve])
                                .map(|(_, f, _)| op.holds(f, *thresh))
                                .unwrap_or(false),
                        };
                        if fires {
                            if let Some(top) = stack.last_mut() {
                                top.1 |= 1u64 << idx;
                            }
                        }
                        i = ve;
                    } else {
                        i = vs;
                    }
                } else {
                    in_string = true;
                    i += 1;
                }
            }
            _ => i += 1,
        }
    }

    out.sort_by_key(|s| s.start);
    out
}

/// Fold numeric values over `spans` into `(int_sum, float_sum, is_float, n)`.
/// Integer spans accumulate into `int_sum`; a single float promotes the
/// whole fold to `float_sum` (which tracks the running total as f64).
/// Spans that don't parse as numbers are skipped.
#[derive(Debug, Clone, Copy, Default)]
pub struct NumFold {
    pub int_sum: i64,
    pub float_sum: f64,
    pub is_float: bool,
    pub count: usize,
    pub min_i: i64,
    pub max_i: i64,
    pub min_f: f64,
    pub max_f: f64,
    pub any: bool,
}

/// Parse a span of JSON numeric bytes. Returns Some((as_i64, as_f64, is_int))
/// or None if not a valid number. Canonical JSON numbers only: `-?\d+(\.\d+)?(e±\d+)?`.
#[inline]
pub fn parse_num_span(s: &[u8]) -> Option<(i64, f64, bool)> {
    let s = std::str::from_utf8(s).ok()?;
    // Integer-looking path (no '.', 'e', 'E') — try i64 first.
    let has_frac_or_exp = s.bytes().any(|b| matches!(b, b'.' | b'e' | b'E'));
    if !has_frac_or_exp {
        if let Ok(n) = s.parse::<i64>() {
            return Some((n, n as f64, true));
        }
    }
    s.parse::<f64>().ok().map(|f| (f as i64, f, false))
}

/// Fold numeric spans for sum/avg/min/max. Walks each span, parses as
/// number, updates the accumulators. Non-numeric spans are skipped.
pub fn fold_nums(bytes: &[u8], spans: &[ValueSpan]) -> NumFold {
    let mut f = NumFold::default();
    for s in spans {
        let slice = &bytes[s.start..s.end];
        let Some((i, x, is_int)) = parse_num_span(slice) else {
            continue;
        };
        f.count += 1;
        if !f.any {
            f.any = true;
            f.min_i = i;
            f.max_i = i;
            f.min_f = x;
            f.max_f = x;
        } else {
            if i < f.min_i {
                f.min_i = i;
            }
            if i > f.max_i {
                f.max_i = i;
            }
            if x < f.min_f {
                f.min_f = x;
            }
            if x > f.max_f {
                f.max_f = x;
            }
        }
        if is_int && !f.is_float {
            f.int_sum = f.int_sum.wrapping_add(i);
            f.float_sum += x;
        } else {
            if !f.is_float {
                f.float_sum = f.int_sum as f64;
                f.is_float = true;
            }
            f.float_sum += x;
        }
    }
    f
}

/// Fold the direct child named `key` of each enclosing object span
/// into a single `NumFold`.  Combines `find_direct_field` +
/// `parse_num_span` without materialising any intermediate `Val`.
/// Spans missing the key or whose value is non-numeric are skipped.
pub fn fold_direct_field_nums(bytes: &[u8], spans: &[ValueSpan], key: &str) -> NumFold {
    let mut f = NumFold::default();
    for s in spans {
        let obj_bytes = &bytes[s.start..s.end];
        let Some(vs) = find_direct_field(obj_bytes, key) else {
            continue;
        };
        let Some((i, x, is_int)) = parse_num_span(&obj_bytes[vs.start..vs.end]) else {
            continue;
        };
        f.count += 1;
        if !f.any {
            f.any = true;
            f.min_i = i;
            f.max_i = i;
            f.min_f = x;
            f.max_f = x;
        } else {
            if i < f.min_i {
                f.min_i = i;
            }
            if i > f.max_i {
                f.max_i = i;
            }
            if x < f.min_f {
                f.min_f = x;
            }
            if x > f.max_f {
                f.max_f = x;
            }
        }
        if is_int && !f.is_float {
            f.int_sum = f.int_sum.wrapping_add(i);
            f.float_sum += x;
        } else {
            if !f.is_float {
                f.float_sum = f.int_sum as f64;
                f.is_float = true;
            }
            f.float_sum += x;
        }
    }
    f
}

/// Walk a JSON value starting at `start`, return the exclusive end offset.
/// Returns `None` on malformed input (missing close, truncated literal).
fn value_end(bytes: &[u8], start: usize) -> Option<usize> {
    if start >= bytes.len() {
        return None;
    }
    match bytes[start] {
        b'"' => {
            // Walk the string respecting escapes.
            let mut i = start + 1;
            let mut escape = false;
            while i < bytes.len() {
                if escape {
                    escape = false;
                    i += 1;
                    continue;
                }
                match bytes[i] {
                    b'\\' => {
                        escape = true;
                        i += 1;
                    }
                    b'"' => return Some(i + 1),
                    _ => i += 1,
                }
            }
            None
        }
        b'{' | b'[' => {
            let open = bytes[start];
            let close = if open == b'{' { b'}' } else { b']' };
            let mut depth = 1usize;
            let mut i = start + 1;
            let mut in_string = false;
            let mut escape = false;
            while i < bytes.len() {
                let b = bytes[i];
                if escape {
                    escape = false;
                    i += 1;
                    continue;
                }
                if in_string {
                    match b {
                        b'\\' => escape = true,
                        b'"' => in_string = false,
                        _ => {}
                    }
                    i += 1;
                    continue;
                }
                match b {
                    b'"' => in_string = true,
                    c if c == open => depth += 1,
                    c if c == close => {
                        depth -= 1;
                        if depth == 0 {
                            return Some(i + 1);
                        }
                    }
                    _ => {}
                }
                i += 1;
            }
            None
        }
        _ => {
            // Scalar (number / bool / null) — scan until structural terminator.
            let mut i = start;
            while i < bytes.len() {
                match bytes[i] {
                    b',' | b'}' | b']' | b' ' | b'\t' | b'\n' | b'\r' => break,
                    _ => i += 1,
                }
            }
            if i == start {
                None
            } else {
                Some(i)
            }
        }
    }
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
        assert_eq!(
            vals,
            vec![
                serde_json::json!(1),
                serde_json::json!(2),
                serde_json::json!(3),
            ]
        );
    }

    #[test]
    fn spans_cover_every_value_kind() {
        let doc = br#"{"a":1,"b":"two","c":[1,2,3],"d":{"x":1},"e":true}"#;
        let keys_and_expected: &[(&str, &[u8])] = &[
            ("a", b"1"),
            ("b", b"\"two\""),
            ("c", b"[1,2,3]"),
            ("d", b"{\"x\":1}"),
            ("e", b"true"),
        ];
        for (k, want) in keys_and_expected {
            let spans = find_key_value_spans(doc, k);
            assert_eq!(spans.len(), 1, "key {} not found", k);
            assert_eq!(&doc[spans[0].start..spans[0].end], *want);
        }
    }

    #[test]
    fn count_eq_matches_only_literal_equals() {
        let doc = br#"{"a":[{"type":"action"},{"type":"idle"},{"type":"action"},{"type":"noop"}]}"#;
        assert_eq!(count_key_value_eq(doc, "type", br#""action""#), 2);
        assert_eq!(count_key_value_eq(doc, "type", br#""missing""#), 0);
    }

    #[test]
    fn count_eq_numeric_literal() {
        let doc = br#"{"xs":[{"n":10},{"n":42},{"n":10},{"n":42}]}"#;
        assert_eq!(count_key_value_eq(doc, "n", b"42"), 2);
        assert_eq!(count_key_value_eq(doc, "n", b"10"), 2);
    }

    #[test]
    fn spans_skip_whitespace_after_colon() {
        let doc = br#"{"a":   42   ,"b":  "x"}"#;
        let a = find_key_value_spans(doc, "a");
        assert_eq!(&doc[a[0].start..a[0].end], b"42");
        let b = find_key_value_spans(doc, "b");
        assert_eq!(&doc[b[0].start..b[0].end], b"\"x\"");
    }

    #[test]
    fn enclosing_object_simple_match() {
        let doc = br#"{"events":[{"type":"action","id":1},{"type":"idle","id":2},{"type":"action","id":3}]}"#;
        let spans = find_enclosing_objects_eq(doc, "type", br#""action""#);
        assert_eq!(spans.len(), 2);
        let objs: Vec<_> = spans
            .iter()
            .map(|s| serde_json::from_slice::<serde_json::Value>(&doc[s.start..s.end]).unwrap())
            .collect();
        assert_eq!(objs[0], serde_json::json!({"type":"action","id":1}));
        assert_eq!(objs[1], serde_json::json!({"type":"action","id":3}));
    }

    #[test]
    fn enclosing_object_nested_both_match() {
        // Outer and inner object both have type:"x" — both must be emitted
        // in start-offset order (matches tree walker DFS pre-order).
        let doc = br#"{"type":"x","child":{"type":"x","n":2}}"#;
        let spans = find_enclosing_objects_eq(doc, "type", br#""x""#);
        assert_eq!(spans.len(), 2);
        assert!(spans[0].start < spans[1].start);
        assert_eq!(&doc[spans[0].start..spans[0].end], doc);
        assert_eq!(&doc[spans[1].start..spans[1].end], br#"{"type":"x","n":2}"#);
    }

    #[test]
    fn enclosing_object_nested_inner_only() {
        let doc = br#"{"type":"a","child":{"type":"b","n":2}}"#;
        let spans = find_enclosing_objects_eq(doc, "type", br#""b""#);
        assert_eq!(spans.len(), 1);
        assert_eq!(&doc[spans[0].start..spans[0].end], br#"{"type":"b","n":2}"#);
    }

    #[test]
    fn enclosing_object_ignores_string_value_containing_needle() {
        let doc = br#"{"comment":"the \"type\":\"action\" label","events":[{"type":"action"}]}"#;
        let spans = find_enclosing_objects_eq(doc, "type", br#""action""#);
        assert_eq!(spans.len(), 1);
        assert_eq!(&doc[spans[0].start..spans[0].end], br#"{"type":"action"}"#);
    }

    #[test]
    fn enclosing_object_numeric_literal() {
        let doc = br#"[{"v":10},{"v":42},{"v":42}]"#;
        let spans = find_enclosing_objects_eq(doc, "v", b"42");
        assert_eq!(spans.len(), 2);
    }

    #[test]
    fn enclosing_object_no_match() {
        let doc = br#"{"xs":[{"v":1},{"v":2}]}"#;
        let spans = find_enclosing_objects_eq(doc, "v", b"99");
        assert!(spans.is_empty());
    }

    #[test]
    fn enclosing_object_multi_and_both_match() {
        let doc = br#"[{"t":"a","v":1},{"t":"a","v":2},{"t":"b","v":1}]"#;
        let c = vec![
            ("t".to_string(), br#""a""#.to_vec()),
            ("v".to_string(), b"1".to_vec()),
        ];
        let spans = find_enclosing_objects_eq_multi(doc, &c);
        assert_eq!(spans.len(), 1);
        assert_eq!(&doc[spans[0].start..spans[0].end], br#"{"t":"a","v":1}"#);
    }

    #[test]
    fn enclosing_object_multi_and_nested_propagates() {
        // Child must match on its own; parent's fields don't leak inward.
        let doc = br#"{"t":"a","child":{"t":"a","v":1},"v":1}"#;
        let c = vec![
            ("t".to_string(), br#""a""#.to_vec()),
            ("v".to_string(), b"1".to_vec()),
        ];
        let spans = find_enclosing_objects_eq_multi(doc, &c);
        assert_eq!(spans.len(), 2);
        assert!(spans[0].start < spans[1].start);
    }

    #[test]
    fn enclosing_object_multi_and_partial_no_match() {
        // Only one conjunct matches → no emit.
        let doc = br#"[{"t":"a","v":2},{"t":"b","v":1}]"#;
        let c = vec![
            ("t".to_string(), br#""a""#.to_vec()),
            ("v".to_string(), b"1".to_vec()),
        ];
        let spans = find_enclosing_objects_eq_multi(doc, &c);
        assert!(spans.is_empty());
    }
}
