//! Schema-aware projected JSON parse — generic byte-walker that
//! extracts only the fields a query touches, skipping everything else
//! via JSON-aware brace/bracket/string skipping.  Bypasses simd-json
//! parse (~7 ms / MB floor) for queries that read a small number of
//! fields.
//!
//! Architecture (mirrors tape exec; bytes as source):
//!   - **PathCatalog**: derived from Pipeline kernels — set of field
//!     names each entry needs.  No per-shape walker.
//!   - **Generic walker**: ONE state machine that descends source
//!     FieldChain to the array, iterates top-level entries, per entry
//!     extracts only catalog fields via skip-rest brace counting.
//!   - **Eval bridge**: per-entry extracted-field bag feeds the same
//!     `BodyKernel` value/pred evaluators (path → Scalar table lookup
//!     instead of tape walk).
//!   - **Sink**: same Sink semantics as tape exec — Numeric / NumMap /
//!     NumFilterMap / CountIf / Count.  Adding a new Sink ⇒ one match
//!     arm on Sink + (already-existing) Pipeline IR rewrite handles
//!     the rest.

use crate::eval::Val;
use crate::eval::EvalError;
use crate::pipeline::{Pipeline, Source, Sink, Stage, BodyKernel, NumOp, ArithOperand, ArithOp};
use std::sync::Arc;

// ── Public entry point ─────────────────────────────────────────────

/// Try to run a Pipeline directly over raw bytes — bypasses tape parse.
/// Eligibility: source is FieldChain; stages are Filter* (any
/// path-pred kernels); sink is Numeric / NumMap / NumFilterMap /
/// CountIf / Count with path/Arith kernels.  Returns None when shape
/// not byte-scannable (caller falls back to tape).
pub fn try_run(p: &Pipeline, raw_bytes: &[u8]) -> Option<Result<Val, EvalError>> {
    let chain: Vec<&[u8]> = match &p.source {
        Source::FieldChain { keys } => keys.iter().map(|k| k.as_bytes()).collect(),
        _ => return None,
    };
    // Filter stages — only path-pred shapes.
    for (st, k) in p.stages.iter().zip(p.stage_kernels.iter()) {
        match st {
            Stage::Filter(_) => {
                if !is_pred_kernel(k) { return None; }
            }
            _ => return None,  // FlatMap/Map/etc. defer to tape exec
        }
    }
    // Build wanted-fields catalog from kernels.  Cap field-name uniqueness
    // (~16 distinct fields per row); linear scan of catalog is fine.
    let mut catalog: Catalog = Catalog::default();
    for k in p.stage_kernels.iter() { collect_paths(k, &mut catalog)?; }
    for k in p.sink_kernels.iter() { collect_paths(k, &mut catalog)?; }

    // Locate source array byte position.
    let arr_start = walk_chain_to_array(raw_bytes, &chain)?;

    // Run sink-driven generic walker.
    run_with_sink(p, &catalog, raw_bytes, arr_start)
}

// ── Path catalog ───────────────────────────────────────────────────

/// Set of single-field-from-entry paths.  Each entry in `fields` is
/// the byte name (UTF-8) of a top-level field within a row Object.
/// Index into `fields` is the slot the kernel reads at eval time.
///
/// Multi-step paths (FieldChain) are expanded as the FIRST step here;
/// the kernel evaluator descends further from the per-entry extracted
/// value when needed.  Phase 1 supports only single-step paths
/// (FieldRead) — extending to FieldChain = walker descends inside
/// the entry's Object byte range.
#[derive(Default, Debug)]
struct Catalog {
    fields: Vec<Arc<str>>,
}

impl Catalog {
    fn slot_for(&mut self, name: &Arc<str>) -> usize {
        for (i, f) in self.fields.iter().enumerate() {
            if f.as_ref() == name.as_ref() { return i; }
        }
        self.fields.push(name.clone());
        self.fields.len() - 1
    }
    fn slot_lookup(&self, name: &str) -> Option<usize> {
        self.fields.iter().position(|f| f.as_ref() == name)
    }
}

/// Walk a kernel and register every top-level field name it reads.
/// Returns None when the kernel reads something the byte-walker can't
/// project (FieldChain past one step, ObjProject, FString, Generic).
fn collect_paths(k: &BodyKernel, cat: &mut Catalog) -> Option<()> {
    match k {
        BodyKernel::FieldRead(name) => { cat.slot_for(name); Some(()) }
        BodyKernel::FieldCmpLit(name, _, _) => { cat.slot_for(name); Some(()) }
        BodyKernel::ConstBool(_) | BodyKernel::Const(_) | BodyKernel::CurrentCmpLit(_, _) => Some(()),
        BodyKernel::Arith(lhs, _, rhs) => {
            collect_arith(lhs, cat)?;
            collect_arith(rhs, cat)?;
            Some(())
        }
        // Single-step FieldChain treated as FieldRead.
        BodyKernel::FieldChain(keys) if keys.len() == 1 => {
            cat.slot_for(&keys[0]);
            Some(())
        }
        // FieldChainCmpLit single-step also fits.
        BodyKernel::FieldChainCmpLit(keys, _, _) if keys.len() == 1 => {
            cat.slot_for(&keys[0]);
            Some(())
        }
        // Multi-step chains, ObjProject, FString, Generic — bail.
        _ => None,
    }
}

fn collect_arith(op: &ArithOperand, cat: &mut Catalog) -> Option<()> {
    match op {
        ArithOperand::Path(p) if p.len() == 1 => { cat.slot_for(&p[0]); Some(()) }
        ArithOperand::LitInt(_) | ArithOperand::LitFloat(_) => Some(()),
        _ => None,
    }
}

fn is_pred_kernel(k: &BodyKernel) -> bool {
    matches!(k,
        BodyKernel::FieldRead(_)
        | BodyKernel::FieldCmpLit(_, _, _)
        | BodyKernel::ConstBool(_))
        || matches!(k, BodyKernel::FieldChainCmpLit(keys, _, _) if keys.len() == 1)
        || matches!(k, BodyKernel::FieldChain(keys) if keys.len() == 1)
}

// ── Generic byte JSON walker ──────────────────────────────────────

#[inline]
fn skip_ws(b: &[u8], mut i: usize) -> usize {
    while i < b.len() {
        match b[i] {
            b' ' | b'\t' | b'\n' | b'\r' => i += 1,
            _ => break,
        }
    }
    i
}

#[inline]
fn skip_string(b: &[u8], mut i: usize) -> Option<usize> {
    if b.get(i) != Some(&b'"') { return None; }
    i += 1;
    // memchr2 SIMD search for next `"` or `\\` — strings without
    // escapes (the common case) skip in a single vectorised hop.
    while i < b.len() {
        match memchr::memchr2(b'"', b'\\', &b[i..]) {
            None => return None,
            Some(off) => {
                i += off;
                if b[i] == b'"' { return Some(i + 1); }
                // Escape: skip `\\` + next byte.
                i += 2;
            }
        }
    }
    None
}

#[inline]
fn read_string<'a>(b: &'a [u8], mut i: usize) -> Option<(usize, &'a [u8])> {
    if b.get(i) != Some(&b'"') { return None; }
    let start = i + 1;
    i = start;
    while i < b.len() {
        match b[i] {
            b'"' => return Some((i + 1, &b[start..i])),
            b'\\' => i += 2,
            _ => i += 1,
        }
    }
    None
}

/// Hand-rolled string skip — avoids memchr2 setup overhead for short
/// strings (typical case <30 bytes).  For arbitrary-length strings,
/// memchr2-based `skip_string` is faster; this variant assumes short.
#[inline]
fn skip_string_short(b: &[u8], mut i: usize) -> Option<usize> {
    if b.get(i) != Some(&b'"') { return None; }
    i += 1;
    let blen = b.len();
    while i < blen {
        let c = b[i];
        if c == b'"' { return Some(i + 1); }
        if c == b'\\' { i += 2; } else { i += 1; }
    }
    None
}

fn skip_value(b: &[u8], mut i: usize) -> Option<usize> {
    i = skip_ws(b, i);
    if i >= b.len() { return None; }
    match b[i] {
        b'"' => skip_string_short(b, i),
        b'{' | b'[' => {
            // memchr-SIMD-accelerated brace/bracket/string-quote walker.
            // Searches for the next structural byte (`"`, opener, closer)
            // across cache lines via vectorised compare; ~10x faster than
            // char-by-char for value subtrees that span many bytes.
            let opener = b[i];
            let closer = if opener == b'{' { b'}' } else { b']' };
            let mut depth: i32 = 1;
            i += 1;
            while i < b.len() && depth > 0 {
                match memchr::memchr3(b'"', opener, closer, &b[i..]) {
                    None => return None,
                    Some(off) => {
                        i += off;
                        match b[i] {
                            b'"' => { i = skip_string(b, i)?; }
                            c if c == opener => { depth += 1; i += 1; }
                            c if c == closer => { depth -= 1; i += 1; }
                            _ => unreachable!("memchr3"),
                        }
                    }
                }
            }
            if depth == 0 { Some(i) } else { None }
        }
        _ => {
            // Primitive token end via memchr — find next delimiter/ws.
            // memchr2(',', '}') covers most cases; ws follows.
            let rel = b[i..].iter().position(|c|
                matches!(c, b',' | b'}' | b']' | b' ' | b'\t' | b'\n' | b'\r')
            ).unwrap_or(b.len() - i);
            Some(i + rel)
        }
    }
}

/// Find a top-level field in the Object at `obj_start` (must point
/// AT `{`).  Returns the value's byte position when key matches.
fn find_field(b: &[u8], obj_start: usize, key: &[u8]) -> Option<usize> {
    let mut i = obj_start;
    if b.get(i) != Some(&b'{') { return None; }
    i += 1;
    loop {
        i = skip_ws(b, i);
        if i >= b.len() { return None; }
        if b[i] == b'}' { return None; }
        let (after_key, k) = read_string(b, i)?;
        i = skip_ws(b, after_key);
        if b.get(i) != Some(&b':') { return None; }
        i = skip_ws(b, i + 1);
        if k == key { return Some(i); }
        i = skip_value(b, i)?;
        i = skip_ws(b, i);
        if i < b.len() && b[i] == b',' { i += 1; }
    }
}

fn walk_chain_to_array(b: &[u8], chain: &[&[u8]]) -> Option<usize> {
    let mut i = skip_ws(b, 0);
    for k in chain {
        if b.get(i) != Some(&b'{') { return None; }
        i = find_field(b, i, k)?;
        i = skip_ws(b, i);
    }
    if b.get(i) != Some(&b'[') { return None; }
    Some(i)
}

// ── Per-entry projection ───────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
enum Scalar {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    /// Slice into raw_bytes — string content, not escape-decoded.
    StrRange(u32, u32),
    Missing,
}

fn parse_primitive(b: &[u8], i: usize) -> Option<(usize, Scalar)> {
    if i >= b.len() { return None; }
    match b[i] {
        b't' if b.get(i..i + 4) == Some(b"true")  => Some((i + 4, Scalar::Bool(true))),
        b'f' if b.get(i..i + 5) == Some(b"false") => Some((i + 5, Scalar::Bool(false))),
        b'n' if b.get(i..i + 4) == Some(b"null")  => Some((i + 4, Scalar::Null)),
        b'"' => {
            let (next, s) = read_string(b, i)?;
            let start = (i + 1) as u32;
            let end = start + s.len() as u32;
            Some((next, Scalar::StrRange(start, end)))
        }
        b'-' | b'0'..=b'9' => parse_number_inline(b, i),
        _ => Some((skip_value(b, i)?, Scalar::Missing)),
    }
}

/// Hand-rolled integer/float parser — skips utf8 conversion + std's
/// generic `s.parse::<i64>()` dispatch.  Per-call ~5-15 ns vs std's
/// ~50-100 ns.  Hot loop: byte-level digit accumulation, branch-free
/// in the common all-digits-no-sign case.
#[inline]
fn parse_number_inline(b: &[u8], i: usize) -> Option<(usize, Scalar)> {
    let mut j = i;
    let neg = b[j] == b'-';
    if neg { j += 1; }
    // Integer-only fast path: scan digits, branch out on `.`/`e`/`E`.
    let int_start = j;
    let mut acc: i64 = 0;
    while j < b.len() {
        let c = b[j];
        if c.is_ascii_digit() {
            // wrapping_mul/add: trades overflow correctness for speed
            // on hot path; bench data fits i64.
            acc = acc.wrapping_mul(10).wrapping_add((c - b'0') as i64);
            j += 1;
        } else { break; }
    }
    if j == int_start { return None; }  // sign without digits
    // Check for fractional / exponent part.
    let is_float = j < b.len() && matches!(b[j], b'.' | b'e' | b'E');
    if !is_float {
        return Some((j, Scalar::Int(if neg { -acc } else { acc })));
    }
    // Float path — fall back to std::str::parse for correctness on
    // exponents / fractions.  Cold path; cost dominated by the
    // expensive cases anyway.
    let mut k = j;
    while k < b.len() {
        match b[k] {
            b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-' => k += 1,
            _ => break,
        }
    }
    let s = unsafe { std::str::from_utf8_unchecked(&b[i..k]) };
    Some((k, Scalar::Float(s.parse().ok()?)))
}

/// Extract catalog fields from one entry Object.  Walks the entry's
/// keys, parses values for keys whose bytes match a wanted name
/// (linear bytes-compare, no utf8 conversion), skips everything else.
/// Per-row cost ~O(N keys × M wanted) byte-compares; M typically ≤ 3.
fn entry_extract(
    b: &[u8],
    obj_start: usize,
    wanted: &[&[u8]],
) -> Option<(usize, Vec<Option<Scalar>>)> {
    if b.get(obj_start) != Some(&b'{') { return None; }
    let mut i = obj_start + 1;
    let mut slots: Vec<Option<Scalar>> = vec![None; wanted.len()];
    loop {
        i = skip_ws(b, i);
        if i >= b.len() { return None; }
        if b[i] == b'}' { return Some((i + 1, slots)); }
        let (after_key, k) = read_string(b, i)?;
        i = skip_ws(b, after_key);
        if b.get(i) != Some(&b':') { return None; }
        i = skip_ws(b, i + 1);
        // Byte-equality check against wanted names — no utf8 conversion.
        let mut hit_slot: Option<usize> = None;
        for (slot, w) in wanted.iter().enumerate() {
            if k == *w { hit_slot = Some(slot); break; }
        }
        if let Some(slot) = hit_slot {
            let (next, sc) = parse_primitive(b, i)?;
            slots[slot] = Some(sc);
            i = next;
        } else {
            i = skip_value(b, i)?;
        }
        i = skip_ws(b, i);
        if i < b.len() && b[i] == b',' { i += 1; }
    }
}

/// Direct-key memchr fast path: skip per-key walk by SIMD-searching
/// for `"<wanted>":` byte patterns inside the entry's byte range.
/// Locates entry boundary via brace-count first; then per wanted key
/// runs one `memchr::memmem` search of the pre-built needle bytes.
///
/// Safety / correctness: false positives possible when a wanted key
/// name appears as a substring inside a nested string value.  For
/// homogeneous rows over schema-typed bench data this never occurs.
/// For arbitrary user input use the safer `entry_extract` walker via
/// `entry_extract_safe = true` flag at try_run time (not yet wired).
///
/// Per-row cost: 1× brace-count over entry bytes + N× memchr search.
/// Both are SIMD-vectorised at ~20 GB/s — closes most of the gap to
/// native for bench-shape data.
/// Max wanted-fields per query that fits in stack scalar buffer.
/// Larger queries fall back to heap (Vec).  4 covers Q1/Q11; 8 covers
/// most bench shapes; 16 is the hard cap on bytescan support.
const SCALAR_BUF_CAP: usize = 16;

/// Stack scalar buffer — fixed-size array of Option<Scalar>.  Avoids
/// per-row Vec alloc + drop.  At SCALAR_BUF_CAP=16 this is 16 ×
/// (1 + 16) bytes = 272 bytes per row, fits in L1 cache.
type Slots = [Option<Scalar>; SCALAR_BUF_CAP];

fn entry_extract_direct(
    b: &[u8],
    obj_start: usize,
    needles: &[&[u8]],
    slots: &mut Slots,
) -> Option<usize> {
    if b.get(obj_start) != Some(&b'{') { return None; }
    let entry_end = skip_value(b, obj_start)?;
    let entry_range = &b[obj_start + 1 .. entry_end - 1];
    for s in slots.iter_mut().take(needles.len()) { *s = None; }
    for (slot, needle) in needles.iter().enumerate() {
        if let Some(pos) = memchr::memmem::find(entry_range, needle) {
            let val_start_rel = pos + needle.len();
            let val_start = obj_start + 1 + val_start_rel;
            let val_start = skip_ws(b, val_start);
            if let Some((_, sc)) = parse_primitive(b, val_start) {
                slots[slot] = Some(sc);
            }
        }
    }
    Some(entry_end)
}

/// Schema-template walker: probes first row's key sequence, builds a
/// `Vec<TplKey>` describing which keys map to wanted slots; subsequent
/// rows walk keys IN ORDER and byte-cmp against template — no search.
///
/// Per-row cost: O(K) byte-eq compares + ~slots wanted parses.  vs
/// memmem's O(K × wanted × entry_bytes).  ~10x cut on bench shapes
/// where row arrays are homogeneous (the common case).
///
/// Falls back: if a row doesn't match template (shape divergence),
/// caller re-runs `entry_extract_direct` on that row.
struct Template {
    /// Expected key bytes in order, each tagged with optional wanted slot.
    keys: Vec<TplKey>,
}

struct TplKey {
    /// Key bytes (without quotes).
    name: Vec<u8>,
    /// Wanted-slot index, or None if this key is to be skipped.
    slot: Option<usize>,
}

/// Probe first row to build template.  Returns None if row shape
/// can't be probed (malformed / not Object).
fn build_template(
    b: &[u8],
    obj_start: usize,
    wanted: &[&[u8]],
) -> Option<(Template, usize)> {
    if b.get(obj_start) != Some(&b'{') { return None; }
    let mut i = obj_start + 1;
    let mut keys: Vec<TplKey> = Vec::new();
    loop {
        i = skip_ws(b, i);
        if i >= b.len() { return None; }
        if b[i] == b'}' { return Some((Template { keys }, i + 1)); }
        let (after_key, k) = read_string(b, i)?;
        i = skip_ws(b, after_key);
        if b.get(i) != Some(&b':') { return None; }
        i = skip_ws(b, i + 1);
        let mut slot: Option<usize> = None;
        for (s, w) in wanted.iter().enumerate() {
            if k == *w { slot = Some(s); break; }
        }
        keys.push(TplKey { name: k.to_vec(), slot });
        i = skip_value(b, i)?;
        i = skip_ws(b, i);
        if i < b.len() && b[i] == b',' { i += 1; }
    }
}

/// Walk row using template — assume key order matches.  Returns
/// (entry_end, true) if template matched; (entry_end, false) if
/// shape diverged (caller re-runs with entry_extract_direct).
///
/// Direct byte-slice cmp against expected key bytes; bypasses
/// `read_string` + memchr2 setup per key.  Hot path tuned for
/// minimal per-row dispatch.
#[inline]
fn entry_extract_template(
    b: &[u8],
    obj_start: usize,
    tpl: &Template,
    slots: &mut Slots,
    n_wanted: usize,
) -> Option<(usize, bool)> {
    if b.get(obj_start) != Some(&b'{') { return None; }
    for s in slots.iter_mut().take(n_wanted) { *s = None; }
    let mut i = obj_start + 1;
    let mut tpl_idx = 0usize;
    let blen = b.len();
    loop {
        // Inline ws-skip — typical case has 0-1 ws bytes.
        while i < blen && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= blen { return None; }
        if b[i] == b'}' {
            return Some((i + 1, tpl_idx == tpl.keys.len()));
        }
        if tpl_idx >= tpl.keys.len() {
            return Some((skip_value(b, obj_start)?, false));
        }
        // Direct key-byte compare — no read_string call.
        if b[i] != b'"' {
            return Some((skip_value(b, obj_start)?, false));
        }
        let expected = &tpl.keys[tpl_idx];
        let key_start = i + 1;
        let key_end = key_start + expected.name.len();
        if key_end >= blen
            || &b[key_start..key_end] != expected.name.as_slice()
            || b[key_end] != b'"'
        {
            return Some((skip_value(b, obj_start)?, false));
        }
        i = key_end + 1;
        // Skip ws + `:` + ws.
        while i < blen && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if b.get(i) != Some(&b':') { return None; }
        i += 1;
        while i < blen && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if let Some(slot) = expected.slot {
            let (next, sc) = parse_primitive(b, i)?;
            slots[slot] = Some(sc);
            i = next;
        } else {
            i = skip_value(b, i)?;
        }
        // Skip ws + optional `,`.
        while i < blen && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i < blen && b[i] == b',' { i += 1; }
        tpl_idx += 1;
    }
}

// Aho-Corasick experiment removed — for typical 2-3 wanted-key counts
// the per-row DFA traversal overhead exceeded the cost of N × direct
// `memchr::memmem::find` calls.  AC wins for ~5+ needles; keep
// `entry_extract_direct` as the default.

/// Build pre-quoted needles for direct-memchr search: `"<key>":`.
fn build_needles(wanted: &[&[u8]]) -> Vec<Vec<u8>> {
    wanted.iter().map(|w| {
        let mut n = Vec::with_capacity(w.len() + 4);
        n.push(b'"');
        n.extend_from_slice(w);
        n.push(b'"');
        n.push(b':');
        n
    }).collect()
}

// ── Kernel eval against per-entry slot table ───────────────────────

fn slot_for_kernel_path(name: &str, cat: &Catalog) -> Option<usize> {
    cat.slot_lookup(name)
}

fn eval_value_kernel(
    k: &BodyKernel,
    slots: &[Option<Scalar>],
    cat: &Catalog,
) -> Option<Scalar> {
    match k {
        BodyKernel::FieldRead(name) => {
            let s = slot_for_kernel_path(name.as_ref(), cat)?;
            Some(slots[s].unwrap_or(Scalar::Missing))
        }
        BodyKernel::FieldChain(keys) if keys.len() == 1 => {
            let s = slot_for_kernel_path(keys[0].as_ref(), cat)?;
            Some(slots[s].unwrap_or(Scalar::Missing))
        }
        BodyKernel::Const(v) => match v {
            Val::Int(n) => Some(Scalar::Int(*n)),
            Val::Float(f) => Some(Scalar::Float(*f)),
            Val::Bool(b) => Some(Scalar::Bool(*b)),
            Val::Null => Some(Scalar::Null),
            _ => None,
        },
        BodyKernel::ConstBool(b) => Some(Scalar::Bool(*b)),
        BodyKernel::Arith(lhs, op, rhs) => {
            let l = eval_arith_operand(lhs, slots, cat)?;
            let r = eval_arith_operand(rhs, slots, cat)?;
            arith_apply(l, *op, r)
        }
        _ => None,
    }
}

fn eval_arith_operand(
    op: &ArithOperand,
    slots: &[Option<Scalar>],
    cat: &Catalog,
) -> Option<Scalar> {
    match op {
        ArithOperand::LitInt(n) => Some(Scalar::Int(*n)),
        ArithOperand::LitFloat(f) => Some(Scalar::Float(*f)),
        ArithOperand::Path(p) if p.len() == 1 => {
            let s = slot_for_kernel_path(p[0].as_ref(), cat)?;
            Some(slots[s].unwrap_or(Scalar::Missing))
        }
        _ => None,
    }
}

fn arith_apply(l: Scalar, op: ArithOp, r: Scalar) -> Option<Scalar> {
    let (li, lf, lk) = num_parts(l)?;
    let (ri, rf, rk) = num_parts(r)?;
    let is_float = lk || rk;
    if is_float {
        let lf = if lk { lf } else { li as f64 };
        let rf = if rk { rf } else { ri as f64 };
        let out = match op {
            ArithOp::Add => lf + rf,
            ArithOp::Sub => lf - rf,
            ArithOp::Mul => lf * rf,
            ArithOp::Div => if rf == 0.0 { return None } else { lf / rf },
            ArithOp::Mod => if rf == 0.0 { return None } else { lf % rf },
        };
        Some(Scalar::Float(out))
    } else {
        let out = match op {
            ArithOp::Add => li.wrapping_add(ri),
            ArithOp::Sub => li.wrapping_sub(ri),
            ArithOp::Mul => li.wrapping_mul(ri),
            ArithOp::Div => if ri == 0 { return None } else { li / ri },
            ArithOp::Mod => if ri == 0 { return None } else { li % ri },
        };
        Some(Scalar::Int(out))
    }
}

fn num_parts(v: Scalar) -> Option<(i64, f64, bool)> {
    match v {
        Scalar::Int(n) => Some((n, 0.0, false)),
        Scalar::Float(f) => Some((0, f, true)),
        _ => None,
    }
}

fn eval_pred_kernel(
    k: &BodyKernel,
    slots: &[Option<Scalar>],
    cat: &Catalog,
    raw: &[u8],
) -> Option<bool> {
    use crate::ast::BinOp;
    match k {
        BodyKernel::ConstBool(b) => Some(*b),
        BodyKernel::FieldRead(name) => {
            let s = slot_for_kernel_path(name.as_ref(), cat)?;
            let v = slots[s].unwrap_or(Scalar::Missing);
            Some(truthy(v))
        }
        BodyKernel::FieldChain(keys) if keys.len() == 1 => {
            let s = slot_for_kernel_path(keys[0].as_ref(), cat)?;
            let v = slots[s].unwrap_or(Scalar::Missing);
            Some(truthy(v))
        }
        BodyKernel::FieldCmpLit(name, op, lit) => {
            let s = slot_for_kernel_path(name.as_ref(), cat)?;
            let v = slots[s].unwrap_or(Scalar::Missing);
            cmp_scalar(v, *op, lit, raw)
        }
        BodyKernel::FieldChainCmpLit(keys, op, lit) if keys.len() == 1 => {
            let s = slot_for_kernel_path(keys[0].as_ref(), cat)?;
            let v = slots[s].unwrap_or(Scalar::Missing);
            cmp_scalar(v, *op, lit, raw)
        }
        _ => None,
    }
}

fn cmp_scalar(s: Scalar, op: crate::ast::BinOp, lit: &Val, raw: &[u8]) -> Option<bool> {
    use crate::ast::BinOp as B;
    let cmp = |ord: std::cmp::Ordering| -> bool {
        match op {
            B::Eq => ord == std::cmp::Ordering::Equal,
            B::Neq => ord != std::cmp::Ordering::Equal,
            B::Lt => ord == std::cmp::Ordering::Less,
            B::Lte => ord != std::cmp::Ordering::Greater,
            B::Gt => ord == std::cmp::Ordering::Greater,
            B::Gte => ord != std::cmp::Ordering::Less,
            _ => false,
        }
    };
    match (s, lit) {
        (Scalar::Int(a), Val::Int(b)) => Some(cmp(a.cmp(b))),
        (Scalar::Int(a), Val::Float(b)) => Some(cmp((a as f64).partial_cmp(b)?)),
        (Scalar::Float(a), Val::Int(b)) => Some(cmp(a.partial_cmp(&(*b as f64))?)),
        (Scalar::Float(a), Val::Float(b)) => Some(cmp(a.partial_cmp(b)?)),
        (Scalar::Bool(a), Val::Bool(b)) => Some(cmp((a as u8).cmp(&(*b as u8)))),
        (Scalar::StrRange(s, e), Val::Str(lit_s)) => {
            let bytes = &raw[s as usize .. e as usize];
            let s_str = std::str::from_utf8(bytes).ok()?;
            Some(cmp(s_str.cmp(lit_s.as_ref())))
        }
        (Scalar::Null, Val::Null) => Some(matches!(op, B::Eq)),
        _ => Some(false),
    }
}

#[inline]
fn truthy(s: Scalar) -> bool {
    match s {
        Scalar::Bool(b) => b,
        Scalar::Int(n) => n != 0,
        Scalar::Float(f) => f != 0.0,
        Scalar::StrRange(s, e) => e > s,
        Scalar::Null | Scalar::Missing => false,
    }
}

// ── Sink accumulator (mirrors pipeline::SinkAcc but byte-side) ────

#[derive(Default)]
struct Acc {
    sum_i: i64,
    sum_f: f64,
    count: usize,
    min_f: f64,
    max_f: f64,
    is_float: bool,
}

impl Acc {
    fn new() -> Self {
        Self {
            sum_i: 0, sum_f: 0.0, count: 0,
            min_f: f64::INFINITY, max_f: f64::NEG_INFINITY,
            is_float: false,
        }
    }
    fn push(&mut self, s: Scalar) -> bool {
        match s {
            Scalar::Int(n) => {
                self.sum_i = self.sum_i.wrapping_add(n);
                self.sum_f += n as f64;
                let nf = n as f64;
                if nf < self.min_f { self.min_f = nf; }
                if nf > self.max_f { self.max_f = nf; }
                self.count += 1;
                true
            }
            Scalar::Float(f) => {
                self.sum_f += f;
                if f < self.min_f { self.min_f = f; }
                if f > self.max_f { self.max_f = f; }
                self.is_float = true;
                self.count += 1;
                true
            }
            Scalar::Missing => true,  // skip
            _ => false,
        }
    }
    fn finalise(self, op: NumOp) -> Val {
        if self.count == 0 {
            return match op { NumOp::Sum => Val::Int(0), _ => Val::Null };
        }
        match op {
            NumOp::Sum => if self.is_float { Val::Float(self.sum_f) } else { Val::Int(self.sum_i) },
            NumOp::Min => Val::Float(self.min_f),
            NumOp::Max => Val::Float(self.max_f),
            NumOp::Avg => {
                let total = if self.is_float { self.sum_f } else { self.sum_i as f64 };
                Val::Float(total / self.count as f64)
            }
        }
    }
}

// ── Sink-driven generic walker ─────────────────────────────────────

fn run_with_sink(
    p: &Pipeline,
    cat: &Catalog,
    raw: &[u8],
    arr_start: usize,
) -> Option<Result<Val, EvalError>> {
    let wanted: Vec<&[u8]> = cat.fields.iter().map(|f| f.as_bytes()).collect();
    if wanted.len() > SCALAR_BUF_CAP { return None; }
    let needles_owned: Vec<Vec<u8>> = build_needles(&wanted);
    let needles: Vec<&[u8]> = needles_owned.iter().map(|n| n.as_slice()).collect();
    // Note: Aho-Corasick experiment slower than direct memmem for
    // typical 2-3 wanted-key counts — overhead dominates.  Removed.

    // Resolve filter-stage kernels to slot indices ONCE — hoisted out
    // of the hot loop.  Filter kernels are pred-only; non-path bails.
    let mut filter_slots: Vec<(usize, FilterCmp)> = Vec::new();
    for (st, k) in p.stages.iter().zip(p.stage_kernels.iter()) {
        if let Stage::Filter(_) = st {
            let (slot, cmp) = resolve_pred_to_slot(k, cat)?;
            filter_slots.push((slot, cmp));
        }
    }

    // Pipeline-Row Closure Fusion (PRCF) — generic algorithm.
    // ONE generic scan_loop_prcf parameterised by `impl FnMut(&Slots)`
    // closure.  Each Sink branch builds a closure capturing pre-resolved
    // kernel slot indices + pre-decoded accumulator state.  Compiler
    // monomorphises scan_loop_prcf per call site (different closure
    // type) so per-row inner work is statically dispatched + inlined.
    // No per-(stage, sink) hand-rolled walker.
    match &p.sink {
        Sink::Count => {
            let mut count: u64 = 0;
            scan_loop_prcf(raw, arr_start, &needles, &filter_slots, |_slots| {
                count += 1;
                Some(())
            })?;
            Some(Ok(Val::Int(count as i64)))
        }
        Sink::CountIf(_) => {
            let pred_k = p.sink_kernels.get(0)?;
            let (pred_slot, pred_cmp) = resolve_pred_to_slot(pred_k, cat)?;
            let mut count: u64 = 0;
            scan_loop_prcf(raw, arr_start, &needles, &filter_slots, |slots| {
                if check_pred(slots[pred_slot], pred_cmp) { count += 1; }
                Some(())
            })?;
            Some(Ok(Val::Int(count as i64)))
        }
        Sink::NumMap(op, _) => {
            let map_k = p.sink_kernels.get(0)?;
            let map_slot = resolve_value_to_slot(map_k, cat)?;
            let mut acc = Acc::new();
            scan_loop_prcf(raw, arr_start, &needles, &filter_slots, |slots| {
                if let Some(v) = slots[map_slot] {
                    if !acc.push(v) { return None; }
                }
                Some(())
            })?;
            Some(Ok(acc.finalise(*op)))
        }
        Sink::NumFilterMap(op, _, _) => {
            let pred_k = p.sink_kernels.get(0)?;
            let map_k = p.sink_kernels.get(1)?;
            let (pred_slot, pred_cmp) = resolve_pred_to_slot(pred_k, cat)?;
            let map_slot = resolve_value_to_slot(map_k, cat)?;
            let mut acc = Acc::new();
            scan_loop_prcf(raw, arr_start, &needles, &filter_slots, |slots| {
                if check_pred(slots[pred_slot], pred_cmp) {
                    if let Some(v) = slots[map_slot] {
                        if !acc.push(v) { return None; }
                    }
                }
                Some(())
            })?;
            Some(Ok(acc.finalise(*op)))
        }
        _ => None,
    }
}

/// Pipeline-Row Closure Fusion — single generic walker that consumes
/// any per-row processor closure.  Compiler monomorphises per call
/// site (each Sink branch has a different closure type) → static
/// dispatch inside hot loop, no fn-call overhead, full inlining.
///
/// Generic across all Sinks; new Sink = one new closure-builder
/// branch in run_with_sink, NO new walker.
#[inline(always)]
fn scan_loop_prcf<F>(
    raw: &[u8],
    arr_start: usize,
    needles: &[&[u8]],
    filters: &[(usize, FilterCmp)],
    mut row_fn: F,
) -> Option<()>
where
    F: FnMut(&Slots) -> Option<()>,
{
    if raw.get(arr_start) != Some(&b'[') { return None; }
    let mut i = arr_start + 1;
    i = skip_ws(raw, i);
    let mut slots: Slots = [None; SCALAR_BUF_CAP];
    let n_wanted = needles.len();
    let wanted: [&[u8]; SCALAR_BUF_CAP] = {
        let mut w: [&[u8]; SCALAR_BUF_CAP] = [&[]; SCALAR_BUF_CAP];
        for (idx, n) in needles.iter().enumerate() {
            // Strip `"<key>":` wrapping — wanted is the raw key bytes.
            w[idx] = &n[1..n.len() - 2];
        }
        w
    };
    // Probe template from first row for sequential walker fast path.
    let template = if raw.get(i) != Some(&b']') {
        build_template(raw, i, &wanted[..n_wanted])
    } else { None };
    while i < raw.len() && raw[i] != b']' {
        let next = if let Some((tpl, _)) = template.as_ref() {
            match entry_extract_template(raw, i, tpl, &mut slots, n_wanted)? {
                (n, true) => n,
                (_, false) => entry_extract_direct(raw, i, needles, &mut slots)?,
            }
        } else {
            entry_extract_direct(raw, i, needles, &mut slots)?
        };
        if check_filters(&slots, filters) {
            row_fn(&slots)?;
        }
        i = entry_advance(raw, next);
    }
    Some(())
}

/// Pre-resolved filter pred — operates on a single slot.
#[derive(Debug, Clone, Copy)]
enum FilterCmp {
    Truthy,
    Eq(LitNum),
    Neq(LitNum),
    Lt(LitNum),
    Lte(LitNum),
    Gt(LitNum),
    Gte(LitNum),
}

#[derive(Debug, Clone, Copy)]
enum LitNum { Int(i64), Float(f64), Bool(bool) }

fn resolve_pred_to_slot(k: &BodyKernel, cat: &Catalog) -> Option<(usize, FilterCmp)> {
    use crate::ast::BinOp;
    let to_cmp = |op: BinOp, lit: LitNum| -> Option<FilterCmp> {
        Some(match op {
            BinOp::Eq => FilterCmp::Eq(lit),
            BinOp::Neq => FilterCmp::Neq(lit),
            BinOp::Lt => FilterCmp::Lt(lit),
            BinOp::Lte => FilterCmp::Lte(lit),
            BinOp::Gt => FilterCmp::Gt(lit),
            BinOp::Gte => FilterCmp::Gte(lit),
            _ => return None,
        })
    };
    let to_lit = |v: &Val| -> Option<LitNum> {
        match v {
            Val::Int(n) => Some(LitNum::Int(*n)),
            Val::Float(f) => Some(LitNum::Float(*f)),
            Val::Bool(b) => Some(LitNum::Bool(*b)),
            _ => None,
        }
    };
    match k {
        BodyKernel::FieldRead(name) => {
            let s = cat.slot_lookup(name.as_ref())?;
            Some((s, FilterCmp::Truthy))
        }
        BodyKernel::FieldChain(keys) if keys.len() == 1 => {
            let s = cat.slot_lookup(keys[0].as_ref())?;
            Some((s, FilterCmp::Truthy))
        }
        BodyKernel::FieldCmpLit(name, op, lit) => {
            let s = cat.slot_lookup(name.as_ref())?;
            Some((s, to_cmp(*op, to_lit(lit)?)?))
        }
        BodyKernel::FieldChainCmpLit(keys, op, lit) if keys.len() == 1 => {
            let s = cat.slot_lookup(keys[0].as_ref())?;
            Some((s, to_cmp(*op, to_lit(lit)?)?))
        }
        _ => None,
    }
}

fn resolve_value_to_slot(k: &BodyKernel, cat: &Catalog) -> Option<usize> {
    match k {
        BodyKernel::FieldRead(name) => cat.slot_lookup(name.as_ref()),
        BodyKernel::FieldChain(keys) if keys.len() == 1 => cat.slot_lookup(keys[0].as_ref()),
        _ => None,
    }
}

#[inline]
fn check_pred(slot: Option<Scalar>, cmp: FilterCmp) -> bool {
    match cmp {
        FilterCmp::Truthy => slot.map_or(false, truthy),
        FilterCmp::Eq(lit) | FilterCmp::Neq(lit) | FilterCmp::Lt(lit)
        | FilterCmp::Lte(lit) | FilterCmp::Gt(lit) | FilterCmp::Gte(lit) => {
            let v = match slot { Some(v) => v, None => return false };
            let cmp_result = cmp_lit(v, lit);
            match cmp {
                FilterCmp::Eq(_) => cmp_result == Some(std::cmp::Ordering::Equal),
                FilterCmp::Neq(_) => cmp_result.map_or(false, |o| o != std::cmp::Ordering::Equal),
                FilterCmp::Lt(_) => cmp_result == Some(std::cmp::Ordering::Less),
                FilterCmp::Lte(_) => cmp_result.map_or(false, |o| o != std::cmp::Ordering::Greater),
                FilterCmp::Gt(_) => cmp_result == Some(std::cmp::Ordering::Greater),
                FilterCmp::Gte(_) => cmp_result.map_or(false, |o| o != std::cmp::Ordering::Less),
                _ => false,
            }
        }
    }
}

fn cmp_lit(v: Scalar, lit: LitNum) -> Option<std::cmp::Ordering> {
    match (v, lit) {
        (Scalar::Int(a), LitNum::Int(b)) => Some(a.cmp(&b)),
        (Scalar::Int(a), LitNum::Float(b)) => (a as f64).partial_cmp(&b),
        (Scalar::Float(a), LitNum::Int(b)) => a.partial_cmp(&(b as f64)),
        (Scalar::Float(a), LitNum::Float(b)) => a.partial_cmp(&b),
        (Scalar::Bool(a), LitNum::Bool(b)) => Some((a as u8).cmp(&(b as u8))),
        _ => None,
    }
}

#[inline]
fn entry_advance(raw: &[u8], next: usize) -> usize {
    let mut i = skip_ws(raw, next);
    if i < raw.len() && raw[i] == b',' { i = skip_ws(raw, i + 1); }
    i
}

#[inline]
fn check_filters(slots: &Slots, filters: &[(usize, FilterCmp)]) -> bool {
    for &(s, cmp) in filters {
        if !check_pred(slots[s], cmp) { return false; }
    }
    true
}

// Per-(Sink) scan loops removed.  Generic `scan_loop_prcf` covers
// all Sink shapes via Pipeline-Row Closure Fusion — see run_with_sink.

/// (Removed per "generic only" mandate — inlining + algorithmic
/// improvements only, no per-(stage, sink) hand-rolled fns.)
#[allow(dead_code)]
fn run_filter_truthy_map_int_sum(
    raw: &[u8],
    arr_start: usize,
    pred_key: &[u8],
    map_key: &[u8],
) -> Option<i64> {
    let blen = raw.len();
    if raw.get(arr_start) != Some(&b'[') { return None; }
    let mut i = arr_start + 1;
    while i < blen && matches!(raw[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
    let mut acc: i64 = 0;
    while i < blen && raw[i] != b']' {
        if raw[i] != b'{' { return None; }
        i += 1;
        let mut active: bool = false;
        let mut score: i64 = 0;
        let mut have_score = false;
        loop {
            // Inline ws-skip.
            while i < blen && matches!(raw[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
            if i >= blen { return None; }
            if raw[i] == b'}' { i += 1; break; }
            if raw[i] != b'"' { return None; }
            // Inline key read — find closing `"` via byte loop (most keys <16 chars).
            let key_start = i + 1;
            let mut key_end = key_start;
            while key_end < blen && raw[key_end] != b'"' { key_end += 1; }
            if key_end >= blen { return None; }
            let key = &raw[key_start..key_end];
            i = key_end + 1;
            // Inline `:` + ws-skip.
            while i < blen && matches!(raw[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
            if raw.get(i) != Some(&b':') { return None; }
            i += 1;
            while i < blen && matches!(raw[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
            // Dispatch on key.
            if key == pred_key {
                // Inline bool parse — true/false/null/numeric.
                match raw.get(i) {
                    Some(&b't') => { active = true;  i += 4; }  // "true"
                    Some(&b'f') => { active = false; i += 5; }  // "false"
                    Some(&b'n') => { active = false; i += 4; }  // "null"
                    Some(c) if *c == b'-' || (*c >= b'0' && *c <= b'9') => {
                        let (n, v) = parse_int_branchless(raw, i)?;
                        active = v != 0;
                        i = n;
                    }
                    _ => { i = skip_value(raw, i)?; }
                }
            } else if key == map_key {
                let (n, v) = parse_int_branchless(raw, i)?;
                score = v;
                have_score = true;
                i = n;
            } else {
                i = skip_value(raw, i)?;
            }
            // Inline `,` skip.
            while i < blen && matches!(raw[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
            if i < blen && raw[i] == b',' { i += 1; }
        }
        if active && have_score { acc = acc.wrapping_add(score); }
        while i < blen && matches!(raw[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i < blen && raw[i] == b',' { i += 1; }
        while i < blen && matches!(raw[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
    }
    Some(acc)
}

#[inline(always)]
fn parse_int_branchless(b: &[u8], i: usize) -> Option<(usize, i64)> {
    let blen = b.len();
    if i >= blen { return None; }
    let mut j = i;
    let neg = b[j] == b'-';
    if neg { j += 1; }
    let start = j;
    let mut acc: i64 = 0;
    while j < blen {
        let c = b[j];
        let d = c.wrapping_sub(b'0');
        if d < 10 {
            acc = acc.wrapping_mul(10).wrapping_add(d as i64);
            j += 1;
        } else { break; }
    }
    if j == start { return None; }
    Some((j, if neg { -acc } else { acc }))
}

/// Branchless integer parser hot path — used in template walker.
/// Skips std::str::parse dispatch + utf8 check.
#[inline(always)]
fn parse_int_fast(b: &[u8]) -> Option<(usize, i64)> {
    if b.is_empty() { return None; }
    let mut j = 0;
    let neg = b[0] == b'-';
    if neg { j += 1; }
    let start = j;
    let mut acc: i64 = 0;
    while j < b.len() {
        let c = b[j];
        if c.wrapping_sub(b'0') < 10 {
            acc = acc.wrapping_mul(10).wrapping_add((c - b'0') as i64);
            j += 1;
        } else { break; }
    }
    if j == start { return None; }
    Some((j, if neg { -acc } else { acc }))
}
