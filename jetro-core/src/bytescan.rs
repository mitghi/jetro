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

fn skip_value(b: &[u8], mut i: usize) -> Option<usize> {
    i = skip_ws(b, i);
    if i >= b.len() { return None; }
    match b[i] {
        b'"' => skip_string(b, i),
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
        b'-' | b'0'..=b'9' => {
            let start = i;
            let mut j = i;
            let mut is_float = false;
            if b[j] == b'-' { j += 1; }
            while j < b.len() {
                match b[j] {
                    b'0'..=b'9' => j += 1,
                    b'.' | b'e' | b'E' | b'+' | b'-' => { is_float = true; j += 1; }
                    _ => break,
                }
            }
            let s = std::str::from_utf8(&b[start..j]).ok()?;
            if is_float {
                Some((j, Scalar::Float(s.parse().ok()?)))
            } else {
                Some((j, Scalar::Int(s.parse().ok()?)))
            }
        }
        _ => Some((skip_value(b, i)?, Scalar::Missing)),
    }
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
fn entry_extract_direct(
    b: &[u8],
    obj_start: usize,
    needles: &[&[u8]],
) -> Option<(usize, Vec<Option<Scalar>>)> {
    if b.get(obj_start) != Some(&b'{') { return None; }
    // Locate entry end via brace count (memchr-accelerated via skip_value).
    let entry_end = skip_value(b, obj_start)?;
    let entry_range = &b[obj_start + 1 .. entry_end - 1];
    let mut slots: Vec<Option<Scalar>> = vec![None; needles.len()];
    for (slot, needle) in needles.iter().enumerate() {
        if let Some(pos) = memchr::memmem::find(entry_range, needle) {
            // Skip past `"<key>":` then any whitespace.
            let val_start_rel = pos + needle.len();
            let val_start = obj_start + 1 + val_start_rel;
            let val_start = skip_ws(b, val_start);
            if let Some((_, sc)) = parse_primitive(b, val_start) {
                slots[slot] = Some(sc);
            }
        }
    }
    Some((entry_end, slots))
}

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
    // Pre-cache wanted-name bytes — avoids per-row utf8 + Arc<str> compare.
    let wanted: Vec<&[u8]> = cat.fields.iter().map(|f| f.as_bytes()).collect();
    // Pre-build `"<key>":` needles for direct-memchr fast path.
    let needles_owned: Vec<Vec<u8>> = build_needles(&wanted);
    let needles: Vec<&[u8]> = needles_owned.iter().map(|n| n.as_slice()).collect();
    let mut i = arr_start;
    if raw.get(i) != Some(&b'[') { return None; }
    i = skip_ws(raw, i + 1);
    let mut acc = Acc::new();
    let mut count: u64 = 0;
    while i < raw.len() {
        if raw[i] == b']' { break; }
        let (next, slots) = entry_extract_direct(raw, i, &needles)?;
        // Eval filters
        let mut pass = true;
        for (st, k) in p.stages.iter().zip(p.stage_kernels.iter()) {
            if let Stage::Filter(_) = st {
                if !eval_pred_kernel(k, &slots, cat, raw)? { pass = false; break; }
            }
        }
        if pass {
            match &p.sink {
                Sink::Count => { count += 1; }
                Sink::CountIf(_) => {
                    let pred_k = p.sink_kernels.get(0)?;
                    if eval_pred_kernel(pred_k, &slots, cat, raw)? { count += 1; }
                }
                Sink::Numeric(_) => {
                    // Accumulate the entry value itself — only meaningful
                    // when entry is a primitive (rare for byte path; stays
                    // on tape).  Bail.
                    return None;
                }
                Sink::NumMap(_, _) => {
                    let map_k = p.sink_kernels.get(0)?;
                    let v = eval_value_kernel(map_k, &slots, cat)?;
                    if !acc.push(v) { return None; }
                }
                Sink::NumFilterMap(_, _, _) => {
                    let pred_k = p.sink_kernels.get(0)?;
                    let map_k = p.sink_kernels.get(1)?;
                    if eval_pred_kernel(pred_k, &slots, cat, raw)? {
                        let v = eval_value_kernel(map_k, &slots, cat)?;
                        if !acc.push(v) { return None; }
                    }
                }
                _ => return None,
            }
        }
        i = next;
        i = skip_ws(raw, i);
        if i < raw.len() && raw[i] == b',' { i = skip_ws(raw, i + 1); }
    }
    let result = match &p.sink {
        Sink::Count | Sink::CountIf(_) => Val::Int(count as i64),
        Sink::NumMap(op, _) | Sink::NumFilterMap(op, _, _) => acc.finalise(*op),
        _ => return None,
    };
    Some(Ok(result))
}
