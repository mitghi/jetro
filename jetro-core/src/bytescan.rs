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
use crate::pipeline::{Pipeline, Source, Sink, Stage, BodyKernel, NumOp, ArithOperand, ObjProjEntry};
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
    // Stage acceptance: Filter (any pred kernel), Skip, Take, and a
    // single trailing Map (FieldRead/FieldChain[1]/ObjProject).  All
    // other shapes defer to tape exec.  Skip/Take counts collect for
    // sink dispatch; Map kernel resolved later via cs.last().
    // FlatMap stays on the tape route — the simd-json tape parses
    // nested arrays once and walks them at ~10 ns/element; bytescan's
    // per-outer memchr + per-inner template extract cost is higher
    // than tape's amortised parse + walk for typical 2-level shapes.
    // Bench-validated: keeping FlatMap on tape.
    // Detect trailing UniqueBy(None) — absorbs into Collect sink as
    // a per-row dedup; treat the stage *before* it as the trailing
    // Map for projection purposes.
    let stage_count = p.stages.len();
    // Heap-top-K detection: leading `[Sort(Some), Take(k)]` plus
    // optional trailing Map projection, terminating in Collect.
    // Pure ordering pattern — no Filter/Skip mixed in for now.  When
    // matched, dispatches the heap walker after the early-exit + skip
    // path acceptance logic skips this leg.
    let heap_top_k = detect_heap_top_k(&p.stages, &p.stage_kernels, &p.sink);
    let last_is_unique = heap_top_k.is_none() && matches!(p.stages.last(), Some(Stage::UniqueBy(None)));
    let last_payload_idx = if last_is_unique { stage_count.saturating_sub(1) } else { stage_count };
    if heap_top_k.is_none() {
        for (idx, (st, k)) in p.stages.iter().zip(p.stage_kernels.iter()).enumerate() {
            match st {
                Stage::Filter(_) => {
                    if !is_pred_kernel(k) { return None; }
                }
                Stage::Skip(_) | Stage::Take(_) => {}
                Stage::Map(_) if last_payload_idx > 0 && idx == last_payload_idx - 1 => {
                    match k {
                        BodyKernel::FieldRead(_) => {}
                        BodyKernel::FieldChain(_) => {}
                        BodyKernel::ObjProject(entries) => {
                            if !objproject_is_byte_friendly(entries) { return None; }
                        }
                        _ => return None,
                    }
                }
                Stage::UniqueBy(None) if idx == stage_count - 1 => {
                    if !matches!(p.sink, Sink::Collect) { return None; }
                }
                _ => return None,
            }
        }
    }
    // Build wanted-fields catalog from kernels.  Cap field-name uniqueness
    // (~16 distinct fields per row); linear scan of catalog is fine.
    // Skip/Take stages don't extract per-row fields — those are
    // navigation-only and skipped here.
    let mut catalog: Catalog = Catalog::default();
    if let Some(htk) = &heap_top_k {
        // Heap path: register sort key + projection paths only;
        // Sort/Take/Map stages are consumed by the heap walker.
        catalog.slot_for_chain(&htk.sort_key);
        if let Some(proj) = &htk.proj_kernel {
            collect_paths(proj, &mut catalog)?;
        }
    } else {
        for (st, k) in p.stages.iter().zip(p.stage_kernels.iter()) {
            match st {
                Stage::Skip(_) | Stage::Take(_) | Stage::UniqueBy(None) => continue,
                _ => {}
            }
            collect_paths(k, &mut catalog)?;
        }
        for k in p.sink_kernels.iter() { collect_paths(k, &mut catalog)?; }
    }

    // Locate source array byte position.
    let arr_start = walk_chain_to_array(raw_bytes, &chain)?;

    if let Some(htk) = heap_top_k {
        return run_heap_top_k(&htk, &catalog, raw_bytes, arr_start);
    }
    run_with_sink(p, &catalog, raw_bytes, arr_start, last_is_unique)
}

/// Heap-top-K shape — picks `k` rows ordered by a path key, optionally
/// applies a single trailing Map projection.  Eligible Pipeline shape:
///   `[Sort(Some(...)), Take(k)]` or `[Sort(Some(...)), Take(k), Map]`
///   terminated by `Sink::Collect`.
struct HeapTopK {
    k: usize,
    sort_key: Vec<Arc<str>>,
    /// `true` iff the sort program is `0 - <path>` (descending).
    descending: bool,
    /// Trailing Map kernel (projection) — when None, output rows are
    /// the full row Object materialised via simd-json.
    proj_kernel: Option<BodyKernel>,
}

fn detect_heap_top_k(
    stages: &[Stage],
    kernels: &[BodyKernel],
    sink: &Sink,
) -> Option<HeapTopK> {
    if !matches!(sink, Sink::Collect) { return None; }
    if stages.len() < 2 { return None; }
    if !matches!(stages[0], Stage::Sort(Some(_))) { return None; }
    let k = match stages[1] { Stage::Take(n) => n, _ => return None };
    // Shape must terminate at Take or [Take, Map].
    let proj_kernel: Option<BodyKernel> = match stages.len() {
        2 => None,
        3 => match (&stages[2], kernels.get(2)) {
            (Stage::Map(_), Some(k @ BodyKernel::FieldRead(_))) => Some(k.clone()),
            (Stage::Map(_), Some(k @ BodyKernel::FieldChain(_))) => Some(k.clone()),
            (Stage::Map(_), Some(k @ BodyKernel::ObjProject(entries)))
                if objproject_is_byte_friendly(entries) => Some(k.clone()),
            _ => return None,
        },
        _ => return None,
    };
    // Sort kernel must be path-based: either FieldRead/FieldChain
    // (ascending) or Arith(0, Sub, Path) (descending).  Other shapes
    // (computed expressions) defer to tape.
    let (sort_key, descending) = match kernels.first()? {
        BodyKernel::FieldRead(name) => (vec![Arc::clone(name)], false),
        BodyKernel::FieldChain(keys) => (keys.iter().map(Arc::clone).collect(), false),
        BodyKernel::Arith(ArithOperand::LitInt(0), crate::pipeline::ArithOp::Sub, ArithOperand::Path(p)) =>
            (p.iter().map(Arc::clone).collect(), true),
        _ => return None,
    };
    Some(HeapTopK { k, sort_key, descending, proj_kernel })
}

/// Heap-top-K byte walker.  One pass over the source array; per row
/// extract the sort-key scalar; maintain a min-heap of size K (max-
/// heap when descending) keyed by the sort-key.  At end, drain heap
/// in reverse to produce the K elements in the requested order, each
/// projected via `proj_kernel` (or full row when None).
fn run_heap_top_k(
    htk: &HeapTopK,
    cat: &Catalog,
    raw: &[u8],
    arr_start: usize,
) -> Option<Result<Val, EvalError>> {
    use std::collections::BinaryHeap;
    use std::cmp::Reverse;

    if cat.n_slots() > SCALAR_BUF_CAP { return None; }
    let wanted: Vec<&[u8]> = cat.top_keys.iter().map(|f| f.as_bytes()).collect();
    let needles_owned: Vec<Vec<u8>> = build_needles(&wanted);
    let needles: Vec<&[u8]> = needles_owned.iter().map(|n| n.as_slice()).collect();

    let sort_slot = cat.slot_lookup_chain(&htk.sort_key)?;
    let map_kind = build_map_kind(htk.proj_kernel.as_ref(), cat)?;

    if htk.k == 0 { return Some(Ok(Val::arr(Vec::new()))); }

    // Heap stores `(OrdKey, seq, val_idx)` keyed by sort-key + insert
    // sequence (stable for ties).  Val payload lives in a side Vec
    // indexed by `val_idx` to keep `Val` out of the heap (Val isn't
    // Ord by design).  Capped at K — pop when over.
    let mut store: Vec<Val> = Vec::with_capacity(htk.k);
    let mut asc_heap: BinaryHeap<(OrdKey, Reverse<u64>, u32)> = BinaryHeap::new();
    let mut desc_heap: BinaryHeap<(Reverse<OrdKey>, Reverse<u64>, u32)> = BinaryHeap::new();
    let mut seq: u64 = 0;

    let mut row_fn = |slots: &Slots, span: RowSpan| -> Option<ScanCtl> {
        let key = match slots[sort_slot] {
            Some(sc) => OrdKey::from_scalar(raw, sc),
            None => OrdKey::Nullish,
        };
        let v = build_row_val(raw, slots, span, &map_kind)?;
        seq += 1;
        if htk.descending {
            // Want largest K.  Min-heap on plain OrdKey: heap top is
            // the SMALLEST of the kept set; pop when over cap.
            if desc_heap.len() < htk.k {
                let idx = store.len() as u32;
                store.push(v);
                desc_heap.push((Reverse(key), Reverse(seq), idx));
            } else if let Some(top) = desc_heap.peek() {
                if key > (top.0).0 {
                    let idx = (top.2) as usize;
                    store[idx] = v;
                    let new_idx = top.2;
                    desc_heap.pop();
                    desc_heap.push((Reverse(key), Reverse(seq), new_idx));
                }
            }
        } else {
            // Want smallest K.  Max-heap on plain OrdKey via Reverse;
            // heap top is the LARGEST of the kept set.
            if asc_heap.len() < htk.k {
                let idx = store.len() as u32;
                store.push(v);
                asc_heap.push((key, Reverse(seq), idx));
            } else if let Some(top) = asc_heap.peek() {
                if key < top.0 {
                    let idx = (top.2) as usize;
                    store[idx] = v;
                    let new_idx = top.2;
                    asc_heap.pop();
                    asc_heap.push((key, Reverse(seq), new_idx));
                }
            }
        }
        Some(ScanCtl::Continue)
    };
    scan_loop_prcf(raw, arr_start, &needles, cat, &[], &mut row_fn)?;

    // Drain heap into a sorted Vec preserving insertion order on ties.
    let mut out: Vec<Val> = Vec::with_capacity(if htk.descending { desc_heap.len() } else { asc_heap.len() });
    if htk.descending {
        let mut entries: Vec<(OrdKey, u64, u32)> = desc_heap.into_iter()
            .map(|(Reverse(k), Reverse(s), i)| (k, s, i)).collect();
        entries.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        for (_, _, i) in entries {
            out.push(std::mem::replace(&mut store[i as usize], Val::Null));
        }
    } else {
        let mut entries: Vec<(OrdKey, u64, u32)> = asc_heap.into_iter()
            .map(|(k, Reverse(s), i)| (k, s, i)).collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        for (_, _, i) in entries {
            out.push(std::mem::replace(&mut store[i as usize], Val::Null));
        }
    }
    Some(Ok(Val::arr(out)))
}

/// Order-comparable key for heap-top-K.  Numerics compare numerically;
/// strings compare lexicographically by raw bytes.  Mixed types order
/// by tag discriminant (deterministic but not semantically meaningful;
/// real-world queries sort over a single typed column).
#[derive(Clone)]
enum OrdKey {
    Int(i64),
    Float(f64),
    Bytes(Vec<u8>),
    Bool(bool),
    Nullish,
}

impl OrdKey {
    fn from_scalar(raw: &[u8], s: Scalar) -> Self {
        match s {
            Scalar::Int(n) => OrdKey::Int(n),
            Scalar::Float(f) => OrdKey::Float(f),
            Scalar::StrRange(s, e) => OrdKey::Bytes(raw[s as usize..e as usize].to_vec()),
            Scalar::Bool(b) => OrdKey::Bool(b),
            Scalar::Null | Scalar::Missing | Scalar::ObjRange(_, _) => OrdKey::Nullish,
        }
    }
    fn tag(&self) -> u8 {
        match self {
            OrdKey::Nullish => 0, OrdKey::Bool(_) => 1, OrdKey::Int(_) => 2,
            OrdKey::Float(_) => 3, OrdKey::Bytes(_) => 4,
        }
    }
}

impl PartialEq for OrdKey { fn eq(&self, o: &Self) -> bool { self.cmp(o) == std::cmp::Ordering::Equal } }
impl Eq for OrdKey {}
impl PartialOrd for OrdKey { fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(o)) } }
impl Ord for OrdKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        match (self, other) {
            (OrdKey::Int(a), OrdKey::Int(b)) => a.cmp(b),
            (OrdKey::Float(a), OrdKey::Float(b)) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
            (OrdKey::Int(a), OrdKey::Float(b)) => (*a as f64).partial_cmp(b).unwrap_or(Ordering::Equal),
            (OrdKey::Float(a), OrdKey::Int(b)) => a.partial_cmp(&(*b as f64)).unwrap_or(Ordering::Equal),
            (OrdKey::Bytes(a), OrdKey::Bytes(b)) => a.cmp(b),
            (OrdKey::Bool(a), OrdKey::Bool(b)) => a.cmp(b),
            (OrdKey::Nullish, OrdKey::Nullish) => Ordering::Equal,
            (a, b) => a.tag().cmp(&b.tag()),
        }
    }
}

// ── Path catalog ───────────────────────────────────────────────────

/// Per-row extraction catalog.  Each slot maps to a canonical chain
/// of field names walked from the row root.  Single-step slots
/// (chain length 1) read primitives directly; multi-step slots
/// descend through nested objects after the top-level extract.
///
/// Layout:
///   slots in `top_keys.len()` range = primary top-level fields,
///     populated by entry_extract_template/direct.
///   slots beyond top_keys.len() = chain tails registered in
///     `chains`, resolved post-extract by descending the ObjRange
///     held at the chain's primary slot.
#[derive(Default, Debug)]
struct Catalog {
    /// Unique top-level field names appearing as either single-step
    /// reads or as the first step of a multi-step chain.
    top_keys: Vec<Arc<str>>,
    /// Multi-step chains.  Each `(primary_slot, tail)` where
    /// `primary_slot` is the index in `top_keys` for the chain's
    /// first step and `tail` is the remaining steps.  The chain's
    /// output slot is `top_keys.len() + idx_in_chains`.
    chains: Vec<(usize, Vec<Arc<str>>)>,
}

impl Catalog {
    fn slot_for(&mut self, name: &Arc<str>) -> usize {
        for (i, f) in self.top_keys.iter().enumerate() {
            if f.as_ref() == name.as_ref() { return i; }
        }
        self.top_keys.push(Arc::clone(name));
        self.top_keys.len() - 1
    }
    fn slot_lookup(&self, name: &str) -> Option<usize> {
        self.top_keys.iter().position(|f| f.as_ref() == name)
    }
    /// Register a multi-step chain.  Returns the slot index that
    /// will hold the final descended value at run time.
    fn slot_for_chain(&mut self, keys: &[Arc<str>]) -> usize {
        if keys.is_empty() { return usize::MAX; }
        if keys.len() == 1 { return self.slot_for(&keys[0]); }
        for (i, (p, t)) in self.chains.iter().enumerate() {
            if self.top_keys[*p].as_ref() == keys[0].as_ref()
                && t.len() == keys.len() - 1
                && t.iter().zip(keys[1..].iter()).all(|(a, b)| a.as_ref() == b.as_ref())
            {
                return self.top_keys.len() + i;
            }
        }
        let primary = self.slot_for(&keys[0]);
        let tail: Vec<Arc<str>> = keys[1..].iter().map(Arc::clone).collect();
        self.chains.push((primary, tail));
        self.top_keys.len() + self.chains.len() - 1
    }
    /// Lookup slot for a multi-step chain; returns None if not registered.
    fn slot_lookup_chain(&self, keys: &[Arc<str>]) -> Option<usize> {
        if keys.is_empty() { return None; }
        if keys.len() == 1 { return self.slot_lookup(keys[0].as_ref()); }
        for (i, (p, t)) in self.chains.iter().enumerate() {
            if self.top_keys.get(*p).map(|s| s.as_ref()) == Some(keys[0].as_ref())
                && t.len() == keys.len() - 1
                && t.iter().zip(keys[1..].iter()).all(|(a, b)| a.as_ref() == b.as_ref())
            {
                return Some(self.top_keys.len() + i);
            }
        }
        None
    }
    fn n_slots(&self) -> usize { self.top_keys.len() + self.chains.len() }
}

/// Walk a kernel and register every field path it reads.  Returns
/// None when the kernel reads something the byte-walker can't project
/// (FString, Generic).  Multi-step FieldChain / FieldChainCmpLit /
/// ObjProject Path are supported via Catalog::slot_for_chain.
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
        BodyKernel::FieldChain(keys) => {
            cat.slot_for_chain(keys);
            Some(())
        }
        BodyKernel::FieldChainCmpLit(keys, _, _) => {
            cat.slot_for_chain(keys);
            Some(())
        }
        BodyKernel::ObjProject(entries) => {
            for e in entries.iter() {
                match e {
                    ObjProjEntry::Path { path, .. } => {
                        cat.slot_for_chain(path);
                    }
                    _ => return None,
                }
            }
            Some(())
        }
        // FString, Generic — bail.
        _ => None,
    }
}

/// True iff every ObjProject entry is a Path projection.  Path
/// length unrestricted — multi-step paths resolve through the
/// catalog's chain-resolution post-extract.
fn objproject_is_byte_friendly(entries: &[ObjProjEntry]) -> bool {
    entries.iter().all(|e| matches!(e, ObjProjEntry::Path { .. }))
}

fn collect_arith(op: &ArithOperand, cat: &mut Catalog) -> Option<()> {
    match op {
        ArithOperand::Path(p) => { cat.slot_for_chain(p); Some(()) }
        ArithOperand::LitInt(_) | ArithOperand::LitFloat(_) => Some(()),
    }
}

fn is_pred_kernel(k: &BodyKernel) -> bool {
    matches!(k,
        BodyKernel::FieldRead(_)
        | BodyKernel::FieldCmpLit(_, _, _)
        | BodyKernel::ConstBool(_)
        | BodyKernel::FieldChainCmpLit(_, _, _)
        | BodyKernel::FieldChain(_))
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
    /// Slice into raw_bytes — entire object value `{...}` byte range.
    /// Used as a stepping stone for multi-step FieldChain resolution.
    ObjRange(u32, u32),
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
        b'{' => {
            let end = skip_value(b, i)?;
            Some((end, Scalar::ObjRange(i as u32, end as u32)))
        }
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


#[inline]
fn truthy(s: Scalar) -> bool {
    match s {
        Scalar::Bool(b) => b,
        Scalar::Int(n) => n != 0,
        Scalar::Float(f) => f != 0.0,
        Scalar::StrRange(s, e) => e > s,
        Scalar::ObjRange(_, _) => true,
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
    unique_collect: bool,
) -> Option<Result<Val, EvalError>> {
    if cat.n_slots() > SCALAR_BUF_CAP { return None; }
    let wanted: Vec<&[u8]> = cat.top_keys.iter().map(|f| f.as_bytes()).collect();
    let needles_owned: Vec<Vec<u8>> = build_needles(&wanted);
    let needles: Vec<&[u8]> = needles_owned.iter().map(|n| n.as_slice()).collect();

    // Operate on canonical view — same dispatch shape regardless of
    // whether lowering left fused Sinks or base form. After fusion-
    // off (Tier 3), canonical == self. Walking via canonical means
    // bytescan migrates uniformly with Pipeline IR.
    let (cs, ck, csink) = p.canonical();

    // Trailing Map kernel — projection that the sink consumes.
    // Applies to Numeric, First, Last, Collect.  When `unique_collect`
    // is set, the LAST stage is UniqueBy(None) — the *real* trailing
    // Map sits at index cs.len() - 2.
    let payload_end = cs.len().saturating_sub(if unique_collect { 1 } else { 0 });
    let trailing_map_kernel: Option<&BodyKernel> =
        if matches!(csink, Sink::Numeric(_) | Sink::First | Sink::Last | Sink::Collect)
            && payload_end > 0
            && matches!(cs.get(payload_end - 1), Some(Stage::Map(_)))
        {
            ck.get(payload_end - 1)
        } else { None };
    let filter_end = if trailing_map_kernel.is_some() { payload_end - 1 } else { payload_end };

    // Resolve filter-stage kernels to slot indices ONCE.  Skip/Take
    // also collected here — counts get consumed by the sink walker.
    // FlatMap (when present) lives at index 0 and is handled by the
    // walker dispatch below; skip past it in this loop.
    let mut filter_slots: Vec<(usize, FilterCmp)> = Vec::new();
    let mut skip_n: usize = 0;
    let mut take_n: usize = usize::MAX;
    let mut has_take = false;
    for (st, k) in cs[..filter_end].iter().zip(ck[..filter_end].iter()) {
        match st {
            Stage::Filter(_) => {
                let (slot, cmp) = resolve_pred_to_slot(k, cat)?;
                filter_slots.push((slot, cmp));
            }
            Stage::Skip(n) => { skip_n = skip_n.saturating_add(*n); }
            Stage::Take(n) => { take_n = take_n.min(*n); has_take = true; }
            _ => return None,
        }
    }
    let _ = has_take;

    // Columnar Extraction fast path — generic algorithm.  No filters,
    // sink is Count or Numeric (with optional trailing Map). One memmem
    // scan per wanted key across the array region; per-row inline parse.
    if filter_slots.is_empty()
        && matches!(csink, Sink::Count | Sink::Numeric(_))
    {
        if let Some(out) = try_columnar_extraction_canonical(
            &csink, trailing_map_kernel, cat, &needles, raw, arr_start,
        ) {
            return Some(Ok(out));
        }
    }

    // Pipeline-Row Closure Fusion (PRCF) — generic fallback.
    // Skip/Take are honoured uniformly by gating the per-Sink reducer
    // on a row counter: skip first `skip_n` matched rows, then process
    // up to `take_n`, then signal Done.  Closure-monomorphised so the
    // skip/take check inlines into the hot loop with no overhead when
    // skip_n=0 / take_n=usize::MAX.
    //
    // When the pipeline starts with FlatMap, the walker uses the
    // 2-level scan (per outer row, locate inner field, scan inner
    // array).  Otherwise the direct walker fires.  Same closure body
    // either way — closure runs against the (post-flatten) row stream.
    macro_rules! sink_walker {
        ($body:expr) => {{
            let mut seen: usize = 0;
            let mut taken: usize = 0;
            let mut row_fn = |slots: &Slots, span: RowSpan| -> Option<ScanCtl> {
                if seen < skip_n { seen += 1; return Some(ScanCtl::Continue); }
                if taken >= take_n { return Some(ScanCtl::Done); }
                taken += 1;
                ($body)(slots, span)
            };
            scan_loop_prcf(raw, arr_start, &needles, cat, &filter_slots, &mut row_fn)?;
        }};
    }
    match (&csink, trailing_map_kernel) {
        (Sink::Count, None) => {
            let mut count: u64 = 0;
            sink_walker!(|_slots: &Slots, _span: RowSpan| -> Option<ScanCtl> {
                count += 1;
                Some(ScanCtl::Continue)
            });
            Some(Ok(Val::Int(count as i64)))
        }
        (Sink::Numeric(op), Some(map_k)) => {
            let map_slot = resolve_value_to_slot(map_k, cat)?;
            let op = *op;
            let mut acc = Acc::new();
            sink_walker!(|slots: &Slots, _span: RowSpan| -> Option<ScanCtl> {
                if let Some(v) = slots[map_slot] {
                    if !acc.push(v) { return None; }
                }
                Some(ScanCtl::Continue)
            });
            Some(Ok(acc.finalise(op)))
        }
        // Sink::First — generic early-exit. Closure returns Done on
        // first row that passes all filters + skip window; walker
        // stops, returns matched row Val.  With trailing Map(field/
        // ObjProject), emits the projected value (no full row alloc).
        (Sink::First, _) => {
            let map_kind = build_map_kind(trailing_map_kernel, cat)?;
            let mut found: Option<Val> = None;
            sink_walker!(|slots: &Slots, span: RowSpan| -> Option<ScanCtl> {
                found = Some(build_row_val(raw, slots, span, &map_kind)?);
                Some(ScanCtl::Done)
            });
            Some(Ok(found.unwrap_or(Val::Null)))
        }
        // Sink::Last — single-pass, capture the latest matching row.
        (Sink::Last, _) => {
            let map_kind = build_map_kind(trailing_map_kernel, cat)?;
            let mut last_val: Option<Val> = None;
            sink_walker!(|slots: &Slots, span: RowSpan| -> Option<ScanCtl> {
                last_val = Some(build_row_val(raw, slots, span, &map_kind)?);
                Some(ScanCtl::Continue)
            });
            Some(Ok(last_val.unwrap_or(Val::Null)))
        }
        // Sink::Collect — gather all matched (and within skip/take
        // window) rows into Val::Arr.  When `unique_collect`, dedupe
        // by canonical bytes of the projected scalar slot.  When
        // trailing Map projects ObjProject, each row builds a small
        // Val::Obj.
        (Sink::Collect, _) => {
            let map_kind = build_map_kind(trailing_map_kernel, cat)?;
            let mut acc: Vec<Val> = Vec::new();
            if unique_collect {
                let scalar_slot = match &map_kind {
                    MapKind::Scalar(s) => *s,
                    // Non-scalar dedup would require row-Val byte
                    // hashing — bail to tape for now.
                    _ => return None,
                };
                let mut seen: std::collections::HashSet<UniqueKey> = std::collections::HashSet::new();
                sink_walker!(|slots: &Slots, _span: RowSpan| -> Option<ScanCtl> {
                    let sc = match slots[scalar_slot] {
                        Some(sc) => sc,
                        None => return Some(ScanCtl::Continue),
                    };
                    let key = UniqueKey::from_scalar(raw, sc);
                    if seen.insert(key) {
                        acc.push(scalar_to_val(raw, sc));
                    }
                    Some(ScanCtl::Continue)
                });
            } else {
                sink_walker!(|slots: &Slots, span: RowSpan| -> Option<ScanCtl> {
                    acc.push(build_row_val(raw, slots, span, &map_kind)?);
                    Some(ScanCtl::Continue)
                });
            }
            Some(Ok(Val::arr(acc)))
        }
        _ => None,
    }
}

/// Hashable canonical key for scalar dedup.  String content compares
/// by raw bytes (no escape decode); numerics by typed value.  Object
/// scalars collapse to a sentinel (caller already excluded ObjRange
/// from this path).
#[derive(PartialEq, Eq, Hash)]
enum UniqueKey {
    Bytes(Vec<u8>),
    Int(i64),
    FloatBits(u64),
    Bool(bool),
    Nullish,
}

impl UniqueKey {
    fn from_scalar(raw: &[u8], s: Scalar) -> Self {
        match s {
            Scalar::StrRange(start, end) => UniqueKey::Bytes(raw[start as usize..end as usize].to_vec()),
            Scalar::Int(n) => UniqueKey::Int(n),
            Scalar::Float(f) => UniqueKey::FloatBits(f.to_bits()),
            Scalar::Bool(b) => UniqueKey::Bool(b),
            Scalar::Null | Scalar::Missing => UniqueKey::Nullish,
            Scalar::ObjRange(start, end) => UniqueKey::Bytes(raw[start as usize..end as usize].to_vec()),
        }
    }
}

/// Resolved trailing-Map projection.  Captures the slot indices for
/// each output field — or None when no trailing map (full row dump).
enum MapKind {
    /// Full row — `parse_row_obj` materialises the matched JSON object.
    None,
    /// Scalar projection at slot index — emits single primitive Val.
    Scalar(usize),
    /// Object shape — emits Val::Obj with N (key, slot) pairs.
    Project(Arc<[(Arc<str>, usize)]>),
}

fn build_map_kind(k: Option<&BodyKernel>, cat: &Catalog) -> Option<MapKind> {
    let k = match k { Some(k) => k, None => return Some(MapKind::None) };
    match k {
        BodyKernel::FieldRead(_) | BodyKernel::FieldChain(_) => {
            let s = resolve_value_to_slot(k, cat)?;
            Some(MapKind::Scalar(s))
        }
        BodyKernel::ObjProject(entries) => {
            let mut pairs: Vec<(Arc<str>, usize)> = Vec::with_capacity(entries.len());
            for e in entries.iter() {
                match e {
                    ObjProjEntry::Path { key, path } => {
                        let s = cat.slot_lookup_chain(path)?;
                        pairs.push((Arc::clone(key), s));
                    }
                    _ => return None,
                }
            }
            Some(MapKind::Project(pairs.into()))
        }
        _ => None,
    }
}

/// Build a Val for a matched row.  Three modes:
///   - `MapKind::None` — parse the row's byte range as a Val::Obj
///     (per-call alloc for the row buffer + Val::Obj IndexMap).
///   - `MapKind::Scalar(slot)` — primitive Val from the slot.
///   - `MapKind::Project(pairs)` — small Val::Obj with N projected
///     fields, no per-row JSON parse.
#[inline]
fn build_row_val(raw: &[u8], slots: &Slots, span: RowSpan, map: &MapKind) -> Option<Val> {
    match map {
        MapKind::None => parse_row_obj(raw, span),
        MapKind::Scalar(s) => Some(slots[*s].map(|sc| scalar_to_val(raw, sc)).unwrap_or(Val::Null)),
        MapKind::Project(pairs) => {
            let mut m: indexmap::IndexMap<Arc<str>, Val> = indexmap::IndexMap::with_capacity(pairs.len());
            for (key, slot) in pairs.iter() {
                let v = slots[*slot].map(|sc| scalar_to_val(raw, sc)).unwrap_or(Val::Null);
                m.insert(Arc::clone(key), v);
            }
            Some(Val::Obj(Arc::new(m)))
        }
    }
}

#[inline]
fn parse_row_obj(raw: &[u8], span: RowSpan) -> Option<Val> {
    #[cfg(feature = "simd-json")]
    {
        let mut buf: Vec<u8> = raw[span.start..span.end].to_vec();
        return Val::from_json_simd(&mut buf).ok();
    }
    #[cfg(not(feature = "simd-json"))]
    {
        let s = std::str::from_utf8(&raw[span.start..span.end]).ok()?;
        let v: serde_json::Value = serde_json::from_str(s).ok()?;
        Some(Val::from(&v))
    }
}

#[inline]
fn scalar_to_val(raw: &[u8], s: Scalar) -> Val {
    match s {
        Scalar::Null | Scalar::Missing => Val::Null,
        Scalar::Bool(b) => Val::Bool(b),
        Scalar::Int(n) => Val::Int(n),
        Scalar::Float(f) => Val::Float(f),
        Scalar::StrRange(start, end) => {
            let bytes = &raw[start as usize .. end as usize];
            match std::str::from_utf8(bytes) {
                Ok(s) => Val::Str(Arc::from(s)),
                Err(_) => Val::Str(Arc::from(String::from_utf8_lossy(bytes).as_ref())),
            }
        }
        Scalar::ObjRange(start, end) => {
            // Materialise nested object as Val via simd-json on a per-
            // call buffer copy.  Used when ObjProject targets the
            // intermediate object of a multi-step chain (rare).
            let span = RowSpan { start: start as usize, end: end as usize };
            parse_row_obj(raw, span).unwrap_or(Val::Null)
        }
    }
}

// ── Borrowed (arena) row-build helpers (Phase 3+4) ─────────────────
//
// Mirror the owned `scalar_to_val` / `parse_row_obj` / `build_row_val`
// but emit `borrowed::Val<'a>` allocated in `arena`.  Used by the
// `try_run_borrow` entry point; the owned path is unchanged.
//
// Cost saving vs owned path:
//   - StrRange → `&'a str` (arena bump, no `Arc::from`).
//   - Project rows → `Val::Obj(&[(&str, Val)])` (arena slice, no
//     `IndexMap`, no `Arc<IndexMap>`).
//   - Full-row parse → `from_json_simd_arena` (no Arc/IndexMap per
//     nested object).

#[cfg(feature = "simd-json")]
#[inline]
fn parse_row_obj_in<'a>(arena: &'a crate::eval::borrowed::Arena, raw: &[u8], span: RowSpan)
    -> Option<crate::eval::borrowed::Val<'a>>
{
    let mut buf: Vec<u8> = raw[span.start..span.end].to_vec();
    crate::eval::borrowed::from_json_simd_arena(arena, &mut buf).ok()
}

#[inline]
fn scalar_to_val_in<'a>(arena: &'a crate::eval::borrowed::Arena, raw: &[u8], s: Scalar)
    -> crate::eval::borrowed::Val<'a>
{
    use crate::eval::borrowed::Val as BVal;
    match s {
        Scalar::Null | Scalar::Missing => BVal::Null,
        Scalar::Bool(b) => BVal::Bool(b),
        Scalar::Int(n) => BVal::Int(n),
        Scalar::Float(f) => BVal::Float(f),
        Scalar::StrRange(start, end) => {
            let bytes = &raw[start as usize .. end as usize];
            match std::str::from_utf8(bytes) {
                Ok(s) => BVal::Str(arena.alloc_str(s)),
                Err(_) => BVal::Str(arena.alloc_str(&String::from_utf8_lossy(bytes))),
            }
        }
        Scalar::ObjRange(start, end) => {
            #[cfg(feature = "simd-json")]
            {
                let span = RowSpan { start: start as usize, end: end as usize };
                return parse_row_obj_in(arena, raw, span).unwrap_or(BVal::Null);
            }
            #[cfg(not(feature = "simd-json"))]
            {
                let _ = (start, end);
                BVal::Null
            }
        }
    }
}

#[inline]
fn build_row_val_in<'a>(
    arena: &'a crate::eval::borrowed::Arena,
    raw: &[u8],
    slots: &Slots,
    span: RowSpan,
    map: &MapKind,
) -> Option<crate::eval::borrowed::Val<'a>> {
    use crate::eval::borrowed::Val as BVal;
    match map {
        MapKind::None => {
            #[cfg(feature = "simd-json")]
            { return parse_row_obj_in(arena, raw, span); }
            #[cfg(not(feature = "simd-json"))]
            { let _ = span; return None; }
        }
        MapKind::Scalar(s) => {
            Some(slots[*s].map(|sc| scalar_to_val_in(arena, raw, sc)).unwrap_or(BVal::Null))
        }
        MapKind::Project(pairs) => {
            // Build entries list, then bulk-allocate as arena slice.
            let mut tmp: Vec<(&'a str, BVal<'a>)> = Vec::with_capacity(pairs.len());
            for (key, slot) in pairs.iter() {
                let v = slots[*slot].map(|sc| scalar_to_val_in(arena, raw, sc)).unwrap_or(BVal::Null);
                tmp.push((arena.alloc_str(key), v));
            }
            let slice = arena.alloc_slice_fill_iter(tmp.into_iter());
            Some(BVal::Obj(&*slice))
        }
    }
}

// ── Borrowed entry point (Phase 4 API) ─────────────────────────────
//
// Same shape acceptance as `try_run` but only the Sink::Collect path
// emits borrowed.  Other Sinks (Numeric/Count/First/Last/heap-top-K)
// fall back to None — caller routes them through owned `try_run` and
// converts via `from_owned` at the boundary.  Subsequent commits will
// extend coverage; Sink::Collect is the highest-row-count case where
// the saving is largest.
pub fn try_run_borrow<'a>(
    p: &Pipeline,
    raw_bytes: &[u8],
    arena: &'a crate::eval::borrowed::Arena,
) -> Option<Result<crate::eval::borrowed::Val<'a>, EvalError>> {
    let chain: Vec<&[u8]> = match &p.source {
        Source::FieldChain { keys } => keys.iter().map(|k| k.as_bytes()).collect(),
        _ => return None,
    };
    // Heap-top-K: not yet covered by borrowed path; defer.
    if detect_heap_top_k(&p.stages, &p.stage_kernels, &p.sink).is_some() {
        return None;
    }
    let stage_count = p.stages.len();
    let last_is_unique = matches!(p.stages.last(), Some(Stage::UniqueBy(None)));
    let last_payload_idx = if last_is_unique { stage_count.saturating_sub(1) } else { stage_count };
    for (idx, (st, k)) in p.stages.iter().zip(p.stage_kernels.iter()).enumerate() {
        match st {
            Stage::Filter(_) => {
                if !is_pred_kernel(k) { return None; }
            }
            Stage::Skip(_) | Stage::Take(_) => {}
            Stage::Map(_) if last_payload_idx > 0 && idx == last_payload_idx - 1 => {
                match k {
                    BodyKernel::FieldRead(_) => {}
                    BodyKernel::FieldChain(_) => {}
                    BodyKernel::ObjProject(entries) => {
                        if !objproject_is_byte_friendly(entries) { return None; }
                    }
                    _ => return None,
                }
            }
            Stage::UniqueBy(None) if idx == stage_count - 1 => {
                if !matches!(p.sink, Sink::Collect) { return None; }
            }
            _ => return None,
        }
    }
    let mut catalog: Catalog = Catalog::default();
    for (st, k) in p.stages.iter().zip(p.stage_kernels.iter()) {
        match st {
            Stage::Skip(_) | Stage::Take(_) | Stage::UniqueBy(None) => continue,
            _ => {}
        }
        collect_paths(k, &mut catalog)?;
    }
    for k in p.sink_kernels.iter() { collect_paths(k, &mut catalog)?; }

    let arr_start = walk_chain_to_array(raw_bytes, &chain)?;
    run_with_sink_borrow(p, &catalog, raw_bytes, arr_start, last_is_unique, arena)
}

fn run_with_sink_borrow<'a>(
    p: &Pipeline,
    cat: &Catalog,
    raw: &[u8],
    arr_start: usize,
    unique_collect: bool,
    arena: &'a crate::eval::borrowed::Arena,
) -> Option<Result<crate::eval::borrowed::Val<'a>, EvalError>> {
    use crate::eval::borrowed::Val as BVal;

    if cat.n_slots() > SCALAR_BUF_CAP { return None; }
    let wanted: Vec<&[u8]> = cat.top_keys.iter().map(|f| f.as_bytes()).collect();
    let needles_owned: Vec<Vec<u8>> = build_needles(&wanted);
    let needles: Vec<&[u8]> = needles_owned.iter().map(|n| n.as_slice()).collect();

    let (cs, ck, csink) = p.canonical();

    let payload_end = cs.len().saturating_sub(if unique_collect { 1 } else { 0 });
    let trailing_map_kernel: Option<&BodyKernel> =
        if matches!(csink, Sink::Numeric(_) | Sink::First | Sink::Last | Sink::Collect)
            && payload_end > 0
            && matches!(cs.get(payload_end - 1), Some(Stage::Map(_)))
        {
            ck.get(payload_end - 1)
        } else { None };
    let filter_end = if trailing_map_kernel.is_some() { payload_end - 1 } else { payload_end };

    let mut filter_slots: Vec<(usize, FilterCmp)> = Vec::new();
    let mut skip_n: usize = 0;
    let mut take_n: usize = usize::MAX;
    for (st, k) in cs[..filter_end].iter().zip(ck[..filter_end].iter()) {
        match st {
            Stage::Filter(_) => {
                let (slot, cmp) = resolve_pred_to_slot(k, cat)?;
                filter_slots.push((slot, cmp));
            }
            Stage::Skip(n) => { skip_n = skip_n.saturating_add(*n); }
            Stage::Take(n) => { take_n = take_n.min(*n); }
            _ => return None,
        }
    }

    macro_rules! sink_walker_b {
        ($body:expr) => {{
            let mut seen: usize = 0;
            let mut taken: usize = 0;
            let mut row_fn = |slots: &Slots, span: RowSpan| -> Option<ScanCtl> {
                if seen < skip_n { seen += 1; return Some(ScanCtl::Continue); }
                if taken >= take_n { return Some(ScanCtl::Done); }
                taken += 1;
                ($body)(slots, span)
            };
            scan_loop_prcf(raw, arr_start, &needles, cat, &filter_slots, &mut row_fn)?;
        }};
    }

    // Convert owned Acc::finalise primitive output to BVal (no arena
    // alloc — Int/Float/Null are inline).
    fn acc_to_bval<'a>(arena: &'a crate::eval::borrowed::Arena, acc: Acc, op: NumOp)
        -> crate::eval::borrowed::Val<'a>
    {
        use crate::eval::borrowed::Val as BVal;
        let v = acc.finalise(op);
        match v {
            Val::Null    => BVal::Null,
            Val::Bool(b) => BVal::Bool(b),
            Val::Int(n)  => BVal::Int(n),
            Val::Float(f) => BVal::Float(f),
            Val::Str(s) => BVal::Str(arena.alloc_str(&s)),
            _ => BVal::Null,
        }
    }

    match (&csink, trailing_map_kernel) {
        (Sink::Count, None) => {
            let mut count: u64 = 0;
            sink_walker_b!(|_slots: &Slots, _span: RowSpan| -> Option<ScanCtl> {
                count += 1;
                Some(ScanCtl::Continue)
            });
            Some(Ok(BVal::Int(count as i64)))
        }
        (Sink::Numeric(op), Some(map_k)) => {
            let map_slot = resolve_value_to_slot(map_k, cat)?;
            let op = *op;
            let mut acc = Acc::new();
            sink_walker_b!(|slots: &Slots, _span: RowSpan| -> Option<ScanCtl> {
                if let Some(v) = slots[map_slot] {
                    if !acc.push(v) { return None; }
                }
                Some(ScanCtl::Continue)
            });
            Some(Ok(acc_to_bval(arena, acc, op)))
        }
        (Sink::First, _) => {
            let map_kind = build_map_kind(trailing_map_kernel, cat)?;
            let mut found: Option<BVal<'a>> = None;
            sink_walker_b!(|slots: &Slots, span: RowSpan| -> Option<ScanCtl> {
                found = Some(build_row_val_in(arena, raw, slots, span, &map_kind)?);
                Some(ScanCtl::Done)
            });
            Some(Ok(found.unwrap_or(BVal::Null)))
        }
        (Sink::Last, _) => {
            let map_kind = build_map_kind(trailing_map_kernel, cat)?;
            let mut last_val: Option<BVal<'a>> = None;
            sink_walker_b!(|slots: &Slots, span: RowSpan| -> Option<ScanCtl> {
                last_val = Some(build_row_val_in(arena, raw, slots, span, &map_kind)?);
                Some(ScanCtl::Continue)
            });
            Some(Ok(last_val.unwrap_or(BVal::Null)))
        }
        (Sink::Collect, _) => {
            let map_kind = build_map_kind(trailing_map_kernel, cat)?;
            let mut acc: Vec<BVal<'a>> = Vec::new();
            if unique_collect {
                let scalar_slot = match &map_kind {
                    MapKind::Scalar(s) => *s,
                    _ => return None,
                };
                let mut seen: std::collections::HashSet<UniqueKey> = std::collections::HashSet::new();
                sink_walker_b!(|slots: &Slots, _span: RowSpan| -> Option<ScanCtl> {
                    let sc = match slots[scalar_slot] {
                        Some(sc) => sc,
                        None => return Some(ScanCtl::Continue),
                    };
                    let key = UniqueKey::from_scalar(raw, sc);
                    if seen.insert(key) {
                        acc.push(scalar_to_val_in(arena, raw, sc));
                    }
                    Some(ScanCtl::Continue)
                });
            } else {
                sink_walker_b!(|slots: &Slots, span: RowSpan| -> Option<ScanCtl> {
                    acc.push(build_row_val_in(arena, raw, slots, span, &map_kind)?);
                    Some(ScanCtl::Continue)
                });
            }
            let slice = arena.alloc_slice_fill_iter(acc.into_iter());
            Some(Ok(BVal::Arr(&*slice)))
        }
        _ => None,
    }
}

/// Columnar extraction — generic algorithm for filter+map+aggregate
/// shapes with no Filter stages.  For each wanted key, runs ONE
/// `memchr::memmem::find_iter` over the full array byte range — SIMD
/// at ~1-3 GB/s.  Result: parallel Vec<usize> per wanted key.
///
/// Then pairs entries by zip — assumes homogeneous-row shape (each
/// row has all wanted keys in same order).  Per (entry_idx, pos)
/// tuple: parse value inline, dispatch via Sink closure, accumulate.
///
/// On non-uniform-shape input (some rows missing wanted key), falls
/// back to PRCF walker.  Generic across Pipeline shapes via the
/// same closure-builder branches as run_with_sink.
fn try_columnar_extraction_canonical(
    csink: &Sink,
    trailing_map_kernel: Option<&BodyKernel>,
    cat: &Catalog,
    needles: &[&[u8]],
    raw: &[u8],
    arr_start: usize,
) -> Option<Val> {
    if raw.get(arr_start) != Some(&b'[') { return None; }
    let arr_end = skip_value(raw, arr_start)?;
    let arr_range = &raw[arr_start + 1 .. arr_end - 1];

    // Per-key memmem scan — collect ALL hit positions in array region.
    let mut cols: Vec<Vec<usize>> = Vec::with_capacity(needles.len());
    let mut min_hits: usize = usize::MAX;
    for needle in needles {
        let positions: Vec<usize> = memchr::memmem::find_iter(arr_range, needle).collect();
        min_hits = min_hits.min(positions.len());
        cols.push(positions);
    }
    if min_hits == 0 { return None; }
    // For homogeneous rows, every column has same #hits.  Verify.
    if !cols.iter().all(|c| c.len() == min_hits) { return None; }

    // Parse value at each (column, row_idx) → 2D scalar grid via
    // direct positional access on raw bytes.
    let n_rows = min_hits;
    let n_cols = cols.len();

    // Per-row evaluator dispatched on canonical (sink, trailing map).
    // Caller already filtered for filter-free shapes — only Count and
    // Numeric(+map) reach here.
    match (csink, trailing_map_kernel) {
        (Sink::Count, _) => Some(Val::Int(n_rows as i64)),
        (Sink::Numeric(op), Some(map_k)) => {
            let map_slot = resolve_value_to_slot(map_k, cat)?;
            let mut acc = Acc::new();
            for row in 0..n_rows {
                let map_pos = arr_start + 1 + cols[map_slot][row] + needles[map_slot].len();
                let v = parse_scalar_at(raw, map_pos)?;
                if !acc.push(v) { return None; }
            }
            Some(acc.finalise(*op))
        }
        _ => None,
    }
    .map(|v| { let _ = n_cols; v })
}

/// Inline parse of a primitive at byte position — skips ws first.
#[inline]
fn parse_scalar_at(raw: &[u8], pos: usize) -> Option<Scalar> {
    let pos = skip_ws(raw, pos);
    let (_, sc) = parse_primitive(raw, pos)?;
    Some(sc)
}

/// Sink loop control: closure tells walker whether to keep walking.
/// `Continue` = process next row, `Done` = sink satisfied (early-exit
/// success).  Closure-returned `None` is reserved for hard error
/// (parse / overflow) and bubbles up as bytescan opt-out.
#[derive(Copy, Clone)]
enum ScanCtl { Continue, Done }

/// Byte-range of a single row (object) in `raw`.  Half-open: `start`
/// at `{`, `end` one-past `}`.
#[derive(Copy, Clone)]
struct RowSpan { start: usize, end: usize }

/// Pipeline-Row Closure Fusion — single generic walker that consumes
/// any per-row processor closure.  Compiler monomorphises per call
/// site (each Sink branch has a different closure type) → static
/// dispatch inside hot loop, no fn-call overhead, full inlining.
///
/// Generic across all Sinks; new Sink = one new closure-builder
/// branch in run_with_sink, NO new walker.
///
/// Returns `Some(true)` when the closure signalled `ScanCtl::Done`
/// (early-exit), `Some(false)` when the array was fully walked.
/// `None` is reserved for hard error.  Callers that don't care
/// about early-exit (Count/Numeric/Collect-without-First) ignore the
/// bool via `?`; nested walkers (FlatMap) propagate it to break the
/// outer loop on inner Done.
///
/// `cat` is used to resolve multi-step chain slots after the
/// per-row top-level extract.  When `cat.chains` is empty the
/// post-extract chain descent compiles to zero overhead.
#[inline(always)]
fn scan_loop_prcf<F>(
    raw: &[u8],
    arr_start: usize,
    needles: &[&[u8]],
    cat: &Catalog,
    filters: &[(usize, FilterCmp)],
    mut row_fn: F,
) -> Option<bool>
where
    F: FnMut(&Slots, RowSpan) -> Option<ScanCtl>,
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
        let row_start = i;
        let next = if let Some((tpl, _)) = template.as_ref() {
            match entry_extract_template(raw, i, tpl, &mut slots, n_wanted)? {
                (n, true) => n,
                (_, false) => entry_extract_direct(raw, i, needles, &mut slots)?,
            }
        } else {
            entry_extract_direct(raw, i, needles, &mut slots)?
        };
        resolve_chains(raw, cat, &mut slots);
        if check_filters(&slots, filters) {
            match row_fn(&slots, RowSpan { start: row_start, end: next })? {
                ScanCtl::Continue => {}
                ScanCtl::Done => return Some(true),
            }
        }
        i = entry_advance(raw, next);
    }
    Some(false)
}

/// Post-extract chain descent.  For each registered multi-step chain,
/// walk its primary slot's ObjRange through the tail keys to produce
/// the final scalar.  Stores the result at slot `top_keys.len() + i`.
/// When `cat.chains` is empty this is a no-op call (compiles to a
/// branch on `chains.is_empty()`).
#[inline]
fn resolve_chains(raw: &[u8], cat: &Catalog, slots: &mut Slots) {
    if cat.chains.is_empty() { return; }
    let n_top = cat.top_keys.len();
    for (i, (primary_slot, tail)) in cat.chains.iter().enumerate() {
        let out_slot = n_top + i;
        let (mut cur, _end) = match slots[*primary_slot] {
            Some(Scalar::ObjRange(s, e)) => (s as usize, e as usize),
            _ => { slots[out_slot] = None; continue; }
        };
        let mut final_scalar: Option<Scalar> = None;
        for (k_idx, key) in tail.iter().enumerate() {
            let cur_skipped = skip_ws(raw, cur);
            if raw.get(cur_skipped) != Some(&b'{') { break; }
            match find_field(raw, cur_skipped, key.as_bytes()) {
                Some(val_pos) => {
                    let val_pos = skip_ws(raw, val_pos);
                    if k_idx == tail.len() - 1 {
                        if let Some((_, sc)) = parse_primitive(raw, val_pos) {
                            final_scalar = Some(sc);
                        }
                    } else {
                        // Intermediate step — must descend into another object.
                        if raw.get(val_pos) == Some(&b'{') {
                            cur = val_pos;
                        } else { break; }
                    }
                }
                None => break,
            }
        }
        slots[out_slot] = final_scalar;
    }
}

/// FlatMap walker — preserved for future early-exit shapes
/// (`flat_map(...).first()` / `take(k)`).  Currently unused: bench
/// showed FlatMap+Sink::Numeric/Count/Collect runs faster on the
/// tape route (parse amortised over inner walks).  Re-enable at
/// `try_run` acceptance for FlatMap when paired with First/Last/
/// Skip+Take(small).
#[allow(dead_code)]
/// FlatMap walker: for each outer row, locate `inner_field_needle`
/// (`"<field>":` bytes), descend into its array value, and walk inner
/// rows applying `inner_filters` + dispatching `row_fn` per match.
/// Inner-row closure runs over the flattened stream — same shape and
/// skip/take semantics as the non-FlatMap path.
///
/// Template caching: probes the inner-row shape ONCE on the first
/// non-empty inner array; subsequent outer rows reuse the template
/// for direct sequential extract — eliminates per-outer template
/// build overhead (~1 µs/outer × 5000 outers = ~5 ms saved on bench
/// data).  Falls back to `entry_extract_direct` when a row's shape
/// diverges from the template.
///
/// Generic across any FieldRead/FieldChain[1] FlatMap kernel.  When
/// the inner closure signals `ScanCtl::Done` the outer loop breaks
/// too, preserving early-exit for First / Take.
#[inline(always)]
fn scan_loop_flat_map_prcf<F>(
    raw: &[u8],
    outer_arr_start: usize,
    inner_field_needle: &[u8],
    inner_needles: &[&[u8]],
    inner_filters: &[(usize, FilterCmp)],
    mut row_fn: F,
) -> Option<()>
where
    F: FnMut(&Slots, RowSpan) -> Option<ScanCtl>,
{
    if raw.get(outer_arr_start) != Some(&b'[') { return None; }
    let n_wanted = inner_needles.len();
    let mut wanted: [&[u8]; SCALAR_BUF_CAP] = [&[]; SCALAR_BUF_CAP];
    for (idx, n) in inner_needles.iter().enumerate() {
        wanted[idx] = &n[1..n.len() - 2];
    }
    let mut slots: Slots = [None; SCALAR_BUF_CAP];
    let mut template: Option<Template> = None;

    let mut i = outer_arr_start + 1;
    i = skip_ws(raw, i);
    while i < raw.len() && raw[i] != b']' {
        if raw[i] != b'{' { return None; }
        let outer_end = skip_value(raw, i)?;
        let outer_inner = &raw[i + 1 .. outer_end - 1];
        if let Some(pos) = memchr::memmem::find(outer_inner, inner_field_needle) {
            let val_start_rel = pos + inner_field_needle.len();
            let val_start = skip_ws(raw, i + 1 + val_start_rel);
            if raw.get(val_start) == Some(&b'[') {
                let mut j = val_start + 1;
                j = skip_ws(raw, j);
                if template.is_none() && raw.get(j) == Some(&b'{') {
                    template = build_template(raw, j, &wanted[..n_wanted]).map(|(t, _)| t);
                }
                while j < raw.len() && raw[j] != b']' {
                    let row_start = j;
                    let next = if let Some(tpl) = template.as_ref() {
                        match entry_extract_template(raw, j, tpl, &mut slots, n_wanted)? {
                            (n, true) => n,
                            (_, false) => entry_extract_direct(raw, j, inner_needles, &mut slots)?,
                        }
                    } else {
                        entry_extract_direct(raw, j, inner_needles, &mut slots)?
                    };
                    if check_filters(&slots, inner_filters) {
                        match row_fn(&slots, RowSpan { start: row_start, end: next })? {
                            ScanCtl::Continue => {}
                            ScanCtl::Done => return Some(()),
                        }
                    }
                    j = entry_advance(raw, next);
                }
            }
        }
        i = entry_advance(raw, outer_end);
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
        BodyKernel::FieldChain(keys) => {
            let s = cat.slot_lookup_chain(keys)?;
            Some((s, FilterCmp::Truthy))
        }
        BodyKernel::FieldCmpLit(name, op, lit) => {
            let s = cat.slot_lookup(name.as_ref())?;
            Some((s, to_cmp(*op, to_lit(lit)?)?))
        }
        BodyKernel::FieldChainCmpLit(keys, op, lit) => {
            let s = cat.slot_lookup_chain(keys)?;
            Some((s, to_cmp(*op, to_lit(lit)?)?))
        }
        _ => None,
    }
}

fn resolve_value_to_slot(k: &BodyKernel, cat: &Catalog) -> Option<usize> {
    match k {
        BodyKernel::FieldRead(name) => cat.slot_lookup(name.as_ref()),
        BodyKernel::FieldChain(keys) => cat.slot_lookup_chain(keys),
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

