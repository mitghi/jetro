//! Tape-source borrowed pipeline substrate — Phase 1.
//!
//! Mirror of `composed_borrow.rs` but reads from a simd-json `TapeData`
//! rather than a materialised `BVal<'a>` tree.  Lets stages run cold
//! against raw tape (no Val build) and emit `BVal<'a>` only when a
//! sink demands materialisation (Collect / First / Last with full row).
//!
//! Closes the gap noted in `pipeline_borrow.rs`: when bytescan_borrow
//! declines, the owned tape path beats arena-Val path because owned
//! avoids Val tree construction.  composed_tape gives the borrowed
//! path the same Val-free advantage.
//!
//! ## Phase 1 (this commit) — substrate only, NOT WIRED
//!
//! - `TapeRow<'a>` cursor: `(&'a TapeData, idx: u32)`.  Methods:
//!   `get_field`, `walk_path`, primitive accessors, `materialise_into`.
//! - `TapeStageOutputT<'a>`: Pass(TapeRow) | Filtered | Done.
//!   No `Many` — FlatMap deferred to Phase 2 (needs sub-array iteration).
//! - `TapeStageT` trait + `IdentityT`, `ComposedT`, `FilterT`,
//!   `MapFieldT`, `MapFieldChainT`, `TakeT`, `SkipT`.
//! - `TapeSinkT` trait (GAT-Acc) + 5 sinks: Count, Sum, Min, Max, Avg.
//!   First/Last/Collect deferred to Phase 2 (need materialisation +
//!   lifetime gymnastics on Acc).
//! - `run_pipeline_t` outer loop over an array node's elements.
//! - 8 round-trip tests.
//!
//! ## Phase 2: First/Last/Collect (materialising sinks)
//!
//! Acc holds `Option<BVal<'a>>` / `Vec<BVal<'a>>`; `fold` materialises
//! the row into the arena.  Phase 1 deferred this because
//! `materialise_into` walks the full subtree — for non-Collect sinks
//! it's wasted work and the tape advantage disappears.
//!
//! ## Phase 3: wire into `Jetro::collect_val_borrow`
//!
//! After bytescan_borrow declines + before owned fallback.  Catches
//! shapes where bytescan misses but owned would have used tape (e.g.
//! string-lit filter predicates).
//!
//! Requires `simd-json` cargo feature — depends on `crate::strref::TapeData`.

#![cfg(feature = "simd-json")]

use crate::eval::borrowed::{Arena, Val as BVal};
use crate::strref::{TapeData, TapeNode};
use simd_json::StaticNode as SN;

// ── TapeRow cursor ──────────────────────────────────────────────────

/// Cursor pointing at a tape node.  Cheap to copy (16 bytes — &TapeData
/// + u32 idx).  Operations walk the tape on demand; primitives are
/// extracted without allocation, materialisation only on explicit
/// `materialise_into(arena)`.
#[derive(Copy, Clone)]
pub struct TapeRow<'a> {
    pub tape: &'a TapeData,
    pub idx:  u32,
}

impl<'a> TapeRow<'a> {
    #[inline]
    pub fn new(tape: &'a TapeData, idx: u32) -> Self {
        Self { tape, idx }
    }

    #[inline]
    fn node(&self) -> TapeNode { self.tape.nodes[self.idx as usize] }

    #[inline]
    pub fn is_object(&self) -> bool { matches!(self.node(), TapeNode::Object { .. }) }

    #[inline]
    pub fn is_array(&self) -> bool { matches!(self.node(), TapeNode::Array { .. }) }

    /// Resolve a field on an Object node.  Returns a sub-cursor at the
    /// value subtree, or None when the node isn't an Object or the
    /// key is absent.  Linear scan over object entries (matches owned
    /// `tape_object_field`); ~5-20 keys typical, no advantage to
    /// hash dispatch at this scale.
    #[inline]
    pub fn get_field(&self, key: &str) -> Option<TapeRow<'a>> {
        let i = self.idx as usize;
        if let TapeNode::Object { len, .. } = self.tape.nodes[i] {
            let mut j = i + 1;
            for _ in 0..len {
                let k = self.tape.str_at(j);
                j += 1;
                if k == key { return Some(TapeRow::new(self.tape, j as u32)); }
                j += self.tape.span(j);
            }
        }
        None
    }

    /// Walk a chain of object-field steps.
    pub fn walk_path(&self, chain: &[&str]) -> Option<TapeRow<'a>> {
        let mut cur = *self;
        for k in chain { cur = cur.get_field(k)?; }
        Some(cur)
    }

    /// Borrow the string contents at this node (zero-copy view into
    /// the tape's `bytes_buf`).  Panics if the node isn't a StringRef.
    #[inline]
    pub fn as_str_unchecked(&self) -> &'a str { self.tape.str_at(self.idx as usize) }

    #[inline]
    pub fn as_str(&self) -> Option<&'a str> {
        match self.node() {
            TapeNode::StringRef { .. } => Some(self.as_str_unchecked()),
            _ => None,
        }
    }

    #[inline]
    pub fn as_int(&self) -> Option<i64> {
        match self.node() {
            TapeNode::Static(SN::I64(n)) => Some(n),
            TapeNode::Static(SN::U64(n)) if n <= i64::MAX as u64 => Some(n as i64),
            _ => None,
        }
    }

    #[inline]
    pub fn as_float(&self) -> Option<f64> {
        match self.node() {
            TapeNode::Static(SN::F64(f)) => Some(f),
            TapeNode::Static(SN::I64(n)) => Some(n as f64),
            TapeNode::Static(SN::U64(n)) => Some(n as f64),
            _ => None,
        }
    }

    #[inline]
    pub fn as_bool(&self) -> Option<bool> {
        match self.node() {
            TapeNode::Static(SN::Bool(b)) => Some(b),
            _ => None,
        }
    }

    #[inline]
    pub fn is_null(&self) -> bool {
        matches!(self.node(), TapeNode::Static(SN::Null))
    }

    /// Materialise this subtree into a `BVal<'a>` allocated in `arena`.
    /// Used by sinks that emit row payloads (First/Last/Collect — Phase 2).
    /// Cost is O(subtree-node-count); for primitives it's a single
    /// match + arena bump (string).
    pub fn materialise_into(&self, arena: &'a Arena) -> BVal<'a> {
        match self.node() {
            TapeNode::Static(SN::Null)    => BVal::Null,
            TapeNode::Static(SN::Bool(b)) => BVal::Bool(b),
            TapeNode::Static(SN::I64(n))  => BVal::Int(n),
            TapeNode::Static(SN::U64(n))  => {
                if n <= i64::MAX as u64 { BVal::Int(n as i64) }
                else { BVal::Float(n as f64) }
            }
            TapeNode::Static(SN::F64(f))  => BVal::Float(f),
            TapeNode::StringRef { .. }    => BVal::Str(arena.alloc_str(self.as_str_unchecked())),
            TapeNode::Array { len, .. } => {
                let mut tmp: Vec<BVal<'a>> = Vec::with_capacity(len as usize);
                let mut j = self.idx as usize + 1;
                for _ in 0..len {
                    let cur = TapeRow::new(self.tape, j as u32);
                    tmp.push(cur.materialise_into(arena));
                    j += self.tape.span(j);
                }
                BVal::Arr(arena.alloc_slice_copy(&tmp))
            }
            TapeNode::Object { len, .. } => {
                let mut tmp: Vec<(&'a str, BVal<'a>)> = Vec::with_capacity(len as usize);
                let mut j = self.idx as usize + 1;
                for _ in 0..len {
                    let k = self.tape.str_at(j);
                    j += 1;
                    let cur = TapeRow::new(self.tape, j as u32);
                    tmp.push((arena.alloc_str(k), cur.materialise_into(arena)));
                    j += self.tape.span(j);
                }
                let slice = arena.alloc_slice_fill_iter(tmp.into_iter());
                BVal::Obj(&*slice)
            }
        }
    }
}

// ── Stage trait + output ────────────────────────────────────────────

pub enum TapeStageOutputT<'a> {
    Pass(TapeRow<'a>),
    Filtered,
    Done,
    /// Expanding stage (FlatMap) — emits a sub-array iterator.  The
    /// outer loop iterates each child cursor in turn, dispatching
    /// downstream stages per element.  Stored as `(tape, parent_idx)`
    /// where parent_idx points at an Array node; iteration walks its
    /// children via tape.span() skip-ahead.
    Many { tape: &'a TapeData, parent_idx: u32 },
}

pub trait TapeStageT {
    fn apply<'a>(&self, x: TapeRow<'a>) -> TapeStageOutputT<'a>;
}

impl<T: TapeStageT + ?Sized> TapeStageT for Box<T> {
    #[inline]
    fn apply<'a>(&self, x: TapeRow<'a>) -> TapeStageOutputT<'a> {
        (**self).apply(x)
    }
}

pub struct IdentityT;
impl TapeStageT for IdentityT {
    #[inline]
    fn apply<'a>(&self, x: TapeRow<'a>) -> TapeStageOutputT<'a> {
        TapeStageOutputT::Pass(x)
    }
}

pub struct ComposedT<A: TapeStageT, B: TapeStageT> {
    pub a: A,
    pub b: B,
}

impl<A: TapeStageT, B: TapeStageT> TapeStageT for ComposedT<A, B> {
    fn apply<'a>(&self, x: TapeRow<'a>) -> TapeStageOutputT<'a> {
        match self.a.apply(x) {
            TapeStageOutputT::Pass(v) => self.b.apply(v),
            TapeStageOutputT::Filtered => TapeStageOutputT::Filtered,
            TapeStageOutputT::Done => TapeStageOutputT::Done,
            // Many propagation: ComposedT cannot flatten Many across
            // sub-stage `b` without buffering, so it surfaces Many to
            // the outer loop unchanged.  The outer loop unrolls Many
            // and re-applies the FULL stage chain (a∘b) per child;
            // since Many can only originate at a FlatMap stage at
            // index 0 of the chain (lower_stage validates), surfacing
            // here yields correct per-child traversal at the outer
            // run_pipeline_t level (which unrolls via apply_chain).
            TapeStageOutputT::Many { tape, parent_idx } =>
                TapeStageOutputT::Many { tape, parent_idx },
        }
    }
}

// ── Stages ──────────────────────────────────────────────────────────

pub struct FilterT<F: Fn(&TapeRow<'_>) -> bool> {
    pub pred: F,
}

impl<F: Fn(&TapeRow<'_>) -> bool> TapeStageT for FilterT<F> {
    #[inline]
    fn apply<'a>(&self, x: TapeRow<'a>) -> TapeStageOutputT<'a> {
        if (self.pred)(&x) { TapeStageOutputT::Pass(x) } else { TapeStageOutputT::Filtered }
    }
}

/// Single field read; new cursor or Filtered when the field is missing
/// or the input isn't an Object.
pub struct MapFieldT {
    pub field: std::sync::Arc<str>,
}

impl TapeStageT for MapFieldT {
    #[inline]
    fn apply<'a>(&self, x: TapeRow<'a>) -> TapeStageOutputT<'a> {
        match x.get_field(&self.field) {
            Some(c) => TapeStageOutputT::Pass(c),
            None    => TapeStageOutputT::Filtered,
        }
    }
}

pub struct MapFieldChainT {
    pub chain: Vec<std::sync::Arc<str>>,
}

impl TapeStageT for MapFieldChainT {
    fn apply<'a>(&self, x: TapeRow<'a>) -> TapeStageOutputT<'a> {
        let chain_refs: Vec<&str> = self.chain.iter().map(|a| a.as_ref()).collect();
        match x.walk_path(&chain_refs) {
            Some(c) => TapeStageOutputT::Pass(c),
            None    => TapeStageOutputT::Filtered,
        }
    }
}

/// FlatMap stage — input must be Object containing the inner array
/// at `field`.  Emits Many pointing at the inner Array node so the
/// outer loop iterates its children.  Filters/Skip/Take/Map stages
/// after FlatMap operate on each inner element.
pub struct FlatMapFieldT {
    pub field: std::sync::Arc<str>,
}

impl TapeStageT for FlatMapFieldT {
    #[inline]
    fn apply<'a>(&self, x: TapeRow<'a>) -> TapeStageOutputT<'a> {
        match x.get_field(&self.field) {
            Some(c) if c.is_array() =>
                TapeStageOutputT::Many { tape: c.tape, parent_idx: c.idx },
            _ => TapeStageOutputT::Filtered,
        }
    }
}

pub struct TakeT {
    pub n: usize,
    seen: std::cell::Cell<usize>,
}
impl TakeT {
    pub fn new(n: usize) -> Self { Self { n, seen: std::cell::Cell::new(0) } }
    pub fn reset(&self) { self.seen.set(0); }
}
impl TapeStageT for TakeT {
    #[inline]
    fn apply<'a>(&self, x: TapeRow<'a>) -> TapeStageOutputT<'a> {
        let s = self.seen.get();
        if s >= self.n { return TapeStageOutputT::Done; }
        self.seen.set(s + 1);
        TapeStageOutputT::Pass(x)
    }
}

pub struct SkipT {
    pub n: usize,
    seen: std::cell::Cell<usize>,
}
impl SkipT {
    pub fn new(n: usize) -> Self { Self { n, seen: std::cell::Cell::new(0) } }
    pub fn reset(&self) { self.seen.set(0); }
}
impl TapeStageT for SkipT {
    #[inline]
    fn apply<'a>(&self, x: TapeRow<'a>) -> TapeStageOutputT<'a> {
        let s = self.seen.get();
        if s < self.n { self.seen.set(s + 1); TapeStageOutputT::Filtered }
        else { TapeStageOutputT::Pass(x) }
    }
}

// ── Sinks (numeric — no materialisation needed) ────────────────────

pub trait TapeSinkT {
    type Acc<'a>;
    fn init<'a>() -> Self::Acc<'a>;
    fn fold<'a>(arena: &'a Arena, acc: Self::Acc<'a>, v: TapeRow<'a>) -> Self::Acc<'a>;
    fn finalise<'a>(arena: &'a Arena, acc: Self::Acc<'a>) -> BVal<'a>;
}

pub struct CountSinkT;
impl TapeSinkT for CountSinkT {
    type Acc<'a> = i64;
    #[inline] fn init<'a>() -> Self::Acc<'a> { 0 }
    #[inline] fn fold<'a>(_: &'a Arena, acc: Self::Acc<'a>, _: TapeRow<'a>) -> Self::Acc<'a> { acc + 1 }
    #[inline] fn finalise<'a>(_: &'a Arena, acc: Self::Acc<'a>) -> BVal<'a> { BVal::Int(acc) }
}

pub struct SumSinkT;
impl TapeSinkT for SumSinkT {
    type Acc<'a> = (i64, f64, bool);
    #[inline] fn init<'a>() -> Self::Acc<'a> { (0, 0.0, false) }
    fn fold<'a>(_: &'a Arena, mut acc: Self::Acc<'a>, v: TapeRow<'a>) -> Self::Acc<'a> {
        if let Some(n) = v.as_int() { acc.0 = acc.0.wrapping_add(n); return acc; }
        if let Some(f) = v.as_float() { acc.1 += f; acc.2 = true; return acc; }
        if let Some(b) = v.as_bool() { acc.0 = acc.0.wrapping_add(b as i64); return acc; }
        acc
    }
    fn finalise<'a>(_: &'a Arena, acc: Self::Acc<'a>) -> BVal<'a> {
        if acc.2 { BVal::Float(acc.0 as f64 + acc.1) } else { BVal::Int(acc.0) }
    }
}

pub struct MinSinkT;
impl TapeSinkT for MinSinkT {
    type Acc<'a> = Option<f64>;
    #[inline] fn init<'a>() -> Self::Acc<'a> { None }
    fn fold<'a>(_: &'a Arena, acc: Self::Acc<'a>, v: TapeRow<'a>) -> Self::Acc<'a> {
        let n = match v.as_float() { Some(f) => f, None => return acc };
        Some(match acc { Some(c) => c.min(n), None => n })
    }
    fn finalise<'a>(_: &'a Arena, acc: Self::Acc<'a>) -> BVal<'a> {
        match acc {
            Some(f) if f.fract() == 0.0 && f.abs() < (i64::MAX as f64) => BVal::Int(f as i64),
            Some(f) => BVal::Float(f),
            None => BVal::Null,
        }
    }
}

pub struct MaxSinkT;
impl TapeSinkT for MaxSinkT {
    type Acc<'a> = Option<f64>;
    #[inline] fn init<'a>() -> Self::Acc<'a> { None }
    fn fold<'a>(_: &'a Arena, acc: Self::Acc<'a>, v: TapeRow<'a>) -> Self::Acc<'a> {
        let n = match v.as_float() { Some(f) => f, None => return acc };
        Some(match acc { Some(c) => c.max(n), None => n })
    }
    fn finalise<'a>(_: &'a Arena, acc: Self::Acc<'a>) -> BVal<'a> {
        match acc {
            Some(f) if f.fract() == 0.0 && f.abs() < (i64::MAX as f64) => BVal::Int(f as i64),
            Some(f) => BVal::Float(f),
            None => BVal::Null,
        }
    }
}

pub struct AvgSinkT;
impl TapeSinkT for AvgSinkT {
    type Acc<'a> = (f64, usize);
    #[inline] fn init<'a>() -> Self::Acc<'a> { (0.0, 0) }
    fn fold<'a>(_: &'a Arena, mut acc: Self::Acc<'a>, v: TapeRow<'a>) -> Self::Acc<'a> {
        if let Some(n) = v.as_float() { acc.0 += n; acc.1 += 1; }
        acc
    }
    fn finalise<'a>(_: &'a Arena, acc: Self::Acc<'a>) -> BVal<'a> {
        if acc.1 == 0 { BVal::Null } else { BVal::Float(acc.0 / acc.1 as f64) }
    }
}

// ── Phase 2: materialising sinks (First/Last/Collect) ──────────────

pub struct FirstSinkT;
impl TapeSinkT for FirstSinkT {
    type Acc<'a> = Option<BVal<'a>>;
    #[inline] fn init<'a>() -> Self::Acc<'a> { None }
    #[inline] fn fold<'a>(arena: &'a Arena, acc: Self::Acc<'a>, v: TapeRow<'a>) -> Self::Acc<'a> {
        if acc.is_some() { acc } else { Some(v.materialise_into(arena)) }
    }
    #[inline] fn finalise<'a>(_: &'a Arena, acc: Self::Acc<'a>) -> BVal<'a> {
        acc.unwrap_or(BVal::Null)
    }
}

pub struct LastSinkT;
impl TapeSinkT for LastSinkT {
    type Acc<'a> = Option<BVal<'a>>;
    #[inline] fn init<'a>() -> Self::Acc<'a> { None }
    #[inline] fn fold<'a>(arena: &'a Arena, _: Self::Acc<'a>, v: TapeRow<'a>) -> Self::Acc<'a> {
        Some(v.materialise_into(arena))
    }
    #[inline] fn finalise<'a>(_: &'a Arena, acc: Self::Acc<'a>) -> BVal<'a> {
        acc.unwrap_or(BVal::Null)
    }
}

pub struct CollectSinkT;
impl TapeSinkT for CollectSinkT {
    type Acc<'a> = Vec<BVal<'a>>;
    #[inline] fn init<'a>() -> Self::Acc<'a> { Vec::new() }
    #[inline] fn fold<'a>(arena: &'a Arena, mut acc: Self::Acc<'a>, v: TapeRow<'a>) -> Self::Acc<'a> {
        acc.push(v.materialise_into(arena)); acc
    }
    #[inline] fn finalise<'a>(arena: &'a Arena, acc: Self::Acc<'a>) -> BVal<'a> {
        let slice = arena.alloc_slice_fill_iter(acc.into_iter());
        BVal::Arr(&*slice)
    }
}

// ── Outer loop ──────────────────────────────────────────────────────

/// Run a composed Stage chain over the elements of an Array node.
/// `arr_idx` must point at an `Array` node in `tape`.
/// Apply a stage chain over a single row, dispatching downstream
/// FlatMap expansion via `inner_stages` (the chain to run per Many
/// child).  Returns the per-row outcome the outer sink loop needs.
///
/// Generic across all Pipeline shapes.  When `inner_stages` is None,
/// no FlatMap is expected — Many is treated as Filtered (caller bug).
fn apply_or_expand<'a, S: TapeSinkT>(
    arena: &'a Arena,
    acc: S::Acc<'a>,
    cur: TapeRow<'a>,
    stages: &dyn TapeStageT,
    inner_stages: Option<&dyn TapeStageT>,
) -> (S::Acc<'a>, bool) {
    // Returns (new_acc, done).  done=true => outer loop terminates.
    match stages.apply(cur) {
        TapeStageOutputT::Pass(p) => (S::fold(arena, acc, p), false),
        TapeStageOutputT::Filtered => (acc, false),
        TapeStageOutputT::Done => (acc, true),
        TapeStageOutputT::Many { tape, parent_idx } => {
            let inner = match inner_stages {
                Some(s) => s,
                None => return (acc, false),
            };
            let i = parent_idx as usize;
            let len = match tape.nodes[i] {
                TapeNode::Array { len, .. } => len as usize,
                _ => return (acc, false),
            };
            let mut acc = acc;
            let mut j = i + 1;
            for _ in 0..len {
                let child = TapeRow::new(tape, j as u32);
                match inner.apply(child) {
                    TapeStageOutputT::Pass(p) => acc = S::fold(arena, acc, p),
                    TapeStageOutputT::Filtered => {}
                    TapeStageOutputT::Done => return (acc, true),
                    TapeStageOutputT::Many { .. } => {
                        // Nested FlatMap not supported in this round
                        // (would need recursion).  Treat as Filtered.
                    }
                }
                j += tape.span(j);
            }
            (acc, false)
        }
    }
}

pub fn run_pipeline_t<'a, S: TapeSinkT>(
    arena: &'a Arena,
    tape: &'a TapeData,
    arr_idx: u32,
    stages: &dyn TapeStageT,
) -> BVal<'a> {
    run_pipeline_t_with_inner::<S>(arena, tape, arr_idx, stages, None)
}

/// FlatMap-aware variant.  `inner_stages` is the stage chain that
/// runs per child of any Many emitted by `stages`.  Typically the
/// caller passes the post-FlatMap suffix of the original chain.
pub fn run_pipeline_t_with_inner<'a, S: TapeSinkT>(
    arena: &'a Arena,
    tape: &'a TapeData,
    arr_idx: u32,
    stages: &dyn TapeStageT,
    inner_stages: Option<&dyn TapeStageT>,
) -> BVal<'a> {
    let mut acc: S::Acc<'a> = S::init();
    let i = arr_idx as usize;
    let len = match tape.nodes[i] {
        TapeNode::Array { len, .. } => len as usize,
        _ => return S::finalise(arena, acc),
    };
    let mut j = i + 1;
    for _ in 0..len {
        let cur = TapeRow::new(tape, j as u32);
        let (new_acc, done) = apply_or_expand::<S>(arena, acc, cur, stages, inner_stages);
        acc = new_acc;
        if done { break; }
        j += tape.span(j);
    }
    S::finalise(arena, acc)
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn parse_doc(s: &[u8]) -> Arc<TapeData> {
        TapeData::parse(s.to_vec()).unwrap()
    }

    #[test]
    fn count_via_tape_pipeline() {
        let doc = br#"{"a":[1,2,3,4,5]}"#;
        let tape = parse_doc(doc);
        let arena = Arena::new();
        let arr_idx = crate::strref::tape_walk_field_chain(&tape, &["a"]).unwrap();
        let out = run_pipeline_t::<CountSinkT>(&arena, &tape, arr_idx as u32, &IdentityT);
        match out { BVal::Int(5) => {} _ => panic!("got {:?}", out), }
    }

    #[test]
    fn sum_via_tape_pipeline() {
        let doc = br#"{"a":[1,2,3,4,5]}"#;
        let tape = parse_doc(doc);
        let arena = Arena::new();
        let arr_idx = crate::strref::tape_walk_field_chain(&tape, &["a"]).unwrap();
        let out = run_pipeline_t::<SumSinkT>(&arena, &tape, arr_idx as u32, &IdentityT);
        match out { BVal::Int(15) => {} _ => panic!("got {:?}", out), }
    }

    #[test]
    fn filter_then_count_via_tape() {
        let doc = br#"{"a":[1,2,3,4,5,6,7,8,9,10]}"#;
        let tape = parse_doc(doc);
        let arena = Arena::new();
        let arr_idx = crate::strref::tape_walk_field_chain(&tape, &["a"]).unwrap();
        let stages = FilterT { pred: |v: &TapeRow<'_>| v.as_int().map_or(false, |n| n % 2 == 0) };
        let out = run_pipeline_t::<CountSinkT>(&arena, &tape, arr_idx as u32, &stages);
        match out { BVal::Int(5) => {} _ => panic!("got {:?}", out), }
    }

    #[test]
    fn map_field_then_sum() {
        let doc = br#"{"orders":[{"total":10},{"total":20},{"total":30}]}"#;
        let tape = parse_doc(doc);
        let arena = Arena::new();
        let arr_idx = crate::strref::tape_walk_field_chain(&tape, &["orders"]).unwrap();
        let stages = MapFieldT { field: Arc::from("total") };
        let out = run_pipeline_t::<SumSinkT>(&arena, &tape, arr_idx as u32, &stages);
        match out { BVal::Int(60) => {} _ => panic!("got {:?}", out), }
    }

    #[test]
    fn map_field_chain_walks_nested() {
        let doc = br#"{"orders":[{"addr":{"n":1}},{"addr":{"n":2}},{"addr":{"n":3}}]}"#;
        let tape = parse_doc(doc);
        let arena = Arena::new();
        let arr_idx = crate::strref::tape_walk_field_chain(&tape, &["orders"]).unwrap();
        let stages = MapFieldChainT {
            chain: vec![Arc::from("addr"), Arc::from("n")],
        };
        let out = run_pipeline_t::<SumSinkT>(&arena, &tape, arr_idx as u32, &stages);
        match out { BVal::Int(6) => {} _ => panic!("got {:?}", out), }
    }

    #[test]
    fn take_caps_output() {
        let doc = br#"{"a":[1,2,3,4,5,6,7,8,9,10]}"#;
        let tape = parse_doc(doc);
        let arena = Arena::new();
        let arr_idx = crate::strref::tape_walk_field_chain(&tape, &["a"]).unwrap();
        let take = TakeT::new(3);
        let out = run_pipeline_t::<CountSinkT>(&arena, &tape, arr_idx as u32, &take);
        match out { BVal::Int(3) => {} _ => panic!("got {:?}", out), }
    }

    #[test]
    fn min_max_avg_sinks() {
        let doc = br#"{"a":[7,2,9,4,5]}"#;
        let tape = parse_doc(doc);
        let arena = Arena::new();
        let arr_idx = crate::strref::tape_walk_field_chain(&tape, &["a"]).unwrap();
        let mn = run_pipeline_t::<MinSinkT>(&arena, &tape, arr_idx as u32, &IdentityT);
        let mx = run_pipeline_t::<MaxSinkT>(&arena, &tape, arr_idx as u32, &IdentityT);
        let av = run_pipeline_t::<AvgSinkT>(&arena, &tape, arr_idx as u32, &IdentityT);
        assert!(matches!(mn, BVal::Int(2)));
        assert!(matches!(mx, BVal::Int(9)));
        match av { BVal::Float(f) => assert!((f - 5.4).abs() < 1e-9), _ => panic!("avg") }
    }

    #[test]
    fn first_sink_real() {
        let doc = br#"{"a":[10,20,30,40]}"#;
        let tape = parse_doc(doc);
        let arena = Arena::new();
        let arr_idx = crate::strref::tape_walk_field_chain(&tape, &["a"]).unwrap();
        let out = run_pipeline_t::<FirstSinkT>(&arena, &tape, arr_idx as u32, &IdentityT);
        assert!(matches!(out, BVal::Int(10)), "got {:?}", out);
    }

    #[test]
    fn last_sink_real() {
        let doc = br#"{"a":[10,20,30,40]}"#;
        let tape = parse_doc(doc);
        let arena = Arena::new();
        let arr_idx = crate::strref::tape_walk_field_chain(&tape, &["a"]).unwrap();
        let out = run_pipeline_t::<LastSinkT>(&arena, &tape, arr_idx as u32, &IdentityT);
        assert!(matches!(out, BVal::Int(40)), "got {:?}", out);
    }

    #[test]
    fn collect_sink_real() {
        let doc = br#"{"a":[1,2,3,4,5]}"#;
        let tape = parse_doc(doc);
        let arena = Arena::new();
        let arr_idx = crate::strref::tape_walk_field_chain(&tape, &["a"]).unwrap();
        let out = run_pipeline_t::<CollectSinkT>(&arena, &tape, arr_idx as u32, &IdentityT);
        match out {
            BVal::Arr(arr) => {
                assert_eq!(arr.len(), 5);
                for (i, v) in arr.iter().enumerate() {
                    assert!(matches!(v, BVal::Int(n) if *n == (i + 1) as i64));
                }
            }
            _ => panic!("got {:?}", out),
        }
    }

    #[test]
    fn collect_with_filter_and_map_field() {
        let doc = br#"{"orders":[{"id":1,"v":10},{"id":2,"v":20},{"id":3,"v":30}]}"#;
        let tape = parse_doc(doc);
        let arena = Arena::new();
        let arr_idx = crate::strref::tape_walk_field_chain(&tape, &["orders"]).unwrap();
        let stages = ComposedT {
            a: FilterT { pred: |v: &TapeRow<'_>| {
                v.get_field("v").and_then(|c| c.as_int()).map_or(false, |n| n >= 20)
            }},
            b: MapFieldT { field: Arc::from("id") },
        };
        let out = run_pipeline_t::<CollectSinkT>(&arena, &tape, arr_idx as u32, &stages);
        match out {
            BVal::Arr(arr) => {
                assert_eq!(arr.len(), 2);
                assert!(matches!(arr[0], BVal::Int(2)));
                assert!(matches!(arr[1], BVal::Int(3)));
            }
            _ => panic!("got {:?}", out),
        }
    }

    #[test]
    fn first_with_filter_short_circuits() {
        let doc = br#"{"orders":[{"v":1},{"v":2},{"v":3},{"v":4}]}"#;
        let tape = parse_doc(doc);
        let arena = Arena::new();
        let arr_idx = crate::strref::tape_walk_field_chain(&tape, &["orders"]).unwrap();
        // Filter v >= 3 then First — result should be {v: 3}
        let stages = FilterT {
            pred: |x: &TapeRow<'_>| {
                x.get_field("v").and_then(|c| c.as_int()).map_or(false, |n| n >= 3)
            }
        };
        let out = run_pipeline_t::<FirstSinkT>(&arena, &tape, arr_idx as u32, &stages);
        match out {
            BVal::Obj(entries) => {
                assert_eq!(entries.len(), 1);
                assert_eq!(entries[0].0, "v");
                assert!(matches!(entries[0].1, BVal::Int(3)));
            }
            _ => panic!("got {:?}", out),
        }
    }

    #[test]
    fn materialise_into_obj_round_trip() {
        let doc = br#"{"x":{"k":"hello","n":42,"f":1.5,"b":true,"z":null}}"#;
        let tape = parse_doc(doc);
        let arena = Arena::new();
        let idx = crate::strref::tape_walk_field_chain(&tape, &["x"]).unwrap();
        let row = TapeRow::new(&tape, idx as u32);
        let bv = row.materialise_into(&arena);
        match bv {
            BVal::Obj(entries) => {
                assert_eq!(entries.len(), 5);
                let pairs: std::collections::HashMap<&str, _> =
                    entries.iter().map(|(k, v)| (*k, *v)).collect();
                assert!(matches!(pairs["k"], BVal::Str("hello")));
                assert!(matches!(pairs["n"], BVal::Int(42)));
                assert!(matches!(pairs["f"], BVal::Float(_)));
                assert!(matches!(pairs["b"], BVal::Bool(true)));
                assert!(matches!(pairs["z"], BVal::Null));
            }
            _ => panic!("expected Obj, got {:?}", bv),
        }
    }
}
