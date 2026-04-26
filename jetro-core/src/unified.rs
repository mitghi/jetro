//! Unified pipeline substrate — Phase 2 of pipeline_unification.
//!
//! Single Stage / Sink trait + run loop generic over `R: Row<'a>`.
//! Replaces composed_borrow's StageB+SinkB + composed_tape's
//! TapeStageT+TapeSinkT with one parameterised module.
//!
//! This commit ships substrate (NOT yet wired into Jetro::collect_val_borrow).
//! Phase 3 will migrate pipeline_borrow + pipeline_tape_borrow callers
//! onto the unified runner and delete the parallel modules.
//!
//! ## Design
//!
//! `Stage<R>` is generic over the row type `R: Row<'a>`.  Concrete
//! stages (Filter, MapField, FlatMapField, Take, Skip) work for ANY
//! substrate that implements `Row<'a>` — same struct, same impl,
//! monomorphised per substrate at the call site.
//!
//! `Sink` uses a GAT `Acc<'a>` so accumulators can hold borrowed
//! payloads (BVal<'a>) for First/Last/Collect.  `fold` is generic
//! over `R: Row<'a>` so the same Sink runs over any substrate.
//!
//! `Composed<A, B>` chains via static dispatch.  For dynamic chain
//! building (lowering), `Box<dyn Stage<R>>` is supported through the
//! blanket impl.
//!
//! ## Phase 3 migration plan
//!
//! 1. Migrate `pipeline_tape_borrow::lower_stages` to emit
//!    `Box<dyn Stage<TapeRow<'a>>>` from this module.
//! 2. Migrate `pipeline_borrow::lower_stages` likewise for BVal.
//! 3. Run regression bench; bench-gate parity vs current per-substrate.
//! 4. Delete composed_borrow.rs / composed_tape.rs / pipeline_borrow.rs
//!    / pipeline_tape_borrow.rs once all wired.

use crate::eval::borrowed::{Arena, Val as BVal};
use crate::row::Row;
use smallvec::SmallVec;

// ── Stage output ────────────────────────────────────────────────────

pub enum StageOutputU<R> {
    Pass(R),
    Filtered,
    Done,
    /// Many — FlatMap expansion.  Stored as `SmallVec` to keep small
    /// inner-array cases inline (typical: 0-4 children).  Larger
    /// flat-maps heap-allocate; the row type is `Copy` and small
    /// (16-24 bytes), so 5000 children = 80-120 KB heap, acceptable.
    Many(SmallVec<[R; 4]>),
}

// ── Stage trait ─────────────────────────────────────────────────────

pub trait Stage<R> {
    fn apply(&self, x: R) -> StageOutputU<R>;
}

impl<R, T: Stage<R> + ?Sized> Stage<R> for Box<T> {
    #[inline]
    fn apply(&self, x: R) -> StageOutputU<R> { (**self).apply(x) }
}

pub struct Identity<R>(std::marker::PhantomData<fn(R)>);
impl<R> Identity<R> {
    pub fn new() -> Self { Self(std::marker::PhantomData) }
}
impl<R> Default for Identity<R> {
    fn default() -> Self { Self::new() }
}
impl<R> Stage<R> for Identity<R> {
    #[inline]
    fn apply(&self, x: R) -> StageOutputU<R> { StageOutputU::Pass(x) }
}

pub struct Composed<R, A: Stage<R>, B: Stage<R>> {
    pub a: A,
    pub b: B,
    _marker: std::marker::PhantomData<fn(R)>,
}
impl<R, A: Stage<R>, B: Stage<R>> Composed<R, A, B> {
    pub fn new(a: A, b: B) -> Self { Self { a, b, _marker: std::marker::PhantomData } }
}
impl<R, A: Stage<R>, B: Stage<R>> Stage<R> for Composed<R, A, B> {
    fn apply(&self, x: R) -> StageOutputU<R> {
        match self.a.apply(x) {
            StageOutputU::Pass(v) => self.b.apply(v),
            StageOutputU::Filtered => StageOutputU::Filtered,
            StageOutputU::Done => StageOutputU::Done,
            StageOutputU::Many(items) => {
                // Apply b to each, collect.  Done short-circuits via
                // early return.
                let mut out: SmallVec<[R; 4]> = SmallVec::new();
                for v in items {
                    match self.b.apply(v) {
                        StageOutputU::Pass(p) => out.push(p),
                        StageOutputU::Filtered => continue,
                        StageOutputU::Many(more) => out.extend(more),
                        StageOutputU::Done => {
                            return if out.is_empty() {
                                StageOutputU::Done
                            } else {
                                StageOutputU::Many(out)
                            };
                        }
                    }
                }
                if out.is_empty() { StageOutputU::Filtered }
                else if out.len() == 1 { StageOutputU::Pass(out.into_iter().next().unwrap()) }
                else { StageOutputU::Many(out) }
            }
        }
    }
}

// ── Stages ──────────────────────────────────────────────────────────

pub struct Filter<R, F: Fn(&R) -> bool> {
    pub pred: F,
    _marker: std::marker::PhantomData<fn(R)>,
}
impl<R, F: Fn(&R) -> bool> Filter<R, F> {
    pub fn new(pred: F) -> Self { Self { pred, _marker: std::marker::PhantomData } }
}
impl<R, F: Fn(&R) -> bool> Stage<R> for Filter<R, F> {
    #[inline]
    fn apply(&self, x: R) -> StageOutputU<R> {
        if (self.pred)(&x) { StageOutputU::Pass(x) } else { StageOutputU::Filtered }
    }
}

pub struct MapField<R> {
    pub field: std::sync::Arc<str>,
    _marker: std::marker::PhantomData<fn(R)>,
}
impl<R> MapField<R> {
    pub fn new(field: std::sync::Arc<str>) -> Self {
        Self { field, _marker: std::marker::PhantomData }
    }
}
impl<'a, R: Row<'a>> Stage<R> for MapField<R> {
    #[inline]
    fn apply(&self, x: R) -> StageOutputU<R> {
        match x.get_field(&self.field) {
            Some(v) => StageOutputU::Pass(v),
            None    => StageOutputU::Filtered,
        }
    }
}

pub struct MapFieldChain<R> {
    pub chain: Vec<std::sync::Arc<str>>,
    _marker: std::marker::PhantomData<fn(R)>,
}
impl<R> MapFieldChain<R> {
    pub fn new(chain: Vec<std::sync::Arc<str>>) -> Self {
        Self { chain, _marker: std::marker::PhantomData }
    }
}
impl<'a, R: Row<'a>> Stage<R> for MapFieldChain<R> {
    fn apply(&self, x: R) -> StageOutputU<R> {
        let chain_refs: Vec<&str> = self.chain.iter().map(|a| a.as_ref()).collect();
        match x.walk_path(&chain_refs) {
            Some(v) => StageOutputU::Pass(v),
            None    => StageOutputU::Filtered,
        }
    }
}

pub struct FlatMapField<R> {
    pub field: std::sync::Arc<str>,
    _marker: std::marker::PhantomData<fn(R)>,
}
impl<R> FlatMapField<R> {
    pub fn new(field: std::sync::Arc<str>) -> Self {
        Self { field, _marker: std::marker::PhantomData }
    }
}
impl<'a, R: Row<'a>> Stage<R> for FlatMapField<R> {
    fn apply(&self, x: R) -> StageOutputU<R> {
        let arr_row = match x.get_field(&self.field) {
            Some(c) => c,
            None    => return StageOutputU::Filtered,
        };
        let iter = match arr_row.array_children() {
            Some(i) => i,
            None    => return StageOutputU::Filtered,
        };
        let items: SmallVec<[R; 4]> = iter.collect();
        if items.is_empty() { StageOutputU::Filtered }
        else if items.len() == 1 { StageOutputU::Pass(items.into_iter().next().unwrap()) }
        else { StageOutputU::Many(items) }
    }
}

pub struct Take<R> {
    pub n: usize,
    seen: std::cell::Cell<usize>,
    _marker: std::marker::PhantomData<fn(R)>,
}
impl<R> Take<R> {
    pub fn new(n: usize) -> Self {
        Self { n, seen: std::cell::Cell::new(0), _marker: std::marker::PhantomData }
    }
}
impl<R> Stage<R> for Take<R> {
    #[inline]
    fn apply(&self, x: R) -> StageOutputU<R> {
        let s = self.seen.get();
        if s >= self.n { return StageOutputU::Done; }
        self.seen.set(s + 1);
        StageOutputU::Pass(x)
    }
}

pub struct Skip<R> {
    pub n: usize,
    seen: std::cell::Cell<usize>,
    _marker: std::marker::PhantomData<fn(R)>,
}
impl<R> Skip<R> {
    pub fn new(n: usize) -> Self {
        Self { n, seen: std::cell::Cell::new(0), _marker: std::marker::PhantomData }
    }
}
impl<R> Stage<R> for Skip<R> {
    #[inline]
    fn apply(&self, x: R) -> StageOutputU<R> {
        let s = self.seen.get();
        if s < self.n { self.seen.set(s + 1); StageOutputU::Filtered }
        else { StageOutputU::Pass(x) }
    }
}

// ── Sink trait ──────────────────────────────────────────────────────

pub trait Sink {
    type Acc<'a>;
    fn init<'a>() -> Self::Acc<'a>;
    fn fold<'a, R: Row<'a>>(arena: &'a Arena, acc: Self::Acc<'a>, v: R) -> Self::Acc<'a>;
    fn finalise<'a>(arena: &'a Arena, acc: Self::Acc<'a>) -> BVal<'a>;
}

// ── Sinks ──────────────────────────────────────────────────────────

pub struct CountSink;
impl Sink for CountSink {
    type Acc<'a> = i64;
    #[inline] fn init<'a>() -> Self::Acc<'a> { 0 }
    #[inline] fn fold<'a, R: Row<'a>>(_: &'a Arena, acc: Self::Acc<'a>, _: R) -> Self::Acc<'a> { acc + 1 }
    #[inline] fn finalise<'a>(_: &'a Arena, acc: Self::Acc<'a>) -> BVal<'a> { BVal::Int(acc) }
}

pub struct SumSink;
impl Sink for SumSink {
    type Acc<'a> = (i64, f64, bool);
    #[inline] fn init<'a>() -> Self::Acc<'a> { (0, 0.0, false) }
    fn fold<'a, R: Row<'a>>(_: &'a Arena, mut acc: Self::Acc<'a>, v: R) -> Self::Acc<'a> {
        if let Some(n) = v.as_int()   { acc.0 = acc.0.wrapping_add(n); return acc; }
        if let Some(f) = v.as_float() { acc.1 += f; acc.2 = true; return acc; }
        if let Some(b) = v.as_bool()  { acc.0 = acc.0.wrapping_add(b as i64); return acc; }
        acc
    }
    fn finalise<'a>(_: &'a Arena, acc: Self::Acc<'a>) -> BVal<'a> {
        if acc.2 { BVal::Float(acc.0 as f64 + acc.1) } else { BVal::Int(acc.0) }
    }
}

pub struct MinSink;
impl Sink for MinSink {
    type Acc<'a> = Option<f64>;
    #[inline] fn init<'a>() -> Self::Acc<'a> { None }
    fn fold<'a, R: Row<'a>>(_: &'a Arena, acc: Self::Acc<'a>, v: R) -> Self::Acc<'a> {
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

pub struct MaxSink;
impl Sink for MaxSink {
    type Acc<'a> = Option<f64>;
    #[inline] fn init<'a>() -> Self::Acc<'a> { None }
    fn fold<'a, R: Row<'a>>(_: &'a Arena, acc: Self::Acc<'a>, v: R) -> Self::Acc<'a> {
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

pub struct AvgSink;
impl Sink for AvgSink {
    type Acc<'a> = (f64, usize);
    #[inline] fn init<'a>() -> Self::Acc<'a> { (0.0, 0) }
    fn fold<'a, R: Row<'a>>(_: &'a Arena, mut acc: Self::Acc<'a>, v: R) -> Self::Acc<'a> {
        if let Some(n) = v.as_float() { acc.0 += n; acc.1 += 1; }
        acc
    }
    fn finalise<'a>(_: &'a Arena, acc: Self::Acc<'a>) -> BVal<'a> {
        if acc.1 == 0 { BVal::Null } else { BVal::Float(acc.0 / acc.1 as f64) }
    }
}

pub struct FirstSink;
impl Sink for FirstSink {
    type Acc<'a> = Option<BVal<'a>>;
    #[inline] fn init<'a>() -> Self::Acc<'a> { None }
    #[inline] fn fold<'a, R: Row<'a>>(arena: &'a Arena, acc: Self::Acc<'a>, v: R) -> Self::Acc<'a> {
        if acc.is_some() { acc } else { Some(v.materialise(arena)) }
    }
    #[inline] fn finalise<'a>(_: &'a Arena, acc: Self::Acc<'a>) -> BVal<'a> {
        acc.unwrap_or(BVal::Null)
    }
}

pub struct LastSink;
impl Sink for LastSink {
    type Acc<'a> = Option<BVal<'a>>;
    #[inline] fn init<'a>() -> Self::Acc<'a> { None }
    #[inline] fn fold<'a, R: Row<'a>>(arena: &'a Arena, _: Self::Acc<'a>, v: R) -> Self::Acc<'a> {
        Some(v.materialise(arena))
    }
    #[inline] fn finalise<'a>(_: &'a Arena, acc: Self::Acc<'a>) -> BVal<'a> {
        acc.unwrap_or(BVal::Null)
    }
}

pub struct CollectSink;
impl Sink for CollectSink {
    type Acc<'a> = Vec<BVal<'a>>;
    #[inline] fn init<'a>() -> Self::Acc<'a> { Vec::new() }
    #[inline] fn fold<'a, R: Row<'a>>(arena: &'a Arena, mut acc: Self::Acc<'a>, v: R) -> Self::Acc<'a> {
        acc.push(v.materialise(arena)); acc
    }
    #[inline] fn finalise<'a>(arena: &'a Arena, acc: Self::Acc<'a>) -> BVal<'a> {
        let slice = arena.alloc_slice_fill_iter(acc.into_iter());
        BVal::Arr(&*slice)
    }
}

// ── Outer loop ──────────────────────────────────────────────────────

pub fn run_pipeline<'a, R, S>(
    arena: &'a Arena,
    rows: impl Iterator<Item = R>,
    stages: &dyn Stage<R>,
) -> BVal<'a>
where
    R: Row<'a>,
    S: Sink,
{
    let mut acc: S::Acc<'a> = S::init();
    for r in rows {
        match stages.apply(r) {
            StageOutputU::Pass(p) => acc = S::fold::<R>(arena, acc, p),
            StageOutputU::Filtered => {}
            StageOutputU::Done => break,
            StageOutputU::Many(items) => {
                for p in items { acc = S::fold::<R>(arena, acc, p); }
            }
        }
    }
    S::finalise(arena, acc)
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn make_orders<'a>(arena: &'a Arena) -> Vec<BVal<'a>> {
        let mut out = Vec::new();
        for i in 1..=5i64 {
            let pairs = vec![
                (arena.alloc_str("id"), BVal::Int(i)),
                (arena.alloc_str("v"), BVal::Int(i * 10)),
            ];
            let slice = arena.alloc_slice_fill_iter(pairs.into_iter());
            out.push(BVal::Obj(&*slice));
        }
        out
    }

    #[test]
    fn count_over_bval() {
        let arena = Arena::new();
        let orders = make_orders(&arena);
        let id: Identity<BVal> = Identity::new();
        let out = run_pipeline::<BVal, CountSink>(&arena, orders.iter().copied(), &id);
        assert!(matches!(out, BVal::Int(5)));
    }

    #[test]
    fn filter_then_sum_over_bval() {
        let arena = Arena::new();
        let orders = make_orders(&arena);
        let stages = Composed::new(
            Filter::<BVal, _>::new(|r: &BVal| {
                r.get_field("v").and_then(|v| v.as_int()).map_or(false, |n| n >= 30)
            }),
            MapField::<BVal>::new(Arc::from("v")),
        );
        let out = run_pipeline::<BVal, SumSink>(&arena, orders.iter().copied(), &stages);
        // v >= 30 -> 30 + 40 + 50 = 120
        assert!(matches!(out, BVal::Int(120)), "got {:?}", out);
    }

    #[test]
    fn skip_take_chain_over_bval() {
        let arena = Arena::new();
        let orders = make_orders(&arena);
        let stages = Composed::new(
            Skip::<BVal>::new(1),
            Take::<BVal>::new(2),
        );
        let out = run_pipeline::<BVal, CountSink>(&arena, orders.iter().copied(), &stages);
        assert!(matches!(out, BVal::Int(2)));
    }

    #[test]
    fn collect_over_bval_materialises() {
        let arena = Arena::new();
        let orders = make_orders(&arena);
        let stages = MapField::<BVal>::new(Arc::from("id"));
        let out = run_pipeline::<BVal, CollectSink>(&arena, orders.iter().copied(), &stages);
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

    #[cfg(feature = "simd-json")]
    #[test]
    fn count_over_tape_row_unified() {
        use crate::composed_tape::TapeRow;
        use crate::strref::TapeData;
        let bytes = br#"{"a":[1,2,3,4,5,6,7]}"#.to_vec();
        let tape = TapeData::parse(bytes).unwrap();
        let arr_idx = crate::strref::tape_walk_field_chain(&tape, &["a"]).unwrap();
        let arr_row = TapeRow::new(&tape, arr_idx as u32);
        let arena = Arena::new();
        let id: Identity<TapeRow<'_>> = Identity::new();
        let iter = arr_row.array_children().unwrap();
        let out = run_pipeline::<TapeRow<'_>, CountSink>(&arena, iter, &id);
        assert!(matches!(out, BVal::Int(7)));
    }

    #[cfg(feature = "simd-json")]
    #[test]
    fn filter_sum_over_tape_row_unified() {
        use crate::composed_tape::TapeRow;
        use crate::strref::TapeData;
        let bytes = br#"{"orders":[{"v":10},{"v":20},{"v":30},{"v":40},{"v":50}]}"#.to_vec();
        let tape = TapeData::parse(bytes).unwrap();
        let arr_idx = crate::strref::tape_walk_field_chain(&tape, &["orders"]).unwrap();
        let arr_row = TapeRow::new(&tape, arr_idx as u32);
        let arena = Arena::new();
        let stages = Composed::new(
            Filter::<TapeRow<'_>, _>::new(|r: &TapeRow<'_>| {
                r.get_field("v").and_then(|v| v.as_int()).map_or(false, |n| n >= 30)
            }),
            MapField::<TapeRow<'_>>::new(Arc::from("v")),
        );
        let iter = arr_row.array_children().unwrap();
        let out = run_pipeline::<TapeRow<'_>, SumSink>(&arena, iter, &stages);
        assert!(matches!(out, BVal::Int(120)), "got {:?}", out);
    }

    #[cfg(feature = "simd-json")]
    #[test]
    fn flat_map_field_unified_over_tape() {
        use crate::composed_tape::TapeRow;
        use crate::strref::TapeData;
        let bytes = br#"{"orders":[{"items":[1,2,3]},{"items":[4,5]},{"items":[6]}]}"#.to_vec();
        let tape = TapeData::parse(bytes).unwrap();
        let arr_idx = crate::strref::tape_walk_field_chain(&tape, &["orders"]).unwrap();
        let arr_row = TapeRow::new(&tape, arr_idx as u32);
        let arena = Arena::new();
        let stages = FlatMapField::<TapeRow<'_>>::new(Arc::from("items"));
        let iter = arr_row.array_children().unwrap();
        let out = run_pipeline::<TapeRow<'_>, SumSink>(&arena, iter, &stages);
        // 1+2+3+4+5+6 = 21
        assert!(matches!(out, BVal::Int(21)), "got {:?}", out);
    }
}
