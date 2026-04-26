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

// Re-exports of stage structs that live in composed.rs.  ONE struct
// per stage; both the owned `composed::Stage` trait and this module's
// substrate-generic `Stage<R>` trait are implemented over the same
// type (see "Bridge" section at end of composed.rs).  Per
// pipeline_unification: stages are not duplicated — they have ONE
// definition with TWO trait impls.
pub use crate::composed::{
    MapField, MapFieldChain, FlatMapField, FlatMapFieldChain,
    Take, Skip,
};

// ── Sink trait ──────────────────────────────────────────────────────

pub trait Sink {
    type Acc<'a>;
    fn init<'a>() -> Self::Acc<'a>;
    fn fold<'a, R: Row<'a>>(arena: &'a Arena, acc: Self::Acc<'a>, v: R) -> Self::Acc<'a>;
    fn finalise<'a>(arena: &'a Arena, acc: Self::Acc<'a>) -> BVal<'a>;
}

// Sink dedup: composed.rs's CountSink/SumSink/MinSink/MaxSink/AvgSink/
// FirstSink/LastSink/CollectSink each carry TWO trait impls — owned
// `composed::Sink` + this module's substrate-generic `Sink`.  Re-export
// the unit-structs so users of the borrow runner don't need to reach
// into composed:: directly.
pub use crate::composed::{
    CountSink, SumSink, MinSink, MaxSink, AvgSink,
    FirstSink, LastSink, CollectSink,
};

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
            MapField::new(Arc::from("v")),
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
            Skip::new(1),
            Take::new(2),
        );
        let out = run_pipeline::<BVal, CountSink>(&arena, orders.iter().copied(), &stages);
        assert!(matches!(out, BVal::Int(2)));
    }

    #[test]
    fn collect_over_bval_materialises() {
        let arena = Arena::new();
        let orders = make_orders(&arena);
        let stages = MapField::new(Arc::from("id"));
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
            MapField::new(Arc::from("v")),
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
        let stages = FlatMapField::new(Arc::from("items"));
        let iter = arr_row.array_children().unwrap();
        let out = run_pipeline::<TapeRow<'_>, SumSink>(&arena, iter, &stages);
        // 1+2+3+4+5+6 = 21
        assert!(matches!(out, BVal::Int(21)), "got {:?}", out);
    }
}
