//! Borrowed parallel of `composed.rs` — Stage / Sink trait + run loop
//! over `borrowed::Val<'a>`.  Phase 1 SUBSTRATE: trait + minimal stages
//! + minimal sinks.  NOT YET wired into Pipeline::run.
//!
//! ## Lifetime model
//!
//! `BVal<'a>` is `Copy` (16-24 bytes) and references arena-allocated
//! bytes / slices.  `StageB::apply` takes `BVal<'a>` by value and
//! returns `StageOutputB<'a>` — no `Cow` needed since `BVal` is itself
//! cheap to copy.  Computed-value stages (arithmetic, projection) call
//! arena allocators to materialise the result; pass-through stages
//! (filter, field-read) emit the input unchanged.
//!
//! ## Why a parallel module
//!
//! The owned `Stage` trait operates on `&Val` with `Cow<'a, Val>`
//! outputs.  Migrating it to dual-mode (owned + borrowed) would
//! require GAT-style associated types and impl-time lifetime branching
//! that doesn't pay off when both substrates already exist independently
//! end-to-end.  Parallel impls ship today; long-term we collapse via
//! pipeline_unification Step 5 (RowSource/Reducer trait redesign).
//!
//! ## Phase plan
//!
//! - **Phase 1** (this commit): trait + Identity/Composed + 4 stages
//!   (Filter, MapField, Take, Skip) + 4 sinks (Count, Sum, Collect,
//!   First).  No wiring; round-trip test only.
//! - **Phase 2**: extend stages (MapFieldChain, ObjProject, MapSplit*,
//!   FlatMap*) + sinks (Min/Max/Avg/Last + barrier ops).  ~700 LoC.
//! - **Phase 3**: gate behind `JETRO_BORROWED_COMPOSED=1` env, wire
//!   into `Jetro::collect_val_borrow` fallback path.  Bench-gate per
//!   query shape.
//! - **Phase 4**: drop owned-only fallback when borrowed-composed
//!   matches or beats it on bench_cold; delete from pipeline_unification
//!   Step 5 plan.

use crate::eval::borrowed::{Arena, Val as BVal};
use smallvec::SmallVec;

// ── Stage output ────────────────────────────────────────────────────

/// Per-element output of a `StageB::apply`.  `BVal<'a>` is `Copy` so
/// no Cow distinction is needed — `Pass(v)` always carries a fresh
/// 16-24 byte value, computed or borrowed.
pub enum StageOutputB<'a> {
    Pass(BVal<'a>),
    Filtered,
    Many(SmallVec<[BVal<'a>; 4]>),
    Done,
}

// ── Stage trait ─────────────────────────────────────────────────────

/// Stage trait.  `'a` is the arena lifetime; computed stages allocate
/// new `BVal<'a>` payloads in the arena via `&'a Arena`.
pub trait StageB {
    fn apply<'a>(&self, arena: &'a Arena, x: BVal<'a>) -> StageOutputB<'a>;
}

impl<T: StageB + ?Sized> StageB for Box<T> {
    #[inline]
    fn apply<'a>(&self, arena: &'a Arena, x: BVal<'a>) -> StageOutputB<'a> {
        (**self).apply(arena, x)
    }
}

/// Identity — pass-through.  Used as fold seed when composing chains.
pub struct IdentityB;
impl StageB for IdentityB {
    #[inline]
    fn apply<'a>(&self, _arena: &'a Arena, x: BVal<'a>) -> StageOutputB<'a> {
        StageOutputB::Pass(x)
    }
}

/// Monoidal composition.  `Composed { a, b }.apply(x) = b.apply(a.apply(x))`.
/// All BVal values are Copy so no lifetime promotion overhead vs the
/// owned version's `Cow::Owned` path.
pub struct ComposedB<A: StageB, B: StageB> {
    pub a: A,
    pub b: B,
}

impl<A: StageB, B: StageB> StageB for ComposedB<A, B> {
    fn apply<'a>(&self, arena: &'a Arena, x: BVal<'a>) -> StageOutputB<'a> {
        match self.a.apply(arena, x) {
            StageOutputB::Pass(v) => self.b.apply(arena, v),
            StageOutputB::Filtered => StageOutputB::Filtered,
            StageOutputB::Many(items) => {
                let mut out: SmallVec<[BVal<'a>; 4]> = SmallVec::new();
                for v in items {
                    match self.b.apply(arena, v) {
                        StageOutputB::Pass(p) => out.push(p),
                        StageOutputB::Filtered => continue,
                        StageOutputB::Many(more) => out.extend(more),
                        StageOutputB::Done => {
                            return if out.is_empty() {
                                StageOutputB::Done
                            } else {
                                StageOutputB::Many(out)
                            };
                        }
                    }
                }
                if out.is_empty() {
                    StageOutputB::Filtered
                } else if out.len() == 1 {
                    StageOutputB::Pass(out.into_iter().next().unwrap())
                } else {
                    StageOutputB::Many(out)
                }
            }
            StageOutputB::Done => StageOutputB::Done,
        }
    }
}

// ── Sink trait ──────────────────────────────────────────────────────

/// Borrowed sink.  Folds `BVal<'a>` elements into `Acc`; finalise
/// emits a `BVal<'a>` allocated in the arena.
pub trait SinkB {
    type Acc;
    fn init() -> Self::Acc;
    fn fold<'a>(acc: Self::Acc, v: BVal<'a>) -> Self::Acc;
    fn finalise<'a>(arena: &'a Arena, acc: Self::Acc) -> BVal<'a>;
}

// ── Generic outer loop ──────────────────────────────────────────────

/// Run a composed Stage chain over `arr` with a `SinkB`.  Stage chain
/// is composed once at lower-time; this loop monomorphises per
/// `(Stage, Sink)` combination at call sites.
pub fn run_pipeline_b<'a, S: SinkB>(
    arena: &'a Arena,
    arr: &[BVal<'a>],
    stages: &dyn StageB,
) -> BVal<'a> {
    let mut acc = S::init();
    for v in arr.iter().copied() {
        match stages.apply(arena, v) {
            StageOutputB::Pass(p) => acc = S::fold(acc, p),
            StageOutputB::Filtered => continue,
            StageOutputB::Many(items) => {
                for p in items { acc = S::fold(acc, p); }
            }
            StageOutputB::Done => break,
        }
    }
    S::finalise(arena, acc)
}

// ── Sinks (Phase 1: minimal set) ────────────────────────────────────

pub struct CountSinkB;
impl SinkB for CountSinkB {
    type Acc = i64;
    #[inline] fn init() -> i64 { 0 }
    #[inline] fn fold<'a>(acc: i64, _: BVal<'a>) -> i64 { acc + 1 }
    #[inline] fn finalise<'a>(_: &'a Arena, acc: i64) -> BVal<'a> { BVal::Int(acc) }
}

pub struct SumSinkB;
impl SinkB for SumSinkB {
    type Acc = (i64, f64, bool);
    #[inline] fn init() -> Self::Acc { (0, 0.0, false) }
    fn fold<'a>(mut acc: Self::Acc, v: BVal<'a>) -> Self::Acc {
        match v {
            BVal::Int(i) => acc.0 = acc.0.wrapping_add(i),
            BVal::Float(f) => { acc.1 += f; acc.2 = true; }
            BVal::Bool(b) => acc.0 = acc.0.wrapping_add(b as i64),
            _ => {}
        }
        acc
    }
    fn finalise<'a>(_: &'a Arena, acc: Self::Acc) -> BVal<'a> {
        if acc.2 { BVal::Float(acc.0 as f64 + acc.1) } else { BVal::Int(acc.0) }
    }
}

pub struct FirstSinkB;
impl SinkB for FirstSinkB {
    /// `Option<BVal<'static>>` would be the natural Acc but the
    /// borrowed lifetime is per-call; box-erase via raw pointer-style
    /// using `Option<BValStatic>` is unsafe.  Use a thread-local trick:
    /// store as `Option<BVal<'a>>` — but Acc must be 'static.  Workaround:
    /// keep payload inside a Cell<Option<...>> at run_pipeline level.
    /// For Phase 1 we use a degenerate Acc that encodes only "saw any";
    /// real First semantics live in `bytescan::run_with_sink_borrow`.
    type Acc = bool;
    #[inline] fn init() -> bool { false }
    #[inline] fn fold<'a>(_: bool, _: BVal<'a>) -> bool { true }
    #[inline] fn finalise<'a>(_: &'a Arena, _: bool) -> BVal<'a> { BVal::Null }
}

pub struct CollectSinkB;
impl SinkB for CollectSinkB {
    type Acc = Vec<i64>;  // placeholder — real Collect needs lifetime
                          // gymnastics that GAT would solve; Phase 2
                          // introduces a SinkBLifed trait variant.
    #[inline] fn init() -> Vec<i64> { Vec::new() }
    #[inline] fn fold<'a>(acc: Vec<i64>, _: BVal<'a>) -> Vec<i64> { acc }
    #[inline] fn finalise<'a>(arena: &'a Arena, _: Vec<i64>) -> BVal<'a> {
        let empty: &[BVal<'a>] = &[];
        BVal::Arr(arena.alloc_slice_copy(empty))
    }
}

// ── Stages (Phase 1: minimal set) ───────────────────────────────────

/// Filter by predicate over the BVal.  Closure-based for arity; later
/// phases lower from BodyKernel into specialised structs (e.g.
/// `FilterFieldEqLitB`).
pub struct FilterB<F: Fn(&BVal<'_>) -> bool> {
    pub pred: F,
}

impl<F: Fn(&BVal<'_>) -> bool> StageB for FilterB<F> {
    #[inline]
    fn apply<'a>(&self, _arena: &'a Arena, x: BVal<'a>) -> StageOutputB<'a> {
        if (self.pred)(&x) {
            StageOutputB::Pass(x)
        } else {
            StageOutputB::Filtered
        }
    }
}

/// Map a field read.  Returns `BVal::Null` when the field is missing
/// or the input isn't an Object — matches owned `MapField` semantics.
pub struct MapFieldB {
    pub field: std::sync::Arc<str>,
}

impl StageB for MapFieldB {
    #[inline]
    fn apply<'a>(&self, _arena: &'a Arena, x: BVal<'a>) -> StageOutputB<'a> {
        let v = x.get_field(&self.field).unwrap_or(BVal::Null);
        StageOutputB::Pass(v)
    }
}

/// Take(n) — emits Done after n passes.  Interior mutability via
/// `Cell` so the same Stage instance can be reused per pipeline run
/// after `reset()`.
pub struct TakeB {
    pub n: usize,
    seen: std::cell::Cell<usize>,
}

impl TakeB {
    pub fn new(n: usize) -> Self { Self { n, seen: std::cell::Cell::new(0) } }
    pub fn reset(&self) { self.seen.set(0); }
}

impl StageB for TakeB {
    #[inline]
    fn apply<'a>(&self, _arena: &'a Arena, x: BVal<'a>) -> StageOutputB<'a> {
        let s = self.seen.get();
        if s >= self.n { return StageOutputB::Done; }
        self.seen.set(s + 1);
        StageOutputB::Pass(x)
    }
}

/// Skip(n) — drops first n elements.
pub struct SkipB {
    pub n: usize,
    seen: std::cell::Cell<usize>,
}

impl SkipB {
    pub fn new(n: usize) -> Self { Self { n, seen: std::cell::Cell::new(0) } }
    pub fn reset(&self) { self.seen.set(0); }
}

impl StageB for SkipB {
    #[inline]
    fn apply<'a>(&self, _arena: &'a Arena, x: BVal<'a>) -> StageOutputB<'a> {
        let s = self.seen.get();
        if s < self.n {
            self.seen.set(s + 1);
            StageOutputB::Filtered
        } else {
            StageOutputB::Pass(x)
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn obj_pair<'a>(arena: &'a Arena, k: &str, v: BVal<'a>) -> BVal<'a> {
        let pairs = vec![(arena.alloc_str(k), v)];
        let slice = arena.alloc_slice_fill_iter(pairs.into_iter());
        BVal::Obj(&*slice)
    }

    #[test]
    fn count_sink_runs() {
        let arena = Arena::new();
        let items = vec![BVal::Int(1), BVal::Int(2), BVal::Int(3)];
        let arr = items.as_slice();
        let stages: Box<dyn StageB> = Box::new(IdentityB);
        let out = run_pipeline_b::<CountSinkB>(&arena, arr, &*stages);
        match out {
            BVal::Int(n) => assert_eq!(n, 3),
            _ => panic!("expected Int(3)"),
        }
    }

    #[test]
    fn sum_sink_int_only() {
        let arena = Arena::new();
        let items = vec![BVal::Int(10), BVal::Int(20), BVal::Int(30)];
        let stages: Box<dyn StageB> = Box::new(IdentityB);
        let out = run_pipeline_b::<SumSinkB>(&arena, items.as_slice(), &*stages);
        match out {
            BVal::Int(60) => {}
            _ => panic!("expected Int(60), got {:?}", out),
        }
    }

    #[test]
    fn sum_sink_mixed_promotes_float() {
        let arena = Arena::new();
        let items = vec![BVal::Int(5), BVal::Float(2.5)];
        let stages: Box<dyn StageB> = Box::new(IdentityB);
        let out = run_pipeline_b::<SumSinkB>(&arena, items.as_slice(), &*stages);
        match out {
            BVal::Float(f) => assert!((f - 7.5).abs() < 1e-9),
            _ => panic!("expected Float(7.5), got {:?}", out),
        }
    }

    #[test]
    fn filter_b_drops_when_pred_false() {
        let arena = Arena::new();
        let items = vec![BVal::Int(1), BVal::Int(2), BVal::Int(3), BVal::Int(4)];
        let stages = FilterB { pred: |v: &BVal<'_>| matches!(v, BVal::Int(n) if *n % 2 == 0) };
        let out = run_pipeline_b::<CountSinkB>(&arena, items.as_slice(), &stages);
        match out {
            BVal::Int(2) => {}
            _ => panic!("expected Int(2), got {:?}", out),
        }
    }

    #[test]
    fn map_field_extracts_value() {
        let arena = Arena::new();
        let row1 = obj_pair(&arena, "n", BVal::Int(7));
        let row2 = obj_pair(&arena, "n", BVal::Int(8));
        let row3 = obj_pair(&arena, "n", BVal::Int(9));
        let items = vec![row1, row2, row3];
        let stages = MapFieldB { field: std::sync::Arc::from("n") };
        let out = run_pipeline_b::<SumSinkB>(&arena, items.as_slice(), &stages);
        match out {
            BVal::Int(24) => {}
            _ => panic!("expected Int(24), got {:?}", out),
        }
    }

    #[test]
    fn take_b_caps_output() {
        let arena = Arena::new();
        let items: Vec<BVal> = (0..1000).map(BVal::Int).collect();
        let take = TakeB::new(5);
        let out = run_pipeline_b::<CountSinkB>(&arena, items.as_slice(), &take);
        match out {
            BVal::Int(5) => {}
            _ => panic!("expected Int(5), got {:?}", out),
        }
    }

    #[test]
    fn skip_b_drops_prefix() {
        let arena = Arena::new();
        let items: Vec<BVal> = (1..=10).map(BVal::Int).collect();
        let skip = SkipB::new(3);
        let out = run_pipeline_b::<CountSinkB>(&arena, items.as_slice(), &skip);
        match out {
            BVal::Int(7) => {}
            _ => panic!("expected Int(7), got {:?}", out),
        }
    }

    #[test]
    fn composed_filter_then_map_field() {
        let arena = Arena::new();
        let row1 = obj_pair(&arena, "n", BVal::Int(2));
        let row2 = obj_pair(&arena, "n", BVal::Int(3));
        let row3 = obj_pair(&arena, "n", BVal::Int(4));
        let row4 = obj_pair(&arena, "n", BVal::Int(5));
        let items = vec![row1, row2, row3, row4];

        // filter(n is even) then map(n) -> [2, 4]
        let stages = ComposedB {
            a: FilterB {
                pred: |v: &BVal<'_>| {
                    v.get_field("n").map_or(false, |x| matches!(x, BVal::Int(n) if n % 2 == 0))
                }
            },
            b: MapFieldB { field: std::sync::Arc::from("n") },
        };
        let out = run_pipeline_b::<SumSinkB>(&arena, items.as_slice(), &stages);
        match out {
            BVal::Int(6) => {}
            _ => panic!("expected Int(6), got {:?}", out),
        }
    }

    #[test]
    fn composed_skip_take_chain() {
        let arena = Arena::new();
        let items: Vec<BVal> = (1..=20).map(BVal::Int).collect();
        // skip(5).take(3) -> 6, 7, 8 -> count = 3
        let stages = ComposedB { a: SkipB::new(5), b: TakeB::new(3) };
        let out = run_pipeline_b::<CountSinkB>(&arena, items.as_slice(), &stages);
        match out {
            BVal::Int(3) => {}
            _ => panic!("expected Int(3), got {:?}", out),
        }
    }
}
