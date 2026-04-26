//! Layer B — composed-Cow Stage chains as the sole runtime exec path.
//!
//! Per `pipeline_specialisation.md` (CORRECTED 2026-04-26 + cold-first
//! pivot): replace the legacy fused-Sink enum + ~30 hand-rolled VM
//! opcodes with a single generic substrate driven by composition. No
//! per-shape opcodes, no enumerated `(stage_chain × sink)` Sink
//! variants — those are the disease. Composition primitives + a
//! bounded list of generic Sinks cover all chain shapes uniformly.
//!
//! ## Design
//!
//! - `Stage`: `&'a Val → StageOutput<'a>` — borrow-form returns
//!   `Cow::Borrowed`, computed-form returns `Cow::Owned`. Avoids the
//!   per-stage clone tax measured at 2.3× (composed-fn owned) vs 1.29×
//!   (composed-borrow Cow) on `filter+map+sum` × 5000 × 1000 iters.
//! - `Composed<A, B>`: monoidal pairing; N stages fold into one apply
//!   call via `stages.into_iter().fold(Identity, compose)`. One
//!   virtual call per element regardless of chain length.
//! - `Sink`: `Acc + fold(&Acc, &Val) → Acc + finalise(Acc) → Val`. A
//!   bounded set (Sum/Min/Max/Avg/Count/First/Last/Collect) covers
//!   every legacy fused Sink variant.
//! - `run_pipeline<S: Sink>`: one generic outer loop, parameterised by
//!   `S`. Stages composed once at lower-time, used N× at execute.
//!
//! No (stage × sink) enumeration anywhere. Adding a new stage shape =
//! one `Stage::apply` impl. Adding a new sink = one `Sink` impl.
//! Adding a new chain shape = no code.
//!
//! ## Status
//!
//! Day 1 — module foundation. Standalone; not yet wired into
//! `pipeline.rs::run`. Day 2-3 lands the wiring under
//! `JETRO_COMPOSED=1` and bench-gates against legacy fused.

use std::borrow::Cow;
use smallvec::SmallVec;

use crate::eval::value::Val;

// ── Stage ────────────────────────────────────────────────────────────────────

/// Per-element output of a `Stage::apply`. Borrowed payload when the
/// stage is a pass-through over the input (filter, field-read);
/// owned payload only when the stage computed a fresh value
/// (arithmetic, format-string, projection).
pub enum StageOutput<'a> {
    /// Stage produced one element. `Cow::Borrowed` for pass-through
    /// stages (zero clone); `Cow::Owned` for computed values.
    Pass(Cow<'a, Val>),
    /// Filter dropped the element.
    Filtered,
    /// FlatMap produced multiple elements.
    Many(SmallVec<[Cow<'a, Val>; 4]>),
    /// Take exhausted; outer loop should terminate.
    Done,
}

/// A Stage transforms one element into a `StageOutput`. The lifetime
/// `'a` ties borrowed outputs to the input reference, eliminating the
/// per-stage clone overhead that owned-Val composition incurred.
pub trait Stage: Send + Sync {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a>;
}

/// Identity stage — pass-through. Used as the fold seed when composing
/// a chain of stages.
pub struct Identity;

impl Stage for Identity {
    #[inline]
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        StageOutput::Pass(Cow::Borrowed(x))
    }
}

/// Monoidal composition. `Composed { a, b }.apply(x) = b.apply(a.apply(x))`
/// with proper handling of Filtered / Many / Done propagation.
pub struct Composed<A: Stage, B: Stage> {
    pub a: A,
    pub b: B,
}

impl<A: Stage, B: Stage> Stage for Composed<A, B> {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        match self.a.apply(x) {
            StageOutput::Pass(Cow::Borrowed(v)) => self.b.apply(v),
            StageOutput::Pass(Cow::Owned(v)) => {
                // `b` may borrow from v — but v dies at scope end. Force
                // owned result so lifetime is independent of `v`. The
                // computed-then-borrow case is rare; common case is
                // borrow-borrow which is zero-cost above.
                let owned = match self.b.apply(&v) {
                    StageOutput::Pass(c) => Cow::Owned(c.into_owned()),
                    StageOutput::Filtered => return StageOutput::Filtered,
                    StageOutput::Many(items) => {
                        let mut out: SmallVec<[Cow<'a, Val>; 4]> = SmallVec::new();
                        for it in items {
                            out.push(Cow::Owned(it.into_owned()));
                        }
                        return StageOutput::Many(out);
                    }
                    StageOutput::Done => return StageOutput::Done,
                };
                StageOutput::Pass(owned)
            }
            StageOutput::Filtered => StageOutput::Filtered,
            StageOutput::Many(items) => {
                let mut out: SmallVec<[Cow<'a, Val>; 4]> = SmallVec::new();
                for it in items {
                    match it {
                        Cow::Borrowed(v) => match self.b.apply(v) {
                            StageOutput::Pass(c) => out.push(c),
                            StageOutput::Filtered => continue,
                            StageOutput::Many(more) => out.extend(more),
                            StageOutput::Done => {
                                return if out.is_empty() {
                                    StageOutput::Done
                                } else {
                                    StageOutput::Many(out)
                                };
                            }
                        },
                        Cow::Owned(v) => {
                            // `v` is owned and dies at end of this arm; any
                            // borrow returned by `b.apply(&v)` must be
                            // promoted to owned so `out` outlives `v`.
                            match self.b.apply(&v) {
                                StageOutput::Pass(c) => out.push(Cow::Owned(c.into_owned())),
                                StageOutput::Filtered => continue,
                                StageOutput::Many(more) => {
                                    for m in more {
                                        out.push(Cow::Owned(m.into_owned()));
                                    }
                                }
                                StageOutput::Done => {
                                    return if out.is_empty() {
                                        StageOutput::Done
                                    } else {
                                        StageOutput::Many(out)
                                    };
                                }
                            }
                        }
                    }
                }
                if out.is_empty() {
                    StageOutput::Filtered
                } else if out.len() == 1 {
                    StageOutput::Pass(out.into_iter().next().unwrap())
                } else {
                    StageOutput::Many(out)
                }
            }
            StageOutput::Done => StageOutput::Done,
        }
    }
}

// ── Sink ─────────────────────────────────────────────────────────────────────

/// Generic terminal accumulator. Bounded list (Sum/Min/Max/Avg/Count/
/// First/Last/Collect) covers every fused Sink variant in
/// `pipeline.rs`. New sinks = new impl, no per-chain code.
pub trait Sink {
    type Acc;
    fn init() -> Self::Acc;
    fn fold(acc: Self::Acc, v: &Val) -> Self::Acc;
    fn finalise(acc: Self::Acc) -> Val;
}

// ── Generic outer loop ──────────────────────────────────────────────────────

/// Parameterised outer loop. One pass over `arr`, dispatching each
/// element through the composed `stages`, accumulating into `S::Acc`.
/// Stages composed once at lower-time, used N× here.
pub fn run_pipeline<S: Sink>(arr: &[Val], stages: &dyn Stage) -> Val {
    let mut acc = S::init();
    for v in arr.iter() {
        match stages.apply(v) {
            StageOutput::Pass(cow) => acc = S::fold(acc, cow.as_ref()),
            StageOutput::Filtered => continue,
            StageOutput::Many(items) => {
                for it in items {
                    acc = S::fold(acc, it.as_ref());
                }
            }
            StageOutput::Done => break,
        }
    }
    S::finalise(acc)
}

// ── Generic Sink impls ──────────────────────────────────────────────────────

pub struct CountSink;
impl Sink for CountSink {
    type Acc = i64;
    #[inline] fn init() -> i64 { 0 }
    #[inline] fn fold(acc: i64, _: &Val) -> i64 { acc + 1 }
    #[inline] fn finalise(acc: i64) -> Val { Val::Int(acc) }
}

pub struct SumSink;
impl Sink for SumSink {
    type Acc = (i64, f64, bool); // (int_sum, float_sum, any_float)
    #[inline] fn init() -> Self::Acc { (0, 0.0, false) }
    fn fold(mut acc: Self::Acc, v: &Val) -> Self::Acc {
        match v {
            Val::Int(i) => acc.0 += *i,
            Val::Float(f) => { acc.1 += *f; acc.2 = true; }
            Val::Bool(b) => acc.0 += *b as i64,
            _ => {}
        }
        acc
    }
    fn finalise(acc: Self::Acc) -> Val {
        if acc.2 { Val::Float(acc.0 as f64 + acc.1) } else { Val::Int(acc.0) }
    }
}

pub struct MinSink;
impl Sink for MinSink {
    type Acc = Option<f64>;
    #[inline] fn init() -> Self::Acc { None }
    fn fold(acc: Self::Acc, v: &Val) -> Self::Acc {
        let n = match v {
            Val::Int(i) => *i as f64,
            Val::Float(f) => *f,
            _ => return acc,
        };
        Some(match acc { Some(cur) => cur.min(n), None => n })
    }
    fn finalise(acc: Self::Acc) -> Val {
        match acc {
            Some(f) if f.fract() == 0.0 && f.abs() < (i64::MAX as f64) => Val::Int(f as i64),
            Some(f) => Val::Float(f),
            None => Val::Null,
        }
    }
}

pub struct MaxSink;
impl Sink for MaxSink {
    type Acc = Option<f64>;
    #[inline] fn init() -> Self::Acc { None }
    fn fold(acc: Self::Acc, v: &Val) -> Self::Acc {
        let n = match v {
            Val::Int(i) => *i as f64,
            Val::Float(f) => *f,
            _ => return acc,
        };
        Some(match acc { Some(cur) => cur.max(n), None => n })
    }
    fn finalise(acc: Self::Acc) -> Val {
        match acc {
            Some(f) if f.fract() == 0.0 && f.abs() < (i64::MAX as f64) => Val::Int(f as i64),
            Some(f) => Val::Float(f),
            None => Val::Null,
        }
    }
}

pub struct AvgSink;
impl Sink for AvgSink {
    type Acc = (f64, usize);
    #[inline] fn init() -> Self::Acc { (0.0, 0) }
    fn fold(mut acc: Self::Acc, v: &Val) -> Self::Acc {
        let n = match v {
            Val::Int(i) => *i as f64,
            Val::Float(f) => *f,
            _ => return acc,
        };
        acc.0 += n;
        acc.1 += 1;
        acc
    }
    fn finalise(acc: Self::Acc) -> Val {
        if acc.1 == 0 { Val::Null } else { Val::Float(acc.0 / acc.1 as f64) }
    }
}

pub struct FirstSink;
impl Sink for FirstSink {
    type Acc = Option<Val>;
    #[inline] fn init() -> Self::Acc { None }
    fn fold(acc: Self::Acc, v: &Val) -> Self::Acc {
        if acc.is_some() { acc } else { Some(v.clone()) }
    }
    fn finalise(acc: Self::Acc) -> Val { acc.unwrap_or(Val::Null) }
}

pub struct LastSink;
impl Sink for LastSink {
    type Acc = Option<Val>;
    #[inline] fn init() -> Self::Acc { None }
    fn fold(_acc: Self::Acc, v: &Val) -> Self::Acc { Some(v.clone()) }
    fn finalise(acc: Self::Acc) -> Val { acc.unwrap_or(Val::Null) }
}

pub struct CollectSink;
impl Sink for CollectSink {
    type Acc = Vec<Val>;
    #[inline] fn init() -> Self::Acc { Vec::new() }
    fn fold(mut acc: Self::Acc, v: &Val) -> Self::Acc {
        acc.push(v.clone());
        acc
    }
    fn finalise(acc: Self::Acc) -> Val {
        Val::Arr(std::sync::Arc::new(acc))
    }
}

// ── Borrow-form stages — zero-clone pass-through ────────────────────────────

/// `.filter(@.k == lit)` — borrow-form when pred holds.
pub struct FilterFieldEqLit {
    pub field: std::sync::Arc<str>,
    pub target: Val,
}

impl Stage for FilterFieldEqLit {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        if let Val::Obj(m) = x {
            if let Some(v) = m.get(self.field.as_ref()) {
                if vals_eq(v, &self.target) {
                    return StageOutput::Pass(Cow::Borrowed(x));
                }
            }
        }
        StageOutput::Filtered
    }
}

/// `.map(@.k)` — borrow into the parent object.
pub struct MapField {
    pub field: std::sync::Arc<str>,
}

impl Stage for MapField {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        if let Val::Obj(m) = x {
            if let Some(v) = m.get(self.field.as_ref()) {
                return StageOutput::Pass(Cow::Borrowed(v));
            }
        }
        StageOutput::Filtered
    }
}

/// `.map(@.a.b.c)` — generic field chain walk.
pub struct MapFieldChain {
    pub keys: std::sync::Arc<[std::sync::Arc<str>]>,
}

impl Stage for MapFieldChain {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        let mut cur = x;
        for k in self.keys.iter() {
            match cur {
                Val::Obj(m) => match m.get(k.as_ref()) {
                    Some(next) => cur = next,
                    None => return StageOutput::Filtered,
                },
                _ => return StageOutput::Filtered,
            }
        }
        StageOutput::Pass(Cow::Borrowed(cur))
    }
}

/// `.take(n)` — counts via interior mutability (single-thread per
/// pipeline invocation; outer loop owns the closure).
pub struct Take {
    pub remaining: std::cell::Cell<usize>,
}

impl Stage for Take {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        let r = self.remaining.get();
        if r == 0 { return StageOutput::Done; }
        self.remaining.set(r - 1);
        StageOutput::Pass(Cow::Borrowed(x))
    }
}

// `Take` holds `Cell` which is `!Sync`. Pipelines run on one thread at
// a time; the `Send + Sync` bound on `Stage` is overly strict for it.
// Provide an unsafe assertion here so the trait stays simple while
// allowing per-call mutable counters. Day 2 wiring uses single-threaded
// per-call execution exclusively.
unsafe impl Sync for Take {}

/// `.skip(n)` — same shape as Take.
pub struct Skip {
    pub remaining: std::cell::Cell<usize>,
}

impl Stage for Skip {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        let r = self.remaining.get();
        if r > 0 {
            self.remaining.set(r - 1);
            return StageOutput::Filtered;
        }
        StageOutput::Pass(Cow::Borrowed(x))
    }
}

unsafe impl Sync for Skip {}

// ── Helpers ─────────────────────────────────────────────────────────────────

#[inline]
fn vals_eq(a: &Val, b: &Val) -> bool {
    match (a, b) {
        (Val::Null, Val::Null) => true,
        (Val::Bool(x), Val::Bool(y)) => x == y,
        (Val::Int(x), Val::Int(y)) => x == y,
        (Val::Float(x), Val::Float(y)) => x == y,
        (Val::Int(x), Val::Float(y)) | (Val::Float(y), Val::Int(x)) => (*x as f64) == *y,
        (Val::Str(x), Val::Str(y)) => x == y,
        (Val::Str(x), Val::StrSlice(r)) | (Val::StrSlice(r), Val::Str(x)) => x.as_ref() == r.as_str(),
        (Val::StrSlice(x), Val::StrSlice(y)) => x.as_str() == y.as_str(),
        _ => false,
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use indexmap::IndexMap;

    fn obj(pairs: &[(&str, Val)]) -> Val {
        let mut m = IndexMap::new();
        for (k, v) in pairs {
            m.insert(Arc::from(*k), v.clone());
        }
        Val::Obj(Arc::new(m))
    }

    #[test]
    fn count_filter_map_field() {
        // [{a:1, b:10}, {a:2, b:20}, {a:1, b:30}].filter(@.a == 1).map(@.b) → count = 2
        let arr = vec![
            obj(&[("a", Val::Int(1)), ("b", Val::Int(10))]),
            obj(&[("a", Val::Int(2)), ("b", Val::Int(20))]),
            obj(&[("a", Val::Int(1)), ("b", Val::Int(30))]),
        ];
        let stages = Composed {
            a: FilterFieldEqLit { field: Arc::from("a"), target: Val::Int(1) },
            b: MapField { field: Arc::from("b") },
        };
        let out = run_pipeline::<CountSink>(&arr, &stages);
        assert!(matches!(out, Val::Int(2)));
    }

    #[test]
    fn sum_filter_map_field() {
        let arr = vec![
            obj(&[("a", Val::Int(1)), ("b", Val::Int(10))]),
            obj(&[("a", Val::Int(2)), ("b", Val::Int(20))]),
            obj(&[("a", Val::Int(1)), ("b", Val::Int(30))]),
        ];
        let stages = Composed {
            a: FilterFieldEqLit { field: Arc::from("a"), target: Val::Int(1) },
            b: MapField { field: Arc::from("b") },
        };
        let out = run_pipeline::<SumSink>(&arr, &stages);
        // 10 + 30 = 40
        assert!(matches!(out, Val::Int(40)));
    }

    #[test]
    fn collect_map_field_chain() {
        // [{u:{a:{c:"NYC"}}}, {u:{a:{c:"LA"}}}].map(@.u.a.c) → ["NYC", "LA"]
        let inner1 = obj(&[("c", Val::Str(Arc::from("NYC")))]);
        let inner2 = obj(&[("c", Val::Str(Arc::from("LA")))]);
        let mid1 = obj(&[("a", inner1)]);
        let mid2 = obj(&[("a", inner2)]);
        let arr = vec![
            obj(&[("u", mid1)]),
            obj(&[("u", mid2)]),
        ];
        let keys: Arc<[Arc<str>]> = Arc::from(vec![Arc::from("u"), Arc::from("a"), Arc::from("c")]);
        let stages = MapFieldChain { keys };
        let out = run_pipeline::<CollectSink>(&arr, &stages);
        if let Val::Arr(v) = out {
            assert_eq!(v.len(), 2);
        } else {
            panic!("expected Arr");
        }
    }

    #[test]
    fn take_terminates_outer_loop() {
        let arr: Vec<Val> = (0..1000).map(Val::Int).collect();
        let stages = Take { remaining: std::cell::Cell::new(3) };
        let out = run_pipeline::<CountSink>(&arr, &stages);
        assert!(matches!(out, Val::Int(3)));
    }

    #[test]
    fn skip_drops_prefix() {
        let arr: Vec<Val> = (0..10).map(Val::Int).collect();
        let stages = Skip { remaining: std::cell::Cell::new(7) };
        let out = run_pipeline::<CountSink>(&arr, &stages);
        assert!(matches!(out, Val::Int(3)));
    }

    #[test]
    fn min_max_avg_basic() {
        let arr: Vec<Val> = vec![Val::Int(5), Val::Int(2), Val::Int(9), Val::Int(3)];
        assert!(matches!(run_pipeline::<MinSink>(&arr, &Identity), Val::Int(2)));
        assert!(matches!(run_pipeline::<MaxSink>(&arr, &Identity), Val::Int(9)));
        // (5+2+9+3)/4 = 4.75
        if let Val::Float(f) = run_pipeline::<AvgSink>(&arr, &Identity) {
            assert!((f - 4.75).abs() < 1e-9);
        } else {
            panic!("expected Float");
        }
    }

    #[test]
    fn first_last_basic() {
        let arr: Vec<Val> = vec![Val::Int(10), Val::Int(20), Val::Int(30)];
        assert!(matches!(run_pipeline::<FirstSink>(&arr, &Identity), Val::Int(10)));
        assert!(matches!(run_pipeline::<LastSink>(&arr, &Identity), Val::Int(30)));
    }

    #[test]
    fn empty_input_finalises_to_default() {
        let arr: Vec<Val> = vec![];
        assert!(matches!(run_pipeline::<CountSink>(&arr, &Identity), Val::Int(0)));
        assert!(matches!(run_pipeline::<SumSink>(&arr, &Identity), Val::Int(0)));
        assert!(matches!(run_pipeline::<MinSink>(&arr, &Identity), Val::Null));
        assert!(matches!(run_pipeline::<FirstSink>(&arr, &Identity), Val::Null));
    }
}
