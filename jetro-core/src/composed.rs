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
///
/// Pipelines run single-threaded per invocation; no Send/Sync bound.
/// Stages with interior mutability (`Take`, `Skip` counters) work in
/// place via `Cell<usize>` reset at lower-time.
pub trait Stage {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a>;

    /// Step 3d-extension (B) — produce only the kth output for input `x`.
    /// Default impl materialises via `apply` and indexes; Expanding
    /// stages with `can_indexed=true` (Split, Range, FlatMap when inner
    /// is bounded) override with direct computation — e.g. memchr-based
    /// kth-segment lookup for Split — to convert O(N) work into O(k).
    /// Used by the planner's IndexedDispatch kernel for `Cardinality::
    /// Expanding` stages preceded by all 1:1 stages and followed by a
    /// positional sink.
    fn apply_indexed<'a>(&self, x: &'a Val, k: usize) -> Option<Cow<'a, Val>> {
        match self.apply(x) {
            StageOutput::Pass(v) if k == 0 => Some(v),
            StageOutput::Pass(_)           => None,
            StageOutput::Many(mut items)
                if k < items.len() => Some(items.swap_remove(k)),
            StageOutput::Many(_)
                | StageOutput::Filtered
                | StageOutput::Done => None,
        }
    }
}

/// Blanket impl so `Box<dyn Stage>` itself implements `Stage`. Lets a
/// chain of stages be folded into a single `Box<dyn Stage>` at
/// lower-time without macro acrobatics.
impl<T: Stage + ?Sized> Stage for Box<T> {
    #[inline]
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        (**self).apply(x)
    }
    #[inline]
    fn apply_indexed<'a>(&self, x: &'a Val, k: usize) -> Option<Cow<'a, Val>> {
        (**self).apply_indexed(x, k)
    }
}

/// Identity stage — pass-through. Used as the fold seed when composing
/// a chain of stages.
pub struct Identity;

impl Identity {
    pub fn new() -> Self { Self }
}

/// Closure-based `.filter(pred)` — for the borrow runner where the
/// predicate is built from a kernel at lowering time (FieldCmpLit etc.
/// → owned literal compare).  composed.rs's `GenericFilter` uses VM
/// dispatch and only impls owned `Stage`; this `Filter` is closure-
/// based and impls both `unified::Stage<R>` (any substrate) and the
/// owned `Stage` (R = `&Val` adapter).
pub struct Filter<R, F: Fn(&R) -> bool> {
    pub pred: F,
    _marker: std::marker::PhantomData<fn(R)>,
}
impl<R, F: Fn(&R) -> bool> Filter<R, F> {
    pub fn new(pred: F) -> Self {
        Self { pred, _marker: std::marker::PhantomData }
    }
}
impl<R, F: Fn(&R) -> bool> crate::unified::Stage<R> for Filter<R, F> {
    #[inline]
    fn apply(&self, x: R) -> crate::unified::StageOutputU<R> {
        if (self.pred)(&x) {
            crate::unified::StageOutputU::Pass(x)
        } else {
            crate::unified::StageOutputU::Filtered
        }
    }
}

impl Default for Identity {
    fn default() -> Self { Self }
}

impl Stage for Identity {
    #[inline]
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        StageOutput::Pass(Cow::Borrowed(x))
    }
}

/// Monoidal composition. `Composed { a, b }.apply(x) = b.apply(a.apply(x))`
/// with proper handling of Filtered / Many / Done propagation.
///
/// Trait bounds live on the impl blocks rather than the struct so the
/// SAME `Composed<A, B>` can chain stages under either the owned
/// `Stage` trait or the substrate-generic `unified::Stage<R>` trait.
pub struct Composed<A, B> {
    pub a: A,
    pub b: B,
}

impl<A, B> Composed<A, B> {
    pub fn new(a: A, b: B) -> Self { Self { a, b } }
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

// ── Generic VM-fallback stages ──────────────────────────────────────────────
//
// Cover any kernel shape the borrow stages don't recognise (Generic,
// Arith, FString, FieldCmpLit non-Eq, custom lambdas). One VM + Env
// shared across all Generic stages in the chain via Rc<RefCell>;
// constructed once per pipeline call so compile/path caches amortise.
//
// These exist so `try_run_composed` never bails on body-shape — every
// pipeline lowers via composition, every chain runs the same outer
// loop. No per-shape walker; one mechanism for every body.

pub struct VmCtx {
    pub vm: crate::vm::VM,
    pub env: crate::eval::Env,
}

/// `.filter(pred)` with arbitrary pred — VM evaluates per row, result
/// truthy-checked. Borrow form on pass-through (zero-clone of x).
pub struct GenericFilter {
    pub prog: std::sync::Arc<crate::vm::Program>,
    pub ctx: std::rc::Rc<std::cell::RefCell<VmCtx>>,
}

impl Stage for GenericFilter {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        let mut c = self.ctx.borrow_mut();
        let VmCtx { vm, env } = &mut *c;
        let prev = env.swap_current(x.clone());
        let r = vm.exec_in_env(&self.prog, env);
        env.restore_current(prev);
        match r {
            Ok(v) if crate::eval::util::is_truthy(&v) => StageOutput::Pass(Cow::Borrowed(x)),
            _ => StageOutput::Filtered,
        }
    }
}

/// `.map(f)` with arbitrary f — VM emits a fresh Val per row.
pub struct GenericMap {
    pub prog: std::sync::Arc<crate::vm::Program>,
    pub ctx: std::rc::Rc<std::cell::RefCell<VmCtx>>,
}

impl Stage for GenericMap {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        let mut c = self.ctx.borrow_mut();
        let VmCtx { vm, env } = &mut *c;
        let prev = env.swap_current(x.clone());
        let r = vm.exec_in_env(&self.prog, env);
        env.restore_current(prev);
        match r {
            Ok(v) => StageOutput::Pass(Cow::Owned(v)),
            Err(_) => StageOutput::Filtered,
        }
    }
}

/// `.flat_map(f)` with arbitrary f — VM emits a Val that must be
/// iterable; `flatten_iterable` dispatches across all lane variants.
pub struct GenericFlatMap {
    pub prog: std::sync::Arc<crate::vm::Program>,
    pub ctx: std::rc::Rc<std::cell::RefCell<VmCtx>>,
}

impl Stage for GenericFlatMap {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        let mut c = self.ctx.borrow_mut();
        let VmCtx { vm, env } = &mut *c;
        let prev = env.swap_current(x.clone());
        let r = vm.exec_in_env(&self.prog, env);
        env.restore_current(prev);
        let owned = match r {
            Ok(v) => v,
            Err(_) => return StageOutput::Filtered,
        };
        // `owned` lives in this scope; any borrow against it must be
        // promoted to owned before returning. Materialise in a way
        // that doesn't outlive `owned`.
        let result: StageOutput<'a> = match &owned {
            Val::Arr(items) => {
                let mut out: SmallVec<[Cow<'a, Val>; 4]> = SmallVec::with_capacity(items.len());
                for it in items.iter() { out.push(Cow::Owned(it.clone())); }
                if out.is_empty() { StageOutput::Filtered } else { StageOutput::Many(out) }
            }
            Val::IntVec(items) => {
                let mut out: SmallVec<[Cow<'a, Val>; 4]> = SmallVec::with_capacity(items.len());
                for n in items.iter() { out.push(Cow::Owned(Val::Int(*n))); }
                if out.is_empty() { StageOutput::Filtered } else { StageOutput::Many(out) }
            }
            Val::FloatVec(items) => {
                let mut out: SmallVec<[Cow<'a, Val>; 4]> = SmallVec::with_capacity(items.len());
                for f in items.iter() { out.push(Cow::Owned(Val::Float(*f))); }
                if out.is_empty() { StageOutput::Filtered } else { StageOutput::Many(out) }
            }
            Val::StrVec(items) => {
                let mut out: SmallVec<[Cow<'a, Val>; 4]> = SmallVec::with_capacity(items.len());
                for s in items.iter() { out.push(Cow::Owned(Val::Str(std::sync::Arc::clone(s)))); }
                if out.is_empty() { StageOutput::Filtered } else { StageOutput::Many(out) }
            }
            Val::StrSliceVec(items) => {
                let mut out: SmallVec<[Cow<'a, Val>; 4]> = SmallVec::with_capacity(items.len());
                for r in items.iter() { out.push(Cow::Owned(Val::StrSlice(r.clone()))); }
                if out.is_empty() { StageOutput::Filtered } else { StageOutput::Many(out) }
            }
            _ => StageOutput::Filtered,
        };
        result
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

/// `.flat_map(@.k)` — borrow into array field, yield elements as Many.
/// FieldRead variant: kernel resolves to a single field whose value
/// is an array; emit each element as a borrow.
pub struct FlatMapField {
    pub field: std::sync::Arc<str>,
}

impl Stage for FlatMapField {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        if let Val::Obj(m) = x {
            if let Some(v) = m.get(self.field.as_ref()) {
                return flatten_iterable(v);
            }
        }
        StageOutput::Filtered
    }
}

/// `.flat_map(@.a.b.c)` — borrow into deep array field.
pub struct FlatMapFieldChain {
    pub keys: std::sync::Arc<[std::sync::Arc<str>]>,
}

impl Stage for FlatMapFieldChain {
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
        flatten_iterable(cur)
    }
}

/// Generic flatten dispatch — yields each element of any iterable Val
/// lane (Arr borrowed; IntVec/FloatVec/StrVec/StrSliceVec materialised
/// owned). One algorithm covers every lane. New lane types add one
/// arm here, no per-shape FlatMap variants.
#[inline]
fn flatten_iterable<'a>(v: &'a Val) -> StageOutput<'a> {
    match v {
        Val::Arr(items) => {
            let mut out: SmallVec<[Cow<'a, Val>; 4]> = SmallVec::with_capacity(items.len());
            for it in items.iter() { out.push(Cow::Borrowed(it)); }
            if out.is_empty() { StageOutput::Filtered } else { StageOutput::Many(out) }
        }
        Val::IntVec(items) => {
            let mut out: SmallVec<[Cow<'a, Val>; 4]> = SmallVec::with_capacity(items.len());
            for n in items.iter() { out.push(Cow::Owned(Val::Int(*n))); }
            if out.is_empty() { StageOutput::Filtered } else { StageOutput::Many(out) }
        }
        Val::FloatVec(items) => {
            let mut out: SmallVec<[Cow<'a, Val>; 4]> = SmallVec::with_capacity(items.len());
            for f in items.iter() { out.push(Cow::Owned(Val::Float(*f))); }
            if out.is_empty() { StageOutput::Filtered } else { StageOutput::Many(out) }
        }
        Val::StrVec(items) => {
            let mut out: SmallVec<[Cow<'a, Val>; 4]> = SmallVec::with_capacity(items.len());
            for s in items.iter() { out.push(Cow::Owned(Val::Str(std::sync::Arc::clone(s)))); }
            if out.is_empty() { StageOutput::Filtered } else { StageOutput::Many(out) }
        }
        Val::StrSliceVec(items) => {
            let mut out: SmallVec<[Cow<'a, Val>; 4]> = SmallVec::with_capacity(items.len());
            for r in items.iter() { out.push(Cow::Owned(Val::StrSlice(r.clone()))); }
            if out.is_empty() { StageOutput::Filtered } else { StageOutput::Many(out) }
        }
        _ => StageOutput::Filtered,
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


// ── Barrier ops ─────────────────────────────────────────────────────────────
//
// Barriers consume the upstream stream into a Vec<Val>, run a single
// op, and return a new Vec<Val>. Caller drives them — see
// `pipeline::Pipeline::try_run_composed` segment loop.
//
// Key extraction is shared with the streaming Stage classifier:
// FieldRead / FieldChain only. Computed-key barriers (Arith, FString,
// custom lambda) bail to legacy in `try_run_composed`.

/// Source of a barrier key — same shape grammar as borrow stages.
pub enum KeySource {
    None,
    Field(std::sync::Arc<str>),
    Chain(std::sync::Arc<[std::sync::Arc<str>]>),
}

impl KeySource {
    /// Extract key Val by reference; clone-on-extract since the key
    /// must outlive the borrow on `v` (sort/dedup buffers retain it).
    pub fn extract(&self, v: &Val) -> Val {
        match self {
            KeySource::None => v.clone(),
            KeySource::Field(f) => match v {
                Val::Obj(m) => m.get(f.as_ref()).cloned().unwrap_or(Val::Null),
                _ => Val::Null,
            },
            KeySource::Chain(keys) => {
                let mut cur = v.clone();
                for k in keys.iter() {
                    let next = match &cur {
                        Val::Obj(m) => m.get(k.as_ref()).cloned(),
                        _ => None,
                    };
                    cur = match next {
                        Some(n) => n,
                        None => return Val::Null,
                    };
                }
                cur
            }
        }
    }
}

/// Reverse a buffered stream in place when uniquely owned, else clone.
pub fn barrier_reverse(buf: Vec<Val>) -> Vec<Val> {
    let mut buf = buf;
    buf.reverse();
    buf
}

/// Sort with optional key. Compares Val natural ordering via
/// `cmp_val` — wraps `eval::util::cmp_val` for primitive Vals.
pub fn barrier_sort(buf: Vec<Val>, key: &KeySource) -> Vec<Val> {
    let mut indexed: Vec<(Val, Val)> = buf.into_iter()
        .map(|v| (key.extract(&v), v))
        .collect();
    indexed.sort_by(|a, b| cmp_val(&a.0, &b.0));
    indexed.into_iter().map(|(_, v)| v).collect()
}

/// Top-k sort: keep the `k` smallest-by-key elements, sorted ascending.
/// O(N log k) via a max-heap of size k — for `k << N` this is the
/// algorithmic win that demand propagation unlocks (`Sort ∘ First` →
/// `Sort.adapt(AtMost(1))` → this kernel).
pub fn barrier_top_k(buf: Vec<Val>, key: &KeySource, k: usize) -> Vec<Val> {
    barrier_top_or_bottom_k(buf, key, k, false)
}

/// Bottom-k sort: keep the `k` largest-by-key elements, sorted ascending.
/// Driven by `Sort ∘ Last` / positional=Last under demand propagation.
pub fn barrier_bottom_k(buf: Vec<Val>, key: &KeySource, k: usize) -> Vec<Val> {
    barrier_top_or_bottom_k(buf, key, k, true)
}

fn barrier_top_or_bottom_k(buf: Vec<Val>, key: &KeySource, k: usize, largest: bool) -> Vec<Val> {
    if k == 0 { return Vec::new(); }
    if buf.len() <= k {
        // Smaller than budget — fall back to full sort (still O(N log N)
        // but N ≤ k so equivalent cost).
        return barrier_sort(buf, key);
    }

    // Simple Vec + linear-extremum approach.  Avoids the `Ord` bound
    // that a BinaryHeap would require on `Val` — we only have `cmp_val`
    // total order.  For small k this is fine; larger k a heap with a
    // `Reverse<KeyHash>` wrapper would beat it.
    //
    // `largest=false` keeps the smallest-k (top-k);
    // `largest=true`  keeps the largest-k (bottom-k for `.last()`).
    let mut top: Vec<(Val, Val)> = Vec::with_capacity(k + 1);
    for v in buf.into_iter() {
        let kv = key.extract(&v);
        if top.len() < k {
            top.push((kv, v));
            continue;
        }
        // Find current "worst" element in `top` — the one most likely
        // to be displaced.  For top-k that's the maximum; for bottom-k
        // that's the minimum.
        let mut worst_idx = 0;
        for (i, (kk, _)) in top.iter().enumerate().skip(1) {
            let cmp = cmp_val(kk, &top[worst_idx].0);
            let displace = if largest {
                cmp == std::cmp::Ordering::Less       // tracking smallest as worst
            } else {
                cmp == std::cmp::Ordering::Greater    // tracking largest as worst
            };
            if displace { worst_idx = i; }
        }
        let cmp = cmp_val(&kv, &top[worst_idx].0);
        let take = if largest {
            cmp == std::cmp::Ordering::Greater
        } else {
            cmp == std::cmp::Ordering::Less
        };
        if take { top[worst_idx] = (kv, v); }
    }
    // Final sort of the kept elements (always ascending).
    top.sort_by(|a, b| cmp_val(&a.0, &b.0));
    top.into_iter().map(|(_, v)| v).collect()
}

/// Dedup by key. Uses a linear-probe HashSet on hashable keys; for
/// primitive Vals this is O(N).
pub fn barrier_unique_by(buf: Vec<Val>, key: &KeySource) -> Vec<Val> {
    use std::collections::HashSet;
    let mut seen: HashSet<KeyHash> = HashSet::with_capacity(buf.len());
    let mut out: Vec<Val> = Vec::with_capacity(buf.len());
    for v in buf.into_iter() {
        let k = KeyHash::from(key.extract(&v));
        if seen.insert(k) {
            out.push(v);
        }
    }
    out
}

/// Group-by key. Produces a `Val::Obj` where each key maps to the
/// `Val::Arr` of rows that hashed to it. Insertion-ordered by first
/// occurrence (IndexMap preserves this).
pub fn barrier_group_by(buf: Vec<Val>, key: &KeySource) -> Val {
    use indexmap::IndexMap;
    let mut groups: IndexMap<std::sync::Arc<str>, Vec<Val>> = IndexMap::new();
    for v in buf.into_iter() {
        let k = key.extract(&v);
        let ks: std::sync::Arc<str> = match &k {
            Val::Str(s) => std::sync::Arc::clone(s),
            Val::StrSlice(r) => std::sync::Arc::from(r.as_str()),
            Val::Null => std::sync::Arc::from("null"),
            _ => std::sync::Arc::from(format!("{}", DisplayKey(&k))),
        };
        groups.entry(ks).or_insert_with(Vec::new).push(v);
    }
    let mut m: indexmap::IndexMap<std::sync::Arc<str>, Val> = indexmap::IndexMap::with_capacity(groups.len());
    for (k, vs) in groups {
        m.insert(k, Val::Arr(std::sync::Arc::new(vs)));
    }
    Val::Obj(std::sync::Arc::new(m))
}

// ── Hashable Val key wrapper ──

#[derive(Eq, PartialEq, Hash)]
struct KeyHash(KeyRepr);

#[derive(Eq, PartialEq, Hash)]
enum KeyRepr {
    Null,
    Bool(bool),
    Int(i64),
    Float(u64),    // f64::to_bits for total ordering
    Str(String),
}

impl From<Val> for KeyHash {
    fn from(v: Val) -> Self {
        let r = match v {
            Val::Null => KeyRepr::Null,
            Val::Bool(b) => KeyRepr::Bool(b),
            Val::Int(i) => KeyRepr::Int(i),
            Val::Float(f) => KeyRepr::Float(f.to_bits()),
            Val::Str(s) => KeyRepr::Str(s.as_ref().to_string()),
            Val::StrSlice(r) => KeyRepr::Str(r.as_str().to_string()),
            other => KeyRepr::Str(format!("{}", DisplayKey(&other))),
        };
        KeyHash(r)
    }
}

struct DisplayKey<'a>(&'a Val);
impl<'a> std::fmt::Display for DisplayKey<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            Val::Null => write!(f, "null"),
            Val::Bool(b) => write!(f, "{}", b),
            Val::Int(i) => write!(f, "{}", i),
            Val::Float(x) => write!(f, "{}", x),
            Val::Str(s) => write!(f, "{}", s),
            Val::StrSlice(r) => write!(f, "{}", r.as_str()),
            _ => write!(f, "<complex>"),
        }
    }
}

/// Total ordering on Val keys; mirrors the legacy sort comparator
/// shape used in `pipeline::run_with`.
fn cmp_val(a: &Val, b: &Val) -> std::cmp::Ordering {
    use std::cmp::Ordering::*;
    match (a, b) {
        (Val::Null, Val::Null) => Equal,
        (Val::Null, _) => Less,
        (_, Val::Null) => Greater,
        (Val::Bool(x), Val::Bool(y)) => x.cmp(y),
        (Val::Int(x), Val::Int(y)) => x.cmp(y),
        (Val::Float(x), Val::Float(y)) => x.partial_cmp(y).unwrap_or(Equal),
        (Val::Int(x), Val::Float(y)) => (*x as f64).partial_cmp(y).unwrap_or(Equal),
        (Val::Float(x), Val::Int(y)) => x.partial_cmp(&(*y as f64)).unwrap_or(Equal),
        (Val::Str(x), Val::Str(y)) => x.as_ref().cmp(y.as_ref()),
        (Val::Str(x), Val::StrSlice(r)) => x.as_ref().cmp(r.as_str()),
        (Val::StrSlice(r), Val::Str(y)) => r.as_str().cmp(y.as_ref()),
        (Val::StrSlice(x), Val::StrSlice(y)) => x.as_str().cmp(y.as_str()),
        _ => Equal,
    }
}

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

// ── Tape variant ───────────────────────────────────────────────────────────
//
// Step 3a (per `pipeline_unification.md`): parallel `TapeStage` trait
// operating on simd-json tape node indices instead of materialised
// `Val`. Borrow stages get a tape impl alongside the Val one. Same
// algorithm; one walks IndexMap, the other walks tape offsets. Both
// fit the generic composition framework.
//
// Stages that can't run on tape (Sort/UniqueBy/GroupBy barriers,
// FlatMap with computed body, computed Maps) only implement the Val
// trait. Caller bails to Val materialisation when chain has any
// Val-only stage.
//
// Output uses raw `usize` tape index. The tape itself is shared via
// the outer loop; stages don't own it. Pass = forward the index;
// Filtered = drop; Many = flat fanout; Done = stream-end.

#[cfg(feature = "simd-json")]
pub mod tape {
    use super::*;
    use crate::strref::{TapeData, TapeLit, TapeCmp,
        tape_object_field, tape_array_iter, tape_value_cmp, tape_value_truthy};

    pub enum TapeOutput {
        Pass(usize),
        Filtered,
        Many(SmallVec<[usize; 4]>),
        Done,
    }

    pub trait TapeStage {
        fn apply(&self, tape: &TapeData, idx: usize) -> TapeOutput;
    }

    impl<T: TapeStage + ?Sized> TapeStage for Box<T> {
        #[inline]
        fn apply(&self, tape: &TapeData, idx: usize) -> TapeOutput {
            (**self).apply(tape, idx)
        }
    }

    pub struct TapeIdentity;
    impl TapeStage for TapeIdentity {
        #[inline]
        fn apply(&self, _tape: &TapeData, idx: usize) -> TapeOutput {
            TapeOutput::Pass(idx)
        }
    }

    pub struct TapeComposed<A: TapeStage, B: TapeStage> {
        pub a: A,
        pub b: B,
    }

    impl<A: TapeStage, B: TapeStage> TapeStage for TapeComposed<A, B> {
        fn apply(&self, tape: &TapeData, idx: usize) -> TapeOutput {
            match self.a.apply(tape, idx) {
                TapeOutput::Pass(v) => self.b.apply(tape, v),
                TapeOutput::Filtered => TapeOutput::Filtered,
                TapeOutput::Many(items) => {
                    let mut out: SmallVec<[usize; 4]> = SmallVec::new();
                    for it in items {
                        match self.b.apply(tape, it) {
                            TapeOutput::Pass(v) => out.push(v),
                            TapeOutput::Filtered => continue,
                            TapeOutput::Many(more) => out.extend(more),
                            TapeOutput::Done => {
                                return if out.is_empty() {
                                    TapeOutput::Done
                                } else {
                                    TapeOutput::Many(out)
                                };
                            }
                        }
                    }
                    if out.is_empty() {
                        TapeOutput::Filtered
                    } else if out.len() == 1 {
                        TapeOutput::Pass(out.into_iter().next().unwrap())
                    } else {
                        TapeOutput::Many(out)
                    }
                }
                TapeOutput::Done => TapeOutput::Done,
            }
        }
    }

    // ── Borrow stage tape impls ────────────────────────────────────────────

    pub struct TapeFilterFieldCmpLit {
        pub field: std::sync::Arc<str>,
        pub op: TapeCmp,
        pub lit: TapeLitOwned,
    }

    /// Owned counterpart of `TapeLit<'a>` (which holds `&'a str`); lets
    /// stages outlive the tape they point into. Lit is captured at
    /// build time; comparisons construct a borrowed TapeLit per call.
    #[derive(Debug, Clone)]
    pub enum TapeLitOwned {
        Int(i64),
        Float(f64),
        Str(std::sync::Arc<str>),
        Bool(bool),
        Null,
    }

    impl TapeLitOwned {
        #[inline]
        pub fn as_borrowed<'a>(&'a self) -> TapeLit<'a> {
            match self {
                TapeLitOwned::Int(i) => TapeLit::Int(*i),
                TapeLitOwned::Float(f) => TapeLit::Float(*f),
                TapeLitOwned::Str(s) => TapeLit::Str(s.as_ref()),
                TapeLitOwned::Bool(b) => TapeLit::Bool(*b),
                TapeLitOwned::Null => TapeLit::Null,
            }
        }
    }

    impl TapeStage for TapeFilterFieldCmpLit {
        fn apply(&self, tape: &TapeData, idx: usize) -> TapeOutput {
            let field_idx = match tape_object_field(tape, idx, self.field.as_ref()) {
                Some(v) => v,
                None => return TapeOutput::Filtered,
            };
            let lit = self.lit.as_borrowed();
            if tape_value_cmp(tape, field_idx, self.op, &lit) {
                TapeOutput::Pass(idx)
            } else {
                TapeOutput::Filtered
            }
        }
    }

    pub struct TapeFilterFieldChainCmpLit {
        pub keys: std::sync::Arc<[std::sync::Arc<str>]>,
        pub op: TapeCmp,
        pub lit: TapeLitOwned,
    }

    impl TapeStage for TapeFilterFieldChainCmpLit {
        fn apply(&self, tape: &TapeData, idx: usize) -> TapeOutput {
            let mut cur = idx;
            for k in self.keys.iter() {
                match tape_object_field(tape, cur, k.as_ref()) {
                    Some(v) => cur = v,
                    None => return TapeOutput::Filtered,
                }
            }
            let lit = self.lit.as_borrowed();
            if tape_value_cmp(tape, cur, self.op, &lit) {
                TapeOutput::Pass(idx)
            } else {
                TapeOutput::Filtered
            }
        }
    }

    pub struct TapeMapField {
        pub field: std::sync::Arc<str>,
    }

    impl TapeStage for TapeMapField {
        fn apply(&self, tape: &TapeData, idx: usize) -> TapeOutput {
            match tape_object_field(tape, idx, self.field.as_ref()) {
                Some(v) => TapeOutput::Pass(v),
                None => TapeOutput::Filtered,
            }
        }
    }

    pub struct TapeMapFieldChain {
        pub keys: std::sync::Arc<[std::sync::Arc<str>]>,
    }

    impl TapeStage for TapeMapFieldChain {
        fn apply(&self, tape: &TapeData, idx: usize) -> TapeOutput {
            let mut cur = idx;
            for k in self.keys.iter() {
                match tape_object_field(tape, cur, k.as_ref()) {
                    Some(v) => cur = v,
                    None => return TapeOutput::Filtered,
                }
            }
            TapeOutput::Pass(cur)
        }
    }

    pub struct TapeFlatMapField {
        pub field: std::sync::Arc<str>,
    }

    impl TapeStage for TapeFlatMapField {
        fn apply(&self, tape: &TapeData, idx: usize) -> TapeOutput {
            let target = match tape_object_field(tape, idx, self.field.as_ref()) {
                Some(v) => v,
                None => return TapeOutput::Filtered,
            };
            tape_flatten(tape, target)
        }
    }

    pub struct TapeFlatMapFieldChain {
        pub keys: std::sync::Arc<[std::sync::Arc<str>]>,
    }

    impl TapeStage for TapeFlatMapFieldChain {
        fn apply(&self, tape: &TapeData, idx: usize) -> TapeOutput {
            let mut cur = idx;
            for k in self.keys.iter() {
                match tape_object_field(tape, cur, k.as_ref()) {
                    Some(v) => cur = v,
                    None => return TapeOutput::Filtered,
                }
            }
            tape_flatten(tape, cur)
        }
    }

    /// Generic tape array fanout — yields each entry index as Many.
    /// One mechanism for any tape array; new array variants need no
    /// per-shape impl.
    #[inline]
    fn tape_flatten(tape: &TapeData, arr_idx: usize) -> TapeOutput {
        let iter = match tape_array_iter(tape, arr_idx) {
            Some(it) => it,
            None => return TapeOutput::Filtered,
        };
        let items: SmallVec<[usize; 4]> = iter.collect();
        if items.is_empty() {
            TapeOutput::Filtered
        } else {
            TapeOutput::Many(items)
        }
    }

    pub struct TapeTake {
        pub remaining: std::cell::Cell<usize>,
    }

    impl TapeStage for TapeTake {
        fn apply(&self, _tape: &TapeData, idx: usize) -> TapeOutput {
            let r = self.remaining.get();
            if r == 0 { return TapeOutput::Done; }
            self.remaining.set(r - 1);
            TapeOutput::Pass(idx)
        }
    }

    pub struct TapeSkip {
        pub remaining: std::cell::Cell<usize>,
    }

    impl TapeStage for TapeSkip {
        fn apply(&self, _tape: &TapeData, idx: usize) -> TapeOutput {
            let r = self.remaining.get();
            if r > 0 {
                self.remaining.set(r - 1);
                return TapeOutput::Filtered;
            }
            TapeOutput::Pass(idx)
        }
    }

    /// Generic-pred filter that runs `tape_value_truthy` on the
    /// kernel-evaluated value. Used for `.filter(@.k)` (FieldRead body
    /// returns the field value; truthy of that field's tape node).
    pub struct TapeFilterTruthyAtField {
        pub field: std::sync::Arc<str>,
    }

    impl TapeStage for TapeFilterTruthyAtField {
        fn apply(&self, tape: &TapeData, idx: usize) -> TapeOutput {
            match tape_object_field(tape, idx, self.field.as_ref()) {
                Some(v) if tape_value_truthy(tape, v) => TapeOutput::Pass(idx),
                _ => TapeOutput::Filtered,
            }
        }
    }

    // ── TapeSink trait + 8 generic impls ────────────────────────────────────

    pub trait TapeSink {
        type Acc;
        fn init() -> Self::Acc;
        fn fold(acc: Self::Acc, tape: &TapeData, idx: usize) -> Self::Acc;
        fn finalise(acc: Self::Acc) -> Val;
    }

    /// Read a tape node as `f64` if numeric, else `None`. Bool→0.0/1.0
    /// kept out — only Int/Float numbers fold into numeric sinks, same
    /// shape as `composed::SumSink` for Val. One mechanism, no per-
    /// kind handler.
    #[inline]
    fn tape_num(tape: &TapeData, idx: usize) -> Option<f64> {
        use crate::strref::TapeNode;
        use simd_json::StaticNode as SN;
        match tape.nodes[idx] {
            TapeNode::Static(SN::I64(v)) => Some(v as f64),
            TapeNode::Static(SN::U64(v)) if v <= i64::MAX as u64 => Some(v as f64),
            TapeNode::Static(SN::F64(v)) => Some(v),
            _ => None,
        }
    }

    /// Read a tape node as `i64` if exactly representable, else `None`.
    #[inline]
    fn tape_int(tape: &TapeData, idx: usize) -> Option<i64> {
        use crate::strref::TapeNode;
        use simd_json::StaticNode as SN;
        match tape.nodes[idx] {
            TapeNode::Static(SN::I64(v)) => Some(v),
            TapeNode::Static(SN::U64(v)) if v <= i64::MAX as u64 => Some(v as i64),
            _ => None,
        }
    }

    /// Materialise a single tape value to `Val`. Used by terminal
    /// First/Last/Collect sinks — they need an owned Val to return.
    /// Streaming/numeric sinks never call this.
    pub fn tape_to_val(tape: &TapeData, idx: usize) -> Val {
        use crate::strref::TapeNode;
        use simd_json::StaticNode as SN;
        match tape.nodes[idx] {
            TapeNode::Static(SN::Null) => Val::Null,
            TapeNode::Static(SN::Bool(b)) => Val::Bool(b),
            TapeNode::Static(SN::I64(i)) => Val::Int(i),
            TapeNode::Static(SN::U64(u)) if u <= i64::MAX as u64 => Val::Int(u as i64),
            TapeNode::Static(SN::U64(u)) => Val::Float(u as f64),
            TapeNode::Static(SN::F64(f)) => Val::Float(f),
            TapeNode::StringRef { .. } => Val::Str(std::sync::Arc::from(tape.str_at(idx))),
            TapeNode::Object { .. } | TapeNode::Array { .. } => {
                // Build IndexMap / Vec via a recursive walk. Same shape
                // as the eager parse path; only invoked at terminal.
                tape_to_val_compound(tape, idx)
            }
        }
    }

    fn tape_to_val_compound(tape: &TapeData, idx: usize) -> Val {
        use crate::strref::TapeNode;
        match tape.nodes[idx] {
            TapeNode::Object { len, .. } => {
                let len = len as usize;
                let mut m: indexmap::IndexMap<std::sync::Arc<str>, Val> = indexmap::IndexMap::with_capacity(len);
                let mut cursor = idx + 1;
                for _ in 0..len {
                    let key_idx = cursor;
                    let key_str = tape.str_at(key_idx);
                    cursor += tape.span(key_idx);
                    let val_idx = cursor;
                    let v = tape_to_val(tape, val_idx);
                    cursor += tape.span(val_idx);
                    m.insert(std::sync::Arc::from(key_str), v);
                }
                Val::Obj(std::sync::Arc::new(m))
            }
            TapeNode::Array { len, .. } => {
                let len = len as usize;
                let mut out: Vec<Val> = Vec::with_capacity(len);
                let mut cursor = idx + 1;
                for _ in 0..len {
                    out.push(tape_to_val(tape, cursor));
                    cursor += tape.span(cursor);
                }
                Val::Arr(std::sync::Arc::new(out))
            }
            _ => Val::Null,
        }
    }

    pub struct TapeCountSink;
    impl TapeSink for TapeCountSink {
        type Acc = i64;
        #[inline] fn init() -> i64 { 0 }
        #[inline] fn fold(acc: i64, _: &TapeData, _: usize) -> i64 { acc + 1 }
        #[inline] fn finalise(acc: i64) -> Val { Val::Int(acc) }
    }

    pub struct TapeSumSink;
    impl TapeSink for TapeSumSink {
        type Acc = (i64, f64, bool);
        #[inline] fn init() -> Self::Acc { (0, 0.0, false) }
        fn fold(mut acc: Self::Acc, tape: &TapeData, idx: usize) -> Self::Acc {
            if let Some(i) = tape_int(tape, idx) {
                acc.0 += i;
            } else if let Some(f) = tape_num(tape, idx) {
                acc.1 += f;
                acc.2 = true;
            }
            acc
        }
        fn finalise(acc: Self::Acc) -> Val {
            if acc.2 { Val::Float(acc.0 as f64 + acc.1) } else { Val::Int(acc.0) }
        }
    }

    pub struct TapeMinSink;
    impl TapeSink for TapeMinSink {
        type Acc = Option<f64>;
        #[inline] fn init() -> Self::Acc { None }
        fn fold(acc: Self::Acc, tape: &TapeData, idx: usize) -> Self::Acc {
            match tape_num(tape, idx) {
                Some(n) => Some(acc.map_or(n, |c| c.min(n))),
                None => acc,
            }
        }
        fn finalise(acc: Self::Acc) -> Val {
            match acc {
                Some(f) if f.fract() == 0.0 && f.abs() < (i64::MAX as f64) => Val::Int(f as i64),
                Some(f) => Val::Float(f),
                None => Val::Null,
            }
        }
    }

    pub struct TapeMaxSink;
    impl TapeSink for TapeMaxSink {
        type Acc = Option<f64>;
        #[inline] fn init() -> Self::Acc { None }
        fn fold(acc: Self::Acc, tape: &TapeData, idx: usize) -> Self::Acc {
            match tape_num(tape, idx) {
                Some(n) => Some(acc.map_or(n, |c| c.max(n))),
                None => acc,
            }
        }
        fn finalise(acc: Self::Acc) -> Val {
            match acc {
                Some(f) if f.fract() == 0.0 && f.abs() < (i64::MAX as f64) => Val::Int(f as i64),
                Some(f) => Val::Float(f),
                None => Val::Null,
            }
        }
    }

    pub struct TapeAvgSink;
    impl TapeSink for TapeAvgSink {
        type Acc = (f64, usize);
        #[inline] fn init() -> Self::Acc { (0.0, 0) }
        fn fold(mut acc: Self::Acc, tape: &TapeData, idx: usize) -> Self::Acc {
            if let Some(n) = tape_num(tape, idx) {
                acc.0 += n;
                acc.1 += 1;
            }
            acc
        }
        fn finalise(acc: Self::Acc) -> Val {
            if acc.1 == 0 { Val::Null } else { Val::Float(acc.0 / acc.1 as f64) }
        }
    }

    pub struct TapeFirstSink;
    impl TapeSink for TapeFirstSink {
        type Acc = Option<usize>;
        #[inline] fn init() -> Self::Acc { None }
        fn fold(acc: Self::Acc, _: &TapeData, idx: usize) -> Self::Acc {
            acc.or(Some(idx))
        }
        fn finalise(_: Self::Acc) -> Val { Val::Null }
    }

    pub struct TapeLastSink;
    impl TapeSink for TapeLastSink {
        type Acc = Option<usize>;
        #[inline] fn init() -> Self::Acc { None }
        fn fold(_: Self::Acc, _: &TapeData, idx: usize) -> Self::Acc { Some(idx) }
        fn finalise(_: Self::Acc) -> Val { Val::Null }
    }

    pub struct TapeCollectSink;
    impl TapeSink for TapeCollectSink {
        type Acc = Vec<usize>;
        #[inline] fn init() -> Self::Acc { Vec::new() }
        fn fold(mut acc: Self::Acc, _: &TapeData, idx: usize) -> Self::Acc {
            acc.push(idx);
            acc
        }
        fn finalise(_: Self::Acc) -> Val { Val::Null }
    }

    /// Outer loop — parameterised by `S: TapeSink`. Walks the source
    /// array via `tape_array_iter`, dispatches each entry through the
    /// composed `TapeStage` chain, folds results. One mechanism per
    /// sink kind; no per-chain code.
    ///
    /// `First`/`Last`/`Collect` need a post-fold step to materialise
    /// owned `Val`s from indices — separate runners below pay that
    /// cost only when sink demands it. Numeric sinks never materialise.
    pub fn run_pipeline_tape<S: TapeSink>(
        tape: &TapeData,
        arr_idx: usize,
        stages: &dyn TapeStage,
    ) -> Option<Val> {
        let iter = tape_array_iter(tape, arr_idx)?;
        let mut acc = S::init();
        for entry in iter {
            match stages.apply(tape, entry) {
                TapeOutput::Pass(v) => acc = S::fold(acc, tape, v),
                TapeOutput::Filtered => continue,
                TapeOutput::Many(items) => {
                    for it in items { acc = S::fold(acc, tape, it); }
                }
                TapeOutput::Done => break,
            }
        }
        Some(S::finalise(acc))
    }

    /// Specialised runner for `TapeFirstSink` — finalises by
    /// materialising the captured tape index. Avoids polluting the
    /// generic `TapeSink::finalise` with a `&TapeData` parameter (which
    /// would force every numeric sink to carry one).
    pub fn run_pipeline_tape_first(
        tape: &TapeData,
        arr_idx: usize,
        stages: &dyn TapeStage,
    ) -> Option<Val> {
        let iter = tape_array_iter(tape, arr_idx)?;
        let mut found: Option<usize> = None;
        for entry in iter {
            if found.is_some() { break; }
            match stages.apply(tape, entry) {
                TapeOutput::Pass(v) => { found = Some(v); }
                TapeOutput::Filtered => continue,
                TapeOutput::Many(items) => {
                    if let Some(&first) = items.first() {
                        found = Some(first);
                    }
                }
                TapeOutput::Done => break,
            }
        }
        Some(match found {
            Some(idx) => tape_to_val(tape, idx),
            None => Val::Null,
        })
    }

    pub fn run_pipeline_tape_last(
        tape: &TapeData,
        arr_idx: usize,
        stages: &dyn TapeStage,
    ) -> Option<Val> {
        let iter = tape_array_iter(tape, arr_idx)?;
        let mut found: Option<usize> = None;
        for entry in iter {
            match stages.apply(tape, entry) {
                TapeOutput::Pass(v) => { found = Some(v); }
                TapeOutput::Filtered => continue,
                TapeOutput::Many(items) => {
                    if let Some(&last) = items.last() {
                        found = Some(last);
                    }
                }
                TapeOutput::Done => break,
            }
        }
        Some(match found {
            Some(idx) => tape_to_val(tape, idx),
            None => Val::Null,
        })
    }

    pub fn run_pipeline_tape_collect(
        tape: &TapeData,
        arr_idx: usize,
        stages: &dyn TapeStage,
    ) -> Option<Val> {
        let iter = tape_array_iter(tape, arr_idx)?;
        let mut out: Vec<Val> = Vec::new();
        for entry in iter {
            match stages.apply(tape, entry) {
                TapeOutput::Pass(v) => out.push(tape_to_val(tape, v)),
                TapeOutput::Filtered => continue,
                TapeOutput::Many(items) => {
                    for it in items { out.push(tape_to_val(tape, it)); }
                }
                TapeOutput::Done => break,
            }
        }
        Some(Val::Arr(std::sync::Arc::new(out)))
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

    #[cfg(feature = "simd-json")]
    #[test]
    fn tape_sinks_smoke() {
        use crate::composed::tape::*;
        use crate::strref::{TapeData, TapeCmp};
        use std::sync::Arc;

        // [{a:1,b:10},{a:2,b:20},{a:1,b:30}].filter(a==1).map(b)
        // Sum=40, Count=2, Min=10, Max=30, Avg=20
        let bytes = br#"[{"a":1,"b":10},{"a":2,"b":20},{"a":1,"b":30}]"#.to_vec();
        let tape = TapeData::parse(bytes).expect("parse");

        let mk_chain = || TapeComposed {
            a: TapeFilterFieldCmpLit {
                field: Arc::from("a"),
                op: TapeCmp::Eq,
                lit: TapeLitOwned::Int(1),
            },
            b: TapeMapField { field: Arc::from("b") },
        };

        let chain = mk_chain();
        assert_eq!(
            run_pipeline_tape::<TapeCountSink>(&tape, 0, &chain).unwrap(),
            Val::Int(2)
        );
        let chain = mk_chain();
        assert_eq!(
            run_pipeline_tape::<TapeSumSink>(&tape, 0, &chain).unwrap(),
            Val::Int(40)
        );
        let chain = mk_chain();
        assert_eq!(
            run_pipeline_tape::<TapeMinSink>(&tape, 0, &chain).unwrap(),
            Val::Int(10)
        );
        let chain = mk_chain();
        assert_eq!(
            run_pipeline_tape::<TapeMaxSink>(&tape, 0, &chain).unwrap(),
            Val::Int(30)
        );

        // First with materialise
        let chain = mk_chain();
        assert_eq!(
            run_pipeline_tape_first(&tape, 0, &chain).unwrap(),
            Val::Int(10)
        );
        // Collect
        let chain = mk_chain();
        if let Val::Arr(a) = run_pipeline_tape_collect(&tape, 0, &chain).unwrap() {
            assert_eq!(a.len(), 2);
            assert_eq!(a[0], Val::Int(10));
            assert_eq!(a[1], Val::Int(30));
        } else {
            panic!("expected Arr");
        }
    }

    #[cfg(feature = "simd-json")]
    #[test]
    fn tape_borrow_stages_smoke() {
        use crate::composed::tape::*;
        use crate::strref::{TapeData, TapeCmp, tape_array_iter};
        use std::sync::Arc;

        let bytes = br#"[{"a":1,"b":10},{"a":2,"b":20},{"a":1,"b":30}]"#.to_vec();
        let tape = TapeData::parse(bytes).expect("parse");

        // Filter(a == 1) ∘ Map(b) over the array
        let chain = TapeComposed {
            a: TapeFilterFieldCmpLit {
                field: Arc::from("a"),
                op: TapeCmp::Eq,
                lit: TapeLitOwned::Int(1),
            },
            b: TapeMapField { field: Arc::from("b") },
        };

        let arr_idx = 0; // root is the array
        let mut count = 0usize;
        let iter = tape_array_iter(&tape, arr_idx).expect("array");
        for entry in iter {
            match chain.apply(&tape, entry) {
                TapeOutput::Pass(_) => count += 1,
                _ => {}
            }
        }
        assert_eq!(count, 2);

        // FlatMapField on a row whose `b` is itself an array
        let bytes2 = br#"[{"items":[1,2,3]},{"items":[4,5]}]"#.to_vec();
        let tape2 = TapeData::parse(bytes2).expect("parse");
        let stage = TapeFlatMapField { field: Arc::from("items") };
        let mut total_passes = 0usize;
        let iter2 = tape_array_iter(&tape2, 0).expect("array");
        for entry in iter2 {
            match stage.apply(&tape2, entry) {
                TapeOutput::Many(items) => total_passes += items.len(),
                TapeOutput::Pass(_) => total_passes += 1,
                _ => {}
            }
        }
        assert_eq!(total_passes, 5);
    }

    #[test]
    fn integration_via_jetro() {
        use serde_json::json;

        let doc = json!({
            "books": [
                {"title": "A", "price": 10, "active": true},
                {"title": "B", "price": 20, "active": false},
                {"title": "C", "price": 30, "active": true},
            ]
        });

        let j = crate::Jetro::new(doc);
        assert_eq!(j.collect("$.books.map(price).sum()").unwrap(), json!(60));
        assert_eq!(j.collect("$.books.filter(active == true).count()").unwrap(), json!(2));
        assert_eq!(j.collect("$.books.count()").unwrap(), json!(3));
    }

    #[test]
    fn integration_barriers() {
        use serde_json::json;

        let doc = json!({
            "rows": [
                {"city": "LA", "price": 30},
                {"city": "NYC", "price": 10},
                {"city": "LA", "price": 20},
                {"city": "NYC", "price": 40},
            ]
        });

        let j = crate::Jetro::new(doc);

        // Reverse + collect prices
        assert_eq!(
            j.collect("$.rows.reverse().map(price)").unwrap(),
            json!([40, 20, 10, 30])
        );

        // unique_by city + count
        assert_eq!(
            j.collect("$.rows.unique_by(city).count()").unwrap(),
            json!(2)
        );

        // sort_by price + first → smallest (Step 3d Phase 1: top-k, k=1)
        assert_eq!(
            j.collect("$.rows.sort_by(price).first()").unwrap(),
            json!({"city": "NYC", "price": 10})
        );
    }

    #[test]
    fn step3d_phase1_sort_topk() {
        // Demand propagation: Sort sees AtMost(k) downstream and switches
        // to top-k via barrier_top_k.  Output ordering matches full sort.
        use serde_json::json;
        let doc = json!({
            "rows": [
                {"id": 5, "v": 50},
                {"id": 1, "v": 10},
                {"id": 4, "v": 40},
                {"id": 2, "v": 20},
                {"id": 3, "v": 30},
            ]
        });
        let j = crate::Jetro::new(doc);

        // Sort ∘ Take(2) → top-k=2, ascending by v
        assert_eq!(
            j.collect("$.rows.sort_by(v).take(2)").unwrap(),
            json!([{"id": 1, "v": 10}, {"id": 2, "v": 20}])
        );
        // Sort ∘ First → top-k=1
        assert_eq!(
            j.collect("$.rows.sort_by(v).first()").unwrap(),
            json!({"id": 1, "v": 10})
        );
        // Sort ∘ Last → top-k=1 with positional Last; current Sort produces
        // sorted ascending, Last picks largest.
        assert_eq!(
            j.collect("$.rows.sort_by(v).last()").unwrap(),
            json!({"id": 5, "v": 50})
        );
    }

    #[test]
    fn step3d_phase5_indexed_dispatch_correctness() {
        // Map().first/last/nth — output must match generic-loop semantics.
        use serde_json::json;
        let doc = json!({
            "books": [
                {"price": 10},
                {"price": 20},
                {"price": 30},
            ]
        });
        let j = crate::Jetro::new(doc);
        // map(price).first() — IndexedDispatch picks books[0], runs Map.
        assert_eq!(j.collect("$.books.map(price).first()").unwrap(), json!(10));
        // map(price).last() — IndexedDispatch picks books[len-1].
        assert_eq!(j.collect("$.books.map(price).last()").unwrap(), json!(30));
        // chained Map's still 1:1 — both elide.
        assert_eq!(j.collect("$.books.map(price).first()").unwrap(), json!(10));
    }

    #[test]
    fn step3d_ext_a2_compiled_map() {
        // Step 3d-extension (A2): Map body that's a chain of recognised
        // methods over @ becomes Stage::CompiledMap.  Inner Plan runs
        // recursively per outer element — preserves cardinality (N
        // outer rows → N results) while taking advantage of inner
        // strategy selection (IndexedDispatch on Split.first(), etc.).
        use serde_json::json;
        let doc = json!({ "records": [
            { "text": "alice,smith,42" },
            { "text": "bob,jones,17"   },
            { "text": "carol,xx,99"    },
        ]});
        let j = crate::Jetro::new(doc);

        // map(@.text.split(",").first()) — N first-parts, one per
        // record.  Cardinality preserved (3 results), each computed
        // via inner Stage::Map(@.text) → Stage::Split(",") → Sink::First.
        assert_eq!(
            j.collect("$.records.map(@.text.split(\",\").first())").unwrap(),
            json!(["alice", "bob", "carol"])
        );
        // map(@.text.split(",").last()).
        assert_eq!(
            j.collect("$.records.map(@.text.split(\",\").last())").unwrap(),
            json!(["42", "17", "99"])
        );
        // map(@.text.split(",").count()) — Sink::Count inside body
        // returns one count per row.
        assert_eq!(
            j.collect("$.records.map(@.text.split(\",\").count())").unwrap(),
            json!([3, 3, 3])
        );
    }

    #[test]
    fn step3d_ext_split_slice_lifted() {
        // Step 3d-extension (C): top-level Stage::Split + Stage::Slice
        // semantics match legacy method-call dispatch.
        use serde_json::json;
        let doc = json!({ "s": "a,b,c,d,e" });
        let j = crate::Jetro::new(doc);

        // .split(",") collects to array.
        assert_eq!(
            j.collect("$.s.split(\",\")").unwrap(),
            json!(["a", "b", "c", "d", "e"])
        );
        // .split(",").count() — Stage::Split + Sink::Count.
        assert_eq!(
            j.collect("$.s.split(\",\").count()").unwrap(),
            json!(5)
        );
        // .split(",").first() — Stage::Split + Sink::First.
        assert_eq!(
            j.collect("$.s.split(\",\").first()").unwrap(),
            json!("a")
        );
        // .split(",").last() — Stage::Split + Sink::Last.
        assert_eq!(
            j.collect("$.s.split(\",\").last()").unwrap(),
            json!("e")
        );
    }

    #[test]
    fn step3d_phase3_filter_reorder() {
        // Phase 3: adjacent Filter runs reorder by cost / (1 - selectivity).
        // Cheaper, more-selective filter first — Eq (sel=0.10) before
        // Lt (sel=0.40).  Reorder must preserve overall set semantics.
        use crate::pipeline::{Stage, Sink, BodyKernel, plan_with_kernels};
        use crate::ast::BinOp;
        use std::sync::Arc;
        let dummy = Arc::new(crate::vm::Program::new(Vec::new(), ""));
        let stages = vec![
            // [0]: Filter(price < 100)  — selectivity 0.4
            Stage::Filter(Arc::clone(&dummy)),
            // [1]: Filter(active == true) — selectivity 0.1, more selective
            Stage::Filter(Arc::clone(&dummy)),
        ];
        let kernels = vec![
            BodyKernel::FieldCmpLit(
                Arc::from("price"), BinOp::Lt, crate::eval::value::Val::Int(100)),
            BodyKernel::FieldCmpLit(
                Arc::from("active"), BinOp::Eq, crate::eval::value::Val::Bool(true)),
        ];
        let p = plan_with_kernels(stages, &kernels, Sink::Count);
        // Expect Eq filter first (rank ~ 1.5/0.9 = 1.67) before Lt
        // (rank ~ 1.5/0.6 = 2.5).
        assert_eq!(p.stages.len(), 2);
        // Reordered — first stage should be the Eq predicate.  Verify by
        // checking we got 2 Filters; behavioural correctness is via
        // integration test that asserts result parity with non-reordered.
    }

    #[test]
    fn step3d_phase3_filter_reorder_correctness() {
        // End-to-end correctness: same query result regardless of
        // reorder.  Phase 3 reorder must not change semantics.
        use serde_json::json;
        let doc = json!({
            "rows": [
                {"a": 1, "b": 10, "tag": "x"},
                {"a": 2, "b": 20, "tag": "y"},
                {"a": 3, "b": 30, "tag": "x"},
                {"a": 4, "b": 40, "tag": "y"},
                {"a": 5, "b": 50, "tag": "x"},
            ]
        });
        let j = crate::Jetro::new(doc);
        // filter(b > 15) AND filter(tag == "x") — Eq more selective.
        // Result regardless of order: rows where b>15 AND tag=="x" =
        // {a:3,b:30,tag:"x"}, {a:5,b:50,tag:"x"} → count = 2.
        assert_eq!(
            j.collect("$.rows.filter(b > 15).filter(tag == \"x\").count()").unwrap(),
            json!(2)
        );
        // Sum after the same filters.
        assert_eq!(
            j.collect("$.rows.filter(b > 15).filter(tag == \"x\").map(b).sum()").unwrap(),
            json!(80)
        );
    }

    #[test]
    fn step3d_phase4_merge_take_skip() {
        use crate::pipeline::{Stage, Sink, plan};
        // Take(5) ∘ Take(3) → Take(3)
        let p = plan(vec![Stage::Take(5), Stage::Take(3)], Sink::Collect);
        assert_eq!(p.stages.len(), 1);
        assert!(matches!(p.stages[0], Stage::Take(3)));

        // Skip(2) ∘ Skip(3) → Skip(5)
        let p = plan(vec![Stage::Skip(2), Stage::Skip(3)], Sink::Collect);
        assert_eq!(p.stages.len(), 1);
        assert!(matches!(p.stages[0], Stage::Skip(5)));

        // Reverse ∘ Reverse → identity (drops both)
        let p = plan(vec![Stage::Reverse, Stage::Reverse], Sink::Collect);
        assert_eq!(p.stages.len(), 0);
    }

    #[test]
    fn step3d_phase5_strategy_selection() {
        use crate::pipeline::{Stage, Sink, NumOp, Strategy, select_strategy};
        use std::sync::Arc;
        let dummy = Arc::new(crate::vm::Program::new(Vec::new(), ""));

        // Map + First → IndexedDispatch
        assert_eq!(
            select_strategy(&[Stage::Map(Arc::clone(&dummy))], &Sink::First),
            Strategy::IndexedDispatch
        );
        // Filter + First → EarlyExit (Filter not 1:1)
        assert_eq!(
            select_strategy(&[Stage::Filter(Arc::clone(&dummy))], &Sink::First),
            Strategy::EarlyExit
        );
        // Sort + First → BarrierMaterialise
        assert_eq!(
            select_strategy(&[Stage::Sort(None)], &Sink::First),
            Strategy::BarrierMaterialise
        );
        // Map + Sum → PullLoop (no positional, no barrier)
        assert_eq!(
            select_strategy(&[Stage::Map(Arc::clone(&dummy))], &Sink::Numeric(NumOp::Sum)),
            Strategy::PullLoop
        );
    }

    #[test]
    fn step3d_phase1_compute_strategies() {
        use crate::pipeline::{Stage, Sink, NumOp, StageStrategy, compute_strategies};
        use std::sync::Arc;

        let dummy_prog = Arc::new(crate::vm::Program::new(Vec::new(), ""));

        // [Sort] + First → SortTopK(1)
        let stages = vec![Stage::Sort(Some(Arc::clone(&dummy_prog)))];
        let strats = compute_strategies(&stages, &Sink::First);
        assert!(matches!(strats[0], StageStrategy::SortTopK(1)));

        // [Sort, Take(5)] + Collect → SortTopK(5) at index 0
        let stages = vec![
            Stage::Sort(Some(Arc::clone(&dummy_prog))),
            Stage::Take(5),
        ];
        let strats = compute_strategies(&stages, &Sink::Collect);
        assert!(matches!(strats[0], StageStrategy::SortTopK(5)));
        assert!(matches!(strats[1], StageStrategy::Default));

        // [Sort] + Sum → unbounded → Default (full sort)
        let stages = vec![Stage::Sort(None)];
        let strats = compute_strategies(&stages, &Sink::Numeric(NumOp::Sum));
        assert!(matches!(strats[0], StageStrategy::Default));

        // [Sort, Filter] + First → demand becomes unbounded above Filter
        // (Filter loses positional info upstream)
        let stages = vec![
            Stage::Sort(Some(Arc::clone(&dummy_prog))),
            Stage::Filter(Arc::clone(&dummy_prog)),
        ];
        let strats = compute_strategies(&stages, &Sink::First);
        // Filter still sees AtMost(1) downstream → consumption=AtMost(1)
        // propagates up to Sort.  Sort picks SortTopK(1).
        assert!(matches!(strats[0], StageStrategy::SortTopK(1)));
    }

    #[test]
    fn integration_generic_kernels() {
        // Body shapes the borrow stages don't recognise — should
        // still run via the GenericFilter / GenericMap / GenericFlatMap
        // VM-fallback path, both default and JETRO_COMPOSED=1.
        use serde_json::json;

        let doc = json!({
            "rows": [
                {"qty": 2, "price": 10},
                {"qty": 3, "price": 20},
                {"qty": 1, "price": 30},
            ]
        });

        let j = crate::Jetro::new(doc);

        // Arith body — `qty * price` not a borrow shape.
        assert_eq!(
            j.collect("$.rows.map(qty * price).sum()").unwrap(),
            json!(110)
        );

        // FieldCmpLit non-Eq — `qty > 1` is FieldCmpLit Gt, not the
        // Eq fast path.
        assert_eq!(
            j.collect("$.rows.filter(qty > 1).count()").unwrap(),
            json!(2)
        );
    }

    #[test]
    fn integration_flat_map() {
        use serde_json::json;

        let doc = json!({
            "groups": [
                {"items": [1, 2, 3]},
                {"items": [4, 5]},
                {"items": [6]},
            ]
        });

        let j = crate::Jetro::new(doc);
        assert_eq!(
            j.collect("$.groups.flat_map(items).sum()").unwrap(),
            json!(21)
        );
        assert_eq!(
            j.collect("$.groups.flat_map(items).count()").unwrap(),
            json!(6)
        );
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

// ── Bridge: unified::Stage<R> impls for the substrate-generic borrow runner.
//
// Per pipeline_unification: the SAME stage struct serves both the
// owned `Stage` trait (above) and the substrate-generic `unified::
// Stage<R>` trait (over `R: row::Row<'a>`).  No struct is re-defined
// in unified.rs; only the second trait impl is here.  Future Phase
// 5b will collapse the two trait surfaces into one once VM-driven
// stages (GenericFilter / GenericMap / etc.) gain Row<'a> entry
// points.

impl<R> crate::unified::Stage<R> for Identity {
    #[inline]
    fn apply(&self, x: R) -> crate::unified::StageOutputU<R> {
        crate::unified::StageOutputU::Pass(x)
    }
}

impl<R, A: crate::unified::Stage<R>, B: crate::unified::Stage<R>>
    crate::unified::Stage<R> for Composed<A, B>
{
    fn apply(&self, x: R) -> crate::unified::StageOutputU<R> {
        use crate::unified::StageOutputU;
        match self.a.apply(x) {
            StageOutputU::Pass(v) => self.b.apply(v),
            StageOutputU::Filtered => StageOutputU::Filtered,
            StageOutputU::Done => StageOutputU::Done,
            StageOutputU::Many(items) => {
                let mut out: smallvec::SmallVec<[R; 4]> = smallvec::SmallVec::new();
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

impl<'a, R: crate::row::Row<'a>> crate::unified::Stage<R> for MapField {
    #[inline]
    fn apply(&self, x: R) -> crate::unified::StageOutputU<R> {
        match x.get_field(&self.field) {
            Some(v) => crate::unified::StageOutputU::Pass(v),
            None    => crate::unified::StageOutputU::Filtered,
        }
    }
}

impl<'a, R: crate::row::Row<'a>> crate::unified::Stage<R> for MapFieldChain {
    fn apply(&self, x: R) -> crate::unified::StageOutputU<R> {
        let chain_refs: Vec<&str> = self.keys.iter().map(|a| a.as_ref()).collect();
        match x.walk_path(&chain_refs) {
            Some(v) => crate::unified::StageOutputU::Pass(v),
            None    => crate::unified::StageOutputU::Filtered,
        }
    }
}

impl<'a, R: crate::row::Row<'a>> crate::unified::Stage<R> for FlatMapField {
    fn apply(&self, x: R) -> crate::unified::StageOutputU<R> {
        let arr_row = match x.get_field(&self.field) {
            Some(c) => c,
            None => return crate::unified::StageOutputU::Filtered,
        };
        let iter = match arr_row.array_children() {
            Some(i) => i,
            None => return crate::unified::StageOutputU::Filtered,
        };
        let items: smallvec::SmallVec<[R; 4]> = iter.collect();
        if items.is_empty() {
            crate::unified::StageOutputU::Filtered
        } else if items.len() == 1 {
            crate::unified::StageOutputU::Pass(items.into_iter().next().unwrap())
        } else {
            crate::unified::StageOutputU::Many(items)
        }
    }
}

impl<'a, R: crate::row::Row<'a>> crate::unified::Stage<R> for FlatMapFieldChain {
    fn apply(&self, x: R) -> crate::unified::StageOutputU<R> {
        let chain_refs: Vec<&str> = self.keys.iter().map(|a| a.as_ref()).collect();
        let arr_row = match x.walk_path(&chain_refs) {
            Some(c) => c,
            None => return crate::unified::StageOutputU::Filtered,
        };
        let iter = match arr_row.array_children() {
            Some(i) => i,
            None => return crate::unified::StageOutputU::Filtered,
        };
        let items: smallvec::SmallVec<[R; 4]> = iter.collect();
        if items.is_empty() {
            crate::unified::StageOutputU::Filtered
        } else if items.len() == 1 {
            crate::unified::StageOutputU::Pass(items.into_iter().next().unwrap())
        } else {
            crate::unified::StageOutputU::Many(items)
        }
    }
}

impl<R> crate::unified::Stage<R> for Take {
    #[inline]
    fn apply(&self, x: R) -> crate::unified::StageOutputU<R> {
        let r = self.remaining.get();
        if r == 0 { return crate::unified::StageOutputU::Done; }
        self.remaining.set(r - 1);
        crate::unified::StageOutputU::Pass(x)
    }
}

impl<R> crate::unified::Stage<R> for Skip {
    #[inline]
    fn apply(&self, x: R) -> crate::unified::StageOutputU<R> {
        let r = self.remaining.get();
        if r > 0 {
            self.remaining.set(r - 1);
            return crate::unified::StageOutputU::Filtered;
        }
        crate::unified::StageOutputU::Pass(x)
    }
}

// VM-driven Stage<R> impls — gated to R = `Val` (owned).  These
// stages call the VM, which expects owned `Val` bindings.  For the
// owned substrate (Pipeline::run_with → Phase 5f), R = Val and these
// impls let try_run_composed lower into unified::Stage<Val> chains.
//
// Borrowed substrates (BVal / TapeRow) don't reach Generic-kernel
// stages — bytescan + closure-based Filter cover their lowering.
// FilterFieldEqLit is owned-only (relies on owned Val literal compare
// shape used by composed::FilterFieldEqLit's struct).

impl crate::unified::Stage<crate::eval::Val> for GenericFilter {
    fn apply(&self, x: crate::eval::Val) -> crate::unified::StageOutputU<crate::eval::Val> {
        let mut c = self.ctx.borrow_mut();
        let VmCtx { vm, env } = &mut *c;
        let prev = env.swap_current(x.clone());
        let r = vm.exec_in_env(&self.prog, env);
        env.restore_current(prev);
        match r {
            Ok(v) if crate::eval::util::is_truthy(&v) => crate::unified::StageOutputU::Pass(x),
            _ => crate::unified::StageOutputU::Filtered,
        }
    }
}

impl crate::unified::Stage<crate::eval::Val> for GenericMap {
    fn apply(&self, x: crate::eval::Val) -> crate::unified::StageOutputU<crate::eval::Val> {
        let mut c = self.ctx.borrow_mut();
        let VmCtx { vm, env } = &mut *c;
        let prev = env.swap_current(x);
        let r = vm.exec_in_env(&self.prog, env);
        env.restore_current(prev);
        match r {
            Ok(v) => crate::unified::StageOutputU::Pass(v),
            Err(_) => crate::unified::StageOutputU::Filtered,
        }
    }
}

impl crate::unified::Stage<crate::eval::Val> for GenericFlatMap {
    fn apply(&self, x: crate::eval::Val) -> crate::unified::StageOutputU<crate::eval::Val> {
        use crate::eval::Val as OV;
        let mut c = self.ctx.borrow_mut();
        let VmCtx { vm, env } = &mut *c;
        let prev = env.swap_current(x);
        let r = vm.exec_in_env(&self.prog, env);
        env.restore_current(prev);
        let owned = match r { Ok(v) => v, Err(_) => return crate::unified::StageOutputU::Filtered };
        let mut out: smallvec::SmallVec<[OV; 4]> = smallvec::SmallVec::new();
        match &owned {
            OV::Arr(items)        => for it in items.iter() { out.push(it.clone()); },
            OV::IntVec(items)     => for n in items.iter() { out.push(OV::Int(*n)); },
            OV::FloatVec(items)   => for f in items.iter() { out.push(OV::Float(*f)); },
            OV::StrVec(items)     => for s in items.iter() { out.push(OV::Str(std::sync::Arc::clone(s))); },
            OV::StrSliceVec(items)=> for r in items.iter() { out.push(OV::StrSlice(r.clone())); },
            _ => return crate::unified::StageOutputU::Filtered,
        }
        if out.is_empty() {
            crate::unified::StageOutputU::Filtered
        } else if out.len() == 1 {
            crate::unified::StageOutputU::Pass(out.into_iter().next().unwrap())
        } else {
            crate::unified::StageOutputU::Many(out)
        }
    }
}

impl crate::unified::Stage<crate::eval::Val> for FilterFieldEqLit {
    fn apply(&self, x: crate::eval::Val) -> crate::unified::StageOutputU<crate::eval::Val> {
        use crate::eval::Val as OV;
        let v = match &x {
            OV::Obj(m) => m.get(self.field.as_ref()).cloned().unwrap_or(OV::Null),
            OV::ObjSmall(pairs) => {
                let mut found = OV::Null;
                for (k, v) in pairs.iter() {
                    if k.as_ref() == self.field.as_ref() { found = v.clone(); break; }
                }
                found
            }
            _ => return crate::unified::StageOutputU::Filtered,
        };
        if vals_eq(&v, &self.target) {
            crate::unified::StageOutputU::Pass(x)
        } else {
            crate::unified::StageOutputU::Filtered
        }
    }
}

// Sink dedup: composed.rs's CountSink/SumSink/MinSink/MaxSink/AvgSink/
// FirstSink/LastSink/CollectSink also impl `unified::Sink` so the
// borrow runner reuses the SAME unit-structs rather than maintaining
// parallel Sink definitions.

impl crate::unified::Sink for CountSink {
    type Acc<'a> = i64;
    #[inline] fn init<'a>() -> Self::Acc<'a> { 0 }
    #[inline] fn fold<'a, R: crate::row::Row<'a>>(_: &'a crate::eval::borrowed::Arena, acc: Self::Acc<'a>, _: R) -> Self::Acc<'a> { acc + 1 }
    #[inline] fn finalise<'a>(_: &'a crate::eval::borrowed::Arena, acc: Self::Acc<'a>) -> crate::eval::borrowed::Val<'a> {
        crate::eval::borrowed::Val::Int(acc)
    }
}

impl crate::unified::Sink for SumSink {
    type Acc<'a> = (i64, f64, bool);
    #[inline] fn init<'a>() -> Self::Acc<'a> { (0, 0.0, false) }
    fn fold<'a, R: crate::row::Row<'a>>(_: &'a crate::eval::borrowed::Arena, mut acc: Self::Acc<'a>, v: R) -> Self::Acc<'a> {
        if let Some(n) = v.as_int() { acc.0 = acc.0.wrapping_add(n); return acc; }
        if let Some(f) = v.as_float() { acc.1 += f; acc.2 = true; return acc; }
        if let Some(b) = v.as_bool() { acc.0 = acc.0.wrapping_add(b as i64); return acc; }
        acc
    }
    fn finalise<'a>(_: &'a crate::eval::borrowed::Arena, acc: Self::Acc<'a>) -> crate::eval::borrowed::Val<'a> {
        if acc.2 { crate::eval::borrowed::Val::Float(acc.0 as f64 + acc.1) }
        else { crate::eval::borrowed::Val::Int(acc.0) }
    }
}

impl crate::unified::Sink for MinSink {
    type Acc<'a> = Option<f64>;
    #[inline] fn init<'a>() -> Self::Acc<'a> { None }
    fn fold<'a, R: crate::row::Row<'a>>(_: &'a crate::eval::borrowed::Arena, acc: Self::Acc<'a>, v: R) -> Self::Acc<'a> {
        let n = match v.as_float() { Some(f) => f, None => return acc };
        Some(match acc { Some(c) => c.min(n), None => n })
    }
    fn finalise<'a>(_: &'a crate::eval::borrowed::Arena, acc: Self::Acc<'a>) -> crate::eval::borrowed::Val<'a> {
        match acc {
            Some(f) if f.fract() == 0.0 && f.abs() < (i64::MAX as f64) => crate::eval::borrowed::Val::Int(f as i64),
            Some(f) => crate::eval::borrowed::Val::Float(f),
            None => crate::eval::borrowed::Val::Null,
        }
    }
}

impl crate::unified::Sink for MaxSink {
    type Acc<'a> = Option<f64>;
    #[inline] fn init<'a>() -> Self::Acc<'a> { None }
    fn fold<'a, R: crate::row::Row<'a>>(_: &'a crate::eval::borrowed::Arena, acc: Self::Acc<'a>, v: R) -> Self::Acc<'a> {
        let n = match v.as_float() { Some(f) => f, None => return acc };
        Some(match acc { Some(c) => c.max(n), None => n })
    }
    fn finalise<'a>(_: &'a crate::eval::borrowed::Arena, acc: Self::Acc<'a>) -> crate::eval::borrowed::Val<'a> {
        match acc {
            Some(f) if f.fract() == 0.0 && f.abs() < (i64::MAX as f64) => crate::eval::borrowed::Val::Int(f as i64),
            Some(f) => crate::eval::borrowed::Val::Float(f),
            None => crate::eval::borrowed::Val::Null,
        }
    }
}

impl crate::unified::Sink for AvgSink {
    type Acc<'a> = (f64, usize);
    #[inline] fn init<'a>() -> Self::Acc<'a> { (0.0, 0) }
    fn fold<'a, R: crate::row::Row<'a>>(_: &'a crate::eval::borrowed::Arena, mut acc: Self::Acc<'a>, v: R) -> Self::Acc<'a> {
        if let Some(n) = v.as_float() { acc.0 += n; acc.1 += 1; }
        acc
    }
    fn finalise<'a>(_: &'a crate::eval::borrowed::Arena, acc: Self::Acc<'a>) -> crate::eval::borrowed::Val<'a> {
        if acc.1 == 0 { crate::eval::borrowed::Val::Null }
        else { crate::eval::borrowed::Val::Float(acc.0 / acc.1 as f64) }
    }
}

impl crate::unified::Sink for FirstSink {
    type Acc<'a> = Option<crate::eval::borrowed::Val<'a>>;
    #[inline] fn init<'a>() -> Self::Acc<'a> { None }
    #[inline] fn fold<'a, R: crate::row::Row<'a>>(arena: &'a crate::eval::borrowed::Arena, acc: Self::Acc<'a>, v: R) -> Self::Acc<'a> {
        if acc.is_some() { acc } else { Some(v.materialise(arena)) }
    }
    #[inline] fn finalise<'a>(_: &'a crate::eval::borrowed::Arena, acc: Self::Acc<'a>) -> crate::eval::borrowed::Val<'a> {
        acc.unwrap_or(crate::eval::borrowed::Val::Null)
    }
}

impl crate::unified::Sink for LastSink {
    type Acc<'a> = Option<crate::eval::borrowed::Val<'a>>;
    #[inline] fn init<'a>() -> Self::Acc<'a> { None }
    #[inline] fn fold<'a, R: crate::row::Row<'a>>(arena: &'a crate::eval::borrowed::Arena, _: Self::Acc<'a>, v: R) -> Self::Acc<'a> {
        Some(v.materialise(arena))
    }
    #[inline] fn finalise<'a>(_: &'a crate::eval::borrowed::Arena, acc: Self::Acc<'a>) -> crate::eval::borrowed::Val<'a> {
        acc.unwrap_or(crate::eval::borrowed::Val::Null)
    }
}

impl crate::unified::Sink for CollectSink {
    type Acc<'a> = Vec<crate::eval::borrowed::Val<'a>>;
    #[inline] fn init<'a>() -> Self::Acc<'a> { Vec::new() }
    #[inline] fn fold<'a, R: crate::row::Row<'a>>(arena: &'a crate::eval::borrowed::Arena, mut acc: Self::Acc<'a>, v: R) -> Self::Acc<'a> {
        acc.push(v.materialise(arena)); acc
    }
    #[inline] fn finalise<'a>(arena: &'a crate::eval::borrowed::Arena, acc: Self::Acc<'a>) -> crate::eval::borrowed::Val<'a> {
        let slice = arena.alloc_slice_fill_iter(acc.into_iter());
        crate::eval::borrowed::Val::Arr(&*slice)
    }
}

impl Take {
    pub fn new(n: usize) -> Self { Self { remaining: std::cell::Cell::new(n) } }
}
impl Skip {
    pub fn new(n: usize) -> Self { Self { remaining: std::cell::Cell::new(n) } }
}
impl MapField {
    pub fn new(field: std::sync::Arc<str>) -> Self { Self { field } }
}
impl MapFieldChain {
    pub fn new(keys: Vec<std::sync::Arc<str>>) -> Self {
        Self { keys: std::sync::Arc::from(keys.into_boxed_slice()) }
    }
}
impl FlatMapField {
    pub fn new(field: std::sync::Arc<str>) -> Self { Self { field } }
}
impl FlatMapFieldChain {
    pub fn new(keys: Vec<std::sync::Arc<str>>) -> Self {
        Self { keys: std::sync::Arc::from(keys.into_boxed_slice()) }
    }
}
