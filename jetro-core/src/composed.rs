//! Physical pipeline operators built from composed-Cow Stage chains.
//!
//! Per `pipeline_specialisation.md` (CORRECTED 2026-04-26 + cold-first
//! pivot): replace the legacy fused-Sink enum + ~30 hand-rolled VM
//! opcodes with a single generic substrate driven by composition. No
//! per-shape opcodes, no enumerated `(stage_chain × sink)` Sink
//! variants. Composition primitives + a
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
//! `pipeline.rs` plans streamable query shapes; this module executes the
//! reusable per-element operators and sinks for those plans.

use smallvec::SmallVec;
use std::borrow::{Borrow, Cow};

use crate::builtins::BuiltinCall;
use crate::chain_ir::PullDemand;
use crate::value::Val;

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
}

/// Blanket impl so `Box<dyn Stage>` itself implements `Stage`. Lets a
/// chain of stages be folded into a single `Box<dyn Stage>` at
/// lower-time without macro acrobatics.
impl<T: Stage + ?Sized> Stage for Box<T> {
    #[inline]
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        (**self).apply(x)
    }
}

/// Identity stage — pass-through. Used as the fold seed when composing
/// a chain of stages.
pub struct Identity;

impl Default for Identity {
    fn default() -> Self {
        Self
    }
}

impl Stage for Identity {
    #[inline]
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        StageOutput::Pass(Cow::Borrowed(x))
    }
}

/// Adapter that lets the composed runtime execute canonical builtins
/// without defining one `Stage` struct per builtin.
pub struct BuiltinStage {
    call: BuiltinCall,
}

impl BuiltinStage {
    pub fn new(call: BuiltinCall) -> Self {
        Self { call }
    }
}

impl Stage for BuiltinStage {
    #[inline]
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        match self.call.apply(x) {
            Some(v) => StageOutput::Pass(Cow::Owned(v)),
            None => StageOutput::Filtered,
        }
    }
}

/// Monoidal composition. `Composed { a, b }.apply(x) = b.apply(a.apply(x))`
/// with proper handling of Filtered / Many / Done propagation.
pub struct Composed<A, B> {
    pub a: A,
    pub b: B,
}

impl<A: Stage, B: Stage> Stage for Composed<A, B> {
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        match self.a.apply(x) {
            StageOutput::Pass(Cow::Borrowed(v)) => self.b.apply(v),
            StageOutput::Pass(Cow::Owned(v)) => {
                // `b` may borrow from v, but v dies at scope end. Force
                // owned result so the returned lifetime is independent of v.
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
                            // `v` dies at end of this arm, so promote any
                            // borrow returned by `b` to an owned value.
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
    #[inline]
    fn done(_acc: &Self::Acc) -> bool {
        false
    }
    fn finalise(acc: Self::Acc) -> Val;
}

// ── Generic outer loop ──────────────────────────────────────────────────────

/// Parameterised outer loop. One pass over `arr`, dispatching each
/// element through the composed `stages`, accumulating into `S::Acc`.
/// Stages composed once at lower-time, used N× here.
pub fn run_pipeline<S: Sink>(arr: &[Val], stages: &dyn Stage) -> Val {
    run_pipeline_with_demand::<S>(arr, stages, PullDemand::All)
}

/// Demand-aware variant of [`run_pipeline`]. `FirstInput(n)` is a hard cap
/// on upstream inputs pulled from `arr`; `UntilOutput(n)` stops after n
/// values have reached the sink, regardless of how many inputs filters
/// had to inspect to produce them.
pub fn run_pipeline_with_demand<S: Sink>(
    arr: &[Val],
    stages: &dyn Stage,
    demand: PullDemand,
) -> Val {
    run_pipeline_iter_with_demand::<S, _>(arr.iter(), stages, demand)
}

pub fn run_pipeline_owned_iter_with_demand<S, I>(
    rows: I,
    stages: &dyn Stage,
    demand: PullDemand,
) -> Val
where
    S: Sink,
    I: IntoIterator<Item = Val>,
{
    run_pipeline_iter_with_demand::<S, _>(rows.into_iter(), stages, demand)
}

fn run_pipeline_iter_with_demand<'a, S, I>(rows: I, stages: &dyn Stage, demand: PullDemand) -> Val
where
    S: Sink,
    I: IntoIterator,
    I::Item: std::borrow::Borrow<Val>,
{
    let mut acc = S::init();
    let mut pulled_inputs = 0usize;
    let mut emitted_outputs = 0usize;
    for v in rows {
        if matches!(demand, PullDemand::FirstInput(n) if pulled_inputs >= n) {
            break;
        }
        pulled_inputs += 1;
        match stages.apply(v.borrow()) {
            StageOutput::Pass(cow) => {
                acc = S::fold(acc, cow.as_ref());
                emitted_outputs += 1;
                if S::done(&acc) {
                    break;
                }
                if matches!(demand, PullDemand::UntilOutput(n) if emitted_outputs >= n) {
                    break;
                }
            }
            StageOutput::Filtered => continue,
            StageOutput::Many(items) => {
                for it in items {
                    acc = S::fold(acc, it.as_ref());
                    emitted_outputs += 1;
                    if S::done(&acc) {
                        break;
                    }
                    if matches!(demand, PullDemand::UntilOutput(n) if emitted_outputs >= n) {
                        break;
                    }
                }
                if S::done(&acc)
                    || matches!(demand, PullDemand::UntilOutput(n) if emitted_outputs >= n)
                {
                    break;
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
    #[inline]
    fn init() -> i64 {
        0
    }
    #[inline]
    fn fold(acc: i64, _: &Val) -> i64 {
        acc + 1
    }
    #[inline]
    fn finalise(acc: i64) -> Val {
        Val::Int(acc)
    }
}

pub struct SumSink;
impl Sink for SumSink {
    type Acc = (i64, f64, bool); // (int_sum, float_sum, any_float)
    #[inline]
    fn init() -> Self::Acc {
        (0, 0.0, false)
    }
    fn fold(mut acc: Self::Acc, v: &Val) -> Self::Acc {
        match v {
            Val::Int(i) => acc.0 += *i,
            Val::Float(f) => {
                acc.1 += *f;
                acc.2 = true;
            }
            Val::Bool(b) => acc.0 += *b as i64,
            _ => {}
        }
        acc
    }
    fn finalise(acc: Self::Acc) -> Val {
        if acc.2 {
            Val::Float(acc.0 as f64 + acc.1)
        } else {
            Val::Int(acc.0)
        }
    }
}

pub struct MinSink;
impl Sink for MinSink {
    type Acc = Option<f64>;
    #[inline]
    fn init() -> Self::Acc {
        None
    }
    fn fold(acc: Self::Acc, v: &Val) -> Self::Acc {
        let n = match v {
            Val::Int(i) => *i as f64,
            Val::Float(f) => *f,
            _ => return acc,
        };
        Some(match acc {
            Some(cur) => cur.min(n),
            None => n,
        })
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
    #[inline]
    fn init() -> Self::Acc {
        None
    }
    fn fold(acc: Self::Acc, v: &Val) -> Self::Acc {
        let n = match v {
            Val::Int(i) => *i as f64,
            Val::Float(f) => *f,
            _ => return acc,
        };
        Some(match acc {
            Some(cur) => cur.max(n),
            None => n,
        })
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
    #[inline]
    fn init() -> Self::Acc {
        (0.0, 0)
    }
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
        if acc.1 == 0 {
            Val::Null
        } else {
            Val::Float(acc.0 / acc.1 as f64)
        }
    }
}

pub struct FirstSink;
impl Sink for FirstSink {
    type Acc = Option<Val>;
    #[inline]
    fn init() -> Self::Acc {
        None
    }
    fn fold(acc: Self::Acc, v: &Val) -> Self::Acc {
        if acc.is_some() {
            acc
        } else {
            Some(v.clone())
        }
    }
    #[inline]
    fn done(acc: &Self::Acc) -> bool {
        acc.is_some()
    }
    fn finalise(acc: Self::Acc) -> Val {
        acc.unwrap_or(Val::Null)
    }
}

pub struct LastSink;
impl Sink for LastSink {
    type Acc = Option<Val>;
    #[inline]
    fn init() -> Self::Acc {
        None
    }
    fn fold(_acc: Self::Acc, v: &Val) -> Self::Acc {
        Some(v.clone())
    }
    fn finalise(acc: Self::Acc) -> Val {
        acc.unwrap_or(Val::Null)
    }
}

pub struct CollectSink;
impl Sink for CollectSink {
    type Acc = Vec<Val>;
    #[inline]
    fn init() -> Self::Acc {
        Vec::new()
    }
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
    pub env: crate::context::Env,
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
        // Single source of truth for filter semantics: route the
        // truthy-check through `crate::builtins::filter_one` so this
        // backend shares its definition with vm.rs + pipeline.rs.
        let kept = crate::builtins::filter_one(x, |item| {
            let prev = env.swap_current(item.clone());
            let r = vm.exec_in_env(&self.prog, env);
            env.restore_current(prev);
            r
        });
        match kept {
            Ok(true) => StageOutput::Pass(Cow::Borrowed(x)),
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
        let result: StageOutput<'a> = match owned.into_vals() {
            Ok(items) => many_from_owned_vals(items),
            Err(_) => StageOutput::Filtered,
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

/// Generic flatten dispatch. `Val::as_vals()` is the single lane
/// adapter: Arr rows stay borrowed, typed lanes and ObjVec materialize
/// only at this compatibility boundary.
#[inline]
fn flatten_iterable<'a>(v: &'a Val) -> StageOutput<'a> {
    match v.as_vals() {
        Some(items) => many_from_vals(items, true),
        None => StageOutput::Filtered,
    }
}

fn many_from_vals<'a>(items: Cow<'a, [Val]>, allow_borrow: bool) -> StageOutput<'a> {
    match items {
        Cow::Borrowed(items) if allow_borrow => {
            let mut out: SmallVec<[Cow<'a, Val>; 4]> = SmallVec::with_capacity(items.len());
            for item in items {
                out.push(Cow::Borrowed(item));
            }
            if out.is_empty() {
                StageOutput::Filtered
            } else {
                StageOutput::Many(out)
            }
        }
        items => many_from_owned_vals(items.into_owned()),
    }
}

fn many_from_owned_vals<'a>(items: Vec<Val>) -> StageOutput<'a> {
    let mut out: SmallVec<[Cow<'a, Val>; 4]> = SmallVec::with_capacity(items.len());
    for item in items {
        out.push(Cow::Owned(item));
    }
    if out.is_empty() {
        StageOutput::Filtered
    } else {
        StageOutput::Many(out)
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
        if r == 0 {
            return StageOutput::Done;
        }
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
/// `cmp_val` — wraps `util::cmp_val` for primitive Vals.
pub fn barrier_sort(buf: Vec<Val>, key: &KeySource) -> Vec<Val> {
    let mut indexed: Vec<(Val, Val)> = buf.into_iter().map(|v| (key.extract(&v), v)).collect();
    indexed.sort_by(|a, b| cmp_val(&a.0, &b.0));
    indexed.into_iter().map(|(_, v)| v).collect()
}

/// Top-k sort: keep the `k` smallest-by-key elements, sorted ascending.
/// O(N log k) via a max-heap of size k — for `k << N` this is the
/// algorithmic win that demand propagation unlocks (`Sort ∘ First` →
/// `Sort.adapt(FirstInput(1))` → this kernel).
pub fn barrier_top_k(buf: Vec<Val>, key: &KeySource, k: usize) -> Vec<Val> {
    barrier_top_or_bottom_k(buf, key, k, false)
}

/// Bottom-k sort: keep the `k` largest-by-key elements, sorted ascending.
/// Driven by `Sort ∘ Last` / positional=Last under demand propagation.
pub fn barrier_bottom_k(buf: Vec<Val>, key: &KeySource, k: usize) -> Vec<Val> {
    barrier_top_or_bottom_k(buf, key, k, true)
}

fn barrier_top_or_bottom_k(buf: Vec<Val>, key: &KeySource, k: usize, largest: bool) -> Vec<Val> {
    let strategy = if largest {
        crate::pipeline::StageStrategy::SortBottomK(k)
    } else {
        crate::pipeline::StageStrategy::SortTopK(k)
    };
    crate::pipeline::bounded_sort_by_key_cmp(buf, false, strategy, |v| Ok(key.extract(v)), cmp_val)
        .unwrap_or_default()
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
    let mut m: indexmap::IndexMap<std::sync::Arc<str>, Val> =
        indexmap::IndexMap::with_capacity(groups.len());
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
    Float(u64), // f64::to_bits for total ordering
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
pub(crate) fn cmp_val(a: &Val, b: &Val) -> std::cmp::Ordering {
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
        (Val::Str(x), Val::StrSlice(r)) | (Val::StrSlice(r), Val::Str(x)) => {
            x.as_ref() == r.as_str()
        }
        (Val::StrSlice(x), Val::StrSlice(y)) => x.as_str() == y.as_str(),
        _ => false,
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::{BuiltinArgs, BuiltinCall, BuiltinMethod};
    use indexmap::IndexMap;
    use std::sync::Arc;

    fn builtin(method: BuiltinMethod, args: BuiltinArgs) -> BuiltinStage {
        BuiltinStage::new(BuiltinCall::new(method, args))
    }

    fn builtin0(method: BuiltinMethod) -> BuiltinStage {
        builtin(method, BuiltinArgs::None)
    }

    fn builtin_str(method: BuiltinMethod, s: &str) -> BuiltinStage {
        builtin(method, BuiltinArgs::Str(Arc::from(s)))
    }

    fn builtin_pair(method: BuiltinMethod, first: &str, second: &str) -> BuiltinStage {
        builtin(
            method,
            BuiltinArgs::StrPair {
                first: Arc::from(first),
                second: Arc::from(second),
            },
        )
    }

    fn builtin_usize(method: BuiltinMethod, n: usize) -> BuiltinStage {
        builtin(method, BuiltinArgs::Usize(n))
    }

    fn builtin_i64(method: BuiltinMethod, n: i64) -> BuiltinStage {
        builtin(method, BuiltinArgs::I64(n))
    }

    fn builtin_pad(method: BuiltinMethod, width: usize, fill: char) -> BuiltinStage {
        builtin(method, BuiltinArgs::Pad { width, fill })
    }

    fn builtin_valvec(method: BuiltinMethod, values: Vec<Val>) -> BuiltinStage {
        builtin(method, BuiltinArgs::ValVec(values))
    }

    fn builtin_strvec(method: BuiltinMethod, values: &[&str]) -> BuiltinStage {
        builtin(
            method,
            BuiltinArgs::StrVec(values.iter().map(|s| Arc::from(*s)).collect()),
        )
    }

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
            a: FilterFieldEqLit {
                field: Arc::from("a"),
                target: Val::Int(1),
            },
            b: MapField {
                field: Arc::from("b"),
            },
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
            a: FilterFieldEqLit {
                field: Arc::from("a"),
                target: Val::Int(1),
            },
            b: MapField {
                field: Arc::from("b"),
            },
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
        let arr = vec![obj(&[("u", mid1)]), obj(&[("u", mid2)])];
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
        let stages = Take {
            remaining: std::cell::Cell::new(3),
        };
        let out = run_pipeline::<CountSink>(&arr, &stages);
        assert!(matches!(out, Val::Int(3)));
    }

    #[test]
    fn skip_drops_prefix() {
        let arr: Vec<Val> = (0..10).map(Val::Int).collect();
        let stages = Skip {
            remaining: std::cell::Cell::new(7),
        };
        let out = run_pipeline::<CountSink>(&arr, &stages);
        assert!(matches!(out, Val::Int(3)));
    }

    #[test]
    fn min_max_avg_basic() {
        let arr: Vec<Val> = vec![Val::Int(5), Val::Int(2), Val::Int(9), Val::Int(3)];
        assert!(matches!(
            run_pipeline::<MinSink>(&arr, &Identity),
            Val::Int(2)
        ));
        assert!(matches!(
            run_pipeline::<MaxSink>(&arr, &Identity),
            Val::Int(9)
        ));
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
        assert!(matches!(
            run_pipeline::<FirstSink>(&arr, &Identity),
            Val::Int(10)
        ));
        assert!(matches!(
            run_pipeline::<LastSink>(&arr, &Identity),
            Val::Int(30)
        ));
    }

    #[test]
    fn first_sink_terminates_outer_loop() {
        struct CountingPass<'a>(&'a std::cell::Cell<usize>);

        impl Stage for CountingPass<'_> {
            fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
                self.0.set(self.0.get() + 1);
                StageOutput::Pass(Cow::Borrowed(x))
            }
        }

        let seen = std::cell::Cell::new(0);
        let arr: Vec<Val> = (0..1000).map(Val::Int).collect();
        let out = run_pipeline::<FirstSink>(&arr, &CountingPass(&seen));
        assert!(matches!(out, Val::Int(0)));
        assert_eq!(seen.get(), 1);
    }

    #[test]
    fn demand_first_input_caps_composed_inputs() {
        struct CountingPass<'a>(&'a std::cell::Cell<usize>);

        impl Stage for CountingPass<'_> {
            fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
                self.0.set(self.0.get() + 1);
                StageOutput::Pass(Cow::Borrowed(x))
            }
        }

        let seen = std::cell::Cell::new(0);
        let arr: Vec<Val> = (0..1000).map(Val::Int).collect();
        let out = run_pipeline_with_demand::<CountSink>(
            &arr,
            &CountingPass(&seen),
            PullDemand::FirstInput(3),
        );
        assert!(matches!(out, Val::Int(3)));
        assert_eq!(seen.get(), 3);
    }

    #[test]
    fn demand_until_output_counts_emitted_values() {
        struct CountingEven<'a>(&'a std::cell::Cell<usize>);

        impl Stage for CountingEven<'_> {
            fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
                self.0.set(self.0.get() + 1);
                match x {
                    Val::Int(n) if n % 2 == 0 => StageOutput::Pass(Cow::Borrowed(x)),
                    _ => StageOutput::Filtered,
                }
            }
        }

        let seen = std::cell::Cell::new(0);
        let arr: Vec<Val> = (1..1000).map(Val::Int).collect();
        let out = run_pipeline_with_demand::<CollectSink>(
            &arr,
            &CountingEven(&seen),
            PullDemand::UntilOutput(2),
        );
        let Val::Arr(items) = out else {
            panic!("expected Arr");
        };
        assert_eq!(items.as_ref(), &vec![Val::Int(2), Val::Int(4)]);
        assert_eq!(seen.get(), 4);
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
        assert_eq!(
            j.collect("$.books.filter(active == true).count()").unwrap(),
            json!(2)
        );
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
        // Demand propagation: Sort sees FirstInput(k) downstream and switches
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
        // via inner Stage::Map(@.text) → Stage::Split(",") → Sink::Terminal(BuiltinMethod::First).
        assert_eq!(
            j.collect("$.records.map(@.text.split(\",\").first())")
                .unwrap(),
            json!(["alice", "bob", "carol"])
        );
        // map(@.text.split(",").last()).
        assert_eq!(
            j.collect("$.records.map(@.text.split(\",\").last())")
                .unwrap(),
            json!(["42", "17", "99"])
        );
        // map(@.text.split(",").count()) — count reducer inside body
        // returns one count per row.
        assert_eq!(
            j.collect("$.records.map(@.text.split(\",\").count())")
                .unwrap(),
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
        // .split(",").count() — Stage::Split + count reducer.
        assert_eq!(j.collect("$.s.split(\",\").count()").unwrap(), json!(5));
        // .split(",").first() — Stage::Split + Sink::Terminal(BuiltinMethod::First).
        assert_eq!(j.collect("$.s.split(\",\").first()").unwrap(), json!("a"));
        // .split(",").last() — Stage::Split + Sink::Terminal(BuiltinMethod::Last).
        assert_eq!(j.collect("$.s.split(\",\").last()").unwrap(), json!("e"));
    }

    #[test]
    fn step3d_phase3_filter_reorder() {
        // Phase 3: adjacent Filter runs reorder by cost / (1 - selectivity).
        // Cheaper, more-selective filter first — Eq (sel=0.10) before
        // Lt (sel=0.40).  Reorder must preserve overall set semantics.
        use crate::ast::BinOp;
        use crate::pipeline::{plan_with_kernels, BodyKernel, Sink, Stage};
        use std::sync::Arc;
        let dummy = Arc::new(crate::vm::Program::new(Vec::new(), ""));
        let stages = vec![
            // [0]: Filter(price < 100)  — selectivity 0.4
            Stage::Filter(
                Arc::clone(&dummy),
                crate::builtins::BuiltinViewStage::Filter,
            ),
            // [1]: Filter(active == true) — selectivity 0.1, more selective
            Stage::Filter(
                Arc::clone(&dummy),
                crate::builtins::BuiltinViewStage::Filter,
            ),
        ];
        let kernels = vec![
            BodyKernel::FieldCmpLit(Arc::from("price"), BinOp::Lt, crate::value::Val::Int(100)),
            BodyKernel::FieldCmpLit(
                Arc::from("active"),
                BinOp::Eq,
                crate::value::Val::Bool(true),
            ),
        ];
        let p = plan_with_kernels(
            stages,
            &kernels,
            Sink::Reducer(crate::pipeline::ReducerSpec::count()),
        );
        // Reorder is immediately followed by predicate fusion, so the
        // adjacent filter run becomes one stage. Behavioural correctness is
        // covered by the parity test below.
        assert_eq!(p.stages.len(), 1);
        assert!(matches!(p.stages[0], Stage::Filter(_, _)));
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
            j.collect("$.rows.filter(b > 15).filter(tag == \"x\").count()")
                .unwrap(),
            json!(2)
        );
        // Sum after the same filters.
        assert_eq!(
            j.collect("$.rows.filter(b > 15).filter(tag == \"x\").map(b).sum()")
                .unwrap(),
            json!(80)
        );
    }

    #[test]
    fn step3d_phase4_merge_take_skip() {
        use crate::builtins::{BuiltinStageMerge, BuiltinViewStage};
        use crate::pipeline::{plan, Sink, Stage};
        // Take(5) ∘ Take(3) → Take(3)
        let p = plan(
            vec![
                Stage::Take(5, BuiltinViewStage::Take, BuiltinStageMerge::UsizeMin),
                Stage::Take(3, BuiltinViewStage::Take, BuiltinStageMerge::UsizeMin),
            ],
            Sink::Collect,
        );
        assert_eq!(p.stages.len(), 1);
        assert!(matches!(p.stages[0], Stage::Take(3, _, _)));

        // Skip(2) ∘ Skip(3) → Skip(5)
        let p = plan(
            vec![
                Stage::Skip(
                    2,
                    BuiltinViewStage::Skip,
                    BuiltinStageMerge::UsizeSaturatingAdd,
                ),
                Stage::Skip(
                    3,
                    BuiltinViewStage::Skip,
                    BuiltinStageMerge::UsizeSaturatingAdd,
                ),
            ],
            Sink::Collect,
        );
        assert_eq!(p.stages.len(), 1);
        assert!(matches!(p.stages[0], Stage::Skip(5, _, _)));

        // Reverse ∘ Reverse → identity (drops both)
        let cancel = crate::builtins::BuiltinMethod::Reverse
            .spec()
            .cancellation
            .unwrap();
        let p = plan(
            vec![Stage::Reverse(cancel), Stage::Reverse(cancel)],
            Sink::Collect,
        );
        assert_eq!(p.stages.len(), 0);
    }

    #[test]
    fn step3d_phase5_strategy_selection() {
        use crate::pipeline::{
            select_strategy, NumOp, ReducerOp, ReducerSpec, Sink, SortSpec, Stage, Strategy,
        };
        let first_sink = Sink::Terminal(BuiltinMethod::First);
        use std::sync::Arc;
        let dummy = Arc::new(crate::vm::Program::new(Vec::new(), ""));

        // Map + First → IndexedDispatch
        assert_eq!(
            select_strategy(
                &[Stage::Map(
                    Arc::clone(&dummy),
                    crate::builtins::BuiltinViewStage::Map
                )],
                &first_sink,
            ),
            Strategy::IndexedDispatch
        );
        // Filter + First → EarlyExit (Filter not 1:1)
        assert_eq!(
            select_strategy(
                &[Stage::Filter(
                    Arc::clone(&dummy),
                    crate::builtins::BuiltinViewStage::Filter,
                )],
                &first_sink,
            ),
            Strategy::EarlyExit
        );
        // Sort + First → BarrierMaterialise
        assert_eq!(
            select_strategy(&[Stage::Sort(SortSpec::identity())], &first_sink),
            Strategy::BarrierMaterialise
        );
        // Map + Sum → PullLoop (no positional, no barrier)
        assert_eq!(
            select_strategy(
                &[Stage::Map(
                    Arc::clone(&dummy),
                    crate::builtins::BuiltinViewStage::Map
                )],
                &Sink::Reducer(ReducerSpec {
                    op: ReducerOp::Numeric(NumOp::Sum),
                    predicate: None,
                    projection: None,
                })
            ),
            Strategy::PullLoop
        );
    }

    #[test]
    fn step3d_phase1_compute_strategies() {
        use crate::pipeline::{
            compute_strategies, NumOp, ReducerOp, ReducerSpec, Sink, SortSpec, Stage, StageStrategy,
        };
        use std::sync::Arc;

        let dummy_prog = Arc::new(crate::vm::Program::new(Vec::new(), ""));
        let first_sink = Sink::Terminal(BuiltinMethod::First);

        // [Sort] + First → SortTopK(1)
        let stages = vec![Stage::Sort(SortSpec::keyed(Arc::clone(&dummy_prog), false))];
        let strats = compute_strategies(&stages, &first_sink);
        assert!(matches!(strats[0], StageStrategy::SortTopK(1)));

        // [Sort, Take(5)] + Collect → SortTopK(5) at index 0
        let stages = vec![
            Stage::Sort(SortSpec::keyed(Arc::clone(&dummy_prog), false)),
            Stage::Take(
                5,
                crate::builtins::BuiltinViewStage::Take,
                crate::builtins::BuiltinStageMerge::UsizeMin,
            ),
        ];
        let strats = compute_strategies(&stages, &Sink::Collect);
        assert!(matches!(strats[0], StageStrategy::SortTopK(5)));
        assert!(matches!(strats[1], StageStrategy::Default));

        // [Sort] + Sum → unbounded → Default (full sort)
        let stages = vec![Stage::Sort(SortSpec::identity())];
        let strats = compute_strategies(
            &stages,
            &Sink::Reducer(ReducerSpec {
                op: ReducerOp::Numeric(NumOp::Sum),
                predicate: None,
                projection: None,
            }),
        );
        assert!(matches!(strats[0], StageStrategy::Default));

        // [Sort, Filter] + First cannot use fixed top-k without a
        // monotonic predicate proof, but it can use a lazy ordered
        // producer that keeps popping sorted rows until First is done.
        let stages = vec![
            Stage::Sort(SortSpec::keyed(Arc::clone(&dummy_prog), false)),
            Stage::Filter(
                Arc::clone(&dummy_prog),
                crate::builtins::BuiltinViewStage::Filter,
            ),
        ];
        let strats = compute_strategies(&stages, &first_sink);
        assert!(matches!(strats[0], StageStrategy::SortUntilOutput(1)));
    }

    #[test]
    fn integration_generic_kernels() {
        // Body shapes the borrow stages don't recognise — should
        // still run via the GenericFilter / GenericMap / GenericFlatMap
        // VM-fallback path.
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
    fn upper_owned_stage_applies() {
        let s = Val::Str(std::sync::Arc::from("hello"));
        let stage = builtin0(BuiltinMethod::Upper);
        let got = stage.apply(&s);
        match got {
            StageOutput::Pass(Cow::Owned(Val::Str(out))) => {
                assert_eq!(out.as_ref(), "HELLO");
            }
            _ => panic!("expected Pass(Owned(Str(\"HELLO\")))"),
        }
    }

    #[test]
    fn builtin_stage_adapts_canonical_builtin_call() {
        let s = Val::Str(std::sync::Arc::from("hello world"));
        let stage = BuiltinStage::new(crate::builtins::BuiltinCall::new(
            crate::builtins::BuiltinMethod::EndsWith,
            crate::builtins::BuiltinArgs::Str(std::sync::Arc::from("world")),
        ));

        match stage.apply(&s) {
            StageOutput::Pass(Cow::Owned(Val::Bool(true))) => {}
            other => panic!(
                "expected owned true bool, got {:?}",
                stage_output_kind(&other)
            ),
        };
    }

    #[test]
    fn upper_filters_non_string() {
        let v = Val::Int(42);
        let stage = builtin0(BuiltinMethod::Upper);
        let got = stage.apply(&v);
        match got {
            StageOutput::Pass(cow) => assert!(matches!(cow.into_owned(), Val::Int(42))),
            _ => panic!("expected pass-through for non-string"),
        }
    }

    #[test]
    fn lower_owned_stage_applies() {
        let s = Val::Str(std::sync::Arc::from("HELLO World"));
        let stage = builtin0(BuiltinMethod::Lower);
        let got = stage.apply(&s);
        match got {
            StageOutput::Pass(Cow::Owned(Val::Str(out))) => {
                assert_eq!(out.as_ref(), "hello world");
            }
            _ => panic!("expected lower"),
        }
    }

    fn extract_str(out: StageOutput<'_>) -> String {
        match out {
            StageOutput::Pass(cow) => match cow.into_owned() {
                Val::Str(s) => s.as_ref().to_owned(),
                other => panic!("expected Str, got {:?}", other),
            },
            _ => panic!("expected Pass"),
        }
    }

    fn stage_output_kind(out: &StageOutput<'_>) -> &'static str {
        match out {
            StageOutput::Pass(_) => "pass",
            StageOutput::Filtered => "filtered",
            StageOutput::Many(_) => "many",
            StageOutput::Done => "done",
        }
    }

    #[test]
    fn trim_stages_strip_whitespace() {
        let s = Val::Str(std::sync::Arc::from("  hello  "));
        let r = extract_str(builtin0(BuiltinMethod::Trim).apply(&s));
        assert_eq!(r, "hello");
        let r = extract_str(builtin0(BuiltinMethod::TrimLeft).apply(&s));
        assert_eq!(r, "hello  ");
        let r = extract_str(builtin0(BuiltinMethod::TrimRight).apply(&s));
        assert_eq!(r, "  hello");
    }

    #[test]
    fn lifted_str_stages_filter_non_string() {
        let v = Val::Int(42);
        assert!(matches!(
            builtin0(BuiltinMethod::Lower).apply(&v),
            StageOutput::Pass(_)
        ));
        assert!(matches!(
            builtin0(BuiltinMethod::Trim).apply(&v),
            StageOutput::Pass(_)
        ));
        assert!(matches!(
            builtin0(BuiltinMethod::TrimLeft).apply(&v),
            StageOutput::Pass(_)
        ));
        assert!(matches!(
            builtin0(BuiltinMethod::TrimRight).apply(&v),
            StageOutput::Pass(_)
        ));
        assert!(matches!(
            builtin0(BuiltinMethod::Capitalize).apply(&v),
            StageOutput::Pass(_)
        ));
        assert!(matches!(
            builtin0(BuiltinMethod::TitleCase).apply(&v),
            StageOutput::Pass(_)
        ));
        assert!(matches!(
            builtin0(BuiltinMethod::HtmlEscape).apply(&v),
            StageOutput::Pass(_)
        ));
        assert!(matches!(
            builtin0(BuiltinMethod::UrlEncode).apply(&v),
            StageOutput::Pass(_)
        ));
    }

    #[test]
    fn capitalize_and_title_case() {
        let s = Val::Str(std::sync::Arc::from("hello world"));
        assert_eq!(
            extract_str(builtin0(BuiltinMethod::Capitalize).apply(&s)),
            "Hello world"
        );
        assert_eq!(
            extract_str(builtin0(BuiltinMethod::TitleCase).apply(&s)),
            "Hello World"
        );
    }

    #[test]
    fn html_escape_runs() {
        let s = Val::Str(std::sync::Arc::from("a<b>&'\"c"));
        assert_eq!(
            extract_str(builtin0(BuiltinMethod::HtmlEscape).apply(&s)),
            "a&lt;b&gt;&amp;&#39;&quot;c"
        );
    }

    #[test]
    fn url_encode_unreserved_passthrough() {
        let s = Val::Str(std::sync::Arc::from("hello world!"));
        assert_eq!(
            extract_str(builtin0(BuiltinMethod::UrlEncode).apply(&s)),
            "hello%20world%21"
        );
    }

    #[test]
    fn url_decode_roundtrip() {
        let s = Val::Str(std::sync::Arc::from("hello%20world%21+a"));
        assert_eq!(
            extract_str(builtin0(BuiltinMethod::UrlDecode).apply(&s)),
            "hello world! a"
        );
    }

    #[test]
    fn html_unescape_runs() {
        let s = Val::Str(std::sync::Arc::from("a&lt;b&gt;&amp;&#39;&quot;c"));
        assert_eq!(
            extract_str(builtin0(BuiltinMethod::HtmlUnescape).apply(&s)),
            "a<b>&'\"c"
        );
    }

    fn extract_arr_len(out: StageOutput<'_>) -> usize {
        match out {
            StageOutput::Pass(cow) => match cow.into_owned() {
                Val::Arr(a) => a.len(),
                other => panic!("expected Arr, got {:?}", other),
            },
            _ => panic!("expected Pass"),
        }
    }

    #[test]
    fn lines_words_chars_split_correctly() {
        let s = Val::Str(std::sync::Arc::from("hello world\nfoo bar"));
        assert_eq!(extract_arr_len(builtin0(BuiltinMethod::Lines).apply(&s)), 2);
        assert_eq!(extract_arr_len(builtin0(BuiltinMethod::Words).apply(&s)), 4);
        let small = Val::Str(std::sync::Arc::from("ábc"));
        assert_eq!(
            extract_arr_len(builtin0(BuiltinMethod::Chars).apply(&small)),
            3
        );
    }

    #[test]
    fn to_number_to_bool_dispatch() {
        let i = Val::Str(std::sync::Arc::from("42"));
        match builtin0(BuiltinMethod::ToNumber).apply(&i) {
            StageOutput::Pass(cow) => match cow.into_owned() {
                Val::Int(42) => {}
                other => panic!("expected Int(42), got {:?}", other),
            },
            _ => panic!("expected Pass"),
        }
        let f = Val::Str(std::sync::Arc::from("3.14"));
        match builtin0(BuiltinMethod::ToNumber).apply(&f) {
            StageOutput::Pass(cow) => match cow.into_owned() {
                Val::Float(_) => {}
                other => panic!("expected Float, got {:?}", other),
            },
            _ => panic!("expected Pass"),
        }
        let bad = Val::Str(std::sync::Arc::from("nope"));
        match builtin0(BuiltinMethod::ToNumber).apply(&bad) {
            StageOutput::Pass(cow) => match cow.into_owned() {
                Val::Null => {}
                other => panic!("expected Null, got {:?}", other),
            },
            _ => panic!("expected Pass"),
        }

        let t = Val::Str(std::sync::Arc::from("true"));
        let r = builtin0(BuiltinMethod::ToBool).apply(&t);
        match r {
            StageOutput::Pass(cow) => match cow.into_owned() {
                Val::Bool(true) => {}
                other => panic!("expected Bool(true), got {:?}", other),
            },
            _ => panic!("expected Pass"),
        }
    }

    fn extract_bool(out: StageOutput<'_>) -> bool {
        match out {
            StageOutput::Pass(cow) => match cow.into_owned() {
                Val::Bool(b) => b,
                other => panic!("expected Bool, got {:?}", other),
            },
            _ => panic!("expected Pass"),
        }
    }

    #[test]
    fn starts_ends_contains() {
        let s = Val::Str(std::sync::Arc::from("hello world"));
        assert!(extract_bool(
            builtin_str(BuiltinMethod::StartsWith, "hello").apply(&s)
        ));
        assert!(!extract_bool(
            builtin_str(BuiltinMethod::StartsWith, "world").apply(&s)
        ));
        assert!(extract_bool(
            builtin_str(BuiltinMethod::EndsWith, "world").apply(&s)
        ));
        assert!(extract_bool(
            builtin_str(BuiltinMethod::Matches, "o w").apply(&s)
        ));
    }

    #[test]
    fn repeat_split_replace() {
        let s = Val::Str(std::sync::Arc::from("ab"));
        assert_eq!(
            extract_str(builtin_usize(BuiltinMethod::Repeat, 3).apply(&s)),
            "ababab"
        );

        let csv = Val::Str(std::sync::Arc::from("a,b,c"));
        assert_eq!(
            extract_arr_len(builtin_str(BuiltinMethod::Split, ",").apply(&csv)),
            3
        );

        let s = Val::Str(std::sync::Arc::from("foo bar foo"));
        let r = builtin_pair(BuiltinMethod::ReplaceAll, "foo", "X").apply(&s);
        assert_eq!(extract_str(r), "X bar X");
    }

    #[test]
    fn strip_prefix_suffix_passthrough() {
        let s = Val::Str(std::sync::Arc::from("foobar"));
        assert_eq!(
            extract_str(builtin_str(BuiltinMethod::StripPrefix, "foo").apply(&s)),
            "bar"
        );
        let s2 = Val::Str(std::sync::Arc::from("xyz"));
        assert_eq!(
            extract_str(builtin_str(BuiltinMethod::StripPrefix, "foo").apply(&s2)),
            "xyz"
        );
        let s3 = Val::Str(std::sync::Arc::from("hello.txt"));
        assert_eq!(
            extract_str(builtin_str(BuiltinMethod::StripSuffix, ".txt").apply(&s3)),
            "hello"
        );
    }

    fn arr_of(items: Vec<Val>) -> Val {
        Val::arr(items)
    }

    fn obj_of(pairs: Vec<(&str, Val)>) -> Val {
        let mut m: indexmap::IndexMap<std::sync::Arc<str>, Val> =
            indexmap::IndexMap::with_capacity(pairs.len());
        for (k, v) in pairs {
            m.insert(std::sync::Arc::from(k), v);
        }
        Val::Obj(std::sync::Arc::new(m))
    }

    #[test]
    fn intersect_union_diff_sets() {
        let a = arr_of(vec![Val::Int(1), Val::Int(2), Val::Int(3), Val::Int(4)]);
        let b = vec![Val::Int(2), Val::Int(4), Val::Int(5)];
        assert_eq!(
            extract_arr_len(builtin_valvec(BuiltinMethod::Intersect, b.clone()).apply(&a)),
            2
        );
        assert_eq!(
            extract_arr_len(builtin_valvec(BuiltinMethod::Union, b.clone()).apply(&a)),
            5
        );
        assert_eq!(
            extract_arr_len(builtin_valvec(BuiltinMethod::Diff, b).apply(&a)),
            2
        );
    }

    #[test]
    fn get_path_has_path() {
        let inner = obj_of(vec![("city", Val::Str(std::sync::Arc::from("NYC")))]);
        let outer = obj_of(vec![("addr", inner)]);
        let r = builtin_str(BuiltinMethod::GetPath, "addr.city").apply(&outer);
        match r {
            StageOutput::Pass(cow) => match cow.into_owned() {
                Val::Str(s) => assert_eq!(s.as_ref(), "NYC"),
                other => panic!("got {:?}", other),
            },
            _ => panic!(),
        }
        assert!(extract_bool(
            builtin_str(BuiltinMethod::HasPath, "addr.city").apply(&outer)
        ));
        assert!(!extract_bool(
            builtin_str(BuiltinMethod::HasPath, "addr.zip").apply(&outer)
        ));
    }

    #[test]
    fn keys_values_entries_obj() {
        let o = obj_of(vec![("a", Val::Int(1)), ("b", Val::Int(2))]);
        assert_eq!(extract_arr_len(builtin0(BuiltinMethod::Keys).apply(&o)), 2);
        assert_eq!(
            extract_arr_len(builtin0(BuiltinMethod::Values).apply(&o)),
            2
        );
        assert_eq!(
            extract_arr_len(builtin0(BuiltinMethod::Entries).apply(&o)),
            2
        );
    }

    #[test]
    fn from_pairs_round_trip() {
        let o = obj_of(vec![("a", Val::Int(1)), ("b", Val::Int(2))]);
        let r = builtin0(BuiltinMethod::Entries).apply(&o);
        let pairs_val = match r {
            StageOutput::Pass(cow) => cow.into_owned(),
            _ => panic!(),
        };
        let r2 = builtin0(BuiltinMethod::FromPairs).apply(&pairs_val);
        match r2 {
            StageOutput::Pass(cow) => match cow.into_owned() {
                Val::Obj(m) => {
                    assert_eq!(m.len(), 2);
                    assert!(matches!(m.get("a"), Some(Val::Int(1))));
                    assert!(matches!(m.get("b"), Some(Val::Int(2))));
                }
                other => panic!("got {:?}", other),
            },
            _ => panic!(),
        }
    }

    #[test]
    fn invert_obj_swaps_kv() {
        let o = obj_of(vec![
            ("a", Val::Str(std::sync::Arc::from("X"))),
            ("b", Val::Str(std::sync::Arc::from("Y"))),
        ]);
        let r = builtin0(BuiltinMethod::Invert).apply(&o);
        match r {
            StageOutput::Pass(cow) => match cow.into_owned() {
                Val::Obj(m) => {
                    assert!(matches!(m.get("X"), Some(Val::Str(s)) if s.as_ref() == "a"));
                    assert!(matches!(m.get("Y"), Some(Val::Str(s)) if s.as_ref() == "b"));
                }
                other => panic!("got {:?}", other),
            },
            _ => panic!(),
        }
    }

    #[test]
    fn has_pick_omit_obj() {
        let o = obj_of(vec![
            ("a", Val::Int(1)),
            ("b", Val::Int(2)),
            ("c", Val::Int(3)),
        ]);
        assert!(extract_bool(builtin_str(BuiltinMethod::Has, "b").apply(&o)));
        assert!(!extract_bool(
            builtin_str(BuiltinMethod::Has, "z").apply(&o)
        ));
        let picked = builtin_strvec(BuiltinMethod::Pick, &["a", "c"]).apply(&o);
        match picked {
            StageOutput::Pass(cow) => match cow.into_owned() {
                Val::Obj(m) => assert_eq!(m.len(), 2),
                _ => panic!(),
            },
            _ => panic!(),
        }
        let omitted = builtin_strvec(BuiltinMethod::Omit, &["b"]).apply(&o);
        match omitted {
            StageOutput::Pass(cow) => match cow.into_owned() {
                Val::Obj(m) => assert_eq!(m.len(), 2),
                _ => panic!(),
            },
            _ => panic!(),
        }
    }

    #[test]
    fn array_len_works() {
        let a = arr_of(vec![Val::Int(1), Val::Int(2), Val::Int(3)]);
        let r = builtin0(BuiltinMethod::Len).apply(&a);
        match r {
            StageOutput::Pass(cow) => match cow.into_owned() {
                Val::Int(3) => {}
                other => panic!("got {:?}", other),
            },
            _ => panic!("expected Pass"),
        }
        let s = Val::Str(std::sync::Arc::from("café"));
        let r = builtin0(BuiltinMethod::Len).apply(&s);
        match r {
            StageOutput::Pass(cow) => match cow.into_owned() {
                Val::Int(4) => {}
                other => panic!("got {:?}", other),
            },
            _ => panic!("expected Pass"),
        }
    }

    #[test]
    fn compact_drops_nulls() {
        let a = arr_of(vec![
            Val::Int(1),
            Val::Null,
            Val::Int(2),
            Val::Null,
            Val::Int(3),
        ]);
        assert_eq!(
            extract_arr_len(builtin0(BuiltinMethod::Compact).apply(&a)),
            3
        );
    }

    #[test]
    fn flatten_one_level() {
        let a = arr_of(vec![
            arr_of(vec![Val::Int(1), Val::Int(2)]),
            arr_of(vec![Val::Int(3)]),
            Val::Int(4),
        ]);
        assert_eq!(
            extract_arr_len(builtin(BuiltinMethod::Flatten, BuiltinArgs::Usize(1)).apply(&a)),
            4
        );
    }

    #[test]
    fn enumerate_emits_pairs() {
        let a = arr_of(vec![Val::Int(10), Val::Int(20), Val::Int(30)]);
        let r = builtin0(BuiltinMethod::Enumerate).apply(&a);
        match r {
            StageOutput::Pass(cow) => match cow.into_owned() {
                Val::Arr(arr) => {
                    assert_eq!(arr.len(), 3);
                    if let Val::Obj(first) = &arr[0] {
                        assert!(matches!(first.get("index"), Some(Val::Int(0))));
                        assert!(matches!(first.get("value"), Some(Val::Int(10))));
                    } else {
                        panic!("expected object pair");
                    }
                }
                other => panic!("got {:?}", other),
            },
            _ => panic!("expected Pass"),
        }
    }

    #[test]
    fn pairwise_emits_adjacent() {
        let a = arr_of(vec![Val::Int(1), Val::Int(2), Val::Int(3), Val::Int(4)]);
        assert_eq!(
            extract_arr_len(builtin0(BuiltinMethod::Pairwise).apply(&a)),
            3
        );
    }

    #[test]
    fn chunk_window_split() {
        let a = arr_of(vec![
            Val::Int(1),
            Val::Int(2),
            Val::Int(3),
            Val::Int(4),
            Val::Int(5),
        ]);
        assert_eq!(
            extract_arr_len(builtin_usize(BuiltinMethod::Chunk, 2).apply(&a)),
            3
        );
        assert_eq!(
            extract_arr_len(builtin_usize(BuiltinMethod::Window, 3).apply(&a)),
            3
        );
    }

    #[test]
    fn nth_with_neg_index() {
        let a = arr_of(vec![Val::Int(10), Val::Int(20), Val::Int(30)]);
        let r0 = builtin_i64(BuiltinMethod::Nth, 0).apply(&a);
        match r0 {
            StageOutput::Pass(cow) => assert!(matches!(cow.into_owned(), Val::Int(10))),
            _ => panic!(),
        }
        let r1 = builtin_i64(BuiltinMethod::Nth, -1).apply(&a);
        match r1 {
            StageOutput::Pass(cow) => assert!(matches!(cow.into_owned(), Val::Int(30))),
            _ => panic!(),
        }
        let r2 = builtin_i64(BuiltinMethod::Nth, 99).apply(&a);
        match r2 {
            StageOutput::Pass(cow) => assert!(matches!(cow.into_owned(), Val::Null)),
            _ => panic!(),
        }
    }

    #[test]
    fn pad_left_right_indent_dedent() {
        let s = Val::Str(std::sync::Arc::from("hi"));
        assert_eq!(
            extract_str(builtin_pad(BuiltinMethod::PadLeft, 5, '-').apply(&s)),
            "---hi"
        );
        assert_eq!(
            extract_str(builtin_pad(BuiltinMethod::PadRight, 5, '-').apply(&s)),
            "hi---"
        );
        let lines = Val::Str(std::sync::Arc::from("foo\nbar"));
        assert_eq!(
            extract_str(builtin_usize(BuiltinMethod::Indent, 2).apply(&lines)),
            "  foo\n  bar"
        );
        let block = Val::Str(std::sync::Arc::from("    foo\n    bar"));
        assert_eq!(
            extract_str(builtin0(BuiltinMethod::Dedent).apply(&block)),
            "foo\nbar"
        );
    }

    #[test]
    fn index_of_and_matches() {
        let s = Val::Str(std::sync::Arc::from("hello world"));
        match builtin_str(BuiltinMethod::IndexOf, "world").apply(&s) {
            StageOutput::Pass(cow) => match cow.into_owned() {
                Val::Int(6) => {}
                other => panic!("got {:?}", other),
            },
            _ => panic!("expected Pass"),
        }
        match builtin_str(BuiltinMethod::LastIndexOf, "o").apply(&s) {
            StageOutput::Pass(cow) => match cow.into_owned() {
                Val::Int(7) => {}
                other => panic!("got {:?}", other),
            },
            _ => panic!("expected Pass"),
        }
        assert!(extract_bool(
            builtin_str(BuiltinMethod::Matches, "world").apply(&s)
        ));
    }

    #[test]
    fn base64_round_trip() {
        let s = Val::Str(std::sync::Arc::from("hello"));
        let enc = extract_str(builtin0(BuiltinMethod::ToBase64).apply(&s));
        let enc_val = Val::Str(std::sync::Arc::from(enc));
        let r = builtin0(BuiltinMethod::FromBase64).apply(&enc_val);
        match r {
            StageOutput::Pass(cow) => match cow.into_owned() {
                Val::Str(out) => assert_eq!(out.as_ref(), "hello"),
                other => panic!("expected Str, got {:?}", other),
            },
            _ => panic!("expected Pass"),
        }
    }

    #[test]
    fn upper_unicode_fallback() {
        let s = Val::Str(std::sync::Arc::from("café"));
        let stage = builtin0(BuiltinMethod::Upper);
        let got = stage.apply(&s);
        match got {
            StageOutput::Pass(Cow::Owned(Val::Str(out))) => {
                assert_eq!(out.as_ref(), "CAFÉ");
            }
            _ => panic!("expected uppercase unicode"),
        }
    }

    #[test]
    fn empty_input_finalises_to_default() {
        let arr: Vec<Val> = vec![];
        assert!(matches!(
            run_pipeline::<CountSink>(&arr, &Identity),
            Val::Int(0)
        ));
        assert!(matches!(
            run_pipeline::<SumSink>(&arr, &Identity),
            Val::Int(0)
        ));
        assert!(matches!(
            run_pipeline::<MinSink>(&arr, &Identity),
            Val::Null
        ));
        assert!(matches!(
            run_pipeline::<FirstSink>(&arr, &Identity),
            Val::Null
        ));
    }
}
