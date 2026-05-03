//! Zero-overhead stage composition using `Cow` borrow semantics.
//!
//! `Stage` transforms one element into a `StageOutput`. `Composed<A,B>` pairs
//! two stages so N stages fold into a single apply call at lower time, with
//! one virtual dispatch per element regardless of chain length. `Sink`
//! accumulates and finalises — a bounded set covers all legacy fused variants.
//! `run_pipeline` is the single generic outer loop parameterised by `Sink`.

use smallvec::SmallVec;
use std::borrow::{Borrow, Cow};

use crate::builtins::BuiltinCall;
use crate::chain_ir::PullDemand;
use crate::value::Val;


/// Per-element output of a `Stage::apply`. `Pass(Cow::Borrowed)` is the
/// hot path for filter and field-read (zero clone); `Cow::Owned` for
/// computed values; `Filtered` for dropped rows; `Many` for flat-map;
/// `Done` signals early termination to the outer loop.
pub enum StageOutput<'a> {
    /// The element passes through; borrowed when the stage did not transform it,
    /// owned when a new value was computed.
    Pass(Cow<'a, Val>),
    /// The element was rejected by this stage and should be skipped.
    Filtered,
    /// The stage expanded one element into multiple outputs (flat-map semantics).
    Many(SmallVec<[Cow<'a, Val>; 4]>),
    /// The stage requests early termination of the outer loop (e.g. `Take`).
    Done,
}


/// Per-element transformation. Implemented by every pipeline stage type;
/// composed monoidally by `Composed<A, B>`.
pub trait Stage {
    /// Apply the stage to a single input element, returning the appropriate
    /// `StageOutput` variant without consuming the input.
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a>;
}


impl<T: Stage + ?Sized> Stage for Box<T> {
    /// Delegate to the inner stage, allowing boxed trait objects to satisfy
    /// the `Stage` bound without an extra allocation path.
    #[inline]
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        (**self).apply(x)
    }
}


/// Identity stage — pass every element through unchanged. Neutral element for
/// `compose`, so the fold over an empty stage list is well-typed.
pub struct Identity;

impl Default for Identity {
    /// Construct the identity stage; equivalent to `Identity`.
    fn default() -> Self {
        Self
    }
}

impl Stage for Identity {
    /// Return the element as a zero-copy borrowed `Pass`, performing no work.
    #[inline]
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        StageOutput::Pass(Cow::Borrowed(x))
    }
}


/// A pipeline stage backed by a statically-dispatched `BuiltinCall`. Wraps the
/// call's `apply` return value into the `StageOutput` protocol.
pub struct BuiltinStage {
    /// The underlying builtin call that performs the actual transformation.
    call: BuiltinCall,
}

impl BuiltinStage {
    /// Construct a `BuiltinStage` from a pre-built `BuiltinCall`.
    pub fn new(call: BuiltinCall) -> Self {
        Self { call }
    }
}

impl Stage for BuiltinStage {
    /// Delegate to `BuiltinCall::apply`; maps `Some(v)` to `Pass(Owned)` and
    /// `None` (no-op / filter) to `Filtered`.
    #[inline]
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        match self.call.apply(x) {
            Some(v) => StageOutput::Pass(Cow::Owned(v)),
            None => StageOutput::Filtered,
        }
    }
}


/// Monoidal pairing of two stages. `Composed<A, B>` applies `A` first, then
/// feeds surviving elements into `B`, lifting ownership correctly across the
/// borrow-checker boundary when `A` returns an owned value.
pub struct Composed<A, B> {
    /// The first stage to apply to each element.
    pub a: A,
    /// The second stage, applied to every element that `a` did not filter out.
    pub b: B,
}

impl<A: Stage, B: Stage> Stage for Composed<A, B> {
    /// Apply `a`, then `b`, propagating `Filtered` and `Done` without
    /// allocating; upcasts borrowed `Cow` values to owned when needed.
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        match self.a.apply(x) {
            StageOutput::Pass(Cow::Borrowed(v)) => self.b.apply(v),
            StageOutput::Pass(Cow::Owned(v)) => {
                // `v` is owned; `self.b` borrows it locally, so any `Borrowed`
                // reference it returns must be re-owned before returning.
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
                            // Same ownership-lifting as the scalar owned case
                            // above; references from `b` cannot outlive `v`.
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


/// Accumulator over the elements that survive a pipeline stage chain.
/// Analogous to a fold; `init` creates the zero, `fold` updates it per element,
/// and `finalise` converts it to the output `Val`.
pub trait Sink {
    /// The type of the running accumulator maintained across all rows.
    type Acc;
    /// Return the zero accumulator before any rows are processed.
    fn init() -> Self::Acc;
    /// Update the accumulator with one passing element; consumes and returns it.
    fn fold(acc: Self::Acc, v: &Val) -> Self::Acc;
    /// Return `true` to signal the outer loop to stop pulling rows early.
    #[inline]
    fn done(_acc: &Self::Acc) -> bool {
        false
    }
    /// Convert the final accumulator into the result `Val`.
    fn finalise(acc: Self::Acc) -> Val;
}


/// Run `stages` over every element of `arr`, collecting results with sink `S`,
/// processing all rows (`PullDemand::All`).
pub fn run_pipeline<S: Sink>(arr: &[Val], stages: &dyn Stage) -> Val {
    run_pipeline_with_demand::<S>(arr, stages, PullDemand::All)
}


/// Run `stages` over `arr` with a `PullDemand` hint that lets callers cap
/// how many input rows are consumed or how many outputs are emitted.
pub fn run_pipeline_with_demand<S: Sink>(
    arr: &[Val],
    stages: &dyn Stage,
    demand: PullDemand,
) -> Val {
    run_pipeline_iter_with_demand::<S, _>(arr.iter(), stages, demand)
}

/// Run `stages` over an owned iterator of `Val` rows with a demand hint.
/// Useful when the source is not a contiguous slice (e.g. a chained iterator).
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

/// Core pipeline loop: iterate `rows`, apply `stages` to each element, feed
/// surviving values into `S::fold`, and honour `PullDemand` early-exit hints.
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


/// Sink that counts every passing element and returns `Val::Int(n)`.
pub struct CountSink;
impl Sink for CountSink {
    type Acc = i64;
    /// Initialise the counter to zero.
    #[inline]
    fn init() -> i64 {
        0
    }
    /// Increment the counter by one for each element, ignoring its value.
    #[inline]
    fn fold(acc: i64, _: &Val) -> i64 {
        acc + 1
    }
    /// Wrap the final count in `Val::Int`.
    #[inline]
    fn finalise(acc: i64) -> Val {
        Val::Int(acc)
    }
}

/// Sink that sums all numeric elements; accumulates integers and floats
/// separately, upgrading the result to `Val::Float` when any float is seen.
pub struct SumSink;
impl Sink for SumSink {
    /// `(int_sum, float_sum, has_float)` — `has_float` tracks whether any
    /// floating-point value was encountered so the final result type is correct.
    type Acc = (i64, f64, bool);
    /// Initialise to zero with no floats seen.
    #[inline]
    fn init() -> Self::Acc {
        (0, 0.0, false)
    }
    /// Add the element to the appropriate accumulator; booleans count as 0/1.
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
    /// Return `Val::Float` when any float was seen, otherwise `Val::Int`.
    fn finalise(acc: Self::Acc) -> Val {
        if acc.2 {
            Val::Float(acc.0 as f64 + acc.1)
        } else {
            Val::Int(acc.0)
        }
    }
}

/// Sink that returns the minimum numeric value seen, or `Val::Null` on empty input.
pub struct MinSink;
impl Sink for MinSink {
    /// `None` before any numeric element is seen, `Some(min)` thereafter.
    type Acc = Option<f64>;
    /// Initialise with no minimum yet.
    #[inline]
    fn init() -> Self::Acc {
        None
    }
    /// Update the running minimum, skipping non-numeric values.
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
    /// Return `Val::Int` when the result is a whole number, otherwise `Val::Float`;
    /// returns `Val::Null` when no numeric elements were processed.
    fn finalise(acc: Self::Acc) -> Val {
        match acc {
            Some(f) if f.fract() == 0.0 && f.abs() < (i64::MAX as f64) => Val::Int(f as i64),
            Some(f) => Val::Float(f),
            None => Val::Null,
        }
    }
}

/// Sink that returns the maximum numeric value seen, or `Val::Null` on empty input.
pub struct MaxSink;
impl Sink for MaxSink {
    /// `None` before any numeric element is seen, `Some(max)` thereafter.
    type Acc = Option<f64>;
    /// Initialise with no maximum yet.
    #[inline]
    fn init() -> Self::Acc {
        None
    }
    /// Update the running maximum, skipping non-numeric values.
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
    /// Return `Val::Int` when the result is a whole number, otherwise `Val::Float`;
    /// returns `Val::Null` when no numeric elements were processed.
    fn finalise(acc: Self::Acc) -> Val {
        match acc {
            Some(f) if f.fract() == 0.0 && f.abs() < (i64::MAX as f64) => Val::Int(f as i64),
            Some(f) => Val::Float(f),
            None => Val::Null,
        }
    }
}

/// Sink that computes the arithmetic mean of all numeric elements, returning
/// `Val::Null` when the input is empty or contains no numbers.
pub struct AvgSink;
impl Sink for AvgSink {
    /// `(running_sum, count)` — both reset to zero on `init`.
    type Acc = (f64, usize);
    /// Initialise to sum=0, count=0.
    #[inline]
    fn init() -> Self::Acc {
        (0.0, 0)
    }
    /// Accumulate numeric values; non-numeric elements are silently skipped.
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
    /// Divide sum by count; returns `Val::Null` when count is zero.
    fn finalise(acc: Self::Acc) -> Val {
        if acc.1 == 0 {
            Val::Null
        } else {
            Val::Float(acc.0 / acc.1 as f64)
        }
    }
}

/// Sink that returns the first element seen, then signals `done` to stop
/// pulling rows, giving O(1) behaviour with any upstream stage.
pub struct FirstSink;
impl Sink for FirstSink {
    /// `None` until the first element arrives, `Some(val)` thereafter.
    type Acc = Option<Val>;
    /// Initialise with no element captured yet.
    #[inline]
    fn init() -> Self::Acc {
        None
    }
    /// Store the first element; subsequent calls are no-ops once filled.
    fn fold(acc: Self::Acc, v: &Val) -> Self::Acc {
        if acc.is_some() {
            acc
        } else {
            Some(v.clone())
        }
    }
    /// Signal early termination as soon as one element has been captured.
    #[inline]
    fn done(acc: &Self::Acc) -> bool {
        acc.is_some()
    }
    /// Return the captured element, or `Val::Null` if the input was empty.
    fn finalise(acc: Self::Acc) -> Val {
        acc.unwrap_or(Val::Null)
    }
}

/// Sink that returns the last element seen; must consume the entire input
/// to determine which element is last.
pub struct LastSink;
impl Sink for LastSink {
    /// The most recently seen element, replaced on every new row.
    type Acc = Option<Val>;
    /// Initialise with no element captured yet.
    #[inline]
    fn init() -> Self::Acc {
        None
    }
    /// Overwrite the accumulator with the newest element unconditionally.
    fn fold(_acc: Self::Acc, v: &Val) -> Self::Acc {
        Some(v.clone())
    }
    /// Return the last element seen, or `Val::Null` if the input was empty.
    fn finalise(acc: Self::Acc) -> Val {
        acc.unwrap_or(Val::Null)
    }
}

/// Sink that collects all passing elements into a `Val::Arr`.
pub struct CollectSink;
impl Sink for CollectSink {
    /// The list of elements accumulated so far.
    type Acc = Vec<Val>;
    /// Initialise with an empty vector.
    #[inline]
    fn init() -> Self::Acc {
        Vec::new()
    }
    /// Append a clone of the element to the accumulator.
    fn fold(mut acc: Self::Acc, v: &Val) -> Self::Acc {
        acc.push(v.clone());
        acc
    }
    /// Wrap the collected vector in `Val::Arr`.
    fn finalise(acc: Self::Acc) -> Val {
        Val::Arr(std::sync::Arc::new(acc))
    }
}


/// Shared VM + environment context threaded through generic pipeline stages
/// that need to re-enter the evaluator (e.g. `GenericFilter`, `GenericMap`).
pub struct VmCtx {
    /// The VM instance used to execute compiled sub-programs.
    pub vm: crate::vm::VM,
    /// The evaluation environment (variables, current value) for sub-program execution.
    pub env: crate::context::Env,
}


/// A pipeline stage that evaluates a compiled boolean sub-program against each
/// element and passes it through only when the result is truthy.
pub struct GenericFilter {
    /// The compiled boolean predicate program to evaluate per row.
    pub prog: std::sync::Arc<crate::vm::Program>,
    /// Shared VM and environment context, wrapped for interior mutability.
    pub ctx: std::rc::Rc<std::cell::RefCell<VmCtx>>,
}

impl Stage for GenericFilter {
    /// Set `@` to `x`, run `prog`; return `Pass` when truthy, `Filtered` otherwise.
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        let mut c = self.ctx.borrow_mut();
        let VmCtx { vm, env } = &mut *c;

        // `filter_one` handles both scalar-bool and wrapped-array results.
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


/// A pipeline stage that maps each element through a compiled expression,
/// producing a new owned `Val` per row.
pub struct GenericMap {
    /// The compiled mapping expression program to evaluate per row.
    pub prog: std::sync::Arc<crate::vm::Program>,
    /// Shared VM and environment context, wrapped for interior mutability.
    pub ctx: std::rc::Rc<std::cell::RefCell<VmCtx>>,
}

impl Stage for GenericMap {
    /// Set `@` to `x`, run `prog`, and return the result as `Pass(Owned)`;
    /// evaluation errors degrade to `Filtered` rather than propagating.
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


/// A pipeline stage that maps each element through a compiled expression and
/// then flattens the resulting array into individual rows.
pub struct GenericFlatMap {
    /// The compiled flat-mapping expression program to evaluate per row.
    pub prog: std::sync::Arc<crate::vm::Program>,
    /// Shared VM and environment context, wrapped for interior mutability.
    pub ctx: std::rc::Rc<std::cell::RefCell<VmCtx>>,
}

impl Stage for GenericFlatMap {
    /// Set `@` to `x`, run `prog`, then expand the result into `Many`; errors
    /// or non-iterable results degrade to `Filtered`.
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

        // `into_vals` extracts the element vector without cloning the `Arc`.
        let result: StageOutput<'a> = match owned.into_vals() {
            Ok(items) => many_from_owned_vals(items),
            Err(_) => StageOutput::Filtered,
        };
        result
    }
}


/// A pipeline stage that keeps only object elements whose named field equals a
/// compile-time literal value; avoids a VM round-trip for simple equality predicates.
pub struct FilterFieldEqLit {
    /// The field name to look up on each object element.
    pub field: std::sync::Arc<str>,
    /// The literal value the field must equal for the row to pass.
    pub target: Val,
}

impl Stage for FilterFieldEqLit {
    /// Pass `x` only when it is an object, has `field`, and `field == target`.
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


/// A pipeline stage that extracts a single named field from each object element,
/// discarding elements that are not objects or lack the field.
pub struct MapField {
    /// The name of the field to extract from each object.
    pub field: std::sync::Arc<str>,
}

impl Stage for MapField {
    /// Return a zero-copy borrowed reference to `x.field`, or `Filtered` when
    /// `x` is not an object or the field is absent.
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        if let Val::Obj(m) = x {
            if let Some(v) = m.get(self.field.as_ref()) {
                return StageOutput::Pass(Cow::Borrowed(v));
            }
        }
        StageOutput::Filtered
    }
}


/// A pipeline stage that extracts a named field from each object and expands it
/// into individual rows when the field value is itself an array.
pub struct FlatMapField {
    /// The name of the field whose value should be flattened into rows.
    pub field: std::sync::Arc<str>,
}

impl Stage for FlatMapField {
    /// Retrieve `x.field` and flatten it; produces `Filtered` when the field is
    /// absent, not an object, or the field value is not iterable.
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        if let Val::Obj(m) = x {
            if let Some(v) = m.get(self.field.as_ref()) {
                return flatten_iterable(v);
            }
        }
        StageOutput::Filtered
    }
}


/// A pipeline stage that follows a sequence of field keys through nested objects
/// and then flattens the final value into individual rows.
pub struct FlatMapFieldChain {
    /// Ordered sequence of field keys forming the nested path to follow.
    pub keys: std::sync::Arc<[std::sync::Arc<str>]>,
}

impl Stage for FlatMapFieldChain {
    /// Traverse `keys` depth-first; flatten the terminal value, or return
    /// `Filtered` when any intermediate step is missing or non-object.
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


/// Expand an iterable `Val` into a `Many` output, borrowing the inner slice
/// when possible to avoid cloning. Returns `Filtered` for non-iterable values.
#[inline]
fn flatten_iterable<'a>(v: &'a Val) -> StageOutput<'a> {
    match v.as_vals() {
        Some(items) => many_from_vals(items, true),
        None => StageOutput::Filtered,
    }
}

/// Convert a `Cow<[Val]>` into `StageOutput::Many`, borrowing each element
/// when `allow_borrow` is `true` and the slice is borrowed, otherwise owning.
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

/// Convert an owned `Vec<Val>` into `StageOutput::Many`, wrapping each element
/// in `Cow::Owned`. Returns `Filtered` when the vector is empty.
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


/// A pipeline stage that traverses a fixed sequence of field keys through
/// nested objects and returns the terminal value as a zero-copy borrow.
pub struct MapFieldChain {
    /// Ordered sequence of field keys forming the nested path to follow.
    pub keys: std::sync::Arc<[std::sync::Arc<str>]>,
}

impl Stage for MapFieldChain {
    /// Follow `keys` depth-first; return a borrowed reference to the terminal
    /// value, or `Filtered` when any step is missing or non-object.
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


/// A pipeline stage that passes at most `remaining` elements, then signals
/// `Done` to terminate the outer loop — enables fused `sort | take(k)`.
pub struct Take {
    /// Interior-mutable counter of elements still allowed to pass through.
    pub remaining: std::cell::Cell<usize>,
}

impl Stage for Take {
    /// Decrement the counter and pass the element; return `Done` when exhausted.
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        let r = self.remaining.get();
        if r == 0 {
            return StageOutput::Done;
        }
        self.remaining.set(r - 1);
        StageOutput::Pass(Cow::Borrowed(x))
    }
}


/// A pipeline stage that discards the first `remaining` elements, then passes
/// every subsequent element through unchanged.
pub struct Skip {
    /// Interior-mutable counter of elements still to be dropped.
    pub remaining: std::cell::Cell<usize>,
}

impl Stage for Skip {
    /// Filter while `remaining > 0`, decrementing; pass once the quota is exhausted.
    fn apply<'a>(&self, x: &'a Val) -> StageOutput<'a> {
        let r = self.remaining.get();
        if r > 0 {
            self.remaining.set(r - 1);
            return StageOutput::Filtered;
        }
        StageOutput::Pass(Cow::Borrowed(x))
    }
}


/// Describes how a barrier operation (`sort_by`, `unique_by`, `group_by`) should
/// extract the comparison or grouping key from each element.
pub enum KeySource {
    /// Use the element itself as the key.
    None,
    /// Extract a single top-level field from the element.
    Field(std::sync::Arc<str>),
    /// Follow a dot-separated chain of fields to reach the key.
    Chain(std::sync::Arc<[std::sync::Arc<str>]>),
}

impl KeySource {
    /// Extract the key from `v` according to the source variant, returning
    /// `Val::Null` when a required field is absent or `v` is not an object.
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


/// Barrier operation: reverse the entire buffered row set in-place, returning
/// the reversed vector.
pub fn barrier_reverse(buf: Vec<Val>) -> Vec<Val> {
    let mut buf = buf;
    buf.reverse();
    buf
}


/// Barrier operation: stable-sort the buffered rows by the key produced by
/// `key`, using `cmp_val` for ordering.
pub fn barrier_sort(buf: Vec<Val>, key: &KeySource) -> Vec<Val> {
    let mut indexed: Vec<(Val, Val)> = buf.into_iter().map(|v| (key.extract(&v), v)).collect();
    indexed.sort_by(|a, b| cmp_val(&a.0, &b.0));
    indexed.into_iter().map(|(_, v)| v).collect()
}


/// Barrier operation: return the `k` smallest rows by `key` using a partial
/// sort, cheaper than sorting the full buffer when `k << buf.len()`.
pub fn barrier_top_k(buf: Vec<Val>, key: &KeySource, k: usize) -> Vec<Val> {
    barrier_top_or_bottom_k(buf, key, k, false)
}


/// Barrier operation: return the `k` largest rows by `key` using a partial
/// sort, cheaper than sorting the full buffer when `k << buf.len()`.
pub fn barrier_bottom_k(buf: Vec<Val>, key: &KeySource, k: usize) -> Vec<Val> {
    barrier_top_or_bottom_k(buf, key, k, true)
}

/// Shared implementation for `barrier_top_k` and `barrier_bottom_k`; delegates
/// to `pipeline::bounded_sort_by_key_cmp` with the appropriate `StageStrategy`.
fn barrier_top_or_bottom_k(buf: Vec<Val>, key: &KeySource, k: usize, largest: bool) -> Vec<Val> {
    let strategy = if largest {
        crate::pipeline::StageStrategy::SortBottomK(k)
    } else {
        crate::pipeline::StageStrategy::SortTopK(k)
    };
    crate::pipeline::bounded_sort_by_key_cmp(buf, false, strategy, |v| Ok(key.extract(v)), cmp_val)
        .unwrap_or_default()
}


/// Barrier operation: deduplicate rows, keeping the first occurrence of each
/// unique key produced by `key`.
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


/// Barrier operation: group rows by the string form of each row's key,
/// returning a `Val::Obj` whose values are `Val::Arr` lists of grouped rows.
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


/// A `Val`-based hash wrapper used as the deduplication key inside `barrier_unique_by`.
/// Stores the key in a canonical `KeyRepr` to enable `HashSet` membership tests.
#[derive(Eq, PartialEq, Hash)]
struct KeyHash(KeyRepr);

/// The canonical form of a `Val` used as a hash key. Floats are stored as
/// raw `u64` bits so that equal floating-point values produce equal hashes.
#[derive(Eq, PartialEq, Hash)]
enum KeyRepr {
    /// Null key, equal to all other nulls.
    Null,
    /// Boolean key.
    Bool(bool),
    /// Integer key.
    Int(i64),
    /// Float stored as raw bits for hash-equality; NaN maps to a distinct value.
    Float(u64),
    /// String key, covering all string-like `Val` variants.
    Str(String),
}

impl From<Val> for KeyHash {
    /// Convert a `Val` into a `KeyHash` by mapping to `KeyRepr`; compound
    /// values (`Arr`, `Obj`) are serialised via `DisplayKey` as a best-effort fallback.
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

/// Minimal `Display` adapter for `Val` used when a complex value must be
/// serialised as a group-by or dedup key string.
struct DisplayKey<'a>(&'a Val);
impl<'a> std::fmt::Display for DisplayKey<'a> {
    /// Format the value as a terse string; compound types render as `"<complex>"`.
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


/// Total order over `Val` used by all barrier sorts. `Null` sorts smallest;
/// cross-type numeric comparisons promote integers to `f64`. Incomparable
/// pairs (e.g. two objects) return `Equal` to preserve insertion order.
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


/// Structural equality over scalar `Val` variants used by `FilterFieldEqLit`.
/// Cross-type numeric comparison (`Int` vs `Float`) is supported; compound
/// types always return `false`.
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
        // filter a==1, map b, count → 2
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
        // rows a==1: b=10, b=30 → sum=40
        assert!(matches!(out, Val::Int(40)));
    }

    #[test]
    fn collect_map_field_chain() {
        // u.a.c for each row
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
        // avg = (5+2+9+3)/4 = 4.75
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

        // reverse then map
        assert_eq!(
            j.collect("$.rows.reverse().map(price)").unwrap(),
            json!([40, 20, 10, 30])
        );

        // unique_by city → 2 cities
        assert_eq!(
            j.collect("$.rows.unique_by(city).count()").unwrap(),
            json!(2)
        );

        // sort_by price ascending → first is the cheapest
        assert_eq!(
            j.collect("$.rows.sort_by(price).first()").unwrap(),
            json!({"city": "NYC", "price": 10})
        );
    }

    #[test]
    fn step3d_phase1_sort_topk() {
        // Verify that sort_by + take(k) uses the fused top-k path and returns
        // correctly sorted results.
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

        // take(2) after sort → two smallest
        assert_eq!(
            j.collect("$.rows.sort_by(v).take(2)").unwrap(),
            json!([{"id": 1, "v": 10}, {"id": 2, "v": 20}])
        );
        // first after sort → smallest
        assert_eq!(
            j.collect("$.rows.sort_by(v).first()").unwrap(),
            json!({"id": 1, "v": 10})
        );

        // last after sort → largest
        assert_eq!(
            j.collect("$.rows.sort_by(v).last()").unwrap(),
            json!({"id": 5, "v": 50})
        );
    }

    #[test]
    fn step3d_phase5_indexed_dispatch_correctness() {
        // Ensure indexed dispatch (first/last after map) returns the right elements.
        use serde_json::json;
        let doc = json!({
            "books": [
                {"price": 10},
                {"price": 20},
                {"price": 30},
            ]
        });
        let j = crate::Jetro::new(doc);
        // first
        assert_eq!(j.collect("$.books.map(price).first()").unwrap(), json!(10));
        // last
        assert_eq!(j.collect("$.books.map(price).last()").unwrap(), json!(30));
        // first again (cache check)
        assert_eq!(j.collect("$.books.map(price).first()").unwrap(), json!(10));
    }

    #[test]
    fn step3d_ext_a2_compiled_map() {
        // Compiled map with chained method calls on @.
        use serde_json::json;
        let doc = json!({ "records": [
            { "text": "alice,smith,42" },
            { "text": "bob,jones,17"   },
            { "text": "carol,xx,99"    },
        ]});
        let j = crate::Jetro::new(doc);

        // first segment
        assert_eq!(
            j.collect("$.records.map(@.text.split(\",\").first())")
                .unwrap(),
            json!(["alice", "bob", "carol"])
        );
        // last segment
        assert_eq!(
            j.collect("$.records.map(@.text.split(\",\").last())")
                .unwrap(),
            json!(["42", "17", "99"])
        );

        // count segments
        assert_eq!(
            j.collect("$.records.map(@.text.split(\",\").count())")
                .unwrap(),
            json!([3, 3, 3])
        );
    }

    #[test]
    fn step3d_ext_split_slice_lifted() {
        // split on a scalar string, then slice/aggregate on the resulting array.
        use serde_json::json;
        let doc = json!({ "s": "a,b,c,d,e" });
        let j = crate::Jetro::new(doc);

        // plain split
        assert_eq!(
            j.collect("$.s.split(\",\")").unwrap(),
            json!(["a", "b", "c", "d", "e"])
        );
        // count
        assert_eq!(j.collect("$.s.split(\",\").count()").unwrap(), json!(5));
        // first
        assert_eq!(j.collect("$.s.split(\",\").first()").unwrap(), json!("a"));
        // last
        assert_eq!(j.collect("$.s.split(\",\").last()").unwrap(), json!("e"));
    }

    #[test]
    fn step3d_phase3_filter_reorder() {
        // Two consecutive Filter stages should be fused/reordered into one by the planner.
        use crate::ast::BinOp;
        use crate::pipeline::{plan_with_kernels, BodyKernel, Sink, Stage};
        use std::sync::Arc;
        let dummy = Arc::new(crate::vm::Program::new(Vec::new(), ""));
        let stages = vec![
            // first filter
            Stage::Filter(
                Arc::clone(&dummy),
                crate::builtins::BuiltinViewStage::Filter,
            ),
            // second filter
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
        // both filters should be merged into a single composed stage
        assert_eq!(p.stages.len(), 1);
        assert!(matches!(p.stages[0], Stage::Filter(_, _)));
    }

    #[test]
    fn step3d_phase3_filter_reorder_correctness() {
        // Two chained filters should yield the same results as either order.
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
        // b>15 AND tag=="x" → rows 3,5 → count=2
        assert_eq!(
            j.collect("$.rows.filter(b > 15).filter(tag == \"x\").count()")
                .unwrap(),
            json!(2)
        );
        // sum of b for those rows → 30+50=80
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
        // take(5).take(3) → take(3)
        let p = plan(
            vec![
                Stage::Take(5, BuiltinViewStage::Take, BuiltinStageMerge::UsizeMin),
                Stage::Take(3, BuiltinViewStage::Take, BuiltinStageMerge::UsizeMin),
            ],
            Sink::Collect,
        );
        assert_eq!(p.stages.len(), 1);
        assert!(matches!(p.stages[0], Stage::Take(3, _, _)));

        // skip(2).skip(3) → skip(5)
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

        // reverse().reverse() → empty
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
        // Filter + First → EarlyExit
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
        // Map + Sum reducer → PullLoop
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
                    predicate_expr: None,
                    projection_expr: None,
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

        // Sort + First → SortTopK(1)
        let stages = vec![Stage::Sort(SortSpec::keyed(Arc::clone(&dummy_prog), false))];
        let strats = compute_strategies(&stages, &first_sink);
        assert!(matches!(strats[0], StageStrategy::SortTopK(1)));

        // Sort + Take(5) + Collect → SortTopK(5)
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

        // Sort + Sum reducer → Default (full materialise required)
        let stages = vec![Stage::Sort(SortSpec::identity())];
        let strats = compute_strategies(
            &stages,
            &Sink::Reducer(ReducerSpec {
                op: ReducerOp::Numeric(NumOp::Sum),
                predicate: None,
                projection: None,
                predicate_expr: None,
                projection_expr: None,
            }),
        );
        assert!(matches!(strats[0], StageStrategy::Default));

        // Sort + Filter + First → SortUntilOutput(1)
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
        // Compiled lambdas in map/filter go through the VM path.
        use serde_json::json;

        let doc = json!({
            "rows": [
                {"qty": 2, "price": 10},
                {"qty": 3, "price": 20},
                {"qty": 1, "price": 30},
            ]
        });

        let j = crate::Jetro::new(doc);

        // computed field in map
        assert_eq!(
            j.collect("$.rows.map(qty * price).sum()").unwrap(),
            json!(110)
        );

        // filter with expression
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
