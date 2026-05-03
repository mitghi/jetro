//! `SinkAccumulator` — stateful wrapper around a `Sink` for the legacy exec
//! path. Receives one element at a time via `step` and materialises the final
//! result via `finish`. Mirrors the `Sink` trait from `composed.rs` but
//! without the generic type parameter, for compatibility with the older paths.

use crate::{
    builtins::{BuiltinSelectionPosition, BuiltinSinkAccumulator},
    value::Val,
};

use super::{ReducerAccumulator, Sink};

/// Stateful wrapper that accumulates one pipeline element at a time on behalf of a `Sink`.
///
/// Owned by the legacy exec path; the composed path uses typed `composed::Sink` impls instead.
pub(crate) struct SinkAccumulator<'a> {
    /// Reference to the sink descriptor that governs accumulation strategy.
    sink: &'a Sink,
    /// Buffer for `Sink::Collect` — elements appended as they arrive.
    collect: Vec<Val>,
    /// Running numeric reducer state, present only for reducer-bearing sinks.
    reducer: Option<ReducerAccumulator>,
    /// Holds the first observed item for `SelectOne(First)` sinks.
    first: Option<Val>,
    /// Holds the most recently observed item for `SelectOne(Last)` sinks.
    last: Option<Val>,
    /// HyperLogLog register array for approximate-distinct-count sinks.
    hll: [u8; HLL_M],
}

impl<'a> SinkAccumulator<'a> {
    /// Creates a fresh accumulator wired to `sink`, pre-initialising the reducer if needed.
    pub(crate) fn new(sink: &'a Sink) -> Self {
        Self {
            sink,
            collect: Vec::new(),
            reducer: sink.reducer_spec().map(ReducerAccumulator::new),
            first: None,
            last: None,
            hll: [0; HLL_M],
        }
    }

    /// Forwards `item` to the appropriate observation method based on sink kind.
    ///
    /// Returns `true` when the sink signals early termination (first-select after capture).
    pub(crate) fn push(&mut self, item: Val) -> bool {
        if let Some(spec) = self.sink.builtin_sink_spec() {
            return self.observe_builtin(spec.accumulator, item);
        }
        match self.sink {
            Sink::Collect => self.observe_collect(item),
            Sink::Reducer(_) => self.observe_reducer(&item),
            Sink::ApproxCountDistinct => self.observe_approx_distinct(&item),
            Sink::Terminal(_) => {}
        }
        false
    }

    /// Dispatches `item` to the correct builtin accumulator without lazy closures.
    ///
    /// Convenience wrapper over `observe_builtin_lazy` for callers that already own the value.
    pub(crate) fn observe_builtin(
        &mut self,
        accumulator: BuiltinSinkAccumulator,
        item: Val,
    ) -> bool {
        self.observe_builtin_lazy(accumulator, || item, || None, || None)
            .unwrap_or(false)
    }

    /// Lazy variant of `observe_builtin` that defers value materialisation to closures.
    ///
    /// `materialize_item` is called at most once; `materialize_numeric` and `hash_key` are
    /// called only when the accumulator kind actually needs them, avoiding redundant work.
    pub(crate) fn observe_builtin_lazy<F, N, K>(
        &mut self,
        accumulator: BuiltinSinkAccumulator,
        materialize_item: F,
        materialize_numeric: N,
        hash_key: K,
    ) -> Option<bool>
    where
        F: FnOnce() -> Val,
        N: FnOnce() -> Option<Val>,
        K: FnOnce() -> Option<String>,
    {
        match accumulator {
            BuiltinSinkAccumulator::Count => {
                self.observe_count();
                Some(false)
            }
            BuiltinSinkAccumulator::Numeric => {
                let numeric_item = materialize_numeric().unwrap_or_else(materialize_item);
                self.observe_numeric(&numeric_item);
                Some(false)
            }
            BuiltinSinkAccumulator::ApproxDistinct => {
                if let Some(key) = hash_key() {
                    self.observe_approx_distinct_key(&key);
                } else {
                    self.observe_approx_distinct(&materialize_item());
                }
                Some(false)
            }
            BuiltinSinkAccumulator::SelectOne(BuiltinSelectionPosition::First) => {
                Some(self.observe_first(materialize_item()))
            }
            BuiltinSinkAccumulator::SelectOne(BuiltinSelectionPosition::Last) => {
                self.observe_last(materialize_item());
                Some(false)
            }
        }
    }

    /// Appends `item` to the collect buffer for later array construction.
    pub(crate) fn observe_collect(&mut self, item: Val) {
        self.collect.push(item);
    }

    /// Forwards `item` into the numeric reducer accumulator if one is present.
    pub(crate) fn observe_reducer(&mut self, item: &Val) {
        if let Some(reducer) = &mut self.reducer {
            reducer.push(item);
        }
    }

    /// Increments the count reducer by pushing a synthetic null item.
    pub(crate) fn observe_count(&mut self) {
        self.observe_reducer(&Val::Null);
    }

    /// Routes `item` into the numeric reducer (sum/min/max/avg path).
    pub(crate) fn observe_numeric(&mut self, item: &Val) {
        self.observe_reducer(item);
    }

    /// Captures `item` as the first-seen value; returns `true` on successful capture.
    ///
    /// Subsequent calls after the first are no-ops that return `false`.
    pub(crate) fn observe_first(&mut self, item: Val) -> bool {
        if self.first.is_none() {
            self.first = Some(item);
            true
        } else {
            false
        }
    }

    /// Overwrites the last-seen slot unconditionally; the final value wins.
    pub(crate) fn observe_last(&mut self, item: Val) {
        self.last = Some(item);
    }

    /// Hashes `item` into the HyperLogLog registers for cardinality estimation.
    pub(crate) fn observe_approx_distinct(&mut self, item: &Val) {
        hll_observe(&mut self.hll, item);
    }

    /// Hashes a pre-serialised string key into the HyperLogLog registers.
    pub(crate) fn observe_approx_distinct_key(&mut self, key: &str) {
        hll_observe_key(&mut self.hll, key);
    }

    /// Feeds an already-projected numeric value directly into the reducer, skipping re-evaluation.
    pub(crate) fn push_projected_numeric(&mut self, numeric_item: &Val) {
        self.observe_reducer(numeric_item);
    }

    /// Consumes the accumulator and produces the final `Val` according to the sink kind.
    ///
    /// When `unwrap_single_collect_obj` is `true` and exactly one object was collected, that
    /// object is returned unwrapped rather than wrapped in a single-element array.
    pub(crate) fn finish(self, unwrap_single_collect_obj: bool) -> Val {
        if let Some(spec) = self.sink.builtin_sink_spec() {
            return self.finish_builtin(spec.accumulator);
        }
        match self.sink {
            Sink::Collect => {
                if unwrap_single_collect_obj
                    && self.collect.len() == 1
                    && matches!(self.collect[0], Val::Obj(_))
                {
                    self.collect.into_iter().next().unwrap()
                } else {
                    Val::arr(self.collect)
                }
            }
            Sink::Reducer(_) => self
                .reducer
                .expect("reducer sinks construct reducer")
                .finish(),
            Sink::ApproxCountDistinct => Val::Int(hll_estimate(&self.hll) as i64),
            Sink::Terminal(_) => Val::Null,
        }
    }

    /// Finalises state for a builtin-registered sink, delegating to the reducer or HLL as needed.
    fn finish_builtin(self, accumulator: BuiltinSinkAccumulator) -> Val {
        match accumulator {
            BuiltinSinkAccumulator::Count | BuiltinSinkAccumulator::Numeric => self
                .reducer
                .expect("reducer sinks construct reducer")
                .finish(),
            BuiltinSinkAccumulator::ApproxDistinct => Val::Int(hll_estimate(&self.hll) as i64),
            BuiltinSinkAccumulator::SelectOne(BuiltinSelectionPosition::First) => {
                self.first.unwrap_or(Val::Null)
            }
            BuiltinSinkAccumulator::SelectOne(BuiltinSelectionPosition::Last) => {
                self.last.unwrap_or(Val::Null)
            }
        }
    }
}

/// HyperLogLog precision parameter; 2^12 = 4096 registers.
const HLL_P: u32 = 12;
/// Number of HyperLogLog registers derived from `HLL_P`.
const HLL_M: usize = 1 << HLL_P;

/// Hashes `key` to a `u64` using a process-stable random state seeded once at startup.
#[inline]
fn hll_hash_key(key: &str) -> u64 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};

    static STATE: std::sync::OnceLock<RandomState> = std::sync::OnceLock::new();
    let bs = STATE.get_or_init(RandomState::new);
    let mut h = bs.build_hasher();
    h.write(key.as_bytes());
    h.finish()
}

/// Serialises `v` to a key string and updates the HLL registers.
fn hll_observe(reg: &mut [u8; HLL_M], v: &Val) {
    use crate::util::val_to_key;
    hll_observe_key(reg, &val_to_key(v));
}

/// Updates the HLL register for `key` using the leading-zeros position encoding.
fn hll_observe_key(reg: &mut [u8; HLL_M], key: &str) {
    let h = hll_hash_key(key);
    let idx = (h >> (64 - HLL_P)) as usize;
    let w = (h << HLL_P) | (1u64 << (HLL_P - 1));
    let lz = w.leading_zeros() as u8 + 1;
    if lz > reg[idx] {
        reg[idx] = lz;
    }
}

/// Computes the HyperLogLog cardinality estimate from the register array.
///
/// Applies small-range linear counting correction when many registers are zero.
fn hll_estimate(reg: &[u8; HLL_M]) -> f64 {
    let mut z: f64 = 0.0;
    let mut zeros: usize = 0;
    for &r in reg.iter() {
        z += 1.0 / (1u64 << r) as f64;
        if r == 0 {
            zeros += 1;
        }
    }
    let m = HLL_M as f64;
    let alpha_m = 0.7213 / (1.0 + 1.079 / m);
    let raw = alpha_m * m * m / z;
    if raw <= 2.5 * m && zeros > 0 {
        return m * (m / zeros as f64).ln();
    }
    raw
}
