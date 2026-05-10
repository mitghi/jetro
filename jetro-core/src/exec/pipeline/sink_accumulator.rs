//! Sink accumulator: stateful wrapper around a `Sink` for the legacy execution path.
//! Receives one element at a time via `push` and materialises the final result via `finish`.
//! Mirrors the composed `Sink` contract but without a generic type parameter, for compatibility
//! with older execution paths.

use std::collections::VecDeque;

use crate::{
    builtins::{BuiltinSelectionPosition, BuiltinSinkAccumulator},
    data::{context::EvalError, value::Val},
};

use super::{cmp_val_total, MembershipSinkOp, PredicateSinkOp, ReducerAccumulator, Sink};

/// Stateful wrapper that accumulates one pipeline element at a time on behalf of a `Sink`.
pub(crate) struct SinkAccumulator<'a> {
    // governs which accumulation strategy is applied
    sink: &'a Sink,
    // elements appended as they arrive for Sink::Collect
    collect: Vec<Val>,
    // present only for reducer-bearing sinks
    reducer: Option<ReducerAccumulator>,
    // first observed item for SelectOne(First) sinks
    first: Option<Val>,
    // most recently observed item for SelectOne(Last) sinks
    last: Option<Val>,
    // nth observed item for Sink::Nth
    nth: Option<Val>,
    nth_seen: usize,
    select_many: VecDeque<Val>,
    predicate_seen: usize,
    predicate_matched: Option<usize>,
    predicate_value: Option<Val>,
    predicate_all: bool,
    predicate_indices: Vec<i64>,
    membership_seen: usize,
    membership_matched: Option<usize>,
    membership_indices: Vec<i64>,
    arg_extreme_key: Option<Val>,
    arg_extreme_value: Option<Val>,
    // HyperLogLog register array for approximate-distinct-count sinks
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
            nth: None,
            nth_seen: 0,
            select_many: VecDeque::new(),
            predicate_seen: 0,
            predicate_matched: None,
            predicate_value: None,
            predicate_all: true,
            predicate_indices: Vec::new(),
            membership_seen: 0,
            membership_matched: None,
            membership_indices: Vec::new(),
            arg_extreme_key: None,
            arg_extreme_value: None,
            hll: [0; HLL_M],
        }
    }

    /// Forwards `item` to the appropriate observation method; returns `true` on early termination.
    pub(crate) fn push(&mut self, item: Val) -> bool {
        if let Some(spec) = self.sink.builtin_sink_spec() {
            return self.observe_builtin(spec.accumulator, item);
        }
        match self.sink {
            Sink::Collect => {
                self.observe_collect(item);
                false
            }
            Sink::Reducer(_) => {
                self.observe_reducer(&item);
                false
            }
            Sink::ApproxCountDistinct => {
                self.observe_approx_distinct(&item);
                false
            }
            Sink::Predicate(_) => false,
            Sink::Membership(_) => false,
            Sink::ArgExtreme(_) => false,
            Sink::SelectMany { n, from_end } => self.observe_select_many(*n, *from_end, item),
            Sink::Nth(idx) => self.observe_nth(*idx, item),
            Sink::Terminal(_) => false,
        }
    }

    /// Dispatches `item` to the correct builtin accumulator; convenience wrapper over `observe_builtin_lazy`.
    pub(crate) fn observe_builtin(
        &mut self,
        accumulator: BuiltinSinkAccumulator,
        item: Val,
    ) -> bool {
        self.observe_builtin_lazy(accumulator, || item, || None, || None)
            .unwrap_or(false)
    }

    /// Lazy variant of `observe_builtin`; closures are called only when the accumulator kind needs them.
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

    /// Captures `item` as the first-seen value and returns `true`; subsequent calls return `false`.
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

    /// Captures the nth observed value, returning true once it is found.
    pub(crate) fn observe_nth(&mut self, idx: usize, item: Val) -> bool {
        self.observe_nth_lazy(idx, || item)
    }

    /// Lazy nth variant; materialises only the selected item.
    pub(crate) fn observe_nth_lazy<F>(&mut self, idx: usize, materialize_item: F) -> bool
    where
        F: FnOnce() -> Val,
    {
        if self.nth.is_some() {
            return true;
        }
        if self.nth_seen == idx {
            self.nth = Some(materialize_item());
            true
        } else {
            self.nth_seen += 1;
            false
        }
    }

    /// Captures a bounded prefix or suffix. Prefix selection stops once full; suffix selection
    /// retains only the latest `n` rows.
    pub(crate) fn observe_select_many(&mut self, n: usize, from_end: bool, item: Val) -> bool {
        self.observe_select_many_lazy(n, from_end, false, || item)
    }

    /// Lazy bounded prefix/suffix selector. `prepend` is used when a reverse source is
    /// satisfying a suffix demand so final output remains in semantic input order.
    pub(crate) fn observe_select_many_lazy<F>(
        &mut self,
        n: usize,
        from_end: bool,
        prepend: bool,
        materialize_item: F,
    ) -> bool
    where
        F: FnOnce() -> Val,
    {
        if n == 0 {
            return true;
        }
        let item = materialize_item();
        if prepend {
            if self.select_many.len() == n {
                self.select_many.pop_back();
            }
            self.select_many.push_front(item);
            return self.select_many.len() >= n;
        }
        if from_end {
            if self.select_many.len() == n {
                self.select_many.pop_front();
            }
            self.select_many.push_back(item);
            false
        } else {
            self.select_many.push_back(item);
            self.select_many.len() >= n
        }
    }

    /// Updates a predicate terminal sink with one predicate result and row value.
    pub(crate) fn observe_predicate_item(
        &mut self,
        op: PredicateSinkOp,
        matched: bool,
        item: Val,
    ) -> Result<bool, EvalError> {
        self.observe_predicate_lazy(op, matched, || item)
    }

    /// Lazy predicate sink variant; materialises the row only for sinks that store it.
    pub(crate) fn observe_predicate_lazy<F>(
        &mut self,
        op: PredicateSinkOp,
        matched: bool,
        materialize_item: F,
    ) -> Result<bool, EvalError>
    where
        F: FnOnce() -> Val,
    {
        match op {
            PredicateSinkOp::Any => {
                if matched {
                    self.predicate_matched = Some(self.predicate_seen);
                    Ok(true)
                } else {
                    self.predicate_seen += 1;
                    Ok(false)
                }
            }
            PredicateSinkOp::All => {
                self.predicate_seen += 1;
                if matched {
                    Ok(false)
                } else {
                    self.predicate_all = false;
                    Ok(true)
                }
            }
            PredicateSinkOp::FindIndex => {
                if matched {
                    self.predicate_matched = Some(self.predicate_seen);
                    Ok(true)
                } else {
                    self.predicate_seen += 1;
                    Ok(false)
                }
            }
            PredicateSinkOp::IndicesWhere => {
                if matched {
                    self.predicate_indices.push(self.predicate_seen as i64);
                }
                self.predicate_seen += 1;
                Ok(false)
            }
            PredicateSinkOp::FindOne => {
                if matched {
                    if self.predicate_matched.is_some() {
                        return Err(EvalError(
                            "find_one: expected exactly one element, got multiple".into(),
                        ));
                    }
                    self.predicate_matched = Some(self.predicate_seen);
                    self.predicate_value = Some(materialize_item());
                }
                self.predicate_seen += 1;
                Ok(false)
            }
        }
    }

    /// Updates a value-membership terminal sink with one row; returns true once decided.
    pub(crate) fn observe_membership(
        &mut self,
        op: MembershipSinkOp,
        item: &Val,
        target: &Val,
    ) -> bool {
        let matched = crate::util::vals_eq(item, target);
        self.observe_membership_match(op, matched)
    }

    /// Updates a value-membership terminal sink with an already-computed comparison result.
    pub(crate) fn observe_membership_match(
        &mut self,
        op: MembershipSinkOp,
        matched: bool,
    ) -> bool {
        match op {
            MembershipSinkOp::Includes => {
                if matched {
                    self.membership_matched = Some(self.membership_seen);
                    true
                } else {
                    self.membership_seen += 1;
                    false
                }
            }
            MembershipSinkOp::Index => {
                if matched {
                    self.membership_matched = Some(self.membership_seen);
                    true
                } else {
                    self.membership_seen += 1;
                    false
                }
            }
            MembershipSinkOp::IndicesOf => {
                if matched {
                    self.membership_indices.push(self.membership_seen as i64);
                }
                self.membership_seen += 1;
                false
            }
        }
    }

    /// Updates an arg-extreme terminal sink with one row and its projected key.
    pub(crate) fn observe_arg_extreme(&mut self, want_max: bool, item: Val, key: Val) {
        self.observe_arg_extreme_lazy(want_max, key, || item);
    }

    /// Lazy arg-extreme update. The row is materialised only when its key becomes the new best.
    pub(crate) fn observe_arg_extreme_lazy<F>(
        &mut self,
        want_max: bool,
        key: Val,
        materialize_item: F,
    ) where
        F: FnOnce() -> Val,
    {
        let should_take = match self.arg_extreme_key.as_ref() {
            None => true,
            Some(best_key) => {
                let ord = cmp_val_total(&key, best_key);
                if want_max {
                    ord.is_gt()
                } else {
                    ord.is_lt()
                }
            }
        };
        if should_take {
            self.arg_extreme_key = Some(key);
            self.arg_extreme_value = Some(materialize_item());
        }
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
            Sink::Predicate(spec) => match spec.op {
                PredicateSinkOp::Any => Val::Bool(self.predicate_matched.is_some()),
                PredicateSinkOp::All => Val::Bool(self.predicate_all),
                PredicateSinkOp::FindIndex => self
                    .predicate_matched
                    .map(|idx| Val::Int(idx as i64))
                    .unwrap_or(Val::Null),
                PredicateSinkOp::IndicesWhere => Val::int_vec(self.predicate_indices),
                PredicateSinkOp::FindOne => self.predicate_value.unwrap_or(Val::Null),
            },
            Sink::Membership(spec) => match spec.op {
                MembershipSinkOp::Includes => Val::Bool(self.membership_matched.is_some()),
                MembershipSinkOp::Index => self
                    .membership_matched
                    .map(|idx| Val::Int(idx as i64))
                    .unwrap_or(Val::Null),
                MembershipSinkOp::IndicesOf => Val::int_vec(self.membership_indices),
            },
            Sink::ArgExtreme(_) => self.arg_extreme_value.unwrap_or(Val::Null),
            Sink::SelectMany { n: 0, .. } => Val::Null,
            Sink::SelectMany { n, .. } if *n == 1 => self
                .select_many
                .into_iter()
                .next()
                .unwrap_or(Val::Null),
            Sink::SelectMany { .. } => Val::arr(self.select_many.into_iter().collect()),
            Sink::Nth(_) => self.nth.unwrap_or(Val::Null),
            Sink::Terminal(_) => Val::Null,
        }
    }

    /// Fallible finalisation for sinks whose terminal semantics can fail.
    pub(crate) fn finish_result(self, unwrap_single_collect_obj: bool) -> Result<Val, EvalError> {
        if matches!(self.sink, Sink::Predicate(spec) if spec.op == PredicateSinkOp::FindOne) {
            return self.predicate_value.ok_or_else(|| {
                EvalError("find_one: expected exactly one element, got 0".into())
            });
        }
        Ok(self.finish(unwrap_single_collect_obj))
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

// HyperLogLog precision: 2^12 = 4096 registers
const HLL_P: u32 = 12;
const HLL_M: usize = 1 << HLL_P;

// process-stable random state seeded once at startup
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

fn hll_observe(reg: &mut [u8; HLL_M], v: &Val) {
    use crate::util::val_to_key;
    hll_observe_key(reg, &val_to_key(v));
}

fn hll_observe_key(reg: &mut [u8; HLL_M], key: &str) {
    let h = hll_hash_key(key);
    let idx = (h >> (64 - HLL_P)) as usize;
    let w = (h << HLL_P) | (1u64 << (HLL_P - 1));
    let lz = w.leading_zeros() as u8 + 1;
    if lz > reg[idx] {
        reg[idx] = lz;
    }
}

// applies small-range linear counting correction when many registers are zero
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

#[cfg(test)]
mod tests {
    use std::{cell::Cell, sync::Arc};

    use crate::builtins::BuiltinMethod;
    use crate::exec::pipeline::{MembershipSinkSpec, MembershipSinkTarget, PredicateSinkSpec};
    use crate::vm::Program;

    use super::*;

    fn empty_program() -> Arc<Program> {
        Arc::new(Program::new(Vec::new(), "<sink-accumulator-test>"))
    }

    fn predicate_sink(op: PredicateSinkOp) -> Sink {
        Sink::Predicate(PredicateSinkSpec {
            op,
            predicate: empty_program(),
        })
    }

    fn membership_sink(op: MembershipSinkOp) -> Sink {
        Sink::Membership(MembershipSinkSpec {
            op,
            target: MembershipSinkTarget::Literal(Val::Int(7)),
            method: BuiltinMethod::Includes,
        })
    }

    #[test]
    fn predicate_short_circuit_sinks_do_not_materialize_rows() {
        let any_sink = predicate_sink(PredicateSinkOp::Any);
        let mut any = SinkAccumulator::new(&any_sink);
        let materialized = Cell::new(0);
        let decided = any
            .observe_predicate_lazy(PredicateSinkOp::Any, true, || {
                materialized.set(materialized.get() + 1);
                Val::Int(1)
            })
            .unwrap();

        assert!(decided);
        assert_eq!(materialized.get(), 0);
        assert_eq!(any.finish(false), Val::Bool(true));

        let all_sink = predicate_sink(PredicateSinkOp::All);
        let mut all = SinkAccumulator::new(&all_sink);
        let materialized = Cell::new(0);
        let decided = all
            .observe_predicate_lazy(PredicateSinkOp::All, false, || {
                materialized.set(materialized.get() + 1);
                Val::Int(1)
            })
            .unwrap();

        assert!(decided);
        assert_eq!(materialized.get(), 0);
        assert_eq!(all.finish(false), Val::Bool(false));
    }

    #[test]
    fn find_one_materializes_only_matching_row_once() {
        let sink = predicate_sink(PredicateSinkOp::FindOne);
        let mut acc = SinkAccumulator::new(&sink);
        let materialized = Cell::new(0);

        assert!(!acc
            .observe_predicate_lazy(PredicateSinkOp::FindOne, false, || {
                materialized.set(materialized.get() + 1);
                Val::Int(0)
            })
            .unwrap());
        assert_eq!(materialized.get(), 0);

        assert!(!acc
            .observe_predicate_lazy(PredicateSinkOp::FindOne, true, || {
                materialized.set(materialized.get() + 1);
                Val::Int(42)
            })
            .unwrap());
        assert_eq!(materialized.get(), 1);

        let err = acc
            .observe_predicate_lazy(PredicateSinkOp::FindOne, true, || {
                materialized.set(materialized.get() + 1);
                Val::Int(99)
            })
            .unwrap_err();
        assert!(err.0.contains("got multiple"));
        assert_eq!(materialized.get(), 1);
    }

    #[test]
    fn find_one_finish_result_requires_exactly_one_match() {
        let empty_sink = predicate_sink(PredicateSinkOp::FindOne);
        let mut empty = SinkAccumulator::new(&empty_sink);
        empty
            .observe_predicate_lazy(PredicateSinkOp::FindOne, false, || Val::Int(0))
            .unwrap();
        let err = empty.finish_result(false).unwrap_err();
        assert!(err.0.contains("got 0"));

        let one_sink = predicate_sink(PredicateSinkOp::FindOne);
        let mut one = SinkAccumulator::new(&one_sink);
        one.observe_predicate_lazy(PredicateSinkOp::FindOne, true, || Val::Int(9))
            .unwrap();
        assert_eq!(one.finish_result(false).unwrap(), Val::Int(9));
    }

    #[test]
    fn membership_short_circuit_sinks_stop_on_first_match() {
        let includes_sink = membership_sink(MembershipSinkOp::Includes);
        let mut includes = SinkAccumulator::new(&includes_sink);
        assert!(!includes.observe_membership_match(MembershipSinkOp::Includes, false));
        assert!(includes.observe_membership_match(MembershipSinkOp::Includes, true));
        assert_eq!(includes.finish(false), Val::Bool(true));

        let index_sink = membership_sink(MembershipSinkOp::Index);
        let mut index = SinkAccumulator::new(&index_sink);
        assert!(!index.observe_membership_match(MembershipSinkOp::Index, false));
        assert!(!index.observe_membership_match(MembershipSinkOp::Index, false));
        assert!(index.observe_membership_match(MembershipSinkOp::Index, true));
        assert_eq!(index.finish(false), Val::Int(2));
    }

    #[test]
    fn scalar_short_circuit_decisions_match_sink_result_demand() {
        use crate::plan::demand::SinkResultDemand;

        for op in [PredicateSinkOp::Any, PredicateSinkOp::FindIndex] {
            let sink = predicate_sink(op);
            assert_eq!(sink.demand().sink_result, SinkResultDemand::UntilMatch);
            let mut acc = SinkAccumulator::new(&sink);
            assert!(!acc.observe_predicate_lazy(op, false, || Val::Null).unwrap());
            assert!(acc.observe_predicate_lazy(op, true, || Val::Null).unwrap());
        }

        let sink = predicate_sink(PredicateSinkOp::All);
        assert_eq!(sink.demand().sink_result, SinkResultDemand::UntilFailure);
        let mut acc = SinkAccumulator::new(&sink);
        assert!(!acc
            .observe_predicate_lazy(PredicateSinkOp::All, true, || Val::Null)
            .unwrap());
        assert!(acc
            .observe_predicate_lazy(PredicateSinkOp::All, false, || Val::Null)
            .unwrap());

        for op in [MembershipSinkOp::Includes, MembershipSinkOp::Index] {
            let sink = membership_sink(op);
            assert_eq!(sink.demand().sink_result, SinkResultDemand::UntilMatch);
            let mut acc = SinkAccumulator::new(&sink);
            assert!(!acc.observe_membership_match(op, false));
            assert!(acc.observe_membership_match(op, true));
        }
    }

    #[test]
    fn membership_indices_sink_retains_all_matches_without_stopping() {
        let sink = membership_sink(MembershipSinkOp::IndicesOf);
        let mut acc = SinkAccumulator::new(&sink);

        assert!(!acc.observe_membership_match(MembershipSinkOp::IndicesOf, true));
        assert!(!acc.observe_membership_match(MembershipSinkOp::IndicesOf, false));
        assert!(!acc.observe_membership_match(MembershipSinkOp::IndicesOf, true));
        assert_eq!(acc.finish(false), Val::int_vec(vec![0, 2]));
    }
}
