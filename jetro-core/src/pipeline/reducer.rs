//! Stateful numeric reducer accumulator for pipeline aggregate sinks.
//! Tracks running sum, count, min, and max in a single pass so `avg`, `sum`,
//! `min`, `max`, and `count` share one loop over the source elements.

use crate::value::Val;

use super::{num_finalise, num_fold, ReducerOp, ReducerSpec};

/// Single-pass accumulator for numeric aggregate sinks (`sum`, `avg`, `min`, `max`, `count`).
///
/// All statistics are computed in one iteration over the source; unused fields incur no cost.
#[derive(Debug, Clone)]
pub(crate) struct ReducerAccumulator {
    /// Governs which aggregate operation is performed and any associated projection/predicate.
    spec: ReducerSpec,
    /// Running row count, incremented for every `push` when `op` is `Count`.
    count: i64,
    /// Integer accumulation for sum; promoted to `sum_f` on first float encounter.
    sum_i: i64,
    /// Floating-point accumulation for sum after integer overflow or float input.
    sum_f: f64,
    /// Set to `true` once `sum_i` has been promoted to `sum_f`.
    sum_floated: bool,
    /// Running minimum tracked as `f64` for uniform comparison.
    min_f: f64,
    /// Running maximum tracked as `f64` for uniform comparison.
    max_f: f64,
    /// Count of numeric observations used by `avg` to compute the mean.
    n_obs: usize,
}

impl ReducerAccumulator {
    /// Creates an accumulator initialised to identity values for all running statistics.
    pub(crate) fn new(spec: ReducerSpec) -> Self {
        Self {
            spec,
            count: 0,
            sum_i: 0,
            sum_f: 0.0,
            sum_floated: false,
            min_f: f64::INFINITY,
            max_f: f64::NEG_INFINITY,
            n_obs: 0,
        }
    }

    /// Folds `item` into the running statistics according to the reducer operation.
    pub(crate) fn push(&mut self, item: &Val) {
        match self.spec.op {
            ReducerOp::Count => {
                self.count += 1;
            }
            ReducerOp::Numeric(op) => {
                num_fold(
                    &mut self.sum_i,
                    &mut self.sum_f,
                    &mut self.sum_floated,
                    &mut self.min_f,
                    &mut self.max_f,
                    &mut self.n_obs,
                    op,
                    item,
                );
            }
        }
    }

    /// Consumes the accumulator and returns the final aggregate `Val`.
    pub(crate) fn finish(self) -> Val {
        match self.spec.op {
            ReducerOp::Count => Val::Int(self.count),
            ReducerOp::Numeric(op) => num_finalise(
                op,
                self.sum_i,
                self.sum_f,
                self.sum_floated,
                self.min_f,
                self.max_f,
                self.n_obs,
            ),
        }
    }
}
