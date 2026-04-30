use crate::value::Val;

use super::{num_finalise, num_fold, ReducerOp, ReducerSpec};

#[derive(Debug, Clone)]
pub(crate) struct ReducerAccumulator {
    spec: ReducerSpec,
    count: i64,
    sum_i: i64,
    sum_f: f64,
    sum_floated: bool,
    min_f: f64,
    max_f: f64,
    n_obs: usize,
}

impl ReducerAccumulator {
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

    pub(crate) fn push(&mut self, item: &Val) {
        debug_assert!(
            self.spec.predicate.is_none(),
            "predicate reducers must evaluate predicates before fold"
        );
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
