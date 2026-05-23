use crate::builtins::BuiltinNumericReducer;
use crate::data::value::Val;
use crate::exec::pipeline::{num_finalise, num_fold, NumOp};
use crate::util::JsonView;

#[derive(Clone, Debug)]
pub(super) struct NumericAccumulator {
    reducer: BuiltinNumericReducer,
    int_acc: i64,
    float_acc: f64,
    floated: bool,
    count: usize,
    min_f: f64,
    max_f: f64,
}

impl NumericAccumulator {
    pub(super) fn from_reducer(reducer: BuiltinNumericReducer) -> Self {
        Self {
            reducer,
            int_acc: 0,
            float_acc: 0.0,
            floated: false,
            count: 0,
            min_f: f64::INFINITY,
            max_f: f64::NEG_INFINITY,
        }
    }

    pub(super) fn add_val(&mut self, value: &Val) {
        let op = self.op();
        num_fold(
            &mut self.int_acc,
            &mut self.float_acc,
            &mut self.floated,
            &mut self.min_f,
            &mut self.max_f,
            &mut self.count,
            op,
            value,
        );
    }

    pub(super) fn add_view(&mut self, value: JsonView<'_>) {
        let value = match value {
            JsonView::Int(n) => Val::Int(n),
            JsonView::UInt(n) => i64::try_from(n)
                .map(Val::Int)
                .unwrap_or(Val::Float(n as f64)),
            JsonView::Float(f) => Val::Float(f),
            _ => return,
        };
        self.add_val(&value);
    }

    pub(super) fn value(&self) -> Val {
        num_finalise(
            self.op(),
            self.int_acc,
            self.float_acc,
            self.floated,
            self.min_f,
            self.max_f,
            self.count,
        )
    }

    pub(super) fn merge(&mut self, other: &Self) {
        debug_assert_eq!(self.reducer, other.reducer);
        if other.count == 0 {
            return;
        }
        match self.reducer {
            BuiltinNumericReducer::Sum | BuiltinNumericReducer::Avg => {
                if self.floated || other.floated {
                    if !self.floated {
                        self.float_acc = self.int_acc as f64;
                        self.floated = true;
                    }
                    self.float_acc += if other.floated {
                        other.float_acc
                    } else {
                        other.int_acc as f64
                    };
                } else {
                    self.int_acc = self.int_acc.wrapping_add(other.int_acc);
                }
                self.count += other.count;
            }
            BuiltinNumericReducer::Min | BuiltinNumericReducer::Max => {
                if self.count == 0 {
                    *self = other.clone();
                    return;
                }
                if self.reducer == BuiltinNumericReducer::Min {
                    if other.min_f < self.min_f {
                        self.min_f = other.min_f;
                    }
                    if !self.floated && !other.floated && other.int_acc < self.int_acc {
                        self.int_acc = other.int_acc;
                    }
                } else {
                    if other.max_f > self.max_f {
                        self.max_f = other.max_f;
                    }
                    if !self.floated && !other.floated && other.int_acc > self.int_acc {
                        self.int_acc = other.int_acc;
                    }
                }
                self.floated |= other.floated;
                self.count += other.count;
            }
        }
    }

    fn op(&self) -> NumOp {
        NumOp::from_builtin_reducer(self.reducer)
    }
}
