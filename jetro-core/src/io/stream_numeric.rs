use crate::builtins::BuiltinMethod;
use crate::data::value::Val;
use crate::util::JsonView;

#[derive(Clone, Debug)]
pub(super) struct NumericAccumulator {
    method: BuiltinMethod,
    int_acc: i64,
    float_acc: f64,
    floated: bool,
    count: usize,
    best: Option<Val>,
    best_f64: f64,
}

impl NumericAccumulator {
    pub(super) fn new(method: BuiltinMethod) -> Self {
        Self {
            method,
            int_acc: 0,
            float_acc: 0.0,
            floated: false,
            count: 0,
            best: None,
            best_f64: 0.0,
        }
    }

    pub(super) fn add_val(&mut self, value: &Val) {
        let Some(number) = value.as_f64() else {
            return;
        };
        if self.add_extreme(number, || value.clone()) {
            return;
        }
        match value {
            Val::Int(n) if !self.floated => {
                self.int_acc = self.int_acc.wrapping_add(*n);
            }
            Val::Int(n) => self.float_acc += *n as f64,
            Val::Float(f) if !self.floated => {
                self.float_acc = self.int_acc as f64 + *f;
                self.floated = true;
            }
            Val::Float(f) => self.float_acc += *f,
            _ => {}
        }
        self.count += 1;
    }

    pub(super) fn add_view(&mut self, value: JsonView<'_>) {
        let number = match value {
            JsonView::Int(n) => n as f64,
            JsonView::UInt(n) => n as f64,
            JsonView::Float(f) => f,
            _ => return,
        };
        if self.add_extreme(number, || match value {
            JsonView::Int(n) => Val::Int(n),
            JsonView::UInt(n) => i64::try_from(n)
                .map(Val::Int)
                .unwrap_or(Val::Float(n as f64)),
            JsonView::Float(f) => Val::Float(f),
            _ => Val::Null,
        }) {
            return;
        }
        match value {
            JsonView::Int(n) if !self.floated => {
                self.int_acc = self.int_acc.wrapping_add(n);
            }
            JsonView::Int(n) => self.float_acc += n as f64,
            JsonView::UInt(n) if !self.floated && i64::try_from(n).is_ok() => {
                self.int_acc = self.int_acc.wrapping_add(n as i64);
            }
            JsonView::UInt(n) if !self.floated => {
                self.float_acc = self.int_acc as f64 + n as f64;
                self.floated = true;
            }
            JsonView::UInt(n) => self.float_acc += n as f64,
            JsonView::Float(f) if !self.floated => {
                self.float_acc = self.int_acc as f64 + f;
                self.floated = true;
            }
            JsonView::Float(f) => self.float_acc += f,
            _ => {}
        }
        self.count += 1;
    }

    pub(super) fn value(&self) -> Val {
        match self.method {
            BuiltinMethod::Sum => {
                if self.floated {
                    Val::Float(self.float_acc)
                } else {
                    Val::Int(self.int_acc)
                }
            }
            BuiltinMethod::Avg => {
                if self.count == 0 {
                    Val::Null
                } else {
                    let sum = if self.floated {
                        self.float_acc
                    } else {
                        self.int_acc as f64
                    };
                    Val::Float(sum / self.count as f64)
                }
            }
            BuiltinMethod::Min | BuiltinMethod::Max => self.best.clone().unwrap_or(Val::Null),
            _ => Val::Null,
        }
    }

    fn add_extreme(&mut self, number: f64, value: impl FnOnce() -> Val) -> bool {
        if !matches!(self.method, BuiltinMethod::Min | BuiltinMethod::Max) {
            return false;
        }
        let replace = match self.method {
            BuiltinMethod::Min => self.best.is_none() || number < self.best_f64,
            BuiltinMethod::Max => self.best.is_none() || number > self.best_f64,
            _ => false,
        };
        if replace {
            self.best = Some(value());
            self.best_f64 = number;
        }
        self.count += 1;
        true
    }
}
