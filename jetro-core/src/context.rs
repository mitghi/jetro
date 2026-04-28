use crate::value::Val;
use smallvec::SmallVec;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct EvalError(pub String);

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "eval error: {}", self.0)
    }
}

impl std::error::Error for EvalError {}

/// Saved-state token returned by `Env::push_lam` and consumed by
/// `Env::pop_lam`. Lets hot loops bind a single lambda slot and swap
/// `current` without cloning the whole Env per iteration.
pub struct LamFrame {
    prev_current: Val,
    prev_var: LamVarPrev,
}

enum LamVarPrev {
    None,
    Pushed,
    Replaced(usize, Val),
}

#[derive(Clone)]
pub struct Env {
    vars: SmallVec<[(Arc<str>, Val); 4]>,
    pub root: Val,
    pub current: Val,
}

impl Env {
    pub fn new(root: Val) -> Self {
        Self {
            vars: SmallVec::new(),
            root: root.clone(),
            current: root,
        }
    }

    #[inline]
    pub fn with_current(&self, current: Val) -> Self {
        Self {
            vars: self.vars.clone(),
            root: self.root.clone(),
            current,
        }
    }

    #[inline]
    pub fn swap_current(&mut self, new: Val) -> Val {
        std::mem::replace(&mut self.current, new)
    }

    #[inline]
    pub fn restore_current(&mut self, old: Val) {
        self.current = old;
    }

    #[inline]
    pub fn get_var(&self, name: &str) -> Option<&Val> {
        self.vars
            .iter()
            .rev()
            .find(|(k, _)| k.as_ref() == name)
            .map(|(_, v)| v)
    }

    pub fn with_var(&self, name: &str, val: Val) -> Self {
        let mut vars = self.vars.clone();
        if let Some(pos) = vars.iter().position(|(k, _)| k.as_ref() == name) {
            vars[pos].1 = val;
        } else {
            vars.push((Arc::from(name), val));
        }
        Self {
            vars,
            root: self.root.clone(),
            current: self.current.clone(),
        }
    }

    #[inline]
    pub fn push_lam(&mut self, name: Option<&str>, val: Val) -> LamFrame {
        let prev_current = std::mem::replace(&mut self.current, val.clone());
        let prev_var = match name {
            None => LamVarPrev::None,
            Some(n) => {
                if let Some(pos) = self.vars.iter().position(|(k, _)| k.as_ref() == n) {
                    let prev = std::mem::replace(&mut self.vars[pos].1, val);
                    LamVarPrev::Replaced(pos, prev)
                } else {
                    self.vars.push((Arc::from(n), val));
                    LamVarPrev::Pushed
                }
            }
        };
        LamFrame {
            prev_current,
            prev_var,
        }
    }

    #[inline]
    pub fn pop_lam(&mut self, frame: LamFrame) {
        self.current = frame.prev_current;
        match frame.prev_var {
            LamVarPrev::None => {}
            LamVarPrev::Pushed => {
                self.vars.pop();
            }
            LamVarPrev::Replaced(pos, prev) => {
                self.vars[pos].1 = prev;
            }
        }
    }
}
