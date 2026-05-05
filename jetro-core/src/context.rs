//! Evaluation context shared by the VM, builtins, and pipeline executor.
//!
//! `Env` carries the three binding forms the language exposes — the document
//! root (`$`), the current item (`@`), and named let-bindings. It is cloned
//! per-scope but kept cheap via `SmallVec` (inline storage for ≤4 vars).

use crate::value::Val;
use smallvec::SmallVec;
use std::sync::Arc;

/// Evaluation error carrying a human-readable message. Propagated through
/// `Result<Val, EvalError>` across all execution layers.
#[derive(Debug, Clone)]
pub struct EvalError(pub String);

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "eval error: {}", self.0)
    }
}

impl std::error::Error for EvalError {}


/// Saved-state token for the hot-loop lambda binding protocol.
/// `push_lam` returns one; `pop_lam` consumes it. Avoids full `Env` clone
/// per iteration — only `current` and the single named binding are swapped.
pub struct LamFrame {
    /// Previous value of `Env::current`, restored by `pop_lam`.
    prev_current: Val,
    /// Describes what happened to the named variable slot so `pop_lam` can undo it.
    prev_var: LamVarPrev,
}

/// Encodes the three possible states of a named-variable slot before `push_lam`.
enum LamVarPrev {
    /// No variable name was bound; only `current` was updated.
    None,
    /// The variable was not present and was appended; `pop_lam` must pop it.
    Pushed,
    /// The variable existed at `usize`; its previous `Val` is saved for restoration.
    Replaced(usize, Val),
}

/// Per-scope evaluation environment. Cloned on scope entry; mutated in place
/// for tight loops via `push_lam`/`pop_lam`. Carries `root` ($), `current`
/// (@), and a flat var list for let-bindings.
#[derive(Clone)]
pub struct Env {
    /// Flat list of named let-bindings; searched in reverse for shadowing.
    vars: SmallVec<[(Arc<str>, Val); 4]>,
    /// The document root bound to `$`; immutable within a single query.
    pub root: Val,
    /// The current focus bound to `@`; updated per iteration in loops and chains.
    pub current: Val,
}

impl Env {
    /// Create a fresh environment with `root` bound to both `$` and `@`.
    pub fn new(root: Val) -> Self {
        Self {
            vars: SmallVec::new(),
            root: root.clone(),
            current: root,
        }
    }

    /// Return a child environment that inherits all vars and root but sets a new `current`.
    #[inline]
    pub fn with_current(&self, current: Val) -> Self {
        Self {
            vars: self.vars.clone(),
            root: self.root.clone(),
            current,
        }
    }

    /// Replace `current` in place and return the displaced value for later restoration.
    #[inline]
    pub fn swap_current(&mut self, new: Val) -> Val {
        std::mem::replace(&mut self.current, new)
    }

    /// Restore a previously swapped `current` value without allocating a new `Env`.
    #[inline]
    pub fn restore_current(&mut self, old: Val) {
        self.current = old;
    }

    /// Look up a named variable; searches in reverse so the innermost binding wins.
    #[inline]
    pub fn get_var(&self, name: &str) -> Option<&Val> {
        self.vars
            .iter()
            .rev()
            .find(|(k, _)| k.as_ref() == name)
            .map(|(_, v)| v)
    }

    /// Return a child environment that shadows (or inserts) `name = val`; does not mutate `self`.
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

    /// Bind `val` to `current` (and optionally to `name`) in place for one loop iteration.
    /// Returns a `LamFrame` that records the displaced state; must be balanced with `pop_lam`.
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

    /// Restore `Env` to the state captured in `frame`; must be called after every `push_lam`.
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
