/// v2 custom method registry — similar to v1's `Callable` / `FuncRegistry`.
///
/// Users implement [`Method`] (or pass a closure) and register it by name.
/// Registered methods are checked after all built-ins in `dispatch_method`.
use std::sync::Arc;
use indexmap::IndexMap;

use super::value::Val;
use super::EvalError;

// ── Method trait ──────────────────────────────────────────────────────────────

/// A custom method that can be registered with [`MethodRegistry`].
///
/// `recv`  — the value the method was called on
/// `args`  — positional arguments, already evaluated to `Val`
pub trait Method: Send + Sync {
    fn call(&self, recv: Val, args: &[Val]) -> Result<Val, EvalError>;
}

/// Blanket impl: any `Fn(Val, &[Val]) -> Result<Val, EvalError>` is a `Method`.
impl<F> Method for F
where
    F: Fn(Val, &[Val]) -> Result<Val, EvalError> + Send + Sync,
{
    #[inline]
    fn call(&self, recv: Val, args: &[Val]) -> Result<Val, EvalError> {
        self(recv, args)
    }
}

// ── Registry ──────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct MethodRegistry {
    methods: IndexMap<String, Arc<dyn Method>>,
}

impl MethodRegistry {
    pub fn new() -> Self {
        Self { methods: IndexMap::new() }
    }

    /// Register a named method. Accepts anything that implements [`Method`],
    /// including closures.
    pub fn register(&mut self, name: impl Into<String>, method: impl Method + 'static) {
        self.methods.insert(name.into(), Arc::new(method));
    }

    /// Look up a method by name.
    #[inline]
    pub fn get(&self, name: &str) -> Option<&Arc<dyn Method>> {
        self.methods.get(name)
    }

    pub fn is_empty(&self) -> bool { self.methods.is_empty() }
}

impl Default for MethodRegistry {
    fn default() -> Self { Self::new() }
}
