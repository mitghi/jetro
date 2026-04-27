//! Top-level execution handle.
//!
//! [`Engine`] wraps the bytecode [`VM`] behind a `Send + Sync` façade so
//! long-lived services can share compile + pointer caches across threads.
//!
//! ```text
//! Engine ──► Mutex<VM> ──► compile cache
//!                      └─► path cache
//! ```
//!
//! Use [`Engine::new`] once at boot, pass `Arc<Engine>` wherever you need
//! to run queries, and the caches warm up organically. [`Jetro`] is the
//! thread-local convenience built on the same VM type — pick whichever
//! fits the call site.

use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::Value;

use crate::eval::methods::MethodRegistry;
use crate::vm::{PassConfig, VM};
use crate::{Error, Result};

/// Shared, thread-safe query engine.
///
/// Internally a `Mutex<VM>`; reads serialise but compilation and caching
/// happen once per expression, so the lock is held only for the duration
/// of an `execute` call. For single-threaded use prefer [`Jetro`], which
/// keeps a dedicated VM per thread.
pub struct Engine {
    vm: Mutex<VM>,
}

impl Engine {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            vm: Mutex::new(VM::new()),
        })
    }

    pub fn with_capacity(compile_cap: usize, path_cap: usize) -> Arc<Self> {
        Arc::new(Self {
            vm: Mutex::new(VM::with_capacity(compile_cap, path_cap)),
        })
    }

    pub fn with_methods(registry: Arc<MethodRegistry>) -> Arc<Self> {
        Arc::new(Self {
            vm: Mutex::new(VM::with_registry(registry)),
        })
    }

    /// Parse, compile (cached), and execute `expr` against `doc`.
    pub fn run(&self, expr: &str, doc: &Value) -> Result<Value> {
        self.vm.lock().run_str(expr, doc).map_err(Error::from)
    }

    pub fn register(
        &self,
        name: impl Into<String>,
        method: impl crate::eval::methods::Method + 'static,
    ) {
        self.vm.lock().register(name, method);
    }

    pub fn set_pass_config(&self, config: PassConfig) {
        self.vm.lock().set_pass_config(config);
    }

    pub fn pass_config(&self) -> PassConfig {
        self.vm.lock().pass_config()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn engine_runs_and_caches() {
        let e = Engine::new();
        let doc = json!({"x": [1, 2, 3, 4]});
        assert_eq!(e.run("$.x.len()", &doc).unwrap(), json!(4));
        // Rerun hits the compile cache — same answer, no panic.
        assert_eq!(e.run("$.x.len()", &doc).unwrap(), json!(4));
    }

    #[test]
    fn engine_is_shareable() {
        let e = Engine::new();
        let e2 = Arc::clone(&e);
        let h = std::thread::spawn(move || {
            let doc = json!({"n": 5});
            e2.run("$.n", &doc).unwrap()
        });
        assert_eq!(h.join().unwrap(), json!(5));
    }
}
