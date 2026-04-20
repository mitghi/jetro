//! Jetro is a tool for querying and transforming JSON.
//!
//! # Quick start
//!
//! ```rust
//! use jetro::Jetro;
//! use serde_json::json;
//!
//! let j = Jetro::new(json!({
//!     "store": {
//!         "books": [
//!             {"title": "Dune",        "price": 12.99},
//!             {"title": "Foundation",  "price":  9.99}
//!         ]
//!     }
//! }));
//!
//! let mut r = j.collect(">/store/books/#len").unwrap();
//! let count: i64 = r.from_index(0).unwrap();
//! assert_eq!(count, 2);
//! ```

extern crate pest_derive;

use pest::Parser;
use pest_derive::Parser as Parse;

pub mod context;
pub mod db;
pub mod graph;
pub mod parser;
pub mod v2;
pub mod vm;
mod fmt;
pub mod func;

// Re-export the two types users need most so they don't have to dig into
// sub-modules for everyday use.
pub use context::{Error, PathResult};

use serde_json::Value;

// ── Jetro ─────────────────────────────────────────────────────────────────────

/// Primary entry point for evaluating Jetro expressions.
///
/// `Jetro` holds a JSON document and evaluates path expressions against it.
/// Internally it delegates to a **thread-local VM** that is shared across every
/// `Jetro` instance on the same thread, so the compile cache and resolution
/// cache accumulate over the lifetime of the thread — not just one query.
///
/// # Example
///
/// ```rust
/// use jetro::Jetro;
/// use serde_json::json;
///
/// let doc = json!({"user": {"name": "Alice", "age": 30}});
/// let j = Jetro::new(doc);
///
/// let mut r = j.collect(">/user/name").unwrap();
/// let name: String = r.from_index(0).unwrap();
/// assert_eq!(name, "Alice");
///
/// // The same instance can be queried multiple times.
/// let mut r2 = j.collect(">/user/age").unwrap();
/// let age: i64 = r2.from_index(0).unwrap();
/// assert_eq!(age, 30);
/// ```
pub struct Jetro {
    document: Value,
}

impl Jetro {
    /// Create a new `Jetro` instance bound to `document`.
    pub fn new(document: Value) -> Self {
        Self { document }
    }

    /// Evaluate `expr` against the document.
    ///
    /// Uses the thread-local VM — the compiled program and resolved pointer
    /// paths are cached for the lifetime of the current thread, so repeated
    /// calls with the same expression skip both parsing and traversal.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the expression cannot be parsed or evaluated.
    pub fn collect<S: Into<String>>(&self, expr: S) -> Result<PathResult, Error> {
        let expr = expr.into();
        vm::THREAD_VM.with(|cell| match cell.try_borrow_mut() {
            Ok(mut vm) => vm.run_str(&expr, &self.document),
            // Re-entrant call (e.g. from inside a function): use a fresh VM.
            Err(_) => vm::VM::new().run_str(&expr, &self.document),
        })
    }
}

/// Convenience: create a `Jetro` instance directly from a `serde_json::Value`.
///
/// ```rust
/// use jetro::Jetro;
/// use serde_json::json;
///
/// let mut r = Jetro::from(json!({"x": 42})).collect(">/x").unwrap();
/// let x: i64 = r.from_index(0).unwrap();
/// assert_eq!(x, 42);
/// ```
impl From<Value> for Jetro {
    fn from(v: Value) -> Self {
        Self::new(v)
    }
}
