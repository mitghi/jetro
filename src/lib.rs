//! Thin façade crate — re-exports `jetro_core` through the public Jetro API.
//!
//! All parsing, optimisation, and execution live in `jetro_core`. This crate
//! provides the byte-oriented `Jetro` handle plus `JetroEngine` for long-lived
//! plan/VM reuse and NDJSON processing.

pub use jetro_core::{introspect, io, EvalError, JetroEngine, JetroEngineError};

/// Byte-oriented query handle. Wraps `jetro_core::Jetro` and exposes document
/// parsing plus query collection.
pub struct Jetro {
    /// The underlying core handle that owns the parsed document and all lazy caches.
    inner: jetro_core::Jetro,
}

impl Jetro {
    /// Parse raw JSON bytes and build a query handle.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, serde_json::Error> {
        Ok(Self {
            inner: jetro_core::Jetro::from_bytes(bytes)?,
        })
    }

    /// Evaluate a Jetro expression and return a `serde_json::Value`.
    pub fn collect<S: AsRef<str>>(&self, expr: S) -> Result<serde_json::Value, EvalError> {
        self.inner.collect(expr)
    }
}
