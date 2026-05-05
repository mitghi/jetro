//! Thin façade crate — re-exports `jetro_core` through a byte-oriented API.
//!
//! All parsing, optimisation, and execution live in `jetro_core`. This crate
//! provides a minimal `Jetro` handle that accepts raw JSON bytes and surfaces
//! `collect` as the single query entry point.

pub use jetro_core::EvalError;

/// Byte-oriented query handle. Wraps `jetro_core::Jetro` and exposes only
/// the two public entry points needed by end users: `from_bytes` and `collect`.
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
