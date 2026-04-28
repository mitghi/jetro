//! Jetro — query JSON bytes.
//!
//! ```rust
//! use jetro::Jetro;
//!
//! let j = Jetro::from_bytes(br#"{"store":{"books":[
//!   {"title":"Dune","price":12.99},
//!   {"title":"Foundation","price":9.99}
//! ]}}"#.to_vec()).unwrap();
//!
//! let count = j.collect("$.store.books.len()").unwrap();
//! assert_eq!(count, serde_json::json!(2));
//! ```

pub use jetro_core::EvalError;

/// Byte-oriented Jetro query handle.
pub struct Jetro {
    inner: jetro_core::Jetro,
}

impl Jetro {
    /// Parse JSON bytes and build a query handle.
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
