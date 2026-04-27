//! Jetro — transform, query, and compare JSON.
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
//! let count = j.collect("$.store.books.len()").unwrap();
//! assert_eq!(count, json!(2));
//! ```

pub mod prelude;

// Engine surface.
pub use jetro_core::{
    CompiledQuery, Compiler, Engine, EvalError, Expr, Jetro, JetroSchema, Method, MethodRegistry,
    ParseError, Program, VM,
};

// Module re-exports for callers that reach into submodules.
pub use jetro_core::ast;
pub use jetro_core::eval;
pub use jetro_core::parser;
pub use jetro_core::vm;

#[cfg(feature = "macros")]
pub use jetro_macros::{jetro, JetroSchema};

use serde_json::Value;
use std::sync::Arc;

/// Engine-side error type.  Either a parse failure or an evaluation failure.
#[derive(Debug)]
pub enum Error {
    Parse(ParseError),
    Eval(EvalError),
}

pub type Result<T> = std::result::Result<T, Error>;

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Parse(e) => write!(f, "{}", e),
            Error::Eval(e)  => write!(f, "{}", e),
        }
    }
}
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Parse(e) => Some(e),
            Error::Eval(_)  => None,
        }
    }
}

impl From<ParseError> for Error { fn from(e: ParseError) -> Self { Error::Parse(e) } }
impl From<EvalError>  for Error { fn from(e: EvalError)  -> Self { Error::Eval(e)  } }

impl From<jetro_core::Error> for Error {
    fn from(e: jetro_core::Error) -> Self {
        match e {
            jetro_core::Error::Parse(p) => Error::Parse(p),
            jetro_core::Error::Eval(v)  => Error::Eval(v),
        }
    }
}

/// Evaluate a Jetro expression against a JSON value.
pub fn query(expr: &str, doc: &Value) -> Result<Value> {
    Ok(jetro_core::query(expr, doc)?)
}

/// Evaluate a Jetro expression with a custom method registry.
pub fn query_with(expr: &str, doc: &Value, registry: Arc<MethodRegistry>) -> Result<Value> {
    Ok(jetro_core::query_with(expr, doc, registry)?)
}
