//! Jetro core — parser, compiler, and VM for the Jetro JSON query language.
//!
//! This crate is storage-free.  For the embedded B+ tree store, named
//! expressions, graph queries, joins, and [`Session`](../jetrodb/struct.Session.html),
//! depend on the sibling `jetrodb` crate, or pull the umbrella `jetro` crate
//! which re-exports both.
//!
//! # Architecture
//!
//! ```text
//!   source text
//!       │
//!       ▼
//!   parser.rs  ── pest grammar → [ast::Expr] tree
//!       │
//!       ▼
//!   vm::Compiler::emit      ── Expr → Vec<Opcode>
//!       │
//!       ▼
//!   vm::Compiler::optimize  ── peephole passes (root_chain, filter/count,
//!                              filter/map fusion, strength reduction,
//!                              constant folding, nullness-driven specialisation)
//!       │
//!       ▼
//!   Compiler::compile runs:
//!       • AST rewrite: reorder_and_operands        (selectivity-based)
//!       • post-pass  : analysis::dedup_subprograms (CSE on Arc<Program>)
//!       │
//!       ▼
//!   vm::VM::execute          ── stack machine over &serde_json::Value
//!                                with thread-local pointer cache.
//! ```
//!
//! # Quick start
//!
//! ```rust
//! use jetro_core::Jetro;
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

pub mod ast;
pub mod engine;
pub mod eval;
pub mod expr;
pub mod graph;
pub mod parser;
pub mod vm;
pub mod analysis;
pub mod schema;
pub mod plan;
pub mod cfg;
pub mod ssa;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod examples;

use std::cell::RefCell;
use std::sync::Arc;
use serde_json::Value;

pub use engine::Engine;
pub use eval::EvalError;
pub use eval::{Method, MethodRegistry};
pub use expr::Expr;
pub use graph::Graph;
pub use parser::ParseError;
pub use vm::{VM, Compiler, Program};

/// Trait implemented by `#[derive(JetroSchema)]` — pairs a type with a
/// fixed set of named expressions.
///
/// ```ignore
/// use jetro_core::JetroSchema;
///
/// #[derive(JetroSchema)]
/// #[expr(titles = "$.books.map(title)")]
/// #[expr(count  = "$.books.len()")]
/// struct BookView;
///
/// for (name, src) in BookView::exprs() { /* register on a bucket */ }
/// ```
pub trait JetroSchema {
    const EXPRS: &'static [(&'static str, &'static str)];
    fn exprs() -> &'static [(&'static str, &'static str)];
    fn names() -> &'static [&'static str];
}

// ── Error ─────────────────────────────────────────────────────────────────────

/// Engine-side error type.  Either a parse failure or an evaluation failure.
///
/// Storage and IO errors are carried by `jetrodb::DbError` in the sibling
/// crate.  The umbrella `jetro` crate unifies both into a flatter
/// `jetro::Error` for callers that want a single match arm per variant.
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

/// Evaluate a Jetro expression against a JSON value.
pub fn query(expr: &str, doc: &Value) -> Result<Value> {
    let ast = parser::parse(expr)?;
    Ok(eval::evaluate(&ast, doc)?)
}

/// Evaluate a Jetro expression with a custom method registry.
pub fn query_with(expr: &str, doc: &Value, registry: Arc<MethodRegistry>) -> Result<Value> {
    let ast = parser::parse(expr)?;
    Ok(eval::evaluate_with(&ast, doc, registry)?)
}

// ── Jetro ─────────────────────────────────────────────────────────────────────

thread_local! {
    static THREAD_VM: RefCell<VM> = RefCell::new(VM::new());
}

/// Primary entry point for evaluating Jetro expressions.
///
/// Holds a JSON document and evaluates expressions against it.  Internally
/// delegates to a thread-local [`VM`] so the compile cache and resolution
/// cache accumulate over the lifetime of the thread.
pub struct Jetro {
    document: Value,
}

impl Jetro {
    pub fn new(document: Value) -> Self {
        Self { document }
    }

    /// Evaluate `expr` against the document.
    pub fn collect<S: AsRef<str>>(&self, expr: S) -> std::result::Result<Value, EvalError> {
        let expr = expr.as_ref();
        THREAD_VM.with(|cell| match cell.try_borrow_mut() {
            Ok(mut vm) => vm.run_str(expr, &self.document),
            Err(_)     => VM::new().run_str(expr, &self.document),
        })
    }
}

impl From<Value> for Jetro {
    fn from(v: Value) -> Self {
        Self::new(v)
    }
}
