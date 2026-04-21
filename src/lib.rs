//! Jetro — query and transform JSON with a compact expression language.
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
//!   vm::Compiler::optimize  ── peephole passes:
//!                              1. root_chain fusion
//!                              2. filter_count fusion
//!                              3. filter / map combinator fusion
//!                              4. find / quantifier fusion
//!                              5. strength reduction
//!                              6. redundant-op removal
//!                              7. kind_check constant fold
//!                              8. method constant fold
//!                              9. expression constant fold
//!                             10. nullness-driven OptField→GetField
//!       │
//!       ▼
//!   Compiler::compile runs:
//!       • AST rewrite: reorder_and_operands        (selectivity-based)
//!       • post-pass : analysis::dedup_subprograms  (CSE on Arc<Program>)
//!       │
//!       ▼
//!   vm::VM::execute          ── stack machine over &serde_json::Value
//!                                with thread-local pointer cache.
//! ```
//!
//! ## Analysis / IR layers
//!
//! Several independent analyses operate on the compiled [`Program`]:
//!
//! | Module         | Produces                                  | Uses                              |
//! |----------------|-------------------------------------------|-----------------------------------|
//! | [`analysis`]   | Type/Nullness/Cardinality, cost, monot.   | Optimizer heuristics, planner     |
//! | [`schema`]     | Shape inference from JSON docs            | Specialise `OptField` → `GetField`|
//! | [`plan`]       | Logical relational plan IR                | Filter push-down, join detection  |
//! | [`cfg`]        | Basic blocks + edges, dominators,         | Liveness, slot allocator          |
//! |                | dominance frontiers, loop headers         |                                   |
//! | [`ssa`]        | SSA-style numbering + phi nodes           | CSE, def-use analysis             |
//!
//! None of these are mandatory for correctness — the tree-walking
//! evaluator in `eval/` is the reference implementation.  They exist
//! to let advanced callers specialise a program for a known document
//! shape, run optimisations the peephole layer cannot express, or
//! export a readable data-flow graph for debugging.
//!
//! # Syntax overview
//!
//! ```text
//! // Navigation
//! $.store.books
//! $.store.books[0]
//! $.store.books[-1]
//! $.store.books[2:5]
//! $..title
//!
//! // Filter
//! $.books.filter(price > 10 and rating >= 4)
//! $.books.filter(lambda b: b.tags.includes("sci-fi"))
//!
//! // Map / reshape
//! $.books.map(title)
//! $.books.map({title, cost: price})
//!
//! // Aggregates
//! $.books.sum(price)
//! $.books.filter(price > 10).count()
//!
//! // Comprehensions
//! [book.title for book in $.books if book.price > 10]
//! {user.id: user.name for user in $.users if user.active}
//!
//! // Let bindings
//! let top = $.books.filter(price > 100) in {count: top.len(), titles: top.map(title)}
//!
//! // Kind checks
//! $.items.filter(price kind number and price > 0)
//! $.data.filter(deleted_at kind null)
//! ```
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
//! let count = j.collect("$.store.books.len()").unwrap();
//! assert_eq!(count, json!(2));
//! ```

pub mod ast;
pub mod engine;
pub mod session;
pub mod eval;
pub mod expr;
pub mod graph;
pub mod parser;
pub mod prelude;
pub mod vm;
pub mod analysis;
pub mod schema;
pub mod plan;
pub mod cfg;
pub mod ssa;
pub mod db;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod examples;

use std::cell::RefCell;
use std::sync::Arc;
use serde_json::Value;

pub use engine::Engine;
pub use session::{Catalog, MemCatalog, Session, SessionBuilder, SessionRow};
pub use eval::EvalError;
pub use eval::{Method, MethodRegistry};
pub use expr::Expr;
pub use graph::Graph;

/// Trait implemented by `#[derive(JetroSchema)]` — pairs a type with a
/// fixed set of named expressions.
///
/// ```ignore
/// use jetro::prelude::*;
/// use jetro::JetroSchema;
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

#[cfg(feature = "macros")]
pub use jetro_macros::{jetro, JetroSchema};
pub use parser::ParseError;
pub use vm::{VM, Compiler, Program};
pub use db::{DbError, Row};

/// Unified error type for jetro operations.
///
/// Covers parsing, evaluation, storage, and IO failures in one enum.
#[derive(Debug)]
pub enum Error {
    Parse(ParseError),
    Eval(EvalError),
    Db(DbError),
    Io(std::io::Error),
}

/// Convenience `Result` alias used across the public API.
pub type Result<T> = std::result::Result<T, Error>;

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Parse(e) => write!(f, "{}", e),
            Error::Eval(e)  => write!(f, "{}", e),
            Error::Db(e)    => write!(f, "{}", e),
            Error::Io(e)    => write!(f, "{}", e),
        }
    }
}
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Parse(e) => Some(e),
            Error::Eval(_)  => None,
            Error::Db(e)    => Some(e),
            Error::Io(e)    => Some(e),
        }
    }
}

impl From<ParseError>      for Error { fn from(e: ParseError)      -> Self { Error::Parse(e) } }
impl From<EvalError>       for Error { fn from(e: EvalError)       -> Self { Error::Eval(e)  } }
impl From<DbError>         for Error { fn from(e: DbError)         -> Self { Error::Db(e)    } }
impl From<std::io::Error>  for Error { fn from(e: std::io::Error)  -> Self { Error::Io(e)    } }

/// Evaluate a Jetro expression against a JSON value.
///
/// ```rust,no_run
/// use serde_json::json;
/// use jetro;
///
/// let doc = json!({
///     "store": {
///         "books": [
///             {"title": "Dune",       "price": 12.99},
///             {"title": "Foundation", "price":  9.99},
///         ]
///     }
/// });
///
/// let result = jetro::query("$.store.books.filter(price > 10).map(title)", &doc).unwrap();
/// assert_eq!(result, json!(["Dune"]));
/// ```
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
/// delegates to a thread-local [`VM`] shared across every `Jetro`
/// instance on the same thread, so the compile cache and resolution cache
/// accumulate over the lifetime of the thread — not just one query.
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
/// assert_eq!(j.collect("$.user.name").unwrap(), json!("Alice"));
/// assert_eq!(j.collect("$.user.age").unwrap(),  json!(30));
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
    /// Uses the thread-local VM — compiled programs and resolved pointer
    /// paths are cached for the lifetime of the current thread.
    pub fn collect<S: AsRef<str>>(&self, expr: S) -> Result<Value> {
        let expr = expr.as_ref();
        THREAD_VM.with(|cell| match cell.try_borrow_mut() {
            Ok(mut vm)  => vm.run_str(expr, &self.document).map_err(Error::Eval),
            // Re-entrant call (e.g. from inside a method): use a fresh VM.
            Err(_)      => VM::new().run_str(expr, &self.document).map_err(Error::Eval),
        })
    }
}

impl From<Value> for Jetro {
    fn from(v: Value) -> Self {
        Self::new(v)
    }
}
