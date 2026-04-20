//! Jetro v2 — redesigned query syntax for JSON.
//!
//! # Architecture
//!
//! v2 is built as a layered pipeline.  Each layer has a single,
//! narrow responsibility so the compiler/optimizer stays tractable
//! and so each layer is independently testable.
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

pub mod ast;
pub mod eval;
pub mod graph;
pub mod parser;
pub mod vm;
pub mod analysis;
pub mod schema;
pub mod plan;
pub mod cfg;
pub mod ssa;
mod tests;
mod examples;

use std::sync::Arc;
use serde_json::Value;

pub use eval::EvalError;
pub use eval::{Method, MethodRegistry};
pub use graph::Graph;
pub use parser::ParseError;
pub use vm::{VM, Compiler, Program};

/// Combined error for parse + eval.
#[derive(Debug)]
pub enum Error {
    Parse(ParseError),
    Eval(EvalError),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Parse(e) => write!(f, "{}", e),
            Error::Eval(e)  => write!(f, "{}", e),
        }
    }
}
impl std::error::Error for Error {}

impl From<ParseError> for Error { fn from(e: ParseError) -> Self { Error::Parse(e) } }
impl From<EvalError>  for Error { fn from(e: EvalError)  -> Self { Error::Eval(e)  } }

/// Evaluate a Jetro v2 expression against a JSON value.
///
/// ```rust,no_run
/// use serde_json::json;
/// use jetro::v2;
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
/// let result = v2::query("$.store.books.filter(price > 10).map(title)", &doc).unwrap();
/// assert_eq!(result, json!(["Dune"]));
/// ```
pub fn query(expr: &str, doc: &Value) -> Result<Value, Error> {
    let ast = parser::parse(expr)?;
    Ok(eval::evaluate(&ast, doc)?)
}

/// Evaluate a Jetro v2 expression with a custom method registry.
pub fn query_with(expr: &str, doc: &Value, registry: Arc<MethodRegistry>) -> Result<Value, Error> {
    let ast = parser::parse(expr)?;
    Ok(eval::evaluate_with(&ast, doc, registry)?)
}
