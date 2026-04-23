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
pub mod scan;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod examples;

use std::cell::{OnceCell, RefCell};
use std::sync::Arc;
use serde_json::Value;
use eval::Val;

pub use engine::Engine;
pub use eval::EvalError;
pub use eval::{Method, MethodRegistry, Val as JetroVal};
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
    /// Cached `Val` tree — built on first `collect()` and reused across
    /// subsequent calls, amortising the `Val::from(&Value)` walk.
    root_val: OnceCell<Val>,
    /// Retained JSON source bytes when the caller built via
    /// [`Jetro::from_bytes`] / [`Jetro::from_slice`].  Enables SIMD
    /// byte-scan fast paths for `$..key` queries.
    raw_bytes: Option<Arc<[u8]>>,
}

impl Jetro {
    pub fn new(document: Value) -> Self {
        Self { document, root_val: OnceCell::new(), raw_bytes: None }
    }

    /// Parse JSON bytes and retain them alongside the parsed document.
    /// Descendant queries (`$..key`) can then take the SIMD byte-scan path
    /// instead of walking the tree.
    pub fn from_bytes(bytes: Vec<u8>) -> std::result::Result<Self, serde_json::Error> {
        let document: Value = serde_json::from_slice(&bytes)?;
        Ok(Self {
            document,
            root_val: OnceCell::new(),
            raw_bytes: Some(Arc::from(bytes.into_boxed_slice())),
        })
    }

    /// Parse JSON from a slice, retaining a copy of the bytes.
    pub fn from_slice(bytes: &[u8]) -> std::result::Result<Self, serde_json::Error> {
        Self::from_bytes(bytes.to_vec())
    }

    fn root_val(&self) -> Val {
        self.root_val.get_or_init(|| Val::from(&self.document)).clone()
    }

    /// Evaluate `expr` against the document.  Routes through the thread-local
    /// VM (compile + path caches); when the Jetro handle carries raw bytes
    /// the VM executes on an env with `raw_bytes` set so `Opcode::Descendant`
    /// can take the SIMD byte-scan fast path.
    pub fn collect<S: AsRef<str>>(&self, expr: S) -> std::result::Result<Value, EvalError> {
        let expr = expr.as_ref();
        THREAD_VM.with(|cell| match (cell.try_borrow_mut(), &self.raw_bytes) {
            (Ok(mut vm), Some(bytes)) => {
                let prog = vm.get_or_compile(expr)?;
                vm.execute_val_with_raw(&prog, self.root_val(), Arc::clone(bytes))
            }
            (Ok(mut vm), None) => {
                let prog = vm.get_or_compile(expr)?;
                vm.execute_val(&prog, self.root_val())
            }
            (Err(_), Some(bytes)) => VM::new().run_str_with_raw(expr, &self.document, Arc::clone(bytes)),
            (Err(_), None)        => VM::new().run_str(expr, &self.document),
        })
    }

    /// Evaluate `expr` and return the raw `Val` without converting to
    /// `serde_json::Value`.  For large structural results (e.g. `group_by`
    /// on 20k+ items) this avoids an expensive materialisation that
    /// otherwise dominates runtime.  The returned `Val` supports cheap
    /// `Arc`-clone and shares structure with the source document.
    ///
    /// Prefer this over `collect` when the caller consumes the result
    /// structurally (further queries, custom walk, re-evaluation) rather
    /// than handing it to `serde_json`-aware code.
    pub fn collect_val<S: AsRef<str>>(&self, expr: S) -> std::result::Result<JetroVal, EvalError> {
        let expr = expr.as_ref();
        THREAD_VM.with(|cell| {
            let mut vm = cell.try_borrow_mut().map_err(|_| EvalError("VM in use".into()))?;
            let prog = vm.get_or_compile(expr)?;
            vm.execute_val_raw(&prog, self.root_val())
        })
    }
}

impl From<Value> for Jetro {
    fn from(v: Value) -> Self {
        Self::new(v)
    }
}
