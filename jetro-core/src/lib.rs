//! Jetro core — parser, compiler, and VM for the Jetro JSON query language.
//!
//! # Execution path
//!
//! ```text
//! source text
//!   │  parse::parser::parse() → Expr AST
//!   │  plan::physical::plan_query() → QueryPlan (physical IR)
//!   │  exec::router::collect_*() → dispatches to:
//!   │    StructuralIndex backend  (jetro-experimental bitmap)
//!   │    ViewPipeline backend     (borrowed tape/Val navigation)
//!   │    Pipeline backend         (pull-based composed stages)
//!   └─  VM fallback               (bytecode stack machine)
//! ```
//!
//! # Quick start
//!
//! ```rust
//! use jetro_core::Jetro;
//! let j = Jetro::from_bytes(br#"{"books":[{"price":12}]}"#.to_vec()).unwrap();
//! assert_eq!(j.collect("$.books.len()").unwrap(), serde_json::json!(1));
//! ```
//!
//! ```rust
//! use jetro_core::JetroEngine;
//! use std::io::Cursor;
//!
//! let engine = JetroEngine::new();
//! let rows = Cursor::new(br#"{"name":"Ada"}
//! {"name":"Bob"}
//! "#);
//! let names = engine.collect_ndjson(rows, "name").unwrap();
//! assert_eq!(names, vec![serde_json::json!("Ada"), serde_json::json!("Bob")]);
//! ```
//!
//! Match-limited NDJSON helpers evaluate a predicate per row, return the
//! original full row for truthy matches, and stop after the requested number of
//! matches:
//!
//! ```rust
//! use jetro_core::JetroEngine;
//! use std::io::Cursor;
//!
//! let engine = JetroEngine::new();
//! let rows = Cursor::new(br#"{"id":1,"active":true}
//! {"id":2,"active":false}
//! {"id":3,"active":true}
//! "#);
//! let first_two = engine.collect_ndjson_matches(rows, "active", 2).unwrap();
//! assert_eq!(first_two, vec![
//!     serde_json::json!({"id": 1, "active": true}),
//!     serde_json::json!({"id": 3, "active": true}),
//! ]);
//! ```

pub(crate) mod builtins;
pub(crate) mod compile;
pub(crate) mod data;
pub(crate) mod exec;
pub mod io;
pub(crate) mod ir;
pub(crate) mod parse;
pub(crate) mod plan;
pub(crate) mod util;
pub(crate) mod vm;

#[cfg(test)]
mod tests;

use data::value::Val;
use serde_json::Value;
use std::cell::{OnceCell, RefCell};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

pub use data::context::EvalError;
#[cfg(test)]
use parse::parser::ParseError;
use vm::VM;

/// Internal parser surface re-exported only when the `fuzz_internal` feature
/// is enabled. Used by the `cargo-fuzz` harness to reach the PEG parser
/// without going through `Jetro::collect`. NOT a stable public API.
#[cfg(feature = "fuzz_internal")]
pub mod __fuzz_internal {
    pub use crate::parse::parser::{parse, ParseError};
    pub use crate::plan::physical::plan_query;
}

#[cfg(test)]
#[derive(Debug)]
pub(crate) enum Error {
    Parse(ParseError),
    Eval(EvalError),
}

#[cfg(test)]
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Parse(e) => write!(f, "{}", e),
            Error::Eval(e) => write!(f, "{}", e),
        }
    }
}
#[cfg(test)]
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Parse(e) => Some(e),
            Error::Eval(_) => None,
        }
    }
}

#[cfg(test)]
impl From<ParseError> for Error {
    fn from(e: ParseError) -> Self {
        Error::Parse(e)
    }
}
#[cfg(test)]
impl From<EvalError> for Error {
    fn from(e: EvalError) -> Self {
        Error::Eval(e)
    }
}

/// Primary entry point. Holds a JSON document and evaluates expressions against
/// it. Lazy fields (`root_val`, `tape`, `structural_index`, `objvec_cache`)
/// are populated on first use so callers only pay for the representations a
/// particular query actually needs.
pub struct Jetro {
    /// The `serde_json::Value` root document; unused for byte-backed handles
    /// where the tape is the authoritative source.
    document: Value,
    /// Cached `Val` tree — built once and reused across `collect()` calls.
    root_val: OnceCell<Val>,
    /// Retained raw bytes for lazy tape and structural-index materialisation.
    raw_bytes: Option<Arc<[u8]>>,

    /// Lazily parsed simd-json tape; `Err` is cached to avoid re-parsing after failure.
    tape: OnceCell<std::result::Result<Arc<crate::data::tape::TapeData>, String>>,

    /// Lazily built bitmap structural index for accelerated key-presence queries.
    structural_index:
        OnceCell<std::result::Result<Arc<jetro_experimental::StructuralIndex>, String>>,

    /// Per-document cache from `Arc<Vec<Val>>` pointer addresses to promoted
    /// `ObjVecData` columnar representations; keyed by pointer to avoid re-promotion.
    pub(crate) objvec_cache:
        std::sync::Mutex<std::collections::HashMap<usize, Arc<crate::data::value::ObjVecData>>>,

    /// Per-document VM cache used by `Jetro::collect`; not shared across document handles.
    vm: RefCell<VM>,
}

/// Long-lived multi-document query engine with an explicit plan cache.
/// Use when the same process evaluates many expressions over many documents —
/// parse/lower/compile work is amortised by this object, not hidden in
/// thread-local state.
pub struct JetroEngine {
    /// Maps `"<context_key>\0<expr>"` to compiled `QueryPlan`; evicted wholesale when full.
    plan_cache: Mutex<HashMap<String, ir::physical::QueryPlan>>,
    /// Maximum number of entries before the cache is cleared; 0 disables caching.
    plan_cache_limit: usize,
    /// The shared `VM` used by all `collect*` calls on this engine instance.
    vm: Mutex<VM>,
    /// Engine-owned JSON object-key intern cache. Used by [`JetroEngine::parse_value`]
    /// and [`JetroEngine::parse_bytes`] (and the `collect_*` shortcuts that go through
    /// them) so each engine instance has an isolated key cache. Documents built via
    /// the standalone `Jetro::from_bytes`/`From<serde_json::Value>` paths use the
    /// process-wide [`crate::data::intern::default_cache`] instead.
    keys: Arc<crate::data::intern::KeyCache>,
}

/// Error returned by `JetroEngine::collect_bytes` and similar methods that
/// may fail during JSON parsing or during expression evaluation.
#[derive(Debug)]
pub enum JetroEngineError {
    /// JSON parsing failed before evaluation could begin.
    Json(serde_json::Error),
    /// Reading from a stream or writing results failed.
    Io(std::io::Error),
    /// NDJSON row parsing failed with row context.
    Ndjson(io::RowError),
    /// Expression evaluation failed (the JSON was valid but the query errored).
    Eval(EvalError),
}

impl std::fmt::Display for JetroEngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Json(err) => write!(f, "{}", err),
            Self::Io(err) => write!(f, "{}", err),
            Self::Ndjson(err) => write!(f, "{}", err),
            Self::Eval(err) => write!(f, "{}", err),
        }
    }
}

impl std::error::Error for JetroEngineError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Json(err) => Some(err),
            Self::Io(err) => Some(err),
            Self::Ndjson(err) => Some(err),
            Self::Eval(_) => None,
        }
    }
}

impl From<serde_json::Error> for JetroEngineError {
    fn from(err: serde_json::Error) -> Self {
        Self::Json(err)
    }
}

impl From<std::io::Error> for JetroEngineError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<io::RowError> for JetroEngineError {
    fn from(err: io::RowError) -> Self {
        Self::Ndjson(err)
    }
}

impl From<EvalError> for JetroEngineError {
    fn from(err: EvalError) -> Self {
        Self::Eval(err)
    }
}

impl Default for JetroEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl JetroEngine {
    /// Default maximum plan-cache size; the cache is cleared wholesale when reached.
    const DEFAULT_PLAN_CACHE_LIMIT: usize = 256;

    /// Create a `JetroEngine` with the default plan-cache limit of 256 entries.
    pub fn new() -> Self {
        Self::with_plan_cache_limit(Self::DEFAULT_PLAN_CACHE_LIMIT)
    }

    /// Create a `JetroEngine` with an explicit plan-cache capacity.
    /// Set `plan_cache_limit` to 0 to disable caching entirely.
    pub fn with_plan_cache_limit(plan_cache_limit: usize) -> Self {
        Self {
            plan_cache: Mutex::new(HashMap::new()),
            plan_cache_limit,
            vm: Mutex::new(VM::new()),
            keys: crate::data::intern::KeyCache::new(),
        }
    }

    /// Borrow this engine's JSON key-intern cache.
    pub fn keys(&self) -> &Arc<crate::data::intern::KeyCache> {
        &self.keys
    }

    /// Discard all cached query plans and the engine's key-intern cache,
    /// forcing re-compilation and re-interning on the next call.
    pub fn clear_cache(&self) {
        self.plan_cache.lock().expect("plan cache poisoned").clear();
        self.keys.clear();
    }

    /// Build a `Jetro` document from a `serde_json::Value` with object keys
    /// interned into this engine's key cache. Use this in place of
    /// `Jetro::from(...)` / the `From<serde_json::Value>` impl when
    /// per-engine key isolation is required.
    pub fn parse_value(&self, document: Value) -> Jetro {
        let root = Val::from_value_with(&self.keys, &document);
        Jetro::from_val_and_value(root, document)
    }

    /// Parse raw JSON bytes into a `Jetro` document with object keys
    /// interned into this engine's key cache. With `simd-json`, the tape
    /// is materialised eagerly so interning happens once at parse time
    /// (subsequent `collect` calls reuse the cached `Val` tree).
    pub fn parse_bytes(&self, bytes: Vec<u8>) -> std::result::Result<Jetro, JetroEngineError> {
        let document = Jetro::from_bytes(bytes)?;
        // Force materialisation so keys are interned through this
        // engine's cache rather than the default thread-local one when
        // `collect` later asks for `root_val`.
        let _ = document.root_val_with(&self.keys)?;
        Ok(document)
    }

    /// Parse raw JSON bytes into a `Jetro` document without forcing the `Val`
    /// tree. This keeps byte-backed callers eligible for tape/view execution;
    /// object keys are interned only if execution later materialises the row.
    pub(crate) fn parse_bytes_lazy(
        &self,
        bytes: Vec<u8>,
    ) -> std::result::Result<Jetro, JetroEngineError> {
        Ok(Jetro::from_bytes(bytes)?)
    }

    /// Evaluate a Jetro expression against an already-constructed `Jetro` document,
    /// using the engine's shared plan cache and `VM`.
    pub fn collect<S: AsRef<str>>(
        &self,
        document: &Jetro,
        expr: S,
    ) -> std::result::Result<Value, EvalError> {
        let expr = expr.as_ref();
        if let Some(rows) = io::collect_document_rows(self, document, expr)? {
            return Ok(Value::from(rows));
        }
        let plan = self.cached_plan(expr, exec::router::planning_context(document));
        self.collect_prepared(document, &plan)
    }

    pub(crate) fn collect_prepared(
        &self,
        document: &Jetro,
        plan: &ir::physical::QueryPlan,
    ) -> std::result::Result<Value, EvalError> {
        self.collect_prepared_val(document, plan).map(Value::from)
    }

    pub(crate) fn collect_prepared_val(
        &self,
        document: &Jetro,
        plan: &ir::physical::QueryPlan,
    ) -> std::result::Result<Val, EvalError> {
        let mut vm = self.vm.lock().expect("vm cache poisoned");
        exec::router::collect_plan_val_with_vm(document, plan, &mut vm)
    }

    pub(crate) fn lock_vm(&self) -> std::sync::MutexGuard<'_, VM> {
        self.vm.lock().expect("vm cache poisoned")
    }

    /// Convenience wrapper: wrap a `serde_json::Value` in a `Jetro` and evaluate `expr`.
    /// Routes through [`JetroEngine::parse_value`] so the document's object keys are
    /// interned into this engine's key cache.
    pub fn collect_value<S: AsRef<str>>(
        &self,
        document: Value,
        expr: S,
    ) -> std::result::Result<Value, EvalError> {
        let document = self.parse_value(document);
        self.collect(&document, expr)
    }

    /// Parse raw JSON bytes into a `Jetro` document and evaluate `expr`,
    /// returning a `JetroEngineError` on either parse or evaluation failure.
    /// Routes through [`JetroEngine::parse_bytes`] so the document's object keys
    /// are interned into this engine's key cache.
    pub fn collect_bytes<S: AsRef<str>>(
        &self,
        bytes: Vec<u8>,
        expr: S,
    ) -> std::result::Result<Value, JetroEngineError> {
        let document = self.parse_bytes(bytes)?;
        Ok(self.collect(&document, expr)?)
    }

    /// Evaluate `query` independently for every non-empty NDJSON row and write
    /// one JSON result per output line.
    pub fn run_ndjson<R, W>(
        &self,
        reader: R,
        query: &str,
        writer: W,
    ) -> std::result::Result<usize, JetroEngineError>
    where
        R: std::io::BufRead,
        W: std::io::Write,
    {
        io::run_ndjson(self, reader, query, writer)
    }

    /// Open an NDJSON file and evaluate `query` independently for every
    /// non-empty row, writing one JSON result per output line.
    pub fn run_ndjson_file<P, W>(
        &self,
        path: P,
        query: &str,
        writer: W,
    ) -> std::result::Result<usize, JetroEngineError>
    where
        P: AsRef<std::path::Path>,
        W: std::io::Write,
    {
        io::run_ndjson_file(self, path, query, writer)
    }

    /// Like [`JetroEngine::run_ndjson_file`] with explicit NDJSON reader options.
    pub fn run_ndjson_file_with_options<P, W>(
        &self,
        path: P,
        query: &str,
        writer: W,
        options: io::NdjsonOptions,
    ) -> std::result::Result<usize, JetroEngineError>
    where
        P: AsRef<std::path::Path>,
        W: std::io::Write,
    {
        io::run_ndjson_file_with_options(self, path, query, writer, options)
    }

    /// Evaluate a file-backed NDJSON query and return a route/counter report.
    pub fn run_ndjson_file_with_report<P, W>(
        &self,
        path: P,
        query: &str,
        writer: W,
    ) -> std::result::Result<io::NdjsonExecutionReport, JetroEngineError>
    where
        P: AsRef<std::path::Path>,
        W: std::io::Write,
    {
        io::run_ndjson_file_with_report(self, path, query, writer)
    }

    /// Like [`JetroEngine::run_ndjson_file_with_report`] with explicit options.
    pub fn run_ndjson_file_with_report_and_options<P, W>(
        &self,
        path: P,
        query: &str,
        writer: W,
        options: io::NdjsonOptions,
    ) -> std::result::Result<io::NdjsonExecutionReport, JetroEngineError>
    where
        P: AsRef<std::path::Path>,
        W: std::io::Write,
    {
        io::run_ndjson_file_with_report_and_options(self, path, query, writer, options)
    }

    /// Open an NDJSON file, write at most `limit` query results, and stop reading.
    pub fn run_ndjson_file_limit<P, W>(
        &self,
        path: P,
        query: &str,
        limit: usize,
        writer: W,
    ) -> std::result::Result<usize, JetroEngineError>
    where
        P: AsRef<std::path::Path>,
        W: std::io::Write,
    {
        io::run_ndjson_file_limit(self, path, query, limit, writer)
    }

    /// Like [`JetroEngine::run_ndjson_file_limit`] with explicit NDJSON reader options.
    pub fn run_ndjson_file_limit_with_options<P, W>(
        &self,
        path: P,
        query: &str,
        limit: usize,
        writer: W,
        options: io::NdjsonOptions,
    ) -> std::result::Result<usize, JetroEngineError>
    where
        P: AsRef<std::path::Path>,
        W: std::io::Write,
    {
        io::run_ndjson_file_limit_with_options(self, path, query, limit, writer, options)
    }

    /// Evaluate a limited file-backed NDJSON query and return a route/counter report.
    pub fn run_ndjson_file_limit_with_report<P, W>(
        &self,
        path: P,
        query: &str,
        limit: usize,
        writer: W,
    ) -> std::result::Result<io::NdjsonExecutionReport, JetroEngineError>
    where
        P: AsRef<std::path::Path>,
        W: std::io::Write,
    {
        io::run_ndjson_file_limit_with_report(self, path, query, limit, writer)
    }

    /// Like [`JetroEngine::run_ndjson_file_limit_with_report`] with explicit options.
    pub fn run_ndjson_file_limit_with_report_and_options<P, W>(
        &self,
        path: P,
        query: &str,
        limit: usize,
        writer: W,
        options: io::NdjsonOptions,
    ) -> std::result::Result<io::NdjsonExecutionReport, JetroEngineError>
    where
        P: AsRef<std::path::Path>,
        W: std::io::Write,
    {
        io::run_ndjson_file_limit_with_report_and_options(self, path, query, limit, writer, options)
    }

    /// Evaluate `query` independently for every row from an [`io::NdjsonSource`].
    pub fn run_ndjson_source<W>(
        &self,
        source: io::NdjsonSource,
        query: &str,
        writer: W,
    ) -> std::result::Result<usize, JetroEngineError>
    where
        W: std::io::Write,
    {
        io::run_ndjson_source(self, source, query, writer)
    }

    /// Like [`JetroEngine::run_ndjson_source`] with explicit NDJSON reader options.
    pub fn run_ndjson_source_with_options<W>(
        &self,
        source: io::NdjsonSource,
        query: &str,
        writer: W,
        options: io::NdjsonOptions,
    ) -> std::result::Result<usize, JetroEngineError>
    where
        W: std::io::Write,
    {
        io::run_ndjson_source_with_options(self, source, query, writer, options)
    }

    /// Evaluate an [`io::NdjsonSource`] query and return a route/counter report.
    pub fn run_ndjson_source_with_report<W>(
        &self,
        source: io::NdjsonSource,
        query: &str,
        writer: W,
    ) -> std::result::Result<io::NdjsonExecutionReport, JetroEngineError>
    where
        W: std::io::Write,
    {
        io::run_ndjson_source_with_report(self, source, query, writer)
    }

    /// Like [`JetroEngine::run_ndjson_source_with_report`] with explicit options.
    pub fn run_ndjson_source_with_report_and_options<W>(
        &self,
        source: io::NdjsonSource,
        query: &str,
        writer: W,
        options: io::NdjsonOptions,
    ) -> std::result::Result<io::NdjsonExecutionReport, JetroEngineError>
    where
        W: std::io::Write,
    {
        io::run_ndjson_source_with_report_and_options(self, source, query, writer, options)
    }

    /// Evaluate `query` for rows from an [`io::NdjsonSource`], write at most
    /// `limit` results, and stop reading.
    pub fn run_ndjson_source_limit<W>(
        &self,
        source: io::NdjsonSource,
        query: &str,
        limit: usize,
        writer: W,
    ) -> std::result::Result<usize, JetroEngineError>
    where
        W: std::io::Write,
    {
        io::run_ndjson_source_limit(self, source, query, limit, writer)
    }

    /// Like [`JetroEngine::run_ndjson_source_limit`] with explicit NDJSON reader options.
    pub fn run_ndjson_source_limit_with_options<W>(
        &self,
        source: io::NdjsonSource,
        query: &str,
        limit: usize,
        writer: W,
        options: io::NdjsonOptions,
    ) -> std::result::Result<usize, JetroEngineError>
    where
        W: std::io::Write,
    {
        io::run_ndjson_source_limit_with_options(self, source, query, limit, writer, options)
    }

    /// Evaluate a limited [`io::NdjsonSource`] query and return a route/counter report.
    pub fn run_ndjson_source_limit_with_report<W>(
        &self,
        source: io::NdjsonSource,
        query: &str,
        limit: usize,
        writer: W,
    ) -> std::result::Result<io::NdjsonExecutionReport, JetroEngineError>
    where
        W: std::io::Write,
    {
        io::run_ndjson_source_limit_with_report(self, source, query, limit, writer)
    }

    /// Like [`JetroEngine::run_ndjson_source_limit_with_report`] with explicit options.
    pub fn run_ndjson_source_limit_with_report_and_options<W>(
        &self,
        source: io::NdjsonSource,
        query: &str,
        limit: usize,
        writer: W,
        options: io::NdjsonOptions,
    ) -> std::result::Result<io::NdjsonExecutionReport, JetroEngineError>
    where
        W: std::io::Write,
    {
        io::run_ndjson_source_limit_with_report_and_options(
            self, source, query, limit, writer, options,
        )
    }

    /// Read an NDJSON file from tail to head and write one query result per row.
    pub fn run_ndjson_rev<P, W>(
        &self,
        path: P,
        query: &str,
        writer: W,
    ) -> std::result::Result<usize, JetroEngineError>
    where
        P: AsRef<std::path::Path>,
        W: std::io::Write,
    {
        io::run_ndjson_rev(self, path, query, writer)
    }

    /// Like [`JetroEngine::run_ndjson_rev`] with explicit NDJSON reader options.
    pub fn run_ndjson_rev_with_options<P, W>(
        &self,
        path: P,
        query: &str,
        writer: W,
        options: io::NdjsonOptions,
    ) -> std::result::Result<usize, JetroEngineError>
    where
        P: AsRef<std::path::Path>,
        W: std::io::Write,
    {
        io::run_ndjson_rev_with_options(self, path, query, writer, options)
    }

    /// Read an NDJSON file from tail to head, write at most `limit` query
    /// results, and stop reading.
    pub fn run_ndjson_rev_limit<P, W>(
        &self,
        path: P,
        query: &str,
        limit: usize,
        writer: W,
    ) -> std::result::Result<usize, JetroEngineError>
    where
        P: AsRef<std::path::Path>,
        W: std::io::Write,
    {
        io::run_ndjson_rev_limit(self, path, query, limit, writer)
    }

    /// Like [`JetroEngine::run_ndjson_rev_limit`] with explicit NDJSON reader options.
    pub fn run_ndjson_rev_limit_with_options<P, W>(
        &self,
        path: P,
        query: &str,
        limit: usize,
        writer: W,
        options: io::NdjsonOptions,
    ) -> std::result::Result<usize, JetroEngineError>
    where
        P: AsRef<std::path::Path>,
        W: std::io::Write,
    {
        io::run_ndjson_rev_limit_with_options(self, path, query, limit, writer, options)
    }

    /// Read an NDJSON file from tail to head, keep only the first row seen for
    /// each `key_query` result in that reverse stream order, write `query` for
    /// retained rows, and stop after `limit` retained rows.
    pub fn run_ndjson_rev_distinct_by<P, W>(
        &self,
        path: P,
        key_query: &str,
        query: &str,
        limit: usize,
        writer: W,
    ) -> std::result::Result<usize, JetroEngineError>
    where
        P: AsRef<std::path::Path>,
        W: std::io::Write,
    {
        io::run_ndjson_rev_distinct_by(self, path, key_query, query, limit, writer)
    }

    /// Like [`JetroEngine::run_ndjson_rev_distinct_by`] with explicit NDJSON
    /// reader options.
    pub fn run_ndjson_rev_distinct_by_with_options<P, W>(
        &self,
        path: P,
        key_query: &str,
        query: &str,
        limit: usize,
        writer: W,
        options: io::NdjsonOptions,
    ) -> std::result::Result<usize, JetroEngineError>
    where
        P: AsRef<std::path::Path>,
        W: std::io::Write,
    {
        io::run_ndjson_rev_distinct_by_with_options(
            self, path, key_query, query, limit, writer, options,
        )
    }

    /// Like [`JetroEngine::run_ndjson_rev_distinct_by`], returning execution
    /// counters for path-selection and duplicate-drop observability.
    pub fn run_ndjson_rev_distinct_by_with_stats<P, W>(
        &self,
        path: P,
        key_query: &str,
        query: &str,
        limit: usize,
        writer: W,
    ) -> std::result::Result<io::NdjsonRevDistinctStats, JetroEngineError>
    where
        P: AsRef<std::path::Path>,
        W: std::io::Write,
    {
        io::run_ndjson_rev_distinct_by_with_stats(self, path, key_query, query, limit, writer)
    }

    /// Like [`JetroEngine::run_ndjson_rev_distinct_by_with_stats`] with explicit
    /// NDJSON reader options.
    pub fn run_ndjson_rev_distinct_by_with_stats_and_options<P, W>(
        &self,
        path: P,
        key_query: &str,
        query: &str,
        limit: usize,
        writer: W,
        options: io::NdjsonOptions,
    ) -> std::result::Result<io::NdjsonRevDistinctStats, JetroEngineError>
    where
        P: AsRef<std::path::Path>,
        W: std::io::Write,
    {
        io::run_ndjson_rev_distinct_by_with_stats_and_options(
            self, path, key_query, query, limit, writer, options,
        )
    }

    /// Like [`JetroEngine::run_ndjson_rev_distinct_by`], returning the shared
    /// NDJSON execution report shape.
    pub fn run_ndjson_rev_distinct_by_with_report<P, W>(
        &self,
        path: P,
        key_query: &str,
        query: &str,
        limit: usize,
        writer: W,
    ) -> std::result::Result<io::NdjsonExecutionReport, JetroEngineError>
    where
        P: AsRef<std::path::Path>,
        W: std::io::Write,
    {
        io::run_ndjson_rev_distinct_by_with_report(
            self, path, key_query, query, limit, writer,
        )
    }

    /// Like [`JetroEngine::run_ndjson_rev_distinct_by_with_report`] with explicit options.
    pub fn run_ndjson_rev_distinct_by_with_report_and_options<P, W>(
        &self,
        path: P,
        key_query: &str,
        query: &str,
        limit: usize,
        writer: W,
        options: io::NdjsonOptions,
    ) -> std::result::Result<io::NdjsonExecutionReport, JetroEngineError>
    where
        P: AsRef<std::path::Path>,
        W: std::io::Write,
    {
        io::run_ndjson_rev_distinct_by_with_report_and_options(
            self, path, key_query, query, limit, writer, options,
        )
    }

    /// Like [`JetroEngine::run_ndjson`] with explicit NDJSON reader options.
    pub fn run_ndjson_with_options<R, W>(
        &self,
        reader: R,
        query: &str,
        writer: W,
        options: io::NdjsonOptions,
    ) -> std::result::Result<usize, JetroEngineError>
    where
        R: std::io::BufRead,
        W: std::io::Write,
    {
        io::run_ndjson_with_options(self, reader, query, writer, options)
    }

    /// Evaluate `query` for NDJSON rows and return a route/counter report.
    pub fn run_ndjson_with_report<R, W>(
        &self,
        reader: R,
        query: &str,
        writer: W,
    ) -> std::result::Result<io::NdjsonExecutionReport, JetroEngineError>
    where
        R: std::io::BufRead,
        W: std::io::Write,
    {
        io::run_ndjson_with_report(self, reader, query, writer)
    }

    /// Like [`JetroEngine::run_ndjson_with_report`] with explicit NDJSON reader options.
    pub fn run_ndjson_with_report_and_options<R, W>(
        &self,
        reader: R,
        query: &str,
        writer: W,
        options: io::NdjsonOptions,
    ) -> std::result::Result<io::NdjsonExecutionReport, JetroEngineError>
    where
        R: std::io::BufRead,
        W: std::io::Write,
    {
        io::run_ndjson_with_report_and_options(self, reader, query, writer, options)
    }

    /// Evaluate `query` for NDJSON rows, write at most `limit` results, and stop reading.
    pub fn run_ndjson_limit<R, W>(
        &self,
        reader: R,
        query: &str,
        limit: usize,
        writer: W,
    ) -> std::result::Result<usize, JetroEngineError>
    where
        R: std::io::BufRead,
        W: std::io::Write,
    {
        io::run_ndjson_limit(self, reader, query, limit, writer)
    }

    /// Like [`JetroEngine::run_ndjson_limit`] with explicit NDJSON reader options.
    pub fn run_ndjson_limit_with_options<R, W>(
        &self,
        reader: R,
        query: &str,
        limit: usize,
        writer: W,
        options: io::NdjsonOptions,
    ) -> std::result::Result<usize, JetroEngineError>
    where
        R: std::io::BufRead,
        W: std::io::Write,
    {
        io::run_ndjson_limit_with_options(self, reader, query, limit, writer, options)
    }

    /// Evaluate a limited NDJSON reader query and return a route/counter report.
    pub fn run_ndjson_limit_with_report<R, W>(
        &self,
        reader: R,
        query: &str,
        limit: usize,
        writer: W,
    ) -> std::result::Result<io::NdjsonExecutionReport, JetroEngineError>
    where
        R: std::io::BufRead,
        W: std::io::Write,
    {
        io::run_ndjson_limit_with_report(self, reader, query, limit, writer)
    }

    /// Like [`JetroEngine::run_ndjson_limit_with_report`] with explicit options.
    pub fn run_ndjson_limit_with_report_and_options<R, W>(
        &self,
        reader: R,
        query: &str,
        limit: usize,
        writer: W,
        options: io::NdjsonOptions,
    ) -> std::result::Result<io::NdjsonExecutionReport, JetroEngineError>
    where
        R: std::io::BufRead,
        W: std::io::Write,
    {
        io::run_ndjson_limit_with_report_and_options(self, reader, query, limit, writer, options)
    }

    /// Evaluate `predicate` for each NDJSON row, write matching original rows,
    /// and stop after `limit` matches.
    pub fn run_ndjson_matches<R, W>(
        &self,
        reader: R,
        predicate: &str,
        limit: usize,
        writer: W,
    ) -> std::result::Result<usize, JetroEngineError>
    where
        R: std::io::BufRead,
        W: std::io::Write,
    {
        io::run_ndjson_matches(self, reader, predicate, limit, writer)
    }

    /// Like [`JetroEngine::run_ndjson_matches`] with explicit NDJSON reader options.
    pub fn run_ndjson_matches_with_options<R, W>(
        &self,
        reader: R,
        predicate: &str,
        limit: usize,
        writer: W,
        options: io::NdjsonOptions,
    ) -> std::result::Result<usize, JetroEngineError>
    where
        R: std::io::BufRead,
        W: std::io::Write,
    {
        io::run_ndjson_matches_with_options(self, reader, predicate, limit, writer, options)
    }

    /// Evaluate a match-limited NDJSON query and return the shared execution report.
    pub fn run_ndjson_matches_with_report<R, W>(
        &self,
        reader: R,
        predicate: &str,
        limit: usize,
        writer: W,
    ) -> std::result::Result<io::NdjsonExecutionReport, JetroEngineError>
    where
        R: std::io::BufRead,
        W: std::io::Write,
    {
        io::run_ndjson_matches_with_report(self, reader, predicate, limit, writer)
    }

    /// Like [`JetroEngine::run_ndjson_matches_with_report`] with explicit options.
    pub fn run_ndjson_matches_with_report_and_options<R, W>(
        &self,
        reader: R,
        predicate: &str,
        limit: usize,
        writer: W,
        options: io::NdjsonOptions,
    ) -> std::result::Result<io::NdjsonExecutionReport, JetroEngineError>
    where
        R: std::io::BufRead,
        W: std::io::Write,
    {
        io::run_ndjson_matches_with_report_and_options(
            self, reader, predicate, limit, writer, options,
        )
    }

    /// Open an NDJSON file, write matching original rows, and stop after `limit` matches.
    pub fn run_ndjson_matches_file<P, W>(
        &self,
        path: P,
        predicate: &str,
        limit: usize,
        writer: W,
    ) -> std::result::Result<usize, JetroEngineError>
    where
        P: AsRef<std::path::Path>,
        W: std::io::Write,
    {
        io::run_ndjson_matches_file(self, path, predicate, limit, writer)
    }

    /// Like [`JetroEngine::run_ndjson_matches_file`] with explicit NDJSON reader options.
    pub fn run_ndjson_matches_file_with_options<P, W>(
        &self,
        path: P,
        predicate: &str,
        limit: usize,
        writer: W,
        options: io::NdjsonOptions,
    ) -> std::result::Result<usize, JetroEngineError>
    where
        P: AsRef<std::path::Path>,
        W: std::io::Write,
    {
        io::run_ndjson_matches_file_with_options(self, path, predicate, limit, writer, options)
    }

    /// Like [`JetroEngine::run_ndjson_matches_file`], returning the shared report.
    pub fn run_ndjson_matches_file_with_report<P, W>(
        &self,
        path: P,
        predicate: &str,
        limit: usize,
        writer: W,
    ) -> std::result::Result<io::NdjsonExecutionReport, JetroEngineError>
    where
        P: AsRef<std::path::Path>,
        W: std::io::Write,
    {
        io::run_ndjson_matches_file_with_report(self, path, predicate, limit, writer)
    }

    /// Like [`JetroEngine::run_ndjson_matches_file_with_report`] with explicit options.
    pub fn run_ndjson_matches_file_with_report_and_options<P, W>(
        &self,
        path: P,
        predicate: &str,
        limit: usize,
        writer: W,
        options: io::NdjsonOptions,
    ) -> std::result::Result<io::NdjsonExecutionReport, JetroEngineError>
    where
        P: AsRef<std::path::Path>,
        W: std::io::Write,
    {
        io::run_ndjson_matches_file_with_report_and_options(
            self, path, predicate, limit, writer, options,
        )
    }

    /// Evaluate `predicate` against each row from an [`io::NdjsonSource`], write
    /// matching original rows, and stop after `limit` matches.
    pub fn run_ndjson_matches_source<W>(
        &self,
        source: io::NdjsonSource,
        predicate: &str,
        limit: usize,
        writer: W,
    ) -> std::result::Result<usize, JetroEngineError>
    where
        W: std::io::Write,
    {
        io::run_ndjson_matches_source(self, source, predicate, limit, writer)
    }

    /// Like [`JetroEngine::run_ndjson_matches_source`] with explicit NDJSON reader options.
    pub fn run_ndjson_matches_source_with_options<W>(
        &self,
        source: io::NdjsonSource,
        predicate: &str,
        limit: usize,
        writer: W,
        options: io::NdjsonOptions,
    ) -> std::result::Result<usize, JetroEngineError>
    where
        W: std::io::Write,
    {
        io::run_ndjson_matches_source_with_options(self, source, predicate, limit, writer, options)
    }

    /// Like [`JetroEngine::run_ndjson_matches_source`], returning the shared report.
    pub fn run_ndjson_matches_source_with_report<W>(
        &self,
        source: io::NdjsonSource,
        predicate: &str,
        limit: usize,
        writer: W,
    ) -> std::result::Result<io::NdjsonExecutionReport, JetroEngineError>
    where
        W: std::io::Write,
    {
        io::run_ndjson_matches_source_with_report(self, source, predicate, limit, writer)
    }

    /// Like [`JetroEngine::run_ndjson_matches_source_with_report`] with explicit options.
    pub fn run_ndjson_matches_source_with_report_and_options<W>(
        &self,
        source: io::NdjsonSource,
        predicate: &str,
        limit: usize,
        writer: W,
        options: io::NdjsonOptions,
    ) -> std::result::Result<io::NdjsonExecutionReport, JetroEngineError>
    where
        W: std::io::Write,
    {
        io::run_ndjson_matches_source_with_report_and_options(
            self, source, predicate, limit, writer, options,
        )
    }

    /// Read an NDJSON file from tail to head, write matching original rows, and
    /// stop after `limit` matches.
    pub fn run_ndjson_rev_matches<P, W>(
        &self,
        path: P,
        predicate: &str,
        limit: usize,
        writer: W,
    ) -> std::result::Result<usize, JetroEngineError>
    where
        P: AsRef<std::path::Path>,
        W: std::io::Write,
    {
        io::run_ndjson_rev_matches(self, path, predicate, limit, writer)
    }

    /// Like [`JetroEngine::run_ndjson_rev_matches`] with explicit NDJSON reader options.
    pub fn run_ndjson_rev_matches_with_options<P, W>(
        &self,
        path: P,
        predicate: &str,
        limit: usize,
        writer: W,
        options: io::NdjsonOptions,
    ) -> std::result::Result<usize, JetroEngineError>
    where
        P: AsRef<std::path::Path>,
        W: std::io::Write,
    {
        io::run_ndjson_rev_matches_with_options(self, path, predicate, limit, writer, options)
    }

    /// Like [`JetroEngine::run_ndjson_rev_matches`], returning the shared report.
    pub fn run_ndjson_rev_matches_with_report<P, W>(
        &self,
        path: P,
        predicate: &str,
        limit: usize,
        writer: W,
    ) -> std::result::Result<io::NdjsonExecutionReport, JetroEngineError>
    where
        P: AsRef<std::path::Path>,
        W: std::io::Write,
    {
        io::run_ndjson_rev_matches_with_report(self, path, predicate, limit, writer)
    }

    /// Like [`JetroEngine::run_ndjson_rev_matches_with_report`] with explicit options.
    pub fn run_ndjson_rev_matches_with_report_and_options<P, W>(
        &self,
        path: P,
        predicate: &str,
        limit: usize,
        writer: W,
        options: io::NdjsonOptions,
    ) -> std::result::Result<io::NdjsonExecutionReport, JetroEngineError>
    where
        P: AsRef<std::path::Path>,
        W: std::io::Write,
    {
        io::run_ndjson_rev_matches_with_report_and_options(
            self, path, predicate, limit, writer, options,
        )
    }

    /// Evaluate `query` independently for every non-empty NDJSON row and collect
    /// the per-row results.
    pub fn collect_ndjson<R>(
        &self,
        reader: R,
        query: &str,
    ) -> std::result::Result<Vec<Value>, JetroEngineError>
    where
        R: std::io::BufRead,
    {
        io::collect_ndjson(self, reader, query)
    }

    /// Open an NDJSON file and collect per-row query results.
    pub fn collect_ndjson_file<P>(
        &self,
        path: P,
        query: &str,
    ) -> std::result::Result<Vec<Value>, JetroEngineError>
    where
        P: AsRef<std::path::Path>,
    {
        io::collect_ndjson_file(self, path, query)
    }

    /// Like [`JetroEngine::collect_ndjson_file`] with explicit NDJSON reader options.
    pub fn collect_ndjson_file_with_options<P>(
        &self,
        path: P,
        query: &str,
        options: io::NdjsonOptions,
    ) -> std::result::Result<Vec<Value>, JetroEngineError>
    where
        P: AsRef<std::path::Path>,
    {
        io::collect_ndjson_file_with_options(self, path, query, options)
    }

    /// Collect per-row query results from an [`io::NdjsonSource`].
    pub fn collect_ndjson_source(
        &self,
        source: io::NdjsonSource,
        query: &str,
    ) -> std::result::Result<Vec<Value>, JetroEngineError> {
        io::collect_ndjson_source(self, source, query)
    }

    /// Like [`JetroEngine::collect_ndjson_source`] with explicit NDJSON reader options.
    pub fn collect_ndjson_source_with_options(
        &self,
        source: io::NdjsonSource,
        query: &str,
        options: io::NdjsonOptions,
    ) -> std::result::Result<Vec<Value>, JetroEngineError> {
        io::collect_ndjson_source_with_options(self, source, query, options)
    }

    /// Read an NDJSON file from tail to head and collect per-row query results.
    pub fn collect_ndjson_rev<P>(
        &self,
        path: P,
        query: &str,
    ) -> std::result::Result<Vec<Value>, JetroEngineError>
    where
        P: AsRef<std::path::Path>,
    {
        io::collect_ndjson_rev(self, path, query)
    }

    /// Like [`JetroEngine::collect_ndjson_rev`] with explicit NDJSON reader options.
    pub fn collect_ndjson_rev_with_options<P>(
        &self,
        path: P,
        query: &str,
        options: io::NdjsonOptions,
    ) -> std::result::Result<Vec<Value>, JetroEngineError>
    where
        P: AsRef<std::path::Path>,
    {
        io::collect_ndjson_rev_with_options(self, path, query, options)
    }

    /// Read an NDJSON file from tail to head and call `f` with each query result
    /// as it is produced.
    pub fn for_each_ndjson_rev<P, F>(
        &self,
        path: P,
        query: &str,
        f: F,
    ) -> std::result::Result<usize, JetroEngineError>
    where
        P: AsRef<std::path::Path>,
        F: FnMut(Value),
    {
        io::for_each_ndjson_rev(self, path, query, f)
    }

    /// Read an NDJSON file from tail to head and call `f` until it returns
    /// [`io::NdjsonControl::Stop`] or input is exhausted.
    pub fn for_each_ndjson_rev_until<P, F>(
        &self,
        path: P,
        query: &str,
        f: F,
    ) -> std::result::Result<usize, JetroEngineError>
    where
        P: AsRef<std::path::Path>,
        F: FnMut(Value) -> std::result::Result<io::NdjsonControl, JetroEngineError>,
    {
        io::for_each_ndjson_rev_with_options(self, path, query, io::NdjsonOptions::default(), f)
    }

    /// Like [`JetroEngine::for_each_ndjson_rev_until`] with explicit NDJSON reader options.
    pub fn for_each_ndjson_rev_until_with_options<P, F>(
        &self,
        path: P,
        query: &str,
        options: io::NdjsonOptions,
        f: F,
    ) -> std::result::Result<usize, JetroEngineError>
    where
        P: AsRef<std::path::Path>,
        F: FnMut(Value) -> std::result::Result<io::NdjsonControl, JetroEngineError>,
    {
        io::for_each_ndjson_rev_with_options(self, path, query, options, f)
    }

    /// Like [`JetroEngine::for_each_ndjson_rev`] with explicit NDJSON reader options.
    pub fn for_each_ndjson_rev_with_options<P, F>(
        &self,
        path: P,
        query: &str,
        options: io::NdjsonOptions,
        mut f: F,
    ) -> std::result::Result<usize, JetroEngineError>
    where
        P: AsRef<std::path::Path>,
        F: FnMut(Value),
    {
        io::for_each_ndjson_rev_with_options(self, path, query, options, |value| {
            f(value);
            Ok(io::NdjsonControl::Continue)
        })
    }

    /// Like [`JetroEngine::collect_ndjson`] with explicit NDJSON reader options.
    pub fn collect_ndjson_with_options<R>(
        &self,
        reader: R,
        query: &str,
        options: io::NdjsonOptions,
    ) -> std::result::Result<Vec<Value>, JetroEngineError>
    where
        R: std::io::BufRead,
    {
        io::collect_ndjson_with_options(self, reader, query, options)
    }

    /// Evaluate `predicate` for each NDJSON row, collect matching original
    /// rows, and stop after `limit` matches.
    pub fn collect_ndjson_matches<R>(
        &self,
        reader: R,
        predicate: &str,
        limit: usize,
    ) -> std::result::Result<Vec<Value>, JetroEngineError>
    where
        R: std::io::BufRead,
    {
        io::collect_ndjson_matches(self, reader, predicate, limit)
    }

    /// Like [`JetroEngine::collect_ndjson_matches`] with explicit NDJSON reader options.
    pub fn collect_ndjson_matches_with_options<R>(
        &self,
        reader: R,
        predicate: &str,
        limit: usize,
        options: io::NdjsonOptions,
    ) -> std::result::Result<Vec<Value>, JetroEngineError>
    where
        R: std::io::BufRead,
    {
        io::collect_ndjson_matches_with_options(self, reader, predicate, limit, options)
    }

    /// Open an NDJSON file, collect matching original rows, and stop after `limit` matches.
    pub fn collect_ndjson_matches_file<P>(
        &self,
        path: P,
        predicate: &str,
        limit: usize,
    ) -> std::result::Result<Vec<Value>, JetroEngineError>
    where
        P: AsRef<std::path::Path>,
    {
        io::collect_ndjson_matches_file(self, path, predicate, limit)
    }

    /// Like [`JetroEngine::collect_ndjson_matches_file`] with explicit NDJSON reader options.
    pub fn collect_ndjson_matches_file_with_options<P>(
        &self,
        path: P,
        predicate: &str,
        limit: usize,
        options: io::NdjsonOptions,
    ) -> std::result::Result<Vec<Value>, JetroEngineError>
    where
        P: AsRef<std::path::Path>,
    {
        io::collect_ndjson_matches_file_with_options(self, path, predicate, limit, options)
    }

    /// Evaluate `predicate` against each row from an [`io::NdjsonSource`],
    /// collect matching original rows, and stop after `limit` matches.
    pub fn collect_ndjson_matches_source(
        &self,
        source: io::NdjsonSource,
        predicate: &str,
        limit: usize,
    ) -> std::result::Result<Vec<Value>, JetroEngineError> {
        io::collect_ndjson_matches_source(self, source, predicate, limit)
    }

    /// Like [`JetroEngine::collect_ndjson_matches_source`] with explicit NDJSON reader options.
    pub fn collect_ndjson_matches_source_with_options(
        &self,
        source: io::NdjsonSource,
        predicate: &str,
        limit: usize,
        options: io::NdjsonOptions,
    ) -> std::result::Result<Vec<Value>, JetroEngineError> {
        io::collect_ndjson_matches_source_with_options(self, source, predicate, limit, options)
    }

    /// Read an NDJSON file from tail to head, collect matching original rows,
    /// and stop after `limit` matches.
    pub fn collect_ndjson_rev_matches<P>(
        &self,
        path: P,
        predicate: &str,
        limit: usize,
    ) -> std::result::Result<Vec<Value>, JetroEngineError>
    where
        P: AsRef<std::path::Path>,
    {
        io::collect_ndjson_rev_matches(self, path, predicate, limit)
    }

    /// Like [`JetroEngine::collect_ndjson_rev_matches`] with explicit NDJSON reader options.
    pub fn collect_ndjson_rev_matches_with_options<P>(
        &self,
        path: P,
        predicate: &str,
        limit: usize,
        options: io::NdjsonOptions,
    ) -> std::result::Result<Vec<Value>, JetroEngineError>
    where
        P: AsRef<std::path::Path>,
    {
        io::collect_ndjson_rev_matches_with_options(self, path, predicate, limit, options)
    }

    /// Evaluate `query` independently for every non-empty NDJSON row and call
    /// `f` with each result as it is produced.
    pub fn for_each_ndjson<R, F>(
        &self,
        reader: R,
        query: &str,
        f: F,
    ) -> std::result::Result<usize, JetroEngineError>
    where
        R: std::io::BufRead,
        F: FnMut(Value),
    {
        io::for_each_ndjson(self, reader, query, f)
    }

    /// Evaluate `query` independently for every non-empty NDJSON row and call
    /// `f` until it returns [`io::NdjsonControl::Stop`] or input is exhausted.
    pub fn for_each_ndjson_until<R, F>(
        &self,
        reader: R,
        query: &str,
        f: F,
    ) -> std::result::Result<usize, JetroEngineError>
    where
        R: std::io::BufRead,
        F: FnMut(Value) -> std::result::Result<io::NdjsonControl, JetroEngineError>,
    {
        io::for_each_ndjson_until(self, reader, query, f)
    }

    /// Evaluate `query` for every row from an [`io::NdjsonSource`] and call
    /// `f` with each result as it is produced.
    pub fn for_each_ndjson_source<F>(
        &self,
        source: io::NdjsonSource,
        query: &str,
        f: F,
    ) -> std::result::Result<usize, JetroEngineError>
    where
        F: FnMut(Value),
    {
        io::for_each_ndjson_source(self, source, query, f)
    }

    /// Evaluate `query` for every row from an [`io::NdjsonSource`] and call
    /// `f` until it returns [`io::NdjsonControl::Stop`] or input is exhausted.
    pub fn for_each_ndjson_source_until<F>(
        &self,
        source: io::NdjsonSource,
        query: &str,
        f: F,
    ) -> std::result::Result<usize, JetroEngineError>
    where
        F: FnMut(Value) -> std::result::Result<io::NdjsonControl, JetroEngineError>,
    {
        io::for_each_ndjson_source_until(self, source, query, f)
    }

    /// Like [`JetroEngine::for_each_ndjson_source_until`] with explicit NDJSON reader options.
    pub fn for_each_ndjson_source_until_with_options<F>(
        &self,
        source: io::NdjsonSource,
        query: &str,
        options: io::NdjsonOptions,
        f: F,
    ) -> std::result::Result<usize, JetroEngineError>
    where
        F: FnMut(Value) -> std::result::Result<io::NdjsonControl, JetroEngineError>,
    {
        io::for_each_ndjson_source_until_with_options(self, source, query, options, f)
    }

    /// Like [`JetroEngine::for_each_ndjson_source`] with explicit NDJSON reader options.
    pub fn for_each_ndjson_source_with_options<F>(
        &self,
        source: io::NdjsonSource,
        query: &str,
        options: io::NdjsonOptions,
        f: F,
    ) -> std::result::Result<usize, JetroEngineError>
    where
        F: FnMut(Value),
    {
        io::for_each_ndjson_source_with_options(self, source, query, options, f)
    }

    /// Like [`JetroEngine::for_each_ndjson`] with explicit NDJSON reader options.
    pub fn for_each_ndjson_with_options<R, F>(
        &self,
        reader: R,
        query: &str,
        options: io::NdjsonOptions,
        f: F,
    ) -> std::result::Result<usize, JetroEngineError>
    where
        R: std::io::BufRead,
        F: FnMut(Value),
    {
        io::for_each_ndjson_with_options(self, reader, query, options, f)
    }

    /// Like [`JetroEngine::for_each_ndjson_until`] with explicit NDJSON reader options.
    pub fn for_each_ndjson_until_with_options<R, F>(
        &self,
        reader: R,
        query: &str,
        options: io::NdjsonOptions,
        f: F,
    ) -> std::result::Result<usize, JetroEngineError>
    where
        R: std::io::BufRead,
        F: FnMut(Value) -> std::result::Result<io::NdjsonControl, JetroEngineError>,
    {
        io::for_each_ndjson_until_with_options(self, reader, query, options, f)
    }

    /// Look up a compiled `QueryPlan` by expression string and planning context,
    /// compiling and inserting it if not already cached; evicts the whole cache if full.
    pub(crate) fn cached_plan(
        &self,
        expr: &str,
        context: plan::physical::PlanningContext,
    ) -> ir::physical::QueryPlan {
        let mut cache = self.plan_cache.lock().expect("plan cache poisoned");
        let cache_key = format!("{}\0{}", context.cache_key(), expr);
        if let Some(plan) = cache.get(&cache_key) {
            return plan.clone();
        }

        let plan = plan::physical::plan_query_with_context(expr, context);
        if self.plan_cache_limit > 0 {
            if cache.len() >= self.plan_cache_limit {
                cache.clear();
            }
            cache.insert(cache_key, plan.clone());
        }
        plan
    }
}

impl exec::pipeline::PipelineData for Jetro {
    fn promote_objvec(&self, arr: &Arc<Vec<Val>>) -> Option<Arc<crate::data::value::ObjVecData>> {
        self.get_or_promote_objvec(arr)
    }
}

impl Jetro {
    /// Return a reference to the lazily parsed simd-json `TapeData`, parsing raw bytes
    /// on first access. Returns `Ok(None)` when no raw bytes are stored.
    pub(crate) fn lazy_tape(
        &self,
    ) -> std::result::Result<Option<&Arc<crate::data::tape::TapeData>>, EvalError> {
        if let Some(result) = self.tape.get() {
            return result
                .as_ref()
                .map(Some)
                .map_err(|err| EvalError(format!("Invalid JSON: {err}")));
        }
        let Some(raw) = self.raw_bytes.as_ref() else {
            return Ok(None);
        };
        let bytes: Vec<u8> = (**raw).to_vec();
        let parsed = crate::data::tape::TapeData::parse(bytes).map_err(|err| err.to_string());
        let _ = self.tape.set(parsed);
        self.tape
            .get()
            .expect("tape cache initialized")
            .as_ref()
            .map(Some)
            .map_err(|err| EvalError(format!("Invalid JSON: {err}")))
    }

    /// Look up or build an `ObjVecData` columnar representation for the given
    /// `Arc<Vec<Val>>` array, caching the result by pointer address.
    pub(crate) fn get_or_promote_objvec(
        &self,
        arr: &Arc<Vec<Val>>,
    ) -> Option<Arc<crate::data::value::ObjVecData>> {
        let key = Arc::as_ptr(arr) as usize;
        if let Ok(cache) = self.objvec_cache.lock() {
            if let Some(d) = cache.get(&key) {
                return Some(Arc::clone(d));
            }
        }
        let promoted = exec::pipeline::Pipeline::try_promote_objvec_arr(arr)?;
        if let Ok(mut cache) = self.objvec_cache.lock() {
            cache.entry(key).or_insert_with(|| Arc::clone(&promoted));
        }
        Some(promoted)
    }

    /// Internal constructor that wraps a `serde_json::Value` without raw bytes.
    pub(crate) fn new(document: Value) -> Self {
        Self {
            document,
            root_val: OnceCell::new(),
            objvec_cache: Default::default(),
            raw_bytes: None,
            tape: OnceCell::new(),
            structural_index: OnceCell::new(),
            vm: RefCell::new(VM::new()),
        }
    }

    /// Build a `Jetro` whose `root_val` is pre-cached with `root` (constructed by the
    /// caller, typically via [`Val::from_value_with`] using an engine-owned key cache).
    /// `document` is retained for value-backed callers and tests that read the
    /// original `serde_json::Value`.
    pub(crate) fn from_val_and_value(root: Val, document: Value) -> Self {
        let root_val = OnceCell::new();
        let _ = root_val.set(root);
        Self {
            document,
            root_val,
            objvec_cache: Default::default(),
            raw_bytes: None,
            tape: OnceCell::new(),
            structural_index: OnceCell::new(),
            vm: RefCell::new(VM::new()),
        }
    }

    /// Like [`Jetro::root_val`] but interns object keys through `keys` instead of the
    /// process-wide default. Used by [`JetroEngine::parse_bytes`] to materialise the
    /// `Val` tree once at parse time so subsequent `collect` calls find a populated
    /// `root_val` cache and skip re-interning.
    pub(crate) fn root_val_with(
        &self,
        keys: &crate::data::intern::KeyCache,
    ) -> std::result::Result<Val, EvalError> {
        if let Some(root) = self.root_val.get() {
            return Ok(root.clone());
        }
        let root = {
            if let Some(tape) = self.lazy_tape()? {
                Val::from_tape_data_with(keys, tape)
            } else {
                Val::from_value_with(keys, &self.document)
            }
        };
        let _ = self.root_val.set(root);
        Ok(self.root_val.get().expect("root val initialized").clone())
    }

    /// Parse raw JSON bytes and build a `Jetro` query handle.
    /// The bytes are not parsed eagerly; the tape is built lazily on the first
    /// query that needs it.
    pub fn from_bytes(bytes: Vec<u8>) -> std::result::Result<Self, serde_json::Error> {
        Ok(Self {
            document: Value::Null,
            root_val: OnceCell::new(),
            objvec_cache: Default::default(),
            raw_bytes: Some(Arc::from(bytes.into_boxed_slice())),
            tape: OnceCell::new(),
            structural_index: OnceCell::new(),
            vm: RefCell::new(VM::new()),
        })
    }

    /// Borrow this document's VM cache, falling back to a temporary VM on re-entrant use.
    pub(crate) fn with_vm<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut VM) -> R,
    {
        match self.vm.try_borrow_mut() {
            Ok(mut vm) => f(&mut vm),
            Err(_) => {
                let mut vm = VM::new();
                f(&mut vm)
            }
        }
    }

    /// Return the raw JSON byte slice if this handle was constructed from bytes,
    /// or `None` if it was constructed from a `serde_json::Value`.
    pub(crate) fn raw_bytes(&self) -> Option<&[u8]> {
        self.raw_bytes.as_deref()
    }

    /// Return a reference to the lazily built `StructuralIndex` for key-presence
    /// queries, constructing it from raw bytes on first access if available.
    pub(crate) fn lazy_structural_index(
        &self,
    ) -> std::result::Result<Option<&Arc<jetro_experimental::StructuralIndex>>, EvalError> {
        if let Some(result) = self.structural_index.get() {
            return result
                .as_ref()
                .map(Some)
                .map_err(|err| EvalError(format!("Invalid JSON: {err}")));
        }
        let Some(raw) = self.raw_bytes.as_ref() else {
            return Ok(None);
        };
        let built = jetro_experimental::from_bytes_with(
            raw.as_ref(),
            jetro_experimental::BuildOptions::keys_only(),
        )
        .map(Arc::new)
        .map_err(|err| err.to_string());
        let _ = self.structural_index.set(built);
        self.structural_index
            .get()
            .expect("structural index cache initialized")
            .as_ref()
            .map(Some)
            .map_err(|err| EvalError(format!("Invalid JSON: {err}")))
    }

    /// Return the root `Val` for the document, building and caching it from the
    /// tape or from the `serde_json::Value` on first access.
    pub(crate) fn root_val(&self) -> std::result::Result<Val, EvalError> {
        if let Some(root) = self.root_val.get() {
            return Ok(root.clone());
        }
        let root = {
            if let Some(tape) = self.lazy_tape()? {
                Val::from_tape_data(tape)
            } else {
                Val::from(&self.document)
            }
        };
        let _ = self.root_val.set(root);
        Ok(self.root_val.get().expect("root val initialized").clone())
    }

    /// Return `true` if the `Val` tree has already been materialised; used in
    /// tests to assert that lazy evaluation is working correctly.
    #[cfg(test)]
    pub(crate) fn root_val_is_materialized(&self) -> bool {
        self.root_val.get().is_some()
    }

    #[cfg(test)]
    pub(crate) fn structural_index_is_built(&self) -> bool {
        self.structural_index.get().is_some()
    }

    #[cfg(test)]
    pub(crate) fn tape_is_built(&self) -> bool {
        self.tape.get().is_some()
    }

    #[cfg(test)]
    pub(crate) fn reset_tape_materialized_subtrees(&self) {
        if let Ok(Some(tape)) = self.lazy_tape() {
            tape.reset_materialized_subtrees();
        }
    }

    #[cfg(test)]
    pub(crate) fn tape_materialized_subtrees(&self) -> usize {
        self.lazy_tape()
            .ok()
            .flatten()
            .map(|tape| tape.materialized_subtrees())
            .unwrap_or(0)
    }

    /// Evaluate a Jetro expression against this document and return the result
    /// as a `serde_json::Value`. Uses this document's VM with compile and
    /// path-resolution caches for repeated calls.
    pub fn collect<S: AsRef<str>>(&self, expr: S) -> std::result::Result<Value, EvalError> {
        exec::router::collect_json(self, expr.as_ref())
    }
}

/// Wrap an existing `serde_json::Value` in a `Jetro` handle without raw bytes.
/// Prefer `Jetro::from_bytes` when you have the original JSON source, as it
/// enables the tape and structural-index lazy backends.
impl From<Value> for Jetro {
    /// Convert a `serde_json::Value` into a `Jetro` query handle.
    fn from(v: Value) -> Self {
        Self::new(v)
    }
}
