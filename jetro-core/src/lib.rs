//! Jetro core — parser, compiler, and VM for the Jetro JSON query language.
//!
//! This crate is storage-free.  For the embedded B+ tree store, named
//! expressions, joins, and [`Session`](../jetrodb/struct.Session.html),
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
//!   planner.rs              ── classify query shape
//!       │                       ├─ Pipeline IR for streamable chains
//!       │                       └─ VM fallback for the general language
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
//!   vm::VM::execute          ── scalar stack machine fallback with
//!                                thread-local compile/path caches.
//! ```
//!
//! # Quick start
//!
//! ```rust
//! use jetro_core::Jetro;
//!
//! let j = Jetro::from_bytes(br#"{"store":{"books":[
//!   {"title":"Dune","price":12.99},
//!   {"title":"Foundation","price":9.99}
//! ]}}"#.to_vec()).unwrap();
//!
//! let count = j.collect("$.store.books.len()").unwrap();
//! assert_eq!(count, serde_json::json!(2));
//! ```

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) mod analysis;
pub(crate) mod ast;
pub(crate) mod builtin_helpers;
pub(crate) mod builtin_registry;
pub(crate) mod builtins;
pub(crate) mod chain_ir;
pub(crate) mod composed;
pub(crate) mod context;
pub(crate) mod executor;
pub(crate) mod parser;
pub(crate) mod physical;
pub(crate) mod physical_eval;
pub(crate) mod pipeline;
pub(crate) mod planner;
pub(crate) mod runtime;
pub(crate) mod strref;
pub(crate) mod structural;
pub(crate) mod util;
pub(crate) mod value;
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) mod value_view;
pub(crate) mod view_pipeline;
pub(crate) mod vm;

#[cfg(test)]
mod examples;
#[cfg(test)]
mod tests;

use serde_json::Value;
use std::cell::{OnceCell, RefCell};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use value::Val;

pub use context::EvalError;
#[cfg(test)]
use parser::ParseError;
use vm::VM;

// ── Error ─────────────────────────────────────────────────────────────────────

/// Query-side error type used by direct VM test helpers.
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

// ── Jetro ─────────────────────────────────────────────────────────────────────

// Thread-local VM with lazy init.  `VM::new()` allocates compile/path
// caches — tiny but not free; deferring until first use saves ~100-200 µs
// of cold-start fixed cost when a thread spins up but never calls
// `collect()`.  `OnceCell<RefCell<VM>>` defers; `with_vm` materialises
// on first access.
thread_local! {
    static THREAD_VM: OnceCell<RefCell<VM>> = const { OnceCell::new() };
}

/// Borrow the thread-local `VM` lazily.  Constructs the VM on first
/// access in the current thread; subsequent calls reuse it.  All
/// existing `with_vm(|cell| ...)` callers route through this
/// adapter so the closure still receives `&RefCell<VM>`.
fn with_vm<F, R>(f: F) -> R
where
    F: FnOnce(&RefCell<VM>) -> R,
{
    THREAD_VM.with(|cell| {
        let inner = cell.get_or_init(|| RefCell::new(VM::new()));
        f(inner)
    })
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
    /// [`Jetro::from_bytes`]. Enables lazy
    /// materialisation without requiring callers to keep their input buffer
    /// alive.
    raw_bytes: Option<Arc<[u8]>>,
    /// Phase-6 tape lane — lazily parsed from `raw_bytes` on first
    /// `lazy_tape()` access.  Defers the simd-json `to_tape` cost
    /// (~5-7 ms / MB) until a query needs the tape.
    #[cfg(feature = "simd-json")]
    tape: OnceCell<std::result::Result<Arc<crate::strref::TapeData>, String>>,
    #[cfg(not(feature = "simd-json"))]
    #[allow(dead_code)]
    tape: OnceCell<()>,
    /// Byte structural sidecar from `jetro-experimental`.  Built only when
    /// the planner selects a structural backend node, so normal tape/Val paths
    /// do not pay the key-bitmap/index cost.
    structural_index:
        OnceCell<std::result::Result<Arc<jetro_experimental::StructuralIndex>, String>>,
    /// Memoised ObjVec promotions, keyed by the source `Arc<Vec<Val>>`'s
    /// pointer identity.  When a Pipeline collects a uniform-shape array
    /// of objects, the first call probes + builds an `ObjVecData`; all
    /// subsequent calls (across queries, across iterations) reuse the
    /// cached columnar layout.  Cost amortised across the lifetime of
    /// the Jetro handle.
    pub(crate) objvec_cache:
        std::sync::Mutex<std::collections::HashMap<usize, Arc<crate::value::ObjVecData>>>,
}

/// Reusable query engine for evaluating many expressions over many documents.
///
/// Unlike [`Jetro::collect`], this owns an explicit physical-plan cache. Use it
/// when the same process evaluates repeated expressions against many `Jetro`
/// documents and you want parse/lower/compile work amortised by this object,
/// not hidden in thread-local state.
pub struct JetroEngine {
    plan_cache: Mutex<HashMap<String, physical::QueryPlan>>,
    plan_cache_limit: usize,
    vm: Mutex<VM>,
}

#[derive(Debug)]
pub enum JetroEngineError {
    Json(serde_json::Error),
    Eval(EvalError),
}

impl std::fmt::Display for JetroEngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Json(err) => write!(f, "{}", err),
            Self::Eval(err) => write!(f, "{}", err),
        }
    }
}

impl std::error::Error for JetroEngineError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Json(err) => Some(err),
            Self::Eval(_) => None,
        }
    }
}

impl From<serde_json::Error> for JetroEngineError {
    fn from(err: serde_json::Error) -> Self {
        Self::Json(err)
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
    const DEFAULT_PLAN_CACHE_LIMIT: usize = 256;

    pub fn new() -> Self {
        Self::with_plan_cache_limit(Self::DEFAULT_PLAN_CACHE_LIMIT)
    }

    pub fn with_plan_cache_limit(plan_cache_limit: usize) -> Self {
        Self {
            plan_cache: Mutex::new(HashMap::new()),
            plan_cache_limit,
            vm: Mutex::new(VM::new()),
        }
    }

    pub fn clear_cache(&self) {
        self.plan_cache.lock().expect("plan cache poisoned").clear();
    }

    pub fn collect<S: AsRef<str>>(
        &self,
        document: &Jetro,
        expr: S,
    ) -> std::result::Result<Value, EvalError> {
        let plan = self.cached_plan(expr.as_ref(), executor::planning_context(document));
        let mut vm = self.vm.lock().expect("vm cache poisoned");
        executor::collect_plan_json_with_vm(document, &plan, &mut vm)
    }

    pub fn collect_value<S: AsRef<str>>(
        &self,
        document: Value,
        expr: S,
    ) -> std::result::Result<Value, EvalError> {
        let document = Jetro::from(document);
        self.collect(&document, expr)
    }

    pub fn collect_bytes<S: AsRef<str>>(
        &self,
        bytes: Vec<u8>,
        expr: S,
    ) -> std::result::Result<Value, JetroEngineError> {
        let document = Jetro::from_bytes(bytes)?;
        Ok(self.collect(&document, expr)?)
    }

    fn cached_plan(&self, expr: &str, context: planner::PlanningContext) -> physical::QueryPlan {
        let mut cache = self.plan_cache.lock().expect("plan cache poisoned");
        let cache_key = format!("{}\0{}", context.cache_key(), expr);
        if let Some(plan) = cache.get(&cache_key) {
            return plan.clone();
        }

        let plan = planner::plan_query_with_context(expr, context);
        if self.plan_cache_limit > 0 {
            if cache.len() >= self.plan_cache_limit {
                cache.clear();
            }
            cache.insert(cache_key, plan.clone());
        }
        plan
    }
}

impl pipeline::PipelineData for Jetro {
    fn promote_objvec(&self, arr: &Arc<Vec<Val>>) -> Option<Arc<crate::value::ObjVecData>> {
        self.get_or_promote_objvec(arr)
    }
}

impl Jetro {
    /// Lazily parse raw_bytes into a TapeData on first access.  Cached
    /// in `tape: OnceCell<Arc<TapeData>>`.  Returns None when no raw
    /// bytes available (handle built from an in-memory serde_json::Value in tests).
    /// Cost: ~5-7 ms / MB on first call (simd-json `to_tape` + node walk
    /// + Arc clone).  Subsequent calls return the cached Arc.
    #[cfg(feature = "simd-json")]
    pub(crate) fn lazy_tape(
        &self,
    ) -> std::result::Result<Option<&Arc<crate::strref::TapeData>>, EvalError> {
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
        let parsed = crate::strref::TapeData::parse(bytes).map_err(|err| err.to_string());
        let _ = self.tape.set(parsed);
        self.tape
            .get()
            .expect("tape cache initialized")
            .as_ref()
            .map(Some)
            .map_err(|err| EvalError(format!("Invalid JSON: {err}")))
    }

    /// Memoised ObjVec promotion.  First call probes the array shape and
    /// builds a columnar ObjVecData; subsequent calls (same Arc<Vec<Val>>
    /// pointer) return the cached layout.  Cache key is the Vec ptr —
    /// safe because `Arc<Vec<Val>>` is immutable in our model.
    ///
    /// Cost: O(N × K) on first miss, O(1) on hit.  Pipeline calls this
    /// once per source array per Jetro lifetime; thereafter every
    /// columnar slot kernel reads slices directly.
    pub(crate) fn get_or_promote_objvec(
        &self,
        arr: &Arc<Vec<Val>>,
    ) -> Option<Arc<crate::value::ObjVecData>> {
        let key = Arc::as_ptr(arr) as usize;
        if let Ok(cache) = self.objvec_cache.lock() {
            if let Some(d) = cache.get(&key) {
                return Some(Arc::clone(d));
            }
        }
        let promoted = pipeline::Pipeline::try_promote_objvec_arr(arr)?;
        if let Ok(mut cache) = self.objvec_cache.lock() {
            cache.entry(key).or_insert_with(|| Arc::clone(&promoted));
        }
        Some(promoted)
    }

    pub(crate) fn new(document: Value) -> Self {
        Self {
            document,
            root_val: OnceCell::new(),
            objvec_cache: Default::default(),
            raw_bytes: None,
            tape: OnceCell::new(),
            structural_index: OnceCell::new(),
        }
    }

    /// Parse JSON bytes and retain them for lazy materialisation.
    pub fn from_bytes(bytes: Vec<u8>) -> std::result::Result<Self, serde_json::Error> {
        // Cold-start path: with simd-json enabled, keep bytes only. The
        // executor plans first, then asks for tape or Val only if the selected
        // representation needs it. This keeps representation choice
        // demand-driven instead of paying a full parse before the expression is
        // known. Parse errors surface from `collect`.
        #[cfg(feature = "simd-json")]
        {
            return Ok(Self {
                document: Value::Null,
                root_val: OnceCell::new(),
                objvec_cache: Default::default(),
                raw_bytes: Some(Arc::from(bytes.into_boxed_slice())),
                tape: OnceCell::new(),
                structural_index: OnceCell::new(),
            });
        }
        #[allow(unreachable_code)]
        {
            let document: Value = serde_json::from_slice(&bytes)?;
            Ok(Self {
                document,
                root_val: OnceCell::new(),
                objvec_cache: Default::default(),
                raw_bytes: Some(Arc::from(bytes.into_boxed_slice())),
                tape: OnceCell::new(),
                structural_index: OnceCell::new(),
            })
        }
    }

    pub(crate) fn raw_bytes(&self) -> Option<&[u8]> {
        self.raw_bytes.as_deref()
    }

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

    pub(crate) fn root_val(&self) -> std::result::Result<Val, EvalError> {
        if let Some(root) = self.root_val.get() {
            return Ok(root.clone());
        }
        let root = {
            #[cfg(feature = "simd-json")]
            {
                if let Some(tape) = self.lazy_tape()? {
                    Val::from_tape_data(tape)
                } else {
                    Val::from(&self.document)
                }
            }
            #[cfg(not(feature = "simd-json"))]
            {
                Val::from(&self.document)
            }
        };
        let _ = self.root_val.set(root);
        Ok(self.root_val.get().expect("root val initialized").clone())
    }

    #[cfg(test)]
    pub(crate) fn root_val_is_materialized(&self) -> bool {
        self.root_val.get().is_some()
    }

    #[cfg(test)]
    pub(crate) fn structural_index_is_built(&self) -> bool {
        self.structural_index.get().is_some()
    }

    #[cfg(all(test, feature = "simd-json"))]
    pub(crate) fn tape_is_built(&self) -> bool {
        self.tape.get().is_some()
    }

    #[cfg(all(test, feature = "simd-json"))]
    pub(crate) fn reset_tape_materialized_subtrees(&self) {
        if let Ok(Some(tape)) = self.lazy_tape() {
            tape.reset_materialized_subtrees();
        }
    }

    #[cfg(all(test, feature = "simd-json"))]
    pub(crate) fn tape_materialized_subtrees(&self) -> usize {
        self.lazy_tape()
            .ok()
            .flatten()
            .map(|tape| tape.materialized_subtrees())
            .unwrap_or(0)
    }

    /// Evaluate `expr` against the document. Routes through the planner and
    /// then either Pipeline IR or the thread-local VM.
    pub fn collect<S: AsRef<str>>(&self, expr: S) -> std::result::Result<Value, EvalError> {
        executor::collect_json(self, expr.as_ref())
    }
}

impl From<Value> for Jetro {
    fn from(v: Value) -> Self {
        Self::new(v)
    }
}
