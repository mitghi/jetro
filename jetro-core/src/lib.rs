//! Jetro core — parser, compiler, and VM for the Jetro JSON query language.
//!
//! # Execution path
//!
//! ```text
//! source text
//!   │  parser::parse()      → Expr AST
//!   │  planner::plan_query()→ QueryPlan (physical IR)
//!   │  physical_eval::run() → dispatches to:
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

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) mod analysis;
pub(crate) mod ast;
pub(crate) mod builtin_helpers;
pub(crate) mod builtin_trait;
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
pub(crate) mod compiler;
pub(crate) mod vm;
pub(crate) mod logical_plan;
pub(crate) mod logical_planner;
pub(crate) mod optimizer;

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


// Thread-local VM, constructed lazily on first `collect()` call.
// Thread-local avoids a Mutex and lets compile/path caches accumulate.
thread_local! {
    static THREAD_VM: OnceCell<RefCell<VM>> = const { OnceCell::new() };
}

/// Borrow the thread-local `VM`, constructing it on first access.
/// All `Jetro::collect` calls on the same thread share one `VM` so that
/// compile and path-resolution caches accumulate across queries.
fn with_vm<F, R>(f: F) -> R
where
    F: FnOnce(&RefCell<VM>) -> R,
{
    THREAD_VM.with(|cell| {
        let inner = cell.get_or_init(|| RefCell::new(VM::new()));
        f(inner)
    })
}


/// Primary entry point. Holds a JSON document and evaluates expressions against
/// it. Lazy fields (`root_val`, `tape`, `structural_index`, `objvec_cache`)
/// are populated on first use so callers only pay for the representations a
/// particular query actually needs.
pub struct Jetro {
    /// The `serde_json::Value` root document; unused when `simd-json` is enabled
    /// (the tape is the authoritative source in that case).
    document: Value,
    /// Cached `Val` tree — built once and reused across `collect()` calls.
    root_val: OnceCell<Val>,
    /// Retained raw bytes for lazy tape and structural-index materialisation.
    raw_bytes: Option<Arc<[u8]>>,

    /// Lazily parsed simd-json tape; `Err` is cached to avoid re-parsing after failure.
    #[cfg(feature = "simd-json")]
    tape: OnceCell<std::result::Result<Arc<crate::strref::TapeData>, String>>,
    /// Unused placeholder so the field name is consistent regardless of features.
    #[cfg(not(feature = "simd-json"))]
    #[allow(dead_code)]
    tape: OnceCell<()>,

    /// Lazily built bitmap structural index for accelerated key-presence queries.
    structural_index:
        OnceCell<std::result::Result<Arc<jetro_experimental::StructuralIndex>, String>>,

    /// Per-document cache from `Arc<Vec<Val>>` pointer addresses to promoted
    /// `ObjVecData` columnar representations; keyed by pointer to avoid re-promotion.
    pub(crate) objvec_cache:
        std::sync::Mutex<std::collections::HashMap<usize, Arc<crate::value::ObjVecData>>>,
}


/// Long-lived multi-document query engine with an explicit plan cache.
/// Use when the same process evaluates many expressions over many documents —
/// parse/lower/compile work is amortised by this object, not hidden in
/// thread-local state.
pub struct JetroEngine {
    /// Maps `"<context_key>\0<expr>"` to compiled `QueryPlan`; evicted wholesale when full.
    plan_cache: Mutex<HashMap<String, physical::QueryPlan>>,
    /// Maximum number of entries before the cache is cleared; 0 disables caching.
    plan_cache_limit: usize,
    /// The shared `VM` used by all `collect*` calls on this engine instance.
    vm: Mutex<VM>,
}

/// Error returned by `JetroEngine::collect_bytes` and similar methods that
/// may fail during JSON parsing or during expression evaluation.
#[derive(Debug)]
pub enum JetroEngineError {
    /// JSON parsing failed before evaluation could begin.
    Json(serde_json::Error),
    /// Expression evaluation failed (the JSON was valid but the query errored).
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
        }
    }

    /// Discard all cached query plans, forcing re-compilation on the next call.
    pub fn clear_cache(&self) {
        self.plan_cache.lock().expect("plan cache poisoned").clear();
    }

    /// Evaluate a Jetro expression against an already-constructed `Jetro` document,
    /// using the engine's shared plan cache and `VM`.
    pub fn collect<S: AsRef<str>>(
        &self,
        document: &Jetro,
        expr: S,
    ) -> std::result::Result<Value, EvalError> {
        let plan = self.cached_plan(expr.as_ref(), executor::planning_context(document));
        let mut vm = self.vm.lock().expect("vm cache poisoned");
        executor::collect_plan_json_with_vm(document, &plan, &mut vm)
    }

    /// Convenience wrapper: wrap a `serde_json::Value` in a `Jetro` and evaluate `expr`.
    pub fn collect_value<S: AsRef<str>>(
        &self,
        document: Value,
        expr: S,
    ) -> std::result::Result<Value, EvalError> {
        let document = Jetro::from(document);
        self.collect(&document, expr)
    }

    /// Parse raw JSON bytes into a `Jetro` document and evaluate `expr`,
    /// returning a `JetroEngineError` on either parse or evaluation failure.
    pub fn collect_bytes<S: AsRef<str>>(
        &self,
        bytes: Vec<u8>,
        expr: S,
    ) -> std::result::Result<Value, JetroEngineError> {
        let document = Jetro::from_bytes(bytes)?;
        Ok(self.collect(&document, expr)?)
    }

    /// Look up a compiled `QueryPlan` by expression string and planning context,
    /// compiling and inserting it if not already cached; evicts the whole cache if full.
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
    /// Return a reference to the lazily parsed simd-json `TapeData`, parsing raw bytes
    /// on first access. Returns `Ok(None)` when no raw bytes are stored.
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

    /// Look up or build an `ObjVecData` columnar representation for the given
    /// `Arc<Vec<Val>>` array, caching the result by pointer address.
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

    /// Internal constructor that wraps a `serde_json::Value` without raw bytes.
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

    /// Parse raw JSON bytes and build a `Jetro` query handle.
    /// When the `simd-json` feature is enabled the bytes are not parsed eagerly;
    /// the tape is built lazily on the first query that needs it.
    pub fn from_bytes(bytes: Vec<u8>) -> std::result::Result<Self, serde_json::Error> {
        
        
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
    /// tape (simd-json) or from the `serde_json::Value` on first access.
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

    /// Evaluate a Jetro expression against this document and return the result
    /// as a `serde_json::Value`. Uses the thread-local `VM` with compile and
    /// path-resolution caches for repeated calls.
    pub fn collect<S: AsRef<str>>(&self, expr: S) -> std::result::Result<Value, EvalError> {
        executor::collect_json(self, expr.as_ref())
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
