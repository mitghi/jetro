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
use std::sync::Arc;
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
pub(crate) type Result<T> = std::result::Result<T, Error>;

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
    tape: OnceCell<Arc<crate::strref::TapeData>>,
    #[cfg(not(feature = "simd-json"))]
    #[allow(dead_code)]
    tape: OnceCell<()>,
    /// Memoised ObjVec promotions, keyed by the source `Arc<Vec<Val>>`'s
    /// pointer identity.  When a Pipeline collects a uniform-shape array
    /// of objects, the first call probes + builds an `ObjVecData`; all
    /// subsequent calls (across queries, across iterations) reuse the
    /// cached columnar layout.  Cost amortised across the lifetime of
    /// the Jetro handle.
    pub(crate) objvec_cache:
        std::sync::Mutex<std::collections::HashMap<usize, Arc<crate::value::ObjVecData>>>,
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
    fn lazy_tape(&self) -> Option<&Arc<crate::strref::TapeData>> {
        // OnceCell::get_or_try_init isn't stable for std OnceCell; do
        // get-then-init manually.
        if let Some(t) = self.tape.get() {
            return Some(t);
        }
        let raw = self.raw_bytes.as_ref()?;
        let bytes: Vec<u8> = (**raw).to_vec();
        let parsed = crate::strref::TapeData::parse(bytes).ok()?;
        let _ = self.tape.set(parsed);
        self.tape.get()
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
        }
    }

    /// Parse JSON bytes and retain them for lazy materialisation.
    pub fn from_bytes(bytes: Vec<u8>) -> std::result::Result<Self, serde_json::Error> {
        // Cold-start path — when the simd-json feature is on, parse to a
        // TapeData first and retain it. The full Val tree builds lazily on
        // first access via root_val().
        // Falls back to the serde_json path on simd-json parse error.
        #[cfg(feature = "simd-json")]
        {
            let raw: Arc<[u8]> = Arc::from(bytes.clone().into_boxed_slice());
            match crate::strref::TapeData::parse(bytes) {
                Ok(tape) => {
                    return Ok(Self {
                        document: Value::Null,
                        // Lazy Val build via root_val() from retained tape.
                        root_val: OnceCell::new(),
                        objvec_cache: Default::default(),
                        raw_bytes: Some(raw),
                        tape: {
                            let c = OnceCell::new();
                            let _ = c.set(tape);
                            c
                        },
                    });
                }
                Err(_) => {
                    let document: Value = serde_json::from_slice(&raw)?;
                    return Ok(Self {
                        document,
                        root_val: OnceCell::new(),
                        objvec_cache: Default::default(),
                        raw_bytes: Some(raw),
                        tape: OnceCell::new(),
                    });
                }
            }
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
            })
        }
    }

    fn root_val(&self) -> Val {
        self.root_val
            .get_or_init(|| {
                #[cfg(feature = "simd-json")]
                {
                    // Prefer the retained simd-json tape when present:
                    // materialise a normal Val tree whose string leaves are
                    // StrSlice views into TapeData.bytes_buf. This keeps the
                    // rest of the VM/Pipeline API lifetime-free while avoiding
                    // per-string Arc<str> allocation on cold VM fallback.
                    if let Some(tape) = self.lazy_tape() {
                        return Val::from_tape_data(tape);
                    }
                    // If tape parsing was unavailable but raw bytes exist,
                    // still use the direct simd-json -> Val parser before
                    // falling back to the serde_json::Value document.
                    if let Some(raw) = &self.raw_bytes {
                        let mut buf: Vec<u8> = (**raw).to_vec();
                        if let Ok(v) = Val::from_json_simd(&mut buf) {
                            return v;
                        }
                    }
                }
                Val::from(&self.document)
            })
            .clone()
    }

    #[cfg(test)]
    pub(crate) fn root_val_is_materialized(&self) -> bool {
        self.root_val.get().is_some()
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
