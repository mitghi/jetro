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
pub mod strref;

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

/// A pre-compiled Jetro query.  Compile once, run against many
/// documents — bypasses the per-call compile cache lookup and lets
/// the same opcode program be shared across threads (it's `Send +
/// Sync`).  See [`Jetro::compile`].
#[derive(Clone)]
pub struct CompiledQuery {
    program: Arc<Program>,
}

impl CompiledQuery {
    /// Run against a `serde_json::Value` document; returns `Value`.
    pub fn run(&self, doc: &Value) -> Result<Value> {
        let mut vm = VM::new();
        Ok(vm.execute(&self.program, doc)?)
    }

    /// Run against a pre-built `JetroVal` root; returns `JetroVal`
    /// (skips the result-side `Val → serde_json::Value` materialisation).
    pub fn run_val(&self, root: JetroVal) -> Result<JetroVal> {
        let mut vm = VM::new();
        Ok(vm.execute_val_raw(&self.program, root)?)
    }

    /// Run against an existing `Jetro` handle, reusing its document /
    /// root_val cache.  Equivalent to `j.collect(expr)` but skips the
    /// thread-local VM compile cache lookup.
    pub fn run_on(&self, j: &Jetro) -> Result<Value> {
        THREAD_VM.with(|cell| {
            let mut vm = cell.try_borrow_mut().map_err(|_| EvalError("VM in use".into()))?;
            Ok(vm.execute_val(&self.program, j.root_val_cached())?)
        })
    }

    /// Borrow the underlying `Arc<Program>` for callers that want to
    /// hand it directly to `VM::execute_val_*`.
    pub fn program(&self) -> &Arc<Program> { &self.program }
}

impl Jetro {
    /// Compile a Jetro expression once for repeated use.  The returned
    /// [`CompiledQuery`] holds an `Arc<Program>` and can be shared
    /// across threads / documents.
    ///
    /// Prefer this over `Jetro::collect` when running the *same* query
    /// against many documents — saves the per-call thread-local
    /// compile-cache lookup and avoids re-parsing.
    ///
    /// ```ignore
    /// let q = jetro::Jetro::compile("$.users.filter(active).count()")?;
    /// for doc in docs {
    ///     let n = q.run(&doc)?;
    /// }
    /// ```
    pub fn compile<S: AsRef<str>>(expr: S) -> Result<CompiledQuery> {
        let prog = vm::Compiler::compile_str(expr.as_ref())?;
        Ok(CompiledQuery { program: Arc::new(prog) })
    }

    /// Internal helper — exposes root_val for `CompiledQuery::run_on`.
    pub(crate) fn root_val_cached(&self) -> Val {
        self.root_val()
    }

    /// Evaluate `expr` and deserialise the result into a caller-chosen
    /// type `T`.  Saves the manual `serde_json::from_value` step after
    /// `.collect()`.
    ///
    /// ```ignore
    /// let titles: Vec<String> = j.collect_typed("$.books.map(title)")?;
    /// let count:  i64         = j.collect_typed("$.books.len()")?;
    /// #[derive(serde::Deserialize)] struct Book { title: String, price: f64 }
    /// let books:  Vec<Book>   = j.collect_typed("$.books")?;
    /// ```
    ///
    /// Internally goes through `collect_val` + `Val::to_json_vec` +
    /// `serde_json::from_slice` — skips the intermediate
    /// `serde_json::Value` tree on the result side.
    pub fn collect_typed<S, T>(&self, expr: S) -> Result<T>
    where
        S: AsRef<str>,
        T: serde::de::DeserializeOwned,
    {
        let val = self.collect_val(expr.as_ref())?;
        let bytes = val.to_json_vec();
        Ok(serde_json::from_slice::<T>(&bytes)
            .map_err(|e| EvalError(format!("collect_typed: {}", e)))?)
    }
}

impl CompiledQuery {
    /// Run + deserialise into `T` in one shot.  Same shape as
    /// [`Jetro::collect_typed`] but uses the pre-compiled program.
    pub fn run_typed<T>(&self, doc: &Value) -> Result<T>
    where T: serde::de::DeserializeOwned,
    {
        let val: Val = (doc).into();
        let mut vm = VM::new();
        let out = vm.execute_val_raw(&self.program, val)?;
        let bytes = out.to_json_vec();
        Ok(serde_json::from_slice::<T>(&bytes)
            .map_err(|e| EvalError(format!("run_typed: {}", e)))?)
    }
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
    /// Phase-6 tape lane — present when the handle was built via
    /// [`Jetro::from_simd_lazy`].  Reserved for future tape-aware
    /// execute paths; today it's just retained alongside the Val.
    #[cfg(feature = "simd-json")]
    tape: Option<Arc<crate::strref::TapeData>>,
    #[cfg(not(feature = "simd-json"))]
    #[allow(dead_code)]
    tape: Option<()>,
}

/// Phase 6 tape-aware fast path classifier + executor.
///
/// Returns `Some(result)` when `expr` matches a supported tape-friendly
/// shape over the given `TapeData`; `None` otherwise (caller falls
/// back to the regular Val-based execute path).
///
/// Currently supported shapes:
///   `$..k.sum()` / `$..k.avg()` / `$..k.min()` / `$..k.max()` /
///   `$..k.count()` / `$..k.len()`.
/// Walks the tape recursively, aggregates numeric values found at
/// every nested key matching `k`, never builds a Val.
#[cfg(feature = "simd-json")]
fn try_tape_descend_aggregate(
    expr: &str,
    tape: &Arc<crate::strref::TapeData>,
) -> Option<std::result::Result<Val, EvalError>> {
    use crate::ast::{Expr, Step, Arg};
    let ast = parser::parse(expr).ok()?;
    let (base, steps) = match &ast {
        Expr::Chain(b, s) => (b.as_ref(), s.as_slice()),
        _ => return None,
    };
    if !matches!(base, Expr::Root) { return None; }
    let _ = (Arg::Pos(Expr::Null),);   // touch Arg so import isn't unused on cold paths

    // Bare `$..k` — collect numerics into IntVec/FloatVec without building Val tree.
    if steps.len() == 1 {
        let key = match &steps[0] {
            Step::Descendant(k) => k.as_str(),
            _ => return None,
        };
        let (ints, floats, is_float) =
            crate::strref::tape_descend_collect_numeric(tape, key)?;
        let out = if is_float {
            Val::FloatVec(Arc::new(floats))
        } else {
            Val::IntVec(Arc::new(ints))
        };
        return Some(Ok(out));
    }

    if steps.len() != 2 { return None; }
    let key = match &steps[0] {
        Step::Descendant(k) => k.as_str(),
        _ => return None,
    };
    let agg = match &steps[1] {
        Step::Method(name, args) if args.is_empty() => name.as_str(),
        _ => return None,
    };

    let (sum_i, sum_f, count, min_f, max_f, is_float) =
        crate::strref::tape_descend_numeric_fold(tape, key);

    let out = match agg {
        "sum"   => if is_float { Val::Float(sum_f) }
                   else        { Val::Int(sum_i) },
        "avg"   => {
            if count == 0 { Val::Null }
            else {
                let total = if is_float { sum_f } else { sum_i as f64 };
                Val::Float(total / count as f64)
            }
        }
        "min"   => if count == 0 { Val::Null } else { Val::Float(min_f) },
        "max"   => if count == 0 { Val::Null } else { Val::Float(max_f) },
        "count" | "len" => Val::Int(count as i64),
        _ => return None,
    };
    Some(Ok(out))
}

/// Trim leading/trailing ASCII whitespace from a `&[u8]`.
#[cfg(feature = "simd-json")]
fn trim_ascii(b: &[u8]) -> &[u8] {
    let mut s = 0;
    let mut e = b.len();
    while s < e && b[s].is_ascii_whitespace() { s += 1; }
    while e > s && b[e - 1].is_ascii_whitespace() { e -= 1; }
    &b[s..e]
}

impl Jetro {
    pub fn new(document: Value) -> Self {
        Self { document, root_val: OnceCell::new(), raw_bytes: None, tape: None }
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
            tape: None,
        })
    }

    /// Parse JSON from a slice, retaining a copy of the bytes.
    pub fn from_slice(bytes: &[u8]) -> std::result::Result<Self, serde_json::Error> {
        Self::from_bytes(bytes.to_vec())
    }

    /// Parse JSON via [simd-json](https://github.com/simd-lite/simd-json) —
    /// SIMD-accelerated structural scanner, typically 2-4x faster than
    /// `serde_json::from_slice` on large inputs.  The parser **mutates**
    /// the input buffer in place, so the buffer is consumed.
    ///
    /// Skips the `serde_json::Value` intermediate tree: simd-json's serde
    /// shim deserializes directly into [`Val`], saving a full extra walk
    /// of the document on cold start.
    ///
    /// On parse error this falls back to `serde_json::from_slice` over the
    /// (possibly partially-mutated) buffer; if both fail, the simd-json
    /// error is returned.  The original bytes (post-mutation) are retained
    /// so `$..key` byte-scan fast paths still work for descendants.
    ///
    /// Requires the `simd-json` cargo feature.
    #[cfg(feature = "simd-json")]
    pub fn from_simd(mut bytes: Vec<u8>) -> std::result::Result<Self, String> {
        // Snapshot the original bytes for byte-scan fast paths *before*
        // simd-json mutates them in place. Cheap one-shot Vec clone; the
        // alternative is rebuilding bytes from the parsed Val later.
        let raw: Arc<[u8]> = Arc::from(bytes.clone().into_boxed_slice());
        match Val::from_json_simd(&mut bytes) {
            Ok(val) => {
                let cell: OnceCell<Val> = OnceCell::new();
                let _ = cell.set(val);
                Ok(Self {
                    document: Value::Null,
                    root_val: cell,
                    raw_bytes: Some(raw),
                    tape: None,
                })
            }
            Err(simd_err) => {
                // simd-json may have mutated the buffer; reparse from the
                // pristine raw_bytes copy via serde_json as fallback.
                let document: Value = serde_json::from_slice(&raw)
                    .map_err(|e| format!("simd-json: {} ; serde_json fallback: {}", simd_err, e))?;
                Ok(Self {
                    document,
                    root_val: OnceCell::new(),
                    raw_bytes: Some(raw),
                    tape: None,
                })
            }
        }
    }

    /// `from_simd` over a borrowed slice. Allocates a writable copy
    /// internally (simd-json mutates in place).
    #[cfg(feature = "simd-json")]
    pub fn from_simd_slice(bytes: &[u8]) -> std::result::Result<Self, String> {
        Self::from_simd(bytes.to_vec())
    }

    /// Tape-aware ingestion path (Phase 6, foundation).  Parses `bytes`
    /// via simd-json and stores a flat `TapeData` (bytes + nodes)
    /// without materialising a `Val` tree.  When subsequent queries
    /// are tape-friendly (descendant + simple-aggregate forms — see
    /// `project_tape_aware_vm.md`), the VM walks the tape directly
    /// without ever building a Val; non-tape-friendly queries
    /// transparently fall back to materialising a Val on first access.
    ///
    /// Currently only the foundation lands: TapeData is built and
    /// retained, but the execute_tape opcode handlers (Day 2 of the
    /// plan) are not yet wired, so this constructor behaves
    /// identically to `from_simd` for execution purposes.  The tape
    /// itself is reachable via `Jetro::tape()` for future use.
    ///
    /// Requires the `simd-json` cargo feature.
    #[cfg(feature = "simd-json")]
    pub fn from_simd_lazy(bytes: Vec<u8>) -> std::result::Result<Self, String> {
        let raw: Arc<[u8]> = Arc::from(bytes.clone().into_boxed_slice());
        let tape = crate::strref::TapeData::parse(bytes)?;
        // Day 3: Val build is now lazy — skipped entirely when every query
        // hits the tape fast path.  `root_val()` materialises on demand
        // by re-parsing `raw_bytes` via simd-json.
        Ok(Self {
            document: Value::Null,
            root_val: OnceCell::new(),
            raw_bytes: Some(raw),
            tape: Some(tape),
        })
    }

    /// Borrow the parsed tape if the handle was built via
    /// [`Jetro::from_simd_lazy`].  `None` for handles built any other
    /// way.  Public so downstream tooling can inspect the flat node
    /// sequence (e.g. tape-aware exec paths in custom workloads).
    #[cfg(feature = "simd-json")]
    pub fn tape(&self) -> Option<&Arc<crate::strref::TapeData>> {
        self.tape.as_ref()
    }

    /// Parse a newline-delimited JSON (NDJSON / JSON-Lines) buffer into a
    /// single `Jetro` handle whose root is a `Val::Arr` of one entry per
    /// non-empty line.  Each line is parsed via simd-json.
    ///
    /// Useful for log streams, event captures, and large append-only
    /// dumps where the producer never writes a wrapping `[…]` array.
    /// After this call, all of jetro's normal queries work over the
    /// concatenated array — `$..error.count()`, `$.filter(level == 'warn')`,
    /// chain-style writes, etc.
    ///
    /// Whitespace-only lines and empty lines are skipped.  On any line
    /// parse error returns `Err` carrying the line number.
    ///
    /// Requires the `simd-json` cargo feature.
    #[cfg(feature = "simd-json")]
    pub fn from_ndjson(bytes: &[u8]) -> std::result::Result<Self, String> {
        let mut out: Vec<Val> = Vec::new();
        for (lineno, raw_line) in bytes.split(|&b| b == b'\n').enumerate() {
            let trimmed = trim_ascii(raw_line);
            if trimmed.is_empty() { continue; }
            let mut buf: Vec<u8> = trimmed.to_vec();
            let v = Val::from_json_simd(&mut buf)
                .map_err(|e| format!("ndjson line {}: {}", lineno + 1, e))?;
            out.push(v);
        }
        let arr = Val::arr(out);
        let cell: OnceCell<Val> = OnceCell::new();
        let _ = cell.set(arr);
        Ok(Self {
            document: Value::Null,
            root_val: cell,
            raw_bytes: None,
            tape: None,
        })
    }

    fn root_val(&self) -> Val {
        self.root_val.get_or_init(|| {
            #[cfg(feature = "simd-json")]
            {
                if self.tape.is_some() {
                    if let Some(raw) = &self.raw_bytes {
                        let mut buf: Vec<u8> = (**raw).to_vec();
                        if let Ok(v) = Val::from_json_simd(&mut buf) {
                            return v;
                        }
                    }
                }
            }
            Val::from(&self.document)
        }).clone()
    }

    /// Evaluate `expr` against the document.  Routes through the thread-local
    /// VM (compile + path caches); when the Jetro handle carries raw bytes
    /// the VM executes on an env with `raw_bytes` set so `Opcode::Descendant`
    /// can take the SIMD byte-scan fast path.
    pub fn collect<S: AsRef<str>>(&self, expr: S) -> std::result::Result<Value, EvalError> {
        let expr = expr.as_ref();
        // Phase 6 tape-aware fast path: `$..k.<aggregate>()` over a
        // handle built via from_simd_lazy walks the tape directly,
        // never building a Val tree.  Bench: closes $..price gap from
        // ~17x native to ~2-3x on bench_complex.
        #[cfg(feature = "simd-json")]
        if let Some(tape) = &self.tape {
            if let Some(out) = try_tape_descend_aggregate(expr, tape) {
                return out.map(|v| v.into());
            }
        }
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
        #[cfg(feature = "simd-json")]
        if let Some(tape) = &self.tape {
            if let Some(out) = try_tape_descend_aggregate(expr, tape) {
                return out;
            }
        }
        THREAD_VM.with(|cell| {
            let mut vm = cell.try_borrow_mut().map_err(|_| EvalError("VM in use".into()))?;
            let prog = vm.get_or_compile(expr)?;
            vm.execute_val_raw(&prog, self.root_val())
        })
    }

    /// Streaming iterator over a query result.
    ///
    /// Returns an [`Iterator<Item = Result<JetroVal, EvalError>>`].
    /// When the query has the shape `$.<path>.filter(...).map(...).take(n).skip(n)`
    /// (any subset, in any order, of `filter` / `map` / `take` / `skip`)
    /// the iterator runs lazily — each `next()` pulls one element from
    /// the path's array, applies the ops, and yields without
    /// materialising the full result Vec.  Useful for memory-bounded
    /// execution on large arrays.
    ///
    /// Anything else (lambdas inside a non-trailing position, sort,
    /// group_by, comprehensions, joins, ...) forces a full eager
    /// evaluation; the returned iterator transparently drains the
    /// materialised `Vec`.
    ///
    /// ```ignore
    /// for item in j.iter("$.orders.filter(price > 100).map(id)")? {
    ///     println!("{}", item?);
    /// }
    /// ```
    pub fn iter<S: AsRef<str>>(&self, expr: S) -> std::result::Result<JetroIter, Error> {
        let ast = parser::parse(expr.as_ref())?;
        Ok(JetroIter::from_expr(self, &ast)?)
    }
}

/// Streaming-aware iterator returned by [`Jetro::iter`].
pub struct JetroIter {
    inner: JetroIterInner,
}

enum JetroIterInner {
    Eager(std::vec::IntoIter<Val>),
    Lazy(Box<LazyState>),
}

struct LazyState {
    items: std::vec::IntoIter<Val>,
    ops:   Vec<LazyOp>,
    env:   eval::Env,
}

enum LazyOp {
    Filter(Arc<ast::Expr>),
    Map(Arc<ast::Expr>),
    Take(usize),
    Skip(usize),
}

impl JetroIter {
    /// Build an iterator from a parsed AST.  Detects the lazy
    /// `Chain(base, [..., Method('filter'|'map'|'take'|'skip', ...) +])`
    /// shape; everything else falls back to eager `collect_val` +
    /// `Vec::into_iter`.
    fn from_expr(j: &Jetro, ast: &ast::Expr) -> std::result::Result<Self, Error> {
        if let Some((base, ops)) = peel_lazy_tail(ast) {
            // Eval the prefix to get the source array.
            let base_val = THREAD_VM.with(|cell| {
                let mut vm = cell.try_borrow_mut()
                    .map_err(|_| EvalError("VM in use".into()))?;
                let prog = Arc::new(vm::Compiler::compile(&base, "<iter>"));
                vm.execute_val_raw(&prog, j.root_val())
            })?;
            let items = match base_val {
                Val::Arr(a)        => Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone()),
                Val::IntVec(a)     => a.iter().map(|n| Val::Int(*n)).collect(),
                Val::FloatVec(a)   => a.iter().map(|f| Val::Float(*f)).collect(),
                Val::StrVec(a)     => a.iter().cloned().map(Val::Str).collect(),
                Val::Null          => Vec::new(),
                other              => vec![other],   // singleton — yields once
            };
            let env = eval::Env::new_with_registry(j.root_val(), Arc::new(eval::MethodRegistry::new()));
            return Ok(JetroIter {
                inner: JetroIterInner::Lazy(Box::new(LazyState {
                    items: items.into_iter(),
                    ops,
                    env,
                })),
            });
        }
        // Eager fallback: full eval, drain Vec.
        let val = THREAD_VM.with(|cell| {
            let mut vm = cell.try_borrow_mut()
                .map_err(|_| EvalError("VM in use".into()))?;
            let prog = Arc::new(vm::Compiler::compile(ast, "<iter>"));
            vm.execute_val_raw(&prog, j.root_val())
        })?;
        let items: Vec<Val> = match val {
            Val::Arr(a)      => Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone()),
            Val::IntVec(a)   => a.iter().map(|n| Val::Int(*n)).collect(),
            Val::FloatVec(a) => a.iter().map(|f| Val::Float(*f)).collect(),
            Val::StrVec(a)   => a.iter().cloned().map(Val::Str).collect(),
            Val::Null        => Vec::new(),
            other            => vec![other],
        };
        Ok(JetroIter { inner: JetroIterInner::Eager(items.into_iter()) })
    }
}

impl Iterator for JetroIter {
    type Item = std::result::Result<Val, EvalError>;
    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            JetroIterInner::Eager(it) => it.next().map(Ok),
            JetroIterInner::Lazy(s)   => s.next_lazy(),
        }
    }
}

impl LazyState {
    fn next_lazy(&mut self) -> Option<std::result::Result<Val, EvalError>> {
        'pull: loop {
            let item = self.items.next()?;
            let mut cur = item;
            for op in &mut self.ops {
                match op {
                    LazyOp::Skip(n) => {
                        if *n > 0 { *n -= 1; continue 'pull; }
                    }
                    LazyOp::Take(n) => {
                        if *n == 0 { return None; }
                        *n -= 1;
                    }
                    LazyOp::Filter(e) => {
                        let prev = self.env.swap_current(cur.clone());
                        let r = eval::eval_in_env(e, &self.env);
                        let _ = self.env.swap_current(prev);
                        match r {
                            Ok(v) if eval::util::is_truthy(&v) => {}
                            Ok(_)  => continue 'pull,
                            Err(err) => return Some(Err(err)),
                        }
                    }
                    LazyOp::Map(e) => {
                        let prev = self.env.swap_current(cur.clone());
                        let r = eval::eval_in_env(e, &self.env);
                        let _ = self.env.swap_current(prev);
                        match r {
                            Ok(v)  => cur = v,
                            Err(err) => return Some(Err(err)),
                        }
                    }
                }
            }
            return Some(Ok(cur));
        }
    }
}

/// Peel the trailing `filter / map / take / skip` chain off an
/// `Expr::Chain(base, steps)` AST.  Returns `Some((base_expr,
/// peeled_ops))` when at least one trailing op is lazy-friendly;
/// `None` otherwise (caller falls back to eager evaluation).
fn peel_lazy_tail(ast: &ast::Expr) -> Option<(ast::Expr, Vec<LazyOp>)> {
    use ast::{Expr, Step, Arg};
    let (base, steps) = match ast {
        Expr::Chain(b, s) => (b.as_ref(), s.as_slice()),
        _ => return None,
    };
    let mut ops: Vec<LazyOp> = Vec::new();
    let mut split: usize = steps.len();
    for (i, st) in steps.iter().enumerate().rev() {
        let lazy_op = match st {
            Step::Method(name, args) => match (name.as_str(), args.as_slice()) {
                ("filter", [Arg::Pos(e)] | [Arg::Named(_, e)]) => {
                    Some(LazyOp::Filter(Arc::new(e.clone())))
                }
                ("map", [Arg::Pos(e)] | [Arg::Named(_, e)]) => {
                    Some(LazyOp::Map(Arc::new(e.clone())))
                }
                ("take", [Arg::Pos(Expr::Int(n))]) if *n >= 0 => Some(LazyOp::Take(*n as usize)),
                ("skip", [Arg::Pos(Expr::Int(n))]) if *n >= 0 => Some(LazyOp::Skip(*n as usize)),
                _ => None,
            },
            _ => None,
        };
        match lazy_op {
            Some(op) => { ops.push(op); split = i; }
            None     => break,
        }
    }
    if split == steps.len() { return None; }
    ops.reverse();
    let base_expr = if split == 0 {
        base.clone()
    } else {
        Expr::Chain(Box::new(base.clone()), steps[..split].to_vec())
    };
    Some((base_expr, ops))
}

impl From<Value> for Jetro {
    fn from(v: Value) -> Self {
        Self::new(v)
    }
}
