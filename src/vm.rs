//! Virtual machine for compiling and executing Jetro expressions.
//!
//! # Architecture
//!
//! ```text
//!  String expression
//!        │  parser::parse()
//!        ▼
//!   Vec<Filter>         ← parsed AST
//!        │  Compiler::compile()
//!        ▼
//!     Program           ← flat opcode sequence  (cached: compile_cache)
//!        │  VM::run()
//!        ▼
//!  Vec<Value>           ← results               (cached: resolution_cache)
//! ```
//!
//! # Optimisations
//!
//! **Compile cache** — every expression string is parsed and compiled once.
//! Repeated calls with the same string skip both the PEG parser and the
//! `Filter → Opcode` compilation pass.
//!
//! **Resolution cache** — for *structural* programs (only field navigation,
//! no filters or functions) the concrete JSON-pointer strings reached during
//! the first traversal are stored under `(program_id, document_hash)`.  On a
//! subsequent call that hits the cache, the traversal is skipped entirely;
//! each cached pointer is resolved in O(depth) with `Value::pointer`.
//!
//! # Example
//!
//! ```rust
//! use jetro::vm::VM;
//! use serde_json::json;
//!
//! let doc = json!({"orders": [{"id": 1, "total": 100}, {"id": 2, "total": 200}]});
//! let mut vm = VM::default();
//!
//! // First call: parse + compile + traverse.
//! let r1 = vm.run_str(">/orders/..total/#sum", &doc).unwrap();
//!
//! // Second call with same expression: program served from compile cache.
//! // If the document is also identical: traversal served from resolution cache.
//! let r2 = vm.run_str(">/orders/..total/#sum", &doc).unwrap();
//!
//! assert_eq!(r1.0, r2.0);
//! ```

use std::{
    cell::RefCell,
    collections::{HashMap, VecDeque},
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    rc::Rc,
    sync::Arc,
};

use serde_json::{Map, Value};

use crate::{
    context::{
        Context, Error, Filter, FilterAST, FilterDescendant,
        Func, FuncArg, MapBody, ObjKey, Path, PathResult, PickFilterInner,
    },
    func::{BuiltinFn, Callable, FuncRegistry},
    parser,
};

// ── CompiledArg ───────────────────────────────────────────────────────────────

/// A pre-compiled function argument stored in `Opcode::CallBuiltin`.
///
/// Sub-expression arguments (`>/path`, `@/field`) are compiled into `Program`s
/// once at compile time rather than re-interpreted as `Vec<Filter>` on every
/// function invocation.  This eliminates per-call re-compilation in `#map`,
/// `#flat_map`, `#resolve`, `#deref`, `#join`, and `#lookup`.
#[derive(Debug, Clone)]
pub(crate) enum CompiledArg {
    Key(Arc<str>),
    Number(f64),
    /// Pre-compiled map body: `@/field` or any sub-path used in `#map`.
    MapPath(Arc<Program>),
    /// Pre-compiled `#map({key: path, ...})` object-construct body.
    MapObjConstruct(Arc<[(ObjKey, Arc<Program>)]>),
    /// Pre-compiled `#map([path, ...])` array-construct body.
    MapArrConstruct(Arc<[Arc<Program>]>),
    /// A filter predicate (`#find(cond)`, `#filter_by(cond)`).
    FilterExpr(Arc<FilterAST>),
    /// Pre-compiled sub-expression for `#join`, `#resolve`, `#deref`, `#lookup`.
    SubExpr(Arc<Program>),
    /// Fallback for arg types that have no pre-compiled form yet.
    Raw(Arc<FuncArg>),
}

// ── Opcode ────────────────────────────────────────────────────────────────────

/// A single VM instruction.
///
/// Programs are a flat `Arc<[Opcode]>` slice executed left-to-right by the VM.
/// Each instruction transforms the current set of live values; fan-out
/// operations (e.g. `AnyChild`, `Descend`) expand the set while predicate
/// operations (e.g. `FilterMulti`) shrink it.
#[derive(Debug, Clone)]
pub enum Opcode {
    /// Replace the current value with the document root (`>`).
    PushRoot,
    /// Keep the current value as-is (`@`).
    PushCurrent,

    // ── Navigation ────────────────────────────────────────────────────────────
    /// Descend into object field `key`.
    GetChild(Arc<str>),
    /// Fan out over every value in an object or array (`*`).
    AnyChild,
    /// Array index `[n]`.
    GetIndex(usize),
    /// Array slice `[from:to]`.
    GetSlice(usize, usize),
    /// Array from `[n:]`.
    GetFrom(usize),
    /// Array to `[:n]`.
    GetTo(usize),
    /// Recursive descendant search (`..key` or `..('k'='v')`).
    Descend(FilterDescendant),
    /// First non-null match among the given keys (`('a'|'b')`).
    GroupedChild(Arc<[String]>),

    // ── Filters ───────────────────────────────────────────────────────────────
    /// Apply compound filter condition (the `#filter(...)` function).
    FilterMulti(Arc<FilterAST>),

    // ── Functions ─────────────────────────────────────────────────────────────
    /// Call a named function with pre-parsed arguments.
    /// Fallback path for unknown / user-registered functions.
    CallFn(Arc<Func>),

    /// Static-dispatch call to a known built-in function.
    ///
    /// `fn_id` is resolved once at compile time from the function name string.
    /// Args are pre-compiled `CompiledArg`s — sub-expressions are `Program`s
    /// rather than raw `Vec<Filter>`, so they are never re-compiled at runtime.
    /// No `BTreeMap` lookup, no vtable, no `pack_slice` allocation for arity-1
    /// pure functions.
    CallBuiltin {
        fn_id: BuiltinFn,
        args: Arc<[CompiledArg]>,
        alias: Option<Arc<str>>,
        should_deref: bool,
    },

    // ── Fused opcodes ─────────────────────────────────────────────────────────
    /// Fused `FilterMulti(ast)` + `CallBuiltin(Len)`.
    ///
    /// Emitted by the peephole optimizer when it detects the pattern
    /// `FilterMulti → CallBuiltin(Len|Count)`.  Counts matching elements
    /// without allocating an intermediate `Value::Array`.
    FilterCount(Arc<FilterAST>),

    // ── Pick ──────────────────────────────────────────────────────────────────
    /// Execute a `#pick(...)` expression.
    Pick(Arc<[PickFilterInner]>),

    // ── Construction ─────────────────────────────────────────────────────────
    /// Build an object: `>{ key: sub_program, ... }`.
    ObjBuild(Arc<[(ObjKey, Program)]>),
    /// Build an array: `>[ sub_program, ... ]`.
    ArrBuild(Arc<[Program]>),

    // ── Cache fast-path ───────────────────────────────────────────────────────
    /// Skip traversal and resolve a pre-computed JSON pointer directly.
    /// Only emitted at runtime when a resolution-cache hit is replayed.
    GetPointer(Arc<str>),

    // ── Peephole-fused fast path ──────────────────────────────────────────────
    /// Emitted by the peephole optimizer when it detects `PushRoot + GetChild*`.
    /// Resolves the full pointer against the document root in one pointer call.
    /// E.g.  `>/customers/[0]/name`  →  `RootPointer("/customers/0/name")`.
    RootPointer(Arc<str>),
}

// ── Program ───────────────────────────────────────────────────────────────────

/// A compiled, immutable Jetro program.
///
/// Programs are cheap to clone (inner data is `Arc`-wrapped) and safe to
/// share across calls.
#[derive(Debug, Clone)]
pub struct Program {
    /// The flat opcode sequence.
    pub ops: Arc<[Opcode]>,
    /// Original source expression (used as compile-cache key).
    pub source: Arc<str>,
    /// Stable hash of `source` used as the first component of the resolution-cache key.
    pub id: u64,
    /// True when the program contains only structural navigation opcodes and is
    /// therefore eligible for resolution caching.
    pub is_structural: bool,
}

impl Program {
    fn new(ops: Vec<Opcode>, source: &str) -> Self {
        let id = hash_str(source);
        let is_structural = ops.iter().all(|op| {
            matches!(
                op,
                Opcode::PushRoot
                    | Opcode::PushCurrent
                    | Opcode::GetChild(_)
                    | Opcode::AnyChild
                    | Opcode::GetIndex(_)
                    | Opcode::GetSlice(..)
                    | Opcode::GetFrom(_)
                    | Opcode::GetTo(_)
                    | Opcode::Descend(FilterDescendant::Single(_))
                    | Opcode::GroupedChild(_)
                    | Opcode::RootPointer(_)  // peephole-fused PushRoot+GetChild*
            )
        });
        Self {
            ops: ops.into(),
            source: source.into(),
            id,
            is_structural,
        }
    }
}

// ── Compiler ──────────────────────────────────────────────────────────────────

/// Converts a `Vec<Filter>` (parser output) into a `Program`.
pub struct Compiler;

impl Compiler {
    /// Compile a pre-parsed filter sequence with the given source string for
    /// cache-key tracking.
    pub fn compile(filters: Vec<Filter>, source: &str) -> Program {
        let ops = Self::optimize(Self::lower(filters));
        Program::new(ops, source)
    }

    /// Peephole optimizer.
    ///
    /// Pass 1 — fuses `PushRoot + GetChild(k)*` chains into `RootPointer`.
    /// Pass 2 — fuses `FilterMulti(ast) + CallBuiltin(Len|Count)` into `FilterCount(ast)`,
    ///           eliminating the intermediate `Value::Array` allocation.
    fn optimize(ops: Vec<Opcode>) -> Vec<Opcode> {
        // Pass 1: RootPointer fusion
        let mut pass1: Vec<Opcode> = Vec::with_capacity(ops.len());
        let mut it = ops.into_iter().peekable();
        while let Some(op) = it.next() {
            match op {
                Opcode::PushRoot => {
                    let mut path = String::new();
                    while matches!(it.peek(), Some(Opcode::GetChild(_))) {
                        if let Some(Opcode::GetChild(k)) = it.next() {
                            path.push('/');
                            path.push_str(k.as_ref());
                        }
                    }
                    if path.is_empty() {
                        pass1.push(Opcode::PushRoot);
                    } else {
                        pass1.push(Opcode::RootPointer(Arc::from(path.as_str())));
                    }
                }
                other => pass1.push(other),
            }
        }

        // Pass 2: FilterCount fusion
        let mut result: Vec<Opcode> = Vec::with_capacity(pass1.len());
        let mut it = pass1.into_iter().peekable();
        while let Some(op) = it.next() {
            match op {
                Opcode::FilterMulti(ref ast) => {
                    let is_len = matches!(
                        it.peek(),
                        Some(Opcode::CallBuiltin { fn_id: BuiltinFn::Len, .. })
                            | Some(Opcode::CallBuiltin { fn_id: BuiltinFn::Count, .. })
                    );
                    if is_len {
                        it.next(); // consume the Len/Count opcode
                        result.push(Opcode::FilterCount(Arc::clone(ast)));
                    } else {
                        result.push(op);
                    }
                }
                other => result.push(other),
            }
        }
        result
    }

    /// Parse and compile an expression string in one step.
    pub fn compile_str(expr: &str) -> Result<Program, Error> {
        let filters = parser::parse(expr).map_err(|e| Error::Parse(e.to_string()))?;
        Ok(Self::compile(filters, expr))
    }

    /// Compile a single `FuncArg` into a `CompiledArg`.
    ///
    /// Sub-expressions (`Vec<Filter>`) are compiled into `Program`s here so
    /// they are never re-compiled at evaluation time.
    fn compile_func_arg(arg: FuncArg) -> CompiledArg {
        match arg {
            FuncArg::Key(k)    => CompiledArg::Key(k.into()),
            FuncArg::Number(n) => CompiledArg::Number(n),
            FuncArg::SubExpr(filters) => {
                CompiledArg::SubExpr(Arc::new(Self::compile(filters, "<sub>")))
            }
            FuncArg::FilterExpr(ast) => CompiledArg::FilterExpr(Arc::new(ast)),
            FuncArg::MapStmt(ref map_ast) => match &map_ast.body {
                MapBody::Subpath(filters) => {
                    CompiledArg::MapPath(Arc::new(Self::compile(filters.clone(), "<map-path>")))
                }
                _ => CompiledArg::Raw(Arc::new(arg)),
            },
            FuncArg::ObjConstruct(fields) => {
                let compiled: Vec<(ObjKey, Arc<Program>)> = fields
                    .into_iter()
                    .map(|(k, f)| (k, Arc::new(Self::compile(f, "<map-obj>"))))
                    .collect();
                CompiledArg::MapObjConstruct(compiled.into())
            }
            FuncArg::ArrConstruct(elems) => {
                let compiled: Vec<Arc<Program>> = elems
                    .into_iter()
                    .map(|f| Arc::new(Self::compile(f, "<map-arr>")))
                    .collect();
                CompiledArg::MapArrConstruct(compiled.into())
            }
            other => CompiledArg::Raw(Arc::new(other)),
        }
    }

    fn lower(filters: Vec<Filter>) -> Vec<Opcode> {
        let mut ops = Vec::with_capacity(filters.len());
        for f in filters {
            Self::lower_one(f, &mut ops);
        }
        ops
    }

    fn lower_one(f: Filter, ops: &mut Vec<Opcode>) {
        match f {
            Filter::Root => ops.push(Opcode::PushRoot),
            Filter::CurrentItem => ops.push(Opcode::PushCurrent),
            Filter::AnyChild => ops.push(Opcode::AnyChild),
            Filter::Child(k) => ops.push(Opcode::GetChild(k.into())),
            Filter::ArrayIndex(n) => ops.push(Opcode::GetIndex(n)),
            Filter::Slice(a, b) => ops.push(Opcode::GetSlice(a, b)),
            Filter::ArrayFrom(n) => ops.push(Opcode::GetFrom(n)),
            Filter::ArrayTo(n) => ops.push(Opcode::GetTo(n)),
            Filter::DescendantChild(d) => ops.push(Opcode::Descend(d)),
            Filter::GroupedChild(keys) => ops.push(Opcode::GroupedChild(
                keys.into_iter().map(String::from).collect::<Vec<_>>().into(),
            )),
            Filter::Filter(cond) => {
                // Single-condition filter: wrap in a trivial FilterAST
                let ast = FilterAST::new(cond);
                ops.push(Opcode::FilterMulti(Arc::new(ast)));
            }
            Filter::MultiFilter(ast) => ops.push(Opcode::FilterMulti(Arc::new(ast))),
            Filter::Function(func) => {
                if let Some(fn_id) = BuiltinFn::from_name(&func.name) {
                    let args: Arc<[CompiledArg]> = func
                        .args
                        .into_iter()
                        .map(Self::compile_func_arg)
                        .collect::<Vec<_>>()
                        .into();
                    let alias = func.alias.map(|s| Arc::from(s.as_str()));
                    ops.push(Opcode::CallBuiltin { fn_id, args, alias, should_deref: func.should_deref });
                } else {
                    ops.push(Opcode::CallFn(Arc::new(func)));
                }
            }
            Filter::Pick(elems) => ops.push(Opcode::Pick(elems.into())),
            Filter::ObjectConstruct(fields) => {
                let compiled: Vec<(ObjKey, Program)> = fields
                    .into_iter()
                    .map(|(key, val_filters)| {
                        let prog = Self::compile(val_filters, "<sub>");
                        (key, prog)
                    })
                    .collect();
                ops.push(Opcode::ObjBuild(compiled.into()));
            }
            Filter::ArrayConstruct(elems) => {
                let compiled: Vec<Program> = elems
                    .into_iter()
                    .map(|elem_filters| Self::compile(elem_filters, "<sub>"))
                    .collect();
                ops.push(Opcode::ArrBuild(compiled.into()));
            }
        }
    }
}

// ── Resolution cache ──────────────────────────────────────────────────────────

/// Bounded LRU cache mapping `(program_id, doc_hash) → concrete JSON pointer paths`.
///
/// A cache entry is only populated for structural programs (no filters or
/// functions) where the set of reachable paths depends only on document
/// *structure*, not on field *values*.
struct ResolutionCache {
    map: HashMap<(u64, u64), Arc<[Arc<str>]>>,
    order: VecDeque<(u64, u64)>,
    capacity: usize,
    hits: u64,
    misses: u64,
}

impl ResolutionCache {
    fn new(capacity: usize) -> Self {
        Self {
            map: HashMap::with_capacity(capacity),
            order: VecDeque::with_capacity(capacity),
            capacity,
            hits: 0,
            misses: 0,
        }
    }

    fn get(&mut self, key: (u64, u64)) -> Option<&Arc<[Arc<str>]>> {
        if self.map.contains_key(&key) {
            // Move to back (most-recently-used)
            self.order.retain(|k| k != &key);
            self.order.push_back(key);
            self.hits += 1;
            self.map.get(&key)
        } else {
            self.misses += 1;
            None
        }
    }

    fn insert(&mut self, key: (u64, u64), paths: Vec<Arc<str>>) {
        if self.map.len() >= self.capacity {
            if let Some(oldest) = self.order.pop_front() {
                self.map.remove(&oldest);
            }
        }
        self.order.push_back(key);
        self.map.insert(key, paths.into());
    }

    /// Cache statistics: `(hits, misses)`.
    pub fn stats(&self) -> (u64, u64) {
        (self.hits, self.misses)
    }
}

// ── VM ────────────────────────────────────────────────────────────────────────

/// The Jetro virtual machine.
///
/// Maintains a compile cache (expression → `Program`) and a resolution cache
/// (structural program + document → concrete JSON pointers) to avoid redundant
/// work across repeated evaluations.
///
/// # Thread safety
///
/// `VM` is single-threaded (`Rc`/`RefCell` internals).  Create one `VM` per
/// thread, or wrap in a `Mutex` for shared use.
pub struct VM {
    registry: FuncRegistry,
    /// Expression string → compiled `Program`.
    compile_cache: HashMap<String, Arc<Program>>,
    /// `(program_id, doc_hash)` → resolved pointer strings.
    resolution_cache: ResolutionCache,
}

impl Default for VM {
    fn default() -> Self {
        Self::new()
    }
}

impl VM {
    /// Create a VM with default cache capacities (512 compiled programs,
    /// 1 024 resolution entries).
    pub fn new() -> Self {
        Self::with_capacity(512, 1024)
    }

    /// Create a VM with explicit cache capacities.
    pub fn with_capacity(compile_cap: usize, resolution_cap: usize) -> Self {
        Self {
            registry: FuncRegistry::default(),
            compile_cache: HashMap::with_capacity(compile_cap),
            resolution_cache: ResolutionCache::new(resolution_cap),
        }
    }

    // ── Public entry-points ───────────────────────────────────────────────────

    /// Parse, compile (with caching) and execute `expr` against `document`.
    pub fn run_str(&mut self, expr: &str, document: &Value) -> Result<PathResult, Error> {
        let program = self.get_or_compile(expr)?;
        self.execute(&program, document)
    }

    /// Execute a pre-compiled `Program` against `document`.
    pub fn execute(&mut self, program: &Program, document: &Value) -> Result<PathResult, Error> {
        // ── Resolution-cache fast path ────────────────────────────────────────
        if program.is_structural {
            let doc_hash = hash_doc_structure(document);
            let key = (program.id, doc_hash);
            if let Some(pointers) = self.resolution_cache.get(key) {
                let results: Vec<Value> = pointers
                    .iter()
                    .filter_map(|p| document.pointer(p.as_ref()))
                    .map(Value::clone)
                    .collect();
                return Ok(PathResult(results));
            }
            // Execute normally, then record the paths taken.
            let (values, paths) = self.run_tracked(program, document)?;
            self.resolution_cache.insert(key, paths);
            return Ok(PathResult(values));
        }

        // ── General execution ─────────────────────────────────────────────────
        let values = self.run(program, document, document)?;
        Ok(PathResult(values))
    }

    /// Return a compile-cache `Program`, compiling if not already cached.
    pub fn get_or_compile(&mut self, expr: &str) -> Result<Arc<Program>, Error> {
        if let Some(p) = self.compile_cache.get(expr) {
            return Ok(Arc::clone(p));
        }
        let prog = Compiler::compile_str(expr)?;
        let arc = Arc::new(prog);
        self.compile_cache.insert(expr.to_string(), Arc::clone(&arc));
        Ok(arc)
    }

    /// Register a custom callable function by name.
    ///
    /// Custom functions are invoked via `#name(...)` in expressions exactly like
    /// built-ins, but dispatched through the legacy `FuncRegistry` path (`CallFn`
    /// opcode) rather than the static `CallBuiltin` path.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use jetro::vm::VM;
    /// use jetro::context::{Context, Error, Func};
    /// use serde_json::Value;
    ///
    /// struct DoubleFn;
    /// impl jetro::func::Callable for DoubleFn {
    ///     fn call(&mut self, _func: &Func, value: &Value, _ctx: Option<&mut Context<'_>>) -> Result<Value, Error> {
    ///         match value.as_f64() {
    ///             Some(n) => Ok(Value::Number(serde_json::Number::from_f64(n * 2.0).unwrap())),
    ///             None => Err(Error::FuncEval("double: expected number".into())),
    ///         }
    ///     }
    /// }
    ///
    /// let mut vm = VM::new();
    /// vm.register("double", DoubleFn);
    /// let result = vm.run_str(">/price/#double", &serde_json::json!({"price": 21.0})).unwrap();
    /// assert_eq!(result.0[0], serde_json::json!(42.0));
    /// ```
    pub fn register(&mut self, name: impl Into<String>, func: impl crate::func::Callable + 'static) {
        self.registry.register(name, Box::new(func));
    }

    /// Return cache statistics `((compile_hits, compile_size), (res_hits, res_misses))`.
    pub fn cache_stats(&self) -> ((usize, usize), (u64, u64)) {
        (
            (0, self.compile_cache.len()),
            self.resolution_cache.stats(),
        )
    }

    // ── Internal execution ────────────────────────────────────────────────────

    /// Run `program` against `input`, collecting values.  `root` is the
    /// document root used by `PushRoot`/`RootPointer` and by functions.
    ///
    /// Uses a two-buffer swap to eliminate per-step `Vec` allocations.
    fn run(&mut self, program: &Program, input: &Value, root: &Value) -> Result<Vec<Value>, Error> {
        let mut current: Vec<Value> = Vec::with_capacity(8);
        let mut scratch: Vec<Value> = Vec::with_capacity(8);
        current.push(input.clone());

        for op in program.ops.iter() {
            if current.is_empty() {
                break;
            }
            scratch.clear();
            self.step_into(op, &current, &mut scratch, root)?;
            std::mem::swap(&mut current, &mut scratch);
        }
        Ok(current)
    }

    /// Like `run`, but also records the concrete JSON-pointer path for each
    /// result value (used to populate the resolution cache).
    fn run_tracked(
        &mut self,
        program: &Program,
        document: &Value,
    ) -> Result<(Vec<Value>, Vec<Arc<str>>), Error> {
        // Each frame: (current_value, pointer_so_far)
        let mut current: Vec<(Value, String)> = vec![(document.clone(), String::new())];

        for op in program.ops.iter() {
            if current.is_empty() {
                break;
            }
            current = self.step_tracked(op, current, document)?;
        }

        let values: Vec<Value> = current.iter().map(|(v, _)| v.clone()).collect();
        let paths: Vec<Arc<str>> = current.into_iter().map(|(_, p)| Arc::from(p.as_str())).collect();
        Ok((values, paths))
    }

    // ── Phase-based step (untracked) ──────────────────────────────────────────

    /// Apply a single opcode to `inputs`, appending results into `out`.
    ///
    /// Taking `inputs` as `&[Value]` and writing into an already-allocated `out`
    /// lets the caller swap two reusable buffers instead of allocating a fresh
    /// `Vec` per step.
    #[inline]
    fn step_into(
        &mut self,
        op: &Opcode,
        inputs: &[Value],
        out: &mut Vec<Value>,
        root: &Value,
    ) -> Result<(), Error> {
        match op {
            // ── Roots ─────────────────────────────────────────────────────────
            Opcode::PushRoot => {
                out.push(root.clone());
            }
            Opcode::PushCurrent => {
                out.extend_from_slice(inputs);
            }

            // ── Peephole-fused root pointer ───────────────────────────────────
            Opcode::RootPointer(ptr) => {
                if let Some(v) = root.pointer(ptr.as_ref()) {
                    out.push(v.clone());
                }
            }

            // ── Navigation ───────────────────────────────────────────────────
            Opcode::GetChild(key) => {
                for v in inputs {
                    if let Value::Object(ref obj) = v {
                        if let Some(child) = obj.get(key.as_ref()) {
                            out.push(child.clone());
                        }
                    }
                }
            }

            Opcode::AnyChild => {
                for v in inputs {
                    match v {
                        Value::Object(ref obj) => out.extend(obj.values().cloned()),
                        Value::Array(ref arr) => out.extend(arr.iter().cloned()),
                        _ => {}
                    }
                }
            }

            Opcode::GetIndex(idx) => {
                for v in inputs {
                    if let Value::Array(ref arr) = v {
                        if *idx < arr.len() {
                            out.push(arr[*idx].clone());
                        }
                    }
                }
            }

            Opcode::GetSlice(from, to) => {
                for v in inputs {
                    if let Value::Array(ref arr) = v {
                        if arr.len() >= *to && from < to {
                            out.push(Value::Array(arr[*from..*to].to_vec()));
                        }
                    }
                }
            }

            Opcode::GetFrom(n) => {
                for v in inputs {
                    if let Value::Array(ref arr) = v {
                        if arr.len() >= *n {
                            out.push(Value::Array(arr[*n..].to_vec()));
                        }
                    }
                }
            }

            Opcode::GetTo(n) => {
                for v in inputs {
                    if let Value::Array(ref arr) = v {
                        if arr.len() >= *n {
                            out.push(Value::Array(arr[..*n].to_vec()));
                        }
                    }
                }
            }

            Opcode::GroupedChild(keys) => {
                for v in inputs {
                    if let Value::Object(ref obj) = v {
                        for key in keys.iter() {
                            if let Some(child) = obj.get(key.as_str()) {
                                if !child.is_null() {
                                    out.push(child.clone());
                                    break;
                                }
                            }
                        }
                    }
                }
            }

            Opcode::Descend(desc) => {
                match desc {
                    FilterDescendant::Single(_) => {
                        for v in inputs {
                            descend(v, desc, out);
                        }
                    }
                    FilterDescendant::Pair(key, target_val) => {
                        // The old interpreter accumulates pair-matched objects in
                        // `step_results` and pushes them as a single `Value::Array`
                        // at the end of execution.  Replicate that here.
                        let mut matched: Vec<Value> = Vec::new();
                        for v in inputs {
                            descend_pair(v, key, target_val, &mut matched);
                        }
                        if !matched.is_empty() {
                            out.push(Value::Array(matched));
                        }
                    }
                }
            }

            // ── Filters ───────────────────────────────────────────────────────
            Opcode::FilterMulti(ast) => {
                for v in inputs {
                    if let Value::Array(ref arr) = v {
                        let matched: Vec<Value> =
                            arr.iter().filter(|item| ast.eval(item)).cloned().collect();
                        if !matched.is_empty() {
                            out.push(Value::Array(matched));
                        }
                    }
                }
            }

            // ── Functions ─────────────────────────────────────────────────────
            Opcode::CallFn(func) => {
                let value = pack_slice(inputs);
                let result = self.call_fn(func, &value, root)?;
                out.push(result);
            }

            Opcode::CallBuiltin { fn_id, args, alias, should_deref } => {
                let result = self.exec_builtin(*fn_id, inputs, args, alias.as_deref(), *should_deref, root)?;
                out.push(result);
            }

            Opcode::FilterCount(ast) => {
                for v in inputs {
                    if let Value::Array(ref arr) = v {
                        let n = arr.iter().filter(|item| ast.eval(item)).count();
                        out.push(Value::Number(serde_json::Number::from(n)));
                    }
                }
            }

            // ── Pick ──────────────────────────────────────────────────────────
            Opcode::Pick(elems) => {
                // Clone the Arc so we release the borrow on `op` before calling
                // `self.exec_pick` (which borrows `self` mutably).
                let elems = Arc::clone(elems);
                for v in inputs {
                    if let Some(picked) = self.exec_pick(&elems, v, root)? {
                        out.push(picked);
                    }
                }
            }

            // ── Construction ──────────────────────────────────────────────────
            Opcode::ObjBuild(fields) => {
                let fields = Arc::clone(fields);
                for v in inputs {
                    let mut map = Map::new();
                    for (obj_key, val_prog) in fields.iter() {
                        let key = match obj_key {
                            ObjKey::Static(s) => s.clone(),
                            ObjKey::Dynamic(kf) => {
                                let kp = Compiler::compile(kf.clone(), "<dyn-key>");
                                let kv = self.run(&kp, v, root)?;
                                match kv.into_iter().next() {
                                    Some(Value::String(s)) => s,
                                    Some(other) => {
                                        let raw = other.to_string();
                                        raw.trim_matches('"').to_string()
                                    }
                                    None => continue,
                                }
                            }
                        };
                        let field_vals = self.run(val_prog, v, root)?;
                        if let Some(fv) = field_vals.into_iter().next() {
                            map.insert(key, fv);
                        }
                    }
                    out.push(Value::Object(map));
                }
            }

            Opcode::ArrBuild(elem_progs) => {
                let elem_progs = Arc::clone(elem_progs);
                for v in inputs {
                    let mut arr: Vec<Value> = Vec::with_capacity(elem_progs.len());
                    for prog in elem_progs.iter() {
                        let vals = self.run(prog, v, root)?;
                        if let Some(first) = vals.into_iter().next() {
                            arr.push(first);
                        }
                    }
                    out.push(Value::Array(arr));
                }
            }

            // ── Cache replay ──────────────────────────────────────────────────
            Opcode::GetPointer(ptr) => {
                for v in inputs {
                    if let Some(child) = v.pointer(ptr.as_ref()) {
                        out.push(child.clone());
                    }
                }
            }
        }
        Ok(())
    }

    // ── Phase-based step (tracked) ────────────────────────────────────────────

    fn step_tracked(
        &mut self,
        op: &Opcode,
        inputs: Vec<(Value, String)>,
        root: &Value,
    ) -> Result<Vec<(Value, String)>, Error> {
        let mut out: Vec<(Value, String)> = Vec::with_capacity(inputs.len());

        match op {
            Opcode::PushRoot => return Ok(vec![(root.clone(), String::new())]),
            Opcode::PushCurrent => return Ok(inputs),

            // Peephole-fused opcode: the pointer IS the full path from root.
            Opcode::RootPointer(ptr) => {
                if let Some(v) = root.pointer(ptr.as_ref()) {
                    out.push((v.clone(), ptr.to_string()));
                }
                return Ok(out);
            }

            Opcode::GetChild(key) => {
                for (v, path) in inputs {
                    if let Value::Object(ref obj) = v {
                        if let Some(child) = obj.get(key.as_ref()) {
                            out.push((child.clone(), format!("{}/{}", path, key)));
                        }
                    }
                }
            }

            Opcode::AnyChild => {
                for (v, path) in inputs {
                    match &v {
                        Value::Object(ref obj) => {
                            for (k, child) in obj {
                                out.push((child.clone(), format!("{}/{}", path, k)));
                            }
                        }
                        Value::Array(ref arr) => {
                            for (i, child) in arr.iter().enumerate() {
                                out.push((child.clone(), format!("{}/{}", path, i)));
                            }
                        }
                        _ => {}
                    }
                }
            }

            Opcode::GetIndex(idx) => {
                for (v, path) in inputs {
                    if let Value::Array(ref arr) = v {
                        if *idx < arr.len() {
                            out.push((arr[*idx].clone(), format!("{}/{}", path, idx)));
                        }
                    }
                }
            }

            Opcode::GetSlice(from, to) => {
                for (v, path) in inputs {
                    if let Value::Array(ref arr) = v {
                        if arr.len() >= *to && from < to {
                            // Slices don't have a single pointer — fall back to value only
                            out.push((Value::Array(arr[*from..*to].to_vec()), path));
                        }
                    }
                }
            }

            Opcode::GetFrom(n) => {
                for (v, path) in inputs {
                    if let Value::Array(ref arr) = v {
                        if arr.len() >= *n {
                            out.push((Value::Array(arr[*n..].to_vec()), path));
                        }
                    }
                }
            }

            Opcode::GetTo(n) => {
                for (v, path) in inputs {
                    if let Value::Array(ref arr) = v {
                        if arr.len() >= *n {
                            out.push((Value::Array(arr[..*n].to_vec()), path));
                        }
                    }
                }
            }

            Opcode::GroupedChild(keys) => {
                for (v, path) in inputs {
                    if let Value::Object(ref obj) = v {
                        for key in keys.iter() {
                            if let Some(child) = obj.get(key.as_str()) {
                                if !child.is_null() {
                                    out.push((child.clone(), format!("{}/{}", path, key)));
                                    break;
                                }
                            }
                        }
                    }
                }
            }

            Opcode::Descend(desc) => {
                match desc {
                    FilterDescendant::Single(_) => {
                        for (v, _path) in inputs {
                            let mut vals: Vec<Value> = Vec::new();
                            descend(&v, desc, &mut vals);
                            for val in vals {
                                out.push((val, String::from("<descendant>")));
                            }
                        }
                    }
                    FilterDescendant::Pair(key, target_val) => {
                        let mut matched: Vec<Value> = Vec::new();
                        for (v, _) in inputs {
                            descend_pair(&v, key, target_val, &mut matched);
                        }
                        if !matched.is_empty() {
                            out.push((Value::Array(matched), String::from("<descendant-pair>")));
                        }
                    }
                }
            }

            // Non-structural ops — fall back to untracked step_into
            other => {
                let plain_inputs: Vec<Value> = inputs.into_iter().map(|(v, _)| v).collect();
                let mut tmp: Vec<Value> = Vec::new();
                self.step_into(other, &plain_inputs, &mut tmp, root)?;
                for v in tmp {
                    out.push((v, String::from("<computed>")));
                }
            }
        }
        Ok(out)
    }

    // ── Function dispatch ─────────────────────────────────────────────────────

    fn call_fn(&mut self, func: &Func, value: &Value, root: &Value) -> Result<Value, Error> {
        // `Context::new(root, &[])` creates a context whose `root` is the
        // document root and whose internal results/step_results are empty.
        // `&[]` is a `&'static [Filter]` so the context lifetime is 'static.
        //
        // Functions that call `ctx.reduce_stack_to_*()` read from
        // `ctx.results` (starts empty → identity values) — correct because the
        // VM already packed all pipeline values into `value` before this call.
        // Functions that need `ctx.root` for sub-expression evaluation (join,
        // resolve, deref, …) receive the document root.
        let mut ctx = Context::new(root.clone(), &[]);
        self.registry
            .call(func, value, Some(&mut ctx))
            .map_err(|e| Error::FuncEval(e.to_string()))
    }

    // ── Static builtin dispatch ───────────────────────────────────────────────

    /// Execute a built-in function via direct static dispatch.
    ///
    /// No `BTreeMap` lookup, no vtable, no `RefCell` borrow.  Pure functions
    /// (everything except `join`, `lookup`, `resolve`, `deref`) skip
    /// `Context::new` and `root.clone()` entirely.  Sub-expression arguments
    /// are pre-compiled `Program`s evaluated inline by the VM.
    #[allow(clippy::too_many_arguments)]
    fn exec_builtin(
        &mut self,
        fn_id: BuiltinFn,
        inputs: &[Value],
        args: &[CompiledArg],
        alias: Option<&str>,
        _should_deref: bool,
        root: &Value,
    ) -> Result<Value, Error> {
        // Helper: resolve a pre-compiled SubExpr arg against root, returning a Vec<Value>.
        // Used by join/lookup/resolve/deref.
        macro_rules! run_sub {
            ($idx:expr) => {{
                match args.get($idx) {
                    Some(CompiledArg::SubExpr(prog)) => self.run(prog, root, root)?,
                    _ => return Err(Error::FuncEval(format!("expected sub-expression at arg {}", $idx))),
                }
            }};
        }
        macro_rules! run_sub_array {
            ($idx:expr) => {{
                let vals = run_sub!($idx);
                match vals.into_iter().next() {
                    Some(Value::Array(arr)) => arr,
                    Some(other) => vec![other],
                    None => vec![],
                }
            }};
        }
        macro_rules! key_arg {
            ($idx:expr, $fn_name:expr) => {
                match args.get($idx) {
                    Some(CompiledArg::Key(k)) => k.as_ref(),
                    _ => return Err(Error::FuncEval(concat!($fn_name, ": expected string key argument").into())),
                }
            };
        }
        macro_rules! num_arg {
            ($idx:expr, $fn_name:expr) => {
                match args.get($idx) {
                    Some(CompiledArg::Number(n)) => *n,
                    Some(CompiledArg::Key(s)) => s.parse::<f64>().map_err(|_| {
                        Error::FuncEval(format!("{}: expected number", $fn_name))
                    })?,
                    _ => return Err(Error::FuncEval(format!("{}: expected numeric argument", $fn_name))),
                }
            };
        }

        // Pack inputs → single Value (same semantics as pack_slice, but inline).
        let value: Value;
        let v: &Value = if inputs.len() == 1 {
            &inputs[0]
        } else {
            value = Value::Array(inputs.to_vec());
            &value
        };

        match fn_id {
            // ── Pure aggregate: sum ───────────────────────────────────────────
            BuiltinFn::Sum => {
                let mut total = 0.0f64;
                match v {
                    Value::Array(arr) => {
                        for item in arr { if let Some(n) = item.as_f64() { total += n; } }
                    }
                    _ => { if let Some(n) = v.as_f64() { total += n; } }
                }
                Ok(Value::Number(serde_json::Number::from_f64(total)
                    .unwrap_or_else(|| serde_json::Number::from(0i64))))
            }

            // ── Pure aggregate: len / count ───────────────────────────────────
            BuiltinFn::Len | BuiltinFn::Count => {
                match v {
                    Value::Array(arr) => Ok(Value::Number(serde_json::Number::from(arr.len()))),
                    Value::Object(obj) => Ok(Value::Number(serde_json::Number::from(obj.len()))),
                    _ => Ok(Value::Number(serde_json::Number::from(inputs.len()))),
                }
            }

            // ── Pure: reverse ─────────────────────────────────────────────────
            BuiltinFn::Reverse => match v {
                Value::Array(arr) => Ok(Value::Array(arr.iter().rev().cloned().collect())),
                _ => Err(Error::FuncEval("reverse: expected array".into())),
            },

            // ── Pure: head ────────────────────────────────────────────────────
            BuiltinFn::Head => match v {
                Value::Array(arr) => {
                    Ok(if arr.is_empty() { v.clone() } else { arr[0].clone() })
                }
                _ => Err(Error::FuncEval("head: expected array".into())),
            },

            // ── Pure: tail ────────────────────────────────────────────────────
            BuiltinFn::Tail => match v {
                Value::Array(arr) => Ok(Value::Array(match arr.len() {
                    0 | 1 => vec![],
                    _ => arr[1..].to_vec(),
                })),
                _ => Err(Error::FuncEval("tail: expected array".into())),
            },

            // ── Pure: last ────────────────────────────────────────────────────
            BuiltinFn::Last => match v {
                Value::Array(arr) => Ok(arr.last().cloned().unwrap_or_else(|| Value::Array(vec![]))),
                _ => Err(Error::FuncEval("last: expected array".into())),
            },

            // ── Pure aggregate: all ───────────────────────────────────────────
            BuiltinFn::All => match v {
                Value::Bool(b) => Ok(Value::Bool(*b)),
                Value::Array(arr) => {
                    let all = arr.iter().all(|x| x.as_bool().unwrap_or(false));
                    Ok(Value::Bool(all))
                }
                _ => Err(Error::FuncEval("all: input is not reducible".into())),
            },

            // ── Pure: any ─────────────────────────────────────────────────────
            BuiltinFn::Any => match v {
                Value::Array(arr) => Ok(Value::Bool(arr.iter().any(|x| x.as_bool().unwrap_or(false)))),
                Value::Bool(b) => Ok(Value::Bool(*b)),
                _ => Err(Error::FuncEval("any: expected array of booleans".into())),
            },

            // ── Pure: not ─────────────────────────────────────────────────────
            BuiltinFn::Not => match v {
                Value::Bool(b) => Ok(Value::Bool(!b)),
                _ => Err(Error::FuncEval("not: expected boolean".into())),
            },

            // ── Pure: keys / values ───────────────────────────────────────────
            BuiltinFn::Keys => match v {
                Value::Object(obj) => Ok(Value::Array(
                    obj.keys().map(|k| Value::String(k.clone())).collect()
                )),
                _ => Err(Error::FuncEval("keys: expected object".into())),
            },
            BuiltinFn::Values => match v {
                Value::Object(obj) => Ok(Value::Array(obj.values().cloned().collect())),
                _ => Err(Error::FuncEval("values: expected object".into())),
            },

            // ── Pure: min / max ───────────────────────────────────────────────
            BuiltinFn::Min => match v {
                Value::Array(arr) => {
                    let mut nums: Vec<f64> = arr.iter().filter_map(|x| x.as_f64()).collect();
                    nums.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    match nums.first() {
                        Some(&n) => Ok(Value::Number(serde_json::Number::from_f64(n).unwrap())),
                        _ => Ok(Value::Array(vec![])),
                    }
                }
                _ => Err(Error::FuncEval("min: expected array".into())),
            },
            BuiltinFn::Max => match v {
                Value::Array(arr) => {
                    let mut nums: Vec<f64> = arr.iter().filter_map(|x| x.as_f64()).collect();
                    nums.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                    match nums.first() {
                        Some(&n) => Ok(Value::Number(serde_json::Number::from_f64(n).unwrap())),
                        _ => Ok(Value::Array(vec![])),
                    }
                }
                _ => Err(Error::FuncEval("max: expected array".into())),
            },

            // ── Pure: avg ─────────────────────────────────────────────────────
            BuiltinFn::Avg => match v {
                Value::Array(arr) => {
                    let nums: Vec<f64> = arr.iter().filter_map(|x| x.as_f64()).collect();
                    if nums.is_empty() { return Ok(Value::Null); }
                    let avg = nums.iter().sum::<f64>() / nums.len() as f64;
                    Ok(Value::Number(serde_json::Number::from_f64(avg).unwrap()))
                }
                _ if v.is_number() => Ok(v.clone()),
                _ => Err(Error::FuncEval("avg: expected array of numbers".into())),
            },

            // ── Pure: math ────────────────────────────────────────────────────
            BuiltinFn::Add => {
                let n = num_arg!(0, "add");
                match v.as_f64() {
                    Some(x) => Ok(Value::Number(serde_json::Number::from_f64(x + n).unwrap())),
                    _ => Err(Error::FuncEval("add: expected number".into())),
                }
            }
            BuiltinFn::Sub => {
                let n = num_arg!(0, "sub");
                match v.as_f64() {
                    Some(x) => Ok(Value::Number(serde_json::Number::from_f64(x - n).unwrap())),
                    _ => Err(Error::FuncEval("sub: expected number".into())),
                }
            }
            BuiltinFn::Mul => {
                let n = num_arg!(0, "mul");
                match v.as_f64() {
                    Some(x) => Ok(Value::Number(serde_json::Number::from_f64(x * n).unwrap())),
                    _ => Err(Error::FuncEval("mul: expected number".into())),
                }
            }
            BuiltinFn::Div => {
                let n = num_arg!(0, "div");
                if n == 0.0 { return Err(Error::FuncEval("div: division by zero".into())); }
                match v.as_f64() {
                    Some(x) => Ok(Value::Number(serde_json::Number::from_f64(x / n).unwrap())),
                    _ => Err(Error::FuncEval("div: expected number".into())),
                }
            }
            BuiltinFn::Abs => match v.as_f64() {
                Some(x) => Ok(Value::Number(serde_json::Number::from_f64(x.abs()).unwrap())),
                _ => Err(Error::FuncEval("abs: expected number".into())),
            },
            BuiltinFn::Round => match v.as_f64() {
                Some(x) => Ok(Value::Number(serde_json::Number::from_f64(x.round()).unwrap())),
                _ => Err(Error::FuncEval("round: expected number".into())),
            },
            BuiltinFn::Floor => match v.as_f64() {
                Some(x) => Ok(Value::Number(serde_json::Number::from_f64(x.floor()).unwrap())),
                _ => Err(Error::FuncEval("floor: expected number".into())),
            },
            BuiltinFn::Ceil => match v.as_f64() {
                Some(x) => Ok(Value::Number(serde_json::Number::from_f64(x.ceil()).unwrap())),
                _ => Err(Error::FuncEval("ceil: expected number".into())),
            },

            // ── Pure: nth ─────────────────────────────────────────────────────
            BuiltinFn::Nth => {
                let idx = num_arg!(0, "nth") as usize;
                match v {
                    Value::Array(arr) => Ok(arr.get(idx).cloned().unwrap_or(Value::Null)),
                    _ => Err(Error::FuncEval("nth: expected array".into())),
                }
            }

            // ── Pure: flatten ─────────────────────────────────────────────────
            BuiltinFn::Flatten => match v {
                Value::Array(arr) => {
                    let mut out: Vec<Value> = Vec::new();
                    for item in arr {
                        match item {
                            Value::Array(inner) => out.extend(inner.clone()),
                            other => out.push(other.clone()),
                        }
                    }
                    Ok(Value::Array(out))
                }
                _ => Err(Error::FuncEval("flatten: expected array".into())),
            },

            // ── Pure: chunk ───────────────────────────────────────────────────
            BuiltinFn::Chunk => {
                let size = num_arg!(0, "chunk") as usize;
                if size == 0 { return Err(Error::FuncEval("chunk: size must be > 0".into())); }
                match v {
                    Value::Array(arr) => Ok(Value::Array(
                        arr.chunks(size).map(|c| Value::Array(c.to_vec())).collect()
                    )),
                    _ => Err(Error::FuncEval("chunk: expected array".into())),
                }
            }

            // ── Pure: unique ──────────────────────────────────────────────────
            BuiltinFn::Unique => match v {
                Value::Array(arr) => {
                    let mut seen: Vec<String> = Vec::new();
                    let mut out: Vec<Value> = Vec::new();
                    for x in arr {
                        let k = x.to_string();
                        if !seen.contains(&k) { seen.push(k); out.push(x.clone()); }
                    }
                    Ok(Value::Array(out))
                }
                _ => Err(Error::FuncEval("unique: expected array".into())),
            },

            // ── Pure: distinct ────────────────────────────────────────────────
            BuiltinFn::Distinct => {
                let key = key_arg!(0, "distinct").to_string();
                match v {
                    Value::Array(arr) => {
                        let mut seen: Vec<String> = Vec::new();
                        let mut out: Vec<Value> = Vec::new();
                        for x in arr {
                            if let Some(field) = x.get(&key) {
                                let k = field.to_string();
                                if !seen.contains(&k) { seen.push(k); out.push(x.clone()); }
                            }
                        }
                        Ok(Value::Array(out))
                    }
                    _ => Err(Error::FuncEval("distinct: expected array".into())),
                }
            }

            // ── Pure: sort_by ─────────────────────────────────────────────────
            BuiltinFn::SortBy => {
                let key = key_arg!(0, "sort_by").to_string();
                let descending = matches!(args.get(1), Some(CompiledArg::Key(d)) if d.as_ref() == "desc");
                match v {
                    Value::Array(arr) => {
                        let mut a = arr.clone();
                        a.sort_by(|x, y| {
                            let xn = x.get(&key).and_then(Value::as_f64);
                            let yn = y.get(&key).and_then(Value::as_f64);
                            let ord = match (xn, yn) {
                                (Some(a), Some(b)) => a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal),
                                _ => {
                                    // compare &str directly — no String allocation per comparison
                                    let xs = x.get(&key).and_then(Value::as_str);
                                    let ys = y.get(&key).and_then(Value::as_str);
                                    xs.cmp(&ys)
                                }
                            };
                            if descending { ord.reverse() } else { ord }
                        });
                        Ok(Value::Array(a))
                    }
                    _ => Err(Error::FuncEval("sort_by: expected array".into())),
                }
            }

            // ── Pure: join_str ────────────────────────────────────────────────
            BuiltinFn::JoinStr => {
                let sep = match args.first() {
                    Some(CompiledArg::Key(s)) => s.as_ref().to_string(),
                    _ => String::new(),
                };
                match v {
                    Value::Array(arr) => {
                        let parts: Vec<String> = arr.iter()
                            .filter_map(|x| x.as_str().map(String::from))
                            .collect();
                        Ok(Value::String(parts.join(&sep)))
                    }
                    _ => Err(Error::FuncEval("join_str: expected array".into())),
                }
            }

            // ── Pure: compact ─────────────────────────────────────────────────
            BuiltinFn::Compact => match v {
                Value::Array(arr) => Ok(Value::Array(arr.iter().filter(|x| !x.is_null()).cloned().collect())),
                _ => Err(Error::FuncEval("compact: expected array".into())),
            },

            // ── Pure: zip ─────────────────────────────────────────────────────
            BuiltinFn::Zip => match v {
                Value::Object(obj) => {
                    let mut stack: std::collections::VecDeque<Value> = std::collections::VecDeque::new();
                    for (i, (k, vs)) in obj.iter().enumerate() {
                        if let Value::Array(arr) = vs {
                            for (j, item) in arr.iter().enumerate() {
                                if i == 0 {
                                    let mut m = serde_json::Map::new();
                                    m.insert(k.clone(), item.clone());
                                    stack.push_back(Value::Object(m));
                                } else {
                                    let entry = stack.remove(j);
                                    if let Some(Value::Object(mut obj2)) = entry {
                                        obj2.insert(k.clone(), item.clone());
                                        stack.insert(j, Value::Object(obj2));
                                    }
                                }
                            }
                        }
                    }
                    Ok(Value::Array(Vec::from(stack)))
                }
                _ => Err(Error::FuncEval("zip: expected object".into())),
            },

            // ── Pure: group_by ────────────────────────────────────────────────
            BuiltinFn::GroupBy => {
                let key = key_arg!(0, "group_by").to_string();
                match v {
                    Value::Array(arr) => {
                        let mut map: serde_json::Map<String, Value> = serde_json::Map::new();
                        for item in arr {
                            if let Some(field) = item.get(&key) {
                                let gk = match field {
                                    Value::String(s) => s.clone(),
                                    other => other.to_string().trim_matches('"').to_string(),
                                };
                                let entry = map.entry(gk).or_insert_with(|| Value::Array(vec![]));
                                if let Value::Array(ref mut a) = entry { a.push(item.clone()); }
                            }
                        }
                        Ok(Value::Object(map))
                    }
                    _ => Err(Error::FuncEval("group_by: expected array".into())),
                }
            }

            // ── Pure: count_by ────────────────────────────────────────────────
            BuiltinFn::CountBy => {
                let key = key_arg!(0, "count_by").to_string();
                match v {
                    Value::Array(arr) => {
                        let mut map: serde_json::Map<String, Value> = serde_json::Map::new();
                        for item in arr {
                            if let Some(field) = item.get(&key) {
                                let gk = match field {
                                    Value::String(s) => s.clone(),
                                    other => other.to_string().trim_matches('"').to_string(),
                                };
                                let entry = map.entry(gk).or_insert(Value::Number(serde_json::Number::from(0i64)));
                                let prev = entry.as_i64().unwrap_or(0);
                                *entry = Value::Number(serde_json::Number::from(prev + 1));
                            }
                        }
                        Ok(Value::Object(map))
                    }
                    _ => Err(Error::FuncEval("count_by: expected array".into())),
                }
            }

            // ── Pure: index_by ────────────────────────────────────────────────
            BuiltinFn::IndexBy => {
                let key = key_arg!(0, "index_by").to_string();
                match v {
                    Value::Array(arr) => {
                        let mut map: serde_json::Map<String, Value> = serde_json::Map::new();
                        for item in arr {
                            if let Some(field) = item.get(&key) {
                                let k = match field {
                                    Value::String(s) => s.clone(),
                                    other => other.to_string().trim_matches('"').to_string(),
                                };
                                map.insert(k, item.clone());
                            }
                        }
                        Ok(Value::Object(map))
                    }
                    _ => Err(Error::FuncEval("index_by: expected array".into())),
                }
            }

            // ── Pure: tally ───────────────────────────────────────────────────
            BuiltinFn::Tally => match v {
                Value::Array(arr) => {
                    let mut map: serde_json::Map<String, Value> = serde_json::Map::new();
                    for item in arr {
                        let k = match item {
                            Value::String(s) => s.clone(),
                            other => other.to_string().trim_matches('"').to_string(),
                        };
                        let entry = map.entry(k).or_insert(Value::Number(serde_json::Number::from(0i64)));
                        let prev = entry.as_i64().unwrap_or(0);
                        *entry = Value::Number(serde_json::Number::from(prev + 1));
                    }
                    Ok(Value::Object(map))
                }
                _ => Err(Error::FuncEval("tally: expected array".into())),
            },

            // ── Pure: merge ───────────────────────────────────────────────────
            BuiltinFn::Merge => match v {
                Value::Array(arr) => {
                    let mut out: serde_json::Map<String, Value> = serde_json::Map::new();
                    for item in arr {
                        if let Value::Object(obj2) = item {
                            for (k, val) in obj2 { out.insert(k.clone(), val.clone()); }
                        }
                    }
                    Ok(Value::Object(out))
                }
                Value::Object(_) => Ok(v.clone()),
                _ => Err(Error::FuncEval("merge: expected array of objects".into())),
            },

            // ── Pure: omit ────────────────────────────────────────────────────
            BuiltinFn::Omit => {
                let keys: Vec<&str> = args.iter().filter_map(|a| {
                    if let CompiledArg::Key(k) = a { Some(k.as_ref()) } else { None }
                }).collect();
                let omit = |obj: &serde_json::Map<String, Value>| -> Value {
                    Value::Object(obj.iter()
                        .filter(|(k, _)| !keys.contains(&k.as_str()))
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect())
                };
                match v {
                    Value::Object(obj) => Ok(omit(obj)),
                    Value::Array(arr) => Ok(Value::Array(arr.iter().filter_map(|item| {
                        if let Value::Object(obj) = item { Some(omit(obj)) } else { None }
                    }).collect())),
                    _ => Err(Error::FuncEval("omit: expected object or array".into())),
                }
            }

            // ── Pure: select ──────────────────────────────────────────────────
            BuiltinFn::Select => {
                let keys: Vec<&str> = args.iter().filter_map(|a| {
                    if let CompiledArg::Key(k) = a { Some(k.as_ref()) } else { None }
                }).collect();
                let sel = |obj: &serde_json::Map<String, Value>| -> Value {
                    Value::Object(obj.iter()
                        .filter(|(k, _)| keys.contains(&k.as_str()))
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect())
                };
                match v {
                    Value::Object(obj) => Ok(sel(obj)),
                    Value::Array(arr) => Ok(Value::Array(arr.iter().filter_map(|item| {
                        if let Value::Object(obj) = item { Some(sel(obj)) } else { None }
                    }).collect())),
                    _ => Err(Error::FuncEval("select: expected object or array".into())),
                }
            }

            // ── Pure: rename ──────────────────────────────────────────────────
            BuiltinFn::Rename => {
                let from = key_arg!(0, "rename").to_string();
                let to   = key_arg!(1, "rename").to_string();
                let ren  = |obj: &serde_json::Map<String, Value>| -> Value {
                    Value::Object(obj.iter().map(|(k, x)| {
                        if k == &from { (to.clone(), x.clone()) } else { (k.clone(), x.clone()) }
                    }).collect())
                };
                match v {
                    Value::Object(obj) => Ok(ren(obj)),
                    Value::Array(arr) => Ok(Value::Array(arr.iter().filter_map(|item| {
                        if let Value::Object(obj) = item { Some(ren(obj)) } else { None }
                    }).collect())),
                    _ => Err(Error::FuncEval("rename: expected object or array".into())),
                }
            }

            // ── Pure: set ─────────────────────────────────────────────────────
            BuiltinFn::Set => {
                let key = key_arg!(0, "set").to_string();
                let val = match args.get(1) {
                    Some(CompiledArg::Key(s)) => Value::String(s.as_ref().to_string()),
                    Some(CompiledArg::Number(n)) => {
                        Value::Number(serde_json::Number::from_f64(*n).unwrap())
                    }
                    _ => return Err(Error::FuncEval("set: expected key and value arguments".into())),
                };
                let ins = |obj: &serde_json::Map<String, Value>| -> Value {
                    let mut m = obj.clone();
                    m.insert(key.clone(), val.clone());
                    Value::Object(m)
                };
                match v {
                    Value::Object(obj) => Ok(ins(obj)),
                    Value::Array(arr) => Ok(Value::Array(arr.iter().filter_map(|item| {
                        if let Value::Object(obj) = item { Some(ins(obj)) } else { None }
                    }).collect())),
                    _ => Err(Error::FuncEval("set: expected object or array".into())),
                }
            }

            // ── Pure: coalesce ────────────────────────────────────────────────
            BuiltinFn::Coalesce => {
                let empty = match v {
                    Value::Null => true,
                    Value::Array(a) => a.is_empty(),
                    Value::String(s) => s.is_empty(),
                    _ => false,
                };
                if !empty { return Ok(v.clone()); }
                match args.first() {
                    Some(CompiledArg::Key(k)) => Ok(Value::String(k.as_ref().to_string())),
                    Some(CompiledArg::Number(n)) => Ok(Value::Number(serde_json::Number::from_f64(*n).unwrap())),
                    _ => Ok(Value::Null),
                }
            }

            // ── Pure: pluck ───────────────────────────────────────────────────
            BuiltinFn::Pluck => {
                let key = key_arg!(0, "pluck").to_string();
                match v {
                    Value::Array(arr) => Ok(Value::Array(
                        arr.iter().filter_map(|item| item.get(&key).cloned()).collect()
                    )),
                    Value::Object(obj) => Ok(obj.get(&key).cloned().unwrap_or(Value::Null)),
                    _ => Err(Error::FuncEval("pluck: expected array or object".into())),
                }
            }

            // ── Pure: get ─────────────────────────────────────────────────────
            BuiltinFn::Get => {
                let key = key_arg!(0, "get").to_string();
                match v {
                    Value::Object(obj) => Ok(obj.get(&key).cloned().unwrap_or(Value::Null)),
                    _ => Err(Error::FuncEval("get: expected object".into())),
                }
            }

            // ── Pure: find ────────────────────────────────────────────────────
            BuiltinFn::Find => {
                let ast = match args.first() {
                    Some(CompiledArg::FilterExpr(a)) => Arc::clone(a),
                    _ => return Err(Error::FuncEval("find: expected filter condition".into())),
                };
                match v {
                    Value::Array(arr) => Ok(arr.iter().find(|item| ast.eval(item)).cloned().unwrap_or(Value::Null)),
                    _ => Err(Error::FuncEval("find: expected array".into())),
                }
            }

            // ── Pure: filter_by ───────────────────────────────────────────────
            BuiltinFn::FilterBy => {
                let ast = match args.first() {
                    Some(CompiledArg::FilterExpr(a)) => Arc::clone(a),
                    _ => return Err(Error::FuncEval("filter_by: expected filter condition".into())),
                };
                match v {
                    Value::Array(arr) => Ok(Value::Array(
                        arr.iter().filter(|item| ast.eval(item)).cloned().collect()
                    )),
                    _ => Err(Error::FuncEval("filter_by: expected array".into())),
                }
            }

            // ── map — uses pre-compiled sub-program ───────────────────────────
            BuiltinFn::Map => {
                match args.first() {
                    // MapPath and SubExpr: run the pre-compiled program with `item`
                    // as both input and root, so Filter::Root / PushRoot within the
                    // sub-expression resolves to the current array element (not the
                    // document root) — mirroring Path::collect_with_filter(item, ...).
                    Some(CompiledArg::MapPath(prog)) | Some(CompiledArg::SubExpr(prog)) => {
                        let prog = Arc::clone(prog);
                        match v {
                            Value::Array(arr) => {
                                let mut out = Vec::with_capacity(arr.len());
                                for item in arr {
                                    let res = self.run(&prog, item, item)?;
                                    if !res.is_empty() { out.push(res[0].clone()); } else {
                                        return Err(Error::FuncEval("map: sub-expression produced no value".into()));
                                    }
                                }
                                Ok(Value::Array(out))
                            }
                            _ => Err(Error::FuncEval("map: expected array".into())),
                        }
                    }
                    Some(CompiledArg::MapObjConstruct(fields)) => {
                        let fields = Arc::clone(fields);
                        match v {
                            Value::Array(arr) => {
                                let mut out = Vec::with_capacity(arr.len());
                                for item in arr {
                                    let mut map = serde_json::Map::new();
                                    for (obj_key, prog) in fields.iter() {
                                        let key = match obj_key {
                                            ObjKey::Static(s) => s.clone(),
                                            ObjKey::Dynamic(kf) => {
                                                let kp = Compiler::compile(kf.clone(), "<dyn-key>");
                                                let kv = self.run(&kp, item, item)?;
                                                match kv.into_iter().next() {
                                                    Some(Value::String(s)) => s,
                                                    Some(other) => other.to_string().trim_matches('"').to_string(),
                                                    None => continue,
                                                }
                                            }
                                        };
                                        let vals = self.run(prog, item, item)?;
                                        if let Some(fv) = vals.into_iter().next() { map.insert(key, fv); }
                                    }
                                    out.push(Value::Object(map));
                                }
                                Ok(Value::Array(out))
                            }
                            _ => Err(Error::FuncEval("map: expected array".into())),
                        }
                    }
                    Some(CompiledArg::MapArrConstruct(elems)) => {
                        let elems = Arc::clone(elems);
                        match v {
                            Value::Array(arr) => {
                                let mut out = Vec::with_capacity(arr.len());
                                for item in arr {
                                    let mut row = Vec::with_capacity(elems.len());
                                    for prog in elems.iter() {
                                        let vals = self.run(prog, item, item)?;
                                        if let Some(fv) = vals.into_iter().next() { row.push(fv); }
                                    }
                                    out.push(Value::Array(row));
                                }
                                Ok(Value::Array(out))
                            }
                            _ => Err(Error::FuncEval("map: expected array".into())),
                        }
                    }
                    // Fallback: Raw FuncArg — delegate to old FuncRegistry
                    Some(CompiledArg::Raw(raw_arg)) => {
                        let func_arg = (**raw_arg).clone();
                        let mut func = Func::new();
                        func.name = "map".to_string();
                        func.args = vec![func_arg];
                        self.call_fn(&func, v, root)
                    }
                    _ => Err(Error::FuncEval("map: expected map statement or subpath".into())),
                }
            }

            // ── flat_map ──────────────────────────────────────────────────────
            BuiltinFn::FlatMap => {
                let prog = match args.first() {
                    Some(CompiledArg::MapPath(p)) => Arc::clone(p),
                    Some(CompiledArg::SubExpr(p)) => Arc::clone(p),
                    _ => return Err(Error::FuncEval("flat_map: expected subpath".into())),
                };
                match v {
                    Value::Array(arr) => {
                        let mut out: Vec<Value> = Vec::new();
                        for item in arr {
                            let res = self.run(&prog, item, item)?;
                            for r in res {
                                match r {
                                    Value::Array(inner) => out.extend(inner),
                                    other => out.push(other),
                                }
                            }
                        }
                        Ok(Value::Array(out))
                    }
                    _ => Err(Error::FuncEval("flat_map: expected array".into())),
                }
            }

            // ── join (requires root) ──────────────────────────────────────────
            BuiltinFn::Join => {
                let left_key = key_arg!(1, "join").to_string();
                let right_key = key_arg!(2, "join").to_string();
                let right = run_sub_array!(0);
                let left_arr = match v {
                    Value::Array(a) => a.clone(),
                    _ => return Err(Error::FuncEval("join: expected array on left".into())),
                };
                let mut out: Vec<Value> = Vec::new();
                for li in &left_arr {
                    let lv = li.get(&left_key);
                    for ri in &right {
                        if ri.get(&right_key) == lv {
                            let mut merged = serde_json::Map::new();
                            if let Value::Object(lo) = li { for (k, x) in lo { merged.insert(k.clone(), x.clone()); } }
                            if let Value::Object(ro) = ri {
                                for (k, x) in ro { if !merged.contains_key(k) { merged.insert(k.clone(), x.clone()); } }
                            }
                            out.push(Value::Object(merged));
                        }
                    }
                }
                Ok(Value::Array(out))
            }

            // ── lookup (requires root) ────────────────────────────────────────
            BuiltinFn::Lookup => {
                let left_key  = key_arg!(1, "lookup").to_string();
                let right_key = key_arg!(2, "lookup").to_string();
                let right = run_sub_array!(0);
                let left_arr = match v {
                    Value::Array(a) => a.clone(),
                    other => vec![other.clone()],
                };
                let mut out: Vec<Value> = Vec::new();
                for li in &left_arr {
                    let lv = li.get(&left_key);
                    let matched = right.iter().find(|ri| ri.get(&right_key) == lv);
                    let mut merged = serde_json::Map::new();
                    if let Value::Object(lo) = li { for (k, x) in lo { merged.insert(k.clone(), x.clone()); } }
                    if let Some(Value::Object(ro)) = matched {
                        for (k, x) in ro { if !merged.contains_key(k) { merged.insert(k.clone(), x.clone()); } }
                    }
                    out.push(Value::Object(merged));
                }
                Ok(Value::Array(out))
            }

            // ── resolve (requires root) ───────────────────────────────────────
            BuiltinFn::Resolve => {
                let ref_field  = key_arg!(0, "resolve").to_string();
                let target = run_sub_array!(1);
                let match_field = match args.get(2) {
                    Some(CompiledArg::Key(k)) => k.as_ref().to_string(),
                    _ => ref_field.clone(),
                };
                let process = |item: &Value| -> Value {
                    let ref_val = match item.get(&ref_field) { Some(x) => x, None => return item.clone() };
                    match target.iter().find(|t| t.get(&match_field) == Some(ref_val)) {
                        None => item.clone(),
                        Some(found) => {
                            let mut merged = serde_json::Map::new();
                            if let Value::Object(o) = item { for (k, x) in o { merged.insert(k.clone(), x.clone()); } }
                            merged.insert(ref_field.clone(), found.clone());
                            Value::Object(merged)
                        }
                    }
                };
                Ok(match v {
                    Value::Array(arr) => Value::Array(arr.iter().map(process).collect()),
                    single => process(single),
                })
            }

            // ── deref (requires root) ─────────────────────────────────────────
            BuiltinFn::Deref => {
                let target = run_sub_array!(0);
                let match_field = match args.get(1) {
                    Some(CompiledArg::Key(k)) => Some(k.as_ref().to_string()),
                    _ => None,
                };
                let result = match match_field {
                    Some(field) => target.into_iter().find(|t| t.get(&field) == Some(v)),
                    None => target.into_iter().find(|t| match t {
                        Value::Object(o) => o.values().any(|x| x == v),
                        other => other == v,
                    }),
                };
                Ok(result.unwrap_or(Value::Null))
            }

            // ── formats (needs alias from Func) ───────────────────────────────
            BuiltinFn::Formats => {
                // alias is required by formats; reconstruct a Func and fall through
                // to the old registry since formats also uses its own key-based args.
                let mut func = Func::new();
                func.name = "formats".to_string();
                func.alias = alias.map(String::from);
                func.args = args.iter().filter_map(|a| match a {
                    CompiledArg::Key(k) => Some(crate::context::FuncArg::Key(k.as_ref().to_string())),
                    CompiledArg::Number(n) => Some(crate::context::FuncArg::Number(*n)),
                    CompiledArg::Raw(r) => Some((**r).clone()),
                    _ => None,
                }).collect();
                self.call_fn(&func, v, root)
            }
        }
    }

    // ── Pick execution ────────────────────────────────────────────────────────

    /// Execute a `#pick(...)` expression against a single value.
    ///
    /// Sub-path expressions inside pick are evaluated with the old
    /// `Path::collect_with_filter` interpreter.  This preserves the exact
    /// traversal ordering (LIFO stack) and accumulation semantics (new-first
    /// merge) that the `match_value!` macro produces, which several tests
    /// depend on.  Simple key lookups are handled directly.
    fn exec_pick(
        &mut self,
        elems: &[PickFilterInner],
        obj: &Value,
        root: &Value,
    ) -> Result<Option<Value>, Error> {
        let mut map = Map::new();
        let descendant_key = "descendant".to_string();
        for elem in elems {
            match elem {
                PickFilterInner::Str(key) => {
                    if let Some(v) = obj.get(key) {
                        map.insert(key.clone(), v.clone());
                    }
                }
                PickFilterInner::KeyedStr { key, alias } => {
                    if let Some(v) = obj.get(key) {
                        map.insert(alias.clone(), v.clone());
                    }
                }
                PickFilterInner::Subpath(subpath, reverse) => {
                    let source = if *reverse { root } else { obj };
                    let result = Path::collect_with_filter(source.clone(), subpath);
                    for v in result.0 {
                        merge_into_map(&mut map, &descendant_key, v);
                    }
                }
                PickFilterInner::KeyedSubpath { subpath, alias, reverse } => {
                    let source = if *reverse { root } else { obj };
                    let result = Path::collect_with_filter(source.clone(), subpath);
                    for v in result.0 {
                        merge_into_map(&mut map, alias, v);
                    }
                }
                PickFilterInner::None => {}
            }
        }
        Ok(Some(Value::Object(map)))
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Merge `value` into `map[key]` using the same accumulation semantics as the
/// `match_value!` macro in `context.rs`.
///
/// - First insertion for a key: value stored directly.
/// - Second insertion (String or Number into existing scalar): converted to
///   `Array[new, existing]`.
/// - Subsequent insertions into an existing Array: value appended.
/// - Arrays and Objects from the sub-path are always stored/replaced directly.
fn merge_into_map(map: &mut Map<String, Value>, key: &str, v: Value) {
    match &v {
        Value::String(_) | Value::Number(_) | Value::Bool(_) => {
            if let Some(existing) = map.remove(key) {
                let merged = match existing {
                    Value::Array(mut arr) => {
                        arr.push(v);
                        Value::Array(arr)
                    }
                    scalar => Value::Array(vec![v, scalar]),
                };
                map.insert(key.to_string(), merged);
            } else {
                map.insert(key.to_string(), v);
            }
        }
        Value::Object(ref sub_obj) => {
            // Object results: spread fields directly into the output map.
            for (k, fv) in sub_obj {
                map.insert(k.clone(), fv.clone());
            }
        }
        Value::Array(_) | Value::Null => {
            map.insert(key.to_string(), v);
        }
    }
}

/// Recursive descendant search — mirrors the context.rs `DescendantChild` logic.
fn descend(value: &Value, desc: &FilterDescendant, out: &mut Vec<Value>) {
    match desc {
        FilterDescendant::Single(key) => descend_single(value, key, out),
        FilterDescendant::Pair(key, target_val) => descend_pair(value, key, target_val, out),
    }
}

fn descend_single(value: &Value, key: &str, out: &mut Vec<Value>) {
    match value {
        Value::Object(ref obj) => {
            for (k, v) in obj {
                if k == key {
                    out.push(v.clone());
                } else {
                    descend_single(v, key, out);
                }
            }
        }
        Value::Array(ref arr) => {
            for v in arr {
                descend_single(v, key, out);
            }
        }
        _ => {}
    }
}

fn descend_pair(value: &Value, key: &str, target_val: &str, out: &mut Vec<Value>) {
    match value {
        Value::Object(ref obj) => {
            let mut matched = false;
            for (k, v) in obj {
                if k == key {
                    if let Value::String(s) = v {
                        if s == target_val {
                            matched = true;
                        }
                    }
                }
            }
            if matched {
                out.push(value.clone());
            } else {
                for (_, v) in obj {
                    descend_pair(v, key, target_val, out);
                }
            }
        }
        Value::Array(ref arr) => {
            for v in arr {
                descend_pair(v, key, target_val, out);
            }
        }
        _ => {}
    }
}

/// Hash a string into a `u64`.
#[inline]
fn hash_str(s: &str) -> u64 {
    let mut h = DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

/// Produce a structural document fingerprint.
///
/// Only the *shape* of the document is hashed — key names, array lengths, and
/// value types — not the actual scalar values.  This gives higher resolution-cache
/// hit rates on schema-stable documents where values change between calls (e.g.
/// live metrics payloads with the same field layout but different numbers).
///
/// The first 8 array elements are sampled to balance accuracy vs. cost for large
/// arrays.  The total array length is always included so length changes are caught.
fn hash_doc_structure(v: &Value) -> u64 {
    let mut h = DefaultHasher::new();
    hash_struct_recursive(v, &mut h, 0);
    h.finish()
}

fn hash_struct_recursive(v: &Value, h: &mut DefaultHasher, depth: u8) {
    // Limit recursion depth to keep hashing bounded for very deep documents.
    if depth > 16 {
        return;
    }
    match v {
        Value::Null    => 0u8.hash(h),
        Value::Bool(_) => 1u8.hash(h),
        Value::Number(_) => 2u8.hash(h),
        Value::String(_) => 3u8.hash(h),
        Value::Array(arr) => {
            4u8.hash(h);
            arr.len().hash(h);
            for item in arr.iter().take(8) {
                hash_struct_recursive(item, h, depth + 1);
            }
        }
        Value::Object(obj) => {
            5u8.hash(h);
            obj.len().hash(h);
            for (k, val) in obj {
                k.hash(h);
                hash_struct_recursive(val, h, depth + 1);
            }
        }
    }
}

/// Pack a `&[Value]` slice into the single value that functions receive.
///
/// - One value  → cloned and returned as-is (avoids wrapping a scalar).
/// - Many values → collected into `Value::Array` so reducing functions like
///   `#sum` see all pipeline items.
#[inline]
fn pack_slice(inputs: &[Value]) -> Value {
    if inputs.len() == 1 {
        inputs[0].clone()
    } else {
        Value::Array(inputs.to_vec())
    }
}

// ── Thread-local VM ───────────────────────────────────────────────────────────

// Per-thread VM shared across all `Path::collect` calls on the same thread.
// The compile cache and resolution cache accumulate for the lifetime of the
// thread, not just a single query — this is the primary source of speedup for
// workloads that repeat the same expression.
//
// Re-entrancy: if `Path::collect` is called from inside a running VM execution
// (e.g. from inside a function implementation), `try_borrow_mut` will return
// `Err` and the caller falls back to a fresh temporary `VM`.
thread_local! {
    pub(crate) static THREAD_VM: RefCell<VM> = RefCell::new(VM::new());
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn doc() -> Value {
        json!({
            "company": "Arctiq",
            "customers": [
                {"id": "c1", "name": "Helios",  "tier": "enterprise", "mrr": 12000},
                {"id": "c2", "name": "Nova",    "tier": "mid-market", "mrr": 3400},
                {"id": "c3", "name": "Stratum", "tier": "enterprise", "mrr": 18500}
            ],
            "orders": [
                {"id": "o1", "customer_id": "c1", "total": 15700, "status": "closed_won"},
                {"id": "o2", "customer_id": "c3", "total": 27275, "status": "closed_won"},
                {"id": "o3", "customer_id": "c2", "total": 4080,  "status": "pending"}
            ]
        })
    }

    // ── Compiler ───────────────────────────────────────────────────────────────

    #[test]
    fn compile_simple_path() {
        let prog = Compiler::compile_str(">/company").unwrap();
        // Peephole optimizer fuses PushRoot + GetChild("company") → RootPointer("/company")
        assert_eq!(prog.ops.len(), 1);
        assert!(matches!(&prog.ops[0], Opcode::RootPointer(_)));
        assert!(prog.is_structural);
    }

    #[test]
    fn compile_fn_not_structural() {
        let prog = Compiler::compile_str(">/customers/#len").unwrap();
        assert!(!prog.is_structural);
    }

    // ── Execution ──────────────────────────────────────────────────────────────

    #[test]
    fn run_child_access() {
        let mut vm = VM::new();
        let r = vm.run_str(">/company", &doc()).unwrap();
        assert_eq!(r.0, vec![json!("Arctiq")]);
    }

    #[test]
    fn run_array_index() {
        let mut vm = VM::new();
        let r = vm.run_str(">/customers/[0]/name", &doc()).unwrap();
        assert_eq!(r.0, vec![json!("Helios")]);
    }

    #[test]
    fn run_descendant() {
        let mut vm = VM::new();
        let r = vm.run_str(">/customers/..mrr", &doc()).unwrap();
        assert_eq!(r.0, vec![json!(12000), json!(3400), json!(18500)]);
    }

    #[test]
    fn run_filter() {
        let mut vm = VM::new();
        let r = vm.run_str(">/orders/#filter('status' == 'closed_won')", &doc()).unwrap();
        assert_eq!(r.0.len(), 1);
        assert_eq!(r.0[0].as_array().unwrap().len(), 2);
    }

    #[test]
    fn run_sum() {
        let mut vm = VM::new();
        let r = vm.run_str(">/orders/..total/#sum", &doc()).unwrap();
        assert_eq!(r.0[0].as_f64().unwrap(), 47055.0);
    }

    #[test]
    fn run_any_child() {
        let mut vm = VM::new();
        let r = vm.run_str(">/customers/[0]/*", &doc()).unwrap();
        // {"id":"c1","name":"Helios","tier":"enterprise","mrr":12000} has 4 fields
        assert_eq!(r.0.len(), 4);
    }

    #[test]
    fn run_grouped_child() {
        let mut vm = VM::new();
        let r = vm.run_str(">/('missing' | 'company')", &doc()).unwrap();
        assert_eq!(r.0, vec![json!("Arctiq")]);
    }

    #[test]
    fn run_obj_construct() {
        let mut vm = VM::new();
        let r = vm
            .run_str(r#">{"name": >/company, "count": >/customers/#len}"#, &doc())
            .unwrap();
        let obj = r.0[0].as_object().unwrap();
        assert_eq!(obj["name"], json!("Arctiq"));
        assert_eq!(obj["count"], json!(3));
    }

    #[test]
    fn run_arr_construct() {
        let mut vm = VM::new();
        let r = vm.run_str(">[>/company, >/customers/#len]", &doc()).unwrap();
        let arr = r.0[0].as_array().unwrap();
        assert_eq!(arr[0], json!("Arctiq"));
        assert_eq!(arr[1], json!(3));
    }

    // ── Compile cache ──────────────────────────────────────────────────────────

    #[test]
    fn compile_cache_hit() {
        let mut vm = VM::new();
        let expr = ">/company";
        let _ = vm.run_str(expr, &doc()).unwrap();
        let _ = vm.run_str(expr, &doc()).unwrap();
        let ((_hits, compile_size), _) = vm.cache_stats();
        assert_eq!(compile_size, 1); // compiled once
    }

    // ── Resolution cache ───────────────────────────────────────────────────────

    #[test]
    fn resolution_cache_hit() {
        let mut vm = VM::new();
        let d = doc();
        let expr = ">/customers/[0]/name"; // structural — should be cached
        let r1 = vm.run_str(expr, &d).unwrap();
        let r2 = vm.run_str(expr, &d).unwrap();
        assert_eq!(r1.0, r2.0);
        let (_, (hits, _)) = vm.cache_stats();
        assert_eq!(hits, 1); // second call was a cache hit
    }

    #[test]
    fn resolution_cache_same_structure_different_values() {
        // With structural hashing, {"x":1} and {"x":2} have identical structure
        // (same keys, same types) so the resolution cache DOES hit on the second
        // call.  Values are still correct because cache replay re-reads via pointer.
        let mut vm = VM::new();
        let d1 = json!({"x": 1});
        let d2 = json!({"x": 2});
        let expr = ">/x";
        let r1 = vm.run_str(expr, &d1).unwrap();
        let r2 = vm.run_str(expr, &d2).unwrap();
        // Results are different even though the cache hit — pointer is re-resolved.
        assert_ne!(r1.0, r2.0);
        let (_, (hits, _)) = vm.cache_stats();
        assert_eq!(hits, 1); // second call hit the cache
    }

    #[test]
    fn resolution_cache_miss_on_structurally_different_doc() {
        // Structurally different documents (different keys) must not hit the cache.
        let mut vm = VM::new();
        let d1 = json!({"x": 1});
        let d2 = json!({"y": 1}); // different key name → different structure hash
        let expr = ">/x";
        let _ = vm.run_str(expr, &d1).unwrap();
        let _ = vm.run_str(expr, &d2).unwrap();
        let (_, (hits, _)) = vm.cache_stats();
        assert_eq!(hits, 0); // no cache hit (different structural hashes)
    }

    #[test]
    fn non_structural_skips_resolution_cache() {
        let mut vm = VM::new();
        let d = doc();
        let expr = ">/customers/..mrr/#sum"; // has CallFn → not structural
        let _ = vm.run_str(expr, &d).unwrap();
        let _ = vm.run_str(expr, &d).unwrap();
        let (_, (hits, _)) = vm.cache_stats();
        assert_eq!(hits, 0); // resolution cache never populated
    }
}
