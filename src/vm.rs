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
        Func, ObjKey, Path, PathResult, PickFilterInner,
    },
    func::{default_registry, Registry},
    parser,
};

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
    CallFn(Arc<Func>),

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

    /// Peephole optimizer: fuses `PushRoot + GetChild(k)*` chains into a single
    /// `RootPointer("/k1/k2/…")` opcode, eliminating repeated hash-map lookups
    /// and reducing the step count for the common `>/a/b/c` access pattern.
    fn optimize(ops: Vec<Opcode>) -> Vec<Opcode> {
        let mut result: Vec<Opcode> = Vec::with_capacity(ops.len());
        let mut ops = ops.into_iter().peekable();

        while let Some(op) = ops.next() {
            match op {
                Opcode::PushRoot => {
                    // Consume as many trailing GetChild ops as possible.
                    let mut path = String::new();
                    while matches!(ops.peek(), Some(Opcode::GetChild(_))) {
                        if let Some(Opcode::GetChild(k)) = ops.next() {
                            path.push('/');
                            path.push_str(k.as_ref());
                        }
                    }
                    if path.is_empty() {
                        result.push(Opcode::PushRoot);
                    } else {
                        result.push(Opcode::RootPointer(Arc::from(path.as_str())));
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
            Filter::Function(func) => ops.push(Opcode::CallFn(Arc::new(func))),
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
    registry: Rc<RefCell<dyn Registry>>,
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
            registry: default_registry(),
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
                // Functions always return their result as a single value.
                // Spreading array results (out.extend) would mis-unwrap functions
                // like #reverse, #map, #zip that intentionally return arrays.
                let value = pack_slice(inputs);
                let result = self.call_fn(func, &value, root)?;
                out.push(result);
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
            .borrow_mut()
            .call(func, value, Some(&mut ctx))
            .map_err(|e| Error::FuncEval(e.to_string()))
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
