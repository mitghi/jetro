# jetro-core

[<img src="https://img.shields.io/badge/docs-jetro--core-blue"></img>](https://docs.rs/jetro-core)

Parser, compiler, and bytecode VM for the Jetro JSON query language.

---

## Pipeline

```text
source string
     │
     ▼  parser::parse        pest PEG → syntax tree
  Expr (AST)
     │
     ▼  Compiler::emit       structural lowering → Vec<Opcode>
     │  Compiler::optimize   10 peephole passes
  Program                    Arc<[Opcode]>, cheap to clone
     │
     ▼  VM::execute          stack machine over &serde_json::Value
  serde_json::Value
```

The tree-walker in `eval/` is the *reference implementation*: when VM behaviour diverges from it, the VM is wrong. Every new language feature lands in the tree-walker first, then is mirrored in the compiler and opcode set.

---

## Layer 1 — parser (pest PEG)

`grammar.pest` defines the full syntax in pest rules. Highlights:

- **Root / current**: `$` (document root), `@` (current item inside filter / comprehension).
- **Navigation**: `.field`, `[idx]`, `[start:end]`, `..descendant`, `?.optional`, `.{dynamic}`.
- **Method calls**: `.name(arg, named: value)`; named + positional args mix freely.
- **Operators**: arithmetic (`+ - * / %`), comparison (`== != < <= > >= ~=`), logical (`and or not`), nullish (`?|`), type cast (`as int`).
- **Literals**: int, float, bool, null, `"str"`, `'str'`, `f"format {expr}"`.
- **Kind checks**: `x kind number`, `x kind not null`.
- **Comprehensions**: list `[e for v in iter if cond]`, dict `{k: v for …}`, set, generator.
- **Let**: `let x = init in body`.
- **Lambdas**: `lambda x, y: body`.
- **Pipelines**: `base | step1 | step2`, optional named bind `-> name |`.
- **Reserved words** (guarded against identifier use): `and or not for in if let lambda kind is as when patch DELETE true false null`.

The parser walks pest's concrete syntax tree once, producing an `ast::Expr`. Identifiers are stored as `Arc<str>` so that cloning a name into an opcode is a refcount bump instead of a `String` allocation.

### `ast::Expr`

```rust
pub enum Expr {
    Null, Bool(bool), Int(i64), Float(f64), Str(String), FString(Vec<FStringPart>),
    Root, Current, Ident(String),
    Chain(Box<Expr>, Vec<Step>),
    BinOp(Box<Expr>, BinOp, Box<Expr>),
    UnaryNeg(Box<Expr>), Not(Box<Expr>),
    Kind { expr, ty, negate },
    Coalesce(Box<Expr>, Box<Expr>),
    Object(Vec<ObjField>), Array(Vec<ArrayElem>),
    Pipeline { base, steps },
    ListComp { expr, vars, iter, cond },
    DictComp { key, val, vars, iter, cond },
    SetComp { … }, GenComp { … },
    Lambda { params, body },
    Let { name, init, body },
    // … patch blocks, cast, f-string parts
}
```

Variants are orthogonal — each maps to one language concept. Subtrees are `Box<Expr>` (owned, not shared) because the compiler mutates the AST in place during `reorder_and_operands`.

---

## Layer 2 — tree-walker (`eval/`)

`eval::evaluate(&ast, &doc)` is the shortest path from AST to result.

### `Val`

```rust
pub enum Val {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(Arc<str>),
    Arr(Arc<Vec<Val>>),
    Obj(Arc<IndexMap<Arc<str>, Val>>),
}
```

Every compound variant wraps its payload in `Arc`. Cloning a `Val` bumps a refcount — no deep copy, no heap churn during chain traversal. Mutation uses `Arc::try_unwrap` with a clone-on-fallback pattern.

`Val` is the engine's internal currency. Conversion to / from `serde_json::Value` happens exactly twice per query: once on entry (`Val::from(&doc)`), once on exit (`Val → Value`).

### `Env`

```rust
struct Env {
    root:     Val,
    current:  Val,
    vars:     SmallVec<[(Arc<str>, Val); 4]>,
    registry: Arc<MethodRegistry>,
}
```

Let-bindings and comprehension variables push onto `vars`; scope exit truncates. The inline capacity of 4 covers every realistic query, and the linear lookup is faster than any hashmap at that size.

### `MethodRegistry`

A `HashMap<&'static str, Arc<dyn Method>>` that the user can extend with custom methods. `Method` is:

```rust
pub trait Method: Send + Sync + 'static {
    fn call(&self, recv: &Val, args: &[Val], env: &Env) -> Result<Val, EvalError>;
}
```

The registry itself is `Clone` (all entries are `Arc<dyn Method>`), so threading it through recursive calls is free.

---

## Layer 3 — bytecode VM (`vm.rs`)

The VM is an accelerated path on top of the tree-walker. It uses:

1. A **compile cache** — parse + compile happen once per unique expression string.
2. A **pointer cache** — purely structural programs (`$.a.b[0]`) cache their resolved pointer path keyed by `(doc_hash, program_id)` and skip traversal on re-run.
3. A **flat opcode array** — `Arc<[Opcode]>`; the execution loop iterates linearly with no recursion for navigation/arithmetic.

### `Program`

```rust
pub struct Program {
    pub ops:           Arc<[Opcode]>,
    pub source:        Arc<str>,
    pub id:            u64,           // hash of source
    pub is_structural: bool,          // eligible for pointer-cache
}
```

A `Program` is immutable and `Arc`-cloneable. The `is_structural` flag is set when every opcode is pure navigation (`PushRoot`, `GetField`, `GetIndex`, `GetSlice`, `OptField`, `RootChain`, `GetPointer`) — those programs are eligible for the resolution cache.

### `Opcode`

Representative slice — the full set is ~60 variants, grouped by purpose:

```rust
pub enum Opcode {
    // Literals
    PushNull, PushBool(bool), PushInt(i64), PushFloat(f64), PushStr(Arc<str>),
    // Context
    PushRoot, PushCurrent,
    // Navigation
    GetField(Arc<str>), GetIndex(i64), GetSlice(Option<i64>, Option<i64>),
    DynIndex(Arc<Program>), OptField(Arc<str>), Descendant(Arc<str>),
    DescendAll, InlineFilter(Arc<Program>), Quantifier(QuantifierKind),

    // Peephole fusions
    RootChain(Arc<[Arc<str>]>),          // PushRoot + GetField+
    FilterCount(Arc<Program>),           // filter(p).count()
    FindFirst(Arc<Program>),             // filter(p).first()
    FindOne(Arc<Program>),               // filter(p).one()
    FilterMap { pred, map },             // filter(p).map(f)
    FilterFilter { p1, p2 },             // filter(p1).filter(p2)
    MapMap { f1, f2 },                   // map(f1).map(f2)
    MapSum(Arc<Program>),                // map(f).sum()
    MapAvg(Arc<Program>),                // map(f).avg()
    MapFlatten(Arc<Program>),            // map(f).flatten()
    MapUnique(Arc<Program>),             // map(f).unique()
    TopN { n: usize, asc: bool },        // sort()[0:n]
    FilterTakeWhile { pred, stop },
    FilterDropWhile { pred, drop },
    EquiJoin { rhs, lhs_key, rhs_key },

    // Idents, ops, short-circuit
    LoadIdent(Arc<str>),
    Add, Sub, Mul, Div, Mod,
    Eq, Neq, Lt, Lte, Gt, Gte, Fuzzy, Not, Neg,
    CastOp(CastType),
    AndOp(Arc<Program>), OrOp(Arc<Program>), CoalesceOp(Arc<Program>),

    // Methods, construction, pipelines, comprehensions, patch
    CallMethod(Arc<CompiledCall>), CallOptMethod(Arc<CompiledCall>),
    MakeObj(Arc<[CompiledObjEntry]>), MakeArr(Arc<[Arc<Program>]>),
    FString(Arc<[CompiledFSPart]>),
    KindCheck { ty, negate },
    SetCurrent, BindVar(Arc<str>), StoreVar(Arc<str>),
    BindObjDestructure(Arc<BindObjSpec>), BindArrDestructure(Arc<[Arc<str>]>),
    LetExpr { name, body }, ListComp(Arc<CompSpec>), DictComp(…), SetComp(…),
    GetPointer(Arc<str>),                // resolution-cache fast-path
    PatchEval(Arc<Expr>),                // delegates to tree-walker
}
```

### Compilation

```rust
Compiler::compile(expr: &Expr, source: &str) -> Program
```

Runs three phases:

1. **AST rewrite** — `reorder_and_operands` commutes `a and b` if `b` is cheaper/more-selective. The estimate is shallow but cheap.
2. **Lowering** — `emit` walks the (possibly rewritten) AST and produces a flat `Vec<Opcode>`. Sub-programs (lambda bodies, method args, comprehension iterators) are recursively compiled and stored as `Arc<Program>`.
3. **Optimisation** — `optimize` runs the peephole passes below.
4. **Post-pass** — `analysis::dedup_subprograms` canonicalises identical `Arc<Program>` sub-trees (common-subexpression elimination at the Program level).

### Peephole passes

Run in this order — each may expose new fusions for the next:

1. **`root_chain`** — `PushRoot` followed by `GetField*` collapses to a single `RootChain` opcode. The whole path is resolved against `doc` via one pointer walk; structural programs made entirely of `RootChain` become eligible for the pointer cache.
2. **`filter_count`** — `filter(p).count()` → `FilterCount(p)`. No intermediate array.
3. **`filter_map / map_map / filter_filter`** — single-pass fusion of adjacent combinators.
4. **`find_quantifier`** — `filter(p).first()` / `.one()` → early-exit `FindFirst` / `FindOne`.
5. **`strength_reduction`** — `len() == 0` → `is_empty`; `.sort()[0:n]` → `TopN`; `map(f).sum()` / `.avg()` fused.
6. **`redundant_op_removal`** — `PushCurrent` followed immediately by `SetCurrent` on the same value cancels.
7. **`kind_check_fold`** — `expr kind T` where `expr` is a literal folds to a `PushBool`.
8. **`method_const_fold`** — builtins applied to literal args fold at compile time (`"ab".len()` → `PushInt(2)`).
9. **`expr_const_fold`** — arithmetic on adjacent integer/float literals folds.
10. **`nullness_specialisation`** — when upstream analysis proves a field cannot be absent, `OptField` downgrades to `GetField` (one fewer branch per execution).

Each pass has a `PassConfig` toggle; `VM::set_pass_config` lets you disable any of them at runtime (the cache keys off the config hash, so toggles don't return stale programs).

### VM state

```rust
pub struct VM {
    registry:      Arc<MethodRegistry>,
    compile_cache: HashMap<(u64, String), Arc<Program>>,
    compile_lru:   VecDeque<(u64, String)>,
    compile_cap:   usize,
    path_cache:    PathCache,
    doc_hash:      u64,
    config:        PassConfig,
}
```

- **`compile_cache`** — keyed by `(pass_config_hash, source)`. LRU eviction at `compile_cap` (default 512).
- **`path_cache`** — structural programs map `(doc_hash, program_id) → resolved Val`. `doc_hash` is computed once per top-level `execute()` via `hash_val_structure`, which hashes both structure *and* primitive values (bounded at depth 8). Two documents with the same shape but different leaf values produce different hashes — otherwise the cache would return stale results across distinct docs.
- **`registry`** — shared via `Arc` so multiple VMs on the same thread share method implementations.

### Execution

```rust
pub fn run_str(&mut self, expr: &str, doc: &Value) -> Result<Value, EvalError>;
pub fn execute(&mut self, program: &Program, doc: &Value) -> Result<Value, EvalError>;
```

`execute` enters a tight loop over `program.ops`, maintaining a `Vec<Val>` operand stack. Opcodes that need sub-programs (method args, lambda bodies, comprehension iterators) recurse through `exec(sub_program, env)`; the doc-hash is *not* recomputed on recursion.

---

## Layer 4 — analyses (`analysis.rs`, `schema.rs`, `plan.rs`, `cfg.rs`, `ssa.rs`)

Optional IRs. None are mandatory for correctness; they exist so advanced callers can specialise programs further or export a readable data-flow graph.

| Module     | Produces                                            | Used for                                    |
|------------|-----------------------------------------------------|---------------------------------------------|
| `analysis` | Type / Nullness / Cardinality / cost / monotonicity | Optimiser heuristics; CSE via `dedup_subprograms` |
| `schema`   | Shape inference from sample JSON                    | Specialise `OptField → GetField`            |
| `plan`     | Logical relational plan IR                          | Filter push-down, join detection            |
| `cfg`      | Basic blocks, edges, dominators, loop headers       | Liveness, slot allocator                    |
| `ssa`      | SSA numbering + phi nodes                           | Common-subexpression elimination, def-use   |

The analyses operate on `Program`, not `Expr` — they observe the already-lowered bytecode.

---

## Public API

```rust
// One-shot tree-walker.
pub fn query(expr: &str, doc: &Value) -> Result<Value>;
pub fn query_with(expr: &str, doc: &Value, registry: Arc<MethodRegistry>) -> Result<Value>;

// Persistent VM (thread-local in the umbrella; fresh in this crate).
pub struct Jetro { … }
impl Jetro {
    pub fn new(doc: Value) -> Self;
    pub fn collect<S: AsRef<str>>(&self, expr: S) -> Result<Value, EvalError>;
}

// Lower-level building blocks.
pub use vm::{VM, Compiler, Program};
pub use expr::Expr;              // typed-phantom wrapper for compile-time expression literals
pub use eval::{Method, MethodRegistry, EvalError};
pub use parser::{parse, ParseError};
pub use graph::Graph;            // multi-document queries
pub use engine::Engine;          // high-level wrapper
```

### `Jetro` — document-bound convenience

```rust
use jetro_core::Jetro;
use serde_json::json;

let j = Jetro::new(json!({
    "store": { "books": [
        {"title": "Dune",       "price": 12.99},
        {"title": "Foundation", "price":  9.99},
    ]}
}));

let titles = j.collect("$.store.books.filter(price > 10).map(title)").unwrap();
assert_eq!(titles, serde_json::json!(["Dune"]));
```

`Jetro::collect` uses a thread-local VM so the compile cache accumulates across calls on the same thread.

### `query` — one-shot

```rust
let n = jetro_core::query("$.store.books.len()", &doc).unwrap();
```

Uses the tree-walker directly — no compile/resolution caches. Use `Jetro` or `VM` for repeated queries.

### Custom methods

```rust
use jetro_core::{VM, Method, Val, Env, EvalError};

struct Shout;
impl Method for Shout {
    fn call(&self, recv: &Val, _: &[Val], _: &Env) -> Result<Val, EvalError> {
        match recv {
            Val::Str(s) => Ok(Val::Str(s.to_uppercase().into())),
            _           => Err(EvalError("shout: expected string".into())),
        }
    }
}

let mut vm = VM::new();
vm.register("shout", Shout);
let r = vm.run_str(r#""hello".shout()"#, &serde_json::json!(null)).unwrap();
assert_eq!(r, serde_json::json!("HELLO"));
```

### `Graph` — multi-document queries

`Graph::new({node: Value, …})` merges named documents into a virtual root and evaluates an expression against the combined object. Joins are expressed inside the query (`.#join('orders', 'customer_id', 'id')`). The `jetrodb` crate's `GraphBucket` is a storage-backed orchestrator on top of this primitive.

---

## Performance notes

- **Clone cost**: `Val::clone` is always O(1) — the cost is one atomic increment. Nothing in the VM copies compound payloads.
- **Pointer cache hits**: A structural program re-run against the same document is a `HashMap` lookup plus an `Arc` bump. No traversal.
- **Compile cache**: First call: parse + compile + optimize (~microseconds for short queries). Subsequent: `HashMap` lookup.
- **Peephole fusions**: `map(f).sum()` runs one pass, no intermediate vec. `filter(p).first()` early-exits on the first match.
- **Dead-code elimination via CSE**: `dedup_subprograms` canonicalises identical sub-programs across the compiled tree — `map(x.price).sum() + map(x.price).avg()` shares one `map(x.price)` sub-program by `Arc` identity after this pass.
- **`smallvec` for scopes**: `Env::vars` keeps the first 4 let-bindings inline. Real queries rarely exceed this.

---

## Error types

```rust
pub enum Error {
    Parse(ParseError),
    Eval(EvalError),
}

pub struct ParseError(pub String);
pub struct EvalError(pub String);
```

Both implement `std::error::Error + Display`. The umbrella `jetro` crate re-exports these and keeps the same shape. `jetrodb` extends with `Db` and `Io` variants.

---

## License

MIT. See [LICENSE](LICENSE).
