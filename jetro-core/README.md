# jetro-core

[<img src="https://img.shields.io/badge/docs-jetro--core-blue"></img>](https://docs.rs/jetro-core)

Parser, compiler, and execution engine for the Jetro JSON query language.

---

## Architecture

A query travels through five systems in order:

```
source string
      │
      ▼  parser
   Expr AST
      │
      ▼  planner
   QueryPlan
      │
      ▼  physical evaluator
      │
      ├─ pipeline executor  ←  preferred path for filter/map/take chains
      └─ VM fallback        ←  general scalar path
      │
      ▼
   serde_json::Value
```

Each system has a narrow interface to the next. The planner decides which execution path to use; the physical evaluator dispatches accordingly.

---

## Parser

Converts source text to an `Expr` AST using a pest PEG grammar (`grammar.pest`). The parser runs once per query string; the resulting AST feeds directly into the planner or compiler.

The language supports field navigation, method chains, comprehensions, let-bindings, lambdas, pipelines, patch writes, and format strings. Root is `$`; current item inside a filter or comprehension is `@`.

---

## Planner

Classifies the AST and produces a `QueryPlan`. Its main job is recognising which execution path is best:

- **Pipeline plan** — for queries that are a chain of source + stages + sink (`filter`, `map`, `take`, `sort`, `group_by`, etc.). The planner lowers these into a typed IR with an explicit source, an ordered stage list, and a terminal sink.
- **VM plan** — for everything else: arbitrary expressions, comprehensions, patch writes, conditionals, let-bindings.

The planner also performs early optimisations: fusing adjacent stages where possible, annotating demand constraints (so a `take(n)` propagates a stop-early signal to upstream stages), and selecting between columnar and row-oriented execution.

---

## Pipeline executor

The preferred execution path for chained queries. A pipeline is a pull-based dataflow: the source produces rows, each stage transforms or filters them, and the sink collects the final result.

Four execution paths are tried in priority order:

1. **Indexed** — single-element position lookups exit immediately without iterating.
2. **Columnar** — when the source is a uniform array of objects (all rows share the same key set), it is promoted to a struct-of-arrays layout. Aggregate operations (`sum`, `avg`, `min`, `max`) over numeric columns run without per-row dispatch overhead.
3. **Composed** — when the pipeline has been fully lowered to the composed substrate, stages are linked into a monoidal chain and evaluated without a per-row VM call.
4. **Legacy** — the general row-streaming fallback. Iterates source rows, runs each stage's compiled sub-program through the VM, and accumulates results into the sink. Always produces a result.

---

## Compiler

Translates an `Expr` AST into a flat bytecode `Program` (`Arc<[Opcode]>`). Once compiled, a program is immutable and cheaply shared via `Arc`.

Compilation runs three phases: an AST rewrite that reorders `and`-operands by selectivity, a lowering pass that emits opcodes recursively, and a sequence of eleven peephole passes. Passes include root-path fusion (eliminating per-field stack traffic for `$.a.b.c`), constant folding, strength reduction (`sort()[0]` → `min()`), redundant-op elimination (`reverse().reverse()` → nothing), and demand annotation (`filter` before `take(n)` learns to stop early).

Compiled programs are cached in the VM's compile cache keyed by expression string, so repeated queries pay the compilation cost only once.

---

## VM

A stack machine that executes a `Program` against a document. It is the fallback for any query the pipeline executor cannot handle, and is also called by the pipeline executor for per-row predicate evaluation inside `Filter` and `Map` stages.

The VM maintains two caches across its lifetime:

- **Compile cache** — maps expression strings to compiled `Program` objects.
- **Path cache** — maps `(document hash, program id)` pairs to resolved values for purely structural navigation programs (e.g. `$.store.books`). On a cache hit the traversal is skipped entirely.

The VM is typically long-lived (thread-local), so both caches warm up across successive queries against the same or similar documents.

---

## Value representation

All internal values are `Val`, an enum with variants for null, bool, int, float, string, array, and object. Compound variants wrap their payload in `Arc`, so cloning a value is always O(1) regardless of its size. Mutation follows copy-on-write using `Arc::try_unwrap`.

Homogeneous arrays get specialised representations: `IntVec`, `FloatVec`, and `ObjVec` (struct-of-arrays for uniform object arrays). These enable the columnar fast paths in the pipeline executor.

Conversion between `Val` and `serde_json::Value` happens exactly twice per query: once on entry, once on exit.

---

## Quick start

```rust
use jetro_core::Jetro;

let j = Jetro::from_bytes(br#"{
  "store": { "books": [
    {"title": "Dune",       "price": 12.99},
    {"title": "Foundation", "price":  9.99}
  ]}
}"#.to_vec()).unwrap();

let titles = j.collect("$.store.books.filter(price > 10).map(title)").unwrap();
assert_eq!(titles, serde_json::json!(["Dune"]));
```

`Jetro::collect` uses a thread-local VM so the compile and path caches accumulate across calls.

---

## License

MIT. See [LICENSE](LICENSE).
