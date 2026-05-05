# jetro-core

[![docs](https://img.shields.io/badge/docs-jetro--core-blue)](https://docs.rs/jetro-core)
[![license](https://img.shields.io/crates/l/jetro-core.svg)](LICENSE)

`jetro-core` is the parser, planner, optimizer, and execution runtime for the
Jetro JSON query language.

The top-level `jetro` crate exposes the small public API. This crate contains
the machinery behind it: AST parsing, physical planning, builtin metadata,
demand propagation, byte/tape navigation, streaming pipelines, and VM fallback.

```rust
use jetro_core::Jetro;

let j = Jetro::from_bytes(br#"{
  "orders": [
    {"id": "o1", "status": "paid", "total": 184.5, "customer": {"name": "Ada"}},
    {"id": "o2", "status": "open", "total": 42.0, "customer": {"name": "Grace"}},
    {"id": "o3", "status": "paid", "total": 312.2, "customer": {"name": "Alan"}}
  ]
}"#.to_vec()).unwrap();

let out = j.collect(r#"
{
  "top_paid": $.orders
    .filter(status == "paid")
    .sort_by(-total)
    .take(2)
    .map({id, customer: customer.name, total}),

  "paid_total": $.orders
    .filter(status == "paid")
    .map(total)
    .sum()
}
"#).unwrap();
```

## Architecture

Jetro is built around one physical plan and several execution backends.

```text
source text
  -> parser
  -> Expr AST
  -> planner
  -> QueryPlan DAG
  -> physical_eval
     -> StructuralIndex backend
     -> ViewPipeline backend
     -> Pipeline backend
     -> VM fallback
  -> serde_json::Value
```

The planner does not commit every expression to one executor too early. It
builds a `QueryPlan`, a DAG of physical nodes. Each node carries:

- the operation to perform, such as `RootPath`, `Pipeline`, `Structural`,
  `Object`, `Array`, `Call`, `Let`, or `Vm`
- backend capabilities
- backend preferences
- execution facts, such as whether a VM fallback is still present

The physical evaluator walks the plan and tries the preferred backend for each
node. If a specialized backend cannot execute a node, execution falls back to
the VM while preserving the same language semantics.

## Core Data Flow

`Jetro::from_bytes` stores the raw JSON bytes. Expensive representations are
lazy:

```text
raw bytes
  -> simd-json tape              when tape/view execution needs it
  -> structural index            when structural search needs it
  -> Val tree                    when owned materialization is required
  -> ObjVec columnar projection  when uniform object arrays benefit from it
```

This lets simple path and streaming queries avoid building a full owned tree.
Materialization happens only when the selected backend or builtin requires it.

## Physical Plan Nodes

Important `PlanNode` variants include:

```text
Root                  document root
RootPath              static field/index path rooted at $
Chain                 path steps applied to another node
Pipeline              source + stages + sink
Structural            deep-search plan with VM fallback
Object / Array         shaped output
Call                  builtin call
Let                   lexical binding
Vm                    compiled bytecode fallback
```

Example:

```text
{
  "names": $.orders.filter(status == "paid").map(customer.name),
  "count": $.orders.count()
}
```

Roughly lowers to:

```text
Object
  field "names" -> Pipeline
    source: RootPath($.orders)
    stages:
      Filter(status == "paid")
      Map(customer.name)
    sink: Collect

  field "count" -> Pipeline
    source: RootPath($.orders)
    stages: []
    sink: Count
```

The object shape remains one physical expression, but each field can use the
best backend available for that sub-expression.

## Pipeline IR

A pipeline is a row source plus an ordered stage list plus a terminal sink.

```text
source -> stage -> stage -> stage -> sink
```

Common sources:

```text
$.orders
@.items
receiver expression
```

Common stages:

```text
filter(predicate)
map(projection)
flat_map(projection)
take(n)
skip(n)
take_while(predicate)
drop_while(predicate)
sort_by(key)
unique_by(key)
```

Common sinks:

```text
collect
first
last
count
sum
avg
min
max
count_by
group_by
index_by
```

The pipeline IR is not only an executor format. It is also where builtin
metadata is used for planning: cardinality, order behavior, materialization
requirements, view compatibility, sink demand, and lowering shape.

## Demand Propagation

Demand propagation is a backward pass through a pipeline. The sink says what it
needs, and each stage translates that demand into the demand it places on its
input.

The core demand model is:

```text
PullDemand:
  All              pull every input
  FirstInput(n)    pull at most n input rows
  UntilOutput(n)   pull until n output rows have survived

ValueNeed:
  None             row payload is not needed
  Predicate        enough value is needed to test a predicate
  Numeric          numeric interpretation is needed
  Whole            full row is needed
```

Example:

```text
$.orders.filter(status == "paid").first()
```

The `first` sink asks for one output. `filter` converts that to
`UntilOutput(1)`, because it may need to inspect many input rows before one row
passes. The executor can stop as soon as the first matching output exists.

```text
sink: First
  demand: FirstInput(1)

filter(status == "paid")
  upstream demand: UntilOutput(1)
```

This is different from a hand-written `filter_first` fusion. The behavior comes
from the builtin demand laws, so any compatible chain can participate without a
new pairwise fused function.

## Barrier Strategy

Some stages are barriers: they need more than one row before they can produce
correct output. Sorting is the most important example.

```text
$.orders.sort_by(-total).take(10)
```

A naive plan fully sorts all rows and then takes ten. A demand-aware plan can
derive a bounded sort strategy:

```text
source: $.orders
stage: SortBy(-total) with SortTopK(10)
stage: Take(10)
sink: Collect
```

More complex chains need careful handling:

```text
$.orders
  .sort_by(-total)
  .take_while(customer.tier == "gold")
  .take(10)
```

`take_while` does not guarantee that the first ten sorted rows satisfy the
predicate. The safe strategy is not simply `SortTopK(10)`. The planner uses
stage metadata to decide whether bounded sorting is safe, whether the sort must
produce until enough downstream outputs survive, or whether full materialization
is required.

The rule is semantic safety first, then performance.

## Builtin Registry

Builtins are intended to be registry-driven. A builtin should describe its
behavior once, and all executors should consume that metadata.

Important metadata includes:

```text
name and aliases
category
cardinality
pipeline lowering
view-stage capability
sink accumulator
sink demand
demand law
order effect
materialization requirement
structural capability
```

This avoids scattering builtin-specific rules across VM, pipeline, view, and
planner code.

For example, `count` declares that it is a sink needing all rows but no row
payload:

```text
pull: All
value: None
order: false
```

That lets executors count rows without materializing full values when the row
source supports it.

## Execution Backends

### Structural Index

Structural plans use `jetro-experimental` bitmap indexing for deep-search and
key-presence style queries.

```text
$..price
$..find(@.status == "paid")
$..shape({id, total})
```

When the structural backend is available, it can answer supported descendant
queries from raw bytes and index data. A VM fallback remains attached for
semantic correctness.

### View Pipeline

The view pipeline executes against a borrowed `ValueView`.

`ValueView` abstracts over:

```text
ValView      borrowed view into an owned Val tree
TapeView     borrowed view into simd-json tape data
```

This is the bridge that allows eligible stages and sinks to run without fully
materializing a `Val` tree.

### Pipeline Backend

The pipeline backend executes lowered source/stage/sink chains. It can use row
streaming, composed stages, barrier strategies, and columnar promotion for
uniform object arrays.

### VM Fallback

The VM is the general executor. It runs compiled bytecode programs and preserves
correctness for expressions that are not yet lowered into a specialized backend.

The VM remains important for:

```text
arbitrary scalar expressions
stage predicates and projections that are not view-native
let bindings
conditionals
patches
complex dynamic expressions
fallback execution
```

## Complex Query Walkthrough

Query:

```text
{
  "leaders": $.orders
    .filter(status == "paid" and total > 100)
    .sort_by(-total)
    .take(3)
    .map({
      id,
      customer: customer.name,
      label: f"{customer.name}: ${total}",
      total
    }),

  "stats": {
    "paid": $.orders.filter(status == "paid").count(),
    "revenue": $.orders.filter(status == "paid").map(total).sum()
  }
}
```

Lowering shape:

```text
Object
  "leaders" -> Pipeline
    source: RootPath($.orders)
    stages:
      Filter(status == "paid" and total > 100)
      SortBy(-total)
      Take(3)
      Map({id, customer, label, total})
    sink: Collect

  "stats" -> Object
    "paid" -> Pipeline
      source: RootPath($.orders)
      stages:
        Filter(status == "paid")
      sink: Count

    "revenue" -> Pipeline
      source: RootPath($.orders)
      stages:
        Filter(status == "paid")
        Map(total)
      sink: Sum
```

Execution behavior:

```text
leaders:
  - source can be read from bytes/tape if available
  - filter is streaming
  - sort is a barrier
  - take(3) pushes bounded demand backward
  - sort may use top-k strategy when safe
  - map shapes only the surviving rows

stats.paid:
  - count needs row existence, not whole row payload
  - filter needs predicate fields
  - executor can avoid building result rows

stats.revenue:
  - sum needs numeric values
  - map(total) can become a numeric projection when view-native
```

The planner does not need a dedicated
`filter_sort_take_map_count_sum_object` implementation. The plan is assembled
from builtin metadata and backend capabilities.

## Design Principles

- One language semantics, multiple execution backends.
- Optimize by metadata and algebraic properties, not by enumerating every chain.
- Keep raw bytes authoritative when possible.
- Materialize only at backend boundaries or when a builtin requires ownership.
- Prefer `ValueView` and tape navigation for read-only streaming work.
- Use the VM as the correctness fallback.
- Make builtin integration registry-driven so adding a builtin does not require
  editing every executor.

## Public Core APIs

The main document handle is:

```rust
let j = jetro_core::Jetro::from_bytes(bytes)?;
let out = j.collect("$.orders.count()")?;
```

For long-lived multi-document use, `JetroEngine` provides an explicit plan cache
and shared VM:

```rust
let engine = jetro_core::JetroEngine::new();
let out = engine.collect_bytes(bytes, "$.orders.count()")?;
```

The top-level `jetro` crate intentionally exposes a smaller facade for end
users.

## Current Direction

The architecture is converging toward:

```text
registry-defined builtins
  -> logical/pipeline lowering
  -> demand-aware physical planning
  -> ValueView/tape execution where possible
  -> VM fallback only where necessary
```

The goal is full tape-aware streaming without duplicating builtin logic across
VM, pipeline, view, and composed execution paths.

## Testing

The core crate contains parser, planner, executor, pipeline, view, structural,
and VM tests.

```bash
cargo test -p jetro-core
cargo test -p jetro-core some_test_name -- --nocapture
```

Use focused tests when changing lowering, demand propagation, or builtin
metadata. Use full core tests before committing executor or value-model changes.

## License

MIT. See [LICENSE](LICENSE).
