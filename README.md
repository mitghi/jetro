# jetro

[![crates.io](https://img.shields.io/crates/v/jetro.svg)](https://crates.io/crates/jetro)
[![docs.rs](https://img.shields.io/badge/docs-jetro-blue)](https://docs.rs/jetro)
[![license](https://img.shields.io/crates/l/jetro.svg)](LICENSE)

<p align="center">
  <img src="media/jetro_logo.png" alt="Jetro" width="320">
</p>

📖 **[The Jetro Book](https://mitghi.github.io/jetro-book/)** is the best
place to start, guided tour, full grammar reference, each builtin with
working examples and some recipes.

Jetro is a compact query language and JSON data processing engine written in Rust. it 
borrows from two playbooks: Haskell compiler and database planners without pretending to 
be either.

```rust
use jetro::Jetro;
use serde_json::json;

let data = br#"{...}"#;
let jetro = Jetro::from_bytes(data.to_vec())?;
let report = jetro.collect(r#"
{
  "top_paid_premium": $.orders
    .filter(status == "paid" and total >= 100)
    .filter(customer.tier == "gold" or customer.tier == "platinum")
    .sort_by(-total)
    .take(2)
    .map({
      order_id: id,
      who: customer.name,
      tier: customer.tier,
      amount: total,
      label: f"order {@.id}: {customer.name} ({customer.tier}) USD {@.total}",
      line_total: items.map(qty * price).sum(),
      last_event: match events.last() with {
          {kind: "delivered", at: t}    -> {state: "ok",     at: t},
          {kind: "shipped",   at: t}    -> {state: "moving", at: t},
          {kind: "refund", reason: r}   -> {state: "refund", reason: r},
          _                             -> {state: "unknown"}
      }
    }),

  "paid_total": $.orders
    .filter(status == "paid")
    .map(total)
    .sum()
}
"#,
)?;

# Ok::<(), Box<dyn std::error::Error>>(())
```

## Install

```toml
[dependencies]
jetro = "0.5.10"
```

## Why Jetro?

- **Expressive, Compact and Familiar**  
  Jetro reads like the functional iterator chains you already know, it helps you get
  productive in minutes.

- **Query, Shape and Mutate**  
  Jetro lets you write rich, multi-stage queries that is easy to read, reshape the result into
  whatever form you need - scalars, arrays, nested objects, computed values, distilled summaries
  and mutate the document in place.
  
- **Laziness: Try Hard to Do Less**  
  Jetro's fastest path is the one it never takes, it tries to figure out the smallest possible amount 
  of work upfront and then does only that. It borrows ideas from lazy functional language implementations 
  like Haskell's demand analysis and stream fusion and adapts them to JSON query engine. The engine pulls 
  data on demand, fuses pipeline stages into a single loop when applicable.  

- **Speed**  
  Jetro is engineered for throughput from initial parsing phase till output. It accelerates 
  your query with a suite of techniques applied at every layer of the stack. SIMD-accelerated 
  parsing and structural scanning at the byte level, demand analysis and stream fusion at the 
  planner level, and a bytecode VM with cached plans and paths at the runtimelevel. For deep-search 
  queries, Jetro can build a bitmap index over the parsed structure: one  bit per structural position, 
  sliced by depth and key, so descendant scans become bitwise  AND/OR over packed words instead of 
  recursive tree walks. Selected automatically by the planner.
  
### Where Jetro is a good fit

  - **Stream processing** — Kafka / NATS / Pulsar consumers, Flink / Spark /
    where each message is JSON and the work is filter -> reshape -> enrich -> 
	forward. Embed Jetro as the per-message transformer, hold one long-lived 
	`JetroEngine` per consumer, and pay only execution cost per record.
  - **API response shaping**, log analytics, ETL transforms - workloads where
    response enrichment and reshaping is dominant.
  - **Ad-hoc and dynamic queries** that aren't known at compile time.
  - **Observability pipelines** — filter / reshape / aggregate spans, events,
    or audit records inline before forwarding.
  - **Document mutation** bulk JSON migrations, config rewrites, fixture
    generation; chain-writes fused into one traversal per document.
  - **jq**: covers common jq use cases with a simpler, more intuitive method-chain syntax.

## Performance

Jetro aims to stay as close as realistically possible to idiomatic, statically typed Rust on common query workloads.

The repository ships a cold-start benchmark that runs each query end-to-end
(parse + execute + serialize), similar to embedded usage.

Four engines are compared on the same input (`N = 8000` records, ~3.9 MB):

| Engine | Style |
|---|---|
| **Rust + serde** | Idiomatic Rust: `serde_json` parse into typed structs, then `Iterator` chains. The hand-tuned baseline. |
| **Go + `encoding/json`** | Idiomatic Go: stdlib JSON parse into typed structs, then `for` loops and `sort.Slice`. |
| **jaq** | Pure-Rust `jq` clone (compile + interpret per call). |
| **jetro** | Single Jetro expression, planned and executed end-to-end. |

### Cold single-run results (lower is better)

| Case | Rust | **Jetro** | Go | [Jaq](https://github.com/01mf02/jaq) |
|---|---|---|---|---|
| active top-100 expensive-item revenue | 8.65ms | **13.18ms (1.52x)** | 51.59ms (5.97x) | 97.49ms (11.28x) |
| flatmap+sort all-items+take+project | 8.09ms | **9.97ms (1.23x)** | 54.42ms (6.73x) | 140.95ms (17.43x) |
| sort+skip+take+project | 7.37ms | **9.99ms (1.36x)** | 51.30ms (6.96x) | 85.22ms (11.56x) |
| filter+flatmap-tags+unique | 7.23ms | **8.56ms (1.18x)** | 52.64ms (7.29x) | 78.65ms (10.89x) |
| flatmap+filter+map-arith+sum | 6.90ms | **9.58ms (1.39x)** | 50.87ms (7.37x) | 135.21ms (19.59x) |
| filter+sort+take+fstring | 6.95ms | **7.72ms (1.11x)** | 53.85ms (7.75x) | 80.19ms (11.54x) |
| filter+flatmap+avg | 6.88ms | **7.26ms (1.06x)** | 53.06ms (7.72x) | 83.29ms (12.11x) |
| sort+take+nested-computed-projection | 6.89ms | **7.47ms (1.09x)** | 53.20ms (7.73x) | 81.88ms (11.89x) |
| 5-stage filter chain + count | 7.02ms | **8.57ms (1.22x)** | 54.19ms (7.72x) | 94.81ms (13.51x) |
| count_by(active) / group_by+map | 7.18ms | **8.23ms (1.15x)** | 52.27ms (7.28x) | 80.04ms (11.15x) |
| sort+take+map+unique (top-300 zips) | 7.32ms | **8.41ms (1.15x)** | 53.89ms (7.36x) | 82.64ms (11.29x) |
| flatmap+map+unique+len (all prices) | 6.93ms | **9.03ms (1.30x)** | 54.44ms (7.86x) | 98.25ms (14.18x) |
| filter+map+sum | 6.92ms | **7.85ms (1.14x)** | 52.74ms (7.62x) | 76.49ms (11.06x) |
| flat_map+filter+count | 6.78ms | **8.30ms (1.22x)** | 53.16ms (7.84x) | 106.33ms (15.69x) |
| filter+flat_map+map+sum | 6.82ms | **7.88ms (1.16x)** | 53.07ms (7.79x) | 93.67ms (13.74x) |
| sort_by+take+map (top10) | 7.05ms | **7.69ms (1.09x)** | 55.23ms (7.83x) | 82.00ms (11.63x) |
| map+unique (cities) | 7.19ms | **7.92ms (1.10x)** | 53.21ms (7.40x) | 75.96ms (10.57x) |
| map (deep projection) | 8.07ms | **12.83ms (1.59x)** | 56.10ms (6.95x) | 180.21ms (22.33x) |
| map f-string | 8.93ms | **9.46ms (1.06x)** | 54.16ms (6.06x) | 110.56ms (12.38x) |
| flat_map+map (all prices) | 8.34ms | **8.96ms (1.07x)** | 54.83ms (6.57x) | 94.72ms (11.35x) |
| filter+first | 6.99ms | **7.25ms (1.04x)** | 52.71ms (7.54x) | 69.00ms (9.87x) |
| skip+take+map (pagination) | 7.25ms | **6.30ms (0.87x)** | 52.16ms (7.20x) | 67.76ms (9.35x) |
| filter+map+avg | 7.09ms | **9.28ms (1.31x)** | 52.14ms (7.36x) | 76.99ms (10.86x) |
| README showcase (2-filter+sort+take+match) | 7.20ms | **8.62ms (1.20x)** | 52.14ms (7.24x) | 85.36ms (11.85x) |

Reproduce:

```bash
cargo run -p jetro-core --release --example bench_cold
(cd bench/go && go run .)
```

## API

```rust
use jetro::{Jetro, JetroEngine};

let jetro = Jetro::from_bytes(json_bytes)?;
let value = jetro.collect("$.some.expression")?;

let engine = JetroEngine::new();
let first_two = engine.collect_ndjson_matches_file(
    "events.ndjson",
    "level == \"error\"",
    2,
)?;
```

`Jetro` is the byte-oriented document handle. `JetroEngine` is the long-lived
engine for cached plans, reusable VM state, and NDJSON processing.

### NDJSON

NDJSON APIs evaluate each non-empty line as an independent JSON document while
reusing one prepared query plan for the stream.

## Quick Language Preview

```text
$                         root document
@                         current item inside map/filter/lambda

$.user.name               field access
$.user?.name              null-safe field access
$.items[0]                index
$.items[1:5]              slice
$..price                  recursive descent
```

### Query

```text
$.books.filter(price > 10)
$.books.sort_by(-rating).take(5)
$.orders.filter(status == "paid").map(total).sum()
```

### Shape

```text
$.books.map({title, price})
$.orders.map({
  id,
  customer: customer.name,
  city: customer.address.city,
  total
})
```

### Compose

```text
{
  "featured": $.books
    .filter(rating >= 4.5)
    .sort_by(-price)
    .take(3)
    .map({title, author, price}),

  "stats": {
    "count": $.books.count(),
    "avg_price": $.books.map(price).avg(),
    "tags": $.books.flat_map(tags).unique().sort()
  }
}
```

### Bind and Format

```text
let min_total = 100 in
$.orders
  .filter(total >= min_total)
  .map({
    id,
    label: f"{customer.name}: ${total}"
  })
```

### Group and Index

```text
$.orders.group_by(status)
$.users.index_by(id)
$.events.count_by(type)
```

### Patch

```text
$.user.name.set("Ada")
$.cart.items.filter(qty == 0).delete()
patch $ { .user.active: true }
```

### Pattern Match

```text
match $.user with {
    {role: "admin"}                    -> "full",
    {role: "user", verified: true}     -> "limited",
    {role: r, ...*rest}                -> {...*rest, role: r},
    _                                  -> "denied"
}

$..match {
    {tag: "click", id: i} -> i,
    _                     -> false
}
```

Full syntax reference: [jetro-core/src/SYNTAX.md](jetro-core/src/SYNTAX.md)

## See

- **[Jetrocli](https://github.com/mitghi/jetrocli)**: For interactive use in Terminal.
- **[Jetro Emacs Plugin](https://github.com/mitghi/jetromacs)** use Jetro in Emacs.
- **[Jetro Python Binding](https://github.com/mitghi/jetro-py)**: Python Binding for Jetro.
- **[Jetro Dart Binding](https://github.com/mitghi/jetro-dart)**: Dart/Flutter Binding for Jetro.

## Learn More

- **[The Jetro Book](https://mitghi.github.io/jetro-book/)** the canonical guide - start here.
- [jetro-core/src/SYNTAX.md](jetro-core/src/SYNTAX.md) - syntax reference
- [CHANGELOG.md](CHANGELOG.md) - release notes

## License

MIT
