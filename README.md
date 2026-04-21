# jetro

[<img src="https://img.shields.io/badge/docs-jetro-blue"></img>](https://docs.rs/jetro)

Jetro is a library for querying, transforming, and comparing JSON. It provides a compact expression language, a bytecode VM with caching, and an optional persistence layer with expression-driven indexes.

Jetro has minimal dependencies and operates on top of [serde_json](https://serde.rs).

This project is in experimental phase, please note that significant changes may occur. 

---

## Quick start

```rust
use jetro::Jetro;
use serde_json::json;

let j = Jetro::new(json!({
    "store": {
        "books": [
            {"title": "Dune",       "price": 12.99},
            {"title": "Foundation", "price":  9.99}
        ]
    }
}));

let titles = j.collect("$.store.books.filter(price > 10).map(title)").unwrap();
assert_eq!(titles, json!(["Dune"]));
```

For one-shot evaluation (tree-walker, no VM caching):

```rust
let result = jetro::query("$.store.books.len()", &doc).unwrap();
```

---

## Rust API tour

A progressive walk through Jetro's Rust API over one rich payload — simple field access all the way to blocking stream-joins. Every snippet in this section runs against the same document:

```rust
use jetro::prelude::*;

let doc = json!({
  "store": {
    "name": "Nova Books",
    "currency": "USD",
    "books": [
      {"id": "b1", "title": "Dune",        "author": "Frank Herbert",  "price": 12.99, "year": 1965, "tags": ["sci-fi","classic"],  "stock": 14, "ratings": [5,5,4,5,3]},
      {"id": "b2", "title": "Foundation",  "author": "Isaac Asimov",   "price":  9.99, "year": 1951, "tags": ["sci-fi","classic"],  "stock":  0, "ratings": [5,5,5]},
      {"id": "b3", "title": "Neuromancer", "author": "William Gibson", "price": 19.50, "year": 1984, "tags": ["cyberpunk"],         "stock":  3, "ratings": [4,5,4,5]},
      {"id": "b4", "title": "The Hobbit",  "author": "J.R.R. Tolkien", "price": 14.25, "year": 1937, "tags": ["fantasy","classic"], "stock": 22, "ratings": [5,5,5,5,4,5]},
      {"id": "b5", "title": "Hyperion",    "author": "Dan Simmons",    "price": 18.00, "year": 1989, "tags": ["sci-fi"],            "stock":  7, "ratings": [5,5,4]}
    ],
    "customers": [
      {"id": "c1", "name": "Ada Lovelace", "tier": "gold",   "credits": 250.0},
      {"id": "c2", "name": "Alan Turing",  "tier": "silver", "credits":  40.0},
      {"id": "c3", "name": "Grace Hopper", "tier": "gold",   "credits": 180.0}
    ],
    "orders": [
      {"id": "o1", "customer_id": "c1", "items": [{"book_id":"b1","qty":2},{"book_id":"b4","qty":1}], "status": "paid"},
      {"id": "o2", "customer_id": "c2", "items": [{"book_id":"b3","qty":1}],                          "status": "pending"},
      {"id": "o3", "customer_id": "c1", "items": [{"book_id":"b5","qty":3}],                          "status": "paid"}
    ]
  }
});

let j = Jetro::new(doc.clone());
```

`Jetro` holds a thread-local VM; a `collect()` call compiles the expression once (then reuses the bytecode + pointer cache), so rerunning the same or similar query is essentially free.

### 1. Field access

```rust
assert_eq!(j.collect("$.store.name")?,        json!("Nova Books"));
assert_eq!(j.collect("$.store.books[0].author")?, json!("Frank Herbert"));
assert_eq!(j.collect("$.store.books[-1].title")?, json!("Hyperion"));
```

### 2. Filter + map

```rust
let cheap = j.collect("$.store.books.filter(price < 15).map(title)")?;
// ["Dune", "Foundation", "The Hobbit"]
```

### 3. Aggregations

```rust
let in_stock_total = j.collect("$.store.books.filter(stock > 0).sum(price)")?;
let mean_price     = j.collect("$.store.books.map(price).avg()")?;
let well_rated     = j.collect("$.store.books.filter(ratings.avg() >= 4.5).len()")?;
```

### 4. Shaped projections

Build a new object per element:

```rust
let summary = j.collect(r#"
    $.store.books.map({
        title,
        mean_rating: ratings.avg(),
        in_stock:    stock > 0
    })
"#)?;
```

### 5. Sort, group, membership

```rust
let scifi_by_year = j.collect(r#"
    $.store.books
      .filter("sci-fi" in tags)
      .sort_by(year)
      .map({year, title})
"#)?;

let by_first_tag = j.collect("$.store.books.group_by(tags[0])")?;
```

### 6. Comprehensions

```rust
let out_of_stock = j.collect("[b.title for b in $.store.books if b.stock == 0]")?;
// ["Foundation"]

let id_to_title = j.collect("{b.id: b.title for b in $.store.books}")?;
// {"b1": "Dune", "b2": "Foundation", ...}
```

### 7. Let bindings + f-strings

```rust
let headline = j.collect(r#"
    let top = $.store.books.sort_by(ratings.avg()).reverse()[0] in
      f"Top-rated: {top.title} ({top.ratings.avg():.2f})"
"#)?;
// "Top-rated: The Hobbit (4.83)"
```

### 8. Pipelines

`|` passes the left value through the right expression as `@`:

```rust
let avg = j.collect("$.store.books.map(price) | @.avg()")?;
let shout = j.collect("$.store.books[0].title | upper")?;  // "DUNE"
```

### 9. Patch blocks

In-place updates that still compose like any other expression:

```rust
let discounted = j.collect(r#"
    patch $ {
        store.books[*].price:                  @ * 0.9,        // 10% off everything
        store.books[* if stock == 0].available: false,          // mark out-of-stock
        store.books[* if year < 1960].vintage:  true            // annotate vintage
    }
"#)?;
```

Patch-field syntax is `path: value when predicate?`. The path always starts with an identifier and may include `.field`, `[n]`, `[*]`, `[* if pred]`, and `..field` steps. `DELETE` is a sentinel value that removes the matched key.

### 10. Custom methods

Register Rust code as a first-class method:

```rust
use jetro::{Method, MethodRegistry, Val, EvalError, Env};

struct Percentile;
impl Method for Percentile {
    fn name(&self) -> &str { "percentile" }
    fn call(&self, subject: &Val, args: &[Val], _env: &Env) -> Result<Val, EvalError> {
        let xs = subject.as_array().ok_or_else(|| EvalError("not an array".into()))?;
        let p = args.first().and_then(|v| v.as_f64()).unwrap_or(50.0) / 100.0;
        let mut v: Vec<f64> = xs.iter().filter_map(|x| x.as_f64()).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let i = ((v.len() as f64 - 1.0) * p).round() as usize;
        Ok(Val::from(v[i]))
    }
}

let mut reg = MethodRegistry::new();
reg.register(std::sync::Arc::new(Percentile));
let p90 = jetro::query_with("$.store.books.map(price).percentile(90)", &doc, reg.into())?;
```

### 11. Typed `Expr<T>`

Jetro carries a phantom return type so the compiler reminds you what a compiled expression produces:

```rust
let titles:  Expr<Vec<String>> = Expr::new("$.store.books.map(title)")?;
let upper:   Expr<Vec<String>> = Expr::new("@.map(upper)")?;

// `|` composes expressions end-to-end
let pipeline = titles | upper;
let shouted  = pipeline.eval(&doc)?;
```

### 12. VM caching explained

Any query through `j.collect(...)` — or any `VM::run_str(...)` — hits two caches:

- **Program cache**: `Expr → Arc<[Opcode]>`, so each expression string is parsed and compiled exactly once.
- **Pointer cache**: resolved field paths keyed by `(expr_hash, doc_hash)`, so repeated traversals of documents with the same shape short-circuit straight to the leaves.

Rerunning a workload over fresh documents of the same shape is effectively free after the first call. For throwaway one-shot queries use `jetro::query(expr, &doc)` — it skips the VM layer entirely.

### 13. Recursive descent + drill-down chains

`$..field` walks every descendant level. Combine it with filters and projections to collect fields from arbitrarily nested shapes:

```rust
// Every price anywhere in the document, then stats:
let stats = j.collect(r#"
    let prices = $..price.filter(@ kind number) in
      {min: prices.min(), max: prices.max(), sum: prices.sum(), n: prices.len()}
"#)?;

// Every id field alongside the field's path — useful for debug dumps:
let keyed = j.collect(r#"
    $..id.filter(@ kind string).map({id: @, kind: @.slice(0, 1)})
"#)?;
```

### 14. Cross-join via nested comprehensions

Two comprehension generators produce the cartesian product; the `if` filter keeps only matching pairs:

```rust
// Expand every order into its line items, joining book title + customer name:
let receipts = j.collect(r#"
    [
      {
        order:    o.id,
        buyer:    ($.store.customers.filter(id == o.customer_id))[0].name,
        title:    ($.store.books.filter(id == li.book_id))[0].title,
        qty:      li.qty,
        subtotal: ($.store.books.filter(id == li.book_id))[0].price * li.qty
      }
      for o  in $.store.orders
      for li in o.items
      if o.status == "paid"
    ]
"#)?;
```

The same query in hand-written Rust would be three nested loops + three lookups; the comprehension folds it into one compiled program and still hits the pointer cache on repeat calls.

### 15. Layered patch transforms

One `patch` block can stack path-scoped clauses, spread merges, and `when` predicates. Each clause runs on the patched result of the previous one — pipelines inside a single pass:

```rust
let restocked = j.collect(r#"
    let tagged = patch $ {
        store.books[* if stock < 5].reorder:                true,
        store.books[* if tags.includes("classic")].badge:   "Vintage",
        store.books[0].title:                               @.upper()
    }
    in patch tagged {
        store.books[* if stock == 0]: DELETE
    }
"#)?;
```

Two filtering layers: `[* if pred]` in the path filters per element using the element's own fields; `when pred` on a field is evaluated against root `$`. Chain blocks with `let ... in patch ...` when a later mutation needs to see the output of an earlier one — composition is free because `patch` returns a value.

### 16. Compile-time expressions — `jetro!`

Enable the `macros` feature and your expressions get a lexical syntax check at compile time, with errors pointing at the exact macro call-site. The macro returns a typed `Expr<Value>` ready to evaluate or compose:

```rust
use jetro::jetro;

let avg_price = jetro!("$.store.books.map(price).avg()");
let n_classic = jetro!(r#"$.store.books.filter("classic" in tags).len()"#);

assert_eq!(avg_price.eval_raw(&doc)?, json!(14.946));
assert_eq!(n_classic.eval_raw(&doc)?, json!(3));
```

Unbalanced parens, unterminated strings, or empty bodies fail the build — not at runtime.

### 17. `#[derive(JetroSchema)]` — attribute-driven schemas

Pair a type with a fixed set of named expressions. Use the derived constants to configure a bucket, then read results back through serde:

```rust
use jetro::JetroSchema;

#[derive(JetroSchema)]
#[expr(titles       = "$.store.books.map(title)")]
#[expr(mean_rating  = "$.store.books.map(ratings.avg()).avg()")]
#[expr(out_of_stock = "[b.title for b in $.store.books if b.stock == 0]")]
struct StoreView;

let mut b = db.bucket("store");
for (name, src) in StoreView::exprs() {
    b = b.with(*name, *src);
}
let bucket = b.open()?;
bucket.insert("nova", &doc["store"])?;

let titles:       Vec<String> = bucket.get("nova", "titles")?.unwrap();
let mean_rating:  f64         = bucket.get("nova", "mean_rating")?.unwrap();
let out_of_stock: Vec<String> = bucket.get("nova", "out_of_stock")?.unwrap();
```

Each `#[expr(...)]` line is also lex-checked at compile time — so an invalid expression in a schema never ships.

---

## Persistence — `jetro::db`

Optional embedded storage layer backed by a B+ tree over memory-mapped files (`memmap2`). Lock-free reads, COW writes. `use jetro::prelude::*` pulls in every type below.

### `Database::memory()` — zero-config sandbox

```rust
use jetro::prelude::*;

let db = Database::memory()?;    // unique temp dir, deleted on drop
```

Swap for `Database::open("path/")` when you want durability.

### Fluent buckets (expressions + documents)

`Bucket` bundles expressions with the documents they derive from. Expressions are registered up front; every insert computes and persists them atomically, and the typed `get<T>` accessor rehydrates the results via `serde`:

```rust
let books = db.bucket("books")
    .with("titles",       "$.books.map(title)")
    .with("sci_fi_count", r#"$.books.filter("sci-fi" in tags).len()"#)
    .with("mean_price",   "$.books.map(price).avg()")
    .open()?;

books.insert("catalog", &doc["store"])?;

let titles: Vec<String> = books.get("catalog", "titles")?;
let count:  usize       = books.get("catalog", "sci_fi_count")?;
let mean:   f64         = books.get("catalog", "mean_price")?;
```

Add another derived view after the fact and rebuild in bulk:

```rust
books.add_expr("classics", r#"$.books.filter("classic" in tags).map(title)"#)?;
books.rebuild("classics")?;     // apply to every stored doc in one pass
```

### `DocIter` — iterator combinators

The `iter_docs()` result is a plain `Vec<(String, Value)>`, but the `DocIter` trait lights up chainable expression-driven operators over it:

```rust
use jetro::db::DocIter;

let expensive_titles: Vec<String> = books.iter_docs()?
    .filter_expr("$.books.map(price) | @.max() > 15")?
    .map_expr_as("$.store.name")?;
```

### `GraphBucket` — cross-document queries

Hold related entities in separate, indexed node trees and query across them:

```rust
let exprs = db.expr_bucket("exprs")?;
let graph = db.graph_bucket("shop", &exprs)?;
graph.add_node("customers")?;
graph.add_node("orders")?;
graph.add_index("customers", "id")?;
graph.add_index("orders",    "customer_id")?;

for c in doc["store"]["customers"].as_array().unwrap() {
    graph.insert("customers", c)?;
}
for o in doc["store"]["orders"].as_array().unwrap() {
    graph.insert("orders", o)?;
}

// Join a single order against the customer tree by indexed field:
let joined = graph.build()
    .inline("orders",    json!([doc["store"]["orders"][0]]))
    .lookup("customers", "id", &json!("c1"))
    .query(r#"{customer: $.customers[0].name, order: $.orders[0].id}"#)?;
```

`preload_hot("customers")` pins reference data in memory for the duration of the handle — cheap repeated joins against rarely-changing tables.

### `Join` — blocking stream-join

Upstream services push pieces of the same logical event at different times. `Join` registers which `kinds` of documents make up a complete record, then blocks `get`/`run` calls until every kind has arrived for a given id:

```rust
let events = db.join("events")
    .id("order_id")
    .kinds(["order", "payment", "shipment"])
    .open()?;

// Producers can arrive in any order:
events.emit("order",    &json!({"order_id": "A1", "item": "Dune"}))?;
events.emit("payment",  &json!({"order_id": "A1", "amount": 12.99}))?;
events.emit("shipment", &json!({"order_id": "A1", "tracking": "TRK-1"}))?;

// Consumers block until all three kinds are present:
let bill = events.on("A1").run(r#"
    {item: $.order.item, paid: $.payment.amount, track: $.shipment.tracking}
"#)?;
```

`on(id).timeout(Duration::from_secs(5))` adds a bounded wait; `peek`, `arrived`, and `partial` are non-blocking inspectors.

### Storage abstraction

Under the hood, every bucket talks to a `Tree` — the byte-oriented ordered KV surface. The default is `BTree` (mmap-backed, persistent). `BTreeMem` provides the same interface over a `BTreeMap` for tests, and `FileStorage` / `MemStorage` open named trees from a directory or an in-process map:

```rust
use jetro::prelude::*;

let st = MemStorage::new();
let t  = st.open_tree("log")?;
t.insert(b"k", b"v")?;
```

### `Engine` — shared VM across threads

`Jetro` gives each thread its own VM. `Engine` is a `Send + Sync` handle around a `Mutex<VM>` so one warm cache can service many worker tasks:

```rust
use jetro::prelude::*;
use std::sync::Arc;

let engine = Engine::new();                          // Arc<Engine>
let shared = Arc::clone(&engine);

std::thread::spawn(move || {
    // Compile cache + pointer cache are shared with the spawner.
    shared.run("$.store.books.len()", &doc).unwrap();
}).join().unwrap();
```

The first call compiles and fills both caches; every subsequent thread hitting the same expression skips straight to execution.

### `Session` — one handle for engine + storage + catalog

The 2.0 façade: bundle a shared `Engine`, a `Storage` backend, and a named-expression `Catalog` into one type. In-memory default takes zero config; swap any piece for durability or sharing:

```rust
use jetro::prelude::*;

let s = Session::in_memory();
s.register_expr("avg_price", "$.store.books.map(price).avg()")?;
let v = s.run_named("avg_price", &doc)?;

// Or mix concrete backends explicitly:
let exprs  = db.expr_bucket("shared")?;        // on-disk Catalog
let custom = Session::builder()
    .engine(Engine::with_capacity(1024, 8192))
    .storage(Arc::new(FileStorage::new(db.path())?))
    .catalog(exprs)
    .build();
```

The `Catalog` trait is implemented by `ExprBucket` (on-disk) and `MemCatalog` (in-process) — so the same `Session` flow works whether you want persistence or not.

### Async — `AsyncJoin` (feature = "async")

Enable the `async` feature and `Join` grows an async sibling. Writes and blocking waits are off-loaded to `tokio::task::spawn_blocking` so the reactor never stalls on a `fsync` or a condvar park:

```rust
use jetro::prelude::*;
use std::time::Duration;

#[tokio::main]
async fn main() -> jetro::Result<()> {
    let db = Database::memory()?;
    let j = db.join("events")
        .id("order_id")
        .kinds(["order", "payment", "shipment"])
        .open()?
        .into_async();

    j.emit("order",    &json!({"order_id": "A1", "item": "Dune"})).await?;
    j.emit("payment",  &json!({"order_id": "A1", "amount": 12.99})).await?;
    j.emit("shipment", &json!({"order_id": "A1", "tracking": "TRK-1"})).await?;

    // Bounded wait — returns Ok(None) on timeout, Ok(Some(_)) on arrival.
    let bill = j.wait_for("A1", Duration::from_secs(5)).await?;
    Ok(())
}
```

`AsyncJoin::run` / `get` evaluate jetro expressions over the joined doc once it completes; `arrived` and `peek` are cheap inspectors.

### Raw `BTree`

```rust
let tree = BTree::open("path/to/file")?;
tree.insert(b"key", b"value")?;
let range = tree.range(b"a", b"z")?;
```

See `examples/` for runnable end-to-end programs.

### Feature flags

| Feature | Pulls in | Unlocks |
|---------|----------|---------|
| `macros` | `jetro-macros` companion crate | `jetro!(...)` macro, `#[derive(JetroSchema)]` |
| `async`  | `tokio` (`sync`, `rt`, `time`)  | `AsyncJoin`, `Join::into_async` |

`Cargo.toml`:

```toml
[dependencies]
jetro = { version = "0.2", features = ["macros", "async"] }
```

---

## Syntax

### Root and context

| Token | Meaning |
|-------|---------|
| `$` | Document root |
| `@` | Current item (inside lambdas, pipes, comprehensions, patch paths) |

### Navigation

```
$.field                // child field
$.a.b.c                // nested
$.user?.name           // null-safe field
$.items[0]             // array index
$.items[-1]            // last
$.items[2:5]           // slice [2, 5)
$.items[2:]            // tail
$.items[:5]            // head
$..title               // recursive descent
```

### Literals

```
null  true  false
42    3.14
"hello"  'world'
f"Hello {$.user.name}!"     // f-string with interpolation
f"{$.price:.2f}"            // format spec
f"{$.name | upper}"         // pipe transform
```

### Operators

```
==  !=  <  <=  >  >=         // comparison
~=                           // fuzzy match (case-insensitive substring)
+  -  *  /  %                // arithmetic
and  or  not                 // logical
$.field ?| "default"         // null coalescing
```

### Methods

All operations are dot-called methods:

```
$.books.filter(price > 10)
$.books.map(title)
$.books.sum(price)
$.books.filter(price > 10).count()
$.books.sort_by(price).reverse()
$.users.group_by(tier)
$.items.map({name, total: price * qty})
```

### Comprehensions

```
[book.title for book in $.books if book.price > 10]
{user.id: user.name for user in $.users if user.active}
```

### Let bindings

```
let top = $.books.filter(price > 100) in
  {count: top.len(), titles: top.map(title)}
```

### Lambdas

```
$.books.filter(lambda b: b.tags.includes("sci-fi"))
```

### Kind checks

```
$.items.filter(price kind number and price > 0)
$.data.filter(deleted_at kind null)
```

### Pipelines

`|` passes the left value through the right expression as `@`:

```
$.name | upper
$.price | (@ * 1.1)
```

### Patch blocks

In-place updates with a compact block-form syntax. Each clause matches a path inside the document and produces either a replacement value or `DELETE`:

```
patch $ {
    books[*].price:             @ * 1.1,         // 10% markup on every book
    books[0].title:             "Dune (1965)",   // replace one field
    books[*].draft:             DELETE,          // drop a field
    books[* if price > 50]:     DELETE           // conditional row delete
}
```

Field form: `path: value when predicate?`. Path must begin with an identifier and may use `.field`, `[n]`, `[*]`, `[* if pred]`, `..field` steps. `@` inside the value is the current matched node.

Patch results compose like any other expression — use them in `map`, `let` bindings, pipes, object literals, etc.

See `src/SYNTAX.md` for the full syntax reference.

---

## Architecture

All code lives under `src/`.

### Expression engine

- `grammar.pest` + `parser.rs` — PEG grammar; produces an `Expr` AST (`ast.rs`)
- `eval/mod.rs` — tree-walking evaluator. Reference implementation; `query()` dispatches here
- `eval/value.rs` — `Val` type with `Arc`-wrapped compound nodes; every clone is O(1)
- `eval/func_*.rs` — built-in method implementations (strings, arrays, objects, paths, aggregates, csv)
- `eval/methods.rs` — `Method` trait + `MethodRegistry` for user-registered custom methods
- `vm.rs` — bytecode compiler + stack machine; peephole passes (fusion, constant folding, strength reduction); caches compiled programs and resolved pointer paths
- `engine.rs` — `Engine` shared `Send + Sync` handle over the VM
- `session.rs` — `Session` + `Catalog` trait + `MemCatalog`; one top-level handle tying Engine, Storage, and named expressions
- `graph.rs` — multi-document queries via virtual root merging
- `analysis.rs` / `schema.rs` / `plan.rs` / `cfg.rs` / `ssa.rs` — optional IR layers (type/nullness/cardinality, shape inference, logical plan, CFG, SSA)

### Persistence

- `db/btree.rs` — B+ tree with COW writes and lock-free reads (bbolt-inspired)
- `db/storage.rs` — `Tree` + `Storage` traits, `FileStorage`, `BTreeMem`, `MemStorage`
- `db/bucket.rs` — `ExprBucket`, `JsonBucket`
- `db/fluent.rs` — fluent `Bucket` builder over `ExprBucket` + `JsonBucket`
- `db/graph_bucket.rs` — cross-document queries with secondary indexes + hot cache
- `db/join.rs` — blocking stream-join bucket
- `db/async_join.rs` — tokio-based `AsyncJoin` (feature = "async")
- `db/mod.rs` — `Database` facade

### Companion crates

- `jetro-macros/` — proc-macro crate exposing `jetro!` and `#[derive(JetroSchema)]` (feature = "macros")

---

## License

MIT — see `LICENSE`.
