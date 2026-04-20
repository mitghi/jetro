# jetro

[<img src="https://img.shields.io/badge/docs-jetro-blue"></img>](https://docs.rs/jetro)
[<img src="https://img.shields.io/badge/try-online%20repl-brightgreen"></img>](https://jetro.io)
![GitHub](https://img.shields.io/github/license/mitghi/jetro)

Jetro is a library for querying, transforming, and comparing JSON. It provides a compact expression language, a bytecode VM with caching, and an optional persistence layer with expression-driven indexes.

Jetro has minimal dependencies and operates on top of [serde_json](https://serde.rs).

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
patch($.books) {
    [*].price        => @ * 1.1,           // 10% markup on every book
    [0].title        => "Dune (1965)",     // replace one field
    [*].draft        => DELETE,             // drop a field
    when price > 50  => DELETE              // conditional
}
```

Patch results compose like any other expression — use them in `map`, `let` bindings, pipes, object literals, etc.

See `src/SYNTAX.md` for the full syntax reference.

---

## Persistence — `jetro::db`

Optional embedded storage layer backed by a B+ tree over memory-mapped files (`memmap2`). Lock-free reads, COW writes. A handful of composable buckets:

### ExprBucket + JsonBucket

Named expressions + JSON documents with automatically derived results:

```rust
use jetro::db::Database;
use serde_json::json;

let db = Database::open("mydb")?;
let exprs = db.expr_bucket("main")?;
exprs.put("book_titles", "$.store.books.map(title)")?;
exprs.put("book_count",  "$.store.books.len()")?;

let docs = db.json_bucket("library", &["book_titles", "book_count"], &exprs)?;
docs.insert("catalog", &catalog_json)?;

// Results are pre-computed at insert time and indexed by (doc_key, expr_key)
let titles = docs.get_result("catalog", "book_titles")?.unwrap();
```

### GraphBucket

Cross-document queries with secondary indexes and a hot cache for reference data:

```rust
use jetro::db::{Database, GraphNode};
use serde_json::json;

let db = Database::open("mydb")?;
let exprs = db.expr_bucket("exprs")?;
let graph = db.graph_bucket("analytics", &exprs)?;

graph.add_node("orders")?;
graph.add_node("customers")?;
graph.add_index("orders",    "customer_id")?;
graph.add_index("customers", "id")?;
graph.preload_hot("customers")?;  // reference data pinned in memory

// Stream-join: inline incoming event + indexed lookup
let summary = graph.query(&[
    GraphNode::Inline  { node: "orders",    value: json!([new_order]) },
    GraphNode::ByField { node: "customers", field: "id", value: &cust_id },
], r#"{customer: $.customers[0].name, total: $.orders[0].total}"#)?;
```

### LinkBucket

Stream-join / blocking bucket — a "link" is complete when one document of every registered `kind` has arrived with the same id. `get`/`query` block until the link completes.

```rust
let lb = db.link_bucket("orders", &["order", "payment", "shipment"], ...)?;

// Producers can arrive in any order; consumers block until all three
// kinds are present for a given id.
lb.insert("order",    &json!({"order_id": "A1", "item": "Chair"}))?;
lb.insert("payment",  &json!({"order_id": "A1", "amount": 99.0}))?;
lb.insert("shipment", &json!({"order_id": "A1", "tracking": "TRK-1"}))?;

let joined = lb.query(
    &json!("A1"),
    r#"{item: $.order.item, paid: $.payment.amount, track: $.shipment.tracking}"#,
)?;
```

### BTree (raw)

The storage primitive is public — use it directly for plain key/value workloads:

```rust
use jetro::db::BTree;

let tree = BTree::open("path/to/file")?;
tree.insert(b"key", b"value")?;
let range = tree.range(b"a", b"z")?;
```

See `examples/` for runnable end-to-end programs.

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
- `graph.rs` — multi-document queries via virtual root merging
- `analysis.rs` / `schema.rs` / `plan.rs` / `cfg.rs` / `ssa.rs` — optional IR layers (type/nullness/cardinality, shape inference, logical plan, CFG, SSA)

### Persistence

- `db/btree.rs` — B+ tree with COW writes and lock-free reads (bbolt-inspired)
- `db/bucket.rs` — `ExprBucket`, `JsonBucket`
- `db/graph_bucket.rs` — cross-document queries with secondary indexes + hot cache
- `db/link_bucket.rs` — blocking stream-join bucket
- `db/mod.rs` — `Database` facade

---

## License

MIT — see `LICENSE`.
