# jetro — in-depth guide

Long-form walk through the Rust API, grammar, and internals. Start at the
[README](README.md) for the high-level pitch; come here for the full
reference.

---

## Quick start

```rust
use jetro::Jetro;
use serde_json::json;

let j = Jetro::from_bytes(br#"{
  "store": {
    "books": [
      {"title": "Dune",       "price": 12.99},
      {"title": "Foundation", "price":  9.99}
    ]
  }
}"#.to_vec()).unwrap();

let titles = j.collect("$.store.books.filter(price > 10).map(title)").unwrap();
assert_eq!(titles, json!(["Dune"]));
```

---

## Rust API tour

Progressive walk through the Rust API over one rich payload. Every snippet runs against this document:

```rust
use jetro::Jetro;
use serde_json::json;

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

let j = Jetro::from_bytes(serde_json::to_vec(&doc).unwrap()).unwrap();
```

`Jetro` holds a thread-local VM; `collect()` compiles each unique expression once (then reuses the bytecode + pointer cache), so rerunning the same or similar query is essentially free.

### 1. Field access

```rust
assert_eq!(j.collect("$.store.name")?,              json!("Nova Books"));
assert_eq!(j.collect("$.store.books[0].author")?,   json!("Frank Herbert"));
assert_eq!(j.collect("$.store.books[-1].title")?,   json!("Hyperion"));
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
let avg   = j.collect("$.store.books.map(price) | @.avg()")?;
let shout = j.collect("$.store.books[0].title | upper")?;  // "DUNE"
```

### 9. Search — shallow and deep

Shallow search / transform helpers (v2.2):

```rust
// .find / .find_all — aliases of .filter
let hit     = j.collect(r#"$.store.books.find(title == "Dune")"#)?;
let classic = j.collect(r#"$.store.books.find_all("classic" in tags)"#)?;

// .pick — field list with optional rename
let cards = j.collect(r#"
    $.store.books.pick(title, slug: id, year)
"#)?;

// .unique_by — dedup by derived key
let first_per_author = j.collect(r#"
    $.store.books.sort_by(year).unique_by(author)
"#)?;

// .collect — scalar → [scalar], array → identity, null → []
let always_array = j.collect("$.store.books[0].tags.collect()")?;
```

Deep search walks every descendant (DFS pre-order):

```rust
// $..find(pred) — every descendant satisfying pred
let cheap_everywhere = j.collect("$..find(@ kind number and @ < 10)")?;

// $..shape({k, …}) — every object that has all listed keys
let books_like = j.collect("$..shape({id, title, price})")?;

// $..like({k: v}) — every object whose listed keys equal the literals
let paid_orders = j.collect(r#"$..like({status: "paid"})"#)?;
```

`.deep_find` / `.deep_shape` / `.deep_like` are the method-form aliases.

### 10. Chain-style writes

Rooted `$.<path>.<op>(...)` chains desugar into a single `patch` block — terse mutation without leaving the query language:

```rust
// .set — replace value at path
let bumped = j.collect(r#"$.store.books[0].price.set(19.99)"#)?;

// .modify — rewrite using @ bound to current leaf
let discounted = j.collect(r#"$.store.books[*].price.modify(@ * 0.9)"#)?;

// .delete — remove the leaf
let pruned = j.collect(r#"$.store.books[* if stock == 0].delete()"#)?;

// .unset(key) — drop a child of the leaf object
let anon = j.collect(r#"$.store.customers[*].unset(credits)"#)?;
```

Breaking change vs v1: `$.field.set(v)` now returns the full doc with the value written back (old behaviour: ignore receiver, return `v`). Pipe form preserves v1 semantics — `$.field | set(v)` returns `v` — so inside `.map(...)` you can still use the pipe form.

### 11. Conditional — Python-style ternary

```rust
let label = j.collect(r#"
    $.store.books.map(
        "out" if stock == 0
        else "low" if stock < 5
        else "ok"
    )
"#)?;
// ["ok", "out", "ok", "ok", "ok"]
```

Right-associative — chains naturally without parens. Short-circuits: only the taken branch runs. A literal `true` / `false` condition folds at compile time so `expensive_expr() if false else cheap_expr()` never compiles the dead branch.

### 12. Patch blocks

In-place updates that compose like any other expression:

```rust
let discounted = j.collect(r#"
    patch $ {
        store.books[*].price:                  @ * 0.9,
        store.books[* if stock == 0].available: false,
        store.books[* if year < 1960].vintage:  true
    }
"#)?;
```

Patch-field syntax: `path: value when predicate?`. Paths start with an identifier and may include `.field`, `[n]`, `[*]`, `[* if pred]`, and `..field`. `DELETE` removes the matched key.

### 13. VM caching

Every query through `Jetro::collect` hits the execution caches:

- **Program cache**: `expr string → Arc<[Opcode]>`. Each unique expression is parsed + compiled exactly once.
- **Pointer cache**: resolved field paths keyed by `(program_id, doc_hash)`. Structural queries short-circuit straight to the leaves on repeat calls.

Rerunning a workload over fresh documents of the same shape is effectively free after the first call.

### 16. Recursive descent + drill-down chains

`$..field` walks every descendant level. Combine with filters and projections:

```rust
let stats = j.collect(r#"
    let prices = $..price.filter(@ kind number) in
      {min: prices.min(), max: prices.max(), sum: prices.sum(), n: prices.len()}
"#)?;

let keyed = j.collect(r#"
    $..id.filter(@ kind string).map({id: @, kind: @.slice(0, 1)})
"#)?;
```

### 17. Cross-join via nested comprehensions

Two generators produce the cartesian product; the `if` filter keeps only matching pairs:

```rust
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

Three nested loops + three lookups folded into one compiled program. Hits the pointer cache on repeat calls.

### 18. Layered patch transforms

One `patch` block stacks path-scoped clauses. Each clause runs on the patched result of the previous one:

```rust
let restocked = j.collect(r#"
    let tagged = patch $ {
        store.books[* if stock < 5].reorder:              true,
        store.books[* if tags.includes("classic")].badge: "Vintage",
        store.books[0].title:                             @.upper()
    }
    in patch tagged {
        store.books[* if stock == 0]: DELETE
    }
"#)?;
```

Two filter layers: `[* if pred]` filters per element against the element's own fields; `when pred` on a field is evaluated against root `$`. Chain blocks with `let ... in patch ...` when a later mutation needs to see earlier output.

---

## Syntax

### Root and context

| Token | Meaning |
|-------|---------|
| `$`   | Document root |
| `@`   | Current item (lambdas, pipes, comprehensions, patch paths) |

### Navigation

```
$.field                // child field
$.a.b.c                // nested
$.user?.name           // null-safe (postfix `?` on .user, not prefix `?.`)
$.items[0]             // array index
$.items[-1]            // last
$.items[2:5]           // slice [2, 5)
$.items[2:]            // tail
$.items[:5]            // head
$..title               // recursive descent
$..services?.first()   // descendant + null-safe + explicit first
```

Postfix `?` is null-safety only. It does not take the first element of
an array — for that use `.first()` explicitly. `!` keeps its
exactly-one-element meaning (errors on 0 or >1).

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

### Search — shallow and deep

```
$.books.find(title == "Dune")            // first match (alias of filter + [0])
$.books.find_all("classic" in tags)      // alias of filter
$.books.pick(title, slug: id, year)      // project + rename
$.books.unique_by(author)                // dedup by key
$.books[0].tags.collect()                // scalar→[scalar], arr→arr, null→[]

$..find(@.price > 10)                    // deep: every descendant satisfying pred
$..shape({id, title, price})             // deep: every obj with all listed keys
$..like({status: "paid"})                // deep: every obj with listed key==lit
```

### Chain-style writes (terminals on rooted paths)

```
$.a.b.set(v)          // replace leaf — returns full doc
$.a.b.modify(@ * 2)   // replace using @ = current leaf
$.a.b.delete()        // remove leaf
$.a.b.unset(k)        // remove child of leaf object
```

### Conditional (Python-style ternary)

```
then_ if cond else else_
"big" if n > 10 else "small"
"a" if n == 1 else "b" if n == 2 else "c"   // right-associative
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

In-place updates with block-form syntax. Each clause matches a path and produces either a replacement value or `DELETE`:

```
patch $ {
    books[*].price:             @ * 1.1,
    books[* if stock == 0]:     DELETE,
    meta.updated_at:            now()
}
```

See [`jetro-core/README.md`](../jetro-core/README.md) for a full technical walkthrough: AST variants, opcode set, peephole passes, cache invariants.

### Companion crates

- [`jetro-core`](../jetro-core) — implementation crate

---

## License

MIT. See [LICENSE](LICENSE).
