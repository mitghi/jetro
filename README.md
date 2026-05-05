# jetro

[![crates.io](https://img.shields.io/crates/v/jetro.svg)](https://crates.io/crates/jetro)
[![docs.rs](https://img.shields.io/badge/docs-jetro-blue)](https://docs.rs/jetro)
[![license](https://img.shields.io/crates/l/jetro.svg)](LICENSE)

**Fast JSON querying and shaping language**

Jetro is a compact expression engine for JSON. It accepts JSON bytes, evaluates
a Jetro expression, and returns a `serde_json::Value`.

```rust
use jetro::Jetro;
use serde_json::json;

let data = br#"
{
  "orders": [
    {
      "id": "ord_1001",
      "status": "paid",
      "total": 184.50,
      "customer": {"name": "Ada", "tier": "gold"}
    },
    {
      "id": "ord_1002",
      "status": "refunded",
      "total": 42.00,
      "customer": {"name": "Grace", "tier": "silver"}
    },
    {
      "id": "ord_1003",
      "status": "paid",
      "total": 312.20,
      "customer": {"name": "Alan", "tier": "gold"}
    }
  ]
}
"#;

let jetro = Jetro::from_bytes(data.to_vec())?;

let report = jetro.collect(r#"
{
  "top_paid": $.orders
    .filter(status == "paid")
    .sort_by(-total)
    .take(2)
    .map({
      id,
      customer: customer.name,
      tier: customer.tier,
      total,
      label: f"{customer.name}: ${total}"
    }),

  "paid_total": $.orders
    .filter(status == "paid")
    .map(total)
    .sum()
}
"#)?;

assert_eq!(report, json!({
  "top_paid": [
    {
      "id": "ord_1003",
      "customer": "Alan",
      "tier": "gold",
      "total": 312.20,
      "label": "Alan: $312.2"
    },
    {
      "id": "ord_1001",
      "customer": "Ada",
      "tier": "gold",
      "total": 184.50,
      "label": "Ada: $184.5"
    }
  ],
  "paid_total": 496.70
}));

# Ok::<(), Box<dyn std::error::Error>>(())
```

## Why Jetro?

- **Byte-first API**  
  The public API is intentionally small: `Jetro::from_bytes(bytes)` and
  `collect(expr)`. With default features, Jetro uses
  [`simd-json`](https://github.com/simd-lite/simd-json).

- **Query and shape in one expression**  
  Return scalars, arrays, objects, nested shapes, filters, projections,
  aggregates, and patches from the same expression language.

- **Demand-aware execution**  
  Jetro avoids adding hand-written fused functions for every useful chain.
  Builtins expose metadata about streaming, ordering, barriers, and demand, so
  the planner can compose optimizations algorithmically.

  ```text
  $.orders
    .filter(status == "paid")
    .sort_by(-total)
    .take_while(customer.tier == "gold")
    .take(10)
    .map({id, customer: customer.name, total})
  ```

- **Lazy materialization**  
  Raw bytes, tape data, structural indexes, and owned value trees are built only
  when a query needs them.

- **Optimized execution paths**  
  Eligible queries can run through structural indexes, borrowed value views,
  streaming pipelines, columnar paths, or the VM fallback.

## Install

```toml
[dependencies]
jetro = "0.4"
```

## API

```rust
use jetro::Jetro;

let jetro = Jetro::from_bytes(json_bytes)?;
let value = jetro.collect("$.some.expression")?;
```

That is the stable top-level API.

## Language Preview

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

Full syntax reference: [jetro-core/src/SYNTAX.md](jetro-core/src/SYNTAX.md)

## Examples

### Pick fields

```rust
let result = jetro.collect("$.books.map({title, price})")?;
```

### Filter, sort, and limit

```rust
let result = jetro.collect("$.books.filter(price > 10).sort_by(-price).take(5)")?;
```

### Build a response object

```rust
let result = jetro.collect(r#"
{
  "titles": $.books.map(title),
  "max_price": $.books.map(price).max(),
  "software": $.books.filter(tags.includes("software")).count()
}
"#)?;
```

### Deep search

```rust
let result = jetro.collect("$..price")?;
```

### Patch-style update

```rust
let result = jetro.collect("$.user.name.set('Ada')")?;
```

## Execution Model

Jetro lowers expressions into a physical plan before execution.

```text
expression
  -> parser
  -> planner
  -> physical plan
  -> structural index / value-view pipeline / streaming pipeline / VM fallback
  -> serde_json::Value
```

Optimized paths are implementation details. They preserve the same language
semantics as the fallback executor.

## CLI

For interactive use, see [`jetrocli`](https://github.com/mitghi/jetrocli).

## Learn More

- [INDEPTH.md](INDEPTH.md) - API and language examples
- [jetro-core/src/SYNTAX.md](jetro-core/src/SYNTAX.md) - syntax reference
- [SAFETY.md](SAFETY.md) - unsafe inventory and safety notes
- [CHANGELOG.md](CHANGELOG.md) - release notes

## License

MIT
