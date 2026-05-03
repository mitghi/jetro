# jetro

[![crates.io](https://img.shields.io/crates/v/jetro.svg)](https://crates.io/crates/jetro)
[![docs.rs](https://img.shields.io/badge/docs-jetro-blue)](https://docs.rs/jetro)
[![license](https://img.shields.io/crates/l/jetro.svg)](LICENSE)

**Query, transform, and patch JSON with a language that fits
on a postcard.**

Jetro compiles a compact expression language to a caching bytecode VM backed
by columnar value lanes.

```rust
use jetro::Jetro;
use serde_json::json;

let j = Jetro::new(json!({
    "store": { "books": [
        {"title": "Dune",       "price": 12.99},
        {"title": "Foundation", "price":  9.99},
    ]}
}));

let titles = j.collect("$.store.books.filter(price > 10).map(title)")?;
assert_eq!(titles, json!(["Dune"]));
# Ok::<(), jetro::EvalError>(())
```

---

## Why jetro

- **One language covers query, transform, patch, and diff.**
  `$.orders.filter(status == 'paid').map({id, total})`, chain-style writes
  `$.user.name.set('Ada')`, full `patch` blocks, and deep search
  `$..find(@.isbn == '978…')` all share the same grammar.

- **Python-style list/dict/set comprehensions & lambdas.** Transform
  pipelines read like Python, compile to tight bytecode:
  ```
  [o.id for o in $.orders if o.total > 500]
  {c.id: c.tier for c in $.customers}
  $.items.map(lambda x: x.price * 1.1)
  $.items.map(x => x.price * 1.1)        // arrow form
  ```

- **Bytecode VM with peephole fusion.** Common shapes —
  `map(f).sum()`, `filter(p).first()`, `map({a, b})`,
  `map(@.split(s).map(len).sum())`, deep `$..find` — collapse to single
  opcodes; repeated queries hit a compile cache.

- **Columnar value lanes.** `IntVec` / `FloatVec` / `StrVec` /
  `StrSliceVec` / `ObjVec` run homogeneous arrays as packed vectors,
  not tagged unions — tight loops beat enum dispatch.

- **SIMD byte-scan for deep-search queries.** `$..id`, `$..k == lit`, chained
  `$..a..b..c` — memchr-accelerated scans over raw document bytes
  when it beats walking the tree.

- **Zero-copy string views.** `slice`, `substring`, `split.first` and
  friends return borrowed slices into the parent `Arc<str>` — no
  per-row allocation on the hot path.

- **Safety.** A tree-walker reference implementation runs
  every query the VM does; 540+ unit tests, Miri-audited `unsafe`
  invariants ([SAFETY.md](SAFETY.md)). Every optimization is a
  specialization on the same semantics.

---

## The language on a postcard

| | |
|---|---|
| `$` root &nbsp;&nbsp; `@` current | `.f`, `.f?` field / null-safe |
| `..k`, `..k?` recursive descent | `[i]`, `[a:b]`, `[*]` index / slice / all |
| `.filter(p)`, `.map(f)` | `.sort_by(k)`, `.group_by(k)` |
| `.find(p)`, `.unique_by(k)` | `.count_by(k)`, `.index_by(k)` |
| `$..find(@.k == v)` | `$..shape({name, price})`, `$..like({…})` |
| `[x for x in xs if pred]` | `{k(x): v(x) for x in xs}` |
| `lambda x: …` &nbsp; `x => …` | `let a = expr in …` |
| `.set(v)`, `.modify(@ * 2)`, `.delete()` | `patch $ { .a.b: 1, .c: DELETE }` |
| `f"hello {$.name}"` | `x when cond else y` |

Full syntax reference: [jetro-core/src/SYNTAX.md](jetro-core/src/SYNTAX.md)

---

## Install

```toml
[dependencies]
jetro = "0.4"
```

One-shot without the VM:

```rust
let result = jetro::query("$.store.books.len()", &doc)?;
```

Give [Jetro CLI](https://github.com/mitghi/jetrocli) a try for an interactive experience.

---

## Learn more

- [**INDEPTH.md**](INDEPTH.md) — complete Rust API tour with worked examples.
- [**jetro-core/src/SYNTAX.md**](jetro-core/src/SYNTAX.md) — language reference.
- [**SAFETY.md**](SAFETY.md) — `unsafe` inventory and Miri audit.
- [**CHANGELOG.md**](CHANGELOG.md) — release notes.

---

## License

MIT.
