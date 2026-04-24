# jetro

[![crates.io](https://img.shields.io/crates/v/jetro.svg)](https://crates.io/crates/jetro)
[![docs.rs](https://img.shields.io/badge/docs-jetro-blue)](https://docs.rs/jetro)
[![license](https://img.shields.io/crates/l/jetro.svg)](LICENSE)

**Query, transform, and patch JSON at native speed — with a language that fits
on a postcard.**

Jetro compiles a compact expression language to a caching bytecode VM backed
by columnar value lanes. On most realistic workloads it runs **within 1–2×
of hand-written Rust + `serde_json`**, and on several specialized ones it's
**faster than native**.

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

- **SIMD byte-scan for deep queries.** `$..id`, `$..k == lit`, chained
  `$..a..b..c` — memchr-accelerated scans over raw document bytes
  when it beats walking the tree.

- **Zero-copy string views.** `slice`, `substring`, `split.first` and
  friends return borrowed slices into the parent `Arc<str>` — no
  per-row allocation on the hot path.

- **Safe and honest.** A tree-walker reference implementation runs
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

---

## Benchmarks

**vs hand-written Rust + `serde_json`** — ratio is `jetro_ms / native_ms`, lower is better, sub‑1.0× means jetro wins.

| workload | vs native Rust |
|---|---|
| `map(@.split('-').map(len).sum())` | **0.51×** — 2× faster |
| `map('prefix-' + @ + '-suffix')` | **0.75×** |
| `map(@.split(s).count())` | **0.77×** |
| `count_by(grp)` | **0.85×** |
| `map({id, grp})` projection | 2.49× |
| `filter(status == 'ok')` | 1.69× |
| `map(@.slice(10, 30))` | 2.34× |
| `map(f"{@.id}_{@.grp}")` | 1.59× |
| `map(@.upper().replace('FOO', 'BAR'))` | 1.13× |

**vs another Rust-based `jq`-style query engine.** Across 13 representative
queries — projection, filter, deep find, group-by, nested aggregates — jetro
is **4×–70× faster**. The biggest gaps show up on deep-search and
reduction workloads (`$..k.sum()`, multi-predicate `$..find`, `map(field).sum()`):
jetro's fusion passes collapse these to single opcodes, while interpreted
`jq`-style engines re-walk the tree per pipeline stage.

**vs idiomatic Go (`encoding/json` + `strings`).** Jetro **matches or beats
the Go standard library** on string-heavy workloads — for example, on
`upper + trim` jetro is about 1.3× faster than Go, on `split + reverse +
join` roughly 4× faster, and on `replace_all` about 3.8× faster. Parity or
better on the simple scans, with the string-method fusions pulling ahead.

542 tests pass on every release. See [CHANGELOG.md](CHANGELOG.md) for the
full v0.4.0 bench delta.

---

## Learn more

- [**INDEPTH.md**](INDEPTH.md) — complete Rust API tour with worked examples.
- [**jetro-core/src/SYNTAX.md**](jetro-core/src/SYNTAX.md) — language reference.
- [**SAFETY.md**](SAFETY.md) — `unsafe` inventory and Miri audit.
- [**CHANGELOG.md**](CHANGELOG.md) — release notes.

---

## License

MIT.
