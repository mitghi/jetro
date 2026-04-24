# jetro

[![crates.io](https://img.shields.io/crates/v/jetro.svg)](https://crates.io/crates/jetro)
[![docs.rs](https://img.shields.io/badge/docs-jetro-blue)](https://docs.rs/jetro)
[![license](https://img.shields.io/crates/l/jetro.svg)](LICENSE)

**A fast JSON query, transform, and comparison engine for Rust.**

Jetro compiles a compact expression language to a caching bytecode VM and
columnar `Val` representation. On most realistic workloads it runs
**within 1–2× of hand-written Rust + serde_json**, and on several
specialized ones — `concat`, `split.count`, `split.map(len).sum`,
`group_by short`, plus the full opt_x_2 numeric suite — it's **faster
than native**.

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

- **One expression language covers query, transform, patch, and diff.** `$.orders.filter(status == 'paid').map({id, total})`, `$.users[*].email?.lower()`, chain-style writes `$.user.name.set('Ada')`, and full `patch` blocks all share the same grammar.
- **Bytecode VM with peephole fusion.** Common shapes (`map(f).sum()`, `filter(p).first()`, deep `$..find`, `map(@.split.map(len).sum())`) collapse to single opcodes; repeated queries hit a compile cache.
- **Columnar `Val` lanes.** `IntVec` / `FloatVec` / `StrVec` / `StrSliceVec` / `ObjVec` run homogeneous arrays as packed vectors — tight loops instead of enum dispatch.
- **SIMD byte-scan for deep queries.** `$..id`, `$..k == lit`, chained `$..a..b..c` — memchr-accelerated scans on raw document bytes when it's faster than walking the tree.
- **Python-like ergonomics.** Comprehensions, lambdas, f-strings, pipelines, destructuring, ternary (`x if cond else y`), deep search (`$..find(…)`, `$..shape({…})`, `$..like({…})`).
- **Zero-copy views.** `Val::StrSlice` / `Val::StrSliceVec` share parent `Arc<str>` buffers so `slice`, `substring`, `split.first` return without allocating.
- **Safe and honest.** Tree-walker reference implementation, 540+ unit tests, Miri-verified `unsafe` invariants ([SAFETY.md](SAFETY.md)). Every optimization is a specialization on the same semantics.

---

## The expression language

| | |
|---|---|
| `$` root | `@` current |
| `.f` field | `.f?` null-safe field |
| `..k` recursive descent | `..k?` null-safe (still array) |
| `[i]` / `[a:b]` index / slice | `[*]` wildcard |
| `.filter(p)`, `.map(f)` | `.sort_by(k)`, `.group_by(k)` |
| `.find(p)`, `.unique_by(k)` | `.count_by(k)`, `.index_by(k)` |
| `$..find(@.k == lit)` | `$..shape({name, price})` |
| `.set(v)`, `.modify(@ * 2)`, `.delete()` | `patch $ { .a.b: 1, .c: DELETE }` |
| `[x for x in xs if pred]` | `{k(x): v(x) for x in xs}` |
| `lambda x: …` / `x => …` | `let a = expr in …` |
| `f"hello {$.name}"` | `x when cond else y` |

Full syntax reference: [jetro-core/src/SYNTAX.md](jetro-core/src/SYNTAX.md)

---

## Install

```toml
[dependencies]
jetro = "0.4"
```

---

## Benchmarks

Ratio = `jetro / hand-written Rust + serde_json`. Lower is better; sub‑1.0× means jetro wins.

| workload | vs native |
|---|---|
| `map(@.split('-').map(len).sum())` | **0.51×** (jetro 2× faster) |
| `map('prefix-' + @ + '-suffix')` | **0.75×** |
| `map(@.split(...).count())` | **0.77×** |
| `count_by(grp)` | **0.85×** |
| `min`, `max`, `kv` ops | **0.27×–0.85×** |
| `$.map({id, grp})` projection | 2.49× |
| `filter(status == 'ok')` | 1.69× |
| `$.map(@.slice(10, 30))` | 2.34× |
| `f"{@.id}_{@.grp}"` | 1.59× |

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
