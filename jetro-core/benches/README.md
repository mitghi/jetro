# jetro-core benches

Microbenchmarks for the query engine. Driven by [criterion].

## Layout

- `match_bench.rs` — pattern-match runtime workloads.

Run with:

```
cargo bench -p jetro-core --bench match_bench
```

To compare against a saved baseline:

```
cargo bench -p jetro-core --bench match_bench -- --baseline match-v0.6
```

To save a new baseline:

```
cargo bench -p jetro-core --bench match_bench -- --save-baseline <name>
```

## Baselines

Baselines live under `target/criterion/<bench>/<baseline>/` and are not
committed (the `target/` directory is gitignored). Each benchmark
target prints its current sample mean to stdout; recorded reference
numbers below are from a Mac M-series host running on macOS.

| Baseline name | Workload | Mean (mid) | Notes |
|---------------|----------|------------|-------|
| `match-v0.6` | `match_filter_take` | ~64 µs | `.filter(match @ ...).take(3)` over a 1024-row int array. |
| `match-v0.6` | `match_dispatch_5_arms` | ~1.04 ms | Tagged-union dispatch over 1024 events with 5 arms (cross-arm prefix sharing active). |
| `match-v0.6` | `match_range_scan` | ~148 µs | Range-pattern dispatch over a 1024-row int array. |

Numbers are end-to-end (parse + compile + execute per iteration). To
benchmark steady-state hot-path execution, build a `JetroEngine` once
and reuse the cached plan across iterations — see `lib.rs::JetroEngine`
for the API.

[criterion]: https://docs.rs/criterion
