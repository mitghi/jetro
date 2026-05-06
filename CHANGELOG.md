# Changelog

## 0.5.1 — 2026-05-06

### Architecture

- Reorganized `jetro-core/src/` around the current execution pipeline:
  `parse/`, `compile/`, `vm/`, `data/`, `ir/`, `plan/`, `exec/`, and
  `builtins/`.
- Moved execution backends under `exec/`, including the router, interpreted
  physical executor, structural backend, view backend, composed helpers, and
  pipeline backend.
- Split compiler and VM code into dedicated modules, with opcodes separated
  from VM execution.
- Consolidated value, tape, view, context, and runtime substrate types under
  `data/`.
- Moved builtin helper/runtime bodies under `builtins/ops/` and kept builtin
  facts in the builtin definition and registry modules.
- Split the large regression test module into focused parser, examples,
  deep-search, chain-write, patch-fusion, and regression modules.

### Demand Propagation

- Added positional pull demand support for `last()` and `nth()` so eligible
  chains can request `LastInput(n)` or `NthInput(i)` instead of scanning every
  row.
- Added projection-aware value demand metadata and broader demand propagation
  tests in the chain IR.
- Added generic late-projection behavior for safe one-to-one maps, allowing
  chains such as `map(...).first()`, `map(...).last()`, `map(...).nth(n)`, and
  `map(...).take(n)` to evaluate projections only for demanded rows.
- Added indexed, reverse, and bounded source access handling for materialized,
  composed, view, and tape-backed execution paths.
- Fixed bounded sort planning for chains where a selective suffix follows
  `sort`/`sort_by`; bounded top-k/bottom-k is now used only when the downstream
  suffix is semantically one-to-one.
- Updated lazy sorted suffix execution so `last()` traverses from the sorted
  tail when safe, while selective suffixes scan until a matching output is
  found.
- Kept `unique()` conservative in the owned pipeline fallback by forcing the
  legacy materialized path where stateful distinct handling is required.

### Patch Fusion

- Added a patch-fusion planner with effect summaries, root-reference tracking,
  alias tables, pending write batches, scope-aware flushing, and path-trie
  batching.
- Added same-root contiguous fusion across pipes, objects, and lets.
- Added fusion support for cross-let alias chains, lambda/comprehension bodies,
  per-iteration chain-write lifting, conditional trie nodes, and final plan
  completion.
- Added soundness coverage for read-after-write behavior, scope isolation,
  alias resolution, conditionals, and atomicity.

### Fuzzing and Benchmarks

- Added a `cargo-fuzz` harness for parse, plan, and collect targets behind the
  `fuzz_internal` feature.
- Added recorded parse timeout artifacts for known parser stress cases.
- Updated the benchmark baseline to `0.5.0` after the module restructure,
  structural backend work, demand propagation improvements, and patch fusion.

### Validation

Verified before release:

```bash
cargo test -p jetro-core
cargo test
git diff --check
```

- Core tests passed: 858 unit tests, 157 integration tests, and 1 doctest;
  1 unit test ignored.
- Workspace tests and doctests passed.

---

## 0.4.0 — 2026-05-05

### Breaking Changes

- **Public API is now byte-first.** The top-level `jetro` crate exposes a
  minimal API centered on `Jetro::from_bytes(bytes)` and `Jetro::collect(expr)`.
  Older direct tree-walker helpers, prelude exports, and custom function
  registration paths have been removed from the public facade.
- **Custom/user-registered function support has been removed for now.** Builtins
  are statically known and dispatched through the builtin system.
- **CamelCase builtin aliases are no longer supported.** Builtin names are
  canonicalized around snake_case.
- **Legacy eval/tree-walker modules have been removed.** The VM and physical
  executor are now the correctness path, with optimized backends selected by
  the planner where possible.

### Architecture

- Added a unified physical planning layer with `QueryPlan`, `PlanNode`, backend
  capabilities, backend preferences, and execution facts.
- Added recursive physical planning for object shaping, nested expressions,
  receiver pipelines, scalar chains, structural prefixes, and fallback nodes.
- Added backend-aware execution through structural index, tape/value-view,
  pipeline, and VM fallback paths.
- Added `JetroEngine`, a long-lived engine with explicit plan caching and a
  shared VM for repeated queries across documents.
- Moved builtin behavior toward a registry/trait-driven model. Builtin identity,
  metadata, demand laws, lowering shape, sink behavior, and execution policy are
  centralized instead of scattered across VM, pipeline, view, and composed
  paths.

### Performance

- `simd-json` is enabled by default.
- `Jetro::from_bytes` keeps raw bytes and lazily builds expensive
  representations only when needed.
- Added lazy simd-json tape handling and `TapeView`/`ValueView` execution paths.
- Added on-demand tape row streaming for eligible pipelines.
- Added structural-index execution for supported deep-search queries.
- Added demand propagation through pipeline chains.
- Added bounded sort/top-k strategies where downstream demand makes them safe.
- Added view-native execution for more scalar, projection, reducer, keyed
  reducer, object-map, f-string, and terminal collection paths.
- Added columnar and object-vector execution paths for uniform object arrays.
- Reduced reliance on hand-written fused VM opcodes in favor of metadata-driven
  pipeline and builtin execution.

### Pipeline and Demand Propagation

- Added a pipeline IR with explicit source, stages, sinks, body kernels, sink
  demand, stage strategy, view capability, and materialization policy.
- Demand now flows backward from sinks through stages:
  - `filter(...).first()` can stop after the first matching output.
  - `take(n)` can cap upstream input demand.
  - `count()` can avoid materializing row payloads where supported.
  - `sort_by(...).take(k)` can use bounded top-k strategy when semantically
    safe.
- Added safety-aware handling for barrier stages such as `sort_by`,
  `take_while`, `unique`, keyed reducers, and materialized suffixes.

### Builtins

- Migrated builtin definitions into `jetro-core/src/builtins/`.
- Added `Builtin` trait metadata and static dispatch hooks.
- Added centralized builtin specs for names and aliases, category, cardinality,
  lowering, streaming behavior, barrier behavior, sink behavior, demand law,
  structural capability, and view capability.
- Removed old `functions.rs`/eval-style builtin shims.
- Removed duplicated dispatch tables where possible.

### Removed

- Removed legacy eval modules.
- Removed legacy graph support.
- Removed old scan/bytescan paths.
- Removed unused schema/plan/cfg/ssa modules.
- Removed the old macro crate from the workspace.
- Removed many obsolete fused VM opcodes and peephole paths now covered by the
  planner/pipeline architecture.

### Documentation

- Rewrote the root README around the new byte-first API.
- Rewrote `jetro-core/README.md` to explain physical planning, backend
  selection, demand propagation, tape/value-view execution, builtin registry
  design, and VM fallback.
- Updated syntax and in-depth documentation for the current API direction.

### Benchmarks

- Expanded `bench_cold` with the full cold benchmark case set from `jqvsjetro`.
- Added and updated benchmark examples covering cold start, nested projections,
  f-strings, jaq comparisons, lock/cache behavior, and complex pipeline chains.

### Validation

Verified before release:

```bash
cargo check -p jetro-core --examples --offline
cargo test -p jetro-core --offline
cargo test --offline
cargo package -p jetro-core --allow-dirty --offline
cargo package --allow-dirty --offline
```

- Core unit tests passed: 749 passed, 1 ignored.
- Integration tests passed.
- Doctests passed.
- Examples compile.
- Package verification passes for both `jetro-core` and `jetro`.

---

## 0.3.0

v2 Tier 1 search / match / collect / chain-style writes.

- `.find / .find_all / .unique_by / .collect / .pick(alias: src)`
- Deep search: `$..find`, `$..shape`, `$..like`
- Chain-style terminal writes: `.set / .modify / .delete / .unset`
- Breaking: `$.field.set(v)` now returns the full doc (was: `v`).
  Pipe form `$.field | set(v)` preserves the old semantics.
- VM Phase 1: inline caches, fusion passes, COW fast-path.
- VM Phase 3: columnar `IntVec` / `FloatVec` / `StrVec` lanes; typed
  aggregate and filter fast paths.
