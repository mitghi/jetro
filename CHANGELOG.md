# Changelog

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
