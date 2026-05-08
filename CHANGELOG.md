# Changelog

## 0.5.6

### Path-receiver scalar unwrap

- **Scalar method on path no longer wraps**. `$.s.upper()` now returns
  `"FOO"` instead of `["FOO"]`. The planner detects pipelines whose source
  is a `FieldChain` and whose every stage is a scalar/object one-to-one
  builtin and bypasses pipeline lowering, dispatching `apply_one` /
  `apply_args` directly on the chain's single value. Bonus consequence:
  `$.users.to_json()` now produces a single JSON document of the array
  rather than an array of per-element JSON strings.
- **`BuiltinSpec::never_unwrap()`**. Per-builtin opt-out flag; when set
  the planner keeps pipeline streaming as the canonical semantic on path
  receivers regardless of category/cardinality.
- **`BuiltinSpec::dispatches_scalar_direct()`**. Spec-level helper that
  decides eligibility (Scalar or Object category, OneToOne cardinality,
  not opted out). Used by both `try_lower_pipeline` and the top-level
  fast-path lowering in `plan_query_with_context`.

### `rec` family

- **`rec(fn, cond)` 2-arg form**. Iterates `fn` while `cond(@)` is truthy,
  capped at 10 000 iterations. Returns the value at the point `cond` first
  becomes falsy.
- **Improved 1-arg cap message**. `rec(fn)` now reports
  `"rec(fn): no fixpoint within 10000 iterations — pass rec(fn, cond) to
  bound the loop or ensure fn is idempotent"` so non-idempotent steps
  surface the cap loudly rather than silently spinning.

### Object-shape arg forms

- **`zip_shape({a, b})`**. Object-literal sugar for the multi-arg
  interleave shape; equivalent to `zip_shape(a, b)`. Mixed shorthand and
  `name: expr` fields supported.
- **`group_shape(key)` 1-arg projection**. Single-arg form keys each
  element by the projected value; bucket value is the original element.
  Accepts bare ident, named arg, `@`-form, `name => …` arrow, and
  `lambda x: …` shapes.
- **`partition(pred)` tuple workflow**. `partition` returns
  `[matching, non_matching]` and now pairs naturally with tuple `let`,
  enabling one-pass splits such as
  `let (active, inactive) = $.store.books.partition(active) in {...}`.

### Parser additions

- **Tuple `let` binding**. `let (a, b) = expr in body` desugars at parse
  time to a synthetic ident plus indexed scalar lets — no runtime tuple
  binding opcode. Tuple bindings compose with ordinary multi-let bindings,
  and names that collide with the synthetic prefix are skipped via a
  counter sweep.
- **Bare-path `.field` in method args**. `$.users.filter(.active)` ≡
  `(@.active)`. The leading-dot shorthand desugars at parse time to a
  `Chain(Current, [Field(name)])` so the planner sees identical opcodes.
- **Object-pattern shorthand `{id, name}` in `match`**. Equivalent to
  `{id: id, name: name}`; rest-capture stays spelled `...*rest` (object)
  or `...tail` (array).
- **Lambda array-destructure with rest**. `([h, ...tail]) => body` lowers
  to a synthetic ident plus `let h = synth[0] in let tail = synth[1:] in
  body`, mirroring the `match` rest pattern.
- **`indent("> ")` string prefix**. The single-arg `indent` builtin now
  accepts a string-literal prefix in addition to the integer count;
  `apply_args` dispatches on `BuiltinArgs::Str` vs `BuiltinArgs::Usize`.
- **Escaped quotes in string literals**. The Pest grammar greedily
  consumes `\X` escapes so `"{\"a\":1}".from_json()` parses; both
  double- and single-quoted forms accept the full escape table.

### Tests

- New integration test `jetro-core/tests/builtin_arg_forms.rs` — 80
  assertions covering every argument-form spelling for builtins that take
  args or lambdas (predicate forms, projection forms, multi-arg, bare
  identifiers, positional values, regex, path mutation, chain-write,
  deep/walk).
- New unit module `tests/v0_5_5_quickfixes.rs` — coverage for the second
  batch of v0.5.5 fixes (`.has(v)` boolean return, `.remove(pred)`,
  `missing(...)`, multi-segment `get_path`, `dedent` common-prefix,
  `enumerate`/`pairwise` on path sources, no-arg `zip_shape`/
  `group_shape`, `partition`, `approx_count_distinct`, string-escape
  table), plus tuple-let regression coverage for `partition`, mixed
  tuple/scalar lets, and synthetic-name collision avoidance.
- Flipped four stale negative-invariant tests in `unsafe_invariants.rs`:
  `map_str_concat_*` and `zip_shape_named_and_bare` now assert correct
  output rather than `is_err()` (the underlying behavior was fixed in
  0.5.5).

## 0.5.5

### Demand propagation and functional batched updates

- **Demand lanes and late projection planning**. Added shared demand models
  for scan needs and result needs, with physical-plan annotations for delayed
  one-to-one projections. This lets chains such as
  `$.books.filter(price > 20).map(isbn).last()` scan for the semantic winner
  first and project only the selected result.
- **Indexed and reverse positional demand**. Pipeline planning now propagates
  first/last/nth/bounded-prefix demand through eligible chains and selects
  indexed, reverse, bounded, or fallback source access from source
  capabilities.
- **Ordering-aware demand paths**. Added lazy ordered suffix handling for
  safe `sort/filter/take/map/last` shapes while preserving prefix barriers
  such as `drop_while` / `take_while`.
- **Functional `.update({...})` batches**. Rooted writes now lower to a
  first-class `UpdateBatch` AST / physical-plan node instead of materializing
  one full document per write. Multi-field updates share selector traversal,
  group static paths into an update trie, and return the full updated root
  once.
- **High-performance update execution**. VM update execution mutates selected
  paths with `Arc::make_mut`, preserving untouched subtree sharing. Wildcard
  and filtered-wildcard updates compact selected deletes correctly, and
  invariant RHS expressions are evaluated once per batch when safe.
- **Functional write examples**:
  `$.books[*].update({ tags: tags.append("test"), reviewed: true })`,
  `$.books[* if year > 1980].update({ tags: tags.append("modern") })`,
  and root batches such as
  `$.update({ "books[*].tags": @.append("test"), active: false })`.

### Builtin runtime fixes (limitations.md sweep)

- **`.has(v)` method** — returns boolean. Previously returned the receiver
  unchanged on arrays and `[true]`/`[false]` (single-element wrap) on
  objects. Spec moved off `scalar_native_element_spec`'s `.element()`
  marker; runtime extended to handle arrays, vectors, strings.
- **`.remove(pred)`** — predicate body is now evaluated. The dispatch
  previously matched only `Expr::Lambda`; `@`-form predicates fell
  through to value-equality. Routes any expression that references
  `Expr::Current` to the predicate path.
- **`missing(...keys)`** — variadic returns the array of absent keys.
  Single-key form keeps the legacy boolean. New `missing_many_apply`
  helper.
- **`update(path, fn)`** — two-arg form reads via `get_path`, applies
  `fn`, writes back via `set_path`. The 1-arg form (single-lambda)
  preserves prior behavior.
- **`get_path("a/b/c")` / `has_path` / `del_path` / `set_path`** — slash
  separator now joins to dot/bracket forms. Numeric segments parse as
  array indices (`users/0/name` walks `users[0].name`).
- **`dedent()`** — strips common leading whitespace per line. Backed by
  the new string-literal escape processing (see Parser below).
- **`now()`** — top-level builtin returning Unix-millis via
  `eval_global_compiled`.
- **`.enumerate()` and `.pairwise()` on path sources** — both removed
  `.element()` from their specs so the streaming pipeline stops wrapping
  the result and discarding the structural pairing.
- **`.zip_shape()` / `.group_shape()`** — no-arg forms wired:
  - `zip_shape()` over an object-of-arrays interleaves to an array of
    objects (parallel-array → row form).
  - `group_shape()` over an array of objects buckets by sorted-key-set.
- **`.partition(pred)`** — returns `[matching, non-matching]` tuple
  (was object `{true, false}`). Pairs with array-pattern destructure
  in lambdas and indexing (`partition(p)[0]`).
- **`.approx_count_distinct()`** — HLL backend with 14-bit precision
  (M=16384 registers, ~16 KiB state, ≈0.81% RSE). Linear-counting
  correction makes small inputs exact. Hash via `DefaultHasher`
  (SipHash) for stable avalanche on small string keys.

### Parser

- **String-literal escapes**: `\n`, `\r`, `\t`, `\0`, `\\`, `\"`, `\'`,
  `\xNN`, `\uXXXX` are now processed during parse. Unknown escapes
  pass through untouched (`\d`, `\w`, `\s` → regex patterns continue
  to work). Pre-fix the parser was raw passthrough — `"a\nb".lines()`
  saw 4 chars `a\nb` instead of `a` `\n` `b`, so any builtin that
  inspected newlines (lines, dedent, indent, words) silently
  misbehaved.

### Tests

40 new tests in `tests::v0_5_5_quickfixes` (HLL, escapes, runtime
fixes). Total: 1285 lib tests pass.

### Grammar

- **Wildcard `[*]`**. Mid-chain expansion: `$.items[*].x.set(0)` lowers to
  `$.items.map(@.x.set(0))`. Trailing `[*]` is identity over the array.
  Read-context only — no special runtime opcode.
- **Slice with step**: `[a:b:c]`, `[::n]`, `[::-1]` (reverse).
  `Step::Slice(a, b, step)`; `step == None | Some(1)` keeps the existing
  step-1 fast path. Negative step walks backward.
- **Lambda array-pattern destructure**: `([k, v]) => body` desugars to a
  synthetic param plus chained `let` bindings. Both arrow and `lambda`
  keyword forms accept the new pattern.
- **Reserved keywords as object/pattern keys**: `{kind: "click"}` now
  parses (in object literals and `match` arm patterns). New
  `loose_ident` rule used in key positions only — `is kind` operator
  unchanged.

### Runtime

- **`Val::StrSlice + Val::Str` string concat**. Path-rooted concat
  (`$.user.first + "-" + $.user.last`) and f-string interpolation across
  borrowed slices both now produce the joined string. Numeric and
  array-concat hot paths unchanged.
- **`entries()` / `keys()` / `values()` triple-wrap fix**. Removing
  `.element()` from `object_element_spec` stops the streaming pipeline
  from wrapping these whole-object results into single-element arrays.
  `$.o.entries()` is now `[[k,v], …]` instead of `[[[k,v], …]]`.
  Restores `group_by().entries()` and `count_by().entries()` to their
  documented shapes.
- **`rec` fixpoint** uses deep structural equality (new
  `vals_deep_eq`), not the scalar-only `vals_eq`. Object and array
  inputs converge in 1–2 iterations instead of looping to the 10000
  ceiling.

### Builtins

- **`parse_int(radix)`**. Accepts radices 2–36, strips `0x` / `0b` /
  `0o` prefix when matching base. No-arg form unchanged (base-10).
- **`to_csv(headers)` / `to_tsv(headers)`**. Optional headers array
  drives explicit column order with a header line emitted first.
  Missing keys produce empty cells. No-arg paths unchanged.
- **`accumulate(init, fn)`**. Two-arg fold variant: explicit initial
  accumulator, one output per input. The single-arg form
  (`accumulate(fn)` — seed from `items[0]`) and IntVec/FloatVec
  specialised binop paths preserved.

### Tests

92 new tests across 4 files: `tests::grammar_extensions` (38),
`tests::strslice_arith` (10), `tests::entries_wrap` (17),
`tests::builtin_migrations` (27). Total: 1245 lib tests pass.

### Bench

`cargo bench -p jetro-core --bench match_bench -- --baseline pre-fix14`:
no regression > 2%; `match_range_scan` -3.6% improved.

## 0.5.4

### Grammar fix

- `!=` operator now parses everywhere — top-level, method args, lambda
  bodies, match guards, inline filters, list-comp guards, ternary
  conditions. The postfix `!` quantifier previously consumed the leading
  `!` of every `!=`, surfacing as a confusing
  "expected kw_and / kw_or / kw_if / kw_kind" diagnostic at the
  whitespace before the comparator. Quantifier rule now uses a `!"="`
  negative lookahead, mirroring the existing `?` quantifier's defensive
  `!("|" | "?")` lookahead. The bare `!` quantifier on its own (e.g.
  `xs!` exactly-one assertion) keeps working.
  
## 0.5.3

### Lambda

- AST-level single-param lambda lowering: `r => r.id` and `lambda r: r.id`
  emit byte-identical opcodes to the equivalent `@`-form. Kernel
  classification (`FieldRead`, `FieldChain`, `FString`, `Object`,
  `FieldCmpLit`, …) fires uniformly across all forms.
- Multi-param fast path: rightmost param substituted to `Current` so
  comparator and binary-HOF bodies skip one `LoadIdent` per row.
- `Opcode::BindLamCurrent` for nested-lambda outer-param refs: emitted
  only when an inner lambda body reads the outer param. One env clone +
  `push_lam` / `pop_lam` per outer row, never per inner element.
- Kernel fast path no longer gates on `lam_param.is_none()` —
  single-param named lambdas hit the same Rust kernels as `@`-form.
- `push_lam` skips the unused-name binding when the AST substitution
  has removed every reference, dropping one env-var insert/remove per
  row across HOFs.
- First-class lambdas via let-bound macro expansion:
  `let f = (x => x*2) in $.xs.map(f)` desugars at parse time. Aliased
  chains (`let g = f`) supported. Method-arg position only.

### Engine

- Filter-pushdown over arithmetic projections fixed: `FilterBeforeMap`
  optimizer rewrites the pushed-down predicate via
  `substitute_current_with_expr(predicate, projection)` so the swapped
  filter sees source rows but tests the same mapped value the original
  placement would have observed.
- `normalize_symbolic` unwraps single-param `Expr::Lambda` wrappers
  before threading them through symbolic substitution, so vm and
  engine paths agree on lambda-bearing pipelines.
- `compile_stage_expr` no longer re-unwraps lambdas; the symbolic pass
  owns that responsibility (prevents corruption of
  `r => (x => x)`-style outer-Lambda-wrapping-inner-Lambda bodies).
- `.sort((a, b) => …)` engine path: pipeline lowering bails out on
  multi-param lambda args so the router falls back to VM
  `sort_comparator_apply`.
- VM `Opcode::DynIndex` accepts `Val::StrSlice` keys (simd-json tape
  paths) — previously fell through to `Val::Null` for borrowed strings.

### Builtins

- `.find(pred)` returns the first match (conventional first-match
  semantics) and `Val::Null` when nothing matches. `.find_all(pred)`
  keeps the previous filter-alias semantics. Spec migrated to
  `BuiltinPipelineLowering::TerminalExprArg { terminal: First }`; VM
  dispatch routes `Find | FindFirst` through `find_first_apply`.

## 0.5.2 — unreleased

### Pattern Matching

- New `match scrutinee with { pat -> body, ... }` expression with first-match
  semantics, optional `when` guards, and full composition with `.map`,
  `.filter`, pipelines, and postfix navigation.
- Pattern forms: wildcard `_`, scalar literal, identifier bind, kind-bind
  `s: string`, kind-only `string`, object pattern `{k: pat}`, array pattern
  `[a, b]`, array head/tail `[head, ...tail]`, or-pattern `a | b | c`,
  numeric range `1..10` / `0..=100` (negative and float bounds supported),
  and parenthesised sub-patterns.
- Object rest binding `{k: pat, ...*rest}` captures every unlisted key as a
  freshly built `Val::Obj`; the `...*expr` sigil is also accepted in object
  body position as a shallow-spread synonym.
- Deep search variants: `$..match { arms }` walks every descendant in DFS
  pre-order and collects truthy arm-body results; `$..match! { arms }`
  is the early-stop variant returning the first truthy result.
- Compile-time analyses: or-pattern linearity check at parse, optional
  exhaustiveness lint via `JETRO_STRICT_MATCH=1`, shape summary
  (`ObjAnyOfKeys` / `KindOnly` / `NumericRange`) for runtime pre-filter and
  structural dispatch.
- Runtime: flat `MatchOp` decision-machine VM with cross-arm shared-prefix
  hoisting (single-key, multi-key, array-length, kind variants) and partial
  prefix sharing for mixed-suffix arm lists; or-of-literals lowers to a flat
  `LitEq + Jump` cascade; or-with-binders falls back to a tree-walk helper.
- View-domain runtime `exec_match_view<V: ValueView>` runs pattern tests
  against borrowed views; pipeline `MatchFilter` / `MatchMap` stages dispatch
  through it directly to skip per-row VM opcode dispatch.
- Bitmap-backed `..match` via `jetro-experimental`: when every arm shares an
  object key prefix, the planner emits `StructuralPlan::DeepMatch` and
  enumerates candidates from the structural index instead of walking the
  full document tree.
- Demand model integration: `ChainOp::Match { Predicate, Transform, Multi }`
  classifies match-bodied stages so `Take`, `find`, and `count` correctly cut
  upstream demand through `.filter(match @ ...)` chains.
- Bench harness: `cargo bench -p jetro-core --bench match_bench` exercises
  `match_filter_take`, `match_dispatch_5_arms`, `match_range_scan`. Reference
  baseline saved as `match-v0.6`.

### Reserved keywords added

- `match`, `with`

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
