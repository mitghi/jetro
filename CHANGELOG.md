# Changelog

## 0.5.10

### Release focus

- **End-to-end demand/tape execution**. Planned work for this release focuses
  on carrying shared demand metadata from builtin definitions through logical
  planning, physical planning, backend selection, tape/view execution, pipeline
  execution, and NDJSON row execution without query-shape-specific fusions.
- **Clean architecture and observability**. Planned work includes explicit
  execution-path labels, fallback reasons, tighter module boundaries, removal
  of unwired prototype code where it is not useful, and regression tests that
  prove hot paths stay on the intended backend.
- **Correctness hardening for all builtins**. Planned work includes auditing
  builtin metadata, VM/pipeline/view/tape consistency, null/missing behavior,
  arity and alias behavior, and adding equivalence tests for optimized builtin
  families.
- **NDJSON performance and proof**. Planned work includes extending generic
  byte/tape direct execution, preserving cold-path performance, documenting
  benchmark methodology, and adding path-selection tests for static projections,
  filtered streams, reducers, and early-stop queries.

### NDJSON observability

- **Direct writer plan kinds are now test-visible**. NDJSON direct planning can
  expose whether a query produced a byte expression plan, a tape root/scalar
  plan, a stream collect/count/numeric plan, or a static projection plan. This
  gives regression tests a stable way to prove hot shapes such as `$.name`,
  `$.a.b.c`, object/array projections, filtered counts, and numeric reducers
  stay on the expected direct execution family.
- **Runtime writer-family selection is now test-visible**. NDJSON tests can
  distinguish pure byte-expression writers, byte-writable tape-plan writers,
  and tape fallback writers, so performance assertions match the actual writer
  branch used by `run_ndjson`.
- **Adaptive structural hint groundwork has started**. NDJSON now has an
  internal schema-hint module that observes simple root object layouts through
  the existing byte scanner, records stable field slots, detects unstable field
  order, and rejects rows the byte scanner cannot validate instead of creating
  unsafe hints.
- **Hintable NDJSON access is derived from direct plan metadata**. Static
  projections and stream plans now feed a generic access-path inventory that
  captures source and projected paths algorithmically, forming the basis for
  schema-guided byte access without query-chain-specific fusions.
- **NDJSON structural hints now have adaptive activation rules**. The internal
  hint state stays in learning mode until enough rows validate the required
  root fields, refuses unstable field orders, and disables itself after too
  many byte-scanner rejections so hinting remains fallback-safe.
- **Stable root layouts can now produce a partial row index**. Learned root
  object layouts can validate a row's field order and expose raw value byte
  spans by slot, providing the primitive needed for projection pushdown without
  building a full per-row structural index.
- **Adaptive hints are wired into static NDJSON projections**. After the
  learning threshold, byte-writable object and array projection plans can use a
  validated root-layout match to emit root-field values by slot, while any
  unsupported value shape or row mismatch falls back to the existing writer.
- **NDJSON hint state now tracks activation counters**. The adaptive hint layer
  records learned rows, rejected rows, hinted rows, and disabled state so future
  explain/debug output can prove when schema-guided byte access is active.
- **NDJSON hints now avoid per-row span allocation**. Active hint matches reuse
  state-owned span scratch storage, keeping the hot projection path allocation
  free after schema learning.
- **Nested and scalar projection values can use hints**. Static object and
  array projections can jump to a learned root slot and then reuse the existing
  byte suffix walker or scalar writer for paths such as `$.profile.name` and
  `$.profile.name.upper()`, without adding query-specific fusion chains.
- **Hint fallback is preflighted before output**. Hinted projection writers now
  prove every projected value can be emitted before writing an object or array,
  so unsupported nested suffixes fall back cleanly without partial output.
- **Stale hints self-disable on layout misses**. Once active, the hint layer
  counts post-activation layout mismatches and disables itself after repeated
  misses, preserving correctness on mixed-shape NDJSON while avoiding repeated
  failed fast-path attempts.
- **Root-slot matching now stops at the demanded fields**. Active hints validate
  only the root slots required by the direct plan and stop after the last needed
  slot, avoiding full root-object scans for early-field projections.
- **Schema learning stops after activation**. The adaptive state machine now
  performs full schema observation only during the learning window; active rows
  go directly through the required-slot matcher. The default learning threshold
  is two stable rows to favor cold NDJSON workloads.
- **NDJSON string scanning uses SIMD byte search**. The byte parser now uses
  `memchr2` for quote/backslash discovery in JSON string scanning, keeping the
  simple-key and value-skip paths conservative while using platform-accelerated
  byte search where available.
- **Stream source hints now cover map, count, and numeric reducers**. Direct
  stream plans can reuse the learned root source slot for collect, filtered
  count, and numeric reducer sinks, then execute the existing raw-byte item
  predicate, projection, and numeric fold logic.
- **Numeric NDJSON streams can stay byte-native**. Queries such as
  `$.attributes.map(@.weight).sum()` now route through byte-writable tape plans
  instead of forcing tape materialization for every row.
- **Stream item projections avoid repeated item scans**. Root-field stream maps
  such as `attributes.map([@.key, @.value])` and object-shaped maps now scan
  each simple item object once, cache the requested field spans for that item,
  and reuse the spans across array/object/scalar projection writers.
- **Filtered stream sinks now share item field spans**. Filtered stream maps,
  filtered counts, and filtered numeric reducers can evaluate root-field
  predicates and projections/reducer inputs from the same per-item byte scan,
  avoiding duplicated path resolution without adding query-shape-specific
  fusion chains.
- **Stream extrema retain projected output directly**. Root-projectable
  `sort_by(...).first()/last()` suffixes can keep only the selected projection
  bytes while scanning candidates, avoiding whole-item retention for hot
  extrema such as `$.attributes.sort_by(@.value).last().key`.
- **NDJSON stream caches validate learned item prefixes**. Constant stream-map
  reuse now proves that learned value offsets still belong to the same item
  field prefix, so reordered or mixed-shape item objects fall back safely.
- **Mixed-shape stream items have explicit regression proof**. Optimized
  stream maps and filtered counts are now covered for missing item fields,
  preserving `null` projection output and non-matching predicate behavior on
  heterogeneous NDJSON arrays.
- **Stream writer scratch storage is smaller and allocation-light**. Common
  stream field sets and span buffers use inline storage, and fixed projection
  keys are written directly through the JSON string writer rather than via
  temporary `Val` allocation.
- **Direct stream projection plans are built once per row source**. Root-field
  array/object stream maps now derive reusable slot plans for raw fields and
  supported scalar calls, reducing per-item projection work while preserving
  fallback through the generic writer for nested or optional shapes.
- **Direct stream projection plans understand nested suffixes**. Stream item
  projections can jump to a root item field and then reuse the byte suffix
  walker for paths and scalar calls such as `@.meta.code.upper()`, keeping
  nested item shaping on the same algorithmic direct projection path.
- **Release validation is green in release mode**. The full workspace test
  suite passes with `cargo test --release --verbose --workspace`, covering the
  optimized NDJSON byte/tape paths alongside the existing VM, parser, planner,
  builtin, and API tests.
- **The 25-query NDJSON CLI profile is re-baselined**. On the local
  4.76M-row benchmark file, direct root projections run in 0.32-0.88s and
  stream projection/filter/reducer cases are generally above 10x versus `jaq`.
  Remaining weaker cases are concentrated around per-row tail/extrema access
  and mixed root-plus-stream output, which are now the next optimization target.
- **NDJSON extrema plans are observable as extrema**. Test-only direct-plan
  labels now distinguish `sort_by(...).first()/last()` stream extrema from
  numeric reducers, making performance and routing regressions easier to pin
  to the correct executor family.
- **Array-element demand uses root-field prefixes**. Direct byte paths for
  `first`, `nth`, and `last` array element access now avoid computing an exact
  root-field value span when the downstream array walker can stop at the
  demanded element boundary. This improves `last()` and mixed root/stream
  projections without adding query-shape-specific fusion.
- **Selective stages distinguish reverse last from nth demand**. Filter-like
  builtins and predicate match stages preserve ordered `LastInput` demand for
  reverse-capable executors, while `nth` after a selective stage remains a
  conservative ordered full scan.
- **Bounded positional demand is explicitly order-sensitive**. Selective,
  distinct-like, expanding, and multi-match demand laws now mark bounded
  positional output as ordered work, so later physical planners cannot treat
  `first`, `nth`, `last`, or `take` after these stages as order-insensitive.
- **Selective and expanding positional semantics have runtime proof**. NDJSON
  coverage now proves `filter(...).last()` returns the last matching output,
  and builtin coverage proves `flat_map(...).last()` follows semantic output
  order.
- **Registry demand invariants guard positional ordering**. Builtin registry
  tests now assert that selective, distinct-like, and expanding demand laws
  preserve ordering for bounded positional sinks, with runtime coverage for
  `unique().last()`.
- **NDJSON string extrema compare simple keys directly**. Stream extrema can
  compare simple JSON string keys from raw bytes before falling back to full
  scalar comparison, improving `sort_by(string).first()/last()` without adding
  query-chain-specific fusion. Escaped string keys have regression coverage for
  the fallback path.
- **NDJSON extrema compare numeric keys directly**. Stream extrema also use the
  lightweight raw scalar comparator for numeric sort keys, with regression
  coverage for integer, negative, and floating-point keys.
- **NDJSON streams have a generic `first` sink**. Filter/map pipelines ending
  in `first()` now lower to a reusable direct stream sink that stops at the
  first matching item and applies the planned projection once. The benchmark
  shape `filter(...).map({...}).first()` improved locally from roughly 4.26s to
  roughly 3.8s on the 4.76M-row file, and the direct-plan test label now
  distinguishes stream-first from stream-collect.
- **Unfiltered stream-first has direct regression proof**.
  `map(...).first()` now has focused NDJSON coverage for empty and non-empty
  arrays, proving the generic stream-first sink also handles demand without a
  predicate stage.
- **Unfiltered stream-first uses first-child source demand**. Byte execution
  now extracts only the demanded first array child for `map(...).first()` when
  no predicate can require a later item; filtered `first()` still scans until
  the first matching semantic output. Locally, `$.attributes.map(@.value).first()`
  improved from roughly 2.56s to roughly 1.97s on the 4.76M-row file.
- **NDJSON streams have a generic `last` sink**. `map(...).last()` and
  `filter(...).map(...).last()` now lower to a reusable stream-last sink that
  keeps only the latest semantic output and applies projection only to the
  retained item. Unfiltered stream-last can select the last array child before
  projection; filtered stream-last still scans for the latest matching output.
- **Filtered stream-last shares item field spans**. Root-field predicates and
  projections in `filter(...).map(...).last()` now reuse the same item scan,
  matching the collect/first span-sharing architecture while retaining only one
  selected output.
- **Demand and NDJSON focused validation is green**. Release-mode focused
  suites for chain demand and NDJSON execution pass after the demand-safety,
  byte-extrema, and stream-first changes.
- **NDJSON module validation is green**. The full `io::ndjson` release-mode
  module suite passes after adding stream-last and the first/last source-demand
  changes, covering byte, hinted, tape, and reverse NDJSON paths.
- **Full release workspace validation is green**.
  `cargo test --release --verbose --workspace` passes after the
  stream-first, byte-extrema, and reverse selective-demand changes.
- **Core compile health is clean**. `cargo check -p jetro-core` passes after
  the demand propagation and NDJSON byte-executor changes.

### Builtin hardening

- **Builtin names and aliases are now collision-checked**. Registry tests prove
  every canonical builtin name and alias resolves back to exactly one
  `BuiltinMethod`, preventing silent lookup drift as the builtin catalog grows.
- **Builtin spec metadata has baseline invariant checks**. Registry tests now
  prove every builtin exposes a finite non-negative planner cost and that
  numeric reducer metadata cannot drift away from numeric sink metadata.
- **Terminal sink demand is checked for every sink builtin**. Registry tests
  now validate count, numeric, approximate-distinct, first, and last sink
  accumulators against the shared `Demand` model used by planners and
  executors.
- **Logical pipeline shapes must participate in demand propagation**. Registry
  tests now assert that every builtin exposed to logical pipeline lowering also
  publishes demand metadata, keeping logical planning and demand planning tied
  together.

## 0.5.9

### Release focus

- **Demand/tape execution is now metadata-driven end to end**. This release
  moves more query behavior out of handwritten shape fusions and into shared
  builtin metadata, planner demand propagation, source capabilities, and common
  executor paths. New builtin execution facts cover logical shape, lowering,
  sink/reducer behavior, view support, order/cardinality effects,
  materialization policy, and demand behavior.
- **Cold-path performance is restored and guarded**. The release benchmark
  suite is back near native Rust for the showcase workloads after fixing a
  view-kernel VM allocation regression. The current `bench_cold` profile keeps
  most cases around 1.0x-1.4x native, with the README showcase around 1.17x in
  the latest validation run.
- **Documentation and benchmark coverage expanded**. The README/showcase was
  refreshed, `bench_cold` covers more representative chains, and a Go benchmark
  harness was added for cross-runtime comparison. The release also includes the
  updated logo asset.

### NDJSON per-row execution

- **`JetroEngine` NDJSON APIs**. Added `run_ndjson`, `collect_ndjson`, and
  `for_each_ndjson` for evaluating one query independently against each
  non-empty NDJSON row. `run_ndjson` writes one JSON result per output line.
  File-path helpers are available for the common cold-path case, with matching
  options-aware variants.
- **Cold-path friendly row execution**. NDJSON rows now enter through the
  engine byte parser, reuse one prepared byte-backed query plan for the whole
  stream, remain lazy/tape-eligible until execution needs materialization, and
  execute with the engine-owned VM instead of performing a per-row plan lookup.
- **Bounded, row-aware input handling**. The reader supports empty input,
  blank-line skipping, CRLF, trailing-newline-less final rows, first-line UTF-8
  BOM stripping, configurable maximum line length, and row-numbered invalid
  JSON errors. `NdjsonOptions` also exposes initial row-buffer sizing for
  callers that know their typical row width.
- **Lower-copy line scanning**. The per-row driver uses `fill_buf` plus
  `memchr` to find line boundaries and transfers the owned row buffer directly
  into JSON parsing, avoiding an extra row byte copy while preserving reusable
  scanner-buffer capacity. `run_ndjson` serializes internal `Val` results
  directly instead of building an intermediate `serde_json::Value` tree.
- **Reverse file scans**. Added tail-to-head NDJSON file helpers backed by a
  chunked `memrchr` reverse reader. `run_ndjson_rev` and
  `collect_ndjson_rev` reuse the same prepared byte-backed plan and direct
  `Val` serialization path as forward per-row execution, with configurable
  reverse chunk sizing and maximum line-length enforcement.
- **Source-dispatch helpers**. Added `NdjsonSource` plus source-based engine
  helpers so callers can route file paths and existing `BufRead` inputs through
  one API while preserving the same options-aware per-row execution paths.
  Callback-based per-row iteration is also available through
  `for_each_ndjson_source`.
- **Early-stop NDJSON matching**. Added `for_each_ndjson_until` plus
  `collect_ndjson_matches*` and `run_ndjson_matches*` APIs for reader, file,
  source-dispatch, and reverse-file inputs. Match helpers evaluate a predicate
  per row, emit the original full row only for truthy matches, and stop as soon
  as the requested match limit is reached, without exposing stream-as-array
  semantics.
- **Public facade exports**. The top-level `jetro` crate now re-exports
  `JetroEngine`, `JetroEngineError`, and `io` so applications can use NDJSON
  APIs directly from the crate they install.
- **Reverse query callbacks**. Added `for_each_ndjson_rev*` APIs so arbitrary
  reverse NDJSON queries can stop through `NdjsonControl` while staying on the
  same byte/tape row execution path as `run_ndjson_rev`.
- **Writer-limit APIs and faster output**. Added writer-based forward, source,
  file, and reverse NDJSON limit helpers so callers can stop after N emitted
  query results without routing through callback `serde_json::Value`
  materialization. NDJSON output now uses a shared direct writer for scalars,
  arrays, objects, small objects, typed lanes, and object-vector rows, with a
  fast no-escape string path and larger options-driven buffering.
- **Raw match-row emission**. `run_ndjson_matches*` and reverse match writer
  APIs now write retained row bytes directly for truthy matches instead of
  materializing the full root value before serialization, preserving original
  row formatting and reducing matched-row overhead.
- **Core NDJSON benchmark**. Added `bench_ndjson` as a core-only cold-path
  benchmark for simple field extraction, array length, nested first access,
  nested mapping, and filter/count-style row queries.
- **Planner-derived NDJSON tape plans**. Common row-local shapes now execute
  directly on reusable simd-json tape scratch: root paths, scalar path calls,
  first/last/nth child access, path maps, filtered maps, filtered counts,
  numeric reducers, and match-limited predicates. These optimizations are
  selected from the physical plan and pipeline kernels rather than handwritten
  query strings, preserving the same fallback semantics for unsupported chains.
- **NDJSON direct-plan cleanup**. Direct tape plan metadata now lives in a
  dedicated internal module, with planner construction separated from row
  driving and tape writing. Focused coverage was added for direct
  first/last/nth element projections.
- **NDJSON direct executor cleanup**. Pure direct tape plans no longer lock the
  engine VM, scalar-call fallback avoids redundant path lookup, and map,
  filtered-map, filtered-count, and numeric reducer paths share one
  array-or-single source traversal helper.
- **Schema-adaptive NDJSON paths and projections**. Direct NDJSON execution now
  caches verified tape-node deltas for stable object field layouts, including
  nested paths, while falling back safely when row field order changes. Static
  object and array projections with path, literal, and view-scalar values write
  directly from tape without materializing per-row `Val` objects.
- **NDJSON projection benchmark coverage**. The core NDJSON benchmark now
  includes object, scalar-call object, and array projection cases alongside
  path, filter, reducer, and match workloads.
- **Wider NDJSON direct projection coverage**. Direct tape execution now covers
  rooted scalar calls, object item methods (`keys`, `values`, `entries`),
  array/object literals inside `map` and `filter(...).map`, and scalar calls on
  first/last/nth array-element receivers. Rooted benchmark query shapes are
  covered by direct-plan tests.
- **Reverse NDJSON direct query writers**. `run_ndjson_rev*` now uses the same
  direct tape row writer as forward NDJSON for eligible queries, including
  reverse limit execution, instead of materializing each row through the
  generic executor.
- **Byte-level NDJSON row scanning for simple projected shapes**. Root field
  projections, root string case calls, root object item methods, and simple
  first/last/nth root-array element projections can now emit directly from row
  bytes and fall back to the tape writer only when the row requires full JSON
  interpretation. This keeps common cold-path CLI projections out of both
  `serde_json::Value` and simd-json tape construction.
- **Rooted NDJSON queries stay on direct byte/tape plans**. NDJSON direct
  planning now treats rooted row-local forms such as `$.id`,
  `$.name.upper()`, and `$.attributes.first().value` equivalently to their
  bare per-row forms, so CLI and API callers get the same fast path without
  rewriting queries.
- **Rooted NDJSON normalization now prefers row-local plans**. Direct NDJSON
  planning normalizes rooted row expressions before accepting a document-root
  fallback, fixing `$.id`/`$.name` style CLI queries so they emit row fields
  instead of `null` while preserving the bare `id`/`name` behavior.
- **Static NDJSON projections can write directly from bytes**. Direct byte
  execution now covers static path projections such as `$.id` and `$.a.b.c`,
  plus simple static object and array shaping such as
  `{test: $.a.b.c, b: $.a.b}` and `[$.id, $.name]`. These plans emit selected
  raw subvalues directly from row bytes and avoid per-row tape or `Val`
  materialization when the shape is safe.
- **Demand-aware byte access for first/nth array elements**. Direct byte
  execution no longer scans the entire root array field before satisfying
  `first()` or `nth()` element projections; it reads only the demanded prefix
  and keeps `last()` on the full/reverse-aware path where the end is required.
- **Byte-level NDJSON match predicates**. Forward match writers and callback
  match APIs can now evaluate direct predicates from raw row bytes for simple
  paths, scalar calls, comparisons, boolean combinations, and first/last/nth
  scalar predicates, falling back to tape evaluation only when the row shape
  requires it.
- **Filtered-count byte fallback**. Direct filtered-count tape plans can count
  supported row-local predicates from bytes before falling back to the shared
  tape writer, preserving the existing physical-plan-driven selection model.
- **10-query CLI NDJSON benchmark now clears the 10x target**. On the
  4.76M-row `/tmp/bench.sh` suite, rebuilt `jetrocli` measured every listed
  rooted NDJSON query above 10x faster than `jaq`; representative timings were
  `$.id` 0.44s vs 28.73s, `$.attributes.first().value` 0.66s vs 29.09s,
  `$.attributes.map([@.key, @.value])` 3.00s vs 54.90s, and
  `$.attributes.filter(@.value.contains("_3")).len()` 1.58s vs 49.34s.

### Demand/tape architecture cleanup

- **Demand metadata stays in planner/executor APIs**. Pipeline bodies now expose
  propagated source and pull demand directly, and segment pull demand is shared
  through the pipeline IR instead of reimplemented by view runners.
- **View source access is capability-driven**. Indexed, reverse, and bounded
  forward access selection now lives with `SourceCapabilities`, including safe
  demotion when a view prefix can change cardinality.
- **View fallback consumes propagated demand**. Generic view-prefix fallback now
  carries bounded demand into the borrowed prefix before materializing suffix
  rows, preserving lazy behavior for safe fallback boundaries.
- **Access modes carry their own bounds**. Reverse and bounded-forward view
  execution honor the selected `SourceAccessMode` output/input counts directly,
  keeping the access plan self-contained.
- **Terminal sink and stage construction cleanup**. Logical/pipeline lowering
  now shares constructors for positional, predicate, membership, arg-extreme,
  count, numeric, and keyed reducer sinks, reducing handwritten builtin
  classification drift.
- **Root positional calls lower statically**. Direct root calls such as
  `$.take(n)` and `$.skip(n)` now decode their static numeric arguments into
  `BuiltinCall` metadata instead of falling back to VM-only execution. This
  makes the same positional facts visible to source demand planning and owned
  value execution.
- **Array RHS `has` is explicit and bounded**. `lhs has [a, b]` now lowers to
  a `has_all` builtin with pre-normalized literal needles, so strings require
  every substring, arrays require every element, and objects require every key.
  Non-literal array RHS forms are rejected instead of silently matching every
  string via empty-substring containment.
- **Nested pipeline plans are first-class execution units**. Nested collection
  maps such as `items.map(...).sum()` now carry their source, stage
  expressions, stage kernels, and sink kernels in the shared plan object.
  Composed and legacy execution both reuse prepared nested-plan metadata across
  rows, while scalar method-chain projections remain ordinary maps so existing
  demand substitution and late projection still apply.
- **Execution VM state is instance-owned**. `Jetro` now keeps its VM cache on
  the document handle instead of using a crate-level thread-local, and planned
  view/composed/tape-row execution paths reuse caller-provided VM state rather
  than allocating private hot-path VMs. A scalar `len()` lowering regression was
  also fixed so string length filters remain view-native scalar calls instead
  of being misclassified as nested array counts.

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

### New builtin: `fold` / `reduce`

- **`fold(init, fn)` and `fold(fn)`**. Left fold returning a single
  value — same loop as `accumulate` but emits only the final acc, no
  intermediate trace. The 1-arg form seeds the accumulator from the
  first element (Iterator::reduce); empty array with no init returns
  `null`. `reduce` is registered as an alias.

  ```jetro
  $.xs.fold(0, (a, b) => a + b)            # → final sum
  $.orders.fold(0, (acc, o) => acc + o.total)
  $.orders.fold({total: 0, n: 0}, (a, o) =>
    {total: a.total + o.total, n: a.n + 1})
  ```

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
