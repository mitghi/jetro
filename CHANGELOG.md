# Changelog

## 0.4.0 — 2026-04-25

### Grammar (breaking)

- **Postfix `?` optional marker.** The null-safety marker now attaches to the
  step it guards, never the next step. Prefix `?.field` / `?.method()` no
  longer parses.
  - `.field?` → null-propagate (same end-result as prior `?.field` on the
    next step).
  - `..descendant?` → still returns an array; `?` is null-safety, not
    first-of-array. Use `.first()` explicitly to extract one:
    `$..services?.first()`.
  - `.method()?` → null-safe method call.
  - `!` (exactly-one quantifier) unchanged.

  Existing query strings that relied on visually writing `?.name` still
  parse — the grammar now interprets the `?` as attached to the field
  before it instead of the one after. End behaviour converges for the
  common null-propagation case.

### New opcodes (VM fusions)

Map-body pattern-specialisations. All are peephole-detected at compile
time; no user-visible API change.

- `MapUpperReplaceLit` / `MapLowerReplaceLit` —
  `map(@.upper().replace(lit, lit))` / lower variant. Single-pass ASCII
  case-fold + memchr needle match into a pre-sized `Arc<[u8]>`.
- `MapStrConcat { prefix, suffix }` —
  `map(prefix + @ + suffix)` in three shapes. Exact-size buffer + one
  Arc alloc per row.
- `MapSplitLenSum { sep }` —
  `map(@.split(sep).map(len).sum())`. Emits `IntVec`. ASCII path:
  memchr / memmem hit-count. Unicode: char-count - hits × sep-chars.
- `MapProject` —
  `map({k1, k2, ..})` with all-`Short` entries. Emits columnar
  `Val::ObjVec`; hoists per-row hashtable allocation.
- `MapStrSlice { start, end }` —
  `map(@.slice(lit, lit))`. ASCII path: zero-alloc `Val::StrSlice` /
  `Val::StrSliceVec` borrowed view into parent Arc.
- `MapFString(parts)` —
  `map(f"...")`. Emits columnar `Val::StrVec` output; skips per-row
  CallMethod dispatch.
- `CountByField(k)` / `UniqueByField(k)` —
  `count_by(k)` / `unique_by(k)` with trivial field key. Direct
  `obj.get(k)` per row instead of lambda dispatch.

### New Val variants

- **`Val::StrSlice(StrRef)`** — borrowed slice into a parent `Arc<str>`.
  Cloning = atomic Arc bump + two u32 offset copies; no heap alloc.
  Emitted by `slice`, `substring`, `split.first` to avoid per-row
  `Arc<str>` allocation.
- **`Val::StrSliceVec(Arc<Vec<StrRef>>)`** — columnar lane of borrowed
  slices. Drops per-row Val enum tag; serializes directly via `ValRef`.
- **`Val::ObjSmall(Arc<[(Arc<str>, Val)]>)`** — inline-object variant
  (flat key-value slice, no hashtable). Used where hashtable build per
  row would dominate.
- **`Val::ObjVec(Arc<ObjVecData>)`** — columnar array-of-objects. Shared
  `keys: Arc<[Arc<str>]>` with `rows: Vec<Vec<Val>>`. One Arc per array
  instead of N Arcs per N rows.

### Other

- **exec_fstring fast paths** — pre-size output `String::with_capacity`;
  per-interp skip `val_to_string`'s temporary String (push_str direct
  for `Val::Str`, `write!` for numeric, `"null"` for null);
  `Arc::<str>::from(String)` transfers buffer instead of reallocating.
  Small interp sub-programs (`[PushCurrent]`, `[PushCurrent, GetIndex]`,
  `[PushCurrent, GetField]`, `[LoadIdent]`) fast-path without full
  `self.exec()` recursion.
- **Zero-alloc MapStrVec Upper/Lower/Trim** — ASCII nop path reuses
  input `Arc`; mut path uses `String::from_utf8_unchecked` +
  `Arc::<str>::from(String)` to transfer buffer.
- **Deserialize key intern** — per-thread `HashMap<Box<str>, Arc<str>>`
  (cap 4096) collapses repeated-key allocs during
  `Val::from(&serde_json::Value)` and the custom `Deserialize` path.

### Benchmarks (v0.3.0 → 0.4.0, N=10k iters=100)

**bench_smallstr** (native = hand-written Rust+serde = 1.0x):

| workload | 0.3.0 | 0.4.0 |
|---|---|---|
| group_by short | 8.42x | **0.85x** (jetro wins) |
| unique_by grp | 7.35x | 1.71x |
| project {id,grp} | 6.88x | 2.49x |
| fstring.short | 1.87x | 1.59x |

**bench_strings**:

| workload | 0.3.0 | 0.4.0 |
|---|---|---|
| split.count | ~3x | **0.77x** (jetro wins) |
| concat | 1.99x | **0.75x** (jetro wins) |
| split.map(len).sum | 5.64x | **0.51x** (jetro wins) |
| upper+replace | 4.54x | 1.13x |
| replace_all | 5.40x | 1.22x |
| slice | 9.73x | 2.34x |
| split-rev-join | 3.98x | 1.36x |
| f-string | 3.53x | 2.71x |
| upper+trim | 3.96x | 1.75x |

**bench_all_native** (opt_x_2): all workloads within 1.15x, 5 jetro wins
(add / min-max / kv / kv-update / to-fromjson).

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
