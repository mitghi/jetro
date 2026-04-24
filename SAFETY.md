# Unsafe Code Inventory

jetro uses `unsafe` in three localised string-builder helpers, all for
one reason: allocate an `Arc<str>` of known size with a single heap
allocation, avoiding the `String → Box<str> → Arc<str>` round-trip.

Every unsafe block is covered by `jetro-core/tests/unsafe_invariants.rs`
and runs under Miri via `scripts/miri.sh`.

## Inventory

| Location | Purpose | Invariants |
|---|---|---|
| `vm.rs::ascii_fold_to_arc_str` | ASCII upper/lower → Arc<str> in one alloc | Input is ASCII; output length equals input; `Arc::get_mut` sound on fresh Arc; `Arc<[u8]>` and `Arc<str>` share layout |
| `vm.rs::StrSplitReverseJoin` exec arm | Reverse-join segments into preallocated buffer | `out_len == src.len()` (permutation invariant); segment spans on UTF-8 boundaries; write loop fills exactly `out_len` bytes (`debug_assert_eq!`) |
| `vm.rs::MapReplaceLit` exec arm | Two-pass literal replace into preallocated buffer | Hit-count predicts exact `out_len` via `src.len() + hits*(wlen-nlen)`; `str::find` returns UTF-8 boundaries; all inserted bytes come from valid UTF-8 inputs; final byte count matches (`debug_assert_eq!`) |

`func_strings.rs::upper`/`lower` formerly used unsafe `as_mut_vec`;
replaced with `String::make_ascii_uppercase` (safe, same perf).

## Running Miri

```bash
rustup +nightly component add miri     # one-time
./scripts/miri.sh                       # exercises every unsafe block
```

## Why unsafe, not safe equivalents

The safe path — `String::with_capacity(n) → Arc::<str>::from(s)` —
allocates twice (the `String`'s buffer, then shrink-fit into the
refcount block). Benchmarked 10–20% slower on `MapReplaceLit`
(`replace_all`: 1.18x → ~1.4x native). Since every unsafe block is
localised, documented, covered by Miri-runnable tests, and asserts its
byte-count invariant in debug builds, the cost of safe-only is judged
not worth the perf regression.

## Adding new unsafe

New unsafe blocks must:

1. Live behind an inline-documented SAFETY comment with invariants.
2. Add a test to `unsafe_invariants.rs` exercising boundary cases.
3. Pass `./scripts/miri.sh` cleanly.
4. Assert byte-count or similar structural invariant in `debug_assert!`.

Drop unsafe if a safe equivalent costs ≤5% — localisation is worth
less than the audit burden.
