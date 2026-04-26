//! Borrowed `Val<'a>` substrate — foundation for cold-start
//! optimisation per `cold_first_roadmap.md` Tier 1 #1.
//!
//! ## Purpose
//!
//! On a cold call (`Jetro::from_simd(bytes).collect_val(query)`), the
//! current `Val` (Arc-wrapped) builder allocates ~50,000 small heap
//! objects for a 1.7 MB / 5000-record document:
//!
//!   - one `Arc<str>` per object key (~5 keys × 5000 rows = 25k Arcs),
//!   - one `Arc<IndexMap<Arc<str>, Val>>` per row (~5k Arcs),
//!   - one `Arc<Vec<Val>>` per array (~few k Arcs),
//!   - per-string content `Arc::from(&str)` allocs.
//!
//! At ~50-75 ns per `Arc::new`/`Box::new`/`Vec::with_capacity` on
//! macOS jemalloc, that's ~3-4 ms of pure allocator tax — about 25%
//! of the cold-call budget.
//!
//! `Val<'a>` mirrors the existing `Val` shape but stores byte content
//! and structural slices as borrows into a per-handle bumpalo `Bump`
//! arena.  All compound nodes share a single arena lifetime → drop
//! is O(1) (arena reset), no per-node refcount, no IndexMap rebuild.
//!
//! ## Status: FOUNDATION ONLY
//!
//! This module ships the type definitions, arena wrapper, and
//! conversion functions.  It is NOT YET wired into the eval / vm /
//! pipeline runners.  Subsequent migration is staged:
//!
//!   - **Phase 1** (this commit): `Val<'a>` skeleton + `Arena` +
//!     converters.  Round-trip equivalence tests.  Zero perf impact.
//!
//!   - **Phase 2**: `Jetro::collect_val_arena()` API that builds
//!     `Val<'a>` directly from simd-json bytes via a borrowed
//!     parser (`Val::from_json_simd_arena`).  Outputs converted to
//!     owned `Val` at boundary for compat.  Bench gate: ≥ neutral
//!     on bench_cold; rollback if regression.
//!
//!   - **Phase 3**: pipeline / bytescan / composed substrates accept
//!     `Val<'a>` source.  Reducer outputs in-arena.  Convert to
//!     owned `Val` only at the public API boundary.  Largest cold-
//!     bench gain expected here (~3-4 ms across hardest queries).
//!
//!   - **Phase 4**: public `Jetro::collect_val_borrow<'h>(&'h self,
//!     expr) -> Val<'h>` API for callers that can hold the handle
//!     alive.  Avoids the boundary conversion entirely.
//!
//! ## Lifetime model
//!
//! The arena outlives every `Val<'a>` it produces.  All borrows in
//! `Val<'a>` (str, slice) point into arena memory — safe to copy /
//! pass around within the arena's scope.  At handle drop, the arena
//! drops, all `Val<'a>` references die together (typical bumpalo
//! pattern).
//!
//! Conversion `Val<'a> -> Val` (owned) deep-clones into Arc-allocated
//! values — used at the API boundary so callers can keep the result
//! after the handle drops.
//!
//! ## Why this isn't an arena-only fix
//!
//! Wrapping `Arc::new` over a `bumpalo::Bump` doesn't change `Val`'s
//! shape — same Arc tag check, same IndexMap allocations on each
//! row, same per-key string copy.  Arc + arena would gain ~1-2 ms
//! at most.  The structural change (drop Arc, drop IndexMap, drop
//! per-string heap allocation) is what unlocks the ~5-7 ms cold-
//! bench saving documented in `cold_first_roadmap.md`.

use bumpalo::Bump;
use std::cell::Cell;

// ── Arena wrapper ──────────────────────────────────────────────────

/// Per-handle arena for `Val<'a>` storage.  Wraps bumpalo's `Bump`
/// with safe allocation helpers.
///
/// Lifetime: `'arena` — `Val<'arena>` references live as long as the
/// `Arena` itself.  Drop the arena → all `Val<'arena>` invalidate
/// (statically tracked by Rust's borrow checker).
///
/// `bytes_allocated` provides a soft observability hook for telemetry
/// + bench validation; cheap atomic-style read on a Cell.
pub struct Arena {
    bump: Bump,
    /// Best-effort live byte count — incremented on every alloc helper
    /// call.  Used for tests + bench instrumentation.
    bytes_allocated: Cell<usize>,
}

impl Arena {
    pub fn new() -> Self {
        Self { bump: Bump::new(), bytes_allocated: Cell::new(0) }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self { bump: Bump::with_capacity(cap), bytes_allocated: Cell::new(0) }
    }

    /// Reset the arena.  All `Val<'a>` references become dangling —
    /// callers must ensure none are reachable.  Safe because Rust's
    /// borrow checker enforces lifetime exclusivity through `&mut self`.
    pub fn reset(&mut self) {
        self.bump.reset();
        self.bytes_allocated.set(0);
    }

    pub fn bytes_allocated(&self) -> usize {
        self.bytes_allocated.get()
    }

    /// Allocate a string slice in the arena.  Returns a `&'arena str`
    /// borrowing the arena bytes — zero refcount, zero further
    /// allocations.  ~3-5 ns per call vs ~75 ns for `Arc::from(&str)`.
    pub fn alloc_str<'arena>(&'arena self, s: &str) -> &'arena str {
        let out = self.bump.alloc_str(s);
        self.bytes_allocated.set(self.bytes_allocated.get() + s.len());
        out
    }

    /// Allocate a slice of values in the arena.  Used for `Val::Arr`
    /// and `Val::Obj` payloads.  Per-call cost = single bump-pointer
    /// move + memcpy, no per-element work.
    pub fn alloc_slice_copy<'arena, T: Copy>(&'arena self, src: &[T]) -> &'arena [T] {
        let out = self.bump.alloc_slice_copy(src);
        self.bytes_allocated.set(self.bytes_allocated.get() + src.len() * std::mem::size_of::<T>());
        out
    }

    /// Allocate a slice of values in the arena, consuming an iterator.
    /// Used when building `Val::Arr` / `Val::Obj` row-by-row.
    pub fn alloc_slice_fill_iter<'arena, T, I>(&'arena self, iter: I) -> &'arena mut [T]
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: ExactSizeIterator,
    {
        self.bump.alloc_slice_fill_iter(iter)
    }

    /// Allocate a single value of type `T` in the arena.  Useful for
    /// `Val<'a>::Box(...)` patterns; rarely needed for the core enum
    /// since the variants embed their payloads directly.
    pub fn alloc<'arena, T>(&'arena self, value: T) -> &'arena mut T {
        self.bump.alloc(value)
    }
}

impl Default for Arena {
    fn default() -> Self { Self::new() }
}

impl std::fmt::Debug for Arena {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Arena")
            .field("bytes_allocated", &self.bytes_allocated.get())
            .finish()
    }
}

// ── Borrowed Val ────────────────────────────────────────────────────

/// Borrowed value type — every compound node references slices in
/// the arena.  Cheap to copy (16-24 bytes), no refcount.
///
/// Mirrors the variants of `eval::value::Val` minus the columnar
/// lanes (`IntVec`, `FloatVec`, `StrVec`, `StrSliceVec`, `ObjVec`).
/// Those stay on the owned `Val` substrate; conversion is a row-by-
/// row materialise during `Val<'a> -> Val` at the boundary.  If
/// future bench data shows columnar lanes are bench-critical for
/// the borrowed path, equivalent variants land here later.
///
/// `Obj` is a flat `&[(&str, Val)]` slice — replaces
/// `Arc<IndexMap<Arc<str>, Val>>`.  Lookup is linear (acceptable
/// for typical N=5-20 keys; matches existing `ObjSmall` ergonomics).
/// IndexMap-style insertion order preserved by slice order.
#[derive(Clone, Copy, Debug)]
pub enum Val<'a> {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(&'a str),
    Arr(&'a [Val<'a>]),
    Obj(&'a [(&'a str, Val<'a>)]),
}

impl<'a> Val<'a> {
    pub fn is_null(&self) -> bool { matches!(self, Val::Null) }
    pub fn is_bool(&self) -> bool { matches!(self, Val::Bool(_)) }
    pub fn is_int(&self) -> bool { matches!(self, Val::Int(_)) }
    pub fn is_float(&self) -> bool { matches!(self, Val::Float(_)) }
    pub fn is_string(&self) -> bool { matches!(self, Val::Str(_)) }
    pub fn is_array(&self) -> bool { matches!(self, Val::Arr(_)) }
    pub fn is_object(&self) -> bool { matches!(self, Val::Obj(_)) }

    pub fn as_str(&self) -> Option<&'a str> {
        match self { Val::Str(s) => Some(*s), _ => None }
    }
    pub fn as_int(&self) -> Option<i64> {
        match self { Val::Int(n) => Some(*n), _ => None }
    }
    pub fn as_float(&self) -> Option<f64> {
        match self { Val::Float(f) => Some(*f), Val::Int(n) => Some(*n as f64), _ => None }
    }
    pub fn as_array(&self) -> Option<&'a [Val<'a>]> {
        match self { Val::Arr(a) => Some(*a), _ => None }
    }
    pub fn as_object(&self) -> Option<&'a [(&'a str, Val<'a>)]> {
        match self { Val::Obj(o) => Some(*o), _ => None }
    }

    /// Linear-scan field lookup.  Matches `IndexMap` insertion order;
    /// returns the first occurrence (consistent with IndexMap semantics
    /// when keys are unique, which they always are post-parse).
    pub fn get_field(&self, key: &str) -> Option<Val<'a>> {
        match self {
            Val::Obj(entries) => {
                for (k, v) in entries.iter() {
                    if *k == key { return Some(*v); }
                }
                None
            }
            _ => None,
        }
    }

    /// Walk a multi-step path; stops at the first non-Object boundary.
    pub fn walk_path(&self, chain: &[&str]) -> Option<Val<'a>> {
        let mut cur = *self;
        for key in chain {
            cur = cur.get_field(key)?;
        }
        Some(cur)
    }
}

// ── Builders ────────────────────────────────────────────────────────

impl<'a> Val<'a> {
    /// Build a `Val::Str` from a string copied into the arena.
    pub fn str_in(arena: &'a Arena, s: &str) -> Val<'a> {
        Val::Str(arena.alloc_str(s))
    }

    /// Build a `Val::Arr` from a slice of values, copied into arena.
    pub fn arr_in(arena: &'a Arena, items: &[Val<'a>]) -> Val<'a> {
        Val::Arr(arena.alloc_slice_copy(items))
    }

    /// Build a `Val::Obj` from a slice of `(key, value)` pairs.
    /// Both keys and entries copied into arena; values must already
    /// reference arena memory (or be primitives).
    pub fn obj_in(arena: &'a Arena, entries: &[(&str, Val<'a>)]) -> Val<'a> {
        let mut out: Vec<(&'a str, Val<'a>)> = Vec::with_capacity(entries.len());
        for (k, v) in entries {
            out.push((arena.alloc_str(k), *v));
        }
        let slice = arena.alloc_slice_fill_iter(out.into_iter());
        // SAFETY: `alloc_slice_fill_iter` returns `&mut [T]`; reborrow
        // as immutable.  Lifetime tied to arena.
        Val::Obj(&*slice)
    }
}

// ── Conversion: borrowed -> owned ───────────────────────────────────

impl<'a> Val<'a> {
    /// Deep-convert into an owned `crate::eval::Val`.  Used at the
    /// API boundary where the result outlives the arena (e.g. the
    /// public `collect_val()` API).
    pub fn to_owned_val(&self) -> crate::eval::Val {
        use crate::eval::Val as Owned;
        use std::sync::Arc;
        match self {
            Val::Null => Owned::Null,
            Val::Bool(b) => Owned::Bool(*b),
            Val::Int(n) => Owned::Int(*n),
            Val::Float(f) => Owned::Float(*f),
            Val::Str(s) => Owned::Str(Arc::from(*s)),
            Val::Arr(items) => {
                let v: Vec<Owned> = items.iter().map(|x| x.to_owned_val()).collect();
                Owned::Arr(Arc::new(v))
            }
            Val::Obj(entries) => {
                let mut m: indexmap::IndexMap<Arc<str>, Owned> =
                    indexmap::IndexMap::with_capacity(entries.len());
                for (k, v) in entries.iter() {
                    m.insert(Arc::from(*k), v.to_owned_val());
                }
                Owned::Obj(Arc::new(m))
            }
        }
    }
}

// ── Conversion: owned -> borrowed ───────────────────────────────────

/// Deep-clone an owned `Val` into the arena, producing a `Val<'a>`.
/// Used at ingestion boundaries where existing owned-Val data needs
/// to feed the borrowed pipeline (e.g. `Jetro::new(serde_json::Value)`
/// path, or test fixtures).
///
/// Cost is the same as building a fresh `Val<'a>`: O(N) allocator
/// pumps + memcpy.  No reuse of source `Arc`s — borrowed Val owns
/// its bytes via the arena.
pub fn from_owned<'a>(arena: &'a Arena, v: &crate::eval::Val) -> Val<'a> {
    use crate::eval::Val as Owned;
    match v {
        Owned::Null => Val::Null,
        Owned::Bool(b) => Val::Bool(*b),
        Owned::Int(n) => Val::Int(*n),
        Owned::Float(f) => Val::Float(*f),
        Owned::Str(s) => Val::Str(arena.alloc_str(s)),
        Owned::StrSlice(sr) => Val::Str(arena.alloc_str(sr.as_str())),
        Owned::Arr(a) => {
            let items: Vec<Val<'a>> = a.iter().map(|x| from_owned(arena, x)).collect();
            Val::Arr(arena.alloc_slice_copy(&items))
        }
        Owned::IntVec(v) => {
            let items: Vec<Val<'a>> = v.iter().map(|n| Val::Int(*n)).collect();
            Val::Arr(arena.alloc_slice_copy(&items))
        }
        Owned::FloatVec(v) => {
            let items: Vec<Val<'a>> = v.iter().map(|f| Val::Float(*f)).collect();
            Val::Arr(arena.alloc_slice_copy(&items))
        }
        Owned::StrVec(v) => {
            let items: Vec<Val<'a>> = v.iter()
                .map(|s| Val::Str(arena.alloc_str(s)))
                .collect();
            Val::Arr(arena.alloc_slice_copy(&items))
        }
        Owned::StrSliceVec(v) => {
            let items: Vec<Val<'a>> = v.iter()
                .map(|sr| Val::Str(arena.alloc_str(sr.as_str())))
                .collect();
            Val::Arr(arena.alloc_slice_copy(&items))
        }
        Owned::Obj(m) => {
            let mut out: Vec<(&'a str, Val<'a>)> = Vec::with_capacity(m.len());
            for (k, v) in m.iter() {
                out.push((arena.alloc_str(k), from_owned(arena, v)));
            }
            let slice = arena.alloc_slice_fill_iter(out.into_iter());
            Val::Obj(&*slice)
        }
        Owned::ObjSmall(entries) => {
            let mut out: Vec<(&'a str, Val<'a>)> = Vec::with_capacity(entries.len());
            for (k, v) in entries.iter() {
                out.push((arena.alloc_str(k), from_owned(arena, v)));
            }
            let slice = arena.alloc_slice_fill_iter(out.into_iter());
            Val::Obj(&*slice)
        }
        Owned::ObjVec(d) => {
            // Materialise rows from columnar layout.  Each row is an
            // Object with `keys` × `cells[row * stride + i]`.
            let stride = d.keys.len();
            let nrows = if stride == 0 { 0 } else { d.cells.len() / stride };
            let mut rows: Vec<Val<'a>> = Vec::with_capacity(nrows);
            for row in 0..nrows {
                let mut entries: Vec<(&'a str, Val<'a>)> = Vec::with_capacity(stride);
                for (i, key) in d.keys.iter().enumerate() {
                    let cell = &d.cells[row * stride + i];
                    entries.push((arena.alloc_str(key), from_owned(arena, cell)));
                }
                let slice = arena.alloc_slice_fill_iter(entries.into_iter());
                rows.push(Val::Obj(&*slice));
            }
            Val::Arr(arena.alloc_slice_copy(&rows))
        }
    }
}

// ── Direct simd-json → Val<'a> builder (Phase 2) ────────────────────

/// Parse a JSON byte buffer directly into `Val<'a>` allocated in
/// `arena`.  Bypasses the owned-`Val` builder entirely — every key,
/// string, array, and object lands in arena memory.  No `Arc`, no
/// `IndexMap`, no per-row heap calls.
///
/// Uses simd-json's tape representation as the parser front-end
/// (same engine as owned `Val::from_json_simd`); only the conversion
/// step changes.
///
/// `bytes` is mutated in place by simd-json (its parser is destructive
/// for SIMD reasons).  Caller must pass a writable buffer.
#[cfg(feature = "simd-json")]
pub fn from_json_simd_arena<'a>(arena: &'a Arena, bytes: &mut [u8]) -> Result<Val<'a>, String> {
    let tape = simd_json::to_tape(bytes).map_err(|e| e.to_string())?;
    let nodes = tape.0;
    let mut idx = 0usize;
    Ok(walk_simd_tape(arena, &nodes, &mut idx))
}

#[cfg(feature = "simd-json")]
fn walk_simd_tape<'a>(
    arena: &'a Arena,
    nodes: &[simd_json::Node<'_>],
    idx: &mut usize,
) -> Val<'a> {
    use simd_json::Node;
    use simd_json::StaticNode as SN;
    let here = nodes[*idx];
    *idx += 1;
    match here {
        Node::Static(SN::Null)    => Val::Null,
        Node::Static(SN::Bool(b)) => Val::Bool(b),
        Node::Static(SN::I64(n))  => Val::Int(n),
        Node::Static(SN::U64(n))  => {
            if n <= i64::MAX as u64 { Val::Int(n as i64) } else { Val::Float(n as f64) }
        }
        Node::Static(SN::F64(f))  => Val::Float(f),
        Node::String(s)           => Val::Str(arena.alloc_str(s)),
        Node::Array { len, .. } => {
            // Build elements into a temporary Vec; bulk-copy into arena
            // at the end.  Bumpalo's slice builders need known length
            // up-front, and per-element length is variable due to nesting.
            let mut tmp: Vec<Val<'a>> = Vec::with_capacity(len);
            for _ in 0..len {
                tmp.push(walk_simd_tape(arena, nodes, idx));
            }
            Val::Arr(arena.alloc_slice_copy(&tmp))
        }
        Node::Object { len, .. } => {
            let mut tmp: Vec<(&'a str, Val<'a>)> = Vec::with_capacity(len);
            for _ in 0..len {
                let key = match nodes[*idx] {
                    Node::String(s) => arena.alloc_str(s),
                    _ => unreachable!("object key must be string"),
                };
                *idx += 1;
                let v = walk_simd_tape(arena, nodes, idx);
                tmp.push((key, v));
            }
            let slice = arena.alloc_slice_fill_iter(tmp.into_iter());
            Val::Obj(&*slice)
        }
    }
}

// ── Round-trip tests ────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::Val as Owned;
    use std::sync::Arc;
    use indexmap::IndexMap;

    fn owned_obj(pairs: &[(&str, Owned)]) -> Owned {
        let mut m: IndexMap<Arc<str>, Owned> = IndexMap::with_capacity(pairs.len());
        for (k, v) in pairs { m.insert(Arc::from(*k), v.clone()); }
        Owned::Obj(Arc::new(m))
    }

    #[test]
    fn primitives_roundtrip() {
        let arena = Arena::new();
        for owned in &[
            Owned::Null, Owned::Bool(true), Owned::Bool(false),
            Owned::Int(42), Owned::Int(-1), Owned::Float(3.14),
            Owned::Str(Arc::from("hello")),
        ] {
            let borrowed = from_owned(&arena, owned);
            let back = borrowed.to_owned_val();
            assert_eq!(format!("{:?}", owned), format!("{:?}", back),
                "round-trip mismatch for {:?}", owned);
        }
    }

    #[test]
    fn arr_roundtrip() {
        let arena = Arena::new();
        let owned = Owned::Arr(Arc::new(vec![
            Owned::Int(1), Owned::Str(Arc::from("two")), Owned::Bool(false),
        ]));
        let borrowed = from_owned(&arena, &owned);
        let back = borrowed.to_owned_val();
        assert_eq!(format!("{:?}", owned), format!("{:?}", back));
    }

    #[test]
    fn obj_roundtrip() {
        let arena = Arena::new();
        let owned = owned_obj(&[
            ("a", Owned::Int(1)),
            ("b", Owned::Str(Arc::from("two"))),
            ("c", Owned::Bool(true)),
        ]);
        let borrowed = from_owned(&arena, &owned);
        let back = borrowed.to_owned_val();
        // IndexMap preserves insertion order; debug repr should match exactly.
        assert_eq!(format!("{:?}", owned), format!("{:?}", back));
    }

    #[test]
    fn nested_obj_roundtrip() {
        let arena = Arena::new();
        let inner = owned_obj(&[
            ("city", Owned::Str(Arc::from("NYC"))),
            ("zip", Owned::Int(10001)),
        ]);
        let owned = owned_obj(&[
            ("name", Owned::Str(Arc::from("Alice"))),
            ("addr", inner),
            ("scores", Owned::Arr(Arc::new(vec![Owned::Int(10), Owned::Int(20)]))),
        ]);
        let borrowed = from_owned(&arena, &owned);
        let back = borrowed.to_owned_val();
        assert_eq!(format!("{:?}", owned), format!("{:?}", back));
    }

    #[test]
    fn intvec_floatvec_collapse_to_arr() {
        let arena = Arena::new();
        let iv = Owned::IntVec(Arc::new(vec![1, 2, 3]));
        let borrowed = from_owned(&arena, &iv);
        let back = borrowed.to_owned_val();
        // After conversion the columnar lane materialises as a plain
        // Arr of Int — same JSON shape, distinct internal repr.
        match back {
            Owned::Arr(a) => {
                assert_eq!(a.len(), 3);
                for (i, v) in a.iter().enumerate() {
                    assert!(matches!(v, Owned::Int(n) if *n == (i as i64 + 1)));
                }
            }
            other => panic!("expected Arr, got {:?}", other),
        }
    }

    #[test]
    fn obj_lookup_walk_path() {
        let arena = Arena::new();
        let inner = owned_obj(&[
            ("city", Owned::Str(Arc::from("NYC"))),
        ]);
        let owned = owned_obj(&[("user", owned_obj(&[("addr", inner)]))]);
        let borrowed = from_owned(&arena, &owned);
        let v = borrowed.walk_path(&["user", "addr", "city"]).expect("walk");
        assert_eq!(v.as_str(), Some("NYC"));
    }

    #[test]
    fn arena_grows_on_alloc() {
        let arena = Arena::new();
        assert_eq!(arena.bytes_allocated(), 0);
        let _s1 = arena.alloc_str("hello");
        assert!(arena.bytes_allocated() >= 5);
        let n0 = arena.bytes_allocated();
        let _s2 = arena.alloc_str("world!");
        assert!(arena.bytes_allocated() >= n0 + 6);
    }

    #[test]
    fn val_copy_is_cheap() {
        // `Val<'a>` is `Copy`; verify by passing-by-value compiles.
        let arena = Arena::new();
        let v = Val::str_in(&arena, "hi");
        fn take(_v: Val<'_>) {}
        take(v); take(v); // would not compile if Val weren't Copy
    }

    #[cfg(feature = "simd-json")]
    #[test]
    fn from_json_simd_arena_primitive() {
        let arena = Arena::new();
        let mut bytes: Vec<u8> = b"42".to_vec();
        let v = from_json_simd_arena(&arena, &mut bytes).unwrap();
        assert!(matches!(v, Val::Int(42)));
    }

    #[cfg(feature = "simd-json")]
    #[test]
    fn from_json_simd_arena_object() {
        let arena = Arena::new();
        let mut bytes: Vec<u8> = br#"{"a":1,"b":"two","c":true}"#.to_vec();
        let v = from_json_simd_arena(&arena, &mut bytes).unwrap();
        let entries = v.as_object().expect("object");
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].0, "a");
        assert!(matches!(entries[0].1, Val::Int(1)));
        assert_eq!(entries[1].0, "b");
        assert_eq!(entries[1].1.as_str(), Some("two"));
        assert_eq!(entries[2].0, "c");
        assert!(matches!(entries[2].1, Val::Bool(true)));
    }

    #[cfg(feature = "simd-json")]
    #[test]
    fn from_json_simd_arena_nested() {
        let arena = Arena::new();
        let mut bytes: Vec<u8> = br#"{"data":[{"id":1,"user":{"name":"alice"}},{"id":2,"user":{"name":"bob"}}]}"#.to_vec();
        let v = from_json_simd_arena(&arena, &mut bytes).unwrap();
        let name = v.walk_path(&["data"])
            .and_then(|d| d.as_array())
            .and_then(|a| a.first().copied())
            .and_then(|r| r.walk_path(&["user", "name"]))
            .and_then(|n| n.as_str())
            .expect("walk");
        assert_eq!(name, "alice");
    }

    #[cfg(feature = "simd-json")]
    #[test]
    fn from_json_simd_arena_matches_owned() {
        // Round-trip equivalence: arena-built Val<'a> → owned must
        // structurally match the owned-direct path on the same input.
        let json: Vec<u8> = br#"{"orders":[{"id":1,"items":[{"sku":"A","qty":2}]}]}"#.to_vec();

        let arena = Arena::new();
        let mut buf1 = json.clone();
        let borrowed = from_json_simd_arena(&arena, &mut buf1).unwrap();
        let from_borrowed = borrowed.to_owned_val();

        let mut buf2 = json.clone();
        let direct = crate::eval::Val::from_json_simd(&mut buf2).unwrap();

        // Compare via JSON byte serialisation for shape equality.
        assert_eq!(from_borrowed.to_json_vec(), direct.to_json_vec());
    }
}
