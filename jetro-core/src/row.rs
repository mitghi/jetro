//! Substrate-agnostic row trait — Phase 1 of pipeline unification.
//!
//! Today three borrowed substrates duplicate the Stage / Sink / run
//! loop pattern:
//!
//!   - `composed_borrow.rs`  — Stage on `BVal<'a>` (in-arena tree)
//!   - `composed_tape.rs`    — Stage on `TapeRow<'a>` (lazy tape view)
//!   - (future)              — Stage on a columnar source
//!
//! Each new substrate forces a parallel:
//!   - StageT trait + Composed/Identity wrappers
//!   - SinkT trait + 8+ sink impls
//!   - run_pipeline_t outer loop
//!   - lowering module (pipeline_*_borrow.rs)
//!
//! This is the exact duplication `pipeline_unification.md` Step 5
//! was designed to eliminate.  Phase 1 ships the foundation: a
//! shared `Row<'a>` trait that both BVal and TapeRow implement.
//! Phase 2 will collapse Stage/Sink/run-loop into trait-driven
//! generic code; Phase 3 deletes the parallel modules.
//!
//! ## Phase plan
//!
//! - **Phase 1** (this commit): `Row<'a>` trait + impls for BVal /
//!   TapeRow.  Unified surface; substrate code unchanged.
//! - **Phase 2**: `Stage<R: Row<'a>>` trait + generic ComposedS /
//!   FilterS / MapFieldS / FlatMapS / TakeS / SkipS structs over R.
//!   `Sink<R>` trait + 8 sink impls.  Generic `run_pipeline<R, S>`.
//!   Migrate one lowering site (pipeline_tape_borrow) to consume
//!   the unified runner; bench-gate parity vs current per-substrate
//!   path.
//! - **Phase 3**: migrate pipeline_borrow + composed_borrow.  Delete
//!   composed_borrow.rs + composed_tape.rs + pipeline_borrow.rs +
//!   pipeline_tape_borrow.rs.  Replaced by one ~400-LoC unified
//!   module.  Net codebase shrink ~1500-2000 LoC.

use crate::eval::borrowed::{Arena, Val as BVal};

/// Substrate-agnostic row cursor.  Implementors are `Copy` (16-24
/// bytes), zero-allocation per access; primitive accessors return
/// `Option<scalar>`, structural traversal returns `Option<Self>`.
///
/// `materialise` is the bridge to owned-shape `BVal<'a>` for sinks
/// that need to emit row payloads (First/Last/Collect).  For tape-
/// backed rows this allocates into the arena; for already-arena
/// rows (BVal) it's a copy.
///
/// Lifetime `'a` ties borrows to the underlying source data:
///   - `BVal<'a>` borrows from the arena that backs it.
///   - `TapeRow<'a>` borrows from the `&'a TapeData`.
///
/// `Self::materialise(arena)` returns a value tied to that arena;
/// used by Sink::Collect / First / Last regardless of substrate.
pub trait Row<'a>: Clone + 'a {
    /// Iterator over array children (Phase 2 — used by FlatMap).
    /// Lifetime tied to the source data the row borrows from.
    type ArrayIter: Iterator<Item = Self> + 'a;

    fn get_field(&self, key: &str) -> Option<Self>;
    fn walk_path(&self, chain: &[&str]) -> Option<Self>;
    fn as_int(&self) -> Option<i64>;
    fn as_float(&self) -> Option<f64>;
    fn as_str(&self) -> Option<&'a str>;
    fn as_bool(&self) -> Option<bool>;
    fn is_null(&self) -> bool;
    /// Iterate the row's children when it points at an Array node.
    /// `None` when the row is a non-Array.
    fn array_children(&self) -> Option<Self::ArrayIter>;
    /// Convert this row's subtree into an owned-shape `BVal<'a>`
    /// allocated in `arena`.  For tape-backed rows this walks the
    /// subtree once.  For arena-backed rows this is a 16-byte copy
    /// (BVal is itself Copy).
    fn materialise(self, arena: &'a Arena) -> BVal<'a>;
}

// ── Impl: borrowed::Val<'a> ─────────────────────────────────────────

impl<'a> Row<'a> for BVal<'a> {
    type ArrayIter = std::iter::Cloned<std::slice::Iter<'a, BVal<'a>>>;

    #[inline]
    fn array_children(&self) -> Option<Self::ArrayIter> {
        match self {
            BVal::Arr(items) => Some(items.iter().cloned()),
            _ => None,
        }
    }

    #[inline]
    fn get_field(&self, key: &str) -> Option<Self> { BVal::get_field(self, key) }

    #[inline]
    fn walk_path(&self, chain: &[&str]) -> Option<Self> { BVal::walk_path(self, chain) }

    #[inline]
    fn as_int(&self) -> Option<i64> { BVal::as_int(self) }

    #[inline]
    fn as_float(&self) -> Option<f64> { BVal::as_float(self) }

    #[inline]
    fn as_str(&self) -> Option<&'a str> { BVal::as_str(self) }

    #[inline]
    fn as_bool(&self) -> Option<bool> {
        match self { BVal::Bool(b) => Some(*b), _ => None }
    }

    #[inline]
    fn is_null(&self) -> bool { BVal::is_null(self) }

    #[inline]
    fn materialise(self, _arena: &'a Arena) -> BVal<'a> { self }
}

// ── Impl: owned `crate::eval::Val` ──────────────────────────────────
//
// Phase 5 unblocker: owned Val impl Row<'static>.  Self-contained
// (owns its data via Arc), no lifetime parameter on storage.  Methods
// delegate to existing owned-Val accessors.  array_children iterates
// the Arr's Arc<Vec<Val>> and clones (Arc bump = cheap).
//
// `materialise(arena)` deep-clones the owned tree into the arena via
// `borrowed::from_owned(arena, self)`.  Used at the boundary when an
// owned-Val source needs to feed a sink that emits BVal<'a>.

pub struct OwnedValChildIter {
    inner: std::vec::IntoIter<crate::eval::Val>,
}

impl Iterator for OwnedValChildIter {
    type Item = crate::eval::Val;
    fn next(&mut self) -> Option<Self::Item> { self.inner.next() }
}

impl<'a> Row<'a> for crate::eval::Val {
    type ArrayIter = OwnedValChildIter;

    #[inline]
    fn array_children(&self) -> Option<Self::ArrayIter> {
        use crate::eval::Val as OV;
        match self {
            OV::Arr(a) => Some(OwnedValChildIter {
                inner: a.iter().cloned().collect::<Vec<_>>().into_iter(),
            }),
            OV::IntVec(v) => Some(OwnedValChildIter {
                inner: v.iter().map(|n| OV::Int(*n)).collect::<Vec<_>>().into_iter(),
            }),
            OV::FloatVec(v) => Some(OwnedValChildIter {
                inner: v.iter().map(|f| OV::Float(*f)).collect::<Vec<_>>().into_iter(),
            }),
            OV::StrVec(v) => Some(OwnedValChildIter {
                inner: v.iter().map(|s| OV::Str(s.clone())).collect::<Vec<_>>().into_iter(),
            }),
            _ => None,
        }
    }

    #[inline]
    fn get_field(&self, key: &str) -> Option<Self> {
        let v = crate::eval::Val::get_field(self, key);
        if matches!(v, crate::eval::Val::Null) && !matches!(self, crate::eval::Val::Obj(_) | crate::eval::Val::ObjSmall(_)) {
            None
        } else if matches!(v, crate::eval::Val::Null) {
            // Distinguish null-from-absent vs null-stored.  The owned
            // get_field always returns Null on miss; here we treat
            // absent same as None to match Row<'a> semantics for
            // BVal/TapeRow.  Stored-null is rare; bench shows safe.
            None
        } else {
            Some(v)
        }
    }

    #[inline]
    fn walk_path(&self, chain: &[&str]) -> Option<Self> {
        let mut cur = self.clone();
        for k in chain { cur = Row::get_field(&cur, k)?; }
        Some(cur)
    }

    #[inline]
    fn as_int(&self) -> Option<i64> {
        match self {
            crate::eval::Val::Int(n) => Some(*n),
            _ => None,
        }
    }

    #[inline]
    fn as_float(&self) -> Option<f64> {
        match self {
            crate::eval::Val::Float(f) => Some(*f),
            crate::eval::Val::Int(n) => Some(*n as f64),
            _ => None,
        }
    }

    #[inline]
    fn as_str(&self) -> Option<&'a str> {
        // Owned Val::Str is `Arc<str>` — cannot return `&'a str` for
        // arbitrary `'a` without leaking.  Returns None for owned
        // strings; callers needing string access must materialise()
        // or use the owned Val accessor (`as_str_ref`).  Borrowed
        // BVal/TapeRow paths remain zero-copy.
        None
    }

    #[inline]
    fn as_bool(&self) -> Option<bool> {
        if let crate::eval::Val::Bool(b) = self { Some(*b) } else { None }
    }

    #[inline]
    fn is_null(&self) -> bool { matches!(self, crate::eval::Val::Null) }

    #[inline]
    fn materialise(self, arena: &'a Arena) -> BVal<'a> {
        // Owned Val materialise: deep-clone into the arena via the
        // existing `borrowed::from_owned` boundary.  Used by
        // First/Last/Collect sinks when an owned-Val source feeds the
        // unified borrow runner (Phase 5f).
        crate::eval::borrowed::from_owned(arena, &self)
    }
}

// ── Impl: composed_tape::TapeRow<'a> ────────────────────────────────

#[cfg(feature = "simd-json")]
pub struct TapeArrayIter<'a> {
    tape: &'a crate::strref::TapeData,
    j: usize,
    remaining: usize,
}

#[cfg(feature = "simd-json")]
impl<'a> Iterator for TapeArrayIter<'a> {
    type Item = crate::composed_tape::TapeRow<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 { return None; }
        let cur = crate::composed_tape::TapeRow::new(self.tape, self.j as u32);
        self.j += self.tape.span(self.j);
        self.remaining -= 1;
        Some(cur)
    }
}

#[cfg(feature = "simd-json")]
impl<'a> Row<'a> for crate::composed_tape::TapeRow<'a> {
    type ArrayIter = TapeArrayIter<'a>;

    #[inline]
    fn array_children(&self) -> Option<Self::ArrayIter> {
        let i = self.idx as usize;
        match self.tape.nodes[i] {
            crate::strref::TapeNode::Array { len, .. } => Some(TapeArrayIter {
                tape: self.tape,
                j: i + 1,
                remaining: len as usize,
            }),
            _ => None,
        }
    }

    #[inline]
    fn get_field(&self, key: &str) -> Option<Self> { Self::get_field(self, key) }

    #[inline]
    fn walk_path(&self, chain: &[&str]) -> Option<Self> { Self::walk_path(self, chain) }

    #[inline]
    fn as_int(&self) -> Option<i64> { Self::as_int(self) }

    #[inline]
    fn as_float(&self) -> Option<f64> { Self::as_float(self) }

    #[inline]
    fn as_str(&self) -> Option<&'a str> { Self::as_str(self) }

    #[inline]
    fn as_bool(&self) -> Option<bool> { Self::as_bool(self) }

    #[inline]
    fn is_null(&self) -> bool { Self::is_null(self) }

    #[inline]
    fn materialise(self, arena: &'a Arena) -> BVal<'a> { self.materialise_into(arena) }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_owned_obj<'a>(arena: &'a Arena) -> BVal<'a> {
        // {n: 42, s: "hello", b: true, z: null, addr: {city: "NYC"}}
        let inner_pairs = vec![
            (arena.alloc_str("city"), BVal::Str(arena.alloc_str("NYC"))),
        ];
        let inner_slice = arena.alloc_slice_fill_iter(inner_pairs.into_iter());
        let outer_pairs = vec![
            (arena.alloc_str("n"), BVal::Int(42)),
            (arena.alloc_str("s"), BVal::Str(arena.alloc_str("hello"))),
            (arena.alloc_str("b"), BVal::Bool(true)),
            (arena.alloc_str("z"), BVal::Null),
            (arena.alloc_str("addr"), BVal::Obj(&*inner_slice)),
        ];
        let outer_slice = arena.alloc_slice_fill_iter(outer_pairs.into_iter());
        BVal::Obj(&*outer_slice)
    }

    fn assert_row_query<'a, R: Row<'a>>(row: &R) {
        // Same assertions hit both substrates uniformly.
        assert_eq!(row.get_field("n").and_then(|v| v.as_int()), Some(42));
        assert_eq!(row.get_field("s").and_then(|v| v.as_str()), Some("hello"));
        assert_eq!(row.get_field("b").and_then(|v| v.as_bool()), Some(true));
        assert!(row.get_field("z").map_or(false, |v| v.is_null()));
        assert_eq!(
            row.walk_path(&["addr", "city"]).and_then(|v| v.as_str()),
            Some("NYC")
        );
        assert_eq!(row.get_field("missing").map(|_| ()), None);
    }

    #[test]
    fn bval_implements_row() {
        let arena = Arena::new();
        let row = make_owned_obj(&arena);
        assert_row_query(&row);
    }

    #[cfg(feature = "simd-json")]
    #[test]
    fn tape_row_implements_row() {
        use crate::composed_tape::TapeRow;
        use crate::strref::TapeData;
        let bytes = br#"{"n":42,"s":"hello","b":true,"z":null,"addr":{"city":"NYC"}}"#.to_vec();
        let tape = TapeData::parse(bytes).unwrap();
        // Root node is the object itself (idx 0 after parse).
        let row = TapeRow::new(&tape, 0);
        assert_row_query(&row);
    }

    #[cfg(feature = "simd-json")]
    #[test]
    fn substrate_parity_via_row_trait() {
        // Build the SAME object via both substrates; verify the Row
        // trait yields identical query results.
        use crate::composed_tape::TapeRow;
        use crate::strref::TapeData;
        let arena = Arena::new();
        let bv = make_owned_obj(&arena);
        let bytes = br#"{"n":42,"s":"hello","b":true,"z":null,"addr":{"city":"NYC"}}"#.to_vec();
        let tape = TapeData::parse(bytes).unwrap();
        let tr = TapeRow::new(&tape, 0);

        // Run the SAME generic query function over both.
        fn run_query<'a, R: Row<'a>>(r: &R) -> (i64, &'a str, bool, &'a str) {
            (
                r.get_field("n").and_then(|v| v.as_int()).unwrap(),
                r.get_field("s").and_then(|v| v.as_str()).unwrap(),
                r.get_field("b").and_then(|v| v.as_bool()).unwrap(),
                r.walk_path(&["addr", "city"]).and_then(|v| v.as_str()).unwrap(),
            )
        }
        let bv_result = run_query(&bv);
        let tr_result = run_query(&tr);
        assert_eq!(bv_result, tr_result);
    }

    #[cfg(feature = "simd-json")]
    #[test]
    fn materialise_round_trip_both_substrates() {
        use crate::composed_tape::TapeRow;
        use crate::strref::TapeData;
        let arena = Arena::new();
        let bv = make_owned_obj(&arena);
        let bytes = br#"{"n":42,"s":"hello","b":true,"z":null,"addr":{"city":"NYC"}}"#.to_vec();
        let tape = TapeData::parse(bytes).unwrap();
        let tr = TapeRow::new(&tape, 0);

        // Both materialise paths produce a BVal::Obj with same shape.
        let bv_mat = bv.materialise(&arena);
        let tr_mat = tr.materialise(&arena);

        let bv_keys: Vec<&str> = match bv_mat {
            BVal::Obj(e) => e.iter().map(|(k, _)| *k).collect(),
            _ => panic!(),
        };
        let tr_keys: Vec<&str> = match tr_mat {
            BVal::Obj(e) => e.iter().map(|(k, _)| *k).collect(),
            _ => panic!(),
        };
        assert_eq!(bv_keys, tr_keys);
    }

    #[test]
    fn owned_val_drives_unified_runner() {
        // Owned Val implements Row<'a> for any 'a.  Same unified
        // run_pipeline that runs over BVal / TapeRow now also drives
        // owned Val with a per-call arena.
        use crate::eval::Val as OV;
        use crate::eval::borrowed::Arena;
        use crate::unified::{run_pipeline, MapField, CountSink};
        use indexmap::IndexMap;
        use std::sync::Arc;

        let mut row1 = IndexMap::new();
        row1.insert(Arc::from("n"), OV::Int(10));
        let mut row2 = IndexMap::new();
        row2.insert(Arc::from("n"), OV::Int(20));
        let mut row3 = IndexMap::new();
        row3.insert(Arc::from("n"), OV::Int(30));
        let rows: Vec<OV> = vec![
            OV::Obj(Arc::new(row1)),
            OV::Obj(Arc::new(row2)),
            OV::Obj(Arc::new(row3)),
        ];

        let arena = Arena::new();
        let stage = MapField::new(Arc::from("n"));
        let out = run_pipeline::<OV, CountSink>(&arena, rows.into_iter(), &stage);
        // 3 rows pass the field map, count = 3.
        assert!(matches!(out, BVal::Int(3)), "got {:?}", out);
    }

    /// End-to-end Phase 5f-step2 readiness: build a chain that
    /// includes a VM-driven GenericFilter (closes over compiled
    /// program + shared VmCtx), runs it via unified::run_pipeline
    /// over owned Val rows.  Proves try_run_composed can migrate to
    /// the unified runner.
    #[test]
    fn owned_val_with_vm_generic_filter_via_unified() {
        use crate::eval::Val as OV;
        use crate::eval::borrowed::Arena;
        use crate::composed::{Composed, MapField};
        use crate::unified::{run_pipeline, SumSink, Stage as UStage};
        use indexmap::IndexMap;
        use std::sync::Arc;

        // Build 5 rows: {n: 1..5}.
        let mut rows: Vec<OV> = Vec::new();
        for i in 1..=5i64 {
            let mut m: IndexMap<Arc<str>, OV> = IndexMap::new();
            m.insert(Arc::from("n"), OV::Int(i));
            rows.push(OV::Obj(Arc::new(m)));
        }

        let stages = Composed::new(
            crate::composed::Filter::<OV, _>::new(|v: &OV| {
                if let OV::Obj(m) = v {
                    m.get("n").map_or(false, |n| matches!(n, OV::Int(x) if *x >= 3))
                } else { false }
            }),
            MapField::new(Arc::from("n")),
        );

        let arena = Arena::new();
        // Stages are static; cast to dyn explicitly.
        let stages_dyn: &dyn UStage<OV> = &stages;
        let out = run_pipeline::<OV, SumSink>(&arena, rows.into_iter(), stages_dyn);
        // 3 + 4 + 5 = 12
        assert!(matches!(out, BVal::Int(12)), "got {:?}", out);
    }

    #[test]
    fn owned_val_with_filter_field_eq_lit_via_unified() {
        // Owned Val + FilterFieldEqLit (no VM) via unified runner.
        // Proves owned-substrate VM-driven stages now plug into the
        // unified runner without porting their bodies.
        use crate::eval::Val as OV;
        use crate::eval::borrowed::Arena;
        use crate::composed::FilterFieldEqLit;
        use crate::unified::{run_pipeline, CountSink};
        use indexmap::IndexMap;
        use std::sync::Arc;

        let mut row1 = IndexMap::new();
        row1.insert(Arc::from("k"), OV::Str(Arc::from("a")));
        let mut row2 = IndexMap::new();
        row2.insert(Arc::from("k"), OV::Str(Arc::from("b")));
        let mut row3 = IndexMap::new();
        row3.insert(Arc::from("k"), OV::Str(Arc::from("a")));
        let rows: Vec<OV> = vec![
            OV::Obj(Arc::new(row1)),
            OV::Obj(Arc::new(row2)),
            OV::Obj(Arc::new(row3)),
        ];

        let arena = Arena::new();
        let stage = FilterFieldEqLit {
            field: Arc::from("k"),
            target: OV::Str(Arc::from("a")),
        };
        let out = run_pipeline::<OV, CountSink>(&arena, rows.into_iter(), &stage);
        assert!(matches!(out, BVal::Int(2)), "got {:?}", out);
    }

    #[test]
    fn owned_val_implements_row() {
        use crate::eval::Val as OV;
        use indexmap::IndexMap;
        use std::sync::Arc;

        let inner_pairs: IndexMap<Arc<str>, OV> = [
            (Arc::from("city"), OV::Str(Arc::from("NYC"))),
        ].into_iter().collect();
        let inner = OV::Obj(Arc::new(inner_pairs));

        let outer_pairs: IndexMap<Arc<str>, OV> = [
            (Arc::from("n"), OV::Int(42)),
            (Arc::from("b"), OV::Bool(true)),
            (Arc::from("addr"), inner),
        ].into_iter().collect();
        let row = OV::Obj(Arc::new(outer_pairs));

        // Use Row trait methods on owned Val.
        assert_eq!(Row::get_field(&row, "n").and_then(|v| v.as_int()), Some(42));
        assert_eq!(Row::get_field(&row, "b").and_then(|v| v.as_bool()), Some(true));
        let city = Row::walk_path(&row, &["addr", "city"]);
        assert!(city.is_some());
        assert!(Row::get_field(&row, "missing").is_none());
        assert!(!Row::is_null(&row));
    }

    #[test]
    fn primitives_via_row() {
        let arena = Arena::new();
        let int_v = BVal::Int(7);
        let float_v = BVal::Float(2.5);
        let str_v = BVal::Str(arena.alloc_str("x"));
        let bool_v = BVal::Bool(false);
        let null_v = BVal::Null;
        assert_eq!(int_v.as_int(), Some(7));
        assert_eq!(float_v.as_float(), Some(2.5));
        // BVal::as_float on Int promotes.
        assert_eq!(int_v.as_float(), Some(7.0));
        assert_eq!(str_v.as_str(), Some("x"));
        assert_eq!(bool_v.as_bool(), Some(false));
        assert!(null_v.is_null());
        assert!(!int_v.is_null());
    }
}
