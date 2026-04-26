//! Tape-source row cursor.
//!
//! `TapeRow<'a>` is a 16-byte `Copy` cursor pointing at a node in a
//! parsed `TapeData`, plus the `materialise_into` boundary that walks
//! a tape subtree into an arena-allocated `BVal<'a>`.
//!
//! Originally this module carried a parallel TapeStageT + TapeSinkT
//! + run_pipeline_t pipeline pattern.  Phase 3 of
//! pipeline_unification migrated all callers onto
//! `unified::Stage<R>` + `unified::Sink` over `R = TapeRow<'a>` via
//! the `row::Row<'a>` impl in `row.rs`.  This module now carries
//! only the cursor type that the unified runner consumes.
//!
//! Requires `simd-json` cargo feature — depends on `crate::strref::TapeData`.

#![cfg(feature = "simd-json")]

use crate::eval::borrowed::{Arena, Val as BVal};
use crate::strref::{TapeData, TapeNode};
use simd_json::StaticNode as SN;

// ── TapeRow cursor ──────────────────────────────────────────────────

/// Cursor pointing at a tape node.  Cheap to copy (16 bytes — &TapeData
/// + u32 idx).  Operations walk the tape on demand; primitives are
/// extracted without allocation, materialisation only on explicit
/// `materialise_into(arena)`.
#[derive(Copy, Clone)]
pub struct TapeRow<'a> {
    pub tape: &'a TapeData,
    pub idx:  u32,
}

impl<'a> TapeRow<'a> {
    #[inline]
    pub fn new(tape: &'a TapeData, idx: u32) -> Self {
        Self { tape, idx }
    }

    #[inline]
    fn node(&self) -> TapeNode { self.tape.nodes[self.idx as usize] }

    #[inline]
    pub fn is_object(&self) -> bool { matches!(self.node(), TapeNode::Object { .. }) }

    #[inline]
    pub fn is_array(&self) -> bool { matches!(self.node(), TapeNode::Array { .. }) }

    /// Resolve a field on an Object node.  Returns a sub-cursor at the
    /// value subtree, or None when the node isn't an Object or the
    /// key is absent.  Linear scan over object entries (matches owned
    /// `tape_object_field`); ~5-20 keys typical, no advantage to
    /// hash dispatch at this scale.
    #[inline]
    pub fn get_field(&self, key: &str) -> Option<TapeRow<'a>> {
        let i = self.idx as usize;
        if let TapeNode::Object { len, .. } = self.tape.nodes[i] {
            let mut j = i + 1;
            for _ in 0..len {
                let k = self.tape.str_at(j);
                j += 1;
                if k == key { return Some(TapeRow::new(self.tape, j as u32)); }
                j += self.tape.span(j);
            }
        }
        None
    }

    /// Walk a chain of object-field steps.
    pub fn walk_path(&self, chain: &[&str]) -> Option<TapeRow<'a>> {
        let mut cur = *self;
        for k in chain { cur = cur.get_field(k)?; }
        Some(cur)
    }

    /// Borrow the string contents at this node (zero-copy view into
    /// the tape's `bytes_buf`).  Panics if the node isn't a StringRef.
    #[inline]
    pub fn as_str_unchecked(&self) -> &'a str { self.tape.str_at(self.idx as usize) }

    #[inline]
    pub fn as_str(&self) -> Option<&'a str> {
        match self.node() {
            TapeNode::StringRef { .. } => Some(self.as_str_unchecked()),
            _ => None,
        }
    }

    #[inline]
    pub fn as_int(&self) -> Option<i64> {
        match self.node() {
            TapeNode::Static(SN::I64(n)) => Some(n),
            TapeNode::Static(SN::U64(n)) if n <= i64::MAX as u64 => Some(n as i64),
            _ => None,
        }
    }

    #[inline]
    pub fn as_float(&self) -> Option<f64> {
        match self.node() {
            TapeNode::Static(SN::F64(f)) => Some(f),
            TapeNode::Static(SN::I64(n)) => Some(n as f64),
            TapeNode::Static(SN::U64(n)) => Some(n as f64),
            _ => None,
        }
    }

    #[inline]
    pub fn as_bool(&self) -> Option<bool> {
        match self.node() {
            TapeNode::Static(SN::Bool(b)) => Some(b),
            _ => None,
        }
    }

    #[inline]
    pub fn is_null(&self) -> bool {
        matches!(self.node(), TapeNode::Static(SN::Null))
    }

    /// Materialise this subtree into a `BVal<'a>` allocated in `arena`.
    /// Used by sinks that emit row payloads (First/Last/Collect — Phase 2).
    /// Cost is O(subtree-node-count); for primitives it's a single
    /// match + arena bump (string).
    pub fn materialise_into(&self, arena: &'a Arena) -> BVal<'a> {
        match self.node() {
            TapeNode::Static(SN::Null)    => BVal::Null,
            TapeNode::Static(SN::Bool(b)) => BVal::Bool(b),
            TapeNode::Static(SN::I64(n))  => BVal::Int(n),
            TapeNode::Static(SN::U64(n))  => {
                if n <= i64::MAX as u64 { BVal::Int(n as i64) }
                else { BVal::Float(n as f64) }
            }
            TapeNode::Static(SN::F64(f))  => BVal::Float(f),
            TapeNode::StringRef { .. }    => BVal::Str(arena.alloc_str(self.as_str_unchecked())),
            TapeNode::Array { len, .. } => {
                let mut tmp: Vec<BVal<'a>> = Vec::with_capacity(len as usize);
                let mut j = self.idx as usize + 1;
                for _ in 0..len {
                    let cur = TapeRow::new(self.tape, j as u32);
                    tmp.push(cur.materialise_into(arena));
                    j += self.tape.span(j);
                }
                BVal::Arr(arena.alloc_slice_copy(&tmp))
            }
            TapeNode::Object { len, .. } => {
                let mut tmp: Vec<(&'a str, BVal<'a>)> = Vec::with_capacity(len as usize);
                let mut j = self.idx as usize + 1;
                for _ in 0..len {
                    let k = self.tape.str_at(j);
                    j += 1;
                    let cur = TapeRow::new(self.tape, j as u32);
                    tmp.push((arena.alloc_str(k), cur.materialise_into(arena)));
                    j += self.tape.span(j);
                }
                let slice = arena.alloc_slice_fill_iter(tmp.into_iter());
                BVal::Obj(&*slice)
            }
        }
    }
}


// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_doc(s: &[u8]) -> std::sync::Arc<TapeData> {
        TapeData::parse(s.to_vec()).unwrap()
    }

    #[test]
    fn materialise_into_obj_round_trip() {
        let doc = br#"{"x":{"k":"hello","n":42,"f":1.5,"b":true,"z":null}}"#;
        let tape = parse_doc(doc);
        let arena = Arena::new();
        let idx = crate::strref::tape_walk_field_chain(&tape, &["x"]).unwrap();
        let row = TapeRow::new(&tape, idx as u32);
        let bv = row.materialise_into(&arena);
        match bv {
            BVal::Obj(entries) => {
                assert_eq!(entries.len(), 5);
                let pairs: std::collections::HashMap<&str, _> =
                    entries.iter().map(|(k, v)| (*k, *v)).collect();
                assert!(matches!(pairs["k"], BVal::Str("hello")));
                assert!(matches!(pairs["n"], BVal::Int(42)));
                assert!(matches!(pairs["f"], BVal::Float(_)));
                assert!(matches!(pairs["b"], BVal::Bool(true)));
                assert!(matches!(pairs["z"], BVal::Null));
            }
            _ => panic!("expected Obj, got {:?}", bv),
        }
    }
}
