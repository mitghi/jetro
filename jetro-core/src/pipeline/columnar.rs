use std::sync::Arc;

use crate::{context::EvalError, value::Val};

use super::{
    eval_cmp_op, num_finalise, num_fold, walk_field_chain, BodyKernel, NumOp, Pipeline,
    PipelineData, Sink, Source, Stage,
};

pub(super) fn run_cached(
    pipeline: &Pipeline,
    root: &Val,
    cache: Option<&dyn PipelineData>,
) -> Option<Result<Val, EvalError>> {
    pipeline.try_columnar_with(root, cache)
}

pub(super) fn run_uncached(pipeline: &Pipeline, root: &Val) -> Option<Result<Val, EvalError>> {
    pipeline.try_columnar(root)
}

/// Build per-slot typed columns from a row-major Val cells matrix.
/// First-row inspection picks the candidate type per slot; subsequent
/// rows must match or that slot falls back to `Mixed`.  Cost O(N×K)
/// — already paid by the cells walk in `try_promote_objvec`.
fn build_typed_cols(cells: &[Val], stride: usize, nrows: usize) -> Vec<crate::value::ObjVecCol> {
    use crate::value::ObjVecCol;
    let mut out: Vec<ObjVecCol> = Vec::with_capacity(stride);
    if stride == 0 || nrows == 0 {
        for _ in 0..stride {
            out.push(ObjVecCol::Mixed);
        }
        return out;
    }
    for slot in 0..stride {
        // Inspect first row's value at this slot to choose target.
        let target_tag: u8 = match &cells[slot] {
            Val::Int(_) => 1,
            Val::Float(_) => 2,
            Val::Str(_) | Val::StrSlice(_) => 3,
            Val::Bool(_) => 4,
            _ => 0, // mixed / unsupported
        };
        if target_tag == 0 {
            out.push(ObjVecCol::Mixed);
            continue;
        }
        // Verify all rows.
        let mut ok = true;
        for r in 0..nrows {
            let v = &cells[r * stride + slot];
            let same = match (target_tag, v) {
                (1, Val::Int(_)) => true,
                (2, Val::Float(_)) => true,
                (3, Val::Str(_) | Val::StrSlice(_)) => true,
                (4, Val::Bool(_)) => true,
                _ => false,
            };
            if !same {
                ok = false;
                break;
            }
        }
        if !ok {
            out.push(ObjVecCol::Mixed);
            continue;
        }
        // Allocate typed lane.
        match target_tag {
            1 => {
                let mut col: Vec<i64> = Vec::with_capacity(nrows);
                for r in 0..nrows {
                    if let Val::Int(n) = &cells[r * stride + slot] {
                        col.push(*n);
                    }
                }
                out.push(ObjVecCol::Ints(col));
            }
            2 => {
                let mut col: Vec<f64> = Vec::with_capacity(nrows);
                for r in 0..nrows {
                    if let Val::Float(f) = &cells[r * stride + slot] {
                        col.push(*f);
                    }
                }
                out.push(ObjVecCol::Floats(col));
            }
            3 => {
                let mut col: Vec<Arc<str>> = Vec::with_capacity(nrows);
                for r in 0..nrows {
                    match &cells[r * stride + slot] {
                        Val::Str(s) => col.push(Arc::clone(s)),
                        Val::StrSlice(s) => col.push(s.to_arc()),
                        _ => {}
                    }
                }
                out.push(ObjVecCol::Strs(col));
            }
            4 => {
                let mut col: Vec<bool> = Vec::with_capacity(nrows);
                for r in 0..nrows {
                    if let Val::Bool(b) = &cells[r * stride + slot] {
                        col.push(*b);
                    }
                }
                out.push(ObjVecCol::Bools(col));
            }
            _ => out.push(ObjVecCol::Mixed),
        }
    }
    out
}

impl Pipeline {
    /// Phase 3 columnar fast path.  Detects pipelines whose source is
    /// an array of objects + zero stages + a single-field `SumMap` /
    /// `CountIf` / `SumFilterMap` sink.  Extracts the projected column
    /// into a flat `Vec<i64>` / `Vec<f64>` once, then folds the whole
    /// slice via the autovec'd reductions in vm.rs.
    ///
    /// Returns `None` if the shape doesn't match the columnar fast
    /// path; caller falls back to the per-row pull loop.
    /// Phase A2 stage-chain columnar fast path:
    ///   `Stage::Filter(FieldCmpLit) ∘ Stage::Map(FieldRead) ∘ Sink::Collect`
    ///   `Stage::Map(FieldRead) ∘ Sink::Collect`
    ///   `Stage::Filter(FieldCmpLit) ∘ Sink::Count`  (already covered by CountIf rule)
    ///   `Stage::Filter(FieldCmpLit) ∘ Sink::Numeric(...)` (already covered)
    /// Walks the column without entering vm.exec per row.
    /// Same as [`try_columnar_stage_chain`] but consults the optional
    /// data context to upgrade Val::Arr → Val::ObjVec; lets the typed
    /// stage-chain path (filter+map on ObjVec typed columns) fire.
    fn try_columnar_stage_chain_with(
        &self,
        root: &Val,
        cache: Option<&dyn PipelineData>,
    ) -> Option<Result<Val, EvalError>> {
        let recv = match &self.source {
            Source::Receiver(v) => v.clone(),
            Source::FieldChain { keys } => walk_field_chain(root, keys),
        };
        // Promote to ObjVec via cache when available.  Unlocks the
        // typed-column stage-chain path below.
        let recv = if let (Some(c), Val::Arr(a)) = (cache, &recv) {
            if let Some(d) = c.promote_objvec(a) {
                Val::ObjVec(d)
            } else {
                recv
            }
        } else {
            recv
        };

        // Typed-column ObjVec group_by: Stage::GroupBy(FieldRead) ∘
        // Sink::Collect over Strs / Ints / Floats / Bools key column.
        // Walks the typed key column directly, partitions row indices
        // per key, materialises Val::Obj { key → Vec<row> }.
        if let (
            Some([Stage::GroupBy(_)]),
            Some([BodyKernel::FieldRead(key)]),
            Val::ObjVec(d),
            Sink::Collect,
        ) = (
            self.stages.get(..),
            self.stage_kernels.get(..),
            &recv,
            &self.sink,
        ) {
            if let Some(out) = objvec_typed_group_by(d, key) {
                return Some(Ok(out));
            }
        }

        // Typed-column ObjVec stage-chain: Filter(FieldCmpLit) ∘
        // Map(FieldRead) ∘ Collect → primitive mask + typed gather.
        if !matches!(self.sink, Sink::Collect) {
            return None;
        }
        if let (
            Some([Stage::Filter(_), Stage::Map(_)]),
            Some([BodyKernel::FieldCmpLit(pk, pop, plit), BodyKernel::FieldRead(mk)]),
            Val::ObjVec(d),
        ) = (self.stages.get(..), self.stage_kernels.get(..), &recv)
        {
            return objvec_typed_filter_map_collect(d, pk, *pop, plit, mk);
        }
        None
    }

    fn try_columnar_stage_chain(&self, root: &Val) -> Option<Result<Val, EvalError>> {
        // Resolve receiver.
        let recv = match &self.source {
            Source::Receiver(v) => v.clone(),
            Source::FieldChain { keys } => walk_field_chain(root, keys),
        };

        // Phase B1 — typed-lane filter chain on IntVec / FloatVec /
        // StrVec receivers.  Stage::Filter(CurrentCmpLit) over a
        // primitive vec → walk slice directly, build typed output.
        // Sinks: Collect / Count.
        if let [Stage::Filter(_)] = self.stages.as_slice() {
            if let [BodyKernel::CurrentCmpLit(op, lit)] = self.stage_kernels.as_slice() {
                match (&recv, &self.sink) {
                    (Val::IntVec(a), Sink::Collect) => {
                        let mut out: Vec<i64> = Vec::with_capacity(a.len());
                        for n in a.iter() {
                            let v = Val::Int(*n);
                            if eval_cmp_op(&v, *op, lit) {
                                out.push(*n);
                            }
                        }
                        return Some(Ok(Val::int_vec(out)));
                    }
                    (Val::IntVec(a), Sink::Count) => {
                        let mut c = 0i64;
                        for n in a.iter() {
                            let v = Val::Int(*n);
                            if eval_cmp_op(&v, *op, lit) {
                                c += 1;
                            }
                        }
                        return Some(Ok(Val::Int(c)));
                    }
                    (Val::FloatVec(a), Sink::Collect) => {
                        let mut out: Vec<f64> = Vec::with_capacity(a.len());
                        for f in a.iter() {
                            let v = Val::Float(*f);
                            if eval_cmp_op(&v, *op, lit) {
                                out.push(*f);
                            }
                        }
                        return Some(Ok(Val::float_vec(out)));
                    }
                    (Val::FloatVec(a), Sink::Count) => {
                        let mut c = 0i64;
                        for f in a.iter() {
                            let v = Val::Float(*f);
                            if eval_cmp_op(&v, *op, lit) {
                                c += 1;
                            }
                        }
                        return Some(Ok(Val::Int(c)));
                    }
                    _ => {}
                }
            }
        }

        if !matches!(self.sink, Sink::Collect) {
            return None;
        }
        let arr = match &recv {
            Val::Arr(a) => Arc::clone(a),
            _ => return None,
        };

        // Build a (kernel, prog) view of the stages.
        let stages = &self.stages;
        let kernels = &self.stage_kernels;
        if stages.len() != kernels.len() {
            return None;
        }

        match (stages.as_slice(), kernels.as_slice()) {
            // Single Map(FieldRead) → Collect: direct projection.
            ([Stage::Map(_)], [BodyKernel::FieldRead(k)]) => {
                let mut out = Vec::with_capacity(arr.len());
                for v in arr.iter() {
                    out.push(v.get_field(k.as_ref()));
                }
                Some(Ok(Val::arr(out)))
            }
            // Single Filter(FieldCmpLit) → Collect: predicate mask copy.
            // Phase C1 — when Arc is uniquely held (refcount 1), take
            // ownership and retain-in-place; saves N Val clones.
            ([Stage::Filter(_)], [BodyKernel::FieldCmpLit(k, op, lit)]) => {
                match Arc::try_unwrap(arr) {
                    Ok(mut owned) => {
                        owned.retain(|v| {
                            let lhs = v.get_field(k.as_ref());
                            eval_cmp_op(&lhs, *op, lit)
                        });
                        Some(Ok(Val::arr(owned)))
                    }
                    Err(arr) => {
                        let mut out = Vec::with_capacity(arr.len());
                        for v in arr.iter() {
                            let lhs = v.get_field(k.as_ref());
                            if eval_cmp_op(&lhs, *op, lit) {
                                out.push(v.clone());
                            }
                        }
                        Some(Ok(Val::arr(out)))
                    }
                }
            }
            // Filter(FieldCmpLit) ∘ Map(FieldRead) → Collect: project filtered column.
            (
                [Stage::Filter(_), Stage::Map(_)],
                [BodyKernel::FieldCmpLit(pk, pop, plit), BodyKernel::FieldRead(mk)],
            ) => {
                let mut out = Vec::with_capacity(arr.len());
                for v in arr.iter() {
                    let lhs = v.get_field(pk.as_ref());
                    if eval_cmp_op(&lhs, *pop, plit) {
                        out.push(v.get_field(mk.as_ref()));
                    }
                }
                Some(Ok(Val::arr(out)))
            }
            // Single Map(FieldChain) → Collect: walk chain per item.
            // IC-cached probe: first item resolves each chain step via
            // IndexMap.get_full (returns slot index); subsequent items
            // try the cached slot first, fall back to hash on miss.
            // Saves ~half the probe cost on uniform-shape arrays.
            ([Stage::Map(_)], [BodyKernel::FieldChain(ks)]) => {
                let mut out = Vec::with_capacity(arr.len());
                let mut slots: Vec<Option<usize>> = vec![None; ks.len()];
                for v in arr.iter() {
                    let mut cur = v.clone();
                    for (i, k) in ks.iter().enumerate() {
                        cur = chain_step_ic(&cur, k.as_ref(), &mut slots[i]);
                        if matches!(cur, Val::Null) {
                            break;
                        }
                    }
                    out.push(cur);
                }
                Some(Ok(Val::arr(out)))
            }
            // Filter(FieldCmpLit) ∘ Map(FieldChain) → Collect.
            (
                [Stage::Filter(_), Stage::Map(_)],
                [BodyKernel::FieldCmpLit(pk, pop, plit), BodyKernel::FieldChain(mks)],
            ) => {
                let mut out = Vec::with_capacity(arr.len());
                for v in arr.iter() {
                    let lhs = v.get_field(pk.as_ref());
                    if eval_cmp_op(&lhs, *pop, plit) {
                        let mut cur = v.clone();
                        for k in mks.iter() {
                            cur = cur.get_field(k.as_ref());
                            if matches!(cur, Val::Null) {
                                break;
                            }
                        }
                        out.push(cur);
                    }
                }
                Some(Ok(Val::arr(out)))
            }
            // Filter(FieldChainCmpLit) ∘ Map(FieldRead) → Collect.
            (
                [Stage::Filter(_), Stage::Map(_)],
                [BodyKernel::FieldChainCmpLit(pks, pop, plit), BodyKernel::FieldRead(mk)],
            ) => {
                let mut out = Vec::with_capacity(arr.len());
                for v in arr.iter() {
                    let mut lhs = v.clone();
                    for k in pks.iter() {
                        lhs = lhs.get_field(k.as_ref());
                        if matches!(lhs, Val::Null) {
                            break;
                        }
                    }
                    if eval_cmp_op(&lhs, *pop, plit) {
                        out.push(v.get_field(mk.as_ref()));
                    }
                }
                Some(Ok(Val::arr(out)))
            }
            _ => None,
        }
    }

    /// Phase 7-lite — speculative ObjVec promotion at pipeline source.
    /// When the resolved receiver is `Val::Arr<Val::Obj>` with uniform
    /// shape (all rows have the same key set, in the same order), build
    /// a columnar `ObjVecData` on the fly so downstream slot-indexed
    /// kernels (objvec_*_slot) can fire.  No schema needed; just probe
    /// the first row + verify subsequent rows match.  Bails on shape
    /// mismatch, returns the original Arr unchanged.
    ///
    /// Cost: O(N × K) where N=row count, K=key count, dominated by
    /// pointer-equality on Arc<str> keys.  Win: subsequent operations
    /// run as O(N) slice walks instead of O(N) IndexMap probes.
    /// Wrapper that returns the promoted `ObjVecData` directly (for the
    /// memoised cache in `Jetro::get_or_promote_objvec`).
    pub fn try_promote_objvec_arr(arr: &Arc<Vec<Val>>) -> Option<Arc<crate::value::ObjVecData>> {
        if let Some(Val::ObjVec(d)) = Self::try_promote_objvec(arr) {
            Some(d)
        } else {
            None
        }
    }

    fn try_promote_objvec(arr: &Arc<Vec<Val>>) -> Option<Val> {
        if arr.is_empty() {
            return None;
        }
        let first = match &arr[0] {
            Val::Obj(m) => m,
            _ => return None,
        };
        let keys: Vec<Arc<str>> = first.keys().cloned().collect();
        if keys.is_empty() {
            return None;
        }
        let stride = keys.len();
        let mut cells: Vec<Val> = Vec::with_capacity(arr.len() * stride);
        for v in arr.iter() {
            let m = match v {
                Val::Obj(m) => m,
                _ => return None,
            };
            if m.len() != stride {
                return None;
            }
            for (i, k) in keys.iter().enumerate() {
                // Pointer-equal Arc check first (cheap path); fall back
                // to hash lookup if Arc identity differs across rows.
                let val = match m.get_index(i) {
                    Some((k2, v)) if Arc::ptr_eq(k2, k) => v.clone(),
                    _ => match m.get(k.as_ref()) {
                        Some(v) => v.clone(),
                        None => return None,
                    },
                };
                cells.push(val);
            }
        }
        // Build typed-column mirror at promotion time.  Costs O(N×K)
        // already spent walking cells; per-slot type lock determined
        // by inspecting the first row + verifying subsequent rows
        // match.  Uniform-type columns light up the typed-fast-path
        // in slot kernels (closes the boxed-Val tag-check tax).
        let stride = keys.len();
        let nrows = if stride == 0 { 0 } else { cells.len() / stride };
        let typed_cols = build_typed_cols(&cells, stride, nrows);

        Some(Val::ObjVec(Arc::new(crate::value::ObjVecData {
            keys: keys.into(),
            cells,
            typed_cols: Some(Arc::new(typed_cols)),
        })))
    }

    /// Cache-aware variant: when cache promotes the source array,
    /// recv is replaced with `Val::ObjVec` and the slot kernels fire.
    pub(super) fn try_columnar_with(
        &self,
        root: &Val,
        cache: Option<&dyn PipelineData>,
    ) -> Option<Result<Val, EvalError>> {
        // Typed ObjVec stage-chain path first. Cold SIMD inputs may also have
        // reached this runner through the tape-to-Val source bridge.
        if let Some(out) = self.try_columnar_stage_chain_with(root, cache) {
            return Some(out);
        }
        if let Some(out) = self.try_columnar_stage_chain(root) {
            return Some(out);
        }

        let recv = match &self.source {
            Source::Receiver(v) => v.clone(),
            Source::FieldChain { keys } => walk_field_chain(root, keys),
        };

        // Cache-driven ObjVec promotion: only if cache provided AND
        // recv is Val::Arr.  Promoted ObjVec memoised across calls.
        let recv = if let (Some(c), Val::Arr(a)) = (cache, &recv) {
            if let Some(d) = c.promote_objvec(a) {
                Val::ObjVec(d)
            } else {
                recv
            }
        } else {
            recv
        };

        // Typed primitive lane fast paths — only when the original
        // pipeline has zero stages (bare aggregate over a primitive
        // vec). Stage'd shapes go through the slot-kernel block below.
        if self.stages.is_empty() {
            match (&recv, &self.sink) {
                (Val::IntVec(a), Sink::Numeric(n)) if n.is_identity() && n.op == NumOp::Sum => {
                    return Some(Ok(Val::Int(a.iter().sum())))
                }
                (Val::IntVec(a), Sink::Numeric(n)) if n.is_identity() && n.op == NumOp::Min => {
                    return Some(Ok(a
                        .iter()
                        .copied()
                        .min()
                        .map(Val::Int)
                        .unwrap_or(Val::Null)))
                }
                (Val::IntVec(a), Sink::Numeric(n)) if n.is_identity() && n.op == NumOp::Max => {
                    return Some(Ok(a
                        .iter()
                        .copied()
                        .max()
                        .map(Val::Int)
                        .unwrap_or(Val::Null)))
                }
                (Val::IntVec(a), Sink::Count) => return Some(Ok(Val::Int(a.len() as i64))),
                (Val::FloatVec(a), Sink::Numeric(n)) if n.is_identity() && n.op == NumOp::Sum => {
                    return Some(Ok(Val::Float(a.iter().sum())))
                }
                (Val::FloatVec(a), Sink::Count) => return Some(Ok(Val::Int(a.len() as i64))),
                (Val::StrVec(a), Sink::Count) => return Some(Ok(Val::Int(a.len() as i64))),
                (Val::StrSliceVec(a), Sink::Count) => return Some(Ok(Val::Int(a.len() as i64))),
                _ => {}
            }
        }
        // ObjVec slot-kernel paths — operate on canonical view so they
        // fire whether the lowered pipeline kept fused Sinks or kept
        // base Sink + last Stage(...). One match arm per kernel; no
        // per-fused-variant duplication.
        if let Val::ObjVec(d) = &recv {
            let (cs, _ck, csink) = self.canonical();
            // FlatMap(FieldRead) → flatmap-count
            if matches!(csink, Sink::Count) && cs.len() == 1 {
                if let Stage::FlatMap(prog) = &cs[0] {
                    if let Some(field) = single_field_prog(prog) {
                        if let Some(slot) = d.slot_of(field) {
                            return Some(Ok(objvec_flatmap_count_slot(d, slot)));
                        }
                    }
                }
            }
            // Map(FieldRead) → numeric-on-slot
            if let Sink::Numeric(n) = &csink {
                if let Some(project) = &n.project {
                    let mf = single_field_prog(project)?;
                    let sm = d.slot_of(mf)?;
                    if cs.is_empty() {
                        return Some(Ok(objvec_num_slot(d, sm, n.op)));
                    }
                    if cs.len() == 1 {
                        if let Stage::Filter(pred) = &cs[0] {
                            let (pf, cop, lit) = single_cmp_prog(pred)?;
                            let sp = d.slot_of(pf)?;
                            return Some(Ok(objvec_filter_num_slots(d, sp, cop, &lit, sm, n.op)));
                        }
                    }
                }
                if cs.len() == 1 {
                    if let Stage::Map(prog) = &cs[0] {
                        let field = single_field_prog(prog)?;
                        let slot = d.slot_of(field)?;
                        return Some(Ok(objvec_num_slot(d, slot, n.op)));
                    }
                }
                if cs.len() == 2 {
                    if let (Stage::Filter(pred), Stage::Map(map)) = (&cs[0], &cs[1]) {
                        let (pf, cop, lit) = single_cmp_prog(pred)?;
                        let mf = single_field_prog(map)?;
                        let sp = d.slot_of(pf)?;
                        let sm = d.slot_of(mf)?;
                        return Some(Ok(objvec_filter_num_slots(d, sp, cop, &lit, sm, n.op)));
                    }
                }
            }
            // Filter(...) → count-if (single cmp or AND chain)
            if matches!(csink, Sink::Count) && cs.len() == 1 {
                if let Stage::Filter(pred) = &cs[0] {
                    if let Some((pf, op, lit)) = single_cmp_prog(pred) {
                        let sp = d.slot_of(pf)?;
                        return Some(Ok(objvec_filter_count_slot(d, sp, op, &lit)));
                    }
                    if let Some(leaves) = and_chain_prog(pred) {
                        let slots: Option<Vec<(usize, crate::ast::BinOp, Val)>> = leaves
                            .iter()
                            .map(|(f, op, lit)| d.slot_of(f).map(|s| (s, *op, lit.clone())))
                            .collect();
                        let slots = slots?;
                        return Some(Ok(objvec_filter_count_and_slots(d, &slots)));
                    }
                }
            }
        }
        None
    }

    pub(super) fn try_columnar(&self, root: &Val) -> Option<Result<Val, EvalError>> {
        // Phase A2 — Stage::Filter(FieldCmpLit) + Stage::Map(FieldRead) +
        // Sink::Collect over Val::Arr (object rows): walk the column
        // directly via slot-known IndexMap probes, build typed output
        // vec.  No per-row vm.exec.
        if let Some(out) = self.try_columnar_stage_chain(root) {
            return Some(out);
        }

        if !self.stages.is_empty() {
            return None;
        }

        let recv = match &self.source {
            Source::Receiver(v) => v.clone(),
            Source::FieldChain { keys } => walk_field_chain(root, keys),
        };
        // ObjVec promotion deferred to Phase 7 (build at parse time);
        // per-call promotion pays its O(N×K) cost on every collect()
        // and dominates short queries.  Keep the helper for a future
        // memoising path that caches the promoted shape on the source.

        // Phase A2 — typed-lane fast path.  When receiver is an
        // already-typed vector (IntVec / FloatVec / StrVec), the
        // sink can read the slice directly with no per-row Val tag
        // dispatch.  Each branch is mechanical: same fold, lane-typed.
        match (&recv, &self.sink) {
            (Val::IntVec(a), Sink::Numeric(n)) if n.is_identity() && n.op == NumOp::Sum => {
                return Some(Ok(Val::Int(a.iter().sum())))
            }
            (Val::IntVec(a), Sink::Numeric(n)) if n.is_identity() && n.op == NumOp::Min => {
                return Some(Ok(a
                    .iter()
                    .copied()
                    .min()
                    .map(Val::Int)
                    .unwrap_or(Val::Null)))
            }
            (Val::IntVec(a), Sink::Numeric(n)) if n.is_identity() && n.op == NumOp::Max => {
                return Some(Ok(a
                    .iter()
                    .copied()
                    .max()
                    .map(Val::Int)
                    .unwrap_or(Val::Null)))
            }
            (Val::IntVec(a), Sink::Numeric(n)) if n.is_identity() && n.op == NumOp::Avg => {
                if a.is_empty() {
                    return Some(Ok(Val::Null));
                }
                let s: i64 = a.iter().sum();
                return Some(Ok(Val::Float(s as f64 / a.len() as f64)));
            }
            (Val::IntVec(a), Sink::Count) => return Some(Ok(Val::Int(a.len() as i64))),
            (Val::FloatVec(a), Sink::Numeric(n)) if n.is_identity() && n.op == NumOp::Sum => {
                return Some(Ok(Val::Float(a.iter().sum())))
            }
            (Val::FloatVec(a), Sink::Numeric(n)) if n.is_identity() && n.op == NumOp::Min => {
                if a.is_empty() {
                    return Some(Ok(Val::Null));
                }
                let m = a.iter().copied().fold(f64::INFINITY, f64::min);
                return Some(Ok(Val::Float(m)));
            }
            (Val::FloatVec(a), Sink::Numeric(n)) if n.is_identity() && n.op == NumOp::Max => {
                if a.is_empty() {
                    return Some(Ok(Val::Null));
                }
                let m = a.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                return Some(Ok(Val::Float(m)));
            }
            (Val::FloatVec(a), Sink::Numeric(n)) if n.is_identity() && n.op == NumOp::Avg => {
                if a.is_empty() {
                    return Some(Ok(Val::Null));
                }
                let s: f64 = a.iter().sum();
                return Some(Ok(Val::Float(s / a.len() as f64)));
            }
            (Val::FloatVec(a), Sink::Count) => return Some(Ok(Val::Int(a.len() as i64))),
            (Val::StrVec(a), Sink::Count) => return Some(Ok(Val::Int(a.len() as i64))),
            (Val::StrSliceVec(a), Sink::Count) => return Some(Ok(Val::Int(a.len() as i64))),
            _ => {}
        }

        // ObjVec slot-kernel paths and Val::Arr columnar paths
        // previously dispatched on fused Sink variants. All gone post
        // fusion-off. Canonical-view consumer in `try_columnar_with`
        // covers the ObjVec shape; primitive-lane fast paths above
        // (IntVec / FloatVec / StrVec) cover Val::Arr-of-primitives.
        let _ = recv;
        None
    }
}

/// Decode a compiled sub-program that reads a single field from `@`
/// — either `[PushCurrent, GetField(k)]` (explicit `@.field`) or
/// `[LoadIdent(k)]` (bare-ident shorthand).  Returns the field name.
fn single_field_prog(prog: &crate::vm::Program) -> Option<&str> {
    use crate::vm::Opcode;
    let ops = prog.ops.as_ref();
    match ops.len() {
        1 => match &ops[0] {
            Opcode::LoadIdent(k) => Some(k.as_ref()),
            _ => None,
        },
        2 => match (&ops[0], &ops[1]) {
            (Opcode::PushCurrent, Opcode::GetField(k)) => Some(k.as_ref()),
            _ => None,
        },
        _ => None,
    }
}

/// Decode a compound AND predicate (a chain of single-cmp predicates
/// joined by `AndOp`) into a flat list of leaves.  Operates directly
/// on the `&[Opcode]` slice so the returned `&str` field references
/// borrow from the original program — no Arc allocation per leaf.
///
/// Accepts the shapes the compiler emits in practice:
///   2-way:  `[<cmp1>, AndOp(<cmp2>)]`
///   3-way:  `[<cmp1>, AndOp([<cmp2>, AndOp(<cmp3>)])]`
fn and_chain_prog<'a>(
    prog: &'a crate::vm::Program,
) -> Option<Vec<(&'a str, crate::ast::BinOp, Val)>> {
    use crate::vm::Opcode;
    let ops = prog.ops.as_ref();
    let (last, head) = ops.split_last()?;
    let rhs = match last {
        Opcode::AndOp(rhs) => rhs,
        _ => return None,
    };
    let head_leaf = decode_cmp_ops(head)?;
    let mut rest =
        and_chain_prog(rhs).or_else(|| decode_cmp_ops(rhs.ops.as_ref()).map(|x| vec![x]))?;
    let mut out = Vec::with_capacity(1 + rest.len());
    out.push(head_leaf);
    out.append(&mut rest);
    Some(out)
}

/// Match the single-cmp opcode prefix and return `(field, op, lit)`.
fn decode_cmp_ops<'a>(ops: &'a [crate::vm::Opcode]) -> Option<(&'a str, crate::ast::BinOp, Val)> {
    use crate::ast::BinOp;
    use crate::vm::Opcode;
    let (field, lit_idx, cmp_idx) = match ops.len() {
        3 => match &ops[0] {
            Opcode::LoadIdent(k) => (k.as_ref(), 1, 2),
            _ => return None,
        },
        4 => match (&ops[0], &ops[1]) {
            (Opcode::PushCurrent, Opcode::GetField(k)) => (k.as_ref(), 2, 3),
            _ => return None,
        },
        _ => return None,
    };
    let lit = match &ops[lit_idx] {
        Opcode::PushInt(n) => Val::Int(*n),
        Opcode::PushFloat(f) => Val::Float(*f),
        Opcode::PushStr(s) => Val::Str(Arc::clone(s)),
        Opcode::PushBool(b) => Val::Bool(*b),
        Opcode::PushNull => Val::Null,
        _ => return None,
    };
    let op = match &ops[cmp_idx] {
        Opcode::Eq => BinOp::Eq,
        Opcode::Neq => BinOp::Neq,
        Opcode::Lt => BinOp::Lt,
        Opcode::Lte => BinOp::Lte,
        Opcode::Gt => BinOp::Gt,
        Opcode::Gte => BinOp::Gte,
        _ => return None,
    };
    Some((field, op, lit))
}

/// Decode a compiled predicate of the shape `<load-field-k>;
/// <push-lit>; <cmp>` into `(field, op, lit)`.  Thin wrapper over
/// `decode_cmp_ops` for backward-compat with existing callers that
/// pass a `Program`.
fn single_cmp_prog<'a>(prog: &'a crate::vm::Program) -> Option<(&'a str, crate::ast::BinOp, Val)> {
    decode_cmp_ops(prog.ops.as_ref())
}
// ── ObjVec slot-indexed kernels (Phase 3.5) ──────────────────────────────────
//
// When the receiver is an ObjVec the row layout is flat
// `cells: Vec<Val>` with stride = keys.len(); a row's field at slot
// `s` lives at `cells[row * stride + s]`.  No IndexMap probe per
// row — direct array index.

/// Columnar `$.<arr>.flat_map(<field>).count()` — sums lengths of the
/// inner sequences without materialising the flattened result.
fn objvec_flatmap_count_slot(d: &Arc<crate::value::ObjVecData>, slot: usize) -> Val {
    let stride = d.stride();
    let nrows = d.nrows();
    let mut count: i64 = 0;
    for row in 0..nrows {
        let v = &d.cells[row * stride + slot];
        match v {
            Val::Arr(a) => count += a.len() as i64,
            Val::IntVec(a) => count += a.len() as i64,
            Val::FloatVec(a) => count += a.len() as i64,
            Val::StrVec(a) => count += a.len() as i64,
            Val::StrSliceVec(a) => count += a.len() as i64,
            Val::ObjVec(ad) => count += ad.nrows() as i64,
            _ => count += 1,
        }
    }
    Val::Int(count)
}

fn objvec_num_slot(d: &Arc<crate::value::ObjVecData>, slot: usize, op: NumOp) -> Val {
    use crate::value::ObjVecCol;
    // Phase 7-typed-columns fast path: typed lane → direct slice walk,
    // no per-row Val tag check.  Closes the boxed-Val tax on numeric
    // aggregates (~3-4× win measured on bench_complex Q12).
    if let Some(cols) = &d.typed_cols {
        match cols.get(slot) {
            Some(ObjVecCol::Ints(col)) => {
                if col.is_empty() {
                    return match op {
                        NumOp::Sum => Val::Int(0),
                        _ => Val::Null,
                    };
                }
                return match op {
                    NumOp::Sum => Val::Int(col.iter().sum()),
                    NumOp::Min => Val::Int(*col.iter().min().unwrap()),
                    NumOp::Max => Val::Int(*col.iter().max().unwrap()),
                    NumOp::Avg => {
                        let s: i64 = col.iter().sum();
                        Val::Float(s as f64 / col.len() as f64)
                    }
                };
            }
            Some(ObjVecCol::Floats(col)) => {
                if col.is_empty() {
                    return match op {
                        NumOp::Sum => Val::Float(0.0),
                        _ => Val::Null,
                    };
                }
                return match op {
                    NumOp::Sum => Val::Float(col.iter().sum()),
                    NumOp::Min => Val::Float(col.iter().copied().fold(f64::INFINITY, f64::min)),
                    NumOp::Max => Val::Float(col.iter().copied().fold(f64::NEG_INFINITY, f64::max)),
                    NumOp::Avg => {
                        let s: f64 = col.iter().sum();
                        Val::Float(s / col.len() as f64)
                    }
                };
            }
            _ => {}
        }
    }
    let stride = d.stride();
    let nrows = d.nrows();
    let mut acc_i: i64 = 0;
    let mut acc_f: f64 = 0.0;
    let mut floated = false;
    let mut min_f = f64::INFINITY;
    let mut max_f = f64::NEG_INFINITY;
    let mut n_obs: usize = 0;
    for row in 0..nrows {
        let v = &d.cells[row * stride + slot];
        num_fold(
            &mut acc_i,
            &mut acc_f,
            &mut floated,
            &mut min_f,
            &mut max_f,
            &mut n_obs,
            op,
            v,
        );
    }
    num_finalise(op, acc_i, acc_f, floated, min_f, max_f, n_obs)
}

fn objvec_filter_count_slot(
    d: &Arc<crate::value::ObjVecData>,
    slot: usize,
    op: crate::ast::BinOp,
    lit: &Val,
) -> Val {
    use crate::ast::BinOp as B;
    use crate::value::ObjVecCol;
    // Typed-column fast path.  Direct slice scan with primitive
    // comparison; no Val tag check, no boxed unbox.
    if let Some(cols) = &d.typed_cols {
        match (cols.get(slot), lit) {
            (Some(ObjVecCol::Ints(col)), Val::Int(rhs)) => {
                let r = *rhs;
                let mut c: i64 = 0;
                for &n in col.iter() {
                    let hit = match op {
                        B::Eq => n == r,
                        B::Neq => n != r,
                        B::Lt => n < r,
                        B::Lte => n <= r,
                        B::Gt => n > r,
                        B::Gte => n >= r,
                        _ => false,
                    };
                    if hit {
                        c += 1;
                    }
                }
                return Val::Int(c);
            }
            (Some(ObjVecCol::Floats(col)), Val::Float(rhs)) => {
                let r = *rhs;
                let mut c: i64 = 0;
                for &f in col.iter() {
                    let hit = match op {
                        B::Eq => f == r,
                        B::Neq => f != r,
                        B::Lt => f < r,
                        B::Lte => f <= r,
                        B::Gt => f > r,
                        B::Gte => f >= r,
                        _ => false,
                    };
                    if hit {
                        c += 1;
                    }
                }
                return Val::Int(c);
            }
            (Some(ObjVecCol::Strs(col)), Val::Str(rhs)) => {
                let r: &str = rhs.as_ref();
                let mut c: i64 = 0;
                for s in col.iter() {
                    let hit = match op {
                        B::Eq => s.as_ref() == r,
                        B::Neq => s.as_ref() != r,
                        _ => false,
                    };
                    if hit {
                        c += 1;
                    }
                }
                return Val::Int(c);
            }
            _ => {}
        }
    }
    let stride = d.stride();
    let nrows = d.nrows();
    let mut count: i64 = 0;
    for row in 0..nrows {
        let v = &d.cells[row * stride + slot];
        if cmp_val_binop_local(v, op, lit) {
            count += 1;
        }
    }
    Val::Int(count)
}

fn objvec_filter_num_slots(
    d: &Arc<crate::value::ObjVecData>,
    pred_slot: usize,
    cop: crate::ast::BinOp,
    lit: &Val,
    map_slot: usize,
    op: NumOp,
) -> Val {
    use crate::ast::BinOp as B;
    use crate::value::ObjVecCol;
    // Typed-column fast path for filter+map slot pair: walk both
    // columns as raw slices, primitive cmp + primitive fold.
    if let Some(cols) = &d.typed_cols {
        // Int pred + Int map (covers `total > 100` then `map(total)`).
        if let (Some(ObjVecCol::Ints(p)), Some(ObjVecCol::Ints(m)), Val::Int(rhs)) =
            (cols.get(pred_slot), cols.get(map_slot), lit)
        {
            let r = *rhs;
            let mut acc_i: i64 = 0;
            let mut acc_f: f64 = 0.0;
            let mut floated = false;
            let mut min_f = f64::INFINITY;
            let mut max_f = f64::NEG_INFINITY;
            let mut n_obs: usize = 0;
            for (i, &pv) in p.iter().enumerate() {
                let hit = match cop {
                    B::Eq => pv == r,
                    B::Neq => pv != r,
                    B::Lt => pv < r,
                    B::Lte => pv <= r,
                    B::Gt => pv > r,
                    B::Gte => pv >= r,
                    _ => false,
                };
                if hit {
                    let v = Val::Int(m[i]);
                    num_fold(
                        &mut acc_i,
                        &mut acc_f,
                        &mut floated,
                        &mut min_f,
                        &mut max_f,
                        &mut n_obs,
                        op,
                        &v,
                    );
                }
            }
            return num_finalise(op, acc_i, acc_f, floated, min_f, max_f, n_obs);
        }
    }
    let stride = d.stride();
    let nrows = d.nrows();
    let mut acc_i: i64 = 0;
    let mut acc_f: f64 = 0.0;
    let mut floated = false;
    let mut min_f = f64::INFINITY;
    let mut max_f = f64::NEG_INFINITY;
    let mut n_obs: usize = 0;
    for row in 0..nrows {
        let off = row * stride;
        if cmp_val_binop_local(&d.cells[off + pred_slot], cop, lit) {
            num_fold(
                &mut acc_i,
                &mut acc_f,
                &mut floated,
                &mut min_f,
                &mut max_f,
                &mut n_obs,
                op,
                &d.cells[off + map_slot],
            );
        }
    }
    num_finalise(op, acc_i, acc_f, floated, min_f, max_f, n_obs)
}

/// Phase 7-typed-columns Q3 path: ObjVec source + Filter(FieldCmpLit)
/// + Map(FieldRead) + Sink::Collect.  When both pred + map slots are
/// typed lanes, walk primitive columns directly: build typed output
/// vec sized by predicate hit count; no Val tag check, no IndexMap probe.
fn objvec_typed_filter_map_collect(
    d: &Arc<crate::value::ObjVecData>,
    pk: &str,
    pop: crate::ast::BinOp,
    plit: &Val,
    mk: &str,
) -> Option<Result<Val, EvalError>> {
    use crate::ast::BinOp as B;
    use crate::value::ObjVecCol;
    let cols = d.typed_cols.as_ref()?;
    let pred_slot = d.slot_of(pk)?;
    let map_slot = d.slot_of(mk)?;
    let pred_col = cols.get(pred_slot)?;
    let map_col = cols.get(map_slot)?;

    // Int pred + Int map (e.g. `filter(total > 500).map(id)`).
    if let (ObjVecCol::Ints(p), ObjVecCol::Ints(m), Val::Int(rhs)) = (pred_col, map_col, plit) {
        let r = *rhs;
        let mut out: Vec<i64> = Vec::with_capacity(p.len());
        for (i, &pv) in p.iter().enumerate() {
            let hit = match pop {
                B::Eq => pv == r,
                B::Neq => pv != r,
                B::Lt => pv < r,
                B::Lte => pv <= r,
                B::Gt => pv > r,
                B::Gte => pv >= r,
                _ => false,
            };
            if hit {
                out.push(m[i]);
            }
        }
        return Some(Ok(Val::int_vec(out)));
    }
    // Float pred + Int map: `filter(total > 500.0).map(id)` —
    // common bench shape (numeric thresholds with int IDs).
    // Promote int literal to f64 for cross-type compare.
    {
        let pred_f64 = match (pred_col, plit) {
            (ObjVecCol::Floats(p), Val::Float(r)) => Some((p, *r)),
            (ObjVecCol::Floats(p), Val::Int(r)) => Some((p, *r as f64)),
            _ => None,
        };
        if let Some((p, r)) = pred_f64 {
            // Map slot can be Int or Float; pick output.
            if let ObjVecCol::Ints(m) = map_col {
                let mut out: Vec<i64> = Vec::with_capacity(p.len());
                for (i, &pv) in p.iter().enumerate() {
                    let hit = match pop {
                        B::Eq => pv == r,
                        B::Neq => pv != r,
                        B::Lt => pv < r,
                        B::Lte => pv <= r,
                        B::Gt => pv > r,
                        B::Gte => pv >= r,
                        _ => false,
                    };
                    if hit {
                        out.push(m[i]);
                    }
                }
                return Some(Ok(Val::int_vec(out)));
            }
            if let ObjVecCol::Floats(m) = map_col {
                let mut out: Vec<f64> = Vec::with_capacity(p.len());
                for (i, &pv) in p.iter().enumerate() {
                    let hit = match pop {
                        B::Eq => pv == r,
                        B::Neq => pv != r,
                        B::Lt => pv < r,
                        B::Lte => pv <= r,
                        B::Gt => pv > r,
                        B::Gte => pv >= r,
                        _ => false,
                    };
                    if hit {
                        out.push(m[i]);
                    }
                }
                return Some(Ok(Val::float_vec(out)));
            }
            if let ObjVecCol::Strs(m) = map_col {
                let mut out: Vec<Arc<str>> = Vec::with_capacity(p.len());
                for (i, &pv) in p.iter().enumerate() {
                    let hit = match pop {
                        B::Eq => pv == r,
                        B::Neq => pv != r,
                        B::Lt => pv < r,
                        B::Lte => pv <= r,
                        B::Gt => pv > r,
                        B::Gte => pv >= r,
                        _ => false,
                    };
                    if hit {
                        out.push(Arc::clone(&m[i]));
                    }
                }
                return Some(Ok(Val::str_vec(out)));
            }
        }
    }
    // Int pred + Str map (e.g. `filter(total > 500).map(name)`).
    if let (ObjVecCol::Ints(p), ObjVecCol::Strs(m), Val::Int(rhs)) = (pred_col, map_col, plit) {
        let r = *rhs;
        let mut out: Vec<Arc<str>> = Vec::with_capacity(p.len());
        for (i, &pv) in p.iter().enumerate() {
            let hit = match pop {
                B::Eq => pv == r,
                B::Neq => pv != r,
                B::Lt => pv < r,
                B::Lte => pv <= r,
                B::Gt => pv > r,
                B::Gte => pv >= r,
                _ => false,
            };
            if hit {
                out.push(Arc::clone(&m[i]));
            }
        }
        return Some(Ok(Val::str_vec(out)));
    }
    // Str pred (== / !=) + any map: status-style filter.
    if let (ObjVecCol::Strs(p), Val::Str(rhs)) = (pred_col, plit) {
        let r: &str = rhs.as_ref();
        let mut hits: Vec<usize> = Vec::with_capacity(p.len());
        for (i, ps) in p.iter().enumerate() {
            let hit = match pop {
                B::Eq => ps.as_ref() == r,
                B::Neq => ps.as_ref() != r,
                _ => false,
            };
            if hit {
                hits.push(i);
            }
        }
        return Some(Ok(materialise_typed_indices(map_col, &hits)));
    }
    None
}

/// Typed columnar group_by: walk the key column directly, partition
/// row indices per distinct key, materialise per-group Val::Arr<Val::Obj>
/// (or a typed lane when all selected rows project the same shape).
/// Avoids per-row IndexMap probe + per-row key Arc clone of the walker
/// path.
fn objvec_typed_group_by(d: &Arc<crate::value::ObjVecData>, key_field: &str) -> Option<Val> {
    use crate::value::ObjVecCol;
    let cols = d.typed_cols.as_ref()?;
    let key_slot = d.slot_of(key_field)?;
    let key_col = cols.get(key_slot)?;
    let stride = d.stride();
    let nrows = d.nrows();

    // Partition row indices by key string.
    let mut groups: indexmap::IndexMap<Arc<str>, Vec<usize>> = indexmap::IndexMap::new();
    match key_col {
        ObjVecCol::Strs(c) => {
            // Use Arc::clone for repeated keys (Arc is interned-ish).
            for (i, s) in c.iter().enumerate() {
                groups.entry(Arc::clone(s)).or_default().push(i);
            }
        }
        ObjVecCol::Ints(c) => {
            for (i, n) in c.iter().enumerate() {
                let k: Arc<str> = Arc::from(n.to_string().as_str());
                groups.entry(k).or_default().push(i);
            }
        }
        ObjVecCol::Bools(c) => {
            for (i, b) in c.iter().enumerate() {
                let k: Arc<str> = if *b {
                    Arc::from("true")
                } else {
                    Arc::from("false")
                };
                groups.entry(k).or_default().push(i);
            }
        }
        _ => return None,
    };

    // Public group_by semantics expose each bucket as an array of
    // objects. Keep the fast key partitioning, but materialise bucket
    // rows to the same shape as the generic builtin path.
    let mut out: indexmap::IndexMap<Arc<str>, Val> =
        indexmap::IndexMap::with_capacity(groups.len());
    for (k, indices) in groups.into_iter() {
        let mut rows: Vec<Val> = Vec::with_capacity(indices.len());
        for r in indices {
            let off = r * stride;
            let mut row: indexmap::IndexMap<Arc<str>, Val> =
                indexmap::IndexMap::with_capacity(stride);
            for slot in 0..stride {
                row.insert(Arc::clone(&d.keys[slot]), d.cells[off + slot].clone());
            }
            rows.push(Val::Obj(Arc::new(row)));
        }
        out.insert(k, Val::arr(rows));
    }
    let _ = nrows;
    Some(Val::Obj(Arc::new(out)))
}

fn materialise_typed_indices(col: &crate::value::ObjVecCol, indices: &[usize]) -> Val {
    use crate::value::ObjVecCol;
    match col {
        ObjVecCol::Ints(c) => {
            let mut o: Vec<i64> = Vec::with_capacity(indices.len());
            for &i in indices {
                o.push(c[i]);
            }
            Val::int_vec(o)
        }
        ObjVecCol::Floats(c) => {
            let mut o: Vec<f64> = Vec::with_capacity(indices.len());
            for &i in indices {
                o.push(c[i]);
            }
            Val::float_vec(o)
        }
        ObjVecCol::Strs(c) => {
            let mut o: Vec<Arc<str>> = Vec::with_capacity(indices.len());
            for &i in indices {
                o.push(Arc::clone(&c[i]));
            }
            Val::str_vec(o)
        }
        ObjVecCol::Bools(c) => {
            let mut o: Vec<Val> = Vec::with_capacity(indices.len());
            for &i in indices {
                o.push(Val::Bool(c[i]));
            }
            Val::arr(o)
        }
        ObjVecCol::Mixed => Val::arr(Vec::new()),
    }
}

fn objvec_filter_count_and_slots(
    d: &Arc<crate::value::ObjVecData>,
    leaves: &[(usize, crate::ast::BinOp, Val)],
) -> Val {
    use crate::ast::BinOp as B;
    use crate::value::ObjVecCol;
    // Phase 7-typed-columns AND-chain path.  Pre-resolve each leaf to
    // a typed checker closure once; per row, run all checkers as
    // primitive comparisons over typed slices.  Skips per-leaf
    // cmp_val_binop_local + Val tag check on every row.
    if let Some(cols) = &d.typed_cols {
        // Snapshot each leaf as a "typed checker": closures over &[i64]
        // / &[f64] / &[Arc<str>] indexed by row.  Bail to scalar path
        // if any leaf can't be typed-resolved.
        enum Checker<'a> {
            IntsEq(&'a [i64], i64),
            IntsNeq(&'a [i64], i64),
            IntsLt(&'a [i64], i64),
            IntsLte(&'a [i64], i64),
            IntsGt(&'a [i64], i64),
            IntsGte(&'a [i64], i64),
            FloatsEq(&'a [f64], f64),
            FloatsNeq(&'a [f64], f64),
            FloatsLt(&'a [f64], f64),
            FloatsLte(&'a [f64], f64),
            FloatsGt(&'a [f64], f64),
            FloatsGte(&'a [f64], f64),
            StrsEq(&'a [Arc<str>], &'a str),
            StrsNeq(&'a [Arc<str>], &'a str),
            BoolsEq(&'a [bool], bool),
            BoolsNeq(&'a [bool], bool),
        }
        impl<'a> Checker<'a> {
            #[inline]
            fn at(&self, i: usize) -> bool {
                match *self {
                    Checker::IntsEq(c, r) => c[i] == r,
                    Checker::IntsNeq(c, r) => c[i] != r,
                    Checker::IntsLt(c, r) => c[i] < r,
                    Checker::IntsLte(c, r) => c[i] <= r,
                    Checker::IntsGt(c, r) => c[i] > r,
                    Checker::IntsGte(c, r) => c[i] >= r,
                    Checker::FloatsEq(c, r) => c[i] == r,
                    Checker::FloatsNeq(c, r) => c[i] != r,
                    Checker::FloatsLt(c, r) => c[i] < r,
                    Checker::FloatsLte(c, r) => c[i] <= r,
                    Checker::FloatsGt(c, r) => c[i] > r,
                    Checker::FloatsGte(c, r) => c[i] >= r,
                    Checker::StrsEq(c, r) => c[i].as_ref() == r,
                    Checker::StrsNeq(c, r) => c[i].as_ref() != r,
                    Checker::BoolsEq(c, r) => c[i] == r,
                    Checker::BoolsNeq(c, r) => c[i] != r,
                }
            }
        }
        let mut typed_checkers: Vec<Checker> = Vec::with_capacity(leaves.len());
        for (slot, op, lit) in leaves {
            let col = match cols.get(*slot) {
                Some(c) => c,
                None => break,
            };
            let chk: Option<Checker> = match (col, lit, *op) {
                (ObjVecCol::Ints(c), Val::Int(r), B::Eq) => Some(Checker::IntsEq(c.as_slice(), *r)),
                (ObjVecCol::Ints(c), Val::Int(r), B::Neq) => {
                    Some(Checker::IntsNeq(c.as_slice(), *r))
                }
                (ObjVecCol::Ints(c), Val::Int(r), B::Lt) => Some(Checker::IntsLt(c.as_slice(), *r)),
                (ObjVecCol::Ints(c), Val::Int(r), B::Lte) => {
                    Some(Checker::IntsLte(c.as_slice(), *r))
                }
                (ObjVecCol::Ints(c), Val::Int(r), B::Gt) => Some(Checker::IntsGt(c.as_slice(), *r)),
                (ObjVecCol::Ints(c), Val::Int(r), B::Gte) => {
                    Some(Checker::IntsGte(c.as_slice(), *r))
                }
                (ObjVecCol::Floats(c), Val::Float(r), B::Eq) => {
                    Some(Checker::FloatsEq(c.as_slice(), *r))
                }
                (ObjVecCol::Floats(c), Val::Float(r), B::Neq) => {
                    Some(Checker::FloatsNeq(c.as_slice(), *r))
                }
                (ObjVecCol::Floats(c), Val::Float(r), B::Lt) => {
                    Some(Checker::FloatsLt(c.as_slice(), *r))
                }
                (ObjVecCol::Floats(c), Val::Float(r), B::Lte) => {
                    Some(Checker::FloatsLte(c.as_slice(), *r))
                }
                (ObjVecCol::Floats(c), Val::Float(r), B::Gt) => {
                    Some(Checker::FloatsGt(c.as_slice(), *r))
                }
                (ObjVecCol::Floats(c), Val::Float(r), B::Gte) => {
                    Some(Checker::FloatsGte(c.as_slice(), *r))
                }
                (ObjVecCol::Strs(c), Val::Str(r), B::Eq) => {
                    Some(Checker::StrsEq(c.as_slice(), r.as_ref()))
                }
                (ObjVecCol::Strs(c), Val::Str(r), B::Neq) => {
                    Some(Checker::StrsNeq(c.as_slice(), r.as_ref()))
                }
                (ObjVecCol::Bools(c), Val::Bool(r), B::Eq) => {
                    Some(Checker::BoolsEq(c.as_slice(), *r))
                }
                (ObjVecCol::Bools(c), Val::Bool(r), B::Neq) => {
                    Some(Checker::BoolsNeq(c.as_slice(), *r))
                }
                _ => None,
            };
            match chk {
                Some(c) => typed_checkers.push(c),
                None => {
                    typed_checkers.clear();
                    break;
                }
            }
        }
        if typed_checkers.len() == leaves.len() {
            let nrows = d.nrows();
            let mut count: i64 = 0;
            'rows_typed: for row in 0..nrows {
                for c in &typed_checkers {
                    if !c.at(row) {
                        continue 'rows_typed;
                    }
                }
                count += 1;
            }
            return Val::Int(count);
        }
    }
    let stride = d.stride();
    let nrows = d.nrows();
    let mut count: i64 = 0;
    'rows: for row in 0..nrows {
        let off = row * stride;
        for (slot, op, lit) in leaves {
            if !cmp_val_binop_local(&d.cells[off + slot], *op, lit) {
                continue 'rows;
            }
        }
        count += 1;
    }
    Val::Int(count)
}
/// Inline numeric/string comparison for the columnar path.  Mirrors
/// the semantics of the VM's existing `cmp_val_binop` helper but
/// accessible from this module.
#[inline]
fn cmp_val_binop_local(a: &Val, op: crate::ast::BinOp, b: &Val) -> bool {
    use crate::ast::BinOp;
    match (a, b) {
        (Val::Int(x), Val::Int(y)) => match op {
            BinOp::Eq => x == y,
            BinOp::Neq => x != y,
            BinOp::Lt => x < y,
            BinOp::Lte => x <= y,
            BinOp::Gt => x > y,
            BinOp::Gte => x >= y,
            _ => false,
        },
        (Val::Int(x), Val::Float(y)) => num_f_cmp(*x as f64, *y, op),
        (Val::Float(x), Val::Int(y)) => num_f_cmp(*x, *y as f64, op),
        (Val::Float(x), Val::Float(y)) => num_f_cmp(*x, *y, op),
        (Val::Str(_) | Val::StrSlice(_), Val::Str(_) | Val::StrSlice(_)) => {
            let Some(x) = a.as_str_ref() else {
                return false;
            };
            let Some(y) = b.as_str_ref() else {
                return false;
            };
            match op {
                BinOp::Eq => x == y,
                BinOp::Neq => x != y,
                BinOp::Lt => x < y,
                BinOp::Lte => x <= y,
                BinOp::Gt => x > y,
                BinOp::Gte => x >= y,
                _ => false,
            }
        }
        (Val::Bool(x), Val::Bool(y)) => match op {
            BinOp::Eq => x == y,
            BinOp::Neq => x != y,
            _ => false,
        },
        _ => false,
    }
}

#[inline]
fn num_f_cmp(a: f64, b: f64, op: crate::ast::BinOp) -> bool {
    use crate::ast::BinOp;
    match op {
        BinOp::Eq => a == b,
        BinOp::Neq => a != b,
        BinOp::Lt => a < b,
        BinOp::Lte => a <= b,
        BinOp::Gt => a > b,
        BinOp::Gte => a >= b,
        _ => false,
    }
}

/// IC-cached chain step.  Cached `Option<usize>` slot survives across
/// rows; missing-key marks slot None.  Used by Map(FieldChain) columnar
/// path to amortise the IndexMap probe cost across the whole array.
#[inline]
fn chain_step_ic(v: &Val, k: &str, ic: &mut Option<usize>) -> Val {
    match v {
        Val::Obj(m) => match lookup_via_ic(m, k, ic) {
            Some(x) => x.clone(),
            None => Val::Null,
        },
        _ => v.get_field(k),
    }
}

fn lookup_via_ic<'a>(
    m: &'a indexmap::IndexMap<Arc<str>, Val>,
    k: &str,
    cached: &mut Option<usize>,
) -> Option<&'a Val> {
    if let Some(i) = *cached {
        if let Some((ki, vi)) = m.get_index(i) {
            if ki.as_ref() == k {
                return Some(vi);
            }
        }
    }
    match m.get_full(k) {
        Some((i, _, v)) => {
            *cached = Some(i);
            Some(v)
        }
        None => {
            *cached = None;
            None
        }
    }
}
