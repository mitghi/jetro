//! Columnar execution path for `ObjVec`-backed pipelines.
//!
//! When the source array has been promoted to `Val::ObjVec` (uniform-shape
//! array of objects stored as a struct-of-arrays), this module runs filter,
//! map, and aggregate stages directly over typed column slices — skipping per-
//! row `Val` enum dispatch. Falls through to the generic exec path otherwise.

use std::sync::Arc;

use crate::builtins::BuiltinColumnarStage;
use crate::{context::EvalError, value::Val};

use super::ReducerOp;
use super::{
    eval_cmp_op, num_finalise, num_fold, walk_field_chain, BodyKernel, NumOp, Pipeline,
    PipelineData, Sink, Source, Stage,
};

/// Entry point for the cache-assisted columnar path.
/// Delegates to `Pipeline::run_cached_columnar_impl`; returns `None` when the
/// pipeline shape does not qualify for columnar execution.
pub(super) fn run_cached(
    pipeline: &Pipeline,
    root: &Val,
    cache: Option<&dyn PipelineData>,
) -> Option<Result<Val, EvalError>> {
    pipeline.run_cached_columnar_impl(root, cache)
}

/// Entry point for the cache-free columnar path.
/// Delegates to `Pipeline::run_uncached_columnar_impl`; returns `None` when
/// the pipeline shape does not qualify for columnar execution.
pub(super) fn run_uncached(pipeline: &Pipeline, root: &Val) -> Option<Result<Val, EvalError>> {
    pipeline.run_uncached_columnar_impl(root)
}

fn stage_kind(stage: &Stage) -> Option<BuiltinColumnarStage> {
    stage.descriptor()?.columnar_stage()
}

fn stage_kernel<'a>(
    stages: &[Stage],
    kernels: &'a [BodyKernel],
    kind: BuiltinColumnarStage,
) -> Option<&'a BodyKernel> {
    let [stage] = stages else {
        return None;
    };
    let [kernel] = kernels else {
        return None;
    };
    (stage_kind(stage)? == kind).then_some(kernel)
}

fn stage_kernel_pair<'a>(
    stages: &[Stage],
    kernels: &'a [BodyKernel],
    first: BuiltinColumnarStage,
    second: BuiltinColumnarStage,
) -> Option<(&'a BodyKernel, &'a BodyKernel)> {
    let [first_stage, second_stage] = stages else {
        return None;
    };
    let [first_kernel, second_kernel] = kernels else {
        return None;
    };
    (stage_kind(first_stage)? == first && stage_kind(second_stage)? == second)
        .then_some((first_kernel, second_kernel))
}

fn stage_program<'a>(
    stage: &'a Stage,
    kind: BuiltinColumnarStage,
) -> Option<&'a crate::vm::Program> {
    if stage_kind(stage)? != kind {
        return None;
    }
    stage.body_program()
}

fn reducer_op(sink: &Sink) -> Option<ReducerOp> {
    match sink {
        Sink::Reducer(spec) if spec.predicate.is_none() => Some(spec.op),
        _ => None,
    }
}

fn is_count_sink(sink: &Sink) -> bool {
    matches!(reducer_op(sink), Some(ReducerOp::Count))
}

fn identity_numeric_sink(sink: &Sink) -> Option<NumOp> {
    match sink {
        Sink::Reducer(spec) if spec.predicate.is_none() && spec.projection.is_none() => {
            spec.numeric_op()
        }
        _ => None,
    }
}

fn projected_numeric_sink(sink: &Sink) -> Option<(&crate::vm::Program, NumOp)> {
    match sink {
        Sink::Reducer(spec) if spec.predicate.is_none() => {
            Some((spec.projection.as_ref()?.as_ref(), spec.numeric_op()?))
        }
        _ => None,
    }
}

fn single_stage_program<'a>(
    stages: &'a [Stage],
    kind: BuiltinColumnarStage,
) -> Option<&'a crate::vm::Program> {
    let [stage] = stages else {
        return None;
    };
    stage_program(stage, kind)
}

fn stage_program_pair<'a>(
    stages: &'a [Stage],
    first: BuiltinColumnarStage,
    second: BuiltinColumnarStage,
) -> Option<(&'a crate::vm::Program, &'a crate::vm::Program)> {
    let [first_stage, second_stage] = stages else {
        return None;
    };
    Some((
        stage_program(first_stage, first)?,
        stage_program(second_stage, second)?,
    ))
}

// Builds per-column typed vectors from a flat row-major cell buffer.
// Enables scalar loops in hot aggregation paths by avoiding per-element Val dispatch.
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
        let target_tag: u8 = match &cells[slot] {
            Val::Int(_) => 1,
            Val::Float(_) => 2,
            Val::Str(_) | Val::StrSlice(_) => 3,
            Val::Bool(_) => 4,
            _ => 0,
        };
        if target_tag == 0 {
            out.push(ObjVecCol::Mixed);
            continue;
        }
        // verify all rows share the same type tag before committing to a typed column
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
    fn try_columnar_stage_chain_with(
        &self,
        root: &Val,
        cache: Option<&dyn PipelineData>,
    ) -> Option<Result<Val, EvalError>> {
        let recv = match &self.source {
            Source::Receiver(v) => v.clone(),
            Source::FieldChain { keys } => walk_field_chain(root, keys),
        };

        // promote a plain Arr to ObjVec via the cache when available
        let recv = if let (Some(c), Val::Arr(a)) = (cache, &recv) {
            if let Some(d) = c.promote_objvec(a) {
                Val::ObjVec(d)
            } else {
                recv
            }
        } else {
            recv
        };

        if matches!(self.sink, Sink::Collect) {
            if let (Some(BodyKernel::FieldRead(key)), Val::ObjVec(d)) = (
                stage_kernel(
                    &self.stages,
                    &self.stage_kernels,
                    BuiltinColumnarStage::GroupBy,
                ),
                &recv,
            ) {
                if let Some(out) = objvec_typed_group_by(d, key) {
                    return Some(Ok(out));
                }
            }
        }

        if !matches!(self.sink, Sink::Collect) {
            return None;
        }
        if let Some((BodyKernel::FieldCmpLit(pk, pop, plit), BodyKernel::FieldRead(mk))) =
            stage_kernel_pair(
                &self.stages,
                &self.stage_kernels,
                BuiltinColumnarStage::Filter,
                BuiltinColumnarStage::Map,
            )
        {
            if let Val::ObjVec(d) = &recv {
                return objvec_typed_filter_map_collect(d, pk, *pop, plit, mk);
            }
        }
        None
    }

    fn try_columnar_stage_chain(&self, root: &Val) -> Option<Result<Val, EvalError>> {
        let recv = match &self.source {
            Source::Receiver(v) => v.clone(),
            Source::FieldChain { keys } => walk_field_chain(root, keys),
        };

        if let Some(BodyKernel::CurrentCmpLit(op, lit)) = stage_kernel(
            &self.stages,
            &self.stage_kernels,
            BuiltinColumnarStage::Filter,
        ) {
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
                (Val::IntVec(a), sink) if is_count_sink(sink) => {
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
                (Val::FloatVec(a), sink) if is_count_sink(sink) => {
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

        if !matches!(self.sink, Sink::Collect) {
            return None;
        }
        let arr = match &recv {
            Val::Arr(a) => Arc::clone(a),
            _ => return None,
        };

        if let Some(BodyKernel::FieldRead(k)) =
            stage_kernel(&self.stages, &self.stage_kernels, BuiltinColumnarStage::Map)
        {
            let mut out = Vec::with_capacity(arr.len());
            for v in arr.iter() {
                out.push(v.get_field(k.as_ref()));
            }
            return Some(Ok(Val::arr(out)));
        }

        if let Some(BodyKernel::FieldCmpLit(k, op, lit)) = stage_kernel(
            &self.stages,
            &self.stage_kernels,
            BuiltinColumnarStage::Filter,
        ) {
            return match Arc::try_unwrap(arr) {
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
            };
        }

        let filter_map = stage_kernel_pair(
            &self.stages,
            &self.stage_kernels,
            BuiltinColumnarStage::Filter,
            BuiltinColumnarStage::Map,
        );

        if let Some((BodyKernel::FieldCmpLit(pk, pop, plit), BodyKernel::FieldRead(mk))) =
            filter_map
        {
            let mut out = Vec::with_capacity(arr.len());
            for v in arr.iter() {
                let lhs = v.get_field(pk.as_ref());
                if eval_cmp_op(&lhs, *pop, plit) {
                    out.push(v.get_field(mk.as_ref()));
                }
            }
            return Some(Ok(Val::arr(out)));
        }

        if let Some(BodyKernel::FieldChain(ks)) =
            stage_kernel(&self.stages, &self.stage_kernels, BuiltinColumnarStage::Map)
        {
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
            return Some(Ok(Val::arr(out)));
        }

        if let Some((BodyKernel::FieldCmpLit(pk, pop, plit), BodyKernel::FieldChain(mks))) =
            filter_map
        {
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
            return Some(Ok(Val::arr(out)));
        }

        if let Some((BodyKernel::FieldChainCmpLit(pks, pop, plit), BodyKernel::FieldRead(mk))) =
            filter_map
        {
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
            return Some(Ok(Val::arr(out)));
        }

        None
    }

    /// Promotes a `Val::Arr` of uniform-shape objects to `ObjVecData`, returning
    /// the inner `Arc` directly. Returns `None` if promotion fails (e.g. mixed shapes).
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
                // fast path: try the same slot index before falling back to a hash lookup
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
        let stride = keys.len();
        let nrows = if stride == 0 { 0 } else { cells.len() / stride };
        let typed_cols = build_typed_cols(&cells, stride, nrows);

        Some(Val::ObjVec(Arc::new(crate::value::ObjVecData {
            keys: keys.into(),
            cells,
            typed_cols: Some(Arc::new(typed_cols)),
        })))
    }

    fn run_cached_columnar_impl(
        &self,
        root: &Val,
        cache: Option<&dyn PipelineData>,
    ) -> Option<Result<Val, EvalError>> {
        // try kernel-chain fast paths before the slot-level numeric/count paths
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

        // promote plain Arr to ObjVec via cache before entering slot-level paths
        let recv = if let (Some(c), Val::Arr(a)) = (cache, &recv) {
            if let Some(d) = c.promote_objvec(a) {
                Val::ObjVec(d)
            } else {
                recv
            }
        } else {
            recv
        };

        if self.stages.is_empty() {
            match (&recv, &self.sink) {
                (Val::IntVec(a), sink) if identity_numeric_sink(sink) == Some(NumOp::Sum) => {
                    return Some(Ok(Val::Int(a.iter().sum())))
                }
                (Val::IntVec(a), sink) if identity_numeric_sink(sink) == Some(NumOp::Min) => {
                    return Some(Ok(a
                        .iter()
                        .copied()
                        .min()
                        .map(Val::Int)
                        .unwrap_or(Val::Null)))
                }
                (Val::IntVec(a), sink) if identity_numeric_sink(sink) == Some(NumOp::Max) => {
                    return Some(Ok(a
                        .iter()
                        .copied()
                        .max()
                        .map(Val::Int)
                        .unwrap_or(Val::Null)))
                }
                (Val::IntVec(a), sink) if is_count_sink(sink) => {
                    return Some(Ok(Val::Int(a.len() as i64)))
                }
                (Val::FloatVec(a), sink) if identity_numeric_sink(sink) == Some(NumOp::Sum) => {
                    return Some(Ok(Val::Float(a.iter().sum())))
                }
                (Val::FloatVec(a), sink) if is_count_sink(sink) => {
                    return Some(Ok(Val::Int(a.len() as i64)))
                }
                (Val::StrVec(a), sink) if is_count_sink(sink) => {
                    return Some(Ok(Val::Int(a.len() as i64)))
                }
                (Val::StrSliceVec(a), sink) if is_count_sink(sink) => {
                    return Some(Ok(Val::Int(a.len() as i64)))
                }
                _ => {}
            }
        }

        if let Val::ObjVec(d) = &recv {
            let (cs, _ck, csink) = self.canonical();

            if is_count_sink(&csink) {
                if let Some(prog) = single_stage_program(&cs, BuiltinColumnarStage::FlatMap) {
                    if let Some(field) = single_field_prog(prog) {
                        if let Some(slot) = d.slot_of(field) {
                            return Some(Ok(objvec_flatmap_count_slot(d, slot)));
                        }
                    }
                }
            }

            if let Some((project, op)) = projected_numeric_sink(&csink) {
                let mf = single_field_prog(project)?;
                let sm = d.slot_of(mf)?;
                if cs.is_empty() {
                    return Some(Ok(objvec_num_slot(d, sm, op)));
                }
                if let Some(pred) = single_stage_program(&cs, BuiltinColumnarStage::Filter) {
                    let (pf, cop, lit) = single_cmp_prog(pred)?;
                    let sp = d.slot_of(pf)?;
                    return Some(Ok(objvec_filter_num_slots(d, sp, cop, &lit, sm, op)));
                }
            }
            if let Some(op) = identity_numeric_sink(&csink) {
                if let Some(prog) = single_stage_program(&cs, BuiltinColumnarStage::Map) {
                    let field = single_field_prog(prog)?;
                    let slot = d.slot_of(field)?;
                    return Some(Ok(objvec_num_slot(d, slot, op)));
                }
                if let Some((pred, map)) =
                    stage_program_pair(&cs, BuiltinColumnarStage::Filter, BuiltinColumnarStage::Map)
                {
                    let (pf, cop, lit) = single_cmp_prog(pred)?;
                    let mf = single_field_prog(map)?;
                    let sp = d.slot_of(pf)?;
                    let sm = d.slot_of(mf)?;
                    return Some(Ok(objvec_filter_num_slots(d, sp, cop, &lit, sm, op)));
                }
            }

            if is_count_sink(&csink) {
                if let Some(pred) = single_stage_program(&cs, BuiltinColumnarStage::Filter) {
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

    fn run_uncached_columnar_impl(&self, root: &Val) -> Option<Result<Val, EvalError>> {
        // kernel-chain fast paths handle typed-vec filter/map without ObjVec promotion
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

        match (&recv, &self.sink) {
            (Val::IntVec(a), sink) if identity_numeric_sink(sink) == Some(NumOp::Sum) => {
                return Some(Ok(Val::Int(a.iter().sum())))
            }
            (Val::IntVec(a), sink) if identity_numeric_sink(sink) == Some(NumOp::Min) => {
                return Some(Ok(a
                    .iter()
                    .copied()
                    .min()
                    .map(Val::Int)
                    .unwrap_or(Val::Null)))
            }
            (Val::IntVec(a), sink) if identity_numeric_sink(sink) == Some(NumOp::Max) => {
                return Some(Ok(a
                    .iter()
                    .copied()
                    .max()
                    .map(Val::Int)
                    .unwrap_or(Val::Null)))
            }
            (Val::IntVec(a), sink) if identity_numeric_sink(sink) == Some(NumOp::Avg) => {
                if a.is_empty() {
                    return Some(Ok(Val::Null));
                }
                let s: i64 = a.iter().sum();
                return Some(Ok(Val::Float(s as f64 / a.len() as f64)));
            }
            (Val::IntVec(a), sink) if is_count_sink(sink) => {
                return Some(Ok(Val::Int(a.len() as i64)))
            }
            (Val::FloatVec(a), sink) if identity_numeric_sink(sink) == Some(NumOp::Sum) => {
                return Some(Ok(Val::Float(a.iter().sum())))
            }
            (Val::FloatVec(a), sink) if identity_numeric_sink(sink) == Some(NumOp::Min) => {
                if a.is_empty() {
                    return Some(Ok(Val::Null));
                }
                let m = a.iter().copied().fold(f64::INFINITY, f64::min);
                return Some(Ok(Val::Float(m)));
            }
            (Val::FloatVec(a), sink) if identity_numeric_sink(sink) == Some(NumOp::Max) => {
                if a.is_empty() {
                    return Some(Ok(Val::Null));
                }
                let m = a.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                return Some(Ok(Val::Float(m)));
            }
            (Val::FloatVec(a), sink) if identity_numeric_sink(sink) == Some(NumOp::Avg) => {
                if a.is_empty() {
                    return Some(Ok(Val::Null));
                }
                let s: f64 = a.iter().sum();
                return Some(Ok(Val::Float(s / a.len() as f64)));
            }
            (Val::FloatVec(a), sink) if is_count_sink(sink) => {
                return Some(Ok(Val::Int(a.len() as i64)))
            }
            (Val::StrVec(a), sink) if is_count_sink(sink) => {
                return Some(Ok(Val::Int(a.len() as i64)))
            }
            (Val::StrSliceVec(a), sink) if is_count_sink(sink) => {
                return Some(Ok(Val::Int(a.len() as i64)))
            }
            _ => {}
        }

        let _ = recv;
        None
    }
}

// Decodes `LoadIdent k` or `PushCurrent GetField k` into the field name.
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

// Decodes a left-associative `AndOp` chain into `(field, op, literal)` leaves
// for multi-predicate slot-based count short-circuits.
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

// Decodes a minimal `(field, literal, cmp)` opcode triple into a typed triple.
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

fn single_cmp_prog<'a>(prog: &'a crate::vm::Program) -> Option<(&'a str, crate::ast::BinOp, Val)> {
    decode_cmp_ops(prog.ops.as_ref())
}

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

// Prefers typed column slices over cell-level dispatch to avoid Val boxing per row.
fn objvec_num_slot(d: &Arc<crate::value::ObjVecData>, slot: usize, op: NumOp) -> Val {
    use crate::value::ObjVecCol;

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

// Prefers typed column slices for the predicate check to avoid Val boxing.
fn objvec_filter_count_slot(
    d: &Arc<crate::value::ObjVecData>,
    slot: usize,
    op: crate::ast::BinOp,
    lit: &Val,
) -> Val {
    use crate::ast::BinOp as B;
    use crate::value::ObjVecCol;

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

// Exploits typed column slices for both predicate and aggregation columns when homogeneous.
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

    if let Some(cols) = &d.typed_cols {
        // int-pred × int-map fast path avoids Val allocation entirely
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

// Returns a typed-vec result (`IntVec`, `FloatVec`, or `StrVec`) when both columns are
// homogeneous; returns `None` otherwise so the caller can fall back to generic execution.
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

    // float predicate: coerce Int rhs to f64 so mixed-type comparisons work
    {
        let pred_f64 = match (pred_col, plit) {
            (ObjVecCol::Floats(p), Val::Float(r)) => Some((p, *r)),
            (ObjVecCol::Floats(p), Val::Int(r)) => Some((p, *r as f64)),
            _ => None,
        };
        if let Some((p, r)) = pred_f64 {
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

// Returns `None` when the key column is not a homogeneous `Strs`, `Ints`, or `Bools` column.
fn objvec_typed_group_by(d: &Arc<crate::value::ObjVecData>, key_field: &str) -> Option<Val> {
    use crate::value::ObjVecCol;
    let cols = d.typed_cols.as_ref()?;
    let key_slot = d.slot_of(key_field)?;
    let key_col = cols.get(key_slot)?;
    let stride = d.stride();
    let nrows = d.nrows();

    let mut groups: indexmap::IndexMap<Arc<str>, Vec<usize>> = indexmap::IndexMap::new();
    match key_col {
        ObjVecCol::Strs(c) => {
            // Arc::clone avoids re-interning the key string on each row
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

// Builds inline `Checker` variants backed by typed column slices to avoid Val allocation per row.
fn objvec_filter_count_and_slots(
    d: &Arc<crate::value::ObjVecData>,
    leaves: &[(usize, crate::ast::BinOp, Val)],
) -> Val {
    use crate::ast::BinOp as B;
    use crate::value::ObjVecCol;

    if let Some(cols) = &d.typed_cols {
        // local Checker avoids heap allocation and dynamic dispatch per predicate
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

// Returns `false` for type combinations that do not define a total order.
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

// Returns `false` for non-comparison operators.
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

// Caches the IndexMap slot index across successive rows to avoid repeated hash lookups.
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

// Falls back to a full `get_full` search and updates the cache on success.
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
