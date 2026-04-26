//! Lower a Pipeline IR to the borrowed `composed_tape` substrate.
//!
//! Phase 3 wire-in for tape-source borrowed pipeline.  Runs AFTER
//! `bytescan::try_run_borrow` declines and uses the same lazy_tape
//! that owned `collect_val` would consume — no re-parse, no Val tree
//! build.  Closes the regression gap that pipeline_borrow.rs had:
//! pipeline_borrow does a full from_json_simd_arena (~3.7 ms); this
//! module reuses the cached TapeData (~0 ms after first call).
//!
//! ## Coverage today (Phase 3 initial wire)
//!
//! Source:    Source::FieldChain
//! Stages:    Filter(FieldCmpLit/FieldChainCmpLit/CurrentCmpLit/ConstBool)
//!            + Map(FieldRead/FieldChain) + Take(n) + Skip(n)
//! Sinks:     Count / Numeric(Sum/Min/Max/Avg) / First / Last / Collect
//!
//! Out of scope (returns None — caller falls back to owned):
//!   FlatMap / Sort / UniqueBy / Reverse / GroupBy / Map(ObjProject) /
//!   Map(FString) / Map(Arith) / Map(Generic) / lambdas.

#![cfg(feature = "simd-json")]

use crate::ast::BinOp;
use crate::eval::Val as OwnedVal;
use crate::eval::borrowed::{Arena, Val as BVal};
use crate::pipeline::{
    Pipeline, Source, Sink, Stage, BodyKernel, NumOp,
};
use crate::composed_tape::{
    TapeStageT, IdentityT, ComposedT, FilterT, MapFieldT, MapFieldChainT,
    TakeT, SkipT,
    CountSinkT, SumSinkT, MinSinkT, MaxSinkT, AvgSinkT,
    FirstSinkT, LastSinkT, CollectSinkT,
    TapeRow, run_pipeline_t,
};
use crate::strref::TapeData;
use std::sync::Arc;

#[derive(Copy, Clone)]
pub enum TapeSinkKind {
    Count,
    Sum, Min, Max, Avg,
    First, Last,
    Collect,
}

impl TapeSinkKind {
    fn from_pipeline_sink(s: &Sink) -> Option<Self> {
        Some(match s {
            Sink::Count               => TapeSinkKind::Count,
            Sink::Numeric(NumOp::Sum) => TapeSinkKind::Sum,
            Sink::Numeric(NumOp::Min) => TapeSinkKind::Min,
            Sink::Numeric(NumOp::Max) => TapeSinkKind::Max,
            Sink::Numeric(NumOp::Avg) => TapeSinkKind::Avg,
            Sink::First               => TapeSinkKind::First,
            Sink::Last                => TapeSinkKind::Last,
            Sink::Collect             => TapeSinkKind::Collect,
        })
    }
}

fn lower_stages(stages: &[Stage], kernels: &[BodyKernel]) -> Option<Box<dyn TapeStageT>> {
    let mut chain: Box<dyn TapeStageT> = Box::new(IdentityT);
    for (st, k) in stages.iter().zip(kernels.iter()) {
        let next: Box<dyn TapeStageT> = lower_stage(st, k)?;
        chain = Box::new(ComposedT { a: chain, b: next });
    }
    Some(chain)
}

fn lower_stage(stage: &Stage, kernel: &BodyKernel) -> Option<Box<dyn TapeStageT>> {
    match (stage, kernel) {
        (Stage::Take(n), _) => Some(Box::new(TakeT::new(*n))),
        (Stage::Skip(n), _) => Some(Box::new(SkipT::new(*n))),

        (Stage::Map(_), BodyKernel::FieldRead(name)) =>
            Some(Box::new(MapFieldT { field: Arc::clone(name) })),
        (Stage::Map(_), BodyKernel::FieldChain(keys)) =>
            Some(Box::new(MapFieldChainT { chain: keys.iter().cloned().collect() })),

        (Stage::Filter(_), BodyKernel::FieldCmpLit(field, op, lit)) => {
            let f = Arc::clone(field);
            let owned_lit = lit.clone();
            let op = *op;
            let pred = move |v: &TapeRow<'_>| -> bool {
                let val = match v.get_field(&f) { Some(x) => x, None => return false };
                trow_cmp_owned(&val, &op, &owned_lit)
            };
            Some(Box::new(FilterT { pred }))
        }
        (Stage::Filter(_), BodyKernel::FieldChainCmpLit(keys, op, lit)) => {
            let chain: Vec<Arc<str>> = keys.iter().cloned().collect();
            let owned_lit = lit.clone();
            let op = *op;
            let pred = move |v: &TapeRow<'_>| -> bool {
                let chain_refs: Vec<&str> = chain.iter().map(|a| a.as_ref()).collect();
                let val = match v.walk_path(&chain_refs) { Some(x) => x, None => return false };
                trow_cmp_owned(&val, &op, &owned_lit)
            };
            Some(Box::new(FilterT { pred }))
        }
        (Stage::Filter(_), BodyKernel::CurrentCmpLit(op, lit)) => {
            let owned_lit = lit.clone();
            let op = *op;
            let pred = move |v: &TapeRow<'_>| -> bool {
                trow_cmp_owned(v, &op, &owned_lit)
            };
            Some(Box::new(FilterT { pred }))
        }
        (Stage::Filter(_), BodyKernel::ConstBool(b)) => {
            let b = *b;
            let pred = move |_: &TapeRow<'_>| -> bool { b };
            Some(Box::new(FilterT { pred }))
        }

        _ => None,
    }
}

fn trow_cmp_owned(b: &TapeRow<'_>, op: &BinOp, lit: &OwnedVal) -> bool {
    use BinOp::*;
    if b.is_null() {
        return matches!((op, lit), (Eq, OwnedVal::Null) | (Neq, _));
    }
    if let (Some(a), OwnedVal::Int(c)) = (b.as_int(), lit) {
        return apply_cmp_i(a, *c, op);
    }
    if let Some(a) = b.as_float() {
        match lit {
            OwnedVal::Int(c) => return apply_cmp_f(a, *c as f64, op),
            OwnedVal::Float(c) => return apply_cmp_f(a, *c, op),
            _ => {}
        }
    }
    if let (Some(a), OwnedVal::Bool(c)) = (b.as_bool(), lit) {
        return apply_cmp_b(a, *c, op);
    }
    if let (Some(a), OwnedVal::Str(c)) = (b.as_str(), lit) {
        return apply_cmp_s(a, c.as_ref(), op);
    }
    matches!(op, Neq) && !matches!(lit, OwnedVal::Null)
}

fn apply_cmp_i(a: i64, b: i64, op: &BinOp) -> bool {
    match op {
        BinOp::Eq => a == b, BinOp::Neq => a != b,
        BinOp::Lt => a < b,  BinOp::Lte => a <= b,
        BinOp::Gt => a > b,  BinOp::Gte => a >= b,
        _ => false,
    }
}
fn apply_cmp_f(a: f64, b: f64, op: &BinOp) -> bool {
    match op {
        BinOp::Eq => a == b, BinOp::Neq => a != b,
        BinOp::Lt => a < b,  BinOp::Lte => a <= b,
        BinOp::Gt => a > b,  BinOp::Gte => a >= b,
        _ => false,
    }
}
fn apply_cmp_b(a: bool, b: bool, op: &BinOp) -> bool {
    match op {
        BinOp::Eq => a == b, BinOp::Neq => a != b,
        _ => false,
    }
}
fn apply_cmp_s(a: &str, b: &str, op: &BinOp) -> bool {
    match op {
        BinOp::Eq => a == b, BinOp::Neq => a != b,
        BinOp::Lt => a < b,  BinOp::Lte => a <= b,
        BinOp::Gt => a > b,  BinOp::Gte => a >= b,
        _ => false,
    }
}

fn dispatch_sink<'a>(
    arena: &'a Arena,
    tape: &'a TapeData,
    arr_idx: u32,
    stages: &dyn TapeStageT,
    kind: TapeSinkKind,
) -> BVal<'a> {
    match kind {
        TapeSinkKind::Count   => run_pipeline_t::<CountSinkT>(arena, tape, arr_idx, stages),
        TapeSinkKind::Sum     => run_pipeline_t::<SumSinkT>(arena, tape, arr_idx, stages),
        TapeSinkKind::Min     => run_pipeline_t::<MinSinkT>(arena, tape, arr_idx, stages),
        TapeSinkKind::Max     => run_pipeline_t::<MaxSinkT>(arena, tape, arr_idx, stages),
        TapeSinkKind::Avg     => run_pipeline_t::<AvgSinkT>(arena, tape, arr_idx, stages),
        TapeSinkKind::First   => run_pipeline_t::<FirstSinkT>(arena, tape, arr_idx, stages),
        TapeSinkKind::Last    => run_pipeline_t::<LastSinkT>(arena, tape, arr_idx, stages),
        TapeSinkKind::Collect => run_pipeline_t::<CollectSinkT>(arena, tape, arr_idx, stages),
    }
}

/// Try to run a Pipeline through the borrowed-composed-tape substrate.
/// Caller passes the cached TapeData (already parsed, ~0 ms cost) and
/// arena.  Returns None when shape unsupported.
pub fn try_run_borrow_tape<'a>(
    p: &Pipeline,
    tape: &'a TapeData,
    arena: &'a Arena,
) -> Option<Result<BVal<'a>, crate::eval::EvalError>> {
    let chain: Vec<Arc<str>> = match &p.source {
        Source::FieldChain { keys } => keys.iter().cloned().collect(),
        _ => return None,
    };

    let sink_kind = TapeSinkKind::from_pipeline_sink(&p.sink)?;
    let stage_chain = lower_stages(&p.stages, &p.stage_kernels)?;

    let chain_refs: Vec<&str> = chain.iter().map(|a| a.as_ref()).collect();
    let arr_idx = crate::strref::tape_walk_field_chain(tape, &chain_refs)?;
    if !matches!(tape.nodes[arr_idx], crate::strref::TapeNode::Array { .. }) {
        return None;
    }

    let out = dispatch_sink(arena, tape, arr_idx as u32, &*stage_chain, sink_kind);
    Some(Ok(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser;

    fn doc() -> Vec<u8> {
        let v = serde_json::json!({
            "orders": [
                {"id": 1, "status": "shipped",  "total": 10},
                {"id": 2, "status": "pending",  "total": 20},
                {"id": 3, "status": "shipped",  "total": 30},
                {"id": 4, "status": "shipped",  "total": 40},
            ]
        });
        serde_json::to_vec(&v).unwrap()
    }

    fn run(expr: &str) -> Option<crate::eval::Val> {
        let arena = Arena::new();
        let parsed = parser::parse(expr).ok()?;
        let p: Pipeline = Pipeline::lower(&parsed)?;
        let bytes = doc();
        let tape = TapeData::parse(bytes).ok()?;
        let r = try_run_borrow_tape(&p, &tape, &arena)?;
        Some(r.ok()?.to_owned_val())
    }

    #[test]
    fn count_via_tape_borrow() {
        assert!(matches!(run("$.orders.count()"), Some(crate::eval::Val::Int(4))));
    }

    #[test]
    fn filter_string_lit_count() {
        // The exact case where bytescan declines (string-lit filter)
        // and pipeline_borrow regressed.  composed_tape handles it.
        assert!(matches!(
            run("$.orders.filter(status == 'shipped').count()"),
            Some(crate::eval::Val::Int(3))
        ));
    }

    #[test]
    fn filter_string_lit_sum() {
        let v = run("$.orders.filter(status == 'shipped').map(total).sum()").unwrap();
        match v {
            crate::eval::Val::Int(80) => {}
            crate::eval::Val::Float(f) if (f - 80.0).abs() < 1e-9 => {}
            other => panic!("got {:?}", other),
        }
    }

    #[test]
    fn filter_first_via_tape_borrow() {
        let v = run("$.orders.filter(status == 'shipped').first()").unwrap();
        match v {
            crate::eval::Val::Obj(_) | crate::eval::Val::ObjSmall(_) => {}
            other => panic!("got {:?}", other),
        }
    }
}
