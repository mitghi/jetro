//! Lower a Pipeline IR to the UNIFIED borrowed substrate over
//! `composed_tape::TapeRow<'a>`.
//!
//! Phase 3 migration: this module previously consumed `composed_tape`'s
//! parallel TapeStageT + TapeSinkT.  It now consumes `unified::Stage<R>`
//! + `unified::Sink` directly, with R = `TapeRow<'a>`.  Same Stage code
//! is shared with the (future) BVal-substrate lowering; future
//! sessions delete pipeline_borrow.rs + composed_borrow.rs once the
//! BVal lowering is also migrated.
//!
//! Wired into `Jetro::collect_val_borrow` after bytescan_borrow declines.
//! Reuses `lazy_tape()` so no re-parse cost.
//!
//! ## Coverage today
//!
//! Source:    Source::FieldChain
//! Stages:    Filter(FieldCmpLit/FieldChainCmpLit/CurrentCmpLit/ConstBool)
//!            + Map(FieldRead/FieldChain) + Take(n) + Skip(n) +
//!            FlatMap(FieldRead) (anywhere in chain — unified Composed
//!            handles Many propagation; no special split required).
//! Sinks:     Count / Numeric(Sum/Min/Max/Avg) / First / Last / Collect

#![cfg(feature = "simd-json")]

use crate::ast::BinOp;
use crate::eval::Val as OwnedVal;
use crate::eval::borrowed::{Arena, Val as BVal};
use crate::pipeline::{
    Pipeline, Source, Sink, Stage, BodyKernel, NumOp,
};
use crate::row::Row;
use crate::composed_tape::TapeRow;
use crate::strref::TapeData;
use crate::unified::{
    Stage as USt,
    Identity as UIdentity,
    Composed as UComposed,
    Filter as UFilter,
    MapField as UMapField,
    MapFieldChain as UMapFieldChain,
    FlatMapField as UFlatMapField,
    Take as UTake,
    Skip as USkip,
    CountSink, SumSink, MinSink, MaxSink, AvgSink,
    FirstSink, LastSink, CollectSink,
    run_pipeline as urun,
};
use std::sync::Arc;

#[derive(Copy, Clone)]
pub enum SinkKind {
    Count, Sum, Min, Max, Avg, First, Last, Collect,
}

impl SinkKind {
    fn from_pipeline_sink(s: &Sink) -> Option<Self> {
        Some(match s {
            Sink::Count               => SinkKind::Count,
            Sink::Numeric(NumOp::Sum) => SinkKind::Sum,
            Sink::Numeric(NumOp::Min) => SinkKind::Min,
            Sink::Numeric(NumOp::Max) => SinkKind::Max,
            Sink::Numeric(NumOp::Avg) => SinkKind::Avg,
            Sink::First               => SinkKind::First,
            Sink::Last                => SinkKind::Last,
            Sink::Collect             => SinkKind::Collect,
        })
    }
}

fn lower_stages<'a>(
    stages: &[Stage],
    kernels: &[BodyKernel],
) -> Option<Box<dyn USt<TapeRow<'a>> + 'a>> {
    let mut chain: Box<dyn USt<TapeRow<'a>> + 'a> = Box::new(UIdentity::new());
    for (st, k) in stages.iter().zip(kernels.iter()) {
        let next: Box<dyn USt<TapeRow<'a>> + 'a> = lower_stage(st, k)?;
        chain = Box::new(UComposed::new(chain, next));
    }
    Some(chain)
}

fn lower_stage<'a>(stage: &Stage, kernel: &BodyKernel) -> Option<Box<dyn USt<TapeRow<'a>> + 'a>> {
    match (stage, kernel) {
        (Stage::FlatMap(_), BodyKernel::FieldRead(name)) =>
            Some(Box::new(UFlatMapField::new(Arc::clone(name)))),
        (Stage::Take(n), _) => Some(Box::new(UTake::new(*n))),
        (Stage::Skip(n), _) => Some(Box::new(USkip::new(*n))),

        (Stage::Map(_), BodyKernel::FieldRead(name)) =>
            Some(Box::new(UMapField::new(Arc::clone(name)))),
        (Stage::Map(_), BodyKernel::FieldChain(keys)) =>
            Some(Box::new(UMapFieldChain::new(keys.iter().cloned().collect()))),

        (Stage::Filter(_), BodyKernel::FieldCmpLit(field, op, lit)) => {
            let f = Arc::clone(field);
            let owned_lit = lit.clone();
            let op = *op;
            let pred = move |v: &TapeRow<'_>| -> bool {
                let val = match v.get_field(&f) { Some(x) => x, None => return false };
                trow_cmp_owned(&val, &op, &owned_lit)
            };
            Some(Box::new(UFilter::<TapeRow<'a>, _>::new(pred)))
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
            Some(Box::new(UFilter::<TapeRow<'a>, _>::new(pred)))
        }
        (Stage::Filter(_), BodyKernel::CurrentCmpLit(op, lit)) => {
            let owned_lit = lit.clone();
            let op = *op;
            let pred = move |v: &TapeRow<'_>| -> bool {
                trow_cmp_owned(v, &op, &owned_lit)
            };
            Some(Box::new(UFilter::<TapeRow<'a>, _>::new(pred)))
        }
        (Stage::Filter(_), BodyKernel::ConstBool(b)) => {
            let b = *b;
            let pred = move |_: &TapeRow<'_>| -> bool { b };
            Some(Box::new(UFilter::<TapeRow<'a>, _>::new(pred)))
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

fn dispatch_sink<'a, I: Iterator<Item = TapeRow<'a>>>(
    arena: &'a Arena,
    rows: I,
    stages: &dyn USt<TapeRow<'a>>,
    kind: SinkKind,
) -> BVal<'a> {
    match kind {
        SinkKind::Count   => urun::<TapeRow<'a>, CountSink>(arena, rows, stages),
        SinkKind::Sum     => urun::<TapeRow<'a>, SumSink>(arena, rows, stages),
        SinkKind::Min     => urun::<TapeRow<'a>, MinSink>(arena, rows, stages),
        SinkKind::Max     => urun::<TapeRow<'a>, MaxSink>(arena, rows, stages),
        SinkKind::Avg     => urun::<TapeRow<'a>, AvgSink>(arena, rows, stages),
        SinkKind::First   => urun::<TapeRow<'a>, FirstSink>(arena, rows, stages),
        SinkKind::Last    => urun::<TapeRow<'a>, LastSink>(arena, rows, stages),
        SinkKind::Collect => urun::<TapeRow<'a>, CollectSink>(arena, rows, stages),
    }
}

/// Try to run a Pipeline through the unified borrowed-tape substrate.
/// Returns None when shape unsupported.
pub fn try_run_borrow_tape<'a>(
    p: &Pipeline,
    tape: &'a TapeData,
    arena: &'a Arena,
) -> Option<Result<BVal<'a>, crate::eval::EvalError>> {
    let chain: Vec<Arc<str>> = match &p.source {
        Source::FieldChain { keys } => keys.iter().cloned().collect(),
        _ => return None,
    };

    let sink_kind = SinkKind::from_pipeline_sink(&p.sink)?;
    let stage_chain = lower_stages(&p.stages, &p.stage_kernels)?;

    let chain_refs: Vec<&str> = chain.iter().map(|a| a.as_ref()).collect();
    let arr_idx = crate::strref::tape_walk_field_chain(tape, &chain_refs)?;
    let arr_row = TapeRow::new(tape, arr_idx as u32);
    let rows = arr_row.array_children()?;

    let out = dispatch_sink(arena, rows, &*stage_chain, sink_kind);
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

    #[test]
    fn flat_map_count_unified() {
        let v = serde_json::json!({
            "orders": [
                {"items": [{"p": 1}, {"p": 2}]},
                {"items": [{"p": 3}, {"p": 4}, {"p": 5}]},
                {"items": [{"p": 6}]},
            ]
        });
        let bytes = serde_json::to_vec(&v).unwrap();
        let arena = Arena::new();
        let parsed = parser::parse("$.orders.flat_map(items).count()").unwrap();
        let p = Pipeline::lower(&parsed).unwrap();
        let tape = TapeData::parse(bytes).unwrap();
        let r = try_run_borrow_tape(&p, &tape, &arena).unwrap().unwrap();
        let owned = r.to_owned_val();
        assert!(matches!(owned, crate::eval::Val::Int(6)), "got {:?}", owned);
    }

    #[test]
    fn flat_map_filter_count_unified() {
        let v = serde_json::json!({
            "orders": [
                {"items": [{"p": 1}, {"p": 20}, {"p": 5}]},
                {"items": [{"p": 30}, {"p": 4}, {"p": 50}]},
            ]
        });
        let bytes = serde_json::to_vec(&v).unwrap();
        let arena = Arena::new();
        let parsed = parser::parse("$.orders.flat_map(items).filter(p > 10).count()").unwrap();
        let p = Pipeline::lower(&parsed).unwrap();
        let tape = TapeData::parse(bytes).unwrap();
        let r = try_run_borrow_tape(&p, &tape, &arena).unwrap().unwrap();
        let owned = r.to_owned_val();
        assert!(matches!(owned, crate::eval::Val::Int(3)), "got {:?}", owned);
    }
}
