//! Lower a Pipeline IR to the UNIFIED borrowed substrate over
//! `borrowed::Val<'a>` (BVal).
//!
//! Phase 3 migration: this module previously consumed
//! `composed_borrow`'s parallel StageB + SinkB.  It now consumes
//! `unified::Stage<R>` + `unified::Sink` directly with R = BVal<'a>,
//! identical to how `pipeline_tape_borrow` does it for TapeRow.
//!
//! NOT WIRED into `Jetro::collect_val_borrow` by default.  Reason
//! documented previously: pipeline_borrow does a full
//! `from_json_simd_arena` parse (~3.7 ms on 1.1 MB) before running
//! the borrowed Stage chain, while `pipeline_tape_borrow` reuses the
//! cached tape (~0 ms after first call).  When bytescan_borrow
//! declines, the tape-source path is preferred.  Substrate stays
//! alive via direct unit tests + regression bench gate; future
//! `collect_val_arena_borrow` API (Phase 4) will route through here
//! for callers that explicitly want the arena Val tree built once.

#![cfg(feature = "simd-json")]

use crate::ast::BinOp;
use crate::eval::Val as OwnedVal;
use crate::eval::borrowed::{Arena, Val as BVal};
use crate::pipeline::{
    Pipeline, Source, Sink, Stage, BodyKernel, NumOp,
};
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
        match s {
            Sink::Count               => Some(SinkKind::Count),
            Sink::Numeric(NumOp::Sum) => Some(SinkKind::Sum),
            Sink::Numeric(NumOp::Min) => Some(SinkKind::Min),
            Sink::Numeric(NumOp::Max) => Some(SinkKind::Max),
            Sink::Numeric(NumOp::Avg) => Some(SinkKind::Avg),
            Sink::First               => Some(SinkKind::First),
            Sink::Last                => Some(SinkKind::Last),
            Sink::Collect             => Some(SinkKind::Collect),
            Sink::ApproxCountDistinct => None,
        }
    }
}

fn lower_stages<'a>(
    stages: &[Stage],
    kernels: &[BodyKernel],
) -> Option<Box<dyn USt<BVal<'a>> + 'a>> {
    let mut chain: Box<dyn USt<BVal<'a>> + 'a> = Box::new(UIdentity::new());
    for (st, k) in stages.iter().zip(kernels.iter()) {
        let next: Box<dyn USt<BVal<'a>> + 'a> = lower_stage(st, k)?;
        chain = Box::new(UComposed::new(chain, next));
    }
    Some(chain)
}

fn lower_stage<'a>(stage: &Stage, kernel: &BodyKernel) -> Option<Box<dyn USt<BVal<'a>> + 'a>> {
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
            let pred = move |v: &BVal<'_>| -> bool {
                let val = match v.get_field(&f) { Some(x) => x, None => return false };
                bval_cmp_owned(&val, &op, &owned_lit)
            };
            Some(Box::new(UFilter::<BVal<'a>, _>::new(pred)))
        }
        (Stage::Filter(_), BodyKernel::FieldChainCmpLit(keys, op, lit)) => {
            let chain: Vec<Arc<str>> = keys.iter().cloned().collect();
            let owned_lit = lit.clone();
            let op = *op;
            let pred = move |v: &BVal<'_>| -> bool {
                let chain_refs: Vec<&str> = chain.iter().map(|a| a.as_ref()).collect();
                let val = match v.walk_path(&chain_refs) { Some(x) => x, None => return false };
                bval_cmp_owned(&val, &op, &owned_lit)
            };
            Some(Box::new(UFilter::<BVal<'a>, _>::new(pred)))
        }
        (Stage::Filter(_), BodyKernel::CurrentCmpLit(op, lit)) => {
            let owned_lit = lit.clone();
            let op = *op;
            let pred = move |v: &BVal<'_>| -> bool {
                bval_cmp_owned(v, &op, &owned_lit)
            };
            Some(Box::new(UFilter::<BVal<'a>, _>::new(pred)))
        }
        (Stage::Filter(_), BodyKernel::ConstBool(b)) => {
            let b = *b;
            let pred = move |_: &BVal<'_>| -> bool { b };
            Some(Box::new(UFilter::<BVal<'a>, _>::new(pred)))
        }

        _ => None,
    }
}

fn bval_cmp_owned(b: &BVal<'_>, op: &BinOp, lit: &OwnedVal) -> bool {
    use BinOp::*;
    match (b, lit) {
        (BVal::Int(a), OwnedVal::Int(c)) => apply_cmp_i(*a, *c, op),
        (BVal::Int(a), OwnedVal::Float(c)) => apply_cmp_f(*a as f64, *c, op),
        (BVal::Float(a), OwnedVal::Int(c)) => apply_cmp_f(*a, *c as f64, op),
        (BVal::Float(a), OwnedVal::Float(c)) => apply_cmp_f(*a, *c, op),
        (BVal::Bool(a), OwnedVal::Bool(c)) => apply_cmp_b(*a, *c, op),
        (BVal::Str(a), OwnedVal::Str(c)) => apply_cmp_s(a, c.as_ref(), op),
        (BVal::Null, OwnedVal::Null) => matches!(op, Eq),
        (BVal::Null, _) | (_, OwnedVal::Null) => matches!(op, Neq),
        _ => false,
    }
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

fn dispatch_sink<'a, I: Iterator<Item = BVal<'a>>>(
    arena: &'a Arena,
    rows: I,
    stages: &dyn USt<BVal<'a>>,
    kind: SinkKind,
) -> BVal<'a> {
    match kind {
        SinkKind::Count   => urun::<BVal<'a>, CountSink>(arena, rows, stages),
        SinkKind::Sum     => urun::<BVal<'a>, SumSink>(arena, rows, stages),
        SinkKind::Min     => urun::<BVal<'a>, MinSink>(arena, rows, stages),
        SinkKind::Max     => urun::<BVal<'a>, MaxSink>(arena, rows, stages),
        SinkKind::Avg     => urun::<BVal<'a>, AvgSink>(arena, rows, stages),
        SinkKind::First   => urun::<BVal<'a>, FirstSink>(arena, rows, stages),
        SinkKind::Last    => urun::<BVal<'a>, LastSink>(arena, rows, stages),
        SinkKind::Collect => urun::<BVal<'a>, CollectSink>(arena, rows, stages),
    }
}

/// Try to run a Pipeline through the unified borrowed-Val substrate.
/// Builds the root via `from_json_simd_arena` (full parse + arena Val
/// tree) then walks the source FieldChain to the array.  Returns None
/// when shape unsupported.
#[cfg(feature = "simd-json")]
pub fn try_run_borrow<'a>(
    p: &Pipeline,
    raw_bytes: &[u8],
    arena: &'a Arena,
) -> Option<Result<BVal<'a>, crate::eval::EvalError>> {
    let chain: Vec<Arc<str>> = match &p.source {
        Source::FieldChain { keys } => keys.iter().cloned().collect(),
        _ => return None,
    };

    let sink_kind = SinkKind::from_pipeline_sink(&p.sink)?;
    let stage_chain = lower_stages(&p.stages, &p.stage_kernels)?;

    let mut buf: Vec<u8> = raw_bytes.to_vec();
    let root = match crate::eval::borrowed::from_json_simd_arena(arena, &mut buf) {
        Ok(v) => v,
        Err(e) => return Some(Err(crate::eval::EvalError(e))),
    };

    let chain_refs: Vec<&str> = chain.iter().map(|a| a.as_ref()).collect();
    let arr_val = root.walk_path(&chain_refs).unwrap_or(BVal::Null);
    let arr = match arr_val {
        BVal::Arr(slice) => slice,
        _ => return None,
    };

    let out = dispatch_sink(arena, arr.iter().copied(), &*stage_chain, sink_kind);
    Some(Ok(out))
}

#[cfg(all(test, feature = "simd-json"))]
mod tests {
    use super::*;
    use crate::pipeline::Pipeline;
    use crate::parser;

    fn run(expr: &str, doc: &[u8]) -> Option<crate::eval::Val> {
        let arena = Arena::new();
        let parsed = parser::parse(expr).ok()?;
        let p: Pipeline = Pipeline::lower(&parsed)?;
        let r = try_run_borrow(&p, doc, &arena)?;
        let bv = r.ok()?;
        Some(bv.to_owned_val())
    }

    fn doc() -> Vec<u8> {
        let v = serde_json::json!({
            "orders": [
                {"id": 1, "total": 10},
                {"id": 2, "total": 20},
                {"id": 3, "total": 30},
                {"id": 4, "total": 40},
            ]
        });
        serde_json::to_vec(&v).unwrap()
    }

    #[test]
    fn count_via_pipeline_borrow() {
        let d = doc();
        let v = run("$.orders.count()", &d).expect("count");
        assert!(matches!(v, crate::eval::Val::Int(4)), "got {:?}", v);
    }

    #[test]
    fn sum_total_via_pipeline_borrow() {
        let d = doc();
        let v = run("$.orders.map(total).sum()", &d).expect("sum");
        match v {
            crate::eval::Val::Int(n) => assert_eq!(n, 100),
            crate::eval::Val::Float(f) => assert!((f - 100.0).abs() < 1e-9),
            other => panic!("got {:?}", other),
        }
    }

    #[test]
    fn first_via_pipeline_borrow() {
        let d = doc();
        let v = run("$.orders.first()", &d).expect("first");
        match v {
            crate::eval::Val::Obj(_) | crate::eval::Val::ObjSmall(_) => {}
            other => panic!("expected Obj, got {:?}", other),
        }
    }

    #[test]
    fn skip_take_count_via_pipeline_borrow() {
        let d = doc();
        let v = run("$.orders.skip(1).take(2).count()", &d).expect("skip+take+count");
        assert!(matches!(v, crate::eval::Val::Int(2)), "got {:?}", v);
    }
}
