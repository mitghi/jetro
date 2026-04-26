//! Lower a Pipeline IR to the borrowed `composed_borrow` substrate.
//!
//! Phase 3 wiring — gated by env var `JETRO_BORROWED_COMPOSED=1`.
//! Runs AFTER `bytescan::try_run_borrow` declines and BEFORE the
//! owned-then-ingest fallback.  Falls back to None on any unsupported
//! shape (StageB chain only covers a subset of Pipeline IR today).
//!
//! ## Coverage today
//!
//! Source:    Source::FieldChain
//! Stages:    Filter(FieldCmpLit) / Filter(FieldChainCmpLit) /
//!            Filter(CurrentCmpLit) /
//!            Map(FieldRead) / Map(FieldChain) /
//!            Take(n) / Skip(n)
//! Sinks:     Count / Numeric(Sum/Min/Max/Avg) / First / Last / Collect
//!
//! Out of scope (returns None — fallback to owned path):
//!   - FlatMap / Sort / UniqueBy / Reverse / GroupBy
//!   - Map(ObjProject) / Map(FString) / Map(Arith) / Map(Generic)
//!   - non-FieldChain Source
//!   - lambdas / closures / user methods

use crate::ast::BinOp;
use crate::eval::Val as OwnedVal;
use crate::eval::borrowed::{Arena, Val as BVal};
use crate::pipeline::{
    Pipeline, Source, Sink, Stage, BodyKernel, NumOp,
};
use crate::composed_borrow::{
    StageB, IdentityB, ComposedB, FilterB, MapFieldB, MapFieldChainB,
    TakeB, SkipB,
    CountSinkB, SumSinkB, MinSinkB, MaxSinkB, AvgSinkB,
    FirstSinkB, LastSinkB, CollectSinkB,
    run_pipeline_b,
};
use std::sync::Arc;

/// Sink dispatch — runtime selection across the 8 SinkB impls.  One
/// monomorphisation per variant; called after stage chain folds.
#[derive(Copy, Clone)]
pub enum SinkBKind {
    Count,
    Sum, Min, Max, Avg,
    First, Last,
    Collect,
}

impl SinkBKind {
    fn from_pipeline_sink(s: &Sink) -> Option<Self> {
        Some(match s {
            Sink::Count => SinkBKind::Count,
            Sink::Numeric(NumOp::Sum) => SinkBKind::Sum,
            Sink::Numeric(NumOp::Min) => SinkBKind::Min,
            Sink::Numeric(NumOp::Max) => SinkBKind::Max,
            Sink::Numeric(NumOp::Avg) => SinkBKind::Avg,
            Sink::First => SinkBKind::First,
            Sink::Last  => SinkBKind::Last,
            Sink::Collect => SinkBKind::Collect,
        })
    }
}

/// Build a `Box<dyn StageB>` chain from the Pipeline's stages + kernels.
/// Returns None on first unsupported stage/kernel.
fn lower_stages(stages: &[Stage], kernels: &[BodyKernel]) -> Option<Box<dyn StageB>> {
    let mut chain: Box<dyn StageB> = Box::new(IdentityB);
    for (st, k) in stages.iter().zip(kernels.iter()) {
        let next: Box<dyn StageB> = lower_stage(st, k)?;
        chain = Box::new(ComposedB { a: chain, b: next });
    }
    Some(chain)
}

fn lower_stage(stage: &Stage, kernel: &BodyKernel) -> Option<Box<dyn StageB>> {
    match (stage, kernel) {
        (Stage::Take(n), _) => Some(Box::new(TakeB::new(*n))),
        (Stage::Skip(n), _) => Some(Box::new(SkipB::new(*n))),

        (Stage::Map(_), BodyKernel::FieldRead(name)) => {
            Some(Box::new(MapFieldB { field: Arc::clone(name) }))
        }
        (Stage::Map(_), BodyKernel::FieldChain(keys)) => {
            Some(Box::new(MapFieldChainB { chain: keys.iter().cloned().collect() }))
        }

        (Stage::Filter(_), BodyKernel::FieldCmpLit(field, op, lit)) => {
            let f = Arc::clone(field);
            let owned_lit = lit.clone();
            let op = *op;
            let pred = move |v: &BVal<'_>| -> bool {
                let val = match v.get_field(&f) { Some(x) => x, None => return false };
                bval_cmp_owned(&val, &op, &owned_lit)
            };
            Some(Box::new(FilterB { pred }))
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
            Some(Box::new(FilterB { pred }))
        }
        (Stage::Filter(_), BodyKernel::CurrentCmpLit(op, lit)) => {
            let owned_lit = lit.clone();
            let op = *op;
            let pred = move |v: &BVal<'_>| -> bool {
                bval_cmp_owned(v, &op, &owned_lit)
            };
            Some(Box::new(FilterB { pred }))
        }
        (Stage::Filter(_), BodyKernel::ConstBool(b)) => {
            let b = *b;
            let pred = move |_: &BVal<'_>| -> bool { b };
            Some(Box::new(FilterB { pred }))
        }

        _ => None,
    }
}

/// Compare a borrowed `BVal` against an owned literal `Val`.  Numeric
/// promotion follows the same rules as `pipeline.rs` filter eval.
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

/// Run a pipeline through the borrowed-composed substrate.  Caller
/// provides the `BVal::Arr` source array (from `walk_path` over the
/// arena-built root) and an arena.
fn dispatch_sink<'a>(
    arena: &'a Arena,
    arr: &[BVal<'a>],
    stages: &dyn StageB,
    kind: SinkBKind,
) -> BVal<'a> {
    match kind {
        SinkBKind::Count   => run_pipeline_b::<CountSinkB>(arena, arr, stages),
        SinkBKind::Sum     => run_pipeline_b::<SumSinkB>(arena, arr, stages),
        SinkBKind::Min     => run_pipeline_b::<MinSinkB>(arena, arr, stages),
        SinkBKind::Max     => run_pipeline_b::<MaxSinkB>(arena, arr, stages),
        SinkBKind::Avg     => run_pipeline_b::<AvgSinkB>(arena, arr, stages),
        SinkBKind::First   => run_pipeline_b::<FirstSinkB>(arena, arr, stages),
        SinkBKind::Last    => run_pipeline_b::<LastSinkB>(arena, arr, stages),
        SinkBKind::Collect => run_pipeline_b::<CollectSinkB>(arena, arr, stages),
    }
}

/// Try to run a Pipeline through the borrowed-composed substrate.
/// Caller passes the arena (lifetime `'a`) and raw bytes; this fn
/// rebuilds the root from bytes via `from_json_simd_arena` and walks
/// the source FieldChain to the array.
///
/// Returns `None` when the Pipeline shape is not yet covered by
/// composed_borrow lowering — caller should fall back to the owned
/// path.
#[cfg(feature = "simd-json")]
pub fn try_run_borrow<'a>(
    p: &Pipeline,
    raw_bytes: &[u8],
    arena: &'a Arena,
) -> Option<Result<BVal<'a>, crate::eval::EvalError>> {
    // Source: FieldChain only.
    let chain: Vec<Arc<str>> = match &p.source {
        Source::FieldChain { keys } => keys.iter().cloned().collect(),
        _ => return None,
    };

    // Sink lowering.
    let sink_kind = SinkBKind::from_pipeline_sink(&p.sink)?;

    // Stage chain lowering.
    let stage_chain = lower_stages(&p.stages, &p.stage_kernels)?;

    // Parse + arena-build root.
    let mut buf: Vec<u8> = raw_bytes.to_vec();
    let root = match crate::eval::borrowed::from_json_simd_arena(arena, &mut buf) {
        Ok(v) => v,
        Err(e) => return Some(Err(crate::eval::EvalError(e))),
    };

    // Walk source chain to array.
    let chain_refs: Vec<&str> = chain.iter().map(|a| a.as_ref()).collect();
    let arr_val = root.walk_path(&chain_refs).unwrap_or(BVal::Null);
    let arr = match arr_val {
        BVal::Arr(slice) => slice,
        _ => return None,
    };

    let out = dispatch_sink(arena, arr, &*stage_chain, sink_kind);
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
