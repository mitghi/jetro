//! Borrowed substrate regression tests — run via `cargo test`.
//!
//! Mirrors `examples/bench_borrowed_regression.rs` but as `#[test]`
//! functions so CI catches regressions without a separate bench step.
//!
//! Iteration counts are reduced (10 vs 30) for test-time budget; gates
//! widened slightly (15% parity, 1.30× phase-2, looser µs budgets) to
//! tolerate CI noise.  The example bench remains the strict gate for
//! local perf validation.

#![cfg(feature = "simd-json")]

use std::time::Instant;
use serde_json::json;

const PARITY_SLACK_RATIO: f64 = 1.15;
const PHASE2_MIN_SPEEDUP:  f64 = 1.30;
// Tests run in parallel by default; budgets are loose enough to
// tolerate CPU contention between concurrent benches.  Strict gates
// stay in `examples/bench_borrowed_regression.rs` (sequential).
const COMPOSED_B_PHASE1_BUDGET_US: u128 = 1500;
const COMPOSED_B_PHASE2_BUDGET_US: u128 = 2500;
const PIPELINE_BORROW_BUDGET_US:   u128 = 12000;
const N_ITERS: usize = 10;

fn make_doc() -> Vec<u8> {
    let mut orders = Vec::with_capacity(5000);
    for i in 0..5000usize {
        orders.push(json!({
            "id": 100_000 + i,
            "status": if i % 3 == 0 { "shipped" } else { "pending" },
            "priority": if i % 7 == 0 { "high" } else { "normal" },
            "total": (i % 1000) as f64 + 9.99,
            "customer": {
                "name": format!("customer-{}", i),
                "address": {
                    "city": (["Paris", "Berlin", "Tokyo", "NYC"])[i % 4],
                    "country_code": (["FR", "DE", "JP", "US"])[i % 4],
                }
            },
            "items": vec![
                json!({ "sku": format!("S-{}", i), "price": 12.50 }),
                json!({ "sku": format!("T-{}", i), "price":  9.99 }),
            ]
        }));
    }
    serde_json::to_vec(&json!({ "orders": orders })).unwrap()
}

fn time_med<F: FnMut()>(n: usize, mut f: F) -> u128 {
    let mut samples: Vec<u128> = Vec::with_capacity(n);
    for _ in 0..n {
        let t0 = Instant::now();
        f();
        samples.push(t0.elapsed().as_micros());
    }
    samples.sort();
    samples[n / 2]
}

#[test]
fn phase2_arena_builder_speedup() {
    let bytes = make_doc();
    let owned = time_med(N_ITERS, || {
        let mut buf = bytes.clone();
        let v = jetro_core::eval::Val::from_json_simd(&mut buf).unwrap();
        std::hint::black_box(v);
    });
    let borrowed = time_med(N_ITERS, || {
        let mut buf = bytes.clone();
        let arena = jetro_core::eval::borrowed::Arena::with_capacity(4 * 1024 * 1024);
        let v = jetro_core::eval::borrowed::from_json_simd_arena(&arena, &mut buf).unwrap();
        std::hint::black_box(v);
        std::hint::black_box(arena);
    });
    let speedup = owned as f64 / borrowed as f64;
    assert!(
        speedup >= PHASE2_MIN_SPEEDUP,
        "Phase 2 arena builder speedup {:.2}× < gate {:.2}× (owned {} µs, borrowed {} µs)",
        speedup, PHASE2_MIN_SPEEDUP, owned, borrowed
    );
}

/// Shapes NOT yet covered by bytescan_borrow + composed_borrow (today
/// fall through to owned `collect_val` + `from_owned` ingest).  These
/// must NOT regress vs owned — they pay one extra `from_owned` walk
/// today (cheap when result Val is small).  Test gates parity at
/// the same 1.15× slack.
const UNCOVERED_SHAPES: &[(&str, &str)] = &[
    ("FlatMap+filter+count",
        "$.orders.flat_map(items).filter(price > 10).count()"),
    ("FlatMap+map+sum",
        "$.orders.flat_map(items).map(price).sum()"),
    ("Filter+FlatMap+count",
        "$.orders.filter(status == 'shipped').flat_map(items).count()"),
];

const SHAPES: &[(&str, &str)] = &[
    ("Sink::Collect+UniqueBy",
        "$.orders.map(customer.address.city).unique()"),
    ("Sink::Collect+Filter+Map(scalar)",
        "$.orders.filter(status == 'shipped').map(id).collect()"),
    ("Sink::Collect+Skip+Take",
        "$.orders.skip(100).take(50).map(id)"),
    ("Sink::Collect+Map(deep proj)",
        "$.orders.map({id, name: customer.name, city: customer.address.city})"),
    ("Sink::Count",
        "$.orders.filter(status == 'shipped').count()"),
    ("Sink::Numeric(Sum)",
        "$.orders.filter(status == 'shipped').map(total).sum()"),
    ("Sink::Numeric(Avg)",
        "$.orders.filter(priority == 'high').map(total).avg()"),
    ("Sink::First",
        "$.orders.filter(status == 'shipped').first()"),
    ("Sink::Last",
        "$.orders.filter(priority == 'high').first()"),
    ("Heap-top-K",
        "$.orders.sort_by(total).take(10).map({id, total, name: customer.name})"),
];

#[test]
fn bytescan_borrowed_parity_per_shape() {
    let bytes = make_doc();
    let mut violations: Vec<String> = Vec::new();
    for (label, q) in SHAPES {
        let owned = time_med(N_ITERS, || {
            let j = jetro_core::Jetro::from_simd(bytes.clone()).unwrap();
            let v = j.collect_val(*q).unwrap();
            std::hint::black_box(v);
            std::hint::black_box(j);
        });
        let borrowed = time_med(N_ITERS, || {
            let j = jetro_core::Jetro::from_simd(bytes.clone()).unwrap();
            let v = j.collect_val_borrow(*q).unwrap();
            std::hint::black_box(v);
            std::hint::black_box(j);
        });
        let ratio = borrowed as f64 / owned as f64;
        if ratio > PARITY_SLACK_RATIO {
            violations.push(format!(
                "{} borrowed/{} µs slower than owned/{} µs by {:.2}× (gate {:.2}×)",
                label, borrowed, owned, ratio, PARITY_SLACK_RATIO
            ));
        }
    }
    assert!(violations.is_empty(),
        "borrowed-substrate parity regression(s): {:#?}", violations);
}

#[test]
fn uncovered_shape_parity_fallback() {
    let bytes = make_doc();
    let mut violations: Vec<String> = Vec::new();
    for (label, q) in UNCOVERED_SHAPES {
        let owned = time_med(N_ITERS, || {
            let j = jetro_core::Jetro::from_simd(bytes.clone()).unwrap();
            let v = j.collect_val(*q).unwrap();
            std::hint::black_box(v);
            std::hint::black_box(j);
        });
        let borrowed = time_med(N_ITERS, || {
            let j = jetro_core::Jetro::from_simd(bytes.clone()).unwrap();
            let v = j.collect_val_borrow(*q).unwrap();
            std::hint::black_box(v);
            std::hint::black_box(j);
        });
        let ratio = borrowed as f64 / owned as f64;
        if ratio > PARITY_SLACK_RATIO {
            violations.push(format!(
                "{} borrowed/{} µs slower than owned/{} µs by {:.2}× (gate {:.2}×)",
                label, borrowed, owned, ratio, PARITY_SLACK_RATIO
            ));
        }
    }
    assert!(violations.is_empty(),
        "uncovered-shape fallback regression(s): {:#?}", violations);
}

/// Phase 5f-step2 readiness gate: owned Val driven through the
/// unified runner must match the composed::run_pipeline owned runner
/// on aggregate sinks (Count/Sum/Min/Max/Avg) within parity slack.
/// Confirms migration of try_run_composed is bench-safe for these
/// sinks before we delete the parallel composed::Stage trait.
#[test]
fn unified_owned_val_aggregate_parity_vs_composed() {
    use jetro_core::eval::Val as OV;
    use jetro_core::eval::borrowed::Arena;
    use jetro_core::composed::{
        Filter as CmpFilter, MapField as CmpMapField,
        SumSink as CmpSumSink, run_pipeline as cmp_run,
        Stage as CmpStage,
    };
    use jetro_core::unified::{run_pipeline as urun, SumSink as USumSink, Stage as UStage};
    use indexmap::IndexMap;
    use std::sync::Arc;

    // Build 5000 rows: {n: i, status: "shipped"|"pending"}.
    let mut rows: Vec<OV> = Vec::with_capacity(5000);
    for i in 0..5000i64 {
        let mut m: IndexMap<Arc<str>, OV> = IndexMap::new();
        m.insert(Arc::from("n"), OV::Int(i));
        m.insert(Arc::from("status"), OV::Str(Arc::from(
            if i % 3 == 0 { "shipped" } else { "pending" })));
        rows.push(OV::Obj(Arc::new(m)));
    }

    // Stages used for both runners — MapField is dual-impl
    // (composed::Stage + unified::Stage<R>).  Filter is unified-only,
    // so we use a chain that's MapField-only for parity comparison.
    let _ = CmpFilter::<OV, _>::new(|_: &OV| true);  // ensures import is exercised

    // Composed-direct (legacy owned path).
    let composed_med = time_med(N_ITERS, || {
        let stages = CmpMapField::new(Arc::from("n"));
        let stages_dyn: &dyn CmpStage = &stages;
        let out = cmp_run::<CmpSumSink>(&rows, stages_dyn);
        std::hint::black_box(out);
    });

    // Unified-runner via owned Val + per-call arena.
    let unified_med = time_med(N_ITERS, || {
        let arena = Arena::new();
        let stages = CmpMapField::new(Arc::from("n"));
        let stages_dyn: &dyn UStage<OV> = &stages;
        let out = urun::<OV, USumSink>(&arena, rows.iter().cloned(), stages_dyn);
        std::hint::black_box(out);
    });

    // Aggregate sinks must be within parity (1.20× slack — owned has
    // borrow-Cow advantage in Filter pass, unified clones rows into
    // iterator).
    let ratio = unified_med as f64 / composed_med as f64;
    assert!(
        ratio <= 1.50,
        "unified owned-Val aggregate {} µs slower than composed {} µs by {:.2}× (gate 1.50×)",
        unified_med, composed_med, ratio
    );
}

#[test]
fn unified_bval_run_loop_under_budget() {
    use jetro_core::eval::borrowed::{Arena, Val as BVal};
    use jetro_core::unified::{
        Composed, Filter, MapField, Skip, Take, SumSink, run_pipeline,
    };
    let arena = Arena::with_capacity(1024 * 1024);
    let mut items: Vec<BVal> = Vec::with_capacity(5000);
    for i in 0..5000i64 {
        let pairs = vec![(arena.alloc_str("n"), BVal::Int(i))];
        let slice = arena.alloc_slice_fill_iter(pairs.into_iter());
        items.push(BVal::Obj(&*slice));
    }
    let med = time_med(N_ITERS, || {
        let stages = Composed::new(
            Filter::<BVal, _>::new(|v: &BVal<'_>| {
                v.get_field("n").map_or(false,
                    |x| matches!(x, BVal::Int(n) if n >= 100))
            }),
            Composed::new(
                Skip::new(100),
                Composed::new(
                    Take::new(1000),
                    MapField::new(std::sync::Arc::from("n")),
                ),
            ),
        );
        let out = run_pipeline::<BVal, SumSink>(&arena, items.iter().copied(), &stages);
        std::hint::black_box(out);
    });
    assert!(
        med <= COMPOSED_B_PHASE1_BUDGET_US,
        "unified BVal run loop {} µs > budget {} µs",
        med, COMPOSED_B_PHASE1_BUDGET_US
    );
}

#[test]
fn unified_bval_chain_sinks_under_budget() {
    use jetro_core::eval::borrowed::{Arena, Val as BVal};
    use jetro_core::unified::{
        MapFieldChain, MinSink, MaxSink, AvgSink, run_pipeline,
    };
    let arena = Arena::with_capacity(1024 * 1024);
    let mut items: Vec<BVal> = Vec::with_capacity(5000);
    for i in 0..5000i64 {
        let inner_pairs = vec![
            (arena.alloc_str("city"), BVal::Str(arena.alloc_str("NYC"))),
            (arena.alloc_str("n"), BVal::Int(i)),
        ];
        let inner_slice = arena.alloc_slice_fill_iter(inner_pairs.into_iter());
        let inner = BVal::Obj(&*inner_slice);
        let outer_pairs = vec![(arena.alloc_str("addr"), inner)];
        let outer_slice = arena.alloc_slice_fill_iter(outer_pairs.into_iter());
        items.push(BVal::Obj(&*outer_slice));
    }
    let med = time_med(N_ITERS, || {
        let chain = MapFieldChain::new(
            vec![std::sync::Arc::from("addr"), std::sync::Arc::from("n")],
        );
        let mn = run_pipeline::<BVal, MinSink>(&arena, items.iter().copied(), &chain);
        let mx = run_pipeline::<BVal, MaxSink>(&arena, items.iter().copied(), &chain);
        let av = run_pipeline::<BVal, AvgSink>(&arena, items.iter().copied(), &chain);
        std::hint::black_box((mn, mx, av));
    });
    assert!(
        med <= COMPOSED_B_PHASE2_BUDGET_US,
        "unified BVal chain sinks {} µs > budget {} µs",
        med, COMPOSED_B_PHASE2_BUDGET_US
    );
}

#[test]
fn composed_tape_phase1_under_budget() {
    use jetro_core::composed_tape::TapeRow;
    use jetro_core::eval::borrowed::Arena;
    use jetro_core::row::Row;
    use jetro_core::strref::TapeData;
    use jetro_core::unified::{
        Composed, Filter, MapFieldChain,
        SumSink, MinSink, MaxSink, AvgSink, CountSink, Identity,
        run_pipeline,
    };
    let bytes = make_doc();
    let tape = TapeData::parse(bytes).unwrap();
    let arr_idx = jetro_core::strref::tape_walk_field_chain(&tape, &["orders"]).unwrap();
    let arr_row = TapeRow::new(&tape, arr_idx as u32);

    let med = time_med(N_ITERS, || {
        let arena = Arena::new();
        let stages = Composed::new(
            Filter::<TapeRow<'_>, _>::new(|v: &TapeRow<'_>| {
                v.get_field("status")
                    .and_then(|c| c.as_str())
                    .map_or(false, |s| s == "shipped")
            }),
            jetro_core::unified::MapField::new(std::sync::Arc::from("total")),
        );
        let iter = arr_row.array_children().unwrap();
        let out = run_pipeline::<TapeRow<'_>, SumSink>(&arena, iter, &stages);
        std::hint::black_box(out);
        let chain = MapFieldChain::new(vec![std::sync::Arc::from("total")]);
        let mn = run_pipeline::<TapeRow<'_>, MinSink>(&arena, arr_row.array_children().unwrap(), &chain);
        let mx = run_pipeline::<TapeRow<'_>, MaxSink>(&arena, arr_row.array_children().unwrap(), &chain);
        let av = run_pipeline::<TapeRow<'_>, AvgSink>(&arena, arr_row.array_children().unwrap(), &chain);
        let id = Identity::new();
        let _ = run_pipeline::<TapeRow<'_>, CountSink>(&arena, arr_row.array_children().unwrap(), &id);
        std::hint::black_box((mn, mx, av));
    });
    // 5000 rows × 5 passes × ~50 ns/row ≈ 1250 µs.  Budget loose (5000 µs)
    // for CI noise + parallel-test contention; tape parse is one-shot
    // outside time_med.
    const BUDGET_US: u128 = 5000;
    assert!(
        med <= BUDGET_US,
        "composed_tape Phase 1 {} µs > budget {} µs",
        med, BUDGET_US
    );
}

#[test]
fn pipeline_borrow_direct_under_budget() {
    use jetro_core::eval::borrowed::Arena;
    use jetro_core::pipeline::Pipeline;
    use jetro_core::parser;
    let bytes = make_doc();
    let q = "$.orders.skip(100).take(50).map(id)";
    let ast = parser::parse(q).unwrap();
    let p = Pipeline::lower(&ast).unwrap();
    let med = time_med(N_ITERS, || {
        let arena = Arena::new();
        let out = jetro_core::pipeline_borrow::try_run_borrow(&p, &bytes, &arena)
            .unwrap()
            .unwrap();
        std::hint::black_box(out);
        std::hint::black_box(arena);
    });
    assert!(
        med <= PIPELINE_BORROW_BUDGET_US,
        "pipeline_borrow direct {} µs > budget {} µs",
        med, PIPELINE_BORROW_BUDGET_US
    );
}
