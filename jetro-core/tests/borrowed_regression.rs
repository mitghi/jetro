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
const COMPOSED_B_PHASE1_BUDGET_US: u128 = 400;
const COMPOSED_B_PHASE2_BUDGET_US: u128 = 800;
const PIPELINE_BORROW_BUDGET_US:   u128 = 9000;
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
fn composed_borrow_phase1_run_loop_under_budget() {
    use jetro_core::composed_borrow::{
        ComposedB, FilterB, MapFieldB, SkipB, TakeB,
        SumSinkB, run_pipeline_b,
    };
    use jetro_core::eval::borrowed::{Arena, Val as BVal};
    let arena = Arena::with_capacity(1024 * 1024);
    let mut items: Vec<BVal> = Vec::with_capacity(5000);
    for i in 0..5000i64 {
        let pairs = vec![(arena.alloc_str("n"), BVal::Int(i))];
        let slice = arena.alloc_slice_fill_iter(pairs.into_iter());
        items.push(BVal::Obj(&*slice));
    }
    let med = time_med(N_ITERS, || {
        let stages = ComposedB {
            a: FilterB { pred: |v: &BVal<'_>| {
                v.get_field("n").map_or(false,
                    |x| matches!(x, BVal::Int(n) if n >= 100))
            }},
            b: ComposedB {
                a: SkipB::new(100),
                b: ComposedB {
                    a: TakeB::new(1000),
                    b: MapFieldB { field: std::sync::Arc::from("n") },
                },
            },
        };
        let out = run_pipeline_b::<SumSinkB>(&arena, items.as_slice(), &stages);
        std::hint::black_box(out);
    });
    assert!(
        med <= COMPOSED_B_PHASE1_BUDGET_US,
        "composed_borrow Phase 1 run loop {} µs > budget {} µs",
        med, COMPOSED_B_PHASE1_BUDGET_US
    );
}

#[test]
fn composed_borrow_phase2_sinks_under_budget() {
    use jetro_core::composed_borrow::{
        MinSinkB, MaxSinkB, AvgSinkB, MapFieldChainB, run_pipeline_b,
    };
    use jetro_core::eval::borrowed::{Arena, Val as BVal};
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
        let chain = MapFieldChainB {
            chain: vec![std::sync::Arc::from("addr"), std::sync::Arc::from("n")],
        };
        let mn = run_pipeline_b::<MinSinkB>(&arena, items.as_slice(), &chain);
        let mx = run_pipeline_b::<MaxSinkB>(&arena, items.as_slice(), &chain);
        let av = run_pipeline_b::<AvgSinkB>(&arena, items.as_slice(), &chain);
        std::hint::black_box((mn, mx, av));
    });
    assert!(
        med <= COMPOSED_B_PHASE2_BUDGET_US,
        "composed_borrow Phase 2 sinks {} µs > budget {} µs",
        med, COMPOSED_B_PHASE2_BUDGET_US
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
