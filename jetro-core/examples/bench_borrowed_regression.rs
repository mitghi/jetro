//! Borrowed-substrate regression suite — locks in Phase 1/2/3+4 lite +
//! heap-top-K + composed_borrow Phase 1 perf so future commits cannot
//! silently regress them.
//!
//! Coverage:
//!   1. Phase 2 microbench  — owned `Val::from_json_simd`
//!                            vs `from_json_simd_arena`.
//!   2. Phase 3+4 lite      — owned `collect_val` vs `collect_val_borrow`
//!                            for every byte-scannable Sink shape.
//!   3. Heap-top-K          — sort_by(k).take(k).map(...) borrowed.
//!   4. Composed_borrow P1  — direct microbench of StageB + SinkB run loop.
//!
//! Regression gate:
//!   - Borrowed never slower than owned by more than `PARITY_SLACK_RATIO`
//!     (default 1.10 = 10%).
//!   - Phase 2 arena builder must be at least 1.4× faster than owned
//!     `Val::from_json_simd`.
//!   - Composed_borrow run loop must process 5000 elements in under
//!     `COMPOSED_B_BUDGET_US` µs.
//!
//! Exit code 0 = pass; exit code 1 = regression detected.
//!
//! Run:
//!   cargo run --release --example bench_borrowed_regression --features simd-json
//!
//! In CI, fail the build on non-zero exit.

use std::time::Instant;
use serde_json::json;

const PARITY_SLACK_RATIO: f64 = 1.10;
const PHASE2_MIN_SPEEDUP: f64 = 1.40;
const COMPOSED_B_BUDGET_US: u128 = 250;
const N_ITERS: usize = 30;

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
    let doc = json!({ "orders": orders });
    serde_json::to_vec(&doc).unwrap()
}

#[derive(Copy, Clone)]
#[allow(dead_code)]
struct Stat { best: u128, median: u128, mean: u128 }

fn time<F: FnMut()>(label: &str, n: usize, mut f: F) -> Stat {
    let mut samples: Vec<u128> = Vec::with_capacity(n);
    for _ in 0..n {
        let t0 = Instant::now();
        f();
        samples.push(t0.elapsed().as_micros());
    }
    samples.sort();
    let median = samples[n / 2];
    let best = samples[0];
    let mean: u128 = samples.iter().sum::<u128>() / n as u128;
    println!("{:<54}  best {:>6} µs  median {:>6} µs  mean {:>6} µs",
        label, best, median, mean);
    Stat { best, median, mean }
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
    ("Sink::Last (via First on different filter)",
        "$.orders.filter(priority == 'high').first()"),
    ("Heap-top-K (sort_by+take+map)",
        "$.orders.sort_by(total).take(10).map({id, total, name: customer.name})"),
];

fn bench_owned_vs_borrowed(bytes: &[u8]) -> Vec<(String, Stat, Stat)> {
    let mut results: Vec<(String, Stat, Stat)> = Vec::new();
    for (label, q) in SHAPES {
        println!("── {} ──", label);
        let owned = time("  owned     collect_val()", N_ITERS, || {
            let j = jetro_core::Jetro::from_simd(bytes.to_vec()).unwrap();
            let v = j.collect_val(*q).unwrap();
            std::hint::black_box(v);
            std::hint::black_box(j);
        });
        let borrowed = time("  borrowed  collect_val_borrow()", N_ITERS, || {
            let j = jetro_core::Jetro::from_simd(bytes.to_vec()).unwrap();
            let v = j.collect_val_borrow(*q).unwrap();
            std::hint::black_box(v);
            std::hint::black_box(j);
        });
        println!();
        results.push(((*label).to_string(), owned, borrowed));
    }
    results
}

fn bench_phase2_arena_build(bytes: &[u8]) -> (Stat, Stat) {
    println!("── Phase 2: arena builder vs owned Val::from_json_simd ──");
    let owned = time("  owned    Val::from_json_simd", N_ITERS, || {
        let mut buf = bytes.to_vec();
        let v = jetro_core::eval::Val::from_json_simd(&mut buf).unwrap();
        std::hint::black_box(v);
    });
    let borrowed = time("  borrowed from_json_simd_arena", N_ITERS, || {
        let mut buf = bytes.to_vec();
        let arena = jetro_core::eval::borrowed::Arena::with_capacity(4 * 1024 * 1024);
        let v = jetro_core::eval::borrowed::from_json_simd_arena(&arena, &mut buf).unwrap();
        std::hint::black_box(v);
        std::hint::black_box(arena);
    });
    println!();
    (owned, borrowed)
}

fn bench_composed_borrow_loop() -> Stat {
    use jetro_core::composed_borrow::{
        ComposedB, FilterB, MapFieldB, SkipB, TakeB,
        SumSinkB, run_pipeline_b,
    };
    use jetro_core::eval::borrowed::{Arena, Val as BVal};
    println!("── composed_borrow Phase 1: StageB run loop (5000 elems) ──");
    let arena = Arena::with_capacity(1024 * 1024);
    let mut items: Vec<BVal> = Vec::with_capacity(5000);
    for i in 0..5000i64 {
        let pairs = vec![(arena.alloc_str("n"), BVal::Int(i))];
        let slice = arena.alloc_slice_fill_iter(pairs.into_iter());
        items.push(BVal::Obj(&*slice));
    }
    let stat = time("  filter(n>=100).skip(100).take(1000).map(n).sum()",
        N_ITERS, || {
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
    println!();
    stat
}

fn bench_composed_borrow_phase2_sinks() -> Stat {
    use jetro_core::composed_borrow::{
        IdentityB, FirstSinkB, LastSinkB, CollectSinkB,
        MinSinkB, MaxSinkB, AvgSinkB, MapFieldChainB, run_pipeline_b,
    };
    use jetro_core::eval::borrowed::{Arena, Val as BVal};
    println!("── composed_borrow Phase 2: GAT-Acc sinks + chain stage ──");
    let arena = Arena::with_capacity(1024 * 1024);
    let mut items: Vec<BVal> = Vec::with_capacity(5000);
    for i in 0..5000i64 {
        // {addr: {city: "X", n: i}}
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
    let identity: Box<dyn jetro_core::composed_borrow::StageB> = Box::new(IdentityB);

    let _ = time("  First/Last/Collect — 5000 elems via Identity",
        N_ITERS, || {
        let f = run_pipeline_b::<FirstSinkB>(&arena, items.as_slice(), &*identity);
        let l = run_pipeline_b::<LastSinkB>(&arena, items.as_slice(), &*identity);
        let c = run_pipeline_b::<CollectSinkB>(&arena, items.as_slice(), &*identity);
        std::hint::black_box((f, l, c));
    });

    // Numeric sinks over MapFieldChain(addr.n)
    let stat = time("  Min/Max/Avg ∘ MapFieldChain[addr.n] — 5000 elems",
        N_ITERS, || {
        let chain = MapFieldChainB {
            chain: vec![std::sync::Arc::from("addr"), std::sync::Arc::from("n")],
        };
        let mn = run_pipeline_b::<MinSinkB>(&arena, items.as_slice(), &chain);
        let mx = run_pipeline_b::<MaxSinkB>(&arena, items.as_slice(), &chain);
        let av = run_pipeline_b::<AvgSinkB>(&arena, items.as_slice(), &chain);
        std::hint::black_box((mn, mx, av));
    });

    println!();
    stat
}

fn main() {
    let bytes = make_doc();
    println!("doc size = {} bytes ({:.2} MB)\n",
        bytes.len(), bytes.len() as f64 / 1.0e6);

    let mut violations: Vec<String> = Vec::new();

    // ── Phase 2 arena builder ──
    let (p2_owned, p2_borrowed) = bench_phase2_arena_build(&bytes);
    let p2_speedup = p2_owned.median as f64 / p2_borrowed.median as f64;
    println!("  Phase 2 speedup (median): {:.2}× (gate ≥ {:.2}×)",
        p2_speedup, PHASE2_MIN_SPEEDUP);
    if p2_speedup < PHASE2_MIN_SPEEDUP {
        violations.push(format!(
            "Phase 2 arena builder speedup {:.2}× < gate {:.2}×",
            p2_speedup, PHASE2_MIN_SPEEDUP));
    }
    println!();

    // ── Phase 3+4 lite parity ──
    let owned_vs_borrowed = bench_owned_vs_borrowed(&bytes);
    for (label, owned, borrowed) in &owned_vs_borrowed {
        let ratio = borrowed.median as f64 / owned.median as f64;
        if ratio > PARITY_SLACK_RATIO {
            violations.push(format!(
                "{} borrowed/{:?}µs slower than owned/{:?}µs by {:.2}× (gate {:.2}×)",
                label, borrowed.median, owned.median, ratio, PARITY_SLACK_RATIO));
        }
    }

    // ── composed_borrow Phase 1 ──
    let cb1 = bench_composed_borrow_loop();
    println!("  composed_borrow Phase 1 median: {} µs (gate ≤ {} µs)",
        cb1.median, COMPOSED_B_BUDGET_US);
    if cb1.median > COMPOSED_B_BUDGET_US {
        violations.push(format!(
            "composed_borrow run loop {} µs > budget {} µs",
            cb1.median, COMPOSED_B_BUDGET_US));
    }
    println!();

    // ── composed_borrow Phase 2 (GAT sinks + chain stage) ──
    let cb2 = bench_composed_borrow_phase2_sinks();
    const COMPOSED_B_PHASE2_BUDGET_US: u128 = 500;
    println!("  composed_borrow Phase 2 median: {} µs (gate ≤ {} µs)",
        cb2.median, COMPOSED_B_PHASE2_BUDGET_US);
    if cb2.median > COMPOSED_B_PHASE2_BUDGET_US {
        violations.push(format!(
            "composed_borrow Phase 2 sinks {} µs > budget {} µs",
            cb2.median, COMPOSED_B_PHASE2_BUDGET_US));
    }
    println!();

    // ── Verdict ──
    if violations.is_empty() {
        println!("REGRESSION GATE: PASS");
        std::process::exit(0);
    } else {
        println!("REGRESSION GATE: FAIL ({} violation(s))", violations.len());
        for v in &violations { println!("  ✗ {}", v); }
        std::process::exit(1);
    }
}
