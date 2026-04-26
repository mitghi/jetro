//! Microbench — `Val::from_json_simd` (owned, Arc + IndexMap) vs
//! `borrowed::from_json_simd_arena` (borrowed, bumpalo arena) on the
//! same 1.7 MB / 5000-record bench document.
//!
//! Measures pure parse + Val-build cost.  No queries.  Confirms the
//! allocator-savings premise of the borrowed Val<'a> migration before
//! wiring it into the runtime.
//!
//! Run:
//!   cargo run --release --example bench_arena_build --features simd-json

use std::time::Instant;
use serde_json::json;

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

fn time<F: FnMut()>(label: &str, n: usize, mut f: F) {
    let mut samples: Vec<u128> = Vec::with_capacity(n);
    for _ in 0..n {
        let t0 = Instant::now();
        f();
        samples.push(t0.elapsed().as_micros());
    }
    samples.sort();
    let med = samples[n / 2];
    let min = samples[0];
    let avg: u128 = samples.iter().sum::<u128>() / n as u128;
    println!("{:<46}  best {:>6} µs  median {:>6} µs  mean {:>6} µs",
        label, min, med, avg);
}

fn main() {
    let bytes = make_doc();
    println!("doc size = {} bytes ({:.2} MB)\n",
        bytes.len(), bytes.len() as f64 / 1.0e6);

    time("owned Val::from_json_simd", 30, || {
        let mut buf = bytes.clone();
        let v = jetro_core::eval::Val::from_json_simd(&mut buf).unwrap();
        std::hint::black_box(v);
    });

    time("borrowed from_json_simd_arena (fresh arena)", 30, || {
        let mut buf = bytes.clone();
        let arena = jetro_core::eval::borrowed::Arena::new();
        let v = jetro_core::eval::borrowed::from_json_simd_arena(&arena, &mut buf).unwrap();
        std::hint::black_box(v);
        std::hint::black_box(arena);
    });

    time("borrowed from_json_simd_arena (presized 4 MB arena)", 30, || {
        let mut buf = bytes.clone();
        let arena = jetro_core::eval::borrowed::Arena::with_capacity(4 * 1024 * 1024);
        let v = jetro_core::eval::borrowed::from_json_simd_arena(&arena, &mut buf).unwrap();
        std::hint::black_box(v);
        std::hint::black_box(arena);
    });

    // Pure simd-json parse cost as floor (no Val build).
    time("simd-json to_tape only (parser floor)", 30, || {
        let mut buf = bytes.clone();
        let tape = simd_json::to_tape(&mut buf).unwrap();
        std::hint::black_box(tape);
    });
}
