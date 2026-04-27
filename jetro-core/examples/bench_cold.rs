//! Cold-start bench — measures the single-call latency for parsing
//! a JSON document and running one query, with no warmup.  This is
//! the cost a CLI invocation, MCP server cold call, or test fixture
//! pays each time.
//!
//! Compares two paths:
//!   1. legacy: `serde_json::from_slice` → `Jetro::new(value)` → `collect`
//!   2. simd-direct: `Jetro::from_bytes(bytes)` (routes through simd-json
//!      direct bytes→Val parser via the cold_start_direct_parse plan)
//!
//! Run:
//!   cargo run --release --example bench_cold --features simd-json
//!
//! Per-iteration: each iteration builds a fresh Jetro, runs ONE query,
//! discards.  No reuse — measures pure cold-start cost.

use jetro_core::Jetro;
use serde_json::json;
use std::time::Instant;

fn make_doc() -> Vec<u8> {
    // Same shape as bench_complex: 5000 records with nested fields.
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
    println!(
        "{:<40}  best {:>6} µs  median {:>6} µs  mean {:>6} µs",
        label, min, med, avg
    );
}

fn main() {
    let bytes = make_doc();
    println!(
        "doc size = {} bytes ({:.2} MB)",
        bytes.len(),
        bytes.len() as f64 / 1.0e6
    );
    println!();

    // Each cold iteration: build Jetro fresh, run one query, drop.
    let q_simple = "$.orders.map(total).sum()";
    let q_filter = "$.orders.filter(total > 500).map(id)";

    println!("Cold-start, 1 query ('map(total).sum()')");
    time("legacy: from_slice + new", 30, || {
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let j = Jetro::new(v);
        let _ = j.collect(q_simple).unwrap();
    });
    time("simd-direct: from_bytes", 30, || {
        let j = Jetro::from_bytes(bytes.clone()).unwrap();
        let _ = j.collect(q_simple).unwrap();
    });

    println!();
    println!("Cold-start, 1 query ('filter+map')");
    time("legacy: from_slice + new", 30, || {
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let j = Jetro::new(v);
        let _ = j.collect(q_filter).unwrap();
    });
    time("simd-direct: from_bytes", 30, || {
        let j = Jetro::from_bytes(bytes.clone()).unwrap();
        let _ = j.collect(q_filter).unwrap();
    });

    println!();
    println!("Warm — same Jetro, 100 queries (amortised)");
    let j = Jetro::from_bytes(bytes.clone()).unwrap();
    // Touch root_val + objvec_cache once so subsequent calls hit cache.
    let _ = j.collect(q_simple).unwrap();
    time("warm: map(total).sum()", 100, || {
        let _ = j.collect(q_simple).unwrap();
    });
    time("warm: filter+map", 100, || {
        let _ = j.collect(q_filter).unwrap();
    });
    time("warm: map(total).max()", 100, || {
        let _ = j.collect("$.orders.map(total).max()").unwrap();
    });

    println!();
    println!("Amortised total — parse once, run N queries:");
    for n in [1usize, 5, 10, 100] {
        let t0 = Instant::now();
        let j = Jetro::from_bytes(bytes.clone()).unwrap();
        let _ = j.collect(q_simple).unwrap(); // warm caches
        for _ in 0..n {
            let _ = j.collect(q_simple).unwrap();
            let _ = j.collect(q_filter).unwrap();
            let _ = j.collect("$.orders.map(total).max()").unwrap();
        }
        let total = t0.elapsed().as_micros();
        println!(
            "{} queries  total {:>7} µs  per-query {:>5} µs",
            n * 3,
            total,
            total / (n as u128 * 3)
        );
    }
}
