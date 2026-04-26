//! Microbench — `Jetro::collect_val` (owned) vs `collect_val_borrow`
//! (Phase 4 borrowed API, arena-backed) on the same 1.7 MB / 5000-record
//! bench document.  Measures end-to-end query cost.
//!
//! Run:
//!   cargo run --release --example bench_collect_borrow --features simd-json

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
    println!("{:<54}  best {:>6} µs  median {:>6} µs  mean {:>6} µs",
        label, min, med, avg);
}

const QUERIES: &[(&str, &str)] = &[
    ("filter+map[city] +unique",
        "$.orders.map(customer.address.city).unique()"),
    ("filter+map(field) collect",
        "$.orders.filter(status == 'shipped').map(id).collect()"),
    ("skip+take+map collect",
        "$.orders.skip(100).take(50).map(id)"),
    ("map deep proj collect",
        "$.orders.map({id, name: customer.name, city: customer.address.city})"),
    ("filter count",
        "$.orders.filter(status == 'shipped').count()"),
    ("filter+map sum",
        "$.orders.filter(status == 'shipped').map(total).sum()"),
    ("filter+map avg",
        "$.orders.filter(priority == 'high').map(total).avg()"),
    ("filter first",
        "$.orders.filter(status == 'shipped').first()"),
    ("filter last",
        "$.orders.filter(priority == 'high').first()"),
    ("sort_by+take+map (heap-top-K)",
        "$.orders.sort_by(total).take(10).map({id, total, name: customer.name})"),
];

fn main() {
    let bytes = make_doc();
    println!("doc size = {} bytes ({:.2} MB)\n",
        bytes.len(), bytes.len() as f64 / 1.0e6);

    for (label, q) in QUERIES {
        println!("── {} ──", label);

        time("  cold owned     collect_val()", 30, || {
            let j = jetro_core::Jetro::from_simd(bytes.clone()).unwrap();
            let v = j.collect_val(*q).unwrap();
            std::hint::black_box(v);
            std::hint::black_box(j);
        });

        time("  cold borrowed  collect_val_borrow()", 30, || {
            let j = jetro_core::Jetro::from_simd(bytes.clone()).unwrap();
            let v = j.collect_val_borrow(*q).unwrap();
            std::hint::black_box(v);
            std::hint::black_box(j);
        });

        println!();
    }
}
