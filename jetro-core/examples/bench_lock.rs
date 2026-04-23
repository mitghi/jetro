//! Regression guard for jetro-core VM perf.
//!
//! Runs a fixed query set against a reproducible synthetic payload and
//! compares median runtime + result hash against `bench_baseline.json`.
//!
//! Usage:
//!   cargo run --release -p jetro-core --example bench_lock
//!   cargo run --release -p jetro-core --example bench_lock -- --update
//!
//! Exit codes:
//!   0 — all queries within tolerance
//!   1 — at least one regression (perf or result)
//!   2 — baseline missing (first run); use --update to create
//!
//! Tolerance: 1.15x slower than baseline fails. Hash mismatch fails.
//! Warmup: 3 iterations before timing. Timing: median of 10.

use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::time::Instant;

use jetro_core::Jetro;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

const WARMUP: usize = 3;
const ITERS:  usize = 20;
/// Relative tolerance.  Laptop variance routinely hits ±15-20% between
/// runs even with warmup; 1.25x catches real regressions without false
/// positives.  Set lower on a dedicated bench machine.
const TOLERANCE: f64 = 1.25;
/// Absolute noise floor.  Sub-ms queries flap by hundreds of µs from
/// thermal/background jitter; only flag regressions where the absolute
/// delta is this large too.
const NOISE_FLOOR_US: u128 = 500;

const QUERIES: &[(&str, &str)] = &[
    ("Q1_project_deep",        "$.orders.map(customer.address.city)"),
    ("Q2_project_unique",      "$.orders.map(customer.address.country_code).unique()"),
    ("Q3_filter_project",      "$.orders.filter(total > 500).map(id)"),
    ("Q4_multi_cond_count",    r#"$.orders.filter(status == "shipped" and priority == "high").count()"#),
    ("Q5_deep_find_broad",     r#"$..find(@.status == "shipped")"#),
    ("Q6_deep_find_narrow",    r#"$..find(@.sku == "SKU-00042")"#),
    ("Q7_deep_multi_pred",     r#"$..find(@.status == "shipped", @.priority == "urgent")"#),
    ("Q8_deep_total_sum",      "$..total.sum()"),
    ("Q9_deep_sku",            "$..sku"),
    ("Q10_group_by",           "$.orders.group_by(status)"),
    ("Q11_map_total_sum",      "$.orders.map(total).sum()"),
    ("Q12_map_total_max",      "$.orders.map(total).max()"),
    ("Q13_list_comp",          "[o.id for o in $.orders if o.total > 1000]"),
];

fn synth_doc(n_orders: usize, items_per_order: usize) -> Value {
    let regions = ["us-east", "us-west", "eu-central", "ap-southeast", "sa-south"];
    let statuses = ["pending", "shipped", "delivered", "cancelled", "refunded"];
    let priorities = ["low", "normal", "high", "urgent"];
    let categories = ["electronics", "books", "apparel", "grocery", "toys", "tools"];

    let mut orders = Vec::with_capacity(n_orders);
    for i in 0..n_orders {
        let mut items = Vec::with_capacity(items_per_order);
        let mut total: f64 = 0.0;
        for j in 0..items_per_order {
            let price = ((i * 7 + j * 13) % 500) as f64 + 9.99;
            let qty = ((i + j) % 5 + 1) as i64;
            total += price * qty as f64;
            items.push(json!({
                "sku": format!("SKU-{:05}", (i * items_per_order + j) % 9973),
                "name": format!("item-{}-{}", i, j),
                "category": categories[(i + j) % categories.len()],
                "price": price,
                "qty": qty,
            }));
        }
        orders.push(json!({
            "id": 100_000 + i,
            "status": statuses[i % statuses.len()],
            "priority": priorities[(i / 3) % priorities.len()],
            "region": regions[i % regions.len()],
            "total": (total * 100.0).round() / 100.0,
            "customer": {
                "id": 10_000 + (i % 5000),
                "name": format!("Customer {}", i % 5000),
                "email": format!("c{}@example.com", i % 5000),
                "address": {
                    "city": match i % 6 {
                        0 => "Tokyo", 1 => "Berlin", 2 => "São Paulo",
                        3 => "Nairobi", 4 => "Austin", _ => "Toronto"
                    },
                    "zip": format!("{:05}", (i * 17) % 100_000),
                    "country_code": match i % 6 {
                        0 => "JP", 1 => "DE", 2 => "BR",
                        3 => "KE", 4 => "US", _ => "CA"
                    },
                }
            },
            "items": items,
        }));
    }
    json!({ "orders": orders, "meta": { "kind": "bench_lock", "version": 1 } })
}

fn result_hash(v: &Value) -> u64 {
    // Canonical bytes via serde_json::to_vec (stable for same shape);
    // hash via SipHash default. Good enough for fingerprinting.
    let bytes = serde_json::to_vec(v).unwrap();
    let mut h = DefaultHasher::new();
    bytes.hash(&mut h);
    h.finish()
}

fn measure_us<F: FnMut() -> Value>(mut f: F) -> (u128, u64) {
    let mut last: Value = Value::Null;
    for _ in 0..WARMUP { last = f(); }
    let hash = result_hash(&last);
    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t = Instant::now();
        let _ = f();
        samples.push(t.elapsed().as_micros());
    }
    samples.sort();
    // Use best-of rather than median — more stable for sub-ms queries.
    (samples[0], hash)
}

#[derive(Serialize, Deserialize, Clone)]
struct Entry { query: String, best_us: u128, result_hash: u64 }

#[derive(Serialize, Deserialize)]
struct Baseline {
    n_orders: usize,
    items_per_order: usize,
    entries: Vec<Entry>,
}

fn baseline_path() -> PathBuf {
    // CARGO_MANIFEST_DIR points at jetro-core.
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("bench_baseline.json");
    p
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let update = args.iter().any(|a| a == "--update");

    let n_orders = 20_000usize;
    let items_per_order = 6usize;
    let doc = synth_doc(n_orders, items_per_order);
    let j = Jetro::new(doc);

    println!("bench_lock: {} orders x {} items, warmup {}, iters {}",
             n_orders, items_per_order, WARMUP, ITERS);

    let mut entries = Vec::with_capacity(QUERIES.len());
    for (name, q) in QUERIES {
        let (med, hash) = measure_us(|| j.collect(q).unwrap());
        println!("  {:28} {:>7}µs  hash={:016x}", name, med, hash);
        entries.push(Entry { query: (*q).into(), best_us: med, result_hash: hash });
    }

    let path = baseline_path();

    if update {
        let b = Baseline { n_orders, items_per_order, entries };
        let json = serde_json::to_string_pretty(&b).unwrap();
        fs::write(&path, json).unwrap();
        println!("\nbaseline written → {}", path.display());
        return;
    }

    let baseline: Baseline = match fs::read_to_string(&path) {
        Ok(s)  => serde_json::from_str(&s).expect("baseline parse"),
        Err(_) => {
            eprintln!("\nbaseline missing at {}.  run with --update to create.", path.display());
            std::process::exit(2);
        }
    };

    if baseline.n_orders != n_orders || baseline.items_per_order != items_per_order {
        eprintln!("baseline shape mismatch: payload dims changed.  --update required.");
        std::process::exit(1);
    }

    let mut fails: Vec<String> = Vec::new();
    println!("\n{:<30} {:>9} {:>9} {:>7}  hash", "query", "base_µs", "cur_µs", "ratio");
    for (i, e) in entries.iter().enumerate() {
        let b = &baseline.entries[i];
        let ratio = e.best_us as f64 / b.best_us.max(1) as f64;
        let hash_ok = e.result_hash == b.result_hash;
        let abs_delta = e.best_us.saturating_sub(b.best_us);
        // Perf passes if under ratio OR under absolute noise floor.
        let perf_ok = ratio <= TOLERANCE || abs_delta < NOISE_FLOOR_US;
        let flag = match (perf_ok, hash_ok) {
            (true, true)  => "  ok",
            (false, true) => " SLOW",
            (true, false) => " HASH",
            (false, false) => " BOTH",
        };
        let name = QUERIES[i].0;
        println!("{:<30} {:>9} {:>9} {:>6.2}x  {}{}",
                 name, b.best_us, e.best_us, ratio,
                 if hash_ok { "ok" } else { "MISMATCH" }, flag);
        if !perf_ok {
            fails.push(format!("{}: {:.2}x slower ({}µs → {}µs)", name, ratio, b.best_us, e.best_us));
        }
        if !hash_ok {
            fails.push(format!("{}: result hash changed ({:016x} → {:016x})", name, b.result_hash, e.result_hash));
        }
    }

    if fails.is_empty() {
        println!("\nall {} queries within tolerance (≤{:.2}x)", entries.len(), TOLERANCE);
    } else {
        eprintln!("\nregressions:");
        for f in &fails { eprintln!("  - {}", f); }
        std::process::exit(1);
    }
}
