//! Micro-benchmark for the SIMD byte-scan path vs the tree walker.
//!
//! Run with:
//!   cargo run --release --example bench_scan -p jetro-core
//!
//! Not a statistically rigorous harness — just a directional read before
//! committing to further SIMD work.  Each query is warmed once and then
//! timed over `ITERS` repetitions; we report best + median + mean.

use jetro_core::Jetro;
use serde_json::{json, Value};
use std::time::Instant;

const ITERS: usize = 5;

fn synth_doc(n_groups: usize, per_group: usize) -> Value {
    let mut groups = Vec::with_capacity(n_groups);
    for g in 0..n_groups {
        let mut rows = Vec::with_capacity(per_group);
        for i in 0..per_group {
            let ty = match i % 4 {
                0 => "action",
                1 => "idle",
                2 => "noop",
                _ => "error",
            };
            rows.push(json!({
                "id": g * per_group + i,
                "type": ty,
                "device": if i % 2 == 0 { "mobile" } else { "desktop" },
                "value": (i as i64) % 101,
                "payload": {
                    "nested": {
                        "tag": format!("t-{}-{}", g, i),
                        "flags": [i % 2 == 0, i % 3 == 0, i % 5 == 0],
                    }
                }
            }));
        }
        groups.push(json!({ "gid": g, "rows": rows }));
    }
    json!({ "groups": groups, "meta": { "kind": "bench" } })
}

#[allow(dead_code)]
struct Stats {
    best: u128,
    median: u128,
    mean: u128,
}

fn run<F: FnMut() -> Value>(name: &str, mut f: F) -> Stats {
    // Warmup.
    let _ = f();
    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t = Instant::now();
        let _ = f();
        samples.push(t.elapsed().as_micros());
    }
    samples.sort();
    let best = samples[0];
    let median = samples[samples.len() / 2];
    let mean = samples.iter().sum::<u128>() / samples.len() as u128;
    println!(
        "  {:<34} best {:>8}µs  median {:>8}µs  mean {:>8}µs",
        name, best, median, mean
    );
    Stats { best, median, mean }
}

fn main() {
    let n_groups = 200usize;
    let per_group = 500usize;
    let n_rows = n_groups * per_group;
    let doc = synth_doc(n_groups, per_group);
    let bytes = serde_json::to_vec(&doc).unwrap();
    let mb = bytes.len() as f64 / 1_048_576.0;
    println!(
        "doc: {} groups × {} rows = {} rows, {:.2} MB",
        n_groups, per_group, n_rows, mb
    );
    println!("iters: {} (best/median/mean across repetitions)\n", ITERS);

    let j_tree = Jetro::new(doc.clone());
    let j_scan = Jetro::from_bytes(bytes.clone()).unwrap();

    println!("Q1  $..id                          (collect all leaf ids)");
    let a = run("tree_walker", || j_tree.collect("$..id").unwrap());
    let b = run("byte_scan", || j_scan.collect("$..id").unwrap());
    speedup(&a, &b);

    println!("\nQ2  $..id.sum()                    (scan + aggregate)");
    let a = run("tree_walker", || j_tree.collect("$..id.sum()").unwrap());
    let b = run("byte_scan", || j_scan.collect("$..id.sum()").unwrap());
    speedup(&a, &b);

    println!("\nQ3  $..type.filter(@ == \"action\")  (literal-match path, string)");
    let a = run("tree_walker", || {
        j_tree.collect(r#"$..type.filter(@ == "action")"#).unwrap()
    });
    let b = run("byte_scan", || {
        j_scan.collect(r#"$..type.filter(@ == "action")"#).unwrap()
    });
    speedup(&a, &b);

    println!("\nQ4  $..value.filter(@ == 42)       (literal-match path, int)");
    let a = run("tree_walker", || {
        j_tree.collect("$..value.filter(@ == 42)").unwrap()
    });
    let b = run("byte_scan", || {
        j_scan.collect("$..value.filter(@ == 42)").unwrap()
    });
    speedup(&a, &b);

    println!("\nQ5  $..tag                         (deeply nested key)");
    let a = run("tree_walker", || j_tree.collect("$..tag").unwrap());
    let b = run("byte_scan", || j_scan.collect("$..tag").unwrap());
    speedup(&a, &b);

    println!("\nQ6  $..missing_key                 (zero hits — early exit behaviour)");
    let a = run("tree_walker", || j_tree.collect("$..missing_key").unwrap());
    let b = run("byte_scan", || j_scan.collect("$..missing_key").unwrap());
    speedup(&a, &b);

    println!("\nQ7a $..find(@.type == \"action\")   (enclosing-obj SIMD scan)");
    let q = r#"$..find(@.type == "action")"#;
    let a = run("tree_walker", || j_tree.collect(q).unwrap());
    let b = run("byte_scan", || j_scan.collect(q).unwrap());
    speedup(&a, &b);

    println!("\nQ7b $..find(@.id == 100)            (enclosing-obj SIMD scan, int)");
    let q = "$..find(@.id == 100)";
    let a = run("tree_walker", || j_tree.collect(q).unwrap());
    let b = run("byte_scan", || j_scan.collect(q).unwrap());
    speedup(&a, &b);

    println!("\nQ7c $..find(@.type==\"action\", @.device==\"mobile\")  (multi-pred AND scan)");
    let q = r#"$..find(@.type == "action", @.device == "mobile")"#;
    let a = run("tree_walker", || j_tree.collect(q).unwrap());
    let b = run("byte_scan", || j_scan.collect(q).unwrap());
    speedup(&a, &b);

    println!("\nQ7  $..groups.first()..rows.first()..tag   (Route C byte chain)");
    let q = "$..groups.first()..rows.first()..tag";
    let a = run("tree_walker", || j_tree.collect(q).unwrap());
    let b = run("byte_scan", || j_scan.collect(q).unwrap());
    speedup(&a, &b);

    // Baseline: raw serde_json parse of the same doc.
    println!("\nBaseline");
    run("serde_json::from_slice (parse)", || {
        let _: Value = serde_json::from_slice(&bytes).unwrap();
        Value::Null
    });
}

fn speedup(tree: &Stats, scan: &Stats) {
    let ratio = tree.median as f64 / scan.median.max(1) as f64;
    println!("  -> scan/tree median speedup: {:.2}x", ratio);
}
