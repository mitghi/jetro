//! Profile where time is going in the scan fast path.
//!
//! Isolates: raw serde parse, bare scan, scan+extract, full Jetro::collect.

use std::time::Instant;
use jetro_core::{Jetro, scan};
use serde_json::{json, Value};

fn synth(n_groups: usize, per_group: usize) -> Value {
    let mut groups = Vec::with_capacity(n_groups);
    for g in 0..n_groups {
        let mut rows = Vec::with_capacity(per_group);
        for i in 0..per_group {
            rows.push(json!({
                "id": g * per_group + i,
                "type": if i % 2 == 0 { "action" } else { "idle" },
                "value": (i as i64) % 101,
                "payload": {"nested": {"tag": format!("t-{}-{}", g, i)}}
            }));
        }
        groups.push(json!({ "gid": g, "rows": rows }));
    }
    json!({ "groups": groups })
}

fn time<T, F: FnMut() -> T>(name: &str, mut f: F) -> T {
    let t = Instant::now();
    let out = f();
    println!("  {:<44} {:>10}µs", name, t.elapsed().as_micros());
    out
}

fn main() {
    let doc = synth(200, 500);
    let bytes = serde_json::to_vec(&doc).unwrap();
    println!("doc: {:.2} MB\n", bytes.len() as f64 / 1_048_576.0);

    println!("A. parse / conversion costs (each is O(doc)):");
    let _ = time("serde_json::from_slice", || {
        let _: Value = serde_json::from_slice(&bytes).unwrap();
    });

    println!("\nB. scan primitives over raw bytes:");
    let _ = time("scan::find_key_positions(id)", || {
        scan::find_key_positions(&bytes, "id").len()
    });
    let _ = time("scan::find_key_positions(missing)", || {
        scan::find_key_positions(&bytes, "does_not_exist").len()
    });
    let _ = time("scan::extract_values(id)", || {
        scan::extract_values(&bytes, "id").len()
    });
    let _ = time("scan::extract_values(tag)", || {
        scan::extract_values(&bytes, "tag").len()
    });

    println!("\nC. full Jetro::collect (includes Val::from + hash_val_structure):");
    let j_scan = Jetro::from_bytes(bytes.clone()).unwrap();
    let j_tree = Jetro::new(doc.clone());
    // warmup
    let _ = j_scan.collect("$..id").unwrap();
    let _ = j_tree.collect("$..id").unwrap();
    let _ = time("Jetro(scan)::collect('$..id')",      || j_scan.collect("$..id").unwrap());
    let _ = time("Jetro(tree)::collect('$..id')",      || j_tree.collect("$..id").unwrap());
    let _ = time("Jetro(scan)::collect('$..missing')", || j_scan.collect("$..missing").unwrap());
    let _ = time("Jetro(tree)::collect('$..missing')", || j_tree.collect("$..missing").unwrap());
}
