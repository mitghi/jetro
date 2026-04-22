//! Isolate time spent in each sub-scan of a chained-descendant query.

use std::time::Instant;
use jetro_core::{Jetro, VM, scan};
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

fn time<T, F: FnMut() -> T>(name: &str, iters: usize, mut f: F) -> T {
    let _ = f();
    let mut best = u128::MAX;
    let mut last = None;
    for _ in 0..iters {
        let t = Instant::now();
        let out = f();
        let e = t.elapsed().as_micros();
        if e < best { best = e; }
        last = Some(out);
    }
    println!("  {:<56} best {:>10}µs", name, best);
    last.unwrap()
}

fn main() {
    let doc = synth(200, 500);
    let bytes = serde_json::to_vec(&doc).unwrap();
    println!("doc: {:.2} MB\n", bytes.len() as f64 / 1_048_576.0);

    println!("Layer-by-layer byte-scan cost:");
    let groups_spans = time("scan groups (doc)", 3, || scan::find_key_value_spans(&bytes, "groups"));
    println!("  groups spans: {}", groups_spans.len());

    let g0 = groups_spans[0];
    let groups_sub = &bytes[g0.start..g0.end];
    println!("  groups sub len: {:.2} MB", groups_sub.len() as f64 / 1_048_576.0);

    let rows_spans = time("scan rows (within groups span)", 3, || scan::find_key_value_spans(groups_sub, "rows"));
    println!("  rows spans: {}", rows_spans.len());

    let r0 = rows_spans[0];
    let rows_sub = &groups_sub[r0.start..r0.end];
    println!("  first rows sub len: {} bytes", rows_sub.len());

    let tag_spans = time("scan tag (within first rows span)", 3, || scan::find_key_value_spans(rows_sub, "tag"));
    println!("  tag spans: {}", tag_spans.len());

    println!("\nManual byte-chain simulation:");
    let _ = time("manual chain end-to-end", 5, || {
        let gs = scan::find_key_value_spans(&bytes, "groups");
        let g = gs[0];
        let sub_g = &bytes[g.start..g.end];
        let rs = scan::find_key_value_spans(sub_g, "rows");
        let r = rs[0];
        let sub_r = &sub_g[r.start..r.end];
        let ts = scan::find_key_value_spans(sub_r, "tag");
        let mut out: Vec<serde_json::Value> = Vec::with_capacity(ts.len());
        for s in &ts {
            if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&sub_r[s.start..s.end]) {
                out.push(v);
            }
        }
        out
    });

    println!("\nFull query via Jetro (warm):");
    let j_scan = Jetro::from_bytes(bytes.clone()).unwrap();
    let j_tree = Jetro::new(doc.clone());
    let q = "$..groups.first()..rows.first()..tag";
    let _ = j_scan.collect(q).unwrap();
    let _ = j_tree.collect(q).unwrap();
    let _ = time("Jetro(scan) full", 5, || j_scan.collect(q).unwrap());
    let _ = time("Jetro(tree) full", 5, || j_tree.collect(q).unwrap());

    // Dump the compiled opcode stream.
    let mut vm = VM::new();
    let prog = vm.get_or_compile(q).unwrap();
    println!("\nCompiled program ({} ops):", prog.ops.len());
    for (i, op) in prog.ops.iter().enumerate() {
        println!("  [{}] {:?}", i, op);
    }
}
