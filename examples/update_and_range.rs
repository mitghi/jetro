//! Demonstrates update (re-applies expressions) and range scanning the raw B-Tree.
//!
//!   cargo run --example update_and_range

use jetro::db::{BTree, Database};
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tmp = tempfile::tempdir()?;
    let db = Database::open(tmp.path())?;

    // ── update example ───────────────────────────────────────────────────────
    let exprs = db.expr_bucket("exprs")?;
    exprs.put("score_sum", ">/scores/#sum")?;

    let docs = db.json_bucket("game", &["score_sum"], &exprs)?;

    docs.insert("player:alice", &json!({"scores": [10, 20, 30]}))?;
    let before = docs.get_result("player:alice", "score_sum")?.unwrap();
    println!("Before update: sum = {}", before[0]);  // 60

    docs.update("player:alice", &json!({"scores": [10, 20, 30, 40, 50]}))?;
    let after = docs.get_result("player:alice", "score_sum")?.unwrap();
    println!("After  update: sum = {}", after[0]);   // 150

    // ── range scan example ───────────────────────────────────────────────────
    let path = tmp.path().join("range_demo.btree");
    let tree = BTree::open(&path)?;

    for i in 0u32..20 {
        let key = format!("entry:{i:03}");
        let val = format!("value-{i}");
        tree.insert(key.as_bytes(), val.as_bytes())?;
    }

    println!("\nRange [entry:005, entry:010):");
    let results = tree.range(b"entry:005", b"entry:010")?;
    for (k, v) in &results {
        println!("  {} => {}", std::str::from_utf8(k)?, std::str::from_utf8(v)?);
    }
    assert_eq!(results.len(), 5);

    println!("\nAll entries from entry:015 onward:");
    let tail = tree.range_from(b"entry:015")?;
    for (k, v) in &tail {
        println!("  {} => {}", std::str::from_utf8(k)?, std::str::from_utf8(v)?);
    }
    assert_eq!(tail.len(), 5); // 015..019

    Ok(())
}
