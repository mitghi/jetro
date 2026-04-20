//! 8 reader threads performing concurrent lookups while a writer inserts.
//!
//!   cargo run --example concurrent_reads

use std::sync::Arc;
use std::thread;
use std::time::Instant;

use jetro::db::Database;
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tmp = tempfile::tempdir()?;
    let db = Database::open(tmp.path())?;

    let exprs = Arc::new(db.expr_bucket("exprs")?);
    exprs.put("total",  "$.values.sum()")?;
    exprs.put("max",    "$.values.max()")?;
    exprs.put("count",  "$.values.len()")?;

    let docs = Arc::new(db.json_bucket("metrics", &["total", "max", "count"], &exprs)?);

    // Pre-populate 100 documents
    for i in 0u32..100 {
        docs.insert(
            &format!("metric:{i:04}"),
            &json!({ "values": [i, i+1, i+2, i+3] }),
        )?;
    }

    println!("Inserted 100 documents. Starting concurrent reads...");

    let start = Instant::now();

    // 8 reader threads, each reads all 100 docs
    let handles: Vec<_> = (0..8)
        .map(|tid| {
            let docs = Arc::clone(&docs);
            thread::spawn(move || {
                let mut hits = 0usize;
                for i in 0u32..100 {
                    let key = format!("metric:{i:04}");
                    let total = docs.get_result(&key, "total").unwrap().unwrap();
                    let expected = 4 * i + 6; // i+(i+1)+(i+2)+(i+3)
                    let got: u64 = serde_json::from_value(total).unwrap();
                    assert_eq!(got, expected as u64, "tid={tid} i={i}");
                    hits += 1;
                }
                hits
            })
        })
        .collect();

    let total_reads: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
    let elapsed = start.elapsed();

    println!(
        "{total_reads} reads completed in {:.1}ms  ({:.0} reads/sec)",
        elapsed.as_secs_f64() * 1000.0,
        total_reads as f64 / elapsed.as_secs_f64()
    );

    Ok(())
}
