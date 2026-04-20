//! Graph-aware storage for stream processing.
//!
//! Pattern:
//!   - "customers" node: small reference table → preloaded into hot cache
//!   - "orders"    node: stream events → secondary index on customer_id
//!   - Each arriving order is joined with customer data via a single expression
//!
//!   cargo run --example graph_stream

use std::sync::Arc;
use std::thread;
use std::time::Instant;

use jetro::db::{Database, GraphNode};
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tmp = tempfile::tempdir()?;
    let db = Database::open(tmp.path())?;
    let exprs = db.expr_bucket("exprs")?;

    // Store a named graph expression
    exprs.put(
        "order_summary",
        r#">{"customer": >/customers/[0]/name, "tier": >/customers/[0]/tier, "total": >/orders/[0]/total}"#,
    )?;

    // ── Setup graph bucket ────────────────────────────────────────────────────
    let graph = Arc::new(db.graph_bucket("analytics", &exprs)?);
    graph.add_node("orders")?;
    graph.add_node("customers")?;
    graph.add_index("orders",    "customer_id")?;
    graph.add_index("customers", "id")?;

    // Reference data: customers
    graph.insert("customers", "c:1",  &json!({"id": 1,  "name": "Alice",   "tier": "gold"}))?;
    graph.insert("customers", "c:2",  &json!({"id": 2,  "name": "Bob",     "tier": "silver"}))?;
    graph.insert("customers", "c:3",  &json!({"id": 3,  "name": "Carol",   "tier": "gold"}))?;
    graph.insert("customers", "c:4",  &json!({"id": 4,  "name": "Dave",    "tier": "bronze"}))?;
    graph.insert("customers", "c:5",  &json!({"id": 5,  "name": "Eve",     "tier": "silver"}))?;

    // Preload customers into hot cache (zero disk I/O on queries)
    graph.preload_hot("customers")?;
    println!("Customers preloaded into hot cache.");

    // ── Simulate stream of orders ─────────────────────────────────────────────
    let graph_ref = Arc::clone(&graph);
    let order_count = 100u32;

    println!("Processing {order_count} stream events...");
    let t0 = Instant::now();

    for i in 0..order_count {
        let cust_id = (i % 5) + 1;
        let order = json!({
            "id": i,
            "customer_id": cust_id,
            "total": (i as f64) * 1.5 + 10.0,
            "items": i % 4 + 1
        });
        let key = format!("o:{i}");
        graph_ref.insert("orders", &key, &order)?;
    }

    let insert_ms = t0.elapsed().as_millis();
    println!("Inserted {order_count} orders in {insert_ms}ms.");

    // ── Query patterns ────────────────────────────────────────────────────────

    // 1. Stream-processing join: event + hot reference lookup
    let event = json!([{"id": 999, "customer_id": 3, "total": 250.0, "items": 5}]);
    let cust_id = event[0]["customer_id"].clone();

    let summary = graph.query_named(&[
        GraphNode::Inline  { node: "orders",    value: event },
        GraphNode::ByField { node: "customers", field: "id", value: &cust_id },
    ], "order_summary")?;
    println!("\nStream event summary: {}", summary[0]);

    // 2. Index lookup: all orders for customer 1
    let alice_orders = graph.query(&[
        GraphNode::ByField { node: "orders",    field: "customer_id", value: &json!(1) },
        GraphNode::Hot     { node: "customers" },
    ], ">/orders/#len")?;
    println!("Alice's orders in DB: {}", alice_orders[0]);

    // 3. Aggregate over all orders with customer join
    let totals = graph.query(&[
        GraphNode::All { node: "orders" },
        GraphNode::Hot { node: "customers" },
    ], ">/orders/..total/#sum")?;
    println!("Total revenue (all orders): {:.2}", totals[0]);

    // ── Concurrent stream processing ─────────────────────────────────────────
    println!("\nConcurrent stream processing (8 threads)...");
    let t1 = Instant::now();
    let processed = Arc::new(std::sync::atomic::AtomicU32::new(0));

    let handles: Vec<_> = (0u32..8).map(|tid| {
        let g = Arc::clone(&graph);
        let counter = Arc::clone(&processed);
        thread::spawn(move || {
            for i in 0u32..50 {
                let cust_id = ((tid * 50 + i) % 5) + 1;
                let order = json!([{"id": 10000 + tid*50+i, "customer_id": cust_id, "total": 42.0}]);
                let id_val = order[0]["customer_id"].clone();

                let res = g.process_stream(
                    "orders", order,
                    &[("customers", "id", &id_val)],
                    r#">{"name": >/customers/[0]/name, "total": >/orders/[0]/total}"#,
                ).unwrap();
                assert!(res[0]["name"].is_string());
                counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        })
    }).collect();
    for h in handles { h.join().unwrap(); }

    let concurrent_ms = t1.elapsed().as_millis();
    let total_ops = processed.load(std::sync::atomic::Ordering::Relaxed);
    println!(
        "{total_ops} graph queries in {concurrent_ms}ms  ({:.0} ops/sec)",
        total_ops as f64 / t1.elapsed().as_secs_f64()
    );

    Ok(())
}
