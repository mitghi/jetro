#[cfg(test)]
mod tests {
    use serde_json::json;
    use tempfile::TempDir;
    use crate::db::{Database, GraphNode};

    fn setup() -> (TempDir, crate::db::Database) {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(dir.path()).unwrap();
        (dir, db)
    }

    fn make_graph(db: &Database) -> std::sync::Arc<crate::db::GraphBucket> {
        let exprs = db.expr_bucket("exprs").unwrap();
        let g = db.graph_bucket("g", &exprs).unwrap();
        g.add_node("orders").unwrap();
        g.add_node("customers").unwrap();
        g.add_index("orders", "customer_id").unwrap();
        g.add_index("customers", "id").unwrap();

        g.insert("customers", "c:1", &json!({"id": 1, "name": "Alice"})).unwrap();
        g.insert("customers", "c:2", &json!({"id": 2, "name": "Bob"})).unwrap();
        g.insert("orders", "o:1", &json!({"id": 10, "customer_id": 1, "total": 99.0})).unwrap();
        g.insert("orders", "o:2", &json!({"id": 11, "customer_id": 2, "total": 45.0})).unwrap();
        g.insert("orders", "o:3", &json!({"id": 12, "customer_id": 1, "total": 30.0})).unwrap();
        g
    }

    // ── Index lookups ─────────────────────────────────────────────────────────

    #[test]
    fn by_field_returns_matching_docs() {
        let (_dir, db) = setup();
        let g = make_graph(&db);

        let results = g.query(&[
            GraphNode::ByField { node: "orders", field: "customer_id", value: &json!(1) },
        ], ">/orders/#len").unwrap();

        let count: i64 = serde_json::from_value(results[0].clone()).unwrap();
        assert_eq!(count, 2, "Alice has 2 orders");
    }

    #[test]
    fn by_key_lookup() {
        let (_dir, db) = setup();
        let g = make_graph(&db);

        let results = g.query(&[
            GraphNode::ByKey { node: "customers", doc_key: "c:1" },
        ], ">/customers/[0]/name").unwrap();

        assert_eq!(results[0], json!("Alice"));
    }

    #[test]
    fn index_updated_on_insert_update() {
        let (_dir, db) = setup();
        let g = make_graph(&db);

        // Re-assign order o:2 from customer 2 to customer 1
        g.insert("orders", "o:2", &json!({"id": 11, "customer_id": 1, "total": 45.0})).unwrap();

        let alice_orders = g.query(&[
            GraphNode::ByField { node: "orders", field: "customer_id", value: &json!(1) },
        ], ">/orders/#len").unwrap();
        let count: i64 = serde_json::from_value(alice_orders[0].clone()).unwrap();
        assert_eq!(count, 3);

        let bob_orders = g.query(&[
            GraphNode::ByField { node: "orders", field: "customer_id", value: &json!(2) },
        ], ">/orders/#len").unwrap();
        let count: i64 = serde_json::from_value(bob_orders[0].clone()).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn index_cleaned_on_delete() {
        let (_dir, db) = setup();
        let g = make_graph(&db);
        g.delete("orders", "o:1").unwrap();

        let results = g.query(&[
            GraphNode::ByField { node: "orders", field: "customer_id", value: &json!(1) },
        ], ">/orders/#len").unwrap();
        let count: i64 = serde_json::from_value(results[0].clone()).unwrap();
        assert_eq!(count, 1); // o:3 remains
    }

    // ── Cross-node graph queries ──────────────────────────────────────────────

    #[test]
    fn all_nodes_inline_and_by_field() {
        let (_dir, db) = setup();
        let g = make_graph(&db);

        // Simulate stream: order arrives, join with customer
        let incoming = json!({"id": 13, "customer_id": 2, "total": 20.0});
        let cust_id = incoming["customer_id"].clone();

        let results = g.query(&[
            GraphNode::Inline { node: "orders",    value: json!([incoming]) },
            GraphNode::ByField { node: "customers", field: "id", value: &cust_id },
        ], r#">{"order_total": >/orders/[0]/total, "customer": >/customers/[0]/name}"#).unwrap();

        let obj = results[0].as_object().unwrap();
        assert_eq!(obj["customer"], json!("Bob"));
        assert_eq!(obj["order_total"], json!(20.0));
    }

    #[test]
    fn process_stream_convenience() {
        let (_dir, db) = setup();
        let g = make_graph(&db);

        let order = json!({"id": 14, "customer_id": 1, "total": 55.5});
        let cust_id = order["customer_id"].clone();

        let results = g.process_stream(
            "orders",
            json!([order]),
            &[("customers", "id", &cust_id)],
            r#">{"name": >/customers/[0]/name, "total": >/orders/[0]/total}"#,
        ).unwrap();

        let obj = results[0].as_object().unwrap();
        assert_eq!(obj["name"], json!("Alice"));
        assert_eq!(obj["total"], json!(55.5));
    }

    // ── Hot cache ─────────────────────────────────────────────────────────────

    #[test]
    fn hot_cache_matches_disk() {
        let (_dir, db) = setup();
        let g = make_graph(&db);
        g.preload_hot("customers").unwrap();

        let hot_results = g.query(&[
            GraphNode::ByField { node: "orders",    field: "customer_id", value: &json!(1) },
            GraphNode::Hot     { node: "customers" },
        ], ">/orders/#len").unwrap();

        let disk_results = g.query(&[
            GraphNode::ByField { node: "orders",    field: "customer_id", value: &json!(1) },
            GraphNode::All     { node: "customers" },
        ], ">/orders/#len").unwrap();

        assert_eq!(hot_results, disk_results);
    }

    #[test]
    fn hot_cache_reflects_updates() {
        let (_dir, db) = setup();
        let g = make_graph(&db);
        g.preload_hot("customers").unwrap();

        g.insert("customers", "c:3", &json!({"id": 3, "name": "Carol"})).unwrap();

        let hot = g.query(&[
            GraphNode::Hot { node: "customers" },
        ], ">/customers/#len").unwrap();
        let count: i64 = serde_json::from_value(hot[0].clone()).unwrap();
        assert_eq!(count, 3);
    }

    // ── All-node scan ─────────────────────────────────────────────────────────

    #[test]
    fn all_node_scan_and_aggregate() {
        let (_dir, db) = setup();
        let g = make_graph(&db);

        let results = g.query(&[
            GraphNode::All { node: "orders" },
        ], ">/orders/..total/#sum").unwrap();

        let sum: f64 = serde_json::from_value(results[0].clone()).unwrap();
        assert!((sum - 174.0).abs() < 0.001);
    }

    // ── Concurrent reads ──────────────────────────────────────────────────────

    #[test]
    fn concurrent_graph_reads() {
        use std::sync::Arc;
        use std::thread;

        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(dir.path()).unwrap();
        let exprs = db.expr_bucket("exprs").unwrap();
        let g = Arc::new(db.graph_bucket("g", &exprs).unwrap());

        g.add_node("items").unwrap();
        g.add_index("items", "tag").unwrap();
        for i in 0u32..50 {
            g.insert("items", &format!("item:{i}"), &json!({"id": i, "tag": i % 5, "v": i * 2})).unwrap();
        }
        g.preload_hot("items").unwrap();

        let handles: Vec<_> = (0..8).map(|_| {
            let g = Arc::clone(&g);
            thread::spawn(move || {
                for tag in 0u32..5 {
                    let by_field = g.query(&[
                        GraphNode::ByField { node: "items", field: "tag", value: &json!(tag) },
                    ], ">/items/#len").unwrap();
                    let count: i64 = serde_json::from_value(by_field[0].clone()).unwrap();
                    assert_eq!(count, 10, "tag={tag}");

                    let hot = g.query(&[
                        GraphNode::Hot { node: "items" },
                    ], ">/items/#len").unwrap();
                    let total: i64 = serde_json::from_value(hot[0].clone()).unwrap();
                    assert_eq!(total, 50);
                }
            })
        }).collect();
        for h in handles { h.join().unwrap(); }
    }
}
