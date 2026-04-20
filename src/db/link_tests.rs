#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    use serde_json::json;
    use tempfile::TempDir;

    use crate::db::{Database, LinkBucket};

    fn setup() -> (TempDir, Database) {
        let dir = tempfile::tempdir().unwrap();
        let db  = Database::open(dir.path()).unwrap();
        (dir, db)
    }

    fn three_way_bucket(db: &Database) -> Arc<LinkBucket> {
        db.link_bucket("events", &[
            ("order",    "order_id"),
            ("payment",  "order_id"),
            ("shipment", "order_id"),
        ]).unwrap()
    }

    // ── Basic completion ──────────────────────────────────────────────────────

    #[test]
    fn complete_after_all_three_kinds() {
        let (_dir, db) = setup();
        let lb = three_way_bucket(&db);

        lb.insert("order",    &json!({"order_id": "A1", "item": "Widget"})).unwrap();
        lb.insert("payment",  &json!({"order_id": "A1", "amount": 49.99})).unwrap();
        lb.insert("shipment", &json!({"order_id": "A1", "tracking": "TRK1"})).unwrap();

        let linked = lb.get(&json!("A1")).unwrap();
        assert_eq!(linked.docs["order"]["item"],        json!("Widget"));
        assert_eq!(linked.docs["payment"]["amount"],    json!(49.99));
        assert_eq!(linked.docs["shipment"]["tracking"], json!("TRK1"));
    }

    #[test]
    fn try_get_returns_none_before_complete() {
        let (_dir, db) = setup();
        let lb = three_way_bucket(&db);

        lb.insert("order", &json!({"order_id": "B1", "item": "Gadget"})).unwrap();
        assert!(lb.try_get(&json!("B1")).unwrap().is_none());

        lb.insert("payment",  &json!({"order_id": "B1", "amount": 9.0})).unwrap();
        assert!(lb.try_get(&json!("B1")).unwrap().is_none());

        lb.insert("shipment", &json!({"order_id": "B1", "tracking": "X"})).unwrap();
        assert!(lb.try_get(&json!("B1")).unwrap().is_some());
    }

    #[test]
    fn get_timeout_returns_none_when_incomplete() {
        let (_dir, db) = setup();
        let lb = three_way_bucket(&db);

        lb.insert("order", &json!({"order_id": "C1", "item": "X"})).unwrap();

        // Only one kind arrived — should timeout immediately.
        let result = lb.get_timeout(&json!("C1"), Some(Duration::from_millis(10))).unwrap();
        assert!(result.is_none());
    }

    // ── Insertion order independence ──────────────────────────────────────────

    #[test]
    fn any_insertion_order_completes_link() {
        let (_dir, db) = setup();
        let lb = three_way_bucket(&db);

        // Reverse order
        lb.insert("shipment", &json!({"order_id": "D1", "tracking": "TRK9"})).unwrap();
        lb.insert("payment",  &json!({"order_id": "D1", "amount": 1.0})).unwrap();
        lb.insert("order",    &json!({"order_id": "D1", "item": "Thing"})).unwrap();

        let linked = lb.try_get(&json!("D1")).unwrap().expect("should be complete");
        assert_eq!(linked.docs["order"]["item"], json!("Thing"));
    }

    // ── Multiple independent ids ──────────────────────────────────────────────

    #[test]
    fn multiple_ids_tracked_independently() {
        let (_dir, db) = setup();
        let lb = three_way_bucket(&db);

        for i in 1u32..=5 {
            let id = json!(format!("ORD-{i}"));
            lb.insert("order",    &json!({"order_id": format!("ORD-{i}"), "item": i})).unwrap();
            lb.insert("payment",  &json!({"order_id": format!("ORD-{i}"), "amount": i})).unwrap();
            lb.insert("shipment", &json!({"order_id": format!("ORD-{i}"), "tracking": format!("T{i}")})).unwrap();
            let linked = lb.try_get(&id).unwrap().expect("complete");
            assert_eq!(linked.docs["order"]["item"], json!(i));
        }
    }

    // ── Blocking get unblocks when last kind arrives ───────────────────────────

    #[test]
    fn blocking_get_unblocks_on_completion() {
        let (_dir, db) = setup();
        let lb = Arc::new(three_way_bucket(&db));

        // Spawn reader before any inserts
        let lb_r = Arc::clone(&lb);
        let reader = thread::spawn(move || {
            lb_r.get(&json!("E1")).unwrap()
        });

        // Insert with delays to ensure reader is waiting
        thread::sleep(Duration::from_millis(5));
        lb.insert("order",    &json!({"order_id": "E1", "item": "Async"})).unwrap();
        thread::sleep(Duration::from_millis(5));
        lb.insert("payment",  &json!({"order_id": "E1", "amount": 7.0})).unwrap();
        thread::sleep(Duration::from_millis(5));
        lb.insert("shipment", &json!({"order_id": "E1", "tracking": "Z"})).unwrap();

        let linked = reader.join().unwrap();
        assert_eq!(linked.docs["order"]["item"], json!("Async"));
        assert_eq!(linked.docs["payment"]["amount"], json!(7.0));
    }

    // ── Multiple concurrent readers ───────────────────────────────────────────

    #[test]
    fn multiple_concurrent_readers_all_unblock() {
        let (_dir, db) = setup();
        let lb = Arc::new(three_way_bucket(&db));

        // Spawn 8 readers all waiting on the same id
        let readers: Vec<_> = (0..8).map(|_| {
            let lb_r = Arc::clone(&lb);
            thread::spawn(move || lb_r.get(&json!("F1")).unwrap())
        }).collect();

        thread::sleep(Duration::from_millis(10));
        lb.insert("order",    &json!({"order_id": "F1", "item": "Multi"})).unwrap();
        lb.insert("payment",  &json!({"order_id": "F1", "amount": 3.0})).unwrap();
        lb.insert("shipment", &json!({"order_id": "F1", "tracking": "M"})).unwrap();

        for r in readers {
            let linked = r.join().unwrap();
            assert_eq!(linked.docs["order"]["item"], json!("Multi"));
        }
    }

    // ── Query with jetro expression ───────────────────────────────────────────

    #[test]
    fn query_runs_jetro_on_linked_doc() {
        let (_dir, db) = setup();
        let lb = three_way_bucket(&db);

        lb.insert("order",    &json!({"order_id": "G1", "item": "Chair", "qty": 2})).unwrap();
        lb.insert("payment",  &json!({"order_id": "G1", "amount": 120.0})).unwrap();
        lb.insert("shipment", &json!({"order_id": "G1", "tracking": "TRK-G"})).unwrap();

        // Cross-kind object construction
        let result = lb.query(
            &json!("G1"),
            r#"{item: $.order.item, paid: $.payment.amount, track: $.shipment.tracking}"#,
        ).unwrap();

        let obj = result.as_object().unwrap();
        assert_eq!(obj["item"],  json!("Chair"));
        assert_eq!(obj["paid"],  json!(120.0));
        assert_eq!(obj["track"], json!("TRK-G"));
    }

    #[test]
    fn query_timeout_returns_none_when_incomplete() {
        let (_dir, db) = setup();
        let lb = three_way_bucket(&db);
        lb.insert("order", &json!({"order_id": "H1", "item": "X"})).unwrap();

        let r = lb.query_timeout(&json!("H1"), "$.order.item", Some(Duration::from_millis(5))).unwrap();
        assert!(r.is_none());
    }

    // ── arrived_kinds ─────────────────────────────────────────────────────────

    #[test]
    fn arrived_kinds_tracks_progress() {
        let (_dir, db) = setup();
        let lb = three_way_bucket(&db);

        assert!(lb.arrived_kinds(&json!("I1")).is_empty());

        lb.insert("order", &json!({"order_id": "I1", "item": "Y"})).unwrap();
        let arrived = lb.arrived_kinds(&json!("I1"));
        assert_eq!(arrived, vec!["order".to_string()]);

        lb.insert("payment", &json!({"order_id": "I1", "amount": 5.0})).unwrap();
        let arrived = lb.arrived_kinds(&json!("I1"));
        assert!(arrived.contains(&"order".to_string()));
        assert!(arrived.contains(&"payment".to_string()));
        assert!(!arrived.contains(&"shipment".to_string()));
    }

    // ── Numeric ids ───────────────────────────────────────────────────────────

    #[test]
    fn numeric_id_works() {
        let (_dir, db) = setup();
        let lb = three_way_bucket(&db);

        lb.insert("order",    &json!({"order_id": 42, "item": "Z"})).unwrap();
        lb.insert("payment",  &json!({"order_id": 42, "amount": 9.0})).unwrap();
        lb.insert("shipment", &json!({"order_id": 42, "tracking": "N42"})).unwrap();

        let linked = lb.try_get(&json!(42)).unwrap().expect("complete");
        assert_eq!(linked.docs["order"]["item"], json!("Z"));
    }

    // ── Two-kind bucket ───────────────────────────────────────────────────────

    #[test]
    fn two_kind_bucket() {
        let (_dir, db) = setup();
        let lb = db.link_bucket("pair", &[
            ("left",  "id"),
            ("right", "id"),
        ]).unwrap();

        lb.insert("left",  &json!({"id": "P1", "val": 1})).unwrap();
        assert!(lb.try_get(&json!("P1")).unwrap().is_none());
        lb.insert("right", &json!({"id": "P1", "val": 2})).unwrap();

        let linked = lb.try_get(&json!("P1")).unwrap().expect("complete");
        assert_eq!(linked.docs["left"]["val"],  json!(1));
        assert_eq!(linked.docs["right"]["val"], json!(2));
    }

    // ── delete_kind / update_kind / get_partial ───────────────────────────────

    #[test]
    fn delete_kind_breaks_complete_link() {
        let (_dir, db) = setup();
        let lb = three_way_bucket(&db);

        lb.insert("order",    &json!({"order_id": "K1", "item": "A"})).unwrap();
        lb.insert("payment",  &json!({"order_id": "K1", "amount": 1.0})).unwrap();
        lb.insert("shipment", &json!({"order_id": "K1", "tracking": "T"})).unwrap();
        assert!(lb.try_get(&json!("K1")).unwrap().is_some());

        lb.delete_kind("payment", &json!("K1")).unwrap();
        assert!(lb.try_get(&json!("K1")).unwrap().is_none());

        let arrived = lb.arrived_kinds(&json!("K1"));
        assert!(arrived.contains(&"order".to_string()));
        assert!(!arrived.contains(&"payment".to_string()));
        assert!(arrived.contains(&"shipment".to_string()));
    }

    #[test]
    fn delete_kind_returns_false_when_not_present() {
        let (_dir, db) = setup();
        let lb = three_way_bucket(&db);

        lb.insert("order", &json!({"order_id": "K2", "item": "B"})).unwrap();
        let existed = lb.delete_kind("payment", &json!("K2")).unwrap();
        assert!(!existed);
    }

    #[test]
    fn blocking_get_unblocks_after_delete_and_reinsert() {
        use std::sync::Arc;
        let (_dir, db) = setup();
        let lb = Arc::new(three_way_bucket(&db));

        lb.insert("order",    &json!({"order_id": "K3", "item": "C"})).unwrap();
        lb.insert("payment",  &json!({"order_id": "K3", "amount": 5.0})).unwrap();
        lb.insert("shipment", &json!({"order_id": "K3", "tracking": "T3"})).unwrap();
        lb.delete_kind("payment", &json!("K3")).unwrap();

        let lb_r = Arc::clone(&lb);
        let reader = thread::spawn(move || {
            lb_r.get_timeout(&json!("K3"), Some(Duration::from_millis(200))).unwrap()
        });

        thread::sleep(Duration::from_millis(10));
        lb.insert("payment", &json!({"order_id": "K3", "amount": 99.0})).unwrap();

        let result = reader.join().unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().docs["payment"]["amount"], json!(99.0));
    }

    #[test]
    fn insert_updates_doc_when_kind_already_present() {
        let (_dir, db) = setup();
        let lb = three_way_bucket(&db);

        lb.insert("order", &json!({"order_id": "N1", "item": "Old"})).unwrap();
        // Re-insert same kind+id with updated payload.
        lb.insert("order", &json!({"order_id": "N1", "item": "New"})).unwrap();

        // Still only one kind arrived.
        assert!(lb.try_get(&json!("N1")).unwrap().is_none());
        let arrived = lb.arrived_kinds(&json!("N1"));
        assert_eq!(arrived.len(), 1);

        lb.insert("payment",  &json!({"order_id": "N1", "amount": 3.0})).unwrap();
        lb.insert("shipment", &json!({"order_id": "N1", "tracking": "T"})).unwrap();

        let linked = lb.try_get(&json!("N1")).unwrap().expect("complete");
        assert_eq!(linked.docs["order"]["item"], json!("New"));
    }

    #[test]
    fn insert_update_on_complete_link_keeps_completion() {
        let (_dir, db) = setup();
        let lb = three_way_bucket(&db);

        lb.insert("order",    &json!({"order_id": "N2", "item": "V1"})).unwrap();
        lb.insert("payment",  &json!({"order_id": "N2", "amount": 1.0})).unwrap();
        lb.insert("shipment", &json!({"order_id": "N2", "tracking": "T"})).unwrap();
        assert!(lb.try_get(&json!("N2")).unwrap().is_some());

        // Update one kind — link must remain complete with new doc.
        lb.insert("order", &json!({"order_id": "N2", "item": "V2"})).unwrap();

        let linked = lb.try_get(&json!("N2")).unwrap().expect("still complete");
        assert_eq!(linked.docs["order"]["item"], json!("V2"));
        assert_eq!(linked.docs["payment"]["amount"], json!(1.0));
    }

    #[test]
    fn update_kind_replaces_doc_and_keeps_link_complete() {
        let (_dir, db) = setup();
        let lb = three_way_bucket(&db);

        lb.insert("order",    &json!({"order_id": "L1", "item": "Old"})).unwrap();
        lb.insert("payment",  &json!({"order_id": "L1", "amount": 1.0})).unwrap();
        lb.insert("shipment", &json!({"order_id": "L1", "tracking": "T"})).unwrap();

        lb.update_kind("order", &json!({"order_id": "L1", "item": "New"})).unwrap();

        let linked = lb.try_get(&json!("L1")).unwrap().expect("still complete");
        assert_eq!(linked.docs["order"]["item"], json!("New"));
        assert_eq!(linked.docs["payment"]["amount"], json!(1.0));
    }

    #[test]
    fn get_partial_returns_arrived_kinds_only() {
        let (_dir, db) = setup();
        let lb = three_way_bucket(&db);

        lb.insert("order",   &json!({"order_id": "M1", "item": "X"})).unwrap();
        lb.insert("payment", &json!({"order_id": "M1", "amount": 7.0})).unwrap();

        let partial = lb.get_partial(&json!("M1")).unwrap();
        assert_eq!(partial.len(), 2);
        assert_eq!(partial["order"]["item"], json!("X"));
        assert_eq!(partial["payment"]["amount"], json!(7.0));
        assert!(!partial.contains_key("shipment"));
    }

    #[test]
    fn get_partial_empty_when_no_kinds_arrived() {
        let (_dir, db) = setup();
        let lb = three_way_bucket(&db);
        let partial = lb.get_partial(&json!("M2")).unwrap();
        assert!(partial.is_empty());
    }

    // ── Remove ────────────────────────────────────────────────────────────────

    #[test]
    fn remove_clears_state() {
        let (_dir, db) = setup();
        let lb = three_way_bucket(&db);

        lb.insert("order",    &json!({"order_id": "J1", "item": "A"})).unwrap();
        lb.insert("payment",  &json!({"order_id": "J1", "amount": 1.0})).unwrap();
        lb.insert("shipment", &json!({"order_id": "J1", "tracking": "T"})).unwrap();
        assert!(lb.try_get(&json!("J1")).unwrap().is_some());

        lb.remove(&json!("J1")).unwrap();
        assert!(lb.try_get(&json!("J1")).unwrap().is_none());

        // Re-insert works after remove
        lb.insert("order",    &json!({"order_id": "J1", "item": "B"})).unwrap();
        lb.insert("payment",  &json!({"order_id": "J1", "amount": 2.0})).unwrap();
        lb.insert("shipment", &json!({"order_id": "J1", "tracking": "T2"})).unwrap();
        let linked = lb.try_get(&json!("J1")).unwrap().expect("complete after re-insert");
        assert_eq!(linked.docs["order"]["item"], json!("B"));
    }
}
