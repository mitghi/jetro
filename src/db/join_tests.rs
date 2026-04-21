#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    use serde_json::json;
    use tempfile::TempDir;

    use crate::db::{Database, Join};

    fn setup() -> (TempDir, Database) {
        let dir = tempfile::tempdir().unwrap();
        let db  = Database::open(dir.path()).unwrap();
        (dir, db)
    }

    fn three_way(db: &Database) -> Arc<Join> {
        db.join("events")
            .id("order_id")
            .kinds(["order", "payment", "shipment"])
            .open()
            .unwrap()
    }

    // ── Basic completion ──────────────────────────────────────────────────────

    #[test]
    fn complete_after_all_three_kinds() {
        let (_dir, db) = setup();
        let j = three_way(&db);

        j.emit("order",    &json!({"order_id": "A1", "item": "Widget"})).unwrap();
        j.emit("payment",  &json!({"order_id": "A1", "amount": 49.99})).unwrap();
        j.emit("shipment", &json!({"order_id": "A1", "tracking": "TRK1"})).unwrap();

        let joined = j.wait("A1").unwrap();
        assert_eq!(joined.docs["order"]["item"],        json!("Widget"));
        assert_eq!(joined.docs["payment"]["amount"],    json!(49.99));
        assert_eq!(joined.docs["shipment"]["tracking"], json!("TRK1"));
    }

    #[test]
    fn peek_returns_none_before_complete() {
        let (_dir, db) = setup();
        let j = three_way(&db);

        j.emit("order", &json!({"order_id": "B1", "item": "Gadget"})).unwrap();
        assert!(j.peek("B1").unwrap().is_none());

        j.emit("payment",  &json!({"order_id": "B1", "amount": 9.0})).unwrap();
        assert!(j.peek("B1").unwrap().is_none());

        j.emit("shipment", &json!({"order_id": "B1", "tracking": "X"})).unwrap();
        assert!(j.peek("B1").unwrap().is_some());
    }

    #[test]
    fn wait_for_returns_none_on_timeout() {
        let (_dir, db) = setup();
        let j = three_way(&db);

        j.emit("order", &json!({"order_id": "C1", "item": "X"})).unwrap();

        let result = j.wait_for("C1", Duration::from_millis(10)).unwrap();
        assert!(result.is_none());
    }

    // ── Insertion order independence ──────────────────────────────────────────

    #[test]
    fn any_emit_order_completes_join() {
        let (_dir, db) = setup();
        let j = three_way(&db);

        j.emit("shipment", &json!({"order_id": "D1", "tracking": "TRK9"})).unwrap();
        j.emit("payment",  &json!({"order_id": "D1", "amount": 1.0})).unwrap();
        j.emit("order",    &json!({"order_id": "D1", "item": "Thing"})).unwrap();

        let joined = j.peek("D1").unwrap().expect("should be complete");
        assert_eq!(joined.docs["order"]["item"], json!("Thing"));
    }

    // ── Multiple independent ids ──────────────────────────────────────────────

    #[test]
    fn multiple_ids_tracked_independently() {
        let (_dir, db) = setup();
        let j = three_way(&db);

        for i in 1u32..=5 {
            let id = format!("ORD-{i}");
            j.emit("order",    &json!({"order_id": &id, "item": i})).unwrap();
            j.emit("payment",  &json!({"order_id": &id, "amount": i})).unwrap();
            j.emit("shipment", &json!({"order_id": &id, "tracking": format!("T{i}")})).unwrap();
            let joined = j.peek(&id).unwrap().expect("complete");
            assert_eq!(joined.docs["order"]["item"], json!(i));
        }
    }

    // ── Blocking wait unblocks when last kind arrives ─────────────────────────

    #[test]
    fn wait_unblocks_on_completion() {
        let (_dir, db) = setup();
        let j = three_way(&db);

        let r = Arc::clone(&j);
        let reader = thread::spawn(move || r.wait("E1").unwrap());

        thread::sleep(Duration::from_millis(5));
        j.emit("order",    &json!({"order_id": "E1", "item": "Async"})).unwrap();
        thread::sleep(Duration::from_millis(5));
        j.emit("payment",  &json!({"order_id": "E1", "amount": 7.0})).unwrap();
        thread::sleep(Duration::from_millis(5));
        j.emit("shipment", &json!({"order_id": "E1", "tracking": "Z"})).unwrap();

        let joined = reader.join().unwrap();
        assert_eq!(joined.docs["order"]["item"], json!("Async"));
        assert_eq!(joined.docs["payment"]["amount"], json!(7.0));
    }

    // ── Multiple concurrent readers ───────────────────────────────────────────

    #[test]
    fn multiple_concurrent_readers_all_unblock() {
        let (_dir, db) = setup();
        let j = three_way(&db);

        let readers: Vec<_> = (0..8).map(|_| {
            let r = Arc::clone(&j);
            thread::spawn(move || r.wait("F1").unwrap())
        }).collect();

        thread::sleep(Duration::from_millis(10));
        j.emit("order",    &json!({"order_id": "F1", "item": "Multi"})).unwrap();
        j.emit("payment",  &json!({"order_id": "F1", "amount": 3.0})).unwrap();
        j.emit("shipment", &json!({"order_id": "F1", "tracking": "M"})).unwrap();

        for r in readers {
            let joined = r.join().unwrap();
            assert_eq!(joined.docs["order"]["item"], json!("Multi"));
        }
    }

    // ── Query with jetro expression ───────────────────────────────────────────

    #[test]
    fn on_run_evaluates_expression() {
        let (_dir, db) = setup();
        let j = three_way(&db);

        j.emit("order",    &json!({"order_id": "G1", "item": "Chair", "qty": 2})).unwrap();
        j.emit("payment",  &json!({"order_id": "G1", "amount": 120.0})).unwrap();
        j.emit("shipment", &json!({"order_id": "G1", "tracking": "TRK-G"})).unwrap();

        let result = j.on("G1").run(
            r#"{item: $.order.item, paid: $.payment.amount, track: $.shipment.tracking}"#,
        ).unwrap();

        let obj = result.as_object().unwrap();
        assert_eq!(obj["item"],  json!("Chair"));
        assert_eq!(obj["paid"],  json!(120.0));
        assert_eq!(obj["track"], json!("TRK-G"));
    }

    #[test]
    fn on_get_typed() {
        let (_dir, db) = setup();
        let j = three_way(&db);

        j.emit("order",    &json!({"order_id": "G2", "item": "Desk", "qty": 1})).unwrap();
        j.emit("payment",  &json!({"order_id": "G2", "amount": 250.0})).unwrap();
        j.emit("shipment", &json!({"order_id": "G2", "tracking": "TRK-G2"})).unwrap();

        let amount: f64 = j.on("G2").get("$.payment.amount").unwrap();
        assert_eq!(amount, 250.0);
    }

    #[test]
    fn on_run_opt_times_out() {
        let (_dir, db) = setup();
        let j = three_way(&db);
        j.emit("order", &json!({"order_id": "H1", "item": "X"})).unwrap();

        let r = j.on("H1")
            .timeout(Duration::from_millis(5))
            .run_opt("$.order.item")
            .unwrap();
        assert!(r.is_none());
    }

    // ── arrived ───────────────────────────────────────────────────────────────

    #[test]
    fn arrived_tracks_progress() {
        let (_dir, db) = setup();
        let j = three_way(&db);

        assert!(j.arrived("I1").is_empty());

        j.emit("order", &json!({"order_id": "I1", "item": "Y"})).unwrap();
        assert_eq!(j.arrived("I1"), vec!["order".to_string()]);

        j.emit("payment", &json!({"order_id": "I1", "amount": 5.0})).unwrap();
        let a = j.arrived("I1");
        assert!(a.contains(&"order".to_string()));
        assert!(a.contains(&"payment".to_string()));
        assert!(!a.contains(&"shipment".to_string()));
    }

    // ── Numeric ids ───────────────────────────────────────────────────────────

    #[test]
    fn numeric_id_works() {
        let (_dir, db) = setup();
        let j = three_way(&db);

        j.emit("order",    &json!({"order_id": 42, "item": "Z"})).unwrap();
        j.emit("payment",  &json!({"order_id": 42, "amount": 9.0})).unwrap();
        j.emit("shipment", &json!({"order_id": 42, "tracking": "N42"})).unwrap();

        let joined = j.peek(42i64).unwrap().expect("complete");
        assert_eq!(joined.docs["order"]["item"], json!("Z"));
    }

    // ── Two-kind join ─────────────────────────────────────────────────────────

    #[test]
    fn two_kind_join() {
        let (_dir, db) = setup();
        let j = db.join("pair")
            .id("id")
            .kinds(["left", "right"])
            .open()
            .unwrap();

        j.emit("left",  &json!({"id": "P1", "val": 1})).unwrap();
        assert!(j.peek("P1").unwrap().is_none());
        j.emit("right", &json!({"id": "P1", "val": 2})).unwrap();

        let joined = j.peek("P1").unwrap().expect("complete");
        assert_eq!(joined.docs["left"]["val"],  json!(1));
        assert_eq!(joined.docs["right"]["val"], json!(2));
    }

    // ── Per-kind id field override ────────────────────────────────────────────

    #[test]
    fn per_kind_id_override() {
        let (_dir, db) = setup();
        let j = db.join("mixed")
            .kind("order",   "order_id")
            .kind("payment", "invoice_id")
            .open()
            .unwrap();

        j.emit("order",   &json!({"order_id": "Q1", "item": "A"})).unwrap();
        j.emit("payment", &json!({"invoice_id": "Q1", "amount": 1.0})).unwrap();

        let joined = j.peek("Q1").unwrap().expect("complete");
        assert_eq!(joined.docs["order"]["item"], json!("A"));
        assert_eq!(joined.docs["payment"]["amount"], json!(1.0));
    }

    // ── unemit / updates / partial ────────────────────────────────────────────

    #[test]
    fn unemit_breaks_complete_join() {
        let (_dir, db) = setup();
        let j = three_way(&db);

        j.emit("order",    &json!({"order_id": "K1", "item": "A"})).unwrap();
        j.emit("payment",  &json!({"order_id": "K1", "amount": 1.0})).unwrap();
        j.emit("shipment", &json!({"order_id": "K1", "tracking": "T"})).unwrap();
        assert!(j.peek("K1").unwrap().is_some());

        j.unemit("payment", "K1").unwrap();
        assert!(j.peek("K1").unwrap().is_none());

        let a = j.arrived("K1");
        assert!(a.contains(&"order".to_string()));
        assert!(!a.contains(&"payment".to_string()));
        assert!(a.contains(&"shipment".to_string()));
    }

    #[test]
    fn unemit_false_when_not_present() {
        let (_dir, db) = setup();
        let j = three_way(&db);

        j.emit("order", &json!({"order_id": "K2", "item": "B"})).unwrap();
        let existed = j.unemit("payment", "K2").unwrap();
        assert!(!existed);
    }

    #[test]
    fn wait_unblocks_after_unemit_and_reemit() {
        let (_dir, db) = setup();
        let j = three_way(&db);

        j.emit("order",    &json!({"order_id": "K3", "item": "C"})).unwrap();
        j.emit("payment",  &json!({"order_id": "K3", "amount": 5.0})).unwrap();
        j.emit("shipment", &json!({"order_id": "K3", "tracking": "T3"})).unwrap();
        j.unemit("payment", "K3").unwrap();

        let r = Arc::clone(&j);
        let reader = thread::spawn(move || {
            r.wait_for("K3", Duration::from_millis(200)).unwrap()
        });

        thread::sleep(Duration::from_millis(10));
        j.emit("payment", &json!({"order_id": "K3", "amount": 99.0})).unwrap();

        let result = reader.join().unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().docs["payment"]["amount"], json!(99.0));
    }

    #[test]
    fn emit_updates_doc_when_kind_already_present() {
        let (_dir, db) = setup();
        let j = three_way(&db);

        j.emit("order", &json!({"order_id": "N1", "item": "Old"})).unwrap();
        j.emit("order", &json!({"order_id": "N1", "item": "New"})).unwrap();

        assert!(j.peek("N1").unwrap().is_none());
        assert_eq!(j.arrived("N1").len(), 1);

        j.emit("payment",  &json!({"order_id": "N1", "amount": 3.0})).unwrap();
        j.emit("shipment", &json!({"order_id": "N1", "tracking": "T"})).unwrap();

        let joined = j.peek("N1").unwrap().expect("complete");
        assert_eq!(joined.docs["order"]["item"], json!("New"));
    }

    #[test]
    fn emit_update_on_complete_keeps_completion() {
        let (_dir, db) = setup();
        let j = three_way(&db);

        j.emit("order",    &json!({"order_id": "N2", "item": "V1"})).unwrap();
        j.emit("payment",  &json!({"order_id": "N2", "amount": 1.0})).unwrap();
        j.emit("shipment", &json!({"order_id": "N2", "tracking": "T"})).unwrap();
        assert!(j.peek("N2").unwrap().is_some());

        j.emit("order", &json!({"order_id": "N2", "item": "V2"})).unwrap();

        let joined = j.peek("N2").unwrap().expect("still complete");
        assert_eq!(joined.docs["order"]["item"], json!("V2"));
        assert_eq!(joined.docs["payment"]["amount"], json!(1.0));
    }

    #[test]
    fn partial_returns_arrived_kinds_only() {
        let (_dir, db) = setup();
        let j = three_way(&db);

        j.emit("order",   &json!({"order_id": "M1", "item": "X"})).unwrap();
        j.emit("payment", &json!({"order_id": "M1", "amount": 7.0})).unwrap();

        let p = j.partial("M1").unwrap();
        assert_eq!(p.len(), 2);
        assert_eq!(p["order"]["item"], json!("X"));
        assert_eq!(p["payment"]["amount"], json!(7.0));
        assert!(!p.contains_key("shipment"));
    }

    #[test]
    fn partial_empty_when_no_kinds_arrived() {
        let (_dir, db) = setup();
        let j = three_way(&db);
        let p = j.partial("M2").unwrap();
        assert!(p.is_empty());
    }

    // ── remove ────────────────────────────────────────────────────────────────

    #[test]
    fn remove_clears_state() {
        let (_dir, db) = setup();
        let j = three_way(&db);

        j.emit("order",    &json!({"order_id": "J1", "item": "A"})).unwrap();
        j.emit("payment",  &json!({"order_id": "J1", "amount": 1.0})).unwrap();
        j.emit("shipment", &json!({"order_id": "J1", "tracking": "T"})).unwrap();
        assert!(j.peek("J1").unwrap().is_some());

        j.remove("J1").unwrap();
        assert!(j.peek("J1").unwrap().is_none());

        j.emit("order",    &json!({"order_id": "J1", "item": "B"})).unwrap();
        j.emit("payment",  &json!({"order_id": "J1", "amount": 2.0})).unwrap();
        j.emit("shipment", &json!({"order_id": "J1", "tracking": "T2"})).unwrap();
        let joined = j.peek("J1").unwrap().expect("complete after re-emit");
        assert_eq!(joined.docs["order"]["item"], json!("B"));
    }
}
