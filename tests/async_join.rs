#![cfg(feature = "async")]

use jetro::prelude::*;
use std::time::Duration;

#[tokio::test]
async fn async_join_end_to_end() {
    let db = Database::memory().unwrap();
    let j = db
        .join("events")
        .id("order_id")
        .kinds(["order", "payment", "shipment"])
        .open()
        .unwrap()
        .into_async();

    j.emit("order", &json!({"order_id": "A1", "item": "Dune"}))
        .await
        .unwrap();
    j.emit("payment", &json!({"order_id": "A1", "amount": 12.99}))
        .await
        .unwrap();
    j.emit("shipment", &json!({"order_id": "A1", "tracking": "TRK-1"}))
        .await
        .unwrap();

    let bill = j
        .run(
            "A1",
            r#"{item: $.order.item, paid: $.payment.amount, track: $.shipment.tracking}"#,
        )
        .await
        .unwrap();
    assert_eq!(
        bill,
        json!({"item": "Dune", "paid": 12.99, "track": "TRK-1"})
    );
}

#[tokio::test]
async fn async_join_wait_for_times_out() {
    let db = Database::memory().unwrap();
    let j = db
        .join("events")
        .id("order_id")
        .kinds(["order", "payment"])
        .open()
        .unwrap()
        .into_async();

    j.emit("order", &json!({"order_id": "B1"})).await.unwrap();

    // payment never arrives
    let r = j.wait_for("B1", Duration::from_millis(150)).await.unwrap();
    assert!(r.is_none());
}

#[tokio::test]
async fn async_join_arrived_and_peek() {
    let db = Database::memory().unwrap();
    let j = db
        .join("events")
        .id("order_id")
        .kinds(["order", "payment", "shipment"])
        .open()
        .unwrap()
        .into_async();

    j.emit("order", &json!({"order_id": "C1"})).await.unwrap();
    j.emit("payment", &json!({"order_id": "C1"})).await.unwrap();

    let got = j.arrived("C1").await;
    assert_eq!(got.len(), 2);
    assert!(got.contains(&"order".to_string()));
    assert!(got.contains(&"payment".to_string()));

    let peek = j.peek("C1").await.unwrap();
    assert!(peek.is_none()); // link not yet complete
}
