use jetro_core::Jetro;
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = br#"
{
  "orders": [
    {
      "id": "ord_1001",
      "status": "paid",
      "total": 184.50,
      "customer": {"name": "Ada", "tier": "gold"},
      "items": [{"sku": "A1", "qty": 1, "price": 184.50}],
      "events": [
        {"kind": "placed",   "at": "2025-04-12T10:00:00Z"},
        {"kind": "shipped",  "at": "2025-04-13T08:30:00Z"},
        {"kind": "delivered","at": "2025-04-14T17:12:00Z"}
      ]
    },
    {
      "id": "ord_1002",
      "status": "refunded",
      "total": 42.00,
      "customer": {"name": "Grace", "tier": "silver"},
      "items": [{"sku": "B2", "qty": 2, "price": 21.00}],
      "events": [
        {"kind": "placed", "at": "2025-04-15T09:00:00Z"},
        {"kind": "refund", "reason": "damaged"}
      ]
    },
    {
      "id": "ord_1003",
      "status": "paid",
      "total": 312.20,
      "customer": {"name": "Alan", "tier": "platinum"},
      "items": [
        {"sku": "C3", "qty": 2, "price": 99.00},
        {"sku": "C4", "qty": 1, "price": 114.20}
      ],
      "events": [
        {"kind": "placed",  "at": "2025-04-20T11:00:00Z"},
        {"kind": "shipped", "at": "2025-04-21T07:45:00Z"}
      ]
    },
    {
      "id": "ord_1004",
      "status": "paid",
      "total": 58.00,
      "customer": {"name": "Lin", "tier": "silver"},
      "items": [{"sku": "D5", "qty": 1, "price": 58.00}],
      "events": [{"kind": "placed", "at": "2025-04-22T12:00:00Z"}]
    }
  ]
}
"#;

    let jetro = Jetro::from_bytes(data.to_vec())?;

    let report = jetro.collect(
        r#"
{
  "top_paid_premium": $.orders
    .filter(status == "paid")
    .filter(total >= 100)
    .filter(customer.tier == "gold" or customer.tier == "platinum")
    .sort_by(-total)
    .take(2)
    .map({
      order_id: id,
      who: customer.name,
      tier: customer.tier,
      amount: total,
      label: f"order {@.id}: {customer.name} ({customer.tier}) USD {@.total}",
      line_total: items.map(qty * price).sum(),
      last_event: match events.last() with {
          {kind: "delivered", at: t}    -> {state: "ok",     at: t},
          {kind: "shipped",   at: t}    -> {state: "moving", at: t},
          {kind: "refund", reason: r}   -> {state: "refund", reason: r},
          _                             -> {state: "unknown"}
      }
    }),

  "paid_total": $.orders
    .filter(status == "paid")
    .map(total)
    .sum()
}
"#,
    )?;

    println!("{}", serde_json::to_string_pretty(&report)?);

    let expected = json!({
      "top_paid_premium": [
        {
          "order_id": "ord_1003",
          "who": "Alan",
          "tier": "platinum",
          "amount": 312.20,
          "label": "order ord_1003: Alan (platinum) USD 312.2",
          "line_total": 312.20,
          "last_event": {"state": "moving", "at": "2025-04-21T07:45:00Z"}
        },
        {
          "order_id": "ord_1001",
          "who": "Ada",
          "tier": "gold",
          "amount": 184.50,
          "label": "order ord_1001: Ada (gold) USD 184.5",
          "line_total": 184.50,
          "last_event": {"state": "ok", "at": "2025-04-14T17:12:00Z"}
        }
      ],
      "paid_total": 554.70
    });

    if report == expected {
        println!("\nMATCH expected.");
    } else {
        println!("\nDIVERGED from expected.");
        println!("expected: {}", serde_json::to_string_pretty(&expected)?);
    }

    Ok(())
}
