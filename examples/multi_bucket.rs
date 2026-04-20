//! Multiple JSON buckets sharing one expression bucket.
//! Orders bucket and users bucket each bind different expression subsets.
//!
//!   cargo run --example multi_bucket

use jetro::db::Database;
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tmp = tempfile::tempdir()?;
    let db = Database::open(tmp.path())?;

    let exprs = db.expr_bucket("shared")?;

    // user expressions
    exprs.put("user_names",   "$.users.map(name)")?;
    exprs.put("adult_users",  "$.users.filter(age >= 18).map(name)")?;

    // order expressions
    exprs.put("order_total",  "$.items.sum(price)")?;
    exprs.put("item_names",   "$.items.map(name)")?;

    // two buckets, different expression subsets
    let users  = db.json_bucket("users",  &["user_names", "adult_users"], &exprs)?;
    let orders = db.json_bucket("orders", &["order_total", "item_names"],  &exprs)?;

    // insert user documents
    users.insert("team_alpha", &json!({
        "users": [
            {"name": "Alice", "age": 32},
            {"name": "Bob",   "age": 17},
            {"name": "Carol", "age": 25}
        ]
    }))?;

    // insert order documents
    orders.insert("order_001", &json!({
        "items": [
            {"name": "Widget A", "price": 4.99},
            {"name": "Widget B", "price": 12.50},
            {"name": "Gadget",   "price": 29.00}
        ]
    }))?;

    // --- results -------------------------------------------------------------
    let names  = users.get_result("team_alpha", "user_names")?.unwrap();
    println!("All users:    {}", names);

    let adults = users.get_result("team_alpha", "adult_users")?.unwrap();
    println!("Adults only:  {}", adults);

    let total  = orders.get_result("order_001", "order_total")?.unwrap();
    println!("Order total:  {}", total);

    let items  = orders.get_result("order_001", "item_names")?.unwrap();
    println!("Item names:   {}", items);

    // expressions from orders bucket are absent from users bucket and vice versa
    assert!(users.get_result("team_alpha", "order_total")?.is_none());
    println!("\nCross-bucket isolation verified.");

    Ok(())
}
