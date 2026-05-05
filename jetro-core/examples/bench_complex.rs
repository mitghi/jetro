

use jetro_core::Jetro;
use serde_json::{json, Value};
use std::time::Instant;

const ITERS: usize = 10;


fn synth_doc(n_orders: usize, items_per_order: usize) -> Value {
    let regions = [
        "us-east",
        "us-west",
        "eu-central",
        "ap-southeast",
        "sa-south",
    ];
    let statuses = ["pending", "shipped", "delivered", "cancelled", "refunded"];
    let priorities = ["low", "normal", "high", "urgent"];
    let categories = [
        "electronics",
        "books",
        "apparel",
        "grocery",
        "toys",
        "tools",
    ];

    let mut orders = Vec::with_capacity(n_orders);
    for i in 0..n_orders {
        let mut items = Vec::with_capacity(items_per_order);
        let mut total: f64 = 0.0;
        for j in 0..items_per_order {
            let price = ((i * 7 + j * 13) % 500) as f64 + 9.99;
            let qty = ((i + j) % 5 + 1) as i64;
            total += price * qty as f64;
            items.push(json!({
                "sku": format!("SKU-{:05}", (i * items_per_order + j) % 9973),
                "name": format!("item-{}-{}", i, j),
                "category": categories[(i + j) % categories.len()],
                "price": price,
                "qty": qty,
                "tags": [
                    format!("t{}", j % 7),
                    format!("c-{}", (i * j) % 13),
                ],
            }));
        }

        orders.push(json!({
            "id": 100_000 + i,
            "status": statuses[i % statuses.len()],
            "priority": priorities[(i / 3) % priorities.len()],
            "region": regions[i % regions.len()],
            "total": (total * 100.0).round() / 100.0,
            "customer": {
                "id": 10_000 + (i % 5000),
                "name": format!("Customer {}", i % 5000),
                "email": format!("c{}@example.com", i % 5000),
                "vip": i % 97 == 0,
                "address": {
                    "street": format!("{} Main St", (i * 31) % 9999),
                    "city": match i % 6 {
                        0 => "Tokyo", 1 => "Berlin", 2 => "São Paulo",
                        3 => "Nairobi", 4 => "Austin", _ => "Toronto"
                    },
                    "zip": format!("{:05}", (i * 17) % 100_000),
                    "country_code": match i % 6 {
                        0 => "JP", 1 => "DE", 2 => "BR",
                        3 => "KE", 4 => "US", _ => "CA"
                    },
                }
            },
            "shipping": {
                "carrier": match i % 4 { 0 => "ups", 1 => "dhl", 2 => "fedex", _ => "usps" },
                "cost": ((i % 40) as f64) + 4.95,
                "expedited": i % 11 == 0,
            },
            "items": items,
            "metadata": {
                "source": if i % 3 == 0 { "web" } else if i % 3 == 1 { "mobile" } else { "partner" },
                "coupon_applied": i % 8 == 0,
                "score": ((i * 37) % 100) as f64 / 10.0,
            }
        }));
    }

    let customers: Vec<Value> = (0..500)
        .map(|c| {
            json!({
                "id": 10_000 + c,
                "tier": match c % 4 { 0 => "bronze", 1 => "silver", 2 => "gold", _ => "platinum" },
                "lifetime_value": (c * 123 % 50_000) as f64,
            })
        })
        .collect();

    json!({
        "orders": orders,
        "customers": customers,
        "meta": { "kind": "bench_complex", "version": 2, "generated_at": "2026-04-23" }
    })
}

#[derive(Clone, Copy)]
#[allow(dead_code)]
struct Stats {
    best: u128,
    median: u128,
    mean: u128,
}

fn run<F: FnMut() -> Value>(label: &str, mut f: F) -> Stats {
    
    let _ = f();
    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t = Instant::now();
        let _ = f();
        samples.push(t.elapsed().as_micros());
    }
    samples.sort();
    let best = samples[0];
    let median = samples[samples.len() / 2];
    let mean = samples.iter().sum::<u128>() / samples.len() as u128;
    println!(
        "  {:<38} best {:>8}µs  median {:>8}µs  mean {:>8}µs",
        label, best, median, mean
    );
    Stats { best, median, mean }
}

fn compare(tree: Stats, bytes: Stats) {
    let ratio = tree.median as f64 / bytes.median.max(1) as f64;
    println!("  -> bytes/tree median: {:.2}x", ratio);
}

fn main() {
    let n_orders = 20_000usize;
    let items_per_order = 6usize;
    let doc = synth_doc(n_orders, items_per_order);
    let bytes = serde_json::to_vec(&doc).unwrap();
    let mb = bytes.len() as f64 / 1_048_576.0;
    println!(
        "payload: {} orders × {} items = {} items, {:.2} MB",
        n_orders,
        items_per_order,
        n_orders * items_per_order,
        mb
    );
    println!(
        "iters: {} (first run warms; cache-hot numbers follow)\n",
        ITERS
    );

    let j_tree = Jetro::from_bytes(serde_json::to_vec(&doc).unwrap()).unwrap();
    let j_bytes = Jetro::from_bytes(bytes.clone()).unwrap();

    
    println!("Q1  $.orders.map(customer.address.city)    (repeat-shape; IC sweet spot)");
    run("tree_walker", || {
        j_tree
            .collect("$.orders.map(customer.address.city)")
            .unwrap()
    });

    println!("\nQ2  $.orders.map(customer.address.country_code).unique()");
    run("tree_walker", || {
        j_tree
            .collect("$.orders.map(customer.address.country_code).unique()")
            .unwrap()
    });

    
    println!("\nQ3  $.orders.filter(total > 500).map(id)   (bare-pred filter → scratch-Env)");
    run("tree_walker", || {
        j_tree
            .collect("$.orders.filter(total > 500).map(id)")
            .unwrap()
    });

    println!("\nQ4  $.orders.filter(status == \"shipped\" and priority == \"high\").count()");
    run("tree_walker", || {
        j_tree
            .collect(r#"$.orders.filter(status == "shipped" and priority == "high").count()"#)
            .unwrap()
    });

    
    println!("\nQ5  $..find(@.status == \"shipped\")   (deep enclosing-object search)");
    let q = r#"$..find(@.status == "shipped")"#;
    let a = run("tree_walker", || j_tree.collect(q).unwrap());
    let b = run("from_bytes", || j_bytes.collect(q).unwrap());
    compare(a, b);

    println!("\nQ6  $..find(@.sku == \"SKU-00042\")    (deep, narrow match)");
    let q = r#"$..find(@.sku == "SKU-00042")"#;
    let a = run("tree_walker", || j_tree.collect(q).unwrap());
    let b = run("from_bytes", || j_bytes.collect(q).unwrap());
    compare(a, b);

    println!("\nQ7  $..find(@.status==\"shipped\", @.priority==\"urgent\")   (multi-pred AND)");
    let q = r#"$..find(@.status == "shipped", @.priority == "urgent")"#;
    let a = run("tree_walker", || j_tree.collect(q).unwrap());
    let b = run("from_bytes", || j_bytes.collect(q).unwrap());
    compare(a, b);

    
    println!("\nQ8  $..total.sum()   (deep key + aggregate)");
    let q = "$..total.sum()";
    let a = run("tree_walker", || j_tree.collect(q).unwrap());
    let b = run("from_bytes", || j_bytes.collect(q).unwrap());
    compare(a, b);

    println!(
        "\nQ9  $..sku         (deep leaf extract, {} values)",
        n_orders * items_per_order
    );
    let q = "$..sku";
    let a = run("tree_walker", || j_tree.collect(q).unwrap());
    let b = run("from_bytes", || j_bytes.collect(q).unwrap());
    compare(a, b);

    
    println!("\nQ10 $.orders.group_by(status)         (group-by count_by-like)");
    run("tree_walker", || {
        j_tree.collect("$.orders.group_by(status)").unwrap()
    });

    println!("\nQ11 $.orders.count_by(region)");
    run("tree_walker", || {
        j_tree.collect("$.orders.count_by(region)").unwrap()
    });

    println!("\nQ12 $.orders.map(total).sum()");
    run("tree_walker", || {
        j_tree.collect("$.orders.map(total).sum()").unwrap()
    });

    
    println!("\nQ13 [o.id for o in $.orders if o.total > 1000]   (list comp)");
    run("tree_walker", || {
        j_tree
            .collect("[o.id for o in $.orders if o.total > 1000]")
            .unwrap()
    });

    println!("\nQ14 $.orders.pick(id, who: customer.name, city: customer.address.city)");
    run("tree_walker", || {
        j_tree
            .collect("$.orders.pick(id, who: customer.name, city: customer.address.city)")
            .unwrap()
    });

    
    println!("\nQ15 $.orders.map(total).max()   (aggregate max)");
    run("tree_walker", || {
        j_tree.collect("$.orders.map(total).max()").unwrap()
    });

    
    println!("\nQ16 $.orders[0].customer.address.set({{...}})   (deep patch; COW)");
    run("tree_walker", || {
        j_tree.collect(r#"$.orders[0].customer.address.set({"city": "Remote", "zip": "00000", "country_code": "XX", "street": "N/A"})"#).unwrap()
    });

    println!("\nQ17 $.orders[0].metadata.score.modify(@ * 2)   (nested path modify)");
    run("tree_walker", || {
        j_tree
            .collect("$.orders[0].metadata.score.modify(@ * 2)")
            .unwrap()
    });

    println!("\nQ18 $.orders[0].items[0].tags.set([])   (deep array-in-obj reset)");
    run("tree_walker", || {
        j_tree.collect("$.orders[0].items[0].tags.set([])").unwrap()
    });

    
    println!("\nBaseline");
    run("serde_json::from_slice (parse)", || {
        let _: Value = serde_json::from_slice(&bytes).unwrap();
        Value::Null
    });
}
