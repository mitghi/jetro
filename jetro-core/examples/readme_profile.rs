use std::time::Instant;

const DATA: &[u8] = br#"
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

const ITERS: u32 = 50_000;

fn time_avg<F: FnMut()>(label: &str, mut f: F) {
    // Warm
    for _ in 0..1000 {
        f();
    }
    let t0 = Instant::now();
    for _ in 0..ITERS {
        f();
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let per = (elapsed / ITERS as f64) * 1e6; // us
    println!("{:<55} {:>9.2} µs/iter", label, per);
}

fn main() {
    let bytes = DATA.to_vec();

    time_avg("01 parse only (Jetro::from_bytes)", || {
        let _j = jetro_core::Jetro::from_bytes(bytes.clone()).unwrap();
    });

    time_avg("02 parse + $.orders", || {
        let j = jetro_core::Jetro::from_bytes(bytes.clone()).unwrap();
        let _v = j.collect("$.orders").unwrap();
    });

    time_avg("03 parse + $.orders.count()", || {
        let j = jetro_core::Jetro::from_bytes(bytes.clone()).unwrap();
        let _v = j.collect("$.orders.count()").unwrap();
    });

    time_avg("04 1-filter + count", || {
        let j = jetro_core::Jetro::from_bytes(bytes.clone()).unwrap();
        let _v = j
            .collect(r#"$.orders.filter(status == "paid").count()"#)
            .unwrap();
    });

    time_avg("05 3-filter + count", || {
        let j = jetro_core::Jetro::from_bytes(bytes.clone()).unwrap();
        let _v = j.collect(r#"$.orders.filter(status == "paid").filter(total >= 100).filter(customer.tier == "gold" or customer.tier == "platinum").count()"#).unwrap();
    });

    time_avg("06 3-filter + sort_by(-total) + count", || {
        let j = jetro_core::Jetro::from_bytes(bytes.clone()).unwrap();
        let _v = j.collect(r#"$.orders.filter(status == "paid").filter(total >= 100).filter(customer.tier == "gold" or customer.tier == "platinum").sort_by(-total).count()"#).unwrap();
    });

    time_avg("07 3-filter + sort + take(2)", || {
        let j = jetro_core::Jetro::from_bytes(bytes.clone()).unwrap();
        let _v = j.collect(r#"$.orders.filter(status == "paid").filter(total >= 100).filter(customer.tier == "gold" or customer.tier == "platinum").sort_by(-total).take(2)"#).unwrap();
    });

    time_avg("08 + map 4 simple fields", || {
        let j = jetro_core::Jetro::from_bytes(bytes.clone()).unwrap();
        let _v = j.collect(r#"$.orders.filter(status == "paid").filter(total >= 100).filter(customer.tier == "gold" or customer.tier == "platinum").sort_by(-total).take(2).map({order_id:id,who:customer.name,tier:customer.tier,amount:total})"#).unwrap();
    });

    time_avg("09 + f-string label", || {
        let j = jetro_core::Jetro::from_bytes(bytes.clone()).unwrap();
        let _v = j.collect(r##"$.orders.filter(status == "paid").filter(total >= 100).filter(customer.tier == "gold" or customer.tier == "platinum").sort_by(-total).take(2).map({order_id:id,who:customer.name,tier:customer.tier,amount:total,label:f"order {@.id}: {customer.name} ({customer.tier}) USD {@.total}"})"##).unwrap();
    });

    time_avg("10 + line_total nested map+sum", || {
        let j = jetro_core::Jetro::from_bytes(bytes.clone()).unwrap();
        let _v = j.collect(r##"$.orders.filter(status == "paid").filter(total >= 100).filter(customer.tier == "gold" or customer.tier == "platinum").sort_by(-total).take(2).map({order_id:id,who:customer.name,tier:customer.tier,amount:total,label:f"order {@.id}: {customer.name} ({customer.tier}) USD {@.total}",line_total:items.map(qty * price).sum()})"##).unwrap();
    });

    time_avg("11 FULL showcase + match last_event", || {
        let j = jetro_core::Jetro::from_bytes(bytes.clone()).unwrap();
        let _v = j.collect(r##"$.orders.filter(status == "paid").filter(total >= 100).filter(customer.tier == "gold" or customer.tier == "platinum").sort_by(-total).take(2).map({order_id:id,who:customer.name,tier:customer.tier,amount:total,label:f"order {@.id}: {customer.name} ({customer.tier}) USD {@.total}",line_total:items.map(qty * price).sum(),last_event:match events.last() with {{kind:"delivered",at:t}->{state:"ok",at:t},{kind:"shipped",at:t}->{state:"moving",at:t},{kind:"refund",reason:r}->{state:"refund",reason:r},_->{state:"unknown"}}})"##).unwrap();
    });

    println!("\n--- planner micro-bench (parse->plan only, no exec) [fuzz_internal] ---");
    #[cfg(feature = "fuzz_internal")]
    {
        let full_q = r##"$.orders.filter(status == "paid").filter(total >= 100).filter(customer.tier == "gold" or customer.tier == "platinum").sort_by(-total).take(2).map({order_id:id,who:customer.name,tier:customer.tier,amount:total,label:f"order {@.id}: {customer.name} ({customer.tier}) USD {@.total}",line_total:items.map(qty * price).sum(),last_event:match events.last() with {{kind:"delivered",at:t}->{state:"ok",at:t},{kind:"shipped",at:t}->{state:"moving",at:t},{kind:"refund",reason:r}->{state:"refund",reason:r},_->{state:"unknown"}}})"##;
        for _ in 0..1000 {
            let _ = jetro_core::__fuzz_internal::parse(full_q);
        }
        let t0 = Instant::now();
        for _ in 0..ITERS {
            let _ = jetro_core::__fuzz_internal::parse(full_q);
        }
        let per = (t0.elapsed().as_secs_f64() / ITERS as f64) * 1e6;
        println!(
            "{:<55} {:>9.2} µs/iter",
            "parse only (pest -> Expr AST)", per
        );

        for _ in 0..1000 {
            let _ = jetro_core::__fuzz_internal::plan_query(full_q);
        }
        let t0 = Instant::now();
        for _ in 0..ITERS {
            let _ = jetro_core::__fuzz_internal::plan_query(full_q);
        }
        let per = (t0.elapsed().as_secs_f64() / ITERS as f64) * 1e6;
        println!(
            "{:<55} {:>9.2} µs/iter",
            "plan_query (parse + plan, full)", per
        );
    }
    #[cfg(not(feature = "fuzz_internal"))]
    println!("(skipped — rebuild with --features fuzz_internal)");

    println!("\n--- isolate parse vs plan vs exec ---");
    // Reuse parsed Jetro across iters: only plan+exec measured.
    let j_persist = jetro_core::Jetro::from_bytes(bytes.clone()).unwrap();
    let full_q = r##"$.orders.filter(status == "paid").filter(total >= 100).filter(customer.tier == "gold" or customer.tier == "platinum").sort_by(-total).take(2).map({order_id:id,who:customer.name,tier:customer.tier,amount:total,label:f"order {@.id}: {customer.name} ({customer.tier}) USD {@.total}",line_total:items.map(qty * price).sum(),last_event:match events.last() with {{kind:"delivered",at:t}->{state:"ok",at:t},{kind:"shipped",at:t}->{state:"moving",at:t},{kind:"refund",reason:r}->{state:"refund",reason:r},_->{state:"unknown"}}})"##;
    for _ in 0..1000 {
        let _ = j_persist.collect(full_q).unwrap();
    }
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = j_persist.collect(full_q).unwrap();
    }
    let per = (t0.elapsed().as_secs_f64() / ITERS as f64) * 1e6;
    println!(
        "{:<55} {:>9.2} µs/iter",
        "Jetro reused, query reused (compile cached)", per
    );

    // Fresh Jetro each iter: parse + plan + exec
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let j = jetro_core::Jetro::from_bytes(bytes.clone()).unwrap();
        let _ = j.collect(full_q).unwrap();
    }
    let per = (t0.elapsed().as_secs_f64() / ITERS as f64) * 1e6;
    println!(
        "{:<55} {:>9.2} µs/iter",
        "Fresh Jetro each iter (parse+plan+exec)", per
    );

    println!("\n--- engine reuse (warm plan cache) ---");
    let eng = jetro_core::JetroEngine::default();
    let full_q = r##"$.orders.filter(status == "paid").filter(total >= 100).filter(customer.tier == "gold" or customer.tier == "platinum").sort_by(-total).take(2).map({order_id:id,who:customer.name,tier:customer.tier,amount:total,label:f"order {@.id}: {customer.name} ({customer.tier}) USD {@.total}",line_total:items.map(qty * price).sum(),last_event:match events.last() with {{kind:"delivered",at:t}->{state:"ok",at:t},{kind:"shipped",at:t}->{state:"moving",at:t},{kind:"refund",reason:r}->{state:"refund",reason:r},_->{state:"unknown"}}})"##;
    // Warm
    for _ in 0..1000 {
        let _ = eng.collect_bytes(bytes.clone(), full_q).unwrap();
    }
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = eng.collect_bytes(bytes.clone(), full_q).unwrap();
    }
    let per = (t0.elapsed().as_secs_f64() / ITERS as f64) * 1e6;
    println!(
        "{:<55} {:>9.2} µs/iter",
        "WARM full showcase via JetroEngine", per
    );
}
