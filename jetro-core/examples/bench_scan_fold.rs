use jetro_core::Jetro;
use serde_json::json;
use std::time::Instant;

fn main() {
    let mut orders = Vec::with_capacity(100_000);
    for i in 0..100_000 {
        orders.push(json!({
            "id": format!("O-{}", i),
            "total": (i % 1000) as f64 * 1.3,
            "count": i % 50,
        }));
    }
    let doc = json!({ "orders": orders });
    let bytes = serde_json::to_vec(&doc).unwrap();
    let mb = bytes.len() as f64 / 1_048_576.0;
    println!("payload: {:.2} MB, 100k orders", mb);

    let j = Jetro::from_bytes(bytes).unwrap();

    for q in ["$..total.sum()", "$..total.avg()", "$..total.min()",
              "$..total.max()", "$..total.count()", "$..total"] {
        let _ = j.collect(q).unwrap();
        let mut ts = Vec::new();
        for _ in 0..30 {
            let t = Instant::now();
            let _ = j.collect(q).unwrap();
            ts.push(t.elapsed().as_micros());
        }
        ts.sort();
        println!("{:30} best {:>6}µs median {:>6}µs",
                 q, ts[0], ts[ts.len() / 2]);
    }
}
