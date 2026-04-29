//! Directional benchmark for native f-string map execution.
//!
//! Run:
//!   cargo run --release -p jetro-core --example bench_fstring

use std::time::Instant;

use jetro_core::Jetro;
use serde_json::{json, Value};

const ITERS: usize = 30;

fn synth_doc(rows: usize) -> Value {
    let cities = ["Berlin", "Tokyo", "Austin", "NYC", "Toronto", "Paris"];
    let names = ["ada", "bob", "cat", "dan", "eve", "max"];
    let mut data = Vec::with_capacity(rows);
    for i in 0..rows {
        data.push(json!({
            "id": i as i64,
            "score": ((i * 37) % 10_000) as i64,
            "user": {
                "name": names[i % names.len()],
                "addr": {
                    "city": cities[i % cities.len()],
                    "zip": format!("{:05}", (i * 19) % 100_000),
                }
            },
            "payload": {
                "ignored": [1, 2, 3, 4, 5, 6]
            }
        }));
    }
    json!({ "data": data, "meta": { "kind": "fstring_bench" } })
}

fn bench<F>(label: &str, mut f: F)
where
    F: FnMut() -> Value,
{
    let _ = f();
    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let start = Instant::now();
        let out = f();
        std::hint::black_box(out);
        samples.push(start.elapsed().as_micros());
    }
    samples.sort_unstable();
    let best = samples[0];
    let median = samples[samples.len() / 2];
    let mean = samples.iter().sum::<u128>() / samples.len() as u128;
    println!("{label:<34} best {best:>8}us  median {median:>8}us  mean {mean:>8}us");
}

fn run_size(rows: usize) {
    let doc = synth_doc(rows);
    let bytes = serde_json::to_vec(&doc).unwrap();
    let mb = bytes.len() as f64 / 1_048_576.0;
    let tree = Jetro::from(doc);
    let tape = Jetro::from_bytes(bytes).unwrap();

    let q_fstring = r##"$.data.map(f"#{id} {user.name} ({user.addr.city}) ${score}")"##;
    let q_fields = "$.data.map(user.addr.city)";
    let q_object = "$.data.map({id, name: user.name, city: user.addr.city, score})";

    println!("\nrows: {rows}, payload: {mb:.2} MB");
    bench("tree f-string", || tree.collect(q_fstring).unwrap());
    bench("tape f-string", || tape.collect(q_fstring).unwrap());
    bench("tape field-chain map", || tape.collect(q_fields).unwrap());
    bench("tape object map", || tape.collect(q_object).unwrap());
}

fn main() {
    for rows in [1_000usize, 10_000, 100_000] {
        run_size(rows);
    }
}
