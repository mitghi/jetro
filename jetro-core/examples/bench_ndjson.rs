use std::io::Cursor;
use std::time::Instant;

use jetro_core::JetroEngine;

fn build_ndjson(rows: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(rows * 128);
    for i in 0..rows {
        let active = if i % 3 == 0 { "true" } else { "false" };
        let score = 10_000usize.saturating_sub(i % 10_000);
        out.extend_from_slice(
            format!(
                r#"{{"id":{i},"name":"user_{i}","active":{active},"score":{score},"attributes":[{{"key":"k1","value":"v_{i}_1","weight":1}},{{"key":"k2","value":"v_{i}_2","weight":2}},{{"key":"k3","value":"v_{i}_3","weight":3}}]}}"#
            )
            .as_bytes(),
        );
        out.push(b'\n');
    }
    out
}

fn bench(engine: &JetroEngine, data: &[u8], label: &str, query: &str) {
    let start = Instant::now();
    let rows = engine
        .run_ndjson(Cursor::new(data), query, std::io::sink())
        .expect("NDJSON query should run");
    let elapsed = start.elapsed();
    let mb = data.len() as f64 / (1024.0 * 1024.0);
    let mb_s = mb / elapsed.as_secs_f64();
    println!("{label:<36} {rows:>8} rows {elapsed:>10.3?} {mb_s:>8.1} MiB/s");
}

fn bench_matches(engine: &JetroEngine, data: &[u8], label: &str, predicate: &str, limit: usize) {
    let start = Instant::now();
    let rows = engine
        .run_ndjson_matches(Cursor::new(data), predicate, limit, std::io::sink())
        .expect("NDJSON match query should run");
    let elapsed = start.elapsed();
    let mb = data.len() as f64 / (1024.0 * 1024.0);
    let mb_s = mb / elapsed.as_secs_f64();
    println!("{label:<36} {rows:>8} rows {elapsed:>10.3?} {mb_s:>8.1} MiB/s");
}

fn main() {
    let rows = std::env::args()
        .nth(1)
        .and_then(|arg| arg.parse::<usize>().ok())
        .unwrap_or(200_000);
    let data = build_ndjson(rows);
    let engine = JetroEngine::new();

    println!(
        "NDJSON cold-path core bench: {} rows, {:.1} MiB",
        rows,
        data.len() as f64 / (1024.0 * 1024.0)
    );
    bench(&engine, &data, "root int field", "id");
    bench(&engine, &data, "root string field", "name");
    bench(&engine, &data, "root string upper", "name.upper()");
    bench(&engine, &data, "array len", "attributes.len()");
    bench(
        &engine,
        &data,
        "first nested value",
        "attributes.first().value",
    );
    bench(&engine, &data, "map nested keys", "attributes.map(@.key)");
    bench(
        &engine,
        &data,
        "map nested pairs",
        "attributes.map([@.key, @.value])",
    );
    bench(
        &engine,
        &data,
        "map nested objects",
        "attributes.map({key: @.key, value: @.value})",
    );
    bench(
        &engine,
        &data,
        "filter nested count",
        r#"attributes.filter(@.value.contains("_3")).len()"#,
    );
    bench(
        &engine,
        &data,
        "filter numeric count",
        "attributes.filter(@.value > \"v_100_1\").len()",
    );
    bench(
        &engine,
        &data,
        "filter nested map keys",
        r#"attributes.filter(@.value.contains("_3")).map(@.key)"#,
    );
    bench(
        &engine,
        &data,
        "filter nested map pairs",
        r#"attributes.filter(@.value.contains("_3")).map([@.key, @.value])"#,
    );
    bench(
        &engine,
        &data,
        "map numeric sum",
        "attributes.map(@.weight).sum()",
    );
    bench(
        &engine,
        &data,
        "filter numeric sum",
        r#"attributes.filter(@.value.contains("_3")).map(@.weight).sum()"#,
    );
    bench(
        &engine,
        &data,
        "object projection",
        r#"{id: id, name: name, score: score, active: active}"#,
    );
    bench(
        &engine,
        &data,
        "object scalar projection",
        r#"{id: id, name: name.upper(), score: score, active: active}"#,
    );
    bench(
        &engine,
        &data,
        "array projection",
        r#"[id, name, score, active]"#,
    );
    bench(&engine, &data, "object keys", "$.keys()");
    bench_matches(&engine, &data, "match active rows", "active", rows);
    bench_matches(&engine, &data, "match score > 9900", "score > 9900", rows);
    bench_matches(
        &engine,
        &data,
        "match nested contains",
        r#"attributes.first().value.contains("_1")"#,
        rows,
    );
}
