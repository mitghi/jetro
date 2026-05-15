use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::time::Instant;

use jetro_core::io::{
    DistinctFrontFilterKind, NdjsonOptions, NdjsonRowFrame, NullPayload, PayloadSide,
};
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

fn bench_with_options(
    engine: &JetroEngine,
    data: &[u8],
    label: &str,
    query: &str,
    options: NdjsonOptions,
) {
    let start = Instant::now();
    let rows = engine
        .run_ndjson_with_options(Cursor::new(data), query, std::io::sink(), options)
        .expect("NDJSON query with options should run");
    let elapsed = start.elapsed();
    let mb = data.len() as f64 / (1024.0 * 1024.0);
    let mb_s = mb / elapsed.as_secs_f64();
    println!("{label:<36} {rows:>8} rows {elapsed:>10.3?} {mb_s:>8.1} MiB/s");
}

fn bench_rows_stream_reader(engine: &JetroEngine, data: &[u8], label: &str, query: &str) {
    let start = Instant::now();
    let rows = engine
        .run_ndjson(Cursor::new(data), query, std::io::sink())
        .expect("NDJSON rows() stream query should run");
    let elapsed = start.elapsed();
    let mb = data.len() as f64 / (1024.0 * 1024.0);
    let mb_s = mb / elapsed.as_secs_f64();
    println!("{label:<36} {rows:>8} rows {elapsed:>10.3?} {mb_s:>8.1} MiB/s");
}

fn bench_rows_stream_file(
    engine: &JetroEngine,
    path: &Path,
    bytes: usize,
    label: &str,
    query: &str,
) {
    let start = Instant::now();
    let rows = engine
        .run_ndjson_file(path, query, std::io::sink())
        .expect("file-backed NDJSON rows() stream query should run");
    let elapsed = start.elapsed();
    let mb = bytes as f64 / (1024.0 * 1024.0);
    let mb_s = mb / elapsed.as_secs_f64();
    println!("{label:<36} {rows:>8} rows {elapsed:>10.3?} {mb_s:>8.1} MiB/s");
}

fn build_compacted_ndjson(rows: usize, keys: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(rows * 96);
    let keys = keys.max(1);
    for i in 0..rows {
        let id = i % keys;
        let version = i / keys;
        out.extend_from_slice(
            format!(
                r#"{{"id":{id},"version":{version},"name":"user_{id}","active":{},"payload":{{"score":{},"tag":"t_{}"}}}}"#,
                i % 2 == 0,
                10_000usize.saturating_sub(i % 10_000),
                i % 17
            )
            .as_bytes(),
        );
        out.push(b'\n');
    }
    out
}

fn build_framed_ndjson(rows: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(rows * 144);
    for i in 0..rows {
        if i % 11 == 0 {
            out.extend_from_slice(format!("key-{i}|null\n").as_bytes());
            continue;
        }
        out.extend_from_slice(
            format!(
                r#"key-{i}|{{"id":{i},"name":"user_{i}","active":{},"score":{}}}"#,
                i % 3 == 0,
                10_000usize.saturating_sub(i % 10_000)
            )
            .as_bytes(),
        );
        out.push(b'\n');
    }
    out
}

fn write_temp_ndjson(label: &str, data: &[u8]) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!("jetro-bench-{label}-{}.ndjson", std::process::id()));
    std::fs::write(&path, data).expect("bench temp file should be writable");
    path
}

fn bench_rev_distinct(
    engine: &JetroEngine,
    path: &Path,
    bytes: usize,
    label: &str,
    key_query: &str,
    query: &str,
    limit: usize,
) {
    let start = Instant::now();
    let stats = engine
        .run_ndjson_rev_distinct_by_with_stats(path, key_query, query, limit, std::io::sink())
        .expect("reverse distinct_by bench should run");
    let elapsed = start.elapsed();
    let mb = bytes as f64 / (1024.0 * 1024.0);
    let mb_s = mb / elapsed.as_secs_f64();
    let front = match stats.front_filter {
        DistinctFrontFilterKind::None => "none",
        DistinctFrontFilterKind::Bloom => "bloom",
        DistinctFrontFilterKind::Cuckoo => "cuckoo",
    };
    println!(
        "{label:<36} {:>8} rows {elapsed:>10.3?} {mb_s:>8.1} MiB/s dup={:<8} key={}/{} val={}/{} front={front}",
        stats.emitted,
        stats.duplicate_rows,
        stats.direct_key_rows,
        stats.fallback_key_rows,
        stats.direct_value_rows,
        stats.fallback_value_rows,
    );
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
    bench(
        &engine,
        &data,
        "first nested upper",
        "attributes.first().key.upper()",
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

    let framed = build_framed_ndjson(rows);
    let framed_options =
        NdjsonOptions::default().with_row_frame(NdjsonRowFrame::DelimitedPayload {
            separator: b'|',
            side: PayloadSide::AfterSeparator,
            null_payload: NullPayload::Skip,
        });
    println!("\nDelimited payload framing:");
    bench_with_options(
        &engine,
        &framed,
        "framed root projection",
        "$.name",
        framed_options,
    );
    bench_with_options(
        &engine,
        &framed,
        "framed rows take",
        "$.rows().take(1000).map($.id)",
        framed_options,
    );

    let high_dup = build_compacted_ndjson(rows, (rows / 100).max(1));
    let low_dup = build_compacted_ndjson(rows, rows);
    let high_dup_path = write_temp_ndjson("high-dup", &high_dup);
    let low_dup_path = write_temp_ndjson("low-dup", &low_dup);
    println!("\nReverse compacted-topic distinct_by:");
    bench_rev_distinct(
        &engine,
        &high_dup_path,
        high_dup.len(),
        "distinct high-dup direct",
        "id",
        r#"{id: id, version: version}"#,
        rows,
    );
    println!("\nExpression-level rows() streams:");
    bench_rows_stream_reader(
        &engine,
        &data,
        "rows take+project",
        "$.rows().take(1000).map($.name)",
    );
    bench_rows_stream_reader(
        &engine,
        &data,
        "rows filter+distinct+take",
        "$.rows().filter($.active == true).distinct_by($.id).take(1000).map({id: $.id, name: $.name})",
    );
    bench_rows_stream_file(
        &engine,
        &high_dup_path,
        high_dup.len(),
        "rows reverse+take",
        "$.rows().reverse().take(1000).map($.id)",
    );
    bench_rows_stream_file(
        &engine,
        &high_dup_path,
        high_dup.len(),
        "rows reverse+distinct+take",
        "$.rows().reverse().distinct_by($.id).take(1000).map({id: $.id, version: $.version})",
    );
    bench_rows_stream_file(
        &engine,
        &high_dup_path,
        high_dup.len(),
        "rows fallback key",
        "$.rows().reverse().distinct_by($.name.upper()).take(1000).map($.payload.score)",
    );
    bench_rev_distinct(
        &engine,
        &high_dup_path,
        high_dup.len(),
        "distinct high-dup limit",
        "id",
        "payload.score",
        100,
    );
    bench_rev_distinct(
        &engine,
        &low_dup_path,
        low_dup.len(),
        "distinct low-dup direct",
        "id",
        "name",
        rows,
    );
    bench_rev_distinct(
        &engine,
        &high_dup_path,
        high_dup.len(),
        "distinct fallback key",
        "name.upper()",
        "payload.score",
        rows,
    );
    let _ = std::fs::remove_file(high_dup_path);
    let _ = std::fs::remove_file(low_dup_path);
}
