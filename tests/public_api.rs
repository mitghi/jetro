use jetro::{
    io::{
        ndjson_rows_plan_kind, ndjson_writer_path_kind, DistinctFrontFilterKind,
        NdjsonRowsPlanKind, NdjsonSource, NdjsonWriterPathKind,
    },
    JetroEngine,
};
use serde_json::json;
use std::io::Cursor;

#[test]
fn facade_exposes_ndjson_route_observability() {
    let engine = JetroEngine::new();
    assert_eq!(
        ndjson_writer_path_kind(&engine, "$.name"),
        Some(NdjsonWriterPathKind::ByteExpr)
    );
    assert_eq!(
        ndjson_rows_plan_kind("$.rows().take(1)").unwrap(),
        Some(NdjsonRowsPlanKind::Stream)
    );
}

#[test]
fn facade_exposes_ndjson_match_api() {
    let engine = JetroEngine::new();
    let rows = Cursor::new(
        br#"{"id":1,"active":true}
{"id":2,"active":false}
{"id":3,"active":true}
"#,
    );

    let out = engine
        .collect_ndjson_matches_source(NdjsonSource::reader(rows), "active", 2)
        .expect("facade re-exported match API should run");

    assert_eq!(
        out,
        vec![
            json!({"id": 1, "active": true}),
            json!({"id": 3, "active": true})
        ]
    );
}

#[test]
fn facade_exposes_reverse_distinct_by_stats_api() {
    let engine = JetroEngine::new();
    let mut path = std::env::temp_dir();
    path.push(format!(
        "jetro-facade-rev-distinct-{}.ndjson",
        std::process::id()
    ));
    std::fs::write(
        &path,
        b"{\"id\":1,\"version\":1}\n{\"id\":2,\"version\":1}\n{\"id\":1,\"version\":2}\n",
    )
    .unwrap();
    let mut out = Vec::new();

    let stats = engine
        .run_ndjson_rev_distinct_by_with_stats(&path, "id", "version", 10, &mut out)
        .expect("facade re-exported reverse distinct_by stats API should run");

    let _ = std::fs::remove_file(&path);
    assert_eq!(String::from_utf8(out).unwrap(), "2\n1\n");
    assert_eq!(stats.emitted, 2);
    assert_eq!(stats.duplicate_rows, 1);
    assert_eq!(stats.front_filter, DistinctFrontFilterKind::None);
}
