use jetro::{
    io::{
        ndjson_explain, ndjson_rows_plan_kind, ndjson_writer_path_kind, DistinctFrontFilterKind,
        NdjsonFallbackReason, NdjsonRouteKind, NdjsonRowsPlanKind, NdjsonSource,
        NdjsonSourceMode, NdjsonWriterPathKind, NdjsonOptions, NdjsonRowFrame, NullPayload,
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
    let row = ndjson_explain(
        &engine,
        NdjsonSourceMode::Reader,
        "$.name",
        Default::default(),
    )
    .unwrap();
    assert_eq!(row.kind, NdjsonRouteKind::RowLocal);
    assert_eq!(row.writer_path, Some(NdjsonWriterPathKind::ByteExpr));

    let unsupported = ndjson_explain(
        &engine,
        NdjsonSourceMode::Reader,
        r#"{head: $.rows().take(1)}"#,
        Default::default(),
    )
    .unwrap();
    assert_eq!(unsupported.kind, NdjsonRouteKind::UnsupportedRows);
    assert_eq!(
        unsupported.fallback_reason,
        Some(NdjsonFallbackReason::FileBackedRowsRequired)
    );

    let framed = ndjson_explain(
        &engine,
        NdjsonSourceMode::File,
        "$.rows().take(1)",
        NdjsonOptions::default().with_row_frame(NdjsonRowFrame::DelimitedPayload {
            separator: b'|',
            null_payload: NullPayload::Skip,
        }),
    )
    .unwrap();
    assert!(framed.source.framed_payload);
    assert!(framed.source.to_string().contains("framed-payload"));
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
fn facade_exposes_ndjson_execution_report_api() {
    let engine = JetroEngine::new();
    let rows = Cursor::new(
        br#"{"id":1,"active":true}
{"id":2,"active":false}
"#,
    );
    let mut out = Vec::new();

    let report = engine
        .run_ndjson_with_report(rows, "$.rows().filter($.active).map($.id)", &mut out)
        .expect("facade report API should run");

    assert_eq!(String::from_utf8(out).unwrap(), "1\n");
    assert_eq!(report.route.kind, NdjsonRouteKind::RowsStream);
    assert_eq!(report.stats.rows_scanned, 2);
    assert_eq!(report.stats.rows_emitted, 1);
}

#[test]
fn facade_exposes_ndjson_match_and_reverse_report_api() {
    let engine = JetroEngine::new();
    let rows = Cursor::new(
        br#"{"id":1,"active":true}
{"id":2,"active":false}
"#,
    );
    let mut out = Vec::new();

    let matches = engine
        .run_ndjson_matches_with_report(rows, "active", 10, &mut out)
        .expect("facade match report API should run");

    assert_eq!(matches.route.kind, NdjsonRouteKind::Matches);
    assert_eq!(matches.stats.rows_scanned, 2);
    assert_eq!(matches.stats.rows_emitted, 1);

    let mut path = std::env::temp_dir();
    path.push(format!(
        "jetro-facade-rev-report-{}.ndjson",
        std::process::id()
    ));
    std::fs::write(
        &path,
        b"{\"id\":1,\"v\":1}\n{\"id\":2,\"v\":1}\n{\"id\":1,\"v\":2}\n",
    )
    .unwrap();
    let mut rev_out = Vec::new();

    let reverse = engine
        .run_ndjson_rev_distinct_by_with_report(&path, "id", "v", 10, &mut rev_out)
        .expect("facade reverse distinct report API should run");

    let _ = std::fs::remove_file(&path);
    assert_eq!(String::from_utf8(rev_out).unwrap(), "2\n1\n");
    assert_eq!(reverse.stats.rows_scanned, 3);
    assert_eq!(reverse.stats.duplicate_rows, 1);

    std::fs::write(
        &path,
        b"{\"id\":1,\"active\":true}\n{\"id\":2,\"active\":false}\n{\"id\":3,\"active\":true}\n",
    )
    .unwrap();
    let mut match_out = Vec::new();
    let reverse_matches = engine
        .run_ndjson_rev_matches_with_report(&path, "active", 2, &mut match_out)
        .expect("facade reverse match report API should run");
    let _ = std::fs::remove_file(&path);
    assert_eq!(reverse_matches.route.kind, NdjsonRouteKind::Matches);
    assert_eq!(reverse_matches.stats.rows_scanned, 3);
    assert_eq!(reverse_matches.stats.rows_emitted, 2);
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
