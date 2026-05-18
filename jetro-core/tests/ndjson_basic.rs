use jetro_core::io::{
    ndjson_rows_plan_kind, ndjson_writer_path_kind, DistinctFrontFilterKind, NdjsonControl,
    NdjsonNullOutput, NdjsonOptions, NdjsonRowFrame, NdjsonRowsPlanKind, NdjsonSource,
    NdjsonWriterPathKind, NullPayload,
};
use jetro_core::{JetroEngine, JetroEngineError};
use serde_json::json;
use std::io::Cursor;
use std::path::PathBuf;

#[test]
fn ndjson_public_plan_observability_reports_routes() {
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
fn collect_ndjson_evaluates_query_per_non_empty_row() {
    let engine = JetroEngine::new();
    let input = br#"
{"name":"Ada","active":true}

  {"name":"Bob","active":false}
"#;

    let out = engine
        .collect_ndjson(Cursor::new(input), "name")
        .expect("ndjson query should run");

    assert_eq!(out, vec![json!("Ada"), json!("Bob")]);
}

#[test]
fn run_ndjson_writes_one_json_result_per_row() {
    let engine = JetroEngine::new();
    let input = br#"{"n":1}
{"n":2}
"#;
    let mut out = Vec::new();

    let rows = engine
        .run_ndjson(Cursor::new(input), "n + 1", &mut out)
        .expect("ndjson query should run");

    assert_eq!(rows, 2);
    assert_eq!(String::from_utf8(out).unwrap(), "2\n3\n");
}

#[test]
fn run_ndjson_skips_null_results_by_default() {
    let engine = JetroEngine::new();
    let input = br#"{"id":1}
{"missing":true}
{"id":3}
"#;
    let mut out = Vec::new();

    let rows = engine
        .run_ndjson(Cursor::new(input), "$.id", &mut out)
        .expect("ndjson query should run");

    assert_eq!(rows, 2);
    assert_eq!(String::from_utf8(out).unwrap(), "1\n3\n");
}

#[test]
fn run_ndjson_can_emit_null_results_when_configured() {
    let engine = JetroEngine::new();
    let input = br#"{"id":1}
{"missing":true}
{"id":3}
"#;
    let mut out = Vec::new();

    let rows = engine
        .run_ndjson_with_options(
            Cursor::new(input),
            "$.id",
            &mut out,
            NdjsonOptions::default().with_null_output(NdjsonNullOutput::Emit),
        )
        .expect("ndjson query should run");

    assert_eq!(rows, 3);
    assert_eq!(String::from_utf8(out).unwrap(), "1\nnull\n3\n");
}

#[test]
fn run_ndjson_delimited_payload_skips_tombstones() {
    let engine = JetroEngine::new();
    let input = br#"key-a|{"id":"a","v":1}
key-b|null
key-c| {"id":"c","v":3}
"#;
    let options = NdjsonOptions::default().with_row_frame(NdjsonRowFrame::DelimitedPayload {
        separator: b'|',
        null_payload: NullPayload::Skip,
    });
    let mut out = Vec::new();

    let rows = engine
        .run_ndjson_with_options(Cursor::new(input), "$.id", &mut out, options)
        .expect("delimited payload rows should run");

    assert_eq!(rows, 2);
    assert_eq!(String::from_utf8(out).unwrap(), "\"a\"\n\"c\"\n");
}

#[test]
fn run_ndjson_delimited_payload_skips_non_payload_records() {
    let engine = JetroEngine::new();
    let options = NdjsonOptions::default().with_row_frame(NdjsonRowFrame::DelimitedPayload {
        separator: b'|',
        null_payload: NullPayload::Skip,
    });
    let mut out = Vec::new();

    let rows = engine
        .run_ndjson_with_options(
            Cursor::new(
                b"missing-separator\nk-empty|\nk-null|null\nk-bad|not-json\nk-ok|{\"id\":\"ok\"}\n",
            ),
            "$.id",
            &mut out,
            options,
        )
        .expect("non-payload records should be skipped before parsing");

    assert_eq!(rows, 1);
    assert_eq!(String::from_utf8(out).unwrap(), "\"ok\"\n");
}

#[test]
fn run_ndjson_delimited_payload_can_keep_or_reject_null() {
    let engine = JetroEngine::new();
    let keep = NdjsonOptions::default()
        .with_row_frame(NdjsonRowFrame::DelimitedPayload {
            separator: b'|',
            null_payload: NullPayload::Keep,
        })
        .with_null_output(NdjsonNullOutput::Emit);
    let reject = NdjsonOptions::default().with_row_frame(NdjsonRowFrame::DelimitedPayload {
        separator: b'|',
        null_payload: NullPayload::Error,
    });
    let mut kept = Vec::new();
    let mut rejected = Vec::new();

    let rows = engine
        .run_ndjson_with_options(Cursor::new(b"k|null\n"), "$", &mut kept, keep)
        .expect("null payload should be valid when configured");
    let err = engine
        .run_ndjson_with_options(Cursor::new(b"k|null\n"), "$", &mut rejected, reject)
        .expect_err("null payload should be rejected when configured");

    assert_eq!(rows, 1);
    assert_eq!(String::from_utf8(kept).unwrap(), "null\n");
    assert!(err.to_string().contains("null framed payload"));
    assert!(rejected.is_empty());
}

#[test]
fn run_ndjson_writes_scalar_results_directly() {
    let engine = JetroEngine::new();
    let input = b"{\"s\":\"a\\\"b\\\\c\\n\",\"b\":true,\"z\":null,\"f\":1.25}\n";

    let mut string_out = Vec::new();
    engine
        .run_ndjson(Cursor::new(input), "s", &mut string_out)
        .expect("string scalar should write");
    assert_eq!(
        String::from_utf8(string_out).unwrap(),
        "\"a\\\"b\\\\c\\n\"\n"
    );

    let mut bool_out = Vec::new();
    engine
        .run_ndjson(Cursor::new(input), "b", &mut bool_out)
        .expect("bool scalar should write");
    assert_eq!(String::from_utf8(bool_out).unwrap(), "true\n");

    let mut null_out = Vec::new();
    engine
        .run_ndjson(Cursor::new(input), "z", &mut null_out)
        .expect("null scalar should write");
    assert_eq!(String::from_utf8(null_out).unwrap(), "");

    let mut float_out = Vec::new();
    engine
        .run_ndjson(Cursor::new(input), "f", &mut float_out)
        .expect("float scalar should write");
    assert_eq!(String::from_utf8(float_out).unwrap(), "1.25\n");

    let mut upper_out = Vec::new();
    engine
        .run_ndjson(Cursor::new(input), "s.upper()", &mut upper_out)
        .expect("view-scalar call should write");
    assert_eq!(
        String::from_utf8(upper_out).unwrap(),
        "\"A\\\"B\\\\C\\n\"\n"
    );

    let mut lower_out = Vec::new();
    engine
        .run_ndjson(
            Cursor::new(b"{\"s\":\"ADA\"}\n"),
            "$.s.lower()",
            &mut lower_out,
        )
        .expect("byte scalar lower call should write");
    assert_eq!(String::from_utf8(lower_out).unwrap(), "\"ada\"\n");
}

#[test]
fn run_ndjson_writes_root_fields_from_byte_path() {
    let engine = JetroEngine::new();
    let input =
        b"{\"id\":1,\"name\":\"Ada\"}\n{\"name\":\"Bob\"}\n{\"i\\u0064\":3,\"name\":\"Cat\"}\n";
    let mut id_out = Vec::new();

    engine
        .run_ndjson(Cursor::new(input), "$.id", &mut id_out)
        .expect("root field should write");

    assert_eq!(String::from_utf8(id_out).unwrap(), "1\n3\n");
}

#[test]
fn run_ndjson_writes_array_and_object_results_directly() {
    let engine = JetroEngine::new();
    let input = br#"{"id":7,"attributes":[{"key":"a","value":1},{"key":"b","value":2}]}
"#;

    let mut array_out = Vec::new();
    engine
        .run_ndjson(Cursor::new(input), "attributes.map(@.key)", &mut array_out)
        .expect("array projection should write");
    assert_eq!(String::from_utf8(array_out).unwrap(), "[\"a\",\"b\"]\n");

    let mut array_pair_out = Vec::new();
    engine
        .run_ndjson(
            Cursor::new(input),
            "attributes.map([@.key, @.value])",
            &mut array_pair_out,
        )
        .expect("array pair projection should write");
    assert_eq!(
        String::from_utf8(array_pair_out).unwrap(),
        "[[\"a\",1],[\"b\",2]]\n"
    );

    let mut object_map_out = Vec::new();
    engine
        .run_ndjson(
            Cursor::new(input),
            "attributes.map({k: @.key, v: @.value})",
            &mut object_map_out,
        )
        .expect("object map projection should write");
    assert_eq!(
        String::from_utf8(object_map_out).unwrap(),
        "[{\"k\":\"a\",\"v\":1},{\"k\":\"b\",\"v\":2}]\n"
    );

    let mut filtered_map_out = Vec::new();
    engine
        .run_ndjson(
            Cursor::new(input),
            "attributes.filter(@.value > 1).map(@.key)",
            &mut filtered_map_out,
        )
        .expect("filtered array projection should write");
    assert_eq!(String::from_utf8(filtered_map_out).unwrap(), "[\"b\"]\n");

    let mut filtered_pair_out = Vec::new();
    engine
        .run_ndjson(
            Cursor::new(input),
            "attributes.filter(@.value > 1).map([@.key, @.value])",
            &mut filtered_pair_out,
        )
        .expect("filtered array pair projection should write");
    assert_eq!(
        String::from_utf8(filtered_pair_out).unwrap(),
        "[[\"b\",2]]\n"
    );

    let mut filtered_object_out = Vec::new();
    engine
        .run_ndjson(
            Cursor::new(input),
            "attributes.filter(@.value > 1).map({k: @.key, v: @.value})",
            &mut filtered_object_out,
        )
        .expect("filtered object projection should write");
    assert_eq!(
        String::from_utf8(filtered_object_out).unwrap(),
        "[{\"k\":\"b\",\"v\":2}]\n"
    );

    let mut filtered_count_out = Vec::new();
    engine
        .run_ndjson(
            Cursor::new(input),
            "attributes.filter(@.value > 1).len()",
            &mut filtered_count_out,
        )
        .expect("filtered count should write");
    assert_eq!(String::from_utf8(filtered_count_out).unwrap(), "1\n");

    let mut numeric_reduce_out = Vec::new();
    engine
        .run_ndjson(
            Cursor::new(input),
            "attributes.map(@.value).sum()",
            &mut numeric_reduce_out,
        )
        .expect("numeric reducer should write");
    assert_eq!(String::from_utf8(numeric_reduce_out).unwrap(), "3\n");

    let mut object_out = Vec::new();
    engine
        .run_ndjson(
            Cursor::new(input),
            "{id: id, first: attributes.first().value}",
            &mut object_out,
        )
        .expect("object projection should write");
    assert_eq!(
        String::from_utf8(object_out).unwrap(),
        "{\"id\":7,\"first\":1}\n"
    );
}

#[test]
fn run_ndjson_writes_direct_array_element_projections() {
    let engine = JetroEngine::new();
    let input =
        br#"{"attributes":[{"key":"a","value":1},{"key":"b","value":2},{"key":"c","value":3}]}
"#;

    let cases = [
        ("attributes.first().key", "\"a\"\n"),
        ("attributes.first().key.upper()", "\"A\"\n"),
        ("attributes.last().key", "\"c\"\n"),
        ("attributes.last().key.upper()", "\"C\"\n"),
        ("attributes.nth(1).value", "2\n"),
    ];

    for (query, expected) in cases {
        let mut out = Vec::new();
        engine
            .run_ndjson(Cursor::new(input), query, &mut out)
            .expect("array element projection should write");
        assert_eq!(String::from_utf8(out).unwrap(), expected, "{query}");
    }
}

#[test]
fn run_ndjson_writes_static_object_projection_directly() {
    let engine = JetroEngine::new();
    let input = br#"{"id":7,"name":"Ada","score":42}
"#;
    let mut out = Vec::new();

    engine
        .run_ndjson(
            Cursor::new(input),
            r#"{id: id, label: name.upper(), score: score, kind: "user"}"#,
            &mut out,
        )
        .expect("static object projection should write");

    assert_eq!(
        String::from_utf8(out).unwrap(),
        r#"{"id":7,"label":"ADA","score":42,"kind":"user"}"#.to_string() + "\n"
    );
}

#[test]
fn run_ndjson_writes_static_array_projection_directly() {
    let engine = JetroEngine::new();
    let input = br#"{"id":7,"name":"Ada","score":42}
"#;
    let mut out = Vec::new();

    engine
        .run_ndjson(Cursor::new(input), r#"[id, name.upper(), score]"#, &mut out)
        .expect("static array projection should write");

    assert_eq!(String::from_utf8(out).unwrap(), "[7,\"ADA\",42]\n");
}

#[test]
fn run_ndjson_writes_object_item_methods_directly() {
    let engine = JetroEngine::new();
    let input = br#"{"id":7,"name":"Ada"}
"#;

    let mut keys_out = Vec::new();
    engine
        .run_ndjson(Cursor::new(input), "$.keys()", &mut keys_out)
        .expect("keys should write");
    assert_eq!(String::from_utf8(keys_out).unwrap(), "[\"id\",\"name\"]\n");

    let mut values_out = Vec::new();
    engine
        .run_ndjson(Cursor::new(input), "$.values()", &mut values_out)
        .expect("values should write");
    assert_eq!(String::from_utf8(values_out).unwrap(), "[7,\"Ada\"]\n");

    let mut entries_out = Vec::new();
    engine
        .run_ndjson(Cursor::new(input), "$.entries()", &mut entries_out)
        .expect("entries should write");
    assert_eq!(
        String::from_utf8(entries_out).unwrap(),
        "[[\"id\",7],[\"name\",\"Ada\"]]\n"
    );
}

#[test]
fn run_ndjson_direct_path_cache_handles_field_order_changes() {
    let engine = JetroEngine::new();
    let input = br#"{"id":1,"name":"Ada","score":10}
{"score":20,"name":"Bob","id":2}
"#;
    let mut out = Vec::new();

    engine
        .run_ndjson(
            Cursor::new(input),
            r#"{id: id, name: name, score: score}"#,
            &mut out,
        )
        .expect("direct projection should handle field order changes");

    assert_eq!(
        String::from_utf8(out).unwrap(),
        "{\"id\":1,\"name\":\"Ada\",\"score\":10}\n{\"id\":2,\"name\":\"Bob\",\"score\":20}\n"
    );
}

#[test]
fn run_ndjson_direct_path_cache_handles_nested_field_order_changes() {
    let engine = JetroEngine::new();
    let input = br#"{"user":{"name":"Ada","profile":{"score":10,"city":"Berlin"}}}
{"user":{"profile":{"city":"Paris","score":20},"name":"Bob"}}
"#;
    let mut out = Vec::new();

    engine
        .run_ndjson(
            Cursor::new(input),
            r#"{name: user.name, city: user.profile.city, score: user.profile.score}"#,
            &mut out,
        )
        .expect("direct projection should handle nested field order changes");

    assert_eq!(
        String::from_utf8(out).unwrap(),
        "{\"name\":\"Ada\",\"city\":\"Berlin\",\"score\":10}\n{\"name\":\"Bob\",\"city\":\"Paris\",\"score\":20}\n"
    );
}

#[test]
fn run_ndjson_filtered_numeric_reduce_honors_scalar_source_predicate() {
    let engine = JetroEngine::new();
    let input = br#"{"attributes":{"active":false,"value":10}}
{"attributes":{"active":true,"value":7}}
"#;
    let mut out = Vec::new();

    engine
        .run_ndjson(
            Cursor::new(input),
            "attributes.filter(@.active).map(@.value).sum()",
            &mut out,
        )
        .expect("filtered scalar-source reducer should write");

    assert_eq!(String::from_utf8(out).unwrap(), "0\n7\n");
}

#[test]
fn run_ndjson_limit_writes_and_stops_without_value_callback() {
    let engine = JetroEngine::new();
    let input = br#"{"n":1}
{"n":2}
not-json
"#;
    let mut out = Vec::new();

    let rows = engine
        .run_ndjson_limit(Cursor::new(input), "n + 1", 2, &mut out)
        .expect("writer limit should stop before invalid tail");

    assert_eq!(rows, 2);
    assert_eq!(String::from_utf8(out).unwrap(), "2\n3\n");
}

#[test]
fn rows_stream_take_map_runs_over_ndjson_rows() {
    let engine = JetroEngine::new();
    let input = br#"{"id":1,"name":"Ada"}
{"id":2,"name":"Bob"}
not-json
"#;
    let mut out = Vec::new();

    let rows = engine
        .run_ndjson(Cursor::new(input), "$.rows().take(2).map($.name)", &mut out)
        .expect("rows stream should stop before the invalid tail");

    assert_eq!(rows, 2);
    assert_eq!(String::from_utf8(out).unwrap(), "\"Ada\"\n\"Bob\"\n");
}

#[test]
fn run_ndjson_with_report_returns_rows_stream_stats() {
    let engine = JetroEngine::new();
    let input = br#"{"id":1,"active":true}
{"id":2,"active":false}
{"id":3,"active":true}
"#;
    let mut out = Vec::new();

    let report = engine
        .run_ndjson_with_report(
            Cursor::new(input),
            "$.rows().filter($.active).map($.id)",
            &mut out,
        )
        .expect("rows stream report should run");

    assert_eq!(String::from_utf8(out).unwrap(), "1\n3\n");
    assert_eq!(report.route.kind.to_string(), "rows-stream");
    assert_eq!(report.stats.rows_scanned, 3);
    assert_eq!(report.stats.rows_emitted, 2);
    assert_eq!(report.stats.rows_filtered, 1);
}

#[test]
fn run_ndjson_with_report_keeps_row_local_route() {
    let engine = JetroEngine::new();
    let input = br#"{"id":1}
{"id":2}
"#;
    let mut out = Vec::new();

    let report = engine
        .run_ndjson_with_report(Cursor::new(input), "$.id", &mut out)
        .expect("row-local report should run");

    assert_eq!(String::from_utf8(out).unwrap(), "1\n2\n");
    assert_eq!(report.route.kind.to_string(), "row-local");
    assert_eq!(report.route.writer_path, Some(NdjsonWriterPathKind::ByteExpr));
    assert_eq!(report.stats.rows_emitted, 2);
}

#[test]
fn run_ndjson_file_with_report_returns_file_route_stats() {
    let engine = JetroEngine::new();
    let path = temp_path("jetro-ndjson-file-report");
    std::fs::write(
        &path,
        b"{\"id\":1,\"active\":true}\n{\"id\":2,\"active\":false}\n{\"id\":3,\"active\":true}\n",
    )
    .unwrap();
    let mut out = Vec::new();

    let report = engine
        .run_ndjson_file_with_report(&path, "$.rows().filter($.active).map($.id)", &mut out)
        .expect("file-backed rows stream report should run");

    let _ = std::fs::remove_file(&path);
    assert_eq!(String::from_utf8(out).unwrap(), "1\n3\n");
    assert_eq!(report.route.source.to_string(), "file+reverse+mmap+partitionable");
    assert_eq!(report.stats.rows_scanned, 3);
    assert_eq!(report.stats.rows_emitted, 2);
}

#[test]
fn run_ndjson_source_with_report_dispatches_file_source() {
    let engine = JetroEngine::new();
    let path = temp_path("jetro-ndjson-source-report");
    std::fs::write(&path, b"{\"id\":1}\n{\"id\":2}\n").unwrap();
    let mut out = Vec::new();

    let report = engine
        .run_ndjson_source_with_report(
            NdjsonSource::file(path.clone()),
            "$.rows().take(1).map($.id)",
            &mut out,
        )
        .expect("source report should dispatch to file route");

    let _ = std::fs::remove_file(&path);
    assert_eq!(String::from_utf8(out).unwrap(), "1\n");
    assert_eq!(report.route.source.mode, jetro_core::io::NdjsonSourceMode::File);
    assert_eq!(report.stats.rows_scanned, 1);
}

#[test]
fn rows_stream_take_writes_original_rows() {
    let engine = JetroEngine::new();
    let input = br#"{"id":1,"name":"Ada"}
{"id":2,"name":"Bob"}
not-json
"#;
    let mut out = Vec::new();

    let rows = engine
        .run_ndjson(Cursor::new(input), "$.rows().take(2)", &mut out)
        .expect("rows stream take should preserve raw row output");

    assert_eq!(rows, 2);
    assert_eq!(
        String::from_utf8(out).unwrap(),
        "{\"id\":1,\"name\":\"Ada\"}\n{\"id\":2,\"name\":\"Bob\"}\n"
    );
}

#[test]
fn rows_stream_take_writes_framed_payload_not_original_record() {
    let engine = JetroEngine::new();
    let input = br#"k1|{"id":1}
k2|null
k3|{"id":3}
"#;
    let options = NdjsonOptions::default().with_row_frame(NdjsonRowFrame::DelimitedPayload {
        separator: b'|',
        null_payload: NullPayload::Skip,
    });
    let mut out = Vec::new();

    let rows = engine
        .run_ndjson_with_options(Cursor::new(input), "$.rows().take(2)", &mut out, options)
        .expect("rows stream should write framed payloads");

    assert_eq!(rows, 2);
    assert_eq!(String::from_utf8(out).unwrap(), "{\"id\":1}\n{\"id\":3}\n");
}

#[test]
fn rows_stream_first_stops_after_one_row() {
    let engine = JetroEngine::new();
    let input = br#"{"id":1}
not-json
"#;
    let mut out = Vec::new();

    let rows = engine
        .run_ndjson(Cursor::new(input), "$.rows().first().map($.id)", &mut out)
        .expect("rows stream first should stop before invalid tail");

    assert_eq!(rows, 1);
    assert_eq!(String::from_utf8(out).unwrap(), "1\n");
}

#[test]
fn rows_stream_reverse_requires_file_backed_ndjson() {
    let engine = JetroEngine::new();
    let input = br#"{"id":1}
"#;
    let mut out = Vec::new();

    let err = engine
        .run_ndjson(Cursor::new(input), "$.rows().reverse().take(1)", &mut out)
        .expect_err("reader-backed reverse rows stream should be rejected");

    assert!(err
        .to_string()
        .contains("$.rows().reverse() requires a file-backed NDJSON source"));
}

#[test]
fn rows_stream_unsupported_method_errors_before_scanning_rows() {
    let engine = JetroEngine::new();
    let input = br#"not-json
"#;
    let mut out = Vec::new();

    let err = engine
        .run_ndjson(Cursor::new(input), "$.rows().sort($.score)", &mut out)
        .expect_err("unsupported rows stream methods should fail in planning");

    assert!(err
        .to_string()
        .contains("unsupported rows() stream method sort()"));
    assert!(out.is_empty());
}

#[test]
fn rows_fanout_requires_file_backed_ndjson_before_scanning_rows() {
    let engine = JetroEngine::new();
    let input = br#"not-json
"#;
    let mut out = Vec::new();

    let err = engine
        .run_ndjson(
            Cursor::new(input),
            r#"let stream = $.rows(), a = stream.take(1), b = stream.count() in {a, b}"#,
            &mut out,
        )
        .expect_err("reader-backed rows fanout should be rejected");

    assert!(err
        .to_string()
        .contains("rows plan requires a file-backed NDJSON source"));
    assert!(out.is_empty());
}

#[test]
fn rows_fanout_with_limit_requires_file_backed_ndjson_before_scanning_rows() {
    let engine = JetroEngine::new();
    let input = br#"not-json
"#;
    let mut out = Vec::new();

    let err = engine
        .run_ndjson_limit(
            Cursor::new(input),
            r#"let stream = $.rows(), a = stream.take(1), b = stream.count() in {a, b}"#,
            1,
            &mut out,
        )
        .expect_err("reader-backed rows fanout limit should be rejected");

    assert!(err
        .to_string()
        .contains("rows plan requires a file-backed NDJSON source"));
    assert!(out.is_empty());
}

#[test]
fn rows_stream_reverse_take_map_runs_from_file_tail() {
    let engine = JetroEngine::new();
    let path = temp_path("jetro-ndjson-rows-reverse");
    std::fs::write(
        &path,
        b"not-json\n{\"id\":1,\"name\":\"Ada\"}\n{\"id\":2,\"name\":\"Bob\"}\n{\"id\":3,\"name\":\"Cid\"}\n",
    )
    .unwrap();
    let mut out = Vec::new();

    let rows = engine
        .run_ndjson_source_with_options(
            NdjsonSource::file(path.clone()),
            "$.rows().reverse().take(2).map($.id)",
            &mut out,
            NdjsonOptions::default().with_reverse_chunk_size(6),
        )
        .expect("file-backed reverse rows stream should stop before invalid head");

    let _ = std::fs::remove_file(&path);
    assert_eq!(rows, 2);
    assert_eq!(String::from_utf8(out).unwrap(), "3\n2\n");
}

#[test]
fn rows_stream_reverse_reads_delimited_payloads_and_skips_tombstones() {
    let engine = JetroEngine::new();
    let path = temp_path("jetro-ndjson-rows-reverse-framed");
    std::fs::write(&path, b"k0|null\nk1|{\"id\":1}\nk2|null\nk3|{\"id\":3}\n").unwrap();
    let options = NdjsonOptions::default()
        .with_reverse_chunk_size(5)
        .with_row_frame(NdjsonRowFrame::DelimitedPayload {
            separator: b'|',
            null_payload: NullPayload::Skip,
        });
    let mut out = Vec::new();

    let rows = engine
        .run_ndjson_file_with_options(&path, "$.rows().reverse().take(2)", &mut out, options)
        .expect("reverse rows stream should run on framed payloads");

    let _ = std::fs::remove_file(&path);
    assert_eq!(rows, 2);
    assert_eq!(String::from_utf8(out).unwrap(), "{\"id\":3}\n{\"id\":1}\n");
}

#[test]
fn rows_stream_reader_writes_scalar_sink_finish() {
    let engine = JetroEngine::new();
    let input = br#"{"active":true,"price":10}
{"active":false,"price":30}
{"active":true,"price":5}
"#;

    let mut count_out = Vec::new();
    let rows = engine
        .run_ndjson(
            Cursor::new(input),
            "$.rows().filter($.active == true).count()",
            &mut count_out,
        )
        .expect("count sink should finish");
    assert_eq!(rows, 1);
    assert_eq!(String::from_utf8(count_out).unwrap(), "2\n");

    let mut sum_out = Vec::new();
    let rows = engine
        .run_ndjson(
            Cursor::new(input),
            "$.rows().filter($.active == true).map($.price).sum()",
            &mut sum_out,
        )
        .expect("sum sink should finish");
    assert_eq!(rows, 1);
    assert_eq!(String::from_utf8(sum_out).unwrap(), "15\n");

    let mut any_out = Vec::new();
    let rows = engine
        .run_ndjson(
            Cursor::new(input),
            "$.rows().any($.price > 20)",
            &mut any_out,
        )
        .expect("any sink should finish");
    assert_eq!(rows, 1);
    assert_eq!(String::from_utf8(any_out).unwrap(), "true\n");
}

#[test]
fn rows_stream_last_retains_final_matching_row() {
    let engine = JetroEngine::new();
    let input = br#"{"active":true,"id":1}
{"active":false,"id":2}
{"active":true,"id":3}
"#;
    let mut out = Vec::new();

    let rows = engine
        .run_ndjson(
            Cursor::new(input),
            "$.rows().filter($.active == true).last()",
            &mut out,
        )
        .expect("last sink should finish");

    assert_eq!(rows, 1);
    assert_eq!(
        String::from_utf8(out).unwrap(),
        "{\"active\":true,\"id\":3}\n"
    );
}

#[test]
fn rows_stream_find_all_alias_filters_rows() {
    let engine = JetroEngine::new();
    let input = br#"{"active":true,"id":1}
{"active":false,"id":2}
{"active":true,"id":3}
"#;
    let mut out = Vec::new();

    let rows = engine
        .run_ndjson(
            Cursor::new(input),
            "$.rows().find_all($.active == true).map($.id)",
            &mut out,
        )
        .expect("find_all should lower as row filter");

    assert_eq!(rows, 2);
    assert_eq!(String::from_utf8(out).unwrap(), "1\n3\n");
}

#[test]
fn rows_stream_scalar_sink_empty_edges_are_finished() {
    let engine = JetroEngine::new();
    let empty = b"";

    let mut any_out = Vec::new();
    let rows = engine
        .run_ndjson(Cursor::new(empty), "$.rows().any($.active)", &mut any_out)
        .expect("empty any should finish");
    assert_eq!(rows, 1);
    assert_eq!(String::from_utf8(any_out).unwrap(), "false\n");

    let mut all_out = Vec::new();
    let rows = engine
        .run_ndjson(Cursor::new(empty), "$.rows().all($.active)", &mut all_out)
        .expect("empty all should finish");
    assert_eq!(rows, 1);
    assert_eq!(String::from_utf8(all_out).unwrap(), "true\n");

    let mut last_out = Vec::new();
    let rows = engine
        .run_ndjson_with_options(
            Cursor::new(
                br#"{"active":false,"id":1}
"#,
            ),
            "$.rows().filter($.active == true).last()",
            &mut last_out,
            NdjsonOptions::default().with_null_output(NdjsonNullOutput::Emit),
        )
        .expect("missing last should finish");
    assert_eq!(rows, 1);
    assert_eq!(String::from_utf8(last_out).unwrap(), "null\n");
}

#[test]
fn rows_stream_parallel_reads_delimited_payloads_and_skips_tombstones() {
    let engine = JetroEngine::new();
    let path = temp_path("jetro-ndjson-rows-parallel-framed");
    std::fs::write(
        &path,
        b"k0|null\nk1|{\"id\":1,\"active\":false}\nk2|{\"id\":2,\"active\":true}\nk3|null\nk4|{\"id\":3,\"active\":true}\n",
    )
    .unwrap();
    let options = NdjsonOptions::default()
        .with_row_frame(NdjsonRowFrame::DelimitedPayload {
            separator: b'|',
            null_payload: NullPayload::Skip,
        })
        .with_parallel_min_bytes(0);
    let mut out = Vec::new();

    let rows = engine
        .run_ndjson_file_with_options(
            &path,
            "$.rows().filter($.active).take(2).map($.id)",
            &mut out,
            options,
        )
        .expect("parallel rows stream should run on framed payloads");

    let _ = std::fs::remove_file(&path);
    assert_eq!(rows, 2);
    assert_eq!(String::from_utf8(out).unwrap(), "2\n3\n");
}

#[test]
fn rows_stream_root_find_can_use_parallel_writer() {
    let engine = JetroEngine::new();
    let path = temp_path("jetro-ndjson-rows-parallel-root-find");
    std::fs::write(
        &path,
        b"{\"id\":1,\"name\":\"old\"}\n{\"id\":2,\"name\":\"target\"}\n{\"id\":3,\"name\":\"new\"}\n",
    )
    .unwrap();
    let mut out = Vec::new();

    let rows = engine
        .run_ndjson_file_limit_with_options(
            &path,
            r#"$.rows().reverse().find($.name == "target").first()"#,
            1,
            &mut out,
            NdjsonOptions::default()
                .with_reverse_chunk_size(8)
                .with_parallel_min_bytes(0),
        )
        .expect("root rows stream find should write selected row");

    let _ = std::fs::remove_file(&path);
    assert_eq!(rows, 1);
    assert_eq!(
        String::from_utf8(out).unwrap(),
        "{\"id\":2,\"name\":\"target\"}\n"
    );
}

#[test]
fn rows_stream_root_take_can_use_parallel_writer() {
    let engine = JetroEngine::new();
    let path = temp_path("jetro-ndjson-rows-parallel-root-take");
    std::fs::write(&path, b"{\"id\":1}\n{\"id\":2}\n{\"id\":3}\n{\"id\":4}\n").unwrap();
    let mut out = Vec::new();

    let rows = engine
        .run_ndjson_file_with_options(
            &path,
            "$.rows().reverse().map($.id).take(3)",
            &mut out,
            NdjsonOptions::default()
                .with_reverse_chunk_size(8)
                .with_parallel_min_bytes(0),
        )
        .expect("root rows stream take should write retained rows");

    let _ = std::fs::remove_file(&path);
    assert_eq!(rows, 3);
    assert_eq!(String::from_utf8(out).unwrap(), "4\n3\n2\n");
}

#[test]
fn rows_stream_reverse_distinct_by_keeps_latest_rows() {
    let engine = JetroEngine::new();
    let path = temp_path("jetro-ndjson-rows-reverse-distinct");
    std::fs::write(
        &path,
        b"not-json\n{\"id\":\"a\",\"v\":1}\n{\"id\":\"b\",\"v\":2}\n{\"id\":\"a\",\"v\":3}\n{\"id\":\"c\",\"v\":4}\n",
    )
    .unwrap();
    let mut out = Vec::new();

    let rows = engine
        .run_ndjson_file_with_options(
            &path,
            "$.rows().reverse().distinct_by($.id).take(2).map({id: $.id, v: $.v})",
            &mut out,
            NdjsonOptions::default().with_reverse_chunk_size(7),
        )
        .expect("reverse rows stream should apply stream-level distinct_by");

    let _ = std::fs::remove_file(&path);
    assert_eq!(rows, 2);
    assert_eq!(
        String::from_utf8(out).unwrap(),
        "{\"id\":\"c\",\"v\":4}\n{\"id\":\"a\",\"v\":3}\n"
    );
}

#[test]
fn rows_stream_subquery_lifts_reverse_find_in_let_wrapper() {
    let engine = JetroEngine::new();
    let path = temp_path("jetro-ndjson-rows-subquery-let");
    std::fs::write(
        &path,
        b"{\"id\":1,\"name\":\"old\"}\n{\"id\":2,\"name\":\"target\"}\n{\"id\":3,\"name\":\"new\"}\n",
    )
    .unwrap();
    let mut out = Vec::new();

    let rows = engine
        .run_ndjson_file_with_options(
            &path,
            r#"let a = $.rows().reverse().find($.name == "target").first() in {id: a.id, name: a.name}"#,
            &mut out,
            NdjsonOptions::default().with_reverse_chunk_size(8),
        )
        .expect("rows stream subquery should run once and bind into wrapper");

    let _ = std::fs::remove_file(&path);
    assert_eq!(rows, 1);
    assert_eq!(
        String::from_utf8(out).unwrap(),
        "{\"id\":2,\"name\":\"target\"}\n"
    );
}

#[test]
fn rows_stream_subquery_lifts_reverse_find_in_object_wrapper() {
    let engine = JetroEngine::new();
    let path = temp_path("jetro-ndjson-rows-subquery-object");
    std::fs::write(
        &path,
        b"{\"id\":1,\"name\":\"old\"}\n{\"id\":2,\"name\":\"target\"}\n{\"id\":3,\"name\":\"new\"}\n",
    )
    .unwrap();
    let mut out = Vec::new();

    let rows = engine
        .run_ndjson_file_with_options(
            &path,
            r#"{hit: $.rows().reverse().find($.name == "target").first().id}"#,
            &mut out,
            NdjsonOptions::default().with_reverse_chunk_size(8),
        )
        .expect("rows stream subquery should bind inside object wrapper");

    let _ = std::fs::remove_file(&path);
    assert_eq!(rows, 1);
    assert_eq!(String::from_utf8(out).unwrap(), "{\"hit\":2}\n");
}

#[test]
fn rows_stream_subquery_lifts_reverse_find_in_if_wrapper() {
    let engine = JetroEngine::new();
    let path = temp_path("jetro-ndjson-rows-subquery-if");
    std::fs::write(
        &path,
        b"{\"id\":1,\"name\":\"old\"}\n{\"id\":2,\"name\":\"target\"}\n{\"id\":3,\"name\":\"new\"}\n",
    )
    .unwrap();
    let mut out = Vec::new();

    let rows = engine
        .run_ndjson_file_with_options(
            &path,
            r#""hit" if $.rows().reverse().find($.name == "target").first().id == 2 else "miss""#,
            &mut out,
            NdjsonOptions::default().with_reverse_chunk_size(8),
        )
        .expect("rows stream subquery should lift from if condition");

    let _ = std::fs::remove_file(&path);
    assert_eq!(rows, 1);
    assert_eq!(String::from_utf8(out).unwrap(), "\"hit\"\n");
}

#[test]
fn rows_stream_subquery_lifts_reverse_find_in_match_wrapper() {
    let engine = JetroEngine::new();
    let path = temp_path("jetro-ndjson-rows-subquery-match");
    std::fs::write(
        &path,
        b"{\"id\":1,\"name\":\"old\"}\n{\"id\":2,\"name\":\"target\"}\n{\"id\":3,\"name\":\"new\"}\n",
    )
    .unwrap();
    let mut out = Vec::new();

    let rows = engine
        .run_ndjson_file_with_options(
            &path,
            r#"match $.rows().reverse().find($.name == "target").first() with {
                {id: id, name: name} -> {id, name},
                _ -> null
            }"#,
            &mut out,
            NdjsonOptions::default().with_reverse_chunk_size(8),
        )
        .expect("rows stream subquery should lift from match scrutinee");

    let _ = std::fs::remove_file(&path);
    assert_eq!(rows, 1);
    assert_eq!(
        String::from_utf8(out).unwrap(),
        "{\"id\":2,\"name\":\"target\"}\n"
    );
}

#[test]
fn rows_stream_subquery_lifts_reverse_find_in_fstring_wrapper() {
    let engine = JetroEngine::new();
    let path = temp_path("jetro-ndjson-rows-subquery-fstring");
    std::fs::write(
        &path,
        b"{\"id\":1,\"name\":\"old\"}\n{\"id\":2,\"name\":\"target\"}\n{\"id\":3,\"name\":\"new\"}\n",
    )
    .unwrap();
    let mut out = Vec::new();

    let rows = engine
        .run_ndjson_file_with_options(
            &path,
            r#"f"hit {$.rows().reverse().find($.name == 'target').first().id}""#,
            &mut out,
            NdjsonOptions::default().with_reverse_chunk_size(8),
        )
        .expect("rows stream subquery should lift from f-string interpolation");

    let _ = std::fs::remove_file(&path);
    assert_eq!(rows, 1);
    assert_eq!(String::from_utf8(out).unwrap(), "\"hit 2\"\n");
}

#[test]
fn rows_stream_subquery_rejects_multiple_streams() {
    let engine = JetroEngine::new();
    let path = temp_path("jetro-ndjson-rows-subquery-multiple");
    std::fs::write(&path, b"{\"id\":1}\n{\"id\":2}\n").unwrap();
    let mut out = Vec::new();

    let err = engine
        .run_ndjson_file_with_options(
            &path,
            r#"{a: $.rows().take(1), b: $.rows().reverse().take(1)}"#,
            &mut out,
            NdjsonOptions::default().with_reverse_chunk_size(8),
        )
        .expect_err("multiple rows streams should be rejected");

    let _ = std::fs::remove_file(&path);
    assert!(err
        .to_string()
        .contains("multiple $.rows() stream subqueries are not supported"));
    assert!(out.is_empty());
}

#[test]
fn rows_stream_distinct_by_canonicalizes_direct_string_keys() {
    let engine = JetroEngine::new();
    let input = br#"{"id":"ab","v":1}
{"id":"a\u0062","v":2}
{"id":"cd","v":3}
"#;
    let mut out = Vec::new();

    let rows = engine
        .run_ndjson(
            Cursor::new(input),
            "$.rows().distinct_by($.id).map($.v)",
            &mut out,
        )
        .expect("rows stream distinct_by should canonicalize direct keys");

    assert_eq!(rows, 2);
    assert_eq!(String::from_utf8(out).unwrap(), "1\n3\n");
}

#[test]
fn rows_stream_filter_distinct_take_projects_retained_rows() {
    let engine = JetroEngine::new();
    let input = br#"{"id":"a","active":false,"v":1}
{"id":"a","active":true,"v":2}
{"id":"b","active":true,"v":3}
{"id":"a","active":true,"v":4}
not-json
"#;
    let mut out = Vec::new();

    let rows = engine
        .run_ndjson(
            Cursor::new(input),
            "$.rows().filter($.active).distinct_by($.id).take(2).map({id: $.id, v: $.v})",
            &mut out,
        )
        .expect("rows stream should stop after retained filtered distinct rows");

    assert_eq!(rows, 2);
    assert_eq!(
        String::from_utf8(out).unwrap(),
        "{\"id\":\"a\",\"v\":2}\n{\"id\":\"b\",\"v\":3}\n"
    );
}

#[test]
fn rows_stream_direct_and_fallback_distinct_keys_match() {
    let engine = JetroEngine::new();
    let input = br#"{"id":1,"v":"a"}
{"id":2,"v":"b"}
{"id":1,"v":"c"}
{"id":3,"v":"d"}
"#;
    let mut direct = Vec::new();
    let mut fallback = Vec::new();

    engine
        .run_ndjson(
            Cursor::new(input),
            "$.rows().distinct_by($.id).map($.v)",
            &mut direct,
        )
        .expect("direct key rows stream should run");
    engine
        .run_ndjson(
            Cursor::new(input),
            "$.rows().distinct_by($.id + 0).map($.v)",
            &mut fallback,
        )
        .expect("fallback key rows stream should run");

    assert_eq!(
        String::from_utf8(direct.clone()).unwrap(),
        "\"a\"\n\"b\"\n\"d\"\n"
    );
    assert_eq!(direct, fallback);
}

#[test]
fn rows_stream_source_dispatch_matches_cli_file_mode() {
    let engine = JetroEngine::new();
    let path = temp_path("jetro-ndjson-rows-cli-dispatch");
    std::fs::write(
        &path,
        b"{\"id\":\"old-a\",\"k\":\"a\",\"v\":1}\n{\"id\":\"new-b\",\"k\":\"b\",\"v\":2}\n{\"id\":\"new-a\",\"k\":\"a\",\"v\":3}\n",
    )
    .unwrap();
    let mut row_local = Vec::new();
    let mut stream = Vec::new();

    let row_local_rows = engine
        .run_ndjson_source(NdjsonSource::file(path.clone()), "$.id", &mut row_local)
        .expect("file-backed NDJSON row-local query should still run per row");
    let stream_rows = engine
        .run_ndjson_source(
            NdjsonSource::file(path.clone()),
            "$.rows().reverse().distinct_by($.k).take(2).map($.id)",
            &mut stream,
        )
        .expect("file-backed NDJSON rows() query should run as one stream");

    let _ = std::fs::remove_file(&path);
    assert_eq!(row_local_rows, 3);
    assert_eq!(
        String::from_utf8(row_local).unwrap(),
        "\"old-a\"\n\"new-b\"\n\"new-a\"\n"
    );
    assert_eq!(stream_rows, 2);
    assert_eq!(String::from_utf8(stream).unwrap(), "\"new-a\"\n\"new-b\"\n");
}

#[test]
fn run_ndjson_source_limit_dispatches_file_and_reader_inputs() {
    let engine = JetroEngine::new();
    let reader = NdjsonSource::reader(Cursor::new(b"{\"n\":1}\n{\"n\":2}\nnot-json\n".to_vec()));
    let mut reader_out = Vec::new();
    let reader_rows = engine
        .run_ndjson_source_limit(reader, "n", 1, &mut reader_out)
        .expect("reader limit should stop after one row");

    let path = temp_path("jetro-ndjson-source-limit");
    std::fs::write(&path, b"{\"n\":3}\n{\"n\":4}\nnot-json\n").unwrap();
    let mut file_out = Vec::new();
    let file_rows = engine
        .run_ndjson_source_limit_with_options(
            NdjsonSource::file(path.clone()),
            "n",
            2,
            &mut file_out,
            NdjsonOptions::default().with_reader_buffer_capacity(8),
        )
        .expect("file limit should stop before invalid tail");

    let _ = std::fs::remove_file(&path);
    assert_eq!(reader_rows, 1);
    assert_eq!(String::from_utf8(reader_out).unwrap(), "1\n");
    assert_eq!(file_rows, 2);
    assert_eq!(String::from_utf8(file_out).unwrap(), "3\n4\n");
}

#[test]
fn collect_ndjson_matches_stops_after_limit() {
    let engine = JetroEngine::new();
    let input = br#"{"name":"Ada","active":true}
{"name":"Bob","active":false}
{"name":"Cid","active":true}
not-json
"#;

    let out = engine
        .collect_ndjson_matches(Cursor::new(input), "active", 2)
        .expect("match query should stop after two matching rows");

    assert_eq!(
        out,
        vec![
            json!({"name": "Ada", "active": true}),
            json!({"name": "Cid", "active": true})
        ]
    );
}

#[test]
fn run_ndjson_matches_writes_matching_original_rows() {
    let engine = JetroEngine::new();
    let input = br#"{"name":"Ada","score":10}
{"name":"Bob","score":5}
{"name":"Cid","score":20}
"#;
    let mut out = Vec::new();

    let rows = engine
        .run_ndjson_matches(Cursor::new(input), "score > 9", 10, &mut out)
        .expect("match query should run");

    assert_eq!(rows, 2);
    assert_eq!(
        String::from_utf8(out).unwrap(),
        "{\"name\":\"Ada\",\"score\":10}\n{\"name\":\"Cid\",\"score\":20}\n"
    );
}

#[test]
fn run_ndjson_matches_writes_raw_matching_rows() {
    let engine = JetroEngine::new();
    let input = br#" { "name" : "Ada" , "score" : 10 }
{"name":"Bob","score":5}
"#;
    let mut out = Vec::new();

    let rows = engine
        .run_ndjson_matches(Cursor::new(input), "score > 9", 10, &mut out)
        .expect("match query should run");

    assert_eq!(rows, 1);
    assert_eq!(
        String::from_utf8(out).unwrap(),
        "{ \"name\" : \"Ada\" , \"score\" : 10 }\n"
    );
}

#[test]
fn file_match_helpers_stop_after_limit() {
    let engine = JetroEngine::new();
    let path = temp_path("jetro-ndjson-match-file");
    std::fs::write(
        &path,
        b"{\"name\":\"Ada\",\"active\":true}\n{\"name\":\"Bob\",\"active\":false}\n{\"name\":\"Cid\",\"active\":true}\nnot-json\n",
    )
    .unwrap();

    let out = engine
        .collect_ndjson_matches_file(&path, "active", 2)
        .expect("file match query should stop before the invalid tail");
    let mut written = Vec::new();
    let rows = engine
        .run_ndjson_matches_file_with_options(
            &path,
            "active",
            1,
            &mut written,
            NdjsonOptions::default().with_reader_buffer_capacity(64),
        )
        .expect("file match writer should stop after one match");

    let _ = std::fs::remove_file(&path);
    assert_eq!(
        out,
        vec![
            json!({"name": "Ada", "active": true}),
            json!({"name": "Cid", "active": true})
        ]
    );
    assert_eq!(rows, 1);
    assert_eq!(
        String::from_utf8(written).unwrap(),
        "{\"name\":\"Ada\",\"active\":true}\n"
    );
}

#[test]
fn source_match_helpers_dispatch_reader_and_file_inputs() {
    let engine = JetroEngine::new();
    let reader = NdjsonSource::reader(Cursor::new(
        br#"{"name":"Ada","score":10}
{"name":"Bob","score":5}
{"name":"Cid","score":20}
"#,
    ));

    let out = engine
        .collect_ndjson_matches_source(reader, "score > 9", 1)
        .expect("reader source match should run");

    let path = temp_path("jetro-ndjson-match-source-file");
    std::fs::write(
        &path,
        b"{\"name\":\"Ada\",\"score\":10}\n{\"name\":\"Bob\",\"score\":5}\n{\"name\":\"Cid\",\"score\":20}\n",
    )
    .unwrap();
    let mut written = Vec::new();
    let rows = engine
        .run_ndjson_matches_source(
            NdjsonSource::file(path.clone()),
            "score > 9",
            2,
            &mut written,
        )
        .expect("file source match should run");

    let _ = std::fs::remove_file(&path);
    assert_eq!(out, vec![json!({"name": "Ada", "score": 10})]);
    assert_eq!(rows, 2);
    assert_eq!(
        String::from_utf8(written).unwrap(),
        "{\"name\":\"Ada\",\"score\":10}\n{\"name\":\"Cid\",\"score\":20}\n"
    );
}

#[test]
fn for_each_ndjson_streams_results_to_callback() {
    let engine = JetroEngine::new();
    let input = br#"{"price":10}
{"price":20}
"#;
    let mut out = Vec::new();

    let rows = engine
        .for_each_ndjson(Cursor::new(input), "price", |value| out.push(value))
        .expect("ndjson query should run");

    assert_eq!(rows, 2);
    assert_eq!(out, vec![json!(10), json!(20)]);
}

#[test]
fn for_each_ndjson_until_stops_reading_when_callback_stops() {
    let engine = JetroEngine::new();
    let input = br#"{"price":10}
not-json
"#;
    let mut out = Vec::new();

    let rows = engine
        .for_each_ndjson_until(Cursor::new(input), "price", |value| {
            out.push(value);
            Ok(NdjsonControl::Stop)
        })
        .expect("callback stop should not read the next row");

    assert_eq!(rows, 1);
    assert_eq!(out, vec![json!(10)]);
}

#[test]
fn empty_input_returns_no_rows() {
    let engine = JetroEngine::new();

    let out = engine
        .collect_ndjson(Cursor::new(b""), "name")
        .expect("empty input should be valid");

    assert!(out.is_empty());
}

#[test]
fn single_line_without_final_newline_is_processed() {
    let engine = JetroEngine::new();

    let out = engine
        .collect_ndjson(Cursor::new(br#"{"name":"Ada"}"#), "name")
        .expect("single row should run");

    assert_eq!(out, vec![json!("Ada")]);
}

#[test]
fn invalid_json_reports_the_physical_line_number() {
    let engine = JetroEngine::new();
    let input = br#"{"ok":true}
not-json
"#;

    let err = engine
        .collect_ndjson(Cursor::new(input), "ok")
        .expect_err("invalid row should fail");

    match err {
        JetroEngineError::Ndjson(row) => {
            assert!(row.to_string().contains("line 2"), "{row}");
        }
        other => panic!("expected row error, got {other:?}"),
    }
}

#[test]
fn options_enforce_max_line_length_after_newline_trim() {
    let engine = JetroEngine::new();
    let input = br#"{"name":"Ada"}
"#;

    let err = engine
        .collect_ndjson_with_options(
            Cursor::new(input),
            "name",
            NdjsonOptions::default().with_max_line_len(4),
        )
        .expect_err("long row should fail");

    match err {
        JetroEngineError::Ndjson(row) => {
            assert!(
                row.to_string().contains("line 1") && row.to_string().contains("too large"),
                "{row}"
            );
        }
        other => panic!("expected row error, got {other:?}"),
    }
}

#[test]
fn crlf_and_trailing_newline_less_rows_are_supported() {
    let engine = JetroEngine::new();
    let input = b"{\"name\":\"Ada\"}\r\n{\"name\":\"Bob\"}";

    let out = engine
        .collect_ndjson(Cursor::new(input), "name")
        .expect("ndjson query should run");

    assert_eq!(out, vec![json!("Ada"), json!("Bob")]);
}

#[test]
fn utf8_bom_is_ignored_only_on_the_first_physical_line() {
    let engine = JetroEngine::new();
    let input = b"\xEF\xBB\xBF{\"name\":\"Ada\"}\n{\"name\":\"Bob\"}\n";

    let out = engine
        .collect_ndjson(Cursor::new(input), "name")
        .expect("ndjson query should run");

    assert_eq!(out, vec![json!("Ada"), json!("Bob")]);
}

#[test]
fn file_helpers_use_the_same_per_row_execution() {
    let engine = JetroEngine::new();
    let path = temp_path("jetro-ndjson-basic");
    std::fs::write(&path, b"{\"name\":\"Ada\"}\n{\"name\":\"Bob\"}\n").unwrap();

    let out = engine
        .collect_ndjson_file(&path, "name")
        .expect("file query should run");
    let out_with_options = engine
        .collect_ndjson_file_with_options(
            &path,
            "name",
            NdjsonOptions::default().with_initial_buffer_capacity(64),
        )
        .expect("file query should run");
    let mut written = Vec::new();
    let rows = engine
        .run_ndjson_file_with_options(
            &path,
            "name",
            &mut written,
            NdjsonOptions::default()
                .with_initial_buffer_capacity(64)
                .with_reader_buffer_capacity(64),
        )
        .expect("file query should run");

    let _ = std::fs::remove_file(&path);
    assert_eq!(out, vec![json!("Ada"), json!("Bob")]);
    assert_eq!(out_with_options, out);
    assert_eq!(rows, 2);
    assert_eq!(String::from_utf8(written).unwrap(), "\"Ada\"\n\"Bob\"\n");
}

#[test]
fn source_helpers_dispatch_reader_and_file_inputs() {
    let engine = JetroEngine::new();
    let reader = NdjsonSource::reader(Cursor::new(
        br#"{"name":"Ada"}
{"name":"Bob"}
"#,
    ));

    let out = engine
        .collect_ndjson_source(reader, "name")
        .expect("source query should run");
    let mut callback_out = Vec::new();
    let callback_rows = engine
        .for_each_ndjson_source(
            NdjsonSource::reader(Cursor::new(
                br#"{"name":"Ada"}
{"name":"Bob"}
"#,
            )),
            "name",
            |value| callback_out.push(value),
        )
        .expect("source query should run");

    let path = temp_path("jetro-ndjson-source-file");
    std::fs::write(&path, b"{\"score\":2}\n{\"score\":3}\n").unwrap();
    let mut written = Vec::new();
    let rows = engine
        .run_ndjson_source(NdjsonSource::file(path.clone()), "score", &mut written)
        .expect("source query should run");

    let _ = std::fs::remove_file(&path);
    assert_eq!(out, vec![json!("Ada"), json!("Bob")]);
    assert_eq!(callback_rows, 2);
    assert_eq!(callback_out, out);
    assert_eq!(rows, 2);
    assert_eq!(String::from_utf8(written).unwrap(), "2\n3\n");
}

#[test]
fn reverse_file_helpers_evaluate_rows_from_tail() {
    let engine = JetroEngine::new();
    let path = temp_path("jetro-ndjson-rev-api");
    std::fs::write(&path, b"{\"name\":\"Ada\"}\n{\"name\":\"Bob\"}\n").unwrap();

    let out = engine
        .collect_ndjson_rev(&path, "name")
        .expect("reverse file query should run");
    let out_with_options = engine
        .collect_ndjson_rev_with_options(
            &path,
            "name",
            NdjsonOptions::default().with_reverse_chunk_size(5),
        )
        .expect("reverse file query should run");
    let mut written = Vec::new();
    let rows = engine
        .run_ndjson_rev_with_options(
            &path,
            "name",
            &mut written,
            NdjsonOptions::default().with_reverse_chunk_size(5),
        )
        .expect("reverse file query should run");

    let _ = std::fs::remove_file(&path);
    assert_eq!(out, vec![json!("Bob"), json!("Ada")]);
    assert_eq!(out_with_options, out);
    assert_eq!(rows, 2);
    assert_eq!(String::from_utf8(written).unwrap(), "\"Bob\"\n\"Ada\"\n");
}

#[test]
fn reverse_run_limit_writes_from_tail_and_stops() {
    let engine = JetroEngine::new();
    let path = temp_path("jetro-ndjson-rev-run-limit");
    std::fs::write(&path, b"not-json\n{\"name\":\"Ada\"}\n{\"name\":\"Bob\"}\n").unwrap();
    let mut out = Vec::new();

    let rows = engine
        .run_ndjson_rev_limit_with_options(
            &path,
            "name",
            2,
            &mut out,
            NdjsonOptions::default().with_reverse_chunk_size(5),
        )
        .expect("reverse writer limit should stop before invalid head");

    let _ = std::fs::remove_file(&path);
    assert_eq!(rows, 2);
    assert_eq!(String::from_utf8(out).unwrap(), "\"Bob\"\n\"Ada\"\n");
}

#[test]
fn reverse_distinct_by_keeps_first_key_seen_from_tail() {
    let engine = JetroEngine::new();
    let path = temp_path("jetro-ndjson-rev-distinct-by");
    std::fs::write(
        &path,
        b"{\"id\":\"a\",\"v\":1}\n{\"id\":\"b\",\"v\":2}\n{\"id\":\"c\",\"v\":3}\n{\"id\":\"a\",\"v\":4}\n{\"id\":\"b\",\"v\":5}\n",
    )
    .unwrap();
    let mut out = Vec::new();

    let emitted = engine
        .run_ndjson_rev_distinct_by_with_options(
            &path,
            "id",
            r#"{id: $.id, v: $.v}"#,
            10,
            &mut out,
            NdjsonOptions::default().with_reverse_chunk_size(8),
        )
        .expect("reverse distinct_by should run");

    let _ = std::fs::remove_file(&path);
    assert_eq!(emitted, 3);
    assert_eq!(
        String::from_utf8(out).unwrap(),
        "{\"id\":\"b\",\"v\":5}\n{\"id\":\"a\",\"v\":4}\n{\"id\":\"c\",\"v\":3}\n"
    );
}

#[test]
fn reverse_distinct_by_limit_stops_before_old_invalid_rows() {
    let engine = JetroEngine::new();
    let path = temp_path("jetro-ndjson-rev-distinct-by-limit");
    std::fs::write(
        &path,
        b"not-json\n{\"id\":\"a\",\"v\":1}\n{\"id\":\"b\",\"v\":2}\n{\"id\":\"a\",\"v\":3}\n",
    )
    .unwrap();
    let mut out = Vec::new();

    let emitted = engine
        .run_ndjson_rev_distinct_by(&path, "id", "v", 2, &mut out)
        .expect("reverse distinct_by should stop after demanded unique rows");

    let _ = std::fs::remove_file(&path);
    assert_eq!(emitted, 2);
    assert_eq!(String::from_utf8(out).unwrap(), "3\n2\n");
}

#[test]
fn reverse_distinct_by_stats_report_fast_paths_and_duplicates() {
    let engine = JetroEngine::new();
    let path = temp_path("jetro-ndjson-rev-distinct-by-stats");
    std::fs::write(
        &path,
        b"{\"id\":\"a\",\"v\":1}\n{\"id\":\"b\",\"v\":2}\n{\"id\":\"a\",\"v\":3}\n",
    )
    .unwrap();
    let mut out = Vec::new();

    let stats = engine
        .run_ndjson_rev_distinct_by_with_stats(&path, "id", "v", 10, &mut out)
        .expect("reverse distinct_by stats should run");

    let _ = std::fs::remove_file(&path);
    assert_eq!(String::from_utf8(out).unwrap(), "3\n2\n");
    assert_eq!(stats.rows_scanned, 3);
    assert_eq!(stats.emitted, 2);
    assert_eq!(stats.duplicate_rows, 1);
    assert_eq!(stats.direct_key_rows, 3);
    assert_eq!(stats.fallback_key_rows, 0);
    assert_eq!(stats.direct_value_rows, 2);
    assert_eq!(stats.fallback_value_rows, 0);
    assert_eq!(stats.front_filter, DistinctFrontFilterKind::None);
}

#[test]
fn reverse_distinct_by_stats_report_front_filter_activation() {
    let engine = JetroEngine::new();
    let path = temp_path("jetro-ndjson-rev-distinct-by-front-filter");
    let mut data = Vec::new();
    for id in 0..5000 {
        data.extend_from_slice(format!(r#"{{"id":{id},"v":{id}}}"#).as_bytes());
        data.push(b'\n');
    }
    std::fs::write(&path, data).unwrap();
    let mut out = Vec::new();

    let stats = engine
        .run_ndjson_rev_distinct_by_with_stats(&path, "id", "v", 5000, &mut out)
        .expect("reverse distinct_by stats should expose front filter activation");

    let _ = std::fs::remove_file(&path);
    assert_eq!(stats.emitted, 5000);
    assert_eq!(stats.duplicate_rows, 0);
    assert_eq!(stats.front_filter, DistinctFrontFilterKind::Cuckoo);
}

#[test]
fn reverse_distinct_by_canonicalizes_escaped_string_keys_directly() {
    let engine = JetroEngine::new();
    let path = temp_path("jetro-ndjson-rev-distinct-by-escaped");
    std::fs::write(
        &path,
        b"{\"id\":\"a\\u0062\",\"v\":1}\n{\"id\":\"ab\",\"v\":2}\n",
    )
    .unwrap();
    let mut out = Vec::new();

    let stats = engine
        .run_ndjson_rev_distinct_by_with_stats(&path, "id", "v", 10, &mut out)
        .expect("reverse distinct_by should canonicalize escaped string keys");

    let _ = std::fs::remove_file(&path);
    assert_eq!(String::from_utf8(out).unwrap(), "2\n");
    assert_eq!(stats.emitted, 1);
    assert_eq!(stats.duplicate_rows, 1);
    assert_eq!(stats.direct_key_rows, 2);
    assert_eq!(stats.fallback_key_rows, 0);
}

#[test]
fn reverse_distinct_by_canonicalizes_compound_keys_directly() {
    let engine = JetroEngine::new();
    let path = temp_path("jetro-ndjson-rev-distinct-by-compound");
    std::fs::write(
        &path,
        b"{\"k\":{\"a\" : 1,\"b\":\"x\\u0079\"},\"v\":1}\n{\"k\":{\"a\":1,\"b\":\"xy\"},\"v\":2}\n",
    )
    .unwrap();
    let mut out = Vec::new();

    let stats = engine
        .run_ndjson_rev_distinct_by_with_stats(&path, "k", "v", 10, &mut out)
        .expect("reverse distinct_by should canonicalize compound keys");

    let _ = std::fs::remove_file(&path);
    assert_eq!(String::from_utf8(out).unwrap(), "2\n");
    assert_eq!(stats.emitted, 1);
    assert_eq!(stats.duplicate_rows, 1);
    assert_eq!(stats.direct_key_rows, 2);
    assert_eq!(stats.fallback_key_rows, 0);
}

#[test]
fn reverse_for_each_until_stops_before_head_rows() {
    let engine = JetroEngine::new();
    let path = temp_path("jetro-ndjson-rev-until");
    std::fs::write(&path, b"not-json\n{\"name\":\"Ada\"}\n{\"name\":\"Bob\"}\n").unwrap();
    let mut out = Vec::new();

    let rows = engine
        .for_each_ndjson_rev_until(&path, "name", |value| {
            out.push(value);
            Ok(NdjsonControl::Stop)
        })
        .expect("reverse callback stop should not read the invalid head row");

    let _ = std::fs::remove_file(&path);
    assert_eq!(rows, 1);
    assert_eq!(out, vec![json!("Bob")]);
}

#[test]
fn reverse_for_each_helpers_stream_query_results() {
    let engine = JetroEngine::new();
    let path = temp_path("jetro-ndjson-rev-callback");
    std::fs::write(&path, b"{\"n\":1}\n{\"n\":2}\n{\"n\":3}\n").unwrap();
    let mut out = Vec::new();

    let rows = engine
        .for_each_ndjson_rev_with_options(
            &path,
            "n + 1",
            NdjsonOptions::default().with_reverse_chunk_size(4),
            |value| out.push(value),
        )
        .expect("reverse callback query should run");

    let _ = std::fs::remove_file(&path);
    assert_eq!(rows, 3);
    assert_eq!(out, vec![json!(4), json!(3), json!(2)]);
}

#[test]
fn reverse_match_helpers_stop_from_tail() {
    let engine = JetroEngine::new();
    let path = temp_path("jetro-ndjson-rev-match");
    std::fs::write(
        &path,
        b"{\"name\":\"Ada\",\"active\":true}\n{\"name\":\"Bob\",\"active\":false}\n{\"name\":\"Cid\",\"active\":true}\n{\"name\":\"Dia\",\"active\":true}\nnot-json\n",
    )
    .unwrap();

    let err = engine
        .collect_ndjson_rev_matches(&path, "active", 2)
        .expect_err("reverse match starts at the invalid tail");
    match err {
        JetroEngineError::Ndjson(row) => {
            assert!(row.to_string().contains("line 1"), "{row}");
        }
        other => panic!("expected row error, got {other:?}"),
    }

    std::fs::write(
        &path,
        b"{\"name\":\"Ada\",\"active\":true}\n{\"name\":\"Bob\",\"active\":false}\n{\"name\":\"Cid\",\"active\":true}\n{\"name\":\"Dia\",\"active\":true}\n",
    )
    .unwrap();
    let out = engine
        .collect_ndjson_rev_matches_with_options(
            &path,
            "active",
            2,
            NdjsonOptions::default().with_reverse_chunk_size(7),
        )
        .expect("reverse match query should run");
    let mut written = Vec::new();
    let rows = engine
        .run_ndjson_rev_matches(&path, "active", 1, &mut written)
        .expect("reverse match writer should stop after one match");

    let _ = std::fs::remove_file(&path);
    assert_eq!(
        out,
        vec![
            json!({"name": "Dia", "active": true}),
            json!({"name": "Cid", "active": true})
        ]
    );
    assert_eq!(rows, 1);
    assert_eq!(
        String::from_utf8(written).unwrap(),
        "{\"name\":\"Dia\",\"active\":true}\n"
    );
}

#[test]
fn reverse_match_writer_preserves_raw_matching_rows() {
    let engine = JetroEngine::new();
    let path = temp_path("jetro-ndjson-rev-match-raw");
    std::fs::write(
        &path,
        b" { \"name\" : \"Ada\" , \"active\" : true }\n{\"name\":\"Bob\",\"active\":false}\n { \"name\" : \"Cid\" , \"active\" : true }\n",
    )
    .unwrap();
    let mut out = Vec::new();

    let rows = engine
        .run_ndjson_rev_matches(&path, "active", 2, &mut out)
        .expect("reverse match writer should run");

    let _ = std::fs::remove_file(&path);
    assert_eq!(rows, 2);
    assert_eq!(
        String::from_utf8(out).unwrap(),
        "{ \"name\" : \"Cid\" , \"active\" : true }\n{ \"name\" : \"Ada\" , \"active\" : true }\n"
    );
}

#[test]
fn reverse_options_enforce_max_line_length() {
    let engine = JetroEngine::new();
    let path = temp_path("jetro-ndjson-rev-max-line");
    std::fs::write(&path, b"{\"name\":\"Ada\"}\n").unwrap();

    let err = engine
        .collect_ndjson_rev_with_options(
            &path,
            "name",
            NdjsonOptions::default()
                .with_reverse_chunk_size(4)
                .with_max_line_len(4),
        )
        .expect_err("long row should fail");

    let _ = std::fs::remove_file(&path);
    match err {
        JetroEngineError::Ndjson(row) => {
            assert!(row.to_string().contains("too large"), "{row}");
        }
        other => panic!("expected row error, got {other:?}"),
    }
}

fn temp_path(name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!("{}-{}.ndjson", name, std::process::id()));
    path
}
