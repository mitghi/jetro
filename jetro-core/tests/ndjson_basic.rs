use jetro_core::io::{NdjsonControl, NdjsonOptions, NdjsonSource};
use jetro_core::{JetroEngine, JetroEngineError};
use serde_json::json;
use std::io::Cursor;
use std::path::PathBuf;

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
fn run_ndjson_writes_scalar_results_directly() {
    let engine = JetroEngine::new();
    let input = b"{\"s\":\"a\\\"b\\\\c\\n\",\"b\":true,\"z\":null,\"f\":1.25}\n";

    let mut string_out = Vec::new();
    engine
        .run_ndjson(Cursor::new(input), "s", &mut string_out)
        .expect("string scalar should write");
    assert_eq!(String::from_utf8(string_out).unwrap(), "\"a\\\"b\\\\c\\n\"\n");

    let mut bool_out = Vec::new();
    engine
        .run_ndjson(Cursor::new(input), "b", &mut bool_out)
        .expect("bool scalar should write");
    assert_eq!(String::from_utf8(bool_out).unwrap(), "true\n");

    let mut null_out = Vec::new();
    engine
        .run_ndjson(Cursor::new(input), "z", &mut null_out)
        .expect("null scalar should write");
    assert_eq!(String::from_utf8(null_out).unwrap(), "null\n");

    let mut float_out = Vec::new();
    engine
        .run_ndjson(Cursor::new(input), "f", &mut float_out)
        .expect("float scalar should write");
    assert_eq!(String::from_utf8(float_out).unwrap(), "1.25\n");
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
        " { \"name\" : \"Ada\" , \"score\" : 10 }\n"
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
