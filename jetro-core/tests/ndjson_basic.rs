use jetro_core::io::NdjsonOptions;
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

fn temp_path(name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!("{}-{}.ndjson", name, std::process::id()));
    path
}
