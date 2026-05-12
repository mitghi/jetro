use jetro::{io::NdjsonSource, JetroEngine};
use serde_json::json;
use std::io::Cursor;

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
