use jetro::JetroEngine;
use std::io::Cursor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = JetroEngine::new();
    let rows = Cursor::new(
        br#"{"id":1,"level":"info","msg":"started"}
{"id":2,"level":"error","msg":"failed"}
{"id":3,"level":"error","msg":"retry failed"}
{"id":4,"level":"error","msg":"unread when limit is reached"}
"#,
    );

    let matches = engine.collect_ndjson_matches(rows, r#"level == "error""#, 2)?;
    println!("{}", serde_json::to_string_pretty(&matches)?);
    Ok(())
}
