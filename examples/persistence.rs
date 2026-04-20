//! Shows data survives process restart: write in one Database::open, read in another.
//!
//!   cargo run --example persistence

use jetro::db::Database;
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Use a fixed path so data outlives the process (unlike TempDir).
    let dir = std::env::temp_dir().join("jetro_persistence_demo");
    std::fs::create_dir_all(&dir)?;

    println!("Database at: {}", dir.display());

    // ── first open: write ────────────────────────────────────────────────────
    {
        let db = Database::open(&dir)?;
        let exprs = db.expr_bucket("exprs")?;
        exprs.put("top_scorers", ">/players/#filter('score' > 80)/#map(@/name)")?;

        let docs = db.json_bucket("game", &["top_scorers"], &exprs)?;
        docs.insert("round:1", &json!({
            "players": [
                {"name": "Alice", "score": 95},
                {"name": "Bob",   "score": 72},
                {"name": "Carol", "score": 88}
            ]
        }))?;
        println!("Wrote round:1 — closing database.");
    }
    // Database dropped; file handles closed.

    // ── second open: read (simulates process restart) ────────────────────────
    {
        let db = Database::open(&dir)?;
        let exprs = db.expr_bucket("exprs")?;

        // Expression persisted
        let expr = exprs.get("top_scorers")?.expect("expression should persist");
        println!("Re-read expression: {expr}");

        let docs = db.json_bucket("game", &["top_scorers"], &exprs)?;
        let top = docs.get_result("round:1", "top_scorers")?.expect("result should persist");
        println!("Top scorers after restart: {}", top[0]);
        assert_eq!(top[0], json!(["Alice", "Carol"]));
    }

    println!("Persistence verified.");

    // clean up
    std::fs::remove_dir_all(&dir)?;
    Ok(())
}
