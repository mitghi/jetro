//! Basic store: store expressions, insert documents, retrieve results.
//!
//!   cargo run --example basic_store

use jetro::db::Database;
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tmp = tempfile::tempdir()?;
    let db = Database::open(tmp.path())?;

    // --- expression bucket ---------------------------------------------------
    let exprs = db.expr_bucket("main")?;

    exprs.put("book_titles",  ">/store/books/#map(@/title)")?;
    exprs.put("book_count",   ">/store/books/#len")?;
    exprs.put("cheap_books",  ">/store/books/#filter('price' < 15)/#map(@/title)")?;

    println!("Stored expressions:");
    for (k, v) in exprs.all()? {
        println!("  {k:20} => {v}");
    }

    // --- json bucket ---------------------------------------------------------
    let docs = db.json_bucket("library", &["book_titles", "book_count", "cheap_books"], &exprs)?;

    let catalog = json!({
        "store": {
            "books": [
                {"title": "Dune",             "author": "Herbert",  "price": 12.99},
                {"title": "Foundation",       "author": "Asimov",   "price":  9.99},
                {"title": "Neuromancer",      "author": "Gibson",   "price": 18.50},
                {"title": "Snow Crash",       "author": "Stephenson","price": 14.00}
            ]
        }
    });
    docs.insert("catalog_2024", &catalog)?;

    // --- retrieve results ----------------------------------------------------
    let titles = docs.get_result("catalog_2024", "book_titles")?.unwrap();
    println!("\nAll titles:  {}", titles[0]);

    let count = docs.get_result("catalog_2024", "book_count")?.unwrap();
    println!("Book count:  {}", count[0]);

    let cheap = docs.get_result("catalog_2024", "cheap_books")?.unwrap();
    println!("Under $15:   {}", cheap[0]);

    // --- original document survives ------------------------------------------
    let original = docs.get_doc("catalog_2024")?.unwrap();
    assert_eq!(original["store"]["books"][0]["title"], "Dune");
    println!("\nOriginal doc round-trips correctly.");

    Ok(())
}
