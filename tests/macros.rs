#![cfg(feature = "macros")]

use jetro::prelude::*;
use jetro::{jetro, JetroSchema};

#[derive(JetroSchema)]
#[expr(titles = "$.books.map(title)")]
#[expr(count  = "$.books.len()")]
struct BookView;

#[test]
fn schema_exposes_pairs() {
    let pairs = BookView::exprs();
    assert_eq!(pairs.len(), 2);
    assert_eq!(pairs[0], ("titles", "$.books.map(title)"));
    assert_eq!(pairs[1], ("count", "$.books.len()"));
    assert_eq!(BookView::names(), &["titles", "count"]);
}

#[test]
fn schema_wires_into_bucket() {
    let db = Database::memory().unwrap();
    let mut b = db.bucket("books");
    for (name, src) in BookView::exprs() {
        b = b.with(*name, *src);
    }
    let bucket = b.open().unwrap();
    bucket
        .insert("cat", &json!({"books": [
            {"title": "Dune"}, {"title": "Foundation"}
        ]}))
        .unwrap();
    let titles: Vec<String> = bucket.get("cat", "titles").unwrap().unwrap();
    let count: usize = bucket.get("cat", "count").unwrap().unwrap();
    assert_eq!(titles, vec!["Dune", "Foundation"]);
    assert_eq!(count, 2);
}

#[test]
fn jetro_macro_compiles_and_evaluates() {
    let e = jetro!("$.x + 1");
    let doc = json!({"x": 41});
    assert_eq!(e.eval_raw(&doc).unwrap(), json!(42));
}

#[test]
fn jetro_macro_nested() {
    let e = jetro!(r#"$.books.filter(price > 10).map(title)"#);
    let doc = json!({"books": [
        {"title": "Dune", "price": 12}, {"title": "Foundation", "price": 9}
    ]});
    assert_eq!(e.eval_raw(&doc).unwrap(), json!(["Dune"]));
}

#[test]
fn typed_row_roundtrip() {
    use jetro::Row;
    let db = Database::memory().unwrap();
    let row: Row<BookView> = db.row("library").unwrap();
    row.insert(
        "cat",
        &json!({"books": [{"title": "Dune"}, {"title": "Foundation"}]}),
    )
    .unwrap();
    let titles: Vec<String> = row.get("cat", "titles").unwrap().unwrap();
    let count: u64 = row.get("cat", "count").unwrap().unwrap();
    assert_eq!(titles, vec!["Dune", "Foundation"]);
    assert_eq!(count, 2);
}

#[test]
fn session_row_roundtrip() {
    use jetro::Session;
    let s = Session::in_memory();
    let row = s.row::<BookView>("library").unwrap();
    row.insert(
        "cat",
        &json!({"books": [{"title": "Dune"}, {"title": "Foundation"}]}),
    )
    .unwrap();
    let count: u64 = row.get("cat", "count").unwrap().unwrap();
    assert_eq!(count, 2);
}

#[test]
fn jetro_macro_let_and_pipeline() {
    // Exercises constructs that a lexical delimiter-only check could never
    // validate: `let` binding + pipeline operator.  If the macro still type-checks,
    // the compile-time grammar check is wired up.
    let e = jetro!("let n = $.nums in n.sum()");
    let doc = json!({"nums": [1, 2, 3, 4]});
    assert_eq!(e.eval_raw(&doc).unwrap(), json!(10));
}
