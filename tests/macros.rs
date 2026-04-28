#![cfg(feature = "macros")]

use jetro::prelude::*;
use jetro::{jetro, JetroSchema};

#[derive(JetroSchema)]
#[expr(titles = "$.books.map(title)")]
#[expr(count = "$.books.len()")]
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
fn jetro_macro_let_and_pipeline() {
    // let + pipeline — non-lexical constructs that require real PEG parse
    let e = jetro!("let n = $.nums in n.sum()");
    let doc = json!({"nums": [1, 2, 3, 4]});
    assert_eq!(e.eval_raw(&doc).unwrap(), json!(10));
}
