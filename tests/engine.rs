use jetro::prelude::*;

#[test]
fn jetro_collect_basic() {
    let j = Jetro::new(json!({
        "store": {
            "books": [
                {"title": "Dune",       "price": 12.99},
                {"title": "Foundation", "price":  9.99}
            ]
        }
    }));
    assert_eq!(j.collect("$.store.books.len()").unwrap(), json!(2));
    assert_eq!(
        j.collect("$.store.books.filter(price > 10).map(title)").unwrap(),
        json!(["Dune"])
    );
}

#[test]
fn query_free_function() {
    let doc = json!({"x": 41});
    assert_eq!(jetro::query("$.x + 1", &doc).unwrap(), json!(42));
}

#[test]
fn parse_error_surfaces() {
    let doc = json!({});
    match jetro::query("$.", &doc) {
        Err(jetro::Error::Parse(_)) => {}
        other => panic!("expected Parse error, got {:?}", other),
    }
}
