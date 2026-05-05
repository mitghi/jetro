use jetro_core::Jetro;
use serde_json::{json, Value};

fn j(document: Value) -> Jetro {
    Jetro::from_bytes(serde_json::to_vec(&document).unwrap()).unwrap()
}

#[test]
fn kvplan_object_shaping_pipeline_ir() {
    let doc = json!({
        "books": [
            {"title": "a", "price": 50},
            {"title": "b", "price": 150},
            {"title": "c", "price": 200}
        ],
        "name": {"first": "alice"}
    });
    let q = r#"{expensive: $.books.filter(@.price > 100).map(@.title), first_name: $.name.first}"#;
    let out = j(doc.into()).collect(q).unwrap();
    let got: Value = out.into();
    assert_eq!(
        got,
        json!({
            "expensive": ["b", "c"],
            "first_name": "alice"
        })
    );
}

#[test]
fn kvplan_take_demand_propagates() {
    let doc = json!({
        "items": (0..1000).map(|i| json!({"v": i})).collect::<Vec<_>>()
    });
    let q = r#"{first_two: $.items.filter(@.v > 100).map(@.v).take(2)}"#;
    let out = j(doc.into()).collect(q).unwrap();
    let got: Value = out.into();
    assert_eq!(got, json!({"first_two": [101, 102]}));
}
