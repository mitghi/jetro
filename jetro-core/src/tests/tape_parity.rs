//! Parity coverage between the byte/tape engine and the VM fallback for
//! demand-sensitive pipeline shapes.

use super::common::vm_query;
use serde_json::{json, Value};

fn assert_tape_vm_eq(query: &str, doc: &Value) {
    let expected = vm_query(query, doc).unwrap_or_else(|err| panic!("VM failed for {query}: {err}"));
    let bytes = serde_json::to_vec(doc).expect("serialize fixture");
    let jetro = crate::Jetro::from_bytes(bytes).expect("build byte engine");
    let actual: Value = jetro
        .collect(query)
        .unwrap_or_else(|err| panic!("tape engine failed for {query}: {err}"));
    assert_eq!(actual, expected, "{query}");
}

#[test]
fn tape_matches_vm_for_borrowed_object_helpers() {
    let doc = json!({
        "profile": {
            "id": 7,
            "name": "Ada",
            "active": true,
            "nested": {"score": 42}
        }
    });
    for query in [
        "$.profile.keys()",
        "$.profile.values()",
        "$.profile.entries()",
        "$.profile.entries().first()",
        "$.profile.entries().last()",
        "$.profile.entries().map(e => e[0])",
        "$.profile.entries().map(e => e[1]).last()",
        "$.profile.pick(\"id\", \"name\")",
        "$.profile.omit(\"active\")",
        "$.profile.has_key(\"nested\")",
    ] {
        assert_tape_vm_eq(query, &doc);
    }
}

#[test]
fn tape_matches_vm_for_bounded_projection_and_reducers() {
    let doc = json!({
        "books": [
            {"id": 1, "title": "Dune", "price": 12.5, "tags": ["sf", "classic"]},
            {"id": 2, "title": "Foundation", "price": 9.0, "tags": ["sf"]},
            {"id": 3, "title": "Hyperion", "price": 14.0, "tags": ["sf", "hugo"]},
            {"id": 4, "title": "Snow Crash", "price": 11.0, "tags": ["sf", "cyberpunk"]}
        ]
    });
    for query in [
        "$.books.map({id, title}).first()",
        "$.books.map({id, title}).last()",
        "$.books.map({id, title}).take(2)",
        "$.books.filter(price > 10).map({id, title}).last()",
        "$.books.filter(tags.includes(\"hugo\")).map(price).sum()",
        "$.books.map(price).avg()",
        "$.books.map(price).min()",
        "$.books.map(price).max()",
        "$.books.count(price > 10)",
    ] {
        assert_tape_vm_eq(query, &doc);
    }
}

#[test]
fn tape_matches_vm_for_scalar_methods_inside_nested_projection() {
    let doc = json!({
        "users": [
            {
                "name": "ada",
                "email": "ada@example.test",
                "score": 10,
                "profile": {"role": "admin", "tier": 2}
            },
            {
                "name": "bob",
                "email": "bob@example.test",
                "score": 20,
                "profile": {"role": "user", "tier": 1}
            }
        ]
    });
    for query in [
        "$.users.map({name: name.upper(), domain: email.split(\"@\").last()})",
        "$.users.map({label: f\"{name}:{score}\", name_len: name.len()}).last()",
        "$.users.filter(name.starts_with(\"a\")).map(email.contains(\"@\"))",
        "$.users.map(profile.entries().first())",
        "$.users.map(profile.entries().map(e => e[0]).last())",
    ] {
        assert_tape_vm_eq(query, &doc);
    }
}
