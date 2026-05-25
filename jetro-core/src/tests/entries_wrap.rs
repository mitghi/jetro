//! Coverage for the `entries`/`keys`/`values` triple-wrap fix.
//!
//! Before the fix, `object_element_spec` marked these methods as
//! `.element()` which made the streaming pipeline treat the object
//! receiver as a 1-element stream. The builtin returned `Val::Arr([pairs])`
//! correctly, but the stream-sink wrapped it again into
//! `Val::Arr([Val::Arr([pairs])])` — surface result `[[pairs]]`.
//!
//! The fix removes `.element()` from the spec for these whole-object
//! transforms; they take a single object and produce a single array.
//!
//! Tests cover bare object input, path-rooted input, and chained pipelines
//! that depend on entries' canonical shape (the common
//! `.group_by().entries()` idiom in particular).

use super::common::vm_query;
use serde_json::json;

#[test]
fn entries_on_path_object() {
    let doc = json!({"o": {"a": 1, "b": 2, "c": 3}});
    let out = vm_query("$.o.entries()", &doc).unwrap();
    assert_eq!(out, json!([["a", 1], ["b", 2], ["c", 3]]));
}

#[test]
fn entries_count_returns_pair_count() {
    // Pre-fix this returned 1 (the outer wrap had a single element).
    let doc = json!({"o": {"a": 1, "b": 2, "c": 3}});
    let out = vm_query("$.o.entries().count()", &doc).unwrap();
    assert_eq!(out, json!(3));
}

#[test]
fn entries_first_returns_pair() {
    let doc = json!({"o": {"x": 10, "y": 20}});
    let out = vm_query("$.o.entries().first()", &doc).unwrap();
    assert_eq!(out, json!(["x", 10]));
}

#[test]
fn entries_map_pick_key() {
    let doc = json!({"o": {"a": 1, "b": 2}});
    let out = vm_query("$.o.entries().map(e => e[0])", &doc).unwrap();
    assert_eq!(out, json!(["a", "b"]));
}

#[test]
fn entries_map_pick_value() {
    let doc = json!({"o": {"a": 1, "b": 2}});
    let out = vm_query("$.o.entries().map(e => e[1])", &doc).unwrap();
    assert_eq!(out, json!([1, 2]));
}

#[test]
fn entries_with_destructure() {
    let doc = json!({"o": {"a": 1, "b": 2}});
    let out = vm_query("$.o.entries().map(([k, v]) => v)", &doc).unwrap();
    assert_eq!(out, json!([1, 2]));
}

#[test]
fn keys_on_path_object() {
    let doc = json!({"o": {"a": 1, "b": 2, "c": 3}});
    let out = vm_query("$.o.keys()", &doc).unwrap();
    assert_eq!(out, json!(["a", "b", "c"]));
}

#[test]
fn keys_count() {
    let doc = json!({"o": {"a": 1, "b": 2, "c": 3}});
    let out = vm_query("$.o.keys().count()", &doc).unwrap();
    assert_eq!(out, json!(3));
}

#[test]
fn values_on_path_object() {
    let doc = json!({"o": {"a": 1, "b": 2, "c": 3}});
    let out = vm_query("$.o.values()", &doc).unwrap();
    assert_eq!(out, json!([1, 2, 3]));
}

#[test]
fn values_sum() {
    let doc = json!({"o": {"a": 10, "b": 20, "c": 30}});
    let out = vm_query("$.o.values().sum()", &doc).unwrap();
    assert_eq!(out, json!(60));
}

#[test]
fn entries_roundtrip_via_from_pairs() {
    let doc = json!({"o": {"a": 1, "b": 2}});
    let out = vm_query("$.o.entries().from_pairs()", &doc).unwrap();
    assert_eq!(out, json!({"a": 1, "b": 2}));
}

#[test]
fn group_by_entries_canonical_idiom() {
    let doc = json!({
        "orders": [
            {"region": "NA", "amount": 100},
            {"region": "EU", "amount": 200},
            {"region": "NA", "amount": 50}
        ]
    });
    let out = vm_query(
        "$.orders.group_by(@.region).entries().map(([region, rows]) =>
            {region: region, total: rows.map(@.amount).sum()})",
        &doc,
    )
    .unwrap();
    // Order is insertion-order: NA appears before EU.
    assert_eq!(
        out,
        json!([
            {"region": "NA", "total": 150},
            {"region": "EU", "total": 200}
        ])
    );
}

#[test]
fn group_by_entries_pick_keys() {
    let doc = json!({
        "items": [
            {"k": "a"}, {"k": "b"}, {"k": "a"}, {"k": "c"}
        ]
    });
    let out = vm_query("$.items.group_by(@.k).entries().map(e => e[0])", &doc).unwrap();
    assert_eq!(out, json!(["a", "b", "c"]));
}

#[test]
fn count_by_entries_to_csv_pipeline() {
    let doc = json!({
        "tweets": [
            {"id": 1, "tag": "foo"},
            {"id": 2, "tag": "bar"},
            {"id": 3, "tag": "foo"}
        ]
    });
    let out = vm_query(
        "$.tweets.count_by(@.tag).entries().map(([tag, count]) => {tag: tag, count: count})",
        &doc,
    )
    .unwrap();
    assert_eq!(
        out,
        json!([
            {"tag": "foo", "count": 2},
            {"tag": "bar", "count": 1}
        ])
    );
}

#[test]
fn entries_then_filter() {
    let doc = json!({"o": {"a": 1, "b": 5, "c": 10}});
    let out = vm_query("$.o.entries().filter(e => e[1] > 3)", &doc).unwrap();
    assert_eq!(out, json!([["b", 5], ["c", 10]]));
}

#[test]
fn entries_inside_comprehension() {
    // Comprehension over entries works (existing fixed iter path).
    let doc = json!({"o": {"a": 1, "b": 2}});
    let out = vm_query("[v for [k, v] in $.o.entries()]", &doc).unwrap();
    assert_eq!(out, json!([1, 2]));
}

#[test]
fn entries_on_literal_object_via_let() {
    // Object literal as receiver works through a let binding (object
    // literal directly is parsed as inline filter — separate concern).
    let out = vm_query("let o = {x: 1, y: 2} in o.entries()", &json!(null)).unwrap();
    assert_eq!(out, json!([["x", 1], ["y", 2]]));
}
