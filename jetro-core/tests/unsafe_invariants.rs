//! Miri-targeted tests covering every unsafe path in jetro-core.
//!
//! Run under Miri:
//!     cargo +nightly miri test -p jetro-core --test unsafe_invariants
//!
//! Each test exercises one `unsafe` block from `vm.rs` /
//! `eval/func_strings.rs`. If Miri accepts these, the documented
//! invariants hold (no OOB write, no uninit read, no Arc layout
//! mismatch, no UTF-8 corruption).

use jetro_core::Jetro;
use serde_json::json;

/// Hits `ascii_fold_to_arc_str` via the StrTrimUpper fused opcode.
#[test]
fn trim_upper_fused_ascii() {
    let doc = json!(["  hello  ", "world", "  mixed CASE  "]);
    let out = Jetro::new(doc)
        .collect_val(r#"$.map(@.trim().upper())"#).unwrap();
    assert_eq!(out.to_string(), r#"["HELLO","WORLD","MIXED CASE"]"#);
}

#[test]
fn upper_trim_fused_ascii() {
    let doc = json!(["  hello  ", "  AbC  "]);
    let out = Jetro::new(doc)
        .collect_val(r#"$.map(@.upper().trim())"#).unwrap();
    assert_eq!(out.to_string(), r#"["HELLO","ABC"]"#);
}

#[test]
fn trim_lower_unicode_fallback() {
    let doc = json!(["  ÉÉ  ", "  Ü  "]);
    let out = Jetro::new(doc)
        .collect_val(r#"$.map(@.trim().lower())"#).unwrap();
    let s = out.to_string();
    assert!(s.contains("éé"));
    assert!(s.contains("ü"));
}

#[test]
fn split_reverse_join_basic() {
    let doc = json!(["a-b-c-d", "one-two-three"]);
    let out = Jetro::new(doc)
        .collect_val(r#"$.map(@.split('-').reverse().join('-'))"#).unwrap();
    assert_eq!(out.to_string(), r#"["d-c-b-a","three-two-one"]"#);
}

#[test]
fn split_reverse_join_single_segment() {
    let doc = json!(["solo", ""]);
    let out = Jetro::new(doc)
        .collect_val(r#"$.map(@.split('-').reverse().join('-'))"#).unwrap();
    assert_eq!(out.to_string(), r#"["solo",""]"#);
}

#[test]
fn split_reverse_join_multichar_sep() {
    let doc = json!(["a::b::c"]);
    let out = Jetro::new(doc)
        .collect_val(r#"$.map(@.split('::').reverse().join('::'))"#).unwrap();
    assert_eq!(out.to_string(), r#"["c::b::a"]"#);
}

#[test]
fn map_replace_literal_single_hit() {
    let doc = json!(["foo bar", "no match", "foo foo foo"]);
    let out = Jetro::new(doc)
        .collect_val(r#"$.map(@.replace('foo', 'X'))"#).unwrap();
    assert_eq!(out.to_string(), r#"["X bar","no match","X foo foo"]"#);
}

#[test]
fn map_replace_all_many_hits() {
    let doc = json!(["aaa", "ababab", "xax"]);
    let out = Jetro::new(doc)
        .collect_val(r#"$.map(@.replace_all('a', 'ZZ'))"#).unwrap();
    assert_eq!(out.to_string(), r#"["ZZZZZZ","ZZbZZbZZb","xZZx"]"#);
}

#[test]
fn map_replace_shrink_replacement() {
    let doc = json!(["foofoo", "foo-foo-foo"]);
    let out = Jetro::new(doc)
        .collect_val(r#"$.map(@.replace_all('foo', 'a'))"#).unwrap();
    assert_eq!(out.to_string(), r#"["aa","a-a-a"]"#);
}

#[test]
fn map_replace_no_hit_shares_parent() {
    let doc = json!(["abc", "def"]);
    let out = Jetro::new(doc)
        .collect_val(r#"$.map(@.replace('zzz', 'X'))"#).unwrap();
    assert_eq!(out.to_string(), r#"["abc","def"]"#);
}

#[test]
fn empty_string_edge_cases() {
    let doc = json!([""]);
    let o1 = Jetro::new(doc.clone()).collect_val(r#"$.map(@.trim().upper())"#).unwrap();
    assert_eq!(o1.to_string(), r#"[""]"#);
    let o2 = Jetro::new(doc.clone()).collect_val(r#"$.map(@.split('-').reverse().join('-'))"#).unwrap();
    assert_eq!(o2.to_string(), r#"[""]"#);
    let o3 = Jetro::new(doc).collect_val(r#"$.map(@.replace_all('x', 'y'))"#).unwrap();
    assert_eq!(o3.to_string(), r#"[""]"#);
}

// ── Tier A: walk + schema ─────────────────────────────────────────────────

#[test]
fn walk_post_order_doubles_numbers() {
    let doc = json!({"a": [1, 2, {"b": 3}], "c": 4});
    let out = Jetro::new(doc)
        .collect_val(r#"$.walk(lambda x: x * 2 if x kind number else x)"#).unwrap();
    assert_eq!(out.to_string(), r#"{"a":[2,4,{"b":6}],"c":8}"#);
}

#[test]
fn walk_pre_uppercases_keys_not_triggered() {
    let doc = json!([1, 2, 3]);
    let out = Jetro::new(doc)
        .collect_val(r#"$.walk(lambda x: x + 1 if x kind number else x)"#).unwrap();
    assert_eq!(out.to_string(), "[2,3,4]");
}

#[test]
fn schema_flat_object() {
    let doc = json!({"id": "a", "n": 1, "active": true});
    let out = Jetro::new(doc).collect_val("$.schema()").unwrap();
    let s = out.to_string();
    assert!(s.contains(r#""type":"Object""#), "got {s}");
    assert!(s.contains(r#""id":{"type":"String"}"#), "got {s}");
    assert!(s.contains(r#""n":{"type":"Int"}"#), "got {s}");
    assert!(s.contains(r#""active":{"type":"Bool"}"#), "got {s}");
}

#[test]
fn schema_array_unifies_items() {
    let doc = json!([{"id": "a", "n": 1}, {"id": "b", "n": 2, "extra": true}]);
    let out = Jetro::new(doc).collect_val("$.schema()").unwrap();
    let s = out.to_string();
    assert!(s.contains(r#""type":"Array""#), "got {s}");
    // "extra" only in second obj → optional
    assert!(s.contains(r#""extra":{"type":"Bool","optional":true}"#), "got {s}");
}

#[test]
fn schema_mixed_scalar_array() {
    let doc = json!([1, "two", 3.0]);
    let out = Jetro::new(doc).collect_val("$.schema()").unwrap();
    let s = out.to_string();
    assert!(s.contains(r#""type":"Mixed""#), "got {s}");
}

// ── Tier B: explode / implode / group_shape ─────────────────────────────

#[test]
fn explode_basic() {
    let doc = json!([
        {"g":"a", "xs":[1,2,3]},
        {"g":"b", "xs":[9]},
        {"g":"c"}
    ]);
    let out = Jetro::new(doc).collect_val(r#"$.explode(xs)"#).unwrap();
    assert_eq!(
        out.to_string(),
        r#"[{"g":"a","xs":1},{"g":"a","xs":2},{"g":"a","xs":3},{"g":"b","xs":9},{"g":"c"}]"#
    );
}

#[test]
fn implode_basic() {
    let doc = json!([
        {"g":"a", "x":1},
        {"g":"a", "x":2},
        {"g":"b", "x":3}
    ]);
    let out = Jetro::new(doc).collect_val(r#"$.implode(x)"#).unwrap();
    assert_eq!(
        out.to_string(),
        r#"[{"g":"a","x":[1,2]},{"g":"b","x":[3]}]"#
    );
}

#[test]
fn explode_implode_roundtrip() {
    let doc = json!([
        {"g":"a", "x":[1,2]},
        {"g":"b", "x":[3]}
    ]);
    let out = Jetro::new(doc).collect_val(r#"$.explode(x).implode(x)"#).unwrap();
    assert_eq!(
        out.to_string(),
        r#"[{"g":"a","x":[1,2]},{"g":"b","x":[3]}]"#
    );
}

#[test]
fn group_shape_sum() {
    let doc = json!([{"g":"a","n":1},{"g":"a","n":2},{"g":"b","n":3}]);
    let out = Jetro::new(doc).collect_val(r#"$.group_shape(g, @.map(n).sum())"#).unwrap();
    let s = out.to_string();
    assert!(s.contains(r#""a":3"#), "got {s}");
    assert!(s.contains(r#""b":3"#), "got {s}");
}

#[test]
fn group_shape_count() {
    let doc = json!([{"g":"a"},{"g":"a"},{"g":"b"}]);
    let out = Jetro::new(doc).collect_val(r#"$.group_shape(g, @.count())"#).unwrap();
    let s = out.to_string();
    assert!(s.contains(r#""a":2"#), "got {s}");
    assert!(s.contains(r#""b":1"#), "got {s}");
}

// ── Tier C/E: fanout / zip_shape / rec / trace_path ──────────────────────

#[test]
fn fanout_multiple_views() {
    let doc = json!({"a": 3, "b": 4});
    let out = Jetro::new(doc).collect_val(r#"$.fanout(a, b, a + b)"#).unwrap();
    assert_eq!(out.to_string(), r#"[3,4,7]"#);
}

#[test]
fn zip_shape_named_and_bare() {
    let doc = json!({"first": "Ada", "last": "Lovelace"});
    let out = Jetro::new(doc)
        .collect_val(r#"$.zip_shape(full: first + " " + last, first)"#).unwrap();
    let s = out.to_string();
    assert!(s.contains(r#""full":"Ada Lovelace""#), "got {s}");
    assert!(s.contains(r#""first":"Ada""#), "got {s}");
}

#[test]
fn rec_fixpoint_cap() {
    // keep halving until 0, then return 0 stably
    let doc = json!(32);
    let out = Jetro::new(doc).collect_val(r#"$.rec(@ / 2 if @ > 0 else 0)"#).unwrap();
    assert_eq!(out.to_string(), "0");
}

#[test]
fn trace_path_collects_paths() {
    let doc = json!({"a": {"b": 42}, "c": [1, 2, 42]});
    let out = Jetro::new(doc)
        .collect_val(r#"$.trace_path(@ == 42)"#).unwrap();
    let s = out.to_string();
    assert!(s.contains(r#""path":"$.a.b""#), "got {s}");
    assert!(s.contains(r#""path":"$.c[2]""#), "got {s}");
}

#[test]
fn split_count_sum_fusion() {
    let doc = json!(["a-b-c-d-e", "one-two", "solo", ""]);
    let total = Jetro::new(doc)
        .collect_val(r#"$.map(@.split('-').count()).sum()"#).unwrap();
    assert_eq!(total.to_string(), "9");
}

#[test]
fn split_consumer_fusions() {
    let doc = json!(["a-b-c-d-e", "one-two", "solo"]);
    let counts = Jetro::new(doc.clone())
        .collect_val(r#"$.map(@.split('-').count())"#).unwrap();
    assert_eq!(counts.to_string(), r#"[5,2,1]"#);
    let firsts = Jetro::new(doc.clone())
        .collect_val(r#"$.map(@.split('-').first())"#).unwrap();
    assert_eq!(firsts.to_string(), r#"["a","one","solo"]"#);
    let nth2 = Jetro::new(doc)
        .collect_val(r#"$.map(@.split('-').nth(2))"#).unwrap();
    assert_eq!(nth2.to_string(), r#"["c",null,null]"#);
}
