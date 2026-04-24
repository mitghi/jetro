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
