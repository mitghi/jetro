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
fn map_upper_replace_fused_ascii() {
    let doc = json!(["foo-bar-foo", "no match", "abc"]);
    let out = Jetro::new(doc)
        .collect_val(r#"$.map(@.upper().replace('FOO', 'BAR'))"#).unwrap();
    assert_eq!(out.to_string(), r#"["BAR-BAR-FOO","NO MATCH","ABC"]"#);
}

#[test]
fn map_upper_replace_all_fused_ascii() {
    let doc = json!(["foo-foo-foo", "fOo fOo"]);
    let out = Jetro::new(doc)
        .collect_val(r#"$.map(@.upper().replace_all('FOO', 'BAR'))"#).unwrap();
    assert_eq!(out.to_string(), r#"["BAR-BAR-BAR","BAR BAR"]"#);
}

#[test]
fn map_lower_replace_all_fused_ascii() {
    let doc = json!(["FOO-FOO", "Foo BAR Foo"]);
    let out = Jetro::new(doc)
        .collect_val(r#"$.map(@.lower().replace_all('foo', 'baz'))"#).unwrap();
    assert_eq!(out.to_string(), r#"["baz-baz","baz bar baz"]"#);
}

#[test]
fn map_upper_replace_no_hit() {
    let doc = json!(["abc", "def"]);
    let out = Jetro::new(doc)
        .collect_val(r#"$.map(@.upper().replace('ZZZ', 'X'))"#).unwrap();
    assert_eq!(out.to_string(), r#"["ABC","DEF"]"#);
}

#[test]
fn map_str_concat_prefix_suffix() {
    let doc = json!(["a", "bb", ""]);
    let out = Jetro::new(doc)
        .collect_val(r#"$.map('P-' + @ + '-S')"#).unwrap();
    assert_eq!(out.to_string(), r#"["P-a-S","P-bb-S","P--S"]"#);
}

#[test]
fn map_str_concat_prefix_only() {
    let doc = json!(["a", "bb"]);
    let out = Jetro::new(doc)
        .collect_val(r#"$.map('P-' + @)"#).unwrap();
    assert_eq!(out.to_string(), r#"["P-a","P-bb"]"#);
}

#[test]
fn map_str_concat_suffix_only() {
    let doc = json!(["a", "bb"]);
    let out = Jetro::new(doc)
        .collect_val(r#"$.map(@ + '-S')"#).unwrap();
    assert_eq!(out.to_string(), r#"["a-S","bb-S"]"#);
}

#[test]
fn map_upper_replace_unicode_fallback() {
    let doc = json!(["café-foo-café"]);
    let out = Jetro::new(doc)
        .collect_val(r#"$.map(@.upper().replace_all('FOO', 'BAR'))"#).unwrap();
    let s = out.to_string();
    assert!(s.contains("BAR"));
    assert!(!s.contains("FOO"));
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

// ── Columnar SIMD filter ─────────────────────────────────────────────────

#[test]
fn filter_intvec_gt_int_literal() {
    let doc = json!([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    let out = Jetro::new(doc).collect_val(r#"$.filter(@ > 5)"#).unwrap();
    assert_eq!(out.to_string(), r#"[6,7,8,9,10]"#);
}

#[test]
fn filter_intvec_eq_int() {
    let doc = json!([1, 2, 3, 2, 1]);
    let out = Jetro::new(doc).collect_val(r#"$.filter(@ == 2)"#).unwrap();
    assert_eq!(out.to_string(), r#"[2,2]"#);
}

#[test]
fn filter_intvec_flipped_lit_lt_current() {
    let doc = json!([1, 2, 3, 4, 5]);
    // `2 < @` → flipped to `@ > 2`
    let out = Jetro::new(doc).collect_val(r#"$.filter(2 < @)"#).unwrap();
    assert_eq!(out.to_string(), r#"[3,4,5]"#);
}

#[test]
fn filter_floatvec_gte_float() {
    let doc = json!([0.5, 1.0, 1.5, 2.0, 2.5]);
    let out = Jetro::new(doc).collect_val(r#"$.filter(@ >= 1.5)"#).unwrap();
    assert_eq!(out.to_string(), r#"[1.5,2.0,2.5]"#);
}

#[test]
fn filter_intvec_preserves_typed_output() {
    // After filter, sum should use the IntVec fast path (not Val::Arr).
    let doc = json!([1, 2, 3, 4, 5]);
    let out = Jetro::new(doc).collect_val(r#"$.filter(@ > 2).sum()"#).unwrap();
    assert_eq!(out.to_string(), "12");
}

#[test]
fn filter_non_columnar_fallback() {
    // Homogeneous string arrays take the StrVec columnar fast path —
    // bytewise compare vs literal, output preserves StrVec lane.
    let doc = json!(["a", "bb", "ccc", "dddd"]);
    let out = Jetro::new(doc).collect_val(r#"$.filter(@ > "b")"#).unwrap();
    assert_eq!(out.to_string(), r#"["bb","ccc","dddd"]"#);
}

#[test]
fn filter_strvec_eq_str() {
    let doc = json!(["alpha", "beta", "gamma", "alpha"]);
    let out = Jetro::new(doc).collect_val(r#"$.filter(@ == "alpha")"#).unwrap();
    assert_eq!(out.to_string(), r#"["alpha","alpha"]"#);
}

#[test]
fn filter_strvec_lt_str() {
    let doc = json!(["aa", "ab", "ba", "zz"]);
    let out = Jetro::new(doc).collect_val(r#"$.filter(@ < "b")"#).unwrap();
    assert_eq!(out.to_string(), r#"["aa","ab"]"#);
}

#[test]
fn filter_strvec_preserves_lane_for_sort() {
    // After StrVec filter, downstream .sort() should still work correctly.
    let doc = json!(["pear", "apple", "banana", "cherry"]);
    let out = Jetro::new(doc).collect_val(r#"$.filter(@ > "b").sort()"#).unwrap();
    assert_eq!(out.to_string(), r#"["banana","cherry","pear"]"#);
}

#[test]
fn filter_strvec_mixed_types_not_columnar() {
    // Non-homogeneous array doesn't promote to StrVec — falls back to Arr path.
    let doc = json!(["a", 1, "b", 2]);
    let out = Jetro::new(doc).collect_val(r#"$.filter(@ == "a")"#).unwrap();
    assert_eq!(out.to_string(), r#"["a"]"#);
}

#[test]
fn filter_strvec_starts_with() {
    let doc = json!(["apple", "apricot", "banana", "avocado"]);
    let out = Jetro::new(doc).collect_val(r#"$.filter(@.starts_with("ap"))"#).unwrap();
    assert_eq!(out.to_string(), r#"["apple","apricot"]"#);
}

#[test]
fn filter_strvec_ends_with() {
    let doc = json!(["config.json", "main.rs", "notes.txt", "lib.rs"]);
    let out = Jetro::new(doc).collect_val(r#"$.filter(@.ends_with(".rs"))"#).unwrap();
    assert_eq!(out.to_string(), r#"["main.rs","lib.rs"]"#);
}

#[test]
fn filter_strvec_contains() {
    let doc = json!(["foobar", "baz", "barfoo", "quux"]);
    let out = Jetro::new(doc).collect_val(r#"$.filter(@.contains("bar"))"#).unwrap();
    assert_eq!(out.to_string(), r#"["foobar","barfoo"]"#);
}

#[test]
fn filter_strvec_contains_empty_needle() {
    // Empty needle should match every string.
    let doc = json!(["a", "bb", "ccc"]);
    let out = Jetro::new(doc).collect_val(r#"$.filter(@.contains(""))"#).unwrap();
    assert_eq!(out.to_string(), r#"["a","bb","ccc"]"#);
}

#[test]
fn filter_strvec_starts_with_no_match() {
    let doc = json!(["hi", "hello"]);
    let out = Jetro::new(doc).collect_val(r#"$.filter(@.starts_with("xyz"))"#).unwrap();
    assert_eq!(out.to_string(), r#"[]"#);
}

#[test]
fn filter_strvec_starts_with_needle_longer_than_item() {
    // Guard: needle length > item length must not panic.
    let doc = json!(["a", "bb", "ccc"]);
    let out = Jetro::new(doc).collect_val(r#"$.filter(@.starts_with("abcd"))"#).unwrap();
    assert_eq!(out.to_string(), r#"[]"#);
}

#[test]
fn map_strvec_upper() {
    let doc = json!(["foo", "Bar", "BAZ"]);
    let out = Jetro::new(doc).collect_val(r#"$.map(@.upper())"#).unwrap();
    assert_eq!(out.to_string(), r#"["FOO","BAR","BAZ"]"#);
}

#[test]
fn map_strvec_lower() {
    let doc = json!(["FOO", "Bar", "baz"]);
    let out = Jetro::new(doc).collect_val(r#"$.map(@.lower())"#).unwrap();
    assert_eq!(out.to_string(), r#"["foo","bar","baz"]"#);
}

#[test]
fn map_strvec_trim() {
    let doc = json!(["  a  ", "b", "\tc\n"]);
    let out = Jetro::new(doc).collect_val(r#"$.map(@.trim())"#).unwrap();
    assert_eq!(out.to_string(), r#"["a","b","c"]"#);
}

#[test]
fn map_strvec_upper_unicode_fallback() {
    // Non-ASCII input uses std to_uppercase() path, not byte-loop.
    let doc = json!(["café", "niño"]);
    let out = Jetro::new(doc).collect_val(r#"$.map(@.upper())"#).unwrap();
    assert_eq!(out.to_string(), r#"["CAFÉ","NIÑO"]"#);
}

#[test]
fn map_intvec_mul_int() {
    let doc = json!([1, 2, 3, 4]);
    let out = Jetro::new(doc).collect_val(r#"$.map(@ * 2)"#).unwrap();
    assert_eq!(out.to_string(), r#"[2,4,6,8]"#);
}

#[test]
fn map_intvec_add_int() {
    let doc = json!([10, 20, 30]);
    let out = Jetro::new(doc).collect_val(r#"$.map(@ + 1)"#).unwrap();
    assert_eq!(out.to_string(), r#"[11,21,31]"#);
}

#[test]
fn map_intvec_sub_rhs() {
    let doc = json!([10, 20, 30]);
    let out = Jetro::new(doc).collect_val(r#"$.map(@ - 5)"#).unwrap();
    assert_eq!(out.to_string(), r#"[5,15,25]"#);
}

#[test]
fn map_intvec_sub_lhs_flipped() {
    let doc = json!([1, 2, 3]);
    let out = Jetro::new(doc).collect_val(r#"$.map(10 - @)"#).unwrap();
    assert_eq!(out.to_string(), r#"[9,8,7]"#);
}

#[test]
fn map_intvec_div_int_promotes_float() {
    // Int / Int → Float (Div has float-returning semantics)
    let doc = json!([1, 2, 4]);
    let out = Jetro::new(doc).collect_val(r#"$.map(@ / 2)"#).unwrap();
    assert_eq!(out.to_string(), r#"[0.5,1.0,2.0]"#);
}

#[test]
fn map_intvec_mod_int() {
    let doc = json!([7, 8, 9, 10]);
    let out = Jetro::new(doc).collect_val(r#"$.map(@ % 3)"#).unwrap();
    assert_eq!(out.to_string(), r#"[1,2,0,1]"#);
}

#[test]
fn map_intvec_times_float_promotes() {
    let doc = json!([1, 2, 3]);
    let out = Jetro::new(doc).collect_val(r#"$.map(@ * 1.5)"#).unwrap();
    assert_eq!(out.to_string(), r#"[1.5,3.0,4.5]"#);
}

#[test]
fn map_floatvec_add() {
    let doc = json!([1.0, 2.5, 3.25]);
    let out = Jetro::new(doc).collect_val(r#"$.map(@ + 1.0)"#).unwrap();
    assert_eq!(out.to_string(), r#"[2.0,3.5,4.25]"#);
}

#[test]
fn map_neg_intvec() {
    let doc = json!([1, -2, 3]);
    let out = Jetro::new(doc).collect_val(r#"$.map(-@)"#).unwrap();
    assert_eq!(out.to_string(), r#"[-1,2,-3]"#);
}

#[test]
fn map_neg_floatvec() {
    let doc = json!([1.5, -2.5, 0.0]);
    let out = Jetro::new(doc).collect_val(r#"$.map(-@)"#).unwrap();
    assert_eq!(out.to_string(), r#"[-1.5,2.5,-0.0]"#);
}

#[test]
fn map_intvec_mul_chain_filter() {
    // IntVec filter → IntVec; then IntVec map → IntVec.
    let doc = json!([1, 2, 3, 4, 5, 6]);
    let out = Jetro::new(doc).collect_val(r#"$.filter(@ > 2).map(@ * 10)"#).unwrap();
    assert_eq!(out.to_string(), r#"[30,40,50,60]"#);
}

#[test]
fn strvec_filter_then_map_chain() {
    // StrVec path propagates through chained filter + map.
    let doc = json!(["apple", "banana", "avocado", "cherry"]);
    let out = Jetro::new(doc)
        .collect_val(r#"$.filter(@.starts_with("a")).map(@.upper())"#).unwrap();
    assert_eq!(out.to_string(), r#"["APPLE","AVOCADO"]"#);
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
