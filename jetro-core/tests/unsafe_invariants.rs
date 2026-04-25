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
fn map_split_len_sum_single_byte() {
    let doc = json!(["a-bb-ccc", "x", "foo-bar"]);
    let out = Jetro::new(doc)
        .collect_val(r#"$.map(@.split('-').map(len).sum())"#).unwrap();
    // "a-bb-ccc".split("-") = [a,bb,ccc] lens=[1,2,3] sum=6
    // "x".split("-") = [x] sum=1
    // "foo-bar".split("-") = [foo,bar] sum=6
    assert_eq!(out.to_string(), "[6,1,6]");
}

#[test]
fn map_split_len_sum_multi_byte_sep() {
    let doc = json!(["ab--cd", "e--f--g"]);
    let out = Jetro::new(doc)
        .collect_val(r#"$.map(@.split('--').map(len).sum())"#).unwrap();
    // "ab--cd" → [ab,cd] = [2,2] = 4
    // "e--f--g" → [e,f,g] = [1,1,1] = 3
    assert_eq!(out.to_string(), "[4,3]");
}

#[test]
fn map_split_len_sum_unicode() {
    let doc = json!(["é-π-ω"]);
    let out = Jetro::new(doc)
        .collect_val(r#"$.map(@.split('-').map(len).sum())"#).unwrap();
    // 3 segments, each is 1 char → sum = 3
    assert_eq!(out.to_string(), "[3]");
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

#[cfg(feature = "simd-json")]
#[test]
fn simd_json_basic_query() {
    let bytes = br#"{"store":{"books":[{"title":"Dune","price":12.99},{"title":"Foundation","price":9.99}]}}"#.to_vec();
    let j = Jetro::from_simd(bytes).unwrap();
    let titles = j.collect("$.store.books.filter(price > 10).map(title)").unwrap();
    assert_eq!(titles.to_string(), r#"["Dune"]"#);
}

#[cfg(feature = "simd-json")]
#[test]
fn simd_json_descendant_byte_scan() {
    // raw_bytes retained so $..key takes the SIMD byte-scan path.
    let bytes = br#"{"a":{"id":1,"sub":{"id":2}},"b":{"id":3}}"#.to_vec();
    let j = Jetro::from_simd(bytes).unwrap();
    let ids = j.collect("$..id").unwrap();
    let s = ids.to_string();
    assert!(s.contains('1') && s.contains('2') && s.contains('3'));
}

#[test]
fn compile_once_run_many() {
    let q = Jetro::compile("$.books.filter(price > 10).map(title)").unwrap();
    let r1 = q.run(&json!({
        "books": [
            {"title": "A", "price": 5.0},
            {"title": "B", "price": 15.0},
        ]
    })).unwrap();
    assert_eq!(r1, json!(["B"]));
    // Same compiled handle, different doc.
    let r2 = q.run(&json!({
        "books": [
            {"title": "X", "price": 99.0},
            {"title": "Y", "price": 1.0},
        ]
    })).unwrap();
    assert_eq!(r2, json!(["X"]));
}

#[test]
fn compile_handle_run_on_jetro() {
    let q = Jetro::compile("$.n").unwrap();
    let j1 = Jetro::new(json!({"n": 1}));
    let j2 = Jetro::new(json!({"n": 2}));
    assert_eq!(q.run_on(&j1).unwrap(), json!(1));
    assert_eq!(q.run_on(&j2).unwrap(), json!(2));
}

// ── has / exists / any ──────────────────────────────────────────────

#[test]
fn has_array_value() {
    assert_eq!(jetro_core::query("$ has 'b'", &json!(["a", "b"])).unwrap(), json!(true));
    assert_eq!(jetro_core::query("$ has 'z'", &json!(["a", "b"])).unwrap(), json!(false));
}

#[test]
fn has_object_key() {
    let doc = json!({"name": "X", "age": 1});
    assert_eq!(jetro_core::query("$ has 'name'",   &doc).unwrap(), json!(true));
    assert_eq!(jetro_core::query("$ has 'absent'", &doc).unwrap(), json!(false));
}

#[test]
fn has_substring() {
    assert_eq!(jetro_core::query("$ has 'foo'", &json!("foobar")).unwrap(), json!(true));
    assert_eq!(jetro_core::query("$ has 'baz'", &json!("foobar")).unwrap(), json!(false));
}

#[test]
fn any_exists_alias() {
    let doc = json!([{"role": "admin"}, {"role": "user"}]);
    let r1 = jetro_core::query(r#"$.any(role == "admin")"#, &doc).unwrap();
    let r2 = jetro_core::query(r#"$.exists(role == "admin")"#, &doc).unwrap();
    assert_eq!(r1, json!(true));
    assert_eq!(r2, json!(true));
}

#[test]
fn elegant_indices_via_any_and_contains() {
    let doc = json!({
        "products": [
            {"id": 1, "reviews": [{"reviewerEmail": "alice@x"}, {"reviewerEmail": "bob@x"}]},
            {"id": 2, "reviews": [{"reviewerEmail": "ruby.andrews@x"}]},
            {"id": 3, "reviews": [{"reviewerEmail": "carol@x"}]},
            {"id": 4, "reviews": [{"reviewerEmail": "ruby.smith@x"}]},
        ]
    });
    let r = jetro_core::query(
        r#"$.products.indices_where(reviews.any(reviewerEmail.contains('ruby')))"#,
        &doc).unwrap();
    assert_eq!(r, json!([1, 3]));
}

// ── streaming iter ──────────────────────────────────────────────────

#[test]
fn iter_eager_array() {
    let j = Jetro::new(json!([1, 2, 3]));
    let collected: Vec<jetro_core::JetroVal> = j.iter("$").unwrap()
        .map(|r| r.unwrap()).collect();
    assert_eq!(collected.len(), 3);
}

#[test]
fn iter_lazy_filter_map() {
    let j = Jetro::new(json!({
        "items": [
            {"id": 1, "active": true},
            {"id": 2, "active": false},
            {"id": 3, "active": true},
            {"id": 4, "active": false},
        ]
    }));
    let ids: Vec<jetro_core::JetroVal> = j
        .iter("$.items.filter(active).map(id)").unwrap()
        .map(|r| r.unwrap()).collect();
    assert_eq!(ids.len(), 2);
    assert_eq!(ids[0].to_string(), "1");
    assert_eq!(ids[1].to_string(), "3");
}

#[test]
fn iter_lazy_take_skip() {
    let j = Jetro::new(json!([10, 20, 30, 40, 50]));
    let xs: Vec<jetro_core::JetroVal> = j.iter("$.skip(1).take(2)").unwrap()
        .map(|r| r.unwrap()).collect();
    assert_eq!(xs.len(), 2);
    assert_eq!(xs[0].to_string(), "20");
    assert_eq!(xs[1].to_string(), "30");
}

#[test]
fn iter_short_circuits_on_take() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let j = Jetro::new(json!([1, 2, 3, 4, 5, 6, 7, 8]));
    static COUNT: AtomicUsize = AtomicUsize::new(0);
    COUNT.store(0, Ordering::SeqCst);
    let mut iter = j.iter("$.filter(@ > 0).take(3)").unwrap();
    while let Some(v) = iter.next() {
        v.unwrap();
        COUNT.fetch_add(1, Ordering::SeqCst);
    }
    assert_eq!(COUNT.load(Ordering::SeqCst), 3, "take(3) must yield exactly 3 items");
}

#[test]
fn iter_eager_fallback_for_sort() {
    // sort() forces materialise; the iterator drains the resulting Vec.
    let j = Jetro::new(json!([3, 1, 2]));
    let xs: Vec<jetro_core::JetroVal> = j.iter("$.sort()").unwrap()
        .map(|r| r.unwrap()).collect();
    assert_eq!(xs.len(), 3);
    assert_eq!(xs[0].to_string(), "1");
    assert_eq!(xs[2].to_string(), "3");
}

// ── index lookup + max_by / min_by ──────────────────────────────────

#[test]
fn find_index_basic() {
    let r = jetro_core::query("$.find_index(@ > 10)", &json!([1, 5, 12, 3, 20])).unwrap();
    assert_eq!(r, json!(2));
    let r = jetro_core::query("$.find_index(@ > 100)", &json!([1, 5, 12])).unwrap();
    assert!(r.is_null());
}

#[test]
fn find_index_with_field_pred() {
    let r = jetro_core::query(r#"$.find_index(name == "Bob")"#, &json!([
        {"name": "Ada"}, {"name": "Bob"}, {"name": "Cara"}
    ])).unwrap();
    assert_eq!(r, json!(1));
}

#[test]
fn index_of_value() {
    let r = jetro_core::query("$.index('urgent')", &json!(["a", "urgent", "x", "urgent"])).unwrap();
    assert_eq!(r, json!(1));
    let r = jetro_core::query("$.index('absent')", &json!(["a", "b"])).unwrap();
    assert!(r.is_null());
}

#[test]
fn indices_where_basic() {
    let r = jetro_core::query("$.indices_where(@ > 5)", &json!([1, 6, 3, 7, 2, 9])).unwrap();
    assert_eq!(r, json!([1, 3, 5]));
}

#[test]
fn indices_of_basic() {
    let r = jetro_core::query("$.indices_of('a')", &json!(["a", "b", "a", "c", "a"])).unwrap();
    assert_eq!(r, json!([0, 2, 4]));
}

#[test]
fn max_by_min_by() {
    let books = json!([
        {"title": "A", "price": 12.0},
        {"title": "B", "price":  9.0},
        {"title": "C", "price": 15.0},
    ]);
    let max = jetro_core::query("$.max_by(price)", &books).unwrap();
    assert_eq!(max["title"], json!("C"));
    let min = jetro_core::query("$.min_by(price)", &books).unwrap();
    assert_eq!(min["title"], json!("B"));
}

#[test]
fn max_by_min_by_lambda_key() {
    let r = jetro_core::query("$.max_by(@.len())", &json!(["a", "abc", "ab"])).unwrap();
    assert_eq!(r, json!("abc"));
    let r = jetro_core::query("$.min_by(@.len())", &json!(["abc", "a", "ab"])).unwrap();
    assert_eq!(r, json!("a"));
}

// ── window-style numeric ops ────────────────────────────────────────

#[test]
fn rolling_avg_basic() {
    let r = jetro_core::query("$.rolling_avg(3)", &json!([1, 2, 3, 4, 5])).unwrap();
    // first 2 entries: null (window not full).  rest: avg.
    assert_eq!(r, json!([null, null, 2.0, 3.0, 4.0]));
}

#[test]
fn rolling_sum_basic() {
    let r = jetro_core::query("$.rolling_sum(2)", &json!([1, 2, 3, 4])).unwrap();
    assert_eq!(r, json!([null, 3.0, 5.0, 7.0]));
}

#[test]
fn rolling_min_max() {
    let r = jetro_core::query("$.rolling_min(3)", &json!([3, 1, 4, 1, 5, 9, 2])).unwrap();
    assert_eq!(r, json!([null, null, 1.0, 1.0, 1.0, 1.0, 2.0]));
    let r = jetro_core::query("$.rolling_max(3)", &json!([3, 1, 4, 1, 5, 9, 2])).unwrap();
    assert_eq!(r, json!([null, null, 4.0, 4.0, 5.0, 9.0, 9.0]));
}

#[test]
fn lag_lead() {
    let r = jetro_core::query("$.lag(1)", &json!([10, 20, 30])).unwrap();
    assert_eq!(r, json!([null, 10.0, 20.0]));
    let r = jetro_core::query("$.lead(1)", &json!([10, 20, 30])).unwrap();
    assert_eq!(r, json!([20.0, 30.0, null]));
}

#[test]
fn diff_window_basic() {
    let r = jetro_core::query("$.diff_window()", &json!([10, 13, 18, 12])).unwrap();
    assert_eq!(r, json!([null, 3.0, 5.0, -6.0]));
}

#[test]
fn pct_change_basic() {
    let r = jetro_core::query("$.pct_change()", &json!([100, 110, 99])).unwrap();
    let arr = r.as_array().unwrap();
    assert!(arr[0].is_null());
    assert!((arr[1].as_f64().unwrap() - 0.1).abs() < 1e-9);
    assert!((arr[2].as_f64().unwrap() - (-0.1)).abs() < 1e-3);
}

#[test]
fn cummax_cummin() {
    let r = jetro_core::query("$.cummax()", &json!([3, 1, 4, 1, 5])).unwrap();
    assert_eq!(r, json!([3.0, 3.0, 4.0, 4.0, 5.0]));
    let r = jetro_core::query("$.cummin()", &json!([3, 1, 4, 1, 5])).unwrap();
    assert_eq!(r, json!([3.0, 1.0, 1.0, 1.0, 1.0]));
}

#[test]
fn zscore_basic() {
    let r = jetro_core::query("$.zscore()", &json!([1, 2, 3, 4, 5])).unwrap();
    let arr = r.as_array().unwrap();
    // mean=3, sd=sqrt(2).  z = (x-3)/sqrt(2).
    assert!((arr[2].as_f64().unwrap() - 0.0).abs() < 1e-9);
    assert!(arr[0].as_f64().unwrap() < 0.0);
    assert!(arr[4].as_f64().unwrap() > 0.0);
}

// ── new string functions ────────────────────────────────────────────

#[test]
fn case_conversions() {
    let j = Jetro::new(json!("helloWorld"));
    assert_eq!(j.collect("$.snake_case()").unwrap(), json!("hello_world"));
    assert_eq!(j.collect("$.kebab_case()").unwrap(), json!("hello-world"));
    let j2 = Jetro::new(json!("hello_world"));
    assert_eq!(j2.collect("$.camel_case()").unwrap(), json!("helloWorld"));
    assert_eq!(j2.collect("$.pascal_case()").unwrap(), json!("HelloWorld"));
}

#[test]
fn predicate_and_parsing() {
    assert_eq!(jetro_core::query("$.is_blank()", &json!("   "))
        .unwrap(), json!(true));
    assert_eq!(jetro_core::query("$.is_numeric()", &json!("12345"))
        .unwrap(), json!(true));
    assert_eq!(jetro_core::query("$.is_alpha()", &json!("abc"))
        .unwrap(), json!(true));
    assert_eq!(jetro_core::query("$.parse_int()", &json!("42"))
        .unwrap(), json!(42));
    assert_eq!(jetro_core::query("$.parse_float()", &json!("3.14"))
        .unwrap(), json!(3.14));
    assert_eq!(jetro_core::query("$.parse_bool()", &json!("yes"))
        .unwrap(), json!(true));
    assert!(jetro_core::query("$.parse_int()", &json!("nope"))
        .unwrap().is_null());
}

#[test]
fn substring_set_predicates() {
    let j = Jetro::new(json!("hello world"));
    assert_eq!(j.collect("$.contains_any(['nope', 'world'])").unwrap(), json!(true));
    assert_eq!(j.collect("$.contains_all(['hello', 'world'])").unwrap(), json!(true));
    assert_eq!(j.collect("$.contains_all(['hello', 'gone'])").unwrap(), json!(false));
}

#[test]
fn pad_center_repeat_reverse() {
    assert_eq!(jetro_core::query("$.center(7, '*')",  &json!("hi")).unwrap(), json!("**hi***"));
    assert_eq!(jetro_core::query("$.repeat_str(3)",   &json!("ab")).unwrap(), json!("ababab"));
    assert_eq!(jetro_core::query("$.reverse_str()",    &json!("abc")).unwrap(), json!("cba"));
}

#[test]
fn byte_introspection() {
    assert_eq!(jetro_core::query("$.byte_len()", &json!("héllo")).unwrap(), json!(6));
    let bs = jetro_core::query("$.bytes()", &json!("abc")).unwrap();
    assert_eq!(bs, json!([97, 98, 99]));
}

// ── regex ───────────────────────────────────────────────────────────

#[test]
fn regex_match_and_captures() {
    let j = Jetro::new(json!("hello-world-2024"));
    assert_eq!(j.collect(r"$.re_match('\d{4}')").unwrap(), json!(true));
    assert_eq!(j.collect(r"$.match_first('\d+')").unwrap(), json!("2024"));
    assert_eq!(j.collect(r"$.match_all('\w+')").unwrap(),
        json!(["hello", "world", "2024"]));
}

#[test]
fn regex_captures_groups() {
    let j = Jetro::new(json!("name=Ada age=42"));
    let r = j.collect(r"$.captures('name=(\w+) age=(\d+)')").unwrap();
    assert_eq!(r, json!(["name=Ada age=42", "Ada", "42"]));
}

#[test]
fn regex_replace_and_split() {
    let j = Jetro::new(json!("a1b2c3"));
    assert_eq!(j.collect(r"$.replace_all_re('\d', '_')").unwrap(),
        json!("a_b_c_"));
    assert_eq!(j.collect(r"$.split_re('\d+')").unwrap(),
        json!(["a", "b", "c", ""]));
}

#[test]
fn regex_compile_error_surfaces() {
    let r = jetro_core::query(r"$.re_match('[unbalanced')", &json!("x"));
    assert!(r.is_err());
}

#[test]
fn collect_typed_primitive() {
    let j = Jetro::new(json!({"books": [{"price": 5.0}, {"price": 15.0}]}));
    let count: i64 = j.collect_typed("$.books.len()").unwrap();
    assert_eq!(count, 2);
}

#[test]
fn collect_typed_vec_of_strings() {
    let j = Jetro::new(json!({"books": [{"title": "A"}, {"title": "B"}]}));
    let titles: Vec<String> = j.collect_typed("$.books.map(title)").unwrap();
    assert_eq!(titles, vec!["A".to_string(), "B".to_string()]);
}

#[test]
fn collect_typed_struct() {
    #[derive(serde::Deserialize, PartialEq, Debug)]
    struct Book { title: String, price: f64 }
    let j = Jetro::new(json!({"books": [
        {"title": "A", "price": 5.0},
        {"title": "B", "price": 15.5}
    ]}));
    let books: Vec<Book> = j.collect_typed("$.books").unwrap();
    assert_eq!(books, vec![
        Book { title: "A".into(), price: 5.0 },
        Book { title: "B".into(), price: 15.5 },
    ]);
}

#[test]
fn compile_query_run_typed() {
    let q = Jetro::compile("$.users.filter(active).map(name)").unwrap();
    let names: Vec<String> = q.run_typed(&json!({
        "users": [
            {"name": "A", "active": true},
            {"name": "B", "active": false},
            {"name": "C", "active": true},
        ]
    })).unwrap();
    assert_eq!(names, vec!["A".to_string(), "C".to_string()]);
}

#[test]
fn compile_handle_run_val_skips_value_materialise() {
    use jetro_core::JetroVal;
    let q = Jetro::compile("$.count").unwrap();
    // Build a Val tree directly and run against it.
    let doc = json!({"count": 42});
    let val: JetroVal = (&doc).into();
    let r = q.run_val(val).unwrap();
    assert_eq!(r.to_string(), "42");
}

#[cfg(feature = "simd-json")]
#[test]
fn simd_json_lazy_tape_foundation() {
    // Phase 6 foundation: Jetro::from_simd_lazy parses the input and
    // retains a TapeData alongside the Val.  Today the execute path
    // still uses the Val; the tape is just available for future
    // tape-aware opcode handlers (Day 2 of the plan).
    let bytes = br#"{"items":[{"id":1,"name":"a"},{"id":2,"name":"b"}]}"#.to_vec();
    let j = Jetro::from_simd_lazy(bytes).unwrap();
    let tape = j.tape().expect("from_simd_lazy must retain tape");
    assert!(tape.nodes.len() > 0);
    assert!(tape.root_len() > 0);
    // Existing query path keeps working through the cached Val.
    let names = j.collect("$.items.map(name)").unwrap();
    assert_eq!(names, json!(["a", "b"]));
}

#[cfg(feature = "simd-json")]
#[test]
fn simd_json_lazy_tape_string_offsets_resolve() {
    let bytes = br#"["alpha","beta","gamma"]"#.to_vec();
    let j = Jetro::from_simd_lazy(bytes).unwrap();
    let tape = j.tape().unwrap();
    let mut found: Vec<&str> = Vec::new();
    for (i, n) in tape.nodes.iter().enumerate() {
        if matches!(n, jetro_core::strref::TapeNode::StringRef { .. }) {
            found.push(tape.str_at(i));
        }
    }
    assert_eq!(found, vec!["alpha", "beta", "gamma"]);
}

#[cfg(feature = "simd-json")]
#[test]
fn simd_json_ndjson_basic() {
    let lines = b"\
{\"id\":1,\"level\":\"info\",\"msg\":\"start\"}
{\"id\":2,\"level\":\"warn\",\"msg\":\"slow query\"}

{\"id\":3,\"level\":\"info\",\"msg\":\"done\"}
";
    let j = Jetro::from_ndjson(lines).unwrap();
    let count = j.collect("$.count()").unwrap();
    assert_eq!(count, json!(3));
    let warns = j.collect(r#"$.filter(level == "warn").map(id)"#).unwrap();
    assert_eq!(warns, json!([2]));
}

#[cfg(feature = "simd-json")]
#[test]
fn simd_json_ndjson_error_carries_line_number() {
    let lines = b"\
{\"id\":1}
this is not json
{\"id\":3}
";
    let r = Jetro::from_ndjson(lines);
    let msg = match r { Err(e) => e, Ok(_) => panic!("expected ndjson error") };
    assert!(msg.contains("ndjson line 2"));
}

#[cfg(feature = "simd-json")]
#[test]
fn simd_json_invalid_falls_back_with_helpful_error() {
    let bad = b"{ this is not json ".to_vec();
    let r = Jetro::from_simd(bad);
    let msg = match r {
        Err(e) => e,
        Ok(_)  => panic!("expected error on invalid JSON"),
    };
    assert!(msg.contains("simd-json") || msg.contains("serde_json"));
}
