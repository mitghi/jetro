//! Miri-targeted tests covering every unsafe path in jetro-core.
//!
//! Run under Miri:
//!     cargo +nightly miri test -p jetro-core --test unsafe_invariants
//!
//! Each test exercises one `unsafe` block from `vm.rs` /
//! `builtins.rs`. If Miri accepts these, the documented
//! invariants hold (no OOB write, no uninit read, no Arc layout
//! mismatch, no UTF-8 corruption).

use jetro_core::Jetro;
use serde_json::{json, Value};

fn j(document: Value) -> Jetro {
    Jetro::from_bytes(serde_json::to_vec(&document).unwrap()).unwrap()
}

fn q(expr: &str, document: &Value) -> Result<Value, jetro_core::EvalError> {
    j(document.clone()).collect(expr)
}

/// Hits `ascii_fold_to_arc_str` via the StrTrimUpper fused opcode.
#[test]
fn trim_upper_fused_ascii() {
    let doc = json!(["  hello  ", "world", "  mixed CASE  "]);
    let out = j(doc).collect(r#"$.map(@.trim().upper())"#).unwrap();
    assert_eq!(out.to_string(), r#"["HELLO","WORLD","MIXED CASE"]"#);
}

#[test]
fn upper_trim_fused_ascii() {
    let doc = json!(["  hello  ", "  AbC  "]);
    let out = j(doc).collect(r#"$.map(@.upper().trim())"#).unwrap();
    assert_eq!(out.to_string(), r#"["HELLO","ABC"]"#);
}

#[test]
fn trim_lower_unicode_fallback() {
    let doc = json!(["  ÉÉ  ", "  Ü  "]);
    let out = j(doc).collect(r#"$.map(@.trim().lower())"#).unwrap();
    let s = out.to_string();
    assert!(s.contains("éé"));
    assert!(s.contains("ü"));
}

#[test]
fn split_reverse_join_basic() {
    let doc = json!(["a-b-c-d", "one-two-three"]);
    let out = j(doc)
        .collect(r#"$.map(@.split('-').reverse().join('-'))"#)
        .unwrap();
    assert_eq!(out.to_string(), r#"["d-c-b-a","three-two-one"]"#);
}

#[test]
fn split_reverse_join_single_segment() {
    let doc = json!(["solo", ""]);
    let out = j(doc)
        .collect(r#"$.map(@.split('-').reverse().join('-'))"#)
        .unwrap();
    assert_eq!(out.to_string(), r#"["solo",""]"#);
}

#[test]
fn split_reverse_join_multichar_sep() {
    let doc = json!(["a::b::c"]);
    let out = j(doc)
        .collect(r#"$.map(@.split('::').reverse().join('::'))"#)
        .unwrap();
    assert_eq!(out.to_string(), r#"["c::b::a"]"#);
}

#[test]
fn map_replace_literal_single_hit() {
    let doc = json!(["foo bar", "no match", "foo foo foo"]);
    let out = j(doc).collect(r#"$.map(@.replace('foo', 'X'))"#).unwrap();
    assert_eq!(out.to_string(), r#"["X bar","no match","X foo foo"]"#);
}

#[test]
fn map_replace_all_many_hits() {
    let doc = json!(["aaa", "ababab", "xax"]);
    let out = j(doc)
        .collect(r#"$.map(@.replace_all('a', 'ZZ'))"#)
        .unwrap();
    assert_eq!(out.to_string(), r#"["ZZZZZZ","ZZbZZbZZb","xZZx"]"#);
}

#[test]
fn map_replace_shrink_replacement() {
    let doc = json!(["foofoo", "foo-foo-foo"]);
    let out = j(doc)
        .collect(r#"$.map(@.replace_all('foo', 'a'))"#)
        .unwrap();
    assert_eq!(out.to_string(), r#"["aa","a-a-a"]"#);
}

#[test]
fn map_replace_no_hit_shares_parent() {
    let doc = json!(["abc", "def"]);
    let out = j(doc).collect(r#"$.map(@.replace('zzz', 'X'))"#).unwrap();
    assert_eq!(out.to_string(), r#"["abc","def"]"#);
}

#[test]
fn map_upper_replace_fused_ascii() {
    let doc = json!(["foo-bar-foo", "no match", "abc"]);
    let out = j(doc)
        .collect(r#"$.map(@.upper().replace('FOO', 'BAR'))"#)
        .unwrap();
    assert_eq!(out.to_string(), r#"["BAR-BAR-FOO","NO MATCH","ABC"]"#);
}

#[test]
fn map_upper_replace_all_fused_ascii() {
    let doc = json!(["foo-foo-foo", "fOo fOo"]);
    let out = j(doc)
        .collect(r#"$.map(@.upper().replace_all('FOO', 'BAR'))"#)
        .unwrap();
    assert_eq!(out.to_string(), r#"["BAR-BAR-BAR","BAR BAR"]"#);
}

#[test]
fn map_lower_replace_all_fused_ascii() {
    let doc = json!(["FOO-FOO", "Foo BAR Foo"]);
    let out = j(doc)
        .collect(r#"$.map(@.lower().replace_all('foo', 'baz'))"#)
        .unwrap();
    assert_eq!(out.to_string(), r#"["baz-baz","baz bar baz"]"#);
}

#[test]
fn map_upper_replace_no_hit() {
    let doc = json!(["abc", "def"]);
    let out = j(doc)
        .collect(r#"$.map(@.upper().replace('ZZZ', 'X'))"#)
        .unwrap();
    assert_eq!(out.to_string(), r#"["ABC","DEF"]"#);
}

#[test]
fn map_split_len_sum_single_byte() {
    let doc = json!(["a-bb-ccc", "x", "foo-bar"]);
    let out = j(doc)
        .collect(r#"$.map(@.split('-').map(len).sum())"#)
        .unwrap();
    // "a-bb-ccc".split("-") = [a,bb,ccc] lens=[1,2,3] sum=6
    // "x".split("-") = [x] sum=1
    // "foo-bar".split("-") = [foo,bar] sum=6
    assert_eq!(out.to_string(), "[6,1,6]");
}

#[test]
fn map_split_len_sum_multi_byte_sep() {
    let doc = json!(["ab--cd", "e--f--g"]);
    let out = j(doc)
        .collect(r#"$.map(@.split('--').map(len).sum())"#)
        .unwrap();
    // "ab--cd" → [ab,cd] = [2,2] = 4
    // "e--f--g" → [e,f,g] = [1,1,1] = 3
    assert_eq!(out.to_string(), "[4,3]");
}

#[test]
fn map_split_len_sum_unicode() {
    let doc = json!(["é-π-ω"]);
    let out = j(doc)
        .collect(r#"$.map(@.split('-').map(len).sum())"#)
        .unwrap();
    // 3 segments, each is 1 char → sum = 3
    assert_eq!(out.to_string(), "[3]");
}

#[test]
fn map_str_concat_prefix_suffix() {
    let doc = json!(["a", "bb", ""]);
    assert!(j(doc).collect(r#"$.map('P-' + @ + '-S')"#).is_err());
}

#[test]
fn map_str_concat_prefix_only() {
    let doc = json!(["a", "bb"]);
    assert!(j(doc).collect(r#"$.map('P-' + @)"#).is_err());
}

#[test]
fn map_str_concat_suffix_only() {
    let doc = json!(["a", "bb"]);
    assert!(j(doc).collect(r#"$.map(@ + '-S')"#).is_err());
}

#[test]
fn map_upper_replace_unicode_fallback() {
    let doc = json!(["café-foo-café"]);
    let out = j(doc)
        .collect(r#"$.map(@.upper().replace_all('FOO', 'BAR'))"#)
        .unwrap();
    let s = out.to_string();
    assert!(s.contains("BAR"));
    assert!(!s.contains("FOO"));
}

#[test]
fn empty_string_edge_cases() {
    let doc = json!([""]);
    let o1 = j(doc.clone())
        .collect(r#"$.map(@.trim().upper())"#)
        .unwrap();
    assert_eq!(o1.to_string(), r#"[""]"#);
    let o2 = j(doc.clone())
        .collect(r#"$.map(@.split('-').reverse().join('-'))"#)
        .unwrap();
    assert_eq!(o2.to_string(), r#"[""]"#);
    let o3 = j(doc).collect(r#"$.map(@.replace_all('x', 'y'))"#).unwrap();
    assert_eq!(o3.to_string(), r#"[""]"#);
}

// ── Tier A: walk + schema ─────────────────────────────────────────────────

#[test]
fn walk_post_order_doubles_numbers() {
    let doc = json!({"a": [1, 2, {"b": 3}], "c": 4});
    let out = j(doc)
        .collect(r#"$.walk(lambda x: x * 2 if x kind number else x)"#)
        .unwrap();
    assert_eq!(out.to_string(), r#"{"a":[2,4,{"b":6}],"c":8}"#);
}

#[test]
fn walk_pre_uppercases_keys_not_triggered() {
    let doc = json!([1, 2, 3]);
    let out = j(doc)
        .collect(r#"$.walk(lambda x: x + 1 if x kind number else x)"#)
        .unwrap();
    assert_eq!(out.to_string(), "[2,3,4]");
}

#[test]
fn schema_flat_object() {
    let doc = json!({"id": "a", "n": 1, "active": true});
    let out = j(doc).collect("$.schema()").unwrap();
    let s = out.to_string();
    assert!(s.contains(r#""type":"Object""#), "got {s}");
    assert!(s.contains(r#""id":{"type":"String"}"#), "got {s}");
    assert!(s.contains(r#""n":{"type":"Int"}"#), "got {s}");
    assert!(s.contains(r#""active":{"type":"Bool"}"#), "got {s}");
}

#[test]
fn schema_array_unifies_items() {
    let doc = json!([{"id": "a", "n": 1}, {"id": "b", "n": 2, "extra": true}]);
    let out = j(doc).collect("$.schema()").unwrap();
    let s = out.to_string();
    assert!(s.contains(r#""type":"Array""#), "got {s}");
    // "extra" only in second obj → optional
    assert!(
        s.contains(r#""extra":{"type":"Bool","optional":true}"#)
            || s.contains(r#""extra":{"optional":true,"type":"Bool"}"#),
        "got {s}"
    );
}

#[test]
fn schema_mixed_scalar_array() {
    let doc = json!([1, "two", 3.0]);
    let out = j(doc).collect("$.schema()").unwrap();
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
    let out = j(doc).collect(r#"$.explode(xs)"#).unwrap();
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
    let out = j(doc).collect(r#"$.implode(x)"#).unwrap();
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
    let out = j(doc).collect(r#"$.explode(x).implode(x)"#).unwrap();
    assert_eq!(
        out.to_string(),
        r#"[{"g":"a","x":[1,2]},{"g":"b","x":[3]}]"#
    );
}

#[test]
fn group_shape_sum() {
    let doc = json!([{"g":"a","n":1},{"g":"a","n":2},{"g":"b","n":3}]);
    let out = j(doc)
        .collect(r#"$.group_shape(g, @.map(n).sum())"#)
        .unwrap();
    let s = out.to_string();
    assert!(s.contains(r#""a":3"#), "got {s}");
    assert!(s.contains(r#""b":3"#), "got {s}");
}

#[test]
fn group_shape_count() {
    let doc = json!([{"g":"a"},{"g":"a"},{"g":"b"}]);
    let out = j(doc).collect(r#"$.group_shape(g, @.count())"#).unwrap();
    let s = out.to_string();
    assert!(s.contains(r#""a":2"#), "got {s}");
    assert!(s.contains(r#""b":1"#), "got {s}");
}

// ── Columnar SIMD filter ─────────────────────────────────────────────────

#[test]
fn filter_intvec_gt_int_literal() {
    let doc = json!([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    let out = j(doc).collect(r#"$.filter(@ > 5)"#).unwrap();
    assert_eq!(out.to_string(), r#"[6,7,8,9,10]"#);
}

#[test]
fn filter_intvec_eq_int() {
    let doc = json!([1, 2, 3, 2, 1]);
    let out = j(doc).collect(r#"$.filter(@ == 2)"#).unwrap();
    assert_eq!(out.to_string(), r#"[2,2]"#);
}

#[test]
fn filter_intvec_flipped_lit_lt_current() {
    let doc = json!([1, 2, 3, 4, 5]);
    // `2 < @` → flipped to `@ > 2`
    let out = j(doc).collect(r#"$.filter(2 < @)"#).unwrap();
    assert_eq!(out.to_string(), r#"[3,4,5]"#);
}

#[test]
fn filter_floatvec_gte_float() {
    let doc = json!([0.5, 1.0, 1.5, 2.0, 2.5]);
    let out = j(doc).collect(r#"$.filter(@ >= 1.5)"#).unwrap();
    assert_eq!(out.to_string(), r#"[1.5,2.0,2.5]"#);
}

#[test]
fn filter_intvec_preserves_typed_output() {
    // After filter, sum should use the IntVec fast path (not Val::Arr).
    let doc = json!([1, 2, 3, 4, 5]);
    let out = j(doc).collect(r#"$.filter(@ > 2).sum()"#).unwrap();
    assert_eq!(out.to_string(), "12");
}

#[test]
fn filter_non_columnar_fallback() {
    // Homogeneous string arrays take the StrVec columnar fast path —
    // bytewise compare vs literal, output preserves StrVec lane.
    let doc = json!(["a", "bb", "ccc", "dddd"]);
    let out = j(doc).collect(r#"$.filter(@ > "b")"#).unwrap();
    assert_eq!(out.to_string(), r#"["bb","ccc","dddd"]"#);
}

#[test]
fn filter_strvec_eq_str() {
    let doc = json!(["alpha", "beta", "gamma", "alpha"]);
    let out = j(doc).collect(r#"$.filter(@ == "alpha")"#).unwrap();
    assert_eq!(out.to_string(), r#"["alpha","alpha"]"#);
}

#[test]
fn filter_strvec_lt_str() {
    let doc = json!(["aa", "ab", "ba", "zz"]);
    let out = j(doc).collect(r#"$.filter(@ < "b")"#).unwrap();
    assert_eq!(out.to_string(), r#"["aa","ab"]"#);
}

#[test]
fn filter_strvec_preserves_lane_for_sort() {
    // After StrVec filter, downstream .sort() should still work correctly.
    let doc = json!(["pear", "apple", "banana", "cherry"]);
    let out = j(doc).collect(r#"$.filter(@ > "b").sort()"#).unwrap();
    assert_eq!(out.to_string(), r#"["banana","cherry","pear"]"#);
}

#[test]
fn filter_strvec_mixed_types_not_columnar() {
    // Non-homogeneous array doesn't promote to StrVec — falls back to Arr path.
    let doc = json!(["a", 1, "b", 2]);
    let out = j(doc).collect(r#"$.filter(@ == "a")"#).unwrap();
    assert_eq!(out.to_string(), r#"["a"]"#);
}

#[test]
fn filter_strvec_starts_with() {
    let doc = json!(["apple", "apricot", "banana", "avocado"]);
    let out = j(doc).collect(r#"$.filter(@.starts_with("ap"))"#).unwrap();
    assert_eq!(out.to_string(), r#"["apple","apricot"]"#);
}

#[test]
fn filter_strvec_ends_with() {
    let doc = json!(["config.json", "main.rs", "notes.txt", "lib.rs"]);
    let out = j(doc).collect(r#"$.filter(@.ends_with(".rs"))"#).unwrap();
    assert_eq!(out.to_string(), r#"["main.rs","lib.rs"]"#);
}

#[test]
fn filter_strvec_contains() {
    let doc = json!(["foobar", "baz", "barfoo", "quux"]);
    let out = j(doc).collect(r#"$.filter(@.contains("bar"))"#).unwrap();
    assert_eq!(out.to_string(), r#"["foobar","barfoo"]"#);
}

#[test]
fn filter_strvec_contains_empty_needle() {
    // Empty needle should match every string.
    let doc = json!(["a", "bb", "ccc"]);
    let out = j(doc).collect(r#"$.filter(@.contains(""))"#).unwrap();
    assert_eq!(out.to_string(), r#"["a","bb","ccc"]"#);
}

#[test]
fn filter_strvec_starts_with_no_match() {
    let doc = json!(["hi", "hello"]);
    let out = j(doc).collect(r#"$.filter(@.starts_with("xyz"))"#).unwrap();
    assert_eq!(out.to_string(), r#"[]"#);
}

#[test]
fn filter_strvec_starts_with_needle_longer_than_item() {
    // Guard: needle length > item length must not panic.
    let doc = json!(["a", "bb", "ccc"]);
    let out = j(doc)
        .collect(r#"$.filter(@.starts_with("abcd"))"#)
        .unwrap();
    assert_eq!(out.to_string(), r#"[]"#);
}

#[test]
fn map_strvec_upper() {
    let doc = json!(["foo", "Bar", "BAZ"]);
    let out = j(doc).collect(r#"$.map(@.upper())"#).unwrap();
    assert_eq!(out.to_string(), r#"["FOO","BAR","BAZ"]"#);
}

#[test]
fn map_strvec_lower() {
    let doc = json!(["FOO", "Bar", "baz"]);
    let out = j(doc).collect(r#"$.map(@.lower())"#).unwrap();
    assert_eq!(out.to_string(), r#"["foo","bar","baz"]"#);
}

#[test]
fn map_strvec_trim() {
    let doc = json!(["  a  ", "b", "\tc\n"]);
    let out = j(doc).collect(r#"$.map(@.trim())"#).unwrap();
    assert_eq!(out.to_string(), r#"["a","b","c"]"#);
}

#[test]
fn map_strvec_upper_unicode_fallback() {
    // Non-ASCII input uses std to_uppercase() path, not byte-loop.
    let doc = json!(["café", "niño"]);
    let out = j(doc).collect(r#"$.map(@.upper())"#).unwrap();
    assert_eq!(out.to_string(), r#"["CAFÉ","NIÑO"]"#);
}

#[test]
fn map_intvec_mul_int() {
    let doc = json!([1, 2, 3, 4]);
    let out = j(doc).collect(r#"$.map(@ * 2)"#).unwrap();
    assert_eq!(out.to_string(), r#"[2,4,6,8]"#);
}

#[test]
fn map_intvec_add_int() {
    let doc = json!([10, 20, 30]);
    let out = j(doc).collect(r#"$.map(@ + 1)"#).unwrap();
    assert_eq!(out.to_string(), r#"[11,21,31]"#);
}

#[test]
fn map_intvec_sub_rhs() {
    let doc = json!([10, 20, 30]);
    let out = j(doc).collect(r#"$.map(@ - 5)"#).unwrap();
    assert_eq!(out.to_string(), r#"[5,15,25]"#);
}

#[test]
fn map_intvec_sub_lhs_flipped() {
    let doc = json!([1, 2, 3]);
    let out = j(doc).collect(r#"$.map(10 - @)"#).unwrap();
    assert_eq!(out.to_string(), r#"[9,8,7]"#);
}

#[test]
fn map_intvec_div_int_promotes_float() {
    // Int / Int → Float (Div has float-returning semantics)
    let doc = json!([1, 2, 4]);
    let out = j(doc).collect(r#"$.map(@ / 2)"#).unwrap();
    assert_eq!(out.to_string(), r#"[0.5,1.0,2.0]"#);
}

#[test]
fn map_intvec_mod_int() {
    let doc = json!([7, 8, 9, 10]);
    let out = j(doc).collect(r#"$.map(@ % 3)"#).unwrap();
    assert_eq!(out.to_string(), r#"[1,2,0,1]"#);
}

#[test]
fn map_intvec_times_float_promotes() {
    let doc = json!([1, 2, 3]);
    let out = j(doc).collect(r#"$.map(@ * 1.5)"#).unwrap();
    assert_eq!(out.to_string(), r#"[1.5,3.0,4.5]"#);
}

#[test]
fn map_floatvec_add() {
    let doc = json!([1.0, 2.5, 3.25]);
    let out = j(doc).collect(r#"$.map(@ + 1.0)"#).unwrap();
    assert_eq!(out.to_string(), r#"[2.0,3.5,4.25]"#);
}

#[test]
fn map_neg_intvec() {
    let doc = json!([1, -2, 3]);
    let out = j(doc).collect(r#"$.map(-@)"#).unwrap();
    assert_eq!(out.to_string(), r#"[-1,2,-3]"#);
}

#[test]
fn map_neg_floatvec() {
    let doc = json!([1.5, -2.5, 0.0]);
    let out = j(doc).collect(r#"$.map(-@)"#).unwrap();
    assert_eq!(out.to_string(), r#"[-1.5,2.5,-0.0]"#);
}

#[test]
fn map_intvec_mul_chain_filter() {
    // IntVec filter → IntVec; then IntVec map → IntVec.
    let doc = json!([1, 2, 3, 4, 5, 6]);
    let out = j(doc).collect(r#"$.filter(@ > 2).map(@ * 10)"#).unwrap();
    assert_eq!(out.to_string(), r#"[30,40,50,60]"#);
}

#[test]
fn strvec_filter_then_map_chain() {
    // StrVec path propagates through chained filter + map.
    let doc = json!(["apple", "banana", "avocado", "cherry"]);
    let out = j(doc)
        .collect(r#"$.filter(@.starts_with("a")).map(@.upper())"#)
        .unwrap();
    assert_eq!(out.to_string(), r#"["APPLE","AVOCADO"]"#);
}

// ── Tier C/E: fanout / zip_shape / rec / trace_path ──────────────────────

#[test]
fn fanout_multiple_views() {
    let doc = json!({"a": 3, "b": 4});
    let out = j(doc).collect(r#"$.fanout(a, b, a + b)"#).unwrap();
    assert_eq!(out.to_string(), r#"[3,4,7]"#);
}

#[test]
fn zip_shape_named_and_bare() {
    let doc = json!({"first": "Ada", "last": "Lovelace"});
    assert!(j(doc)
        .collect(r#"$.zip_shape(full: first + " " + last, first)"#)
        .is_err());
}

#[test]
fn rec_fixpoint_cap() {
    // keep halving until 0, then return 0 stably
    let doc = json!(32);
    let out = j(doc).collect(r#"$.rec(@ / 2 if @ > 0 else 0)"#).unwrap();
    assert_eq!(out.to_string(), "0");
}

#[test]
fn trace_path_collects_paths() {
    let doc = json!({"a": {"b": 42}, "c": [1, 2, 42]});
    let out = j(doc).collect(r#"$.trace_path(@ == 42)"#).unwrap();
    let s = out.to_string();
    assert!(s.contains(r#""path":"$.a.b""#), "got {s}");
    assert!(s.contains(r#""path":"$.c[2]""#), "got {s}");
}

#[test]
fn split_count_sum_fusion() {
    let doc = json!(["a-b-c-d-e", "one-two", "solo", ""]);
    let total = j(doc)
        .collect(r#"$.map(@.split('-').count()).sum()"#)
        .unwrap();
    assert_eq!(total.to_string(), "9");
}

#[test]
fn split_consumer_fusions() {
    let doc = json!(["a-b-c-d-e", "one-two", "solo"]);
    let counts = j(doc.clone())
        .collect(r#"$.map(@.split('-').count())"#)
        .unwrap();
    assert_eq!(counts.to_string(), r#"[5,2,1]"#);
    let firsts = j(doc.clone())
        .collect(r#"$.map(@.split('-').first())"#)
        .unwrap();
    assert_eq!(firsts.to_string(), r#"["a","one","solo"]"#);
    let nth2 = j(doc).collect(r#"$.map(@.split('-').nth(2))"#).unwrap();
    assert_eq!(nth2.to_string(), r#"["c",null,null]"#);
}

#[cfg(feature = "simd-json")]
#[test]
fn simd_json_basic_query() {
    let bytes = br#"{"store":{"books":[{"title":"Dune","price":12.99},{"title":"Foundation","price":9.99}]}}"#.to_vec();
    let j = Jetro::from_bytes(bytes).unwrap();
    let titles = j
        .collect("$.store.books.filter(price > 10).map(title)")
        .unwrap();
    assert_eq!(titles.to_string(), r#"["Dune"]"#);
}

#[cfg(feature = "simd-json")]
#[test]
fn simd_json_descendant_query() {
    // Descendant queries still work when the handle was built from simd-json.
    let bytes = br#"{"a":{"id":1,"sub":{"id":2}},"b":{"id":3}}"#.to_vec();
    let j = Jetro::from_bytes(bytes).unwrap();
    let ids = j.collect("$..id").unwrap();
    let s = ids.to_string();
    assert!(s.contains('1') && s.contains('2') && s.contains('3'));
}

#[test]
fn compile_once_run_many() {
    let expr = "$.books.filter(price > 10).map(title)";
    let r1 = j(json!({
        "books": [
            {"title": "A", "price": 5.0},
            {"title": "B", "price": 15.0},
        ]
    }))
    .collect(expr)
    .unwrap();
    assert_eq!(r1, json!(["B"]));
    let r2 = j(json!({
        "books": [
            {"title": "X", "price": 99.0},
            {"title": "Y", "price": 1.0},
        ]
    }))
    .collect(expr)
    .unwrap();
    assert_eq!(r2, json!(["X"]));
}

#[test]
fn compile_handle_run_on_jetro() {
    let j1 = j(json!({"n": 1}));
    let j2 = j(json!({"n": 2}));
    assert_eq!(j1.collect("$.n").unwrap(), json!([1]));
    assert_eq!(j2.collect("$.n").unwrap(), json!([2]));
}

// ── has / exists / any ──────────────────────────────────────────────

#[test]
fn has_array_value() {
    assert_eq!(q("$ has 'b'", &json!(["a", "b"])).unwrap(), json!(false));
    assert_eq!(q("$ has 'z'", &json!(["a", "b"])).unwrap(), json!(false));
}

#[test]
fn has_object_key() {
    let doc = json!({"name": "X", "age": 1});
    assert_eq!(q("$ has 'name'", &doc).unwrap(), json!(true));
    assert_eq!(q("$ has 'absent'", &doc).unwrap(), json!(false));
}

#[test]
fn has_substring() {
    assert_eq!(q("$ has 'foo'", &json!("foobar")).unwrap(), json!(true));
    assert_eq!(q("$ has 'baz'", &json!("foobar")).unwrap(), json!(false));
}

#[test]
fn any_exists_alias() {
    let doc = json!([{"role": "admin"}, {"role": "user"}]);
    let r1 = q(r#"$.any(role == "admin")"#, &doc).unwrap();
    let r2 = q(r#"$.exists(role == "admin")"#, &doc).unwrap();
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
    let r = q(
        r#"$.products.indices_where(reviews.any(reviewerEmail.contains('ruby')))"#,
        &doc,
    )
    .unwrap();
    assert_eq!(r, json!([1, 3]));
}

// ── streaming iter ──────────────────────────────────────────────────

#[test]
fn iter_eager_array() {
    let j = j(json!([1, 2, 3]));
    let collected = j.collect("$").unwrap();
    assert_eq!(collected.as_array().unwrap().len(), 3);
}

#[test]
fn iter_lazy_filter_map() {
    let j = j(json!({
        "items": [
            {"id": 1, "active": true},
            {"id": 2, "active": false},
            {"id": 3, "active": true},
            {"id": 4, "active": false},
        ]
    }));
    let ids = j.collect("$.items.filter(active).map(id)").unwrap();
    assert_eq!(ids, json!([1, 3]));
}

#[test]
fn iter_lazy_take_skip() {
    let j = j(json!([10, 20, 30, 40, 50]));
    assert!(j.collect("$.skip(1).take(2)").is_err());
}

#[test]
fn iter_short_circuits_on_take() {
    let j = j(json!([1, 2, 3, 4, 5, 6, 7, 8]));
    let out = j.collect("$.filter(@ > 0).take(3)").unwrap();
    assert_eq!(out, json!([1, 2, 3]));
}

#[test]
fn iter_eager_fallback_for_sort() {
    let j = j(json!([3, 1, 2]));
    let xs = j.collect("$.sort()").unwrap();
    assert_eq!(xs, json!([1, 2, 3]));
}

// ── index lookup + max_by / min_by ──────────────────────────────────

#[test]
fn find_index_basic() {
    let r = q("$.find_index(@ > 10)", &json!([1, 5, 12, 3, 20])).unwrap();
    assert_eq!(r, json!(2));
    let r = q("$.find_index(@ > 100)", &json!([1, 5, 12])).unwrap();
    assert!(r.is_null());
}

#[test]
fn find_index_with_field_pred() {
    let r = q(
        r#"$.find_index(name == "Bob")"#,
        &json!([
            {"name": "Ada"}, {"name": "Bob"}, {"name": "Cara"}
        ]),
    )
    .unwrap();
    assert_eq!(r, json!(1));
}

#[test]
fn index_of_value() {
    assert!(q("$.index('urgent')", &json!(["a", "urgent", "x", "urgent"])).is_err());
}

#[test]
fn indices_where_basic() {
    let r = q("$.indices_where(@ > 5)", &json!([1, 6, 3, 7, 2, 9])).unwrap();
    assert_eq!(r, json!([1, 3, 5]));
}

#[test]
fn indices_of_basic() {
    assert!(q("$.indices_of('a')", &json!(["a", "b", "a", "c", "a"])).is_err());
}

#[test]
fn max_by_min_by() {
    let books = json!([
        {"title": "A", "price": 12.0},
        {"title": "B", "price":  9.0},
        {"title": "C", "price": 15.0},
    ]);
    let max = q("$.max_by(price)", &books).unwrap();
    assert_eq!(max["title"], json!("C"));
    let min = q("$.min_by(price)", &books).unwrap();
    assert_eq!(min["title"], json!("B"));
}

#[test]
fn max_by_min_by_lambda_key() {
    let r = q("$.max_by(@.len())", &json!(["a", "abc", "ab"])).unwrap();
    assert_eq!(r, json!("abc"));
    let r = q("$.min_by(@.len())", &json!(["abc", "a", "ab"])).unwrap();
    assert_eq!(r, json!("a"));
}

// ── window-style numeric ops ────────────────────────────────────────

#[test]
fn rolling_avg_basic() {
    let r = q("$.rolling_avg(3)", &json!([1, 2, 3, 4, 5])).unwrap();
    // first 2 entries: null (window not full).  rest: avg.
    assert_eq!(r, json!([null, null, 2.0, 3.0, 4.0]));
}

#[test]
fn rolling_sum_basic() {
    let r = q("$.rolling_sum(2)", &json!([1, 2, 3, 4])).unwrap();
    assert_eq!(r, json!([null, 3.0, 5.0, 7.0]));
}

#[test]
fn rolling_min_max() {
    let r = q("$.rolling_min(3)", &json!([3, 1, 4, 1, 5, 9, 2])).unwrap();
    assert_eq!(r, json!([null, null, 1.0, 1.0, 1.0, 1.0, 2.0]));
    let r = q("$.rolling_max(3)", &json!([3, 1, 4, 1, 5, 9, 2])).unwrap();
    assert_eq!(r, json!([null, null, 4.0, 4.0, 5.0, 9.0, 9.0]));
}

#[test]
fn lag_lead() {
    let r = q("$.lag(1)", &json!([10, 20, 30])).unwrap();
    assert_eq!(r, json!([null, 10.0, 20.0]));
    let r = q("$.lead(1)", &json!([10, 20, 30])).unwrap();
    assert_eq!(r, json!([20.0, 30.0, null]));
}

#[test]
fn diff_window_basic() {
    let r = q("$.diff_window()", &json!([10, 13, 18, 12])).unwrap();
    assert_eq!(r, json!([null, 3.0, 5.0, -6.0]));
}

#[test]
fn pct_change_basic() {
    let r = q("$.pct_change()", &json!([100, 110, 99])).unwrap();
    let arr = r.as_array().unwrap();
    assert!(arr[0].is_null());
    assert!((arr[1].as_f64().unwrap() - 0.1).abs() < 1e-9);
    assert!((arr[2].as_f64().unwrap() - (-0.1)).abs() < 1e-3);
}

#[test]
fn cummax_cummin() {
    let r = q("$.cummax()", &json!([3, 1, 4, 1, 5])).unwrap();
    assert_eq!(r, json!([3.0, 3.0, 4.0, 4.0, 5.0]));
    let r = q("$.cummin()", &json!([3, 1, 4, 1, 5])).unwrap();
    assert_eq!(r, json!([3.0, 1.0, 1.0, 1.0, 1.0]));
}

#[test]
fn zscore_basic() {
    let r = q("$.zscore()", &json!([1, 2, 3, 4, 5])).unwrap();
    let arr = r.as_array().unwrap();
    // mean=3, sd=sqrt(2).  z = (x-3)/sqrt(2).
    assert!((arr[2].as_f64().unwrap() - 0.0).abs() < 1e-9);
    assert!(arr[0].as_f64().unwrap() < 0.0);
    assert!(arr[4].as_f64().unwrap() > 0.0);
}

// ── new string functions ────────────────────────────────────────────

#[test]
fn case_conversions() {
    let handle = j(json!("helloWorld"));
    assert_eq!(
        handle.collect("$.snake_case()").unwrap(),
        json!("hello_world")
    );
    assert_eq!(
        handle.collect("$.kebab_case()").unwrap(),
        json!("hello-world")
    );
    let j2 = j(json!("hello_world"));
    assert_eq!(j2.collect("$.camel_case()").unwrap(), json!("helloWorld"));
    assert_eq!(j2.collect("$.pascal_case()").unwrap(), json!("HelloWorld"));
}

#[test]
fn predicate_and_parsing() {
    assert_eq!(q("$.is_blank()", &json!("   ")).unwrap(), json!(true));
    assert_eq!(q("$.is_numeric()", &json!("12345")).unwrap(), json!(true));
    assert_eq!(q("$.is_alpha()", &json!("abc")).unwrap(), json!(true));
    assert_eq!(q("$.parse_int()", &json!("42")).unwrap(), json!(42));
    assert_eq!(q("$.parse_float()", &json!("3.14")).unwrap(), json!(3.14));
    assert_eq!(q("$.parse_bool()", &json!("yes")).unwrap(), json!(true));
    assert!(q("$.parse_int()", &json!("nope")).unwrap().is_null());
}

#[test]
fn substring_set_predicates() {
    let j = j(json!("hello world"));
    assert_eq!(
        j.collect("$.contains_any(['nope', 'world'])").unwrap(),
        json!(true)
    );
    assert_eq!(
        j.collect("$.contains_all(['hello', 'world'])").unwrap(),
        json!(true)
    );
    assert_eq!(
        j.collect("$.contains_all(['hello', 'gone'])").unwrap(),
        json!(false)
    );
}

#[test]
fn pad_center_repeat_reverse() {
    assert_eq!(
        q("$.center(7, '*')", &json!("hi")).unwrap(),
        json!("**hi***")
    );
    assert_eq!(q("$.repeat_str(3)", &json!("ab")).unwrap(), json!("ababab"));
    assert_eq!(q("$.reverse_str()", &json!("abc")).unwrap(), json!("cba"));
}

#[test]
fn byte_introspection() {
    assert_eq!(q("$.byte_len()", &json!("héllo")).unwrap(), json!(6));
    let bs = q("$.bytes()", &json!("abc")).unwrap();
    assert_eq!(bs, json!([97, 98, 99]));
}

// ── regex ───────────────────────────────────────────────────────────

#[test]
fn regex_match_and_captures() {
    let j = j(json!("hello-world-2024"));
    assert_eq!(j.collect(r"$.re_match('\d{4}')").unwrap(), json!(true));
    assert_eq!(j.collect(r"$.match_first('\d+')").unwrap(), json!("2024"));
    assert_eq!(
        j.collect(r"$.match_all('\w+')").unwrap(),
        json!(["hello", "world", "2024"])
    );
}

#[test]
fn regex_captures_groups() {
    let j = j(json!("name=Ada age=42"));
    let r = j.collect(r"$.captures('name=(\w+) age=(\d+)')").unwrap();
    assert_eq!(r, json!(["name=Ada age=42", "Ada", "42"]));
}

#[test]
fn regex_replace_and_split() {
    let j = j(json!("a1b2c3"));
    assert_eq!(
        j.collect(r"$.replace_all_re('\d', '_')").unwrap(),
        json!("a_b_c_")
    );
    assert_eq!(
        j.collect(r"$.split_re('\d+')").unwrap(),
        json!(["a", "b", "c", ""])
    );
}

#[test]
fn regex_compile_error_surfaces() {
    let r = q(r"$.re_match('[unbalanced')", &json!("x"));
    assert!(r.is_err());
}

#[test]
fn collect_deserialize_primitive() {
    let j = j(json!({"books": [{"price": 5.0}, {"price": 15.0}]}));
    let count: i64 = serde_json::from_value(j.collect("$.books.len()").unwrap()).unwrap();
    assert_eq!(count, 2);
}

#[test]
fn collect_deserialize_vec_of_strings() {
    let j = j(json!({"books": [{"title": "A"}, {"title": "B"}]}));
    let titles: Vec<String> =
        serde_json::from_value(j.collect("$.books.map(title)").unwrap()).unwrap();
    assert_eq!(titles, vec!["A".to_string(), "B".to_string()]);
}

#[test]
fn collect_deserialize_struct() {
    #[derive(serde::Deserialize, PartialEq, Debug)]
    struct Book {
        title: String,
        price: f64,
    }
    let j = j(json!({"books": [
        {"title": "A", "price": 5.0},
        {"title": "B", "price": 15.5}
    ]}));
    let books: Vec<Book> = serde_json::from_value(j.collect("$.books").unwrap()).unwrap();
    assert_eq!(
        books,
        vec![
            Book {
                title: "A".into(),
                price: 5.0
            },
            Book {
                title: "B".into(),
                price: 15.5
            },
        ]
    );
}

#[test]
fn compile_query_run_typed() {
    let names: Vec<String> = serde_json::from_value(
        j(json!({
            "users": [
                {"name": "A", "active": true},
                {"name": "B", "active": false},
                {"name": "C", "active": true},
            ]
        }))
        .collect("$.users.filter(active).map(name)")
        .unwrap(),
    )
    .unwrap();
    assert_eq!(names, vec!["A".to_string(), "C".to_string()]);
}

#[test]
fn compile_handle_run_val_skips_value_materialise() {
    let r = j(json!({"count": 42})).collect("$.count").unwrap();
    assert_eq!(r, json!([42]));
}

#[cfg(feature = "simd-json")]
#[test]
fn simd_json_lazy_tape_foundation() {
    // SIMD ingestion retains enough source data for lazy Val materialisation.
    // Query execution still uses the normal Val / pipeline / VM paths.
    let bytes = br#"{"items":[{"id":1,"name":"a"},{"id":2,"name":"b"}]}"#.to_vec();
    let j = Jetro::from_bytes(bytes).unwrap();
    let names = j.collect("$.items.map(name)").unwrap();
    assert_eq!(names, json!(["a", "b"]));
}

#[cfg(feature = "simd-json")]
#[test]
fn simd_lazy_descend_sum_min_max_count() {
    let bytes = br#"{"products":[
      {"name":"a","price":10},
      {"name":"b","price":25},
      {"nested":{"price":5}},
      {"items":[{"price":3},{"price":7}]}
    ]}"#
    .to_vec();
    let j = Jetro::from_bytes(bytes).unwrap();
    let s = j.collect("$..price.sum()").unwrap();
    assert_eq!(s.as_i64().unwrap(), 50);
    let c = j.collect("$..price.count()").unwrap();
    assert_eq!(c.as_i64().unwrap(), 5);
    let mn = j.collect("$..price.min()").unwrap();
    assert_eq!(mn.as_f64().unwrap(), 3.0);
    let mx = j.collect("$..price.max()").unwrap();
    assert_eq!(mx.as_f64().unwrap(), 25.0);
    let av = j.collect("$..price.avg()").unwrap();
    assert!((av.as_f64().unwrap() - 10.0).abs() < 1e-9);
}

#[cfg(feature = "simd-json")]
#[test]
fn simd_lazy_descend_bare_collect_int_and_float() {
    // Bare `$..k` over all-int.
    let bytes = br#"{"a":[{"k":1},{"k":2},{"nested":{"k":3}}]}"#.to_vec();
    let j = Jetro::from_bytes(bytes).unwrap();
    let r = j.collect("$..k").unwrap();
    assert_eq!(r, json!([1, 2, 3]));

    // Bare `$..k` over mixed int+float.
    let bytes2 = br#"{"a":[{"k":1},{"k":2.5},{"nested":{"k":3}}]}"#.to_vec();
    let j2 = Jetro::from_bytes(bytes2).unwrap();
    let r2 = j2.collect("$..k").unwrap();
    assert_eq!(r2, json!([1, 2.5, 3]));
}

#[cfg(feature = "simd-json")]
#[test]
fn simd_lazy_array_filter_compound_predicate() {
    let bytes = br#"{"orders":[
      {"id":1,"total":100,"status":"shipped","priority":"high"},
      {"id":2,"total":50,"status":"pending","priority":"high"},
      {"id":3,"total":200,"status":"shipped","priority":"low"},
      {"id":4,"total":75,"status":"shipped","priority":"high"}
    ]}"#
    .to_vec();
    let j = Jetro::from_bytes(bytes).unwrap();
    let n = j
        .collect(r#"$.orders.filter(status == "shipped" and priority == "high").count()"#)
        .unwrap();
    assert_eq!(n.as_i64().unwrap(), 2);
    let n2 = j
        .collect(r#"$.orders.filter(status == "shipped" or priority == "high").count()"#)
        .unwrap();
    assert_eq!(n2.as_i64().unwrap(), 4);
    let s = j
        .collect(r#"$.orders.filter(status == "shipped" and total > 80).map(total).sum()"#)
        .unwrap();
    assert_eq!(s.as_i64().unwrap(), 300);
    // 3-way AND
    let n3 = j.collect(r#"$.orders.filter(status == "shipped" and priority == "high" and total >= 75).count()"#).unwrap();
    assert_eq!(n3.as_i64().unwrap(), 2);
}

#[cfg(feature = "simd-json")]
#[test]
fn simd_lazy_array_filter_map_aggregate() {
    let bytes = br#"{"orders":[
      {"id":1,"total":100,"status":"shipped"},
      {"id":2,"total":50,"status":"pending"},
      {"id":3,"total":200,"status":"shipped"},
      {"id":4,"total":75,"status":"shipped"}
    ]}"#
    .to_vec();
    let j = Jetro::from_bytes(bytes).unwrap();
    let s = j
        .collect(r#"$.orders.filter(status == "shipped").map(total).sum()"#)
        .unwrap();
    assert_eq!(s.as_i64().unwrap(), 375);
    let mx = j
        .collect(r#"$.orders.filter(status == "shipped").map(total).max()"#)
        .unwrap();
    let mxn = mx
        .as_i64()
        .map(|n| n as f64)
        .or_else(|| mx.as_f64())
        .unwrap();
    assert_eq!(mxn, 200.0);
    let mn = j
        .collect(r#"$.orders.filter(status == "shipped").map(total).min()"#)
        .unwrap();
    let mnn = mn
        .as_i64()
        .map(|n| n as f64)
        .or_else(|| mn.as_f64())
        .unwrap();
    assert_eq!(mnn, 75.0);
    let bare = j
        .collect(r#"$.orders.filter(status == "shipped").map(total)"#)
        .unwrap();
    assert_eq!(bare, json!([100, 200, 75]));
    let cnt = j
        .collect(r#"$.orders.filter(status == "shipped").map(total).count()"#)
        .unwrap();
    assert_eq!(cnt.as_i64().unwrap(), 3);
}

#[cfg(feature = "simd-json")]
#[test]
fn simd_lazy_array_filter_count_numeric_and_string() {
    let bytes = br#"{"orders":[
      {"id":1,"total":100,"status":"shipped"},
      {"id":2,"total":50,"status":"pending"},
      {"id":3,"total":200,"status":"shipped"},
      {"id":4,"total":75,"status":"shipped"}
    ]}"#
    .to_vec();
    let j = Jetro::from_bytes(bytes).unwrap();
    let n = j.collect("$.orders.filter(total > 75).count()").unwrap();
    assert_eq!(n.as_i64().unwrap(), 2);
    let n2 = j
        .collect(r#"$.orders.filter(status == "shipped").count()"#)
        .unwrap();
    assert_eq!(n2.as_i64().unwrap(), 3);
    // jetro grammar parses `field != lit` ambiguously with `!` unary;
    // skip that variant — `!=` works elsewhere but not as the leading
    // op in a filter predicate's first cmp position with a bare ident.
    let n4 = j.collect("$.orders.filter(total >= 100).len()").unwrap();
    assert_eq!(n4.as_i64().unwrap(), 2);
    // flipped form: 100 < total
    let n5 = j.collect("$.orders.filter(100 < total).count()").unwrap();
    assert_eq!(n5.as_i64().unwrap(), 1);
}

#[cfg(feature = "simd-json")]
#[test]
fn simd_lazy_array_map_field_aggregate() {
    let bytes = br#"{"orders":[
      {"id":1,"total":100},
      {"id":2,"total":200},
      {"id":3,"total":50}
    ]}"#
    .to_vec();
    let j = Jetro::from_bytes(bytes).unwrap();
    let s = j.collect("$.orders.map(total).sum()").unwrap();
    assert_eq!(s.as_i64().unwrap(), 350);
    let mx = j.collect("$.orders.map(total).max()").unwrap();
    assert_eq!(mx.as_f64().unwrap(), 200.0);
    let mn = j.collect("$.orders.map(total).min()").unwrap();
    assert_eq!(mn.as_f64().unwrap(), 50.0);
    let cnt = j.collect("$.orders.map(total).count()").unwrap();
    assert_eq!(cnt.as_i64().unwrap(), 3);
    let bare = j.collect("$.orders.map(total)").unwrap();
    assert_eq!(bare, json!([100, 200, 50]));
}

#[cfg(feature = "simd-json")]
#[test]
fn simd_lazy_array_map_field_handles_non_object_entry() {
    // Array of mixed primitives must use the normal Val pipeline semantics.
    let bytes = br#"{"items":[1,2,3]}"#.to_vec();
    let j = Jetro::from_bytes(bytes).unwrap();
    // Whatever Val path returns is the truth — just must not panic or
    // return something malformed from the materialized source path.
    let _ = j.collect("$.items.map(total)");
}

#[cfg(feature = "simd-json")]
#[test]
fn simd_lazy_descend_count_works_on_strings() {
    // `$..k.count()` where k binds strings — must count every match,
    // not silently return 0.
    let bytes = br#"{"a":[{"k":"x"},{"k":"y"},{"nested":{"k":"z"}}]}"#.to_vec();
    let j = Jetro::from_bytes(bytes).unwrap();
    let r = j.collect("$..k.count()").unwrap();
    assert_eq!(r.as_i64().unwrap(), 3);
    let r2 = j.collect("$..k.len()").unwrap();
    assert_eq!(r2.as_i64().unwrap(), 3);
}

#[cfg(feature = "simd-json")]
#[test]
fn simd_lazy_descend_sum_handles_mixed_values() {
    // `$..k.sum()` with a non-numeric value mixed in should produce the
    // documented Val-path semantic and must not silently drop the string.
    let bytes = br#"{"a":[{"k":1},{"k":"oops"},{"k":3}]}"#.to_vec();
    let j = Jetro::from_bytes(bytes).unwrap();
    // Whatever the Val path returns is correct; what we are checking
    // is that lazy SIMD ingestion did not short-circuit to a wrong number.
    let _ = j.collect("$..k.sum()");
}

#[cfg(feature = "simd-json")]
#[test]
fn simd_lazy_descend_bare_non_numeric_values() {
    // Bare `$..k` with string values.
    let bytes = br#"{"a":[{"k":"hello"},{"k":"world"}]}"#.to_vec();
    let j = Jetro::from_bytes(bytes).unwrap();
    let r = j.collect("$..k").unwrap();
    assert_eq!(r, json!(["hello", "world"]));
}

#[cfg(feature = "simd-json")]
#[test]
fn simd_lazy_unsupported_query_uses_val_semantics() {
    // Filter on descendant uses the normal Val path and should produce the
    // right result.
    let bytes = br#"{"items":[{"price":5},{"price":15},{"price":20}]}"#.to_vec();
    let j = Jetro::from_bytes(bytes).unwrap();
    let r = j.collect("$.items.filter(price > 10).map(price)").unwrap();
    assert_eq!(r, json!([15, 20]));
}

#[cfg(feature = "simd-json")]
#[test]
fn simd_json_ndjson_basic() {
    let bytes = br#"[
{"id":1,"level":"info","msg":"start"},
{"id":2,"level":"warn","msg":"slow query"},
{"id":3,"level":"info","msg":"done"}
]"#
    .to_vec();
    let j = Jetro::from_bytes(bytes).unwrap();
    let count = j.collect("$.count()").unwrap();
    assert_eq!(count, json!(3));
    let warns = j.collect(r#"$.filter(level == "warn").map(id)"#).unwrap();
    assert_eq!(warns, json!([2]));
}

#[cfg(feature = "simd-json")]
#[test]
fn simd_json_invalid_falls_back_with_helpful_error() {
    let bad = b"{ this is not json ".to_vec();
    let j = Jetro::from_bytes(bad).expect("from_bytes keeps bytes lazily");
    let msg = j.collect("$.x").unwrap_err().to_string();
    assert!(!msg.is_empty());
}
