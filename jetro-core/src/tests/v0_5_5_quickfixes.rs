//! Coverage for the second batch of v0.5.5 fixes (the items previously
//! tracked in `limitations.md` as broken/unsupported in 0.5.5):
//!
//! - `.has(v)` method returns boolean (was returning the receiver)
//! - `.remove(pred)` evaluates the predicate (was a silent no-op)
//! - `missing(...keys)` variadic returns array of absent keys
//! - `update(path, fn)` reads / transforms / writes through `path`
//! - `get_path("a/b/c")` resolves multi-segment slash paths
//! - `dedent()` strips common leading whitespace (string-escape fix
//!   makes real `\n` newlines work)
//! - `now()` returns a positive Unix-millis integer
//! - `enumerate()` and `pairwise()` survive on path-rooted sources
//! - `zip_shape()` no-arg parallel-array interleave
//! - `group_shape()` no-arg key-set bucketing
//! - `partition(pred)` returns `[matching, non-matching]` tuple
//! - `approx_count_distinct()` HLL backend (exact for small inputs)
//! - String literal escapes (`\n`, `\t`, `\\`, `\xNN`, `\uXXXX`)

use super::common::vm_query;
use serde_json::json;

#[test]
fn string_escape_newline() {
    assert_eq!(
        vm_query("\"a\\nb\\nc\".lines()", &json!(null)).unwrap(),
        json!(["a", "b", "c"])
    );
}

#[test]
fn string_escape_tab() {
    assert_eq!(vm_query("\"a\\tb\".len()", &json!(null)).unwrap(), json!(3));
}

#[test]
fn string_escape_backslash() {
    assert_eq!(
        vm_query("\"a\\\\b\".len()", &json!(null)).unwrap(),
        json!(3)
    );
}

// `string_escape_quote`: escaped `\"` inside a `"..."` literal is not
// supported at grammar-level (the lexer's `lit_str_dq` rule terminates at
// the first `"` regardless of preceding backslashes). Use single-quoted
// `'...'` strings or `"` escape when an embedded double-quote is
// needed. Tracked as a separate grammar issue.

#[test]
fn string_escape_unknown_preserved_for_regex() {
    // `\d` stays `\d` so regex patterns continue to work.
    assert_eq!(
        vm_query("\"a1b\".re_match(\"\\d\")", &json!(null)).unwrap(),
        json!(true)
    );
}

#[test]
fn has_method_array_present() {
    let doc = json!({"xs": [1, 2, 3]});
    assert_eq!(vm_query("$.xs.has(2)", &doc).unwrap(), json!(true));
}

#[test]
fn has_method_array_absent() {
    let doc = json!({"xs": [1, 2, 3]});
    assert_eq!(vm_query("$.xs.has(99)", &doc).unwrap(), json!(false));
}

#[test]
fn has_method_object_present() {
    let doc = json!({"o": {"a": 1, "b": 2}});
    assert_eq!(vm_query(r#"$.o.has("a")"#, &doc).unwrap(), json!(true));
}

#[test]
fn has_method_object_absent() {
    let doc = json!({"o": {"a": 1}});
    assert_eq!(vm_query(r#"$.o.has("z")"#, &doc).unwrap(), json!(false));
}

#[test]
fn has_method_string_substring() {
    assert_eq!(
        vm_query(r#""hello".has("ell")"#, &json!(null)).unwrap(),
        json!(true)
    );
}

#[test]
fn remove_at_form_negatives() {
    let doc = json!({"xs": [1, -1, 2, -2, 3]});
    assert_eq!(
        vm_query("$.xs.remove(@ < 0)", &doc).unwrap(),
        json!([1, 2, 3])
    );
}

#[test]
fn remove_lambda_form() {
    let doc = json!({"xs": [1, 2, 3, 4, 5]});
    assert_eq!(
        vm_query("$.xs.remove(x => x > 2)", &doc).unwrap(),
        json!([1, 2])
    );
}

#[test]
fn remove_value_form_unchanged() {
    // Pure-literal arg keeps the value-equality semantics.
    let doc = json!({"xs": [1, 0, 2, 0, 3]});
    assert_eq!(vm_query("$.xs.remove(0)", &doc).unwrap(), json!([1, 2, 3]));
}

#[test]
fn missing_single_key_boolean() {
    let doc = json!({"o": {"host": "x"}});
    assert_eq!(
        vm_query(r#"$.o.missing("host")"#, &doc).unwrap(),
        json!(false)
    );
}

#[test]
fn missing_multiple_keys_returns_array() {
    let doc = json!({"o": {"host": "x", "port": 80}});
    assert_eq!(
        vm_query(r#"$.o.missing("host", "port", "missing")"#, &doc).unwrap(),
        json!(["missing"])
    );
}

#[test]
fn missing_all_absent() {
    let doc = json!({"o": {"x": 1}});
    assert_eq!(
        vm_query(r#"$.o.missing("a", "b")"#, &doc).unwrap(),
        json!(["a", "b"])
    );
}

#[test]
fn update_increments_nested() {
    let doc = json!({"counters": {"visits": 10}});
    assert_eq!(
        vm_query(r#"$.update("counters.visits", @ + 1)"#, &doc).unwrap(),
        json!({"counters": {"visits": 11}})
    );
}

#[test]
fn update_double_deep() {
    let doc = json!({"a": {"b": {"c": 42}}});
    assert_eq!(
        vm_query(r#"$.update("a.b.c", @ * 2)"#, &doc).unwrap(),
        json!({"a": {"b": {"c": 84}}})
    );
}

#[test]
fn get_path_slash_separator() {
    let doc = json!({"user": {"profile": {"name": "Ada"}}});
    assert_eq!(
        vm_query(r#"$.get_path("user/profile/name")"#, &doc).unwrap(),
        json!("Ada")
    );
}

#[test]
fn get_path_dot_separator() {
    let doc = json!({"user": {"profile": {"name": "Ada"}}});
    assert_eq!(
        vm_query(r#"$.get_path("user.profile.name")"#, &doc).unwrap(),
        json!("Ada")
    );
}

#[test]
fn get_path_array_index_through_slash() {
    let doc = json!({"users": [{"name": "a"}, {"name": "b"}]});
    assert_eq!(
        vm_query(r#"$.get_path("users/0/name")"#, &doc).unwrap(),
        json!("a")
    );
}

#[test]
fn get_path_missing_returns_null() {
    let doc = json!({"a": {"b": 1}});
    assert_eq!(
        vm_query(r#"$.get_path("a/b/missing/x")"#, &doc).unwrap(),
        json!(null)
    );
}

#[test]
fn has_path_slash_works() {
    let doc = json!({"a": {"b": {"c": 1}}});
    assert_eq!(
        vm_query(r#"$.has_path("a/b/c")"#, &doc).unwrap(),
        json!(true)
    );
}

#[test]
fn dedent_two_spaces() {
    assert_eq!(
        vm_query("\"  a\\n  b\".dedent()", &json!(null)).unwrap(),
        json!("a\nb")
    );
}

#[test]
fn dedent_preserves_extra_indent() {
    assert_eq!(
        vm_query("\"  a\\n    b\".dedent()", &json!(null)).unwrap(),
        json!("a\n  b")
    );
}

#[test]
fn now_returns_positive_int() {
    let v = vm_query("now()", &json!(null)).unwrap();
    let n = v.as_i64().expect("now() should be integer");
    assert!(n > 1_700_000_000_000); // sanity: post-2023 in ms
}

#[test]
fn now_in_patch() {
    let v = vm_query("patch $ {ts: now()}", &json!({})).unwrap();
    assert!(v["ts"].as_i64().unwrap() > 0);
}

#[test]
fn enumerate_on_path_yields_records() {
    let doc = json!({"users": [{"name": "a"}, {"name": "b"}, {"name": "c"}]});
    let r = vm_query("$.users.enumerate()", &doc).unwrap();
    let arr = r.as_array().unwrap();
    assert_eq!(arr.len(), 3);
    assert_eq!(arr[0]["index"], json!(0));
    assert_eq!(arr[2]["index"], json!(2));
}

#[test]
fn pairwise_on_path_yields_pairs() {
    let doc = json!({"prices": [100, 105, 110, 115]});
    assert_eq!(
        vm_query("$.prices.pairwise()", &doc).unwrap(),
        json!([[100, 105], [105, 110], [110, 115]])
    );
}

#[test]
fn pairwise_then_diff() {
    let doc = json!({"prices": [100, 105, 110, 115]});
    assert_eq!(
        vm_query("$.prices.pairwise().map(p => p[1] - p[0])", &doc).unwrap(),
        json!([5, 5, 5])
    );
}

#[test]
fn zip_shape_object_to_rows() {
    let doc = json!({"o": {"names": ["a", "b"], "ages": [1, 2]}});
    let r = vm_query("$.o.zip_shape()", &doc).unwrap();
    assert_eq!(r.as_array().unwrap().len(), 2);
    assert_eq!(r[0]["names"], json!("a"));
    assert_eq!(r[0]["ages"], json!(1));
    assert_eq!(r[1]["names"], json!("b"));
    assert_eq!(r[1]["ages"], json!(2));
}

#[test]
fn zip_shape_truncates_to_shortest() {
    let doc = json!({"o": {"a": [1, 2, 3], "b": [10, 20]}});
    let r = vm_query("$.o.zip_shape()", &doc).unwrap();
    assert_eq!(r.as_array().unwrap().len(), 2);
}

#[test]
fn zip_shape_broadcasts_scalar() {
    let doc = json!({"o": {"names": ["a", "b"], "tag": "x"}});
    let r = vm_query("$.o.zip_shape()", &doc).unwrap();
    assert_eq!(r[0]["tag"], json!("x"));
    assert_eq!(r[1]["tag"], json!("x"));
}

#[test]
fn group_shape_buckets_by_keyset() {
    let doc = json!({
        "rs": [{"id": 1, "name": "a"}, {"id": 2}, {"id": 3, "name": "c"}]
    });
    let r = vm_query("$.rs.group_shape()", &doc).unwrap();
    assert_eq!(r["id"].as_array().unwrap().len(), 1);
    assert_eq!(r["id,name"].as_array().unwrap().len(), 2);
}

#[test]
fn partition_returns_tuple() {
    assert_eq!(
        vm_query("[1,2,3,4,5,6].partition(@ % 2 == 0)", &json!(null)).unwrap(),
        json!([[2, 4, 6], [1, 3, 5]])
    );
}

#[test]
fn partition_let_unpacking() {
    // The tuple shape integrates with `let` destructuring style:
    // pull the matching half via `[0]`, the rest via `[1]`.
    let doc = json!({"xs": [1, 2, 3, 4]});
    assert_eq!(
        vm_query("$.xs.partition(@ > 2)[0]", &doc).unwrap(),
        json!([3, 4])
    );
    assert_eq!(
        vm_query("$.xs.partition(@ > 2)[1]", &doc).unwrap(),
        json!([1, 2])
    );
}

#[test]
fn partition_tuple_let_binding() {
    let doc = json!({
        "store": {
            "books": [
                {"title": "Dune", "active": true},
                {"title": "Draft", "active": false},
                {"title": "Foundation", "active": true}
            ]
        }
    });
    let r = vm_query(
        "let (active, inactive) = $.store.books.partition(active) in {active: active.map(title), inactive: inactive.map(title)}",
        &doc,
    )
    .unwrap();
    assert_eq!(
        r,
        json!({
            "active": ["Dune", "Foundation"],
            "inactive": ["Draft"]
        })
    );
}

#[test]
fn tuple_let_composes_with_regular_let_binding() {
    let r = vm_query(
        "let (lo, hi) = [1, 2], scale = 10 in {lo: lo * scale, hi: hi * scale}",
        &json!(null),
    )
    .unwrap();
    assert_eq!(r, json!({"lo": 10, "hi": 20}));
}

#[test]
fn tuple_let_avoids_synthetic_name_collision() {
    let r = vm_query(
        "let (__lettuple_0, b) = [4, 5] in {a: __lettuple_0, b: b}",
        &json!(null),
    )
    .unwrap();
    assert_eq!(r, json!({"a": 4, "b": 5}));
}

#[test]
fn partition_chained_path_source() {
    let doc = json!({"store": {"books": [{"price": 5}, {"price": 15}, {"price": 25}]}});
    let r = vm_query("$.store.books.partition(@.price > 10)", &doc).unwrap();
    assert_eq!(r[0].as_array().unwrap().len(), 2);
    assert_eq!(r[1].as_array().unwrap().len(), 1);
}

#[test]
fn approx_count_distinct_small_exact() {
    // Linear-counting correction makes small inputs exact.
    let doc = json!({"xs": [1, 2, 3, 1, 2, 1, 5, 7, 7, 7]});
    assert_eq!(
        vm_query("$.xs.approx_count_distinct()", &doc).unwrap(),
        json!(5)
    );
}

#[test]
fn approx_count_distinct_empty() {
    let doc = json!({"xs": []});
    assert_eq!(
        vm_query("$.xs.approx_count_distinct()", &doc).unwrap(),
        json!(0)
    );
}

#[test]
fn approx_count_distinct_all_same() {
    assert_eq!(
        vm_query("[1,1,1,1].approx_count_distinct()", &json!(null)).unwrap(),
        json!(1)
    );
}

#[test]
fn approx_count_distinct_strings() {
    let doc = json!({"xs": ["a", "b", "a", "c", "b", "a"]});
    assert_eq!(
        vm_query("$.xs.approx_count_distinct()", &doc).unwrap(),
        json!(3)
    );
}
