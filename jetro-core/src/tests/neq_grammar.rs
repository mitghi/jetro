//! Regression for `!=` parse failure.
//!
//! Before the fix, the postfix `quantifier` rule accepted a bare `"!"`
//! greedily and ate the leading `!` of every `!=` operator, so any `a != b`
//! expression failed to parse with a confusing "expected kw_and / kw_or /
//! kw_if / kw_kind" diagnostic. The grammar fix adds a `!"="` lookahead to
//! the quantifier alternative, leaving `!=` for `cmp_op` to consume.

use super::common::vm_query;
use serde_json::json;

#[test]
fn neq_top_level() {
    let doc = json!({"a": 1, "b": 2});
    assert_eq!(vm_query("$.a != 1", &doc).unwrap(), json!(false));
    assert_eq!(vm_query("$.a != 2", &doc).unwrap(), json!(true));
}

#[test]
fn neq_in_filter_at_form() {
    let doc = json!({"xs": [1, 2, 3]});
    assert_eq!(
        vm_query("$.xs.filter(@ != 2)", &doc).unwrap(),
        json!([1, 3])
    );
}

#[test]
fn neq_in_filter_arrow_lambda() {
    let doc = json!({"xs": [1, 2, 3]});
    assert_eq!(
        vm_query("$.xs.filter(r => r != 2)", &doc).unwrap(),
        json!([1, 3])
    );
}

#[test]
fn neq_in_filter_lambda_keyword() {
    let doc = json!({"xs": [1, 2, 3]});
    assert_eq!(
        vm_query("$.xs.filter(lambda r: r != 2)", &doc).unwrap(),
        json!([1, 3])
    );
}

#[test]
fn neq_string_arg() {
    let doc = json!({"xs": ["a", "b", "c"]});
    assert_eq!(
        vm_query(r#"$.xs.filter(@ != "b")"#, &doc).unwrap(),
        json!(["a", "c"])
    );
}

#[test]
fn neq_filter_keys_named_lambda() {
    let doc = json!({"o": {"a": 1, "b": 2, "c": 3}});
    let out = vm_query(r#"$.o.filter_keys(k => k != "b")"#, &doc).unwrap();
    let map = out.as_object().expect("object");
    assert!(map.contains_key("a"));
    assert!(map.contains_key("c"));
    assert!(!map.contains_key("b"));
}

#[test]
fn neq_inline_filter_in_chain() {
    // Jetro's inline filter is `{expr}`. `[expr]` is index/slice syntax.
    let doc = json!({"xs": [{"v": 1}, {"v": 2}, {"v": 3}]});
    assert_eq!(
        vm_query("$.xs{@.v != 2}", &doc).unwrap(),
        json!([{"v": 1}, {"v": 3}])
    );
}

#[test]
fn neq_in_match_guard() {
    let doc = json!({"x": 5});
    let r = vm_query("match $.x with { v when v != 0 -> v, _ -> -1 }", &doc).unwrap();
    assert_eq!(r, json!(5));
    let zero = vm_query(
        "match $.x with { v when v != 0 -> v, _ -> -1 }",
        &json!({"x": 0}),
    )
    .unwrap();
    assert_eq!(zero, json!(-1));
}

#[test]
fn neq_parses_in_list_comprehension_position() {
    // Confirm `!=` parses inside a list comprehension's `if` guard;
    // execution semantics over path-collected sources is a separate
    // concern. The fix here is purely the grammar lookahead.
    use crate::parse::parser::parse;
    assert!(parse("[r for r in xs if r != 0]").is_ok());
}

#[test]
fn neq_combined_with_and_or() {
    let doc = json!({"xs": [1, 2, 3, 4]});
    assert_eq!(
        vm_query("$.xs.filter(@ != 1 and @ != 4)", &doc).unwrap(),
        json!([2, 3])
    );
    assert_eq!(
        vm_query("$.xs.filter(@ != 1 or @ != 4)", &doc).unwrap(),
        json!([1, 2, 3, 4]),
    );
}

#[test]
fn neq_inside_arrow_compound_predicate() {
    let doc = json!({"xs": [{"k": "a", "v": 1}, {"k": "b", "v": 2}, {"k": "a", "v": 3}]});
    assert_eq!(
        vm_query(r#"$.xs.filter(r => r.k != "b" and r.v > 0)"#, &doc).unwrap(),
        json!([{"k": "a", "v": 1}, {"k": "a", "v": 3}])
    );
}

#[test]
fn neq_does_not_break_one_shot_quantifier() {
    // The bare `!` postfix is the exactly-one quantifier — it must
    // still parse at EOI without being absorbed into `!=`.
    let doc = json!({"xs": [42]});
    assert_eq!(vm_query("$.xs!", &doc).unwrap(), json!([42]));
}

#[test]
fn neq_does_not_break_index_quantifier_chain() {
    // `[0]!` style postfix — quantifier after index must keep firing.
    let doc = json!({"xs": [42, 99]});
    assert_eq!(vm_query("$.xs[0]!", &doc).unwrap(), json!(42));
}

#[test]
fn neq_chained_with_fields() {
    let doc = json!({"xs": [{"u": {"id": 1}}, {"u": {"id": 2}}, {"u": {"id": 3}}]});
    assert_eq!(
        vm_query("$.xs.filter(r => r.u.id != 2).map(r => r.u.id)", &doc).unwrap(),
        json!([1, 3])
    );
}

#[test]
fn neq_at_form_inside_method_arg_chain() {
    let doc = json!({"xs": [{"k": 1}, {"k": 2}, {"k": 3}]});
    // `.find` returns the first match (conventional first-match
    // semantics on this branch).
    assert_eq!(
        vm_query("$.xs.find(@.k != 1)", &doc).unwrap(),
        json!({"k": 2})
    );
}

#[test]
fn neq_with_null() {
    let doc = json!({"xs": [{"v": 1}, {"v": null}, {"v": 3}]});
    assert_eq!(
        vm_query("$.xs.filter(@.v != null).map(r => r.v)", &doc).unwrap(),
        json!([1, 3])
    );
}

#[test]
fn neq_in_ternary_condition() {
    let doc = json!({"xs": [1, 2, 3]});
    assert_eq!(
        vm_query("$.xs.map(r => 99 if r != 2 else 0)", &doc).unwrap(),
        json!([99, 0, 99])
    );
}
