//! Pattern-match parser and runtime tests.

use crate::parse::ast::{Expr, KindType, Pat, PatLit};
use crate::parse::parser::parse;
use crate::Jetro;
use serde_json::json;

fn run(json_src: &[u8], expr: &str) -> serde_json::Value {
    let j = Jetro::from_bytes(json_src.to_vec()).expect("json parse");
    j.collect(expr).expect("eval")
}

fn run_err(json_src: &[u8], expr: &str) -> String {
    let j = Jetro::from_bytes(json_src.to_vec()).expect("json parse");
    j.collect(expr).err().expect("expected error").to_string()
}

fn parse_match(src: &str) -> (Expr, Vec<crate::parse::ast::MatchArm>) {
    match parse(src).expect("parse ok") {
        Expr::Match { scrutinee, arms } => (*scrutinee, arms),
        other => panic!("expected Match, got {other:?}"),
    }
}

#[test]
fn parses_wildcard_arm() {
    let (_, arms) = parse_match("match $.x with { _ -> 1 }");
    assert_eq!(arms.len(), 1);
    assert!(matches!(arms[0].pat, Pat::Wild));
    assert!(arms[0].guard.is_none());
}

#[test]
fn parses_literal_arms() {
    let (_, arms) = parse_match(
        r#"match $.k with {
            null    -> 0,
            true    -> 1,
            "x"     -> 2,
            42      -> 3,
            3.14    -> 4,
            _       -> 5
        }"#,
    );
    assert_eq!(arms.len(), 6);
    assert!(matches!(arms[0].pat, Pat::Lit(PatLit::Null)));
    assert!(matches!(arms[1].pat, Pat::Lit(PatLit::Bool(true))));
    assert!(matches!(arms[2].pat, Pat::Lit(PatLit::Str(ref s)) if s == "x"));
    assert!(matches!(arms[3].pat, Pat::Lit(PatLit::Int(42))));
    assert!(matches!(arms[4].pat, Pat::Lit(PatLit::Float(_))));
    assert!(matches!(arms[5].pat, Pat::Wild));
}

#[test]
fn parses_object_pattern() {
    let (_, arms) = parse_match(r#"match $.u with { {role: "admin"} -> 1, _ -> 0 }"#);
    let Pat::Obj { fields, open } = &arms[0].pat else {
        panic!("expected Obj pattern");
    };
    assert!(!open);
    assert_eq!(fields.len(), 1);
    assert_eq!(fields[0].0, "role");
    assert!(matches!(fields[0].1, Pat::Lit(PatLit::Str(ref s)) if s == "admin"));
}

#[test]
fn parses_object_open_pattern() {
    let (_, arms) = parse_match(r#"match $.u with { {role: "admin", ...} -> 1, _ -> 0 }"#);
    let Pat::Obj { open, .. } = &arms[0].pat else {
        panic!("expected Obj pattern");
    };
    assert!(*open);
}

#[test]
fn parses_array_pattern_with_rest() {
    let (_, arms) = parse_match(r#"match $.xs with { [a, b, ...rest] -> a, _ -> 0 }"#);
    let Pat::Arr { elems, rest } = &arms[0].pat else {
        panic!("expected Arr pattern");
    };
    assert_eq!(elems.len(), 2);
    assert!(matches!(rest, Some(Some(ref s)) if s == "rest"));
}

#[test]
fn parses_or_pattern() {
    let (_, arms) = parse_match(r#"match $.m with { "GET" | "HEAD" -> 1, _ -> 0 }"#);
    let Pat::Or(alts) = &arms[0].pat else {
        panic!("expected Or pattern");
    };
    assert_eq!(alts.len(), 2);
}

#[test]
fn parses_kind_bind_pattern() {
    let (_, arms) = parse_match(r#"match $.v with { s: string -> s, _ -> "no" }"#);
    let Pat::Kind { name, kind } = &arms[0].pat else {
        panic!("expected Kind pattern");
    };
    assert!(matches!(name, Some(ref n) if n == "s"));
    assert!(matches!(kind, KindType::Str));
}

#[test]
fn parses_guard_arm() {
    let (_, arms) =
        parse_match(r#"match $.x with { n when n > 10 -> "big", _ -> "small" }"#);
    assert!(arms[0].guard.is_some());
    assert!(arms[1].guard.is_none());
}

#[test]
fn parses_bind_only_pattern() {
    let (_, arms) = parse_match(r#"match $.x with { v -> v }"#);
    assert!(matches!(arms[0].pat, Pat::Bind(ref n) if n == "v"));
}

#[test]
fn runtime_wildcard_returns_body() {
    let v = run(br#"{"x": 1}"#, r#"match $.x with { _ -> "any" }"#);
    assert_eq!(v, json!("any"));
}

#[test]
fn runtime_literal_dispatch() {
    let src = br#"{"k": "ok"}"#;
    assert_eq!(
        run(src, r#"match $.k with { "ok" -> 1, _ -> 0 }"#),
        json!(1)
    );
    assert_eq!(
        run(src, r#"match $.k with { "no" -> 1, _ -> 0 }"#),
        json!(0)
    );
}

#[test]
fn runtime_int_literal_dispatch() {
    let src = br#"{"n": 42}"#;
    assert_eq!(
        run(src, r#"match $.n with { 1 -> "one", 42 -> "answer", _ -> "?" }"#),
        json!("answer")
    );
}

#[test]
fn runtime_or_pattern() {
    let src = br#"{"m": "HEAD"}"#;
    assert_eq!(
        run(
            src,
            r#"match $.m with { "GET" | "HEAD" -> "safe", _ -> "other" }"#
        ),
        json!("safe")
    );
}

#[test]
fn runtime_object_pattern_partial_match() {
    let src = br#"{"u": {"role": "admin", "id": 9}}"#;
    let v = run(
        src,
        r#"match $.u with { {role: "admin"} -> "ok", _ -> "no" }"#,
    );
    assert_eq!(v, json!("ok"));
}

#[test]
fn runtime_array_rest_binding() {
    let src = br#"{"xs": [1, 2, 3, 4]}"#;
    let v = run(
        src,
        r#"match $.xs with { [a, b, ...rest] -> rest, _ -> [] }"#,
    );
    assert_eq!(v, json!([3, 4]));
}

#[test]
fn runtime_kind_bind() {
    let src = br#"{"v": "hello"}"#;
    let v = run(
        src,
        r#"match $.v with { s: string -> s, _ -> "other" }"#,
    );
    assert_eq!(v, json!("hello"));
}

#[test]
fn runtime_guard_filters_arm() {
    let src = br#"{"x": 5}"#;
    assert_eq!(
        run(
            src,
            r#"match $.x with { n when n > 10 -> "big", n -> "small" }"#
        ),
        json!("small")
    );
    let src = br#"{"x": 50}"#;
    assert_eq!(
        run(
            src,
            r#"match $.x with { n when n > 10 -> "big", n -> "small" }"#
        ),
        json!("big")
    );
}

#[test]
fn runtime_bind_captures_value() {
    let src = br#"{"x": 7}"#;
    assert_eq!(run(src, "match $.x with { v -> v }"), json!(7));
}

#[test]
fn runtime_no_arm_match_is_error() {
    let err = run_err(br#"{"x": 1}"#, r#"match $.x with { 99 -> 0 }"#);
    assert!(err.contains("match"), "got: {err}");
    assert!(err.contains("no arm matched") || err.contains("no arm"), "got: {err}");
}

#[test]
fn runtime_nested_object_pattern() {
    let src = br#"{"e": {"type": "click", "target": {"tag": "a"}}}"#;
    let v = run(
        src,
        r#"match $.e with {
            {type: "click", target: {tag: "a"}} -> "anchor",
            {type: "click"} -> "click",
            _ -> "other"
        }"#,
    );
    assert_eq!(v, json!("anchor"));
}
