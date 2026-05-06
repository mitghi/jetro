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
fn runtime_first_arm_wins() {
    // Earlier arms shadow later ones even when both would match.
    let v = run(
        br#"{"x": 1}"#,
        r#"match $.x with { _ -> "first", _ -> "second" }"#,
    );
    assert_eq!(v, json!("first"));
}

#[test]
fn runtime_object_pattern_shadowing_keys() {
    // Arms that share a prefix key still run independently; first match wins.
    let src = br#"{"u": {"role": "user", "id": 9}}"#;
    let v = run(
        src,
        r#"match $.u with {
            {role: "admin"} -> "admin",
            {role: "user"}  -> "user",
            _               -> "other"
        }"#,
    );
    assert_eq!(v, json!("user"));
}

#[test]
fn runtime_array_exact_length() {
    // No rest pattern means length must match exactly.
    let v = run(
        br#"{"xs": [1, 2]}"#,
        r#"match $.xs with { [a, b] -> a, _ -> 0 }"#,
    );
    assert_eq!(v, json!(1));
    let v = run(
        br#"{"xs": [1, 2, 3]}"#,
        r#"match $.xs with { [a, b] -> a, _ -> 99 }"#,
    );
    assert_eq!(v, json!(99));
}

#[test]
fn runtime_or_pattern_inside_object() {
    let src = br#"{"r": {"method": "POST"}}"#;
    let v = run(
        src,
        r#"match $.r with {
            {method: "GET" | "HEAD"}        -> "safe",
            {method: "POST" | "PUT"}        -> "write",
            _                               -> "other"
        }"#,
    );
    assert_eq!(v, json!("write"));
}

#[test]
fn runtime_kind_only_pattern() {
    let v = run(
        br#"{"v": 42}"#,
        r#"match $.v with { number -> "num", _ -> "other" }"#,
    );
    assert_eq!(v, json!("num"));
    let v = run(
        br#"{"v": "x"}"#,
        r#"match $.v with { number -> "num", _ -> "other" }"#,
    );
    assert_eq!(v, json!("other"));
}

#[test]
fn runtime_wildcard_inside_array() {
    let v = run(
        br#"{"xs": [1, 2, 3]}"#,
        r#"match $.xs with { [_, mid, _] -> mid, _ -> 0 }"#,
    );
    assert_eq!(v, json!(2));
}

#[test]
fn runtime_int_to_float_literal_coerces() {
    // Integer pattern literal should match a Val::Float with the same value.
    let v = run(
        br#"{"n": 1.0}"#,
        r#"match $.n with { 1 -> "one", _ -> "other" }"#,
    );
    assert_eq!(v, json!("one"));
}

#[test]
fn runtime_guard_sees_arm_bindings() {
    // Guard expressions can reference bindings introduced by the pattern.
    let v = run(
        br#"{"u": {"name": "alice", "age": 31}}"#,
        r#"match $.u with {
            {name: n, age: a} when a >= 18 -> n,
            _                              -> "minor"
        }"#,
    );
    assert_eq!(v, json!("alice"));
}

#[test]
fn runtime_bindings_reset_between_arms() {
    // A failed arm must not leak its bindings into later arm bodies.
    let v = run(
        br#"{"x": 5}"#,
        r#"match $.x with {
            n when n > 100 -> n,
            v              -> v
        }"#,
    );
    assert_eq!(v, json!(5));
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

#[test]
fn runtime_match_inside_map() {
    // Match used as the body of a `.map` lambda over an array.
    // Field `kind` is a reserved keyword in the jetro grammar, so the
    // events here use `tag` instead.
    let src = br#"{"events": [
        {"tag": "view", "path": "/home"},
        {"tag": "click", "id": "btn"},
        {"tag": "error", "code": 500}
    ]}"#;
    let v = run(
        src,
        r#"$.events.map(match @ with {
            {tag: "view", path: p}  -> p,
            {tag: "click", id: i}   -> i,
            {tag: "error", code: c} -> c,
            _                       -> null
        })"#,
    );
    assert_eq!(v, json!(["/home", "btn", 500]));
}

#[test]
fn runtime_match_inside_filter_predicate() {
    // Match used as a boolean predicate inside `.filter`.
    let src = br#"{"reqs": [
        {"method": "GET"},
        {"method": "POST"},
        {"method": "DELETE"}
    ]}"#;
    let v = run(
        src,
        r#"$.reqs.filter(match @ with {
            {method: "GET" | "HEAD"} -> true,
            _                        -> false
        })"#,
    );
    assert_eq!(v, json!([{"method": "GET"}]));
}

#[test]
fn runtime_match_inside_pipeline() {
    // Match used as the right-hand side of a `|` pipeline step.
    let v = run(
        br#"{"x": 7}"#,
        r#"$.x | (match @ with { n when n > 5 -> "big", _ -> "small" })"#,
    );
    assert_eq!(v, json!("big"));
}

#[test]
fn runtime_match_returns_object() {
    let src = br#"{"u": {"name": "alice", "role": "admin"}}"#;
    let v = run(
        src,
        r#"match $.u with {
            {role: "admin", name: n} -> {tag: "admin", name: n},
            _                        -> {tag: "other"}
        }"#,
    );
    assert_eq!(v, json!({"tag": "admin", "name": "alice"}));
}

#[test]
fn runtime_shared_first_key_dispatch() {
    // Tagged-union dispatch: every arm tests the same first object key.
    // The compiler hoists the `ObjCheck + LoadField` for that key out of
    // every per-arm prologue; correctness here exercises the shared path.
    let cases = [
        (br#"{"u": {"role": "admin", "id": 1}}"# as &[u8], json!({"k": "admin", "n": 1})),
        (br#"{"u": {"role": "user",  "id": 2}}"#, json!({"k": "user", "n": 2})),
        (br#"{"u": {"role": "guest", "id": 3}}"#, json!({"k": "guest", "n": 0})),
    ];
    for (src, expected) in cases {
        let v = run(
            src,
            r#"match $.u with {
                {role: "admin", id: i} -> {k: "admin", n: i},
                {role: "user", id: i}  -> {k: "user",  n: i},
                {role: r}              -> {k: r, n: 0}
            }"#,
        );
        assert_eq!(v, expected, "src={}", std::str::from_utf8(src).unwrap());
    }
}

#[test]
fn runtime_shared_key_preserves_arm_bindings() {
    // The key value loaded by the shared prelude (slot 1) must remain
    // addressable inside per-arm bodies via the K-field sub-pattern.
    let v = run(
        br#"{"u": {"role": "admin", "level": 9}}"#,
        r#"match $.u with {
            {role: r, level: l} when l > 5 -> r,
            {role: r}                      -> r
        }"#,
    );
    assert_eq!(v, json!("admin"));
}

#[test]
fn runtime_shared_key_no_arm_match_falls_through_to_fail() {
    // No arm tests "missing" so a value with that role hits the trailing
    // `Fail` op via the per-arm miss path (not the prelude's else_pc).
    let err = run_err(
        br#"{"u": {"role": "missing"}}"#,
        r#"match $.u with {
            {role: "admin"} -> 1,
            {role: "user"}  -> 2
        }"#,
    );
    assert!(err.contains("no arm matched"), "{err}");
}

#[test]
fn runtime_shared_key_prelude_failure_when_not_object() {
    // Scrutinee is not an object, so the shared prelude's `ObjCheck`
    // jumps directly to `Fail` without entering any arm.
    let err = run_err(
        br#"{"u": 42}"#,
        r#"match $.u with {
            {role: "admin"} -> 1,
            {role: "user"}  -> 2
        }"#,
    );
    assert!(err.contains("no arm matched"), "{err}");
}

#[test]
fn runtime_shared_key_prelude_failure_when_key_missing() {
    // Scrutinee is an object but the shared key is absent — prelude's
    // `LoadField` jumps to `Fail`.
    let err = run_err(
        br#"{"u": {"name": "alice"}}"#,
        r#"match $.u with {
            {role: "admin"} -> 1,
            {role: "user"}  -> 2
        }"#,
    );
    assert!(err.contains("no arm matched"), "{err}");
}

#[test]
fn parse_rejects_or_pattern_with_inconsistent_bindings() {
    let err = parse(r#"match $.x with { {a: x} | {b: y} -> x }"#)
        .err()
        .expect("expected linearity error");
    let s = err.to_string();
    assert!(s.contains("or-pattern arms must bind"), "got: {s}");
}

#[test]
fn parse_accepts_or_pattern_with_matching_bindings() {
    // Both alts bind `n`; linearity holds.
    let v = run(
        br#"{"e": {"warn": 5}}"#,
        r#"match $.e with {
            {warn: n} | {error: n} -> n,
            _                      -> 0
        }"#,
    );
    assert_eq!(v, json!(5));
}

#[test]
fn runtime_or_of_literals_cascade_three_alts() {
    // Three-way or-pattern over literals exercises the flat cascade path
    // (LitEq + Jump + LitEq + Jump + LitEq) emitted by the compiler when
    // every alt is a `Pat::Lit`.
    let cases = [
        ("a", json!("first")),
        ("b", json!("first")),
        ("c", json!("first")),
        ("d", json!("other")),
    ];
    for (k, expected) in cases {
        let v = run(
            format!(r#"{{"k": "{k}"}}"#).as_bytes(),
            r#"match $.k with {
                "a" | "b" | "c" -> "first",
                _               -> "other"
            }"#,
        );
        assert_eq!(v, expected, "k={k}");
    }
}

#[test]
fn runtime_or_of_literals_with_int_alts() {
    let v = run(
        br#"{"n": 2}"#,
        r#"match $.n with {
            1 | 2 | 3 -> "small",
            _         -> "big"
        }"#,
    );
    assert_eq!(v, json!("small"));
}

#[test]
fn parse_accepts_literal_only_or_pattern() {
    // Linearity is trivial when no alt binds anything.
    let v = run(
        br#"{"m": "PUT"}"#,
        r#"match $.m with {
            "GET" | "HEAD"   -> "safe",
            "POST" | "PUT"   -> "write",
            _                -> "other"
        }"#,
    );
    assert_eq!(v, json!("write"));
}

#[test]
fn runtime_match_chained_with_postfix() {
    // Result of a match flows into a subsequent method call.
    let src = br#"{"xs": [1, 2, 3]}"#;
    let v = run(
        src,
        r#"(match $.xs with { [_, ...rest] -> rest, _ -> [] }).len()"#,
    );
    assert_eq!(v, json!(2));
}
