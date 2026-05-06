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
fn runtime_shared_two_key_prefix_dispatch() {
    // Every arm tests the same two leading keys (`tag` then `target`) in
    // the same order. The compiler hoists `ObjCheck` + two `LoadField`
    // ops out of the per-arm prologue.
    let v = run(
        br#"{"e": {"tag": "click", "target": "btn-a"}}"#,
        r#"match $.e with {
            {tag: "click", target: t} -> {sort: "click", t: t},
            {tag: "view",  target: t} -> {sort: "view",  t: t},
            _                         -> {sort: "other"}
        }"#,
    );
    assert_eq!(v, json!({"sort": "click", "t": "btn-a"}));
    let v = run(
        br#"{"e": {"tag": "view", "target": "/home"}}"#,
        r#"match $.e with {
            {tag: "click", target: t} -> {sort: "click", t: t},
            {tag: "view",  target: t} -> {sort: "view",  t: t},
            _                         -> {sort: "other"}
        }"#,
    );
    assert_eq!(v, json!({"sort": "view", "t": "/home"}));
}

#[test]
fn runtime_shared_arr_len_dispatch() {
    // Every arm is a fixed-length array pattern of the same length, so
    // the compiler hoists the `LenCheck` into a single match-level
    // prelude and emits only `LoadIndex` ops per arm.
    let v = run(
        br#"{"p": [1, 2]}"#,
        r#"match $.p with {
            [0, x] -> {tag: "zero", x: x},
            [1, x] -> {tag: "one",  x: x},
            [_, x] -> {tag: "any",  x: x}
        }"#,
    );
    assert_eq!(v, json!({"tag": "one", "x": 2}));
}

#[test]
fn runtime_shared_arr_len_disabled_when_lengths_differ() {
    // Arm 0 has length 2, arm 1 has length 3 — sharing is disabled and
    // each arm runs its own `LenCheck`.
    let v = run(
        br#"{"p": [1, 2, 3]}"#,
        r#"match $.p with {
            [a, b]    -> {tag: "two", a: a},
            [a, b, c] -> {tag: "three", c: c},
            _         -> {tag: "other"}
        }"#,
    );
    assert_eq!(v, json!({"tag": "three", "c": 3}));
}

#[test]
fn runtime_shared_prefix_with_trailing_catchall_routes_prelude_failure() {
    // Mixing typed `Pat::Obj` arms with a trailing wildcard catch-all
    // still enables shared-prefix sharing; prelude failures route to
    // the catch-all arm rather than the global `Fail` op.
    // Scrutinee is not an object — `ObjCheck` in the prelude fails and
    // jumps to the catch-all body.
    let v = run(
        br#"{"u": 42}"#,
        r#"match $.u with {
            {role: "admin"} -> "admin",
            {role: "user"}  -> "user",
            _               -> "fallback"
        }"#,
    );
    assert_eq!(v, json!("fallback"));

    // Object missing the shared key — prelude `LoadField` fails and
    // routes to the catch-all arm.
    let v = run(
        br#"{"u": {"name": "alice"}}"#,
        r#"match $.u with {
            {role: "admin"} -> "admin",
            {role: "user"}  -> "user",
            _               -> "fallback"
        }"#,
    );
    assert_eq!(v, json!("fallback"));

    // Object with shared key but no typed arm matches — last typed arm's
    // miss routes to the catch-all (via the standard pending-else patch).
    let v = run(
        br#"{"u": {"role": "guest"}}"#,
        r#"match $.u with {
            {role: "admin"} -> "admin",
            {role: "user"}  -> "user",
            _               -> "fallback"
        }"#,
    );
    assert_eq!(v, json!("fallback"));
}

#[test]
fn runtime_arr_length_share_with_trailing_catchall() {
    // Same idea for the array-length sharing path: a trailing wildcard
    // catch-all participates without disabling the shared `LenCheck`.
    let v = run(
        br#"{"p": [1, 2, 3]}"#,
        r#"match $.p with {
            [a, b]    -> {tag: "two", a: a},
            [a, b, _] -> {tag: "three", a: a},
            _         -> {tag: "other"}
        }"#,
    );
    // First two arms have lengths 2 and 3 — sharing disabled. Catch-all
    // present but not used. Test exercises the no-sharing-with-catchall
    // path for completeness.
    assert_eq!(v, json!({"tag": "three", "a": 1}));

    // All typed arms agree on length 3, plus a trailing catch-all.
    let v = run(
        br#"{"p": [1, 2]}"#,
        r#"match $.p with {
            [a, _, _] -> {tag: "first",  a: a},
            [_, b, _] -> {tag: "second", b: b},
            _         -> {tag: "other"}
        }"#,
    );
    // Scrutinee has length 2, prelude `LenCheck { len: 3, exact: true }`
    // fails and routes to the catch-all.
    assert_eq!(v, json!({"tag": "other"}));
}

#[test]
fn runtime_non_exhaustive_error_includes_value_snippet() {
    // The trailing `Fail` op renders a short snippet of the scrutinee
    // alongside its kind so users can spot the offending value.
    let err = run_err(
        br#"{"x": 7}"#,
        r#"match $.x with { 1 -> "a", 2 -> "b" }"#,
    );
    assert!(err.contains("no arm matched"), "{err}");
    assert!(err.contains("7"), "expected scrutinee value in error: {err}");
}

#[test]
fn runtime_shared_prefix_disabled_on_key_order_divergence() {
    // The two arms list the same keys but in different orders. The
    // shared-prefix optimization is disabled (common prefix length 0)
    // and per-arm codegen handles every key independently.
    let v = run(
        br#"{"u": {"role": "admin", "id": 9}}"#,
        r#"match $.u with {
            {role: r, id: i} -> {r: r, i: i},
            {id: i, role: r} -> {r: r, i: i}
        }"#,
    );
    assert_eq!(v, json!({"r": "admin", "i": 9}));
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
fn runtime_filter_match_take_demand_cuts_through() {
    // Pipeline: `.filter(match @ ...) | take(3)` should emit only the
    // first three matches without exhausting the source. The chain IR
    // recognises `Stage::Filter` whose body is a single `Opcode::Match`
    // as `ChainOp::Match { Predicate }` so demand widens to
    // `UntilOutput(3)` instead of bounded `FirstInput`.
    let src = br#"{"xs": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}"#;
    let v = run(
        src,
        r#"$.xs.filter(match @ with {
            n when n > 2 -> true,
            _            -> false
        }).take(3)"#,
    );
    assert_eq!(v, json!([3, 4, 5]));
}

#[test]
fn runtime_map_match_passthrough() {
    // Pipeline: `.map(match @ ...)` is 1:1 — every input row produces
    // exactly one output. Used as a tagged-decoder over an array.
    let src = br#"{"events": [
        {"tag": "view", "p": "/home"},
        {"tag": "click", "id": "btn"},
        {"tag": "error", "code": 500}
    ]}"#;
    let v = run(
        src,
        r#"$.events.map(match @ with {
            {tag: "view",  p: x}    -> {sort: "view",  v: x},
            {tag: "click", id: x}   -> {sort: "click", v: x},
            {tag: "error", code: x} -> {sort: "error", v: x},
            _                       -> {sort: "other"}
        })"#,
    );
    assert_eq!(
        v,
        json!([
            {"sort": "view",  "v": "/home"},
            {"sort": "click", "v": "btn"},
            {"sort": "error", "v": 500}
        ])
    );
}

#[test]
fn runtime_range_pattern_inclusive() {
    // `1..=10` matches integers 1 through 10 inclusive.
    let cases = [
        (1, "low"),
        (5, "low"),
        (10, "low"),
        (11, "high"),
        (0, "high"),
    ];
    for (n, expected) in cases {
        let v = run(
            format!(r#"{{"x": {n}}}"#).as_bytes(),
            r#"match $.x with {
                1..=10 -> "low",
                _      -> "high"
            }"#,
        );
        assert_eq!(v, json!(expected), "n={n}");
    }
}

#[test]
fn runtime_range_pattern_exclusive() {
    // `1..10` matches integers 1 through 9; 10 falls into the catch-all.
    let cases = [(1, "lo"), (9, "lo"), (10, "hi")];
    for (n, expected) in cases {
        let v = run(
            format!(r#"{{"x": {n}}}"#).as_bytes(),
            r#"match $.x with {
                1..10 -> "lo",
                _     -> "hi"
            }"#,
        );
        assert_eq!(v, json!(expected), "n={n}");
    }
}

#[test]
fn runtime_range_pattern_float() {
    // Floats accepted as bounds and as scrutinees.
    let v = run(
        br#"{"x": 0.5}"#,
        r#"match $.x with {
            0.0..=1.0 -> "unit",
            _         -> "other"
        }"#,
    );
    assert_eq!(v, json!("unit"));
}

#[test]
fn runtime_range_pattern_int_widens_to_float() {
    // Integer scrutinee widens to f64 for range comparison.
    let v = run(
        br#"{"x": 3}"#,
        r#"match $.x with {
            0.0..5.0 -> "in",
            _        -> "out"
        }"#,
    );
    assert_eq!(v, json!("in"));
}

#[test]
fn runtime_range_pattern_inside_or() {
    // Range patterns compose with or-patterns via the tree-walker path.
    let v = run(
        br#"{"x": 7}"#,
        r#"match $.x with {
            1..=3 | 5..=10 -> "ok",
            _              -> "no"
        }"#,
    );
    assert_eq!(v, json!("ok"));
}

#[test]
fn runtime_range_pattern_non_numeric_skipped() {
    // String scrutinee falls through to catch-all because RangeCheck
    // requires a numeric value.
    let v = run(
        br#"{"x": "hello"}"#,
        r#"match $.x with {
            1..=10 -> "num",
            _      -> "other"
        }"#,
    );
    assert_eq!(v, json!("other"));
}

#[test]
fn runtime_range_pattern_inside_object() {
    // Range patterns work inside object sub-pattern positions.
    let v = run(
        br#"{"u": {"age": 65}}"#,
        r#"match $.u with {
            {age: 0..=17}    -> "minor",
            {age: 18..=64}   -> "adult",
            {age: 65..=120}  -> "senior",
            _                -> "unknown"
        }"#,
    );
    assert_eq!(v, json!("senior"));
}

#[test]
fn runtime_shared_kind_dispatch() {
    // Every typed arm is a `Pat::Kind` testing the same scalar kind, so
    // the compiler hoists the `KindCheck` into a single prelude. Each
    // arm just pushes its binding then runs the guard / body.
    let v = run(
        br#"{"x": "hello"}"#,
        r#"match $.x with {
            s: string when s.len() > 3 -> "long",
            s: string                  -> "short",
            _                          -> "other"
        }"#,
    );
    assert_eq!(v, json!("long"));
    let v = run(
        br#"{"x": "hi"}"#,
        r#"match $.x with {
            s: string when s.len() > 3 -> "long",
            s: string                  -> "short",
            _                          -> "other"
        }"#,
    );
    assert_eq!(v, json!("short"));
    let v = run(
        br#"{"x": 42}"#,
        r#"match $.x with {
            s: string when s.len() > 3 -> "long",
            s: string                  -> "short",
            _                          -> "other"
        }"#,
    );
    assert_eq!(v, json!("other"));
}

#[test]
fn runtime_range_pattern_negative_bounds() {
    let cases = [(-5, "neg"), (-1, "neg"), (0, "zero"), (10, "pos")];
    for (n, expected) in cases {
        let v = run(
            format!(r#"{{"x": {n}}}"#).as_bytes(),
            r#"match $.x with {
                -10..0 -> "neg",
                0      -> "zero",
                _      -> "pos"
            }"#,
        );
        assert_eq!(v, json!(expected), "n={n}");
    }
}

#[test]
fn view_domain_runtime_runs_against_borrowed_view() {
    // Compile a match expression, then dispatch the view-domain runtime
    // (`exec_match_view`) directly against a `ValView` borrow of the
    // scrutinee. This exercises the pattern test path that runs against
    // borrowed scalars / sub-projections without materialising the
    // missed-arm subtree.
    use crate::compile::compiler::Compiler;
    use crate::data::context::Env;
    use crate::data::value::Val as DVal;
    use crate::data::view::ValView;
    use crate::parse::parser::parse;
    use crate::vm::{Opcode, VM};

    // Build the AST and locate the embedded `Match` opcode by compiling
    // the expression and inspecting its op stream.
    let expr = parse(r#"match @ with {
        {role: "admin", id: i} -> {sort: "admin", n: i},
        {role: "user", id: i}  -> {sort: "user", n: i},
        {role: r}              -> {sort: r, n: 0}
    }"#)
    .expect("parse");
    let prog = Compiler::compile(&expr, "match-view-test");
    let cm = prog
        .ops
        .iter()
        .find_map(|op| match op {
            Opcode::Match(cm) => Some(cm.clone()),
            _ => None,
        })
        .expect("compiled match opcode");

    // Drive `exec_match_view` directly against a borrowed view of the
    // input object. The runtime should pick the first arm and produce
    // the corresponding output object.
    let scrutinee_val: DVal = DVal::from(&json!({"role": "admin", "id": 9}));
    let view = ValView::new(&scrutinee_val);
    let mut vm = VM::new();
    let env = Env::new(scrutinee_val.clone());
    let result = crate::vm::exec::exec_match_view(&mut vm, &cm, view, &env)
        .expect("view-domain match should succeed");
    let as_json: serde_json::Value = result.into();
    assert_eq!(as_json, json!({"sort": "admin", "n": 9}));
}

#[test]
fn runtime_partial_shared_prefix_with_mixed_suffix() {
    // The first two arms share a leading object key (`role`); the
    // remaining arms have unrelated shapes (length-2 array, raw int,
    // catch-all). Cross-arm sharing now hoists the `ObjCheck +
    // LoadField` for the leading run while later arms run their own
    // standard codegen.
    //
    // Object scrutinee — admin: shared run hits arm 0.
    let v = run(
        br#"{"x": {"role": "admin", "id": 1}}"#,
        r#"match $.x with {
            {role: "admin"} -> "admin",
            {role: "user"}  -> "user",
            [a, b]          -> "pair",
            42              -> "answer",
            _               -> "other"
        }"#,
    );
    assert_eq!(v, json!("admin"));

    // Array scrutinee — prelude `LoadField` would fail; jumps to arm 2
    // (first non-shared arm) which matches.
    let v = run(
        br#"{"x": [1, 2]}"#,
        r#"match $.x with {
            {role: "admin"} -> "admin",
            {role: "user"}  -> "user",
            [a, b]          -> "pair",
            42              -> "answer",
            _               -> "other"
        }"#,
    );
    assert_eq!(v, json!("pair"));

    // Integer scrutinee — falls through past arrays to arm 3.
    let v = run(
        br#"{"x": 42}"#,
        r#"match $.x with {
            {role: "admin"} -> "admin",
            {role: "user"}  -> "user",
            [a, b]          -> "pair",
            42              -> "answer",
            _               -> "other"
        }"#,
    );
    assert_eq!(v, json!("answer"));

    // String scrutinee — falls through to wildcard catch-all.
    let v = run(
        br#"{"x": "hello"}"#,
        r#"match $.x with {
            {role: "admin"} -> "admin",
            {role: "user"}  -> "user",
            [a, b]          -> "pair",
            42              -> "answer",
            _               -> "other"
        }"#,
    );
    assert_eq!(v, json!("other"));
}

#[test]
fn runtime_deep_match_collects_truthy_arm_bodies() {
    // `..match { arms }` walks every descendant in DFS order and
    // collects the truthy arm-body results. The catch-all arm here
    // returns `false`, so descendants that hit it are dropped from
    // the output.
    let src = br#"{
        "page": "/home",
        "events": [
            {"tag": "click", "id": "btn-a"},
            {"tag": "view", "url": "/about"},
            {"nested": {"tag": "click", "id": "btn-b"}}
        ],
        "meta": {"title": "site"}
    }"#;
    let v = run(
        src,
        r#"$..match {
            {tag: "click", id: i} -> i,
            _                     -> false
        }"#,
    );
    // Two click events match — IDs in DFS order.
    assert_eq!(v, json!(["btn-a", "btn-b"]));
}

#[test]
fn runtime_deep_match_with_range_pattern() {
    // Range pattern over numeric descendants: collect every number in
    // 100..=999 anywhere in the tree.
    let src = br#"{
        "stats": {"a": 50, "b": 250, "c": 1500},
        "more":  [42, 200, 800, 9999]
    }"#;
    let v = run(
        src,
        r#"$..match {
            n: number when n >= 100 and n < 1000 -> n,
            _                                     -> false
        }"#,
    );
    assert_eq!(v, json!([250, 200, 800]));
}

#[test]
fn runtime_deep_match_first_returns_first_truthy() {
    // `..match! { arms }` aborts the descent at the first truthy
    // arm-body and returns it directly (not wrapped in an array).
    let src = br#"{
        "events": [
            {"tag": "view", "id": "a"},
            {"tag": "click", "id": "b"},
            {"tag": "click", "id": "c"}
        ]
    }"#;
    let v = run(
        src,
        r#"$..match! {
            {tag: "click", id: i} -> i,
            _                     -> false
        }"#,
    );
    assert_eq!(v, json!("b"));
}

#[test]
fn runtime_deep_match_first_returns_null_when_nothing_matches() {
    let v = run(
        br#"{"a": 1, "b": "x"}"#,
        r#"$..match! {
            {role: "admin"} -> "found",
            _               -> false
        }"#,
    );
    assert_eq!(v, json!(null));
}

#[test]
fn runtime_deep_match_no_matches_returns_empty() {
    let v = run(
        br#"{"a": 1, "b": 2}"#,
        r#"$..match {
            {role: "admin"} -> "found",
            _               -> false
        }"#,
    );
    assert_eq!(v, json!([]));
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
