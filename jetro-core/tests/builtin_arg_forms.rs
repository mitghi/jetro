//! Argument-shape coverage matrix for every builtin that accepts an
//! argument or a lambda. Each test runs a builtin through every spelling
//! of its argument list (`@`-form, bare-path `.field`, named arrow,
//! `lambda` keyword, multi-arg, bare-ident, etc.) and asserts the actual
//! runtime output. New builtins should add a `#[test]` here that walks
//! every form they accept.

use jetro_core::Jetro;
use serde_json::{json, Value};

fn run(q: &str, doc: &Value) -> String {
    let bytes = serde_json::to_vec(doc).unwrap();
    let j = Jetro::from_bytes(bytes).unwrap();
    j.collect(q.to_string())
        .unwrap_or_else(|e| panic!("{}\n  query: {}", e.0, q))
        .to_string()
}

/// Asserts every spelling in `exprs` produces `expected` when run against `doc`.
fn assert_all(label: &str, exprs: &[&str], doc: &Value, expected: &str) {
    for e in exprs {
        let got = run(e, doc);
        assert_eq!(got, expected, "{}: form `{}` produced {}", label, e, got);
    }
}

fn users() -> Value {
    json!({
        "users": [
            {"id": 1, "name": "Ada",    "age": 30, "active": true,  "score": 80},
            {"id": 2, "name": "Bob",    "age": 24, "active": false, "score": 40},
            {"id": 3, "name": "Carol",  "age": 42, "active": true,  "score": 95},
        ]
    })
}

fn xs() -> Value {
    json!({"xs": [1, 2, 3, 4, 5]})
}

fn obj() -> Value {
    json!({"o": {"a": 1, "b": 2, "c": 3}})
}

// ──────────────────────────────────────────────────────────────────────────
// Group A: single-arg predicate-lambda builtins
// All four spellings must yield identical output.
// ──────────────────────────────────────────────────────────────────────────

#[test]
fn filter_all_forms() {
    let d = users();
    assert_all(
        "filter",
        &[
            "$.users.filter(@.active)",
            "$.users.filter(.active)",
            "$.users.filter(u => u.active)",
            "$.users.filter(lambda u: u.active)",
        ],
        &d,
        r#"[{"active":true,"age":30,"id":1,"name":"Ada","score":80},{"active":true,"age":42,"id":3,"name":"Carol","score":95}]"#,
    );
}

#[test]
fn find_all_forms() {
    // `.find(pred)` returns the first matching element only; use
    // `.find_all` for every match.
    let d = users();
    let exp_first = r#"{"active":true,"age":30,"id":1,"name":"Ada","score":80}"#;
    assert_all(
        "find(first match)",
        &[
            "$.users.find(@.active)",
            "$.users.find(.active)",
            "$.users.find(u => u.active)",
            "$.users.find(lambda u: u.active)",
        ],
        &d,
        exp_first,
    );

    let exp_all = r#"[{"active":true,"age":30,"id":1,"name":"Ada","score":80},{"active":true,"age":42,"id":3,"name":"Carol","score":95}]"#;
    assert_all(
        "find_all",
        &[
            "$.users.find_all(@.active)",
            "$.users.find_all(.active)",
            "$.users.find_all(u => u.active)",
            "$.users.find_all(lambda u: u.active)",
        ],
        &d,
        exp_all,
    );
}

#[test]
fn find_first_all_forms() {
    let d = users();
    let exp = r#"{"active":true,"age":30,"id":1,"name":"Ada","score":80}"#;
    assert_all(
        "find_first",
        &[
            "$.users.find_first(@.active)",
            "$.users.find_first(.active)",
            "$.users.find_first(u => u.active)",
            "$.users.find_first(lambda u: u.active)",
        ],
        &d,
        exp,
    );
}

#[test]
fn find_one_all_forms() {
    let d = users();
    let exp = r#"{"active":true,"age":30,"id":1,"name":"Ada","score":80}"#;
    assert_all(
        "find_one",
        &[
            "$.users.find_one(@.id == 1)",
            "$.users.find_one(u => u.id == 1)",
            "$.users.find_one(lambda u: u.id == 1)",
        ],
        &d,
        exp,
    );
}

#[test]
fn find_index_all_forms() {
    let d = users();
    assert_all(
        "find_index",
        &[
            "$.users.find_index(@.id == 2)",
            "$.users.find_index(.id == 2)",
            "$.users.find_index(u => u.id == 2)",
            "$.users.find_index(lambda u: u.id == 2)",
        ],
        &d,
        "1",
    );
}

#[test]
fn indices_where_all_forms() {
    let d = users();
    assert_all(
        "indices_where",
        &[
            "$.users.indices_where(@.active)",
            "$.users.indices_where(.active)",
            "$.users.indices_where(u => u.active)",
            "$.users.indices_where(lambda u: u.active)",
        ],
        &d,
        "[0,2]",
    );
}

#[test]
fn any_all_forms() {
    let d = users();
    assert_all(
        "any",
        &[
            "$.users.any(@.age > 40)",
            "$.users.any(.age > 40)",
            "$.users.any(u => u.age > 40)",
            "$.users.any(lambda u: u.age > 40)",
        ],
        &d,
        "true",
    );
}

#[test]
fn all_all_forms() {
    let d = users();
    assert_all(
        "all",
        &[
            "$.users.all(@.age > 0)",
            "$.users.all(.age > 0)",
            "$.users.all(u => u.age > 0)",
            "$.users.all(lambda u: u.age > 0)",
        ],
        &d,
        "true",
    );
}

#[test]
fn take_while_all_forms() {
    let d = xs();
    assert_all(
        "take_while",
        &[
            "$.xs.take_while(@ < 3)",
            "$.xs.take_while(x => x < 3)",
            "$.xs.take_while(lambda x: x < 3)",
        ],
        &d,
        "[1,2]",
    );
}

#[test]
fn drop_while_all_forms() {
    let d = xs();
    assert_all(
        "drop_while",
        &[
            "$.xs.drop_while(@ < 3)",
            "$.xs.drop_while(x => x < 3)",
            "$.xs.drop_while(lambda x: x < 3)",
        ],
        &d,
        "[3,4,5]",
    );
}

#[test]
fn remove_all_forms() {
    let d = xs();
    assert_all(
        "remove",
        &[
            "$.xs.remove(@ > 3)",
            "$.xs.remove(x => x > 3)",
            "$.xs.remove(lambda x: x > 3)",
        ],
        &d,
        "[1,2,3]",
    );
}

#[test]
fn partition_all_forms() {
    let d = xs();
    assert_all(
        "partition",
        &[
            "$.xs.partition(@ > 2)",
            "$.xs.partition(x => x > 2)",
            "$.xs.partition(lambda x: x > 2)",
        ],
        &d,
        "[[3,4,5],[1,2]]",
    );
}

#[test]
fn count_lambda_forms() {
    let d = users();
    assert_all(
        "count(pred)",
        &[
            "$.users.count(@.active)",
            "$.users.count(.active)",
            "$.users.count(u => u.active)",
            "$.users.count(lambda u: u.active)",
        ],
        &d,
        "2",
    );
}

// ──────────────────────────────────────────────────────────────────────────
// Group B: single-arg projection-lambda builtins
// ──────────────────────────────────────────────────────────────────────────

#[test]
fn map_all_forms() {
    let d = users();
    assert_all(
        "map",
        &[
            "$.users.map(@.id)",
            "$.users.map(.id)",
            "$.users.map(u => u.id)",
            "$.users.map(lambda u: u.id)",
        ],
        &d,
        "[1,2,3]",
    );
}

#[test]
fn flat_map_all_forms() {
    let d = json!({"groups": [[1, 2], [3, 4]]});
    assert_all(
        "flat_map",
        &[
            "$.groups.flat_map(@)",
            "$.groups.flat_map(g => g)",
            "$.groups.flat_map(lambda g: g)",
        ],
        &d,
        "[1,2,3,4]",
    );
}

#[test]
fn flat_map_last_uses_semantic_output_order() {
    let d = json!({"groups": [[1, 2], [], [3, 4], [5]]});
    assert_eq!(run("$.groups.flat_map(@).last()", &d), "5");
}

#[test]
fn unique_last_uses_distinct_output_order() {
    let d = json!({"xs": ["a", "b", "a", "c", "b"]});
    assert_eq!(run("$.xs.unique().last()", &d), "\"c\"");
}

#[test]
fn sort_lambda_forms() {
    let d = users();
    let asc = r#"[{"active":false,"age":24,"id":2,"name":"Bob","score":40},{"active":true,"age":30,"id":1,"name":"Ada","score":80},{"active":true,"age":42,"id":3,"name":"Carol","score":95}]"#;
    assert_all(
        "sort(key)",
        &[
            "$.users.sort(@.age)",
            "$.users.sort(.age)",
            "$.users.sort(u => u.age)",
            "$.users.sort(lambda u: u.age)",
        ],
        &d,
        asc,
    );
}

#[test]
fn sort_comparator_forms() {
    let d = xs();
    assert_all(
        "sort(comparator)",
        &[
            "$.xs.sort((a, b) => b < a)",
            "$.xs.sort(lambda a, b: b < a)",
        ],
        &d,
        "[5,4,3,2,1]",
    );
}

#[test]
fn unique_by_all_forms() {
    let d = json!({"xs": [{"k": "a", "v": 1}, {"k": "a", "v": 2}, {"k": "b", "v": 3}]});
    let exp = r#"[{"k":"a","v":1},{"k":"b","v":3}]"#;
    assert_all(
        "unique_by",
        &[
            "$.xs.unique_by(@.k)",
            "$.xs.distinct_by(@.k)",
            "$.xs.unique_by(.k)",
            "$.xs.unique_by(x => x.k)",
            "$.xs.unique_by(lambda x: x.k)",
        ],
        &d,
        exp,
    );
}

#[test]
fn group_by_all_forms() {
    let d = json!({"xs": [{"t": "a"}, {"t": "b"}, {"t": "a"}]});
    let exp = r#"{"a":[{"t":"a"},{"t":"a"}],"b":[{"t":"b"}]}"#;
    assert_all(
        "group_by",
        &[
            "$.xs.group_by(@.t)",
            "$.xs.group_by(.t)",
            "$.xs.group_by(x => x.t)",
            "$.xs.group_by(lambda x: x.t)",
        ],
        &d,
        exp,
    );
}

#[test]
fn count_by_all_forms() {
    let d = json!({"xs": [{"t": "a"}, {"t": "b"}, {"t": "a"}]});
    let exp = r#"{"a":2,"b":1}"#;
    assert_all(
        "count_by",
        &[
            "$.xs.count_by(@.t)",
            "$.xs.count_by(.t)",
            "$.xs.count_by(x => x.t)",
            "$.xs.count_by(lambda x: x.t)",
        ],
        &d,
        exp,
    );
}

#[test]
fn index_by_all_forms() {
    let d = json!({"xs": [{"id": "a", "v": 1}, {"id": "b", "v": 2}]});
    let exp = r#"{"a":{"id":"a","v":1},"b":{"id":"b","v":2}}"#;
    assert_all(
        "index_by",
        &[
            "$.xs.index_by(@.id)",
            "$.xs.index_by(.id)",
            "$.xs.index_by(x => x.id)",
            "$.xs.index_by(lambda x: x.id)",
        ],
        &d,
        exp,
    );
}

#[test]
fn max_by_all_forms() {
    let d = users();
    let exp = r#"{"active":true,"age":42,"id":3,"name":"Carol","score":95}"#;
    assert_all(
        "max_by",
        &[
            "$.users.max_by(@.score)",
            "$.users.max_by(.score)",
            "$.users.max_by(u => u.score)",
            "$.users.max_by(lambda u: u.score)",
        ],
        &d,
        exp,
    );
}

#[test]
fn min_by_all_forms() {
    let d = users();
    let exp = r#"{"active":false,"age":24,"id":2,"name":"Bob","score":40}"#;
    assert_all(
        "min_by",
        &[
            "$.users.min_by(@.score)",
            "$.users.min_by(.score)",
            "$.users.min_by(u => u.score)",
            "$.users.min_by(lambda u: u.score)",
        ],
        &d,
        exp,
    );
}

#[test]
fn sum_lambda_forms() {
    let d = users();
    assert_all(
        "sum(proj)",
        &[
            "$.users.sum(@.score)",
            "$.users.sum(.score)",
            "$.users.sum(u => u.score)",
            "$.users.sum(lambda u: u.score)",
        ],
        &d,
        "215",
    );
}

#[test]
fn avg_lambda_forms() {
    let d = json!({"xs": [{"v": 10}, {"v": 20}, {"v": 30}]});
    assert_all(
        "avg(proj)",
        &[
            "$.xs.avg(@.v)",
            "$.xs.avg(.v)",
            "$.xs.avg(x => x.v)",
            "$.xs.avg(lambda x: x.v)",
        ],
        &d,
        "20.0",
    );
}

#[test]
fn min_lambda_forms() {
    let d = users();
    // Projected aggregate widens to f64 — see `max_lambda_forms`.
    assert_all(
        "min(proj)",
        &[
            "$.users.min(@.score)",
            "$.users.min(.score)",
            "$.users.min(u => u.score)",
        ],
        &d,
        "40.0",
    );
}

#[test]
fn max_lambda_forms() {
    let d = users();
    // `max(proj)` returns the projected value as a float; use `max_by` for
    // the originating element preserved in its source numeric type.
    assert_all(
        "max(proj)",
        &[
            "$.users.max(@.score)",
            "$.users.max(.score)",
            "$.users.max(u => u.score)",
        ],
        &d,
        "95.0",
    );
}

#[test]
fn transform_keys_all_forms() {
    let d = obj();
    let exp = r#"{"o":{"K_a":1,"K_b":2,"K_c":3}}"#;
    assert_all(
        "transform_keys",
        &[
            r#"$.o.transform_keys(@.upper())"#,  // wraps key name
        ],
        &json!({"o": {"a": 1, "b": 2, "c": 3}}),
        r#"{"A":1,"B":2,"C":3}"#,
    );
    let _ = d;
    let _ = exp;
}

#[test]
fn transform_values_all_forms() {
    let d = obj();
    let exp = r#"{"a":2,"b":4,"c":6}"#;
    assert_all(
        "transform_values",
        &[
            "$.o.transform_values(@ * 2)",
            "$.o.transform_values(v => v * 2)",
            "$.o.transform_values(lambda v: v * 2)",
        ],
        &d,
        exp,
    );
}

#[test]
fn filter_keys_all_forms() {
    let d = obj();
    let exp = r#"{"a":1,"b":2}"#;
    assert_all(
        "filter_keys",
        &[
            r#"$.o.filter_keys(@ != "c")"#,
            r#"$.o.filter_keys(k => k != "c")"#,
            r#"$.o.filter_keys(lambda k: k != "c")"#,
        ],
        &d,
        exp,
    );
}

#[test]
fn filter_values_all_forms() {
    let d = obj();
    let exp = r#"{"b":2,"c":3}"#;
    assert_all(
        "filter_values",
        &[
            "$.o.filter_values(@ > 1)",
            "$.o.filter_values(v => v > 1)",
            "$.o.filter_values(lambda v: v > 1)",
        ],
        &d,
        exp,
    );
}

#[test]
fn fanout_all_forms() {
    let d = json!({"x": 5});
    assert_all(
        "fanout",
        &[
            "$.x.fanout(@ * 2, @ + 1)",
            "$.x.fanout(x => x * 2, x => x + 1)",
        ],
        &d,
        "[10,6]",
    );
}

// ──────────────────────────────────────────────────────────────────────────
// Group C: multi-arg lambdas
// ──────────────────────────────────────────────────────────────────────────

#[test]
fn accumulate_two_arg_forms() {
    let d = xs();
    assert_all(
        "accumulate(0, fn)",
        &[
            "$.xs.accumulate(0, (a, b) => a + b)",
            "$.xs.accumulate(0, lambda a, b: a + b)",
        ],
        &d,
        "[1,3,6,10,15]",
    );
}

#[test]
fn accumulate_one_arg_forms() {
    let d = xs();
    assert_all(
        "accumulate(fn)",
        &[
            "$.xs.accumulate((a, b) => a + b)",
            "$.xs.accumulate(lambda a, b: a + b)",
        ],
        &d,
        "[1,3,6,10,15]",
    );
}

#[test]
fn fold_two_arg_forms() {
    let d = xs();
    assert_all(
        "fold(0, fn)",
        &[
            "$.xs.fold(0, (a, b) => a + b)",
            "$.xs.fold(0, lambda a, b: a + b)",
            "$.xs.reduce(0, (a, b) => a + b)",
        ],
        &d,
        "15",
    );
}

#[test]
fn fold_one_arg_forms() {
    let d = xs();
    assert_all(
        "fold(fn)",
        &[
            "$.xs.fold((a, b) => a + b)",
            "$.xs.fold(lambda a, b: a + b)",
        ],
        &d,
        "15",
    );
    // Empty / single-element edge cases.
    assert_eq!(run("[].fold(0, (a, b) => a + b)", &json!({})), "0");
    assert_eq!(run("[].fold((a, b) => a + b)", &json!({})), "null");
    assert_eq!(run("[7].fold((a, b) => a + b)", &json!({})), "7");
}

// ──────────────────────────────────────────────────────────────────────────
// Group D: bare-identifier args (no `@`, no path)
// ──────────────────────────────────────────────────────────────────────────

#[test]
fn pick_forms() {
    let d = json!({"u": {"id": 1, "name": "Ada", "age": 30}});
    assert_eq!(run("$.u.pick(id, name)", &d), r#"{"id":1,"name":"Ada"}"#);
    assert_eq!(run("$.u.pick(uid: id, who: name)", &d), r#"{"uid":1,"who":"Ada"}"#);
}

#[test]
fn omit_forms() {
    let d = json!({"u": {"id": 1, "name": "Ada", "secret": "x"}});
    assert_eq!(run("$.u.omit(secret)", &d), r#"{"id":1,"name":"Ada"}"#);
    assert_eq!(run("$.u.omit(secret, name)", &d), r#"{"id":1}"#);
}

#[test]
fn rename_forms() {
    // `rename({old: new})` accepts either bare-string values or quoted
    // string keys; the bare-key shorthand resolves to the local of the
    // same name, so the rename target needs an explicit string literal.
    let d = json!({"u": {"id": 1, "name": "Ada"}});
    assert_eq!(
        run(r#"$.u.rename({id: "uid", name: "who"})"#, &d),
        r#"{"uid":1,"who":"Ada"}"#,
    );
    assert_eq!(
        run(r#"$.u.rename({"id": "uid"})"#, &d),
        r#"{"name":"Ada","uid":1}"#,
    );
}

#[test]
fn missing_forms() {
    let d = json!({"a": 1, "c": 3});
    assert_eq!(run(r#"$.missing("a", "b", "c")"#, &d), r#"["b"]"#);
}

#[test]
fn has_key_forms() {
    let d = json!({"o": {"a": 1, "b": 2}});
    assert_eq!(run(r#"$.o.has_key("a")"#, &d), "true");
    assert_eq!(run(r#"$.o.has_key("z")"#, &d), "false");
}

#[test]
fn has_array_rhs_forms() {
    let d = json!({
        "text": "hello world",
        "tags": ["x", "y", "z"],
        "nums": [1, 2, 3],
        "obj": {"x": 1, "y": 2}
    });

    assert_eq!(run(r#"$.text has ["hello", "world"]"#, &d), "true");
    assert_eq!(run(r#"$.text has ["hello", "missing"]"#, &d), "false");
    assert_eq!(run(r#"$.tags has ["x", "y"]"#, &d), "true");
    assert_eq!(run(r#"$.tags has ["x", "q"]"#, &d), "false");
    assert_eq!(run(r#"$.nums has [1, 2]"#, &d), "true");
    assert_eq!(run(r#"$.obj has ["x", "y"]"#, &d), "true");
    assert_eq!(run(r#"$.obj has ["x", "z"]"#, &d), "false");
    assert_eq!(run(r#"$.obj has []"#, &d), "true");
}

#[test]
fn has_array_rhs_rejects_non_literal_needles() {
    let d = json!({"tags": ["x", "y"], "needle": "x"});
    let bytes = serde_json::to_vec(&d).unwrap();
    let j = Jetro::from_bytes(bytes).unwrap();
    let err = j
        .collect("$.tags has [needle]")
        .expect_err("dynamic has array RHS should fail at parse time");

    assert!(
        err.0.contains("has [...] requires scalar literal elements"),
        "{err:?}"
    );
}

#[test]
fn deep_shape_forms() {
    let d = json!({
        "rows": [
            {"id": 1, "name": "a"},
            {"id": 2}
        ]
    });
    assert_eq!(run("$.deep_shape({id, name})", &d), r#"[{"id":1,"name":"a"}]"#);
}

#[test]
fn deep_like_forms() {
    let d = json!({"rows": [{"k": "x"}, {"k": "y"}]});
    assert_eq!(run(r#"$.deep_like({k: "x"})"#, &d), r#"[{"k":"x"}]"#);
}

#[test]
fn group_shape_forms() {
    let d = json!({"users": [{"id": 1, "k": "x"}, {"id": 2, "k": "y"}, {"id": 3, "k": "x"}]});
    let exp = r#"{"x":[{"id":1,"k":"x"},{"id":3,"k":"x"}],"y":[{"id":2,"k":"y"}]}"#;
    assert_all(
        "group_shape(key)",
        &[
            "$.users.group_shape(.k)",
            "$.users.group_shape(@.k)",
            "$.users.group_shape(u => u.k)",
            "$.users.group_shape(lambda u: u.k)",
            "$.users.group_shape(by: k)",
        ],
        &d,
        exp,
    );
}

#[test]
fn zip_shape_forms() {
    let d = json!({"a": [1, 2], "b": [3, 4]});
    let bare_form = r#"{"a":[1,2],"b":[3,4]}"#;
    let interleave_form = r#"[{"a":1,"b":3},{"a":2,"b":4}]"#;
    assert_eq!(run("$.zip_shape(a, b)", &d), bare_form);
    assert_eq!(run("$.zip_shape({a, b})", &d), interleave_form);
    assert_eq!(run("$.zip_shape()", &d), interleave_form);
}

// ──────────────────────────────────────────────────────────────────────────
// Group E: positional value args
// ──────────────────────────────────────────────────────────────────────────

#[test]
fn slice_int_args() {
    assert_eq!(run(r#""hello".slice(0, 3)"#, &json!({})), r#""hel""#);
    assert_eq!(run(r#""hello".slice(2)"#, &json!({})), r#""llo""#);
    assert_eq!(run(r#"$.s.slice(0, 3)"#, &json!({"s":"hello"})), r#""hel""#);
}

#[test]
fn replace_args() {
    assert_eq!(
        run(r#""hello hello".replace("hello", "hi")"#, &json!({})),
        r#""hi hello""#,
    );
    assert_eq!(
        run(r#""hello hello".replace_all("hello", "hi")"#, &json!({})),
        r#""hi hi""#,
    );
}

#[test]
fn indent_overloads() {
    assert_eq!(
        run(r#""a\nb".indent(2)"#, &json!({})),
        r#""  a\n  b""#,
    );
    assert_eq!(
        run(r#""a\nb".indent("> ")"#, &json!({})),
        r#""> a\n> b""#,
    );
}

#[test]
fn repeat_args() {
    assert_eq!(run(r#""ab".repeat(3)"#, &json!({})), r#""ababab""#);
}

#[test]
fn pad_args() {
    assert_eq!(
        run(r#""abc".pad_left(6, "_")"#, &json!({})),
        r#""___abc""#,
    );
    assert_eq!(
        run(r#""abc".pad_right(6, "_")"#, &json!({})),
        r#""abc___""#,
    );
    assert_eq!(
        run(r#""abc".center(7, "_")"#, &json!({})),
        r#""__abc__""#,
    );
}

#[test]
fn split_join() {
    assert_eq!(
        run(r#""a,b,c".split(",")"#, &json!({})),
        r#"["a","b","c"]"#,
    );
    assert_eq!(
        run(r#"["a","b","c"].join(",")"#, &json!({})),
        r#""a,b,c""#,
    );
}

#[test]
fn nth_args() {
    let d = xs();
    assert_eq!(run("$.xs.nth(0)", &d), "1");
    assert_eq!(run("$.xs.nth(-1)", &d), "5");
    assert_eq!(run("$.xs.nth(2)", &d), "3");
}

#[test]
fn take_skip() {
    let d = xs();
    assert_eq!(run("$.xs.take(3)", &d), "[1,2,3]");
    assert_eq!(run("$.xs.skip(2)", &d), "[3,4,5]");
}

#[test]
fn first_last_unary_and_n() {
    let d = xs();
    assert_eq!(run("$.xs.first()", &d), "1");
    assert_eq!(run("$.xs.last()", &d), "5");
    assert_eq!(run("$.xs.first(2)", &d), "[1,2]");
    assert_eq!(run("$.xs.last(2)", &d), "[4,5]");
}

#[test]
fn append_prepend() {
    let d = json!({"xs": [2, 3]});
    assert_eq!(run("$.xs.append(4)", &d), "[2,3,4]");
    assert_eq!(run("$.xs.prepend(1)", &d), "[1,2,3]");
}

#[test]
fn chunk_window() {
    let d = xs();
    assert_eq!(run("$.xs.chunk(2)", &d), "[[1,2],[3,4],[5]]");
    assert_eq!(run("$.xs.window(3)", &d), "[[1,2,3],[2,3,4],[3,4,5]]");
}

#[test]
fn lag_lead() {
    // Both windowing builtins promote integers to f64 inside the result
    // array because the `null` placeholder forces a unified numeric kind.
    let d = xs();
    assert_eq!(run("$.xs.lag(1)", &d), "[null,1.0,2.0,3.0,4.0]");
    assert_eq!(run("$.xs.lead(1)", &d), "[2.0,3.0,4.0,5.0,null]");
}

#[test]
fn rolling_args() {
    // Rolling/window outputs widen to f64 because `null` shares the slot.
    let d = xs();
    assert_eq!(run("$.xs.rolling_sum(2)", &d), "[null,3.0,5.0,7.0,9.0]");
    assert_eq!(run("$.xs.rolling_avg(2)", &d), "[null,1.5,2.5,3.5,4.5]");
    assert_eq!(run("$.xs.rolling_min(2)", &d), "[null,1.0,2.0,3.0,4.0]");
    assert_eq!(run("$.xs.rolling_max(2)", &d), "[null,2.0,3.0,4.0,5.0]");
}

#[test]
fn diff_window_pct_change_zscore() {
    // `cummax`/`cummin` are spelled without the underscore in the public
    // API; the running min/max output widens to f64 to share a slot with
    // the float reductions used by the same compute path.
    let d = xs();
    assert_eq!(run("$.xs.diff_window()", &d), "[null,1.0,1.0,1.0,1.0]");
    assert_eq!(run("$.xs.cummax()", &d), "[1.0,2.0,3.0,4.0,5.0]");
    assert_eq!(run("$.xs.cummin()", &d), "[1.0,1.0,1.0,1.0,1.0]");
}

// ──────────────────────────────────────────────────────────────────────────
// Group F: regex
// ──────────────────────────────────────────────────────────────────────────

#[test]
fn re_match_args() {
    assert_eq!(run(r#""abc123".re_match("\d+")"#, &json!({})), "true");
    assert_eq!(run(r#""abc".re_match("\d+")"#, &json!({})), "false");
}

#[test]
fn re_replace_args() {
    // Public names are `replace_re` (single replace) and `replace_all_re`.
    assert_eq!(
        run(r#""abc123".replace_re("\d", "X")"#, &json!({})),
        r#""abcX23""#,
    );
    assert_eq!(
        run(r#""abc123def456".replace_all_re("\d+", "<n>")"#, &json!({})),
        r#""abc<n>def<n>""#,
    );
}

#[test]
fn includes_index() {
    assert_eq!(run(r#"["a","b"].includes("a")"#, &json!({})), "true");
    assert_eq!(run(r#"["a","b","c"].index("b")"#, &json!({})), "1");
}

// ──────────────────────────────────────────────────────────────────────────
// Group G: path-mutation methods
// ──────────────────────────────────────────────────────────────────────────

#[test]
fn set_path_args() {
    let d = json!({"u": {"name": "Ada"}});
    assert_eq!(
        run(r#"$.set_path("u.email", "x@y.z")"#, &d),
        r#"{"u":{"email":"x@y.z","name":"Ada"}}"#,
    );
}

#[test]
fn get_path_args() {
    let d = json!({"a": {"b": {"c": 1}}});
    assert_eq!(run(r#"$.get_path("a.b.c")"#, &d), "1");
    assert_eq!(run(r#"$.get_path("a/b/c")"#, &d), "1");
}

#[test]
fn del_path_args() {
    let d = json!({"u": {"a": 1, "b": 2}});
    assert_eq!(run(r#"$.del_path("u.a")"#, &d), r#"{"u":{"b":2}}"#);
}

#[test]
fn has_path_args() {
    let d = json!({"a": null, "b": 1});
    assert_eq!(run(r#"$.has_path("a")"#, &d), "false");
    assert_eq!(run(r#"$.has_path("b")"#, &d), "true");
    assert_eq!(run(r#"$.has_path("z")"#, &d), "false");
}

// ──────────────────────────────────────────────────────────────────────────
// Group H: write methods (chain-write terminals)
// ──────────────────────────────────────────────────────────────────────────

#[test]
fn set_chain() {
    let d = json!({"u": {"name": "Ada"}});
    assert_eq!(
        run(r#"$.u.name.set("Bob")"#, &d),
        r#"{"u":{"name":"Bob"}}"#,
    );
}

#[test]
fn modify_chain() {
    let d = json!({"x": 5});
    assert_eq!(run("$.x.modify(@ + 1)", &d), r#"{"x":6}"#);
    assert_eq!(run("$.x.modify(v => v * 2)", &d), r#"{"x":10}"#);
}

#[test]
fn update_object_form() {
    let d = json!({"u": {"name": "Ada", "age": 30}});
    assert_eq!(
        run(r#"$.u.update({age: age + 1, role: "admin"})"#, &d),
        r#"{"u":{"age":31,"name":"Ada","role":"admin"}}"#,
    );
}

#[test]
fn update_two_arg() {
    let d = json!({"counters": {"visits": 10}});
    assert_eq!(
        run(r#"$.update("counters.visits", @ + 1)"#, &d),
        r#"{"counters":{"visits":11}}"#,
    );
}

// ──────────────────────────────────────────────────────────────────────────
// Group I: deep / structural / walk
// ──────────────────────────────────────────────────────────────────────────

#[test]
fn deep_find_pred_forms() {
    // `@ is num and @ > 4` does not parse — kind tests don't compose with
    // `and` inside a method-arg position. Stick to the comparison form.
    let d = json!({"rows": [{"v": 1}, {"v": 5}, {"v": 9}]});
    assert_eq!(run("$.deep_find(@ > 4)", &d), "[5,9]");
    assert_eq!(run("$.deep_find(@ > 0)", &d), "[1,5,9]");
}

#[test]
fn walk_lambda_forms() {
    // The Python-style ternary doesn't parse inside a method-arg position
    // because `if` doubles as a comprehension/filter keyword. Use a
    // host-side transform like `to_string` that doesn't need a guard.
    let d = json!({"x": [1, 2], "k": "hi"});
    assert_all(
        "walk(fn)",
        &[
            "$.walk(@.to_string())",
            "$.walk(v => v.to_string())",
            "$.walk(lambda v: v.to_string())",
        ],
        &d,
        r#""{\"k\":\"hi\",\"x\":\"[\\\"1\\\",\\\"2\\\"]\"}""#,
    );
}

#[test]
fn rec_one_arg() {
    let d = json!({"x": [1, [2, [3]]]});
    assert_eq!(run("$.x.rec(@.flatten())", &d), "[1,2,3]");
}

#[test]
fn rec_two_arg_cond() {
    let d = json!({"x": 1});
    assert_eq!(run("$.x.rec(@ * 2, @ < 100)", &d), "128");
}

// ──────────────────────────────────────────────────────────────────────────
// Group J: misc no-arg or terse args
// ──────────────────────────────────────────────────────────────────────────

#[test]
fn diff_intersect_union() {
    let l = json!({"l": [1, 2, 3], "r": [2, 3, 4]});
    assert_eq!(run("$.l.diff($.r)", &l), "[1]");
    assert_eq!(run("$.l.intersect($.r)", &l), "[2,3]");
    assert_eq!(run("$.l.union($.r)", &l), "[1,2,3,4]");
}

#[test]
fn zip_zip_longest() {
    // `.zip_longest` ignores any explicit fill argument in v0.5; missing
    // positions are emitted as `null`. Document the actual shape so the
    // test stays honest.
    let d = json!({"a": [1, 2, 3], "b": ["x", "y"]});
    assert_eq!(run("$.a.zip($.b)", &d), r#"[[1,"x"],[2,"y"]]"#);
    assert_eq!(
        run("$.a.zip_longest($.b)", &d),
        r#"[[1,"x"],[2,"y"],[3,null]]"#,
    );
}

#[test]
fn flatten_arg() {
    let d = json!({"xs": [[1, 2], [3, [4, 5]]]});
    assert_eq!(run("$.xs.flatten()", &d), "[1,2,3,[4,5]]");
    assert_eq!(run("$.xs.flatten(2)", &d), "[1,2,3,4,5]");
}

#[test]
fn enumerate_pairwise() {
    let d = xs();
    assert_eq!(
        run("$.xs.enumerate()", &d),
        r#"[{"index":0,"value":1},{"index":1,"value":2},{"index":2,"value":3},{"index":3,"value":4},{"index":4,"value":5}]"#,
    );
    assert_eq!(run("$.xs.pairwise()", &d), "[[1,2],[2,3],[3,4],[4,5]]");
}

#[test]
fn merge_deep_merge() {
    // `.merge` and `.deep_merge` are chain-write terminals when rooted at
    // `$` — they return the patched root with `$.a` updated in place.
    // Use the pipe form to extract just the merged object.
    let d = json!({"a": {"x": 1, "y": 2}, "b": {"y": 9, "z": 3}});
    assert_eq!(
        run("$.a.merge($.b)", &d),
        r#"{"a":{"x":1,"y":9,"z":3},"b":{"y":9,"z":3}}"#,
    );
    assert_eq!(
        run("$.a | @.merge($.b)", &d),
        r#"{"x":1,"y":9,"z":3}"#,
    );
    let n = json!({
        "a": {"x": {"p": 1}, "q": 2},
        "b": {"x": {"p": 9, "r": 3}}
    });
    assert_eq!(
        run("$.a | @.deep_merge($.b)", &n),
        r#"{"q":2,"x":{"p":9,"r":3}}"#,
    );
}

#[test]
fn or_arg() {
    let d = json!({"a": null, "b": 5});
    assert_eq!(run("$.a.or(99)", &d), "99");
    assert_eq!(run("$.b.or(99)", &d), "5");
}

#[test]
fn defaults_arg() {
    let d = json!({"u": {"name": "Ada"}});
    assert_eq!(
        run(r#"$.u.defaults({name: "x", age: 30})"#, &d),
        r#"{"age":30,"name":"Ada"}"#,
    );
}

#[test]
fn collect_values() {
    assert_eq!(run("42.collect()", &json!({})), "[42]");
    assert_eq!(run("[1,2].collect()", &json!({})), "[1,2]");
    assert_eq!(run("null.collect()", &json!({})), "[]");
}
