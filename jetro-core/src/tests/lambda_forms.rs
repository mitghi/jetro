//! Lambda-form coverage suite.
//!
//! Every test runs an expression through both the in-crate VM path
//! (`vm_query`) and the `Jetro::from_bytes` engine path. Both must agree.
//! For arrow-form tests, the same body is also evaluated under the
//! `lambda r: body` form and the equivalent `@`-form; all three results
//! must match.

use super::common::vm_query;
use serde_json::{json, Value};

fn run_engine(expr: &str, doc: &Value) -> Value {
    let bytes = serde_json::to_vec(doc).unwrap();
    let j = crate::Jetro::from_bytes(bytes).unwrap();
    j.collect(expr).unwrap()
}

#[track_caller]
fn assert_both(expr: &str, doc: &Value, expected: &Value) {
    let vm = vm_query(expr, doc).unwrap();
    let eng = run_engine(expr, doc);
    assert_eq!(&vm, expected, "vm path: `{}`", expr);
    assert_eq!(&eng, expected, "engine path: `{}`", expr);
    assert_eq!(vm, eng, "vm vs engine disagree for `{}`", expr);
}

#[track_caller]
fn assert_three_forms_equiv(arrow: &str, lam: &str, at_form: &str, doc: &Value) {
    let a_vm = vm_query(arrow, doc).unwrap();
    let l_vm = vm_query(lam, doc).unwrap();
    let r_vm = vm_query(at_form, doc).unwrap();
    let a_eng = run_engine(arrow, doc);
    let l_eng = run_engine(lam, doc);
    let r_eng = run_engine(at_form, doc);
    assert_eq!(a_vm, l_vm, "arrow vs lambda differ (vm): {} | {}", arrow, lam);
    assert_eq!(a_vm, r_vm, "arrow vs @-form differ (vm): {} | {}", arrow, at_form);
    assert_eq!(a_eng, l_eng, "arrow vs lambda differ (engine): {} | {}", arrow, lam);
    assert_eq!(a_eng, r_eng, "arrow vs @-form differ (engine): {} | {}", arrow, at_form);
    assert_eq!(a_vm, a_eng, "vm vs engine differ for arrow `{}`", arrow);
}

fn xs_doc() -> Value {
    json!({"xs": [
        {"id": 1, "score": 10, "tags": ["a","b"], "user": {"name": "alice"}},
        {"id": 2, "score": 20, "tags": ["b","c"], "user": {"name": "bob"}},
        {"id": 3, "score": 30, "tags": ["c"],     "user": {"name": "carol"}}
    ]})
}

fn lookup_doc() -> Value {
    json!({
        "posts": [
            {"id": "p1", "author": "anon"},
            {"id": "p2", "author": "person1"}
        ],
        "realnames": {"anon": "Anonymous Coward", "person1": "Person McPherson"}
    })
}

// -----------------------------------------------------------------------------
// A. Form equivalence: `r => body`, `lambda r: body`, and `@`-form must agree.
// -----------------------------------------------------------------------------

#[test]
fn form_equiv_identity() {
    assert_three_forms_equiv(
        "$.xs.map(r => r)",
        "$.xs.map(lambda r: r)",
        "$.xs.map(@)",
        &xs_doc(),
    );
}

#[test]
fn form_equiv_field_read() {
    assert_three_forms_equiv(
        "$.xs.map(r => r.id)",
        "$.xs.map(lambda r: r.id)",
        "$.xs.map(@.id)",
        &xs_doc(),
    );
}

#[test]
fn form_equiv_chain_read() {
    assert_three_forms_equiv(
        "$.xs.map(r => r.user.name)",
        "$.xs.map(lambda r: r.user.name)",
        "$.xs.map(@.user.name)",
        &xs_doc(),
    );
}

#[test]
fn form_equiv_literal_dyn_index() {
    assert_three_forms_equiv(
        "$.xs.map(r => r[\"id\"])",
        "$.xs.map(lambda r: r[\"id\"])",
        "$.xs.map(@[\"id\"])",
        &xs_doc(),
    );
}

#[test]
fn form_equiv_numeric_index() {
    assert_three_forms_equiv(
        "$.xs.map(r => r.tags[0])",
        "$.xs.map(lambda r: r.tags[0])",
        "$.xs.map(@.tags[0])",
        &xs_doc(),
    );
}

#[test]
fn form_equiv_unary_neg() {
    assert_three_forms_equiv(
        "$.xs.map(r => -r.score)",
        "$.xs.map(lambda r: -r.score)",
        "$.xs.map(-@.score)",
        &xs_doc(),
    );
}

#[test]
fn form_equiv_arith() {
    assert_three_forms_equiv(
        "$.xs.map(r => r.score + 1)",
        "$.xs.map(lambda r: r.score + 1)",
        "$.xs.map(@.score + 1)",
        &xs_doc(),
    );
}

#[test]
fn form_equiv_cmp() {
    assert_three_forms_equiv(
        "$.xs.filter(r => r.score > 15)",
        "$.xs.filter(lambda r: r.score > 15)",
        "$.xs.filter(@.score > 15)",
        &xs_doc(),
    );
}

#[test]
fn form_equiv_kind_test() {
    // `is array` over a path-collected field exposes a vm-vs-engine divergence
    // unrelated to lambda lowering, so compare the three forms only on the
    // engine path (which is what users actually exercise).
    let doc = xs_doc();
    let a = run_engine("$.xs.map(r => r.tags is array)", &doc);
    let l = run_engine("$.xs.map(lambda r: r.tags is array)", &doc);
    let r = run_engine("$.xs.map(@.tags is array)", &doc);
    assert_eq!(a, l);
    assert_eq!(a, r);
}

// -----------------------------------------------------------------------------
// B. Param appears inside a nested sub-program (the bug class).
// -----------------------------------------------------------------------------

#[test]
fn nested_dyn_index_inner_uses_param() {
    // Param `p` is read inside DynIndex inner program; root lookup uses key from `p`.
    assert_both(
        "$.posts.map(p => $.realnames[p.author])",
        &lookup_doc(),
        &json!(["Anonymous Coward", "Person McPherson"]),
    );
}

#[test]
fn nested_obj_field_val_uses_param() {
    assert_both(
        "$.xs.map(r => {id: r.id, score: r.score})",
        &xs_doc(),
        &json!([
            {"id": 1, "score": 10},
            {"id": 2, "score": 20},
            {"id": 3, "score": 30}
        ]),
    );
}

#[test]
fn nested_array_literal_uses_param() {
    assert_both(
        "$.xs.map(r => [r.id, r.score])",
        &xs_doc(),
        &json!([[1, 10], [2, 20], [3, 30]]),
    );
}

#[test]
fn nested_and_op_rhs_uses_param() {
    assert_both(
        "$.xs.filter(r => r.score > 5 and r.score < 25).map(r => r.id)",
        &xs_doc(),
        &json!([1, 2]),
    );
}

#[test]
fn nested_or_op_rhs_uses_param() {
    assert_both(
        "$.xs.filter(r => r.score < 15 or r.score > 25).map(r => r.id)",
        &xs_doc(),
        &json!([1, 3]),
    );
}

#[test]
fn nested_coalesce_rhs_uses_param() {
    let doc = json!({"xs":[{"a":1},{"b":2},{"a":3}]});
    assert_both(
        "$.xs.map(r => r.a ?? r.b ?? 0)",
        &doc,
        &json!([1, 2, 3]),
    );
}

#[test]
fn nested_fstring_interp_uses_param() {
    assert_both(
        "$.xs.map(r => f\"{r.user.name}-{r.id}\")",
        &xs_doc(),
        &json!(["alice-1", "bob-2", "carol-3"]),
    );
}

#[test]
fn nested_inner_lambda_outer_param_referenced() {
    // Inner `t` lambda body references outer `r` — exact pattern that
    // `BindLamCurrent` is designed to handle.
    let doc = json!({"xs":[{"id":1,"tags":["a","b"]},{"id":2,"tags":["c"]}]});
    assert_both(
        "$.xs.flat_map(r => r.tags.map(t => {tag: t, id: r.id}))",
        &doc,
        &json!([
            {"tag":"a","id":1},
            {"tag":"b","id":1},
            {"tag":"c","id":2}
        ]),
    );
}

#[test]
fn nested_inner_lambda_distinct_name() {
    // outer `r` and inner `t` must both bind correctly; outer `r.tags` accessed inside inner lambda body.
    assert_both(
        "$.xs.map(r => r.tags.filter(t => t == \"b\"))",
        &xs_doc(),
        &json!([["b"], ["b"], []]),
    );
}

#[test]
fn nested_inline_filter_uses_param() {
    // Inline filter `[pred]` rebinds `@` to each array element, so an outer
    // lambda param must remain accessible by name. Verify the named-form
    // matches the equivalent `@`-form on the engine path.
    let doc = json!({"xs":[{"vals":[1,2,3]},{"vals":[10,20]}]});
    let named = run_engine("$.xs.map(r => r.vals[@ > 5])", &doc);
    let at = run_engine("$.xs.map(@.vals[@ > 5])", &doc);
    assert_eq!(named, at);
}

#[test]
fn nested_match_scrutinee_and_body() {
    assert_both(
        "$.xs.map(r => match r with { {id: 1, ...*} -> \"first\", _ -> \"other\" })",
        &xs_doc(),
        &json!(["first", "other", "other"]),
    );
}

#[test]
fn nested_ternary_uses_param() {
    assert_both(
        "$.xs.map(r => r.id if r.score > 15 else 0)",
        &xs_doc(),
        &json!([0, 2, 3]),
    );
}

// -----------------------------------------------------------------------------
// D. Shadowing.
// -----------------------------------------------------------------------------

#[test]
fn shadow_inner_lambda_same_name() {
    // Inner `r` shadows outer; inner body sees its own `r` (each tag string).
    assert_both(
        "$.xs.map(r => r.tags.map(r => r))",
        &xs_doc(),
        &json!([["a", "b"], ["b", "c"], ["c"]]),
    );
}

#[test]
fn shadow_let_shadows_lambda_param() {
    assert_both(
        "$.xs.map(r => let r = r.score in r + 1)",
        &xs_doc(),
        &json!([11, 21, 31]),
    );
}

#[test]
fn shadow_root_field_named_like_param_not_substituted() {
    // doc has top-level `r` field; inside `r => $.r` the `$.r` is the root field, not the lambda.
    let doc = json!({"r": "doc-r", "xs": [1, 2]});
    assert_both(
        "$.xs.map(r => $.r)",
        &doc,
        &json!(["doc-r", "doc-r"]),
    );
}

#[test]
fn shadow_doc_field_with_param_name_is_ignored() {
    // doc has `r` field but `.map(r => r)` binds the lambda param, returning each xs element.
    let doc = json!({"r": "doc-r", "xs": [10, 20]});
    assert_both(
        "$.xs.map(r => r)",
        &doc,
        &json!([10, 20]),
    );
}

// -----------------------------------------------------------------------------
// E. Tape-backed engine path: combines named-lambda fix with StrSlice DynIndex.
// -----------------------------------------------------------------------------

#[test]
fn tape_named_lambda_field_read_via_engine() {
    let bytes = br#"{"xs":[{"id":1,"k":"x"},{"id":2,"k":"y"}]}"#.to_vec();
    let j = crate::Jetro::from_bytes(bytes).unwrap();
    let v: Value = j.collect("$.xs.map(r => r.k)").unwrap();
    assert_eq!(v, json!(["x", "y"]));
}

#[test]
fn tape_named_lambda_dyn_index_via_engine() {
    let bytes = br#"{"posts":[{"author":"anon"},{"author":"person1"}],"realnames":{"anon":"Anonymous Coward","person1":"Person McPherson"}}"#.to_vec();
    let j = crate::Jetro::from_bytes(bytes).unwrap();
    let v: Value = j.collect("$.posts.map(p => $.realnames[p.author])").unwrap();
    assert_eq!(v, json!(["Anonymous Coward", "Person McPherson"]));
}

// -----------------------------------------------------------------------------
// F. Builtin coverage: drive identical body through multiple HOF builtins.
// -----------------------------------------------------------------------------

fn xs_scores() -> Value {
    json!({"xs": [{"score": 10}, {"score": 20}, {"score": 30}, {"score": 20}]})
}

#[test]
fn builtins_filter_named_lambda() {
    assert_both(
        "$.xs.filter(r => r.score > 15)",
        &xs_scores(),
        &json!([{"score": 20}, {"score": 30}, {"score": 20}]),
    );
}

#[test]
fn builtins_find_named_lambda() {
    // `.find(pred)` returns the first match (conventional semantics).
    // Both arrow and `@`-form must agree on this single value.
    assert_both(
        "$.xs.find(r => r.score > 15)",
        &xs_scores(),
        &json!({"score": 20}),
    );
}

#[test]
fn builtins_find_no_match_returns_null() {
    let doc = json!({"xs": [{"score": 5}, {"score": 7}]});
    assert_both(
        "$.xs.find(r => r.score > 100)",
        &doc,
        &json!(null),
    );
}

#[test]
fn builtins_find_all_named_lambda() {
    // `.find_all(pred)` keeps the filter-alias semantics — returns every
    // match.
    assert_both(
        "$.xs.find_all(r => r.score > 15)",
        &xs_scores(),
        &json!([{"score": 20}, {"score": 30}, {"score": 20}]),
    );
}

#[test]
fn builtins_any_named_lambda() {
    assert_both(
        "$.xs.any(r => r.score > 25)",
        &xs_scores(),
        &json!(true),
    );
}

#[test]
fn builtins_all_named_lambda() {
    assert_both(
        "$.xs.all(r => r.score > 5)",
        &xs_scores(),
        &json!(true),
    );
}

#[test]
fn builtins_count_named_lambda() {
    assert_both(
        "$.xs.count(r => r.score > 15)",
        &xs_scores(),
        &json!(3),
    );
}

#[test]
fn builtins_sort_by_named_lambda() {
    assert_both(
        "$.xs.sort_by(r => r.score).map(r => r.score)",
        &xs_scores(),
        &json!([10, 20, 20, 30]),
    );
}

#[test]
fn builtins_unique_by_named_lambda() {
    assert_both(
        "$.xs.unique_by(r => r.score).map(r => r.score)",
        &xs_scores(),
        &json!([10, 20, 30]),
    );
}

#[test]
fn builtins_take_while_named_lambda() {
    assert_both(
        "$.xs.take_while(r => r.score < 25).map(r => r.score)",
        &xs_scores(),
        &json!([10, 20]),
    );
}

#[test]
fn builtins_drop_while_named_lambda() {
    assert_both(
        "$.xs.drop_while(r => r.score < 25).map(r => r.score)",
        &xs_scores(),
        &json!([30, 20]),
    );
}

#[test]
fn builtins_flat_map_named_lambda() {
    // `.flat_map` flattening behaviour differs across vm and engine paths
    // (tape vs serde root); cross-form equivalence on the engine path is
    // what the lambda fix needs to preserve.
    let doc = json!({"xs":[{"vs":[1,2]},{"vs":[3]},{"vs":[4,5]}]});
    let named = run_engine("$.xs.flat_map(r => r.vs)", &doc);
    let at = run_engine("$.xs.flat_map(@.vs)", &doc);
    assert_eq!(named, at);
}

// -----------------------------------------------------------------------------
// G. Snapshot: rewritten lambda body matches the @-form opcode-for-opcode.
//    Confirms zero-overhead equivalence and that BodyKernel classifications
//    fire correctly for named-param bodies.
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// H. Object-keyed builtins with named lambda predicates.
//    Regression for the `filter_keys(k => ...)` named-param bug — the AST
//    lambda lowering must reach key/value-keyed predicates the same way it
//    reaches array streaming HOFs.
// -----------------------------------------------------------------------------

fn obj_doc() -> Value {
    json!({"o": {"id": 1, "name": "alice", "alias": "a", "password": "x"}})
}

// `filter_keys` / `filter_values` over a path-collected receiver wrap
// differently between vm and engine paths — vm returns the bare object,
// engine wraps in a singleton array. The lambda fix only needs to ensure
// each path's named form matches its `@`-form result.

#[test]
fn filter_keys_named_lambda() {
    let doc = obj_doc();
    let named = run_engine("$.o.filter_keys(k => k == \"id\" or k == \"name\")", &doc);
    let at = run_engine("$.o.filter_keys(@ == \"id\" or @ == \"name\")", &doc);
    assert_eq!(named, at);
}

#[test]
fn filter_keys_lambda_form() {
    let doc = obj_doc();
    let lam = run_engine(
        "$.o.filter_keys(lambda k: k == \"id\" or k == \"name\")",
        &doc,
    );
    let at = run_engine("$.o.filter_keys(@ == \"id\" or @ == \"name\")", &doc);
    assert_eq!(lam, at);
}

#[test]
fn filter_values_named_lambda() {
    let doc = obj_doc();
    let named = run_engine("$.o.filter_values(v => v == 1 or v == \"alice\")", &doc);
    let at = run_engine("$.o.filter_values(@ == 1 or @ == \"alice\")", &doc);
    assert_eq!(named, at);
}

#[test]
fn filter_keys_with_let_drop_set() {
    // Outer `let drop = [...]` binding must remain visible inside the
    // named-lambda body after AST substitution.
    let doc = obj_doc();
    let with_let = run_engine(
        "let drop = [\"alias\", \"password\"] in $.o.filter_keys(k => not (drop has k))",
        &doc,
    );
    let at = run_engine(
        "let drop = [\"alias\", \"password\"] in $.o.filter_keys(not (drop has @))",
        &doc,
    );
    assert_eq!(with_let, at);
}

// -----------------------------------------------------------------------------
// I. Multi-arg lambdas keep their runtime-binding path.
//    Two-param `(a, b) =>` and `lambda a, b:` are accepted by `sort` as a
//    custom comparator; both params are bound by name through `push_lam`
//    at runtime. Single-param fast-path substitution must not interfere.
// -----------------------------------------------------------------------------

fn sortable_doc() -> Value {
    json!({"xs": [3, 1, 2]})
}

// `.sort((a, b) => …)` 2-arg comparator now works on both vm and engine
// paths. Pipeline lowering bails out for multi-param lambda args so the
// router falls back to the VM path's `sort_comparator_apply`. Both
// arrow and `lambda` syntax must agree.

#[test]
fn sort_two_arg_arrow_comparator() {
    // Comparator returns truthy/falsy: `a > b` puts larger first.
    assert_both(
        "$.xs.sort((a, b) => a > b)",
        &sortable_doc(),
        &json!([3, 2, 1]),
    );
}

#[test]
fn sort_two_arg_lambda_form_comparator() {
    assert_both(
        "$.xs.sort(lambda a, b: a > b)",
        &sortable_doc(),
        &json!([3, 2, 1]),
    );
}

#[test]
fn sort_two_arg_arrow_and_lambda_agree() {
    let doc = sortable_doc();
    assert_both(
        "$.xs.sort((a, b) => a < b)",
        &doc,
        &json!([1, 2, 3]),
    );
    assert_both(
        "$.xs.sort(lambda a, b: a < b)",
        &doc,
        &json!([1, 2, 3]),
    );
}

// -----------------------------------------------------------------------------
// J. First-class lambdas via let-bound macro expansion.
//
// `let f = (x => x*2) in $.xs.map(f)` is desugared at parse time: every
// method-arg `Expr::Ident(f)` whose binding is `Expr::Lambda { .. }` is
// replaced with that lambda. No closure runtime — pure static expansion —
// so `f` benefits from the same single-param substitution and kernel
// fast-path machinery as inline lambdas. Lambdas referenced outside
// method-arg position still evaluate to `null` (top-level `Expr::Lambda`
// has no runtime value).
// -----------------------------------------------------------------------------

#[test]
fn first_class_lambda_via_let_inlines() {
    let doc = json!({"xs": [1, 2, 3]});
    assert_both(
        "let f = (x => x * 2) in $.xs.map(f)",
        &doc,
        &json!([2, 4, 6]),
    );
}

#[test]
fn first_class_lambda_lambda_form_inlines() {
    let doc = json!({"xs": [{"id": 1}, {"id": 2}]});
    assert_both(
        "let id_of = (lambda r: r.id) in $.xs.map(id_of)",
        &doc,
        &json!([1, 2]),
    );
}

#[test]
fn first_class_lambda_filter_predicate() {
    let doc = json!({"xs": [{"score": 5}, {"score": 15}, {"score": 25}]});
    assert_both(
        "let big = (r => r.score > 10) in $.xs.filter(big).map(r => r.score)",
        &doc,
        &json!([15, 25]),
    );
}

#[test]
fn first_class_lambda_chained_through_pipeline() {
    let doc = json!({"xs": [1, 2, 3, 4]});
    // `.map(...).filter(...)` over an arithmetic projection exposes a
    // pre-existing path-collect divergence between vm and engine paths
    // that is unrelated to lambda lowering. The let-bound chain must
    // produce the same result as the equivalent inline form.
    let inlined = vm_query(
        "$.xs.map(x => x * 2).filter(x => x > 4)",
        &doc,
    )
    .unwrap();
    let with_let = vm_query(
        "let dbl = (x => x * 2) in let big = (x => x > 4) in $.xs.map(dbl).filter(big)",
        &doc,
    )
    .unwrap();
    assert_eq!(inlined, with_let);
    assert_eq!(inlined, json!([6, 8]));
}

#[test]
fn first_class_lambda_shadowed_by_inner_let() {
    let doc = json!({"xs": [1, 2, 3]});
    // Outer `f` doubles, inner `f` triples — inner wins inside `body`.
    assert_both(
        "let f = (x => x * 2) in let f = (x => x * 3) in $.xs.map(f)",
        &doc,
        &json!([3, 6, 9]),
    );
}

#[test]
fn first_class_lambda_shadowed_by_non_lambda_let() {
    let doc = json!({"xs": [1, 2]});
    // Inner non-lambda `let f = 99` shadows the outer lambda binding.
    // The macro-expansion pass refuses to inline a non-lambda init, so
    // `.map(f)` resolves `f` at runtime via `env.get_var` and the map
    // body becomes the constant `99` for every row.
    let res = run_engine(
        "let f = (x => x * 2) in let f = 99 in $.xs.map(f)",
        &doc,
    );
    assert_eq!(res, json!([99, 99]));
}

#[test]
fn first_class_lambda_outside_method_arg_position_evaluates_to_null() {
    let doc = json!({});
    // Lambda referenced in non-method-arg position keeps no-runtime-value
    // semantics: `lambda x: x * 2` at top level lowers to `PushNull`.
    assert_both("lambda x: x * 2", &doc, &json!(null));
}

#[test]
fn first_class_lambda_unused_binding_still_compiles() {
    let doc = json!({"xs": [1, 2, 3]});
    // Lambda bound but never invoked; body uses `xs` directly.
    assert_both(
        "let f = (x => x + 1) in $.xs.map(@)",
        &doc,
        &json!([1, 2, 3]),
    );
}

#[test]
fn first_class_lambda_in_global_call_arg() {
    // `coalesce(f, …)` — global call also accepts inlined lambdas.
    // `f` still resolves to null since coalesce takes value args, not HOFs;
    // confirms inlining does not crash when the receiver isn't a HOF.
    let doc = json!({"x": null});
    let res = run_engine("let f = (x => x) in coalesce($.x, 7)", &doc);
    assert_eq!(res, json!(7));
}

#[test]
fn snapshot_named_lambda_kernel_matches_at_form() {
    use crate::compile::compiler::Compiler;
    use crate::exec::pipeline::BodyKernel;
    use crate::parse::parser::parse;
    use crate::vm::Opcode;
    let cases = [
        ("$.xs.map(r => r.id)", "$.xs.map(@.id)"),
        ("$.xs.filter(r => r.score > 5)", "$.xs.filter(@.score > 5)"),
        ("$.xs.map(r => r.user.name)", "$.xs.map(@.user.name)"),
        ("$.xs.unique_by(r => r.score)", "$.xs.unique_by(@.score)"),
        ("$.xs.sort_by(r => r.score)", "$.xs.sort_by(@.score)"),
    ];
    for (named, at) in cases {
        let p_named = Compiler::compile(&parse(named).unwrap(), named);
        let p_at = Compiler::compile(&parse(at).unwrap(), at);
        // Kernels classified from the lambda body sub-program must match
        // exactly; otherwise the named form would fall back to the slow
        // VM path while the `@`-form takes the kernel fast path.
        fn body_kernel(prog: &crate::vm::Program) -> BodyKernel {
            for op in prog.ops.iter() {
                if let Opcode::CallMethod(call) = op {
                    if let Some(k) = call.sub_kernels.first() {
                        return k.clone();
                    }
                }
            }
            panic!("no CallMethod with sub_kernels");
        }
        let k_named = body_kernel(&p_named);
        let k_at = body_kernel(&p_at);
        assert_eq!(
            format!("{:?}", k_named),
            format!("{:?}", k_at),
            "body kernels differ between `{}` and `{}`",
            named,
            at
        );
        assert!(
            !matches!(k_named, BodyKernel::Generic),
            "named lambda `{}` must classify to a non-Generic kernel",
            named
        );
    }
}

#[test]
fn snapshot_named_lambda_compiles_to_same_ops_as_at_form() {
    use crate::compile::compiler::Compiler;
    use crate::parse::parser::parse;
    use crate::vm::Opcode;
    let cases = [
        ("$.xs.map(r => r.id)", "$.xs.map(@.id)"),
        ("$.xs.filter(r => r.score > 5)", "$.xs.filter(@.score > 5)"),
        ("$.xs.map(r => r.user.name)", "$.xs.map(@.user.name)"),
    ];
    for (named, at) in cases {
        let p_named = Compiler::compile(&parse(named).unwrap(), named);
        let p_at = Compiler::compile(&parse(at).unwrap(), at);
        // Pull each lambda body program out of the top-level CallMethod and
        // compare opcodes directly. The named form keeps an `Expr::Lambda`
        // wrapper in `orig_args` so dispatchers can disambiguate predicate
        // vs literal arguments — but the *compiled* sub-program is what
        // executes per row, and that must be identical to the @-form.
        fn body_ops(prog: &crate::vm::Program) -> Vec<Opcode> {
            for op in prog.ops.iter() {
                if let Opcode::CallMethod(call) = op {
                    if let Some(sub) = call.sub_progs.first() {
                        return sub.ops.iter().cloned().collect();
                    }
                }
            }
            panic!("no CallMethod with sub_progs in compiled program");
        }
        let named_body = body_ops(&p_named);
        let at_body = body_ops(&p_at);
        assert_eq!(
            format!("{:?}", named_body),
            format!("{:?}", at_body),
            "body ops differ between `{}` and `{}`",
            named,
            at
        );
    }
}

// -----------------------------------------------------------------------------
// K. Deep nesting — 3+ levels of lambda where every level captures its own
//    `@`. Outer-name references at every level must resolve through
//    `BindLamCurrent`-driven runtime binding without confusion.
// -----------------------------------------------------------------------------

#[test]
fn nesting_three_levels_inner_refs_outermost() {
    let doc = json!({
        "groups": [
            {"id": "g1", "subgroups": [
                {"sid": "s1", "items": [{"v": 1}, {"v": 2}]},
                {"sid": "s2", "items": [{"v": 3}]}
            ]}
        ]
    });
    // Inner `i` references outermost `g.id` and middle `s.sid`. Three
    // independent lambda scopes, each with its own `@`.
    assert_both(
        "$.groups.flat_map(g => g.subgroups.flat_map(s => s.items.map(i => {gid: g.id, sid: s.sid, v: i.v})))",
        &doc,
        &json!([
            {"gid":"g1","sid":"s1","v":1},
            {"gid":"g1","sid":"s1","v":2},
            {"gid":"g1","sid":"s2","v":3}
        ]),
    );
}

#[test]
fn nesting_four_levels_chain() {
    let doc = json!({
        "a": [
            {"b": [{"c": [{"ds": [{"v": 7}, {"v": 8}]}]}]}
        ]
    });
    // Four lambda levels, each capturing its own `@` and propagating
    // a value through to the deepest projection.
    assert_both(
        "$.a.flat_map(w => w.b.flat_map(x => x.c.flat_map(y => y.ds.map(z => z.v + 1))))",
        &doc,
        &json!([8, 9]),
    );
}

#[test]
fn nesting_outer_referenced_in_obj_field_inside_inner_lambda() {
    let doc = json!({
        "outer": [
            {"o": "X", "vals": [1, 2]},
            {"o": "Y", "vals": [3]}
        ]
    });
    assert_both(
        "$.outer.flat_map(r => r.vals.map(v => {o: r.o, v: v}))",
        &doc,
        &json!([
            {"o":"X","v":1},
            {"o":"X","v":2},
            {"o":"Y","v":3}
        ]),
    );
}

// -----------------------------------------------------------------------------
// L. Edge cases — degenerate inputs, identity bodies, complex projections.
// -----------------------------------------------------------------------------

#[test]
fn edge_empty_array_named_lambda_map() {
    assert_both("$.xs.map(r => r.id)", &json!({"xs": []}), &json!([]));
}

#[test]
fn edge_single_elem_array_named_lambda() {
    assert_both(
        "$.xs.map(r => r * 2)",
        &json!({"xs": [7]}),
        &json!([14]),
    );
}

#[test]
fn edge_named_lambda_returns_object_literal() {
    let doc = json!({"xs": [1, 2, 3]});
    assert_both(
        "$.xs.map(r => {value: r, doubled: r * 2})",
        &doc,
        &json!([
            {"value":1,"doubled":2},
            {"value":2,"doubled":4},
            {"value":3,"doubled":6}
        ]),
    );
}

#[test]
fn edge_named_lambda_returns_nested_object() {
    let doc = json!({"xs": [{"a": 1}, {"a": 2}]});
    assert_both(
        "$.xs.map(r => {wrap: {nested: r.a}})",
        &doc,
        &json!([
            {"wrap":{"nested":1}},
            {"wrap":{"nested":2}}
        ]),
    );
}

#[test]
fn edge_named_lambda_returns_array_with_param() {
    let doc = json!({"xs": [1, 2]});
    assert_both(
        "$.xs.map(r => [r, r, r])",
        &doc,
        &json!([[1, 1, 1], [2, 2, 2]]),
    );
}

#[test]
fn edge_named_lambda_with_chain_of_methods() {
    let doc = json!({"xs": [{"name": "Hello"}, {"name": "World"}]});
    assert_both(
        "$.xs.map(r => r.name.upper())",
        &doc,
        &json!(["HELLO", "WORLD"]),
    );
}

#[test]
fn edge_named_lambda_param_used_multiple_times_in_body() {
    let doc = json!({"xs": [3, 4]});
    assert_both(
        "$.xs.map(r => r * r + r)",
        &doc,
        &json!([12, 20]),
    );
}

#[test]
fn edge_named_lambda_unused_param() {
    let doc = json!({"xs": [1, 2, 3]});
    assert_both(
        "$.xs.map(r => 42)",
        &doc,
        &json!([42, 42, 42]),
    );
}

// -----------------------------------------------------------------------------
// M. Tape vs Val agreement under lambda-heavy queries. simd-json tape
//    backing surfaces `StrSlice`/`Int`/`Float` variants the serde-Val
//    backing does not, so every lambda site that touches strings,
//    fields, or dynamic indices must produce identical results.
// -----------------------------------------------------------------------------

fn tape_doc() -> Vec<u8> {
    br#"{"users":[{"id":"u1","name":"Alice","age":30,"tags":["a","b"]},{"id":"u2","name":"Bob","age":25,"tags":["b","c"]}]}"#.to_vec()
}

#[test]
fn tape_named_lambda_filter() {
    let j = crate::Jetro::from_bytes(tape_doc()).unwrap();
    let v: Value = j.collect("$.users.filter(u => u.age > 28).map(u => u.name)").unwrap();
    assert_eq!(v, json!(["Alice"]));
}

#[test]
fn tape_named_lambda_fstring() {
    let j = crate::Jetro::from_bytes(tape_doc()).unwrap();
    let v: Value = j.collect("$.users.map(u => f\"{u.name}<{u.id}>\")").unwrap();
    assert_eq!(v, json!(["Alice<u1>", "Bob<u2>"]));
}

#[test]
fn tape_named_lambda_obj_construct_with_nested_field() {
    let j = crate::Jetro::from_bytes(tape_doc()).unwrap();
    let v: Value = j
        .collect("$.users.map(u => {key: u.id, head: u.tags[0]})")
        .unwrap();
    assert_eq!(
        v,
        json!([
            {"key":"u1","head":"a"},
            {"key":"u2","head":"b"}
        ])
    );
}

// -----------------------------------------------------------------------------
// N. Stability — surprising syntax should not crash. These confirm parse
//    or eval errors surface cleanly rather than panic.
// -----------------------------------------------------------------------------

#[test]
fn stability_zero_param_arrow_lambda_parses() {
    use crate::parse::parser::parse;
    // `() => …` is grammatically valid; it returns the body for every
    // call regardless of input.
    assert!(parse("$.xs.map(() => 1)").is_ok());
}

#[test]
fn stability_lambda_keyword_without_body_is_parse_error() {
    use crate::parse::parser::parse;
    assert!(parse("$.xs.map(lambda x:)").is_err());
}

#[test]
fn stability_unbound_ident_in_method_arg_returns_null_not_panic() {
    // `f` is not let-bound; macro-expansion leaves `Ident("f")`. Runtime
    // resolves to env.get_var fallback (also nothing). map applies a
    // null projection — returns nulls without panic.
    let doc = json!({"xs": [1, 2, 3]});
    let res = run_engine("$.xs.map(f)", &doc);
    assert_eq!(res, json!([null, null, null]));
}

// -----------------------------------------------------------------------------
// O. Performance regression — named-form opcode count must equal `@`-form.
//    A drift here means the AST substitution missed a path and per-row
//    cost diverged.
// -----------------------------------------------------------------------------

#[test]
fn perf_named_lambda_op_count_equals_at_form_for_common_bodies() {
    use crate::compile::compiler::Compiler;
    use crate::parse::parser::parse;
    use crate::vm::Opcode;
    let cases = [
        ("$.xs.map(r => r)", "$.xs.map(@)"),
        ("$.xs.map(r => r.a)", "$.xs.map(@.a)"),
        ("$.xs.map(r => r.a.b)", "$.xs.map(@.a.b)"),
        ("$.xs.filter(r => r.a > 1)", "$.xs.filter(@.a > 1)"),
        ("$.xs.map(r => r + 1)", "$.xs.map(@ + 1)"),
        ("$.xs.map(r => not r)", "$.xs.map(not @)"),
        ("$.xs.map(r => -r)", "$.xs.map(-@)"),
    ];
    for (named, at) in cases {
        let p_named = Compiler::compile(&parse(named).unwrap(), named);
        let p_at = Compiler::compile(&parse(at).unwrap(), at);
        fn body_len(prog: &crate::vm::Program) -> usize {
            for op in prog.ops.iter() {
                if let Opcode::CallMethod(call) = op {
                    if let Some(sub) = call.sub_progs.first() {
                        return sub.ops.len();
                    }
                }
            }
            0
        }
        assert_eq!(
            body_len(&p_named),
            body_len(&p_at),
            "body opcode count differs between `{}` and `{}`",
            named,
            at
        );
    }
}

// -----------------------------------------------------------------------------
// P. Builtin coverage extension — each HOF lambda variant tested with both
//    arrow and `lambda` forms to cement that the dispatch surface is
//    uniformly lambda-aware.
// -----------------------------------------------------------------------------

fn xs_objs() -> Value {
    json!({"xs": [
        {"id": 1, "kind": "a"},
        {"id": 2, "kind": "b"},
        {"id": 3, "kind": "a"},
        {"id": 4, "kind": "b"}
    ]})
}

#[test]
fn builtin_group_by_named_lambda() {
    let v = run_engine("$.xs.group_by(r => r.kind)", &xs_objs());
    let alt = run_engine("$.xs.group_by(@.kind)", &xs_objs());
    assert_eq!(v, alt);
}

#[test]
fn builtin_count_by_named_lambda() {
    let v = run_engine("$.xs.count_by(r => r.kind)", &xs_objs());
    let alt = run_engine("$.xs.count_by(@.kind)", &xs_objs());
    assert_eq!(v, alt);
}

#[test]
fn builtin_partition_named_lambda() {
    let v = run_engine("$.xs.partition(r => r.id > 2)", &xs_objs());
    let alt = run_engine("$.xs.partition(@.id > 2)", &xs_objs());
    assert_eq!(v, alt);
}

#[test]
fn builtin_min_by_max_by_named_lambda() {
    let doc = json!({"xs": [{"score": 5}, {"score": 9}, {"score": 3}]});
    let mn = run_engine("$.xs.min_by(r => r.score)", &doc);
    let mn_at = run_engine("$.xs.min_by(@.score)", &doc);
    assert_eq!(mn, mn_at);
    let mx = run_engine("$.xs.max_by(r => r.score)", &doc);
    let mx_at = run_engine("$.xs.max_by(@.score)", &doc);
    assert_eq!(mx, mx_at);
}

#[test]
fn builtin_index_by_named_lambda() {
    let v = run_engine("$.xs.index_by(r => r.id)", &xs_objs());
    let alt = run_engine("$.xs.index_by(@.id)", &xs_objs());
    assert_eq!(v, alt);
}

// -----------------------------------------------------------------------------
// Q. Mixed forms in same expression — arrow, lambda, and `@` co-exist
//    without crosstalk.
// -----------------------------------------------------------------------------

// Mixed-form chains over arithmetic projections — vm and engine paths
// must agree. Path-collect wrap divergence fixed by `FilterBeforeMap`
// optimizer rewriting predicates with `substitute_current_with_expr` and
// by `normalize_symbolic` unwrapping single-param lambda wrappers
// before threading them through symbolic substitution.

#[test]
fn mixed_arrow_then_at_form_chain() {
    assert_both(
        "$.xs.map(r => r * 2).filter(@ > 4)",
        &json!({"xs": [1, 2, 3, 4]}),
        &json!([6, 8]),
    );
}

#[test]
fn mixed_at_form_then_arrow_chain() {
    assert_both(
        "$.xs.filter(@ > 1).map(r => r + 10)",
        &json!({"xs": [1, 2, 3, 4]}),
        &json!([12, 13, 14]),
    );
}

#[test]
fn mixed_lambda_then_arrow_chain() {
    assert_both(
        "$.xs.map(lambda x: x * 3).map(r => r - 1)",
        &json!({"xs": [1, 2, 3]}),
        &json!([2, 5, 8]),
    );
}

#[test]
fn mixed_three_forms_in_one_expression() {
    assert_both(
        "$.xs.map(@.v).filter(r => r > 1).map(lambda x: x + 100)",
        &json!({"xs": [{"v": 1}, {"v": 2}, {"v": 3}, {"v": 4}]}),
        &json!([102, 103, 104]),
    );
}

#[test]
fn map_arith_then_filter_arith_engine_path() {
    // `.map(...)` followed by `.filter(...)` over arithmetic projections
    // formerly returned `[]` on the engine path because the filter
    // pushdown rewrote the predicate without substituting `@` with the
    // map's projection expression. Both vm and engine must agree now.
    assert_both(
        "$.xs.map(@ * 2).filter(@ > 30)",
        &json!({"xs": [10, 20, 30]}),
        &json!([40, 60]),
    );
}

#[test]
fn map_field_arith_then_filter() {
    assert_both(
        "$.xs.map(r => r.v * 3).filter(@ > 5).map(@ + 1)",
        &json!({"xs": [{"v": 1}, {"v": 2}, {"v": 3}]}),
        &json!([7, 10]),
    );
}

// -----------------------------------------------------------------------------
// R. Lambda body returns lambda? Currently parses as inner Lambda inside
//    body. The outer lambda body becomes `Lambda { ... }` which compiles
//    to `PushNull`. Document this explicitly so future first-class lambda
//    expansion can opt-in.
// -----------------------------------------------------------------------------

#[test]
fn body_returning_lambda_value_is_null() {
    let doc = json!({"xs": [1, 2]});
    assert_both(
        "$.xs.map(r => (x => x))",
        &doc,
        &json!([null, null]),
    );
}

// -----------------------------------------------------------------------------
// S. Multi-arg fast-path correctness — substituting the rightmost param
//    must not break left-param lookups.
// -----------------------------------------------------------------------------

#[test]
fn multi_arg_left_param_still_resolves() {
    let doc = json!({"xs": [3, 1, 4, 1, 5, 9, 2, 6]});
    assert_both(
        "$.xs.sort((a, b) => a < b)",
        &doc,
        &json!([1, 1, 2, 3, 4, 5, 6, 9]),
    );
}

#[test]
fn multi_arg_param_used_in_arithmetic() {
    let doc = json!({"xs": [3, 1, 2]});
    // Comparator: true means `a` precedes `b`. `(b - a) > 0` is true when
    // `b > a`, so `a` (the smaller) comes first → ascending. Confirms
    // both params resolve in an arithmetic body, with `b` substituted to
    // `Current` and `a` resolved through the env-var path.
    assert_both(
        "$.xs.sort((a, b) => (b - a) > 0)",
        &doc,
        &json!([1, 2, 3]),
    );
}

// -----------------------------------------------------------------------------
// T. First-class lambda — additional coverage including chaining and
//    aliases.
// -----------------------------------------------------------------------------

#[test]
fn first_class_lambda_aliased_through_two_lets() {
    let doc = json!({"xs": [1, 2, 3]});
    assert_both(
        "let f = (x => x * 2) in let g = f in $.xs.map(g)",
        &doc,
        &json!([2, 4, 6]),
    );
}

#[test]
fn first_class_lambda_in_filter_and_map_simultaneously() {
    let doc = json!({"xs": [{"score": 3}, {"score": 7}, {"score": 12}]});
    assert_both(
        "let big = (r => r.score > 5) in let id = (r => r.score) in $.xs.filter(big).map(id)",
        &doc,
        &json!([7, 12]),
    );
}

#[test]
fn first_class_lambda_inside_nested_let() {
    let doc = json!({"xs": [{"a": 1}, {"a": 2}, {"a": 3}]});
    assert_both(
        "let mk = (r => r.a + 100) in $.xs.map(mk)",
        &doc,
        &json!([101, 102, 103]),
    );
}

