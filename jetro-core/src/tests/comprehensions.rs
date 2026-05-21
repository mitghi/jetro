//! Comprehensive coverage for list / dict / set / generator comprehensions.
//!
//! Comprehensions historically had two issues that this suite locks down:
//!
//! 1. **Path-source iteration.** Iterating `$.xs` (where `$.xs` is a path
//!    that evaluates to a columnar `Val::IntVec` / `FloatVec` / `StrVec` /
//!    `StrSliceVec` / `ObjVec`) used to fall through `exec_iter_vals` and
//!    produce a single-element wrapping instead of per-element iteration.
//!    The fix is in `vm::exec::exec_iter_vals`.
//!
//! 2. **Two-variable destructure.** `for k, v in [[k1,v1],...]` used to look
//!    for `index`/`value` keys on the item, returning null/null pairs for
//!    array elements. `bind_comp_vars` now binds from arrays as a pair.
//!
//! 3. **Multiple `if` clauses.** The grammar previously allowed only one
//!    trailing `if` per comprehension. Multiple `if`s are now folded into a
//!    conjunction at parse time.
//!
//! 4. **Array-pattern destructure.** `for [k, v] in pairs` is sugar for the
//!    `for k, v in pairs` form. Both now parse.
//!
//! Tests cover list, dict, set, and generator comprehensions for each of
//! these axes, plus standard regressions over scalar, string, object, and
//! nested sources.
//!
//! Conventions:
//! - `vm_query` runs through the bytecode VM with a fresh state.
//! - `assert_eq!(out, json!(...))` for exact JSON equality.

use super::common::vm_query;
use serde_json::json;

// ════════════════════════════════════════════════════════════════════════════
// 1. List comprehensions
// ════════════════════════════════════════════════════════════════════════════

mod list {
    use super::*;

    #[test]
    fn literal_source_simple() {
        let out = vm_query("[n for n in [1,2,3]]", &json!(null)).unwrap();
        assert_eq!(out, json!([1, 2, 3]));
    }

    #[test]
    fn literal_source_transform() {
        let out = vm_query("[n*n for n in [1,2,3]]", &json!(null)).unwrap();
        assert_eq!(out, json!([1, 4, 9]));
    }

    #[test]
    fn path_source_int_vec() {
        // `$.xs` resolves to a `Val::IntVec` columnar fast path. Iteration
        // must descend into elements, not yield the whole vec as one item.
        let doc = json!({"xs": [1, 2, 3, 4, 5]});
        let out = vm_query("[n for n in $.xs]", &doc).unwrap();
        assert_eq!(out, json!([1, 2, 3, 4, 5]));
    }

    #[test]
    fn path_source_int_vec_transform() {
        let doc = json!({"xs": [1, 2, 3, 4, 5]});
        let out = vm_query("[n*n for n in $.xs]", &doc).unwrap();
        assert_eq!(out, json!([1, 4, 9, 16, 25]));
    }

    #[test]
    fn path_source_float_vec() {
        let doc = json!({"xs": [1.5, 2.5, 3.5]});
        let out = vm_query("[x for x in $.xs]", &doc).unwrap();
        assert_eq!(out, json!([1.5, 2.5, 3.5]));
    }

    #[test]
    fn path_source_str_vec() {
        let doc = json!({"xs": ["a", "b", "c"]});
        let out = vm_query("[s for s in $.xs]", &doc).unwrap();
        assert_eq!(out, json!(["a", "b", "c"]));
    }

    #[test]
    fn path_source_obj_vec() {
        // `Val::ObjVec` columnar form (uniform-key objects).
        let doc = json!({"xs": [{"x": 1, "y": 10}, {"x": 2, "y": 20}, {"x": 3, "y": 30}]});
        let out = vm_query("[o.x for o in $.xs]", &doc).unwrap();
        assert_eq!(out, json!([1, 2, 3]));
    }

    #[test]
    fn path_source_after_method() {
        // Iter source produced by a method chain (`reverse`).
        let doc = json!({"xs": [1, 2, 3]});
        let out = vm_query("[n for n in $.xs.reverse()]", &doc).unwrap();
        assert_eq!(out, json!([3, 2, 1]));
    }

    #[test]
    fn single_if_clause() {
        let doc = json!({"xs": [1, 2, 3, 4, 5]});
        let out = vm_query("[n for n in $.xs if n > 2]", &doc).unwrap();
        assert_eq!(out, json!([3, 4, 5]));
    }

    #[test]
    fn two_if_clauses_conjoined() {
        let doc = json!({"xs": [1, 2, 3, 4, 5, 6, 7]});
        let out = vm_query("[n for n in $.xs if n > 1 if n < 5]", &doc).unwrap();
        assert_eq!(out, json!([2, 3, 4]));
    }

    #[test]
    fn three_if_clauses_conjoined() {
        let doc = json!({"xs": [1, 2, 3, 4, 5, 6, 7]});
        let out = vm_query("[n for n in $.xs if n > 1 if n < 7 if n != 4]", &doc).unwrap();
        assert_eq!(out, json!([2, 3, 5, 6]));
    }

    #[test]
    fn empty_source_yields_empty() {
        let doc = json!({"xs": []});
        let out = vm_query("[n*2 for n in $.xs]", &doc).unwrap();
        assert_eq!(out, json!([]));
    }

    #[test]
    fn empty_source_with_cond() {
        let doc = json!({"xs": []});
        let out = vm_query("[n for n in $.xs if n > 0]", &doc).unwrap();
        assert_eq!(out, json!([]));
    }

    #[test]
    fn cond_filters_all_out() {
        let doc = json!({"xs": [1, 2, 3]});
        let out = vm_query("[n for n in $.xs if n > 100]", &doc).unwrap();
        assert_eq!(out, json!([]));
    }

    #[test]
    fn ternary_in_body() {
        let doc = json!({"xs": [1, 5, 10]});
        let out = vm_query(r#"["big" if n > 4 else "small" for n in $.xs]"#, &doc).unwrap();
        assert_eq!(out, json!(["small", "big", "big"]));
    }

    #[test]
    fn fstring_in_body() {
        let doc = json!({"xs": [1, 2]});
        let out = vm_query(r#"[f"item-{n}" for n in $.xs]"#, &doc).unwrap();
        assert_eq!(out, json!(["item-1", "item-2"]));
    }

    #[test]
    fn method_call_on_var() {
        let doc = json!({"xs": ["foo", "bar"]});
        let out = vm_query("[s.upper() for s in $.xs]", &doc).unwrap();
        assert_eq!(out, json!(["FOO", "BAR"]));
    }

    #[test]
    fn nested_field_access_on_var() {
        let doc = json!({"users": [{"profile": {"name": "Ada"}}, {"profile": {"name": "Bob"}}]});
        let out = vm_query("[u.profile.name for u in $.users]", &doc).unwrap();
        assert_eq!(out, json!(["Ada", "Bob"]));
    }


    #[test]
    fn two_var_destructure_arrays_of_pairs() {
        // `for k, v in [[..],[..]]` binds first/second elements.
        let doc = json!({"pairs": [["a", 1], ["b", 2], ["c", 3]]});
        let out = vm_query("[k for k, v in $.pairs]", &doc).unwrap();
        assert_eq!(out, json!(["a", "b", "c"]));
    }

    #[test]
    fn two_var_destructure_value_only() {
        let doc = json!({"pairs": [["a", 1], ["b", 2]]});
        let out = vm_query("[v for k, v in $.pairs]", &doc).unwrap();
        assert_eq!(out, json!([1, 2]));
    }

    #[test]
    fn two_var_array_pattern_destructure() {
        // `for [k, v] in ...` — explicit array-pattern form, sugar for 2-var.
        let doc = json!({"pairs": [["a", 1], ["b", 2]]});
        let out = vm_query("[k for [k, v] in $.pairs]", &doc).unwrap();
        assert_eq!(out, json!(["a", "b"]));
    }

    #[test]
    fn two_var_array_pattern_value() {
        let doc = json!({"pairs": [["a", 1], ["b", 2]]});
        let out = vm_query("[v for [k, v] in $.pairs]", &doc).unwrap();
        assert_eq!(out, json!([1, 2]));
    }

    #[test]
    fn two_var_arithmetic_on_pair() {
        let out = vm_query("[a*b for [a, b] in [[2,3],[4,5]]]", &json!(null)).unwrap();
        assert_eq!(out, json!([6, 20]));
    }

    #[test]
    fn two_var_with_filter() {
        let doc = json!({"pairs": [["a", 1], ["b", 2], ["c", 3]]});
        let out = vm_query("[k for [k, v] in $.pairs if v >= 2]", &doc).unwrap();
        assert_eq!(out, json!(["b", "c"]));
    }


    #[test]
    fn object_source_yields_entries() {
        // Iterating an object produces `{key, value}` records.
        let doc = json!({"o": {"a": 1, "b": 2}});
        let out = vm_query("[e.value for e in $.o]", &doc).unwrap();
        assert_eq!(out, json!([1, 2]));
    }

    #[test]
    fn object_source_keys() {
        let doc = json!({"o": {"a": 1, "b": 2}});
        let out = vm_query("[e.key for e in $.o]", &doc).unwrap();
        assert_eq!(out, json!(["a", "b"]));
    }


    #[test]
    fn iter_via_let_binding() {
        // Verifies the let-binding short path that worked before the fix
        // continues to work.
        let doc = json!({"xs": [1, 2, 3]});
        let out = vm_query("let x = $.xs in [n*2 for n in x]", &doc).unwrap();
        assert_eq!(out, json!([2, 4, 6]));
    }

    #[test]
    fn nested_comprehension() {
        // Inner comp inside outer body.
        let doc = json!({"rows": [[1, 2], [3, 4]]});
        let out = vm_query("[[n*10 for n in row] for row in $.rows]", &doc).unwrap();
        assert_eq!(out, json!([[10, 20], [30, 40]]));
    }

    #[test]
    fn cond_referring_to_var_method() {
        let doc = json!({"strs": ["foo", "bar", "baz"]});
        let out = vm_query(r#"[s for s in $.strs if s.starts_with("b")]"#, &doc).unwrap();
        assert_eq!(out, json!(["bar", "baz"]));
    }
}

// ════════════════════════════════════════════════════════════════════════════
// 2. Dict comprehensions
// ════════════════════════════════════════════════════════════════════════════

mod dict {
    use super::*;

    #[test]
    fn simple_literal() {
        let out = vm_query(r#"{k: 1 for k in ["a","b","c"]}"#, &json!(null)).unwrap();
        assert_eq!(out, json!({"a": 1, "b": 1, "c": 1}));
    }

    #[test]
    fn key_to_self_squared() {
        // Note: keys coerce to strings.
        let out = vm_query("{n: n*n for n in [1,2,3]}", &json!(null)).unwrap();
        assert_eq!(out, json!({"1": 1, "2": 4, "3": 9}));
    }

    #[test]
    fn path_source_int_vec() {
        // Was previously affected by the Val::IntVec iter fallthrough.
        let doc = json!({"xs": [1, 2, 3]});
        let out = vm_query("{n: n for n in $.xs}", &doc).unwrap();
        assert_eq!(out, json!({"1": 1, "2": 2, "3": 3}));
    }

    #[test]
    fn cond_filter() {
        let doc = json!({"xs": [1, 2, 3, 4, 5]});
        let out = vm_query("{n: n for n in $.xs if n > 3}", &doc).unwrap();
        assert_eq!(out, json!({"4": 4, "5": 5}));
    }

    #[test]
    fn multiple_if_clauses() {
        let doc = json!({"xs": [1, 2, 3, 4, 5]});
        let out = vm_query("{n: n*n for n in $.xs if n > 1 if n < 5}", &doc).unwrap();
        assert_eq!(out, json!({"2": 4, "3": 9, "4": 16}));
    }

    #[test]
    fn two_var_comma_form() {
        let doc = json!({"pairs": [["a", 1], ["b", 2]]});
        let out = vm_query("{k: v for k, v in $.pairs}", &doc).unwrap();
        assert_eq!(out, json!({"a": 1, "b": 2}));
    }

    #[test]
    fn two_var_array_pattern_form() {
        let doc = json!({"pairs": [["a", 1], ["b", 2]]});
        let out = vm_query("{k: v for [k, v] in $.pairs}", &doc).unwrap();
        assert_eq!(out, json!({"a": 1, "b": 2}));
    }

    #[test]
    fn two_var_swap_kv() {
        // Build a value→key inverse.
        let out = vm_query(
            r#"{v: k for [k, v] in [["x",1],["y",2]]}"#,
            &json!(null),
        )
        .unwrap();
        assert_eq!(out, json!({"1": "x", "2": "y"}));
    }

    #[test]
    fn object_source_iterates_entries() {
        // Iterating an object yields {key, value} records.
        let doc = json!({"o": {"a": 1, "b": 2}});
        let out = vm_query("{e.key: e.value*10 for e in $.o}", &doc).unwrap();
        assert_eq!(out, json!({"a": 10, "b": 20}));
    }

    #[test]
    fn fstring_key() {
        let doc = json!({"xs": [1, 2]});
        let out = vm_query(r#"{f"k-{n}": n for n in $.xs}"#, &doc).unwrap();
        assert_eq!(out, json!({"k-1": 1, "k-2": 2}));
    }

    #[test]
    fn empty_source() {
        let out = vm_query("{n: n for n in []}", &json!(null)).unwrap();
        assert_eq!(out, json!({}));
    }

    #[test]
    fn duplicate_keys_last_wins() {
        // Two pairs collapse to one (last value wins).
        let out = vm_query(
            r#"{k: v for [k, v] in [["a",1],["a",2]]}"#,
            &json!(null),
        )
        .unwrap();
        assert_eq!(out, json!({"a": 2}));
    }

    #[test]
    fn cond_skip_drops_key() {
        let doc = json!({"pairs": [["a", 1], ["b", 2], ["c", 3]]});
        let out = vm_query("{k: v for [k, v] in $.pairs if v != 2}", &doc).unwrap();
        assert_eq!(out, json!({"a": 1, "c": 3}));
    }

    #[test]
    fn conditional_via_ternary_in_value() {
        let doc = json!({"xs": [1, 2, 3]});
        let out = vm_query(
            r#"{n: "yes" if n > 1 else "no" for n in $.xs}"#,
            &doc,
        )
        .unwrap();
        assert_eq!(out, json!({"1": "no", "2": "yes", "3": "yes"}));
    }
}

// ════════════════════════════════════════════════════════════════════════════
// 3. Set comprehensions
// ════════════════════════════════════════════════════════════════════════════

mod set {
    use super::*;

    #[test]
    fn simple_dedup() {
        let out = vm_query("{n for n in [1,1,2,2,3]}", &json!(null)).unwrap();
        assert_eq!(out, json!([1, 2, 3]));
    }

    #[test]
    fn transform_dedup() {
        let out = vm_query("{n*n for n in [-1,1,-2,2,-3,3]}", &json!(null)).unwrap();
        assert_eq!(out, json!([1, 4, 9]));
    }

    #[test]
    fn path_source() {
        let doc = json!({"xs": [1, 2, 2, 3, 3, 3]});
        let out = vm_query("{n for n in $.xs}", &doc).unwrap();
        assert_eq!(out, json!([1, 2, 3]));
    }

    #[test]
    fn with_filter() {
        let doc = json!({"strs": ["foo", "bar", "foo"]});
        let out = vm_query(r#"{s for s in $.strs if s != "bar"}"#, &doc).unwrap();
        assert_eq!(out, json!(["foo"]));
    }

    #[test]
    fn multi_if_clauses() {
        let out = vm_query("{n*n for n in [1,2,3,4,5,6] if n > 1 if n < 5}", &json!(null))
            .unwrap();
        assert_eq!(out, json!([4, 9, 16]));
    }
}

// ════════════════════════════════════════════════════════════════════════════
// 4. Generator comprehensions
// ════════════════════════════════════════════════════════════════════════════

mod gen {
    use super::*;

    #[test]
    fn simple() {
        let out = vm_query("(n for n in [1,2,3])", &json!(null)).unwrap();
        assert_eq!(out, json!([1, 2, 3]));
    }

    #[test]
    fn path_source() {
        let doc = json!({"xs": [1, 2, 3]});
        let out = vm_query("(n*2 for n in $.xs)", &doc).unwrap();
        assert_eq!(out, json!([2, 4, 6]));
    }

    #[test]
    fn with_cond() {
        let doc = json!({"xs": [1, 2, 3, 4, 5]});
        let out = vm_query("(n for n in $.xs if n > 2)", &doc).unwrap();
        assert_eq!(out, json!([3, 4, 5]));
    }

    #[test]
    fn multi_if() {
        let doc = json!({"xs": [1, 2, 3, 4, 5]});
        let out = vm_query("(n for n in $.xs if n > 1 if n < 5)", &doc).unwrap();
        assert_eq!(out, json!([2, 3, 4]));
    }

    #[test]
    fn array_pattern_destructure() {
        let out = vm_query("(a+b for [a, b] in [[1,2],[3,4]])", &json!(null)).unwrap();
        assert_eq!(out, json!([3, 7]));
    }
}

// ════════════════════════════════════════════════════════════════════════════
// 5. Cross-cutting interactions
// ════════════════════════════════════════════════════════════════════════════

mod misc {
    use super::*;

    #[test]
    fn comprehension_inside_method_arg() {
        let doc = json!({"xs": [1, 2, 3, 4]});
        let out = vm_query(
            "[n for n in $.xs if n > 1].len()",
            &doc,
        )
        .unwrap();
        assert_eq!(out, json!(3));
    }

    #[test]
    fn comprehension_pipes_into_reducer() {
        let doc = json!({"xs": [1, 2, 3, 4]});
        let out = vm_query("[n*n for n in $.xs].sum()", &doc).unwrap();
        assert_eq!(out, json!(30));
    }

    #[test]
    fn list_comp_then_filter_chain() {
        let doc = json!({"xs": [1, 2, 3, 4, 5]});
        let out = vm_query("[n*n for n in $.xs].filter(@ > 5)", &doc).unwrap();
        assert_eq!(out, json!([9, 16, 25]));
    }

    #[test]
    fn dict_comp_then_entries_chain() {
        let doc = json!({"xs": [1, 2, 3]});
        let out = vm_query(
            "{n: n*n for n in $.xs}.entries().count()",
            &doc,
        )
        .unwrap();
        assert_eq!(out, json!(3));
    }

    #[test]
    fn comp_var_shadows_outer_let() {
        // Inner `n` shadows outer `n`.
        let out = vm_query(
            "let n = 100 in [n for n in [1,2,3]]",
            &json!(null),
        )
        .unwrap();
        assert_eq!(out, json!([1, 2, 3]));
    }

    #[test]
    fn comp_inherits_outer_let_in_cond() {
        // Outer let-bound var visible in cond expression.
        let out = vm_query(
            "let cutoff = 2 in [n for n in [1,2,3,4] if n > cutoff]",
            &json!(null),
        )
        .unwrap();
        assert_eq!(out, json!([3, 4]));
    }

    #[test]
    fn comp_as_pipeline_source() {
        // Comprehension feeds a pipeline.
        let doc = json!({"xs": [1, 2, 3, 4, 5]});
        let out = vm_query(
            "[n*2 for n in $.xs if n > 2].map(@ + 1)",
            &doc,
        )
        .unwrap();
        assert_eq!(out, json!([7, 9, 11]));
    }

    #[test]
    fn deeply_nested_comp() {
        // Triple nesting.
        let out = vm_query(
            "[[[k for k in [1,2]] for j in [1]] for i in [1]]",
            &json!(null),
        )
        .unwrap();
        assert_eq!(out, json!([[[1, 2]]]));
    }

    #[test]
    fn comp_in_let_body_inherits_root() {
        let doc = json!({"xs": [10, 20, 30]});
        let out = vm_query(
            "let f = (x => x + 1) in [n for n in $.xs]",
            &doc,
        )
        .unwrap();
        assert_eq!(out, json!([10, 20, 30]));
    }

    #[test]
    fn condition_uses_root_lookup() {
        // Predicate references `$` — comp var must not shadow it.
        let doc = json!({"xs": [1, 2, 3, 4, 5], "limit": 3});
        let out = vm_query("[n for n in $.xs if n <= $.limit]", &doc).unwrap();
        assert_eq!(out, json!([1, 2, 3]));
    }
}
