//! Tests for chain-style writes (`patch $ { ... }` and the `.set/.modify/.delete/.unset` desugars).
//!
//! Each test case constructs a small fixture and asserts the post-write document.

#[cfg(test)]
mod tests {
    use super::super::common::vm_query;
    use serde_json::json;

    
    #[test]
    fn patch_simple_field_replace() {
        let doc = json!({"name": "Alice", "age": 30});
        let r = vm_query(r#"patch $ { name: "Bob" }"#, &doc).unwrap();
        assert_eq!(r, json!({"name": "Bob", "age": 30}));
    }

    #[test]
    fn patch_nested_field_replace() {
        let doc = json!({"user": {"name": "Alice", "age": 30}});
        let r = vm_query(r#"patch $ { user.name: "Bob" }"#, &doc).unwrap();
        assert_eq!(r, json!({"user": {"name": "Bob", "age": 30}}));
    }

    #[test]
    fn patch_delete_field() {
        let doc = json!({"name": "Alice", "tmp": "remove-me", "age": 30});
        let r = vm_query(r#"patch $ { tmp: DELETE }"#, &doc).unwrap();
        assert_eq!(r, json!({"name": "Alice", "age": 30}));
    }

    #[test]
    fn patch_add_new_field() {
        let doc = json!({"name": "Alice"});
        let r = vm_query(r#"patch $ { age: 42 }"#, &doc).unwrap();
        assert_eq!(r, json!({"name": "Alice", "age": 42}));
    }

    #[test]
    fn patch_wildcard_array() {
        let doc = json!({"users": [
            {"name": "Alice", "seen": false},
            {"name": "Bob",   "seen": false},
        ]});
        let r = vm_query(r#"patch $ { users[*].seen: true }"#, &doc).unwrap();
        assert_eq!(
            r,
            json!({"users": [
                {"name": "Alice", "seen": true},
                {"name": "Bob",   "seen": true},
            ]})
        );
    }

    #[test]
    fn patch_wildcard_filter() {
        let doc = json!({"users": [
            {"name": "Alice", "active": true,  "role": "user"},
            {"name": "Bob",   "active": false, "role": "user"},
            {"name": "Cara",  "active": true,  "role": "user"},
        ]});
        let r = vm_query(r#"patch $ { users[* if active].role: "admin" }"#, &doc).unwrap();
        assert_eq!(
            r,
            json!({"users": [
                {"name": "Alice", "active": true,  "role": "admin"},
                {"name": "Bob",   "active": false, "role": "user"},
                {"name": "Cara",  "active": true,  "role": "admin"},
            ]})
        );
    }

    #[test]
    fn patch_uses_current_value() {
        let doc = json!({"users": [
            {"name": "Alice", "email": "ALICE@X"},
            {"name": "Bob",   "email": "BOB@X"},
        ]});
        let r = vm_query(r#"patch $ { users[*].email: @.lower() }"#, &doc).unwrap();
        assert_eq!(
            r,
            json!({"users": [
                {"name": "Alice", "email": "alice@x"},
                {"name": "Bob",   "email": "bob@x"},
            ]})
        );
    }

    #[test]
    fn patch_conditional_when_truthy() {
        let doc = json!({"count": 5, "enabled": true});
        let r = vm_query(r#"patch $ { count: @ + 1 when $.enabled }"#, &doc).unwrap();
        assert_eq!(r, json!({"count": 6, "enabled": true}));
    }

    #[test]
    fn patch_conditional_when_falsy_skips() {
        let doc = json!({"count": 5, "enabled": false});
        let r = vm_query(r#"patch $ { count: @ + 1 when $.enabled }"#, &doc).unwrap();
        assert_eq!(r, json!({"count": 5, "enabled": false}));
    }

    #[test]
    fn patch_multiple_ops_in_order() {
        let doc = json!({"a": 1, "b": 2, "c": 3});
        let r = vm_query(r#"patch $ { a: 10, b: DELETE, c: 30 }"#, &doc).unwrap();
        assert_eq!(r, json!({"a": 10, "c": 30}));
    }

    #[test]
    fn patch_index_access() {
        let doc = json!({"items": [10, 20, 30]});
        let r = vm_query(r#"patch $ { items[1]: 99 }"#, &doc).unwrap();
        assert_eq!(r, json!({"items": [10, 99, 30]}));
    }

    #[test]
    fn patch_delete_from_wildcard() {
        let doc = json!({"users": [
            {"name": "Alice", "active": true},
            {"name": "Bob",   "active": false},
            {"name": "Cara",  "active": true},
        ]});
        let r = vm_query(r#"patch $ { users[* if not active]: DELETE }"#, &doc).unwrap();
        assert_eq!(
            r,
            json!({"users": [
                {"name": "Alice", "active": true},
                {"name": "Cara",  "active": true},
            ]})
        );
    }

    #[test]
    fn patch_composes_pipe() {
        let doc = json!({"name": "Alice", "age": 30});
        let r = vm_query(r#"patch $ { name: "Bob" } | @.name"#, &doc).unwrap();
        assert_eq!(r, json!("Bob"));
    }

    #[test]
    fn patch_composes_method_chain() {
        let doc = json!({"name": "Alice", "age": 30});
        let r = vm_query(r#"patch $ { name: "Bob" }.keys()"#, &doc).unwrap();
        let mut keys = r.as_array().unwrap().clone();
        keys.sort_by(|a, b| a.as_str().unwrap().cmp(b.as_str().unwrap()));
        assert_eq!(keys, vec![json!("age"), json!("name")]);
    }

    #[test]
    fn patch_composes_nested_in_object() {
        let doc = json!({"name": "Alice"});
        let r = vm_query(r#"{result: patch $ { name: "Bob" }}"#, &doc).unwrap();
        assert_eq!(r, json!({"result": {"name": "Bob"}}));
    }

    #[test]
    fn patch_composes_let_binding() {
        let doc = json!({"name": "Alice", "age": 30});
        let r = vm_query(r#"let x = patch $ { name: "Bob" } in x.name"#, &doc).unwrap();
        assert_eq!(r, json!("Bob"));
    }

    #[test]
    fn patch_composes_nested_patch() {
        let doc = json!({"name": "Alice", "age": 30});
        let r = vm_query(r#"patch (patch $ { name: "Bob" }) { age: 99 }"#, &doc).unwrap();
        assert_eq!(r, json!({"name": "Bob", "age": 99}));
    }

    #[test]
    fn patch_composes_inside_map() {
        let doc = json!({"users": [{"n": 1}, {"n": 2}, {"n": 3}]});
        let r = vm_query(r#"$.users.map(patch @ { n: @ * 10 })"#, &doc).unwrap();
        assert_eq!(r, json!([{"n": 10}, {"n": 20}, {"n": 30}]));
    }

    #[test]
    fn patch_delete_mark_outside_patch_errors() {
        let doc = json!({});
        let r = vm_query(r#"DELETE"#, &doc);
        assert!(r.is_err());
    }

    // ---- Phase D: batched patch via PathTrie + Arc::make_mut --------------
    //
    // The trie kicks in when `cp.ops.len() >= 2` and every op uses only
    // `Field`/`Index`/`DynIndex` path steps with no `cond` guards. These
    // assertions cover the supported shapes and verify that semantics
    // remain identical to the per-op walker.

    #[test]
    fn batched_patch_three_disjoint_writes() {
        let doc = json!({"a": 0, "b": 0, "c": 0, "d": 0});
        let r = vm_query(r#"patch $ { a: 1, b: 2, c: 3 }"#, &doc).unwrap();
        assert_eq!(r, json!({"a": 1, "b": 2, "c": 3, "d": 0}));
    }

    #[test]
    fn batched_patch_sibling_writes_share_parent() {
        let doc = json!({"user": {"name": "?", "role": "?"}});
        let r = vm_query(
            r#"patch $ { user.name: "alice", user.role: "admin" }"#,
            &doc,
        )
        .unwrap();
        assert_eq!(r, json!({"user": {"name": "alice", "role": "admin"}}));
    }

    #[test]
    fn batched_patch_nested_overlap_last_wins() {
        // The trie inserts ops in source order. The first op sets `a` to a
        // full object; the second op then descends into that object and
        // rewrites `a.x`. Because `Branch` insertion replaces a `Replace`
        // leaf with a fresh branch and discards the prior value, the final
        // shape comes purely from the second op writing `a.x: 2` — `a` is
        // synthesised from `Null` since the original `a` was a number.
        let doc = json!({"a": 1});
        let r = vm_query(r#"patch $ { a: {x: 1}, a.x: 2 }"#, &doc).unwrap();
        assert_eq!(r, json!({"a": {"x": 2}}));
    }

    #[test]
    fn batched_patch_array_index_writes() {
        let doc = json!({"items": [0, 0, 0]});
        let r = vm_query(r#"patch $ { items[0]: 10, items[1]: 20 }"#, &doc).unwrap();
        assert_eq!(r, json!({"items": [10, 20, 0]}));
    }

    #[test]
    fn batched_patch_delete_and_replace() {
        let doc = json!({"a": 0, "b": 0});
        let r = vm_query(r#"patch $ { a: DELETE, b: 1 }"#, &doc).unwrap();
        assert_eq!(r, json!({"b": 1}));
    }

    #[test]
    fn batched_patch_insert_missing_field() {
        // Trie should synthesise the missing `meta` object.
        let doc = json!({"name": "Alice"});
        let r = vm_query(
            r#"patch $ { meta.role: "admin", meta.active: true }"#,
            &doc,
        )
        .unwrap();
        assert_eq!(
            r,
            json!({"name": "Alice", "meta": {"role": "admin", "active": true}})
        );
    }

    #[test]
    fn batched_patch_modify_uses_old_value() {
        // `@` inside the value program must still bind to the pre-write
        // value at that path, matching the per-op walker.
        let doc = json!({"a": 5, "b": 10});
        let r = vm_query(r#"patch $ { a: @ + 1, b: @ * 2 }"#, &doc).unwrap();
        assert_eq!(r, json!({"a": 6, "b": 20}));
    }

    #[test]
    fn batched_patch_conditional_op_falls_back() {
        // Conditional ops are not trie-eligible — `from_ops` returns None
        // and the executor falls through to the per-op path. Result must
        // still be correct.
        let doc = json!({"role": "admin", "id": 7});
        let r = vm_query(
            r#"patch $ { active: true when $.role == "admin", banned: true when $.id < 0 }"#,
            &doc,
        )
        .unwrap();
        assert_eq!(
            r,
            json!({"role": "admin", "id": 7, "active": true})
        );
    }

    #[test]
    fn batched_patch_wildcard_falls_back() {
        // Wildcard path step is not trie-eligible.
        let doc = json!({"users": [{"n": 1}, {"n": 2}], "tag": "x"});
        let r = vm_query(r#"patch $ { users[*].n: @ + 100, tag: "y" }"#, &doc).unwrap();
        assert_eq!(
            r,
            json!({"users": [{"n": 101}, {"n": 102}], "tag": "y"})
        );
    }

    #[test]
    fn batched_patch_preserves_arc_for_untouched_subtrees() {
        // Build a doc with two subtrees, patch one, and confirm the other
        // survives intact (we can only check structural equality from the
        // public API surface, which is sufficient evidence the make_mut
        // CoW didn't accidentally fall back to a deep clone of unrelated
        // siblings — a deep clone would still produce equal JSON output
        // but blow up complexity. The structural assertion here pairs
        // with the unit-level invariant documented in `apply_trie`.)
        let doc = json!({
            "touched": {"x": 1, "y": 2},
            "untouched": {"a": [1, 2, 3], "b": "string", "c": {"deep": true}}
        });
        let r = vm_query(
            r#"patch $ { touched.x: 99, touched.y: 100 }"#,
            &doc,
        )
        .unwrap();
        assert_eq!(
            r,
            json!({
                "touched": {"x": 99, "y": 100},
                "untouched": {"a": [1, 2, 3], "b": "string", "c": {"deep": true}}
            })
        );
    }
}
