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
}
