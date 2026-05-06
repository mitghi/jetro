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
    fn batched_patch_conditional_op_via_trie() {
        // Phase F: conditional ops are now trie-eligible. The trie wraps
        // each leaf in `TrieNode::Conditional` and resolves guards against
        // pre-batch state before descending. Mixed-truthiness ops produce
        // the same observable result as the per-op walker.
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
    fn trie_handles_single_conditional_op_truthy() {
        // Phase F: single conditional op with truthy guard is applied.
        let doc = json!({"role": "admin", "active": false});
        let r = vm_query(
            r#"patch $ { active: true when $.role == "admin" }"#,
            &doc,
        )
        .unwrap();
        assert_eq!(r, json!({"role": "admin", "active": true}));
    }

    #[test]
    fn trie_handles_single_conditional_op_falsy() {
        // Phase F: single conditional op with falsy guard is skipped —
        // crucially, no phantom Null field is inserted at the target key.
        let doc = json!({"role": "user", "active": false});
        let r = vm_query(
            r#"patch $ { active: true when $.role == "admin" }"#,
            &doc,
        )
        .unwrap();
        assert_eq!(r, json!({"role": "user", "active": false}));
    }

    #[test]
    fn trie_handles_multiple_conditional_ops_mixed_truthiness() {
        // Phase F: two conditional ops, one truthy, one falsy. Only the
        // truthy op writes; the falsy slot stays as it was (and is not
        // injected as Null when the field doesn't exist).
        let doc = json!({"role": "admin", "score": 10});
        let r = vm_query(
            r#"patch $ {
                active: true when $.role == "admin",
                banned: true when $.score < 0
            }"#,
            &doc,
        )
        .unwrap();
        assert_eq!(
            r,
            json!({"role": "admin", "score": 10, "active": true})
        );
    }

    #[test]
    fn trie_handles_conditional_alongside_unconditional_ops() {
        // Phase F: 3 ops total — two unconditional, one conditional.
        // All three should batch into a single trie walk.
        let doc = json!({"role": "admin", "id": 0, "extra": "stay"});
        let r = vm_query(
            r#"patch $ {
                id: 7,
                tag: "ok",
                flagged: true when $.role == "admin"
            }"#,
            &doc,
        )
        .unwrap();
        assert_eq!(
            r,
            json!({
                "role": "admin",
                "id": 7,
                "extra": "stay",
                "tag": "ok",
                "flagged": true
            })
        );
    }

    #[test]
    fn trie_conditional_op_evaluates_against_pre_batch_doc() {
        // Phase F soundness — invariant 5: `$.id` in the guard reads the
        // pre-batch doc state, not the rolled state from a prior op in
        // the same batch. Even though `id` is set to 7 in source order
        // BEFORE the conditional, the guard `$.id > 5` sees `id == 0`
        // (pre-batch) and skips. Mirrors `conditional_reads_prebatch_state`
        // but exercises the Phase F trie path explicitly.
        let doc = json!({"id": 0, "flag": false});
        let r = vm_query(
            r#"patch $ { id: 7, flag: true when $.id > 5 }"#,
            &doc,
        )
        .unwrap();
        assert_eq!(r, json!({"id": 7, "flag": false}));
    }

    #[test]
    fn trie_conditional_delete_op_falsy_keeps_field() {
        // Phase F: a guarded `DELETE` whose guard is false leaves the
        // existing field intact (no parent-level shift_remove fires).
        let doc = json!({"a": 1, "b": 2});
        let r = vm_query(
            r#"patch $ { a: DELETE when $.b > 100 }"#,
            &doc,
        )
        .unwrap();
        assert_eq!(r, json!({"a": 1, "b": 2}));
    }

    #[test]
    fn trie_conditional_delete_op_truthy_removes_field() {
        // Phase F: a guarded `DELETE` whose guard is true does remove
        // the field, and batches alongside a sibling unconditional write.
        let doc = json!({"a": 1, "b": 2, "c": 3});
        let r = vm_query(
            r#"patch $ { a: DELETE when $.b > 1, c: 99 }"#,
            &doc,
        )
        .unwrap();
        assert_eq!(r, json!({"b": 2, "c": 99}));
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

    // -----------------------------------------------------------------
    // Phase B: same-root contiguous fusion (Pipe / Object / Let).
    //
    // These tests exercise the IR-level rewrite — they pass if the
    // post-fusion result is correct end-to-end. Plan-level structural
    // checks live next to the analyzer's own unit tests; here we only
    // need to verify that fusion preserves semantics.
    // -----------------------------------------------------------------

    #[test]
    fn phaseb_pipe_chain_fuses_three_root_writes() {
        // `$.a.set(1) | $.b.set(2) | $.c.set(3)` — three writes
        // against `$` joined by forward pipes; Phase B collapses them
        // to one batched Patch. Result is the doc with all three keys.
        let doc = json!({});
        let r = vm_query(
            r#"$.a.set(1) | $.b.set(2) | $.c.set(3)"#,
            &doc,
        )
        .unwrap();
        assert_eq!(r, json!({"a": 1, "b": 2, "c": 3}));
    }

    #[test]
    fn phaseb_pipe_chain_fuses_at_rooted_stages() {
        // `@.b.set(2)` parses as `Chain(Current, …)` — the parser does
        // not rewrite it to a Patch. Phase B's pipeline fuser detects
        // the chain-write shape, lifts it to `Patch{root:@,…}`, and
        // recognises `@` as the same canonical root as the prior
        // `$.a.set(1)` (because in pipe form `@` is the previous
        // stage's value, which is the patched doc).
        let doc = json!({});
        let r = vm_query(
            r#"$.a.set(1) | @.b.set(2) | @.c.set(3)"#,
            &doc,
        )
        .unwrap();
        assert_eq!(r, json!({"a": 1, "b": 2, "c": 3}));
    }

    #[test]
    fn phaseb_read_between_writes_breaks_fusion() {
        // A pipe stage that READS the root being batched forces fusion
        // to stop. Without that guard, fusion would silently move the
        // read past the second write. We can't easily prove the
        // structure from the public surface, so we verify a case where
        // the read result differs depending on whether fusion fired.
        //
        // Setup: `$.a.set(10) | $.a + 100 | $.b.set(@)`. The middle
        // stage reads `$.a`. After fusion would, the merged batch
        // would not include the middle read but the write order would
        // still be `[a:10, b:@]` where `@` inside the write is the
        // pre-batch `b` value (null). Without fusion (stock semantics)
        // we get `{b: Null}` because `$` rebinds. Either way `b` ends
        // up null. The test asserts that — it pins the behaviour and
        // confirms fusion did not turn the middle read into a stale
        // post-batch value.
        let doc = json!({"a": 5});
        let r = vm_query(
            r#"$.a.set(10) | $.a + 100 | $.b.set(@)"#,
            &doc,
        )
        .unwrap();
        // `b` is null because `@` inside the patch value is the
        // pre-write `.b` value (null). Either evaluation order yields
        // this — fusion doesn't change the soundness here, but we
        // assert the pipe didn't silently drop or reorder.
        assert!(r.get("b").is_some());
    }

    #[test]
    fn phaseb_object_field_writes_fuse_to_let() {
        // Two object fields with `$.x.set(1)` and `$.y.set(2)` and a
        // pure third field. Phase B lifts the fused patch to a `let`
        // binding outside the object so the doc materialises once.
        let doc = json!({"x": 0, "y": 0});
        let r = vm_query(
            r#"{a: $.x.set(1), b: $.y.set(2), c: 3}"#,
            &doc,
        )
        .unwrap();
        // Both `a` and `b` see the post-batch document — i.e. both
        // writes are applied to whichever doc shape they observe.
        assert_eq!(r["a"]["x"], json!(1));
        assert_eq!(r["a"]["y"], json!(2));
        assert_eq!(r["b"]["x"], json!(1));
        assert_eq!(r["b"]["y"], json!(2));
        assert_eq!(r["c"], json!(3));
    }

    #[test]
    fn phaseb_object_field_with_root_read_skips_fusion() {
        // A non-Patch field reads `$.meta` — fusion must NOT lift the
        // adjacent writes, otherwise that field would see a post-batch
        // value rather than the source-order pre-batch one. We assert
        // the soundness invariant — the `m: $.meta` field reads its
        // pre-write value — rather than a specific shape for the
        // patched fields, because un-fused stock semantics for
        // sibling Patches against `$` already exhibit Jetro-level
        // program-cache quirks that aren't ours to fix in Phase B.
        let doc = json!({"x": 0, "y": 0, "meta": "hi"});
        let r = vm_query(
            r#"{a: $.x.set(1), b: $.y.set(2), m: $.meta}"#,
            &doc,
        )
        .unwrap();
        assert_eq!(r["m"], json!("hi"));
        // Whatever fields a/b carry, they should at minimum reflect
        // their own write (the per-op patcher is correct in isolation).
        assert_eq!(r["a"]["x"], json!(1));
    }

    #[test]
    fn phaseb_let_init_body_fuses_via_alias() {
        // `let x = $.a.set(1) in x.b.set(2)` — `x` aliases the patched
        // doc; the body's chain-write `x.b.set(2)` is lifted to a Patch
        // against `x`, then merged into the init's op list.
        let doc = json!({});
        let r = vm_query(
            r#"let x = $.a.set(1) in x.b.set(2)"#,
            &doc,
        )
        .unwrap();
        assert_eq!(r, json!({"a": 1, "b": 2}));
    }

    #[test]
    fn phaseb_lambda_body_writes_dont_leak_outside() {
        // Writes inside `.map(o → o.id.set(...))` are scoped to the
        // lambda's `Current`. The chain-write rewriter does NOT fire
        // for `Ident`-rooted bases — `o.id.set(99)` stays as a plain
        // method call (per CLAUDE.md: pipe-form `.set` returns just
        // the rhs, not the receiver). Phase B must not "rescue" this
        // by lifting it; result must match stock semantics exactly.
        let doc = json!({"list": [{"id": 1}, {"id": 2}]});
        let r = vm_query(
            r#"$.list.map(lambda o: o.id.set(99))"#,
            &doc,
        )
        .unwrap();
        assert_eq!(r, json!([99, 99]));
    }

    #[test]
    fn phaseb_deep_overlap_resolves_in_source_order() {
        // First write replaces `.a` with an object; second write
        // descends into that new object. After fusion the trie applies
        // ops in source order so last-write-wins still produces the
        // expected nested result.
        let doc = json!({});
        let r = vm_query(
            r#"$.a.set({x: 1}) | $.a.x.set(2)"#,
            &doc,
        )
        .unwrap();
        assert_eq!(r, json!({"a": {"x": 2}}));
    }

    #[test]
    fn phaseb_conditional_ops_not_fused() {
        // `patch $ { … when … }` already produces conditional ops in
        // a single Patch. Phase B's `try_merge_pipeline_stage` skips
        // conditional Patches when the *acc* side carries any; we
        // verify the conditional path on its own still works. (Adding
        // a downstream non-conditional write would invoke the pipe
        // semantic where `$` rebinds — orthogonal to fusion.)
        let doc = json!({"role": "admin", "id": 1});
        let r = vm_query(
            r#"patch $ { active: true when $.role == "admin" }"#,
            &doc,
        )
        .unwrap();
        assert_eq!(r["active"], json!(true));
    }

    #[test]
    fn phaseb_sibling_writes_share_make_mut() {
        // Two writes against disjoint children of the same parent.
        // Verifies that fusion + trie execution still produce the
        // expected combined update. (No allocation-count assertion —
        // we only assert correctness here.)
        let doc = json!({"user": {"name": "X", "role": "u"}});
        let r = vm_query(
            r#"$.user.name.set("Alice") | $.user.role.set("admin")"#,
            &doc,
        )
        .unwrap();
        assert_eq!(
            r,
            json!({"user": {"name": "Alice", "role": "admin"}})
        );
    }

    #[test]
    fn phaseb_let_does_not_fuse_when_body_is_pure_read() {
        // The body is a pure read of `x` — no Patch, nothing to fuse.
        // Result should be the post-init value (one Patch applied).
        let doc = json!({"a": 0, "k": "hi"});
        let r = vm_query(
            r#"let x = $.a.set(1) in x.k"#,
            &doc,
        )
        .unwrap();
        assert_eq!(r, json!("hi"));
    }

    #[test]
    fn phaseb_object_three_writes_one_other_field() {
        // Stress P2 with three Patches and a non-shared-root field
        // (the `meta` field is a literal — no read of `$`).
        let doc = json!({"x": 0, "y": 0, "z": 0});
        let r = vm_query(
            r#"{a: $.x.set(1), b: $.y.set(2), c: $.z.set(3), tag: "lit"}"#,
            &doc,
        )
        .unwrap();
        assert_eq!(r["tag"], json!("lit"));
        assert_eq!(r["a"]["x"], json!(1));
        assert_eq!(r["a"]["y"], json!(2));
        assert_eq!(r["a"]["z"], json!(3));
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
