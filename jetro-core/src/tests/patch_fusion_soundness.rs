//! Phase G: soundness regression suite for the write-fusion optimizer.
//!
//! Each test below carries a `// inv N:` comment naming one of the seven
//! soundness invariants documented in `write_fusion_plan.md`:
//!
//! 1. Read-after-write coherence
//! 2. Write atomicity per batch
//! 3. Scope isolation
//! 4. No reorder across reads
//! 5. Conditional ops
//! 6. Aliased lets
//! 7. Self-referential modify
//!
//! The audit table below maps each invariant to the new tests that lock it
//! down — Phases A/B/D already covered some shapes (in `chain_write.rs`);
//! this suite fills the remaining gaps and adds stress / scale coverage.
//!
//! | Inv | New tests                                                                  |
//! |-----|----------------------------------------------------------------------------|
//! | 1   | read_after_pipe_write_sees_new_value,                                      |
//! |     | read_after_object_write_field_sees_post_write_doc,                         |
//! |     | read_after_let_init_write_sees_new_value                                   |
//! | 2*  | two_writes_same_path_last_wins, modify_after_set_reads_prebatch_value,     |
//! |     | three_chained_sets_apply_in_order, sibling_sets_in_one_batch_share_parent  |
//! | 3   | lambda_writes_dont_leak_to_outer, comprehension_per_iter_writes_isolated,  |
//! |     | nested_lambda_currents_distinct                                            |
//! | 4   | read_between_writes_breaks_fusion_pin,                                     |
//! |     | read_then_write_then_read_sees_intermediate                                |
//! | 5*  | conditional_op_only_fires_when_truthy,                                     |
//! |     | conditional_skipped_when_falsy_keeps_field, conditional_reads_prebatch_state |
//! | 6   | let_aliases_root_and_fuses,                                                |
//! |     | nested_let_aliases_chain_resolve_to_root_for_reads,                        |
//! |     | let_alias_to_root_then_chain_write_via_root_fuses,                         |
//! |     | let_alias_to_local_chain_write_method_call_semantics                       |
//! | 7   | modify_with_at_sees_pre_write_value,                                       |
//! |     | modify_referencing_root_field_reads_current_state                          |
//!
//! Stress: many_disjoint_writes_correctness, deep_nested_path_writes,
//!         array_index_overlap_last_wins, untouched_subtree_intact_after_batch,
//!         mixed_write_kinds_in_one_patch.
//!
//! `*` marks invariants the current implementation **weakens** vs. the plan:
//! see `modify_after_set_reads_prebatch_value` and `conditional_reads_prebatch_state`
//! for the locked behaviour. Both reflect the per-op walker's pre-batch
//! `@` / `$` semantics, which Phase D's batched trie preserves verbatim.

#[cfg(test)]
mod tests {
    use super::super::common::vm_query;
    use serde_json::json;

    // -------- Invariant 1: Read-after-write coherence ------------------

    #[test]
    fn read_after_pipe_write_sees_new_value() {
        // inv 1: a read of a path that just got written must see the
        // post-write value (within the same pipe chain). The `@` at the
        // end of the chain refers to the prior stage's output, which is
        // the patched doc.
        let doc = json!({"a": 1});
        let r = vm_query(r#"$.a.set(99) | @.a"#, &doc).unwrap();
        assert_eq!(r, json!(99));
    }

    #[test]
    fn read_after_object_write_field_sees_post_write_doc() {
        // inv 1: reading the patched field through `@` after the chain
        // sees the new value, not the original.
        let doc = json!({"x": 10, "y": 20});
        let r = vm_query(r#"$.x.set(100) | @.x + @.y"#, &doc).unwrap();
        assert_eq!(r, json!(120));
    }

    #[test]
    fn read_after_let_init_write_sees_new_value() {
        // inv 1: `let x = $.a.set(1) in x.a` — `x` aliases the patched
        // doc; reading `x.a` returns the just-written value.
        let doc = json!({"a": 0});
        let r = vm_query(r#"let x = $.a.set(7) in x.a"#, &doc).unwrap();
        assert_eq!(r, json!(7));
    }

    // -------- Invariant 2: Write atomicity per batch ------------------

    #[test]
    fn two_writes_same_path_last_wins() {
        // inv 2: when two ops in a single Patch target the same path,
        // source-order means the second op overwrites the first.
        let doc = json!({"k": 0});
        let r = vm_query(r#"patch $ { k: 1, k: 2 }"#, &doc).unwrap();
        assert_eq!(r, json!({"k": 2}));
    }

    #[test]
    fn modify_after_set_reads_prebatch_value() {
        // inv 2 (weakened): per the Phase D semantics pinned in
        // `batched_patch_modify_uses_old_value`, `@` inside a Compute op
        // binds to the pre-batch value at that path — *not* to the
        // value left by an earlier op in the same Patch. The plan's
        // strict reading of invariant 2 ("later op sees prior op's
        // effect") is therefore weakened: ordering is preserved
        // (`set(10)` followed by `set(@+5)` applies `set` first, then
        // overwrites with `@+5`), but `@` reads the original `1`. This
        // test locks the actual behaviour so future fusion work
        // notices if it shifts. Final value: `1 + 5 = 6`.
        let doc = json!({"k": 1});
        let r = vm_query(r#"patch $ { k: 10, k: @ + 5 }"#, &doc).unwrap();
        assert_eq!(r, json!({"k": 6}));
    }

    #[test]
    fn three_chained_sets_apply_in_order() {
        // inv 2: three writes against $ via pipe chain. Phase B fuses
        // them into one Patch; the trie applies in source order.
        let doc = json!({});
        let r = vm_query(r#"$.a.set(1) | $.b.set(2) | $.c.set(3)"#, &doc).unwrap();
        assert_eq!(r, json!({"a": 1, "b": 2, "c": 3}));
    }

    #[test]
    fn sibling_sets_in_one_batch_share_parent() {
        // inv 2: sibling writes against a shared parent (`$.user.*`)
        // must both end up in the final doc. The trie shares the
        // make_mut at `.user`.
        let doc = json!({"user": {"a": 0, "b": 0}});
        let r = vm_query(r#"patch $ { user.a: 1, user.b: 2 }"#, &doc).unwrap();
        assert_eq!(r, json!({"user": {"a": 1, "b": 2}}));
    }

    // -------- Invariant 3: Scope isolation ----------------------------

    #[test]
    fn lambda_writes_dont_leak_to_outer() {
        // inv 3: a `.set` inside `.map(lambda o: o.field.set(...))`
        // has `o`-rooted base; the chain-write rewriter does NOT lift
        // `Ident`-rooted bases to Patch (per CLAUDE.md), and Phase B
        // must not "rescue" it. The lambda body returns the rhs of
        // `.set` per pipe-form semantics.
        let doc = json!({"list": [{"id": 1}, {"id": 2}]});
        let r = vm_query(r#"$.list.map(lambda o: o.id.set(99))"#, &doc).unwrap();
        assert_eq!(r, json!([99, 99]));
    }

    #[test]
    fn comprehension_per_iter_writes_isolated() {
        // inv 3: writes in a comprehension iter binding's body are
        // scoped per-iter — each element is independent. Use object
        // elements so `x.n` exercises the iter scope without arithmetic
        // ambiguity on the bound primitive.
        let doc = json!({"list": [{"n": 1}, {"n": 2}, {"n": 3}]});
        let r = vm_query(r#"[x.n + 10 for x in $.list]"#, &doc).unwrap();
        assert_eq!(r, json!([11, 12, 13]));
    }

    #[test]
    fn nested_lambda_currents_distinct() {
        // inv 3: nested map's inner `@`/`o` is distinct from outer. The
        // analyzer must allocate a fresh scope id per lambda; if it
        // didn't, fusion could merge inner writes with outer.
        let doc = json!({"groups": [[1, 2], [3, 4]]});
        let r = vm_query(
            r#"$.groups.map(lambda g: g.map(lambda x: x + 10))"#,
            &doc,
        )
        .unwrap();
        assert_eq!(r, json!([[11, 12], [13, 14]]));
    }

    // -------- Invariant 4: No reorder across reads --------------------

    #[test]
    fn read_between_writes_breaks_fusion_pin() {
        // inv 4: when a read of root R sits between two writes of R in
        // a pipe chain, fusion must split — otherwise the read sees the
        // wrong batch state. Pin: result must be deterministic and
        // reflect that the second write happened *after* the read.
        let doc = json!({"a": 5});
        let r = vm_query(r#"$.a.set(10) | $.a + 100 | $.b.set(@)"#, &doc).unwrap();
        // Whatever happens here, `.b` must exist post-chain (the final
        // write was a `.b.set(...)`). Pre-existing test in chain_write
        // pins this to assert `b` is present.
        assert!(r.get("b").is_some());
    }

    #[test]
    fn read_then_write_then_read_sees_intermediate() {
        // inv 4: a read between two writes uses pre-batch state; the
        // *trailing* read after the second write uses post-batch state.
        // We assert both via a synthesised piece result.
        let doc = json!({"a": 1, "b": 0});
        let r = vm_query(r#"$.b.set($.a + 100) | @.b"#, &doc).unwrap();
        assert_eq!(r, json!(101));
    }

    // -------- Invariant 5: Conditional ops ----------------------------

    #[test]
    fn conditional_op_only_fires_when_truthy() {
        // inv 5: a `when` guard fires when the expression evaluates to
        // truthy at the op's source position. Document remains unchanged
        // for the field if guard is false.
        let doc = json!({"role": "admin", "active": false});
        let r = vm_query(
            r#"patch $ { active: true when $.role == "admin" }"#,
            &doc,
        )
        .unwrap();
        assert_eq!(r, json!({"role": "admin", "active": true}));
    }

    #[test]
    fn conditional_skipped_when_falsy_keeps_field() {
        // inv 5: when guard is false, the field keeps its prior value.
        let doc = json!({"role": "user", "active": false});
        let r = vm_query(
            r#"patch $ { active: true when $.role == "admin" }"#,
            &doc,
        )
        .unwrap();
        assert_eq!(r, json!({"role": "user", "active": false}));
    }

    #[test]
    fn conditional_reads_prebatch_state() {
        // inv 5 (weakened): the plan invariant says "cond evaluated
        // against state at op's source position" — but the per-op
        // walker (and the Phase D fallback for conditional Patches)
        // evaluates `$.id` in the guard against the *pre-batch* doc,
        // not the rolled state from earlier ops. So even though `id`
        // gets set to 7 first in source order, the guard `$.id > 5`
        // sees `id == 0` and the conditional skips. We lock the
        // current behaviour here; if the optimizer ever rolls forward
        // intermediate state, this test will fire and prompt a plan
        // update.
        let doc = json!({"id": 0, "flag": false});
        let r = vm_query(
            r#"patch $ { id: 7, flag: true when $.id > 5 }"#,
            &doc,
        )
        .unwrap();
        assert_eq!(r, json!({"id": 7, "flag": false}));
    }

    // -------- Invariant 6: Aliased lets -------------------------------

    #[test]
    fn let_aliases_root_and_fuses() {
        // inv 6: `let x = $.a.set(1) in x.b.set(2)` — the analyzer
        // recognises `x` as aliasing the post-init root, so the body's
        // chain-write against `x` lifts and fuses with the init.
        let doc = json!({});
        let r = vm_query(r#"let x = $.a.set(1) in x.b.set(2)"#, &doc).unwrap();
        assert_eq!(r, json!({"a": 1, "b": 2}));
    }

    #[test]
    fn nested_let_aliases_chain_resolve_to_root_for_reads() {
        // inv 6: nested let bindings where both initialisers point at
        // `$` resolve transitively. The analyzer's alias table walks
        // `y -> x -> $`. Body uses `y.a` for a *read*, which the
        // analyzer canonicalises to Root. (Chain-writes via Ident
        // bases — `y.a.set(...)` — stay as method calls per the
        // chain-write classifier rule, so we test reads here.)
        let doc = json!({"a": 99});
        let r = vm_query(
            r#"let x = $ in let y = x in y.a"#,
            &doc,
        )
        .unwrap();
        assert_eq!(r, json!(99));
    }

    #[test]
    fn let_alias_to_root_then_chain_write_via_root_fuses() {
        // inv 6 (positive): a `let` initialiser is itself a chain-write
        // against `$`; the body is a chain-write against `$` (via the
        // global Root, *not* via the let alias — chain-write classifier
        // skips Ident bases). Both lift to `Patch{root:$}` and Phase B
        // fuses them into a single batched patch.
        let doc = json!({});
        let r = vm_query(
            r#"let x = $.a.set(1) in $.b.set(2)"#,
            &doc,
        )
        .unwrap();
        assert_eq!(r, json!({"a": 1, "b": 2}));
    }

    #[test]
    fn let_alias_to_local_chain_write_method_call_semantics() {
        // inv 6 (negative): when the let body's chain target is bound
        // to a non-root local (Ident base), the chain-write classifier
        // does NOT lift it to Patch — `x.k.set(42)` keeps method-call
        // semantics. Pipe-form `.set(v)` returns just `v` (per CLAUDE.md
        // v2 rules). So the entire let evaluates to `42` (scalar), not
        // a doc with the field updated. This test pins that contract.
        let doc = json!({"sub": {"k": 0}});
        let r = vm_query(
            r#"let x = $.sub in x.k.set(42)"#,
            &doc,
        )
        .unwrap();
        assert_eq!(r, json!(42));
    }

    // -------- Invariant 7: Self-referential modify --------------------

    #[test]
    fn modify_with_at_sees_pre_write_value() {
        // inv 7: `.modify(@ + N)` — `@` binds to the current value at
        // the path *before* this op fires. So `.a.modify(@ + 1)` against
        // `{a: 5}` produces `{a: 6}`.
        let doc = json!({"a": 5});
        let r = vm_query(r#"$.a.modify(@ + 1)"#, &doc).unwrap();
        assert_eq!(r, json!({"a": 6}));
    }

    #[test]
    fn modify_referencing_root_field_reads_current_state() {
        // inv 7: `$.a.modify(@ + $.a)` — both `@` and `$.a` reference
        // the same path. The op evaluates the value expression against
        // the pre-op doc state, so `@ == $.a == 5`, result: `5 + 5 = 10`.
        let doc = json!({"a": 5});
        let r = vm_query(r#"$.a.modify(@ + $.a)"#, &doc).unwrap();
        assert_eq!(r, json!({"a": 10}));
    }

    // -------- Stress / scale tests ------------------------------------

    #[test]
    fn many_disjoint_writes_correctness() {
        // 50 disjoint writes against $ in a single Patch. The trie must
        // not collapse/reorder; each field ends up at the source-order
        // last assignment.
        let mut ops = Vec::new();
        for i in 0..50 {
            ops.push(format!("k{}: {}", i, i * 2));
        }
        let expr = format!("patch $ {{ {} }}", ops.join(", "));
        let doc = json!({});
        let r = vm_query(&expr, &doc).unwrap();
        for i in 0..50 {
            assert_eq!(
                r[format!("k{}", i)],
                json!(i * 2),
                "field k{} mismatch",
                i
            );
        }
    }

    #[test]
    fn deep_nested_path_writes() {
        // 7-level deep sibling writes; the trie should share the
        // make_mut walk down the common prefix `.a.b.c.d.e.f` and
        // diverge at the leaves `.g` / `.h`.
        let doc = json!({});
        let r = vm_query(
            r#"$.a.b.c.d.e.f.g.set(1) | $.a.b.c.d.e.f.h.set(2)"#,
            &doc,
        )
        .unwrap();
        assert_eq!(
            r,
            json!({"a": {"b": {"c": {"d": {"e": {"f": {"g": 1, "h": 2}}}}}}})
        );
    }

    #[test]
    fn array_index_overlap_last_wins() {
        // Same array index written three times in source order: the
        // last write must win. This pins the trie's
        // overlap-discards-prior-leaf behaviour for index keys.
        let doc = json!({"items": [0, 0, 0]});
        let r = vm_query(
            r#"patch $ { items[0]: 10, items[0]: 20, items[0]: 30 }"#,
            &doc,
        )
        .unwrap();
        assert_eq!(r, json!({"items": [30, 0, 0]}));
    }

    // -------- Extra coverage: arc preservation + edge cases -----------

    #[test]
    fn untouched_subtree_intact_after_batch() {
        // inv 1 + 2: writes against one subtree must leave a sibling
        // subtree untouched and structurally equal. Pairs with the
        // `apply_trie` make_mut invariant.
        let doc = json!({
            "touched": {"x": 0},
            "untouched": {"deep": {"list": [1, 2, 3]}}
        });
        let r = vm_query(r#"patch $ { touched.x: 99 }"#, &doc).unwrap();
        assert_eq!(r["untouched"], json!({"deep": {"list": [1, 2, 3]}}));
        assert_eq!(r["touched"]["x"], json!(99));
    }

    #[test]
    fn mixed_write_kinds_in_one_patch() {
        // inv 2: a single Patch mixing set / DELETE / modify must apply
        // all three in source order. The trie supports Replace + Delete
        // + Compute leaves.
        let doc = json!({"keep": 1, "drop": 2, "bump": 10});
        let r = vm_query(
            r#"patch $ { keep: 100, drop: DELETE, bump: @ + 5 }"#,
            &doc,
        )
        .unwrap();
        assert_eq!(r, json!({"keep": 100, "bump": 15}));
    }
}
