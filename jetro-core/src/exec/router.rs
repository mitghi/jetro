//! Top-level query execution router.
//!
//! Receives a `Jetro` document and a `QueryPlan` produced by `planner`,
//! then dispatches to either `physical_eval` (for structured IR nodes) or the
//! document/engine-owned VM (for the `SourceVm` fallback when planning is bypassed).
//! The only job here is routing — no evaluation logic lives in this module.

use serde_json::Value;

use crate::data::context::EvalError;
use crate::data::value::Val;
use crate::exec::interpreted as physical_eval;
use crate::ir::physical::{QueryPlan, QueryRoot};
use crate::plan::physical as planner;
use crate::{Jetro, VM};

/// Plans `expr` against `j`'s input mode and then executes the resulting plan, returning JSON.
///
/// This is the single call path used by `Jetro::collect` for one-shot queries.
pub(crate) fn collect_json(j: &Jetro, expr: &str) -> Result<Value, EvalError> {
    let plan = planner::plan_query_with_context(expr, planning_context(j));
    collect_plan_json(j, &plan)
}

/// Executes a pre-built `QueryPlan` against `j`, routing to `physical_eval` or the VM fallback.
///
/// Used by `JetroEngine::collect` when the plan was already retrieved from cache.
pub(crate) fn collect_plan_json(j: &Jetro, plan: &QueryPlan) -> Result<Value, EvalError> {
    j.with_vm(|vm| collect_plan_json_with_vm(j, plan, vm))
}

/// Executes a pre-built plan using a caller-supplied `VM` instance owned by `JetroEngine`,
/// avoiding the thread-local VM cell and enabling re-entrant use within the same thread.
pub(crate) fn collect_plan_json_with_vm(
    j: &Jetro,
    plan: &QueryPlan,
    vm: &mut VM,
) -> Result<Value, EvalError> {
    collect_plan_val_with_vm(j, plan, vm).map(Value::from)
}

/// Executes a pre-built plan and returns the internal `Val` result, avoiding a
/// `serde_json::Value` tree when the caller can serialize or consume `Val`.
pub(crate) fn collect_plan_val_with_vm(
    j: &Jetro,
    plan: &QueryPlan,
    vm: &mut VM,
) -> Result<Val, EvalError> {
    match plan.root() {
        QueryRoot::Node(root) => physical_eval::run_with_vm(j, plan, *root, vm),
        QueryRoot::SourceVm(source) => {
            let prog = vm.get_or_compile(source.as_ref())?;
            vm.execute_val_raw(&prog, j.root_val()?)
        }
    }
}

/// Derives the appropriate `PlanningContext` from the document handle's backing representation.
///
/// Documents backed by raw bytes use `Bytes` mode; in-memory `Val` documents use `Val` mode.
#[inline]
pub(crate) fn planning_context(j: &Jetro) -> planner::PlanningContext {
    if j.raw_bytes().is_some() {
        planner::PlanningContext::bytes()
    } else {
        planner::PlanningContext::val()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::json;

    use crate::data::value::Val;
    use crate::exec::pipeline::{BodyKernel, NumOp, ReducerOp, Sink, Stage};
    use crate::ir::physical::QueryRoot;
    use crate::ir::physical::{
        BackendPlan, BackendPreference, BackendSet, ExecutionFacts, NodeId, PhysicalArrayElem,
        PhysicalChainStep, PhysicalNode, PhysicalObjField, PhysicalPathStep, PipelinePlanSource,
        PlanNode, QueryPlan,
    };
    use crate::plan::physical as planner;
    use crate::{Jetro, JetroEngine};

    fn assert_no_vm_fallback(plan: &crate::ir::physical::QueryPlan, id: NodeId) {
        match plan.node(id) {
            PlanNode::Vm(_) => panic!("unexpected VM fallback in physical plan"),
            PlanNode::Literal(_)
            | PlanNode::Root
            | PlanNode::Current
            | PlanNode::Ident(_)
            | PlanNode::Local(_)
            | PlanNode::RootPath(_)
            | PlanNode::Structural { .. } => {}
            PlanNode::Pipeline { source, .. } => {
                if let PipelinePlanSource::Expr(source) = source {
                    assert_no_vm_fallback(plan, *source);
                }
            }
            PlanNode::Call { receiver, .. } => assert_no_vm_fallback(plan, *receiver),
            PlanNode::Chain { base, steps } => {
                assert_no_vm_fallback(plan, *base);
                for step in steps {
                    if let PhysicalChainStep::DynIndex(expr) = step {
                        assert_no_vm_fallback(plan, *expr);
                    }
                }
            }
            PlanNode::UnaryNeg(inner) | PlanNode::Not(inner) => assert_no_vm_fallback(plan, *inner),
            PlanNode::Binary { lhs, rhs, .. } => {
                assert_no_vm_fallback(plan, *lhs);
                assert_no_vm_fallback(plan, *rhs);
            }
            PlanNode::Kind { expr, .. } => assert_no_vm_fallback(plan, *expr),
            PlanNode::Coalesce { lhs, rhs } => {
                assert_no_vm_fallback(plan, *lhs);
                assert_no_vm_fallback(plan, *rhs);
            }
            PlanNode::IfElse { cond, then_, else_ } => {
                assert_no_vm_fallback(plan, *cond);
                assert_no_vm_fallback(plan, *then_);
                assert_no_vm_fallback(plan, *else_);
            }
            PlanNode::Try { body, default } => {
                assert_no_vm_fallback(plan, *body);
                assert_no_vm_fallback(plan, *default);
            }
            PlanNode::Object(fields) => {
                for field in fields {
                    match field {
                        PhysicalObjField::Kv { val, cond, .. } => {
                            assert_no_vm_fallback(plan, *val);
                            if let Some(cond) = cond {
                                assert_no_vm_fallback(plan, *cond);
                            }
                        }
                        PhysicalObjField::Short(_) => {}
                        PhysicalObjField::Dynamic { key, val } => {
                            assert_no_vm_fallback(plan, *key);
                            assert_no_vm_fallback(plan, *val);
                        }
                        PhysicalObjField::Spread(expr) | PhysicalObjField::SpreadDeep(expr) => {
                            assert_no_vm_fallback(plan, *expr);
                        }
                    }
                }
            }
            PlanNode::Array(elems) => {
                for elem in elems {
                    match elem {
                        PhysicalArrayElem::Expr(expr) | PhysicalArrayElem::Spread(expr) => {
                            assert_no_vm_fallback(plan, *expr);
                        }
                    }
                }
            }
            PlanNode::Let { init, body, .. } => {
                assert_no_vm_fallback(plan, *init);
                assert_no_vm_fallback(plan, *body);
            }
            PlanNode::UpdateBatch { .. } => panic!("unexpected update fallback in physical plan"),
        }
    }

    fn collect_test_val(j: &Jetro, expr: &str) -> Val {
        let plan = planner::plan_query(expr);
        match plan.root() {
            QueryRoot::Node(root) => crate::exec::interpreted::run(j, &plan, *root).unwrap(),
            QueryRoot::SourceVm(_) => panic!("unexpected source VM fallback"),
        }
    }

    #[test]
    fn engine_reuses_cached_physical_plan_across_documents() {
        let engine = JetroEngine::new();
        let j = Jetro::from(json!({
            "rows": [
                {"name": "low", "score": 1},
                {"name": "ada", "score": 901},
                {"name": "bob", "score": 902}
            ]
        }));
        let j2 = Jetro::from(json!({
            "rows": [
                {"name": "cat", "score": 3},
                {"name": "dan", "score": 903}
            ]
        }));

        let expr = "$.rows.filter(score > 900).first()";
        let first = engine.collect(&j, expr).unwrap();
        let second = engine.collect(&j2, expr).unwrap();

        assert_eq!(first, json!({"name": "ada", "score": 901}));
        assert_eq!(second, json!({"name": "dan", "score": 903}));
    }

    #[test]
    fn object_shape_executes_pipeline_children() {
        let j = Jetro::from(json!({
            "books": [
                {"id": 1, "price": 5},
                {"id": 2, "price": 15},
                {"id": 3, "price": 25}
            ],
            "test": "ok"
        }));

        let out = j
            .collect(r#"{"ids": $.books.filter(price > 10).map(id), "test": $.test}"#)
            .unwrap();

        assert_eq!(out, json!({"ids": [2, 3], "test": "ok"}));
    }

    #[test]
    fn object_shape_executes_multiple_pipeline_children() {
        let expr = r#"{"top": $.books.filter(score > 900).take(2).map(title), "first": $.books.filter(score > 900).first(), "meta": $.meta.version}"#;
        let plan = planner::plan_query(expr);
        let QueryRoot::Node(root) = plan.root() else {
            panic!("expected physical expression plan");
        };
        let PlanNode::Object(fields) = plan.node(*root) else {
            panic!("expected object root");
        };
        assert_eq!(fields.len(), 3);
        for idx in [0usize, 1] {
            let PhysicalObjField::Kv { val, .. } = &fields[idx] else {
                panic!("expected pipeline kv field");
            };
            assert!(matches!(plan.node(*val), PlanNode::Pipeline { .. }));
        }
        let PhysicalObjField::Kv { val, .. } = &fields[2] else {
            panic!("expected scalar kv field");
        };
        assert!(matches!(plan.node(*val), PlanNode::RootPath(_)));

        let j = Jetro::from(json!({
            "books": [
                {"title": "low", "score": 1},
                {"title": "a", "score": 901},
                {"title": "b", "score": 902},
                {"title": "c", "score": 903}
            ],
            "meta": {"version": 7}
        }));

        let out = j.collect(expr).unwrap();

        assert_eq!(
            out,
            json!({
                "top": ["a", "b"],
                "first": {"title": "a", "score": 901},
                "meta": 7
            })
        );
    }

    #[test]
    fn array_shape_executes_pipeline_children() {
        let expr = r#"[$.books.filter(score > 900).take(2).map(title), {"first": $.books.filter(score > 900).first()}, $.meta.version]"#;
        let plan = planner::plan_query(expr);
        let QueryRoot::Node(root) = plan.root() else {
            panic!("expected physical expression plan");
        };
        let PlanNode::Array(elems) = plan.node(*root) else {
            panic!("expected array root");
        };
        let PhysicalArrayElem::Expr(first) = &elems[0] else {
            panic!("expected array expr");
        };
        assert!(matches!(plan.node(*first), PlanNode::Pipeline { .. }));
        let PhysicalArrayElem::Expr(second) = &elems[1] else {
            panic!("expected array expr");
        };
        let PlanNode::Object(fields) = plan.node(*second) else {
            panic!("expected nested object");
        };
        let PhysicalObjField::Kv { val, .. } = &fields[0] else {
            panic!("expected nested kv field");
        };
        assert!(matches!(plan.node(*val), PlanNode::Pipeline { .. }));
        let PhysicalArrayElem::Expr(third) = &elems[2] else {
            panic!("expected array expr");
        };
        assert!(matches!(plan.node(*third), PlanNode::RootPath(_)));

        let j = Jetro::from(json!({
            "books": [
                {"title": "low", "score": 1},
                {"title": "a", "score": 901},
                {"title": "b", "score": 902},
                {"title": "c", "score": 903}
            ],
            "meta": {"version": 7}
        }));

        let out = j.collect(expr).unwrap();

        assert_eq!(
            out,
            json!([
                ["a", "b"],
                {"first": {"title": "a", "score": 901}},
                7
            ])
        );
    }

    #[test]
    fn nested_structural_shapes_execute_pipeline_children() {
        let expr = r#"{"groups": [{"top": $.books.filter(score > 900).take(2).map(title)}], "meta": [$.meta.version]}"#;
        let plan = planner::plan_query(expr);
        let QueryRoot::Node(root) = plan.root() else {
            panic!("expected physical expression plan");
        };
        assert_no_vm_fallback(&plan, *root);

        let j = Jetro::from(json!({
            "books": [
                {"title": "low", "score": 1},
                {"title": "a", "score": 901},
                {"title": "b", "score": 902},
                {"title": "c", "score": 903}
            ],
            "meta": {"version": 7}
        }));

        let out = j.collect(expr).unwrap();

        assert_eq!(
            out,
            json!({
                "groups": [{"top": ["a", "b"]}],
                "meta": [7]
            })
        );
    }

    #[test]
    fn object_shape_lowers_filter_map_sum_and_runs_map() {
        let expr = r#"{"total": $.data.filter(active).map(score).sum()}"#;
        let plan = planner::plan_query(expr);
        let QueryRoot::Node(root) = plan.root() else {
            panic!("expected physical expression plan");
        };
        let PlanNode::Object(fields) = plan.node(*root) else {
            panic!("expected object root");
        };
        let PhysicalObjField::Kv { val, .. } = &fields[0] else {
            panic!("expected object key/value field");
        };
        let PlanNode::Pipeline { source, body } = plan.node(*val) else {
            panic!("expected pipeline child");
        };

        match source {
            PipelinePlanSource::FieldChain { keys } => {
                let keys: Vec<&str> = keys.iter().map(|k| k.as_ref()).collect();
                assert_eq!(keys, vec!["data"]);
            }
            PipelinePlanSource::RootPath { .. } | PipelinePlanSource::Expr(_) => {
                panic!("expected $.data field-chain source")
            }
        }
        assert_eq!(body.stages.len(), 1);
        assert!(matches!(body.stages[0], Stage::Filter(_, _)));
        assert!(
            matches!(&body.sink, Sink::Reducer(spec) if spec.op == ReducerOp::Numeric(NumOp::Sum) && spec.projection.is_some())
        );

        let j = Jetro::from(json!({
            "data": [
                {"active": true, "score": 10},
                {"active": false, "score": 1000},
                {"active": true, "score": 15}
            ]
        }));

        let out = j.collect(expr).unwrap();

        assert_eq!(out, json!({"total": 25}));
    }

    #[test]
    fn top_level_lowers_filter_map_sum_and_runs_map() {
        let expr = "$.data.filter(active).map(score).sum()";
        let plan = planner::plan_query(expr);
        let QueryRoot::Node(root) = plan.root() else {
            panic!("expected physical expression plan");
        };
        let PlanNode::Pipeline { source, body } = plan.node(*root) else {
            panic!("expected pipeline root");
        };

        match source {
            PipelinePlanSource::FieldChain { keys } => {
                let keys: Vec<&str> = keys.iter().map(|k| k.as_ref()).collect();
                assert_eq!(keys, vec!["data"]);
            }
            PipelinePlanSource::RootPath { .. } | PipelinePlanSource::Expr(_) => {
                panic!("expected $.data field-chain source")
            }
        }
        assert_eq!(body.stages.len(), 1);
        assert!(matches!(body.stages[0], Stage::Filter(_, _)));
        assert!(
            matches!(&body.sink, Sink::Reducer(spec) if spec.op == ReducerOp::Numeric(NumOp::Sum) && spec.projection.is_some())
        );

        let j = Jetro::from(json!({
            "data": [
                {"active": true, "score": 10},
                {"active": false, "score": 1000},
                {"active": true, "score": 15}
            ]
        }));

        let out = j.collect(expr).unwrap();

        assert_eq!(out, json!(25));
    }

    #[test]
    fn object_shape_executes_scalar_root_path_without_collecting() {
        let j = Jetro::from(json!({
            "a": {"b": [{"c": "ok"}]}
        }));

        let out = j.collect(r#"{"value": $.a.b[0].c}"#).unwrap();

        assert_eq!(out, json!({"value": "ok"}));
    }
    #[test]
    fn object_shape_root_paths_read_from_tape_without_materializing_root_val() {
        let j =
            Jetro::from_bytes(br#"{"a":{"b":[{"c":"ok"},{"c":"next"}]},"n":7}"#.to_vec()).unwrap();

        let out = j.collect(r#"{"value": $.a.b[1].c, "n": $.n}"#).unwrap();

        assert_eq!(out, json!({"value": "next", "n": 7}));
        assert!(!j.root_val_is_materialized());
    }

    #[test]
    fn object_builtin_root_path_calls_read_from_tape_without_materializing_root_val() {
        let j = Jetro::from_bytes(
            br#"{"user":{"name":"Ada","email":"ada@example.test","meta":{"id":7}},"unused":{"large":[1,2,3]}}"#.to_vec(),
        )
        .unwrap();

        let out = j
            .collect(
                r#"{"has_email": $.user.has_key("email"), "missing_phone": $.user.missing("phone"), "missing_many": $.user.missing("email", "phone", "meta.name"), "id": $.user.get_path("meta.id"), "picked": $.user.pick("name"), "keys": $.user.keys()}"#,
            )
            .unwrap();

        assert_eq!(
            out,
            json!({"has_email": true, "missing_phone": true, "missing_many": ["phone", "meta.name"], "id": 7, "picked": {"name": "Ada"}, "keys": ["name", "email", "meta"]})
        );
        assert!(!j.root_val_is_materialized());
    }

    #[test]
    fn deep_match_obj_keys_routes_through_structural_index() {
        // `$..match { {role: "admin"} -> ..., ... }` is lowered to a
        // `StructuralPlan::DeepMatch` whose candidate enumeration draws
        // on the bitmap index, so the body of the document never has
        // to be materialised end-to-end.
        let j = Jetro::from_bytes(
            br#"{"u":[{"role":"admin","id":1},{"role":"user","id":2},{"name":"none"}]}"#.to_vec(),
        )
        .unwrap();

        let out = j
            .collect(
                r#"$..match {
                    {role: "admin", id: i} -> {tag: "a", n: i},
                    {role: "user",  id: i} -> {tag: "u", n: i},
                    _                      -> false
                }"#,
            )
            .unwrap();

        assert_eq!(
            out,
            serde_json::json!([
                {"tag": "a", "n": 1},
                {"tag": "u", "n": 2}
            ])
        );
        assert!(j.structural_index_is_built());
    }
    #[test]
    fn deep_match_first_obj_keys_via_structural_index() {
        // The early-stop variant returns the first truthy arm body
        // without walking the rest of the document.
        let j = Jetro::from_bytes(
            br#"{"events":[{"role":"viewer"},{"role":"admin","id":7},{"role":"editor"}]}"#.to_vec(),
        )
        .unwrap();

        let out = j
            .collect(
                r#"$..match! {
                    {role: "admin", id: i} -> i,
                    _                      -> false
                }"#,
            )
            .unwrap();

        assert_eq!(out, serde_json::json!(7));
        assert!(j.structural_index_is_built());
    }
    #[test]
    fn deep_shape_reads_from_structural_index_without_tape_or_root_val() {
        let j = Jetro::from_bytes(
            br#"{"users":[{"email":"a@x","role":"lead"},{"name":"missing"},{"team":{"email":"b@x","role":"dev"}}]}"#.to_vec(),
        )
        .unwrap();

        let out = j.collect(r#"$.deep_shape({email})"#).unwrap();

        assert_eq!(
            out,
            json!([
                {"email": "a@x", "role": "lead"},
                {"email": "b@x", "role": "dev"}
            ])
        );
        assert!(j.structural_index_is_built());
        assert!(!j.root_val_is_materialized());
        assert!(!j.tape_is_built());
    }
    #[test]
    fn deep_like_literal_pattern_reads_from_structural_index() {
        let j = Jetro::from_bytes(
            br#"{"users":[{"email":"a@x","role":"lead","active":true},{"email":"b@x","role":"lead","active":false},{"email":"c@x","role":"dev","active":true}]}"#.to_vec(),
        )
        .unwrap();

        let out = j
            .collect(r#"$.deep_like({role: "lead", active: true})"#)
            .unwrap();

        assert_eq!(
            out,
            json!([{"email": "a@x", "role": "lead", "active": true}])
        );
        assert!(j.structural_index_is_built());
        assert!(!j.root_val_is_materialized());
        assert!(!j.tape_is_built());
    }
    #[test]
    fn deep_find_field_literal_predicate_reads_from_structural_index() {
        let j = Jetro::from_bytes(
            br#"{"rows":[{"status":"open","id":1},{"status":"closed","id":2},{"nested":{"status":"open","id":3}}]}"#.to_vec(),
        )
        .unwrap();

        let out = j.collect(r#"$.deep_find(status == "open")"#).unwrap();

        assert_eq!(
            out,
            json!([{"status": "open", "id": 1}, {"status": "open", "id": 3}])
        );
        assert!(j.structural_index_is_built());
        assert!(!j.root_val_is_materialized());
        assert!(!j.tape_is_built());
    }
    #[test]
    fn deep_find_kind_and_field_predicate_reads_from_structural_index() {
        let j = Jetro::from_bytes(
            br#"{"rows":[{"status":"open","id":1},{"status":"closed","id":2},{"nested":{"status":"open","id":3}}]}"#.to_vec(),
        )
        .unwrap();

        let out = j
            .collect(r#"$.deep_find(@ kind object and @.status == "open")"#)
            .unwrap();

        assert_eq!(
            out,
            json!([{"status": "open", "id": 1}, {"status": "open", "id": 3}])
        );
        assert!(j.structural_index_is_built());
        assert!(!j.root_val_is_materialized());
        assert!(!j.tape_is_built());
    }
    #[test]
    fn anchored_deep_find_executes_pipeline_suffix() {
        let j = Jetro::from_bytes(
            br#"{"outside":{"status":"open","id":0},"org":{"rows":[{"status":"open","id":1},{"status":"open","id":2},{"status":"closed","id":3}]}}"#.to_vec(),
        )
        .unwrap();

        let out = j
            .collect(r#"$.org.rows.deep_find(status == "open").take(1)"#)
            .unwrap();

        assert_eq!(out, json!([{"status": "open", "id": 1}]));
        assert!(j.structural_index_is_built());
        assert!(!j.root_val_is_materialized());
        assert!(!j.tape_is_built());
    }
    #[test]
    fn anchored_deep_shape_reads_only_structural_subtree() {
        let j = Jetro::from_bytes(
            br#"{"outside":{"email":"outside@x"},"org":{"users":[{"email":"a@x"},{"team":{"email":"b@x"}}]}}"#.to_vec(),
        )
        .unwrap();

        let out = j.collect(r#"$.org.users.deep_shape({email})"#).unwrap();

        assert_eq!(out, json!([{"email": "a@x"}, {"email": "b@x"}]));
        assert!(j.structural_index_is_built());
        assert!(!j.root_val_is_materialized());
        assert!(!j.tape_is_built());
    }
    #[test]
    fn structural_prefix_executes_pipeline_suffix() {
        let j = Jetro::from_bytes(
            br#"{"org":{"users":[{"email":"a@x"},{"email":"b@x"},{"name":"missing"}]}}"#.to_vec(),
        )
        .unwrap();

        let out = j
            .collect(r#"$.org.users.deep_shape({email}).take(1)"#)
            .unwrap();

        assert_eq!(out, json!([{"email": "a@x"}]));
        assert!(j.structural_index_is_built());
        assert!(!j.root_val_is_materialized());
        assert!(!j.tape_is_built());
    }
    #[test]
    fn structural_prefix_executes_call_suffix() {
        let j = Jetro::from_bytes(
            br#"{"org":{"users":[{"email":"a@x"},{"email":"b@x"},{"name":"missing"}]}}"#.to_vec(),
        )
        .unwrap();

        let out = j
            .collect(r#"$.org.users.deep_shape({email}).count()"#)
            .unwrap();

        assert_eq!(out, json!(2));
        assert!(j.structural_index_is_built());
        assert!(!j.tape_is_built());
    }
    #[test]
    fn top_level_pipeline_source_reads_from_tape_without_materializing_root_val() {
        let j = Jetro::from_bytes(
            br#"{"books":[{"title":"low","score":1},{"title":"a","score":901},{"title":"b","score":902}],"unused":{"large":[1,2,3]}}"#.to_vec(),
        )
        .unwrap();

        let out = j
            .collect(r#"$.books.filter(score > 900).take(1).map(title)"#)
            .unwrap();

        assert_eq!(out, json!(["a"]));
        assert!(!j.root_val_is_materialized());
    }

    #[test]
    fn indexed_root_path_first_index_suffix_stays_tape_native() {
        let j = Jetro::from_bytes(
            br#"{"groups":[{"items":[{"id":"a","tags":[{"name":"sf"},{"name":"classic"}]},{"id":"b","tags":[{"name":"fantasy"}]}]}],"unused":{"large":[1,2,3]}}"#
                .to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"$.groups[0].items.first().tags[0].name"#)
            .unwrap();

        assert_eq!(out, json!("sf"));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }

    #[test]
    fn view_fstring_map_reads_from_tape_without_materializing_root_val() {
        let j = Jetro::from_bytes(
            br#"{"data":[{"id":1,"score":10,"user":{"name":"ada","addr":{"city":"NYC"}}},{"id":2,"score":20,"user":{"name":"bob","addr":{"city":"LA"}}}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();

        let out = j
            .collect(r##"$.data.map(f"#{id} {user.name} ({user.addr.city}) ${score}")"##)
            .unwrap();

        assert_eq!(out, json!(["#1 ada (NYC) $10", "#2 bob (LA) $20"]));
        assert!(!j.root_val_is_materialized());
    }
    #[test]
    fn view_object_map_reads_from_tape_without_materializing_root_val() {
        let j = Jetro::from_bytes(
            br#"{"data":[{"id":1,"score":10,"user":{"name":"ada","addr":{"city":"NYC"}}},{"id":2,"score":20,"user":{"name":"bob","addr":{"city":"LA"}}}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();

        let out = j
            .collect(r#"$.data.map({id, name: user.name, city: user.addr.city, score})"#)
            .unwrap();

        assert_eq!(
            out,
            json!([
                {"id": 1, "name": "ada", "city": "NYC", "score": 10},
                {"id": 2, "name": "bob", "city": "LA", "score": 20}
            ])
        );
        assert!(!j.root_val_is_materialized());
    }
    #[test]
    fn view_object_map_collect_uses_terminal_objvec_collector() {
        let j = Jetro::from_bytes(
            br#"{"data":[{"id":1,"score":10,"user":{"name":"ada"}},{"id":2,"score":20,"user":{"name":"bob"}}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = collect_test_val(&j, r#"$.data.map({id, name: user.name, score})"#);

        match out {
            Val::ObjVec(rows) => {
                assert_eq!(rows.nrows(), 2);
                assert_eq!(
                    rows.keys.iter().map(|key| key.as_ref()).collect::<Vec<_>>(),
                    vec!["id", "name", "score"]
                );
            }
            other => panic!("expected terminal object map to collect ObjVec, got {other:?}"),
        }
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn view_scalar_map_collects_without_materializing_subtrees() {
        let j = Jetro::from_bytes(
            br#"{"books":[{"title":"low","score":1},{"title":"a","score":901},{"title":"b","score":902}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j.collect(r#"$.books.map(score)"#).unwrap();

        assert_eq!(out, json!([1, 901, 902]));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn view_object_key_projection_last_uses_tape_native_helpers() {
        let j = Jetro::from_bytes(
            br#"{"books":[{"title":"low","score":1,"debug":true},{"title":"b","score":902,"debug":false}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let keys = j.collect(r#"$.books.map(@.keys()).last()"#).unwrap();
        let picked = j
            .collect(r#"$.books.map(@.pick("title", "score")).last()"#)
            .unwrap();
        let omitted = j.collect(r#"$.books.map(@.omit("debug")).last()"#).unwrap();

        assert_eq!(keys, json!(["title", "score", "debug"]));
        assert_eq!(picked, json!({"title": "b", "score": 902}));
        assert_eq!(omitted, json!({"title": "b", "score": 902}));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn view_object_values_entries_use_tape_native_helpers() {
        let j = Jetro::from_bytes(
            br#"{"books":[{"title":"a","score":1},{"title":"b","score":2}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let values = j.collect(r#"$.books.map(@.values()).last()"#).unwrap();
        let entries = j.collect(r#"$.books.map(@.entries()).last()"#).unwrap();

        assert_eq!(values, json!(["b", 2]));
        assert_eq!(entries, json!([["title", "b"], ["score", 2]]));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn view_object_key_predicates_use_tape_native_helpers() {
        let j = Jetro::from_bytes(
            br#"{"books":[{"title":"low","score":1},{"title":"b","isbn":"x"}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let has = j.collect(r#"$.books.map(@.has("isbn")).last()"#).unwrap();
        let has_key = j
            .collect(r#"$.books.map(@.has_key("isbn")).last()"#)
            .unwrap();
        let missing = j
            .collect(r#"$.books.map(@.missing("score")).last()"#)
            .unwrap();

        assert_eq!(has, json!(true));
        assert_eq!(has_key, json!(true));
        assert_eq!(missing, json!(true));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn view_object_predicate_and_projection_chain_stays_borrowed() {
        let j = Jetro::from_bytes(
            br#"{"books":[{"meta":{"title":"low"}},{"meta":{"isbn":"x","price":10,"debug":true}},{"meta":{"isbn":"y","price":20,"debug":false}}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"$.books.filter(@.meta.has_key("isbn")).map(@.meta.pick("isbn", "price")).last()"#)
            .unwrap();

        assert_eq!(out, json!({"isbn": "y", "price": 20}));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }

    #[test]
    fn view_nested_object_scalar_projection_chain_stays_borrowed() {
        let data = br#"{"books":[{"meta":{"author":{"name":"ada"},"isbn":"x"}},{"meta":{"author":{"name":"bob"},"isbn":"y"}},{"meta":{"title":"missing"}}],"unused":{"large":[1,2,3,4]}}"#.to_vec();
        let from_tape = Jetro::from_bytes(data.clone()).unwrap();
        let from_value = Jetro::from_bytes(data).unwrap();
        from_tape.reset_tape_materialized_subtrees();

        let query = r#"$.books.filter(@.meta.has_path("author.name")).map(@.meta.get_path("author.name").upper()).last()"#;
        let tape_out = from_tape.collect(query).unwrap();
        let value_out = from_value.collect(query).unwrap();

        assert_eq!(tape_out, value_out);
        assert_eq!(tape_out, json!("BOB"));
        assert!(!from_tape.root_val_is_materialized());
        assert_eq!(from_tape.tape_materialized_subtrees(), 0);
    }

    #[test]
    fn view_owned_object_projection_followup_chain_keeps_tape_input_borrowed() {
        let data = br#"{"books":[{"meta":{"author":{"name":"ada"},"isbn":"x","debug":1}},{"meta":{"author":{"name":"bob"},"isbn":"y","debug":2}},{"meta":{"author":{"name":"cat"},"debug":3}}],"unused":{"large":[1,2,3,4]}}"#.to_vec();
        let from_tape = Jetro::from_bytes(data.clone()).unwrap();
        let from_value = Jetro::from_bytes(data).unwrap();
        from_tape.reset_tape_materialized_subtrees();

        let query = r#"$.books.filter(@.meta.has_key("isbn")).map(@.meta.pick("isbn", "author").get_path("author.name")).take(2)"#;
        let tape_out = from_tape.collect(query).unwrap();
        let value_out = from_value.collect(query).unwrap();

        assert_eq!(tape_out, value_out);
        assert_eq!(tape_out, json!(["ada", "bob"]));
        assert!(!from_tape.root_val_is_materialized());
        assert_eq!(from_tape.tape_materialized_subtrees(), 0);
    }

    #[test]
    fn view_owned_object_projection_scalar_followup_stays_late() {
        let data = br#"{"books":[{"meta":{"isbn":"x","price":10,"debug":1}},{"meta":{"isbn":"y","price":20,"debug":2}},{"meta":{"title":"missing","debug":3}}],"unused":{"large":[1,2,3,4]}}"#.to_vec();
        let from_tape = Jetro::from_bytes(data.clone()).unwrap();
        let from_value = Jetro::from_bytes(data).unwrap();
        from_tape.reset_tape_materialized_subtrees();

        let query = r#"$.books.filter(@.meta.has_key("isbn")).map(@.meta.pick("isbn", "price").len()).last()"#;
        let tape_out = from_tape.collect(query).unwrap();
        let value_out = from_value.collect(query).unwrap();

        assert_eq!(tape_out, value_out);
        assert_eq!(tape_out, json!(2));
        assert!(!from_tape.root_val_is_materialized());
        assert_eq!(from_tape.tape_materialized_subtrees(), 0);
    }

    #[test]
    fn view_owned_object_helper_chains_keep_tape_input_borrowed() {
        let data = br#"{"books":[{"meta":{"author":{"name":"ada"},"isbn":"x","price":10,"debug":1}},{"meta":{"author":{"name":"bob"},"isbn":"y","price":20,"debug":2}},{"meta":{"author":{"name":"cat"},"debug":3}}],"unused":{"large":[1,2,3,4]}}"#.to_vec();
        let from_tape = Jetro::from_bytes(data.clone()).unwrap();
        let from_value = Jetro::from_bytes(data).unwrap();
        from_tape.reset_tape_materialized_subtrees();

        let query = r#"$.books.filter(@.meta.has_key("isbn")).map({
            last_key: @.meta.pick("isbn", "price").entries().last()[0],
            first_value: @.meta.omit("debug", "author").values().first()
        }).last()"#;
        let tape_out = from_tape.collect(query).unwrap();
        let value_out = from_value.collect(query).unwrap();

        assert_eq!(tape_out, value_out);
        assert_eq!(tape_out, json!({"last_key": "price", "first_value": "y"}));
        assert!(!from_tape.root_val_is_materialized());
        assert_eq!(from_tape.tape_materialized_subtrees(), 0);
    }

    #[test]
    fn view_flat_map_object_helper_outputs_materialize_only_demanded_rows() {
        let data = br#"{"books":[{"meta":{"isbn":"x","price":10,"debug":1}},{"meta":{"isbn":"y","price":20,"debug":2}},{"meta":{"isbn":"z","debug":3}}],"unused":{"large":[1,2,3,4]}}"#.to_vec();
        let from_tape = Jetro::from_bytes(data.clone()).unwrap();
        let from_value = Jetro::from_bytes(data).unwrap();
        from_tape.reset_tape_materialized_subtrees();

        let values_query =
            r#"$.books.flat_map(@.meta.entries()).filter(@[0] != "debug").map(@[1]).take(4)"#;
        let values = from_tape.collect(values_query).unwrap();
        let expected_values = from_value.collect(values_query).unwrap();
        assert_eq!(values, expected_values);
        assert_eq!(values, json!(["x", 10, "y", 20]));

        let sum_query =
            r#"$.books.flat_map(@.meta.entries()).filter(@[0] == "price").map(@[1]).sum()"#;
        let total = from_tape.collect(sum_query).unwrap();
        let expected_total = from_value.collect(sum_query).unwrap();
        assert_eq!(total, expected_total);
        assert_eq!(total, json!(30));

        assert!(!from_tape.root_val_is_materialized());
        assert_eq!(from_tape.tape_materialized_subtrees(), 0);
    }

    #[test]
    fn view_nested_projection_reducers_keep_tape_input_borrowed() {
        let data = br#"{"books":[{"meta":{"author":{"country":"uk"},"price":10}},{"meta":{"author":{"country":"us"},"price":20}},{"meta":{"author":{"country":"uk"},"price":30}}],"unused":{"large":[1,2,3,4]}}"#.to_vec();
        let from_tape = Jetro::from_bytes(data.clone()).unwrap();
        let from_value = Jetro::from_bytes(data).unwrap();
        from_tape.reset_tape_materialized_subtrees();

        let counts = from_tape
            .collect(r#"$.books.count_by(@.meta.get_path("author.country").upper())"#)
            .unwrap();
        let expected_counts = from_value
            .collect(r#"$.books.count_by(@.meta.get_path("author.country").upper())"#)
            .unwrap();
        assert_eq!(counts, expected_counts);
        assert_eq!(counts, json!({"UK": 2, "US": 1}));

        let total = from_tape
            .collect(r#"$.books.map(@.meta.get_path("price")).sum()"#)
            .unwrap();
        let expected_total = from_value
            .collect(r#"$.books.map(@.meta.get_path("price")).sum()"#)
            .unwrap();
        assert_eq!(total, expected_total);
        assert_eq!(total, json!(60));
        assert!(!from_tape.root_val_is_materialized());
        assert_eq!(from_tape.tape_materialized_subtrees(), 0);
    }

    #[test]
    fn view_has_preserves_array_and_string_membership_without_materialization() {
        let j = Jetro::from_bytes(
            br#"{"books":[{"tags":["sf"],"title":"Dune"},{"tags":["sf","hugo"],"title":"Foundation"}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let tag = j
            .collect(r#"$.books.map(@.tags.has("hugo")).last()"#)
            .unwrap();
        let title = j
            .collect(r#"$.books.map(@.title.has("dation")).last()"#)
            .unwrap();

        assert_eq!(tag, json!(true));
        assert_eq!(title, json!(true));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn view_missing_treats_null_as_missing_without_materialization() {
        let j = Jetro::from_bytes(
            br#"{"books":[{"isbn":"x","meta":{"rating":5}},{"isbn":null,"meta":{}}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let single = j
            .collect(r#"$.books.map(@.missing("isbn")).last()"#)
            .unwrap();
        let many = j
            .collect(r#"$.books.map(@.missing("isbn", "meta.rating")).last()"#)
            .unwrap();

        assert_eq!(single, json!(true));
        assert_eq!(many, json!(["isbn", "meta.rating"]));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn view_path_helpers_use_tape_native_navigation() {
        let j = Jetro::from_bytes(
            br#"{"books":[{"user":{"name":"ada"}},{"user":{"name":"bob"}}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let name = j
            .collect(r#"$.books.map(@.get_path("user.name")).last()"#)
            .unwrap();
        let found = j
            .collect(r#"$.books.map(@.has_path("user.name")).last()"#)
            .unwrap();
        let missing = j
            .collect(r#"$.books.map(@.has_path("user.missing")).last()"#)
            .unwrap();

        assert_eq!(name, json!("bob"));
        assert_eq!(found, json!(true));
        assert_eq!(missing, json!(false));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }

    #[test]
    fn view_del_path_projects_from_tape_without_materializing_receiver() {
        let j = Jetro::from_bytes(
            br#"{"books":[{"id":1,"user":{"name":"ada","tmp":true}},{"id":2,"user":{"name":"bob","tmp":false}}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"$.books.map(@.del_path("user.tmp")).last()"#)
            .unwrap();

        assert_eq!(out, json!({"id": 2, "user": {"name": "bob"}}));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }

    #[test]
    fn view_set_path_projects_from_tape_without_materializing_receiver() {
        let j = Jetro::from_bytes(
            br#"{"books":[{"id":1,"user":{"name":"ada"}},{"id":2,"user":{"name":"bob"}}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"$.books.map(@.set_path("user.reviewed", true)).last()"#)
            .unwrap();

        assert_eq!(
            out,
            json!({"id": 2, "user": {"name": "bob", "reviewed": true}})
        );
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }

    #[test]
    fn view_del_paths_projects_from_tape_without_materializing_receiver() {
        let j = Jetro::from_bytes(
            br#"{"books":[{"id":1,"user":{"name":"ada","tmp":true},"debug":1},{"id":2,"user":{"name":"bob","tmp":false},"debug":2}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"$.books.map(@.del_paths("user.tmp", "debug")).last()"#)
            .unwrap();

        assert_eq!(out, json!({"id": 2, "user": {"name": "bob"}}));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }

    #[test]
    fn view_delimited_serializers_write_from_tape_without_materializing_receiver() {
        let j = Jetro::from_bytes(
            br#"{"docs":[{"rows":[[0,"old"]],"objects":[{"id":0,"name":"old"}]},{"rows":[[1,"a,b"],[2,"plain"]],"objects":[{"id":1,"name":"ada"},{"id":2,"name":"bob"}]}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let csv = j.collect(r#"$.docs.map(@.rows.to_csv()).last()"#).unwrap();
        assert_eq!(j.tape_materialized_subtrees(), 0);
        let tsv = j
            .collect(r#"$.docs.map(@.objects.to_tsv(["id", "name"])).last()"#)
            .unwrap();

        assert_eq!(csv, json!("1,\"a,b\"\n2,plain"));
        assert_eq!(tsv, json!("id\tname\n1\tada\n2\tbob"));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }

    #[test]
    fn view_zip_shape_projects_from_tape_without_materializing_receiver() {
        let j = Jetro::from_bytes(
            br#"{"docs":[{"shape":{"id":[0],"name":["old"],"kind":"book"}},{"shape":{"id":[1,2],"name":["ada","bob"],"kind":"book"}}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"$.docs.map(@.shape.zip_shape()).last()"#)
            .unwrap();

        assert_eq!(
            out,
            json!([
                {"id": 1, "name": "ada", "kind": "book"},
                {"id": 2, "name": "bob", "kind": "book"}
            ])
        );
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }

    #[test]
    fn view_from_json_projects_from_tape_without_materializing_receiver() {
        let j = Jetro::from_bytes(
            br#"{"docs":[{"raw":"{\"id\":1,\"name\":\"old\"}"},{"raw":"{\"id\":2,\"name\":\"bob\"}"}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"$.docs.map(@.raw.from_json()).last()"#)
            .unwrap();

        assert_eq!(out, json!({"id": 2, "name": "bob"}));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }

    #[test]
    fn view_implode_projects_selected_tape_receiver() {
        let j = Jetro::from_bytes(
            br#"{"docs":[{"rows":[{"g":"old","x":1}]},{"rows":[{"g":"a","x":1},{"g":"a","x":2},{"g":"b","x":3}]}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"$.docs.map(@.rows.implode(x)).last()"#)
            .unwrap();

        assert_eq!(out, json!([{"g":"a","x":[1,2]},{"g":"b","x":[3]}]));
        assert!(!j.root_val_is_materialized());
        assert!(j.tape_materialized_subtrees() < 8);
    }

    #[test]
    fn view_group_shape_projects_selected_tape_receiver() {
        let j = Jetro::from_bytes(
            br#"{"docs":[{"rows":[{"old":true}]},{"rows":[{"a":1,"b":2},{"b":3,"a":4},{"a":5},7]}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"$.docs.map(@.rows.group_shape()).last()"#)
            .unwrap();

        assert_eq!(
            out,
            json!({"a,b":[{"a":1,"b":2},{"b":3,"a":4}],"a":[{"a":5}],"<scalar>":[7]})
        );
        assert!(!j.root_val_is_materialized());
    }

    #[test]
    fn view_pivot_projects_selected_tape_receiver() {
        let j = Jetro::from_bytes(
            br#"{"docs":[{"rows":[{"region":"old","product":"X","sales":1}]},{"rows":[{"region":"north","product":"A","sales":100},{"region":"north","product":"B","sales":120},{"region":"south","product":"A","sales":150}]}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"$.docs.map(@.rows.pivot("region", "product", "sales")).last()"#)
            .unwrap();

        assert_eq!(out, json!({"north":{"A":100,"B":120},"south":{"A":150}}));
        assert!(!j.root_val_is_materialized());
    }

    #[test]
    fn view_object_lambdas_project_selected_tape_receiver() {
        let j = Jetro::from_bytes(
            br#"{"profiles":[{"settings":{"_debug":true,"feature_a":1,"feature_b":null}},{"settings":{"_debug":false,"feature_a":2,"feature_b":3}}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(
                r#"$.profiles.map(settings.filter_keys(k => not k.starts_with("_")).transform_values(v => v.to_string())).last()"#,
            )
            .unwrap();

        assert_eq!(out, json!({"feature_a": "2", "feature_b": "3"}));
        assert!(!j.root_val_is_materialized());
    }

    #[test]
    fn view_transform_keys_projects_selected_tape_receiver() {
        let j = Jetro::from_bytes(
            br#"{"profiles":[{"settings":{"old":1}},{"settings":{"feature_a":2,"feature_b":3}}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"$.profiles.map(settings.transform_keys(k => k.upper())).last()"#)
            .unwrap();

        assert_eq!(out, json!({"FEATURE_A": 2, "FEATURE_B": 3}));
        assert!(!j.root_val_is_materialized());
    }

    #[test]
    fn view_partition_streams_tape_receiver_until_flush() {
        let j = Jetro::from_bytes(
            br#"{"books":[{"title":"old","active":false},{"title":"dune","active":true},{"title":"foundation","active":true},{"title":"draft","active":false}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j.collect(r#"$.books.partition(active)"#).unwrap();

        assert_eq!(
            out,
            json!([
                [{"title":"dune","active":true},{"title":"foundation","active":true}],
                [{"title":"old","active":false},{"title":"draft","active":false}]
            ])
        );
        assert!(!j.root_val_is_materialized());
    }

    #[test]
    fn view_get_path_projects_selected_tape_receiver() {
        let j = Jetro::from_bytes(
            br#"{"books":[{"meta":{"isbn":"old","flags":{"reviewed":false}}},{"meta":{"isbn":"dune","flags":{"reviewed":true}}},{"meta":{"isbn":"foundation","flags":{"reviewed":true}}}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"$.books.map(@.get_path("meta.isbn")).last()"#)
            .unwrap();

        assert_eq!(out, json!("foundation"));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }

    #[test]
    fn view_has_path_filters_tape_receiver_without_materializing_root() {
        let j = Jetro::from_bytes(
            br#"{"books":[{"title":"old","meta":{}},{"title":"dune","meta":{"flags":{"reviewed":true}}},{"title":"draft","meta":{"flags":{}}}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"$.books.filter(@.has_path("meta.flags.reviewed")).map(title)"#)
            .unwrap();

        assert_eq!(out, json!(["dune"]));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }

    #[test]
    fn view_object_values_first_streams_object_items_from_tape() {
        let j = Jetro::from_bytes(
            br#"{"profile":{"name":"Ada","bio":{"large":[1,2,3,4]},"city":"London"},"unused":{"large":[5,6,7,8]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j.collect(r#"$.profile.values().first()"#).unwrap();

        assert_eq!(out, json!("Ada"));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }

    #[test]
    fn root_view_object_values_first_streams_object_items_from_tape() {
        let j = Jetro::from_bytes(
            br#"{"name":"Ada","bio":{"large":[1,2,3,4]},"city":"London"}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j.collect(r#"$.values().first()"#).unwrap();

        assert_eq!(out, json!("Ada"));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }

    #[test]
    fn view_object_entries_first_streams_only_selected_entry_from_tape() {
        let j = Jetro::from_bytes(
            br#"{"profile":{"name":"Ada","bio":{"large":[1,2,3,4]},"city":"London"},"unused":{"large":[5,6,7,8]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j.collect(r#"$.profile.entries().first()"#).unwrap();

        assert_eq!(out, json!(["name", "Ada"]));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }

    #[test]
    fn nested_view_object_values_first_streams_object_items_from_tape() {
        let j = Jetro::from_bytes(
            br#"{"profile":{"name":"Ada","bio":{"large":[1,2,3,4]},"city":"London"},"unused":{"large":[5,6,7,8]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"{first_name: $.profile.values().first()}"#)
            .unwrap();

        assert_eq!(out, json!({"first_name":"Ada"}));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }

    #[test]
    fn indexed_view_object_values_first_streams_object_items_from_tape() {
        let j = Jetro::from_bytes(
            br#"{"profiles":[{"name":"Ada","bio":{"large":[1,2,3,4]}},{"name":"Grace"}],"unused":{"large":[5,6,7,8]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j.collect(r#"$.profiles[0].values().first()"#).unwrap();

        assert_eq!(out, json!("Ada"));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }

    #[test]
    fn indexed_array_pipeline_streams_from_static_root_path() {
        let j = Jetro::from_bytes(
            br#"{"groups":[{"items":[{"id":"a","payload":{"large":[1,2,3]}},{"id":"b"}]},{"items":[{"id":"c"}]}],"unused":{"large":[5,6,7,8]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j.collect(r#"$.groups[0].items.map(id).first()"#).unwrap();

        assert_eq!(out, json!("a"));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }

    #[test]
    fn indexed_array_pipeline_suffix_streams_selected_row_from_static_root_path() {
        let j = Jetro::from_bytes(
            br#"{"groups":[{"items":[{"id":"a","payload":{"large":[1,2,3]}},{"id":"b"}]},{"items":[{"id":"c"}]}],"unused":{"large":[5,6,7,8]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j.collect(r#"$.groups[0].items.first().id"#).unwrap();

        assert_eq!(out, json!("a"));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }

    #[test]
    fn view_object_map_collects_scalar_cells_without_materializing_subtrees() {
        let j = Jetro::from_bytes(
            br#"{"books":[{"title":"low","score":1},{"title":"a","score":901},{"title":"b","score":902}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j.collect(r#"$.books.map({title, score})"#).unwrap();

        assert_eq!(
            out,
            json!([
                {"title": "low", "score": 1},
                {"title": "a", "score": 901},
                {"title": "b", "score": 902}
            ])
        );
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }

    #[test]
    fn nested_field_chain_pipeline_in_projection_stays_tape_streamed() {
        let j = Jetro::from_bytes(
            br#"{"books":[{"title":"a","items":[{"isbn":"old","price":5},{"isbn":"new","price":25}],"payload":{"large":[1,2,3]}},{"title":"b","items":[{"isbn":"x","price":30}],"payload":{"large":[4,5,6]}}],"unused":{"large":[7,8,9]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"$.books.map({title, last: items.filter(price > 20).map(isbn).last()})"#)
            .unwrap();

        assert_eq!(
            out,
            json!([
                {"title": "a", "last": "new"},
                {"title": "b", "last": "x"}
            ])
        );
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }

    #[test]
    fn guarded_object_projection_stays_tape_streamed() {
        let j = Jetro::from_bytes(
            br#"{"books":[{"title":"a","isbn":"x","active":true,"payload":{"large":[1,2,3]}},{"title":"b","isbn":"y","active":false,"payload":{"large":[4,5,6]}}],"unused":{"large":[7,8,9]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"$.books.map({title, isbn: isbn when active})"#)
            .unwrap();

        assert_eq!(
            out,
            json!([
                {"title": "a", "isbn": "x"},
                {"title": "b"}
            ])
        );
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }

    #[test]
    fn object_shape_pipeline_child_reads_from_tape_without_materializing_root_val() {
        let j = Jetro::from_bytes(
            br#"{"books":[{"title":"low","score":1},{"title":"a","score":901},{"title":"b","score":902}],"meta":{"version":3}}"#.to_vec(),
        )
        .unwrap();

        let out = j
            .collect(
                r#"{"top": $.books.filter(score > 900).take(1).map(title), "v": $.meta.version}"#,
            )
            .unwrap();

        assert_eq!(out, json!({"top": ["a"], "v": 3}));
        assert!(!j.root_val_is_materialized());
    }
    #[test]
    fn tape_pipeline_generic_first_stage_uses_row_bridge() {
        let j = Jetro::from_bytes(
            br#"{"people":[{"name":"al"},{"name":"ada"},{"name":"bob"},{"name":"carol"}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();

        let out = j
            .collect(r#"$.people.filter(name.len() == 3).take(1).map(name)"#)
            .unwrap();

        assert_eq!(out, json!(["ada"]));
        assert!(!j.root_val_is_materialized());
    }
    #[test]
    fn tape_view_native_take_materializes_only_output_subtree() {
        let j = Jetro::from_bytes(
            br#"{"people":[{"name":"al","score":1},{"name":"ada","score":901},{"name":"bob","score":902}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"$.people.filter(score > 900).take(1).map(name)"#)
            .unwrap();

        assert_eq!(out, json!(["ada"]));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn tape_view_current_row_collect_materializes_only_output_subtree() {
        let j = Jetro::from_bytes(
            br#"{"people":[{"name":"al","score":1},{"name":"ada","score":901},{"name":"bob","score":902}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"$.people.filter(score > 900).take(1)"#)
            .unwrap();

        assert_eq!(out, json!([{"name": "ada", "score": 901}]));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 1);
    }
    #[test]
    fn tape_view_remove_last_stays_borrowed() {
        let j = Jetro::from_bytes(
            br#"{"xs":[{"id":1},{"id":2},{"id":3},{"id":2}],"unused":{"large":[1,2,3,4]}}"#
                .to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j.collect(r#"$.xs.map(id).remove(2).last()"#).unwrap();

        assert_eq!(out, json!(3));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn tape_view_remove_last_ignores_removed_physical_tail() {
        let j = Jetro::from_bytes(
            br#"{"xs":[{"id":1},{"id":3},{"id":2}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j.collect(r#"$.xs.map(id).remove(2).last()"#).unwrap();

        assert_eq!(out, json!(3));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn tape_view_remove_take_stays_borrowed() {
        let j = Jetro::from_bytes(
            br#"{"xs":[{"id":1},{"id":2},{"id":3},{"id":4}],"unused":{"large":[1,2,3,4]}}"#
                .to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j.collect(r#"$.xs.map(id).remove(2).take(2)"#).unwrap();

        assert_eq!(out, json!([1, 3]));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn tape_view_compact_take_stays_borrowed() {
        let j = Jetro::from_bytes(
            br#"{"xs":[null,{"id":1},null,{"id":2},{"id":3}],"unused":{"large":[1,2,3,4]}}"#
                .to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j.collect(r#"$.xs.compact().take(2).map(id)"#).unwrap();

        assert_eq!(out, json!([1, 2]));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn tape_view_compact_last_stays_borrowed() {
        let j = Jetro::from_bytes(
            br#"{"xs":[null,{"id":1},null,{"id":2}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j.collect(r#"$.xs.compact().map(id).last()"#).unwrap();

        assert_eq!(out, json!(2));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn tape_view_compact_last_ignores_physical_null_tail() {
        let j = Jetro::from_bytes(
            br#"{"xs":[{"id":1},{"id":2},null],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j.collect(r#"$.xs.compact().map(id).last()"#).unwrap();

        assert_eq!(out, json!(2));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn tape_view_map_first_reads_head_without_materializing_result_row() {
        let j = Jetro::from_bytes(
            br#"{"people":[{"name":"al","score":1},{"name":"ada","score":901},{"name":"bob","score":902}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j.collect(r#"$.people.map(name).first()"#).unwrap();

        assert_eq!(out, json!("al"));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn tape_view_map_last_reads_tail_and_materializes_one_result() {
        let j = Jetro::from_bytes(
            br#"{"people":[{"name":"al","score":1},{"name":"ada","score":901},{"name":"bob","score":902}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j.collect(r#"$.people.map(name).last()"#).unwrap();

        assert_eq!(out, json!("bob"));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn tape_view_chained_map_last_composes_late_projection() {
        let j = Jetro::from_bytes(
            br#"{"people":[{"user":{"name":"al"}},{"user":{"name":"bob"}}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"$.people.map(@.user).map(@.name).last()"#)
            .unwrap();

        assert_eq!(out, json!("bob"));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn tape_view_chained_map_take_composes_bounded_projection() {
        let j = Jetro::from_bytes(
            br#"{"people":[{"user":{"name":"al"}},{"user":{"name":"bob"}},{"user":{"name":"cy"}}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"$.people.map(@.user).map(@.name).take(2)"#)
            .unwrap();

        assert_eq!(out, json!(["al", "bob"]));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn tape_view_chained_map_nth_uses_indexed_projection() {
        let j = Jetro::from_bytes(
            br#"{"people":[{"user":{"name":"al"}},{"user":{"name":"bob"}},{"user":{"name":"cy"}}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"$.people.map(@.user).map(@.name).nth(1)"#)
            .unwrap();

        assert_eq!(out, json!("bob"));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn tape_view_filter_map_last_scans_from_tail_until_match() {
        let j = Jetro::from_bytes(
            br#"{"people":[{"name":"al","score":1},{"name":"ada","score":901},{"name":"bob","score":2},{"name":"cy","score":903}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"$.people.filter(score < 900).map(name).last()"#)
            .unwrap();

        assert_eq!(out, json!("bob"));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn tape_view_filter_last_ignores_failing_physical_tail() {
        let j = Jetro::from_bytes(
            br#"{"people":[{"name":"al","score":1},{"name":"bob","score":2},{"name":"cy","score":903}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"$.people.filter(score < 900).map(name).last()"#)
            .unwrap();

        assert_eq!(out, json!("bob"));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn tape_view_map_nth_reads_indexed_row() {
        let j = Jetro::from_bytes(
            br#"{"people":[{"name":"al","score":1},{"name":"ada","score":901},{"name":"bob","score":902}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j.collect(r#"$.people.map(name).nth(1)"#).unwrap();

        assert_eq!(out, json!("ada"));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn tape_view_filter_map_nth_preserves_filtered_semantics() {
        let j = Jetro::from_bytes(
            br#"{"people":[{"name":"al","score":1},{"name":"ada","score":901},{"name":"bob","score":902},{"name":"cy","score":903}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"$.people.filter(score > 900).map(name).nth(1)"#)
            .unwrap();

        assert_eq!(out, json!("bob"));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn tape_view_prefix_keeps_projection_builtin_suffix_as_tape_views() {
        let j = Jetro::from_bytes(
            br#"{"people":[{"name":"al","score":1},{"name":"ada","score":901},{"name":"bob","score":902},{"name":"cat","score":3}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"$.people.filter(score > 900).map(name).upper()"#)
            .unwrap();

        assert_eq!(out, json!(["ADA", "BOB"]));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn tape_row_bridge_materializes_only_demanded_rows_for_generic_prefix() {
        let j = Jetro::from_bytes(
            br#"{"people":[{"name":"al"},{"name":"ada"},{"name":"bob"},{"name":"carol"}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"$.people.filter(name.len() == 3).take(1).map(name)"#)
            .unwrap();

        assert_eq!(out, json!(["ada"]));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn tape_find_one_materializes_only_matching_result_row() {
        let j = Jetro::from_bytes(
            br#"{"people":[{"name":"al","score":1},{"name":"ada","score":9},{"name":"bob","score":2}],"unused":{"large":[1,2,3,4]}}"#
                .to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j.collect(r#"$.people.find_one(name == "ada")"#).unwrap();

        assert_eq!(out, json!({"name": "ada", "score": 9}));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 1);
    }
    #[test]
    fn tape_predicate_scalar_sinks_do_not_materialize_rows() {
        let j = Jetro::from_bytes(
            br#"{"people":[{"name":"al","score":1},{"name":"ada","score":9},{"name":"bob","score":2}],"unused":{"large":[1,2,3,4]}}"#
                .to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let any = j.collect(r#"$.people.any(score > 8)"#).unwrap();
        let all = j.collect(r#"$.people.all(score > 0)"#).unwrap();

        assert_eq!(any, json!(true));
        assert_eq!(all, json!(true));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn tape_membership_scalar_sinks_do_not_materialize_rows() {
        let j = Jetro::from_bytes(
            br#"{"people":[{"name":"al"},{"name":"ada"},{"name":"bob"}],"unused":{"large":[1,2,3,4]}}"#
                .to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let includes = j.collect(r#"$.people.map(name).includes("ada")"#).unwrap();
        let index = j.collect(r#"$.people.map(name).index("bob")"#).unwrap();

        assert_eq!(includes, json!(true));
        assert_eq!(index, json!(2));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn tape_static_arg_scalar_filter_materializes_only_output_subtree() {
        let j = Jetro::from_bytes(
            br#"{"people":[{"name":"bob"},{"name":"ada"},{"name":"amy"}],"unused":{"large":[1,2,3,4]}}"#
                .to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"$.people.filter(name.starts_with("a")).take(1).map(name)"#)
            .unwrap();

        assert_eq!(out, json!(["ada"]));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn tape_static_arg_scalar_filter_reuses_view_builtin_metadata() {
        let j = Jetro::from_bytes(
            br#"{"people":[{"name":"bob"},{"name":"zoe"},{"name":"ada"}],"unused":{"large":[1,2,3,4]}}"#
                .to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"$.people.filter(name.ends_with("a")).take(1).map(name)"#)
            .unwrap();

        assert_eq!(out, json!(["ada"]));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn tape_string_predicate_scalar_filter_stays_view_native() {
        let j = Jetro::from_bytes(
            br#"{"people":[{"name":"bob"},{"name":"ada"},{"name":"amy"}],"unused":{"large":[1,2,3,4]}}"#
                .to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"$.people.filter(name.matches("ad")).take(1).map(name)"#)
            .unwrap();

        assert_eq!(out, json!(["ada"]));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn tape_string_index_scalar_filter_stays_view_native() {
        let j = Jetro::from_bytes(
            br#"{"people":[{"name":"bob"},{"name":"ada"},{"name":"amy"}],"unused":{"large":[1,2,3,4]}}"#
                .to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"$.people.filter(name.index_of("d") >= 1).take(1).map(name)"#)
            .unwrap();

        assert_eq!(out, json!(["ada"]));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn tape_no_arg_string_scalar_filter_preserves_output_demand() {
        let j = Jetro::from_bytes(
            br#"{"people":[{"code":"abc"},{"code":"123"},{"code":"456"}],"unused":{"large":[1,2,3,4]}}"#
                .to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"$.people.filter(code.is_numeric()).take(1).map(code)"#)
            .unwrap();

        assert_eq!(out, json!(["123"]));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn tape_no_arg_numeric_string_scalar_filter_preserves_output_demand() {
        let j = Jetro::from_bytes(
            br#"{"people":[{"code":"xx"},{"code":"abc"},{"code":"def"}],"unused":{"large":[1,2,3,4]}}"#
                .to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"$.people.filter(code.byte_len() == 3).take(1).map(code)"#)
            .unwrap();

        assert_eq!(out, json!(["abc"]));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn tape_numeric_scalar_filter_preserves_output_demand() {
        let j = Jetro::from_bytes(
            br#"{"people":[{"score":3},{"score":-12},{"score":20}],"unused":{"large":[1,2,3,4]}}"#
                .to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"$.people.filter(score.abs() > 10).take(1).map(score)"#)
            .unwrap();

        assert_eq!(out, json!([-12]));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn tape_float_numeric_scalar_filter_preserves_output_demand() {
        let j = Jetro::from_bytes(
            br#"{"people":[{"score":8.2},{"score":9.7},{"score":10.2}],"unused":{"large":[1,2,3,4]}}"#
                .to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"$.people.filter(score.round() == 10).take(1).map(score)"#)
            .unwrap();

        assert_eq!(out, json!([9.7]));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn object_shape_tape_pipeline_generic_first_stage_uses_row_bridge() {
        let j = Jetro::from_bytes(
            br#"{"people":[{"name":"al"},{"name":"ada"},{"name":"bob"},{"name":"carol"}],"meta":{"version":3}}"#.to_vec(),
        )
        .unwrap();

        let out = j
            .collect(
                r#"{"first": $.people.filter(name.len() == 3).take(1).map(name), "v": $.meta.version}"#,
            )
            .unwrap();

        assert_eq!(out, json!({"first": ["ada"], "v": 3}));
        assert!(!j.root_val_is_materialized());
    }
    #[test]
    fn view_pipeline_count_and_sum_read_from_tape_without_materializing_root_val() {
        let j = Jetro::from_bytes(
            br#"{"books":[{"score":1},{"score":901},{"score":902},{"score":2}]}"#.to_vec(),
        )
        .unwrap();

        let count = j.collect(r#"$.books.filter(score > 900).count()"#).unwrap();
        let sum = j
            .collect(r#"$.books.filter(score > 900).map(score).sum()"#)
            .unwrap();
        let direct_count = j.collect(r#"$.books.count(score > 900)"#).unwrap();
        let direct_sum = j.collect(r#"$.books.sum(score)"#).unwrap();

        assert_eq!(count, json!(2));
        assert_eq!(sum, json!(1803));
        assert_eq!(direct_count, json!(2));
        assert_eq!(direct_sum, json!(1806));
        assert!(!j.root_val_is_materialized());
    }
    #[test]
    fn view_numeric_projection_sink_reads_scalar_keys_without_materializing_subtrees() {
        let j = Jetro::from_bytes(
            br#"{"books":[{"score":1},{"score":901},{"score":902},{"score":2}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j.collect(r#"$.books.sum(score)"#).unwrap();

        assert_eq!(out, json!(1806));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn view_approx_count_distinct_hashes_tape_scalars_without_materializing_rows() {
        let j = Jetro::from_bytes(
            br#"{"books":[{"tag":"a"},{"tag":"b"},{"tag":"a"},{"tag":"c"}],"unused":{"large":[1,2,3,4]}}"#
                .to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"$.books.map(tag).approx_count_distinct()"#)
            .unwrap();

        assert!(out.as_i64().unwrap() >= 3);
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }

    #[test]
    fn view_count_by_reduces_tape_rows_without_materializing_rows() {
        let j = Jetro::from_bytes(
            br#"{"books":[{"genre":"sf","title":"a"},{"genre":"fantasy","title":"b"},{"genre":"sf","title":"c"}],"unused":{"large":[1,2,3,4]}}"#
                .to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j.collect(r#"$.books.count_by(genre)"#).unwrap();

        assert_eq!(out, json!({"sf": 2, "fantasy": 1}));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }

    #[test]
    fn view_index_by_materializes_only_indexed_result_rows() {
        let j = Jetro::from_bytes(
            br#"{"books":[{"id":"a","title":"old"},{"id":"b","title":"bee"},{"id":"a","title":"new"}],"unused":{"large":[1,2,3,4]}}"#
                .to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j.collect(r#"$.books.index_by(id)"#).unwrap();

        assert_eq!(
            out,
            json!({
                "a": {"id": "a", "title": "new"},
                "b": {"id": "b", "title": "bee"}
            })
        );
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 2);
    }

    #[test]
    fn view_group_by_reduces_tape_rows_without_materializing_root() {
        let j = Jetro::from_bytes(
            br#"{"books":[{"genre":"sf","title":"a"},{"genre":"fantasy","title":"b"},{"genre":"sf","title":"c"}],"unused":{"large":[1,2,3,4]}}"#
                .to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j.collect(r#"$.books.group_by(genre)"#).unwrap();

        assert_eq!(
            out,
            json!({
                "sf": [
                    {"genre": "sf", "title": "a"},
                    {"genre": "sf", "title": "c"}
                ],
                "fantasy": [
                    {"genre": "fantasy", "title": "b"}
                ]
            })
        );
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 3);
    }
    #[test]
    fn view_flat_map_then_map_reads_from_tape_without_materializing_root_val() {
        let j = Jetro::from_bytes(
            br#"{"data":[{"items":[{"price":10},{"price":20}]},{"items":[{"price":30}]},{"items":[]}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();

        let out = j.collect(r#"$.data.flat_map(items).map(price)"#).unwrap();

        assert_eq!(out, json!([10, 20, 30]));
        assert!(!j.root_val_is_materialized());
    }
    #[test]
    fn view_flat_map_take_stops_after_expanded_rows_without_materializing_root_val() {
        let j = Jetro::from_bytes(
            br#"{"data":[{"items":[{"price":10},{"price":20},{"price":30}]},{"items":[{"price":40}]}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();

        let out = j
            .collect(r#"$.data.flat_map(items).take(2).map(price)"#)
            .unwrap();

        assert_eq!(out, json!([10, 20]));
        assert!(!j.root_val_is_materialized());
    }
    #[test]
    fn view_flat_map_take_collects_expanded_rows_without_materializing_root_val() {
        let j = Jetro::from_bytes(
            br#"{"data":[{"items":[{"price":10},{"price":20},{"price":30}]},{"items":[{"price":40}]}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j.collect(r#"$.data.flat_map(items).take(2)"#).unwrap();

        assert_eq!(out, json!([{"price": 10}, {"price": 20}]));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 2);
    }
    #[test]
    fn view_prefix_materializes_boundary_rows_not_root_for_suffix_builtin() {
        let j = Jetro::from_bytes(
            br#"{"people":[{"name":"low","score":1},{"name":"ada","score":901},{"name":"bob","score":902}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();

        let out = j
            .collect(r#"$.people.filter(score > 900).map(name).upper()"#)
            .unwrap();

        assert_eq!(out, json!(["ADA", "BOB"]));
        assert!(!j.root_val_is_materialized());
    }
    #[test]
    fn view_prefix_allows_current_only_generic_suffix_without_materializing_root() {
        let j = Jetro::from_bytes(
            br#"{"people":[{"name":"low","score":1},{"name":"ada","score":901},{"name":"bob","score":902}],"target":"ada"}"#.to_vec(),
        )
        .unwrap();

        let out = j
            .collect(r#"$.people.filter(score > 900).map(name).take(10).filter(@.len() == 3)"#)
            .unwrap();

        assert_eq!(out, json!(["ada", "bob"]));
        assert!(!j.root_val_is_materialized());
    }
    #[test]
    fn view_prefix_uses_stage_metadata_for_materialized_suffix_barriers() {
        let j = Jetro::from_bytes(
            br#"{"people":[{"name":"low","score":1},{"name":"bob","score":902},{"name":"ada","score":901},{"name":"bob","score":903}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();

        let out = j
            .collect(r#"$.people.filter(score > 900).map(name).sort().unique().count()"#)
            .unwrap();

        assert_eq!(out, json!(2));
        assert!(!j.root_val_is_materialized());
    }
    #[test]
    fn view_sort_topk_materializes_only_winners_for_current_projection_suffix() {
        let j = Jetro::from_bytes(
            br#"{"data":[{"id":1,"score":10,"user":{"name":"low"}},{"id":2,"score":30,"user":{"name":"top"}},{"id":3,"score":20,"user":{"name":"mid"}}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"$.data.sort_by(-score).take(2).map({id, name: user.name, score})"#)
            .unwrap();

        assert_eq!(
            out,
            json!([
                {"id": 2, "name": "top", "score": 30},
                {"id": 3, "name": "mid", "score": 20}
            ])
        );
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn view_sort_topk_keeps_projection_builtin_suffix_as_tape_views() {
        let j = Jetro::from_bytes(
            br#"{"data":[{"name":"low","score":10},{"name":"top","score":30},{"name":"mid","score":20}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"$.data.sort_by(-score).take(2).map(name).upper()"#)
            .unwrap();

        assert_eq!(out, json!(["TOP", "MID"]));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn view_sort_topk_keeps_object_key_suffix_as_tape_views() {
        let j = Jetro::from_bytes(
            br#"{"data":[{"score":10},{"isbn":"top","score":30},{"isbn":"mid","score":20}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"$.data.sort_by(-score).take(2).has_key("isbn")"#)
            .unwrap();

        assert_eq!(out, json!([true, true]));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn view_sort_tail_keeps_object_key_terminal_projection_as_tape_view() {
        let j = Jetro::from_bytes(
            br#"{"data":[{"isbn":"top","score":30},{"score":20},{"isbn":"low","score":10}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"$.data.sort_by(-score).has_key("isbn").last()"#)
            .unwrap();

        assert_eq!(out, json!(true));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn view_sort_tail_keeps_scalar_terminal_projection_as_tape_view() {
        let j = Jetro::from_bytes(
            br#"{"data":[{"name":"top","score":30},{"name":"mid","score":20},{"name":"low","score":10}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"$.data.sort_by(-score).map(name).upper().last()"#)
            .unwrap();

        assert_eq!(out, json!("LOW"));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn view_sort_tail_many_keeps_projection_as_tape_views() {
        let j = Jetro::from_bytes(
            br#"{"data":[{"isbn":"top","score":30},{"isbn":"mid","score":20},{"isbn":"low","score":10}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"$.data.sort_by(-score).map(isbn).last(2)"#)
            .unwrap();

        assert_eq!(out, json!(["mid", "low"]));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn view_sort_head_many_keeps_projection_as_tape_views() {
        let j = Jetro::from_bytes(
            br#"{"data":[{"isbn":"top","score":30},{"isbn":"mid","score":20},{"isbn":"low","score":10}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"$.data.sort_by(-score).map(isbn).first(2)"#)
            .unwrap();

        assert_eq!(out, json!(["top", "mid"]));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn view_sort_nth_keeps_projection_as_tape_view() {
        let j = Jetro::from_bytes(
            br#"{"data":[{"isbn":"top","score":30},{"isbn":"mid","score":20},{"isbn":"low","score":10}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"$.data.sort_by(-score).map(isbn).nth(1)"#)
            .unwrap();

        assert_eq!(out, json!("mid"));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn view_prefix_streams_into_sort_topk_without_materializing_prefix_rows() {
        let j = Jetro::from_bytes(
            br#"{"data":[{"name":"low","score":10},{"name":"top","score":30},{"name":"mid","score":20},{"name":"skip","score":5}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"$.data.filter(score > 10).sort_by(-score).take(2).map(name)"#)
            .unwrap();

        assert_eq!(out, json!(["top", "mid"]));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn view_sort_until_output_feeds_take_while_suffix_as_tape_views() {
        let j = Jetro::from_bytes(
            br#"{"data":[{"name":"top","score":40,"price":20},{"name":"mid","score":30,"price":30},{"name":"stop","score":20,"price":5},{"name":"late","score":10,"price":99}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"$.data.sort_by(-score).take_while(price > 10).take(2).map(name)"#)
            .unwrap();

        assert_eq!(out, json!(["top", "mid"]));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }

    #[test]
    fn view_sort_drop_while_filter_map_last_matches_value_backend() {
        let doc = json!({
            "data": [
                {"name": "skip_test", "isbn": "skip", "score": 100, "price": 90},
                {"name": "keep-high", "isbn": "high", "score": 90, "price": 30},
                {"name": "keep-low", "isbn": "low", "score": 80, "price": 25},
                {"name": "cheap", "isbn": "cheap", "score": 70, "price": 5}
            ],
            "unused": {"large": [1, 2, 3, 4]}
        });
        let data = serde_json::to_vec(&doc).unwrap();
        let from_tape = Jetro::from_bytes(data).unwrap();
        let engine = JetroEngine::new();
        from_tape.reset_tape_materialized_subtrees();

        let query = r#"$.data.sort_by(-score).drop_while(name.contains("_test")).filter(price > 20).map(isbn).last()"#;
        let tape_out = from_tape.collect(query).unwrap();
        let value_out = engine.collect_value(doc, query).unwrap();

        assert_eq!(tape_out, value_out);
        assert_eq!(tape_out, json!("low"));
        assert!(!from_tape.root_val_is_materialized());
        assert_eq!(from_tape.tape_materialized_subtrees(), 0);
    }

    #[test]
    fn view_drop_while_map_first_streams_without_materializing_root() {
        let j = Jetro::from_bytes(
            br#"{"data":[{"isbn":"skip","price":10},{"isbn":"answer","price":30},{"isbn":"late","price":40}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"$.data.drop_while(price < 20).map(isbn).first()"#)
            .unwrap();

        assert_eq!(out, json!("answer"));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }

    #[test]
    fn view_sort_filter_map_last_scans_sorted_tail_until_match() {
        let j = Jetro::from_bytes(
            br#"{"data":[{"isbn":"top-fails","score":100,"price":10},{"isbn":"answer","score":90,"price":30},{"isbn":"tail-fails","score":80,"price":5}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"$.data.sort_by(-score).filter(price > 20).map(isbn).last()"#)
            .unwrap();

        assert_eq!(out, json!("answer"));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn view_sort_filter_map_nth_stays_borrowed() {
        let j = Jetro::from_bytes(
            br#"{"data":[{"isbn":"skip","score":100,"price":10},{"isbn":"first","score":90,"price":30},{"isbn":"second","score":80,"price":40},{"isbn":"tail","score":70,"price":50}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"$.data.sort_by(-score).filter(price > 20).map(isbn).nth(1)"#)
            .unwrap();

        assert_eq!(out, json!("second"));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn view_sort_filter_map_last_many_stays_borrowed() {
        let j = Jetro::from_bytes(
            br#"{"data":[{"isbn":"skip","score":100,"price":10},{"isbn":"first","score":90,"price":30},{"isbn":"second","score":80,"price":40},{"isbn":"tail","score":70,"price":50}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"$.data.sort_by(-score).filter(price > 20).map(isbn).last(2)"#)
            .unwrap();

        assert_eq!(out, json!(["second", "tail"]));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }

    #[test]
    fn view_dynamic_object_key_projection_stays_tape_streamed() {
        let j = Jetro::from_bytes(
            br#"{"data":[{"kind":"isbn","value":"978"},{"kind":"sku","value":"A-1"}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j.collect(r#"$.data.map({[@.kind]: value})"#).unwrap();

        assert_eq!(out, json!([{"isbn": "978"}, {"sku": "A-1"}]));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }

    #[test]
    fn view_sort_string_predicate_map_last_stays_borrowed() {
        let j = Jetro::from_bytes(
            br#"{"data":[{"name":"prod","score":100},{"name":"skip_test","score":90},{"name":"answer","score":80}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"$.data.sort_by(-score).filter(name.ends_with("er")).map(name).last()"#)
            .unwrap();

        assert_eq!(out, json!("answer"));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn view_sort_numeric_predicate_map_last_stays_borrowed() {
        let j = Jetro::from_bytes(
            br#"{"data":[{"isbn":"top","score":100,"delta":-1.2},{"isbn":"mid","score":90,"delta":-2.4},{"isbn":"answer","score":80,"delta":2.6}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"$.data.sort_by(-score).filter(delta.abs() > 2.0).map(isbn).last()"#)
            .unwrap();

        assert_eq!(out, json!("answer"));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn view_membership_dynamic_target_stops_without_row_materialization() {
        let j = Jetro::from_bytes(
            br#"{"xs":["a","b","needle","tail"],"needle":"needle","unused":{"large":[1,2,3,4]}}"#
                .to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j.collect(r#"$.xs.includes($.needle)"#).unwrap();

        assert_eq!(out, json!(true));
        assert!(j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn view_receiver_includes_filters_nested_tape_arrays_without_materializing_rows() {
        let j = Jetro::from_bytes(
            br#"{"books":[{"price":12.5,"tags":["sf","classic"]},{"price":14,"tags":["sf","hugo"]},{"price":9,"tags":["sf"]}]}"#
                .to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(r#"$.books.filter(tags.includes("hugo")).map(price).sum()"#)
            .unwrap();

        assert_eq!(out, json!(14));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn view_index_dynamic_target_stops_without_row_materialization() {
        let j = Jetro::from_bytes(
            br#"{"xs":["a","b","needle","tail"],"needle":"needle","unused":{"large":[1,2,3,4]}}"#
                .to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j.collect(r#"$.xs.index($.needle)"#).unwrap();

        assert_eq!(out, json!(2));
        assert!(j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn view_indices_dynamic_target_scans_without_row_materialization() {
        let j = Jetro::from_bytes(
            br#"{"xs":["needle","b","needle","tail"],"needle":"needle","unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j.collect(r#"$.xs.indices_of($.needle)"#).unwrap();

        assert_eq!(out, json!([0, 2]));
        assert!(j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn view_sort_tail_pick_omit_helpers_only_materialize_outputs() {
        let j = Jetro::from_bytes(
            br#"{"data":[{"isbn":"top","score":30,"debug":1},{"isbn":"low","score":10,"debug":2}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let picked = j
            .collect(r#"$.data.sort_by(-score).map(@.pick("isbn")).last()"#)
            .unwrap();
        let omitted = j
            .collect(r#"$.data.sort_by(-score).map(@.omit("debug")).last()"#)
            .unwrap();

        assert_eq!(picked, json!({"isbn": "low"}));
        assert_eq!(omitted, json!({"isbn": "low", "score": 10}));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn view_sort_tail_object_collection_helpers_stay_borrowed() {
        let j = Jetro::from_bytes(
            br#"{"data":[{"isbn":"top","score":30},{"isbn":"low","score":10}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let keys = j
            .collect(r#"$.data.sort_by(-score).map(@.keys()).last()"#)
            .unwrap();
        let values = j
            .collect(r#"$.data.sort_by(-score).map(@.values()).last()"#)
            .unwrap();
        let entries = j
            .collect(r#"$.data.sort_by(-score).map(@.entries()).last()"#)
            .unwrap();

        assert_eq!(keys, json!(["isbn", "score"]));
        assert_eq!(values, json!(["low", 10]));
        assert_eq!(entries, json!([["isbn", "low"], ["score", 10]]));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn view_sort_tail_path_helpers_stay_borrowed() {
        let j = Jetro::from_bytes(
            br#"{"data":[{"user":{"name":"top"},"score":30},{"user":{"name":"low"},"score":10}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let name = j
            .collect(r#"$.data.sort_by(-score).map(@.get_path("user.name")).last()"#)
            .unwrap();
        let found = j
            .collect(r#"$.data.sort_by(-score).map(@.has_path("user.name")).last()"#)
            .unwrap();

        assert_eq!(name, json!("low"));
        assert_eq!(found, json!(true));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn view_object_predicate_and_projection_after_sort_stay_borrowed() {
        let j = Jetro::from_bytes(
            br#"{"data":[{"meta":{"isbn":"top","price":30,"debug":1},"score":30},{"meta":{"price":20,"debug":2},"score":20},{"meta":{"isbn":"low","price":10,"debug":3},"score":10}],"unused":{"large":[1,2,3,4]}}"#.to_vec(),
        )
        .unwrap();
        j.reset_tape_materialized_subtrees();

        let out = j
            .collect(
                r#"$.data.sort_by(-score).filter(@.meta.has_key("isbn")).map(@.meta.pick("isbn", "price")).last()"#,
            )
            .unwrap();

        assert_eq!(out, json!({"isbn": "low", "price": 10}));
        assert!(!j.root_val_is_materialized());
        assert_eq!(j.tape_materialized_subtrees(), 0);
    }
    #[test]
    fn view_sort_tail_object_helper_chain_stays_borrowed() {
        let data = br#"{"data":[{"meta":{"isbn":"top","author":{"name":"ada"},"debug":1},"score":30},{"meta":{"isbn":"mid","author":{"name":"bea"},"debug":2},"score":20},{"meta":{"isbn":"low","author":{"name":"cat"},"debug":3},"score":10}],"unused":{"large":[1,2,3,4]}}"#.to_vec();
        let from_tape = Jetro::from_bytes(data.clone()).unwrap();
        let from_value = Jetro::from_bytes(data).unwrap();
        from_tape.reset_tape_materialized_subtrees();

        let query = r#"$.data.sort_by(-score).map(@.meta.omit("debug").pick("isbn", "author").get_path("author.name").upper()).last()"#;
        let tape_out = from_tape.collect(query).unwrap();
        let value_out = from_value.collect(query).unwrap();

        assert_eq!(tape_out, value_out);
        assert_eq!(tape_out, json!("CAT"));
        assert!(!from_tape.root_val_is_materialized());
        assert_eq!(from_tape.tape_materialized_subtrees(), 0);
    }

    #[test]
    fn view_sort_topk_scalar_helper_reducer_chain_stays_borrowed() {
        let doc = json!({
            "data": [
                {"meta": {"author": {"name": " Ada "}}, "score": 40},
                {"meta": {"author": {"name": "Bea"}}, "score": 30},
                {"meta": {"author": {"name": "Cyd"}}, "score": 20},
                {"meta": {"author": {"name": "Dee"}}, "score": 10}
            ],
            "unused": {"large": [1, 2, 3, 4]}
        });
        let data = serde_json::to_vec(&doc).unwrap();
        let from_tape = Jetro::from_bytes(data).unwrap();
        let engine = JetroEngine::new();
        from_tape.reset_tape_materialized_subtrees();

        let query = r#"$.data.sort_by(-score).take(3).map(@.meta.get_path("author.name").trim().upper().byte_len()).sum()"#;
        let tape_out = from_tape.collect(query).unwrap();
        let value_out = engine.collect_value(doc, query).unwrap();

        assert_eq!(tape_out, value_out);
        assert_eq!(tape_out, json!(9));
        assert!(!from_tape.root_val_is_materialized());
        assert_eq!(from_tape.tape_materialized_subtrees(), 0);
    }

    #[test]
    fn view_prefix_and_full_execution_share_stage_semantics() {
        let data = br#"{"people":[{"name":"low","score":1},{"name":"ada","score":901},{"name":"bob","score":902},{"name":"cat","score":903},{"name":"dan","score":904}],"unused":{"large":[1,2,3,4]}}"#.to_vec();
        let full = Jetro::from_bytes(data.clone()).unwrap();
        let prefix = Jetro::from_bytes(data).unwrap();

        let full_out = full
            .collect(r#"$.people.skip(1).take(3).filter(score > 901).map(name).count()"#)
            .unwrap();
        let prefix_out = prefix
            .collect(r#"$.people.skip(1).take(3).filter(score > 901).map(name).upper().count()"#)
            .unwrap();

        assert_eq!(full_out, prefix_out);
        assert_eq!(full_out, json!(2));
        assert!(!full.root_val_is_materialized());
        assert!(!prefix.root_val_is_materialized());
    }
    #[test]
    fn view_prefix_rejects_root_dependent_generic_suffix() {
        let j = Jetro::from_bytes(
            br#"{"people":[{"name":"low","score":1},{"name":"ada","score":901},{"name":"bob","score":902}],"target":"ada"}"#.to_vec(),
        )
        .unwrap();

        let out = j
            .collect(r#"$.people.filter(score > 900).map(name).take(10).filter(@ == $.target)"#)
            .unwrap();

        assert_eq!(out, json!(["ada"]));
        assert!(j.root_val_is_materialized());
    }
    #[test]
    fn byte_native_root_facts_match_no_root_materialization_execution() {
        let expr = r#"{"a": $.rows.filter(score > 10).take(1), "b": $.meta.version}"#;
        let plan = planner::plan_query_with_context(expr, planner::PlanningContext::bytes());
        assert!(plan.root_execution_facts().is_byte_native());

        let j = Jetro::from_bytes(
            br#"{"rows":[{"score":11},{"score":3}],"meta":{"version":1},"unused":{"large":[1,2,3,4]}}"#
                .to_vec(),
        )
        .unwrap();

        let out = super::collect_plan_json(&j, &plan).unwrap();

        assert_eq!(out, json!({"a": [{"score": 11}], "b": 1}));
        assert!(!j.root_val_is_materialized());
    }
    #[test]
    fn byte_native_nodes_reject_interpreted_backend_at_runtime() {
        let node = PlanNode::RootPath(vec![PhysicalPathStep::Field(Arc::from("meta"))]);
        let plan = QueryPlan::from_physical_nodes(
            NodeId(0),
            vec![PhysicalNode::with_backend_plan(
                node,
                BackendPlan::new(&[BackendPreference::Interpreted]),
            )],
        );
        assert!(plan.root_execution_facts().is_byte_native());

        let j = Jetro::from_bytes(br#"{"meta":1}"#.to_vec()).unwrap();
        let err = super::collect_plan_json(&j, &plan).unwrap_err();

        assert!(err
            .0
            .contains("no planned backend could execute physical node"));
        assert!(!j.root_val_is_materialized());
    }

    #[test]
    fn executor_skips_backends_not_advertised_by_capabilities() {
        let node = PlanNode::RootPath(vec![PhysicalPathStep::Field(Arc::from("meta"))]);
        let plan = QueryPlan::from_physical_nodes(
            NodeId(0),
            vec![PhysicalNode::with_backend_plan_capabilities_and_facts(
                node,
                BackendPlan::new(&[BackendPreference::Interpreted]),
                BackendSet::TAPE_PATH,
                ExecutionFacts::default(),
            )],
        );

        let j = Jetro::from(json!({"meta": 1}));
        let err = super::collect_plan_json(&j, &plan).unwrap_err();

        assert!(err
            .0
            .contains("no planned backend could execute physical node"));
    }
    #[test]
    fn byte_native_dynamic_index_chain_executes_without_root_materialization() {
        let expr = r#"{"item": $.items[$.index].name, "field": $.object[$.key]}"#;
        let plan = planner::plan_query_with_context(expr, planner::PlanningContext::bytes());
        assert!(plan.root_execution_facts().is_byte_native());

        let j = Jetro::from_bytes(
            br#"{"items":[{"name":"zero"},{"name":"one"}],"index":1,"object":{"x":7},"key":"x"}"#
                .to_vec(),
        )
        .unwrap();

        let out = super::collect_plan_json(&j, &plan).unwrap();

        assert_eq!(out, json!({"item": "one", "field": 7}));
        assert!(!j.root_val_is_materialized());
    }
    #[test]
    fn byte_native_scalar_expressions_execute_without_root_materialization() {
        let expr = r#"{"gt": $.n > 1, "sum": $.n + 4, "picked": "yes" if $.ok else "no", "fallback": $.missing ?? $.n}"#;
        let plan = planner::plan_query_with_context(expr, planner::PlanningContext::bytes());
        assert!(plan.root_execution_facts().is_byte_native());

        let j = Jetro::from_bytes(br#"{"n":3,"ok":true}"#.to_vec()).unwrap();

        let out = super::collect_plan_json(&j, &plan).unwrap();

        assert_eq!(
            out,
            json!({"gt": true, "sum": 7, "picked": "yes", "fallback": 3})
        );
        assert!(!j.root_val_is_materialized());
    }
    #[test]
    fn byte_native_composite_fields_execute_without_root_materialization() {
        let expr =
            r#"[{"a": $.a when $.ok, [$.key]: $.value, ...$.base, ...**$.deep}, ...$.items]"#;
        let plan = planner::plan_query_with_context(expr, planner::PlanningContext::bytes());
        assert!(plan.root_execution_facts().is_byte_native());

        let j = Jetro::from_bytes(
            br#"{"a":1,"ok":true,"key":"dyn","value":2,"base":{"b":3},"deep":{"nested":{"c":4}},"items":[5,6]}"#
                .to_vec(),
        )
        .unwrap();

        let out = super::collect_plan_json(&j, &plan).unwrap();

        assert_eq!(
            out,
            json!([{"a": 1, "dyn": 2, "b": 3, "nested": {"c": 4}}, 5, 6])
        );
        assert!(!j.root_val_is_materialized());
    }
    #[test]
    fn vm_fallback_root_facts_match_materialized_execution() {
        let expr = r#"{"a": [x for x in $.rows if x.score > 10], "b": $.meta.version}"#;
        let plan = planner::plan_query_with_context(expr, planner::PlanningContext::bytes());
        assert!(!plan.root_execution_facts().is_byte_native());
        assert!(plan.root_execution_facts().contains_vm_fallback);

        let j = Jetro::from_bytes(
            br#"{"rows":[{"score":11},{"score":3}],"meta":{"version":1}}"#.to_vec(),
        )
        .unwrap();

        let out = super::collect_plan_json(&j, &plan).unwrap();

        assert_eq!(out, json!({"a": [{"score": 11}], "b": 1}));
        assert!(j.root_val_is_materialized());
    }
    #[test]
    fn root_receiver_sort_executes_without_root_val_cache_materialization() {
        let plan = planner::plan_query_with_context("$.sort()", planner::PlanningContext::bytes());
        let QueryRoot::Node(root) = plan.root() else {
            panic!("expected physical expression plan");
        };
        assert_no_vm_fallback(&plan, *root);

        let j = Jetro::from_bytes(br#"[3,1,2]"#.to_vec()).unwrap();
        let out = super::collect_plan_json(&j, &plan).unwrap();

        assert_eq!(out, json!([1, 2, 3]));
        assert!(!j.root_val_is_materialized());
    }

    #[test]
    fn object_shape_executes_common_scalar_nodes_without_vm() {
        let expr = r#"{"gt": $.a > 1, "sum": $.a + 4, "picked": "yes" if $.ok else "no"}"#;
        let plan = planner::plan_query(expr);
        let QueryRoot::Node(root) = plan.root() else {
            panic!("expected physical expression plan");
        };
        assert_no_vm_fallback(&plan, *root);

        let j = Jetro::from(json!({
            "a": 3,
            "ok": true
        }));

        let out = j.collect(expr).unwrap();

        assert_eq!(out, json!({"gt": true, "sum": 7, "picked": "yes"}));
    }

    #[test]
    fn object_shape_executes_scalar_chains_without_vm() {
        let expr = r#"let k = "name" in {"current": @.user.name, "ident": user.name, "dyn": user[k], "method": user.name.upper().trim()}"#;
        let plan = planner::plan_query(expr);
        let QueryRoot::Node(root) = plan.root() else {
            panic!("expected physical expression plan");
        };
        assert_no_vm_fallback(&plan, *root);

        let j = Jetro::from(json!({
            "user": {"name": " ada "}
        }));

        let out = j.collect(expr).unwrap();

        assert_eq!(
            out,
            json!({
                "current": " ada ",
                "ident": " ada ",
                "dyn": " ada ",
                "method": "ADA"
            })
        );
    }

    #[test]
    fn let_body_can_contain_object_shape_with_pipeline_child() {
        let j = Jetro::from(json!({
            "books": [
                {"id": 1, "price": 5},
                {"id": 2, "price": 15}
            ]
        }));

        let out = j
            .collect(r#"let x = 1 in {"ids": $.books.filter(price > 10).map(id), "x": x}"#)
            .unwrap();

        assert_eq!(out, json!({"ids": [2], "x": 1}));
    }

    #[test]
    fn let_bound_values_are_visible_inside_pipeline_children() {
        let expr = r#"let min_score = 900 in {"top": $.books.filter(score > min_score).take(2).map(title), "first": $.books.filter(score > min_score).first()}"#;
        let plan = planner::plan_query(expr);
        let QueryRoot::Node(root) = plan.root() else {
            panic!("expected physical expression plan");
        };
        let PlanNode::Let { body, .. } = plan.node(*root) else {
            panic!("expected let root");
        };
        let PlanNode::Object(fields) = plan.node(*body) else {
            panic!("expected object body");
        };
        for idx in [0usize, 1] {
            let PhysicalObjField::Kv { val, .. } = &fields[idx] else {
                panic!("expected pipeline kv field");
            };
            assert!(matches!(plan.node(*val), PlanNode::Pipeline { .. }));
        }

        let j = Jetro::from(json!({
            "books": [
                {"title": "low", "score": 1},
                {"title": "a", "score": 901},
                {"title": "b", "score": 902},
                {"title": "c", "score": 903}
            ]
        }));

        let out = j.collect(expr).unwrap();

        assert_eq!(
            out,
            json!({
                "top": ["a", "b"],
                "first": {"title": "a", "score": 901}
            })
        );
    }
    #[test]
    fn local_ident_pipeline_body_uses_env_not_row_field_kernel() {
        let expr = r#"let title = "fixed" in $.books.map(title).take(2)"#;
        let plan = planner::plan_query_with_context(expr, planner::PlanningContext::bytes());
        let QueryRoot::Node(root) = plan.root() else {
            panic!("expected physical expression plan");
        };
        let PlanNode::Let { body, .. } = plan.node(*root) else {
            panic!("expected let root");
        };
        let PlanNode::Pipeline { body, .. } = plan.node(*body) else {
            panic!("expected pipeline body");
        };
        assert!(matches!(
            body.stage_kernels.first(),
            Some(BodyKernel::Generic)
        ));

        let j = Jetro::from_bytes(
            br#"{"books":[{"title":"row-a"},{"title":"row-b"},{"title":"row-c"}]}"#.to_vec(),
        )
        .unwrap();
        let out = super::collect_plan_json(&j, &plan).unwrap();

        assert_eq!(out, json!(["fixed", "fixed"]));
        assert!(!j.root_val_is_materialized());
    }
    #[test]
    fn local_ident_reducer_projection_uses_env_not_row_field_kernel() {
        let expr = r#"let bonus = 10 in $.books.sum(bonus)"#;
        let plan = planner::plan_query_with_context(expr, planner::PlanningContext::bytes());
        let QueryRoot::Node(root) = plan.root() else {
            panic!("expected physical expression plan");
        };
        let PlanNode::Let { body, .. } = plan.node(*root) else {
            panic!("expected let root");
        };
        let PlanNode::Pipeline { body, .. } = plan.node(*body) else {
            panic!("expected pipeline body");
        };
        assert!(matches!(
            body.sink_kernels.first(),
            Some(BodyKernel::Generic)
        ));

        let j = Jetro::from_bytes(br#"{"books":[{"bonus":1},{"bonus":2},{"bonus":3}]}"#.to_vec())
            .unwrap();
        let out = super::collect_plan_json(&j, &plan).unwrap();

        assert_eq!(out, json!(30));
        assert!(!j.root_val_is_materialized());
    }
    #[test]
    fn local_ident_predicate_sink_uses_env_not_row_field_kernel() {
        let expr = r#"let threshold = 10 in $.books.any(threshold > 5)"#;
        let plan = planner::plan_query_with_context(expr, planner::PlanningContext::bytes());
        let QueryRoot::Node(root) = plan.root() else {
            panic!("expected physical expression plan");
        };
        let PlanNode::Let { body, .. } = plan.node(*root) else {
            panic!("expected let root");
        };
        let PlanNode::Pipeline { body, .. } = plan.node(*body) else {
            panic!("expected pipeline body");
        };
        assert!(matches!(
            body.sink_kernels.first(),
            Some(BodyKernel::Generic)
        ));

        let j = Jetro::from_bytes(
            br#"{"books":[{"threshold":1},{"threshold":2},{"threshold":3}]}"#.to_vec(),
        )
        .unwrap();
        let out = super::collect_plan_json(&j, &plan).unwrap();

        assert_eq!(out, json!(true));
        assert!(!j.root_val_is_materialized());
    }
    #[test]
    fn local_ident_arg_extreme_sink_uses_env_not_row_field_kernel() {
        let expr = r#"let key = 10 in $.books.max_by(key).title"#;
        let plan = planner::plan_query_with_context(expr, planner::PlanningContext::bytes());
        let QueryRoot::Node(root) = plan.root() else {
            panic!("expected physical expression plan");
        };
        let PlanNode::Let { body, .. } = plan.node(*root) else {
            panic!("expected let root");
        };
        let PlanNode::Chain { base, .. } = plan.node(*body) else {
            panic!("expected chain body");
        };
        let PlanNode::Pipeline { body, .. } = plan.node(*base) else {
            panic!("expected pipeline base");
        };
        assert!(matches!(
            body.sink_kernels.first(),
            Some(BodyKernel::Generic)
        ));

        let j = Jetro::from_bytes(
            br#"{"books":[{"title":"first","key":1},{"title":"second","key":99}]}"#.to_vec(),
        )
        .unwrap();
        let out = super::collect_plan_json(&j, &plan).unwrap();

        assert_eq!(out, json!("first"));
        assert!(!j.root_val_is_materialized());
    }
    #[test]
    fn local_ident_compiled_map_body_uses_env_not_inner_row_field() {
        let expr = r#"let label = "fixed" in $.books.map(@.items.map(label).take(1))"#;
        let plan = planner::plan_query_with_context(expr, planner::PlanningContext::bytes());
        let QueryRoot::Node(root) = plan.root() else {
            panic!("expected physical expression plan");
        };
        let PlanNode::Let { body, .. } = plan.node(*root) else {
            panic!("expected let root");
        };
        let PlanNode::Pipeline { body, .. } = plan.node(*body) else {
            panic!("expected pipeline body");
        };
        assert!(matches!(
            body.stage_kernels.first(),
            Some(BodyKernel::Generic)
        ));

        let j = Jetro::from_bytes(
            br#"{"books":[{"items":[{"label":"row-a"}]},{"items":[{"label":"row-b"}]}]}"#.to_vec(),
        )
        .unwrap();
        let out = super::collect_plan_json(&j, &plan).unwrap();

        assert_eq!(out, json!([["fixed"], ["fixed"]]));
        assert!(!j.root_val_is_materialized());
    }
    #[test]
    fn byte_native_let_keeps_locals_visible_without_root_materialization() {
        let expr =
            r#"let min_score = 900 in {"root": $.meta.version, "min": min_score, min_score}"#;
        let plan = planner::plan_query_with_context(expr, planner::PlanningContext::bytes());
        assert!(plan.root_execution_facts().is_byte_native());

        let j = Jetro::from_bytes(br#"{"meta":{"version":7}}"#.to_vec()).unwrap();

        let out = super::collect_plan_json(&j, &plan).unwrap();

        assert_eq!(out, json!({"root": 7, "min": 900, "min_score": 900}));
        assert!(!j.root_val_is_materialized());
    }

    #[test]
    fn let_bound_receiver_chain_executes_as_pipeline_source() {
        let expr = r#"let books = $.books in books.filter(score > 900).take(2).map(title)"#;
        let plan = planner::plan_query(expr);
        let QueryRoot::Node(root) = plan.root() else {
            panic!("expected physical expression plan");
        };
        let PlanNode::Let { body, .. } = plan.node(*root) else {
            panic!("expected let root");
        };
        let PlanNode::Pipeline {
            source: PipelinePlanSource::Expr(source),
            body,
        } = plan.node(*body)
        else {
            panic!("expected receiver pipeline source");
        };
        assert!(matches!(plan.node(*source), PlanNode::Local(name) if name.as_ref() == "books"));
        assert_eq!(body.stages.len(), 3);

        let j = Jetro::from(json!({
            "books": [
                {"title": "low", "score": 1},
                {"title": "a", "score": 901},
                {"title": "b", "score": 902},
                {"title": "c", "score": 903}
            ]
        }));

        let out = j.collect(expr).unwrap();

        assert_eq!(out, json!(["a", "b"]));
    }

    #[test]
    fn object_shape_executes_receiver_pipeline_children() {
        let expr = r#"let books = $.books in {"top": books.filter(score > 900).take(2).map(title), "first": books.filter(score > 900).first()}"#;
        let plan = planner::plan_query(expr);
        let QueryRoot::Node(root) = plan.root() else {
            panic!("expected physical expression plan");
        };
        let PlanNode::Let { body, .. } = plan.node(*root) else {
            panic!("expected let root");
        };
        let PlanNode::Object(fields) = plan.node(*body) else {
            panic!("expected object body");
        };
        for idx in [0usize, 1] {
            let PhysicalObjField::Kv { val, .. } = &fields[idx] else {
                panic!("expected kv field");
            };
            assert!(matches!(plan.node(*val), PlanNode::Pipeline { .. }));
        }

        let j = Jetro::from(json!({
            "books": [
                {"title": "low", "score": 1},
                {"title": "a", "score": 901},
                {"title": "b", "score": 902},
                {"title": "c", "score": 903}
            ]
        }));

        let out = j.collect(expr).unwrap();

        assert_eq!(
            out,
            json!({
                "top": ["a", "b"],
                "first": {"title": "a", "score": 901}
            })
        );
    }

    #[test]
    fn array_shape_executes_receiver_pipeline_children() {
        let expr = r#"let books = $.books in [books.filter(score > 900).take(2).map(title), books.filter(score > 900).first()]"#;
        let plan = planner::plan_query(expr);
        let QueryRoot::Node(root) = plan.root() else {
            panic!("expected physical expression plan");
        };
        let PlanNode::Let { body, .. } = plan.node(*root) else {
            panic!("expected let root");
        };
        let PlanNode::Array(elems) = plan.node(*body) else {
            panic!("expected array body");
        };
        for idx in [0usize, 1] {
            let PhysicalArrayElem::Expr(val) = &elems[idx] else {
                panic!("expected array expr");
            };
            assert!(matches!(plan.node(*val), PlanNode::Pipeline { .. }));
        }

        let j = Jetro::from(json!({
            "books": [
                {"title": "low", "score": 1},
                {"title": "a", "score": 901},
                {"title": "b", "score": 902},
                {"title": "c", "score": 903}
            ]
        }));

        let out = j.collect(expr).unwrap();

        assert_eq!(
            out,
            json!([
                ["a", "b"],
                {"title": "a", "score": 901}
            ])
        );
    }
}
