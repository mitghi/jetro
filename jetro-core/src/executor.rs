//! Top-level query execution routing.
//!
//! `planner.rs` builds a single-use `QueryPlan`. This module only chooses
//! between a parsed plan root and the source-level VM fallback used when
//! parsing fails.

use serde_json::Value;

use crate::context::EvalError;
use crate::physical::{QueryPlan, QueryRoot};
use crate::physical_eval;
use crate::planner;
use crate::{with_vm, Jetro, VM};

pub(crate) fn collect_json(j: &Jetro, expr: &str) -> Result<Value, EvalError> {
    let plan = planner::plan_query(expr);
    collect_plan_json(j, &plan)
}

pub(crate) fn collect_plan_json(j: &Jetro, plan: &QueryPlan) -> Result<Value, EvalError> {
    match plan.root() {
        QueryRoot::Node(root) => physical_eval::run(j, plan, *root).map(Value::from),
        QueryRoot::SourceVm(source) => run_vm_json(j, source.as_ref()),
    }
}

fn run_vm_json(j: &Jetro, expr: &str) -> Result<Value, EvalError> {
    with_vm(|cell| match cell.try_borrow_mut() {
        Ok(mut vm) => {
            let prog = vm.get_or_compile(expr)?;
            vm.execute_val(&prog, j.root_val()?)
        }
        Err(_) => {
            let mut vm = VM::new();
            let prog = vm.get_or_compile(expr)?;
            vm.execute_val(&prog, j.root_val()?)
        }
    })
}

pub(crate) fn collect_plan_json_with_vm(
    j: &Jetro,
    plan: &QueryPlan,
    vm: &mut VM,
) -> Result<Value, EvalError> {
    match plan.root() {
        QueryRoot::Node(root) => physical_eval::run(j, plan, *root).map(Value::from),
        QueryRoot::SourceVm(source) => {
            let prog = vm.get_or_compile(source.as_ref())?;
            vm.execute_val(&prog, j.root_val()?)
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::physical::QueryRoot;
    use crate::physical::{
        NodeId, PhysicalArrayElem, PhysicalChainStep, PhysicalObjField, PipelinePlanSource,
        PlanNode,
    };
    use crate::pipeline::{NumOp, ReducerOp, Sink, Stage};
    use crate::planner;
    use crate::value::Val;
    use crate::{Jetro, JetroEngine};

    fn assert_no_vm_fallback(plan: &crate::physical::QueryPlan, id: NodeId) {
        match plan.node(id) {
            PlanNode::Vm(_) => panic!("unexpected VM fallback in physical plan"),
            PlanNode::Literal(_)
            | PlanNode::Root
            | PlanNode::Current
            | PlanNode::Ident(_)
            | PlanNode::RootPath(_)
            | PlanNode::Structural(_) => {}
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
        }
    }

    fn collect_test_val(j: &Jetro, expr: &str) -> Val {
        let plan = planner::plan_query(expr);
        match plan.root() {
            QueryRoot::Node(root) => crate::physical_eval::run(j, &plan, *root).unwrap(),
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
            PipelinePlanSource::Expr(_) => panic!("expected $.data field-chain source"),
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
            PipelinePlanSource::Expr(_) => panic!("expected $.data field-chain source"),
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

    #[cfg(feature = "simd-json")]
    #[test]
    fn object_shape_root_paths_read_from_tape_without_materializing_root_val() {
        let j =
            Jetro::from_bytes(br#"{"a":{"b":[{"c":"ok"},{"c":"next"}]},"n":7}"#.to_vec()).unwrap();

        let out = j.collect(r#"{"value": $.a.b[1].c, "n": $.n}"#).unwrap();

        assert_eq!(out, json!({"value": "next", "n": 7}));
        assert!(!j.root_val_is_materialized());
    }

    #[cfg(feature = "simd-json")]
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

    #[cfg(feature = "simd-json")]
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

    #[cfg(feature = "simd-json")]
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

    #[cfg(feature = "simd-json")]
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

    #[cfg(feature = "simd-json")]
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

    #[cfg(feature = "simd-json")]
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

    #[cfg(feature = "simd-json")]
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

    #[cfg(feature = "simd-json")]
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

    #[cfg(feature = "simd-json")]
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

    #[cfg(feature = "simd-json")]
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

    #[cfg(feature = "simd-json")]
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

    #[cfg(feature = "simd-json")]
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

    #[cfg(feature = "simd-json")]
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

    #[cfg(feature = "simd-json")]
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

    #[cfg(feature = "simd-json")]
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

    #[cfg(feature = "simd-json")]
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

    #[cfg(feature = "simd-json")]
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

    #[cfg(feature = "simd-json")]
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

    #[cfg(feature = "simd-json")]
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

    #[cfg(feature = "simd-json")]
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

    #[cfg(feature = "simd-json")]
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

    #[cfg(feature = "simd-json")]
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

    #[cfg(feature = "simd-json")]
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

    #[cfg(feature = "simd-json")]
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

    #[cfg(feature = "simd-json")]
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

    #[cfg(feature = "simd-json")]
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

    #[cfg(feature = "simd-json")]
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

    #[cfg(feature = "simd-json")]
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

    #[cfg(feature = "simd-json")]
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

    #[cfg(feature = "simd-json")]
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

    #[cfg(feature = "simd-json")]
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

    #[cfg(feature = "simd-json")]
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

    #[cfg(feature = "simd-json")]
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

    #[cfg(feature = "simd-json")]
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

    #[cfg(feature = "simd-json")]
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

    #[cfg(feature = "simd-json")]
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

    #[cfg(feature = "simd-json")]
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

    #[cfg(feature = "simd-json")]
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

    #[cfg(feature = "simd-json")]
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

    #[cfg(feature = "simd-json")]
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

    #[cfg(feature = "simd-json")]
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

    #[cfg(feature = "simd-json")]
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
        assert!(matches!(plan.node(*source), PlanNode::Ident(name) if name.as_ref() == "books"));
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
