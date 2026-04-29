//! Top-level query execution routing.
//!
//! `planner.rs` builds a single-use `QueryPlan`. This module only chooses
//! between a parsed plan root and the source-level VM fallback used when
//! parsing fails.

use serde_json::Value;

use crate::context::EvalError;
use crate::physical::QueryRoot;
use crate::physical_eval;
use crate::planner;
use crate::{with_vm, Jetro, VM};

pub(crate) fn collect_json(j: &Jetro, expr: &str) -> Result<Value, EvalError> {
    let plan = planner::plan_query(expr);

    match plan.root() {
        QueryRoot::Node(root) => physical_eval::run(j, &plan, *root).map(Value::from),
        QueryRoot::SourceVm(source) => run_vm_json(j, source.as_ref()),
    }
}

fn run_vm_json(j: &Jetro, expr: &str) -> Result<Value, EvalError> {
    with_vm(|cell| match cell.try_borrow_mut() {
        Ok(mut vm) => {
            let prog = vm.get_or_compile(expr)?;
            vm.execute_val(&prog, j.root_val())
        }
        Err(_) => VM::new().run_str(expr, &j.document),
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::physical::QueryRoot;
    use crate::physical::{
        NodeId, PhysicalArrayElem, PhysicalChainStep, PhysicalObjField, PlanNode,
    };
    use crate::pipeline::{NumOp, Sink, Source, Stage};
    use crate::planner;
    use crate::Jetro;

    fn assert_no_vm_fallback(plan: &crate::physical::QueryPlan, id: NodeId) {
        match plan.node(id) {
            PlanNode::Vm(_) => panic!("unexpected VM fallback in physical plan"),
            PlanNode::Literal(_)
            | PlanNode::Root
            | PlanNode::Current
            | PlanNode::Ident(_)
            | PlanNode::Pipeline(_)
            | PlanNode::RootPath(_) => {}
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
            assert!(matches!(plan.node(*val), PlanNode::Pipeline(_)));
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
        let PlanNode::Pipeline(pipeline) = plan.node(*val) else {
            panic!("expected pipeline child");
        };

        match &pipeline.source {
            Source::FieldChain { keys } => {
                let keys: Vec<&str> = keys.iter().map(|k| k.as_ref()).collect();
                assert_eq!(keys, vec!["data"]);
            }
            Source::Receiver(_) => panic!("expected $.data field-chain source"),
        }
        assert_eq!(pipeline.stages.len(), 1);
        assert!(matches!(pipeline.stages[0], Stage::Filter(_)));
        assert!(
            matches!(&pipeline.sink, Sink::Numeric(n) if n.op == NumOp::Sum && n.project.is_some())
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
        let PlanNode::Pipeline(pipeline) = plan.node(*root) else {
            panic!("expected pipeline root");
        };

        match &pipeline.source {
            Source::FieldChain { keys } => {
                let keys: Vec<&str> = keys.iter().map(|k| k.as_ref()).collect();
                assert_eq!(keys, vec!["data"]);
            }
            Source::Receiver(_) => panic!("expected $.data field-chain source"),
        }
        assert_eq!(pipeline.stages.len(), 1);
        assert!(matches!(pipeline.stages[0], Stage::Filter(_)));
        assert!(
            matches!(&pipeline.sink, Sink::Numeric(n) if n.op == NumOp::Sum && n.project.is_some())
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
            assert!(matches!(plan.node(*val), PlanNode::Pipeline(_)));
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
}
