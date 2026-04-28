//! Top-level query execution routing.
//!
//! `planner.rs` classifies the expression. This module owns the runtime
//! decision flow for a `Jetro` handle: pipeline first when planned, scalar VM
//! fallback otherwise. Keeping this out of `lib.rs` makes data flow explicit
//! without changing the physical loops in `pipeline.rs` or `vm.rs`.

use serde_json::Value;

use crate::context::EvalError;
use crate::physical_eval;
use crate::pipeline;
use crate::planner;
use crate::value::Val;
use crate::{with_vm, Jetro, VM};

pub(crate) fn collect_json(j: &Jetro, expr: &str) -> Result<Value, EvalError> {
    let plan = planner::plan_query(expr);

    match &plan {
        planner::ExecutionPlan::Pipeline(_) => {
            if let Some(out) = run_pipeline(j, &plan) {
                return out.map(Value::from);
            }
        }
        planner::ExecutionPlan::Expr(expr) => {
            return physical_eval::run(j, expr).map(Value::from);
        }
        planner::ExecutionPlan::Vm => {}
    }

    run_vm_json(j, expr)
}

fn run_pipeline(j: &Jetro, plan: &planner::ExecutionPlan) -> Option<Result<Val, EvalError>> {
    let p = plan.pipeline()?;
    Some(p.run_with(&j.root_val(), Some(j as &dyn pipeline::PipelineData)))
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

    use crate::physical::{PhysicalArrayElem, PhysicalChainStep, PhysicalExpr, PhysicalObjField};
    use crate::planner::{self, ExecutionPlan};
    use crate::Jetro;

    fn assert_no_vm_fallback(expr: &PhysicalExpr) {
        match expr {
            PhysicalExpr::Vm(_) => panic!("unexpected VM fallback in physical plan"),
            PhysicalExpr::Literal(_)
            | PhysicalExpr::Root
            | PhysicalExpr::Current
            | PhysicalExpr::Ident(_)
            | PhysicalExpr::Pipeline(_)
            | PhysicalExpr::RootPath(_) => {}
            PhysicalExpr::Chain { base, steps } => {
                assert_no_vm_fallback(base);
                for step in steps {
                    if let PhysicalChainStep::DynIndex(expr) = step {
                        assert_no_vm_fallback(expr);
                    }
                }
            }
            PhysicalExpr::UnaryNeg(inner) | PhysicalExpr::Not(inner) => {
                assert_no_vm_fallback(inner)
            }
            PhysicalExpr::Binary { lhs, rhs, .. } => {
                assert_no_vm_fallback(lhs);
                assert_no_vm_fallback(rhs);
            }
            PhysicalExpr::Kind { expr, .. } => assert_no_vm_fallback(expr),
            PhysicalExpr::Coalesce { lhs, rhs } => {
                assert_no_vm_fallback(lhs);
                assert_no_vm_fallback(rhs);
            }
            PhysicalExpr::IfElse { cond, then_, else_ } => {
                assert_no_vm_fallback(cond);
                assert_no_vm_fallback(then_);
                assert_no_vm_fallback(else_);
            }
            PhysicalExpr::Try { body, default } => {
                assert_no_vm_fallback(body);
                assert_no_vm_fallback(default);
            }
            PhysicalExpr::Object(fields) => {
                for field in fields {
                    match field {
                        PhysicalObjField::Kv { val, cond, .. } => {
                            assert_no_vm_fallback(val);
                            if let Some(cond) = cond {
                                assert_no_vm_fallback(cond);
                            }
                        }
                        PhysicalObjField::Short(_) => {}
                        PhysicalObjField::Dynamic { key, val } => {
                            assert_no_vm_fallback(key);
                            assert_no_vm_fallback(val);
                        }
                        PhysicalObjField::Spread(expr) | PhysicalObjField::SpreadDeep(expr) => {
                            assert_no_vm_fallback(expr);
                        }
                    }
                }
            }
            PhysicalExpr::Array(elems) => {
                for elem in elems {
                    match elem {
                        PhysicalArrayElem::Expr(expr) | PhysicalArrayElem::Spread(expr) => {
                            assert_no_vm_fallback(expr);
                        }
                    }
                }
            }
            PhysicalExpr::Let { init, body, .. } => {
                assert_no_vm_fallback(init);
                assert_no_vm_fallback(body);
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
        let ExecutionPlan::Expr(plan) = planner::plan_query(expr) else {
            panic!("expected physical expression plan");
        };
        assert_no_vm_fallback(&plan);

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
        let ExecutionPlan::Expr(plan) = planner::plan_query(expr) else {
            panic!("expected physical expression plan");
        };
        assert_no_vm_fallback(&plan);

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
}
