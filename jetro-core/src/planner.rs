//! Query execution planner.
//!
//! This module is the explicit decision point between scalar VM execution,
//! Pipeline IR execution, and scalar VM fallback. The hot execution loops
//! remain in `vm.rs`, `pipeline.rs`, and `composed.rs`; the planner only
//! classifies a query so callers do not duplicate routing logic.

use std::sync::Arc;

use crate::ast::{ArrayElem, Expr, ObjField, Step};
use crate::builtins::BuiltinCall;
use crate::parser;
use crate::physical::{
    PhysicalArrayElem, PhysicalChainStep, PhysicalExpr, PhysicalObjField, PhysicalPathStep,
};
use crate::pipeline::Pipeline;
use crate::value::Val;
use crate::vm::Compiler;

/// Top-level physical execution choice for a query.
#[derive(Clone)]
pub enum ExecutionPlan {
    /// The expression lowered to Pipeline IR. Pipeline execution may still
    /// choose columnar or composed physical loops internally.
    Pipeline(Pipeline),
    /// A recursively planned expression tree. Individual nodes may still
    /// use Pipeline IR while unsupported leaves fall back to VM programs.
    Expr(PhysicalExpr),
    /// Fallback to the scalar bytecode VM.
    Vm,
}

impl ExecutionPlan {
    #[inline]
    pub fn pipeline(&self) -> Option<&Pipeline> {
        match self {
            ExecutionPlan::Pipeline(p) => Some(p),
            ExecutionPlan::Expr(_) | ExecutionPlan::Vm => None,
        }
    }
}

#[inline]
pub fn plan_expr(expr: &Expr) -> PhysicalExpr {
    try_lower_pipeline(expr)
        .or_else(|| try_lower_root_path(expr))
        .or_else(|| try_lower_chain(expr))
        .or_else(|| try_lower_scalar(expr))
        .or_else(|| try_lower_structural(expr))
        .unwrap_or_else(|| fallback_vm(expr))
}

fn try_lower_pipeline(expr: &Expr) -> Option<PhysicalExpr> {
    let pipeline = Pipeline::lower(expr)?;
    (!is_trivial_collect_pipeline(&pipeline)).then_some(PhysicalExpr::Pipeline(pipeline))
}

fn is_trivial_collect_pipeline(pipeline: &Pipeline) -> bool {
    pipeline.stages.is_empty() && matches!(pipeline.sink, crate::pipeline::Sink::Collect)
}

fn try_lower_root_path(expr: &Expr) -> Option<PhysicalExpr> {
    match expr {
        Expr::Root => Some(PhysicalExpr::RootPath(Vec::new())),
        Expr::Chain(base, steps) => {
            if !matches!(base.as_ref(), Expr::Root) {
                return None;
            }
            let mut out = Vec::with_capacity(steps.len());
            for step in steps {
                match step {
                    Step::Field(key) | Step::OptField(key) => {
                        out.push(PhysicalPathStep::Field(Arc::from(key.as_str())));
                    }
                    Step::Index(idx) => out.push(PhysicalPathStep::Index(*idx)),
                    _ => return None,
                }
            }
            Some(PhysicalExpr::RootPath(out))
        }
        _ => None,
    }
}

fn try_lower_chain(expr: &Expr) -> Option<PhysicalExpr> {
    let Expr::Chain(base, steps) = expr else {
        return None;
    };

    let mut out = Vec::with_capacity(steps.len());
    for step in steps {
        match step {
            Step::Field(key) | Step::OptField(key) => {
                out.push(PhysicalChainStep::Field(Arc::from(key.as_str())));
            }
            Step::Index(idx) => out.push(PhysicalChainStep::Index(*idx)),
            Step::DynIndex(expr) => out.push(PhysicalChainStep::DynIndex(plan_expr(expr))),
            Step::Method(name, args) => {
                let call = BuiltinCall::from_literal_ast_args(name, args)?;
                out.push(PhysicalChainStep::Method {
                    call,
                    optional: false,
                });
            }
            Step::OptMethod(name, args) => {
                let call = BuiltinCall::from_literal_ast_args(name, args)?;
                out.push(PhysicalChainStep::Method {
                    call,
                    optional: true,
                });
            }
            _ => return None,
        }
    }

    Some(PhysicalExpr::Chain {
        base: Box::new(plan_expr(base)),
        steps: out,
    })
}

fn try_lower_scalar(expr: &Expr) -> Option<PhysicalExpr> {
    match expr {
        Expr::Null => Some(PhysicalExpr::Literal(Val::Null)),
        Expr::Bool(b) => Some(PhysicalExpr::Literal(Val::Bool(*b))),
        Expr::Int(n) => Some(PhysicalExpr::Literal(Val::Int(*n))),
        Expr::Float(f) => Some(PhysicalExpr::Literal(Val::Float(*f))),
        Expr::Str(s) => Some(PhysicalExpr::Literal(Val::Str(Arc::from(s.as_str())))),
        Expr::Root => Some(PhysicalExpr::Root),
        Expr::Current => Some(PhysicalExpr::Current),
        Expr::Ident(name) => Some(PhysicalExpr::Ident(Arc::from(name.as_str()))),
        Expr::UnaryNeg(inner) => Some(PhysicalExpr::UnaryNeg(Box::new(plan_expr(inner)))),
        Expr::Not(inner) => Some(PhysicalExpr::Not(Box::new(plan_expr(inner)))),
        Expr::BinOp(lhs, op, rhs) => Some(PhysicalExpr::Binary {
            lhs: Box::new(plan_expr(lhs)),
            op: *op,
            rhs: Box::new(plan_expr(rhs)),
        }),
        Expr::Kind { expr, ty, negate } => Some(PhysicalExpr::Kind {
            expr: Box::new(plan_expr(expr)),
            ty: *ty,
            negate: *negate,
        }),
        Expr::Coalesce(lhs, rhs) => Some(PhysicalExpr::Coalesce {
            lhs: Box::new(plan_expr(lhs)),
            rhs: Box::new(plan_expr(rhs)),
        }),
        Expr::IfElse { cond, then_, else_ } => Some(PhysicalExpr::IfElse {
            cond: Box::new(plan_expr(cond)),
            then_: Box::new(plan_expr(then_)),
            else_: Box::new(plan_expr(else_)),
        }),
        Expr::Try { body, default } => Some(PhysicalExpr::Try {
            body: Box::new(plan_expr(body)),
            default: Box::new(plan_expr(default)),
        }),
        _ => None,
    }
}

fn try_lower_structural(expr: &Expr) -> Option<PhysicalExpr> {
    match expr {
        Expr::Object(fields) => Some(PhysicalExpr::Object(
            fields.iter().map(plan_obj_field).collect(),
        )),
        Expr::Array(elems) => Some(PhysicalExpr::Array(
            elems.iter().map(plan_array_elem).collect(),
        )),
        Expr::Let { name, init, body } => Some(PhysicalExpr::Let {
            name: Arc::from(name.as_str()),
            init: Box::new(plan_expr(init)),
            body: Box::new(plan_expr(body)),
        }),
        _ => None,
    }
}

fn fallback_vm(expr: &Expr) -> PhysicalExpr {
    PhysicalExpr::Vm(Arc::new(Compiler::compile(expr, "<planned-expr>")))
}

fn plan_obj_field(field: &ObjField) -> PhysicalObjField {
    match field {
        ObjField::Kv {
            key,
            val,
            optional,
            cond,
        } => PhysicalObjField::Kv {
            key: Arc::from(key.as_str()),
            val: plan_expr(val),
            optional: *optional,
            cond: cond.as_ref().map(plan_expr),
        },
        ObjField::Short(name) => PhysicalObjField::Short(Arc::from(name.as_str())),
        ObjField::Dynamic { key, val } => PhysicalObjField::Dynamic {
            key: plan_expr(key),
            val: plan_expr(val),
        },
        ObjField::Spread(expr) => PhysicalObjField::Spread(plan_expr(expr)),
        ObjField::SpreadDeep(expr) => PhysicalObjField::SpreadDeep(plan_expr(expr)),
    }
}

fn plan_array_elem(elem: &ArrayElem) -> PhysicalArrayElem {
    match elem {
        ArrayElem::Expr(expr) => PhysicalArrayElem::Expr(plan_expr(expr)),
        ArrayElem::Spread(expr) => PhysicalArrayElem::Spread(plan_expr(expr)),
    }
}

/// Parse and classify a query once.
#[inline]
pub fn plan_query(expr: &str) -> ExecutionPlan {
    let Ok(ast) = parser::parse(expr) else {
        return ExecutionPlan::Vm;
    };
    if let Some(pipeline) = Pipeline::lower(&ast) {
        return ExecutionPlan::Pipeline(pipeline);
    }
    match &ast {
        Expr::Object(_) | Expr::Array(_) | Expr::Let { .. } => ExecutionPlan::Expr(plan_expr(&ast)),
        _ => ExecutionPlan::Vm,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::BinOp;
    use crate::physical::{PhysicalExpr, PhysicalObjField};

    #[test]
    fn object_shape_keeps_pipeline_children() {
        let plan = plan_query(r#"{"a": $.books.filter(price > 10).map(id), "b": $.test}"#);
        let ExecutionPlan::Expr(PhysicalExpr::Object(fields)) = plan else {
            panic!("expected physical object plan");
        };
        assert_eq!(fields.len(), 2);
        match &fields[0] {
            PhysicalObjField::Kv { val, .. } => {
                assert!(matches!(val, PhysicalExpr::Pipeline(_)));
            }
            _ => panic!("expected kv field"),
        }
        assert!(matches!(
            &fields[1],
            PhysicalObjField::Kv {
                val: PhysicalExpr::RootPath(_),
                ..
            }
        ));
    }

    #[test]
    fn object_shape_uses_scalar_root_path_for_simple_field_chain() {
        let plan = plan_query(r#"{"b": $.a.b[0]}"#);
        let ExecutionPlan::Expr(PhysicalExpr::Object(fields)) = plan else {
            panic!("expected physical object plan");
        };
        assert!(matches!(
            &fields[0],
            PhysicalObjField::Kv {
                val: PhysicalExpr::RootPath(_),
                ..
            }
        ));
    }

    #[test]
    fn object_shape_plans_common_scalar_nodes_without_vm() {
        let plan = plan_query(r#"{"b": $.a > 1, "c": "x", "d": true if $.ok else false}"#);
        let ExecutionPlan::Expr(PhysicalExpr::Object(fields)) = plan else {
            panic!("expected physical object plan");
        };
        assert!(matches!(
            &fields[0],
            PhysicalObjField::Kv {
                val: PhysicalExpr::Binary { op: BinOp::Gt, .. },
                ..
            }
        ));
        assert!(matches!(
            &fields[1],
            PhysicalObjField::Kv {
                val: PhysicalExpr::Literal(_),
                ..
            }
        ));
        assert!(matches!(
            &fields[2],
            PhysicalObjField::Kv {
                val: PhysicalExpr::IfElse { .. },
                ..
            }
        ));
    }
}
