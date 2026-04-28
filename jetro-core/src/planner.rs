//! Query execution planner.
//!
//! This module lowers a parsed expression into one lightweight, single-use
//! `QueryPlan`. Pipeline and VM are backend nodes inside that plan, not
//! separate top-level execution modes.

use std::sync::Arc;

use crate::ast::{ArrayElem, Expr, ObjField, Step};
use crate::builtins::BuiltinCall;
use crate::parser;
use crate::physical::{
    NodeId, PhysicalArrayElem, PhysicalChainStep, PhysicalObjField, PhysicalPathStep, PlanNode,
    QueryPlan,
};
use crate::pipeline::Pipeline;
use crate::value::Val;
use crate::vm::Compiler;

#[derive(Default)]
struct PlanBuilder {
    nodes: Vec<PlanNode>,
}

impl PlanBuilder {
    #[inline]
    fn finish(self, root: NodeId) -> QueryPlan {
        QueryPlan::from_nodes(root, self.nodes)
    }

    #[inline]
    fn push(&mut self, node: PlanNode) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(node);
        id
    }
}

#[inline]
fn lower_expr(builder: &mut PlanBuilder, expr: &Expr) -> NodeId {
    try_lower_pipeline(expr)
        .map(|node| builder.push(node))
        .or_else(|| try_lower_root_path(expr).map(|node| builder.push(node)))
        .or_else(|| try_lower_chain(builder, expr))
        .or_else(|| try_lower_scalar(builder, expr))
        .or_else(|| try_lower_structural(builder, expr))
        .unwrap_or_else(|| fallback_vm(builder, expr))
}

fn try_lower_pipeline(expr: &Expr) -> Option<PlanNode> {
    let pipeline = Pipeline::lower(expr)?;
    (!is_trivial_collect_pipeline(&pipeline)).then_some(PlanNode::Pipeline(pipeline))
}

fn is_trivial_collect_pipeline(pipeline: &Pipeline) -> bool {
    pipeline.stages.is_empty() && matches!(pipeline.sink, crate::pipeline::Sink::Collect)
}

fn try_lower_root_path(expr: &Expr) -> Option<PlanNode> {
    match expr {
        Expr::Root => Some(PlanNode::RootPath(Vec::new())),
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
            Some(PlanNode::RootPath(out))
        }
        _ => None,
    }
}

fn try_lower_chain(builder: &mut PlanBuilder, expr: &Expr) -> Option<NodeId> {
    let Expr::Chain(base, steps) = expr else {
        return None;
    };

    let mut cur = lower_expr(builder, base);
    let mut out = Vec::new();
    for step in steps {
        match step {
            Step::Field(key) | Step::OptField(key) => {
                out.push(PhysicalChainStep::Field(Arc::from(key.as_str())));
            }
            Step::Index(idx) => out.push(PhysicalChainStep::Index(*idx)),
            Step::DynIndex(expr) => {
                out.push(PhysicalChainStep::DynIndex(lower_expr(builder, expr)))
            }
            Step::Method(name, args) => {
                let call = BuiltinCall::from_literal_ast_args(name, args)?;
                cur = flush_chain(builder, cur, &mut out);
                cur = builder.push(PlanNode::Call {
                    receiver: cur,
                    call,
                    optional: false,
                });
            }
            Step::OptMethod(name, args) => {
                let call = BuiltinCall::from_literal_ast_args(name, args)?;
                cur = flush_chain(builder, cur, &mut out);
                cur = builder.push(PlanNode::Call {
                    receiver: cur,
                    call,
                    optional: true,
                });
            }
            _ => return None,
        }
    }

    Some(flush_chain(builder, cur, &mut out))
}

fn flush_chain(
    builder: &mut PlanBuilder,
    base: NodeId,
    steps: &mut Vec<PhysicalChainStep>,
) -> NodeId {
    if steps.is_empty() {
        return base;
    }
    builder.push(PlanNode::Chain {
        base,
        steps: std::mem::take(steps),
    })
}

fn try_lower_scalar(builder: &mut PlanBuilder, expr: &Expr) -> Option<NodeId> {
    match expr {
        Expr::Null => Some(builder.push(PlanNode::Literal(Val::Null))),
        Expr::Bool(b) => Some(builder.push(PlanNode::Literal(Val::Bool(*b)))),
        Expr::Int(n) => Some(builder.push(PlanNode::Literal(Val::Int(*n)))),
        Expr::Float(f) => Some(builder.push(PlanNode::Literal(Val::Float(*f)))),
        Expr::Str(s) => Some(builder.push(PlanNode::Literal(Val::Str(Arc::from(s.as_str()))))),
        Expr::Root => Some(builder.push(PlanNode::Root)),
        Expr::Current => Some(builder.push(PlanNode::Current)),
        Expr::Ident(name) => Some(builder.push(PlanNode::Ident(Arc::from(name.as_str())))),
        Expr::UnaryNeg(inner) => {
            let inner = lower_expr(builder, inner);
            Some(builder.push(PlanNode::UnaryNeg(inner)))
        }
        Expr::Not(inner) => {
            let inner = lower_expr(builder, inner);
            Some(builder.push(PlanNode::Not(inner)))
        }
        Expr::BinOp(lhs, op, rhs) => {
            let lhs = lower_expr(builder, lhs);
            let rhs = lower_expr(builder, rhs);
            Some(builder.push(PlanNode::Binary { lhs, op: *op, rhs }))
        }
        Expr::Kind { expr, ty, negate } => {
            let expr = lower_expr(builder, expr);
            Some(builder.push(PlanNode::Kind {
                expr,
                ty: *ty,
                negate: *negate,
            }))
        }
        Expr::Coalesce(lhs, rhs) => {
            let lhs = lower_expr(builder, lhs);
            let rhs = lower_expr(builder, rhs);
            Some(builder.push(PlanNode::Coalesce { lhs, rhs }))
        }
        Expr::IfElse { cond, then_, else_ } => {
            let cond = lower_expr(builder, cond);
            let then_ = lower_expr(builder, then_);
            let else_ = lower_expr(builder, else_);
            Some(builder.push(PlanNode::IfElse { cond, then_, else_ }))
        }
        Expr::Try { body, default } => {
            let body = lower_expr(builder, body);
            let default = lower_expr(builder, default);
            Some(builder.push(PlanNode::Try { body, default }))
        }
        _ => None,
    }
}

fn try_lower_structural(builder: &mut PlanBuilder, expr: &Expr) -> Option<NodeId> {
    match expr {
        Expr::Object(fields) => {
            let fields = fields
                .iter()
                .map(|field| plan_obj_field(builder, field))
                .collect();
            Some(builder.push(PlanNode::Object(fields)))
        }
        Expr::Array(elems) => {
            let elems = elems
                .iter()
                .map(|elem| plan_array_elem(builder, elem))
                .collect();
            Some(builder.push(PlanNode::Array(elems)))
        }
        Expr::Let { name, init, body } => {
            let init = lower_expr(builder, init);
            let body = lower_expr(builder, body);
            Some(builder.push(PlanNode::Let {
                name: Arc::from(name.as_str()),
                init,
                body,
            }))
        }
        _ => None,
    }
}

fn fallback_vm(builder: &mut PlanBuilder, expr: &Expr) -> NodeId {
    builder.push(PlanNode::Vm(Arc::new(Compiler::compile(
        expr,
        "<planned-expr>",
    ))))
}

fn plan_obj_field(builder: &mut PlanBuilder, field: &ObjField) -> PhysicalObjField {
    match field {
        ObjField::Kv {
            key,
            val,
            optional,
            cond,
        } => PhysicalObjField::Kv {
            key: Arc::from(key.as_str()),
            val: lower_expr(builder, val),
            optional: *optional,
            cond: cond.as_ref().map(|cond| lower_expr(builder, cond)),
        },
        ObjField::Short(name) => PhysicalObjField::Short(Arc::from(name.as_str())),
        ObjField::Dynamic { key, val } => PhysicalObjField::Dynamic {
            key: lower_expr(builder, key),
            val: lower_expr(builder, val),
        },
        ObjField::Spread(expr) => PhysicalObjField::Spread(lower_expr(builder, expr)),
        ObjField::SpreadDeep(expr) => PhysicalObjField::SpreadDeep(lower_expr(builder, expr)),
    }
}

fn plan_array_elem(builder: &mut PlanBuilder, elem: &ArrayElem) -> PhysicalArrayElem {
    match elem {
        ArrayElem::Expr(expr) => PhysicalArrayElem::Expr(lower_expr(builder, expr)),
        ArrayElem::Spread(expr) => PhysicalArrayElem::Spread(lower_expr(builder, expr)),
    }
}

/// Parse and classify a query once.
#[inline]
pub fn plan_query(expr: &str) -> QueryPlan {
    let Ok(ast) = parser::parse(expr) else {
        return QueryPlan::source_vm(expr);
    };
    let mut builder = PlanBuilder::default();
    if let Some(pipeline) = Pipeline::lower(&ast) {
        let root = builder.push(PlanNode::Pipeline(pipeline));
        return builder.finish(root);
    }
    let root = match &ast {
        Expr::Object(_) | Expr::Array(_) | Expr::Let { .. } => lower_expr(&mut builder, &ast),
        _ => fallback_vm(&mut builder, &ast),
    };
    builder.finish(root)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::BinOp;
    use crate::physical::{PhysicalObjField, PlanNode, QueryRoot};

    fn root_node(plan: &QueryPlan) -> &PlanNode {
        let QueryRoot::Node(root) = plan.root() else {
            panic!("expected physical plan");
        };
        plan.node(*root)
    }

    #[test]
    fn object_shape_keeps_pipeline_children() {
        let plan = plan_query(r#"{"a": $.books.filter(price > 10).map(id), "b": $.test}"#);
        let PlanNode::Object(fields) = root_node(&plan) else {
            panic!("expected physical object plan");
        };
        assert_eq!(fields.len(), 2);
        match &fields[0] {
            PhysicalObjField::Kv { val, .. } => {
                assert!(matches!(plan.node(*val), PlanNode::Pipeline(_)));
            }
            _ => panic!("expected kv field"),
        }
        let PhysicalObjField::Kv { val, .. } = &fields[1] else {
            panic!("expected kv field");
        };
        assert!(matches!(plan.node(*val), PlanNode::RootPath(_)));
    }

    #[test]
    fn object_shape_uses_scalar_root_path_for_simple_field_chain() {
        let plan = plan_query(r#"{"b": $.a.b[0]}"#);
        let PlanNode::Object(fields) = root_node(&plan) else {
            panic!("expected physical object plan");
        };
        let PhysicalObjField::Kv { val, .. } = &fields[0] else {
            panic!("expected kv field");
        };
        assert!(matches!(plan.node(*val), PlanNode::RootPath(_)));
    }

    #[test]
    fn object_shape_plans_common_scalar_nodes_without_vm() {
        let plan = plan_query(r#"{"b": $.a > 1, "c": "x", "d": true if $.ok else false}"#);
        let PlanNode::Object(fields) = root_node(&plan) else {
            panic!("expected physical object plan");
        };
        let PhysicalObjField::Kv { val, .. } = &fields[0] else {
            panic!("expected kv field");
        };
        assert!(matches!(
            plan.node(*val),
            PlanNode::Binary { op: BinOp::Gt, .. }
        ));
        let PhysicalObjField::Kv { val, .. } = &fields[1] else {
            panic!("expected kv field");
        };
        assert!(matches!(plan.node(*val), PlanNode::Literal(_)));
        let PhysicalObjField::Kv { val, .. } = &fields[2] else {
            panic!("expected kv field");
        };
        assert!(matches!(plan.node(*val), PlanNode::IfElse { .. }));
    }

    #[test]
    fn method_chain_lowers_builtin_methods_to_call_nodes() {
        let plan =
            plan_query(r#"let user = {"name": " ada "} in {"name": user.name.upper().trim()}"#);
        let PlanNode::Let { body, .. } = root_node(&plan) else {
            panic!("expected let plan");
        };
        let PlanNode::Object(fields) = plan.node(*body) else {
            panic!("expected object body");
        };
        let PhysicalObjField::Kv { val, .. } = &fields[0] else {
            panic!("expected kv field");
        };
        let PlanNode::Call { receiver, .. } = plan.node(*val) else {
            panic!("expected trim call");
        };
        assert!(matches!(plan.node(*receiver), PlanNode::Call { .. }));
    }
}
