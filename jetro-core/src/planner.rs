//! Query execution planner.
//!
//! This module lowers a parsed expression into one lightweight, single-use
//! `QueryPlan`. Pipeline and VM are backend nodes inside that plan, not
//! separate top-level execution modes.

use std::sync::Arc;

use crate::analysis;
use crate::ast::{ArrayElem, Expr, ObjField, Step};
use crate::builtins::{BuiltinCall, BuiltinMethod};
use crate::parser;
use crate::physical::{
    BackendPlan, ExecutionFacts, NodeId, PhysicalArrayElem, PhysicalChainStep, PhysicalNode,
    PhysicalObjField, PhysicalPathStep, PipelinePlanSource, PlanNode, QueryPlan,
};
use crate::pipeline::{Pipeline, Source};
use crate::structural::{StructuralPathStep, StructuralPlan};
use crate::value::Val;
use crate::vm::Compiler;

#[derive(Default)]
struct PlanBuilder {
    nodes: Vec<PhysicalNode>,
    context: PlanningContext,
    locals: Vec<Arc<str>>,
}

impl PlanBuilder {
    #[inline]
    fn is_local(&self, name: &str) -> bool {
        self.locals.iter().rev().any(|local| local.as_ref() == name)
    }

    #[inline]
    fn push_local(&mut self, name: Arc<str>) {
        self.locals.push(name);
    }

    #[inline]
    fn pop_local(&mut self) {
        self.locals.pop();
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum InputMode {
    Bytes,
    Val,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PlanningContext {
    input: InputMode,
}

impl Default for PlanningContext {
    #[inline]
    fn default() -> Self {
        Self::bytes()
    }
}

impl PlanningContext {
    #[inline]
    pub(crate) const fn bytes() -> Self {
        Self {
            input: InputMode::Bytes,
        }
    }

    #[inline]
    pub(crate) const fn val() -> Self {
        Self {
            input: InputMode::Val,
        }
    }

    #[inline]
    pub(crate) const fn cache_key(self) -> &'static str {
        match self.input {
            InputMode::Bytes => "bytes",
            InputMode::Val => "val",
        }
    }
}

impl PlanBuilder {
    #[inline]
    fn finish(self, root: NodeId) -> QueryPlan {
        QueryPlan::from_physical_nodes(root, self.nodes)
    }

    #[inline]
    fn push(&mut self, node: PlanNode) -> NodeId {
        let facts = self.execution_facts_for_node(&node);
        let backends = self.backend_plan_for_node(&node, facts);
        let facts = adjust_facts_for_backend_plan(&node, backends, facts);
        self.push_with_backends_and_facts(node, backends, facts)
    }

    #[inline]
    fn push_with_backends_and_facts(
        &mut self,
        node: PlanNode,
        backends: BackendPlan,
        facts: ExecutionFacts,
    ) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(PhysicalNode::with_backend_plan_and_facts(
            node, backends, facts,
        ));
        id
    }

    #[inline]
    fn backend_plan_for_node(&self, node: &PlanNode, facts: ExecutionFacts) -> BackendPlan {
        select_backend_plan(self.context, node, facts)
    }

    #[inline]
    fn node_facts(&self, id: NodeId) -> ExecutionFacts {
        self.nodes[id.0].execution_facts()
    }

    fn execution_facts_for_node(&self, node: &PlanNode) -> ExecutionFacts {
        match node {
            PlanNode::Pipeline {
                source: PipelinePlanSource::Expr(source),
                body,
            } => {
                let local = ExecutionFacts::for_node(node);
                let source = self.node_facts(*source);
                let receiver_only = body.can_run_with_materialized_receiver();
                ExecutionFacts {
                    can_avoid_root_materialization: source.can_avoid_root_materialization
                        && !source.contains_vm_fallback
                        && receiver_only,
                    can_stream_rows: local.can_stream_rows || source.can_stream_rows,
                    can_use_tape: local.can_use_tape || source.can_use_tape,
                    contains_vm_fallback: source.contains_vm_fallback,
                    may_materialize_source: local.may_materialize_source
                        || source.may_materialize_source,
                }
            }
            PlanNode::Chain { base, steps } => {
                let children = std::iter::once(self.node_facts(*base)).chain(
                    steps.iter().filter_map(|step| match step {
                        PhysicalChainStep::DynIndex(id) => Some(self.node_facts(*id)),
                        _ => None,
                    }),
                );
                ExecutionFacts::combine_all(children)
            }
            PlanNode::Local(_) => ExecutionFacts::constant(),
            PlanNode::Call { receiver, .. }
            | PlanNode::UnaryNeg(receiver)
            | PlanNode::Not(receiver)
            | PlanNode::Kind { expr: receiver, .. } => {
                ExecutionFacts::combine_all([self.node_facts(*receiver)])
            }
            PlanNode::Binary { lhs, rhs, .. }
            | PlanNode::Coalesce { lhs, rhs }
            | PlanNode::Try {
                body: lhs,
                default: rhs,
            } => ExecutionFacts::combine_all([self.node_facts(*lhs), self.node_facts(*rhs)]),
            PlanNode::Let { init, body, .. } => {
                ExecutionFacts::combine_all([self.node_facts(*init), self.node_facts(*body)])
            }
            PlanNode::IfElse { cond, then_, else_ } => ExecutionFacts::combine_all([
                self.node_facts(*cond),
                self.node_facts(*then_),
                self.node_facts(*else_),
            ]),
            PlanNode::Object(fields) => {
                let children = fields.iter().flat_map(|field| match field {
                    PhysicalObjField::Kv { val, cond, .. } => {
                        let mut out = Vec::with_capacity(2);
                        if let Some(cond) = cond {
                            out.push(self.node_facts(*cond));
                        }
                        out.push(self.node_facts(*val));
                        out
                    }
                    PhysicalObjField::Dynamic { key, val } => {
                        vec![self.node_facts(*key), self.node_facts(*val)]
                    }
                    PhysicalObjField::Spread(id) | PhysicalObjField::SpreadDeep(id) => {
                        vec![self.node_facts(*id)]
                    }
                    PhysicalObjField::Short(_) => vec![ExecutionFacts::default()],
                });
                ExecutionFacts::combine_all(children)
            }
            PlanNode::Array(elems) => {
                let children = elems.iter().map(|elem| match elem {
                    PhysicalArrayElem::Expr(id) | PhysicalArrayElem::Spread(id) => {
                        self.node_facts(*id)
                    }
                });
                ExecutionFacts::combine_all(children)
            }
            _ => ExecutionFacts::for_node(node),
        }
    }
}

#[inline]
fn select_backend_plan(
    context: PlanningContext,
    node: &PlanNode,
    facts: ExecutionFacts,
) -> BackendPlan {
    match (context.input, node) {
        (
            InputMode::Val,
            PlanNode::Pipeline {
                source: PipelinePlanSource::FieldChain { .. },
                ..
            },
        ) => BackendPlan::new(&[
            crate::physical::BackendPreference::ValView,
            crate::physical::BackendPreference::Interpreted,
        ]),
        (InputMode::Val, PlanNode::RootPath(_) | PlanNode::Structural { .. }) => {
            BackendPlan::new(&[crate::physical::BackendPreference::Interpreted])
        }
        (InputMode::Bytes, PlanNode::Structural { .. }) => {
            BackendPlan::new(&[crate::physical::BackendPreference::Structural])
        }
        (
            InputMode::Bytes,
            PlanNode::Pipeline {
                source: PipelinePlanSource::FieldChain { .. },
                ..
            },
        ) if facts.can_stream_rows => BackendPlan::new(&[
            crate::physical::BackendPreference::TapeView,
            crate::physical::BackendPreference::TapeRows,
            crate::physical::BackendPreference::MaterializedSource,
            crate::physical::BackendPreference::ValView,
            crate::physical::BackendPreference::Interpreted,
        ])
        .without_interpreted_if(facts.is_byte_native()),
        _ if context.input == InputMode::Bytes
            && facts.is_byte_native()
            && !matches!(node, PlanNode::Pipeline { .. }) =>
        {
            BackendPlan::for_node(node).without_interpreted()
        }
        _ => BackendPlan::for_node(node),
    }
}

#[inline]
fn adjust_facts_for_backend_plan(
    node: &PlanNode,
    backends: BackendPlan,
    mut facts: ExecutionFacts,
) -> ExecutionFacts {
    if matches!(node, PlanNode::Structural { .. })
        && backends
            .as_slice()
            .contains(&crate::physical::BackendPreference::Structural)
    {
        facts.contains_vm_fallback = false;
    }
    facts
}

#[inline]
fn lower_expr(builder: &mut PlanBuilder, expr: &Expr) -> NodeId {
    try_lower_structural_op(expr)
        .map(|node| builder.push(node))
        .or_else(|| try_lower_pipeline(builder, expr).map(|node| builder.push(node)))
        .or_else(|| try_lower_root_path(expr).map(|node| builder.push(node)))
        .or_else(|| try_lower_receiver_pipeline(builder, expr))
        .or_else(|| try_lower_structural_chain_prefix(builder, expr))
        .or_else(|| try_lower_chain(builder, expr))
        .or_else(|| try_lower_scalar(builder, expr))
        .or_else(|| try_lower_structural(builder, expr))
        .unwrap_or_else(|| fallback_vm(builder, expr))
}

fn try_lower_pipeline(builder: &PlanBuilder, expr: &Expr) -> Option<PlanNode> {
    let pipeline = Pipeline::lower(expr)?;
    if is_trivial_collect_pipeline(&pipeline) {
        return None;
    }
    let (source, mut body) = pipeline.into_source_body();
    mask_active_local_stage_kernels(&mut body, builder);
    pipeline_parts_to_plan_node(source, body)
}

fn pipeline_parts_to_plan_node(
    source: Source,
    body: crate::pipeline::PipelineBody,
) -> Option<PlanNode> {
    let source = match source {
        Source::FieldChain { keys } => PipelinePlanSource::FieldChain { keys },
        Source::Receiver(_) => return None,
    };
    Some(PlanNode::Pipeline { source, body })
}

fn is_trivial_collect_pipeline(pipeline: &Pipeline) -> bool {
    pipeline.stages.is_empty() && matches!(pipeline.sink, crate::pipeline::Sink::Collect)
}

fn mask_active_local_stage_kernels(body: &mut crate::pipeline::PipelineBody, builder: &PlanBuilder) {
    if builder.locals.is_empty() || body.stage_exprs.len() != body.stage_kernels.len() {
        return;
    }
    for (expr, kernel) in body.stage_exprs.iter().zip(body.stage_kernels.iter_mut()) {
        let Some(expr) = expr else {
            continue;
        };
        if builder
            .locals
            .iter()
            .any(|local| analysis::expr_uses_ident(expr, local.as_ref()))
        {
            *kernel = crate::pipeline::BodyKernel::Generic;
        }
    }
}

fn try_lower_structural_op(expr: &Expr) -> Option<PlanNode> {
    let Expr::Chain(base, steps) = expr else {
        return None;
    };
    let (plan, fallback, consumed) = lower_structural_prefix(base, steps)?;
    if consumed == steps.len() {
        Some(PlanNode::Structural { plan, fallback })
    } else {
        None
    }
}

fn try_lower_structural_chain_prefix(builder: &mut PlanBuilder, expr: &Expr) -> Option<NodeId> {
    let Expr::Chain(base, steps) = expr else {
        return None;
    };
    let (plan, fallback, consumed) = lower_structural_prefix(base, steps)?;
    if consumed >= steps.len() {
        return None;
    }
    let mut cur = builder.push(PlanNode::Structural { plan, fallback });
    let mut out = Vec::new();
    for step in &steps[consumed..] {
        match step {
            Step::Field(key) | Step::OptField(key) => {
                out.push(PhysicalChainStep::Field(Arc::from(key.as_str())));
            }
            Step::Index(idx) => out.push(PhysicalChainStep::Index(*idx)),
            Step::DynIndex(expr) => {
                out.push(PhysicalChainStep::DynIndex(lower_expr(builder, expr)));
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

fn lower_structural_prefix(
    base: &Expr,
    steps: &[Step],
) -> Option<(StructuralPlan, Arc<crate::vm::Program>, usize)> {
    if !matches!(base, Expr::Root) {
        return None;
    }
    let mut anchor = Vec::new();
    for (idx, step) in steps.iter().enumerate() {
        match step {
            Step::Field(key) | Step::OptField(key) => {
                anchor.push(StructuralPathStep::Field(Arc::from(key.as_str())));
            }
            Step::Index(index) => anchor.push(StructuralPathStep::Index(*index)),
            Step::Method(name, args) | Step::OptMethod(name, args) => {
                let anchor = Arc::from(anchor);
                let method = BuiltinMethod::from_name(name);
                let plan = StructuralPlan::lower_builtin(anchor, method, args)?;
                let fallback_expr = base.clone().maybe_chain(steps[..=idx].to_vec());
                let fallback = Arc::new(Compiler::compile(&fallback_expr, "<structural-fallback>"));
                return Some((plan, fallback, idx + 1));
            }
            _ => return None,
        }
    }
    None
}

fn try_lower_receiver_pipeline(builder: &mut PlanBuilder, expr: &Expr) -> Option<NodeId> {
    let Expr::Chain(base, steps) = expr else {
        return None;
    };

    for method_start in steps
        .iter()
        .enumerate()
        .filter_map(|(idx, step)| Pipeline::is_receiver_pipeline_start(step).then_some(idx))
    {
        if matches!(base.as_ref(), Expr::Root) && method_start == 0 {
            continue;
        }
        let Some(mut body) = Pipeline::lower_body_from_steps(&steps[method_start..]) else {
            continue;
        };
        mask_active_local_stage_kernels(&mut body, builder);
        let source_expr = base
            .as_ref()
            .clone()
            .maybe_chain(steps[..method_start].to_vec());
        let source = lower_expr(builder, &source_expr);
        return Some(builder.push(PlanNode::Pipeline {
            source: PipelinePlanSource::Expr(source),
            body,
        }));
    }
    None
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
        Expr::Ident(name) if builder.is_local(name) => {
            Some(builder.push(PlanNode::Local(Arc::from(name.as_str()))))
        }
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
            let name = Arc::from(name.as_str());
            builder.push_local(Arc::clone(&name));
            let body = lower_expr(builder, body);
            builder.pop_local();
            Some(builder.push(PlanNode::Let { name, init, body }))
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
        ObjField::Short(name) if builder.is_local(name) => PhysicalObjField::Kv {
            key: Arc::from(name.as_str()),
            val: builder.push(PlanNode::Local(Arc::from(name.as_str()))),
            optional: false,
            cond: None,
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
    plan_query_with_context(expr, PlanningContext::default())
}

/// Parse and classify a query once with input-representation preferences.
#[inline]
pub(crate) fn plan_query_with_context(expr: &str, context: PlanningContext) -> QueryPlan {
    let Ok(ast) = parser::parse(expr) else {
        return QueryPlan::source_vm(expr);
    };
    let mut builder = PlanBuilder {
        nodes: Vec::new(),
        context,
        locals: Vec::new(),
    };
    if let Some(pipeline) = Pipeline::lower(&ast) {
        let (source, body) = pipeline.into_source_body();
        if let Some(node) = pipeline_parts_to_plan_node(source, body) {
            let root = builder.push(node);
            return builder.finish(root);
        }
    }
    let root = lower_expr(&mut builder, &ast);
    builder.finish(root)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::BinOp;
    use crate::physical::{
        BackendPreference, PhysicalObjField, PipelinePlanSource, PlanNode, QueryRoot,
    };

    fn root_node(plan: &QueryPlan) -> &PlanNode {
        let QueryRoot::Node(root) = plan.root() else {
            panic!("expected physical plan");
        };
        plan.node(*root)
    }

    #[test]
    fn deep_shape_lowers_to_structural_plan() {
        let plan = plan_query(r#"$.deep_shape({email})"#);
        assert!(matches!(root_node(&plan), PlanNode::Structural { .. }));
    }

    #[test]
    fn byte_context_prefers_tape_pipeline_backends() {
        let plan =
            plan_query_with_context(r#"$.rows.filter(score > 10)"#, PlanningContext::bytes());
        let QueryRoot::Node(root) = plan.root() else {
            panic!("expected physical plan");
        };
        assert_eq!(
            plan.backend_preferences(*root),
            &[
                BackendPreference::TapeView,
                BackendPreference::TapeRows,
                BackendPreference::MaterializedSource,
                BackendPreference::ValView,
            ]
        );
    }

    #[test]
    fn backend_policy_is_centralized_for_input_context() {
        let node = PlanNode::Pipeline {
            source: PipelinePlanSource::FieldChain {
                keys: Arc::from([Arc::<str>::from("rows")]),
            },
            body: crate::pipeline::PipelineBody {
                stages: Vec::new(),
                stage_exprs: Vec::new(),
                sink: crate::pipeline::Sink::Collect,
                stage_kernels: Vec::new(),
                sink_kernels: Vec::new(),
            },
        };

        assert_eq!(
            select_backend_plan(
                PlanningContext::val(),
                &node,
                ExecutionFacts::for_node(&node)
            )
            .as_slice(),
            &[BackendPreference::ValView, BackendPreference::Interpreted]
        );
        assert_eq!(
            select_backend_plan(
                PlanningContext::bytes(),
                &node,
                ExecutionFacts::for_node(&node)
            )
            .as_slice()[0],
            BackendPreference::TapeView
        );
    }

    #[test]
    fn val_context_prefers_val_pipeline_backend() {
        let plan = plan_query_with_context(r#"$.rows.filter(score > 10)"#, PlanningContext::val());
        let QueryRoot::Node(root) = plan.root() else {
            panic!("expected physical plan");
        };
        assert_eq!(
            plan.backend_preferences(*root),
            &[BackendPreference::ValView, BackendPreference::Interpreted]
        );
        assert!(plan
            .backend_capabilities(*root)
            .contains(crate::physical::BackendSet::TAPE_VIEW));
        assert!(plan
            .backend_capabilities(*root)
            .contains(crate::physical::BackendSet::VAL_VIEW));
    }

    #[test]
    fn val_context_avoids_tape_only_root_path_backend() {
        let plan = plan_query_with_context(r#"{"x": $.a.b}"#, PlanningContext::val());
        let PlanNode::Object(fields) = root_node(&plan) else {
            panic!("expected object plan");
        };
        let PhysicalObjField::Kv { val, .. } = &fields[0] else {
            panic!("expected kv field");
        };
        assert_eq!(
            plan.backend_preferences(*val),
            &[BackendPreference::Interpreted]
        );
    }

    #[test]
    fn val_context_prefers_val_backend_for_top_level_field_chain_pipeline() {
        let plan = plan_query_with_context(r#"$.a.b"#, PlanningContext::val());
        let QueryRoot::Node(root) = plan.root() else {
            panic!("expected physical plan");
        };
        assert_eq!(
            plan.backend_preferences(*root),
            &[BackendPreference::ValView, BackendPreference::Interpreted]
        );
    }

    #[test]
    fn object_shape_facts_aggregate_tape_pipeline_and_root_path_children() {
        let plan = plan_query_with_context(
            r#"{"a": $.rows.filter(score > 10).take(1), "b": $.meta.version}"#,
            PlanningContext::bytes(),
        );
        let QueryRoot::Node(root) = plan.root() else {
            panic!("expected physical plan");
        };
        let facts = plan.execution_facts(*root);
        assert!(facts.can_avoid_root_materialization);
        assert!(facts.can_stream_rows);
        assert!(facts.can_use_tape);
        assert!(!facts.contains_vm_fallback);
    }

    #[test]
    fn root_facts_classify_byte_native_object_shape() {
        let plan = plan_query_with_context(
            r#"{"a": $.rows.filter(score > 10).take(1), "b": $.meta.version}"#,
            PlanningContext::bytes(),
        );
        let facts = plan.root_execution_facts();
        assert!(facts.is_byte_native());
        assert!(facts.can_use_tape);
    }

    #[test]
    fn root_facts_classify_prefix_only_pipeline_as_needing_fallback() {
        let plan = plan_query_with_context(
            r#"$.people.filter(score > 900).map(name).take(10).filter(@ == $.target)"#,
            PlanningContext::bytes(),
        );
        let QueryRoot::Node(root) = plan.root() else {
            panic!("expected physical plan");
        };
        let facts = plan.root_execution_facts();
        assert!(facts.can_stream_rows);
        assert!(facts.can_use_tape);
        assert!(!facts.is_byte_native());
        assert!(plan
            .backend_preferences(*root)
            .contains(&BackendPreference::Interpreted));
    }

    #[test]
    fn object_shape_facts_report_vm_fallback_children() {
        let plan = plan_query_with_context(
            r#"{"a": [x for x in $.rows if x.score > 10], "b": $.meta.version}"#,
            PlanningContext::bytes(),
        );
        let QueryRoot::Node(root) = plan.root() else {
            panic!("expected physical plan");
        };
        let facts = plan.execution_facts(*root);
        assert!(facts.contains_vm_fallback);
        assert!(!facts.can_avoid_root_materialization);
    }

    #[test]
    fn root_facts_classify_vm_fallback_object_shape_as_not_byte_native() {
        let plan = plan_query_with_context(
            r#"{"a": [x for x in $.rows if x.score > 10], "b": $.meta.version}"#,
            PlanningContext::bytes(),
        );
        let facts = plan.root_execution_facts();
        assert!(facts.contains_vm_fallback);
        assert!(!facts.is_byte_native());
    }

    #[test]
    fn root_facts_classify_byte_structural_plan_as_byte_native() {
        let plan = plan_query_with_context(
            r#"$.deep_find(@ kind object and status == "open")"#,
            PlanningContext::bytes(),
        );
        let facts = plan.root_execution_facts();
        assert!(matches!(root_node(&plan), PlanNode::Structural { .. }));
        assert!(facts.is_byte_native());
        assert!(!facts.contains_vm_fallback);
    }

    #[test]
    fn root_facts_keep_val_structural_plan_as_vm_fallback() {
        let plan = plan_query_with_context(
            r#"$.deep_find(@ kind object and status == "open")"#,
            PlanningContext::val(),
        );
        let facts = plan.root_execution_facts();
        assert!(matches!(root_node(&plan), PlanNode::Structural { .. }));
        assert!(facts.contains_vm_fallback);
        assert!(!facts.is_byte_native());
    }

    #[test]
    fn source_vm_plan_is_not_byte_native() {
        let plan = QueryPlan::source_vm("$[");
        let facts = plan.root_execution_facts();
        assert!(facts.contains_vm_fallback);
        assert!(!facts.is_byte_native());
    }

    #[test]
    fn let_root_facts_are_byte_native_when_body_is_byte_native() {
        let plan =
            plan_query_with_context(r#"let x = 1 in $.meta.version"#, PlanningContext::bytes());
        let facts = plan.root_execution_facts();
        assert!(!facts.contains_vm_fallback);
        assert!(facts.is_byte_native());
    }

    #[test]
    fn deep_find_supported_predicate_lowers_to_structural_plan() {
        let plan = plan_query(r#"$.deep_find(@ kind object and status == "open")"#);
        assert!(matches!(root_node(&plan), PlanNode::Structural { .. }));
    }

    #[test]
    fn deep_find_unsupported_predicate_does_not_lower_to_structural_plan() {
        let plan = plan_query(r#"$.deep_find(score > 10)"#);
        assert!(!matches!(root_node(&plan), PlanNode::Structural { .. }));
    }

    #[test]
    fn deep_like_lowers_literal_pattern_to_structural_plan() {
        let plan = plan_query(r#"$.deep_like({role: "lead", active: true})"#);
        assert!(matches!(root_node(&plan), PlanNode::Structural { .. }));
    }

    #[test]
    fn anchored_deep_shape_lowers_to_structural_plan() {
        let plan = plan_query(r#"$.org.users.deep_shape({email})"#);
        assert!(matches!(root_node(&plan), PlanNode::Structural { .. }));
    }

    #[test]
    fn structural_prefix_can_feed_suffix_call() {
        let plan = plan_query(r#"$.org.users.deep_shape({email}).count()"#);
        let PlanNode::Pipeline { source, .. } = root_node(&plan) else {
            panic!("expected receiver pipeline");
        };
        let PipelinePlanSource::Expr(source) = source else {
            panic!("expected structural expression source");
        };
        assert!(matches!(plan.node(*source), PlanNode::Structural { .. }));
    }

    #[test]
    fn structural_prefix_can_feed_receiver_pipeline() {
        let plan = plan_query(r#"$.org.users.deep_shape({email}).take(1)"#);
        let PlanNode::Pipeline { source, .. } = root_node(&plan) else {
            panic!("expected receiver pipeline");
        };
        let PipelinePlanSource::Expr(source) = source else {
            panic!("expected structural expression source");
        };
        assert!(matches!(plan.node(*source), PlanNode::Structural { .. }));
    }

    #[test]
    fn structural_receiver_pipeline_facts_require_receiver_only_suffix() {
        let fast = plan_query_with_context(
            r#"$.org.users.deep_shape({email}).take(1)"#,
            PlanningContext::bytes(),
        );
        assert!(fast.root_execution_facts().is_byte_native());

        let fallback = plan_query_with_context(
            r#"$.org.users.deep_shape({email}).filter(@ == $.target)"#,
            PlanningContext::bytes(),
        );
        let QueryRoot::Node(root) = fallback.root() else {
            panic!("expected physical plan");
        };
        let facts = fallback.root_execution_facts();
        assert!(!facts.is_byte_native());
        assert!(fallback
            .backend_preferences(*root)
            .contains(&BackendPreference::Interpreted));
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
                assert!(matches!(plan.node(*val), PlanNode::Pipeline { .. }));
            }
            _ => panic!("expected kv field"),
        }
        let PhysicalObjField::Kv { val, .. } = &fields[1] else {
            panic!("expected kv field");
        };
        assert!(matches!(plan.node(*val), PlanNode::RootPath(_)));
    }

    #[test]
    fn descendant_prefix_can_feed_receiver_pipeline() {
        let plan =
            plan_query(r#"$..books?.first().sort_by(-price).take_while(price > 10).take(2)"#);
        let PlanNode::Pipeline { source, body } = root_node(&plan) else {
            panic!("expected physical receiver pipeline");
        };
        assert!(matches!(source, PipelinePlanSource::Expr(_)));
        assert!(matches!(body.stages[0], crate::pipeline::Stage::Sort(_)));
        assert!(matches!(
            body.stages[1],
            crate::pipeline::Stage::TakeWhile(_)
        ));
        assert!(matches!(
            body.stages[2],
            crate::pipeline::Stage::Take(2, _, _)
        ));
    }

    #[test]
    fn object_shape_keeps_multiple_nested_pipeline_children() {
        let plan = plan_query(
            r#"{"top": $.books.filter(score > 900).take(2).map(title), "first": $.books.filter(score > 900).first(), "meta": $.meta.version}"#,
        );
        let PlanNode::Object(fields) = root_node(&plan) else {
            panic!("expected physical object plan");
        };
        assert_eq!(fields.len(), 3);

        let PhysicalObjField::Kv { val, .. } = &fields[0] else {
            panic!("expected top kv field");
        };
        assert!(matches!(plan.node(*val), PlanNode::Pipeline { .. }));

        let PhysicalObjField::Kv { val, .. } = &fields[1] else {
            panic!("expected first kv field");
        };
        assert!(matches!(plan.node(*val), PlanNode::Pipeline { .. }));

        let PhysicalObjField::Kv { val, .. } = &fields[2] else {
            panic!("expected meta kv field");
        };
        assert!(matches!(plan.node(*val), PlanNode::RootPath(_)));
    }

    #[test]
    fn array_shape_keeps_pipeline_children() {
        let plan = plan_query(
            r#"[$.books.filter(score > 900).take(2).map(title), {"first": $.books.filter(score > 900).first()}, $.meta.version]"#,
        );
        let PlanNode::Array(elems) = root_node(&plan) else {
            panic!("expected physical array plan");
        };
        assert_eq!(elems.len(), 3);

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
            panic!("expected nested object kv field");
        };
        assert!(matches!(plan.node(*val), PlanNode::Pipeline { .. }));

        let PhysicalArrayElem::Expr(third) = &elems[2] else {
            panic!("expected array expr");
        };
        assert!(matches!(plan.node(*third), PlanNode::RootPath(_)));
    }

    #[test]
    fn nested_structural_shapes_keep_pipeline_children() {
        let plan = plan_query(
            r#"{"groups": [{"top": $.books.filter(score > 900).take(2).map(title)}], "meta": [$.meta.version]}"#,
        );
        let PlanNode::Object(fields) = root_node(&plan) else {
            panic!("expected physical object plan");
        };
        assert_eq!(fields.len(), 2);

        let PhysicalObjField::Kv { val: groups, .. } = &fields[0] else {
            panic!("expected groups kv field");
        };
        let PlanNode::Array(items) = plan.node(*groups) else {
            panic!("expected groups array");
        };
        let PhysicalArrayElem::Expr(item) = &items[0] else {
            panic!("expected groups array expr");
        };
        let PlanNode::Object(group_fields) = plan.node(*item) else {
            panic!("expected nested group object");
        };
        let PhysicalObjField::Kv { val, .. } = &group_fields[0] else {
            panic!("expected top kv field");
        };
        assert!(matches!(plan.node(*val), PlanNode::Pipeline { .. }));

        let PhysicalObjField::Kv { val: meta, .. } = &fields[1] else {
            panic!("expected meta kv field");
        };
        let PlanNode::Array(meta_items) = plan.node(*meta) else {
            panic!("expected meta array");
        };
        let PhysicalArrayElem::Expr(version) = &meta_items[0] else {
            panic!("expected meta version expr");
        };
        assert!(matches!(plan.node(*version), PlanNode::RootPath(_)));
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

    #[test]
    fn let_bound_receiver_chain_lowers_to_pipeline_source() {
        let plan =
            plan_query(r#"let books = $.books in books.filter(score > 900).take(2).map(title)"#);
        let PlanNode::Let { body, .. } = root_node(&plan) else {
            panic!("expected let plan");
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
    }

    #[test]
    fn object_shape_keeps_receiver_pipeline_children() {
        let plan = plan_query(
            r#"let books = $.books in {"top": books.filter(score > 900).take(2).map(title), "first": books.filter(score > 900).first()}"#,
        );
        let PlanNode::Let { body, .. } = root_node(&plan) else {
            panic!("expected let plan");
        };
        let PlanNode::Object(fields) = plan.node(*body) else {
            panic!("expected object body");
        };
        assert_eq!(fields.len(), 2);

        for idx in [0usize, 1] {
            let PhysicalObjField::Kv { val, .. } = &fields[idx] else {
                panic!("expected kv field");
            };
            let PlanNode::Pipeline {
                source: PipelinePlanSource::Expr(source),
                body,
            } = plan.node(*val)
            else {
                panic!("expected receiver pipeline source");
            };
            assert!(
                matches!(plan.node(*source), PlanNode::Local(name) if name.as_ref() == "books")
            );
            assert!(!body.stages.is_empty());
        }
    }

    #[test]
    fn array_shape_keeps_receiver_pipeline_children() {
        let plan = plan_query(
            r#"let books = $.books in [books.filter(score > 900).take(2).map(title), books.filter(score > 900).first()]"#,
        );
        let PlanNode::Let { body, .. } = root_node(&plan) else {
            panic!("expected let plan");
        };
        let PlanNode::Array(elems) = plan.node(*body) else {
            panic!("expected array body");
        };
        assert_eq!(elems.len(), 2);

        for idx in [0usize, 1] {
            let PhysicalArrayElem::Expr(val) = &elems[idx] else {
                panic!("expected array expr");
            };
            let PlanNode::Pipeline {
                source: PipelinePlanSource::Expr(source),
                body,
            } = plan.node(*val)
            else {
                panic!("expected receiver pipeline source");
            };
            assert!(
                matches!(plan.node(*source), PlanNode::Local(name) if name.as_ref() == "books")
            );
            assert!(!body.stages.is_empty());
        }
    }
}
