use super::report::{
    BackendInspection, BackendKind, BackendStatus, ExecutionFactsInspection, InspectionSummary,
    PhysicalInspection, PhysicalNodeInspection,
};
use crate::ir::physical::{
    BackendPreference, ExecutionFacts, NodeId, PhysicalArrayElem, PhysicalObjField, PlanNode,
    QueryPlan, QueryRoot,
};

pub(crate) fn inspect_physical_plan(plan: &QueryPlan) -> (InspectionSummary, PhysicalInspection) {
    let root = root_label(plan.root());
    let summary_facts = root_facts(plan);
    let selected_executor = match plan.root() {
        QueryRoot::Node(root) => plan
            .backend_preferences(*root)
            .first()
            .copied()
            .map(backend_kind),
        QueryRoot::SourceVm(_) => Some(BackendKind::Vm),
    };
    let summary = InspectionSummary {
        root: root.clone(),
        selected_executor,
        materializes_root: !summary_facts.can_avoid_root_materialization,
        contains_vm_fallback: summary_facts.contains_vm_fallback,
        can_run_byte_native: summary_facts.is_byte_native(),
    };
    let nodes = plan
        .node_ids()
        .map(|id| inspect_node(plan, id))
        .collect::<Vec<_>>();
    (summary, PhysicalInspection { root, nodes })
}

fn root_label(root: &QueryRoot) -> String {
    match root {
        QueryRoot::Node(id) => format!("node:{}", id.0),
        QueryRoot::SourceVm(_) => "source-vm".to_string(),
    }
}

fn root_facts(plan: &QueryPlan) -> ExecutionFacts {
    match plan.root() {
        QueryRoot::Node(root) => plan.execution_facts(*root),
        QueryRoot::SourceVm(_) => ExecutionFacts {
            contains_vm_fallback: true,
            ..ExecutionFacts::default()
        },
    }
}

fn inspect_node(plan: &QueryPlan, id: NodeId) -> PhysicalNodeInspection {
    let node = plan.node(id);
    PhysicalNodeInspection {
        id: id.0,
        kind: node_kind(node).to_string(),
        children: node_children(node).into_iter().map(|id| id.0).collect(),
        facts: facts_inspection(plan.execution_facts(id)),
        backends: inspect_backends(plan.backend_preferences(id)),
        detail: node_detail(node),
    }
}

fn facts_inspection(facts: ExecutionFacts) -> ExecutionFactsInspection {
    ExecutionFactsInspection {
        can_avoid_root_materialization: facts.can_avoid_root_materialization,
        contains_vm_fallback: facts.contains_vm_fallback,
        can_run_byte_native: facts.is_byte_native(),
    }
}

fn inspect_backends(backends: &[BackendPreference]) -> Vec<BackendInspection> {
    backends
        .iter()
        .enumerate()
        .map(|(index, backend)| BackendInspection {
            backend: backend_kind(*backend),
            status: if index == 0 {
                BackendStatus::Selected
            } else if *backend == BackendPreference::Interpreted {
                BackendStatus::Fallback
            } else {
                BackendStatus::Planned
            },
            reason: None,
        })
        .collect()
}

fn backend_kind(backend: BackendPreference) -> BackendKind {
    match backend {
        BackendPreference::Structural => BackendKind::StructuralIndex,
        BackendPreference::TapeView => BackendKind::ViewPipeline,
        BackendPreference::TapeRows => BackendKind::TapeRows,
        BackendPreference::TapePath => BackendKind::TapePath,
        BackendPreference::ValView => BackendKind::ValView,
        BackendPreference::MaterializedSource => BackendKind::MaterializedSource,
        BackendPreference::FastChildren => BackendKind::FastChildren,
        BackendPreference::Interpreted => BackendKind::Interpreted,
    }
}

fn node_kind(node: &PlanNode) -> &'static str {
    match node {
        PlanNode::Literal(_) => "literal",
        PlanNode::Root => "root",
        PlanNode::Current => "current",
        PlanNode::Ident(_) => "ident",
        PlanNode::Local(_) => "local",
        PlanNode::Pipeline { .. } => "pipeline",
        PlanNode::Structural { .. } => "structural",
        PlanNode::RootPath(_) => "root-path",
        PlanNode::Chain { .. } => "chain",
        PlanNode::Call { .. } => "call",
        PlanNode::UnaryNeg(_) => "unary-neg",
        PlanNode::Not(_) => "not",
        PlanNode::Binary { .. } => "binary",
        PlanNode::Kind { .. } => "kind",
        PlanNode::Coalesce { .. } => "coalesce",
        PlanNode::IfElse { .. } => "if-else",
        PlanNode::Try { .. } => "try",
        PlanNode::Object(_) => "object",
        PlanNode::Array(_) => "array",
        PlanNode::Let { .. } => "let",
        PlanNode::UpdateBatch { .. } => "update-batch",
        PlanNode::Vm(_) => "vm",
    }
}

fn node_detail(node: &PlanNode) -> Option<String> {
    match node {
        PlanNode::Ident(name) | PlanNode::Local(name) => Some(name.to_string()),
        PlanNode::Pipeline { source, body } => Some(format!(
            "source={}, stages={}, sink={}",
            pipeline_source_label(source),
            body.stages.len(),
            pipeline_sink_label(&body.sink)
        )),
        PlanNode::RootPath(steps) => Some(format!("steps={}", steps.len())),
        PlanNode::Chain { steps, .. } => Some(format!("steps={}", steps.len())),
        PlanNode::Call { call, optional, .. } => {
            Some(format!("method={:?}, optional={optional}", call.method))
        }
        PlanNode::Binary { op, .. } => Some(format!("op={op:?}")),
        PlanNode::Kind { ty, negate, .. } => Some(format!("kind={ty:?}, negate={negate}")),
        PlanNode::Object(fields) => Some(format!("fields={}", fields.len())),
        PlanNode::Array(items) => Some(format!("items={}", items.len())),
        PlanNode::Let { name, .. } => Some(format!("name={name}")),
        PlanNode::UpdateBatch { selector, ops, .. } => Some(format!(
            "selector_steps={}, ops={}",
            selector.len(),
            ops.len()
        )),
        _ => None,
    }
}

fn pipeline_source_label(source: &crate::ir::physical::PipelinePlanSource) -> String {
    match source {
        crate::ir::physical::PipelinePlanSource::FieldChain { keys } => {
            format!("field-chain({})", keys.len())
        }
        crate::ir::physical::PipelinePlanSource::RootPath { steps } => {
            format!("root-path({})", steps.len())
        }
        crate::ir::physical::PipelinePlanSource::Expr(id) => format!("expr:{}", id.0),
    }
}

fn pipeline_sink_label(sink: &crate::exec::pipeline::Sink) -> &'static str {
    match sink {
        crate::exec::pipeline::Sink::Collect => "collect",
        crate::exec::pipeline::Sink::Nth(_) => "nth",
        crate::exec::pipeline::Sink::Reducer(_) => "reducer",
        crate::exec::pipeline::Sink::Terminal(_) => "terminal",
        crate::exec::pipeline::Sink::SelectMany { .. } => "select-many",
        crate::exec::pipeline::Sink::Membership(_) => "membership",
        crate::exec::pipeline::Sink::ArgExtreme(_) => "arg-extreme",
        crate::exec::pipeline::Sink::Predicate(_) => "predicate",
        crate::exec::pipeline::Sink::ApproxCountDistinct => "approx-count-distinct",
    }
}

fn node_children(node: &PlanNode) -> Vec<NodeId> {
    match node {
        PlanNode::Chain { base, .. } => vec![*base],
        PlanNode::Call { receiver, .. } => vec![*receiver],
        PlanNode::UnaryNeg(inner) | PlanNode::Not(inner) | PlanNode::Kind { expr: inner, .. } => {
            vec![*inner]
        }
        PlanNode::Binary { lhs, rhs, .. } | PlanNode::Coalesce { lhs, rhs } => vec![*lhs, *rhs],
        PlanNode::IfElse { cond, then_, else_ } => vec![*cond, *then_, *else_],
        PlanNode::Try { body, default } => vec![*body, *default],
        PlanNode::Object(fields) => object_children(fields),
        PlanNode::Array(items) => array_children(items),
        PlanNode::Let { init, body, .. } => vec![*init, *body],
        PlanNode::Pipeline {
            source: crate::ir::physical::PipelinePlanSource::Expr(source),
            ..
        } => vec![*source],
        PlanNode::UpdateBatch { root, .. } => vec![*root],
        _ => Vec::new(),
    }
}

fn object_children(fields: &[PhysicalObjField]) -> Vec<NodeId> {
    let mut out = Vec::new();
    for field in fields {
        match field {
            PhysicalObjField::Kv { val, cond, .. } => {
                out.push(*val);
                if let Some(cond) = cond {
                    out.push(*cond);
                }
            }
            PhysicalObjField::Short(_) => {}
            PhysicalObjField::Dynamic { key, val } => {
                out.push(*key);
                out.push(*val);
            }
            PhysicalObjField::Spread(expr) => out.push(*expr),
            PhysicalObjField::SpreadDeep(expr) => out.push(*expr),
        }
    }
    out
}

fn array_children(items: &[PhysicalArrayElem]) -> Vec<NodeId> {
    items
        .iter()
        .map(|item| match item {
            PhysicalArrayElem::Expr(id) | PhysicalArrayElem::Spread(id) => *id,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn physical_inspection_exports_nodes_and_backend_preferences() {
        let plan = crate::plan::physical::plan_query_with_context(
            "$.books.filter(price > 10).map(title)",
            crate::plan::physical::PlanningContext::bytes(),
        );

        let (summary, physical) = super::inspect_physical_plan(&plan);

        assert!(!physical.nodes.is_empty());
        assert!(physical.nodes.iter().any(|node| node.kind == "pipeline"));
        assert!(physical.nodes.iter().any(|node| !node.backends.is_empty()));
        assert!(summary.selected_executor.is_some());
    }
}
