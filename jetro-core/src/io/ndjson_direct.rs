use crate::data::value::Val;
use crate::ir::physical::{PhysicalPathStep, PlanNode, QueryPlan};
use crate::plan::physical::PlanningContext;
use crate::JetroEngine;
use std::sync::Arc;

/// Planner-side description of NDJSON row work that can run directly on
/// simd-json tape scratch. Execution stays in `ndjson.rs`; this module owns
/// only physical-plan recognition and compact metadata for the row runner.
pub(super) type NdjsonPhysicalPath = Vec<PhysicalPathStep>;

#[derive(Clone)]
pub(super) enum NdjsonDirectTapePlan {
    RootPath(NdjsonPhysicalPath),
    ViewScalarCall {
        steps: NdjsonPhysicalPath,
        call: crate::builtins::BuiltinCall,
        optional: bool,
    },
    ObjectItems {
        steps: NdjsonPhysicalPath,
        method: crate::builtins::BuiltinMethod,
    },
    ArrayElementPath {
        source_steps: NdjsonPhysicalPath,
        element: NdjsonDirectElement,
        suffix_steps: NdjsonPhysicalPath,
    },
    MapPath {
        source_steps: NdjsonPhysicalPath,
        suffix_steps: NdjsonPhysicalPath,
    },
    MapArray {
        source_steps: NdjsonPhysicalPath,
        items: Vec<NdjsonDirectProjectionValue>,
    },
    MapObject {
        source_steps: NdjsonPhysicalPath,
        fields: Vec<NdjsonDirectObjectField>,
    },
    FilterMapPath {
        source_steps: NdjsonPhysicalPath,
        predicate: NdjsonDirectItemPredicate,
        suffix_steps: NdjsonPhysicalPath,
    },
    FilterMapArray {
        source_steps: NdjsonPhysicalPath,
        predicate: NdjsonDirectItemPredicate,
        items: Vec<NdjsonDirectProjectionValue>,
    },
    FilterMapObject {
        source_steps: NdjsonPhysicalPath,
        predicate: NdjsonDirectItemPredicate,
        fields: Vec<NdjsonDirectObjectField>,
    },
    CountFiltered {
        source_steps: NdjsonPhysicalPath,
        predicate: NdjsonDirectItemPredicate,
    },
    NumericReducePath {
        source_steps: NdjsonPhysicalPath,
        suffix_steps: NdjsonPhysicalPath,
        op: crate::exec::pipeline::NumOp,
    },
    FilterNumericReducePath {
        source_steps: NdjsonPhysicalPath,
        predicate: NdjsonDirectItemPredicate,
        suffix_steps: NdjsonPhysicalPath,
        op: crate::exec::pipeline::NumOp,
    },
    Object(Vec<NdjsonDirectObjectField>),
    Array(Vec<NdjsonDirectProjectionValue>),
    ViewPipeline {
        source_steps: NdjsonPhysicalPath,
        body: crate::exec::pipeline::PipelineBody,
    },
}

#[derive(Clone)]
pub(super) struct NdjsonDirectObjectField {
    pub(super) key: Arc<str>,
    pub(super) value: NdjsonDirectProjectionValue,
    pub(super) optional: bool,
}

#[derive(Clone)]
pub(super) enum NdjsonDirectProjectionValue {
    Path(NdjsonPhysicalPath),
    ViewScalarCall {
        steps: NdjsonPhysicalPath,
        call: crate::builtins::BuiltinCall,
        optional: bool,
    },
    Literal(Val),
}

impl NdjsonDirectTapePlan {
    pub(super) fn needs_vm(&self) -> bool {
        matches!(self, Self::ViewPipeline { .. })
    }
}

#[derive(Clone, Copy)]
pub(super) enum NdjsonDirectElement {
    First,
    Last,
    Nth(usize),
}

#[derive(Clone)]
pub(super) enum NdjsonDirectPredicate {
    Path(NdjsonPhysicalPath),
    Literal(Val),
    Not(Box<NdjsonDirectPredicate>),
    Binary {
        lhs: Box<NdjsonDirectPredicate>,
        op: crate::parse::ast::BinOp,
        rhs: Box<NdjsonDirectPredicate>,
    },
    ViewScalarCall {
        steps: NdjsonPhysicalPath,
        call: crate::builtins::BuiltinCall,
    },
    ArrayElementViewScalarCall {
        source_steps: NdjsonPhysicalPath,
        element: NdjsonDirectElement,
        suffix_steps: NdjsonPhysicalPath,
        call: crate::builtins::BuiltinCall,
    },
    ViewPipeline {
        source_steps: NdjsonPhysicalPath,
        body: crate::exec::pipeline::PipelineBody,
    },
}

#[derive(Clone)]
pub(super) enum NdjsonDirectItemPredicate {
    Path(NdjsonPhysicalPath),
    Literal(Val),
    Binary {
        lhs: Box<NdjsonDirectItemPredicate>,
        op: crate::parse::ast::BinOp,
        rhs: Box<NdjsonDirectItemPredicate>,
    },
    CmpLit {
        lhs: NdjsonPhysicalPath,
        op: crate::parse::ast::BinOp,
        lit: Val,
    },
    ViewScalarCall {
        suffix_steps: NdjsonPhysicalPath,
        call: crate::builtins::BuiltinCall,
    },
}

pub(super) fn direct_tape_plan(engine: &JetroEngine, query: &str) -> Option<NdjsonDirectTapePlan> {
    use crate::builtins::{BuiltinArgs, BuiltinMethod};
    use crate::ir::physical::QueryRoot;

    let plan = engine.cached_plan(query, PlanningContext::bytes());
    let QueryRoot::Node(root) = plan.root() else {
        return None;
    };
    if let PlanNode::Chain { base, steps } = plan.node(*root) {
        let (source_steps, element) = direct_array_element_source(&plan, *base)?;
        return Some(NdjsonDirectTapePlan::ArrayElementPath {
            source_steps,
            element,
            suffix_steps: physical_chain_to_path(steps)?,
        });
    }
    if let Some((source_steps, element)) = direct_array_element_source(&plan, *root) {
        return Some(NdjsonDirectTapePlan::ArrayElementPath {
            source_steps,
            element,
            suffix_steps: Vec::new(),
        });
    }
    match plan.node(*root) {
        PlanNode::RootPath(steps) => Some(NdjsonDirectTapePlan::RootPath(steps.clone())),
        PlanNode::Pipeline {
            source: crate::ir::physical::PipelinePlanSource::FieldChain { keys },
            body,
        } if body.stages.is_empty() && is_plain_count_sink(body) => {
            Some(NdjsonDirectTapePlan::ViewScalarCall {
                steps: keys_to_path(keys),
                call: crate::builtins::BuiltinCall::new(BuiltinMethod::Len, BuiltinArgs::None),
                optional: false,
            })
        }
        PlanNode::Pipeline {
            source: crate::ir::physical::PipelinePlanSource::Expr(source),
            body,
        } if body.stages.is_empty() && is_plain_count_sink(body) => {
            Some(NdjsonDirectTapePlan::ViewScalarCall {
                steps: root_path_steps(&plan, *source)?,
                call: crate::builtins::BuiltinCall::new(BuiltinMethod::Len, BuiltinArgs::None),
                optional: false,
            })
        }
        PlanNode::Call {
            receiver,
            call,
            optional,
        } if call.method == BuiltinMethod::Len && matches!(call.args, BuiltinArgs::None) => {
            Some(NdjsonDirectTapePlan::ViewScalarCall {
                steps: root_path_steps(&plan, *receiver)?,
                call: call.clone(),
                optional: *optional,
            })
        }
        PlanNode::Call {
            receiver,
            call,
            optional,
        } if matches!(
            call.method,
            BuiltinMethod::Keys | BuiltinMethod::Values | BuiltinMethod::Entries
        ) && matches!(call.args, BuiltinArgs::None) && !*optional =>
        {
            Some(NdjsonDirectTapePlan::ObjectItems {
                steps: root_path_steps(&plan, *receiver)?,
                method: call.method,
            })
        }
        PlanNode::Pipeline { source, body } => {
            if let Some(plan) = direct_tape_filter_numeric_reduce_path_plan(&plan, source, body) {
                return Some(plan);
            }
            if let Some(plan) = direct_tape_numeric_reduce_path_plan(&plan, source, body) {
                return Some(plan);
            }
            if let Some(plan) = direct_tape_count_filtered_plan(&plan, source, body) {
                return Some(plan);
            }
            if let Some(plan) = direct_tape_filter_map_path_plan(&plan, source, body) {
                return Some(plan);
            }
            if let Some(plan) = direct_tape_map_path_plan(&plan, source, body) {
                return Some(plan);
            }
            if !body.can_run_with_view() {
                return None;
            }
            Some(NdjsonDirectTapePlan::ViewPipeline {
                source_steps: pipeline_source_to_steps(&plan, source)?,
                body: body.clone(),
            })
        }
        PlanNode::Object(fields) => direct_tape_object_plan(&plan, fields),
        PlanNode::Array(elems) => direct_tape_array_plan(&plan, elems),
        _ => None,
    }
}

fn direct_object_value_from_node(
    plan: &QueryPlan,
    id: crate::ir::physical::NodeId,
) -> Option<NdjsonDirectProjectionValue> {
    match plan.node(id) {
        PlanNode::RootPath(steps) => Some(NdjsonDirectProjectionValue::Path(steps.clone())),
        PlanNode::Call {
            receiver,
            call,
            optional,
        } if call.spec().view_scalar => Some(NdjsonDirectProjectionValue::ViewScalarCall {
            steps: root_path_steps(plan, *receiver)?,
            call: call.clone(),
            optional: *optional,
        }),
        PlanNode::Literal(value) => Some(NdjsonDirectProjectionValue::Literal(value.clone())),
        _ => None,
    }
}

fn direct_tape_object_plan(
    plan: &QueryPlan,
    fields: &[crate::ir::physical::PhysicalObjField],
) -> Option<NdjsonDirectTapePlan> {
    use crate::ir::physical::PhysicalObjField;

    let mut out = Vec::with_capacity(fields.len());
    for field in fields {
        let PhysicalObjField::Kv {
            key,
            val,
            optional,
            cond: None,
        } = field
        else {
            return None;
        };
        let value = direct_object_value_from_node(plan, *val)?;
        out.push(NdjsonDirectObjectField {
            key: key.clone(),
            value,
            optional: *optional,
        });
    }
    Some(NdjsonDirectTapePlan::Object(out))
}

fn direct_tape_array_plan(
    plan: &QueryPlan,
    elems: &[crate::ir::physical::PhysicalArrayElem],
) -> Option<NdjsonDirectTapePlan> {
    use crate::ir::physical::PhysicalArrayElem;

    let mut out = Vec::with_capacity(elems.len());
    for elem in elems {
        let PhysicalArrayElem::Expr(id) = elem else {
            return None;
        };
        out.push(direct_object_value_from_node(plan, *id)?);
    }
    Some(NdjsonDirectTapePlan::Array(out))
}

fn pipeline_source_to_steps(
    plan: &QueryPlan,
    source: &crate::ir::physical::PipelinePlanSource,
) -> Option<NdjsonPhysicalPath> {
    match source {
        crate::ir::physical::PipelinePlanSource::FieldChain { keys } => Some(keys_to_path(keys)),
        crate::ir::physical::PipelinePlanSource::Expr(source) => root_path_steps(plan, *source),
    }
}

fn is_plain_count_sink(body: &crate::exec::pipeline::PipelineBody) -> bool {
    matches!(
        body.sink,
        crate::exec::pipeline::Sink::Reducer(ref spec)
            if spec.op == crate::exec::pipeline::ReducerOp::Count && spec.predicate.is_none()
    )
}

fn keys_to_path(keys: &[Arc<str>]) -> NdjsonPhysicalPath {
    keys.iter()
        .map(|key| PhysicalPathStep::Field(key.clone()))
        .collect()
}

fn root_path_steps(
    plan: &QueryPlan,
    id: crate::ir::physical::NodeId,
) -> Option<NdjsonPhysicalPath> {
    let PlanNode::RootPath(steps) = plan.node(id) else {
        return None;
    };
    Some(steps.clone())
}

pub(super) fn direct_tape_predicate(
    engine: &JetroEngine,
    predicate: &str,
) -> Option<NdjsonDirectPredicate> {
    let plan = engine.cached_plan(predicate, PlanningContext::bytes());
    let crate::ir::physical::QueryRoot::Node(root) = plan.root() else {
        return None;
    };
    direct_tape_predicate_node(&plan, *root)
}

fn direct_tape_predicate_node(
    plan: &QueryPlan,
    id: crate::ir::physical::NodeId,
) -> Option<NdjsonDirectPredicate> {
    match plan.node(id) {
        PlanNode::Literal(value) => Some(NdjsonDirectPredicate::Literal(value.clone())),
        PlanNode::RootPath(steps) => Some(NdjsonDirectPredicate::Path(steps.clone())),
        PlanNode::Not(inner) => Some(NdjsonDirectPredicate::Not(Box::new(
            direct_tape_predicate_node(plan, *inner)?,
        ))),
        PlanNode::Binary { lhs, op, rhs } => Some(NdjsonDirectPredicate::Binary {
            lhs: Box::new(direct_tape_predicate_node(plan, *lhs)?),
            op: *op,
            rhs: Box::new(direct_tape_predicate_node(plan, *rhs)?),
        }),
        PlanNode::Call {
            receiver,
            call,
            optional,
        } if !*optional && call.spec().view_scalar => {
            direct_tape_predicate_scalar_call(plan, *receiver, call.clone())
        }
        PlanNode::Pipeline { source, body } => {
            if let Some(predicate) = direct_tape_predicate_membership_sink(plan, source, body) {
                return Some(predicate);
            }
            if !body.can_run_with_view() {
                return None;
            }
            Some(NdjsonDirectPredicate::ViewPipeline {
                source_steps: pipeline_source_to_steps(plan, source)?,
                body: body.clone(),
            })
        }
        _ => None,
    }
}

fn direct_tape_map_path_plan(
    plan: &QueryPlan,
    source: &crate::ir::physical::PipelinePlanSource,
    body: &crate::exec::pipeline::PipelineBody,
) -> Option<NdjsonDirectTapePlan> {
    use crate::exec::pipeline::{Sink, Stage};

    if !matches!(body.sink, Sink::Collect) || body.stages.len() != 1 {
        return None;
    }
    let Stage::Map(_, _) = body.stages.first()? else {
        return None;
    };
    let source_steps = pipeline_source_to_steps(plan, source)?;
    let kernel = body.stage_kernels.first()?;
    if let Some(suffix_steps) = kernel_to_physical_path(kernel) {
        return Some(NdjsonDirectTapePlan::MapPath {
            source_steps,
            suffix_steps,
        });
    }
    if let crate::exec::pipeline::BodyKernel::Array(items) = kernel {
        let items = items
            .iter()
            .map(direct_projection_value_from_kernel)
            .collect::<Option<Vec<_>>>()?;
        return Some(NdjsonDirectTapePlan::MapArray {
            source_steps,
            items,
        });
    }
    if let crate::exec::pipeline::BodyKernel::Object(object) = kernel {
        return Some(NdjsonDirectTapePlan::MapObject {
            source_steps,
            fields: direct_object_fields_from_kernel(object)?,
        });
    }
    None
}

fn direct_tape_count_filtered_plan(
    plan: &QueryPlan,
    source: &crate::ir::physical::PipelinePlanSource,
    body: &crate::exec::pipeline::PipelineBody,
) -> Option<NdjsonDirectTapePlan> {
    use crate::exec::pipeline::{ReducerOp, Sink, Stage};

    if body.stages.len() != 1 {
        return None;
    }
    let Stage::Filter(_, _) = body.stages.first()? else {
        return None;
    };
    let Sink::Reducer(spec) = &body.sink else {
        return None;
    };
    if spec.op != ReducerOp::Count || spec.predicate.is_some() {
        return None;
    }
    Some(NdjsonDirectTapePlan::CountFiltered {
        source_steps: pipeline_source_to_steps(plan, source)?,
        predicate: direct_item_predicate_from_kernel(body.stage_kernels.first()?)?,
    })
}

fn direct_tape_numeric_reduce_path_plan(
    plan: &QueryPlan,
    source: &crate::ir::physical::PipelinePlanSource,
    body: &crate::exec::pipeline::PipelineBody,
) -> Option<NdjsonDirectTapePlan> {
    use crate::exec::pipeline::{ReducerOp, Sink, Stage};

    let Sink::Reducer(spec) = &body.sink else {
        return None;
    };
    if spec.predicate.is_some() {
        return None;
    }
    let ReducerOp::Numeric(op) = spec.op else {
        return None;
    };
    let suffix_steps = match body.stages.as_slice() {
        [Stage::Map(_, _)] if spec.projection.is_none() => {
            kernel_to_physical_path(body.stage_kernels.first()?)?
        }
        [] if spec.projection.is_some() => kernel_to_physical_path(body.sink_kernels.first()?)?,
        _ => return None,
    };
    Some(NdjsonDirectTapePlan::NumericReducePath {
        source_steps: pipeline_source_to_steps(plan, source)?,
        suffix_steps,
        op,
    })
}

fn direct_tape_filter_numeric_reduce_path_plan(
    plan: &QueryPlan,
    source: &crate::ir::physical::PipelinePlanSource,
    body: &crate::exec::pipeline::PipelineBody,
) -> Option<NdjsonDirectTapePlan> {
    use crate::exec::pipeline::{ReducerOp, Sink, Stage};

    let Sink::Reducer(spec) = &body.sink else {
        return None;
    };
    if spec.predicate.is_some() {
        return None;
    }
    let ReducerOp::Numeric(op) = spec.op else {
        return None;
    };
    let (predicate, suffix_steps) = match body.stages.as_slice() {
        [Stage::Filter(_, _), Stage::Map(_, _)] if spec.projection.is_none() => (
            direct_item_predicate_from_kernel(body.stage_kernels.first()?)?,
            kernel_to_physical_path(body.stage_kernels.get(1)?)?,
        ),
        [Stage::Filter(_, _)] if spec.projection.is_some() => (
            direct_item_predicate_from_kernel(body.stage_kernels.first()?)?,
            kernel_to_physical_path(body.sink_kernels.first()?)?,
        ),
        _ => return None,
    };
    Some(NdjsonDirectTapePlan::FilterNumericReducePath {
        source_steps: pipeline_source_to_steps(plan, source)?,
        predicate,
        suffix_steps,
        op,
    })
}

fn direct_tape_filter_map_path_plan(
    plan: &QueryPlan,
    source: &crate::ir::physical::PipelinePlanSource,
    body: &crate::exec::pipeline::PipelineBody,
) -> Option<NdjsonDirectTapePlan> {
    use crate::exec::pipeline::{Sink, Stage};

    if !matches!(body.sink, Sink::Collect) || body.stages.len() != 2 {
        return None;
    }
    let [Stage::Filter(_, _), Stage::Map(_, _)] = body.stages.as_slice() else {
        return None;
    };
    let source_steps = pipeline_source_to_steps(plan, source)?;
    let predicate = direct_item_predicate_from_kernel(body.stage_kernels.first()?)?;
    let kernel = body.stage_kernels.get(1)?;
    if let Some(suffix_steps) = kernel_to_physical_path(kernel) {
        return Some(NdjsonDirectTapePlan::FilterMapPath {
            source_steps,
            predicate,
            suffix_steps,
        });
    }
    if let crate::exec::pipeline::BodyKernel::Array(items) = kernel {
        let items = items
            .iter()
            .map(direct_projection_value_from_kernel)
            .collect::<Option<Vec<_>>>()?;
        return Some(NdjsonDirectTapePlan::FilterMapArray {
            source_steps,
            predicate,
            items,
        });
    }
    if let crate::exec::pipeline::BodyKernel::Object(object) = kernel {
        return Some(NdjsonDirectTapePlan::FilterMapObject {
            source_steps,
            predicate,
            fields: direct_object_fields_from_kernel(object)?,
        });
    }
    None
}

fn direct_item_predicate_from_kernel(
    kernel: &crate::exec::pipeline::BodyKernel,
) -> Option<NdjsonDirectItemPredicate> {
    match kernel {
        crate::exec::pipeline::BodyKernel::Current => {
            Some(NdjsonDirectItemPredicate::Path(Vec::new()))
        }
        crate::exec::pipeline::BodyKernel::Const(value) => {
            Some(NdjsonDirectItemPredicate::Literal(value.clone()))
        }
        crate::exec::pipeline::BodyKernel::ConstBool(value) => {
            Some(NdjsonDirectItemPredicate::Literal(Val::Bool(*value)))
        }
        crate::exec::pipeline::BodyKernel::FieldRead(_)
        | crate::exec::pipeline::BodyKernel::FieldChain(_) => Some(
            NdjsonDirectItemPredicate::Path(kernel_to_physical_path(kernel)?),
        ),
        crate::exec::pipeline::BodyKernel::FieldCmpLit(field, op, lit) => {
            Some(NdjsonDirectItemPredicate::CmpLit {
                lhs: vec![PhysicalPathStep::Field(field.clone())],
                op: *op,
                lit: lit.clone(),
            })
        }
        crate::exec::pipeline::BodyKernel::FieldChainCmpLit(keys, op, lit) => {
            Some(NdjsonDirectItemPredicate::CmpLit {
                lhs: keys_to_path(keys),
                op: *op,
                lit: lit.clone(),
            })
        }
        crate::exec::pipeline::BodyKernel::CurrentCmpLit(op, lit) => {
            Some(NdjsonDirectItemPredicate::CmpLit {
                lhs: Vec::new(),
                op: *op,
                lit: lit.clone(),
            })
        }
        crate::exec::pipeline::BodyKernel::CmpLit { lhs, op, lit } => {
            Some(NdjsonDirectItemPredicate::CmpLit {
                lhs: kernel_to_physical_path(lhs)?,
                op: *op,
                lit: lit.clone(),
            })
        }
        crate::exec::pipeline::BodyKernel::And(items) => {
            let mut iter = items.iter().map(direct_item_predicate_from_kernel);
            let mut acc = iter.next()??;
            for item in iter {
                acc = NdjsonDirectItemPredicate::Binary {
                    lhs: Box::new(acc),
                    op: crate::parse::ast::BinOp::And,
                    rhs: Box::new(item?),
                };
            }
            Some(acc)
        }
        crate::exec::pipeline::BodyKernel::Or(items) => {
            let mut iter = items.iter().map(direct_item_predicate_from_kernel);
            let mut acc = iter.next()??;
            for item in iter {
                acc = NdjsonDirectItemPredicate::Binary {
                    lhs: Box::new(acc),
                    op: crate::parse::ast::BinOp::Or,
                    rhs: Box::new(item?),
                };
            }
            Some(acc)
        }
        crate::exec::pipeline::BodyKernel::BuiltinCall { receiver, call }
            if call.spec().view_scalar =>
        {
            Some(NdjsonDirectItemPredicate::ViewScalarCall {
                suffix_steps: kernel_to_physical_path(receiver)?,
                call: call.clone(),
            })
        }
        _ => None,
    }
}

fn kernel_to_physical_path(
    kernel: &crate::exec::pipeline::BodyKernel,
) -> Option<NdjsonPhysicalPath> {
    match kernel {
        crate::exec::pipeline::BodyKernel::FieldRead(key) => {
            Some(vec![PhysicalPathStep::Field(key.clone())])
        }
        crate::exec::pipeline::BodyKernel::FieldChain(keys) => Some(keys_to_path(keys)),
        crate::exec::pipeline::BodyKernel::Current => Some(Vec::new()),
        _ => None,
    }
}

fn direct_projection_value_from_kernel(
    kernel: &crate::exec::pipeline::BodyKernel,
) -> Option<NdjsonDirectProjectionValue> {
    match kernel {
        crate::exec::pipeline::BodyKernel::Current
        | crate::exec::pipeline::BodyKernel::FieldRead(_)
        | crate::exec::pipeline::BodyKernel::FieldChain(_) => {
            Some(NdjsonDirectProjectionValue::Path(kernel_to_physical_path(kernel)?))
        }
        crate::exec::pipeline::BodyKernel::Const(value) => {
            Some(NdjsonDirectProjectionValue::Literal(value.clone()))
        }
        crate::exec::pipeline::BodyKernel::ConstBool(value) => {
            Some(NdjsonDirectProjectionValue::Literal(Val::Bool(*value)))
        }
        crate::exec::pipeline::BodyKernel::BuiltinCall { receiver, call }
            if call.spec().view_scalar =>
        {
            Some(NdjsonDirectProjectionValue::ViewScalarCall {
                steps: kernel_to_physical_path(receiver)?,
                call: call.clone(),
                optional: false,
            })
        }
        _ => None,
    }
}

fn direct_object_fields_from_kernel(
    object: &crate::exec::pipeline::ObjectKernel,
) -> Option<Vec<NdjsonDirectObjectField>> {
    object
        .entries()
        .iter()
        .map(|entry| {
            Some(NdjsonDirectObjectField {
                key: entry.key().clone(),
                value: direct_projection_value_from_kernel(entry.value())?,
                optional: entry.optional() || entry.omit_null(),
            })
        })
        .collect()
}

fn direct_tape_predicate_membership_sink(
    plan: &QueryPlan,
    source: &crate::ir::physical::PipelinePlanSource,
    body: &crate::exec::pipeline::PipelineBody,
) -> Option<NdjsonDirectPredicate> {
    use crate::builtins::{BuiltinArgs, BuiltinCall};
    use crate::exec::pipeline::{MembershipSinkOp, MembershipSinkTarget, Sink};

    if !body.stages.is_empty() {
        return None;
    }
    let Sink::Membership(spec) = &body.sink else {
        return None;
    };
    if spec.op != MembershipSinkOp::Includes {
        return None;
    }
    let MembershipSinkTarget::Literal(target) = &spec.target else {
        return None;
    };
    let call = BuiltinCall::new(spec.method, BuiltinArgs::Val(target.clone()));
    direct_tape_predicate_source_scalar_call(plan, source, call)
}

fn direct_tape_predicate_source_scalar_call(
    plan: &QueryPlan,
    source: &crate::ir::physical::PipelinePlanSource,
    call: crate::builtins::BuiltinCall,
) -> Option<NdjsonDirectPredicate> {
    match source {
        crate::ir::physical::PipelinePlanSource::FieldChain { keys } => {
            Some(NdjsonDirectPredicate::ViewScalarCall {
                steps: keys_to_path(keys),
                call,
            })
        }
        crate::ir::physical::PipelinePlanSource::Expr(receiver) => {
            direct_tape_predicate_scalar_call(plan, *receiver, call)
        }
    }
}

fn direct_tape_predicate_scalar_call(
    plan: &QueryPlan,
    receiver: crate::ir::physical::NodeId,
    call: crate::builtins::BuiltinCall,
) -> Option<NdjsonDirectPredicate> {
    if let PlanNode::RootPath(steps) = plan.node(receiver) {
        return Some(NdjsonDirectPredicate::ViewScalarCall {
            steps: steps.clone(),
            call,
        });
    }

    let PlanNode::Chain { base, steps } = plan.node(receiver) else {
        return None;
    };
    let (source_steps, element) = direct_array_element_source(plan, *base)?;
    Some(NdjsonDirectPredicate::ArrayElementViewScalarCall {
        source_steps,
        element,
        suffix_steps: physical_chain_to_path(steps)?,
        call,
    })
}

fn direct_array_element_source(
    plan: &QueryPlan,
    id: crate::ir::physical::NodeId,
) -> Option<(NdjsonPhysicalPath, NdjsonDirectElement)> {
    use crate::builtins::BuiltinMethod;
    use crate::exec::pipeline::Sink;
    use crate::ir::physical::PipelinePlanSource;

    if let PlanNode::Call {
        receiver,
        call,
        optional,
    } = plan.node(id)
    {
        if *optional {
            return None;
        }
        let element = match call.method {
            BuiltinMethod::First => NdjsonDirectElement::First,
            BuiltinMethod::Last => NdjsonDirectElement::Last,
            _ => return None,
        };
        let PlanNode::RootPath(steps) = plan.node(*receiver) else {
            return None;
        };
        return Some((steps.clone(), element));
    }

    let PlanNode::Pipeline { source, body } = plan.node(id) else {
        return None;
    };
    if !body.stages.is_empty() {
        return None;
    }
    let element = match body.sink {
        Sink::Terminal(BuiltinMethod::First) => NdjsonDirectElement::First,
        Sink::Terminal(BuiltinMethod::Last) => NdjsonDirectElement::Last,
        Sink::SelectMany {
            n: 1,
            from_end: false,
        } => NdjsonDirectElement::First,
        Sink::SelectMany {
            n: 1,
            from_end: true,
        } => NdjsonDirectElement::Last,
        Sink::Nth(n) => NdjsonDirectElement::Nth(n),
        _ => return None,
    };
    let source_steps = match source {
        PipelinePlanSource::FieldChain { keys } => keys_to_path(keys),
        PipelinePlanSource::Expr(source) => root_path_steps(plan, *source)?,
    };
    Some((source_steps, element))
}

fn physical_chain_to_path(
    steps: &[crate::ir::physical::PhysicalChainStep],
) -> Option<NdjsonPhysicalPath> {
    steps
        .iter()
        .map(|step| match step {
            crate::ir::physical::PhysicalChainStep::Field(key) => {
                Some(PhysicalPathStep::Field(key.clone()))
            }
            crate::ir::physical::PhysicalChainStep::Index(idx) => {
                Some(PhysicalPathStep::Index(*idx))
            }
            crate::ir::physical::PhysicalChainStep::DynIndex(_) => None,
        })
        .collect()
}
