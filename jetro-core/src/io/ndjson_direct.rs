use crate::data::value::Val;
use crate::plan::physical::PlanningContext;
use crate::JetroEngine;

pub(super) type NdjsonPhysicalPath = Vec<crate::ir::physical::PhysicalPathStep>;

#[derive(Clone)]
pub(super) enum NdjsonDirectTapePlan {
    RootPath(NdjsonPhysicalPath),
    ViewScalarCall {
        steps: NdjsonPhysicalPath,
        call: crate::builtins::BuiltinCall,
        optional: bool,
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
    FilterMapPath {
        source_steps: NdjsonPhysicalPath,
        predicate: NdjsonDirectItemPredicate,
        suffix_steps: NdjsonPhysicalPath,
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
    ViewPipeline {
        source_steps: NdjsonPhysicalPath,
        body: crate::exec::pipeline::PipelineBody,
    },
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
    use crate::ir::physical::{PlanNode, QueryRoot};

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
        } if body.stages.is_empty()
            && matches!(
                body.sink,
                crate::exec::pipeline::Sink::Reducer(ref spec)
                    if spec.op == crate::exec::pipeline::ReducerOp::Count
                        && spec.predicate.is_none()
            ) =>
        {
            Some(NdjsonDirectTapePlan::ViewScalarCall {
                steps: keys
                    .iter()
                    .map(|key| crate::ir::physical::PhysicalPathStep::Field(key.clone()))
                    .collect(),
                call: crate::builtins::BuiltinCall::new(BuiltinMethod::Len, BuiltinArgs::None),
                optional: false,
            })
        }
        PlanNode::Pipeline {
            source: crate::ir::physical::PipelinePlanSource::Expr(source),
            body,
        } if body.stages.is_empty()
            && matches!(
                body.sink,
                crate::exec::pipeline::Sink::Reducer(ref spec)
                    if spec.op == crate::exec::pipeline::ReducerOp::Count
                        && spec.predicate.is_none()
            ) =>
        {
            let PlanNode::RootPath(steps) = plan.node(*source) else {
                return None;
            };
            Some(NdjsonDirectTapePlan::ViewScalarCall {
                steps: steps.clone(),
                call: crate::builtins::BuiltinCall::new(BuiltinMethod::Len, BuiltinArgs::None),
                optional: false,
            })
        }
        PlanNode::Call {
            receiver,
            call,
            optional,
        } if call.method == BuiltinMethod::Len && matches!(call.args, BuiltinArgs::None) => {
            let PlanNode::RootPath(steps) = plan.node(*receiver) else {
                return None;
            };
            Some(NdjsonDirectTapePlan::ViewScalarCall {
                steps: steps.clone(),
                call: call.clone(),
                optional: *optional,
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
        _ => None,
    }
}

fn pipeline_source_to_steps(
    plan: &crate::ir::physical::QueryPlan,
    source: &crate::ir::physical::PipelinePlanSource,
) -> Option<NdjsonPhysicalPath> {
    match source {
        crate::ir::physical::PipelinePlanSource::FieldChain { keys } => Some(
            keys.iter()
                .map(|key| crate::ir::physical::PhysicalPathStep::Field(key.clone()))
                .collect(),
        ),
        crate::ir::physical::PipelinePlanSource::Expr(source) => {
            let crate::ir::physical::PlanNode::RootPath(steps) = plan.node(*source) else {
                return None;
            };
            Some(steps.clone())
        }
    }
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
    plan: &crate::ir::physical::QueryPlan,
    id: crate::ir::physical::NodeId,
) -> Option<NdjsonDirectPredicate> {
    use crate::ir::physical::PlanNode;

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
    plan: &crate::ir::physical::QueryPlan,
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
    Some(NdjsonDirectTapePlan::MapPath {
        source_steps: pipeline_source_to_steps(plan, source)?,
        suffix_steps: kernel_to_physical_path(body.stage_kernels.first()?)?,
    })
}

fn direct_tape_count_filtered_plan(
    plan: &crate::ir::physical::QueryPlan,
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
    plan: &crate::ir::physical::QueryPlan,
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
    plan: &crate::ir::physical::QueryPlan,
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
    plan: &crate::ir::physical::QueryPlan,
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
    Some(NdjsonDirectTapePlan::FilterMapPath {
        source_steps: pipeline_source_to_steps(plan, source)?,
        predicate: direct_item_predicate_from_kernel(body.stage_kernels.first()?)?,
        suffix_steps: kernel_to_physical_path(body.stage_kernels.get(1)?)?,
    })
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
                lhs: vec![crate::ir::physical::PhysicalPathStep::Field(field.clone())],
                op: *op,
                lit: lit.clone(),
            })
        }
        crate::exec::pipeline::BodyKernel::FieldChainCmpLit(keys, op, lit) => {
            Some(NdjsonDirectItemPredicate::CmpLit {
                lhs: keys
                    .iter()
                    .map(|key| crate::ir::physical::PhysicalPathStep::Field(key.clone()))
                    .collect(),
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
            Some(vec![crate::ir::physical::PhysicalPathStep::Field(
                key.clone(),
            )])
        }
        crate::exec::pipeline::BodyKernel::FieldChain(keys) => Some(
            keys.iter()
                .map(|key| crate::ir::physical::PhysicalPathStep::Field(key.clone()))
                .collect(),
        ),
        crate::exec::pipeline::BodyKernel::Current => Some(Vec::new()),
        _ => None,
    }
}

fn direct_tape_predicate_membership_sink(
    plan: &crate::ir::physical::QueryPlan,
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
    plan: &crate::ir::physical::QueryPlan,
    source: &crate::ir::physical::PipelinePlanSource,
    call: crate::builtins::BuiltinCall,
) -> Option<NdjsonDirectPredicate> {
    match source {
        crate::ir::physical::PipelinePlanSource::FieldChain { keys } => {
            Some(NdjsonDirectPredicate::ViewScalarCall {
                steps: keys
                    .iter()
                    .map(|key| crate::ir::physical::PhysicalPathStep::Field(key.clone()))
                    .collect(),
                call,
            })
        }
        crate::ir::physical::PipelinePlanSource::Expr(receiver) => {
            direct_tape_predicate_scalar_call(plan, *receiver, call)
        }
    }
}

fn direct_tape_predicate_scalar_call(
    plan: &crate::ir::physical::QueryPlan,
    receiver: crate::ir::physical::NodeId,
    call: crate::builtins::BuiltinCall,
) -> Option<NdjsonDirectPredicate> {
    use crate::ir::physical::PlanNode;

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
    plan: &crate::ir::physical::QueryPlan,
    id: crate::ir::physical::NodeId,
) -> Option<(NdjsonPhysicalPath, NdjsonDirectElement)> {
    use crate::builtins::BuiltinMethod;
    use crate::exec::pipeline::Sink;
    use crate::ir::physical::{PipelinePlanSource, PlanNode};

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
        PipelinePlanSource::FieldChain { keys } => keys
            .iter()
            .map(|key| crate::ir::physical::PhysicalPathStep::Field(key.clone()))
            .collect(),
        PipelinePlanSource::Expr(source) => {
            let PlanNode::RootPath(steps) = plan.node(*source) else {
                return None;
            };
            steps.clone()
        }
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
                Some(crate::ir::physical::PhysicalPathStep::Field(key.clone()))
            }
            crate::ir::physical::PhysicalChainStep::Index(idx) => {
                Some(crate::ir::physical::PhysicalPathStep::Index(*idx))
            }
            crate::ir::physical::PhysicalChainStep::DynIndex(_) => None,
        })
        .collect()
}
