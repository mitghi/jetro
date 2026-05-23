use crate::builtins::registry::{
    array_selector as builtin_array_selector, by_name as builtin_by_name,
    direct_view_call, logical_shape, raw_json_scalar_call, BuiltinDirectViewCall, BuiltinId,
};
use crate::builtins::BuiltinLogicalShape;
use crate::data::value::Val;
use crate::ir::physical::{PhysicalPathStep, PlanNode, QueryPlan};
use crate::parse::ast::{Arg, BinOp, Expr, Step};
use crate::plan::physical::{plan_ast_with_context, PlanningContext};
use crate::JetroEngine;
use std::sync::Arc;

/// Planner-side description of NDJSON row work that can run directly on
/// simd-json tape scratch. Execution stays in `ndjson.rs`; this module owns
/// only physical-plan recognition and compact metadata for the row runner.
pub(super) type NdjsonPhysicalPath = Vec<PhysicalPathStep>;

pub(super) fn physical_paths_equal(a: &[PhysicalPathStep], b: &[PhysicalPathStep]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(a, b)| match (a, b) {
            (PhysicalPathStep::Field(a), PhysicalPathStep::Field(b)) => a == b,
            (PhysicalPathStep::Index(a), PhysicalPathStep::Index(b)) => a == b,
            _ => false,
        })
}

#[derive(Clone)]
pub(super) enum NdjsonDirectBytePlan {
    Expr(NdjsonDirectByteExpr),
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum NdjsonDirectPlanKind {
    ByteExpr,
    TapeRootPath,
    TapeScalarCall,
    TapeArrayElement,
    TapeObjectItems,
    TapeStreamCollect,
    TapeStreamFirst,
    TapeStreamLast,
    TapeStreamCount,
    TapeStreamNumeric,
    TapeStreamExtreme,
    TapeObjectProjection,
    TapeArrayProjection,
    TapeViewPipeline,
}

#[cfg(test)]
impl NdjsonDirectBytePlan {
    pub(super) fn kind(&self) -> NdjsonDirectPlanKind {
        match self {
            Self::Expr(_) => NdjsonDirectPlanKind::ByteExpr,
        }
    }
}

#[cfg(test)]
impl NdjsonDirectTapePlan {
    pub(super) fn kind(&self) -> NdjsonDirectPlanKind {
        match self {
            Self::RootPath(_) => NdjsonDirectPlanKind::TapeRootPath,
            Self::ViewScalarCall { .. } | Self::ArrayElementViewScalarCall { .. } => {
                NdjsonDirectPlanKind::TapeScalarCall
            }
            Self::ArrayElementPath { .. } => NdjsonDirectPlanKind::TapeArrayElement,
            Self::ObjectItems { .. } => NdjsonDirectPlanKind::TapeObjectItems,
            Self::Stream(stream) => match &stream.sink {
                NdjsonDirectStreamSink::Collect(_) => NdjsonDirectPlanKind::TapeStreamCollect,
                NdjsonDirectStreamSink::First(_) => NdjsonDirectPlanKind::TapeStreamFirst,
                NdjsonDirectStreamSink::Last(_) => NdjsonDirectPlanKind::TapeStreamLast,
                NdjsonDirectStreamSink::Count => NdjsonDirectPlanKind::TapeStreamCount,
                NdjsonDirectStreamSink::Numeric { .. } => NdjsonDirectPlanKind::TapeStreamNumeric,
                NdjsonDirectStreamSink::Extreme { .. } => NdjsonDirectPlanKind::TapeStreamExtreme,
            },
            Self::Object(_) => NdjsonDirectPlanKind::TapeObjectProjection,
            Self::Array(_) => NdjsonDirectPlanKind::TapeArrayProjection,
            Self::ViewPipeline { .. } => NdjsonDirectPlanKind::TapeViewPipeline,
        }
    }
}

#[derive(Clone)]
pub(super) enum NdjsonDirectByteExpr {
    Path(NdjsonPhysicalPath),
    ScalarCall {
        value: Box<NdjsonDirectByteExpr>,
        call: crate::builtins::BuiltinCall,
    },
    ObjectItems {
        path: NdjsonPhysicalPath,
        method: crate::builtins::BuiltinMethod,
    },
    ArrayElementPath {
        source_steps: NdjsonPhysicalPath,
        element: NdjsonDirectElement,
        suffix_steps: NdjsonPhysicalPath,
    },
}

#[derive(Clone)]
pub(super) enum NdjsonDirectTapePlan {
    RootPath(NdjsonPhysicalPath),
    ViewScalarCall {
        steps: NdjsonPhysicalPath,
        call: crate::builtins::BuiltinCall,
        optional: bool,
    },
    ArrayElementViewScalarCall {
        source_steps: NdjsonPhysicalPath,
        element: NdjsonDirectElement,
        suffix_steps: NdjsonPhysicalPath,
        call: crate::builtins::BuiltinCall,
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
    Stream(NdjsonDirectStreamPlan),
    Object(Vec<NdjsonDirectObjectField>),
    Array(Vec<NdjsonDirectProjectionValue>),
    ViewPipeline {
        source_steps: NdjsonPhysicalPath,
        body: crate::exec::pipeline::PipelineBody,
    },
}

#[derive(Clone)]
pub(super) struct NdjsonDirectStreamPlan {
    pub(super) source_steps: NdjsonPhysicalPath,
    pub(super) predicate: Option<NdjsonDirectItemPredicate>,
    pub(super) sink: NdjsonDirectStreamSink,
}

#[derive(Clone)]
pub(super) enum NdjsonDirectStreamMap {
    Value(NdjsonDirectProjectionValue),
    Array(Vec<NdjsonDirectProjectionValue>),
    Object(Vec<NdjsonDirectObjectField>),
}

impl NdjsonDirectStreamMap {
    pub(super) fn value_path(self) -> Option<NdjsonPhysicalPath> {
        match self {
            Self::Value(value) => value.into_path(),
            Self::Array(_) | Self::Object(_) => None,
        }
    }

    pub(super) fn values(&self) -> NdjsonDirectStreamMapValues<'_> {
        NdjsonDirectStreamMapValues {
            inner: match self {
                Self::Value(value) => NdjsonDirectStreamMapValuesInner::One(Some(value)),
                Self::Array(items) => NdjsonDirectStreamMapValuesInner::Array(items.iter()),
                Self::Object(fields) => NdjsonDirectStreamMapValuesInner::Object(fields.iter()),
            },
        }
    }
}

pub(super) struct NdjsonDirectStreamMapValues<'a> {
    inner: NdjsonDirectStreamMapValuesInner<'a>,
}

enum NdjsonDirectStreamMapValuesInner<'a> {
    One(Option<&'a NdjsonDirectProjectionValue>),
    Array(std::slice::Iter<'a, NdjsonDirectProjectionValue>),
    Object(std::slice::Iter<'a, NdjsonDirectObjectField>),
}

impl<'a> Iterator for NdjsonDirectStreamMapValues<'a> {
    type Item = &'a NdjsonDirectProjectionValue;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            NdjsonDirectStreamMapValuesInner::One(value) => value.take(),
            NdjsonDirectStreamMapValuesInner::Array(items) => items.next(),
            NdjsonDirectStreamMapValuesInner::Object(fields) => fields.next().map(|field| &field.value),
        }
    }
}

#[derive(Clone)]
pub(super) enum NdjsonDirectStreamSink {
    Collect(NdjsonDirectStreamMap),
    First(NdjsonDirectStreamMap),
    Last(NdjsonDirectStreamMap),
    Count,
    Numeric {
        suffix_steps: NdjsonPhysicalPath,
        op: crate::exec::pipeline::NumOp,
    },
    Extreme {
        key_steps: NdjsonPhysicalPath,
        op: crate::builtins::BuiltinArgExtremeSink,
        value: NdjsonDirectProjectionValue,
    },
}

#[cfg(test)]
pub(super) fn direct_byte_plan(engine: &JetroEngine, query: &str) -> Option<NdjsonDirectBytePlan> {
    rootless_ndjson_query(query)
        .and_then(|query| direct_byte_plan_inner(engine, query))
        .or_else(|| direct_byte_plan_inner(engine, query))
}

pub(super) fn direct_writer_plans(
    engine: &JetroEngine,
    query: &str,
) -> Option<(Option<NdjsonDirectBytePlan>, NdjsonDirectTapePlan)> {
    rootless_ndjson_query(query)
        .and_then(|query| direct_writer_plans_inner(engine, query))
        .or_else(|| direct_writer_plans_inner(engine, query))
}

#[cfg(test)]
pub(super) fn direct_writer_plan_kind(
    engine: &JetroEngine,
    query: &str,
) -> Option<(Option<NdjsonDirectPlanKind>, NdjsonDirectPlanKind)> {
    let (byte, tape) = direct_writer_plans(engine, query)?;
    Some((byte.as_ref().map(NdjsonDirectBytePlan::kind), tape.kind()))
}

fn direct_writer_plans_inner(
    engine: &JetroEngine,
    query: &str,
) -> Option<(Option<NdjsonDirectBytePlan>, NdjsonDirectTapePlan)> {
    let plan = engine.cached_plan(query, PlanningContext::bytes());
    let tape = direct_tape_plan_from_plan(&plan)?;
    let byte = direct_byte_plan_from_plan(&plan);
    Some((byte, tape))
}

#[cfg(test)]
fn direct_byte_plan_inner(engine: &JetroEngine, query: &str) -> Option<NdjsonDirectBytePlan> {
    let plan = engine.cached_plan(query, PlanningContext::bytes());
    direct_byte_plan_from_plan(&plan)
}

fn direct_byte_plan_from_plan(plan: &QueryPlan) -> Option<NdjsonDirectBytePlan> {
    use crate::ir::physical::QueryRoot;

    let QueryRoot::Node(root) = plan.root() else {
        return None;
    };
    match plan.node(*root) {
        PlanNode::Chain { base, steps } => {
            let (source_steps, element) = direct_array_element_source(&plan, *base)?;
            if !byte_path_has_root_field(&source_steps) {
                return None;
            }
            Some(NdjsonDirectBytePlan::Expr(
                NdjsonDirectByteExpr::ArrayElementPath {
                    source_steps,
                    element,
                    suffix_steps: physical_chain_to_path(steps)?,
                },
            ))
        }
        PlanNode::RootPath(steps) => {
            if !byte_path_has_root_field(steps) {
                return None;
            }
            Some(NdjsonDirectBytePlan::Expr(NdjsonDirectByteExpr::Path(
                steps.clone(),
            )))
        }
        PlanNode::Call {
            receiver,
            call,
            optional,
        } if !*optional && is_direct_raw_json_scalar_call(call) =>
        {
            let value = direct_byte_expr_from_receiver(&plan, *receiver)?;
            Some(NdjsonDirectBytePlan::Expr(
                NdjsonDirectByteExpr::ScalarCall {
                    value: Box::new(value),
                    call: call.clone(),
                },
            ))
        }
        PlanNode::Call {
            receiver,
            call,
            optional,
        } if !*optional && is_direct_object_items_call(call) =>
        {
            let steps = root_path_steps(&plan, *receiver)?;
            byte_path_has_root_field(&steps)
                .then_some(())
                .or_else(|| byte_path_is_root(&steps).then_some(()))?;
            Some(NdjsonDirectBytePlan::Expr(
                NdjsonDirectByteExpr::ObjectItems {
                    path: steps,
                    method: call.method,
                },
            ))
        }
        PlanNode::Pipeline {
            source,
            body,
        } => direct_byte_plain_sink_pipeline_plan(plan, source, body),
        _ => {
            if let Some((source_steps, element)) = direct_array_element_source(&plan, *root) {
                if !byte_path_has_root_field(&source_steps) {
                    return None;
                }
                return Some(NdjsonDirectBytePlan::Expr(
                    NdjsonDirectByteExpr::ArrayElementPath {
                        source_steps,
                        element,
                        suffix_steps: Vec::new(),
                    },
                ));
            }
            None
        }
    }
}

fn direct_byte_plain_sink_pipeline_plan(
    plan: &QueryPlan,
    source: &crate::ir::physical::PipelinePlanSource,
    body: &crate::exec::pipeline::PipelineBody,
) -> Option<NdjsonDirectBytePlan> {
    let call = plain_sink_direct_raw_json_scalar_call(body)?;
    let steps = pipeline_source_to_steps(plan, source)?;
    if !byte_path_has_root_field(&steps) {
        return None;
    }
    Some(NdjsonDirectBytePlan::Expr(
        NdjsonDirectByteExpr::ScalarCall {
            value: Box::new(NdjsonDirectByteExpr::Path(steps)),
            call,
        },
    ))
}

fn byte_path_has_root_field(steps: &[PhysicalPathStep]) -> bool {
    matches!(steps.first(), Some(PhysicalPathStep::Field(_)))
}

fn byte_path_is_root(steps: &[PhysicalPathStep]) -> bool {
    steps.is_empty()
}

fn direct_byte_expr_from_receiver(
    plan: &QueryPlan,
    receiver: crate::ir::physical::NodeId,
) -> Option<NdjsonDirectByteExpr> {
    if let Some(steps) = root_path_steps(plan, receiver) {
        return byte_path_has_root_field(&steps).then_some(NdjsonDirectByteExpr::Path(steps));
    }

    let PlanNode::Chain { base, steps } = plan.node(receiver) else {
        return None;
    };
    let (source_steps, element) = direct_array_element_source(plan, *base)?;
    byte_path_has_root_field(&source_steps).then_some(NdjsonDirectByteExpr::ArrayElementPath {
        source_steps,
        element,
        suffix_steps: physical_chain_to_path(steps)?,
    })
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
    Nested(Box<NdjsonDirectTapePlan>),
    Literal(Val),
}

impl NdjsonDirectProjectionValue {
    pub(super) fn path_projection(&self) -> Option<NdjsonDirectPathProjection<'_>> {
        match self {
            Self::Path(steps) => Some(NdjsonDirectPathProjection::Raw(steps)),
            Self::ViewScalarCall {
                steps,
                call,
                optional,
            } => Some(NdjsonDirectPathProjection::Scalar {
                steps,
                call,
                optional: *optional,
            }),
            Self::Nested(_) | Self::Literal(_) => None,
        }
    }

    pub(super) fn path_steps(&self) -> Option<&[PhysicalPathStep]> {
        self.path_projection().map(|projection| projection.steps())
    }

    pub(super) fn into_path(self) -> Option<NdjsonPhysicalPath> {
        match self {
            Self::Path(steps) => Some(steps),
            Self::ViewScalarCall { .. } | Self::Nested(_) | Self::Literal(_) => None,
        }
    }
}

pub(super) enum NdjsonDirectPathProjection<'a> {
    Raw(&'a [PhysicalPathStep]),
    Scalar {
        steps: &'a [PhysicalPathStep],
        call: &'a crate::builtins::BuiltinCall,
        optional: bool,
    },
}

impl<'a> NdjsonDirectPathProjection<'a> {
    pub(super) fn steps(&self) -> &'a [PhysicalPathStep] {
        match self {
            Self::Raw(steps) | Self::Scalar { steps, .. } => steps,
        }
    }

    pub(super) fn scalar_call(&self) -> Option<&'a crate::builtins::BuiltinCall> {
        match self {
            Self::Raw(_) => None,
            Self::Scalar { call, .. } => Some(call),
        }
    }
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
    ArrayAny {
        source_steps: NdjsonPhysicalPath,
        predicate: NdjsonDirectItemPredicate,
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

pub(super) fn direct_cmp_literal_predicate(
    predicate: &NdjsonDirectPredicate,
) -> Option<(NdjsonPhysicalPath, BinOp, Val)> {
    let NdjsonDirectPredicate::Binary { lhs, op, rhs } = predicate else {
        return None;
    };
    if !op.is_scalar_comparison() {
        return None;
    }
    match (lhs.as_ref(), rhs.as_ref()) {
        (NdjsonDirectPredicate::Path(steps), NdjsonDirectPredicate::Literal(lit)) => {
            Some((steps.clone(), *op, lit.clone()))
        }
        (NdjsonDirectPredicate::Literal(lit), NdjsonDirectPredicate::Path(steps)) => {
            Some((steps.clone(), op.flipped_comparison()?, lit.clone()))
        }
        _ => None,
    }
}

pub(super) fn direct_tape_plan(engine: &JetroEngine, query: &str) -> Option<NdjsonDirectTapePlan> {
    rootless_ndjson_query(query)
        .and_then(|query| direct_tape_plan_inner(engine, query))
        .or_else(|| direct_tape_plan_inner(engine, query))
}

pub(super) fn direct_tape_plan_for_expr(expr: &Expr) -> Option<NdjsonDirectTapePlan> {
    if let Some(steps) = direct_tape_row_path_for_expr(expr) {
        return Some(NdjsonDirectTapePlan::RootPath(steps));
    }
    let plan = plan_ast_with_context(expr.clone(), PlanningContext::bytes());
    direct_tape_plan_from_plan(&plan)
}

pub(super) fn direct_tape_root_path_for_expr(expr: &Expr) -> Option<NdjsonPhysicalPath> {
    direct_root_path_expr(expr)
}

pub(super) fn direct_tape_row_path_for_expr(expr: &Expr) -> Option<NdjsonPhysicalPath> {
    direct_item_path_expr(expr).or_else(|| direct_tape_root_path_for_expr(expr))
}

fn direct_root_path_expr(expr: &Expr) -> Option<NdjsonPhysicalPath> {
    let Expr::Chain(base, steps) = expr else {
        return None;
    };
    if !matches!(base.as_ref(), Expr::Root) {
        return None;
    }
    steps
        .iter()
        .map(|step| match step {
            Step::Field(key) | Step::OptField(key) => {
                Some(PhysicalPathStep::Field(Arc::from(key.as_str())))
            }
            Step::Index(index) => Some(PhysicalPathStep::Index(*index)),
            _ => None,
        })
        .collect()
}

fn direct_tape_plan_inner(engine: &JetroEngine, query: &str) -> Option<NdjsonDirectTapePlan> {
    let plan = engine.cached_plan(query, PlanningContext::bytes());
    direct_tape_plan_from_plan(&plan)
}

fn direct_tape_plan_from_plan(plan: &QueryPlan) -> Option<NdjsonDirectTapePlan> {
    use crate::ir::physical::QueryRoot;

    let QueryRoot::Node(root) = plan.root() else {
        return None;
    };
    direct_tape_plan_for_node(plan, *root)
}

fn direct_tape_plan_for_node(
    plan: &QueryPlan,
    id: crate::ir::physical::NodeId,
) -> Option<NdjsonDirectTapePlan> {
    if let PlanNode::Chain { base, steps } = plan.node(id) {
        if let Some(plan) =
            direct_tape_sort_extreme_plan_for_node(plan, *base, physical_chain_to_path(steps)?)
        {
            return Some(plan);
        }
        let (source_steps, element) = direct_array_element_source(plan, *base)?;
        return Some(NdjsonDirectTapePlan::ArrayElementPath {
            source_steps,
            element,
            suffix_steps: physical_chain_to_path(steps)?,
        });
    }
    if let Some((source_steps, element)) = direct_array_element_source(plan, id) {
        return Some(NdjsonDirectTapePlan::ArrayElementPath {
            source_steps,
            element,
            suffix_steps: Vec::new(),
        });
    }
    match plan.node(id) {
        PlanNode::RootPath(steps) => Some(NdjsonDirectTapePlan::RootPath(steps.clone())),
        PlanNode::Call {
            receiver,
            call,
            optional,
        } if is_direct_view_scalar_call(call) =>
        {
            if let Some(steps) = node_path_steps(plan, *receiver) {
                return Some(NdjsonDirectTapePlan::ViewScalarCall {
                    steps,
                    call: call.clone(),
                    optional: *optional,
                });
            }
            let PlanNode::Chain { base, steps } = plan.node(*receiver) else {
                return None;
            };
            let (source_steps, element) = direct_array_element_source(plan, *base)?;
            Some(NdjsonDirectTapePlan::ArrayElementViewScalarCall {
                source_steps,
                element,
                suffix_steps: physical_chain_to_path(steps)?,
                call: call.clone(),
            })
        }
        PlanNode::Call {
            receiver,
            call,
            optional,
        } if !*optional && is_direct_object_items_call(call) =>
        {
            Some(NdjsonDirectTapePlan::ObjectItems {
                steps: node_path_steps(plan, *receiver)?,
                method: call.method,
            })
        }
        PlanNode::Pipeline { source, body } => {
            if let Some(plan) = direct_tape_plain_sink_pipeline_plan(plan, source, body) {
                return Some(plan);
            }
            if let Some(plan) = direct_tape_sort_extreme_plan(plan, source, body, Vec::new()) {
                return Some(plan);
            }
            if let Some(plan) = direct_tape_numeric_stream_plan(plan, source, body) {
                return Some(plan);
            }
            if let Some(plan) = direct_tape_count_stream_plan(plan, source, body) {
                return Some(plan);
            }
            if let Some(plan) = direct_tape_positional_stream_plan(plan, source, body) {
                return Some(plan);
            }
            if let Some(plan) = direct_tape_collect_stream_plan(plan, source, body) {
                return Some(plan);
            }
            if !body.can_run_with_view() {
                return None;
            }
            Some(NdjsonDirectTapePlan::ViewPipeline {
                source_steps: pipeline_source_to_steps(plan, source)?,
                body: body.clone(),
            })
        }
        PlanNode::Object(fields) => direct_tape_object_plan(plan, fields),
        PlanNode::Array(elems) => direct_tape_array_plan(plan, elems),
        _ => None,
    }
}

fn direct_tape_plain_sink_pipeline_plan(
    plan: &QueryPlan,
    source: &crate::ir::physical::PipelinePlanSource,
    body: &crate::exec::pipeline::PipelineBody,
) -> Option<NdjsonDirectTapePlan> {
    let call = plain_sink_direct_scalar_call(body)?;
    Some(NdjsonDirectTapePlan::ViewScalarCall {
        steps: pipeline_source_to_steps(plan, source)?,
        call,
        optional: false,
    })
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
        } if is_direct_view_scalar_call(call) => {
            Some(NdjsonDirectProjectionValue::ViewScalarCall {
                steps: node_path_steps(plan, *receiver)?,
                call: call.clone(),
                optional: *optional,
            })
        }
        PlanNode::Literal(value) => Some(NdjsonDirectProjectionValue::Literal(value.clone())),
        _ => direct_tape_plan_for_node(plan, id)
            .map(Box::new)
            .map(NdjsonDirectProjectionValue::Nested),
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

fn direct_tape_sort_extreme_plan_for_node(
    plan: &QueryPlan,
    id: crate::ir::physical::NodeId,
    suffix_steps: NdjsonPhysicalPath,
) -> Option<NdjsonDirectTapePlan> {
    if let PlanNode::Call {
        receiver,
        call,
        optional,
    } = plan.node(id)
    {
        if *optional {
            return None;
        }
        let want_last =
            crate::exec::pipeline::Sink::terminal_builtin(call.method)?.select_one_wants_last()?;
        let PlanNode::Pipeline { source, body } = plan.node(*receiver) else {
            return None;
        };
        return direct_tape_sort_extreme_plan_with_position(
            plan,
            source,
            body,
            suffix_steps,
            want_last,
        );
    }
    let PlanNode::Pipeline { source, body } = plan.node(id) else {
        return None;
    };
    direct_tape_sort_extreme_plan(plan, source, body, suffix_steps)
}

fn direct_tape_sort_extreme_plan(
    plan: &QueryPlan,
    source: &crate::ir::physical::PipelinePlanSource,
    body: &crate::exec::pipeline::PipelineBody,
    suffix_steps: NdjsonPhysicalPath,
) -> Option<NdjsonDirectTapePlan> {
    use crate::exec::pipeline::Stage;

    let [Stage::Sort(_)] = body.stages.as_slice() else {
        return None;
    };
    let want_last = body.sink.select_one_wants_last()?;
    direct_tape_sort_extreme_plan_with_position(plan, source, body, suffix_steps, want_last)
}

fn direct_tape_sort_extreme_plan_with_position(
    plan: &QueryPlan,
    source: &crate::ir::physical::PipelinePlanSource,
    body: &crate::exec::pipeline::PipelineBody,
    suffix_steps: NdjsonPhysicalPath,
    want_last: bool,
) -> Option<NdjsonDirectTapePlan> {
    use crate::exec::pipeline::Stage;

    let [Stage::Sort(sort)] = body.stages.as_slice() else {
        return None;
    };
    let key_steps = kernel_to_physical_path(body.stage_kernels.first()?)?;
    let op = if want_last ^ sort.descending {
        crate::builtins::BuiltinArgExtremeSink::MaxBy
    } else {
        crate::builtins::BuiltinArgExtremeSink::MinBy
    };
    Some(NdjsonDirectTapePlan::Stream(NdjsonDirectStreamPlan {
        source_steps: pipeline_source_to_steps(plan, source)?,
        predicate: None,
        sink: NdjsonDirectStreamSink::Extreme {
            key_steps,
            op,
            value: NdjsonDirectProjectionValue::Path(suffix_steps),
        },
    }))
}

fn pipeline_source_to_steps(
    plan: &QueryPlan,
    source: &crate::ir::physical::PipelinePlanSource,
) -> Option<NdjsonPhysicalPath> {
    match source {
        crate::ir::physical::PipelinePlanSource::FieldChain { keys } => Some(keys_to_path(keys)),
        crate::ir::physical::PipelinePlanSource::Expr(source) => node_path_steps(plan, *source),
    }
}

fn plain_sink_direct_scalar_call(
    body: &crate::exec::pipeline::PipelineBody,
) -> Option<crate::builtins::BuiltinCall> {
    if !body.stages.is_empty() {
        return None;
    }
    body.sink.scalar_call_for_plain_sink()
}

fn plain_sink_direct_raw_json_scalar_call(
    body: &crate::exec::pipeline::PipelineBody,
) -> Option<crate::builtins::BuiltinCall> {
    let call = plain_sink_direct_scalar_call(body)?;
    is_direct_raw_json_scalar_call(&call).then_some(call)
}

#[inline]
fn is_direct_view_scalar_call(call: &crate::builtins::BuiltinCall) -> bool {
    direct_view_call(BuiltinId::from_method(call.method), &call.args)
        == Some(BuiltinDirectViewCall::ScalarValue)
}

#[inline]
fn is_direct_raw_json_scalar_call(call: &crate::builtins::BuiltinCall) -> bool {
    raw_json_scalar_call(BuiltinId::from_method(call.method), &call.args)
}

#[inline]
fn is_direct_object_items_call(call: &crate::builtins::BuiltinCall) -> bool {
    matches!(
        direct_view_call(BuiltinId::from_method(call.method), &call.args),
        Some(BuiltinDirectViewCall::ObjectItems(_))
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
    rootless_ndjson_query(predicate)
        .and_then(|query| direct_tape_predicate_inner(engine, query))
        .or_else(|| direct_tape_predicate_inner(engine, predicate))
}

pub(super) fn direct_tape_predicate_for_expr(expr: &Expr) -> Option<NdjsonDirectPredicate> {
    if let Some(predicate) = direct_array_any_predicate_expr(expr) {
        return Some(predicate);
    }
    if let Some(steps) = direct_tape_row_path_for_expr(expr) {
        return Some(NdjsonDirectPredicate::Path(steps));
    }
    let plan = plan_ast_with_context(expr.clone(), PlanningContext::bytes());
    direct_tape_predicate_from_plan(&plan)
}

pub(super) fn direct_tape_predicates_for_exprs<'a>(
    exprs: impl IntoIterator<Item = &'a Expr>,
) -> Option<Vec<NdjsonDirectPredicate>> {
    exprs
        .into_iter()
        .map(direct_tape_predicate_for_expr)
        .collect()
}

fn direct_array_any_predicate_expr(expr: &Expr) -> Option<NdjsonDirectPredicate> {
    let Expr::Chain(base, steps) = expr else {
        return None;
    };
    let (last, prefix) = steps.split_last()?;
    let Step::Method(name, args) = last else {
        return None;
    };
    if logical_shape(builtin_by_name(name.as_str())?) != Some(BuiltinLogicalShape::FilterThenFirst)
    {
        return None;
    }
    let [Arg::Pos(predicate_expr)] = args.as_slice() else {
        return None;
    };
    let mut source_steps = direct_tape_row_path_for_expr(base)?;
    source_steps.extend(physical_path_from_steps(prefix)?);
    let predicate = direct_item_predicate_from_expr(predicate_expr)?;
    if !item_predicate_guarantees_object_match(&predicate) {
        return None;
    }
    Some(NdjsonDirectPredicate::ArrayAny {
        source_steps,
        predicate,
    })
}

fn direct_item_predicate_from_expr(expr: &Expr) -> Option<NdjsonDirectItemPredicate> {
    let kernel = crate::exec::pipeline::BodyKernel::classify_expr(expr);
    if !matches!(kernel, crate::exec::pipeline::BodyKernel::Generic) {
        if let Some(predicate) = direct_item_predicate_from_kernel(&kernel) {
            return Some(predicate);
        }
    }
    match expr {
        Expr::Bool(value) => Some(NdjsonDirectItemPredicate::Literal(Val::Bool(*value))),
        Expr::Null => Some(NdjsonDirectItemPredicate::Literal(Val::Null)),
        Expr::Int(value) => Some(NdjsonDirectItemPredicate::Literal(Val::Int(*value))),
        Expr::Float(value) => Some(NdjsonDirectItemPredicate::Literal(Val::Float(*value))),
        Expr::Str(value) => Some(NdjsonDirectItemPredicate::Literal(Val::Str(Arc::from(
            value.as_str(),
        )))),
        Expr::Current | Expr::Ident(_) | Expr::Chain(_, _) => direct_item_path_expr(expr)
            .map(NdjsonDirectItemPredicate::Path),
        Expr::BinOp(lhs, op @ (BinOp::And | BinOp::Or), rhs) => {
            Some(NdjsonDirectItemPredicate::Binary {
                lhs: Box::new(direct_item_predicate_from_expr(lhs)?),
                op: *op,
                rhs: Box::new(direct_item_predicate_from_expr(rhs)?),
            })
        }
        Expr::BinOp(lhs, op, rhs) if op.is_scalar_comparison() => {
            if let (Some(lhs), Some(lit)) = (direct_item_path_expr(lhs), literal_val_expr(rhs)) {
                return Some(NdjsonDirectItemPredicate::CmpLit { lhs, op: *op, lit });
            }
            if let (Some(lit), Some(lhs)) = (literal_val_expr(lhs), direct_item_path_expr(rhs)) {
                return Some(NdjsonDirectItemPredicate::CmpLit {
                    lhs,
                    op: op.flipped_comparison()?,
                    lit,
                });
            }
            None
        }
        _ => None,
    }
}

fn direct_item_path_expr(expr: &Expr) -> Option<NdjsonPhysicalPath> {
    match expr {
        Expr::Current => Some(Vec::new()),
        Expr::Ident(name) => Some(vec![PhysicalPathStep::Field(Arc::from(name.as_str()))]),
        Expr::Chain(base, steps) => {
            let mut path = match base.as_ref() {
                Expr::Current => Vec::new(),
                Expr::Ident(name) => vec![PhysicalPathStep::Field(Arc::from(name.as_str()))],
                _ => return None,
            };
            path.extend(physical_path_from_steps(steps)?);
            Some(path)
        }
        _ => None,
    }
}

fn physical_path_from_steps(steps: &[Step]) -> Option<NdjsonPhysicalPath> {
    steps
        .iter()
        .map(|step| match step {
            Step::Field(name) | Step::OptField(name) => {
                Some(PhysicalPathStep::Field(Arc::from(name.as_str())))
            }
            Step::Index(index) => Some(PhysicalPathStep::Index(*index)),
            _ => None,
        })
        .collect()
}

fn literal_val_expr(expr: &Expr) -> Option<Val> {
    match expr {
        Expr::Null => Some(Val::Null),
        Expr::Bool(value) => Some(Val::Bool(*value)),
        Expr::Int(value) => Some(Val::Int(*value)),
        Expr::Float(value) => Some(Val::Float(*value)),
        Expr::Str(value) => Some(Val::Str(Arc::from(value.as_str()))),
        _ => None,
    }
}

fn item_predicate_guarantees_object_match(predicate: &NdjsonDirectItemPredicate) -> bool {
    match predicate {
        NdjsonDirectItemPredicate::CmpLit { lhs, .. }
        | NdjsonDirectItemPredicate::Path(lhs)
        | NdjsonDirectItemPredicate::ViewScalarCall {
            suffix_steps: lhs, ..
        } => !lhs.is_empty(),
        NdjsonDirectItemPredicate::Binary { lhs, op, rhs } if *op == BinOp::And => {
            item_predicate_guarantees_object_match(lhs)
                || item_predicate_guarantees_object_match(rhs)
        }
        NdjsonDirectItemPredicate::Binary { lhs, op, rhs } if *op == BinOp::Or => {
            item_predicate_guarantees_object_match(lhs)
                && item_predicate_guarantees_object_match(rhs)
        }
        _ => false,
    }
}

fn direct_tape_predicate_inner(
    engine: &JetroEngine,
    predicate: &str,
) -> Option<NdjsonDirectPredicate> {
    let plan = engine.cached_plan(predicate, PlanningContext::bytes());
    direct_tape_predicate_from_plan(&plan)
}

fn direct_tape_predicate_from_plan(plan: &QueryPlan) -> Option<NdjsonDirectPredicate> {
    let crate::ir::physical::QueryRoot::Node(root) = plan.root() else {
        return None;
    };
    direct_tape_predicate_node(plan, *root)
}

fn rootless_ndjson_query(query: &str) -> Option<&str> {
    query.strip_prefix("$.").filter(|query| {
        query
            .as_bytes()
            .first()
            .is_some_and(|byte| byte.is_ascii_alphabetic() || *byte == b'_')
    })
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
        } if !*optional && is_direct_view_scalar_call(call) => {
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

fn direct_tape_count_stream_plan(
    plan: &QueryPlan,
    source: &crate::ir::physical::PipelinePlanSource,
    body: &crate::exec::pipeline::PipelineBody,
) -> Option<NdjsonDirectTapePlan> {
    if !body.sink.is_plain_count_reducer() {
        return None;
    }
    let stream = direct_stream_shape(body)?;
    if stream.map.is_some() {
        return None;
    }
    Some(NdjsonDirectTapePlan::Stream(NdjsonDirectStreamPlan {
        source_steps: pipeline_source_to_steps(plan, source)?,
        predicate: stream.predicate,
        sink: NdjsonDirectStreamSink::Count,
    }))
}

fn direct_tape_numeric_stream_plan(
    plan: &QueryPlan,
    source: &crate::ir::physical::PipelinePlanSource,
    body: &crate::exec::pipeline::PipelineBody,
) -> Option<NdjsonDirectTapePlan> {
    let stream = direct_stream_shape(body)?;
    let (op, predicate, suffix_steps) = if let Some((_projection, op)) =
        body.sink.projected_numeric_reducer()
    {
        if stream.map.is_some() {
            return None;
        }
        (
            op,
            stream.predicate,
            kernel_to_physical_path(body.sink_kernels.first()?)?,
        )
    } else if let Some(op) = body.sink.identity_numeric_reducer() {
        let suffix_steps = direct_stream_map_path(stream.map?)?;
        (op, stream.predicate, suffix_steps)
    } else {
        return None;
    };
    Some(NdjsonDirectTapePlan::Stream(NdjsonDirectStreamPlan {
        source_steps: pipeline_source_to_steps(plan, source)?,
        predicate,
        sink: NdjsonDirectStreamSink::Numeric { suffix_steps, op },
    }))
}

fn direct_tape_collect_stream_plan(
    plan: &QueryPlan,
    source: &crate::ir::physical::PipelinePlanSource,
    body: &crate::exec::pipeline::PipelineBody,
) -> Option<NdjsonDirectTapePlan> {
    use crate::exec::pipeline::Sink;

    if !matches!(body.sink, Sink::Collect) {
        return None;
    }
    let stream = direct_stream_shape(body)?;
    let Some(map) = stream.map else {
        return None;
    };
    Some(NdjsonDirectTapePlan::Stream(NdjsonDirectStreamPlan {
        source_steps: pipeline_source_to_steps(plan, source)?,
        predicate: stream.predicate,
        sink: NdjsonDirectStreamSink::Collect(map),
    }))
}

fn direct_tape_positional_stream_plan(
    plan: &QueryPlan,
    source: &crate::ir::physical::PipelinePlanSource,
    body: &crate::exec::pipeline::PipelineBody,
) -> Option<NdjsonDirectTapePlan> {
    let want_last = body.sink.select_one_wants_last()?;
    let stream = direct_stream_shape(body)?;
    let map = stream
        .map
        .unwrap_or(NdjsonDirectStreamMap::Value(NdjsonDirectProjectionValue::Path(
            Vec::new(),
        )));
    Some(NdjsonDirectTapePlan::Stream(NdjsonDirectStreamPlan {
        source_steps: pipeline_source_to_steps(plan, source)?,
        predicate: stream.predicate,
        sink: if want_last {
            NdjsonDirectStreamSink::Last(map)
        } else {
            NdjsonDirectStreamSink::First(map)
        },
    }))
}

struct DirectStreamShape {
    predicate: Option<NdjsonDirectItemPredicate>,
    map: Option<NdjsonDirectStreamMap>,
}

fn direct_stream_shape(
    body: &crate::exec::pipeline::PipelineBody,
) -> Option<DirectStreamShape> {
    let mut predicate = None;
    let mut map = None;
    for (idx, stage) in body.stages.iter().enumerate() {
        if stage.is_symbolic_filter_stage() && map.is_none() {
            let next = direct_item_predicate_from_kernel(body.stage_kernels.get(idx)?)?;
            predicate = Some(combine_direct_item_predicate(predicate, next));
        } else if stage.is_symbolic_map_stage() && map.is_none() && idx + 1 == body.stages.len() {
            map = Some(direct_stream_map_from_kernel(body.stage_kernels.get(idx)?)?);
        } else {
            return None;
        }
    }
    Some(DirectStreamShape { predicate, map })
}

fn combine_direct_item_predicate(
    lhs: Option<NdjsonDirectItemPredicate>,
    rhs: NdjsonDirectItemPredicate,
) -> NdjsonDirectItemPredicate {
    match lhs {
        Some(lhs) => NdjsonDirectItemPredicate::Binary {
            lhs: Box::new(lhs),
            op: crate::parse::ast::BinOp::And,
            rhs: Box::new(rhs),
        },
        None => rhs,
    }
}

fn direct_stream_map_from_kernel(
    kernel: &crate::exec::pipeline::BodyKernel,
) -> Option<NdjsonDirectStreamMap> {
    if let Some(value) = direct_projection_value_from_kernel(kernel) {
        return Some(NdjsonDirectStreamMap::Value(value));
    }
    if let crate::exec::pipeline::BodyKernel::Array(items) = kernel {
        let items = items
            .iter()
            .map(direct_projection_value_from_kernel)
            .collect::<Option<Vec<_>>>()?;
        return Some(NdjsonDirectStreamMap::Array(items));
    }
    if let crate::exec::pipeline::BodyKernel::Object(object) = kernel {
        return Some(NdjsonDirectStreamMap::Object(
            direct_object_fields_from_kernel(object)?,
        ));
    }
    None
}

fn direct_stream_map_path(map: NdjsonDirectStreamMap) -> Option<NdjsonPhysicalPath> {
    map.value_path()
}

fn direct_item_predicate_from_kernel(
    kernel: &crate::exec::pipeline::BodyKernel,
) -> Option<NdjsonDirectItemPredicate> {
    if let Some(value) = kernel.literal_value() {
        return Some(NdjsonDirectItemPredicate::Literal(value));
    }
    if let Some((keys, op, lit)) = kernel.field_path_literal_cmp() {
        return Some(NdjsonDirectItemPredicate::CmpLit {
            lhs: keys_to_path(&keys),
            op,
            lit,
        });
    }
    match kernel {
        crate::exec::pipeline::BodyKernel::Current => {
            Some(NdjsonDirectItemPredicate::Path(Vec::new()))
        }
        crate::exec::pipeline::BodyKernel::FieldRead(_)
        | crate::exec::pipeline::BodyKernel::FieldChain(_) => Some(
            NdjsonDirectItemPredicate::Path(kernel_to_physical_path(kernel)?),
        ),
        crate::exec::pipeline::BodyKernel::And(items) => {
            combine_direct_item_predicate_kernels(items, crate::parse::ast::BinOp::And)
        }
        crate::exec::pipeline::BodyKernel::Or(items) => {
            combine_direct_item_predicate_kernels(items, crate::parse::ast::BinOp::Or)
        }
        crate::exec::pipeline::BodyKernel::BuiltinCall { .. } => {
            let (suffix_steps, call) = direct_scalar_call_from_kernel(kernel)?;
            Some(NdjsonDirectItemPredicate::ViewScalarCall {
                suffix_steps,
                call,
            })
        }
        _ => None,
    }
}

fn combine_direct_item_predicate_kernels(
    items: &[crate::exec::pipeline::BodyKernel],
    op: crate::parse::ast::BinOp,
) -> Option<NdjsonDirectItemPredicate> {
    let mut iter = items.iter().map(direct_item_predicate_from_kernel);
    let mut acc = iter.next()??;
    for item in iter {
        acc = NdjsonDirectItemPredicate::Binary {
            lhs: Box::new(acc),
            op,
            rhs: Box::new(item?),
        };
    }
    Some(acc)
}

fn kernel_to_physical_path(
    kernel: &crate::exec::pipeline::BodyKernel,
) -> Option<NdjsonPhysicalPath> {
    kernel.field_path_keys().map(|keys| keys_to_path(&keys))
}

fn direct_projection_value_from_kernel(
    kernel: &crate::exec::pipeline::BodyKernel,
) -> Option<NdjsonDirectProjectionValue> {
    if let Some(value) = kernel.literal_value() {
        return Some(NdjsonDirectProjectionValue::Literal(value));
    }
    match kernel {
        crate::exec::pipeline::BodyKernel::Current
        | crate::exec::pipeline::BodyKernel::FieldRead(_)
        | crate::exec::pipeline::BodyKernel::FieldChain(_) => Some(
            NdjsonDirectProjectionValue::Path(kernel_to_physical_path(kernel)?),
        ),
        crate::exec::pipeline::BodyKernel::BuiltinCall { .. } => {
            let (steps, call) = direct_scalar_call_from_kernel(kernel)?;
            Some(NdjsonDirectProjectionValue::ViewScalarCall {
                steps,
                call,
                optional: false,
            })
        }
        _ => None,
    }
}

fn direct_scalar_call_from_kernel(
    kernel: &crate::exec::pipeline::BodyKernel,
) -> Option<(NdjsonPhysicalPath, crate::builtins::BuiltinCall)> {
    let crate::exec::pipeline::BodyKernel::BuiltinCall { receiver, call } = kernel else {
        return None;
    };
    if !is_direct_view_scalar_call(call) {
        return None;
    }
    Some((kernel_to_physical_path(receiver)?, call.clone()))
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
    use crate::exec::pipeline::{MembershipSinkTarget, Sink};

    if !body.stages.is_empty() {
        return None;
    }
    let Sink::Membership(spec) = &body.sink else {
        return None;
    };
    if !spec.is_includes() {
        return None;
    }
    let MembershipSinkTarget::Literal(target) = &spec.target else {
        return None;
    };
    let call = BuiltinCall::new(spec.method(), BuiltinArgs::Val(target.clone()));
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
    if let Some(steps) = node_path_steps(plan, receiver) {
        return Some(NdjsonDirectPredicate::ViewScalarCall { steps, call });
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
    use crate::exec::pipeline::SingleElementSelection;
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
        let arg = match &call.args {
            crate::builtins::BuiltinArgs::I64(n) => Some(*n),
            _ => None,
        };
        let selection = SingleElementSelection::from_array_selector(
            builtin_array_selector(BuiltinId::from_method(call.method))?,
            arg,
        )?;
        return Some((
            node_path_steps(plan, *receiver)?,
            direct_element_from_selection(selection),
        ));
    }

    let PlanNode::Pipeline { source, body } = plan.node(id) else {
        return None;
    };
    if !body.stages.is_empty() {
        return None;
    }
    let element = direct_element_from_selection(body.sink.single_element_selection()?);
    let source_steps = match source {
        PipelinePlanSource::FieldChain { keys } => keys_to_path(keys),
        PipelinePlanSource::Expr(source) => node_path_steps(plan, *source)?,
    };
    Some((source_steps, element))
}

fn direct_element_from_selection(
    selection: crate::exec::pipeline::SingleElementSelection,
) -> NdjsonDirectElement {
    match selection {
        crate::exec::pipeline::SingleElementSelection::First => NdjsonDirectElement::First,
        crate::exec::pipeline::SingleElementSelection::Last => NdjsonDirectElement::Last,
        crate::exec::pipeline::SingleElementSelection::Nth(n) => NdjsonDirectElement::Nth(n),
    }
}

fn node_path_steps(
    plan: &QueryPlan,
    id: crate::ir::physical::NodeId,
) -> Option<NdjsonPhysicalPath> {
    match plan.node(id) {
        PlanNode::RootPath(steps) => Some(steps.clone()),
        PlanNode::Chain { base, steps } => {
            let mut out = node_path_steps(plan, *base)?;
            out.extend(physical_chain_to_path(steps)?);
            Some(out)
        }
        _ => None,
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_element_selection_reuses_pipeline_metadata() {
        use crate::exec::pipeline::SingleElementSelection;

        assert!(matches!(
            direct_element_from_selection(SingleElementSelection::First),
            NdjsonDirectElement::First
        ));
        assert!(matches!(
            direct_element_from_selection(SingleElementSelection::Last),
            NdjsonDirectElement::Last
        ));
        assert!(matches!(
            direct_element_from_selection(SingleElementSelection::Nth(2)),
            NdjsonDirectElement::Nth(2)
        ));
    }

    #[test]
    fn physical_path_equality_compares_field_and_index_steps() {
        let a = vec![
            PhysicalPathStep::Field(Arc::from("items")),
            PhysicalPathStep::Index(0),
            PhysicalPathStep::Field(Arc::from("sku")),
        ];
        let b = vec![
            PhysicalPathStep::Field(Arc::from("items")),
            PhysicalPathStep::Index(0),
            PhysicalPathStep::Field(Arc::from("sku")),
        ];
        let c = vec![
            PhysicalPathStep::Field(Arc::from("items")),
            PhysicalPathStep::Index(1),
            PhysicalPathStep::Field(Arc::from("sku")),
        ];

        assert!(physical_paths_equal(&a, &b));
        assert!(!physical_paths_equal(&a, &c));
        assert!(!physical_paths_equal(&a, &a[..2]));
    }

    #[test]
    fn direct_tape_root_path_extracts_only_rooted_paths() {
        let path = crate::parse::parser::parse("$.items[0].sku").expect("parse rooted path");
        let steps = direct_tape_root_path_for_expr(&path).expect("root path");
        assert!(matches!(
            steps.as_slice(),
            [
                PhysicalPathStep::Field(items),
                PhysicalPathStep::Index(0),
                PhysicalPathStep::Field(sku)
            ] if items.as_ref() == "items" && sku.as_ref() == "sku"
        ));

        let current_path = crate::parse::parser::parse("@.items[0].sku").expect("parse current");
        assert!(direct_tape_root_path_for_expr(&current_path).is_none());

        let computed = crate::parse::parser::parse("$.items.map(sku)").expect("parse computed");
        assert!(direct_tape_root_path_for_expr(&computed).is_none());
    }

    #[test]
    fn direct_tape_row_path_accepts_row_local_paths() {
        let bare = crate::parse::parser::parse("sku").expect("parse bare field");
        let current = crate::parse::parser::parse("@.sku").expect("parse current field");
        let rooted = crate::parse::parser::parse("$.sku").expect("parse root field");

        for expr in [&bare, &current, &rooted] {
            let steps = direct_tape_row_path_for_expr(expr).expect("row path");
            assert!(matches!(
                steps.as_slice(),
                [PhysicalPathStep::Field(sku)] if sku.as_ref() == "sku"
            ));
        }
    }

    #[test]
    fn direct_tape_predicate_accepts_row_local_truthy_path() {
        let bare = crate::parse::parser::parse("active").expect("parse bare field");
        let current = crate::parse::parser::parse("@.active").expect("parse current field");

        for expr in [&bare, &current] {
            let predicate = direct_tape_predicate_for_expr(expr).expect("direct predicate");
            assert!(matches!(
                predicate,
                NdjsonDirectPredicate::Path(ref steps)
                    if matches!(
                        steps.as_slice(),
                        [PhysicalPathStep::Field(active)] if active.as_ref() == "active"
                    )
            ));
        }
    }

    #[test]
    fn recognizes_array_find_field_comparison_predicate() {
        let expr = crate::parse::parser::parse(r#"@.custom_attributes.find(@.value == "z")"#)
            .expect("parse");
        let Some(NdjsonDirectPredicate::ArrayAny {
            source_steps,
            predicate,
        }) = direct_array_any_predicate_expr(&expr)
        else {
            panic!("expected array-any predicate");
        };

        assert!(matches!(
            source_steps.as_slice(),
            [PhysicalPathStep::Field(name)] if name.as_ref() == "custom_attributes"
        ));
        assert!(matches!(
            predicate,
            NdjsonDirectItemPredicate::CmpLit {
                ref lhs,
                op: BinOp::Eq,
                lit: Val::Str(ref value),
            } if matches!(
                lhs.as_slice(),
                [PhysicalPathStep::Field(name)] if name.as_ref() == "value"
            ) && value.as_ref() == "z"
        ));
    }

    #[test]
    fn recognizes_array_find_bare_field_comparison_predicate() {
        let expr = crate::parse::parser::parse(r#"@.custom_attributes.find(value == "z")"#)
            .expect("parse");
        assert!(matches!(
            direct_array_any_predicate_expr(&expr),
            Some(NdjsonDirectPredicate::ArrayAny { .. })
        ));
    }

    #[test]
    fn recognizes_array_find_on_bare_row_local_source() {
        let expr = crate::parse::parser::parse(r#"custom_attributes.find(value == "z")"#)
            .expect("parse");
        let Some(NdjsonDirectPredicate::ArrayAny { source_steps, .. }) =
            direct_array_any_predicate_expr(&expr)
        else {
            panic!("expected array-any predicate");
        };
        assert!(matches!(
            source_steps.as_slice(),
            [PhysicalPathStep::Field(name)] if name.as_ref() == "custom_attributes"
        ));
    }

    #[test]
    fn recognizes_array_find_item_view_scalar_predicate_from_kernel_metadata() {
        let expr = crate::parse::parser::parse(r#"custom_attributes.find(value.len())"#)
            .expect("parse");
        let Some(NdjsonDirectPredicate::ArrayAny { predicate, .. }) =
            direct_array_any_predicate_expr(&expr)
        else {
            panic!("expected array-any predicate");
        };
        assert!(matches!(
            predicate,
            NdjsonDirectItemPredicate::ViewScalarCall {
                ref suffix_steps,
                call: crate::builtins::BuiltinCall {
                    method: crate::builtins::BuiltinMethod::Len,
                    ..
                },
            } if matches!(
                suffix_steps.as_slice(),
                [PhysicalPathStep::Field(name)] if name.as_ref() == "value"
            )
        ));
    }

    #[test]
    fn direct_tape_predicates_collects_all_or_rejects_whole_prefix() {
        let active = crate::parse::parser::parse("$.active == true").expect("parse active");
        let score = crate::parse::parser::parse("$.score > 10").expect("parse score");
        let unsupported = crate::parse::parser::parse(r#"match @ with { _ -> true }"#)
            .expect("parse unsupported");

        let predicates = direct_tape_predicates_for_exprs([&active, &score]).expect("predicates");
        assert_eq!(predicates.len(), 2);
        assert!(direct_tape_predicates_for_exprs([&active, &unsupported]).is_none());
    }

    #[test]
    fn rejects_array_find_scalar_current_predicate() {
        let expr = crate::parse::parser::parse(r#"@.items.find(@ == 0)"#).expect("parse");
        assert!(direct_array_any_predicate_expr(&expr).is_none());
    }

    #[test]
    fn direct_cmp_literal_predicate_normalizes_literal_lhs() {
        let predicate = direct_tape_predicate_for_expr(
            &crate::parse::parser::parse(r#""Ada" == $.name"#).expect("parse"),
        )
        .expect("predicate");
        let Some((steps, op, lit)) = direct_cmp_literal_predicate(&predicate) else {
            panic!("expected direct literal comparison");
        };
        assert!(matches!(
            steps.as_slice(),
            [PhysicalPathStep::Field(name)] if name.as_ref() == "name"
        ));
        assert_eq!(op, BinOp::Eq);
        assert_eq!(lit, Val::Str(Arc::from("Ada")));

        let predicate = direct_tape_predicate_for_expr(
            &crate::parse::parser::parse(r#"10 < $.score"#).expect("parse"),
        )
        .expect("predicate");
        let Some((steps, op, lit)) = direct_cmp_literal_predicate(&predicate) else {
            panic!("expected flipped direct comparison");
        };
        assert!(matches!(
            steps.as_slice(),
            [PhysicalPathStep::Field(name)] if name.as_ref() == "score"
        ));
        assert_eq!(op, BinOp::Gt);
        assert_eq!(lit, Val::Int(10));
    }

    #[test]
    fn direct_count_on_path_uses_scalar_sink_projection() {
        let engine = JetroEngine::new();
        assert_eq!(
            direct_writer_plan_kind(&engine, "$.attributes.count()"),
            Some((
                Some(NdjsonDirectPlanKind::ByteExpr),
                NdjsonDirectPlanKind::TapeScalarCall,
            ))
        );
        assert_eq!(
            direct_writer_plan_kind(&engine, "$.meta.attributes.count()"),
            Some((
                Some(NdjsonDirectPlanKind::ByteExpr),
                NdjsonDirectPlanKind::TapeScalarCall,
            ))
        );
        assert_eq!(
            direct_writer_plan_kind(&engine, "$.attributes.filter(@.active).count()"),
            Some((None, NdjsonDirectPlanKind::TapeStreamCount))
        );
    }

    #[test]
    fn stream_map_values_iterates_projection_values() {
        let map = NdjsonDirectStreamMap::Object(vec![
            NdjsonDirectObjectField {
                key: Arc::from("id"),
                value: NdjsonDirectProjectionValue::Path(vec![PhysicalPathStep::Field(
                    Arc::from("id"),
                )]),
                optional: false,
            },
            NdjsonDirectObjectField {
                key: Arc::from("active"),
                value: NdjsonDirectProjectionValue::Literal(Val::Bool(true)),
                optional: false,
            },
        ]);
        assert_eq!(map.values().count(), 2);

        let path = NdjsonDirectStreamMap::Value(NdjsonDirectProjectionValue::Path(vec![
            PhysicalPathStep::Field(Arc::from("name")),
        ]))
        .value_path();
        assert!(matches!(
            path.as_deref(),
            Some([PhysicalPathStep::Field(name)]) if name.as_ref() == "name"
        ));

        let value = NdjsonDirectProjectionValue::ViewScalarCall {
            steps: vec![PhysicalPathStep::Field(Arc::from("email"))],
            call: crate::builtins::BuiltinCall::new(
                crate::builtins::BuiltinMethod::Len,
                crate::builtins::BuiltinArgs::None,
            ),
            optional: false,
        };
        assert!(matches!(
            value.path_steps(),
            Some([PhysicalPathStep::Field(name)]) if name.as_ref() == "email"
        ));
        let projection = value.path_projection().expect("path projection");
        assert_eq!(projection.steps().len(), 1);
        assert!(projection.scalar_call().is_some());
    }
}
