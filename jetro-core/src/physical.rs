//! Physical expression IR used by the planner and physical evaluator.

use std::sync::Arc;

use crate::ast::{BinOp, KindType};
use crate::builtins::BuiltinCall;
use crate::pipeline::PipelineBody;
use crate::structural::StructuralPlan;
use crate::value::Val;
use crate::vm::Program;

#[derive(Clone)]
pub struct QueryPlan {
    root: QueryRoot,
    nodes: Vec<PlanNode>,
}

impl QueryPlan {
    #[inline]
    pub(crate) fn from_nodes(root: NodeId, nodes: Vec<PlanNode>) -> Self {
        Self {
            root: QueryRoot::Node(root),
            nodes,
        }
    }

    #[inline]
    pub fn source_vm(expr: &str) -> Self {
        Self {
            root: QueryRoot::SourceVm(Arc::from(expr)),
            nodes: Vec::new(),
        }
    }

    #[inline]
    pub fn root(&self) -> &QueryRoot {
        &self.root
    }

    #[inline]
    pub(crate) fn node(&self, id: NodeId) -> &PlanNode {
        &self.nodes[id.0]
    }
}

#[derive(Clone)]
pub enum QueryRoot {
    Node(NodeId),
    SourceVm(Arc<str>),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NodeId(pub(crate) usize);

#[derive(Clone)]
pub enum PlanNode {
    Literal(Val),
    Root,
    Current,
    Ident(Arc<str>),
    Pipeline {
        source: PipelinePlanSource,
        body: PipelineBody,
    },
    Structural {
        plan: StructuralPlan,
        fallback: Arc<Program>,
    },
    RootPath(Vec<PhysicalPathStep>),
    Chain {
        base: NodeId,
        steps: Vec<PhysicalChainStep>,
    },
    Call {
        receiver: NodeId,
        call: BuiltinCall,
        optional: bool,
    },
    UnaryNeg(NodeId),
    Not(NodeId),
    Binary {
        lhs: NodeId,
        op: BinOp,
        rhs: NodeId,
    },
    Kind {
        expr: NodeId,
        ty: KindType,
        negate: bool,
    },
    Coalesce {
        lhs: NodeId,
        rhs: NodeId,
    },
    IfElse {
        cond: NodeId,
        then_: NodeId,
        else_: NodeId,
    },
    Try {
        body: NodeId,
        default: NodeId,
    },
    Object(Vec<PhysicalObjField>),
    Array(Vec<PhysicalArrayElem>),
    Let {
        name: Arc<str>,
        init: NodeId,
        body: NodeId,
    },
    Vm(Arc<Program>),
}

#[derive(Clone)]
pub enum PipelinePlanSource {
    FieldChain { keys: Arc<[Arc<str>]> },
    Expr(NodeId),
}

#[derive(Clone)]
pub enum PhysicalPathStep {
    Field(Arc<str>),
    Index(i64),
}

#[derive(Clone)]
pub enum PhysicalChainStep {
    Field(Arc<str>),
    Index(i64),
    DynIndex(NodeId),
}

#[derive(Clone)]
pub enum PhysicalObjField {
    Kv {
        key: Arc<str>,
        val: NodeId,
        optional: bool,
        cond: Option<NodeId>,
    },
    Short(Arc<str>),
    Dynamic {
        key: NodeId,
        val: NodeId,
    },
    Spread(NodeId),
    SpreadDeep(NodeId),
}

#[derive(Clone)]
pub enum PhysicalArrayElem {
    Expr(NodeId),
    Spread(NodeId),
}
