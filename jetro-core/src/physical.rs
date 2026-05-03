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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BackendSet(u16);

impl BackendSet {
    pub(crate) const NONE: Self = Self(0);
    pub(crate) const STRUCTURAL: Self = Self(1 << 0);
    pub(crate) const TAPE_VIEW: Self = Self(1 << 1);
    pub(crate) const TAPE_ROWS: Self = Self(1 << 2);
    pub(crate) const TAPE_PATH: Self = Self(1 << 3);
    pub(crate) const VAL_VIEW: Self = Self(1 << 4);
    pub(crate) const MATERIALIZED_SOURCE: Self = Self(1 << 5);
    pub(crate) const FAST_CHILDREN: Self = Self(1 << 6);

    #[inline]
    pub(crate) const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    #[inline]
    pub(crate) const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
}

impl PlanNode {
    #[inline]
    pub(crate) fn backends(&self) -> BackendSet {
        match self {
            Self::Pipeline { source, .. } => match source {
                PipelinePlanSource::FieldChain { .. } => BackendSet::TAPE_VIEW
                    .union(BackendSet::TAPE_ROWS)
                    .union(BackendSet::VAL_VIEW)
                    .union(BackendSet::MATERIALIZED_SOURCE),
                PipelinePlanSource::Expr(_) => BackendSet::FAST_CHILDREN,
            },
            Self::Structural { .. } => BackendSet::STRUCTURAL,
            Self::RootPath(_) => BackendSet::TAPE_PATH,
            Self::Object(_) | Self::Array(_) | Self::Call { .. } | Self::Chain { .. } => {
                BackendSet::FAST_CHILDREN
            }
            _ => BackendSet::NONE,
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::{BodyKernel, Sink};

    fn empty_body() -> PipelineBody {
        PipelineBody {
            stages: Vec::new(),
            stage_exprs: Vec::new(),
            sink: Sink::Collect,
            stage_kernels: Vec::new(),
            sink_kernels: Vec::new(),
        }
    }

    #[test]
    fn pipeline_field_chain_advertises_view_and_source_backends() {
        let node = PlanNode::Pipeline {
            source: PipelinePlanSource::FieldChain {
                keys: Arc::from([Arc::<str>::from("rows")]),
            },
            body: empty_body(),
        };
        let backends = node.backends();
        assert!(backends.contains(BackendSet::TAPE_VIEW));
        assert!(backends.contains(BackendSet::TAPE_ROWS));
        assert!(backends.contains(BackendSet::VAL_VIEW));
        assert!(backends.contains(BackendSet::MATERIALIZED_SOURCE));
        assert!(!backends.contains(BackendSet::FAST_CHILDREN));
    }

    #[test]
    fn structural_nodes_and_composite_nodes_advertise_distinct_backends() {
        let root_path = PlanNode::RootPath(vec![PhysicalPathStep::Field(Arc::from("meta"))]);
        assert!(root_path.backends().contains(BackendSet::TAPE_PATH));

        let object = PlanNode::Object(vec![PhysicalObjField::Kv {
            key: Arc::from("a"),
            val: NodeId(0),
            optional: false,
            cond: None,
        }]);
        assert!(object.backends().contains(BackendSet::FAST_CHILDREN));

        let pipeline = PlanNode::Pipeline {
            source: PipelinePlanSource::Expr(NodeId(0)),
            body: PipelineBody {
                stage_kernels: vec![BodyKernel::Generic],
                ..empty_body()
            },
        };
        let backends = pipeline.backends();
        assert!(backends.contains(BackendSet::FAST_CHILDREN));
        assert!(!backends.contains(BackendSet::TAPE_VIEW));
    }
}
