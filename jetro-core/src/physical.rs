//! Physical query plan IR used by the planner and `physical_eval`.
//!
//! A `QueryPlan` is a DAG of `PhysicalNode`s produced by `planner`. Each node
//! carries a `PlanNode` kind, a preference list of backends (`Structural`,
//! `View`, `Pipeline`, `Vm`), and `ExecutionFacts` that inform the executor
//! about what the node produces. The executor walks the DAG and tries each
//! backend preference in order, falling back to VM as the last resort.

use std::sync::Arc;

use crate::ast::{BinOp, KindType};
use crate::builtins::BuiltinCall;
use crate::pipeline::PipelineBody;
use crate::structural::StructuralPlan;
use crate::data::value::Val;
use crate::vm::Program;

/// Compiled query plan: a DAG of `PhysicalNode`s plus a root selector.
/// Cloneable so `JetroEngine` can cache and reuse plans across calls.
#[derive(Clone)]
pub struct QueryPlan {
    root: QueryRoot,
    nodes: Vec<PhysicalNode>,
}

impl QueryPlan {
    /// Constructs a plan from bare `PlanNode`s, auto-computing backend preferences; for tests only.
    #[inline]
    #[cfg(test)]
    pub(crate) fn from_nodes(root: NodeId, nodes: Vec<PlanNode>) -> Self {
        Self {
            root: QueryRoot::Node(root),
            nodes: nodes.into_iter().map(PhysicalNode::new).collect(),
        }
    }

    /// Constructs a plan from fully-annotated `PhysicalNode`s produced by the planner.
    #[inline]
    pub(crate) fn from_physical_nodes(root: NodeId, nodes: Vec<PhysicalNode>) -> Self {
        Self {
            root: QueryRoot::Node(root),
            nodes,
        }
    }

    /// Creates a parse-failed fallback plan that executes `expr` directly through the VM.
    #[inline]
    pub fn source_vm(expr: &str) -> Self {
        Self {
            root: QueryRoot::SourceVm(Arc::from(expr)),
            nodes: Vec::new(),
        }
    }

    /// Returns the root selector indicating whether this plan has physical nodes or a VM source.
    #[inline]
    pub fn root(&self) -> &QueryRoot {
        &self.root
    }

    /// Returns the `PlanNode` kind for the node at `id`.
    #[inline]
    pub(crate) fn node(&self, id: NodeId) -> &PlanNode {
        &self.nodes[id.0].kind
    }

    /// Returns the ordered backend preference slice for node `id` as set by the planner.
    #[inline]
    pub(crate) fn backend_preferences(&self, id: NodeId) -> &[BackendPreference] {
        self.nodes[id.0].backends.as_slice()
    }

    /// Returns the full set of backends that `id` is *capable* of using (may differ from preferences).
    #[inline]
    pub(crate) fn backend_capabilities(&self, id: NodeId) -> BackendSet {
        self.nodes[id.0].capabilities
    }

    /// Returns the `ExecutionFacts` metadata attached to node `id` by the planner.
    #[inline]
    pub(crate) fn execution_facts(&self, id: NodeId) -> ExecutionFacts {
        self.nodes[id.0].facts
    }

    /// Returns the `ExecutionFacts` for the root node; used by tests to assert byte-native status.
    #[inline]
    #[cfg(test)]
    pub(crate) fn root_execution_facts(&self) -> ExecutionFacts {
        match self.root {
            QueryRoot::Node(root) => self.execution_facts(root),
            QueryRoot::SourceVm(_) => ExecutionFacts {
                contains_vm_fallback: true,
                ..ExecutionFacts::default()
            },
        }
    }
}

/// Selects the execution entry point for a `QueryPlan`.
#[derive(Clone)]
pub enum QueryRoot {
    /// The plan was successfully lowered; evaluation begins at the given node.
    Node(NodeId),
    /// Parsing failed; the raw expression string must be run directly through the VM.
    SourceVm(Arc<str>),
}

/// Typed index into the `QueryPlan`'s node arena.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NodeId(pub(crate) usize);

/// Describes the computational role of a single node in the physical plan DAG.
#[derive(Clone)]
pub enum PlanNode {
    /// A compile-time constant value requiring no document access.
    Literal(Val),
    /// The document root (`$`); materialises or navigates the whole document.
    Root,
    /// The current pipeline item (`@`); valid only inside pipeline stage bodies.
    Current,
    /// An unresolved identifier that will be looked up in `Env` at runtime.
    Ident(Arc<str>),
    /// A name that was bound by an enclosing `let` expression; resolved via the fast-local cache.
    Local(Arc<str>),
    /// A pull-IR pipeline with a field-chain or receiver expression as its row source.
    Pipeline {
        /// Where the rows come from (a field path or an evaluated sub-expression).
        source: PipelinePlanSource,
        /// Stages, sink, and their associated kernels and programs.
        body: PipelineBody,
    },
    /// A deep-search operation that the structural (bitmap index) backend can satisfy.
    Structural {
        /// The structural search plan describing the index traversal.
        plan: StructuralPlan,
        /// Compiled VM program used when the structural backend is unavailable.
        fallback: Arc<Program>,
    },
    /// A straight field/index path rooted at `$`; tape-navigable without full materialisation.
    RootPath(Vec<PhysicalPathStep>),
    /// A sequence of field/index/dynamic-index steps applied to a base node.
    Chain {
        /// The value being navigated.
        base: NodeId,
        /// Ordered navigation steps applied left-to-right.
        steps: Vec<PhysicalChainStep>,
    },
    /// A built-in method call applied to a receiver value.
    Call {
        /// The node whose value becomes the method receiver.
        receiver: NodeId,
        /// The specific built-in call to dispatch.
        call: BuiltinCall,
        /// If `true`, propagate `null` from the receiver instead of calling the method.
        optional: bool,
    },
    /// Arithmetic negation of a numeric node.
    UnaryNeg(NodeId),
    /// Logical `not` of a node.
    Not(NodeId),
    /// A binary infix operation between two nodes.
    Binary {
        /// Left-hand operand.
        lhs: NodeId,
        /// The operator to apply.
        op: BinOp,
        /// Right-hand operand.
        rhs: NodeId,
    },
    /// A type-kind predicate check (`@ kind object`, `@ not kind array`, etc.).
    Kind {
        /// The value to test.
        expr: NodeId,
        /// The kind to check against.
        ty: KindType,
        /// When `true`, the result is inverted (`not kind`).
        negate: bool,
    },
    /// Returns the left operand unless it is null, in which case evaluates the right.
    Coalesce {
        /// Preferred value; returned if not null.
        lhs: NodeId,
        /// Fallback value used when `lhs` is null.
        rhs: NodeId,
    },
    /// Conditional expression: evaluates `then_` or `else_` based on `cond`.
    IfElse {
        /// Condition node; truthy value selects `then_`.
        cond: NodeId,
        /// Value produced when `cond` is truthy.
        then_: NodeId,
        /// Value produced when `cond` is falsy.
        else_: NodeId,
    },
    /// Evaluates `body`; on null or error, evaluates `default` instead.
    Try {
        /// Primary expression to attempt.
        body: NodeId,
        /// Fallback evaluated when `body` returns null or errors.
        default: NodeId,
    },
    /// Object literal with physical field descriptors.
    Object(Vec<PhysicalObjField>),
    /// Array literal with physical element descriptors.
    Array(Vec<PhysicalArrayElem>),
    /// Lexical `let name = init in body` binding.
    Let {
        /// The variable name being bound.
        name: Arc<str>,
        /// The initialiser expression whose result is bound to `name`.
        init: NodeId,
        /// The body expression evaluated with `name` in scope.
        body: NodeId,
    },
    /// Direct VM bytecode execution for expressions that no other path could lower.
    Vm(Arc<Program>),
}

/// A fully-annotated plan node combining its kind with planner-selected metadata.
#[derive(Clone)]
pub(crate) struct PhysicalNode {
    /// The logical operation this node performs.
    kind: PlanNode,
    /// Full set of backends the node's kind supports (may include options not in `backends`).
    capabilities: BackendSet,
    /// Propagated metadata used by the executor to decide whether to materialise the document.
    facts: ExecutionFacts,
    /// Ordered preference list chosen by the planner for this specific context.
    backends: BackendPlan,
}

impl PhysicalNode {
    /// Creates a node with auto-derived backend preferences and facts; for tests only.
    #[inline]
    #[cfg(test)]
    pub(crate) fn new(kind: PlanNode) -> Self {
        let backends = BackendPlan::for_node(&kind);
        Self::with_backend_plan(kind, backends)
    }

    /// Creates a node with the given backend plan; facts are derived from the node kind.
    #[inline]
    #[cfg(test)]
    pub(crate) fn with_backend_plan(kind: PlanNode, backends: BackendPlan) -> Self {
        let facts = ExecutionFacts::for_node(&kind);
        Self::with_backend_plan_and_facts(kind, backends, facts)
    }

    /// Creates a node with explicitly provided backend plan and facts; derives capabilities from kind.
    #[inline]
    pub(crate) fn with_backend_plan_and_facts(
        kind: PlanNode,
        backends: BackendPlan,
        facts: ExecutionFacts,
    ) -> Self {
        let capabilities = kind.backends();
        Self::with_backend_plan_capabilities_and_facts(kind, backends, capabilities, facts)
    }

    /// Creates a fully-specified node with all four fields supplied externally.
    #[cfg(test)]
    #[inline]
    pub(crate) fn with_backend_plan_capabilities_and_facts(
        kind: PlanNode,
        backends: BackendPlan,
        capabilities: BackendSet,
        facts: ExecutionFacts,
    ) -> Self {
        Self {
            kind,
            capabilities,
            facts,
            backends,
        }
    }

    /// Creates a fully-specified node with all four fields supplied externally (non-test).
    #[cfg(not(test))]
    #[inline]
    fn with_backend_plan_capabilities_and_facts(
        kind: PlanNode,
        backends: BackendPlan,
        capabilities: BackendSet,
        facts: ExecutionFacts,
    ) -> Self {
        Self {
            kind,
            capabilities,
            facts,
            backends,
        }
    }

    /// Returns a copy of the `ExecutionFacts` stored on this node.
    #[inline]
    pub(crate) fn execution_facts(&self) -> ExecutionFacts {
        self.facts
    }
}

/// A compact bitset recording which backend families a node can use.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BackendSet(u16);

/// Identifies the concrete execution strategy the executor should attempt for a node.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BackendPreference {
    /// Use the bitmap structural index (deep-search operations on raw bytes).
    Structural,
    /// Navigate the simd-json tape via a zero-copy view pipeline.
    TapeView,
    /// Materialise individual tape rows on demand via the row-bridge pipeline.
    TapeRows,
    /// Navigate a field/index path directly on the tape without materialising values.
    TapePath,
    /// Navigate or iterate a borrowed `ValView` without copying the whole document.
    ValView,
    /// Materialise the source array from the tape, then run the pipeline on a `Val`.
    MaterializedSource,
    /// Evaluate child nodes without constructing an `Env`; fastest for scalars and composites.
    FastChildren,
    /// Full tree-walking interpreter with `Env` materialisation; the universal fallback.
    Interpreted,
}

/// Propagated metadata about what a physical node produces, used to avoid unnecessary work.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ExecutionFacts {
    /// `true` when every leaf can execute without materialising the entire document as a `Val`.
    pub(crate) can_avoid_root_materialization: bool,
    /// `true` when at least one child produces a streamable row sequence.
    pub(crate) can_stream_rows: bool,
    /// `true` when at least one child can read directly from the simd-json tape.
    pub(crate) can_use_tape: bool,
    /// `true` when at least one descendant will unconditionally fall through to the VM.
    pub(crate) contains_vm_fallback: bool,
    /// `true` when at least one child may need to materialise its source array.
    pub(crate) may_materialize_source: bool,
}

impl ExecutionFacts {
    /// Returns `true` when the plan can execute entirely on raw bytes, bypassing `Val` construction.
    #[inline]
    pub(crate) fn is_byte_native(self) -> bool {
        self.can_avoid_root_materialization && !self.contains_vm_fallback
    }

    /// Returns facts for a node that produces a compile-time constant and needs no document access.
    #[inline]
    pub(crate) fn constant() -> Self {
        Self {
            can_avoid_root_materialization: true,
            ..Self::default()
        }
    }

    /// Folds an iterator of child facts into a single summary using conservative bit-propagation.
    #[inline]
    pub(crate) fn combine_all(children: impl IntoIterator<Item = Self>) -> Self {
        let mut saw_child = false;
        let mut out = Self {
            can_avoid_root_materialization: true,
            ..Self::default()
        };
        for child in children {
            saw_child = true;
            out.can_avoid_root_materialization &=
                child.can_avoid_root_materialization && !child.contains_vm_fallback;
            out.can_stream_rows |= child.can_stream_rows;
            out.can_use_tape |= child.can_use_tape;
            out.contains_vm_fallback |= child.contains_vm_fallback;
            out.may_materialize_source |= child.may_materialize_source;
        }
        if saw_child {
            out
        } else {
            Self::constant()
        }
    }

    /// Computes the initial `ExecutionFacts` for a node based solely on its own kind (no children).
    #[inline]
    pub(crate) fn for_node(node: &PlanNode) -> Self {
        match node {
            PlanNode::Literal(_) => Self::constant(),
            PlanNode::Pipeline { source, body } => {
                let field_chain = matches!(source, PipelinePlanSource::FieldChain { .. });
                let view_native = crate::pipeline::view_capabilities(body).is_some();
                let view_prefix = crate::pipeline::view_prefix_capabilities(body).is_some();
                let materialized_source = field_chain && body.can_run_with_materialized_receiver();
                let can_complete_without_root = field_chain && (view_native || materialized_source);
                Self {
                    can_avoid_root_materialization: can_complete_without_root,
                    can_stream_rows: field_chain && (view_native || view_prefix),
                    can_use_tape: field_chain,
                    contains_vm_fallback: false,
                    may_materialize_source: materialized_source,
                }
            }
            PlanNode::Structural { .. } => Self {
                can_avoid_root_materialization: true,
                can_stream_rows: false,
                can_use_tape: false,
                contains_vm_fallback: true,
                may_materialize_source: false,
            },
            PlanNode::RootPath(_) => Self {
                can_avoid_root_materialization: true,
                can_stream_rows: false,
                can_use_tape: true,
                contains_vm_fallback: false,
                may_materialize_source: false,
            },
            PlanNode::Vm(_) => Self {
                contains_vm_fallback: true,
                ..Self::default()
            },
            _ => Self::default(),
        }
    }
}

/// An inline-stored ordered list of up to five backend preferences chosen by the planner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BackendPlan {
    /// Number of valid entries in `items`.
    len: u8,
    /// Fixed-size buffer holding up to five preference entries; unused slots are ignored.
    items: [BackendPreference; 5],
}

impl BackendPlan {
    /// An empty plan with no preferences.
    const EMPTY: Self = Self {
        len: 0,
        items: [BackendPreference::FastChildren; 5],
    };

    /// Derives the default preference list for `node` from its static `backend_preferences`.
    #[inline]
    pub(crate) fn for_node(node: &PlanNode) -> Self {
        Self::new(node.backend_preferences())
    }

    /// Constructs a plan from a slice of at most five preferences.
    #[inline]
    pub(crate) fn new(backends: &[BackendPreference]) -> Self {
        debug_assert!(backends.len() <= 5);
        let mut out = Self::EMPTY;
        out.len = backends.len().min(5) as u8;
        out.items[..out.len as usize].copy_from_slice(&backends[..out.len as usize]);
        out
    }

    /// Returns the active slice of preference entries.
    #[inline]
    pub(crate) fn as_slice(&self) -> &[BackendPreference] {
        &self.items[..self.len as usize]
    }

    /// Converts the preference list into a `BackendSet` union for capability checks.
    #[inline]
    pub(crate) fn as_set(&self) -> BackendSet {
        let mut out = BackendSet::NONE;
        for backend in self.as_slice() {
            out = out.union(backend.backend_set());
        }
        out
    }

    /// Returns a copy of this plan with the `Interpreted` entry removed.
    #[inline]
    pub(crate) fn without_interpreted(self) -> Self {
        let mut items = [BackendPreference::FastChildren; 5];
        let mut len = 0usize;
        for backend in self.as_slice() {
            if *backend != BackendPreference::Interpreted {
                items[len] = *backend;
                len += 1;
            }
        }
        Self {
            len: len as u8,
            items,
        }
    }

    /// Conditionally strips `Interpreted` from the plan based on a runtime flag.
    #[inline]
    pub(crate) fn without_interpreted_if(self, condition: bool) -> Self {
        if condition {
            self.without_interpreted()
        } else {
            self
        }
    }
}

impl BackendSet {
    /// Empty set — node advertises no backends.
    pub(crate) const NONE: Self = Self(0);
    /// Bit for the structural (bitmap index) backend.
    pub(crate) const STRUCTURAL: Self = Self(1 << 0);
    /// Bit for the tape-view zero-copy pipeline backend.
    pub(crate) const TAPE_VIEW: Self = Self(1 << 1);
    /// Bit for the tape-rows on-demand materialisation backend.
    pub(crate) const TAPE_ROWS: Self = Self(1 << 2);
    /// Bit for direct tape path navigation without value materialisation.
    pub(crate) const TAPE_PATH: Self = Self(1 << 3);
    /// Bit for the borrowed `ValView` pipeline backend.
    pub(crate) const VAL_VIEW: Self = Self(1 << 4);
    /// Bit for the materialized-source pipeline variant.
    pub(crate) const MATERIALIZED_SOURCE: Self = Self(1 << 5);
    /// Bit for the fast child-evaluation path (no `Env` construction).
    pub(crate) const FAST_CHILDREN: Self = Self(1 << 6);
    /// Bit for the full interpreted tree-walker with `Env`.
    pub(crate) const INTERPRETED: Self = Self(1 << 7);

    /// Returns the bitwise union of `self` and `other`.
    #[inline]
    pub(crate) const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Returns `true` when all bits of `other` are set in `self`.
    #[inline]
    pub(crate) const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
}

impl PlanNode {
    /// Returns the default ordered backend preference slice for this node kind.
    ///
    /// The planner may override this list via `select_backend_plan`; this method is the baseline.
    #[inline]
    pub(crate) fn backend_preferences(&self) -> &'static [BackendPreference] {
        match self {
            Self::Pipeline { source, .. } => match source {
                PipelinePlanSource::FieldChain { .. } => &[
                    BackendPreference::TapeView,
                    BackendPreference::TapeRows,
                    BackendPreference::MaterializedSource,
                    BackendPreference::ValView,
                    BackendPreference::Interpreted,
                ],
                PipelinePlanSource::Expr(_) => &[
                    BackendPreference::FastChildren,
                    BackendPreference::Interpreted,
                ],
            },
            Self::Structural { .. } => &[
                BackendPreference::Structural,
                BackendPreference::Interpreted,
            ],
            Self::RootPath(_) => &[BackendPreference::TapePath, BackendPreference::Interpreted],
            Self::Literal(_)
            | Self::Local(_)
            | Self::Object(_)
            | Self::Array(_)
            | Self::Call { .. }
            | Self::Chain { .. }
            | Self::UnaryNeg(_)
            | Self::Not(_)
            | Self::Binary { .. }
            | Self::Kind { .. }
            | Self::Coalesce { .. }
            | Self::IfElse { .. }
            | Self::Try { .. }
            | Self::Let { .. } => &[
                BackendPreference::FastChildren,
                BackendPreference::Interpreted,
            ],
            _ => &[BackendPreference::Interpreted],
        }
    }

    /// Returns the full `BackendSet` of all backends this node kind is capable of using.
    #[inline]
    pub(crate) fn backends(&self) -> BackendSet {
        BackendPlan::new(self.backend_preferences()).as_set()
    }
}

impl BackendPreference {
    /// Maps this preference variant to its single-bit `BackendSet` representation.
    #[inline]
    pub(crate) const fn backend_set(self) -> BackendSet {
        match self {
            Self::Structural => BackendSet::STRUCTURAL,
            Self::TapeView => BackendSet::TAPE_VIEW,
            Self::TapeRows => BackendSet::TAPE_ROWS,
            Self::TapePath => BackendSet::TAPE_PATH,
            Self::ValView => BackendSet::VAL_VIEW,
            Self::MaterializedSource => BackendSet::MATERIALIZED_SOURCE,
            Self::FastChildren => BackendSet::FAST_CHILDREN,
            Self::Interpreted => BackendSet::INTERPRETED,
        }
    }
}

/// Describes where a `Pipeline` node draws its input rows from.
#[derive(Clone)]
pub enum PipelinePlanSource {
    /// A dot-separated key sequence rooted at `$`, eligible for tape/view backends.
    FieldChain { keys: Arc<[Arc<str>]> },
    /// An arbitrary expression whose result becomes the pipeline receiver at runtime.
    Expr(NodeId),
}

/// A single step in a `RootPath` — purely field or integer-index navigation.
#[derive(Clone)]
pub enum PhysicalPathStep {
    /// Access a named object field.
    Field(Arc<str>),
    /// Access an array element by zero-based (or negative) integer index.
    Index(i64),
}

/// A single step in a `Chain` node, including dynamically-computed subscripts.
#[derive(Clone)]
pub enum PhysicalChainStep {
    /// Access a named object field.
    Field(Arc<str>),
    /// Access an array element by integer index.
    Index(i64),
    /// Access a field or element using a runtime-computed key from another node.
    DynIndex(NodeId),
}

/// Describes one field in a physical `Object` node.
#[derive(Clone)]
pub enum PhysicalObjField {
    /// A key/value pair with optional omit-when-null and guard-condition flags.
    Kv {
        /// The string key for this field.
        key: Arc<str>,
        /// Node that produces the field value.
        val: NodeId,
        /// If `true`, the field is omitted when `val` evaluates to null.
        optional: bool,
        /// Optional guard; the field is omitted when this node evaluates to falsy.
        cond: Option<NodeId>,
    },
    /// Shorthand `{name}` — resolves `name` from the current `Env` or document root.
    Short(Arc<str>),
    /// A field whose key is computed at runtime from another node.
    Dynamic {
        /// Node producing the key string.
        key: NodeId,
        /// Node producing the field value.
        val: NodeId,
    },
    /// Shallow-merges all key/value pairs from an object into the enclosing object.
    Spread(NodeId),
    /// Deep-merges (recursively) all nested key/value pairs into the enclosing object.
    SpreadDeep(NodeId),
}

/// Describes one element in a physical `Array` node.
#[derive(Clone)]
pub enum PhysicalArrayElem {
    /// A single element produced by the given node.
    Expr(NodeId),
    /// Spreads all items of the given array node into the enclosing array.
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
        assert!(backends.contains(BackendSet::INTERPRETED));
        assert!(!backends.contains(BackendSet::FAST_CHILDREN));
        assert_eq!(
            node.backend_preferences(),
            &[
                BackendPreference::TapeView,
                BackendPreference::TapeRows,
                BackendPreference::MaterializedSource,
                BackendPreference::ValView,
                BackendPreference::Interpreted,
            ]
        );
    }

    #[test]
    fn query_plan_stores_backend_preferences_per_node() {
        let plan = QueryPlan::from_nodes(
            NodeId(0),
            vec![PlanNode::Pipeline {
                source: PipelinePlanSource::FieldChain {
                    keys: Arc::from([Arc::<str>::from("rows")]),
                },
                body: empty_body(),
            }],
        );

        assert_eq!(
            plan.backend_preferences(NodeId(0)),
            &[
                BackendPreference::TapeView,
                BackendPreference::TapeRows,
                BackendPreference::MaterializedSource,
                BackendPreference::ValView,
                BackendPreference::Interpreted,
            ]
        );
    }

    #[test]
    fn query_plan_allows_planner_selected_backend_order() {
        let node = PlanNode::Pipeline {
            source: PipelinePlanSource::FieldChain {
                keys: Arc::from([Arc::<str>::from("rows")]),
            },
            body: empty_body(),
        };
        let plan = QueryPlan::from_physical_nodes(
            NodeId(0),
            vec![PhysicalNode::with_backend_plan(
                node,
                BackendPlan::new(&[BackendPreference::ValView, BackendPreference::TapeView]),
            )],
        );

        assert_eq!(
            plan.backend_preferences(NodeId(0)),
            &[BackendPreference::ValView, BackendPreference::TapeView]
        );
        let capabilities = plan.backend_capabilities(NodeId(0));
        assert!(capabilities.contains(BackendSet::TAPE_VIEW));
        assert!(capabilities.contains(BackendSet::TAPE_ROWS));
        assert!(capabilities.contains(BackendSet::MATERIALIZED_SOURCE));
        assert!(capabilities.contains(BackendSet::VAL_VIEW));
        let facts = plan.execution_facts(NodeId(0));
        assert!(facts.can_avoid_root_materialization);
        assert!(facts.can_stream_rows);
        assert!(facts.can_use_tape);
    }

    #[test]
    fn structural_nodes_and_composite_nodes_advertise_distinct_backends() {
        let root_path = PlanNode::RootPath(vec![PhysicalPathStep::Field(Arc::from("meta"))]);
        assert!(root_path.backends().contains(BackendSet::TAPE_PATH));
        let facts = ExecutionFacts::for_node(&root_path);
        assert!(facts.can_avoid_root_materialization);
        assert!(facts.can_use_tape);

        let object = PlanNode::Object(vec![PhysicalObjField::Kv {
            key: Arc::from("a"),
            val: NodeId(0),
            optional: false,
            cond: None,
        }]);
        assert!(object.backends().contains(BackendSet::FAST_CHILDREN));
        assert!(object.backends().contains(BackendSet::INTERPRETED));

        let pipeline = PlanNode::Pipeline {
            source: PipelinePlanSource::Expr(NodeId(0)),
            body: PipelineBody {
                stage_kernels: vec![BodyKernel::Generic],
                ..empty_body()
            },
        };
        let backends = pipeline.backends();
        assert!(backends.contains(BackendSet::FAST_CHILDREN));
        assert!(backends.contains(BackendSet::INTERPRETED));
        assert!(!backends.contains(BackendSet::TAPE_VIEW));
        assert_eq!(
            pipeline.backend_preferences(),
            &[
                BackendPreference::FastChildren,
                BackendPreference::Interpreted
            ]
        );
    }
}
