//! Query planner: parser output → physical `QueryPlan`.
//!
//! `plan_query_with_context` is the single entry point. It parses the
//! expression string, walks the AST, and emits a `QueryPlan` whose nodes
//! carry ordered backend-preference lists. Pipeline-eligible chains get a
//! `Pipeline` node preference; structural deep-search gets a `Structural`
//! preference; everything else gets a VM-compiled `Program`. No evaluation
//! happens here — the plan is a pure data structure for `physical_eval`.

use std::sync::Arc;

use crate::builtins::registry::{by_name as builtin_by_name, view_object_items_projection_call};
use crate::builtins::BuiltinCall;
use crate::compile::compiler::Compiler;
use crate::data::value::Val;
use crate::exec::pipeline::{Pipeline, Source};
use crate::exec::structural::{StructuralPathStep, StructuralPlan};
use crate::ir::physical::{
    BackendPlan, ExecutionFacts, NodeId, PhysicalArrayElem, PhysicalChainStep, PhysicalNode,
    PhysicalObjField, PhysicalPathStep, PipelinePlanSource, PlanNode, QueryPlan,
};
use crate::parse::ast::{ArrayElem, Expr, ObjField, Step};
use crate::parse::parser;
use crate::plan::analysis;

/// Accumulates `PhysicalNode`s as the AST is lowered and tracks lexical state
/// needed to distinguish let-bound locals from bare field identifiers.
#[derive(Default)]
struct PlanBuilder {
    /// Flat arena of physical nodes indexed by `NodeId`.
    nodes: Vec<PhysicalNode>,
    /// Input-mode context that controls which backends are eligible.
    context: PlanningContext,
    /// Stack of names currently bound by enclosing `let` expressions.
    locals: Vec<Arc<str>>,
}

impl PlanBuilder {
    /// Returns `true` if `name` is currently bound by an enclosing `let`.
    #[inline]
    fn is_local(&self, name: &str) -> bool {
        self.locals.iter().rev().any(|local| local.as_ref() == name)
    }

    /// Records a new `let`-binding name as entering scope.
    #[inline]
    fn push_local(&mut self, name: Arc<str>) {
        self.locals.push(name);
    }

    /// Removes the innermost `let`-binding name when leaving its scope.
    #[inline]
    fn pop_local(&mut self) {
        self.locals.pop();
    }
}

/// Whether the `Jetro` handle was built from raw bytes or an in-memory Value.
/// Governs which backend representations (tape, structural index) are eligible.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum InputMode {
    Bytes,
    Val,
}

/// Planner configuration derived from the `Jetro` document handle at
/// call time. Feeds into the plan-cache key so that the same expression
/// string planned against bytes vs. Value hits different cache slots.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PlanningContext {
    input: InputMode,
}

impl Default for PlanningContext {
    /// Defaults to `Bytes` mode, matching the most capable backend tier.
    #[inline]
    fn default() -> Self {
        Self::bytes()
    }
}

impl PlanningContext {
    /// Constructs a context for a document backed by raw bytes (tape/structural eligible).
    #[inline]
    pub(crate) const fn bytes() -> Self {
        Self {
            input: InputMode::Bytes,
        }
    }

    /// Constructs a context for an in-memory `Val` document (no tape or structural index).
    #[inline]
    pub(crate) const fn val() -> Self {
        Self {
            input: InputMode::Val,
        }
    }

    /// Returns a short static string suitable for use as a plan-cache namespace key.
    #[inline]
    pub(crate) const fn cache_key(self) -> &'static str {
        match self.input {
            InputMode::Bytes => "bytes",
            InputMode::Val => "val",
        }
    }
}

impl PlanBuilder {
    /// Consumes the builder and wraps all accumulated nodes into a `QueryPlan`.
    #[inline]
    fn finish(self, root: NodeId) -> QueryPlan {
        QueryPlan::from_physical_nodes(root, self.nodes)
    }

    /// Appends a node, auto-computing its `ExecutionFacts` and `BackendPlan`, and returns its id.
    #[inline]
    fn push(&mut self, node: PlanNode) -> NodeId {
        let facts = self.execution_facts_for_node(&node);
        let backends = self.backend_plan_for_node(&node, facts);
        let facts = adjust_facts_for_backend_plan(&node, backends, facts);
        self.push_with_backends_and_facts(node, backends, facts)
    }

    /// Appends a node with pre-computed backends and facts, returning its assigned `NodeId`.
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

    /// Selects the ordered backend preference list for `node` given the current planning context.
    #[inline]
    fn backend_plan_for_node(&self, node: &PlanNode, facts: ExecutionFacts) -> BackendPlan {
        select_backend_plan(self.context, node, facts)
    }

    /// Retrieves the cached `ExecutionFacts` for an already-pushed node.
    #[inline]
    fn node_facts(&self, id: NodeId) -> ExecutionFacts {
        self.nodes[id.0].execution_facts()
    }

    /// Derives `ExecutionFacts` for `node` by propagating facts from its children.
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
            PlanNode::Call { receiver, call, .. } => {
                let receiver = self.node_facts(*receiver);
                if receiver.is_byte_native() && call.is_view_projection() {
                    ExecutionFacts {
                        can_avoid_root_materialization: true,
                        can_use_tape: receiver.can_use_tape,
                        contains_vm_fallback: false,
                        may_materialize_source: false,
                        can_stream_rows: false,
                    }
                } else {
                    ExecutionFacts::combine_all([receiver])
                }
            }
            PlanNode::UnaryNeg(receiver)
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
            PlanNode::UpdateBatch { root, .. } => {
                let root = self.node_facts(*root);
                ExecutionFacts {
                    contains_vm_fallback: true,
                    may_materialize_source: root.may_materialize_source,
                    ..root
                }
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

/// Maps a `(context, node, facts)` triple to the ordered `BackendPlan` preference list.
///
/// This is the single policy point that decides backend ordering; all callers go through here.
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
            crate::ir::physical::BackendPreference::ValView,
            crate::ir::physical::BackendPreference::Interpreted,
        ]),
        (InputMode::Val, PlanNode::RootPath(_) | PlanNode::Structural { .. }) => {
            BackendPlan::new(&[crate::ir::physical::BackendPreference::Interpreted])
        }
        (InputMode::Bytes, PlanNode::Structural { .. }) => BackendPlan::new(&[
            crate::ir::physical::BackendPreference::Structural,
            crate::ir::physical::BackendPreference::Interpreted,
        ]),
        (
            InputMode::Bytes,
            PlanNode::Pipeline {
                source: PipelinePlanSource::FieldChain { .. },
                ..
            },
        ) if facts.can_stream_rows => BackendPlan::new(&[
            crate::ir::physical::BackendPreference::TapeView,
            crate::ir::physical::BackendPreference::TapeRows,
            crate::ir::physical::BackendPreference::MaterializedSource,
            crate::ir::physical::BackendPreference::ValView,
            crate::ir::physical::BackendPreference::Interpreted,
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

/// Clears the `contains_vm_fallback` flag on `Structural` nodes when the structural backend is
/// actually selected, since no VM execution will occur for those nodes at runtime.
#[inline]
fn adjust_facts_for_backend_plan(
    node: &PlanNode,
    backends: BackendPlan,
    mut facts: ExecutionFacts,
) -> ExecutionFacts {
    if matches!(node, PlanNode::Structural { .. })
        && backends
            .as_slice()
            .contains(&crate::ir::physical::BackendPreference::Structural)
    {
        facts.contains_vm_fallback = false;
    }
    facts
}

/// Top-level AST-to-plan dispatcher: tries each lowering strategy in priority order and falls
/// back to a `PlanNode::Vm` wrapper when no specialised path applies.
#[inline]
fn lower_expr(builder: &mut PlanBuilder, expr: &Expr) -> NodeId {
    try_lower_structural_op(expr)
        .map(|node| builder.push(node))
        .or_else(|| try_lower_pipeline(builder, expr).map(|node| builder.push(node)))
        .or_else(|| try_lower_root_path(expr).map(|node| builder.push(node)))
        .or_else(|| try_lower_implicit_root_path(builder, expr).map(|node| builder.push(node)))
        .or_else(|| try_lower_object_items_pipeline(builder, expr))
        .or_else(|| try_lower_receiver_pipeline(builder, expr))
        .or_else(|| try_lower_structural_chain_prefix(builder, expr))
        .or_else(|| try_lower_pipeline_path_suffix(builder, expr))
        .or_else(|| try_lower_chain(builder, expr))
        .or_else(|| try_lower_scalar(builder, expr))
        .or_else(|| try_lower_structural(builder, expr))
        .unwrap_or_else(|| fallback_vm(builder, expr))
}

/// Attempts to lower `expr` as a field-chain pipeline node; returns `None` for non-pipeline
/// expressions and for trivial collect-only pipelines that add no value.
///
/// Tries the logical path (`logical_planner → optimizer → logical_lower`) first; falls back
/// to the legacy `Pipeline::lower()` for shapes the logical planner cannot classify.
fn try_lower_pipeline(builder: &PlanBuilder, expr: &Expr) -> Option<PlanNode> {
    if should_skip_field_chain_pipeline(expr) {
        return None;
    }
    let pipeline = lower_via_logical(expr).or_else(|| Pipeline::lower(expr))?;
    if is_trivial_collect_pipeline(&pipeline) {
        return None;
    }
    if is_scalar_unwrap_pipeline(&pipeline) {
        return None;
    }
    let (source, mut body) = pipeline.into_source_body();
    mask_active_local_stage_kernels(&mut body, builder);
    pipeline_parts_to_plan_node(source, body)
}

fn should_skip_field_chain_pipeline(expr: &Expr) -> bool {
    expr_is_direct_view_projection_chain(expr)
        || field_chain_pipeline_starts_with_direct_view_projection(expr)
}

fn field_chain_pipeline_starts_with_direct_view_projection(expr: &Expr) -> bool {
    let Expr::Chain(base, steps) = expr else {
        return false;
    };
    if !matches!(base.as_ref(), Expr::Root) {
        return false;
    }
    let Some(first_method) = steps
        .iter()
        .find(|step| !matches!(step, Step::Field(_) | Step::OptField(_) | Step::Index(_)))
    else {
        return false;
    };
    analysis::starts_with_direct_view_projection(std::slice::from_ref(first_method))
}

/// Returns `true` for path-receiver pipelines whose every stage is a
/// scalar-direct-dispatch builtin (e.g. `$.s.upper().lower()`). Rejecting these
/// from pipeline lowering causes `lower_expr` to fall through to
/// `try_lower_chain`, which emits direct `apply_one` calls and returns a
/// scalar — eliminating the legacy one-element array wrap on path receivers.
fn is_scalar_unwrap_pipeline(pipeline: &Pipeline) -> bool {
    pipeline.is_field_chain_scalar_direct_collect()
}

fn is_scalar_unwrap_body(body: &crate::exec::pipeline::PipelineBody) -> bool {
    body.is_scalar_direct_collect()
}

/// Runs `expr` through the logical planner, optimizer, and logical lowerer. Returns `None` if
/// any stage cannot classify the expression.
fn lower_via_logical(expr: &Expr) -> Option<Pipeline> {
    let logical = crate::plan::logical::try_lower(expr)?;
    let optimized = crate::plan::optimize::Optimizer::default_rules().optimize(logical);
    crate::exec::pipeline::logical_lower::try_lower(optimized)
}

/// Converts a decomposed pipeline `(source, body)` pair into a `PlanNode::Pipeline`, returning
/// `None` when the source is a `Receiver` (those go through `try_lower_receiver_pipeline`).
fn pipeline_parts_to_plan_node(
    source: Source,
    body: crate::exec::pipeline::PipelineBody,
) -> Option<PlanNode> {
    let source = match source {
        Source::FieldChain { keys } => PipelinePlanSource::FieldChain { keys },
        Source::Receiver(_) => return None,
    };
    Some(PlanNode::Pipeline { source, body })
}

/// Lowers chains like `$.xs.sort_by(key).last().field` as a pipeline prefix followed by
/// a plain path suffix. The pipeline lowerer requires terminal sinks to end the method
/// chain, but the physical IR can represent the selected row as a `Pipeline` node and
/// keep the final field/index lookup as a `Chain`.
fn try_lower_pipeline_path_suffix(builder: &mut PlanBuilder, expr: &Expr) -> Option<NodeId> {
    let Expr::Chain(base, steps) = expr else {
        return None;
    };
    let mut split = steps.len();
    let mut suffix = Vec::new();
    while split > 0 {
        let step = &steps[split - 1];
        if let Some(physical) = physical_chain_path_step(step) {
            suffix.push(physical);
            split -= 1;
            continue;
        }
        break;
    }
    if suffix.is_empty() || split == 0 {
        return None;
    }
    suffix.reverse();
    let prefix = Expr::Chain(base.clone(), steps[..split].to_vec());
    let pipeline = lower_via_logical(&prefix).or_else(|| Pipeline::lower(&prefix))?;
    if is_trivial_collect_pipeline(&pipeline) || is_scalar_unwrap_pipeline(&pipeline) {
        return None;
    }
    let (source, mut body) = pipeline.into_source_body();
    mask_active_local_stage_kernels(&mut body, builder);
    let node = pipeline_parts_to_plan_node(source, body)?;
    let base = builder.push(node);
    Some(builder.push(PlanNode::Chain {
        base,
        steps: suffix,
    }))
}

fn physical_chain_path_step(step: &Step) -> Option<PhysicalChainStep> {
    match step {
        Step::Field(key) | Step::OptField(key) => {
            Some(PhysicalChainStep::Field(Arc::from(key.as_str())))
        }
        Step::Index(idx) => Some(PhysicalChainStep::Index(*idx)),
        _ => None,
    }
}

/// Returns `true` when the pipeline has no stages and sinks straight to `Collect`,
/// meaning it is semantically equivalent to just evaluating the source expression.
fn is_trivial_collect_pipeline(pipeline: &Pipeline) -> bool {
    pipeline.stages.is_empty() && matches!(pipeline.sink, crate::exec::pipeline::Sink::Collect)
}

/// Demotes any stage or sink kernel that references an in-scope local variable to `Generic`,
/// ensuring the pipeline evaluator resolves the name through the `Env` rather than as a row field.
fn mask_active_local_stage_kernels(
    body: &mut crate::exec::pipeline::PipelineBody,
    builder: &PlanBuilder,
) {
    if builder.locals.is_empty() {
        return;
    }

    if body.stage_exprs.len() == body.stage_kernels.len() {
        for idx in 0..body.stage_exprs.len() {
            let expr = &body.stage_exprs[idx];
            let Some(expr) = expr else {
                continue;
            };
            if builder
                .locals
                .iter()
                .any(|local| analysis::expr_uses_ident(expr, local.as_ref()))
            {
                recompile_stage_body_for_lexical_env(&mut body.stages[idx], expr);
                body.stage_kernels[idx] = crate::exec::pipeline::BodyKernel::Generic;
            }
        }
    }

    match &mut body.sink {
        crate::exec::pipeline::Sink::Reducer(spec) => {
            let mut kernel_idx = 0usize;
            if let (Some(program), Some(expr)) = (&mut spec.predicate, spec.predicate_expr.as_ref())
            {
                recompile_sink_program_for_lexical_env(
                    program,
                    expr,
                    body.sink_kernels.get_mut(kernel_idx),
                    builder,
                );
                kernel_idx += 1;
            }
            if let (Some(program), Some(expr)) =
                (&mut spec.projection, spec.projection_expr.as_ref())
            {
                recompile_sink_program_for_lexical_env(
                    program,
                    expr,
                    body.sink_kernels.get_mut(kernel_idx),
                    builder,
                );
            }
        }
        crate::exec::pipeline::Sink::Predicate(spec) => {
            if let Some(expr) = spec.predicate_expr.as_ref() {
                let kernel_idx = spec.predicate_kernel_index();
                recompile_sink_program_for_lexical_env(
                    &mut spec.predicate,
                    expr,
                    body.sink_kernels.get_mut(kernel_idx),
                    builder,
                );
            }
        }
        crate::exec::pipeline::Sink::ArgExtreme(spec) => {
            if let Some(expr) = spec.key_expr.as_ref() {
                let kernel_idx = spec.key_kernel_index();
                recompile_sink_program_for_lexical_env(
                    &mut spec.key,
                    expr,
                    body.sink_kernels.get_mut(kernel_idx),
                    builder,
                );
            }
        }
        _ => {
            for kernel in &mut body.sink_kernels {
                if kernel_mentions_active_local(kernel, &builder.locals) {
                    *kernel = crate::exec::pipeline::BodyKernel::Generic;
                }
            }
        }
    }
}

fn recompile_sink_program_for_lexical_env(
    program: &mut Arc<crate::vm::Program>,
    expr: &Expr,
    kernel: Option<&mut crate::exec::pipeline::BodyKernel>,
    builder: &PlanBuilder,
) {
    if builder
        .locals
        .iter()
        .any(|local| analysis::expr_uses_ident(expr, local.as_ref()))
    {
        let lowered = crate::compile::lambda_lower::unwrap_single_lambda(expr);
        *program = Arc::new(Compiler::compile(&lowered, "<local-aware-pipeline-sink>"));
        if let Some(kernel) = kernel {
            *kernel = crate::exec::pipeline::BodyKernel::Generic;
        }
    }
}

/// Recompiles the stored kernel program of a pipeline stage so it will be evaluated inside
/// a full `Env` (picking up let-bound variables) rather than against a bare row.
fn recompile_stage_body_for_lexical_env(stage: &mut crate::exec::pipeline::Stage, expr: &Expr) {
    let lowered = crate::compile::lambda_lower::unwrap_single_lambda(expr);
    let program = Arc::new(Compiler::compile(&lowered, "<local-aware-pipeline-stage>"));
    stage.replace_body_program(program);
}

/// Returns `true` if `kernel` references any identifier that is currently a let-bound local.
fn kernel_mentions_active_local(
    kernel: &crate::exec::pipeline::BodyKernel,
    locals: &[Arc<str>],
) -> bool {
    kernel.mentions_any_field_like_ident(locals)
}

/// Tries to lower `expr` as a complete `Structural` node when every step can be handled by the
/// bitmap index (all steps must be consumed — no residual suffix is allowed here).
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

/// Lowers an expression whose prefix qualifies for the structural backend while additional
/// field, index, or method steps follow; emits a `Structural` node feeding into a `Chain`.
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

/// Attempts to extract a contiguous leading `Structural` prefix from `(base, steps)`.
///
/// Returns `(plan, fallback_program, consumed_step_count)` on success; `None` when no prefix
/// can be mapped to the structural backend.
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
                let id = builtin_by_name(name.as_str())?;
                let plan = StructuralPlan::lower_builtin(anchor, id, args)?;
                let fallback_expr = base.clone().maybe_chain(steps[..=idx].to_vec());
                let fallback = Arc::new(Compiler::compile(&fallback_expr, "<structural-fallback>"));
                return Some((plan, fallback, idx + 1));
            }
            Step::DeepMatch { arms, early_stop } => {
                // Lower `..match { ... }` to a structural plan when the
                // compile-time shape summary admits bitmap candidate
                // enumeration (currently `ObjAnyOfKeys`). All other
                // shape summaries fall through to the VM tree-walk
                // runtime by returning `None` here.
                let cm = Arc::new(crate::compile::compiler::compile_match(
                    &Expr::Current,
                    arms,
                    &crate::compile::compiler::VarCtx::default(),
                ));
                let candidate_keys = match cm.shape_summary.as_ref()? {
                    crate::vm::MatchShapeSummary::ObjAnyOfKeys(keys) => Arc::clone(keys),
                    _ => return None,
                };
                let plan = StructuralPlan::DeepMatch {
                    anchor: Arc::from(anchor),
                    candidate_keys,
                    cm,
                    early_stop: *early_stop,
                };
                let fallback_expr = base.clone().maybe_chain(steps[..=idx].to_vec());
                let fallback = Arc::new(Compiler::compile(&fallback_expr, "<structural-fallback>"));
                return Some((plan, fallback, idx + 1));
            }
            _ => return None,
        }
    }
    None
}

/// Lowers a pipeline whose source is an arbitrary sub-expression (e.g. a let-bound variable or
/// a structural result), emitting a `Pipeline { source: Expr(_), body }` node.
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
        if receiver_pipeline_step_is_direct_view_projection(&steps[method_start]) {
            continue;
        }
        let Some(mut body) = Pipeline::lower_body_from_steps(&steps[method_start..]) else {
            continue;
        };
        if is_scalar_unwrap_body(&body) {
            continue;
        }
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

fn receiver_pipeline_step_is_direct_view_projection(step: &Step) -> bool {
    analysis::step_is_direct_view_projection(step)
}

/// Lowers `$` or a pure `$.field[idx]...` chain into a `RootPath` node, enabling tape-native
/// path navigation without materialising the full document value.
fn try_lower_root_path(expr: &Expr) -> Option<PlanNode> {
    match expr {
        Expr::Root => Some(PlanNode::RootPath(Vec::new())),
        Expr::Chain(base, steps) => {
            if !matches!(base.as_ref(), Expr::Root) {
                return None;
            }
            Some(PlanNode::RootPath(physical_path_steps(steps)?))
        }
        _ => None,
    }
}

/// Lowers bare field identifiers in byte-backed root expressions to `RootPath` nodes.
///
/// Jetro's expression environment resolves an unbound identifier as a field on the
/// current value. At the top level the current value is the document root, so
/// `id`, `user.name`, and `score > 9` can read directly from the tape instead of
/// materialising the whole row into `Val` just to build an `Env`.
fn try_lower_implicit_root_path(builder: &PlanBuilder, expr: &Expr) -> Option<PlanNode> {
    if builder.context.input != InputMode::Bytes {
        return None;
    }

    match expr {
        Expr::Ident(name) if !builder.is_local(name) => {
            Some(PlanNode::RootPath(vec![PhysicalPathStep::Field(
                Arc::from(name.as_str()),
            )]))
        }
        Expr::Chain(base, steps) => {
            let Expr::Ident(name) = base.as_ref() else {
                return None;
            };
            if builder.is_local(name) {
                return None;
            }

            let mut out = Vec::with_capacity(steps.len() + 1);
            out.push(PhysicalPathStep::Field(Arc::from(name.as_str())));
            out.extend(physical_path_steps(steps)?);
            Some(PlanNode::RootPath(out))
        }
        _ => None,
    }
}

pub(crate) fn physical_path_steps(steps: &[Step]) -> Option<Vec<PhysicalPathStep>> {
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
    Some(out)
}

pub(crate) fn physical_field_keys_to_path_steps(keys: &[Arc<str>]) -> Vec<PhysicalPathStep> {
    keys.iter()
        .map(|key| PhysicalPathStep::Field(Arc::clone(key)))
        .collect()
}

/// Lowers a general `Expr::Chain` into a sequence of `PhysicalChainStep`s, flushing accumulated
/// steps into `Chain` nodes whenever a method call interrupts the sequence.
fn try_lower_chain(builder: &mut PlanBuilder, expr: &Expr) -> Option<NodeId> {
    let Expr::Chain(base, steps) = expr else {
        return None;
    };

    let mut cur =
        if builder.context.input == InputMode::Bytes {
            match base.as_ref() {
                Expr::Ident(name) if !builder.is_local(name) => builder.push(PlanNode::RootPath(
                    vec![PhysicalPathStep::Field(Arc::from(name.as_str()))],
                )),
                _ => lower_expr(builder, base),
            }
        } else {
            lower_expr(builder, base)
        };
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

/// Emits a `Chain` node for any pending `steps`; returns `base` unchanged when `steps` is empty.
fn flush_chain(
    builder: &mut PlanBuilder,
    base: NodeId,
    steps: &mut Vec<PhysicalChainStep>,
) -> NodeId {
    if steps.is_empty() {
        return base;
    }
    if let PlanNode::RootPath(prefix) = builder.nodes[base.0].kind() {
        if let Some(suffix) = physical_steps_to_path_steps(steps) {
            let mut out = Vec::with_capacity(prefix.len() + suffix.len());
            out.extend(prefix.iter().cloned());
            out.extend(suffix);
            steps.clear();
            return builder.push(PlanNode::RootPath(out));
        }
    }
    builder.push(PlanNode::Chain {
        base,
        steps: std::mem::take(steps),
    })
}

pub(crate) fn physical_steps_to_path_steps(
    steps: &[PhysicalChainStep],
) -> Option<Vec<PhysicalPathStep>> {
    let mut out = Vec::with_capacity(steps.len());
    for step in steps {
        match step {
            PhysicalChainStep::Field(key) => {
                out.push(PhysicalPathStep::Field(Arc::clone(key)));
            }
            PhysicalChainStep::Index(index) => out.push(PhysicalPathStep::Index(*index)),
            PhysicalChainStep::DynIndex(_) => return None,
        }
    }
    Some(out)
}

/// Lowers scalar and simple compound expressions (literals, identifiers, unary/binary ops,
/// conditionals) that do not require a structural or pipeline-level representation.
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

/// Lowers object literals, array literals, and `let` expressions into their physical
/// counterparts, recursively lowering all child sub-expressions.
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
        Expr::UpdateBatch {
            root,
            selector,
            ops,
        } => {
            let dependencies = crate::plan::update::analyze_update_batch(root, selector, ops);
            let trie = crate::plan::update::build_update_trie_plan(ops);
            let root_node = lower_expr(builder, root);
            Some(builder.push(PlanNode::UpdateBatch {
                root: root_node,
                selector: selector.clone(),
                ops: ops.clone(),
                dependencies,
                trie,
                fallback: Arc::new(Compiler::compile(expr, "<planned-update>")),
            }))
        }
        _ => None,
    }
}

/// Creates a `PlanNode::Vm` wrapping a compiled `Program` as the last-resort fallback for
/// expressions that no specialised lowering path could handle.
fn fallback_vm(builder: &mut PlanBuilder, expr: &Expr) -> NodeId {
    builder.push(PlanNode::Vm(Arc::new(Compiler::compile(
        expr,
        "<planned-expr>",
    ))))
}

/// Converts an AST `ObjField` into a `PhysicalObjField`, recursively lowering value and
/// condition sub-expressions and promoting shorthand locals to explicit `Kv` nodes.
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

/// Converts an AST `ArrayElem` into a `PhysicalArrayElem`, lowering the contained expression.
fn plan_array_elem(builder: &mut PlanBuilder, elem: &ArrayElem) -> PhysicalArrayElem {
    match elem {
        ArrayElem::Expr(expr) => PhysicalArrayElem::Expr(lower_expr(builder, expr)),
        ArrayElem::Spread(expr) => PhysicalArrayElem::Spread(lower_expr(builder, expr)),
    }
}

/// Plans `expr` using the default (`Bytes`) context; exposed for tests + fuzz.
#[inline]
#[cfg(any(test, feature = "fuzz_internal"))]
pub fn plan_query(expr: &str) -> QueryPlan {
    plan_query_with_context(expr, PlanningContext::default())
}

/// Parses `expr`, walks the resulting AST through `PlanBuilder`, and returns a `QueryPlan`.
///
/// Falls back to a `SourceVm` plan when parsing fails, so callers always receive a usable plan.
#[inline]
pub(crate) fn plan_query_with_context(expr: &str, context: PlanningContext) -> QueryPlan {
    let Ok(ast) = parser::parse(expr) else {
        return QueryPlan::source_vm(expr);
    };
    plan_ast_with_context(ast, context)
}

pub(crate) fn plan_ast_with_context(ast: Expr, context: PlanningContext) -> QueryPlan {
    // Phase B: fuse contiguous same-root chain-writes into multi-op
    // `Expr::Patch` nodes before lowering. The resulting Patches are
    // automatically routed to Phase D's PathTrie execution path by the
    // bytecode compiler. The compiler runs the same pass on its inputs,
    // but doing it here as well lets the structural / pipeline lowerers
    // see fused Patches before they decide whether to fall back to the
    // VM source path.
    let ast = crate::plan::patch_fusion::fuse_writes(ast);
    let mut builder = PlanBuilder {
        nodes: Vec::new(),
        context,
        locals: Vec::new(),
    };
    if let Some(node) = try_lower_object_items_pipeline(&mut builder, &ast) {
        return builder.finish(node);
    }
    let top_level_pipeline = if should_skip_field_chain_pipeline(&ast) {
        None
    } else {
        lower_via_logical(&ast).or_else(|| Pipeline::lower(&ast))
    };
    if let Some(pipeline) = top_level_pipeline {
        if !is_scalar_unwrap_pipeline(&pipeline) {
            let (source, mut body) = pipeline.into_source_body();
            mask_active_local_stage_kernels(&mut body, &builder);
            if let Some(node) = pipeline_parts_to_plan_node(source, body) {
                let root = builder.push(node);
                return builder.finish(root);
            }
        }
    }
    let root = lower_expr(&mut builder, &ast);
    builder.finish(root)
}

fn expr_is_direct_view_projection_chain(expr: &Expr) -> bool {
    let Expr::Chain(base, steps) = expr else {
        return false;
    };
    if steps.is_empty() || !matches!(base.as_ref(), Expr::Root | Expr::Ident(_)) {
        return false;
    }
    let Some((last, prefix)) = steps.split_last() else {
        return false;
    };
    if !receiver_pipeline_step_is_direct_view_projection(last) {
        return false;
    }
    prefix.iter().all(|step| {
        matches!(
            step,
            Step::Field(_) | Step::OptField(_) | Step::Index(_) | Step::DynIndex(_)
        )
    })
}

fn try_lower_object_items_pipeline(builder: &mut PlanBuilder, expr: &Expr) -> Option<NodeId> {
    let Expr::Chain(base, steps) = expr else {
        return None;
    };
    if !matches!(base.as_ref(), Expr::Root) {
        return None;
    }

    let mut field_end = 0;
    for step in steps {
        match step {
            Step::Field(_) => field_end += 1,
            _ => break,
        }
    }
    if field_end >= steps.len() {
        return None;
    }
    let Step::Method(name, args) = &steps[field_end] else {
        return None;
    };
    let call = BuiltinCall::from_literal_ast_args(name.as_str(), args)?;
    let projection = view_object_items_projection_call(call.id(), &call.args)?;

    let keys: Arc<[Arc<str>]> = steps[..field_end]
        .iter()
        .map(|step| match step {
            Step::Field(key) => Arc::from(key.as_str()),
            _ => unreachable!(),
        })
        .collect::<Vec<_>>()
        .into();
    let mut stages = vec![crate::exec::pipeline::Stage::ObjectItems(projection)];
    let mut stage_exprs = vec![None];
    let sink = if field_end + 1 < steps.len() {
        let tail = Pipeline::lower_body_from_steps(&steps[field_end + 1..])?;
        stages.extend(tail.stages);
        stage_exprs.extend(tail.stage_exprs);
        tail.sink
    } else {
        crate::exec::pipeline::Sink::Collect
    };
    let body = crate::exec::pipeline::PipelineBody::planned(stages, stage_exprs, sink);
    Some(builder.push(PlanNode::Pipeline {
        source: PipelinePlanSource::FieldChain { keys },
        body,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::physical::{
        BackendPreference, PhysicalObjField, PipelinePlanSource, PlanNode, QueryRoot,
    };
    use crate::parse::ast::BinOp;

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
    fn byte_context_lowers_bare_ident_to_root_path() {
        let plan = plan_query_with_context("name", PlanningContext::bytes());
        assert!(matches!(
            root_node(&plan),
            PlanNode::RootPath(steps)
                if matches!(steps.as_slice(), [PhysicalPathStep::Field(key)] if key.as_ref() == "name")
        ));
        assert!(plan.root_execution_facts().is_byte_native());
    }

    #[test]
    fn byte_context_lowers_bare_ident_chain_to_root_path() {
        let plan = plan_query_with_context("user.name", PlanningContext::bytes());
        assert!(matches!(
            root_node(&plan),
            PlanNode::RootPath(steps)
                if matches!(
                    steps.as_slice(),
                    [PhysicalPathStep::Field(user), PhysicalPathStep::Field(name)]
                        if user.as_ref() == "user" && name.as_ref() == "name"
                )
        ));
        assert!(plan.root_execution_facts().is_byte_native());
    }

    #[test]
    fn byte_context_lowers_bare_ident_method_receiver_to_root_path() {
        let plan = plan_query_with_context("attributes.len()", PlanningContext::bytes());
        let PlanNode::Call { receiver, .. } = root_node(&plan) else {
            panic!("expected direct call");
        };
        assert!(matches!(
            plan.node(*receiver),
            PlanNode::RootPath(steps)
                if matches!(steps.as_slice(), [PhysicalPathStep::Field(key)] if key.as_ref() == "attributes")
        ));
        assert!(plan.root_execution_facts().is_byte_native());
    }

    #[test]
    fn byte_context_lowers_bare_filter_count_to_view_pipeline() {
        let plan = plan_query_with_context(
            r#"attributes.filter(@.value.contains("_3")).len()"#,
            PlanningContext::bytes(),
        );
        let PlanNode::Pipeline { body, .. } = root_node(&plan) else {
            panic!("expected pipeline");
        };
        assert!(body.can_run_with_view(), "{body:?}");
    }

    #[test]
    fn byte_context_lowers_bare_first_suffix_to_pipeline_chain() {
        let plan = plan_query_with_context("attributes.first().value", PlanningContext::bytes());
        let PlanNode::Chain { base, steps } = root_node(&plan) else {
            panic!(
                "expected chain, got {:?}",
                std::mem::discriminant(root_node(&plan))
            );
        };
        assert!(
            matches!(steps.as_slice(), [PhysicalChainStep::Field(key)] if key.as_ref() == "value")
        );
        assert!(
            matches!(
                plan.node(*base),
                PlanNode::Pipeline { .. } | PlanNode::Call { .. }
            ),
            "{:?}",
            std::mem::discriminant(plan.node(*base))
        );
        if let PlanNode::Pipeline { body, .. } = plan.node(*base) {
            assert!(body.stages.is_empty(), "{body:?}");
        }
    }

    #[test]
    fn byte_context_lowers_terminal_pipeline_with_path_suffix() {
        let plan = plan_query_with_context(
            "$.attributes.sort_by(@.value).last().key",
            PlanningContext::bytes(),
        );
        let PlanNode::Chain { base, steps } = root_node(&plan) else {
            panic!("expected suffix chain over terminal pipeline");
        };
        assert!(
            matches!(steps.as_slice(), [PhysicalChainStep::Field(key)] if key.as_ref() == "key")
        );
        assert!(matches!(plan.node(*base), PlanNode::Pipeline { .. }));
    }

    #[test]
    fn val_context_keeps_bare_ident_semantics() {
        let plan = plan_query_with_context("name", PlanningContext::val());
        assert!(matches!(root_node(&plan), PlanNode::Ident(name) if name.as_ref() == "name"));
    }

    #[test]
    fn functional_update_lowers_to_physical_update_batch() {
        let plan = plan_query(r#"$.books[*].update({ tags: tags.append("test") })"#);
        let QueryRoot::Node(root) = plan.root() else {
            panic!("expected physical plan");
        };
        let PlanNode::UpdateBatch {
            selector,
            ops,
            dependencies,
            trie,
            ..
        } = plan.node(*root)
        else {
            panic!("expected update batch plan");
        };
        assert_eq!(selector.len(), 2);
        assert_eq!(ops.len(), 1);
        assert!(!dependencies.reads_root);
        assert!(dependencies.reads_focus);
        assert!(!dependencies.reads_current);
        assert!(!dependencies.has_dynamic_path);
        assert!(trie.static_prefixes_only);
        assert_eq!(trie.op_count, 1);
        assert_eq!(
            plan.backend_preferences(*root),
            &[BackendPreference::Interpreted]
        );
        let facts = plan.execution_facts(*root);
        assert!(facts.contains_vm_fallback);
        assert!(!facts.may_materialize_source);
    }

    #[test]
    fn backend_policy_is_centralized_for_input_context() {
        let node = PlanNode::Pipeline {
            source: PipelinePlanSource::FieldChain {
                keys: Arc::from([Arc::<str>::from("rows")]),
            },
            body: crate::exec::pipeline::PipelineBody {
                stages: Vec::new(),
                stage_exprs: Vec::new(),
                sink: crate::exec::pipeline::Sink::Collect,
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
            .contains(crate::ir::physical::BackendSet::TAPE_VIEW));
        assert!(plan
            .backend_capabilities(*root)
            .contains(crate::ir::physical::BackendSet::VAL_VIEW));
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
        assert!(matches!(
            body.stages[0],
            crate::exec::pipeline::Stage::Sort(_)
        ));
        assert!(matches!(
            body.stages[1],
            crate::exec::pipeline::Stage::ExprBuiltin {
                method: crate::builtins::BuiltinMethod::TakeWhile,
                ..
            }
        ));
        assert!(matches!(
            body.stages[2],
            crate::exec::pipeline::Stage::UsizeBuiltin {
                method: crate::builtins::BuiltinMethod::Take,
                value: 2
            }
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
    fn root_path_view_projection_lowers_to_byte_native_call() {
        let plan = plan_query(r#"$.user.pick("name")"#);
        let QueryRoot::Node(root) = plan.root() else {
            panic!("expected physical plan");
        };
        assert!(plan.execution_facts(*root).is_byte_native());
        let PlanNode::Call { receiver, .. } = plan.node(*root) else {
            panic!("expected direct call");
        };
        assert!(matches!(
            plan.node(*receiver),
            PlanNode::RootPath(steps)
                if matches!(steps.as_slice(), [PhysicalPathStep::Field(key)] if key.as_ref() == "user")
        ));
    }

    #[test]
    fn root_path_scalar_view_projection_lowers_to_byte_native_call() {
        let plan = plan_query(r#"$.user.name.upper()"#);
        let QueryRoot::Node(root) = plan.root() else {
            panic!("expected physical plan");
        };
        assert!(plan.execution_facts(*root).is_byte_native());
        let PlanNode::Call { receiver, .. } = plan.node(*root) else {
            panic!("expected direct call");
        };
        assert!(matches!(
            plan.node(*receiver),
            PlanNode::RootPath(steps)
                if matches!(
                    steps.as_slice(),
                    [PhysicalPathStep::Field(user), PhysicalPathStep::Field(name)]
                    if user.as_ref() == "user" && name.as_ref() == "name"
                )
        ));
    }

    #[test]
    fn object_item_projection_prefix_lowers_as_field_chain_pipeline() {
        let plan = plan_query(r#"$.profile.entries().first()"#);
        let QueryRoot::Node(root) = plan.root() else {
            panic!("expected physical plan");
        };
        match plan.node(*root) {
            PlanNode::Pipeline {
                source: PipelinePlanSource::FieldChain { .. },
                body,
            } => assert!(matches!(
                body.stages.first(),
                Some(crate::exec::pipeline::Stage::ObjectItems(
                    crate::builtins::BuiltinViewObjectProjection::Entries
                ))
            )),
            _ => panic!("entries().first() must stream object items from a field-chain source"),
        }
    }

    #[test]
    fn root_object_item_projection_lowers_as_empty_field_chain_pipeline() {
        let plan = plan_query(r#"$.values().first()"#);
        let QueryRoot::Node(root) = plan.root() else {
            panic!("expected physical plan");
        };
        match plan.node(*root) {
            PlanNode::Pipeline {
                source: PipelinePlanSource::FieldChain { keys },
                body,
            } => {
                assert!(keys.is_empty());
                assert!(matches!(
                    body.stages.first(),
                    Some(crate::exec::pipeline::Stage::ObjectItems(
                        crate::builtins::BuiltinViewObjectProjection::Values
                    ))
                ));
            }
            _ => panic!("root values().first() must stream root object items"),
        }
    }

    #[test]
    fn nested_object_item_projection_lowers_as_field_chain_pipeline() {
        let plan = plan_query(r#"{first: $.profile.values().first()}"#);
        let PlanNode::Object(fields) = root_node(&plan) else {
            panic!("expected object plan");
        };
        let PhysicalObjField::Kv { val, .. } = &fields[0] else {
            panic!("expected object field");
        };
        match plan.node(*val) {
            PlanNode::Pipeline {
                source: PipelinePlanSource::FieldChain { .. },
                body,
            } => assert!(matches!(
                body.stages.first(),
                Some(crate::exec::pipeline::Stage::ObjectItems(
                    crate::builtins::BuiltinViewObjectProjection::Values
                ))
            )),
            _ => panic!("nested values().first() must stream object items from a field-chain source"),
        }
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
    fn root_receiver_method_chain_lowers_without_vm_fallback() {
        let plan = plan_query(r#"$.sort()"#);
        let QueryRoot::Node(root) = plan.root() else {
            panic!("expected physical plan");
        };
        assert!(!plan.execution_facts(*root).contains_vm_fallback);
        let PlanNode::Call { receiver, .. } = plan.node(*root) else {
            panic!("expected root receiver call");
        };
        assert!(matches!(plan.node(*receiver), PlanNode::RootPath(steps) if steps.is_empty()));
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
