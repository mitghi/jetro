//! Pipeline IR type accessors and descriptor helpers.
//!
//! This module defines the descriptor-level types (`StageDescriptor`, `StageShape`,
//! `StageStrategy`, `Strategy`, `Plan`, `Position`, `SinkDemand`) and the `impl` blocks for
//! `Stage`, `PipelineBody`, `Sink`, and `Pipeline` that expose IR-level metadata used by the
//! planner.  Pure planning algorithms (filter reordering, strategy selection, etc.) live in
//! `plan.rs`.

use std::sync::Arc;

use crate::builtins::registry::{
    effective_pipeline_order_effect, participates_in_demand, pipeline_composed_barrier,
    pipeline_legacy_materialized, pipeline_shape, pipeline_streams,
    sink_demand as builtin_sink_demand, BuiltinId,
};
use crate::builtins::{
    BuiltinCardinality, BuiltinDemandLaw, BuiltinMethod, BuiltinPipelineOrderEffect,
    BuiltinSelectionPosition, BuiltinSinkAccumulator, BuiltinSinkSpec, BuiltinViewStage,
};
use crate::parse::ast::Expr;
use crate::plan::chain_ir::{ChainOp, MatchRole};
use crate::plan::demand::{
    Demand as ChainDemand, DemandLanes, FieldDemand, PullDemand, SinkResultDemand, ValueNeed,
};
use crate::vm::{CompiledObjEntry, Opcode, Program};

use super::{
    BodyKernel, Pipeline, PipelineBody, PredicateSinkOp, ReducerOp, Sink, Stage,
    ViewMembershipTarget, ViewSinkCapability, ViewStageCapability,
};

/// Indicates whether a positional terminal sink wants the first or the last qualifying element.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Position {
    /// The sink wants only the first element that satisfies all upstream stages.
    First,
    /// The sink wants only the last element that satisfies all upstream stages.
    Last,
}

/// Combined demand description for a sink: how many elements to pull from upstream, plus an
/// optional positional preference used to pick between top-K and bottom-K sort strategies.
#[derive(Debug, Clone, Copy)]
pub struct SinkDemand {
    /// Element-level pull/value/order requirements propagated to the source chain.
    pub chain: ChainDemand,
    /// Scalar sink-result decision demand, separate from row-output pull demand.
    pub sink_result: SinkResultDemand,
    /// Set when the sink selects exactly one element by position (first/last).
    pub positional: Option<Position>,
}

impl SinkDemand {
    /// A "pull everything, materialise fully" demand used as the baseline when no tighter
    /// demand can be computed.
    pub const RESULT: SinkDemand = SinkDemand {
        chain: ChainDemand::RESULT,
        sink_result: SinkResultDemand::None,
        positional: None,
    };

    /// Returns true when the terminal scalar result can stop the executor loop
    /// before row-output pull demand is satisfied.
    #[cfg(test)]
    pub(crate) fn has_scalar_short_circuit(self) -> bool {
        self.sink_result.can_short_circuit()
    }
}

/// Source payload demand split into scan-time and result-row lanes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PayloadDemand {
    /// Fields/value needed while deciding which rows survive.
    pub scan_need: FieldDemand,
    /// Fields/value needed only for rows that are emitted or retained by the sink.
    pub result_need: FieldDemand,
}

/// Explicit pending projection run discovered at the tail of a pipeline.
#[derive(Debug, Clone)]
pub struct LateProjection {
    /// Number of leading stages that must run before the delayed projection is applied.
    pub prefix_len: usize,
    /// Composed projection kernel to apply to selected rows.
    pub kernel: BodyKernel,
}

/// Boundary where the pipeline must leave the current low-materialization path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackBoundary {
    /// No known materialization/fallback boundary in this pipeline.
    None,
    /// Stage at `index` requires legacy materialization.
    LegacyStage {
        /// Index of the first legacy-materialized stage.
        index: usize,
    },
    /// The selected physical path can fall back to materialized execution.
    MaterializedExecution,
}

impl From<DemandLanes> for PayloadDemand {
    fn from(value: DemandLanes) -> Self {
        Self {
            scan_need: value.scan_need,
            result_need: value.result_need,
        }
    }
}

impl Sink {
    /// Computes the `SinkDemand` for this sink by consulting its builtin metadata, falling back
    /// to `SinkDemand::RESULT` for sinks with no registered spec.
    pub fn demand(&self) -> SinkDemand {
        if let Sink::Nth(idx) = self {
            return SinkDemand {
                chain: ChainDemand {
                    pull: PullDemand::NthInput(*idx),
                    value: ValueNeed::Whole,
                    order: false,
                },
                sink_result: SinkResultDemand::None,
                positional: Some(Position::First),
            };
        }
        if matches!(self, Sink::SelectMany { .. }) {
            return SinkDemand {
                chain: self.select_many_demand().expect("checked SelectMany"),
                sink_result: SinkResultDemand::None,
                positional: None,
            };
        }
        if let Sink::Predicate(spec) = self {
            return SinkDemand {
                chain: spec.demand(),
                sink_result: spec.sink_result_demand(),
                positional: None,
            };
        }
        if let Sink::Membership(spec) = self {
            return SinkDemand {
                chain: spec.demand(),
                sink_result: spec.sink_result_demand(),
                positional: None,
            };
        }
        if let Sink::ArgExtreme(spec) = self {
            return SinkDemand {
                chain: spec.demand(),
                sink_result: SinkResultDemand::None,
                positional: None,
            };
        }
        if let Some(spec) = self.builtin_sink_spec() {
            return sink_demand_from_builtin(spec);
        }
        SinkDemand::RESULT
    }

    /// Returns true when this sink can apply a delayed projection only to the
    /// rows it actually emits or retains.
    pub(crate) fn supports_late_projection(&self, demand: PullDemand) -> bool {
        match self {
            Sink::Collect => !matches!(demand, PullDemand::LastInput(_)),
            Sink::Terminal(BuiltinMethod::First | BuiltinMethod::Last)
            | Sink::Nth(_)
            | Sink::SelectMany { .. } => true,
            _ => false,
        }
    }

    /// Returns `true` when every sub-program in the sink (predicate, projection) satisfies
    /// `program_ok`, meaning the sink can execute against a materialised receiver without a
    /// document-root lookup.
    pub(crate) fn can_run_with_receiver_only<F>(&self, mut program_ok: F) -> bool
    where
        F: FnMut(&crate::vm::Program) -> bool,
    {
        match self {
            Sink::Collect
            | Sink::Terminal(_)
            | Sink::SelectMany { .. }
            | Sink::Nth(_)
            | Sink::ApproxCountDistinct => true,
            Sink::Membership(spec) => spec.sink_programs().all(|prog| program_ok(prog)),
            Sink::Predicate(spec) => program_ok(&spec.predicate),
            Sink::ArgExtreme(spec) => program_ok(&spec.key),
            Sink::Reducer(spec) => spec.sink_programs().all(|prog| program_ok(prog)),
        }
    }

    /// Classifies embedded sink programs into body kernels for payload planning and view routing.
    pub(crate) fn body_kernels(&self) -> Vec<BodyKernel> {
        match self {
            Sink::Reducer(spec) => spec
                .sink_programs()
                .map(|p| BodyKernel::classify(p))
                .collect(),
            Sink::Predicate(spec) => spec
                .sink_programs()
                .map(|p| BodyKernel::classify(p))
                .collect(),
            Sink::Membership(spec) => spec
                .sink_programs()
                .map(|p| BodyKernel::classify(p))
                .collect(),
            Sink::ArgExtreme(spec) => spec
                .sink_programs()
                .map(|p| BodyKernel::classify(p))
                .collect(),
            _ => Vec::new(),
        }
    }

    /// Returns the `ViewSinkCapability` if the sink can operate in the borrowed `ValueView`
    /// domain, or `None` if full materialisation is required.
    pub(crate) fn view_capability(
        &self,
        sink_kernels: &[BodyKernel],
    ) -> Option<ViewSinkCapability> {
        if matches!(self, Sink::Collect) {
            return Some(ViewSinkCapability::Collect);
        }
        if let Sink::SelectMany { n, from_end } = self {
            return Some(ViewSinkCapability::SelectMany {
                n: *n,
                from_end: *from_end,
                source_reversed: false,
            });
        }
        if let Sink::Nth(index) = self {
            return Some(ViewSinkCapability::Nth { index: *index });
        }
        if let Sink::Predicate(spec) = self {
            return Some(ViewSinkCapability::Predicate {
                op: spec.op,
                predicate_kernel: view_native_sink_kernel(
                    sink_kernels,
                    spec.predicate_kernel_index(),
                )?,
            });
        }
        if let Sink::Membership(spec) = self {
            return Some(ViewSinkCapability::Membership {
                op: spec.op,
                target: ViewMembershipTarget::from(&spec.target),
            });
        }
        if let Sink::ArgExtreme(spec) = self {
            return Some(ViewSinkCapability::ArgExtreme {
                want_max: spec.want_max,
                key_kernel: view_native_sink_kernel(sink_kernels, spec.key_kernel_index())?,
            });
        }

        let sink_spec = self.builtin_sink_spec()?;
        let reducer = self.reducer_spec();
        let predicate_kernel = match reducer
            .as_ref()
            .and_then(|spec| spec.predicate_kernel_index())
        {
            Some(idx) => Some(view_native_sink_kernel(sink_kernels, idx)?),
            None => None,
        };
        let project_kernel = match reducer
            .as_ref()
            .and_then(|spec| spec.projection_kernel_index())
        {
            Some(idx) => Some(view_native_sink_kernel(sink_kernels, idx)?),
            None => None,
        };
        if sink_spec.accumulator == BuiltinSinkAccumulator::Numeric {
            reducer.as_ref()?.numeric_op()?;
        }

        Some(ViewSinkCapability::from_sink_spec(
            sink_spec,
            predicate_kernel,
            project_kernel,
        ))
    }

    /// Looks up the `BuiltinSinkSpec` for this sink from the builtin registry, returning `None`
    /// for `Sink::Collect` which has no associated spec.
    pub(crate) fn builtin_sink_spec(&self) -> Option<BuiltinSinkSpec> {
        match self {
            Sink::Terminal(method) => method.spec().sink,
            Sink::Nth(_) => None,
            Sink::SelectMany { .. } => None,
            Sink::Predicate(_) => None,
            Sink::Membership(_) => None,
            Sink::ArgExtreme(_) => None,
            Sink::Reducer(spec) => spec.method()?.spec().sink,
            Sink::ApproxCountDistinct => BuiltinMethod::ApproxCountDistinct.spec().sink,
            Sink::Collect => None,
        }
    }

    fn select_many_demand(&self) -> Option<ChainDemand> {
        let Sink::SelectMany { n, from_end } = self else {
            return None;
        };
        Some(ChainDemand {
            pull: if *from_end {
                PullDemand::LastInput(*n)
            } else {
                PullDemand::FirstInput(*n)
            },
            value: ValueNeed::Whole,
            order: true,
        })
    }
}

fn view_native_sink_kernel(sink_kernels: &[BodyKernel], idx: usize) -> Option<usize> {
    sink_kernels.get(idx)?.is_view_native().then_some(idx)
}

fn sink_demand_from_builtin(spec: BuiltinSinkSpec) -> SinkDemand {
    SinkDemand {
        chain: builtin_sink_demand(spec),
        sink_result: SinkResultDemand::None,
        positional: match spec.accumulator {
            BuiltinSinkAccumulator::SelectOne(position) => Some(position.into()),
            _ => None,
        },
    }
}

impl From<BuiltinSelectionPosition> for Position {
    fn from(value: BuiltinSelectionPosition) -> Self {
        match value {
            BuiltinSelectionPosition::First => Position::First,
            BuiltinSelectionPosition::Last => Position::Last,
        }
    }
}

/// Execution strategy chosen by the planner for a `Stage::Sort` when downstream demand is bounded.
#[derive(Debug, Clone, Copy)]
pub enum StageStrategy {
    /// No special treatment; execute the stage with its default implementation.
    Default,
    /// Use a min-heap of size `k` to produce the `k` smallest (or top-key) elements without a full
    /// sort.
    SortTopK(usize),
    /// Use a max-heap of size `k` to produce the `k` largest elements without a full sort.
    SortBottomK(usize),
    /// Sort lazily and stop emitting once `k` outputs have passed all downstream filters.
    SortUntilOutput(usize),
}

/// Static cost/cardinality metadata for a pipeline stage, used by the planner to pick
/// execution strategies and reorder filter runs.
#[derive(Debug, Clone, Copy)]
pub struct StageShape {
    /// Whether the stage emits one-to-one, fewer, more, or barrier-level output rows.
    pub cardinality: BuiltinCardinality,
    /// `true` when the stage supports position-indexed execution (used for `IndexedDispatch`).
    pub can_indexed: bool,
    /// Estimated relative CPU cost per element passing through the stage.
    pub cost: f64,
    /// Fraction of elements expected to pass through (1.0 = all pass, 0.0 = none pass).
    pub selectivity: f64,
}

impl StageShape {
    pub(crate) fn from_view_stage(stage: BuiltinViewStage) -> Self {
        Self {
            cardinality: stage.cardinality(),
            can_indexed: stage.can_indexed(),
            cost: stage.cost(),
            selectivity: stage.selectivity(),
        }
    }

    pub(crate) fn from_builtin(method: BuiltinMethod) -> Self {
        use crate::builtins::BuiltinCategory;

        let spec = method.spec();
        if let Some(shape) = pipeline_shape(BuiltinId::from_method(method)) {
            return Self {
                cardinality: shape.cardinality,
                can_indexed: shape.can_indexed,
                cost: shape.cost,
                selectivity: shape.selectivity,
            };
        }
        Self {
            cardinality: spec.cardinality,
            can_indexed: spec.can_indexed,
            cost: spec.cost,
            selectivity: if matches!(spec.category, BuiltinCategory::StreamingFilter) {
                0.5
            } else {
                1.0
            },
        }
    }
}

/// A unified descriptor for a `Stage`, providing the canonical method, body program,
/// numeric argument, and execution/view-stage overrides needed by the planner.
#[derive(Debug, Clone, Copy)]
pub(crate) struct StageDescriptor<'a> {
    /// The `BuiltinMethod` this stage maps to, if any.
    pub method: Option<BuiltinMethod>,
    /// The compiled predicate or projection program carried by the stage, if any.
    pub body: Option<&'a Program>,
    /// Integer argument (e.g. the `n` in `take(n)`), if applicable.
    pub usize_arg: Option<usize>,
    view_stage_override: Option<BuiltinViewStage>,
    // when true, a one-to-one stage may fall back to Preserves order effect
    allow_one_to_one_order_fallback: bool,
    // when true, the stage is safe to run against a materialised receiver with no body program
    receiver_safe_without_body: bool,
}

impl<'a> StageDescriptor<'a> {
    #[inline]
    pub(crate) fn new(method: BuiltinMethod) -> Self {
        Self {
            method: Some(method),
            body: None,
            usize_arg: None,
            view_stage_override: None,
            allow_one_to_one_order_fallback: false,
            receiver_safe_without_body: true,
        }
    }

    // Used for stages with no builtin method (SortedDedup, CompiledMap synthetics).
    #[inline]
    pub(crate) fn special() -> Self {
        Self {
            method: None,
            body: None,
            usize_arg: None,
            view_stage_override: None,
            allow_one_to_one_order_fallback: false,
            receiver_safe_without_body: true,
        }
    }

    #[inline]
    pub(crate) fn body(mut self, body: &'a Program) -> Self {
        self.body = Some(body);
        self
    }

    #[inline]
    pub(crate) fn usize_arg(mut self, usize_arg: usize) -> Self {
        self.usize_arg = Some(usize_arg);
        self
    }

    #[inline]
    pub(crate) fn with_view_stage(mut self, stage: BuiltinViewStage) -> Self {
        self.view_stage_override = Some(stage);
        self
    }

    #[inline]
    pub(crate) fn allow_one_to_one_order_fallback(mut self) -> Self {
        self.allow_one_to_one_order_fallback = true;
        self
    }

    // Marks stages like `CompiledMap` that have no fallback without their body program.
    #[inline]
    pub(crate) fn receiver_unsafe_without_body(mut self) -> Self {
        self.receiver_safe_without_body = false;
        self
    }

    /// Returns the effective `BuiltinViewStage` for this descriptor, using the override if set
    /// or falling back to the method's registered view stage.
    #[inline]
    pub(crate) fn view_stage(self) -> Option<BuiltinViewStage> {
        self.view_stage_override
            .or_else(|| self.method.and_then(|method| method.spec().view_stage))
    }

    /// Returns the columnar-stage metadata for the method, if it supports columnar execution.
    #[inline]
    pub(crate) fn columnar_stage(self) -> Option<crate::builtins::BuiltinColumnarStage> {
        self.method.and_then(|method| method.spec().columnar_stage)
    }

    /// Returns how this stage affects the sort order of its input stream.
    #[inline]
    pub(crate) fn pipeline_order_effect(self) -> BuiltinPipelineOrderEffect {
        let Some(method) = self.method else {
            return BuiltinPipelineOrderEffect::Blocks;
        };
        effective_pipeline_order_effect(
            BuiltinId::from_method(method),
            self.allow_one_to_one_order_fallback,
        )
    }

    /// Returns `true` when the stage can run using only a materialised receiver, delegating
    /// to `program_ok` for the body program if one is present.
    #[inline]
    pub(crate) fn can_run_with_receiver_only<F>(self, program_ok: F) -> bool
    where
        F: FnMut(&crate::vm::Program) -> bool,
    {
        self.body
            .map(program_ok)
            .unwrap_or(self.receiver_safe_without_body)
    }
}

macro_rules! view_body_stage_descriptor {
    ($stage:expr, { $($variant:ident => $method:ident),+ $(,)? }) => {
        match $stage {
            $(
                Stage::$variant(prog, view_stage) => {
                    Some(StageDescriptor::new(BuiltinMethod::$method).body(prog).with_view_stage(*view_stage))
                },
            )+
            _ => None,
        }
    };
}

macro_rules! method_stage_descriptor {
    ($stage:expr, { $($pattern:pat => $method:ident),+ $(,)? }) => {
        match $stage {
            $(
                $pattern => Some(StageDescriptor::new(BuiltinMethod::$method)),
            )+
            _ => None,
        }
    };
}

impl Stage {
    /// Classifies the stage body program, returning `Generic` for stages without a body.
    pub(crate) fn body_kernel(&self) -> BodyKernel {
        self.body_program()
            .map(BodyKernel::classify)
            .unwrap_or(BodyKernel::Generic)
    }

    /// Classifies all stage body programs in order.
    pub(crate) fn body_kernels(stages: &[Stage]) -> Vec<BodyKernel> {
        stages.iter().map(Self::body_kernel).collect()
    }

    /// Returns `true` when this stage requires a composed-barrier materialisation pass before
    /// the next stage can begin.
    pub(crate) fn is_composed_barrier(&self) -> bool {
        self.descriptor()
            .and_then(|desc| desc.method)
            .is_some_and(|method| pipeline_composed_barrier(BuiltinId::from_method(method)))
    }

    /// Returns `true` when the stage cannot participate in the streaming pull loop and must be
    /// executed via the legacy materialisation path.
    pub(crate) fn requires_legacy_materialization(&self) -> bool {
        matches!(self, Stage::SortedDedup(_))
            || (!matches!(self, Stage::CompiledMap(_))
                && self
                    .descriptor()
                    .and_then(|desc| desc.method)
                    .is_some_and(|method| !pipeline_streams(BuiltinId::from_method(method))))
    }

    /// Returns `true` when the stage cannot use a composed/view barrier and must fall back to the
    /// legacy full-materialisation executor.
    pub(crate) fn requires_legacy_fallback(&self) -> bool {
        matches!(self, Stage::SortedDedup(_))
            || self
                .descriptor()
                .and_then(|desc| desc.method)
                .is_some_and(|method| pipeline_legacy_materialized(BuiltinId::from_method(method)))
    }

    /// Returns the `ViewStageCapability` for this stage at position `idx` in the kernel list,
    /// or `None` if the stage or its kernel cannot operate in the borrowed `ValueView` domain.
    pub(crate) fn view_capability(
        &self,
        idx: usize,
        kernel: Option<&BodyKernel>,
    ) -> Option<ViewStageCapability> {
        let desc = self.descriptor()?;
        let stage = desc.view_stage()?;
        if stage == BuiltinViewStage::RemoveValue {
            return match self {
                Stage::Builtin(call) => match &call.args {
                    crate::builtins::BuiltinArgs::Val(target) => {
                        Some(ViewStageCapability::RemoveValue(target.clone()))
                    }
                    _ => None,
                },
                _ => None,
            };
        }
        if stage == BuiltinViewStage::Distinct {
            return match desc.body {
                Some(_) if kernel.is_some_and(BodyKernel::is_view_native) => {
                    Some(ViewStageCapability::Distinct { kernel: Some(idx) })
                }
                Some(_) => None,
                None => Some(ViewStageCapability::Distinct { kernel: None }),
            };
        }
        if stage == BuiltinViewStage::KeyedReduce {
            return match (desc.method, desc.body) {
                (Some(method), Some(_)) if kernel.is_some_and(BodyKernel::is_view_native) => {
                    let kind = method.spec().keyed_reducer?;
                    Some(ViewStageCapability::KeyedReduce { kind, kernel: idx })
                }
                _ => None,
            };
        }
        ViewStageCapability::from_stage_metadata(
            stage,
            desc.usize_arg,
            idx,
            kernel.is_some_and(BodyKernel::is_view_native),
        )
    }

    /// Builds a `StageDescriptor` for this stage, providing the canonical method / body /
    /// executor metadata used throughout the planner; returns `None` for unrecognised variants.
    pub(crate) fn descriptor(&self) -> Option<StageDescriptor<'_>> {
        if let Some(desc) = view_body_stage_descriptor!(self, {
            Filter => Filter,
            Map => Map,
            FlatMap => FlatMap,
        }) {
            return Some(desc);
        }
        if let Some(desc) = method_stage_descriptor!(self, {
            Stage::Reverse(_) => Reverse,
            Stage::UniqueBy(None) => Unique,
        }) {
            return Some(desc);
        }

        match self {
            Stage::UniqueBy(Some(prog)) => {
                Some(StageDescriptor::new(BuiltinMethod::UniqueBy).body(prog))
            }
            Stage::Sort(super::SortSpec { key, .. }) => {
                let desc = StageDescriptor::new(BuiltinMethod::Sort);
                Some(if let Some(prog) = key {
                    desc.body(prog)
                } else {
                    desc
                })
            }
            Stage::UsizeBuiltin { method, value } => {
                Some(StageDescriptor::new(*method).usize_arg(*value))
            }
            Stage::StringBuiltin { method, .. } | Stage::StringPairBuiltin { method, .. } => {
                Some(StageDescriptor::new(*method))
            }
            Stage::IntRangeBuiltin { method, .. } => Some(StageDescriptor::new(*method)),
            Stage::ExprBuiltin { method, body } => Some(StageDescriptor::new(*method).body(body)),
            Stage::Builtin(call) => {
                Some(StageDescriptor::new(call.method).allow_one_to_one_order_fallback())
            }
            Stage::SortedDedup(prog) => {
                let desc = StageDescriptor::special();
                Some(if let Some(prog) = prog {
                    desc.body(prog)
                } else {
                    desc
                })
            }
            Stage::CompiledMap(_) => {
                Some(StageDescriptor::special().receiver_unsafe_without_body())
            }
            _ => None,
        }
    }

    /// Returns `true` when this stage can execute using only the materialised receiver, with
    /// `program_ok` used to validate the stage's body program if present.
    pub(crate) fn can_run_with_receiver_only<F>(&self, mut program_ok: F) -> bool
    where
        F: FnMut(&crate::vm::Program) -> bool,
    {
        self.descriptor()
            .is_some_and(|desc| desc.can_run_with_receiver_only(&mut program_ok))
    }

    /// Returns the compiled body program carried by this stage, or `None` if the stage has no
    /// sub-expression (e.g. `Reverse`, `Take`).
    pub(crate) fn body_program(&self) -> Option<&crate::vm::Program> {
        self.descriptor().and_then(|desc| desc.body)
    }

    /// Returns `true` when this stage can be consumed by the `TerminalMapCollector`
    /// optimisation (requires Map shape and a body program).
    pub(crate) fn can_use_terminal_map_collector(&self) -> bool {
        match self {
            Stage::Map(_, _) => true,
            Stage::CompiledMap(_) => true,
            _ => false,
        }
    }

    /// Returns `true` when this stage performs a per-element value transformation that the
    /// demand optimiser can track symbolically (substitute `@` in downstream predicates).
    /// Direct Stage-variant match — no executor enum lookup.
    pub(crate) fn is_symbolic_map_stage(&self) -> bool {
        matches!(self, Stage::CompiledMap(_) | Stage::Map(_, _))
    }

    /// Returns `true` when this stage is a filter whose predicate can be substituted symbolically
    /// by the demand optimiser after a map transformation.
    pub(crate) fn is_symbolic_filter_stage(&self) -> bool {
        matches!(self, Stage::Filter(_, _))
    }

    /// Returns `true` when this stage uses a positional / bounded executor (e.g. `Take`, `Skip`),
    /// meaning order must be preserved upstream.
    pub(crate) fn is_positional_stage(&self) -> bool {
        matches!(
            self,
            Stage::UsizeBuiltin {
                method: BuiltinMethod::Take | BuiltinMethod::Skip,
                ..
            }
        )
    }

    /// Returns `true` when this stage only changes element order without affecting membership
    /// (e.g. `Sort`, `Reverse`), allowing the demand optimiser to drop it when order is unused.
    pub(crate) fn is_order_only_stage(&self) -> bool {
        matches!(self, Stage::Sort(_) | Stage::Reverse(_))
    }

    /// Returns `true` when the stage reads the actual element value rather than just membership
    /// metadata, meaning downstream value-demand cannot be eliminated.
    /// Direct Stage-variant match — Sort with key, lambdas (Filter/Map/FlatMap),
    /// keyed reducers, prefix predicates, and ExprBuiltin all read element value.
    pub(crate) fn consumes_input_value(&self) -> bool {
        match self {
            Stage::Filter(_, _)
            | Stage::Map(_, _)
            | Stage::FlatMap(_, _)
            | Stage::CompiledMap(_)
            | Stage::ExprBuiltin { .. }
            | Stage::UniqueBy(Some(_))
            | Stage::SortedDedup(Some(_)) => true,
            Stage::Sort(spec) => spec.key.is_some(),
            _ => false,
        }
    }

    /// Returns `true` when the stage can be safely eliminated by the demand optimiser when its
    /// output value is never consumed (one-to-one, order-preserving, and pure).
    pub(crate) fn can_drop_when_value_unused(&self) -> bool {
        let Some(desc) = self.descriptor() else {
            return false;
        };
        if !matches!(self.shape().cardinality, BuiltinCardinality::OneToOne) {
            return false;
        }
        if desc.pipeline_order_effect() != BuiltinPipelineOrderEffect::Preserves {
            return false;
        }
        // Element-wise scalar: keep iff method is pure.
        // ObjectLambda variants (TransformKeys/TransformValues/FilterKeys/FilterValues): always droppable.
        match self {
            Stage::Builtin(_) | Stage::IntRangeBuiltin { .. } | Stage::StringPairBuiltin { .. } => {
                desc.method.is_some_and(|m| m.spec().pure)
            }
            Stage::ExprBuiltin {
                method:
                    BuiltinMethod::TransformKeys
                    | BuiltinMethod::TransformValues
                    | BuiltinMethod::FilterKeys
                    | BuiltinMethod::FilterValues,
                ..
            } => true,
            _ => false,
        }
    }

    /// Returns the `ChainOp` representing this stage in the chain-IR demand propagation graph,
    /// or `None` for stages that do not participate in demand propagation.
    pub fn chain_op(&self) -> Option<ChainOp> {
        match self {
            Stage::CompiledMap(_) => Some(ChainOp::builtin(BuiltinMethod::Map)),
            Stage::SortedDedup(_) => None,
            // Filter and Map whose body is a single `match` expression
            // are reported as `ChainOp::Match` so the demand model can
            // reason about them with the role-specific propagation rules
            // (`Predicate` widens upstream demand to scan-until-output;
            // `Transform` is 1:1).
            Stage::Filter(prog, _) if program_is_match_only(prog) => {
                Some(ChainOp::match_role(MatchRole::Predicate))
            }
            Stage::Map(prog, _) if program_is_match_only(prog) => {
                Some(ChainOp::match_role(MatchRole::Transform))
            }
            _ => self.chain_demand_op(),
        }
    }

    fn chain_demand_op(&self) -> Option<ChainOp> {
        let desc = self.descriptor()?;
        let method = desc.method?;
        match self {
            _ if desc.usize_arg.is_some() => Some(ChainOp::builtin_usize(method, desc.usize_arg?)),
            Stage::Builtin(_) => Some(ChainOp::builtin(method)),
            _ if participates_in_demand(BuiltinId::from_method(method)) => {
                Some(ChainOp::builtin(method))
            }
            _ => None,
        }
    }

    /// Propagates `demand` backwards through this stage, returning the demand that must be
    /// satisfied by the stage immediately upstream.
    pub fn upstream_demand(&self, demand: SinkDemand) -> SinkDemand {
        let chain = match self.chain_op() {
            Some(op) => op.propagate_demand(demand.chain),
            None => ChainDemand::RESULT,
        };
        let positional = if matches!(self.shape().cardinality, BuiltinCardinality::OneToOne) {
            demand.positional
        } else {
            None
        };
        SinkDemand {
            chain,
            sink_result: demand.sink_result,
            positional,
        }
    }

    pub(crate) fn ordered_prefix_effect(
        &self,
        sort: &super::SortSpec,
        sort_kernel: &BodyKernel,
        kernel: &BodyKernel,
    ) -> bool {
        match self.pipeline_order_effect() {
            BuiltinPipelineOrderEffect::Preserves => true,
            BuiltinPipelineOrderEffect::PredicatePrefix => {
                super::plan::predicate_is_order_prefix(sort, sort_kernel, kernel)
            }
            BuiltinPipelineOrderEffect::Blocks => false,
        }
    }

    // Hard-coded overrides for CompiledMap (Preserves) and SortedDedup (Blocks).
    fn pipeline_order_effect(&self) -> BuiltinPipelineOrderEffect {
        match self {
            Stage::CompiledMap(_) => BuiltinPipelineOrderEffect::Preserves,
            Stage::SortedDedup(_) => BuiltinPipelineOrderEffect::Blocks,
            _ => self
                .descriptor()
                .map(StageDescriptor::pipeline_order_effect)
                .unwrap_or(BuiltinPipelineOrderEffect::Blocks),
        }
    }

    /// Returns the static `StageShape` (cardinality, cost, selectivity, indexed flag) for this
    /// stage, used by the planner for strategy selection and filter reordering.
    pub fn shape(&self) -> StageShape {
        match self {
            Stage::CompiledMap(_) => StageShape {
                cardinality: BuiltinCardinality::OneToOne,
                can_indexed: true,
                cost: 10.0,
                selectivity: 1.0,
            },
            Stage::SortedDedup(_) => StageShape {
                cardinality: BuiltinCardinality::OneToOne,
                can_indexed: true,
                cost: 1.0,
                selectivity: 1.0,
            },
            _ => self.descriptor().map_or(
                StageShape {
                    cardinality: BuiltinCardinality::OneToOne,
                    can_indexed: false,
                    cost: 1.0,
                    selectivity: 1.0,
                },
                |desc| {
                    desc.view_stage()
                        .map(StageShape::from_view_stage)
                        .or_else(|| desc.method.map(StageShape::from_builtin))
                        .unwrap_or(StageShape {
                            cardinality: BuiltinCardinality::OneToOne,
                            can_indexed: false,
                            cost: 1.0,
                            selectivity: 1.0,
                        })
                },
            ),
        }
    }

    /// Attempts to merge `self` with `other` into a single equivalent stage, returning `None`
    /// when the two stages cannot be combined.
    pub fn merge_with(&self, other: &Self) -> Option<Self> {
        if let Some(merged) = self.merge_with_usize_stage(other) {
            return Some(merged);
        }
        match (self, other) {
            (Stage::Sort(_), Stage::Sort(b)) => Some(Stage::Sort(b.clone())),
            (Stage::UniqueBy(_), Stage::UniqueBy(b)) => Some(Stage::UniqueBy(b.clone())),
            (Stage::UniqueBy(None), Stage::Sort(super::SortSpec { key: None, .. }))
            | (Stage::Sort(super::SortSpec { key: None, .. }), Stage::UniqueBy(None)) => {
                Some(Stage::SortedDedup(None))
            }
            (Stage::UniqueBy(Some(a)), Stage::Sort(super::SortSpec { key: Some(b), .. }))
            | (Stage::Sort(super::SortSpec { key: Some(a), .. }), Stage::UniqueBy(Some(b)))
                if Arc::ptr_eq(a, b) =>
            {
                Some(Stage::SortedDedup(Some(a.clone())))
            }
            (Stage::Builtin(a), Stage::Builtin(b)) if a.method == b.method && a.is_idempotent() => {
                Some(Stage::Builtin(a.clone()))
            }
            _ => None,
        }
    }

    fn merge_with_usize_stage(&self, other: &Self) -> Option<Self> {
        let lhs = self.usize_stage_merge_parts()?;
        let rhs = other.usize_stage_merge_parts()?;
        if lhs.stage != rhs.stage || lhs.merge != rhs.merge {
            return None;
        }
        self.with_usize_stage_value(lhs.merge.combine_usize(lhs.value, rhs.value))
    }

    fn usize_stage_merge_parts(&self) -> Option<UsizeStageMergeParts> {
        match self {
            Stage::UsizeBuiltin { method, value } => Some(UsizeStageMergeParts {
                value: *value,
                stage: method.spec().view_stage?,
                merge: method.spec().stage_merge?,
            }),
            _ => None,
        }
    }

    fn with_usize_stage_value(&self, value: usize) -> Option<Self> {
        match self {
            Stage::UsizeBuiltin { method, .. } => Some(Stage::UsizeBuiltin {
                method: *method,
                value,
            }),
            _ => None,
        }
    }

    /// Returns `true` when `self` and `other` are each other's inverse and can be eliminated
    /// together during plan fusion (e.g. two consecutive `Reverse` stages).
    pub fn cancels_with(&self, other: &Self) -> bool {
        match (self.cancellation(), other.cancellation()) {
            (Some(a), Some(b)) => a.cancels_with(b),
            _ => false,
        }
    }

    fn cancellation(&self) -> Option<crate::builtins::BuiltinCancellation> {
        match self {
            Stage::Reverse(cancel) => Some(*cancel),
            Stage::Builtin(call) => call.spec().cancellation,
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct UsizeStageMergeParts {
    value: usize,
    stage: BuiltinViewStage,
    merge: crate::builtins::BuiltinStageMerge,
}

/// Top-level execution strategy chosen by `select_strategy` for a complete pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    /// All stages support indexed access and the sink selects by position; use a direct
    /// index dispatch without a pull loop.
    IndexedDispatch,
    /// At least one stage is a barrier that must materialise all upstream data before
    /// the next stage can begin.
    BarrierMaterialise,
    /// The sink needs only the first `n` inputs; break out of the pull loop once they arrive.
    EarlyExit,
    /// The default pull-loop path: iterate elements one-by-one through all stages into the sink.
    PullLoop,
}

/// Physical execution path chosen once at lower time; replaces the runtime 4-way fallthrough
/// in `exec.rs` with a static first-eligible-path dispatch.
///
/// Variants are ordered by priority: each skips all earlier paths that static analysis proved
/// cannot fire for this pipeline shape. Legacy is always the last-resort fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicalExecPath {
    /// Try indexed dispatch first; if it returns None, fall through to columnar → composed → legacy.
    Indexed,
    /// Skip indexed (it cannot apply); try columnar → composed → legacy.
    Columnar,
    /// Skip indexed and columnar (neither can apply); try composed → legacy.
    Composed,
    /// Skip all specialised paths; execute directly through the legacy interpreter.
    Legacy,
}

/// An optimised stage/sink plan produced by `plan_with_exprs`, ready for execution or further
/// wrapping into a `Pipeline`.
#[derive(Debug, Clone)]
pub struct Plan {
    /// Optimised, fused, and reordered stages.
    pub stages: Vec<Stage>,
    /// Preserved AST expressions parallel to `stages` after symbolic optimisation.
    pub stage_exprs: Vec<Option<Arc<Expr>>>,
    /// The terminal sink after demand optimisation.
    pub sink: Sink,
}

pub(super) fn stages_can_run_with_materialized_receiver(stages: &[Stage]) -> bool {
    stages
        .iter()
        .all(|stage| stage.can_run_with_receiver_only(program_is_current_only))
}

/// Return the `CompiledMatch` payload when `program`'s op stream is a
/// single `Opcode::Match` instruction (any leading `SetCurrent` /
/// `PushCurrent` is permitted because the pipeline lowering wraps lambda
/// bodies that bind the row to `@`). Returning the payload — rather than
/// just a boolean — lets callers in `composed.rs` build a dedicated
/// `MatchFilter` / `MatchMap` stage that dispatches directly into the
/// flat-IR runtime, skipping VM opcode-dispatch overhead per row.
pub(super) fn program_match_only(program: &Program) -> Option<Arc<crate::vm::CompiledMatch>> {
    let mut ops = program.ops.iter();
    while let Some(op) = ops.next() {
        match op {
            Opcode::SetCurrent | Opcode::PushCurrent => continue,
            Opcode::Match(cm) => {
                if ops.next().is_none() {
                    return Some(Arc::clone(cm));
                }
                return None;
            }
            _ => return None,
        }
    }
    None
}

/// Boolean wrapper retained for the chain-IR demand classifier, which
/// only needs to know whether a stage program *is* a match expression.
fn program_is_match_only(program: &Program) -> bool {
    program_match_only(program).is_some()
}

pub(super) fn program_is_current_only(program: &Program) -> bool {
    program.ops.iter().all(opcode_is_current_only)
}

pub(super) fn opcode_is_current_only(opcode: &Opcode) -> bool {
    match opcode {
        Opcode::PushRoot | Opcode::RootChain(_) => false,
        Opcode::PipelineRun { .. }
        | Opcode::LetExpr { .. }
        | Opcode::ListComp(_)
        | Opcode::DictComp(_)
        | Opcode::SetComp(_)
        | Opcode::PatchEval(_)
        | Opcode::UpdateBatchEval(_)
        | Opcode::Match(_)
        | Opcode::DeepMatchAll(_)
        | Opcode::DeepMatchFirst(_) => false,
        Opcode::DynIndex(prog)
        | Opcode::InlineFilter(prog)
        | Opcode::AndOp(prog)
        | Opcode::OrOp(prog)
        | Opcode::CoalesceOp(prog) => program_is_current_only(prog),
        Opcode::BindLamCurrent { body, .. } => program_is_current_only(body),
        Opcode::CallMethod(call) | Opcode::CallOptMethod(call) => call
            .sub_progs
            .iter()
            .all(|prog| program_is_current_only(prog)),
        Opcode::IfElse { then_, else_ } => {
            program_is_current_only(then_) && program_is_current_only(else_)
        }
        Opcode::TryExpr { body, default } => {
            program_is_current_only(body) && program_is_current_only(default)
        }
        Opcode::MakeArr(items) => items
            .iter()
            .all(|(prog, _spread)| program_is_current_only(prog)),
        Opcode::FString(parts) => parts.iter().all(|part| match part {
            crate::vm::CompiledFSPart::Lit(_) => true,
            crate::vm::CompiledFSPart::Interp { prog, .. } => program_is_current_only(prog),
        }),
        Opcode::MakeObj(entries) => entries.iter().all(obj_entry_is_current_only),
        Opcode::PushNull
        | Opcode::PushBool(_)
        | Opcode::PushInt(_)
        | Opcode::PushFloat(_)
        | Opcode::PushStr(_)
        | Opcode::PushCurrent
        | Opcode::LoadIdent(_)
        | Opcode::GetField(_)
        | Opcode::GetIndex(_)
        | Opcode::GetSlice(_, _, _)
        | Opcode::OptField(_)
        | Opcode::Descendant(_)
        | Opcode::DescendAll
        | Opcode::Quantifier(_)
        | Opcode::FieldChain(_)
        | Opcode::Add
        | Opcode::Sub
        | Opcode::Mul
        | Opcode::Div
        | Opcode::Mod
        | Opcode::Eq
        | Opcode::Neq
        | Opcode::Lt
        | Opcode::Lte
        | Opcode::Gt
        | Opcode::Gte
        | Opcode::Fuzzy
        | Opcode::Not
        | Opcode::Neg
        | Opcode::CastOp(_)
        | Opcode::KindCheck { .. }
        | Opcode::SetCurrent
        | Opcode::DeleteMarkErr => true,
    }
}

pub(super) fn obj_entry_is_current_only(entry: &CompiledObjEntry) -> bool {
    match entry {
        CompiledObjEntry::Short { .. } | CompiledObjEntry::KvPath { .. } => true,
        CompiledObjEntry::Kv { prog, cond, .. } => {
            program_is_current_only(prog)
                && cond
                    .as_ref()
                    .is_none_or(|cond| program_is_current_only(cond))
        }
        CompiledObjEntry::Dynamic { key, val } => {
            program_is_current_only(key) && program_is_current_only(val)
        }
        CompiledObjEntry::Spread(prog) | CompiledObjEntry::SpreadDeep(prog) => {
            program_is_current_only(prog)
        }
    }
}

impl PipelineBody {
    /// Returns `true` when all stages and the sink can execute using only the materialised
    /// receiver value, without needing a document-root (`$`) reference.
    pub(crate) fn can_run_with_materialized_receiver(&self) -> bool {
        stages_can_run_with_materialized_receiver(&self.stages)
            && self
                .sink
                .can_run_with_receiver_only(program_is_current_only)
    }

    /// Like `can_run_with_materialized_receiver` but only checks stages starting at index
    /// `consumed_stages`, used after a view-pipeline prefix has already been executed.
    pub(crate) fn suffix_can_run_with_materialized_receiver(&self, consumed_stages: usize) -> bool {
        consumed_stages <= self.stages.len()
            && stages_can_run_with_materialized_receiver(&self.stages[consumed_stages..])
            && self
                .sink
                .can_run_with_receiver_only(program_is_current_only)
    }
}

impl Pipeline {
    /// Folds the sink demand backwards through `stages`, returning the demand that must be
    /// satisfied by the source of the given stage/sink segment.
    pub fn segment_source_demand(stages: &[Stage], sink: &Sink) -> SinkDemand {
        stages
            .iter()
            .rev()
            .fold(sink.demand(), |demand, stage| stage.upstream_demand(demand))
    }

    /// Returns the `SinkDemand` that the pipeline's source must satisfy after propagating the
    /// sink demand through all stages.
    pub fn source_demand(&self) -> SinkDemand {
        self.source_demand
    }

    /// Computes precise payload demand at the source, split into fields needed while scanning
    /// and fields needed only for selected output rows.
    pub fn segment_payload_demand(
        stages: &[Stage],
        stage_kernels: &[BodyKernel],
        sink: &Sink,
        sink_kernels: &[BodyKernel],
    ) -> PayloadDemand {
        let mut lanes = sink_payload_lanes(sink, sink_kernels);
        for (idx, stage) in stages.iter().enumerate().rev() {
            let kernel = stage_kernels.get(idx).unwrap_or(&BodyKernel::Generic);
            lanes = stage_payload_lanes(stage, kernel, lanes);
        }
        lanes.into()
    }

    /// Computes precise payload demand at this pipeline's source.
    #[allow(dead_code)]
    pub fn payload_demand(&self) -> PayloadDemand {
        self.payload_demand.clone()
    }

    /// Finds a trailing run of pure one-to-one projection stages and composes it into a
    /// single late-projection annotation.
    pub fn late_projection_for(
        stages: &[Stage],
        stage_kernels: &[BodyKernel],
    ) -> Option<LateProjection> {
        let mut idx = stages.len();
        let mut kernel = BodyKernel::Current;
        let mut found = false;

        while idx > 0 {
            let stage_idx = idx - 1;
            let Some(stage_kernel) =
                trailing_projection_kernel(&stages[stage_idx], stage_kernels.get(stage_idx))
            else {
                break;
            };
            kernel = compose_projection_kernel(stage_kernel, kernel);
            found = true;
            idx -= 1;
        }

        found.then_some(LateProjection {
            prefix_len: idx,
            kernel,
        })
    }

    /// Returns true when the stored late projection can be applied after a
    /// prefix that starts at `start` without crossing a composed barrier.
    pub(crate) fn can_apply_late_projection_from(&self, start: usize) -> bool {
        let Some(projection) = self.late_projection.as_ref() else {
            return false;
        };
        projection.prefix_len >= start
            && projection.prefix_len < self.stages.len()
            && !self.stages[start..projection.prefix_len]
                .iter()
                .any(Stage::requires_legacy_fallback)
            && !self.stages[projection.prefix_len..]
                .iter()
                .any(Stage::is_composed_barrier)
    }

    /// Computes the first explicit materialization/fallback boundary for this physical path.
    pub fn fallback_boundary_for(
        stages: &[Stage],
        exec_path: PhysicalExecPath,
    ) -> FallbackBoundary {
        if let Some(index) = stages.iter().position(Stage::requires_legacy_fallback) {
            return FallbackBoundary::LegacyStage { index };
        }
        match exec_path {
            PhysicalExecPath::Legacy => FallbackBoundary::MaterializedExecution,
            PhysicalExecPath::Indexed | PhysicalExecPath::Columnar | PhysicalExecPath::Composed => {
                FallbackBoundary::None
            }
        }
    }
}

fn trailing_projection_kernel(stage: &Stage, kernel: Option<&BodyKernel>) -> Option<BodyKernel> {
    match stage {
        Stage::Map(_, _) => {
            let kernel = kernel?;
            kernel.is_view_native().then(|| kernel.clone())
        }
        Stage::Builtin(call)
            if call.spec().pure
                && call.method.is_view_projection_method()
                && call.spec().cardinality == crate::builtins::BuiltinCardinality::OneToOne =>
        {
            Some(BodyKernel::BuiltinCall {
                receiver: Box::new(BodyKernel::Current),
                call: call.clone(),
            })
        }
        _ => None,
    }
}

fn compose_projection_kernel(first: BodyKernel, then: BodyKernel) -> BodyKernel {
    if matches!(then, BodyKernel::Current) {
        return first;
    }
    BodyKernel::Compose {
        first: Box::new(first),
        then: Box::new(then),
    }
}

fn sink_payload_lanes(sink: &Sink, sink_kernels: &[BodyKernel]) -> DemandLanes {
    match sink {
        Sink::Collect | Sink::Terminal(_) | Sink::SelectMany { .. } | Sink::Nth(_) => {
            DemandLanes::RESULT
        }
        Sink::Reducer(spec) if spec.op == ReducerOp::Count => {
            let mut lanes = DemandLanes::NONE;
            if let Some(idx) = spec.predicate_kernel_index() {
                lanes.merge_scan(kernel_payload_need(sink_kernels, idx));
            }
            lanes
        }
        Sink::Reducer(spec) => {
            let mut lanes = DemandLanes::NONE;
            if let Some(idx) = spec.predicate_kernel_index() {
                lanes.merge_scan(kernel_payload_need(sink_kernels, idx));
            }
            lanes.merge_scan(match spec.projection_kernel_index() {
                Some(idx) => kernel_payload_need(sink_kernels, idx),
                None => FieldDemand::Whole,
            });
            lanes
        }
        Sink::Predicate(spec) => {
            let mut lanes = DemandLanes::NONE;
            lanes.merge_scan(kernel_payload_need(
                sink_kernels,
                spec.predicate_kernel_index(),
            ));
            if spec.op == PredicateSinkOp::FindOne {
                lanes.merge_result(FieldDemand::Whole);
            }
            lanes
        }
        Sink::Membership(_) | Sink::ApproxCountDistinct => DemandLanes {
            scan_need: FieldDemand::Whole,
            result_need: FieldDemand::None,
        },
        Sink::ArgExtreme(spec) => {
            let mut lanes = DemandLanes::RESULT;
            lanes.merge_scan(kernel_payload_need(sink_kernels, spec.key_kernel_index()));
            lanes
        }
    }
}

fn stage_payload_lanes(stage: &Stage, kernel: &BodyKernel, downstream: DemandLanes) -> DemandLanes {
    match stage {
        Stage::Filter(_, _) => {
            let mut lanes = downstream;
            lanes.merge_scan(kernel.field_demand());
            lanes
        }
        Stage::Map(_, _) | Stage::CompiledMap(_) => DemandLanes {
            scan_need: map_lane_payload(&downstream.scan_need, kernel),
            result_need: map_lane_payload(&downstream.result_need, kernel),
        },
        Stage::FlatMap(_, _) => DemandLanes {
            scan_need: FieldDemand::Whole,
            result_need: FieldDemand::Whole,
        },
        Stage::Sort(spec) => {
            let mut lanes = downstream;
            lanes.merge_scan(match spec.key {
                Some(_) => kernel.field_demand(),
                None => FieldDemand::Whole,
            });
            lanes
        }
        Stage::UniqueBy(Some(_)) | Stage::SortedDedup(Some(_)) => {
            let mut lanes = downstream;
            lanes.merge_scan(kernel.field_demand());
            lanes
        }
        Stage::UniqueBy(None) | Stage::SortedDedup(None) => {
            let mut lanes = downstream;
            lanes.merge_scan(FieldDemand::Whole);
            lanes
        }
        Stage::ExprBuiltin { method, .. }
            if matches!(
                method,
                BuiltinMethod::TakeWhile | BuiltinMethod::DropWhile | BuiltinMethod::FilterKeys
            ) =>
        {
            let mut lanes = downstream;
            lanes.merge_scan(kernel.field_demand());
            lanes
        }
        Stage::ExprBuiltin {
            method:
                BuiltinMethod::Map
                | BuiltinMethod::TransformKeys
                | BuiltinMethod::TransformValues
                | BuiltinMethod::FilterValues,
            ..
        } => DemandLanes {
            scan_need: map_lane_payload(&downstream.scan_need, kernel),
            result_need: map_lane_payload(&downstream.result_need, kernel),
        },
        Stage::ExprBuiltin { method, .. }
            if matches!(method.spec().demand_law, BuiltinDemandLaw::KeyOnlyReducer) =>
        {
            DemandLanes {
                scan_need: kernel.field_demand(),
                result_need: FieldDemand::None,
            }
        }
        Stage::ExprBuiltin { method, .. }
            if matches!(method.spec().demand_law, BuiltinDemandLaw::RowKeyedReducer) =>
        {
            DemandLanes {
                scan_need: FieldDemand::Whole,
                result_need: FieldDemand::None,
            }
        }
        Stage::Builtin(call)
            if call.method.spec().cardinality == crate::builtins::BuiltinCardinality::OneToOne =>
        {
            if downstream.scan_need.is_none() && downstream.result_need.is_none() {
                downstream
            } else {
                DemandLanes {
                    scan_need: if downstream.scan_need.is_none() {
                        FieldDemand::None
                    } else {
                        FieldDemand::Whole
                    },
                    result_need: if downstream.result_need.is_none() {
                        FieldDemand::None
                    } else {
                        FieldDemand::Whole
                    },
                }
            }
        }
        _ if stage.consumes_input_value() => DemandLanes {
            scan_need: FieldDemand::Whole,
            result_need: downstream.result_need,
        },
        _ => downstream,
    }
}

fn map_lane_payload(demand: &FieldDemand, kernel: &BodyKernel) -> FieldDemand {
    match demand {
        FieldDemand::None => FieldDemand::None,
        FieldDemand::Fields(fields) => match kernel {
            BodyKernel::FieldRead(key) => FieldDemand::Fields(fields.prefixed(&[Arc::clone(key)])),
            BodyKernel::FieldChain(keys) => FieldDemand::Fields(fields.prefixed(keys)),
            _ => kernel.field_demand(),
        },
        FieldDemand::Whole => kernel.field_demand(),
    }
}

fn kernel_payload_need(kernels: &[BodyKernel], idx: usize) -> FieldDemand {
    kernels
        .get(idx)
        .map(BodyKernel::field_demand)
        .unwrap_or(FieldDemand::Whole)
}
