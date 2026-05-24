//! Composed execution path: builds a `Stage` chain once at lower time and drives it through `run_pipeline`.
//! Avoids the per-shape dispatch in `materialized_exec`; any combination of stages and sinks executes
//! through one generic loop.
//! Returns `None` from `run` to fall through to the legacy path when lowering cannot complete.
//!
//! This module consolidates the composed-execution helpers: barrier-stage handling,
//! segment chain building, sink dispatch, and the per-stage builder.

use std::borrow::{Borrow, Cow};
use std::cell::{Cell, OnceCell, RefCell};
use std::collections::VecDeque;
use std::ops::Range;
use std::rc::Rc;
use std::sync::Arc;

use crate::builtins::{
    registry::{
        keyed_reducer as builtin_keyed_reducer, numeric_reducer as builtin_numeric_reducer,
        pipeline_stage_caps_input_prefix, pipeline_stage_is_positional, BuiltinComposedStage,
    },
    BuiltinKeyedReducer, BuiltinNumericReducer, BuiltinSelectionPosition, BuiltinSinkAccumulator,
};
use crate::data::context::{Env, EvalError};
use crate::data::value::Val;
use crate::exec::composed as cmp;
use crate::plan::demand::PullDemand;
use crate::vm::{Program, VM};

use super::ir::program_match_only;
use super::{
    compute_strategies_with_kernels, eval_kernel_view_first_with_vm, ordered_by_key_cmp,
    row_source, BodyKernel, Pipeline, Position, Sink, Source, Stage, StageStrategy,
};

// ---------------------------------------------------------------------------
// Stage builder
// ---------------------------------------------------------------------------

/// Constructs concrete `composed::Stage` objects from `Stage` IR nodes and their `BodyKernel`.
pub(super) struct ComposedStageBuilder<'a> {
    // inherited from the pipeline's outer scope
    base_env: &'a Env,
    // lazily allocated; shared by all generic program-based stages so it is created at most once
    vm_ctx: OnceCell<Rc<RefCell<cmp::VmCtx>>>,
    vm_seed: RefCell<Option<VM>>,
}

impl<'a> ComposedStageBuilder<'a> {
    /// Creates a builder that borrows `base_env` for the duration of pipeline compilation.
    pub(super) fn new(base_env: &'a Env, vm: &mut VM) -> Self {
        Self {
            base_env,
            vm_ctx: OnceCell::new(),
            vm_seed: RefCell::new(Some(std::mem::take(vm))),
        }
    }

    /// Builds a specialised `composed::Stage` for `(stage, kernel)`; returns `None` for barrier stages.
    pub(super) fn build(&self, stage: &Stage, kernel: &BodyKernel) -> Option<Box<dyn cmp::Stage>> {
        let stage_id = stage.descriptor().and_then(|desc| desc.builtin_id());
        Some(match (stage, kernel) {
            (Stage::CompiledMap(plan), _) => Box::new(NestedPlanStage {
                plan: super::nested::PreparedPlan::new(plan),
            }),
            (Stage::Filter(_, _), BodyKernel::FieldCmpLit(field, op, lit))
                if matches!(op, crate::parse::ast::BinOp::Eq) =>
            {
                Box::new(cmp::FilterFieldEqLit {
                    field: Arc::clone(field),
                    target: lit.clone(),
                })
            }
            (Stage::Map(_, _), BodyKernel::FieldRead(field)) => Box::new(cmp::MapField {
                field: Arc::clone(field),
            }),
            (Stage::Map(_, _), BodyKernel::FieldChain(keys)) => Box::new(cmp::MapFieldChain {
                keys: Arc::clone(keys),
            }),
            (Stage::FlatMap(_, _), BodyKernel::FieldRead(field)) => Box::new(cmp::FlatMapField {
                field: Arc::clone(field),
            }),
            (Stage::FlatMap(_, _), BodyKernel::FieldChain(keys)) => {
                Box::new(cmp::FlatMapFieldChain {
                    keys: Arc::clone(keys),
                })
            }
            (Stage::UsizeBuiltin { value, .. }, _)
                if stage_id.is_some_and(pipeline_stage_caps_input_prefix) =>
            {
                Box::new(cmp::Take {
                    remaining: Cell::new(*value),
                })
            }
            (Stage::UsizeBuiltin { value, .. }, _)
                if stage_id.is_some_and(pipeline_stage_is_positional) =>
            {
                Box::new(cmp::Skip {
                    remaining: Cell::new(*value),
                })
            }
            (Stage::Builtin(call), _)
                if call.composed_stage() == Some(BuiltinComposedStage::Compact) =>
            {
                Box::new(cmp::CompactFilterStage)
            }
            (Stage::Builtin(call), _)
                if call.composed_stage() == Some(BuiltinComposedStage::RemoveValue) =>
            {
                match &call.args {
                    crate::builtins::BuiltinArgs::Val(target) => {
                        Box::new(cmp::RemoveValueFilterStage::new(target.clone()))
                    }
                    _ => return None,
                }
            }
            (Stage::Builtin(call), _) => Box::new(cmp::BuiltinStage::new(call.clone())),
            // When a filter / map body is a single `match` expression we
            // dispatch directly into the flat-IR runtime, skipping VM
            // stack and opcode-dispatch overhead per row. The detector
            // accepts a leading `SetCurrent`/`PushCurrent` from lambda
            // binding so `.filter(match @ with {...})` matches.
            (Stage::Filter(p, _), _) => {
                if let Some(cm) = program_match_only(p) {
                    Box::new(cmp::MatchFilter {
                        cm,
                        ctx: self.vm_ctx(),
                    })
                } else {
                    Box::new(cmp::GenericFilter {
                        prog: Arc::clone(p),
                        ctx: self.vm_ctx(),
                    })
                }
            }
            (Stage::Map(p, _), _) => {
                if let Some(cm) = program_match_only(p) {
                    Box::new(cmp::MatchMap {
                        cm,
                        ctx: self.vm_ctx(),
                    })
                } else {
                    Box::new(cmp::GenericMap {
                        prog: Arc::clone(p),
                        ctx: self.vm_ctx(),
                    })
                }
            }
            (Stage::FlatMap(p, _), _) => Box::new(cmp::GenericFlatMap {
                prog: Arc::clone(p),
                ctx: self.vm_ctx(),
            }),
            _ => return None,
        })
    }

    /// Builds a filter stage from `prog`, specialising on field-equality kernels where possible.
    pub(super) fn build_filter_program(
        &self,
        prog: &Arc<Program>,
        kernel: &BodyKernel,
    ) -> Box<dyn cmp::Stage> {
        match kernel {
            BodyKernel::FieldCmpLit(field, op, lit)
                if matches!(op, crate::parse::ast::BinOp::Eq) =>
            {
                Box::new(cmp::FilterFieldEqLit {
                    field: Arc::clone(field),
                    target: lit.clone(),
                })
            }
            _ => Box::new(cmp::GenericFilter {
                prog: Arc::clone(prog),
                ctx: self.vm_ctx(),
            }),
        }
    }

    /// Builds a map stage from `prog`, specialising on single-field and chain-read kernels.
    pub(super) fn build_map_program(
        &self,
        prog: &Arc<Program>,
        kernel: &BodyKernel,
    ) -> Box<dyn cmp::Stage> {
        match kernel {
            BodyKernel::FieldRead(field) => Box::new(cmp::MapField {
                field: Arc::clone(field),
            }),
            BodyKernel::FieldChain(keys) => Box::new(cmp::MapFieldChain {
                keys: Arc::clone(keys),
            }),
            _ => Box::new(cmp::GenericMap {
                prog: Arc::clone(prog),
                ctx: self.vm_ctx(),
            }),
        }
    }

    // initialises the shared VmCtx on first call
    fn vm_ctx(&self) -> Rc<RefCell<cmp::VmCtx>> {
        Rc::clone(self.vm_ctx.get_or_init(|| {
            Rc::new(RefCell::new(cmp::VmCtx {
                vm: self.vm_seed.borrow_mut().take().unwrap_or_default(),
                env: self.base_env.clone(),
            }))
        }))
    }

    fn restore_vm(&self, vm: &mut VM) {
        if let Some(ctx) = self.vm_ctx.get() {
            *vm = std::mem::take(&mut ctx.borrow_mut().vm);
        } else if let Some(seed) = self.vm_seed.borrow_mut().take() {
            *vm = seed;
        }
    }

    fn with_vm<T>(&self, f: impl FnOnce(&mut VM) -> T) -> T {
        if let Some(ctx) = self.vm_ctx.get() {
            return f(&mut ctx.borrow_mut().vm);
        }
        f(self
            .vm_seed
            .borrow_mut()
            .as_mut()
            .expect("composed VM seed must be present before restore"))
    }
}

struct NestedPlanStage {
    plan: super::nested::PreparedPlan,
}

impl cmp::Stage for NestedPlanStage {
    fn apply<'a>(&self, x: &'a Val) -> cmp::StageOutput<'a> {
        match self.plan.run(x.clone()) {
            Ok(value) => cmp::StageOutput::Pass(Cow::Owned(value)),
            Err(_) => cmp::StageOutput::Filtered,
        }
    }
}

/// Extracts a `KeySource` from `kernel`; returns `None` for generic kernels.
pub(super) fn key_from_kernel(kernel: &BodyKernel) -> Option<cmp::KeySource> {
    let keys = kernel.field_path_keys()?;
    match keys.as_slice() {
        [] => Some(cmp::KeySource::None),
        [field] => Some(cmp::KeySource::Field(Arc::clone(field))),
        _ => Some(cmp::KeySource::Chain(keys.into())),
    }
}

// ---------------------------------------------------------------------------
// Segment chain construction
// ---------------------------------------------------------------------------

/// Builds a composed stage chain for the `stages[range]` slice using `kernels` for specialisation.
///
/// Returns `None` if any stage in the range cannot be lowered to a composed equivalent.
fn build_chain(
    stages: &[Stage],
    kernels: &[BodyKernel],
    range: Range<usize>,
    builder: &ComposedStageBuilder<'_>,
) -> Option<Box<dyn cmp::Stage>> {
    let mut chain: Box<dyn cmp::Stage> = Box::new(cmp::Identity);
    for idx in range {
        let stage = &stages[idx];
        let kernel = kernels.get(idx).unwrap_or(&BodyKernel::Generic);
        let next = builder.build(stage, kernel)?;
        chain = Box::new(cmp::Composed { a: chain, b: next });
    }
    Some(chain)
}

/// Runs `chain` over `rows` with a `CollectSink` and unwraps the resulting `Val::Arr`.
///
/// Returns `None` if the pipeline result is not an array (should not happen in normal use).
fn segment_collect(rows: &[Val], chain: &dyn cmp::Stage) -> Option<Vec<Val>> {
    match cmp::run_pipeline::<cmp::CollectSink>(rows, chain) {
        Val::Arr(items) => Some(items.as_ref().clone()),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Barrier-stage execution
// ---------------------------------------------------------------------------

/// Result of a barrier stage: either a new row list for downstream stages, or a finished value.
enum BarrierOutput {
    /// Transformed row set to continue through the pipeline.
    Rows(Vec<Val>),
    /// Final result produced by a consuming barrier (e.g. `group_by`).
    Done(Val),
}

/// Executes a barrier stage over `buf`; returns `None` for unrecognised barriers or missing keys.
fn run_barrier(
    stage: &Stage,
    kernel: &BodyKernel,
    strategy: StageStrategy,
    sink: &Sink,
    is_terminal: bool,
    buf: Vec<Val>,
) -> Option<BarrierOutput> {
    let rows = match stage {
        Stage::Reverse(_) => cmp::barrier_reverse(buf),
        Stage::Sort(spec) => {
            let key = match &spec.key {
                None => cmp::KeySource::None,
                Some(_) => key_from_kernel(kernel)?,
            };
            let mut out = match (strategy, spec.descending) {
                (StageStrategy::SortTopK(k), false) | (StageStrategy::SortBottomK(k), true) => {
                    cmp::barrier_top_k(buf, &key, k)
                }
                (StageStrategy::SortTopK(k), true) | (StageStrategy::SortBottomK(k), false) => {
                    cmp::barrier_bottom_k(buf, &key, k)
                }
                (_, false) | (_, true) => cmp::barrier_sort(buf, &key),
            };
            if spec.descending {
                out.reverse();
            }
            out
        }
        Stage::UniqueBy(None) => cmp::barrier_unique_by(buf, &cmp::KeySource::None),
        Stage::UniqueBy(Some(_)) => {
            let key = key_from_kernel(kernel)?;
            cmp::barrier_unique_by(buf, &key)
        }
        Stage::SortedDedup(None) => {
            let sorted = cmp::barrier_sort(buf, &cmp::KeySource::None);
            cmp::barrier_unique_by(sorted, &cmp::KeySource::None)
        }
        Stage::SortedDedup(Some(_)) => {
            let key = key_from_kernel(kernel)?;
            let sorted = cmp::barrier_sort(buf, &key);
            cmp::barrier_unique_by(sorted, &key)
        }
        Stage::ExprBuiltin { .. } if stage
            .descriptor()
            .and_then(|desc| desc.builtin_id())
            .is_some_and(|id| builtin_keyed_reducer(id).is_some()) =>
        {
            let id = stage.descriptor()?.builtin_id()?;
            let key = key_from_kernel(kernel)?;
            let value = match builtin_keyed_reducer(id)? {
                BuiltinKeyedReducer::Group => cmp::barrier_group_by(buf, &key),
                BuiltinKeyedReducer::Count => cmp::barrier_count_by(buf, &key),
                BuiltinKeyedReducer::Index => cmp::barrier_index_by(buf, &key),
            };
            if matches!(sink, Sink::Collect) && is_terminal {
                return Some(BarrierOutput::Done(value));
            }
            return Some(BarrierOutput::Rows(vec![value]));
        }
        _ => return None,
    };

    Some(BarrierOutput::Rows(rows))
}

// ---------------------------------------------------------------------------
// Sink dispatch
// ---------------------------------------------------------------------------

// dispatches a borrowed-slice run; expands to a monomorphised cmp::$runner call
macro_rules! run_composed_sink {
    ($runner:ident, $rows:expr, $chain:expr, $demand:expr, $sink:expr) => {
        match $sink.builtin_sink_spec()?.accumulator {
            BuiltinSinkAccumulator::Count => cmp::$runner::<cmp::CountSink>($rows, $chain, $demand),
            BuiltinSinkAccumulator::Numeric => match numeric_reducer($sink)? {
                BuiltinNumericReducer::Sum => cmp::$runner::<cmp::SumSink>($rows, $chain, $demand),
                BuiltinNumericReducer::Min => cmp::$runner::<cmp::MinSink>($rows, $chain, $demand),
                BuiltinNumericReducer::Max => cmp::$runner::<cmp::MaxSink>($rows, $chain, $demand),
                BuiltinNumericReducer::Avg => cmp::$runner::<cmp::AvgSink>($rows, $chain, $demand),
            },
            BuiltinSinkAccumulator::SelectOne(BuiltinSelectionPosition::First) => {
                cmp::$runner::<cmp::FirstSink>($rows, $chain, $demand)
            }
            BuiltinSinkAccumulator::SelectOne(BuiltinSelectionPosition::Last) => {
                cmp::$runner::<cmp::LastSink>($rows, $chain, $demand)
            }
            BuiltinSinkAccumulator::ApproxDistinct => return None,
        }
    };
}

// like run_composed_sink! but accepts any IntoIterator<Item = Val> as the row source
macro_rules! run_composed_owned_sink {
    ($runner:ident, $rows:expr, $chain:expr, $demand:expr, $sink:expr) => {
        match $sink.builtin_sink_spec()?.accumulator {
            BuiltinSinkAccumulator::Count => {
                cmp::$runner::<cmp::CountSink, _>($rows, $chain, $demand)
            }
            BuiltinSinkAccumulator::Numeric => match numeric_reducer($sink)? {
                BuiltinNumericReducer::Sum => {
                    cmp::$runner::<cmp::SumSink, _>($rows, $chain, $demand)
                }
                BuiltinNumericReducer::Min => {
                    cmp::$runner::<cmp::MinSink, _>($rows, $chain, $demand)
                }
                BuiltinNumericReducer::Max => {
                    cmp::$runner::<cmp::MaxSink, _>($rows, $chain, $demand)
                }
                BuiltinNumericReducer::Avg => {
                    cmp::$runner::<cmp::AvgSink, _>($rows, $chain, $demand)
                }
            },
            BuiltinSinkAccumulator::SelectOne(BuiltinSelectionPosition::First) => {
                cmp::$runner::<cmp::FirstSink, _>($rows, $chain, $demand)
            }
            BuiltinSinkAccumulator::SelectOne(BuiltinSelectionPosition::Last) => {
                cmp::$runner::<cmp::LastSink, _>($rows, $chain, $demand)
            }
            BuiltinSinkAccumulator::ApproxDistinct => return None,
        }
    };
}

/// Runs `chain` over `rows`, collecting into the sink; returns `None` for `ApproxCountDistinct`.
fn run_sink(sink: &Sink, rows: &[Val], chain: &dyn cmp::Stage, demand: PullDemand) -> Option<Val> {
    let out = match sink {
        Sink::Collect => cmp::run_pipeline_with_demand::<cmp::CollectSink>(rows, chain, demand),
        Sink::Nth(idx) => cmp::run_pipeline_nth_with_demand(rows, chain, demand, *idx),
        Sink::Reducer(_) | Sink::Terminal(_) => {
            run_composed_sink!(run_pipeline_with_demand, rows, chain, demand, sink)
        }
        Sink::Predicate(_)
        | Sink::Membership(_)
        | Sink::ArgExtreme(_)
        | Sink::SelectMany { .. } => return None,
        Sink::ApproxCountDistinct => return None,
    };

    Some(out)
}

/// Runs `chain` over an owned iterator `rows`, collecting into the sink.
fn run_sink_owned_iter<I>(
    sink: &Sink,
    rows: I,
    chain: &dyn cmp::Stage,
    demand: PullDemand,
) -> Option<Val>
where
    I: IntoIterator<Item = Val>,
{
    let out = match sink {
        Sink::Collect => {
            cmp::run_pipeline_owned_iter_with_demand::<cmp::CollectSink, _>(rows, chain, demand)
        }
        Sink::Nth(idx) => cmp::run_pipeline_owned_iter_nth_with_demand(rows, chain, demand, *idx),
        Sink::Reducer(_) | Sink::Terminal(_) => run_composed_owned_sink!(
            run_pipeline_owned_iter_with_demand,
            rows,
            chain,
            demand,
            sink
        ),
        Sink::Predicate(_)
        | Sink::Membership(_)
        | Sink::ArgExtreme(_)
        | Sink::SelectMany { .. } => return None,
        Sink::ApproxCountDistinct => return None,
    };

    Some(out)
}

fn numeric_reducer(sink: &Sink) -> Option<BuiltinNumericReducer> {
    builtin_numeric_reducer(sink.builtin_id()?)
}

// ---------------------------------------------------------------------------
// Source resolution
// ---------------------------------------------------------------------------

/// Resolves `source` against `root` and returns a `Rows` iterator for composed execution.
///
/// Returns `None` when the resolved value is not array-like (scalar or null source).
fn source_rows(source: &Source, root: &Val) -> Option<row_source::Rows<'static>> {
    let recv = row_source::resolve(source, root);
    row_source::resolved_array_like_rows(recv)
}

// ---------------------------------------------------------------------------
// Top-level driver
// ---------------------------------------------------------------------------

/// Entry point for composed execution using caller-owned VM cache/state.
pub(super) fn run_with_vm(
    pipeline: &Pipeline,
    root: &Val,
    base_env: &Env,
    vm: &mut VM,
) -> Option<Result<Val, EvalError>> {
    let result = run_inner(pipeline, root, base_env, vm);
    result
}

fn run_inner(
    pipeline: &Pipeline,
    root: &Val,
    base_env: &Env,
    vm: &mut VM,
) -> Option<Result<Val, EvalError>> {
    let (eff_stages, eff_kernels, eff_sink) = pipeline.canonical();
    let stage_builder = ComposedStageBuilder::new(base_env, vm);
    let result = run_with_builder(
        pipeline,
        root,
        &eff_stages,
        &eff_kernels,
        &eff_sink,
        &stage_builder,
    );
    stage_builder.restore_vm(vm);
    result
}

fn run_with_builder(
    pipeline: &Pipeline,
    root: &Val,
    eff_stages: &[Stage],
    eff_kernels: &[BodyKernel],
    eff_sink: &Sink,
    stage_builder: &ComposedStageBuilder<'_>,
) -> Option<Result<Val, EvalError>> {
    let mut buf = source_rows(&pipeline.source, root)?;

    let kernels = eff_kernels;
    let stages_ref = eff_stages;

    let strategies = compute_strategies_with_kernels(stages_ref, kernels, &eff_sink);

    let mut last_split = 0usize;
    for (i, stage) in stages_ref.iter().enumerate() {
        if !stage.is_composed_barrier() {
            continue;
        }

        if i > last_split {
            let chain = build_chain(stages_ref, kernels, last_split..i, &stage_builder)?;
            buf = super::row_source::Rows::Owned(segment_collect(buf.as_slice(), chain.as_ref())?);
        }

        let kernel = kernels.get(i).unwrap_or(&BodyKernel::Generic);
        let strategy = strategies.get(i).copied().unwrap_or(StageStrategy::Default);
        if let StageStrategy::SortUntilOutput(target_outputs) = strategy {
            let _ = target_outputs;
            if let Some(out) = run_lazy_ordered_suffix(
                pipeline,
                stage,
                kernel,
                &eff_sink,
                &pipeline.sink_kernels,
                stages_ref,
                kernels,
                i,
                &stage_builder,
                buf.into_vec(),
            ) {
                return Some(out);
            }
            return None;
        }
        match run_barrier(
            stage,
            kernel,
            strategy,
            &eff_sink,
            i + 1 == stages_ref.len(),
            buf.into_vec(),
        )? {
            BarrierOutput::Rows(rows) => buf = super::row_source::Rows::Owned(rows),
            BarrierOutput::Done(val) => return Some(Ok(val)),
        };

        last_split = i + 1;
    }

    let chain = build_chain(
        stages_ref,
        kernels,
        last_split..stages_ref.len(),
        &stage_builder,
    )?;
    let final_demand = Pipeline::segment_source_demand(&stages_ref[last_split..], &eff_sink)
        .chain
        .pull;
    if let Some(out) = run_late_projection_sink(
        pipeline,
        &eff_sink,
        &stage_builder,
        stages_ref,
        kernels,
        last_split,
        buf.as_slice(),
    ) {
        return Some(out);
    }

    let (sink, chain) =
        append_reducer_sink_stages(&eff_sink, &pipeline.sink_kernels, &stage_builder, chain)?;
    let out = run_sink(&sink, buf.as_slice(), chain.as_ref(), final_demand)?;

    Some(Ok(out))
}

/// Runs a terminal sink through only the non-projection prefix, applying the delayed projection
/// inside the sink so rows rejected or skipped by the prefix never pay projection cost.
fn run_late_projection_sink(
    pipeline: &Pipeline,
    sink: &Sink,
    stage_builder: &ComposedStageBuilder<'_>,
    stages: &[Stage],
    kernels: &[BodyKernel],
    start: usize,
    rows: &[Val],
) -> Option<Result<Val, EvalError>> {
    let projection = pipeline.late_projection.as_ref()?;
    if !pipeline.can_apply_late_projection_from(start) {
        return None;
    }

    let prefix = build_chain(stages, kernels, start..projection.prefix_len, stage_builder)?;
    let demand = Pipeline::segment_source_demand(&stages[start..projection.prefix_len], sink)
        .chain
        .pull;

    let projecting_sink = projecting_sink_for(sink, demand)?;
    run_projecting_iter(
        demand_rows(rows, demand),
        prefix.as_ref(),
        demand,
        &projection.kernel,
        projecting_sink,
        stage_builder,
    )
}

fn projecting_sink_for(sink: &Sink, demand: PullDemand) -> Option<ProjectingSink> {
    if !sink.supports_late_projection(demand) {
        return None;
    }
    match sink {
        Sink::Collect => Some(ProjectingSink::Collect(Vec::new())),
        Sink::Terminal(_) => match sink.select_one_position()? {
            Position::First => Some(ProjectingSink::First(None)),
            Position::Last => Some(ProjectingSink::Last(None)),
        },
        Sink::Nth(idx) => {
            let target = if demand.is_nth_input() { 0 } else { *idx };
            Some(ProjectingSink::Nth {
                target,
                seen: 0,
                value: None,
            })
        }
        Sink::SelectMany { n, from_end } => Some(ProjectingSink::SelectMany {
            n: *n,
            from_end: *from_end,
            prepend: *from_end && demand.is_suffix(),
            items: VecDeque::new(),
        }),
        _ => None,
    }
}

enum DemandRows<'a> {
    Forward(std::slice::Iter<'a, Val>),
    Reverse(std::iter::Rev<std::slice::Iter<'a, Val>>),
    One(std::option::IntoIter<&'a Val>),
}

impl<'a> Iterator for DemandRows<'a> {
    type Item = &'a Val;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            DemandRows::Forward(iter) => iter.next(),
            DemandRows::Reverse(iter) => iter.next(),
            DemandRows::One(iter) => iter.next(),
        }
    }
}

fn demand_rows(rows: &[Val], demand: PullDemand) -> DemandRows<'_> {
    match demand {
        PullDemand::LastInput(_) => DemandRows::Reverse(rows.iter().rev()),
        PullDemand::NthInput(i) => DemandRows::One(rows.get(i).into_iter()),
        _ => DemandRows::Forward(rows.iter()),
    }
}

enum ProjectingSink {
    Collect(Vec<Val>),
    First(Option<Val>),
    Last(Option<Val>),
    Nth {
        target: usize,
        seen: usize,
        value: Option<Val>,
    },
    SelectMany {
        n: usize,
        from_end: bool,
        prepend: bool,
        items: VecDeque<Val>,
    },
}

impl ProjectingSink {
    fn observe_with_vm(
        &mut self,
        item: &Val,
        projection: &BodyKernel,
        stage_builder: &ComposedStageBuilder<'_>,
    ) -> Result<bool, EvalError> {
        stage_builder.with_vm(|vm| self.observe(item, projection, vm))
    }

    fn observe(
        &mut self,
        item: &Val,
        projection: &BodyKernel,
        vm: &mut VM,
    ) -> Result<bool, EvalError> {
        match self {
            ProjectingSink::Collect(items) => {
                items.push(eval_late_projection(projection, item, vm)?);
                Ok(false)
            }
            ProjectingSink::First(slot) => {
                if slot.is_none() {
                    *slot = Some(eval_late_projection(projection, item, vm)?);
                }
                Ok(true)
            }
            ProjectingSink::Last(slot) => {
                *slot = Some(eval_late_projection(projection, item, vm)?);
                Ok(false)
            }
            ProjectingSink::Nth {
                target,
                seen,
                value,
            } => {
                if *seen == *target {
                    *value = Some(eval_late_projection(projection, item, vm)?);
                    return Ok(true);
                }
                *seen += 1;
                Ok(false)
            }
            ProjectingSink::SelectMany {
                n,
                from_end,
                prepend,
                items,
            } => {
                if *n == 0 {
                    return Ok(true);
                }
                let item = eval_late_projection(projection, item, vm)?;
                if *prepend {
                    if items.len() == *n {
                        items.pop_back();
                    }
                    items.push_front(item);
                    return Ok(items.len() >= *n);
                }
                if *from_end {
                    if items.len() == *n {
                        items.pop_front();
                    }
                    items.push_back(item);
                    Ok(false)
                } else {
                    items.push_back(item);
                    Ok(items.len() >= *n)
                }
            }
        }
    }

    fn done(&self) -> bool {
        matches!(
            self,
            ProjectingSink::First(Some(_)) | ProjectingSink::Nth { value: Some(_), .. }
        )
    }

    fn finish(self) -> Val {
        match self {
            ProjectingSink::Collect(items) => Val::Arr(Arc::new(items)),
            ProjectingSink::First(value) | ProjectingSink::Last(value) => {
                value.unwrap_or(Val::Null)
            }
            ProjectingSink::Nth { value, .. } => value.unwrap_or(Val::Null),
            ProjectingSink::SelectMany { n: 0, .. } => Val::Null,
            ProjectingSink::SelectMany { n, items, .. } if n == 1 => {
                items.into_iter().next().unwrap_or(Val::Null)
            }
            ProjectingSink::SelectMany { items, .. } => Val::arr(items.into_iter().collect()),
        }
    }
}

fn eval_late_projection(
    projection: &BodyKernel,
    item: &Val,
    vm: &mut VM,
) -> Result<Val, EvalError> {
    eval_kernel_view_first_with_vm(projection, item, vm, |_, _| {
        Err(EvalError(
            "late projection requires a native body kernel".to_string(),
        ))
    })
}

fn run_projecting_iter<I>(
    rows: I,
    stages: &dyn cmp::Stage,
    demand: PullDemand,
    projection: &BodyKernel,
    mut sink: ProjectingSink,
    stage_builder: &ComposedStageBuilder<'_>,
) -> Option<Result<Val, EvalError>>
where
    I: IntoIterator,
    I::Item: Borrow<Val>,
{
    let mut pulled_inputs = 0usize;
    let mut emitted_outputs = 0usize;
    for row in rows {
        if demand.input_satisfied_by(pulled_inputs) {
            break;
        }
        pulled_inputs += 1;

        match stages.apply(row.borrow()) {
            cmp::StageOutput::Pass(cow) => {
                let done = match sink.observe_with_vm(cow.as_ref(), projection, stage_builder) {
                    Ok(done) => done,
                    Err(err) => return Some(Err(err)),
                };
                emitted_outputs += 1;
                if done || sink.done() {
                    break;
                }
                if demand.output_satisfied_by(emitted_outputs) {
                    break;
                }
            }
            cmp::StageOutput::Filtered => continue,
            cmp::StageOutput::Many(items) => {
                for item in items {
                    let done = match sink.observe_with_vm(item.as_ref(), projection, stage_builder)
                    {
                        Ok(done) => done,
                        Err(err) => return Some(Err(err)),
                    };
                    emitted_outputs += 1;
                    if done || sink.done() {
                        break;
                    }
                    if demand.output_satisfied_by(emitted_outputs) {
                        break;
                    }
                }
                if sink.done() || demand.output_satisfied_by(emitted_outputs) {
                    break;
                }
            }
            cmp::StageOutput::Done => break,
        }
    }

    Some(Ok(sink.finish()))
}

/// Sorts `rows` by key and feeds the ordered iterator into the composed sink for top-N short-circuit.
fn run_lazy_ordered_suffix(
    pipeline: &Pipeline,
    stage: &Stage,
    kernel: &BodyKernel,
    sink: &Sink,
    sink_kernels: &[BodyKernel],
    stages: &[Stage],
    kernels: &[BodyKernel],
    sort_idx: usize,
    stage_builder: &ComposedStageBuilder<'_>,
    rows: Vec<Val>,
) -> Option<Result<Val, EvalError>> {
    let Stage::Sort(spec) = stage else {
        return None;
    };
    if stages[sort_idx + 1..]
        .iter()
        .any(Stage::is_composed_barrier)
    {
        return None;
    }

    let key = match &spec.key {
        None => cmp::KeySource::None,
        Some(_) => key_from_kernel(kernel)?,
    };
    let final_demand = Pipeline::segment_source_demand(&stages[sort_idx + 1..], sink)
        .chain
        .pull;
    let ordered_descending = if final_demand.is_suffix() {
        !spec.descending
    } else {
        spec.descending
    };
    let ordered = match ordered_by_key_cmp(
        rows,
        ordered_descending,
        |v| Ok(key.extract(v)),
        cmp::cmp_val,
    ) {
        Ok(ordered) => ordered,
        Err(err) => return Some(Err(err)),
    };
    let suffix_start = sort_idx + 1;
    if let Some(projection) = pipeline.late_projection.as_ref() {
        if pipeline.can_apply_late_projection_from(suffix_start) {
            if let Some(projecting_sink) = projecting_sink_for(sink, final_demand) {
                if let Some(prefix) = build_chain(
                    stages,
                    kernels,
                    suffix_start..projection.prefix_len,
                    stage_builder,
                ) {
                    return run_projecting_iter(
                        ordered,
                        prefix.as_ref(),
                        final_demand,
                        &projection.kernel,
                        projecting_sink,
                        stage_builder,
                    );
                }
            }
        }
    }

    let chain = build_chain(stages, kernels, sort_idx + 1..stages.len(), stage_builder)?;
    let (sink, chain) = append_reducer_sink_stages(sink, sink_kernels, stage_builder, chain)?;
    run_sink_owned_iter(&sink, ordered, chain.as_ref(), final_demand).map(Ok)
}

/// Promotes reducer predicate and projection into composed stages appended to `chain`, stripping them from the returned sink.
fn append_reducer_sink_stages(
    sink: &Sink,
    sink_kernels: &[BodyKernel],
    stage_builder: &ComposedStageBuilder<'_>,
    mut chain: Box<dyn cmp::Stage>,
) -> Option<(Sink, Box<dyn cmp::Stage>)> {
    let Sink::Reducer(spec) = sink else {
        return Some((sink.clone(), chain));
    };

    let mut sink = sink.clone();
    let Sink::Reducer(out_spec) = &mut sink else {
        unreachable!("cloned reducer sink changed variant");
    };

    if let Some(predicate) = &spec.predicate {
        let idx = spec.predicate_kernel_index()?;
        let kernel = sink_kernels.get(idx).unwrap_or(&BodyKernel::Generic);
        let stage = stage_builder.build_filter_program(predicate, kernel);
        chain = Box::new(cmp::Composed { a: chain, b: stage });
        out_spec.predicate = None;
    }

    if let Some(projection) = &spec.projection {
        let idx = spec.projection_kernel_index()?;
        let kernel = sink_kernels.get(idx).unwrap_or(&BodyKernel::Generic);
        let stage = stage_builder.build_map_program(projection, kernel);
        chain = Box::new(cmp::Composed { a: chain, b: stage });
        out_spec.projection = None;
    }

    Some((sink, chain))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use indexmap::IndexMap;

    use crate::data::value::Val;
    use crate::exec::pipeline::BodyKernel;
    use crate::parse::parser::parse;

    use super::key_from_kernel;

    fn obj(pairs: impl IntoIterator<Item = (&'static str, Val)>) -> Val {
        let mut out = IndexMap::new();
        for (key, value) in pairs {
            out.insert(Arc::from(key), value);
        }
        Val::Obj(Arc::new(out))
    }

    #[test]
    fn composed_key_source_uses_shared_field_path_metadata() {
        let kernel = BodyKernel::classify_expr(
            &parse(r#"profile.get_path("author.name")"#).expect("parse get_path key"),
        );
        let key = key_from_kernel(&kernel).expect("key source");
        let row = obj([(
            "profile",
            obj([("author", obj([("name", Val::Str(Arc::from("Ada")))]))]),
        )]);

        assert_eq!(key.extract(&row), Val::Str(Arc::from("Ada")));
        assert_eq!(key.extract(&obj([])), Val::Null);
    }
}
