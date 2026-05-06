//! Composed execution path: builds a `Stage` chain once at lower time and drives it through `run_pipeline`.
//! Avoids the per-shape dispatch in `materialized_exec`; any combination of stages and sinks executes
//! through one generic loop.
//! Returns `None` from `run` to fall through to the legacy path when lowering cannot complete.
//!
//! This module consolidates the composed-execution helpers: barrier-stage handling,
//! segment chain building, sink dispatch, and the per-stage builder.

use std::cell::{Cell, OnceCell, RefCell};
use std::ops::Range;
use std::rc::Rc;
use std::sync::Arc;

use crate::builtins::{BuiltinNumericReducer, BuiltinSelectionPosition, BuiltinSinkAccumulator};
use crate::parse::chain_ir::PullDemand;
use crate::exec::composed as cmp;
use crate::data::context::{Env, EvalError};
use crate::data::value::Val;
use crate::vm::Program;

use super::{
    compute_strategies_with_kernels, ordered_by_key_cmp, row_source, BodyKernel, Pipeline, Sink,
    Source, Stage, StageStrategy,
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
}

impl<'a> ComposedStageBuilder<'a> {
    /// Creates a builder that borrows `base_env` for the duration of pipeline compilation.
    pub(super) fn new(base_env: &'a Env) -> Self {
        Self {
            base_env,
            vm_ctx: OnceCell::new(),
        }
    }

    /// Builds a specialised `composed::Stage` for `(stage, kernel)`; returns `None` for barrier stages.
    pub(super) fn build(&self, stage: &Stage, kernel: &BodyKernel) -> Option<Box<dyn cmp::Stage>> {
        Some(match (stage, kernel) {
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
            (
                Stage::UsizeBuiltin {
                    method: crate::builtins::BuiltinMethod::Take,
                    value,
                },
                _,
            ) => Box::new(cmp::Take {
                remaining: Cell::new(*value),
            }),
            (
                Stage::UsizeBuiltin {
                    method: crate::builtins::BuiltinMethod::Skip,
                    value,
                },
                _,
            ) => Box::new(cmp::Skip {
                remaining: Cell::new(*value),
            }),
            (Stage::Builtin(call), _) => Box::new(cmp::BuiltinStage::new(call.clone())),
            (Stage::Filter(p, _), _) => Box::new(cmp::GenericFilter {
                prog: Arc::clone(p),
                ctx: self.vm_ctx(),
            }),
            (Stage::Map(p, _), _) => Box::new(cmp::GenericMap {
                prog: Arc::clone(p),
                ctx: self.vm_ctx(),
            }),
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
            BodyKernel::FieldCmpLit(field, op, lit) if matches!(op, crate::parse::ast::BinOp::Eq) => {
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
                vm: crate::vm::VM::new(),
                env: self.base_env.clone(),
            }))
        }))
    }
}

/// Extracts a `KeySource` from `kernel`; returns `None` for generic kernels.
pub(super) fn key_from_kernel(kernel: &BodyKernel) -> Option<cmp::KeySource> {
    match kernel {
        BodyKernel::FieldRead(field) => Some(cmp::KeySource::Field(Arc::clone(field))),
        BodyKernel::FieldChain(keys) => Some(cmp::KeySource::Chain(Arc::clone(keys))),
        _ => None,
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
        Stage::ExprBuiltin {
            method: crate::builtins::BuiltinMethod::GroupBy,
            ..
        } => {
            if !matches!(sink, Sink::Collect) || !is_terminal {
                return None;
            }
            let key = key_from_kernel(kernel)?;
            return Some(BarrierOutput::Done(cmp::barrier_group_by(buf, &key)));
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
fn run_sink(
    sink: &Sink,
    rows: &[Val],
    chain: &dyn cmp::Stage,
    demand: PullDemand,
) -> Option<Val> {
    let out = match sink {
        Sink::Collect => cmp::run_pipeline_with_demand::<cmp::CollectSink>(rows, chain, demand),
        Sink::Nth(idx) => cmp::run_pipeline_nth_with_demand(rows, chain, demand, *idx),
        Sink::Reducer(_) | Sink::Terminal(_) => {
            run_composed_sink!(run_pipeline_with_demand, rows, chain, demand, sink)
        }
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
        Sink::ApproxCountDistinct => return None,
    };

    Some(out)
}

fn numeric_reducer(sink: &Sink) -> Option<BuiltinNumericReducer> {
    sink.reducer_spec()?.method()?.spec().numeric_reducer
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

/// Entry point for composed execution; returns `None` when any stage or sink cannot be lowered.
pub(super) fn run(
    pipeline: &Pipeline,
    root: &Val,
    base_env: &Env,
) -> Option<Result<Val, EvalError>> {
    let (eff_stages, eff_kernels, eff_sink) = pipeline.canonical();
    let stage_builder = ComposedStageBuilder::new(base_env);

    let mut buf = source_rows(&pipeline.source, root)?;

    let kernels = &eff_kernels;
    let stages_ref = &eff_stages;

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

    let chain = build_chain(stages_ref, kernels, last_split..stages_ref.len(), &stage_builder)?;
    let final_demand = Pipeline::segment_source_demand(&stages_ref[last_split..], &eff_sink)
        .chain
        .pull;
    let (sink, chain) =
        append_reducer_sink_stages(&eff_sink, &pipeline.sink_kernels, &stage_builder, chain)?;
    let out = run_sink(&sink, buf.as_slice(), chain.as_ref(), final_demand)?;

    Some(Ok(out))
}

/// Sorts `rows` by key and feeds the ordered iterator into the composed sink for top-N short-circuit.
fn run_lazy_ordered_suffix(
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
    let ordered_descending = if matches!(final_demand, PullDemand::LastInput(_)) {
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
