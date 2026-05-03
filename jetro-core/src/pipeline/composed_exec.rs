//! Composed execution path: builds a `Stage` chain once at lower time and drives it through `run_pipeline`.
//! Avoids the per-shape dispatch in `legacy_exec`; any combination of stages and sinks executes
//! through one generic loop.
//! Returns `None` from `run` to fall through to the legacy path when lowering cannot complete.

use crate::{
    context::{Env, EvalError},
    value::Val,
};

use super::composed_barrier::{self, BarrierOutput};
use super::composed_segment;
use super::composed_sink;
use super::composed_source;
use super::composed_stage::ComposedStageBuilder;
use super::{
    compute_strategies_with_kernels, ordered_by_key_cmp, BodyKernel, Pipeline, Stage, StageStrategy,
};

/// Entry point for composed execution; returns `None` when any stage or sink cannot be lowered.
pub(super) fn run(
    pipeline: &Pipeline,
    root: &Val,
    base_env: &Env,
) -> Option<Result<Val, EvalError>> {
    let (eff_stages, eff_kernels, eff_sink) = pipeline.canonical();
    let stage_builder = ComposedStageBuilder::new(base_env);

    let mut buf = composed_source::rows(&pipeline.source, root)?;

    
    let kernels = &eff_kernels;
    let stages_ref = &eff_stages;

    let strategies = compute_strategies_with_kernels(stages_ref, kernels, &eff_sink);

    let mut last_split = 0usize;
    for (i, stage) in stages_ref.iter().enumerate() {
        if !stage.is_composed_barrier() {
            continue;
        }

        if i > last_split {
            let chain =
                composed_segment::build_chain(stages_ref, kernels, last_split..i, &stage_builder)?;
            buf = super::row_source::Rows::Owned(composed_segment::collect(
                buf.as_slice(),
                chain.as_ref(),
            )?);
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
        match composed_barrier::run(
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

    let chain = composed_segment::build_chain(
        stages_ref,
        kernels,
        last_split..stages_ref.len(),
        &stage_builder,
    )?;
    let final_demand = Pipeline::segment_source_demand(&stages_ref[last_split..], &eff_sink)
        .chain
        .pull;
    let (sink, chain) =
        append_reducer_sink_stages(&eff_sink, &pipeline.sink_kernels, &stage_builder, chain)?;
    let out = composed_sink::run(&sink, buf.as_slice(), chain.as_ref(), final_demand)?;

    Some(Ok(out))
}

/// Sorts `rows` by key and feeds the ordered iterator into the composed sink for top-N short-circuit.
fn run_lazy_ordered_suffix(
    stage: &Stage,
    kernel: &BodyKernel,
    sink: &super::Sink,
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
        None => crate::composed::KeySource::None,
        Some(_) => super::composed_stage::key_from_kernel(kernel)?,
    };
    let ordered = match ordered_by_key_cmp(
        rows,
        spec.descending,
        |v| Ok(key.extract(v)),
        crate::composed::cmp_val,
    ) {
        Ok(ordered) => ordered,
        Err(err) => return Some(Err(err)),
    };
    let chain =
        composed_segment::build_chain(stages, kernels, sort_idx + 1..stages.len(), stage_builder)?;
    let final_demand = Pipeline::segment_source_demand(&stages[sort_idx + 1..], sink)
        .chain
        .pull;
    let (sink, chain) = append_reducer_sink_stages(sink, sink_kernels, stage_builder, chain)?;
    composed_sink::run_owned_iter(&sink, ordered, chain.as_ref(), final_demand).map(Ok)
}

/// Promotes reducer predicate and projection into composed stages appended to `chain`, stripping them from the returned sink.
fn append_reducer_sink_stages(
    sink: &super::Sink,
    sink_kernels: &[BodyKernel],
    stage_builder: &ComposedStageBuilder<'_>,
    mut chain: Box<dyn crate::composed::Stage>,
) -> Option<(super::Sink, Box<dyn crate::composed::Stage>)> {
    let super::Sink::Reducer(spec) = sink else {
        return Some((sink.clone(), chain));
    };

    let mut sink = sink.clone();
    let super::Sink::Reducer(out_spec) = &mut sink else {
        unreachable!("cloned reducer sink changed variant");
    };

    if let Some(predicate) = &spec.predicate {
        let idx = spec.predicate_kernel_index()?;
        let kernel = sink_kernels.get(idx).unwrap_or(&BodyKernel::Generic);
        let stage = stage_builder.build_filter_program(predicate, kernel);
        chain = Box::new(crate::composed::Composed { a: chain, b: stage });
        out_spec.predicate = None;
    }

    if let Some(projection) = &spec.projection {
        let idx = spec.projection_kernel_index()?;
        let kernel = sink_kernels.get(idx).unwrap_or(&BodyKernel::Generic);
        let stage = stage_builder.build_map_program(projection, kernel);
        chain = Box::new(crate::composed::Composed { a: chain, b: stage });
        out_spec.projection = None;
    }

    Some((sink, chain))
}
