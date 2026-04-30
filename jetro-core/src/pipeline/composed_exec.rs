use crate::{
    context::{Env, EvalError},
    value::Val,
};

use super::composed_barrier::{self, BarrierOutput};
use super::composed_segment;
use super::composed_sink;
use super::composed_source;
use super::composed_stage::ComposedStageBuilder;
use super::{compute_strategies, BodyKernel, Pipeline, StageStrategy};

pub(super) fn run(
    pipeline: &Pipeline,
    root: &Val,
    base_env: &Env,
) -> Option<Result<Val, EvalError>> {
    let (eff_stages, eff_kernels, eff_sink) = pipeline.canonical();
    let stage_builder = ComposedStageBuilder::new(base_env);

    let mut buf = composed_source::rows(&pipeline.source, root)?;

    // Walk stages, splitting at barriers. Each streaming run uses a
    // composed-Cow chain into a CollectSink to materialise the
    // intermediate Vec<Val>; each barrier consumes Vec, returns Vec.
    let kernels = &eff_kernels;
    let stages_ref = &eff_stages;

    // Demand propagation is the generic performance hook: stages pick
    // algorithms from downstream demand instead of query-specific rewrites.
    let strategies = compute_strategies(stages_ref, &eff_sink);

    let mut last_split = 0usize;
    for (i, stage) in stages_ref.iter().enumerate() {
        if !stage.is_composed_barrier() {
            continue;
        }

        if i > last_split {
            let chain =
                composed_segment::build_chain(stages_ref, kernels, last_split..i, &stage_builder)?;
            buf = composed_segment::collect(&buf, chain.as_ref())?;
        }

        let kernel = kernels.get(i).unwrap_or(&BodyKernel::Generic);
        let strategy = strategies.get(i).copied().unwrap_or(StageStrategy::Default);
        match composed_barrier::run(
            stage,
            kernel,
            strategy,
            &eff_sink,
            i + 1 == stages_ref.len(),
            buf,
        )? {
            BarrierOutput::Rows(rows) => buf = rows,
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
    let out = composed_sink::run(&eff_sink, &buf, chain.as_ref(), final_demand)?;

    Some(Ok(out))
}
