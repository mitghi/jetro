//! Per-method builtin definitions implementing the `Builtin` trait.
//!
//! One zero-sized struct per `BuiltinMethod` variant. Each struct's `impl Builtin` block
//! is the single source of truth for that method's identity, spec, and (future) runtime
//! behaviour. As migration proceeds, more methods move from the legacy `BuiltinMethod::spec()`
//! match into this file (or category-split children).

use super::{
    builtin_def::Builtin, BuiltinCardinality, BuiltinCategory, BuiltinColumnarStage,
    BuiltinDemandLaw, BuiltinMethod, BuiltinPipelineLowering, BuiltinPipelineOrderEffect,
    BuiltinSpec, BuiltinViewStage,
};

// ── Streaming filters ────────────────────────────────────────────────────────

/// Predicate filter: keeps elements for which the lambda yields a truthy value.
pub(crate) struct Filter;
impl Builtin for Filter {
    const METHOD: BuiltinMethod = BuiltinMethod::Filter;
    const NAME: &'static str = "filter";

    fn spec() -> BuiltinSpec {
        BuiltinSpec::new(BuiltinCategory::StreamingFilter, BuiltinCardinality::Filtering)
            .view_stage(BuiltinViewStage::Filter)
            .columnar_stage(BuiltinColumnarStage::Filter)
            .cost(10.0)
            .demand_law(BuiltinDemandLaw::FilterLike)
            .order_effect(BuiltinPipelineOrderEffect::PredicatePrefix)
            .lowering(BuiltinPipelineLowering::ExprArg)
    }
}
