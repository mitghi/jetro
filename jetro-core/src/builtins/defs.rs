//! Per-method builtin definitions implementing the `Builtin` trait.
//!
//! One zero-sized struct per `BuiltinMethod` variant. Each struct's `impl Builtin` block
//! is the single source of truth for that method's identity, spec, and (future) runtime
//! behaviour. As migration proceeds, more methods move from the legacy `BuiltinMethod::spec()`
//! match into this file (or category-split children).

use super::{
    builtin_def::Builtin, BuiltinCardinality, BuiltinCategory, BuiltinColumnarStage,
    BuiltinDemandLaw, BuiltinMethod, BuiltinPipelineLowering, BuiltinPipelineMaterialization,
    BuiltinPipelineOrderEffect, BuiltinPipelineShape, BuiltinSelectionPosition,
    BuiltinSpec, BuiltinStageMerge, BuiltinViewStage,
};

// ── Streaming filters ────────────────────────────────────────────────────────

/// Helper: shared spec body for the predicate filter family (Filter / Find / FindAll).
/// All three are streaming filters with identical pipeline characteristics; they only
/// differ in their parser-level surface (semantic aliasing).
#[inline]
fn filter_spec() -> BuiltinSpec {
    BuiltinSpec::new(BuiltinCategory::StreamingFilter, BuiltinCardinality::Filtering)
        .view_stage(BuiltinViewStage::Filter)
        .columnar_stage(BuiltinColumnarStage::Filter)
        .cost(10.0)
        .demand_law(BuiltinDemandLaw::FilterLike)
        .order_effect(BuiltinPipelineOrderEffect::PredicatePrefix)
        .lowering(BuiltinPipelineLowering::ExprArg)
}

/// Predicate filter: keeps elements for which the lambda yields a truthy value.
pub(crate) struct Filter;
impl Builtin for Filter {
    const METHOD: BuiltinMethod = BuiltinMethod::Filter;
    const NAME: &'static str = "filter";

    fn spec() -> BuiltinSpec {
        filter_spec()
    }
}

/// Surface alias of `Filter` (same semantics; user-facing v2 name).
pub(crate) struct Find;
impl Builtin for Find {
    const METHOD: BuiltinMethod = BuiltinMethod::Find;
    const NAME: &'static str = "find";

    fn spec() -> BuiltinSpec {
        filter_spec()
    }
}

/// Surface alias of `Filter` (same semantics; user-facing v2 name).
pub(crate) struct FindAll;
impl Builtin for FindAll {
    const METHOD: BuiltinMethod = BuiltinMethod::FindAll;
    const NAME: &'static str = "find_all";

    fn spec() -> BuiltinSpec {
        filter_spec()
    }
}

/// Removes nullish elements; degenerate filter with no lambda.
pub(crate) struct Compact;
impl Builtin for Compact {
    const METHOD: BuiltinMethod = BuiltinMethod::Compact;
    const NAME: &'static str = "compact";

    fn spec() -> BuiltinSpec {
        BuiltinSpec::new(BuiltinCategory::StreamingFilter, BuiltinCardinality::Filtering).cost(10.0)
    }
}

/// Removes elements equal to the literal argument; degenerate equality filter.
pub(crate) struct Remove;
impl Builtin for Remove {
    const METHOD: BuiltinMethod = BuiltinMethod::Remove;
    const NAME: &'static str = "remove";

    fn spec() -> BuiltinSpec {
        BuiltinSpec::new(BuiltinCategory::StreamingFilter, BuiltinCardinality::Filtering).cost(10.0)
    }
}

// ── Streaming one-to-one ─────────────────────────────────────────────────────

/// Per-element projection via lambda; preserves cardinality and order.
pub(crate) struct Map;
impl Builtin for Map {
    const METHOD: BuiltinMethod = BuiltinMethod::Map;
    const NAME: &'static str = "map";

    fn spec() -> BuiltinSpec {
        BuiltinSpec::new(BuiltinCategory::StreamingOneToOne, BuiltinCardinality::OneToOne)
            .indexed()
            .view_stage(BuiltinViewStage::Map)
            .columnar_stage(BuiltinColumnarStage::Map)
            .cost(10.0)
            .demand_law(BuiltinDemandLaw::MapLike)
            .order_effect(BuiltinPipelineOrderEffect::Preserves)
            .lowering(BuiltinPipelineLowering::ExprArg)
            .element()
    }
}

/// Expanding projection: each element produces an array; outputs are concatenated.
pub(crate) struct FlatMap;
impl Builtin for FlatMap {
    const METHOD: BuiltinMethod = BuiltinMethod::FlatMap;
    const NAME: &'static str = "flat_map";

    fn spec() -> BuiltinSpec {
        BuiltinSpec::new(BuiltinCategory::StreamingExpand, BuiltinCardinality::Expanding)
            .view_stage(BuiltinViewStage::FlatMap)
            .columnar_stage(BuiltinColumnarStage::FlatMap)
            .cost(10.0)
            .demand_law(BuiltinDemandLaw::FlatMapLike)
            .materialization(BuiltinPipelineMaterialization::LegacyMaterialized)
            .lowering(BuiltinPipelineLowering::ExprArg)
    }
}

// ── Bounded prefix / positional ──────────────────────────────────────────────

/// Take first N elements; bounded positional slice.
pub(crate) struct Take;
impl Builtin for Take {
    const METHOD: BuiltinMethod = BuiltinMethod::Take;
    const NAME: &'static str = "take";

    fn spec() -> BuiltinSpec {
        BuiltinSpec::new(BuiltinCategory::Positional, BuiltinCardinality::Bounded)
            .view_native()
            .view_stage(BuiltinViewStage::Take)
            .stage_merge(BuiltinStageMerge::UsizeMin)
            .demand_law(BuiltinDemandLaw::Take)
            .order_effect(BuiltinPipelineOrderEffect::Preserves)
            .lowering(BuiltinPipelineLowering::UsizeArg { min: 0 })
    }
}

/// Skip first N elements; bounded positional offset.
pub(crate) struct Skip;
impl Builtin for Skip {
    const METHOD: BuiltinMethod = BuiltinMethod::Skip;
    const NAME: &'static str = "skip";

    fn spec() -> BuiltinSpec {
        BuiltinSpec::new(BuiltinCategory::Positional, BuiltinCardinality::Bounded)
            .view_native()
            .view_stage(BuiltinViewStage::Skip)
            .stage_merge(BuiltinStageMerge::UsizeSaturatingAdd)
            .demand_law(BuiltinDemandLaw::Skip)
            .order_effect(BuiltinPipelineOrderEffect::Preserves)
            .lowering(BuiltinPipelineLowering::UsizeArg { min: 0 })
    }
}

/// Selects the first element; terminal positional sink.
pub(crate) struct First;
impl Builtin for First {
    const METHOD: BuiltinMethod = BuiltinMethod::First;
    const NAME: &'static str = "first";

    fn spec() -> BuiltinSpec {
        BuiltinSpec::new(BuiltinCategory::Positional, BuiltinCardinality::Bounded)
            .view_native()
            .select_one_sink(BuiltinSelectionPosition::First)
            .demand_law(BuiltinDemandLaw::First)
            .lowering(BuiltinPipelineLowering::TerminalSink)
    }
}

/// Selects the last element; terminal positional sink.
pub(crate) struct Last;
impl Builtin for Last {
    const METHOD: BuiltinMethod = BuiltinMethod::Last;
    const NAME: &'static str = "last";

    fn spec() -> BuiltinSpec {
        BuiltinSpec::new(BuiltinCategory::Positional, BuiltinCardinality::Bounded)
            .view_native()
            .select_one_sink(BuiltinSelectionPosition::Last)
            .demand_law(BuiltinDemandLaw::Last)
            .lowering(BuiltinPipelineLowering::TerminalSink)
    }
}

// ── Bounded prefix predicates ────────────────────────────────────────────────

/// Take elements while predicate holds; stops at first failure.
pub(crate) struct TakeWhile;
impl Builtin for TakeWhile {
    const METHOD: BuiltinMethod = BuiltinMethod::TakeWhile;
    const NAME: &'static str = "take_while";

    fn spec() -> BuiltinSpec {
        BuiltinSpec::new(BuiltinCategory::StreamingFilter, BuiltinCardinality::Filtering)
            .view_stage(BuiltinViewStage::TakeWhile)
            .cost(10.0)
            .demand_law(BuiltinDemandLaw::TakeWhile)
            .pipeline_shape(BuiltinPipelineShape::new(
                BuiltinCardinality::Filtering,
                true,
                10.0,
                0.5,
            ))
            .order_effect(BuiltinPipelineOrderEffect::PredicatePrefix)
            .lowering(BuiltinPipelineLowering::ExprArg)
    }
}

/// Skip elements while predicate holds; emits the remainder.
pub(crate) struct DropWhile;
impl Builtin for DropWhile {
    const METHOD: BuiltinMethod = BuiltinMethod::DropWhile;
    const NAME: &'static str = "drop_while";

    fn spec() -> BuiltinSpec {
        BuiltinSpec::new(BuiltinCategory::StreamingFilter, BuiltinCardinality::Filtering)
            .view_stage(BuiltinViewStage::DropWhile)
            .cost(10.0)
            .materialization(BuiltinPipelineMaterialization::LegacyMaterialized)
            .pipeline_shape(BuiltinPipelineShape::new(
                BuiltinCardinality::Filtering,
                true,
                10.0,
                0.5,
            ))
            .order_effect(BuiltinPipelineOrderEffect::Blocks)
            .lowering(BuiltinPipelineLowering::ExprArg)
    }
}
