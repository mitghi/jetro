//! Per-method builtin definitions implementing the `Builtin` trait.
//!
//! One zero-sized struct per `BuiltinMethod` variant. Each struct's `impl Builtin` block
//! is the single source of truth for that method's identity, spec, and (future) runtime
//! behaviour. As migration proceeds, more methods move from the legacy `BuiltinMethod::spec()`
//! match into this file (or category-split children).

use super::{
    builtin::Builtin, BuiltinCancelGroup, BuiltinCancelSide, BuiltinCancellation,
    BuiltinCardinality, BuiltinCategory, BuiltinColumnarStage, BuiltinDemandLaw,
    BuiltinKeyedReducer, BuiltinMethod, BuiltinNumericReducer, BuiltinPipelineLowering,
    BuiltinPipelineMaterialization, BuiltinPipelineOrderEffect, BuiltinPipelineShape,
    BuiltinSelectionPosition, BuiltinSpec, BuiltinStageMerge, BuiltinStructural, BuiltinViewStage,
};

// ── Helpers shared across reducer family ─────────────────────────────────────

/// Numeric reducer (sum/avg/min/max) skeleton; same demand/lowering across the four.
#[inline]
fn numeric_reducer_spec(reducer: BuiltinNumericReducer) -> BuiltinSpec {
    BuiltinSpec::new(BuiltinCategory::Reducer, BuiltinCardinality::Reducing)
        .view_native()
        .numeric_sink(reducer)
        .cost(10.0)
        .demand_law(BuiltinDemandLaw::NumericReducer)
        .lowering(BuiltinPipelineLowering::TerminalSink)
}

/// Predicate-driven reducer-with-take-first skeleton (FindIndex / IndicesWhere / MaxBy / MinBy).
#[inline]
fn predicate_reducer_spec() -> BuiltinSpec {
    BuiltinSpec::new(BuiltinCategory::Reducer, BuiltinCardinality::Reducing)
        .view_native()
        .cost(10.0)
        .materialization(BuiltinPipelineMaterialization::LegacyMaterialized)
        .pipeline_shape(BuiltinPipelineShape::new(
            BuiltinCardinality::OneToOne,
            true,
            1.0,
            1.0,
        ))
        .lowering(BuiltinPipelineLowering::TerminalExprArg {
            terminal: BuiltinMethod::First,
        })
}

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

    #[inline]
    fn apply_stream(
        ctx: &mut super::builtin::StreamCtx<'_, '_>,
        item: crate::data::value::Val,
        body: Option<&crate::vm::Program>,
    ) -> Result<crate::exec::pipeline::StageFlow<crate::data::value::Val>, crate::data::context::EvalError> {
        let prog = body.expect("filter body");
        let keep = super::filter_one(&item, |v| {
            crate::exec::pipeline::eval_kernel(ctx.kernel, v, |it| {
                crate::exec::pipeline::apply_item_in_env(ctx.vm, ctx.env, it, prog)
            })
        })?;
        Ok(if keep {
            crate::exec::pipeline::StageFlow::Continue(item)
        } else {
            crate::exec::pipeline::StageFlow::SkipRow
        })
    }
#[inline]
    fn apply_barrier(
        ctx: &mut super::builtin::BarrierCtx<'_>,
        buf: &mut Vec<crate::data::value::Val>,
        body: Option<&crate::vm::Program>,
    ) -> Option<Result<(), crate::data::context::EvalError>> {
        let prog = body?;
        let result = super::filter_apply(std::mem::take(buf), |v| {
            crate::exec::pipeline::eval_kernel(ctx.kernel, v, |item| {
                crate::exec::pipeline::apply_item_in_env(ctx.vm, ctx.env, item, prog)
            })
        });
        match result {
            Ok(out) => { *buf = out; Some(Ok(())) }
            Err(err) => Some(Err(err)),
        }
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
        BuiltinSpec::new(BuiltinCategory::StreamingFilter, BuiltinCardinality::Filtering)
            .cost(10.0)
            .demand_law(BuiltinDemandLaw::FilterLike)
            .order_effect(BuiltinPipelineOrderEffect::PredicatePrefix)
    }
    #[inline]
    fn apply_one(recv: &crate::data::value::Val) -> Option<crate::data::value::Val> {
        Some(super::compact_apply(recv).unwrap_or_else(|| recv.clone()))
    }
}

/// Removes elements equal to the literal argument; degenerate equality filter.
pub(crate) struct Remove;
impl Builtin for Remove {
    const METHOD: BuiltinMethod = BuiltinMethod::Remove;
    const NAME: &'static str = "remove";

    fn spec() -> BuiltinSpec {
        BuiltinSpec::new(BuiltinCategory::StreamingFilter, BuiltinCardinality::Filtering)
            .cost(10.0)
            .demand_law(BuiltinDemandLaw::FilterLike)
            .order_effect(BuiltinPipelineOrderEffect::PredicatePrefix)
    }
    #[inline]
    fn apply_args(recv: &crate::data::value::Val, args: &super::BuiltinArgs) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Val(item) => Some(super::remove_value_apply(recv, item).unwrap_or_else(|| recv.clone())),
            _ => None,
        }
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

    #[inline]
    fn apply_stream(
        ctx: &mut super::builtin::StreamCtx<'_, '_>,
        item: crate::data::value::Val,
        body: Option<&crate::vm::Program>,
    ) -> Result<crate::exec::pipeline::StageFlow<crate::data::value::Val>, crate::data::context::EvalError> {
        let prog = body.expect("map body");
        // Terminal-map collector short-circuit (avoid allocating intermediate Val).
        if Some(ctx.stage_idx) == ctx.terminal_map_idx {
            ctx.terminal_map_collect
                .as_mut()
                .expect("terminal map collector")
                .push_val_row(&item, ctx.kernel, |it| {
                    crate::exec::pipeline::apply_item_in_env(ctx.vm, ctx.env, it, prog)
                })?;
            return Ok(crate::exec::pipeline::StageFlow::TerminalCollected);
        }
        let mapped = super::map_one(&item, |v| {
            crate::exec::pipeline::eval_kernel(ctx.kernel, v, |it| {
                crate::exec::pipeline::apply_item_in_env(ctx.vm, ctx.env, it, prog)
            })
        })?;
        Ok(crate::exec::pipeline::StageFlow::Continue(mapped))
    }

    #[inline]
    fn apply_barrier(
        ctx: &mut super::builtin::BarrierCtx<'_>,
        buf: &mut Vec<crate::data::value::Val>,
        body: Option<&crate::vm::Program>,
    ) -> Option<Result<(), crate::data::context::EvalError>> {
        let prog = body?;
        let result = super::map_apply(std::mem::take(buf), |v| {
            crate::exec::pipeline::eval_kernel(ctx.kernel, v, |item| {
                crate::exec::pipeline::apply_item_in_env(ctx.vm, ctx.env, item, prog)
            })
        });
        match result {
            Ok(out) => { *buf = out; Some(Ok(())) }
            Err(err) => Some(Err(err)),
        }
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

    #[inline]
    fn apply_barrier(
        ctx: &mut super::builtin::BarrierCtx<'_>,
        buf: &mut Vec<crate::data::value::Val>,
        body: Option<&crate::vm::Program>,
    ) -> Option<Result<(), crate::data::context::EvalError>> {
        let prog = body?;
        let mut out: Vec<crate::data::value::Val> = Vec::new();
        for v in buf.iter() {
            let inner = match crate::exec::pipeline::eval_kernel(ctx.kernel, v, |item| {
                crate::exec::pipeline::apply_item_in_env(ctx.vm, ctx.env, item, prog)
            }) {
                Ok(inner) => inner,
                Err(err) => return Some(Err(err)),
            };
            if let Some(arr) = inner.as_vals() {
                out.extend(arr.iter().cloned());
            } else {
                out.push(inner);
            }
        }
        *buf = out;
        Some(Ok(()))
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

    #[inline]
    fn apply_stream(
        ctx: &mut super::builtin::StreamCtx<'_, '_>,
        item: crate::data::value::Val,
        _body: Option<&crate::vm::Program>,
    ) -> Result<crate::exec::pipeline::StageFlow<crate::data::value::Val>, crate::data::context::EvalError> {
        let n = match ctx.stage.descriptor().and_then(|d| d.usize_arg) {
            Some(n) => n,
            None => return Ok(crate::exec::pipeline::StageFlow::Continue(item)),
        };
        if ctx.stage_taken[ctx.stage_idx] >= n {
            Ok(crate::exec::pipeline::StageFlow::Stop)
        } else {
            ctx.stage_taken[ctx.stage_idx] += 1;
            Ok(crate::exec::pipeline::StageFlow::Continue(item))
        }
    }

    #[inline]
    fn apply_barrier(
        ctx: &mut super::builtin::BarrierCtx<'_>,
        buf: &mut Vec<crate::data::value::Val>,
        _body: Option<&crate::vm::Program>,
    ) -> Option<Result<(), crate::data::context::EvalError>> {
        let n = ctx.stage.descriptor().and_then(|d| d.usize_arg)?;
        buf.truncate(n);
        Some(Ok(()))
    }
}

/// Skip first N elements; bounded positional offset.
pub(crate) struct Skip;
impl Builtin for Skip {
    const METHOD: BuiltinMethod = BuiltinMethod::Skip;
    const NAME: &'static str = "skip";
    const ALIASES: &'static [&'static str] = &["drop"];

    fn spec() -> BuiltinSpec {
        BuiltinSpec::new(BuiltinCategory::Positional, BuiltinCardinality::Bounded)
            .view_native()
            .view_stage(BuiltinViewStage::Skip)
            .stage_merge(BuiltinStageMerge::UsizeSaturatingAdd)
            .demand_law(BuiltinDemandLaw::Skip)
            .order_effect(BuiltinPipelineOrderEffect::Preserves)
            .lowering(BuiltinPipelineLowering::UsizeArg { min: 0 })
    }

    #[inline]
    fn apply_stream(
        ctx: &mut super::builtin::StreamCtx<'_, '_>,
        item: crate::data::value::Val,
        _body: Option<&crate::vm::Program>,
    ) -> Result<crate::exec::pipeline::StageFlow<crate::data::value::Val>, crate::data::context::EvalError> {
        let n = match ctx.stage.descriptor().and_then(|d| d.usize_arg) {
            Some(n) => n,
            None => return Ok(crate::exec::pipeline::StageFlow::Continue(item)),
        };
        if ctx.stage_skipped[ctx.stage_idx] < n {
            ctx.stage_skipped[ctx.stage_idx] += 1;
            Ok(crate::exec::pipeline::StageFlow::SkipRow)
        } else {
            Ok(crate::exec::pipeline::StageFlow::Continue(item))
        }
    }

    #[inline]
    fn apply_barrier(
        ctx: &mut super::builtin::BarrierCtx<'_>,
        buf: &mut Vec<crate::data::value::Val>,
        _body: Option<&crate::vm::Program>,
    ) -> Option<Result<(), crate::data::context::EvalError>> {
        let n = ctx.stage.descriptor().and_then(|d| d.usize_arg)?;
        if buf.len() <= n {
            buf.clear();
        } else {
            buf.drain(..n);
        }
        Some(Ok(()))
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
    #[inline]
    fn apply_args(recv: &crate::data::value::Val, args: &super::BuiltinArgs) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::I64(n) => { super::first_apply(recv, *n) }
            _ => None,
        }
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
    #[inline]
    fn apply_args(recv: &crate::data::value::Val, args: &super::BuiltinArgs) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::I64(n) => { super::last_apply(recv, *n) }
            _ => None,
        }
    }
}

// ── Bounded prefix predicates ────────────────────────────────────────────────

/// Take elements while predicate holds; stops at first failure.
pub(crate) struct TakeWhile;
impl Builtin for TakeWhile {
    const METHOD: BuiltinMethod = BuiltinMethod::TakeWhile;
    const NAME: &'static str = "take_while";
    const ALIASES: &'static [&'static str] = &["takewhile"];

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

    #[inline]
    fn apply_stream(
        ctx: &mut super::builtin::StreamCtx<'_, '_>,
        item: crate::data::value::Val,
        body: Option<&crate::vm::Program>,
    ) -> Result<crate::exec::pipeline::StageFlow<crate::data::value::Val>, crate::data::context::EvalError> {
        let prog = body.expect("take_while body");
        let pass = super::take_while_one(&item, |v| {
            crate::exec::pipeline::eval_kernel(ctx.kernel, v, |it| {
                crate::exec::pipeline::apply_item_in_env(ctx.vm, ctx.env, it, prog)
            })
        })?;
        Ok(if pass {
            crate::exec::pipeline::StageFlow::Continue(item)
        } else {
            crate::exec::pipeline::StageFlow::Stop
        })
    }

    #[inline]
    fn apply_barrier(
        ctx: &mut super::builtin::BarrierCtx<'_>,
        buf: &mut Vec<crate::data::value::Val>,
        body: Option<&crate::vm::Program>,
    ) -> Option<Result<(), crate::data::context::EvalError>> {
        let prog = body?;
        let result = super::take_while_apply(std::mem::take(buf), |v| {
            crate::exec::pipeline::eval_kernel(ctx.kernel, v, |item| {
                crate::exec::pipeline::apply_item_in_env(ctx.vm, ctx.env, item, prog)
            })
        });
        match result {
            Ok(out) => { *buf = out; Some(Ok(())) }
            Err(err) => Some(Err(err)),
        }
    }
}

/// Skip elements while predicate holds; emits the remainder.
pub(crate) struct DropWhile;
impl Builtin for DropWhile {
    const METHOD: BuiltinMethod = BuiltinMethod::DropWhile;
    const NAME: &'static str = "drop_while";
    const ALIASES: &'static [&'static str] = &["dropwhile"];

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

    /// DropWhile in the streaming loop is a no-op pass-through (the materialised
    /// barrier path handles the actual drop semantics in materialized_exec). Mirrors
    /// the original `PrefixWhile { take: false }` arm in val_stage_flow.
    #[inline]
    fn apply_stream(
        _ctx: &mut super::builtin::StreamCtx<'_, '_>,
        item: crate::data::value::Val,
        _body: Option<&crate::vm::Program>,
    ) -> Result<crate::exec::pipeline::StageFlow<crate::data::value::Val>, crate::data::context::EvalError> {
        Ok(crate::exec::pipeline::StageFlow::Continue(item))
    }

    #[inline]
    fn apply_barrier(
        ctx: &mut super::builtin::BarrierCtx<'_>,
        buf: &mut Vec<crate::data::value::Val>,
        body: Option<&crate::vm::Program>,
    ) -> Option<Result<(), crate::data::context::EvalError>> {
        let prog = body?;
        let result = super::drop_while_apply(std::mem::take(buf), |v| {
            crate::exec::pipeline::eval_kernel(ctx.kernel, v, |item| {
                crate::exec::pipeline::apply_item_in_env(ctx.vm, ctx.env, item, prog)
            })
        });
        match result {
            Ok(out) => { *buf = out; Some(Ok(())) }
            Err(err) => Some(Err(err)),
        }
    }
}

// ── Reducer sinks ────────────────────────────────────────────────────────────

/// Element count via scalar view sink; degenerate non-numeric reducer.
pub(crate) struct Len;
impl Builtin for Len {
    const METHOD: BuiltinMethod = BuiltinMethod::Len;
    const NAME: &'static str = "len";

    fn spec() -> BuiltinSpec {
        BuiltinSpec::new(BuiltinCategory::Reducer, BuiltinCardinality::Reducing)
            .indexed()
            .view_scalar()
            .count_sink()
    }
    #[inline]
    fn apply_one(recv: &crate::data::value::Val) -> Option<crate::data::value::Val> {
        Some(super::len_apply(recv).unwrap_or_else(|| recv.clone()))
    }
}

/// Sum of numeric stream elements.
pub(crate) struct Sum;
impl Builtin for Sum {
    const METHOD: BuiltinMethod = BuiltinMethod::Sum;
    const NAME: &'static str = "sum";

    fn spec() -> BuiltinSpec {
        numeric_reducer_spec(BuiltinNumericReducer::Sum)
    }
}

/// Arithmetic mean of numeric stream elements.
pub(crate) struct Avg;
impl Builtin for Avg {
    const METHOD: BuiltinMethod = BuiltinMethod::Avg;
    const NAME: &'static str = "avg";

    fn spec() -> BuiltinSpec {
        numeric_reducer_spec(BuiltinNumericReducer::Avg)
    }
}

/// Smallest numeric element.
pub(crate) struct Min;
impl Builtin for Min {
    const METHOD: BuiltinMethod = BuiltinMethod::Min;
    const NAME: &'static str = "min";

    fn spec() -> BuiltinSpec {
        numeric_reducer_spec(BuiltinNumericReducer::Min)
    }
}

/// Largest numeric element.
pub(crate) struct Max;
impl Builtin for Max {
    const METHOD: BuiltinMethod = BuiltinMethod::Max;
    const NAME: &'static str = "max";

    fn spec() -> BuiltinSpec {
        numeric_reducer_spec(BuiltinNumericReducer::Max)
    }
}

/// Stream length count; differs from `Len` in being a streaming reducer (not scalar).
pub(crate) struct Count;
impl Builtin for Count {
    const METHOD: BuiltinMethod = BuiltinMethod::Count;
    const NAME: &'static str = "count";

    fn spec() -> BuiltinSpec {
        BuiltinSpec::new(BuiltinCategory::Reducer, BuiltinCardinality::Reducing)
            .view_native()
            .count_sink()
            .cost(10.0)
            .demand_law(BuiltinDemandLaw::Count)
            .lowering(BuiltinPipelineLowering::TerminalSink)
    }
}

/// HyperLogLog-style approximate distinct count.
pub(crate) struct ApproxCountDistinct;
impl Builtin for ApproxCountDistinct {
    const METHOD: BuiltinMethod = BuiltinMethod::ApproxCountDistinct;
    const NAME: &'static str = "approx_count_distinct";

    fn spec() -> BuiltinSpec {
        BuiltinSpec::new(BuiltinCategory::Reducer, BuiltinCardinality::Reducing)
            .view_native()
            .approx_distinct_sink()
            .cost(10.0)
            .demand_law(BuiltinDemandLaw::KeyedReducer)
            .lowering(BuiltinPipelineLowering::TerminalSink)
    }
}

/// Boolean reducer: true if any element matches predicate.
pub(crate) struct Any;
impl Builtin for Any {
    const METHOD: BuiltinMethod = BuiltinMethod::Any;
    const NAME: &'static str = "any";
    const ALIASES: &'static [&'static str] = &["exists"];

    fn spec() -> BuiltinSpec {
        BuiltinSpec::new(BuiltinCategory::Reducer, BuiltinCardinality::Reducing)
            .view_native()
            .cost(10.0)
    }
}

/// Boolean reducer: true if all elements match predicate.
pub(crate) struct All;
impl Builtin for All {
    const METHOD: BuiltinMethod = BuiltinMethod::All;
    const NAME: &'static str = "all";

    fn spec() -> BuiltinSpec {
        BuiltinSpec::new(BuiltinCategory::Reducer, BuiltinCardinality::Reducing)
            .view_native()
            .cost(10.0)
    }
}

/// Index of the first element satisfying the predicate.
pub(crate) struct FindIndex;
impl Builtin for FindIndex {
    const METHOD: BuiltinMethod = BuiltinMethod::FindIndex;
    const NAME: &'static str = "find_index";

    fn spec() -> BuiltinSpec {
        predicate_reducer_spec()
    }

    #[inline]
    fn apply_barrier(
        ctx: &mut super::builtin::BarrierCtx<'_>,
        buf: &mut Vec<crate::data::value::Val>,
        body: Option<&crate::vm::Program>,
    ) -> Option<Result<(), crate::data::context::EvalError>> {
        let prog = body?;
        let mut found: crate::data::value::Val = crate::data::value::Val::Null;
        for (i, v) in buf.iter().enumerate() {
            match super::filter_one(v, |item| {
                crate::exec::pipeline::eval_kernel(ctx.kernel, item, |it| {
                    crate::exec::pipeline::apply_item_in_env(ctx.vm, ctx.env, it, prog)
                })
            }) {
                Ok(true) => {
                    found = crate::data::value::Val::Int(i as i64);
                    break;
                }
                Ok(false) => {}
                Err(err) => return Some(Err(err)),
            }
        }
        *buf = vec![found];
        Some(Ok(()))
    }
}

/// Indices of all elements satisfying the predicate.
pub(crate) struct IndicesWhere;
impl Builtin for IndicesWhere {
    const METHOD: BuiltinMethod = BuiltinMethod::IndicesWhere;
    const NAME: &'static str = "indices_where";

    fn spec() -> BuiltinSpec {
        predicate_reducer_spec()
    }

    #[inline]
    fn apply_barrier(
        ctx: &mut super::builtin::BarrierCtx<'_>,
        buf: &mut Vec<crate::data::value::Val>,
        body: Option<&crate::vm::Program>,
    ) -> Option<Result<(), crate::data::context::EvalError>> {
        let prog = body?;
        let mut out: Vec<i64> = Vec::new();
        for (i, v) in buf.iter().enumerate() {
            match super::filter_one(v, |item| {
                crate::exec::pipeline::eval_kernel(ctx.kernel, item, |it| {
                    crate::exec::pipeline::apply_item_in_env(ctx.vm, ctx.env, it, prog)
                })
            }) {
                Ok(true) => out.push(i as i64),
                Ok(false) => {}
                Err(err) => return Some(Err(err)),
            }
        }
        *buf = vec![crate::data::value::Val::int_vec(out)];
        Some(Ok(()))
    }
}

/// Shared barrier body for MaxBy / MinBy (ArgExtreme).
#[inline]
fn arg_extreme_apply_barrier(
    ctx: &mut super::builtin::BarrierCtx<'_>,
    buf: &mut Vec<crate::data::value::Val>,
    body: Option<&crate::vm::Program>,
    max: bool,
) -> Option<Result<(), crate::data::context::EvalError>> {
    let prog = body?;
    if buf.is_empty() {
        *buf = vec![crate::data::value::Val::Null];
        return Some(Ok(()));
    }
    let mut best_idx = 0usize;
    let mut best_key = match crate::exec::pipeline::eval_kernel(ctx.kernel, &buf[0], |item| {
        crate::exec::pipeline::apply_item_in_env(ctx.vm, ctx.env, item, prog)
    }) {
        Ok(key) => key,
        Err(err) => return Some(Err(err)),
    };
    for i in 1..buf.len() {
        let key = match crate::exec::pipeline::eval_kernel(ctx.kernel, &buf[i], |item| {
            crate::exec::pipeline::apply_item_in_env(ctx.vm, ctx.env, item, prog)
        }) {
            Ok(key) => key,
            Err(err) => return Some(Err(err)),
        };
        let cmp = crate::exec::pipeline::cmp_val_total(&key, &best_key);
        let take = if max {
            cmp == std::cmp::Ordering::Greater
        } else {
            cmp == std::cmp::Ordering::Less
        };
        if take {
            best_idx = i;
            best_key = key;
        }
    }
    let best = std::mem::take(buf).into_iter().nth(best_idx).unwrap();
    *buf = vec![best];
    Some(Ok(()))
}

/// Element with the largest projected key.
pub(crate) struct MaxBy;
impl Builtin for MaxBy {
    const METHOD: BuiltinMethod = BuiltinMethod::MaxBy;
    const NAME: &'static str = "max_by";

    fn spec() -> BuiltinSpec {
        predicate_reducer_spec()
    }

    #[inline]
    fn apply_barrier(
        ctx: &mut super::builtin::BarrierCtx<'_>,
        buf: &mut Vec<crate::data::value::Val>,
        body: Option<&crate::vm::Program>,
    ) -> Option<Result<(), crate::data::context::EvalError>> {
        arg_extreme_apply_barrier(ctx, buf, body, true)
    }
}

/// Element with the smallest projected key.
pub(crate) struct MinBy;
impl Builtin for MinBy {
    const METHOD: BuiltinMethod = BuiltinMethod::MinBy;
    const NAME: &'static str = "min_by";

    fn spec() -> BuiltinSpec {
        predicate_reducer_spec()
    }

    #[inline]
    fn apply_barrier(
        ctx: &mut super::builtin::BarrierCtx<'_>,
        buf: &mut Vec<crate::data::value::Val>,
        body: Option<&crate::vm::Program>,
    ) -> Option<Result<(), crate::data::context::EvalError>> {
        arg_extreme_apply_barrier(ctx, buf, body, false)
    }
}

// ── Indexed/element-only streaming ───────────────────────────────────────────

/// `enumerate` — pairs each element with its index; element-wise.
pub(crate) struct Enumerate;
impl Builtin for Enumerate {
    const METHOD: BuiltinMethod = BuiltinMethod::Enumerate;
    const NAME: &'static str = "enumerate";
    fn spec() -> BuiltinSpec {
        BuiltinSpec::new(BuiltinCategory::StreamingOneToOne, BuiltinCardinality::OneToOne)
            .indexed()
            .cost(10.0)
            .element()
    }
    #[inline]
    fn apply_one(recv: &crate::data::value::Val) -> Option<crate::data::value::Val> {
        Some(super::enumerate_apply(recv).unwrap_or_else(|| recv.clone()))
    }
}

/// `pairwise` — yields adjacent pairs; element-wise indexed.
pub(crate) struct Pairwise;
impl Builtin for Pairwise {
    const METHOD: BuiltinMethod = BuiltinMethod::Pairwise;
    const NAME: &'static str = "pairwise";
    fn spec() -> BuiltinSpec {
        BuiltinSpec::new(BuiltinCategory::StreamingOneToOne, BuiltinCardinality::OneToOne)
            .indexed()
            .cost(10.0)
            .element()
    }
    #[inline]
    fn apply_one(recv: &crate::data::value::Val) -> Option<crate::data::value::Val> {
        Some(super::pairwise_apply(recv).unwrap_or_else(|| recv.clone()))
    }
}

// ── Expanding (no lambda) ────────────────────────────────────────────────────

#[inline]
fn expand_simple_spec() -> BuiltinSpec {
    BuiltinSpec::new(BuiltinCategory::StreamingExpand, BuiltinCardinality::Expanding).cost(10.0)
}

/// `flatten` — concatenates nested arrays.
pub(crate) struct Flatten;
impl Builtin for Flatten {
    const METHOD: BuiltinMethod = BuiltinMethod::Flatten;
    const NAME: &'static str = "flatten";
    fn spec() -> BuiltinSpec { expand_simple_spec() }
    #[inline]
    fn apply_args(recv: &crate::data::value::Val, args: &super::BuiltinArgs) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Usize(depth) => { super::flatten_depth_apply(recv, *depth) }
            _ => None,
        }
    }
}

/// `explode` — same as flatten with object semantics.
pub(crate) struct Explode;
impl Builtin for Explode {
    const METHOD: BuiltinMethod = BuiltinMethod::Explode;
    const NAME: &'static str = "explode";
    fn spec() -> BuiltinSpec { expand_simple_spec() }
    #[inline]
    fn apply_args(recv: &crate::data::value::Val, args: &super::BuiltinArgs) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Str(field) => { super::explode_apply(recv, field) }
            _ => None,
        }
    }
}

/// `split(sep)` — string-arg expansion stage.
pub(crate) struct Split;
impl Builtin for Split {
    const METHOD: BuiltinMethod = BuiltinMethod::Split;
    const NAME: &'static str = "split";
    fn spec() -> BuiltinSpec {
        BuiltinSpec::new(BuiltinCategory::StreamingExpand, BuiltinCardinality::Expanding)
            .cost(10.0)
            .materialization(BuiltinPipelineMaterialization::LegacyMaterialized)
            .pipeline_shape(BuiltinPipelineShape::new(
                BuiltinCardinality::Expanding,
                true,
                2.0,
                1.0,
            ))
            .lowering(BuiltinPipelineLowering::StringArg)
    }
    #[inline]
    fn apply_args(recv: &crate::data::value::Val, args: &super::BuiltinArgs) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Str(p) => { super::split_apply(recv, p) }
            _ => None,
        }
    }
}

#[inline]
fn expand_element_spec() -> BuiltinSpec {
    BuiltinSpec::new(BuiltinCategory::StreamingExpand, BuiltinCardinality::Expanding)
        .cost(10.0)
        .element()
}

/// `lines` — split string on newlines.
pub(crate) struct Lines;
impl Builtin for Lines {
    const METHOD: BuiltinMethod = BuiltinMethod::Lines;
    const NAME: &'static str = "lines";
    fn spec() -> BuiltinSpec { expand_element_spec() }
    #[inline]
    fn apply_one(recv: &crate::data::value::Val) -> Option<crate::data::value::Val> {
        Some(super::lines_apply(recv).unwrap_or_else(|| recv.clone()))
    }
}

/// `words` — whitespace-tokenise string.
pub(crate) struct Words;
impl Builtin for Words {
    const METHOD: BuiltinMethod = BuiltinMethod::Words;
    const NAME: &'static str = "words";
    fn spec() -> BuiltinSpec { expand_element_spec() }
    #[inline]
    fn apply_one(recv: &crate::data::value::Val) -> Option<crate::data::value::Val> {
        Some(super::words_apply(recv).unwrap_or_else(|| recv.clone()))
    }
}

/// `chars` — string char iterator.
pub(crate) struct Chars;
impl Builtin for Chars {
    const METHOD: BuiltinMethod = BuiltinMethod::Chars;
    const NAME: &'static str = "chars";
    fn spec() -> BuiltinSpec { expand_element_spec() }
    #[inline]
    fn apply_one(recv: &crate::data::value::Val) -> Option<crate::data::value::Val> {
        Some(super::chars_apply(recv).unwrap_or_else(|| recv.clone()))
    }
}

/// `chars_of` — chars at given positions.
pub(crate) struct CharsOf;
impl Builtin for CharsOf {
    const METHOD: BuiltinMethod = BuiltinMethod::CharsOf;
    const NAME: &'static str = "chars_of";
    fn spec() -> BuiltinSpec { expand_element_spec() }
    #[inline]
    fn apply_one(recv: &crate::data::value::Val) -> Option<crate::data::value::Val> {
        Some(super::chars_of_apply(recv).unwrap_or_else(|| recv.clone()))
    }
}

/// `bytes` — string byte iterator.
pub(crate) struct Bytes;
impl Builtin for Bytes {
    const METHOD: BuiltinMethod = BuiltinMethod::Bytes;
    const NAME: &'static str = "bytes";
    fn spec() -> BuiltinSpec { expand_element_spec() }
    #[inline]
    fn apply_one(recv: &crate::data::value::Val) -> Option<crate::data::value::Val> {
        Some(super::bytes_of_apply(recv).unwrap_or_else(|| recv.clone()))
    }
}

// ── Find-first / find-one ────────────────────────────────────────────────────

/// `find_first(pred)` — terminal expr-arg returning first match with First demand.
pub(crate) struct FindFirst;
impl Builtin for FindFirst {
    const METHOD: BuiltinMethod = BuiltinMethod::FindFirst;
    const NAME: &'static str = "find_first";
    fn spec() -> BuiltinSpec {
        BuiltinSpec::new(BuiltinCategory::StreamingFilter, BuiltinCardinality::Filtering)
            .cost(10.0)
            .demand_law(BuiltinDemandLaw::FilterLike)
            .lowering(BuiltinPipelineLowering::TerminalExprArg {
                terminal: BuiltinMethod::First,
            })
    }
}

/// `find_one(pred)` — terminal expr-arg without demand annotation.
pub(crate) struct FindOne;
impl Builtin for FindOne {
    const METHOD: BuiltinMethod = BuiltinMethod::FindOne;
    const NAME: &'static str = "find_one";
    fn spec() -> BuiltinSpec {
        BuiltinSpec::new(BuiltinCategory::StreamingFilter, BuiltinCardinality::Filtering)
            .cost(10.0)
            .lowering(BuiltinPipelineLowering::TerminalExprArg {
                terminal: BuiltinMethod::First,
            })
    }
}

// ── Positional miscellaneous ─────────────────────────────────────────────────

#[inline]
fn positional_native_spec() -> BuiltinSpec {
    BuiltinSpec::new(BuiltinCategory::Positional, BuiltinCardinality::Bounded).view_native()
}

/// `nth(i)` — positional select by index.
pub(crate) struct Nth;
impl Builtin for Nth {
    const METHOD: BuiltinMethod = BuiltinMethod::Nth;
    const NAME: &'static str = "nth";
    fn spec() -> BuiltinSpec {
        positional_native_spec()
            .demand_law(BuiltinDemandLaw::Nth)
            .lowering(BuiltinPipelineLowering::TerminalUsizeSink { min: 0 })
    }
    #[inline]
    fn apply_args(recv: &crate::data::value::Val, args: &super::BuiltinArgs) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::I64(n) => { super::nth_any_apply(recv, *n) }
            _ => None,
        }
    }
}

/// `collect()` — materialise stream to Vec; positional pass-through.
pub(crate) struct Collect;
impl Builtin for Collect {
    const METHOD: BuiltinMethod = BuiltinMethod::Collect;
    const NAME: &'static str = "collect";
    fn spec() -> BuiltinSpec { positional_native_spec() }
    #[inline]
    fn apply_one(recv: &crate::data::value::Val) -> Option<crate::data::value::Val> {
        Some(super::collect_apply(recv))
    }
}

// ── Barrier family ───────────────────────────────────────────────────────────

#[inline]
fn barrier_default_spec() -> BuiltinSpec {
    BuiltinSpec::new(BuiltinCategory::Barrier, BuiltinCardinality::Barrier).cost(20.0)
}

/// `sort` — full-barrier comparison sort, optional key.
pub(crate) struct Sort;
impl Builtin for Sort {
    const METHOD: BuiltinMethod = BuiltinMethod::Sort;
    const NAME: &'static str = "sort";
    const ALIASES: &'static [&'static str] = &["sort_by"];
    fn spec() -> BuiltinSpec {
        barrier_default_spec()
            .demand_law(BuiltinDemandLaw::OrderBarrier)
            .materialization(BuiltinPipelineMaterialization::ComposedBarrier)
            .lowering(BuiltinPipelineLowering::Sort)
    }
    #[inline]
    fn apply_barrier(
        ctx: &mut super::builtin::BarrierCtx<'_>,
        buf: &mut Vec<crate::data::value::Val>,
        body: Option<&crate::vm::Program>,
    ) -> Option<Result<(), crate::data::context::EvalError>> {
        let _ = body;
        let _ = ctx;
        let crate::exec::pipeline::Stage::Sort(spec) = ctx.stage else {
            return None;
        };
        let descending = spec.descending;
        let strategy = ctx.strategy;
        let result = match &spec.key {
            None => crate::exec::pipeline::bounded_sort_by_key(
                std::mem::take(buf), descending, strategy, |v| Ok(v.clone()),
            ),
            Some(prog) => {
                let key_prog = prog.clone();
                crate::exec::pipeline::bounded_sort_by_key(
                    std::mem::take(buf), descending, strategy, |v| {
                        Ok(crate::exec::pipeline::eval_kernel(ctx.kernel, v, |item| {
                            crate::exec::pipeline::apply_item_in_env(ctx.vm, ctx.env, item, &key_prog)
                        }).unwrap_or(crate::data::value::Val::Null))
                    },
                )
            }
        };
        match result {
            Ok(sorted) => { *buf = sorted; Some(Ok(())) }
            Err(err) => Some(Err(err)),
        }
    }
}

/// `group_shape` — barrier returning per-shape buckets.
pub(crate) struct GroupShape;
impl Builtin for GroupShape {
    const METHOD: BuiltinMethod = BuiltinMethod::GroupShape;
    const NAME: &'static str = "group_shape";
    fn spec() -> BuiltinSpec { barrier_default_spec() }
}

/// `partition` — splits stream by predicate; barrier.
pub(crate) struct Partition;
impl Builtin for Partition {
    const METHOD: BuiltinMethod = BuiltinMethod::Partition;
    const NAME: &'static str = "partition";
    fn spec() -> BuiltinSpec { barrier_default_spec() }
}

/// `window(n)` — sliding window barrier.
pub(crate) struct Window;
impl Builtin for Window {
    const METHOD: BuiltinMethod = BuiltinMethod::Window;
    const NAME: &'static str = "window";
    fn spec() -> BuiltinSpec {
        barrier_default_spec()
            .materialization(BuiltinPipelineMaterialization::LegacyMaterialized)
            .pipeline_shape(BuiltinPipelineShape::new(
                BuiltinCardinality::Barrier,
                true,
                2.0,
                1.0,
            ))
            .lowering(BuiltinPipelineLowering::UsizeArg { min: 1 })
    }

    #[inline]
    fn apply_barrier(
        ctx: &mut super::builtin::BarrierCtx<'_>,
        buf: &mut Vec<crate::data::value::Val>,
        _body: Option<&crate::vm::Program>,
    ) -> Option<Result<(), crate::data::context::EvalError>> {
        let n = ctx.stage.descriptor().and_then(|d| d.usize_arg)?;
        *buf = super::window_apply(buf, n);
        Some(Ok(()))
    }
}

/// `chunk(n)` — non-overlapping fixed-size buckets.
pub(crate) struct Chunk;
impl Builtin for Chunk {
    const METHOD: BuiltinMethod = BuiltinMethod::Chunk;
    const NAME: &'static str = "chunk";
    const ALIASES: &'static [&'static str] = &["batch"];
    fn spec() -> BuiltinSpec {
        barrier_default_spec()
            .materialization(BuiltinPipelineMaterialization::LegacyMaterialized)
            .pipeline_shape(BuiltinPipelineShape::new(
                BuiltinCardinality::Barrier,
                true,
                2.0,
                1.0,
            ))
            .lowering(BuiltinPipelineLowering::UsizeArg { min: 1 })
    }

    #[inline]
    fn apply_barrier(
        ctx: &mut super::builtin::BarrierCtx<'_>,
        buf: &mut Vec<crate::data::value::Val>,
        _body: Option<&crate::vm::Program>,
    ) -> Option<Result<(), crate::data::context::EvalError>> {
        let n = ctx.stage.descriptor().and_then(|d| d.usize_arg)?;
        *buf = super::chunk_apply(buf, n);
        Some(Ok(()))
    }
}

/// `rolling_sum(n)` — windowed sum barrier.
pub(crate) struct RollingSum;
impl Builtin for RollingSum {
    const METHOD: BuiltinMethod = BuiltinMethod::RollingSum;
    const NAME: &'static str = "rolling_sum";
    fn spec() -> BuiltinSpec { barrier_default_spec() }
    #[inline]
    fn apply_args(recv: &crate::data::value::Val, args: &super::BuiltinArgs) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Usize(n) => { super::rolling_sum_apply(recv, *n) }
            _ => None,
        }
    }
}

/// `rolling_avg(n)` — windowed mean barrier.
pub(crate) struct RollingAvg;
impl Builtin for RollingAvg {
    const METHOD: BuiltinMethod = BuiltinMethod::RollingAvg;
    const NAME: &'static str = "rolling_avg";
    fn spec() -> BuiltinSpec { barrier_default_spec() }
    #[inline]
    fn apply_args(recv: &crate::data::value::Val, args: &super::BuiltinArgs) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Usize(n) => { super::rolling_avg_apply(recv, *n) }
            _ => None,
        }
    }
}

/// `rolling_min(n)` — windowed min barrier.
pub(crate) struct RollingMin;
impl Builtin for RollingMin {
    const METHOD: BuiltinMethod = BuiltinMethod::RollingMin;
    const NAME: &'static str = "rolling_min";
    fn spec() -> BuiltinSpec { barrier_default_spec() }
    #[inline]
    fn apply_args(recv: &crate::data::value::Val, args: &super::BuiltinArgs) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Usize(n) => { super::rolling_min_apply(recv, *n) }
            _ => None,
        }
    }
}

/// `rolling_max(n)` — windowed max barrier.
pub(crate) struct RollingMax;
impl Builtin for RollingMax {
    const METHOD: BuiltinMethod = BuiltinMethod::RollingMax;
    const NAME: &'static str = "rolling_max";
    fn spec() -> BuiltinSpec { barrier_default_spec() }
    #[inline]
    fn apply_args(recv: &crate::data::value::Val, args: &super::BuiltinArgs) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Usize(n) => { super::rolling_max_apply(recv, *n) }
            _ => None,
        }
    }
}

/// `accumulate` — running fold barrier.
pub(crate) struct Accumulate;
impl Builtin for Accumulate {
    const METHOD: BuiltinMethod = BuiltinMethod::Accumulate;
    const NAME: &'static str = "accumulate";
    fn spec() -> BuiltinSpec { barrier_default_spec() }
}

// ── Keyed reducers ───────────────────────────────────────────────────────────

/// `group_by(key)` — keyed reducer collecting elements per key.
pub(crate) struct GroupBy;
impl Builtin for GroupBy {
    const METHOD: BuiltinMethod = BuiltinMethod::GroupBy;
    const NAME: &'static str = "group_by";
    fn spec() -> BuiltinSpec {
        BuiltinSpec::new(BuiltinCategory::Reducer, BuiltinCardinality::Reducing)
            .view_stage(BuiltinViewStage::KeyedReduce)
            .keyed_reducer(BuiltinKeyedReducer::Group)
            .columnar_stage(BuiltinColumnarStage::GroupBy)
            .cost(20.0)
            .demand_law(BuiltinDemandLaw::KeyedReducer)
            .materialization(BuiltinPipelineMaterialization::ComposedBarrier)
            .lowering(BuiltinPipelineLowering::ExprArg)
    }
    #[inline]
    fn apply_barrier(
        ctx: &mut super::builtin::BarrierCtx<'_>,
        buf: &mut Vec<crate::data::value::Val>,
        body: Option<&crate::vm::Program>,
    ) -> Option<Result<(), crate::data::context::EvalError>> {
        let _ = body;
        let _ = ctx;
        let prog = match body {
            Some(p) => p,
            None => return Some(Ok(())),
        };
        let result = super::group_by_apply(std::mem::take(buf), |v| {
            crate::exec::pipeline::eval_kernel(ctx.kernel, v, |item| {
                crate::exec::pipeline::apply_item_in_env(ctx.vm, ctx.env, item, prog)
            })
        });
        match result {
            Ok(out_obj) => {
                *buf = vec![crate::data::value::Val::Obj(std::sync::Arc::new(out_obj))];
                Some(Ok(()))
            }
            Err(err) => Some(Err(err)),
        }
    }
}

/// `count_by(key)` — keyed reducer counting per key.
pub(crate) struct CountBy;
impl Builtin for CountBy {
    const METHOD: BuiltinMethod = BuiltinMethod::CountBy;
    const NAME: &'static str = "count_by";
    fn spec() -> BuiltinSpec {
        BuiltinSpec::new(BuiltinCategory::Reducer, BuiltinCardinality::Reducing)
            .view_stage(BuiltinViewStage::KeyedReduce)
            .keyed_reducer(BuiltinKeyedReducer::Count)
            .cost(10.0)
            .demand_law(BuiltinDemandLaw::KeyedReducer)
            .pipeline_shape(BuiltinPipelineShape::new(
                BuiltinCardinality::OneToOne,
                true,
                1.0,
                1.0,
            ))
            .lowering(BuiltinPipelineLowering::TerminalExprArg {
                terminal: BuiltinMethod::First,
            })
    }

    #[inline]
    fn apply_barrier(
        ctx: &mut super::builtin::BarrierCtx<'_>,
        buf: &mut Vec<crate::data::value::Val>,
        body: Option<&crate::vm::Program>,
    ) -> Option<Result<(), crate::data::context::EvalError>> {
        let prog = body?;
        let result = super::count_by_apply(std::mem::take(buf), |v| {
            crate::exec::pipeline::eval_kernel(ctx.kernel, v, |item| {
                crate::exec::pipeline::apply_item_in_env(ctx.vm, ctx.env, item, prog)
            })
        });
        match result {
            Ok(map) => {
                *buf = vec![crate::data::value::Val::obj(map)];
                Some(Ok(()))
            }
            Err(err) => Some(Err(err)),
        }
    }
}

/// `index_by(key)` — keyed reducer with last-write-wins.
pub(crate) struct IndexBy;
impl Builtin for IndexBy {
    const METHOD: BuiltinMethod = BuiltinMethod::IndexBy;
    const NAME: &'static str = "index_by";
    fn spec() -> BuiltinSpec {
        BuiltinSpec::new(BuiltinCategory::Reducer, BuiltinCardinality::Reducing)
            .view_stage(BuiltinViewStage::KeyedReduce)
            .keyed_reducer(BuiltinKeyedReducer::Index)
            .cost(10.0)
            .demand_law(BuiltinDemandLaw::KeyedReducer)
            .pipeline_shape(BuiltinPipelineShape::new(
                BuiltinCardinality::OneToOne,
                true,
                1.0,
                1.0,
            ))
            .lowering(BuiltinPipelineLowering::TerminalExprArg {
                terminal: BuiltinMethod::First,
            })
    }

    #[inline]
    fn apply_barrier(
        ctx: &mut super::builtin::BarrierCtx<'_>,
        buf: &mut Vec<crate::data::value::Val>,
        body: Option<&crate::vm::Program>,
    ) -> Option<Result<(), crate::data::context::EvalError>> {
        let prog = body?;
        let result = super::index_by_apply(std::mem::take(buf), |v| {
            crate::exec::pipeline::eval_kernel(ctx.kernel, v, |item| {
                crate::exec::pipeline::apply_item_in_env(ctx.vm, ctx.env, item, prog)
            })
        });
        match result {
            Ok(map) => {
                *buf = vec![crate::data::value::Val::obj(map)];
                Some(Ok(()))
            }
            Err(err) => Some(Err(err)),
        }
    }
}

// ── Distinct / unique ────────────────────────────────────────────────────────

#[inline]
fn unique_spec() -> BuiltinSpec {
    BuiltinSpec::new(BuiltinCategory::StreamingFilter, BuiltinCardinality::Filtering)
        .view_stage(BuiltinViewStage::Distinct)
        .cost(10.0)
        .demand_law(BuiltinDemandLaw::UniqueLike)
        .pipeline_shape(BuiltinPipelineShape::new(
            BuiltinCardinality::Filtering,
            true,
            10.0,
            1.0,
        ))
        .order_effect(BuiltinPipelineOrderEffect::Preserves)
        .materialization(BuiltinPipelineMaterialization::LegacyMaterialized)
}

/// Shared barrier body for Unique / UniqueBy.
#[inline]
fn unique_apply_barrier(
    ctx: &mut super::builtin::BarrierCtx<'_>,
    buf: &mut Vec<crate::data::value::Val>,
    body: Option<&crate::vm::Program>,
) -> Option<Result<(), crate::data::context::EvalError>> {
    match body {
        None => {
            let mut seen: std::collections::HashSet<String> = Default::default();
            buf.retain(|v| seen.insert(format!("{:?}", v)));
        }
        Some(prog) => {
            let mut seen: std::collections::HashSet<String> = Default::default();
            let mut keep: Vec<bool> = Vec::with_capacity(buf.len());
            for v in buf.iter() {
                let key = crate::exec::pipeline::eval_kernel(ctx.kernel, v, |item| {
                    crate::exec::pipeline::apply_item_in_env(ctx.vm, ctx.env, item, prog)
                })
                .unwrap_or(crate::data::value::Val::Null);
                keep.push(seen.insert(format!("{:?}", key)));
            }
            let mut out: Vec<crate::data::value::Val> = Vec::with_capacity(buf.len());
            for (i, v) in std::mem::take(buf).into_iter().enumerate() {
                if keep[i] {
                    out.push(v);
                }
            }
            *buf = out;
        }
    }
    Some(Ok(()))
}

/// `unique` — argument-free distinct.
pub(crate) struct Unique;
impl Builtin for Unique {
    const METHOD: BuiltinMethod = BuiltinMethod::Unique;
    const NAME: &'static str = "unique";
    const ALIASES: &'static [&'static str] = &["distinct"];
    fn spec() -> BuiltinSpec {
        unique_spec().lowering(BuiltinPipelineLowering::Nullary)
    }
    #[inline]
    fn apply_one(recv: &crate::data::value::Val) -> Option<crate::data::value::Val> {
        Some(super::unique_arr_apply(recv).unwrap_or_else(|| recv.clone()))
    }
    #[inline]
    fn apply_barrier(
        ctx: &mut super::builtin::BarrierCtx<'_>,
        buf: &mut Vec<crate::data::value::Val>,
        body: Option<&crate::vm::Program>,
    ) -> Option<Result<(), crate::data::context::EvalError>> {
        unique_apply_barrier(ctx, buf, body)
    }
}

/// `unique_by(key)` — distinct by projected key.
pub(crate) struct UniqueBy;
impl Builtin for UniqueBy {
    const METHOD: BuiltinMethod = BuiltinMethod::UniqueBy;
    const NAME: &'static str = "unique_by";
    fn spec() -> BuiltinSpec {
        unique_spec().lowering(BuiltinPipelineLowering::ExprArg)
    }
    #[inline]
    fn apply_barrier(
        ctx: &mut super::builtin::BarrierCtx<'_>,
        buf: &mut Vec<crate::data::value::Val>,
        body: Option<&crate::vm::Program>,
    ) -> Option<Result<(), crate::data::context::EvalError>> {
        unique_apply_barrier(ctx, buf, body)
    }
}

// ── Reverse ──────────────────────────────────────────────────────────────────

/// `reverse` — full-barrier order reversal; cancels with adjacent reverse.
pub(crate) struct Reverse;
impl Builtin for Reverse {
    const METHOD: BuiltinMethod = BuiltinMethod::Reverse;
    const NAME: &'static str = "reverse";
    fn spec() -> BuiltinSpec {
        BuiltinSpec::new(BuiltinCategory::Barrier, BuiltinCardinality::Barrier)
            .cost(10.0)
            .cancellation(BuiltinCancellation::SelfInverse(BuiltinCancelGroup::Reverse))
            .demand_law(BuiltinDemandLaw::OrderBarrier)
            .materialization(BuiltinPipelineMaterialization::ComposedBarrier)
            .lowering(BuiltinPipelineLowering::Nullary)
    }
    #[inline]
    fn apply_one(recv: &crate::data::value::Val) -> Option<crate::data::value::Val> {
        Some(super::reverse_any_apply(recv).unwrap_or_else(|| recv.clone()))
    }

    #[inline]
    fn apply_barrier(
        _ctx: &mut super::builtin::BarrierCtx<'_>,
        buf: &mut Vec<crate::data::value::Val>,
        _body: Option<&crate::vm::Program>,
    ) -> Option<Result<(), crate::data::context::EvalError>> {
        buf.reverse();
        Some(Ok(()))
    }
}

// ── Set / array combiners (barriers, no extra metadata) ──────────────────────

#[inline]
fn barrier_simple_spec() -> BuiltinSpec {
    BuiltinSpec::new(BuiltinCategory::Barrier, BuiltinCardinality::Barrier).cost(10.0)
}

/// `append(arr)` — concatenates barrier.
pub(crate) struct Append;
impl Builtin for Append {
    const METHOD: BuiltinMethod = BuiltinMethod::Append;
    const NAME: &'static str = "append";
    fn spec() -> BuiltinSpec { barrier_simple_spec() }
    #[inline]
    fn apply_args(recv: &crate::data::value::Val, args: &super::BuiltinArgs) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Val(item) => Some(super::append_apply(recv, item).unwrap_or_else(|| recv.clone())),
            _ => None,
        }
    }
}

/// `prepend(arr)` — prepend barrier.
pub(crate) struct Prepend;
impl Builtin for Prepend {
    const METHOD: BuiltinMethod = BuiltinMethod::Prepend;
    const NAME: &'static str = "prepend";
    fn spec() -> BuiltinSpec { barrier_simple_spec() }
    #[inline]
    fn apply_args(recv: &crate::data::value::Val, args: &super::BuiltinArgs) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Val(item) => Some(super::prepend_apply(recv, item).unwrap_or_else(|| recv.clone())),
            _ => None,
        }
    }
}

/// `diff(arr)` — set difference.
pub(crate) struct Diff;
impl Builtin for Diff {
    const METHOD: BuiltinMethod = BuiltinMethod::Diff;
    const NAME: &'static str = "diff";
    fn spec() -> BuiltinSpec { barrier_simple_spec() }
    #[inline]
    fn apply_args(recv: &crate::data::value::Val, args: &super::BuiltinArgs) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::ValVec(other) => { let arr_recv = recv.clone().into_vec().map(crate::data::value::Val::arr)?; super::diff_apply(&arr_recv, other) }
            _ => None,
        }
    }
}

/// `intersect(arr)` — set intersection.
pub(crate) struct Intersect;
impl Builtin for Intersect {
    const METHOD: BuiltinMethod = BuiltinMethod::Intersect;
    const NAME: &'static str = "intersect";
    fn spec() -> BuiltinSpec { barrier_simple_spec() }
    #[inline]
    fn apply_args(recv: &crate::data::value::Val, args: &super::BuiltinArgs) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::ValVec(other) => { let arr_recv = recv.clone().into_vec().map(crate::data::value::Val::arr)?; super::intersect_apply(&arr_recv, other) }
            _ => None,
        }
    }
}

/// `union(arr)` — set union.
pub(crate) struct Union;
impl Builtin for Union {
    const METHOD: BuiltinMethod = BuiltinMethod::Union;
    const NAME: &'static str = "union";
    fn spec() -> BuiltinSpec { barrier_simple_spec() }
    #[inline]
    fn apply_args(recv: &crate::data::value::Val, args: &super::BuiltinArgs) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::ValVec(other) => { let arr_recv = recv.clone().into_vec().map(crate::data::value::Val::arr)?; super::union_apply(&arr_recv, other) }
            _ => None,
        }
    }
}

/// `join(sep)` — string join barrier.
pub(crate) struct Join;
impl Builtin for Join {
    const METHOD: BuiltinMethod = BuiltinMethod::Join;
    const NAME: &'static str = "join";
    fn spec() -> BuiltinSpec { barrier_simple_spec() }
    #[inline]
    fn apply_args(recv: &crate::data::value::Val, args: &super::BuiltinArgs) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Str(sep) => Some(super::join_apply(recv, sep).unwrap_or_else(|| recv.clone())),
            _ => None,
        }
    }
}

/// `zip(arr)` — element pairing.
pub(crate) struct Zip;
impl Builtin for Zip {
    const METHOD: BuiltinMethod = BuiltinMethod::Zip;
    const NAME: &'static str = "zip";
    fn spec() -> BuiltinSpec { barrier_simple_spec() }
}

/// `zip_longest(arr)` — pad-shorter zip.
pub(crate) struct ZipLongest;
impl Builtin for ZipLongest {
    const METHOD: BuiltinMethod = BuiltinMethod::ZipLongest;
    const NAME: &'static str = "zip_longest";
    fn spec() -> BuiltinSpec { barrier_simple_spec() }
}

/// `fanout(...)` — multi-projection.
pub(crate) struct Fanout;
impl Builtin for Fanout {
    const METHOD: BuiltinMethod = BuiltinMethod::Fanout;
    const NAME: &'static str = "fanout";
    fn spec() -> BuiltinSpec { barrier_simple_spec() }
}

/// `zip_shape(...)` — shape-preserving zip.
pub(crate) struct ZipShape;
impl Builtin for ZipShape {
    const METHOD: BuiltinMethod = BuiltinMethod::ZipShape;
    const NAME: &'static str = "zip_shape";
    fn spec() -> BuiltinSpec { barrier_simple_spec() }
}

// ── Object operations ────────────────────────────────────────────────────────

#[inline]
fn object_element_spec() -> BuiltinSpec {
    BuiltinSpec::new(BuiltinCategory::Object, BuiltinCardinality::OneToOne).element()
}

/// `keys` — extract keys of an object (element-wise).
pub(crate) struct Keys;
impl Builtin for Keys {
    const METHOD: BuiltinMethod = BuiltinMethod::Keys;
    const NAME: &'static str = "keys";
    fn spec() -> BuiltinSpec { object_element_spec() }
    #[inline]
    fn apply_one(recv: &crate::data::value::Val) -> Option<crate::data::value::Val> {
        Some(super::keys_apply(recv))
    }
}

/// `values` — extract values of an object (element-wise).
pub(crate) struct Values;
impl Builtin for Values {
    const METHOD: BuiltinMethod = BuiltinMethod::Values;
    const NAME: &'static str = "values";
    fn spec() -> BuiltinSpec { object_element_spec() }
    #[inline]
    fn apply_one(recv: &crate::data::value::Val) -> Option<crate::data::value::Val> {
        Some(super::values_apply(recv))
    }
}

/// `entries` — extract (key, value) pairs (element-wise).
pub(crate) struct Entries;
impl Builtin for Entries {
    const METHOD: BuiltinMethod = BuiltinMethod::Entries;
    const NAME: &'static str = "entries";
    fn spec() -> BuiltinSpec { object_element_spec() }
    #[inline]
    fn apply_one(recv: &crate::data::value::Val) -> Option<crate::data::value::Val> {
        Some(super::entries_apply(recv))
    }
}

#[inline]
fn object_simple_spec() -> BuiltinSpec {
    BuiltinSpec::new(BuiltinCategory::Object, BuiltinCardinality::OneToOne)
}

/// `to_pairs` — convert object to array of `[k, v]` pairs.
pub(crate) struct ToPairs;
impl Builtin for ToPairs {
    const METHOD: BuiltinMethod = BuiltinMethod::ToPairs;
    const NAME: &'static str = "to_pairs";
    fn spec() -> BuiltinSpec { object_simple_spec() }
    #[inline]
    fn apply_one(recv: &crate::data::value::Val) -> Option<crate::data::value::Val> {
        Some(super::to_pairs_apply(recv).unwrap_or_else(|| recv.clone()))
    }
}

/// `from_pairs` — invert `to_pairs`.
pub(crate) struct FromPairs;
impl Builtin for FromPairs {
    const METHOD: BuiltinMethod = BuiltinMethod::FromPairs;
    const NAME: &'static str = "from_pairs";
    fn spec() -> BuiltinSpec { object_simple_spec() }
    #[inline]
    fn apply_one(recv: &crate::data::value::Val) -> Option<crate::data::value::Val> {
        Some(super::from_pairs_apply(recv).unwrap_or_else(|| recv.clone()))
    }
}

/// `invert` — swap keys and values.
pub(crate) struct Invert;
impl Builtin for Invert {
    const METHOD: BuiltinMethod = BuiltinMethod::Invert;
    const NAME: &'static str = "invert";
    fn spec() -> BuiltinSpec { object_simple_spec() }
    #[inline]
    fn apply_one(recv: &crate::data::value::Val) -> Option<crate::data::value::Val> {
        Some(super::invert_apply(recv).unwrap_or_else(|| recv.clone()))
    }
}

/// `pick(...keys)` — restrict object to given keys.
pub(crate) struct Pick;
impl Builtin for Pick {
    const METHOD: BuiltinMethod = BuiltinMethod::Pick;
    const NAME: &'static str = "pick";
    fn spec() -> BuiltinSpec { object_simple_spec() }
    #[inline]
    fn apply_args(recv: &crate::data::value::Val, args: &super::BuiltinArgs) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::StrVec(keys) => { super::pick_apply(recv, keys) }
            _ => None,
        }
    }
}

/// `omit(...keys)` — drop given keys from object.
pub(crate) struct Omit;
impl Builtin for Omit {
    const METHOD: BuiltinMethod = BuiltinMethod::Omit;
    const NAME: &'static str = "omit";
    fn spec() -> BuiltinSpec { object_simple_spec() }
    #[inline]
    fn apply_args(recv: &crate::data::value::Val, args: &super::BuiltinArgs) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::StrVec(keys) => { super::omit_apply(recv, keys) }
            _ => None,
        }
    }
}

/// `merge(...objs)` — shallow merge objects.
pub(crate) struct Merge;
impl Builtin for Merge {
    const METHOD: BuiltinMethod = BuiltinMethod::Merge;
    const NAME: &'static str = "merge";
    fn spec() -> BuiltinSpec { object_simple_spec() }
    #[inline]
    fn apply_args(recv: &crate::data::value::Val, args: &super::BuiltinArgs) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Val(other) => { super::merge_apply(recv, other) }
            _ => None,
        }
    }
}

/// `deep_merge(...objs)` — recursive merge.
pub(crate) struct DeepMerge;
impl Builtin for DeepMerge {
    const METHOD: BuiltinMethod = BuiltinMethod::DeepMerge;
    const NAME: &'static str = "deep_merge";
    fn spec() -> BuiltinSpec { object_simple_spec() }
    #[inline]
    fn apply_args(recv: &crate::data::value::Val, args: &super::BuiltinArgs) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Val(other) => { super::deep_merge_apply(recv, other) }
            _ => None,
        }
    }
}

/// `defaults(...objs)` — fill-in defaults without overwriting.
pub(crate) struct Defaults;
impl Builtin for Defaults {
    const METHOD: BuiltinMethod = BuiltinMethod::Defaults;
    const NAME: &'static str = "defaults";
    fn spec() -> BuiltinSpec { object_simple_spec() }
    #[inline]
    fn apply_args(recv: &crate::data::value::Val, args: &super::BuiltinArgs) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Val(other) => { super::defaults_apply(recv, other) }
            _ => None,
        }
    }
}

/// `rename({...})` — rename object keys.
pub(crate) struct Rename;
impl Builtin for Rename {
    const METHOD: BuiltinMethod = BuiltinMethod::Rename;
    const NAME: &'static str = "rename";
    fn spec() -> BuiltinSpec { object_simple_spec() }
    #[inline]
    fn apply_args(recv: &crate::data::value::Val, args: &super::BuiltinArgs) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Val(other) => { super::rename_apply(recv, other) }
            _ => None,
        }
    }
}

/// `pivot(...)` — reshape object axes.
pub(crate) struct Pivot;
impl Builtin for Pivot {
    const METHOD: BuiltinMethod = BuiltinMethod::Pivot;
    const NAME: &'static str = "pivot";
    fn spec() -> BuiltinSpec { object_simple_spec() }
}

/// `implode(sep)` — array-to-string with separator.
pub(crate) struct Implode;
impl Builtin for Implode {
    const METHOD: BuiltinMethod = BuiltinMethod::Implode;
    const NAME: &'static str = "implode";
    fn spec() -> BuiltinSpec { object_simple_spec() }
    #[inline]
    fn apply_args(recv: &crate::data::value::Val, args: &super::BuiltinArgs) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Str(field) => { super::implode_apply(recv, field) }
            _ => None,
        }
    }
}

#[inline]
fn object_lambda_spec() -> BuiltinSpec {
    BuiltinSpec::new(BuiltinCategory::Object, BuiltinCardinality::OneToOne)
        .pipeline_shape(BuiltinPipelineShape::new(
            BuiltinCardinality::OneToOne,
            true,
            1.0,
            1.0,
        ))
        .order_effect(BuiltinPipelineOrderEffect::Preserves)
        .lowering(BuiltinPipelineLowering::ExprArg)
}

/// `transform_keys(lam)` — map over keys of an object.
pub(crate) struct TransformKeys;
impl Builtin for TransformKeys {
    const METHOD: BuiltinMethod = BuiltinMethod::TransformKeys;
    const NAME: &'static str = "transform_keys";
    fn spec() -> BuiltinSpec { object_lambda_spec() }

    #[inline]
    fn apply_stream(
        ctx: &mut super::builtin::StreamCtx<'_, '_>,
        item: crate::data::value::Val,
        body: Option<&crate::vm::Program>,
    ) -> Result<crate::exec::pipeline::StageFlow<crate::data::value::Val>, crate::data::context::EvalError> {
        object_lambda_apply_stream(ctx, item, body)
    }

    #[inline]
    fn apply_barrier(
        ctx: &mut super::builtin::BarrierCtx<'_>,
        buf: &mut Vec<crate::data::value::Val>,
        body: Option<&crate::vm::Program>,
    ) -> Option<Result<(), crate::data::context::EvalError>> {
        object_lambda_apply_barrier(ctx, buf, body)
    }
}

/// Helper used by all ObjectLambda variants — single body shared across
/// TransformKeys / TransformValues / FilterKeys / FilterValues.
#[inline]
fn object_lambda_apply_stream(
    ctx: &mut super::builtin::StreamCtx<'_, '_>,
    item: crate::data::value::Val,
    body: Option<&crate::vm::Program>,
) -> Result<crate::exec::pipeline::StageFlow<crate::data::value::Val>, crate::data::context::EvalError> {
    let prog = body.expect("object lambda body");
    let result = crate::exec::pipeline::materialized_exec::apply_lambda_obj(
        ctx.stage, &item, ctx.vm, ctx.env, ctx.kernel, prog,
    )?;
    Ok(crate::exec::pipeline::StageFlow::Continue(result))
}

/// Helper used by all ObjectLambda variants for barrier (whole-buffer) execution.
#[inline]
fn object_lambda_apply_barrier(
    ctx: &mut super::builtin::BarrierCtx<'_>,
    buf: &mut Vec<crate::data::value::Val>,
    body: Option<&crate::vm::Program>,
) -> Option<Result<(), crate::data::context::EvalError>> {
    let prog = body?;
    let mut out: Vec<crate::data::value::Val> = Vec::with_capacity(buf.len());
    for v in std::mem::take(buf) {
        match crate::exec::pipeline::materialized_exec::apply_lambda_obj(
            ctx.stage, &v, ctx.vm, ctx.env, ctx.kernel, prog,
        ) {
            Ok(mapped) => out.push(mapped),
            Err(err) => {
                *buf = out;
                return Some(Err(err));
            }
        }
    }
    *buf = out;
    Some(Ok(()))
}

/// `transform_values(lam)` — map over values of an object.
pub(crate) struct TransformValues;
impl Builtin for TransformValues {
    const METHOD: BuiltinMethod = BuiltinMethod::TransformValues;
    const NAME: &'static str = "transform_values";
    fn spec() -> BuiltinSpec { object_lambda_spec() }
    #[inline]
    fn apply_stream(
        ctx: &mut super::builtin::StreamCtx<'_, '_>,
        item: crate::data::value::Val,
        body: Option<&crate::vm::Program>,
    ) -> Result<crate::exec::pipeline::StageFlow<crate::data::value::Val>, crate::data::context::EvalError> {
        object_lambda_apply_stream(ctx, item, body)
    }

    #[inline]
    fn apply_barrier(
        ctx: &mut super::builtin::BarrierCtx<'_>,
        buf: &mut Vec<crate::data::value::Val>,
        body: Option<&crate::vm::Program>,
    ) -> Option<Result<(), crate::data::context::EvalError>> {
        object_lambda_apply_barrier(ctx, buf, body)
    }
}

/// `filter_keys(pred)` — drop entries by key predicate.
pub(crate) struct FilterKeys;
impl Builtin for FilterKeys {
    const METHOD: BuiltinMethod = BuiltinMethod::FilterKeys;
    const NAME: &'static str = "filter_keys";
    fn spec() -> BuiltinSpec { object_lambda_spec() }
    #[inline]
    fn apply_stream(
        ctx: &mut super::builtin::StreamCtx<'_, '_>,
        item: crate::data::value::Val,
        body: Option<&crate::vm::Program>,
    ) -> Result<crate::exec::pipeline::StageFlow<crate::data::value::Val>, crate::data::context::EvalError> {
        object_lambda_apply_stream(ctx, item, body)
    }

    #[inline]
    fn apply_barrier(
        ctx: &mut super::builtin::BarrierCtx<'_>,
        buf: &mut Vec<crate::data::value::Val>,
        body: Option<&crate::vm::Program>,
    ) -> Option<Result<(), crate::data::context::EvalError>> {
        object_lambda_apply_barrier(ctx, buf, body)
    }
}

/// `filter_values(pred)` — drop entries by value predicate.
pub(crate) struct FilterValues;
impl Builtin for FilterValues {
    const METHOD: BuiltinMethod = BuiltinMethod::FilterValues;
    const NAME: &'static str = "filter_values";
    fn spec() -> BuiltinSpec { object_lambda_spec() }
    #[inline]
    fn apply_stream(
        ctx: &mut super::builtin::StreamCtx<'_, '_>,
        item: crate::data::value::Val,
        body: Option<&crate::vm::Program>,
    ) -> Result<crate::exec::pipeline::StageFlow<crate::data::value::Val>, crate::data::context::EvalError> {
        object_lambda_apply_stream(ctx, item, body)
    }

    #[inline]
    fn apply_barrier(
        ctx: &mut super::builtin::BarrierCtx<'_>,
        buf: &mut Vec<crate::data::value::Val>,
        body: Option<&crate::vm::Program>,
    ) -> Option<Result<(), crate::data::context::EvalError>> {
        object_lambda_apply_barrier(ctx, buf, body)
    }
}

// ── Path operations ──────────────────────────────────────────────────────────

#[inline]
fn path_element_spec() -> BuiltinSpec {
    BuiltinSpec::new(BuiltinCategory::Path, BuiltinCardinality::OneToOne)
        .indexed()
        .element()
}

/// `get_path(path)` — navigate path lookup.
pub(crate) struct GetPath;
impl Builtin for GetPath {
    const METHOD: BuiltinMethod = BuiltinMethod::GetPath;
    const NAME: &'static str = "get_path";
    fn spec() -> BuiltinSpec { path_element_spec() }
    #[inline]
    fn apply_args(recv: &crate::data::value::Val, args: &super::BuiltinArgs) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Str(p) => { super::get_path_apply(recv, p) }
            _ => None,
        }
    }
}

/// `del_path(path)` — remove value at path.
pub(crate) struct DelPath;
impl Builtin for DelPath {
    const METHOD: BuiltinMethod = BuiltinMethod::DelPath;
    const NAME: &'static str = "del_path";
    fn spec() -> BuiltinSpec { path_element_spec() }
    #[inline]
    fn apply_args(recv: &crate::data::value::Val, args: &super::BuiltinArgs) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Str(p) => { super::del_path_apply(recv, p) }
            _ => None,
        }
    }
}

/// `has_path(path)` — existence test.
pub(crate) struct HasPath;
impl Builtin for HasPath {
    const METHOD: BuiltinMethod = BuiltinMethod::HasPath;
    const NAME: &'static str = "has_path";
    fn spec() -> BuiltinSpec { path_element_spec() }
    #[inline]
    fn apply_args(recv: &crate::data::value::Val, args: &super::BuiltinArgs) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Str(p) => { super::has_path_apply(recv, p) }
            _ => None,
        }
    }
}

#[inline]
fn path_indexed_spec() -> BuiltinSpec {
    BuiltinSpec::new(BuiltinCategory::Path, BuiltinCardinality::OneToOne).indexed()
}

/// `set_path(path, val)` — write value at path.
pub(crate) struct SetPath;
impl Builtin for SetPath {
    const METHOD: BuiltinMethod = BuiltinMethod::SetPath;
    const NAME: &'static str = "set_path";
    fn spec() -> BuiltinSpec { path_indexed_spec() }
}

/// `del_paths([...])` — bulk path removal.
pub(crate) struct DelPaths;
impl Builtin for DelPaths {
    const METHOD: BuiltinMethod = BuiltinMethod::DelPaths;
    const NAME: &'static str = "del_paths";
    fn spec() -> BuiltinSpec { path_indexed_spec() }
}

/// `flatten_keys` — flatten nested object into dotted keys.
pub(crate) struct FlattenKeys;
impl Builtin for FlattenKeys {
    const METHOD: BuiltinMethod = BuiltinMethod::FlattenKeys;
    const NAME: &'static str = "flatten_keys";
    fn spec() -> BuiltinSpec { path_indexed_spec() }
    #[inline]
    fn apply_args(recv: &crate::data::value::Val, args: &super::BuiltinArgs) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Str(p) => { super::flatten_keys_apply(recv, p) }
            _ => None,
        }
    }
}

/// `unflatten_keys` — invert `flatten_keys`.
pub(crate) struct UnflattenKeys;
impl Builtin for UnflattenKeys {
    const METHOD: BuiltinMethod = BuiltinMethod::UnflattenKeys;
    const NAME: &'static str = "unflatten_keys";
    fn spec() -> BuiltinSpec { path_indexed_spec() }
    #[inline]
    fn apply_args(recv: &crate::data::value::Val, args: &super::BuiltinArgs) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Str(p) => { super::unflatten_keys_apply(recv, p) }
            _ => None,
        }
    }
}

// ── Deep operations ──────────────────────────────────────────────────────────

#[inline]
fn deep_simple_spec() -> BuiltinSpec {
    BuiltinSpec::new(BuiltinCategory::Deep, BuiltinCardinality::Expanding).cost(20.0)
}

/// `walk(fn)` — post-order walk.
pub(crate) struct Walk;
impl Builtin for Walk {
    const METHOD: BuiltinMethod = BuiltinMethod::Walk;
    const NAME: &'static str = "walk";
    fn spec() -> BuiltinSpec { deep_simple_spec() }
}

/// `walk_pre(fn)` — pre-order walk.
pub(crate) struct WalkPre;
impl Builtin for WalkPre {
    const METHOD: BuiltinMethod = BuiltinMethod::WalkPre;
    const NAME: &'static str = "walk_pre";
    fn spec() -> BuiltinSpec { deep_simple_spec() }
}

/// `rec(fn)` — recursive descent map.
pub(crate) struct Rec;
impl Builtin for Rec {
    const METHOD: BuiltinMethod = BuiltinMethod::Rec;
    const NAME: &'static str = "rec";
    fn spec() -> BuiltinSpec { deep_simple_spec() }
}

/// `trace_path()` — collect all paths.
pub(crate) struct TracePath;
impl Builtin for TracePath {
    const METHOD: BuiltinMethod = BuiltinMethod::TracePath;
    const NAME: &'static str = "trace_path";
    fn spec() -> BuiltinSpec { deep_simple_spec() }
}

/// `deep_find(pred)` — descend and collect all matches.
pub(crate) struct DeepFind;
impl Builtin for DeepFind {
    const METHOD: BuiltinMethod = BuiltinMethod::DeepFind;
    const NAME: &'static str = "deep_find";
    fn spec() -> BuiltinSpec {
        BuiltinSpec::new(BuiltinCategory::Deep, BuiltinCardinality::Expanding)
            .structural(BuiltinStructural::DeepFind)
            .cost(20.0)
    }
}

/// `deep_shape({...})` — descend and collect by shape.
pub(crate) struct DeepShape;
impl Builtin for DeepShape {
    const METHOD: BuiltinMethod = BuiltinMethod::DeepShape;
    const NAME: &'static str = "deep_shape";
    fn spec() -> BuiltinSpec {
        BuiltinSpec::new(BuiltinCategory::Deep, BuiltinCardinality::Expanding)
            .structural(BuiltinStructural::DeepShape)
            .cost(20.0)
    }
}

/// `deep_like({...})` — descend and collect by literal match.
pub(crate) struct DeepLike;
impl Builtin for DeepLike {
    const METHOD: BuiltinMethod = BuiltinMethod::DeepLike;
    const NAME: &'static str = "deep_like";
    fn spec() -> BuiltinSpec {
        BuiltinSpec::new(BuiltinCategory::Deep, BuiltinCardinality::Expanding)
            .structural(BuiltinStructural::DeepLike)
            .cost(20.0)
    }
}

// ── Serialization / relational / mutation ────────────────────────────────────

#[inline]
fn serialization_spec() -> BuiltinSpec {
    BuiltinSpec::new(BuiltinCategory::Serialization, BuiltinCardinality::OneToOne)
        .indexed()
        .cost(20.0)
}

/// `to_csv()` — CSV serialiser.
pub(crate) struct ToCsv;
impl Builtin for ToCsv {
    const METHOD: BuiltinMethod = BuiltinMethod::ToCsv;
    const NAME: &'static str = "to_csv";
    fn spec() -> BuiltinSpec { serialization_spec() }
    #[inline]
    fn apply_one(recv: &crate::data::value::Val) -> Option<crate::data::value::Val> {
        Some(super::to_csv_apply(recv).unwrap_or_else(|| recv.clone()))
    }
}

/// `to_tsv()` — TSV serialiser.
pub(crate) struct ToTsv;
impl Builtin for ToTsv {
    const METHOD: BuiltinMethod = BuiltinMethod::ToTsv;
    const NAME: &'static str = "to_tsv";
    fn spec() -> BuiltinSpec { serialization_spec() }
    #[inline]
    fn apply_one(recv: &crate::data::value::Val) -> Option<crate::data::value::Val> {
        Some(super::to_tsv_apply(recv).unwrap_or_else(|| recv.clone()))
    }
}

/// `equi_join(left, right, on)` — relational join barrier.
pub(crate) struct EquiJoin;
impl Builtin for EquiJoin {
    const METHOD: BuiltinMethod = BuiltinMethod::EquiJoin;
    const NAME: &'static str = "equi_join";
    fn spec() -> BuiltinSpec {
        BuiltinSpec::new(BuiltinCategory::Relational, BuiltinCardinality::Barrier).cost(20.0)
    }
}

/// `set(path, val)` — element-wise mutation.
pub(crate) struct Set;
impl Builtin for Set {
    const METHOD: BuiltinMethod = BuiltinMethod::Set;
    const NAME: &'static str = "set";
    fn spec() -> BuiltinSpec {
        BuiltinSpec::new(BuiltinCategory::Mutation, BuiltinCardinality::OneToOne)
            .indexed()
            .element()
    }
    #[inline]
    fn apply_args(_recv: &crate::data::value::Val, args: &super::BuiltinArgs) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Val(item) => Some(item.clone()),
            _ => None,
        }
    }
}

/// `update(path, fn)` — mutation via lambda.
pub(crate) struct Update;
impl Builtin for Update {
    const METHOD: BuiltinMethod = BuiltinMethod::Update;
    const NAME: &'static str = "update";
    fn spec() -> BuiltinSpec {
        BuiltinSpec::new(BuiltinCategory::Mutation, BuiltinCardinality::OneToOne).indexed()
    }
}

// ── Streaming OneToOne (no lambda, indexed, element) ─────────────────────────

#[inline]
fn streaming_one_to_one_element_spec() -> BuiltinSpec {
    BuiltinSpec::new(BuiltinCategory::StreamingOneToOne, BuiltinCardinality::OneToOne)
        .indexed()
        .cost(10.0)
        .element()
}

/// `lag(n)` — element shifted by N positions.
pub(crate) struct Lag;
impl Builtin for Lag {
    const METHOD: BuiltinMethod = BuiltinMethod::Lag;
    const NAME: &'static str = "lag";
    fn spec() -> BuiltinSpec { streaming_one_to_one_element_spec() }
    #[inline]
    fn apply_args(recv: &crate::data::value::Val, args: &super::BuiltinArgs) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Usize(n) => { super::lag_apply(recv, *n) }
            _ => None,
        }
    }
}

/// `lead(n)` — element shifted forward by N positions.
pub(crate) struct Lead;
impl Builtin for Lead {
    const METHOD: BuiltinMethod = BuiltinMethod::Lead;
    const NAME: &'static str = "lead";
    fn spec() -> BuiltinSpec { streaming_one_to_one_element_spec() }
    #[inline]
    fn apply_args(recv: &crate::data::value::Val, args: &super::BuiltinArgs) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Usize(n) => { super::lead_apply(recv, *n) }
            _ => None,
        }
    }
}

/// `diff_window(n)` — pairwise diff at lag N.
pub(crate) struct DiffWindow;
impl Builtin for DiffWindow {
    const METHOD: BuiltinMethod = BuiltinMethod::DiffWindow;
    const NAME: &'static str = "diff_window";
    fn spec() -> BuiltinSpec { streaming_one_to_one_element_spec() }
    #[inline]
    fn apply_args(recv: &crate::data::value::Val, args: &super::BuiltinArgs) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::None => { Some(super::diff_window_apply(recv).unwrap_or_else(|| recv.clone())) }
            _ => None,
        }
    }
}

/// `pct_change(n)` — pairwise relative change at lag N.
pub(crate) struct PctChange;
impl Builtin for PctChange {
    const METHOD: BuiltinMethod = BuiltinMethod::PctChange;
    const NAME: &'static str = "pct_change";
    fn spec() -> BuiltinSpec { streaming_one_to_one_element_spec() }
    #[inline]
    fn apply_args(recv: &crate::data::value::Val, args: &super::BuiltinArgs) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::None => { Some(super::pct_change_apply(recv).unwrap_or_else(|| recv.clone())) }
            _ => None,
        }
    }
}

/// `cummax()` — running maximum.
pub(crate) struct CumMax;
impl Builtin for CumMax {
    const METHOD: BuiltinMethod = BuiltinMethod::CumMax;
    const NAME: &'static str = "cummax";
    fn spec() -> BuiltinSpec { streaming_one_to_one_element_spec() }
    #[inline]
    fn apply_args(recv: &crate::data::value::Val, args: &super::BuiltinArgs) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::None => { Some(super::cummax_apply(recv).unwrap_or_else(|| recv.clone())) }
            _ => None,
        }
    }
}

/// `cummin()` — running minimum.
pub(crate) struct CumMin;
impl Builtin for CumMin {
    const METHOD: BuiltinMethod = BuiltinMethod::CumMin;
    const NAME: &'static str = "cummin";
    fn spec() -> BuiltinSpec { streaming_one_to_one_element_spec() }
    #[inline]
    fn apply_args(recv: &crate::data::value::Val, args: &super::BuiltinArgs) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::None => { Some(super::cummin_apply(recv).unwrap_or_else(|| recv.clone())) }
            _ => None,
        }
    }
}

/// `zscore()` — element standardised by mean/std.
pub(crate) struct Zscore;
impl Builtin for Zscore {
    const METHOD: BuiltinMethod = BuiltinMethod::Zscore;
    const NAME: &'static str = "zscore";
    fn spec() -> BuiltinSpec { streaming_one_to_one_element_spec() }
    #[inline]
    fn apply_args(recv: &crate::data::value::Val, args: &super::BuiltinArgs) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::None => { Some(super::zscore_apply(recv).unwrap_or_else(|| recv.clone())) }
            _ => None,
        }
    }
}

// ── Scalar element-only (basic) ──────────────────────────────────────────────

#[inline]
fn scalar_native_element_spec() -> BuiltinSpec {
    BuiltinSpec::new(BuiltinCategory::Scalar, BuiltinCardinality::OneToOne)
        .indexed()
        .view_native()
        .element()
}

#[inline]
fn scalar_view_scalar_element_spec() -> BuiltinSpec {
    BuiltinSpec::new(BuiltinCategory::Scalar, BuiltinCardinality::OneToOne)
        .indexed()
        .view_native()
        .view_scalar()
        .element()
}

// Native-element (no view_scalar):
// `apply` clause wraps with recv.clone() fallback so trait dispatch fully owns this method
// (no fall-through to legacy match on type mismatch).
macro_rules! scalar_native_element {
    ( $( $ty:ident => $variant:ident, $name:literal
         $( , aliases: [ $( $alias:literal ),* $(,)? ] )?
         $( , apply: $apply:ident )? ; )* ) => {
        $(
            pub(crate) struct $ty;
            impl Builtin for $ty {
                const METHOD: BuiltinMethod = BuiltinMethod::$variant;
                const NAME: &'static str = $name;
                $( const ALIASES: &'static [&'static str] = &[ $( $alias ),* ]; )?
                fn spec() -> BuiltinSpec { scalar_native_element_spec() }
                $(
                    #[inline]
                    fn apply_one(recv: &crate::data::value::Val) -> Option<crate::data::value::Val> {
                        Some(super::$apply(recv).unwrap_or_else(|| recv.clone()))
                    }
                )?
            }
        )*
    };
}

// View-scalar element:
macro_rules! scalar_view_scalar_element {
    ( $( $ty:ident => $variant:ident, $name:literal
         $( , aliases: [ $( $alias:literal ),* $(,)? ] )?
         $( , apply: $apply:ident )? ; )* ) => {
        $(
            pub(crate) struct $ty;
            impl Builtin for $ty {
                const METHOD: BuiltinMethod = BuiltinMethod::$variant;
                const NAME: &'static str = $name;
                $( const ALIASES: &'static [&'static str] = &[ $( $alias ),* ]; )?
                fn spec() -> BuiltinSpec { scalar_view_scalar_element_spec() }
                $(
                    #[inline]
                    fn apply_one(recv: &crate::data::value::Val) -> Option<crate::data::value::Val> {
                        Some(super::$apply(recv).unwrap_or_else(|| recv.clone()))
                    }
                )?
            }
        )*
    };
}

scalar_native_element! {
    Capitalize => Capitalize, "capitalize", apply: capitalize_apply;
    TitleCase => TitleCase, "title_case", apply: title_case_apply;
    SnakeCase => SnakeCase, "snake_case", apply: snake_case_apply;
    KebabCase => KebabCase, "kebab_case", apply: kebab_case_apply;
    CamelCase => CamelCase, "camel_case", apply: camel_case_apply;
    PascalCase => PascalCase, "pascal_case", apply: pascal_case_apply;
    ParseInt => ParseInt, "parse_int", apply: parse_int_apply;
    ParseFloat => ParseFloat, "parse_float", apply: parse_float_apply;
    ParseBool => ParseBool, "parse_bool", apply: parse_bool_apply;
    Schema => Schema, "schema", apply: schema_apply;
    Type => Type, "type", apply: type_name_apply;
    ToString => ToString, "to_string", apply: to_string_apply;
    ToJson => ToJson, "to_json", apply: to_json_apply;
    Dedent => Dedent, "dedent", apply: dedent_apply;
}

scalar_view_scalar_element! {
    Ceil => Ceil, "ceil";
    Floor => Floor, "floor";
    Round => Round, "round";
    Abs => Abs, "abs";
    Upper => Upper, "upper", apply: upper_apply;
    Lower => Lower, "lower", apply: lower_apply;
    Trim => Trim, "trim", apply: trim_apply;
    TrimLeft => TrimLeft, "trim_left", aliases: ["lstrip"], apply: trim_left_apply;
    TrimRight => TrimRight, "trim_right", aliases: ["rstrip"], apply: trim_right_apply;
    IsBlank => IsBlank, "is_blank";
    IsNumeric => IsNumeric, "is_numeric";
    IsAlpha => IsAlpha, "is_alpha";
    IsAscii => IsAscii, "is_ascii";
    ToNumber => ToNumber, "to_number";
    ToBool => ToBool, "to_bool";
    StartsWith => StartsWith, "starts_with";
    EndsWith => EndsWith, "ends_with";
    IndexOf => IndexOf, "index_of";
    LastIndexOf => LastIndexOf, "last_index_of";
    Matches => Matches, "matches";
    ByteLen => ByteLen, "byte_len";
}

// ── Scalar with pipeline lowerings ───────────────────────────────────────────

/// `slice(start, end?)` — int-range scalar element with pipeline lowering.
pub(crate) struct Slice;
impl Builtin for Slice {
    const METHOD: BuiltinMethod = BuiltinMethod::Slice;
    const NAME: &'static str = "slice";
    fn spec() -> BuiltinSpec {
        BuiltinSpec::new(BuiltinCategory::Scalar, BuiltinCardinality::OneToOne)
            .indexed()
            .view_native()
            .pipeline_shape(BuiltinPipelineShape::new(
                BuiltinCardinality::OneToOne,
                true,
                1.0,
                1.0,
            ))
            .order_effect(BuiltinPipelineOrderEffect::Preserves)
            .lowering(BuiltinPipelineLowering::IntRangeArg)
    }
    #[inline]
    fn apply_args(recv: &crate::data::value::Val, args: &super::BuiltinArgs) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::I64Opt { first, second } => { Some(super::slice_apply(recv.clone(), *first, *second)) }
            _ => None,
        }
    }
}

/// `replace(needle, with)` — single-replace string-pair scalar.
pub(crate) struct Replace;
impl Builtin for Replace {
    const METHOD: BuiltinMethod = BuiltinMethod::Replace;
    const NAME: &'static str = "replace";
    fn spec() -> BuiltinSpec {
        BuiltinSpec::new(BuiltinCategory::Scalar, BuiltinCardinality::OneToOne)
            .indexed()
            .view_native()
            .pipeline_shape(BuiltinPipelineShape::new(
                BuiltinCardinality::OneToOne,
                true,
                2.0,
                1.0,
            ))
            .order_effect(BuiltinPipelineOrderEffect::Preserves)
            .lowering(BuiltinPipelineLowering::StringPairArg)
    }
    #[inline]
    fn apply_args(recv: &crate::data::value::Val, args: &super::BuiltinArgs) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::StrPair { first, second } => { super::replace_apply(recv.clone(), first, second, false) }
            _ => None,
        }
    }
}

/// `replace_all(needle, with)` — replace-all string-pair scalar.
pub(crate) struct ReplaceAll;
impl Builtin for ReplaceAll {
    const METHOD: BuiltinMethod = BuiltinMethod::ReplaceAll;
    const NAME: &'static str = "replace_all";
    fn spec() -> BuiltinSpec {
        BuiltinSpec::new(BuiltinCategory::Scalar, BuiltinCardinality::OneToOne)
            .indexed()
            .view_native()
            .pipeline_shape(BuiltinPipelineShape::new(
                BuiltinCardinality::OneToOne,
                true,
                2.0,
                1.0,
            ))
            .order_effect(BuiltinPipelineOrderEffect::Preserves)
            .lowering(BuiltinPipelineLowering::StringPairArg)
    }
    #[inline]
    fn apply_args(recv: &crate::data::value::Val, args: &super::BuiltinArgs) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::StrPair { first, second } => { super::replace_apply(recv.clone(), first, second, true) }
            _ => None,
        }
    }
}

/// `unknown` — sentinel for unrecognised methods (impure).
/// Canonical name uses angle brackets so it can never collide with user-callable names.
pub(crate) struct Unknown;
impl Builtin for Unknown {
    const METHOD: BuiltinMethod = BuiltinMethod::Unknown;
    const NAME: &'static str = "<unknown>";
    fn spec() -> BuiltinSpec {
        BuiltinSpec {
            pure: false,
            ..BuiltinSpec::new(BuiltinCategory::Unknown, BuiltinCardinality::OneToOne)
        }
    }
}

// ── Wildcard-default methods now made explicit (so all methods have defs entries) ──

/// `from_json` — string → JSON value (default scalar element).
pub(crate) struct FromJson;
impl Builtin for FromJson {
    const METHOD: BuiltinMethod = BuiltinMethod::FromJson;
    const NAME: &'static str = "from_json";
    fn spec() -> BuiltinSpec { default_scalar_spec(BuiltinMethod::FromJson) }
    #[inline]
    fn apply_one(recv: &crate::data::value::Val) -> Option<crate::data::value::Val> {
        Some(super::from_json_apply(recv).unwrap_or_else(|| recv.clone()))
    }
}

/// `includes(item)` / `contains(item)` — array membership scalar.
pub(crate) struct Includes;
impl Builtin for Includes {
    const METHOD: BuiltinMethod = BuiltinMethod::Includes;
    const NAME: &'static str = "includes";
    const ALIASES: &'static [&'static str] = &["contains"];
    fn spec() -> BuiltinSpec { default_scalar_spec(BuiltinMethod::Includes) }
    #[inline]
    fn apply_args(recv: &crate::data::value::Val, args: &super::BuiltinArgs) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Val(item) => Some(super::includes_apply(recv, item)),
            _ => None,
        }
    }
}

/// `index(item)` — first index of element.
pub(crate) struct Index;
impl Builtin for Index {
    const METHOD: BuiltinMethod = BuiltinMethod::Index;
    const NAME: &'static str = "index";
    fn spec() -> BuiltinSpec { default_scalar_spec(BuiltinMethod::Index) }
    #[inline]
    fn apply_args(recv: &crate::data::value::Val, args: &super::BuiltinArgs) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Val(item) => Some(super::index_value_apply(recv, item).unwrap_or_else(|| recv.clone())),
            _ => None,
        }
    }
}

/// `indices_of(item)` — all indices of element.
pub(crate) struct IndicesOf;
impl Builtin for IndicesOf {
    const METHOD: BuiltinMethod = BuiltinMethod::IndicesOf;
    const NAME: &'static str = "indices_of";
    fn spec() -> BuiltinSpec { default_scalar_spec(BuiltinMethod::IndicesOf) }
    #[inline]
    fn apply_args(recv: &crate::data::value::Val, args: &super::BuiltinArgs) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Val(item) => Some(super::indices_of_apply(recv, item).unwrap_or_else(|| recv.clone())),
            _ => None,
        }
    }
}

/// `missing` — sentinel for missing key/path.
pub(crate) struct Missing;
impl Builtin for Missing {
    const METHOD: BuiltinMethod = BuiltinMethod::Missing;
    const NAME: &'static str = "missing";
    fn spec() -> BuiltinSpec { default_scalar_spec(BuiltinMethod::Missing) }
    #[inline]
    fn apply_args(recv: &crate::data::value::Val, args: &super::BuiltinArgs) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Str(key) => Some(super::missing_apply(recv, key)),
            _ => None,
        }
    }
}

/// Default scalar fallback used by methods that previously fell to the wildcard arm.
/// Mirrors the `_ => { ... }` body in legacy `BuiltinMethod::spec()`.
fn default_scalar_spec(method: BuiltinMethod) -> BuiltinSpec {
    let spec = BuiltinSpec::new(BuiltinCategory::Scalar, BuiltinCardinality::OneToOne)
        .indexed()
        .view_native();
    if method.is_view_scalar_method() {
        spec.view_scalar()
    } else {
        spec
    }
}

// ── Cancellation-aware encode/decode pairs ───────────────────────────────────
// Each is a scalar element with the same spec body as `scalar_native_element_spec`
// but advertises an algebraic cancellation rule used by the optimizer to fuse
// adjacent inverse pairs (e.g. `to_base64(from_base64(x))` → identity).

/// `to_base64` — Forward base64 encode.
pub(crate) struct ToBase64;
impl Builtin for ToBase64 {
    const METHOD: BuiltinMethod = BuiltinMethod::ToBase64;
    const NAME: &'static str = "to_base64";
    fn spec() -> BuiltinSpec { scalar_native_element_spec() }
    #[inline]
    fn apply_one(recv: &crate::data::value::Val) -> Option<crate::data::value::Val> {
        Some(super::to_base64_apply(recv).unwrap_or_else(|| recv.clone()))
    }
    #[inline]
    fn cancellation() -> Option<BuiltinCancellation> {
        Some(BuiltinCancellation::Inverse {
            group: BuiltinCancelGroup::Base64,
            side: BuiltinCancelSide::Forward,
        })
    }
}

/// `from_base64` — Inverse of `to_base64`.
pub(crate) struct FromBase64;
impl Builtin for FromBase64 {
    const METHOD: BuiltinMethod = BuiltinMethod::FromBase64;
    const NAME: &'static str = "from_base64";
    fn spec() -> BuiltinSpec { scalar_native_element_spec() }
    #[inline]
    fn apply_one(recv: &crate::data::value::Val) -> Option<crate::data::value::Val> {
        Some(super::from_base64_apply(recv).unwrap_or_else(|| recv.clone()))
    }
    #[inline]
    fn cancellation() -> Option<BuiltinCancellation> {
        Some(BuiltinCancellation::Inverse {
            group: BuiltinCancelGroup::Base64,
            side: BuiltinCancelSide::Backward,
        })
    }
}

/// `url_encode` — Forward URL percent-encode.
pub(crate) struct UrlEncode;
impl Builtin for UrlEncode {
    const METHOD: BuiltinMethod = BuiltinMethod::UrlEncode;
    const NAME: &'static str = "url_encode";
    fn spec() -> BuiltinSpec { scalar_native_element_spec() }
    #[inline]
    fn apply_one(recv: &crate::data::value::Val) -> Option<crate::data::value::Val> {
        Some(super::url_encode_apply(recv).unwrap_or_else(|| recv.clone()))
    }
    #[inline]
    fn cancellation() -> Option<BuiltinCancellation> {
        Some(BuiltinCancellation::Inverse {
            group: BuiltinCancelGroup::Url,
            side: BuiltinCancelSide::Forward,
        })
    }
}

/// `url_decode` — Inverse of `url_encode`.
pub(crate) struct UrlDecode;
impl Builtin for UrlDecode {
    const METHOD: BuiltinMethod = BuiltinMethod::UrlDecode;
    const NAME: &'static str = "url_decode";
    fn spec() -> BuiltinSpec { scalar_native_element_spec() }
    #[inline]
    fn apply_one(recv: &crate::data::value::Val) -> Option<crate::data::value::Val> {
        Some(super::url_decode_apply(recv).unwrap_or_else(|| recv.clone()))
    }
    #[inline]
    fn cancellation() -> Option<BuiltinCancellation> {
        Some(BuiltinCancellation::Inverse {
            group: BuiltinCancelGroup::Url,
            side: BuiltinCancelSide::Backward,
        })
    }
}

/// `html_escape` — Forward HTML-entity escape.
pub(crate) struct HtmlEscape;
impl Builtin for HtmlEscape {
    const METHOD: BuiltinMethod = BuiltinMethod::HtmlEscape;
    const NAME: &'static str = "html_escape";
    fn spec() -> BuiltinSpec { scalar_native_element_spec() }
    #[inline]
    fn apply_one(recv: &crate::data::value::Val) -> Option<crate::data::value::Val> {
        Some(super::html_escape_apply(recv).unwrap_or_else(|| recv.clone()))
    }
    #[inline]
    fn cancellation() -> Option<BuiltinCancellation> {
        Some(BuiltinCancellation::Inverse {
            group: BuiltinCancelGroup::Html,
            side: BuiltinCancelSide::Forward,
        })
    }
}

/// `html_unescape` — Inverse of `html_escape`.
pub(crate) struct HtmlUnescape;
impl Builtin for HtmlUnescape {
    const METHOD: BuiltinMethod = BuiltinMethod::HtmlUnescape;
    const NAME: &'static str = "html_unescape";
    fn spec() -> BuiltinSpec { scalar_native_element_spec() }
    #[inline]
    fn apply_one(recv: &crate::data::value::Val) -> Option<crate::data::value::Val> {
        Some(super::html_unescape_apply(recv).unwrap_or_else(|| recv.clone()))
    }
    #[inline]
    fn cancellation() -> Option<BuiltinCancellation> {
        Some(BuiltinCancellation::Inverse {
            group: BuiltinCancelGroup::Html,
            side: BuiltinCancelSide::Backward,
        })
    }
}

/// `reverse_str` — Self-inverse string reversal (cancels with adjacent reverse_str).
pub(crate) struct ReverseStr;
impl Builtin for ReverseStr {
    const METHOD: BuiltinMethod = BuiltinMethod::ReverseStr;
    const NAME: &'static str = "reverse_str";
    fn spec() -> BuiltinSpec { scalar_native_element_spec() }
    #[inline]
    fn apply_one(recv: &crate::data::value::Val) -> Option<crate::data::value::Val> {
        Some(super::reverse_str_apply(recv).unwrap_or_else(|| recv.clone()))
    }
    #[inline]
    fn cancellation() -> Option<BuiltinCancellation> {
        Some(BuiltinCancellation::SelfInverse(BuiltinCancelGroup::Reverse))
    }
}

// ── Re-export Builtin trait constants used by cancellation impls (already imported above) ──

/// `or(default)` — coalesce: returns recv unless null/missing, else default.
pub(crate) struct Or;
impl Builtin for Or {
    const METHOD: BuiltinMethod = BuiltinMethod::Or;
    const NAME: &'static str = "or";
    fn spec() -> BuiltinSpec { scalar_native_element_spec() }
    #[inline]
    fn apply_args(recv: &crate::data::value::Val, args: &super::BuiltinArgs) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Val(default) => Some(super::or_apply(recv, default)),
            _ => None,
        }
    }
}

// ── Multi-arg scalar element methods (need apply_args) ──

macro_rules! str_arg_scalar_native {
    ( $( $ty:ident, $name:literal $( , aliases: [ $( $alias:literal ),* $(,)? ] )?, $apply:ident ; )* ) => {
        $(
            pub(crate) struct $ty;
            impl Builtin for $ty {
                const METHOD: BuiltinMethod = BuiltinMethod::$ty;
                const NAME: &'static str = $name;
                $( const ALIASES: &'static [&'static str] = &[ $( $alias ),* ]; )?
                fn spec() -> BuiltinSpec { scalar_native_element_spec() }
                #[inline]
                fn apply_args(recv: &crate::data::value::Val, args: &super::BuiltinArgs) -> Option<crate::data::value::Val> {
                    match args {
                        super::BuiltinArgs::Str(p) => {
                            Some(super::$apply(recv, p).unwrap_or_else(|| recv.clone()))
                        }
                        _ => None,
                    }
                }
            }
        )*
    };
}

str_arg_scalar_native! {
    Has, "has", has_apply;
    StripPrefix, "strip_prefix", strip_prefix_apply;
    StripSuffix, "strip_suffix", strip_suffix_apply;
    Scan, "scan", scan_apply;
    ReMatch, "re_match", re_match_apply;
    ReMatchFirst, "match_first", re_match_first_apply;
    ReMatchAll, "match_all", re_match_all_apply;
    ReCaptures, "captures", re_captures_apply;
}

// ── More multi-arg scalar element methods ──

// Str-arg cases that extend the str_arg_scalar_native pattern.
str_arg_scalar_native! {
    ReCapturesAll, "captures_all", re_captures_all_apply;
    ReSplit, "split_re", re_split_apply;
}

macro_rules! str_vec_arg_scalar_native {
    ( $( $ty:ident, $name:literal, $apply:ident ; )* ) => {
        $(
            pub(crate) struct $ty;
            impl Builtin for $ty {
                const METHOD: BuiltinMethod = BuiltinMethod::$ty;
                const NAME: &'static str = $name;
                fn spec() -> BuiltinSpec { scalar_native_element_spec() }
                #[inline]
                fn apply_args(recv: &crate::data::value::Val, args: &super::BuiltinArgs) -> Option<crate::data::value::Val> {
                    match args {
                        super::BuiltinArgs::StrVec(v) => {
                            Some(super::$apply(recv, v).unwrap_or_else(|| recv.clone()))
                        }
                        _ => None,
                    }
                }
            }
        )*
    };
}
str_vec_arg_scalar_native! {
    ContainsAny, "contains_any", contains_any_apply;
    ContainsAll, "contains_all", contains_all_apply;
}

macro_rules! usize_arg_scalar_native {
    ( $( $ty:ident, $name:literal, $apply:ident $( , aliases: [ $( $alias:literal ),* $(,)? ] )? ; )* ) => {
        $(
            pub(crate) struct $ty;
            impl Builtin for $ty {
                const METHOD: BuiltinMethod = BuiltinMethod::$ty;
                const NAME: &'static str = $name;
                $( const ALIASES: &'static [&'static str] = &[ $( $alias ),* ]; )?
                fn spec() -> BuiltinSpec { scalar_native_element_spec() }
                #[inline]
                fn apply_args(recv: &crate::data::value::Val, args: &super::BuiltinArgs) -> Option<crate::data::value::Val> {
                    match args {
                        super::BuiltinArgs::Usize(n) => {
                            Some(super::$apply(recv, *n).unwrap_or_else(|| recv.clone()))
                        }
                        _ => None,
                    }
                }
            }
        )*
    };
}
usize_arg_scalar_native! {
    Repeat, "repeat", repeat_apply, aliases: ["repeat_str"];
    Indent, "indent", indent_apply;
}

macro_rules! pad_arg_scalar_native {
    ( $( $ty:ident, $name:literal, $apply:ident ; )* ) => {
        $(
            pub(crate) struct $ty;
            impl Builtin for $ty {
                const METHOD: BuiltinMethod = BuiltinMethod::$ty;
                const NAME: &'static str = $name;
                fn spec() -> BuiltinSpec { scalar_native_element_spec() }
                #[inline]
                fn apply_args(recv: &crate::data::value::Val, args: &super::BuiltinArgs) -> Option<crate::data::value::Val> {
                    match args {
                        super::BuiltinArgs::Pad { width, fill } => {
                            Some(super::$apply(recv, *width, *fill).unwrap_or_else(|| recv.clone()))
                        }
                        _ => None,
                    }
                }
            }
        )*
    };
}
pad_arg_scalar_native! {
    PadLeft, "pad_left", pad_left_apply;
    PadRight, "pad_right", pad_right_apply;
    Center, "center", center_apply;
}

macro_rules! str_pair_scalar_native {
    ( $( $ty:ident, $name:literal, $apply:expr ; )* ) => {
        $(
            pub(crate) struct $ty;
            impl Builtin for $ty {
                const METHOD: BuiltinMethod = BuiltinMethod::$ty;
                const NAME: &'static str = $name;
                fn spec() -> BuiltinSpec { scalar_native_element_spec() }
                #[inline]
                fn apply_args(recv: &crate::data::value::Val, args: &super::BuiltinArgs) -> Option<crate::data::value::Val> {
                    match args {
                        super::BuiltinArgs::StrPair { first, second } => {
                            Some($apply(recv, first, second).unwrap_or_else(|| recv.clone()))
                        }
                        _ => None,
                    }
                }
            }
        )*
    };
}
str_pair_scalar_native! {
    ReReplace, "replace_re", super::re_replace_apply;
    ReReplaceAll, "replace_all_re", super::re_replace_all_apply;
}
