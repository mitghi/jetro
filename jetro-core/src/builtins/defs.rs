//! Per-method builtin definitions implementing the `Builtin` trait.
//!
//! One zero-sized struct per `BuiltinMethod` variant. Each struct's `impl Builtin` block
//! is the single source of truth for that method's identity, spec, and (future) runtime
//! behaviour. As migration proceeds, more methods move from the legacy `BuiltinMethod::spec()`
//! match into this file (or category-split children).

use super::{
    builtin::Builtin, BuiltinArgExtremeSink, BuiltinArraySelector, BuiltinCancelGroup,
    BuiltinCancelSide, BuiltinCancellation, BuiltinCardinality, BuiltinCategory,
    BuiltinColumnarStage, BuiltinDemandLaw, BuiltinExprPayload, BuiltinExprStage,
    BuiltinKeyedReducer, BuiltinLogicalShape, BuiltinMembershipSink, BuiltinMethod,
    BuiltinNullaryStage, BuiltinNumericReducer, BuiltinObjectLambda, BuiltinPipelineLowering,
    BuiltinPipelineMaterialization, BuiltinPipelineOrderEffect, BuiltinPipelineShape,
    BuiltinPredicateSink, BuiltinRawJsonScalar, BuiltinRowStreamOp, BuiltinRuntimeHook,
    BuiltinSelectionRewrite, BuiltinSpec, BuiltinStageMerge, BuiltinStreamingBoundary,
    BuiltinStringPairStage, BuiltinStructural, BuiltinViewNumericFullInput,
    BuiltinViewNumericScan, BuiltinViewRolling, BuiltinViewObjectProjection, BuiltinViewScalarOp,
    BuiltinViewSetFilter, BuiltinViewStage, BuiltinViewStringExpand, BuiltinViewValueProjection,
};

/// Numeric reducer (sum/avg/min/max) skeleton; same demand/lowering across the four.
#[inline]
fn numeric_reducer_spec(reducer: BuiltinNumericReducer) -> BuiltinSpec {
    BuiltinSpec::new(BuiltinCategory::Reducer, BuiltinCardinality::Reducing)
        .view_native()
        .numeric_sink(reducer)
        .cost(10.0)
        .demand_law(reducer.demand_law())
        .streaming_boundary(BuiltinStreamingBoundary::FullInputState)
        .logical_shape(reducer.logical_shape())
        .row_stream_op(reducer.row_stream_op())
        .lowering(BuiltinPipelineLowering::TerminalSink)
}

/// Arg-extreme reducer (`max_by` / `min_by`) skeleton.
#[inline]
fn arg_extreme_reducer_spec(sink: BuiltinArgExtremeSink) -> BuiltinSpec {
    BuiltinSpec::new(BuiltinCategory::Reducer, BuiltinCardinality::Reducing)
        .view_native()
        .arg_extreme_sink(sink)
        .cost(10.0)
        .demand_law(sink.demand_law())
        .lowering(BuiltinPipelineLowering::TerminalSink)
}

/// Predicate terminal sink skeleton for short-circuiting reducers.
#[inline]
fn predicate_terminal_sink_spec(sink: BuiltinPredicateSink) -> BuiltinSpec {
    BuiltinSpec::new(BuiltinCategory::Reducer, BuiltinCardinality::Reducing)
        .view_native()
        .predicate_sink(sink)
        .cost(10.0)
        .demand_law(sink.demand_law())
        .lowering(BuiltinPipelineLowering::TerminalSink)
}

/// Helper: shared spec body for streaming predicate filters (`filter` and `find_all`).
/// `find` has first-match terminal semantics and declares its own filter + first lowering.
#[inline]
fn filter_spec() -> BuiltinSpec {
    BuiltinSpec::new(
        BuiltinCategory::StreamingFilter,
        BuiltinCardinality::Filtering,
    )
    .view_native()
    .view_stage(BuiltinViewStage::Filter)
    .columnar_stage(BuiltinColumnarStage::Filter)
    .cost(10.0)
    .demand_law(BuiltinDemandLaw::FilterLike)
    .order_effect(BuiltinPipelineOrderEffect::PredicatePrefix)
    .expr_stage(BuiltinExprStage::Filter)
    .expr_payload(BuiltinExprPayload::PredicateScan)
    .logical_shape(BuiltinLogicalShape::Filter)
    .row_stream_op(BuiltinRowStreamOp::Filter)
    .runtime_hook(BuiltinRuntimeHook::SharedFilter)
    .output_cap_receiver()
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
    ) -> Result<
        crate::exec::pipeline::StageFlow<crate::data::value::Val>,
        crate::data::context::EvalError,
    > {
        let prog = body.expect("filter body");
        let keep = super::filter_one(&item, |v| {
            crate::exec::pipeline::eval_kernel_with_vm(ctx.kernel, v, ctx.vm, |it, vm| {
                crate::exec::pipeline::apply_item_in_env(vm, ctx.env, it, prog)
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
            crate::exec::pipeline::eval_kernel_with_vm(ctx.kernel, v, ctx.vm, |item, vm| {
                crate::exec::pipeline::apply_item_in_env(vm, ctx.env, item, prog)
            })
        });
        match result {
            Ok(out) => {
                *buf = out;
                Some(Ok(()))
            }
            Err(err) => Some(Err(err)),
        }
    }
}

/// `find(pred)` — returns the first element for which `pred` is truthy,
/// or `null` when nothing matches. Matches the conventional first-match
/// semantics found in JavaScript / Rust / Python iterators. Use
/// `find_all` (filter alias) when every match is desired.
pub(crate) struct Find;
impl Builtin for Find {
    const METHOD: BuiltinMethod = BuiltinMethod::Find;
    const NAME: &'static str = "find";

    fn spec() -> BuiltinSpec {
        BuiltinSpec::new(
            BuiltinCategory::StreamingFilter,
            BuiltinCardinality::Filtering,
        )
        .view_native()
        .view_stage(BuiltinViewStage::Filter)
        .cost(10.0)
        .demand_law(BuiltinDemandLaw::FilterLike)
        .expr_stage(BuiltinExprStage::Filter)
        .logical_shape(BuiltinLogicalShape::FilterThenFirst)
        .row_stream_op(BuiltinRowStreamOp::FindFirst)
        .runtime_hook(BuiltinRuntimeHook::SharedFilter)
        .lowering(BuiltinPipelineLowering::TerminalExprArg {
            terminal: BuiltinMethod::First,
        })
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
        BuiltinSpec::new(
            BuiltinCategory::StreamingFilter,
            BuiltinCardinality::Filtering,
        )
        .view_native()
        .view_stage(BuiltinViewStage::Compact)
        .cost(10.0)
        .demand_law(BuiltinDemandLaw::FilterLike)
        .runtime_hook(BuiltinRuntimeHook::StreamAndBarrier)
        .order_effect(BuiltinPipelineOrderEffect::PredicatePrefix)
    }
    #[inline]
    fn apply_one(recv: &crate::data::value::Val) -> Option<crate::data::value::Val> {
        Some(super::compact_apply(recv).unwrap_or_else(|| recv.clone()))
    }

    #[inline]
    fn apply_stream(
        _ctx: &mut super::builtin::StreamCtx<'_, '_>,
        item: crate::data::value::Val,
        _body: Option<&crate::vm::Program>,
    ) -> Result<
        crate::exec::pipeline::StageFlow<crate::data::value::Val>,
        crate::data::context::EvalError,
    > {
        Ok(if matches!(item, crate::data::value::Val::Null) {
            crate::exec::pipeline::StageFlow::SkipRow
        } else {
            crate::exec::pipeline::StageFlow::Continue(item)
        })
    }

    #[inline]
    fn apply_barrier(
        _ctx: &mut super::builtin::BarrierCtx<'_>,
        buf: &mut Vec<crate::data::value::Val>,
        _body: Option<&crate::vm::Program>,
    ) -> Option<Result<(), crate::data::context::EvalError>> {
        buf.retain(|v| !matches!(v, crate::data::value::Val::Null));
        Some(Ok(()))
    }
}

/// Removes elements equal to the literal argument; degenerate equality filter.
pub(crate) struct Remove;
impl Builtin for Remove {
    const METHOD: BuiltinMethod = BuiltinMethod::Remove;
    const NAME: &'static str = "remove";

    fn spec() -> BuiltinSpec {
        BuiltinSpec::new(
            BuiltinCategory::StreamingFilter,
            BuiltinCardinality::Filtering,
        )
        .view_native()
        .view_stage(BuiltinViewStage::RemoveValue)
        .cost(10.0)
        .demand_law(BuiltinDemandLaw::FilterLike)
        .runtime_hook(BuiltinRuntimeHook::StreamAndBarrier)
        .order_effect(BuiltinPipelineOrderEffect::PredicatePrefix)
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Val(item) => {
                Some(super::remove_value_apply(recv, item).unwrap_or_else(|| recv.clone()))
            }
            _ => None,
        }
    }

    #[inline]
    fn apply_stream(
        ctx: &mut super::builtin::StreamCtx<'_, '_>,
        item: crate::data::value::Val,
        _body: Option<&crate::vm::Program>,
    ) -> Result<
        crate::exec::pipeline::StageFlow<crate::data::value::Val>,
        crate::data::context::EvalError,
    > {
        let crate::exec::pipeline::Stage::Builtin(call) = ctx.stage else {
            return Ok(crate::exec::pipeline::StageFlow::Continue(item));
        };
        match &call.args {
            super::BuiltinArgs::Val(target) if crate::util::vals_deep_eq(&item, target) => {
                Ok(crate::exec::pipeline::StageFlow::SkipRow)
            }
            _ => Ok(crate::exec::pipeline::StageFlow::Continue(item)),
        }
    }

    #[inline]
    fn apply_barrier(
        ctx: &mut super::builtin::BarrierCtx<'_>,
        buf: &mut Vec<crate::data::value::Val>,
        _body: Option<&crate::vm::Program>,
    ) -> Option<Result<(), crate::data::context::EvalError>> {
        let crate::exec::pipeline::Stage::Builtin(call) = ctx.stage else {
            return Some(Ok(()));
        };
        if let super::BuiltinArgs::Val(target) = &call.args {
            buf.retain(|v| !crate::util::vals_deep_eq(v, target));
        }
        Some(Ok(()))
    }
}

/// Per-element projection via lambda; preserves cardinality and order.
pub(crate) struct Map;
impl Builtin for Map {
    const METHOD: BuiltinMethod = BuiltinMethod::Map;
    const NAME: &'static str = "map";

    fn spec() -> BuiltinSpec {
        BuiltinSpec::new(
            BuiltinCategory::StreamingOneToOne,
            BuiltinCardinality::OneToOne,
        )
        .indexed()
        .view_native()
        .view_stage(BuiltinViewStage::Map)
        .columnar_stage(BuiltinColumnarStage::Map)
        .cost(10.0)
        .demand_law(BuiltinDemandLaw::MapLike)
        .order_effect(BuiltinPipelineOrderEffect::Preserves)
        .expr_stage(BuiltinExprStage::Map)
        .expr_payload(BuiltinExprPayload::Projection)
        .logical_shape(BuiltinLogicalShape::Map)
        .row_stream_op(BuiltinRowStreamOp::Map)
        .runtime_hook(BuiltinRuntimeHook::StreamAndBarrier)
        .lowering(BuiltinPipelineLowering::ExprArg)
        .output_cap_receiver()
        .element()
    }

    #[inline]
    fn apply_stream(
        ctx: &mut super::builtin::StreamCtx<'_, '_>,
        item: crate::data::value::Val,
        body: Option<&crate::vm::Program>,
    ) -> Result<
        crate::exec::pipeline::StageFlow<crate::data::value::Val>,
        crate::data::context::EvalError,
    > {
        let prog = body.expect("map body");
        // Terminal-map collector short-circuit (avoid allocating intermediate Val).
        if Some(ctx.stage_idx) == ctx.terminal_map_idx {
            ctx.terminal_map_collect
                .as_mut()
                .expect("terminal map collector")
                .push_val_row(&item, ctx.kernel, ctx.vm, |it, vm| {
                    crate::exec::pipeline::apply_item_in_env(vm, ctx.env, it, prog)
                })?;
            return Ok(crate::exec::pipeline::StageFlow::TerminalCollected);
        }
        let mapped = super::map_one(&item, |v| {
            crate::exec::pipeline::eval_kernel_with_vm(ctx.kernel, v, ctx.vm, |it, vm| {
                crate::exec::pipeline::apply_item_in_env(vm, ctx.env, it, prog)
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
            crate::exec::pipeline::eval_kernel_with_vm(ctx.kernel, v, ctx.vm, |item, vm| {
                crate::exec::pipeline::apply_item_in_env(vm, ctx.env, item, prog)
            })
        });
        match result {
            Ok(out) => {
                *buf = out;
                Some(Ok(()))
            }
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
        BuiltinSpec::new(
            BuiltinCategory::StreamingExpand,
            BuiltinCardinality::Expanding,
        )
        .view_native()
        .view_stage(BuiltinViewStage::FlatMap)
        .columnar_stage(BuiltinColumnarStage::FlatMap)
        .cost(10.0)
        .demand_law(BuiltinDemandLaw::FlatMapLike)
        .streaming_boundary(BuiltinStreamingBoundary::Expanding)
        .expr_stage(BuiltinExprStage::FlatMap)
        .logical_shape(BuiltinLogicalShape::FlatMap)
        .runtime_hook(BuiltinRuntimeHook::StreamAndBarrier)
        .lowering(BuiltinPipelineLowering::ExprArg)
    }

    #[inline]
    fn apply_stream(
        ctx: &mut super::builtin::StreamCtx<'_, '_>,
        item: crate::data::value::Val,
        body: Option<&crate::vm::Program>,
    ) -> Result<
        crate::exec::pipeline::StageFlow<crate::data::value::Val>,
        crate::data::context::EvalError,
    > {
        let Some(prog) = body else {
            return Ok(crate::exec::pipeline::StageFlow::Continue(item));
        };
        let inner =
            crate::exec::pipeline::eval_kernel_with_vm(ctx.kernel, &item, ctx.vm, |row, vm| {
                crate::exec::pipeline::apply_item_in_env(vm, ctx.env, row, prog)
            })?;
        if let Some(arr) = inner.as_vals() {
            Ok(crate::exec::pipeline::StageFlow::Expand(
                arr.iter().cloned().collect(),
            ))
        } else {
            Ok(crate::exec::pipeline::StageFlow::Continue(inner))
        }
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
            let inner = match crate::exec::pipeline::eval_kernel_with_vm(
                ctx.kernel,
                v,
                ctx.vm,
                |item, vm| crate::exec::pipeline::apply_item_in_env(vm, ctx.env, item, prog),
            ) {
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
            .streaming_boundary(BuiltinStreamingBoundary::BoundedState)
            .order_effect(BuiltinPipelineOrderEffect::Preserves)
            .logical_shape(BuiltinLogicalShape::Take)
            .row_stream_op(BuiltinRowStreamOp::Take)
            .runtime_hook(BuiltinRuntimeHook::StreamAndBarrier)
            .lowering(BuiltinPipelineLowering::UsizeArg { min: 0 })
    }

    #[inline]
    fn apply_stream(
        ctx: &mut super::builtin::StreamCtx<'_, '_>,
        item: crate::data::value::Val,
        _body: Option<&crate::vm::Program>,
    ) -> Result<
        crate::exec::pipeline::StageFlow<crate::data::value::Val>,
        crate::data::context::EvalError,
    > {
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

    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Usize(n) => super::take_apply(recv, *n),
            _ => None,
        }
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
            .streaming_boundary(BuiltinStreamingBoundary::BoundedState)
            .order_effect(BuiltinPipelineOrderEffect::Preserves)
            .logical_shape(BuiltinLogicalShape::Skip)
            .runtime_hook(BuiltinRuntimeHook::StreamAndBarrier)
            .lowering(BuiltinPipelineLowering::UsizeArg { min: 0 })
    }

    #[inline]
    fn apply_stream(
        ctx: &mut super::builtin::StreamCtx<'_, '_>,
        item: crate::data::value::Val,
        _body: Option<&crate::vm::Program>,
    ) -> Result<
        crate::exec::pipeline::StageFlow<crate::data::value::Val>,
        crate::data::context::EvalError,
    > {
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

    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Usize(n) => super::skip_apply(recv, *n),
            _ => None,
        }
    }
}

/// Selects the first element; terminal positional sink.
pub(crate) struct First;
impl Builtin for First {
    const METHOD: BuiltinMethod = BuiltinMethod::First;
    const NAME: &'static str = "first";

    fn spec() -> BuiltinSpec {
        let selector = BuiltinArraySelector::First;
        BuiltinSpec::new(BuiltinCategory::Positional, BuiltinCardinality::Bounded)
            .view_native()
            .array_selector(selector)
            .select_one_sink(
                selector
                    .selection_position()
                    .expect("first selector position"),
            )
            .demand_law(selector.demand_law())
            .streaming_boundary(BuiltinStreamingBoundary::BoundedState)
            .logical_shape(BuiltinLogicalShape::First)
            .row_stream_op(selector.row_stream_op().expect("first row stream op"))
            .lowering(selector.pipeline_lowering())
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::I64(n) => super::first_apply(recv, *n),
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
        let selector = BuiltinArraySelector::Last;
        BuiltinSpec::new(BuiltinCategory::Positional, BuiltinCardinality::Bounded)
            .view_native()
            .array_selector(selector)
            .select_one_sink(
                selector
                    .selection_position()
                    .expect("last selector position"),
            )
            .demand_law(selector.demand_law())
            .streaming_boundary(BuiltinStreamingBoundary::BoundedState)
            .logical_shape(BuiltinLogicalShape::Last)
            .row_stream_op(selector.row_stream_op().expect("last row stream op"))
            .lowering(selector.pipeline_lowering())
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::I64(n) => super::last_apply(recv, *n),
            _ => None,
        }
    }
}

/// Take elements while predicate holds; stops at first failure.
pub(crate) struct TakeWhile;
impl Builtin for TakeWhile {
    const METHOD: BuiltinMethod = BuiltinMethod::TakeWhile;
    const NAME: &'static str = "take_while";
    const ALIASES: &'static [&'static str] = &["takewhile"];

    fn spec() -> BuiltinSpec {
        BuiltinSpec::new(
            BuiltinCategory::StreamingFilter,
            BuiltinCardinality::Filtering,
        )
        .view_native()
        .view_stage(BuiltinViewStage::TakeWhile)
        .cost(10.0)
        .demand_law(BuiltinDemandLaw::TakeWhile)
        .streaming_boundary(BuiltinStreamingBoundary::PrefixState)
        .expr_payload(BuiltinExprPayload::PredicateScan)
        .logical_shape(BuiltinLogicalShape::TakeWhile)
        .runtime_hook(BuiltinRuntimeHook::StreamAndBarrier)
        .pipeline_shape(BuiltinPipelineShape::new(
            BuiltinCardinality::Filtering,
            false,
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
    ) -> Result<
        crate::exec::pipeline::StageFlow<crate::data::value::Val>,
        crate::data::context::EvalError,
    > {
        let prog = body.expect("take_while body");
        let pass = super::take_while_one(&item, |v| {
            crate::exec::pipeline::eval_kernel_with_vm(ctx.kernel, v, ctx.vm, |it, vm| {
                crate::exec::pipeline::apply_item_in_env(vm, ctx.env, it, prog)
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
            crate::exec::pipeline::eval_kernel_with_vm(ctx.kernel, v, ctx.vm, |item, vm| {
                crate::exec::pipeline::apply_item_in_env(vm, ctx.env, item, prog)
            })
        });
        match result {
            Ok(out) => {
                *buf = out;
                Some(Ok(()))
            }
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
        BuiltinSpec::new(
            BuiltinCategory::StreamingFilter,
            BuiltinCardinality::Filtering,
        )
        .view_native()
        .view_stage(BuiltinViewStage::DropWhile)
        .cost(10.0)
        .demand_law(BuiltinDemandLaw::DropWhile)
        .expr_payload(BuiltinExprPayload::PredicateScan)
        .logical_shape(BuiltinLogicalShape::DropWhile)
        .streaming_boundary(BuiltinStreamingBoundary::PrefixState)
        .runtime_hook(BuiltinRuntimeHook::StreamAndBarrier)
        .pipeline_shape(BuiltinPipelineShape::new(
            BuiltinCardinality::Filtering,
            false,
            10.0,
            0.5,
        ))
        .order_effect(BuiltinPipelineOrderEffect::Blocks)
        .lowering(BuiltinPipelineLowering::ExprArg)
    }

    #[inline]
    fn apply_stream(
        ctx: &mut super::builtin::StreamCtx<'_, '_>,
        item: crate::data::value::Val,
        body: Option<&crate::vm::Program>,
    ) -> Result<
        crate::exec::pipeline::StageFlow<crate::data::value::Val>,
        crate::data::context::EvalError,
    > {
        if ctx.stage_taken[ctx.stage_idx] != 0 {
            return Ok(crate::exec::pipeline::StageFlow::Continue(item));
        }
        let prog = body.expect("drop_while body");
        let drop = super::filter_one(&item, |v| {
            crate::exec::pipeline::eval_kernel_with_vm(ctx.kernel, v, ctx.vm, |it, vm| {
                crate::exec::pipeline::apply_item_in_env(vm, ctx.env, it, prog)
            })
        })?;
        if drop {
            ctx.stage_skipped[ctx.stage_idx] += 1;
            Ok(crate::exec::pipeline::StageFlow::SkipRow)
        } else {
            ctx.stage_taken[ctx.stage_idx] = 1;
            Ok(crate::exec::pipeline::StageFlow::Continue(item))
        }
    }

    #[inline]
    fn apply_barrier(
        ctx: &mut super::builtin::BarrierCtx<'_>,
        buf: &mut Vec<crate::data::value::Val>,
        body: Option<&crate::vm::Program>,
    ) -> Option<Result<(), crate::data::context::EvalError>> {
        let prog = body?;
        let result = super::drop_while_apply(std::mem::take(buf), |v| {
            crate::exec::pipeline::eval_kernel_with_vm(ctx.kernel, v, ctx.vm, |item, vm| {
                crate::exec::pipeline::apply_item_in_env(vm, ctx.env, item, prog)
            })
        });
        match result {
            Ok(out) => {
                *buf = out;
                Some(Ok(()))
            }
            Err(err) => Some(Err(err)),
        }
    }
}

/// Element count via scalar view sink; degenerate non-numeric reducer.
pub(crate) struct Len;
impl Builtin for Len {
    const METHOD: BuiltinMethod = BuiltinMethod::Len;
    const NAME: &'static str = "len";

    fn spec() -> BuiltinSpec {
        let raw = BuiltinRawJsonScalar::Len;
        BuiltinSpec::new(BuiltinCategory::Reducer, BuiltinCardinality::Reducing)
            .indexed()
            .view_scalar_op(BuiltinViewScalarOp::Len)
            .raw_json_scalar(raw)
            .row_stream_op(BuiltinRowStreamOp::Count)
            .demand_law(raw.demand_law())
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
            .count_sink_with_predicate()
            .cost(10.0)
            .demand_law(BuiltinDemandLaw::Count)
            .logical_shape(BuiltinLogicalShape::Count)
            .row_stream_op(BuiltinRowStreamOp::Count)
            .lowering(BuiltinPipelineLowering::TerminalSink)
    }
}

/// HyperLogLog-style approximate distinct count.
///
/// Backed by the shared small-range-corrected HyperLogLog estimator used by
/// buffered, pipeline, and view/tape execution. Tiny inputs remain exact while
/// large streams keep a fixed-size register array.
pub(crate) struct ApproxCountDistinct;
impl Builtin for ApproxCountDistinct {
    const METHOD: BuiltinMethod = BuiltinMethod::ApproxCountDistinct;
    const NAME: &'static str = "approx_count_distinct";

    fn spec() -> BuiltinSpec {
        BuiltinSpec::new(BuiltinCategory::Reducer, BuiltinCardinality::Reducing)
            .view_native()
            .approx_distinct_sink()
            .cost(10.0)
            .demand_law(BuiltinDemandLaw::RowKeyedReducer)
            .logical_shape(BuiltinLogicalShape::ApproxCountDistinct)
            .lowering(BuiltinPipelineLowering::TerminalSink)
    }
    #[inline]
    fn apply_one(recv: &crate::data::value::Val) -> Option<crate::data::value::Val> {
        let items = recv.as_vals()?;
        Some(crate::data::value::Val::Int(
            super::hll_count_distinct(&items) as i64,
        ))
    }
}

/// Boolean reducer: true if any element matches predicate.
pub(crate) struct Any;
impl Builtin for Any {
    const METHOD: BuiltinMethod = BuiltinMethod::Any;
    const NAME: &'static str = "any";
    const ALIASES: &'static [&'static str] = &["exists"];

    fn spec() -> BuiltinSpec {
        predicate_terminal_sink_spec(BuiltinPredicateSink::Any)
            .row_stream_op(BuiltinRowStreamOp::Any)
    }
}

/// Boolean reducer: true if all elements match predicate.
pub(crate) struct All;
impl Builtin for All {
    const METHOD: BuiltinMethod = BuiltinMethod::All;
    const NAME: &'static str = "all";

    fn spec() -> BuiltinSpec {
        predicate_terminal_sink_spec(BuiltinPredicateSink::All)
            .row_stream_op(BuiltinRowStreamOp::All)
    }
}

/// Index of the first element satisfying the predicate.
pub(crate) struct FindIndex;
impl Builtin for FindIndex {
    const METHOD: BuiltinMethod = BuiltinMethod::FindIndex;
    const NAME: &'static str = "find_index";

    fn spec() -> BuiltinSpec {
        predicate_terminal_sink_spec(BuiltinPredicateSink::FindIndex)
            .runtime_hook(BuiltinRuntimeHook::Barrier)
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
                crate::exec::pipeline::eval_kernel_with_vm(ctx.kernel, item, ctx.vm, |it, vm| {
                    crate::exec::pipeline::apply_item_in_env(vm, ctx.env, it, prog)
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
        predicate_terminal_sink_spec(BuiltinPredicateSink::IndicesWhere)
            .runtime_hook(BuiltinRuntimeHook::Barrier)
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
                crate::exec::pipeline::eval_kernel_with_vm(ctx.kernel, item, ctx.vm, |it, vm| {
                    crate::exec::pipeline::apply_item_in_env(vm, ctx.env, it, prog)
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
    let mut best_key =
        match crate::exec::pipeline::eval_kernel_with_vm(ctx.kernel, &buf[0], ctx.vm, |item, vm| {
            crate::exec::pipeline::apply_item_in_env(vm, ctx.env, item, prog)
        }) {
            Ok(key) => key,
            Err(err) => return Some(Err(err)),
        };
    for i in 1..buf.len() {
        let key = match crate::exec::pipeline::eval_kernel_with_vm(
            ctx.kernel,
            &buf[i],
            ctx.vm,
            |item, vm| crate::exec::pipeline::apply_item_in_env(vm, ctx.env, item, prog),
        ) {
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
        arg_extreme_reducer_spec(BuiltinArgExtremeSink::MaxBy)
            .runtime_hook(BuiltinRuntimeHook::Barrier)
    }

    #[inline]
    fn apply_barrier(
        ctx: &mut super::builtin::BarrierCtx<'_>,
        buf: &mut Vec<crate::data::value::Val>,
        body: Option<&crate::vm::Program>,
    ) -> Option<Result<(), crate::data::context::EvalError>> {
        arg_extreme_apply_barrier(ctx, buf, body, BuiltinArgExtremeSink::MaxBy.wants_max())
    }
}

/// Element with the smallest projected key.
pub(crate) struct MinBy;
impl Builtin for MinBy {
    const METHOD: BuiltinMethod = BuiltinMethod::MinBy;
    const NAME: &'static str = "min_by";

    fn spec() -> BuiltinSpec {
        arg_extreme_reducer_spec(BuiltinArgExtremeSink::MinBy)
            .runtime_hook(BuiltinRuntimeHook::Barrier)
    }

    #[inline]
    fn apply_barrier(
        ctx: &mut super::builtin::BarrierCtx<'_>,
        buf: &mut Vec<crate::data::value::Val>,
        body: Option<&crate::vm::Program>,
    ) -> Option<Result<(), crate::data::context::EvalError>> {
        arg_extreme_apply_barrier(ctx, buf, body, BuiltinArgExtremeSink::MinBy.wants_max())
    }
}

/// `enumerate` — pairs each element with its index. Operates on the
/// whole receiver array as one unit; NOT marked `.element()` because the
/// streaming pipeline would otherwise treat the receiver as a 1-element
/// stream and discard the pairing (visible as `[items]` not `[{index,
/// value}, ...]`).
pub(crate) struct Enumerate;
impl Builtin for Enumerate {
    const METHOD: BuiltinMethod = BuiltinMethod::Enumerate;
    const NAME: &'static str = "enumerate";
    fn spec() -> BuiltinSpec {
        BuiltinSpec::new(
            BuiltinCategory::StreamingOneToOne,
            BuiltinCardinality::OneToOne,
        )
        .view_native()
        .view_stage(BuiltinViewStage::Enumerate)
        .demand_law(BuiltinDemandLaw::MapLike)
        .runtime_hook(BuiltinRuntimeHook::StreamAndBarrier)
        .lowering(BuiltinPipelineLowering::Nullary)
        .cost(10.0)
    }
    #[inline]
    fn apply_stream(
        ctx: &mut super::builtin::StreamCtx<'_, '_>,
        item: crate::data::value::Val,
        _body: Option<&crate::vm::Program>,
    ) -> Result<
        crate::exec::pipeline::StageFlow<crate::data::value::Val>,
        crate::data::context::EvalError,
    > {
        let index = ctx.stage_taken[ctx.stage_idx];
        ctx.stage_taken[ctx.stage_idx] = index.saturating_add(1);
        Ok(crate::exec::pipeline::StageFlow::Continue(
            crate::util::obj2("index", crate::data::value::Val::Int(index as i64), "value", item),
        ))
    }
    #[inline]
    fn apply_one(recv: &crate::data::value::Val) -> Option<crate::data::value::Val> {
        Some(super::enumerate_apply(recv).unwrap_or_else(|| recv.clone()))
    }
}

/// `pairwise` — yields adjacent pairs as `[a, b]` tuples. Like
/// `enumerate`, NOT marked `.element()` because the streaming pipeline
/// would treat the receiver as a 1-element stream and discard the pair
/// formation, returning the bare array instead of `[[a,b], ...]`.
pub(crate) struct Pairwise;
impl Builtin for Pairwise {
    const METHOD: BuiltinMethod = BuiltinMethod::Pairwise;
    const NAME: &'static str = "pairwise";
    fn spec() -> BuiltinSpec {
        BuiltinSpec::new(
            BuiltinCategory::StreamingFilter,
            BuiltinCardinality::Filtering,
        )
        .view_native()
        .view_stage(BuiltinViewStage::Pairwise)
        .demand_law(BuiltinDemandLaw::Pairwise)
        .runtime_hook(BuiltinRuntimeHook::StreamAndBarrier)
        .streaming_boundary(BuiltinStreamingBoundary::BoundedState)
        .pipeline_shape(BuiltinPipelineShape::new(
            BuiltinCardinality::Filtering,
            false,
            2.0,
            1.0,
        ))
        .lowering(BuiltinPipelineLowering::Nullary)
        .cost(10.0)
    }
    #[inline]
    fn apply_stream(
        ctx: &mut super::builtin::StreamCtx<'_, '_>,
        item: crate::data::value::Val,
        _body: Option<&crate::vm::Program>,
    ) -> Result<
        crate::exec::pipeline::StageFlow<crate::data::value::Val>,
        crate::data::context::EvalError,
    > {
        let buffer = &mut ctx.stage_window_buffers[ctx.stage_idx];
        buffer.push_back(item);
        while buffer.len() > 2 {
            buffer.pop_front();
        }
        if buffer.len() < 2 {
            return Ok(crate::exec::pipeline::StageFlow::SkipRow);
        }
        Ok(crate::exec::pipeline::StageFlow::Continue(
            crate::data::value::Val::arr(buffer.iter().cloned().collect()),
        ))
    }
    #[inline]
    fn apply_one(recv: &crate::data::value::Val) -> Option<crate::data::value::Val> {
        Some(super::pairwise_apply(recv).unwrap_or_else(|| recv.clone()))
    }
}

#[inline]
fn expand_simple_spec() -> BuiltinSpec {
    BuiltinSpec::new(
        BuiltinCategory::StreamingExpand,
        BuiltinCardinality::Expanding,
    )
    .cost(10.0)
    .demand_law(BuiltinDemandLaw::FlatMapLike)
}

/// `flatten` — concatenates nested arrays.
pub(crate) struct Flatten;
impl Builtin for Flatten {
    const METHOD: BuiltinMethod = BuiltinMethod::Flatten;
    const NAME: &'static str = "flatten";
    fn spec() -> BuiltinSpec {
        expand_simple_spec()
            .view_native()
            .view_stage(BuiltinViewStage::Flatten)
            .lowering(BuiltinPipelineLowering::UsizeArg { min: 0 })
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Usize(depth) => super::flatten_depth_apply(recv, *depth),
            _ => None,
        }
    }
}

/// `explode` — same as flatten with object semantics.
pub(crate) struct Explode;
impl Builtin for Explode {
    const METHOD: BuiltinMethod = BuiltinMethod::Explode;
    const NAME: &'static str = "explode";
    fn spec() -> BuiltinSpec {
        expand_simple_spec()
            .view_native()
            .view_stage(BuiltinViewStage::Explode)
            .lowering(BuiltinPipelineLowering::StringArg)
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Str(field) => super::explode_apply(recv, field),
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
        BuiltinSpec::new(
            BuiltinCategory::StreamingExpand,
            BuiltinCardinality::Expanding,
        )
        .cost(10.0)
        .demand_law(BuiltinDemandLaw::FlatMapLike)
        .runtime_hook(BuiltinRuntimeHook::StreamAndBarrier)
        .streaming_boundary(BuiltinStreamingBoundary::Expanding)
        .pipeline_shape(BuiltinPipelineShape::new(
            BuiltinCardinality::Expanding,
            false,
            2.0,
            1.0,
        ))
        .view_native()
        .view_string_expand(BuiltinViewStringExpand::Split)
        .lowering(BuiltinPipelineLowering::StringArg)
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Str(p) => super::split_apply(recv, p),
            _ => None,
        }
    }
    #[inline]
    fn apply_stream(
        ctx: &mut super::builtin::StreamCtx<'_, '_>,
        item: crate::data::value::Val,
        _body: Option<&crate::vm::Program>,
    ) -> Result<
        crate::exec::pipeline::StageFlow<crate::data::value::Val>,
        crate::data::context::EvalError,
    > {
        let crate::exec::pipeline::Stage::StringBuiltin { value, .. } = ctx.stage else {
            return Ok(crate::exec::pipeline::StageFlow::Continue(item));
        };
        Ok(match super::split_apply(&item, value) {
            Some(crate::data::value::Val::Arr(items)) => crate::exec::pipeline::StageFlow::Expand(
                std::sync::Arc::try_unwrap(items).unwrap_or_else(|items| (*items).clone()),
            ),
            Some(other) => crate::exec::pipeline::StageFlow::Continue(other),
            None => crate::exec::pipeline::StageFlow::Continue(item),
        })
    }
}

#[inline]
fn expand_element_spec(expand: BuiltinViewStringExpand) -> BuiltinSpec {
    BuiltinSpec::new(
        BuiltinCategory::StreamingExpand,
        BuiltinCardinality::Expanding,
    )
    .view_string_expand(expand)
    .cost(10.0)
    .demand_law(BuiltinDemandLaw::FlatMapLike)
    .runtime_hook(BuiltinRuntimeHook::StreamAndBarrier)
    .streaming_boundary(BuiltinStreamingBoundary::Expanding)
    .pipeline_shape(BuiltinPipelineShape::new(
        BuiltinCardinality::Expanding,
        false,
        2.0,
        1.0,
    ))
    .lowering(BuiltinPipelineLowering::Nullary)
    .element()
}

#[inline]
fn expand_string_stage_flow(
    item: crate::data::value::Val,
    apply: impl FnOnce(&crate::data::value::Val) -> Option<crate::data::value::Val>,
) -> crate::exec::pipeline::StageFlow<crate::data::value::Val> {
    match apply(&item) {
        Some(crate::data::value::Val::Arr(items)) => crate::exec::pipeline::StageFlow::Expand(
            std::sync::Arc::try_unwrap(items).unwrap_or_else(|items| (*items).clone()),
        ),
        Some(other) => crate::exec::pipeline::StageFlow::Continue(other),
        None => crate::exec::pipeline::StageFlow::Continue(item),
    }
}

#[inline]
fn expand_string_barrier(
    buf: &mut Vec<crate::data::value::Val>,
    apply: impl Fn(&crate::data::value::Val) -> Option<crate::data::value::Val>,
) {
    let mut out = Vec::with_capacity(buf.len());
    for value in std::mem::take(buf) {
        match apply(&value) {
            Some(crate::data::value::Val::Arr(items)) => out
                .extend(std::sync::Arc::try_unwrap(items).unwrap_or_else(|items| (*items).clone())),
            Some(other) => out.push(other),
            None => out.push(value),
        }
    }
    *buf = out;
}

macro_rules! string_expand_builtin {
    ($ty:ident, $method:ident, $name:literal, $expand:ident, $apply:path) => {
        pub(crate) struct $ty;
        impl Builtin for $ty {
            const METHOD: BuiltinMethod = BuiltinMethod::$method;
            const NAME: &'static str = $name;
            fn spec() -> BuiltinSpec {
                expand_element_spec(BuiltinViewStringExpand::$expand)
            }
            #[inline]
            fn apply_one(recv: &crate::data::value::Val) -> Option<crate::data::value::Val> {
                Some($apply(recv).unwrap_or_else(|| recv.clone()))
            }
            #[inline]
            fn apply_stream(
                _ctx: &mut super::builtin::StreamCtx<'_, '_>,
                item: crate::data::value::Val,
                _body: Option<&crate::vm::Program>,
            ) -> Result<
                crate::exec::pipeline::StageFlow<crate::data::value::Val>,
                crate::data::context::EvalError,
            > {
                Ok(expand_string_stage_flow(item, $apply))
            }
            #[inline]
            fn apply_barrier(
                _ctx: &mut super::builtin::BarrierCtx<'_>,
                buf: &mut Vec<crate::data::value::Val>,
                _body: Option<&crate::vm::Program>,
            ) -> Option<Result<(), crate::data::context::EvalError>> {
                expand_string_barrier(buf, $apply);
                Some(Ok(()))
            }
        }
    };
}

string_expand_builtin!(Lines, Lines, "lines", Lines, super::lines_apply);
string_expand_builtin!(Words, Words, "words", Words, super::words_apply);
string_expand_builtin!(Chars, Chars, "chars", Chars, super::chars_apply);
string_expand_builtin!(CharsOf, CharsOf, "chars_of", CharsOf, super::chars_of_apply);
string_expand_builtin!(Bytes, Bytes, "bytes", Bytes, super::bytes_of_apply);

/// `find_first(pred)` — terminal expr-arg returning first match with First demand.
pub(crate) struct FindFirst;
impl Builtin for FindFirst {
    const METHOD: BuiltinMethod = BuiltinMethod::FindFirst;
    const NAME: &'static str = "find_first";
    fn spec() -> BuiltinSpec {
        BuiltinSpec::new(
            BuiltinCategory::StreamingFilter,
            BuiltinCardinality::Filtering,
        )
        .view_native()
        .view_stage(BuiltinViewStage::Filter)
        .cost(10.0)
        .demand_law(BuiltinDemandLaw::FilterLike)
        .expr_stage(BuiltinExprStage::Filter)
        .logical_shape(BuiltinLogicalShape::FilterThenFirst)
        .lowering(BuiltinPipelineLowering::TerminalExprArg {
            terminal: BuiltinMethod::First,
        })
    }
}

/// `find_one(pred)` — terminal predicate sink requiring exactly one match.
pub(crate) struct FindOne;
impl Builtin for FindOne {
    const METHOD: BuiltinMethod = BuiltinMethod::FindOne;
    const NAME: &'static str = "find_one";
    fn spec() -> BuiltinSpec {
        BuiltinSpec::new(
            BuiltinCategory::StreamingFilter,
            BuiltinCardinality::Filtering,
        )
        .predicate_sink(BuiltinPredicateSink::FindOne)
        .cost(10.0)
        .demand_law(BuiltinDemandLaw::PredicateMapLike)
        .row_stream_op(BuiltinRowStreamOp::FindOne)
        .lowering(BuiltinPipelineLowering::TerminalSink)
    }
}

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
        let selector = BuiltinArraySelector::Nth;
        positional_native_spec()
            .array_selector(selector)
            .demand_law(selector.demand_law())
            .streaming_boundary(BuiltinStreamingBoundary::BoundedState)
            .lowering(selector.pipeline_lowering())
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::I64(n) => super::nth_any_apply(recv, *n),
            _ => None,
        }
    }
}

/// `collect()` — materialise stream to Vec; positional pass-through.
pub(crate) struct Collect;
impl Builtin for Collect {
    const METHOD: BuiltinMethod = BuiltinMethod::Collect;
    const NAME: &'static str = "collect";
    fn spec() -> BuiltinSpec {
        positional_native_spec()
    }
    #[inline]
    fn apply_one(recv: &crate::data::value::Val) -> Option<crate::data::value::Val> {
        Some(super::collect_apply(recv))
    }
}

#[inline]
fn barrier_default_spec() -> BuiltinSpec {
    BuiltinSpec::new(BuiltinCategory::Barrier, BuiltinCardinality::Barrier)
        .cost(20.0)
        .demand_law(BuiltinDemandLaw::OrderBarrier)
        .streaming_boundary(BuiltinStreamingBoundary::FullInputOrder)
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
            .order_only()
            .selection_rewrite(
                BuiltinSelectionRewrite::new()
                    .first(BuiltinMethod::Min)
                    .last(BuiltinMethod::Max)
                    .index_zero(BuiltinMethod::Min)
                    .index_minus_one(BuiltinMethod::Max),
            )
            .idempotent()
            .logical_shape(BuiltinLogicalShape::Sort)
            .runtime_hook(BuiltinRuntimeHook::Barrier)
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
                std::mem::take(buf),
                descending,
                strategy,
                |v| Ok(v.clone()),
            ),
            Some(prog) => {
                let key_prog = prog.clone();
                crate::exec::pipeline::bounded_sort_by_key(
                    std::mem::take(buf),
                    descending,
                    strategy,
                    |v| {
                        Ok(crate::exec::pipeline::eval_kernel_with_vm(
                            ctx.kernel,
                            v,
                            ctx.vm,
                            |item, vm| {
                                crate::exec::pipeline::apply_item_in_env(
                                    vm, ctx.env, item, &key_prog,
                                )
                            },
                        )
                        .unwrap_or(crate::data::value::Val::Null))
                    },
                )
            }
        };
        match result {
            Ok(sorted) => {
                *buf = sorted;
                Some(Ok(()))
            }
            Err(err) => Some(Err(err)),
        }
    }
}

/// `group_shape()` — bucket an array of objects by their structural key
/// set (sorted, comma-joined). Output is `{shape_key: [items]}` with
/// first-occurrence order preserved. The 2-arg form `group_shape(key,
/// shape)` (key projection + per-group shape transform) dispatches via
/// the lambda-method runtime path.
pub(crate) struct GroupShape;
impl Builtin for GroupShape {
    const METHOD: BuiltinMethod = BuiltinMethod::GroupShape;
    const NAME: &'static str = "group_shape";
    fn spec() -> BuiltinSpec {
        barrier_default_spec()
            .view_native()
            .view_value_projection(BuiltinViewValueProjection::GroupShape)
            .demand_law(BuiltinViewValueProjection::GroupShape.demand_law())
            .order_effect(BuiltinPipelineOrderEffect::Preserves)
    }
    #[inline]
    fn apply_one(recv: &crate::data::value::Val) -> Option<crate::data::value::Val> {
        super::group_shape_by_keys_apply(recv.clone())
    }
}

/// `partition` — splits stream by predicate; barrier.
pub(crate) struct Partition;
impl Builtin for Partition {
    const METHOD: BuiltinMethod = BuiltinMethod::Partition;
    const NAME: &'static str = "partition";
    fn spec() -> BuiltinSpec {
        barrier_default_spec().lambda_arg()
    }
}

/// `window(n)` — sliding window barrier.
pub(crate) struct Window;
impl Builtin for Window {
    const METHOD: BuiltinMethod = BuiltinMethod::Window;
    const NAME: &'static str = "window";
    fn spec() -> BuiltinSpec {
        barrier_default_spec()
            .materialization(BuiltinPipelineMaterialization::ComposedBarrier)
            .streaming_boundary(BuiltinStreamingBoundary::BoundedState)
            .view_native()
            .view_stage(BuiltinViewStage::Window)
            .pipeline_shape(BuiltinPipelineShape::new(
                BuiltinCardinality::Barrier,
                false,
                2.0,
                1.0,
            ))
            .demand_law(BuiltinDemandLaw::Window)
            .runtime_hook(BuiltinRuntimeHook::StreamAndBarrier)
            .lowering(BuiltinPipelineLowering::UsizeArg { min: 1 })
    }

    #[inline]
    fn apply_stream(
        ctx: &mut super::builtin::StreamCtx<'_, '_>,
        item: crate::data::value::Val,
        _body: Option<&crate::vm::Program>,
    ) -> Result<
        crate::exec::pipeline::StageFlow<crate::data::value::Val>,
        crate::data::context::EvalError,
    > {
        let Some(n) = ctx.stage.descriptor().and_then(|d| d.usize_arg) else {
            return Ok(crate::exec::pipeline::StageFlow::Continue(item));
        };
        let buffer = &mut ctx.stage_window_buffers[ctx.stage_idx];
        buffer.push_back(item);
        if buffer.len() < n {
            return Ok(crate::exec::pipeline::StageFlow::SkipRow);
        }
        while buffer.len() > n {
            buffer.pop_front();
        }
        let window = buffer.iter().cloned().collect();
        Ok(crate::exec::pipeline::StageFlow::Continue(
            crate::data::value::Val::arr(window),
        ))
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
            .materialization(BuiltinPipelineMaterialization::ComposedBarrier)
            .streaming_boundary(BuiltinStreamingBoundary::BoundedState)
            .view_native()
            .view_stage(BuiltinViewStage::Chunk)
            .pipeline_shape(BuiltinPipelineShape::new(
                BuiltinCardinality::Barrier,
                false,
                2.0,
                1.0,
            ))
            .demand_law(BuiltinDemandLaw::Chunk)
            .runtime_hook(BuiltinRuntimeHook::StreamAndBarrier)
            .lowering(BuiltinPipelineLowering::UsizeArg { min: 1 })
    }

    #[inline]
    fn apply_stream(
        ctx: &mut super::builtin::StreamCtx<'_, '_>,
        item: crate::data::value::Val,
        _body: Option<&crate::vm::Program>,
    ) -> Result<
        crate::exec::pipeline::StageFlow<crate::data::value::Val>,
        crate::data::context::EvalError,
    > {
        let Some(n) = ctx.stage.descriptor().and_then(|d| d.usize_arg) else {
            return Ok(crate::exec::pipeline::StageFlow::Continue(item));
        };
        let buffer = &mut ctx.stage_window_buffers[ctx.stage_idx];
        buffer.push_back(item);
        if buffer.len() < n {
            return Ok(crate::exec::pipeline::StageFlow::SkipRow);
        }
        let chunk = buffer.drain(..).collect();
        Ok(crate::exec::pipeline::StageFlow::Continue(
            crate::data::value::Val::arr(chunk),
        ))
    }

    #[inline]
    fn finish_stream(
        ctx: &mut super::builtin::StreamCtx<'_, '_>,
        _body: Option<&crate::vm::Program>,
    ) -> Result<Vec<crate::data::value::Val>, crate::data::context::EvalError> {
        let buffer = &mut ctx.stage_window_buffers[ctx.stage_idx];
        if buffer.is_empty() {
            return Ok(Vec::new());
        }
        let chunk = buffer.drain(..).collect();
        Ok(vec![crate::data::value::Val::arr(chunk)])
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

#[inline]
fn rolling_numeric_spec(op: BuiltinViewRolling) -> BuiltinSpec {
    BuiltinSpec::new(
        BuiltinCategory::StreamingOneToOne,
        BuiltinCardinality::OneToOne,
    )
    .view_native()
    .view_stage(BuiltinViewStage::Rolling(op))
    .cost(10.0)
    .demand_law(BuiltinDemandLaw::OrderBarrier)
    .runtime_hook(BuiltinRuntimeHook::StreamAndBarrier)
    .streaming_boundary(BuiltinStreamingBoundary::BoundedState)
    .pipeline_shape(BuiltinPipelineShape::new(
        BuiltinCardinality::OneToOne,
        false,
        2.0,
        1.0,
    ))
    .lowering(BuiltinPipelineLowering::UsizeArg { min: 1 })
}

#[inline]
fn rolling_stream_value(
    buffer: &mut std::collections::VecDeque<crate::data::value::Val>,
    item: crate::data::value::Val,
    width: usize,
    op: BuiltinViewRolling,
) -> crate::data::value::Val {
    buffer.push_back(opt_float_val(numeric_val(&item)));
    while buffer.len() > width {
        buffer.pop_front();
    }
    if buffer.len() < width {
        return crate::data::value::Val::Null;
    }
    let mut sum = 0.0;
    let mut count = 0usize;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for value in buffer.iter().filter_map(numeric_val) {
        sum += value;
        count += 1;
        min = min.min(value);
        max = max.max(value);
    }
    match op {
        BuiltinViewRolling::Sum => crate::data::value::Val::Float(sum),
        BuiltinViewRolling::Avg if count > 0 => crate::data::value::Val::Float(sum / count as f64),
        BuiltinViewRolling::Avg => crate::data::value::Val::Null,
        BuiltinViewRolling::Min if min.is_finite() => crate::data::value::Val::Float(min),
        BuiltinViewRolling::Min => crate::data::value::Val::Null,
        BuiltinViewRolling::Max if max.is_finite() => crate::data::value::Val::Float(max),
        BuiltinViewRolling::Max => crate::data::value::Val::Null,
    }
}

#[inline]
fn rolling_apply_stream(
    ctx: &mut super::builtin::StreamCtx<'_, '_>,
    item: crate::data::value::Val,
    op: BuiltinViewRolling,
) -> Result<
    crate::exec::pipeline::StageFlow<crate::data::value::Val>,
    crate::data::context::EvalError,
> {
    let Some(width) = ctx.stage.descriptor().and_then(|d| d.usize_arg) else {
        return Ok(crate::exec::pipeline::StageFlow::Continue(item));
    };
    if width == 0 {
        return Ok(crate::exec::pipeline::StageFlow::Continue(item));
    }
    let out = rolling_stream_value(
        &mut ctx.stage_window_buffers[ctx.stage_idx],
        item,
        width,
        op,
    );
    Ok(crate::exec::pipeline::StageFlow::Continue(out))
}

/// `rolling_sum(n)` — windowed sum barrier.
pub(crate) struct RollingSum;
impl Builtin for RollingSum {
    const METHOD: BuiltinMethod = BuiltinMethod::RollingSum;
    const NAME: &'static str = "rolling_sum";
    fn spec() -> BuiltinSpec {
        rolling_numeric_spec(BuiltinViewRolling::Sum)
    }
    #[inline]
    fn apply_stream(
        ctx: &mut super::builtin::StreamCtx<'_, '_>,
        item: crate::data::value::Val,
        _body: Option<&crate::vm::Program>,
    ) -> Result<
        crate::exec::pipeline::StageFlow<crate::data::value::Val>,
        crate::data::context::EvalError,
    > {
        rolling_apply_stream(ctx, item, BuiltinViewRolling::Sum)
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Usize(n) => super::rolling_sum_apply(recv, *n),
            _ => None,
        }
    }
}

/// `rolling_avg(n)` — windowed mean barrier.
pub(crate) struct RollingAvg;
impl Builtin for RollingAvg {
    const METHOD: BuiltinMethod = BuiltinMethod::RollingAvg;
    const NAME: &'static str = "rolling_avg";
    fn spec() -> BuiltinSpec {
        rolling_numeric_spec(BuiltinViewRolling::Avg)
    }
    #[inline]
    fn apply_stream(
        ctx: &mut super::builtin::StreamCtx<'_, '_>,
        item: crate::data::value::Val,
        _body: Option<&crate::vm::Program>,
    ) -> Result<
        crate::exec::pipeline::StageFlow<crate::data::value::Val>,
        crate::data::context::EvalError,
    > {
        rolling_apply_stream(ctx, item, BuiltinViewRolling::Avg)
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Usize(n) => super::rolling_avg_apply(recv, *n),
            _ => None,
        }
    }
}

/// `rolling_min(n)` — windowed min barrier.
pub(crate) struct RollingMin;
impl Builtin for RollingMin {
    const METHOD: BuiltinMethod = BuiltinMethod::RollingMin;
    const NAME: &'static str = "rolling_min";
    fn spec() -> BuiltinSpec {
        rolling_numeric_spec(BuiltinViewRolling::Min)
    }
    #[inline]
    fn apply_stream(
        ctx: &mut super::builtin::StreamCtx<'_, '_>,
        item: crate::data::value::Val,
        _body: Option<&crate::vm::Program>,
    ) -> Result<
        crate::exec::pipeline::StageFlow<crate::data::value::Val>,
        crate::data::context::EvalError,
    > {
        rolling_apply_stream(ctx, item, BuiltinViewRolling::Min)
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Usize(n) => super::rolling_min_apply(recv, *n),
            _ => None,
        }
    }
}

/// `rolling_max(n)` — windowed max barrier.
pub(crate) struct RollingMax;
impl Builtin for RollingMax {
    const METHOD: BuiltinMethod = BuiltinMethod::RollingMax;
    const NAME: &'static str = "rolling_max";
    fn spec() -> BuiltinSpec {
        rolling_numeric_spec(BuiltinViewRolling::Max)
    }
    #[inline]
    fn apply_stream(
        ctx: &mut super::builtin::StreamCtx<'_, '_>,
        item: crate::data::value::Val,
        _body: Option<&crate::vm::Program>,
    ) -> Result<
        crate::exec::pipeline::StageFlow<crate::data::value::Val>,
        crate::data::context::EvalError,
    > {
        rolling_apply_stream(ctx, item, BuiltinViewRolling::Max)
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Usize(n) => super::rolling_max_apply(recv, *n),
            _ => None,
        }
    }
}

/// `accumulate` — running fold barrier.
pub(crate) struct Accumulate;
impl Builtin for Accumulate {
    const METHOD: BuiltinMethod = BuiltinMethod::Accumulate;
    const NAME: &'static str = "accumulate";
    fn spec() -> BuiltinSpec {
        barrier_default_spec().lambda_arg()
    }
}

/// `fold(init, fn)` / `fold(fn)` — like `accumulate(...).last()` but
/// emits a single value instead of the running-trace array. Equivalent
/// to `Iterator::fold` (with init) or `Iterator::reduce` (without).
pub(crate) struct Fold;
impl Builtin for Fold {
    const METHOD: BuiltinMethod = BuiltinMethod::Fold;
    const NAME: &'static str = "fold";
    const ALIASES: &'static [&'static str] = &["reduce"];
    fn spec() -> BuiltinSpec {
        barrier_default_spec().lambda_arg()
    }
}

/// `group_by(key)` — keyed reducer collecting elements per key.
pub(crate) struct GroupBy;
impl Builtin for GroupBy {
    const METHOD: BuiltinMethod = BuiltinMethod::GroupBy;
    const NAME: &'static str = "group_by";
    fn spec() -> BuiltinSpec {
        BuiltinSpec::new(BuiltinCategory::Reducer, BuiltinCardinality::Reducing)
            .view_native()
            .view_stage(BuiltinViewStage::KeyedReduce)
            .keyed_reducer(BuiltinKeyedReducer::Group)
            .columnar_stage(BuiltinColumnarStage::GroupBy)
            .cost(20.0)
            .demand_law(BuiltinKeyedReducer::Group.demand_law())
            .materialization(BuiltinPipelineMaterialization::ComposedBarrier)
            .streaming_boundary(BuiltinStreamingBoundary::FullInputState)
            .logical_shape(BuiltinLogicalShape::GroupBy)
            .runtime_hook(BuiltinRuntimeHook::Barrier)
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
            crate::exec::pipeline::eval_kernel_with_vm(ctx.kernel, v, ctx.vm, |item, vm| {
                crate::exec::pipeline::apply_item_in_env(vm, ctx.env, item, prog)
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
            .view_native()
            .view_stage(BuiltinViewStage::KeyedReduce)
            .keyed_reducer(BuiltinKeyedReducer::Count)
            .cost(10.0)
            .demand_law(BuiltinKeyedReducer::Count.demand_law())
            .materialization(BuiltinPipelineMaterialization::ComposedBarrier)
            .streaming_boundary(BuiltinStreamingBoundary::FullInputState)
            .pipeline_shape(BuiltinPipelineShape::new(
                BuiltinCardinality::Reducing,
                false,
                1.0,
                1.0,
            ))
            .logical_shape(BuiltinLogicalShape::CountBy)
            .runtime_hook(BuiltinRuntimeHook::Barrier)
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
            crate::exec::pipeline::eval_kernel_with_vm(ctx.kernel, v, ctx.vm, |item, vm| {
                crate::exec::pipeline::apply_item_in_env(vm, ctx.env, item, prog)
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
            .view_native()
            .view_stage(BuiltinViewStage::KeyedReduce)
            .keyed_reducer(BuiltinKeyedReducer::Index)
            .cost(10.0)
            .demand_law(BuiltinKeyedReducer::Index.demand_law())
            .materialization(BuiltinPipelineMaterialization::ComposedBarrier)
            .streaming_boundary(BuiltinStreamingBoundary::FullInputState)
            .pipeline_shape(BuiltinPipelineShape::new(
                BuiltinCardinality::Reducing,
                false,
                1.0,
                1.0,
            ))
            .logical_shape(BuiltinLogicalShape::IndexBy)
            .runtime_hook(BuiltinRuntimeHook::Barrier)
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
            crate::exec::pipeline::eval_kernel_with_vm(ctx.kernel, v, ctx.vm, |item, vm| {
                crate::exec::pipeline::apply_item_in_env(vm, ctx.env, item, prog)
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

#[inline]
fn unique_spec() -> BuiltinSpec {
    BuiltinSpec::new(
        BuiltinCategory::StreamingFilter,
        BuiltinCardinality::Filtering,
    )
    .view_native()
    .view_stage(BuiltinViewStage::Distinct)
    .cost(10.0)
    .demand_law(BuiltinDemandLaw::UniqueLike)
    .pipeline_shape(BuiltinPipelineShape::new(
        BuiltinCardinality::Filtering,
        false,
        10.0,
        1.0,
    ))
    .order_effect(BuiltinPipelineOrderEffect::Preserves)
    .runtime_hook(BuiltinRuntimeHook::StreamAndBarrier)
    .streaming_boundary(BuiltinStreamingBoundary::FullInputState)
}

#[inline]
fn unique_apply_stream(
    ctx: &mut super::builtin::StreamCtx<'_, '_>,
    item: crate::data::value::Val,
    body: Option<&crate::vm::Program>,
) -> Result<
    crate::exec::pipeline::StageFlow<crate::data::value::Val>,
    crate::data::context::EvalError,
> {
    let key = match body {
        None => item.clone(),
        Some(prog) => {
            crate::exec::pipeline::eval_kernel_with_vm(ctx.kernel, &item, ctx.vm, |row, vm| {
                crate::exec::pipeline::apply_item_in_env(vm, ctx.env, row, prog)
            })
            .unwrap_or(crate::data::value::Val::Null)
        }
    };
    if ctx.stage_unique_seen[ctx.stage_idx].insert(&key) {
        Ok(crate::exec::pipeline::StageFlow::Continue(item))
    } else {
        Ok(crate::exec::pipeline::StageFlow::SkipRow)
    }
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
            let mut seen = crate::util::StructuralValueSet::with_capacity(buf.len());
            buf.retain(|v| seen.insert(v));
        }
        Some(prog) => {
            let mut seen = crate::util::StructuralValueSet::with_capacity(buf.len());
            let mut keep: Vec<bool> = Vec::with_capacity(buf.len());
            for v in buf.iter() {
                let key = crate::exec::pipeline::eval_kernel_with_vm(
                    ctx.kernel,
                    v,
                    ctx.vm,
                    |item, vm| crate::exec::pipeline::apply_item_in_env(vm, ctx.env, item, prog),
                )
                .unwrap_or(crate::data::value::Val::Null);
                keep.push(seen.insert(&key));
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
        unique_spec()
            .nullary_stage(BuiltinNullaryStage::Unique)
            .idempotent()
            .logical_shape(BuiltinLogicalShape::Unique)
            .lowering(BuiltinPipelineLowering::Nullary)
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
    #[inline]
    fn apply_stream(
        ctx: &mut super::builtin::StreamCtx<'_, '_>,
        item: crate::data::value::Val,
        body: Option<&crate::vm::Program>,
    ) -> Result<
        crate::exec::pipeline::StageFlow<crate::data::value::Val>,
        crate::data::context::EvalError,
    > {
        unique_apply_stream(ctx, item, body)
    }
}

/// `unique_by(key)` — distinct by projected key.
pub(crate) struct UniqueBy;
impl Builtin for UniqueBy {
    const METHOD: BuiltinMethod = BuiltinMethod::UniqueBy;
    const NAME: &'static str = "unique_by";
    const ALIASES: &'static [&'static str] = &["distinct_by"];
    fn spec() -> BuiltinSpec {
        unique_spec()
            .lowering(BuiltinPipelineLowering::ExprArg)
            .expr_stage(BuiltinExprStage::UniqueBy)
            .logical_shape(BuiltinLogicalShape::UniqueBy)
            .row_stream_op(BuiltinRowStreamOp::DistinctBy)
    }
    #[inline]
    fn apply_barrier(
        ctx: &mut super::builtin::BarrierCtx<'_>,
        buf: &mut Vec<crate::data::value::Val>,
        body: Option<&crate::vm::Program>,
    ) -> Option<Result<(), crate::data::context::EvalError>> {
        unique_apply_barrier(ctx, buf, body)
    }
    #[inline]
    fn apply_stream(
        ctx: &mut super::builtin::StreamCtx<'_, '_>,
        item: crate::data::value::Val,
        body: Option<&crate::vm::Program>,
    ) -> Result<
        crate::exec::pipeline::StageFlow<crate::data::value::Val>,
        crate::data::context::EvalError,
    > {
        unique_apply_stream(ctx, item, body)
    }
}

/// `reverse` — full-barrier order reversal; cancels with adjacent reverse.
pub(crate) struct Reverse;
impl Builtin for Reverse {
    const METHOD: BuiltinMethod = BuiltinMethod::Reverse;
    const NAME: &'static str = "reverse";
    fn spec() -> BuiltinSpec {
        BuiltinSpec::new(BuiltinCategory::Barrier, BuiltinCardinality::Barrier)
            .cost(10.0)
            .cancellation(BuiltinCancellation::SelfInverse(
                BuiltinCancelGroup::Reverse,
            ))
            .demand_law(BuiltinDemandLaw::Reverse)
            .materialization(BuiltinPipelineMaterialization::ComposedBarrier)
            .streaming_boundary(BuiltinStreamingBoundary::FullInputOrder)
            .order_only()
            .selection_rewrite(
                BuiltinSelectionRewrite::new()
                    .first(BuiltinMethod::Last)
                    .last(BuiltinMethod::First),
            )
            .nullary_stage(BuiltinNullaryStage::Reverse)
            .logical_shape(BuiltinLogicalShape::Reverse)
            .row_stream_op(BuiltinRowStreamOp::Reverse)
            .runtime_hook(BuiltinRuntimeHook::Barrier)
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

#[inline]
fn barrier_simple_spec() -> BuiltinSpec {
    BuiltinSpec::new(BuiltinCategory::Barrier, BuiltinCardinality::Barrier)
        .cost(10.0)
        .demand_law(BuiltinDemandLaw::OrderBarrier)
        .streaming_boundary(BuiltinStreamingBoundary::FullInputOrder)
}

#[inline]
fn set_filter_spec(op: BuiltinViewSetFilter) -> BuiltinSpec {
    BuiltinSpec::new(
        BuiltinCategory::StreamingFilter,
        BuiltinCardinality::Filtering,
    )
    .cost(10.0)
    .demand_law(BuiltinDemandLaw::OrderBarrier)
    .streaming_boundary(BuiltinStreamingBoundary::FullInputOrder)
    .view_native()
    .view_stage(BuiltinViewStage::SetFilter(op))
}

/// `append(arr)` — concatenates barrier.
pub(crate) struct Append;
impl Builtin for Append {
    const METHOD: BuiltinMethod = BuiltinMethod::Append;
    const NAME: &'static str = "append";
    fn spec() -> BuiltinSpec {
        barrier_simple_spec()
            .view_native()
            .view_stage(BuiltinViewStage::AppendValue)
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Val(item) => {
                Some(super::append_apply(recv, item).unwrap_or_else(|| recv.clone()))
            }
            _ => None,
        }
    }
}

/// `prepend(arr)` — prepend barrier.
pub(crate) struct Prepend;
impl Builtin for Prepend {
    const METHOD: BuiltinMethod = BuiltinMethod::Prepend;
    const NAME: &'static str = "prepend";
    fn spec() -> BuiltinSpec {
        barrier_simple_spec()
            .view_native()
            .view_stage(BuiltinViewStage::PrependValue)
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Val(item) => {
                Some(super::prepend_apply(recv, item).unwrap_or_else(|| recv.clone()))
            }
            _ => None,
        }
    }
}

/// `diff(arr)` — set difference.
pub(crate) struct Diff;
impl Builtin for Diff {
    const METHOD: BuiltinMethod = BuiltinMethod::Diff;
    const NAME: &'static str = "diff";
    fn spec() -> BuiltinSpec {
        set_filter_spec(BuiltinViewSetFilter::Diff)
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::ValVec(other) => {
                let arr_recv = recv.clone().into_vec().map(crate::data::value::Val::arr)?;
                super::diff_apply(&arr_recv, other)
            }
            _ => None,
        }
    }
}

/// `intersect(arr)` — set intersection.
pub(crate) struct Intersect;
impl Builtin for Intersect {
    const METHOD: BuiltinMethod = BuiltinMethod::Intersect;
    const NAME: &'static str = "intersect";
    fn spec() -> BuiltinSpec {
        set_filter_spec(BuiltinViewSetFilter::Intersect)
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::ValVec(other) => {
                let arr_recv = recv.clone().into_vec().map(crate::data::value::Val::arr)?;
                super::intersect_apply(&arr_recv, other)
            }
            _ => None,
        }
    }
}

/// `union(arr)` — set union.
pub(crate) struct Union;
impl Builtin for Union {
    const METHOD: BuiltinMethod = BuiltinMethod::Union;
    const NAME: &'static str = "union";
    fn spec() -> BuiltinSpec {
        barrier_simple_spec()
            .view_native()
            .view_stage(BuiltinViewStage::SetUnion)
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::ValVec(other) => {
                let arr_recv = recv.clone().into_vec().map(crate::data::value::Val::arr)?;
                super::union_apply(&arr_recv, other)
            }
            _ => None,
        }
    }
}

/// `join(sep)` — string join barrier.
pub(crate) struct Join;
impl Builtin for Join {
    const METHOD: BuiltinMethod = BuiltinMethod::Join;
    const NAME: &'static str = "join";
    fn spec() -> BuiltinSpec {
        barrier_simple_spec()
            .view_native()
            .view_stage(BuiltinViewStage::JoinString)
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Str(sep) => {
                Some(super::join_apply(recv, sep).unwrap_or_else(|| recv.clone()))
            }
            _ => None,
        }
    }
}

/// `zip(arr)` — element pairing.
pub(crate) struct Zip;
impl Builtin for Zip {
    const METHOD: BuiltinMethod = BuiltinMethod::Zip;
    const NAME: &'static str = "zip";
    fn spec() -> BuiltinSpec {
        barrier_simple_spec()
            .view_native()
            .view_stage(BuiltinViewStage::ZipStatic)
    }
}

/// `zip_longest(arr)` — pad-shorter zip.
pub(crate) struct ZipLongest;
impl Builtin for ZipLongest {
    const METHOD: BuiltinMethod = BuiltinMethod::ZipLongest;
    const NAME: &'static str = "zip_longest";
    fn spec() -> BuiltinSpec {
        barrier_simple_spec()
            .view_native()
            .view_stage(BuiltinViewStage::ZipLongestStatic)
    }
}

/// `fanout(...)` — multi-projection.
pub(crate) struct Fanout;
impl Builtin for Fanout {
    const METHOD: BuiltinMethod = BuiltinMethod::Fanout;
    const NAME: &'static str = "fanout";
    fn spec() -> BuiltinSpec {
        barrier_simple_spec()
    }
}

/// `zip_shape(...)` — two callable shapes:
///
/// - **No-arg, object receiver**: parallel-array interleave. Receiver is
///   `{k1: arr1, k2: arr2, ...}`; output is an array of objects, one per
///   index, with each key holding `arr_i[index]`. Non-array values are
///   broadcast to every row. Output length = min array length.
/// - **Named-args, any receiver**: build an object `{name0: expr0(recv),
///   name1: expr1(recv), ...}` (legacy form, dispatched separately).
pub(crate) struct ZipShape;
impl Builtin for ZipShape {
    const METHOD: BuiltinMethod = BuiltinMethod::ZipShape;
    const NAME: &'static str = "zip_shape";
    fn spec() -> BuiltinSpec {
        barrier_simple_spec()
            .view_native()
            .view_value_projection(BuiltinViewValueProjection::ZipShape)
            .demand_law(BuiltinViewValueProjection::ZipShape.demand_law())
            .order_effect(BuiltinPipelineOrderEffect::Preserves)
    }
    #[inline]
    fn apply_one(recv: &crate::data::value::Val) -> Option<crate::data::value::Val> {
        super::zip_shape_obj_apply(recv)
    }
}

#[inline]
fn object_element_spec() -> BuiltinSpec {
    // Note: NOT marked `.element()`. Methods that share this spec (`keys`,
    // `values`, `entries`) take a single object and produce a single array
    // — they are not per-element vectorisable. Marking them element-wise
    // caused the streaming pipeline to wrap their already-array result in
    // an outer `Val::Arr`, producing the `[[pairs]]` triple-wrap bug.
    BuiltinSpec::new(BuiltinCategory::Object, BuiltinCardinality::OneToOne)
        .view_native()
        .demand_law(BuiltinDemandLaw::MapLike)
        .order_effect(BuiltinPipelineOrderEffect::Preserves)
        .lowering(BuiltinPipelineLowering::Nullary)
}

/// `keys` — extract keys of an object (element-wise).
pub(crate) struct Keys;
impl Builtin for Keys {
    const METHOD: BuiltinMethod = BuiltinMethod::Keys;
    const NAME: &'static str = "keys";
    fn spec() -> BuiltinSpec {
        object_element_spec().view_object_projection(BuiltinViewObjectProjection::Keys)
    }
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
    fn spec() -> BuiltinSpec {
        object_element_spec().view_object_projection(BuiltinViewObjectProjection::Values)
    }
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
    fn spec() -> BuiltinSpec {
        object_element_spec().view_object_projection(BuiltinViewObjectProjection::Entries)
    }
    #[inline]
    fn apply_one(recv: &crate::data::value::Val) -> Option<crate::data::value::Val> {
        Some(super::entries_apply(recv))
    }
}

#[inline]
fn object_simple_spec() -> BuiltinSpec {
    BuiltinSpec::new(BuiltinCategory::Object, BuiltinCardinality::OneToOne)
        .demand_law(BuiltinDemandLaw::MapLike)
        .order_effect(BuiltinPipelineOrderEffect::Preserves)
}

/// `to_pairs` — convert object to array of `[k, v]` pairs.
pub(crate) struct ToPairs;
impl Builtin for ToPairs {
    const METHOD: BuiltinMethod = BuiltinMethod::ToPairs;
    const NAME: &'static str = "to_pairs";
    fn spec() -> BuiltinSpec {
        object_simple_spec()
            .view_native()
            .view_object_projection(BuiltinViewObjectProjection::ToPairs)
            .lowering(BuiltinPipelineLowering::Nullary)
    }
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
    fn spec() -> BuiltinSpec {
        object_simple_spec()
            .view_native()
            .view_value_projection(BuiltinViewValueProjection::FromPairs)
            .demand_law(BuiltinViewValueProjection::FromPairs.demand_law())
            .order_effect(BuiltinPipelineOrderEffect::Preserves)
    }
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
    fn spec() -> BuiltinSpec {
        object_simple_spec()
            .view_native()
            .view_value_projection(BuiltinViewValueProjection::Invert)
            .demand_law(BuiltinViewValueProjection::Invert.demand_law())
            .order_effect(BuiltinPipelineOrderEffect::Preserves)
    }
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
    fn spec() -> BuiltinSpec {
        let projection = BuiltinViewObjectProjection::Pick;
        object_simple_spec()
            .view_native()
            .view_object_projection(projection)
            .demand_law(projection.demand_law())
            .order_effect(BuiltinPipelineOrderEffect::Preserves)
            .element()
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::StrVec(keys) => super::pick_apply(recv, keys),
            _ => None,
        }
    }
}

/// `omit(...keys)` — drop given keys from object.
pub(crate) struct Omit;
impl Builtin for Omit {
    const METHOD: BuiltinMethod = BuiltinMethod::Omit;
    const NAME: &'static str = "omit";
    fn spec() -> BuiltinSpec {
        let projection = BuiltinViewObjectProjection::Omit;
        object_simple_spec()
            .view_native()
            .view_object_projection(projection)
            .demand_law(projection.demand_law())
            .order_effect(BuiltinPipelineOrderEffect::Preserves)
            .element()
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::StrVec(keys) => super::omit_apply(recv, keys),
            _ => None,
        }
    }
}

/// `merge(...objs)` — shallow merge objects.
pub(crate) struct Merge;
impl Builtin for Merge {
    const METHOD: BuiltinMethod = BuiltinMethod::Merge;
    const NAME: &'static str = "merge";
    fn spec() -> BuiltinSpec {
        object_simple_spec()
            .view_native()
            .view_value_projection(BuiltinViewValueProjection::Merge)
            .demand_law(BuiltinViewValueProjection::Merge.demand_law())
            .order_effect(BuiltinPipelineOrderEffect::Preserves)
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Val(other) => super::merge_apply(recv, other),
            _ => None,
        }
    }
}

/// `deep_merge(...objs)` — recursive merge.
pub(crate) struct DeepMerge;
impl Builtin for DeepMerge {
    const METHOD: BuiltinMethod = BuiltinMethod::DeepMerge;
    const NAME: &'static str = "deep_merge";
    fn spec() -> BuiltinSpec {
        object_simple_spec()
            .view_native()
            .view_value_projection(BuiltinViewValueProjection::DeepMerge)
            .demand_law(BuiltinViewValueProjection::DeepMerge.demand_law())
            .order_effect(BuiltinPipelineOrderEffect::Preserves)
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Val(other) => super::deep_merge_apply(recv, other),
            _ => None,
        }
    }
}

/// `defaults(...objs)` — fill-in defaults without overwriting.
pub(crate) struct Defaults;
impl Builtin for Defaults {
    const METHOD: BuiltinMethod = BuiltinMethod::Defaults;
    const NAME: &'static str = "defaults";
    fn spec() -> BuiltinSpec {
        object_simple_spec()
            .view_native()
            .view_value_projection(BuiltinViewValueProjection::Defaults)
            .demand_law(BuiltinViewValueProjection::Defaults.demand_law())
            .order_effect(BuiltinPipelineOrderEffect::Preserves)
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Val(other) => super::defaults_apply(recv, other),
            _ => None,
        }
    }
}

/// `rename({...})` — rename object keys.
pub(crate) struct Rename;
impl Builtin for Rename {
    const METHOD: BuiltinMethod = BuiltinMethod::Rename;
    const NAME: &'static str = "rename";
    fn spec() -> BuiltinSpec {
        object_simple_spec()
            .view_native()
            .view_value_projection(BuiltinViewValueProjection::Rename)
            .demand_law(BuiltinViewValueProjection::Rename.demand_law())
            .order_effect(BuiltinPipelineOrderEffect::Preserves)
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Val(other) => super::rename_apply(recv, other),
            _ => None,
        }
    }
}

/// `pivot(...)` — reshape object axes.
pub(crate) struct Pivot;
impl Builtin for Pivot {
    const METHOD: BuiltinMethod = BuiltinMethod::Pivot;
    const NAME: &'static str = "pivot";
    fn spec() -> BuiltinSpec {
        object_simple_spec().lambda_arg()
    }
}

/// `implode(sep)` — array-to-string with separator.
pub(crate) struct Implode;
impl Builtin for Implode {
    const METHOD: BuiltinMethod = BuiltinMethod::Implode;
    const NAME: &'static str = "implode";
    fn spec() -> BuiltinSpec {
        object_simple_spec()
            .view_native()
            .view_value_projection(BuiltinViewValueProjection::Implode)
            .demand_law(BuiltinViewValueProjection::Implode.demand_law())
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Str(field) => super::implode_apply(recv, field),
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
    fn spec() -> BuiltinSpec {
        let lambda = BuiltinObjectLambda::TransformKeys;
        object_lambda_spec()
            .object_lambda(lambda)
            .demand_law(lambda.demand_law())
            .runtime_hook(BuiltinRuntimeHook::StreamAndBarrier)
            .expr_payload(lambda.expr_payload())
    }

    #[inline]
    fn apply_stream(
        ctx: &mut super::builtin::StreamCtx<'_, '_>,
        item: crate::data::value::Val,
        body: Option<&crate::vm::Program>,
    ) -> Result<
        crate::exec::pipeline::StageFlow<crate::data::value::Val>,
        crate::data::context::EvalError,
    > {
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
) -> Result<
    crate::exec::pipeline::StageFlow<crate::data::value::Val>,
    crate::data::context::EvalError,
> {
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
    fn spec() -> BuiltinSpec {
        let lambda = BuiltinObjectLambda::TransformValues;
        object_lambda_spec()
            .object_lambda(lambda)
            .demand_law(lambda.demand_law())
            .runtime_hook(BuiltinRuntimeHook::StreamAndBarrier)
            .expr_payload(lambda.expr_payload())
    }
    #[inline]
    fn apply_stream(
        ctx: &mut super::builtin::StreamCtx<'_, '_>,
        item: crate::data::value::Val,
        body: Option<&crate::vm::Program>,
    ) -> Result<
        crate::exec::pipeline::StageFlow<crate::data::value::Val>,
        crate::data::context::EvalError,
    > {
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
    fn spec() -> BuiltinSpec {
        let lambda = BuiltinObjectLambda::FilterKeys;
        object_lambda_spec()
            .object_lambda(lambda)
            .demand_law(lambda.demand_law())
            .runtime_hook(BuiltinRuntimeHook::StreamAndBarrier)
            .expr_payload(lambda.expr_payload())
    }
    #[inline]
    fn apply_stream(
        ctx: &mut super::builtin::StreamCtx<'_, '_>,
        item: crate::data::value::Val,
        body: Option<&crate::vm::Program>,
    ) -> Result<
        crate::exec::pipeline::StageFlow<crate::data::value::Val>,
        crate::data::context::EvalError,
    > {
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
    fn spec() -> BuiltinSpec {
        let lambda = BuiltinObjectLambda::FilterValues;
        object_lambda_spec()
            .object_lambda(lambda)
            .demand_law(lambda.demand_law())
            .runtime_hook(BuiltinRuntimeHook::StreamAndBarrier)
            .expr_payload(lambda.expr_payload())
    }
    #[inline]
    fn apply_stream(
        ctx: &mut super::builtin::StreamCtx<'_, '_>,
        item: crate::data::value::Val,
        body: Option<&crate::vm::Program>,
    ) -> Result<
        crate::exec::pipeline::StageFlow<crate::data::value::Val>,
        crate::data::context::EvalError,
    > {
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

#[inline]
fn path_element_spec() -> BuiltinSpec {
    BuiltinSpec::new(BuiltinCategory::Path, BuiltinCardinality::OneToOne)
        .indexed()
        .demand_law(BuiltinDemandLaw::MapLike)
        .order_effect(BuiltinPipelineOrderEffect::Preserves)
        .element()
}

/// `get_path(path)` — navigate path lookup.
pub(crate) struct GetPath;
impl Builtin for GetPath {
    const METHOD: BuiltinMethod = BuiltinMethod::GetPath;
    const NAME: &'static str = "get_path";
    fn spec() -> BuiltinSpec {
        path_element_spec().view_object_projection(BuiltinViewObjectProjection::GetPath)
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Str(p) => super::get_path_apply(recv, p),
            super::BuiltinArgs::Path(path) => Some(super::get_path_impl(recv, path)),
            _ => None,
        }
    }
}

/// `del_path(path)` — remove value at path.
pub(crate) struct DelPath;
impl Builtin for DelPath {
    const METHOD: BuiltinMethod = BuiltinMethod::DelPath;
    const NAME: &'static str = "del_path";
    fn spec() -> BuiltinSpec {
        let projection = BuiltinViewValueProjection::DelPath;
        path_element_spec()
            .view_native()
            .view_value_projection(projection)
            .demand_law(projection.demand_law())
            .order_effect(BuiltinPipelineOrderEffect::Preserves)
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Str(p) => super::del_path_apply(recv, p),
            _ => None,
        }
    }
}

/// `has_path(path)` — existence test.
pub(crate) struct HasPath;
impl Builtin for HasPath {
    const METHOD: BuiltinMethod = BuiltinMethod::HasPath;
    const NAME: &'static str = "has_path";
    fn spec() -> BuiltinSpec {
        let projection = BuiltinViewObjectProjection::HasPath;
        path_element_spec()
            .view_object_projection(projection)
            .demand_law(projection.demand_law())
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Str(p) => super::has_path_apply(recv, p),
            super::BuiltinArgs::Path(path) => Some(crate::data::value::Val::Bool(
                !super::get_path_impl(recv, path).is_null(),
            )),
            _ => None,
        }
    }
}

#[inline]
fn path_indexed_spec() -> BuiltinSpec {
    BuiltinSpec::new(BuiltinCategory::Path, BuiltinCardinality::OneToOne)
        .indexed()
        .demand_law(BuiltinDemandLaw::MapLike)
        .order_effect(BuiltinPipelineOrderEffect::Preserves)
}

/// `set_path(path, val)` — write value at path.
pub(crate) struct SetPath;
impl Builtin for SetPath {
    const METHOD: BuiltinMethod = BuiltinMethod::SetPath;
    const NAME: &'static str = "set_path";
    fn spec() -> BuiltinSpec {
        let projection = BuiltinViewValueProjection::SetPath;
        path_indexed_spec()
            .view_native()
            .view_value_projection(projection)
            .demand_law(projection.demand_law())
            .order_effect(BuiltinPipelineOrderEffect::Preserves)
    }
}

/// `del_paths([...])` — bulk path removal.
pub(crate) struct DelPaths;
impl Builtin for DelPaths {
    const METHOD: BuiltinMethod = BuiltinMethod::DelPaths;
    const NAME: &'static str = "del_paths";
    fn spec() -> BuiltinSpec {
        let projection = BuiltinViewValueProjection::DelPaths;
        path_indexed_spec()
            .view_native()
            .view_value_projection(projection)
            .demand_law(projection.demand_law())
            .order_effect(BuiltinPipelineOrderEffect::Preserves)
    }
}

/// `flatten_keys` — flatten nested object into dotted keys.
pub(crate) struct FlattenKeys;
impl Builtin for FlattenKeys {
    const METHOD: BuiltinMethod = BuiltinMethod::FlattenKeys;
    const NAME: &'static str = "flatten_keys";
    fn spec() -> BuiltinSpec {
        path_indexed_spec()
            .view_native()
            .view_value_projection(BuiltinViewValueProjection::FlattenKeys)
            .demand_law(BuiltinViewValueProjection::FlattenKeys.demand_law())
            .order_effect(BuiltinPipelineOrderEffect::Preserves)
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Str(p) => super::flatten_keys_apply(recv, p),
            _ => None,
        }
    }
}

/// `unflatten_keys` — invert `flatten_keys`.
pub(crate) struct UnflattenKeys;
impl Builtin for UnflattenKeys {
    const METHOD: BuiltinMethod = BuiltinMethod::UnflattenKeys;
    const NAME: &'static str = "unflatten_keys";
    fn spec() -> BuiltinSpec {
        path_indexed_spec()
            .view_native()
            .view_value_projection(BuiltinViewValueProjection::UnflattenKeys)
            .demand_law(BuiltinViewValueProjection::UnflattenKeys.demand_law())
            .order_effect(BuiltinPipelineOrderEffect::Preserves)
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Str(p) => super::unflatten_keys_apply(recv, p),
            _ => None,
        }
    }
}

#[inline]
fn deep_simple_spec() -> BuiltinSpec {
    BuiltinSpec::new(BuiltinCategory::Deep, BuiltinCardinality::Expanding)
        .demand_law(BuiltinDemandLaw::FlatMapLike)
        .order_effect(BuiltinPipelineOrderEffect::Preserves)
        .cost(20.0)
}

/// `walk(fn)` — post-order walk.
pub(crate) struct Walk;
impl Builtin for Walk {
    const METHOD: BuiltinMethod = BuiltinMethod::Walk;
    const NAME: &'static str = "walk";
    fn spec() -> BuiltinSpec {
        deep_simple_spec()
    }
}

/// `walk_pre(fn)` — pre-order walk.
pub(crate) struct WalkPre;
impl Builtin for WalkPre {
    const METHOD: BuiltinMethod = BuiltinMethod::WalkPre;
    const NAME: &'static str = "walk_pre";
    fn spec() -> BuiltinSpec {
        deep_simple_spec()
    }
}

/// `rec(fn)` — recursive descent map.
pub(crate) struct Rec;
impl Builtin for Rec {
    const METHOD: BuiltinMethod = BuiltinMethod::Rec;
    const NAME: &'static str = "rec";
    fn spec() -> BuiltinSpec {
        deep_simple_spec()
    }
}

/// `trace_path()` — collect all paths.
pub(crate) struct TracePath;
impl Builtin for TracePath {
    const METHOD: BuiltinMethod = BuiltinMethod::TracePath;
    const NAME: &'static str = "trace_path";
    fn spec() -> BuiltinSpec {
        deep_simple_spec()
    }
}

/// `deep_find(pred)` — descend and collect all matches.
pub(crate) struct DeepFind;
impl Builtin for DeepFind {
    const METHOD: BuiltinMethod = BuiltinMethod::DeepFind;
    const NAME: &'static str = "deep_find";
    fn spec() -> BuiltinSpec {
        BuiltinSpec::new(BuiltinCategory::Deep, BuiltinCardinality::Expanding)
            .demand_law(BuiltinDemandLaw::FlatMapLike)
            .order_effect(BuiltinPipelineOrderEffect::Preserves)
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
            .demand_law(BuiltinDemandLaw::FlatMapLike)
            .order_effect(BuiltinPipelineOrderEffect::Preserves)
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
            .demand_law(BuiltinDemandLaw::FlatMapLike)
            .order_effect(BuiltinPipelineOrderEffect::Preserves)
            .structural(BuiltinStructural::DeepLike)
            .cost(20.0)
    }
}

#[inline]
fn serialization_spec() -> BuiltinSpec {
    BuiltinSpec::new(BuiltinCategory::Serialization, BuiltinCardinality::OneToOne)
        .indexed()
        .demand_law(BuiltinDemandLaw::MapLike)
        .order_effect(BuiltinPipelineOrderEffect::Preserves)
        .cost(20.0)
}

#[inline]
fn view_serialization_spec(projection: BuiltinViewValueProjection) -> BuiltinSpec {
    serialization_spec()
        .view_native()
        .view_value_projection(projection)
        .demand_law(projection.demand_law())
        .order_effect(BuiltinPipelineOrderEffect::Preserves)
}

/// `to_csv(headers?)` — CSV serialiser. Optional header-array argument
/// drives explicit column ordering with the headers as the first row.
pub(crate) struct ToCsv;
impl Builtin for ToCsv {
    const METHOD: BuiltinMethod = BuiltinMethod::ToCsv;
    const NAME: &'static str = "to_csv";
    fn spec() -> BuiltinSpec {
        view_serialization_spec(BuiltinViewValueProjection::ToCsv)
    }
    #[inline]
    fn apply_one(recv: &crate::data::value::Val) -> Option<crate::data::value::Val> {
        Some(super::to_csv_apply(recv).unwrap_or_else(|| recv.clone()))
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        if let super::BuiltinArgs::StrVec(headers) = args {
            return super::to_csv_with_headers_apply(recv, headers);
        }
        None
    }
}

/// `to_tsv(headers?)` — TSV serialiser. Same header semantics as `to_csv`.
pub(crate) struct ToTsv;
impl Builtin for ToTsv {
    const METHOD: BuiltinMethod = BuiltinMethod::ToTsv;
    const NAME: &'static str = "to_tsv";
    fn spec() -> BuiltinSpec {
        view_serialization_spec(BuiltinViewValueProjection::ToTsv)
    }
    #[inline]
    fn apply_one(recv: &crate::data::value::Val) -> Option<crate::data::value::Val> {
        Some(super::to_tsv_apply(recv).unwrap_or_else(|| recv.clone()))
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        if let super::BuiltinArgs::StrVec(headers) = args {
            return super::to_tsv_with_headers_apply(recv, headers);
        }
        None
    }
}

/// `equi_join(left, right, on)` — relational join barrier.
pub(crate) struct EquiJoin;
impl Builtin for EquiJoin {
    const METHOD: BuiltinMethod = BuiltinMethod::EquiJoin;
    const NAME: &'static str = "equi_join";
    fn spec() -> BuiltinSpec {
        BuiltinSpec::new(BuiltinCategory::Relational, BuiltinCardinality::Barrier)
            .demand_law(BuiltinDemandLaw::OrderBarrier)
            .order_effect(BuiltinPipelineOrderEffect::Blocks)
            .cost(20.0)
    }
}

/// `set(path, val)` — element-wise mutation.
pub(crate) struct Set;
impl Builtin for Set {
    const METHOD: BuiltinMethod = BuiltinMethod::Set;
    const NAME: &'static str = "set";
    fn spec() -> BuiltinSpec {
        BuiltinSpec::new(BuiltinCategory::Mutation, BuiltinCardinality::OneToOne)
            .demand_law(BuiltinDemandLaw::MapLike)
            .element()
    }
    #[inline]
    fn apply_args(
        _recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
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
        BuiltinSpec::new(BuiltinCategory::Mutation, BuiltinCardinality::OneToOne)
            .demand_law(BuiltinDemandLaw::MapLike)
            .lambda_arg()
    }
}

#[inline]
fn numeric_scan_spec(op: BuiltinViewNumericScan) -> BuiltinSpec {
    BuiltinSpec::new(
        BuiltinCategory::StreamingOneToOne,
        BuiltinCardinality::OneToOne,
    )
    .view_native()
    .view_stage(BuiltinViewStage::NumericScan(op))
    .cost(10.0)
    .demand_law(BuiltinDemandLaw::OrderBarrier)
    .runtime_hook(BuiltinRuntimeHook::StreamAndBarrier)
    .streaming_boundary(BuiltinStreamingBoundary::BoundedState)
    .pipeline_shape(BuiltinPipelineShape::new(
        BuiltinCardinality::OneToOne,
        false,
        2.0,
        1.0,
    ))
    .lowering(BuiltinPipelineLowering::Nullary)
}

#[inline]
fn numeric_full_input_spec(op: BuiltinViewNumericFullInput) -> BuiltinSpec {
    BuiltinSpec::new(
        BuiltinCategory::StreamingOneToOne,
        BuiltinCardinality::OneToOne,
    )
    .view_native()
    .view_stage(BuiltinViewStage::NumericFullInput(op))
    .cost(10.0)
    .demand_law(BuiltinDemandLaw::OrderBarrier)
    .runtime_hook(BuiltinRuntimeHook::StreamAndBarrier)
    .streaming_boundary(BuiltinStreamingBoundary::FullInputState)
    .pipeline_shape(BuiltinPipelineShape::new(
        BuiltinCardinality::OneToOne,
        false,
        2.0,
        1.0,
    ))
    .lowering(BuiltinPipelineLowering::Nullary)
}

#[inline]
fn numeric_val(item: &crate::data::value::Val) -> Option<f64> {
    match item {
        crate::data::value::Val::Int(n) => Some(*n as f64),
        crate::data::value::Val::Float(f) => Some(*f),
        _ => None,
    }
}

#[inline]
fn opt_float_val(value: Option<f64>) -> crate::data::value::Val {
    value.map_or(crate::data::value::Val::Null, crate::data::value::Val::Float)
}

#[inline]
fn numeric_scan_apply_stream(
    ctx: &mut super::builtin::StreamCtx<'_, '_>,
    item: crate::data::value::Val,
    op: BuiltinViewNumericScan,
) -> Result<
    crate::exec::pipeline::StageFlow<crate::data::value::Val>,
    crate::data::context::EvalError,
> {
    let current = numeric_val(&item);
    let buffer = &mut ctx.stage_window_buffers[ctx.stage_idx];
    let out = match op {
        BuiltinViewNumericScan::DiffWindow => {
            let previous = buffer.front().and_then(numeric_val);
            buffer.clear();
            buffer.push_back(opt_float_val(current));
            match (previous, current) {
                (Some(previous), Some(current)) => Some(current - previous),
                _ => None,
            }
        }
        BuiltinViewNumericScan::PctChange => {
            let previous = buffer.front().and_then(numeric_val);
            buffer.clear();
            buffer.push_back(opt_float_val(current));
            match (previous, current) {
                (Some(previous), Some(current)) if previous != 0.0 => {
                    Some((current - previous) / previous)
                }
                _ => None,
            }
        }
        BuiltinViewNumericScan::CumMax => {
            let best = buffer.front().and_then(numeric_val);
            let next = match (current, best) {
                (Some(current), Some(best)) => Some(current.max(best)),
                (Some(current), None) => Some(current),
                (None, best) => best,
            };
            buffer.clear();
            buffer.push_back(opt_float_val(next));
            next
        }
        BuiltinViewNumericScan::CumMin => {
            let best = buffer.front().and_then(numeric_val);
            let next = match (current, best) {
                (Some(current), Some(best)) => Some(current.min(best)),
                (Some(current), None) => Some(current),
                (None, best) => best,
            };
            buffer.clear();
            buffer.push_back(opt_float_val(next));
            next
        }
    };
    Ok(crate::exec::pipeline::StageFlow::Continue(opt_float_val(out)))
}

#[inline]
fn numeric_lag_spec() -> BuiltinSpec {
    BuiltinSpec::new(
        BuiltinCategory::StreamingOneToOne,
        BuiltinCardinality::OneToOne,
    )
    .view_native()
    .view_stage(BuiltinViewStage::Lag)
    .cost(10.0)
    .demand_law(BuiltinDemandLaw::OrderBarrier)
    .runtime_hook(BuiltinRuntimeHook::StreamAndBarrier)
    .streaming_boundary(BuiltinStreamingBoundary::BoundedState)
    .pipeline_shape(BuiltinPipelineShape::new(
        BuiltinCardinality::OneToOne,
        false,
        2.0,
        1.0,
    ))
    .lowering(BuiltinPipelineLowering::UsizeArg { min: 0 })
}

#[inline]
fn numeric_lead_spec() -> BuiltinSpec {
    BuiltinSpec::new(
        BuiltinCategory::StreamingOneToOne,
        BuiltinCardinality::OneToOne,
    )
    .view_native()
    .view_stage(BuiltinViewStage::Lead)
    .cost(10.0)
    .demand_law(BuiltinDemandLaw::OrderBarrier)
    .runtime_hook(BuiltinRuntimeHook::StreamAndBarrier)
    .streaming_boundary(BuiltinStreamingBoundary::BoundedState)
    .pipeline_shape(BuiltinPipelineShape::new(
        BuiltinCardinality::OneToOne,
        false,
        2.0,
        1.0,
    ))
    .lowering(BuiltinPipelineLowering::UsizeArg { min: 0 })
}

/// `lag(n)` — element shifted by N positions.
pub(crate) struct Lag;
impl Builtin for Lag {
    const METHOD: BuiltinMethod = BuiltinMethod::Lag;
    const NAME: &'static str = "lag";
    fn spec() -> BuiltinSpec {
        numeric_lag_spec()
    }
    #[inline]
    fn apply_stream(
        ctx: &mut super::builtin::StreamCtx<'_, '_>,
        item: crate::data::value::Val,
        _body: Option<&crate::vm::Program>,
    ) -> Result<
        crate::exec::pipeline::StageFlow<crate::data::value::Val>,
        crate::data::context::EvalError,
    > {
        let Some(n) = ctx.stage.descriptor().and_then(|d| d.usize_arg) else {
            return Ok(crate::exec::pipeline::StageFlow::Continue(item));
        };
        let current = opt_float_val(numeric_val(&item));
        if n == 0 {
            return Ok(crate::exec::pipeline::StageFlow::Continue(current));
        }
        let buffer = &mut ctx.stage_window_buffers[ctx.stage_idx];
        let out = if buffer.len() >= n {
            buffer.pop_front().unwrap_or(crate::data::value::Val::Null)
        } else {
            crate::data::value::Val::Null
        };
        buffer.push_back(current);
        Ok(crate::exec::pipeline::StageFlow::Continue(out))
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Usize(n) => super::lag_apply(recv, *n),
            _ => None,
        }
    }
}

/// `lead(n)` — element shifted forward by N positions.
pub(crate) struct Lead;
impl Builtin for Lead {
    const METHOD: BuiltinMethod = BuiltinMethod::Lead;
    const NAME: &'static str = "lead";
    fn spec() -> BuiltinSpec {
        numeric_lead_spec()
    }
    #[inline]
    fn apply_stream(
        ctx: &mut super::builtin::StreamCtx<'_, '_>,
        item: crate::data::value::Val,
        _body: Option<&crate::vm::Program>,
    ) -> Result<
        crate::exec::pipeline::StageFlow<crate::data::value::Val>,
        crate::data::context::EvalError,
    > {
        let Some(n) = ctx.stage.descriptor().and_then(|d| d.usize_arg) else {
            return Ok(crate::exec::pipeline::StageFlow::Continue(item));
        };
        let current = opt_float_val(numeric_val(&item));
        if n == 0 {
            return Ok(crate::exec::pipeline::StageFlow::Continue(current));
        }
        let seen = &mut ctx.stage_taken[ctx.stage_idx];
        *seen = seen.saturating_add(1);
        if *seen <= n {
            return Ok(crate::exec::pipeline::StageFlow::SkipRow);
        }
        Ok(crate::exec::pipeline::StageFlow::Continue(current))
    }
    #[inline]
    fn finish_stream(
        ctx: &mut super::builtin::StreamCtx<'_, '_>,
        _body: Option<&crate::vm::Program>,
    ) -> Result<Vec<crate::data::value::Val>, crate::data::context::EvalError> {
        let Some(n) = ctx.stage.descriptor().and_then(|d| d.usize_arg) else {
            return Ok(Vec::new());
        };
        if n == 0 {
            return Ok(Vec::new());
        }
        let seen = ctx.stage_taken[ctx.stage_idx];
        let tail = n.min(seen);
        Ok((0..tail).map(|_| crate::data::value::Val::Null).collect())
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Usize(n) => super::lead_apply(recv, *n),
            _ => None,
        }
    }
}

/// `diff_window(n)` — pairwise diff at lag N.
pub(crate) struct DiffWindow;
impl Builtin for DiffWindow {
    const METHOD: BuiltinMethod = BuiltinMethod::DiffWindow;
    const NAME: &'static str = "diff_window";
    fn spec() -> BuiltinSpec {
        numeric_scan_spec(BuiltinViewNumericScan::DiffWindow)
    }
    #[inline]
    fn apply_stream(
        ctx: &mut super::builtin::StreamCtx<'_, '_>,
        item: crate::data::value::Val,
        _body: Option<&crate::vm::Program>,
    ) -> Result<
        crate::exec::pipeline::StageFlow<crate::data::value::Val>,
        crate::data::context::EvalError,
    > {
        numeric_scan_apply_stream(ctx, item, BuiltinViewNumericScan::DiffWindow)
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::None => {
                Some(super::diff_window_apply(recv).unwrap_or_else(|| recv.clone()))
            }
            _ => None,
        }
    }
}

/// `pct_change(n)` — pairwise relative change at lag N.
pub(crate) struct PctChange;
impl Builtin for PctChange {
    const METHOD: BuiltinMethod = BuiltinMethod::PctChange;
    const NAME: &'static str = "pct_change";
    fn spec() -> BuiltinSpec {
        numeric_scan_spec(BuiltinViewNumericScan::PctChange)
    }
    #[inline]
    fn apply_stream(
        ctx: &mut super::builtin::StreamCtx<'_, '_>,
        item: crate::data::value::Val,
        _body: Option<&crate::vm::Program>,
    ) -> Result<
        crate::exec::pipeline::StageFlow<crate::data::value::Val>,
        crate::data::context::EvalError,
    > {
        numeric_scan_apply_stream(ctx, item, BuiltinViewNumericScan::PctChange)
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::None => {
                Some(super::pct_change_apply(recv).unwrap_or_else(|| recv.clone()))
            }
            _ => None,
        }
    }
}

/// `cummax()` — running maximum.
pub(crate) struct CumMax;
impl Builtin for CumMax {
    const METHOD: BuiltinMethod = BuiltinMethod::CumMax;
    const NAME: &'static str = "cummax";
    fn spec() -> BuiltinSpec {
        numeric_scan_spec(BuiltinViewNumericScan::CumMax)
    }
    #[inline]
    fn apply_stream(
        ctx: &mut super::builtin::StreamCtx<'_, '_>,
        item: crate::data::value::Val,
        _body: Option<&crate::vm::Program>,
    ) -> Result<
        crate::exec::pipeline::StageFlow<crate::data::value::Val>,
        crate::data::context::EvalError,
    > {
        numeric_scan_apply_stream(ctx, item, BuiltinViewNumericScan::CumMax)
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::None => {
                Some(super::cummax_apply(recv).unwrap_or_else(|| recv.clone()))
            }
            _ => None,
        }
    }
}

/// `cummin()` — running minimum.
pub(crate) struct CumMin;
impl Builtin for CumMin {
    const METHOD: BuiltinMethod = BuiltinMethod::CumMin;
    const NAME: &'static str = "cummin";
    fn spec() -> BuiltinSpec {
        numeric_scan_spec(BuiltinViewNumericScan::CumMin)
    }
    #[inline]
    fn apply_stream(
        ctx: &mut super::builtin::StreamCtx<'_, '_>,
        item: crate::data::value::Val,
        _body: Option<&crate::vm::Program>,
    ) -> Result<
        crate::exec::pipeline::StageFlow<crate::data::value::Val>,
        crate::data::context::EvalError,
    > {
        numeric_scan_apply_stream(ctx, item, BuiltinViewNumericScan::CumMin)
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::None => {
                Some(super::cummin_apply(recv).unwrap_or_else(|| recv.clone()))
            }
            _ => None,
        }
    }
}

/// `zscore()` — element standardised by mean/std.
pub(crate) struct Zscore;
impl Builtin for Zscore {
    const METHOD: BuiltinMethod = BuiltinMethod::Zscore;
    const NAME: &'static str = "zscore";
    fn spec() -> BuiltinSpec {
        numeric_full_input_spec(BuiltinViewNumericFullInput::Zscore)
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::None => {
                Some(super::zscore_apply(recv).unwrap_or_else(|| recv.clone()))
            }
            _ => None,
        }
    }
}

#[inline]
fn scalar_native_element_spec() -> BuiltinSpec {
    BuiltinSpec::new(BuiltinCategory::Scalar, BuiltinCardinality::OneToOne)
        .indexed()
        .view_native()
        .demand_law(BuiltinDemandLaw::MapLike)
        .order_effect(BuiltinPipelineOrderEffect::Preserves)
        .element()
}

#[inline]
fn scalar_native_predicate_element_spec() -> BuiltinSpec {
    BuiltinSpec::new(BuiltinCategory::Scalar, BuiltinCardinality::OneToOne)
        .indexed()
        .view_native()
        .demand_law(BuiltinDemandLaw::PredicateMapLike)
        .order_effect(BuiltinPipelineOrderEffect::Preserves)
        .element()
}

#[inline]
fn scalar_view_scalar_element_spec() -> BuiltinSpec {
    BuiltinSpec::new(BuiltinCategory::Scalar, BuiltinCardinality::OneToOne)
        .indexed()
        .view_native()
        .view_scalar()
        .demand_law(BuiltinDemandLaw::MapLike)
        .order_effect(BuiltinPipelineOrderEffect::Preserves)
        .element()
}

#[inline]
fn scalar_view_predicate_element_spec() -> BuiltinSpec {
    BuiltinSpec::new(BuiltinCategory::Scalar, BuiltinCardinality::OneToOne)
        .indexed()
        .view_native()
        .view_scalar()
        .demand_law(BuiltinDemandLaw::PredicateMapLike)
        .order_effect(BuiltinPipelineOrderEffect::Preserves)
        .element()
}

#[inline]
fn scalar_view_value_element_spec(projection: super::BuiltinViewValueProjection) -> BuiltinSpec {
    BuiltinSpec::new(BuiltinCategory::Scalar, BuiltinCardinality::OneToOne)
        .indexed()
        .view_native()
        .view_value_projection(projection)
        .demand_law(projection.demand_law())
        .order_effect(BuiltinPipelineOrderEffect::Preserves)
        .element()
}

// Native-element (no view_scalar):
// `apply` clause wraps with recv.clone() fallback so trait dispatch fully owns this method
// (no fall-through to legacy match on type mismatch).
macro_rules! scalar_native_element {
    ( $( $ty:ident => $variant:ident, $name:literal
         $( , aliases: [ $( $alias:literal ),* $(,)? ] )?
         $( , idempotent: $idempotent:literal )?
         $( , view_op: $view_op:ident )?
         $( , view_value: $view_value:ident )?
         $( , apply: $apply:ident )? ; )* ) => {
        $(
            pub(crate) struct $ty;
            impl Builtin for $ty {
                const METHOD: BuiltinMethod = BuiltinMethod::$variant;
                const NAME: &'static str = $name;
                $( const ALIASES: &'static [&'static str] = &[ $( $alias ),* ]; )?
                fn spec() -> BuiltinSpec {
                    let spec = scalar_native_element_spec()
                        $( .view_scalar_op(BuiltinViewScalarOp::$view_op) )?
                        $( .view_value_projection(super::BuiltinViewValueProjection::$view_value) )?;
                    $( let spec = if $idempotent { spec.idempotent() } else { spec }; )?
                    spec
                }
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
         , view_op: $view_op:ident
         $( , aliases: [ $( $alias:literal ),* $(,)? ] )?
         $( , idempotent: $idempotent:literal )?
         $( , apply: $apply:ident )? ; )* ) => {
        $(
            pub(crate) struct $ty;
            impl Builtin for $ty {
                const METHOD: BuiltinMethod = BuiltinMethod::$variant;
                const NAME: &'static str = $name;
                $( const ALIASES: &'static [&'static str] = &[ $( $alias ),* ]; )?
                fn spec() -> BuiltinSpec {
                    let spec = scalar_view_scalar_element_spec()
                        .view_scalar_op(BuiltinViewScalarOp::$view_op);
                    $( let spec = if $idempotent { spec.idempotent() } else { spec }; )?
                    spec
                }
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

macro_rules! scalar_view_predicate_element {
    ( $( $ty:ident => $variant:ident, $name:literal
         , view_op: $view_op:ident
         $( , aliases: [ $( $alias:literal ),* $(,)? ] )?
         $( , apply: $apply:ident )? ; )* ) => {
        $(
            pub(crate) struct $ty;
            impl Builtin for $ty {
                const METHOD: BuiltinMethod = BuiltinMethod::$variant;
                const NAME: &'static str = $name;
                $( const ALIASES: &'static [&'static str] = &[ $( $alias ),* ]; )?
                fn spec() -> BuiltinSpec {
                    scalar_view_predicate_element_spec()
                        .view_scalar_op(BuiltinViewScalarOp::$view_op)
                }
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
    Capitalize => Capitalize, "capitalize", idempotent: true, view_value: Capitalize, apply: capitalize_apply;
    TitleCase => TitleCase, "title_case", idempotent: true, view_value: TitleCase, apply: title_case_apply;
    SnakeCase => SnakeCase, "snake_case", idempotent: true, view_value: SnakeCase, apply: snake_case_apply;
    KebabCase => KebabCase, "kebab_case", idempotent: true, view_value: KebabCase, apply: kebab_case_apply;
    CamelCase => CamelCase, "camel_case", idempotent: true, view_value: CamelCase, apply: camel_case_apply;
    PascalCase => PascalCase, "pascal_case", idempotent: true, view_value: PascalCase, apply: pascal_case_apply;
    ParseFloat => ParseFloat, "parse_float", view_op: StringNoArg, apply: parse_float_apply;
    ParseBool => ParseBool, "parse_bool", view_op: StringNoArg, apply: parse_bool_apply;
    Schema => Schema, "schema", apply: schema_apply;
    Dedent => Dedent, "dedent", idempotent: true, view_value: Dedent, apply: dedent_apply;
}

pub(crate) struct Type;
impl Builtin for Type {
    const METHOD: BuiltinMethod = BuiltinMethod::Type;
    const NAME: &'static str = "type";
    fn spec() -> BuiltinSpec {
        scalar_view_scalar_element_spec().view_scalar_op(BuiltinViewScalarOp::TypeName)
    }
    #[inline]
    fn apply_one(recv: &crate::data::value::Val) -> Option<crate::data::value::Val> {
        super::type_name_apply(recv)
    }
}

pub(crate) struct ToString;
impl Builtin for ToString {
    const METHOD: BuiltinMethod = BuiltinMethod::ToString;
    const NAME: &'static str = "to_string";
    fn spec() -> BuiltinSpec {
        scalar_view_value_element_spec(super::BuiltinViewValueProjection::ToString)
    }
    #[inline]
    fn apply_one(recv: &crate::data::value::Val) -> Option<crate::data::value::Val> {
        super::to_string_apply(recv)
    }
}

pub(crate) struct ToJson;
impl Builtin for ToJson {
    const METHOD: BuiltinMethod = BuiltinMethod::ToJson;
    const NAME: &'static str = "to_json";
    fn spec() -> BuiltinSpec {
        scalar_view_value_element_spec(super::BuiltinViewValueProjection::ToJson)
    }
    #[inline]
    fn apply_one(recv: &crate::data::value::Val) -> Option<crate::data::value::Val> {
        super::to_json_apply(recv)
    }
}

/// `parse_int(radix)` — string → integer with optional radix (2–36).
/// Strips a leading `0b` / `0x` for binary / hex when the radix matches,
/// so both `"0xff".parse_int(16)` and `"ff".parse_int(16)` produce 255.
/// No-arg form is base 10.
pub(crate) struct ParseInt;
impl Builtin for ParseInt {
    const METHOD: BuiltinMethod = BuiltinMethod::ParseInt;
    const NAME: &'static str = "parse_int";
    fn spec() -> BuiltinSpec {
        scalar_native_element_spec().view_scalar_op(BuiltinViewScalarOp::ParseInt)
    }
    #[inline]
    fn apply_one(recv: &crate::data::value::Val) -> Option<crate::data::value::Val> {
        Some(super::parse_int_apply(recv).unwrap_or_else(|| recv.clone()))
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        // Radix arrives via the static-args decoder as `BuiltinArgs::Usize`
        // for a positive integer literal, or via `BuiltinArgs::I64` if
        // the parser took a different path. Anything else falls through
        // to the no-arg `apply_one` semantics (decoder hands us a base-10
        // parse).
        let radix: u32 = match args {
            super::BuiltinArgs::Usize(n) => *n as u32,
            super::BuiltinArgs::I64(n) if *n > 0 => *n as u32,
            super::BuiltinArgs::None => return None,
            _ => return None,
        };
        if !(2..=36).contains(&radix) {
            return Some(crate::data::value::Val::Null);
        }
        super::ops::string::map_str_val(recv, |s| super::parse_int_radix_str(s, radix))
    }
}

scalar_view_scalar_element! {
    Ceil => Ceil, "ceil", view_op: NumericNoArg;
    Floor => Floor, "floor", view_op: NumericNoArg;
    Round => Round, "round", view_op: NumericNoArg;
    Abs => Abs, "abs", view_op: NumericNoArg;
    Trim => Trim, "trim", view_op: StringNoArg, idempotent: true, apply: trim_apply;
    TrimLeft => TrimLeft, "trim_left", view_op: StringNoArg, aliases: ["lstrip"], idempotent: true, apply: trim_left_apply;
    TrimRight => TrimRight, "trim_right", view_op: StringNoArg, aliases: ["rstrip"], idempotent: true, apply: trim_right_apply;
    ToNumber => ToNumber, "to_number", view_op: StringNoArg;
    ToBool => ToBool, "to_bool", view_op: StringNoArg;
    IndexOf => IndexOf, "index_of", view_op: StringArg;
    LastIndexOf => LastIndexOf, "last_index_of", view_op: StringArg;
    ByteLen => ByteLen, "byte_len", view_op: StringNoArg;
}

/// `upper` — ASCII raw-json scalar capable uppercase transform.
pub(crate) struct Upper;
impl Builtin for Upper {
    const METHOD: BuiltinMethod = BuiltinMethod::Upper;
    const NAME: &'static str = "upper";
    fn spec() -> BuiltinSpec {
        let raw = BuiltinRawJsonScalar::AsciiUpper;
        scalar_view_scalar_element_spec()
            .idempotent()
            .view_scalar_op(BuiltinViewScalarOp::StringNoArg)
            .raw_json_scalar(raw)
            .demand_law(raw.demand_law())
    }
    #[inline]
    fn apply_one(recv: &crate::data::value::Val) -> Option<crate::data::value::Val> {
        Some(super::upper_apply(recv).unwrap_or_else(|| recv.clone()))
    }
}

/// `lower` — ASCII raw-json scalar capable lowercase transform.
pub(crate) struct Lower;
impl Builtin for Lower {
    const METHOD: BuiltinMethod = BuiltinMethod::Lower;
    const NAME: &'static str = "lower";
    fn spec() -> BuiltinSpec {
        let raw = BuiltinRawJsonScalar::AsciiLower;
        scalar_view_scalar_element_spec()
            .idempotent()
            .view_scalar_op(BuiltinViewScalarOp::StringNoArg)
            .raw_json_scalar(raw)
            .demand_law(raw.demand_law())
    }
    #[inline]
    fn apply_one(recv: &crate::data::value::Val) -> Option<crate::data::value::Val> {
        Some(super::lower_apply(recv).unwrap_or_else(|| recv.clone()))
    }
}

scalar_view_predicate_element! {
    IsBlank => IsBlank, "is_blank", view_op: StringNoArg;
    IsNumeric => IsNumeric, "is_numeric", view_op: StringNoArg;
    IsAlpha => IsAlpha, "is_alpha", view_op: StringNoArg;
    IsAscii => IsAscii, "is_ascii", view_op: StringNoArg;
    StartsWith => StartsWith, "starts_with", view_op: StringArg;
    EndsWith => EndsWith, "ends_with", view_op: StringArg;
    Matches => Matches, "matches", view_op: StringArg;
}

/// `slice(start, end?)` — int-range scalar element with pipeline lowering.
pub(crate) struct Slice;
impl Builtin for Slice {
    const METHOD: BuiltinMethod = BuiltinMethod::Slice;
    const NAME: &'static str = "slice";
    fn spec() -> BuiltinSpec {
        scalar_view_value_element_spec(super::BuiltinViewValueProjection::Slice)
            .pipeline_shape(BuiltinPipelineShape::new(
                BuiltinCardinality::OneToOne,
                true,
                1.0,
                1.0,
            ))
            .lowering(BuiltinPipelineLowering::IntRangeArg)
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::I64Opt { first, second } => {
                Some(super::slice_apply(recv.clone(), *first, *second))
            }
            _ => None,
        }
    }
}

#[inline]
fn scalar_string_pair_spec(
    stage: BuiltinStringPairStage,
    projection: super::BuiltinViewValueProjection,
) -> BuiltinSpec {
    scalar_view_value_element_spec(projection)
        .pipeline_shape(BuiltinPipelineShape::new(
            BuiltinCardinality::OneToOne,
            true,
            2.0,
            1.0,
        ))
        .string_pair_stage(stage)
        .lowering(BuiltinPipelineLowering::StringPairArg)
}

/// `replace(needle, with)` — single-replace string-pair scalar.
pub(crate) struct Replace;
impl Builtin for Replace {
    const METHOD: BuiltinMethod = BuiltinMethod::Replace;
    const NAME: &'static str = "replace";
    fn spec() -> BuiltinSpec {
        scalar_string_pair_spec(
            BuiltinStringPairStage::Replace { all: false },
            super::BuiltinViewValueProjection::Replace,
        )
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::StrPair { first, second } => {
                super::replace_apply(recv.clone(), first, second, false)
            }
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
        scalar_string_pair_spec(
            BuiltinStringPairStage::Replace { all: true },
            super::BuiltinViewValueProjection::ReplaceAll,
        )
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::StrPair { first, second } => {
                super::replace_apply(recv.clone(), first, second, true)
            }
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

/// `rows()` — source-lifting marker used by stream planners.
///
/// Runtime row-local dispatch intentionally leaves the receiver unchanged; the
/// planner is responsible for recognizing root `$.rows()` as a stream boundary.
pub(crate) struct Rows;
impl Builtin for Rows {
    const METHOD: BuiltinMethod = BuiltinMethod::Rows;
    const NAME: &'static str = "rows";
    fn spec() -> BuiltinSpec {
        BuiltinSpec::new(BuiltinCategory::Object, BuiltinCardinality::OneToOne)
            .stream_source()
            .streaming_boundary(BuiltinStreamingBoundary::SourceStream)
            .never_unwrap()
    }
    #[inline]
    fn apply_one(recv: &crate::data::value::Val) -> Option<crate::data::value::Val> {
        Some(recv.clone())
    }
}

// Wildcard-default methods are explicit so all methods have defs entries.

/// `from_json` — string → JSON value (default scalar element).
pub(crate) struct FromJson;
impl Builtin for FromJson {
    const METHOD: BuiltinMethod = BuiltinMethod::FromJson;
    const NAME: &'static str = "from_json";
    fn spec() -> BuiltinSpec {
        default_scalar_spec(BuiltinMethod::FromJson)
            .view_value_projection(BuiltinViewValueProjection::FromJson)
            .demand_law(BuiltinViewValueProjection::FromJson.demand_law())
    }
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
    fn spec() -> BuiltinSpec {
        default_scalar_spec(BuiltinMethod::Includes)
            .view_scalar_op(BuiltinViewScalarOp::StringContainsArg)
            .view_value_projection(super::BuiltinViewValueProjection::Includes)
            .membership_sink(BuiltinMembershipSink::Includes)
            .demand_law(BuiltinMembershipSink::Includes.demand_law())
            .lowering(BuiltinPipelineLowering::TerminalSink)
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
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
    fn spec() -> BuiltinSpec {
        default_scalar_spec(BuiltinMethod::Index)
            .membership_sink(BuiltinMembershipSink::Index)
            .demand_law(BuiltinMembershipSink::Index.demand_law())
            .lowering(BuiltinPipelineLowering::TerminalSink)
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Val(item) => {
                Some(super::index_value_apply(recv, item).unwrap_or_else(|| recv.clone()))
            }
            _ => None,
        }
    }
}

/// `indices_of(item)` — all indices of element.
pub(crate) struct IndicesOf;
impl Builtin for IndicesOf {
    const METHOD: BuiltinMethod = BuiltinMethod::IndicesOf;
    const NAME: &'static str = "indices_of";
    fn spec() -> BuiltinSpec {
        default_scalar_spec(BuiltinMethod::IndicesOf)
            .membership_sink(BuiltinMembershipSink::IndicesOf)
            .demand_law(BuiltinMembershipSink::IndicesOf.demand_law())
            .lowering(BuiltinPipelineLowering::TerminalSink)
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Val(item) => {
                Some(super::indices_of_apply(recv, item).unwrap_or_else(|| recv.clone()))
            }
            _ => None,
        }
    }
}

/// `missing(...keys)` — variadic key-existence audit. With one key,
/// returns `Bool(true)` iff the key is absent (legacy form). With one or
/// more keys passed as a string list, returns `Val::Arr<Str>` containing
/// the subset of keys that are absent or null. Empty input → `[]`; all
/// keys present → `[]`.
pub(crate) struct Missing;
impl Builtin for Missing {
    const METHOD: BuiltinMethod = BuiltinMethod::Missing;
    const NAME: &'static str = "missing";
    fn spec() -> BuiltinSpec {
        let projection = BuiltinViewObjectProjection::Missing;
        default_scalar_spec(BuiltinMethod::Missing)
            .view_object_projection(projection)
            .demand_law(projection.demand_law())
            .order_effect(BuiltinPipelineOrderEffect::Preserves)
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Str(key) => Some(super::missing_apply(recv, key)),
            super::BuiltinArgs::StrVec(keys) => Some(super::missing_many_apply(recv, keys)),
            _ => None,
        }
    }
}

/// Default scalar fallback used by methods that previously fell to the wildcard arm.
/// Mirrors the `_ => { ... }` body in legacy `BuiltinMethod::spec()`.
fn default_scalar_spec(_method: BuiltinMethod) -> BuiltinSpec {
    BuiltinSpec::new(BuiltinCategory::Scalar, BuiltinCardinality::OneToOne)
        .indexed()
        .view_native()
        .demand_law(BuiltinDemandLaw::MapLike)
        .order_effect(BuiltinPipelineOrderEffect::Preserves)
}

// Each is a scalar element with the same spec body as `scalar_native_element_spec`
// but advertises an algebraic cancellation rule used by the optimizer to fuse
// adjacent inverse pairs (e.g. `to_base64(from_base64(x))` → identity).

/// `to_base64` — Forward base64 encode.
pub(crate) struct ToBase64;
impl Builtin for ToBase64 {
    const METHOD: BuiltinMethod = BuiltinMethod::ToBase64;
    const NAME: &'static str = "to_base64";
    fn spec() -> BuiltinSpec {
        scalar_view_value_element_spec(super::BuiltinViewValueProjection::ToBase64)
    }
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
    fn spec() -> BuiltinSpec {
        scalar_view_value_element_spec(super::BuiltinViewValueProjection::FromBase64)
    }
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
    fn spec() -> BuiltinSpec {
        scalar_view_value_element_spec(super::BuiltinViewValueProjection::UrlEncode)
    }
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
    fn spec() -> BuiltinSpec {
        scalar_view_value_element_spec(super::BuiltinViewValueProjection::UrlDecode)
    }
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
    fn spec() -> BuiltinSpec {
        scalar_view_value_element_spec(super::BuiltinViewValueProjection::HtmlEscape)
    }
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
    fn spec() -> BuiltinSpec {
        scalar_view_value_element_spec(super::BuiltinViewValueProjection::HtmlUnescape)
    }
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
    fn spec() -> BuiltinSpec {
        scalar_view_value_element_spec(super::BuiltinViewValueProjection::ReverseStr)
    }
    #[inline]
    fn apply_one(recv: &crate::data::value::Val) -> Option<crate::data::value::Val> {
        Some(super::reverse_str_apply(recv).unwrap_or_else(|| recv.clone()))
    }
    #[inline]
    fn cancellation() -> Option<BuiltinCancellation> {
        Some(BuiltinCancellation::SelfInverse(
            BuiltinCancelGroup::Reverse,
        ))
    }
}

// Re-export Builtin trait constants used by cancellation impls.

/// `or(default)` — coalesce: returns recv unless null/missing, else default.
pub(crate) struct Or;
impl Builtin for Or {
    const METHOD: BuiltinMethod = BuiltinMethod::Or;
    const NAME: &'static str = "or";
    fn spec() -> BuiltinSpec {
        scalar_view_value_element_spec(super::BuiltinViewValueProjection::Or)
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Val(default) => Some(super::or_apply(recv, default)),
            _ => None,
        }
    }
}

// Multi-arg scalar element methods need apply_args.

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

pub(crate) struct StripPrefix;
impl Builtin for StripPrefix {
    const METHOD: BuiltinMethod = BuiltinMethod::StripPrefix;
    const NAME: &'static str = "strip_prefix";
    fn spec() -> BuiltinSpec {
        scalar_view_value_element_spec(super::BuiltinViewValueProjection::StripPrefix)
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Str(prefix) => {
                Some(super::strip_prefix_apply(recv, prefix).unwrap_or_else(|| recv.clone()))
            }
            _ => None,
        }
    }
}

pub(crate) struct StripSuffix;
impl Builtin for StripSuffix {
    const METHOD: BuiltinMethod = BuiltinMethod::StripSuffix;
    const NAME: &'static str = "strip_suffix";
    fn spec() -> BuiltinSpec {
        scalar_view_value_element_spec(super::BuiltinViewValueProjection::StripSuffix)
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Str(suffix) => {
                Some(super::strip_suffix_apply(recv, suffix).unwrap_or_else(|| recv.clone()))
            }
            _ => None,
        }
    }
}

str_arg_scalar_native! {
    Scan, "scan", scan_apply;
    ReMatch, "re_match", re_match_apply;
    ReMatchFirst, "match_first", re_match_first_apply;
    ReMatchAll, "match_all", re_match_all_apply;
    ReCaptures, "captures", re_captures_apply;
}

/// `has(key)` — scalar membership test. Object: key existence. Array:
/// element-wise equality. String: substring. Returns `Val::Bool` always.
/// Spec is non-element so the pipeline does not wrap the boolean result
/// in a single-element array (was the cause of `$.o.has("a")` →
/// `[true]`).
pub(crate) struct Has;
impl Builtin for Has {
    const METHOD: BuiltinMethod = BuiltinMethod::Has;
    const NAME: &'static str = "has";
    fn spec() -> BuiltinSpec {
        let projection = BuiltinViewObjectProjection::Has;
        BuiltinSpec::new(BuiltinCategory::Scalar, BuiltinCardinality::OneToOne)
            .indexed()
            .view_native()
            .view_object_projection(projection)
            .demand_law(projection.demand_law())
            .order_effect(BuiltinPipelineOrderEffect::Preserves)
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Str(p) => super::has_apply(recv, p),
            super::BuiltinArgs::Val(v) => {
                let key = crate::util::val_to_key(v);
                super::has_apply(recv, &key)
            }
            _ => None,
        }
    }
}

/// `has_all([a, b, ...])` — every literal needle is present in the receiver.
pub(crate) struct HasAll;
impl Builtin for HasAll {
    const METHOD: BuiltinMethod = BuiltinMethod::HasAll;
    const NAME: &'static str = "has_all";
    fn spec() -> BuiltinSpec {
        let projection = BuiltinViewObjectProjection::HasAll;
        BuiltinSpec::new(BuiltinCategory::Scalar, BuiltinCardinality::OneToOne)
            .indexed()
            .view_native()
            .view_object_projection(projection)
            .demand_law(projection.demand_law())
            .order_effect(BuiltinPipelineOrderEffect::Preserves)
            .element()
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Val(v) => super::has_all_apply(recv, v),
            super::BuiltinArgs::StrVec(keys) => super::has_all_keys_apply(recv, keys),
            _ => None,
        }
    }
}

/// `has_key(key)` — object key existence test with a view/tape-native backend.
pub(crate) struct HasKey;
impl Builtin for HasKey {
    const METHOD: BuiltinMethod = BuiltinMethod::HasKey;
    const NAME: &'static str = "has_key";
    fn spec() -> BuiltinSpec {
        let projection = BuiltinViewObjectProjection::HasKey;
        BuiltinSpec::new(BuiltinCategory::Scalar, BuiltinCardinality::OneToOne)
            .indexed()
            .view_native()
            .view_object_projection(projection)
            .demand_law(projection.demand_law())
            .order_effect(BuiltinPipelineOrderEffect::Preserves)
            .element()
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Str(p) => Some(super::has_key_apply(recv, p)),
            _ => None,
        }
    }
}

// Additional multi-arg scalar element methods.

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
                fn spec() -> BuiltinSpec {
                    scalar_native_predicate_element_spec()
                        .view_scalar_op(BuiltinViewScalarOp::StringVecArg)
                }
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

pub(crate) struct Repeat;
impl Builtin for Repeat {
    const METHOD: BuiltinMethod = BuiltinMethod::Repeat;
    const NAME: &'static str = "repeat";
    const ALIASES: &'static [&'static str] = &["repeat_str"];
    fn spec() -> BuiltinSpec {
        scalar_view_value_element_spec(super::BuiltinViewValueProjection::Repeat)
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Usize(n) => {
                Some(super::repeat_apply(recv, *n).unwrap_or_else(|| recv.clone()))
            }
            _ => None,
        }
    }
}

/// `indent(n_or_prefix)` — prepend each line with `n` spaces (when `n` is a
/// non-negative integer) or with the literal `prefix` string (when a string
/// is supplied). Both forms preserve trailing-newline semantics of `lines()`.
pub(crate) struct Indent;
impl Builtin for Indent {
    const METHOD: BuiltinMethod = BuiltinMethod::Indent;
    const NAME: &'static str = "indent";
    fn spec() -> BuiltinSpec {
        scalar_view_value_element_spec(super::BuiltinViewValueProjection::Indent)
    }
    #[inline]
    fn apply_args(
        recv: &crate::data::value::Val,
        args: &super::BuiltinArgs,
    ) -> Option<crate::data::value::Val> {
        match args {
            super::BuiltinArgs::Usize(n) => {
                Some(super::indent_apply(recv, *n).unwrap_or_else(|| recv.clone()))
            }
            super::BuiltinArgs::Str(prefix) => Some(
                super::indent_with_prefix_apply(recv, prefix.as_ref())
                    .unwrap_or_else(|| recv.clone()),
            ),
            _ => None,
        }
    }
}

macro_rules! pad_arg_scalar_view {
    ( $( $ty:ident, $name:literal, $projection:ident, $apply:ident ; )* ) => {
        $(
            pub(crate) struct $ty;
            impl Builtin for $ty {
                const METHOD: BuiltinMethod = BuiltinMethod::$ty;
                const NAME: &'static str = $name;
                fn spec() -> BuiltinSpec {
                    scalar_view_value_element_spec(super::BuiltinViewValueProjection::$projection)
                }
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
pad_arg_scalar_view! {
    PadLeft, "pad_left", PadLeft, pad_left_apply;
    PadRight, "pad_right", PadRight, pad_right_apply;
    Center, "center", Center, center_apply;
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
