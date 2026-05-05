//! `Builtin` trait — the per-method definition contract.
//!
//! Each builtin gets its own zero-sized struct implementing `Builtin`. The trait collects
//! all metadata (identity, spec, eventually runtime apply) in one place so a single struct
//! impl block fully describes the builtin. Static dispatch: `BuiltinMethod::spec()` matches
//! on the variant and calls `<TheStruct>::spec()` — monomorphised, no vtable.
//!
//! This replaces the prior split where:
//! - `BuiltinMethod::spec()` carried metadata in one giant match
//! - `builtin_registry!` macro carried names + aliases
//! - `executor_for_method` carried streaming classification
//! - `push_expr_stage` carried lowering
//!
//! After full migration, each builtin = one `impl Builtin for X` block.

use crate::context::{Env, EvalError};
use crate::value::Val;
use crate::vm::{Program, VM};

use super::{BuiltinCancellation, BuiltinMethod, BuiltinSpec};

/// Concrete context passed to `Builtin::apply_stream`.
/// Carries all per-element streaming state — VM, env, kernel, positional counters,
/// and the optional terminal-map collector — by mutable reference so individual
/// builtin streaming bodies can update counters and reach into VM state without
/// allocating per-call.
#[allow(dead_code)]
pub(crate) struct StreamCtx<'a, 'b> {
    pub vm: &'a mut VM,
    pub env: &'a mut Env,
    pub kernel: &'a super::super::pipeline::BodyKernel,
    pub stage: &'a super::super::pipeline::Stage,
    pub stage_idx: usize,
    pub stage_taken: &'a mut [usize],
    pub stage_skipped: &'a mut [usize],
    pub terminal_map_idx: Option<usize>,
    pub terminal_map_collect: &'a mut Option<super::super::pipeline::TerminalMapCollector<'b>>,
}

/// Concrete context passed to `Builtin::apply_barrier`.
/// Carries the materialised buffer plus VM/env/kernel/stage references so individual
/// barrier bodies can run their full-buffer transforms without allocating per-call.
#[allow(dead_code)]
pub(crate) struct BarrierCtx<'a> {
    pub vm: &'a mut VM,
    pub env: &'a mut Env,
    pub kernel: &'a super::super::pipeline::BodyKernel,
    pub stage: &'a super::super::pipeline::Stage,
    pub strategy: super::super::pipeline::StageStrategy,
}

/// Per-method definition trait. Each `BuiltinMethod` variant has a corresponding zero-sized
/// struct in `builtins::defs::*` that implements this trait.
///
/// `METHOD`, `NAME`, and `ALIASES` are reserved for later migration phases (consumer code in
/// `builtin_registry` will derive name lookups from them); `#[allow(dead_code)]` keeps the
/// foundation intact while the spec piece is migrated first.
#[allow(dead_code)]
pub(crate) trait Builtin {
    /// The `BuiltinMethod` enum variant this struct corresponds to.
    const METHOD: BuiltinMethod;

    /// Canonical name used in source code (e.g. `"filter"`).
    const NAME: &'static str;

    /// Alternate names accepted by the parser (e.g. `&["where"]` for `Filter`).
    const ALIASES: &'static [&'static str] = &[];

    /// Returns the full `BuiltinSpec` for this method.
    fn spec() -> BuiltinSpec;

    /// Algebraic cancellation rule for this method, if any.
    /// Default `None`; override for inverse pairs (Base64/Url/Html encode/decode) and
    /// self-inverses (ReverseStr). Used by the optimizer to fuse-cancel adjacent stages.
    #[inline]
    fn cancellation() -> Option<BuiltinCancellation> {
        None
    }

    /// Element-wise pure runtime apply: takes a receiver value, returns the transformed
    /// value or `None` if the receiver type is not applicable.
    /// Default `None` (caller falls back to legacy dispatch).
    /// Override on element-wise scalar methods (Upper, Lower, Trim, encoders, ...).
    #[inline]
    fn apply_one(_recv: &Val) -> Option<Val> {
        None
    }

    /// Multi-arg pure runtime apply: takes a receiver and pre-decoded args, returns the
    /// transformed value or `None` to fall through to legacy dispatch.
    /// Override on element-wise scalar methods that take literal arguments
    /// (Or/Includes/Append/Replace/Slice/Window/etc.).
    #[inline]
    fn apply_args(_recv: &Val, _args: &super::BuiltinArgs) -> Option<Val> {
        None
    }

    /// Streaming row-stage runtime: takes an item plus the per-stage `StreamCtx` and
    /// returns a `StageFlow` (Continue/SkipRow/Stop/TerminalCollected). Default returns
    /// `Continue(item)` (pass-through). Override on streaming-shaped builtins:
    /// Filter / Find / FindAll → row-predicate filter
    /// Map / FlatMap → row projection
    /// TakeWhile / DropWhile → bounded prefix predicate
    /// Take / Skip → positional slice
    /// TransformKeys / TransformValues / FilterKeys / FilterValues → object lambda
    #[inline]
    fn apply_stream(
        _ctx: &mut StreamCtx<'_, '_>,
        item: Val,
        _body: Option<&Program>,
    ) -> Result<super::super::pipeline::StageFlow<Val>, EvalError> {
        Ok(super::super::pipeline::StageFlow::Continue(item))
    }

    /// Barrier full-buffer runtime: transforms `buf` in place. Default leaves `buf`
    /// unchanged. Override on barrier-shaped builtins (Sort, Reverse, Unique, GroupBy,
    /// Window, Chunk, RowFilter materialised, RowMap materialised, RowFlatMap, ...).
    /// Returns `None` when method is not migrated to barrier dispatch (caller handles).
    #[inline]
    fn apply_barrier(
        _ctx: &mut BarrierCtx<'_>,
        _buf: &mut Vec<Val>,
        _body: Option<&Program>,
    ) -> Option<Result<(), EvalError>> {
        None
    }
}
