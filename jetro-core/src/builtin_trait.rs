//! Unified builtin descriptor and demand-law types.
//!
//! `BuiltinDescriptor` consolidates all pipeline-registry properties for a
//! single builtin into one value. `builtin_registry` uses it as the sole
//! source of truth, replacing the nine separate per-property match functions
//! that the old `builtin_registry!` macro generated.

use crate::builtins::{
    BuiltinPipelineExecutor, BuiltinPipelineLowering, BuiltinPipelineMaterialization,
    BuiltinPipelineOrderEffect, BuiltinPipelineShape, BuiltinStructural,
};

/// How a builtin transforms downstream demand into the demand it places on
/// its upstream source. Unknown builtins default to `Identity`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BuiltinDemandLaw {
    /// Pass downstream demand through unchanged (e.g. purely transforming builtins).
    Identity,
    /// Like filter: must scan until `n` outputs are produced, so converts `FirstInput(n)` to `UntilOutput(n)`.
    FilterLike,
    /// Like `take_while`: stops at the first predicate failure, so `UntilOutput(n)` becomes `FirstInput(n)`.
    TakeWhile,
    /// Like `unique`/`unique_by`: scan until enough distinct outputs are observed.
    UniqueLike,
    /// Like map: the output count equals the input count; passes demand through but requires whole values.
    MapLike,
    /// Like `flat_map`: output count is unbounded relative to input, so always requests all input.
    FlatMapLike,
    /// Cap the upstream pull to the provided count argument.
    Take,
    /// Shift the upstream pull window by the provided count argument.
    Skip,
    /// Only the first element is needed; translates any downstream demand to `FirstInput(1)`.
    First,
    /// The last element is needed; requires all ordered input.
    Last,
    /// Only a count is needed; requires all inputs but no value payloads.
    Count,
    /// A numeric aggregate (sum/min/max/avg); requires all inputs with numeric-only payload.
    NumericReducer,
    /// A predicate/keyed aggregate; requires all inputs and predicate/key evaluation.
    KeyedReducer,
    /// A full-input ordering barrier; downstream limits can choose strategy, but source scan remains all input.
    OrderBarrier,
}

/// Pipeline-registry properties for a single builtin, consolidated into one value.
///
/// Identity fields (`method`, `canonical`, `aliases`) are excluded — the caller
/// already holds the `BuiltinMethod` key, and name lookup lives in `by_name`.
/// `get_descriptor(method)` is the single dispatch point; the nine per-property
/// wrapper functions read exactly one field each.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BuiltinDescriptor {
    /// Demand-propagation law (default: `Identity`).
    pub demand_law: BuiltinDemandLaw,
    /// Streaming executor variant, if any.
    pub executor: Option<BuiltinPipelineExecutor>,
    /// Materialisation policy (default: `Streaming`).
    pub materialization: BuiltinPipelineMaterialization,
    /// Cardinality/cost shape annotation, if registered.
    pub pipeline_shape: Option<BuiltinPipelineShape>,
    /// How this builtin affects element ordering, if registered.
    pub order_effect: Option<BuiltinPipelineOrderEffect>,
    /// Physical stage lowering strategy, if registered.
    pub lowering: Option<BuiltinPipelineLowering>,
    /// Whether the builtin is element-wise vectorisable.
    pub is_element: bool,
    /// Structural traversal backend hint, if any.
    pub structural: Option<BuiltinStructural>,
}
