//! Stable numeric IDs for builtins and per-builtin demand-propagation laws.
//!
//! `BuiltinMethod` (the original enum in `builtins.rs`) remains the execution
//! identity used by the VM and pipeline. `BuiltinId` is a compact numeric
//! alias for the same set, stable across refactors, that new planner and
//! analysis code carries without depending on the legacy enum directly.

use crate::{
    builtins::{
        BuiltinCardinality, BuiltinExprStage, BuiltinMethod, BuiltinNullaryStage,
        BuiltinPipelineExecutor, BuiltinPipelineLowering, BuiltinPipelineMaterialization,
        BuiltinPipelineOrderEffect, BuiltinPipelineShape, BuiltinSinkAccumulator,
        BuiltinStringPairStage, BuiltinStringStage, BuiltinStructural, BuiltinUsizeStage,
    },
    chain_ir::{Demand, PullDemand, ValueNeed},
};

/// Compact, stable numeric identity for a builtin. One-to-one with
/// `BuiltinMethod`; used by planner/analysis to avoid re-matching names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct BuiltinId(pub(crate) u16);

/// Complete compile-time description of a single builtin, used only in tests
/// to verify round-trip consistency between names, aliases, and IDs.
#[derive(Debug, Clone, Copy)]
#[cfg(test)]
pub(crate) struct BuiltinDescriptor {
    /// Stable numeric ID for this builtin.
    pub(crate) id: BuiltinId,
    /// Canonical `BuiltinMethod` variant associated with this descriptor.
    pub(crate) method: BuiltinMethod,
    /// Primary name as it appears in Jetro expressions.
    pub(crate) canonical_name: &'static str,
    /// Alternative names that resolve to the same builtin.
    pub(crate) aliases: &'static [&'static str],
}

/// Optional numeric argument carried alongside a builtin's demand law.
/// `Take(n)` and `Skip(n)` pass their count here so `propagate_demand` can
/// tighten or loosen the upstream `PullDemand` accordingly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BuiltinDemandArg {
    /// No numeric argument; the law is applied unconditionally.
    None,
    /// A specific count (e.g. the `n` in `.take(n)` or `.skip(n)`).
    Usize(usize),
}

/// Encodes how a builtin transforms downstream demand into the demand it
/// places on its own upstream source. The planner matches each builtin to
/// exactly one law; unknown builtins default to `Identity`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BuiltinDemandLaw {
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

/// Canonical argument-count contract for pipeline lowering. This keeps
/// receiver-start checks, stage construction, and tests from re-encoding
/// per-builtin arity rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BuiltinPipelineArity {
    /// Accepts exactly N arguments.
    Exact(usize),
    /// Accepts any count in the inclusive range.
    Range { min: usize, max: usize },
}

impl BuiltinPipelineArity {
    #[inline]
    pub(crate) fn accepts(self, arity: usize) -> bool {
        match self {
            Self::Exact(n) => arity == n,
            Self::Range { min, max } => (min..=max).contains(&arity),
        }
    }
}

/// Compute the upstream `Demand` that builtin `id` must place on its source
/// given the `downstream` demand from the next stage and optional numeric `arg`.
#[inline]
pub(crate) fn propagate_demand(id: BuiltinId, arg: BuiltinDemandArg, downstream: Demand) -> Demand {
    match demand_law(id) {
        BuiltinDemandLaw::Identity => downstream,
        BuiltinDemandLaw::FilterLike => Demand {
            pull: match downstream.pull {
                PullDemand::All => PullDemand::All,
                PullDemand::FirstInput(n) | PullDemand::UntilOutput(n) => {
                    PullDemand::UntilOutput(n)
                }
            },
            value: downstream.value.merge(ValueNeed::Predicate),
            order: downstream.order,
        },
        BuiltinDemandLaw::TakeWhile => Demand {
            pull: match downstream.pull {
                PullDemand::All => PullDemand::All,
                PullDemand::FirstInput(n) | PullDemand::UntilOutput(n) => PullDemand::FirstInput(n),
            },
            value: downstream.value.merge(ValueNeed::Predicate),
            order: downstream.order,
        },
        BuiltinDemandLaw::UniqueLike => Demand {
            pull: match downstream.pull {
                PullDemand::All => PullDemand::All,
                PullDemand::FirstInput(n) | PullDemand::UntilOutput(n) => {
                    PullDemand::UntilOutput(n)
                }
            },
            value: downstream.value.merge(ValueNeed::Whole),
            order: downstream.order,
        },
        BuiltinDemandLaw::MapLike => Demand {
            value: downstream.value.merge(ValueNeed::Whole),
            ..downstream
        },
        BuiltinDemandLaw::FlatMapLike => Demand::all(ValueNeed::Whole),
        BuiltinDemandLaw::Take => match arg {
            BuiltinDemandArg::Usize(n) => Demand {
                pull: downstream.pull.cap_inputs(n),
                ..downstream
            },
            BuiltinDemandArg::None => downstream,
        },
        BuiltinDemandLaw::Skip => match arg {
            BuiltinDemandArg::Usize(n) => Demand {
                pull: match downstream.pull {
                    PullDemand::FirstInput(m) => PullDemand::FirstInput(n.saturating_add(m)),
                    PullDemand::All | PullDemand::UntilOutput(_) => PullDemand::All,
                },
                ..downstream
            },
            BuiltinDemandArg::None => downstream,
        },
        BuiltinDemandLaw::First => Demand::first(ValueNeed::Whole),
        BuiltinDemandLaw::Last => Demand {
            pull: PullDemand::All,
            value: ValueNeed::Whole,
            order: true,
        },
        BuiltinDemandLaw::Count => Demand {
            pull: PullDemand::All,
            value: ValueNeed::None,
            order: false,
        },
        BuiltinDemandLaw::NumericReducer => Demand {
            pull: PullDemand::All,
            value: ValueNeed::Numeric,
            order: false,
        },
        BuiltinDemandLaw::KeyedReducer => Demand {
            pull: PullDemand::All,
            value: ValueNeed::Predicate,
            order: false,
        },
        BuiltinDemandLaw::OrderBarrier => Demand {
            pull: PullDemand::All,
            value: downstream.value.merge(ValueNeed::Whole),
            order: true,
        },
    }
}

/// Return `true` if builtin `id` has a non-trivial demand law that can
/// restrict the amount of input the planner must pull from its source.
#[inline]
pub(crate) fn participates_in_demand(id: BuiltinId) -> bool {
    demand_law(id) != BuiltinDemandLaw::Identity
}

/// Return the pipeline executor variant for builtin `id`, or `None` if the
/// builtin has no specialised streaming executor.
#[inline]
pub(crate) fn pipeline_executor(id: BuiltinId) -> Option<BuiltinPipelineExecutor> {
    registry_pipeline_executor(id)
}

/// Return the materialization policy for builtin `id`; defaults to `Streaming`
/// when the builtin has no explicit registry entry.
#[inline]
pub(crate) fn pipeline_materialization(id: BuiltinId) -> BuiltinPipelineMaterialization {
    registry_pipeline_materialization(id).unwrap_or(BuiltinPipelineMaterialization::Streaming)
}

/// Return the cardinality/cost shape annotation for builtin `id`, used by
/// the pipeline cost estimator during plan selection.
#[inline]
pub(crate) fn pipeline_shape(id: BuiltinId) -> Option<BuiltinPipelineShape> {
    registry_pipeline_shape(id)
}

/// Return how builtin `id` affects element ordering in the pipeline, or
/// `None` if the builtin has no registered ordering behaviour.
#[inline]
pub(crate) fn pipeline_order_effect(id: BuiltinId) -> Option<BuiltinPipelineOrderEffect> {
    registry_pipeline_order_effect(id)
}

/// Return the pipeline lowering strategy for builtin `id`, indicating which
/// physical stage type and arguments the builtin compiles to.
#[inline]
pub(crate) fn pipeline_lowering(id: BuiltinId) -> Option<BuiltinPipelineLowering> {
    registry_pipeline_lowering(id)
}

/// Return `true` if builtin `id` can be lowered in pipeline position with
/// `arity` arguments. Terminal sinks are only accepted when `is_last` is true.
#[inline]
pub(crate) fn pipeline_accepts_arity(id: BuiltinId, arity: usize, is_last: bool) -> bool {
    pipeline_arity(id, is_last).is_some_and(|accepted| accepted.accepts(arity))
}

/// Return the canonical accepted pipeline arity for builtin `id`. Terminal
/// sinks are only exposed when `is_last` is true.
#[inline]
pub(crate) fn pipeline_arity(id: BuiltinId, is_last: bool) -> Option<BuiltinPipelineArity> {
    let Some(method) = id.method() else {
        return None;
    };
    match pipeline_lowering(id) {
        Some(BuiltinPipelineLowering::ExprStage(_))
        | Some(BuiltinPipelineLowering::TerminalExprStage { .. })
        | Some(BuiltinPipelineLowering::UsizeStage { .. })
        | Some(BuiltinPipelineLowering::StringStage(_)) => Some(BuiltinPipelineArity::Exact(1)),
        Some(BuiltinPipelineLowering::NullaryStage(_)) => Some(BuiltinPipelineArity::Exact(0)),
        Some(BuiltinPipelineLowering::StringPairStage(_)) => Some(BuiltinPipelineArity::Exact(2)),
        Some(BuiltinPipelineLowering::Sort) => Some(BuiltinPipelineArity::Range { min: 0, max: 1 }),
        Some(BuiltinPipelineLowering::Slice) => {
            Some(BuiltinPipelineArity::Range { min: 1, max: 2 })
        }
        Some(BuiltinPipelineLowering::TerminalSink) => {
            is_last.then(|| terminal_sink_arity(method))?
        }
        None => is_last.then(|| terminal_sink_arity(method))?,
    }
}

#[inline]
fn terminal_sink_arity(method: BuiltinMethod) -> Option<BuiltinPipelineArity> {
    let Some(sink) = method.spec().sink else {
        return None;
    };
    Some(match sink.accumulator {
        BuiltinSinkAccumulator::Count => {
            if method == BuiltinMethod::Count {
                BuiltinPipelineArity::Range { min: 0, max: 1 }
            } else {
                BuiltinPipelineArity::Exact(0)
            }
        }
        BuiltinSinkAccumulator::Numeric => BuiltinPipelineArity::Range { min: 0, max: 1 },
        BuiltinSinkAccumulator::SelectOne(_) | BuiltinSinkAccumulator::ApproxDistinct => {
            BuiltinPipelineArity::Exact(0)
        }
    })
}

/// Return `true` if builtin `id` is an element-wise operation that can be
/// applied independently to each item in a vectorised column.
#[inline]
pub(crate) fn pipeline_element(id: BuiltinId) -> bool {
    registry_pipeline_element(id).unwrap_or(false)
}

/// Return the structural traversal variant for builtin `id` (`DeepFind`,
/// `DeepShape`, `DeepLike`), or `None` for non-structural builtins.
#[inline]
pub(crate) fn structural(id: BuiltinId) -> Option<BuiltinStructural> {
    registry_structural(id)
}

/// Look up the demand law for `id`, returning `Identity` for any unregistered builtin.
#[inline]
fn demand_law(id: BuiltinId) -> BuiltinDemandLaw {
    registry_demand_law(id).unwrap_or(BuiltinDemandLaw::Identity)
}

impl BuiltinId {
    /// Construct a `BuiltinId` from a `BuiltinMethod` by casting its discriminant to `u16`.
    #[inline]
    pub(crate) fn from_method(method: BuiltinMethod) -> Self {
        BuiltinId(method as u16)
    }

    /// Resolve this `BuiltinId` back to its `BuiltinMethod`, returning `None`
    /// for IDs that do not correspond to any registered method.
    #[inline]
    pub(crate) fn method(self) -> Option<BuiltinMethod> {
        method_from_id(self)
    }
}

macro_rules! builtin_meta_demand {
    () => {
        None
    };
    (demand: $value:expr $(, $($rest:tt)*)?) => {
        Some($value)
    };
    ($key:ident : $value:expr, $($rest:tt)*) => {
        builtin_meta_demand!($($rest)*)
    };
    ($key:ident : $value:expr) => {
        None
    };
}

macro_rules! builtin_meta_executor {
    () => {
        None
    };
    (executor: $value:expr $(, $($rest:tt)*)?) => {
        Some($value)
    };
    ($key:ident : $value:expr, $($rest:tt)*) => {
        builtin_meta_executor!($($rest)*)
    };
    ($key:ident : $value:expr) => {
        None
    };
}

macro_rules! builtin_meta_materialization {
    () => {
        None
    };
    (materialization: $value:expr $(, $($rest:tt)*)?) => {
        Some($value)
    };
    ($key:ident : $value:expr, $($rest:tt)*) => {
        builtin_meta_materialization!($($rest)*)
    };
    ($key:ident : $value:expr) => {
        None
    };
}

macro_rules! builtin_meta_shape {
    () => {
        None
    };
    (shape: $value:expr $(, $($rest:tt)*)?) => {
        Some($value)
    };
    ($key:ident : $value:expr, $($rest:tt)*) => {
        builtin_meta_shape!($($rest)*)
    };
    ($key:ident : $value:expr) => {
        None
    };
}

macro_rules! builtin_meta_order {
    () => {
        None
    };
    (order: $value:expr $(, $($rest:tt)*)?) => {
        Some($value)
    };
    ($key:ident : $value:expr, $($rest:tt)*) => {
        builtin_meta_order!($($rest)*)
    };
    ($key:ident : $value:expr) => {
        None
    };
}

macro_rules! builtin_meta_lowering {
    () => {
        None
    };
    (lowering: $value:expr $(, $($rest:tt)*)?) => {
        Some($value)
    };
    ($key:ident : $value:expr, $($rest:tt)*) => {
        builtin_meta_lowering!($($rest)*)
    };
    ($key:ident : $value:expr) => {
        None
    };
}

macro_rules! builtin_meta_element {
    () => {
        None
    };
    (element: $value:expr $(, $($rest:tt)*)?) => {
        Some($value)
    };
    ($key:ident : $value:expr, $($rest:tt)*) => {
        builtin_meta_element!($($rest)*)
    };
    ($key:ident : $value:expr) => {
        None
    };
}

macro_rules! builtin_meta_structural {
    () => {
        None
    };
    (structural: $value:expr $(, $($rest:tt)*)?) => {
        Some($value)
    };
    ($key:ident : $value:expr, $($rest:tt)*) => {
        builtin_meta_structural!($($rest)*)
    };
    ($key:ident : $value:expr) => {
        None
    };
}

macro_rules! builtin_registry {
    ($( $method:ident => $canonical:literal [ $( $alias:literal ),* $(,)? ] $( { $($meta:tt)* } )? ; )*) => {
        #[cfg(test)]
        pub(crate) static BUILTIN_DESCRIPTORS: &[BuiltinDescriptor] = &[
            $(
                BuiltinDescriptor {
                    id: BuiltinId(BuiltinMethod::$method as u16),
                    method: BuiltinMethod::$method,
                    canonical_name: $canonical,
                    aliases: &[$($alias),*],
                },
            )*
        ];

        #[inline]
        pub(crate) fn method_from_id(id: BuiltinId) -> Option<BuiltinMethod> {
            match id.0 {
                $(x if x == BuiltinMethod::$method as u16 => Some(BuiltinMethod::$method),)*
                _ => None,
            }
        }

        #[inline]
        #[cfg(test)]
        pub(crate) fn descriptor(id: BuiltinId) -> Option<BuiltinDescriptor> {
            match id.0 {
                $(x if x == BuiltinMethod::$method as u16 => Some(BuiltinDescriptor {
                    id,
                    method: BuiltinMethod::$method,
                    canonical_name: $canonical,
                    aliases: &[$($alias),*],
                }),)*
                _ => None,
            }
        }

        #[inline]
        pub(crate) fn by_name(name: &str) -> Option<BuiltinId> {
            match name {
                $($canonical $(| $alias)* => Some(BuiltinId(BuiltinMethod::$method as u16)),)*
                _ => None,
            }
        }

        #[inline]
        fn registry_demand_law(id: BuiltinId) -> Option<BuiltinDemandLaw> {
            match id.0 {
                $(x if x == BuiltinMethod::$method as u16 => builtin_meta_demand!($($($meta)*)?),)*
                _ => None,
            }
        }

        #[inline]
        fn registry_pipeline_executor(id: BuiltinId) -> Option<BuiltinPipelineExecutor> {
            match id.0 {
                $(x if x == BuiltinMethod::$method as u16 => builtin_meta_executor!($($($meta)*)?),)*
                _ => None,
            }
        }

        #[inline]
        fn registry_pipeline_materialization(
            id: BuiltinId,
        ) -> Option<BuiltinPipelineMaterialization> {
            match id.0 {
                $(x if x == BuiltinMethod::$method as u16 => builtin_meta_materialization!($($($meta)*)?),)*
                _ => None,
            }
        }

        #[inline]
        fn registry_pipeline_shape(id: BuiltinId) -> Option<BuiltinPipelineShape> {
            match id.0 {
                $(x if x == BuiltinMethod::$method as u16 => builtin_meta_shape!($($($meta)*)?),)*
                _ => None,
            }
        }

        #[inline]
        fn registry_pipeline_order_effect(id: BuiltinId) -> Option<BuiltinPipelineOrderEffect> {
            match id.0 {
                $(x if x == BuiltinMethod::$method as u16 => builtin_meta_order!($($($meta)*)?),)*
                _ => None,
            }
        }

        #[inline]
        fn registry_pipeline_lowering(id: BuiltinId) -> Option<BuiltinPipelineLowering> {
            match id.0 {
                $(x if x == BuiltinMethod::$method as u16 => builtin_meta_lowering!($($($meta)*)?),)*
                _ => None,
            }
        }

        #[inline]
        fn registry_pipeline_element(id: BuiltinId) -> Option<bool> {
            match id.0 {
                $(x if x == BuiltinMethod::$method as u16 => builtin_meta_element!($($($meta)*)?),)*
                _ => None,
            }
        }

        #[inline]
        fn registry_structural(id: BuiltinId) -> Option<BuiltinStructural> {
            match id.0 {
                $(x if x == BuiltinMethod::$method as u16 => builtin_meta_structural!($($($meta)*)?),)*
                _ => None,
            }
        }
    };
}

builtin_registry! {
    Len => "len" [];
    Keys => "keys" [] { element: true };
    Values => "values" [] { element: true };
    Entries => "entries" [] { element: true };
    ToPairs => "to_pairs" [];
    FromPairs => "from_pairs" [];
    Invert => "invert" [];
    Reverse => "reverse" [] {
        demand: BuiltinDemandLaw::OrderBarrier,
        executor: BuiltinPipelineExecutor::Reverse,
        materialization: BuiltinPipelineMaterialization::ComposedBarrier,
        lowering: BuiltinPipelineLowering::NullaryStage(BuiltinNullaryStage::Reverse)
    };
    Type => "type" [] { element: true };
    ToString => "to_string" [] { element: true };
    ToJson => "to_json" [] { element: true };
    FromJson => "from_json" [];
    Sum => "sum" [] {
        demand: BuiltinDemandLaw::NumericReducer,
        lowering: BuiltinPipelineLowering::TerminalSink
    };
    Avg => "avg" [] {
        demand: BuiltinDemandLaw::NumericReducer,
        lowering: BuiltinPipelineLowering::TerminalSink
    };
    Min => "min" [] {
        demand: BuiltinDemandLaw::NumericReducer,
        lowering: BuiltinPipelineLowering::TerminalSink
    };
    Max => "max" [] {
        demand: BuiltinDemandLaw::NumericReducer,
        lowering: BuiltinPipelineLowering::TerminalSink
    };
    Count => "count" [] {
        demand: BuiltinDemandLaw::Count,
        lowering: BuiltinPipelineLowering::TerminalSink
    };
    Any => "any" ["exists"];
    All => "all" [];
    FindIndex => "find_index" [] {
        executor: BuiltinPipelineExecutor::FindIndex,
        materialization: BuiltinPipelineMaterialization::LegacyMaterialized,
        shape: BuiltinPipelineShape::new(BuiltinCardinality::OneToOne, true, 1.0, 1.0),
        lowering: BuiltinPipelineLowering::TerminalExprStage {
            stage: BuiltinExprStage::FindIndex,
            terminal: BuiltinMethod::First,
        }
    };
    IndicesWhere => "indices_where" [] {
        executor: BuiltinPipelineExecutor::IndicesWhere,
        materialization: BuiltinPipelineMaterialization::LegacyMaterialized,
        shape: BuiltinPipelineShape::new(BuiltinCardinality::OneToOne, true, 1.0, 1.0),
        lowering: BuiltinPipelineLowering::TerminalExprStage {
            stage: BuiltinExprStage::IndicesWhere,
            terminal: BuiltinMethod::First,
        }
    };
    MaxBy => "max_by" [] {
        executor: BuiltinPipelineExecutor::ArgExtreme { max: true },
        materialization: BuiltinPipelineMaterialization::LegacyMaterialized,
        shape: BuiltinPipelineShape::new(BuiltinCardinality::OneToOne, true, 1.0, 1.0),
        lowering: BuiltinPipelineLowering::TerminalExprStage {
            stage: BuiltinExprStage::MaxBy,
            terminal: BuiltinMethod::First,
        }
    };
    MinBy => "min_by" [] {
        executor: BuiltinPipelineExecutor::ArgExtreme { max: false },
        materialization: BuiltinPipelineMaterialization::LegacyMaterialized,
        shape: BuiltinPipelineShape::new(BuiltinCardinality::OneToOne, true, 1.0, 1.0),
        lowering: BuiltinPipelineLowering::TerminalExprStage {
            stage: BuiltinExprStage::MinBy,
            terminal: BuiltinMethod::First,
        }
    };
    GroupBy => "group_by" [] {
        demand: BuiltinDemandLaw::KeyedReducer,
        executor: BuiltinPipelineExecutor::GroupBy,
        materialization: BuiltinPipelineMaterialization::ComposedBarrier,
        lowering: BuiltinPipelineLowering::ExprStage(BuiltinExprStage::GroupBy)
    };
    CountBy => "count_by" [] {
        demand: BuiltinDemandLaw::KeyedReducer,
        executor: BuiltinPipelineExecutor::CountBy,
        materialization: BuiltinPipelineMaterialization::Streaming,
        shape: BuiltinPipelineShape::new(BuiltinCardinality::OneToOne, true, 1.0, 1.0),
        lowering: BuiltinPipelineLowering::TerminalExprStage {
            stage: BuiltinExprStage::CountBy,
            terminal: BuiltinMethod::First,
        }
    };
    IndexBy => "index_by" [] {
        demand: BuiltinDemandLaw::KeyedReducer,
        executor: BuiltinPipelineExecutor::IndexBy,
        materialization: BuiltinPipelineMaterialization::Streaming,
        shape: BuiltinPipelineShape::new(BuiltinCardinality::OneToOne, true, 1.0, 1.0),
        lowering: BuiltinPipelineLowering::TerminalExprStage {
            stage: BuiltinExprStage::IndexBy,
            terminal: BuiltinMethod::First,
        }
    };
    GroupShape => "group_shape" [];
    Explode => "explode" [];
    Implode => "implode" [];
    Filter => "filter" [] {
        demand: BuiltinDemandLaw::FilterLike,
        executor: BuiltinPipelineExecutor::RowFilter,
        order: BuiltinPipelineOrderEffect::PredicatePrefix,
        lowering: BuiltinPipelineLowering::ExprStage(BuiltinExprStage::Filter)
    };
    Map => "map" [] {
        demand: BuiltinDemandLaw::MapLike,
        executor: BuiltinPipelineExecutor::RowMap,
        order: BuiltinPipelineOrderEffect::Preserves,
        lowering: BuiltinPipelineLowering::ExprStage(BuiltinExprStage::Map),
        element: true
    };
    FlatMap => "flat_map" [] {
        demand: BuiltinDemandLaw::FlatMapLike,
        executor: BuiltinPipelineExecutor::RowFlatMap,
        materialization: BuiltinPipelineMaterialization::LegacyMaterialized,
        lowering: BuiltinPipelineLowering::ExprStage(BuiltinExprStage::FlatMap)
    };
    Find => "find" [] {
        demand: BuiltinDemandLaw::FilterLike,
        executor: BuiltinPipelineExecutor::RowFilter,
        order: BuiltinPipelineOrderEffect::PredicatePrefix,
        lowering: BuiltinPipelineLowering::ExprStage(BuiltinExprStage::Filter)
    };
    FindAll => "find_all" [] {
        demand: BuiltinDemandLaw::FilterLike,
        executor: BuiltinPipelineExecutor::RowFilter,
        order: BuiltinPipelineOrderEffect::PredicatePrefix,
        lowering: BuiltinPipelineLowering::ExprStage(BuiltinExprStage::Filter)
    };
    Sort => "sort" ["sort_by"] {
        demand: BuiltinDemandLaw::OrderBarrier,
        executor: BuiltinPipelineExecutor::Sort,
        materialization: BuiltinPipelineMaterialization::ComposedBarrier,
        lowering: BuiltinPipelineLowering::Sort
    };
    Unique => "unique" ["distinct"] {
        demand: BuiltinDemandLaw::UniqueLike,
        executor: BuiltinPipelineExecutor::UniqueBy,
        materialization: BuiltinPipelineMaterialization::Streaming,
        shape: BuiltinPipelineShape::new(BuiltinCardinality::Filtering, true, 10.0, 1.0),
        order: BuiltinPipelineOrderEffect::Preserves,
        lowering: BuiltinPipelineLowering::NullaryStage(BuiltinNullaryStage::Unique)
    };
    UniqueBy => "unique_by" [] {
        demand: BuiltinDemandLaw::UniqueLike,
        executor: BuiltinPipelineExecutor::UniqueBy,
        materialization: BuiltinPipelineMaterialization::Streaming,
        shape: BuiltinPipelineShape::new(BuiltinCardinality::Filtering, true, 10.0, 1.0),
        order: BuiltinPipelineOrderEffect::Preserves,
        lowering: BuiltinPipelineLowering::ExprStage(BuiltinExprStage::UniqueBy)
    };
    Collect => "collect" [];
    DeepFind => "deep_find" [] {
        structural: BuiltinStructural::DeepFind
    };
    DeepShape => "deep_shape" [] {
        structural: BuiltinStructural::DeepShape
    };
    DeepLike => "deep_like" [] {
        structural: BuiltinStructural::DeepLike
    };
    Walk => "walk" [];
    WalkPre => "walk_pre" [];
    Rec => "rec" [];
    TracePath => "trace_path" [];
    Flatten => "flatten" [];
    Compact => "compact" [];
    Join => "join" [];
    First => "first" [] {
        demand: BuiltinDemandLaw::First,
        lowering: BuiltinPipelineLowering::TerminalSink
    };
    Last => "last" [] {
        demand: BuiltinDemandLaw::Last,
        lowering: BuiltinPipelineLowering::TerminalSink
    };
    Nth => "nth" [];
    Take => "take" [] {
        demand: BuiltinDemandLaw::Take,
        executor: BuiltinPipelineExecutor::Position { take: true },
        order: BuiltinPipelineOrderEffect::Preserves,
        lowering: BuiltinPipelineLowering::UsizeStage {
            stage: BuiltinUsizeStage::Take,
            min: 0,
        }
    };
    Skip => "skip" ["drop"] {
        demand: BuiltinDemandLaw::Skip,
        executor: BuiltinPipelineExecutor::Position { take: false },
        order: BuiltinPipelineOrderEffect::Preserves,
        lowering: BuiltinPipelineLowering::UsizeStage {
            stage: BuiltinUsizeStage::Skip,
            min: 0,
        }
    };
    Append => "append" [];
    Prepend => "prepend" [];
    Remove => "remove" [];
    Diff => "diff" [];
    Intersect => "intersect" [];
    Union => "union" [];
    Enumerate => "enumerate" [] { element: true };
    Pairwise => "pairwise" [] { element: true };
    Window => "window" [] {
        executor: BuiltinPipelineExecutor::Window,
        materialization: BuiltinPipelineMaterialization::LegacyMaterialized,
        shape: BuiltinPipelineShape::new(BuiltinCardinality::Barrier, true, 2.0, 1.0),
        lowering: BuiltinPipelineLowering::UsizeStage {
            stage: BuiltinUsizeStage::Window,
            min: 1,
        }
    };
    Chunk => "chunk" ["batch"] {
        executor: BuiltinPipelineExecutor::Chunk,
        materialization: BuiltinPipelineMaterialization::LegacyMaterialized,
        shape: BuiltinPipelineShape::new(BuiltinCardinality::Barrier, true, 2.0, 1.0),
        lowering: BuiltinPipelineLowering::UsizeStage {
            stage: BuiltinUsizeStage::Chunk,
            min: 1,
        }
    };
    TakeWhile => "take_while" ["takewhile"] {
        demand: BuiltinDemandLaw::TakeWhile,
        executor: BuiltinPipelineExecutor::PrefixWhile { take: true },
        materialization: BuiltinPipelineMaterialization::Streaming,
        shape: BuiltinPipelineShape::new(BuiltinCardinality::Filtering, true, 10.0, 0.5),
        order: BuiltinPipelineOrderEffect::PredicatePrefix,
        lowering: BuiltinPipelineLowering::ExprStage(BuiltinExprStage::TakeWhile)
    };
    DropWhile => "drop_while" ["dropwhile"] {
        executor: BuiltinPipelineExecutor::PrefixWhile { take: false },
        materialization: BuiltinPipelineMaterialization::LegacyMaterialized,
        shape: BuiltinPipelineShape::new(BuiltinCardinality::Filtering, true, 10.0, 0.5),
        order: BuiltinPipelineOrderEffect::Blocks,
        lowering: BuiltinPipelineLowering::ExprStage(BuiltinExprStage::DropWhile)
    };
    FindFirst => "find_first" [] {
        demand: BuiltinDemandLaw::First,
        lowering: BuiltinPipelineLowering::TerminalExprStage {
            stage: BuiltinExprStage::Filter,
            terminal: BuiltinMethod::First,
        }
    };
    FindOne => "find_one" [] {
        lowering: BuiltinPipelineLowering::TerminalExprStage {
            stage: BuiltinExprStage::Filter,
            terminal: BuiltinMethod::First,
        }
    };
    ApproxCountDistinct => "approx_count_distinct" [] {
        demand: BuiltinDemandLaw::KeyedReducer,
        lowering: BuiltinPipelineLowering::TerminalSink
    };
    Accumulate => "accumulate" [];
    Partition => "partition" [];
    Zip => "zip" [];
    ZipLongest => "zip_longest" [];
    Fanout => "fanout" [];
    ZipShape => "zip_shape" [];
    Pick => "pick" [];
    Omit => "omit" [];
    Merge => "merge" [];
    DeepMerge => "deep_merge" [];
    Defaults => "defaults" [];
    Rename => "rename" [];
    TransformKeys => "transform_keys" [] {
        executor: BuiltinPipelineExecutor::ObjectLambda,
        shape: BuiltinPipelineShape::new(BuiltinCardinality::OneToOne, true, 1.0, 1.0),
        order: BuiltinPipelineOrderEffect::Preserves,
        lowering: BuiltinPipelineLowering::ExprStage(BuiltinExprStage::TransformKeys)
    };
    TransformValues => "transform_values" [] {
        executor: BuiltinPipelineExecutor::ObjectLambda,
        shape: BuiltinPipelineShape::new(BuiltinCardinality::OneToOne, true, 1.0, 1.0),
        order: BuiltinPipelineOrderEffect::Preserves,
        lowering: BuiltinPipelineLowering::ExprStage(BuiltinExprStage::TransformValues)
    };
    FilterKeys => "filter_keys" [] {
        executor: BuiltinPipelineExecutor::ObjectLambda,
        shape: BuiltinPipelineShape::new(BuiltinCardinality::OneToOne, true, 1.0, 1.0),
        order: BuiltinPipelineOrderEffect::Preserves,
        lowering: BuiltinPipelineLowering::ExprStage(BuiltinExprStage::FilterKeys)
    };
    FilterValues => "filter_values" [] {
        executor: BuiltinPipelineExecutor::ObjectLambda,
        shape: BuiltinPipelineShape::new(BuiltinCardinality::OneToOne, true, 1.0, 1.0),
        order: BuiltinPipelineOrderEffect::Preserves,
        lowering: BuiltinPipelineLowering::ExprStage(BuiltinExprStage::FilterValues)
    };
    Pivot => "pivot" [];
    GetPath => "get_path" [] { element: true };
    SetPath => "set_path" [];
    DelPath => "del_path" [] { element: true };
    DelPaths => "del_paths" [];
    HasPath => "has_path" [] { element: true };
    FlattenKeys => "flatten_keys" [];
    UnflattenKeys => "unflatten_keys" [];
    ToCsv => "to_csv" [];
    ToTsv => "to_tsv" [];
    Or => "or" [] { element: true };
    Has => "has" [] { element: true };
    Missing => "missing" [];
    Includes => "includes" ["contains"];
    Index => "index" [];
    IndicesOf => "indices_of" [];
    Set => "set" [] { element: true };
    Update => "update" [];
    Ceil => "ceil" [] { element: true };
    Floor => "floor" [] { element: true };
    Round => "round" [] { element: true };
    Abs => "abs" [] { element: true };
    RollingSum => "rolling_sum" [];
    RollingAvg => "rolling_avg" [];
    RollingMin => "rolling_min" [];
    RollingMax => "rolling_max" [];
    Lag => "lag" [] { element: true };
    Lead => "lead" [] { element: true };
    DiffWindow => "diff_window" [] { element: true };
    PctChange => "pct_change" [] { element: true };
    CumMax => "cummax" [] { element: true };
    CumMin => "cummin" [] { element: true };
    Zscore => "zscore" [] { element: true };
    Upper => "upper" [] { element: true };
    Lower => "lower" [] { element: true };
    Capitalize => "capitalize" [] { element: true };
    TitleCase => "title_case" [] { element: true };
    Trim => "trim" [] { element: true };
    TrimLeft => "trim_left" ["lstrip"] { element: true };
    TrimRight => "trim_right" ["rstrip"] { element: true };
    SnakeCase => "snake_case" [] { element: true };
    KebabCase => "kebab_case" [] { element: true };
    CamelCase => "camel_case" [] { element: true };
    PascalCase => "pascal_case" [] { element: true };
    ReverseStr => "reverse_str" [] { element: true };
    Lines => "lines" [] { element: true };
    Words => "words" [] { element: true };
    Chars => "chars" [] { element: true };
    CharsOf => "chars_of" [] { element: true };
    Bytes => "bytes" [] { element: true };
    ByteLen => "byte_len" [] { element: true };
    IsBlank => "is_blank" [] { element: true };
    IsNumeric => "is_numeric" [] { element: true };
    IsAlpha => "is_alpha" [] { element: true };
    IsAscii => "is_ascii" [] { element: true };
    ToNumber => "to_number" [] { element: true };
    ToBool => "to_bool" [] { element: true };
    ParseInt => "parse_int" [] { element: true };
    ParseFloat => "parse_float" [] { element: true };
    ParseBool => "parse_bool" [] { element: true };
    ToBase64 => "to_base64" [] { element: true };
    FromBase64 => "from_base64" [] { element: true };
    UrlEncode => "url_encode" [] { element: true };
    UrlDecode => "url_decode" [] { element: true };
    HtmlEscape => "html_escape" [] { element: true };
    HtmlUnescape => "html_unescape" [] { element: true };
    Repeat => "repeat" ["repeat_str"] { element: true };
    PadLeft => "pad_left" [] { element: true };
    PadRight => "pad_right" [] { element: true };
    Center => "center" [] { element: true };
    StartsWith => "starts_with" [] { element: true };
    EndsWith => "ends_with" [] { element: true };
    IndexOf => "index_of" [] { element: true };
    LastIndexOf => "last_index_of" [] { element: true };
    Replace => "replace" [] {
        executor: BuiltinPipelineExecutor::ElementBuiltin,
        shape: BuiltinPipelineShape::new(BuiltinCardinality::OneToOne, true, 2.0, 1.0),
        order: BuiltinPipelineOrderEffect::Preserves,
        lowering: BuiltinPipelineLowering::StringPairStage(
            BuiltinStringPairStage::Replace { all: false }
        )
    };
    ReplaceAll => "replace_all" [] {
        executor: BuiltinPipelineExecutor::ElementBuiltin,
        shape: BuiltinPipelineShape::new(BuiltinCardinality::OneToOne, true, 2.0, 1.0),
        order: BuiltinPipelineOrderEffect::Preserves,
        lowering: BuiltinPipelineLowering::StringPairStage(
            BuiltinStringPairStage::Replace { all: true }
        )
    };
    StripPrefix => "strip_prefix" [] { element: true };
    StripSuffix => "strip_suffix" [] { element: true };
    Slice => "slice" [] {
        executor: BuiltinPipelineExecutor::ElementBuiltin,
        shape: BuiltinPipelineShape::new(BuiltinCardinality::OneToOne, true, 1.0, 1.0),
        order: BuiltinPipelineOrderEffect::Preserves,
        lowering: BuiltinPipelineLowering::Slice
    };
    Split => "split" [] {
        executor: BuiltinPipelineExecutor::ExpandingBuiltin,
        materialization: BuiltinPipelineMaterialization::LegacyMaterialized,
        shape: BuiltinPipelineShape::new(BuiltinCardinality::Expanding, true, 2.0, 1.0),
        lowering: BuiltinPipelineLowering::StringStage(BuiltinStringStage::Split)
    };
    Indent => "indent" [] { element: true };
    Dedent => "dedent" [] { element: true };
    Matches => "matches" [] { element: true };
    Scan => "scan" [] { element: true };
    ReMatch => "re_match" [] { element: true };
    ReMatchFirst => "match_first" [] { element: true };
    ReMatchAll => "match_all" [] { element: true };
    ReCaptures => "captures" [] { element: true };
    ReCapturesAll => "captures_all" [] { element: true };
    ReSplit => "split_re" [] { element: true };
    ReReplace => "replace_re" [] { element: true };
    ReReplaceAll => "replace_all_re" [] { element: true };
    ContainsAny => "contains_any" [] { element: true };
    ContainsAll => "contains_all" [] { element: true };
    Schema => "schema" [] { element: true };
    EquiJoin => "equi_join" [];
    Unknown => "<unknown>" [];
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::{
        BuiltinExprStage, BuiltinNullaryStage, BuiltinPipelineExecutor, BuiltinPipelineLowering,
        BuiltinPipelineMaterialization, BuiltinPipelineOrderEffect, BuiltinStringPairStage,
        BuiltinStringStage, BuiltinUsizeStage,
    };

    #[test]
    fn registry_name_lookup_matches_legacy_lookup() {
        for descriptor in BUILTIN_DESCRIPTORS {
            assert_eq!(
                by_name(descriptor.canonical_name).and_then(BuiltinId::method),
                Some(descriptor.method)
            );
            for alias in descriptor.aliases {
                assert_eq!(
                    by_name(alias).and_then(BuiltinId::method),
                    Some(descriptor.method)
                );
            }
        }
        assert_eq!(by_name("missing_builtin"), None);
    }

    #[test]
    fn registry_does_not_accept_obsolete_camel_case_aliases() {
        for name in [
            "toString",
            "flatMap",
            "groupBy",
            "sortBy",
            "uniqueBy",
            "transformKeys",
            "getPath",
            "isBlank",
            "parseInt",
            "startsWith",
            "replaceAll",
        ] {
            assert_eq!(by_name(name), None);
            assert_eq!(BuiltinMethod::from_name(name), BuiltinMethod::Unknown);
        }

        assert_eq!(BuiltinMethod::from_name("group_by"), BuiltinMethod::GroupBy);
        assert_eq!(BuiltinMethod::from_name("exists"), BuiltinMethod::Any);
        assert_eq!(BuiltinMethod::from_name("distinct"), BuiltinMethod::Unique);
        assert_eq!(BuiltinMethod::from_name("lstrip"), BuiltinMethod::TrimLeft);
    }

    #[test]
    fn registry_propagates_core_streaming_demands() {
        let filter = BuiltinId::from_method(BuiltinMethod::Filter);
        let take = BuiltinId::from_method(BuiltinMethod::Take);
        let count = BuiltinId::from_method(BuiltinMethod::Count);
        let unique = BuiltinId::from_method(BuiltinMethod::Unique);
        let count_by = BuiltinId::from_method(BuiltinMethod::CountBy);
        let sort = BuiltinId::from_method(BuiltinMethod::Sort);

        let demand = propagate_demand(take, BuiltinDemandArg::Usize(3), Demand::RESULT);
        assert_eq!(demand.pull, PullDemand::FirstInput(3));

        let demand = propagate_demand(filter, BuiltinDemandArg::None, demand);
        assert_eq!(demand.pull, PullDemand::UntilOutput(3));
        assert_eq!(demand.value, ValueNeed::Whole);

        let demand = propagate_demand(count, BuiltinDemandArg::None, Demand::RESULT);
        assert_eq!(demand.pull, PullDemand::All);
        assert_eq!(demand.value, ValueNeed::None);
        assert!(!demand.order);

        let downstream = Demand {
            pull: PullDemand::FirstInput(2),
            value: ValueNeed::Whole,
            order: true,
        };
        let demand = propagate_demand(unique, BuiltinDemandArg::None, downstream);
        assert_eq!(demand.pull, PullDemand::UntilOutput(2));
        assert_eq!(demand.value, ValueNeed::Whole);
        assert!(demand.order);

        let demand = propagate_demand(count_by, BuiltinDemandArg::None, Demand::RESULT);
        assert_eq!(demand.pull, PullDemand::All);
        assert_eq!(demand.value, ValueNeed::Predicate);
        assert!(!demand.order);

        let downstream = Demand {
            pull: PullDemand::FirstInput(5),
            value: ValueNeed::Predicate,
            order: false,
        };
        let demand = propagate_demand(sort, BuiltinDemandArg::None, downstream);
        assert_eq!(demand.pull, PullDemand::All);
        assert_eq!(demand.value, ValueNeed::Whole);
        assert!(demand.order);
    }

    #[test]
    fn registry_drives_pipeline_executor_classification() {
        assert_eq!(
            pipeline_executor(BuiltinId::from_method(BuiltinMethod::Slice)),
            Some(BuiltinPipelineExecutor::ElementBuiltin)
        );
        assert_eq!(
            pipeline_executor(BuiltinId::from_method(BuiltinMethod::Replace)),
            Some(BuiltinPipelineExecutor::ElementBuiltin)
        );
        assert_eq!(
            pipeline_executor(BuiltinId::from_method(BuiltinMethod::Split)),
            Some(BuiltinPipelineExecutor::ExpandingBuiltin)
        );
        assert_eq!(
            pipeline_executor(BuiltinId::from_method(BuiltinMethod::TransformValues)),
            Some(BuiltinPipelineExecutor::ObjectLambda)
        );
        assert_eq!(
            pipeline_executor(BuiltinId::from_method(BuiltinMethod::Filter)),
            Some(BuiltinPipelineExecutor::RowFilter)
        );
        assert_eq!(
            pipeline_executor(BuiltinId::from_method(BuiltinMethod::Map)),
            Some(BuiltinPipelineExecutor::RowMap)
        );
        assert_eq!(
            pipeline_executor(BuiltinId::from_method(BuiltinMethod::FlatMap)),
            Some(BuiltinPipelineExecutor::RowFlatMap)
        );
        assert_eq!(
            pipeline_executor(BuiltinId::from_method(BuiltinMethod::Take)),
            Some(BuiltinPipelineExecutor::Position { take: true })
        );
        assert_eq!(
            pipeline_executor(BuiltinId::from_method(BuiltinMethod::Skip)),
            Some(BuiltinPipelineExecutor::Position { take: false })
        );
        assert_eq!(
            pipeline_executor(BuiltinId::from_method(BuiltinMethod::Reverse)),
            Some(BuiltinPipelineExecutor::Reverse)
        );
        assert_eq!(
            pipeline_executor(BuiltinId::from_method(BuiltinMethod::Sort)),
            Some(BuiltinPipelineExecutor::Sort)
        );
        assert_eq!(
            pipeline_executor(BuiltinId::from_method(BuiltinMethod::GroupBy)),
            Some(BuiltinPipelineExecutor::GroupBy)
        );
        assert_eq!(
            pipeline_executor(BuiltinId::from_method(BuiltinMethod::CountBy)),
            Some(BuiltinPipelineExecutor::CountBy)
        );
        assert_eq!(
            pipeline_executor(BuiltinId::from_method(BuiltinMethod::MaxBy)),
            Some(BuiltinPipelineExecutor::ArgExtreme { max: true })
        );
        assert_eq!(
            pipeline_executor(BuiltinId::from_method(BuiltinMethod::FindIndex)),
            Some(BuiltinPipelineExecutor::FindIndex)
        );
        assert_eq!(
            pipeline_executor(BuiltinId::from_method(BuiltinMethod::UniqueBy)),
            Some(BuiltinPipelineExecutor::UniqueBy)
        );
        assert_eq!(
            pipeline_executor(BuiltinId::from_method(BuiltinMethod::Chunk)),
            Some(BuiltinPipelineExecutor::Chunk)
        );
        assert_eq!(
            pipeline_executor(BuiltinId::from_method(BuiltinMethod::Window)),
            Some(BuiltinPipelineExecutor::Window)
        );
        assert_eq!(
            pipeline_executor(BuiltinId::from_method(BuiltinMethod::TakeWhile)),
            Some(BuiltinPipelineExecutor::PrefixWhile { take: true })
        );
        assert_eq!(
            pipeline_executor(BuiltinId::from_method(BuiltinMethod::DropWhile)),
            Some(BuiltinPipelineExecutor::PrefixWhile { take: false })
        );
    }

    #[test]
    fn registry_drives_pipeline_execution_policy() {
        assert_eq!(
            pipeline_materialization(BuiltinId::from_method(BuiltinMethod::Sort)),
            BuiltinPipelineMaterialization::ComposedBarrier
        );
        assert_eq!(
            pipeline_materialization(BuiltinId::from_method(BuiltinMethod::Reverse)),
            BuiltinPipelineMaterialization::ComposedBarrier
        );
        assert_eq!(
            pipeline_materialization(BuiltinId::from_method(BuiltinMethod::Split)),
            BuiltinPipelineMaterialization::LegacyMaterialized
        );
        assert_eq!(
            pipeline_materialization(BuiltinId::from_method(BuiltinMethod::TakeWhile)),
            BuiltinPipelineMaterialization::Streaming
        );
        assert_eq!(
            pipeline_shape(BuiltinId::from_method(BuiltinMethod::Split))
                .unwrap()
                .can_indexed,
            true
        );
        assert_eq!(
            pipeline_shape(BuiltinId::from_method(BuiltinMethod::Chunk))
                .unwrap()
                .cost,
            2.0
        );
        assert_eq!(
            pipeline_order_effect(BuiltinId::from_method(BuiltinMethod::Filter)),
            Some(BuiltinPipelineOrderEffect::PredicatePrefix)
        );
        assert_eq!(
            pipeline_order_effect(BuiltinId::from_method(BuiltinMethod::Replace)),
            Some(BuiltinPipelineOrderEffect::Preserves)
        );
    }

    #[test]
    fn registry_drives_pipeline_lowering() {
        assert_eq!(
            pipeline_lowering(BuiltinId::from_method(BuiltinMethod::Filter)),
            Some(BuiltinPipelineLowering::ExprStage(BuiltinExprStage::Filter))
        );
        assert_eq!(
            pipeline_lowering(BuiltinId::from_method(BuiltinMethod::Map)),
            Some(BuiltinPipelineLowering::ExprStage(BuiltinExprStage::Map))
        );
        assert_eq!(
            pipeline_lowering(BuiltinId::from_method(BuiltinMethod::FindOne)),
            Some(BuiltinPipelineLowering::TerminalExprStage {
                stage: BuiltinExprStage::Filter,
                terminal: BuiltinMethod::First,
            })
        );
        assert_eq!(
            pipeline_lowering(BuiltinId::from_method(BuiltinMethod::Take)),
            Some(BuiltinPipelineLowering::UsizeStage {
                stage: BuiltinUsizeStage::Take,
                min: 0,
            })
        );
        assert_eq!(
            pipeline_lowering(BuiltinId::from_method(BuiltinMethod::Sort)),
            Some(BuiltinPipelineLowering::Sort)
        );
        assert_eq!(
            pipeline_lowering(BuiltinId::from_method(BuiltinMethod::Reverse)),
            Some(BuiltinPipelineLowering::NullaryStage(
                BuiltinNullaryStage::Reverse
            ))
        );
        assert_eq!(
            pipeline_lowering(BuiltinId::from_method(BuiltinMethod::Split)),
            Some(BuiltinPipelineLowering::StringStage(
                BuiltinStringStage::Split
            ))
        );
        assert_eq!(
            pipeline_lowering(BuiltinId::from_method(BuiltinMethod::ReplaceAll)),
            Some(BuiltinPipelineLowering::StringPairStage(
                BuiltinStringPairStage::Replace { all: true }
            ))
        );
        assert_eq!(
            pipeline_lowering(BuiltinId::from_method(BuiltinMethod::Count)),
            Some(BuiltinPipelineLowering::TerminalSink)
        );
    }

    #[test]
    fn registry_classifies_pipeline_arity_without_method_special_cases() {
        assert_eq!(
            pipeline_arity(BuiltinId::from_method(BuiltinMethod::Filter), false),
            Some(BuiltinPipelineArity::Exact(1))
        );
        assert!(pipeline_accepts_arity(
            BuiltinId::from_method(BuiltinMethod::Filter),
            1,
            false
        ));
        assert!(!pipeline_accepts_arity(
            BuiltinId::from_method(BuiltinMethod::Filter),
            0,
            false
        ));
        assert!(pipeline_accepts_arity(
            BuiltinId::from_method(BuiltinMethod::Sort),
            0,
            false
        ));
        assert_eq!(
            pipeline_arity(BuiltinId::from_method(BuiltinMethod::Sort), false),
            Some(BuiltinPipelineArity::Range { min: 0, max: 1 })
        );
        assert!(pipeline_accepts_arity(
            BuiltinId::from_method(BuiltinMethod::Sort),
            1,
            false
        ));
        assert!(!pipeline_accepts_arity(
            BuiltinId::from_method(BuiltinMethod::Sort),
            2,
            false
        ));
        assert!(pipeline_accepts_arity(
            BuiltinId::from_method(BuiltinMethod::Slice),
            2,
            false
        ));
        assert!(!pipeline_accepts_arity(
            BuiltinId::from_method(BuiltinMethod::Count),
            1,
            false
        ));
        assert_eq!(
            pipeline_arity(BuiltinId::from_method(BuiltinMethod::Count), false),
            None
        );
        assert_eq!(
            pipeline_arity(BuiltinId::from_method(BuiltinMethod::Count), true),
            Some(BuiltinPipelineArity::Range { min: 0, max: 1 })
        );
        assert!(pipeline_accepts_arity(
            BuiltinId::from_method(BuiltinMethod::Count),
            1,
            true
        ));
        assert!(pipeline_accepts_arity(
            BuiltinId::from_method(BuiltinMethod::Sum),
            1,
            true
        ));
        assert!(!pipeline_accepts_arity(
            BuiltinId::from_method(BuiltinMethod::First),
            1,
            true
        ));
    }

    #[test]
    fn registry_drives_pipeline_element_classification() {
        for method in [
            BuiltinMethod::Upper,
            BuiltinMethod::StripPrefix,
            BuiltinMethod::IsNumeric,
            BuiltinMethod::Abs,
            BuiltinMethod::ParseInt,
            BuiltinMethod::Has,
            BuiltinMethod::Lines,
            BuiltinMethod::GetPath,
        ] {
            assert!(pipeline_element(BuiltinId::from_method(method)));
        }

        for method in [
            BuiltinMethod::Len,
            BuiltinMethod::FromJson,
            BuiltinMethod::Sort,
            BuiltinMethod::Flatten,
        ] {
            assert!(!pipeline_element(BuiltinId::from_method(method)));
        }
    }

    #[test]
    fn registry_drives_structural_lowering() {
        assert_eq!(
            structural(BuiltinId::from_method(BuiltinMethod::DeepFind)),
            Some(BuiltinStructural::DeepFind)
        );
        assert_eq!(
            structural(BuiltinId::from_method(BuiltinMethod::DeepShape)),
            Some(BuiltinStructural::DeepShape)
        );
        assert_eq!(
            structural(BuiltinId::from_method(BuiltinMethod::DeepLike)),
            Some(BuiltinStructural::DeepLike)
        );
        assert_eq!(
            structural(BuiltinId::from_method(BuiltinMethod::Walk)),
            None
        );
        assert_eq!(
            structural(BuiltinId::from_method(BuiltinMethod::Filter)),
            None
        );
    }

    #[test]
    fn unknown_builtin_demand_is_identity() {
        let downstream = Demand {
            pull: PullDemand::FirstInput(7),
            value: ValueNeed::Predicate,
            order: false,
        };
        assert_eq!(
            propagate_demand(
                BuiltinId::from_method(BuiltinMethod::Unknown),
                BuiltinDemandArg::None,
                downstream
            ),
            downstream
        );
    }

    #[test]
    fn descriptor_round_trips_method_identity() {
        for desc in BUILTIN_DESCRIPTORS {
            assert_eq!(desc.id.method(), Some(desc.method));
            assert_eq!(
                descriptor(desc.id).map(|descriptor| descriptor.method),
                Some(desc.method)
            );
        }
    }
}
