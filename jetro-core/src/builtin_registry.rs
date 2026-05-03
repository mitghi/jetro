//! Macro-generated builtin registry facade.
//!
//! This is the first v2.6 adapter layer: `BuiltinMethod` remains the
//! compatibility identity for existing executors, while `BuiltinId` becomes the
//! stable identity new planner/runtime code can carry without depending on the
//! old enum directly.

use crate::{
    builtins::{
        BuiltinCardinality, BuiltinExprStage, BuiltinMethod, BuiltinNullaryStage,
        BuiltinPipelineExecutor, BuiltinPipelineLowering, BuiltinPipelineMaterialization,
        BuiltinPipelineOrderEffect, BuiltinPipelineShape, BuiltinPipelineStage, BuiltinSpec,
        BuiltinStringPairStage, BuiltinStringStage, BuiltinStructural, BuiltinUsizeStage,
    },
    chain_ir::{Demand, PullDemand, ValueNeed},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct BuiltinId(pub(crate) u16);

#[derive(Debug, Clone, Copy)]
pub(crate) struct BuiltinDescriptor {
    pub(crate) id: BuiltinId,
    pub(crate) method: BuiltinMethod,
    pub(crate) canonical_name: &'static str,
    pub(crate) aliases: &'static [&'static str],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BuiltinDemandArg {
    None,
    Usize(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BuiltinDemandLaw {
    Identity,
    FilterLike,
    TakeWhile,
    MapLike,
    FlatMapLike,
    Take,
    Skip,
    First,
    Last,
    Count,
    NumericReducer,
}

impl BuiltinDescriptor {
    #[inline]
    pub(crate) fn spec(self) -> BuiltinSpec {
        self.method.spec()
    }
}

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
    }
}

#[inline]
pub(crate) fn participates_in_demand(id: BuiltinId) -> bool {
    demand_law(id) != BuiltinDemandLaw::Identity
}

#[inline]
pub(crate) fn pipeline_executor(id: BuiltinId) -> Option<BuiltinPipelineExecutor> {
    registry_pipeline_executor(id)
}

#[inline]
pub(crate) fn pipeline_materialization(id: BuiltinId) -> BuiltinPipelineMaterialization {
    registry_pipeline_materialization(id).unwrap_or(BuiltinPipelineMaterialization::Streaming)
}

#[inline]
pub(crate) fn pipeline_shape(id: BuiltinId) -> Option<BuiltinPipelineShape> {
    registry_pipeline_shape(id)
}

#[inline]
pub(crate) fn pipeline_order_effect(id: BuiltinId) -> Option<BuiltinPipelineOrderEffect> {
    registry_pipeline_order_effect(id)
}

#[inline]
pub(crate) fn pipeline_stage(id: BuiltinId) -> Option<BuiltinPipelineStage> {
    match pipeline_lowering(id)? {
        BuiltinPipelineLowering::ExprStage(_)
        | BuiltinPipelineLowering::TerminalExprStage { .. }
        | BuiltinPipelineLowering::UsizeStage { .. } => Some(BuiltinPipelineStage::Unary),
        BuiltinPipelineLowering::NullaryStage(_) | BuiltinPipelineLowering::Sort => {
            Some(BuiltinPipelineStage::Nullary)
        }
        BuiltinPipelineLowering::StringStage(_)
        | BuiltinPipelineLowering::StringPairStage(_)
        | BuiltinPipelineLowering::Slice
        | BuiltinPipelineLowering::TerminalSink => None,
    }
}

#[inline]
pub(crate) fn pipeline_lowering(id: BuiltinId) -> Option<BuiltinPipelineLowering> {
    registry_pipeline_lowering(id)
}

#[inline]
pub(crate) fn pipeline_element(id: BuiltinId) -> bool {
    registry_pipeline_element(id).unwrap_or(false)
}

#[inline]
pub(crate) fn structural(id: BuiltinId) -> Option<BuiltinStructural> {
    registry_structural(id)
}

#[inline]
fn demand_law(id: BuiltinId) -> BuiltinDemandLaw {
    registry_demand_law(id).unwrap_or(BuiltinDemandLaw::Identity)
}

impl BuiltinId {
    #[inline]
    pub(crate) fn from_method(method: BuiltinMethod) -> Self {
        BuiltinId(method as u16)
    }

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
        executor: BuiltinPipelineExecutor::GroupBy,
        materialization: BuiltinPipelineMaterialization::ComposedBarrier,
        lowering: BuiltinPipelineLowering::ExprStage(BuiltinExprStage::GroupBy)
    };
    CountBy => "count_by" [] {
        executor: BuiltinPipelineExecutor::CountBy,
        materialization: BuiltinPipelineMaterialization::Streaming,
        shape: BuiltinPipelineShape::new(BuiltinCardinality::OneToOne, true, 1.0, 1.0),
        lowering: BuiltinPipelineLowering::TerminalExprStage {
            stage: BuiltinExprStage::CountBy,
            terminal: BuiltinMethod::First,
        }
    };
    IndexBy => "index_by" [] {
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
        executor: BuiltinPipelineExecutor::Sort,
        materialization: BuiltinPipelineMaterialization::ComposedBarrier,
        lowering: BuiltinPipelineLowering::Sort
    };
    Unique => "unique" ["distinct"] {
        executor: BuiltinPipelineExecutor::UniqueBy,
        materialization: BuiltinPipelineMaterialization::Streaming,
        shape: BuiltinPipelineShape::new(BuiltinCardinality::Filtering, true, 10.0, 1.0),
        order: BuiltinPipelineOrderEffect::Preserves,
        lowering: BuiltinPipelineLowering::NullaryStage(BuiltinNullaryStage::Unique)
    };
    UniqueBy => "unique_by" [] {
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
        BuiltinPipelineMaterialization, BuiltinPipelineOrderEffect, BuiltinPipelineStage,
        BuiltinStringPairStage, BuiltinStringStage, BuiltinUsizeStage,
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

        let demand = propagate_demand(take, BuiltinDemandArg::Usize(3), Demand::RESULT);
        assert_eq!(demand.pull, PullDemand::FirstInput(3));

        let demand = propagate_demand(filter, BuiltinDemandArg::None, demand);
        assert_eq!(demand.pull, PullDemand::UntilOutput(3));
        assert_eq!(demand.value, ValueNeed::Whole);

        let demand = propagate_demand(count, BuiltinDemandArg::None, Demand::RESULT);
        assert_eq!(demand.pull, PullDemand::All);
        assert_eq!(demand.value, ValueNeed::None);
        assert!(!demand.order);
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
    fn registry_drives_pipeline_stage_classification() {
        assert_eq!(
            pipeline_stage(BuiltinId::from_method(BuiltinMethod::Filter)),
            Some(BuiltinPipelineStage::Unary)
        );
        assert_eq!(
            pipeline_stage(BuiltinId::from_method(BuiltinMethod::CountBy)),
            Some(BuiltinPipelineStage::Unary)
        );
        assert_eq!(
            pipeline_stage(BuiltinId::from_method(BuiltinMethod::Sort)),
            Some(BuiltinPipelineStage::Nullary)
        );
        assert_eq!(
            pipeline_stage(BuiltinId::from_method(BuiltinMethod::Reverse)),
            Some(BuiltinPipelineStage::Nullary)
        );
        assert_eq!(
            pipeline_stage(BuiltinId::from_method(BuiltinMethod::Take)),
            Some(BuiltinPipelineStage::Unary)
        );
        assert_eq!(
            pipeline_stage(BuiltinId::from_method(BuiltinMethod::FindFirst)),
            Some(BuiltinPipelineStage::Unary)
        );

        assert_eq!(
            pipeline_stage(BuiltinId::from_method(BuiltinMethod::Len)),
            None
        );
        assert_eq!(
            pipeline_stage(BuiltinId::from_method(BuiltinMethod::FromJson)),
            None
        );
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
