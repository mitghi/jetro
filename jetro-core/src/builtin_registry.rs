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
        BuiltinStringPairStage, BuiltinStringStage, BuiltinUsizeStage,
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
    match method_from_id(id) {
        Some(BuiltinMethod::Slice | BuiltinMethod::Replace | BuiltinMethod::ReplaceAll) => {
            Some(BuiltinPipelineExecutor::ElementBuiltin)
        }
        Some(BuiltinMethod::Split) => Some(BuiltinPipelineExecutor::ExpandingBuiltin),
        Some(
            BuiltinMethod::TransformKeys
            | BuiltinMethod::TransformValues
            | BuiltinMethod::FilterKeys
            | BuiltinMethod::FilterValues,
        ) => Some(BuiltinPipelineExecutor::ObjectLambda),
        Some(BuiltinMethod::Filter | BuiltinMethod::Find | BuiltinMethod::FindAll) => {
            Some(BuiltinPipelineExecutor::RowFilter)
        }
        Some(BuiltinMethod::Map) => Some(BuiltinPipelineExecutor::RowMap),
        Some(BuiltinMethod::FlatMap) => Some(BuiltinPipelineExecutor::RowFlatMap),
        Some(BuiltinMethod::Take) => Some(BuiltinPipelineExecutor::Position { take: true }),
        Some(BuiltinMethod::Skip) => Some(BuiltinPipelineExecutor::Position { take: false }),
        Some(BuiltinMethod::Reverse) => Some(BuiltinPipelineExecutor::Reverse),
        Some(BuiltinMethod::Sort) => Some(BuiltinPipelineExecutor::Sort),
        Some(BuiltinMethod::Unique | BuiltinMethod::UniqueBy) => {
            Some(BuiltinPipelineExecutor::UniqueBy)
        }
        Some(BuiltinMethod::GroupBy) => Some(BuiltinPipelineExecutor::GroupBy),
        Some(BuiltinMethod::CountBy) => Some(BuiltinPipelineExecutor::CountBy),
        Some(BuiltinMethod::IndexBy) => Some(BuiltinPipelineExecutor::IndexBy),
        Some(BuiltinMethod::FindIndex) => Some(BuiltinPipelineExecutor::FindIndex),
        Some(BuiltinMethod::IndicesWhere) => Some(BuiltinPipelineExecutor::IndicesWhere),
        Some(BuiltinMethod::MaxBy) => Some(BuiltinPipelineExecutor::ArgExtreme { max: true }),
        Some(BuiltinMethod::MinBy) => Some(BuiltinPipelineExecutor::ArgExtreme { max: false }),
        Some(BuiltinMethod::Chunk) => Some(BuiltinPipelineExecutor::Chunk),
        Some(BuiltinMethod::Window) => Some(BuiltinPipelineExecutor::Window),
        Some(BuiltinMethod::TakeWhile) => Some(BuiltinPipelineExecutor::PrefixWhile { take: true }),
        Some(BuiltinMethod::DropWhile) => {
            Some(BuiltinPipelineExecutor::PrefixWhile { take: false })
        }
        _ => None,
    }
}

#[inline]
pub(crate) fn pipeline_materialization(id: BuiltinId) -> BuiltinPipelineMaterialization {
    match method_from_id(id) {
        Some(
            BuiltinMethod::Sort
            | BuiltinMethod::Unique
            | BuiltinMethod::UniqueBy
            | BuiltinMethod::GroupBy
            | BuiltinMethod::Reverse,
        ) => BuiltinPipelineMaterialization::ComposedBarrier,
        Some(
            BuiltinMethod::FlatMap
            | BuiltinMethod::Split
            | BuiltinMethod::DropWhile
            | BuiltinMethod::FindIndex
            | BuiltinMethod::IndicesWhere
            | BuiltinMethod::MaxBy
            | BuiltinMethod::MinBy
            | BuiltinMethod::CountBy
            | BuiltinMethod::IndexBy
            | BuiltinMethod::Chunk
            | BuiltinMethod::Window,
        ) => BuiltinPipelineMaterialization::LegacyMaterialized,
        _ => BuiltinPipelineMaterialization::Streaming,
    }
}

#[inline]
pub(crate) fn pipeline_shape(id: BuiltinId) -> Option<BuiltinPipelineShape> {
    use BuiltinCardinality as Card;

    match method_from_id(id) {
        Some(BuiltinMethod::Split) => {
            Some(BuiltinPipelineShape::new(Card::Expanding, true, 2.0, 1.0))
        }
        Some(BuiltinMethod::TakeWhile | BuiltinMethod::DropWhile) => {
            Some(BuiltinPipelineShape::new(Card::Filtering, true, 10.0, 0.5))
        }
        Some(
            BuiltinMethod::FindIndex
            | BuiltinMethod::IndicesWhere
            | BuiltinMethod::MaxBy
            | BuiltinMethod::MinBy
            | BuiltinMethod::CountBy
            | BuiltinMethod::IndexBy
            | BuiltinMethod::TransformKeys
            | BuiltinMethod::TransformValues
            | BuiltinMethod::FilterKeys
            | BuiltinMethod::FilterValues
            | BuiltinMethod::Slice,
        ) => Some(BuiltinPipelineShape::new(Card::OneToOne, true, 1.0, 1.0)),
        Some(BuiltinMethod::Chunk | BuiltinMethod::Window) => {
            Some(BuiltinPipelineShape::new(Card::Barrier, true, 2.0, 1.0))
        }
        Some(BuiltinMethod::Replace | BuiltinMethod::ReplaceAll) => {
            Some(BuiltinPipelineShape::new(Card::OneToOne, true, 2.0, 1.0))
        }
        _ => None,
    }
}

#[inline]
pub(crate) fn pipeline_order_effect(id: BuiltinId) -> Option<BuiltinPipelineOrderEffect> {
    match method_from_id(id) {
        Some(BuiltinMethod::Filter | BuiltinMethod::Find | BuiltinMethod::FindAll) => {
            Some(BuiltinPipelineOrderEffect::PredicatePrefix)
        }
        Some(BuiltinMethod::TakeWhile) => Some(BuiltinPipelineOrderEffect::PredicatePrefix),
        Some(
            BuiltinMethod::Map
            | BuiltinMethod::Take
            | BuiltinMethod::Skip
            | BuiltinMethod::TransformKeys
            | BuiltinMethod::TransformValues
            | BuiltinMethod::FilterKeys
            | BuiltinMethod::FilterValues
            | BuiltinMethod::Slice
            | BuiltinMethod::Replace
            | BuiltinMethod::ReplaceAll,
        ) => Some(BuiltinPipelineOrderEffect::Preserves),
        Some(BuiltinMethod::DropWhile) => Some(BuiltinPipelineOrderEffect::Blocks),
        _ => None,
    }
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
    match method_from_id(id) {
        Some(BuiltinMethod::Filter | BuiltinMethod::Find | BuiltinMethod::FindAll) => {
            Some(BuiltinPipelineLowering::ExprStage(BuiltinExprStage::Filter))
        }
        Some(BuiltinMethod::Map) => Some(BuiltinPipelineLowering::ExprStage(BuiltinExprStage::Map)),
        Some(BuiltinMethod::FlatMap) => Some(BuiltinPipelineLowering::ExprStage(
            BuiltinExprStage::FlatMap,
        )),
        Some(BuiltinMethod::Split) => Some(BuiltinPipelineLowering::StringStage(
            BuiltinStringStage::Split,
        )),
        Some(BuiltinMethod::TakeWhile) => Some(BuiltinPipelineLowering::ExprStage(
            BuiltinExprStage::TakeWhile,
        )),
        Some(BuiltinMethod::DropWhile) => Some(BuiltinPipelineLowering::ExprStage(
            BuiltinExprStage::DropWhile,
        )),
        Some(BuiltinMethod::FindFirst | BuiltinMethod::FindOne) => {
            Some(BuiltinPipelineLowering::TerminalExprStage {
                stage: BuiltinExprStage::Filter,
                terminal: BuiltinMethod::First,
            })
        }
        Some(BuiltinMethod::Take) => Some(BuiltinPipelineLowering::UsizeStage {
            stage: BuiltinUsizeStage::Take,
            min: 0,
        }),
        Some(BuiltinMethod::Skip) => Some(BuiltinPipelineLowering::UsizeStage {
            stage: BuiltinUsizeStage::Skip,
            min: 0,
        }),
        Some(
            BuiltinMethod::First
            | BuiltinMethod::Last
            | BuiltinMethod::Sum
            | BuiltinMethod::Avg
            | BuiltinMethod::Min
            | BuiltinMethod::Max
            | BuiltinMethod::Count
            | BuiltinMethod::ApproxCountDistinct,
        ) => Some(BuiltinPipelineLowering::TerminalSink),
        Some(BuiltinMethod::FindIndex) => Some(BuiltinPipelineLowering::TerminalExprStage {
            stage: BuiltinExprStage::FindIndex,
            terminal: BuiltinMethod::First,
        }),
        Some(BuiltinMethod::IndicesWhere) => Some(BuiltinPipelineLowering::TerminalExprStage {
            stage: BuiltinExprStage::IndicesWhere,
            terminal: BuiltinMethod::First,
        }),
        Some(BuiltinMethod::MaxBy) => Some(BuiltinPipelineLowering::TerminalExprStage {
            stage: BuiltinExprStage::MaxBy,
            terminal: BuiltinMethod::First,
        }),
        Some(BuiltinMethod::MinBy) => Some(BuiltinPipelineLowering::TerminalExprStage {
            stage: BuiltinExprStage::MinBy,
            terminal: BuiltinMethod::First,
        }),
        Some(BuiltinMethod::Sort) => Some(BuiltinPipelineLowering::Sort),
        Some(BuiltinMethod::Unique) => Some(BuiltinPipelineLowering::NullaryStage(
            BuiltinNullaryStage::Unique,
        )),
        Some(BuiltinMethod::UniqueBy) => Some(BuiltinPipelineLowering::ExprStage(
            BuiltinExprStage::UniqueBy,
        )),
        Some(BuiltinMethod::GroupBy) => Some(BuiltinPipelineLowering::ExprStage(
            BuiltinExprStage::GroupBy,
        )),
        Some(BuiltinMethod::CountBy) => Some(BuiltinPipelineLowering::TerminalExprStage {
            stage: BuiltinExprStage::CountBy,
            terminal: BuiltinMethod::First,
        }),
        Some(BuiltinMethod::IndexBy) => Some(BuiltinPipelineLowering::TerminalExprStage {
            stage: BuiltinExprStage::IndexBy,
            terminal: BuiltinMethod::First,
        }),
        Some(BuiltinMethod::Chunk) => Some(BuiltinPipelineLowering::UsizeStage {
            stage: BuiltinUsizeStage::Chunk,
            min: 1,
        }),
        Some(BuiltinMethod::Window) => Some(BuiltinPipelineLowering::UsizeStage {
            stage: BuiltinUsizeStage::Window,
            min: 1,
        }),
        Some(BuiltinMethod::Reverse) => Some(BuiltinPipelineLowering::NullaryStage(
            BuiltinNullaryStage::Reverse,
        )),
        Some(BuiltinMethod::TransformKeys) => Some(BuiltinPipelineLowering::ExprStage(
            BuiltinExprStage::TransformKeys,
        )),
        Some(BuiltinMethod::TransformValues) => Some(BuiltinPipelineLowering::ExprStage(
            BuiltinExprStage::TransformValues,
        )),
        Some(BuiltinMethod::FilterKeys) => Some(BuiltinPipelineLowering::ExprStage(
            BuiltinExprStage::FilterKeys,
        )),
        Some(BuiltinMethod::FilterValues) => Some(BuiltinPipelineLowering::ExprStage(
            BuiltinExprStage::FilterValues,
        )),
        Some(BuiltinMethod::Slice) => Some(BuiltinPipelineLowering::Slice),
        Some(BuiltinMethod::Replace) => Some(BuiltinPipelineLowering::StringPairStage(
            BuiltinStringPairStage::Replace { all: false },
        )),
        Some(BuiltinMethod::ReplaceAll) => Some(BuiltinPipelineLowering::StringPairStage(
            BuiltinStringPairStage::Replace { all: true },
        )),
        _ => None,
    }
}

#[inline]
fn demand_law(id: BuiltinId) -> BuiltinDemandLaw {
    match method_from_id(id) {
        Some(BuiltinMethod::Filter | BuiltinMethod::Find | BuiltinMethod::FindAll) => {
            BuiltinDemandLaw::FilterLike
        }
        Some(BuiltinMethod::TakeWhile) => BuiltinDemandLaw::TakeWhile,
        Some(BuiltinMethod::Map) => BuiltinDemandLaw::MapLike,
        Some(BuiltinMethod::FlatMap) => BuiltinDemandLaw::FlatMapLike,
        Some(BuiltinMethod::Take) => BuiltinDemandLaw::Take,
        Some(BuiltinMethod::Skip) => BuiltinDemandLaw::Skip,
        Some(BuiltinMethod::First | BuiltinMethod::FindFirst) => BuiltinDemandLaw::First,
        Some(BuiltinMethod::Last) => BuiltinDemandLaw::Last,
        Some(BuiltinMethod::Count) => BuiltinDemandLaw::Count,
        Some(BuiltinMethod::Sum | BuiltinMethod::Avg | BuiltinMethod::Min | BuiltinMethod::Max) => {
            BuiltinDemandLaw::NumericReducer
        }
        _ => BuiltinDemandLaw::Identity,
    }
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

macro_rules! builtin_registry {
    ($( $method:ident => $canonical:literal [ $( $alias:literal ),* $(,)? ]; )*) => {
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
    };
}

builtin_registry! {
    Len => "len" [];
    Keys => "keys" [];
    Values => "values" [];
    Entries => "entries" [];
    ToPairs => "to_pairs" [];
    FromPairs => "from_pairs" [];
    Invert => "invert" [];
    Reverse => "reverse" [];
    Type => "type" [];
    ToString => "to_string" [];
    ToJson => "to_json" [];
    FromJson => "from_json" [];
    Sum => "sum" [];
    Avg => "avg" [];
    Min => "min" [];
    Max => "max" [];
    Count => "count" [];
    Any => "any" ["exists"];
    All => "all" [];
    FindIndex => "find_index" [];
    IndicesWhere => "indices_where" [];
    MaxBy => "max_by" [];
    MinBy => "min_by" [];
    GroupBy => "group_by" [];
    CountBy => "count_by" [];
    IndexBy => "index_by" [];
    GroupShape => "group_shape" [];
    Explode => "explode" [];
    Implode => "implode" [];
    Filter => "filter" [];
    Map => "map" [];
    FlatMap => "flat_map" [];
    Find => "find" [];
    FindAll => "find_all" [];
    Sort => "sort" ["sort_by"];
    Unique => "unique" ["distinct"];
    UniqueBy => "unique_by" [];
    Collect => "collect" [];
    DeepFind => "deep_find" [];
    DeepShape => "deep_shape" [];
    DeepLike => "deep_like" [];
    Walk => "walk" [];
    WalkPre => "walk_pre" [];
    Rec => "rec" [];
    TracePath => "trace_path" [];
    Flatten => "flatten" [];
    Compact => "compact" [];
    Join => "join" [];
    First => "first" [];
    Last => "last" [];
    Nth => "nth" [];
    Take => "take" [];
    Skip => "skip" ["drop"];
    Append => "append" [];
    Prepend => "prepend" [];
    Remove => "remove" [];
    Diff => "diff" [];
    Intersect => "intersect" [];
    Union => "union" [];
    Enumerate => "enumerate" [];
    Pairwise => "pairwise" [];
    Window => "window" [];
    Chunk => "chunk" ["batch"];
    TakeWhile => "take_while" ["takewhile"];
    DropWhile => "drop_while" ["dropwhile"];
    FindFirst => "find_first" [];
    FindOne => "find_one" [];
    ApproxCountDistinct => "approx_count_distinct" [];
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
    TransformKeys => "transform_keys" [];
    TransformValues => "transform_values" [];
    FilterKeys => "filter_keys" [];
    FilterValues => "filter_values" [];
    Pivot => "pivot" [];
    GetPath => "get_path" [];
    SetPath => "set_path" [];
    DelPath => "del_path" [];
    DelPaths => "del_paths" [];
    HasPath => "has_path" [];
    FlattenKeys => "flatten_keys" [];
    UnflattenKeys => "unflatten_keys" [];
    ToCsv => "to_csv" [];
    ToTsv => "to_tsv" [];
    Or => "or" [];
    Has => "has" [];
    Missing => "missing" [];
    Includes => "includes" ["contains"];
    Index => "index" [];
    IndicesOf => "indices_of" [];
    Set => "set" [];
    Update => "update" [];
    Ceil => "ceil" [];
    Floor => "floor" [];
    Round => "round" [];
    Abs => "abs" [];
    RollingSum => "rolling_sum" [];
    RollingAvg => "rolling_avg" [];
    RollingMin => "rolling_min" [];
    RollingMax => "rolling_max" [];
    Lag => "lag" [];
    Lead => "lead" [];
    DiffWindow => "diff_window" [];
    PctChange => "pct_change" [];
    CumMax => "cummax" [];
    CumMin => "cummin" [];
    Zscore => "zscore" [];
    Upper => "upper" [];
    Lower => "lower" [];
    Capitalize => "capitalize" [];
    TitleCase => "title_case" [];
    Trim => "trim" [];
    TrimLeft => "trim_left" ["lstrip"];
    TrimRight => "trim_right" ["rstrip"];
    SnakeCase => "snake_case" [];
    KebabCase => "kebab_case" [];
    CamelCase => "camel_case" [];
    PascalCase => "pascal_case" [];
    ReverseStr => "reverse_str" [];
    Lines => "lines" [];
    Words => "words" [];
    Chars => "chars" [];
    CharsOf => "chars_of" [];
    Bytes => "bytes" [];
    ByteLen => "byte_len" [];
    IsBlank => "is_blank" [];
    IsNumeric => "is_numeric" [];
    IsAlpha => "is_alpha" [];
    IsAscii => "is_ascii" [];
    ToNumber => "to_number" [];
    ToBool => "to_bool" [];
    ParseInt => "parse_int" [];
    ParseFloat => "parse_float" [];
    ParseBool => "parse_bool" [];
    ToBase64 => "to_base64" [];
    FromBase64 => "from_base64" [];
    UrlEncode => "url_encode" [];
    UrlDecode => "url_decode" [];
    HtmlEscape => "html_escape" [];
    HtmlUnescape => "html_unescape" [];
    Repeat => "repeat" ["repeat_str"];
    PadLeft => "pad_left" [];
    PadRight => "pad_right" [];
    Center => "center" [];
    StartsWith => "starts_with" [];
    EndsWith => "ends_with" [];
    IndexOf => "index_of" [];
    LastIndexOf => "last_index_of" [];
    Replace => "replace" [];
    ReplaceAll => "replace_all" [];
    StripPrefix => "strip_prefix" [];
    StripSuffix => "strip_suffix" [];
    Slice => "slice" [];
    Split => "split" [];
    Indent => "indent" [];
    Dedent => "dedent" [];
    Matches => "matches" [];
    Scan => "scan" [];
    ReMatch => "re_match" [];
    ReMatchFirst => "match_first" [];
    ReMatchAll => "match_all" [];
    ReCaptures => "captures" [];
    ReCapturesAll => "captures_all" [];
    ReSplit => "split_re" [];
    ReReplace => "replace_re" [];
    ReReplaceAll => "replace_all_re" [];
    ContainsAny => "contains_any" [];
    ContainsAll => "contains_all" [];
    Schema => "schema" [];
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
