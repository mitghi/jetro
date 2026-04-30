//! Unified built-in kernels — single home for builtin algorithm bodies
//! shared across `vm.rs`, `pipeline.rs`, and `composed.rs`.
//!
//! Per-builtin shape: an algorithm helper that takes a closure
//! evaluator from the caller. Each backend supplies its own per-row
//! evaluator (kernel-classified eval, lambda-body executor, VM
//! re-entry); the algorithm (loop / truthy-check / push-if-true) lives
//! here exactly once.
//!
//! ```text
//!   vm.rs            ─┐
//!   pipeline.rs      ─┼── all call ── builtins::filter_one / filter_apply / map_* / ...
//!   composed.rs      ─┘
//! ```
//!
//! Two primitive shapes:
//!   - `*_one(item, eval)` — per-row decision/transform; building block.
//!   - `*_apply(items, eval)` — buffered form, built on `*_one`.
//!
//! Streaming consumers call `*_one`; barrier consumers call `*_apply`.

use crate::context::EvalError;
use crate::util::{cmp_vals, is_truthy, val_key, zip_arrays};
use crate::value::Val;
use indexmap::IndexMap;
use std::sync::Arc;

// ── BuiltinMethod ─────────────────────────────────────────────────────────────

/// Pre-resolved method identifier shared by VM, pipeline analysis, and
/// builtin dispatch. Keeps method name resolution out of backend-specific
/// modules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BuiltinMethod {
    // Navigation / basics
    Len = 0,
    Keys,
    Values,
    Entries,
    ToPairs,
    FromPairs,
    Invert,
    Reverse,
    Type,
    ToString,
    ToJson,
    FromJson,
    // Aggregates
    Sum,
    Avg,
    Min,
    Max,
    Count,
    Any,
    All,
    FindIndex,
    IndicesWhere,
    MaxBy,
    MinBy,
    GroupBy,
    CountBy,
    IndexBy,
    GroupShape,
    Explode,
    Implode,
    // Array ops
    Filter,
    Map,
    FlatMap,
    Find,
    FindAll,
    Sort,
    Unique,
    UniqueBy,
    Collect,
    DeepFind,
    DeepShape,
    DeepLike,
    Walk,
    WalkPre,
    Rec,
    TracePath,
    Flatten,
    Compact,
    Join,
    First,
    Last,
    Nth,
    Take,
    Skip,
    Append,
    Prepend,
    Remove,
    Diff,
    Intersect,
    Union,
    Enumerate,
    Pairwise,
    Window,
    Chunk,
    TakeWhile,
    DropWhile,
    FindFirst,
    FindOne,
    ApproxCountDistinct,
    Accumulate,
    Partition,
    Zip,
    ZipLongest,
    Fanout,
    ZipShape,
    // Object ops
    Pick,
    Omit,
    Merge,
    DeepMerge,
    Defaults,
    Rename,
    TransformKeys,
    TransformValues,
    FilterKeys,
    FilterValues,
    Pivot,
    // Path ops
    GetPath,
    SetPath,
    DelPath,
    DelPaths,
    HasPath,
    FlattenKeys,
    UnflattenKeys,
    // CSV
    ToCsv,
    ToTsv,
    // Null / predicate
    Or,
    Has,
    Missing,
    Includes,
    Index,
    IndicesOf,
    Set,
    Update,
    // Numeric scalar ops
    Ceil,
    Floor,
    Round,
    Abs,
    RollingSum,
    RollingAvg,
    RollingMin,
    RollingMax,
    Lag,
    Lead,
    DiffWindow,
    PctChange,
    CumMax,
    CumMin,
    Zscore,
    // String methods
    Upper,
    Lower,
    Capitalize,
    TitleCase,
    Trim,
    TrimLeft,
    TrimRight,
    SnakeCase,
    KebabCase,
    CamelCase,
    PascalCase,
    ReverseStr,
    Lines,
    Words,
    Chars,
    CharsOf,
    Bytes,
    ByteLen,
    IsBlank,
    IsNumeric,
    IsAlpha,
    IsAscii,
    ToNumber,
    ToBool,
    ParseInt,
    ParseFloat,
    ParseBool,
    ToBase64,
    FromBase64,
    UrlEncode,
    UrlDecode,
    HtmlEscape,
    HtmlUnescape,
    Repeat,
    PadLeft,
    PadRight,
    Center,
    StartsWith,
    EndsWith,
    IndexOf,
    LastIndexOf,
    Replace,
    ReplaceAll,
    StripPrefix,
    StripSuffix,
    Slice,
    Split,
    Indent,
    Dedent,
    Matches,
    Scan,
    ReMatch,
    ReMatchFirst,
    ReMatchAll,
    ReCaptures,
    ReCapturesAll,
    ReSplit,
    ReReplace,
    ReReplaceAll,
    ContainsAny,
    ContainsAll,
    Schema,
    // Relational
    EquiJoin,
    // Sentinel for custom/unknown
    Unknown,
}

impl BuiltinMethod {
    pub fn from_name(name: &str) -> Self {
        match name {
            "len" => Self::Len,
            "keys" => Self::Keys,
            "values" => Self::Values,
            "entries" => Self::Entries,
            "to_pairs" | "toPairs" => Self::ToPairs,
            "from_pairs" | "fromPairs" => Self::FromPairs,
            "invert" => Self::Invert,
            "reverse" => Self::Reverse,
            "type" => Self::Type,
            "to_string" | "toString" => Self::ToString,
            "to_json" | "toJson" => Self::ToJson,
            "from_json" | "fromJson" => Self::FromJson,
            "sum" => Self::Sum,
            "avg" => Self::Avg,
            "min" => Self::Min,
            "max" => Self::Max,
            "count" => Self::Count,
            "any" | "exists" => Self::Any,
            "all" => Self::All,
            "find_index" | "findIndex" => Self::FindIndex,
            "indices_where" | "indicesWhere" => Self::IndicesWhere,
            "max_by" | "maxBy" => Self::MaxBy,
            "min_by" | "minBy" => Self::MinBy,
            "groupBy" | "group_by" => Self::GroupBy,
            "countBy" | "count_by" => Self::CountBy,
            "indexBy" | "index_by" => Self::IndexBy,
            "group_shape" | "groupShape" => Self::GroupShape,
            "explode" => Self::Explode,
            "implode" => Self::Implode,
            "filter" => Self::Filter,
            "map" => Self::Map,
            "flatMap" | "flat_map" => Self::FlatMap,
            "find" => Self::Find,
            "find_all" | "findAll" => Self::FindAll,
            "sort" | "sort_by" | "sortBy" => Self::Sort,
            "unique" | "distinct" => Self::Unique,
            "unique_by" | "uniqueBy" => Self::UniqueBy,
            "collect" => Self::Collect,
            "deep_find" | "deepFind" => Self::DeepFind,
            "deep_shape" | "deepShape" => Self::DeepShape,
            "deep_like" | "deepLike" => Self::DeepLike,
            "walk" => Self::Walk,
            "walk_pre" | "walkPre" => Self::WalkPre,
            "rec" => Self::Rec,
            "trace_path" | "tracePath" => Self::TracePath,
            "flatten" => Self::Flatten,
            "compact" => Self::Compact,
            "join" => Self::Join,
            "equi_join" | "equiJoin" => Self::EquiJoin,
            "first" => Self::First,
            "last" => Self::Last,
            "nth" => Self::Nth,
            "take" => Self::Take,
            "skip" | "drop" => Self::Skip,
            "append" => Self::Append,
            "prepend" => Self::Prepend,
            "remove" => Self::Remove,
            "diff" => Self::Diff,
            "intersect" => Self::Intersect,
            "union" => Self::Union,
            "enumerate" => Self::Enumerate,
            "pairwise" => Self::Pairwise,
            "window" => Self::Window,
            "chunk" | "batch" => Self::Chunk,
            "takewhile" | "take_while" => Self::TakeWhile,
            "dropwhile" | "drop_while" => Self::DropWhile,
            "find_first" | "findFirst" => Self::FindFirst,
            "find_one" | "findOne" => Self::FindOne,
            "approx_count_distinct" | "approxCountDistinct" => Self::ApproxCountDistinct,
            "accumulate" => Self::Accumulate,
            "partition" => Self::Partition,
            "zip" => Self::Zip,
            "zip_longest" | "zipLongest" => Self::ZipLongest,
            "fanout" => Self::Fanout,
            "zip_shape" | "zipShape" => Self::ZipShape,
            "pick" => Self::Pick,
            "omit" => Self::Omit,
            "merge" => Self::Merge,
            "deep_merge" | "deepMerge" => Self::DeepMerge,
            "defaults" => Self::Defaults,
            "rename" => Self::Rename,
            "transform_keys" | "transformKeys" => Self::TransformKeys,
            "transform_values" | "transformValues" => Self::TransformValues,
            "filter_keys" | "filterKeys" => Self::FilterKeys,
            "filter_values" | "filterValues" => Self::FilterValues,
            "pivot" => Self::Pivot,
            "get_path" | "getPath" => Self::GetPath,
            "set_path" | "setPath" => Self::SetPath,
            "del_path" | "delPath" => Self::DelPath,
            "del_paths" | "delPaths" => Self::DelPaths,
            "has_path" | "hasPath" => Self::HasPath,
            "flatten_keys" | "flattenKeys" => Self::FlattenKeys,
            "unflatten_keys" | "unflattenKeys" => Self::UnflattenKeys,
            "to_csv" | "toCsv" => Self::ToCsv,
            "to_tsv" | "toTsv" => Self::ToTsv,
            "or" => Self::Or,
            "has" => Self::Has,
            "missing" => Self::Missing,
            "includes" | "contains" => Self::Includes,
            "index" => Self::Index,
            "indices_of" | "indicesOf" => Self::IndicesOf,
            "set" => Self::Set,
            "update" => Self::Update,
            "ceil" => Self::Ceil,
            "floor" => Self::Floor,
            "round" => Self::Round,
            "abs" => Self::Abs,
            "rolling_sum" | "rollingSum" => Self::RollingSum,
            "rolling_avg" | "rollingAvg" => Self::RollingAvg,
            "rolling_min" | "rollingMin" => Self::RollingMin,
            "rolling_max" | "rollingMax" => Self::RollingMax,
            "lag" => Self::Lag,
            "lead" => Self::Lead,
            "diff_window" | "diffWindow" => Self::DiffWindow,
            "pct_change" | "pctChange" => Self::PctChange,
            "cummax" => Self::CumMax,
            "cummin" => Self::CumMin,
            "zscore" => Self::Zscore,
            "upper" => Self::Upper,
            "lower" => Self::Lower,
            "capitalize" => Self::Capitalize,
            "title_case" | "titleCase" => Self::TitleCase,
            "trim" => Self::Trim,
            "trim_left" | "trimLeft" | "lstrip" => Self::TrimLeft,
            "trim_right" | "trimRight" | "rstrip" => Self::TrimRight,
            "snake_case" | "snakeCase" => Self::SnakeCase,
            "kebab_case" | "kebabCase" => Self::KebabCase,
            "camel_case" | "camelCase" => Self::CamelCase,
            "pascal_case" | "pascalCase" => Self::PascalCase,
            "reverse_str" | "reverseStr" => Self::ReverseStr,
            "lines" => Self::Lines,
            "words" => Self::Words,
            "chars" => Self::Chars,
            "chars_of" | "charsOf" => Self::CharsOf,
            "bytes" => Self::Bytes,
            "byte_len" | "byteLen" => Self::ByteLen,
            "is_blank" | "isBlank" => Self::IsBlank,
            "is_numeric" | "isNumeric" => Self::IsNumeric,
            "is_alpha" | "isAlpha" => Self::IsAlpha,
            "is_ascii" | "isAscii" => Self::IsAscii,
            "to_number" | "toNumber" => Self::ToNumber,
            "to_bool" | "toBool" => Self::ToBool,
            "parse_int" | "parseInt" => Self::ParseInt,
            "parse_float" | "parseFloat" => Self::ParseFloat,
            "parse_bool" | "parseBool" => Self::ParseBool,
            "to_base64" | "toBase64" => Self::ToBase64,
            "from_base64" | "fromBase64" => Self::FromBase64,
            "url_encode" | "urlEncode" => Self::UrlEncode,
            "url_decode" | "urlDecode" => Self::UrlDecode,
            "html_escape" | "htmlEscape" => Self::HtmlEscape,
            "html_unescape" | "htmlUnescape" => Self::HtmlUnescape,
            "repeat" | "repeat_str" | "repeatStr" => Self::Repeat,
            "pad_left" | "padLeft" => Self::PadLeft,
            "pad_right" | "padRight" => Self::PadRight,
            "center" => Self::Center,
            "starts_with" | "startsWith" => Self::StartsWith,
            "ends_with" | "endsWith" => Self::EndsWith,
            "index_of" | "indexOf" => Self::IndexOf,
            "last_index_of" | "lastIndexOf" => Self::LastIndexOf,
            "replace" => Self::Replace,
            "replace_all" | "replaceAll" => Self::ReplaceAll,
            "strip_prefix" | "stripPrefix" => Self::StripPrefix,
            "strip_suffix" | "stripSuffix" => Self::StripSuffix,
            "slice" => Self::Slice,
            "split" => Self::Split,
            "indent" => Self::Indent,
            "dedent" => Self::Dedent,
            "matches" => Self::Matches,
            "scan" => Self::Scan,
            "re_match" | "reMatch" => Self::ReMatch,
            "match_first" | "matchFirst" => Self::ReMatchFirst,
            "match_all" | "matchAll" => Self::ReMatchAll,
            "captures" => Self::ReCaptures,
            "captures_all" | "capturesAll" => Self::ReCapturesAll,
            "split_re" | "splitRe" => Self::ReSplit,
            "replace_re" | "replaceRe" => Self::ReReplace,
            "replace_all_re" | "replaceAllRe" => Self::ReReplaceAll,
            "contains_any" | "containsAny" => Self::ContainsAny,
            "contains_all" | "containsAll" => Self::ContainsAll,
            "schema" => Self::Schema,
            _ => Self::Unknown,
        }
    }

    /// True for methods that receive a sub-program to run per item.
    pub(crate) fn is_lambda_method(self) -> bool {
        matches!(
            self,
            Self::Filter
                | Self::Map
                | Self::FlatMap
                | Self::Sort
                | Self::Any
                | Self::All
                | Self::Count
                | Self::GroupBy
                | Self::CountBy
                | Self::IndexBy
                | Self::TakeWhile
                | Self::DropWhile
                | Self::Accumulate
                | Self::Partition
                | Self::TransformKeys
                | Self::TransformValues
                | Self::FilterKeys
                | Self::FilterValues
                | Self::Pivot
                | Self::Update
        )
    }
}

// ── Resolved builtin calls ──────────────────────────────────────────────────

/// Literal arguments attached to a resolved builtin call.
///
/// Backends that can only lower compile-time literal arguments use this
/// carrier instead of keeping their own per-runtime argument enum.
#[derive(Debug, Clone)]
pub enum BuiltinArgs {
    None,
    Str(Arc<str>),
    StrPair { first: Arc<str>, second: Arc<str> },
    StrVec(Vec<Arc<str>>),
    I64(i64),
    I64Opt { first: i64, second: Option<i64> },
    Usize(usize),
    Val(Val),
    ValVec(Vec<Val>),
    Pad { width: usize, fill: char },
}

#[derive(Debug, Clone)]
pub struct BuiltinCall {
    pub method: BuiltinMethod,
    pub args: BuiltinArgs,
}

struct StaticArgDecoder<'a, E, I> {
    name: &'a str,
    eval_arg: E,
    ident_arg: I,
}

impl<E, I> StaticArgDecoder<'_, E, I>
where
    E: FnMut(usize) -> Result<Option<Val>, EvalError>,
    I: FnMut(usize) -> Option<Arc<str>>,
{
    fn val(&mut self, idx: usize) -> Result<Val, EvalError> {
        (self.eval_arg)(idx)?.ok_or_else(|| EvalError(format!("{}: missing argument", self.name)))
    }

    fn str(&mut self, idx: usize) -> Result<Arc<str>, EvalError> {
        if let Some(value) = (self.ident_arg)(idx) {
            return Ok(value);
        }
        match self.val(idx)? {
            Val::Str(s) => Ok(s),
            other => Ok(Arc::from(crate::util::val_to_string(&other).as_str())),
        }
    }

    fn i64(&mut self, idx: usize) -> Result<i64, EvalError> {
        match self.val(idx)? {
            Val::Int(n) => Ok(n),
            Val::Float(f) => Ok(f as i64),
            _ => Err(EvalError(format!(
                "{}: expected number argument",
                self.name
            ))),
        }
    }

    fn usize(&mut self, idx: usize) -> Result<usize, EvalError> {
        Ok(self.i64(idx)?.max(0) as usize)
    }

    fn vec(&mut self, idx: usize) -> Result<Vec<Val>, EvalError> {
        self.val(idx).and_then(|value| {
            value
                .into_vec()
                .ok_or_else(|| EvalError(format!("{}: expected array arg", self.name)))
        })
    }

    fn str_vec(&mut self, idx: usize) -> Result<Vec<Arc<str>>, EvalError> {
        Ok(self
            .vec(idx)?
            .iter()
            .map(|v| match v {
                Val::Str(s) => s.clone(),
                other => Arc::from(crate::util::val_to_string(other).as_str()),
            })
            .collect())
    }

    fn char(&mut self, idx: usize, arg_len: usize) -> Result<char, EvalError> {
        if idx >= arg_len {
            return Ok(' ');
        }
        match self.str(idx)? {
            s if s.chars().count() == 1 => Ok(s.chars().next().unwrap()),
            _ => Err(EvalError(format!(
                "{}: filler must be a single-char string",
                self.name
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BuiltinSpec {
    pub pure: bool,
    pub category: BuiltinCategory,
    pub cardinality: BuiltinCardinality,
    pub can_indexed: bool,
    pub view_native: bool,
    pub view_scalar: bool,
    pub view_stage: Option<BuiltinViewStage>,
    pub view_sink: Option<BuiltinViewSink>,
    pub pipeline_stage: Option<BuiltinPipelineStage>,
    pub pipeline_sink: Option<BuiltinPipelineSink>,
    pub pipeline_element: bool,
    pub cost: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinViewStage {
    Filter,
    Map,
    FlatMap,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinViewSink {
    Count,
    Numeric,
    First,
    Last,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinPipelineStage {
    Nullary,
    Unary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinPipelineSink {
    ApproxCountDistinct,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinViewMaterialization {
    Never,
    SinkFinalRow,
    SinkNumericInput,
}

impl BuiltinViewSink {
    pub fn materialization(self) -> BuiltinViewMaterialization {
        match self {
            Self::Count => BuiltinViewMaterialization::Never,
            Self::Numeric => BuiltinViewMaterialization::SinkNumericInput,
            Self::First | Self::Last => BuiltinViewMaterialization::SinkFinalRow,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinCategory {
    Scalar,
    StreamingOneToOne,
    StreamingFilter,
    StreamingExpand,
    Reducer,
    Positional,
    Barrier,
    Object,
    Path,
    Deep,
    Serialization,
    Relational,
    Mutation,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinCardinality {
    OneToOne,
    Filtering,
    Expanding,
    Bounded,
    Reducing,
    Barrier,
}

impl BuiltinSpec {
    fn new(category: BuiltinCategory, cardinality: BuiltinCardinality) -> Self {
        Self {
            pure: true,
            category,
            cardinality,
            can_indexed: false,
            view_native: false,
            view_scalar: false,
            view_stage: None,
            view_sink: None,
            pipeline_stage: None,
            pipeline_sink: None,
            pipeline_element: false,
            cost: 1.0,
        }
    }

    fn indexed(mut self) -> Self {
        self.can_indexed = true;
        self
    }

    fn view_native(mut self) -> Self {
        self.view_native = true;
        self
    }

    fn view_stage(mut self, stage: BuiltinViewStage) -> Self {
        self.view_stage = Some(stage);
        self
    }

    fn view_scalar(mut self) -> Self {
        self.view_scalar = true;
        self.view_native = true;
        self
    }

    fn pipeline_stage(mut self, stage: BuiltinPipelineStage) -> Self {
        self.pipeline_stage = Some(stage);
        self
    }

    fn pipeline_sink(mut self, sink: BuiltinPipelineSink) -> Self {
        self.pipeline_sink = Some(sink);
        self
    }

    fn view_sink(mut self, sink: BuiltinViewSink) -> Self {
        self.view_sink = Some(sink);
        self
    }

    fn pipeline_element(mut self) -> Self {
        self.pipeline_element = true;
        self
    }

    fn cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }
}

impl BuiltinMethod {
    #[inline]
    pub fn spec(self) -> BuiltinSpec {
        use BuiltinCardinality as Card;
        use BuiltinCategory as Cat;

        let spec = match self {
            Self::Filter | Self::Find | Self::FindAll => {
                BuiltinSpec::new(Cat::StreamingFilter, Card::Filtering)
                    .view_stage(BuiltinViewStage::Filter)
                    .pipeline_stage(BuiltinPipelineStage::Unary)
                    .cost(10.0)
            }
            Self::Compact | Self::Remove => {
                BuiltinSpec::new(Cat::StreamingFilter, Card::Filtering).cost(10.0)
            }
            Self::Map => BuiltinSpec::new(Cat::StreamingOneToOne, Card::OneToOne)
                .indexed()
                .view_stage(BuiltinViewStage::Map)
                .pipeline_stage(BuiltinPipelineStage::Unary)
                .pipeline_element()
                .cost(10.0),
            Self::Enumerate | Self::Pairwise => {
                BuiltinSpec::new(Cat::StreamingOneToOne, Card::OneToOne)
                    .indexed()
                    .pipeline_element()
                    .cost(10.0)
            }
            Self::FlatMap => BuiltinSpec::new(Cat::StreamingExpand, Card::Expanding)
                .view_stage(BuiltinViewStage::FlatMap)
                .pipeline_stage(BuiltinPipelineStage::Unary)
                .cost(10.0),
            Self::Flatten | Self::Explode | Self::Split => {
                BuiltinSpec::new(Cat::StreamingExpand, Card::Expanding).cost(10.0)
            }
            Self::Lines | Self::Words | Self::Chars | Self::CharsOf | Self::Bytes => {
                BuiltinSpec::new(Cat::StreamingExpand, Card::Expanding)
                    .pipeline_element()
                    .cost(10.0)
            }
            Self::TakeWhile | Self::DropWhile => {
                BuiltinSpec::new(Cat::StreamingFilter, Card::Filtering)
                    .pipeline_stage(BuiltinPipelineStage::Unary)
                    .cost(10.0)
            }
            Self::FindFirst | Self::FindOne => {
                BuiltinSpec::new(Cat::StreamingFilter, Card::Filtering)
                    .pipeline_stage(BuiltinPipelineStage::Unary)
                    .cost(10.0)
            }
            Self::Take | Self::Skip => BuiltinSpec::new(Cat::Positional, Card::Bounded)
                .view_native()
                .pipeline_stage(BuiltinPipelineStage::Unary),
            Self::First => BuiltinSpec::new(Cat::Positional, Card::Bounded)
                .view_native()
                .view_sink(BuiltinViewSink::First),
            Self::Last => BuiltinSpec::new(Cat::Positional, Card::Bounded)
                .view_native()
                .view_sink(BuiltinViewSink::Last),
            Self::Nth | Self::Collect => {
                BuiltinSpec::new(Cat::Positional, Card::Bounded).view_native()
            }
            Self::Len => BuiltinSpec::new(Cat::Reducer, Card::Reducing)
                .indexed()
                .view_scalar()
                .view_sink(BuiltinViewSink::Count),
            Self::Sum | Self::Avg | Self::Min | Self::Max => {
                BuiltinSpec::new(Cat::Reducer, Card::Reducing)
                    .view_native()
                    .view_sink(BuiltinViewSink::Numeric)
                    .cost(10.0)
            }
            Self::Count => BuiltinSpec::new(Cat::Reducer, Card::Reducing)
                .view_native()
                .view_sink(BuiltinViewSink::Count)
                .cost(10.0),
            Self::ApproxCountDistinct => BuiltinSpec::new(Cat::Reducer, Card::Reducing)
                .pipeline_sink(BuiltinPipelineSink::ApproxCountDistinct)
                .cost(10.0),
            Self::Any
            | Self::All
            | Self::FindIndex
            | Self::IndicesWhere
            | Self::MaxBy
            | Self::MinBy => {
                let spec = BuiltinSpec::new(Cat::Reducer, Card::Reducing)
                    .view_native()
                    .cost(10.0);
                match self {
                    Self::FindIndex | Self::IndicesWhere | Self::MaxBy | Self::MinBy => {
                        spec.pipeline_stage(BuiltinPipelineStage::Unary)
                    }
                    _ => spec,
                }
            }
            Self::Sort
            | Self::Unique
            | Self::UniqueBy
            | Self::GroupBy
            | Self::CountBy
            | Self::IndexBy
            | Self::GroupShape
            | Self::Partition
            | Self::Window
            | Self::Chunk
            | Self::RollingSum
            | Self::RollingAvg
            | Self::RollingMin
            | Self::RollingMax
            | Self::Accumulate => {
                let spec = BuiltinSpec::new(Cat::Barrier, Card::Barrier).cost(20.0);
                match self {
                    Self::Sort | Self::Unique => spec.pipeline_stage(BuiltinPipelineStage::Nullary),
                    Self::UniqueBy
                    | Self::GroupBy
                    | Self::CountBy
                    | Self::IndexBy
                    | Self::Chunk
                    | Self::Window => spec.pipeline_stage(BuiltinPipelineStage::Unary),
                    _ => spec,
                }
            }
            Self::Reverse
            | Self::Append
            | Self::Prepend
            | Self::Diff
            | Self::Intersect
            | Self::Union
            | Self::Join
            | Self::Zip
            | Self::ZipLongest
            | Self::Fanout
            | Self::ZipShape => {
                let spec = BuiltinSpec::new(Cat::Barrier, Card::Barrier).cost(10.0);
                match self {
                    Self::Reverse => spec.pipeline_stage(BuiltinPipelineStage::Nullary),
                    _ => spec,
                }
            }
            Self::Keys | Self::Values | Self::Entries => {
                BuiltinSpec::new(Cat::Object, Card::OneToOne).pipeline_element()
            }
            Self::ToPairs
            | Self::FromPairs
            | Self::Invert
            | Self::Pick
            | Self::Omit
            | Self::Merge
            | Self::DeepMerge
            | Self::Defaults
            | Self::Rename
            | Self::TransformKeys
            | Self::TransformValues
            | Self::FilterKeys
            | Self::FilterValues
            | Self::Pivot
            | Self::Implode => BuiltinSpec::new(Cat::Object, Card::OneToOne),
            Self::GetPath | Self::DelPath | Self::HasPath => {
                BuiltinSpec::new(Cat::Path, Card::OneToOne)
                    .indexed()
                    .pipeline_element()
            }
            Self::SetPath | Self::DelPaths | Self::FlattenKeys | Self::UnflattenKeys => {
                BuiltinSpec::new(Cat::Path, Card::OneToOne).indexed()
            }
            Self::DeepFind
            | Self::DeepShape
            | Self::DeepLike
            | Self::Walk
            | Self::WalkPre
            | Self::Rec
            | Self::TracePath => BuiltinSpec::new(Cat::Deep, Card::Expanding).cost(20.0),
            Self::ToCsv | Self::ToTsv => BuiltinSpec::new(Cat::Serialization, Card::OneToOne)
                .indexed()
                .cost(20.0),
            Self::EquiJoin => BuiltinSpec::new(Cat::Relational, Card::Barrier).cost(20.0),
            Self::Set => BuiltinSpec::new(Cat::Mutation, Card::OneToOne)
                .indexed()
                .pipeline_element(),
            Self::Update => BuiltinSpec::new(Cat::Mutation, Card::OneToOne).indexed(),
            Self::Lag
            | Self::Lead
            | Self::DiffWindow
            | Self::PctChange
            | Self::CumMax
            | Self::CumMin
            | Self::Zscore => BuiltinSpec::new(Cat::StreamingOneToOne, Card::OneToOne)
                .indexed()
                .pipeline_element()
                .cost(10.0),
            Self::Unknown => BuiltinSpec {
                pure: false,
                ..BuiltinSpec::new(Cat::Unknown, Card::OneToOne)
            },
            _ => {
                let pipeline_element = matches!(
                    self,
                    Self::Upper
                        | Self::Lower
                        | Self::Trim
                        | Self::TrimLeft
                        | Self::TrimRight
                        | Self::Capitalize
                        | Self::TitleCase
                        | Self::SnakeCase
                        | Self::KebabCase
                        | Self::CamelCase
                        | Self::PascalCase
                        | Self::ReverseStr
                        | Self::HtmlEscape
                        | Self::HtmlUnescape
                        | Self::UrlEncode
                        | Self::UrlDecode
                        | Self::ToBase64
                        | Self::FromBase64
                        | Self::Dedent
                        | Self::ByteLen
                        | Self::IsBlank
                        | Self::IsNumeric
                        | Self::IsAlpha
                        | Self::IsAscii
                        | Self::Ceil
                        | Self::Floor
                        | Self::Round
                        | Self::Abs
                        | Self::ToNumber
                        | Self::ToBool
                        | Self::ParseInt
                        | Self::ParseFloat
                        | Self::ParseBool
                        | Self::Or
                        | Self::Type
                        | Self::ToString
                        | Self::ToJson
                        | Self::Schema
                        | Self::Has
                        | Self::StartsWith
                        | Self::EndsWith
                        | Self::StripPrefix
                        | Self::StripSuffix
                        | Self::Matches
                        | Self::IndexOf
                        | Self::LastIndexOf
                        | Self::Scan
                        | Self::ReMatch
                        | Self::ReMatchFirst
                        | Self::ReMatchAll
                        | Self::ReCaptures
                        | Self::ReCapturesAll
                        | Self::ReSplit
                        | Self::ReReplace
                        | Self::ReReplaceAll
                        | Self::ContainsAny
                        | Self::ContainsAll
                        | Self::Repeat
                        | Self::Indent
                        | Self::PadLeft
                        | Self::PadRight
                        | Self::Center
                );
                let spec = BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                    .indexed()
                    .view_native();
                let spec = match self {
                    Self::StartsWith | Self::EndsWith => spec.view_scalar(),
                    _ => spec,
                };
                if pipeline_element {
                    spec.pipeline_element()
                } else {
                    spec
                }
            }
        };
        spec
    }
}

impl BuiltinCall {
    #[inline]
    pub fn new(method: BuiltinMethod, args: BuiltinArgs) -> Self {
        Self { method, args }
    }

    #[inline]
    pub fn spec(&self) -> BuiltinSpec {
        let mut spec = self.method.spec();
        let (cost, can_indexed) = match self.method {
            BuiltinMethod::Keys | BuiltinMethod::Values | BuiltinMethod::Entries => (1.0, false),
            BuiltinMethod::Repeat
            | BuiltinMethod::Indent
            | BuiltinMethod::PadLeft
            | BuiltinMethod::PadRight
            | BuiltinMethod::Center => (2.0, true),
            BuiltinMethod::IndexOf
            | BuiltinMethod::LastIndexOf
            | BuiltinMethod::Scan
            | BuiltinMethod::StartsWith
            | BuiltinMethod::EndsWith
            | BuiltinMethod::StripPrefix
            | BuiltinMethod::StripSuffix
            | BuiltinMethod::Matches
            | BuiltinMethod::ReMatch
            | BuiltinMethod::ReMatchFirst
            | BuiltinMethod::ReMatchAll
            | BuiltinMethod::ReCaptures
            | BuiltinMethod::ReCapturesAll
            | BuiltinMethod::ReSplit
            | BuiltinMethod::ReReplace
            | BuiltinMethod::ReReplaceAll
            | BuiltinMethod::ContainsAny
            | BuiltinMethod::ContainsAll => (2.0, true),
            _ => (spec.cost, spec.can_indexed),
        };
        spec.cost = cost;
        spec.can_indexed = can_indexed;
        spec
    }

    #[inline]
    pub fn is_idempotent(&self) -> bool {
        matches!(
            self.method,
            BuiltinMethod::Upper
                | BuiltinMethod::Lower
                | BuiltinMethod::Trim
                | BuiltinMethod::TrimLeft
                | BuiltinMethod::TrimRight
                | BuiltinMethod::Capitalize
                | BuiltinMethod::TitleCase
                | BuiltinMethod::SnakeCase
                | BuiltinMethod::KebabCase
                | BuiltinMethod::CamelCase
                | BuiltinMethod::PascalCase
                | BuiltinMethod::Dedent
        )
    }

    #[inline]
    pub fn cancels_with(&self, other: &Self) -> bool {
        matches!(
            (self.method, other.method),
            (BuiltinMethod::ToBase64, BuiltinMethod::FromBase64)
                | (BuiltinMethod::FromBase64, BuiltinMethod::ToBase64)
                | (BuiltinMethod::UrlEncode, BuiltinMethod::UrlDecode)
                | (BuiltinMethod::UrlDecode, BuiltinMethod::UrlEncode)
                | (BuiltinMethod::HtmlEscape, BuiltinMethod::HtmlUnescape)
                | (BuiltinMethod::HtmlUnescape, BuiltinMethod::HtmlEscape)
                | (BuiltinMethod::ReverseStr, BuiltinMethod::ReverseStr)
        )
    }

    pub fn apply(&self, recv: &Val) -> Option<Val> {
        macro_rules! apply_or_recv {
            ($expr:expr) => {
                return Some($expr.unwrap_or_else(|| recv.clone()))
            };
        }
        match (self.method, &self.args) {
            (BuiltinMethod::Upper, BuiltinArgs::None) => apply_or_recv!(upper_apply(recv)),
            (BuiltinMethod::Lower, BuiltinArgs::None) => apply_or_recv!(lower_apply(recv)),
            (BuiltinMethod::Trim, BuiltinArgs::None) => apply_or_recv!(trim_apply(recv)),
            (BuiltinMethod::TrimLeft, BuiltinArgs::None) => apply_or_recv!(trim_left_apply(recv)),
            (BuiltinMethod::TrimRight, BuiltinArgs::None) => {
                apply_or_recv!(trim_right_apply(recv))
            }
            (BuiltinMethod::Capitalize, BuiltinArgs::None) => {
                apply_or_recv!(capitalize_apply(recv))
            }
            (BuiltinMethod::TitleCase, BuiltinArgs::None) => {
                apply_or_recv!(title_case_apply(recv))
            }
            (BuiltinMethod::SnakeCase, BuiltinArgs::None) => apply_or_recv!(snake_case_apply(recv)),
            (BuiltinMethod::KebabCase, BuiltinArgs::None) => apply_or_recv!(kebab_case_apply(recv)),
            (BuiltinMethod::CamelCase, BuiltinArgs::None) => apply_or_recv!(camel_case_apply(recv)),
            (BuiltinMethod::PascalCase, BuiltinArgs::None) => {
                apply_or_recv!(pascal_case_apply(recv))
            }
            (BuiltinMethod::ReverseStr, BuiltinArgs::None) => {
                apply_or_recv!(reverse_str_apply(recv))
            }
            (BuiltinMethod::HtmlEscape, BuiltinArgs::None) => {
                apply_or_recv!(html_escape_apply(recv))
            }
            (BuiltinMethod::HtmlUnescape, BuiltinArgs::None) => {
                apply_or_recv!(html_unescape_apply(recv))
            }
            (BuiltinMethod::UrlEncode, BuiltinArgs::None) => {
                apply_or_recv!(url_encode_apply(recv))
            }
            (BuiltinMethod::UrlDecode, BuiltinArgs::None) => {
                apply_or_recv!(url_decode_apply(recv))
            }
            (BuiltinMethod::ToBase64, BuiltinArgs::None) => {
                apply_or_recv!(to_base64_apply(recv))
            }
            (BuiltinMethod::FromBase64, BuiltinArgs::None) => {
                apply_or_recv!(from_base64_apply(recv))
            }
            (BuiltinMethod::Dedent, BuiltinArgs::None) => apply_or_recv!(dedent_apply(recv)),
            (BuiltinMethod::Lines, BuiltinArgs::None) => apply_or_recv!(lines_apply(recv)),
            (BuiltinMethod::Words, BuiltinArgs::None) => apply_or_recv!(words_apply(recv)),
            (BuiltinMethod::Chars, BuiltinArgs::None) => apply_or_recv!(chars_apply(recv)),
            (BuiltinMethod::CharsOf, BuiltinArgs::None) => apply_or_recv!(chars_of_apply(recv)),
            (BuiltinMethod::Bytes, BuiltinArgs::None) => apply_or_recv!(bytes_of_apply(recv)),
            (BuiltinMethod::ByteLen, BuiltinArgs::None) => apply_or_recv!(byte_len_apply(recv)),
            (BuiltinMethod::IsBlank, BuiltinArgs::None) => apply_or_recv!(is_blank_apply(recv)),
            (BuiltinMethod::IsNumeric, BuiltinArgs::None) => {
                apply_or_recv!(is_numeric_apply(recv))
            }
            (BuiltinMethod::IsAlpha, BuiltinArgs::None) => apply_or_recv!(is_alpha_apply(recv)),
            (BuiltinMethod::IsAscii, BuiltinArgs::None) => apply_or_recv!(is_ascii_apply(recv)),
            (BuiltinMethod::ToNumber, BuiltinArgs::None) => {
                apply_or_recv!(to_number_apply(recv))
            }
            (BuiltinMethod::ToBool, BuiltinArgs::None) => apply_or_recv!(to_bool_apply(recv)),
            (BuiltinMethod::ParseInt, BuiltinArgs::None) => apply_or_recv!(parse_int_apply(recv)),
            (BuiltinMethod::ParseFloat, BuiltinArgs::None) => {
                apply_or_recv!(parse_float_apply(recv))
            }
            (BuiltinMethod::ParseBool, BuiltinArgs::None) => {
                apply_or_recv!(parse_bool_apply(recv))
            }
            (BuiltinMethod::Sum, BuiltinArgs::None)
            | (BuiltinMethod::Avg, BuiltinArgs::None)
            | (BuiltinMethod::Min, BuiltinArgs::None)
            | (BuiltinMethod::Max, BuiltinArgs::None) => {
                return Some(numeric_aggregate_apply(recv, self.method));
            }
            (BuiltinMethod::Len, BuiltinArgs::None) | (BuiltinMethod::Count, BuiltinArgs::None) => {
                apply_or_recv!(len_apply(recv))
            }
            (BuiltinMethod::Keys, BuiltinArgs::None) => return Some(keys_apply(recv)),
            (BuiltinMethod::Values, BuiltinArgs::None) => return Some(values_apply(recv)),
            (BuiltinMethod::Entries, BuiltinArgs::None) => return Some(entries_apply(recv)),
            (BuiltinMethod::Reverse, BuiltinArgs::None) => apply_or_recv!(reverse_any_apply(recv)),
            (BuiltinMethod::Unique, BuiltinArgs::None) => apply_or_recv!(unique_arr_apply(recv)),
            (BuiltinMethod::Collect, BuiltinArgs::None) => return Some(collect_apply(recv)),
            (BuiltinMethod::Invert, BuiltinArgs::None) => apply_or_recv!(invert_apply(recv)),
            (BuiltinMethod::Type, BuiltinArgs::None) => apply_or_recv!(type_name_apply(recv)),
            (BuiltinMethod::ToString, BuiltinArgs::None) => apply_or_recv!(to_string_apply(recv)),
            (BuiltinMethod::ToJson, BuiltinArgs::None) => apply_or_recv!(to_json_apply(recv)),
            (BuiltinMethod::FromJson, BuiltinArgs::None) => return from_json_apply(recv),
            (BuiltinMethod::ToCsv, BuiltinArgs::None) => apply_or_recv!(to_csv_apply(recv)),
            (BuiltinMethod::ToTsv, BuiltinArgs::None) => apply_or_recv!(to_tsv_apply(recv)),
            (BuiltinMethod::ToPairs, BuiltinArgs::None) => apply_or_recv!(to_pairs_apply(recv)),
            (BuiltinMethod::FromPairs, BuiltinArgs::None) => {
                apply_or_recv!(from_pairs_apply(recv))
            }
            (BuiltinMethod::Ceil, BuiltinArgs::None) => return ceil_apply(recv),
            (BuiltinMethod::Floor, BuiltinArgs::None) => return floor_apply(recv),
            (BuiltinMethod::Round, BuiltinArgs::None) => return round_apply(recv),
            (BuiltinMethod::Abs, BuiltinArgs::None) => return abs_apply(recv),
            (BuiltinMethod::Or, BuiltinArgs::Val(default)) => return Some(or_apply(recv, default)),
            (BuiltinMethod::Missing, BuiltinArgs::Str(k)) => return Some(missing_apply(recv, k)),
            (BuiltinMethod::Includes, BuiltinArgs::Val(item)) => {
                return Some(includes_apply(recv, item))
            }
            (BuiltinMethod::Index, BuiltinArgs::Val(item)) => return index_value_apply(recv, item),
            (BuiltinMethod::IndicesOf, BuiltinArgs::Val(item)) => {
                return indices_of_apply(recv, item)
            }
            (BuiltinMethod::Set, BuiltinArgs::Val(item)) => return Some(item.clone()),
            (BuiltinMethod::Compact, BuiltinArgs::None) => apply_or_recv!(compact_apply(recv)),
            (BuiltinMethod::Join, BuiltinArgs::Str(sep)) => return join_apply(recv, sep),
            (BuiltinMethod::Enumerate, BuiltinArgs::None) => return enumerate_apply(recv),
            (BuiltinMethod::Pairwise, BuiltinArgs::None) => apply_or_recv!(pairwise_apply(recv)),
            (BuiltinMethod::Schema, BuiltinArgs::None) => apply_or_recv!(schema_apply(recv)),
            (BuiltinMethod::Flatten, BuiltinArgs::Usize(depth)) => {
                apply_or_recv!(flatten_depth_apply(recv, *depth))
            }
            (BuiltinMethod::First, BuiltinArgs::I64(n)) => apply_or_recv!(first_apply(recv, *n)),
            (BuiltinMethod::Last, BuiltinArgs::I64(n)) => apply_or_recv!(last_apply(recv, *n)),
            (BuiltinMethod::Nth, BuiltinArgs::I64(n)) => apply_or_recv!(nth_any_apply(recv, *n)),
            (BuiltinMethod::Append, BuiltinArgs::Val(item)) => {
                apply_or_recv!(append_apply(recv, item))
            }
            (BuiltinMethod::Prepend, BuiltinArgs::Val(item)) => {
                apply_or_recv!(prepend_apply(recv, item))
            }
            (BuiltinMethod::Remove, BuiltinArgs::Val(item)) => {
                apply_or_recv!(remove_value_apply(recv, item))
            }
            (BuiltinMethod::Diff, BuiltinArgs::ValVec(other)) => {
                let arr_recv = recv.clone().into_vec().map(Val::arr)?;
                apply_or_recv!(diff_apply(&arr_recv, other))
            }
            (BuiltinMethod::Intersect, BuiltinArgs::ValVec(other)) => {
                let arr_recv = recv.clone().into_vec().map(Val::arr)?;
                apply_or_recv!(intersect_apply(&arr_recv, other))
            }
            (BuiltinMethod::Union, BuiltinArgs::ValVec(other)) => {
                let arr_recv = recv.clone().into_vec().map(Val::arr)?;
                apply_or_recv!(union_apply(&arr_recv, other))
            }
            (BuiltinMethod::Window, BuiltinArgs::Usize(n)) => {
                let arr_recv = recv.clone().into_vec().map(Val::arr)?;
                apply_or_recv!(window_arr_apply(&arr_recv, *n))
            }
            (BuiltinMethod::Chunk, BuiltinArgs::Usize(n)) => {
                let arr_recv = recv.clone().into_vec().map(Val::arr)?;
                apply_or_recv!(chunk_arr_apply(&arr_recv, *n))
            }
            (BuiltinMethod::RollingSum, BuiltinArgs::Usize(n)) => {
                apply_or_recv!(rolling_sum_apply(recv, *n))
            }
            (BuiltinMethod::RollingAvg, BuiltinArgs::Usize(n)) => {
                apply_or_recv!(rolling_avg_apply(recv, *n))
            }
            (BuiltinMethod::RollingMin, BuiltinArgs::Usize(n)) => {
                apply_or_recv!(rolling_min_apply(recv, *n))
            }
            (BuiltinMethod::RollingMax, BuiltinArgs::Usize(n)) => {
                apply_or_recv!(rolling_max_apply(recv, *n))
            }
            (BuiltinMethod::Lag, BuiltinArgs::Usize(n)) => apply_or_recv!(lag_apply(recv, *n)),
            (BuiltinMethod::Lead, BuiltinArgs::Usize(n)) => apply_or_recv!(lead_apply(recv, *n)),
            (BuiltinMethod::DiffWindow, BuiltinArgs::None) => {
                apply_or_recv!(diff_window_apply(recv))
            }
            (BuiltinMethod::PctChange, BuiltinArgs::None) => {
                apply_or_recv!(pct_change_apply(recv))
            }
            (BuiltinMethod::CumMax, BuiltinArgs::None) => apply_or_recv!(cummax_apply(recv)),
            (BuiltinMethod::CumMin, BuiltinArgs::None) => apply_or_recv!(cummin_apply(recv)),
            (BuiltinMethod::Zscore, BuiltinArgs::None) => apply_or_recv!(zscore_apply(recv)),
            (BuiltinMethod::Merge, BuiltinArgs::Val(other)) => {
                apply_or_recv!(merge_apply(recv, other))
            }
            (BuiltinMethod::DeepMerge, BuiltinArgs::Val(other)) => {
                apply_or_recv!(deep_merge_apply(recv, other))
            }
            (BuiltinMethod::Defaults, BuiltinArgs::Val(other)) => {
                apply_or_recv!(defaults_apply(recv, other))
            }
            (BuiltinMethod::Rename, BuiltinArgs::Val(other)) => {
                apply_or_recv!(rename_apply(recv, other))
            }
            (BuiltinMethod::Explode, BuiltinArgs::Str(field)) => {
                apply_or_recv!(explode_apply(recv, field))
            }
            (BuiltinMethod::Implode, BuiltinArgs::Str(field)) => {
                apply_or_recv!(implode_apply(recv, field))
            }
            (BuiltinMethod::Has, BuiltinArgs::Str(k)) => apply_or_recv!(has_apply(recv, k)),
            (BuiltinMethod::GetPath, BuiltinArgs::Str(p)) => {
                apply_or_recv!(get_path_apply(recv, p))
            }
            (BuiltinMethod::HasPath, BuiltinArgs::Str(p)) => {
                apply_or_recv!(has_path_apply(recv, p))
            }
            (BuiltinMethod::DelPath, BuiltinArgs::Str(p)) => {
                apply_or_recv!(del_path_apply(recv, p))
            }
            (BuiltinMethod::FlattenKeys, BuiltinArgs::Str(p)) => {
                apply_or_recv!(flatten_keys_apply(recv, p))
            }
            (BuiltinMethod::UnflattenKeys, BuiltinArgs::Str(p)) => {
                apply_or_recv!(unflatten_keys_apply(recv, p))
            }
            (BuiltinMethod::StartsWith, BuiltinArgs::Str(p)) => {
                apply_or_recv!(starts_with_apply(recv, p))
            }
            (BuiltinMethod::EndsWith, BuiltinArgs::Str(p)) => {
                apply_or_recv!(ends_with_apply(recv, p))
            }
            (BuiltinMethod::StripPrefix, BuiltinArgs::Str(p)) => {
                apply_or_recv!(strip_prefix_apply(recv, p))
            }
            (BuiltinMethod::StripSuffix, BuiltinArgs::Str(p)) => {
                apply_or_recv!(strip_suffix_apply(recv, p))
            }
            (BuiltinMethod::Matches, BuiltinArgs::Str(p)) => {
                apply_or_recv!(contains_apply(recv, p))
            }
            (BuiltinMethod::IndexOf, BuiltinArgs::Str(p)) => {
                apply_or_recv!(index_of_apply(recv, p))
            }
            (BuiltinMethod::LastIndexOf, BuiltinArgs::Str(p)) => {
                apply_or_recv!(last_index_of_apply(recv, p))
            }
            (BuiltinMethod::Scan, BuiltinArgs::Str(p)) => apply_or_recv!(scan_apply(recv, p)),
            (BuiltinMethod::Split, BuiltinArgs::Str(p)) => apply_or_recv!(split_apply(recv, p)),
            (BuiltinMethod::Slice, BuiltinArgs::I64Opt { first, second }) => {
                return Some(slice_apply(recv.clone(), *first, *second));
            }
            (BuiltinMethod::Replace, BuiltinArgs::StrPair { first, second }) => {
                apply_or_recv!(replace_apply(recv.clone(), first, second, false))
            }
            (BuiltinMethod::ReplaceAll, BuiltinArgs::StrPair { first, second }) => {
                apply_or_recv!(replace_apply(recv.clone(), first, second, true))
            }
            (BuiltinMethod::ReMatch, BuiltinArgs::Str(p)) => {
                apply_or_recv!(re_match_apply(recv, p))
            }
            (BuiltinMethod::ReMatchFirst, BuiltinArgs::Str(p)) => {
                apply_or_recv!(re_match_first_apply(recv, p))
            }
            (BuiltinMethod::ReMatchAll, BuiltinArgs::Str(p)) => {
                apply_or_recv!(re_match_all_apply(recv, p))
            }
            (BuiltinMethod::ReCaptures, BuiltinArgs::Str(p)) => {
                apply_or_recv!(re_captures_apply(recv, p))
            }
            (BuiltinMethod::ReCapturesAll, BuiltinArgs::Str(p)) => {
                apply_or_recv!(re_captures_all_apply(recv, p))
            }
            (BuiltinMethod::ReSplit, BuiltinArgs::Str(p)) => {
                apply_or_recv!(re_split_apply(recv, p))
            }
            (BuiltinMethod::ReReplace, BuiltinArgs::StrPair { first, second }) => {
                apply_or_recv!(re_replace_apply(recv, first, second))
            }
            (BuiltinMethod::ReReplaceAll, BuiltinArgs::StrPair { first, second }) => {
                apply_or_recv!(re_replace_all_apply(recv, first, second))
            }
            (BuiltinMethod::ContainsAny, BuiltinArgs::StrVec(ns)) => {
                apply_or_recv!(contains_any_apply(recv, ns))
            }
            (BuiltinMethod::ContainsAll, BuiltinArgs::StrVec(ns)) => {
                apply_or_recv!(contains_all_apply(recv, ns))
            }
            (BuiltinMethod::Pick, BuiltinArgs::StrVec(keys)) => {
                apply_or_recv!(pick_apply(recv, keys))
            }
            (BuiltinMethod::Omit, BuiltinArgs::StrVec(keys)) => {
                apply_or_recv!(omit_apply(recv, keys))
            }
            (BuiltinMethod::Repeat, BuiltinArgs::Usize(n)) => {
                apply_or_recv!(repeat_apply(recv, *n))
            }
            (BuiltinMethod::Indent, BuiltinArgs::Usize(n)) => {
                apply_or_recv!(indent_apply(recv, *n))
            }
            (BuiltinMethod::PadLeft, BuiltinArgs::Pad { width, fill }) => {
                apply_or_recv!(pad_left_apply(recv, *width, *fill))
            }
            (BuiltinMethod::PadRight, BuiltinArgs::Pad { width, fill }) => {
                apply_or_recv!(pad_right_apply(recv, *width, *fill))
            }
            (BuiltinMethod::Center, BuiltinArgs::Pad { width, fill }) => {
                apply_or_recv!(center_apply(recv, *width, *fill))
            }
            _ => None,
        }
    }

    pub fn try_apply(&self, recv: &Val) -> Result<Option<Val>, EvalError> {
        match (self.method, &self.args) {
            (BuiltinMethod::ReMatch, BuiltinArgs::Str(p)) => try_re_match_apply(recv, p),
            (BuiltinMethod::ReMatchFirst, BuiltinArgs::Str(p)) => try_re_match_first_apply(recv, p),
            (BuiltinMethod::ReMatchAll, BuiltinArgs::Str(p)) => try_re_match_all_apply(recv, p),
            (BuiltinMethod::ReCaptures, BuiltinArgs::Str(p)) => try_re_captures_apply(recv, p),
            (BuiltinMethod::ReCapturesAll, BuiltinArgs::Str(p)) => {
                try_re_captures_all_apply(recv, p)
            }
            (BuiltinMethod::ReSplit, BuiltinArgs::Str(p)) => try_re_split_apply(recv, p),
            (BuiltinMethod::ReReplace, BuiltinArgs::StrPair { first, second }) => {
                try_re_replace_apply(recv, first, second)
            }
            (BuiltinMethod::ReReplaceAll, BuiltinArgs::StrPair { first, second }) => {
                try_re_replace_all_apply(recv, first, second)
            }
            (BuiltinMethod::FromJson, BuiltinArgs::None) => try_from_json_apply(recv),
            (BuiltinMethod::Join, BuiltinArgs::Str(sep)) => join_apply(recv, sep)
                .map(Some)
                .ok_or_else(|| EvalError("join: expected array".into())),
            (BuiltinMethod::Enumerate, BuiltinArgs::None) => enumerate_apply(recv)
                .map(Some)
                .ok_or_else(|| EvalError("enumerate: expected array".into())),
            (BuiltinMethod::Index, BuiltinArgs::Val(item)) => index_value_apply(recv, item)
                .map(Some)
                .ok_or_else(|| EvalError("index: expected array".into())),
            (BuiltinMethod::IndicesOf, BuiltinArgs::Val(item)) => indices_of_apply(recv, item)
                .map(Some)
                .ok_or_else(|| EvalError("indices_of: expected array".into())),
            (BuiltinMethod::Ceil, BuiltinArgs::None) => try_ceil_apply(recv),
            (BuiltinMethod::Floor, BuiltinArgs::None) => try_floor_apply(recv),
            (BuiltinMethod::Round, BuiltinArgs::None) => try_round_apply(recv),
            (BuiltinMethod::Abs, BuiltinArgs::None) => try_abs_apply(recv),
            (BuiltinMethod::RollingSum, BuiltinArgs::Usize(0)) => {
                Err(EvalError("rolling_sum: window must be > 0".into()))
            }
            (BuiltinMethod::RollingAvg, BuiltinArgs::Usize(0)) => {
                Err(EvalError("rolling_avg: window must be > 0".into()))
            }
            (BuiltinMethod::RollingMin, BuiltinArgs::Usize(0)) => {
                Err(EvalError("rolling_min: window must be > 0".into()))
            }
            (BuiltinMethod::RollingMax, BuiltinArgs::Usize(0)) => {
                Err(EvalError("rolling_max: window must be > 0".into()))
            }
            (BuiltinMethod::RollingSum, BuiltinArgs::Usize(_))
            | (BuiltinMethod::RollingAvg, BuiltinArgs::Usize(_))
            | (BuiltinMethod::RollingMin, BuiltinArgs::Usize(_))
            | (BuiltinMethod::RollingMax, BuiltinArgs::Usize(_))
            | (BuiltinMethod::Lag, BuiltinArgs::Usize(_))
            | (BuiltinMethod::Lead, BuiltinArgs::Usize(_))
            | (BuiltinMethod::DiffWindow, BuiltinArgs::None)
            | (BuiltinMethod::PctChange, BuiltinArgs::None)
            | (BuiltinMethod::CumMax, BuiltinArgs::None)
            | (BuiltinMethod::CumMin, BuiltinArgs::None)
            | (BuiltinMethod::Zscore, BuiltinArgs::None) => self
                .apply(recv)
                .map(Some)
                .ok_or_else(|| EvalError("expected numeric array".into())),
            _ => Ok(self.apply(recv)),
        }
    }

    pub fn from_static_args<E, I>(
        method: BuiltinMethod,
        name: &str,
        arg_len: usize,
        eval_arg: E,
        ident_arg: I,
    ) -> Result<Option<Self>, EvalError>
    where
        E: FnMut(usize) -> Result<Option<Val>, EvalError>,
        I: FnMut(usize) -> Option<Arc<str>>,
    {
        if method == BuiltinMethod::Unknown {
            return Ok(None);
        }

        let mut args = StaticArgDecoder {
            name,
            eval_arg,
            ident_arg,
        };

        let call = match method {
            BuiltinMethod::Flatten => {
                let depth = if arg_len > 0 { args.usize(0)? } else { 1 };
                Self::new(method, BuiltinArgs::Usize(depth))
            }
            BuiltinMethod::First | BuiltinMethod::Last => {
                let n = if arg_len > 0 { args.i64(0)? } else { 1 };
                Self::new(method, BuiltinArgs::I64(n))
            }
            BuiltinMethod::Nth => Self::new(method, BuiltinArgs::I64(args.i64(0)?)),
            BuiltinMethod::Append | BuiltinMethod::Prepend | BuiltinMethod::Set => {
                let item = if arg_len > 0 { args.val(0)? } else { Val::Null };
                Self::new(method, BuiltinArgs::Val(item))
            }
            BuiltinMethod::Or => {
                let default = if arg_len > 0 { args.val(0)? } else { Val::Null };
                Self::new(method, BuiltinArgs::Val(default))
            }
            BuiltinMethod::Includes | BuiltinMethod::Index | BuiltinMethod::IndicesOf => {
                Self::new(method, BuiltinArgs::Val(args.val(0)?))
            }
            BuiltinMethod::Diff | BuiltinMethod::Intersect | BuiltinMethod::Union => {
                Self::new(method, BuiltinArgs::ValVec(args.vec(0)?))
            }
            BuiltinMethod::Window
            | BuiltinMethod::Chunk
            | BuiltinMethod::RollingSum
            | BuiltinMethod::RollingAvg
            | BuiltinMethod::RollingMin
            | BuiltinMethod::RollingMax => Self::new(method, BuiltinArgs::Usize(args.usize(0)?)),
            BuiltinMethod::Lag | BuiltinMethod::Lead => {
                let n = if arg_len > 0 { args.usize(0)? } else { 1 };
                Self::new(method, BuiltinArgs::Usize(n))
            }
            BuiltinMethod::Merge
            | BuiltinMethod::DeepMerge
            | BuiltinMethod::Defaults
            | BuiltinMethod::Rename => Self::new(method, BuiltinArgs::Val(args.val(0)?)),
            BuiltinMethod::Slice => {
                let start = args.i64(0)?;
                let end = if arg_len > 1 {
                    Some(args.i64(1)?)
                } else {
                    None
                };
                Self::new(
                    method,
                    BuiltinArgs::I64Opt {
                        first: start,
                        second: end,
                    },
                )
            }
            BuiltinMethod::GetPath
            | BuiltinMethod::HasPath
            | BuiltinMethod::Has
            | BuiltinMethod::Join
            | BuiltinMethod::Explode
            | BuiltinMethod::Implode
            | BuiltinMethod::DelPath
            | BuiltinMethod::FlattenKeys
            | BuiltinMethod::UnflattenKeys
            | BuiltinMethod::Missing
            | BuiltinMethod::StartsWith
            | BuiltinMethod::EndsWith
            | BuiltinMethod::IndexOf
            | BuiltinMethod::LastIndexOf
            | BuiltinMethod::StripPrefix
            | BuiltinMethod::StripSuffix
            | BuiltinMethod::Matches
            | BuiltinMethod::Scan
            | BuiltinMethod::Split
            | BuiltinMethod::ReMatch
            | BuiltinMethod::ReMatchFirst
            | BuiltinMethod::ReMatchAll
            | BuiltinMethod::ReCaptures
            | BuiltinMethod::ReCapturesAll
            | BuiltinMethod::ReSplit => {
                let s = if arg_len > 0 {
                    args.str(0)?
                } else if matches!(method, BuiltinMethod::Join) {
                    Arc::from("")
                } else if matches!(
                    method,
                    BuiltinMethod::FlattenKeys | BuiltinMethod::UnflattenKeys
                ) {
                    Arc::from(".")
                } else {
                    return Ok(None);
                };
                Self::new(method, BuiltinArgs::Str(s))
            }
            BuiltinMethod::Replace
            | BuiltinMethod::ReplaceAll
            | BuiltinMethod::ReReplace
            | BuiltinMethod::ReReplaceAll => Self::new(
                method,
                BuiltinArgs::StrPair {
                    first: args.str(0)?,
                    second: args.str(1)?,
                },
            ),
            BuiltinMethod::ContainsAny | BuiltinMethod::ContainsAll => {
                Self::new(method, BuiltinArgs::StrVec(args.str_vec(0)?))
            }
            BuiltinMethod::Repeat => Self::new(method, BuiltinArgs::Usize(args.usize(0)?)),
            BuiltinMethod::Indent => {
                let n = if arg_len > 0 { args.usize(0)? } else { 2 };
                Self::new(method, BuiltinArgs::Usize(n))
            }
            BuiltinMethod::PadLeft | BuiltinMethod::PadRight | BuiltinMethod::Center => Self::new(
                method,
                BuiltinArgs::Pad {
                    width: args.usize(0)?,
                    fill: args.char(1, arg_len)?,
                },
            ),
            _ if arg_len == 0 => Self::new(method, BuiltinArgs::None),
            _ => return Ok(None),
        };
        Ok(Some(call))
    }

    pub fn from_literal_ast_args(name: &str, args: &[crate::ast::Arg]) -> Option<Self> {
        use crate::ast::{Arg, ArrayElem, Expr, ObjField};

        let method = BuiltinMethod::from_name(name);
        if method == BuiltinMethod::Unknown {
            return None;
        }

        fn literal_val(expr: &Expr) -> Option<Val> {
            match expr {
                Expr::Null => Some(Val::Null),
                Expr::Bool(b) => Some(Val::Bool(*b)),
                Expr::Int(n) => Some(Val::Int(*n)),
                Expr::Float(f) => Some(Val::Float(*f)),
                Expr::Str(s) => Some(Val::Str(Arc::from(s.as_str()))),
                Expr::Array(elems) => {
                    let mut out = Vec::with_capacity(elems.len());
                    for elem in elems {
                        match elem {
                            ArrayElem::Expr(expr) => out.push(literal_val(expr)?),
                            ArrayElem::Spread(_) => return None,
                        }
                    }
                    Some(Val::Arr(Arc::new(out)))
                }
                Expr::Object(fields) => {
                    let mut out = IndexMap::with_capacity(fields.len());
                    for field in fields {
                        match field {
                            ObjField::Kv {
                                key,
                                val,
                                optional: false,
                                cond: None,
                            } => {
                                out.insert(Arc::from(key.as_str()), literal_val(val)?);
                            }
                            _ => return None,
                        }
                    }
                    Some(Val::Obj(Arc::new(out)))
                }
                _ => None,
            }
        }

        Self::from_static_args(
            method,
            name,
            args.len(),
            |idx| {
                Ok(match args.get(idx) {
                    Some(Arg::Pos(expr)) => literal_val(expr),
                    _ => None,
                })
            },
            |idx| match args.get(idx) {
                Some(Arg::Pos(Expr::Ident(value))) => Some(Arc::from(value.as_str())),
                _ => None,
            },
        )
        .ok()
        .flatten()
    }

    /// Lower a method call into a pure per-element pipeline builtin.
    ///
    /// This intentionally excludes whole-receiver methods such as `compact`,
    /// `flatten`, `join`, `rolling_*`, etc.  Pipeline lowering applies stages
    /// to each row yielded by the source stream, so only methods whose
    /// semantics are valid per input value belong here.
    pub fn from_pipeline_literal_args(name: &str, args: &[crate::ast::Arg]) -> Option<Self> {
        let call = Self::from_literal_ast_args(name, args)?;
        call.method.is_pipeline_element_method().then_some(call)
    }

    pub fn try_apply_json_view(&self, recv: crate::util::JsonView<'_>) -> Option<Val> {
        if !self.spec().view_scalar {
            return None;
        }
        match (self.method, &self.args) {
            (BuiltinMethod::Len, BuiltinArgs::None) => json_view_len(recv).map(Val::Int),
            (BuiltinMethod::StartsWith, BuiltinArgs::Str(prefix)) => {
                json_view_str(recv).map(|value| Val::Bool(value.starts_with(prefix.as_ref())))
            }
            (BuiltinMethod::EndsWith, BuiltinArgs::Str(suffix)) => {
                json_view_str(recv).map(|value| Val::Bool(value.ends_with(suffix.as_ref())))
            }
            _ => None,
        }
    }
}

#[inline]
fn json_view_len(recv: crate::util::JsonView<'_>) -> Option<i64> {
    match recv {
        crate::util::JsonView::Str(s) => Some(s.chars().count() as i64),
        crate::util::JsonView::ArrayLen(n) | crate::util::JsonView::ObjectLen(n) => Some(n as i64),
        _ => None,
    }
}

#[inline]
fn json_view_str(recv: crate::util::JsonView<'_>) -> Option<&str> {
    match recv {
        crate::util::JsonView::Str(s) => Some(s),
        _ => None,
    }
}

/// Direct VM fallback adapter for builtins with AST arguments.
///
/// This replaces the old dynamic builtin hash table. It resolves by
/// `BuiltinMethod`, evaluates only the arguments needed for the resolved
/// builtin, then executes the canonical `BuiltinCall` implementation in this
/// module.
pub(crate) fn eval_builtin_method<F, G, H>(
    recv: Val,
    name: &str,
    args: &[crate::ast::Arg],
    mut eval_arg: F,
    mut eval_item: G,
    mut eval_pair: H,
) -> Result<Val, EvalError>
where
    F: FnMut(&crate::ast::Arg) -> Result<Val, EvalError>,
    G: FnMut(&Val, &crate::ast::Arg) -> Result<Val, EvalError>,
    H: FnMut(&Val, &Val, &crate::ast::Arg) -> Result<Val, EvalError>,
{
    use crate::ast::{Arg, Expr, ObjField};

    let method = BuiltinMethod::from_name(name);
    if method == BuiltinMethod::Unknown {
        return Err(EvalError(format!("unknown method '{}'", name)));
    }

    macro_rules! arg_val {
        ($idx:expr) => {{
            let arg = args
                .get($idx)
                .ok_or_else(|| EvalError(format!("{}: missing argument", name)))?;
            eval_arg(arg)
        }};
    }

    macro_rules! str_arg {
        ($idx:expr) => {{
            match args.get($idx) {
                Some(Arg::Pos(Expr::Ident(s))) => Ok(Arc::from(s.as_str())),
                Some(_) => match arg_val!($idx)? {
                    Val::Str(s) => Ok(s),
                    other => Ok(Arc::from(crate::util::val_to_string(&other).as_str())),
                },
                None => Err(EvalError(format!("{}: missing argument", name))),
            }
        }};
    }

    macro_rules! i64_arg {
        ($idx:expr) => {{
            match arg_val!($idx)? {
                Val::Int(n) => Ok(n),
                Val::Float(f) => Ok(f as i64),
                _ => Err(EvalError(format!("{}: expected number argument", name))),
            }
        }};
    }

    macro_rules! vec_arg {
        ($idx:expr) => {{
            arg_val!($idx)?
                .into_vec()
                .ok_or_else(|| EvalError(format!("{}: expected array arg", name)))
        }};
    }

    macro_rules! str_vec_arg {
        ($idx:expr) => {{
            Ok(vec_arg!($idx)?
                .iter()
                .map(|v| match v {
                    Val::Str(s) => s.clone(),
                    other => Arc::from(crate::util::val_to_string(other).as_str()),
                })
                .collect())
        }};
    }

    macro_rules! fill_arg {
        ($idx:expr) => {{
            match args.get($idx) {
                None => Ok(' '),
                Some(_) => {
                    let s = str_arg!($idx)?;
                    if s.chars().count() == 1 {
                        Ok(s.chars().next().unwrap())
                    } else {
                        Err(EvalError(format!(
                            "{}: filler must be a single-char string",
                            name
                        )))
                    }
                }
            }
        }};
    }

    let call = match method {
        BuiltinMethod::Len
        | BuiltinMethod::Count
        | BuiltinMethod::Sum
        | BuiltinMethod::Avg
        | BuiltinMethod::Min
        | BuiltinMethod::Max
        | BuiltinMethod::Keys
        | BuiltinMethod::Values
        | BuiltinMethod::Entries
        | BuiltinMethod::Reverse
        | BuiltinMethod::Unique
        | BuiltinMethod::Collect
        | BuiltinMethod::Compact
        | BuiltinMethod::FromJson
        | BuiltinMethod::FromPairs
        | BuiltinMethod::ToPairs
        | BuiltinMethod::Invert
        | BuiltinMethod::Enumerate
        | BuiltinMethod::Pairwise
        | BuiltinMethod::Ceil
        | BuiltinMethod::Floor
        | BuiltinMethod::Round
        | BuiltinMethod::Abs
        | BuiltinMethod::DiffWindow
        | BuiltinMethod::PctChange
        | BuiltinMethod::CumMax
        | BuiltinMethod::CumMin
        | BuiltinMethod::Zscore
        | BuiltinMethod::Upper
        | BuiltinMethod::Lower
        | BuiltinMethod::Trim
        | BuiltinMethod::TrimLeft
        | BuiltinMethod::TrimRight
        | BuiltinMethod::Capitalize
        | BuiltinMethod::TitleCase
        | BuiltinMethod::SnakeCase
        | BuiltinMethod::KebabCase
        | BuiltinMethod::CamelCase
        | BuiltinMethod::PascalCase
        | BuiltinMethod::ReverseStr
        | BuiltinMethod::HtmlEscape
        | BuiltinMethod::HtmlUnescape
        | BuiltinMethod::UrlEncode
        | BuiltinMethod::UrlDecode
        | BuiltinMethod::ToBase64
        | BuiltinMethod::FromBase64
        | BuiltinMethod::Dedent
        | BuiltinMethod::Lines
        | BuiltinMethod::Words
        | BuiltinMethod::Chars
        | BuiltinMethod::CharsOf
        | BuiltinMethod::Bytes
        | BuiltinMethod::ByteLen
        | BuiltinMethod::IsBlank
        | BuiltinMethod::IsNumeric
        | BuiltinMethod::IsAlpha
        | BuiltinMethod::IsAscii
        | BuiltinMethod::ToNumber
        | BuiltinMethod::ToBool
        | BuiltinMethod::ParseInt
        | BuiltinMethod::ParseFloat
        | BuiltinMethod::ParseBool
        | BuiltinMethod::Type
        | BuiltinMethod::ToString
        | BuiltinMethod::ToJson
        | BuiltinMethod::ToCsv
        | BuiltinMethod::ToTsv
        | BuiltinMethod::Schema
            if args.is_empty() =>
        {
            BuiltinCall::new(method, BuiltinArgs::None)
        }
        BuiltinMethod::Sum | BuiltinMethod::Avg | BuiltinMethod::Min | BuiltinMethod::Max => {
            return numeric_aggregate_projected_apply(&recv, method, |item| {
                eval_item(item, &args[0])
            });
        }
        BuiltinMethod::Count => {
            let items = recv
                .as_vals()
                .ok_or_else(|| EvalError("count: expected array".into()))?;
            let mut n: i64 = 0;
            for item in items.iter() {
                if crate::util::is_truthy(&eval_item(item, &args[0])?) {
                    n += 1;
                }
            }
            return Ok(Val::Int(n));
        }
        BuiltinMethod::Find | BuiltinMethod::FindAll => {
            return find_apply(recv, args.len(), |item, idx| eval_item(item, &args[idx]));
        }
        BuiltinMethod::FindIndex => {
            return find_index_apply(recv, args.len(), |item, idx| eval_item(item, &args[idx]));
        }
        BuiltinMethod::IndicesWhere => {
            return indices_where_apply(recv, args.len(), |item, idx| eval_item(item, &args[idx]));
        }
        BuiltinMethod::UniqueBy => {
            let key_arg = args
                .first()
                .ok_or_else(|| EvalError("unique_by: requires key fn".into()))?;
            return unique_by_apply(recv, |item| eval_item(item, key_arg));
        }
        BuiltinMethod::MaxBy | BuiltinMethod::MinBy => {
            let key_arg = args
                .first()
                .ok_or_else(|| EvalError(format!("{}: requires a key expression", name)))?;
            return extreme_by_apply(recv, method == BuiltinMethod::MaxBy, |item| {
                eval_item(item, key_arg)
            });
        }
        BuiltinMethod::DeepFind => {
            return deep_find_apply(recv, args.len(), |item, idx| eval_item(item, &args[idx]));
        }
        BuiltinMethod::DeepShape => {
            let arg = args
                .first()
                .ok_or_else(|| EvalError("shape: requires pattern".into()))?;
            let expr = match arg {
                Arg::Pos(e) | Arg::Named(_, e) => e,
            };
            let Expr::Object(fields) = expr else {
                return Err(EvalError(
                    "shape: expected `{k1, k2, ...}` object pattern".into(),
                ));
            };
            let mut keys = Vec::with_capacity(fields.len());
            for field in fields {
                match field {
                    ObjField::Short(k) => keys.push(Arc::from(k.as_str())),
                    ObjField::Kv { key, val, .. } if matches!(val, Expr::Ident(n) if n == key) => {
                        keys.push(Arc::from(key.as_str()));
                    }
                    _ => return Err(EvalError("shape: unsupported pattern field".into())),
                }
            }
            return deep_shape_apply(recv, &keys);
        }
        BuiltinMethod::DeepLike => {
            let arg = args
                .first()
                .ok_or_else(|| EvalError("like: requires pattern".into()))?;
            let expr = match arg {
                Arg::Pos(e) | Arg::Named(_, e) => e,
            };
            let Expr::Object(fields) = expr else {
                return Err(EvalError(
                    "like: expected `{k: lit, ...}` object pattern".into(),
                ));
            };
            let mut pats = Vec::with_capacity(fields.len());
            for field in fields {
                match field {
                    ObjField::Kv { key, val, .. } => {
                        pats.push((Arc::from(key.as_str()), eval_arg(&Arg::Pos(val.clone()))?));
                    }
                    ObjField::Short(k) => {
                        pats.push((
                            Arc::from(k.as_str()),
                            eval_arg(&Arg::Pos(Expr::Ident(k.clone())))?,
                        ));
                    }
                    _ => return Err(EvalError("like: unsupported pattern field".into())),
                }
            }
            return deep_like_apply(recv, &pats);
        }
        BuiltinMethod::Walk | BuiltinMethod::WalkPre => {
            let arg = args
                .first()
                .ok_or_else(|| EvalError("walk: requires fn".into()))?;
            let pre = method == BuiltinMethod::WalkPre;
            let mut eval = |value: Val| eval_item(&value, arg);
            return walk_apply(recv, pre, &mut eval);
        }
        BuiltinMethod::Rec => {
            let arg = args
                .first()
                .ok_or_else(|| EvalError("rec: requires step expression".into()))?;
            return rec_apply(recv, |value| eval_item(&value, arg));
        }
        BuiltinMethod::TracePath => {
            let arg = args
                .first()
                .ok_or_else(|| EvalError("trace_path: requires predicate".into()))?;
            return trace_path_apply(recv, |value| eval_item(value, arg));
        }
        BuiltinMethod::Fanout => {
            return fanout_apply(&recv, args.len(), |value, idx| eval_item(value, &args[idx]));
        }
        BuiltinMethod::ZipShape => {
            let mut names = Vec::with_capacity(args.len());
            for arg in args {
                let name: Arc<str> = match arg {
                    Arg::Named(n, _) => Arc::from(n.as_str()),
                    Arg::Pos(Expr::Ident(n)) => Arc::from(n.as_str()),
                    _ => {
                        return Err(EvalError(
                            "zip_shape: args must be `name = expr` or bare identifier".into(),
                        ))
                    }
                };
                names.push(name);
            }
            return zip_shape_apply(&recv, &names, |value, idx| eval_item(value, &args[idx]));
        }
        BuiltinMethod::GroupShape => {
            let key_arg = args
                .first()
                .ok_or_else(|| EvalError("group_shape: requires key".into()))?;
            let shape_arg = args
                .get(1)
                .ok_or_else(|| EvalError("group_shape: requires shape".into()))?;
            return group_shape_apply(recv, |value, idx| {
                if idx == 0 {
                    eval_item(&value, key_arg)
                } else {
                    eval_item(&value, shape_arg)
                }
            });
        }
        BuiltinMethod::Sort => {
            if args.is_empty() {
                return sort_apply(recv);
            }
            let mut key_args = Vec::with_capacity(args.len());
            let mut desc = Vec::with_capacity(args.len());
            for arg in args {
                match arg {
                    Arg::Pos(Expr::Lambda { params, .. })
                    | Arg::Named(_, Expr::Lambda { params, .. })
                        if params.len() == 2 =>
                    {
                        return sort_comparator_apply(recv, |left, right| {
                            eval_pair(left, right, arg)
                        });
                    }
                    Arg::Pos(Expr::UnaryNeg(inner)) => {
                        desc.push(true);
                        key_args.push(Arg::Pos((**inner).clone()));
                    }
                    Arg::Pos(e) => {
                        desc.push(false);
                        key_args.push(Arg::Pos(e.clone()));
                    }
                    Arg::Named(name, Expr::UnaryNeg(inner)) => {
                        desc.push(true);
                        key_args.push(Arg::Named(name.clone(), (**inner).clone()));
                    }
                    Arg::Named(name, e) => {
                        desc.push(false);
                        key_args.push(Arg::Named(name.clone(), e.clone()));
                    }
                }
            }
            return sort_by_apply(recv, &desc, |item, idx| eval_item(item, &key_args[idx]));
        }
        BuiltinMethod::Flatten => {
            let depth = if args.is_empty() {
                1
            } else {
                i64_arg!(0)?.max(0) as usize
            };
            BuiltinCall::new(method, BuiltinArgs::Usize(depth))
        }
        BuiltinMethod::First | BuiltinMethod::Last => {
            let n = if args.is_empty() { 1 } else { i64_arg!(0)? };
            BuiltinCall::new(method, BuiltinArgs::I64(n))
        }
        BuiltinMethod::Nth => BuiltinCall::new(method, BuiltinArgs::I64(i64_arg!(0)?)),
        BuiltinMethod::Append | BuiltinMethod::Prepend | BuiltinMethod::Set => {
            let item = if args.is_empty() {
                Val::Null
            } else {
                arg_val!(0)?
            };
            BuiltinCall::new(method, BuiltinArgs::Val(item))
        }
        BuiltinMethod::Or => {
            let default = if args.is_empty() {
                Val::Null
            } else {
                arg_val!(0)?
            };
            BuiltinCall::new(method, BuiltinArgs::Val(default))
        }
        BuiltinMethod::Includes | BuiltinMethod::Index | BuiltinMethod::IndicesOf => {
            BuiltinCall::new(method, BuiltinArgs::Val(arg_val!(0)?))
        }
        BuiltinMethod::Diff | BuiltinMethod::Intersect | BuiltinMethod::Union => {
            BuiltinCall::new(method, BuiltinArgs::ValVec(vec_arg!(0)?))
        }
        BuiltinMethod::Window
        | BuiltinMethod::Chunk
        | BuiltinMethod::RollingSum
        | BuiltinMethod::RollingAvg
        | BuiltinMethod::RollingMin
        | BuiltinMethod::RollingMax => {
            BuiltinCall::new(method, BuiltinArgs::Usize(i64_arg!(0)?.max(0) as usize))
        }
        BuiltinMethod::Lag | BuiltinMethod::Lead => {
            let n = if args.is_empty() {
                1
            } else {
                i64_arg!(0)?.max(0) as usize
            };
            BuiltinCall::new(method, BuiltinArgs::Usize(n))
        }
        BuiltinMethod::Merge
        | BuiltinMethod::DeepMerge
        | BuiltinMethod::Defaults
        | BuiltinMethod::Rename => BuiltinCall::new(method, BuiltinArgs::Val(arg_val!(0)?)),
        BuiltinMethod::Remove => match args.first() {
            Some(Arg::Pos(Expr::Lambda { .. })) | Some(Arg::Named(_, Expr::Lambda { .. })) => {
                return remove_predicate_apply(recv, |item| eval_item(item, &args[0]));
            }
            Some(_) => BuiltinCall::new(method, BuiltinArgs::Val(arg_val!(0)?)),
            None => return Err(EvalError("remove: requires arg".into())),
        },
        BuiltinMethod::Zip => {
            let other = args
                .first()
                .map(|arg| eval_arg(arg))
                .transpose()?
                .unwrap_or_else(|| Val::arr(Vec::new()));
            return zip_apply(recv, other);
        }
        BuiltinMethod::ZipLongest => {
            let mut other = Val::arr(Vec::new());
            let mut fill = Val::Null;
            for arg in args {
                match arg {
                    Arg::Pos(_) => other = eval_arg(arg)?,
                    Arg::Named(n, _) if n == "fill" => fill = eval_arg(arg)?,
                    Arg::Named(_, _) => {}
                }
            }
            return zip_longest_apply(recv, other, fill);
        }
        BuiltinMethod::EquiJoin => {
            let other = arg_val!(0)?;
            let lhs_key = str_arg!(1)?;
            let rhs_key = str_arg!(2)?;
            return equi_join_apply(recv, other, &lhs_key, &rhs_key);
        }
        BuiltinMethod::Pivot => {
            return pivot_apply(recv, args.len(), |item, idx| match &args[idx] {
                Arg::Pos(Expr::Str(s)) | Arg::Named(_, Expr::Str(s)) => {
                    Ok(item.get_field(s.as_str()))
                }
                arg => eval_item(item, arg),
            });
        }
        BuiltinMethod::Slice => {
            let start = i64_arg!(0)?;
            let end = if args.len() > 1 {
                Some(i64_arg!(1)?)
            } else {
                None
            };
            BuiltinCall::new(
                method,
                BuiltinArgs::I64Opt {
                    first: start,
                    second: end,
                },
            )
        }
        BuiltinMethod::Join => {
            let sep = if args.is_empty() {
                Arc::from("")
            } else {
                str_arg!(0)?
            };
            BuiltinCall::new(method, BuiltinArgs::Str(sep))
        }
        BuiltinMethod::FlattenKeys | BuiltinMethod::UnflattenKeys if args.is_empty() => {
            BuiltinCall::new(method, BuiltinArgs::Str(Arc::from(".")))
        }
        BuiltinMethod::GetPath
        | BuiltinMethod::HasPath
        | BuiltinMethod::Has
        | BuiltinMethod::Missing
        | BuiltinMethod::Explode
        | BuiltinMethod::Implode
        | BuiltinMethod::DelPath
        | BuiltinMethod::FlattenKeys
        | BuiltinMethod::UnflattenKeys
        | BuiltinMethod::StartsWith
        | BuiltinMethod::EndsWith
        | BuiltinMethod::IndexOf
        | BuiltinMethod::LastIndexOf
        | BuiltinMethod::StripPrefix
        | BuiltinMethod::StripSuffix
        | BuiltinMethod::Matches
        | BuiltinMethod::Scan
        | BuiltinMethod::Split
        | BuiltinMethod::ReMatch
        | BuiltinMethod::ReMatchFirst
        | BuiltinMethod::ReMatchAll
        | BuiltinMethod::ReCaptures
        | BuiltinMethod::ReCapturesAll
        | BuiltinMethod::ReSplit => BuiltinCall::new(method, BuiltinArgs::Str(str_arg!(0)?)),
        BuiltinMethod::Replace
        | BuiltinMethod::ReplaceAll
        | BuiltinMethod::ReReplace
        | BuiltinMethod::ReReplaceAll => BuiltinCall::new(
            method,
            BuiltinArgs::StrPair {
                first: str_arg!(0)?,
                second: str_arg!(1)?,
            },
        ),
        BuiltinMethod::ContainsAny | BuiltinMethod::ContainsAll => {
            BuiltinCall::new(method, BuiltinArgs::StrVec(str_vec_arg!(0)?))
        }
        BuiltinMethod::Pick => {
            let mut specs = Vec::with_capacity(args.len());
            for arg in args {
                let resolved: Option<(Arc<str>, Arc<str>)> = match arg {
                    Arg::Pos(Expr::Ident(s)) => {
                        let key: Arc<str> = Arc::from(s.as_str());
                        Some((key.clone(), key))
                    }
                    Arg::Pos(_) => match eval_arg(arg)? {
                        Val::Str(s) => {
                            let out_key: Arc<str> = if s.contains('.') || s.contains('[') {
                                match parse_path_segs(&s).first() {
                                    Some(PathSeg::Field(f)) => Arc::from(f.as_str()),
                                    Some(PathSeg::Index(i)) => Arc::from(i.to_string().as_str()),
                                    None => s.clone(),
                                }
                            } else {
                                s.clone()
                            };
                            Some((out_key, s))
                        }
                        _ => None,
                    },
                    Arg::Named(alias, Expr::Ident(src)) => {
                        Some((Arc::from(alias.as_str()), Arc::from(src.as_str())))
                    }
                    Arg::Named(alias, _) => match eval_arg(arg)? {
                        Val::Str(s) => Some((Arc::from(alias.as_str()), s)),
                        _ => None,
                    },
                };
                let Some((out_key, src)) = resolved else {
                    continue;
                };
                let source = if src.contains('.') || src.contains('[') {
                    PickSource::Path(parse_path_segs(&src))
                } else {
                    PickSource::Field(src)
                };
                specs.push(PickSpec { out_key, source });
            }
            return pick_specs_apply(&recv, &specs)
                .ok_or_else(|| EvalError("pick: expected object or array of objects".into()));
        }
        BuiltinMethod::Omit => {
            let mut keys = Vec::with_capacity(args.len());
            for idx in 0..args.len() {
                keys.push(str_arg!(idx)?);
            }
            BuiltinCall::new(method, BuiltinArgs::StrVec(keys))
        }
        BuiltinMethod::Repeat | BuiltinMethod::Indent => {
            let n = if args.is_empty() {
                if matches!(method, BuiltinMethod::Indent) {
                    2
                } else {
                    1
                }
            } else {
                i64_arg!(0)?.max(0) as usize
            };
            BuiltinCall::new(method, BuiltinArgs::Usize(n))
        }
        BuiltinMethod::PadLeft | BuiltinMethod::PadRight | BuiltinMethod::Center => {
            BuiltinCall::new(
                method,
                BuiltinArgs::Pad {
                    width: i64_arg!(0)?.max(0) as usize,
                    fill: fill_arg!(1)?,
                },
            )
        }
        BuiltinMethod::SetPath => {
            return set_path_apply(&recv, &str_arg!(0)?, &arg_val!(1)?)
                .ok_or_else(|| EvalError("set_path: builtin unsupported".into()));
        }
        BuiltinMethod::DelPaths => {
            let mut paths = Vec::with_capacity(args.len());
            for idx in 0..args.len() {
                paths.push(str_arg!(idx)?);
            }
            return del_paths_apply(&recv, &paths)
                .ok_or_else(|| EvalError("del_paths: builtin unsupported".into()));
        }
        _ => {
            return Err(EvalError(format!(
                "{}: builtin not migrated to builtins.rs AST adapter",
                name
            )));
        }
    };

    call.try_apply(&recv)?
        .ok_or_else(|| EvalError(format!("{}: builtin unsupported", name)))
}

pub(crate) fn eval_builtin_no_args(recv: Val, name: &str) -> Result<Val, EvalError> {
    eval_builtin_method(
        recv,
        name,
        &[],
        |_| {
            Err(EvalError(format!(
                "{}: unexpected argument evaluation",
                name
            )))
        },
        |_, _| Err(EvalError(format!("{}: unexpected item evaluation", name))),
        |_, _, _| Err(EvalError(format!("{}: unexpected pair evaluation", name))),
    )
}

impl BuiltinMethod {
    #[inline]
    pub fn is_pipeline_element_method(self) -> bool {
        self.spec().pipeline_element
    }
}

// ── filter ──────────────────────────────────────────────────────────

/// Per-row filter decision — single source of truth for `.filter()`
/// semantics across all backends (Pipeline streaming arm, Pipeline
/// barrier arm, VM `BuiltinMethod::Filter`, composed runner
/// `GenericFilter`).
#[inline]
pub fn filter_one<F>(item: &Val, mut eval: F) -> Result<bool, EvalError>
where
    F: FnMut(&Val) -> Result<Val, EvalError>,
{
    Ok(is_truthy(&eval(item)?))
}

/// Buffered filter — `Vec<Val> → Vec<Val>` form, built on
/// `filter_one`.
#[inline]
pub fn filter_apply<F>(items: Vec<Val>, mut eval: F) -> Result<Vec<Val>, EvalError>
where
    F: FnMut(&Val) -> Result<Val, EvalError>,
{
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        if filter_one(&item, &mut eval)? {
            out.push(item);
        }
    }
    Ok(out)
}

/// Demand-aware filter — stops after `max_keep` items pass the
/// predicate.
///
/// Used by the planner / VM-peephole-fused `Filter ∘ Take(n)` shape
/// to avoid evaluating the predicate over the entire array when only
/// the first N keeps are needed. When `max_keep` is `None`, behaves
/// identically to `filter_apply`.
///
/// Trivial generalisation of `filter_apply` — same algorithm, one
/// extra termination check. Single source of truth keeps drift
/// impossible.
#[inline]
pub fn filter_apply_bounded<F>(
    items: Vec<Val>,
    max_keep: Option<usize>,
    mut eval: F,
) -> Result<Vec<Val>, EvalError>
where
    F: FnMut(&Val) -> Result<Val, EvalError>,
{
    let cap = match max_keep {
        Some(n) => n.min(items.len()),
        None => items.len(),
    };
    let mut out = Vec::with_capacity(cap);
    for item in items {
        if filter_one(&item, &mut eval)? {
            out.push(item);
            if let Some(n) = max_keep {
                if out.len() >= n {
                    break;
                }
            }
        }
    }
    Ok(out)
}

// ── map ─────────────────────────────────────────────────────────────

/// Per-row map transform — single source of truth for `.map()`.
#[inline]
pub fn map_one<F>(item: &Val, mut eval: F) -> Result<Val, EvalError>
where
    F: FnMut(&Val) -> Result<Val, EvalError>,
{
    eval(item)
}

/// Buffered map — `Vec<Val> → Vec<Val>`.
#[inline]
pub fn map_apply<F>(items: Vec<Val>, mut eval: F) -> Result<Vec<Val>, EvalError>
where
    F: FnMut(&Val) -> Result<Val, EvalError>,
{
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        out.push(map_one(&item, &mut eval)?);
    }
    Ok(out)
}

/// Demand-aware map — stops after `max_emit` items.
///
/// Used by the planner / VM-peephole-fused `Map ∘ Take(n)` shape.
/// Map is 1:1 so `max_emit = max_keep` here, but the helper is
/// kept distinct for symmetry with the filter shape.
#[inline]
pub fn map_apply_bounded<F>(
    items: Vec<Val>,
    max_emit: Option<usize>,
    mut eval: F,
) -> Result<Vec<Val>, EvalError>
where
    F: FnMut(&Val) -> Result<Val, EvalError>,
{
    let cap = match max_emit {
        Some(n) => n.min(items.len()),
        None => items.len(),
    };
    let mut out = Vec::with_capacity(cap);
    for item in items {
        out.push(map_one(&item, &mut eval)?);
        if let Some(n) = max_emit {
            if out.len() >= n {
                break;
            }
        }
    }
    Ok(out)
}

// ── flat_map ────────────────────────────────────────────────────────

/// Per-row flat_map — yields zero or more results. Backend
/// evaluator returns a single Val; if Arr, elements are flattened
/// one level into the output stream.
#[inline]
pub fn flat_map_one<F>(item: &Val, mut eval: F) -> Result<smallvec::SmallVec<[Val; 1]>, EvalError>
where
    F: FnMut(&Val) -> Result<Val, EvalError>,
{
    use std::sync::Arc;
    let r = eval(item)?;
    Ok(match r {
        Val::Arr(a) => {
            let v = Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone());
            v.into_iter().collect()
        }
        v => smallvec::smallvec![v],
    })
}

/// Buffered flat_map — `Vec<Val> → Vec<Val>`.
#[inline]
pub fn flat_map_apply<F>(items: Vec<Val>, mut eval: F) -> Result<Vec<Val>, EvalError>
where
    F: FnMut(&Val) -> Result<Val, EvalError>,
{
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        out.extend(flat_map_one(&item, &mut eval)?);
    }
    Ok(out)
}

// ── sort ───────────────────────────────────────────────────────────

#[inline]
pub fn sort_apply(recv: Val) -> Result<Val, EvalError> {
    match recv {
        Val::IntVec(a) => {
            let mut v = Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone());
            v.sort();
            Ok(Val::int_vec(v))
        }
        Val::FloatVec(a) => {
            let mut v = Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone());
            v.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
            Ok(Val::float_vec(v))
        }
        other => {
            let mut items = other
                .into_vec()
                .ok_or_else(|| EvalError("sort: expected array".into()))?;
            items.sort_by(cmp_vals);
            Ok(Val::arr(items))
        }
    }
}

#[inline]
pub fn sort_by_apply<F>(recv: Val, desc: &[bool], mut eval: F) -> Result<Val, EvalError>
where
    F: FnMut(&Val, usize) -> Result<Val, EvalError>,
{
    let items = recv
        .into_vec()
        .ok_or_else(|| EvalError("sort: expected array".into()))?;
    let mut keyed: Vec<(Vec<Val>, Val)> = Vec::with_capacity(items.len());
    for item in items {
        let mut keys = Vec::with_capacity(desc.len());
        for idx in 0..desc.len() {
            keys.push(eval(&item, idx)?);
        }
        keyed.push((keys, item));
    }
    keyed.sort_by(|(xk, _), (yk, _)| {
        for (idx, is_desc) in desc.iter().enumerate() {
            let ord = cmp_vals(&xk[idx], &yk[idx]);
            if ord != std::cmp::Ordering::Equal {
                return if *is_desc { ord.reverse() } else { ord };
            }
        }
        std::cmp::Ordering::Equal
    });
    Ok(Val::arr(keyed.into_iter().map(|(_, v)| v).collect()))
}

#[inline]
pub fn sort_comparator_apply<F>(recv: Val, mut eval_pair: F) -> Result<Val, EvalError>
where
    F: FnMut(&Val, &Val) -> Result<Val, EvalError>,
{
    let mut items = recv
        .into_vec()
        .ok_or_else(|| EvalError("sort: expected array".into()))?;
    let mut err_cell: Option<EvalError> = None;
    items.sort_by(|x, y| {
        if err_cell.is_some() {
            return std::cmp::Ordering::Equal;
        }
        match eval_pair(x, y) {
            Ok(Val::Bool(true)) => std::cmp::Ordering::Less,
            Ok(_) => std::cmp::Ordering::Greater,
            Err(e) => {
                err_cell = Some(e);
                std::cmp::Ordering::Equal
            }
        }
    });
    if let Some(e) = err_cell {
        Err(e)
    } else {
        Ok(Val::arr(items))
    }
}

#[inline]
pub fn remove_predicate_apply<F>(recv: Val, mut eval: F) -> Result<Val, EvalError>
where
    F: FnMut(&Val) -> Result<Val, EvalError>,
{
    let items = recv
        .into_vec()
        .ok_or_else(|| EvalError("remove: expected array".into()))?;
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        if !is_truthy(&eval(&item)?) {
            out.push(item);
        }
    }
    Ok(Val::arr(out))
}

#[inline]
pub fn find_apply<F>(recv: Val, pred_count: usize, mut eval: F) -> Result<Val, EvalError>
where
    F: FnMut(&Val, usize) -> Result<Val, EvalError>,
{
    if pred_count == 0 {
        return Err(EvalError("find: requires at least one predicate".into()));
    }
    let items = recv
        .into_vec()
        .ok_or_else(|| EvalError("find: expected array".into()))?;
    let mut out = Vec::with_capacity(items.len());
    'outer: for item in items {
        for idx in 0..pred_count {
            if !is_truthy(&eval(&item, idx)?) {
                continue 'outer;
            }
        }
        out.push(item);
    }
    Ok(Val::arr(out))
}

#[inline]
pub fn unique_by_apply<F>(recv: Val, mut eval: F) -> Result<Val, EvalError>
where
    F: FnMut(&Val) -> Result<Val, EvalError>,
{
    let items = recv
        .into_vec()
        .ok_or_else(|| EvalError("unique_by: expected array".into()))?;
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        let key = eval(&item)?;
        if seen.insert(crate::util::val_to_key(&key)) {
            out.push(item);
        }
    }
    Ok(Val::arr(out))
}

#[inline]
pub fn find_index_apply<F>(recv: Val, pred_count: usize, mut eval: F) -> Result<Val, EvalError>
where
    F: FnMut(&Val, usize) -> Result<Val, EvalError>,
{
    if pred_count == 0 {
        return Err(EvalError("find_index: requires a predicate".into()));
    }
    let items = recv
        .into_vec()
        .ok_or_else(|| EvalError("find_index: expected array".into()))?;
    'outer: for (idx, item) in items.iter().enumerate() {
        for pred_idx in 0..pred_count {
            if !is_truthy(&eval(item, pred_idx)?) {
                continue 'outer;
            }
        }
        return Ok(Val::Int(idx as i64));
    }
    Ok(Val::Null)
}

#[inline]
pub fn indices_where_apply<F>(recv: Val, pred_count: usize, mut eval: F) -> Result<Val, EvalError>
where
    F: FnMut(&Val, usize) -> Result<Val, EvalError>,
{
    if pred_count == 0 {
        return Err(EvalError("indices_where: requires a predicate".into()));
    }
    let items = recv
        .into_vec()
        .ok_or_else(|| EvalError("indices_where: expected array".into()))?;
    let mut out = Vec::new();
    'outer: for (idx, item) in items.iter().enumerate() {
        for pred_idx in 0..pred_count {
            if !is_truthy(&eval(item, pred_idx)?) {
                continue 'outer;
            }
        }
        out.push(idx as i64);
    }
    Ok(Val::int_vec(out))
}

#[inline]
pub fn extreme_by_apply<F>(recv: Val, want_max: bool, mut eval: F) -> Result<Val, EvalError>
where
    F: FnMut(&Val) -> Result<Val, EvalError>,
{
    let items = recv
        .into_vec()
        .ok_or_else(|| EvalError("max_by/min_by: expected array".into()))?;
    if items.is_empty() {
        return Ok(Val::Null);
    }
    let mut best_idx = 0usize;
    let mut best_key: Option<Val> = None;
    for (idx, item) in items.iter().enumerate() {
        let key = eval(item)?;
        let take = match &best_key {
            None => true,
            Some(best) => {
                let ord = cmp_vals(&key, best);
                if want_max {
                    ord == std::cmp::Ordering::Greater
                } else {
                    ord == std::cmp::Ordering::Less
                }
            }
        };
        if take {
            best_idx = idx;
            best_key = Some(key);
        }
    }
    Ok(items.into_iter().nth(best_idx).unwrap_or(Val::Null))
}

#[inline]
pub fn collect_apply(recv: &Val) -> Val {
    match recv {
        Val::Null => Val::arr(Vec::new()),
        Val::Arr(_) | Val::IntVec(_) | Val::FloatVec(_) | Val::StrVec(_) | Val::StrSliceVec(_) => {
            recv.clone()
        }
        other => Val::arr(vec![other.clone()]),
    }
}

#[inline]
pub fn zip_apply(recv: Val, other: Val) -> Result<Val, EvalError> {
    zip_arrays(recv, other, false, Val::Null)
}

#[inline]
pub fn zip_longest_apply(recv: Val, other: Val, fill: Val) -> Result<Val, EvalError> {
    zip_arrays(recv, other, true, fill)
}

#[inline]
pub fn global_zip_apply(arrs: &[Val]) -> Val {
    let len = arrs.iter().filter_map(|a| a.arr_len()).min().unwrap_or(0);
    Val::arr(
        (0..len)
            .map(|i| Val::arr(arrs.iter().map(|a| a.get_index(i as i64)).collect()))
            .collect(),
    )
}

#[inline]
pub fn global_zip_longest_apply(arrs: &[Val], fill: &Val) -> Val {
    let len = arrs.iter().filter_map(|a| a.arr_len()).max().unwrap_or(0);
    Val::arr(
        (0..len)
            .map(|i| {
                Val::arr(
                    arrs.iter()
                        .map(|a| {
                            if (i as usize) < a.arr_len().unwrap_or(0) {
                                a.get_index(i as i64)
                            } else {
                                fill.clone()
                            }
                        })
                        .collect(),
                )
            })
            .collect(),
    )
}

#[inline]
pub fn global_product_apply(arrs: &[Val]) -> Val {
    let arrays: Vec<Vec<Val>> = arrs
        .iter()
        .map(|v| v.clone().into_vec().unwrap_or_default())
        .collect();
    Val::arr(
        crate::util::cartesian(&arrays)
            .into_iter()
            .map(Val::arr)
            .collect(),
    )
}

#[inline]
pub fn range_apply(nums: &[i64]) -> Result<Val, EvalError> {
    if nums.is_empty() || nums.len() > 3 {
        return Err(EvalError(format!(
            "range: expected 1..3 args, got {}",
            nums.len()
        )));
    }
    let (from, upto, step) = match nums {
        [n] => (0, *n, 1i64),
        [f, u] => (*f, *u, 1i64),
        [f, u, s] => (*f, *u, *s),
        _ => unreachable!(),
    };
    if step == 0 {
        return Ok(Val::int_vec(Vec::new()));
    }
    let len_hint = if step > 0 && upto > from {
        (((upto - from) + step - 1) / step).max(0) as usize
    } else if step < 0 && upto < from {
        (((from - upto) + (-step) - 1) / (-step)).max(0) as usize
    } else {
        0
    };
    let mut out = Vec::with_capacity(len_hint);
    let mut i = from;
    if step > 0 {
        while i < upto {
            out.push(i);
            i += step;
        }
    } else {
        while i > upto {
            out.push(i);
            i += step;
        }
    }
    Ok(Val::int_vec(out))
}

#[inline]
pub fn equi_join_apply(
    recv: Val,
    other: Val,
    lhs_key: &str,
    rhs_key: &str,
) -> Result<Val, EvalError> {
    use std::collections::HashMap;

    let left = recv
        .into_vec()
        .ok_or_else(|| EvalError("equi_join: lhs not array".into()))?;
    let right = other
        .into_vec()
        .ok_or_else(|| EvalError("equi_join: rhs not array".into()))?;
    let mut idx: HashMap<String, Vec<Val>> = HashMap::new();
    for r in right {
        let key = match &r {
            Val::Obj(o) => o.get(rhs_key).map(crate::util::val_to_key),
            _ => None,
        };
        if let Some(k) = key {
            idx.entry(k).or_default().push(r);
        }
    }

    let mut out = Vec::new();
    for l in left {
        let key = match &l {
            Val::Obj(o) => o.get(lhs_key).map(crate::util::val_to_key),
            _ => None,
        };
        let Some(k) = key else {
            continue;
        };
        let Some(matches) = idx.get(&k) else {
            continue;
        };
        for r in matches {
            out.push(merge_pair(&l, r));
        }
    }
    Ok(Val::arr(out))
}

fn merge_pair(left: &Val, right: &Val) -> Val {
    match (left, right) {
        (Val::Obj(lo), Val::Obj(ro)) => {
            let mut out = (**lo).clone();
            for (k, v) in ro.iter() {
                out.insert(k.clone(), v.clone());
            }
            Val::obj(out)
        }
        _ => left.clone(),
    }
}

#[inline]
pub fn pivot_apply<F>(recv: Val, arg_count: usize, mut eval: F) -> Result<Val, EvalError>
where
    F: FnMut(&Val, usize) -> Result<Val, EvalError>,
{
    let items = recv
        .into_vec()
        .ok_or_else(|| EvalError("pivot: expected array".into()))?;

    #[inline]
    fn to_key(v: Val) -> Arc<str> {
        match v {
            Val::Str(s) => s,
            other => Arc::<str>::from(crate::util::val_to_key(&other)),
        }
    }

    if arg_count >= 3 {
        let mut map: IndexMap<Arc<str>, IndexMap<Arc<str>, Val>> = IndexMap::new();
        for item in &items {
            let row = to_key(eval(item, 0)?);
            let col = to_key(eval(item, 1)?);
            let value = eval(item, 2)?;
            map.entry(row).or_default().insert(col, value);
        }
        let out = map
            .into_iter()
            .map(|(k, inner)| (k, Val::obj(inner)))
            .collect();
        return Ok(Val::obj(out));
    }

    if arg_count < 2 {
        return Err(EvalError("pivot: requires key arg and value arg".into()));
    }

    let mut map = IndexMap::with_capacity(items.len());
    for item in &items {
        let key = to_key(eval(item, 0)?);
        let value = eval(item, 1)?;
        map.insert(key, value);
    }
    Ok(Val::obj(map))
}

fn walk_pre<F: FnMut(&Val)>(value: &Val, f: &mut F) {
    f(value);
    match value {
        Val::Arr(items) => {
            for child in items.iter() {
                walk_pre(child, f);
            }
        }
        Val::Obj(map) => {
            for (_, child) in map.iter() {
                walk_pre(child, f);
            }
        }
        _ => {}
    }
}

#[inline]
pub fn deep_find_apply<F>(recv: Val, pred_count: usize, mut eval: F) -> Result<Val, EvalError>
where
    F: FnMut(&Val, usize) -> Result<Val, EvalError>,
{
    if pred_count == 0 {
        return Err(EvalError("find: requires at least one predicate".into()));
    }
    let mut out = Vec::new();
    let mut err_cell: Option<EvalError> = None;
    walk_pre(&recv, &mut |node| {
        if err_cell.is_some() {
            return;
        }
        for idx in 0..pred_count {
            match eval(node, idx) {
                Ok(v) if is_truthy(&v) => {}
                Ok(_) => return,
                Err(e) => {
                    err_cell = Some(e);
                    return;
                }
            }
        }
        out.push(node.clone());
    });
    if let Some(e) = err_cell {
        Err(e)
    } else {
        Ok(Val::arr(out))
    }
}

#[inline]
pub fn deep_shape_apply(recv: Val, keys: &[Arc<str>]) -> Result<Val, EvalError> {
    if keys.is_empty() {
        return Err(EvalError("shape: empty pattern".into()));
    }
    let mut out = Vec::new();
    walk_pre(&recv, &mut |node| {
        if let Val::Obj(map) = node {
            if keys.iter().all(|k| map.contains_key(k.as_ref())) {
                out.push(node.clone());
            }
        }
    });
    Ok(Val::arr(out))
}

#[inline]
pub fn deep_like_apply(recv: Val, pats: &[(Arc<str>, Val)]) -> Result<Val, EvalError> {
    if pats.is_empty() {
        return Err(EvalError("like: empty pattern".into()));
    }
    let mut out = Vec::new();
    walk_pre(&recv, &mut |node| {
        if let Val::Obj(map) = node {
            let ok = pats.iter().all(|(key, want)| {
                map.get(key.as_ref())
                    .map(|got| crate::util::vals_eq(got, want))
                    .unwrap_or(false)
            });
            if ok {
                out.push(node.clone());
            }
        }
    });
    Ok(Val::arr(out))
}

pub fn walk_apply<F>(recv: Val, pre: bool, eval: &mut F) -> Result<Val, EvalError>
where
    F: FnMut(Val) -> Result<Val, EvalError>,
{
    let transformed = if pre { eval(recv)? } else { recv };
    let after_children = match transformed {
        Val::Arr(a) => {
            let items = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
            let mut out = Vec::with_capacity(items.len());
            for child in items {
                out.push(walk_apply(child, pre, eval)?);
            }
            Val::arr(out)
        }
        Val::IntVec(a) => {
            let mut out = Vec::with_capacity(a.len());
            for n in a.iter() {
                out.push(walk_apply(Val::Int(*n), pre, eval)?);
            }
            Val::arr(out)
        }
        Val::FloatVec(a) => {
            let mut out = Vec::with_capacity(a.len());
            for n in a.iter() {
                out.push(walk_apply(Val::Float(*n), pre, eval)?);
            }
            Val::arr(out)
        }
        Val::Obj(m) => {
            let items = Arc::try_unwrap(m).unwrap_or_else(|arc| (*arc).clone());
            let mut out = IndexMap::with_capacity(items.len());
            for (k, child) in items {
                out.insert(k, walk_apply(child, pre, eval)?);
            }
            Val::obj(out)
        }
        other => other,
    };
    if pre {
        Ok(after_children)
    } else {
        eval(after_children)
    }
}

#[inline]
pub fn rec_apply<F>(mut recv: Val, mut eval: F) -> Result<Val, EvalError>
where
    F: FnMut(Val) -> Result<Val, EvalError>,
{
    for _ in 0..10_000 {
        let next = eval(recv.clone())?;
        if crate::util::vals_eq(&recv, &next) {
            return Ok(next);
        }
        recv = next;
    }
    Err(EvalError(
        "rec: exceeded 10000 iterations without reaching fixpoint".into(),
    ))
}

pub fn trace_path_apply<F>(recv: Val, mut eval: F) -> Result<Val, EvalError>
where
    F: FnMut(&Val) -> Result<Val, EvalError>,
{
    fn walk<F>(value: &Val, path: String, eval: &mut F, out: &mut Vec<Val>) -> Result<(), EvalError>
    where
        F: FnMut(&Val) -> Result<Val, EvalError>,
    {
        if is_truthy(&eval(value)?) {
            let mut row = IndexMap::with_capacity(2);
            row.insert(Arc::from("path"), Val::Str(Arc::from(path.as_str())));
            row.insert(Arc::from("value"), value.clone());
            out.push(Val::obj(row));
        }
        match value {
            Val::Arr(items) => {
                for (idx, child) in items.iter().enumerate() {
                    walk(child, format!("{}[{}]", path, idx), eval, out)?;
                }
            }
            Val::IntVec(items) => {
                for (idx, n) in items.iter().enumerate() {
                    walk(&Val::Int(*n), format!("{}[{}]", path, idx), eval, out)?;
                }
            }
            Val::FloatVec(items) => {
                for (idx, n) in items.iter().enumerate() {
                    walk(&Val::Float(*n), format!("{}[{}]", path, idx), eval, out)?;
                }
            }
            Val::Obj(map) => {
                for (key, child) in map.iter() {
                    walk(child, format!("{}.{}", path, key), eval, out)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    let mut out = Vec::new();
    walk(&recv, String::from("$"), &mut eval, &mut out)?;
    Ok(Val::arr(out))
}

#[inline]
pub fn fanout_apply<F>(recv: &Val, count: usize, mut eval: F) -> Result<Val, EvalError>
where
    F: FnMut(&Val, usize) -> Result<Val, EvalError>,
{
    if count == 0 {
        return Err(EvalError("fanout: requires at least one expression".into()));
    }
    let mut out = Vec::with_capacity(count);
    for idx in 0..count {
        out.push(eval(recv, idx)?);
    }
    Ok(Val::arr(out))
}

#[inline]
pub fn zip_shape_apply<F>(recv: &Val, names: &[Arc<str>], mut eval: F) -> Result<Val, EvalError>
where
    F: FnMut(&Val, usize) -> Result<Val, EvalError>,
{
    if names.is_empty() {
        return Err(EvalError("zip_shape: requires at least one field".into()));
    }
    let mut out = IndexMap::with_capacity(names.len());
    for (idx, name) in names.iter().enumerate() {
        out.insert(name.clone(), eval(recv, idx)?);
    }
    Ok(Val::obj(out))
}

#[inline]
pub fn group_shape_apply<F>(recv: Val, mut eval: F) -> Result<Val, EvalError>
where
    F: FnMut(Val, usize) -> Result<Val, EvalError>,
{
    let items = recv
        .into_vec()
        .ok_or_else(|| EvalError("group_shape: expected array".into()))?;
    let mut buckets: IndexMap<Arc<str>, Vec<Val>> = IndexMap::with_capacity(items.len());
    for item in items {
        let key = match eval(item.clone(), 0)? {
            Val::Str(s) => s,
            other => Arc::<str>::from(crate::util::val_to_key(&other)),
        };
        buckets.entry(key).or_default().push(item);
    }
    let mut out = IndexMap::with_capacity(buckets.len());
    for (key, group) in buckets {
        out.insert(key, eval(Val::arr(group), 1)?);
    }
    Ok(Val::obj(out))
}

// ── take_while / drop_while / any / all ─────────────────────────────

/// Per-row take_while decision — `Ok(true)` continue, `Ok(false)`
/// stop the stream (caller breaks the outer loop).
#[inline]
pub fn take_while_one<F>(item: &Val, eval: F) -> Result<bool, EvalError>
where
    F: FnMut(&Val) -> Result<Val, EvalError>,
{
    filter_one(item, eval)
}

/// Per-row any — `Ok(true)` short-circuits the loop. Caller breaks
/// on first true.
#[inline]
pub fn any_one<F>(item: &Val, eval: F) -> Result<bool, EvalError>
where
    F: FnMut(&Val) -> Result<Val, EvalError>,
{
    filter_one(item, eval)
}

/// Per-row all — `Ok(false)` short-circuits the loop. Caller breaks
/// on first false.
#[inline]
pub fn all_one<F>(item: &Val, eval: F) -> Result<bool, EvalError>
where
    F: FnMut(&Val) -> Result<Val, EvalError>,
{
    filter_one(item, eval)
}

/// Buffered take_while — keeps prefix while predicate holds, then
/// stops.
#[inline]
pub fn take_while_apply<F>(items: Vec<Val>, mut eval: F) -> Result<Vec<Val>, EvalError>
where
    F: FnMut(&Val) -> Result<Val, EvalError>,
{
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        if !take_while_one(&item, &mut eval)? {
            break;
        }
        out.push(item);
    }
    Ok(out)
}

/// Buffered drop_while — drops prefix while predicate holds,
/// keeps the rest unconditionally.
#[inline]
pub fn drop_while_apply<F>(items: Vec<Val>, mut eval: F) -> Result<Vec<Val>, EvalError>
where
    F: FnMut(&Val) -> Result<Val, EvalError>,
{
    let mut dropping = true;
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        if dropping {
            if filter_one(&item, &mut eval)? {
                continue;
            }
            dropping = false;
        }
        out.push(item);
    }
    Ok(out)
}

// ── partition / group_by / count_by / index_by ──────────────────────

/// Buffered partition — splits items by predicate into (truthy, falsy).
#[inline]
pub fn partition_apply<F>(items: Vec<Val>, mut eval: F) -> Result<(Vec<Val>, Vec<Val>), EvalError>
where
    F: FnMut(&Val) -> Result<Val, EvalError>,
{
    let mut yes = Vec::with_capacity(items.len());
    let mut no = Vec::with_capacity(items.len());
    for item in items {
        if filter_one(&item, &mut eval)? {
            yes.push(item);
        } else {
            no.push(item);
        }
    }
    Ok((yes, no))
}

/// Buffered group_by — bucket items by key produced by `eval(item)`.
/// Key is val_to_key'd to `Arc<str>` for IndexMap insertion.
#[inline]
pub fn group_by_apply<F>(
    items: Vec<Val>,
    mut eval: F,
) -> Result<indexmap::IndexMap<std::sync::Arc<str>, Val>, EvalError>
where
    F: FnMut(&Val) -> Result<Val, EvalError>,
{
    use std::sync::Arc;
    let mut map: indexmap::IndexMap<Arc<str>, Val> = indexmap::IndexMap::new();
    for item in items {
        let k: Arc<str> = Arc::from(crate::util::val_to_key(&eval(&item)?).as_str());
        let bucket = map.entry(k).or_insert_with(|| Val::arr(Vec::new()));
        bucket.as_array_mut().unwrap().push(item);
    }
    Ok(map)
}

/// Buffered count_by — count items per `eval(item)` key.
#[inline]
pub fn count_by_apply<F>(
    items: Vec<Val>,
    mut eval: F,
) -> Result<indexmap::IndexMap<std::sync::Arc<str>, Val>, EvalError>
where
    F: FnMut(&Val) -> Result<Val, EvalError>,
{
    use std::sync::Arc;
    let mut map: indexmap::IndexMap<Arc<str>, Val> = indexmap::IndexMap::new();
    for item in items {
        let k: Arc<str> = Arc::from(crate::util::val_to_key(&eval(&item)?).as_str());
        let counter = map.entry(k).or_insert(Val::Int(0));
        if let Val::Int(n) = counter {
            *n += 1;
        }
    }
    Ok(map)
}

/// Buffered index_by — index items by `eval(item)` key (last write wins).
#[inline]
pub fn index_by_apply<F>(
    items: Vec<Val>,
    mut eval: F,
) -> Result<indexmap::IndexMap<std::sync::Arc<str>, Val>, EvalError>
where
    F: FnMut(&Val) -> Result<Val, EvalError>,
{
    use std::sync::Arc;
    let mut map: indexmap::IndexMap<Arc<str>, Val> = indexmap::IndexMap::new();
    for item in items {
        let k: Arc<str> = Arc::from(crate::util::val_to_key(&eval(&item)?).as_str());
        map.insert(k, item);
    }
    Ok(map)
}

// ── object-shaped (filter / transform on IndexMap) ──────────────────

/// Object form of filter — keep entries where predicate on `(k, v)`
/// holds. Caller supplies which side feeds the predicate (key for
/// `filter_keys`, value for `filter_values`) via the closure.
#[inline]
pub fn filter_object_apply<F>(
    map: indexmap::IndexMap<std::sync::Arc<str>, Val>,
    mut keep: F,
) -> Result<indexmap::IndexMap<std::sync::Arc<str>, Val>, EvalError>
where
    F: FnMut(&std::sync::Arc<str>, &Val) -> Result<bool, EvalError>,
{
    let mut out: indexmap::IndexMap<std::sync::Arc<str>, Val> = indexmap::IndexMap::new();
    for (k, v) in map {
        if keep(&k, &v)? {
            out.insert(k, v);
        }
    }
    Ok(out)
}

/// Object form of map — rename keys via `eval(key)`. Values
/// preserved.
#[inline]
pub fn transform_keys_apply<F>(
    map: indexmap::IndexMap<std::sync::Arc<str>, Val>,
    mut eval: F,
) -> Result<indexmap::IndexMap<std::sync::Arc<str>, Val>, EvalError>
where
    F: FnMut(&std::sync::Arc<str>) -> Result<Val, EvalError>,
{
    use std::sync::Arc;
    let mut out: indexmap::IndexMap<Arc<str>, Val> = indexmap::IndexMap::with_capacity(map.len());
    for (k, v) in map {
        let new_key: Arc<str> = Arc::from(crate::util::val_to_key(&eval(&k)?).as_str());
        out.insert(new_key, v);
    }
    Ok(out)
}

// ── Pure 1:1 kernels (no closure) ───────────────────────────────────
//
// `&Val → Val` shape — these are the canonical bodies for built-ins
// that lifted natively to Pipeline IR `Stage::*` enum variants per
// `lift_native_pattern.md`. Called from BOTH the Pipeline runtime arm
// AND the VM fallback adapter (for nested sub-program invocation via
// `Opcode::CallMethod`).

/// Canonical `.keys()` impl shared by `Stage::Keys` runtime arm and
/// the `.keys` VM adapter.  Non-object receivers yield an
/// empty array.
#[inline]
pub fn keys_apply(recv: &Val) -> Val {
    Val::arr(
        recv.as_object()
            .map(|m| m.keys().map(|k| Val::Str(k.clone())).collect())
            .unwrap_or_default(),
    )
}

/// Canonical `.values()` impl.
#[inline]
pub fn values_apply(recv: &Val) -> Val {
    Val::arr(
        recv.as_object()
            .map(|m| m.values().cloned().collect())
            .unwrap_or_default(),
    )
}

/// Canonical `.entries()` impl.  Each entry is `Val::arr([key, value])`.
#[inline]
pub fn entries_apply(recv: &Val) -> Val {
    Val::arr(
        recv.as_object()
            .map(|m| {
                m.iter()
                    .map(|(k, v)| Val::arr(vec![Val::Str(k.clone()), v.clone()]))
                    .collect()
            })
            .unwrap_or_default(),
    )
}

/// Canonical `.slice(start[, end])` impl shared by `Stage::Slice`
/// runtime arms and the `.slice` builtin dispatch shim.
/// Handles `Val::Str` + `Val::StrSlice` receivers; ASCII fast path
/// returns zero-alloc `Val::StrSlice` via `StrRef`; Unicode walks
/// `char_indices`. Negative indexes count from end (Python slice
/// semantics); `end=None` means "to end". Non-string receiver
/// returns the value unchanged.
pub fn slice_apply(recv: Val, start: i64, end: Option<i64>) -> Val {
    let (parent, base_off, view_len): (Arc<str>, usize, usize) = match recv {
        Val::Str(s) => {
            let l = s.len();
            (s, 0, l)
        }
        Val::StrSlice(r) => {
            let parent = r.to_arc();
            let plen = parent.len();
            (parent, 0, plen)
        }
        other => return other,
    };
    let view = &parent[base_off..base_off + view_len];
    let blen = view.len();
    if view.is_ascii() {
        let start_u = if start < 0 {
            blen.saturating_sub((-start) as usize)
        } else {
            (start as usize).min(blen)
        };
        let end_u = match end {
            Some(e) if e < 0 => blen.saturating_sub((-e) as usize),
            Some(e) => (e as usize).min(blen),
            None => blen,
        };
        let start_u = start_u.min(end_u);
        if start_u == 0 && end_u == blen {
            return Val::Str(parent);
        }
        return Val::StrSlice(crate::strref::StrRef::slice(
            parent,
            base_off + start_u,
            base_off + end_u,
        ));
    }
    let chars: Vec<(usize, char)> = view.char_indices().collect();
    let n = chars.len() as i64;
    let resolve = |i: i64| -> usize {
        let r = if i < 0 { n + i } else { i };
        r.clamp(0, n) as usize
    };
    let s_idx = resolve(start);
    let e_idx = match end {
        Some(e) => resolve(e),
        None => n as usize,
    };
    let s_idx = s_idx.min(e_idx);
    let s_b = chars.get(s_idx).map(|c| c.0).unwrap_or(view.len());
    let e_b = chars.get(e_idx).map(|c| c.0).unwrap_or(view.len());
    if s_b == 0 && e_b == view.len() {
        return Val::Str(parent);
    }
    Val::StrSlice(crate::strref::StrRef::slice(
        parent,
        base_off + s_b,
        base_off + e_b,
    ))
}

/// Canonical `.split(sep)` impl shared by `Stage::Split` runtime arms
/// and the `.split` builtin dispatch shim. Returns the split segments
/// as fresh `Val::Str` allocations wrapped in `Val::Arr`. `None` on
/// non-string receiver.
#[inline]
pub fn split_apply(recv: &Val, sep: &str) -> Option<Val> {
    let s: &str = match recv {
        Val::Str(s) => s.as_ref(),
        Val::StrSlice(r) => r.as_str(),
        _ => return None,
    };
    Some(Val::arr(
        s.split(sep)
            .map(|p| Val::Str(Arc::<str>::from(p)))
            .collect(),
    ))
}

/// Canonical `.chunk(n)` partition into chunks of size `n` (last may
/// be shorter). Each emitted Val is `Val::arr` of up to `n` source
/// elements.
#[inline]
pub fn chunk_apply(items: &[Val], n: usize) -> Vec<Val> {
    let n = n.max(1);
    items.chunks(n).map(|c| Val::arr(c.to_vec())).collect()
}

/// Canonical `.window(n)` sliding window of size `n` over the source
/// stream. Emits `len.saturating_sub(n) + 1` overlapping windows;
/// empty when `n > len`.
#[inline]
pub fn window_apply(items: &[Val], n: usize) -> Vec<Val> {
    let n = n.max(1);
    items.windows(n).map(|w| Val::arr(w.to_vec())).collect()
}

/// Canonical `.replace(needle, repl)` (all=false) and `.replace_all`
/// (all=true). Shared by `Stage::Replace` runtime arms and the
/// dispatch shim. Returns receiver unchanged when needle is absent
/// (no alloc fast-path). `None` on non-string receiver.
#[inline]
pub fn replace_apply(recv: Val, needle: &str, replacement: &str, all: bool) -> Option<Val> {
    let s: Arc<str> = match recv {
        Val::Str(s) => s,
        Val::StrSlice(r) => r.to_arc(),
        _ => return None,
    };
    if !s.contains(needle) {
        return Some(Val::Str(s));
    }
    let out = if all {
        s.replace(needle, replacement)
    } else {
        s.replacen(needle, replacement, 1)
    };
    Some(Val::Str(Arc::<str>::from(out)))
}

// ── String 1:1 transforms (Phase D batch 1) ─────────────────────────
//
// Per `lift_native_pattern.md` — each builtin's body lives here as a
// `pub fn *_apply(recv: &Val) -> Option<Val>` free fn. Pipeline IR
// runtime arm calls these directly via `lifted_apply` dispatch.
//
// Helper: lift any `&str → String` transform into a `&Val → Option<Val>`
// kernel that filters non-string receivers. Accepts both owned strings
// and tape-backed `StrSlice` views so simd-json materialisation can
// share string storage without changing builtin semantics.

#[inline]
fn map_str_owned(recv: &Val, f: impl FnOnce(&str) -> String) -> Option<Val> {
    let s = recv.as_str_ref()?;
    Some(Val::Str(Arc::<str>::from(f(s).as_str())))
}

/// `.upper()` — ASCII fast path; full Unicode fallback.
#[inline]
pub fn upper_apply(recv: &Val) -> Option<Val> {
    map_str_owned(recv, |s| {
        if s.is_ascii() {
            let mut buf = s.to_owned();
            buf.make_ascii_uppercase();
            buf
        } else {
            s.to_uppercase()
        }
    })
}

/// `.lower()` — ASCII fast path; full Unicode fallback.
#[inline]
pub fn lower_apply(recv: &Val) -> Option<Val> {
    map_str_owned(recv, |s| {
        if s.is_ascii() {
            let mut buf = s.to_owned();
            buf.make_ascii_lowercase();
            buf
        } else {
            s.to_lowercase()
        }
    })
}

/// `.trim()` — strip whitespace from both ends.
#[inline]
pub fn trim_apply(recv: &Val) -> Option<Val> {
    map_str_owned(recv, |s| s.trim().to_owned())
}

/// `.trim_left()` / `.trim_start()` — strip leading whitespace.
#[inline]
pub fn trim_left_apply(recv: &Val) -> Option<Val> {
    map_str_owned(recv, |s| s.trim_start().to_owned())
}

/// `.trim_right()` / `.trim_end()` — strip trailing whitespace.
#[inline]
pub fn trim_right_apply(recv: &Val) -> Option<Val> {
    map_str_owned(recv, |s| s.trim_end().to_owned())
}

/// `.capitalize()` — uppercase first char, lowercase rest (per Unicode).
#[inline]
pub fn capitalize_apply(recv: &Val) -> Option<Val> {
    map_str_owned(recv, |s| {
        let mut out = String::with_capacity(s.len());
        let mut chars = s.chars();
        if let Some(first) = chars.next() {
            for c in first.to_uppercase() {
                out.push(c);
            }
            out.push_str(&chars.as_str().to_lowercase());
        }
        out
    })
}

/// `.title_case()` — uppercase first char of each word, lowercase rest.
#[inline]
pub fn title_case_apply(recv: &Val) -> Option<Val> {
    map_str_owned(recv, |s| {
        let mut out = String::with_capacity(s.len());
        let mut at_start = true;
        for c in s.chars() {
            if c.is_whitespace() {
                out.push(c);
                at_start = true;
            } else if at_start {
                for u in c.to_uppercase() {
                    out.push(u);
                }
                at_start = false;
            } else {
                for l in c.to_lowercase() {
                    out.push(l);
                }
            }
        }
        out
    })
}

/// `.html_escape()` — replace HTML special chars with entities.
#[inline]
pub fn html_escape_apply(recv: &Val) -> Option<Val> {
    map_str_owned(recv, |s| {
        let mut out = String::with_capacity(s.len());
        for c in s.chars() {
            match c {
                '<' => out.push_str("&lt;"),
                '>' => out.push_str("&gt;"),
                '&' => out.push_str("&amp;"),
                '"' => out.push_str("&quot;"),
                '\'' => out.push_str("&#39;"),
                _ => out.push(c),
            }
        }
        out
    })
}

/// `.html_unescape()` — reverse the 5 entity replacements.
#[inline]
pub fn html_unescape_apply(recv: &Val) -> Option<Val> {
    map_str_owned(recv, |s| {
        s.replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", "\"")
            .replace("&#39;", "'")
    })
}

/// `.url_encode()` — percent-encode per RFC 3986 unreserved set.
#[inline]
pub fn url_encode_apply(recv: &Val) -> Option<Val> {
    map_str_owned(recv, |s| {
        let mut out = String::with_capacity(s.len());
        for b in s.as_bytes() {
            let b = *b;
            match b {
                b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                    out.push(b as char)
                }
                _ => {
                    use std::fmt::Write;
                    let _ = write!(out, "%{:02X}", b);
                }
            }
        }
        out
    })
}

/// `.url_decode()` — percent-decode + plus-to-space.
#[inline]
pub fn url_decode_apply(recv: &Val) -> Option<Val> {
    map_str_owned(recv, |s| {
        let bytes = s.as_bytes();
        let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'%' && i + 2 < bytes.len() {
                let h1 = char::from(bytes[i + 1]).to_digit(16);
                let h2 = char::from(bytes[i + 2]).to_digit(16);
                if let (Some(h1), Some(h2)) = (h1, h2) {
                    out.push((h1 * 16 + h2) as u8);
                    i += 3;
                    continue;
                }
            } else if bytes[i] == b'+' {
                out.push(b' ');
                i += 1;
                continue;
            }
            out.push(bytes[i]);
            i += 1;
        }
        String::from_utf8_lossy(&out).into_owned()
    })
}

/// `.to_base64()` — RFC 4648 base64 encoding.
#[inline]
pub fn to_base64_apply(recv: &Val) -> Option<Val> {
    map_str_owned(recv, |s| {
        crate::builtin_helpers::base64_encode(s.as_bytes())
    })
}

/// `.dedent()` — strip common leading whitespace from all non-empty lines.
#[inline]
pub fn dedent_apply(recv: &Val) -> Option<Val> {
    map_str_owned(recv, |s| {
        let min_indent = s
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.len() - l.trim_start().len())
            .min()
            .unwrap_or(0);
        s.lines()
            .map(|l| {
                if l.len() >= min_indent {
                    &l[min_indent..]
                } else {
                    l
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    })
}

/// `.snake_case()` — lower-case words joined by `_`.
#[inline]
pub fn snake_case_apply(recv: &Val) -> Option<Val> {
    map_str_owned(recv, |s| {
        crate::builtin_helpers::split_words_lower(s).join("_")
    })
}

/// `.kebab_case()` — lower-case words joined by `-`.
#[inline]
pub fn kebab_case_apply(recv: &Val) -> Option<Val> {
    map_str_owned(recv, |s| {
        crate::builtin_helpers::split_words_lower(s).join("-")
    })
}

/// `.camel_case()` — first word lower, rest capitalised, no separator.
#[inline]
pub fn camel_case_apply(recv: &Val) -> Option<Val> {
    map_str_owned(recv, |s| {
        let parts = crate::builtin_helpers::split_words_lower(s);
        let mut out = String::with_capacity(s.len());
        for (i, p) in parts.iter().enumerate() {
            if i == 0 {
                out.push_str(p);
            } else {
                crate::builtin_helpers::upper_first_into(p, &mut out);
            }
        }
        out
    })
}

/// `.pascal_case()` — every word capitalised, no separator.
#[inline]
pub fn pascal_case_apply(recv: &Val) -> Option<Val> {
    map_str_owned(recv, |s| {
        let parts = crate::builtin_helpers::split_words_lower(s);
        let mut out = String::with_capacity(s.len());
        for p in parts.iter() {
            crate::builtin_helpers::upper_first_into(p, &mut out);
        }
        out
    })
}

/// `.reverse()` on a string — Unicode-aware char reverse.
#[inline]
pub fn reverse_str_apply(recv: &Val) -> Option<Val> {
    map_str_owned(recv, |s| s.chars().rev().collect::<String>())
}

/// Helper: lift any `&str → Val` transform into a kernel that
/// filters non-string receivers (matches prior `lifted_str_to_val!`
/// macro semantics).
#[inline]
fn map_str_val(recv: &Val, f: impl FnOnce(&str) -> Val) -> Option<Val> {
    Some(f(recv.as_str_ref()?))
}

// ── Phase D batch 2: string → Val transforms (was lifted_str_to_val!) ──

/// `.lines()` — split into Val::Arr of Val::Str by newline.
#[inline]
pub fn lines_apply(recv: &Val) -> Option<Val> {
    map_str_val(recv, |s| {
        Val::arr(s.lines().map(|l| Val::Str(Arc::from(l))).collect())
    })
}

/// `.words()` — whitespace split into Val::Arr of Val::Str.
#[inline]
pub fn words_apply(recv: &Val) -> Option<Val> {
    map_str_val(recv, |s| {
        Val::arr(
            s.split_whitespace()
                .map(|w| Val::Str(Arc::from(w)))
                .collect(),
        )
    })
}

/// `.chars()` — codepoint split into Val::Arr of Val::Str.
#[inline]
pub fn chars_apply(recv: &Val) -> Option<Val> {
    map_str_val(recv, |s| {
        Val::arr(
            s.chars()
                .map(|c| Val::Str(Arc::from(c.to_string())))
                .collect(),
        )
    })
}

/// `.chars_of()` — Str → Arr<Str> (one Str per Unicode char).
#[inline]
pub fn chars_of_apply(recv: &Val) -> Option<Val> {
    map_str_val(recv, |s| {
        let mut out: Vec<Val> = Vec::new();
        let mut tmp = [0u8; 4];
        for c in s.chars() {
            let utf8 = c.encode_utf8(&mut tmp);
            out.push(Val::Str(Arc::from(utf8.as_ref())));
        }
        Val::arr(out)
    })
}

/// `.bytes()` — Str → IntVec of byte values.
#[inline]
pub fn bytes_of_apply(recv: &Val) -> Option<Val> {
    map_str_val(recv, |s| {
        let v: Vec<i64> = s.as_bytes().iter().map(|&b| b as i64).collect();
        Val::int_vec(v)
    })
}

/// `.byte_len()` — Str → Int byte count.
#[inline]
pub fn byte_len_apply(recv: &Val) -> Option<Val> {
    map_str_val(recv, |s| Val::Int(s.len() as i64))
}

/// `.is_blank()` — all whitespace.
#[inline]
pub fn is_blank_apply(recv: &Val) -> Option<Val> {
    map_str_val(recv, |s| Val::Bool(s.chars().all(|c| c.is_whitespace())))
}

/// `.is_numeric()` — all ASCII digits, non-empty.
#[inline]
pub fn is_numeric_apply(recv: &Val) -> Option<Val> {
    map_str_val(recv, |s| {
        Val::Bool(!s.is_empty() && s.chars().all(|c| c.is_ascii_digit()))
    })
}

/// `.is_alpha()` — all alphabetic, non-empty.
#[inline]
pub fn is_alpha_apply(recv: &Val) -> Option<Val> {
    map_str_val(recv, |s| {
        Val::Bool(!s.is_empty() && s.chars().all(|c| c.is_alphabetic()))
    })
}

/// `.is_ascii()` — all ASCII bytes.
#[inline]
pub fn is_ascii_apply(recv: &Val) -> Option<Val> {
    map_str_val(recv, |s| Val::Bool(s.is_ascii()))
}

/// `.ceil()` — numeric ceiling as Int.
#[inline]
pub fn ceil_apply(recv: &Val) -> Option<Val> {
    match recv {
        Val::Int(n) => Some(Val::Int(*n)),
        Val::Float(f) => Some(Val::Int(f.ceil() as i64)),
        _ => None,
    }
}

#[inline]
pub fn try_ceil_apply(recv: &Val) -> Result<Option<Val>, EvalError> {
    ceil_apply(recv)
        .map(Some)
        .ok_or_else(|| EvalError("ceil: expected number".into()))
}

/// `.floor()` — numeric floor as Int.
#[inline]
pub fn floor_apply(recv: &Val) -> Option<Val> {
    match recv {
        Val::Int(n) => Some(Val::Int(*n)),
        Val::Float(f) => Some(Val::Int(f.floor() as i64)),
        _ => None,
    }
}

#[inline]
pub fn try_floor_apply(recv: &Val) -> Result<Option<Val>, EvalError> {
    floor_apply(recv)
        .map(Some)
        .ok_or_else(|| EvalError("floor: expected number".into()))
}

/// `.round()` — numeric round as Int.
#[inline]
pub fn round_apply(recv: &Val) -> Option<Val> {
    match recv {
        Val::Int(n) => Some(Val::Int(*n)),
        Val::Float(f) => Some(Val::Int(f.round() as i64)),
        _ => None,
    }
}

#[inline]
pub fn try_round_apply(recv: &Val) -> Result<Option<Val>, EvalError> {
    round_apply(recv)
        .map(Some)
        .ok_or_else(|| EvalError("round: expected number".into()))
}

/// `.abs()` — numeric absolute value.
#[inline]
pub fn abs_apply(recv: &Val) -> Option<Val> {
    match recv {
        Val::Int(n) => Some(Val::Int(n.wrapping_abs())),
        Val::Float(f) => Some(Val::Float(f.abs())),
        _ => None,
    }
}

#[inline]
pub fn try_abs_apply(recv: &Val) -> Result<Option<Val>, EvalError> {
    abs_apply(recv)
        .map(Some)
        .ok_or_else(|| EvalError("abs: expected number".into()))
}

/// `.to_number()` — parse i64 or f64; null on parse failure.
#[inline]
pub fn to_number_apply(recv: &Val) -> Option<Val> {
    map_str_val(recv, |s| {
        if let Ok(i) = s.parse::<i64>() {
            return Val::Int(i);
        }
        if let Ok(f) = s.parse::<f64>() {
            return Val::Float(f);
        }
        Val::Null
    })
}

/// `.to_bool()` — recognise "true"/"false" (Stage form). Errs on
/// non-recognised in dispatch fn; here we yield Null.
#[inline]
pub fn to_bool_apply(recv: &Val) -> Option<Val> {
    map_str_val(recv, |s| match s {
        "true" => Val::Bool(true),
        "false" => Val::Bool(false),
        _ => Val::Null,
    })
}

/// `.parse_int()` — trim + parse i64; Null on failure.
#[inline]
pub fn parse_int_apply(recv: &Val) -> Option<Val> {
    map_str_val(recv, |s| {
        s.trim().parse::<i64>().map(Val::Int).unwrap_or(Val::Null)
    })
}

/// `.parse_float()` — trim + parse f64; Null on failure.
#[inline]
pub fn parse_float_apply(recv: &Val) -> Option<Val> {
    map_str_val(recv, |s| {
        s.trim().parse::<f64>().map(Val::Float).unwrap_or(Val::Null)
    })
}

/// `.parse_bool()` — recognise extended forms (true/yes/1/on, false/no/0/off);
/// Null on unrecognised.
#[inline]
pub fn parse_bool_apply(recv: &Val) -> Option<Val> {
    map_str_val(recv, |s| match s.trim().to_ascii_lowercase().as_str() {
        "true" | "yes" | "1" | "on" => Val::Bool(true),
        "false" | "no" | "0" | "off" => Val::Bool(false),
        _ => Val::Null,
    })
}

/// `.from_base64()` — decode; lossy UTF-8 conversion. Null on decode failure.
#[inline]
pub fn from_base64_apply(recv: &Val) -> Option<Val> {
    map_str_val(recv, |s| match crate::builtin_helpers::base64_decode(s) {
        Ok(bytes) => Val::Str(Arc::from(String::from_utf8_lossy(&bytes).as_ref())),
        Err(_) => Val::Null,
    })
}

// ── Phase D batch 3: string args (one-arg / two-arg search/transform) ──

/// `.starts_with(prefix)` — returns Val::Bool.
#[inline]
pub fn starts_with_apply(recv: &Val, prefix: &str) -> Option<Val> {
    Some(Val::Bool(recv.as_str_ref()?.starts_with(prefix)))
}

/// `.ends_with(suffix)` — returns Val::Bool.
#[inline]
pub fn ends_with_apply(recv: &Val, suffix: &str) -> Option<Val> {
    Some(Val::Bool(recv.as_str_ref()?.ends_with(suffix)))
}

/// `.contains(needle)` / `.str_matches(needle)` — returns Val::Bool.
#[inline]
pub fn contains_apply(recv: &Val, needle: &str) -> Option<Val> {
    Some(Val::Bool(recv.as_str_ref()?.contains(needle)))
}

/// `.repeat(n)` — repeat string n times.
#[inline]
pub fn repeat_apply(recv: &Val, n: usize) -> Option<Val> {
    Some(Val::Str(Arc::from(recv.as_str_ref()?.repeat(n))))
}

/// `.strip_prefix(prefix)` — strip if present, else return original.
#[inline]
pub fn strip_prefix_apply(recv: &Val, prefix: &str) -> Option<Val> {
    let s = recv.as_str_ref()?;
    Some(match s.strip_prefix(prefix) {
        Some(stripped) => Val::Str(Arc::<str>::from(stripped)),
        None => recv.clone(),
    })
}

/// `.strip_suffix(suffix)` — strip if present, else return original.
#[inline]
pub fn strip_suffix_apply(recv: &Val, suffix: &str) -> Option<Val> {
    let s = recv.as_str_ref()?;
    Some(match s.strip_suffix(suffix) {
        Some(stripped) => Val::Str(Arc::<str>::from(stripped)),
        None => recv.clone(),
    })
}

/// `.pad_left(width, fill)` — left-pad to width with fill char.
#[inline]
pub fn pad_left_apply(recv: &Val, width: usize, fill: char) -> Option<Val> {
    let s = recv.as_str_ref()?;
    let n = s.chars().count();
    if n >= width {
        return Some(recv.clone());
    }
    let pad: String = std::iter::repeat(fill).take(width - n).collect();
    Some(Val::Str(Arc::from(pad + s)))
}

/// `.pad_right(width, fill)` — right-pad to width with fill char.
#[inline]
pub fn pad_right_apply(recv: &Val, width: usize, fill: char) -> Option<Val> {
    let s = recv.as_str_ref()?;
    let n = s.chars().count();
    if n >= width {
        return Some(recv.clone());
    }
    let pad: String = std::iter::repeat(fill).take(width - n).collect();
    Some(Val::Str(Arc::from(s.to_string() + &pad)))
}

/// `.center(width, fill)` — center-pad to width with fill char.
#[inline]
pub fn center_apply(recv: &Val, width: usize, fill: char) -> Option<Val> {
    let s = recv.as_str_ref()?;
    let cur = s.chars().count();
    if cur >= width {
        return Some(recv.clone());
    }
    let total = width - cur;
    let left = total / 2;
    let right = total - left;
    let mut out = String::with_capacity(s.len() + total);
    for _ in 0..left {
        out.push(fill);
    }
    out.push_str(s);
    for _ in 0..right {
        out.push(fill);
    }
    Some(Val::Str(Arc::from(out)))
}

/// `.indent(n)` — prepend n spaces to each line.
#[inline]
pub fn indent_apply(recv: &Val, n: usize) -> Option<Val> {
    let s = recv.as_str_ref()?;
    let prefix: String = std::iter::repeat(' ').take(n).collect();
    let out = s
        .lines()
        .map(|l| format!("{}{}", prefix, l))
        .collect::<Vec<_>>()
        .join("\n");
    Some(Val::Str(Arc::from(out)))
}

/// `.index_of(needle)` — char-position of first match; -1 on miss.
#[inline]
pub fn index_of_apply(recv: &Val, needle: &str) -> Option<Val> {
    let s = recv.as_str_ref()?;
    Some(match s.find(needle) {
        Some(i) => Val::Int(s[..i].chars().count() as i64),
        None => Val::Int(-1),
    })
}

/// `.last_index_of(needle)` — char-position of last match; -1 on miss.
#[inline]
pub fn last_index_of_apply(recv: &Val, needle: &str) -> Option<Val> {
    let s = recv.as_str_ref()?;
    Some(match s.rfind(needle) {
        Some(i) => Val::Int(s[..i].chars().count() as i64),
        None => Val::Int(-1),
    })
}

/// `.scan(pat)` — collect all occurrences of `pat` in `s`.
#[inline]
pub fn scan_apply(recv: &Val, pat: &str) -> Option<Val> {
    let s = recv.as_str_ref()?;
    let mut out: Vec<Val> = Vec::new();
    if !pat.is_empty() {
        let mut start = 0usize;
        while let Some(pos) = s[start..].find(pat) {
            out.push(Val::Str(Arc::from(pat)));
            start += pos + pat.len();
        }
    }
    Some(Val::arr(out))
}

// ── Phase D batch 4: array family ───────────────────────────────────

/// Numeric aggregate over a receiver without projection. Typed vectors stay on
/// slice loops; mixed arrays skip non-numeric values.
#[inline]
pub fn numeric_aggregate_apply(recv: &Val, method: BuiltinMethod) -> Val {
    match recv {
        Val::IntVec(a) => return numeric_aggregate_i64(a, method),
        Val::FloatVec(a) => return numeric_aggregate_f64(a, method),
        Val::Arr(a) => numeric_aggregate_values(a, method),
        _ => Val::Null,
    }
}

/// Numeric aggregate with per-item projection. This path is used for
/// `.sum(field)` / `.avg(lambda ...)`; bare `.sum()` stays on the typed
/// no-projection kernel above.
#[inline]
pub fn numeric_aggregate_projected_apply<F>(
    recv: &Val,
    method: BuiltinMethod,
    mut eval: F,
) -> Result<Val, EvalError>
where
    F: FnMut(&Val) -> Result<Val, EvalError>,
{
    let items = recv
        .as_vals()
        .ok_or_else(|| EvalError("expected array for numeric aggregate".into()))?;

    let mut vals = Vec::with_capacity(items.len());
    for item in items.iter() {
        let v = eval(item)?;
        if v.is_number() {
            vals.push(v);
        }
    }
    Ok(numeric_aggregate_values(&vals, method))
}

#[inline]
fn numeric_aggregate_i64(a: &[i64], method: BuiltinMethod) -> Val {
    match method {
        BuiltinMethod::Sum => Val::Int(a.iter().fold(0i64, |acc, n| acc.wrapping_add(*n))),
        BuiltinMethod::Avg => {
            if a.is_empty() {
                Val::Null
            } else {
                let s = a.iter().fold(0i64, |acc, n| acc.wrapping_add(*n));
                Val::Float(s as f64 / a.len() as f64)
            }
        }
        BuiltinMethod::Min => a.iter().min().copied().map(Val::Int).unwrap_or(Val::Null),
        BuiltinMethod::Max => a.iter().max().copied().map(Val::Int).unwrap_or(Val::Null),
        _ => Val::Null,
    }
}

#[inline]
fn numeric_aggregate_f64(a: &[f64], method: BuiltinMethod) -> Val {
    match method {
        BuiltinMethod::Sum => Val::Float(a.iter().sum()),
        BuiltinMethod::Avg => {
            if a.is_empty() {
                Val::Null
            } else {
                Val::Float(a.iter().sum::<f64>() / a.len() as f64)
            }
        }
        BuiltinMethod::Min => a
            .iter()
            .copied()
            .reduce(f64::min)
            .map(Val::Float)
            .unwrap_or(Val::Null),
        BuiltinMethod::Max => a
            .iter()
            .copied()
            .reduce(f64::max)
            .map(Val::Float)
            .unwrap_or(Val::Null),
        _ => Val::Null,
    }
}

#[inline]
fn numeric_aggregate_values(a: &[Val], method: BuiltinMethod) -> Val {
    match method {
        BuiltinMethod::Sum => {
            let mut i_acc: i64 = 0;
            let mut f_acc: f64 = 0.0;
            let mut floated = false;
            for v in a {
                match v {
                    Val::Int(n) if !floated => i_acc = i_acc.wrapping_add(*n),
                    Val::Int(n) => f_acc += *n as f64,
                    Val::Float(f) if !floated => {
                        f_acc = i_acc as f64 + *f;
                        floated = true;
                    }
                    Val::Float(f) => f_acc += *f,
                    _ => {}
                }
            }
            if floated {
                Val::Float(f_acc)
            } else {
                Val::Int(i_acc)
            }
        }
        BuiltinMethod::Avg => {
            let mut sum = 0.0;
            let mut n = 0usize;
            for v in a {
                match v {
                    Val::Int(i) => {
                        sum += *i as f64;
                        n += 1;
                    }
                    Val::Float(f) => {
                        sum += *f;
                        n += 1;
                    }
                    _ => {}
                }
            }
            if n == 0 {
                Val::Null
            } else {
                Val::Float(sum / n as f64)
            }
        }
        BuiltinMethod::Min | BuiltinMethod::Max => {
            let want_max = method == BuiltinMethod::Max;
            let mut best: Option<Val> = None;
            let mut best_f = 0.0;
            for v in a {
                if !v.is_number() {
                    continue;
                }
                let vf = v.as_f64().unwrap_or(0.0);
                let replace = match best {
                    None => true,
                    Some(_) if want_max => vf > best_f,
                    Some(_) => vf < best_f,
                };
                if replace {
                    best_f = vf;
                    best = Some(v.clone());
                }
            }
            best.unwrap_or(Val::Null)
        }
        _ => Val::Null,
    }
}

/// `.len()` / `.count()` — element count for Arr / typed vecs / Obj /
/// Str (chars).
#[inline]
pub fn len_apply(recv: &Val) -> Option<Val> {
    let n = match recv {
        Val::Arr(a) => a.len(),
        Val::IntVec(a) => a.len(),
        Val::FloatVec(a) => a.len(),
        Val::StrVec(a) => a.len(),
        Val::StrSliceVec(a) => a.len(),
        Val::Obj(m) => m.len(),
        Val::Str(s) => s.chars().count(),
        Val::StrSlice(r) => r.as_str().chars().count(),
        _ => return None,
    };
    Some(Val::Int(n as i64))
}

/// `.compact()` — drop Null entries from Arr.
#[inline]
pub fn compact_apply(recv: &Val) -> Option<Val> {
    let items_cow = recv.as_vals()?;
    let kept: Vec<Val> = items_cow
        .iter()
        .filter(|v| !matches!(v, Val::Null))
        .cloned()
        .collect();
    Some(Val::arr(kept))
}

/// `.flatten(depth)` — recursive flatten up to depth levels.
#[inline]
pub fn flatten_depth_apply(recv: &Val, depth: usize) -> Option<Val> {
    if matches!(recv, Val::Arr(_)) {
        Some(crate::util::flatten_val(recv.clone(), depth))
    } else {
        None
    }
}

/// `.reverse()` — reverse Arr / IntVec / FloatVec / StrVec / Str.
#[inline]
pub fn reverse_any_apply(recv: &Val) -> Option<Val> {
    Some(match recv {
        Val::Arr(a) => {
            let mut v: Vec<Val> = a.as_ref().clone();
            v.reverse();
            Val::arr(v)
        }
        Val::IntVec(a) => {
            let mut v: Vec<i64> = a.as_ref().clone();
            v.reverse();
            Val::int_vec(v)
        }
        Val::FloatVec(a) => {
            let mut v: Vec<f64> = a.as_ref().clone();
            v.reverse();
            Val::float_vec(v)
        }
        Val::StrVec(a) => {
            let mut v: Vec<Arc<str>> = a.as_ref().clone();
            v.reverse();
            Val::str_vec(v)
        }
        Val::Str(s) => Val::Str(Arc::<str>::from(s.chars().rev().collect::<String>())),
        Val::StrSlice(r) => Val::Str(Arc::<str>::from(
            r.as_str().chars().rev().collect::<String>(),
        )),
        _ => return None,
    })
}

/// `.unique()` / `.distinct()` — dedup Arr by val_to_key.
#[inline]
pub fn unique_arr_apply(recv: &Val) -> Option<Val> {
    let items_cow = recv.as_vals()?;
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let kept: Vec<Val> = items_cow
        .iter()
        .filter(|v| seen.insert(crate::util::val_to_key(v)))
        .cloned()
        .collect();
    Some(Val::arr(kept))
}

fn numeric_options(recv: &Val) -> Option<Vec<Option<f64>>> {
    match recv {
        Val::IntVec(a) => Some(a.iter().map(|n| Some(*n as f64)).collect()),
        Val::FloatVec(a) => Some(a.iter().map(|f| Some(*f)).collect()),
        Val::Arr(a) => Some(
            a.iter()
                .map(|v| match v {
                    Val::Int(n) => Some(*n as f64),
                    Val::Float(f) => Some(*f),
                    _ => None,
                })
                .collect(),
        ),
        _ => None,
    }
}

fn numeric_options_to_val(out: Vec<Option<f64>>) -> Val {
    if out.iter().all(|v| v.is_some()) {
        Val::float_vec(out.into_iter().map(|v| v.unwrap()).collect())
    } else {
        Val::arr(
            out.into_iter()
                .map(|v| match v {
                    Some(f) => Val::Float(f),
                    None => Val::Null,
                })
                .collect(),
        )
    }
}

/// `.rolling_sum(n)` — rolling numeric sum.
#[inline]
pub fn rolling_sum_apply(recv: &Val, n: usize) -> Option<Val> {
    if n == 0 {
        return None;
    }
    let xs = numeric_options(recv)?;
    let mut out: Vec<Option<f64>> = Vec::with_capacity(xs.len());
    let mut sum: f64 = 0.0;
    for (i, v) in xs.iter().enumerate() {
        if let Some(x) = v {
            sum += x;
        }
        if i >= n {
            if let Some(old) = xs[i - n] {
                sum -= old;
            }
        }
        if i + 1 >= n {
            out.push(Some(sum));
        } else {
            out.push(None);
        }
    }
    Some(numeric_options_to_val(out))
}

/// `.rolling_avg(n)` — rolling numeric average over present values.
#[inline]
pub fn rolling_avg_apply(recv: &Val, n: usize) -> Option<Val> {
    if n == 0 {
        return None;
    }
    let xs = numeric_options(recv)?;
    let mut out: Vec<Option<f64>> = Vec::with_capacity(xs.len());
    let mut sum: f64 = 0.0;
    let mut count: usize = 0;
    for (i, v) in xs.iter().enumerate() {
        if let Some(x) = v {
            sum += x;
            count += 1;
        }
        if i >= n {
            if let Some(old) = xs[i - n] {
                sum -= old;
                count -= 1;
            }
        }
        if i + 1 >= n && count > 0 {
            out.push(Some(sum / count as f64));
        } else {
            out.push(None);
        }
    }
    Some(numeric_options_to_val(out))
}

/// `.rolling_min(n)` — rolling numeric min.
#[inline]
pub fn rolling_min_apply(recv: &Val, n: usize) -> Option<Val> {
    if n == 0 {
        return None;
    }
    let xs = numeric_options(recv)?;
    let mut out: Vec<Option<f64>> = Vec::with_capacity(xs.len());
    for i in 0..xs.len() {
        if i + 1 < n {
            out.push(None);
            continue;
        }
        let lo = i + 1 - n;
        let m = xs[lo..=i]
            .iter()
            .filter_map(|v| *v)
            .fold(f64::INFINITY, |a, b| a.min(b));
        out.push(if m.is_finite() { Some(m) } else { None });
    }
    Some(numeric_options_to_val(out))
}

/// `.rolling_max(n)` — rolling numeric max.
#[inline]
pub fn rolling_max_apply(recv: &Val, n: usize) -> Option<Val> {
    if n == 0 {
        return None;
    }
    let xs = numeric_options(recv)?;
    let mut out: Vec<Option<f64>> = Vec::with_capacity(xs.len());
    for i in 0..xs.len() {
        if i + 1 < n {
            out.push(None);
            continue;
        }
        let lo = i + 1 - n;
        let m = xs[lo..=i]
            .iter()
            .filter_map(|v| *v)
            .fold(f64::NEG_INFINITY, |a, b| a.max(b));
        out.push(if m.is_finite() { Some(m) } else { None });
    }
    Some(numeric_options_to_val(out))
}

/// `.lag(n)` — shift numeric values backward, filling leading nulls.
#[inline]
pub fn lag_apply(recv: &Val, n: usize) -> Option<Val> {
    let xs = numeric_options(recv)?;
    let mut out: Vec<Option<f64>> = Vec::with_capacity(xs.len());
    for i in 0..xs.len() {
        out.push(if i >= n { xs[i - n] } else { None });
    }
    Some(numeric_options_to_val(out))
}

/// `.lead(n)` — shift numeric values forward, filling trailing nulls.
#[inline]
pub fn lead_apply(recv: &Val, n: usize) -> Option<Val> {
    let xs = numeric_options(recv)?;
    let mut out: Vec<Option<f64>> = Vec::with_capacity(xs.len());
    for i in 0..xs.len() {
        let j = i + n;
        out.push(if j < xs.len() { xs[j] } else { None });
    }
    Some(numeric_options_to_val(out))
}

/// `.diff_window()` — current numeric value minus previous numeric value.
#[inline]
pub fn diff_window_apply(recv: &Val) -> Option<Val> {
    let xs = numeric_options(recv)?;
    let mut out: Vec<Option<f64>> = Vec::with_capacity(xs.len());
    for i in 0..xs.len() {
        out.push(match (i.checked_sub(1).and_then(|j| xs[j]), xs[i]) {
            (Some(p), Some(c)) => Some(c - p),
            _ => None,
        });
    }
    Some(numeric_options_to_val(out))
}

/// `.pct_change()` — fractional change from previous numeric value.
#[inline]
pub fn pct_change_apply(recv: &Val) -> Option<Val> {
    let xs = numeric_options(recv)?;
    let mut out: Vec<Option<f64>> = Vec::with_capacity(xs.len());
    for i in 0..xs.len() {
        out.push(match (i.checked_sub(1).and_then(|j| xs[j]), xs[i]) {
            (Some(p), Some(c)) if p != 0.0 => Some((c - p) / p),
            _ => None,
        });
    }
    Some(numeric_options_to_val(out))
}

/// `.cummax()` — cumulative numeric maximum, carrying previous best over nulls.
#[inline]
pub fn cummax_apply(recv: &Val) -> Option<Val> {
    let xs = numeric_options(recv)?;
    let mut out: Vec<Option<f64>> = Vec::with_capacity(xs.len());
    let mut best: Option<f64> = None;
    for v in xs.iter() {
        match (*v, best) {
            (Some(x), Some(b)) => {
                best = Some(x.max(b));
                out.push(best);
            }
            (Some(x), None) => {
                best = Some(x);
                out.push(best);
            }
            (None, _) => out.push(best),
        }
    }
    Some(numeric_options_to_val(out))
}

/// `.cummin()` — cumulative numeric minimum, carrying previous best over nulls.
#[inline]
pub fn cummin_apply(recv: &Val) -> Option<Val> {
    let xs = numeric_options(recv)?;
    let mut out: Vec<Option<f64>> = Vec::with_capacity(xs.len());
    let mut best: Option<f64> = None;
    for v in xs.iter() {
        match (*v, best) {
            (Some(x), Some(b)) => {
                best = Some(x.min(b));
                out.push(best);
            }
            (Some(x), None) => {
                best = Some(x);
                out.push(best);
            }
            (None, _) => out.push(best),
        }
    }
    Some(numeric_options_to_val(out))
}

/// `.zscore()` — numeric standard score.
#[inline]
pub fn zscore_apply(recv: &Val) -> Option<Val> {
    let xs = numeric_options(recv)?;
    let nums: Vec<f64> = xs.iter().filter_map(|v| *v).collect();
    if nums.is_empty() {
        return Some(numeric_options_to_val(vec![None; xs.len()]));
    }
    let mean = nums.iter().sum::<f64>() / nums.len() as f64;
    let var = nums.iter().map(|y| (y - mean).powi(2)).sum::<f64>() / nums.len() as f64;
    let sd = var.sqrt();
    let mut out: Vec<Option<f64>> = Vec::with_capacity(xs.len());
    for v in xs.iter() {
        out.push(match v {
            Some(y) if sd > 0.0 => Some((y - mean) / sd),
            Some(_) => Some(0.0),
            None => None,
        });
    }
    Some(numeric_options_to_val(out))
}

/// `.first(n)` — n==1 returns scalar, else Arr of first n.
#[inline]
pub fn first_apply(recv: &Val, n: i64) -> Option<Val> {
    if let Val::Arr(a) = recv {
        Some(if n == 1 {
            a.first().cloned().unwrap_or(Val::Null)
        } else {
            Val::arr(a.iter().take(n.max(0) as usize).cloned().collect())
        })
    } else {
        Some(Val::Null)
    }
}

/// `.last(n)` — n==1 returns scalar, else Arr of last n.
#[inline]
pub fn last_apply(recv: &Val, n: i64) -> Option<Val> {
    if let Val::Arr(a) = recv {
        Some(if n == 1 {
            a.last().cloned().unwrap_or(Val::Null)
        } else {
            let s = a.len().saturating_sub(n.max(0) as usize);
            Val::arr(a[s..].to_vec())
        })
    } else {
        Some(Val::Null)
    }
}

/// `.nth(i)` — index lookup (negative = from end).
#[inline]
pub fn nth_any_apply(recv: &Val, i: i64) -> Option<Val> {
    Some(recv.get_index(i))
}

/// `.append(item)` — push item onto end. Coerces typed vecs.
#[inline]
pub fn append_apply(recv: &Val, item: &Val) -> Option<Val> {
    let mut v = recv.clone().into_vec()?;
    v.push(item.clone());
    Some(Val::arr(v))
}

/// `.prepend(item)` — insert item at front.
#[inline]
pub fn prepend_apply(recv: &Val, item: &Val) -> Option<Val> {
    let mut v = recv.clone().into_vec()?;
    v.insert(0, item.clone());
    Some(Val::arr(v))
}

/// `.remove(target)` — array-like receiver without values equal to target.
#[inline]
pub fn remove_value_apply(recv: &Val, target: &Val) -> Option<Val> {
    use crate::util::val_to_key;
    let items_cow = recv.as_vals()?;
    let key = val_to_key(target);
    let out: Vec<Val> = items_cow
        .iter()
        .filter(|v| val_to_key(v) != key)
        .cloned()
        .collect();
    Some(Val::arr(out))
}

/// `.enumerate()` — Arr → Arr<{index, value}>.
#[inline]
pub fn enumerate_apply(recv: &Val) -> Option<Val> {
    let items_cow = recv.as_vals()?;
    let out: Vec<Val> = items_cow
        .iter()
        .enumerate()
        .map(|(i, v)| crate::util::obj2("index", Val::Int(i as i64), "value", v.clone()))
        .collect();
    Some(Val::arr(out))
}

/// `.join(sep)` — array-like receiver to string.
#[inline]
pub fn join_apply(recv: &Val, sep: &str) -> Option<Val> {
    use crate::util::val_to_string;
    use std::fmt::Write as _;

    let items_cow = recv.as_vals()?;
    let items: &[Val] = items_cow.as_ref();
    if items.is_empty() {
        return Some(Val::Str(Arc::from("")));
    }
    if items.iter().all(|v| matches!(v, Val::Str(_))) {
        let total_len: usize = items
            .iter()
            .map(|v| if let Val::Str(s) = v { s.len() } else { 0 })
            .sum::<usize>()
            + sep.len() * (items.len() - 1);
        let mut out = String::with_capacity(total_len);
        for (idx, v) in items.iter().enumerate() {
            if idx > 0 {
                out.push_str(sep);
            }
            if let Val::Str(s) = v {
                out.push_str(s);
            }
        }
        return Some(Val::Str(Arc::from(out)));
    }

    let mut out = String::with_capacity(items.len() * 8 + sep.len() * items.len());
    for (idx, v) in items.iter().enumerate() {
        if idx > 0 {
            out.push_str(sep);
        }
        match v {
            Val::Str(s) => out.push_str(s),
            Val::Int(n) => {
                let _ = write!(out, "{}", n);
            }
            Val::Float(f) => {
                let _ = write!(out, "{}", f);
            }
            Val::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Val::Null => out.push_str("null"),
            other => out.push_str(&val_to_string(other)),
        }
    }
    Some(Val::Str(Arc::from(out)))
}

/// `.index(target)` — first index of target, else null.
#[inline]
pub fn index_value_apply(recv: &Val, target: &Val) -> Option<Val> {
    let items_cow = recv.as_vals()?;
    for (i, item) in items_cow.iter().enumerate() {
        if crate::util::vals_eq(item, target) {
            return Some(Val::Int(i as i64));
        }
    }
    Some(Val::Null)
}

/// `.indices_of(target)` — all indices of target.
#[inline]
pub fn indices_of_apply(recv: &Val, target: &Val) -> Option<Val> {
    let items_cow = recv.as_vals()?;
    let out: Vec<i64> = items_cow
        .iter()
        .enumerate()
        .filter(|(_, v)| crate::util::vals_eq(v, target))
        .map(|(i, _)| i as i64)
        .collect();
    Some(Val::int_vec(out))
}

/// `.explode(field)` — expand array-valued object field into one row per element.
#[inline]
pub fn explode_apply(recv: &Val, field: &str) -> Option<Val> {
    let items_cow = recv.as_vals()?;
    let items: &[Val] = items_cow.as_ref();
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        match item {
            Val::Obj(m) => {
                let sub = m.get(field).cloned();
                if sub.as_ref().map(|v| v.is_array()).unwrap_or(false) {
                    let elts = sub.unwrap().into_vec().unwrap();
                    for e in elts {
                        let mut row = (**m).clone();
                        row.insert(Arc::from(field), e);
                        out.push(Val::obj(row));
                    }
                } else {
                    out.push(item.clone());
                }
            }
            other => out.push(other.clone()),
        }
    }
    Some(Val::arr(out))
}

/// `.implode(field)` — inverse of explode, grouping rows by all non-field values.
#[inline]
pub fn implode_apply(recv: &Val, field: &str) -> Option<Val> {
    use crate::util::val_to_key;
    let items_cow = recv.as_vals()?;
    let items: &[Val] = items_cow.as_ref();
    let mut groups: indexmap::IndexMap<Arc<str>, (indexmap::IndexMap<Arc<str>, Val>, Vec<Val>)> =
        indexmap::IndexMap::new();
    for item in items {
        let m = match item {
            Val::Obj(m) => m,
            _ => return None,
        };
        let mut rest = (**m).clone();
        let val = rest.shift_remove(field).unwrap_or(Val::Null);
        let key_src: indexmap::IndexMap<Arc<str>, Val> = rest.clone();
        let key = Arc::<str>::from(val_to_key(&Val::obj(key_src)));
        groups
            .entry(key)
            .or_insert_with(|| (rest, Vec::new()))
            .1
            .push(val);
    }
    let mut out = Vec::with_capacity(groups.len());
    for (_, (mut rest, vals)) in groups {
        rest.insert(Arc::from(field), Val::arr(vals));
        out.push(Val::obj(rest));
    }
    Some(Val::arr(out))
}

/// `.pairwise()` — Arr → Arr<[arr[i], arr[i+1]]>.
#[inline]
pub fn pairwise_apply(recv: &Val) -> Option<Val> {
    let items_cow = recv.as_vals()?;
    let a = items_cow.as_ref();
    let mut out: Vec<Val> = Vec::with_capacity(a.len().saturating_sub(1));
    for w in a.windows(2) {
        out.push(Val::arr(vec![w[0].clone(), w[1].clone()]));
    }
    Some(Val::arr(out))
}

/// `.chunk(n)` Stage — Val::Arr → Arr<Arr<n>>.
#[inline]
pub fn chunk_arr_apply(recv: &Val, n: usize) -> Option<Val> {
    if n == 0 {
        return None;
    }
    if let Val::Arr(a) = recv {
        let chunks: Vec<Val> = a.chunks(n).map(|c| Val::arr(c.to_vec())).collect();
        Some(Val::arr(chunks))
    } else {
        None
    }
}

/// `.window(n)` Stage — Val::Arr → Arr<Arr<n>>.
#[inline]
pub fn window_arr_apply(recv: &Val, n: usize) -> Option<Val> {
    if n == 0 {
        return None;
    }
    if let Val::Arr(a) = recv {
        let windows: Vec<Val> = a.windows(n).map(|w| Val::arr(w.to_vec())).collect();
        Some(Val::arr(windows))
    } else {
        None
    }
}

/// `.intersect(other)` — keep elements in both arrays.
#[inline]
pub fn intersect_apply(recv: &Val, other: &[Val]) -> Option<Val> {
    if let Val::Arr(a) = recv {
        let other_keys: std::collections::HashSet<String> =
            other.iter().map(crate::util::val_to_key).collect();
        let kept: Vec<Val> = a
            .iter()
            .filter(|v| other_keys.contains(&crate::util::val_to_key(v)))
            .cloned()
            .collect();
        Some(Val::arr(kept))
    } else {
        None
    }
}

/// `.union(other)` — combine, preserve order, dedup.
#[inline]
pub fn union_apply(recv: &Val, other: &[Val]) -> Option<Val> {
    if let Val::Arr(a) = recv {
        let mut out: Vec<Val> = a.as_ref().clone();
        let a_keys: std::collections::HashSet<String> =
            a.iter().map(crate::util::val_to_key).collect();
        for v in other {
            if !a_keys.contains(&crate::util::val_to_key(v)) {
                out.push(v.clone());
            }
        }
        Some(Val::arr(out))
    } else {
        None
    }
}

/// `.diff(other)` — keep elements in self but not other.
#[inline]
pub fn diff_apply(recv: &Val, other: &[Val]) -> Option<Val> {
    if let Val::Arr(a) = recv {
        let other_keys: std::collections::HashSet<String> =
            other.iter().map(crate::util::val_to_key).collect();
        let kept: Vec<Val> = a
            .iter()
            .filter(|v| !other_keys.contains(&crate::util::val_to_key(v)))
            .cloned()
            .collect();
        Some(Val::arr(kept))
    } else {
        None
    }
}

// ── Phase D batch 5: object family ──────────────────────────────────

/// `.from_pairs()` — Arr<{key,val}> or Arr<[Str, Val]> → Obj.
#[inline]
pub fn from_pairs_apply(recv: &Val) -> Option<Val> {
    let items = recv.as_vals()?;
    let mut m: indexmap::IndexMap<Arc<str>, Val> = indexmap::IndexMap::with_capacity(items.len());
    for item in items.iter() {
        match item {
            Val::Arr(kv) if kv.len() == 2 => {
                if let Some(k) = kv[0].as_str_ref() {
                    m.insert(Arc::<str>::from(k), kv[1].clone());
                }
            }
            _ => {
                let k_val = item
                    .get("key")
                    .or_else(|| item.get("k"))
                    .cloned()
                    .unwrap_or(Val::Null);
                let v = item
                    .get("val")
                    .or_else(|| item.get("value"))
                    .or_else(|| item.get("v"))
                    .cloned()
                    .unwrap_or(Val::Null);
                if let Val::Str(k) = k_val {
                    m.insert(k, v);
                }
            }
        }
    }
    Some(Val::Obj(Arc::new(m)))
}

/// `.invert()` — Obj{k → v} → Obj{v_str → k}.
#[inline]
pub fn invert_apply(recv: &Val) -> Option<Val> {
    let m = recv.as_object()?;
    let mut out: indexmap::IndexMap<Arc<str>, Val> = indexmap::IndexMap::with_capacity(m.len());
    for (k, v) in m.iter() {
        let new_key: Arc<str> = match v {
            Val::Str(s) => s.clone(),
            Val::StrSlice(r) => Arc::<str>::from(r.as_str()),
            other => Arc::<str>::from(crate::util::val_to_key(other).as_str()),
        };
        out.insert(new_key, Val::Str(k.clone()));
    }
    Some(Val::Obj(Arc::new(out)))
}

/// `.merge(other)` — shallow merge; other wins on conflict.
#[inline]
pub fn merge_apply(recv: &Val, other: &Val) -> Option<Val> {
    let base = recv.as_object()?;
    let other = other.as_object()?;
    let mut out: indexmap::IndexMap<Arc<str>, Val> = base.clone();
    for (k, v) in other.iter() {
        out.insert(k.clone(), v.clone());
    }
    Some(Val::Obj(Arc::new(out)))
}

/// `.deep_merge(other)` — recursive merge.
#[inline]
pub fn deep_merge_apply(recv: &Val, other: &Val) -> Option<Val> {
    Some(crate::util::deep_merge(recv.clone(), other.clone()))
}

/// `.defaults(other)` — fill null/missing keys from other.
#[inline]
pub fn defaults_apply(recv: &Val, other: &Val) -> Option<Val> {
    let base = recv.as_object()?;
    let defs = other.as_object()?;
    let mut out: indexmap::IndexMap<Arc<str>, Val> = base.clone();
    for (k, v) in defs.iter() {
        let entry = out.entry(k.clone()).or_insert(Val::Null);
        if entry.is_null() {
            *entry = v.clone();
        }
    }
    Some(Val::Obj(Arc::new(out)))
}

/// `.rename({old: new, ...})` — rename keys per mapping.
#[inline]
pub fn rename_apply(recv: &Val, renames: &Val) -> Option<Val> {
    let base = recv.as_object()?;
    let renames = renames.as_object()?;
    let mut out: indexmap::IndexMap<Arc<str>, Val> = base.clone();
    for (old, new_val) in renames.iter() {
        if let Some(v) = out.shift_remove(old.as_ref()) {
            let new_key: Arc<str> = new_val
                .as_str_ref()
                .map(Arc::<str>::from)
                .unwrap_or_else(|| old.clone());
            out.insert(new_key, v);
        }
    }
    Some(Val::Obj(Arc::new(out)))
}

// ── Phase D batch 6: path family ────────────────────────────────────

pub(crate) enum PathSeg {
    Field(String),
    Index(i64),
}

pub(crate) enum PickSource {
    Field(Arc<str>),
    Path(Vec<PathSeg>),
}

pub(crate) struct PickSpec {
    pub out_key: Arc<str>,
    pub source: PickSource,
}

pub(crate) fn parse_path_segs(path: &str) -> Vec<PathSeg> {
    let mut segs = Vec::new();
    let mut cur = String::new();
    let mut chars = path.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '.' => {
                if !cur.is_empty() {
                    segs.push(PathSeg::Field(std::mem::take(&mut cur)));
                }
            }
            '[' => {
                if !cur.is_empty() {
                    segs.push(PathSeg::Field(std::mem::take(&mut cur)));
                }
                let mut idx = String::new();
                for c2 in chars.by_ref() {
                    if c2 == ']' {
                        break;
                    }
                    idx.push(c2);
                }
                segs.push(PathSeg::Index(idx.parse().unwrap_or(0)));
            }
            _ => cur.push(c),
        }
    }
    if !cur.is_empty() {
        segs.push(PathSeg::Field(cur));
    }
    segs
}

pub(crate) fn get_path_impl(val: &Val, segs: &[PathSeg]) -> Val {
    if segs.is_empty() {
        return val.clone();
    }
    let next = match &segs[0] {
        PathSeg::Field(f) => val.get(f).cloned().unwrap_or(Val::Null),
        PathSeg::Index(i) => val.get_index(*i),
    };
    get_path_impl(&next, &segs[1..])
}

pub(crate) fn set_path_impl(val: Val, segs: &[PathSeg], new_val: Val) -> Val {
    if segs.is_empty() {
        return new_val;
    }
    match (&segs[0], val) {
        (PathSeg::Field(f), Val::Obj(m)) => {
            let mut map = Arc::try_unwrap(m).unwrap_or_else(|m| (*m).clone());
            let child = map.shift_remove(f.as_str()).unwrap_or(Val::Null);
            map.insert(
                Arc::from(f.as_str()),
                set_path_impl(child, &segs[1..], new_val),
            );
            Val::obj(map)
        }
        (PathSeg::Index(i), Val::Arr(a)) => {
            let mut arr = Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone());
            let idx = resolve_path_idx(*i, arr.len() as i64);
            if idx < arr.len() {
                let child = arr[idx].clone();
                arr[idx] = set_path_impl(child, &segs[1..], new_val);
            }
            Val::arr(arr)
        }
        (PathSeg::Field(f), _) => {
            let mut m = IndexMap::new();
            m.insert(
                Arc::from(f.as_str()),
                set_path_impl(Val::Null, &segs[1..], new_val),
            );
            Val::obj(m)
        }
        (_, v) => v,
    }
}

pub(crate) fn del_path_impl(val: Val, segs: &[PathSeg]) -> Val {
    if segs.is_empty() {
        return Val::Null;
    }
    match (&segs[0], val) {
        (PathSeg::Field(f), Val::Obj(m)) => {
            let mut map = Arc::try_unwrap(m).unwrap_or_else(|m| (*m).clone());
            if segs.len() == 1 {
                map.shift_remove(f.as_str());
            } else if let Some(child) = map.shift_remove(f.as_str()) {
                map.insert(Arc::from(f.as_str()), del_path_impl(child, &segs[1..]));
            }
            Val::obj(map)
        }
        (PathSeg::Index(i), Val::Arr(a)) => {
            let mut arr = Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone());
            let idx = resolve_path_idx(*i, arr.len() as i64);
            if segs.len() == 1 {
                if idx < arr.len() {
                    arr.remove(idx);
                }
            } else if idx < arr.len() {
                let child = arr[idx].clone();
                arr[idx] = del_path_impl(child, &segs[1..]);
            }
            Val::arr(arr)
        }
        (_, v) => v,
    }
}

fn resolve_path_idx(i: i64, len: i64) -> usize {
    (if i < 0 { (len + i).max(0) } else { i }) as usize
}

pub(crate) fn flatten_keys_impl(
    prefix: &str,
    val: &Val,
    sep: &str,
    out: &mut IndexMap<Arc<str>, Val>,
) {
    match val {
        Val::Obj(m) => {
            for (k, v) in m.iter() {
                let full = if prefix.is_empty() {
                    k.to_string()
                } else {
                    format!("{}{}{}", prefix, sep, k)
                };
                flatten_keys_impl(&full, v, sep, out);
            }
        }
        _ => {
            out.insert(Arc::from(prefix), val.clone());
        }
    }
}

pub(crate) fn unflatten_keys_impl(m: &IndexMap<Arc<str>, Val>, sep: &str) -> Val {
    let mut root: IndexMap<Arc<str>, Val> = IndexMap::new();
    for (key, val) in m {
        let parts: Vec<&str> = key.split(sep).collect();
        insert_nested(&mut root, &parts, val.clone());
    }
    Val::obj(root)
}

fn insert_nested(obj: &mut IndexMap<Arc<str>, Val>, parts: &[&str], val: Val) {
    if parts.is_empty() {
        return;
    }
    if parts.len() == 1 {
        obj.insert(val_key(parts[0]), val);
        return;
    }
    let entry = obj
        .entry(val_key(parts[0]))
        .or_insert_with(|| Val::obj(IndexMap::new()));
    if let Val::Obj(child) = entry {
        insert_nested(Arc::make_mut(child), &parts[1..], val);
    }
}

/// `.get_path(path)` — read leaf at dotted/bracket path.
#[inline]
pub fn get_path_apply(recv: &Val, path: &str) -> Option<Val> {
    let segs = parse_path_segs(path);
    Some(get_path_impl(recv, &segs))
}

/// `.has_path(path)` — Val::Bool, true if path resolves non-null.
#[inline]
pub fn has_path_apply(recv: &Val, path: &str) -> Option<Val> {
    let segs = parse_path_segs(path);
    let found = !get_path_impl(recv, &segs).is_null();
    Some(Val::Bool(found))
}

/// `.has(key)` — Val::Bool, true if Obj has key.
#[inline]
pub fn has_apply(recv: &Val, key: &str) -> Option<Val> {
    let m = recv.as_object()?;
    Some(Val::Bool(m.contains_key(key)))
}

/// `.pick(keys...)` — keep selected object keys. For array receivers, applies
/// the projection to each object element.
#[inline]
pub fn pick_apply(recv: &Val, keys: &[Arc<str>]) -> Option<Val> {
    use indexmap::IndexMap;

    fn pick_obj(m: &IndexMap<Arc<str>, Val>, keys: &[Arc<str>]) -> Val {
        let mut out = IndexMap::with_capacity(keys.len());
        for key in keys {
            if let Some(v) = m.get(key.as_ref()) {
                out.insert(key.clone(), v.clone());
            }
        }
        Val::obj(out)
    }

    match recv {
        Val::Obj(m) => Some(pick_obj(m, keys)),
        Val::Arr(a) => Some(Val::arr(
            a.iter()
                .filter_map(|v| match v {
                    Val::Obj(m) => Some(pick_obj(m, keys)),
                    _ => None,
                })
                .collect(),
        )),
        _ => None,
    }
}

#[inline]
pub(crate) fn pick_specs_apply(recv: &Val, specs: &[PickSpec]) -> Option<Val> {
    fn pick_obj(m: &IndexMap<Arc<str>, Val>, specs: &[PickSpec]) -> Val {
        let mut out = IndexMap::with_capacity(specs.len());
        let wrapped = Val::Obj(Arc::new(m.clone()));
        for spec in specs {
            match &spec.source {
                PickSource::Field(src) => {
                    if let Some(v) = m.get(src.as_ref()) {
                        out.insert(spec.out_key.clone(), v.clone());
                    }
                }
                PickSource::Path(segs) => {
                    let v = get_path_impl(&wrapped, segs);
                    if !v.is_null() {
                        out.insert(spec.out_key.clone(), v);
                    }
                }
            }
        }
        Val::obj(out)
    }

    match recv {
        Val::Obj(m) => Some(pick_obj(m, specs)),
        Val::Arr(a) => Some(Val::arr(
            a.iter()
                .filter_map(|v| match v {
                    Val::Obj(m) => Some(pick_obj(m, specs)),
                    _ => None,
                })
                .collect(),
        )),
        _ => None,
    }
}

/// `.omit(keys...)` — drop selected object keys. For array receivers, applies
/// the projection to each object element.
#[inline]
pub fn omit_apply(recv: &Val, keys: &[Arc<str>]) -> Option<Val> {
    fn omit_obj(m: &indexmap::IndexMap<Arc<str>, Val>, keys: &[Arc<str>]) -> Val {
        let mut out = m.clone();
        for key in keys {
            out.shift_remove(key.as_ref());
        }
        Val::obj(out)
    }

    match recv {
        Val::Obj(m) => Some(omit_obj(m, keys)),
        Val::Arr(a) => Some(Val::arr(
            a.iter()
                .filter_map(|v| match v {
                    Val::Obj(m) => Some(omit_obj(m, keys)),
                    _ => None,
                })
                .collect(),
        )),
        _ => None,
    }
}

/// `.del_path(path)` — remove value at dotted path.
#[inline]
pub fn del_path_apply(recv: &Val, path: &str) -> Option<Val> {
    let segs = parse_path_segs(path);
    Some(del_path_impl(recv.clone(), &segs))
}

/// `.set_path(path, value)` — set value at dotted path.
#[inline]
pub fn set_path_apply(recv: &Val, path: &str, value: &Val) -> Option<Val> {
    let segs = parse_path_segs(path);
    Some(set_path_impl(recv.clone(), &segs, value.clone()))
}

/// `.del_paths(p1, p2, ...)` — remove multiple dotted paths.
#[inline]
pub fn del_paths_apply(recv: &Val, paths: &[Arc<str>]) -> Option<Val> {
    let mut out = recv.clone();
    for path in paths {
        let segs = parse_path_segs(path.as_ref());
        out = del_path_impl(out, &segs);
    }
    Some(out)
}

/// `.flatten_keys(sep)` — Obj → flat-Obj with `sep`-joined keys.
#[inline]
pub fn flatten_keys_apply(recv: &Val, sep: &str) -> Option<Val> {
    let mut out: indexmap::IndexMap<Arc<str>, Val> = indexmap::IndexMap::new();
    flatten_keys_impl("", recv, sep, &mut out);
    Some(Val::obj(out))
}

/// `.unflatten_keys(sep)` — flat-Obj → nested Obj.
#[inline]
pub fn unflatten_keys_apply(recv: &Val, sep: &str) -> Option<Val> {
    if let Val::Obj(m) = recv {
        Some(unflatten_keys_impl(m, sep))
    } else {
        None
    }
}

// ── Phase D batch 7: regex family ───────────────────────────────────

#[inline]
fn compile_regex_eval(pat: &str) -> Result<Arc<regex::Regex>, EvalError> {
    crate::builtin_helpers::compile_regex(pat).map_err(EvalError)
}

/// `.match(pat)` — Bool, regex match anywhere.
#[inline]
pub fn re_match_apply(recv: &Val, pat: &str) -> Option<Val> {
    try_re_match_apply(recv, pat).ok().flatten()
}

#[inline]
pub fn try_re_match_apply(recv: &Val, pat: &str) -> Result<Option<Val>, EvalError> {
    let Some(s) = recv.as_str_ref() else {
        return Ok(None);
    };
    let re = compile_regex_eval(pat)?;
    Ok(Some(Val::Bool(re.is_match(s))))
}

/// `.match_first(pat)` — Str of first match or Null.
#[inline]
pub fn re_match_first_apply(recv: &Val, pat: &str) -> Option<Val> {
    try_re_match_first_apply(recv, pat).ok().flatten()
}

#[inline]
pub fn try_re_match_first_apply(recv: &Val, pat: &str) -> Result<Option<Val>, EvalError> {
    let Some(s) = recv.as_str_ref() else {
        return Ok(None);
    };
    let re = compile_regex_eval(pat)?;
    Ok(Some(
        re.find(s)
            .map(|m| Val::Str(Arc::from(m.as_str())))
            .unwrap_or(Val::Null),
    ))
}

/// `.match_all(pat)` — StrVec of all matches.
#[inline]
pub fn re_match_all_apply(recv: &Val, pat: &str) -> Option<Val> {
    try_re_match_all_apply(recv, pat).ok().flatten()
}

#[inline]
pub fn try_re_match_all_apply(recv: &Val, pat: &str) -> Result<Option<Val>, EvalError> {
    let Some(s) = recv.as_str_ref() else {
        return Ok(None);
    };
    let re = compile_regex_eval(pat)?;
    let out: Vec<Arc<str>> = re
        .find_iter(s)
        .map(|m| Arc::<str>::from(m.as_str()))
        .collect();
    Ok(Some(Val::str_vec(out)))
}

/// `.captures(pat)` — Arr of capture groups (group 0 first); Null on miss.
#[inline]
pub fn re_captures_apply(recv: &Val, pat: &str) -> Option<Val> {
    try_re_captures_apply(recv, pat).ok().flatten()
}

#[inline]
pub fn try_re_captures_apply(recv: &Val, pat: &str) -> Result<Option<Val>, EvalError> {
    let Some(s) = recv.as_str_ref() else {
        return Ok(None);
    };
    let re = compile_regex_eval(pat)?;
    Ok(Some(match re.captures(s) {
        Some(c) => {
            let mut out: Vec<Val> = Vec::with_capacity(c.len());
            for i in 0..c.len() {
                out.push(
                    c.get(i)
                        .map(|m| Val::Str(Arc::from(m.as_str())))
                        .unwrap_or(Val::Null),
                );
            }
            Val::arr(out)
        }
        None => Val::Null,
    }))
}

/// `.captures_all(pat)` — Arr<Arr> of capture groups for every match.
#[inline]
pub fn re_captures_all_apply(recv: &Val, pat: &str) -> Option<Val> {
    try_re_captures_all_apply(recv, pat).ok().flatten()
}

#[inline]
pub fn try_re_captures_all_apply(recv: &Val, pat: &str) -> Result<Option<Val>, EvalError> {
    let Some(s) = recv.as_str_ref() else {
        return Ok(None);
    };
    let re = compile_regex_eval(pat)?;
    let mut all: Vec<Val> = Vec::new();
    for c in re.captures_iter(s) {
        let mut row: Vec<Val> = Vec::with_capacity(c.len());
        for i in 0..c.len() {
            row.push(
                c.get(i)
                    .map(|m| Val::Str(Arc::from(m.as_str())))
                    .unwrap_or(Val::Null),
            );
        }
        all.push(Val::arr(row));
    }
    Ok(Some(Val::arr(all)))
}

/// `.replace_re(pat, with)` — single regex replacement.
#[inline]
pub fn re_replace_apply(recv: &Val, pat: &str, with: &str) -> Option<Val> {
    try_re_replace_apply(recv, pat, with).ok().flatten()
}

#[inline]
pub fn try_re_replace_apply(recv: &Val, pat: &str, with: &str) -> Result<Option<Val>, EvalError> {
    let Some(s) = recv.as_str_ref() else {
        return Ok(None);
    };
    let re = compile_regex_eval(pat)?;
    let out = re.replace(s, with);
    Ok(Some(Val::Str(Arc::from(out.as_ref()))))
}

/// `.replace_all_re(pat, with)` — all regex replacements.
#[inline]
pub fn re_replace_all_apply(recv: &Val, pat: &str, with: &str) -> Option<Val> {
    try_re_replace_all_apply(recv, pat, with).ok().flatten()
}

#[inline]
pub fn try_re_replace_all_apply(
    recv: &Val,
    pat: &str,
    with: &str,
) -> Result<Option<Val>, EvalError> {
    let Some(s) = recv.as_str_ref() else {
        return Ok(None);
    };
    let re = compile_regex_eval(pat)?;
    let out = re.replace_all(s, with);
    Ok(Some(Val::Str(Arc::from(out.as_ref()))))
}

/// `.split_re(pat)` — regex split into StrVec.
#[inline]
pub fn re_split_apply(recv: &Val, pat: &str) -> Option<Val> {
    try_re_split_apply(recv, pat).ok().flatten()
}

#[inline]
pub fn try_re_split_apply(recv: &Val, pat: &str) -> Result<Option<Val>, EvalError> {
    let Some(s) = recv.as_str_ref() else {
        return Ok(None);
    };
    let re = compile_regex_eval(pat)?;
    let out: Vec<Arc<str>> = re.split(s).map(Arc::<str>::from).collect();
    Ok(Some(Val::str_vec(out)))
}

/// `.contains_any([needles])` — Bool, true if any needle appears.
#[inline]
pub fn contains_any_apply(recv: &Val, needles: &[Arc<str>]) -> Option<Val> {
    let s = recv.as_str_ref()?;
    Some(Val::Bool(needles.iter().any(|n| s.contains(n.as_ref()))))
}

/// `.contains_all([needles])` — Bool, true if every needle appears.
#[inline]
pub fn contains_all_apply(recv: &Val, needles: &[Arc<str>]) -> Option<Val> {
    let s = recv.as_str_ref()?;
    Some(Val::Bool(needles.iter().all(|n| s.contains(n.as_ref()))))
}

// ── Phase D batch 8: csv / cast / type-name ────────────────────────

/// `.to_csv()` — Val → Str (comma sep).
#[inline]
pub fn to_csv_apply(recv: &Val) -> Option<Val> {
    Some(Val::Str(Arc::from(
        crate::builtin_helpers::csv_emit(recv, ",").as_str(),
    )))
}

/// `.to_tsv()` — Val → Str (tab sep).
#[inline]
pub fn to_tsv_apply(recv: &Val) -> Option<Val> {
    Some(Val::Str(Arc::from(
        crate::builtin_helpers::csv_emit(recv, "\t").as_str(),
    )))
}

/// `.to_pairs()` — Obj → Arr<{key, val}> (named-obj form).
#[inline]
pub fn to_pairs_apply(recv: &Val) -> Option<Val> {
    use crate::util::obj2;
    let arr: Vec<Val> = recv
        .as_object()
        .map(|m| {
            m.iter()
                .map(|(k, v)| obj2("key", Val::Str(k.clone()), "val", v.clone()))
                .collect()
        })
        .unwrap_or_default();
    Some(Val::arr(arr))
}

/// `.type()` — Val → Str (type name).
#[inline]
pub fn type_name_apply(recv: &Val) -> Option<Val> {
    Some(Val::Str(Arc::from(recv.type_name())))
}

/// `.to_string()` — Val → Str (display form).
#[inline]
pub fn to_string_apply(recv: &Val) -> Option<Val> {
    Some(Val::Str(Arc::from(
        crate::util::val_to_string(recv).as_str(),
    )))
}

/// `.to_json()` — Val → Str (JSON encoding). Inline fast paths for primitives.
#[inline]
pub fn to_json_apply(recv: &Val) -> Option<Val> {
    let out = match recv {
        Val::Int(n) => n.to_string(),
        Val::Float(f) => {
            if f.is_finite() {
                let v = serde_json::Value::from(*f);
                serde_json::to_string(&v).unwrap_or_default()
            } else {
                "null".to_string()
            }
        }
        Val::Bool(b) => (if *b { "true" } else { "false" }).to_string(),
        Val::Null => "null".to_string(),
        Val::Str(s) => {
            let v = serde_json::Value::String(s.to_string());
            serde_json::to_string(&v).unwrap_or_default()
        }
        other => {
            let sv: serde_json::Value = other.clone().into();
            serde_json::to_string(&sv).unwrap_or_default()
        }
    };
    Some(Val::Str(Arc::from(out)))
}

/// `.from_json()` — parse receiver as JSON.
#[inline]
pub fn from_json_apply(recv: &Val) -> Option<Val> {
    try_from_json_apply(recv).ok().flatten()
}

#[inline]
pub fn try_from_json_apply(recv: &Val) -> Result<Option<Val>, EvalError> {
    #[cfg(feature = "simd-json")]
    {
        let bytes_owned: Vec<u8> = match recv {
            Val::Str(s) => s.as_bytes().to_vec(),
            _ => crate::util::val_to_string(recv).into_bytes(),
        };
        let mut bytes = bytes_owned;
        return Val::from_json_simd(&mut bytes)
            .map(Some)
            .map_err(|e| EvalError(format!("from_json: {}", e)));
    }
    #[cfg(not(feature = "simd-json"))]
    {
        match recv {
            Val::Str(s) => Val::from_json_str(s.as_ref())
                .map(Some)
                .map_err(|e| EvalError(format!("from_json: {}", e))),
            _ => {
                let s = crate::util::val_to_string(recv);
                Val::from_json_str(&s)
                    .map(Some)
                    .map_err(|e| EvalError(format!("from_json: {}", e)))
            }
        }
    }
}

/// `.or(default)` — default only when receiver is null.
#[inline]
pub fn or_apply(recv: &Val, default: &Val) -> Val {
    if recv.is_null() {
        default.clone()
    } else {
        recv.clone()
    }
}

/// `.missing(key)` — negated nested field existence test.
#[inline]
pub fn missing_apply(recv: &Val, key: &str) -> Val {
    Val::Bool(!crate::util::field_exists_nested(recv, key))
}

/// `.includes(item)` / `.contains(item)` — membership or substring check.
#[inline]
pub fn includes_apply(recv: &Val, item: &Val) -> Val {
    use crate::util::val_to_key;
    let key = val_to_key(item);
    Val::Bool(match recv {
        Val::Arr(a) => a.iter().any(|v| val_to_key(v) == key),
        Val::IntVec(a) => a.iter().any(|n| val_to_key(&Val::Int(*n)) == key),
        Val::FloatVec(a) => a.iter().any(|f| val_to_key(&Val::Float(*f)) == key),
        Val::StrVec(a) => match item.as_str() {
            Some(needle) => a.iter().any(|s| s.as_ref() == needle),
            None => false,
        },
        Val::Str(s) => s.contains(item.as_str().unwrap_or_default()),
        Val::StrSlice(s) => s.as_str().contains(item.as_str().unwrap_or_default()),
        Val::Obj(m) => match item.as_str() {
            Some(k) => m.contains_key(k),
            None => false,
        },
        Val::ObjSmall(p) => match item.as_str() {
            Some(k) => p.iter().any(|(kk, _)| kk.as_ref() == k),
            None => false,
        },
        _ => false,
    })
}

pub(crate) fn schema_of(v: &Val) -> Val {
    match v {
        Val::Null => ty_obj("Null"),
        Val::Bool(_) => ty_obj("Bool"),
        Val::Int(_) => ty_obj("Int"),
        Val::Float(_) => ty_obj("Float"),
        Val::Str(_) | Val::StrSlice(_) => ty_obj("String"),
        Val::IntVec(a) => array_schema(a.len(), ty_obj("Int")),
        Val::FloatVec(a) => array_schema(a.len(), ty_obj("Float")),
        Val::StrVec(a) => array_schema(a.len(), ty_obj("String")),
        Val::StrSliceVec(a) => array_schema(a.len(), ty_obj("String")),
        Val::ObjVec(d) => array_schema(d.nrows(), ty_obj("Object")),
        Val::Arr(a) => {
            let items = if a.is_empty() {
                ty_obj("Unknown")
            } else {
                let mut acc = schema_of(&a[0]);
                for el in a.iter().skip(1) {
                    acc = unify_schema(acc, schema_of(el));
                }
                acc
            };
            array_schema(a.len(), items)
        }
        Val::Obj(m) => schema_object(m.iter().map(|(k, v)| (k.clone(), v))),
        Val::ObjSmall(pairs) => schema_object(pairs.iter().map(|(k, v)| (k.clone(), v))),
    }
}

fn schema_object<'a>(pairs: impl Iterator<Item = (Arc<str>, &'a Val)>) -> Val {
    let mut required = Vec::new();
    let mut fields = IndexMap::new();
    for (k, child) in pairs {
        let mut field = schema_of(child);
        if matches!(child, Val::Null) {
            field = set_schema_field(field, "nullable", Val::Bool(true));
        } else {
            required.push(Val::Str(k.clone()));
        }
        fields.insert(k, field);
    }
    let mut out: IndexMap<Arc<str>, Val> = IndexMap::with_capacity(3);
    out.insert(Arc::from("type"), Val::Str(Arc::from("Object")));
    out.insert(Arc::from("required"), Val::arr(required));
    out.insert(Arc::from("fields"), Val::obj(fields));
    Val::obj(out)
}

fn ty_obj(name: &str) -> Val {
    let mut m: IndexMap<Arc<str>, Val> = IndexMap::with_capacity(1);
    m.insert(Arc::from("type"), Val::Str(Arc::from(name)));
    Val::obj(m)
}

fn array_schema(len: usize, items: Val) -> Val {
    let mut m: IndexMap<Arc<str>, Val> = IndexMap::with_capacity(3);
    m.insert(Arc::from("type"), Val::Str(Arc::from("Array")));
    m.insert(Arc::from("len"), Val::Int(len as i64));
    m.insert(Arc::from("items"), items);
    Val::obj(m)
}

fn set_schema_field(obj: Val, key: &str, v: Val) -> Val {
    if let Val::Obj(m) = obj {
        let mut m = Arc::try_unwrap(m).unwrap_or_else(|arc| (*arc).clone());
        m.insert(Arc::from(key), v);
        Val::obj(m)
    } else {
        obj
    }
}

fn schema_type(v: &Val) -> Option<&str> {
    if let Val::Obj(m) = v {
        if let Some(Val::Str(s)) = m.get("type") {
            return Some(s.as_ref());
        }
    }
    None
}

fn unify_schema(a: Val, b: Val) -> Val {
    match (schema_type(&a), schema_type(&b)) {
        (Some(x), Some(y)) if x == y => match x {
            "Object" => unify_object_schemas(a, b),
            "Array" => unify_array_schemas(a, b),
            _ => mark_nullable_if_either(a, b),
        },
        (Some("Null"), _) => set_schema_field(b, "nullable", Val::Bool(true)),
        (_, Some("Null")) => set_schema_field(a, "nullable", Val::Bool(true)),
        _ => ty_obj("Mixed"),
    }
}

fn mark_nullable_if_either(a: Val, b: Val) -> Val {
    if is_schema_nullable(&a) || is_schema_nullable(&b) {
        set_schema_field(a, "nullable", Val::Bool(true))
    } else {
        a
    }
}

fn is_schema_nullable(v: &Val) -> bool {
    matches!(
        v,
        Val::Obj(m) if matches!(m.get("nullable"), Some(Val::Bool(true)))
    )
}

fn unify_array_schemas(a: Val, b: Val) -> Val {
    let items = match (
        extract_schema_field(&a, "items"),
        extract_schema_field(&b, "items"),
    ) {
        (Some(x), Some(y)) => unify_schema(x, y),
        (Some(x), None) => x,
        (None, Some(y)) => y,
        (None, None) => ty_obj("Unknown"),
    };
    let la = extract_schema_int(&a, "len").unwrap_or(0);
    let lb = extract_schema_int(&b, "len").unwrap_or(0);
    array_schema((la + lb) as usize, items)
}

fn extract_schema_field(v: &Val, key: &str) -> Option<Val> {
    if let Val::Obj(m) = v {
        m.get(key).cloned()
    } else {
        None
    }
}

fn extract_schema_int(v: &Val, key: &str) -> Option<i64> {
    if let Some(Val::Int(n)) = extract_schema_field(v, key) {
        Some(n)
    } else {
        None
    }
}

fn unify_object_schemas(a: Val, b: Val) -> Val {
    let (Some(Val::Obj(a_fields)), Some(Val::Obj(b_fields))) = (
        extract_schema_field(&a, "fields"),
        extract_schema_field(&b, "fields"),
    ) else {
        return ty_obj("Object");
    };
    let a_map = Arc::try_unwrap(a_fields).unwrap_or_else(|arc| (*arc).clone());
    let b_map = Arc::try_unwrap(b_fields).unwrap_or_else(|arc| (*arc).clone());
    let a_req = extract_required_set(&a);
    let b_req = extract_required_set(&b);

    let mut out_fields: IndexMap<Arc<str>, Val> =
        IndexMap::with_capacity(a_map.len().max(b_map.len()));
    let mut all_keys: Vec<Arc<str>> = Vec::with_capacity(a_map.len() + b_map.len());
    for (k, _) in &a_map {
        all_keys.push(k.clone());
    }
    for (k, _) in &b_map {
        if !a_map.contains_key(k) {
            all_keys.push(k.clone());
        }
    }

    let mut required = Vec::new();
    for k in all_keys {
        let av = a_map.get(&k).cloned();
        let bv = b_map.get(&k).cloned();
        let field = match (av, bv) {
            (Some(x), Some(y)) => unify_schema(x, y),
            (Some(x), None) => set_schema_field(x, "optional", Val::Bool(true)),
            (None, Some(y)) => set_schema_field(y, "optional", Val::Bool(true)),
            _ => ty_obj("Unknown"),
        };
        if a_req.contains(k.as_ref()) && b_req.contains(k.as_ref()) {
            required.push(Val::Str(k.clone()));
        }
        out_fields.insert(k, field);
    }

    let mut out: IndexMap<Arc<str>, Val> = IndexMap::with_capacity(3);
    out.insert(Arc::from("type"), Val::Str(Arc::from("Object")));
    out.insert(Arc::from("required"), Val::arr(required));
    out.insert(Arc::from("fields"), Val::obj(out_fields));
    Val::obj(out)
}

fn extract_required_set(v: &Val) -> std::collections::HashSet<String> {
    let mut set = std::collections::HashSet::new();
    if let Some(Val::Arr(a)) = extract_schema_field(v, "required") {
        for el in a.iter() {
            if let Val::Str(k) = el {
                set.insert(k.to_string());
            }
        }
    }
    set
}

/// `.schema()` — Val → schema Obj describing types/required/array shape.
#[inline]
pub fn schema_apply(recv: &Val) -> Option<Val> {
    Some(schema_of(recv))
}

#[cfg(test)]
mod spec_tests {
    use super::{
        BuiltinCardinality, BuiltinCategory, BuiltinMethod, BuiltinPipelineSink,
        BuiltinPipelineStage, BuiltinViewMaterialization, BuiltinViewSink, BuiltinViewStage,
    };

    #[test]
    fn builtin_specs_describe_execution_shape() {
        let map = BuiltinMethod::Map.spec();
        assert_eq!(map.category, BuiltinCategory::StreamingOneToOne);
        assert_eq!(map.cardinality, BuiltinCardinality::OneToOne);

        let flat_map = BuiltinMethod::FlatMap.spec();
        assert_eq!(flat_map.category, BuiltinCategory::StreamingExpand);
        assert_eq!(flat_map.cardinality, BuiltinCardinality::Expanding);

        let sum = BuiltinMethod::Sum.spec();
        assert_eq!(sum.category, BuiltinCategory::Reducer);
        assert_eq!(sum.cardinality, BuiltinCardinality::Reducing);

        let sort = BuiltinMethod::Sort.spec();
        assert_eq!(sort.category, BuiltinCategory::Barrier);
        assert_eq!(sort.cardinality, BuiltinCardinality::Barrier);
    }

    #[test]
    fn builtin_specs_drive_pipeline_element_lowering() {
        assert!(BuiltinMethod::Upper.spec().pipeline_element);
        assert!(BuiltinMethod::Lines.spec().pipeline_element);
        assert!(BuiltinMethod::GetPath.spec().pipeline_element);

        assert!(!BuiltinMethod::Len.spec().pipeline_element);
        assert!(!BuiltinMethod::FromJson.spec().pipeline_element);
        assert!(!BuiltinMethod::Sort.spec().pipeline_element);
        assert!(!BuiltinMethod::Flatten.spec().pipeline_element);
    }

    #[test]
    fn builtin_specs_drive_view_stage_lowering() {
        assert_eq!(
            BuiltinMethod::Filter.spec().view_stage,
            Some(BuiltinViewStage::Filter)
        );
        assert_eq!(
            BuiltinMethod::Map.spec().view_stage,
            Some(BuiltinViewStage::Map)
        );
        assert_eq!(
            BuiltinMethod::FlatMap.spec().view_stage,
            Some(BuiltinViewStage::FlatMap)
        );

        assert_eq!(BuiltinMethod::Sort.spec().view_stage, None);
        assert_eq!(BuiltinMethod::Upper.spec().view_stage, None);
    }

    #[test]
    fn builtin_specs_drive_view_sink_lowering() {
        assert_eq!(
            BuiltinMethod::Count.spec().view_sink,
            Some(BuiltinViewSink::Count)
        );
        assert_eq!(
            BuiltinMethod::Len.spec().view_sink,
            Some(BuiltinViewSink::Count)
        );
        assert_eq!(
            BuiltinMethod::Sum.spec().view_sink,
            Some(BuiltinViewSink::Numeric)
        );
        assert_eq!(
            BuiltinMethod::First.spec().view_sink,
            Some(BuiltinViewSink::First)
        );
        assert_eq!(
            BuiltinMethod::Last.spec().view_sink,
            Some(BuiltinViewSink::Last)
        );

        assert_eq!(
            BuiltinViewSink::Count.materialization(),
            BuiltinViewMaterialization::Never
        );
        assert_eq!(
            BuiltinViewSink::Numeric.materialization(),
            BuiltinViewMaterialization::SinkNumericInput
        );
        assert_eq!(BuiltinMethod::Sort.spec().view_sink, None);
    }

    #[test]
    fn builtin_specs_drive_view_scalar_kernels() {
        assert!(BuiltinMethod::Len.spec().view_scalar);
        assert!(BuiltinMethod::StartsWith.spec().view_scalar);
        assert!(BuiltinMethod::EndsWith.spec().view_scalar);
        assert!(!BuiltinMethod::Sort.spec().view_scalar);
        assert!(!BuiltinMethod::FromJson.spec().view_scalar);
    }

    #[test]
    fn builtin_specs_drive_pipeline_stage_lowering() {
        assert_eq!(
            BuiltinMethod::Filter.spec().pipeline_stage,
            Some(BuiltinPipelineStage::Unary)
        );
        assert_eq!(
            BuiltinMethod::CountBy.spec().pipeline_stage,
            Some(BuiltinPipelineStage::Unary)
        );
        assert_eq!(
            BuiltinMethod::Sort.spec().pipeline_stage,
            Some(BuiltinPipelineStage::Nullary)
        );
        assert_eq!(
            BuiltinMethod::Reverse.spec().pipeline_stage,
            Some(BuiltinPipelineStage::Nullary)
        );
        assert_eq!(
            BuiltinMethod::Take.spec().pipeline_stage,
            Some(BuiltinPipelineStage::Unary)
        );
        assert_eq!(
            BuiltinMethod::FindFirst.spec().pipeline_stage,
            Some(BuiltinPipelineStage::Unary)
        );

        assert_eq!(BuiltinMethod::Len.spec().pipeline_stage, None);
        assert_eq!(BuiltinMethod::FromJson.spec().pipeline_stage, None);
    }

    #[test]
    fn builtin_specs_drive_pipeline_sink_lowering() {
        assert_eq!(
            BuiltinMethod::ApproxCountDistinct.spec().pipeline_sink,
            Some(BuiltinPipelineSink::ApproxCountDistinct)
        );
        assert_eq!(BuiltinMethod::Count.spec().pipeline_sink, None);
    }
}
