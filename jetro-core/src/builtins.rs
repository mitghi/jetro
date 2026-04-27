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

use crate::eval::util::is_truthy;
use crate::eval::value::Val;
use crate::eval::EvalError;
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
    GroupBy,
    CountBy,
    IndexBy,
    // Array ops
    Filter,
    Map,
    FlatMap,
    Sort,
    Unique,
    Flatten,
    Compact,
    Join,
    First,
    Last,
    Nth,
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
    Accumulate,
    Partition,
    Zip,
    ZipLongest,
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
    Set,
    Update,
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
            "any" => Self::Any,
            "all" => Self::All,
            "groupBy" | "group_by" => Self::GroupBy,
            "countBy" | "count_by" => Self::CountBy,
            "indexBy" | "index_by" => Self::IndexBy,
            "filter" => Self::Filter,
            "map" => Self::Map,
            "flatMap" | "flat_map" => Self::FlatMap,
            "sort" => Self::Sort,
            "unique" | "distinct" => Self::Unique,
            "flatten" => Self::Flatten,
            "compact" => Self::Compact,
            "join" => Self::Join,
            "equi_join" | "equiJoin" => Self::EquiJoin,
            "first" => Self::First,
            "last" => Self::Last,
            "nth" => Self::Nth,
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
            "accumulate" => Self::Accumulate,
            "partition" => Self::Partition,
            "zip" => Self::Zip,
            "zip_longest" | "zipLongest" => Self::ZipLongest,
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
            "set" => Self::Set,
            "update" => Self::Update,
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
            "repeat" => Self::Repeat,
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
    Usize(usize),
    Pad { width: usize, fill: char },
}

#[derive(Debug, Clone)]
pub struct BuiltinCall {
    pub method: BuiltinMethod,
    pub args: BuiltinArgs,
}

#[derive(Debug, Clone, Copy)]
pub struct BuiltinSpec {
    pub pure: bool,
    pub one_to_one: bool,
    pub can_indexed: bool,
    pub cost: f64,
}

impl BuiltinCall {
    #[inline]
    pub fn new(method: BuiltinMethod, args: BuiltinArgs) -> Self {
        Self { method, args }
    }

    #[inline]
    pub fn spec(&self) -> BuiltinSpec {
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
            _ => (1.0, true),
        };
        BuiltinSpec {
            pure: true,
            one_to_one: true,
            can_indexed,
            cost,
        }
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
            (BuiltinMethod::Keys, BuiltinArgs::None) => return Some(keys_apply(recv)),
            (BuiltinMethod::Values, BuiltinArgs::None) => return Some(values_apply(recv)),
            (BuiltinMethod::Entries, BuiltinArgs::None) => return Some(entries_apply(recv)),
            (BuiltinMethod::Invert, BuiltinArgs::None) => apply_or_recv!(invert_apply(recv)),
            (BuiltinMethod::Type, BuiltinArgs::None) => apply_or_recv!(type_name_apply(recv)),
            (BuiltinMethod::ToString, BuiltinArgs::None) => apply_or_recv!(to_string_apply(recv)),
            (BuiltinMethod::ToJson, BuiltinArgs::None) => apply_or_recv!(to_json_apply(recv)),
            (BuiltinMethod::ToCsv, BuiltinArgs::None) => apply_or_recv!(to_csv_apply(recv)),
            (BuiltinMethod::ToTsv, BuiltinArgs::None) => apply_or_recv!(to_tsv_apply(recv)),
            (BuiltinMethod::ToPairs, BuiltinArgs::None) => apply_or_recv!(to_pairs_apply(recv)),
            (BuiltinMethod::Compact, BuiltinArgs::None) => apply_or_recv!(compact_apply(recv)),
            (BuiltinMethod::Pairwise, BuiltinArgs::None) => apply_or_recv!(pairwise_apply(recv)),
            (BuiltinMethod::Schema, BuiltinArgs::None) => apply_or_recv!(schema_apply(recv)),
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

    pub fn from_literal_ast_args(name: &str, args: &[crate::ast::Arg]) -> Option<Self> {
        use crate::ast::{Arg, Expr};

        let method = BuiltinMethod::from_name(name);
        if method == BuiltinMethod::Unknown {
            return None;
        }

        let str_arg = |idx: usize| -> Option<Arc<str>> {
            match args.get(idx)? {
                Arg::Pos(Expr::Str(s)) => Some(Arc::from(s.as_str())),
                _ => None,
            }
        };
        let usize_arg = |idx: usize| -> Option<usize> {
            match args.get(idx)? {
                Arg::Pos(Expr::Int(n)) if *n >= 0 => Some(*n as usize),
                _ => None,
            }
        };
        let str_vec_arg = |idx: usize| -> Option<Vec<Arc<str>>> {
            match args.get(idx)? {
                Arg::Pos(Expr::Array(elems)) => {
                    let mut out = Vec::with_capacity(elems.len());
                    for elem in elems {
                        match elem {
                            crate::ast::ArrayElem::Expr(Expr::Str(s)) => {
                                out.push(Arc::from(s.as_str()));
                            }
                            _ => return None,
                        }
                    }
                    Some(out)
                }
                _ => None,
            }
        };

        match (method, args.len()) {
            (
                BuiltinMethod::Keys
                | BuiltinMethod::Values
                | BuiltinMethod::Entries
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
                | BuiltinMethod::Schema,
                0,
            ) => Some(Self::new(method, BuiltinArgs::None)),
            (
                BuiltinMethod::GetPath
                | BuiltinMethod::HasPath
                | BuiltinMethod::Has
                | BuiltinMethod::DelPath,
                1,
            ) => Some(Self::new(method, BuiltinArgs::Str(str_arg(0)?))),
            (
                BuiltinMethod::StartsWith
                | BuiltinMethod::EndsWith
                | BuiltinMethod::StripPrefix
                | BuiltinMethod::StripSuffix
                | BuiltinMethod::Matches
                | BuiltinMethod::IndexOf
                | BuiltinMethod::LastIndexOf
                | BuiltinMethod::Scan
                | BuiltinMethod::ReMatch
                | BuiltinMethod::ReMatchFirst
                | BuiltinMethod::ReMatchAll
                | BuiltinMethod::ReCaptures
                | BuiltinMethod::ReCapturesAll
                | BuiltinMethod::ReSplit,
                1,
            ) => Some(Self::new(method, BuiltinArgs::Str(str_arg(0)?))),
            (BuiltinMethod::ReReplace | BuiltinMethod::ReReplaceAll, 2) => Some(Self::new(
                method,
                BuiltinArgs::StrPair {
                    first: str_arg(0)?,
                    second: str_arg(1)?,
                },
            )),
            (BuiltinMethod::ContainsAny | BuiltinMethod::ContainsAll, 1) => {
                Some(Self::new(method, BuiltinArgs::StrVec(str_vec_arg(0)?)))
            }
            (BuiltinMethod::Repeat, 1) => {
                Some(Self::new(method, BuiltinArgs::Usize(usize_arg(0)?)))
            }
            (BuiltinMethod::Indent, 0) => Some(Self::new(method, BuiltinArgs::Usize(2))),
            (BuiltinMethod::Indent, 1) => {
                Some(Self::new(method, BuiltinArgs::Usize(usize_arg(0)?)))
            }
            (BuiltinMethod::PadLeft | BuiltinMethod::PadRight | BuiltinMethod::Center, 1) => {
                Some(Self::new(
                    method,
                    BuiltinArgs::Pad {
                        width: usize_arg(0)?,
                        fill: ' ',
                    },
                ))
            }
            (BuiltinMethod::PadLeft | BuiltinMethod::PadRight | BuiltinMethod::Center, 2) => {
                let fill = str_arg(1)?.chars().next().unwrap_or(' ');
                Some(Self::new(
                    method,
                    BuiltinArgs::Pad {
                        width: usize_arg(0)?,
                        fill,
                    },
                ))
            }
            _ => None,
        }
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
}

impl BuiltinMethod {
    #[inline]
    pub fn is_pipeline_element_method(self) -> bool {
        matches!(
            self,
            Self::Keys
                | Self::Values
                | Self::Entries
                | Self::Upper
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
                | Self::Lines
                | Self::Words
                | Self::Chars
                | Self::CharsOf
                | Self::Bytes
                | Self::ByteLen
                | Self::IsBlank
                | Self::IsNumeric
                | Self::IsAlpha
                | Self::IsAscii
                | Self::ToNumber
                | Self::ToBool
                | Self::ParseInt
                | Self::ParseFloat
                | Self::ParseBool
                | Self::Type
                | Self::ToString
                | Self::ToJson
                | Self::Schema
                | Self::GetPath
                | Self::HasPath
                | Self::Has
                | Self::DelPath
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
        )
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
        let k: Arc<str> = Arc::from(crate::eval::util::val_to_key(&eval(&item)?).as_str());
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
        let k: Arc<str> = Arc::from(crate::eval::util::val_to_key(&eval(&item)?).as_str());
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
        let k: Arc<str> = Arc::from(crate::eval::util::val_to_key(&eval(&item)?).as_str());
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
        let new_key: Arc<str> = Arc::from(crate::eval::util::val_to_key(&eval(&k)?).as_str());
        out.insert(new_key, v);
    }
    Ok(out)
}

// ── Pure 1:1 kernels (no closure) ───────────────────────────────────
//
// `&Val → Val` shape — these are the canonical bodies for built-ins
// that lifted natively to Pipeline IR `Stage::*` enum variants per
// `lift_native_pattern.md`. Called from BOTH the Pipeline runtime arm
// AND the `eval/builtins.rs` dispatch fn (for nested sub-program
// invocation via `Opcode::CallMethod`).

/// Canonical `.keys()` impl shared by `Stage::Keys` runtime arm and
/// the `.keys` builtin dispatch shim.  Non-object receivers yield an
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
// runtime arm calls these directly via `lifted_apply` dispatch;
// `eval::builtins` dispatch shim calls them via `composed::shims::*`
// (transitional — shim deletes in a later cleanup pass).
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
    map_str_owned(recv, |s| crate::functions::base64_encode(s.as_bytes()))
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
    map_str_owned(recv, |s| crate::functions::split_words_lower(s).join("_"))
}

/// `.kebab_case()` — lower-case words joined by `-`.
#[inline]
pub fn kebab_case_apply(recv: &Val) -> Option<Val> {
    map_str_owned(recv, |s| crate::functions::split_words_lower(s).join("-"))
}

/// `.camel_case()` — first word lower, rest capitalised, no separator.
#[inline]
pub fn camel_case_apply(recv: &Val) -> Option<Val> {
    map_str_owned(recv, |s| {
        let parts = crate::functions::split_words_lower(s);
        let mut out = String::with_capacity(s.len());
        for (i, p) in parts.iter().enumerate() {
            if i == 0 {
                out.push_str(p);
            } else {
                crate::functions::upper_first_into(p, &mut out);
            }
        }
        out
    })
}

/// `.pascal_case()` — every word capitalised, no separator.
#[inline]
pub fn pascal_case_apply(recv: &Val) -> Option<Val> {
    map_str_owned(recv, |s| {
        let parts = crate::functions::split_words_lower(s);
        let mut out = String::with_capacity(s.len());
        for p in parts.iter() {
            crate::functions::upper_first_into(p, &mut out);
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
    map_str_val(recv, |s| match crate::functions::base64_decode(s) {
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

/// `.split(sep)` Stage form — fresh Val::Str segments. (Distinct from
/// `split_apply` above which is the canonical Stage::Split runtime
/// kernel; this variant takes Arc<str> sep — kept for symmetry with
/// the others.)
#[inline]
pub fn split_str_apply(recv: &Val, sep: &str) -> Option<Val> {
    let s = recv.as_str_ref()?;
    let items: Vec<Val> = s.split(sep).map(|p| Val::Str(Arc::from(p))).collect();
    Some(Val::arr(items))
}

/// `.replace(needle, with)` Stage form — single substitution per
/// match (matches `lifted_apply Stage::Replace` semantics — different
/// from the chain-Replace which uses `replace_apply` with a `bool`
/// `all` flag).
#[inline]
pub fn replace_str_apply(recv: &Val, needle: &str, with: &str) -> Option<Val> {
    Some(Val::Str(Arc::from(
        recv.as_str_ref()?.replace(needle, with),
    )))
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

/// `.flatten()` — one level of array flattening.
#[inline]
pub fn flatten_one_apply(recv: &Val) -> Option<Val> {
    if let Val::Arr(outer) = recv {
        let mut out: Vec<Val> = Vec::new();
        for v in outer.iter() {
            match v {
                Val::Arr(inner) => out.extend(inner.iter().cloned()),
                other => out.push(other.clone()),
            }
        }
        Some(Val::arr(out))
    } else {
        None
    }
}

/// `.flatten(depth)` — recursive flatten up to depth levels.
#[inline]
pub fn flatten_depth_apply(recv: &Val, depth: usize) -> Option<Val> {
    if matches!(recv, Val::Arr(_)) {
        Some(crate::eval::util::flatten_val(recv.clone(), depth))
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
    if let Val::Arr(a) = recv {
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        let kept: Vec<Val> = a
            .iter()
            .filter(|v| seen.insert(crate::eval::util::val_to_key(v)))
            .cloned()
            .collect();
        Some(Val::arr(kept))
    } else {
        None
    }
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

/// `.enumerate()` — Arr → Arr<[index, item]>.
#[inline]
pub fn enumerate_apply(recv: &Val) -> Option<Val> {
    if let Val::Arr(a) = recv {
        let pairs: Vec<Val> = a
            .iter()
            .enumerate()
            .map(|(i, v)| Val::arr(vec![Val::Int(i as i64), v.clone()]))
            .collect();
        Some(Val::arr(pairs))
    } else {
        None
    }
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
            other.iter().map(crate::eval::util::val_to_key).collect();
        let kept: Vec<Val> = a
            .iter()
            .filter(|v| other_keys.contains(&crate::eval::util::val_to_key(v)))
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
            a.iter().map(crate::eval::util::val_to_key).collect();
        for v in other {
            if !a_keys.contains(&crate::eval::util::val_to_key(v)) {
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
            other.iter().map(crate::eval::util::val_to_key).collect();
        let kept: Vec<Val> = a
            .iter()
            .filter(|v| !other_keys.contains(&crate::eval::util::val_to_key(v)))
            .cloned()
            .collect();
        Some(Val::arr(kept))
    } else {
        None
    }
}

// ── Phase D batch 5: object family ──────────────────────────────────

/// `.from_pairs()` — Arr<[Str, Val]> → Obj.
#[inline]
pub fn from_pairs_apply(recv: &Val) -> Option<Val> {
    if let Val::Arr(pairs) = recv {
        let mut m: indexmap::IndexMap<Arc<str>, Val> =
            indexmap::IndexMap::with_capacity(pairs.len());
        for p in pairs.iter() {
            if let Val::Arr(kv) = p {
                if kv.len() == 2 {
                    if let Some(k) = kv[0].as_str_ref() {
                        m.insert(Arc::<str>::from(k), kv[1].clone());
                    }
                }
            }
        }
        Some(Val::Obj(Arc::new(m)))
    } else {
        None
    }
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
            other => Arc::<str>::from(crate::eval::util::val_to_key(other).as_str()),
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
    Some(crate::eval::util::deep_merge(recv.clone(), other.clone()))
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

/// `.get_path(path)` — read leaf at dotted/bracket path.
#[inline]
pub fn get_path_apply(recv: &Val, path: &str) -> Option<Val> {
    let segs = crate::eval::func_paths::parse_path_segs(path);
    Some(crate::eval::func_paths::get_path_impl(recv, &segs))
}

/// `.has_path(path)` — Val::Bool, true if path resolves non-null.
#[inline]
pub fn has_path_apply(recv: &Val, path: &str) -> Option<Val> {
    let segs = crate::eval::func_paths::parse_path_segs(path);
    let found = !crate::eval::func_paths::get_path_impl(recv, &segs).is_null();
    Some(Val::Bool(found))
}

/// `.has(key)` — Val::Bool, true if Obj has key.
#[inline]
pub fn has_apply(recv: &Val, key: &str) -> Option<Val> {
    let m = recv.as_object()?;
    Some(Val::Bool(m.contains_key(key)))
}

/// `.del_path(path)` — remove value at dotted path.
#[inline]
pub fn del_path_apply(recv: &Val, path: &str) -> Option<Val> {
    let segs = crate::eval::func_paths::parse_path_segs(path);
    Some(crate::eval::func_paths::del_path_impl(recv.clone(), &segs))
}

/// `.flatten_keys(sep)` — Obj → flat-Obj with `sep`-joined keys.
#[inline]
pub fn flatten_keys_apply(recv: &Val, sep: &str) -> Option<Val> {
    let mut out: indexmap::IndexMap<Arc<str>, Val> = indexmap::IndexMap::new();
    crate::eval::func_paths::flatten_keys_impl("", recv, sep, &mut out);
    Some(Val::obj(out))
}

/// `.unflatten_keys(sep)` — flat-Obj → nested Obj.
#[inline]
pub fn unflatten_keys_apply(recv: &Val, sep: &str) -> Option<Val> {
    if let Val::Obj(m) = recv {
        Some(crate::eval::func_paths::unflatten_keys_impl(m, sep))
    } else {
        None
    }
}

// ── Phase D batch 7: regex family ───────────────────────────────────

/// `.match(pat)` — Bool, regex match anywhere.
#[inline]
pub fn re_match_apply(recv: &Val, pat: &str) -> Option<Val> {
    let s = recv.as_str_ref()?;
    let re = crate::functions::compile_regex(pat).ok()?;
    Some(Val::Bool(re.is_match(s)))
}

/// `.match_first(pat)` — Str of first match or Null.
#[inline]
pub fn re_match_first_apply(recv: &Val, pat: &str) -> Option<Val> {
    let s = recv.as_str_ref()?;
    let re = crate::functions::compile_regex(pat).ok()?;
    Some(
        re.find(s)
            .map(|m| Val::Str(Arc::from(m.as_str())))
            .unwrap_or(Val::Null),
    )
}

/// `.match_all(pat)` — StrVec of all matches.
#[inline]
pub fn re_match_all_apply(recv: &Val, pat: &str) -> Option<Val> {
    let s = recv.as_str_ref()?;
    let re = crate::functions::compile_regex(pat).ok()?;
    let out: Vec<Arc<str>> = re
        .find_iter(s)
        .map(|m| Arc::<str>::from(m.as_str()))
        .collect();
    Some(Val::str_vec(out))
}

/// `.captures(pat)` — Arr of capture groups (group 0 first); Null on miss.
#[inline]
pub fn re_captures_apply(recv: &Val, pat: &str) -> Option<Val> {
    let s = recv.as_str_ref()?;
    let re = crate::functions::compile_regex(pat).ok()?;
    Some(match re.captures(s) {
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
    })
}

/// `.captures_all(pat)` — Arr<Arr> of capture groups for every match.
#[inline]
pub fn re_captures_all_apply(recv: &Val, pat: &str) -> Option<Val> {
    let s = recv.as_str_ref()?;
    let re = crate::functions::compile_regex(pat).ok()?;
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
    Some(Val::arr(all))
}

/// `.replace_re(pat, with)` — single regex replacement.
#[inline]
pub fn re_replace_apply(recv: &Val, pat: &str, with: &str) -> Option<Val> {
    let s = recv.as_str_ref()?;
    let re = crate::functions::compile_regex(pat).ok()?;
    let out = re.replace(s, with);
    Some(Val::Str(Arc::from(out.as_ref())))
}

/// `.replace_all_re(pat, with)` — all regex replacements.
#[inline]
pub fn re_replace_all_apply(recv: &Val, pat: &str, with: &str) -> Option<Val> {
    let s = recv.as_str_ref()?;
    let re = crate::functions::compile_regex(pat).ok()?;
    let out = re.replace_all(s, with);
    Some(Val::Str(Arc::from(out.as_ref())))
}

/// `.split_re(pat)` — regex split into StrVec.
#[inline]
pub fn re_split_apply(recv: &Val, pat: &str) -> Option<Val> {
    let s = recv.as_str_ref()?;
    let re = crate::functions::compile_regex(pat).ok()?;
    let out: Vec<Arc<str>> = re.split(s).map(Arc::<str>::from).collect();
    Some(Val::str_vec(out))
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
        crate::functions::csv_emit(recv, ",").as_str(),
    )))
}

/// `.to_tsv()` — Val → Str (tab sep).
#[inline]
pub fn to_tsv_apply(recv: &Val) -> Option<Val> {
    Some(Val::Str(Arc::from(
        crate::functions::csv_emit(recv, "\t").as_str(),
    )))
}

/// `.to_pairs()` — Obj → Arr<{key, val}> (named-obj form).
#[inline]
pub fn to_pairs_apply(recv: &Val) -> Option<Val> {
    use crate::eval::util::obj2;
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
        crate::eval::util::val_to_string(recv).as_str(),
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

/// Direct dispatcher for pure zero-argument builtins.
///
/// This is the shared fast path used by backend runtimes that already
/// resolved a method name to `BuiltinMethod`. It intentionally excludes
/// lambda methods and arg-bearing methods; those still need backend-specific
/// argument evaluation.
#[inline]
pub fn apply_zero_arg_method(method: BuiltinMethod, recv: &Val) -> Option<Val> {
    macro_rules! apply_or_recv {
        ($f:expr) => {
            return Some($f(recv).unwrap_or_else(|| recv.clone()))
        };
    }

    match method {
        BuiltinMethod::Upper => apply_or_recv!(upper_apply),
        BuiltinMethod::Lower => apply_or_recv!(lower_apply),
        BuiltinMethod::Trim => apply_or_recv!(trim_apply),
        BuiltinMethod::TrimLeft => apply_or_recv!(trim_left_apply),
        BuiltinMethod::TrimRight => apply_or_recv!(trim_right_apply),
        BuiltinMethod::Capitalize => apply_or_recv!(capitalize_apply),
        BuiltinMethod::TitleCase => apply_or_recv!(title_case_apply),
        BuiltinMethod::Dedent => apply_or_recv!(dedent_apply),
        BuiltinMethod::HtmlEscape => apply_or_recv!(html_escape_apply),
        BuiltinMethod::HtmlUnescape => apply_or_recv!(html_unescape_apply),
        BuiltinMethod::UrlEncode => apply_or_recv!(url_encode_apply),
        BuiltinMethod::UrlDecode => apply_or_recv!(url_decode_apply),
        BuiltinMethod::ToBase64 => apply_or_recv!(to_base64_apply),
        BuiltinMethod::FromBase64 => apply_or_recv!(from_base64_apply),
        BuiltinMethod::Lines => apply_or_recv!(lines_apply),
        BuiltinMethod::Words => apply_or_recv!(words_apply),
        BuiltinMethod::Chars => apply_or_recv!(chars_apply),
        BuiltinMethod::ToNumber => apply_or_recv!(to_number_apply),
        BuiltinMethod::ToBool => apply_or_recv!(to_bool_apply),
        BuiltinMethod::Keys => return Some(keys_apply(recv)),
        BuiltinMethod::Values => return Some(values_apply(recv)),
        BuiltinMethod::Entries => return Some(entries_apply(recv)),
        BuiltinMethod::Invert => apply_or_recv!(invert_apply),
        BuiltinMethod::Compact => apply_or_recv!(compact_apply),
        BuiltinMethod::Pairwise => apply_or_recv!(pairwise_apply),
        BuiltinMethod::Type => apply_or_recv!(type_name_apply),
        BuiltinMethod::ToString => apply_or_recv!(to_string_apply),
        BuiltinMethod::ToJson => apply_or_recv!(to_json_apply),
        BuiltinMethod::ToCsv => apply_or_recv!(to_csv_apply),
        BuiltinMethod::ToTsv => apply_or_recv!(to_tsv_apply),
        BuiltinMethod::ToPairs => apply_or_recv!(to_pairs_apply),
        BuiltinMethod::Schema => apply_or_recv!(schema_apply),
        _ => None,
    }
}

/// `.schema()` — Val → schema Obj describing types/required/array shape.
#[inline]
pub fn schema_apply(recv: &Val) -> Option<Val> {
    Some(crate::eval::func_objects::schema_of(recv))
}
