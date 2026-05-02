//! Macro-generated builtin registry facade.
//!
//! This is the first v2.6 adapter layer: `BuiltinMethod` remains the
//! compatibility identity for existing executors, while `BuiltinId` becomes the
//! stable identity new planner/runtime code can carry without depending on the
//! old enum directly.

use crate::builtins::{BuiltinMethod, BuiltinSpec};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct BuiltinId(pub(crate) u16);

#[derive(Debug, Clone, Copy)]
pub(crate) struct BuiltinDescriptor {
    pub(crate) id: BuiltinId,
    pub(crate) method: BuiltinMethod,
    pub(crate) canonical_name: &'static str,
    pub(crate) aliases: &'static [&'static str],
}

impl BuiltinDescriptor {
    #[inline]
    pub(crate) fn spec(self) -> BuiltinSpec {
        self.method.spec()
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
    ToPairs => "to_pairs" ["toPairs"];
    FromPairs => "from_pairs" ["fromPairs"];
    Invert => "invert" [];
    Reverse => "reverse" [];
    Type => "type" [];
    ToString => "to_string" ["toString"];
    ToJson => "to_json" ["toJson"];
    FromJson => "from_json" ["fromJson"];
    Sum => "sum" [];
    Avg => "avg" [];
    Min => "min" [];
    Max => "max" [];
    Count => "count" [];
    Any => "any" ["exists"];
    All => "all" [];
    FindIndex => "find_index" ["findIndex"];
    IndicesWhere => "indices_where" ["indicesWhere"];
    MaxBy => "max_by" ["maxBy"];
    MinBy => "min_by" ["minBy"];
    GroupBy => "group_by" ["groupBy"];
    CountBy => "count_by" ["countBy"];
    IndexBy => "index_by" ["indexBy"];
    GroupShape => "group_shape" ["groupShape"];
    Explode => "explode" [];
    Implode => "implode" [];
    Filter => "filter" [];
    Map => "map" [];
    FlatMap => "flat_map" ["flatMap"];
    Find => "find" [];
    FindAll => "find_all" ["findAll"];
    Sort => "sort" ["sort_by", "sortBy"];
    Unique => "unique" ["distinct"];
    UniqueBy => "unique_by" ["uniqueBy"];
    Collect => "collect" [];
    DeepFind => "deep_find" ["deepFind"];
    DeepShape => "deep_shape" ["deepShape"];
    DeepLike => "deep_like" ["deepLike"];
    Walk => "walk" [];
    WalkPre => "walk_pre" ["walkPre"];
    Rec => "rec" [];
    TracePath => "trace_path" ["tracePath"];
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
    FindFirst => "find_first" ["findFirst"];
    FindOne => "find_one" ["findOne"];
    ApproxCountDistinct => "approx_count_distinct" ["approxCountDistinct"];
    Accumulate => "accumulate" [];
    Partition => "partition" [];
    Zip => "zip" [];
    ZipLongest => "zip_longest" ["zipLongest"];
    Fanout => "fanout" [];
    ZipShape => "zip_shape" ["zipShape"];
    Pick => "pick" [];
    Omit => "omit" [];
    Merge => "merge" [];
    DeepMerge => "deep_merge" ["deepMerge"];
    Defaults => "defaults" [];
    Rename => "rename" [];
    TransformKeys => "transform_keys" ["transformKeys"];
    TransformValues => "transform_values" ["transformValues"];
    FilterKeys => "filter_keys" ["filterKeys"];
    FilterValues => "filter_values" ["filterValues"];
    Pivot => "pivot" [];
    GetPath => "get_path" ["getPath"];
    SetPath => "set_path" ["setPath"];
    DelPath => "del_path" ["delPath"];
    DelPaths => "del_paths" ["delPaths"];
    HasPath => "has_path" ["hasPath"];
    FlattenKeys => "flatten_keys" ["flattenKeys"];
    UnflattenKeys => "unflatten_keys" ["unflattenKeys"];
    ToCsv => "to_csv" ["toCsv"];
    ToTsv => "to_tsv" ["toTsv"];
    Or => "or" [];
    Has => "has" [];
    Missing => "missing" [];
    Includes => "includes" ["contains"];
    Index => "index" [];
    IndicesOf => "indices_of" ["indicesOf"];
    Set => "set" [];
    Update => "update" [];
    Ceil => "ceil" [];
    Floor => "floor" [];
    Round => "round" [];
    Abs => "abs" [];
    RollingSum => "rolling_sum" ["rollingSum"];
    RollingAvg => "rolling_avg" ["rollingAvg"];
    RollingMin => "rolling_min" ["rollingMin"];
    RollingMax => "rolling_max" ["rollingMax"];
    Lag => "lag" [];
    Lead => "lead" [];
    DiffWindow => "diff_window" ["diffWindow"];
    PctChange => "pct_change" ["pctChange"];
    CumMax => "cummax" [];
    CumMin => "cummin" [];
    Zscore => "zscore" [];
    Upper => "upper" [];
    Lower => "lower" [];
    Capitalize => "capitalize" [];
    TitleCase => "title_case" ["titleCase"];
    Trim => "trim" [];
    TrimLeft => "trim_left" ["trimLeft", "lstrip"];
    TrimRight => "trim_right" ["trimRight", "rstrip"];
    SnakeCase => "snake_case" ["snakeCase"];
    KebabCase => "kebab_case" ["kebabCase"];
    CamelCase => "camel_case" ["camelCase"];
    PascalCase => "pascal_case" ["pascalCase"];
    ReverseStr => "reverse_str" ["reverseStr"];
    Lines => "lines" [];
    Words => "words" [];
    Chars => "chars" [];
    CharsOf => "chars_of" ["charsOf"];
    Bytes => "bytes" [];
    ByteLen => "byte_len" ["byteLen"];
    IsBlank => "is_blank" ["isBlank"];
    IsNumeric => "is_numeric" ["isNumeric"];
    IsAlpha => "is_alpha" ["isAlpha"];
    IsAscii => "is_ascii" ["isAscii"];
    ToNumber => "to_number" ["toNumber"];
    ToBool => "to_bool" ["toBool"];
    ParseInt => "parse_int" ["parseInt"];
    ParseFloat => "parse_float" ["parseFloat"];
    ParseBool => "parse_bool" ["parseBool"];
    ToBase64 => "to_base64" ["toBase64"];
    FromBase64 => "from_base64" ["fromBase64"];
    UrlEncode => "url_encode" ["urlEncode"];
    UrlDecode => "url_decode" ["urlDecode"];
    HtmlEscape => "html_escape" ["htmlEscape"];
    HtmlUnescape => "html_unescape" ["htmlUnescape"];
    Repeat => "repeat" ["repeat_str", "repeatStr"];
    PadLeft => "pad_left" ["padLeft"];
    PadRight => "pad_right" ["padRight"];
    Center => "center" [];
    StartsWith => "starts_with" ["startsWith"];
    EndsWith => "ends_with" ["endsWith"];
    IndexOf => "index_of" ["indexOf"];
    LastIndexOf => "last_index_of" ["lastIndexOf"];
    Replace => "replace" [];
    ReplaceAll => "replace_all" ["replaceAll"];
    StripPrefix => "strip_prefix" ["stripPrefix"];
    StripSuffix => "strip_suffix" ["stripSuffix"];
    Slice => "slice" [];
    Split => "split" [];
    Indent => "indent" [];
    Dedent => "dedent" [];
    Matches => "matches" [];
    Scan => "scan" [];
    ReMatch => "re_match" ["reMatch"];
    ReMatchFirst => "match_first" ["matchFirst"];
    ReMatchAll => "match_all" ["matchAll"];
    ReCaptures => "captures" [];
    ReCapturesAll => "captures_all" ["capturesAll"];
    ReSplit => "split_re" ["splitRe"];
    ReReplace => "replace_re" ["replaceRe"];
    ReReplaceAll => "replace_all_re" ["replaceAllRe"];
    ContainsAny => "contains_any" ["containsAny"];
    ContainsAll => "contains_all" ["containsAll"];
    Schema => "schema" [];
    EquiJoin => "equi_join" ["equiJoin"];
    Unknown => "<unknown>" [];
}

#[cfg(test)]
mod tests {
    use super::*;

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
