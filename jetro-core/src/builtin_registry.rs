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
