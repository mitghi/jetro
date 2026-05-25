//! Builtin method catalog and shared algorithm implementations.
//!
//! All three execution backends (VM, pipeline, composed) dispatch here for
//! algorithm bodies. Each builtin exposes two primitives:
//! `*_one(item, eval)` for per-row work and `*_apply(items, eval)` for
//! buffered work. Streaming consumers call `*_one`; barrier consumers call
//! `*_apply`. This module owns the loop and truthy-check logic exactly once.

use crate::data::context::EvalError;
use crate::data::value::Val;
use indexmap::IndexMap;
use std::sync::Arc;

/// Pre-resolved method identifier. Carried by `CompiledCall` and pipeline
/// plan nodes so method dispatch is an O(1) integer match, not a string hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BuiltinMethod {
    /// Returns the number of elements in an array, object, or string.
    Len = 0,
    /// Returns an array of all keys of an object.
    Keys,
    /// Returns an array of all values of an object.
    Values,
    /// Returns `[[key, value], ...]` pairs for each object entry.
    Entries,
    /// Converts an object to `[{key, val}, ...]` form.
    ToPairs,
    /// Inverse of `to_pairs`; reconstructs an object from key/value pairs.
    FromPairs,
    /// Swaps keys and values of an object.
    Invert,
    /// Reverses an array or string.
    Reverse,
    /// Returns a string name for the runtime type of a value.
    Type,
    /// Converts any value to its display string representation.
    ToString,
    /// Serialises a value to a JSON string.
    ToJson,
    /// Parses a JSON string back to a value.
    FromJson,
    /// Lifts the current input source into a row stream.
    Rows,

    /// Sums all numeric elements; accepts an optional projection lambda.
    Sum,
    /// Computes the arithmetic mean; accepts an optional projection lambda.
    Avg,
    /// Returns the minimum numeric element; accepts an optional projection.
    Min,
    /// Returns the maximum numeric element; accepts an optional projection.
    Max,
    /// Counts elements, or truthy results of a predicate lambda.
    Count,
    /// Returns true if any element satisfies the predicate.
    Any,
    /// Returns true only when every element satisfies the predicate.
    All,
    /// Returns the index of the first element satisfying the predicate.
    FindIndex,
    /// Returns all indices whose elements satisfy the predicate.
    IndicesWhere,
    /// Returns the element whose projected key is the greatest.
    MaxBy,
    /// Returns the element whose projected key is the smallest.
    MinBy,
    /// Groups elements into an object keyed by the lambda result.
    GroupBy,
    /// Counts elements per key produced by the lambda.
    CountBy,
    /// Indexes elements into a map keyed by the lambda result (last wins).
    IndexBy,
    /// Groups elements by a key lambda, then applies a shape lambda to each group.
    GroupShape,
    /// Unnests an array field so each nested value becomes its own row.
    Explode,
    /// Inverse of `explode`; collapses rows sharing the same non-field keys.
    Implode,

    /// Keeps only elements for which the predicate is truthy.
    Filter,
    /// Projects each element through the lambda.
    Map,
    /// Maps each element and flattens one level of the resulting arrays.
    FlatMap,
    /// Alias of `filter`; keeps elements matching the predicate.
    Find,
    /// Alias of `filter`; keeps all elements matching the predicate.
    FindAll,
    /// Sorts an array; supports key expressions and comparator lambdas.
    Sort,
    /// Removes duplicate values from an array.
    Unique,
    /// Removes duplicates by comparing the value of a key lambda.
    UniqueBy,
    /// Wraps a scalar in `[scalar]`; passes arrays through unchanged.
    Collect,
    /// DFS pre-order search across the entire value tree.
    DeepFind,
    /// Collects all objects that contain every key in the shape pattern.
    DeepShape,
    /// Collects all objects whose listed keys equal the given literals.
    DeepLike,
    /// Post-order recursive tree transform (bottom-up).
    Walk,
    /// Pre-order recursive tree transform (top-down).
    WalkPre,
    /// Applies a step expression repeatedly until a fixpoint is reached.
    Rec,
    /// Walks the tree collecting `{path, value}` rows for matching nodes.
    TracePath,
    /// Flattens nested arrays up to a given depth (default 1).
    Flatten,
    /// Removes `null` values from an array.
    Compact,
    /// Joins array elements into a string with a separator.
    Join,
    /// Returns the first element, or the first N elements as an array.
    First,
    /// Returns the last element, or the last N elements as an array.
    Last,
    /// Returns the element at a given index (supports negative indexing).
    Nth,
    /// Keeps at most N elements from the front of the array.
    Take,
    /// Drops the first N elements and returns the rest.
    Skip,
    /// Appends an element to the end of an array.
    Append,
    /// Inserts an element at the front of an array.
    Prepend,
    /// Removes occurrences of a value from an array, or items matching a predicate.
    Remove,
    /// Returns elements of the receiver not present in the argument array.
    Diff,
    /// Returns elements present in both arrays.
    Intersect,
    /// Returns the union of two arrays without duplicates.
    Union,
    /// Produces `[{index, value}, ...]` pairs for each element.
    Enumerate,
    /// Returns consecutive overlapping pairs as `[[a, b], ...]`.
    Pairwise,
    /// Slides a window of size N over the array.
    Window,
    /// Splits an array into non-overlapping chunks of size N.
    Chunk,
    /// Keeps elements from the front as long as the predicate holds.
    TakeWhile,
    /// Drops elements from the front while the predicate holds, then keeps the rest.
    DropWhile,
    /// Returns the first element satisfying the predicate, or null.
    FindFirst,
    /// Returns the only element satisfying the predicate, erroring on zero or multiple matches.
    FindOne,
    /// Counts approximate distinct values using a HyperLogLog-style sketch.
    ApproxCountDistinct,
    /// Produces a running accumulation using the lambda.
    Accumulate,
    /// Folds the array to a single value with `fn(acc, row) -> acc`. Equivalent
    /// to `accumulate(init, fn).last()` but avoids the intermediate array.
    Fold,
    /// Splits an array into two arrays: elements that pass and those that fail the predicate.
    Partition,
    /// Zips two arrays element-wise into `[[a0, b0], ...]`.
    Zip,
    /// Like `zip` but pads the shorter array with a fill value.
    ZipLongest,
    /// Applies multiple expressions to the same receiver and collects results.
    Fanout,
    /// Applies named expressions to one value and collects them into an object.
    ZipShape,

    /// Selects a named subset of fields from an object or array of objects.
    Pick,
    /// Removes named fields from an object or array of objects.
    Omit,
    /// Shallow-merges two objects (right wins on collision).
    Merge,
    /// Recursively merges two objects.
    DeepMerge,
    /// Fills in missing or null fields from a defaults object.
    Defaults,
    /// Renames object keys according to a `{old: new}` map.
    Rename,
    /// Maps a lambda over each key, replacing the key with the result.
    TransformKeys,
    /// Maps a lambda over each value, replacing the value with the result.
    TransformValues,
    /// Keeps only the object entries for which the lambda is truthy.
    FilterKeys,
    /// Keeps only the object entries whose values satisfy the lambda.
    FilterValues,
    /// Pivots an array of objects into a nested object or flat map.
    Pivot,

    /// Retrieves a value at a dot-notation path.
    GetPath,
    /// Sets a value at a dot-notation path, returning the modified document.
    SetPath,
    /// Deletes the value at a dot-notation path.
    DelPath,
    /// Deletes values at multiple dot-notation paths.
    DelPaths,
    /// Returns true if a non-null value exists at the given path.
    HasPath,
    /// Flattens a nested object to dot-notation keys with a given separator.
    FlattenKeys,
    /// Reconstructs a nested object from dot-notation flat keys.
    UnflattenKeys,

    /// Serialises an array/object to CSV text.
    ToCsv,
    /// Serialises an array/object to TSV text.
    ToTsv,

    /// Returns the receiver if non-null; otherwise returns the argument.
    Or,
    /// Returns true if the object contains the given key.
    Has,
    /// Returns true if every literal needle is present in the receiver.
    HasAll,
    /// Returns true if the object contains the given key.
    HasKey,
    /// Returns true if a field path is absent or null in the receiver.
    Missing,
    /// Returns true if the array/string/object contains the given item.
    Includes,
    /// Returns the first index of a value in an array, or -1.
    Index,
    /// Returns all indices where a value occurs in an array.
    IndicesOf,
    /// Replaces the receiver with the argument value (chain-write terminal).
    Set,
    /// Mutates the receiver in place using a lambda (chain-write terminal).
    Update,

    /// Rounds up to the nearest integer.
    Ceil,
    /// Rounds down to the nearest integer.
    Floor,
    /// Rounds to the nearest integer.
    Round,
    /// Returns the absolute value.
    Abs,
    /// Computes a rolling sum over a sliding window of size N.
    RollingSum,
    /// Computes a rolling mean over a sliding window of size N.
    RollingAvg,
    /// Computes a rolling minimum over a sliding window of size N.
    RollingMin,
    /// Computes a rolling maximum over a sliding window of size N.
    RollingMax,
    /// Shifts values backward by N positions (fills leading positions with null).
    Lag,
    /// Shifts values forward by N positions (fills trailing positions with null).
    Lead,
    /// Computes element-wise first differences.
    DiffWindow,
    /// Computes element-wise percentage change from the previous value.
    PctChange,
    /// Running maximum up to each position.
    CumMax,
    /// Running minimum up to each position.
    CumMin,
    /// Normalises each element to its z-score relative to the array mean/std.
    Zscore,

    /// Converts a string to all-uppercase.
    Upper,
    /// Converts a string to all-lowercase.
    Lower,
    /// Uppercases the first character and lowercases the rest.
    Capitalize,
    /// Title-cases every word in the string.
    TitleCase,
    /// Strips leading and trailing ASCII whitespace.
    Trim,
    /// Strips leading ASCII whitespace.
    TrimLeft,
    /// Strips trailing ASCII whitespace.
    TrimRight,
    /// Converts a string to `snake_case`.
    SnakeCase,
    /// Converts a string to `kebab-case`.
    KebabCase,
    /// Converts a string to `camelCase`.
    CamelCase,
    /// Converts a string to `PascalCase`.
    PascalCase,
    /// Reverses the characters of a string.
    ReverseStr,
    /// Splits a string on newlines and returns an array of lines.
    Lines,
    /// Splits a string on whitespace and returns an array of words.
    Words,
    /// Returns each Unicode grapheme cluster as a single-element string.
    Chars,
    /// Returns each Unicode code point as a UTF-8 encoded string.
    CharsOf,
    /// Returns each byte of the string as an integer.
    Bytes,
    /// Returns the byte length (not char count) of a string.
    ByteLen,
    /// Returns true if the string is empty or contains only whitespace.
    IsBlank,
    /// Returns true if the string consists entirely of ASCII digits.
    IsNumeric,
    /// Returns true if the string consists entirely of alphabetic characters.
    IsAlpha,
    /// Returns true if the string is valid ASCII.
    IsAscii,
    /// Parses a string as an integer or float; returns null on failure.
    ToNumber,
    /// Parses `"true"` / `"false"` to a boolean; returns null otherwise.
    ToBool,
    /// Parses the string as a base-10 integer; returns null on failure.
    ParseInt,
    /// Parses the string as a float; returns null on failure.
    ParseFloat,
    /// Parses common truthy/falsy string representations to a boolean.
    ParseBool,
    /// Encodes a string as standard Base64.
    ToBase64,
    /// Decodes a Base64-encoded string.
    FromBase64,
    /// Percent-encodes a string for use in a URL.
    UrlEncode,
    /// Decodes a percent-encoded URL string.
    UrlDecode,
    /// Escapes `<`, `>`, `&`, `"`, `'` to their HTML entities.
    HtmlEscape,
    /// Converts HTML entities back to their literal characters.
    HtmlUnescape,
    /// Repeats the string N times.
    Repeat,
    /// Left-pads the string to the given width with a fill character.
    PadLeft,
    /// Right-pads the string to the given width with a fill character.
    PadRight,
    /// Centers the string within the given width using a fill character.
    Center,
    /// Returns true if the string starts with the given prefix.
    StartsWith,
    /// Returns true if the string ends with the given suffix.
    EndsWith,
    /// Returns the char index of the first occurrence, or -1.
    IndexOf,
    /// Returns the char index of the last occurrence, or -1.
    LastIndexOf,
    /// Replaces the first occurrence of `needle` with `replacement`.
    Replace,
    /// Replaces all occurrences of `needle` with `replacement`.
    ReplaceAll,
    /// Strips the given prefix if present; returns the receiver unchanged otherwise.
    StripPrefix,
    /// Strips the given suffix if present; returns the receiver unchanged otherwise.
    StripSuffix,
    /// Returns a substring by character indices (supports negative indexing).
    Slice,
    /// Splits a string on a separator and returns an array of parts.
    Split,
    /// Prepends N spaces to every line of a string.
    Indent,
    /// Removes the common leading whitespace from every line.
    Dedent,
    /// Returns true if the string contains the given substring.
    Matches,
    /// Returns an array of every non-overlapping occurrence of a pattern.
    Scan,
    /// Returns true if the regex matches the string.
    ReMatch,
    /// Returns the first regex match as a string, or null.
    ReMatchFirst,
    /// Returns all non-overlapping regex matches as an array of strings.
    ReMatchAll,
    /// Returns capture groups of the first regex match as an array, or null.
    ReCaptures,
    /// Returns all capture groups for every match as an array of arrays.
    ReCapturesAll,
    /// Splits a string on a regex pattern.
    ReSplit,
    /// Replaces the first regex match with a replacement string.
    ReReplace,
    /// Replaces all regex matches with a replacement string.
    ReReplaceAll,
    /// Returns true if the string contains any of the given substrings.
    ContainsAny,
    /// Returns true if the string contains all of the given substrings.
    ContainsAll,
    /// Infers a structural schema description from the value.
    Schema,

    /// Performs an inner equi-join of two arrays of objects on matching key fields.
    EquiJoin,

    /// Sentinel returned by `from_name` when the method string is unrecognised.
    Unknown,
}

/// Expands `$macro!(...)` once per `BuiltinMethod` variant — the single source of truth for
/// "all builtin methods" used by name lookup, registry exports, and any future cross-cutting
/// per-method generation. Variant names match the corresponding `defs::*` struct names.
#[macro_export]
macro_rules! for_each_builtin {
    ($macro:ident) => {
        $macro!(
            Abs,
            Accumulate,
            All,
            Any,
            Append,
            ApproxCountDistinct,
            Avg,
            ByteLen,
            Bytes,
            CamelCase,
            Capitalize,
            Ceil,
            Center,
            Chars,
            CharsOf,
            Chunk,
            Collect,
            Compact,
            ContainsAll,
            ContainsAny,
            Count,
            CountBy,
            CumMax,
            CumMin,
            Dedent,
            DeepFind,
            DeepLike,
            DeepMerge,
            DeepShape,
            Defaults,
            DelPath,
            DelPaths,
            Diff,
            DiffWindow,
            DropWhile,
            EndsWith,
            Entries,
            Enumerate,
            EquiJoin,
            Explode,
            Fanout,
            Filter,
            FilterKeys,
            FilterValues,
            Find,
            FindAll,
            FindFirst,
            FindIndex,
            FindOne,
            First,
            FlatMap,
            Flatten,
            FlattenKeys,
            Floor,
            Fold,
            FromBase64,
            FromJson,
            FromPairs,
            GetPath,
            GroupBy,
            GroupShape,
            Has,
            HasAll,
            HasKey,
            HasPath,
            HtmlEscape,
            HtmlUnescape,
            Implode,
            Includes,
            Indent,
            Index,
            IndexBy,
            IndexOf,
            IndicesOf,
            IndicesWhere,
            Intersect,
            Invert,
            IsAlpha,
            IsAscii,
            IsBlank,
            IsNumeric,
            Join,
            KebabCase,
            Keys,
            Lag,
            Last,
            LastIndexOf,
            Lead,
            Len,
            Lines,
            Lower,
            Map,
            Matches,
            Max,
            MaxBy,
            Merge,
            Min,
            MinBy,
            Missing,
            Nth,
            Omit,
            Or,
            PadLeft,
            PadRight,
            Pairwise,
            ParseBool,
            ParseFloat,
            ParseInt,
            Partition,
            PascalCase,
            PctChange,
            Pick,
            Pivot,
            Prepend,
            Rec,
            ReCaptures,
            ReCapturesAll,
            ReMatch,
            ReMatchAll,
            ReMatchFirst,
            Remove,
            Rename,
            Repeat,
            Replace,
            ReplaceAll,
            ReReplace,
            ReReplaceAll,
            ReSplit,
            Reverse,
            ReverseStr,
            RollingAvg,
            RollingMax,
            RollingMin,
            RollingSum,
            Round,
            Rows,
            Scan,
            Schema,
            Set,
            SetPath,
            Skip,
            Slice,
            SnakeCase,
            Sort,
            Split,
            StartsWith,
            StripPrefix,
            StripSuffix,
            Sum,
            Take,
            TakeWhile,
            TitleCase,
            ToBase64,
            ToBool,
            ToCsv,
            ToJson,
            ToNumber,
            ToPairs,
            ToString,
            ToTsv,
            TracePath,
            TransformKeys,
            TransformValues,
            Trim,
            TrimLeft,
            TrimRight,
            Type,
            UnflattenKeys,
            Union,
            Unique,
            UniqueBy,
            Unknown,
            Update,
            Upper,
            UrlDecode,
            UrlEncode,
            Values,
            Walk,
            WalkPre,
            Window,
            Words,
            Zip,
            ZipLongest,
            ZipShape,
            Zscore
        )
    };
}

impl BuiltinMethod {
    /// Resolves a method name string to the corresponding `BuiltinMethod` variant.
    /// Returns [`BuiltinMethod::Unknown`] when the name is not registered.
    pub fn from_name(name: &str) -> Self {
        crate::builtins::registry::by_name(name)
            .and_then(|id| id.method())
            .unwrap_or(Self::Unknown)
    }

    /// Returns true when the method requires a lambda expression as its first argument.
    /// The pipeline planner uses this to distinguish element vs. expression stages.
    pub(crate) fn is_lambda_method(self) -> bool {
        registry::accepts_lambda_arg(registry::BuiltinId::from_method(self))
    }
}

/// Statically-typed argument payload stored inside a [`BuiltinCall`].
/// Each variant corresponds to the argument signature of a group of builtins,
/// enabling argument decoding without heap allocation at call time.
#[derive(Debug, Clone)]
pub enum BuiltinArgs {
    /// No arguments.
    None,
    /// A single string argument (field name, separator, pattern, etc.).
    Str(Arc<str>),
    /// A pre-parsed dot/bracket path used by hot path helpers.
    Path(Arc<[PathSeg]>),
    /// Multiple pre-parsed dot/bracket paths used by bulk path helpers.
    PathList(Vec<Arc<[PathSeg]>>),
    /// A pre-parsed dot/bracket path plus an owned value payload.
    PathVal { path: Arc<[PathSeg]>, value: Val },
    /// Two string arguments (needle + replacement, pattern + replacement).
    StrPair { first: Arc<str>, second: Arc<str> },
    /// A list of string arguments (field list for `pick`, `omit`, etc.).
    StrVec(Vec<Arc<str>>),
    /// A single signed-integer argument (index, count).
    I64(i64),
    /// A primary integer plus an optional second integer (start + optional end for `slice`).
    I64Opt { first: i64, second: Option<i64> },
    /// A single unsigned-integer argument (window size, chunk size, etc.).
    Usize(usize),
    /// A single pre-evaluated `Val` argument.
    Val(Val),
    /// A list of pre-evaluated `Val` arguments (`diff`, `intersect`, `union`).
    ValVec(Vec<Val>),
    /// Padding width and fill character (`pad_left`, `pad_right`, `center`).
    Pad { width: usize, fill: char },
}

/// A pre-compiled builtin call ready for stateless execution.
/// Stored in pipeline plan nodes and the `CompiledCall` opcode payload.
#[derive(Debug, Clone)]
pub struct BuiltinCall {
    /// Which builtin to invoke.
    pub method: BuiltinMethod,
    /// The decoded static arguments for this call.
    pub args: BuiltinArgs,
}

/// Internal helper that decodes static (non-lambda) arguments for [`BuiltinCall::from_static_args`].
/// Wraps the `eval_arg` and `ident_arg` closures with typed accessor methods.
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
    /// Evaluates the argument at `idx`, returning an error if it is absent.
    fn val(&mut self, idx: usize) -> Result<Val, EvalError> {
        (self.eval_arg)(idx)?.ok_or_else(|| EvalError(format!("{}: missing argument", self.name)))
    }

    /// Evaluates the argument at `idx` as a string, accepting bare identifiers.
    fn str(&mut self, idx: usize) -> Result<Arc<str>, EvalError> {
        if let Some(value) = (self.ident_arg)(idx) {
            return Ok(value);
        }
        match self.val(idx)? {
            Val::Str(s) => Ok(s),
            other => Ok(Arc::from(crate::util::val_to_string(&other).as_str())),
        }
    }

    /// Returns `Some(prefix)` only when the argument is a string-typed
    /// value, leaving non-string arguments untouched. Used to disambiguate
    /// overloaded scalar-arg builtins (e.g. `indent(2)` vs `indent("> ")`).
    fn str_lit(&mut self, idx: usize) -> Option<Arc<str>> {
        match (self.eval_arg)(idx).ok().flatten()? {
            Val::Str(s) => Some(s),
            Val::StrSlice(r) => Some(r.to_arc()),
            _ => None,
        }
    }

    /// Evaluates the argument at `idx` as a signed 64-bit integer.
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

    /// Evaluates the argument at `idx` as a `usize` (clamped to 0 from below).
    fn usize(&mut self, idx: usize) -> Result<usize, EvalError> {
        Ok(self.i64(idx)?.max(0) as usize)
    }

    /// Evaluates the argument at `idx` as a `Vec<Val>`, failing if not an array.
    fn vec(&mut self, idx: usize) -> Result<Vec<Val>, EvalError> {
        self.val(idx).and_then(|value| {
            value
                .into_vec()
                .ok_or_else(|| EvalError(format!("{}: expected array arg", self.name)))
        })
    }

    /// Evaluates the argument at `idx` as a vector of strings.
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

    /// Evaluates the argument at `idx` as a single character for padding operations.
    /// Defaults to `' '` when the argument index is out of range.
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

/// Capability and cost descriptor for a single builtin method.
/// The pipeline planner reads these fields to decide how to lower each stage.
#[derive(Debug, Clone, Copy)]
pub struct BuiltinSpec {
    /// Whether the method is pure (no side effects); impure methods are never fused.
    pub pure: bool,
    /// Broad classification used for planning and display.
    pub category: BuiltinCategory,
    /// Input-to-output row-count relationship.
    pub cardinality: BuiltinCardinality,
    /// Whether the builtin may be used as an indexed projection (e.g. inside `map`).
    pub can_indexed: bool,
    /// Whether the builtin has a native view-path implementation.
    pub view_native: bool,
    /// Whether the builtin can execute directly on a `JsonView` without materialising.
    pub view_scalar: bool,
    /// Concrete borrowed-view scalar dispatch family, if any.
    pub view_scalar_op: Option<BuiltinViewScalarOp>,
    /// View-native whole-value projection operation, if any.
    pub view_value_projection: Option<BuiltinViewValueProjection>,
    /// View-native object/path projection operation, if any.
    pub view_object_projection: Option<BuiltinViewObjectProjection>,
    /// View-native string expansion operation, if any.
    pub view_string_expand: Option<BuiltinViewStringExpand>,
    /// Raw-byte JSON scalar operation, if any.
    pub raw_json_scalar: Option<BuiltinRawJsonScalar>,
    /// Object-lambda operation behavior, if any.
    pub object_lambda: Option<BuiltinObjectLambda>,
    /// Two-string-argument pipeline stage behavior, if any.
    pub string_pair_stage: Option<BuiltinStringPairStage>,
    /// Nullary pipeline stage behavior, if any.
    pub nullary_stage: Option<BuiltinNullaryStage>,
    /// Expression-argument pipeline stage behavior, if any.
    pub expr_stage: Option<BuiltinExprStage>,
    /// Payload-demand behavior for expression-bearing stages, if any.
    pub expr_payload: Option<BuiltinExprPayload>,
    /// Logical planner node shape, if this builtin participates in logical lowering.
    pub logical_shape: Option<BuiltinLogicalShape>,
    /// Source-level `$.rows()` stream operation behavior, if legal in stream position.
    pub row_stream_op: Option<BuiltinRowStreamOp>,
    /// View-stage lowering target, if the builtin maps to one of the view stages.
    pub view_stage: Option<BuiltinViewStage>,
    /// Sink (terminal aggregation) descriptor, present for reducing builtins.
    pub sink: Option<BuiltinSinkSpec>,
    /// Keyed reducer kind (group/count/index), used for grouped output planning.
    pub keyed_reducer: Option<BuiltinKeyedReducer>,
    /// Numeric reducer kind, used by the numeric sink path.
    pub numeric_reducer: Option<BuiltinNumericReducer>,
    /// Arg-extreme sink kind, used by `max_by` / `min_by` planning.
    pub arg_extreme_sink: Option<BuiltinArgExtremeSink>,
    /// Predicate terminal sink kind, used by predicate reducers.
    pub predicate_sink: Option<BuiltinPredicateSink>,
    /// Membership terminal sink kind, used by target-value reducers.
    pub membership_sink: Option<BuiltinMembershipSink>,
    /// Array selector kind, used for direct element projection.
    pub array_selector: Option<BuiltinArraySelector>,
    /// Selection rewrite for an ordered stage followed by a terminal pick or index.
    pub selection_rewrite: Option<BuiltinSelectionRewrite>,
    /// How adjacent stages of the same kind can be merged (e.g. `take(3).take(2)` → `take(2)`).
    pub stage_merge: Option<BuiltinStageMerge>,
    /// Algebraic cancellation rule (e.g. `reverse().reverse()` = identity).
    pub cancellation: Option<BuiltinCancellation>,
    /// Whether applying the builtin twice is equivalent to applying it once.
    pub idempotent: bool,
    /// Whether the builtin accepts a lambda/expression argument at runtime.
    pub accepts_lambda_arg: bool,
    /// Whether the builtin changes only row order, not row membership or values.
    pub order_only: bool,
    /// Runtime hook implementation target when public builtins share executor behavior.
    pub runtime_hook: Option<BuiltinRuntimeHook>,
    /// Whether receiver-mode VM execution can stop after a downstream output cap.
    pub output_cap_receiver: bool,
    /// Columnar stage kind for backends that work on typed column vectors.
    pub columnar_stage: Option<BuiltinColumnarStage>,
    /// Structural index backend hint (deep search variants).
    pub structural: Option<BuiltinStructural>,
    /// Relative cost used by the planner's heuristic optimizer.
    pub cost: f64,
    /// Demand-propagation law for pipeline planning (default: `Identity`).
    pub demand_law: BuiltinDemandLaw,
    /// Materialisation policy (default: `Streaming`).
    pub materialization: BuiltinPipelineMaterialization,
    /// Semantic streaming boundary class. Executors use this to explain and
    /// minimize ownership boundaries without rediscovering builtin behavior.
    pub streaming_boundary: BuiltinStreamingBoundary,
    /// Cardinality/cost shape annotation for the pipeline cost estimator.
    pub pipeline_shape: Option<BuiltinPipelineShape>,
    /// How this builtin affects element ordering in the pipeline.
    pub order_effect: Option<BuiltinPipelineOrderEffect>,
    /// Physical stage lowering strategy, if registered.
    pub lowering: Option<BuiltinPipelineLowering>,
    /// Whether the builtin is element-wise vectorisable.
    pub is_element: bool,
    /// Opt out of the path-receiver scalar-unwrap rewrite. When `true`, the
    /// planner does **not** lower `$.path.method()` directly to `apply_one`,
    /// even if `category` and `cardinality` would otherwise allow it. Use for
    /// methods whose pipeline-streaming behavior is the desired semantic on
    /// path receivers (e.g. per-element serialization).
    pub never_unwrap: bool,
    /// Marks this method as a source-lifting stream boundary. Such methods are
    /// planned by source/stream planners instead of normal row-local dispatch.
    pub stream_source: bool,
}

/// How a builtin transforms downstream demand into the demand it places on
/// its upstream source. Unknown builtins default to `Identity`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinDemandLaw {
    /// Pass downstream demand through unchanged (e.g. purely transforming builtins).
    Identity,
    /// Like filter: must scan until `n` outputs are produced, so converts `FirstInput(n)` to `UntilOutput(n)`.
    FilterLike,
    /// Like `take_while`: stops at the first predicate failure, so `UntilOutput(n)` becomes `FirstInput(n)`.
    TakeWhile,
    /// Like `drop_while`: prefix predicate barrier; safe upstream demand is a full ordered scan.
    DropWhile,
    /// Like `unique`/`unique_by`: scan until enough distinct outputs are observed.
    UniqueLike,
    /// Like map: the output count equals the input count; passes demand through but requires whole values.
    MapLike,
    /// Like a scalar predicate projection: one output per input, but only predicate-relevant
    /// payload is needed from each input value.
    PredicateMapLike,
    /// Like scalar `slice`: one-to-one and order-preserving, but consumes the whole input value.
    Slice,
    /// Like `flat_map`: output count is unbounded relative to input, so always requests all input.
    FlatMapLike,
    /// Cap the upstream pull to the provided count argument.
    Take,
    /// Shift the upstream pull window by the provided count argument.
    Skip,
    /// Fixed-size chunking; bounded output demand maps to a bounded input prefix.
    Chunk,
    /// Sliding window; bounded output demand maps to a bounded input prefix.
    Window,
    /// Adjacent pair demand; equivalent to a fixed width-2 sliding window.
    Pairwise,
    /// Only the first element is needed; translates any downstream demand to `FirstInput(1)`.
    First,
    /// The last element is needed; requires all ordered input.
    Last,
    /// A specific positional element is needed.
    Nth,
    /// Only a count is needed; requires all inputs but no value payloads.
    Count,
    /// A numeric aggregate (sum/min/max/avg); requires all inputs with numeric-only payload.
    NumericReducer,
    /// Key-only aggregate such as `count_by`; requires all inputs and key evaluation.
    KeyOnlyReducer,
    /// Row-retaining keyed aggregate; requires all full input rows.
    RowKeyedReducer,
    /// A full-input ordering barrier; downstream limits can choose strategy, but source scan remains all input.
    OrderBarrier,
    /// Reverses one-to-one output order, swapping first/last positional demand.
    Reverse,
}

/// Concrete borrowed-view scalar execution family for a builtin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinViewScalarOp {
    /// Return the length of an array, object, or string view.
    Len,
    /// Return the JSON type name from the view tag.
    TypeName,
    /// String receiver, no static argument.
    StringNoArg,
    /// String receiver parsed as an integer with optional radix.
    ParseInt,
    /// Numeric receiver, no static argument.
    NumericNoArg,
    /// String receiver with a single string argument.
    StringArg,
    /// String containment using a literal string target value.
    StringContainsArg,
    /// String receiver with a static string-vector argument.
    StringVecArg,
}

/// View-native projection that needs the whole value but can still avoid
/// materialising the receiver by traversing borrowed child views.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinViewValueProjection {
    /// Convert `[key, value]` / `{key, val}` pairs into an object.
    FromPairs,
    /// Parse a JSON string into a value.
    FromJson,
    /// Bucket array rows by object key shape.
    GroupShape,
    /// Group row objects by all fields except one, collecting that field into arrays.
    Implode,
    /// Swap object keys and values, coercing values to object keys.
    Invert,
    /// Recursively merge an owned value argument into the receiver value.
    DeepMerge,
    /// Set a nested dot/bracket path to an owned value.
    SetPath,
    /// Delete a nested dot/bracket path from an object or array value.
    DelPath,
    /// Delete multiple nested dot/bracket paths from an object or array value.
    DelPaths,
    /// Flatten nested object keys into a separator-joined object.
    FlattenKeys,
    /// Rebuild nested object keys from a separator-joined object.
    UnflattenKeys,
    /// Shallow-merge an owned object argument into the receiver object.
    Merge,
    /// Fill missing or null receiver object keys from an owned defaults object.
    Defaults,
    /// Rename receiver object keys from an owned `{old: new}` object.
    Rename,
    /// Pivot array rows using literal field names.
    Pivot,
    /// Convert to camelCase.
    CamelCase,
    /// Uppercase first character and lowercase the rest.
    Capitalize,
    /// Decode Base64 text as UTF-8.
    FromBase64,
    /// Remove common leading indentation.
    Dedent,
    /// Unescape HTML entities.
    HtmlUnescape,
    /// Escape HTML-sensitive characters.
    HtmlEscape,
    /// Broad receiver membership/containment check.
    Includes,
    /// Prepend a prefix to each line.
    Indent,
    /// Return a default value when the receiver is null/missing.
    Or,
    /// Center-pad a string value.
    Center,
    /// Left-pad a string value.
    PadLeft,
    /// Right-pad a string value.
    PadRight,
    /// Replace the first matching substring in a string value.
    Replace,
    /// Replace all matching substrings in a string value.
    ReplaceAll,
    /// Repeat a string value N times.
    Repeat,
    /// Reverse a string value by Unicode scalar values.
    ReverseStr,
    /// Slice a string value by character offsets.
    Slice,
    /// Convert to snake_case.
    SnakeCase,
    /// Remove a matching string prefix.
    StripPrefix,
    /// Remove a matching string suffix.
    StripSuffix,
    /// Coerce the value to Jetro's human-readable string form.
    ToString,
    /// Serialize the value to compact JSON text.
    ToJson,
    /// Serialize the value to CSV text.
    ToCsv,
    /// Serialize the value to TSV text.
    ToTsv,
    /// Convert an object of parallel arrays into row objects.
    ZipShape,
    /// Title-case whitespace-delimited words.
    TitleCase,
    /// Encode string bytes as Base64 text.
    ToBase64,
    /// Convert to kebab-case.
    KebabCase,
    /// Convert to PascalCase.
    PascalCase,
    /// Percent-decode URL text.
    UrlDecode,
    /// Percent-encode URL text.
    UrlEncode,
}

impl BuiltinViewValueProjection {
    /// Demand law implied by this view-native whole-value operation.
    #[inline]
    pub(crate) const fn demand_law(self) -> BuiltinDemandLaw {
        match self {
            BuiltinViewValueProjection::Slice => BuiltinDemandLaw::Slice,
            BuiltinViewValueProjection::GroupShape | BuiltinViewValueProjection::ZipShape => {
                BuiltinDemandLaw::OrderBarrier
            }
            BuiltinViewValueProjection::CamelCase
            | BuiltinViewValueProjection::Capitalize
            | BuiltinViewValueProjection::Center
            | BuiltinViewValueProjection::Dedent
            | BuiltinViewValueProjection::DelPath
            | BuiltinViewValueProjection::DelPaths
            | BuiltinViewValueProjection::DeepMerge
            | BuiltinViewValueProjection::Defaults
            | BuiltinViewValueProjection::FlattenKeys
            | BuiltinViewValueProjection::FromJson
            | BuiltinViewValueProjection::FromPairs
            | BuiltinViewValueProjection::FromBase64
            | BuiltinViewValueProjection::HtmlEscape
            | BuiltinViewValueProjection::HtmlUnescape
            | BuiltinViewValueProjection::Implode
            | BuiltinViewValueProjection::Invert
            | BuiltinViewValueProjection::Merge
            | BuiltinViewValueProjection::Indent
            | BuiltinViewValueProjection::KebabCase
            | BuiltinViewValueProjection::Or
            | BuiltinViewValueProjection::PadLeft
            | BuiltinViewValueProjection::PadRight
            | BuiltinViewValueProjection::PascalCase
            | BuiltinViewValueProjection::Pivot
            | BuiltinViewValueProjection::Replace
            | BuiltinViewValueProjection::ReplaceAll
            | BuiltinViewValueProjection::Rename
            | BuiltinViewValueProjection::Repeat
            | BuiltinViewValueProjection::ReverseStr
            | BuiltinViewValueProjection::SetPath
            | BuiltinViewValueProjection::SnakeCase
            | BuiltinViewValueProjection::TitleCase
            | BuiltinViewValueProjection::ToCsv
            | BuiltinViewValueProjection::ToString
            | BuiltinViewValueProjection::ToJson
            | BuiltinViewValueProjection::ToTsv
            | BuiltinViewValueProjection::ToBase64
            | BuiltinViewValueProjection::UnflattenKeys
            | BuiltinViewValueProjection::StripPrefix
            | BuiltinViewValueProjection::StripSuffix
            | BuiltinViewValueProjection::UrlDecode
            | BuiltinViewValueProjection::UrlEncode => BuiltinDemandLaw::MapLike,
            BuiltinViewValueProjection::Includes => BuiltinDemandLaw::PredicateMapLike,
        }
    }
}

/// View-native object/path projection operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinViewObjectProjection {
    /// Broad membership check (`has`).
    Has,
    /// All listed keys/items must be present.
    HasAll,
    /// Object-key-only existence check.
    HasKey,
    /// Object key/path is missing or null.
    Missing,
    /// Return a nested path view.
    GetPath,
    /// Test whether a nested path exists.
    HasPath,
    /// Return object keys.
    Keys,
    /// Return object values.
    Values,
    /// Return object entries.
    Entries,
    /// Return object entries as `{key, val}` objects.
    ToPairs,
    /// Keep selected keys.
    Pick,
    /// Drop selected keys.
    Omit,
}

/// View-native string expansion operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinViewStringExpand {
    /// Split on a supplied separator.
    Split,
    /// Split on line boundaries.
    Lines,
    /// Split on whitespace.
    Words,
    /// Emit Unicode scalar values as strings.
    Chars,
    /// Emit Unicode scalar values re-encoded as UTF-8 strings.
    CharsOf,
    /// Emit UTF-8 bytes as integers.
    Bytes,
}

impl BuiltinViewObjectProjection {
    /// Demand law implied by this view-native object/path operation.
    #[inline]
    pub(crate) const fn demand_law(self) -> BuiltinDemandLaw {
        match self {
            Self::Has | Self::HasAll | Self::HasKey | Self::Missing | Self::HasPath => {
                BuiltinDemandLaw::PredicateMapLike
            }
            Self::GetPath
            | Self::Keys
            | Self::Values
            | Self::Entries
            | Self::ToPairs
            | Self::Pick
            | Self::Omit => BuiltinDemandLaw::MapLike,
        }
    }

    /// Whether the projection enumerates object keys, values, or entries.
    #[inline]
    pub(crate) const fn is_item_projection(self) -> bool {
        matches!(
            self,
            Self::Keys | Self::Values | Self::Entries | Self::ToPairs
        )
    }

    /// Whether applying this projection yields an owned value instead of a borrowed child view.
    #[inline]
    pub(crate) const fn returns_owned(self) -> bool {
        !matches!(self, Self::GetPath)
    }
}

/// Raw-byte JSON scalar operation that can be executed before building a
/// `JsonView` or materialising a `Val`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinRawJsonScalar {
    /// Compute string/array/object length directly from raw JSON.
    Len,
    /// ASCII-only string uppercasing can be written without allocation.
    AsciiUpper,
    /// ASCII-only string lowercasing can be written without allocation.
    AsciiLower,
}

impl BuiltinRawJsonScalar {
    /// Demand law implied by this raw-byte scalar operation.
    #[inline]
    pub(crate) const fn demand_law(self) -> BuiltinDemandLaw {
        match self {
            Self::Len => BuiltinDemandLaw::Count,
            Self::AsciiUpper | Self::AsciiLower => BuiltinDemandLaw::MapLike,
        }
    }

    /// Whether this raw operation writes a transformed JSON string.
    #[inline]
    pub(crate) const fn writes_string(self) -> bool {
        matches!(self, Self::AsciiUpper | Self::AsciiLower)
    }

    /// Whether this raw operation writes the length of the current JSON view.
    #[inline]
    pub(crate) const fn writes_view_len(self) -> bool {
        matches!(self, Self::Len)
    }
}

/// Object-lambda operation behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinObjectLambda {
    /// Map object keys through the lambda.
    TransformKeys,
    /// Map object values through the lambda.
    TransformValues,
    /// Keep entries whose key satisfies the predicate.
    FilterKeys,
    /// Keep entries whose value satisfies the predicate.
    FilterValues,
}

impl BuiltinObjectLambda {
    /// Demand law implied by object-lambda stages.
    #[inline]
    pub(crate) const fn demand_law(self) -> BuiltinDemandLaw {
        BuiltinDemandLaw::MapLike
    }

    /// Expression payload lane used by this object-lambda operation.
    #[inline]
    pub(crate) const fn expr_payload(self) -> BuiltinExprPayload {
        match self {
            Self::FilterKeys => BuiltinExprPayload::PredicateScan,
            Self::TransformKeys | Self::TransformValues | Self::FilterValues => {
                BuiltinExprPayload::Projection
            }
        }
    }
}

/// Concrete pipeline stage behavior for builtins with two string arguments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinStringPairStage {
    /// String replacement; `all` selects first-hit vs all-hit replacement.
    Replace {
        /// Replace every occurrence when true, otherwise only the first.
        all: bool,
    },
}

/// Concrete pipeline stage shape for nullary stage builtins.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinNullaryStage {
    /// Reverse stage with cancellation metadata.
    Reverse,
    /// Deduplicate by full row value.
    Unique,
    /// Generic no-argument element builtin.
    Element,
}

/// Concrete pipeline stage shape for builtins with one expression argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinExprStage {
    /// Predicate filter stage.
    Filter,
    /// One-to-one map stage.
    Map,
    /// Expanding flat-map stage.
    FlatMap,
    /// Deduplicate by key stage.
    UniqueBy,
    /// Generic expression-bearing builtin stage.
    ExprBuiltin,
}

impl BuiltinExprStage {
    /// View-stage shape used when an expression-stage builtin lowers through
    /// another concrete streaming stage, such as terminal `find_first`.
    #[inline]
    pub fn view_stage(self) -> Option<BuiltinViewStage> {
        match self {
            Self::Filter => Some(BuiltinViewStage::Filter),
            Self::Map => Some(BuiltinViewStage::Map),
            Self::FlatMap => Some(BuiltinViewStage::FlatMap),
            Self::UniqueBy | Self::ExprBuiltin => None,
        }
    }
}

/// Payload-demand behavior for expression-bearing pipeline stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinExprPayload {
    /// The expression is used only to decide scan-time membership/prefix state.
    PredicateScan,
    /// The expression is a one-to-one projection that can rewrite downstream field demand.
    Projection,
    /// The expression computes an aggregate key; retained rows are not emitted downstream.
    KeyOnlyReducer,
    /// The expression computes a row-retaining aggregate key and therefore needs whole rows.
    RowKeyedReducer,
}

/// Concrete runtime hook implementation shared by one or more builtin names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinRuntimeHook {
    /// Predicate filter stream/barrier hook.
    SharedFilter,
    /// Builtin's own `Builtin::apply_barrier` implementation is available.
    Barrier,
    /// Builtin has both stream and barrier trait implementations.
    StreamAndBarrier,
}

impl BuiltinRuntimeHook {
    /// Whether this hook can run in the per-row streaming executor.
    #[inline]
    pub(crate) const fn has_stream(self) -> bool {
        matches!(self, Self::SharedFilter | Self::StreamAndBarrier)
    }

    /// Whether this hook can run against a materialized barrier buffer.
    #[inline]
    pub(crate) const fn has_barrier(self) -> bool {
        matches!(
            self,
            Self::SharedFilter | Self::Barrier | Self::StreamAndBarrier
        )
    }
}

/// Logical planner shape for pipeline-position builtins.
///
/// This keeps builtin classification in the builtin definitions while allowing
/// `plan::logical` to own construction of `LogicalPlan` nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinLogicalShape {
    /// `filter(expr)`-style streaming predicate.
    Filter,
    /// `filter(expr)` followed by `first()` when terminal.
    FilterThenFirst,
    /// `map(expr)` one-to-one projection.
    Map,
    /// `flat_map(expr)` expansion.
    FlatMap,
    /// `take(n)` positional prefix.
    Take,
    /// `skip(n)` positional offset.
    Skip,
    /// Nullary terminal first.
    First,
    /// Nullary terminal last.
    Last,
    /// Nullary numeric reducer.
    Sum,
    /// Nullary numeric reducer.
    Avg,
    /// Nullary numeric reducer.
    Min,
    /// Nullary numeric reducer.
    Max,
    /// Nullary count reducer.
    Count,
    /// Nullary reverse barrier.
    Reverse,
    /// Prefix predicate barrier.
    TakeWhile,
    /// Prefix predicate barrier.
    DropWhile,
    /// Sort with optional key.
    Sort,
    /// Nullary unique.
    Unique,
    /// `unique_by(expr)`.
    UniqueBy,
    /// `group_by(expr)`.
    GroupBy,
    /// `count_by(expr)` followed by first when terminal.
    CountBy,
    /// `index_by(expr)` followed by first when terminal.
    IndexBy,
    /// Nullary approximate distinct reducer.
    ApproxCountDistinct,
}

/// Source-level `$.rows()` stream operation behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinRowStreamOp {
    /// Toggle stream direction.
    Reverse,
    /// Keep rows matching the predicate.
    Filter,
    /// Keep the first row matching the predicate.
    FindFirst,
    /// Return the only row matching the predicate, erroring on zero or multiple matches.
    FindOne,
    /// Deduplicate rows by a key expression.
    DistinctBy,
    /// Keep a bounded prefix.
    Take,
    /// Keep the first row.
    First,
    /// Keep the last row.
    Last,
    /// Count retained rows.
    Count,
    /// Numeric sum over retained rows.
    Sum,
    /// Numeric average over retained rows.
    Avg,
    /// Numeric minimum over retained rows.
    Min,
    /// Numeric maximum over retained rows.
    Max,
    /// Predicate existential sink.
    Any,
    /// Predicate universal sink.
    All,
    /// Project each retained row.
    Map,
}

/// Argument contract for a source-level `$.rows()` stream operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinRowStreamArg {
    /// The operation accepts no argument.
    None,
    /// The operation accepts one expression/lambda argument.
    Expr,
    /// The operation accepts one non-negative integer argument.
    Usize,
}

impl BuiltinRowStreamOp {
    /// Argument kind required by this row-stream operation.
    #[inline]
    pub(crate) const fn arg(self) -> BuiltinRowStreamArg {
        match self {
            Self::Reverse
            | Self::First
            | Self::Last
            | Self::Count
            | Self::Sum
            | Self::Avg
            | Self::Min
            | Self::Max => BuiltinRowStreamArg::None,
            Self::Filter
            | Self::FindFirst
            | Self::FindOne
            | Self::DistinctBy
            | Self::Any
            | Self::All
            | Self::Map => BuiltinRowStreamArg::Expr,
            Self::Take => BuiltinRowStreamArg::Usize,
        }
    }

    /// Whether this operation finalizes a `$.rows()` stream.
    #[inline]
    pub(crate) const fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Last
                | Self::Count
                | Self::Any
                | Self::All
                | Self::FindOne
                | Self::Sum
                | Self::Avg
                | Self::Min
                | Self::Max
        )
    }

    /// Semantic streaming boundary for this source-level row operation.
    #[inline]
    pub(crate) const fn streaming_boundary(self) -> BuiltinStreamingBoundary {
        match self {
            Self::Reverse => BuiltinStreamingBoundary::FullInputOrder,
            Self::DistinctBy => BuiltinStreamingBoundary::FullInputState,
            Self::Take | Self::First | Self::Last | Self::FindFirst => {
                BuiltinStreamingBoundary::BoundedState
            }
            Self::Count | Self::Sum | Self::Avg | Self::Min | Self::Max => {
                BuiltinStreamingBoundary::FullInputState
            }
            Self::Filter | Self::FindOne | Self::Any | Self::All | Self::Map => {
                BuiltinStreamingBoundary::RowLocal
            }
        }
    }

    /// Fixed number of rows retained by this operation, independent of user
    /// arguments. Argument-bearing operations such as `take(n)` report their
    /// dynamic limit from the lowered stage.
    #[inline]
    pub(crate) const fn fixed_retained_limit(self) -> Option<usize> {
        match self {
            Self::First | Self::FindFirst | Self::Last => Some(1),
            _ => None,
        }
    }

    /// Whether this operation prevents partitioned file execution.
    #[inline]
    pub(crate) const fn blocks_parallel_partitioning(self) -> bool {
        matches!(self, Self::DistinctBy | Self::Last)
    }

    /// Whether this operation tests row membership with a predicate while
    /// preserving the surviving row order.
    #[inline]
    pub(crate) const fn is_filter_like(self) -> bool {
        matches!(self, Self::Filter | Self::FindFirst)
    }

    /// Whether this operation projects retained rows one-to-one.
    #[inline]
    pub(crate) const fn is_projector(self) -> bool {
        matches!(self, Self::Map)
    }

    /// Whether this operation evaluates a key expression for retained-row
    /// state such as stream deduplication.
    #[inline]
    pub(crate) const fn is_keyed_state(self) -> bool {
        matches!(self, Self::DistinctBy)
    }

    /// Whether this operation selects which rows continue downstream without
    /// changing their order.
    #[inline]
    pub(crate) const fn is_row_selection(self) -> bool {
        matches!(self, Self::Filter | Self::FindFirst | Self::DistinctBy)
    }

    /// Whether this operation can appear before a retained limit while keeping
    /// source-order early stop semantics conservative and unchanged.
    #[inline]
    pub(crate) const fn preserves_order_before_limit(self) -> bool {
        matches!(self, Self::Filter | Self::FindFirst | Self::Map)
    }

    /// Numeric reducer represented by this terminal stream operation, if any.
    #[inline]
    pub(crate) const fn numeric_reducer(self) -> Option<BuiltinNumericReducer> {
        match self {
            Self::Sum => Some(BuiltinNumericReducer::Sum),
            Self::Avg => Some(BuiltinNumericReducer::Avg),
            Self::Min => Some(BuiltinNumericReducer::Min),
            Self::Max => Some(BuiltinNumericReducer::Max),
            _ => None,
        }
    }

    /// Predicate sink represented by this terminal stream operation, if any.
    #[inline]
    pub(crate) const fn predicate_sink(self) -> Option<BuiltinPredicateSink> {
        match self {
            Self::Any => Some(BuiltinPredicateSink::Any),
            Self::All => Some(BuiltinPredicateSink::All),
            Self::FindOne => Some(BuiltinPredicateSink::FindOne),
            _ => None,
        }
    }
}

/// Marker that a builtin has a structural (index-based) execution backend.
/// The query planner may choose the structural path over the generic DFS walk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinStructural {
    /// Structural backend for `deep_find`.
    DeepFind,
    /// Structural backend for `deep_shape`.
    DeepShape,
    /// Structural backend for `deep_like`.
    DeepLike,
}

/// View-layer stage that a builtin can be lowered into.
/// Each variant corresponds to a distinct operation in the view execution path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinViewStage {
    /// Predicate-driven row filter stage.
    Filter,
    /// Null-removal filter stage.
    Compact,
    /// Literal equality removal filter stage.
    RemoveValue,
    /// Per-row projection stage.
    Map,
    /// Per-row expansion stage (one-to-many).
    FlatMap,
    /// Object key/value/entry expansion stage.
    ObjectItems(BuiltinViewObjectProjection),
    /// Array flattening expansion stage.
    Flatten,
    /// Object-field array explosion stage.
    Explode,
    /// Pair each row with its zero-based stream index.
    Enumerate,
    /// Adjacent pair stage.
    Pairwise,
    /// Stateful numeric one-pass scan stage.
    NumericScan(BuiltinViewNumericScan),
    /// Numeric stage that must see all input before it can emit rows.
    NumericFullInput(BuiltinViewNumericFullInput),
    /// Numeric lag by a fixed row offset.
    Lag,
    /// Numeric lead by a fixed row offset.
    Lead,
    /// Rolling numeric aggregate over a fixed-width window.
    Rolling(BuiltinViewRolling),
    /// Non-overlapping fixed-size chunk stage.
    Chunk,
    /// Sliding fixed-size window stage.
    Window,
    /// Prefix filter that stops at the first non-matching row.
    TakeWhile,
    /// Skips leading matching rows and passes the rest.
    DropWhile,
    /// Deduplication stage (keeps first occurrence of each key).
    Distinct,
    /// Keyed reduce stage (groups, counts, or indexes by key).
    KeyedReduce,
    /// Split rows into truthy/falsy predicate buckets.
    Partition,
    /// Set membership filter against a static argument list.
    SetFilter(BuiltinViewSetFilter),
    /// Set union with a static argument list.
    SetUnion,
    /// Join all receiver rows into one string with a static separator.
    JoinString,
    /// Zip receiver rows with a static array argument.
    ZipStatic,
    /// Zip receiver rows with a static array argument, padding the shorter side.
    ZipLongestStatic,
    /// Append one static value after all receiver rows.
    AppendValue,
    /// Prepend one static value before receiver rows.
    PrependValue,
    /// Positional limit stage.
    Take,
    /// Positional skip stage.
    Skip,
}

/// Whether a view stage needs to iterate the source view or can skip it entirely.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinViewInputMode {
    /// Stage reads values from the underlying view one by one.
    ReadsView,
    /// Stage does not consult the view at all (e.g. positional `take`/`skip`).
    SkipsViewRead,
}

/// How a view stage produces its output relative to the source view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinViewOutputMode {
    /// Output is a sub-slice of the input view (filter, take, skip, etc.).
    PreservesInputView,
    /// Output is a single borrowed subview derived from one input element (map).
    BorrowedSubview,
    /// Output is multiple borrowed subviews derived from one element (flat_map).
    BorrowedSubviews,
    /// Output is a freshly constructed owned value (keyed reduce, etc.).
    EmitsOwnedValue,
}

/// One-pass numeric scan operation for borrowed view rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinViewNumericScan {
    /// Difference from the previous numeric row.
    DiffWindow,
    /// Percentage change from the previous numeric row.
    PctChange,
    /// Cumulative maximum, carrying previous best over null/non-numeric rows.
    CumMax,
    /// Cumulative minimum, carrying previous best over null/non-numeric rows.
    CumMin,
}

/// Full-input numeric operation for borrowed view rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinViewNumericFullInput {
    /// Standardize numeric rows by full-input mean and standard deviation.
    Zscore,
}

/// Rolling numeric operation for borrowed view rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinViewRolling {
    /// Rolling sum over numeric rows, treating non-numeric/null rows as absent.
    Sum,
    /// Rolling average over numeric rows, null when the window has no numeric rows.
    Avg,
    /// Rolling minimum over numeric rows.
    Min,
    /// Rolling maximum over numeric rows.
    Max,
}

/// Set-filter operation for borrowed view rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinViewSetFilter {
    /// Keep rows absent from the static set.
    Diff,
    /// Keep rows present in the static set.
    Intersect,
}

/// Extra executor data required to construct a borrowed-view stage capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinViewCapabilityShape {
    /// Standard stage capability; body/native compatibility is enough.
    Generic,
    /// Stage removes rows matching a literal target value.
    RemoveValueTarget,
    /// Stage carries one literal value argument.
    ValueArg,
    /// Stage carries a static list of literal values.
    ValVecArg,
    /// Stage may carry an optional distinct key body.
    OptionalKeyBody,
    /// Stage requires keyed-reducer metadata plus a key body.
    KeyedReducer,
}

/// When, if ever, a view-stage or sink must materialise elements into owned
/// values. This is builtin metadata; executors consume it instead of
/// rediscovering materialization policy from builtin identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinViewMaterialization {
    /// No materialisation is needed; the stage/sink can operate entirely on borrowed views.
    Never,
    /// The stage must materialise the final value it emits (e.g. keyed reduce output).
    StageFinalValue,
    /// The sink materialises each output row into the result array (e.g. collect).
    SinkOutputRows,
    /// The sink materialises only the single selected row (e.g. first / last).
    SinkFinalRow,
    /// The sink materialises each element's numeric input for folding (e.g. sum).
    SinkNumericInput,
    /// The sink materialises input rows for its own comparison/state, not for output.
    SinkInputRows,
}

/// Canonical argument-count contract for pipeline lowering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinPipelineArity {
    /// Accepts exactly N arguments.
    Exact(usize),
    /// Accepts any count in the inclusive range.
    Range {
        /// Minimum accepted argument count.
        min: usize,
        /// Maximum accepted argument count.
        max: usize,
    },
}

impl BuiltinPipelineArity {
    /// Returns whether this contract accepts `arity` arguments.
    #[inline]
    pub(crate) const fn accepts(self, arity: usize) -> bool {
        match self {
            Self::Exact(n) => arity == n,
            Self::Range { min, max } => arity >= min && arity <= max,
        }
    }
}

/// Describes how a terminal reducing builtin accumulates its final result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BuiltinSinkSpec {
    /// Which accumulator algorithm to use.
    pub accumulator: BuiltinSinkAccumulator,
    /// How many rows the sink needs to see before it can emit a result.
    pub demand: BuiltinSinkDemand,
    /// Whether this sink accepts an optional predicate expression.
    pub accepts_predicate: bool,
}

impl BuiltinSinkSpec {
    /// Returns which portion of each input row this sink needs for its
    /// accumulator, excluding optional predicate/projection programs.
    #[inline]
    pub(crate) const fn value_need(self) -> BuiltinSinkValueNeed {
        self.demand.value_need()
    }

    /// Returns the argument-count contract for pipeline terminal lowering.
    #[inline]
    pub(crate) const fn pipeline_arity(self) -> BuiltinPipelineArity {
        match self.accumulator {
            BuiltinSinkAccumulator::Count => {
                if self.accepts_predicate {
                    BuiltinPipelineArity::Range { min: 0, max: 1 }
                } else {
                    BuiltinPipelineArity::Exact(0)
                }
            }
            BuiltinSinkAccumulator::Numeric | BuiltinSinkAccumulator::SelectOne(_) => {
                BuiltinPipelineArity::Range { min: 0, max: 1 }
            }
            BuiltinSinkAccumulator::ApproxDistinct => BuiltinPipelineArity::Exact(0),
        }
    }

    /// Returns when a borrowed-view executor must materialise input or result
    /// rows to run this sink.
    #[inline]
    pub(crate) const fn view_materialization(self) -> BuiltinViewMaterialization {
        match self.accumulator {
            BuiltinSinkAccumulator::Count | BuiltinSinkAccumulator::ApproxDistinct => {
                BuiltinViewMaterialization::Never
            }
            BuiltinSinkAccumulator::Numeric => BuiltinViewMaterialization::SinkNumericInput,
            BuiltinSinkAccumulator::SelectOne(_) => BuiltinViewMaterialization::SinkFinalRow,
        }
    }

    /// Whether this sink requires a numeric reducer operation to execute.
    #[inline]
    pub(crate) const fn requires_numeric_reducer(self) -> bool {
        matches!(self.accumulator, BuiltinSinkAccumulator::Numeric)
    }
}

/// The accumulation strategy for a terminal reducing builtin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinSinkAccumulator {
    /// Counts the number of rows.
    Count,
    /// Applies a numeric reduction (sum, avg, min, max).
    Numeric,
    /// Counts approximate distinct values using a probabilistic sketch.
    ApproxDistinct,
    /// Selects either the first or last observed row.
    SelectOne(BuiltinSelectionPosition),
}

impl BuiltinSinkAccumulator {
    /// Whether this accumulator stores its final result in reducer state.
    #[inline]
    pub(crate) const fn finishes_from_reducer_state(self) -> bool {
        matches!(self, Self::Count | Self::Numeric)
    }

    /// Return the selected stream position for first/last-style sinks.
    #[inline]
    pub(crate) const fn selection_position(self) -> Option<BuiltinSelectionPosition> {
        match self {
            Self::SelectOne(position) => Some(position),
            _ => None,
        }
    }

    /// Result for an empty input stream when no numeric reducer state is needed.
    #[inline]
    pub(crate) fn empty_stream_result(self) -> Option<Val> {
        match self {
            Self::Count | Self::ApproxDistinct => Some(Val::Int(0)),
            Self::SelectOne(_) => Some(Val::Null),
            Self::Numeric => None,
        }
    }
}

/// The keyed-reduction algorithm used by `group_by` / `count_by` / `index_by`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinKeyedReducer {
    /// Counts occurrences per key (`count_by`).
    Count,
    /// Maps each key to its last value (`index_by`).
    Index,
    /// Maps each key to a list of its values (`group_by`).
    Group,
}

impl BuiltinKeyedReducer {
    /// Demand law implied by the keyed reducer's accumulator semantics.
    #[inline]
    pub(crate) const fn demand_law(self) -> BuiltinDemandLaw {
        if self.needs_row_payload() {
            BuiltinDemandLaw::RowKeyedReducer
        } else {
            BuiltinDemandLaw::KeyOnlyReducer
        }
    }

    /// Expression payload lane implied by the keyed reducer's accumulator semantics.
    #[inline]
    pub(crate) const fn expr_payload(self) -> BuiltinExprPayload {
        match self {
            Self::Count => BuiltinExprPayload::KeyOnlyReducer,
            Self::Index | Self::Group => BuiltinExprPayload::RowKeyedReducer,
        }
    }

    /// Whether the accumulator must retain/materialise source rows.
    #[inline]
    pub(crate) const fn needs_row_payload(self) -> bool {
        matches!(self, Self::Index | Self::Group)
    }
}

/// Arg-extreme terminal sink behavior for key-based row selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinArgExtremeSink {
    /// Keep the row with the largest key.
    MaxBy,
    /// Keep the row with the smallest key.
    MinBy,
}

impl BuiltinArgExtremeSink {
    /// Registry id represented by this terminal sink.
    #[inline]
    pub(crate) const fn id(self) -> registry::BuiltinId {
        match self {
            Self::MaxBy => registry::BuiltinId::MAX_BY,
            Self::MinBy => registry::BuiltinId::MIN_BY,
        }
    }

    /// Builtin method represented by this terminal sink.
    #[inline]
    #[cfg(test)]
    pub(crate) const fn method(self) -> BuiltinMethod {
        match self {
            Self::MaxBy => BuiltinMethod::MaxBy,
            Self::MinBy => BuiltinMethod::MinBy,
        }
    }

    /// Demand law implied by arg-extreme selection.
    #[inline]
    pub(crate) const fn demand_law(self) -> BuiltinDemandLaw {
        BuiltinDemandLaw::RowKeyedReducer
    }

    /// Whether this sink keeps the largest projected key.
    #[inline]
    pub(crate) const fn wants_max(self) -> bool {
        matches!(self, Self::MaxBy)
    }
}

/// Predicate terminal sink behavior for builtins with a predicate argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinPredicateSink {
    /// Returns true when any row matches the predicate.
    Any,
    /// Returns true when every row matches the predicate.
    All,
    /// Returns the zero-based index of the first matching row, or null.
    FindIndex,
    /// Returns all zero-based indices whose rows match.
    IndicesWhere,
    /// Returns exactly one matching row, erroring on zero or multiple matches.
    FindOne,
}

impl BuiltinPredicateSink {
    /// Registry id represented by this terminal sink.
    #[inline]
    pub(crate) const fn id(self) -> registry::BuiltinId {
        match self {
            Self::Any => registry::BuiltinId::ANY,
            Self::All => registry::BuiltinId::ALL,
            Self::FindIndex => registry::BuiltinId::FIND_INDEX,
            Self::IndicesWhere => registry::BuiltinId::INDICES_WHERE,
            Self::FindOne => registry::BuiltinId::FIND_ONE,
        }
    }

    /// Builtin method represented by this terminal sink.
    #[inline]
    #[cfg(test)]
    pub(crate) const fn method(self) -> BuiltinMethod {
        match self {
            Self::Any => BuiltinMethod::Any,
            Self::All => BuiltinMethod::All,
            Self::FindIndex => BuiltinMethod::FindIndex,
            Self::IndicesWhere => BuiltinMethod::IndicesWhere,
            Self::FindOne => BuiltinMethod::FindOne,
        }
    }

    /// Demand law implied by predicate terminal sinks.
    #[inline]
    pub(crate) const fn demand_law(self) -> BuiltinDemandLaw {
        BuiltinDemandLaw::PredicateMapLike
    }

    /// Value need implied by this sink's result shape.
    #[inline]
    pub(crate) const fn value_need(self) -> crate::plan::demand::ValueNeed {
        match self {
            Self::FindOne => crate::plan::demand::ValueNeed::Whole,
            Self::Any | Self::All | Self::FindIndex | Self::IndicesWhere => {
                crate::plan::demand::ValueNeed::Predicate
            }
        }
    }

    /// Whether this terminal result can short-circuit.
    #[inline]
    pub(crate) const fn result_demand(self) -> crate::plan::demand::SinkResultDemand {
        match self {
            Self::Any | Self::FindIndex => crate::plan::demand::SinkResultDemand::UntilMatch,
            Self::All => crate::plan::demand::SinkResultDemand::UntilFailure,
            Self::IndicesWhere | Self::FindOne => crate::plan::demand::SinkResultDemand::None,
        }
    }

    /// Returns when a borrowed-view executor must materialise rows for this sink.
    #[inline]
    pub(crate) const fn view_materialization(self) -> BuiltinViewMaterialization {
        if self.returns_matching_row() {
            BuiltinViewMaterialization::SinkFinalRow
        } else {
            BuiltinViewMaterialization::Never
        }
    }

    /// Whether this predicate sink returns the matching input row itself.
    #[inline]
    pub(crate) const fn returns_matching_row(self) -> bool {
        matches!(self, Self::FindOne)
    }

    /// Result for an empty input stream, when the sink can complete without
    /// observing any rows.
    #[inline]
    pub(crate) fn empty_stream_result(self) -> Option<Val> {
        match self {
            Self::Any => Some(Val::Bool(false)),
            Self::All => Some(Val::Bool(true)),
            Self::FindIndex => Some(Val::Null),
            Self::IndicesWhere => Some(Val::arr(Vec::new())),
            Self::FindOne => None,
        }
    }

    /// Result when every row in a stream observes the same predicate result.
    #[inline]
    pub(crate) fn constant_predicate_stream_result(
        self,
        matched: bool,
        count: usize,
    ) -> Option<Val> {
        if count == 0 {
            return self.empty_stream_result();
        }
        match self {
            Self::Any => Some(Val::Bool(matched)),
            Self::All => Some(Val::Bool(matched)),
            Self::FindIndex => Some(if matched { Val::Int(0) } else { Val::Null }),
            Self::IndicesWhere => {
                if matched {
                    Some(Val::int_vec((0..count).map(|idx| idx as i64).collect()))
                } else {
                    Some(Val::arr(Vec::new()))
                }
            }
            Self::FindOne => None,
        }
    }
}

/// Membership terminal sink behavior for builtins with a target value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinMembershipSink {
    /// Returns true when any row equals the target.
    Includes,
    /// Returns the zero-based index of the first matching row, or null.
    Index,
    /// Returns all zero-based indices matching the target.
    IndicesOf,
}

impl BuiltinMembershipSink {
    /// Registry id represented by this terminal sink.
    #[inline]
    pub(crate) const fn id(self) -> registry::BuiltinId {
        match self {
            Self::Includes => registry::BuiltinId::INCLUDES,
            Self::Index => registry::BuiltinId::INDEX,
            Self::IndicesOf => registry::BuiltinId::INDICES_OF,
        }
    }

    /// Builtin method represented by this terminal sink.
    #[inline]
    pub(crate) const fn method(self) -> BuiltinMethod {
        match self {
            Self::Includes => BuiltinMethod::Includes,
            Self::Index => BuiltinMethod::Index,
            Self::IndicesOf => BuiltinMethod::IndicesOf,
        }
    }

    /// Demand law implied by membership terminal sinks.
    #[inline]
    pub(crate) const fn demand_law(self) -> BuiltinDemandLaw {
        BuiltinDemandLaw::PredicateMapLike
    }

    /// Value need implied by membership matching.
    #[inline]
    pub(crate) const fn value_need(self) -> crate::plan::demand::ValueNeed {
        crate::plan::demand::ValueNeed::Whole
    }

    /// Whether this terminal result can short-circuit.
    #[inline]
    pub(crate) const fn result_demand(self) -> crate::plan::demand::SinkResultDemand {
        match self {
            Self::Includes | Self::Index => crate::plan::demand::SinkResultDemand::UntilMatch,
            Self::IndicesOf => crate::plan::demand::SinkResultDemand::None,
        }
    }

    /// Whether this membership sink returns only a boolean answer.
    #[inline]
    pub(crate) const fn returns_bool(self) -> bool {
        matches!(self, Self::Includes)
    }

    /// Result for an empty input stream.
    #[inline]
    pub(crate) fn empty_stream_result(self) -> Val {
        match self {
            Self::Includes => Val::Bool(false),
            Self::Index => Val::Null,
            Self::IndicesOf => Val::arr(Vec::new()),
        }
    }
}

/// Builtin array-child selector behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinArraySelector {
    /// First child.
    First,
    /// Last child.
    Last,
    /// Index supplied by an argument.
    Nth,
}

impl BuiltinArraySelector {
    /// Canonical builtin id represented by this selector.
    #[cfg(test)]
    #[inline]
    pub(crate) const fn id(self) -> registry::BuiltinId {
        match self {
            Self::First => registry::BuiltinId::FIRST,
            Self::Last => registry::BuiltinId::LAST,
            Self::Nth => registry::BuiltinId::NTH,
        }
    }

    /// Demand law implied by this positional selector.
    #[inline]
    pub(crate) const fn demand_law(self) -> BuiltinDemandLaw {
        match self {
            Self::First => BuiltinDemandLaw::First,
            Self::Last => BuiltinDemandLaw::Last,
            Self::Nth => BuiltinDemandLaw::Nth,
        }
    }

    /// Select-one sink position, when this selector is a terminal first/last sink.
    #[inline]
    pub(crate) const fn selection_position(self) -> Option<BuiltinSelectionPosition> {
        match self {
            Self::First => Some(BuiltinSelectionPosition::First),
            Self::Last => Some(BuiltinSelectionPosition::Last),
            Self::Nth => None,
        }
    }

    /// Row-stream operation represented by this selector, when supported.
    #[inline]
    pub(crate) const fn row_stream_op(self) -> Option<BuiltinRowStreamOp> {
        match self {
            Self::First => Some(BuiltinRowStreamOp::First),
            Self::Last => Some(BuiltinRowStreamOp::Last),
            Self::Nth => None,
        }
    }

    /// Pipeline lowering implied by this selector.
    #[inline]
    pub(crate) const fn pipeline_lowering(self) -> BuiltinPipelineLowering {
        match self {
            Self::First | Self::Last => BuiltinPipelineLowering::TerminalSink,
            Self::Nth => BuiltinPipelineLowering::TerminalUsizeSink { min: 0 },
        }
    }
}

/// Rewrite available after a stage has rearranged rows but before selecting
/// one end. For example, `sort().first()` can become `min()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BuiltinSelectionRewrite {
    /// Replacement for `.first()`.
    pub first: Option<BuiltinMethod>,
    /// Replacement for `.last()`.
    pub last: Option<BuiltinMethod>,
    /// Replacement for `[0]`.
    pub index_zero: Option<BuiltinMethod>,
    /// Replacement for `[-1]`.
    pub index_minus_one: Option<BuiltinMethod>,
}

impl BuiltinSelectionRewrite {
    /// Empty rewrite table.
    #[inline]
    pub const fn new() -> Self {
        Self {
            first: None,
            last: None,
            index_zero: None,
            index_minus_one: None,
        }
    }

    #[inline]
    pub const fn first(mut self, method: BuiltinMethod) -> Self {
        self.first = Some(method);
        self
    }

    #[inline]
    pub const fn last(mut self, method: BuiltinMethod) -> Self {
        self.last = Some(method);
        self
    }

    #[inline]
    pub const fn index_zero(mut self, method: BuiltinMethod) -> Self {
        self.index_zero = Some(method);
        self
    }

    #[inline]
    pub const fn index_minus_one(mut self, method: BuiltinMethod) -> Self {
        self.index_minus_one = Some(method);
        self
    }
}

/// Which end of the stream the `SelectOne` sink picks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinSelectionPosition {
    /// Pick the first row seen (short-circuits on `first`).
    First,
    /// Pick the last row seen (must consume the whole stream for `last`).
    Last,
}

impl BuiltinSelectionPosition {
    /// Whether this selection keeps the last retained row.
    #[inline]
    pub(crate) const fn wants_last(self) -> bool {
        matches!(self, Self::Last)
    }
}

/// How many rows a terminal sink must consume to produce its result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinSinkDemand {
    /// Must see every row; `order` indicates whether row order matters.
    All {
        /// Which aspect of each row value is needed.
        value: BuiltinSinkValueNeed,
        /// Whether the sink is order-sensitive (affects fusion legality).
        order: bool,
    },
    /// Can stop after the first qualifying row.
    First {
        /// Which aspect of the first row's value is needed.
        value: BuiltinSinkValueNeed,
    },
    /// Can satisfy selection by reading from the tail of a reversible/indexed source.
    Last {
        /// Which aspect of the last row's value is needed.
        value: BuiltinSinkValueNeed,
    },
}

impl BuiltinSinkDemand {
    /// Returns which part of each input row this demand reads.
    #[inline]
    pub(crate) const fn value_need(self) -> BuiltinSinkValueNeed {
        match self {
            Self::All { value, .. } | Self::First { value } | Self::Last { value } => value,
        }
    }
}

/// Which portion of each row value the sink algorithm actually reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinSinkValueNeed {
    /// The sink counts rows only and never dereferences their values.
    None,
    /// The sink needs the complete `Val` (e.g. `first`, `last`).
    Whole,
    /// The sink only reads the numeric representation of each value (sum, avg, min, max).
    Numeric,
}

/// Which numeric aggregation the `Numeric` sink accumulator performs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinNumericReducer {
    /// Accumulate by addition.
    Sum,
    /// Accumulate sum and count, emit mean.
    Avg,
    /// Track the running minimum.
    Min,
    /// Track the running maximum.
    Max,
}

impl BuiltinNumericReducer {
    /// Registry id represented by this reducer.
    #[inline]
    pub(crate) const fn id(self) -> registry::BuiltinId {
        match self {
            Self::Sum => registry::BuiltinId::SUM,
            Self::Avg => registry::BuiltinId::AVG,
            Self::Min => registry::BuiltinId::MIN,
            Self::Max => registry::BuiltinId::MAX,
        }
    }

    /// Builtin method represented by this reducer.
    #[inline]
    #[cfg(test)]
    pub(crate) const fn method(self) -> BuiltinMethod {
        match self {
            Self::Sum => BuiltinMethod::Sum,
            Self::Avg => BuiltinMethod::Avg,
            Self::Min => BuiltinMethod::Min,
            Self::Max => BuiltinMethod::Max,
        }
    }

    /// Logical reducer shape represented by this numeric accumulator.
    #[inline]
    pub(crate) const fn logical_shape(self) -> BuiltinLogicalShape {
        match self {
            Self::Sum => BuiltinLogicalShape::Sum,
            Self::Avg => BuiltinLogicalShape::Avg,
            Self::Min => BuiltinLogicalShape::Min,
            Self::Max => BuiltinLogicalShape::Max,
        }
    }

    /// Row-stream operation represented by this numeric accumulator.
    #[inline]
    pub(crate) const fn row_stream_op(self) -> BuiltinRowStreamOp {
        match self {
            Self::Sum => BuiltinRowStreamOp::Sum,
            Self::Avg => BuiltinRowStreamOp::Avg,
            Self::Min => BuiltinRowStreamOp::Min,
            Self::Max => BuiltinRowStreamOp::Max,
        }
    }

    /// Demand law shared by numeric reducers.
    #[inline]
    pub(crate) const fn demand_law(self) -> BuiltinDemandLaw {
        BuiltinDemandLaw::NumericReducer
    }

    /// Whether this reducer tracks the lower extreme among observed values.
    #[inline]
    pub(crate) const fn selects_min(self) -> bool {
        matches!(self, Self::Min)
    }
}

/// Describes how two adjacent identical stages can be collapsed into one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinStageMerge {
    /// `take(a).take(b)` → `take(min(a, b))`.
    UsizeMin,
    /// `skip(a).skip(b)` → `skip(a + b)` (saturating to avoid overflow).
    UsizeSaturatingAdd,
}

/// Algebraic cancellation rule for a builtin.
/// Two adjacent stages cancel when `a.cancels_with(b)` is true.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinCancellation {
    /// The operation is its own inverse (`reverse().reverse()` = identity).
    SelfInverse(BuiltinCancelGroup),
    /// The operation has a paired inverse (encode/decode, escape/unescape).
    Inverse {
        /// Which encode/decode group this operation belongs to.
        group: BuiltinCancelGroup,
        /// Whether this is the forward (encoding) or backward (decoding) member.
        side: BuiltinCancelSide,
    },
}

/// Identifies which encode/decode pair a cancellation belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinCancelGroup {
    /// String reversal (`reverse_str` is self-inverse).
    Reverse,
    /// Base64 encode/decode pair.
    Base64,
    /// URL percent-encode/decode pair.
    Url,
    /// HTML escape/unescape pair.
    Html,
}

/// Which side of a forward/backward cancellation pair this builtin occupies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinCancelSide {
    /// The encoding or escaping direction.
    Forward,
    /// The decoding or unescaping direction.
    Backward,
}

impl BuiltinCancellation {
    /// Returns true if `self` and `other` are algebraically inverse and can be eliminated.
    #[inline]
    pub fn cancels_with(self, other: Self) -> bool {
        match (self, other) {
            (Self::SelfInverse(a), Self::SelfInverse(b)) => a == b,
            (Self::Inverse { group: a, side: sa }, Self::Inverse { group: b, side: sb }) => {
                a == b && sa != sb
            }
            _ => false,
        }
    }
}

impl BuiltinStageMerge {
    /// Combines two stage arguments according to the merge rule.
    #[inline]
    pub fn combine_usize(self, a: usize, b: usize) -> usize {
        match self {
            Self::UsizeMin => a.min(b),
            Self::UsizeSaturatingAdd => a.saturating_add(b),
        }
    }
}

impl BuiltinViewStage {
    /// Returns whether this stage reads values from the source view or can skip it.
    #[inline]
    pub fn input_mode(self) -> BuiltinViewInputMode {
        match self {
            Self::Filter
            | Self::Compact
            | Self::RemoveValue
            | Self::Map
            | Self::FlatMap
            | Self::ObjectItems(_)
            | Self::Flatten
            | Self::Explode
            | Self::Enumerate
            | Self::Pairwise
            | Self::NumericScan(_)
            | Self::NumericFullInput(_)
            | Self::Lag
            | Self::Lead
            | Self::Rolling(_)
            | Self::Chunk
            | Self::Window
            | Self::TakeWhile
            | Self::DropWhile
            | Self::Distinct
            | Self::KeyedReduce
            | Self::Partition
            | Self::SetFilter(_)
            | Self::SetUnion
            | Self::JoinString
            | Self::ZipStatic
            | Self::ZipLongestStatic
            | Self::AppendValue
            | Self::PrependValue => BuiltinViewInputMode::ReadsView,
            Self::Take | Self::Skip => BuiltinViewInputMode::SkipsViewRead,
        }
    }

    /// Returns how this stage relates its output to the source view's memory.
    #[inline]
    pub fn output_mode(self) -> BuiltinViewOutputMode {
        match self {
            Self::Map => BuiltinViewOutputMode::BorrowedSubview,
            Self::ObjectItems(BuiltinViewObjectProjection::Values) => {
                BuiltinViewOutputMode::BorrowedSubviews
            }
            Self::ObjectItems(_) => BuiltinViewOutputMode::EmitsOwnedValue,
            Self::FlatMap | Self::Flatten => BuiltinViewOutputMode::BorrowedSubviews,
            Self::Explode
            | Self::Enumerate
            | Self::Pairwise
            | Self::NumericScan(_)
            | Self::NumericFullInput(_)
            | Self::Lag
            | Self::Lead
            | Self::Rolling(_)
            | Self::Chunk
            | Self::Window
            | Self::KeyedReduce
            | Self::Partition
            | Self::SetUnion
            | Self::JoinString
            | Self::ZipStatic
            | Self::ZipLongestStatic
            | Self::AppendValue
            | Self::PrependValue => BuiltinViewOutputMode::EmitsOwnedValue,
            Self::Filter
            | Self::Compact
            | Self::RemoveValue
            | Self::TakeWhile
            | Self::DropWhile
            | Self::Distinct
            | Self::SetFilter(_)
            | Self::Take
            | Self::Skip => BuiltinViewOutputMode::PreservesInputView,
        }
    }

    /// Whether the stage body must return a borrowed view-backed result.
    #[inline]
    pub fn requires_borrowed_body_result(self) -> bool {
        false
    }

    /// Returns the executor capability construction shape for this stage.
    #[inline]
    pub fn capability_shape(self) -> BuiltinViewCapabilityShape {
        match self {
            Self::RemoveValue => BuiltinViewCapabilityShape::RemoveValueTarget,
            Self::AppendValue | Self::PrependValue | Self::ZipStatic => {
                BuiltinViewCapabilityShape::ValueArg
            }
            Self::ZipLongestStatic => BuiltinViewCapabilityShape::ValVecArg,
            Self::SetFilter(_) | Self::SetUnion => BuiltinViewCapabilityShape::ValVecArg,
            Self::Distinct => BuiltinViewCapabilityShape::OptionalKeyBody,
            Self::KeyedReduce => BuiltinViewCapabilityShape::KeyedReducer,
            _ => BuiltinViewCapabilityShape::Generic,
        }
    }

    /// Returns the output row-count relationship of this stage.
    #[inline]
    pub fn cardinality(self) -> BuiltinCardinality {
        match self {
            Self::Filter | Self::Compact | Self::RemoveValue | Self::SetFilter(_) => {
                BuiltinCardinality::Filtering
            }
            Self::Map => BuiltinCardinality::OneToOne,
            Self::FlatMap | Self::ObjectItems(_) | Self::Flatten | Self::Explode => {
                BuiltinCardinality::Expanding
            }
            Self::Enumerate => BuiltinCardinality::OneToOne,
            Self::Pairwise => BuiltinCardinality::Filtering,
            Self::NumericScan(_) => BuiltinCardinality::OneToOne,
            Self::NumericFullInput(_) => BuiltinCardinality::OneToOne,
            Self::Lag | Self::Lead | Self::Rolling(_) => BuiltinCardinality::OneToOne,
            Self::Chunk | Self::Window => BuiltinCardinality::Barrier,
            Self::TakeWhile | Self::DropWhile => BuiltinCardinality::Filtering,
            Self::Distinct => BuiltinCardinality::Filtering,
            Self::KeyedReduce => BuiltinCardinality::Reducing,
            Self::Partition => BuiltinCardinality::Barrier,
            Self::SetUnion
            | Self::JoinString
            | Self::ZipStatic
            | Self::ZipLongestStatic
            | Self::AppendValue
            | Self::PrependValue => BuiltinCardinality::Barrier,
            Self::Take | Self::Skip => BuiltinCardinality::Bounded,
        }
    }

    /// Whether this view stage emits exactly one row for each input row.
    #[inline]
    pub fn preserves_cardinality(self) -> bool {
        matches!(self.cardinality(), BuiltinCardinality::OneToOne)
    }

    /// Returns when this view stage must materialise data while executing.
    #[inline]
    pub fn materialization(self) -> BuiltinViewMaterialization {
        match self {
            Self::KeyedReduce | Self::Partition => BuiltinViewMaterialization::StageFinalValue,
            Self::Filter
            | Self::Compact
            | Self::RemoveValue
            | Self::Map
            | Self::FlatMap
            | Self::ObjectItems(_)
            | Self::Flatten
            | Self::Explode
            | Self::Enumerate
            | Self::Pairwise
            | Self::NumericScan(_)
            | Self::NumericFullInput(_)
            | Self::Lag
            | Self::Lead
            | Self::Rolling(_)
            | Self::Chunk
            | Self::Window
            | Self::TakeWhile
            | Self::DropWhile
            | Self::Distinct
            | Self::SetFilter(_)
            | Self::SetUnion
            | Self::JoinString
            | Self::ZipStatic
            | Self::ZipLongestStatic
            | Self::AppendValue
            | Self::PrependValue
            | Self::Take
            | Self::Skip => BuiltinViewMaterialization::Never,
        }
    }

    /// Returns whether this stage can participate in indexed (random-access) evaluation.
    #[inline]
    pub fn can_indexed(self) -> bool {
        matches!(self, Self::Map)
    }

    /// Returns the relative per-row cost estimate used by the planner.
    #[inline]
    pub fn cost(self) -> f64 {
        match self {
            Self::Filter
            | Self::Compact
            | Self::RemoveValue
            | Self::Map
            | Self::FlatMap
            | Self::ObjectItems(_)
            | Self::Flatten
            | Self::Explode
            | Self::Enumerate
            | Self::Pairwise
            | Self::NumericScan(_)
            | Self::NumericFullInput(_)
            | Self::Lag
            | Self::Lead
            | Self::Rolling(_)
            | Self::Chunk
            | Self::Window
            | Self::TakeWhile
            | Self::DropWhile
            | Self::Distinct
            | Self::SetFilter(_)
            | Self::SetUnion
            | Self::JoinString
            | Self::ZipStatic
            | Self::ZipLongestStatic
            | Self::KeyedReduce
            | Self::Partition
            | Self::AppendValue
            | Self::PrependValue => 10.0,
            Self::Take | Self::Skip => 0.5,
        }
    }

    /// Returns the estimated output-to-input row ratio (1.0 = no change, 0.5 = half the rows).
    #[inline]
    pub fn selectivity(self) -> f64 {
        match self {
            Self::Filter
            | Self::Compact
            | Self::RemoveValue
            | Self::TakeWhile
            | Self::DropWhile
            | Self::SetFilter(_) => 0.5,
            Self::Distinct => 1.0,
            Self::Map
            | Self::FlatMap
            | Self::ObjectItems(_)
            | Self::Flatten
            | Self::Explode
            | Self::Enumerate
            | Self::Pairwise
            | Self::NumericScan(_)
            | Self::NumericFullInput(_)
            | Self::Lag
            | Self::Lead
            | Self::Rolling(_)
            | Self::Chunk
            | Self::Window
            | Self::KeyedReduce
            | Self::Partition
            | Self::SetUnion
            | Self::JoinString
            | Self::ZipStatic
            | Self::ZipLongestStatic
            | Self::AppendValue
            | Self::PrependValue => 1.0,
            Self::Take | Self::Skip => 0.5,
        }
    }
}

/// Planning metadata for a builtin in the pipeline execution path.
/// The planner uses these fields to order and fuse pipeline stages.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BuiltinPipelineShape {
    /// Row-count relationship of this stage.
    pub cardinality: BuiltinCardinality,
    /// Whether the stage supports indexed access.
    pub can_indexed: bool,
    /// Relative per-row cost used for ordering heuristics.
    pub cost: f64,
    /// Estimated output/input row ratio.
    pub selectivity: f64,
}

/// When/how a pipeline stage materialises its output into a concrete `Vec<Val>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinPipelineMaterialization {
    /// Stage processes rows one-at-a-time without buffering.
    Streaming,
    /// Stage buffers all input (barrier), then emits via the composed path.
    ComposedBarrier,
}

/// Semantic reason a builtin can or cannot stay in the borrowed streaming
/// domain. This is intentionally separate from the concrete executor
/// materialization policy: a full-input ordering boundary may still use a
/// bounded top-k algorithm for a downstream limit, while a row-local stage
/// should never force whole-row ownership by itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinStreamingBoundary {
    /// Row-local execution; no global state or ownership boundary is inherent.
    RowLocal,
    /// Source-lifting stream boundary such as `rows()`.
    SourceStream,
    /// Bounded positional/prefix state only, such as `take`, `skip`, `window`,
    /// or `chunk` when downstream demand keeps it bounded.
    BoundedState,
    /// Predicate-defined prefix state such as `take_while` / `drop_while`.
    PrefixState,
    /// One input may emit many outputs, but the operation can stream when the
    /// executor supports borrowed expansion.
    Expanding,
    /// Full-input state keyed by values, without requiring output order sort.
    FullInputState,
    /// Full-input ordering boundary such as `sort` or `reverse`.
    FullInputOrder,
    /// The current implementation still requires legacy materialized
    /// execution. This is the highest-priority target for future tape work.
    LegacyMaterialized,
}

/// Describes how a pipeline stage interacts with the ordering of its input stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinPipelineOrderEffect {
    /// Stage forwards rows in the same order it receives them.
    Preserves,
    /// Stage emits a contiguous prefix determined by a predicate (take_while, drop_while).
    PredicatePrefix,
    /// Stage may reorder or buffer all rows (sort, group_by, etc.).
    Blocks,
}

/// Stage variant for columnar (typed-array) execution backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinColumnarStage {
    /// Columnar predicate filter.
    Filter,
    /// Columnar projection.
    Map,
    /// Columnar expansion.
    FlatMap,
    /// Columnar keyed grouping.
    GroupBy,
}

impl BuiltinPipelineShape {
    /// Constructs a `BuiltinPipelineShape` from its four planning fields.
    #[inline]
    pub fn new(
        cardinality: BuiltinCardinality,
        can_indexed: bool,
        cost: f64,
        selectivity: f64,
    ) -> Self {
        Self {
            cardinality,
            can_indexed,
            cost,
            selectivity,
        }
    }
}

/// Describes the *shape* of a builtin's lowering call (arg count + arg type).
///
/// The lowering routine dispatches first on this shape to validate args, then on
/// `BuiltinMethod` itself to emit the right `Stage` variant. Shape tags do not
/// duplicate method identity — the method is already passed alongside the spec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinPipelineLowering {
    /// One Expr argument (a lambda).
    ExprArg,
    /// One Expr argument followed by a terminal builtin that collapses the stream when last.
    TerminalExprArg {
        /// The terminal method applied after the stage.
        terminal: BuiltinMethod,
    },
    /// No arguments.
    Nullary,
    /// One `usize` argument with a minimum legal value.
    UsizeArg {
        /// Minimum legal argument value; arguments below this are rejected.
        min: usize,
    },
    /// One string argument.
    StringArg,
    /// Two string arguments.
    StringPairArg,
    /// One or two integer arguments (e.g. slice).
    IntRangeArg,
    /// Sort with optional key expression (zero or one arg).
    Sort,
    /// Terminal sink (no stage emitted).
    TerminalSink,
    /// Terminal sink with one `usize` argument (e.g. `nth(i)`).
    TerminalUsizeSink {
        /// Minimum legal argument value; arguments below this are rejected.
        min: usize,
    },
}

/// Broad category for a builtin, used for grouping and display purposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinCategory {
    /// Operates on a single scalar value (string transforms, math, type ops).
    Scalar,
    /// Streaming one-to-one transform over array elements (map, enumerate, etc.).
    StreamingOneToOne,
    /// Streaming predicate filter (filter, take_while, drop_while, compact, etc.).
    StreamingFilter,
    /// Streaming expansion (flat_map, flatten, explode, split, etc.).
    StreamingExpand,
    /// Reduces many rows to one value (sum, count, any, all, group_by, etc.).
    Reducer,
    /// Positional slice (first, last, nth, take, skip).
    Positional,
    /// Full barrier: must buffer all input before emitting (sort, reverse, window, etc.).
    Barrier,
    /// Object-manipulation builtin (pick, omit, merge, keys, values, etc.).
    Object,
    /// Dot-path navigation and mutation (get_path, set_path, del_path, etc.).
    Path,
    /// Deep tree traversal (deep_find, deep_shape, walk, rec, etc.).
    Deep,
    /// Serialisation / deserialisation (to_csv, to_json, from_json, etc.).
    Serialization,
    /// Set-theory or join operations across multiple collections (equi_join, etc.).
    Relational,
    /// In-place mutation chain write (set, update).
    Mutation,
    /// Category is not known at compile time.
    Unknown,
}

/// Row-count relationship between a builtin's input and output.
/// Used by the pipeline planner to reason about stream length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinCardinality {
    /// Every input row produces exactly one output row.
    OneToOne,
    /// Output has at most as many rows as the input (subset).
    Filtering,
    /// Output may have more rows than the input (flat_map, flatten, etc.).
    Expanding,
    /// Output is bounded by a fixed constant regardless of input size.
    Bounded,
    /// Multiple input rows collapse to one output value.
    Reducing,
    /// Must buffer the full input stream before emitting; output size may vary.
    Barrier,
}

impl BuiltinSpec {
    /// Creates a minimal `BuiltinSpec` with sensible defaults (pure, cost 1.0, no optional features).
    fn new(category: BuiltinCategory, cardinality: BuiltinCardinality) -> Self {
        Self {
            pure: true,
            category,
            cardinality,
            can_indexed: false,
            view_native: false,
            view_scalar: false,
            view_scalar_op: None,
            view_value_projection: None,
            view_object_projection: None,
            view_string_expand: None,
            raw_json_scalar: None,
            object_lambda: None,
            string_pair_stage: None,
            nullary_stage: None,
            expr_stage: None,
            expr_payload: None,
            logical_shape: None,
            row_stream_op: None,
            view_stage: None,
            sink: None,
            keyed_reducer: None,
            numeric_reducer: None,
            arg_extreme_sink: None,
            predicate_sink: None,
            membership_sink: None,
            array_selector: None,
            selection_rewrite: None,
            stage_merge: None,
            cancellation: None,
            idempotent: false,
            accepts_lambda_arg: false,
            order_only: false,
            runtime_hook: None,
            output_cap_receiver: false,
            columnar_stage: None,
            structural: None,
            cost: 1.0,
            demand_law: BuiltinDemandLaw::Identity,
            materialization: BuiltinPipelineMaterialization::Streaming,
            streaming_boundary: BuiltinStreamingBoundary::RowLocal,
            pipeline_shape: None,
            order_effect: None,
            lowering: None,
            is_element: false,
            never_unwrap: false,
            stream_source: false,
        }
    }

    /// Returns `true` when `$.path.method()` should bypass pipeline streaming
    /// and dispatch as a direct `apply_one` (or `apply_args`) call on the
    /// single value produced by the chain. Eligibility covers scalar and
    /// object one-to-one builtins (e.g. `upper`, `type`, `ceil`, `omit`,
    /// `transform_values`) that have not opted out via `never_unwrap`.
    /// Streaming and reducing categories keep pipeline lowering — their
    /// per-element behavior is the canonical semantic on path receivers.
    pub fn dispatches_scalar_direct(&self) -> bool {
        matches!(
            self.category,
            BuiltinCategory::Scalar | BuiltinCategory::Object
        ) && matches!(self.cardinality, BuiltinCardinality::OneToOne)
            && !self.never_unwrap
    }

    /// Marks this builtin as safe for indexed (random-access) evaluation.
    fn indexed(mut self) -> Self {
        self.can_indexed = true;
        self
    }

    /// Marks this builtin as having a native view-path implementation.
    fn view_native(mut self) -> Self {
        self.view_native = true;
        self
    }

    /// Attaches the view stage lowering target for this builtin.
    fn view_stage(mut self, stage: BuiltinViewStage) -> Self {
        self.view_stage = Some(stage);
        self
    }

    /// Marks this builtin as a view-scalar method (implies `view_native`).
    fn view_scalar(mut self) -> Self {
        self.view_scalar = true;
        self.view_native = true;
        self
    }

    /// Attaches the concrete borrowed-view scalar dispatch family.
    fn view_scalar_op(mut self, op: BuiltinViewScalarOp) -> Self {
        self.view_scalar_op = Some(op);
        self.view_scalar()
    }

    /// Attaches a view-native whole-value projection operation.
    fn view_value_projection(mut self, projection: BuiltinViewValueProjection) -> Self {
        self.view_value_projection = Some(projection);
        self.view_native = true;
        self
    }

    /// Attaches a view-native object/path projection operation.
    fn view_object_projection(mut self, projection: BuiltinViewObjectProjection) -> Self {
        self.view_object_projection = Some(projection);
        self.view_native = true;
        self
    }

    /// Attaches a view-native string expansion operation.
    fn view_string_expand(mut self, expand: BuiltinViewStringExpand) -> Self {
        self.view_string_expand = Some(expand);
        self.view_native = true;
        self
    }

    /// Attaches a raw-byte JSON scalar operation.
    fn raw_json_scalar(mut self, scalar: BuiltinRawJsonScalar) -> Self {
        self.raw_json_scalar = Some(scalar);
        self.view_native = true;
        self
    }

    /// Marks a builtin as algebraically idempotent.
    fn idempotent(mut self) -> Self {
        self.idempotent = true;
        self
    }

    /// Attaches a stage-to-selection rewrite table.
    fn selection_rewrite(mut self, rewrite: BuiltinSelectionRewrite) -> Self {
        self.selection_rewrite = Some(rewrite);
        self
    }

    /// Marks a builtin as accepting a lambda/expression argument.
    fn lambda_arg(mut self) -> Self {
        self.accepts_lambda_arg = true;
        self
    }

    /// Marks a pipeline stage as changing only row order.
    fn order_only(mut self) -> Self {
        self.order_only = true;
        self
    }

    /// Attaches a shared runtime hook implementation target.
    fn runtime_hook(mut self, hook: BuiltinRuntimeHook) -> Self {
        self.runtime_hook = Some(hook);
        self
    }

    /// Marks this builtin as supporting receiver-mode bounded output in the VM.
    fn output_cap_receiver(mut self) -> Self {
        self.output_cap_receiver = true;
        self
    }

    /// Attaches object-lambda operation behavior.
    fn object_lambda(mut self, lambda: BuiltinObjectLambda) -> Self {
        self.object_lambda = Some(lambda);
        self
    }

    /// Attaches two-string-argument pipeline stage behavior.
    fn string_pair_stage(mut self, stage: BuiltinStringPairStage) -> Self {
        self.string_pair_stage = Some(stage);
        self
    }

    /// Attaches nullary pipeline stage behavior.
    fn nullary_stage(mut self, stage: BuiltinNullaryStage) -> Self {
        self.nullary_stage = Some(stage);
        self
    }

    /// Attaches expression-argument pipeline stage behavior.
    fn expr_stage(mut self, stage: BuiltinExprStage) -> Self {
        self.expr_stage = Some(stage);
        self
    }

    /// Attaches expression payload-demand behavior.
    fn expr_payload(mut self, payload: BuiltinExprPayload) -> Self {
        self.expr_payload = Some(payload);
        self
    }

    /// Attaches logical planner node shape.
    fn logical_shape(mut self, shape: BuiltinLogicalShape) -> Self {
        self.logical_shape = Some(shape);
        self
    }

    /// Attaches source-level row-stream operation behavior.
    fn row_stream_op(mut self, op: BuiltinRowStreamOp) -> Self {
        self.row_stream_op = Some(op);
        self
    }

    /// Attaches a columnar stage kind for typed-array execution backends.
    fn columnar_stage(mut self, stage: BuiltinColumnarStage) -> Self {
        self.columnar_stage = Some(stage);
        self
    }

    /// Configures a counting sink (demand all rows, value not needed, order-insensitive).
    fn count_sink(mut self) -> Self {
        self.sink = Some(BuiltinSinkSpec {
            accumulator: BuiltinSinkAccumulator::Count,
            demand: BuiltinSinkDemand::All {
                value: BuiltinSinkValueNeed::None,
                order: false,
            },
            accepts_predicate: false,
        });
        self
    }

    /// Configures a count sink that can optionally filter rows with a predicate.
    fn count_sink_with_predicate(mut self) -> Self {
        self.sink = Some(BuiltinSinkSpec {
            accumulator: BuiltinSinkAccumulator::Count,
            demand: BuiltinSinkDemand::All {
                value: BuiltinSinkValueNeed::None,
                order: false,
            },
            accepts_predicate: true,
        });
        self
    }

    /// Configures a select-one sink that picks the first or last row.
    fn select_one_sink(mut self, position: BuiltinSelectionPosition) -> Self {
        self.sink = Some(BuiltinSinkSpec {
            accumulator: BuiltinSinkAccumulator::SelectOne(position),
            demand: match position {
                BuiltinSelectionPosition::First => BuiltinSinkDemand::First {
                    value: BuiltinSinkValueNeed::Whole,
                },
                BuiltinSelectionPosition::Last => BuiltinSinkDemand::Last {
                    value: BuiltinSinkValueNeed::Whole,
                },
            },
            accepts_predicate: false,
        });
        self
    }

    /// Configures a numeric sink (sum, avg, min, max) that needs numeric values from every row.
    fn numeric_sink(mut self, reducer: BuiltinNumericReducer) -> Self {
        self.sink = Some(BuiltinSinkSpec {
            accumulator: BuiltinSinkAccumulator::Numeric,
            demand: BuiltinSinkDemand::All {
                value: BuiltinSinkValueNeed::Numeric,
                order: false,
            },
            accepts_predicate: false,
        });
        self.numeric_reducer = Some(reducer);
        self
    }

    /// Configures an approximate distinct-count sink.
    fn approx_distinct_sink(mut self) -> Self {
        self.sink = Some(BuiltinSinkSpec {
            accumulator: BuiltinSinkAccumulator::ApproxDistinct,
            demand: BuiltinSinkDemand::All {
                value: BuiltinSinkValueNeed::Whole,
                order: false,
            },
            accepts_predicate: false,
        });
        self
    }

    /// Attaches a keyed reducer kind (group, count, or index).
    fn keyed_reducer(mut self, reducer: BuiltinKeyedReducer) -> Self {
        self.keyed_reducer = Some(reducer);
        self
    }

    /// Attaches an arg-extreme sink kind (`max_by` / `min_by`).
    fn arg_extreme_sink(mut self, sink: BuiltinArgExtremeSink) -> Self {
        self.arg_extreme_sink = Some(sink);
        self
    }

    /// Attaches a predicate terminal sink kind.
    fn predicate_sink(mut self, sink: BuiltinPredicateSink) -> Self {
        self.predicate_sink = Some(sink);
        self
    }

    /// Attaches a membership terminal sink kind.
    fn membership_sink(mut self, sink: BuiltinMembershipSink) -> Self {
        self.membership_sink = Some(sink);
        self
    }

    /// Attaches direct array-child selector metadata.
    fn array_selector(mut self, selector: BuiltinArraySelector) -> Self {
        self.array_selector = Some(selector);
        self
    }

    /// Attaches a stage-merge rule so adjacent identical stages can be collapsed.
    fn stage_merge(mut self, merge: BuiltinStageMerge) -> Self {
        self.stage_merge = Some(merge);
        self
    }

    /// Attaches an algebraic cancellation rule for this builtin.
    fn cancellation(mut self, cancellation: BuiltinCancellation) -> Self {
        self.cancellation = Some(cancellation);
        self
    }

    /// Marks this builtin as having a structural index backend.
    fn structural(mut self, structural: BuiltinStructural) -> Self {
        self.structural = Some(structural);
        self
    }

    /// Overrides the default relative cost estimate.
    fn cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }

    /// Sets the demand-propagation law for pipeline planning.
    fn demand_law(mut self, law: BuiltinDemandLaw) -> Self {
        self.demand_law = law;
        self
    }

    /// Sets the materialization policy.
    fn materialization(mut self, m: BuiltinPipelineMaterialization) -> Self {
        self.materialization = m;
        self
    }

    /// Sets the semantic streaming-boundary class for this builtin.
    fn streaming_boundary(mut self, boundary: BuiltinStreamingBoundary) -> Self {
        self.streaming_boundary = boundary;
        self
    }

    /// Sets the cardinality/cost pipeline shape annotation.
    fn pipeline_shape(mut self, s: BuiltinPipelineShape) -> Self {
        self.pipeline_shape = Some(s);
        self
    }

    /// Sets the ordering-effect annotation.
    fn order_effect(mut self, o: BuiltinPipelineOrderEffect) -> Self {
        self.order_effect = Some(o);
        self
    }

    /// Sets the physical stage lowering strategy.
    fn lowering(mut self, l: BuiltinPipelineLowering) -> Self {
        self.lowering = Some(l);
        self
    }

    /// Marks this builtin as element-wise vectorisable.
    fn element(mut self) -> Self {
        self.is_element = true;
        self
    }

    /// Opts this builtin out of the path-receiver scalar-unwrap rewrite. The
    /// pipeline-streaming lowering remains the canonical path for
    /// `$.path.method()`, even when category/cardinality would otherwise be
    /// eligible for direct dispatch.
    fn never_unwrap(mut self) -> Self {
        self.never_unwrap = true;
        self
    }

    /// Marks this builtin as a stream source boundary.
    fn stream_source(mut self) -> Self {
        self.stream_source = true;
        self
    }
}

impl BuiltinMethod {
    /// Returns the full capability descriptor for this builtin.
    /// Called by the pipeline planner and VM to query cardinality, cost, and feature flags.
    #[inline]
    pub fn spec(self) -> BuiltinSpec {
        macro_rules! spec_arm {
            ( $( $variant:ident ),* $(,)? ) => {
                match self {
                    $( Self::$variant => {
                        debug_assert_eq!(
                            <defs::$variant as builtin::Builtin>::METHOD,
                            Self::$variant
                        );
                        <defs::$variant as builtin::Builtin>::spec()
                    }, )*
                }
            };
        }
        let spec = crate::for_each_builtin!(spec_arm);
        // Apply per-method cancellation override (defaults to None for most methods).
        macro_rules! cancel_arm {
            ( $( $variant:ident ),* $(,)? ) => {
                match self {
                    $( Self::$variant => <defs::$variant as builtin::Builtin>::cancellation(), )*
                }
            };
        }
        match crate::for_each_builtin!(cancel_arm) {
            Some(c) => spec.cancellation(c),
            None => spec,
        }
    }
}

impl BuiltinCall {
    /// Constructs a `BuiltinCall` from a resolved method and its decoded arguments.
    #[inline]
    pub fn new(method: BuiltinMethod, args: BuiltinArgs) -> Self {
        Self { method, args }
    }

    /// Returns this call's registry id.
    #[inline]
    pub(crate) fn id(&self) -> registry::BuiltinId {
        registry::BuiltinId::from_method(self.method)
    }

    /// Returns the direct borrowed-view call family for this call, if any.
    #[inline]
    pub(crate) fn direct_view_call(&self) -> Option<registry::BuiltinDirectViewCall> {
        registry::direct_view_call(self.id(), &self.args)
    }

    /// Returns true when this call can be evaluated as a scalar borrowed-view projection.
    #[inline]
    pub(crate) fn is_direct_view_scalar_call(&self) -> bool {
        self.direct_view_call() == Some(registry::BuiltinDirectViewCall::ScalarValue)
    }

    /// Returns true when this call can be applied directly to raw JSON scalar bytes.
    #[inline]
    pub(crate) fn is_raw_json_scalar_call(&self) -> bool {
        registry::raw_json_scalar_call(self.id(), &self.args)
    }

    /// Returns the direct raw JSON scalar operation for this call, if any.
    #[inline]
    pub(crate) fn raw_json_scalar(&self) -> Option<BuiltinRawJsonScalar> {
        registry::raw_json_scalar(self.id(), &self.args)
    }

    /// Returns true when this call can run as a borrowed-view projection.
    #[inline]
    pub(crate) fn is_view_projection(&self) -> bool {
        registry::view_projection(self.id())
    }

    /// Returns true when this view projection returns an owned value.
    #[inline]
    pub(crate) fn view_projection_returns_owned(&self) -> bool {
        registry::view_projection_returns_owned(self.id(), &self.args)
    }

    /// Returns array-selector metadata for this call, if it is a selector.
    #[inline]
    pub(crate) fn array_selector(&self) -> Option<BuiltinArraySelector> {
        registry::array_selector(self.id())
    }

    /// Returns composed-pipeline execution metadata for this call, if any.
    #[inline]
    pub(crate) fn composed_stage(&self) -> Option<registry::BuiltinComposedStage> {
        registry::composed_builtin_stage(self.id())
    }

    /// Returns stage cancellation metadata for this call, if any.
    #[inline]
    pub(crate) fn cancellation(&self) -> Option<BuiltinCancellation> {
        registry::cancellation(self.id())
    }

    /// Returns cardinality metadata for this call, if known.
    #[inline]
    pub(crate) fn cardinality(&self) -> Option<BuiltinCardinality> {
        registry::builtin_cardinality(self.id())
    }

    /// Returns true when this call preserves one output row for each input row.
    #[inline]
    pub(crate) fn preserves_cardinality(&self) -> bool {
        self.cardinality()
            .is_some_and(|cardinality| matches!(cardinality, BuiltinCardinality::OneToOne))
    }

    /// Returns true when this view projection can be delayed by the planner.
    #[inline]
    pub(crate) fn is_stage_delayable_view_projection(&self) -> bool {
        registry::stage_delayable_view_projection(self.id())
    }

    /// Returns true when this call projects object items from a borrowed view.
    #[inline]
    pub(crate) fn is_direct_object_items_call(&self) -> bool {
        self.direct_object_items_projection().is_some()
    }

    /// Returns object-items projection metadata for this call, if it can be
    /// executed directly over a borrowed view.
    #[inline]
    pub(crate) fn direct_object_items_projection(&self) -> Option<BuiltinViewObjectProjection> {
        registry::view_object_items_projection_call(self.id(), &self.args)
    }

    /// Returns true if applying this builtin twice is equivalent to applying it once.
    /// The pipeline optimizer uses this to eliminate redundant stages.
    #[inline]
    pub fn is_idempotent(&self) -> bool {
        registry::is_idempotent(registry::BuiltinId::from_method(self.method))
    }

    /// Executes the builtin against `recv` with its pre-decoded static arguments.
    /// Returns `None` when the receiver type is not applicable (caller may fall back).
    /// For methods that can return errors, prefer [`BuiltinCall::try_apply`].
    pub fn apply(&self, recv: &Val) -> Option<Val> {
        macro_rules! apply_or_recv {
            ($expr:expr) => {
                return Some($expr.unwrap_or_else(|| recv.clone()))
            };
        }
        if let Some(value) = registry::apply_scalar_hook(self.method, &self.args, recv) {
            return Some(value);
        }
        match (self.method, &self.args) {
            (BuiltinMethod::ByteLen, BuiltinArgs::None)
            | (BuiltinMethod::IsBlank, BuiltinArgs::None)
            | (BuiltinMethod::IsNumeric, BuiltinArgs::None)
            | (BuiltinMethod::IsAlpha, BuiltinArgs::None)
            | (BuiltinMethod::IsAscii, BuiltinArgs::None)
            | (BuiltinMethod::ToNumber, BuiltinArgs::None)
            | (BuiltinMethod::ToBool, BuiltinArgs::None) => {
                apply_or_recv!(str_no_arg_scalar_val_apply(self.method, recv))
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
            (BuiltinMethod::Collect, BuiltinArgs::None) => return Some(collect_apply(recv)),
            (BuiltinMethod::FromJson, BuiltinArgs::None) => return from_json_apply(recv),
            (BuiltinMethod::Ceil, BuiltinArgs::None)
            | (BuiltinMethod::Floor, BuiltinArgs::None)
            | (BuiltinMethod::Round, BuiltinArgs::None)
            | (BuiltinMethod::Abs, BuiltinArgs::None) => {
                return numeric_no_arg_scalar_val_apply(self.method, recv)
            }
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
            (BuiltinMethod::Join, BuiltinArgs::Str(sep)) => return join_apply(recv, sep),
            (BuiltinMethod::Enumerate, BuiltinArgs::None) => return enumerate_apply(recv),
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
            (BuiltinMethod::Pivot, BuiltinArgs::StrVec(fields)) => {
                apply_or_recv!(pivot_fields_apply(recv, fields).ok())
            }
            (BuiltinMethod::Has, BuiltinArgs::Str(k)) => {
                apply_or_recv!(has_apply(recv, k))
            }
            (BuiltinMethod::HasAll, BuiltinArgs::Val(v)) => {
                apply_or_recv!(has_all_apply(recv, v))
            }
            (BuiltinMethod::HasAll, BuiltinArgs::StrVec(keys)) => {
                apply_or_recv!(has_all_keys_apply(recv, keys))
            }
            (BuiltinMethod::HasKey, BuiltinArgs::Str(k)) => return Some(has_key_apply(recv, k)),
            (BuiltinMethod::GetPath, BuiltinArgs::Str(p)) => {
                apply_or_recv!(get_path_apply(recv, p))
            }
            (BuiltinMethod::GetPath, BuiltinArgs::Path(path)) => {
                return Some(get_path_impl(recv, path))
            }
            (BuiltinMethod::HasPath, BuiltinArgs::Str(p)) => {
                apply_or_recv!(has_path_apply(recv, p))
            }
            (BuiltinMethod::HasPath, BuiltinArgs::Path(path)) => {
                return Some(Val::Bool(!get_path_impl(recv, path).is_null()))
            }
            (BuiltinMethod::DelPath, BuiltinArgs::Str(p)) => {
                apply_or_recv!(del_path_apply(recv, p))
            }
            (BuiltinMethod::DelPaths, BuiltinArgs::PathList(paths)) => {
                let mut out = recv.clone();
                for path in paths {
                    out = del_path_impl(out, path);
                }
                return Some(out);
            }
            (BuiltinMethod::SetPath, BuiltinArgs::PathVal { path, value }) => {
                return Some(set_path_impl(recv.clone(), path, value.clone()));
            }
            (BuiltinMethod::FlattenKeys, BuiltinArgs::Str(p)) => {
                apply_or_recv!(flatten_keys_apply(recv, p))
            }
            (BuiltinMethod::UnflattenKeys, BuiltinArgs::Str(p)) => {
                apply_or_recv!(unflatten_keys_apply(recv, p))
            }
            (BuiltinMethod::StartsWith, BuiltinArgs::Str(p))
            | (BuiltinMethod::EndsWith, BuiltinArgs::Str(p))
            | (BuiltinMethod::Matches, BuiltinArgs::Str(p))
            | (BuiltinMethod::IndexOf, BuiltinArgs::Str(p))
            | (BuiltinMethod::LastIndexOf, BuiltinArgs::Str(p)) => {
                apply_or_recv!(str_arg_scalar_val_apply(self.method, recv, p))
            }
            (BuiltinMethod::StripPrefix, BuiltinArgs::Str(p)) => {
                apply_or_recv!(strip_prefix_apply(recv, p))
            }
            (BuiltinMethod::StripSuffix, BuiltinArgs::Str(p)) => {
                apply_or_recv!(strip_suffix_apply(recv, p))
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
            (BuiltinMethod::Indent, BuiltinArgs::Str(prefix)) => {
                apply_or_recv!(indent_with_prefix_apply(recv, prefix.as_ref()))
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

    /// Like [`BuiltinCall::apply`] but propagates evaluation errors (regex compilation,
    /// window-size-zero, JSON parse failures, etc.) as `EvalError`.
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
            (BuiltinMethod::Sort, BuiltinArgs::None) => sort_apply(recv.clone()).map(Some),
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

    /// Decodes static (non-lambda) arguments for `method` and constructs a `BuiltinCall`.
    /// `eval_arg` evaluates positional argument expressions; `ident_arg` extracts bare
    /// identifier names (used to accept field names without quote syntax).
    /// Returns `Ok(None)` for methods that require lambda arguments (handled separately).
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
            BuiltinMethod::Take | BuiltinMethod::Skip => {
                Self::new(method, BuiltinArgs::Usize(args.usize(0)?))
            }
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
            BuiltinMethod::SetPath => {
                let path = args.str(0)?;
                Self::new(
                    method,
                    BuiltinArgs::PathVal {
                        path: parse_path_segs(path.as_ref()).into(),
                        value: args.val(1)?,
                    },
                )
            }
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
            // `missing(...keys)`: multi-arg form returns the array of
            // absent keys; single-arg keeps the legacy boolean. Listed
            // BEFORE the catch-all `Missing` so the multi-arg case wins.
            BuiltinMethod::Missing if arg_len >= 2 => {
                let mut keys = Vec::with_capacity(arg_len);
                for i in 0..arg_len {
                    keys.push(args.str(i)?);
                }
                Self::new(method, BuiltinArgs::StrVec(keys))
            }
            BuiltinMethod::GetPath | BuiltinMethod::HasPath => {
                let path = args.str(0)?;
                Self::new(
                    method,
                    BuiltinArgs::Path(parse_path_segs(path.as_ref()).into()),
                )
            }
            BuiltinMethod::DelPaths => {
                let mut paths = Vec::with_capacity(arg_len);
                for idx in 0..arg_len {
                    paths.push(parse_path_segs(args.str(idx)?.as_ref()).into());
                }
                Self::new(method, BuiltinArgs::PathList(paths))
            }
            BuiltinMethod::HasAll => Self::new(method, BuiltinArgs::Val(args.val(0)?)),
            BuiltinMethod::Has
            | BuiltinMethod::HasKey
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
            BuiltinMethod::ToCsv | BuiltinMethod::ToTsv if arg_len > 0 => {
                Self::new(method, BuiltinArgs::StrVec(args.str_vec(0)?))
            }
            BuiltinMethod::Pick | BuiltinMethod::Omit => {
                let mut keys = Vec::with_capacity(arg_len);
                for idx in 0..arg_len {
                    keys.push(args.str(idx)?);
                }
                Self::new(method, BuiltinArgs::StrVec(keys))
            }
            BuiltinMethod::Repeat => Self::new(method, BuiltinArgs::Usize(args.usize(0)?)),
            BuiltinMethod::Indent => {
                if arg_len > 0 {
                    if let Some(prefix) = args.str_lit(0) {
                        Self::new(method, BuiltinArgs::Str(prefix))
                    } else {
                        Self::new(method, BuiltinArgs::Usize(args.usize(0)?))
                    }
                } else {
                    Self::new(method, BuiltinArgs::Usize(2))
                }
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

    /// Decodes static arguments from parser argument forms. Named arguments
    /// stay on the semantic fallback path for builtins such as `pick` where
    /// names mean output aliases rather than field-list keys.
    pub fn from_static_ast_args<E>(
        method: BuiltinMethod,
        name: &str,
        args: &[crate::parse::ast::Arg],
        eval_arg: E,
    ) -> Result<Option<Self>, EvalError>
    where
        E: FnMut(usize) -> Result<Option<Val>, EvalError>,
    {
        use crate::parse::ast::{Arg, Expr};

        let has_named_arg = args.iter().any(|arg| matches!(arg, Arg::Named(_, _)));
        if method == BuiltinMethod::Pick && has_named_arg {
            return Ok(None);
        }

        Self::from_static_args(method, name, args.len(), eval_arg, |idx| {
            match args.get(idx) {
                Some(Arg::Pos(Expr::Ident(value))) => Some(Arc::from(value.as_str())),
                _ => None,
            }
        })
    }

    /// Attempts to construct a `BuiltinCall` from AST arguments that are all compile-time
    /// literals. Non-literal or lambda arguments cause `None` to be returned, falling back
    /// to runtime evaluation.
    pub fn from_literal_ast_args(name: &str, args: &[crate::parse::ast::Arg]) -> Option<Self> {
        use crate::parse::ast::{Arg, ArrayElem, Expr, ObjField};

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

        if method == BuiltinMethod::Remove {
            return match args {
                [Arg::Pos(expr)] => Some(Self::new(method, BuiltinArgs::Val(literal_val(expr)?))),
                _ => None,
            };
        }

        if method == BuiltinMethod::HasAll {
            return match args {
                [Arg::Pos(Expr::Array(elems))] => {
                    let mut keys = Vec::with_capacity(elems.len());
                    for elem in elems {
                        let ArrayElem::Expr(expr) = elem else {
                            return None;
                        };
                        keys.push(Arc::from(crate::util::val_to_key(&literal_val(expr)?)));
                    }
                    Some(Self::new(method, BuiltinArgs::StrVec(keys)))
                }
                [Arg::Pos(expr)] => Some(Self::new(method, BuiltinArgs::Val(literal_val(expr)?))),
                _ => None,
            };
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

    /// Like [`BuiltinCall::from_literal_ast_args`] but also requires the method to be
    /// accepted as a builtin-call pipeline stage, returning `None` otherwise.
    pub fn from_pipeline_literal_args(name: &str, args: &[crate::parse::ast::Arg]) -> Option<Self> {
        let call = Self::from_literal_ast_args(name, args)?;
        crate::builtins::registry::pipeline_builtin_call_stage(
            crate::builtins::registry::BuiltinId::from_method(call.method),
        )
        .then_some(call)
    }

    /// Evaluates this builtin directly on a zero-copy `JsonView` without materialising a `Val`.
    /// Only works for view-scalar methods; returns `None` for all other builtins.
    pub fn try_apply_json_view(&self, recv: crate::util::JsonView<'_>) -> Option<Val> {
        registry::apply_json_view_scalar_hook(self.method, &self.args, recv)
    }
}

/// Applies a zero-argument numeric scalar method (`ceil`, `floor`, `round`, `abs`) to a `JsonView`.
#[inline]
fn numeric_no_arg_scalar_apply(
    method: BuiltinMethod,
    recv: crate::util::JsonView<'_>,
) -> Option<Val> {
    match (method, recv) {
        (
            BuiltinMethod::Ceil | BuiltinMethod::Floor | BuiltinMethod::Round,
            crate::util::JsonView::Int(n),
        ) => Some(Val::Int(n)),
        (
            BuiltinMethod::Ceil | BuiltinMethod::Floor | BuiltinMethod::Round,
            crate::util::JsonView::UInt(n),
        ) => Some(uint_to_val(n)),
        (BuiltinMethod::Ceil, crate::util::JsonView::Float(f)) => Some(Val::Int(f.ceil() as i64)),
        (BuiltinMethod::Floor, crate::util::JsonView::Float(f)) => Some(Val::Int(f.floor() as i64)),
        (BuiltinMethod::Round, crate::util::JsonView::Float(f)) => Some(Val::Int(f.round() as i64)),
        (BuiltinMethod::Abs, crate::util::JsonView::Int(n)) => Some(Val::Int(n.wrapping_abs())),
        (BuiltinMethod::Abs, crate::util::JsonView::UInt(n)) => Some(uint_to_val(n)),
        (BuiltinMethod::Abs, crate::util::JsonView::Float(f)) => Some(Val::Float(f.abs())),
        _ => None,
    }
}

/// Applies a zero-argument numeric scalar method to a materialised `Val`.
#[inline]
fn numeric_no_arg_scalar_val_apply(method: BuiltinMethod, recv: &Val) -> Option<Val> {
    numeric_no_arg_scalar_apply(method, crate::util::JsonView::from_val(recv))
}

/// Converts a `u64` to `Val::Int` if it fits, otherwise `Val::Float`.
#[inline]
fn uint_to_val(n: u64) -> Val {
    if n <= i64::MAX as u64 {
        Val::Int(n as i64)
    } else {
        Val::Float(n as f64)
    }
}

/// Applies a zero-argument string scalar method to a `&str`, returning the result as a `Val`.
#[inline]
fn str_no_arg_scalar_apply(method: BuiltinMethod, value: &str) -> Option<Val> {
    match method {
        BuiltinMethod::Upper => {
            if value.is_ascii() {
                let mut buf = value.to_owned();
                buf.make_ascii_uppercase();
                Some(Val::Str(Arc::from(buf)))
            } else {
                Some(Val::Str(Arc::from(value.to_uppercase())))
            }
        }
        BuiltinMethod::Lower => {
            if value.is_ascii() {
                let mut buf = value.to_owned();
                buf.make_ascii_lowercase();
                Some(Val::Str(Arc::from(buf)))
            } else {
                Some(Val::Str(Arc::from(value.to_lowercase())))
            }
        }
        BuiltinMethod::Trim => Some(Val::Str(Arc::from(value.trim()))),
        BuiltinMethod::TrimLeft => Some(Val::Str(Arc::from(value.trim_start()))),
        BuiltinMethod::TrimRight => Some(Val::Str(Arc::from(value.trim_end()))),
        BuiltinMethod::ByteLen => Some(Val::Int(value.len() as i64)),
        BuiltinMethod::IsBlank => Some(Val::Bool(value.chars().all(|c| c.is_whitespace()))),
        BuiltinMethod::IsNumeric => Some(Val::Bool(
            !value.is_empty() && value.chars().all(|c| c.is_ascii_digit()),
        )),
        BuiltinMethod::IsAlpha => Some(Val::Bool(
            !value.is_empty() && value.chars().all(|c| c.is_alphabetic()),
        )),
        BuiltinMethod::IsAscii => Some(Val::Bool(value.is_ascii())),
        BuiltinMethod::ToNumber => {
            if let Ok(i) = value.parse::<i64>() {
                return Some(Val::Int(i));
            }
            if let Ok(f) = value.parse::<f64>() {
                return Some(Val::Float(f));
            }
            Some(Val::Null)
        }
        BuiltinMethod::ToBool => Some(match value {
            "true" => Val::Bool(true),
            "false" => Val::Bool(false),
            _ => Val::Null,
        }),
        BuiltinMethod::ParseInt => Some(parse_int_str(value)),
        BuiltinMethod::ParseFloat => Some(parse_float_str(value)),
        BuiltinMethod::ParseBool => Some(parse_bool_str(value)),
        _ => None,
    }
}

/// Applies a zero-argument string scalar method to a `Val`, extracting the string slice first.
#[inline]
fn str_no_arg_scalar_val_apply(method: BuiltinMethod, recv: &Val) -> Option<Val> {
    str_no_arg_scalar_apply(method, recv.as_str_ref()?)
}

/// Applies a single-string-argument scalar method to a `&str` value with the argument.
#[inline]
fn str_arg_scalar_apply(method: BuiltinMethod, value: &str, arg: &str) -> Option<Val> {
    match method {
        BuiltinMethod::StartsWith => Some(Val::Bool(value.starts_with(arg))),
        BuiltinMethod::EndsWith => Some(Val::Bool(value.ends_with(arg))),
        BuiltinMethod::Matches => Some(Val::Bool(value.contains(arg))),
        BuiltinMethod::IndexOf => Some(str_index_of(value, arg, false)),
        BuiltinMethod::LastIndexOf => Some(str_index_of(value, arg, true)),
        _ => None,
    }
}

/// Applies a single-string-argument scalar method to a `Val` receiver.
#[inline]
fn str_arg_scalar_val_apply(method: BuiltinMethod, recv: &Val, arg: &str) -> Option<Val> {
    str_arg_scalar_apply(method, recv.as_str_ref()?, arg)
}

#[inline]
fn str_vec_arg_scalar_apply(method: BuiltinMethod, value: &str, args: &[Arc<str>]) -> Option<Val> {
    match method {
        BuiltinMethod::ContainsAny => Some(Val::Bool(
            args.iter().any(|needle| value.contains(needle.as_ref())),
        )),
        BuiltinMethod::ContainsAll => Some(Val::Bool(
            args.iter().all(|needle| value.contains(needle.as_ref())),
        )),
        _ => None,
    }
}

/// Returns the character index of `needle` in `value`; uses `rfind` when `last` is true.
/// Returns `Val::Int(-1)` when not found.
#[inline]
fn str_index_of(value: &str, needle: &str, last: bool) -> Val {
    let offset = if last {
        value.rfind(needle)
    } else {
        value.find(needle)
    };
    match offset {
        Some(i) => Val::Int(value[..i].chars().count() as i64),
        None => Val::Int(-1),
    }
}

/// Extracts the logical length from a `JsonView` (char count for strings, element count for collections).
#[inline]
fn json_view_len(recv: crate::util::JsonView<'_>) -> Option<i64> {
    match recv {
        crate::util::JsonView::Str(s) => Some(s.chars().count() as i64),
        crate::util::JsonView::ArrayLen(n) | crate::util::JsonView::ObjectLen(n) => Some(n as i64),
        _ => None,
    }
}

/// Extracts the JSON type name from a `JsonView` tag without materialising the receiver.
#[inline]
fn json_view_type_name(recv: crate::util::JsonView<'_>) -> &'static str {
    match recv {
        crate::util::JsonView::Null => "null",
        crate::util::JsonView::Bool(_) => "bool",
        crate::util::JsonView::Int(_)
        | crate::util::JsonView::UInt(_)
        | crate::util::JsonView::Float(_) => "number",
        crate::util::JsonView::Str(_) => "string",
        crate::util::JsonView::ArrayLen(_) => "array",
        crate::util::JsonView::ObjectLen(_) => "object",
    }
}

/// Extracts a `&str` from a `JsonView::Str` variant; returns `None` for other variants.
#[inline]
fn json_view_str(recv: crate::util::JsonView<'_>) -> Option<&str> {
    match recv {
        crate::util::JsonView::Str(s) => Some(s),
        _ => None,
    }
}

/// Main dispatch entry point called by the tree-walking evaluator.
///
/// Resolves `name` to a [`BuiltinMethod`], decodes arguments, and invokes the
/// appropriate algorithm body. Three evaluator closures supply the backend's
/// expression evaluation strategy:
/// - `eval_arg`: evaluates a standalone argument expression.
/// - `eval_item`: evaluates a lambda body with `@` bound to an array element.
/// - `eval_pair`: evaluates a two-parameter comparator lambda (`sort` with a custom comparator).
pub(crate) fn eval_builtin_method<F, G, H>(
    recv: Val,
    name: &str,
    args: &[crate::parse::ast::Arg],
    mut eval_arg: F,
    mut eval_item: G,
    mut eval_pair: H,
) -> Result<Val, EvalError>
where
    F: FnMut(&crate::parse::ast::Arg) -> Result<Val, EvalError>,
    G: FnMut(&Val, &crate::parse::ast::Arg) -> Result<Val, EvalError>,
    H: FnMut(&Val, &Val, &crate::parse::ast::Arg) -> Result<Val, EvalError>,
{
    use crate::parse::ast::{Arg, Expr, ObjField};

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
        | BuiltinMethod::ApproxCountDistinct
        | BuiltinMethod::ZipShape
        | BuiltinMethod::GroupShape
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
        BuiltinMethod::Find | BuiltinMethod::FindFirst => {
            return find_first_apply(recv, args.len(), |item, idx| eval_item(item, &args[idx]));
        }
        BuiltinMethod::FindAll => {
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
            if let Some(cond_arg) = args.get(1) {
                let eval_cell = std::cell::RefCell::new(eval_item);
                return rec_cond_apply(
                    recv,
                    |value| eval_cell.borrow_mut()(&value, arg),
                    |value| eval_cell.borrow_mut()(value, cond_arg),
                );
            }
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
            // No-arg form: parallel-array interleave. Receiver is an
            // object whose values are arrays of the same length; emit
            // one row per index with each key holding the array's
            // i-th element. Non-array values are broadcast.
            if args.is_empty() {
                return zip_shape_obj_apply(&recv)
                    .ok_or_else(|| EvalError("zip_shape: expected object receiver".into()));
            }
            // Object-literal sugar: `zip_shape({a, b})` ≡ `zip_shape(a, b)`.
            // The single `{a, b}` arg is evaluated as an object literal
            // against the receiver, then dispatched through the no-arg
            // parallel-array interleave.
            if args.len() == 1 {
                if let Arg::Pos(Expr::Object(fields)) = &args[0] {
                    let all_short = fields
                        .iter()
                        .all(|f| matches!(f, crate::parse::ast::ObjField::Short(_)));
                    if all_short {
                        let obj = eval_arg(&args[0])?;
                        return zip_shape_obj_apply(&obj).ok_or_else(|| {
                            EvalError("zip_shape: expected object receiver".into())
                        });
                    }
                }
            }
            let mut names = Vec::with_capacity(args.len());
            for arg in args {
                let name: Arc<str> = match arg {
                    Arg::Named(n, _) => Arc::from(n.as_str()),
                    Arg::Pos(Expr::Ident(n)) => Arc::from(n.as_str()),
                    _ => {
                        return Err(EvalError(
                            "zip_shape: args must be `name: expr` or bare identifier".into(),
                        ))
                    }
                };
                names.push(name);
            }
            return zip_shape_apply(&recv, &names, |value, idx| eval_item(value, &args[idx]));
        }
        BuiltinMethod::GroupShape => {
            // No-arg form: group an array of objects by their
            // structural key set (the sorted keys joined with `,`).
            // Output is `{shape_key: [items]}`. Useful for partitioning
            // a heterogeneous collection by which keys each row has.
            if args.is_empty() {
                return group_shape_by_keys_apply(recv)
                    .ok_or_else(|| EvalError("group_shape: expected array".into()));
            }
            // 1-arg form: `group_shape(key_expr)` keys each element by the
            // projected value and emits `{key_value: [items]}` with the
            // original element preserved as the bucket value.
            if args.len() == 1 {
                let key_arg = &args[0];
                return group_shape_apply(recv, |value, idx| {
                    if idx == 0 {
                        eval_item(&value, key_arg)
                    } else {
                        Ok(value)
                    }
                });
            }
            let key_arg = &args[0];
            let shape_arg = &args[1];
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
        BuiltinMethod::Take | BuiltinMethod::Skip => {
            BuiltinCall::new(method, BuiltinArgs::Usize(i64_arg!(0)?.max(0) as usize))
        }
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
        BuiltinMethod::ParseInt if !args.is_empty() => {
            // `parse_int(radix)` — package the radix as a `Usize` so the
            // trait dispatch in `BuiltinCall::apply_args` (defs::ParseInt)
            // picks it up. Falls through to base-10 no-arg form when the
            // arg is missing.
            let radix = i64_arg!(0)?;
            BuiltinCall::new(method, BuiltinArgs::Usize(radix.max(0) as usize))
        }
        BuiltinMethod::ToCsv | BuiltinMethod::ToTsv if !args.is_empty() => {
            // `to_csv(headers)` / `to_tsv(headers)` — headers must be a
            // string array; package as `BuiltinArgs::StrVec`.
            let headers = str_vec_arg!(0)?;
            BuiltinCall::new(method, BuiltinArgs::StrVec(headers))
        }
        BuiltinMethod::Remove => match args.first() {
            // Treat anything that touches the current item (`@.x > 0`,
            // `@ != null`, comparison/binop/chain on `@`) as a per-element
            // predicate — same path as an explicit lambda. The original
            // dispatch only matched `Expr::Lambda`, so `@`-form predicates
            // fell through to the value-equality path and silently kept
            // every element.
            Some(Arg::Pos(Expr::Lambda { .. })) | Some(Arg::Named(_, Expr::Lambda { .. })) => {
                return remove_predicate_apply(recv, |item| eval_item(item, &args[0]));
            }
            Some(arg) if arg_uses_current(arg) => {
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
        BuiltinMethod::Pivot
            if args.len() >= 2
                && args.iter().all(|arg| {
                    matches!(arg, Arg::Pos(Expr::Str(_)) | Arg::Named(_, Expr::Str(_)))
                }) =>
        {
            let fields = args
                .iter()
                .map(|arg| match arg {
                    Arg::Pos(Expr::Str(value)) | Arg::Named(_, Expr::Str(value)) => {
                        Ok(Arc::from(value.as_str()))
                    }
                    _ => unreachable!("pivot literal guard checked above"),
                })
                .collect::<Result<Vec<_>, EvalError>>()?;
            BuiltinCall::new(method, BuiltinArgs::StrVec(fields))
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
        // `missing(...keys)` — variadic key-existence audit. Multi-key
        // form returns the array of absent keys; single-key form keeps the
        // legacy boolean.
        BuiltinMethod::Missing if args.len() >= 2 => {
            let keys = (0..args.len())
                .map(|i| str_arg!(i))
                .collect::<Result<Vec<_>, _>>()?;
            BuiltinCall::new(method, BuiltinArgs::StrVec(keys))
        }
        BuiltinMethod::GetPath | BuiltinMethod::HasPath => BuiltinCall::new(
            method,
            BuiltinArgs::Path(parse_path_segs(str_arg!(0)?.as_ref()).into()),
        ),
        BuiltinMethod::HasAll => BuiltinCall::new(method, BuiltinArgs::Val(arg_val!(0)?)),
        BuiltinMethod::Has
        | BuiltinMethod::HasKey
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
            // `indent("> ")` accepts a string prefix argument. Detect the
            // string-literal shape before falling through to the integer
            // count coercion used by both `repeat` and the spaces-form of
            // `indent`.
            let prefix_arg = if matches!(method, BuiltinMethod::Indent) && !args.is_empty() {
                match &args[0] {
                    Arg::Pos(Expr::Str(s)) => Some(Arc::<str>::from(s.as_str())),
                    Arg::Named(_, Expr::Str(s)) => Some(Arc::<str>::from(s.as_str())),
                    _ => None,
                }
            } else {
                None
            };
            if let Some(prefix) = prefix_arg {
                BuiltinCall::new(method, BuiltinArgs::Str(prefix))
            } else {
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
        BuiltinMethod::SetPath => BuiltinCall::new(
            method,
            BuiltinArgs::PathVal {
                path: parse_path_segs(str_arg!(0)?.as_ref()).into(),
                value: arg_val!(1)?,
            },
        ),
        BuiltinMethod::DelPaths => {
            let mut paths = Vec::with_capacity(args.len());
            for idx in 0..args.len() {
                paths.push(parse_path_segs(str_arg!(idx)?.as_ref()).into());
            }
            BuiltinCall::new(method, BuiltinArgs::PathList(paths))
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

/// Returns `true` if `arg`'s expression tree references `Expr::Current`
/// (the `@` placeholder), indicating it should be treated as a per-element
/// predicate / projection rather than a literal value. Used by builtins
/// like `remove` whose semantics depend on whether the user supplied a
/// per-element predicate or a literal needle.
fn arg_uses_current(arg: &crate::parse::ast::Arg) -> bool {
    use crate::parse::ast::{Arg, Expr};
    fn walk(e: &Expr) -> bool {
        match e {
            Expr::Current => true,
            Expr::Lambda { .. } => false, // lambda introduces its own scope
            Expr::Chain(base, _) => walk(base),
            Expr::UnaryNeg(x) | Expr::Not(x) => walk(x),
            Expr::BinOp(l, _, r) => walk(l) || walk(r),
            Expr::Coalesce(l, r) => walk(l) || walk(r),
            Expr::IfElse { cond, then_, else_ } => walk(cond) || walk(then_) || walk(else_),
            Expr::Try { body, default } => walk(body) || walk(default),
            Expr::Cast { expr, .. } => walk(expr),
            Expr::FString(parts) => parts.iter().any(|p| match p {
                crate::parse::ast::FStringPart::Interp { expr, .. } => walk(expr),
                _ => false,
            }),
            Expr::Let { init, body, .. } => walk(init) || walk(body),
            // Comprehensions, match arms, patches, identifiers, literals,
            // root, and ident references do not transitively bind `@` in
            // the dispatch position we care about.
            _ => false,
        }
    }
    match arg {
        Arg::Pos(e) | Arg::Named(_, e) => walk(e),
    }
}

/// Convenience wrapper over [`eval_builtin_method`] for zero-argument builtins.
/// Panics (via `EvalError`) if any argument evaluation closure is unexpectedly invoked.
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

pub mod ops;

pub(crate) mod builtin;
pub(crate) mod defs;
pub(crate) mod helpers;
pub(crate) mod registry;

pub use ops::array::*;
pub use ops::collection::*;
pub use ops::misc::*;
pub use ops::path::*;
pub use ops::regex::*;
pub use ops::schema::*;
pub use ops::string::*;
