//! Builtin method catalog and shared algorithm implementations.
//!
//! All three execution backends (VM, pipeline, composed) dispatch here for
//! algorithm bodies. Each builtin exposes two primitives:
//! `*_one(item, eval)` for per-row work and `*_apply(items, eval)` for
//! buffered work. Streaming consumers call `*_one`; barrier consumers call
//! `*_apply`. This module owns the loop and truthy-check logic exactly once.

use crate::context::EvalError;
use crate::util::{cmp_vals, is_truthy, val_key, zip_arrays};
use crate::value::Val;
use indexmap::IndexMap;
use std::sync::Arc;

/// Pre-resolved method identifier. Carried by `CompiledCall` and pipeline
/// plan nodes so method dispatch is an O(1) integer match, not a string hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BuiltinMethod {
    // ── Object / structural inspection ────────────────────────────────────
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

    // ── Numeric aggregates ─────────────────────────────────────────────────
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

    // ── Streaming / array transforms ──────────────────────────────────────
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
    /// Alias of `find_first`.
    FindOne,
    /// Counts approximate distinct values using a HyperLogLog-style sketch.
    ApproxCountDistinct,
    /// Produces a running accumulation using the lambda.
    Accumulate,
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

    // ── Object transforms ──────────────────────────────────────────────────
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

    // ── Path operations ────────────────────────────────────────────────────
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

    // ── Serialisation ──────────────────────────────────────────────────────
    /// Serialises an array/object to CSV text.
    ToCsv,
    /// Serialises an array/object to TSV text.
    ToTsv,

    // ── Miscellaneous scalar helpers ───────────────────────────────────────
    /// Returns the receiver if non-null; otherwise returns the argument.
    Or,
    /// Returns true if the object contains the given key.
    Has,
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

    // ── Numeric / math ─────────────────────────────────────────────────────
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

    // ── String transforms ──────────────────────────────────────────────────
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

    // ── Relational ─────────────────────────────────────────────────────────
    /// Performs an inner equi-join of two arrays of objects on matching key fields.
    EquiJoin,

    /// Sentinel returned by `from_name` when the method string is unrecognised.
    Unknown,
}

impl BuiltinMethod {
    /// Resolves a method name string to the corresponding `BuiltinMethod` variant.
    /// Returns [`BuiltinMethod::Unknown`] when the name is not registered.
    pub fn from_name(name: &str) -> Self {
        crate::builtin_registry::by_name(name)
            .and_then(|id| id.method())
            .unwrap_or(Self::Unknown)
    }

    /// Returns true when the method requires a lambda expression as its first argument.
    /// The pipeline planner uses this to distinguish element vs. expression stages.
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

/// Statically-typed argument payload stored inside a [`BuiltinCall`].
/// Each variant corresponds to the argument signature of a group of builtins,
/// enabling argument decoding without heap allocation at call time.
#[derive(Debug, Clone)]
pub enum BuiltinArgs {
    /// No arguments.
    None,
    /// A single string argument (field name, separator, pattern, etc.).
    Str(Arc<str>),
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
    /// View-stage lowering target, if the builtin maps to one of the view stages.
    pub view_stage: Option<BuiltinViewStage>,
    /// Sink (terminal aggregation) descriptor, present for reducing builtins.
    pub sink: Option<BuiltinSinkSpec>,
    /// Keyed reducer kind (group/count/index), used for grouped output planning.
    pub keyed_reducer: Option<BuiltinKeyedReducer>,
    /// Numeric reducer kind, used by the numeric sink path.
    pub numeric_reducer: Option<BuiltinNumericReducer>,
    /// How adjacent stages of the same kind can be merged (e.g. `take(3).take(2)` → `take(2)`).
    pub stage_merge: Option<BuiltinStageMerge>,
    /// Algebraic cancellation rule (e.g. `reverse().reverse()` = identity).
    pub cancellation: Option<BuiltinCancellation>,
    /// Columnar stage kind for backends that work on typed column vectors.
    pub columnar_stage: Option<BuiltinColumnarStage>,
    /// Structural index backend hint (deep search variants).
    pub structural: Option<BuiltinStructural>,
    /// Relative cost used by the planner's heuristic optimizer.
    pub cost: f64,
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
    /// Per-row projection stage.
    Map,
    /// Per-row expansion stage (one-to-many).
    FlatMap,
    /// Prefix filter that stops at the first non-matching row.
    TakeWhile,
    /// Skips leading matching rows and passes the rest.
    DropWhile,
    /// Deduplication stage (keeps first occurrence of each key).
    Distinct,
    /// Keyed reduce stage (groups, counts, or indexes by key).
    KeyedReduce,
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

/// Describes how a terminal reducing builtin accumulates its final result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BuiltinSinkSpec {
    /// Which accumulator algorithm to use.
    pub accumulator: BuiltinSinkAccumulator,
    /// How many rows the sink needs to see before it can emit a result.
    pub demand: BuiltinSinkDemand,
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

/// Which end of the stream the `SelectOne` sink picks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinSelectionPosition {
    /// Pick the first row seen (short-circuits on `first`).
    First,
    /// Pick the last row seen (must consume the whole stream for `last`).
    Last,
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
            | Self::Map
            | Self::FlatMap
            | Self::TakeWhile
            | Self::DropWhile
            | Self::Distinct
            | Self::KeyedReduce => BuiltinViewInputMode::ReadsView,
            Self::Take | Self::Skip => BuiltinViewInputMode::SkipsViewRead,
        }
    }

    /// Returns how this stage relates its output to the source view's memory.
    #[inline]
    pub fn output_mode(self) -> BuiltinViewOutputMode {
        match self {
            Self::Map => BuiltinViewOutputMode::BorrowedSubview,
            Self::FlatMap => BuiltinViewOutputMode::BorrowedSubviews,
            Self::KeyedReduce => BuiltinViewOutputMode::EmitsOwnedValue,
            Self::Filter
            | Self::TakeWhile
            | Self::DropWhile
            | Self::Distinct
            | Self::Take
            | Self::Skip => BuiltinViewOutputMode::PreservesInputView,
        }
    }

    /// Returns the materialization policy; currently always `Never` for all view stages.
    #[inline]
    pub fn materialization(self) -> BuiltinViewMaterialization {
        BuiltinViewMaterialization::Never
    }

    /// Returns the output row-count relationship of this stage.
    #[inline]
    pub fn cardinality(self) -> BuiltinCardinality {
        match self {
            Self::Filter => BuiltinCardinality::Filtering,
            Self::Map => BuiltinCardinality::OneToOne,
            Self::FlatMap => BuiltinCardinality::Expanding,
            Self::TakeWhile | Self::DropWhile => BuiltinCardinality::Filtering,
            Self::Distinct => BuiltinCardinality::Filtering,
            Self::KeyedReduce => BuiltinCardinality::Reducing,
            Self::Take | Self::Skip => BuiltinCardinality::Bounded,
        }
    }

    /// Returns whether this stage can participate in indexed (random-access) evaluation.
    #[inline]
    pub fn can_indexed(self) -> bool {
        matches!(self, Self::Map | Self::KeyedReduce)
    }

    /// Returns the relative per-row cost estimate used by the planner.
    #[inline]
    pub fn cost(self) -> f64 {
        match self {
            Self::Filter
            | Self::Map
            | Self::FlatMap
            | Self::TakeWhile
            | Self::DropWhile
            | Self::Distinct
            | Self::KeyedReduce => 10.0,
            Self::Take | Self::Skip => 0.5,
        }
    }

    /// Returns the estimated output-to-input row ratio (1.0 = no change, 0.5 = half the rows).
    #[inline]
    pub fn selectivity(self) -> f64 {
        match self {
            Self::Filter | Self::TakeWhile | Self::DropWhile => 0.5,
            Self::Distinct => 1.0,
            Self::Map | Self::FlatMap | Self::KeyedReduce => 1.0,
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
    /// Stage uses the legacy full-materialisation path.
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

/// Identifies the concrete executor kernel used to run a pipeline stage.
/// Selected by the lowering pass when translating a `BuiltinPipelineLowering` node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinPipelineExecutor {
    /// Element-level builtin with no lambda (scalar transforms, string ops, etc.).
    ElementBuiltin,
    /// Element-level builtin that may expand one row to many (flatten, explode, etc.).
    ExpandingBuiltin,
    /// Lambda applied over the fields of an object.
    ObjectLambda,
    /// Row-level predicate filter using a lambda.
    RowFilter,
    /// Row-level projection using a lambda.
    RowMap,
    /// Row-level expansion using a lambda.
    RowFlatMap,
    /// Positional slice operator; `take: true` = limit, `false` = offset.
    Position { take: bool },
    /// In-place array reversal.
    Reverse,
    /// Full-barrier comparison sort.
    Sort,
    /// Deduplication keyed by a lambda.
    UniqueBy,
    /// Group elements into an object keyed by a lambda.
    GroupBy,
    /// Count elements per key produced by a lambda.
    CountBy,
    /// Index elements (last write wins) keyed by a lambda.
    IndexBy,
    /// Return the index of the first matching element.
    FindIndex,
    /// Return all indices of matching elements.
    IndicesWhere,
    /// Select the element with the extreme (max or min) key value.
    ArgExtreme { max: bool },
    /// Split array into fixed-size chunks.
    Chunk,
    /// Slide a fixed-size window over the array.
    Window,
    /// Emit/skip elements from a contiguous prefix; `take: true` = take_while, `false` = drop_while.
    PrefixWhile { take: bool },
    /// Deduplication that assumes the input is already sorted.
    SortedDedup,
}

impl BuiltinPipelineExecutor {
    /// Returns true if this executor performs a per-row projection.
    #[inline]
    pub fn is_row_map(self) -> bool {
        matches!(self, Self::RowMap)
    }

    /// Returns true if this executor performs a per-row predicate filter.
    #[inline]
    pub fn is_row_filter(self) -> bool {
        matches!(self, Self::RowFilter)
    }

    /// Returns true if this executor operates on positions rather than values.
    #[inline]
    pub fn is_positional(self) -> bool {
        matches!(self, Self::Position { .. })
    }

    /// Returns true if the executor only reorders rows and never inspects their content.
    #[inline]
    pub fn is_order_only(self) -> bool {
        matches!(self, Self::Reverse | Self::Sort)
    }

    /// Returns true if the executor passes each input value into the lambda or comparator.
    #[inline]
    pub fn consumes_input_value(self) -> bool {
        matches!(
            self,
            Self::ObjectLambda
                | Self::RowFilter
                | Self::RowFlatMap
                | Self::UniqueBy
                | Self::GroupBy
                | Self::CountBy
                | Self::IndexBy
                | Self::FindIndex
                | Self::IndicesWhere
                | Self::ArgExtreme { .. }
                | Self::PrefixWhile { .. }
                | Self::SortedDedup
        )
    }
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

/// Describes the lowering target for a builtin in the pipeline compiler.
/// Each variant maps to a distinct code-generation path in the pipeline backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinPipelineLowering {
    /// Lambda-based stage (filter, map, flat_map, sort_by, etc.).
    ExprStage(BuiltinExprStage),
    /// Lambda-based stage followed by a terminal builtin that collapses the stream.
    TerminalExprStage {
        /// The upstream streaming stage.
        stage: BuiltinExprStage,
        /// The terminal method applied after the stage.
        terminal: BuiltinMethod,
    },
    /// Argument-free stage (reverse, unique).
    NullaryStage(BuiltinNullaryStage),
    /// Stage parameterised by a single `usize` (take, skip, chunk, window).
    UsizeStage {
        /// Which usize-parameterised stage.
        stage: BuiltinUsizeStage,
        /// Minimum legal argument value; arguments below this are rejected.
        min: usize,
    },
    /// Stage parameterised by a single string (split).
    StringStage(BuiltinStringStage),
    /// Stage parameterised by a pair of strings (replace, re_replace).
    StringPairStage(BuiltinStringPairStage),
    /// Stage parameterised by integer start/end bounds.
    IntRangeStage(BuiltinIntRangeStage),
    /// Full-barrier comparison sort with optional key expressions.
    Sort,
    /// Terminal sink (count, sum, avg, first, last, etc.).
    TerminalSink,
}

/// Lambda-driven pipeline stage variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinExprStage {
    /// Predicate filter.
    Filter,
    /// Projection.
    Map,
    /// Expanding projection.
    FlatMap,
    /// Contiguous prefix filter.
    TakeWhile,
    /// Drop prefix, pass rest.
    DropWhile,
    /// Collect all matching indices.
    IndicesWhere,
    /// Return the first matching index.
    FindIndex,
    /// Select element with the greatest key.
    MaxBy,
    /// Select element with the smallest key.
    MinBy,
    /// Deduplicate by key.
    UniqueBy,
    /// Group elements by key.
    GroupBy,
    /// Count elements by key.
    CountBy,
    /// Index elements by key.
    IndexBy,
    /// Map over values of an object.
    TransformValues,
    /// Map over keys of an object.
    TransformKeys,
    /// Filter entries of an object by value predicate.
    FilterValues,
    /// Filter entries of an object by key predicate.
    FilterKeys,
}

/// Argument-free pipeline stage variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinNullaryStage {
    /// Reverse the order of the array.
    Reverse,
    /// Remove duplicate values.
    Unique,
}

/// Pipeline stages parameterised by a single `usize` argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinUsizeStage {
    /// Keep the first N elements.
    Take,
    /// Drop the first N elements.
    Skip,
    /// Split into non-overlapping chunks of size N.
    Chunk,
    /// Slide a window of size N.
    Window,
}

/// Pipeline stages parameterised by a single string argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinStringStage {
    /// Split on a separator string.
    Split,
}

/// Pipeline stages parameterised by two string arguments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinStringPairStage {
    /// Replace using a literal needle; `all: true` replaces every occurrence.
    Replace { all: bool },
}

/// Pipeline stages parameterised by integer range arguments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinIntRangeStage {
    /// Slice an array or string by start/end index.
    Slice,
}

/// Materialisation policy for a view stage's output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinViewMaterialization {
    /// The stage never forces materialisation of the underlying view.
    Never,
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
            view_stage: None,
            sink: None,
            keyed_reducer: None,
            numeric_reducer: None,
            stage_merge: None,
            cancellation: None,
            columnar_stage: None,
            structural: None,
            cost: 1.0,
        }
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
                BuiltinSelectionPosition::Last => BuiltinSinkDemand::All {
                    value: BuiltinSinkValueNeed::Whole,
                    order: true,
                },
            },
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
        });
        self
    }

    /// Attaches a keyed reducer kind (group, count, or index).
    fn keyed_reducer(mut self, reducer: BuiltinKeyedReducer) -> Self {
        self.keyed_reducer = Some(reducer);
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
}

impl BuiltinMethod {
    /// Returns true for string scalar methods that take a single string argument and
    /// can execute directly on a `JsonView` without materialising the receiver.
    #[inline]
    pub(crate) fn is_string_arg_view_scalar(self) -> bool {
        matches!(
            self,
            Self::StartsWith | Self::EndsWith | Self::Matches | Self::IndexOf | Self::LastIndexOf
        )
    }

    /// Returns true for zero-argument string scalar methods that can execute on a `JsonView`.
    #[inline]
    pub(crate) fn is_string_no_arg_view_scalar(self) -> bool {
        matches!(
            self,
            Self::Upper
                | Self::Lower
                | Self::Trim
                | Self::TrimLeft
                | Self::TrimRight
                | Self::ByteLen
                | Self::IsBlank
                | Self::IsNumeric
                | Self::IsAlpha
                | Self::IsAscii
                | Self::ToNumber
                | Self::ToBool
        )
    }

    /// Returns true for zero-argument numeric scalar methods that can execute on a `JsonView`.
    #[inline]
    pub(crate) fn is_numeric_no_arg_view_scalar(self) -> bool {
        matches!(self, Self::Ceil | Self::Floor | Self::Round | Self::Abs)
    }

    /// Returns true if this method can be evaluated on a raw `JsonView` without materialising.
    #[inline]
    pub(crate) fn is_view_scalar_method(self) -> bool {
        self == Self::Len
            || self.is_string_arg_view_scalar()
            || self.is_string_no_arg_view_scalar()
            || self.is_numeric_no_arg_view_scalar()
    }

    /// Returns the full capability descriptor for this builtin.
    /// Called by the pipeline planner and VM to query cardinality, cost, and feature flags.
    #[inline]
    pub fn spec(self) -> BuiltinSpec {
        use BuiltinCardinality as Card;
        use BuiltinCategory as Cat;

        let spec = match self {
            Self::Filter | Self::Find | Self::FindAll => {
                BuiltinSpec::new(Cat::StreamingFilter, Card::Filtering)
                    .view_stage(BuiltinViewStage::Filter)
                    .columnar_stage(BuiltinColumnarStage::Filter)
                    .cost(10.0)
            }
            Self::Compact | Self::Remove => {
                BuiltinSpec::new(Cat::StreamingFilter, Card::Filtering).cost(10.0)
            }
            Self::Map => BuiltinSpec::new(Cat::StreamingOneToOne, Card::OneToOne)
                .indexed()
                .view_stage(BuiltinViewStage::Map)
                .columnar_stage(BuiltinColumnarStage::Map)
                .cost(10.0),
            Self::Enumerate | Self::Pairwise => {
                BuiltinSpec::new(Cat::StreamingOneToOne, Card::OneToOne)
                    .indexed()
                    .cost(10.0)
            }
            Self::FlatMap => BuiltinSpec::new(Cat::StreamingExpand, Card::Expanding)
                .view_stage(BuiltinViewStage::FlatMap)
                .columnar_stage(BuiltinColumnarStage::FlatMap)
                .cost(10.0),
            Self::Flatten | Self::Explode => {
                BuiltinSpec::new(Cat::StreamingExpand, Card::Expanding).cost(10.0)
            }
            Self::Split => BuiltinSpec::new(Cat::StreamingExpand, Card::Expanding).cost(10.0),
            Self::Lines | Self::Words | Self::Chars | Self::CharsOf | Self::Bytes => {
                BuiltinSpec::new(Cat::StreamingExpand, Card::Expanding).cost(10.0)
            }
            Self::TakeWhile => BuiltinSpec::new(Cat::StreamingFilter, Card::Filtering)
                .view_stage(BuiltinViewStage::TakeWhile)
                .cost(10.0),
            Self::DropWhile => BuiltinSpec::new(Cat::StreamingFilter, Card::Filtering)
                .view_stage(BuiltinViewStage::DropWhile)
                .cost(10.0),
            Self::FindFirst | Self::FindOne => {
                BuiltinSpec::new(Cat::StreamingFilter, Card::Filtering).cost(10.0)
            }
            Self::Take => BuiltinSpec::new(Cat::Positional, Card::Bounded)
                .view_native()
                .view_stage(BuiltinViewStage::Take)
                .stage_merge(BuiltinStageMerge::UsizeMin),
            Self::Skip => BuiltinSpec::new(Cat::Positional, Card::Bounded)
                .view_native()
                .view_stage(BuiltinViewStage::Skip)
                .stage_merge(BuiltinStageMerge::UsizeSaturatingAdd),
            Self::First => BuiltinSpec::new(Cat::Positional, Card::Bounded)
                .view_native()
                .select_one_sink(BuiltinSelectionPosition::First),
            Self::Last => BuiltinSpec::new(Cat::Positional, Card::Bounded)
                .view_native()
                .select_one_sink(BuiltinSelectionPosition::Last),
            Self::Nth | Self::Collect => {
                BuiltinSpec::new(Cat::Positional, Card::Bounded).view_native()
            }
            Self::Len => BuiltinSpec::new(Cat::Reducer, Card::Reducing)
                .indexed()
                .view_scalar()
                .count_sink(),
            Self::Sum => BuiltinSpec::new(Cat::Reducer, Card::Reducing)
                .view_native()
                .numeric_sink(BuiltinNumericReducer::Sum)
                .cost(10.0),
            Self::Avg => BuiltinSpec::new(Cat::Reducer, Card::Reducing)
                .view_native()
                .numeric_sink(BuiltinNumericReducer::Avg)
                .cost(10.0),
            Self::Min => BuiltinSpec::new(Cat::Reducer, Card::Reducing)
                .view_native()
                .numeric_sink(BuiltinNumericReducer::Min)
                .cost(10.0),
            Self::Max => BuiltinSpec::new(Cat::Reducer, Card::Reducing)
                .view_native()
                .numeric_sink(BuiltinNumericReducer::Max)
                .cost(10.0),
            Self::Count => BuiltinSpec::new(Cat::Reducer, Card::Reducing)
                .view_native()
                .count_sink()
                .cost(10.0),
            Self::ApproxCountDistinct => BuiltinSpec::new(Cat::Reducer, Card::Reducing)
                .view_native()
                .approx_distinct_sink()
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
                spec
            }
            Self::Sort
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
                    Self::Sort | Self::Unique | Self::UniqueBy => spec,
                    Self::Chunk | Self::Window => spec,
                    _ => spec,
                }
            }
            Self::GroupBy => BuiltinSpec::new(Cat::Reducer, Card::Reducing)
                .view_stage(BuiltinViewStage::KeyedReduce)
                .keyed_reducer(BuiltinKeyedReducer::Group)
                .columnar_stage(BuiltinColumnarStage::GroupBy)
                .cost(20.0),
            Self::CountBy => BuiltinSpec::new(Cat::Reducer, Card::Reducing)
                .view_stage(BuiltinViewStage::KeyedReduce)
                .keyed_reducer(BuiltinKeyedReducer::Count)
                .cost(10.0),
            Self::IndexBy => BuiltinSpec::new(Cat::Reducer, Card::Reducing)
                .view_stage(BuiltinViewStage::KeyedReduce)
                .keyed_reducer(BuiltinKeyedReducer::Index)
                .cost(10.0),
            Self::Unique | Self::UniqueBy => {
                BuiltinSpec::new(Cat::StreamingFilter, Card::Filtering)
                    .view_stage(BuiltinViewStage::Distinct)
                    .cost(10.0)
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
                    Self::Reverse => spec.cancellation(BuiltinCancellation::SelfInverse(
                        BuiltinCancelGroup::Reverse,
                    )),
                    _ => spec,
                }
            }
            Self::Keys | Self::Values | Self::Entries => {
                BuiltinSpec::new(Cat::Object, Card::OneToOne)
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
            | Self::Pivot
            | Self::Implode => BuiltinSpec::new(Cat::Object, Card::OneToOne),
            Self::TransformKeys | Self::TransformValues | Self::FilterKeys | Self::FilterValues => {
                BuiltinSpec::new(Cat::Object, Card::OneToOne)
            }
            Self::GetPath | Self::DelPath | Self::HasPath => {
                BuiltinSpec::new(Cat::Path, Card::OneToOne).indexed()
            }
            Self::SetPath | Self::DelPaths | Self::FlattenKeys | Self::UnflattenKeys => {
                BuiltinSpec::new(Cat::Path, Card::OneToOne).indexed()
            }
            Self::Walk | Self::WalkPre | Self::Rec | Self::TracePath => {
                BuiltinSpec::new(Cat::Deep, Card::Expanding).cost(20.0)
            }
            Self::DeepFind => BuiltinSpec::new(Cat::Deep, Card::Expanding)
                .structural(BuiltinStructural::DeepFind)
                .cost(20.0),
            Self::DeepShape => BuiltinSpec::new(Cat::Deep, Card::Expanding)
                .structural(BuiltinStructural::DeepShape)
                .cost(20.0),
            Self::DeepLike => BuiltinSpec::new(Cat::Deep, Card::Expanding)
                .structural(BuiltinStructural::DeepLike)
                .cost(20.0),
            Self::ToCsv | Self::ToTsv => BuiltinSpec::new(Cat::Serialization, Card::OneToOne)
                .indexed()
                .cost(20.0),
            Self::EquiJoin => BuiltinSpec::new(Cat::Relational, Card::Barrier).cost(20.0),
            Self::Set => BuiltinSpec::new(Cat::Mutation, Card::OneToOne).indexed(),
            Self::Update => BuiltinSpec::new(Cat::Mutation, Card::OneToOne).indexed(),
            Self::Lag
            | Self::Lead
            | Self::DiffWindow
            | Self::PctChange
            | Self::CumMax
            | Self::CumMin
            | Self::Zscore => BuiltinSpec::new(Cat::StreamingOneToOne, Card::OneToOne)
                .indexed()
                .cost(10.0),
            Self::Unknown => BuiltinSpec {
                pure: false,
                ..BuiltinSpec::new(Cat::Unknown, Card::OneToOne)
            },
            Self::Slice => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native(),
            Self::Replace | Self::ReplaceAll => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native(),
            _ => {
                let spec = BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                    .indexed()
                    .view_native();
                let spec = if self.is_view_scalar_method() {
                    spec.view_scalar()
                } else {
                    spec
                };
                spec
            }
        };
        match self {
            Self::ToBase64 => spec.cancellation(BuiltinCancellation::Inverse {
                group: BuiltinCancelGroup::Base64,
                side: BuiltinCancelSide::Forward,
            }),
            Self::FromBase64 => spec.cancellation(BuiltinCancellation::Inverse {
                group: BuiltinCancelGroup::Base64,
                side: BuiltinCancelSide::Backward,
            }),
            Self::UrlEncode => spec.cancellation(BuiltinCancellation::Inverse {
                group: BuiltinCancelGroup::Url,
                side: BuiltinCancelSide::Forward,
            }),
            Self::UrlDecode => spec.cancellation(BuiltinCancellation::Inverse {
                group: BuiltinCancelGroup::Url,
                side: BuiltinCancelSide::Backward,
            }),
            Self::HtmlEscape => spec.cancellation(BuiltinCancellation::Inverse {
                group: BuiltinCancelGroup::Html,
                side: BuiltinCancelSide::Forward,
            }),
            Self::HtmlUnescape => spec.cancellation(BuiltinCancellation::Inverse {
                group: BuiltinCancelGroup::Html,
                side: BuiltinCancelSide::Backward,
            }),
            Self::ReverseStr => spec.cancellation(BuiltinCancellation::SelfInverse(
                BuiltinCancelGroup::Reverse,
            )),
            _ => spec,
        }
    }
}

impl BuiltinCall {
    /// Constructs a `BuiltinCall` from a resolved method and its decoded arguments.
    #[inline]
    pub fn new(method: BuiltinMethod, args: BuiltinArgs) -> Self {
        Self { method, args }
    }

    /// Returns the capability descriptor for this call, potentially overriding the
    /// method-level spec with argument-specific cost or indexability adjustments.
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

    /// Returns true if applying this builtin twice is equivalent to applying it once.
    /// The pipeline optimizer uses this to eliminate redundant stages.
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

    /// Executes the builtin against `recv` with its pre-decoded static arguments.
    /// Returns `None` when the receiver type is not applicable (caller may fall back).
    /// For methods that can return errors, prefer [`BuiltinCall::try_apply`].
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
            (BuiltinMethod::ByteLen, BuiltinArgs::None)
            | (BuiltinMethod::IsBlank, BuiltinArgs::None)
            | (BuiltinMethod::IsNumeric, BuiltinArgs::None)
            | (BuiltinMethod::IsAlpha, BuiltinArgs::None)
            | (BuiltinMethod::IsAscii, BuiltinArgs::None)
            | (BuiltinMethod::ToNumber, BuiltinArgs::None)
            | (BuiltinMethod::ToBool, BuiltinArgs::None) => {
                apply_or_recv!(str_no_arg_scalar_val_apply(self.method, recv))
            }
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

    /// Attempts to construct a `BuiltinCall` from AST arguments that are all compile-time
    /// literals. Non-literal or lambda arguments cause `None` to be returned, falling back
    /// to runtime evaluation.
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

    /// Like [`BuiltinCall::from_literal_ast_args`] but also requires the method to be a
    /// registered pipeline element method, returning `None` otherwise.
    pub fn from_pipeline_literal_args(name: &str, args: &[crate::ast::Arg]) -> Option<Self> {
        let call = Self::from_literal_ast_args(name, args)?;
        call.method.is_pipeline_element_method().then_some(call)
    }

    /// Evaluates this builtin directly on a zero-copy `JsonView` without materialising a `Val`.
    /// Only works for view-scalar methods; returns `None` for all other builtins.
    pub fn try_apply_json_view(&self, recv: crate::util::JsonView<'_>) -> Option<Val> {
        if !self.spec().view_scalar {
            return None;
        }
        match (self.method, &self.args) {
            (BuiltinMethod::Len, BuiltinArgs::None) => json_view_len(recv).map(Val::Int),
            (method, BuiltinArgs::None) if method.is_string_no_arg_view_scalar() => {
                let value = json_view_str(recv)?;
                str_no_arg_scalar_apply(method, value)
            }
            (method, BuiltinArgs::None) if method.is_numeric_no_arg_view_scalar() => {
                numeric_no_arg_scalar_apply(method, recv)
            }
            (method, BuiltinArgs::Str(arg)) if method.is_string_arg_view_scalar() => {
                let value = json_view_str(recv)?;
                str_arg_scalar_apply(method, value, arg.as_ref())
            }
            _ => None,
        }
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

impl BuiltinMethod {
    /// Returns true if this method is registered as a pipeline element method.
    /// Pipeline element methods operate on individual values and can run in-stream.
    #[inline]
    pub fn is_pipeline_element_method(self) -> bool {
        crate::builtin_registry::pipeline_element(crate::builtin_registry::BuiltinId::from_method(
            self,
        ))
    }
}

/// Per-row filter primitive: evaluates `eval` on `item` and returns its truthiness.
/// Streaming consumers call this once per row instead of buffering the entire array.
#[inline]
pub fn filter_one<F>(item: &Val, mut eval: F) -> Result<bool, EvalError>
where
    F: FnMut(&Val) -> Result<Val, EvalError>,
{
    Ok(is_truthy(&eval(item)?))
}

/// Buffered filter: applies the predicate to every element and returns all passing items.
/// Barrier consumers call this after collecting the full input.
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

/// Bounded filter: like [`filter_apply`] but stops after collecting `max_keep` matching items.
/// Pass `None` for `max_keep` to collect all matches (equivalent to `filter_apply`).
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

/// Per-row map primitive: evaluates `eval` on `item` and returns the projected value.
#[inline]
pub fn map_one<F>(item: &Val, mut eval: F) -> Result<Val, EvalError>
where
    F: FnMut(&Val) -> Result<Val, EvalError>,
{
    eval(item)
}

/// Buffered map: applies the projection to every element and returns the results.
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

/// Bounded map: like [`map_apply`] but stops after emitting `max_emit` projected values.
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

/// Per-row flat_map primitive: evaluates `eval`, then flattens one level if the result is an array.
/// Returns a `SmallVec` to avoid heap allocation for the common single-element case.
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

/// Buffered flat_map: maps and flattens every element into a single output vector.
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

/// Natural (ascending) sort. Specialises for homogeneous `IntVec` and `FloatVec` arrays
/// before falling back to the generic `cmp_vals` comparator.
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

/// Multi-key sort: evaluates one or more key expressions per element, then sorts using
/// the resulting key tuples. Each entry in `desc` controls ascending/descending order
/// for the corresponding key position.
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

/// Sorts an array using a two-argument comparator lambda (returns `true` when left < right).
/// Errors from the comparator are captured and surfaced after the sort completes.
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

/// Removes all elements for which the predicate is truthy (inverse of `filter`).
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

/// Filters an array keeping only elements that satisfy all `pred_count` predicates.
/// Multiple predicates are ANDed together; `eval(item, idx)` evaluates the `idx`-th predicate.
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

/// Deduplicates an array by a key expression, keeping the first occurrence of each distinct key.
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

/// Returns the index of the first element satisfying all predicates, or `Val::Null`.
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

/// Returns all indices where every predicate evaluates to truthy; result is `Val::IntVec`.
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

/// Returns the element with the greatest (or smallest) key from the key expression.
/// `want_max = true` for `max_by`, `false` for `min_by`. Returns `Val::Null` for empty arrays.
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

/// Normalises a value to an array: `null` → `[]`, array → identity, scalar → `[scalar]`.
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

/// Zips two arrays element-wise, stopping at the shorter array.
#[inline]
pub fn zip_apply(recv: Val, other: Val) -> Result<Val, EvalError> {
    zip_arrays(recv, other, false, Val::Null)
}

/// Zips two arrays element-wise, padding the shorter array with `fill`.
#[inline]
pub fn zip_longest_apply(recv: Val, other: Val, fill: Val) -> Result<Val, EvalError> {
    zip_arrays(recv, other, true, fill)
}

/// Zips N arrays element-wise into `[[a0, b0, ...], ...]`, truncating to the shortest.
#[inline]
pub fn global_zip_apply(arrs: &[Val]) -> Val {
    let len = arrs.iter().filter_map(|a| a.arr_len()).min().unwrap_or(0);
    Val::arr(
        (0..len)
            .map(|i| Val::arr(arrs.iter().map(|a| a.get_index(i as i64)).collect()))
            .collect(),
    )
}

/// Zips N arrays element-wise, padding shorter arrays with `fill`.
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

/// Computes the Cartesian product of N arrays, returning all combinations as `[[...], ...]`.
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

/// Generates an integer range. Accepts 1–3 arguments: `(end)`, `(start, end)`, or
/// `(start, end, step)`. Returns an empty array when `step == 0` or the range is empty.
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

/// Inner equi-join of two arrays of objects on matching key fields.
/// Builds a hash index over the right-hand array, then iterates the left, merging matches.
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

/// Shallow-merges two objects (right wins on collision). Used internally by `equi_join`.
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

/// Pivots an array of objects into a flat or nested map.
/// Two-arg form: `pivot(key_expr, val_expr)` → `{key: val, ...}`.
/// Three-arg form: `pivot(row_expr, col_expr, val_expr)` → `{row: {col: val, ...}, ...}`.
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

/// DFS pre-order visitor: calls `f` on every node (parents before children).
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

/// DFS pre-order search: collects every node in the tree that satisfies all `pred_count` predicates.
/// Visits every descendant including nested arrays and objects.
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

/// DFS pre-order search: collects every object node that contains all of the given `keys`.
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

/// DFS pre-order search: collects every object node whose listed keys equal the given literal values.
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

/// Recursive tree transform. When `pre = true` the transform runs top-down (pre-order);
/// when `pre = false` it runs bottom-up (post-order). All array and object children
/// are recursively transformed, then the lambda is applied.
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

/// Applies `eval` repeatedly until the value reaches a fixpoint (output equals input).
/// Errors if the fixpoint is not reached within 10 000 iterations.
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

/// Walks the entire value tree and, for every node where the predicate is truthy, emits
/// a `{path: "$...", value: ...}` object. Paths use `$` as the root and `.field` / `[idx]` syntax.
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

/// Evaluates `count` independent expressions against the same receiver and returns
/// the results as an array `[expr0(recv), expr1(recv), ...]`.
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

/// Evaluates named expressions against the receiver and collects results into an object
/// `{name0: expr0(recv), name1: expr1(recv), ...}`.
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

/// Groups elements by a key expression (arg 0), then applies a shape expression (arg 1)
/// to each group, returning `{key: shape(group), ...}`.
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

/// Per-row primitive for `take_while`: returns true while the predicate holds.
#[inline]
pub fn take_while_one<F>(item: &Val, eval: F) -> Result<bool, EvalError>
where
    F: FnMut(&Val) -> Result<Val, EvalError>,
{
    filter_one(item, eval)
}

/// Per-row primitive for `any`: returns true when the predicate is truthy.
#[inline]
pub fn any_one<F>(item: &Val, eval: F) -> Result<bool, EvalError>
where
    F: FnMut(&Val) -> Result<Val, EvalError>,
{
    filter_one(item, eval)
}

/// Per-row primitive for `all`: returns true when the predicate is truthy.
#[inline]
pub fn all_one<F>(item: &Val, eval: F) -> Result<bool, EvalError>
where
    F: FnMut(&Val) -> Result<Val, EvalError>,
{
    filter_one(item, eval)
}

/// Buffered `take_while`: keeps the leading elements satisfying the predicate, stops at the first falsy result.
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

/// Buffered `drop_while`: skips leading elements satisfying the predicate, then passes the rest.
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

/// Splits elements into two groups: those satisfying the predicate (first) and those that don't (second).
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

/// Groups elements by a key expression, returning an `IndexMap<key, [elements]>`.
/// Insertion order of the first occurrence of each key is preserved.
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

/// Counts elements per key expression, returning an `IndexMap<key, Int>`.
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

/// Indexes elements by a key expression, returning `IndexMap<key, last_matching_element>`.
/// When two elements share a key, the last one wins.
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

/// Filters an object's entries, keeping only those for which `keep(key, value)` is truthy.
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

/// Applies `eval` to every key of the object and rebuilds the map with the new keys.
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

/// Returns an array of every key in the object, or an empty array for non-objects.
#[inline]
pub fn keys_apply(recv: &Val) -> Val {
    Val::arr(
        recv.as_object()
            .map(|m| m.keys().map(|k| Val::Str(k.clone())).collect())
            .unwrap_or_default(),
    )
}

/// Returns an array of every value in the object, or an empty array for non-objects.
#[inline]
pub fn values_apply(recv: &Val) -> Val {
    Val::arr(
        recv.as_object()
            .map(|m| m.values().cloned().collect())
            .unwrap_or_default(),
    )
}

/// Returns `[[key, value], ...]` pairs for each entry in the object.
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

/// Returns a substring by character indices, supporting negative indexing.
/// Returns a zero-copy `StrSlice` view when the input is ASCII; allocates otherwise.
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

/// Splits a string on `sep` and returns the parts as an array of strings.
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

/// Splits a slice into non-overlapping chunks of size `n` (last chunk may be smaller).
#[inline]
pub fn chunk_apply(items: &[Val], n: usize) -> Vec<Val> {
    let n = n.max(1);
    items.chunks(n).map(|c| Val::arr(c.to_vec())).collect()
}

/// Produces all contiguous windows of size `n` from a slice of values.
#[inline]
pub fn window_apply(items: &[Val], n: usize) -> Vec<Val> {
    let n = n.max(1);
    items.windows(n).map(|w| Val::arr(w.to_vec())).collect()
}

/// Replaces `needle` with `replacement` in a string. When `all` is true replaces every
/// occurrence; otherwise only the first. Returns the original value unchanged if `needle` is absent.
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

/// Applies a `&str → String` transform to the string inside `recv`, wrapping the result in `Val::Str`.
#[inline]
fn map_str_owned(recv: &Val, f: impl FnOnce(&str) -> String) -> Option<Val> {
    let s = recv.as_str_ref()?;
    Some(Val::Str(Arc::<str>::from(f(s).as_str())))
}

/// Converts the string to all-uppercase (ASCII fast path).
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

/// Converts the string to all-lowercase (ASCII fast path).
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

/// Strips leading and trailing whitespace from a string.
#[inline]
pub fn trim_apply(recv: &Val) -> Option<Val> {
    map_str_owned(recv, |s| s.trim().to_owned())
}

/// Strips leading whitespace from a string.
#[inline]
pub fn trim_left_apply(recv: &Val) -> Option<Val> {
    map_str_owned(recv, |s| s.trim_start().to_owned())
}

/// Strips trailing whitespace from a string.
#[inline]
pub fn trim_right_apply(recv: &Val) -> Option<Val> {
    map_str_owned(recv, |s| s.trim_end().to_owned())
}

/// Uppercases the first character and lowercases the rest of a string.
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

/// Capitalises the first letter of each whitespace-delimited word.
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

/// Escapes `<`, `>`, `&`, `"`, and `'` to their HTML entity equivalents.
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

/// Converts HTML entities (`&lt;`, `&gt;`, `&amp;`, `&quot;`, `&#39;`) back to their characters.
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

/// Percent-encodes a string using RFC 3986 unreserved characters (`A-Z a-z 0-9 - _ . ~`).
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

/// Decodes a percent-encoded URL string, also converting `+` to space.
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

/// Encodes a string's bytes as standard (non-padded) Base64.
#[inline]
pub fn to_base64_apply(recv: &Val) -> Option<Val> {
    map_str_owned(recv, |s| {
        crate::builtin_helpers::base64_encode(s.as_bytes())
    })
}

/// Removes the common leading whitespace prefix from every non-blank line.
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

/// Converts a string to `snake_case` by splitting on word boundaries and joining with `_`.
#[inline]
pub fn snake_case_apply(recv: &Val) -> Option<Val> {
    map_str_owned(recv, |s| {
        crate::builtin_helpers::split_words_lower(s).join("_")
    })
}

/// Converts a string to `kebab-case` by splitting on word boundaries and joining with `-`.
#[inline]
pub fn kebab_case_apply(recv: &Val) -> Option<Val> {
    map_str_owned(recv, |s| {
        crate::builtin_helpers::split_words_lower(s).join("-")
    })
}

/// Converts a string to `camelCase` (first word lowercase, subsequent words title-cased).
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

/// Converts a string to `PascalCase` (every word title-cased, no separator).
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

/// Reverses the Unicode codepoints of a string.
#[inline]
pub fn reverse_str_apply(recv: &Val) -> Option<Val> {
    map_str_owned(recv, |s| s.chars().rev().collect::<String>())
}

/// Applies a `&str → Val` transform to the string inside `recv`.
#[inline]
fn map_str_val(recv: &Val, f: impl FnOnce(&str) -> Val) -> Option<Val> {
    Some(f(recv.as_str_ref()?))
}

/// Splits a string on newlines and returns each line as a `Val::Str`.
#[inline]
pub fn lines_apply(recv: &Val) -> Option<Val> {
    map_str_val(recv, |s| {
        Val::arr(s.lines().map(|l| Val::Str(Arc::from(l))).collect())
    })
}

/// Splits a string on whitespace and returns each token as a `Val::Str`.
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

/// Returns each Unicode character as a single-char `Val::Str`.
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

/// Returns each Unicode code point re-encoded as a UTF-8 `Val::Str` (same as `chars` for BMP).
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

/// Returns each byte of the string's UTF-8 encoding as a `Val::Int`.
#[inline]
pub fn bytes_of_apply(recv: &Val) -> Option<Val> {
    map_str_val(recv, |s| {
        let v: Vec<i64> = s.as_bytes().iter().map(|&b| b as i64).collect();
        Val::int_vec(v)
    })
}

/// Returns the ceiling (round-up) of a numeric value as `Val::Int`.
#[inline]
pub fn ceil_apply(recv: &Val) -> Option<Val> {
    match recv {
        Val::Int(n) => Some(Val::Int(*n)),
        Val::Float(f) => Some(Val::Int(f.ceil() as i64)),
        _ => None,
    }
}

/// Like [`ceil_apply`] but returns an error for non-numeric receivers.
#[inline]
pub fn try_ceil_apply(recv: &Val) -> Result<Option<Val>, EvalError> {
    ceil_apply(recv)
        .map(Some)
        .ok_or_else(|| EvalError("ceil: expected number".into()))
}

/// Returns the floor (round-down) of a numeric value as `Val::Int`.
#[inline]
pub fn floor_apply(recv: &Val) -> Option<Val> {
    match recv {
        Val::Int(n) => Some(Val::Int(*n)),
        Val::Float(f) => Some(Val::Int(f.floor() as i64)),
        _ => None,
    }
}

/// Like [`floor_apply`] but returns an error for non-numeric receivers.
#[inline]
pub fn try_floor_apply(recv: &Val) -> Result<Option<Val>, EvalError> {
    floor_apply(recv)
        .map(Some)
        .ok_or_else(|| EvalError("floor: expected number".into()))
}

/// Rounds a numeric value to the nearest integer.
#[inline]
pub fn round_apply(recv: &Val) -> Option<Val> {
    match recv {
        Val::Int(n) => Some(Val::Int(*n)),
        Val::Float(f) => Some(Val::Int(f.round() as i64)),
        _ => None,
    }
}

/// Like [`round_apply`] but returns an error for non-numeric receivers.
#[inline]
pub fn try_round_apply(recv: &Val) -> Result<Option<Val>, EvalError> {
    round_apply(recv)
        .map(Some)
        .ok_or_else(|| EvalError("round: expected number".into()))
}

/// Returns the absolute value of an integer or float.
#[inline]
pub fn abs_apply(recv: &Val) -> Option<Val> {
    match recv {
        Val::Int(n) => Some(Val::Int(n.wrapping_abs())),
        Val::Float(f) => Some(Val::Float(f.abs())),
        _ => None,
    }
}

/// Like [`abs_apply`] but returns an error for non-numeric receivers.
#[inline]
pub fn try_abs_apply(recv: &Val) -> Result<Option<Val>, EvalError> {
    abs_apply(recv)
        .map(Some)
        .ok_or_else(|| EvalError("abs: expected number".into()))
}

/// Parses the string as a base-10 `i64`; returns `Val::Null` on failure.
#[inline]
pub fn parse_int_apply(recv: &Val) -> Option<Val> {
    map_str_val(recv, |s| {
        s.trim().parse::<i64>().map(Val::Int).unwrap_or(Val::Null)
    })
}

/// Parses the string as an `f64`; returns `Val::Null` on failure.
#[inline]
pub fn parse_float_apply(recv: &Val) -> Option<Val> {
    map_str_val(recv, |s| {
        s.trim().parse::<f64>().map(Val::Float).unwrap_or(Val::Null)
    })
}

/// Parses common truthy/falsy string representations to `Val::Bool`; returns `Val::Null` otherwise.
/// Recognises `true/yes/1/on` and `false/no/0/off` (case-insensitive).
#[inline]
pub fn parse_bool_apply(recv: &Val) -> Option<Val> {
    map_str_val(recv, |s| match s.trim().to_ascii_lowercase().as_str() {
        "true" | "yes" | "1" | "on" => Val::Bool(true),
        "false" | "no" | "0" | "off" => Val::Bool(false),
        _ => Val::Null,
    })
}

/// Decodes a Base64 string to its UTF-8 representation; returns `Val::Null` for invalid input.
#[inline]
pub fn from_base64_apply(recv: &Val) -> Option<Val> {
    map_str_val(recv, |s| match crate::builtin_helpers::base64_decode(s) {
        Ok(bytes) => Val::Str(Arc::from(String::from_utf8_lossy(&bytes).as_ref())),
        Err(_) => Val::Null,
    })
}

/// Returns the string repeated `n` times.
#[inline]
pub fn repeat_apply(recv: &Val, n: usize) -> Option<Val> {
    Some(Val::Str(Arc::from(recv.as_str_ref()?.repeat(n))))
}

/// Removes `prefix` from the beginning of the string if present; returns the original otherwise.
#[inline]
pub fn strip_prefix_apply(recv: &Val, prefix: &str) -> Option<Val> {
    let s = recv.as_str_ref()?;
    Some(match s.strip_prefix(prefix) {
        Some(stripped) => Val::Str(Arc::<str>::from(stripped)),
        None => recv.clone(),
    })
}

/// Removes `suffix` from the end of the string if present; returns the original otherwise.
#[inline]
pub fn strip_suffix_apply(recv: &Val, suffix: &str) -> Option<Val> {
    let s = recv.as_str_ref()?;
    Some(match s.strip_suffix(suffix) {
        Some(stripped) => Val::Str(Arc::<str>::from(stripped)),
        None => recv.clone(),
    })
}

/// Left-pads the string to `width` characters with `fill`; returns the original when already wide enough.
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

/// Right-pads the string to `width` characters with `fill`; returns the original when already wide enough.
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

/// Centers the string within `width` characters by padding both sides with `fill`.
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

/// Prepends `n` spaces to each line of the string.
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

/// Finds every non-overlapping occurrence of `pat` and returns an array of the matched strings.
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

/// Applies a numeric aggregate (`sum`, `avg`, `min`, `max`) to an array or typed numeric vector.
/// Returns `Val::Null` when the receiver is not an array-like type.
#[inline]
pub fn numeric_aggregate_apply(recv: &Val, method: BuiltinMethod) -> Val {
    match recv {
        Val::IntVec(a) => return numeric_aggregate_i64(a, method),
        Val::FloatVec(a) => return numeric_aggregate_f64(a, method),
        Val::Arr(a) => numeric_aggregate_values(a, method),
        _ => Val::Null,
    }
}

/// Numeric aggregate with a projection: evaluates `eval` on each element first,
/// then aggregates all numeric results. Non-numeric projected values are silently skipped.
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

/// Numeric aggregate specialised for homogeneous `i64` slices.
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

/// Numeric aggregate specialised for homogeneous `f64` slices.
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

/// Numeric aggregate for heterogeneous `Val` slices; skips non-numeric elements.
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

/// Returns the logical length of an array, object, or string (char count), or `None` for scalars.
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

/// Removes all `Val::Null` elements from an array.
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

/// Recursively flattens nested arrays up to `depth` levels deep.
#[inline]
pub fn flatten_depth_apply(recv: &Val, depth: usize) -> Option<Val> {
    if matches!(recv, Val::Arr(_)) {
        Some(crate::util::flatten_val(recv.clone(), depth))
    } else {
        None
    }
}

/// Reverses any sequence type: arrays, typed vectors, and strings (by Unicode codepoints).
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

/// Removes duplicate elements from an array, preserving first-seen order.
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

/// Extracts numeric values from any array-like `Val` as `Option<f64>`, preserving nulls as `None`.
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

/// Converts a `Vec<Option<f64>>` back to a `Val`: returns `FloatVec` when all are `Some`, otherwise `Arr` with nulls.
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

/// Computes a rolling sum over a window of size `n`; positions before the first full window are `Null`.
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

/// Computes a rolling average over a window of size `n`; positions before the first full window are `Null`.
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

/// Computes a rolling minimum over a window of size `n`; positions before the first full window are `Null`.
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

/// Computes a rolling maximum over a window of size `n`; positions before the first full window are `Null`.
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

/// Shifts values backward by `n` positions; the first `n` positions are `Null`.
#[inline]
pub fn lag_apply(recv: &Val, n: usize) -> Option<Val> {
    let xs = numeric_options(recv)?;
    let mut out: Vec<Option<f64>> = Vec::with_capacity(xs.len());
    for i in 0..xs.len() {
        out.push(if i >= n { xs[i - n] } else { None });
    }
    Some(numeric_options_to_val(out))
}

/// Shifts values forward by `n` positions; the last `n` positions are `Null`.
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

/// Returns element-wise first differences (`v[i] - v[i-1]`); the first element is `Null`.
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

/// Returns element-wise percentage change `(v[i] - v[i-1]) / v[i-1]`; division by zero and the first element yield `Null`.
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

/// Computes a cumulative maximum: each position holds the running max up to that index.
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

/// Computes a cumulative minimum: each position holds the running min up to that index.
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

/// Normalises each element to its z-score `(v - mean) / stddev`; returns 0 when stddev is zero, `Null` for non-numeric.
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

/// Returns the first `n` elements of an array; when `n == 1` returns a scalar instead of a single-element array.
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

/// Returns the last `n` elements of an array; when `n == 1` returns a scalar instead of a single-element array.
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

/// Returns the element at index `i` (negative indices count from the end); delegates to `Val::get_index`.
#[inline]
pub fn nth_any_apply(recv: &Val, i: i64) -> Option<Val> {
    Some(recv.get_index(i))
}

/// Appends `item` to the end of an array, returning a new array.
#[inline]
pub fn append_apply(recv: &Val, item: &Val) -> Option<Val> {
    let mut v = recv.clone().into_vec()?;
    v.push(item.clone());
    Some(Val::arr(v))
}

/// Inserts `item` at the beginning of an array, returning a new array.
#[inline]
pub fn prepend_apply(recv: &Val, item: &Val) -> Option<Val> {
    let mut v = recv.clone().into_vec()?;
    v.insert(0, item.clone());
    Some(Val::arr(v))
}

/// Removes all elements from an array that are structurally equal to `target`.
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

/// Pairs each element with its zero-based index, producing `[{index, value}, …]`.
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

/// Joins all array elements into a single string separated by `sep`; non-string elements are coerced.
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

/// Returns the zero-based index of the first occurrence of `target`, or `Val::Null` if not found.
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

/// Returns all zero-based indices where `target` appears in the array.
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

/// Unnests the array-valued `field` of each row object: each element of the nested array becomes
/// its own row, copying all other fields.
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

/// Inverse of `explode`: groups rows by all fields except `field`, collecting the `field` values
/// into an array on each merged row.
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

/// Produces all adjacent pairs `[[a,b],[b,c],…]` from an array.
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

/// Splits an array into non-overlapping chunks of size `n`; the last chunk may be smaller.
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

/// Produces all overlapping sliding windows of size `n` from an array.
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

/// Returns elements that appear in both `recv` and `other` (set intersection, order from `recv`).
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

/// Returns all elements from `recv` plus elements in `other` not already present (set union).
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

/// Returns elements from `recv` that do not appear in `other` (set difference).
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

/// Converts an array of `[key, value]` pairs or `{key, val}` objects into an object.
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

/// Swaps keys and values of an object; values are coerced to strings to become new keys.
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

/// Shallow-merges `other` into `recv`; `other` keys overwrite `recv` keys.
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

/// Recursively merges `other` into `recv`, combining nested objects rather than replacing them.
#[inline]
pub fn deep_merge_apply(recv: &Val, other: &Val) -> Option<Val> {
    Some(crate::util::deep_merge(recv.clone(), other.clone()))
}

/// Fills missing or null keys of `recv` with values from `other` (non-destructive merge).
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

/// Renames keys in an object according to `renames` (`{old: new, …}`), preserving other keys.
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

/// A single resolved segment of a dot/bracket path string.
pub(crate) enum PathSeg {
    /// A named object field (`.foo`).
    Field(String),
    /// A numeric array index (`[0]` or `[-1]`).
    Index(i64),
}

/// Describes where to read a value from when executing a `pick` specification.
pub(crate) enum PickSource {
    /// A single top-level field name.
    Field(Arc<str>),
    /// A multi-segment dot/bracket path, pre-parsed into [`PathSeg`]s.
    Path(Vec<PathSeg>),
}

/// One entry in a compiled `pick` call: the output key name and where to read the value from.
pub(crate) struct PickSpec {
    /// Key used in the output object.
    pub out_key: Arc<str>,
    /// Source location (field or path) inside the input object.
    pub source: PickSource,
}

/// Parses a dot/bracket path string (e.g. `"a.b[0].c"`) into a `Vec<PathSeg>`.
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

/// Traverses `val` following `segs`, returning the found value or `Val::Null` when any step is missing.
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

/// Returns a copy of `val` with the node at `segs` replaced by `new_val`; creates missing intermediate objects.
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

/// Returns a copy of `val` with the node at `segs` removed; no-ops if the path does not exist.
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

/// Converts a possibly-negative index into an absolute `usize`, clamped to `[0, len)`.
fn resolve_path_idx(i: i64, len: i64) -> usize {
    (if i < 0 { (len + i).max(0) } else { i }) as usize
}

/// Recursively flattens nested object keys into dot-separated (or `sep`-separated) flat keys,
/// writing results into `out`. Arrays and scalars terminate the recursion.
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

/// Reconstructs a nested object from a flat `{sep}-joined-key: value` map.
pub(crate) fn unflatten_keys_impl(m: &IndexMap<Arc<str>, Val>, sep: &str) -> Val {
    let mut root: IndexMap<Arc<str>, Val> = IndexMap::new();
    for (key, val) in m {
        let parts: Vec<&str> = key.split(sep).collect();
        insert_nested(&mut root, &parts, val.clone());
    }
    Val::obj(root)
}

/// Recursively inserts `val` at the nested path `parts` inside `obj`, creating intermediate objects as needed.
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

/// Retrieves the value at a dot/bracket `path` string, returning `Val::Null` for missing nodes.
#[inline]
pub fn get_path_apply(recv: &Val, path: &str) -> Option<Val> {
    let segs = parse_path_segs(path);
    Some(get_path_impl(recv, &segs))
}

/// Returns `Val::Bool(true)` when a value exists (non-null) at the given dot/bracket path.
#[inline]
pub fn has_path_apply(recv: &Val, path: &str) -> Option<Val> {
    let segs = parse_path_segs(path);
    let found = !get_path_impl(recv, &segs).is_null();
    Some(Val::Bool(found))
}

/// Returns `Val::Bool(true)` when the object has a top-level key named `key`.
#[inline]
pub fn has_apply(recv: &Val, key: &str) -> Option<Val> {
    let m = recv.as_object()?;
    Some(Val::Bool(m.contains_key(key)))
}

/// Keeps only the listed `keys` from an object (or each object in an array), dropping all others.
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

/// Richer version of `pick_apply` that supports aliasing and deep-path sources via [`PickSpec`].
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

/// Removes the listed `keys` from an object (or each object in an array), keeping all others.
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

/// Returns a copy of `recv` with the node at the dot/bracket `path` removed.
#[inline]
pub fn del_path_apply(recv: &Val, path: &str) -> Option<Val> {
    let segs = parse_path_segs(path);
    Some(del_path_impl(recv.clone(), &segs))
}

/// Returns a copy of `recv` with the node at the dot/bracket `path` replaced by `value`.
#[inline]
pub fn set_path_apply(recv: &Val, path: &str, value: &Val) -> Option<Val> {
    let segs = parse_path_segs(path);
    Some(set_path_impl(recv.clone(), &segs, value.clone()))
}

/// Deletes multiple dot/bracket paths from `recv` sequentially, returning the final result.
#[inline]
pub fn del_paths_apply(recv: &Val, paths: &[Arc<str>]) -> Option<Val> {
    let mut out = recv.clone();
    for path in paths {
        let segs = parse_path_segs(path.as_ref());
        out = del_path_impl(out, &segs);
    }
    Some(out)
}

/// Collapses a nested object into a flat object using `sep`-joined key paths (e.g. `"a.b.c": v`).
#[inline]
pub fn flatten_keys_apply(recv: &Val, sep: &str) -> Option<Val> {
    let mut out: indexmap::IndexMap<Arc<str>, Val> = indexmap::IndexMap::new();
    flatten_keys_impl("", recv, sep, &mut out);
    Some(Val::obj(out))
}

/// Reconstructs a nested object from a flat `sep`-delimited key map; inverse of `flatten_keys_apply`.
#[inline]
pub fn unflatten_keys_apply(recv: &Val, sep: &str) -> Option<Val> {
    if let Val::Obj(m) = recv {
        Some(unflatten_keys_impl(m, sep))
    } else {
        None
    }
}

/// Compiles a regex pattern, converting any compilation error into an `EvalError`.
#[inline]
fn compile_regex_eval(pat: &str) -> Result<Arc<regex::Regex>, EvalError> {
    crate::builtin_helpers::compile_regex(pat).map_err(EvalError)
}

/// Returns `Val::Bool` indicating whether the full string matches `pat`; returns `None` for non-strings.
#[inline]
pub fn re_match_apply(recv: &Val, pat: &str) -> Option<Val> {
    try_re_match_apply(recv, pat).ok().flatten()
}

/// Fallible variant of [`re_match_apply`]; propagates regex compilation errors as `EvalError`.
#[inline]
pub fn try_re_match_apply(recv: &Val, pat: &str) -> Result<Option<Val>, EvalError> {
    let Some(s) = recv.as_str_ref() else {
        return Ok(None);
    };
    let re = compile_regex_eval(pat)?;
    Ok(Some(Val::Bool(re.is_match(s))))
}

/// Returns the first substring matching `pat`, or `Val::Null` if no match is found.
#[inline]
pub fn re_match_first_apply(recv: &Val, pat: &str) -> Option<Val> {
    try_re_match_first_apply(recv, pat).ok().flatten()
}

/// Fallible variant of [`re_match_first_apply`]; propagates regex compilation errors.
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

/// Returns all non-overlapping substrings matching `pat` as a `StrVec`.
#[inline]
pub fn re_match_all_apply(recv: &Val, pat: &str) -> Option<Val> {
    try_re_match_all_apply(recv, pat).ok().flatten()
}

/// Fallible variant of [`re_match_all_apply`]; propagates regex compilation errors.
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

/// Returns capture groups of the first match as an array, or `Val::Null` if no match.
#[inline]
pub fn re_captures_apply(recv: &Val, pat: &str) -> Option<Val> {
    try_re_captures_apply(recv, pat).ok().flatten()
}

/// Fallible variant of [`re_captures_apply`]; propagates regex compilation errors.
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

/// Returns an array of capture-group arrays for every match of `pat` in the string.
#[inline]
pub fn re_captures_all_apply(recv: &Val, pat: &str) -> Option<Val> {
    try_re_captures_all_apply(recv, pat).ok().flatten()
}

/// Fallible variant of [`re_captures_all_apply`]; propagates regex compilation errors.
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

/// Replaces the first occurrence of `pat` in the string with `with`.
#[inline]
pub fn re_replace_apply(recv: &Val, pat: &str, with: &str) -> Option<Val> {
    try_re_replace_apply(recv, pat, with).ok().flatten()
}

/// Fallible variant of [`re_replace_apply`]; propagates regex compilation errors.
#[inline]
pub fn try_re_replace_apply(recv: &Val, pat: &str, with: &str) -> Result<Option<Val>, EvalError> {
    let Some(s) = recv.as_str_ref() else {
        return Ok(None);
    };
    let re = compile_regex_eval(pat)?;
    let out = re.replace(s, with);
    Ok(Some(Val::Str(Arc::from(out.as_ref()))))
}

/// Replaces all non-overlapping occurrences of `pat` in the string with `with`.
#[inline]
pub fn re_replace_all_apply(recv: &Val, pat: &str, with: &str) -> Option<Val> {
    try_re_replace_all_apply(recv, pat, with).ok().flatten()
}

/// Fallible variant of [`re_replace_all_apply`]; propagates regex compilation errors.
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

/// Splits the string on all matches of `pat`, returning a `StrVec` of tokens.
#[inline]
pub fn re_split_apply(recv: &Val, pat: &str) -> Option<Val> {
    try_re_split_apply(recv, pat).ok().flatten()
}

/// Fallible variant of [`re_split_apply`]; propagates regex compilation errors.
#[inline]
pub fn try_re_split_apply(recv: &Val, pat: &str) -> Result<Option<Val>, EvalError> {
    let Some(s) = recv.as_str_ref() else {
        return Ok(None);
    };
    let re = compile_regex_eval(pat)?;
    let out: Vec<Arc<str>> = re.split(s).map(Arc::<str>::from).collect();
    Ok(Some(Val::str_vec(out)))
}

/// Returns `Val::Bool(true)` when the string contains at least one of the `needles`.
#[inline]
pub fn contains_any_apply(recv: &Val, needles: &[Arc<str>]) -> Option<Val> {
    let s = recv.as_str_ref()?;
    Some(Val::Bool(needles.iter().any(|n| s.contains(n.as_ref()))))
}

/// Returns `Val::Bool(true)` when the string contains every one of the `needles`.
#[inline]
pub fn contains_all_apply(recv: &Val, needles: &[Arc<str>]) -> Option<Val> {
    let s = recv.as_str_ref()?;
    Some(Val::Bool(needles.iter().all(|n| s.contains(n.as_ref()))))
}

/// Serialises an array of arrays/objects to CSV format (comma-delimited).
#[inline]
pub fn to_csv_apply(recv: &Val) -> Option<Val> {
    Some(Val::Str(Arc::from(
        crate::builtin_helpers::csv_emit(recv, ",").as_str(),
    )))
}

/// Serialises an array of arrays/objects to TSV format (tab-delimited).
#[inline]
pub fn to_tsv_apply(recv: &Val) -> Option<Val> {
    Some(Val::Str(Arc::from(
        crate::builtin_helpers::csv_emit(recv, "\t").as_str(),
    )))
}

/// Converts an object into `[{key, val}, …]`; returns an empty array for non-objects.
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

/// Returns the runtime type name of `recv` as a `Val::Str` (e.g. `"Int"`, `"Array"`, `"Object"`).
#[inline]
pub fn type_name_apply(recv: &Val) -> Option<Val> {
    Some(Val::Str(Arc::from(recv.type_name())))
}

/// Coerces any `Val` to its human-readable string representation.
#[inline]
pub fn to_string_apply(recv: &Val) -> Option<Val> {
    Some(Val::Str(Arc::from(
        crate::util::val_to_string(recv).as_str(),
    )))
}

/// Serialises `recv` to a compact JSON string; non-finite floats become `"null"`.
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

/// Parses a JSON string into a `Val`; silently returns `None` on parse errors.
#[inline]
pub fn from_json_apply(recv: &Val) -> Option<Val> {
    try_from_json_apply(recv).ok().flatten()
}

/// Fallible variant of [`from_json_apply`]; returns an `EvalError` on invalid JSON.
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

/// Returns `recv` if it is non-null, otherwise returns `default`.
#[inline]
pub fn or_apply(recv: &Val, default: &Val) -> Val {
    if recv.is_null() {
        default.clone()
    } else {
        recv.clone()
    }
}

/// Returns `Val::Bool(true)` when `key` is absent or null at any nesting level inside `recv`.
#[inline]
pub fn missing_apply(recv: &Val, key: &str) -> Val {
    Val::Bool(!crate::util::field_exists_nested(recv, key))
}

/// Membership test: arrays/vectors check element presence, strings check substring, objects check key.
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

/// Infers a JSON-Schema-like descriptor `Val` from a `Val` instance, recursing into objects and arrays.
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

/// Builds an `Object` schema descriptor from an iterator of `(key, value)` pairs.
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

/// Constructs a minimal `{type: name}` schema object.
fn ty_obj(name: &str) -> Val {
    let mut m: IndexMap<Arc<str>, Val> = IndexMap::with_capacity(1);
    m.insert(Arc::from("type"), Val::Str(Arc::from(name)));
    Val::obj(m)
}

/// Constructs an `{type: "Array", len, items}` schema object.
fn array_schema(len: usize, items: Val) -> Val {
    let mut m: IndexMap<Arc<str>, Val> = IndexMap::with_capacity(3);
    m.insert(Arc::from("type"), Val::Str(Arc::from("Array")));
    m.insert(Arc::from("len"), Val::Int(len as i64));
    m.insert(Arc::from("items"), items);
    Val::obj(m)
}

/// Inserts or overwrites a single field in a schema object; returns `obj` unchanged if not an `Obj`.
fn set_schema_field(obj: Val, key: &str, v: Val) -> Val {
    if let Val::Obj(m) = obj {
        let mut m = Arc::try_unwrap(m).unwrap_or_else(|arc| (*arc).clone());
        m.insert(Arc::from(key), v);
        Val::obj(m)
    } else {
        obj
    }
}

/// Extracts the `"type"` string from a schema object, returning `None` for non-schema values.
fn schema_type(v: &Val) -> Option<&str> {
    if let Val::Obj(m) = v {
        if let Some(Val::Str(s)) = m.get("type") {
            return Some(s.as_ref());
        }
    }
    None
}

/// Merges two schema descriptors into one, widening types as needed (same type → recurse, mismatch → `Mixed`).
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

/// Marks schema `a` as nullable if either `a` or `b` is already nullable; otherwise returns `a` unchanged.
fn mark_nullable_if_either(a: Val, b: Val) -> Val {
    if is_schema_nullable(&a) || is_schema_nullable(&b) {
        set_schema_field(a, "nullable", Val::Bool(true))
    } else {
        a
    }
}

/// Returns `true` when the schema object carries `nullable: true`.
fn is_schema_nullable(v: &Val) -> bool {
    matches!(
        v,
        Val::Obj(m) if matches!(m.get("nullable"), Some(Val::Bool(true)))
    )
}

/// Unifies two `Array` schemas: recursively unifies item schemas and sums lengths.
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

/// Extracts a field from a schema object by key, returning `None` when absent.
fn extract_schema_field(v: &Val, key: &str) -> Option<Val> {
    if let Val::Obj(m) = v {
        m.get(key).cloned()
    } else {
        None
    }
}

/// Extracts an integer field from a schema object; returns `None` when absent or not an integer.
fn extract_schema_int(v: &Val, key: &str) -> Option<i64> {
    if let Some(Val::Int(n)) = extract_schema_field(v, key) {
        Some(n)
    } else {
        None
    }
}

/// Unifies two `Object` schemas: merges field schemas, marks fields present in only one as optional.
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

/// Extracts the set of required field names from a schema object's `"required"` array.
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

/// Public adapter for [`schema_of`]: infers and returns a schema descriptor for any `Val`.
#[inline]
pub fn schema_apply(recv: &Val) -> Option<Val> {
    Some(schema_of(recv))
}

#[cfg(test)]
mod spec_tests {
    use super::{
        BuiltinCardinality, BuiltinCategory, BuiltinColumnarStage, BuiltinKeyedReducer,
        BuiltinMethod, BuiltinNumericReducer, BuiltinSelectionPosition, BuiltinSinkAccumulator,
        BuiltinSinkDemand, BuiltinSinkValueNeed, BuiltinStageMerge, BuiltinStructural,
        BuiltinViewInputMode, BuiltinViewMaterialization, BuiltinViewOutputMode, BuiltinViewStage,
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
        assert_eq!(
            BuiltinMethod::Take.spec().view_stage,
            Some(BuiltinViewStage::Take)
        );
        assert_eq!(
            BuiltinMethod::Take.spec().stage_merge,
            Some(BuiltinStageMerge::UsizeMin)
        );
        assert_eq!(
            BuiltinMethod::Skip.spec().view_stage,
            Some(BuiltinViewStage::Skip)
        );
        assert_eq!(
            BuiltinMethod::Skip.spec().stage_merge,
            Some(BuiltinStageMerge::UsizeSaturatingAdd)
        );

        assert_eq!(BuiltinMethod::Sort.spec().view_stage, None);
        assert_eq!(BuiltinMethod::Upper.spec().view_stage, None);
    }

    #[test]
    fn builtin_specs_drive_structural_lowering() {
        assert_eq!(
            BuiltinMethod::DeepShape.spec().structural,
            Some(BuiltinStructural::DeepShape)
        );
        assert_eq!(
            BuiltinMethod::DeepLike.spec().structural,
            Some(BuiltinStructural::DeepLike)
        );
        assert_eq!(
            BuiltinMethod::DeepFind.spec().structural,
            Some(BuiltinStructural::DeepFind)
        );
    }

    #[test]
    fn builtin_view_stage_metadata_describes_view_flow() {
        assert_eq!(
            BuiltinViewStage::Filter.input_mode(),
            BuiltinViewInputMode::ReadsView
        );
        assert_eq!(
            BuiltinViewStage::Take.input_mode(),
            BuiltinViewInputMode::SkipsViewRead
        );
        assert_eq!(
            BuiltinViewStage::Map.output_mode(),
            BuiltinViewOutputMode::BorrowedSubview
        );
        assert_eq!(
            BuiltinViewStage::FlatMap.output_mode(),
            BuiltinViewOutputMode::BorrowedSubviews
        );
        assert_eq!(
            BuiltinViewStage::Skip.output_mode(),
            BuiltinViewOutputMode::PreservesInputView
        );
        assert_eq!(
            BuiltinViewStage::Filter.materialization(),
            BuiltinViewMaterialization::Never
        );
    }

    #[test]
    fn builtin_specs_drive_sink_lowering() {
        assert_eq!(
            BuiltinMethod::Count.spec().sink.unwrap().accumulator,
            BuiltinSinkAccumulator::Count
        );
        assert_eq!(
            BuiltinMethod::Len.spec().sink.unwrap().accumulator,
            BuiltinSinkAccumulator::Count
        );
        assert_eq!(
            BuiltinMethod::Sum.spec().sink.unwrap().accumulator,
            BuiltinSinkAccumulator::Numeric
        );
        assert_eq!(
            BuiltinMethod::Sum.spec().numeric_reducer,
            Some(BuiltinNumericReducer::Sum)
        );
        assert_eq!(
            BuiltinMethod::Avg.spec().numeric_reducer,
            Some(BuiltinNumericReducer::Avg)
        );
        assert_eq!(
            BuiltinMethod::Min.spec().numeric_reducer,
            Some(BuiltinNumericReducer::Min)
        );
        assert_eq!(
            BuiltinMethod::Max.spec().numeric_reducer,
            Some(BuiltinNumericReducer::Max)
        );
        assert_eq!(
            BuiltinMethod::First.spec().sink.unwrap().accumulator,
            BuiltinSinkAccumulator::SelectOne(BuiltinSelectionPosition::First)
        );
        assert_eq!(
            BuiltinMethod::First.spec().sink.unwrap().demand,
            BuiltinSinkDemand::First {
                value: BuiltinSinkValueNeed::Whole
            }
        );
        assert_eq!(
            BuiltinMethod::Last.spec().sink.unwrap().accumulator,
            BuiltinSinkAccumulator::SelectOne(BuiltinSelectionPosition::Last)
        );
        assert_eq!(
            BuiltinMethod::Last.spec().sink.unwrap().demand,
            BuiltinSinkDemand::All {
                value: BuiltinSinkValueNeed::Whole,
                order: true
            }
        );
        assert_eq!(
            BuiltinMethod::Count.spec().sink.unwrap().demand,
            BuiltinSinkDemand::All {
                value: BuiltinSinkValueNeed::None,
                order: false
            }
        );
        assert_eq!(
            BuiltinMethod::Sum.spec().sink.unwrap().demand,
            BuiltinSinkDemand::All {
                value: BuiltinSinkValueNeed::Numeric,
                order: false
            }
        );

        assert!(BuiltinMethod::Sort.spec().sink.is_none());
    }

    #[test]
    fn builtin_specs_drive_columnar_stage_metadata() {
        assert_eq!(
            BuiltinMethod::Filter.spec().columnar_stage,
            Some(BuiltinColumnarStage::Filter)
        );
        assert_eq!(
            BuiltinMethod::Map.spec().columnar_stage,
            Some(BuiltinColumnarStage::Map)
        );
        assert_eq!(
            BuiltinMethod::FlatMap.spec().columnar_stage,
            Some(BuiltinColumnarStage::FlatMap)
        );
        assert_eq!(
            BuiltinMethod::GroupBy.spec().columnar_stage,
            Some(BuiltinColumnarStage::GroupBy)
        );
        assert_eq!(
            BuiltinMethod::CountBy.spec().keyed_reducer,
            Some(BuiltinKeyedReducer::Count)
        );
        assert_eq!(
            BuiltinMethod::IndexBy.spec().keyed_reducer,
            Some(BuiltinKeyedReducer::Index)
        );
        assert_eq!(
            BuiltinMethod::GroupBy.spec().keyed_reducer,
            Some(BuiltinKeyedReducer::Group)
        );
        assert_eq!(BuiltinMethod::Sort.spec().columnar_stage, None);
    }

    #[test]
    fn builtin_specs_drive_view_scalar_kernels() {
        let supported = [
            BuiltinMethod::Len,
            BuiltinMethod::StartsWith,
            BuiltinMethod::EndsWith,
            BuiltinMethod::Matches,
            BuiltinMethod::IndexOf,
            BuiltinMethod::LastIndexOf,
            BuiltinMethod::ByteLen,
            BuiltinMethod::IsBlank,
            BuiltinMethod::IsNumeric,
            BuiltinMethod::IsAlpha,
            BuiltinMethod::IsAscii,
            BuiltinMethod::ToNumber,
            BuiltinMethod::ToBool,
            BuiltinMethod::Ceil,
            BuiltinMethod::Floor,
            BuiltinMethod::Round,
            BuiltinMethod::Abs,
        ];
        for method in supported {
            assert!(method.is_view_scalar_method());
            assert!(method.spec().view_scalar);
        }
        assert!(!BuiltinMethod::Sort.spec().view_scalar);
        assert!(!BuiltinMethod::FromJson.spec().view_scalar);
    }
}
