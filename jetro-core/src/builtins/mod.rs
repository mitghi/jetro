//! Builtin method catalog and shared algorithm implementations.
//!
//! All three execution backends (VM, pipeline, composed) dispatch here for
//! algorithm bodies. Each builtin exposes two primitives:
//! `*_one(item, eval)` for per-row work and `*_apply(items, eval)` for
//! buffered work. Streaming consumers call `*_one`; barrier consumers call
//! `*_apply`. This module owns the loop and truthy-check logic exactly once.

use crate::context::EvalError;
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
    /// Demand-propagation law for pipeline planning (default: `Identity`).
    pub demand_law: BuiltinDemandLaw,
    /// Streaming executor variant, if any.
    pub executor: Option<BuiltinPipelineExecutor>,
    /// Materialisation policy (default: `Streaming`).
    pub materialization: BuiltinPipelineMaterialization,
    /// Cardinality/cost shape annotation for the pipeline cost estimator.
    pub pipeline_shape: Option<BuiltinPipelineShape>,
    /// How this builtin affects element ordering in the pipeline.
    pub order_effect: Option<BuiltinPipelineOrderEffect>,
    /// Physical stage lowering strategy, if registered.
    pub lowering: Option<BuiltinPipelineLowering>,
    /// Whether the builtin is element-wise vectorisable.
    pub is_element: bool,
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
            demand_law: BuiltinDemandLaw::Identity,
            executor: None,
            materialization: BuiltinPipelineMaterialization::Streaming,
            pipeline_shape: None,
            order_effect: None,
            lowering: None,
            is_element: false,
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

    /// Sets the demand-propagation law for pipeline planning.
    fn demand_law(mut self, law: BuiltinDemandLaw) -> Self {
        self.demand_law = law;
        self
    }

    /// Sets the streaming executor variant.
    fn executor(mut self, ex: BuiltinPipelineExecutor) -> Self {
        self.executor = Some(ex);
        self
    }

    /// Sets the materialization policy.
    fn materialization(mut self, m: BuiltinPipelineMaterialization) -> Self {
        self.materialization = m;
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
            Self::Filter => {
                BuiltinSpec::new(Cat::StreamingFilter, Card::Filtering)
                    .view_stage(BuiltinViewStage::Filter)
                    .columnar_stage(BuiltinColumnarStage::Filter)
                    .cost(10.0)
                    .demand_law(BuiltinDemandLaw::FilterLike)
                    .executor(BuiltinPipelineExecutor::RowFilter)
                    .order_effect(BuiltinPipelineOrderEffect::PredicatePrefix)
                    .lowering(BuiltinPipelineLowering::ExprStage(BuiltinExprStage::Filter))
            }
            Self::Find => {
                BuiltinSpec::new(Cat::StreamingFilter, Card::Filtering)
                    .view_stage(BuiltinViewStage::Filter)
                    .columnar_stage(BuiltinColumnarStage::Filter)
                    .cost(10.0)
                    .demand_law(BuiltinDemandLaw::FilterLike)
                    .executor(BuiltinPipelineExecutor::RowFilter)
                    .order_effect(BuiltinPipelineOrderEffect::PredicatePrefix)
                    .lowering(BuiltinPipelineLowering::ExprStage(BuiltinExprStage::Filter))
            }
            Self::FindAll => {
                BuiltinSpec::new(Cat::StreamingFilter, Card::Filtering)
                    .view_stage(BuiltinViewStage::Filter)
                    .columnar_stage(BuiltinColumnarStage::Filter)
                    .cost(10.0)
                    .demand_law(BuiltinDemandLaw::FilterLike)
                    .executor(BuiltinPipelineExecutor::RowFilter)
                    .order_effect(BuiltinPipelineOrderEffect::PredicatePrefix)
                    .lowering(BuiltinPipelineLowering::ExprStage(BuiltinExprStage::Filter))
            }
            Self::Compact | Self::Remove => {
                BuiltinSpec::new(Cat::StreamingFilter, Card::Filtering).cost(10.0)
            }
            Self::Map => BuiltinSpec::new(Cat::StreamingOneToOne, Card::OneToOne)
                .indexed()
                .view_stage(BuiltinViewStage::Map)
                .columnar_stage(BuiltinColumnarStage::Map)
                .cost(10.0)
                .demand_law(BuiltinDemandLaw::MapLike)
                .executor(BuiltinPipelineExecutor::RowMap)
                .order_effect(BuiltinPipelineOrderEffect::Preserves)
                .lowering(BuiltinPipelineLowering::ExprStage(BuiltinExprStage::Map))
                .element(),
            Self::Enumerate => {
                BuiltinSpec::new(Cat::StreamingOneToOne, Card::OneToOne)
                    .indexed()
                    .cost(10.0)
                    .element()
            }
            Self::Pairwise => {
                BuiltinSpec::new(Cat::StreamingOneToOne, Card::OneToOne)
                    .indexed()
                    .cost(10.0)
                    .element()
            }
            Self::FlatMap => BuiltinSpec::new(Cat::StreamingExpand, Card::Expanding)
                .view_stage(BuiltinViewStage::FlatMap)
                .columnar_stage(BuiltinColumnarStage::FlatMap)
                .cost(10.0)
                .demand_law(BuiltinDemandLaw::FlatMapLike)
                .executor(BuiltinPipelineExecutor::RowFlatMap)
                .materialization(BuiltinPipelineMaterialization::LegacyMaterialized)
                .lowering(BuiltinPipelineLowering::ExprStage(BuiltinExprStage::FlatMap)),
            Self::Flatten | Self::Explode => {
                BuiltinSpec::new(Cat::StreamingExpand, Card::Expanding).cost(10.0)
            }
            Self::Split => BuiltinSpec::new(Cat::StreamingExpand, Card::Expanding)
                .cost(10.0)
                .executor(BuiltinPipelineExecutor::ExpandingBuiltin)
                .materialization(BuiltinPipelineMaterialization::LegacyMaterialized)
                .pipeline_shape(BuiltinPipelineShape::new(BuiltinCardinality::Expanding, true, 2.0, 1.0))
                .lowering(BuiltinPipelineLowering::StringStage(BuiltinStringStage::Split)),
            Self::Lines | Self::Words | Self::Chars | Self::CharsOf | Self::Bytes => {
                BuiltinSpec::new(Cat::StreamingExpand, Card::Expanding)
                    .cost(10.0)
                    .element()
            }
            Self::TakeWhile => BuiltinSpec::new(Cat::StreamingFilter, Card::Filtering)
                .view_stage(BuiltinViewStage::TakeWhile)
                .cost(10.0)
                .demand_law(BuiltinDemandLaw::TakeWhile)
                .executor(BuiltinPipelineExecutor::PrefixWhile { take: true })
                .pipeline_shape(BuiltinPipelineShape::new(BuiltinCardinality::Filtering, true, 10.0, 0.5))
                .order_effect(BuiltinPipelineOrderEffect::PredicatePrefix)
                .lowering(BuiltinPipelineLowering::ExprStage(BuiltinExprStage::TakeWhile)),
            Self::DropWhile => BuiltinSpec::new(Cat::StreamingFilter, Card::Filtering)
                .view_stage(BuiltinViewStage::DropWhile)
                .cost(10.0)
                .executor(BuiltinPipelineExecutor::PrefixWhile { take: false })
                .materialization(BuiltinPipelineMaterialization::LegacyMaterialized)
                .pipeline_shape(BuiltinPipelineShape::new(BuiltinCardinality::Filtering, true, 10.0, 0.5))
                .order_effect(BuiltinPipelineOrderEffect::Blocks)
                .lowering(BuiltinPipelineLowering::ExprStage(BuiltinExprStage::DropWhile)),
            Self::FindFirst => {
                BuiltinSpec::new(Cat::StreamingFilter, Card::Filtering)
                    .cost(10.0)
                    .demand_law(BuiltinDemandLaw::First)
                    .lowering(BuiltinPipelineLowering::TerminalExprStage {
                        stage: BuiltinExprStage::Filter,
                        terminal: BuiltinMethod::First,
                    })
            }
            Self::FindOne => {
                BuiltinSpec::new(Cat::StreamingFilter, Card::Filtering)
                    .cost(10.0)
                    .lowering(BuiltinPipelineLowering::TerminalExprStage {
                        stage: BuiltinExprStage::Filter,
                        terminal: BuiltinMethod::First,
                    })
            }
            Self::Take => BuiltinSpec::new(Cat::Positional, Card::Bounded)
                .view_native()
                .view_stage(BuiltinViewStage::Take)
                .stage_merge(BuiltinStageMerge::UsizeMin)
                .demand_law(BuiltinDemandLaw::Take)
                .executor(BuiltinPipelineExecutor::Position { take: true })
                .order_effect(BuiltinPipelineOrderEffect::Preserves)
                .lowering(BuiltinPipelineLowering::UsizeStage {
                    stage: BuiltinUsizeStage::Take,
                    min: 0,
                }),
            Self::Skip => BuiltinSpec::new(Cat::Positional, Card::Bounded)
                .view_native()
                .view_stage(BuiltinViewStage::Skip)
                .stage_merge(BuiltinStageMerge::UsizeSaturatingAdd)
                .demand_law(BuiltinDemandLaw::Skip)
                .executor(BuiltinPipelineExecutor::Position { take: false })
                .order_effect(BuiltinPipelineOrderEffect::Preserves)
                .lowering(BuiltinPipelineLowering::UsizeStage {
                    stage: BuiltinUsizeStage::Skip,
                    min: 0,
                }),
            Self::First => BuiltinSpec::new(Cat::Positional, Card::Bounded)
                .view_native()
                .select_one_sink(BuiltinSelectionPosition::First)
                .demand_law(BuiltinDemandLaw::First)
                .lowering(BuiltinPipelineLowering::TerminalSink),
            Self::Last => BuiltinSpec::new(Cat::Positional, Card::Bounded)
                .view_native()
                .select_one_sink(BuiltinSelectionPosition::Last)
                .demand_law(BuiltinDemandLaw::Last)
                .lowering(BuiltinPipelineLowering::TerminalSink),
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
                .cost(10.0)
                .demand_law(BuiltinDemandLaw::NumericReducer)
                .lowering(BuiltinPipelineLowering::TerminalSink),
            Self::Avg => BuiltinSpec::new(Cat::Reducer, Card::Reducing)
                .view_native()
                .numeric_sink(BuiltinNumericReducer::Avg)
                .cost(10.0)
                .demand_law(BuiltinDemandLaw::NumericReducer)
                .lowering(BuiltinPipelineLowering::TerminalSink),
            Self::Min => BuiltinSpec::new(Cat::Reducer, Card::Reducing)
                .view_native()
                .numeric_sink(BuiltinNumericReducer::Min)
                .cost(10.0)
                .demand_law(BuiltinDemandLaw::NumericReducer)
                .lowering(BuiltinPipelineLowering::TerminalSink),
            Self::Max => BuiltinSpec::new(Cat::Reducer, Card::Reducing)
                .view_native()
                .numeric_sink(BuiltinNumericReducer::Max)
                .cost(10.0)
                .demand_law(BuiltinDemandLaw::NumericReducer)
                .lowering(BuiltinPipelineLowering::TerminalSink),
            Self::Count => BuiltinSpec::new(Cat::Reducer, Card::Reducing)
                .view_native()
                .count_sink()
                .cost(10.0)
                .demand_law(BuiltinDemandLaw::Count)
                .lowering(BuiltinPipelineLowering::TerminalSink),
            Self::ApproxCountDistinct => BuiltinSpec::new(Cat::Reducer, Card::Reducing)
                .view_native()
                .approx_distinct_sink()
                .cost(10.0)
                .demand_law(BuiltinDemandLaw::KeyedReducer)
                .lowering(BuiltinPipelineLowering::TerminalSink),
            Self::Any | Self::All => {
                BuiltinSpec::new(Cat::Reducer, Card::Reducing)
                    .view_native()
                    .cost(10.0)
            }
            Self::FindIndex => {
                BuiltinSpec::new(Cat::Reducer, Card::Reducing)
                    .view_native()
                    .cost(10.0)
                    .executor(BuiltinPipelineExecutor::FindIndex)
                    .materialization(BuiltinPipelineMaterialization::LegacyMaterialized)
                    .pipeline_shape(BuiltinPipelineShape::new(BuiltinCardinality::OneToOne, true, 1.0, 1.0))
                    .lowering(BuiltinPipelineLowering::TerminalExprStage {
                        stage: BuiltinExprStage::FindIndex,
                        terminal: BuiltinMethod::First,
                    })
            }
            Self::IndicesWhere => {
                BuiltinSpec::new(Cat::Reducer, Card::Reducing)
                    .view_native()
                    .cost(10.0)
                    .executor(BuiltinPipelineExecutor::IndicesWhere)
                    .materialization(BuiltinPipelineMaterialization::LegacyMaterialized)
                    .pipeline_shape(BuiltinPipelineShape::new(BuiltinCardinality::OneToOne, true, 1.0, 1.0))
                    .lowering(BuiltinPipelineLowering::TerminalExprStage {
                        stage: BuiltinExprStage::IndicesWhere,
                        terminal: BuiltinMethod::First,
                    })
            }
            Self::MaxBy => {
                BuiltinSpec::new(Cat::Reducer, Card::Reducing)
                    .view_native()
                    .cost(10.0)
                    .executor(BuiltinPipelineExecutor::ArgExtreme { max: true })
                    .materialization(BuiltinPipelineMaterialization::LegacyMaterialized)
                    .pipeline_shape(BuiltinPipelineShape::new(BuiltinCardinality::OneToOne, true, 1.0, 1.0))
                    .lowering(BuiltinPipelineLowering::TerminalExprStage {
                        stage: BuiltinExprStage::MaxBy,
                        terminal: BuiltinMethod::First,
                    })
            }
            Self::MinBy => {
                BuiltinSpec::new(Cat::Reducer, Card::Reducing)
                    .view_native()
                    .cost(10.0)
                    .executor(BuiltinPipelineExecutor::ArgExtreme { max: false })
                    .materialization(BuiltinPipelineMaterialization::LegacyMaterialized)
                    .pipeline_shape(BuiltinPipelineShape::new(BuiltinCardinality::OneToOne, true, 1.0, 1.0))
                    .lowering(BuiltinPipelineLowering::TerminalExprStage {
                        stage: BuiltinExprStage::MinBy,
                        terminal: BuiltinMethod::First,
                    })
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
                    Self::Sort => spec
                        .demand_law(BuiltinDemandLaw::OrderBarrier)
                        .executor(BuiltinPipelineExecutor::Sort)
                        .materialization(BuiltinPipelineMaterialization::ComposedBarrier)
                        .lowering(BuiltinPipelineLowering::Sort),
                    Self::Window => spec
                        .executor(BuiltinPipelineExecutor::Window)
                        .materialization(BuiltinPipelineMaterialization::LegacyMaterialized)
                        .pipeline_shape(BuiltinPipelineShape::new(BuiltinCardinality::Barrier, true, 2.0, 1.0))
                        .lowering(BuiltinPipelineLowering::UsizeStage {
                            stage: BuiltinUsizeStage::Window,
                            min: 1,
                        }),
                    Self::Chunk => spec
                        .executor(BuiltinPipelineExecutor::Chunk)
                        .materialization(BuiltinPipelineMaterialization::LegacyMaterialized)
                        .pipeline_shape(BuiltinPipelineShape::new(BuiltinCardinality::Barrier, true, 2.0, 1.0))
                        .lowering(BuiltinPipelineLowering::UsizeStage {
                            stage: BuiltinUsizeStage::Chunk,
                            min: 1,
                        }),
                    _ => spec,
                }
            }
            Self::GroupBy => BuiltinSpec::new(Cat::Reducer, Card::Reducing)
                .view_stage(BuiltinViewStage::KeyedReduce)
                .keyed_reducer(BuiltinKeyedReducer::Group)
                .columnar_stage(BuiltinColumnarStage::GroupBy)
                .cost(20.0)
                .demand_law(BuiltinDemandLaw::KeyedReducer)
                .executor(BuiltinPipelineExecutor::GroupBy)
                .materialization(BuiltinPipelineMaterialization::ComposedBarrier)
                .lowering(BuiltinPipelineLowering::ExprStage(BuiltinExprStage::GroupBy)),
            Self::CountBy => BuiltinSpec::new(Cat::Reducer, Card::Reducing)
                .view_stage(BuiltinViewStage::KeyedReduce)
                .keyed_reducer(BuiltinKeyedReducer::Count)
                .cost(10.0)
                .demand_law(BuiltinDemandLaw::KeyedReducer)
                .executor(BuiltinPipelineExecutor::CountBy)
                .pipeline_shape(BuiltinPipelineShape::new(BuiltinCardinality::OneToOne, true, 1.0, 1.0))
                .lowering(BuiltinPipelineLowering::TerminalExprStage {
                    stage: BuiltinExprStage::CountBy,
                    terminal: BuiltinMethod::First,
                }),
            Self::IndexBy => BuiltinSpec::new(Cat::Reducer, Card::Reducing)
                .view_stage(BuiltinViewStage::KeyedReduce)
                .keyed_reducer(BuiltinKeyedReducer::Index)
                .cost(10.0)
                .demand_law(BuiltinDemandLaw::KeyedReducer)
                .executor(BuiltinPipelineExecutor::IndexBy)
                .pipeline_shape(BuiltinPipelineShape::new(BuiltinCardinality::OneToOne, true, 1.0, 1.0))
                .lowering(BuiltinPipelineLowering::TerminalExprStage {
                    stage: BuiltinExprStage::IndexBy,
                    terminal: BuiltinMethod::First,
                }),
            Self::Unique => {
                BuiltinSpec::new(Cat::StreamingFilter, Card::Filtering)
                    .view_stage(BuiltinViewStage::Distinct)
                    .cost(10.0)
                    .demand_law(BuiltinDemandLaw::UniqueLike)
                    .executor(BuiltinPipelineExecutor::UniqueBy)
                    .pipeline_shape(BuiltinPipelineShape::new(BuiltinCardinality::Filtering, true, 10.0, 1.0))
                    .order_effect(BuiltinPipelineOrderEffect::Preserves)
                    .lowering(BuiltinPipelineLowering::NullaryStage(BuiltinNullaryStage::Unique))
            }
            Self::UniqueBy => {
                BuiltinSpec::new(Cat::StreamingFilter, Card::Filtering)
                    .view_stage(BuiltinViewStage::Distinct)
                    .cost(10.0)
                    .demand_law(BuiltinDemandLaw::UniqueLike)
                    .executor(BuiltinPipelineExecutor::UniqueBy)
                    .pipeline_shape(BuiltinPipelineShape::new(BuiltinCardinality::Filtering, true, 10.0, 1.0))
                    .order_effect(BuiltinPipelineOrderEffect::Preserves)
                    .lowering(BuiltinPipelineLowering::ExprStage(BuiltinExprStage::UniqueBy))
            }
            Self::Reverse => {
                BuiltinSpec::new(Cat::Barrier, Card::Barrier)
                    .cost(10.0)
                    .cancellation(BuiltinCancellation::SelfInverse(BuiltinCancelGroup::Reverse))
                    .demand_law(BuiltinDemandLaw::OrderBarrier)
                    .executor(BuiltinPipelineExecutor::Reverse)
                    .materialization(BuiltinPipelineMaterialization::ComposedBarrier)
                    .lowering(BuiltinPipelineLowering::NullaryStage(BuiltinNullaryStage::Reverse))
            }
            Self::Append
            | Self::Prepend
            | Self::Diff
            | Self::Intersect
            | Self::Union
            | Self::Join
            | Self::Zip
            | Self::ZipLongest
            | Self::Fanout
            | Self::ZipShape => {
                BuiltinSpec::new(Cat::Barrier, Card::Barrier).cost(10.0)
            }
            Self::Keys => BuiltinSpec::new(Cat::Object, Card::OneToOne).element(),
            Self::Values => BuiltinSpec::new(Cat::Object, Card::OneToOne).element(),
            Self::Entries => BuiltinSpec::new(Cat::Object, Card::OneToOne).element(),
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
            Self::TransformKeys => {
                BuiltinSpec::new(Cat::Object, Card::OneToOne)
                    .executor(BuiltinPipelineExecutor::ObjectLambda)
                    .pipeline_shape(BuiltinPipelineShape::new(BuiltinCardinality::OneToOne, true, 1.0, 1.0))
                    .order_effect(BuiltinPipelineOrderEffect::Preserves)
                    .lowering(BuiltinPipelineLowering::ExprStage(BuiltinExprStage::TransformKeys))
            }
            Self::TransformValues => {
                BuiltinSpec::new(Cat::Object, Card::OneToOne)
                    .executor(BuiltinPipelineExecutor::ObjectLambda)
                    .pipeline_shape(BuiltinPipelineShape::new(BuiltinCardinality::OneToOne, true, 1.0, 1.0))
                    .order_effect(BuiltinPipelineOrderEffect::Preserves)
                    .lowering(BuiltinPipelineLowering::ExprStage(BuiltinExprStage::TransformValues))
            }
            Self::FilterKeys => {
                BuiltinSpec::new(Cat::Object, Card::OneToOne)
                    .executor(BuiltinPipelineExecutor::ObjectLambda)
                    .pipeline_shape(BuiltinPipelineShape::new(BuiltinCardinality::OneToOne, true, 1.0, 1.0))
                    .order_effect(BuiltinPipelineOrderEffect::Preserves)
                    .lowering(BuiltinPipelineLowering::ExprStage(BuiltinExprStage::FilterKeys))
            }
            Self::FilterValues => {
                BuiltinSpec::new(Cat::Object, Card::OneToOne)
                    .executor(BuiltinPipelineExecutor::ObjectLambda)
                    .pipeline_shape(BuiltinPipelineShape::new(BuiltinCardinality::OneToOne, true, 1.0, 1.0))
                    .order_effect(BuiltinPipelineOrderEffect::Preserves)
                    .lowering(BuiltinPipelineLowering::ExprStage(BuiltinExprStage::FilterValues))
            }
            Self::GetPath => {
                BuiltinSpec::new(Cat::Path, Card::OneToOne).indexed().element()
            }
            Self::DelPath => {
                BuiltinSpec::new(Cat::Path, Card::OneToOne).indexed().element()
            }
            Self::HasPath => {
                BuiltinSpec::new(Cat::Path, Card::OneToOne).indexed().element()
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
            Self::Set => BuiltinSpec::new(Cat::Mutation, Card::OneToOne).indexed().element(),
            Self::Update => BuiltinSpec::new(Cat::Mutation, Card::OneToOne).indexed(),
            Self::Lag => BuiltinSpec::new(Cat::StreamingOneToOne, Card::OneToOne)
                .indexed()
                .cost(10.0)
                .element(),
            Self::Lead => BuiltinSpec::new(Cat::StreamingOneToOne, Card::OneToOne)
                .indexed()
                .cost(10.0)
                .element(),
            Self::DiffWindow => BuiltinSpec::new(Cat::StreamingOneToOne, Card::OneToOne)
                .indexed()
                .cost(10.0)
                .element(),
            Self::PctChange => BuiltinSpec::new(Cat::StreamingOneToOne, Card::OneToOne)
                .indexed()
                .cost(10.0)
                .element(),
            Self::CumMax => BuiltinSpec::new(Cat::StreamingOneToOne, Card::OneToOne)
                .indexed()
                .cost(10.0)
                .element(),
            Self::CumMin => BuiltinSpec::new(Cat::StreamingOneToOne, Card::OneToOne)
                .indexed()
                .cost(10.0)
                .element(),
            Self::Zscore => BuiltinSpec::new(Cat::StreamingOneToOne, Card::OneToOne)
                .indexed()
                .cost(10.0)
                .element(),
            Self::Or => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .element(),
            Self::Has => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .element(),
            Self::Ceil => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .view_scalar()
                .element(),
            Self::Floor => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .view_scalar()
                .element(),
            Self::Round => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .view_scalar()
                .element(),
            Self::Abs => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .view_scalar()
                .element(),
            Self::Upper => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .view_scalar()
                .element(),
            Self::Lower => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .view_scalar()
                .element(),
            Self::Capitalize => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .element(),
            Self::TitleCase => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .element(),
            Self::Trim => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .view_scalar()
                .element(),
            Self::TrimLeft => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .view_scalar()
                .element(),
            Self::TrimRight => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .view_scalar()
                .element(),
            Self::SnakeCase => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .element(),
            Self::KebabCase => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .element(),
            Self::CamelCase => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .element(),
            Self::PascalCase => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .element(),
            Self::ReverseStr => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .element(),
            Self::IsBlank => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .view_scalar()
                .element(),
            Self::IsNumeric => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .view_scalar()
                .element(),
            Self::IsAlpha => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .view_scalar()
                .element(),
            Self::IsAscii => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .view_scalar()
                .element(),
            Self::ToNumber => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .view_scalar()
                .element(),
            Self::ToBool => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .view_scalar()
                .element(),
            Self::ParseInt => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .element(),
            Self::ParseFloat => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .element(),
            Self::ParseBool => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .element(),
            Self::ToBase64 => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .element(),
            Self::FromBase64 => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .element(),
            Self::UrlEncode => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .element(),
            Self::UrlDecode => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .element(),
            Self::HtmlEscape => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .element(),
            Self::HtmlUnescape => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .element(),
            Self::Repeat => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .element(),
            Self::PadLeft => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .element(),
            Self::PadRight => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .element(),
            Self::Center => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .element(),
            Self::StartsWith => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .view_scalar()
                .element(),
            Self::EndsWith => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .view_scalar()
                .element(),
            Self::IndexOf => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .view_scalar()
                .element(),
            Self::LastIndexOf => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .view_scalar()
                .element(),
            Self::StripPrefix => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .element(),
            Self::StripSuffix => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .element(),
            Self::Matches => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .view_scalar()
                .element(),
            Self::Scan => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .element(),
            Self::ReMatch => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .element(),
            Self::ReMatchFirst => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .element(),
            Self::ReMatchAll => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .element(),
            Self::ReCaptures => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .element(),
            Self::ReCapturesAll => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .element(),
            Self::ReSplit => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .element(),
            Self::ReReplace => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .element(),
            Self::ReReplaceAll => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .element(),
            Self::ContainsAny => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .element(),
            Self::ContainsAll => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .element(),
            Self::Schema => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .element(),
            Self::ByteLen => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .view_scalar()
                .element(),
            Self::Unknown => BuiltinSpec {
                pure: false,
                ..BuiltinSpec::new(Cat::Unknown, Card::OneToOne)
            },
            Self::Slice => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .executor(BuiltinPipelineExecutor::ElementBuiltin)
                .pipeline_shape(BuiltinPipelineShape::new(BuiltinCardinality::OneToOne, true, 1.0, 1.0))
                .order_effect(BuiltinPipelineOrderEffect::Preserves)
                .lowering(BuiltinPipelineLowering::IntRangeStage(BuiltinIntRangeStage::Slice)),
            Self::Replace => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .executor(BuiltinPipelineExecutor::ElementBuiltin)
                .pipeline_shape(BuiltinPipelineShape::new(BuiltinCardinality::OneToOne, true, 2.0, 1.0))
                .order_effect(BuiltinPipelineOrderEffect::Preserves)
                .lowering(BuiltinPipelineLowering::StringPairStage(
                    BuiltinStringPairStage::Replace { all: false }
                )),
            Self::ReplaceAll => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .executor(BuiltinPipelineExecutor::ElementBuiltin)
                .pipeline_shape(BuiltinPipelineShape::new(BuiltinCardinality::OneToOne, true, 2.0, 1.0))
                .order_effect(BuiltinPipelineOrderEffect::Preserves)
                .lowering(BuiltinPipelineLowering::StringPairStage(
                    BuiltinStringPairStage::Replace { all: true }
                )),
            Self::Type => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .element(),
            Self::ToString => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .element(),
            Self::ToJson => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .element(),
            Self::Indent => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .element(),
            Self::Dedent => BuiltinSpec::new(Cat::Scalar, Card::OneToOne)
                .indexed()
                .view_native()
                .element(),
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


pub mod array;
pub mod collection;
pub mod misc;
pub mod path;
pub mod regex;
pub mod schema;
pub mod string;

pub use array::*;
pub use collection::*;
pub use misc::*;
pub use path::*;
pub use regex::*;
pub use schema::*;
pub use string::*;
