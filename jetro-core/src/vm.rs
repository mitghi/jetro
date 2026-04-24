//! High-performance bytecode VM for v2 Jetro expressions.
//!
//! # Architecture
//!
//! ```text
//!  String expression
//!        │  parser::parse()
//!        ▼
//!     Expr (AST)
//!        │  Compiler::compile()
//!        ▼
//!     Program              ← flat Arc<[Opcode]>  (cached: compile_cache)
//!        │  VM::execute()
//!        ▼
//!      Val                 ← result              (structural: resolution_cache)
//! ```
//!
//! # Optimisations over the tree-walker
//!
//! 1. **Compile cache** — parse + compile once per unique expression string.
//! 2. **Val type** — `Arc`-wrapped compound nodes; every clone is O(1).
//! 3. **BuiltinMethod enum** — O(1) method dispatch (jump-table vs string hash).
//! 4. **Pre-compiled sub-programs** — lambda/arg bodies compiled to `Arc<Program>`
//!    once at compile time; never re-compiled per call.
//! 5. **Resolution cache** — structural programs (`$.a.b[0]`) cache their
//!    pointer path after the first traversal; subsequent calls skip traversal.
//! 6. **Peephole pass 1 — RootChain** — `PushRoot + GetField+` fused into a
//!    single pointer-resolve opcode.
//! 7. **Peephole pass 2 — FilterCount** — `CallMethod(filter) +
//!    CallMethod(len/count)` fused; counts matches without materialising the
//!    intermediate filtered array.
//! 8. **Peephole pass 3 — ConstFold** — arithmetic on adjacent integer literals
//!    folded at compile time.
//! 9. **Stack machine** — iterative `exec()` loop; no per-opcode stack-frame
//!    overhead for simple navigation / arithmetic opcodes.

use std::{
    collections::{HashMap, VecDeque},
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    sync::Arc,
    sync::atomic::{AtomicU64, Ordering},
};
use indexmap::IndexMap;
use memchr::memchr;
use memchr::memmem;
use smallvec::SmallVec;

use crate::ast::*;
use super::eval::{
    Env, EvalError, Val,
    dispatch_method, eval,
};
use super::eval::util::{
    is_truthy, kind_matches, vals_eq, cmp_vals, val_to_key, val_to_string,
    add_vals, num_op, obj2,
};
use super::eval::methods::MethodRegistry;

macro_rules! pop {
    ($stack:expr) => {
        $stack.pop().ok_or_else(|| EvalError("stack underflow".into()))?
    };
}
macro_rules! err {
    ($($t:tt)*) => { Err(EvalError(format!($($t)*))) };
}

// ── BuiltinMethod ─────────────────────────────────────────────────────────────

/// Pre-resolved method identifier — eliminates string comparison at dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BuiltinMethod {
    // Navigation / basics
    Len = 0, Keys, Values, Entries, ToPairs, FromPairs, Invert, Reverse, Type,
    ToString, ToJson, FromJson,
    // Aggregates
    Sum, Avg, Min, Max, Count, Any, All,
    GroupBy, CountBy, IndexBy,
    // Array ops
    Filter, Map, FlatMap, Sort, Unique, Flatten, Compact,
    Join, First, Last, Nth, Append, Prepend, Remove,
    Diff, Intersect, Union, Enumerate, Pairwise, Window, Chunk,
    TakeWhile, DropWhile, Accumulate, Partition, Zip, ZipLongest,
    // Object ops
    Pick, Omit, Merge, DeepMerge, Defaults, Rename,
    TransformKeys, TransformValues, FilterKeys, FilterValues, Pivot,
    // Path ops
    GetPath, SetPath, DelPath, DelPaths, HasPath, FlattenKeys, UnflattenKeys,
    // CSV
    ToCsv, ToTsv,
    // Null / predicate
    Or, Has, Missing, Includes, Set, Update,
    // String methods
    Upper, Lower, Capitalize, TitleCase, Trim, TrimLeft, TrimRight,
    Lines, Words, Chars, ToNumber, ToBool, ToBase64, FromBase64,
    UrlEncode, UrlDecode, HtmlEscape, HtmlUnescape,
    Repeat, PadLeft, PadRight, StartsWith, EndsWith,
    IndexOf, LastIndexOf, Replace, ReplaceAll, StripPrefix, StripSuffix,
    Slice, Split, Indent, Dedent, Matches, Scan,
    // Relational
    EquiJoin,
    // Sentinel for custom/unknown
    Unknown,
}

impl BuiltinMethod {
    pub fn from_name(name: &str) -> Self {
        match name {
            "len"            => Self::Len,
            "keys"           => Self::Keys,
            "values"         => Self::Values,
            "entries"        => Self::Entries,
            "to_pairs"|"toPairs" => Self::ToPairs,
            "from_pairs"|"fromPairs" => Self::FromPairs,
            "invert"         => Self::Invert,
            "reverse"        => Self::Reverse,
            "type"           => Self::Type,
            "to_string"|"toString" => Self::ToString,
            "to_json"|"toJson" => Self::ToJson,
            "from_json"|"fromJson" => Self::FromJson,
            "sum"            => Self::Sum,
            "avg"            => Self::Avg,
            "min"            => Self::Min,
            "max"            => Self::Max,
            "count"          => Self::Count,
            "any"            => Self::Any,
            "all"            => Self::All,
            "groupBy"|"group_by" => Self::GroupBy,
            "countBy"|"count_by" => Self::CountBy,
            "indexBy"|"index_by" => Self::IndexBy,
            "filter"         => Self::Filter,
            "map"            => Self::Map,
            "flatMap"|"flat_map" => Self::FlatMap,
            "sort"           => Self::Sort,
            "unique"|"distinct" => Self::Unique,
            "flatten"        => Self::Flatten,
            "compact"        => Self::Compact,
            "join"           => Self::Join,
            "equi_join"|"equiJoin" => Self::EquiJoin,
            "first"          => Self::First,
            "last"           => Self::Last,
            "nth"            => Self::Nth,
            "append"         => Self::Append,
            "prepend"        => Self::Prepend,
            "remove"         => Self::Remove,
            "diff"           => Self::Diff,
            "intersect"      => Self::Intersect,
            "union"          => Self::Union,
            "enumerate"      => Self::Enumerate,
            "pairwise"       => Self::Pairwise,
            "window"         => Self::Window,
            "chunk"|"batch"  => Self::Chunk,
            "takewhile"|"take_while" => Self::TakeWhile,
            "dropwhile"|"drop_while" => Self::DropWhile,
            "accumulate"     => Self::Accumulate,
            "partition"      => Self::Partition,
            "zip"            => Self::Zip,
            "zip_longest"|"zipLongest" => Self::ZipLongest,
            "pick"           => Self::Pick,
            "omit"           => Self::Omit,
            "merge"          => Self::Merge,
            "deep_merge"|"deepMerge" => Self::DeepMerge,
            "defaults"       => Self::Defaults,
            "rename"         => Self::Rename,
            "transform_keys"|"transformKeys" => Self::TransformKeys,
            "transform_values"|"transformValues" => Self::TransformValues,
            "filter_keys"|"filterKeys" => Self::FilterKeys,
            "filter_values"|"filterValues" => Self::FilterValues,
            "pivot"          => Self::Pivot,
            "get_path"|"getPath" => Self::GetPath,
            "set_path"|"setPath" => Self::SetPath,
            "del_path"|"delPath" => Self::DelPath,
            "del_paths"|"delPaths" => Self::DelPaths,
            "has_path"|"hasPath" => Self::HasPath,
            "flatten_keys"|"flattenKeys" => Self::FlattenKeys,
            "unflatten_keys"|"unflattenKeys" => Self::UnflattenKeys,
            "to_csv"|"toCsv" => Self::ToCsv,
            "to_tsv"|"toTsv" => Self::ToTsv,
            "or"             => Self::Or,
            "has"            => Self::Has,
            "missing"        => Self::Missing,
            "includes"|"contains" => Self::Includes,
            "set"            => Self::Set,
            "update"         => Self::Update,
            "upper"          => Self::Upper,
            "lower"          => Self::Lower,
            "capitalize"     => Self::Capitalize,
            "title_case"|"titleCase" => Self::TitleCase,
            "trim"           => Self::Trim,
            "trim_left"|"trimLeft"|"lstrip" => Self::TrimLeft,
            "trim_right"|"trimRight"|"rstrip" => Self::TrimRight,
            "lines"          => Self::Lines,
            "words"          => Self::Words,
            "chars"          => Self::Chars,
            "to_number"|"toNumber" => Self::ToNumber,
            "to_bool"|"toBool" => Self::ToBool,
            "to_base64"|"toBase64" => Self::ToBase64,
            "from_base64"|"fromBase64" => Self::FromBase64,
            "url_encode"|"urlEncode" => Self::UrlEncode,
            "url_decode"|"urlDecode" => Self::UrlDecode,
            "html_escape"|"htmlEscape" => Self::HtmlEscape,
            "html_unescape"|"htmlUnescape" => Self::HtmlUnescape,
            "repeat"         => Self::Repeat,
            "pad_left"|"padLeft" => Self::PadLeft,
            "pad_right"|"padRight" => Self::PadRight,
            "starts_with"|"startsWith" => Self::StartsWith,
            "ends_with"|"endsWith" => Self::EndsWith,
            "index_of"|"indexOf" => Self::IndexOf,
            "last_index_of"|"lastIndexOf" => Self::LastIndexOf,
            "replace"        => Self::Replace,
            "replace_all"|"replaceAll" => Self::ReplaceAll,
            "strip_prefix"|"stripPrefix" => Self::StripPrefix,
            "strip_suffix"|"stripSuffix" => Self::StripSuffix,
            "slice"          => Self::Slice,
            "split"          => Self::Split,
            "indent"         => Self::Indent,
            "dedent"         => Self::Dedent,
            "matches"        => Self::Matches,
            "scan"           => Self::Scan,
            _                => Self::Unknown,
        }
    }

    /// True for methods that receive a sub-program to run per item.
    fn is_lambda_method(self) -> bool {
        matches!(self,
            Self::Filter | Self::Map | Self::FlatMap | Self::Sort |
            Self::Any | Self::All | Self::Count | Self::GroupBy |
            Self::CountBy | Self::IndexBy | Self::TakeWhile |
            Self::DropWhile | Self::Accumulate | Self::Partition |
            Self::TransformKeys | Self::TransformValues |
            Self::FilterKeys | Self::FilterValues | Self::Pivot | Self::Update
        )
    }
}

// ── Compiled sub-structures ───────────────────────────────────────────────────

/// A compiled method call stored inside `Opcode::CallMethod`.
#[derive(Debug, Clone)]
pub struct CompiledCall {
    pub method:   BuiltinMethod,
    pub name:     Arc<str>,
    /// Compiled lambda/expression sub-programs (one per arg, in order).
    pub sub_progs: Arc<[Arc<Program>]>,
    /// Original AST args kept for non-lambda dispatch fallback.
    pub orig_args: Arc<[Arg]>,
}

/// A compiled object field for `Opcode::MakeObj`.
#[derive(Debug, Clone)]
pub enum CompiledObjEntry {
    /// `{ name }` / `{ name, … }` shorthand — reads `env.current.name`
    /// (or a bound variable of that name).  `ic` is a per-entry inline
    /// cache hint so that repeated MakeObj calls over objects that share
    /// shape skip the IndexMap key-hash on hit.
    Short { name: Arc<str>, ic: Arc<AtomicU64> },
    Kv     { key: Arc<str>, prog: Arc<Program>, optional: bool, cond: Option<Arc<Program>> },
    /// Specialised `Kv` where the value is a pure path from current:
    /// `{ key: @.a.b[0] }` compiles to `KvPath` so `exec_make_obj` can
    /// walk `env.current` through the pre-resolved steps without a
    /// sub-program exec.  `optional=true` mirrors `?` in the source —
    /// the field is omitted when the walk lands on `Null`.
    /// `ics[i]` is an inline-cache slot for `steps[i]` — only used when
    /// the step is `Field`.
    KvPath { key: Arc<str>, steps: Arc<[KvStep]>, optional: bool, ics: Arc<[AtomicU64]> },
    Dynamic { key: Arc<Program>, val: Arc<Program> },
    Spread(Arc<Program>),
    SpreadDeep(Arc<Program>),
}

/// Single step in a pre-resolved `KvPath` projection.
#[derive(Debug, Clone)]
pub enum KvStep {
    Field(Arc<str>),
    Index(i64),
}

/// A compiled f-string interpolation part.
#[derive(Debug, Clone)]
pub enum CompiledFSPart {
    Lit(Arc<str>),
    Interp { prog: Arc<Program>, fmt: Option<FmtSpec> },
}

/// Compiled bind-object destructure spec.
#[derive(Debug, Clone)]
pub struct BindObjSpec {
    pub fields: Arc<[Arc<str>]>,
    pub rest:   Option<Arc<str>>,
}

/// Compiled comprehension spec.
#[derive(Debug, Clone)]
pub struct CompSpec {
    pub expr: Arc<Program>,
    pub vars: Arc<[Arc<str>]>,
    pub iter: Arc<Program>,
    pub cond: Option<Arc<Program>>,
}

#[derive(Debug, Clone)]
pub struct DictCompSpec {
    pub key:  Arc<Program>,
    pub val:  Arc<Program>,
    pub vars: Arc<[Arc<str>]>,
    pub iter: Arc<Program>,
    pub cond: Option<Arc<Program>>,
}

// ── Opcode ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum Opcode {
    // ── Literals ─────────────────────────────────────────────────────────────
    PushNull,
    PushBool(bool),
    PushInt(i64),
    PushFloat(f64),
    PushStr(Arc<str>),

    // ── Context ───────────────────────────────────────────────────────────────
    PushRoot,
    PushCurrent,

    // ── Navigation ────────────────────────────────────────────────────────────
    GetField(Arc<str>),
    GetIndex(i64),
    GetSlice(Option<i64>, Option<i64>),
    DynIndex(Arc<Program>),
    OptField(Arc<str>),
    Descendant(Arc<str>),
    DescendAll,
    InlineFilter(Arc<Program>),
    Quantifier(QuantifierKind),

    // ── Peephole fusions ──────────────────────────────────────────────────────
    /// PushRoot + GetField* fused — resolves chain via pointer arithmetic.
    RootChain(Arc<[Arc<str>]>),
    /// GetField(k1) + GetField(k2) + … fused — walks TOS through N fields.
    /// Applies mid-program where `RootChain` does not match (e.g. after a
    /// method call or filter produces an object on stack).
    /// Carries a per-step inline-cache array so that `map(a.b.c)` over a
    /// shape-uniform array of objects hits `get_index(cached_slot)`
    /// instead of re-hashing the key at every iteration.
    FieldChain(Arc<FieldChainData>),
    /// filter(pred) + len/count fused — counts matches without temp array.
    FilterCount(Arc<Program>),
    /// filter(pred) + First quantifier fused — early-exit on first match.
    FindFirst(Arc<Program>),
    /// filter(pred) + One quantifier fused — early-exit at 2nd match (error).
    FindOne(Arc<Program>),
    /// filter(pred) + map(f) fused — single pass, no intermediate array.
    FilterMap { pred: Arc<Program>, map: Arc<Program> },
    /// filter(p1) + filter(p2) fused — single pass, both predicates.
    FilterFilter { p1: Arc<Program>, p2: Arc<Program> },
    /// map(f1) + map(f2) fused — single pass, composed.
    MapMap { f1: Arc<Program>, f2: Arc<Program> },
    /// map(f) + filter(p) fused — single pass; emit `f(x)` only when `p(f(x))` holds.
    MapFilter { map: Arc<Program>, pred: Arc<Program> },
    /// Fused `map(f).sum()` — evaluates `f` per item, accumulates numeric sum.
    MapSum(Arc<Program>),
    /// Fused `map(@.to_json()).join(sep)` — single-pass stringify + concat.
    /// Skips the intermediate Vec<Val::Str> (N Arc allocations) and writes
    /// each item's JSON form straight into one output buffer.  Columnar
    /// receivers (`Val::IntVec` / `Val::FloatVec`) shortcut further via
    /// native number-formatting in a tight loop.
    MapToJsonJoin { sep_prog: Arc<Program> },
    /// Fused `.trim().upper()` — one allocation instead of two, ASCII fast-path.
    StrTrimUpper,
    /// Fused `.trim().lower()` — one allocation instead of two, ASCII fast-path.
    StrTrimLower,
    /// Fused `.upper().trim()` — one allocation instead of two.
    StrUpperTrim,
    /// Fused `.lower().trim()` — one allocation instead of two.
    StrLowerTrim,
    /// Fused `.split(sep).reverse().join(sep)` — byte-scan segments and
    /// emit reversed join into one buffer.  No intermediate `Vec<Arc<str>>`.
    StrSplitReverseJoin { sep: Arc<str> },
    /// Fused `map(@.replace(lit, lit))` (and `replace_all`) — literal needle
    /// and replacement inlined; skips per-item sub_prog evaluation for arg
    /// strings.  `all=true` means replace every occurrence, else only first.
    MapReplaceLit { needle: Arc<str>, with: Arc<str>, all: bool },
    /// Fused `map(@.upper().replace(lit, lit))` (and `replace_all`) — scan
    /// bytes once: ASCII-upper + memchr needle scan into a single pre-sized
    /// output String per item. Falls back to non-ASCII path for Unicode.
    MapUpperReplaceLit { needle: Arc<str>, with: Arc<str>, all: bool },
    /// Fused `map(@.lower().replace(lit, lit))` (and `replace_all`) — same as
    /// above but ASCII-lower.
    MapLowerReplaceLit { needle: Arc<str>, with: Arc<str>, all: bool },
    /// Fused `map(prefix + @ + suffix)` — per item, allocate exact-size
    /// `Arc<str>` with one uninit slice + copy_nonoverlapping. Either
    /// prefix or suffix may be empty for the 2-operand forms.
    MapStrConcat { prefix: Arc<str>, suffix: Arc<str> },
    /// Fused `map(@.split(sep).map(len).sum())` — emits IntVec of per-row
    /// sum-of-segment-char-lengths. Uses byte-scan (memchr/memmem) for ASCII
    /// source; falls back to char counting for Unicode.
    MapSplitLenSum { sep: Arc<str> },
    /// Fused `map({k1, k2, ..})` — map over an array projecting each object
    /// to a fixed set of `Short`-form fields (bare identifiers). Avoids the
    /// nested `MakeObj` dispatch per row and hoists key `Arc<str>` clones
    /// outside the inner loop. Uses one IC slot per key for shape lookup.
    MapProject { keys: Arc<[Arc<str>]>, ics: Arc<[std::sync::atomic::AtomicU64]> },
    /// Fused `map(@.slice(lit, lit))` — per-row ASCII byte-range slice via
    /// borrowed `Val::StrSlice` into the parent Arc.  Non-ASCII rows fall
    /// through to character-index resolution.  Zero allocation per row
    /// when the input is already `Val::Str` (just an Arc bump for the
    /// borrowed view).
    MapStrSlice { start: i64, end: Option<i64> },
    /// Fused `map(f"…")` — map over an array applying an f-string to each
    /// element. Skips the inner CallMethod dispatch / FString Arc-clone
    /// per row; runs the f-string parts in a tight loop.
    MapFString(Arc<[CompiledFSPart]>),
    /// Fused `map(@.split(sep).count())` — byte-scan per row, returns Int;
    /// zero per-row allocations.
    MapSplitCount { sep: Arc<str> },
    /// Fused `map(@.split(sep).count()).sum()` — scalar Int, no intermediate
    /// `[Int,Int,...]` array. One memchr-backed scan per row, accumulated.
    MapSplitCountSum { sep: Arc<str> },
    /// Fused `map(@.split(sep).first())` — first segment only; one Arc per
    /// row instead of N.
    MapSplitFirst { sep: Arc<str> },
    /// Fused `map(@.split(sep).nth(n))` — nth segment; one Arc per row.
    MapSplitNth { sep: Arc<str>, n: usize },
    /// Fused `map(f).avg()` — evaluates `f` per item, computes mean as float.
    MapAvg(Arc<Program>),
    /// Fused `filter(p).map(f).sum()` — single pass, numeric sum of mapped
    /// values that pass the predicate.  No intermediate array.
    FilterMapSum { pred: Arc<Program>, map: Arc<Program> },
    /// Fused `filter(p).map(f).avg()` — mean as float over mapped values
    /// that pass the predicate.
    FilterMapAvg { pred: Arc<Program>, map: Arc<Program> },
    /// Fused `filter(p).map(f).first()` — early-exit: apply `map` once,
    /// to the first item that satisfies `pred`.
    FilterMapFirst { pred: Arc<Program>, map: Arc<Program> },
    /// Fused `filter(p).map(f).min()` — single pass, numeric min over
    /// mapped values that pass the predicate.  No intermediate array.
    FilterMapMin { pred: Arc<Program>, map: Arc<Program> },
    /// Fused `filter(p).map(f).max()` — single pass, numeric max over
    /// mapped values that pass the predicate.
    FilterMapMax { pred: Arc<Program>, map: Arc<Program> },
    /// Fused `filter(p).last()` — reverse scan, return last item
    /// satisfying `pred` (or Null when none match / input is Null).
    FilterLast { pred: Arc<Program> },
    /// Fused `sort()` + `[0:n]` — partial-sort smallest N using BinaryHeap.
    /// `asc=true` → smallest N; `asc=false` → largest N.
    TopN { n: usize, asc: bool },
    /// Fused `unique()` + `count()`/`len()` — count distinct elements without
    /// materialising the deduped array.
    UniqueCount,
    /// Fused `sort_by(k).first()` / `.last()` — O(N) single pass instead of
    /// an O(N log N) sort followed by discard.  Preserves stable-sort ordering:
    /// `max=false` returns the *earliest* item whose key is minimal
    /// (matches `sort_by(k).first()`); `max=true` returns the *latest* item
    /// whose key is maximal (matches `sort_by(k).last()`).
    ArgExtreme { key: Arc<Program>, lam_param: Option<Arc<str>>, max: bool },
    /// Fused `map(f).flatten()` — single-pass concat of mapped arrays.
    MapFlatten(Arc<Program>),
    /// Fused `map(f).first()` — apply `f` only to the first element.
    /// Empty input → Null (matches plain `first()` on `[]`).
    MapFirst(Arc<Program>),
    /// Fused `map(f).last()` — apply `f` only to the last element.
    MapLast(Arc<Program>),
    /// Fused `map(f).min()` — single-pass numeric min over mapped values.
    MapMin(Arc<Program>),
    /// Fused `map(f).max()` — single-pass numeric max over mapped values.
    MapMax(Arc<Program>),

    // ── Field-specialised fusions (Tier 3) ────────────────────────────────────
    /// `map(k).sum()` where `k` is a single field ident. Skips sub-program exec.
    MapFieldSum(Arc<str>),
    /// `map(k).avg()` where `k` is a single field ident.
    MapFieldAvg(Arc<str>),
    /// `map(k).min()` where `k` is a single field ident.
    MapFieldMin(Arc<str>),
    /// `map(k).max()` where `k` is a single field ident.
    MapFieldMax(Arc<str>),
    /// `map(k)` where `k` is a single field ident — emit array of field values.
    MapField(Arc<str>),
    /// `map(a.b.c)` on arr-of-obj → walk chain per item, push resulting
    /// Val (Null if any step hits a non-Obj or missing key).
    MapFieldChain(Arc<[Arc<str>]>),
    /// `map(k).unique()` where `k` is a single field ident. FxHashSet dedup.
    MapFieldUnique(Arc<str>),
    /// `map(a.b.c).unique()` — walk chain + inline dedup, no intermediate array.
    MapFieldChainUnique(Arc<[Arc<str>]>),

    // ── Flatten-chain fusion (Tier 1) ─────────────────────────────────────────
    /// `.map(k1).flatten().map(k2).flatten()…` collapsed into a single walk.
    /// Input is an array of objects; each step descends the named array-valued
    /// field and concatenates. `N` levels → `N+1` buffers (current+next) instead
    /// of `2N` allocations.
    FlatMapChain(Arc<[Arc<str>]>),

    // ── Predicate specialisation (Tier 4) ─────────────────────────────────────
    /// `filter(k == lit)` — predicate is equality of a single field to a literal.
    FilterFieldEqLit(Arc<str>, Val),
    /// `filter(k <op> lit)` — predicate is a comparison of a single field to a literal.
    FilterFieldCmpLit(Arc<str>, super::ast::BinOp, Val),
    /// `filter(k1 <op> k2)` — predicate compares two fields of the same item.
    FilterFieldCmpField(Arc<str>, super::ast::BinOp, Arc<str>),
    /// `filter(kp == lit).map(kproj)` fused — single pass, no intermediate array.
    FilterFieldEqLitMapField(Arc<str>, Val, Arc<str>),
    /// `filter(kp <cop> lit).map(kproj)` fused — single pass, no intermediate array.
    FilterFieldCmpLitMapField(Arc<str>, super::ast::BinOp, Val, Arc<str>),
    /// `filter(f1 == l1 AND f2 == l2 AND …).count()` fused — zero alloc,
    /// one IC slot per conjunct field.
    FilterFieldsAllEqLitCount(Arc<[(Arc<str>, Val)]>),
    /// `filter(f1 <o1> l1 AND f2 <o2> l2 AND …).count()` fused — general cmp.
    FilterFieldsAllCmpLitCount(Arc<[(Arc<str>, super::ast::BinOp, Val)]>),
    /// `filter(k == lit).count()` — count without materialising.
    FilterFieldEqLitCount(Arc<str>, Val),
    /// `filter(k <op> lit).count()` — count cmp without materialising.
    FilterFieldCmpLitCount(Arc<str>, super::ast::BinOp, Val),
    /// `filter(k1 <op> k2).count()` — cross-field count.
    FilterFieldCmpFieldCount(Arc<str>, super::ast::BinOp, Arc<str>),
    /// `filter(@ <op> lit)` — predicate on the current element itself.
    /// Columnar fast path: IntVec/FloatVec receivers loop on the raw
    /// slice and emit a typed vec; Arr falls back to element iteration.
    FilterCurrentCmpLit(super::ast::BinOp, Val),
    /// `filter(@.starts_with(lit))` — columnar prefix compare on StrVec.
    FilterStrVecStartsWith(Arc<str>),
    /// `filter(@.ends_with(lit))` — columnar suffix compare on StrVec.
    FilterStrVecEndsWith(Arc<str>),
    /// `filter(@.contains(lit))` — SIMD substring (memchr::memmem) on StrVec.
    FilterStrVecContains(Arc<str>),
    /// `map(@.upper())` — ASCII-fast in-lane StrVec→StrVec.
    MapStrVecUpper,
    /// `map(@.lower())` — ASCII-fast in-lane StrVec→StrVec.
    MapStrVecLower,
    /// `map(@.trim())` — in-lane StrVec→StrVec.
    MapStrVecTrim,
    /// `map(@ <op> lit)` / `map(lit <op> @)` — columnar arith over
    /// IntVec / FloatVec receivers.  `flipped=true` means the literal is
    /// on the LHS (matters for Sub/Div).  Output lane:
    ///   IntVec   × Int   × {Add,Sub,Mul,Mod} → IntVec
    ///   IntVec   × Int   × Div               → FloatVec
    ///   IntVec   × Float × *                 → FloatVec
    ///   FloatVec × Int/Float × *             → FloatVec
    MapNumVecArith { op: super::ast::BinOp, lit: Val, flipped: bool },
    /// `map(-@)` — unary negation per element, preserves lane.
    MapNumVecNeg,

    // ── group_by specialisation (Tier 2) ──────────────────────────────────────
    /// `group_by(k)` where `k` is a single field ident. Uses FxHashMap with
    /// primitive-key fast path.
    GroupByField(Arc<str>),
    /// `.count_by(k)` with trivial field key — per-row `obj.get(k)` instead
    /// of lambda dispatch; builds Val::Obj<Arc<str>, Int>.
    CountByField(Arc<str>),
    /// `.unique_by(k)` with trivial field key — per-row `obj.get(k)` direct.
    UniqueByField(Arc<str>),
    /// Fused `filter(p).take_while(q)` — scan while both predicates hold.
    FilterTakeWhile { pred: Arc<Program>, stop: Arc<Program> },
    /// Fused `filter(p).drop_while(q)` — skip leading matches of q on
    /// p-filtered elements.
    FilterDropWhile { pred: Arc<Program>, drop: Arc<Program> },
    /// Fused `map(f).unique()` — apply f and dedup by resulting value.
    MapUnique(Arc<Program>),
    /// Fused equi-join: TOS is lhs array; `rhs` program evaluates
    /// to rhs array; join by (lhs_key, rhs_key) string field names.
    /// Produces array of merged objects (rhs wins on collision).
    EquiJoin { rhs: Arc<Program>, lhs_key: Arc<str>, rhs_key: Arc<str> },

    // ── Ident lookup (var, then current field) ────────────────────────────────
    LoadIdent(Arc<str>),

    // ── Binary / unary ops ────────────────────────────────────────────────────
    Add, Sub, Mul, Div, Mod,
    Eq, Neq, Lt, Lte, Gt, Gte, Fuzzy, Not, Neg,

    // ── Type cast (`as T`) ───────────────────────────────────────────────────
    CastOp(super::ast::CastType),

    // ── Short-circuit ops (embed rhs as sub-program) ──────────────────────────
    AndOp(Arc<Program>),
    OrOp(Arc<Program>),
    CoalesceOp(Arc<Program>),

    // ── Method calls ─────────────────────────────────────────────────────────
    CallMethod(Arc<CompiledCall>),
    CallOptMethod(Arc<CompiledCall>),

    // ── Construction ─────────────────────────────────────────────────────────
    MakeObj(Arc<[CompiledObjEntry]>),
    MakeArr(Arc<[Arc<Program>]>),

    // ── F-string ─────────────────────────────────────────────────────────────
    FString(Arc<[CompiledFSPart]>),

    // ── Kind check ───────────────────────────────────────────────────────────
    KindCheck { ty: KindType, negate: bool },

    // ── Pipeline helpers ──────────────────────────────────────────────────────
    /// Pop TOS → env.current, then push it back (pass-through with context update).
    SetCurrent,
    /// TOS → env var by name, TOS remains (for `->` bind).
    BindVar(Arc<str>),
    /// Pop TOS → env var (for `let` init).
    StoreVar(Arc<str>),
    /// Object destructure bind: TOS obj → multiple vars.
    BindObjDestructure(Arc<BindObjSpec>),
    /// Array destructure bind: TOS arr → multiple vars.
    BindArrDestructure(Arc<[Arc<str>]>),

    // ── Complex (recursive sub-programs) ─────────────────────────────────────
    LetExpr { name: Arc<str>, body: Arc<Program> },
    /// Python-style ternary: TOS is cond; branch into `then_` or `else_`.
    /// Short-circuits — only the taken branch is executed.
    IfElse { then_: Arc<Program>, else_: Arc<Program> },
    ListComp(Arc<CompSpec>),
    DictComp(Arc<DictCompSpec>),
    SetComp(Arc<CompSpec>),

    // ── Resolution cache fast-path ────────────────────────────────────────────
    GetPointer(Arc<str>),

    // ── Patch block (delegates to tree-walker eval) ──────────────────────────
    PatchEval(Arc<super::ast::Expr>),
}

// ── Program ───────────────────────────────────────────────────────────────────

/// A compiled, immutable v2 program.  Cheap to clone (`Arc` internals).
#[derive(Debug, Clone)]
pub struct Program {
    pub ops:          Arc<[Opcode]>,
    pub source:       Arc<str>,
    pub id:           u64,
    /// True when the program contains only structural navigation opcodes
    /// (eligible for resolution caching).
    pub is_structural: bool,
    /// Inline caches — one `AtomicU64` slot per opcode.  Populated by
    /// `Opcode::GetField` / `Opcode::OptField` / `Opcode::FieldChain`.
    ///
    /// Encoding: `stored_slot = slot_idx + 1` (0 reserved for "unset").
    /// No Arc-ptr gating — the hit path is `get_index(slot)` + byte-eq
    /// key verify.  That lets a single slot survive across different
    /// `Arc<IndexMap>` instances of the same shape, which is the common
    /// case for repeated queries over distinct docs and for shape-uniform
    /// array iteration reaching the opcode inside a sub-program.
    pub ics:          Arc<[AtomicU64]>,
}

impl Program {
    fn new(ops: Vec<Opcode>, source: &str) -> Self {
        let id = hash_str(source);
        let is_structural = ops.iter().all(|op| matches!(op,
            Opcode::PushRoot | Opcode::PushCurrent |
            Opcode::GetField(_) | Opcode::GetIndex(_) |
            Opcode::GetSlice(..) | Opcode::OptField(_) |
            Opcode::RootChain(_) | Opcode::FieldChain(_) |
            Opcode::GetPointer(_)
        ));
        let ics = fresh_ics(ops.len());
        Self {
            ops: ops.into(),
            source: source.into(),
            id,
            is_structural,
            ics,
        }
    }
}

/// Per-step inline caches for `Opcode::FieldChain`.  One `AtomicU64` slot per
/// key in the chain — same encoding as `Program.ics` (`slot_idx + 1`, 0 unset).
/// Lives inside the opcode rather than the top-level side-table because the
/// chain length is known only at compile time of that specific opcode.
#[derive(Debug)]
pub struct FieldChainData {
    pub keys: Arc<[Arc<str>]>,
    pub ics:  Box<[AtomicU64]>,
}

impl FieldChainData {
    pub fn new(keys: Arc<[Arc<str>]>) -> Self {
        let n = keys.len();
        let mut ics = Vec::with_capacity(n);
        for _ in 0..n { ics.push(AtomicU64::new(0)); }
        Self { keys, ics: ics.into_boxed_slice() }
    }
    #[inline] pub fn len(&self) -> usize { self.keys.len() }
    #[inline] pub fn is_empty(&self) -> bool { self.keys.is_empty() }
}

impl std::ops::Deref for FieldChainData {
    type Target = [Arc<str>];
    #[inline] fn deref(&self) -> &[Arc<str>] { &self.keys }
}

/// Build a fresh IC side-table with one zeroed `AtomicU64` per opcode.
/// Kept public so other modules that fabricate `Program` values (schema
/// specialisation, analysis passes) can populate the field.
pub fn fresh_ics(len: usize) -> Arc<[AtomicU64]> {
    let mut v = Vec::with_capacity(len);
    for _ in 0..len { v.push(AtomicU64::new(0)); }
    v.into()
}

/// Look up `key` in `m`, using the IC slot as a speculative hint.
///
/// IC is ptr-independent: slot survives across different `Arc<IndexMap>`
/// instances as long as shape (key ordering) is the same.  Hit path is
/// `get_index(slot) + byte-eq key verify`; miss path is one `get_full`
/// that also refreshes the slot.  Slot is encoded as `idx + 1` so zero
/// stays reserved for "unset".
#[inline]
fn ic_get_field(m: &Arc<IndexMap<Arc<str>, Val>>, key: &str, ic: &AtomicU64) -> Val {
    let cached = ic.load(Ordering::Relaxed);
    if cached != 0 {
        let slot = (cached - 1) as usize;
        if let Some((k, v)) = m.get_index(slot) {
            if k.as_ref() == key { return v.clone(); }
        }
    }
    if let Some((idx, _, v)) = m.get_full(key) {
        ic.store((idx as u64) + 1, Ordering::Relaxed);
        v.clone()
    } else {
        Val::Null
    }
}

/// Recognise `.map(k)` sub-programs that reduce to a single field access
/// from the current item: `[PushCurrent, GetField(k)]` or bare `[GetField(k)]`.
/// Lets MapSum/Min/Max/Avg skip the per-item `exec` dispatch.
#[inline]
fn trivial_push_str(ops: &[Opcode]) -> Option<Arc<str>> {
    match ops {
        [Opcode::PushStr(s)] => Some(s.clone()),
        _ => None,
    }
}

/// Allocate an `Arc<str>` of exactly `bytes.len()` and write ASCII-folded
/// contents directly into the Arc payload — one allocation, no intermediate
/// `String`.
///
/// # Safety invariants
/// - Caller must ensure `bytes` is pure ASCII (all bytes < 128).
///   ASCII case-fold preserves ASCII, which is valid UTF-8.
/// - `Arc::get_mut(&mut arc).unwrap()` succeeds because `arc` was just
///   returned by `new_uninit_slice`, so no other strong/weak refs exist.
/// - All `bytes.len()` bytes are initialised before `assume_init`.
/// - `Arc::from_raw(... as *const str)` layout-reinterprets the `Arc<[u8]>`
///   as `Arc<str>`: both share `ArcInner<[u8]>` layout (fat pointer =
///   data ptr + length), and the payload is valid UTF-8 by invariant 1.
#[inline]
fn ascii_fold_to_arc_str(bytes: &[u8], upper: bool) -> Arc<str> {
    debug_assert!(bytes.is_ascii(), "ascii_fold_to_arc_str: non-ASCII input");
    let mut arc = Arc::<[u8]>::new_uninit_slice(bytes.len());
    let slot = Arc::get_mut(&mut arc).unwrap();
    // SAFETY: see invariants above. `dst` points to `bytes.len()` uninit
    // bytes owned exclusively by this Arc; writes stay in bounds.
    unsafe {
        let dst = slot.as_mut_ptr() as *mut u8;
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
        if upper {
            for i in 0..bytes.len() { *dst.add(i) = (*dst.add(i)).to_ascii_uppercase(); }
        } else {
            for i in 0..bytes.len() { *dst.add(i) = (*dst.add(i)).to_ascii_lowercase(); }
        }
    }
    // SAFETY: all bytes initialised by the loop above.
    let arc_bytes: Arc<[u8]> = unsafe { arc.assume_init() };
    // SAFETY: `Arc<[u8]>` and `Arc<str>` share layout (fat pointer over
    // `ArcInner<T>`). Payload is valid UTF-8: ASCII in + ASCII-preserving
    // transform = ASCII out.
    unsafe { Arc::from_raw(Arc::into_raw(arc_bytes) as *const str) }
}

fn trivial_field(ops: &[Opcode]) -> Option<Arc<str>> {
    match ops {
        [Opcode::PushCurrent, Opcode::GetField(k)] => Some(k.clone()),
        [Opcode::GetField(k)] => Some(k.clone()),
        // Bare idents in lambda bodies compile to `LoadIdent(k)` which does
        // var-lookup-then-field fallback; in sub-progs of map/filter/group_by
        // the var slot is almost never shadowed by a field-name, so treat as
        // a field read.
        [Opcode::LoadIdent(k)] => Some(k.clone()),
        _ => None,
    }
}

/// Recognise `.map(a.b.c)` sub-programs that reduce to a walk of a
/// nested field chain from the current item.  Patterns accepted:
///   `[PushCurrent, GetField(k1), GetField(k2), …]`
///   `[PushCurrent, FieldChain([k1, k2, …])]`
///   `[GetField(k1), GetField(k2), …]`
///   `[FieldChain([k1, k2, …])]`
///   `[LoadIdent(k1), GetField(k2), …]`
///   `[LoadIdent(k1), FieldChain([k2, k3, …])]`
/// Returns `None` for single-field patterns (those go via `trivial_field`).
#[inline]
fn trivial_field_chain(ops: &[Opcode]) -> Option<Arc<[Arc<str>]>> {
    let mut out: Vec<Arc<str>> = Vec::new();
    let mut slice = ops;
    // Optional leading PushCurrent is absorbed silently.
    if let [Opcode::PushCurrent, rest @ ..] = slice { slice = rest; }
    // First step: LoadIdent, GetField, or FieldChain.
    match slice {
        [Opcode::LoadIdent(k), rest @ ..] => { out.push(k.clone()); slice = rest; }
        [Opcode::GetField(k), rest @ ..]  => { out.push(k.clone()); slice = rest; }
        [Opcode::FieldChain(ks), rest @ ..] => {
            for k in ks.iter() { out.push(k.clone()); }
            slice = rest;
        }
        _ => return None,
    }
    // Remaining steps: any mix of GetField / FieldChain.
    while !slice.is_empty() {
        match slice {
            [Opcode::GetField(k), rest @ ..]  => { out.push(k.clone()); slice = rest; }
            [Opcode::FieldChain(ks), rest @ ..] => {
                for k in ks.iter() { out.push(k.clone()); }
                slice = rest;
            }
            _ => return None,
        }
    }
    if out.len() < 2 { None } else { Some(Arc::from(out)) }
}

/// A literal primitive suitable for filter-predicate fusion.
#[inline]
fn trivial_literal(op: &Opcode) -> Option<Val> {
    match op {
        Opcode::PushNull => Some(Val::Null),
        Opcode::PushBool(b) => Some(Val::Bool(*b)),
        Opcode::PushInt(n) => Some(Val::Int(*n)),
        Opcode::PushFloat(f) => Some(Val::Float(*f)),
        Opcode::PushStr(s) => Some(Val::Str(s.clone())),
        _ => None,
    }
}

/// Detect `@ <op> lit` or `lit <op> @` — a filter predicate comparing
/// the current element directly to a literal.  Used to lower filter
/// on columnar IntVec/FloatVec receivers to a tight slice loop.
fn detect_current_cmp_lit(ops: &[Opcode]) -> Option<(super::ast::BinOp, Val)> {
    // Form: [PushCurrent, <lit>, <cmp>]
    if let [Opcode::PushCurrent, a, b] = ops {
        if let (Some(lit), Some(op)) = (trivial_literal(a), cmp_opcode(b)) {
            return Some((op, lit));
        }
    }
    // Form: [<lit>, PushCurrent, <cmp>]  →  flip cmp
    if let [a, Opcode::PushCurrent, b] = ops {
        if let (Some(lit), Some(op)) = (trivial_literal(a), cmp_opcode(b)) {
            return Some((flip_cmp(op), lit));
        }
    }
    None
}

/// Which StrVec string predicate is recognised at a filter site.
#[derive(Debug, Clone, Copy)]
enum StrVecPred { StartsWith, EndsWith, Contains }

/// Detect `@.starts_with(lit)` / `@.ends_with(lit)` / `@.contains(lit)` /
/// `@.includes(lit)` filter bodies.  Returns which predicate kind and the
/// literal needle as `Arc<str>`.
fn detect_current_str_method(ops: &[Opcode]) -> Option<(StrVecPred, Arc<str>)> {
    // Form: [PushCurrent, CallMethod{<str method>, sub_progs:[[PushStr(lit)]]}]
    if let [Opcode::PushCurrent, Opcode::CallMethod(b)] = ops {
        if b.sub_progs.len() != 1 { return None; }
        let sub = &b.sub_progs[0];
        if sub.ops.len() != 1 { return None; }
        let lit = match &sub.ops[0] {
            Opcode::PushStr(s) => s.clone(),
            _ => return None,
        };
        let kind = match b.method {
            BuiltinMethod::StartsWith => StrVecPred::StartsWith,
            BuiltinMethod::EndsWith   => StrVecPred::EndsWith,
            // `contains` aliases to `includes` at parse time.
            BuiltinMethod::Includes   => StrVecPred::Contains,
            _ => return None,
        };
        return Some((kind, lit));
    }
    None
}

/// Detect `@.upper()` / `@.lower()` / `@.trim()` map bodies → in-lane StrVec op.
#[derive(Debug, Clone, Copy)]
enum StrVecMap { Upper, Lower, Trim }

/// Detect `@ <op> lit` / `lit <op> @` arith map bodies.
/// `op` is one of Add/Sub/Mul/Div/Mod.  `flipped=true` → literal on LHS.
fn detect_current_arith_lit(ops: &[Opcode]) -> Option<(super::ast::BinOp, Val, bool)> {
    use super::ast::BinOp::*;
    let arith_op = |o: &Opcode| -> Option<super::ast::BinOp> {
        Some(match o {
            Opcode::Add => Add, Opcode::Sub => Sub,
            Opcode::Mul => Mul, Opcode::Div => Div,
            Opcode::Mod => Mod,
            _ => return None,
        })
    };
    // Form: [PushCurrent, <lit>, <arith>]
    if let [Opcode::PushCurrent, a, b] = ops {
        if let (Some(lit), Some(op)) = (trivial_literal(a), arith_op(b)) {
            if matches!(lit, Val::Int(_) | Val::Float(_)) {
                return Some((op, lit, false));
            }
        }
    }
    // Form: [<lit>, PushCurrent, <arith>]
    if let [a, Opcode::PushCurrent, b] = ops {
        if let (Some(lit), Some(op)) = (trivial_literal(a), arith_op(b)) {
            if matches!(lit, Val::Int(_) | Val::Float(_)) {
                return Some((op, lit, true));
            }
        }
    }
    None
}

/// Detect `[-@]` — unary negation of the current element.
fn detect_current_neg(ops: &[Opcode]) -> bool {
    matches!(ops, [Opcode::PushCurrent, Opcode::Neg])
}

fn detect_current_str_nullary(ops: &[Opcode]) -> Option<StrVecMap> {
    if let [Opcode::PushCurrent, Opcode::CallMethod(b)] = ops {
        if !b.sub_progs.is_empty() { return None; }
        return Some(match b.method {
            BuiltinMethod::Upper => StrVecMap::Upper,
            BuiltinMethod::Lower => StrVecMap::Lower,
            BuiltinMethod::Trim  => StrVecMap::Trim,
            _ => return None,
        });
    }
    None
}

/// Detect a filter-predicate sub-program of shape `field <op> literal`,
/// `literal <op> field`, or `field1 <op> field2`. Returns one of three variants
/// so the caller can pick the right fused opcode.
#[derive(Debug)]
enum FieldPred {
    FieldCmpLit(Arc<str>, super::ast::BinOp, Val),
    FieldCmpField(Arc<str>, super::ast::BinOp, Arc<str>),
}

fn flip_cmp(op: super::ast::BinOp) -> super::ast::BinOp {
    use super::ast::BinOp::*;
    match op {
        Lt => Gt, Gt => Lt, Lte => Gte, Gte => Lte,
        other => other,
    }
}

fn cmp_opcode(op: &Opcode) -> Option<super::ast::BinOp> {
    use super::ast::BinOp::*;
    Some(match op {
        Opcode::Eq => Eq, Opcode::Neq => Neq,
        Opcode::Lt => Lt, Opcode::Lte => Lte,
        Opcode::Gt => Gt, Opcode::Gte => Gte,
        _ => return None,
    })
}

/// Patterns recognised for a filter-lambda body:
///   `[PushCurrent, GetField(k), PushLit, <cmp>]`
///   `[PushLit, PushCurrent, GetField(k), <cmp>]`
///   `[PushCurrent, GetField(k1), PushCurrent, GetField(k2), <cmp>]`
fn detect_field_pred(ops: &[Opcode]) -> Option<FieldPred> {
    // Helper: match a single-op "field read" — PushCurrent+GetField, GetField
    // alone, or LoadIdent (var fallback to field).
    #[inline]
    fn field_read_prefix(ops: &[Opcode]) -> Option<(Arc<str>, usize)> {
        match ops.first()? {
            Opcode::LoadIdent(k) => Some((k.clone(), 1)),
            Opcode::GetField(k) => Some((k.clone(), 1)),
            Opcode::PushCurrent => {
                if let Some(Opcode::GetField(k)) = ops.get(1) {
                    Some((k.clone(), 2))
                } else { None }
            }
            _ => None,
        }
    }
    // Form 1: field <op> literal
    if let Some((k, n)) = field_read_prefix(ops) {
        if let (Some(lit_op), Some(cmp_op)) = (ops.get(n), ops.get(n + 1)) {
            if ops.len() == n + 2 {
                if let (Some(lit), Some(op)) = (trivial_literal(lit_op), cmp_opcode(cmp_op)) {
                    return Some(FieldPred::FieldCmpLit(k, op, lit));
                }
            }
        }
        // Form 3: field1 <op> field2
        if let Some((k2, n2_extra)) = ops.get(n).and_then(|_| {
            let tail = &ops[n..];
            field_read_prefix(tail).map(|(kk, nn)| (kk, nn))
        }) {
            if let Some(cmp_op) = ops.get(n + n2_extra) {
                if ops.len() == n + n2_extra + 1 {
                    if let Some(op) = cmp_opcode(cmp_op) {
                        return Some(FieldPred::FieldCmpField(k, op, k2));
                    }
                }
            }
        }
    }
    // Form 2: literal <op> field (flip)
    if let Some(lit) = ops.first().and_then(trivial_literal) {
        if let Some((k, n2)) = field_read_prefix(&ops[1..]) {
            if let Some(cmp_op) = ops.get(1 + n2) {
                if ops.len() == 1 + n2 + 1 {
                    if let Some(op) = cmp_opcode(cmp_op) {
                        return Some(FieldPred::FieldCmpLit(k, flip_cmp(op), lit));
                    }
                }
            }
        }
    }
    None
}

/// Detect a predicate body that is a conjunction (AND-chain) of
/// `field <cmp> lit` comparisons.  Returns the flat list of
/// `(field, cmp_op, lit)` triples when the entire pred reduces to
/// `f1 <o1> l1 AND f2 <o2> l2 AND ...`.
///
/// Pattern accepted (N ≥ 2):
///   `[⟨field cmp lit⟩, AndOp(⟨field cmp lit⟩), AndOp(⟨field cmp lit⟩), …]`
fn detect_field_cmp_conjuncts(ops: &[Opcode]) -> Option<Vec<(Arc<str>, super::ast::BinOp, Val)>> {
    let mut triples: Vec<(Arc<str>, super::ast::BinOp, Val)> = Vec::new();
    let first_and = ops.iter().position(|o| matches!(o, Opcode::AndOp(_)));
    let first_len = first_and.unwrap_or(ops.len());
    match detect_field_pred(&ops[..first_len])? {
        FieldPred::FieldCmpLit(k, op, lit) => triples.push((k, op, lit)),
        _ => return None,
    }
    for op in &ops[first_len..] {
        if let Opcode::AndOp(sub) = op {
            match detect_field_pred(&sub.ops)? {
                FieldPred::FieldCmpLit(k, op, lit) => triples.push((k, op, lit)),
                _ => return None,
            }
        } else {
            return None;
        }
    }
    if triples.len() >= 2 { Some(triples) } else { None }
}

/// Convenience: all conjuncts are Eq → produce the flat (field, lit) form
/// used by `FilterFieldsAllEqLitCount`.
fn detect_field_eq_conjuncts(ops: &[Opcode]) -> Option<Vec<(Arc<str>, Val)>> {
    let triples = detect_field_cmp_conjuncts(ops)?;
    triples.into_iter()
        .map(|(k, op, v)| if matches!(op, super::ast::BinOp::Eq) { Some((k, v)) } else { None })
        .collect()
}

/// Compare two `Val`s using a binary comparison operator. Only implements the
/// semantics needed for filter predicate fusion; falls back to cmp_vals for
/// ordering comparisons.
#[inline]
fn cmp_val_binop(a: &Val, op: super::ast::BinOp, b: &Val) -> bool {
    use super::ast::BinOp::*;
    use std::cmp::Ordering;
    match op {
        Eq => crate::eval::util::vals_eq(a, b),
        Neq => !crate::eval::util::vals_eq(a, b),
        Lt | Lte | Gt | Gte => {
            let ord = crate::eval::util::cmp_vals(a, b);
            match op {
                Lt  => ord == Ordering::Less,
                Lte => ord != Ordering::Greater,
                Gt  => ord == Ordering::Greater,
                Gte => ord != Ordering::Less,
                _ => unreachable!(),
            }
        }
        _ => false,
    }
}

/// `group_by(k)` where `k` is a bare field ident. Builds an object whose keys
/// are the distinct field values stringified, mapping to arrays of items.
/// Preserves first-seen key order.
/// Resolve char-index based slice bounds into byte offsets.
/// Returns `(start_byte, end_byte)` within `src`.
fn slice_unicode_bounds(src: &str, start: i64, end: Option<i64>) -> (usize, usize) {
    let total_chars = src.chars().count() as i64;
    let start_u = if start < 0 {
        total_chars.saturating_sub(-start).max(0) as usize
    } else { start as usize };
    let end_u = match end {
        Some(e) if e < 0 => total_chars.saturating_sub(-e).max(0) as usize,
        Some(e) => e as usize,
        None    => total_chars as usize,
    };
    let mut start_b = src.len();
    let mut end_b = src.len();
    let mut found_start = false;
    for (ci, (bi, _)) in src.char_indices().enumerate() {
        if !found_start && ci == start_u {
            start_b = bi;
            found_start = true;
        }
        if ci == end_u {
            end_b = bi;
            return (start_b.min(end_b), end_b);
        }
    }
    if !found_start { start_b = src.len(); }
    (start_b, end_b)
}

fn count_by_field(recv: &Val, k: &str) -> Val {
    let a = match recv {
        Val::Arr(a) => a,
        _ => return Val::obj(indexmap::IndexMap::new()),
    };
    let mut out: indexmap::IndexMap<Arc<str>, i64> = indexmap::IndexMap::with_capacity(16);
    let mut cached: Option<usize> = None;
    for item in a.iter() {
        let key: Arc<str> = if let Val::Obj(m) = item {
            let v = lookup_field_by_str_cached(m, k, &mut cached);
            match v {
                Some(Val::Str(s)) => s.clone(),
                Some(Val::StrSlice(r)) => r.to_arc(),
                Some(Val::Int(n)) => Arc::from(n.to_string()),
                Some(Val::Float(x)) => Arc::from(x.to_string()),
                Some(Val::Bool(b)) => Arc::from(if *b { "true" } else { "false" }),
                Some(Val::Null) | None => Arc::from("null"),
                Some(other) => Arc::from(format!("{:?}", other)),
            }
        } else if let Val::ObjSmall(ps) = item {
            let mut found: Option<&Val> = None;
            for (kk, vv) in ps.iter() {
                if kk.as_ref() == k { found = Some(vv); break; }
            }
            match found {
                Some(Val::Str(s)) => s.clone(),
                Some(Val::StrSlice(r)) => r.to_arc(),
                Some(Val::Int(n)) => Arc::from(n.to_string()),
                Some(Val::Float(x)) => Arc::from(x.to_string()),
                Some(Val::Bool(b)) => Arc::from(if *b { "true" } else { "false" }),
                Some(Val::Null) | None => Arc::from("null"),
                Some(other) => Arc::from(format!("{:?}", other)),
            }
        } else {
            Arc::from("null")
        };
        *out.entry(key).or_insert(0) += 1;
    }
    let finalised: indexmap::IndexMap<Arc<str>, Val> = out.into_iter()
        .map(|(k, n)| (k, Val::Int(n)))
        .collect();
    Val::obj(finalised)
}

fn unique_by_field(recv: &Val, k: &str) -> Val {
    let a = match recv {
        Val::Arr(a) => a,
        _ => return Val::arr(Vec::new()),
    };
    let mut seen: indexmap::IndexSet<Arc<str>> = indexmap::IndexSet::with_capacity(a.len());
    let mut out: Vec<Val> = Vec::with_capacity(a.len());
    let mut cached: Option<usize> = None;
    for item in a.iter() {
        let key: Arc<str> = if let Val::Obj(m) = item {
            let v = lookup_field_by_str_cached(m, k, &mut cached);
            match v {
                Some(Val::Str(s)) => s.clone(),
                Some(Val::StrSlice(r)) => r.to_arc(),
                Some(Val::Int(n)) => Arc::from(n.to_string()),
                Some(Val::Float(x)) => Arc::from(x.to_string()),
                Some(Val::Bool(b)) => Arc::from(if *b { "true" } else { "false" }),
                Some(Val::Null) | None => Arc::from("null"),
                Some(other) => Arc::from(format!("{:?}", other)),
            }
        } else {
            Arc::from("null")
        };
        if seen.insert(key) { out.push(item.clone()); }
    }
    Val::arr(out)
}

fn group_by_field(recv: &Val, k: &str) -> Val {
    let a = match recv {
        Val::Arr(a) => a,
        _ => return Val::obj(indexmap::IndexMap::new()),
    };
    let mut out: indexmap::IndexMap<Arc<str>, Vec<Val>> = indexmap::IndexMap::with_capacity(16);
    let mut cached: Option<usize> = None;
    for item in a.iter() {
        let key = if let Val::Obj(m) = item {
            let v = lookup_field_by_str_cached(m, k, &mut cached);
            match v {
                Some(Val::Str(s)) => s.clone(),
                Some(Val::Int(n)) => Arc::from(n.to_string()),
                Some(Val::Float(x)) => Arc::from(x.to_string()),
                Some(Val::Bool(b)) => Arc::from(if *b { "true" } else { "false" }),
                Some(Val::Null) | None => Arc::from("null"),
                Some(other) => Arc::from(format!("{:?}", other)),
            }
        } else {
            Arc::from("null")
        };
        out.entry(key).or_insert_with(|| Vec::with_capacity(4)).push(item.clone());
    }
    let finalised: indexmap::IndexMap<Arc<str>, Val> = out.into_iter()
        .map(|(k, v)| (k, Val::arr(v)))
        .collect();
    Val::obj(finalised)
}

/// Shape-index cache: cheap inline-cache for repeated `m.get(k)` on arrays of
/// same-shape objects. First call stores `get_index_of(k)`; subsequent calls
/// try `get_index(i)` and verify key identity (Arc<str> pointer or bytes).
/// Fallback to `m.get(k)` on miss.
#[inline]
fn lookup_field_cached<'a>(
    m: &'a indexmap::IndexMap<Arc<str>, Val>,
    k: &Arc<str>,
    cached: &mut Option<usize>,
) -> Option<&'a Val> {
    if let Some(i) = *cached {
        if let Some((ki, vi)) = m.get_index(i) {
            if Arc::ptr_eq(ki, k) || ki.as_ref() == k.as_ref() {
                return Some(vi);
            }
        }
    }
    match m.get_full(k.as_ref()) {
        Some((i, _, v)) => { *cached = Some(i); Some(v) }
        None => { *cached = None; None }
    }
}

/// Initial capacity hint for filter `out` Vecs.
/// Assumes ~25% selectivity; avoids zero-cap realloc storm while not
/// over-reserving on highly selective predicates.  Capped at receiver len.
#[inline]
fn filter_cap_hint(recv_len: usize) -> usize {
    (recv_len / 4 + 4).min(recv_len)
}

/// Accumulate lambda pattern tag — selects which fused binop to run.
#[derive(Copy, Clone)]
enum AccumOp { Add, Sub, Mul }

// ── Typed-numeric aggregate fast-paths ────────────────────────────────────────
// Direct loops over `&[Val]` for mono-typed or mixed Int/Float arrays.  Used
// by the bare `.sum()/.min()/.max()/.avg()` no-arg method-call fast path in
// `exec_call` — skips registry dispatch, `into_vec()` clone, and the extra
// `.filter().collect()` that `func_aggregates::collect_nums` performs.
//
// Semantics match `func_aggregates`: non-numeric items are skipped; Int-only
// arrays stay on `i64` (no lossy widening); Float appearance widens once.

#[inline]
fn agg_sum_typed(a: &[Val]) -> Val {
    // Tight i64 loop until first Float; then switch to f64 loop.
    let mut i_acc: i64 = 0;
    let mut it = a.iter();
    while let Some(v) = it.next() {
        match v {
            Val::Int(n) => i_acc = i_acc.wrapping_add(*n),
            Val::Float(x) => {
                let mut f_acc = i_acc as f64 + *x;
                for v in it {
                    match v {
                        Val::Int(n)   => f_acc += *n as f64,
                        Val::Float(x) => f_acc += *x,
                        _ => {}
                    }
                }
                return Val::Float(f_acc);
            }
            _ => {} // skip non-numeric
        }
    }
    Val::Int(i_acc)
}

#[inline]
fn agg_avg_typed(a: &[Val]) -> Val {
    let mut sum: f64 = 0.0;
    let mut n: usize = 0;
    for v in a {
        match v {
            Val::Int(x)   => { sum += *x as f64; n += 1; }
            Val::Float(x) => { sum += *x;        n += 1; }
            _ => {}
        }
    }
    if n == 0 { Val::Null } else { Val::Float(sum / n as f64) }
}

#[inline]
fn agg_minmax_typed(a: &[Val], want_max: bool) -> Val {
    let mut it = a.iter();
    // Find first number.
    let first = loop {
        match it.next() {
            Some(v) if v.is_number() => break v,
            Some(_) => continue,
            None    => return Val::Null,
        }
    };
    match first {
        Val::Int(n0) => {
            let mut best: i64 = *n0;
            // Mono-Int tight loop; on first Float, promote.
            while let Some(v) = it.next() {
                match v {
                    Val::Int(n) => {
                        let n = *n;
                        if want_max { if n > best { best = n; } }
                        else        { if n < best { best = n; } }
                    }
                    Val::Float(x) => {
                        let x = *x;
                        let mut best_f = best as f64;
                        if want_max { if x > best_f { best_f = x; } }
                        else        { if x < best_f { best_f = x; } }
                        for v in it {
                            match v {
                                Val::Int(n) => {
                                    let n = *n as f64;
                                    if want_max { if n > best_f { best_f = n; } }
                                    else        { if n < best_f { best_f = n; } }
                                }
                                Val::Float(x) => {
                                    let x = *x;
                                    if want_max { if x > best_f { best_f = x; } }
                                    else        { if x < best_f { best_f = x; } }
                                }
                                _ => {}
                            }
                        }
                        return Val::Float(best_f);
                    }
                    _ => {}
                }
            }
            Val::Int(best)
        }
        Val::Float(x0) => {
            let mut best_f: f64 = *x0;
            for v in it {
                match v {
                    Val::Int(n) => {
                        let n = *n as f64;
                        if want_max { if n > best_f { best_f = n; } }
                        else        { if n < best_f { best_f = n; } }
                    }
                    Val::Float(x) => {
                        let x = *x;
                        if want_max { if x > best_f { best_f = x; } }
                        else        { if x < best_f { best_f = x; } }
                    }
                    _ => {}
                }
            }
            Val::Float(best_f)
        }
        _ => Val::Null,
    }
}

/// `&str`-keyed variant of `lookup_field_cached`; ptr-eq shortcut is skipped
/// (caller doesn't hold an `Arc<str>`), so the hit path is byte-eq only.
#[inline]
fn lookup_field_by_str_cached<'a>(
    m: &'a indexmap::IndexMap<Arc<str>, Val>,
    k: &str,
    cached: &mut Option<usize>,
) -> Option<&'a Val> {
    if let Some(i) = *cached {
        if let Some((ki, vi)) = m.get_index(i) {
            if ki.as_ref() == k {
                return Some(vi);
            }
        }
    }
    match m.get_full(k) {
        Some((i, _, v)) => { *cached = Some(i); Some(v) }
        None => { *cached = None; None }
    }
}

// ── Variable context (compile-time) ──────────────────────────────────────────

#[derive(Clone, Default)]
struct VarCtx {
    known: SmallVec<[Arc<str>; 4]>,
}

impl VarCtx {
    fn with_var(&self, name: &str) -> Self {
        let mut v = self.clone();
        if !v.known.iter().any(|k| k.as_ref() == name) {
            v.known.push(Arc::from(name));
        }
        v
    }
    fn with_vars(&self, names: &[String]) -> Self {
        let mut v = self.clone();
        for n in names {
            if !v.known.iter().any(|k| k.as_ref() == n.as_str()) {
                v.known.push(Arc::from(n.as_str()));
            }
        }
        v
    }
    fn has(&self, name: &str) -> bool {
        self.known.iter().any(|k| k.as_ref() == name)
    }
}

// ── Compiler ─────────────────────────────────────────────────────────────────

pub struct Compiler;

impl Compiler {
    pub fn compile(expr: &Expr, source: &str) -> Program {
        let mut e = expr.clone();
        Self::reorder_and_operands(&mut e);
        let ctx = VarCtx::default();
        let ops = Self::optimize(Self::emit(&e, &ctx));
        let prog = Program::new(ops, source);
        // Post-pass: canonicalise identical sub-programs.
        let deduped = super::analysis::dedup_subprograms(&prog);
        let ics = fresh_ics(deduped.ops.len());
        Program {
            ops:           deduped.ops.clone(),
            source:        prog.source,
            id:            prog.id,
            is_structural: prog.is_structural,
            ics,
        }
    }

    /// AST rewrite: for each `a and b`, if `b` is more selective than `a`,
    /// swap operands so the cheaper/selective predicate runs first.  Safe
    /// because `and` is commutative on pure, side-effect-free expressions.
    fn reorder_and_operands(expr: &mut Expr) {
        use super::analysis::selectivity_score;
        match expr {
            Expr::BinOp(l, op, r) if *op == BinOp::And => {
                Self::reorder_and_operands(l);
                Self::reorder_and_operands(r);
                if selectivity_score(r) < selectivity_score(l) {
                    std::mem::swap(l, r);
                }
            }
            Expr::BinOp(l, _, r) => {
                Self::reorder_and_operands(l);
                Self::reorder_and_operands(r);
            }
            Expr::UnaryNeg(e) | Expr::Not(e) | Expr::Kind { expr: e, .. } =>
                Self::reorder_and_operands(e),
            Expr::Coalesce(l, r) => {
                Self::reorder_and_operands(l);
                Self::reorder_and_operands(r);
            }
            Expr::Chain(base, steps) => {
                Self::reorder_and_operands(base);
                for s in steps {
                    match s {
                        super::ast::Step::DynIndex(e) | super::ast::Step::InlineFilter(e) =>
                            Self::reorder_and_operands(e),
                        super::ast::Step::Method(_, args) | super::ast::Step::OptMethod(_, args) =>
                            for a in args { match a {
                                super::ast::Arg::Pos(e) | super::ast::Arg::Named(_, e) =>
                                    Self::reorder_and_operands(e),
                            } },
                        _ => {}
                    }
                }
            }
            Expr::Let { init, body, .. } => {
                Self::reorder_and_operands(init);
                Self::reorder_and_operands(body);
            }
            Expr::Pipeline { base, steps } => {
                Self::reorder_and_operands(base);
                for s in steps {
                    if let super::ast::PipeStep::Forward(e) = s {
                        Self::reorder_and_operands(e);
                    }
                }
            }
            Expr::Object(fields) => for f in fields { match f {
                super::ast::ObjField::Kv { val, .. } => Self::reorder_and_operands(val),
                super::ast::ObjField::Dynamic { key, val } => {
                    Self::reorder_and_operands(key);
                    Self::reorder_and_operands(val);
                }
                super::ast::ObjField::Spread(e) => Self::reorder_and_operands(e),
                _ => {}
            } },
            Expr::Array(elems) => for e in elems { match e {
                super::ast::ArrayElem::Expr(e) | super::ast::ArrayElem::Spread(e) =>
                    Self::reorder_and_operands(e),
            } },
            Expr::ListComp { expr, iter, cond, .. }
            | Expr::SetComp  { expr, iter, cond, .. }
            | Expr::GenComp  { expr, iter, cond, .. } => {
                Self::reorder_and_operands(expr);
                Self::reorder_and_operands(iter);
                if let Some(c) = cond { Self::reorder_and_operands(c); }
            }
            Expr::DictComp { key, val, iter, cond, .. } => {
                Self::reorder_and_operands(key);
                Self::reorder_and_operands(val);
                Self::reorder_and_operands(iter);
                if let Some(c) = cond { Self::reorder_and_operands(c); }
            }
            Expr::Lambda { body, .. } => Self::reorder_and_operands(body),
            Expr::GlobalCall { args, .. } => for a in args { match a {
                super::ast::Arg::Pos(e) | super::ast::Arg::Named(_, e) =>
                    Self::reorder_and_operands(e),
            } },
            _ => {}
        }
    }

    pub fn compile_str(input: &str) -> Result<Program, EvalError> {
        let expr = super::parser::parse(input)
            .map_err(|e| EvalError(e.to_string()))?;
        Ok(Self::compile(&expr, input))
    }

    /// Compile with explicit pass configuration.  Cached by callers
    /// under `(config.hash(), expr)`.
    pub fn compile_str_with_config(input: &str, config: PassConfig) -> Result<Program, EvalError> {
        let expr = super::parser::parse(input)
            .map_err(|e| EvalError(e.to_string()))?;
        let mut e = expr.clone();
        if config.reorder_and { Self::reorder_and_operands(&mut e); }
        let ctx = VarCtx::default();
        let ops = Self::optimize_with(Self::emit(&e, &ctx), config);
        let prog = Program::new(ops, input);
        if config.dedup_subprogs {
            let deduped = super::analysis::dedup_subprograms(&prog);
            let ics = fresh_ics(deduped.ops.len());
            Ok(Program {
                ops:           deduped.ops.clone(),
                source:        prog.source,
                id:            prog.id,
                is_structural: prog.is_structural,
                ics,
            })
        } else {
            Ok(prog)
        }
    }

    // ── Peephole optimizer ────────────────────────────────────────────────────

    fn optimize(ops: Vec<Opcode>) -> Vec<Opcode> {
        Self::optimize_with(ops, PassConfig::default())
    }

    fn optimize_with(ops: Vec<Opcode>, cfg: PassConfig) -> Vec<Opcode> {
        let ops = if cfg.root_chain      { Self::pass_root_chain(ops) }      else { ops };
        let ops = if cfg.field_chain     { Self::pass_field_chain(ops) }     else { ops };
        let ops = if cfg.filter_count    { Self::pass_filter_count(ops) }    else { ops };
        let ops = if cfg.filter_fusion   { Self::pass_filter_fusion(ops) }   else { ops };
        let ops = if cfg.filter_fusion   { Self::pass_string_chain_fusion(ops) } else { ops };
        let ops = if cfg.find_quantifier { Self::pass_find_quantifier(ops) } else { ops };
        let ops = if cfg.filter_fusion   { Self::pass_field_specialise(ops) } else { ops };
        let ops = Self::pass_list_comp_specialise(ops);
        let ops = if cfg.strength_reduce { Self::pass_strength_reduce(ops) } else { ops };
        let ops = if cfg.redundant_ops   { Self::pass_redundant_ops(ops) }   else { ops };
        let ops = if cfg.kind_check_fold { Self::pass_kind_check_fold(ops) } else { ops };
        let ops = if cfg.method_const    { Self::pass_method_const_fold(ops)} else { ops };
        let ops = if cfg.const_fold      { Self::pass_const_fold(ops) }      else { ops };
        let ops = if cfg.nullness        { Self::pass_nullness_opt_field(ops)} else { ops };
        let ops = if cfg.equi_join       { Self::pass_equi_join_fusion(ops) } else { ops };
        ops
    }

    /// Rewrite `CallMethod(equi_join, [rhs, PushStr(lk), PushStr(rk)])`
    /// to the fused `EquiJoin` opcode — removes runtime method dispatch
    /// and extracts string keys into the opcode so the executor can
    /// hash directly.
    fn pass_equi_join_fusion(ops: Vec<Opcode>) -> Vec<Opcode> {
        let mut out: Vec<Opcode> = Vec::with_capacity(ops.len());
        for op in ops {
            if let Opcode::CallMethod(c) = &op {
                if c.method == BuiltinMethod::EquiJoin && c.sub_progs.len() == 3 {
                    let rhs = Arc::clone(&c.sub_progs[0]);
                    let lhs_key = const_str_program(&c.sub_progs[1]);
                    let rhs_key = const_str_program(&c.sub_progs[2]);
                    if let (Some(lk), Some(rk)) = (lhs_key, rhs_key) {
                        out.push(Opcode::EquiJoin { rhs, lhs_key: lk, rhs_key: rk });
                        continue;
                    }
                }
            }
            out.push(op);
        }
        out
    }

    /// Nullness-driven: when the preceding op provably leaves a non-null
    /// receiver on the stack, rewrite `OptField(k)` → `GetField(k)`.
    /// Conservative: only folds when the predecessor is a construction
    /// opcode (MakeObj / MakeArr / PushStr / RootChain / GetField).
    fn pass_nullness_opt_field(ops: Vec<Opcode>) -> Vec<Opcode> {
        let mut out: Vec<Opcode> = Vec::with_capacity(ops.len());
        for op in ops {
            if let Opcode::OptField(k) = &op {
                // Only safe when receiver is provably a non-null object —
                // MakeObj is the only opcode that guarantees this without
                // a schema.  Other cases are handled by schema::specialize.
                let non_null = matches!(out.last(), Some(Opcode::MakeObj(_)));
                if non_null {
                    out.push(Opcode::GetField(k.clone()));
                    continue;
                }
            }
            out.push(op);
        }
        out
    }

    /// Fold built-in methods when receiver is a literal with known length/content:
    ///   PushStr(s) + .len()    → PushInt(utf8 char count)
    ///   PushStr(s) + .upper()  → PushStr(upper)
    ///   PushStr(s) + .lower()  → PushStr(lower)
    ///   PushStr(s) + .trim()   → PushStr(trim)
    ///   MakeArr(n elems) + .len()  → PushInt(n)  (only for non-spread arrays)
    fn pass_method_const_fold(ops: Vec<Opcode>) -> Vec<Opcode> {
        let mut out: Vec<Opcode> = Vec::with_capacity(ops.len());
        for op in ops {
            if let Opcode::CallMethod(c) = &op {
                if c.sub_progs.is_empty() {
                    match (out.last(), c.method) {
                        (Some(Opcode::PushStr(s)), BuiltinMethod::Len) => {
                            let n = s.chars().count() as i64;
                            out.pop();
                            out.push(Opcode::PushInt(n));
                            continue;
                        }
                        (Some(Opcode::PushStr(s)), BuiltinMethod::Upper) => {
                            let u: Arc<str> = Arc::from(s.to_uppercase());
                            out.pop();
                            out.push(Opcode::PushStr(u));
                            continue;
                        }
                        (Some(Opcode::PushStr(s)), BuiltinMethod::Lower) => {
                            let u: Arc<str> = Arc::from(s.to_lowercase());
                            out.pop();
                            out.push(Opcode::PushStr(u));
                            continue;
                        }
                        (Some(Opcode::PushStr(s)), BuiltinMethod::Trim) => {
                            let u: Arc<str> = Arc::from(s.trim());
                            out.pop();
                            out.push(Opcode::PushStr(u));
                            continue;
                        }
                        (Some(Opcode::MakeArr(progs)), BuiltinMethod::Len) => {
                            let n = progs.len() as i64;
                            out.pop();
                            out.push(Opcode::PushInt(n));
                            continue;
                        }
                        _ => {}
                    }
                }
            }
            out.push(op);
        }
        out
    }

    /// Fold `KindCheck` when its input type is a literal push:
    ///   PushInt(n)  + KindCheck{number, neg} → PushBool(!neg)
    ///   PushStr(_)  + KindCheck{string, neg} → PushBool(!neg)
    ///   PushNull    + KindCheck{null, neg}   → PushBool(!neg)
    ///   PushBool(_) + KindCheck{bool, neg}   → PushBool(!neg)
    ///   mismatches fold to opposite.
    fn pass_kind_check_fold(ops: Vec<Opcode>) -> Vec<Opcode> {
        use super::analysis::{fold_kind_check, VType};
        let mut out = Vec::with_capacity(ops.len());
        for op in ops {
            if let Opcode::KindCheck { ty, negate } = &op {
                let prev_ty: Option<VType> = match out.last() {
                    Some(Opcode::PushNull)     => Some(VType::Null),
                    Some(Opcode::PushBool(_))  => Some(VType::Bool),
                    Some(Opcode::PushInt(_))   => Some(VType::Int),
                    Some(Opcode::PushFloat(_)) => Some(VType::Float),
                    Some(Opcode::PushStr(_))   => Some(VType::Str),
                    Some(Opcode::MakeArr(_))   => Some(VType::Arr),
                    Some(Opcode::MakeObj(_))   => Some(VType::Obj),
                    _ => None,
                };
                if let Some(vt) = prev_ty {
                    if let Some(b) = fold_kind_check(vt, *ty, *negate) {
                        out.pop();
                        out.push(Opcode::PushBool(b));
                        continue;
                    }
                }
            }
            out.push(op);
        }
        out
    }

    /// Fuse adjacent method calls into single-pass fused opcodes:
    ///   filter(p) + map(f)     → FilterMap
    ///   filter(p1) + filter(p2)→ FilterFilter
    ///   map(f1) + map(f2)      → MapMap
    fn pass_filter_fusion(ops: Vec<Opcode>) -> Vec<Opcode> {
        let mut out: Vec<Opcode> = Vec::with_capacity(ops.len());
        for op in ops {
            // FilterMap + sum()/avg()/first() → three-way fusion
            if let Opcode::CallMethod(b) = &op {
                if b.sub_progs.is_empty() {
                    if let Some(Opcode::FilterMap { pred, map }) = out.last() {
                        let pred = Arc::clone(pred);
                        let map = Arc::clone(map);
                        let fused = match b.method {
                            BuiltinMethod::Sum => Some(Opcode::FilterMapSum { pred, map }),
                            BuiltinMethod::Avg => Some(Opcode::FilterMapAvg { pred, map }),
                            BuiltinMethod::First => Some(Opcode::FilterMapFirst { pred, map }),
                            BuiltinMethod::Min => Some(Opcode::FilterMapMin { pred, map }),
                            BuiltinMethod::Max => Some(Opcode::FilterMapMax { pred, map }),
                            _ => None,
                        };
                        if let Some(o) = fused {
                            out.pop();
                            out.push(o);
                            continue;
                        }
                    }
                }
            }
            if let (Opcode::CallMethod(b), Some(Opcode::CallMethod(a))) = (&op, out.last()) {
                // Two-arg fusions (both have sub_progs)
                if a.sub_progs.len() >= 1 && b.sub_progs.len() >= 1 {
                    let (am, bm) = (a.method, b.method);
                    let p1 = Arc::clone(&a.sub_progs[0]);
                    let p2 = Arc::clone(&b.sub_progs[0]);
                    let fused = match (am, bm) {
                        (BuiltinMethod::Filter, BuiltinMethod::Map) =>
                            Some(Opcode::FilterMap { pred: p1, map: p2 }),
                        (BuiltinMethod::Filter, BuiltinMethod::Filter) =>
                            Some(Opcode::FilterFilter { p1, p2 }),
                        (BuiltinMethod::Map, BuiltinMethod::Map) =>
                            Some(Opcode::MapMap { f1: p1, f2: p2 }),
                        (BuiltinMethod::Map, BuiltinMethod::Filter) =>
                            Some(Opcode::MapFilter { map: p1, pred: p2 }),
                        _ => None,
                    };
                    if let Some(f) = fused {
                        out.pop();
                        out.push(f);
                        continue;
                    }
                }
                // map(f) + sum()/avg()/min()/max()/flatten()/first()/last()
                if a.method == BuiltinMethod::Map && a.sub_progs.len() >= 1
                   && b.sub_progs.is_empty() {
                    let f = Arc::clone(&a.sub_progs[0]);
                    let fused = match b.method {
                        BuiltinMethod::Sum => Some(Opcode::MapSum(f)),
                        BuiltinMethod::Avg => Some(Opcode::MapAvg(f)),
                        BuiltinMethod::Min => Some(Opcode::MapMin(f)),
                        BuiltinMethod::Max => Some(Opcode::MapMax(f)),
                        BuiltinMethod::Flatten => Some(Opcode::MapFlatten(f)),
                        BuiltinMethod::First => Some(Opcode::MapFirst(f)),
                        BuiltinMethod::Last => Some(Opcode::MapLast(f)),
                        _ => None,
                    };
                    if let Some(o) = fused {
                        out.pop();
                        out.push(o);
                        continue;
                    }
                }
                // filter(p) + last() → FilterLast (reverse scan, early exit).
                // filter(p) + first() is handled by pass_find_quantifier
                // (emits FindFirst).
                if a.method == BuiltinMethod::Filter && a.sub_progs.len() >= 1
                   && b.method == BuiltinMethod::Last && b.sub_progs.is_empty() {
                    let pred = Arc::clone(&a.sub_progs[0]);
                    out.pop();
                    out.push(Opcode::FilterLast { pred });
                    continue;
                }
                // filter(p) + take_while(q) → FilterTakeWhile
                if a.method == BuiltinMethod::Filter && a.sub_progs.len() >= 1
                   && b.method == BuiltinMethod::TakeWhile && b.sub_progs.len() >= 1 {
                    let pred = Arc::clone(&a.sub_progs[0]);
                    let stop = Arc::clone(&b.sub_progs[0]);
                    out.pop();
                    out.push(Opcode::FilterTakeWhile { pred, stop });
                    continue;
                }
                // filter(p) + drop_while(q) → FilterDropWhile
                if a.method == BuiltinMethod::Filter && a.sub_progs.len() >= 1
                   && b.method == BuiltinMethod::DropWhile && b.sub_progs.len() >= 1 {
                    let pred = Arc::clone(&a.sub_progs[0]);
                    let drop = Arc::clone(&b.sub_progs[0]);
                    out.pop();
                    out.push(Opcode::FilterDropWhile { pred, drop });
                    continue;
                }
                // map(f) + unique() → MapUnique
                if a.method == BuiltinMethod::Map && a.sub_progs.len() >= 1
                   && b.method == BuiltinMethod::Unique && b.sub_progs.is_empty() {
                    let f = Arc::clone(&a.sub_progs[0]);
                    out.pop();
                    out.push(Opcode::MapUnique(f));
                    continue;
                }
                // trim + upper/lower  and  upper/lower + trim  → fused StrXY.
                // Both calls take no arguments.
                if a.sub_progs.is_empty() && b.sub_progs.is_empty() {
                    let fused_str = match (a.method, b.method) {
                        (BuiltinMethod::Trim,  BuiltinMethod::Upper) => Some(Opcode::StrTrimUpper),
                        (BuiltinMethod::Trim,  BuiltinMethod::Lower) => Some(Opcode::StrTrimLower),
                        (BuiltinMethod::Upper, BuiltinMethod::Trim)  => Some(Opcode::StrUpperTrim),
                        (BuiltinMethod::Lower, BuiltinMethod::Trim)  => Some(Opcode::StrLowerTrim),
                        _ => None,
                    };
                    if let Some(o) = fused_str {
                        out.pop();
                        out.push(o);
                        continue;
                    }
                }
                // split(sep) + reverse() — detect; only actually fuse when next
                // op is join(sep) with the same literal sep.  Done in a
                // dedicated 3-way pass below via lookahead buffer.
                // map(@.to_json()) + join(sep) → MapToJsonJoin { sep_prog }
                // Body is one of:
                //   [PushCurrent, CallMethod(ToJson, empty)]     — `@.to_json()`
                //   [LoadIdent(_), CallMethod(ToJson, empty)]    — `lambda x: x.to_json()`
                if a.method == BuiltinMethod::Map && a.sub_progs.len() == 1
                   && b.method == BuiltinMethod::Join && b.sub_progs.len() == 1 {
                    let body = &a.sub_progs[0].ops;
                    let is_to_json_body = matches!(&body[..],
                        [Opcode::PushCurrent, Opcode::CallMethod(c)]
                            if c.method == BuiltinMethod::ToJson
                               && c.sub_progs.is_empty())
                        || matches!(&body[..],
                        [Opcode::LoadIdent(_), Opcode::CallMethod(c)]
                            if c.method == BuiltinMethod::ToJson
                               && c.sub_progs.is_empty());
                    if is_to_json_body {
                        let sep_prog = Arc::clone(&b.sub_progs[0]);
                        out.pop();
                        out.push(Opcode::MapToJsonJoin { sep_prog });
                        continue;
                    }
                }
            }
            // map(@.replace(lit, lit))   or   map(@.replace_all(lit, lit))
            // → MapReplaceLit { needle, with, all } — single-op CallMethod.
            //
            // Also detects two-step chains:
            //   map(@.upper().replace(lit, lit))  → MapUpperReplaceLit
            //   map(@.lower().replace(lit, lit))  → MapLowerReplaceLit
            if let Opcode::CallMethod(a) = &op {
                if a.method == BuiltinMethod::Map && a.sub_progs.len() == 1 {
                    let body = &a.sub_progs[0].ops;
                    // Body shape 1: [PushCurrent, CallMethod(Replace|ReplaceAll, [PushStr, PushStr])]
                    let fused = if let [Opcode::PushCurrent, Opcode::CallMethod(inner)] = &body[..] {
                        let is_replace = inner.method == BuiltinMethod::Replace
                                      || inner.method == BuiltinMethod::ReplaceAll;
                        if is_replace && inner.sub_progs.len() == 2 {
                            let n = trivial_push_str(&inner.sub_progs[0].ops);
                            let w = trivial_push_str(&inner.sub_progs[1].ops);
                            match (n, w) {
                                (Some(needle), Some(with)) => {
                                    let all = inner.method == BuiltinMethod::ReplaceAll;
                                    Some(Opcode::MapReplaceLit { needle, with, all })
                                }
                                _ => None,
                            }
                        } else { None }
                    } else if let [Opcode::PushCurrent,
                                    Opcode::CallMethod(case_op),
                                    Opcode::CallMethod(inner)] = &body[..] {
                        // Body shape 2: [PushCurrent, CallMethod(Upper|Lower,[]),
                        //                CallMethod(Replace|ReplaceAll, [PushStr, PushStr])]
                        let is_replace = inner.method == BuiltinMethod::Replace
                                      || inner.method == BuiltinMethod::ReplaceAll;
                        let is_case_nullary = case_op.sub_progs.is_empty()
                            && (case_op.method == BuiltinMethod::Upper
                             || case_op.method == BuiltinMethod::Lower);
                        if is_case_nullary && is_replace && inner.sub_progs.len() == 2 {
                            let n = trivial_push_str(&inner.sub_progs[0].ops);
                            let w = trivial_push_str(&inner.sub_progs[1].ops);
                            match (n, w) {
                                (Some(needle), Some(with)) => {
                                    let all = inner.method == BuiltinMethod::ReplaceAll;
                                    if case_op.method == BuiltinMethod::Upper {
                                        Some(Opcode::MapUpperReplaceLit { needle, with, all })
                                    } else {
                                        Some(Opcode::MapLowerReplaceLit { needle, with, all })
                                    }
                                }
                                _ => None,
                            }
                        } else { None }
                    } else { None };
                    if let Some(o) = fused {
                        out.push(o);
                        continue;
                    }
                }
            }
            // map(f"...") with no ident captures → MapFString(parts)
            // Body shape: [FString(parts)]
            if let Opcode::CallMethod(a) = &op {
                if a.method == BuiltinMethod::Map && a.sub_progs.len() == 1 {
                    let body = &a.sub_progs[0].ops;
                    if let [Opcode::FString(parts)] = &body[..] {
                        out.push(Opcode::MapFString(Arc::clone(parts)));
                        continue;
                    }
                }
            }
            // map(@.slice(lit, lit)) → MapStrSlice { start, end }
            // Body shape: [PushCurrent, CallMethod(Slice, [PushInt(a), PushInt(b)])]
            if let Opcode::CallMethod(a) = &op {
                if a.method == BuiltinMethod::Map && a.sub_progs.len() == 1 {
                    let body = &a.sub_progs[0].ops;
                    if let [Opcode::PushCurrent, Opcode::CallMethod(inner)] = &body[..] {
                        if inner.method == BuiltinMethod::Slice {
                            let start = match inner.sub_progs.first()
                                .map(|p| p.ops.as_ref()) {
                                Some([Opcode::PushInt(n)]) => Some(*n),
                                _ => None,
                            };
                            let end = match inner.sub_progs.get(1)
                                .map(|p| p.ops.as_ref()) {
                                Some([Opcode::PushInt(n)]) => Some(Some(*n)),
                                None => Some(None),
                                _ => None,
                            };
                            if let (Some(s), Some(e)) = (start, end) {
                                out.push(Opcode::MapStrSlice { start: s, end: e });
                                continue;
                            }
                        }
                    }
                }
            }
            // map({k1, k2, ..}) with all `Short` entries → MapProject
            if let Opcode::CallMethod(a) = &op {
                if a.method == BuiltinMethod::Map && a.sub_progs.len() == 1 {
                    let body = &a.sub_progs[0].ops;
                    if let [Opcode::MakeObj(entries)] = &body[..] {
                        let all_short: Option<Vec<Arc<str>>> = entries.iter()
                            .map(|e| match e {
                                CompiledObjEntry::Short { name, .. } => Some(name.clone()),
                                _ => None,
                            })
                            .collect();
                        if let Some(keys) = all_short {
                            if !keys.is_empty() {
                                let ics: Vec<std::sync::atomic::AtomicU64> =
                                    keys.iter().map(|_| std::sync::atomic::AtomicU64::new(0)).collect();
                                out.push(Opcode::MapProject {
                                    keys: keys.into(),
                                    ics: ics.into(),
                                });
                                continue;
                            }
                        }
                    }
                }
            }
            // map(@.split(sep).map(len).sum()) → MapSplitLenSum { sep }
            // Body shape (post pass_field_specialise rewrote map(len) as
            // MapFieldSum("len")):
            //   [PushCurrent, CallMethod(Split, [PushStr(sep)]), MapFieldSum("len")]
            if let Opcode::CallMethod(a) = &op {
                if a.method == BuiltinMethod::Map && a.sub_progs.len() == 1 {
                    let body = &a.sub_progs[0].ops;
                    let fused = if let [Opcode::PushCurrent,
                                         Opcode::CallMethod(split),
                                         Opcode::MapFieldSum(field)] = &body[..] {
                        if split.method == BuiltinMethod::Split
                           && split.sub_progs.len() == 1
                           && field.as_ref() == "len" {
                            let sep_opt = trivial_push_str(&split.sub_progs[0].ops);
                            sep_opt.map(|sep| Opcode::MapSplitLenSum { sep })
                        } else { None }
                    } else { None };
                    if let Some(o) = fused {
                        out.push(o);
                        continue;
                    }
                }
            }
            // map(prefix + @ + suffix), map(prefix + @), map(@ + suffix)
            //   → MapStrConcat { prefix, suffix }
            // Body shapes:
            //   [PushStr(p), PushCurrent, Add, PushStr(s), Add]     prefix+suffix
            //   [PushStr(p), PushCurrent, Add]                      prefix only
            //   [PushCurrent, PushStr(s), Add]                      suffix only
            if let Opcode::CallMethod(a) = &op {
                if a.method == BuiltinMethod::Map && a.sub_progs.len() == 1 {
                    let body = &a.sub_progs[0].ops;
                    let empty: Arc<str> = Arc::from("");
                    let fused = match &body[..] {
                        [Opcode::PushStr(p), Opcode::PushCurrent, Opcode::Add,
                         Opcode::PushStr(s), Opcode::Add] => Some(Opcode::MapStrConcat {
                            prefix: p.clone(), suffix: s.clone(),
                        }),
                        [Opcode::PushStr(p), Opcode::PushCurrent, Opcode::Add] =>
                            Some(Opcode::MapStrConcat {
                                prefix: p.clone(), suffix: empty.clone(),
                            }),
                        [Opcode::PushCurrent, Opcode::PushStr(s), Opcode::Add] =>
                            Some(Opcode::MapStrConcat {
                                prefix: empty.clone(), suffix: s.clone(),
                            }),
                        _ => None,
                    };
                    if let Some(o) = fused {
                        out.push(o);
                        continue;
                    }
                }
            }
            // MapSplitCount followed by Sum → MapSplitCountSum (scalar, no
            // intermediate Int array).
            if let Opcode::CallMethod(b) = &op {
                if b.method == BuiltinMethod::Sum && b.sub_progs.is_empty() {
                    if let Some(Opcode::MapSplitCount { sep }) = out.last() {
                        let sep = Arc::clone(sep);
                        out.pop();
                        out.push(Opcode::MapSplitCountSum { sep });
                        continue;
                    }
                }
            }
            // map(@.split(lit).count()|.first()|.nth(lit)) → MapSplitCount /
            // MapSplitFirst / MapSplitNth.  Eliminates N per-row Arcs from
            // split materialisation when the consumer only needs the count
            // or a single segment.
            if let Opcode::CallMethod(a) = &op {
                if a.method == BuiltinMethod::Map && a.sub_progs.len() == 1 {
                    let body = &a.sub_progs[0].ops;
                    let fused = if let [Opcode::PushCurrent,
                                         Opcode::CallMethod(split),
                                         Opcode::CallMethod(cons)] = &body[..] {
                        if split.method == BuiltinMethod::Split && split.sub_progs.len() == 1 {
                            let sep_opt = trivial_push_str(&split.sub_progs[0].ops);
                            match (sep_opt, cons.method, cons.sub_progs.len()) {
                                (Some(sep), BuiltinMethod::Count, 0)
                              | (Some(sep), BuiltinMethod::Len,   0) =>
                                    Some(Opcode::MapSplitCount { sep }),
                                (Some(sep), BuiltinMethod::First, 0) =>
                                    Some(Opcode::MapSplitFirst { sep }),
                                (Some(sep), BuiltinMethod::Nth,   1) => {
                                    if let [Opcode::PushInt(n)] = &cons.sub_progs[0].ops[..] {
                                        if *n >= 0 {
                                            Some(Opcode::MapSplitNth { sep, n: *n as usize })
                                        } else { None }
                                    } else { None }
                                }
                                _ => None,
                            }
                        } else { None }
                    } else { None };
                    if let Some(o) = fused {
                        out.push(o);
                        continue;
                    }
                }
            }
            out.push(op);
        }
        out
    }

    /// Three-way string-method fusion: `split(s).reverse().join(s)` with
    /// matching string literal `s` collapses to `StrSplitReverseJoin`.
    fn pass_string_chain_fusion(ops: Vec<Opcode>) -> Vec<Opcode> {
        let mut out: Vec<Opcode> = Vec::with_capacity(ops.len());
        let mut i = 0;
        while i < ops.len() {
            if i + 2 < ops.len() {
                if let (Opcode::CallMethod(a),
                        Opcode::CallMethod(b),
                        Opcode::CallMethod(c)) = (&ops[i], &ops[i + 1], &ops[i + 2]) {
                    if a.method == BuiltinMethod::Split && a.sub_progs.len() == 1
                       && b.method == BuiltinMethod::Reverse && b.sub_progs.is_empty()
                       && c.method == BuiltinMethod::Join && c.sub_progs.len() == 1 {
                        let sep_a = trivial_push_str(&a.sub_progs[0].ops);
                        let sep_c = trivial_push_str(&c.sub_progs[0].ops);
                        if let (Some(s1), Some(s2)) = (sep_a, sep_c) {
                            if s1 == s2 {
                                out.push(Opcode::StrSplitReverseJoin { sep: s1 });
                                i += 3;
                                continue;
                            }
                        }
                    }
                }
            }
            out.push(ops[i].clone());
            i += 1;
        }
        out
    }

    /// Lower generic fused opcodes to field-specialised variants when the
    /// sub-program is a trivial `GetField(k)` read. Runs AFTER
    /// pass_find_quantifier / pass_filter_count so those passes see the
    /// generic `CallMethod(Filter)` / `MapSum` forms first.
    fn pass_field_specialise(ops: Vec<Opcode>) -> Vec<Opcode> {
        let mut out2: Vec<Opcode> = Vec::with_capacity(ops.len());
        for op in ops {
            match op {
                Opcode::MapSum(ref f) => {
                    if let Some(k) = trivial_field(&f.ops) {
                        out2.push(Opcode::MapFieldSum(k)); continue;
                    }
                }
                Opcode::MapAvg(ref f) => {
                    if let Some(k) = trivial_field(&f.ops) {
                        out2.push(Opcode::MapFieldAvg(k)); continue;
                    }
                }
                Opcode::MapMin(ref f) => {
                    if let Some(k) = trivial_field(&f.ops) {
                        out2.push(Opcode::MapFieldMin(k)); continue;
                    }
                }
                Opcode::MapMax(ref f) => {
                    if let Some(k) = trivial_field(&f.ops) {
                        out2.push(Opcode::MapFieldMax(k)); continue;
                    }
                }
                Opcode::MapUnique(ref f) => {
                    if let Some(k) = trivial_field(&f.ops) {
                        out2.push(Opcode::MapFieldUnique(k)); continue;
                    }
                    if let Some(chain) = trivial_field_chain(&f.ops) {
                        out2.push(Opcode::MapFieldChainUnique(chain)); continue;
                    }
                }
                Opcode::FilterCount(ref pred) => {
                    if let Some(pairs) = detect_field_eq_conjuncts(&pred.ops) {
                        out2.push(Opcode::FilterFieldsAllEqLitCount(Arc::from(pairs)));
                        continue;
                    }
                    if let Some(triples) = detect_field_cmp_conjuncts(&pred.ops) {
                        out2.push(Opcode::FilterFieldsAllCmpLitCount(Arc::from(triples)));
                        continue;
                    }
                }
                Opcode::CallMethod(ref b) => {
                    // map(k)    → MapField(k)
                    if b.method == BuiltinMethod::Map && b.sub_progs.len() == 1 {
                        if let Some(k) = trivial_field(&b.sub_progs[0].ops) {
                            out2.push(Opcode::MapField(k)); continue;
                        }
                        if let Some(chain) = trivial_field_chain(&b.sub_progs[0].ops) {
                            out2.push(Opcode::MapFieldChain(chain)); continue;
                        }
                    }
                    // group_by(k) → GroupByField(k)
                    if b.method == BuiltinMethod::GroupBy && b.sub_progs.len() == 1 {
                        if let Some(k) = trivial_field(&b.sub_progs[0].ops) {
                            out2.push(Opcode::GroupByField(k)); continue;
                        }
                    }
                    // count_by(k) → CountByField(k)
                    if b.method == BuiltinMethod::CountBy && b.sub_progs.len() == 1 {
                        if let Some(k) = trivial_field(&b.sub_progs[0].ops) {
                            out2.push(Opcode::CountByField(k)); continue;
                        }
                    }
                    // unique_by(k) / uniqueBy(k) → UniqueByField(k)
                    if b.method == BuiltinMethod::Unknown
                       && matches!(b.name.as_ref(), "unique_by" | "uniqueBy")
                       && b.sub_progs.len() == 1 {
                        if let Some(k) = trivial_field(&b.sub_progs[0].ops) {
                            out2.push(Opcode::UniqueByField(k)); continue;
                        }
                    }
                    // filter(field <cmp> lit|field) → FilterField*
                    if b.method == BuiltinMethod::Filter && b.sub_progs.len() == 1 {
                        if let Some(p) = detect_field_pred(&b.sub_progs[0].ops) {
                            let lowered = match p {
                                FieldPred::FieldCmpLit(k, super::ast::BinOp::Eq, lit) =>
                                    Opcode::FilterFieldEqLit(k, lit),
                                FieldPred::FieldCmpLit(k, op, lit) =>
                                    Opcode::FilterFieldCmpLit(k, op, lit),
                                FieldPred::FieldCmpField(k1, op, k2) =>
                                    Opcode::FilterFieldCmpField(k1, op, k2),
                            };
                            out2.push(lowered); continue;
                        }
                        // filter(@ <cmp> lit) → FilterCurrentCmpLit
                        if let Some((op, lit)) = detect_current_cmp_lit(&b.sub_progs[0].ops) {
                            out2.push(Opcode::FilterCurrentCmpLit(op, lit));
                            continue;
                        }
                        // filter(@.starts_with/ends_with/contains(lit)) → FilterStrVec*
                        if let Some((kind, lit)) = detect_current_str_method(&b.sub_progs[0].ops) {
                            out2.push(match kind {
                                StrVecPred::StartsWith => Opcode::FilterStrVecStartsWith(lit),
                                StrVecPred::EndsWith   => Opcode::FilterStrVecEndsWith(lit),
                                StrVecPred::Contains   => Opcode::FilterStrVecContains(lit),
                            });
                            continue;
                        }
                    }
                    // map(@.upper/lower/trim()) → MapStrVec*
                    if b.method == BuiltinMethod::Map && b.sub_progs.len() == 1 {
                        if let Some(kind) = detect_current_str_nullary(&b.sub_progs[0].ops) {
                            out2.push(match kind {
                                StrVecMap::Upper => Opcode::MapStrVecUpper,
                                StrVecMap::Lower => Opcode::MapStrVecLower,
                                StrVecMap::Trim  => Opcode::MapStrVecTrim,
                            });
                            continue;
                        }
                        // map(@ <arith> lit) / map(lit <arith> @) → MapNumVecArith
                        if let Some((op, lit, flipped)) =
                            detect_current_arith_lit(&b.sub_progs[0].ops) {
                            out2.push(Opcode::MapNumVecArith { op, lit, flipped });
                            continue;
                        }
                        // map(-@) → MapNumVecNeg
                        if detect_current_neg(&b.sub_progs[0].ops) {
                            out2.push(Opcode::MapNumVecNeg);
                            continue;
                        }
                    }
                }
                _ => {}
            }
            out2.push(op);
        }
        // Third pass: fold `FilterField* + count()` into FilterField*Count,
        // and collapse chains of `MapField(k1) + MapFlatten(trivial k2) + ...`
        // into a single `FlatMapChain([k1,k2,...])`.
        let mut out3: Vec<Opcode> = Vec::with_capacity(out2.len());
        for op in out2 {
            // FilterField* + count()
            if let Opcode::CallMethod(ref b) = op {
                if b.method == BuiltinMethod::Count && b.sub_progs.is_empty() {
                    match out3.last().cloned() {
                        Some(Opcode::FilterFieldEqLit(k, lit)) => {
                            out3.pop();
                            out3.push(Opcode::FilterFieldEqLitCount(k, lit));
                            continue;
                        }
                        Some(Opcode::FilterFieldCmpLit(k, cop, lit)) => {
                            out3.pop();
                            out3.push(Opcode::FilterFieldCmpLitCount(k, cop, lit));
                            continue;
                        }
                        Some(Opcode::FilterFieldCmpField(k1, cop, k2)) => {
                            out3.pop();
                            out3.push(Opcode::FilterFieldCmpFieldCount(k1, cop, k2));
                            continue;
                        }
                        _ => {}
                    }
                }
            }
            // FilterField* + MapField(k) → FilterField*MapField (single pass)
            if let Opcode::MapField(ref kp) = op {
                match out3.last().cloned() {
                    Some(Opcode::FilterFieldEqLit(k, lit)) => {
                        out3.pop();
                        out3.push(Opcode::FilterFieldEqLitMapField(k, lit, kp.clone()));
                        continue;
                    }
                    Some(Opcode::FilterFieldCmpLit(k, cop, lit)) => {
                        out3.pop();
                        out3.push(Opcode::FilterFieldCmpLitMapField(k, cop, lit, kp.clone()));
                        continue;
                    }
                    _ => {}
                }
            }
            // MapField(k) + MapFlatten(trivial k2) → FlatMapChain([k, k2])
            if let Opcode::MapFlatten(ref f) = op {
                if let Some(k2) = trivial_field(&f.ops) {
                    match out3.last().cloned() {
                        Some(Opcode::MapField(k1)) => {
                            out3.pop();
                            out3.push(Opcode::FlatMapChain(Arc::from(vec![k1, k2])));
                            continue;
                        }
                        Some(Opcode::FlatMapChain(ks)) => {
                            let mut v: Vec<Arc<str>> = ks.iter().cloned().collect();
                            v.push(k2);
                            out3.pop();
                            out3.push(Opcode::FlatMapChain(Arc::from(v)));
                            continue;
                        }
                        _ => {}
                    }
                }
            }
            out3.push(op);
        }
        out3
    }

    /// Lower simple single-var list comprehensions to the same fused
    /// opcodes we already emit for `.filter(k op lit).map(k2)` chains.
    /// Matches `[x.k2 for x in <iter> (if x.k op lit)]` — a common shape
    /// that otherwise pays ~4 opcode dispatches per iteration.
    fn pass_list_comp_specialise(ops: Vec<Opcode>) -> Vec<Opcode> {
        #[inline]
        fn proj_key(ops: &[Opcode], var: &str) -> Option<Arc<str>> {
            match ops {
                [Opcode::LoadIdent(v), Opcode::GetField(k)] if v.as_ref() == var =>
                    Some(k.clone()),
                _ => None,
            }
        }
        #[inline]
        fn cond_pred(ops: &[Opcode], var: &str)
            -> Option<(Arc<str>, super::ast::BinOp, Val)>
        {
            if ops.len() != 4 { return None; }
            let k = match (&ops[0], &ops[1]) {
                (Opcode::LoadIdent(v), Opcode::GetField(k)) if v.as_ref() == var =>
                    k.clone(),
                _ => return None,
            };
            let lit = trivial_literal(&ops[2])?;
            let op = cmp_opcode(&ops[3])?;
            Some((k, op, lit))
        }

        let mut out: Vec<Opcode> = Vec::with_capacity(ops.len());
        for op in ops {
            if let Opcode::ListComp(ref spec) = op {
                if spec.vars.len() == 1 {
                    let var = spec.vars[0].as_ref();
                    if let Some(proj) = proj_key(&spec.expr.ops, var) {
                        match &spec.cond {
                            Some(cond) => {
                                if let Some((pk, cop, lit)) = cond_pred(&cond.ops, var) {
                                    for iop in spec.iter.ops.iter() {
                                        out.push(iop.clone());
                                    }
                                    if matches!(cop, super::ast::BinOp::Eq) {
                                        out.push(Opcode::FilterFieldEqLitMapField(pk, lit, proj));
                                    } else {
                                        out.push(Opcode::FilterFieldCmpLitMapField(pk, cop, lit, proj));
                                    }
                                    continue;
                                }
                            }
                            None => {
                                for iop in spec.iter.ops.iter() {
                                    out.push(iop.clone());
                                }
                                out.push(Opcode::MapField(proj));
                                continue;
                            }
                        }
                    }
                }
            }
            out.push(op);
        }
        out
    }

    fn sort_lam_param(prev: &CompiledCall) -> Option<Arc<str>> {
        match prev.orig_args.first() {
            Some(Arg::Pos(Expr::Lambda { params, .. })) if !params.is_empty() =>
                Some(Arc::from(params[0].as_str())),
            _ => None,
        }
    }

    /// Replace expensive ops with cheaper equivalents:
    ///   sort() + first()    → min()
    ///   sort() + last()     → max()
    ///   sort() + [0]        → min()
    ///   sort() + [-1]       → max()
    ///   reverse() + first() → last()
    ///   reverse() + last()  → first()
    ///   sort_by(k) + first()/last() → ArgExtreme (O(N) scan)
    fn pass_strength_reduce(ops: Vec<Opcode>) -> Vec<Opcode> {
        let mut out: Vec<Opcode> = Vec::with_capacity(ops.len());
        for op in ops {
            // Pattern: [..., prev_method_call, current_op]
            if let Some(Opcode::CallMethod(prev)) = out.last().cloned() {
                let replaced = match (prev.method, &op) {
                    // sort() + [0] → min()
                    (BuiltinMethod::Sort, Opcode::GetIndex(0)) if prev.sub_progs.is_empty() =>
                        Some(make_noarg_call(BuiltinMethod::Min, "min")),
                    // sort() + [-1] → max()
                    (BuiltinMethod::Sort, Opcode::GetIndex(-1)) if prev.sub_progs.is_empty() =>
                        Some(make_noarg_call(BuiltinMethod::Max, "max")),
                    // sort() + first() → min()
                    (BuiltinMethod::Sort, Opcode::CallMethod(next))
                        if prev.sub_progs.is_empty() && next.method == BuiltinMethod::First =>
                        Some(make_noarg_call(BuiltinMethod::Min, "min")),
                    // sort() + last() → max()
                    (BuiltinMethod::Sort, Opcode::CallMethod(next))
                        if prev.sub_progs.is_empty() && next.method == BuiltinMethod::Last =>
                        Some(make_noarg_call(BuiltinMethod::Max, "max")),
                    // sort_by(k) + first() → ArgExtreme{key, max=false}
                    (BuiltinMethod::Sort, Opcode::CallMethod(next))
                        if prev.sub_progs.len() == 1
                           && next.method == BuiltinMethod::First
                           && next.sub_progs.is_empty() =>
                        Some(Opcode::ArgExtreme {
                            key: Arc::clone(&prev.sub_progs[0]),
                            lam_param: Self::sort_lam_param(&prev),
                            max: false,
                        }),
                    // sort_by(k) + last() → ArgExtreme{key, max=true}
                    (BuiltinMethod::Sort, Opcode::CallMethod(next))
                        if prev.sub_progs.len() == 1
                           && next.method == BuiltinMethod::Last
                           && next.sub_progs.is_empty() =>
                        Some(Opcode::ArgExtreme {
                            key: Arc::clone(&prev.sub_progs[0]),
                            lam_param: Self::sort_lam_param(&prev),
                            max: true,
                        }),
                    // reverse() + first() → last()
                    (BuiltinMethod::Reverse, Opcode::CallMethod(next))
                        if next.method == BuiltinMethod::First =>
                        Some(make_noarg_call(BuiltinMethod::Last, "last")),
                    // reverse() + last() → first()
                    (BuiltinMethod::Reverse, Opcode::CallMethod(next))
                        if next.method == BuiltinMethod::Last =>
                        Some(make_noarg_call(BuiltinMethod::First, "first")),
                    // sort() + [0:n] → TopN(n, asc=true)
                    (BuiltinMethod::Sort, Opcode::GetSlice(from, Some(to)))
                        if prev.sub_progs.is_empty()
                           && (from.is_none() || *from == Some(0))
                           && *to > 0 =>
                        Some(Opcode::TopN { n: *to as usize, asc: true }),
                    // Cardinality-preserving op + len/count → drop the first op.
                    // sort / reverse preserve length by definition; map is
                    // 1:1 so it also preserves length, and `count` only needs
                    // the input array length.
                    (BuiltinMethod::Sort | BuiltinMethod::Reverse | BuiltinMethod::Map,
                     Opcode::CallMethod(next))
                        if next.sub_progs.is_empty()
                           && (next.method == BuiltinMethod::Len
                               || next.method == BuiltinMethod::Count) =>
                        Some(Opcode::CallMethod(Arc::clone(next))),
                    // Order-independent aggregate after sort/reverse → drop
                    // the reorder.  sum / avg / min / max only inspect the
                    // multiset of elements, not their order.
                    (BuiltinMethod::Sort | BuiltinMethod::Reverse,
                     Opcode::CallMethod(next))
                        if prev.sub_progs.is_empty()
                           && next.sub_progs.is_empty()
                           && matches!(next.method,
                                BuiltinMethod::Sum | BuiltinMethod::Avg
                              | BuiltinMethod::Min | BuiltinMethod::Max) =>
                        Some(Opcode::CallMethod(Arc::clone(next))),
                    // Idempotent: f(f(x)) == f(x).  `sort(k)` is idempotent
                    // only when both calls use the same key, so we restrict
                    // the no-arg case; `unique()` dedup is always idempotent.
                    (BuiltinMethod::Sort, Opcode::CallMethod(next))
                        if prev.sub_progs.is_empty()
                           && next.method == BuiltinMethod::Sort
                           && next.sub_progs.is_empty() =>
                        Some(Opcode::CallMethod(Arc::clone(next))),
                    (BuiltinMethod::Unique, Opcode::CallMethod(next))
                        if next.method == BuiltinMethod::Unique =>
                        Some(Opcode::CallMethod(Arc::clone(next))),
                    // unique() + count()/len() → UniqueCount (skip materialising dedup array).
                    (BuiltinMethod::Unique, Opcode::CallMethod(next))
                        if prev.sub_progs.is_empty()
                           && next.sub_progs.is_empty()
                           && (next.method == BuiltinMethod::Count
                               || next.method == BuiltinMethod::Len) =>
                        Some(Opcode::UniqueCount),
                    _ => None,
                };
                if let Some(rep) = replaced {
                    out.pop();
                    out.push(rep);
                    continue;
                }
                // Involution: reverse().reverse() → drop both.
                if prev.method == BuiltinMethod::Reverse && prev.sub_progs.is_empty() {
                    if let Opcode::CallMethod(next) = &op {
                        if next.method == BuiltinMethod::Reverse && next.sub_progs.is_empty() {
                            out.pop();
                            continue;
                        }
                    }
                }
            }
            out.push(op);
        }
        out
    }

    /// Fuse runs of `GetField` not consumed by `pass_root_chain` into a
    /// single `FieldChain`.  Applies mid-program where the object on TOS
    /// came from elsewhere (method return, filter, comprehension).  Singletons
    /// are left as-is — fusion only triggers at length ≥ 2.
    fn pass_field_chain(ops: Vec<Opcode>) -> Vec<Opcode> {
        // Both GetField(k) and OptField(k) devolve to `get_field(k)` which
        // returns Null for non-objects, so OptField can be absorbed into a
        // FieldChain: null propagates through the remaining get_field calls.
        fn field_key(op: &Opcode) -> Option<Arc<str>> {
            match op {
                Opcode::GetField(k) | Opcode::OptField(k) => Some(Arc::clone(k)),
                _ => None,
            }
        }
        let mut out = Vec::with_capacity(ops.len());
        let mut it = ops.into_iter().peekable();
        while let Some(op) = it.next() {
            if let Some(k0) = field_key(&op) {
                if it.peek().and_then(field_key).is_some() {
                    let mut chain: Vec<Arc<str>> = vec![k0];
                    while let Some(k) = it.peek().and_then(field_key) {
                        it.next();
                        chain.push(k);
                    }
                    out.push(Opcode::FieldChain(Arc::new(FieldChainData::new(chain.into()))));
                    continue;
                }
                out.push(op);
            } else {
                out.push(op);
            }
        }
        out
    }

    /// Fuse `PushRoot + GetField(k1) + GetField(k2) ...` → `RootChain([k1,k2,...])`.
    fn pass_root_chain(ops: Vec<Opcode>) -> Vec<Opcode> {
        let mut out = Vec::with_capacity(ops.len());
        let mut it = ops.into_iter().peekable();
        while let Some(op) = it.next() {
            if matches!(op, Opcode::PushRoot) {
                let mut chain: Vec<Arc<str>> = Vec::new();
                while let Some(Opcode::GetField(_)) = it.peek() {
                    if let Some(Opcode::GetField(k)) = it.next() {
                        chain.push(k);
                    }
                }
                if chain.is_empty() {
                    out.push(Opcode::PushRoot);
                } else {
                    out.push(Opcode::RootChain(chain.into()));
                }
            } else {
                out.push(op);
            }
        }
        out
    }

    /// Fuse `CallMethod(filter/pred) + CallMethod(len/count)` → `FilterCount(pred)`.
    fn pass_filter_count(ops: Vec<Opcode>) -> Vec<Opcode> {
        let mut out = Vec::with_capacity(ops.len());
        let mut it = ops.into_iter().peekable();
        while let Some(op) = it.next() {
            if let Opcode::CallMethod(ref call) = op {
                let is_filter_like = call.method == BuiltinMethod::Filter
                    || (call.method == BuiltinMethod::Unknown
                        && matches!(call.name.as_ref(), "find" | "find_all" | "findAll"));
                if is_filter_like && call.sub_progs.len() == 1 {
                    let is_len = matches!(it.peek(),
                        Some(Opcode::CallMethod(c))
                            if c.method == BuiltinMethod::Len || c.method == BuiltinMethod::Count
                    );
                    if is_len {
                        let pred = Arc::clone(&call.sub_progs[0]);
                        it.next(); // consume Len/Count
                        out.push(Opcode::FilterCount(pred));
                        continue;
                    }
                }
            }
            out.push(op);
        }
        out
    }

    /// Fuse `InlineFilter(pred) + Quantifier(First/One)` → `FindFirst/FindOne(pred)`.
    /// Also fuses `CallMethod(Filter, pred) + Quantifier(...)` for explicit `.filter()`.
    fn pass_find_quantifier(ops: Vec<Opcode>) -> Vec<Opcode> {
        let mut out = Vec::with_capacity(ops.len());
        let mut it = ops.into_iter().peekable();
        while let Some(op) = it.next() {
            let pred_opt: Option<Arc<Program>> = match &op {
                Opcode::InlineFilter(p) => Some(Arc::clone(p)),
                Opcode::CallMethod(c) if c.method == BuiltinMethod::Filter && !c.sub_progs.is_empty()
                    => Some(Arc::clone(&c.sub_progs[0])),
                _ => None,
            };
            if let Some(pred) = pred_opt {
                match it.peek() {
                    Some(Opcode::Quantifier(QuantifierKind::First)) => {
                        it.next();
                        out.push(Opcode::FindFirst(pred));
                        continue;
                    }
                    Some(Opcode::Quantifier(QuantifierKind::One)) => {
                        it.next();
                        out.push(Opcode::FindOne(pred));
                        continue;
                    }
                    // `.filter(p).first()` — scans until predicate holds, returns
                    // that item or null.  Skips materialising a filtered array.
                    Some(Opcode::CallMethod(c))
                        if c.method == BuiltinMethod::First && c.sub_progs.is_empty() =>
                    {
                        it.next();
                        out.push(Opcode::FindFirst(pred));
                        continue;
                    }
                    _ => {}
                }
            }
            out.push(op);
        }
        out
    }

    /// Eliminate redundant adjacent method calls:
    ///   reverse() + reverse()         → identity (both dropped)
    ///   unique() + unique()           → unique()
    ///   compact() + compact()         → compact()
    ///   sort() + sort(k)              → sort(k)      (later sort wins on same-array)
    ///   sort(k) + sort(k)             → sort(k)
    ///   Quantifier + Quantifier       → second only  (first wrap is scalar anyway)
    fn pass_redundant_ops(ops: Vec<Opcode>) -> Vec<Opcode> {
        let mut out: Vec<Opcode> = Vec::with_capacity(ops.len());
        for op in ops {
            match (&op, out.last()) {
                // reverse + reverse: drop both
                (Opcode::CallMethod(b), Some(Opcode::CallMethod(a)))
                    if a.method == BuiltinMethod::Reverse && b.method == BuiltinMethod::Reverse =>
                {
                    out.pop();
                    continue;
                }
                // idempotent method pairs: keep second only (drop first)
                (Opcode::CallMethod(b), Some(Opcode::CallMethod(a)))
                    if a.method == b.method && matches!(a.method,
                        BuiltinMethod::Unique | BuiltinMethod::Compact)
                        && a.sub_progs.is_empty() && b.sub_progs.is_empty() =>
                {
                    out.pop();
                    out.push(op);
                    continue;
                }
                // sort + sort(_): later sort wins, drop the first
                (Opcode::CallMethod(b), Some(Opcode::CallMethod(a)))
                    if a.method == BuiltinMethod::Sort && b.method == BuiltinMethod::Sort =>
                {
                    out.pop();
                    out.push(op);
                    continue;
                }
                // Quantifier + Quantifier: second wins (first unwraps scalar,
                // second is no-op on scalar — but keeping second preserves error
                // semantics of `!`). Drop first.
                (Opcode::Quantifier(_), Some(Opcode::Quantifier(_))) => {
                    out.pop();
                    out.push(op);
                    continue;
                }
                // reverse + last → first (strength reduction fallback after sort)
                // already handled in pass_strength_reduce
                // Not + Not: double negation → drop both
                (Opcode::Not, Some(Opcode::Not)) => {
                    out.pop();
                    continue;
                }
                // Neg + Neg: --x → x
                (Opcode::Neg, Some(Opcode::Neg)) => {
                    out.pop();
                    continue;
                }
                _ => {}
            }
            out.push(op);
        }
        out
    }

    /// Constant-fold adjacent integer arithmetic + bool short-circuits.
    fn pass_const_fold(ops: Vec<Opcode>) -> Vec<Opcode> {
        let mut out = Vec::with_capacity(ops.len());
        let mut i = 0;
        while i < ops.len() {
            // 2-op bool short-circuit folds:
            //   PushBool(false) + AndOp(_)  → PushBool(false)
            //   PushBool(true)  + OrOp(_)   → PushBool(true)
            if i + 1 < ops.len() {
                let folded = match (&ops[i], &ops[i+1]) {
                    (Opcode::PushBool(false), Opcode::AndOp(_)) =>
                        Some(Opcode::PushBool(false)),
                    (Opcode::PushBool(true),  Opcode::OrOp(_)) =>
                        Some(Opcode::PushBool(true)),
                    _ => None,
                };
                if let Some(folded) = folded {
                    out.push(folded);
                    i += 2;
                    continue;
                }
            }
            // 2-op unary folds
            if i + 1 < ops.len() {
                let folded = match (&ops[i], &ops[i+1]) {
                    (Opcode::PushBool(b), Opcode::Not) =>
                        Some(Opcode::PushBool(!b)),
                    (Opcode::PushInt(n), Opcode::Neg) =>
                        Some(Opcode::PushInt(-n)),
                    (Opcode::PushFloat(f), Opcode::Neg) =>
                        Some(Opcode::PushFloat(-f)),
                    _ => None,
                };
                if let Some(folded) = folded {
                    out.push(folded);
                    i += 2;
                    continue;
                }
            }
            // 3-op arithmetic + comparison folds
            if i + 2 < ops.len() {
                let folded = match (&ops[i], &ops[i+1], &ops[i+2]) {
                    (Opcode::PushInt(a), Opcode::PushInt(b), Opcode::Add) =>
                        Some(Opcode::PushInt(a + b)),
                    (Opcode::PushInt(a), Opcode::PushInt(b), Opcode::Sub) =>
                        Some(Opcode::PushInt(a - b)),
                    (Opcode::PushInt(a), Opcode::PushInt(b), Opcode::Mul) =>
                        Some(Opcode::PushInt(a * b)),
                    (Opcode::PushInt(a), Opcode::PushInt(b), Opcode::Mod) if *b != 0 =>
                        Some(Opcode::PushInt(a % b)),
                    (Opcode::PushInt(a), Opcode::PushInt(b), Opcode::Div) if *b != 0 =>
                        Some(Opcode::PushFloat(*a as f64 / *b as f64)),
                    (Opcode::PushFloat(a), Opcode::PushFloat(b), Opcode::Add) =>
                        Some(Opcode::PushFloat(a + b)),
                    (Opcode::PushFloat(a), Opcode::PushFloat(b), Opcode::Sub) =>
                        Some(Opcode::PushFloat(a - b)),
                    (Opcode::PushFloat(a), Opcode::PushFloat(b), Opcode::Mul) =>
                        Some(Opcode::PushFloat(a * b)),
                    (Opcode::PushFloat(a), Opcode::PushFloat(b), Opcode::Div) if *b != 0.0 =>
                        Some(Opcode::PushFloat(a / b)),
                    // Mixed int/float arithmetic
                    (Opcode::PushInt(a), Opcode::PushFloat(b), Opcode::Add) =>
                        Some(Opcode::PushFloat(*a as f64 + b)),
                    (Opcode::PushFloat(a), Opcode::PushInt(b), Opcode::Add) =>
                        Some(Opcode::PushFloat(a + *b as f64)),
                    (Opcode::PushInt(a), Opcode::PushFloat(b), Opcode::Sub) =>
                        Some(Opcode::PushFloat(*a as f64 - b)),
                    (Opcode::PushFloat(a), Opcode::PushInt(b), Opcode::Sub) =>
                        Some(Opcode::PushFloat(a - *b as f64)),
                    (Opcode::PushInt(a), Opcode::PushFloat(b), Opcode::Mul) =>
                        Some(Opcode::PushFloat(*a as f64 * b)),
                    (Opcode::PushFloat(a), Opcode::PushInt(b), Opcode::Mul) =>
                        Some(Opcode::PushFloat(a * *b as f64)),
                    (Opcode::PushInt(a), Opcode::PushFloat(b), Opcode::Div) if *b != 0.0 =>
                        Some(Opcode::PushFloat(*a as f64 / b)),
                    (Opcode::PushFloat(a), Opcode::PushInt(b), Opcode::Div) if *b != 0 =>
                        Some(Opcode::PushFloat(a / *b as f64)),
                    // Mixed int/float comparisons
                    (Opcode::PushInt(a), Opcode::PushFloat(b), Opcode::Lt) =>
                        Some(Opcode::PushBool((*a as f64) < *b)),
                    (Opcode::PushFloat(a), Opcode::PushInt(b), Opcode::Lt) =>
                        Some(Opcode::PushBool(*a < (*b as f64))),
                    (Opcode::PushInt(a), Opcode::PushFloat(b), Opcode::Gt) =>
                        Some(Opcode::PushBool((*a as f64) > *b)),
                    (Opcode::PushFloat(a), Opcode::PushInt(b), Opcode::Gt) =>
                        Some(Opcode::PushBool(*a > (*b as f64))),
                    (Opcode::PushInt(a), Opcode::PushFloat(b), Opcode::Lte) =>
                        Some(Opcode::PushBool((*a as f64) <= *b)),
                    (Opcode::PushFloat(a), Opcode::PushInt(b), Opcode::Lte) =>
                        Some(Opcode::PushBool(*a <= (*b as f64))),
                    (Opcode::PushInt(a), Opcode::PushFloat(b), Opcode::Gte) =>
                        Some(Opcode::PushBool((*a as f64) >= *b)),
                    (Opcode::PushFloat(a), Opcode::PushInt(b), Opcode::Gte) =>
                        Some(Opcode::PushBool(*a >= (*b as f64))),
                    // Float comparisons (parity with int)
                    (Opcode::PushFloat(a), Opcode::PushFloat(b), Opcode::Lt) =>
                        Some(Opcode::PushBool(a < b)),
                    (Opcode::PushFloat(a), Opcode::PushFloat(b), Opcode::Lte) =>
                        Some(Opcode::PushBool(a <= b)),
                    (Opcode::PushFloat(a), Opcode::PushFloat(b), Opcode::Gt) =>
                        Some(Opcode::PushBool(a > b)),
                    (Opcode::PushFloat(a), Opcode::PushFloat(b), Opcode::Gte) =>
                        Some(Opcode::PushBool(a >= b)),
                    (Opcode::PushInt(a), Opcode::PushInt(b), Opcode::Eq) =>
                        Some(Opcode::PushBool(a == b)),
                    (Opcode::PushInt(a), Opcode::PushInt(b), Opcode::Neq) =>
                        Some(Opcode::PushBool(a != b)),
                    (Opcode::PushInt(a), Opcode::PushInt(b), Opcode::Lt) =>
                        Some(Opcode::PushBool(a < b)),
                    (Opcode::PushInt(a), Opcode::PushInt(b), Opcode::Lte) =>
                        Some(Opcode::PushBool(a <= b)),
                    (Opcode::PushInt(a), Opcode::PushInt(b), Opcode::Gt) =>
                        Some(Opcode::PushBool(a > b)),
                    (Opcode::PushInt(a), Opcode::PushInt(b), Opcode::Gte) =>
                        Some(Opcode::PushBool(a >= b)),
                    (Opcode::PushStr(a), Opcode::PushStr(b), Opcode::Eq) =>
                        Some(Opcode::PushBool(a == b)),
                    (Opcode::PushStr(a), Opcode::PushStr(b), Opcode::Neq) =>
                        Some(Opcode::PushBool(a != b)),
                    (Opcode::PushStr(a), Opcode::PushStr(b), Opcode::Lt) =>
                        Some(Opcode::PushBool(a < b)),
                    (Opcode::PushStr(a), Opcode::PushStr(b), Opcode::Lte) =>
                        Some(Opcode::PushBool(a <= b)),
                    (Opcode::PushStr(a), Opcode::PushStr(b), Opcode::Gt) =>
                        Some(Opcode::PushBool(a > b)),
                    (Opcode::PushStr(a), Opcode::PushStr(b), Opcode::Gte) =>
                        Some(Opcode::PushBool(a >= b)),
                    (Opcode::PushStr(a), Opcode::PushStr(b), Opcode::Add) =>
                        Some(Opcode::PushStr(Arc::<str>::from(format!("{}{}", a, b)))),
                    (Opcode::PushBool(a), Opcode::PushBool(b), Opcode::Eq) =>
                        Some(Opcode::PushBool(a == b)),
                    _ => None,
                };
                if let Some(folded) = folded {
                    out.push(folded);
                    i += 3;
                    continue;
                }
            }
            out.push(ops[i].clone());
            i += 1;
        }
        out
    }

    // ── Main emit ─────────────────────────────────────────────────────────────

    fn emit(expr: &Expr, ctx: &VarCtx) -> Vec<Opcode> {
        let mut ops = Vec::new();
        Self::emit_into(expr, ctx, &mut ops);
        ops
    }

    fn emit_into(expr: &Expr, ctx: &VarCtx, ops: &mut Vec<Opcode>) {
        match expr {
            Expr::Null    => ops.push(Opcode::PushNull),
            Expr::Bool(b) => ops.push(Opcode::PushBool(*b)),
            Expr::Int(n)  => ops.push(Opcode::PushInt(*n)),
            Expr::Float(f)=> ops.push(Opcode::PushFloat(*f)),
            Expr::Str(s)  => ops.push(Opcode::PushStr(Arc::from(s.as_str()))),
            Expr::Root    => ops.push(Opcode::PushRoot),
            Expr::Current => ops.push(Opcode::PushCurrent),

            Expr::FString(parts) => {
                let compiled: Vec<CompiledFSPart> = parts.iter().map(|p| match p {
                    FStringPart::Lit(s) => CompiledFSPart::Lit(Arc::from(s.as_str())),
                    FStringPart::Interp { expr, fmt } => CompiledFSPart::Interp {
                        prog: Arc::new(Self::compile_sub(expr, ctx)),
                        fmt: fmt.clone(),
                    },
                }).collect();
                ops.push(Opcode::FString(compiled.into()));
            }

            Expr::Ident(name) => ops.push(Opcode::LoadIdent(Arc::from(name.as_str()))),

            Expr::Chain(base, steps) => {
                Self::emit_into(base, ctx, ops);
                for step in steps {
                    Self::emit_step(step, ctx, ops);
                }
            }

            Expr::UnaryNeg(e) => {
                Self::emit_into(e, ctx, ops);
                ops.push(Opcode::Neg);
            }
            Expr::Not(e) => {
                Self::emit_into(e, ctx, ops);
                ops.push(Opcode::Not);
            }

            Expr::BinOp(l, op, r) => Self::emit_binop(l, *op, r, ctx, ops),

            Expr::Coalesce(lhs, rhs) => {
                Self::emit_into(lhs, ctx, ops);
                let rhs_prog = Arc::new(Self::compile_sub(rhs, ctx));
                ops.push(Opcode::CoalesceOp(rhs_prog));
            }

            Expr::Kind { expr, ty, negate } => {
                Self::emit_into(expr, ctx, ops);
                ops.push(Opcode::KindCheck { ty: *ty, negate: *negate });
            }

            Expr::Object(fields) => {
                let entries: Vec<CompiledObjEntry> = fields.iter().map(|f| match f {
                    ObjField::Short(name) =>
                        CompiledObjEntry::Short {
                            name: Arc::from(name.as_str()),
                            ic: Arc::new(AtomicU64::new(0)),
                        },
                    ObjField::Kv { key, val, optional, cond }
                        if cond.is_none()
                            && Self::try_kv_path_steps(val).is_some()
                    => {
                        let steps: Vec<KvStep> = Self::try_kv_path_steps(val).unwrap();
                        let n = steps.len();
                        let mut ics_vec: Vec<AtomicU64> = Vec::with_capacity(n);
                        for _ in 0..n { ics_vec.push(AtomicU64::new(0)); }
                        CompiledObjEntry::KvPath {
                            key: Arc::from(key.as_str()),
                            steps: steps.into(),
                            optional: *optional,
                            ics: ics_vec.into(),
                        }
                    }
                    ObjField::Kv { key, val, optional, cond } =>
                        CompiledObjEntry::Kv {
                            key: Arc::from(key.as_str()),
                            prog: Arc::new(Self::compile_sub(val, ctx)),
                            optional: *optional,
                            cond: cond.as_ref().map(|c| Arc::new(Self::compile_sub(c, ctx))),
                        },
                    ObjField::Dynamic { key, val } =>
                        CompiledObjEntry::Dynamic {
                            key: Arc::new(Self::compile_sub(key, ctx)),
                            val: Arc::new(Self::compile_sub(val, ctx)),
                        },
                    ObjField::Spread(e) =>
                        CompiledObjEntry::Spread(Arc::new(Self::compile_sub(e, ctx))),
                    ObjField::SpreadDeep(e) =>
                        CompiledObjEntry::SpreadDeep(Arc::new(Self::compile_sub(e, ctx))),
                }).collect();
                ops.push(Opcode::MakeObj(entries.into()));
            }

            Expr::Array(elems) => {
                // Compile each elem as a sub-program.
                // Spread elems are handled by a special marker.
                let _progs: Vec<Arc<Program>> = elems.iter().map(|e| match e {
                    ArrayElem::Expr(ex)   => Arc::new(Self::compile_sub(ex, ctx)),
                    // Spread: compile the inner expr with a spread marker opcode
                    ArrayElem::Spread(ex) => {
                        let mut sub = Self::emit(ex, ctx);
                        // Prepend a sentinel to distinguish spread from normal
                        sub.insert(0, Opcode::PushNull); // placeholder
                        // Actually, encode spread differently via a wrapper
                        Arc::new(Self::compile_array_spread(ex, ctx))
                    }
                }).collect();
                // Simpler: build a mixed elem list
                let progs = elems.iter().map(|e| match e {
                    ArrayElem::Expr(ex) => {
                        Arc::new(Self::compile_sub(ex, ctx))
                    }
                    ArrayElem::Spread(ex) => {
                        Arc::new(Self::compile_sub_spread(ex, ctx))
                    }
                }).collect::<Vec<_>>();
                ops.push(Opcode::MakeArr(progs.into()));
            }

            Expr::Pipeline { base, steps } => {
                Self::emit_pipeline(base, steps, ctx, ops);
            }

            Expr::ListComp { expr, vars, iter, cond } => {
                let inner_ctx = ctx.with_vars(vars);
                ops.push(Opcode::ListComp(Arc::new(CompSpec {
                    expr: Arc::new(Self::compile_sub(expr, &inner_ctx)),
                    vars: vars.iter().map(|v| Arc::from(v.as_str())).collect::<Vec<_>>().into(),
                    iter: Arc::new(Self::compile_sub(iter, ctx)),
                    cond: cond.as_ref().map(|c| Arc::new(Self::compile_sub(c, &inner_ctx))),
                })));
            }

            Expr::DictComp { key, val, vars, iter, cond } => {
                let inner_ctx = ctx.with_vars(vars);
                ops.push(Opcode::DictComp(Arc::new(DictCompSpec {
                    key:  Arc::new(Self::compile_sub(key, &inner_ctx)),
                    val:  Arc::new(Self::compile_sub(val, &inner_ctx)),
                    vars: vars.iter().map(|v| Arc::from(v.as_str())).collect::<Vec<_>>().into(),
                    iter: Arc::new(Self::compile_sub(iter, ctx)),
                    cond: cond.as_ref().map(|c| Arc::new(Self::compile_sub(c, &inner_ctx))),
                })));
            }

            Expr::SetComp { expr, vars, iter, cond } |
            Expr::GenComp { expr, vars, iter, cond } => {
                let inner_ctx = ctx.with_vars(vars);
                ops.push(Opcode::SetComp(Arc::new(CompSpec {
                    expr: Arc::new(Self::compile_sub(expr, &inner_ctx)),
                    vars: vars.iter().map(|v| Arc::from(v.as_str())).collect::<Vec<_>>().into(),
                    iter: Arc::new(Self::compile_sub(iter, ctx)),
                    cond: cond.as_ref().map(|c| Arc::new(Self::compile_sub(c, &inner_ctx))),
                })));
            }

            Expr::Lambda { .. } => {
                // Lambdas as standalone values are errors; they only appear as args
                ops.push(Opcode::PushNull);
            }

            Expr::Let { name, init, body } => {
                // Dead-let: if body never references `name` and init is pure,
                // drop the binding entirely and emit body only.
                if super::analysis::expr_is_pure(init)
                    && !super::analysis::expr_uses_ident(body, name) {
                    Self::emit_into(body, ctx, ops);
                } else {
                    Self::emit_into(init, ctx, ops);
                    let body_ctx = ctx.with_var(name);
                    let body_prog = Arc::new(Self::compile_sub(body, &body_ctx));
                    ops.push(Opcode::LetExpr { name: Arc::from(name.as_str()), body: body_prog });
                }
            }

            Expr::IfElse { cond, then_, else_ } => {
                // Compile-time fold when cond is a literal bool.
                match cond.as_ref() {
                    Expr::Bool(true)  => { Self::emit_into(then_, ctx, ops); }
                    Expr::Bool(false) => { Self::emit_into(else_, ctx, ops); }
                    _ => {
                        Self::emit_into(cond, ctx, ops);
                        let then_prog = Arc::new(Self::compile_sub(then_, ctx));
                        let else_prog = Arc::new(Self::compile_sub(else_, ctx));
                        ops.push(Opcode::IfElse { then_: then_prog, else_: else_prog });
                    }
                }
            }

            Expr::GlobalCall { name, args } => {
                // Compile as a sequence of sub-progs + a special dispatch
                let sub_progs: Vec<Arc<Program>> = args.iter().map(|a| match a {
                    Arg::Pos(e) | Arg::Named(_, e) => Arc::new(Self::compile_sub(e, ctx)),
                }).collect();
                let call = Arc::new(CompiledCall {
                    method:    BuiltinMethod::Unknown,
                    name:      Arc::from(name.as_str()),
                    sub_progs: sub_progs.into(),
                    orig_args: args.iter().cloned().collect::<Vec<_>>().into(),
                });
                ops.push(Opcode::PushRoot); // global calls need root pushed first
                ops.push(Opcode::CallMethod(call));
            }

            Expr::Cast { expr, ty } => {
                Self::emit_into(expr, ctx, ops);
                ops.push(Opcode::CastOp(*ty));
            }

            Expr::Patch { .. } => {
                // Patch block: structural transform with COW writes, conditional
                // leaves, DELETE sentinel.  Emit opaque opcode; the VM delegates
                // to the tree-walker at runtime (patch is rare enough that the
                // opcode compile pays no dividend here).
                ops.push(Opcode::PatchEval(Arc::new(expr.clone())));
            }

            Expr::DeleteMark => {
                // DELETE outside a patch-field value is a static error; the
                // tree-walker raises it at runtime, so emit a sentinel that
                // triggers the same path.
                ops.push(Opcode::PatchEval(Arc::new(Expr::DeleteMark)));
            }
        }
    }

    fn emit_step(step: &Step, ctx: &VarCtx, ops: &mut Vec<Opcode>) {
        match step {
            Step::Field(name)    => ops.push(Opcode::GetField(Arc::from(name.as_str()))),
            Step::OptField(name) => ops.push(Opcode::OptField(Arc::from(name.as_str()))),
            Step::Descendant(n)  => ops.push(Opcode::Descendant(Arc::from(n.as_str()))),
            Step::DescendAll     => ops.push(Opcode::DescendAll),
            Step::Index(i)       => ops.push(Opcode::GetIndex(*i)),
            Step::DynIndex(e)    => ops.push(Opcode::DynIndex(Arc::new(Self::compile_sub(e, ctx)))),
            Step::Slice(a, b)    => ops.push(Opcode::GetSlice(*a, *b)),
            Step::Method(name, method_args) => {
                let call = Self::compile_call(name, method_args, ctx);
                ops.push(Opcode::CallMethod(Arc::new(call)));
            }
            Step::OptMethod(name, method_args) => {
                let call = Self::compile_call(name, method_args, ctx);
                ops.push(Opcode::CallOptMethod(Arc::new(call)));
            }
            Step::InlineFilter(pred) => {
                ops.push(Opcode::InlineFilter(Arc::new(Self::compile_sub(pred, ctx))));
            }
            Step::Quantifier(k) => ops.push(Opcode::Quantifier(*k)),
        }
    }

    fn compile_call(name: &str, args: &[Arg], ctx: &VarCtx) -> CompiledCall {
        let method = BuiltinMethod::from_name(name);
        let sub_progs: Vec<Arc<Program>> = args.iter().map(|a| match a {
            Arg::Pos(e) | Arg::Named(_, e) => Arc::new(Self::compile_lambda_or_expr(e, ctx)),
        }).collect();
        CompiledCall {
            method,
            name: Arc::from(name),
            sub_progs: sub_progs.into(),
            orig_args: args.iter().cloned().collect::<Vec<_>>().into(),
        }
    }

    /// Compile an argument expression; for lambdas, the lambda param becomes a
    /// known var in the inner context so `Ident(param)` emits `LoadIdent`.
    fn compile_lambda_or_expr(expr: &Expr, ctx: &VarCtx) -> Program {
        match expr {
            Expr::Lambda { params, body } => {
                let inner = ctx.with_vars(params);
                Self::compile_sub(body, &inner)
            }
            other => Self::compile_sub(other, ctx),
        }
    }

    fn emit_binop(l: &Expr, op: BinOp, r: &Expr, ctx: &VarCtx, ops: &mut Vec<Opcode>) {
        match op {
            BinOp::And => {
                Self::emit_into(l, ctx, ops);
                let rhs_prog = Arc::new(Self::compile_sub(r, ctx));
                ops.push(Opcode::AndOp(rhs_prog));
            }
            BinOp::Or => {
                Self::emit_into(l, ctx, ops);
                let rhs_prog = Arc::new(Self::compile_sub(r, ctx));
                ops.push(Opcode::OrOp(rhs_prog));
            }
            BinOp::Add => { Self::emit_into(l, ctx, ops); Self::emit_into(r, ctx, ops); ops.push(Opcode::Add); }
            BinOp::Sub => { Self::emit_into(l, ctx, ops); Self::emit_into(r, ctx, ops); ops.push(Opcode::Sub); }
            BinOp::Mul => { Self::emit_into(l, ctx, ops); Self::emit_into(r, ctx, ops); ops.push(Opcode::Mul); }
            BinOp::Div => { Self::emit_into(l, ctx, ops); Self::emit_into(r, ctx, ops); ops.push(Opcode::Div); }
            BinOp::Mod => { Self::emit_into(l, ctx, ops); Self::emit_into(r, ctx, ops); ops.push(Opcode::Mod); }
            BinOp::Eq  => { Self::emit_into(l, ctx, ops); Self::emit_into(r, ctx, ops); ops.push(Opcode::Eq); }
            BinOp::Neq => { Self::emit_into(l, ctx, ops); Self::emit_into(r, ctx, ops); ops.push(Opcode::Neq); }
            BinOp::Lt  => { Self::emit_into(l, ctx, ops); Self::emit_into(r, ctx, ops); ops.push(Opcode::Lt); }
            BinOp::Lte => { Self::emit_into(l, ctx, ops); Self::emit_into(r, ctx, ops); ops.push(Opcode::Lte); }
            BinOp::Gt  => { Self::emit_into(l, ctx, ops); Self::emit_into(r, ctx, ops); ops.push(Opcode::Gt); }
            BinOp::Gte => { Self::emit_into(l, ctx, ops); Self::emit_into(r, ctx, ops); ops.push(Opcode::Gte); }
            BinOp::Fuzzy => { Self::emit_into(l, ctx, ops); Self::emit_into(r, ctx, ops); ops.push(Opcode::Fuzzy); }
        }
    }

    fn emit_pipeline(base: &Expr, steps: &[PipeStep], ctx: &VarCtx, ops: &mut Vec<Opcode>) {
        Self::emit_into(base, ctx, ops);
        let mut cur_ctx = ctx.clone();
        for step in steps {
            match step {
                PipeStep::Forward(rhs) => {
                    Self::emit_pipe_forward(rhs, &cur_ctx, ops);
                }
                PipeStep::Bind(target) => {
                    Self::emit_bind(target, &mut cur_ctx, ops);
                }
            }
        }
    }

    fn emit_pipe_forward(rhs: &Expr, ctx: &VarCtx, ops: &mut Vec<Opcode>) {
        match rhs {
            Expr::Ident(name) if !ctx.has(name) => {
                // No-arg method call on TOS
                let call = CompiledCall {
                    method:    BuiltinMethod::from_name(name),
                    name:      Arc::from(name.as_str()),
                    sub_progs: Arc::from(&[] as &[Arc<Program>]),
                    orig_args: Arc::from(&[] as &[Arg]),
                };
                ops.push(Opcode::CallMethod(Arc::new(call)));
            }
            Expr::Chain(base, steps) if !steps.is_empty() => {
                if let Expr::Ident(name) = base.as_ref() {
                    if !ctx.has(name) {
                        // method(args...) — base is method, steps are chained
                        let call = CompiledCall {
                            method:    BuiltinMethod::from_name(name),
                            name:      Arc::from(name.as_str()),
                            sub_progs: Arc::from(&[] as &[Arc<Program>]),
                            orig_args: Arc::from(&[] as &[Arg]),
                        };
                        ops.push(Opcode::CallMethod(Arc::new(call)));
                        for step in steps { Self::emit_step(step, ctx, ops); }
                        return;
                    }
                }
                ops.push(Opcode::SetCurrent);
                Self::emit_into(rhs, ctx, ops);
            }
            _ => {
                // Arbitrary expression; set current to pipe input, then eval
                ops.push(Opcode::SetCurrent);
                Self::emit_into(rhs, ctx, ops);
            }
        }
    }

    fn emit_bind(target: &BindTarget, ctx: &mut VarCtx, ops: &mut Vec<Opcode>) {
        match target {
            BindTarget::Name(name) => {
                ops.push(Opcode::BindVar(Arc::from(name.as_str())));
                *ctx = ctx.with_var(name);
            }
            BindTarget::Obj { fields, rest } => {
                let spec = BindObjSpec {
                    fields: fields.iter().map(|f| Arc::from(f.as_str())).collect::<Vec<_>>().into(),
                    rest: rest.as_ref().map(|r| Arc::from(r.as_str())),
                };
                ops.push(Opcode::BindObjDestructure(Arc::new(spec)));
                for f in fields { *ctx = ctx.with_var(f); }
                if let Some(r) = rest { *ctx = ctx.with_var(r); }
            }
            BindTarget::Arr(names) => {
                let ns: Vec<Arc<str>> = names.iter().map(|n| Arc::from(n.as_str())).collect();
                ops.push(Opcode::BindArrDestructure(ns.into()));
                for n in names { *ctx = ctx.with_var(n); }
            }
        }
    }

    fn compile_sub(expr: &Expr, ctx: &VarCtx) -> Program {
        let ops = Self::optimize(Self::emit(expr, ctx));
        Program::new(ops, "<sub>")
    }

    /// Classify an object-value expression as a pure path on `current`:
    /// a chain of `Field(name)` / `Index(i)` steps rooted at `Expr::Current`.
    /// Returns `None` for anything else — the caller falls back to full
    /// sub-program compilation.
    fn try_kv_path_steps(expr: &Expr) -> Option<Vec<KvStep>> {
        use super::ast::Step;
        let (base, steps) = match expr {
            Expr::Chain(b, s) => (&**b, s.as_slice()),
            _ => return None,
        };
        if !matches!(base, Expr::Current) { return None; }
        if steps.is_empty() { return None; }
        let mut out = Vec::with_capacity(steps.len());
        for s in steps {
            match s {
                Step::Field(name) => out.push(KvStep::Field(Arc::from(name.as_str()))),
                Step::Index(i)    => out.push(KvStep::Index(*i)),
                _ => return None,
            }
        }
        Some(out)
    }

    fn compile_array_spread(_expr: &Expr, _ctx: &VarCtx) -> Program {
        // Not reached — handled in MakeArr execution
        Program::new(vec![], "<spread>")
    }

    /// Compile a spread array element — wrapped with a special marker.
    fn compile_sub_spread(expr: &Expr, ctx: &VarCtx) -> Program {
        let mut ops = Self::emit(expr, ctx);
        // Prefix with a sentinel bool to mark this as a spread
        ops.insert(0, Opcode::PushBool(true));
        // Append a sentinel for reading: bool(true) + actual val
        // Actually use a dedicated approach: GetSlice-like marker
        // For simplicity, just compile the expr normally;
        // MakeArr handles spread by checking if the result is an array
        // when the corresponding ArrayElem is Spread.
        // Re-do: just compile normally, MakeArr knows which slots are spreads.
        Self::compile_sub(expr, ctx) // caller has separate spread tracking
    }
}

// ── Path cache ────────────────────────────────────────────────────────────────
//
// Key: (doc_hash, json_pointer) → Val
//
// Doc-scoped (no program_id): any program resolving the same path on the same
// document gets a hit.  Intermediate nodes are cached so a prefix of a longer
// path can be reused without re-traversal.

struct PathCache {
    /// doc_hash → (pointer_string → Val)
    docs:     HashMap<u64, HashMap<Arc<str>, Val>>,
    /// FIFO eviction order
    order:    VecDeque<(u64, Arc<str>)>,
    capacity: usize,
}

impl PathCache {
    fn new(cap: usize) -> Self {
        Self {
            docs:     HashMap::new(),
            order:    VecDeque::with_capacity(cap),
            capacity: cap,
        }
    }

    /// O(1) immutable lookup — returns cloned Val (Val::clone is O(1)).
    #[inline]
    fn get(&self, doc_hash: u64, ptr: &str) -> Option<Val> {
        self.docs.get(&doc_hash)?.get(ptr).cloned()
    }

    fn insert(&mut self, doc_hash: u64, ptr: Arc<str>, val: Val) {
        if self.order.len() >= self.capacity {
            if let Some((old_hash, old_ptr)) = self.order.pop_front() {
                if let Some(inner) = self.docs.get_mut(&old_hash) {
                    inner.remove(old_ptr.as_ref());
                    if inner.is_empty() { self.docs.remove(&old_hash); }
                }
            }
        }
        self.order.push_back((doc_hash, ptr.clone()));
        self.docs.entry(doc_hash).or_insert_with(HashMap::new).insert(ptr, val);
    }

    fn len(&self) -> usize { self.order.len() }
}

// ── VM ────────────────────────────────────────────────────────────────────────

/// High-performance v2 virtual machine.
///
/// Maintains:
/// - **Compile cache** — expression string → `Program` (parse + compile once).
/// - **Path cache** — `(doc_hash, json_pointer)` → `Val`; doc-scoped so any
///   program navigating the same path on the same document shares cached nodes.
///   Intermediate nodes are populated as a side-effect of every traversal,
///   enabling prefix reuse without re-traversal.
///
/// One VM per thread; wrap in `Mutex` for shared use.
/// Toggle each optimiser pass independently.  Default enables every
/// pass.  Disabling a pass invalidates the compile cache for the next
/// compilation by changing `hash()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PassConfig {
    pub root_chain:      bool,
    pub field_chain:     bool,
    pub filter_count:    bool,
    pub filter_fusion:   bool,
    pub find_quantifier: bool,
    pub strength_reduce: bool,
    pub redundant_ops:   bool,
    pub kind_check_fold: bool,
    pub method_const:    bool,
    pub const_fold:      bool,
    pub nullness:        bool,
    pub equi_join:       bool,
    pub reorder_and:     bool,
    pub dedup_subprogs:  bool,
}

impl Default for PassConfig {
    fn default() -> Self {
        Self {
            root_chain: true, field_chain: true, filter_count: true, filter_fusion: true,
            find_quantifier: true, strength_reduce: true, redundant_ops: true,
            kind_check_fold: true, method_const: true, const_fold: true,
            nullness: true, equi_join: true,
            reorder_and: true, dedup_subprogs: true,
        }
    }
}

impl PassConfig {
    /// Disable every pass — emit raw opcodes.
    pub fn none() -> Self {
        Self {
            root_chain: false, field_chain: false, filter_count: false, filter_fusion: false,
            find_quantifier: false, strength_reduce: false, redundant_ops: false,
            kind_check_fold: false, method_const: false, const_fold: false,
            nullness: false, equi_join: false,
            reorder_and: false, dedup_subprogs: false,
        }
    }

    pub fn hash(&self) -> u64 {
        let mut bits: u64 = 0;
        for (i, b) in [self.root_chain, self.field_chain, self.filter_count, self.filter_fusion,
                       self.find_quantifier, self.strength_reduce, self.redundant_ops,
                       self.kind_check_fold, self.method_const, self.const_fold,
                       self.nullness, self.equi_join,
                       self.reorder_and, self.dedup_subprogs].iter().enumerate() {
            if *b { bits |= 1u64 << i; }
        }
        bits
    }
}

pub struct VM {
    registry:      Arc<MethodRegistry>,
    /// Cache key = (pass_config_hash, expr_string).  Changing `config`
    /// invalidates prior entries automatically via key divergence.
    compile_cache: HashMap<(u64, String), Arc<Program>>,
    /// LRU ordering for `compile_cache`; front = least recently used.
    /// Entries are moved to back on hit; oldest evicted when over cap.
    compile_lru:   std::collections::VecDeque<(u64, String)>,
    compile_cap:   usize,
    path_cache:    PathCache,
    /// Per-exec RootChain resolution cache.  Key = raw address of the
    /// `chain` Arc slice; value = resolved Val.  Cleared on every top-level
    /// `execute()` call so stale entries never outlive the doc they
    /// reference.  Avoids rebuilding the `/a/b/c` pointer string and
    /// consulting `path_cache` when the same RootChain opcode fires
    /// repeatedly inside a loop.
    root_chain_cache: HashMap<usize, Val>,
    /// Hash of the document currently being executed — set once by `execute()`,
    /// reused by all recursive `exec()` calls within the same top-level call.
    doc_hash:      u64,
    /// Cache of root Arc pointer → structural hash.  Lets repeated calls
    /// against the same cached root (e.g. via `Jetro::collect`) skip the
    /// O(doc) structural walk entirely.  Keyed on the inner Arc ptr of
    /// the root `Val::Obj`/`Val::Arr`.
    root_hash_cache: Option<(usize, u64)>,
    /// Optimiser pass toggles.  Default: all on.
    config:        PassConfig,
}

impl Default for VM {
    fn default() -> Self { Self::new() }
}

impl VM {
    pub fn new() -> Self { Self::with_capacity(512, 4096) }

    pub fn with_capacity(compile_cap: usize, path_cap: usize) -> Self {
        Self {
            registry:      Arc::new(MethodRegistry::new()),
            compile_cache: HashMap::with_capacity(compile_cap),
            compile_lru:   std::collections::VecDeque::with_capacity(compile_cap),
            compile_cap,
            path_cache:    PathCache::new(path_cap),
            root_chain_cache: HashMap::new(),
            doc_hash:      0,
            root_hash_cache: None,
            config:        PassConfig::default(),
        }
    }

    /// Build a VM that shares an existing method registry.
    pub fn with_registry(registry: Arc<MethodRegistry>) -> Self {
        let mut vm = Self::new();
        vm.registry = registry;
        vm
    }

    /// Register a method already wrapped in `Arc`.
    pub fn register_arc(&mut self, name: &str, method: Arc<dyn crate::eval::methods::Method>) {
        Arc::make_mut(&mut self.registry).register_arc(name, method);
    }

    /// Replace the pass configuration.  The compile cache is not purged,
    /// but future lookups key off the new config hash so old entries
    /// are effectively invalidated for the new regime.
    pub fn set_pass_config(&mut self, config: PassConfig) { self.config = config; }

    pub fn pass_config(&self) -> PassConfig { self.config }

    /// Register a custom method (callable via `.method_name(...)` in expressions).
    pub fn register(&mut self, name: impl Into<String>, method: impl super::eval::methods::Method + 'static) {
        Arc::make_mut(&mut self.registry).register(name, method);
    }

    // ── Public entry-points ───────────────────────────────────────────────────

    /// Parse, compile (cached), and execute `expr` against `doc`.
    pub fn run_str(&mut self, expr: &str, doc: &serde_json::Value) -> Result<serde_json::Value, EvalError> {
        let prog = self.get_or_compile(expr)?;
        self.execute(&prog, doc)
    }

    /// Parse, compile, and execute with raw JSON source bytes retained so
    /// that SIMD byte-scan can short-circuit `Opcode::Descendant` at the
    /// document root.
    pub fn run_str_with_raw(
        &mut self,
        expr: &str,
        doc: &serde_json::Value,
        raw_bytes: Arc<[u8]>,
    ) -> Result<serde_json::Value, EvalError> {
        let prog = self.get_or_compile(expr)?;
        self.execute_with_raw(&prog, doc, raw_bytes)
    }

    /// Execute a pre-compiled `Program` against `doc`.
    pub fn execute(&mut self, program: &Program, doc: &serde_json::Value) -> Result<serde_json::Value, EvalError> {
        let root = Val::from(doc);
        self.doc_hash = self.compute_or_cache_root_hash(&root);
        // Per-exec RootChain cache: entries key off raw Arc addresses that
        // must not outlive this document.  Clear before every run.
        self.root_chain_cache.clear();
        let env = self.make_env(root);
        let result = self.exec(program, &env)?;
        Ok(result.into())
    }

    /// Reuse the cached structural hash if `root`'s inner Arc pointer
    /// matches a prior call; otherwise walk the tree once and cache.
    /// Primitive roots bypass the cache (hashing is already O(1)).
    fn compute_or_cache_root_hash(&mut self, root: &Val) -> u64 {
        let ptr: Option<usize> = match root {
            Val::Obj(m)      => Some(Arc::as_ptr(m) as *const () as usize),
            Val::Arr(a)      => Some(Arc::as_ptr(a) as *const () as usize),
            Val::IntVec(a)   => Some(Arc::as_ptr(a) as *const () as usize),
            Val::FloatVec(a) => Some(Arc::as_ptr(a) as *const () as usize),
            _ => None,
        };
        if let Some(p) = ptr {
            if let Some((cp, h)) = self.root_hash_cache {
                if cp == p { return h; }
            }
            let h = hash_val_structure(root);
            self.root_hash_cache = Some((p, h));
            h
        } else {
            hash_val_structure(root)
        }
    }

    /// Execute with raw JSON source bytes retained on the environment so
    /// that descendant opcodes at document root can take the SIMD byte-scan
    /// fast path instead of walking the tree.
    pub fn execute_with_raw(
        &mut self,
        program: &Program,
        doc: &serde_json::Value,
        raw_bytes: Arc<[u8]>,
    ) -> Result<serde_json::Value, EvalError> {
        let root = Val::from(doc);
        self.execute_val_with_raw(program, root, raw_bytes)
    }

    /// Execute against a pre-built `Val` root (skips the `Val::from(&Value)`
    /// conversion on every call).  With raw bytes, the `doc_hash` pointer-
    /// cache path is also skipped — byte-scan handles descendants directly
    /// and `RootChain` reads are O(chain length) against the already-built
    /// tree.
    pub fn execute_val_with_raw(
        &mut self,
        program: &Program,
        root: Val,
        raw_bytes: Arc<[u8]>,
    ) -> Result<serde_json::Value, EvalError> {
        // doc_hash seeds the path cache; on the scan fast path we bypass the
        // cache entirely, so skip the O(doc) structural hash walk.
        self.doc_hash = 0;
        self.root_chain_cache.clear();
        let env = Env::new_with_raw(root, Arc::clone(&self.registry), raw_bytes);
        let result = self.exec(program, &env)?;
        Ok(result.into())
    }

    /// Execute against a pre-built `Val` root without raw bytes.  Skips the
    /// `Val::from` conversion only — path cache and doc hash still behave
    /// as in `execute()`.
    pub fn execute_val(
        &mut self,
        program: &Program,
        root: Val,
    ) -> Result<serde_json::Value, EvalError> {
        Ok(self.execute_val_raw(program, root)?.into())
    }

    /// Execute against a pre-built `Val` root and return the raw `Val` —
    /// no `serde_json::Value` materialisation.  Use when the caller will
    /// consume results structurally (further queries, custom walk) and
    /// wants to skip a potentially expensive `Val → Value` conversion.
    pub fn execute_val_raw(
        &mut self,
        program: &Program,
        root: Val,
    ) -> Result<Val, EvalError> {
        self.doc_hash = self.compute_or_cache_root_hash(&root);
        self.root_chain_cache.clear();
        let env = self.make_env(root);
        self.exec(program, &env)
    }

    /// Execute a compiled program against a document, first specialising
    /// against the given shape (turns `OptField` → `GetField` where safe,
    /// folds `KindCheck` where type is known, etc.).
    pub fn execute_with_schema(
        &mut self,
        program: &Program,
        doc: &serde_json::Value,
        shape: &super::schema::Shape,
    ) -> Result<serde_json::Value, EvalError> {
        let specialized = super::schema::specialize(program, shape);
        self.execute(&specialized, doc)
    }

    /// Execute a program; infer the shape from the document itself.  Costs
    /// an O(doc) shape walk before execution; useful when the same
    /// compiled program is reused across many docs with similar shapes.
    pub fn execute_with_inferred_schema(
        &mut self,
        program: &Program,
        doc: &serde_json::Value,
    ) -> Result<serde_json::Value, EvalError> {
        let shape = super::schema::Shape::of(doc);
        self.execute_with_schema(program, doc, &shape)
    }

    /// Get or compile an expression string (compile cache).
    /// Cache key is (pass_config_hash, expr) so that different pass
    /// configurations yield different compiled programs.
    pub fn get_or_compile(&mut self, expr: &str) -> Result<Arc<Program>, EvalError> {
        let key = (self.config.hash(), expr.to_string());
        if let Some(p) = self.compile_cache.get(&key) {
            let arc = Arc::clone(p);
            self.touch_lru(&key);
            return Ok(arc);
        }
        let prog = Compiler::compile_str_with_config(expr, self.config)?;
        let arc = Arc::new(prog);
        self.insert_compile(key, Arc::clone(&arc));
        Ok(arc)
    }

    /// Move `key` to the back of the LRU queue (most recently used).
    fn touch_lru(&mut self, key: &(u64, String)) {
        if let Some(pos) = self.compile_lru.iter().position(|k| k == key) {
            let k = self.compile_lru.remove(pos).unwrap();
            self.compile_lru.push_back(k);
        }
    }

    /// Insert into compile cache with LRU eviction at `compile_cap`.
    fn insert_compile(&mut self, key: (u64, String), prog: Arc<Program>) {
        while self.compile_cache.len() >= self.compile_cap && self.compile_cap > 0 {
            if let Some(old) = self.compile_lru.pop_front() {
                self.compile_cache.remove(&old);
            } else {
                break;
            }
        }
        self.compile_lru.push_back(key.clone());
        self.compile_cache.insert(key, prog);
    }

    /// Cache statistics: `(compile_entries, path_entries)`.
    pub fn cache_stats(&self) -> (usize, usize) {
        (self.compile_cache.len(), self.path_cache.len())
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    fn make_env(&self, root: Val) -> Env {
        Env::new_with_registry(root, Arc::clone(&self.registry))
    }


    // ── Core execution loop ───────────────────────────────────────────────────

    /// Execute `program` in environment `env`, returning the top-of-stack value.
    pub fn exec(&mut self, program: &Program, env: &Env) -> Result<Val, EvalError> {
        let mut stack: SmallVec<[Val; 16]> = SmallVec::new();
        let ops_slice: &[Opcode] = &program.ops;
        let mut skip_ahead: usize = 0;

        for (op_idx, op) in ops_slice.iter().enumerate() {
            if skip_ahead > 0 { skip_ahead -= 1; continue; }
            match op {
                // ── Literals ──────────────────────────────────────────────────
                Opcode::PushNull        => stack.push(Val::Null),
                Opcode::PushBool(b)     => stack.push(Val::Bool(*b)),
                Opcode::PushInt(n)      => stack.push(Val::Int(*n)),
                Opcode::PushFloat(f)    => stack.push(Val::Float(*f)),
                Opcode::PushStr(s)      => stack.push(Val::Str(s.clone())),

                // ── Context ───────────────────────────────────────────────────
                Opcode::PushRoot        => stack.push(env.root.clone()),
                Opcode::PushCurrent     => stack.push(env.current.clone()),

                // ── Navigation ────────────────────────────────────────────────
                Opcode::GetField(k) => {
                    let v = pop!(stack);
                    let out = match &v {
                        Val::Obj(m) => ic_get_field(m, k.as_ref(), &program.ics[op_idx]),
                        _ => Val::Null,
                    };
                    stack.push(out);
                }
                Opcode::FieldChain(chain) => {
                    let mut cur = pop!(stack);
                    for (i, k) in chain.keys.iter().enumerate() {
                        cur = if let Val::Obj(m) = &cur {
                            ic_get_field(m, k.as_ref(), &chain.ics[i])
                        } else {
                            cur.get_field(k.as_ref())
                        };
                    }
                    stack.push(cur);
                }
                Opcode::GetIndex(i) => {
                    let v = pop!(stack);
                    stack.push(v.get_index(*i));
                }
                Opcode::DynIndex(prog) => {
                    let v = pop!(stack);
                    let key = self.exec(prog, env)?;
                    stack.push(match key {
                        Val::Int(i) => v.get_index(i),
                        Val::Str(s) => v.get_field(s.as_ref()),
                        _ => Val::Null,
                    });
                }
                Opcode::GetSlice(from, to) => {
                    let v = pop!(stack);
                    stack.push(exec_slice(v, *from, *to));
                }
                Opcode::OptField(k) => {
                    let v = pop!(stack);
                    let out = match &v {
                        Val::Null => Val::Null,
                        Val::Obj(m) => ic_get_field(m, k.as_ref(), &program.ics[op_idx]),
                        _ => v.get_field(k.as_ref()),
                    };
                    stack.push(out);
                }
                Opcode::Descendant(k) => {
                    let v = pop!(stack);
                    // (D) When descending from root, track pointer paths and
                    // cache each discovered node for future RootChain lookups.
                    let from_root = match (&v, &env.root) {
                        (Val::Obj(a), Val::Obj(b)) => Arc::ptr_eq(a, b),
                        (Val::Arr(a), Val::Arr(b)) => Arc::ptr_eq(a, b),
                        _ => matches!((&v, &env.root), (Val::Null, Val::Null)),
                    };
                    // SIMD fast path: descending from root with raw JSON bytes
                    // retained → byte-scan instead of walking the tree.  Path
                    // cache is skipped on this path (cost/benefit unfavorable
                    // vs avoiding the tree walk entirely).
                    if from_root {
                        if let Some(bytes) = env.raw_bytes.as_ref() {
                            let (val, extra) = byte_chain_exec(
                                bytes, k.as_ref(), &ops_slice[op_idx + 1..]
                            );
                            stack.push(val);
                            skip_ahead = extra;
                            continue;
                        }
                    }
                    // Tree-walker early exit: `$..k.first()` / `$..k!` materialises
                    // only the first self-first DFS hit.  SIMD byte_chain_exec
                    // already covers the raw-bytes case; this catches tree-only
                    // receivers.  Skips pointer-cache population (single-hit
                    // caller doesn't benefit from storing siblings).
                    if let Some(next) = ops_slice.get(op_idx + 1) {
                        if is_first_selector_op(next) {
                            let hit = find_desc_first(&v, k.as_ref()).unwrap_or(Val::Null);
                            stack.push(hit);
                            skip_ahead = 1;
                            continue;
                        }
                    }
                    let mut found = Vec::new();
                    if from_root {
                        let mut prefix = String::new();
                        let mut cached: Vec<(Arc<str>, Val)> = Vec::new();
                        collect_desc_with_paths(&v, k.as_ref(), &mut prefix, &mut found, &mut cached);
                        let doc_hash = self.doc_hash;
                        for (ptr, val) in cached {
                            self.path_cache.insert(doc_hash, ptr, val);
                        }
                    } else {
                        collect_desc(&v, k.as_ref(), &mut found);
                    }
                    stack.push(Val::arr(found));
                }
                Opcode::DescendAll => {
                    let v = pop!(stack);
                    let mut found = Vec::new();
                    collect_all(&v, &mut found);
                    stack.push(Val::arr(found));
                }
                Opcode::InlineFilter(pred) => {
                    let val = pop!(stack);
                    let items = match val {
                        Val::Arr(a) => Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone()),
                        other => vec![other],
                    };
                    let mut out = Vec::with_capacity(items.len());
                    let mut scratch = env.clone();
                    for item in items {
                        let prev = scratch.swap_current(item.clone());
                        let keep = is_truthy(&self.exec(pred, &scratch)?);
                        scratch.restore_current(prev);
                        if keep { out.push(item); }
                    }
                    stack.push(Val::arr(out));
                }
                Opcode::Quantifier(kind) => {
                    let val = pop!(stack);
                    stack.push(match kind {
                        QuantifierKind::First => match val {
                            Val::Arr(a) => a.first().cloned().unwrap_or(Val::Null),
                            other => other,
                        },
                        QuantifierKind::One => match val {
                            Val::Arr(a) if a.len() == 1 => a[0].clone(),
                            Val::Arr(a) => return err!("quantifier !: expected exactly one element, got {}", a.len()),
                            other => other,
                        },
                    });
                }

                // ── Peephole fusions ──────────────────────────────────────────
                Opcode::RootChain(chain) => {
                    // Fast path — same RootChain opcode fired earlier in this
                    // execute() call on this same doc.  Arc pointer identity
                    // is stable; cache is cleared per top-level execute().
                    let key = Arc::as_ptr(chain) as *const () as usize;
                    if let Some(v) = self.root_chain_cache.get(&key) {
                        stack.push(v.clone());
                        continue;
                    }

                    let doc_hash = self.doc_hash;
                    let mut current = env.root.clone();
                    let mut ptr = String::new();
                    let mut resumed_from_cache = false;
                    for k in chain.iter() {
                        ptr.push('/');
                        ptr.push_str(k.as_ref());
                        // Try resuming from a longer cached prefix before
                        // stepping through get_field.
                        if !resumed_from_cache {
                            if let Some(cached) = self.path_cache.get(doc_hash, &ptr) {
                                current = cached;
                                continue;
                            }
                            resumed_from_cache = true;
                        }
                        current = current.get_field(k.as_ref());
                        self.path_cache.insert(doc_hash, Arc::from(ptr.as_str()), current.clone());
                    }

                    self.root_chain_cache.insert(key, current.clone());
                    stack.push(current);
                }
                Opcode::FilterCount(pred) => {
                    let recv = pop!(stack);
                    let n = match &recv {
                        Val::Arr(a) => {
                            let mut count = 0u64;
                            let mut scratch = env.clone();
                            for item in a.iter() {
                                let prev = scratch.swap_current(item.clone());
                                let t = is_truthy(&self.exec(pred, &scratch)?);
                                scratch.restore_current(prev);
                                if t { count += 1; }
                            }
                            count
                        }
                        Val::IntVec(a) => {
                            let mut count = 0u64;
                            let mut scratch = env.clone();
                            for &n in a.iter() {
                                let prev = scratch.swap_current(Val::Int(n));
                                let t = is_truthy(&self.exec(pred, &scratch)?);
                                scratch.restore_current(prev);
                                if t { count += 1; }
                            }
                            count
                        }
                        Val::FloatVec(a) => {
                            let mut count = 0u64;
                            let mut scratch = env.clone();
                            for &f in a.iter() {
                                let prev = scratch.swap_current(Val::Float(f));
                                let t = is_truthy(&self.exec(pred, &scratch)?);
                                scratch.restore_current(prev);
                                if t { count += 1; }
                            }
                            count
                        }
                        _ => 0,
                    };
                    stack.push(Val::Int(n as i64));
                }
                Opcode::FindFirst(pred) => {
                    let recv = pop!(stack);
                    let mut found = Val::Null;
                    if let Val::Arr(a) = &recv {
                        let mut scratch = env.clone();
                        for item in a.iter() {
                            let prev = scratch.swap_current(item.clone());
                            let t = is_truthy(&self.exec(pred, &scratch)?);
                            scratch.restore_current(prev);
                            if t { found = item.clone(); break; }
                        }
                    } else if !recv.is_null() {
                        let sub_env = env.with_current(recv.clone());
                        if is_truthy(&self.exec(pred, &sub_env)?) { found = recv; }
                    }
                    stack.push(found);
                }
                Opcode::FilterMap { pred, map } => {
                    let recv = pop!(stack);
                    let recv = match recv {
                        Val::StrVec(_) | Val::IntVec(_) | Val::FloatVec(_) => recv.into_arr(),
                        v => v,
                    };
                    if let Val::Arr(a) = recv {
                        let mut out = Vec::with_capacity(a.len());
                        let mut scratch = env.clone();
                        for item in a.iter() {
                            let prev = scratch.swap_current(item.clone());
                            if is_truthy(&self.exec(pred, &scratch)?) {
                                out.push(self.exec(map, &scratch)?);
                            }
                            scratch.restore_current(prev);
                        }
                        stack.push(Val::arr(out));
                    } else {
                        stack.push(Val::arr(Vec::new()));
                    }
                }
                Opcode::MapFilter { map, pred } => {
                    let recv = pop!(stack);
                    let recv = match recv {
                        Val::StrVec(_) | Val::IntVec(_) | Val::FloatVec(_) => recv.into_arr(),
                        v => v,
                    };
                    if let Val::Arr(a) = recv {
                        let mut out = Vec::with_capacity(a.len());
                        let mut scratch = env.clone();
                        for item in a.iter() {
                            let prev = scratch.swap_current(item.clone());
                            let mapped = self.exec(map, &scratch)?;
                            let pscratch = scratch.with_current(mapped.clone());
                            if is_truthy(&self.exec(pred, &pscratch)?) {
                                out.push(mapped);
                            }
                            scratch.restore_current(prev);
                        }
                        stack.push(Val::arr(out));
                    } else {
                        stack.push(Val::arr(Vec::new()));
                    }
                }
                Opcode::FilterFilter { p1, p2 } => {
                    let recv = pop!(stack);
                    let recv = match recv {
                        Val::StrVec(_) | Val::IntVec(_) | Val::FloatVec(_) => recv.into_arr(),
                        v => v,
                    };
                    if let Val::Arr(a) = recv {
                        let mut out = Vec::with_capacity(a.len());
                        let mut scratch = env.clone();
                        for item in a.iter() {
                            let prev = scratch.swap_current(item.clone());
                            let keep = is_truthy(&self.exec(p1, &scratch)?)
                                    && is_truthy(&self.exec(p2, &scratch)?);
                            scratch.restore_current(prev);
                            if keep { out.push(item.clone()); }
                        }
                        stack.push(Val::arr(out));
                    } else {
                        stack.push(Val::arr(Vec::new()));
                    }
                }
                Opcode::MapToJsonJoin { sep_prog } => {
                    use std::fmt::Write as _;
                    let recv = pop!(stack);
                    let sep_val = self.exec(sep_prog, env)?;
                    let sep: &str = match &sep_val {
                        Val::Str(s) => s.as_ref(),
                        _ => "",
                    };
                    // Columnar shortcut: IntVec / FloatVec -> tight
                    // number-format loop without per-item Val alloc.
                    match &recv {
                        Val::IntVec(a) => {
                            let mut out = String::with_capacity(a.len() * 6);
                            let mut first = true;
                            for n in a.iter() {
                                if !first { out.push_str(sep); } first = false;
                                let _ = write!(out, "{}", n);
                            }
                            stack.push(Val::Str(Arc::<str>::from(out)));
                        }
                        Val::FloatVec(a) => {
                            let mut out = String::with_capacity(a.len() * 8);
                            let mut first = true;
                            for f in a.iter() {
                                if !first { out.push_str(sep); } first = false;
                                if f.is_finite() {
                                    let v = serde_json::Value::from(*f);
                                    out.push_str(&serde_json::to_string(&v).unwrap_or_default());
                                } else {
                                    out.push_str("null");
                                }
                            }
                            stack.push(Val::Str(Arc::<str>::from(out)));
                        }
                        Val::Arr(a) => {
                            let mut out = String::with_capacity(a.len() * 8);
                            let mut first = true;
                            for item in a.iter() {
                                if !first { out.push_str(sep); } first = false;
                                match item {
                                    Val::Int(n)  => { let _ = write!(out, "{}", n); }
                                    Val::Float(f) => {
                                        if f.is_finite() {
                                            let v = serde_json::Value::from(*f);
                                            out.push_str(&serde_json::to_string(&v).unwrap_or_default());
                                        } else { out.push_str("null"); }
                                    }
                                    Val::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
                                    Val::Null    => out.push_str("null"),
                                    Val::Str(s)  => {
                                        let src = s.as_ref();
                                        let mut needs_escape = false;
                                        for &b in src.as_bytes() {
                                            if b < 0x20 || b == b'"' || b == b'\\' { needs_escape = true; break; }
                                        }
                                        if !needs_escape {
                                            out.push('"'); out.push_str(src); out.push('"');
                                        } else {
                                            let v = serde_json::Value::String(s.to_string());
                                            out.push_str(&serde_json::to_string(&v).unwrap_or_default());
                                        }
                                    }
                                    _ => {
                                        let sv: serde_json::Value = item.clone().into();
                                        out.push_str(&serde_json::to_string(&sv).unwrap_or_default());
                                    }
                                }
                            }
                            stack.push(Val::Str(Arc::<str>::from(out)));
                        }
                        _ => stack.push(Val::Str(Arc::<str>::from(""))),
                    }
                }
                // ── Fused string-method chains ────────────────────────────────
                Opcode::StrTrimUpper | Opcode::StrTrimLower => {
                    let v = pop!(stack);
                    let out = if let Val::Str(s) = &v {
                        let t = s.trim();
                        let bytes = t.as_bytes();
                        if bytes.is_ascii() {
                            let upper = matches!(op, Opcode::StrTrimUpper);
                            Val::Str(ascii_fold_to_arc_str(bytes, upper))
                        } else {
                            let s2 = match op {
                                Opcode::StrTrimUpper => t.to_uppercase(),
                                _                    => t.to_lowercase(),
                            };
                            Val::Str(Arc::<str>::from(s2))
                        }
                    } else {
                        return Err(EvalError(format!("{:?}: expected string", op)));
                    };
                    stack.push(out);
                }
                Opcode::StrUpperTrim | Opcode::StrLowerTrim => {
                    let v = pop!(stack);
                    let out = if let Val::Str(s) = &v {
                        let bytes = s.as_bytes();
                        let t = s.trim();
                        let tb = t.as_bytes();
                        if bytes.is_ascii() {
                            let upper = matches!(op, Opcode::StrUpperTrim);
                            Val::Str(ascii_fold_to_arc_str(tb, upper))
                        } else {
                            let s2 = match op {
                                Opcode::StrUpperTrim => t.to_uppercase(),
                                _                    => t.to_lowercase(),
                            };
                            Val::Str(Arc::<str>::from(s2))
                        }
                    } else {
                        return Err(EvalError(format!("{:?}: expected string", op)));
                    };
                    stack.push(out);
                }
                Opcode::StrSplitReverseJoin { sep } => {
                    let v = pop!(stack);
                    let out = if let Val::Str(s) = &v {
                        let src = s.as_ref();
                        let sep_s = sep.as_ref();
                        // Collect segment (start, end) pairs via one byte-scan.
                        let mut spans: SmallVec<[(usize, usize); 8]> = SmallVec::new();
                        if sep_s.is_empty() {
                            // Mirror str::split("") behaviour: every char boundary.
                            let mut prev = 0usize;
                            for (i, _) in src.char_indices() {
                                if i > 0 { spans.push((prev, i)); prev = i; }
                            }
                            spans.push((prev, src.len()));
                        } else {
                            let mut prev = 0usize;
                            let sb = sep_s.as_bytes();
                            let slen = sb.len();
                            let bytes = src.as_bytes();
                            let mut i = 0usize;
                            while i + slen <= bytes.len() {
                                if &bytes[i..i + slen] == sb {
                                    spans.push((prev, i));
                                    i += slen; prev = i;
                                } else { i += 1; }
                            }
                            spans.push((prev, src.len()));
                        }
                        // SAFETY plan for the write-loop below:
                        // - `out_len == src.len()`: reverse join over the
                        //   same segments + same separator count produces
                        //   exactly the input byte count.
                        // - `arc` just returned from `new_uninit_slice`, no
                        //   other refs: `get_mut().unwrap()` is sound.
                        // - Segment spans `(a, b)` came from `str::char_indices`
                        //   or ASCII byte-scan of valid UTF-8 → on UTF-8
                        //   boundaries.
                        // - Final cast to `*const str`: payload is valid
                        //   UTF-8 because it is a permutation of the source
                        //   bytes joined by the same `sep` (both already
                        //   valid UTF-8 at known boundaries).
                        let out_len = src.len();
                        let mut arc = Arc::<[u8]>::new_uninit_slice(out_len);
                        let slot = Arc::get_mut(&mut arc).unwrap();
                        let src_b = src.as_bytes();
                        let sep_b = sep_s.as_bytes();
                        let slen = sep_b.len();
                        let n = spans.len();
                        // SAFETY: see plan above.
                        unsafe {
                            let dst = slot.as_mut_ptr() as *mut u8;
                            let mut widx = 0usize;
                            for idx in 0..n {
                                let (a, b) = spans[n - 1 - idx];
                                if idx > 0 && slen > 0 {
                                    std::ptr::copy_nonoverlapping(sep_b.as_ptr(), dst.add(widx), slen);
                                    widx += slen;
                                }
                                let seg_len = b - a;
                                std::ptr::copy_nonoverlapping(src_b.as_ptr().add(a), dst.add(widx), seg_len);
                                widx += seg_len;
                            }
                            debug_assert_eq!(widx, out_len);
                        }
                        // SAFETY: all `out_len` bytes initialised by the
                        // write-loop above (asserted).
                        let arc_bytes: Arc<[u8]> = unsafe { arc.assume_init() };
                        // SAFETY: `Arc<[u8]>` and `Arc<str>` share layout;
                        // payload is valid UTF-8 as argued above.
                        let arc_str: Arc<str> = unsafe {
                            Arc::from_raw(Arc::into_raw(arc_bytes) as *const str)
                        };
                        Val::Str(arc_str)
                    } else {
                        return Err(EvalError("split_reverse_join: expected string".into()));
                    };
                    stack.push(out);
                }
                Opcode::MapReplaceLit { needle, with, all } => {
                    let v = pop!(stack);
                    let v = if matches!(&v, Val::StrVec(_)) { v.into_arr() } else { v };
                    let n: &str = needle.as_ref();
                    let w: &str = with.as_ref();
                    let nlen = n.len();
                    let wlen = w.len();
                    let mut out_vec: Vec<Val> = match &v {
                        Val::Arr(a) => Vec::with_capacity(a.len()),
                        _ => Vec::new(),
                    };
                    if let Val::Arr(a) = &v {
                        for item in a.iter() {
                            if let Val::Str(s) = item {
                                let src = s.as_ref();
                                let Some(first_idx) = src.find(n) else {
                                    out_vec.push(Val::Str(s.clone()));
                                    continue;
                                };
                                // Two-pass: (1) count hits to compute exact
                                // size, (2) allocate Arc<str> directly with
                                // new_uninit_slice and write bytes once —
                                // avoids intermediate String alloc + copy
                                // that Arc::<str>::from(String) would do.
                                let hit_count = if *all {
                                    let mut c: usize = 1;
                                    let mut pos = first_idx + nlen;
                                    while let Some(i) = src[pos..].find(n) {
                                        c += 1;
                                        pos += i + nlen;
                                    }
                                    c
                                } else { 1 };
                                let out_len = src.len() + hit_count * wlen - hit_count * nlen;
                                let mut arc = Arc::<[u8]>::new_uninit_slice(out_len);
                                // SAFETY: unique new allocation; no other refs exist yet.
                                let slot = Arc::get_mut(&mut arc).unwrap();
                                let src_b = src.as_bytes();
                                let w_b = w.as_bytes();
                                let mut widx = 0usize;
                                // Write prefix up to first hit.
                                // SAFETY: MaybeUninit<u8> has same layout as u8.
                                unsafe {
                                    let dst = slot.as_mut_ptr() as *mut u8;
                                    std::ptr::copy_nonoverlapping(src_b.as_ptr(), dst, first_idx);
                                    widx += first_idx;
                                    std::ptr::copy_nonoverlapping(w_b.as_ptr(), dst.add(widx), wlen);
                                    widx += wlen;
                                    let mut last_end = first_idx + nlen;
                                    if *all {
                                        while let Some(i) = src[last_end..].find(n) {
                                            let abs = last_end + i;
                                            let len = abs - last_end;
                                            std::ptr::copy_nonoverlapping(src_b.as_ptr().add(last_end), dst.add(widx), len);
                                            widx += len;
                                            std::ptr::copy_nonoverlapping(w_b.as_ptr(), dst.add(widx), wlen);
                                            widx += wlen;
                                            last_end = abs + nlen;
                                        }
                                    }
                                    let tail = src_b.len() - last_end;
                                    std::ptr::copy_nonoverlapping(src_b.as_ptr().add(last_end), dst.add(widx), tail);
                                    debug_assert_eq!(widx + tail, out_len,
                                        "MapReplaceLit: hit-count predicted {} bytes, wrote {}",
                                        out_len, widx + tail);
                                }
                                // SAFETY: all `out_len` bytes are initialised above; the
                                // bytes are valid UTF-8 because src is valid UTF-8 and
                                // every substitution inserts a valid UTF-8 `w`.
                                let arc_bytes: Arc<[u8]> = unsafe { arc.assume_init() };
                                let arc_str: Arc<str> = unsafe {
                                    Arc::from_raw(Arc::into_raw(arc_bytes) as *const str)
                                };
                                out_vec.push(Val::Str(arc_str));
                            } else {
                                out_vec.push(item.clone());
                            }
                        }
                    }
                    stack.push(Val::arr(out_vec));
                }
                Opcode::MapUpperReplaceLit { needle, with, all }
                | Opcode::MapLowerReplaceLit { needle, with, all } => {
                    let to_upper = matches!(op, Opcode::MapUpperReplaceLit { .. });
                    let v = pop!(stack);
                    let v = if matches!(&v, Val::StrVec(_)) { v.into_arr() } else { v };
                    let n: &str = needle.as_ref();
                    let w: &str = with.as_ref();
                    let nlen = n.len();
                    let wlen = w.len();
                    let mut out_vec: Vec<Val> = match &v {
                        Val::Arr(a) => Vec::with_capacity(a.len()),
                        _ => Vec::new(),
                    };
                    if let Val::Arr(a) = &v {
                        for item in a.iter() {
                            if let Val::Str(s) = item {
                                let src = s.as_ref();
                                if src.is_ascii() {
                                    // ASCII fast path: compute needle hits against
                                    // cased source (upper needle vs upper haystack /
                                    // lower vs lower) so match semantics equal
                                    // `s.upper().replace(n, w)`.
                                    let src_b = src.as_bytes();
                                    let n_b = n.as_bytes();
                                    let w_b = w.as_bytes();
                                    let mut hits: Vec<usize> = Vec::new();
                                    if nlen > 0 && nlen <= src_b.len() {
                                        let mut i = 0usize;
                                        'outer: while i + nlen <= src_b.len() {
                                            for j in 0..nlen {
                                                let c = src_b[i + j];
                                                let cased = if to_upper {
                                                    if c.is_ascii_lowercase() { c - 32 } else { c }
                                                } else if c.is_ascii_uppercase() { c + 32 } else { c };
                                                if cased != n_b[j] {
                                                    i += 1;
                                                    continue 'outer;
                                                }
                                            }
                                            hits.push(i);
                                            if !*all { break; }
                                            i += nlen;
                                        }
                                    }
                                    let out_len = src_b.len() + hits.len() * wlen - hits.len() * nlen;
                                    let mut arc = Arc::<[u8]>::new_uninit_slice(out_len);
                                    let slot = Arc::get_mut(&mut arc).unwrap();
                                    unsafe {
                                        let dst = slot.as_mut_ptr() as *mut u8;
                                        let mut widx = 0usize;
                                        let mut last_end = 0usize;
                                        for &hit in &hits {
                                            // Copy [last_end..hit), case-transformed.
                                            for k in last_end..hit {
                                                let c = src_b[k];
                                                let cased = if to_upper {
                                                    if c.is_ascii_lowercase() { c - 32 } else { c }
                                                } else if c.is_ascii_uppercase() { c + 32 } else { c };
                                                *dst.add(widx) = cased;
                                                widx += 1;
                                            }
                                            std::ptr::copy_nonoverlapping(w_b.as_ptr(), dst.add(widx), wlen);
                                            widx += wlen;
                                            last_end = hit + nlen;
                                        }
                                        for k in last_end..src_b.len() {
                                            let c = src_b[k];
                                            let cased = if to_upper {
                                                if c.is_ascii_lowercase() { c - 32 } else { c }
                                            } else if c.is_ascii_uppercase() { c + 32 } else { c };
                                            *dst.add(widx) = cased;
                                            widx += 1;
                                        }
                                        debug_assert_eq!(widx, out_len);
                                    }
                                    let arc_bytes: Arc<[u8]> = unsafe { arc.assume_init() };
                                    let arc_str: Arc<str> = unsafe {
                                        Arc::from_raw(Arc::into_raw(arc_bytes) as *const str)
                                    };
                                    out_vec.push(Val::Str(arc_str));
                                } else {
                                    // Unicode fallback: build cased String, then replace.
                                    let cased = if to_upper { src.to_uppercase() } else { src.to_lowercase() };
                                    let replaced = if *all {
                                        cased.replace(n, w)
                                    } else {
                                        cased.replacen(n, w, 1)
                                    };
                                    out_vec.push(Val::Str(Arc::<str>::from(replaced)));
                                }
                            } else {
                                out_vec.push(item.clone());
                            }
                        }
                    }
                    stack.push(Val::arr(out_vec));
                }
                Opcode::MapStrConcat { prefix, suffix } => {
                    let v = pop!(stack);
                    let v = if matches!(&v, Val::StrVec(_)) { v.into_arr() } else { v };
                    let p_b = prefix.as_bytes();
                    let s_b = suffix.as_bytes();
                    let pl = p_b.len();
                    let sl = s_b.len();
                    let mut out_vec: Vec<Val> = match &v {
                        Val::Arr(a) => Vec::with_capacity(a.len()),
                        _ => Vec::new(),
                    };
                    if let Val::Arr(a) = &v {
                        for item in a.iter() {
                            if let Val::Str(s) = item {
                                let src_b = s.as_bytes();
                                let out_len = pl + src_b.len() + sl;
                                let mut arc = Arc::<[u8]>::new_uninit_slice(out_len);
                                let slot = Arc::get_mut(&mut arc).unwrap();
                                unsafe {
                                    let dst = slot.as_mut_ptr() as *mut u8;
                                    if pl > 0 {
                                        std::ptr::copy_nonoverlapping(p_b.as_ptr(), dst, pl);
                                    }
                                    std::ptr::copy_nonoverlapping(
                                        src_b.as_ptr(), dst.add(pl), src_b.len());
                                    if sl > 0 {
                                        std::ptr::copy_nonoverlapping(
                                            s_b.as_ptr(), dst.add(pl + src_b.len()), sl);
                                    }
                                }
                                let arc_bytes: Arc<[u8]> = unsafe { arc.assume_init() };
                                let arc_str: Arc<str> = unsafe {
                                    Arc::from_raw(Arc::into_raw(arc_bytes) as *const str)
                                };
                                out_vec.push(Val::Str(arc_str));
                            } else {
                                // Non-string element: fall back to add_vals semantics.
                                let a1 = super::eval::util::add_vals(
                                    Val::Str(prefix.clone()), item.clone())
                                    .unwrap_or(Val::Null);
                                let a2 = super::eval::util::add_vals(
                                    a1, Val::Str(suffix.clone()))
                                    .unwrap_or(Val::Null);
                                out_vec.push(a2);
                            }
                        }
                    }
                    stack.push(Val::arr(out_vec));
                }
                Opcode::MapSplitLenSum { sep } => {
                    let v = pop!(stack);
                    let v = if matches!(&v, Val::StrVec(_)) { v.into_arr() } else { v };
                    let sep_b = sep.as_bytes();
                    let slen = sep_b.len();
                    let sep_chars = sep.chars().count() as i64;
                    let mut out: Vec<i64> = match &v {
                        Val::Arr(a) => Vec::with_capacity(a.len()),
                        _ => Vec::new(),
                    };
                    if let Val::Arr(a) = &v {
                        for item in a.iter() {
                            if let Val::Str(s) = item {
                                let src = s.as_ref();
                                let row: i64 = if slen == 0 {
                                    // split("") yields one segment per char.
                                    // Each segment is one char, count = char_count.
                                    // Sum of lens = char_count. But split("") also
                                    // yields an empty prefix + per char + empty
                                    // suffix; match existing split(.).count()
                                    // semantics by falling back for empty sep.
                                    0
                                } else if src.is_ascii() && slen == 1 {
                                    // ASCII 1-byte sep: sum(segment_byte_len)
                                    //   = src.len() - hits
                                    let byte = sep_b[0];
                                    let hits = memchr::memchr_iter(byte, src.as_bytes())
                                                 .count() as i64;
                                    src.len() as i64 - hits
                                } else if src.is_ascii() {
                                    // Multi-byte ASCII sep.
                                    let hits = memchr::memmem::find_iter(src.as_bytes(), sep_b)
                                                 .count() as i64;
                                    src.len() as i64 - hits * slen as i64
                                } else {
                                    // Unicode source: count source chars, then
                                    // subtract hits * sep_chars.
                                    let src_chars = src.chars().count() as i64;
                                    let hits = if slen == 1 {
                                        memchr::memchr_iter(sep_b[0], src.as_bytes())
                                            .count() as i64
                                    } else {
                                        memchr::memmem::find_iter(src.as_bytes(), sep_b)
                                            .count() as i64
                                    };
                                    src_chars - hits * sep_chars
                                };
                                out.push(row);
                            } else {
                                out.push(0);
                            }
                        }
                    }
                    // Fallback for empty sep: recompute via generic path.
                    if slen == 0 {
                        // Classic split("") behavior = char count per seg +
                        // extra; match non-fused by computing via char count.
                        out.clear();
                        if let Val::Arr(a) = &v {
                            for item in a.iter() {
                                if let Val::Str(s) = item {
                                    // `split("").count()` is chars+1; sum of
                                    // individual char lens is just char count.
                                    out.push(s.chars().count() as i64);
                                } else {
                                    out.push(0);
                                }
                            }
                        }
                    }
                    stack.push(Val::int_vec(out));
                }
                Opcode::MapSplitCountSum { sep } => {
                    let v = pop!(stack);
                    let v = if matches!(&v, Val::StrVec(_)) { v.into_arr() } else { v };
                    let sep_b = sep.as_bytes();
                    let slen = sep_b.len();
                    let mut total: i64 = 0;
                    if let Val::Arr(a) = &v {
                        for item in a.iter() {
                            if let Val::Str(s) = item {
                                let sb = s.as_bytes();
                                let c: i64 = if slen == 0 {
                                    s.as_ref().chars().count() as i64 + 1
                                } else if slen == 1 {
                                    let byte = sep_b[0];
                                    let mut c: i64 = 1;
                                    let mut hay = sb;
                                    while let Some(pos) = memchr(byte, hay) {
                                        c += 1;
                                        hay = &hay[pos + 1..];
                                    }
                                    c
                                } else {
                                    let mut c: i64 = 1;
                                    let mut i = 0usize;
                                    while i + slen <= sb.len() {
                                        if &sb[i..i + slen] == sep_b {
                                            c += 1; i += slen;
                                        } else { i += 1; }
                                    }
                                    c
                                };
                                total += c;
                            }
                        }
                    }
                    stack.push(Val::Int(total));
                }
                Opcode::MapSplitCount { sep } => {
                    let v = pop!(stack);
                    let v = if matches!(&v, Val::StrVec(_)) { v.into_arr() } else { v };
                    let sep_b = sep.as_bytes();
                    let slen = sep_b.len();
                    let out = if let Val::Arr(a) = &v {
                        let mut out = Vec::with_capacity(a.len());
                        for item in a.iter() {
                            if let Val::Str(s) = item {
                                let sb = s.as_bytes();
                                let c = if slen == 0 {
                                    // str::split("") ≡ char boundary count + 1.
                                    s.as_ref().chars().count() as i64 + 1
                                } else if slen == 1 {
                                    // memchr SIMD scan for single-byte sep.
                                    let byte = sep_b[0];
                                    let mut c: i64 = 1;
                                    let mut hay = sb;
                                    while let Some(pos) = memchr(byte, hay) {
                                        c += 1;
                                        hay = &hay[pos + 1..];
                                    }
                                    c
                                } else {
                                    let mut c: i64 = 1;
                                    let mut i = 0usize;
                                    while i + slen <= sb.len() {
                                        if &sb[i..i + slen] == sep_b {
                                            c += 1; i += slen;
                                        } else { i += 1; }
                                    }
                                    c
                                };
                                out.push(Val::Int(c));
                            } else {
                                out.push(Val::Null);
                            }
                        }
                        Val::arr(out)
                    } else { Val::arr(Vec::new()) };
                    stack.push(out);
                }
                Opcode::MapSplitFirst { sep } => {
                    let v = pop!(stack);
                    let sep_s = sep.as_ref();
                    // StrVec / homogeneous Arr<Str> input → emit columnar
                    // StrSliceVec (Vec<StrRef>), no per-row Val enum tag.
                    if let Val::StrVec(a) = &v {
                        let mut out: Vec<crate::strref::StrRef> = Vec::with_capacity(a.len());
                        for s in a.iter() {
                            let src = s.as_ref();
                            if sep_s.is_empty() {
                                out.push(crate::strref::StrRef::slice(s.clone(), 0, 0));
                            } else {
                                let end = src.find(sep_s).unwrap_or(src.len());
                                out.push(crate::strref::StrRef::slice(s.clone(), 0, end));
                            }
                        }
                        stack.push(Val::StrSliceVec(Arc::new(out)));
                        continue;
                    }
                    let out = if let Val::Arr(a) = &v {
                        let all_str = a.iter().all(|it| matches!(it, Val::Str(_)));
                        if all_str {
                            let mut out: Vec<crate::strref::StrRef> = Vec::with_capacity(a.len());
                            for item in a.iter() {
                                if let Val::Str(s) = item {
                                    let src = s.as_ref();
                                    if sep_s.is_empty() {
                                        out.push(crate::strref::StrRef::slice(s.clone(), 0, 0));
                                    } else {
                                        let end = src.find(sep_s).unwrap_or(src.len());
                                        out.push(crate::strref::StrRef::slice(s.clone(), 0, end));
                                    }
                                }
                            }
                            stack.push(Val::StrSliceVec(Arc::new(out)));
                            continue;
                        }
                        let mut out = Vec::with_capacity(a.len());
                        for item in a.iter() {
                            if let Val::Str(s) = item {
                                let src = s.as_ref();
                                if sep_s.is_empty() {
                                    out.push(Val::Str(Arc::<str>::from("")));
                                    continue;
                                }
                                match src.find(sep_s) {
                                    Some(idx) => {
                                        out.push(Val::StrSlice(
                                            crate::strref::StrRef::slice(s.clone(), 0, idx)
                                        ));
                                    }
                                    None => out.push(Val::Str(s.clone())),
                                }
                            } else {
                                out.push(Val::Null);
                            }
                        }
                        Val::arr(out)
                    } else { Val::arr(Vec::new()) };
                    stack.push(out);
                }
                Opcode::MapSplitNth { sep, n } => {
                    let v = pop!(stack);
                    let v = if matches!(&v, Val::StrVec(_)) { v.into_arr() } else { v };
                    let sep_s = sep.as_ref();
                    let want = *n;
                    let out = if let Val::Arr(a) = &v {
                        let mut out = Vec::with_capacity(a.len());
                        for item in a.iter() {
                            if let Val::Str(s) = item {
                                let src = s.as_ref();
                                let mut pushed = false;
                                if sep_s.is_empty() {
                                    // nth char boundary
                                    if let Some((i, _)) = src.char_indices().nth(want) {
                                        let end = src[i..].chars().next().map(|c| i + c.len_utf8()).unwrap_or(i);
                                        out.push(Val::Str(Arc::<str>::from(&src[i..end])));
                                        pushed = true;
                                    }
                                } else {
                                    let mut prev = 0usize;
                                    let mut idx = 0usize;
                                    let sb = sep_s.as_bytes();
                                    let slen = sb.len();
                                    let bytes = src.as_bytes();
                                    if slen == 1 {
                                        let byte = sb[0];
                                        let mut cursor = 0usize;
                                        let mut hay = bytes;
                                        while let Some(off) = memchr(byte, hay) {
                                            let i = cursor + off;
                                            if idx == want {
                                                out.push(Val::Str(Arc::<str>::from(&src[prev..i])));
                                                pushed = true;
                                                break;
                                            }
                                            idx += 1;
                                            cursor = i + 1;
                                            prev = cursor;
                                            hay = &bytes[cursor..];
                                        }
                                    } else {
                                        let mut i = 0usize;
                                        while i + slen <= bytes.len() {
                                            if &bytes[i..i + slen] == sb {
                                                if idx == want {
                                                    out.push(Val::Str(Arc::<str>::from(&src[prev..i])));
                                                    pushed = true;
                                                    break;
                                                }
                                                idx += 1;
                                                i += slen;
                                                prev = i;
                                            } else { i += 1; }
                                        }
                                    }
                                    if !pushed && idx == want {
                                        let arc: Arc<str> = if prev == 0 { s.clone() }
                                                            else { Arc::<str>::from(&src[prev..]) };
                                        out.push(Val::Str(arc));
                                        pushed = true;
                                    }
                                }
                                if !pushed { out.push(Val::Null); }
                            } else {
                                out.push(Val::Null);
                            }
                        }
                        Val::arr(out)
                    } else { Val::arr(Vec::new()) };
                    stack.push(out);
                }
                Opcode::MapSum(f) => {
                    let recv = pop!(stack);
                    let mut acc_i: i64 = 0;
                    let mut acc_f: f64 = 0.0;
                    let mut is_float = false;
                    if let Val::Arr(a) = &recv {
                        if let Some(k) = trivial_field(&f.ops) {
                            let mut idx: Option<usize> = None;
                            for item in a.iter() {
                                if let Val::Obj(m) = item {
                                    match lookup_field_cached(m, &k, &mut idx) {
                                        Some(Val::Int(n))   => { if is_float { acc_f += *n as f64; } else { acc_i += *n; } }
                                        Some(Val::Float(x)) => { if !is_float { acc_f = acc_i as f64; is_float = true; } acc_f += *x; }
                                        Some(Val::Null) | None => {}
                                        _ => return err!("map(..).sum(): non-numeric mapped value"),
                                    }
                                }
                            }
                        } else {
                            let mut scratch = env.clone();
                            for item in a.iter() {
                                let prev = scratch.swap_current(item.clone());
                                let v = self.exec(f, &scratch)?;
                                scratch.restore_current(prev);
                                match v {
                                    Val::Int(n) => {
                                        if is_float { acc_f += n as f64; } else { acc_i += n; }
                                    }
                                    Val::Float(x) => {
                                        if !is_float { acc_f = acc_i as f64; is_float = true; }
                                        acc_f += x;
                                    }
                                    Val::Null => {}
                                    _ => return err!("map(..).sum(): non-numeric mapped value"),
                                }
                            }
                        }
                    }
                    stack.push(if is_float { Val::Float(acc_f) } else { Val::Int(acc_i) });
                }
                Opcode::MapAvg(f) => {
                    let recv = pop!(stack);
                    let mut sum: f64 = 0.0;
                    let mut n: usize = 0;
                    if let Val::Arr(a) = &recv {
                        if let Some(k) = trivial_field(&f.ops) {
                            let mut idx: Option<usize> = None;
                            for item in a.iter() {
                                if let Val::Obj(m) = item {
                                    match lookup_field_cached(m, &k, &mut idx) {
                                        Some(Val::Int(x))   => { sum += *x as f64; n += 1; }
                                        Some(Val::Float(x)) => { sum += *x;        n += 1; }
                                        Some(Val::Null) | None => {}
                                        _ => return err!("map(..).avg(): non-numeric mapped value"),
                                    }
                                }
                            }
                        } else {
                            let mut scratch = env.clone();
                            for item in a.iter() {
                                let prev = scratch.swap_current(item.clone());
                                let v = self.exec(f, &scratch)?;
                                scratch.restore_current(prev);
                                match v {
                                    Val::Int(x)   => { sum += x as f64; n += 1; }
                                    Val::Float(x) => { sum += x;        n += 1; }
                                    Val::Null => {}
                                    _ => return err!("map(..).avg(): non-numeric mapped value"),
                                }
                            }
                        }
                    }
                    stack.push(if n == 0 { Val::Null } else { Val::Float(sum / n as f64) });
                }
                Opcode::MapMin(f) => {
                    let recv = pop!(stack);
                    let mut best_i: Option<i64> = None;
                    let mut best_f: Option<f64> = None;
                    macro_rules! fold_min {
                        ($v:expr) => { match $v {
                            Val::Int(n) => {
                                let n = *n;
                                if let Some(bf) = best_f { if (n as f64) < bf { best_f = Some(n as f64); } }
                                else if let Some(bi) = best_i { if n < bi { best_i = Some(n); } }
                                else { best_i = Some(n); }
                            }
                            Val::Float(x) => {
                                let x = *x;
                                if best_f.is_none() { best_f = Some(best_i.map(|i| i as f64).unwrap_or(x)); best_i = None; }
                                if x < best_f.unwrap() { best_f = Some(x); }
                            }
                            Val::Null => {}
                            _ => return err!("map(..).min(): non-numeric mapped value"),
                        } }
                    }
                    if let Val::Arr(a) = &recv {
                        if let Some(k) = trivial_field(&f.ops) {
                            let mut idx: Option<usize> = None;
                            for item in a.iter() {
                                if let Val::Obj(m) = item {
                                    if let Some(v) = lookup_field_cached(m, &k, &mut idx) { fold_min!(v); }
                                }
                            }
                        } else {
                            let mut scratch = env.clone();
                            for item in a.iter() {
                                let prev = scratch.swap_current(item.clone());
                                let v = self.exec(f, &scratch)?;
                                scratch.restore_current(prev);
                                fold_min!(&v);
                            }
                        }
                    }
                    stack.push(match (best_i, best_f) {
                        (_, Some(x)) => Val::Float(x),
                        (Some(i), _) => Val::Int(i),
                        _ => Val::Null,
                    });
                }
                Opcode::MapMax(f) => {
                    let recv = pop!(stack);
                    let mut best_i: Option<i64> = None;
                    let mut best_f: Option<f64> = None;
                    macro_rules! fold_max {
                        ($v:expr) => { match $v {
                            Val::Int(n) => {
                                let n = *n;
                                if let Some(bf) = best_f { if (n as f64) > bf { best_f = Some(n as f64); } }
                                else if let Some(bi) = best_i { if n > bi { best_i = Some(n); } }
                                else { best_i = Some(n); }
                            }
                            Val::Float(x) => {
                                let x = *x;
                                if best_f.is_none() { best_f = Some(best_i.map(|i| i as f64).unwrap_or(x)); best_i = None; }
                                if x > best_f.unwrap() { best_f = Some(x); }
                            }
                            Val::Null => {}
                            _ => return err!("map(..).max(): non-numeric mapped value"),
                        } }
                    }
                    if let Val::Arr(a) = &recv {
                        if let Some(k) = trivial_field(&f.ops) {
                            let mut idx: Option<usize> = None;
                            for item in a.iter() {
                                if let Val::Obj(m) = item {
                                    if let Some(v) = lookup_field_cached(m, &k, &mut idx) { fold_max!(v); }
                                }
                            }
                        } else {
                            let mut scratch = env.clone();
                            for item in a.iter() {
                                let prev = scratch.swap_current(item.clone());
                                let v = self.exec(f, &scratch)?;
                                scratch.restore_current(prev);
                                fold_max!(&v);
                            }
                        }
                    }
                    stack.push(match (best_i, best_f) {
                        (_, Some(x)) => Val::Float(x),
                        (Some(i), _) => Val::Int(i),
                        _ => Val::Null,
                    });
                }

                // ── Field-specialised fusions (Tier 3) ────────────────────
                Opcode::MapField(k) => {
                    let recv = pop!(stack);
                    if let Val::Arr(a) = &recv {
                        let mut out = Vec::with_capacity(a.len());
                        let mut idx: Option<usize> = None;
                        for item in a.iter() {
                            match item {
                                Val::Obj(m) => out.push(
                                    lookup_field_cached(m, k, &mut idx)
                                        .cloned()
                                        .unwrap_or(Val::Null),
                                ),
                                _ => out.push(Val::Null),
                            }
                        }
                        stack.push(Val::arr(out));
                    } else {
                        stack.push(Val::arr(Vec::new()));
                    }
                }
                Opcode::MapFieldChain(ks) => {
                    let recv = pop!(stack);
                    if let Val::Arr(a) = &recv {
                        let mut out = Vec::with_capacity(a.len());
                        // One IC slot per hop — `lookup_field_cached` keys on
                        // slot index + key verify (ptr-independent), so it
                        // hits across different Arcs of the same shape.
                        let mut ic: SmallVec<[Option<usize>; 4]> = SmallVec::new();
                        ic.resize(ks.len(), None);
                        for item in a.iter() {
                            let mut cur: Val = match item {
                                Val::Obj(m) => lookup_field_cached(m, &ks[0], &mut ic[0])
                                    .cloned()
                                    .unwrap_or(Val::Null),
                                _ => Val::Null,
                            };
                            for (hop, k) in ks[1..].iter().enumerate() {
                                cur = match &cur {
                                    Val::Obj(m) => lookup_field_cached(m, k, &mut ic[hop + 1])
                                        .cloned()
                                        .unwrap_or(Val::Null),
                                    _ => Val::Null,
                                };
                                if matches!(cur, Val::Null) { break; }
                            }
                            out.push(cur);
                        }
                        stack.push(Val::arr(out));
                    } else {
                        stack.push(Val::arr(Vec::new()));
                    }
                }
                Opcode::MapFieldSum(k) => {
                    let recv = pop!(stack);
                    let mut acc_i: i64 = 0;
                    let mut acc_f: f64 = 0.0;
                    let mut is_float = false;
                    let mut idx: Option<usize> = None;
                    if let Val::Arr(a) = &recv {
                        for item in a.iter() {
                            if let Val::Obj(m) = item {
                                match lookup_field_cached(m, k, &mut idx) {
                                    Some(Val::Int(n))   => { if is_float { acc_f += *n as f64; } else { acc_i += *n; } }
                                    Some(Val::Float(x)) => { if !is_float { acc_f = acc_i as f64; is_float = true; } acc_f += *x; }
                                    Some(Val::Null) | None => {}
                                    _ => return err!("map(k).sum(): non-numeric field"),
                                }
                            }
                        }
                    }
                    stack.push(if is_float { Val::Float(acc_f) } else { Val::Int(acc_i) });
                }
                Opcode::MapFieldAvg(k) => {
                    let recv = pop!(stack);
                    let mut sum: f64 = 0.0;
                    let mut n: usize = 0;
                    let mut idx: Option<usize> = None;
                    if let Val::Arr(a) = &recv {
                        for item in a.iter() {
                            if let Val::Obj(m) = item {
                                match lookup_field_cached(m, k, &mut idx) {
                                    Some(Val::Int(x))   => { sum += *x as f64; n += 1; }
                                    Some(Val::Float(x)) => { sum += *x;        n += 1; }
                                    Some(Val::Null) | None => {}
                                    _ => return err!("map(k).avg(): non-numeric field"),
                                }
                            }
                        }
                    }
                    stack.push(if n == 0 { Val::Null } else { Val::Float(sum / n as f64) });
                }
                Opcode::MapFieldMin(k) => {
                    let recv = pop!(stack);
                    let mut best_i: Option<i64> = None;
                    let mut best_f: Option<f64> = None;
                    let mut idx: Option<usize> = None;
                    if let Val::Arr(a) = &recv {
                        for item in a.iter() {
                            if let Val::Obj(m) = item {
                                match lookup_field_cached(m, k, &mut idx) {
                                    Some(Val::Int(n)) => {
                                        let n = *n;
                                        if let Some(bf) = best_f { if (n as f64) < bf { best_f = Some(n as f64); } }
                                        else if let Some(bi) = best_i { if n < bi { best_i = Some(n); } }
                                        else { best_i = Some(n); }
                                    }
                                    Some(Val::Float(x)) => {
                                        let x = *x;
                                        if best_f.is_none() { best_f = Some(best_i.map(|i| i as f64).unwrap_or(x)); best_i = None; }
                                        if x < best_f.unwrap() { best_f = Some(x); }
                                    }
                                    Some(Val::Null) | None => {}
                                    _ => return err!("map(k).min(): non-numeric field"),
                                }
                            }
                        }
                    }
                    stack.push(match (best_i, best_f) {
                        (_, Some(x)) => Val::Float(x),
                        (Some(i), _) => Val::Int(i),
                        _ => Val::Null,
                    });
                }
                Opcode::MapFieldMax(k) => {
                    let recv = pop!(stack);
                    let mut best_i: Option<i64> = None;
                    let mut best_f: Option<f64> = None;
                    let mut idx: Option<usize> = None;
                    if let Val::Arr(a) = &recv {
                        for item in a.iter() {
                            if let Val::Obj(m) = item {
                                match lookup_field_cached(m, k, &mut idx) {
                                    Some(Val::Int(n)) => {
                                        let n = *n;
                                        if let Some(bf) = best_f { if (n as f64) > bf { best_f = Some(n as f64); } }
                                        else if let Some(bi) = best_i { if n > bi { best_i = Some(n); } }
                                        else { best_i = Some(n); }
                                    }
                                    Some(Val::Float(x)) => {
                                        let x = *x;
                                        if best_f.is_none() { best_f = Some(best_i.map(|i| i as f64).unwrap_or(x)); best_i = None; }
                                        if x > best_f.unwrap() { best_f = Some(x); }
                                    }
                                    Some(Val::Null) | None => {}
                                    _ => return err!("map(k).max(): non-numeric field"),
                                }
                            }
                        }
                    }
                    stack.push(match (best_i, best_f) {
                        (_, Some(x)) => Val::Float(x),
                        (Some(i), _) => Val::Int(i),
                        _ => Val::Null,
                    });
                }
                Opcode::MapFieldUnique(k) => {
                    let recv = pop!(stack);
                    let mut out: Vec<Val> = Vec::new();
                    let mut seen_int: std::collections::HashSet<i64> = std::collections::HashSet::new();
                    let mut seen_str: std::collections::HashSet<Arc<str>> = std::collections::HashSet::new();
                    let mut seen_other: Vec<Val> = Vec::new();
                    let mut idx: Option<usize> = None;
                    if let Val::Arr(a) = &recv {
                        for item in a.iter() {
                            if let Val::Obj(m) = item {
                                if let Some(v) = lookup_field_cached(m, k, &mut idx) {
                                    match v {
                                        Val::Int(n) => {
                                            if seen_int.insert(*n) { out.push(v.clone()); }
                                        }
                                        Val::Str(s) => {
                                            if seen_str.insert(s.clone()) { out.push(v.clone()); }
                                        }
                                        _ => {
                                            if !seen_other.iter().any(|o| crate::eval::util::vals_eq(o, v)) {
                                                seen_other.push(v.clone());
                                                out.push(v.clone());
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    stack.push(Val::arr(out));
                }
                Opcode::MapFieldChainUnique(ks) => {
                    let recv = pop!(stack);
                    let mut out: Vec<Val> = Vec::new();
                    let mut seen_int: std::collections::HashSet<i64> = std::collections::HashSet::new();
                    let mut seen_str: std::collections::HashSet<Arc<str>> = std::collections::HashSet::new();
                    let mut seen_other: Vec<Val> = Vec::new();
                    let mut ic: SmallVec<[Option<usize>; 4]> = SmallVec::new();
                    ic.resize(ks.len(), None);
                    if let Val::Arr(a) = &recv {
                        for item in a.iter() {
                            let mut cur: Val = match item {
                                Val::Obj(m) => lookup_field_cached(m, &ks[0], &mut ic[0])
                                    .cloned()
                                    .unwrap_or(Val::Null),
                                _ => Val::Null,
                            };
                            for (hop, k) in ks[1..].iter().enumerate() {
                                cur = match &cur {
                                    Val::Obj(m) => lookup_field_cached(m, k, &mut ic[hop + 1])
                                        .cloned()
                                        .unwrap_or(Val::Null),
                                    _ => Val::Null,
                                };
                                if matches!(cur, Val::Null) { break; }
                            }
                            match &cur {
                                Val::Int(n) => {
                                    if seen_int.insert(*n) { out.push(cur); }
                                }
                                Val::Str(s) => {
                                    if seen_str.insert(s.clone()) { out.push(cur); }
                                }
                                _ => {
                                    if !seen_other.iter().any(|o| crate::eval::util::vals_eq(o, &cur)) {
                                        seen_other.push(cur.clone());
                                        out.push(cur);
                                    }
                                }
                            }
                        }
                    }
                    stack.push(Val::arr(out));
                }

                // ── FlatMapChain (Tier 1) ─────────────────────────────────
                Opcode::FlatMapChain(keys) => {
                    let recv = pop!(stack);
                    let mut cur: Vec<Val> = match recv {
                        Val::Arr(a) => a.as_ref().clone(),
                        _ => Vec::new(),
                    };
                    for k in keys.iter() {
                        let mut next: Vec<Val> = Vec::with_capacity(cur.len() * 4);
                        let mut idx: Option<usize> = None;
                        for item in cur.drain(..) {
                            if let Val::Obj(m) = item {
                                if let Some(Val::Arr(inner)) = lookup_field_cached(&m, k, &mut idx) {
                                    for v in inner.iter() { next.push(v.clone()); }
                                }
                            }
                        }
                        cur = next;
                    }
                    stack.push(Val::arr(cur));
                }

                // ── Predicate specialisation (Tier 4) ─────────────────────
                Opcode::FilterFieldEqLit(k, lit) => {
                    let recv = pop!(stack);
                    let hint = match &recv { Val::Arr(a) => filter_cap_hint(a.len()), _ => 0 };
                    let mut out = Vec::with_capacity(hint);
                    let mut idx: Option<usize> = None;
                    if let Val::Arr(a) = &recv {
                        for item in a.iter() {
                            if let Val::Obj(m) = item {
                                if let Some(v) = lookup_field_cached(m, k, &mut idx) {
                                    if crate::eval::util::vals_eq(v, lit) {
                                        out.push(item.clone());
                                    }
                                }
                            }
                        }
                    }
                    stack.push(Val::arr(out));
                }
                Opcode::FilterFieldCmpLit(k, op, lit) => {
                    let recv = pop!(stack);
                    let hint = match &recv { Val::Arr(a) => filter_cap_hint(a.len()), _ => 0 };
                    let mut out = Vec::with_capacity(hint);
                    let mut idx: Option<usize> = None;
                    if let Val::Arr(a) = &recv {
                        for item in a.iter() {
                            if let Val::Obj(m) = item {
                                if let Some(v) = lookup_field_cached(m, k, &mut idx) {
                                    if cmp_val_binop(v, *op, lit) {
                                        out.push(item.clone());
                                    }
                                }
                            }
                        }
                    }
                    stack.push(Val::arr(out));
                }
                Opcode::FilterCurrentCmpLit(op, lit) => {
                    use super::ast::BinOp;
                    let recv = pop!(stack);
                    // Columnar fast paths — IntVec / FloatVec receivers
                    // walk the raw slice, produce a typed vec.  This is
                    // the autovectoriser-friendly shape (branchless
                    // body, one stride, no heap traffic per hit).
                    match (&recv, lit) {
                        (Val::IntVec(a), Val::Int(rhs)) => {
                            let rhs = *rhs;
                            let mut out: Vec<i64> = Vec::with_capacity(a.len());
                            match op {
                                BinOp::Eq  => for &n in a.iter() { if n == rhs { out.push(n); } }
                                BinOp::Neq => for &n in a.iter() { if n != rhs { out.push(n); } }
                                BinOp::Lt  => for &n in a.iter() { if n <  rhs { out.push(n); } }
                                BinOp::Lte => for &n in a.iter() { if n <= rhs { out.push(n); } }
                                BinOp::Gt  => for &n in a.iter() { if n >  rhs { out.push(n); } }
                                BinOp::Gte => for &n in a.iter() { if n >= rhs { out.push(n); } }
                                _ => {
                                    stack.push(recv);
                                    continue;
                                }
                            }
                            stack.push(Val::int_vec(out));
                            continue;
                        }
                        (Val::IntVec(a), Val::Float(rhs)) => {
                            let rhs = *rhs;
                            let mut out: Vec<i64> = Vec::with_capacity(a.len());
                            match op {
                                BinOp::Eq  => for &n in a.iter() { if (n as f64) == rhs { out.push(n); } }
                                BinOp::Neq => for &n in a.iter() { if (n as f64) != rhs { out.push(n); } }
                                BinOp::Lt  => for &n in a.iter() { if (n as f64) <  rhs { out.push(n); } }
                                BinOp::Lte => for &n in a.iter() { if (n as f64) <= rhs { out.push(n); } }
                                BinOp::Gt  => for &n in a.iter() { if (n as f64) >  rhs { out.push(n); } }
                                BinOp::Gte => for &n in a.iter() { if (n as f64) >= rhs { out.push(n); } }
                                _ => { stack.push(recv); continue; }
                            }
                            stack.push(Val::int_vec(out));
                            continue;
                        }
                        (Val::FloatVec(a), Val::Float(rhs)) => {
                            let rhs = *rhs;
                            let mut out: Vec<f64> = Vec::with_capacity(a.len());
                            match op {
                                BinOp::Eq  => for &f in a.iter() { if f == rhs { out.push(f); } }
                                BinOp::Neq => for &f in a.iter() { if f != rhs { out.push(f); } }
                                BinOp::Lt  => for &f in a.iter() { if f <  rhs { out.push(f); } }
                                BinOp::Lte => for &f in a.iter() { if f <= rhs { out.push(f); } }
                                BinOp::Gt  => for &f in a.iter() { if f >  rhs { out.push(f); } }
                                BinOp::Gte => for &f in a.iter() { if f >= rhs { out.push(f); } }
                                _ => { stack.push(recv); continue; }
                            }
                            stack.push(Val::float_vec(out));
                            continue;
                        }
                        (Val::FloatVec(a), Val::Int(rhs)) => {
                            let rhs = *rhs as f64;
                            let mut out: Vec<f64> = Vec::with_capacity(a.len());
                            match op {
                                BinOp::Eq  => for &f in a.iter() { if f == rhs { out.push(f); } }
                                BinOp::Neq => for &f in a.iter() { if f != rhs { out.push(f); } }
                                BinOp::Lt  => for &f in a.iter() { if f <  rhs { out.push(f); } }
                                BinOp::Lte => for &f in a.iter() { if f <= rhs { out.push(f); } }
                                BinOp::Gt  => for &f in a.iter() { if f >  rhs { out.push(f); } }
                                BinOp::Gte => for &f in a.iter() { if f >= rhs { out.push(f); } }
                                _ => { stack.push(recv); continue; }
                            }
                            stack.push(Val::float_vec(out));
                            continue;
                        }
                        (Val::StrVec(a), Val::Str(rhs)) => {
                            let rhs_b = rhs.as_bytes();
                            let mut out: Vec<Arc<str>> = Vec::with_capacity(a.len());
                            match op {
                                BinOp::Eq  => for s in a.iter() { if s.as_bytes() == rhs_b { out.push(s.clone()); } }
                                BinOp::Neq => for s in a.iter() { if s.as_bytes() != rhs_b { out.push(s.clone()); } }
                                BinOp::Lt  => for s in a.iter() { if s.as_bytes() <  rhs_b { out.push(s.clone()); } }
                                BinOp::Lte => for s in a.iter() { if s.as_bytes() <= rhs_b { out.push(s.clone()); } }
                                BinOp::Gt  => for s in a.iter() { if s.as_bytes() >  rhs_b { out.push(s.clone()); } }
                                BinOp::Gte => for s in a.iter() { if s.as_bytes() >= rhs_b { out.push(s.clone()); } }
                                _ => { stack.push(recv); continue; }
                            }
                            stack.push(Val::str_vec(out));
                            continue;
                        }
                        _ => {}
                    }
                    // Generic fallback — walk Val::Arr, compare each.
                    let hint = match &recv { Val::Arr(a) => filter_cap_hint(a.len()), _ => 0 };
                    let mut out = Vec::with_capacity(hint);
                    if let Val::Arr(a) = &recv {
                        for item in a.iter() {
                            if cmp_val_binop(item, *op, lit) { out.push(item.clone()); }
                        }
                    }
                    stack.push(Val::arr(out));
                }
                Opcode::FilterStrVecStartsWith(needle) => {
                    let recv = pop!(stack);
                    let n_b = needle.as_bytes();
                    match &recv {
                        Val::StrVec(a) => {
                            let mut out: Vec<Arc<str>> = Vec::with_capacity(a.len());
                            for s in a.iter() {
                                let b = s.as_bytes();
                                if b.len() >= n_b.len() && &b[..n_b.len()] == n_b {
                                    out.push(s.clone());
                                }
                            }
                            stack.push(Val::str_vec(out));
                        }
                        Val::Arr(a) => {
                            let mut out: Vec<Val> = Vec::with_capacity(filter_cap_hint(a.len()));
                            for item in a.iter() {
                                if let Val::Str(s) = item {
                                    let b = s.as_bytes();
                                    if b.len() >= n_b.len() && &b[..n_b.len()] == n_b {
                                        out.push(item.clone());
                                    }
                                }
                            }
                            stack.push(Val::arr(out));
                        }
                        _ => stack.push(Val::arr(Vec::new())),
                    }
                }
                Opcode::FilterStrVecEndsWith(needle) => {
                    let recv = pop!(stack);
                    let n_b = needle.as_bytes();
                    match &recv {
                        Val::StrVec(a) => {
                            let mut out: Vec<Arc<str>> = Vec::with_capacity(a.len());
                            for s in a.iter() {
                                let b = s.as_bytes();
                                if b.len() >= n_b.len() && &b[b.len() - n_b.len()..] == n_b {
                                    out.push(s.clone());
                                }
                            }
                            stack.push(Val::str_vec(out));
                        }
                        Val::Arr(a) => {
                            let mut out: Vec<Val> = Vec::with_capacity(filter_cap_hint(a.len()));
                            for item in a.iter() {
                                if let Val::Str(s) = item {
                                    let b = s.as_bytes();
                                    if b.len() >= n_b.len() && &b[b.len() - n_b.len()..] == n_b {
                                        out.push(item.clone());
                                    }
                                }
                            }
                            stack.push(Val::arr(out));
                        }
                        _ => stack.push(Val::arr(Vec::new())),
                    }
                }
                Opcode::FilterStrVecContains(needle) => {
                    let recv = pop!(stack);
                    let n_b = needle.as_bytes();
                    // Empty needle → every string matches (std::str::contains semantics).
                    match &recv {
                        Val::StrVec(a) => {
                            if n_b.is_empty() {
                                stack.push(recv);
                            } else {
                                let finder = memmem::Finder::new(n_b);
                                let mut out: Vec<Arc<str>> = Vec::with_capacity(a.len());
                                for s in a.iter() {
                                    if finder.find(s.as_bytes()).is_some() {
                                        out.push(s.clone());
                                    }
                                }
                                stack.push(Val::str_vec(out));
                            }
                        }
                        Val::Arr(a) => {
                            let finder = memmem::Finder::new(n_b);
                            let mut out: Vec<Val> = Vec::with_capacity(filter_cap_hint(a.len()));
                            for item in a.iter() {
                                if let Val::Str(s) = item {
                                    if n_b.is_empty() || finder.find(s.as_bytes()).is_some() {
                                        out.push(item.clone());
                                    }
                                }
                            }
                            stack.push(Val::arr(out));
                        }
                        _ => stack.push(Val::arr(Vec::new())),
                    }
                }
                Opcode::MapStrVecUpper => {
                    let recv = pop!(stack);
                    match &recv {
                        Val::StrVec(a) => {
                            let mut out: Vec<Arc<str>> = Vec::with_capacity(a.len());
                            for s in a.iter() {
                                // ASCII fast-path: scan for any lowercase; if none,
                                // reuse input Arc (no-alloc). Otherwise clone+mutate
                                // in a Vec<u8>, hand ownership to Arc<str> via String.
                                let b = s.as_bytes();
                                if b.is_ascii() {
                                    if !b.iter().any(|c| c.is_ascii_lowercase()) {
                                        out.push(s.clone());
                                    } else {
                                        let mut v = b.to_vec();
                                        for c in v.iter_mut() {
                                            if c.is_ascii_lowercase() { *c -= 32; }
                                        }
                                        // SAFETY: v was ASCII in, byte-shifted within ASCII → still UTF-8.
                                        let owned = unsafe { String::from_utf8_unchecked(v) };
                                        out.push(Arc::<str>::from(owned));
                                    }
                                } else {
                                    out.push(Arc::<str>::from(s.to_uppercase()));
                                }
                            }
                            stack.push(Val::str_vec(out));
                        }
                        Val::Arr(a) => {
                            let mut out: Vec<Val> = Vec::with_capacity(a.len());
                            for item in a.iter() {
                                if let Val::Str(s) = item {
                                    out.push(Val::Str(Arc::<str>::from(s.to_uppercase())));
                                } else {
                                    out.push(Val::Null);
                                }
                            }
                            stack.push(Val::arr(out));
                        }
                        Val::Str(s) => stack.push(Val::Str(Arc::<str>::from(s.to_uppercase()))),
                        _ => stack.push(recv),
                    }
                }
                Opcode::MapStrVecLower => {
                    let recv = pop!(stack);
                    match &recv {
                        Val::StrVec(a) => {
                            let mut out: Vec<Arc<str>> = Vec::with_capacity(a.len());
                            for s in a.iter() {
                                let b = s.as_bytes();
                                if b.is_ascii() {
                                    if !b.iter().any(|c| c.is_ascii_uppercase()) {
                                        out.push(s.clone());
                                    } else {
                                        let mut v = b.to_vec();
                                        for c in v.iter_mut() {
                                            if c.is_ascii_uppercase() { *c += 32; }
                                        }
                                        let owned = unsafe { String::from_utf8_unchecked(v) };
                                        out.push(Arc::<str>::from(owned));
                                    }
                                } else {
                                    out.push(Arc::<str>::from(s.to_lowercase()));
                                }
                            }
                            stack.push(Val::str_vec(out));
                        }
                        Val::Arr(a) => {
                            let mut out: Vec<Val> = Vec::with_capacity(a.len());
                            for item in a.iter() {
                                if let Val::Str(s) = item {
                                    out.push(Val::Str(Arc::<str>::from(s.to_lowercase())));
                                } else {
                                    out.push(Val::Null);
                                }
                            }
                            stack.push(Val::arr(out));
                        }
                        Val::Str(s) => stack.push(Val::Str(Arc::<str>::from(s.to_lowercase()))),
                        _ => stack.push(recv),
                    }
                }
                Opcode::MapStrVecTrim => {
                    let recv = pop!(stack);
                    match &recv {
                        Val::StrVec(a) => {
                            let mut out: Vec<Arc<str>> = Vec::with_capacity(a.len());
                            for s in a.iter() {
                                let t = s.trim();
                                if t.len() == s.len() {
                                    out.push(s.clone());
                                } else {
                                    out.push(Arc::from(t));
                                }
                            }
                            stack.push(Val::str_vec(out));
                        }
                        Val::Arr(a) => {
                            let mut out: Vec<Val> = Vec::with_capacity(a.len());
                            for item in a.iter() {
                                if let Val::Str(s) = item {
                                    let t = s.trim();
                                    if t.len() == s.len() {
                                        out.push(Val::Str(s.clone()));
                                    } else {
                                        out.push(Val::Str(Arc::from(t)));
                                    }
                                } else {
                                    out.push(Val::Null);
                                }
                            }
                            stack.push(Val::arr(out));
                        }
                        Val::Str(s) => {
                            let t = s.trim();
                            if t.len() == s.len() {
                                stack.push(Val::Str(s.clone()));
                            } else {
                                stack.push(Val::Str(Arc::from(t)));
                            }
                        }
                        _ => stack.push(recv),
                    }
                }
                Opcode::MapNumVecArith { op, lit, flipped } => {
                    use super::ast::BinOp;
                    let recv = pop!(stack);
                    // Int literal or Float literal — determine output lane.
                    let (lit_is_float, lit_i, lit_f) = match lit {
                        Val::Int(n) => (false, *n, *n as f64),
                        Val::Float(f) => (true, *f as i64, *f),
                        _ => { stack.push(recv); continue; }
                    };
                    match (&recv, *op, lit_is_float, *flipped) {
                        // IntVec × Int × {Add,Sub,Mul,Mod} → IntVec
                        (Val::IntVec(a), BinOp::Add, false, _) => {
                            let mut out: Vec<i64> = Vec::with_capacity(a.len());
                            for &n in a.iter() { out.push(n + lit_i); }
                            stack.push(Val::int_vec(out));
                        }
                        (Val::IntVec(a), BinOp::Sub, false, false) => {
                            let mut out: Vec<i64> = Vec::with_capacity(a.len());
                            for &n in a.iter() { out.push(n - lit_i); }
                            stack.push(Val::int_vec(out));
                        }
                        (Val::IntVec(a), BinOp::Sub, false, true) => {
                            let mut out: Vec<i64> = Vec::with_capacity(a.len());
                            for &n in a.iter() { out.push(lit_i - n); }
                            stack.push(Val::int_vec(out));
                        }
                        (Val::IntVec(a), BinOp::Mul, false, _) => {
                            let mut out: Vec<i64> = Vec::with_capacity(a.len());
                            for &n in a.iter() { out.push(n * lit_i); }
                            stack.push(Val::int_vec(out));
                        }
                        (Val::IntVec(a), BinOp::Mod, false, false) if lit_i != 0 => {
                            let mut out: Vec<i64> = Vec::with_capacity(a.len());
                            for &n in a.iter() { out.push(n % lit_i); }
                            stack.push(Val::int_vec(out));
                        }
                        // IntVec × Int × Div → FloatVec (matches Val::Div semantics)
                        (Val::IntVec(a), BinOp::Div, false, false) if lit_i != 0 => {
                            let mut out: Vec<f64> = Vec::with_capacity(a.len());
                            let div = lit_i as f64;
                            for &n in a.iter() { out.push(n as f64 / div); }
                            stack.push(Val::float_vec(out));
                        }
                        (Val::IntVec(a), BinOp::Div, false, true) => {
                            let mut out: Vec<f64> = Vec::with_capacity(a.len());
                            let num = lit_i as f64;
                            for &n in a.iter() {
                                out.push(if n != 0 { num / n as f64 } else { f64::INFINITY });
                            }
                            stack.push(Val::float_vec(out));
                        }
                        // IntVec × Float → FloatVec
                        (Val::IntVec(a), _, true, _) => {
                            let mut out: Vec<f64> = Vec::with_capacity(a.len());
                            for &n in a.iter() {
                                let x = n as f64;
                                let r = match (op, flipped) {
                                    (BinOp::Add, _) => x + lit_f,
                                    (BinOp::Sub, false) => x - lit_f,
                                    (BinOp::Sub, true)  => lit_f - x,
                                    (BinOp::Mul, _) => x * lit_f,
                                    (BinOp::Div, false) => x / lit_f,
                                    (BinOp::Div, true)  => lit_f / x,
                                    (BinOp::Mod, false) => x % lit_f,
                                    (BinOp::Mod, true)  => lit_f % x,
                                    _ => { out.clear(); break; }
                                };
                                out.push(r);
                            }
                            if out.len() == a.len() { stack.push(Val::float_vec(out)); }
                            else { stack.push(recv); }
                        }
                        // FloatVec × (Int|Float) → FloatVec
                        (Val::FloatVec(a), _, _, _) => {
                            let mut out: Vec<f64> = Vec::with_capacity(a.len());
                            for &x in a.iter() {
                                let r = match (op, flipped) {
                                    (BinOp::Add, _) => x + lit_f,
                                    (BinOp::Sub, false) => x - lit_f,
                                    (BinOp::Sub, true)  => lit_f - x,
                                    (BinOp::Mul, _) => x * lit_f,
                                    (BinOp::Div, false) => x / lit_f,
                                    (BinOp::Div, true)  => lit_f / x,
                                    (BinOp::Mod, false) => x % lit_f,
                                    (BinOp::Mod, true)  => lit_f % x,
                                    _ => { out.clear(); break; }
                                };
                                out.push(r);
                            }
                            if out.len() == a.len() { stack.push(Val::float_vec(out)); }
                            else { stack.push(recv); }
                        }
                        // Arr fallback — per-item numeric arithmetic.
                        (Val::Arr(a), _, _, _) => {
                            let a = Arc::clone(a);
                            drop(recv);
                            let mut out: Vec<Val> = Vec::with_capacity(a.len());
                            for item in a.iter() {
                                let (ix, ifl, is_flt) = match item {
                                    Val::Int(n) => (*n, *n as f64, false),
                                    Val::Float(f) => (*f as i64, *f, true),
                                    _ => { out.push(Val::Null); continue; }
                                };
                                let r = if is_flt || lit_is_float {
                                    let (a, b) = if *flipped { (lit_f, ifl) } else { (ifl, lit_f) };
                                    match op {
                                        BinOp::Add => Val::Float(a + b),
                                        BinOp::Sub => Val::Float(a - b),
                                        BinOp::Mul => Val::Float(a * b),
                                        BinOp::Div => Val::Float(a / b),
                                        BinOp::Mod => Val::Float(a % b),
                                        _ => Val::Null,
                                    }
                                } else {
                                    let (a, b) = if *flipped { (lit_i, ix) } else { (ix, lit_i) };
                                    match op {
                                        BinOp::Add => Val::Int(a + b),
                                        BinOp::Sub => Val::Int(a - b),
                                        BinOp::Mul => Val::Int(a * b),
                                        BinOp::Div if b != 0 => Val::Float(a as f64 / b as f64),
                                        BinOp::Mod if b != 0 => Val::Int(a % b),
                                        _ => Val::Null,
                                    }
                                };
                                out.push(r);
                            }
                            stack.push(Val::arr(out));
                        }
                        _ => stack.push(recv),
                    }
                }
                Opcode::MapNumVecNeg => {
                    let recv = pop!(stack);
                    match &recv {
                        Val::IntVec(a) => {
                            let mut out: Vec<i64> = Vec::with_capacity(a.len());
                            for &n in a.iter() { out.push(-n); }
                            stack.push(Val::int_vec(out));
                        }
                        Val::FloatVec(a) => {
                            let mut out: Vec<f64> = Vec::with_capacity(a.len());
                            for &f in a.iter() { out.push(-f); }
                            stack.push(Val::float_vec(out));
                        }
                        Val::Arr(a) => {
                            let mut out: Vec<Val> = Vec::with_capacity(a.len());
                            for item in a.iter() {
                                out.push(match item {
                                    Val::Int(n) => Val::Int(-n),
                                    Val::Float(f) => Val::Float(-f),
                                    _ => Val::Null,
                                });
                            }
                            stack.push(Val::arr(out));
                        }
                        Val::Int(n) => stack.push(Val::Int(-n)),
                        Val::Float(f) => stack.push(Val::Float(-f)),
                        _ => stack.push(recv),
                    }
                }
                Opcode::FilterFieldEqLitMapField(kp, lit, kproj) => {
                    let recv = pop!(stack);
                    let hint = match &recv { Val::Arr(a) => filter_cap_hint(a.len()), _ => 0 };
                    let mut out = Vec::with_capacity(hint);
                    let mut ip: Option<usize> = None;
                    let mut iq: Option<usize> = None;
                    if let Val::Arr(a) = &recv {
                        for item in a.iter() {
                            if let Val::Obj(m) = item {
                                if let Some(v) = lookup_field_cached(m, kp, &mut ip) {
                                    if crate::eval::util::vals_eq(v, lit) {
                                        out.push(
                                            lookup_field_cached(m, kproj, &mut iq)
                                                .cloned()
                                                .unwrap_or(Val::Null),
                                        );
                                    }
                                }
                            }
                        }
                    }
                    stack.push(Val::arr(out));
                }
                Opcode::FilterFieldCmpLitMapField(kp, op, lit, kproj) => {
                    let recv = pop!(stack);
                    let hint = match &recv { Val::Arr(a) => filter_cap_hint(a.len()), _ => 0 };
                    let mut out = Vec::with_capacity(hint);
                    let mut ip: Option<usize> = None;
                    let mut iq: Option<usize> = None;
                    if let Val::Arr(a) = &recv {
                        for item in a.iter() {
                            if let Val::Obj(m) = item {
                                if let Some(v) = lookup_field_cached(m, kp, &mut ip) {
                                    if cmp_val_binop(v, *op, lit) {
                                        out.push(
                                            lookup_field_cached(m, kproj, &mut iq)
                                                .cloned()
                                                .unwrap_or(Val::Null),
                                        );
                                    }
                                }
                            }
                        }
                    }
                    stack.push(Val::arr(out));
                }
                Opcode::FilterFieldCmpField(k1, op, k2) => {
                    let recv = pop!(stack);
                    let hint = match &recv { Val::Arr(a) => filter_cap_hint(a.len()), _ => 0 };
                    let mut out = Vec::with_capacity(hint);
                    let mut i1: Option<usize> = None;
                    let mut i2: Option<usize> = None;
                    if let Val::Arr(a) = &recv {
                        for item in a.iter() {
                            if let Val::Obj(m) = item {
                                let v1 = lookup_field_cached(m, k1, &mut i1);
                                let v2 = lookup_field_cached(m, k2, &mut i2);
                                if let (Some(v1), Some(v2)) = (v1, v2) {
                                    if cmp_val_binop(v1, *op, v2) {
                                        out.push(item.clone());
                                    }
                                }
                            }
                        }
                    }
                    stack.push(Val::arr(out));
                }
                Opcode::FilterFieldsAllEqLitCount(pairs) => {
                    let recv = pop!(stack);
                    let mut n: i64 = 0;
                    let mut ics: SmallVec<[Option<usize>; 4]> = SmallVec::new();
                    ics.resize(pairs.len(), None);
                    if let Val::Arr(a) = &recv {
                        'item: for item in a.iter() {
                            if let Val::Obj(m) = item {
                                for (i, (k, lit)) in pairs.iter().enumerate() {
                                    match lookup_field_cached(m, k, &mut ics[i]) {
                                        Some(v) if crate::eval::util::vals_eq(v, lit) => {}
                                        _ => continue 'item,
                                    }
                                }
                                n += 1;
                            }
                        }
                    }
                    stack.push(Val::Int(n));
                }
                Opcode::FilterFieldsAllCmpLitCount(triples) => {
                    let recv = pop!(stack);
                    let mut n: i64 = 0;
                    let mut ics: SmallVec<[Option<usize>; 4]> = SmallVec::new();
                    ics.resize(triples.len(), None);
                    if let Val::Arr(a) = &recv {
                        'item: for item in a.iter() {
                            if let Val::Obj(m) = item {
                                for (i, (k, cop, lit)) in triples.iter().enumerate() {
                                    match lookup_field_cached(m, k, &mut ics[i]) {
                                        Some(v) if cmp_val_binop(v, *cop, lit) => {}
                                        _ => continue 'item,
                                    }
                                }
                                n += 1;
                            }
                        }
                    }
                    stack.push(Val::Int(n));
                }
                Opcode::FilterFieldEqLitCount(k, lit) => {
                    let recv = pop!(stack);
                    let mut n: i64 = 0;
                    let mut idx: Option<usize> = None;
                    if let Val::Arr(a) = &recv {
                        for item in a.iter() {
                            if let Val::Obj(m) = item {
                                if let Some(v) = lookup_field_cached(m, k, &mut idx) {
                                    if crate::eval::util::vals_eq(v, lit) { n += 1; }
                                }
                            }
                        }
                    }
                    stack.push(Val::Int(n));
                }
                Opcode::FilterFieldCmpLitCount(k, op, lit) => {
                    let recv = pop!(stack);
                    let mut n: i64 = 0;
                    let mut idx: Option<usize> = None;
                    if let Val::Arr(a) = &recv {
                        for item in a.iter() {
                            if let Val::Obj(m) = item {
                                if let Some(v) = lookup_field_cached(m, k, &mut idx) {
                                    if cmp_val_binop(v, *op, lit) { n += 1; }
                                }
                            }
                        }
                    }
                    stack.push(Val::Int(n));
                }
                Opcode::FilterFieldCmpFieldCount(k1, op, k2) => {
                    let recv = pop!(stack);
                    let mut n: i64 = 0;
                    let mut i1: Option<usize> = None;
                    let mut i2: Option<usize> = None;
                    if let Val::Arr(a) = &recv {
                        for item in a.iter() {
                            if let Val::Obj(m) = item {
                                let v1 = lookup_field_cached(m, k1, &mut i1);
                                let v2 = lookup_field_cached(m, k2, &mut i2);
                                if let (Some(v1), Some(v2)) = (v1, v2) {
                                    if cmp_val_binop(v1, *op, v2) { n += 1; }
                                }
                            }
                        }
                    }
                    stack.push(Val::Int(n));
                }

                // ── GroupByField (Tier 2) ─────────────────────────────────
                Opcode::GroupByField(k) => {
                    let recv = pop!(stack);
                    stack.push(group_by_field(&recv, k.as_ref()));
                }
                Opcode::CountByField(k) => {
                    let recv = pop!(stack);
                    stack.push(count_by_field(&recv, k.as_ref()));
                }
                Opcode::UniqueByField(k) => {
                    let recv = pop!(stack);
                    stack.push(unique_by_field(&recv, k.as_ref()));
                }
                Opcode::FilterMapSum { pred, map } => {
                    let recv = pop!(stack);
                    let mut acc_i: i64 = 0;
                    let mut acc_f: f64 = 0.0;
                    let mut is_float = false;
                    let run = |this: &mut Self, item: Val, scratch: &mut Env,
                               acc_i: &mut i64, acc_f: &mut f64, is_float: &mut bool|
                        -> Result<(), EvalError>
                    {
                        let prev = scratch.swap_current(item);
                        if !is_truthy(&this.exec(pred, scratch)?) {
                            scratch.restore_current(prev);
                            return Ok(());
                        }
                        let v = this.exec(map, scratch)?;
                        scratch.restore_current(prev);
                        match v {
                            Val::Int(n) => {
                                if *is_float { *acc_f += n as f64; } else { *acc_i += n; }
                            }
                            Val::Float(x) => {
                                if !*is_float { *acc_f = *acc_i as f64; *is_float = true; }
                                *acc_f += x;
                            }
                            Val::Null => {}
                            _ => return Err(EvalError("filter(..).map(..).sum(): non-numeric mapped value".into())),
                        }
                        Ok(())
                    };
                    match &recv {
                        Val::Arr(a) => {
                            let mut scratch = env.clone();
                            for item in a.iter() {
                                run(self, item.clone(), &mut scratch, &mut acc_i, &mut acc_f, &mut is_float)?;
                            }
                        }
                        Val::IntVec(a) => {
                            let mut scratch = env.clone();
                            for &n in a.iter() {
                                run(self, Val::Int(n), &mut scratch, &mut acc_i, &mut acc_f, &mut is_float)?;
                            }
                        }
                        Val::FloatVec(a) => {
                            let mut scratch = env.clone();
                            for &f in a.iter() {
                                run(self, Val::Float(f), &mut scratch, &mut acc_i, &mut acc_f, &mut is_float)?;
                            }
                        }
                        _ => {}
                    }
                    stack.push(if is_float { Val::Float(acc_f) } else { Val::Int(acc_i) });
                }
                Opcode::FilterMapAvg { pred, map } => {
                    let recv = pop!(stack);
                    let mut sum: f64 = 0.0;
                    let mut n: usize = 0;
                    if let Val::Arr(a) = &recv {
                        let mut scratch = env.clone();
                        for item in a.iter() {
                            let prev = scratch.swap_current(item.clone());
                            if !is_truthy(&self.exec(pred, &scratch)?) {
                                scratch.restore_current(prev);
                                continue;
                            }
                            let v = self.exec(map, &scratch)?;
                            scratch.restore_current(prev);
                            match v {
                                Val::Int(x)   => { sum += x as f64; n += 1; }
                                Val::Float(x) => { sum += x;        n += 1; }
                                Val::Null => {}
                                _ => return err!("filter(..).map(..).avg(): non-numeric mapped value"),
                            }
                        }
                    }
                    stack.push(if n == 0 { Val::Null } else { Val::Float(sum / n as f64) });
                }
                Opcode::FilterMapFirst { pred, map } => {
                    let recv = pop!(stack);
                    let mut out = Val::Null;
                    if let Val::Arr(a) = &recv {
                        let mut scratch = env.clone();
                        for item in a.iter() {
                            let prev = scratch.swap_current(item.clone());
                            if is_truthy(&self.exec(pred, &scratch)?) {
                                let mapped = self.exec(map, &scratch)?;
                                scratch.restore_current(prev);
                                out = mapped;
                                break;
                            }
                            scratch.restore_current(prev);
                        }
                    } else if !recv.is_null() {
                        let sub = env.with_current(recv.clone());
                        if is_truthy(&self.exec(pred, &sub)?) {
                            out = self.exec(map, &sub)?;
                        }
                    }
                    stack.push(out);
                }
                Opcode::FilterMapMin { pred, map } => {
                    let recv = pop!(stack);
                    stack.push(self.filter_map_minmax(recv, pred, map, env, true)?);
                }
                Opcode::FilterMapMax { pred, map } => {
                    let recv = pop!(stack);
                    stack.push(self.filter_map_minmax(recv, pred, map, env, false)?);
                }
                Opcode::FilterLast { pred } => {
                    let recv = pop!(stack);
                    let mut out = Val::Null;
                    if let Val::Arr(a) = &recv {
                        let mut scratch = env.clone();
                        for item in a.iter().rev() {
                            let prev = scratch.swap_current(item.clone());
                            if is_truthy(&self.exec(pred, &scratch)?) {
                                scratch.restore_current(prev);
                                out = item.clone();
                                break;
                            }
                            scratch.restore_current(prev);
                        }
                    } else if !recv.is_null() {
                        let sub = env.with_current(recv.clone());
                        if is_truthy(&self.exec(pred, &sub)?) {
                            out = recv;
                        }
                    }
                    stack.push(out);
                }
                Opcode::MapFirst(f) => {
                    let recv = pop!(stack);
                    let first = match recv {
                        Val::Arr(a) => match Arc::try_unwrap(a) {
                            Ok(mut v) if !v.is_empty() => Some(v.swap_remove(0)),
                            Ok(_) => None,
                            Err(a) => a.first().cloned(),
                        },
                        Val::Null => None,
                        other => Some(other),
                    };
                    let out = match first {
                        None => Val::Null,
                        Some(item) => {
                            let sub = env.with_current(item);
                            self.exec(f, &sub)?
                        }
                    };
                    stack.push(out);
                }
                Opcode::MapLast(f) => {
                    let recv = pop!(stack);
                    let last = match recv {
                        Val::Arr(a) => match Arc::try_unwrap(a) {
                            Ok(mut v) => v.pop(),
                            Err(a) => a.last().cloned(),
                        },
                        Val::Null => None,
                        other => Some(other),
                    };
                    let out = match last {
                        None => Val::Null,
                        Some(item) => {
                            let sub = env.with_current(item);
                            self.exec(f, &sub)?
                        }
                    };
                    stack.push(out);
                }
                Opcode::MapFlatten(f) => {
                    let recv = pop!(stack);
                    let items = match recv {
                        Val::Arr(a) => Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone()),
                        _ => Vec::new(),
                    };
                    let mut out = Vec::with_capacity(items.len());
                    let mut scratch = env.clone();
                    for item in items {
                        let prev = scratch.swap_current(item);
                        let mapped = self.exec(f, &scratch)?;
                        scratch.restore_current(prev);
                        match mapped {
                            Val::Arr(a) => {
                                let v = Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone());
                                out.extend(v);
                            }
                            other => out.push(other),
                        }
                    }
                    stack.push(Val::arr(out));
                }
                Opcode::FilterTakeWhile { pred, stop } => {
                    let recv = pop!(stack);
                    let items = match recv {
                        Val::Arr(a) => Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone()),
                        _ => Vec::new(),
                    };
                    let mut out = Vec::with_capacity(items.len());
                    let mut scratch = env.clone();
                    for item in items {
                        let prev = scratch.swap_current(item.clone());
                        let pass_pred = is_truthy(&self.exec(pred, &scratch)?);
                        if !pass_pred { scratch.restore_current(prev); continue; }
                        let stop_ok = is_truthy(&self.exec(stop, &scratch)?);
                        scratch.restore_current(prev);
                        if !stop_ok { break; }
                        out.push(item);
                    }
                    stack.push(Val::arr(out));
                }
                Opcode::FilterDropWhile { pred, drop } => {
                    let recv = pop!(stack);
                    let items = match recv {
                        Val::Arr(a) => Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone()),
                        _ => Vec::new(),
                    };
                    let mut out = Vec::with_capacity(items.len());
                    let mut dropping = true;
                    let mut scratch = env.clone();
                    for item in items {
                        let prev = scratch.swap_current(item.clone());
                        let pass_pred = is_truthy(&self.exec(pred, &scratch)?);
                        if !pass_pred { scratch.restore_current(prev); continue; }
                        if dropping {
                            let still_drop = is_truthy(&self.exec(drop, &scratch)?);
                            scratch.restore_current(prev);
                            if still_drop { continue; }
                            dropping = false;
                            out.push(item);
                            continue;
                        }
                        scratch.restore_current(prev);
                        out.push(item);
                    }
                    stack.push(Val::arr(out));
                }
                Opcode::EquiJoin { rhs, lhs_key, rhs_key } => {
                    use std::collections::HashMap;
                    let left_val = pop!(stack);
                    let right_val = self.exec(rhs, env)?;
                    let left = match left_val {
                        Val::Arr(a) => Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone()),
                        _ => Vec::new(),
                    };
                    let right = match right_val {
                        Val::Arr(a) => Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone()),
                        _ => Vec::new(),
                    };
                    let mut idx: HashMap<String, Vec<Val>> = HashMap::with_capacity(right.len());
                    let mut r_slot: Option<usize> = None;
                    for r in right {
                        let key = match &r {
                            Val::Obj(o) => lookup_field_cached(o, rhs_key, &mut r_slot)
                                .map(super::eval::util::val_to_key),
                            _ => None,
                        };
                        if let Some(k) = key { idx.entry(k).or_default().push(r); }
                    }
                    let mut out = Vec::with_capacity(left.len());
                    let mut l_slot: Option<usize> = None;
                    for l in left {
                        let key = match &l {
                            Val::Obj(o) => lookup_field_cached(o, lhs_key, &mut l_slot)
                                .map(super::eval::util::val_to_key),
                            _ => None,
                        };
                        let Some(k) = key else { continue };
                        let Some(matches) = idx.get(&k) else { continue };
                        for r in matches {
                            match (&l, r) {
                                (Val::Obj(lo), Val::Obj(ro)) => {
                                    let mut m = (**lo).clone();
                                    for (k, v) in ro.iter() { m.insert(k.clone(), v.clone()); }
                                    out.push(Val::obj(m));
                                }
                                _ => out.push(l.clone()),
                            }
                        }
                    }
                    stack.push(Val::arr(out));
                }
                Opcode::MapUnique(f) => {
                    let recv = pop!(stack);
                    let items = match recv {
                        Val::Arr(a) => Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone()),
                        _ => Vec::new(),
                    };
                    let mut seen: std::collections::HashSet<String> =
                        std::collections::HashSet::with_capacity(items.len());
                    let mut out = Vec::with_capacity(items.len());
                    let mut scratch = env.clone();
                    for item in items {
                        let prev = scratch.swap_current(item);
                        let mapped = self.exec(f, &scratch)?;
                        scratch.restore_current(prev);
                        if seen.insert(super::eval::util::val_to_key(&mapped)) {
                            out.push(mapped);
                        }
                    }
                    stack.push(Val::arr(out));
                }
                Opcode::TopN { n, asc } => {
                    use std::collections::BinaryHeap;
                    use std::cmp::Reverse;
                    let recv = pop!(stack);
                    let items = match recv {
                        Val::Arr(a) => Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone()),
                        Val::IntVec(a) => Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone())
                            .into_iter().map(Val::Int).collect(),
                        Val::FloatVec(a) => Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone())
                            .into_iter().map(Val::Float).collect(),
                        _ => Vec::new(),
                    };
                    if *n >= items.len() {
                        let mut v = items;
                        v.sort_by(|x, y| super::eval::util::cmp_vals(x, y));
                        if !*asc { v.reverse(); }
                        stack.push(Val::arr(v));
                    } else if *asc {
                        // Max-heap of size n; pop largest to keep smallest n.
                        let mut heap: BinaryHeap<WrapVal> = BinaryHeap::with_capacity(*n);
                        for item in items {
                            if heap.len() < *n {
                                heap.push(WrapVal(item));
                            } else if super::eval::util::cmp_vals(&item, &heap.peek().unwrap().0)
                                      == std::cmp::Ordering::Less {
                                heap.pop();
                                heap.push(WrapVal(item));
                            }
                        }
                        let mut v: Vec<Val> = heap.into_iter().map(|w| w.0).collect();
                        v.sort_by(|x, y| super::eval::util::cmp_vals(x, y));
                        stack.push(Val::arr(v));
                    } else {
                        // Min-heap via Reverse; keep largest n.
                        let mut heap: BinaryHeap<Reverse<WrapVal>> = BinaryHeap::with_capacity(*n);
                        for item in items {
                            if heap.len() < *n {
                                heap.push(Reverse(WrapVal(item)));
                            } else if super::eval::util::cmp_vals(&item, &heap.peek().unwrap().0.0)
                                      == std::cmp::Ordering::Greater {
                                heap.pop();
                                heap.push(Reverse(WrapVal(item)));
                            }
                        }
                        let mut v: Vec<Val> = heap.into_iter().map(|w| w.0.0).collect();
                        v.sort_by(|x, y| super::eval::util::cmp_vals(y, x));
                        stack.push(Val::arr(v));
                    }
                }
                Opcode::UniqueCount => {
                    use super::eval::util::val_to_key;
                    let recv = pop!(stack);
                    let items = match recv {
                        Val::Arr(a) => Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone()),
                        Val::Null   => { stack.push(Val::Int(0)); continue; }
                        other       => vec![other],
                    };
                    let mut seen: std::collections::HashSet<String> =
                        std::collections::HashSet::with_capacity(items.len());
                    let mut n: i64 = 0;
                    for it in &items {
                        if seen.insert(val_to_key(it)) { n += 1; }
                    }
                    stack.push(Val::Int(n));
                }
                Opcode::ArgExtreme { key, lam_param, max } => {
                    let recv = pop!(stack);
                    let items = match recv {
                        Val::Arr(a) => a,
                        _ => { stack.push(Val::Null); continue; }
                    };
                    if items.is_empty() { stack.push(Val::Null); continue; }
                    let mut scratch = env.clone();
                    let param = lam_param.as_deref();
                    let mut best_idx: usize = 0;
                    let mut best_key = self.exec_lam_body_scratch(
                        key, &items[0], param, &mut scratch)?;
                    for (i, item) in items.iter().enumerate().skip(1) {
                        let k = self.exec_lam_body_scratch(
                            key, item, param, &mut scratch)?;
                        let ord = super::eval::util::cmp_vals(&k, &best_key);
                        let take = if *max {
                            // .last() on sorted asc → last occurrence of max;
                            // ties update to later index.
                            ord != std::cmp::Ordering::Less
                        } else {
                            // .first() → earliest occurrence of min;
                            // strict less only (keep earliest on ties).
                            ord == std::cmp::Ordering::Less
                        };
                        if take { best_idx = i; best_key = k; }
                    }
                    let mut items_vec = Arc::try_unwrap(items).unwrap_or_else(|a| (*a).clone());
                    let winner = std::mem::replace(&mut items_vec[best_idx], Val::Null);
                    stack.push(winner);
                }
                Opcode::MapMap { f1, f2 } => {
                    let recv = pop!(stack);
                    let recv = match recv {
                        Val::StrVec(_) | Val::IntVec(_) | Val::FloatVec(_) => recv.into_arr(),
                        v => v,
                    };
                    if let Val::Arr(a) = recv {
                        // COW fast-path: if the Arc is unique, reuse the Vec
                        // storage (writing mapped values back in place).
                        let mut scratch = env.clone();
                        match Arc::try_unwrap(a) {
                            Ok(mut v) => {
                                for slot in v.iter_mut() {
                                    let prev = scratch.swap_current(std::mem::replace(slot, Val::Null));
                                    let mid = self.exec(f1, &scratch)?;
                                    scratch.swap_current(mid);
                                    let res = self.exec(f2, &scratch)?;
                                    scratch.restore_current(prev);
                                    *slot = res;
                                }
                                stack.push(Val::arr(v));
                            }
                            Err(a) => {
                                let mut out = Vec::with_capacity(a.len());
                                for item in a.iter() {
                                    let prev = scratch.swap_current(item.clone());
                                    let mid = self.exec(f1, &scratch)?;
                                    scratch.swap_current(mid);
                                    let res = self.exec(f2, &scratch)?;
                                    scratch.restore_current(prev);
                                    out.push(res);
                                }
                                stack.push(Val::arr(out));
                            }
                        }
                    } else {
                        stack.push(Val::arr(Vec::new()));
                    }
                }
                Opcode::FindOne(pred) => {
                    let recv = pop!(stack);
                    let mut found: Option<Val> = None;
                    if let Val::Arr(a) = &recv {
                        let mut scratch = env.clone();
                        for item in a.iter() {
                            let prev = scratch.swap_current(item.clone());
                            let keep = is_truthy(&self.exec(pred, &scratch)?);
                            scratch.restore_current(prev);
                            if keep {
                                if found.is_some() {
                                    return err!("quantifier !: expected exactly one match, found multiple");
                                }
                                found = Some(item.clone());
                            }
                        }
                    } else if !recv.is_null() {
                        let sub_env = env.with_current(recv.clone());
                        if is_truthy(&self.exec(pred, &sub_env)?) { found = Some(recv); }
                    }
                    match found {
                        Some(v) => stack.push(v),
                        None => return err!("quantifier !: expected exactly one match, found none"),
                    }
                }

                // ── Ident ─────────────────────────────────────────────────────
                Opcode::LoadIdent(name) => {
                    let v = if let Some(v) = env.get_var(name.as_ref()) {
                        v.clone()
                    } else {
                        env.current.get_field(name.as_ref())
                    };
                    stack.push(v);
                }

                // ── Operators ─────────────────────────────────────────────────
                Opcode::Add  => { let r = pop!(stack); let l = pop!(stack); stack.push(add_vals(l, r)?); }
                Opcode::Sub  => { let r = pop!(stack); let l = pop!(stack); stack.push(num_op(l, r, |a,b|a-b, |a,b|a-b)?); }
                Opcode::Mul  => { let r = pop!(stack); let l = pop!(stack); stack.push(num_op(l, r, |a,b|a*b, |a,b|a*b)?); }
                Opcode::Div  => {
                    let r = pop!(stack); let l = pop!(stack);
                    let b = r.as_f64().unwrap_or(0.0);
                    if b == 0.0 { return err!("division by zero"); }
                    stack.push(Val::Float(l.as_f64().unwrap_or(0.0) / b));
                }
                Opcode::Mod  => { let r = pop!(stack); let l = pop!(stack); stack.push(num_op(l, r, |a,b|a%b, |a,b|a%b)?); }
                Opcode::Eq   => { let r = pop!(stack); let l = pop!(stack); stack.push(Val::Bool(vals_eq(&l,&r))); }
                Opcode::Neq  => { let r = pop!(stack); let l = pop!(stack); stack.push(Val::Bool(!vals_eq(&l,&r))); }
                Opcode::Lt   => { let r = pop!(stack); let l = pop!(stack); stack.push(Val::Bool(cmp_vals(&l,&r) == std::cmp::Ordering::Less)); }
                Opcode::Lte  => { let r = pop!(stack); let l = pop!(stack); stack.push(Val::Bool(cmp_vals(&l,&r) != std::cmp::Ordering::Greater)); }
                Opcode::Gt   => { let r = pop!(stack); let l = pop!(stack); stack.push(Val::Bool(cmp_vals(&l,&r) == std::cmp::Ordering::Greater)); }
                Opcode::Gte  => { let r = pop!(stack); let l = pop!(stack); stack.push(Val::Bool(cmp_vals(&l,&r) != std::cmp::Ordering::Less)); }
                Opcode::Fuzzy => {
                    let r = pop!(stack); let l = pop!(stack);
                    let ls = match &l { Val::Str(s) => s.to_lowercase(), _ => val_to_string(&l).to_lowercase() };
                    let rs = match &r { Val::Str(s) => s.to_lowercase(), _ => val_to_string(&r).to_lowercase() };
                    stack.push(Val::Bool(ls.contains(&rs) || rs.contains(&ls)));
                }
                Opcode::Not  => { let v = pop!(stack); stack.push(Val::Bool(!is_truthy(&v))); }
                Opcode::Neg  => {
                    let v = pop!(stack);
                    stack.push(match v {
                        Val::Int(n) => Val::Int(-n),
                        Val::Float(f) => Val::Float(-f),
                        _ => return err!("unary minus requires a number"),
                    });
                }
                Opcode::CastOp(ty) => {
                    let v = pop!(stack);
                    stack.push(exec_cast(&v, *ty)?);
                }

                // ── Short-circuit ops ─────────────────────────────────────────
                Opcode::AndOp(rhs) => {
                    let lv = pop!(stack);
                    if !is_truthy(&lv) {
                        stack.push(Val::Bool(false));
                    } else {
                        let rv = self.exec(rhs, env)?;
                        stack.push(Val::Bool(is_truthy(&rv)));
                    }
                }
                Opcode::OrOp(rhs) => {
                    let lv = pop!(stack);
                    if is_truthy(&lv) {
                        stack.push(lv);
                    } else {
                        stack.push(self.exec(rhs, env)?);
                    }
                }
                Opcode::CoalesceOp(rhs) => {
                    let lv = pop!(stack);
                    if !lv.is_null() {
                        stack.push(lv);
                    } else {
                        stack.push(self.exec(rhs, env)?);
                    }
                }
                Opcode::IfElse { then_, else_ } => {
                    let cv = pop!(stack);
                    let branch = if is_truthy(&cv) { then_ } else { else_ };
                    stack.push(self.exec(branch, env)?);
                }

                // ── Method calls ──────────────────────────────────────────────
                Opcode::CallMethod(call) => {
                    let recv = pop!(stack);
                    // SIMD fast path: `$..find(@.k == lit)` on root with raw
                    // bytes → scan enclosing objects, skip tree walk.
                    if call.method == BuiltinMethod::Unknown
                        && call.name.as_ref() == "deep_find"
                        && !call.orig_args.is_empty()
                    {
                        if let Some(bytes) = env.raw_bytes.as_ref() {
                            let recv_is_root = match (&recv, &env.root) {
                                (Val::Obj(a), Val::Obj(b)) => Arc::ptr_eq(a, b),
                                (Val::Arr(a), Val::Arr(b)) => Arc::ptr_eq(a, b),
                                _ => false,
                            };
                            if recv_is_root {
                                let tail = &ops_slice[op_idx + 1..];
                                if let Some(conjuncts) =
                                    super::eval::canonical_field_eq_literals(&call.orig_args)
                                {
                                    let spans = if conjuncts.len() == 1 {
                                        super::scan::find_enclosing_objects_eq(
                                            bytes, &conjuncts[0].0, &conjuncts[0].1,
                                        )
                                    } else {
                                        super::scan::find_enclosing_objects_eq_multi(
                                            bytes, &conjuncts,
                                        )
                                    };
                                    let (arr, extra) = materialise_find_scan_spans(
                                        bytes, &spans, tail,
                                    );
                                    stack.push(arr);
                                    skip_ahead = extra;
                                    continue;
                                }
                                // Single-conjunct numeric-range scan.
                                if call.orig_args.len() == 1 {
                                    let e = match &call.orig_args[0] {
                                        super::ast::Arg::Pos(e)
                                        | super::ast::Arg::Named(_, e) => e,
                                    };
                                    if let Some((field, op, thresh)) =
                                        super::eval::canonical_field_cmp_literal(e)
                                    {
                                        let spans = super::scan::find_enclosing_objects_cmp(
                                            bytes, &field, op, thresh,
                                        );
                                        let (arr, extra) = materialise_find_scan_spans(
                                            bytes, &spans, tail,
                                        );
                                        stack.push(arr);
                                        skip_ahead = extra;
                                        continue;
                                    }
                                }
                                // Mixed multi-conjunct (eq + numeric-range).
                                if let Some(conjuncts) =
                                    super::eval::canonical_field_mixed_predicates(&call.orig_args)
                                {
                                    let spans = super::scan::find_enclosing_objects_mixed(
                                        bytes, &conjuncts,
                                    );
                                    let (arr, extra) = materialise_find_scan_spans(
                                        bytes, &spans, tail,
                                    );
                                    stack.push(arr);
                                    skip_ahead = extra;
                                    continue;
                                }
                            }
                        }
                    }
                    let result = self.exec_call(recv, call, env)?;
                    stack.push(result);
                }
                Opcode::CallOptMethod(call) => {
                    let recv = pop!(stack);
                    if recv.is_null() {
                        stack.push(Val::Null);
                    } else {
                        stack.push(self.exec_call(recv, call, env)?);
                    }
                }

                Opcode::MapFString(parts) => {
                    let parts = Arc::clone(parts);
                    let recv = pop!(stack);
                    let recv = if matches!(&recv, Val::StrVec(_) | Val::IntVec(_) | Val::FloatVec(_)) {
                        recv.into_arr()
                    } else { recv };
                    let out_vec: Vec<Val> = if let Val::Arr(a) = &recv {
                        let mut out = Vec::with_capacity(a.len());
                        let mut scratch = env.clone();
                        for item in a.iter() {
                            let prev = scratch.swap_current(item.clone());
                            let result = self.exec_fstring(&parts, &scratch)?;
                            scratch.restore_current(prev);
                            out.push(result);
                        }
                        out
                    } else { Vec::new() };
                    stack.push(Val::arr(out_vec));
                }
                Opcode::MapStrSlice { start, end } => {
                    let v = pop!(stack);
                    let start = *start;
                    let end_opt = *end;
                    // StrVec input: emit columnar StrSliceVec — single Arc
                    // wrapping Vec<StrRef>, no per-row Val enum tag.
                    if let Val::StrVec(a) = &v {
                        let mut out: Vec<crate::strref::StrRef> = Vec::with_capacity(a.len());
                        for s in a.iter() {
                            let src = s.as_ref();
                            if src.is_ascii() {
                                let blen = src.len();
                                let start_u = if start < 0 {
                                    blen.saturating_sub((-start) as usize)
                                } else { start as usize };
                                let end_u = match end_opt {
                                    Some(e) if e < 0 =>
                                        blen.saturating_sub((-e) as usize),
                                    Some(e) => (e as usize).min(blen),
                                    None    => blen,
                                };
                                let start_u = start_u.min(end_u).min(blen);
                                out.push(crate::strref::StrRef::slice(s.clone(), start_u, end_u));
                            } else {
                                // Unicode path — char boundaries.
                                let (start_b, end_b) = slice_unicode_bounds(src, start, end_opt);
                                out.push(crate::strref::StrRef::slice(s.clone(), start_b, end_b));
                            }
                        }
                        stack.push(Val::StrSliceVec(Arc::new(out)));
                        continue;
                    }
                    let out_vec: Vec<Val> = if let Val::Arr(a) = &v {
                        // Homogeneous Str input → emit columnar StrSliceVec.
                        let all_str = a.iter().all(|it| matches!(it, Val::Str(_)));
                        if all_str {
                            let mut out: Vec<crate::strref::StrRef> = Vec::with_capacity(a.len());
                            for item in a.iter() {
                                if let Val::Str(s) = item {
                                    let src = s.as_ref();
                                    if src.is_ascii() {
                                        let blen = src.len();
                                        let start_u = if start < 0 {
                                            blen.saturating_sub((-start) as usize)
                                        } else { start as usize };
                                        let end_u = match end_opt {
                                            Some(e) if e < 0 =>
                                                blen.saturating_sub((-e) as usize),
                                            Some(e) => (e as usize).min(blen),
                                            None    => blen,
                                        };
                                        let start_u = start_u.min(end_u).min(blen);
                                        out.push(crate::strref::StrRef::slice(s.clone(), start_u, end_u));
                                    } else {
                                        let (start_b, end_b) = slice_unicode_bounds(src, start, end_opt);
                                        out.push(crate::strref::StrRef::slice(s.clone(), start_b, end_b));
                                    }
                                }
                            }
                            stack.push(Val::StrSliceVec(Arc::new(out)));
                            continue;
                        }
                        let mut out = Vec::with_capacity(a.len());
                        for item in a.iter() {
                            if let Val::Str(s) = item {
                                let src = s.as_ref();
                                if src.is_ascii() {
                                    let blen = src.len();
                                    let start_u = if start < 0 {
                                        blen.saturating_sub((-start) as usize)
                                    } else { start as usize };
                                    let end_u = match end_opt {
                                        Some(e) if e < 0 =>
                                            blen.saturating_sub((-e) as usize),
                                        Some(e) => (e as usize).min(blen),
                                        None    => blen,
                                    };
                                    let start_u = start_u.min(end_u).min(blen);
                                    if start_u == 0 && end_u == blen {
                                        out.push(Val::Str(s.clone()));
                                    } else {
                                        out.push(Val::StrSlice(
                                            crate::strref::StrRef::slice(
                                                s.clone(), start_u, end_u)
                                        ));
                                    }
                                } else {
                                    // Unicode fallback: char-indices walk.
                                    let mut start_b = src.len();
                                    let mut end_b = src.len();
                                    let mut found_start = false;
                                    let start_want = if start < 0 { 0 } else { start as usize };
                                    let end_want = end_opt.and_then(|e|
                                        if e < 0 { None } else { Some(e as usize) });
                                    for (ci, (bi, _)) in src.char_indices().enumerate() {
                                        if !found_start && ci == start_want {
                                            start_b = bi;
                                            found_start = true;
                                        }
                                        if let Some(ew) = end_want {
                                            if ci == ew { end_b = bi; break; }
                                        }
                                    }
                                    if !found_start { start_b = src.len(); }
                                    if end_want.is_none() { end_b = src.len(); }
                                    if start_b > end_b { start_b = end_b; }
                                    if start_b == 0 && end_b == src.len() {
                                        out.push(Val::Str(s.clone()));
                                    } else {
                                        out.push(Val::StrSlice(
                                            crate::strref::StrRef::slice(
                                                s.clone(), start_b, end_b)
                                        ));
                                    }
                                }
                            } else {
                                out.push(Val::Null);
                            }
                        }
                        out
                    } else { Vec::new() };
                    stack.push(Val::arr(out_vec));
                }
                Opcode::MapProject { keys, ics } => {
                    let recv = pop!(stack);
                    let recv = if matches!(&recv, Val::StrVec(_) | Val::IntVec(_) | Val::FloatVec(_)) {
                        recv.into_arr()
                    } else { recv };
                    // Emit Val::ObjVec — columnar struct-of-arrays with one
                    // shared keys schema and a `Vec<Vec<Val>>` of rows.
                    // No per-row Arc wrapping, no per-row hashtable.
                    if let Val::Arr(a) = &recv {
                        let mut rows: Vec<Vec<Val>> = Vec::with_capacity(a.len());
                        for item in a.iter() {
                            if let Val::Obj(m) = item {
                                let mut row: Vec<Val> = Vec::with_capacity(keys.len());
                                for (i, k) in keys.iter().enumerate() {
                                    row.push(ic_get_field(m, k.as_ref(), &ics[i]));
                                }
                                rows.push(row);
                            } else if let Val::ObjSmall(ps) = item {
                                let mut row: Vec<Val> = Vec::with_capacity(keys.len());
                                for k in keys.iter() {
                                    let mut v = Val::Null;
                                    for (kk, vv) in ps.iter() {
                                        if kk.as_ref() == k.as_ref() {
                                            v = vv.clone();
                                            break;
                                        }
                                    }
                                    row.push(v);
                                }
                                rows.push(row);
                            } else {
                                rows.push(vec![Val::Null; keys.len()]);
                            }
                        }
                        stack.push(Val::ObjVec(Arc::new(super::eval::value::ObjVecData {
                            keys: Arc::clone(keys),
                            rows,
                        })));
                    } else {
                        stack.push(Val::arr(Vec::new()));
                    }
                }

                // ── Construction ──────────────────────────────────────────────
                Opcode::MakeObj(entries) => {
                    let entries = Arc::clone(entries);
                    let result = self.exec_make_obj(&entries, env)?;
                    stack.push(result);
                }
                Opcode::MakeArr(progs) => {
                    let progs = Arc::clone(progs);
                    let mut out = Vec::with_capacity(progs.len());
                    for p in progs.iter() {
                        let v = self.exec(p, env)?;
                        // If the program produces an array from a spread,
                        // check if it was tagged; for simplicity, just push.
                        out.push(v);
                    }
                    stack.push(Val::arr(out));
                }

                // ── F-string ──────────────────────────────────────────────────
                Opcode::FString(parts) => {
                    let parts = Arc::clone(parts);
                    let result = self.exec_fstring(&parts, env)?;
                    stack.push(result);
                }

                // ── Kind check ────────────────────────────────────────────────
                Opcode::KindCheck { ty, negate } => {
                    let v = pop!(stack);
                    let m = kind_matches(&v, *ty);
                    stack.push(Val::Bool(if *negate { !m } else { m }));
                }

                // ── Pipeline helpers ──────────────────────────────────────────
                Opcode::SetCurrent => {
                    // This is emitted before an arbitrary pipe-forward expression.
                    // The value flowing through the pipe is now on the stack as TOS.
                    // However we can't mutate env here since env is immutable.
                    // The actual "set current" happens by the caller preparing a new env.
                    // In practice, SetCurrent should not appear in isolation in the
                    // flat opcode stream because pipeline steps that need SetCurrent
                    // are compiled as sub-programs. Skip.
                }
                Opcode::BindVar(name) => {
                    // TOS becomes a named var; TOS remains (pass-through for ->) .
                    // We can't mutate env here. This is handled at the pipeline level.
                    // For now, just keep TOS.
                    let _ = name;
                }
                Opcode::StoreVar(name) => {
                    // Pop and discard (the LetExpr opcode handles binding properly).
                    let _ = name;
                    pop!(stack);
                }
                Opcode::BindObjDestructure(_) | Opcode::BindArrDestructure(_) => {
                    // Pipeline bind destructure — handled at pipeline level.
                }

                // ── Complex recursive ops ─────────────────────────────────────
                Opcode::LetExpr { name, body } => {
                    let init_val = pop!(stack);
                    let body_env = env.with_var(name.as_ref(), init_val);
                    stack.push(self.exec(body, &body_env)?);
                }

                Opcode::ListComp(spec) => {
                    let items = self.exec_iter_vals(&spec.iter, env)?;
                    let mut out = Vec::with_capacity(items.len());
                    // Fast path: single-var `for x in iter` — reuse one
                    // scratch Env via push_lam / pop_lam instead of
                    // cloning per iteration.
                    if spec.vars.len() == 1 {
                        let vname = spec.vars[0].clone();
                        let mut scratch = env.clone();
                        for item in items {
                            let frame = scratch.push_lam(Some(vname.as_ref()), item);
                            let keep = match &spec.cond {
                                Some(c) => is_truthy(&self.exec(c, &scratch)?),
                                None => true,
                            };
                            if keep {
                                let v = self.exec(&spec.expr, &scratch)?;
                                out.push(v);
                            }
                            scratch.pop_lam(frame);
                        }
                    } else {
                        for item in items {
                            let ie = bind_comp_vars(env, &spec.vars, item);
                            if let Some(cond) = &spec.cond {
                                if !is_truthy(&self.exec(cond, &ie)?) { continue; }
                            }
                            out.push(self.exec(&spec.expr, &ie)?);
                        }
                    }
                    stack.push(Val::arr(out));
                }

                Opcode::DictComp(spec) => {
                    let items = self.exec_iter_vals(&spec.iter, env)?;
                    let mut map: IndexMap<Arc<str>, Val> = IndexMap::with_capacity(items.len());
                    // Single-var fast path: reuse scratch Env via push_lam/pop_lam
                    // instead of cloning per iteration.
                    if spec.vars.len() == 1 {
                        let vname = spec.vars[0].clone();
                        // Hot pattern: `{ f(x): x for x in iter }` with no
                        // condition.  Both key and val depend only on `x`,
                        // which is also `current`, so we can elide the
                        // env rebind and the two `self.exec` calls by
                        // dispatching the common shapes inline.
                        let val_is_ident = matches!(
                            spec.val.ops.as_ref(),
                            [Opcode::LoadIdent(v)] if v.as_ref() == vname.as_ref()
                        );
                        let key_shape = classify_dict_key(&spec.key, vname.as_ref());
                        if spec.cond.is_none() && val_is_ident && key_shape.is_some() {
                            let shape = key_shape.unwrap();
                            for item in items {
                                let k: Arc<str> = match shape {
                                    DictKeyShape::Ident => match &item {
                                        Val::Str(s) => s.clone(),
                                        other       => Arc::<str>::from(val_to_key(other)),
                                    },
                                    DictKeyShape::IdentToString => match &item {
                                        Val::Str(s) => s.clone(),
                                        Val::Int(n)   => Arc::<str>::from(n.to_string()),
                                        Val::Float(f) => Arc::<str>::from(f.to_string()),
                                        Val::Bool(b)  => Arc::<str>::from(if *b { "true" } else { "false" }),
                                        Val::Null     => Arc::<str>::from("null"),
                                        other         => Arc::<str>::from(val_to_key(other)),
                                    },
                                };
                                map.insert(k, item);
                            }
                            stack.push(Val::obj(map));
                            continue;
                        }
                        let mut scratch = env.clone();
                        for item in items {
                            let frame = scratch.push_lam(Some(vname.as_ref()), item);
                            let keep = match &spec.cond {
                                Some(c) => is_truthy(&self.exec(c, &scratch)?),
                                None    => true,
                            };
                            if keep {
                                let k: Arc<str> = match self.exec(&spec.key, &scratch)? {
                                    Val::Str(s) => s,
                                    other       => Arc::<str>::from(val_to_key(&other)),
                                };
                                let v = self.exec(&spec.val, &scratch)?;
                                map.insert(k, v);
                            }
                            scratch.pop_lam(frame);
                        }
                    } else {
                        for item in items {
                            let ie = bind_comp_vars(env, &spec.vars, item);
                            if let Some(cond) = &spec.cond {
                                if !is_truthy(&self.exec(cond, &ie)?) { continue; }
                            }
                            let k: Arc<str> = match self.exec(&spec.key, &ie)? {
                                Val::Str(s) => s,
                                other       => Arc::<str>::from(val_to_key(&other)),
                            };
                            let v = self.exec(&spec.val, &ie)?;
                            map.insert(k, v);
                        }
                    }
                    stack.push(Val::obj(map));
                }

                Opcode::SetComp(spec) => {
                    let items = self.exec_iter_vals(&spec.iter, env)?;
                    let mut seen: std::collections::HashSet<String> =
                        std::collections::HashSet::with_capacity(items.len());
                    let mut out = Vec::with_capacity(items.len());
                    if spec.vars.len() == 1 {
                        let vname = spec.vars[0].clone();
                        let mut scratch = env.clone();
                        for item in items {
                            let frame = scratch.push_lam(Some(vname.as_ref()), item);
                            let keep = match &spec.cond {
                                Some(c) => is_truthy(&self.exec(c, &scratch)?),
                                None    => true,
                            };
                            if keep {
                                let v = self.exec(&spec.expr, &scratch)?;
                                let k = val_to_key(&v);
                                if seen.insert(k) { out.push(v); }
                            }
                            scratch.pop_lam(frame);
                        }
                    } else {
                        for item in items {
                            let ie = bind_comp_vars(env, &spec.vars, item);
                            if let Some(cond) = &spec.cond {
                                if !is_truthy(&self.exec(cond, &ie)?) { continue; }
                            }
                            let v = self.exec(&spec.expr, &ie)?;
                            let k = val_to_key(&v);
                            if seen.insert(k) { out.push(v); }
                        }
                    }
                    stack.push(Val::arr(out));
                }

                // ── Path cache lookup ─────────────────────────────────────────
                Opcode::GetPointer(ptr) => {
                    let doc_hash = self.doc_hash;
                    let v = if let Some(cached) = self.path_cache.get(doc_hash, ptr.as_ref()) {
                        cached
                    } else {
                        let v = resolve_pointer(&env.root, ptr.as_ref());
                        self.path_cache.insert(doc_hash, ptr.clone(), v.clone());
                        v
                    };
                    stack.push(v);
                }

                // ── Patch block ───────────────────────────────────────────────
                Opcode::PatchEval(e) => {
                    stack.push(eval(e, env)?);
                }
            }
        }

        stack.pop().ok_or_else(|| EvalError("program produced no value".into()))
    }

    // ── Helper: single-pass numeric min/max over filter.map ───────────────────
    /// Shared body for `FilterMapMin` / `FilterMapMax`.  `is_min=true` keeps the
    /// smallest value, `false` keeps the largest.  Factored out of the main
    /// exec match so the hot dispatch loop stays lean.
    #[cold]
    #[inline(never)]
    fn filter_map_minmax(
        &mut self,
        recv: Val,
        pred: &Program,
        map:  &Program,
        env:  &Env,
        is_min: bool,
    ) -> Result<Val, EvalError> {
        let mut best_i: Option<i64> = None;
        let mut best_f: Option<f64> = None;
        let better = |new: f64, old: f64| if is_min { new < old } else { new > old };
        let better_i = |new: i64, old: i64| if is_min { new < old } else { new > old };
        let label = if is_min { "min" } else { "max" };
        if let Val::Arr(a) = &recv {
            let mut scratch = env.clone();
            for item in a.iter() {
                let prev = scratch.swap_current(item.clone());
                if !is_truthy(&self.exec(pred, &scratch)?) {
                    scratch.restore_current(prev);
                    continue;
                }
                let v = self.exec(map, &scratch)?;
                scratch.restore_current(prev);
                match v {
                    Val::Int(n) => {
                        if let Some(bf) = best_f {
                            if better(n as f64, bf) { best_f = Some(n as f64); }
                        } else if let Some(bi) = best_i {
                            if better_i(n, bi) { best_i = Some(n); }
                        } else { best_i = Some(n); }
                    }
                    Val::Float(x) => {
                        if best_f.is_none() {
                            best_f = Some(best_i.map(|i| i as f64).unwrap_or(x));
                            best_i = None;
                        }
                        if better(x, best_f.unwrap()) { best_f = Some(x); }
                    }
                    Val::Null => {}
                    _ => return Err(EvalError(format!(
                        "filter(..).map(..).{}(): non-numeric mapped value", label))),
                }
            }
        }
        Ok(match (best_i, best_f) {
            (_, Some(x)) => Val::Float(x),
            (Some(i), _) => Val::Int(i),
            _ => Val::Null,
        })
    }

    // ── Method call dispatch ──────────────────────────────────────────────────

    fn exec_call(&mut self, recv: Val, call: &CompiledCall, env: &Env) -> Result<Val, EvalError> {
        // Global-call opcodes push Root before calling; handle them
        if call.method == BuiltinMethod::Unknown {
            // Custom registry or global function
            if !env.registry_is_empty() {
                if let Some(method) = env.registry_get(call.name.as_ref()) {
                    let evaled: Result<Vec<Val>, _> = call.sub_progs.iter()
                        .map(|p| self.exec(p, env)).collect();
                    return method.call(recv, &evaled?);
                }
            }
            // Global function (coalesce, zip, etc.) or fallback to dispatch_method
            return dispatch_method(recv, call.name.as_ref(), &call.orig_args, env);
        }

        // Lambda methods — VM handles iteration, running sub-programs per item
        if call.method.is_lambda_method() {
            return self.exec_lambda_method(recv, call, env);
        }

        // Typed-numeric aggregate fast-path: bare `.sum()/.min()/.max()/.avg()`
        // on an array.  Skips registry dispatch + `collect_nums` extra Vec.
        if call.sub_progs.is_empty() && call.orig_args.is_empty() {
            if let Val::Arr(a) = &recv {
                match call.method {
                    BuiltinMethod::Sum => return Ok(agg_sum_typed(a)),
                    BuiltinMethod::Avg => return Ok(agg_avg_typed(a)),
                    BuiltinMethod::Min => return Ok(agg_minmax_typed(a, false)),
                    BuiltinMethod::Max => return Ok(agg_minmax_typed(a, true)),
                    _ => {}
                }
            }
            // Columnar IntVec — pure i64 tight loops, native parity.
            if let Val::IntVec(a) = &recv {
                match call.method {
                    BuiltinMethod::Sum => {
                        let s: i64 = a.iter().fold(0i64, |a, b| a.wrapping_add(*b));
                        return Ok(Val::Int(s));
                    }
                    BuiltinMethod::Avg => {
                        if a.is_empty() { return Ok(Val::Null); }
                        let s: f64 = a.iter().map(|n| *n as f64).sum();
                        return Ok(Val::Float(s / a.len() as f64));
                    }
                    BuiltinMethod::Min => {
                        return Ok(a.iter().copied().min().map(Val::Int).unwrap_or(Val::Null));
                    }
                    BuiltinMethod::Max => {
                        return Ok(a.iter().copied().max().map(Val::Int).unwrap_or(Val::Null));
                    }
                    BuiltinMethod::Count | BuiltinMethod::Len => {
                        return Ok(Val::Int(a.len() as i64));
                    }
                    _ => {}
                }
            }
            // Homogeneous-Int `Val::Arr` receivers get routed through
            // columnar reverse/sort — 3x less memory bandwidth than the
            // Vec<Val> path.  Clone cost is identical (O(N) in both cases
            // because the Arr is typically shared); the win is on write.
            if let Val::Arr(a) = &recv {
                let is_all_int = a.iter().all(|v| matches!(v, Val::Int(_)));
                if is_all_int && !a.is_empty() {
                    match call.method {
                        BuiltinMethod::Reverse => {
                            let mut v: Vec<i64> = a.iter().map(|x|
                                if let Val::Int(n) = x { *n } else { 0 }
                            ).collect();
                            v.reverse();
                            return Ok(Val::int_vec(v));
                        }
                        BuiltinMethod::Sort => {
                            let mut v: Vec<i64> = a.iter().map(|x|
                                if let Val::Int(n) = x { *n } else { 0 }
                            ).collect();
                            v.sort_unstable();
                            return Ok(Val::int_vec(v));
                        }
                        BuiltinMethod::Sum => {
                            let s: i64 = a.iter().fold(0i64, |acc, v|
                                if let Val::Int(n) = v { acc.wrapping_add(*n) } else { acc });
                            return Ok(Val::Int(s));
                        }
                        _ => {}
                    }
                }
            }
            // Columnar IntVec receiver — reverse/sort in-place on Vec<i64>.
            if let Val::IntVec(a) = &recv {
                match call.method {
                    BuiltinMethod::Reverse => {
                        let mut v: Vec<i64> = Arc::try_unwrap(a.clone()).unwrap_or_else(|a| (*a).clone());
                        v.reverse();
                        return Ok(Val::int_vec(v));
                    }
                    BuiltinMethod::Sort => {
                        let mut v: Vec<i64> = Arc::try_unwrap(a.clone()).unwrap_or_else(|a| (*a).clone());
                        v.sort_unstable();
                        return Ok(Val::int_vec(v));
                    }
                    _ => {}
                }
            }
            if let Val::FloatVec(a) = &recv {
                match call.method {
                    BuiltinMethod::Sum => {
                        let s: f64 = a.iter().sum();
                        return Ok(Val::Float(s));
                    }
                    BuiltinMethod::Avg => {
                        if a.is_empty() { return Ok(Val::Null); }
                        let s: f64 = a.iter().sum();
                        return Ok(Val::Float(s / a.len() as f64));
                    }
                    BuiltinMethod::Min => {
                        let mut best: Option<f64> = None;
                        for &f in a.iter() {
                            best = Some(match best { Some(b) => if f < b { f } else { b }, None => f });
                        }
                        return Ok(best.map(Val::Float).unwrap_or(Val::Null));
                    }
                    BuiltinMethod::Max => {
                        let mut best: Option<f64> = None;
                        for &f in a.iter() {
                            best = Some(match best { Some(b) => if f > b { f } else { b }, None => f });
                        }
                        return Ok(best.map(Val::Float).unwrap_or(Val::Null));
                    }
                    BuiltinMethod::Count | BuiltinMethod::Len => {
                        return Ok(Val::Int(a.len() as i64));
                    }
                    _ => {}
                }
            }
            // Bare `.flatten()` (depth=1) — inline depth-1 flatten with exact
            // preallocation.  Skips `dispatch_method` + its arg-parse path.
            if call.method == BuiltinMethod::Flatten {
                if let Val::Arr(a) = &recv {
                    // Columnar fast-path: all inners Int-only → emit Val::IntVec.
                    let all_int_inner = a.iter().all(|it| match it {
                        Val::IntVec(_) => true,
                        Val::Arr(inner) => inner.iter().all(|v| matches!(v, Val::Int(_))),
                        Val::Int(_) => true,
                        _ => false,
                    });
                    if all_int_inner {
                        let cap: usize = a.iter().map(|it| match it {
                            Val::IntVec(inner) => inner.len(),
                            Val::Arr(inner)    => inner.len(),
                            _ => 1,
                        }).sum();
                        let mut out: Vec<i64> = Vec::with_capacity(cap);
                        for item in a.iter() {
                            match item {
                                Val::IntVec(inner) => out.extend(inner.iter().copied()),
                                Val::Arr(inner)    => out.extend(inner.iter().filter_map(|v| v.as_i64())),
                                Val::Int(n)        => out.push(*n),
                                _ => {}
                            }
                        }
                        return Ok(Val::int_vec(out));
                    }
                    let cap: usize = a.iter().map(|it| match it {
                        Val::Arr(inner) => inner.len(),
                        Val::IntVec(inner) => inner.len(),
                        Val::FloatVec(inner) => inner.len(),
                        _ => 1,
                    }).sum();
                    let mut out = Vec::with_capacity(cap);
                    for item in a.iter() {
                        match item {
                            Val::Arr(inner) => out.extend(inner.iter().cloned()),
                            Val::IntVec(inner) => out.extend(inner.iter().map(|n| Val::Int(*n))),
                            Val::FloatVec(inner) => out.extend(inner.iter().map(|f| Val::Float(*f))),
                            other => out.push(other.clone()),
                        }
                    }
                    return Ok(Val::arr(out));
                }
                // Columnar receiver itself — already flat; return as-is.
                if matches!(&recv, Val::IntVec(_) | Val::FloatVec(_)) {
                    return Ok(recv);
                }
            }
            // Scalar `.to_string()` — inline the conversion.  The registry
            // path allocates a fresh `Val::Str` via `val_to_string`; this
            // does the same work without the dispatch + argslice copy.
            if call.method == BuiltinMethod::ToString {
                let s: Arc<str> = match &recv {
                    Val::Str(s)   => return Ok(Val::Str(s.clone())),
                    Val::Int(n)   => Arc::from(n.to_string()),
                    Val::Float(f) => Arc::from(f.to_string()),
                    Val::Bool(b)  => Arc::from(b.to_string()),
                    Val::Null     => Arc::from("null"),
                    other         => Arc::from(super::eval::util::val_to_string(other).as_str()),
                };
                return Ok(Val::Str(s));
            }
            // Scalar `.to_json()` — skip the `Val -> serde_json::Value ->
            // String` round-trip for primitives.  Big-ticket users are
            // `.map(@.to_json())` in hot pipelines.
            if call.method == BuiltinMethod::ToJson {
                match &recv {
                    Val::Int(n)   => return Ok(Val::Str(Arc::from(n.to_string()))),
                    Val::Float(f) => {
                        let s = if f.is_finite() { f.to_string() } else { "null".to_string() };
                        return Ok(Val::Str(Arc::from(s)));
                    }
                    Val::Bool(b)  => return Ok(Val::Str(Arc::from(if *b { "true" } else { "false" }))),
                    Val::Null     => return Ok(Val::Str(Arc::from("null"))),
                    Val::Str(s) => {
                        // JSON-escape — fall through for now; cheap escape
                        // added here to keep the fast-path.  Handles the
                        // common no-escape case with a single scan.
                        let src = s.as_ref();
                        let mut needs_escape = false;
                        for &b in src.as_bytes() {
                            if b < 0x20 || b == b'"' || b == b'\\' { needs_escape = true; break; }
                        }
                        if !needs_escape {
                            let mut out = String::with_capacity(src.len() + 2);
                            out.push('"'); out.push_str(src); out.push('"');
                            return Ok(Val::Str(Arc::from(out)));
                        }
                        // Fall through to serde path for escape handling.
                    }
                    _ => {}
                }
            }
        }

        // Value methods — delegate to the existing dispatch with orig_args
        dispatch_method(recv, call.name.as_ref(), &call.orig_args, env)
    }

    fn exec_lambda_method(&mut self, recv: Val, call: &CompiledCall, env: &Env) -> Result<Val, EvalError> {
        let sub = call.sub_progs.first();
        // Hoist the lambda param name out of the per-item loop — otherwise
        // each iteration would re-scan `orig_args` for the Lambda pattern.
        let lam_param: Option<&str> = match call.orig_args.first() {
            Some(Arg::Pos(Expr::Lambda { params, .. })) if !params.is_empty() =>
                Some(params[0].as_str()),
            _ => None,
        };
        // Single scratch env per call — reused across every item iteration
        // below via `push_lam` / `pop_lam` instead of a fresh clone per item.
        let mut scratch = env.clone();

        match call.method {
            BuiltinMethod::Filter => {
                let pred = sub.ok_or_else(|| EvalError("filter: requires predicate".into()))?;
                let items = recv.into_vec().ok_or_else(|| EvalError("filter: expected array".into()))?;
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    if is_truthy(&self.exec_lam_body_scratch(pred, &item, lam_param, &mut scratch)?) {
                        out.push(item);
                    }
                }
                Ok(Val::arr(out))
            }
            BuiltinMethod::Map => {
                let mapper = sub.ok_or_else(|| EvalError("map: requires mapper".into()))?;
                let items = recv.into_vec().ok_or_else(|| EvalError("map: expected array".into()))?;
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    out.push(self.exec_lam_body_scratch(mapper, &item, lam_param, &mut scratch)?);
                }
                Ok(Val::arr(out))
            }
            BuiltinMethod::FlatMap => {
                let mapper = sub.ok_or_else(|| EvalError("flatMap: requires mapper".into()))?;
                let items = recv.into_vec().ok_or_else(|| EvalError("flatMap: expected array".into()))?;
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    match self.exec_lam_body_scratch(mapper, &item, lam_param, &mut scratch)? {
                        Val::Arr(a) => out.extend(Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone())),
                        v => out.push(v),
                    }
                }
                Ok(Val::arr(out))
            }
            BuiltinMethod::Sort => {
                // Delegate to func_arrays::sort which already handles lambda args
                dispatch_method(recv, call.name.as_ref(), &call.orig_args, env)
            }
            BuiltinMethod::Any => {
                if let Val::Arr(a) = &recv {
                    let pred = sub.ok_or_else(|| EvalError("any: requires predicate".into()))?;
                    for item in a.iter() {
                        if is_truthy(&self.exec_lam_body_scratch(pred, item, lam_param, &mut scratch)?) {
                            return Ok(Val::Bool(true));
                        }
                    }
                    Ok(Val::Bool(false))
                } else { Ok(Val::Bool(false)) }
            }
            BuiltinMethod::All => {
                if let Val::Arr(a) = &recv {
                    if a.is_empty() { return Ok(Val::Bool(true)); }
                    let pred = sub.ok_or_else(|| EvalError("all: requires predicate".into()))?;
                    for item in a.iter() {
                        if !is_truthy(&self.exec_lam_body_scratch(pred, item, lam_param, &mut scratch)?) {
                            return Ok(Val::Bool(false));
                        }
                    }
                    Ok(Val::Bool(true))
                } else { Ok(Val::Bool(false)) }
            }
            BuiltinMethod::Count if !call.sub_progs.is_empty() => {
                if let Val::Arr(a) = &recv {
                    let pred = &call.sub_progs[0];
                    let mut n: i64 = 0;
                    for item in a.iter() {
                        if is_truthy(&self.exec_lam_body_scratch(pred, item, lam_param, &mut scratch)?) {
                            n += 1;
                        }
                    }
                    Ok(Val::Int(n))
                } else { Ok(Val::Int(0)) }
            }
            BuiltinMethod::GroupBy => {
                let key_prog = sub.ok_or_else(|| EvalError("groupBy: requires key".into()))?;
                let items = recv.into_vec().ok_or_else(|| EvalError("groupBy: expected array".into()))?;
                // Pattern specialisation: `lambda x: x % K` where K is a small
                // positive Int.  Skips per-item exec + string-keying.  Uses a
                // dense Vec<Vec<Val>> indexed by (n % K).rem_euclid — collapsed
                // into an IndexMap<Arc<str>, Val> once at the end.
                if let Some(param) = lam_param {
                    if let [Opcode::LoadIdent(p), Opcode::PushInt(k_lit), Opcode::Mod]
                        = key_prog.ops.as_ref()
                    {
                        if p.as_ref() == param && *k_lit > 0 && *k_lit <= 4096 {
                            let k_lit = *k_lit;
                            let k_u = k_lit as usize;
                            let mut buckets: Vec<Vec<Val>> = vec![Vec::new(); k_u];
                            let mut seen: Vec<bool> = vec![false; k_u];
                            let mut order: Vec<usize> = Vec::new();
                            // All-numeric fast path; error on non-numeric
                            // (matches tree-walker `x % K` dispatch).
                            for item in items {
                                let idx = match &item {
                                    Val::Int(n)   => n.rem_euclid(k_lit) as usize,
                                    Val::Float(x) => (x.trunc() as i64).rem_euclid(k_lit) as usize,
                                    _ => return err!("group_by(x % K): non-numeric item"),
                                };
                                if !seen[idx] { seen[idx] = true; order.push(idx); }
                                buckets[idx].push(item);
                            }
                            let mut map: IndexMap<Arc<str>, Val> = IndexMap::with_capacity(order.len());
                            for idx in order {
                                let k: Arc<str> = Arc::from(idx.to_string());
                                let bucket = std::mem::take(&mut buckets[idx]);
                                map.insert(k, Val::arr(bucket));
                            }
                            return Ok(Val::obj(map));
                        }
                    }
                }
                // General compiled-bytecode path.
                let mut map: IndexMap<Arc<str>, Val> = IndexMap::new();
                for item in items {
                    let k: Arc<str> = Arc::from(val_to_key(&self.exec_lam_body_scratch(key_prog, &item, lam_param, &mut scratch)?).as_str());
                    let bucket = map.entry(k).or_insert_with(|| Val::arr(Vec::new()));
                    bucket.as_array_mut().unwrap().push(item);
                }
                Ok(Val::obj(map))
            }
            BuiltinMethod::CountBy => {
                let key_prog = sub.ok_or_else(|| EvalError("countBy: requires key".into()))?;
                let items = recv.into_vec().ok_or_else(|| EvalError("countBy: expected array".into()))?;
                let mut map: IndexMap<Arc<str>, Val> = IndexMap::new();
                for item in items {
                    let k: Arc<str> = Arc::from(val_to_key(&self.exec_lam_body_scratch(key_prog, &item, lam_param, &mut scratch)?).as_str());
                    let counter = map.entry(k).or_insert(Val::Int(0));
                    if let Val::Int(n) = counter { *n += 1; }
                }
                Ok(Val::obj(map))
            }
            BuiltinMethod::IndexBy => {
                let key_prog = sub.ok_or_else(|| EvalError("indexBy: requires key".into()))?;
                let items = recv.into_vec().ok_or_else(|| EvalError("indexBy: expected array".into()))?;
                let mut map: IndexMap<Arc<str>, Val> = IndexMap::new();
                for item in items {
                    let k: Arc<str> = Arc::from(val_to_key(&self.exec_lam_body_scratch(key_prog, &item, lam_param, &mut scratch)?).as_str());
                    map.insert(k, item);
                }
                Ok(Val::obj(map))
            }
            BuiltinMethod::TakeWhile => {
                let pred = sub.ok_or_else(|| EvalError("takeWhile: requires predicate".into()))?;
                let items = recv.into_vec().ok_or_else(|| EvalError("takeWhile: expected array".into()))?;
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    if !is_truthy(&self.exec_lam_body_scratch(pred, &item, lam_param, &mut scratch)?) { break; }
                    out.push(item);
                }
                Ok(Val::arr(out))
            }
            BuiltinMethod::DropWhile => {
                let pred = sub.ok_or_else(|| EvalError("dropWhile: requires predicate".into()))?;
                let items = recv.into_vec().ok_or_else(|| EvalError("dropWhile: expected array".into()))?;
                let mut dropping = true;
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    if dropping {
                        let still_drop = is_truthy(&self.exec_lam_body_scratch(pred, &item, lam_param, &mut scratch)?);
                        if still_drop { continue; }
                        dropping = false;
                    }
                    out.push(item);
                }
                Ok(Val::arr(out))
            }
            BuiltinMethod::Accumulate => {
                // VM-accelerate the common 2-param form `accumulate(lambda a, x: …)`
                // with no `start:` named arg.  Pattern-specialise for a handful
                // of tight-loop shapes (`a + x`, `a - x`, `a * x`, `max/min`)
                // that the compiled bytecode would otherwise re-dispatch per
                // iteration.  Other shapes fall through to a compiled-bytecode
                // VM loop; unsupported shapes fall back to the tree-walker.
                let lam_body = sub.ok_or_else(|| EvalError("accumulate: requires lambda".into()))?;
                let (p1, p2) = match call.orig_args.first() {
                    Some(Arg::Pos(Expr::Lambda { params, .. })) if params.len() >= 2 =>
                        (params[0].as_str(), params[1].as_str()),
                    _ => return dispatch_method(recv, call.name.as_ref(), &call.orig_args, env),
                };
                if call.orig_args.iter().any(|a|
                    matches!(a, Arg::Named(n, _) if n.as_str() == "start"))
                {
                    return dispatch_method(recv, call.name.as_ref(), &call.orig_args, env);
                }
                // Try pattern specialisation: `LoadIdent(p1), LoadIdent(p2), <BinOp>`.
                let specialised_binop = match lam_body.ops.as_ref() {
                    [Opcode::LoadIdent(a), Opcode::LoadIdent(b), op]
                        if a.as_ref() == p1 && b.as_ref() == p2 =>
                    {
                        match op {
                            Opcode::Add => Some(AccumOp::Add),
                            Opcode::Sub => Some(AccumOp::Sub),
                            Opcode::Mul => Some(AccumOp::Mul),
                            _           => None,
                        }
                    }
                    _ => None,
                };
                // Columnar IntVec input → IntVec output (native-parity).
                if let (Val::IntVec(a), Some(bop)) = (&recv, specialised_binop.as_ref().copied()) {
                    let mut out: Vec<i64> = Vec::with_capacity(a.len());
                    let mut acc: i64 = 0;
                    let mut first = true;
                    for &n in a.iter() {
                        if first { acc = n; first = false; }
                        else { acc = match bop {
                            AccumOp::Add => acc.wrapping_add(n),
                            AccumOp::Sub => acc.wrapping_sub(n),
                            AccumOp::Mul => acc.wrapping_mul(n),
                        }; }
                        out.push(acc);
                    }
                    return Ok(Val::int_vec(out));
                }
                if let (Val::FloatVec(a), Some(bop)) = (&recv, specialised_binop.as_ref().copied()) {
                    let mut out: Vec<f64> = Vec::with_capacity(a.len());
                    let mut acc: f64 = 0.0;
                    let mut first = true;
                    for &n in a.iter() {
                        if first { acc = n; first = false; }
                        else { acc = match bop {
                            AccumOp::Add => acc + n,
                            AccumOp::Sub => acc - n,
                            AccumOp::Mul => acc * n,
                        }; }
                        out.push(acc);
                    }
                    return Ok(Val::float_vec(out));
                }
                let items = recv.into_vec()
                    .ok_or_else(|| EvalError("accumulate: expected array".into()))?;
                let mut out = Vec::with_capacity(items.len());
                if let Some(bop) = specialised_binop {
                    // Typed i64 tight-loop when every item is Int.  No Val
                    // match, no add_vals dispatch, no per-step Val clone —
                    // native parity.
                    if items.iter().all(|v| matches!(v, Val::Int(_))) {
                        let mut acc_out: Vec<i64> = Vec::with_capacity(items.len());
                        let mut acc: i64 = 0;
                        let mut first = true;
                        for item in &items {
                            let n = if let Val::Int(n) = item { *n } else { unreachable!() };
                            if first { acc = n; first = false; }
                            else { acc = match bop {
                                AccumOp::Add => acc.wrapping_add(n),
                                AccumOp::Sub => acc.wrapping_sub(n),
                                AccumOp::Mul => acc.wrapping_mul(n),
                            }; }
                            acc_out.push(acc);
                        }
                        return Ok(Val::int_vec(acc_out));
                    }
                    // Typed f64 tight-loop when every item is Float.
                    if items.iter().all(|v| matches!(v, Val::Float(_))) {
                        let mut acc_out: Vec<f64> = Vec::with_capacity(items.len());
                        let mut acc: f64 = 0.0;
                        let mut first = true;
                        for item in &items {
                            let n = if let Val::Float(n) = item { *n } else { unreachable!() };
                            if first { acc = n; first = false; }
                            else { acc = match bop {
                                AccumOp::Add => acc + n,
                                AccumOp::Sub => acc - n,
                                AccumOp::Mul => acc * n,
                            }; }
                            acc_out.push(acc);
                        }
                        return Ok(Val::float_vec(acc_out));
                    }
                    // Mixed / non-numeric — inline fold via add_vals/num_op.
                    let mut running: Option<Val> = None;
                    for item in items {
                        let next = match running.take() {
                            Some(acc) => match bop {
                                AccumOp::Add => add_vals(acc, item)?,
                                AccumOp::Sub => num_op(acc, item, |a,b| a-b, |a,b| a-b)?,
                                AccumOp::Mul => num_op(acc, item, |a,b| a*b, |a,b| a*b)?,
                            },
                            None => item,
                        };
                        out.push(next.clone());
                        running = Some(next);
                    }
                    return Ok(Val::arr(out));
                }
                // General path: compiled-bytecode VM loop.
                let mut running: Option<Val> = None;
                for item in items {
                    let next = if let Some(acc) = running.take() {
                        let f1 = scratch.push_lam(Some(p1), acc);
                        let f2 = scratch.push_lam(Some(p2), item.clone());
                        let r = self.exec(lam_body, &scratch)?;
                        scratch.pop_lam(f2);
                        scratch.pop_lam(f1);
                        r
                    } else {
                        item
                    };
                    out.push(next.clone());
                    running = Some(next);
                }
                Ok(Val::arr(out))
            }
            BuiltinMethod::Partition => {
                let pred = sub.ok_or_else(|| EvalError("partition: requires predicate".into()))?;
                let items = recv.into_vec().ok_or_else(|| EvalError("partition: expected array".into()))?;
                let (mut yes, mut no) = (Vec::with_capacity(items.len()), Vec::with_capacity(items.len()));
                for item in items {
                    if is_truthy(&self.exec_lam_body_scratch(pred, &item, lam_param, &mut scratch)?) {
                        yes.push(item);
                    } else {
                        no.push(item);
                    }
                }
                Ok(Val::arr(vec![Val::arr(yes), Val::arr(no)]))
            }
            BuiltinMethod::TransformKeys => {
                let lam = sub.ok_or_else(|| EvalError("transformKeys: requires lambda".into()))?;
                let map = recv.into_map().ok_or_else(|| EvalError("transformKeys: expected object".into()))?;
                let mut out: IndexMap<Arc<str>, Val> = IndexMap::new();
                for (k, v) in map {
                    let new_key = Arc::from(val_to_key(&self.exec_lam_body_scratch(lam, &Val::Str(k), lam_param, &mut scratch)?).as_str());
                    out.insert(new_key, v);
                }
                Ok(Val::obj(out))
            }
            BuiltinMethod::TransformValues => {
                let lam = sub.ok_or_else(|| EvalError("transformValues: requires lambda".into()))?;
                // COW: if the receiver's Arc is unique we mutate in place,
                // otherwise deep-clone once and mutate the clone.  Either
                // way, no fresh IndexMap allocation and no key-Arc clone
                // per entry (IndexMap::values_mut preserves slot identity).
                let mut map = recv.into_map().ok_or_else(|| EvalError("transformValues: expected object".into()))?;
                // Pattern-specialise `[PushCurrent, PushInt(K), <BinOp>]` —
                // the body of `transform_values(@ + K)` / `(@ - K)` / `(@ * K)`.
                let pat = match lam.ops.as_ref() {
                    [Opcode::PushCurrent, Opcode::PushInt(k), op] => match op {
                        Opcode::Add => Some((AccumOp::Add, *k)),
                        Opcode::Sub => Some((AccumOp::Sub, *k)),
                        Opcode::Mul => Some((AccumOp::Mul, *k)),
                        _ => None,
                    },
                    _ => None,
                };
                if let Some((op, k_lit)) = pat {
                    let kf = k_lit as f64;
                    for v in map.values_mut() {
                        match v {
                            Val::Int(n) => *n = match op {
                                AccumOp::Add => n.wrapping_add(k_lit),
                                AccumOp::Sub => n.wrapping_sub(k_lit),
                                AccumOp::Mul => n.wrapping_mul(k_lit),
                            },
                            Val::Float(x) => *x = match op {
                                AccumOp::Add => *x + kf,
                                AccumOp::Sub => *x - kf,
                                AccumOp::Mul => *x * kf,
                            },
                            _ => {
                                let new = self.exec_lam_body_scratch(lam, v, lam_param, &mut scratch)?;
                                *v = new;
                            }
                        }
                    }
                    return Ok(Val::obj(map));
                }
                // General path — mutate in place via values_mut; no new map,
                // no key reinsertion.
                for v in map.values_mut() {
                    let new = self.exec_lam_body_scratch(lam, v, lam_param, &mut scratch)?;
                    *v = new;
                }
                Ok(Val::obj(map))
            }
            BuiltinMethod::FilterKeys => {
                let lam = sub.ok_or_else(|| EvalError("filterKeys: requires predicate".into()))?;
                let map = recv.into_map().ok_or_else(|| EvalError("filterKeys: expected object".into()))?;
                let mut out: IndexMap<Arc<str>, Val> = IndexMap::new();
                for (k, v) in map {
                    if is_truthy(&self.exec_lam_body_scratch(lam, &Val::Str(k.clone()), lam_param, &mut scratch)?) {
                        out.insert(k, v);
                    }
                }
                Ok(Val::obj(out))
            }
            BuiltinMethod::FilterValues => {
                let lam = sub.ok_or_else(|| EvalError("filterValues: requires predicate".into()))?;
                let map = recv.into_map().ok_or_else(|| EvalError("filterValues: expected object".into()))?;
                let mut out: IndexMap<Arc<str>, Val> = IndexMap::new();
                for (k, v) in map {
                    if is_truthy(&self.exec_lam_body_scratch(lam, &v, lam_param, &mut scratch)?) {
                        out.insert(k, v);
                    }
                }
                Ok(Val::obj(out))
            }
            BuiltinMethod::Pivot => {
                dispatch_method(recv, call.name.as_ref(), &call.orig_args, env)
            }
            BuiltinMethod::Update => {
                let lam = sub.ok_or_else(|| EvalError("update: requires lambda".into()))?;
                self.exec_lam_body(lam, &recv, lam_param, env)
            }
            _ => dispatch_method(recv, call.name.as_ref(), &call.orig_args, env),
        }
    }

    /// Convenience wrapper: clones `env` once, runs the prog, discards
    /// the scratch.  Hot loops should use `exec_lam_body_scratch` to
    /// reuse a single scratch env instead.
    fn exec_lam_body(&mut self, prog: &Program, item: &Val, lam_param: Option<&str>, env: &Env)
        -> Result<Val, EvalError>
    {
        let mut scratch = env.clone();
        self.exec_lam_body_scratch(prog, item, lam_param, &mut scratch)
    }

    /// Scratch-reusing variant: mutates `scratch` in place via
    /// `Env::push_lam` / `Env::pop_lam` instead of cloning per item.
    /// The caller provides (and reuses) the scratch env across loop
    /// iterations.
    fn exec_lam_body_scratch(&mut self, prog: &Program, item: &Val,
                              lam_param: Option<&str>, scratch: &mut Env)
        -> Result<Val, EvalError>
    {
        let frame = scratch.push_lam(lam_param, item.clone());
        let r = self.exec(prog, scratch);
        scratch.pop_lam(frame);
        r
    }

    // ── Object construction ───────────────────────────────────────────────────

    fn exec_make_obj(&mut self, entries: &[CompiledObjEntry], env: &Env) -> Result<Val, EvalError> {
        let mut map: IndexMap<Arc<str>, Val> = IndexMap::with_capacity(entries.len());
        for entry in entries {
            match entry {
                CompiledObjEntry::Short { name, ic } => {
                    let v = if let Some(v) = env.get_var(name.as_ref()) {
                        v.clone()
                    } else if let Val::Obj(m) = &env.current {
                        ic_get_field(m, name.as_ref(), ic)
                    } else {
                        env.current.get_field(name.as_ref())
                    };
                    if !v.is_null() { map.insert(name.clone(), v); }
                }
                CompiledObjEntry::Kv { key, prog, optional, cond } => {
                    if let Some(c) = cond {
                        if !super::eval::util::is_truthy(&self.exec(c, env)?) { continue; }
                    }
                    let v = self.exec(prog, env)?;
                    if *optional && v.is_null() { continue; }
                    map.insert(key.clone(), v);
                }
                CompiledObjEntry::KvPath { key, steps, optional, ics } => {
                    let mut v = env.current.clone();
                    for (i, st) in steps.iter().enumerate() {
                        v = match st {
                            KvStep::Field(f) => {
                                if let Val::Obj(m) = &v {
                                    ic_get_field(m, f.as_ref(), &ics[i])
                                } else {
                                    v.get_field(f.as_ref())
                                }
                            }
                            KvStep::Index(i) => v.get_index(*i),
                        };
                        if v.is_null() { break; }
                    }
                    if *optional && v.is_null() { continue; }
                    map.insert(key.clone(), v);
                }
                CompiledObjEntry::Dynamic { key, val } => {
                    let k: Arc<str> = Arc::from(val_to_key(&self.exec(key, env)?).as_str());
                    let v = self.exec(val, env)?;
                    map.insert(k, v);
                }
                CompiledObjEntry::Spread(prog) => {
                    if let Val::Obj(other) = self.exec(prog, env)? {
                        let entries = Arc::try_unwrap(other).unwrap_or_else(|m| (*m).clone());
                        for (k, v) in entries { map.insert(k, v); }
                    }
                }
                CompiledObjEntry::SpreadDeep(prog) => {
                    if let Val::Obj(other) = self.exec(prog, env)? {
                        let base = std::mem::take(&mut map);
                        let merged = super::eval::util::deep_merge_concat(
                            Val::obj(base), Val::Obj(other));
                        if let Val::Obj(m) = merged {
                            map = Arc::try_unwrap(m).unwrap_or_else(|m| (*m).clone());
                        }
                    }
                }
            }
        }
        Ok(Val::obj(map))
    }

    // ── F-string ──────────────────────────────────────────────────────────────

    fn exec_fstring(&mut self, parts: &[CompiledFSPart], env: &Env) -> Result<Val, EvalError> {
        use std::fmt::Write as _;
        // Pre-size by summing literal lengths + rough 8 bytes per interp.
        let lit_len: usize = parts.iter().map(|p| match p {
            CompiledFSPart::Lit(s) => s.len(),
            CompiledFSPart::Interp { .. } => 8,
        }).sum();
        let mut out = String::with_capacity(lit_len);
        for part in parts {
            match part {
                CompiledFSPart::Lit(s) => out.push_str(s.as_ref()),
                CompiledFSPart::Interp { prog, fmt } => {
                    // Fast path: very small interp programs that just extract
                    // a field / index / current — avoid full self.exec() recursion
                    // (stack alloc, ic vec, etc.) per row.
                    let val: Val = match &prog.ops[..] {
                        [Opcode::PushCurrent] => env.current.clone(),
                        [Opcode::PushCurrent, Opcode::GetIndex(n)] => {
                            match &env.current {
                                Val::Arr(a) => {
                                    let idx = if *n >= 0 { *n as usize }
                                              else { a.len().saturating_sub((-*n) as usize) };
                                    a.get(idx).cloned().unwrap_or(Val::Null)
                                }
                                _ => self.exec(prog, env)?,
                            }
                        }
                        [Opcode::PushCurrent, Opcode::GetField(k)] => {
                            match &env.current {
                                Val::Obj(m) => m.get(k.as_ref()).cloned().unwrap_or(Val::Null),
                                _ => self.exec(prog, env)?,
                            }
                        }
                        [Opcode::LoadIdent(name)] => env.get_var(name).cloned().unwrap_or(Val::Null),
                        _ => self.exec(prog, env)?,
                    };
                    match fmt {
                        None => match &val {
                            // Fast paths: avoid val_to_string's temporary String.
                            Val::Str(s)   => out.push_str(s.as_ref()),
                            Val::Int(n)   => { let _ = write!(out, "{}", n); }
                            Val::Float(f) => { let _ = write!(out, "{}", f); }
                            Val::Bool(b)  => { let _ = write!(out, "{}", b); }
                            Val::Null     => out.push_str("null"),
                            _             => out.push_str(&val_to_string(&val)),
                        },
                        Some(FmtSpec::Spec(spec)) => {
                            out.push_str(&apply_fmt_spec(&val, spec));
                        }
                        Some(FmtSpec::Pipe(method)) => {
                            let piped = dispatch_method(val, method, &[], env)?;
                            match &piped {
                                Val::Str(s)   => out.push_str(s.as_ref()),
                                Val::Int(n)   => { let _ = write!(out, "{}", n); }
                                Val::Float(f) => { let _ = write!(out, "{}", f); }
                                Val::Bool(b)  => { let _ = write!(out, "{}", b); }
                                Val::Null     => out.push_str("null"),
                                _             => out.push_str(&val_to_string(&piped)),
                            }
                        }
                    }
                }
            }
        }
        // `Arc::<str>::from(String)` transfers the buffer — no realloc.
        Ok(Val::Str(Arc::<str>::from(out)))
    }

    // ── Comprehension helpers ─────────────────────────────────────────────────

    fn exec_iter_vals(&mut self, iter_prog: &Program, env: &Env) -> Result<Vec<Val>, EvalError> {
        match self.exec(iter_prog, env)? {
            Val::Arr(a) => Ok(Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone())),
            Val::Obj(m) => {
                let entries = Arc::try_unwrap(m).unwrap_or_else(|m| (*m).clone());
                Ok(entries.into_iter().map(|(k, v)| obj2("key", Val::Str(k), "value", v)).collect())
            }
            other => Ok(vec![other]),
        }
    }
}

// ── Env extensions for VM ─────────────────────────────────────────────────────

impl Env {
    fn registry_is_empty(&self) -> bool { self.registry_ref().is_empty() }
    fn registry_get(&self, name: &str) -> Option<Arc<dyn super::eval::methods::Method>> {
        self.registry_ref().get(name).cloned()
    }
}

// ── Free helpers ──────────────────────────────────────────────────────────────

/// Route C — chained-descendant byte chain.
///
/// Entry: `root_key` is the key of the root `Descendant` that triggered the
/// scan.  `tail` is the opcode slice immediately after that `Descendant`.
///
/// Consumes as many tail opcodes as can be handled on byte spans:
///   - `Descendant(k)` — re-scan within each current span.
///   - `Quantifier(First)` — keep first span; `Quantifier(One)` when exactly one.
///   - `InlineFilter(pred)` / `CallMethod(Filter, [pred])` with a canonical
///     equality literal (int/string/bool/null) — bytewise retain.
///
/// Any other opcode terminates the chain; remaining spans are materialised
/// into `Val`s and returned, and the caller resumes normal opcode dispatch.
/// A "first-selector" opcode: bare `.first()` / `Quantifier::First`.
/// When `Descendant(k)` is followed by one of these, the byte scan
/// can stop at the first match per span.
fn is_first_selector_op(op: &Opcode) -> bool {
    match op {
        Opcode::Quantifier(QuantifierKind::First) => true,
        Opcode::CallMethod(c)
            if c.sub_progs.is_empty() && c.method == BuiltinMethod::First => true,
        _ => false,
    }
}

/// Materialise byte-scan `spans` into a `Val::Arr`, peeking at `tail`
/// for a trailing `.map(<field>)` (compiled to `Opcode::MapField`).
/// When present the direct child is extracted from each span without
/// paying a full `serde_json::from_slice` on the enclosing object.
/// Returned `skip` tells the caller how many opcodes were consumed
/// beyond the current `CallMethod`.
///
/// Outlined (`#[cold] #[inline(never)]`) to keep the `exec` dispatch
/// match compact — inlining the span loop here was enough to cost
/// Q4/Q13 icache footprint in `bench_lock`.
#[cold]
#[inline(never)]
fn materialise_find_scan_spans(
    bytes: &[u8],
    spans: &[super::scan::ValueSpan],
    tail: &[Opcode],
) -> (Val, usize) {
    // Consume any leading span-refiner tail ops — compiled forms of
    // `.filter(@.k == lit)` / `.filter(@.k <cmp> lit)` sitting after the
    // byte-scan.  Each narrows `spans` in place via `find_direct_field`
    // + bytewise / numeric comparison; the remaining tail then feeds the
    // existing map/aggregate dispatch below.
    let mut consumed_refiners = 0usize;
    let mut owned: Option<Vec<super::scan::ValueSpan>>;
    let mut spans_view: &[super::scan::ValueSpan] = spans;
    loop {
        match tail.get(consumed_refiners) {
            Some(Opcode::FilterFieldEqLit(k, lit_val)) => {
                let Some(lit) = val_to_canonical_lit_bytes(lit_val) else { break };
                let next: Vec<_> = spans_view.iter().copied().filter(|s| {
                    let obj = &bytes[s.start..s.end];
                    match super::scan::find_direct_field(obj, k.as_ref()) {
                        Some(vs) => vs.end - vs.start == lit.len()
                            && obj[vs.start..vs.end] == lit[..],
                        None => false,
                    }
                }).collect();
                owned = Some(next);
                spans_view = owned.as_deref().unwrap();
                consumed_refiners += 1;
            }
            Some(Opcode::FilterFieldCmpLit(k, op, lit_val)) => {
                let Some(thresh) = lit_val.as_f64() else { break };
                let holds: fn(f64, f64) -> bool = match op {
                    super::ast::BinOp::Lt  => |a, b| a <  b,
                    super::ast::BinOp::Lte => |a, b| a <= b,
                    super::ast::BinOp::Gt  => |a, b| a >  b,
                    super::ast::BinOp::Gte => |a, b| a >= b,
                    _ => break,
                };
                let next: Vec<_> = spans_view.iter().copied().filter(|s| {
                    let obj = &bytes[s.start..s.end];
                    let Some(vs) = super::scan::find_direct_field(obj, k.as_ref())
                        else { return false };
                    match super::scan::parse_num_span(&obj[vs.start..vs.end]) {
                        Some((_, f, _)) => holds(f, thresh),
                        None => false,
                    }
                }).collect();
                owned = Some(next);
                spans_view = owned.as_deref().unwrap();
                consumed_refiners += 1;
            }
            _ => break,
        }
    }
    let spans = spans_view;
    let tail = &tail[consumed_refiners..];
    let (val, inner) = materialise_find_scan_spans_tail(bytes, spans, tail);
    (val, consumed_refiners + inner)
}

/// Convert a `Val` literal to its canonical JSON byte encoding — the
/// same form `find_direct_field` produces when it locates a scalar value
/// inside an object.  Returns `None` for non-scalar / non-canonical values.
#[inline]
fn val_to_canonical_lit_bytes(v: &Val) -> Option<Vec<u8>> {
    match v {
        Val::Int(n)   => Some(n.to_string().into_bytes()),
        Val::Bool(b)  => Some(if *b { b"true".to_vec() } else { b"false".to_vec() }),
        Val::Null     => Some(b"null".to_vec()),
        Val::Str(s)   => serde_json::to_vec(
            &serde_json::Value::String(s.to_string())
        ).ok(),
        _ => None,
    }
}

#[cold]
#[inline(never)]
fn materialise_find_scan_spans_tail(
    bytes: &[u8],
    spans: &[super::scan::ValueSpan],
    tail: &[Opcode],
) -> (Val, usize) {
    // Trailing `.count()` / `.len()` — just the span count, no parse.
    if let Some(Opcode::CallMethod(c)) = tail.first() {
        if c.sub_progs.is_empty()
            && matches!(c.method, BuiltinMethod::Count | BuiltinMethod::Len)
        {
            return (Val::Int(spans.len() as i64), 1);
        }
    }
    // Trailing fused `.filter(@.kp op lit).map(kproj)` — peephole fused
    // forms `FilterFieldEqLitMapField` / `FilterFieldCmpLitMapField`.
    // Refine spans by predicate, then project direct-field values; peek
    // for a numeric aggregate right after to fold on the projection.
    if let Some(op) = tail.first() {
        let refined = match op {
            Opcode::FilterFieldEqLitMapField(kp, lit_v, kproj) => {
                let lit = val_to_canonical_lit_bytes(lit_v);
                lit.map(|lit| {
                    let spans2: Vec<_> = spans.iter().copied().filter(|s| {
                        let obj = &bytes[s.start..s.end];
                        match super::scan::find_direct_field(obj, kp.as_ref()) {
                            Some(vs) => vs.end - vs.start == lit.len()
                                && obj[vs.start..vs.end] == lit[..],
                            None => false,
                        }
                    }).collect();
                    (spans2, kproj.clone())
                })
            }
            Opcode::FilterFieldCmpLitMapField(kp, cop, lit_v, kproj) => {
                let thresh_opt = lit_v.as_f64();
                let holds_opt: Option<fn(f64, f64) -> bool> = match cop {
                    super::ast::BinOp::Lt  => Some(|a, b| a <  b),
                    super::ast::BinOp::Lte => Some(|a, b| a <= b),
                    super::ast::BinOp::Gt  => Some(|a, b| a >  b),
                    super::ast::BinOp::Gte => Some(|a, b| a >= b),
                    _ => None,
                };
                match (thresh_opt, holds_opt) {
                    (Some(thresh), Some(holds)) => {
                        let spans2: Vec<_> = spans.iter().copied().filter(|s| {
                            let obj = &bytes[s.start..s.end];
                            let Some(vs) = super::scan::find_direct_field(obj, kp.as_ref())
                                else { return false };
                            match super::scan::parse_num_span(&obj[vs.start..vs.end]) {
                                Some((_, f, _)) => holds(f, thresh),
                                None => false,
                            }
                        }).collect();
                        Some((spans2, kproj.clone()))
                    }
                    _ => None,
                }
            }
            _ => None,
        };
        if let Some((spans2, k)) = refined {
            // Aggregate fold on the projection without materialising.
            if let Some(Opcode::CallMethod(c)) = tail.get(1) {
                if c.sub_progs.is_empty() {
                    match c.method {
                        BuiltinMethod::Count | BuiltinMethod::Len => {
                            let f = super::scan::fold_direct_field_nums(bytes, &spans2, k.as_ref());
                            return (Val::Int(f.count as i64), 2);
                        }
                        BuiltinMethod::Sum => {
                            let f = super::scan::fold_direct_field_nums(bytes, &spans2, k.as_ref());
                            let v = if f.count == 0 { Val::Int(0) }
                                    else if f.is_float { Val::Float(f.float_sum) }
                                    else { Val::Int(f.int_sum) };
                            return (v, 2);
                        }
                        BuiltinMethod::Avg => {
                            let f = super::scan::fold_direct_field_nums(bytes, &spans2, k.as_ref());
                            let v = if f.count == 0 { Val::Null }
                                    else { Val::Float(f.float_sum / f.count as f64) };
                            return (v, 2);
                        }
                        BuiltinMethod::Min => {
                            let f = super::scan::fold_direct_field_nums(bytes, &spans2, k.as_ref());
                            let v = if !f.any { Val::Null }
                                    else if f.is_float { Val::Float(f.min_f) }
                                    else { Val::Int(f.min_i) };
                            return (v, 2);
                        }
                        BuiltinMethod::Max => {
                            let f = super::scan::fold_direct_field_nums(bytes, &spans2, k.as_ref());
                            let v = if !f.any { Val::Null }
                                    else if f.is_float { Val::Float(f.max_f) }
                                    else { Val::Int(f.max_i) };
                            return (v, 2);
                        }
                        _ => {}
                    }
                }
            }
            let mut vals: Vec<Val> = Vec::with_capacity(spans2.len());
            for s in &spans2 {
                let obj_bytes = &bytes[s.start..s.end];
                let v = match super::scan::find_direct_field(obj_bytes, k.as_ref()) {
                    Some(vs) => serde_json::from_slice::<serde_json::Value>(
                        &obj_bytes[vs.start..vs.end],
                    ).ok().map(|sv| Val::from(&sv)).unwrap_or(Val::Null),
                    None => Val::Null,
                };
                vals.push(v);
            }
            return (Val::arr(vals), 1);
        }
    }
    // Trailing `.map(<field>)` — peek once more for a numeric aggregate
    // that can fold straight from the per-span direct field, skipping
    // both the full-object parse and the Val array construction.
    if let Some(Opcode::MapField(k)) = tail.first() {
        if let Some(Opcode::CallMethod(c)) = tail.get(1) {
            if c.sub_progs.is_empty() {
                match c.method {
                    BuiltinMethod::Count | BuiltinMethod::Len => {
                        // count of extracted fields == count of spans
                        // where the key parses successfully.  Fold gives
                        // us that as `f.count`.
                        let f = super::scan::fold_direct_field_nums(bytes, spans, k.as_ref());
                        return (Val::Int(f.count as i64), 2);
                    }
                    BuiltinMethod::Sum => {
                        let f = super::scan::fold_direct_field_nums(bytes, spans, k.as_ref());
                        let v = if f.count == 0 { Val::Int(0) }
                                else if f.is_float { Val::Float(f.float_sum) }
                                else { Val::Int(f.int_sum) };
                        return (v, 2);
                    }
                    BuiltinMethod::Avg => {
                        let f = super::scan::fold_direct_field_nums(bytes, spans, k.as_ref());
                        let v = if f.count == 0 { Val::Null }
                                else { Val::Float(f.float_sum / f.count as f64) };
                        return (v, 2);
                    }
                    BuiltinMethod::Min => {
                        let f = super::scan::fold_direct_field_nums(bytes, spans, k.as_ref());
                        let v = if !f.any { Val::Null }
                                else if f.is_float { Val::Float(f.min_f) }
                                else { Val::Int(f.min_i) };
                        return (v, 2);
                    }
                    BuiltinMethod::Max => {
                        let f = super::scan::fold_direct_field_nums(bytes, spans, k.as_ref());
                        let v = if !f.any { Val::Null }
                                else if f.is_float { Val::Float(f.max_f) }
                                else { Val::Int(f.max_i) };
                        return (v, 2);
                    }
                    _ => {}
                }
            }
        }
        let mut vals: Vec<Val> = Vec::with_capacity(spans.len());
        for s in spans {
            let obj_bytes = &bytes[s.start..s.end];
            let v = match super::scan::find_direct_field(obj_bytes, k.as_ref()) {
                Some(vs) => serde_json::from_slice::<serde_json::Value>(
                    &obj_bytes[vs.start..vs.end],
                ).ok().map(|sv| Val::from(&sv)).unwrap_or(Val::Null),
                None => Val::Null,
            };
            vals.push(v);
        }
        return (Val::arr(vals), 1);
    }
    let mut vals: Vec<Val> = Vec::with_capacity(spans.len());
    for s in spans {
        if let Ok(v) = serde_json::from_slice::<serde_json::Value>(
            &bytes[s.start..s.end],
        ) {
            vals.push(Val::from(&v));
        }
    }
    (Val::arr(vals), 0)
}

fn byte_chain_exec(
    bytes: &[u8],
    root_key: &str,
    tail: &[Opcode],
) -> (Val, usize) {
    // Early-exit: `$..key.first()` / `$..key!` needs only the first match.
    let first_after_initial = tail.first().map(is_first_selector_op).unwrap_or(false);
    let mut spans: Vec<super::scan::ValueSpan> = if first_after_initial {
        super::scan::find_first_key_value_span(bytes, root_key)
            .into_iter().collect()
    } else {
        super::scan::find_key_value_spans(bytes, root_key)
    };
    let mut scalar = false;
    let mut consumed = 0usize;

    for (idx, op) in tail.iter().enumerate() {
        match op {
            Opcode::Descendant(k) => {
                let next_first = tail.get(idx + 1)
                    .map(is_first_selector_op).unwrap_or(false);
                let mut next = Vec::with_capacity(spans.len());
                for s in &spans {
                    let sub = &bytes[s.start..s.end];
                    if next_first {
                        if let Some(s2) = super::scan::find_first_key_value_span(sub, k.as_ref()) {
                            next.push(super::scan::ValueSpan {
                                start: s.start + s2.start,
                                end:   s.start + s2.end,
                            });
                        }
                    } else {
                        for s2 in super::scan::find_key_value_spans(sub, k.as_ref()) {
                            next.push(super::scan::ValueSpan {
                                start: s.start + s2.start,
                                end:   s.start + s2.end,
                            });
                        }
                    }
                }
                spans = next;
                scalar = false;
            }
            Opcode::Quantifier(QuantifierKind::First) => {
                spans.truncate(1);
                scalar = true;
            }
            Opcode::Quantifier(QuantifierKind::One) => {
                if spans.len() != 1 { break; }
                scalar = true;
            }
            // `.first()` / `.last()` compile to `CallMethod(First|Last)` not
            // `Quantifier`.  Handle them as scalar terminators.
            Opcode::CallMethod(call) if call.sub_progs.is_empty() => {
                match call.method {
                    BuiltinMethod::First => {
                        spans.truncate(1);
                        scalar = true;
                    }
                    BuiltinMethod::Last => {
                        if let Some(last) = spans.pop() { spans = vec![last]; }
                        scalar = true;
                    }
                    _ => break,
                }
            }
            Opcode::InlineFilter(prog) => {
                match canonical_eq_literal_from_program(prog) {
                    Some(lit) => {
                        spans.retain(|s| {
                            s.end - s.start == lit.len()
                                && &bytes[s.start..s.end] == &lit[..]
                        });
                        scalar = false;
                    }
                    None => break,
                }
            }
            Opcode::CallMethod(call)
                if call.method == BuiltinMethod::Filter && call.sub_progs.len() == 1 =>
            {
                match canonical_eq_literal_from_program(&call.sub_progs[0]) {
                    Some(lit) => {
                        spans.retain(|s| {
                            s.end - s.start == lit.len()
                                && &bytes[s.start..s.end] == &lit[..]
                        });
                        scalar = false;
                    }
                    None => break,
                }
            }
            _ => break,
        }
        consumed += 1;
    }

    // Numeric-fold fast path: trailing `.sum()/.avg()/.min()/.max()/.count()/.len()`
    // — skip Val materialisation, parse numbers inline from byte spans.
    if !scalar {
        if let Some(Opcode::CallMethod(call)) = tail.get(consumed) {
            if call.sub_progs.is_empty() && tail.len() == consumed + 1 {
                match call.method {
                    BuiltinMethod::Count | BuiltinMethod::Len => {
                        return (Val::Int(spans.len() as i64), consumed + 1);
                    }
                    BuiltinMethod::Sum => {
                        let f = super::scan::fold_nums(bytes, &spans);
                        let v = if f.count == 0 { Val::Int(0) }
                            else if f.is_float { Val::Float(f.float_sum) }
                            else { Val::Int(f.int_sum) };
                        return (v, consumed + 1);
                    }
                    BuiltinMethod::Avg => {
                        let f = super::scan::fold_nums(bytes, &spans);
                        let v = if f.count == 0 { Val::Null }
                            else { Val::Float(f.float_sum / f.count as f64) };
                        return (v, consumed + 1);
                    }
                    BuiltinMethod::Min => {
                        let f = super::scan::fold_nums(bytes, &spans);
                        let v = if !f.any { Val::Null }
                            else if f.is_float { Val::Float(f.min_f) }
                            else { Val::Int(f.min_i) };
                        return (v, consumed + 1);
                    }
                    BuiltinMethod::Max => {
                        let f = super::scan::fold_nums(bytes, &spans);
                        let v = if !f.any { Val::Null }
                            else if f.is_float { Val::Float(f.max_f) }
                            else { Val::Int(f.max_i) };
                        return (v, consumed + 1);
                    }
                    _ => {}
                }
            }
        }
    }

    let mut materialised: Vec<Val> = Vec::with_capacity(spans.len());
    for s in &spans {
        if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes[s.start..s.end]) {
            materialised.push(Val::from(&v));
        }
    }

    let out = if scalar {
        materialised.into_iter().next().unwrap_or(Val::Null)
    } else {
        Val::arr(materialised)
    };
    (out, consumed)
}

/// Recognise `@ == lit` / `lit == @` compiled to a sub-program.  Matches
/// the shape `[PushCurrent, Push<Lit>, Eq]` or `[Push<Lit>, PushCurrent, Eq]`
/// and returns the canonical-serialised literal bytes.  Floats rejected
/// (representation variance vs `1` / `1.0`).
fn canonical_eq_literal_from_program(prog: &Program) -> Option<Vec<u8>> {
    if prog.ops.len() != 3 { return None; }
    if !matches!(prog.ops[2], Opcode::Eq) { return None; }
    let (lit_op, has_current) = match (&prog.ops[0], &prog.ops[1]) {
        (Opcode::PushCurrent, lit) => (lit, true),
        (lit, Opcode::PushCurrent) => (lit, true),
        _ => (&prog.ops[0], false),
    };
    if !has_current { return None; }
    match lit_op {
        Opcode::PushInt(n)  => Some(n.to_string().into_bytes()),
        Opcode::PushBool(b) => Some(if *b { b"true".to_vec() } else { b"false".to_vec() }),
        Opcode::PushNull    => Some(b"null".to_vec()),
        Opcode::PushStr(s)  => serde_json::to_vec(&serde_json::Value::String(s.to_string())).ok(),
        _ => None,
    }
}

fn exec_slice(v: Val, from: Option<i64>, to: Option<i64>) -> Val {
    match v {
        Val::Arr(a) => {
            let len = a.len() as i64;
            let s = resolve_idx(from.unwrap_or(0), len);
            let e = resolve_idx(to.unwrap_or(len), len);
            let items = Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone());
            let s = s.min(items.len());
            let e = e.min(items.len());
            Val::arr(items[s..e].to_vec())
        }
        Val::IntVec(a) => {
            let len = a.len() as i64;
            let s = resolve_idx(from.unwrap_or(0), len);
            let e = resolve_idx(to.unwrap_or(len), len);
            let items = Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone());
            let s = s.min(items.len());
            let e = e.min(items.len());
            Val::int_vec(items[s..e].to_vec())
        }
        Val::FloatVec(a) => {
            let len = a.len() as i64;
            let s = resolve_idx(from.unwrap_or(0), len);
            let e = resolve_idx(to.unwrap_or(len), len);
            let items = Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone());
            let s = s.min(items.len());
            let e = e.min(items.len());
            Val::float_vec(items[s..e].to_vec())
        }
        _ => Val::Null,
    }
}

fn resolve_idx(i: i64, len: i64) -> usize {
    (if i < 0 { (len + i).max(0) } else { i }) as usize
}

fn collect_desc(v: &Val, name: &str, out: &mut Vec<Val>) {
    match v {
        Val::Obj(m) => {
            if let Some(v) = m.get(name) { out.push(v.clone()); }
            for v in m.values() { collect_desc(v, name, out); }
        }
        Val::Arr(a) => { for item in a.as_ref() { collect_desc(item, name, out); } }
        _ => {}
    }
}

/// Early-exit variant of `collect_desc`: returns the first self-first DFS
/// hit for `name`, matching the order that `collect_desc` would produce.
/// Powers the `$..key.first()` fast path when raw JSON bytes aren't
/// available (SIMD `byte_chain_exec` handles the raw-bytes case).
fn find_desc_first(v: &Val, name: &str) -> Option<Val> {
    match v {
        Val::Obj(m) => {
            if let Some(v) = m.get(name) { return Some(v.clone()); }
            for child in m.values() {
                if let Some(hit) = find_desc_first(child, name) { return Some(hit); }
            }
            None
        }
        Val::Arr(a) => {
            for item in a.as_ref() {
                if let Some(hit) = find_desc_first(item, name) { return Some(hit); }
            }
            None
        }
        _ => None,
    }
}

fn collect_all(v: &Val, out: &mut Vec<Val>) {
    match v {
        Val::Obj(m) => {
            out.push(v.clone());
            for child in m.values() { collect_all(child, out); }
        }
        Val::Arr(a) => {
            for item in a.as_ref() { collect_all(item, out); }
        }
        other => out.push(other.clone()),
    }
}

/// Path-tracking variant — called when descending from root so paths are
/// root-relative and can be cached for future `RootChain` lookups.
/// `prefix` is mutated in-place (push/truncate) to avoid allocations.
fn collect_desc_with_paths(
    v: &Val, name: &str, prefix: &mut String,
    out: &mut Vec<Val>, cached: &mut Vec<(Arc<str>, Val)>,
) {
    match v {
        Val::Obj(m) => {
            if let Some(found) = m.get(name) {
                let mut path = prefix.clone();
                path.push('/');
                path.push_str(name);
                out.push(found.clone());
                cached.push((Arc::from(path.as_str()), found.clone()));
            }
            for (k, child) in m.iter() {
                let prev = prefix.len();
                prefix.push('/');
                prefix.push_str(k.as_ref());
                collect_desc_with_paths(child, name, prefix, out, cached);
                prefix.truncate(prev);
            }
        }
        Val::Arr(a) => {
            for (i, item) in a.iter().enumerate() {
                let prev = prefix.len();
                prefix.push('/');
                let idx = i.to_string();
                prefix.push_str(&idx);
                collect_desc_with_paths(item, name, prefix, out, cached);
                prefix.truncate(prev);
            }
        }
        _ => {}
    }
}

fn resolve_pointer(root: &Val, ptr: &str) -> Val {
    let mut cur = root.clone();
    for seg in ptr.split('/').filter(|s| !s.is_empty()) {
        cur = cur.get_field(seg);
    }
    cur
}

#[derive(Clone, Copy)]
enum DictKeyShape {
    /// key prog is `[LoadIdent(v)]` — use item directly (stringify non-Str).
    Ident,
    /// key prog is `[LoadIdent(v), CallMethod{ToString, no args}]`.
    IdentToString,
}

fn classify_dict_key(prog: &Program, vname: &str) -> Option<DictKeyShape> {
    match prog.ops.as_ref() {
        [Opcode::LoadIdent(v)] if v.as_ref() == vname => Some(DictKeyShape::Ident),
        [Opcode::LoadIdent(v), Opcode::CallMethod(call)]
            if v.as_ref() == vname
                && call.method == BuiltinMethod::ToString
                && call.sub_progs.is_empty()
                && call.orig_args.is_empty() =>
            Some(DictKeyShape::IdentToString),
        _ => None,
    }
}

fn bind_comp_vars(env: &Env, vars: &[Arc<str>], item: Val) -> Env {
    match vars {
        [] => env.with_current(item),
        [v] => { let mut e = env.with_var(v.as_ref(), item.clone()); e.current = item; e }
        [v1, v2, ..] => {
            let idx = item.get("index").cloned().unwrap_or(Val::Null);
            let val = item.get("value").cloned().unwrap_or_else(|| item.clone());
            let mut e = env.with_var(v1.as_ref(), idx).with_var(v2.as_ref(), val.clone());
            e.current = val;
            e
        }
    }
}

fn exec_cast(v: &Val, ty: super::ast::CastType) -> Result<Val, EvalError> {
    use super::ast::CastType;
    match ty {
        CastType::Str => Ok(Val::Str(Arc::from(match v {
            Val::Null     => "null".to_string(),
            Val::Bool(b)  => b.to_string(),
            Val::Int(n)   => n.to_string(),
            Val::Float(f) => f.to_string(),
            Val::Str(s)   => s.to_string(),
            other         => super::eval::util::val_to_string(other),
        }.as_str()))),
        CastType::Bool => Ok(Val::Bool(match v {
            Val::Null         => false,
            Val::Bool(b)      => *b,
            Val::Int(n)       => *n != 0,
            Val::Float(f)     => *f != 0.0,
            Val::Str(s)       => !s.is_empty(),
            Val::StrSlice(r)  => !r.is_empty(),
            Val::Arr(a)       => !a.is_empty(),
            Val::IntVec(a)    => !a.is_empty(),
            Val::FloatVec(a)  => !a.is_empty(),
            Val::StrVec(a)       => !a.is_empty(),
            Val::StrSliceVec(a)  => !a.is_empty(),
            Val::ObjVec(d)       => !d.rows.is_empty(),
            Val::Obj(o)       => !o.is_empty(),
            Val::ObjSmall(p)  => !p.is_empty(),
        })),
        CastType::Number | CastType::Float => match v {
            Val::Int(n)   => Ok(Val::Float(*n as f64)),
            Val::Float(_) => Ok(v.clone()),
            Val::Str(s)   => s.parse::<f64>().map(Val::Float)
                              .map_err(|e| EvalError(format!("as float: {}", e))),
            Val::Bool(b)  => Ok(Val::Float(if *b { 1.0 } else { 0.0 })),
            Val::Null     => Ok(Val::Float(0.0)),
            _             => err!("as float: cannot convert"),
        },
        CastType::Int => match v {
            Val::Int(_)   => Ok(v.clone()),
            Val::Float(f) => Ok(Val::Int(*f as i64)),
            Val::Str(s)   => s.parse::<i64>().map(Val::Int)
                              .or_else(|_| s.parse::<f64>().map(|f| Val::Int(f as i64)))
                              .map_err(|e| EvalError(format!("as int: {}", e))),
            Val::Bool(b)  => Ok(Val::Int(if *b { 1 } else { 0 })),
            Val::Null     => Ok(Val::Int(0)),
            _             => err!("as int: cannot convert"),
        },
        CastType::Array => match v {
            Val::Arr(_)   => Ok(v.clone()),
            Val::Null     => Ok(Val::arr(Vec::new())),
            other         => Ok(Val::arr(vec![other.clone()])),
        },
        CastType::Object => match v {
            Val::Obj(_)   => Ok(v.clone()),
            _             => err!("as object: cannot convert non-object"),
        },
        CastType::Null => Ok(Val::Null),
    }
}

fn apply_fmt_spec(val: &Val, spec: &str) -> String {
    if let Some(rest) = spec.strip_suffix('f') {
        if let Some(prec_str) = rest.strip_prefix('.') {
            if let Ok(prec) = prec_str.parse::<usize>() {
                if let Some(f) = val.as_f64() { return format!("{:.prec$}", f); }
            }
        }
    }
    if spec == "d" { if let Some(i) = val.as_i64() { return format!("{}", i); } }
    let s = val_to_string(val);
    if let Some(w) = spec.strip_prefix('>').and_then(|s| s.parse::<usize>().ok()) { return format!("{:>w$}", s); }
    if let Some(w) = spec.strip_prefix('<').and_then(|s| s.parse::<usize>().ok()) { return format!("{:<w$}", s); }
    if let Some(w) = spec.strip_prefix('^').and_then(|s| s.parse::<usize>().ok()) { return format!("{:^w$}", s); }
    if let Some(w) = spec.strip_prefix('0').and_then(|s| s.parse::<usize>().ok()) {
        if let Some(i) = val.as_i64() { return format!("{:0>w$}", i); }
    }
    s
}

// ── Opcode helpers ────────────────────────────────────────────────────────────

/// Build a no-arg `CallMethod` opcode (used by peephole strength reduction).
/// Newtype wrapper giving `Val` an `Ord` derived from `cmp_vals`,
/// so it can be used in `BinaryHeap` for TopN.
struct WrapVal(Val);
impl PartialEq for WrapVal { fn eq(&self, o: &Self) -> bool {
    super::eval::util::cmp_vals(&self.0, &o.0) == std::cmp::Ordering::Equal
} }
impl Eq for WrapVal {}
impl PartialOrd for WrapVal { fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
    Some(self.cmp(o))
} }
impl Ord for WrapVal { fn cmp(&self, o: &Self) -> std::cmp::Ordering {
    super::eval::util::cmp_vals(&self.0, &o.0)
} }

/// Extract a literal string from a single-op program `[PushStr(s)]`.
fn const_str_program(p: &Arc<Program>) -> Option<Arc<str>> {
    match p.ops.as_ref() {
        [Opcode::PushStr(s)] => Some(s.clone()),
        _ => None,
    }
}

fn make_noarg_call(method: BuiltinMethod, name: &str) -> Opcode {
    Opcode::CallMethod(Arc::new(CompiledCall {
        method,
        name:      Arc::from(name),
        sub_progs: Arc::from(&[] as &[Arc<Program>]),
        orig_args: Arc::from(&[] as &[Arg]),
    }))
}

// ── Hash helpers ──────────────────────────────────────────────────────────────

fn hash_str(s: &str) -> u64 {
    let mut h = DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

/// Hash the *structure* (keys + types) of a Val for the resolution cache key.
/// Does NOT hash values, only shape, so structural navigation results are
/// stable across different values in the same-shaped document.
fn hash_val_structure(v: &Val) -> u64 {
    let mut h = DefaultHasher::new();
    hash_structure_into(v, &mut h, 0);
    h.finish()
}

fn hash_structure_into(v: &Val, h: &mut DefaultHasher, depth: usize) {
    if depth > 8 { return; }
    match v {
        Val::Null       => 0u8.hash(h),
        Val::Bool(b)    => { 1u8.hash(h); b.hash(h); }
        Val::Int(n)     => { 2u8.hash(h); n.hash(h); }
        Val::Float(f)   => { 3u8.hash(h); f.to_bits().hash(h); }
        Val::Str(s)       => { 4u8.hash(h); s.hash(h); }
        Val::StrSlice(r)  => { 4u8.hash(h); r.as_str().hash(h); }
        Val::Arr(a)     => { 5u8.hash(h); a.len().hash(h); for item in a.iter() { hash_structure_into(item, h, depth+1); } }
        Val::IntVec(a)  => { 5u8.hash(h); a.len().hash(h); for n in a.iter() { 2u8.hash(h); n.hash(h); } }
        Val::FloatVec(a) => { 5u8.hash(h); a.len().hash(h); for f in a.iter() { 3u8.hash(h); f.to_bits().hash(h); } }
        Val::StrVec(a)  => { 5u8.hash(h); a.len().hash(h); for s in a.iter() { 4u8.hash(h); s.hash(h); } }
        Val::StrSliceVec(a) => { 5u8.hash(h); a.len().hash(h); for r in a.iter() { 4u8.hash(h); r.as_str().hash(h); } }
        Val::ObjVec(d)  => { 6u8.hash(h); d.rows.len().hash(h); for k in d.keys.iter() { k.hash(h); } }
        Val::Obj(m)     => { 6u8.hash(h); m.len().hash(h); for (k, v) in m.iter() { k.hash(h); hash_structure_into(v, h, depth+1); } }
        Val::ObjSmall(p) => { 6u8.hash(h); p.len().hash(h); for (k, v) in p.iter() { k.hash(h); hash_structure_into(v, h, depth+1); } }
    }
}
