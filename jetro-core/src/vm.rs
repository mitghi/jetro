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
    Short(Arc<str>),
    Kv     { key: Arc<str>, prog: Arc<Program>, optional: bool, cond: Option<Arc<Program>> },
    Dynamic { key: Arc<Program>, val: Arc<Program> },
    Spread(Arc<Program>),
    SpreadDeep(Arc<Program>),
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
    FieldChain(Arc<[Arc<str>]>),
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
    /// Fused `filter(p).last()` — reverse scan, return last item
    /// satisfying `pred` (or Null when none match / input is Null).
    FilterLast { pred: Arc<Program> },
    /// Fused `sort()` + `[0:n]` — partial-sort smallest N using BinaryHeap.
    /// `asc=true` → smallest N; `asc=false` → largest N.
    TopN { n: usize, asc: bool },
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
    /// `filter(k == lit).count()` — count without materialising.
    FilterFieldEqLitCount(Arc<str>, Val),
    /// `filter(k <op> lit).count()` — count cmp without materialising.
    FilterFieldCmpLitCount(Arc<str>, super::ast::BinOp, Val),
    /// `filter(k1 <op> k2).count()` — cross-field count.
    FilterFieldCmpFieldCount(Arc<str>, super::ast::BinOp, Arc<str>),

    // ── group_by specialisation (Tier 2) ──────────────────────────────────────
    /// `group_by(k)` where `k` is a single field ident. Uses FxHashMap with
    /// primitive-key fast path.
    GroupByField(Arc<str>),
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
    /// Inline caches — one `AtomicU64` slot per opcode.  Currently only
    /// `Opcode::GetField` / `Opcode::OptField` populate a slot:
    ///   bits 63..32 = truncated `Arc::as_ptr(&obj_map) as u32`
    ///   bits 31..0  = cached IndexMap slot index
    /// A value of 0 means "miss" (Arc ptr hash 0 is vanishingly rare —
    /// minor correctness cost: worst case one extra slow lookup).
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

/// Build a fresh IC side-table with one zeroed `AtomicU64` per opcode.
/// Kept public so other modules that fabricate `Program` values (schema
/// specialisation, analysis passes) can populate the field.
pub fn fresh_ics(len: usize) -> Arc<[AtomicU64]> {
    let mut v = Vec::with_capacity(len);
    for _ in 0..len { v.push(AtomicU64::new(0)); }
    v.into()
}

/// Look up `key` in `m`, using the IC slot as a speculative hint.
#[inline]
fn ic_get_field(m: &Arc<IndexMap<Arc<str>, Val>>, key: &str, ic: &AtomicU64) -> Val {
    let ptr_hash = ((Arc::as_ptr(m) as usize) >> 4) as u32;
    let cached = ic.load(Ordering::Relaxed);
    let cached_ptr = (cached >> 32) as u32;
    let cached_slot = (cached & 0xFFFF_FFFF) as usize;
    if cached_ptr == ptr_hash && ptr_hash != 0 {
        if let Some((k, v)) = m.get_index(cached_slot) {
            if k.as_ref() == key { return v.clone(); }
        }
    }
    // Slow path: full hash lookup, populate cache on hit.
    if let Some((idx, _, v)) = m.get_full(key) {
        let packed = ((ptr_hash as u64) << 32) | (idx as u64);
        ic.store(packed, Ordering::Relaxed);
        v.clone()
    } else {
        Val::Null
    }
}

/// Recognise `.map(k)` sub-programs that reduce to a single field access
/// from the current item: `[PushCurrent, GetField(k)]` or bare `[GetField(k)]`.
/// Lets MapSum/Min/Max/Avg skip the per-item `exec` dispatch.
#[inline]
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
        Opcode::PushStr(s) => Some(Val::Str(s.clone())),
        _ => None,
    }
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
/// `field == lit` equalities.  Returns the flat list of `(field, lit)`
/// pairs when the entire pred reduces to `f1 == l1 AND f2 == l2 AND ...`.
///
/// Pattern accepted (N ≥ 2):
///   `[⟨field==lit⟩, AndOp(⟨field==lit⟩), AndOp(⟨field==lit⟩), …]`
/// where `⟨field==lit⟩` is one of the trivial-field-eq-lit forms
/// (`[LoadIdent k, PushLit, Eq]`, `[PushCurrent, GetField k, PushLit, Eq]`,
/// or the lit-first flipped form).
fn detect_field_eq_conjuncts(ops: &[Opcode]) -> Option<Vec<(Arc<str>, Val)>> {
    let mut pairs: Vec<(Arc<str>, Val)> = Vec::new();
    // First conjunct: a leading ⟨field==lit⟩ using detect_field_pred.
    // Find the split point: everything before the first AndOp is conjunct #1.
    let first_and = ops.iter().position(|o| matches!(o, Opcode::AndOp(_)));
    let first_len = first_and.unwrap_or(ops.len());
    let first_slice = &ops[..first_len];
    match detect_field_pred(first_slice)? {
        FieldPred::FieldCmpLit(k, super::ast::BinOp::Eq, lit) => pairs.push((k, lit)),
        _ => return None,
    }
    // Subsequent conjuncts: each an Opcode::AndOp(sub) where sub itself is
    // a ⟨field==lit⟩. Walk every remaining op; reject if any non-AndOp
    // appears past the first split.
    for op in &ops[first_len..] {
        if let Opcode::AndOp(sub) = op {
            match detect_field_pred(&sub.ops)? {
                FieldPred::FieldCmpLit(k, super::ast::BinOp::Eq, lit) => pairs.push((k, lit)),
                _ => return None,
            }
        } else {
            return None;
        }
    }
    if pairs.len() >= 2 { Some(pairs) } else { None }
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
fn group_by_field(recv: &Val, k: &str) -> Val {
    let a = match recv {
        Val::Arr(a) => a,
        _ => return Val::obj(indexmap::IndexMap::new()),
    };
    let mut out: indexmap::IndexMap<Arc<str>, Vec<Val>> = indexmap::IndexMap::with_capacity(16);
    for item in a.iter() {
        let key = if let Val::Obj(m) = item {
            match m.get(k) {
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
    let v = m.get(k.as_ref());
    *cached = m.get_index_of(k.as_ref());
    v
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
        let ops = if cfg.find_quantifier { Self::pass_find_quantifier(ops) } else { ops };
        let ops = if cfg.filter_fusion   { Self::pass_field_specialise(ops) } else { ops };
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
            }
            out.push(op);
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
                }
                Opcode::FilterCount(ref pred) => {
                    if let Some(pairs) = detect_field_eq_conjuncts(&pred.ops) {
                        out2.push(Opcode::FilterFieldsAllEqLitCount(Arc::from(pairs)));
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

    /// Replace expensive ops with cheaper equivalents:
    ///   sort() + first()    → min()
    ///   sort() + last()     → max()
    ///   sort() + [0]        → min()
    ///   sort() + [-1]       → max()
    ///   reverse() + first() → last()
    ///   reverse() + last()  → first()
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
                    out.push(Opcode::FieldChain(chain.into()));
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
                        CompiledObjEntry::Short(Arc::from(name.as_str())),
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
            Val::Obj(m) => Some(Arc::as_ptr(m) as *const () as usize),
            Val::Arr(a) => Some(Arc::as_ptr(a) as *const () as usize),
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
                    for k in chain.iter() {
                        cur = cur.get_field(k.as_ref());
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
                    let n = if let Val::Arr(a) = &recv {
                        let mut count = 0u64;
                        let mut scratch = env.clone();
                        for item in a.iter() {
                            let prev = scratch.swap_current(item.clone());
                            let t = is_truthy(&self.exec(pred, &scratch)?);
                            scratch.restore_current(prev);
                            if t { count += 1; }
                        }
                        count
                    } else { 0 };
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
                Opcode::MapSum(f) => {
                    let recv = pop!(stack);
                    let mut acc_i: i64 = 0;
                    let mut acc_f: f64 = 0.0;
                    let mut is_float = false;
                    if let Val::Arr(a) = &recv {
                        if let Some(k) = trivial_field(&f.ops) {
                            for item in a.iter() {
                                if let Val::Obj(m) = item {
                                    match m.get(k.as_ref()) {
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
                            for item in a.iter() {
                                if let Val::Obj(m) = item {
                                    match m.get(k.as_ref()) {
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
                            for item in a.iter() {
                                if let Val::Obj(m) = item {
                                    if let Some(v) = m.get(k.as_ref()) { fold_min!(v); }
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
                            for item in a.iter() {
                                if let Val::Obj(m) = item {
                                    if let Some(v) = m.get(k.as_ref()) { fold_max!(v); }
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
                        for item in a.iter() {
                            match item {
                                Val::Obj(m) => out.push(m.get(k.as_ref()).cloned().unwrap_or(Val::Null)),
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
                    let mut out = Vec::new();
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
                    let mut out = Vec::new();
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
                Opcode::FilterFieldEqLitMapField(kp, lit, kproj) => {
                    let recv = pop!(stack);
                    let mut out = Vec::new();
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
                    let mut out = Vec::new();
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
                    let mut out = Vec::new();
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
                Opcode::FilterMapSum { pred, map } => {
                    let recv = pop!(stack);
                    let mut acc_i: i64 = 0;
                    let mut acc_f: f64 = 0.0;
                    let mut is_float = false;
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
                                    if is_float { acc_f += n as f64; } else { acc_i += n; }
                                }
                                Val::Float(x) => {
                                    if !is_float { acc_f = acc_i as f64; is_float = true; }
                                    acc_f += x;
                                }
                                Val::Null => {}
                                _ => return err!("filter(..).map(..).sum(): non-numeric mapped value"),
                            }
                        }
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
                    for r in right {
                        let key = match &r {
                            Val::Obj(o) => o.get(rhs_key.as_ref()).map(super::eval::util::val_to_key),
                            _ => None,
                        };
                        if let Some(k) = key { idx.entry(k).or_default().push(r); }
                    }
                    let mut out = Vec::with_capacity(left.len());
                    for l in left {
                        let key = match &l {
                            Val::Obj(o) => o.get(lhs_key.as_ref()).map(super::eval::util::val_to_key),
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
                Opcode::MapMap { f1, f2 } => {
                    let recv = pop!(stack);
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
                                    let mut vals: Vec<Val> = Vec::with_capacity(spans.len());
                                    for s in &spans {
                                        if let Ok(v) = serde_json::from_slice::<serde_json::Value>(
                                            &bytes[s.start..s.end],
                                        ) {
                                            vals.push(Val::from(&v));
                                        }
                                    }
                                    stack.push(Val::arr(vals));
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
                    for item in items {
                        let ie = bind_comp_vars(env, &spec.vars, item);
                        if let Some(cond) = &spec.cond {
                            if !is_truthy(&self.exec(cond, &ie)?) { continue; }
                        }
                        out.push(self.exec(&spec.expr, &ie)?);
                    }
                    stack.push(Val::arr(out));
                }

                Opcode::DictComp(spec) => {
                    let items = self.exec_iter_vals(&spec.iter, env)?;
                    let mut map: IndexMap<Arc<str>, Val> = IndexMap::with_capacity(items.len());
                    for item in items {
                        let ie = bind_comp_vars(env, &spec.vars, item);
                        if let Some(cond) = &spec.cond {
                            if !is_truthy(&self.exec(cond, &ie)?) { continue; }
                        }
                        // Reuse existing Arc<str> when the key is already a string.
                        let k: Arc<str> = match self.exec(&spec.key, &ie)? {
                            Val::Str(s) => s,
                            other       => Arc::<str>::from(val_to_key(&other)),
                        };
                        let v = self.exec(&spec.val, &ie)?;
                        map.insert(k, v);
                    }
                    stack.push(Val::obj(map));
                }

                Opcode::SetComp(spec) => {
                    let items = self.exec_iter_vals(&spec.iter, env)?;
                    let mut seen: std::collections::HashSet<String> =
                        std::collections::HashSet::with_capacity(items.len());
                    let mut out = Vec::with_capacity(items.len());
                    for item in items {
                        let ie = bind_comp_vars(env, &spec.vars, item);
                        if let Some(cond) = &spec.cond {
                            if !is_truthy(&self.exec(cond, &ie)?) { continue; }
                        }
                        let v = self.exec(&spec.expr, &ie)?;
                        let k = val_to_key(&v);
                        if seen.insert(k) { out.push(v); }
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
                dispatch_method(recv, call.name.as_ref(), &call.orig_args, env)
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
                let map = recv.into_map().ok_or_else(|| EvalError("transformValues: expected object".into()))?;
                let mut out: IndexMap<Arc<str>, Val> = IndexMap::new();
                for (k, v) in map {
                    out.insert(k, self.exec_lam_body_scratch(lam, &v, lam_param, &mut scratch)?);
                }
                Ok(Val::obj(out))
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
        let mut map: IndexMap<Arc<str>, Val> = IndexMap::new();
        for entry in entries {
            match entry {
                CompiledObjEntry::Short(name) => {
                    let v = if let Some(v) = env.get_var(name.as_ref()) { v.clone() }
                            else { env.current.get_field(name.as_ref()) };
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
        let mut out = String::new();
        for part in parts {
            match part {
                CompiledFSPart::Lit(s) => out.push_str(s.as_ref()),
                CompiledFSPart::Interp { prog, fmt } => {
                    let val = self.exec(prog, env)?;
                    let s = match fmt {
                        None                      => val_to_string(&val),
                        Some(FmtSpec::Spec(spec)) => apply_fmt_spec(&val, spec),
                        Some(FmtSpec::Pipe(method)) => {
                            val_to_string(&dispatch_method(val, method, &[], env)?)
                        }
                    };
                    out.push_str(&s);
                }
            }
        }
        Ok(Val::Str(Arc::from(out.as_str())))
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
fn byte_chain_exec(
    bytes: &[u8],
    root_key: &str,
    tail: &[Opcode],
) -> (Val, usize) {
    let mut spans: Vec<super::scan::ValueSpan> =
        super::scan::find_key_value_spans(bytes, root_key);
    let mut scalar = false;
    let mut consumed = 0usize;

    for op in tail {
        match op {
            Opcode::Descendant(k) => {
                let mut next = Vec::with_capacity(spans.len());
                for s in &spans {
                    let sub = &bytes[s.start..s.end];
                    for s2 in super::scan::find_key_value_spans(sub, k.as_ref()) {
                        next.push(super::scan::ValueSpan {
                            start: s.start + s2.start,
                            end:   s.start + s2.end,
                        });
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
    if let Val::Arr(a) = v {
        let len = a.len() as i64;
        let s = resolve_idx(from.unwrap_or(0), len);
        let e = resolve_idx(to.unwrap_or(len), len);
        let items = Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone());
        let s = s.min(items.len());
        let e = e.min(items.len());
        Val::arr(items[s..e].to_vec())
    } else { Val::Null }
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
            Val::Null     => false,
            Val::Bool(b)  => *b,
            Val::Int(n)   => *n != 0,
            Val::Float(f) => *f != 0.0,
            Val::Str(s)   => !s.is_empty(),
            Val::Arr(a)   => !a.is_empty(),
            Val::Obj(o)   => !o.is_empty(),
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
        Val::Str(s)     => { 4u8.hash(h); s.hash(h); }
        Val::Arr(a)     => { 5u8.hash(h); a.len().hash(h); for item in a.iter() { hash_structure_into(item, h, depth+1); } }
        Val::Obj(m)     => { 6u8.hash(h); m.len().hash(h); for (k, v) in m.iter() { k.hash(h); hash_structure_into(v, h, depth+1); } }
    }
}
