//! Static analysis passes over compiled `Program` IR.
//!
//! Forward-flow analyses that produce *abstract domains* for each opcode
//! position.  Callers (compiler, planner, caller code) can use these to:
//!
//! - emit specialised opcodes where a value's type / nullness / cardinality
//!   is statically known,
//! - reject ill-typed expressions at compile time,
//! - enable further peephole passes that require type awareness.
//!
//! The analyses run on the flat `Arc<[Opcode]>` IR and are intentionally
//! conservative — when uncertain, they return the `Unknown` top element of
//! the lattice.

use std::sync::Arc;
use std::collections::HashMap;

use super::vm::{Program, Opcode, BuiltinMethod};
use super::ast::KindType;

// ── Type lattice ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VType {
    /// Bottom: no value (unreachable).
    Bottom,
    Null,
    Bool,
    Int,
    Float,
    /// Numeric — Int or Float (partially refined).
    Num,
    Str,
    Arr,
    Obj,
    /// Top: any type possible.
    Unknown,
}

impl VType {
    /// Least upper bound (join) for the lattice.
    pub fn join(self, other: VType) -> VType {
        if self == other { return self; }
        match (self, other) {
            (VType::Bottom, x) | (x, VType::Bottom) => x,
            (VType::Int, VType::Float) | (VType::Float, VType::Int)
                | (VType::Int, VType::Num) | (VType::Num, VType::Int)
                | (VType::Float, VType::Num) | (VType::Num, VType::Float)
                => VType::Num,
            _ => VType::Unknown,
        }
    }

    pub fn is_array_like(self) -> bool { matches!(self, VType::Arr) }
    pub fn is_object_like(self) -> bool { matches!(self, VType::Obj) }
    pub fn is_numeric(self) -> bool { matches!(self, VType::Int | VType::Float | VType::Num) }
    pub fn is_string(self) -> bool { matches!(self, VType::Str) }
}

// ── Null-ness lattice ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Nullness {
    /// Proven null.
    AlwaysNull,
    /// Proven non-null.
    NonNull,
    /// May or may not be null.
    MaybeNull,
}

impl Nullness {
    pub fn join(self, other: Nullness) -> Nullness {
        if self == other { return self; }
        Nullness::MaybeNull
    }
}

// ── Cardinality lattice ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Cardinality {
    /// Exactly 0 elements (empty).
    Zero,
    /// Exactly 1 element (scalar-wrapped or unwrapped).
    One,
    /// 0 or 1 (e.g. result of `?` quantifier).
    ZeroOrOne,
    /// Multiple elements possible.
    Many,
    /// Not an array (scalar domain).
    NotArray,
    /// Unknown shape.
    Unknown,
}

impl Cardinality {
    pub fn join(self, other: Cardinality) -> Cardinality {
        if self == other { return self; }
        match (self, other) {
            (Cardinality::Zero, Cardinality::One)
                | (Cardinality::One, Cardinality::Zero) => Cardinality::ZeroOrOne,
            _ => Cardinality::Unknown,
        }
    }
}

// ── Abstract value ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AbstractVal {
    pub ty:   VType,
    pub null: Nullness,
    pub card: Cardinality,
}

impl AbstractVal {
    pub const UNKNOWN: Self = Self {
        ty: VType::Unknown, null: Nullness::MaybeNull, card: Cardinality::Unknown,
    };
    pub const NULL: Self = Self {
        ty: VType::Null, null: Nullness::AlwaysNull, card: Cardinality::NotArray,
    };
    pub fn scalar(ty: VType) -> Self {
        Self { ty, null: Nullness::NonNull, card: Cardinality::NotArray }
    }
    pub fn array() -> Self {
        Self { ty: VType::Arr, null: Nullness::NonNull, card: Cardinality::Many }
    }
    pub fn object() -> Self {
        Self { ty: VType::Obj, null: Nullness::NonNull, card: Cardinality::NotArray }
    }
    pub fn join(self, other: AbstractVal) -> AbstractVal {
        AbstractVal {
            ty:   self.ty.join(other.ty),
            null: self.null.join(other.null),
            card: self.card.join(other.card),
        }
    }
}

// ── Forward type inference ────────────────────────────────────────────────────

/// Walk opcodes of `program` forward, simulating a stack of `AbstractVal`s.
/// Returns the top-of-stack abstract value at program end (i.e. the result type).
pub fn infer_result_type(program: &Program) -> AbstractVal {
    let mut stack: Vec<AbstractVal> = Vec::with_capacity(16);
    let mut env: HashMap<Arc<str>, AbstractVal> = HashMap::new();
    for op in program.ops.iter() { apply_op_env(op, &mut stack, &mut env); }
    stack.pop().unwrap_or(AbstractVal::UNKNOWN)
}

/// Same as `infer_result_type` but exposes the bindings environment after
/// the program finishes — useful for debugging the interprocedural flow.
pub fn infer_result_type_with_env(program: &Program)
    -> (AbstractVal, HashMap<Arc<str>, AbstractVal>)
{
    let mut stack: Vec<AbstractVal> = Vec::with_capacity(16);
    let mut env: HashMap<Arc<str>, AbstractVal> = HashMap::new();
    for op in program.ops.iter() { apply_op_env(op, &mut stack, &mut env); }
    (stack.pop().unwrap_or(AbstractVal::UNKNOWN), env)
}

fn apply_op_env(op: &Opcode, stack: &mut Vec<AbstractVal>,
                env: &mut HashMap<Arc<str>, AbstractVal>) {
    // Handle binding & ident lookup here, delegate scalar cases to apply_op.
    match op {
        Opcode::LoadIdent(name) => {
            let av = env.get(name).copied().unwrap_or(AbstractVal::UNKNOWN);
            stack.push(av);
        }
        Opcode::BindVar(name) => {
            // TOS preserved; record type.
            if let Some(top) = stack.last().copied() { env.insert(name.clone(), top); }
        }
        Opcode::StoreVar(name) => {
            let v = stack.pop().unwrap_or(AbstractVal::UNKNOWN);
            env.insert(name.clone(), v);
        }
        Opcode::LetExpr { name, body } => {
            let init = stack.pop().unwrap_or(AbstractVal::UNKNOWN);
            let saved = env.get(name).copied();
            env.insert(name.clone(), init);
            let mut sub_stack: Vec<AbstractVal> = Vec::with_capacity(8);
            for op2 in body.ops.iter() { apply_op_env(op2, &mut sub_stack, env); }
            let res = sub_stack.pop().unwrap_or(AbstractVal::UNKNOWN);
            // Restore shadowed binding.
            match saved {
                Some(v) => { env.insert(name.clone(), v); }
                None    => { env.remove(name); }
            }
            stack.push(res);
        }
        _ => apply_op(op, stack),
    }
}

fn apply_op(op: &Opcode, stack: &mut Vec<AbstractVal>) {
    macro_rules! pop2 { () => {{ let r = stack.pop().unwrap_or(AbstractVal::UNKNOWN); let l = stack.pop().unwrap_or(AbstractVal::UNKNOWN); (l, r) }} }
    macro_rules! pop1 { () => { stack.pop().unwrap_or(AbstractVal::UNKNOWN) } }
    match op {
        Opcode::PushNull      => stack.push(AbstractVal::NULL),
        Opcode::PushBool(_)   => stack.push(AbstractVal::scalar(VType::Bool)),
        Opcode::PushInt(_)    => stack.push(AbstractVal::scalar(VType::Int)),
        Opcode::PushFloat(_)  => stack.push(AbstractVal::scalar(VType::Float)),
        Opcode::PushStr(_)    => stack.push(AbstractVal::scalar(VType::Str)),
        Opcode::PushRoot | Opcode::PushCurrent | Opcode::LoadIdent(_)
                              => stack.push(AbstractVal::UNKNOWN),
        Opcode::GetField(_) | Opcode::OptField(_) | Opcode::FieldChain(_) => {
            pop1!();
            stack.push(AbstractVal::UNKNOWN);
        }
        Opcode::GetIndex(_) | Opcode::DynIndex(_) => {
            pop1!();
            stack.push(AbstractVal::UNKNOWN);
        }
        Opcode::GetSlice(_, _) => {
            pop1!();
            stack.push(AbstractVal::array());
        }
        Opcode::Descendant(_) | Opcode::DescendAll => {
            pop1!();
            stack.push(AbstractVal::array());
        }
        Opcode::InlineFilter(_) => {
            pop1!();
            stack.push(AbstractVal::array());
        }
        Opcode::Quantifier(kind) => {
            pop1!();
            use super::ast::QuantifierKind;
            let card = match kind {
                QuantifierKind::First => Cardinality::ZeroOrOne,
                QuantifierKind::One   => Cardinality::One,
            };
            stack.push(AbstractVal { ty: VType::Unknown, null: Nullness::MaybeNull, card });
        }
        Opcode::RootChain(_) => stack.push(AbstractVal::UNKNOWN),
        Opcode::FilterCount(_) => {
            pop1!();
            stack.push(AbstractVal::scalar(VType::Int));
        }
        Opcode::FindFirst(_) | Opcode::FindOne(_) => {
            pop1!();
            stack.push(AbstractVal::UNKNOWN);
        }
        Opcode::FilterMap { .. } | Opcode::FilterFilter { .. }
            | Opcode::MapMap { .. } | Opcode::MapFilter { .. } => {
            pop1!();
            stack.push(AbstractVal::array());
        }
        Opcode::MapSum(_) | Opcode::FilterMapSum { .. }
        | Opcode::MapMin(_) | Opcode::MapMax(_)
        | Opcode::MapFieldSum(_) | Opcode::MapFieldMin(_) | Opcode::MapFieldMax(_) => {
            pop1!();
            stack.push(AbstractVal::scalar(VType::Num));
        }
        Opcode::MapAvg(_) | Opcode::FilterMapAvg { .. } | Opcode::MapFieldAvg(_) => {
            pop1!();
            stack.push(AbstractVal::scalar(VType::Float));
        }
        Opcode::FilterFieldEqLitCount(_, _)
        | Opcode::FilterFieldCmpLitCount(_, _, _)
        | Opcode::FilterFieldCmpFieldCount(_, _, _) => {
            pop1!();
            stack.push(AbstractVal::scalar(VType::Int));
        }
        Opcode::TopN { .. } | Opcode::MapFlatten(_) | Opcode::FilterTakeWhile { .. }
            | Opcode::FilterDropWhile { .. } | Opcode::MapUnique(_)
            | Opcode::EquiJoin { .. }
            | Opcode::MapField(_) | Opcode::MapFieldUnique(_)
            | Opcode::FlatMapChain(_)
            | Opcode::FilterFieldEqLit(_, _) | Opcode::FilterFieldCmpLit(_, _, _)
            | Opcode::FilterFieldCmpField(_, _, _)
            | Opcode::GroupByField(_) => {
            pop1!();
            stack.push(AbstractVal::array());
        }
        Opcode::MapFirst(_) | Opcode::MapLast(_) | Opcode::FilterMapFirst { .. }
            | Opcode::FilterLast { .. } => {
            pop1!();
            stack.push(AbstractVal::UNKNOWN);
        }
        Opcode::Add | Opcode::Sub | Opcode::Mul | Opcode::Mod => {
            let (l, r) = pop2!();
            stack.push(AbstractVal::scalar(l.ty.join(r.ty)));
        }
        Opcode::Div => {
            pop2!();
            stack.push(AbstractVal::scalar(VType::Float));
        }
        Opcode::Eq | Opcode::Neq | Opcode::Lt | Opcode::Lte
            | Opcode::Gt | Opcode::Gte | Opcode::Fuzzy => {
            pop2!();
            stack.push(AbstractVal::scalar(VType::Bool));
        }
        Opcode::Not => { pop1!(); stack.push(AbstractVal::scalar(VType::Bool)); }
        Opcode::Neg => { let v = pop1!(); stack.push(AbstractVal::scalar(v.ty)); }
        Opcode::AndOp(_) | Opcode::OrOp(_) => {
            pop1!();
            stack.push(AbstractVal::scalar(VType::Bool));
        }
        Opcode::CoalesceOp(_) => { pop1!(); stack.push(AbstractVal::UNKNOWN); }
        Opcode::CallMethod(call) | Opcode::CallOptMethod(call) => {
            pop1!();
            stack.push(method_result_type(call.method));
        }
        Opcode::MakeObj(_) => stack.push(AbstractVal::object()),
        Opcode::MakeArr(_) => stack.push(AbstractVal::array()),
        Opcode::FString(_) => stack.push(AbstractVal::scalar(VType::Str)),
        Opcode::KindCheck { .. } => { pop1!(); stack.push(AbstractVal::scalar(VType::Bool)); }
        Opcode::SetCurrent => {} // TOS stays as current
        Opcode::BindVar(_) => {} // TOS preserved
        Opcode::StoreVar(_) => { pop1!(); }
        Opcode::BindObjDestructure(_) | Opcode::BindArrDestructure(_) => {} // TOS preserved
        Opcode::LetExpr { .. } => {
            pop1!();
            stack.push(AbstractVal::UNKNOWN);
        }
        Opcode::ListComp(_) | Opcode::SetComp(_) => stack.push(AbstractVal::array()),
        Opcode::DictComp(_) => stack.push(AbstractVal::object()),
        Opcode::GetPointer(_) => stack.push(AbstractVal::UNKNOWN),
        Opcode::PatchEval(_) => stack.push(AbstractVal::UNKNOWN),
        Opcode::CastOp(ty) => {
            pop1!();
            use super::ast::CastType;
            let av = match ty {
                CastType::Int    => AbstractVal::scalar(VType::Int),
                CastType::Float  => AbstractVal::scalar(VType::Float),
                CastType::Number => AbstractVal::scalar(VType::Num),
                CastType::Str    => AbstractVal::scalar(VType::Str),
                CastType::Bool   => AbstractVal::scalar(VType::Bool),
                CastType::Array  => AbstractVal::array(),
                CastType::Object => AbstractVal::object(),
                CastType::Null   => AbstractVal::NULL,
            };
            stack.push(av);
        }
    }
}

/// Static result-type mapping for builtin methods (conservative).
pub fn method_result_type(m: BuiltinMethod) -> AbstractVal {
    use BuiltinMethod::*;
    match m {
        // → Int
        Len | Count | Sum | IndexOf | LastIndexOf => AbstractVal::scalar(VType::Int),
        // → Bool
        Any | All | Has | Missing | Includes | StartsWith | EndsWith => AbstractVal::scalar(VType::Bool),
        // → Str
        Upper | Lower | Capitalize | TitleCase | Trim | TrimLeft | TrimRight
            | ToString | ToJson | ToBase64 | FromBase64 | UrlEncode | UrlDecode
            | HtmlEscape | HtmlUnescape | Repeat | PadLeft | PadRight
            | Replace | ReplaceAll | StripPrefix | StripSuffix | Indent | Dedent
            | Join | ToCsv | ToTsv | Type
            => AbstractVal::scalar(VType::Str),
        // → Float
        Avg => AbstractVal::scalar(VType::Float),
        // → Num (Min/Max depend on input; treat as unknown scalar)
        Min | Max | ToNumber => AbstractVal::scalar(VType::Num),
        // → Bool (to_bool)
        ToBool => AbstractVal::scalar(VType::Bool),
        // → Arr
        Keys | Values | Entries | ToPairs | Reverse | Unique | Flatten | Compact
            | Chars | Lines | Words | Split | Sort | Filter | Map | FlatMap
            | Enumerate | Pairwise | Window | Chunk | TakeWhile | DropWhile
            | Accumulate | Zip | ZipLongest | Diff | Intersect | Union
            | Append | Prepend | Remove | Matches | Scan | Slice
            => AbstractVal::array(),
        // → Obj
        FromPairs | Invert | Pick | Omit | Merge | DeepMerge | Defaults | Rename
            | TransformKeys | TransformValues | FilterKeys | FilterValues | Pivot
            | GroupBy | CountBy | IndexBy | Partition | FlattenKeys | UnflattenKeys
            | SetPath | DelPath | DelPaths | Set | Update
            => AbstractVal::object(),
        // → various
        First | Last | Nth | GetPath => AbstractVal::UNKNOWN,
        HasPath => AbstractVal::scalar(VType::Bool),
        FromJson | Or => AbstractVal::UNKNOWN,
        EquiJoin => AbstractVal::array(),
        Unknown => AbstractVal::UNKNOWN,
    }
}

// ── Alias / use-count analysis ────────────────────────────────────────────────

/// Count `LoadIdent(name)` references across an entire program (including sub-programs).
pub fn count_ident_uses(program: &Program, name: &str) -> usize {
    let mut n = 0;
    count_ident_uses_in_ops(&program.ops, name, &mut n);
    n
}

fn count_ident_uses_in_ops(ops: &[Opcode], name: &str, acc: &mut usize) {
    for op in ops {
        match op {
            Opcode::LoadIdent(s) if s.as_ref() == name => *acc += 1,
            Opcode::AndOp(p) | Opcode::OrOp(p) | Opcode::CoalesceOp(p)
                | Opcode::InlineFilter(p) | Opcode::FilterCount(p)
                | Opcode::FindFirst(p) | Opcode::FindOne(p)
                | Opcode::DynIndex(p)
                | Opcode::MapSum(p) | Opcode::MapAvg(p)
                | Opcode::MapMin(p) | Opcode::MapMax(p)
                | Opcode::MapFlatten(p)
                | Opcode::MapFirst(p) | Opcode::MapLast(p)
                | Opcode::FilterLast { pred: p }
                => count_ident_uses_in_ops(&p.ops, name, acc),
            Opcode::FilterTakeWhile { pred, stop } => {
                count_ident_uses_in_ops(&pred.ops, name, acc);
                count_ident_uses_in_ops(&stop.ops, name, acc);
            }
            Opcode::FilterMap { pred, map }
                | Opcode::FilterMapSum { pred, map }
                | Opcode::FilterMapAvg { pred, map }
                | Opcode::FilterMapFirst { pred, map } => {
                count_ident_uses_in_ops(&pred.ops, name, acc);
                count_ident_uses_in_ops(&map.ops, name, acc);
            }
            Opcode::MapFilter { map, pred } => {
                count_ident_uses_in_ops(&map.ops, name, acc);
                count_ident_uses_in_ops(&pred.ops, name, acc);
            }
            Opcode::FilterFilter { p1, p2 } => {
                count_ident_uses_in_ops(&p1.ops, name, acc);
                count_ident_uses_in_ops(&p2.ops, name, acc);
            }
            Opcode::MapMap { f1, f2 } => {
                count_ident_uses_in_ops(&f1.ops, name, acc);
                count_ident_uses_in_ops(&f2.ops, name, acc);
            }
            Opcode::CallMethod(c) | Opcode::CallOptMethod(c) => {
                for p in c.sub_progs.iter() { count_ident_uses_in_ops(&p.ops, name, acc); }
            }
            Opcode::LetExpr { body, .. } => count_ident_uses_in_ops(&body.ops, name, acc),
            Opcode::ListComp(spec) | Opcode::SetComp(spec) => {
                count_ident_uses_in_ops(&spec.expr.ops, name, acc);
                count_ident_uses_in_ops(&spec.iter.ops, name, acc);
                if let Some(c) = &spec.cond { count_ident_uses_in_ops(&c.ops, name, acc); }
            }
            Opcode::DictComp(spec) => {
                count_ident_uses_in_ops(&spec.key.ops, name, acc);
                count_ident_uses_in_ops(&spec.val.ops, name, acc);
                count_ident_uses_in_ops(&spec.iter.ops, name, acc);
                if let Some(c) = &spec.cond { count_ident_uses_in_ops(&c.ops, name, acc); }
            }
            Opcode::MakeObj(entries) => {
                use super::vm::CompiledObjEntry;
                for e in entries.iter() {
                    match e {
                        CompiledObjEntry::Short(_) => {}
                        CompiledObjEntry::Kv { prog, cond, .. } => {
                            count_ident_uses_in_ops(&prog.ops, name, acc);
                            if let Some(c) = cond { count_ident_uses_in_ops(&c.ops, name, acc); }
                        }
                        CompiledObjEntry::Dynamic { key, val } => {
                            count_ident_uses_in_ops(&key.ops, name, acc);
                            count_ident_uses_in_ops(&val.ops, name, acc);
                        }
                        CompiledObjEntry::Spread(p) => count_ident_uses_in_ops(&p.ops, name, acc),
                        CompiledObjEntry::SpreadDeep(p) => count_ident_uses_in_ops(&p.ops, name, acc),
                    }
                }
            }
            Opcode::MakeArr(progs) => {
                for p in progs.iter() { count_ident_uses_in_ops(&p.ops, name, acc); }
            }
            Opcode::FString(parts) => {
                use super::vm::CompiledFSPart;
                for p in parts.iter() {
                    if let CompiledFSPart::Interp { prog, .. } = p {
                        count_ident_uses_in_ops(&prog.ops, name, acc);
                    }
                }
            }
            _ => {}
        }
    }
}

// ── Projection set (field access pattern) ─────────────────────────────────────

/// Collect all field names directly accessed from a program.  Used by
/// projection-pushdown analysis: if an object is later accessed only via these
/// fields, other fields can be trimmed early.
pub fn collect_accessed_fields(program: &Program) -> Vec<Arc<str>> {
    let mut set = Vec::new();
    collect_fields_in_ops(&program.ops, &mut set);
    set
}

fn collect_fields_in_ops(ops: &[Opcode], acc: &mut Vec<Arc<str>>) {
    for op in ops {
        match op {
            Opcode::GetField(k) | Opcode::OptField(k) | Opcode::Descendant(k)
                | Opcode::MapFieldSum(k) | Opcode::MapFieldAvg(k)
                | Opcode::MapFieldMin(k) | Opcode::MapFieldMax(k)
                | Opcode::MapField(k) | Opcode::MapFieldUnique(k)
                | Opcode::GroupByField(k)
                | Opcode::FilterFieldEqLit(k, _) | Opcode::FilterFieldCmpLit(k, _, _)
                | Opcode::FilterFieldEqLitCount(k, _) | Opcode::FilterFieldCmpLitCount(k, _, _)
                => {
                if !acc.iter().any(|a: &Arc<str>| a == k) { acc.push(k.clone()); }
            }
            Opcode::FilterFieldCmpField(k1, _, k2)
                | Opcode::FilterFieldCmpFieldCount(k1, _, k2) => {
                if !acc.iter().any(|a: &Arc<str>| a == k1) { acc.push(k1.clone()); }
                if !acc.iter().any(|a: &Arc<str>| a == k2) { acc.push(k2.clone()); }
            }
            Opcode::FlatMapChain(ks) => {
                for k in ks.iter() {
                    if !acc.iter().any(|a: &Arc<str>| a == k) { acc.push(k.clone()); }
                }
            }
            Opcode::RootChain(chain) => {
                for k in chain.iter() {
                    if !acc.iter().any(|a: &Arc<str>| a == k) { acc.push(k.clone()); }
                }
            }
            Opcode::AndOp(p) | Opcode::OrOp(p) | Opcode::CoalesceOp(p)
                | Opcode::InlineFilter(p) | Opcode::FilterCount(p)
                | Opcode::FindFirst(p) | Opcode::FindOne(p)
                | Opcode::DynIndex(p)
                | Opcode::MapSum(p) | Opcode::MapAvg(p)
                | Opcode::MapMin(p) | Opcode::MapMax(p)
                | Opcode::MapFlatten(p)
                | Opcode::MapFirst(p) | Opcode::MapLast(p)
                | Opcode::FilterLast { pred: p }
                => collect_fields_in_ops(&p.ops, acc),
            Opcode::FilterTakeWhile { pred, stop } => {
                collect_fields_in_ops(&pred.ops, acc);
                collect_fields_in_ops(&stop.ops, acc);
            }
            Opcode::FilterMap { pred, map }
                | Opcode::FilterMapSum { pred, map }
                | Opcode::FilterMapAvg { pred, map }
                | Opcode::FilterMapFirst { pred, map } => {
                collect_fields_in_ops(&pred.ops, acc);
                collect_fields_in_ops(&map.ops, acc);
            }
            Opcode::MapFilter { map, pred } => {
                collect_fields_in_ops(&map.ops, acc);
                collect_fields_in_ops(&pred.ops, acc);
            }
            Opcode::FilterFilter { p1, p2 } => {
                collect_fields_in_ops(&p1.ops, acc);
                collect_fields_in_ops(&p2.ops, acc);
            }
            Opcode::MapMap { f1, f2 } => {
                collect_fields_in_ops(&f1.ops, acc);
                collect_fields_in_ops(&f2.ops, acc);
            }
            Opcode::CallMethod(c) | Opcode::CallOptMethod(c) => {
                for p in c.sub_progs.iter() { collect_fields_in_ops(&p.ops, acc); }
            }
            Opcode::LetExpr { body, .. } => collect_fields_in_ops(&body.ops, acc),
            _ => {}
        }
    }
}

// ── Structural opcode hashing (for CSE) ───────────────────────────────────────

/// Hash a program's opcode sequence into a stable identifier.  Two programs
/// with the same opcodes hash to the same value — enables CSE across `let`
/// initialisers or parallel `->` binds.
pub fn program_signature(program: &Program) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;
    let mut h = DefaultHasher::new();
    hash_ops(&program.ops, &mut h);
    h.finish()
}

fn hash_ops(ops: &[Opcode], h: &mut impl std::hash::Hasher) {
    use std::hash::Hash;
    for op in ops {
        // Discriminant + data for stable signature.
        std::mem::discriminant(op).hash(h);
        match op {
            Opcode::PushInt(n) => n.hash(h),
            Opcode::PushStr(s) => s.as_bytes().hash(h),
            Opcode::PushBool(b) => b.hash(h),
            Opcode::GetField(k) | Opcode::OptField(k) | Opcode::Descendant(k)
                | Opcode::LoadIdent(k) | Opcode::GetPointer(k)
                => k.as_bytes().hash(h),
            Opcode::GetIndex(i) => i.hash(h),
            Opcode::CallMethod(c) | Opcode::CallOptMethod(c) => {
                (c.method as u8).hash(h);
                for p in c.sub_progs.iter() { hash_ops(&p.ops, h); }
            }
            Opcode::AndOp(p) | Opcode::OrOp(p) | Opcode::CoalesceOp(p)
                | Opcode::InlineFilter(p) | Opcode::FilterCount(p)
                | Opcode::FindFirst(p) | Opcode::FindOne(p)
                | Opcode::DynIndex(p)
                | Opcode::MapSum(p) | Opcode::MapAvg(p)
                | Opcode::MapMin(p) | Opcode::MapMax(p)
                | Opcode::MapFlatten(p)
                | Opcode::MapFirst(p) | Opcode::MapLast(p)
                | Opcode::FilterLast { pred: p }
                => hash_ops(&p.ops, h),
            Opcode::FilterTakeWhile { pred, stop } => {
                hash_ops(&pred.ops, h);
                hash_ops(&stop.ops, h);
            }
            Opcode::RootChain(chain) => {
                for k in chain.iter() { k.as_bytes().hash(h); }
            }
            _ => {}
        }
    }
}

// ── Common-subexpression detection ────────────────────────────────────────────

/// Find sub-programs (via `Arc<Program>` pointers inside opcodes) that appear
/// multiple times across the program tree.  Returns a map of
/// `signature → count` for analysis / potential reuse.
pub fn find_common_subexprs(program: &Program) -> HashMap<u64, usize> {
    let mut map: HashMap<u64, usize> = HashMap::new();
    walk_subprograms(&program.ops, &mut map);
    map.retain(|_, &mut n| n >= 2);
    map
}

fn walk_subprograms(ops: &[Opcode], map: &mut HashMap<u64, usize>) {
    for op in ops {
        let sub_progs: Vec<&Arc<Program>> = match op {
            Opcode::AndOp(p) | Opcode::OrOp(p) | Opcode::CoalesceOp(p)
                | Opcode::InlineFilter(p) | Opcode::FilterCount(p)
                | Opcode::FindFirst(p) | Opcode::FindOne(p)
                | Opcode::DynIndex(p)
                | Opcode::MapSum(p) | Opcode::MapAvg(p)
                | Opcode::MapMin(p) | Opcode::MapMax(p)
                | Opcode::MapFlatten(p)
                | Opcode::MapFirst(p) | Opcode::MapLast(p)
                | Opcode::FilterLast { pred: p }
                => vec![p],
            Opcode::FilterTakeWhile { pred, stop } => vec![pred, stop],
            Opcode::FilterMap { pred, map: m }
                | Opcode::FilterMapSum { pred, map: m }
                | Opcode::FilterMapAvg { pred, map: m }
                | Opcode::FilterMapFirst { pred, map: m } => vec![pred, m],
            Opcode::MapFilter { map: m, pred } => vec![m, pred],
            Opcode::FilterFilter { p1, p2 } => vec![p1, p2],
            Opcode::MapMap { f1, f2 } => vec![f1, f2],
            Opcode::CallMethod(c) | Opcode::CallOptMethod(c) =>
                c.sub_progs.iter().collect(),
            Opcode::LetExpr { body, .. } => vec![body],
            Opcode::MakeArr(progs) => progs.iter().collect(),
            Opcode::MakeObj(entries) => {
                use super::vm::CompiledObjEntry;
                let mut v = Vec::new();
                for e in entries.iter() {
                    match e {
                        CompiledObjEntry::Short(_) => {}
                        CompiledObjEntry::Kv { prog, cond, .. } => {
                            v.push(prog);
                            if let Some(c) = cond { v.push(c); }
                        }
                        CompiledObjEntry::Dynamic { key, val } => { v.push(key); v.push(val); }
                        CompiledObjEntry::Spread(p) => v.push(p),
                        CompiledObjEntry::SpreadDeep(p) => v.push(p),
                    }
                }
                v
            }
            _ => continue,
        };
        for p in sub_progs {
            let sig = program_signature(p);
            *map.entry(sig).or_insert(0) += 1;
            walk_subprograms(&p.ops, map);
        }
    }
}

// ── AST-level ident use walker (for dead-let) ────────────────────────────────

/// True if any sub-expression references `name` as a bare identifier.
/// Walks the AST without compiling.  Shadowing by inner `let` / `lambda` /
/// comprehension binders is respected: inner bindings with the same name
/// hide the outer one.
pub fn expr_uses_ident(expr: &super::ast::Expr, name: &str) -> bool {
    use super::ast::{Expr, Step, PipeStep, BindTarget, FStringPart, ArrayElem, ObjField, Arg};
    match expr {
        Expr::Ident(n) => n == name,
        Expr::Null | Expr::Bool(_) | Expr::Int(_) | Expr::Float(_)
            | Expr::Str(_) | Expr::Root | Expr::Current => false,
        Expr::FString(parts) => parts.iter().any(|p| match p {
            FStringPart::Lit(_) => false,
            FStringPart::Interp { expr, .. } => expr_uses_ident(expr, name),
        }),
        Expr::Chain(base, steps) => {
            if expr_uses_ident(base, name) { return true; }
            steps.iter().any(|s| match s {
                Step::DynIndex(e) | Step::InlineFilter(e) => expr_uses_ident(e, name),
                Step::Method(_, args) | Step::OptMethod(_, args) =>
                    args.iter().any(|a| match a {
                        Arg::Pos(e) | Arg::Named(_, e) => expr_uses_ident(e, name),
                    }),
                _ => false,
            })
        }
        Expr::BinOp(l, _, r) => expr_uses_ident(l, name) || expr_uses_ident(r, name),
        Expr::UnaryNeg(e) | Expr::Not(e) => expr_uses_ident(e, name),
        Expr::Kind { expr, .. } => expr_uses_ident(expr, name),
        Expr::Coalesce(l, r) => expr_uses_ident(l, name) || expr_uses_ident(r, name),
        Expr::Object(fields) => fields.iter().any(|f| match f {
            ObjField::Kv { val, cond, .. } =>
                expr_uses_ident(val, name)
                || cond.as_ref().map_or(false, |c| expr_uses_ident(c, name)),
            ObjField::Short(n) => n == name,
            ObjField::Dynamic { key, val } =>
                expr_uses_ident(key, name) || expr_uses_ident(val, name),
            ObjField::Spread(e) => expr_uses_ident(e, name),
            ObjField::SpreadDeep(e) => expr_uses_ident(e, name),
        }),
        Expr::Array(elems) => elems.iter().any(|e| match e {
            ArrayElem::Expr(e) | ArrayElem::Spread(e) => expr_uses_ident(e, name),
        }),
        Expr::Pipeline { base, steps } => {
            if expr_uses_ident(base, name) { return true; }
            steps.iter().any(|s| match s {
                PipeStep::Forward(e) => expr_uses_ident(e, name),
                PipeStep::Bind(bt) => match bt {
                    BindTarget::Name(n) => n == name,
                    BindTarget::Obj { fields, rest } =>
                        fields.iter().any(|f| f == name)
                        || rest.as_ref().map_or(false, |r| r == name),
                    BindTarget::Arr(ns) => ns.iter().any(|n| n == name),
                },
            })
        }
        Expr::ListComp { expr, vars, iter, cond }
        | Expr::SetComp  { expr, vars, iter, cond }
        | Expr::GenComp  { expr, vars, iter, cond } => {
            if expr_uses_ident(iter, name) { return true; }
            if vars.iter().any(|v| v == name) { return false; } // shadowed
            expr_uses_ident(expr, name)
                || cond.as_ref().map_or(false, |c| expr_uses_ident(c, name))
        }
        Expr::DictComp { key, val, vars, iter, cond } => {
            if expr_uses_ident(iter, name) { return true; }
            if vars.iter().any(|v| v == name) { return false; }
            expr_uses_ident(key, name) || expr_uses_ident(val, name)
                || cond.as_ref().map_or(false, |c| expr_uses_ident(c, name))
        }
        Expr::Lambda { params, body } => {
            if params.iter().any(|p| p == name) { return false; }
            expr_uses_ident(body, name)
        }
        Expr::Let { name: n, init, body } => {
            if expr_uses_ident(init, name) { return true; }
            if n == name { return false; } // inner let shadows
            expr_uses_ident(body, name)
        }
        Expr::GlobalCall { args, .. } => args.iter().any(|a| match a {
            Arg::Pos(e) | Arg::Named(_, e) => expr_uses_ident(e, name),
        }),
        Expr::Cast { expr, .. } => expr_uses_ident(expr, name),
        Expr::Patch { root, ops } => {
            use super::ast::PathStep;
            if expr_uses_ident(root, name) { return true; }
            ops.iter().any(|op| {
                op.path.iter().any(|s| match s {
                    PathStep::WildcardFilter(e) => expr_uses_ident(e, name),
                    _ => false,
                })
                || expr_uses_ident(&op.val, name)
                || op.cond.as_ref().map_or(false, |c| expr_uses_ident(c, name))
            })
        }
        Expr::DeleteMark => false,
    }
}

/// True if the expression is pure — no side-effecting global calls or
/// unknown methods.  Enables dropping unused `let` initialisers safely.
pub fn expr_is_pure(expr: &super::ast::Expr) -> bool {
    use super::ast::{Expr, Step, Arg};
    match expr {
        Expr::Null | Expr::Bool(_) | Expr::Int(_) | Expr::Float(_)
            | Expr::Str(_) | Expr::Root | Expr::Current | Expr::Ident(_) => true,
        Expr::FString(_) => true,
        Expr::Chain(base, steps) => {
            if !expr_is_pure(base) { return false; }
            steps.iter().all(|s| match s {
                Step::DynIndex(e) | Step::InlineFilter(e) => expr_is_pure(e),
                Step::Method(_, args) | Step::OptMethod(_, args) =>
                    args.iter().all(|a| match a {
                        Arg::Pos(e) | Arg::Named(_, e) => expr_is_pure(e),
                    }),
                _ => true,
            })
        }
        Expr::BinOp(l, _, r) | Expr::Coalesce(l, r) =>
            expr_is_pure(l) && expr_is_pure(r),
        Expr::UnaryNeg(e) | Expr::Not(e) | Expr::Kind { expr: e, .. } =>
            expr_is_pure(e),
        // All Jetro exprs are pure in practice; global calls may throw but no side effects.
        _ => true,
    }
}

// ── CSE: canonicalise identical sub-programs ─────────────────────────────────

/// Walk `program` and replace every `Arc<Program>` inside opcodes with a
/// canonical `Arc` keyed by `program_signature`.  Structurally-identical
/// sub-programs end up pointing at the same allocation, reducing memory
/// and enabling downstream caches to hit on the same key.
///
/// Returns a new `Program` with deduplicated sub-programs.
pub fn dedup_subprograms(program: &Program) -> Arc<Program> {
    let mut cache: HashMap<u64, Arc<Program>> = HashMap::new();
    dedup_rec(program, &mut cache)
}

fn dedup_rec(program: &Program, cache: &mut HashMap<u64, Arc<Program>>) -> Arc<Program> {
    let sig = program_signature(program);
    if let Some(a) = cache.get(&sig) { return Arc::clone(a); }
    let new_ops: Vec<Opcode> = program.ops.iter().map(|op| rewrite_op(op, cache)).collect();
    let ics = crate::vm::fresh_ics(new_ops.len());
    let out = Arc::new(Program {
        ops:           new_ops.into(),
        source:        program.source.clone(),
        id:            program.id,
        is_structural: program.is_structural,
        ics,
    });
    cache.insert(sig, Arc::clone(&out));
    out
}

fn rewrite_op(op: &Opcode, cache: &mut HashMap<u64, Arc<Program>>) -> Opcode {
    use super::vm::{CompiledObjEntry, CompiledFSPart, CompSpec, DictCompSpec};
    match op {
        Opcode::AndOp(p)        => Opcode::AndOp(dedup_rec(p, cache)),
        Opcode::OrOp(p)         => Opcode::OrOp(dedup_rec(p, cache)),
        Opcode::CoalesceOp(p)   => Opcode::CoalesceOp(dedup_rec(p, cache)),
        Opcode::InlineFilter(p) => Opcode::InlineFilter(dedup_rec(p, cache)),
        Opcode::FilterCount(p)  => Opcode::FilterCount(dedup_rec(p, cache)),
        Opcode::MapSum(p)       => Opcode::MapSum(dedup_rec(p, cache)),
        Opcode::MapAvg(p)       => Opcode::MapAvg(dedup_rec(p, cache)),
        Opcode::MapMin(p)       => Opcode::MapMin(dedup_rec(p, cache)),
        Opcode::MapMax(p)       => Opcode::MapMax(dedup_rec(p, cache)),
        Opcode::MapFlatten(p)   => Opcode::MapFlatten(dedup_rec(p, cache)),
        Opcode::MapFirst(p)     => Opcode::MapFirst(dedup_rec(p, cache)),
        Opcode::MapLast(p)      => Opcode::MapLast(dedup_rec(p, cache)),
        Opcode::FilterLast  { pred } => Opcode::FilterLast  { pred: dedup_rec(pred, cache) },
        Opcode::FilterTakeWhile { pred, stop } => Opcode::FilterTakeWhile {
            pred: dedup_rec(pred, cache),
            stop: dedup_rec(stop, cache),
        },
        Opcode::FindFirst(p)    => Opcode::FindFirst(dedup_rec(p, cache)),
        Opcode::FindOne(p)      => Opcode::FindOne(dedup_rec(p, cache)),
        Opcode::DynIndex(p)     => Opcode::DynIndex(dedup_rec(p, cache)),
        Opcode::FilterMap { pred, map } => Opcode::FilterMap {
            pred: dedup_rec(pred, cache),
            map:  dedup_rec(map,  cache),
        },
        Opcode::FilterMapSum { pred, map } => Opcode::FilterMapSum {
            pred: dedup_rec(pred, cache),
            map:  dedup_rec(map,  cache),
        },
        Opcode::FilterMapAvg { pred, map } => Opcode::FilterMapAvg {
            pred: dedup_rec(pred, cache),
            map:  dedup_rec(map,  cache),
        },
        Opcode::FilterMapFirst { pred, map } => Opcode::FilterMapFirst {
            pred: dedup_rec(pred, cache),
            map:  dedup_rec(map,  cache),
        },
        Opcode::MapFilter { map, pred } => Opcode::MapFilter {
            map:  dedup_rec(map,  cache),
            pred: dedup_rec(pred, cache),
        },
        Opcode::FilterFilter { p1, p2 } => Opcode::FilterFilter {
            p1: dedup_rec(p1, cache),
            p2: dedup_rec(p2, cache),
        },
        Opcode::MapMap { f1, f2 } => Opcode::MapMap {
            f1: dedup_rec(f1, cache),
            f2: dedup_rec(f2, cache),
        },
        Opcode::LetExpr { name, body } => Opcode::LetExpr {
            name: name.clone(),
            body: dedup_rec(body, cache),
        },
        Opcode::CallMethod(c) => Opcode::CallMethod(rewrite_call(c, cache)),
        Opcode::CallOptMethod(c) => Opcode::CallOptMethod(rewrite_call(c, cache)),
        Opcode::MakeArr(progs) => {
            let new_progs: Vec<Arc<Program>> = progs.iter().map(|p| dedup_rec(p, cache)).collect();
            Opcode::MakeArr(new_progs.into())
        }
        Opcode::MakeObj(entries) => {
            let new_entries: Vec<CompiledObjEntry> = entries.iter().map(|e| match e {
                CompiledObjEntry::Short(s) => CompiledObjEntry::Short(s.clone()),
                CompiledObjEntry::Kv { key, prog, optional, cond } => CompiledObjEntry::Kv {
                    key: key.clone(),
                    prog: dedup_rec(prog, cache),
                    optional: *optional,
                    cond: cond.as_ref().map(|c| dedup_rec(c, cache)),
                },
                CompiledObjEntry::Dynamic { key, val } => CompiledObjEntry::Dynamic {
                    key: dedup_rec(key, cache),
                    val: dedup_rec(val, cache),
                },
                CompiledObjEntry::Spread(p) => CompiledObjEntry::Spread(dedup_rec(p, cache)),
                CompiledObjEntry::SpreadDeep(p) => CompiledObjEntry::SpreadDeep(dedup_rec(p, cache)),
            }).collect();
            Opcode::MakeObj(new_entries.into())
        }
        Opcode::FString(parts) => {
            let new_parts: Vec<CompiledFSPart> = parts.iter().map(|p| match p {
                CompiledFSPart::Lit(s) => CompiledFSPart::Lit(s.clone()),
                CompiledFSPart::Interp { prog, fmt } => CompiledFSPart::Interp {
                    prog: dedup_rec(prog, cache),
                    fmt:  fmt.clone(),
                },
            }).collect();
            Opcode::FString(new_parts.into())
        }
        Opcode::ListComp(spec) => {
            let new_spec = CompSpec {
                expr:  dedup_rec(&spec.expr, cache),
                iter:  dedup_rec(&spec.iter, cache),
                cond:  spec.cond.as_ref().map(|c| dedup_rec(c, cache)),
                vars:  spec.vars.clone(),
            };
            Opcode::ListComp(Arc::new(new_spec))
        }
        Opcode::SetComp(spec) => {
            let new_spec = CompSpec {
                expr:  dedup_rec(&spec.expr, cache),
                iter:  dedup_rec(&spec.iter, cache),
                cond:  spec.cond.as_ref().map(|c| dedup_rec(c, cache)),
                vars:  spec.vars.clone(),
            };
            Opcode::SetComp(Arc::new(new_spec))
        }
        Opcode::DictComp(spec) => {
            let new_spec = DictCompSpec {
                key:   dedup_rec(&spec.key, cache),
                val:   dedup_rec(&spec.val, cache),
                iter:  dedup_rec(&spec.iter, cache),
                cond:  spec.cond.as_ref().map(|c| dedup_rec(c, cache)),
                vars:  spec.vars.clone(),
            };
            Opcode::DictComp(Arc::new(new_spec))
        }
        _ => op.clone(),
    }
}

fn rewrite_call(c: &Arc<super::vm::CompiledCall>, cache: &mut HashMap<u64, Arc<Program>>) -> Arc<super::vm::CompiledCall> {
    use super::vm::CompiledCall;
    let new_subs: Vec<Arc<Program>> = c.sub_progs.iter().map(|p| dedup_rec(p, cache)).collect();
    Arc::new(CompiledCall {
        method:    c.method,
        name:      c.name.clone(),
        sub_progs: new_subs.into(),
        orig_args: c.orig_args.clone(),
    })
}

// ── Cost model ────────────────────────────────────────────────────────────────

/// Rough, ordinal cost estimate per opcode.  Not a wall-clock number — only
/// useful to compare relative cost (e.g. for AndOp operand reordering).
pub fn opcode_cost(op: &Opcode) -> u32 {
    match op {
        Opcode::PushNull | Opcode::PushBool(_) | Opcode::PushInt(_)
            | Opcode::PushFloat(_) | Opcode::PushStr(_)
            | Opcode::PushRoot | Opcode::PushCurrent | Opcode::LoadIdent(_) => 1,
        Opcode::GetField(_) | Opcode::OptField(_) | Opcode::GetIndex(_)
            | Opcode::RootChain(_) | Opcode::FieldChain(_)
            | Opcode::GetPointer(_) => 2,
        Opcode::GetSlice(..) | Opcode::Descendant(_) => 5,
        Opcode::DescendAll => 20,
        Opcode::Not | Opcode::Neg | Opcode::SetCurrent
            | Opcode::BindVar(_) | Opcode::StoreVar(_)
            | Opcode::BindObjDestructure(_) | Opcode::BindArrDestructure(_) => 1,
        Opcode::Add | Opcode::Sub | Opcode::Mul | Opcode::Div | Opcode::Mod
            | Opcode::Eq | Opcode::Neq | Opcode::Lt | Opcode::Lte
            | Opcode::Gt | Opcode::Gte | Opcode::Fuzzy => 2,
        Opcode::KindCheck { .. } => 2,
        Opcode::AndOp(p) | Opcode::OrOp(p) | Opcode::CoalesceOp(p) => 2 + program_cost(p),
        Opcode::InlineFilter(p) | Opcode::FilterCount(p)
            | Opcode::FindFirst(p) | Opcode::FindOne(p)
            | Opcode::MapSum(p) | Opcode::MapAvg(p)
            | Opcode::MapMin(p) | Opcode::MapMax(p)
            | Opcode::MapFlatten(p)
            | Opcode::MapFirst(p) | Opcode::MapLast(p)
            | Opcode::FilterLast { pred: p }
            | Opcode::DynIndex(p) => 10 + program_cost(p),
        Opcode::FilterTakeWhile { pred, stop } => 10 + program_cost(pred) + program_cost(stop),
        Opcode::FilterDropWhile { pred, drop } => 10 + program_cost(pred) + program_cost(drop),
        Opcode::MapUnique(p) => 15 + program_cost(p),
        Opcode::EquiJoin { rhs, .. } => 25 + program_cost(rhs),
        Opcode::FilterMap { pred, map }
            | Opcode::FilterMapSum { pred, map }
            | Opcode::FilterMapAvg { pred, map }
            | Opcode::FilterMapFirst { pred, map } => 10 + program_cost(pred) + program_cost(map),
        Opcode::MapFilter { map, pred } => 10 + program_cost(map) + program_cost(pred),
        Opcode::FilterFilter { p1, p2 } => 10 + program_cost(p1) + program_cost(p2),
        Opcode::MapMap { f1, f2 } => 10 + program_cost(f1) + program_cost(f2),
        Opcode::TopN { n, .. } => 15 + *n as u32,
        Opcode::CallMethod(c) | Opcode::CallOptMethod(c) => {
            let base = match c.method {
                BuiltinMethod::Filter | BuiltinMethod::Map | BuiltinMethod::FlatMap => 10,
                BuiltinMethod::Sort   => 30,
                BuiltinMethod::GroupBy | BuiltinMethod::IndexBy => 25,
                BuiltinMethod::Len    | BuiltinMethod::Count => 2,
                _ => 8,
            };
            base + c.sub_progs.iter().map(|p| program_cost(p)).sum::<u32>()
        }
        Opcode::MakeObj(entries) => {
            use super::vm::CompiledObjEntry;
            entries.iter().map(|e| match e {
                CompiledObjEntry::Short(_) => 2,
                CompiledObjEntry::Kv { prog, cond, .. } =>
                    2 + program_cost(prog) + cond.as_ref().map_or(0, |c| program_cost(c)),
                CompiledObjEntry::Dynamic { key, val } => 3 + program_cost(key) + program_cost(val),
                CompiledObjEntry::Spread(p) => 5 + program_cost(p),
                CompiledObjEntry::SpreadDeep(p) => 8 + program_cost(p),
            }).sum()
        }
        Opcode::MakeArr(progs) => progs.iter().map(|p| 1 + program_cost(p)).sum(),
        Opcode::FString(parts) => {
            use super::vm::CompiledFSPart;
            parts.iter().map(|p| match p {
                CompiledFSPart::Lit(_) => 1,
                CompiledFSPart::Interp { prog, .. } => 3 + program_cost(prog),
            }).sum()
        }
        Opcode::ListComp(s) | Opcode::SetComp(s) =>
            15 + program_cost(&s.expr) + program_cost(&s.iter)
               + s.cond.as_ref().map_or(0, |c| program_cost(c)),
        Opcode::DictComp(s) =>
            15 + program_cost(&s.key) + program_cost(&s.val) + program_cost(&s.iter)
               + s.cond.as_ref().map_or(0, |c| program_cost(c)),
        Opcode::LetExpr { body, .. } => 2 + program_cost(body),
        Opcode::Quantifier(_) => 2,
        Opcode::CastOp(_) => 2,
        Opcode::PatchEval(_) => 50,
        Opcode::MapFieldSum(_) | Opcode::MapFieldAvg(_)
            | Opcode::MapFieldMin(_) | Opcode::MapFieldMax(_)
            | Opcode::MapField(_) => 5,
        Opcode::MapFieldUnique(_) => 8,
        Opcode::FlatMapChain(ks) => 5 + ks.len() as u32 * 3,
        Opcode::FilterFieldEqLit(_, _) | Opcode::FilterFieldCmpLit(_, _, _)
            | Opcode::FilterFieldCmpField(_, _, _) => 5,
        Opcode::FilterFieldEqLitCount(_, _) | Opcode::FilterFieldCmpLitCount(_, _, _)
            | Opcode::FilterFieldCmpFieldCount(_, _, _) => 4,
        Opcode::GroupByField(_) => 15,
    }
}

/// Total cost of a program (sum of per-op costs).
pub fn program_cost(program: &Program) -> u32 {
    program.ops.iter().map(opcode_cost).sum()
}

// ── Monotonicity lattice ─────────────────────────────────────────────────────

/// Tracks ordering properties of array-like values through the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Monotonicity {
    /// Order unknown.
    Unknown,
    /// Ascending by natural comparator.
    Asc,
    /// Descending by natural comparator.
    Desc,
    /// Not an ordered collection.
    NotArray,
}

impl Monotonicity {
    pub fn after(self, op: &Opcode) -> Monotonicity {
        match op {
            Opcode::CallMethod(c) if c.sub_progs.is_empty() => match c.method {
                BuiltinMethod::Sort    => Monotonicity::Asc,
                BuiltinMethod::Reverse => match self {
                    Monotonicity::Asc  => Monotonicity::Desc,
                    Monotonicity::Desc => Monotonicity::Asc,
                    x => x,
                },
                BuiltinMethod::Filter  => self, // order preserved
                BuiltinMethod::Map     => Monotonicity::Unknown, // key changes
                _ => Monotonicity::Unknown,
            }
            Opcode::TopN { asc, .. } => if *asc { Monotonicity::Asc } else { Monotonicity::Desc },
            Opcode::MakeArr(_) | Opcode::ListComp(_) => Monotonicity::Unknown,
            _ => self,
        }
    }
}

/// Walk program and determine monotonicity of the final result.
pub fn infer_monotonicity(program: &Program) -> Monotonicity {
    let mut m = Monotonicity::Unknown;
    for op in program.ops.iter() { m = m.after(op); }
    m
}

// ── Escape analysis ───────────────────────────────────────────────────────────

/// Simple escape check: does the program's final value contain references to
/// the input document (i.e. survive returning)?  If not, the compiler may
/// emit value-copying ops rather than Arc-sharing to free the original doc.
///
/// Returns `false` only when the result is a scalar or newly-constructed
/// object/array with no root references.
pub fn escapes_doc(program: &Program) -> bool {
    for op in program.ops.iter() {
        match op {
            Opcode::PushRoot | Opcode::PushCurrent | Opcode::RootChain(_)
                | Opcode::GetField(_) | Opcode::GetIndex(_) | Opcode::GetSlice(..)
                | Opcode::Descendant(_) | Opcode::DescendAll
                | Opcode::GetPointer(_) | Opcode::OptField(_) => return true,
            Opcode::CallMethod(c) | Opcode::CallOptMethod(c)
                if c.sub_progs.iter().any(|p| escapes_doc(p)) => return true,
            _ => {}
        }
    }
    false
}

// ── Selectivity scoring ──────────────────────────────────────────────────────

/// Rough selectivity estimate for AST predicates.  Lower score → more
/// selective (filters out more rows).  Used to reorder `and` operands so
/// cheaper / more-selective predicate runs first (short-circuit friendly).
pub fn selectivity_score(expr: &super::ast::Expr) -> u32 {
    use super::ast::{Expr, BinOp};
    match expr {
        Expr::Bool(true) => 1000,        // no filtering
        Expr::Bool(false) => 0,          // max filtering
        Expr::BinOp(_, BinOp::Eq, _)    => 1,
        Expr::BinOp(_, BinOp::Neq, _)   => 5,
        Expr::BinOp(_, BinOp::Lt, _) | Expr::BinOp(_, BinOp::Gt, _)
            | Expr::BinOp(_, BinOp::Lte, _) | Expr::BinOp(_, BinOp::Gte, _) => 3,
        Expr::BinOp(_, BinOp::Fuzzy, _) => 2,
        Expr::BinOp(l, BinOp::And, r) =>
            selectivity_score(l).min(selectivity_score(r)),
        Expr::BinOp(l, BinOp::Or, r)  =>
            selectivity_score(l) + selectivity_score(r),
        Expr::Not(e) => 10u32.saturating_sub(selectivity_score(e)),
        Expr::Kind { .. } => 2,
        _ => 5,
    }
}

// ── Kind-check specialisation ────────────────────────────────────────────────

/// When a `KindCheck` is applied to a value with a statically-known type,
/// the check can be constant-folded to true/false at compile time.
pub fn fold_kind_check(val_ty: VType, target: KindType, negate: bool) -> Option<bool> {
    let matches = match (val_ty, target) {
        (VType::Null, KindType::Null)   => true,
        (VType::Bool, KindType::Bool)   => true,
        (VType::Int | VType::Float | VType::Num, KindType::Number) => true,
        (VType::Str,  KindType::Str)    => true,
        (VType::Arr,  KindType::Array)  => true,
        (VType::Obj,  KindType::Object) => true,
        (VType::Unknown, _)             => return None,
        _                               => false,
    };
    Some(if negate { !matches } else { matches })
}
