//! Forward-flow static analyses over compiled `Program` IR.
//!
//! Analyses run on the flat `Arc<[Opcode]>` representation after compilation.
//! They are conservative (return the `Unknown` top element when uncertain)
//! and used by the compiler for peephole specialisation and by the planner
//! for CSE (`dedup_subprograms`). None affect runtime correctness.

use std::collections::HashMap;
use std::sync::Arc;

use crate::parse::ast::KindType;
use crate::vm::{CompiledPipeStep, Opcode, Program};
use crate::builtins::BuiltinMethod;


/// Type lattice element. Ordered: `Bottom` ⊑ concrete types ⊑ `Unknown`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VType {
    /// Unreachable position; join with any type yields that type.
    Bottom,
    /// Definitely `Val::Null`.
    Null,
    /// Definitely `Val::Bool`.
    Bool,
    /// Definitely `Val::Int` (i64).
    Int,
    /// Definitely `Val::Float` (f64).
    Float,
    /// Either `Int` or `Float`; the join of both concrete numeric types.
    Num,
    /// Definitely a string value.
    Str,
    /// Definitely `Val::Arr`.
    Arr,
    /// Definitely `Val::Obj`.
    Obj,
    /// Any type possible — top element, used when analysis cannot determine the type.
    Unknown,
}

impl VType {
    /// Lattice join: return the least upper bound of `self` and `other`.
    /// `Int ⊔ Float = Num`; incompatible concrete types collapse to `Unknown`.
    pub fn join(self, other: VType) -> VType {
        if self == other {
            return self;
        }
        match (self, other) {
            (VType::Bottom, x) | (x, VType::Bottom) => x,
            (VType::Int, VType::Float)
            | (VType::Float, VType::Int)
            | (VType::Int, VType::Num)
            | (VType::Num, VType::Int)
            | (VType::Float, VType::Num)
            | (VType::Num, VType::Float) => VType::Num,
            _ => VType::Unknown,
        }
    }

    /// Return `true` only for `Arr`; used to guard array-specific optimisations.
    pub fn is_array_like(self) -> bool {
        matches!(self, VType::Arr)
    }
    /// Return `true` only for `Obj`; used to guard object-specific optimisations.
    pub fn is_object_like(self) -> bool {
        matches!(self, VType::Obj)
    }
    /// Return `true` for any numeric variant (`Int`, `Float`, or `Num`).
    pub fn is_numeric(self) -> bool {
        matches!(self, VType::Int | VType::Float | VType::Num)
    }
    /// Return `true` only when the type is definitely a string.
    pub fn is_string(self) -> bool {
        matches!(self, VType::Str)
    }
}


/// Nullness lattice element tracking whether a value can ever be `null`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Nullness {
    /// The value is always `Val::Null` at this program point.
    AlwaysNull,
    /// The value is never null at this program point.
    NonNull,
    /// The value may or may not be null; the conservative top element.
    MaybeNull,
}

impl Nullness {
    /// Lattice join: any disagreement between `AlwaysNull` and `NonNull` yields `MaybeNull`.
    pub fn join(self, other: Nullness) -> Nullness {
        if self == other {
            return self;
        }
        Nullness::MaybeNull
    }
}


/// Cardinality lattice element describing how many values a program position produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Cardinality {
    /// Produces no values (empty result / unreachable branch).
    Zero,
    /// Produces exactly one value.
    One,
    /// Produces zero or one values (e.g. optional field access).
    ZeroOrOne,
    /// Produces two or more values (array output).
    Many,
    /// The value is a scalar — not wrapped in an array.
    NotArray,
    /// Cardinality is indeterminate; conservative top element.
    Unknown,
}

impl Cardinality {
    /// Lattice join: `Zero ⊔ One = ZeroOrOne`; all other mixed pairs collapse to `Unknown`.
    pub fn join(self, other: Cardinality) -> Cardinality {
        if self == other {
            return self;
        }
        match (self, other) {
            (Cardinality::Zero, Cardinality::One) | (Cardinality::One, Cardinality::Zero) => {
                Cardinality::ZeroOrOne
            }
            _ => Cardinality::Unknown,
        }
    }
}


/// Product of all three lattice dimensions for a single program point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AbstractVal {
    /// Inferred type of the value.
    pub ty: VType,
    /// Nullness of the value.
    pub null: Nullness,
    /// Cardinality (scalar vs. array, count) of the value.
    pub card: Cardinality,
}

impl AbstractVal {
    /// Fully conservative element; used as the initial stack value and when analysis fails.
    pub const UNKNOWN: Self = Self {
        ty: VType::Unknown,
        null: Nullness::MaybeNull,
        card: Cardinality::Unknown,
    };
    /// Abstract value for a definite null literal.
    pub const NULL: Self = Self {
        ty: VType::Null,
        null: Nullness::AlwaysNull,
        card: Cardinality::NotArray,
    };
    /// Construct an abstract scalar (non-null, non-array) with the given type.
    pub fn scalar(ty: VType) -> Self {
        Self {
            ty,
            null: Nullness::NonNull,
            card: Cardinality::NotArray,
        }
    }
    /// Construct the canonical abstract value for a non-null array result.
    pub fn array() -> Self {
        Self {
            ty: VType::Arr,
            null: Nullness::NonNull,
            card: Cardinality::Many,
        }
    }
    /// Construct the canonical abstract value for a non-null object result.
    pub fn object() -> Self {
        Self {
            ty: VType::Obj,
            null: Nullness::NonNull,
            card: Cardinality::NotArray,
        }
    }
    /// Component-wise lattice join over all three dimensions.
    pub fn join(self, other: AbstractVal) -> AbstractVal {
        AbstractVal {
            ty: self.ty.join(other.ty),
            null: self.null.join(other.null),
            card: self.card.join(other.card),
        }
    }
}


/// Run the forward-flow type analysis over `program` and return the abstract
/// value at the top of the stack after the last opcode.
pub fn infer_result_type(program: &Program) -> AbstractVal {
    let mut stack: Vec<AbstractVal> = Vec::with_capacity(16);
    let mut env: HashMap<Arc<str>, AbstractVal> = HashMap::new();
    for op in program.ops.iter() {
        apply_op_env(op, &mut stack, &mut env);
    }
    stack.pop().unwrap_or(AbstractVal::UNKNOWN)
}


/// Like `infer_result_type` but also returns the variable environment so the
/// caller can inspect inferred types for named bindings.
pub fn infer_result_type_with_env(
    program: &Program,
) -> (AbstractVal, HashMap<Arc<str>, AbstractVal>) {
    let mut stack: Vec<AbstractVal> = Vec::with_capacity(16);
    let mut env: HashMap<Arc<str>, AbstractVal> = HashMap::new();
    for op in program.ops.iter() {
        apply_op_env(op, &mut stack, &mut env);
    }
    (stack.pop().unwrap_or(AbstractVal::UNKNOWN), env)
}

/// Apply a single opcode to the abstract stack, threading the variable environment
/// for `LetExpr` and `LoadIdent`; delegates to `apply_op` for all other opcodes.
fn apply_op_env(
    op: &Opcode,
    stack: &mut Vec<AbstractVal>,
    env: &mut HashMap<Arc<str>, AbstractVal>,
) {
    // Handle the two opcodes that read/write the variable environment.
    match op {
        Opcode::LoadIdent(name) => {
            let av = env.get(name).copied().unwrap_or(AbstractVal::UNKNOWN);
            stack.push(av);
        }
        Opcode::LetExpr { name, body } => {
            let init = stack.pop().unwrap_or(AbstractVal::UNKNOWN);
            let saved = env.get(name).copied();
            env.insert(name.clone(), init);
            let mut sub_stack: Vec<AbstractVal> = Vec::with_capacity(8);
            for op2 in body.ops.iter() {
                apply_op_env(op2, &mut sub_stack, env);
            }
            let res = sub_stack.pop().unwrap_or(AbstractVal::UNKNOWN);
            // Restore the variable environment to its state before the let binding.
            match saved {
                Some(v) => {
                    env.insert(name.clone(), v);
                }
                None => {
                    env.remove(name);
                }
            }
            stack.push(res);
        }
        _ => apply_op(op, stack),
    }
}

/// Apply a single opcode to the abstract stack, producing abstract output values
/// from abstract input values without touching the variable environment.
fn apply_op(op: &Opcode, stack: &mut Vec<AbstractVal>) {
    macro_rules! pop2 {
        () => {{
            let r = stack.pop().unwrap_or(AbstractVal::UNKNOWN);
            let l = stack.pop().unwrap_or(AbstractVal::UNKNOWN);
            (l, r)
        }};
    }
    macro_rules! pop1 {
        () => {
            stack.pop().unwrap_or(AbstractVal::UNKNOWN)
        };
    }
    match op {
        Opcode::PushNull => stack.push(AbstractVal::NULL),
        Opcode::PushBool(_) => stack.push(AbstractVal::scalar(VType::Bool)),
        Opcode::PushInt(_) => stack.push(AbstractVal::scalar(VType::Int)),
        Opcode::PushFloat(_) => stack.push(AbstractVal::scalar(VType::Float)),
        Opcode::PushStr(_) => stack.push(AbstractVal::scalar(VType::Str)),
        Opcode::PushRoot | Opcode::PushCurrent | Opcode::LoadIdent(_) => {
            stack.push(AbstractVal::UNKNOWN)
        }
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
            use crate::parse::ast::QuantifierKind;
            let card = match kind {
                QuantifierKind::First => Cardinality::ZeroOrOne,
                QuantifierKind::One => Cardinality::One,
            };
            stack.push(AbstractVal {
                ty: VType::Unknown,
                null: Nullness::MaybeNull,
                card,
            });
        }
        Opcode::RootChain(_) => stack.push(AbstractVal::UNKNOWN),
        Opcode::Add | Opcode::Sub | Opcode::Mul | Opcode::Mod => {
            let (l, r) = pop2!();
            stack.push(AbstractVal::scalar(l.ty.join(r.ty)));
        }
        Opcode::Div => {
            pop2!();
            stack.push(AbstractVal::scalar(VType::Float));
        }
        Opcode::Eq
        | Opcode::Neq
        | Opcode::Lt
        | Opcode::Lte
        | Opcode::Gt
        | Opcode::Gte
        | Opcode::Fuzzy => {
            pop2!();
            stack.push(AbstractVal::scalar(VType::Bool));
        }
        Opcode::Not => {
            pop1!();
            stack.push(AbstractVal::scalar(VType::Bool));
        }
        Opcode::Neg => {
            let v = pop1!();
            stack.push(AbstractVal::scalar(v.ty));
        }
        Opcode::AndOp(_) | Opcode::OrOp(_) => {
            pop1!();
            stack.push(AbstractVal::scalar(VType::Bool));
        }
        Opcode::CoalesceOp(_) => {
            pop1!();
            stack.push(AbstractVal::UNKNOWN);
        }
        Opcode::IfElse { .. } => {
            pop1!();
            stack.push(AbstractVal::UNKNOWN);
        }
        Opcode::TryExpr { .. } => {
            stack.push(AbstractVal::UNKNOWN);
        }
        Opcode::CallMethod(call) | Opcode::CallOptMethod(call) => {
            pop1!();
            stack.push(method_result_type(call.method));
        }
        Opcode::MakeObj(_) => stack.push(AbstractVal::object()),
        Opcode::MakeArr(_) => stack.push(AbstractVal::array()),
        Opcode::FString(_) => stack.push(AbstractVal::scalar(VType::Str)),
        Opcode::KindCheck { .. } => {
            pop1!();
            stack.push(AbstractVal::scalar(VType::Bool));
        }
        Opcode::SetCurrent => {} // Side-effecting; does not push a value.
        Opcode::LetExpr { .. } => {
            pop1!();
            stack.push(AbstractVal::UNKNOWN);
        }
        Opcode::ListComp(_) | Opcode::SetComp(_) => stack.push(AbstractVal::array()),
        Opcode::DictComp(_) => stack.push(AbstractVal::object()),
        Opcode::PatchEval(_) => stack.push(AbstractVal::UNKNOWN),
        Opcode::CastOp(ty) => {
            pop1!();
            use crate::parse::ast::CastType;
            let av = match ty {
                CastType::Int => AbstractVal::scalar(VType::Int),
                CastType::Float => AbstractVal::scalar(VType::Float),
                CastType::Number => AbstractVal::scalar(VType::Num),
                CastType::Str => AbstractVal::scalar(VType::Str),
                CastType::Bool => AbstractVal::scalar(VType::Bool),
                CastType::Array => AbstractVal::array(),
                CastType::Object => AbstractVal::object(),
                CastType::Null => AbstractVal::NULL,
            };
            stack.push(av);
        }
        Opcode::PipelineRun { .. } => stack.push(AbstractVal::UNKNOWN),
        Opcode::DeleteMarkErr => stack.push(AbstractVal::UNKNOWN),
        Opcode::Match(_) => stack.push(AbstractVal::UNKNOWN),
    }
}


/// Return the statically known result type of a builtin method call.
/// Grouped by return-type family; methods whose return type is data-dependent
/// fall through to `AbstractVal::UNKNOWN`.
pub fn method_result_type(m: BuiltinMethod) -> AbstractVal {
    use BuiltinMethod::*;
    match m {
        // Integer-returning methods.
        Len | Count | Sum | ApproxCountDistinct | IndexOf | LastIndexOf | ByteLen | ParseInt
        | Ceil | Floor | Round => AbstractVal::scalar(VType::Int),
        // Boolean-returning methods.
        Any | All | Has | Missing | Includes | StartsWith | EndsWith | IsBlank | IsNumeric
        | IsAlpha | IsAscii | ParseBool | ReMatch | ContainsAny | ContainsAll => {
            AbstractVal::scalar(VType::Bool)
        }
        // String-returning methods.
        Upper | Lower | Capitalize | TitleCase | Trim | TrimLeft | TrimRight | ToString
        | ToJson | ToBase64 | FromBase64 | UrlEncode | UrlDecode | HtmlEscape | HtmlUnescape
        | Repeat | PadLeft | PadRight | Replace | ReplaceAll | StripPrefix | StripSuffix
        | Indent | Dedent | Join | ToCsv | ToTsv | Type | SnakeCase | KebabCase | CamelCase
        | PascalCase | ReverseStr | Center => AbstractVal::scalar(VType::Str),
        // Float-returning methods.
        Avg | ParseFloat => AbstractVal::scalar(VType::Float),
        // Polymorphic-numeric methods; exact type depends on input.
        Min | Max | ToNumber | Abs => AbstractVal::scalar(VType::Num),
        // Explicit bool coercion.
        ToBool => AbstractVal::scalar(VType::Bool),
        // Array-returning methods (includes collection, transform, and window operations).
        Keys | Values | Entries | ToPairs | Reverse | Unique | Collect | Flatten | Compact
        | Chars | CharsOf | Lines | Words | Split | Sort | Filter | Map | FlatMap | Find
        | FindAll | UniqueBy | DeepFind | DeepShape | DeepLike | IndicesWhere | Fanout
        | TracePath | Enumerate | Pairwise | Window | Chunk | TakeWhile | DropWhile | Take
        | Skip | Accumulate | Zip | ZipLongest | Diff | Intersect | Union | Append | Prepend
        | Remove | Matches | Scan | Slice | Bytes | IndicesOf | Explode | Implode | RollingSum
        | RollingAvg | RollingMin | RollingMax | Lag | Lead | DiffWindow | PctChange | CumMax
        | CumMin | Zscore => AbstractVal::array(),
        // Object-returning methods.
        FromPairs | Invert | Pick | Omit | Merge | DeepMerge | Defaults | Rename
        | TransformKeys | TransformValues | FilterKeys | FilterValues | Pivot | GroupBy
        | CountBy | IndexBy | GroupShape | ZipShape | Partition | FlattenKeys | UnflattenKeys
        | SetPath | DelPath | DelPaths | Update | Schema => AbstractVal::object(),
        // Scalar-returning methods whose type cannot be determined without runtime information.
        First | Last | Nth | FindFirst | FindOne | FindIndex | MaxBy | MinBy | Walk | WalkPre
        | Rec | GetPath | ReMatchFirst | ReCaptures => AbstractVal::UNKNOWN,
        HasPath => AbstractVal::scalar(VType::Bool),
        ReMatchAll | ReCapturesAll | ReSplit => AbstractVal::array(),
        ReReplace | ReReplaceAll => AbstractVal::scalar(VType::Str),
        FromJson | Or | Set | Index => AbstractVal::UNKNOWN,
        EquiJoin => AbstractVal::array(),
        Unknown => AbstractVal::UNKNOWN,
    }
}


/// Count how many times `name` is referenced as `Opcode::LoadIdent` across
/// `program` and all nested sub-programs; used for inlining decisions.
pub fn count_ident_uses(program: &Program, name: &str) -> usize {
    let mut n = 0;
    count_ident_uses_in_ops(&program.ops, name, &mut n);
    n
}

/// Recursive helper for `count_ident_uses`; descends into every embedded `Program`.
fn count_ident_uses_in_ops(ops: &[Opcode], name: &str, acc: &mut usize) {
    for op in ops {
        match op {
            Opcode::LoadIdent(s) if s.as_ref() == name => *acc += 1,
            Opcode::AndOp(p)
            | Opcode::OrOp(p)
            | Opcode::CoalesceOp(p)
            | Opcode::InlineFilter(p)
            | Opcode::DynIndex(p) => count_ident_uses_in_ops(&p.ops, name, acc),
            Opcode::IfElse { then_, else_ } => {
                count_ident_uses_in_ops(&then_.ops, name, acc);
                count_ident_uses_in_ops(&else_.ops, name, acc);
            }
            Opcode::CallMethod(c) | Opcode::CallOptMethod(c) => {
                for p in c.sub_progs.iter() {
                    count_ident_uses_in_ops(&p.ops, name, acc);
                }
            }
            Opcode::LetExpr { body, .. } => count_ident_uses_in_ops(&body.ops, name, acc),
            Opcode::ListComp(spec) | Opcode::SetComp(spec) => {
                count_ident_uses_in_ops(&spec.expr.ops, name, acc);
                count_ident_uses_in_ops(&spec.iter.ops, name, acc);
                if let Some(c) = &spec.cond {
                    count_ident_uses_in_ops(&c.ops, name, acc);
                }
            }
            Opcode::DictComp(spec) => {
                count_ident_uses_in_ops(&spec.key.ops, name, acc);
                count_ident_uses_in_ops(&spec.val.ops, name, acc);
                count_ident_uses_in_ops(&spec.iter.ops, name, acc);
                if let Some(c) = &spec.cond {
                    count_ident_uses_in_ops(&c.ops, name, acc);
                }
            }
            Opcode::MakeObj(entries) => {
                use crate::vm::CompiledObjEntry;
                for e in entries.iter() {
                    match e {
                        CompiledObjEntry::Short { .. } => {}
                        CompiledObjEntry::Kv { prog, cond, .. } => {
                            count_ident_uses_in_ops(&prog.ops, name, acc);
                            if let Some(c) = cond {
                                count_ident_uses_in_ops(&c.ops, name, acc);
                            }
                        }
                        CompiledObjEntry::KvPath { .. } => {}
                        CompiledObjEntry::Dynamic { key, val } => {
                            count_ident_uses_in_ops(&key.ops, name, acc);
                            count_ident_uses_in_ops(&val.ops, name, acc);
                        }
                        CompiledObjEntry::Spread(p) => count_ident_uses_in_ops(&p.ops, name, acc),
                        CompiledObjEntry::SpreadDeep(p) => {
                            count_ident_uses_in_ops(&p.ops, name, acc)
                        }
                    }
                }
            }
            Opcode::MakeArr(progs) => {
                for (p, _) in progs.iter() {
                    count_ident_uses_in_ops(&p.ops, name, acc);
                }
            }
            Opcode::FString(parts) => {
                use crate::vm::CompiledFSPart;
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


/// Collect every field name statically accessed by `program` (via `GetField`,
/// `OptField`, `Descendant`, or `RootChain`). De-duplicated; order is discovery order.
pub fn collect_accessed_fields(program: &Program) -> Vec<Arc<str>> {
    let mut set = Vec::new();
    collect_fields_in_ops(&program.ops, &mut set);
    set
}

/// Recursive helper for `collect_accessed_fields`.
fn collect_fields_in_ops(ops: &[Opcode], acc: &mut Vec<Arc<str>>) {
    for op in ops {
        match op {
            Opcode::GetField(k) | Opcode::OptField(k) | Opcode::Descendant(k) => {
                if !acc.iter().any(|a: &Arc<str>| a == k) {
                    acc.push(k.clone());
                }
            }
            Opcode::RootChain(chain) => {
                for k in chain.iter() {
                    if !acc.iter().any(|a: &Arc<str>| a == k) {
                        acc.push(k.clone());
                    }
                }
            }
            Opcode::AndOp(p)
            | Opcode::OrOp(p)
            | Opcode::CoalesceOp(p)
            | Opcode::InlineFilter(p)
            | Opcode::DynIndex(p) => collect_fields_in_ops(&p.ops, acc),
            Opcode::IfElse { then_, else_ } => {
                collect_fields_in_ops(&then_.ops, acc);
                collect_fields_in_ops(&else_.ops, acc);
            }
            Opcode::CallMethod(c) | Opcode::CallOptMethod(c) => {
                for p in c.sub_progs.iter() {
                    collect_fields_in_ops(&p.ops, acc);
                }
            }
            Opcode::LetExpr { body, .. } => collect_fields_in_ops(&body.ops, acc),
            _ => {}
        }
    }
}


/// Compute a structural hash of `program` that identifies its opcode sequence.
/// Used as a key for CSE deduplication and the compiled-program cache.
pub fn program_signature(program: &Program) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;
    let mut h = DefaultHasher::new();
    hash_ops(&program.ops, &mut h);
    h.finish()
}

/// Recursively hash an opcode slice; only structurally significant fields are hashed
/// (discriminant + key literals + sub-program signatures).
fn hash_ops(ops: &[Opcode], h: &mut impl std::hash::Hasher) {
    use std::hash::Hash;
    for op in ops {
        // Always hash the variant discriminant so different opcodes don't collide.
        std::mem::discriminant(op).hash(h);
        match op {
            Opcode::PushInt(n) => n.hash(h),
            Opcode::PushStr(s) => s.as_bytes().hash(h),
            Opcode::PushBool(b) => b.hash(h),
            Opcode::GetField(k)
            | Opcode::OptField(k)
            | Opcode::Descendant(k)
            | Opcode::LoadIdent(k) => k.as_bytes().hash(h),
            Opcode::GetIndex(i) => i.hash(h),
            Opcode::CallMethod(c) | Opcode::CallOptMethod(c) => {
                (c.method as u8).hash(h);
                for p in c.sub_progs.iter() {
                    hash_ops(&p.ops, h);
                }
            }
            Opcode::AndOp(p)
            | Opcode::OrOp(p)
            | Opcode::CoalesceOp(p)
            | Opcode::InlineFilter(p)
            | Opcode::DynIndex(p) => hash_ops(&p.ops, h),
            Opcode::IfElse { then_, else_ } => {
                hash_ops(&then_.ops, h);
                hash_ops(&else_.ops, h);
            }
            Opcode::RootChain(chain) => {
                for k in chain.iter() {
                    k.as_bytes().hash(h);
                }
            }
            Opcode::MakeObj(entries) => {
                use crate::vm::CompiledObjEntry;
                for e in entries.iter() {
                    match e {
                        CompiledObjEntry::Short { name, .. } => {
                            0u8.hash(h);
                            name.as_bytes().hash(h);
                        }
                        CompiledObjEntry::Kv {
                            key,
                            prog,
                            optional,
                            cond,
                        } => {
                            1u8.hash(h);
                            key.as_bytes().hash(h);
                            optional.hash(h);
                            hash_ops(&prog.ops, h);
                            if let Some(c) = cond {
                                hash_ops(&c.ops, h);
                            }
                        }
                        CompiledObjEntry::KvPath {
                            key,
                            steps,
                            optional,
                            ..
                        } => {
                            2u8.hash(h);
                            key.as_bytes().hash(h);
                            optional.hash(h);
                            for st in steps.iter() {
                                use crate::vm::KvStep;
                                match st {
                                    KvStep::Field(f) => {
                                        0u8.hash(h);
                                        f.as_bytes().hash(h);
                                    }
                                    KvStep::Index(i) => {
                                        1u8.hash(h);
                                        i.hash(h);
                                    }
                                }
                            }
                        }
                        CompiledObjEntry::Dynamic { key, val } => {
                            3u8.hash(h);
                            hash_ops(&key.ops, h);
                            hash_ops(&val.ops, h);
                        }
                        CompiledObjEntry::Spread(p) => {
                            4u8.hash(h);
                            hash_ops(&p.ops, h);
                        }
                        CompiledObjEntry::SpreadDeep(p) => {
                            5u8.hash(h);
                            hash_ops(&p.ops, h);
                        }
                    }
                }
            }
            Opcode::MakeArr(entries) => {
                for (p, sp) in entries.iter() {
                    sp.hash(h);
                    hash_ops(&p.ops, h);
                }
            }
            _ => {}
        }
    }
}


/// Walk `program` and its nested sub-programs, recording every sub-program
/// signature and how many times it appears; entries with count ≥ 2 are CSE candidates.
pub fn find_common_subexprs(program: &Program) -> HashMap<u64, usize> {
    let mut map: HashMap<u64, usize> = HashMap::new();
    walk_subprograms(&program.ops, &mut map);
    map.retain(|_, &mut n| n >= 2);
    map
}

/// Recursive helper for `find_common_subexprs`; increments a counter for each
/// sub-program encountered and recurses into its opcodes.
fn walk_subprograms(ops: &[Opcode], map: &mut HashMap<u64, usize>) {
    for op in ops {
        let sub_progs: Vec<&Arc<Program>> = match op {
            Opcode::AndOp(p)
            | Opcode::OrOp(p)
            | Opcode::CoalesceOp(p)
            | Opcode::InlineFilter(p)
            | Opcode::DynIndex(p) => vec![p],
            Opcode::IfElse { then_, else_ } => vec![then_, else_],
            Opcode::CallMethod(c) | Opcode::CallOptMethod(c) => c.sub_progs.iter().collect(),
            Opcode::LetExpr { body, .. } => vec![body],
            Opcode::MakeArr(progs) => progs.iter().map(|(p, _)| p).collect(),
            Opcode::MakeObj(entries) => {
                use crate::vm::CompiledObjEntry;
                let mut v = Vec::new();
                for e in entries.iter() {
                    match e {
                        CompiledObjEntry::Short { .. } => {}
                        CompiledObjEntry::Kv { prog, cond, .. } => {
                            v.push(prog);
                            if let Some(c) = cond {
                                v.push(c);
                            }
                        }
                        CompiledObjEntry::KvPath { .. } => {}
                        CompiledObjEntry::Dynamic { key, val } => {
                            v.push(key);
                            v.push(val);
                        }
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


/// Return `true` when `expr` contains a free reference to the variable `name`,
/// respecting lexical scope (bindings introduced inside comprehensions / lambdas / let shadow `name`).
pub fn expr_uses_ident(expr: &crate::parse::ast::Expr, name: &str) -> bool {
    use crate::parse::ast::{Arg, ArrayElem, BindTarget, Expr, FStringPart, ObjField, PipeStep, Step};
    match expr {
        Expr::Ident(n) => n == name,
        Expr::Null
        | Expr::Bool(_)
        | Expr::Int(_)
        | Expr::Float(_)
        | Expr::Str(_)
        | Expr::Root
        | Expr::Current => false,
        Expr::FString(parts) => parts.iter().any(|p| match p {
            FStringPart::Lit(_) => false,
            FStringPart::Interp { expr, .. } => expr_uses_ident(expr, name),
        }),
        Expr::Chain(base, steps) => {
            if expr_uses_ident(base, name) {
                return true;
            }
            steps.iter().any(|s| match s {
                Step::DynIndex(e) | Step::InlineFilter(e) => expr_uses_ident(e, name),
                Step::Method(_, args) | Step::OptMethod(_, args) => args.iter().any(|a| match a {
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
            ObjField::Kv { val, cond, .. } => {
                expr_uses_ident(val, name)
                    || cond.as_ref().map_or(false, |c| expr_uses_ident(c, name))
            }
            ObjField::Short(n) => n == name,
            ObjField::Dynamic { key, val } => {
                expr_uses_ident(key, name) || expr_uses_ident(val, name)
            }
            ObjField::Spread(e) => expr_uses_ident(e, name),
            ObjField::SpreadDeep(e) => expr_uses_ident(e, name),
        }),
        Expr::Array(elems) => elems.iter().any(|e| match e {
            ArrayElem::Expr(e) | ArrayElem::Spread(e) => expr_uses_ident(e, name),
        }),
        Expr::Pipeline { base, steps } => {
            if expr_uses_ident(base, name) {
                return true;
            }
            steps.iter().any(|s| match s {
                PipeStep::Forward(e) => expr_uses_ident(e, name),
                PipeStep::Bind(bt) => match bt {
                    BindTarget::Name(n) => n == name,
                    BindTarget::Obj { fields, rest } => {
                        fields.iter().any(|f| f == name)
                            || rest.as_ref().map_or(false, |r| r == name)
                    }
                    BindTarget::Arr(ns) => ns.iter().any(|n| n == name),
                },
            })
        }
        Expr::ListComp {
            expr,
            vars,
            iter,
            cond,
        }
        | Expr::SetComp {
            expr,
            vars,
            iter,
            cond,
        }
        | Expr::GenComp {
            expr,
            vars,
            iter,
            cond,
        } => {
            if expr_uses_ident(iter, name) {
                return true;
            }
            if vars.iter().any(|v| v == name) {
                return false; // `name` is shadowed by the comprehension binding.
            }
            expr_uses_ident(expr, name) || cond.as_ref().map_or(false, |c| expr_uses_ident(c, name))
        }
        Expr::DictComp {
            key,
            val,
            vars,
            iter,
            cond,
        } => {
            if expr_uses_ident(iter, name) {
                return true;
            }
            if vars.iter().any(|v| v == name) {
                return false; // `name` is shadowed by the comprehension binding.
            }
            expr_uses_ident(key, name)
                || expr_uses_ident(val, name)
                || cond.as_ref().map_or(false, |c| expr_uses_ident(c, name))
        }
        Expr::Lambda { params, body } => {
            if params.iter().any(|p| p == name) {
                return false; // Parameter shadows the outer binding.
            }
            expr_uses_ident(body, name)
        }
        Expr::Let {
            name: n,
            init,
            body,
        } => {
            if expr_uses_ident(init, name) {
                return true;
            }
            if n == name {
                return false; // The let binding itself shadows `name` in `body`.
            }
            expr_uses_ident(body, name)
        }
        Expr::IfElse { cond, then_, else_ } => {
            expr_uses_ident(cond, name)
                || expr_uses_ident(then_, name)
                || expr_uses_ident(else_, name)
        }
        Expr::Try { body, default } => {
            expr_uses_ident(body, name) || expr_uses_ident(default, name)
        }
        Expr::GlobalCall { args, .. } => args.iter().any(|a| match a {
            Arg::Pos(e) | Arg::Named(_, e) => expr_uses_ident(e, name),
        }),
        Expr::Cast { expr, .. } => expr_uses_ident(expr, name),
        Expr::Patch { root, ops } => {
            use crate::parse::ast::PathStep;
            if expr_uses_ident(root, name) {
                return true;
            }
            ops.iter().any(|op| {
                op.path.iter().any(|s| match s {
                    PathStep::WildcardFilter(e) => expr_uses_ident(e, name),
                    _ => false,
                }) || expr_uses_ident(&op.val, name)
                    || op.cond.as_ref().map_or(false, |c| expr_uses_ident(c, name))
            })
        }
        Expr::DeleteMark => false,
        Expr::Match { scrutinee, arms } => {
            expr_uses_ident(scrutinee, name)
                || arms.iter().any(|a| {
                    a.guard.as_ref().is_some_and(|g| expr_uses_ident(g, name))
                        || expr_uses_ident(&a.body, name)
                })
        }
    }
}


/// Return `true` when `expr` is side-effect-free and may be safely eliminated
/// or reordered. Conservatively returns `true` for most compound forms.
pub fn expr_is_pure(expr: &crate::parse::ast::Expr) -> bool {
    use crate::parse::ast::{Arg, Expr, Step};
    match expr {
        Expr::Null
        | Expr::Bool(_)
        | Expr::Int(_)
        | Expr::Float(_)
        | Expr::Str(_)
        | Expr::Root
        | Expr::Current
        | Expr::Ident(_) => true,
        Expr::FString(_) => true,
        Expr::Chain(base, steps) => {
            if !expr_is_pure(base) {
                return false;
            }
            steps.iter().all(|s| match s {
                Step::DynIndex(e) | Step::InlineFilter(e) => expr_is_pure(e),
                Step::Method(_, args) | Step::OptMethod(_, args) => args.iter().all(|a| match a {
                    Arg::Pos(e) | Arg::Named(_, e) => expr_is_pure(e),
                }),
                _ => true,
            })
        }
        Expr::BinOp(l, _, r) | Expr::Coalesce(l, r) => expr_is_pure(l) && expr_is_pure(r),
        Expr::UnaryNeg(e) | Expr::Not(e) | Expr::Kind { expr: e, .. } => expr_is_pure(e),
        // Conservatively treat all other forms (lambdas, patches, comprehensions) as pure
        // since they don't mutate shared state in the current runtime.
        _ => true,
    }
}


/// CSE pass over `program`: replace duplicate sub-programs (identified by
/// `program_signature`) with shared `Arc` pointers, reducing re-compilation and
/// memory pressure in programs with repeated sub-expressions.
pub fn dedup_subprograms(program: &Program) -> Arc<Program> {
    let mut cache: HashMap<u64, Arc<Program>> = HashMap::new();
    dedup_rec(program, &mut cache)
}

/// Recursive implementation of `dedup_subprograms`; returns a cached `Arc<Program>`
/// if one with the same signature already exists, otherwise rebuilds with deduped children.
fn dedup_rec(program: &Program, cache: &mut HashMap<u64, Arc<Program>>) -> Arc<Program> {
    let sig = program_signature(program);
    if let Some(a) = cache.get(&sig) {
        return Arc::clone(a);
    }
    let new_ops: Vec<Opcode> = program.ops.iter().map(|op| rewrite_op(op, cache)).collect();
    let ics = crate::vm::fresh_ics(new_ops.len());
    let out = Arc::new(Program {
        ops: new_ops.into(),
        source: program.source.clone(),
        id: program.id,
        is_structural: program.is_structural,
        ics,
    });
    cache.insert(sig, Arc::clone(&out));
    out
}

/// Rewrite a single opcode so that all embedded sub-programs are replaced with
/// their deduplicated equivalents from `cache`.
fn rewrite_op(op: &Opcode, cache: &mut HashMap<u64, Arc<Program>>) -> Opcode {
    use crate::vm::{CompSpec, CompiledFSPart, CompiledObjEntry, DictCompSpec};
    match op {
        Opcode::AndOp(p) => Opcode::AndOp(dedup_rec(p, cache)),
        Opcode::OrOp(p) => Opcode::OrOp(dedup_rec(p, cache)),
        Opcode::CoalesceOp(p) => Opcode::CoalesceOp(dedup_rec(p, cache)),
        Opcode::IfElse { then_, else_ } => Opcode::IfElse {
            then_: dedup_rec(then_, cache),
            else_: dedup_rec(else_, cache),
        },
        Opcode::InlineFilter(p) => Opcode::InlineFilter(dedup_rec(p, cache)),
        Opcode::DynIndex(p) => Opcode::DynIndex(dedup_rec(p, cache)),
        Opcode::LetExpr { name, body } => Opcode::LetExpr {
            name: name.clone(),
            body: dedup_rec(body, cache),
        },
        Opcode::CallMethod(c) => Opcode::CallMethod(rewrite_call(c, cache)),
        Opcode::CallOptMethod(c) => Opcode::CallOptMethod(rewrite_call(c, cache)),
        Opcode::MakeArr(progs) => {
            let new_progs: Vec<(Arc<Program>, bool)> = progs
                .iter()
                .map(|(p, sp)| (dedup_rec(p, cache), *sp))
                .collect();
            Opcode::MakeArr(new_progs.into())
        }
        Opcode::MakeObj(entries) => {
            let new_entries: Vec<CompiledObjEntry> = entries
                .iter()
                .map(|e| match e {
                    CompiledObjEntry::Short { name, ic } => CompiledObjEntry::Short {
                        name: name.clone(),
                        ic: ic.clone(),
                    },
                    CompiledObjEntry::Kv {
                        key,
                        prog,
                        optional,
                        cond,
                    } => CompiledObjEntry::Kv {
                        key: key.clone(),
                        prog: dedup_rec(prog, cache),
                        optional: *optional,
                        cond: cond.as_ref().map(|c| dedup_rec(c, cache)),
                    },
                    CompiledObjEntry::KvPath {
                        key,
                        steps,
                        optional,
                        ics,
                    } => CompiledObjEntry::KvPath {
                        key: key.clone(),
                        steps: steps.clone(),
                        optional: *optional,
                        ics: ics.clone(),
                    },
                    CompiledObjEntry::Dynamic { key, val } => CompiledObjEntry::Dynamic {
                        key: dedup_rec(key, cache),
                        val: dedup_rec(val, cache),
                    },
                    CompiledObjEntry::Spread(p) => CompiledObjEntry::Spread(dedup_rec(p, cache)),
                    CompiledObjEntry::SpreadDeep(p) => {
                        CompiledObjEntry::SpreadDeep(dedup_rec(p, cache))
                    }
                })
                .collect();
            Opcode::MakeObj(new_entries.into())
        }
        Opcode::FString(parts) => {
            let new_parts: Vec<CompiledFSPart> = parts
                .iter()
                .map(|p| match p {
                    CompiledFSPart::Lit(s) => CompiledFSPart::Lit(s.clone()),
                    CompiledFSPart::Interp { prog, fmt } => CompiledFSPart::Interp {
                        prog: dedup_rec(prog, cache),
                        fmt: fmt.clone(),
                    },
                })
                .collect();
            Opcode::FString(new_parts.into())
        }
        Opcode::ListComp(spec) => {
            let new_spec = CompSpec {
                expr: dedup_rec(&spec.expr, cache),
                iter: dedup_rec(&spec.iter, cache),
                cond: spec.cond.as_ref().map(|c| dedup_rec(c, cache)),
                vars: spec.vars.clone(),
            };
            Opcode::ListComp(Arc::new(new_spec))
        }
        Opcode::SetComp(spec) => {
            let new_spec = CompSpec {
                expr: dedup_rec(&spec.expr, cache),
                iter: dedup_rec(&spec.iter, cache),
                cond: spec.cond.as_ref().map(|c| dedup_rec(c, cache)),
                vars: spec.vars.clone(),
            };
            Opcode::SetComp(Arc::new(new_spec))
        }
        Opcode::DictComp(spec) => {
            let new_spec = DictCompSpec {
                key: dedup_rec(&spec.key, cache),
                val: dedup_rec(&spec.val, cache),
                iter: dedup_rec(&spec.iter, cache),
                cond: spec.cond.as_ref().map(|c| dedup_rec(c, cache)),
                vars: spec.vars.clone(),
            };
            Opcode::DictComp(Arc::new(new_spec))
        }
        _ => op.clone(),
    }
}

/// Rebuild a `CompiledCall` with all sub-programs replaced by their deduplicated equivalents.
fn rewrite_call(
    c: &Arc<crate::vm::CompiledCall>,
    cache: &mut HashMap<u64, Arc<Program>>,
) -> Arc<crate::vm::CompiledCall> {
    use crate::vm::CompiledCall;
    let new_subs: Vec<Arc<Program>> = c.sub_progs.iter().map(|p| dedup_rec(p, cache)).collect();
    Arc::new(CompiledCall {
        method: c.method,
        name: c.name.clone(),
        sub_progs: new_subs.into(),
        orig_args: c.orig_args.clone(),
        demand_max_keep: c.demand_max_keep,
    })
}


/// Return an estimated execution cost for a single opcode, used by the planner
/// to order filter predicates cheapest-first and to guide inlining decisions.
pub fn opcode_cost(op: &Opcode) -> u32 {
    match op {
        Opcode::PushNull
        | Opcode::PushBool(_)
        | Opcode::PushInt(_)
        | Opcode::PushFloat(_)
        | Opcode::PushStr(_)
        | Opcode::PushRoot
        | Opcode::PushCurrent
        | Opcode::LoadIdent(_) => 1,
        Opcode::GetField(_)
        | Opcode::OptField(_)
        | Opcode::GetIndex(_)
        | Opcode::RootChain(_)
        | Opcode::FieldChain(_) => 2,
        Opcode::GetSlice(..) | Opcode::Descendant(_) => 5,
        Opcode::DescendAll => 20,
        Opcode::Not | Opcode::Neg | Opcode::SetCurrent => 1,
        Opcode::Add
        | Opcode::Sub
        | Opcode::Mul
        | Opcode::Div
        | Opcode::Mod
        | Opcode::Eq
        | Opcode::Neq
        | Opcode::Lt
        | Opcode::Lte
        | Opcode::Gt
        | Opcode::Gte
        | Opcode::Fuzzy => 2,
        Opcode::KindCheck { .. } => 2,
        Opcode::AndOp(p) | Opcode::OrOp(p) | Opcode::CoalesceOp(p) => 2 + program_cost(p),
        Opcode::IfElse { then_, else_ } => 2 + program_cost(then_) + program_cost(else_),
        Opcode::TryExpr { body, default } => 2 + program_cost(body) + program_cost(default),
        Opcode::InlineFilter(p) | Opcode::DynIndex(p) => 10 + program_cost(p),
        Opcode::CallMethod(c) | Opcode::CallOptMethod(c) => {
            let base = match c.method {
                BuiltinMethod::Filter | BuiltinMethod::Map | BuiltinMethod::FlatMap => 10,
                BuiltinMethod::Sort => 30,
                BuiltinMethod::GroupBy | BuiltinMethod::IndexBy => 25,
                BuiltinMethod::Len | BuiltinMethod::Count => 2,
                _ => 8,
            };
            base + c.sub_progs.iter().map(|p| program_cost(p)).sum::<u32>()
        }
        Opcode::MakeObj(entries) => {
            use crate::vm::CompiledObjEntry;
            entries
                .iter()
                .map(|e| match e {
                    CompiledObjEntry::Short { .. } => 2,
                    CompiledObjEntry::Kv { prog, cond, .. } => {
                        2 + program_cost(prog) + cond.as_ref().map_or(0, |c| program_cost(c))
                    }
                    CompiledObjEntry::KvPath { steps, .. } => 2 + steps.len() as u32,
                    CompiledObjEntry::Dynamic { key, val } => {
                        3 + program_cost(key) + program_cost(val)
                    }
                    CompiledObjEntry::Spread(p) => 5 + program_cost(p),
                    CompiledObjEntry::SpreadDeep(p) => 8 + program_cost(p),
                })
                .sum()
        }
        Opcode::MakeArr(progs) => progs.iter().map(|(p, _)| 1 + program_cost(p)).sum(),
        Opcode::FString(parts) => {
            use crate::vm::CompiledFSPart;
            parts
                .iter()
                .map(|p| match p {
                    CompiledFSPart::Lit(_) => 1,
                    CompiledFSPart::Interp { prog, .. } => 3 + program_cost(prog),
                })
                .sum()
        }
        Opcode::ListComp(s) | Opcode::SetComp(s) => {
            15 + program_cost(&s.expr)
                + program_cost(&s.iter)
                + s.cond.as_ref().map_or(0, |c| program_cost(c))
        }
        Opcode::DictComp(s) => {
            15 + program_cost(&s.key)
                + program_cost(&s.val)
                + program_cost(&s.iter)
                + s.cond.as_ref().map_or(0, |c| program_cost(c))
        }
        Opcode::LetExpr { body, .. } => 2 + program_cost(body),
        Opcode::Quantifier(_) => 2,
        Opcode::CastOp(_) => 2,
        Opcode::PatchEval(_) => 50,
        Opcode::DeleteMarkErr => 1,
        Opcode::Match(_) => 1,
        Opcode::PipelineRun { base, steps } => {
            program_cost(base)
                + steps
                    .iter()
                    .map(|s| match s {
                        CompiledPipeStep::Forward(p) => program_cost(p),
                        _ => 1,
                    })
                    .sum::<u32>()
        }
    }
}


/// Sum `opcode_cost` over all opcodes in `program`; used as a proxy for execution
/// time when comparing alternative sub-expressions for predicate reordering.
pub fn program_cost(program: &Program) -> u32 {
    program.ops.iter().map(opcode_cost).sum()
}


/// Monotonicity of an array-valued program with respect to its natural order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Monotonicity {
    /// Order is unknown or has been disrupted by a non-monotone operation.
    Unknown,
    /// The array is in ascending order (e.g. after `.sort()`).
    Asc,
    /// The array is in descending order (e.g. after `.sort().reverse()`).
    Desc,
    /// The value is not an array; monotonicity is not applicable.
    NotArray,
}

impl Monotonicity {
    /// Compute the monotonicity that results from applying `op` to a value
    /// with the current monotonicity. Used to track sort order through pipelines.
    pub fn after(self, op: &Opcode) -> Monotonicity {
        match op {
            Opcode::CallMethod(c) if c.sub_progs.is_empty() => match c.method {
                BuiltinMethod::Sort => Monotonicity::Asc,
                BuiltinMethod::Reverse => match self {
                    Monotonicity::Asc => Monotonicity::Desc,
                    Monotonicity::Desc => Monotonicity::Asc,
                    x => x,
                },
                BuiltinMethod::Filter => self, // Filter preserves order.
                BuiltinMethod::Map => Monotonicity::Unknown, // Map may reorder.
                _ => Monotonicity::Unknown,
            },
            Opcode::MakeArr(_) | Opcode::ListComp(_) => Monotonicity::Unknown,
            _ => self,
        }
    }
}


/// Infer the output monotonicity of `program` by stepping through each opcode
/// sequentially from `Unknown`, updating the state with `Monotonicity::after`.
pub fn infer_monotonicity(program: &Program) -> Monotonicity {
    let mut m = Monotonicity::Unknown;
    for op in program.ops.iter() {
        m = m.after(op);
    }
    m
}


/// Return `true` when `program` reads from the input document (`PushRoot`,
/// `PushCurrent`, field/index accesses, descendants). Used to detect programs
/// that are fully constant and need not be re-evaluated per document.
pub fn escapes_doc(program: &Program) -> bool {
    for op in program.ops.iter() {
        match op {
            Opcode::PushRoot
            | Opcode::PushCurrent
            | Opcode::RootChain(_)
            | Opcode::GetField(_)
            | Opcode::GetIndex(_)
            | Opcode::GetSlice(..)
            | Opcode::Descendant(_)
            | Opcode::DescendAll
            | Opcode::OptField(_) => return true,
            Opcode::CallMethod(c) | Opcode::CallOptMethod(c)
                if c.sub_progs.iter().any(|p| escapes_doc(p)) =>
            {
                return true
            }
            _ => {}
        }
    }
    false
}


/// Estimate the selectivity of a filter predicate expression; lower scores mean
/// the predicate eliminates more candidates and should be tested first.
/// The planner uses this to reorder `And` operands cheapest/most-selective first.
pub fn selectivity_score(expr: &crate::parse::ast::Expr) -> u32 {
    use crate::parse::ast::{BinOp, Expr};
    match expr {
        Expr::Bool(true) => 1000, // Always passes — effectively no filter.
        Expr::Bool(false) => 0,   // Never passes — maximally selective.
        Expr::BinOp(_, BinOp::Eq, _) => 1,
        Expr::BinOp(_, BinOp::Neq, _) => 5,
        Expr::BinOp(_, BinOp::Lt, _)
        | Expr::BinOp(_, BinOp::Gt, _)
        | Expr::BinOp(_, BinOp::Lte, _)
        | Expr::BinOp(_, BinOp::Gte, _) => 3,
        Expr::BinOp(_, BinOp::Fuzzy, _) => 2,
        Expr::BinOp(l, BinOp::And, r) => selectivity_score(l).min(selectivity_score(r)),
        Expr::BinOp(l, BinOp::Or, r) => selectivity_score(l) + selectivity_score(r),
        Expr::Not(e) => 10u32.saturating_sub(selectivity_score(e)),
        Expr::Kind { .. } => 2,
        _ => 5,
    }
}


/// Attempt to evaluate a kind-check at compile time given a statically known
/// `VType`. Returns `Some(bool)` when the result is certain, `None` when
/// `val_ty` is `Unknown` and the check must be deferred to runtime.
pub fn fold_kind_check(val_ty: VType, target: KindType, negate: bool) -> Option<bool> {
    let matches = match (val_ty, target) {
        (VType::Null, KindType::Null) => true,
        (VType::Bool, KindType::Bool) => true,
        (VType::Int | VType::Float | VType::Num, KindType::Number) => true,
        (VType::Str, KindType::Str) => true,
        (VType::Arr, KindType::Array) => true,
        (VType::Obj, KindType::Object) => true,
        (VType::Unknown, _) => return None,
        _ => false,
    };
    Some(if negate { !matches } else { matches })
}
