//! Classified expression kernels for pipeline stage evaluation.
//!
//! `BodyKernel` is a pre-classified form of a stage body expression that lets
//! the pipeline executor skip the VM for common patterns (field reads,
//! literal comparisons, `FieldCmpLit` fusions). Generic/unknown expressions
//! fall back to `BodyKernel::Generic`, which re-enters the VM.

use std::sync::Arc;

use crate::builtins::BuiltinCall;
use crate::context::EvalError;
use crate::util::JsonView;
use crate::value::Val;
use crate::value_view::{scalar_view_to_owned_val, ValueView};

/// Pre-classified stage body expression; variants are ordered least-to-most expensive, `Generic` re-enters the VM.
#[derive(Debug, Clone)]
pub enum BodyKernel {
    /// Expression not classifiable into a faster form; falls back to full VM evaluation.
    Generic,
    /// Returns the current element unchanged (`@`).
    Current,
    /// Reads a single named field from the current element object.
    FieldRead(Arc<str>),
    /// Traverses a chain of field names left-to-right, returning the final value.
    FieldChain(Arc<[Arc<str>]>),
    /// Applies a view-scalar builtin to the result of the receiver kernel.
    BuiltinCall {
        /// Sub-kernel that computes the value the builtin is called on.
        receiver: Box<BodyKernel>,
        /// The resolved builtin method and its static arguments.
        call: BuiltinCall,
    },
    /// Chains two kernels: applies `first`, then feeds the result into `then`.
    Compose {
        /// The first kernel in the composition chain.
        first: Box<BodyKernel>,
        /// The kernel applied to the output of `first`.
        then: Box<BodyKernel>,
    },
    /// Compares the result of `lhs` to a literal using `op`, returning a boolean.
    CmpLit {
        /// The sub-kernel whose result is the left-hand side of the comparison.
        lhs: Box<BodyKernel>,
        /// The comparison operator.
        op: crate::ast::BinOp,
        /// The literal right-hand side value.
        lit: Val,
    },
    /// Short-circuits through a list of predicates, returning `false` on the first failure.
    And(Arc<[BodyKernel]>),
    /// Reads a single field and compares it to a literal in one fused step.
    FieldCmpLit(Arc<str>, crate::ast::BinOp, Val),
    /// Traverses a field chain and compares the result to a literal in one fused step.
    FieldChainCmpLit(Arc<[Arc<str>]>, crate::ast::BinOp, Val),
    /// Compares the current element directly to a literal.
    CurrentCmpLit(crate::ast::BinOp, Val),
    /// Always produces the given boolean constant, regardless of the current element.
    ConstBool(bool),
    /// Always produces the given `Val` constant.
    Const(Val),
    /// Evaluates an interpolated format string by evaluating each part kernel.
    FString(FStringKernel),
    /// Evaluates an object literal by evaluating each field-value kernel.
    Object(ObjectKernel),
}

/// Pre-classified kernel for a format-string expression, avoiding VM re-entry for each part.
#[derive(Debug, Clone)]
pub struct FStringKernel {
    // ordered parts (literals and interpolated sub-kernels) that make up the format string
    parts: Arc<[FStringKernelPart]>,
    // pre-computed lower-bound capacity hint for the output string buffer
    base_capacity: usize,
}

/// A single part of an `FStringKernel`: either a fixed literal or a dynamic interpolation.
#[derive(Debug, Clone)]
pub enum FStringKernelPart {
    /// A constant string segment that is copied verbatim into the output.
    Lit(Arc<str>),
    /// A sub-kernel whose result is formatted and appended to the output string.
    Interp(BodyKernel),
}

/// Pre-classified kernel for an object-literal expression; bypasses the VM's object-construction opcodes.
#[derive(Debug, Clone)]
pub struct ObjectKernel {
    // ordered key/value entries that constitute the produced object
    entries: Arc<[ObjectKernelEntry]>,
}

/// A single key/value entry in an `ObjectKernel`.
#[derive(Debug, Clone)]
pub struct ObjectKernelEntry {
    // key name in the produced object
    key: Arc<str>,
    // kernel used to compute the value for this key
    value: BodyKernel,
    // when true, a null result causes this entry to be silently omitted
    optional: bool,
    // when true, null values are omitted regardless of the optional flag
    omit_null: bool,
}

impl ObjectKernel {
    /// Returns the number of key/value entries in this object kernel.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    /// Collects and returns the ordered list of key names from this object kernel.
    pub(crate) fn keys(&self) -> Arc<[Arc<str>]> {
        self.entries
            .iter()
            .map(|entry| Arc::clone(&entry.key))
            .collect::<Vec<_>>()
            .into()
    }

    /// Evaluates each entry against `item`, appending to `cells`; returns `Some(false)` on null-optional, `None` on view failure.
    pub(crate) fn eval_view_row_cells<'a, V>(&self, item: &V, cells: &mut Vec<Val>) -> Option<bool>
    where
        V: ValueView<'a>,
    {
        let start = cells.len();
        for entry in self.entries.iter() {
            let value = match eval_view_kernel(&entry.value, item)? {
                ViewKernelValue::View(view) => {
                    scalar_view_to_owned_val(view.scalar()).unwrap_or_else(|| view.materialize())
                }
                ViewKernelValue::Owned(value) => value,
            };
            if (entry.optional || entry.omit_null) && value.is_null() {
                cells.truncate(start);
                return Some(false);
            }
            cells.push(value);
        }
        Some(true)
    }

    /// Evaluates each entry against `item`, appending to `cells`; returns `false` on null-optional skip.
    pub(crate) fn eval_val_row_cells(&self, item: &Val, cells: &mut Vec<Val>) -> bool {
        let start = cells.len();
        for entry in self.entries.iter() {
            let value = eval_native_kernel(&entry.value, item).unwrap_or(Val::Null);
            if (entry.optional || entry.omit_null) && value.is_null() {
                cells.truncate(start);
                return false;
            }
            cells.push(value);
        }
        true
    }

    /// Evaluates all entries against `item` into a `Val::ObjSmall`, returning `Val::Null` on sub-kernel failure.
    pub(crate) fn eval_val(&self, item: &Val) -> Val {
        eval_object_kernel(self, |kernel| eval_native_kernel(kernel, item)).unwrap_or(Val::Null)
    }
}

impl BodyKernel {
    /// Returns `true` when this kernel references any name in `names` as a field-like access.
    pub(crate) fn mentions_any_field_like_ident(&self, names: &[Arc<str>]) -> bool {
        fn matches_name(name: &str, names: &[Arc<str>]) -> bool {
            names.iter().any(|candidate| candidate.as_ref() == name)
        }

        match self {
            Self::FieldRead(name) | Self::FieldCmpLit(name, _, _) => {
                matches_name(name.as_ref(), names)
            }
            Self::FieldChain(keys) | Self::FieldChainCmpLit(keys, _, _) => keys
                .first()
                .is_some_and(|name| matches_name(name.as_ref(), names)),
            Self::BuiltinCall { receiver, .. } => receiver.mentions_any_field_like_ident(names),
            Self::Compose { first, then } => {
                first.mentions_any_field_like_ident(names)
                    || then.mentions_any_field_like_ident(names)
            }
            Self::CmpLit { lhs, .. } => lhs.mentions_any_field_like_ident(names),
            Self::And(predicates) => predicates
                .iter()
                .any(|predicate| predicate.mentions_any_field_like_ident(names)),
            Self::FString(fstring) => fstring.parts.iter().any(|part| match part {
                FStringKernelPart::Lit(_) => false,
                FStringKernelPart::Interp(kernel) => kernel.mentions_any_field_like_ident(names),
            }),
            Self::Object(object) => object
                .entries
                .iter()
                .any(|entry| entry.value.mentions_any_field_like_ident(names)),
            Self::Generic
            | Self::Current
            | Self::CurrentCmpLit(_, _)
            | Self::ConstBool(_)
            | Self::Const(_) => false,
        }
    }

    /// Returns `true` when the kernel can operate entirely on borrowed `ValueView` without materialising.
    pub(crate) fn is_view_native(&self) -> bool {
        match self {
            Self::Generic => false,
            Self::BuiltinCall { receiver, call } => {
                receiver.is_view_native() && call.spec().view_scalar
            }
            Self::Compose { first, then } => first.is_view_native() && then.is_view_native(),
            Self::CmpLit { lhs, .. } => lhs.is_view_native(),
            Self::And(predicates) => predicates.iter().all(Self::is_view_native),
            _ => true,
        }
    }

    /// Returns the `CollectLayout` hint indicating whether outputs form a uniform-object or generic collection.
    pub(crate) fn collect_layout(&self) -> CollectLayout<'_> {
        match self {
            Self::Object(object) if object.len() > 0 => CollectLayout::UniformObject(object),
            _ => CollectLayout::Values,
        }
    }

    /// Classifies a compiled `Program` into the most specific `BodyKernel` variant, falling back to `Generic`.
    pub fn classify(prog: &crate::vm::Program) -> Self {
        use crate::vm::Opcode;
        let ops = prog.ops.as_ref();
        if ops.len() == 1 {
            if let Some(lit) = trivial_lit(&ops[0]) {
                return match &ops[0] {
                    Opcode::PushBool(b) => Self::ConstBool(*b),
                    _ => Self::Const(lit),
                };
            }
        }
        match ops {
            [Opcode::MakeObj(entries)] => {
                let mut out = Vec::with_capacity(entries.len());
                for entry in entries.iter() {
                    let (key, value, optional, omit_null) = match entry {
                        crate::vm::CompiledObjEntry::Short { name, .. } => (
                            Arc::clone(name),
                            BodyKernel::FieldRead(Arc::clone(name)),
                            false,
                            true,
                        ),
                        crate::vm::CompiledObjEntry::Kv {
                            key,
                            prog,
                            optional,
                            cond: None,
                        } => {
                            let value = BodyKernel::classify(prog);
                            if matches!(value, BodyKernel::Generic) {
                                return Self::Generic;
                            }
                            (Arc::clone(key), value, *optional, false)
                        }
                        crate::vm::CompiledObjEntry::KvPath {
                            key,
                            steps,
                            optional,
                            ..
                        } => {
                            let Some(value) = classify_kv_path(steps) else {
                                return Self::Generic;
                            };
                            (Arc::clone(key), value, *optional, false)
                        }
                        _ => return Self::Generic,
                    };
                    out.push(ObjectKernelEntry {
                        key,
                        value,
                        optional,
                        omit_null,
                    });
                }
                return Self::Object(ObjectKernel {
                    entries: out.into(),
                });
            }
            [Opcode::FString(parts)] => {
                let mut out = Vec::with_capacity(parts.len());
                let mut base_capacity = 0usize;
                for part in parts.iter() {
                    match part {
                        crate::vm::CompiledFSPart::Lit(value) => {
                            base_capacity += value.len();
                            out.push(FStringKernelPart::Lit(Arc::clone(value)));
                        }
                        crate::vm::CompiledFSPart::Interp { prog, fmt } if fmt.is_none() => {
                            let kernel = BodyKernel::classify(prog);
                            if matches!(kernel, BodyKernel::Generic | BodyKernel::FString(_)) {
                                return Self::Generic;
                            }
                            base_capacity += 8;
                            out.push(FStringKernelPart::Interp(kernel));
                        }
                        crate::vm::CompiledFSPart::Interp { .. } => return Self::Generic,
                    }
                }
                return Self::FString(FStringKernel {
                    parts: out.into(),
                    base_capacity,
                });
            }
            [Opcode::PushCurrent, Opcode::GetField(k)]
            | [Opcode::GetField(k)]
            | [Opcode::LoadIdent(k)] => return Self::FieldRead(k.clone()),
            [Opcode::PushCurrent, Opcode::FieldChain(fc)] | [Opcode::FieldChain(fc)] => {
                return Self::FieldChain(fc.keys.clone())
            }
            [Opcode::LoadIdent(k1), rest @ ..]
                if rest.iter().all(|o| matches!(o, Opcode::GetField(_))) =>
            {
                let mut keys = vec![k1.clone()];
                for o in rest {
                    if let Opcode::GetField(k) = o {
                        keys.push(k.clone());
                    }
                }
                return Self::FieldChain(keys.into());
            }
            [Opcode::LoadIdent(k1), Opcode::FieldChain(fc)] => {
                let mut keys = vec![k1.clone()];
                for k in fc.keys.iter() {
                    keys.push(k.clone());
                }
                return Self::FieldChain(keys.into());
            }
            _ => {}
        }
        let rest: &[Opcode] = if matches!(ops.first(), Some(Opcode::PushCurrent)) {
            &ops[1..]
        } else {
            ops
        };
        if rest.len() == 3 {
            if matches!(&rest[0], Opcode::PushCurrent) {
                if let Some(lit) = trivial_lit(&rest[1]) {
                    if let Some(bo) = cmp_to_binop(&rest[2]) {
                        return Self::CurrentCmpLit(bo, lit);
                    }
                }
            }
            let single_key = match &rest[0] {
                Opcode::LoadIdent(k) | Opcode::GetField(k) => Some(k.clone()),
                _ => None,
            };
            if let Some(k) = single_key {
                if let Some(lit) = trivial_lit(&rest[1]) {
                    if let Some(bo) = cmp_to_binop(&rest[2]) {
                        return Self::FieldCmpLit(k, bo, lit);
                    }
                }
            }
            if let Opcode::FieldChain(fc) = &rest[0] {
                if let Some(lit) = trivial_lit(&rest[1]) {
                    if let Some(bo) = cmp_to_binop(&rest[2]) {
                        return Self::FieldChainCmpLit(fc.keys.clone(), bo, lit);
                    }
                }
            }
        }
        if let Some(kernel) = classify_structural_view_kernel(rest) {
            return kernel;
        }
        if let Some(kernel) = classify_and_kernel(ops) {
            return kernel;
        }
        Self::Generic
    }
}

// returns None rather than wrapping a Generic sub-kernel, which would defeat specialisation
fn classify_and_kernel(ops: &[crate::vm::Opcode]) -> Option<BodyKernel> {
    let (lhs_ops, rhs) = match ops {
        [lhs @ .., crate::vm::Opcode::AndOp(rhs)] if !lhs.is_empty() => (lhs, rhs),
        _ => return None,
    };
    let lhs_prog = crate::vm::Program::new(lhs_ops.to_vec(), "<pipeline-and-lhs>");
    let lhs = BodyKernel::classify(&lhs_prog);
    let rhs = BodyKernel::classify(rhs);
    if matches!(lhs, BodyKernel::Generic) || matches!(rhs, BodyKernel::Generic) {
        return None;
    }
    let mut predicates = Vec::new();
    flatten_and_kernel(lhs, &mut predicates);
    flatten_and_kernel(rhs, &mut predicates);
    Some(BodyKernel::And(predicates.into()))
}

fn flatten_and_kernel(kernel: BodyKernel, out: &mut Vec<BodyKernel>) {
    match kernel {
        BodyKernel::And(predicates) => out.extend(predicates.iter().cloned()),
        other => out.push(other),
    }
}

/// Describes how a Map stage's output elements should be collected by the sink.
pub(crate) enum CollectLayout<'a> {
    /// Output elements are heterogeneous `Val`s; collect into a plain array.
    Values,
    /// Every output element is a uniform object with the same key schema; collect into a columnar layout.
    UniformObject(&'a ObjectKernel),
}

// numeric index steps cannot be represented as a FieldChain, so return None
fn classify_kv_path(steps: &[crate::vm::KvStep]) -> Option<BodyKernel> {
    if steps.is_empty() {
        return None;
    }
    let mut keys = Vec::with_capacity(steps.len());
    for step in steps {
        match step {
            crate::vm::KvStep::Field(key) => keys.push(Arc::clone(key)),
            crate::vm::KvStep::Index(_) => return None,
        }
    }
    match keys.len() {
        0 => None,
        1 => Some(BodyKernel::FieldRead(keys.pop().unwrap())),
        _ => Some(BodyKernel::FieldChain(keys.into())),
    }
}

#[inline]
fn trivial_lit(op: &crate::vm::Opcode) -> Option<Val> {
    use crate::vm::Opcode;
    match op {
        Opcode::PushInt(n) => Some(Val::Int(*n)),
        Opcode::PushFloat(f) => Some(Val::Float(*f)),
        Opcode::PushStr(s) => Some(Val::Str(s.clone())),
        Opcode::PushBool(b) => Some(Val::Bool(*b)),
        Opcode::PushNull => Some(Val::Null),
        _ => None,
    }
}

// called after the simple fused patterns are exhausted; handles structural view patterns
fn classify_structural_view_kernel(ops: &[crate::vm::Opcode]) -> Option<BodyKernel> {
    use crate::vm::Opcode;

    let ops = match ops {
        [Opcode::PushCurrent] => return Some(BodyKernel::Current),
        [Opcode::PushCurrent, rest @ ..] => rest,
        other => other,
    };

    if let [lhs @ .., lit_op, cmp_op] = ops {
        if let Some(lit) = trivial_lit(lit_op) {
            if let Some(op) = cmp_to_binop(cmp_op) {
                let lhs = classify_structural_view_kernel(lhs)?;
                return Some(BodyKernel::CmpLit {
                    lhs: Box::new(lhs),
                    op,
                    lit,
                });
            }
        }
    }

    match ops {
        [Opcode::LoadIdent(k) | Opcode::GetField(k)] => Some(BodyKernel::FieldRead(k.clone())),
        [Opcode::FieldChain(fc)] => Some(BodyKernel::FieldChain(fc.keys.clone())),
        [Opcode::LoadIdent(k1), rest @ ..]
            if rest.iter().all(|op| matches!(op, Opcode::GetField(_))) =>
        {
            let mut keys = Vec::with_capacity(rest.len() + 1);
            keys.push(k1.clone());
            for op in rest {
                if let Opcode::GetField(k) = op {
                    keys.push(k.clone());
                }
            }
            Some(BodyKernel::FieldChain(keys.into()))
        }
        [receiver @ .., Opcode::CallMethod(call)] if call.method.spec().view_scalar => {
            let receiver = classify_structural_view_kernel(receiver)?;
            let builtin_call = BuiltinCall::from_static_args(
                call.method,
                call.name.as_ref(),
                call.orig_args.len(),
                |idx| {
                    Ok(call
                        .sub_progs
                        .get(idx)
                        .and_then(|prog| static_prog_val(prog)))
                },
                |idx| match call.orig_args.get(idx) {
                    Some(crate::ast::Arg::Pos(crate::ast::Expr::Ident(value))) => {
                        Some(Arc::from(value.as_str()))
                    }
                    _ => None,
                },
            )
            .ok()
            .flatten()?;
            if !builtin_call.spec().view_scalar {
                return None;
            }
            Some(BodyKernel::BuiltinCall {
                receiver: Box::new(receiver),
                call: builtin_call,
            })
        }
        _ => None,
    }
}

fn static_prog_val(prog: &crate::vm::Program) -> Option<Val> {
    match prog.ops.as_ref() {
        [op] => trivial_lit(op),
        _ => None,
    }
}

#[inline]
fn cmp_to_binop(op: &crate::vm::Opcode) -> Option<crate::ast::BinOp> {
    use crate::ast::BinOp as B;
    use crate::vm::Opcode as O;
    match op {
        O::Eq => Some(B::Eq),
        O::Neq => Some(B::Neq),
        O::Lt => Some(B::Lt),
        O::Lte => Some(B::Lte),
        O::Gt => Some(B::Gt),
        O::Gte => Some(B::Gte),
        _ => None,
    }
}

/// Evaluates `kernel` against `item`, invoking `fallback` for VM re-entry only on `Generic`.
#[inline]
pub fn eval_kernel<F>(kernel: &BodyKernel, item: &Val, fallback: F) -> Result<Val, EvalError>
where
    F: FnOnce(&Val) -> Result<Val, EvalError>,
{
    if matches!(kernel, BodyKernel::Generic) {
        return fallback(item);
    }
    eval_native_kernel(kernel, item)
}

// panics on Generic — callers must route Generic through eval_kernel's fallback instead
fn eval_native_kernel(kernel: &BodyKernel, item: &Val) -> Result<Val, EvalError> {
    match kernel {
        BodyKernel::Current => Ok(item.clone()),
        BodyKernel::FieldRead(k) => Ok(item.get_field(k.as_ref())),
        BodyKernel::FieldChain(ks) => {
            let mut v = item.clone();
            for k in ks.iter() {
                v = v.get_field(k.as_ref());
                if matches!(v, Val::Null) {
                    break;
                }
            }
            Ok(v)
        }
        BodyKernel::ConstBool(b) => Ok(Val::Bool(*b)),
        BodyKernel::Const(v) => Ok(v.clone()),
        BodyKernel::FString(fstring) => {
            eval_fstring_kernel(fstring, |kernel| eval_native_kernel(kernel, item))
        }
        BodyKernel::Object(object) => {
            eval_object_kernel(object, |kernel| eval_native_kernel(kernel, item))
        }
        BodyKernel::BuiltinCall { receiver, call } => {
            let recv = eval_native_kernel(receiver, item)?;
            call.try_apply(&recv)?
                .ok_or_else(|| EvalError(format!("{:?}: unsupported receiver", call.method)))
        }
        BodyKernel::Compose { first, then } => {
            let recv = eval_native_kernel(first, item)?;
            eval_native_kernel(then, &recv)
        }
        BodyKernel::CmpLit { lhs, op, lit } => {
            let lhs = eval_native_kernel(lhs, item)?;
            Ok(Val::Bool(eval_cmp_op(&lhs, *op, lit)))
        }
        BodyKernel::And(predicates) => {
            for predicate in predicates.iter() {
                if !crate::util::is_truthy(&eval_native_kernel(predicate, item)?) {
                    return Ok(Val::Bool(false));
                }
            }
            Ok(Val::Bool(true))
        }
        BodyKernel::FieldCmpLit(k, op, lit) => {
            let lhs = item.get_field(k.as_ref());
            Ok(Val::Bool(eval_cmp_op(&lhs, *op, lit)))
        }
        BodyKernel::FieldChainCmpLit(ks, op, lit) => {
            let mut v = item.clone();
            for k in ks.iter() {
                v = v.get_field(k.as_ref());
                if matches!(v, Val::Null) {
                    break;
                }
            }
            Ok(Val::Bool(eval_cmp_op(&v, *op, lit)))
        }
        BodyKernel::CurrentCmpLit(op, lit) => Ok(Val::Bool(eval_cmp_op(item, *op, lit))),
        BodyKernel::Generic => unreachable!("generic body kernels are handled by eval_kernel"),
    }
}

fn eval_object_kernel<F>(object: &ObjectKernel, mut eval: F) -> Result<Val, EvalError>
where
    F: FnMut(&BodyKernel) -> Result<Val, EvalError>,
{
    let mut pairs = Vec::with_capacity(object.entries.len());
    for entry in object.entries.iter() {
        let value = eval(&entry.value)?;
        if (entry.optional || entry.omit_null) && value.is_null() {
            continue;
        }
        pairs.push((Arc::clone(&entry.key), value));
    }
    Ok(Val::ObjSmall(pairs.into()))
}

fn eval_fstring_kernel<F>(fstring: &FStringKernel, mut eval: F) -> Result<Val, EvalError>
where
    F: FnMut(&BodyKernel) -> Result<Val, EvalError>,
{
    let mut out = String::with_capacity(fstring.base_capacity);
    for part in fstring.parts.iter() {
        match part {
            FStringKernelPart::Lit(value) => out.push_str(value),
            FStringKernelPart::Interp(kernel) => append_val_to_string(&mut out, &eval(kernel)?)?,
        }
    }
    Ok(Val::Str(Arc::from(out)))
}

// uses itoa/ryu for numeric fast paths; val_to_string only for compound types
fn append_val_to_string(out: &mut String, value: &Val) -> Result<(), EvalError> {
    match value {
        Val::Str(value) => out.push_str(value),
        Val::StrSlice(value) => out.push_str(value.as_str()),
        Val::Int(value) => out.push_str(itoa::Buffer::new().format(*value)),
        Val::Float(value) => out.push_str(ryu::Buffer::new().format(*value)),
        Val::Bool(true) => out.push_str("true"),
        Val::Bool(false) => out.push_str("false"),
        Val::Null => out.push_str("null"),
        other => out.push_str(&crate::util::val_to_string(other)),
    }
    Ok(())
}

// compound scalars (array/object len variants) fall through to view.materialize()
fn append_json_view_to_string<'a, V>(
    out: &mut String,
    view: &V,
    scalar: JsonView<'_>,
) -> Result<(), EvalError>
where
    V: ValueView<'a>,
{
    match scalar {
        JsonView::Null => out.push_str("null"),
        JsonView::Bool(true) => out.push_str("true"),
        JsonView::Bool(false) => out.push_str("false"),
        JsonView::Int(value) => out.push_str(itoa::Buffer::new().format(value)),
        JsonView::UInt(value) => out.push_str(itoa::Buffer::new().format(value)),
        JsonView::Float(value) => out.push_str(ryu::Buffer::new().format(value)),
        JsonView::Str(value) => out.push_str(value),
        JsonView::ArrayLen(_) | JsonView::ObjectLen(_) => {
            out.push_str(&crate::util::val_to_string(&view.materialize()));
        }
    }
    Ok(())
}

/// Result of a view-native kernel evaluation: a borrowed sub-view or a newly-owned `Val`.
pub(crate) enum ViewKernelValue<V> {
    /// The kernel produced a borrowed sub-view of the input without materialising.
    View(V),
    /// The kernel produced an owned `Val` (e.g. a literal, comparison result, or builtin output).
    Owned(Val),
}

/// Evaluates `kernel` on the borrowed `item` view, returning a sub-view or owned `Val`; `None` for `Generic`.
#[inline]
pub(crate) fn eval_view_kernel<'a, V>(kernel: &BodyKernel, item: &V) -> Option<ViewKernelValue<V>>
where
    V: ValueView<'a>,
{
    match kernel {
        BodyKernel::Current => Some(ViewKernelValue::View(item.clone())),
        BodyKernel::FieldRead(key) => Some(ViewKernelValue::View(item.field(key))),
        BodyKernel::FieldChain(keys) => Some(ViewKernelValue::View(walk_view_fields(
            item.clone(),
            keys.as_ref(),
        ))),
        BodyKernel::ConstBool(value) => Some(ViewKernelValue::Owned(Val::Bool(*value))),
        BodyKernel::Const(value) => Some(ViewKernelValue::Owned(value.clone())),
        BodyKernel::FString(fstring) => {
            let mut out = String::with_capacity(fstring.base_capacity);
            for part in fstring.parts.iter() {
                match part {
                    FStringKernelPart::Lit(value) => out.push_str(value),
                    FStringKernelPart::Interp(kernel) => match eval_view_kernel(kernel, item)? {
                        ViewKernelValue::View(view) => {
                            append_json_view_to_string(&mut out, &view, view.scalar()).ok()?;
                        }
                        ViewKernelValue::Owned(value) => {
                            append_val_to_string(&mut out, &value).ok()?;
                        }
                    },
                }
            }
            Some(ViewKernelValue::Owned(Val::Str(Arc::from(out))))
        }
        BodyKernel::Object(object) => {
            let mut pairs = Vec::with_capacity(object.entries.len());
            for entry in object.entries.iter() {
                let value = match eval_view_kernel(&entry.value, item)? {
                    ViewKernelValue::View(view) => view.materialize(),
                    ViewKernelValue::Owned(value) => value,
                };
                if (entry.optional || entry.omit_null) && value.is_null() {
                    continue;
                }
                pairs.push((Arc::clone(&entry.key), value));
            }
            Some(ViewKernelValue::Owned(Val::ObjSmall(pairs.into())))
        }
        BodyKernel::BuiltinCall { receiver, call } => match eval_view_kernel(receiver, item)? {
            ViewKernelValue::View(view) => call
                .try_apply_json_view(view.scalar())
                .map(ViewKernelValue::Owned),
            ViewKernelValue::Owned(value) => call
                .try_apply(&value)
                .ok()
                .flatten()
                .map(ViewKernelValue::Owned),
        },
        BodyKernel::Compose { first, then } => match eval_view_kernel(first, item)? {
            ViewKernelValue::View(view) => eval_view_kernel(then, &view),
            ViewKernelValue::Owned(value) => eval_native_kernel(then, &value)
                .ok()
                .map(ViewKernelValue::Owned),
        },
        BodyKernel::CmpLit { lhs, op, lit } => {
            let passes = match eval_view_kernel(lhs, item)? {
                ViewKernelValue::View(view) => crate::util::json_cmp_binop(
                    view.scalar(),
                    *op,
                    crate::util::JsonView::from_val(lit),
                ),
                ViewKernelValue::Owned(value) => crate::util::json_cmp_binop(
                    JsonView::from_val(&value),
                    *op,
                    crate::util::JsonView::from_val(lit),
                ),
            };
            Some(ViewKernelValue::Owned(Val::Bool(passes)))
        }
        BodyKernel::And(predicates) => {
            for predicate in predicates.iter() {
                let passes = match eval_view_kernel(predicate, item)? {
                    ViewKernelValue::View(view) => view.scalar().truthy(),
                    ViewKernelValue::Owned(value) => crate::util::is_truthy(&value),
                };
                if !passes {
                    return Some(ViewKernelValue::Owned(Val::Bool(false)));
                }
            }
            Some(ViewKernelValue::Owned(Val::Bool(true)))
        }
        BodyKernel::FieldCmpLit(key, op, lit) => {
            let lhs = item.field(key);
            Some(ViewKernelValue::Owned(Val::Bool(
                crate::util::json_cmp_binop(
                    lhs.scalar(),
                    *op,
                    crate::util::JsonView::from_val(lit),
                ),
            )))
        }
        BodyKernel::FieldChainCmpLit(keys, op, lit) => {
            let lhs = walk_view_fields(item.clone(), keys.as_ref());
            Some(ViewKernelValue::Owned(Val::Bool(
                crate::util::json_cmp_binop(
                    lhs.scalar(),
                    *op,
                    crate::util::JsonView::from_val(lit),
                ),
            )))
        }
        BodyKernel::CurrentCmpLit(op, lit) => Some(ViewKernelValue::Owned(Val::Bool(
            crate::util::json_cmp_binop(item.scalar(), *op, crate::util::JsonView::from_val(lit)),
        ))),
        BodyKernel::Generic => None,
    }
}

#[inline]
fn walk_view_fields<'a, V>(mut cur: V, keys: &[Arc<str>]) -> V
where
    V: ValueView<'a>,
{
    for key in keys {
        cur = cur.field(key.as_ref());
    }
    cur
}

/// Evaluates `lhs op rhs` using JSON-view comparison semantics, returning the boolean result.
#[inline]
pub fn eval_cmp_op(lhs: &Val, op: crate::ast::BinOp, rhs: &Val) -> bool {
    crate::util::json_cmp_binop(
        crate::util::JsonView::from_val(lhs),
        op,
        crate::util::JsonView::from_val(rhs),
    )
}
