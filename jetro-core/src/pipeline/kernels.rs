use std::sync::Arc;

use crate::builtins::{BuiltinArgs, BuiltinCall};
use crate::context::EvalError;
use crate::util::JsonView;
use crate::value::Val;
use crate::value_view::ValueView;

#[derive(Debug, Clone)]
pub enum BodyKernel {
    Generic,
    Current,
    FieldRead(Arc<str>),
    FieldChain(Arc<[Arc<str>]>),
    BuiltinCall {
        receiver: Box<BodyKernel>,
        call: BuiltinCall,
    },
    CmpLit {
        lhs: Box<BodyKernel>,
        op: crate::ast::BinOp,
        lit: Val,
    },
    FieldCmpLit(Arc<str>, crate::ast::BinOp, Val),
    FieldChainCmpLit(Arc<[Arc<str>]>, crate::ast::BinOp, Val),
    CurrentCmpLit(crate::ast::BinOp, Val),
    ConstBool(bool),
    Const(Val),
    FString(FStringKernel),
    Object(ObjectKernel),
}

#[derive(Debug, Clone)]
pub struct FStringKernel {
    parts: Arc<[FStringKernelPart]>,
    base_capacity: usize,
}

#[derive(Debug, Clone)]
pub enum FStringKernelPart {
    Lit(Arc<str>),
    Interp(BodyKernel),
}

#[derive(Debug, Clone)]
pub struct ObjectKernel {
    entries: Arc<[ObjectKernelEntry]>,
}

#[derive(Debug, Clone)]
pub struct ObjectKernelEntry {
    key: Arc<str>,
    value: BodyKernel,
    optional: bool,
    omit_null: bool,
}

impl ObjectKernel {
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn keys(&self) -> Arc<[Arc<str>]> {
        self.entries
            .iter()
            .map(|entry| Arc::clone(&entry.key))
            .collect::<Vec<_>>()
            .into()
    }

    pub(crate) fn eval_view_row_cells<'a, V>(&self, item: &V, cells: &mut Vec<Val>) -> Option<bool>
    where
        V: ValueView<'a>,
    {
        let start = cells.len();
        for entry in self.entries.iter() {
            let value = match eval_view_kernel(&entry.value, item)? {
                ViewKernelValue::View(view) => view.materialize(),
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

    pub(crate) fn eval_val(&self, item: &Val) -> Val {
        eval_object_kernel(self, |kernel| eval_native_kernel(kernel, item)).unwrap_or(Val::Null)
    }
}

impl BodyKernel {
    pub(crate) fn is_view_native(&self) -> bool {
        match self {
            Self::Generic => false,
            Self::BuiltinCall { receiver, call } => {
                receiver.is_view_native() && call.spec().view_scalar
            }
            Self::CmpLit { lhs, .. } => lhs.is_view_native(),
            _ => true,
        }
    }

    pub(crate) fn collect_layout(&self) -> CollectLayout<'_> {
        match self {
            Self::Object(object) if object.len() > 0 => CollectLayout::UniformObject(object),
            _ => CollectLayout::Values,
        }
    }

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
        Self::Generic
    }
}

pub(crate) enum CollectLayout<'a> {
    Values,
    UniformObject(&'a ObjectKernel),
}

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
        [receiver @ .., Opcode::CallMethod(call)]
            if call.orig_args.is_empty()
                && call.sub_progs.is_empty()
                && call.method.spec().view_scalar =>
        {
            let receiver = classify_structural_view_kernel(receiver)?;
            Some(BodyKernel::BuiltinCall {
                receiver: Box::new(receiver),
                call: BuiltinCall::new(call.method, BuiltinArgs::None),
            })
        }
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
        BodyKernel::CmpLit { lhs, op, lit } => {
            let lhs = eval_native_kernel(lhs, item)?;
            Ok(Val::Bool(eval_cmp_op(&lhs, *op, lit)))
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

pub(crate) enum ViewKernelValue<V> {
    View(V),
    Owned(Val),
}

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

#[inline]
pub fn eval_cmp_op(lhs: &Val, op: crate::ast::BinOp, rhs: &Val) -> bool {
    crate::util::json_cmp_binop(
        crate::util::JsonView::from_val(lhs),
        op,
        crate::util::JsonView::from_val(rhs),
    )
}
