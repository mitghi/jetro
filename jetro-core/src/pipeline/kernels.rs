use std::sync::Arc;

use crate::context::EvalError;
use crate::value::Val;

#[derive(Debug, Clone)]
pub enum BodyKernel {
    Generic,
    FieldRead(Arc<str>),
    FieldChain(Arc<[Arc<str>]>),
    FieldCmpLit(Arc<str>, crate::ast::BinOp, Val),
    FieldChainCmpLit(Arc<[Arc<str>]>, crate::ast::BinOp, Val),
    CurrentCmpLit(crate::ast::BinOp, Val),
    ConstBool(bool),
    Const(Val),
}

impl BodyKernel {
    pub(crate) fn is_view_native(&self) -> bool {
        !matches!(self, Self::Generic)
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
        Self::Generic
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
    match kernel {
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
        BodyKernel::Generic => fallback(item),
    }
}

#[inline]
pub fn eval_cmp_op(lhs: &Val, op: crate::ast::BinOp, rhs: &Val) -> bool {
    crate::util::json_cmp_binop(
        crate::util::JsonView::from_val(lhs),
        op,
        crate::util::JsonView::from_val(rhs),
    )
}
