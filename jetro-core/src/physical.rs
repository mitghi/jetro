//! Physical expression IR used by the planner and physical evaluator.

use std::sync::Arc;

use crate::ast::{BinOp, KindType};
use crate::builtins::BuiltinCall;
use crate::pipeline::Pipeline;
use crate::value::Val;
use crate::vm::Program;

#[derive(Clone)]
pub enum PhysicalExpr {
    Literal(Val),
    Root,
    Current,
    Ident(Arc<str>),
    Pipeline(Pipeline),
    RootPath(Vec<PhysicalPathStep>),
    Chain {
        base: Box<PhysicalExpr>,
        steps: Vec<PhysicalChainStep>,
    },
    UnaryNeg(Box<PhysicalExpr>),
    Not(Box<PhysicalExpr>),
    Binary {
        lhs: Box<PhysicalExpr>,
        op: BinOp,
        rhs: Box<PhysicalExpr>,
    },
    Kind {
        expr: Box<PhysicalExpr>,
        ty: KindType,
        negate: bool,
    },
    Coalesce {
        lhs: Box<PhysicalExpr>,
        rhs: Box<PhysicalExpr>,
    },
    IfElse {
        cond: Box<PhysicalExpr>,
        then_: Box<PhysicalExpr>,
        else_: Box<PhysicalExpr>,
    },
    Try {
        body: Box<PhysicalExpr>,
        default: Box<PhysicalExpr>,
    },
    Object(Vec<PhysicalObjField>),
    Array(Vec<PhysicalArrayElem>),
    Let {
        name: Arc<str>,
        init: Box<PhysicalExpr>,
        body: Box<PhysicalExpr>,
    },
    Vm(Arc<Program>),
}

#[derive(Clone)]
pub enum PhysicalPathStep {
    Field(Arc<str>),
    Index(i64),
}

#[derive(Clone)]
pub enum PhysicalChainStep {
    Field(Arc<str>),
    Index(i64),
    DynIndex(PhysicalExpr),
    Method { call: BuiltinCall, optional: bool },
}

#[derive(Clone)]
pub enum PhysicalObjField {
    Kv {
        key: Arc<str>,
        val: PhysicalExpr,
        optional: bool,
        cond: Option<PhysicalExpr>,
    },
    Short(Arc<str>),
    Dynamic {
        key: PhysicalExpr,
        val: PhysicalExpr,
    },
    Spread(PhysicalExpr),
    SpreadDeep(PhysicalExpr),
}

#[derive(Clone)]
pub enum PhysicalArrayElem {
    Expr(PhysicalExpr),
    Spread(PhysicalExpr),
}
