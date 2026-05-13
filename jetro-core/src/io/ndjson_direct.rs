use crate::data::value::Val;

pub(super) type NdjsonPhysicalPath = Vec<crate::ir::physical::PhysicalPathStep>;

#[derive(Clone)]
pub(super) enum NdjsonDirectTapePlan {
    RootPath(NdjsonPhysicalPath),
    ViewScalarCall {
        steps: NdjsonPhysicalPath,
        call: crate::builtins::BuiltinCall,
        optional: bool,
    },
    ArrayElementPath {
        source_steps: NdjsonPhysicalPath,
        element: NdjsonDirectElement,
        suffix_steps: NdjsonPhysicalPath,
    },
    MapPath {
        source_steps: NdjsonPhysicalPath,
        suffix_steps: NdjsonPhysicalPath,
    },
    FilterMapPath {
        source_steps: NdjsonPhysicalPath,
        predicate: NdjsonDirectItemPredicate,
        suffix_steps: NdjsonPhysicalPath,
    },
    CountFiltered {
        source_steps: NdjsonPhysicalPath,
        predicate: NdjsonDirectItemPredicate,
    },
    NumericReducePath {
        source_steps: NdjsonPhysicalPath,
        suffix_steps: NdjsonPhysicalPath,
        op: crate::exec::pipeline::NumOp,
    },
    FilterNumericReducePath {
        source_steps: NdjsonPhysicalPath,
        predicate: NdjsonDirectItemPredicate,
        suffix_steps: NdjsonPhysicalPath,
        op: crate::exec::pipeline::NumOp,
    },
    ViewPipeline {
        source_steps: NdjsonPhysicalPath,
        body: crate::exec::pipeline::PipelineBody,
    },
}

#[derive(Clone, Copy)]
pub(super) enum NdjsonDirectElement {
    First,
    Last,
    Nth(usize),
}

#[derive(Clone)]
pub(super) enum NdjsonDirectPredicate {
    Path(NdjsonPhysicalPath),
    Literal(Val),
    Not(Box<NdjsonDirectPredicate>),
    Binary {
        lhs: Box<NdjsonDirectPredicate>,
        op: crate::parse::ast::BinOp,
        rhs: Box<NdjsonDirectPredicate>,
    },
    ViewScalarCall {
        steps: NdjsonPhysicalPath,
        call: crate::builtins::BuiltinCall,
    },
    ArrayElementViewScalarCall {
        source_steps: NdjsonPhysicalPath,
        element: NdjsonDirectElement,
        suffix_steps: NdjsonPhysicalPath,
        call: crate::builtins::BuiltinCall,
    },
    ViewPipeline {
        source_steps: NdjsonPhysicalPath,
        body: crate::exec::pipeline::PipelineBody,
    },
}

#[derive(Clone)]
pub(super) enum NdjsonDirectItemPredicate {
    Path(NdjsonPhysicalPath),
    Literal(Val),
    Binary {
        lhs: Box<NdjsonDirectItemPredicate>,
        op: crate::parse::ast::BinOp,
        rhs: Box<NdjsonDirectItemPredicate>,
    },
    CmpLit {
        lhs: NdjsonPhysicalPath,
        op: crate::parse::ast::BinOp,
        lit: Val,
    },
    ViewScalarCall {
        suffix_steps: NdjsonPhysicalPath,
        call: crate::builtins::BuiltinCall,
    },
}
