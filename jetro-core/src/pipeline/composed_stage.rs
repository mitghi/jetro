use std::cell::{Cell, OnceCell, RefCell};
use std::rc::Rc;
use std::sync::Arc;

use crate::composed as cmp;
use crate::context::Env;

use super::{BodyKernel, Stage};

pub(super) struct ComposedStageBuilder<'a> {
    base_env: &'a Env,
    vm_ctx: OnceCell<Rc<RefCell<cmp::VmCtx>>>,
}

impl<'a> ComposedStageBuilder<'a> {
    pub(super) fn new(base_env: &'a Env) -> Self {
        Self {
            base_env,
            vm_ctx: OnceCell::new(),
        }
    }

    pub(super) fn build(&self, stage: &Stage, kernel: &BodyKernel) -> Option<Box<dyn cmp::Stage>> {
        Some(match (stage, kernel) {
            (Stage::Filter(_), BodyKernel::FieldCmpLit(field, op, lit))
                if matches!(op, crate::ast::BinOp::Eq) =>
            {
                Box::new(cmp::FilterFieldEqLit {
                    field: Arc::clone(field),
                    target: lit.clone(),
                })
            }
            (Stage::Map(_), BodyKernel::FieldRead(field)) => Box::new(cmp::MapField {
                field: Arc::clone(field),
            }),
            (Stage::Map(_), BodyKernel::FieldChain(keys)) => Box::new(cmp::MapFieldChain {
                keys: Arc::clone(keys),
            }),
            (Stage::FlatMap(_), BodyKernel::FieldRead(field)) => Box::new(cmp::FlatMapField {
                field: Arc::clone(field),
            }),
            (Stage::FlatMap(_), BodyKernel::FieldChain(keys)) => Box::new(cmp::FlatMapFieldChain {
                keys: Arc::clone(keys),
            }),
            (Stage::Take(n, _), _) => Box::new(cmp::Take {
                remaining: Cell::new(*n),
            }),
            (Stage::Skip(n, _), _) => Box::new(cmp::Skip {
                remaining: Cell::new(*n),
            }),
            (Stage::Builtin(call), _) => Box::new(cmp::BuiltinStage::new(call.clone())),
            // VM-fallback for any unrecognised body: generic kernels,
            // FieldCmpLit non-Eq, and custom lambdas.
            (Stage::Filter(p), _) => Box::new(cmp::GenericFilter {
                prog: Arc::clone(p),
                ctx: self.vm_ctx(),
            }),
            (Stage::Map(p), _) => Box::new(cmp::GenericMap {
                prog: Arc::clone(p),
                ctx: self.vm_ctx(),
            }),
            (Stage::FlatMap(p), _) => Box::new(cmp::GenericFlatMap {
                prog: Arc::clone(p),
                ctx: self.vm_ctx(),
            }),
            _ => return None,
        })
    }

    fn vm_ctx(&self) -> Rc<RefCell<cmp::VmCtx>> {
        Rc::clone(self.vm_ctx.get_or_init(|| {
            Rc::new(RefCell::new(cmp::VmCtx {
                vm: crate::vm::VM::new(),
                env: self.base_env.clone(),
            }))
        }))
    }
}

pub(super) fn key_from_kernel(kernel: &BodyKernel) -> Option<cmp::KeySource> {
    match kernel {
        BodyKernel::FieldRead(field) => Some(cmp::KeySource::Field(Arc::clone(field))),
        BodyKernel::FieldChain(keys) => Some(cmp::KeySource::Chain(Arc::clone(keys))),
        _ => None,
    }
}
