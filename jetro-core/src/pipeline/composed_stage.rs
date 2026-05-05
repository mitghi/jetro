//! Builds concrete `composed::Stage` implementations from `BodyKernel` and `Stage` IR nodes.
//! The resulting chain is passed to `run_pipeline` for execution with no per-shape dispatch
//! in the hot loop.

use std::cell::{Cell, OnceCell, RefCell};
use std::rc::Rc;
use std::sync::Arc;

use crate::composed as cmp;
use crate::context::Env;
use crate::vm::Program;

use super::{BodyKernel, Stage};

/// Constructs concrete `composed::Stage` objects from `Stage` IR nodes and their `BodyKernel`.
pub(super) struct ComposedStageBuilder<'a> {
    // inherited from the pipeline's outer scope
    base_env: &'a Env,
    // lazily allocated; shared by all generic program-based stages so it is created at most once
    vm_ctx: OnceCell<Rc<RefCell<cmp::VmCtx>>>,
}

impl<'a> ComposedStageBuilder<'a> {
    /// Creates a builder that borrows `base_env` for the duration of pipeline compilation.
    pub(super) fn new(base_env: &'a Env) -> Self {
        Self {
            base_env,
            vm_ctx: OnceCell::new(),
        }
    }

    /// Builds a specialised `composed::Stage` for `(stage, kernel)`; returns `None` for barrier stages.
    pub(super) fn build(&self, stage: &Stage, kernel: &BodyKernel) -> Option<Box<dyn cmp::Stage>> {
        Some(match (stage, kernel) {
            (Stage::Filter(_, _), BodyKernel::FieldCmpLit(field, op, lit))
                if matches!(op, crate::ast::BinOp::Eq) =>
            {
                Box::new(cmp::FilterFieldEqLit {
                    field: Arc::clone(field),
                    target: lit.clone(),
                })
            }
            (Stage::Map(_, _), BodyKernel::FieldRead(field)) => Box::new(cmp::MapField {
                field: Arc::clone(field),
            }),
            (Stage::Map(_, _), BodyKernel::FieldChain(keys)) => Box::new(cmp::MapFieldChain {
                keys: Arc::clone(keys),
            }),
            (Stage::FlatMap(_, _), BodyKernel::FieldRead(field)) => Box::new(cmp::FlatMapField {
                field: Arc::clone(field),
            }),
            (Stage::FlatMap(_, _), BodyKernel::FieldChain(keys)) => {
                Box::new(cmp::FlatMapFieldChain {
                    keys: Arc::clone(keys),
                })
            }
            (
                Stage::UsizeBuiltin {
                    method: crate::builtins::BuiltinMethod::Take,
                    value,
                },
                _,
            ) => Box::new(cmp::Take {
                remaining: Cell::new(*value),
            }),
            (
                Stage::UsizeBuiltin {
                    method: crate::builtins::BuiltinMethod::Skip,
                    value,
                },
                _,
            ) => Box::new(cmp::Skip {
                remaining: Cell::new(*value),
            }),
            (Stage::Builtin(call), _) => Box::new(cmp::BuiltinStage::new(call.clone())),
            (Stage::Filter(p, _), _) => Box::new(cmp::GenericFilter {
                prog: Arc::clone(p),
                ctx: self.vm_ctx(),
            }),
            (Stage::Map(p, _), _) => Box::new(cmp::GenericMap {
                prog: Arc::clone(p),
                ctx: self.vm_ctx(),
            }),
            (Stage::FlatMap(p, _), _) => Box::new(cmp::GenericFlatMap {
                prog: Arc::clone(p),
                ctx: self.vm_ctx(),
            }),
            _ => return None,
        })
    }

    /// Builds a filter stage from `prog`, specialising on field-equality kernels where possible.
    pub(super) fn build_filter_program(
        &self,
        prog: &Arc<Program>,
        kernel: &BodyKernel,
    ) -> Box<dyn cmp::Stage> {
        match kernel {
            BodyKernel::FieldCmpLit(field, op, lit) if matches!(op, crate::ast::BinOp::Eq) => {
                Box::new(cmp::FilterFieldEqLit {
                    field: Arc::clone(field),
                    target: lit.clone(),
                })
            }
            _ => Box::new(cmp::GenericFilter {
                prog: Arc::clone(prog),
                ctx: self.vm_ctx(),
            }),
        }
    }

    /// Builds a map stage from `prog`, specialising on single-field and chain-read kernels.
    pub(super) fn build_map_program(
        &self,
        prog: &Arc<Program>,
        kernel: &BodyKernel,
    ) -> Box<dyn cmp::Stage> {
        match kernel {
            BodyKernel::FieldRead(field) => Box::new(cmp::MapField {
                field: Arc::clone(field),
            }),
            BodyKernel::FieldChain(keys) => Box::new(cmp::MapFieldChain {
                keys: Arc::clone(keys),
            }),
            _ => Box::new(cmp::GenericMap {
                prog: Arc::clone(prog),
                ctx: self.vm_ctx(),
            }),
        }
    }

    // initialises the shared VmCtx on first call
    fn vm_ctx(&self) -> Rc<RefCell<cmp::VmCtx>> {
        Rc::clone(self.vm_ctx.get_or_init(|| {
            Rc::new(RefCell::new(cmp::VmCtx {
                vm: crate::vm::VM::new(),
                env: self.base_env.clone(),
            }))
        }))
    }
}

/// Extracts a `KeySource` from `kernel`; returns `None` for generic kernels.
pub(super) fn key_from_kernel(kernel: &BodyKernel) -> Option<cmp::KeySource> {
    match kernel {
        BodyKernel::FieldRead(field) => Some(cmp::KeySource::Field(Arc::clone(field))),
        BodyKernel::FieldChain(keys) => Some(cmp::KeySource::Chain(Arc::clone(keys))),
        _ => None,
    }
}
