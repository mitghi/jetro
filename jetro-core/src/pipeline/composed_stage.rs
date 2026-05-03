//! Builds concrete `composed::Stage` implementations from `BodyKernel`
//! and `Stage` IR nodes. The resulting composed chain is then passed to
//! `run_pipeline` for execution — no per-shape dispatch in the hot loop.

use std::cell::{Cell, OnceCell, RefCell};
use std::rc::Rc;
use std::sync::Arc;

use crate::composed as cmp;
use crate::context::Env;
use crate::vm::Program;

use super::{BodyKernel, Stage};

/// Constructs concrete `composed::Stage` objects from `Stage` IR nodes and their `BodyKernel`.
///
/// A single builder is shared across all stages in one pipeline compilation so the lazily
/// initialised `VmCtx` is allocated at most once.
pub(super) struct ComposedStageBuilder<'a> {
    /// Evaluation environment inherited from the pipeline's outer scope.
    base_env: &'a Env,
    /// Lazily allocated VM context shared by all generic (program-based) stages.
    vm_ctx: OnceCell<Rc<RefCell<cmp::VmCtx>>>,
}

impl<'a> ComposedStageBuilder<'a> {
    /// Creates a builder that borrows `base_env` for the lifetime of the pipeline compilation.
    pub(super) fn new(base_env: &'a Env) -> Self {
        Self {
            base_env,
            vm_ctx: OnceCell::new(),
        }
    }

    /// Attempts to build a specialised `composed::Stage` for `(stage, kernel)`.
    ///
    /// Returns `None` for IR nodes that have no composed equivalent (barrier stages, etc.).
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
            (Stage::Take(n, _, _), _) => Box::new(cmp::Take {
                remaining: Cell::new(*n),
            }),
            (Stage::Skip(n, _, _), _) => Box::new(cmp::Skip {
                remaining: Cell::new(*n),
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

    /// Builds a filter stage from a compiled `Program`, specialising on field-equality kernels.
    ///
    /// Falls back to `GenericFilter` when the kernel cannot be mapped to a cheaper specialisation.
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

    /// Builds a map stage from a compiled `Program`, specialising on single-field and chain reads.
    ///
    /// Falls back to `GenericMap` when the kernel cannot be mapped to a cheaper specialisation.
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

    /// Returns a reference-counted handle to the shared `VmCtx`, initialising it on first call.
    fn vm_ctx(&self) -> Rc<RefCell<cmp::VmCtx>> {
        Rc::clone(self.vm_ctx.get_or_init(|| {
            Rc::new(RefCell::new(cmp::VmCtx {
                vm: crate::vm::VM::new(),
                env: self.base_env.clone(),
            }))
        }))
    }
}

/// Extracts a `KeySource` from `kernel` for group-by, sort, and unique-by operations.
///
/// Returns `None` for generic kernels that cannot be reduced to a field or chain key.
pub(super) fn key_from_kernel(kernel: &BodyKernel) -> Option<cmp::KeySource> {
    match kernel {
        BodyKernel::FieldRead(field) => Some(cmp::KeySource::Field(Arc::clone(field))),
        BodyKernel::FieldChain(keys) => Some(cmp::KeySource::Chain(Arc::clone(keys))),
        _ => None,
    }
}
