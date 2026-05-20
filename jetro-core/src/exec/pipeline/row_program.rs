//! Generic row-local expression programs.
//!
//! `RowProgram` is the planner-facing name for expressions that can run against
//! a single row without re-entering the VM. It deliberately wraps `BodyKernel`
//! instead of duplicating expression classification: the existing kernel tree is
//! already the generic row IR for field reads, scalar calls, object shaping,
//! f-strings, arithmetic, comparisons, and match expressions.

use crate::data::view::ValueView;
#[cfg(test)]
use crate::{
    data::{context::EvalError, value::Val},
    vm::Program,
};

use super::{eval_view_kernel, BodyKernel, ViewKernelValue};
#[cfg(test)]
use super::eval_kernel;

/// A compiled row-local expression that can evaluate on owned values or borrowed views.
#[derive(Debug, Clone)]
pub(crate) struct RowProgram {
    kernel: BodyKernel,
}

impl RowProgram {
    /// Builds a row program from a pre-classified body kernel.
    pub(crate) fn from_kernel(kernel: BodyKernel) -> Option<Self> {
        (!matches!(kernel, BodyKernel::Generic)).then_some(Self { kernel })
    }

    /// Classifies a VM program as a row program when it has a native kernel representation.
    #[cfg(test)]
    pub(crate) fn classify(program: &Program) -> Option<Self> {
        Self::from_kernel(BodyKernel::classify(program))
    }

    /// Returns the underlying kernel for planner code that still consumes `BodyKernel`.
    #[inline]
    pub(crate) fn kernel(&self) -> &BodyKernel {
        &self.kernel
    }

    /// Consumes the row program and returns the underlying kernel.
    #[inline]
    pub(crate) fn into_kernel(self) -> BodyKernel {
        self.kernel
    }

    /// Evaluates this row program against an owned/materialized row.
    #[cfg(test)]
    pub(crate) fn eval_val(&self, row: &Val) -> Result<Val, EvalError> {
        eval_kernel(&self.kernel, row, |_| {
            Err(EvalError(
                "row program unexpectedly requires VM fallback".to_string(),
            ))
        })
    }

    /// Evaluates this row program against a borrowed row view.
    pub(crate) fn eval_view<'a, V>(&self, row: &V) -> Option<ViewKernelValue<V>>
    where
        V: ValueView<'a>,
    {
        eval_view_kernel(&self.kernel, row)
    }
}

impl From<RowProgram> for BodyKernel {
    fn from(program: RowProgram) -> Self {
        program.into_kernel()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::{compile::compiler::Compiler, data::view::ValView};

    use super::*;

    #[test]
    fn row_program_runs_object_projection_on_val_and_view() {
        let expr = crate::parse::parser::parse("{id, total: price * qty}").unwrap();
        let program = Compiler::compile(&expr, "<row-program-test>");
        let row_program = RowProgram::classify(&program).expect("row program");
        assert!(matches!(row_program.kernel(), BodyKernel::Object(_)));

        let row: Val = (&json!({"id": 7, "price": 12, "qty": 3})).into();
        let val_out = row_program.eval_val(&row).unwrap();
        let view_out = match row_program.eval_view(&ValView::new(&row)).unwrap() {
            ViewKernelValue::View(view) => view.materialize(),
            ViewKernelValue::Owned(value) => value,
        };

        assert_eq!(val_out.get_field("id"), Val::Int(7));
        assert_eq!(val_out.get_field("total"), Val::Int(36));
        assert_eq!(view_out.get_field("id"), Val::Int(7));
        assert_eq!(view_out.get_field("total"), Val::Int(36));
    }

    #[test]
    fn row_program_rejects_generic_vm_fallback() {
        let expr = crate::parse::parser::parse("let x = price in x").unwrap();
        let program = Compiler::compile(&expr, "<row-program-test>");
        assert!(RowProgram::classify(&program).is_none());
    }
}
