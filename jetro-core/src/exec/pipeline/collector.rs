//! Terminal result collectors for the view-pipeline and composed paths.
//!
//! `TerminalCollector` decides at collection time whether the output should
//! materialise as a plain `Vec<Val>` or as a `Val::ObjVec` (columnar struct-
//! of-arrays) when all collected rows have uniform shape.

use std::sync::Arc;

use crate::data::value::{ObjVecData, Val};
use crate::data::view::ValueView;

use super::{BodyKernel, CollectLayout, ObjectKernel, RowProgram};

/// Output collector for the terminal stage of a pipeline.
pub(crate) enum TerminalCollector<'a> {
    /// Collects heterogeneous or scalar rows into a plain `Val::Arr`.
    Values(Vec<Val>),
    /// Collects uniform-shape object rows into a `Val::ObjVec` columnar layout when possible.
    UniformObject(UniformObjectCollector<'a>),
}

/// Accumulates uniform-shape object rows into flat cell storage for `Val::ObjVec` construction.
pub(crate) struct UniformObjectCollector<'a> {
    // describes expected field names and projections
    object: &'a ObjectKernel,
    // interned key slice shared across all output rows
    keys: Arc<[Arc<str>]>,
    // rows × columns values in row-major order
    cells: Vec<Val>,
    // overflow buffer activated once a row breaks uniformity
    rows: Option<Vec<Val>>,
}

impl<'a> TerminalCollector<'a> {
    /// Creates a collector with the layout dictated by `kernel`'s `CollectLayout`.
    pub(crate) fn new(kernel: &'a BodyKernel) -> Self {
        match kernel.collect_layout() {
            CollectLayout::Values => Self::Values(Vec::new()),
            CollectLayout::UniformObject(object) => Self::UniformObject(UniformObjectCollector {
                object,
                keys: object.keys(),
                cells: Vec::new(),
                rows: None,
            }),
        }
    }

    /// Evaluates `item` via a row program using a caller-provided view evaluator.
    pub(crate) fn push_view_program_with_evaluator<'v, V, F>(
        &mut self,
        item: &V,
        program: &RowProgram,
        vm: &mut crate::vm::VM,
        mut eval: F,
    ) -> Option<()>
    where
        V: ValueView<'v> + 'v,
        F: FnMut(&BodyKernel, &V, &mut crate::vm::VM) -> Option<Val>,
    {
        match self {
            Self::Values(values) => values.push(eval(program.kernel(), item, vm)?),
            Self::UniformObject(collector) => {
                collector.push_view_row_with_evaluator(item, vm, eval)?
            }
        }
        Some(())
    }

    /// Evaluates `item` via `kernel` (owned `Val` path); calls `fallback` when the VM is needed.
    pub(crate) fn push_val_row<F>(
        &mut self,
        item: &Val,
        kernel: &BodyKernel,
        vm: &mut crate::vm::VM,
        fallback: F,
    ) -> Result<(), crate::data::context::EvalError>
    where
        F: FnOnce(&Val, &mut crate::vm::VM) -> Result<Val, crate::data::context::EvalError>,
    {
        match self {
            Self::Values(values) => {
                values.push(super::eval_kernel_with_vm(kernel, item, vm, fallback)?)
            }
            Self::UniformObject(collector) => collector.push_val_row(item, vm),
        }
        Ok(())
    }

    /// Consumes the collector and returns either `Val::Arr` or `Val::ObjVec`.
    pub(crate) fn finish(self) -> Val {
        match self {
            Self::Values(values) => Val::arr(values),
            Self::UniformObject(collector) => collector.finish(),
        }
    }
}

/// Alias used by the streaming execution path for terminal map collection.
pub(crate) type TerminalMapCollector<'a> = TerminalCollector<'a>;

impl<'a> UniformObjectCollector<'a> {
    fn push_view_row_with_evaluator<'v, V, F>(
        &mut self,
        item: &V,
        vm: &mut crate::vm::VM,
        mut eval: F,
    ) -> Option<()>
    where
        V: ValueView<'v> + 'v,
        F: FnMut(&BodyKernel, &V, &mut crate::vm::VM) -> Option<Val>,
    {
        if let Some(rows) = self.rows.as_mut() {
            rows.push(eval_view_object_value_with_evaluator(
                item,
                self.object,
                vm,
                &mut eval,
            )?);
            return Some(());
        }

        if !eval_view_object_cells_with_evaluator(
            self.object,
            item,
            &mut self.cells,
            vm,
            &mut eval,
        )? {
            self.flush_cells_to_rows_with(eval_view_object_value_with_evaluator(
                item,
                self.object,
                vm,
                &mut eval,
            )?);
        }
        Some(())
    }

    fn push_val_row(&mut self, item: &Val, vm: &mut crate::vm::VM) {
        if let Some(rows) = self.rows.as_mut() {
            rows.push(self.object.eval_val_with_vm(item, vm));
            return;
        }

        if !self
            .object
            .eval_val_row_cells_with_vm(item, &mut self.cells, vm)
        {
            self.flush_cells_to_rows_with(self.object.eval_val_with_vm(item, vm));
        }
    }

    // drains flat cells into ObjSmall rows and initialises the overflow buffer
    fn flush_cells_to_rows_with(&mut self, current: Val) {
        let mut rows = Vec::with_capacity(self.cells.len() / self.object.len().max(1) + 1);
        for row_cells in self.cells.chunks_exact(self.object.len()) {
            rows.push(row_small_object(&self.keys, row_cells));
        }
        self.cells.clear();
        rows.push(current);
        self.rows = Some(rows);
    }

    fn finish(self) -> Val {
        if let Some(rows) = self.rows {
            return Val::arr(rows);
        }
        Val::ObjVec(Arc::new(ObjVecData {
            keys: self.keys,
            cells: self.cells,
            typed_cols: None,
        }))
    }
}

fn eval_view_object_value_with_evaluator<'a, V, F>(
    item: &V,
    object: &ObjectKernel,
    vm: &mut crate::vm::VM,
    eval: &mut F,
) -> Option<Val>
where
    V: ValueView<'a> + 'a,
    F: FnMut(&BodyKernel, &V, &mut crate::vm::VM) -> Option<Val>,
{
    let mut pairs = Vec::with_capacity(object.entries().len());
    for entry in object.entries() {
        if let Some(cond) = entry.cond() {
            let keep = eval(cond, item, vm).map(|value| crate::util::is_truthy(&value))?;
            if !keep {
                continue;
            }
        }
        let value = eval(entry.value(), item, vm)?;
        if entry.omits_null() && value.is_null() {
            continue;
        }
        pairs.push((Arc::clone(entry.key()), value));
    }
    Some(Val::ObjSmall(pairs.into()))
}

fn eval_view_object_cells_with_evaluator<'a, V, F>(
    object: &ObjectKernel,
    item: &V,
    cells: &mut Vec<Val>,
    vm: &mut crate::vm::VM,
    eval: &mut F,
) -> Option<bool>
where
    V: ValueView<'a> + 'a,
    F: FnMut(&BodyKernel, &V, &mut crate::vm::VM) -> Option<Val>,
{
    let start = cells.len();
    for entry in object.entries() {
        if let Some(cond) = entry.cond() {
            let keep = eval(cond, item, vm).map(|value| crate::util::is_truthy(&value))?;
            if !keep {
                cells.truncate(start);
                return Some(false);
            }
        }
        let value = eval(entry.value(), item, vm)?;
        if entry.omits_null() && value.is_null() {
            cells.truncate(start);
            return Some(false);
        }
        cells.push(value);
    }
    Some(true)
}

fn row_small_object(keys: &[Arc<str>], cells: &[Val]) -> Val {
    Val::ObjSmall(
        keys.iter()
            .zip(cells.iter())
            .map(|(key, value)| (Arc::clone(key), value.clone()))
            .collect::<Vec<_>>()
            .into(),
    )
}
