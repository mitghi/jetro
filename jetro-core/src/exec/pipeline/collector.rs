//! Terminal result collectors for the view-pipeline and composed paths.
//!
//! `TerminalCollector` decides at collection time whether the output should
//! materialise as a plain `Vec<Val>` or as a `Val::ObjVec` (columnar struct-
//! of-arrays) when all collected rows have uniform shape.

use std::sync::Arc;

use crate::data::value::{ObjVecData, Val};
use crate::data::view::{scalar_view_to_owned_val, ValueView};

use super::{BodyKernel, CollectLayout, ObjectKernel, ViewKernelValue};

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

    /// Evaluates `item` via `kernel` on the zero-copy view path; returns `None` on failure.
    pub(crate) fn push_view_row<'v, V>(&mut self, item: &V, kernel: &BodyKernel) -> Option<()>
    where
        V: ValueView<'v>,
    {
        match self {
            Self::Values(values) => values.push(eval_view_value(item, kernel)?),
            Self::UniformObject(collector) => collector.push_view_row(item)?,
        }
        Some(())
    }

    /// Evaluates `item` via `kernel` (owned `Val` path); calls `fallback` when the VM is needed.
    pub(crate) fn push_val_row<F>(
        &mut self,
        item: &Val,
        kernel: &BodyKernel,
        fallback: F,
    ) -> Result<(), crate::data::context::EvalError>
    where
        F: FnOnce(&Val) -> Result<Val, crate::data::context::EvalError>,
    {
        match self {
            Self::Values(values) => values.push(super::eval_kernel(kernel, item, fallback)?),
            Self::UniformObject(collector) => collector.push_val_row(item),
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
    // switches to the overflow path when a row breaks shape uniformity
    fn push_view_row<'v, V>(&mut self, item: &V) -> Option<()>
    where
        V: ValueView<'v>,
    {
        if let Some(rows) = self.rows.as_mut() {
            rows.push(eval_view_object_value(item, self.object)?);
            return Some(());
        }

        if !self.object.eval_view_row_cells(item, &mut self.cells)? {
            self.flush_cells_to_rows_with(eval_view_object_value(item, self.object)?);
        }
        Some(())
    }

    fn push_val_row(&mut self, item: &Val) {
        if let Some(rows) = self.rows.as_mut() {
            rows.push(self.object.eval_val(item));
            return;
        }

        if !self.object.eval_val_row_cells(item, &mut self.cells) {
            self.flush_cells_to_rows_with(self.object.eval_val(item));
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

fn eval_view_value<'a, V>(item: &V, kernel: &BodyKernel) -> Option<Val>
where
    V: ValueView<'a>,
{
    match super::eval_view_kernel(kernel, item)? {
        ViewKernelValue::View(view) => {
            scalar_view_to_owned_val(view.scalar()).or_else(|| Some(view.materialize()))
        }
        ViewKernelValue::Owned(value) => Some(value),
    }
}

fn eval_view_object_value<'a, V>(item: &V, object: &ObjectKernel) -> Option<Val>
where
    V: ValueView<'a>,
{
    eval_view_value(item, &BodyKernel::Object(object.clone()))
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
