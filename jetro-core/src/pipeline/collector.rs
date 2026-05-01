use std::sync::Arc;

use crate::value::{ObjVecData, Val};
use crate::value_view::ValueView;

use super::{BodyKernel, CollectLayout, ObjectKernel, ViewKernelValue};

pub(crate) enum TerminalCollector<'a> {
    Values(Vec<Val>),
    UniformObject(UniformObjectCollector<'a>),
}

pub(crate) struct UniformObjectCollector<'a> {
    object: &'a ObjectKernel,
    keys: Arc<[Arc<str>]>,
    cells: Vec<Val>,
    rows: Option<Vec<Val>>,
}

impl<'a> TerminalCollector<'a> {
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

    pub(crate) fn push_val_row<F>(
        &mut self,
        item: &Val,
        kernel: &BodyKernel,
        fallback: F,
    ) -> Result<(), crate::context::EvalError>
    where
        F: FnOnce(&Val) -> Result<Val, crate::context::EvalError>,
    {
        match self {
            Self::Values(values) => values.push(super::eval_kernel(kernel, item, fallback)?),
            Self::UniformObject(collector) => collector.push_val_row(item),
        }
        Ok(())
    }

    pub(crate) fn finish(self) -> Val {
        match self {
            Self::Values(values) => Val::arr(values),
            Self::UniformObject(collector) => collector.finish(),
        }
    }
}

pub(crate) type TerminalMapCollector<'a> = TerminalCollector<'a>;

impl<'a> UniformObjectCollector<'a> {
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
        ViewKernelValue::View(view) => Some(view.materialize()),
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
