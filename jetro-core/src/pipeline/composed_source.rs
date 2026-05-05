//! Source resolution for the composed execution path. Extracts a `Rows`
//! iterator from the pipeline `Source` without copying the backing data.

use crate::value::Val;

use super::{row_source, Source};

/// Resolves `source` against `root` and returns a `Rows` iterator for composed execution.
///
/// Returns `None` when the resolved value is not array-like (scalar or null source).
pub(super) fn rows(source: &Source, root: &Val) -> Option<row_source::Rows<'static>> {
    let recv = row_source::resolve(source, root);
    row_source::resolved_array_like_rows(recv)
}
