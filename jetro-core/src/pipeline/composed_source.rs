use crate::value::Val;

use super::{row_source, Source};

pub(super) fn rows(source: &Source, root: &Val) -> Option<row_source::Rows<'static>> {
    let recv = row_source::resolve(source, root);
    row_source::resolved_array_like_rows(recv)
}
