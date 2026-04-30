use crate::value::Val;

use super::{row_source, Source};

pub(super) fn rows(source: &Source, root: &Val) -> Option<Vec<Val>> {
    let recv = row_source::resolve(source, root);
    row_source::array_like_rows(&recv)
}
