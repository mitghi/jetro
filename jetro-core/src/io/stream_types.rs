use crate::data::value::Val;

pub(super) enum RowStreamRowResult {
    Emit(Val),
    EmitBytes(Vec<u8>),
    Skip,
    Stop,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(super) struct RowStreamStats {
    pub rows_scanned: usize,
    pub rows_emitted: usize,
    pub rows_filtered: usize,
    pub duplicate_rows: usize,
    pub direct_filter_rows: usize,
    pub fallback_filter_rows: usize,
    pub direct_key_rows: usize,
    pub fallback_key_rows: usize,
    pub direct_project_rows: usize,
    pub fallback_project_rows: usize,
}
