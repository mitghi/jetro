use super::stream_plan::{RowStreamDirection, RowStreamSourceKind};
use crate::data::value::Val;

pub(super) enum RowStreamRowResult {
    Emit(Val),
    EmitBytes(Vec<u8>),
    Skip,
    Stop,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct RowStreamStats {
    pub source: RowStreamSourceKind,
    pub direction: RowStreamDirection,
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

impl Default for RowStreamStats {
    fn default() -> Self {
        Self {
            source: RowStreamSourceKind::DocumentRows,
            direction: RowStreamDirection::Forward,
            rows_scanned: 0,
            rows_emitted: 0,
            rows_filtered: 0,
            duplicate_rows: 0,
            direct_filter_rows: 0,
            fallback_filter_rows: 0,
            direct_key_rows: 0,
            fallback_key_rows: 0,
            direct_project_rows: 0,
            fallback_project_rows: 0,
        }
    }
}
