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
    pub parallel_partitions: usize,
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
            parallel_partitions: 0,
        }
    }
}

impl RowStreamStats {
    pub(super) fn merge_partition(&mut self, part: &Self) {
        self.rows_scanned += part.rows_scanned;
        self.rows_emitted += part.rows_emitted;
        self.rows_filtered += part.rows_filtered;
        self.duplicate_rows += part.duplicate_rows;
        self.direct_filter_rows += part.direct_filter_rows;
        self.fallback_filter_rows += part.fallback_filter_rows;
        self.direct_key_rows += part.direct_key_rows;
        self.fallback_key_rows += part.fallback_key_rows;
        self.direct_project_rows += part.direct_project_rows;
        self.fallback_project_rows += part.fallback_project_rows;
    }
}
