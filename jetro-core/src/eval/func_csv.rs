//! CSV / TSV emission — thin re-export shim.
//!
//! The body lives in `composed::csv_emit` (single source of truth via
//! `composed::ToCsv` / `composed::ToTsv` Stages).  These wrappers keep
//! the legacy `func_csv::to_csv` / `to_tsv` callable surface during
//! the lift transition; they delegate to the canonical impl.

use super::value::Val;

pub fn to_csv(recv: &Val) -> String {
    crate::composed::csv_emit(recv, ",")
}
pub fn to_tsv(recv: &Val) -> String {
    crate::composed::csv_emit(recv, "\t")
}
