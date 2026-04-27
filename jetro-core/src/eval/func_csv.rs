//! CSV / TSV emission — thin re-export shim.
//!
//! The body lives in `builtin_helpers::csv_emit`. These wrappers keep
//! the legacy `func_csv::to_csv` / `to_tsv` callable surface during
//! the builtin migration.

use super::value::Val;

pub fn to_csv(recv: &Val) -> String {
    crate::builtin_helpers::csv_emit(recv, ",")
}
pub fn to_tsv(recv: &Val) -> String {
    crate::builtin_helpers::csv_emit(recv, "\t")
}
