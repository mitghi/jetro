//! Pipeline normalisation entry-point.
//!
//! Structural and syntactic normalisation passes (stage canonicalization, filter
//! reordering, merge operations) live here.  Symbolic constant-folding over
//! `Expr` trees lives in `symbolic.rs` and is re-exported below.

pub(super) use super::symbolic::normalize_symbolic;
