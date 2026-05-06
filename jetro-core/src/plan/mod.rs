//! Planning algorithms: AST → logical IR → physical plan.
//!
//! `logical` lowers an `Expr` to the logical IR; `physical` chooses an
//! executable shape for it; `optimize` rewrites the resulting plans;
//! `analysis` provides shared shape, nullability, and selectivity passes.

pub(crate) mod analysis;
pub(crate) mod logical;
pub(crate) mod optimize;
pub(crate) mod physical;
