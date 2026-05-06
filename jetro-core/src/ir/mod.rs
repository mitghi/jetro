//! Pure data structures for the planning pipeline.
//!
//! `logical` holds the relational/declarative IR; `physical` holds the
//! executable plan IR consumed by the executor backends. Algorithms that
//! produce or rewrite these shapes live in `crate::plan`.

pub(crate) mod logical;
pub(crate) mod physical;
