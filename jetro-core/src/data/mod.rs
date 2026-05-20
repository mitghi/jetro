//! Substrate types for the query engine.
//!
//! Groups the foundational data representations:
//! - [`value`] — the `Val` type and its compound variants (`Arr`, `Obj`, …).
//! - [`view`] — borrowed `ValueView` projections over tape-backed documents.
//! - [`tape`] — simd-json tape representation and `StrRef` slices.
//! - [`runtime`] — per-evaluation runtime state shared across the engine.

pub(crate) mod context;
pub(crate) mod intern;
pub(crate) mod runtime;
pub(crate) mod tape;
pub(crate) mod value;
pub(crate) mod view;
