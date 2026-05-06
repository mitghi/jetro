//! Substrate types for the query engine.
//!
//! Groups the foundational data representations:
//! - [`value`] — the `Val` type and its compound variants (`Arr`, `Obj`, …).
//! - [`view`] — borrowed `ValueView` projections over tape-backed documents.
//! - [`tape`] — simd-json tape representation and `StrRef` slices.
//! - [`runtime`] — per-evaluation runtime state shared across the engine.

pub(crate) mod context;
pub(crate) mod runtime;
pub(crate) mod tape;
pub(crate) mod value;
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) mod view;
