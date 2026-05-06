//! Execution backends and the dispatcher that selects between them.
//!
//! `router` is the public entry that takes a `Jetro` document plus a
//! `QueryPlan` and chooses one of the backends below:
//!
//! - `interpreted` — opcode tree-walker; the always-correct fallback.
//! - `structural` — bitmap key-presence shortcut for shape-only queries.
//! - `view` — borrowed traversal over `ValueView`s, no `Val` materialisation.
//! - `pipeline` — pull-based stage chain for streamable shapes.
//! - `composed` — fused multi-stage variant of `pipeline`.

pub(crate) mod composed;
pub(crate) mod interpreted;
pub(crate) mod pipeline;
pub(crate) mod router;
pub(crate) mod structural;
pub(crate) mod view;
