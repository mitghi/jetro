//! Convenience re-exports.
//!
//! `use jetro::prelude::*;` pulls in the engine-side types so call-sites
//! don't need a dozen individual `use` lines.
//!

pub use crate::query;
pub use crate::{Engine, Expr, Jetro, VM};
pub use crate::{Error, Result};
pub use serde_json::{json, Value};
