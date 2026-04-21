//! Convenience re-exports.
//!
//! `use jetro::prelude::*;` pulls in the engine-side types so call-sites
//! don't need a dozen individual `use` lines.
//!

pub use crate::{Error, Result};
pub use crate::{Jetro, Graph, VM, Expr, Engine};
pub use crate::{Method, MethodRegistry};
pub use crate::{query, query_with};
pub use serde_json::{json, Value};
