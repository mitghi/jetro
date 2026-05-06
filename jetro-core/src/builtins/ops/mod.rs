//! Built-in runtime helpers grouped by category.
//!
//! Each submodule provides the runtime implementations for one functional
//! family of built-in methods (array, string, collection, path, regex, schema,
//! misc). The trait-based dispatch tables in `defs.rs` call into these helpers
//! via static dispatch.

pub mod array;
pub mod collection;
pub mod misc;
pub mod path;
pub mod regex;
pub mod schema;
pub mod string;
