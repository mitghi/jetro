//! On-disk B+ Tree storage for jetro expressions and JSON query results.
//!
//! # Architecture
//!
//! ```text
//! Database (directory)
//!   ├── <name>.expr  ← ExprBucket  (key → expression string)
//!   └── <name>.json  ← JsonBucket  (composite key → JSON result)
//! ```
//!
//! # Example
//!
//! ```rust,no_run
//! use jetro::db::Database;
//! use serde_json::json;
//!
//! let db = Database::open("./mydb").unwrap();
//!
//! // Store named expressions
//! let exprs = db.expr_bucket("main").unwrap();
//! exprs.put("adults", ">/users/#filter('age' > 18)").unwrap();
//! exprs.put("names",  ">/users/#map(@/name)").unwrap();
//!
//! // Create a JSON bucket that applies those expressions on every insert
//! let store = db.json_bucket("docs", &["adults", "names"], &exprs).unwrap();
//!
//! store.insert("doc1", &json!({"users": [
//!     {"name": "Alice", "age": 30},
//!     {"name": "Bob",   "age": 15},
//! ]})).unwrap();
//!
//! // Retrieve the pre-computed result
//! let adults = store.get_result("doc1", "adults").unwrap();
//! ```

pub mod btree;
#[cfg(feature = "async")]
mod async_join;
mod bucket;
mod error;
mod fluent;
mod graph_bucket;
mod graph_fluent;
mod join;
mod node;
mod page;
mod row;
mod storage;
mod store;
mod graph_tests;
mod join_tests;
mod tests;

pub use btree::BTree;
#[cfg(feature = "async")]
pub use async_join::AsyncJoin;
pub use bucket::{ExprBucket, JsonBucket};
pub use error::DbError;
pub use fluent::{Bucket, BucketBuilder};
pub use graph_bucket::{GraphBucket, GraphNode};
pub use graph_fluent::{GraphQueryBuilder};
pub use join::{Join, JoinBuilder, JoinQuery, JoinedDoc, IntoJoinId};
pub use row::Row;
pub use storage::{BTreeMem, FileStorage, MemStorage, Storage, Tree};
pub use store::{Store, BTreeBulk, BTreePrefix, DocIter};

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::{fs, io};

/// Top-level handle for a jetro database directory.
///
/// A `Database` is a directory that contains one file per bucket.
/// Construct with:
///
/// - [`Database::open`]     — existing directory (creates if missing)
/// - [`Database::open_dir`] — alias, explicit about intent
/// - [`Database::memory`]   — ephemeral, unique temp directory that
///   is removed automatically when this handle is dropped.
pub struct Database {
    dir:    PathBuf,
    /// Held until drop so the temp directory is cleaned up.  `None`
    /// for on-disk databases.
    _tmp:   Option<tempfile::TempDir>,
}

impl Database {
    /// Open (or create) a database at `dir`.
    pub fn open(dir: impl AsRef<Path>) -> io::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)?;
        Ok(Self { dir, _tmp: None })
    }

    /// Alias for [`open`] — spells out that the argument is a
    /// directory so users don't pass a file path by accident.
    pub fn open_dir(dir: impl AsRef<Path>) -> io::Result<Self> {
        Self::open(dir)
    }

    /// Open a zero-config, in-process database backed by a unique
    /// temp directory that is deleted when the returned handle
    /// is dropped.  Intended for tests, examples, and short-lived
    /// tools.
    pub fn memory() -> io::Result<Self> {
        let tmp = tempfile::Builder::new()
            .prefix("jetro-mem-")
            .tempdir()?;
        Ok(Self { dir: tmp.path().to_path_buf(), _tmp: Some(tmp) })
    }

    /// Path of the backing directory (stable across calls).
    pub fn path(&self) -> &Path { &self.dir }

    /// Open (or create) a named expression bucket.
    pub fn expr_bucket(&self, name: &str) -> Result<Arc<ExprBucket>, DbError> {
        let path = self.dir.join(format!("{name}.expr"));
        ExprBucket::open(path)
    }

    /// Open (or create) a named JSON bucket bound to an expression bucket.
    ///
    /// `expr_keys` are the expression names that will be applied on every insert
    /// or update. They must exist in `exprs` before any document is inserted.
    pub fn json_bucket(
        &self,
        name: &str,
        expr_keys: &[&str],
        exprs: &Arc<ExprBucket>,
    ) -> Result<Arc<JsonBucket>, DbError> {
        let path = self.dir.join(format!("{name}.json"));
        let keys = expr_keys.iter().map(|s| s.to_string()).collect();
        JsonBucket::open(path, keys, Arc::clone(exprs))
    }

    /// Begin a fluent [`Join`] definition for stream-join workloads.
    ///
    /// ```rust,no_run
    /// use jetro::prelude::*;
    /// let db = Database::memory()?;
    /// let orders = db.join("orders")
    ///     .id("order_id")
    ///     .kinds(["order", "payment", "shipment"])
    ///     .open()?;
    /// # Ok::<(), jetro::Error>(())
    /// ```
    pub fn join(&self, name: impl Into<String>) -> join::JoinBuilder {
        join::JoinBuilder::from_dir(&self.dir, name)
            .expect("Database::join: FileStorage open failed")
    }

    /// Open (or create) a named graph bucket for cross-document queries.
    ///
    /// The bucket manages its own per-node BTree files and secondary index files
    /// inside `dir`. Call [`GraphBucket::add_node`] and [`GraphBucket::add_index`]
    /// before inserting documents.
    pub fn graph_bucket(
        &self,
        name: &str,
        exprs: &Arc<ExprBucket>,
    ) -> Result<Arc<GraphBucket>, DbError> {
        graph_bucket::GraphBucket::open(&self.dir, name, Arc::clone(exprs))
    }
}
