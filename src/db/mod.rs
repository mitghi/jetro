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
mod bucket;
mod error;
mod graph_bucket;
mod link_bucket;
mod node;
mod page;
mod graph_tests;
mod link_tests;
mod tests;

pub use btree::BTree;
pub use bucket::{ExprBucket, JsonBucket};
pub use error::DbError;
pub use graph_bucket::{GraphBucket, GraphNode};
pub use link_bucket::{LinkBucket, LinkedDoc};

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::{fs, io};

/// Top-level handle for a jetro database directory.
pub struct Database {
    dir: PathBuf,
}

impl Database {
    /// Open (or create) a database at `dir`.
    pub fn open(dir: impl AsRef<Path>) -> io::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)?;
        Ok(Self { dir })
    }

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

    /// Open (or create) a named link bucket for blocking stream-join operations.
    ///
    /// `kinds` is a slice of `(kind_name, id_field)` pairs, e.g.:
    /// ```text
    /// &[("order", "order_id"), ("payment", "order_id"), ("shipment", "order_id")]
    /// ```
    ///
    /// A link is complete when one document of every kind has been inserted with
    /// the **same id value** (extracted from the specified field).  [`LinkBucket::get`]
    /// and [`LinkBucket::query`] block until the link is complete.
    pub fn link_bucket(
        &self,
        name: &str,
        kinds: &[(&str, &str)],
    ) -> Result<Arc<LinkBucket>, DbError> {
        link_bucket::LinkBucket::open(&self.dir, name, kinds)
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
