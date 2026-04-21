//! Fluent, `Arc`-hiding wrappers over [`ExprBucket`] + [`JsonBucket`].
//!
//! The builder replaces the two-step expression-then-bucket ceremony
//! with a single chain:
//!
//! ```rust,no_run
//! use jetro::prelude::*;
//!
//! let db = Database::memory()?;
//! let books = db.bucket("library")
//!     .with("count",  "$.books.len()")
//!     .with("titles", "$.books.map(title)")
//!     .open()?;
//!
//! books.insert("catalog", &json!({"books": [{"title":"Dune","price":12.0}]}))?;
//! let n: u64 = books.get("catalog", "count")?.unwrap();
//! assert_eq!(n, 1);
//! # Ok::<(), jetro::Error>(())
//! ```
//!
//! The builder owns a per-bucket [`ExprBucket`] named `<bucket>.expr`
//! so expressions belonging to different buckets cannot collide.
//! Callers who want a shared expression namespace can still use the
//! underlying `db.expr_bucket(...)` + `db.json_bucket(...)` APIs.

use std::path::PathBuf;
use std::sync::Arc;

use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::Expr;
use super::bucket::{ExprBucket, JsonBucket};
use super::error::DbError;
use super::Database;

// ── Builder ───────────────────────────────────────────────────────────────────

/// Fluent builder returned by [`Database::bucket`].
pub struct BucketBuilder<'a> {
    db:    &'a Database,
    name:  String,
    exprs: Vec<(String, String)>,
}

impl<'a> BucketBuilder<'a> {
    pub(super) fn new(db: &'a Database, name: impl Into<String>) -> Self {
        Self { db, name: name.into(), exprs: Vec::new() }
    }

    /// Add a named expression.  The source is parse-checked when
    /// [`open`](Self::open) runs.
    pub fn with(mut self, key: impl Into<String>, src: impl Into<String>) -> Self {
        self.exprs.push((key.into(), src.into()));
        self
    }

    /// Add a pre-validated [`Expr`].  Same effect as [`with`] but the
    /// phantom type is lost at the storage boundary.
    pub fn with_expr<T>(mut self, key: impl Into<String>, expr: &Expr<T>) -> Self {
        self.exprs.push((key.into(), expr.as_str().to_string()));
        self
    }

    /// Open the underlying files and return a [`Bucket`] handle.
    ///
    /// Creates (or reuses) `<name>.expr` for this bucket's expression
    /// strings and `<name>.json` for documents and derived results.
    pub fn open(self) -> Result<Bucket, DbError> {
        let exprs = self.db.expr_bucket(&self.name)?;
        for (k, src) in &self.exprs {
            exprs.put(k, src)?;
        }
        let keys: Vec<&str> = self.exprs.iter().map(|(k, _)| k.as_str()).collect();
        let docs = self.db.json_bucket(&self.name, &keys, &exprs)?;
        Ok(Bucket {
            docs,
            exprs,
            name: self.name,
            _dir: self.db.path().to_path_buf(),
        })
    }
}

// ── Bucket ────────────────────────────────────────────────────────────────────

/// `Arc`-free handle over a [`JsonBucket`] paired with its private
/// [`ExprBucket`].
///
/// Clones are cheap — internal state is `Arc<_>`.
#[derive(Clone)]
pub struct Bucket {
    docs:  Arc<JsonBucket>,
    exprs: Arc<ExprBucket>,
    name:  String,
    _dir:  PathBuf,
}

impl Bucket {
    /// Bucket name as passed to [`Database::bucket`].
    pub fn name(&self) -> &str { &self.name }

    /// Insert or replace a document; all configured expressions run
    /// and their results are persisted.
    pub fn insert(&self, key: &str, doc: &Value) -> Result<(), DbError> {
        self.docs.insert(key, doc)
    }

    /// Alias for [`insert`] — reads better when updating.
    pub fn update(&self, key: &str, doc: &Value) -> Result<(), DbError> {
        self.docs.update(key, doc)
    }

    /// Delete a document and all its derived results.
    pub fn delete(&self, key: &str) -> Result<(), DbError> {
        self.docs.delete_doc(key)
    }

    /// Fetch the original stored document.
    pub fn get_doc(&self, key: &str) -> Result<Option<Value>, DbError> {
        self.docs.get_doc(key)
    }

    /// Fetch a derived result as raw [`Value`].
    pub fn get_value(&self, doc_key: &str, expr_key: &str) -> Result<Option<Value>, DbError> {
        self.docs.get_result(doc_key, expr_key)
    }

    /// Fetch a derived result and deserialise to `T`.
    pub fn get<T: DeserializeOwned>(
        &self,
        doc_key: &str,
        expr_key: &str,
    ) -> Result<Option<T>, DbError> {
        match self.docs.get_result(doc_key, expr_key)? {
            None => Ok(None),
            Some(v) => serde_json::from_value(v)
                .map(Some)
                .map_err(|e| DbError::Serialize(e.to_string())),
        }
    }

    /// Re-run one expression across every stored document.
    pub fn rebuild(&self, expr_key: &str) -> Result<usize, DbError> {
        self.docs.rebuild(expr_key)
    }

    /// Re-run every expression across every stored document.
    pub fn rebuild_all(&self) -> Result<usize, DbError> {
        self.docs.rebuild_all()
    }

    /// Register a new expression on this bucket after it has been opened.
    /// Existing docs are *not* back-filled — call [`rebuild`] if you need that.
    pub fn add_expr(&self, key: &str, src: &str) -> Result<(), DbError> {
        self.exprs.put(key, src)?;
        self.docs.add_expr_key(key);
        Ok(())
    }

    /// Register a pre-validated [`Expr`].  Same as [`add_expr`] but
    /// typed.
    pub fn add_expr_typed<T>(&self, key: &str, expr: &Expr<T>) -> Result<(), DbError> {
        self.exprs.put(key, expr.as_str())?;
        self.docs.add_expr_key(key);
        Ok(())
    }

    /// Snapshot of expression keys bound to this bucket.
    pub fn expr_keys(&self) -> Vec<String> { self.docs.expr_keys() }

    /// Every document as `(key, value)` pairs.
    pub fn iter_docs(&self) -> Result<Vec<(String, Value)>, DbError> {
        self.docs.iter_docs()
    }

    /// Direct access to the underlying [`JsonBucket`] — escape hatch
    /// for callers that need the full surface.
    pub fn json(&self) -> &Arc<JsonBucket> { &self.docs }

    /// Direct access to the underlying [`ExprBucket`].
    pub fn exprs(&self) -> &Arc<ExprBucket> { &self.exprs }
}

// ── Database convenience ──────────────────────────────────────────────────────

impl Database {
    /// Begin a fluent bucket definition.  See [`BucketBuilder`].
    pub fn bucket(&self, name: impl Into<String>) -> BucketBuilder<'_> {
        BucketBuilder::new(self, name)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn build_insert_get_typed() {
        let db = Database::memory().unwrap();
        let b  = db.bucket("books")
            .with("count",  "$.books.len()")
            .with("titles", "$.books.map(title)")
            .open().unwrap();
        b.insert("catalog", &json!({
            "books": [
                {"title": "Dune",       "price": 12.0},
                {"title": "Foundation", "price":  9.99}
            ]
        })).unwrap();
        let n: u64 = b.get("catalog", "count").unwrap().unwrap();
        assert_eq!(n, 2);
        let titles: Vec<String> = b.get("catalog", "titles").unwrap().unwrap();
        assert_eq!(titles, vec!["Dune", "Foundation"]);
    }

    #[test]
    fn with_expr_typed_path() {
        let db = Database::memory().unwrap();
        let len: Expr<u64> = Expr::new("$.xs.len()").unwrap();
        let b = db.bucket("nums").with_expr("n", &len).open().unwrap();
        b.insert("doc", &json!({"xs": [10, 20, 30]})).unwrap();
        let n: u64 = b.get("doc", "n").unwrap().unwrap();
        assert_eq!(n, 3);
    }

    #[test]
    fn rebuild_picks_up_new_expr() {
        let db = Database::memory().unwrap();
        let b  = db.bucket("x")
            .with("a", "$.xs.len()")
            .open().unwrap();
        b.insert("d1", &json!({"xs": [1,2,3]})).unwrap();
        b.insert("d2", &json!({"xs": [1,2]})).unwrap();
        // Add a new expression after the bucket is open.
        b.add_expr("double", "$.xs.len() * 2").unwrap();
        // The existing docs don't have a result for "double" yet.
        assert!(b.get_value("d1", "double").unwrap().is_none());
        // After rebuild they do.
        let n = b.rebuild("double").unwrap();
        assert_eq!(n, 2);
        let d1: i64 = b.get("d1", "double").unwrap().unwrap();
        assert_eq!(d1, 6);
    }

    #[test]
    fn delete_removes_everything() {
        let db = Database::memory().unwrap();
        let b = db.bucket("y").with("k", "$.x").open().unwrap();
        b.insert("d", &json!({"x": 1})).unwrap();
        assert!(b.get_doc("d").unwrap().is_some());
        b.delete("d").unwrap();
        assert!(b.get_doc("d").unwrap().is_none());
        assert!(b.get_value("d", "k").unwrap().is_none());
    }

    #[test]
    fn iter_docs_returns_originals_only() {
        let db = Database::memory().unwrap();
        let b = db.bucket("z").with("k", "$.x.len()").open().unwrap();
        b.insert("d1", &json!({"x": [1]})).unwrap();
        b.insert("d2", &json!({"x": [1,2]})).unwrap();
        let docs = b.iter_docs().unwrap();
        assert_eq!(docs.len(), 2);
        let keys: Vec<String> = docs.iter().map(|(k, _)| k.clone()).collect();
        assert!(keys.contains(&"d1".to_string()));
        assert!(keys.contains(&"d2".to_string()));
    }
}
