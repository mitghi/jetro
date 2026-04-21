//! [`Row<T>`] — a typed projection over a [`Bucket`] for a `T: JetroSchema`.
//!
//! Where a [`Bucket`] accepts arbitrary expression keys as `&str`, a
//! `Row<T>` pins its expression set to the pairs declared on `T` by
//! `#[derive(JetroSchema)]`.  The schema is registered once on open so
//! that subsequent `get` / `insert` calls can be purely typed:
//!
//! ```rust,no_run
//! use jetro::prelude::*;
//! use jetro::JetroSchema;
//!
//! #[derive(JetroSchema)]
//! #[expr(titles = "$.books.map(title)")]
//! #[expr(count  = "$.books.len()")]
//! struct BookView;
//!
//! let db   = Database::memory()?;
//! let row  = db.row::<BookView>("library")?;
//! row.insert("catalog", &json!({"books": [{"title": "Dune"}]}))?;
//! let n: u64 = row.get("catalog", "count")?.unwrap();
//! # Ok::<(), jetro::Error>(())
//! ```

use std::marker::PhantomData;

use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::JetroSchema;
use super::error::DbError;
use super::fluent::Bucket;
use super::Database;

/// Typed projection of a [`Bucket`] pinned to `T`'s schema.
///
/// Cheap to clone — `Row<T>` wraps a `Bucket`, which is itself `Arc`-backed.
#[derive(Clone)]
pub struct Row<T: JetroSchema> {
    bucket:  Bucket,
    _schema: PhantomData<fn() -> T>,
}

impl<T: JetroSchema> Row<T> {
    /// Wrap an already-open [`Bucket`] as a typed `Row<T>`.
    ///
    /// No schema registration is performed — the caller is trusted to have
    /// already bound the expressions declared by `T` via
    /// [`BucketBuilder::with`].  Prefer [`Row::open`] when you want the
    /// schema to be enforced automatically.
    pub fn from_bucket(bucket: Bucket) -> Self {
        Self { bucket, _schema: PhantomData }
    }

    /// Open a bucket on `db` named `name`, registering every expression
    /// declared on `T`.  Idempotent — re-opening the same row does not
    /// duplicate entries.
    pub fn open(db: &Database, name: impl Into<String>) -> Result<Self, DbError> {
        let mut b = db.bucket(name);
        for (k, src) in T::exprs() {
            b = b.with(*k, *src);
        }
        Ok(Self::from_bucket(b.open()?))
    }

    /// Bucket name.
    pub fn name(&self) -> &str {
        self.bucket.name()
    }

    /// Escape hatch to the untyped bucket.
    pub fn bucket(&self) -> &Bucket {
        &self.bucket
    }

    /// Insert or replace a document; all `T::EXPRS` are evaluated and stored.
    pub fn insert(&self, key: &str, doc: &Value) -> Result<(), DbError> {
        self.bucket.insert(key, doc)
    }

    /// Fetch a derived result and deserialise into `R`.
    pub fn get<R: DeserializeOwned>(
        &self,
        key: &str,
        expr: &str,
    ) -> Result<Option<R>, DbError> {
        self.bucket.get(key, expr)
    }

    /// Fetch a derived result as raw [`Value`].
    pub fn get_value(&self, key: &str, expr: &str) -> Result<Option<Value>, DbError> {
        self.bucket.get_value(key, expr)
    }

    /// The original stored document.
    pub fn get_doc(&self, key: &str) -> Result<Option<Value>, DbError> {
        self.bucket.get_doc(key)
    }

    /// Re-run every declared expression across every stored document.
    pub fn rebuild_all(&self) -> Result<usize, DbError> {
        self.bucket.rebuild_all()
    }

    /// The schema-declared expression pairs, re-exported for convenience.
    pub fn schema_pairs() -> &'static [(&'static str, &'static str)] {
        T::exprs()
    }
}

// ── Database convenience ──────────────────────────────────────────────────────

impl Database {
    /// Open a typed [`Row<T>`] on this database.  See [`Row::open`].
    pub fn row<T: JetroSchema>(&self, name: impl Into<String>) -> Result<Row<T>, DbError> {
        Row::<T>::open(self, name)
    }
}

// Tests live in `tests/macros.rs` — the `#[derive(JetroSchema)]` macro
// expands to a path rooted at `::jetro`, which the crate's own unit tests
// cannot resolve against itself.  Integration tests treat the crate as an
// external dependency, so the derive works there.
