//! [`Session`] — a single handle tying [`Engine`] + [`Storage`] + [`Catalog`].
//!
//! This is the 2.0 top-level façade: one type in one place for services
//! that want to share caches, named expressions, and a backing store
//! across their call-sites. The existing free functions and `Jetro` /
//! `Database` APIs keep working; `Session` is purely additive.
//!
//! ```
//! use jetro::prelude::*;
//! use jetro::{Session, MemCatalog};
//! use std::sync::Arc;
//!
//! # fn main() -> jetro::Result<()> {
//! let s = Session::builder()
//!     .storage(Arc::new(MemStorage::new()))
//!     .catalog(Arc::new(MemCatalog::new()))
//!     .build();
//!
//! s.register_expr("sum_x", "$.items.sum(x)")?;
//! let total = s.run_named("sum_x", &json!({"items": [{"x": 1}, {"x": 2}, {"x": 3}]}))?;
//! assert_eq!(total, json!(6));
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

use parking_lot::RwLock;
use serde_json::Value;

use crate::db::{
    DbError, ExprBucket, GraphBucket, Join, JoinBuilder, JsonBucket, MemStorage, Storage,
};
use crate::{Engine, Error, Result};

// ── Catalog ───────────────────────────────────────────────────────────────────

/// Named jetro expressions. Abstracts over on-disk [`ExprBucket`] and
/// in-memory [`MemCatalog`].
pub trait Catalog: Send + Sync {
    fn put(&self, name: &str, src: &str) -> std::result::Result<(), DbError>;
    fn get(&self, name: &str) -> std::result::Result<Option<String>, DbError>;
    fn delete(&self, name: &str) -> std::result::Result<bool, DbError>;
    fn all(&self) -> std::result::Result<Vec<(String, String)>, DbError>;
}

impl Catalog for ExprBucket {
    fn put(&self, name: &str, src: &str) -> std::result::Result<(), DbError> {
        ExprBucket::put(self, name, src)
    }
    fn get(&self, name: &str) -> std::result::Result<Option<String>, DbError> {
        ExprBucket::get(self, name)
    }
    fn delete(&self, name: &str) -> std::result::Result<bool, DbError> {
        ExprBucket::delete(self, name)
    }
    fn all(&self) -> std::result::Result<Vec<(String, String)>, DbError> {
        ExprBucket::all(self)
    }
}

// ── MemCatalog ────────────────────────────────────────────────────────────────

/// In-memory [`Catalog`] — a `RwLock<IndexMap<String, String>>` under
/// the hood. Intended for tests and short-lived tools.
pub struct MemCatalog {
    inner: RwLock<indexmap::IndexMap<String, String>>,
}

impl MemCatalog {
    pub fn new() -> Self {
        Self { inner: RwLock::new(indexmap::IndexMap::new()) }
    }
}

impl Default for MemCatalog {
    fn default() -> Self {
        Self::new()
    }
}

impl Catalog for MemCatalog {
    fn put(&self, name: &str, src: &str) -> std::result::Result<(), DbError> {
        self.inner.write().insert(name.to_string(), src.to_string());
        Ok(())
    }
    fn get(&self, name: &str) -> std::result::Result<Option<String>, DbError> {
        Ok(self.inner.read().get(name).cloned())
    }
    fn delete(&self, name: &str) -> std::result::Result<bool, DbError> {
        Ok(self.inner.write().shift_remove(name).is_some())
    }
    fn all(&self) -> std::result::Result<Vec<(String, String)>, DbError> {
        Ok(self.inner.read().iter().map(|(k, v)| (k.clone(), v.clone())).collect())
    }
}

// ── Session ───────────────────────────────────────────────────────────────────

/// One-stop handle carrying the shared engine, storage, and catalog.
#[derive(Clone)]
pub struct Session {
    engine:  Arc<Engine>,
    storage: Arc<dyn Storage>,
    catalog: Arc<dyn Catalog>,
}

impl Session {
    pub fn builder() -> SessionBuilder {
        SessionBuilder::default()
    }

    /// Ergonomic default — in-memory storage + in-memory catalog, fresh engine.
    pub fn in_memory() -> Self {
        Self {
            engine:  Engine::new(),
            storage: Arc::new(MemStorage::new()),
            catalog: Arc::new(MemCatalog::new()),
        }
    }

    pub fn engine(&self) -> &Arc<Engine> {
        &self.engine
    }
    pub fn storage(&self) -> &Arc<dyn Storage> {
        &self.storage
    }
    pub fn catalog(&self) -> &Arc<dyn Catalog> {
        &self.catalog
    }

    /// Register a named expression.
    pub fn register_expr(&self, name: &str, src: &str) -> Result<()> {
        self.catalog.put(name, src).map_err(Error::from)
    }

    /// Run a named expression against `doc`.
    pub fn run_named(&self, name: &str, doc: &Value) -> Result<Value> {
        let src = self
            .catalog
            .get(name)
            .map_err(Error::from)?
            .ok_or_else(|| Error::Db(DbError::ExprNotFound(name.to_string())))?;
        self.engine.run(&src, doc)
    }

    /// Run a raw expression without touching the catalog.
    pub fn run(&self, expr: &str, doc: &Value) -> Result<Value> {
        self.engine.run(expr, doc)
    }

    // ── Bucket accessors ──────────────────────────────────────────────────────
    //
    // Wire bucket types through the session's shared [`Storage`], so a
    // `Session::in_memory()` yields fully in-memory buckets, a file-backed
    // session yields file-backed buckets, and custom storages (test shims,
    // remote blob stores, etc.) plug in unchanged.

    /// Open (or create) an [`ExprBucket`] on this session's storage.
    pub fn expr_bucket(&self, name: &str) -> Result<Arc<ExprBucket>> {
        ExprBucket::from_storage(self.storage.clone(), name).map_err(Error::from)
    }

    /// Open (or create) a [`JsonBucket`] bound to `exprs`.
    pub fn json_bucket(
        &self,
        name: &str,
        expr_keys: Vec<String>,
        exprs: Arc<ExprBucket>,
    ) -> Result<Arc<JsonBucket>> {
        JsonBucket::from_storage(self.storage.clone(), name, expr_keys, exprs)
            .map_err(Error::from)
    }

    /// Open (or create) a [`GraphBucket`] backed by this session's storage.
    pub fn graph_bucket(
        &self,
        name: &str,
        exprs: Arc<ExprBucket>,
    ) -> Result<Arc<GraphBucket>> {
        GraphBucket::from_storage(self.storage.clone(), name, exprs).map_err(Error::from)
    }

    /// Begin a fluent [`Join`] definition wired to this session's storage.
    pub fn join(&self, name: impl Into<String>) -> JoinBuilder {
        JoinBuilder::new(self.storage.clone(), name)
    }

    /// Open a typed row on this session.  Registers every expression
    /// declared on `T: JetroSchema` into an [`ExprBucket`] named `<name>`
    /// and opens a [`JsonBucket`] of the same name, returning a
    /// schema-pinned handle.
    ///
    /// Unlike [`Database::row`], this works against any [`Storage`]
    /// backing the session (in-memory, file-backed, or custom).
    pub fn row<T: crate::JetroSchema>(
        &self,
        name: &str,
    ) -> Result<SessionRow<T>> {
        let exprs = self.expr_bucket(name)?;
        let mut keys: Vec<String> = Vec::with_capacity(T::exprs().len());
        for (k, src) in T::exprs() {
            exprs.put(k, src).map_err(Error::from)?;
            keys.push((*k).to_string());
        }
        let docs = self.json_bucket(name, keys, Arc::clone(&exprs))?;
        Ok(SessionRow { docs, _schema: std::marker::PhantomData })
    }
}

// ── SessionRow ────────────────────────────────────────────────────────────────

/// Session-flavoured counterpart of [`crate::db::Row`]: typed view over a
/// [`JsonBucket`] backed by a [`Session`]'s shared storage.
pub struct SessionRow<T: crate::JetroSchema> {
    docs:    Arc<JsonBucket>,
    _schema: std::marker::PhantomData<fn() -> T>,
}

impl<T: crate::JetroSchema> SessionRow<T> {
    /// Insert or replace a document; registered expressions run eagerly.
    pub fn insert(&self, key: &str, doc: &Value) -> Result<()> {
        self.docs.insert(key, doc).map_err(Error::from)
    }

    /// Fetch a derived result and deserialise into `R`.
    pub fn get<R: serde::de::DeserializeOwned>(
        &self,
        key: &str,
        expr: &str,
    ) -> Result<Option<R>> {
        match self.docs.get_result(key, expr).map_err(Error::from)? {
            None => Ok(None),
            Some(v) => serde_json::from_value(v)
                .map(Some)
                .map_err(|e| Error::Db(DbError::Serialize(e.to_string()))),
        }
    }

    /// Fetch a derived result as raw [`Value`].
    pub fn get_value(&self, key: &str, expr: &str) -> Result<Option<Value>> {
        self.docs.get_result(key, expr).map_err(Error::from)
    }

    /// Original stored document.
    pub fn get_doc(&self, key: &str) -> Result<Option<Value>> {
        self.docs.get_doc(key).map_err(Error::from)
    }

    /// Schema-declared pairs.
    pub fn schema_pairs() -> &'static [(&'static str, &'static str)] {
        T::exprs()
    }
}

// ── SessionBuilder ────────────────────────────────────────────────────────────

#[derive(Default)]
pub struct SessionBuilder {
    engine:  Option<Arc<Engine>>,
    storage: Option<Arc<dyn Storage>>,
    catalog: Option<Arc<dyn Catalog>>,
}

impl SessionBuilder {
    pub fn engine(mut self, engine: Arc<Engine>) -> Self {
        self.engine = Some(engine);
        self
    }
    pub fn storage(mut self, storage: Arc<dyn Storage>) -> Self {
        self.storage = Some(storage);
        self
    }
    pub fn catalog(mut self, catalog: Arc<dyn Catalog>) -> Self {
        self.catalog = Some(catalog);
        self
    }
    pub fn build(self) -> Session {
        Session {
            engine:  self.engine.unwrap_or_else(Engine::new),
            storage: self.storage.unwrap_or_else(|| Arc::new(MemStorage::new())),
            catalog: self.catalog.unwrap_or_else(|| Arc::new(MemCatalog::new())),
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn session_in_memory_roundtrip() {
        let s = Session::in_memory();
        s.register_expr("sum_x", "$.items.sum(x)").unwrap();
        let total = s
            .run_named(
                "sum_x",
                &json!({"items": [{"x": 1}, {"x": 2}, {"x": 3}]}),
            )
            .unwrap();
        assert_eq!(total, json!(6));
    }

    #[test]
    fn session_run_raw() {
        let s = Session::in_memory();
        let r = s.run("$.a + $.b", &json!({"a": 5, "b": 7})).unwrap();
        assert_eq!(r, json!(12));
    }

    #[test]
    fn mem_catalog_operations() {
        let c = MemCatalog::new();
        c.put("k", "$.x").unwrap();
        assert_eq!(c.get("k").unwrap().as_deref(), Some("$.x"));
        assert_eq!(c.all().unwrap().len(), 1);
        assert!(c.delete("k").unwrap());
        assert_eq!(c.get("k").unwrap(), None);
    }

    #[test]
    fn session_unknown_expr_errors() {
        let s = Session::in_memory();
        let r = s.run_named("missing", &json!({}));
        assert!(matches!(r, Err(Error::Db(DbError::ExprNotFound(_)))));
    }

    #[test]
    fn session_expr_bucket_shares_storage() {
        let s = Session::in_memory();
        let a = s.expr_bucket("named").unwrap();
        a.put("greet", "$.who").unwrap();

        // A second handle from the same session + name hits the same tree.
        let b = s.expr_bucket("named").unwrap();
        assert_eq!(b.get("greet").unwrap().as_deref(), Some("$.who"));
    }

    #[test]
    fn session_json_bucket_applies_expressions() {
        let s = Session::in_memory();
        let exprs = s.expr_bucket("exprs").unwrap();
        exprs.put("total", "$.items.sum(x)").unwrap();

        let docs = s
            .json_bucket("docs", vec!["total".to_string()], Arc::clone(&exprs))
            .unwrap();
        docs.insert("d1", &json!({"items": [{"x": 1}, {"x": 2}, {"x": 3}]})).unwrap();

        assert_eq!(docs.get_result("d1", "total").unwrap(), Some(json!(6)));
    }

    #[test]
    fn session_graph_bucket_wired() {
        let s = Session::in_memory();
        let exprs = s.expr_bucket("g_exprs").unwrap();
        let g = s.graph_bucket("g", Arc::clone(&exprs)).unwrap();
        g.add_node("user").unwrap();
        g.insert("user", "u1", &json!({"name": "alice"})).unwrap();
        assert_eq!(g.get("user", "u1").unwrap(), Some(json!({"name": "alice"})));
    }

    #[test]
    fn session_join_wired() {
        let s = Session::in_memory();
        let j = s
            .join("orders")
            .id("order_id")
            .kinds(["order", "payment"])
            .open()
            .unwrap();
        j.emit("order", &json!({"order_id": "o1", "total": 10})).unwrap();
        assert!(j.peek("o1").unwrap().is_none());
        j.emit("payment", &json!({"order_id": "o1", "paid": true})).unwrap();
        assert!(j.peek("o1").unwrap().is_some());
    }
}
