use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use parking_lot::RwLock;
use serde_json::Value;

use crate::Jetro;
use crate::parser;
use super::btree::BTree;
use super::error::DbError;
use super::storage::{Storage, Tree};

// ── ExprBucket ────────────────────────────────────────────────────────────────

/// Stores named jetro expression strings.
///
/// Expressions are stored as UTF-8 strings. On first use they are compiled by
/// the thread-local VM (compile cache means repeated use is near-zero cost).
pub struct ExprBucket {
    tree: Arc<dyn Tree>,
}

impl ExprBucket {
    /// Open a file-backed bucket directly from a path.
    ///
    /// Convenience wrapper around [`ExprBucket::from_storage`] for callers that
    /// just want a single on-disk tree without plumbing a [`Storage`] through.
    pub fn open(path: impl AsRef<Path>) -> Result<Arc<Self>, DbError> {
        let tree = BTree::open(path)?;
        Ok(Arc::new(Self { tree: tree as Arc<dyn Tree> }))
    }

    /// Open a bucket backed by any [`Storage`] — file-based, in-memory, or custom.
    ///
    /// `name` is the logical tree name within the storage; two buckets opened
    /// with the same `(storage, name)` share the same underlying tree.
    pub fn from_storage(storage: Arc<dyn Storage>, name: &str) -> Result<Arc<Self>, DbError> {
        let tree = storage.open_tree(name).map_err(DbError::Io)?;
        Ok(Arc::new(Self { tree }))
    }

    /// Store or replace a named expression.
    pub fn put(&self, key: &str, expr: &str) -> Result<(), DbError> {
        parser::parse(expr).map_err(|e| DbError::InvalidExpr(e.to_string()))?;
        self.tree.insert(key.as_bytes(), expr.as_bytes())?;
        Ok(())
    }

    /// Retrieve a stored expression string.
    pub fn get(&self, key: &str) -> Result<Option<String>, DbError> {
        match self.tree.get(key.as_bytes())? {
            Some(bytes) => Ok(Some(String::from_utf8(bytes).map_err(|_| {
                DbError::Corrupt("expression not valid UTF-8".into())
            })?)),
            None => Ok(None),
        }
    }

    pub fn delete(&self, key: &str) -> Result<bool, DbError> {
        Ok(self.tree.delete(key.as_bytes())?)
    }

    /// All stored (key, expression) pairs in key order.
    pub fn all(&self) -> Result<Vec<(String, String)>, DbError> {
        self.tree
            .all()?
            .into_iter()
            .map(|(k, v)| {
                let key = String::from_utf8(k)
                    .map_err(|_| DbError::Corrupt("key not UTF-8".into()))?;
                let expr = String::from_utf8(v)
                    .map_err(|_| DbError::Corrupt("expr not UTF-8".into()))?;
                Ok((key, expr))
            })
            .collect()
    }
}

// ── JsonBucket ────────────────────────────────────────────────────────────────

/// Stores JSON documents and derived expression results.
///
/// On [`insert`] / [`update`], each configured expression (fetched from an
/// [`ExprBucket`]) is applied to the document and the result is persisted under
/// a composite key `<doc_key>\x00<expr_key>`.
///
/// The original document is stored under `<doc_key>\x00`.
pub struct JsonBucket {
    tree: Arc<dyn Tree>,
    expr_keys: RwLock<Vec<String>>,
    exprs: Arc<ExprBucket>,
}

impl JsonBucket {
    /// File-backed open. Wrapper around [`JsonBucket::from_storage`].
    pub fn open(
        path: impl AsRef<Path>,
        expr_keys: Vec<String>,
        exprs: Arc<ExprBucket>,
    ) -> Result<Arc<Self>, DbError> {
        let tree = BTree::open(path)?;
        Ok(Arc::new(Self { tree: tree as Arc<dyn Tree>, expr_keys: RwLock::new(expr_keys), exprs }))
    }

    /// Open a bucket backed by any [`Storage`]. `name` identifies the tree
    /// within the storage.
    pub fn from_storage(
        storage: Arc<dyn Storage>,
        name: &str,
        expr_keys: Vec<String>,
        exprs: Arc<ExprBucket>,
    ) -> Result<Arc<Self>, DbError> {
        let tree = storage.open_tree(name).map_err(DbError::Io)?;
        Ok(Arc::new(Self { tree, expr_keys: RwLock::new(expr_keys), exprs }))
    }

    /// Add an expression key after open-time.  Future inserts will
    /// apply the corresponding expression; existing docs are *not*
    /// back-filled — call [`rebuild`](Self::rebuild) for that.
    ///
    /// No-op if the key is already registered.
    pub fn add_expr_key(&self, key: &str) {
        let mut w = self.expr_keys.write();
        if !w.iter().any(|k| k == key) {
            w.push(key.to_string());
        }
    }

    /// Insert a document. Applies all configured expressions and stores results.
    pub fn insert(&self, key: &str, doc: &Value) -> Result<(), DbError> {
        self.write_doc(key, doc)
    }

    /// Replace an existing document (same as insert — overwrites).
    pub fn update(&self, key: &str, doc: &Value) -> Result<(), DbError> {
        self.write_doc(key, doc)
    }

    /// Get the stored result for a specific expression key.
    pub fn get_result(
        &self,
        doc_key: &str,
        expr_key: &str,
    ) -> Result<Option<Value>, DbError> {
        let composite = composite_key(doc_key, expr_key);
        match self.tree.get(composite.as_bytes())? {
            Some(bytes) => {
                let v: Value = serde_json::from_slice(&bytes)
                    .map_err(|e| DbError::Corrupt(e.to_string()))?;
                Ok(Some(v))
            }
            None => Ok(None),
        }
    }

    /// Get the original stored document.
    pub fn get_doc(&self, doc_key: &str) -> Result<Option<Value>, DbError> {
        let composite = composite_key(doc_key, "");
        match self.tree.get(composite.as_bytes())? {
            Some(bytes) => {
                let v: Value = serde_json::from_slice(&bytes)
                    .map_err(|e| DbError::Corrupt(e.to_string()))?;
                Ok(Some(v))
            }
            None => Ok(None),
        }
    }

    /// Get all expression results for a document key.
    pub fn get_all_results(
        &self,
        doc_key: &str,
    ) -> Result<HashMap<String, Value>, DbError> {
        let mut out = HashMap::new();
        let keys = self.expr_keys.read().clone();
        for ek in &keys {
            if let Some(result) = self.get_result(doc_key, ek)? {
                out.insert(ek.clone(), result);
            }
        }
        Ok(out)
    }

    pub fn delete_doc(&self, doc_key: &str) -> Result<(), DbError> {
        // Delete original doc
        self.tree.delete(composite_key(doc_key, "").as_bytes())?;
        // Delete all expression results
        let keys = self.expr_keys.read().clone();
        for ek in &keys {
            self.tree.delete(composite_key(doc_key, ek).as_bytes())?;
        }
        Ok(())
    }

    /// Snapshot of the expression keys this bucket knows about.
    pub fn expr_keys(&self) -> Vec<String> { self.expr_keys.read().clone() }

    /// Iterate every original document as `(doc_key, Value)` pairs.
    pub fn iter_docs(&self) -> Result<Vec<(String, Value)>, DbError> {
        let mut out = Vec::new();
        for (kb, vb) in self.tree.all()? {
            // Original docs are stored under `<doc_key>\x00` (expr_key empty).
            // The composite key always contains exactly one \x00 separator;
            // docs have nothing after it.
            let Some(sep) = kb.iter().position(|&b| b == 0x00) else { continue };
            if sep + 1 != kb.len() { continue; } // result, not a doc
            let dk = std::str::from_utf8(&kb[..sep])
                .map_err(|_| DbError::Corrupt("non-UTF8 doc key".into()))?
                .to_string();
            let v: Value = serde_json::from_slice(&vb)
                .map_err(|e| DbError::Corrupt(e.to_string()))?;
            out.push((dk, v));
        }
        Ok(out)
    }

    /// Re-run `expr_key` across every stored document, overwriting
    /// any existing result.  Returns the number of docs processed.
    pub fn rebuild(&self, expr_key: &str) -> Result<usize, DbError> {
        let known = self.expr_keys.read().iter().any(|k| k == expr_key);
        if !known {
            return Err(DbError::ExprNotFound(expr_key.into()));
        }
        let expr = self
            .exprs
            .get(expr_key)?
            .ok_or_else(|| DbError::ExprNotFound(expr_key.into()))?;
        let docs = self.iter_docs()?;
        let mut n = 0;
        for (doc_key, doc) in docs {
            let result = Jetro::new(doc)
                .collect(&expr)
                .map_err(|e| DbError::EvalError(e.to_string()))?;
            let bytes = serde_json::to_vec(&result)
                .map_err(|e| DbError::Serialize(e.to_string()))?;
            self.tree.insert(composite_key(&doc_key, expr_key).as_bytes(), &bytes)?;
            n += 1;
        }
        Ok(n)
    }

    /// Re-run every configured expression across every stored document.
    pub fn rebuild_all(&self) -> Result<usize, DbError> {
        let docs = self.iter_docs()?;
        let keys = self.expr_keys.read().clone();
        let jetro_cache: Vec<(String, String)> = keys
            .iter()
            .map(|k| self.exprs.get(k).and_then(|o| {
                o.ok_or_else(|| DbError::ExprNotFound(k.clone())).map(|e| (k.clone(), e))
            }))
            .collect::<Result<_, _>>()?;
        let mut n = 0;
        for (doc_key, doc) in docs {
            let jetro = Jetro::new(doc);
            for (ek, src) in &jetro_cache {
                let result = jetro
                    .collect(src)
                    .map_err(|e| DbError::EvalError(e.to_string()))?;
                let bytes = serde_json::to_vec(&result)
                    .map_err(|e| DbError::Serialize(e.to_string()))?;
                self.tree.insert(composite_key(&doc_key, ek).as_bytes(), &bytes)?;
            }
            n += 1;
        }
        Ok(n)
    }

    // ── Internal ─────────────────────────────────────────────────────────────

    fn write_doc(&self, key: &str, doc: &Value) -> Result<(), DbError> {
        // Store original document
        let doc_bytes = serde_json::to_vec(doc)
            .map_err(|e| DbError::Serialize(e.to_string()))?;
        self.tree.insert(composite_key(key, "").as_bytes(), &doc_bytes)?;

        // Apply each expression and store result
        let jetro = Jetro::new(doc.clone());
        let keys = self.expr_keys.read().clone();
        for ek in &keys {
            let expr = self
                .exprs
                .get(ek)?
                .ok_or_else(|| DbError::ExprNotFound(ek.clone()))?;
            let result = jetro
                .collect(&expr)
                .map_err(|e| DbError::EvalError(e.to_string()))?;
            let result_bytes = serde_json::to_vec(&result)
                .map_err(|e| DbError::Serialize(e.to_string()))?;
            self.tree.insert(composite_key(key, ek).as_bytes(), &result_bytes)?;
        }
        Ok(())
    }
}

// composite_key encodes doc_key + "\x00" + expr_key.
// Empty expr_key stores the original document.
fn composite_key(doc_key: &str, expr_key: &str) -> String {
    format!("{}\x00{}", doc_key, expr_key)
}
