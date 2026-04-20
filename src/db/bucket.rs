use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use serde_json::Value;

use crate::Jetro;
use super::btree::BTree;
use super::error::DbError;

// ── ExprBucket ────────────────────────────────────────────────────────────────

/// Stores named jetro expression strings.
///
/// Expressions are stored as UTF-8 strings. On first use they are compiled by
/// the thread-local VM (compile cache means repeated use is near-zero cost).
pub struct ExprBucket {
    tree: Arc<BTree>,
}

impl ExprBucket {
    pub fn open(path: impl AsRef<Path>) -> Result<Arc<Self>, DbError> {
        let tree = BTree::open(path)?;
        Ok(Arc::new(Self { tree }))
    }

    /// Store or replace a named expression.
    pub fn put(&self, key: &str, expr: &str) -> Result<(), DbError> {
        // Validate the expression before storing by parsing it.
        crate::parser::parse(expr).map_err(|e| DbError::InvalidExpr(e.to_string()))?;
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
    tree: Arc<BTree>,
    expr_keys: Vec<String>,
    exprs: Arc<ExprBucket>,
}

impl JsonBucket {
    pub fn open(
        path: impl AsRef<Path>,
        expr_keys: Vec<String>,
        exprs: Arc<ExprBucket>,
    ) -> Result<Arc<Self>, DbError> {
        let tree = BTree::open(path)?;
        Ok(Arc::new(Self { tree, expr_keys, exprs }))
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
    ) -> Result<Option<Vec<Value>>, DbError> {
        let composite = composite_key(doc_key, expr_key);
        match self.tree.get(composite.as_bytes())? {
            Some(bytes) => {
                let v: Vec<Value> = serde_json::from_slice(&bytes)
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
    ) -> Result<HashMap<String, Vec<Value>>, DbError> {
        let mut out = HashMap::new();
        for ek in &self.expr_keys {
            if let Some(results) = self.get_result(doc_key, ek)? {
                out.insert(ek.clone(), results);
            }
        }
        Ok(out)
    }

    pub fn delete_doc(&self, doc_key: &str) -> Result<(), DbError> {
        // Delete original doc
        self.tree.delete(composite_key(doc_key, "").as_bytes())?;
        // Delete all expression results
        for ek in &self.expr_keys {
            self.tree.delete(composite_key(doc_key, ek).as_bytes())?;
        }
        Ok(())
    }

    // ── Internal ─────────────────────────────────────────────────────────────

    fn write_doc(&self, key: &str, doc: &Value) -> Result<(), DbError> {
        // Store original document
        let doc_bytes = serde_json::to_vec(doc)
            .map_err(|e| DbError::Serialize(e.to_string()))?;
        self.tree.insert(composite_key(key, "").as_bytes(), &doc_bytes)?;

        // Apply each expression and store result
        let jetro = Jetro::new(doc.clone());
        for ek in &self.expr_keys {
            let expr = self
                .exprs
                .get(ek)?
                .ok_or_else(|| DbError::ExprNotFound(ek.clone()))?;
            let result = jetro
                .collect(&expr)
                .map_err(|e| DbError::EvalError(e.to_string()))?;
            let result_bytes = serde_json::to_vec(&result.0)
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
