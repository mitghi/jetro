//! Graph-aware storage for cross-document jetro queries.
//!
//! # Architecture
//!
//! ```text
//! GraphBucket
//!   ├── NodeStore("orders")
//!   │     ├── docs BTree      doc_key → JSON bytes
//!   │     └── index BTree     {encoded_field}\x00{doc_key} → ""
//!   │         (per indexed field)
//!   └── NodeStore("customers")
//!         ├── docs BTree
//!         ├── index BTree
//!         └── hot cache       RwLock<HashMap<doc_key, Value>>  ← zero disk I/O
//! ```
//!
//! # Stream processing pattern
//!
//! ```text
//! 1. Incoming event (order JSON) → GraphBucket::insert("orders", key, doc)
//!    - Stores doc in orders.docs BTree
//!    - Updates orders.customer_id index BTree
//!
//! 2. query(&[
//!      GraphNode::Inline("orders",   order_json),          // event provided directly
//!      GraphNode::ByField("customers", "id", &cust_id),    // O(log n) index lookup
//!    ], ">/orders/#join('customers', 'customer_id', 'id')")
//!    - Builds jetro::Graph({orders: [...], customers: [...]})
//!    - Evaluates expression via thread-local VM
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::RwLock;
use serde_json::Value;

use crate::Graph;
use super::btree::BTree;
use super::bucket::ExprBucket;
use super::error::DbError;

// ── Index key encoding ────────────────────────────────────────────────────────
//
// Key = {1-byte type tag}{encoded value}\x00{doc_key}
//
// Type tags keep different JSON types sorted in separate namespaces.
// The \x00 separator lets us range-scan a prefix to get all doc_keys for a value.

const TAG_NULL:   u8 = b'z';
const TAG_BOOL:   u8 = b'b';
const TAG_NUMBER: u8 = b'n';
const TAG_STRING: u8 = b's';
const TAG_OTHER:  u8 = b'j'; // arrays/objects: fall back to JSON string

fn encode_field_prefix(val: &Value) -> Vec<u8> {
    let mut v = match val {
        Value::Null       => vec![TAG_NULL],
        Value::Bool(b)    => vec![TAG_BOOL, *b as u8],
        Value::Number(n)  => {
            let mut v = vec![TAG_NUMBER];
            v.extend_from_slice(n.to_string().as_bytes());
            v
        }
        Value::String(s)  => {
            let mut v = vec![TAG_STRING];
            v.extend_from_slice(s.as_bytes());
            v
        }
        other => {
            let mut v = vec![TAG_OTHER];
            v.extend_from_slice(serde_json::to_string(other).unwrap().as_bytes());
            v
        }
    };
    v.push(0x00); // separator between value and doc_key
    v
}

fn encode_field_key(val: &Value, doc_key: &str) -> Vec<u8> {
    let mut prefix = encode_field_prefix(val);
    prefix.extend_from_slice(doc_key.as_bytes());
    prefix
}

/// Returns the exclusive end of a prefix range. Increments the \x00 separator
/// byte to \x01, giving [prefix, prefix_end) = all entries with this field value.
fn prefix_end(prefix: &[u8]) -> Vec<u8> {
    let mut end = prefix.to_vec();
    *end.last_mut().unwrap() = 0x01;
    end
}

fn extract_doc_key(index_key: &[u8]) -> Option<String> {
    // Find the \x00 separator and take everything after it.
    let sep = index_key.iter().position(|&b| b == 0x00)?;
    String::from_utf8(index_key[sep + 1..].to_vec()).ok()
}

// ── NodeStore ─────────────────────────────────────────────────────────────────

struct NodeStore {
    /// Primary storage: doc_key (UTF-8) → JSON bytes.
    docs: Arc<BTree>,
    /// Per-field secondary indexes: field_name → BTree({prefix}{doc_key} → empty).
    indexes: HashMap<String, Arc<BTree>>,
    /// Optional in-memory hot cache for reference nodes.
    /// Populated by `preload_hot` and kept up-to-date on every insert/update/delete.
    hot: Option<RwLock<HashMap<String, Value>>>,
}

impl NodeStore {
    fn new(docs: Arc<BTree>) -> Self {
        Self { docs, indexes: HashMap::new(), hot: None }
    }

    fn enable_hot(&mut self) {
        if self.hot.is_none() {
            self.hot = Some(RwLock::new(HashMap::new()));
        }
    }

    fn hot_put(&self, doc_key: &str, val: Value) {
        if let Some(cache) = &self.hot {
            cache.write().insert(doc_key.to_string(), val);
        }
    }

    fn hot_remove(&self, doc_key: &str) {
        if let Some(cache) = &self.hot {
            cache.write().remove(doc_key);
        }
    }

    fn hot_snapshot(&self) -> Option<Vec<Value>> {
        self.hot.as_ref().map(|c| c.read().values().cloned().collect())
    }

    // ── Index helpers ─────────────────────────────────────────────────────────

    fn index_insert_doc(&self, doc_key: &str, doc: &Value) -> Result<(), DbError> {
        for (field, idx) in &self.indexes {
            if let Some(field_val) = doc.get(field) {
                let key = encode_field_key(field_val, doc_key);
                idx.insert(&key, b"")?;
            }
        }
        Ok(())
    }

    fn index_delete_doc(&self, doc_key: &str, doc: &Value) -> Result<(), DbError> {
        for (field, idx) in &self.indexes {
            if let Some(field_val) = doc.get(field) {
                let key = encode_field_key(field_val, doc_key);
                idx.delete(&key)?;
            }
        }
        Ok(())
    }

    fn index_lookup_keys(&self, field: &str, val: &Value) -> Result<Vec<String>, DbError> {
        let idx = self.indexes.get(field)
            .ok_or_else(|| DbError::Corrupt(format!("no index on field '{field}'")))?;
        let prefix = encode_field_prefix(val);
        let end = prefix_end(&prefix);
        let pairs = idx.range(&prefix, &end)?;
        pairs.iter()
            .filter_map(|(k, _)| extract_doc_key(k))
            .map(Ok)
            .collect()
    }
}

// ── GraphNode: describes how to load one node for a query ─────────────────────

/// Specifies how to populate one named node of the graph for a single query.
pub enum GraphNode<'a> {
    /// Value provided directly (e.g. the incoming stream event).
    Inline { node: &'a str, value: Value },

    /// Load a single document by its storage key.
    ByKey { node: &'a str, doc_key: &'a str },

    /// Load all documents whose `field` equals `value` (uses index if present,
    /// falls back to full scan otherwise).
    ByField { node: &'a str, field: &'a str, value: &'a Value },

    /// Load all documents in the node. Suitable for small reference tables.
    All { node: &'a str },

    /// Read from the in-memory hot cache (zero disk I/O). Requires `preload_hot`
    /// to have been called for this node.
    Hot { node: &'a str },
}

// ── GraphBucket ───────────────────────────────────────────────────────────────

/// Cross-document graph storage optimised for stream processing.
pub struct GraphBucket {
    nodes: RwLock<HashMap<String, NodeStore>>,
    exprs: Arc<ExprBucket>,
    dir: PathBuf,
    name: String,
}

impl GraphBucket {
    pub(super) fn open(
        dir: impl AsRef<Path>,
        name: &str,
        exprs: Arc<ExprBucket>,
    ) -> Result<Arc<Self>, DbError> {
        Ok(Arc::new(Self {
            nodes: RwLock::new(HashMap::new()),
            exprs,
            dir: dir.as_ref().to_path_buf(),
            name: name.to_string(),
        }))
    }

    // ── Node management ───────────────────────────────────────────────────────

    /// Register a new node. Must be called before inserting documents.
    pub fn add_node(&self, node: &str) -> Result<(), DbError> {
        let path = self.node_docs_path(node);
        let tree = BTree::open(&path)?;
        self.nodes.write().insert(node.to_string(), NodeStore::new(tree));
        Ok(())
    }

    /// Add a secondary index on `field` for `node`. Rebuilds from existing data.
    pub fn add_index(&self, node: &str, field: &str) -> Result<(), DbError> {
        let path = self.node_index_path(node, field);
        let idx = BTree::open(&path)?;

        // Index all existing documents.
        let docs = {
            let guard = self.nodes.read();
            let ns = guard.get(node).ok_or_else(|| DbError::ExprNotFound(node.to_string()))?;
            ns.docs.all()?
        };
        for (key_bytes, val_bytes) in &docs {
            let doc_key = std::str::from_utf8(key_bytes)
                .map_err(|_| DbError::Corrupt("non-UTF8 doc key".into()))?;
            let doc: Value = serde_json::from_slice(val_bytes)
                .map_err(|e| DbError::Corrupt(e.to_string()))?;
            if let Some(field_val) = doc.get(field) {
                let ikey = encode_field_key(field_val, doc_key);
                idx.insert(&ikey, b"")?;
            }
        }

        let mut guard = self.nodes.write();
        let ns = guard.get_mut(node).ok_or_else(|| DbError::ExprNotFound(node.to_string()))?;
        ns.indexes.insert(field.to_string(), idx);
        Ok(())
    }

    /// Load all documents for `node` into the hot in-memory cache.
    /// After this call, `GraphNode::Hot` for this node performs no disk I/O.
    pub fn preload_hot(&self, node: &str) -> Result<(), DbError> {
        let mut guard = self.nodes.write();
        let ns = guard.get_mut(node).ok_or_else(|| DbError::ExprNotFound(node.to_string()))?;
        ns.enable_hot();
        let all = ns.docs.all()?;
        let cache = ns.hot.as_ref().unwrap();
        let mut w = cache.write();
        for (k, v) in &all {
            let key = String::from_utf8(k.clone())
                .map_err(|_| DbError::Corrupt("non-UTF8 key".into()))?;
            let val: Value = serde_json::from_slice(v)
                .map_err(|e| DbError::Corrupt(e.to_string()))?;
            w.insert(key, val);
        }
        Ok(())
    }

    // ── Document operations ───────────────────────────────────────────────────

    /// Insert or replace a document in `node`. Updates all indexes and hot cache.
    pub fn insert(&self, node: &str, doc_key: &str, doc: &Value) -> Result<(), DbError> {
        let guard = self.nodes.read();
        let ns = guard.get(node).ok_or_else(|| DbError::ExprNotFound(node.to_string()))?;

        // If update: remove old index entries first.
        if let Some(old_bytes) = ns.docs.get(doc_key.as_bytes())? {
            let old: Value = serde_json::from_slice(&old_bytes)
                .map_err(|e| DbError::Corrupt(e.to_string()))?;
            ns.index_delete_doc(doc_key, &old)?;
        }

        let bytes = serde_json::to_vec(doc)
            .map_err(|e| DbError::Serialize(e.to_string()))?;
        ns.docs.insert(doc_key.as_bytes(), &bytes)?;
        ns.index_insert_doc(doc_key, doc)?;
        ns.hot_put(doc_key, doc.clone());
        Ok(())
    }

    /// Delete a document from `node`. Removes index entries and hot cache.
    pub fn delete(&self, node: &str, doc_key: &str) -> Result<bool, DbError> {
        let guard = self.nodes.read();
        let ns = guard.get(node).ok_or_else(|| DbError::ExprNotFound(node.to_string()))?;
        if let Some(bytes) = ns.docs.get(doc_key.as_bytes())? {
            let doc: Value = serde_json::from_slice(&bytes)
                .map_err(|e| DbError::Corrupt(e.to_string()))?;
            ns.index_delete_doc(doc_key, &doc)?;
        }
        let deleted = ns.docs.delete(doc_key.as_bytes())?;
        if deleted {
            ns.hot_remove(doc_key);
        }
        Ok(deleted)
    }

    /// Fetch a single document by key.
    pub fn get(&self, node: &str, doc_key: &str) -> Result<Option<Value>, DbError> {
        let guard = self.nodes.read();
        let ns = guard.get(node).ok_or_else(|| DbError::ExprNotFound(node.to_string()))?;
        match ns.docs.get(doc_key.as_bytes())? {
            Some(b) => Ok(Some(serde_json::from_slice(&b).map_err(|e| DbError::Corrupt(e.to_string()))?)),
            None => Ok(None),
        }
    }

    // ── Query ─────────────────────────────────────────────────────────────────

    /// Evaluate `expr` on a jetro Graph built from the given node loading specs.
    ///
    /// Each `GraphNode` describes how to populate one named node:
    /// - `Inline`  — value provided directly; no disk access.
    /// - `ByKey`   — single point lookup: O(log n).
    /// - `ByField` — secondary index scan: O(log n + k).
    /// - `All`     — full node scan: O(n). Use only for small reference data.
    /// - `Hot`     — in-memory read: O(1). Requires `preload_hot` first.
    pub fn query(&self, nodes: &[GraphNode<'_>], expr: &str) -> Result<Value, DbError> {
        let guard = self.nodes.read();
        let mut graph = Graph::new();

        for node_spec in nodes {
            let (name, value) = self.resolve_node(&guard, node_spec)?;
            graph.add_node(name, value);
        }

        graph.query(expr).map_err(|e| DbError::EvalError(e.to_string()))
    }

    /// Convenience for the common stream processing pattern:
    /// - `stream_node`: the node receiving the event.
    /// - `event`: the incoming JSON document.
    /// - `lookups`: slice of `(node, field, value)` — index lookups to populate
    ///   reference nodes.
    /// - `expr`: jetro graph expression to evaluate.
    ///
    /// # Example
    /// ```text
    /// graph.process_stream(
    ///     "orders",
    ///     &order_json,
    ///     &[("customers", "id", &order_json["customer_id"])],
    ///     ">/orders/#join('customers', 'customer_id', 'id')",
    /// )
    /// ```
    pub fn process_stream(
        &self,
        stream_node: &str,
        event: Value,
        lookups: &[(&str, &str, &Value)],
        expr: &str,
    ) -> Result<Value, DbError> {
        let mut node_specs: Vec<GraphNode<'_>> = vec![
            GraphNode::Inline { node: stream_node, value: event },
        ];
        for (node, field, val) in lookups {
            node_specs.push(GraphNode::ByField { node, field, value: val });
        }
        self.query(&node_specs, expr)
    }

    /// Evaluate a named expression (looked up from the ExprBucket) on the graph.
    pub fn query_named(&self, nodes: &[GraphNode<'_>], expr_key: &str) -> Result<Value, DbError> {
        let expr = self.exprs.get(expr_key)?
            .ok_or_else(|| DbError::ExprNotFound(expr_key.to_string()))?;
        self.query(nodes, &expr)
    }

    // ── Internal ──────────────────────────────────────────────────────────────

    fn resolve_node<'g>(
        &self,
        guard: &'g parking_lot::RwLockReadGuard<'g, HashMap<String, NodeStore>>,
        spec: &GraphNode<'_>,
    ) -> Result<(String, Value), DbError> {
        match spec {
            GraphNode::Inline { node, value } => {
                Ok((node.to_string(), value.clone()))
            }
            GraphNode::ByKey { node, doc_key } => {
                let ns = get_ns(guard, node)?;
                let val = match ns.docs.get(doc_key.as_bytes())? {
                    Some(b) => serde_json::from_slice(&b).map_err(|e| DbError::Corrupt(e.to_string()))?,
                    None => Value::Null,
                };
                Ok((node.to_string(), Value::Array(vec![val])))
            }
            GraphNode::ByField { node, field, value } => {
                let ns = get_ns(guard, node)?;
                let doc_keys = ns.index_lookup_keys(field, value)?;
                let docs = load_docs_by_keys(ns, &doc_keys)?;
                Ok((node.to_string(), Value::Array(docs)))
            }
            GraphNode::All { node } => {
                let ns = get_ns(guard, node)?;
                let all = ns.docs.all()?;
                let docs = deserialize_docs(&all)?;
                Ok((node.to_string(), Value::Array(docs)))
            }
            GraphNode::Hot { node } => {
                let ns = get_ns(guard, node)?;
                let docs = ns.hot_snapshot()
                    .ok_or_else(|| DbError::Corrupt(format!("node '{node}' has no hot cache; call preload_hot first")))?;
                Ok((node.to_string(), Value::Array(docs)))
            }
        }
    }

    fn node_docs_path(&self, node: &str) -> PathBuf {
        self.dir.join(format!("{}.{}.docs", self.name, node))
    }

    fn node_index_path(&self, node: &str, field: &str) -> PathBuf {
        self.dir.join(format!("{}.{}.idx.{}", self.name, node, field))
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn get_ns<'g>(
    guard: &'g parking_lot::RwLockReadGuard<'g, HashMap<String, NodeStore>>,
    node: &str,
) -> Result<&'g NodeStore, DbError> {
    guard.get(node).ok_or_else(|| DbError::ExprNotFound(node.to_string()))
}

fn load_docs_by_keys(ns: &NodeStore, keys: &[String]) -> Result<Vec<Value>, DbError> {
    keys.iter()
        .filter_map(|k| ns.docs.get(k.as_bytes()).transpose())
        .map(|r| {
            r.map_err(DbError::Io).and_then(|b| {
                serde_json::from_slice(&b).map_err(|e| DbError::Corrupt(e.to_string()))
            })
        })
        .collect()
}

fn deserialize_docs(pairs: &[(Vec<u8>, Vec<u8>)]) -> Result<Vec<Value>, DbError> {
    pairs.iter()
        .map(|(_, v)| serde_json::from_slice(v).map_err(|e| DbError::Corrupt(e.to_string())))
        .collect()
}

