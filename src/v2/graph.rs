//! Graph layer — query across multiple named JSON documents using the v2 VM.
//!
//! A [`Graph`] holds a set of named JSON values (nodes). When you call
//! [`Graph::query`], all nodes are merged into a single virtual root object
//! `{ "node_name": <value>, ... }` and the expression is evaluated against it.
//! Reference node data as `$.node_name...` in v2 syntax.
//!
//! # Example
//!
//! ```rust,no_run
//! use jetro::v2::graph::Graph;
//! use serde_json::json;
//!
//! let mut g = Graph::new();
//! g.add_node("orders",    json!([{"id": 1, "price": 9.99}]))
//!  .add_node("customers", json!([{"id": 1, "name": "Alice"}]));
//!
//! // Sum all prices
//! let result = g.query("$.orders.sum(price)").expect("query failed");
//! ```

use std::sync::Arc;
use indexmap::IndexMap;
use serde_json::Value;

use super::eval::EvalError;
use super::eval::{MethodRegistry, Method};
use super::vm::VM;

// ── Error ─────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum GraphError {
    Eval(EvalError),
    NodeNotFound(String),
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphError::Eval(e)         => write!(f, "{}", e),
            GraphError::NodeNotFound(n) => write!(f, "node '{}' not found", n),
        }
    }
}
impl std::error::Error for GraphError {}
impl From<EvalError> for GraphError { fn from(e: EvalError) -> Self { GraphError::Eval(e) } }

// ── Graph ─────────────────────────────────────────────────────────────────────

/// A collection of named JSON documents that can be queried together via the v2 VM.
///
/// # Thread safety
///
/// `Graph` is `Send + Sync` (nodes are plain `serde_json::Value`). The internal
/// [`VM`] is *not* `Sync`, so wrap in `Mutex` for concurrent use.
pub struct Graph {
    nodes:    IndexMap<String, Value>,
    vm:       VM,
}

impl Graph {
    /// Create an empty graph with a fresh VM.
    pub fn new() -> Self {
        Self { nodes: IndexMap::new(), vm: VM::new() }
    }

    /// Create a graph backed by a VM with custom capacity hints.
    ///
    /// `compile_cap` — compile-cache slots; `resolution_cap` — resolution-cache slots.
    pub fn with_capacity(compile_cap: usize, resolution_cap: usize) -> Self {
        Self {
            nodes: IndexMap::new(),
            vm:    VM::with_capacity(compile_cap, resolution_cap),
        }
    }

    // ── Node management ───────────────────────────────────────────────────────

    /// Add (or replace) a named node. Returns `&mut self` for chaining.
    pub fn add_node<S: Into<String>>(&mut self, name: S, value: Value) -> &mut Self {
        self.nodes.insert(name.into(), value);
        self
    }

    /// Return a reference to a node value, if it exists.
    pub fn get_node(&self, name: &str) -> Option<&Value> {
        self.nodes.get(name)
    }

    /// Remove a node, returning its value if it existed.
    pub fn remove_node(&mut self, name: &str) -> Option<Value> {
        self.nodes.shift_remove(name)
    }

    /// Return the number of nodes in the graph.
    pub fn len(&self) -> usize { self.nodes.len() }

    /// Return `true` if the graph has no nodes.
    pub fn is_empty(&self) -> bool { self.nodes.is_empty() }

    /// Return an iterator over node names.
    pub fn node_names(&self) -> impl Iterator<Item = &str> {
        self.nodes.keys().map(|s| s.as_str())
    }

    // ── Query API ─────────────────────────────────────────────────────────────

    /// Evaluate a v2 expression against the **virtual root** — a JSON object
    /// `{ "node_name": value, ... }` containing all nodes.
    ///
    /// Navigate into a node via `$.node_name...`.
    pub fn query(&mut self, expr: &str) -> Result<Value, GraphError> {
        let root = self.virtual_root();
        Ok(self.vm.run_str(expr, &root)?)
    }

    /// Evaluate a v2 expression against a **single named node** directly.
    pub fn query_node(&mut self, node: &str, expr: &str) -> Result<Value, GraphError> {
        let value = self.nodes.get(node)
            .ok_or_else(|| GraphError::NodeNotFound(node.to_string()))?
            .clone();
        Ok(self.vm.run_str(expr, &value)?)
    }

    /// Register a custom method callable from query expressions.
    pub fn register_method(&mut self, name: impl Into<String>, method: impl Method + 'static) {
        self.vm.register(name, method);
    }

    // ── VM access ─────────────────────────────────────────────────────────────

    /// Access the underlying VM (e.g. to inspect cache stats).
    pub fn vm(&self) -> &VM { &self.vm }

    /// Mutable access to the underlying VM.
    pub fn vm_mut(&mut self) -> &mut VM { &mut self.vm }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Build the virtual root object containing all nodes as top-level fields.
    pub fn virtual_root(&self) -> Value {
        let map: serde_json::Map<String, Value> = self.nodes.iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        Value::Object(map)
    }

    /// Alias for `query` — reads better when constructing message schemas.
    ///
    /// ```text
    /// graph.message(r#"{ "total": $.orders.sum(price), "count": $.orders.len() }"#)
    /// ```
    pub fn message(&mut self, schema: &str) -> Result<Value, GraphError> {
        self.query(schema)
    }
}

impl Default for Graph {
    fn default() -> Self { Self::new() }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_graph() -> Graph {
        let mut g = Graph::new();
        g.add_node("orders", json!([
            {"id": 1, "customer_id": 10, "price": 9.99, "is_gratis": false},
            {"id": 2, "customer_id": 20, "price": 4.50, "is_gratis": false},
            {"id": 3, "customer_id": 10, "price": 0.00, "is_gratis": true},
        ])).add_node("customers", json!([
            {"id": 10, "name": "Alice"},
            {"id": 20, "name": "Bob"},
        ]));
        g
    }

    #[test]
    fn test_node_count() {
        let g = make_graph();
        assert_eq!(g.len(), 2);
    }

    #[test]
    fn test_virtual_root_shape() {
        let g = make_graph();
        let root = g.virtual_root();
        let obj = root.as_object().unwrap();
        assert!(obj.contains_key("orders"));
        assert!(obj.contains_key("customers"));
    }

    #[test]
    fn test_query_len() {
        let mut g = make_graph();
        let r = g.query("$.orders.len()").unwrap();
        assert_eq!(r, json!(3));
    }

    #[test]
    fn test_query_sum() {
        let mut g = make_graph();
        let r = g.query("$.orders.sum(price)").unwrap();
        let total = r.as_f64().unwrap();
        assert!((total - 14.49).abs() < 0.001);
    }

    #[test]
    fn test_query_filter_sum() {
        let mut g = make_graph();
        let r = g.query("$.orders.filter(is_gratis == false).sum(price)").unwrap();
        let total = r.as_f64().unwrap();
        assert!((total - 14.49).abs() < 0.001);
    }

    #[test]
    fn test_query_node_direct() {
        let mut g = make_graph();
        let r = g.query_node("orders", "$.len()").unwrap();
        assert_eq!(r, json!(3));
    }

    #[test]
    fn test_query_node_not_found() {
        let mut g = make_graph();
        let result = g.query_node("missing", "$.len()");
        assert!(matches!(result, Err(GraphError::NodeNotFound(_))));
    }

    #[test]
    fn test_group_by() {
        let mut g = make_graph();
        let r = g.query("$.orders.groupBy(customer_id)").unwrap();
        let obj = r.as_object().unwrap();
        assert_eq!(obj.len(), 2);
    }

    #[test]
    fn test_remove_node() {
        let mut g = make_graph();
        assert!(g.remove_node("orders").is_some());
        assert_eq!(g.len(), 1);
        assert!(g.get_node("orders").is_none());
    }

    #[test]
    fn test_compile_cache() {
        let mut g = make_graph();
        // Run same query twice — second should hit compile cache
        g.query("$.orders.len()").unwrap();
        g.query("$.orders.len()").unwrap();
        let (cache_size, _) = g.vm().cache_stats();
        assert_eq!(cache_size, 1);
    }

    #[test]
    fn test_message_alias() {
        let mut g = make_graph();
        let r = g.message(r#"{"count": $.orders.len(), "names": $.customers.map(name)}"#).unwrap();
        let obj = r.as_object().unwrap();
        assert!(obj.contains_key("count"));
        assert!(obj.contains_key("names"));
    }

    #[test]
    fn test_custom_method() {
        let mut g = make_graph();
        g.register_method("double_len", |recv: super::super::eval::value::Val, _args: &[super::super::eval::value::Val]| {
            use super::super::eval::value::Val;
            let n = match &recv {
                Val::Arr(a) => a.len() as i64 * 2,
                _ => 0,
            };
            Ok(Val::Int(n))
        });
        let r = g.query("$.orders.double_len()").unwrap();
        assert_eq!(r, json!(6));
    }
}
