//! Graph layer — query across multiple named JSON documents.
//!
//! A [`Graph`] holds a set of named JSON values (nodes). When you call
//! [`Graph::query`], all nodes are merged into a single virtual root object
//! `{ "node_name": <value>, ... }` and the Jetro expression is evaluated
//! against it. This means you reference node data as `>/node_name/...`.
//!
//! # Example
//!
//! ```rust
//! use jetro::graph::Graph;
//! use serde_json::json;
//!
//! let mut g = Graph::new();
//! g.add_node("orders",    json!([{"id": 1, "price": 9.99}, {"id": 2, "price": 4.50}]))
//!  .add_node("customers", json!([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]));
//!
//! // Total revenue across all orders
//! let mut result = g.query(">/orders/..price/#sum").expect("query failed");
//! ```

use std::collections::HashMap;
use serde_json::Value;
use crate::context::{Error, Path, PathResult};

/// A collection of named JSON documents that can be queried together.
pub struct Graph {
    nodes: HashMap<String, Value>,
}

impl Graph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
        }
    }

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
        self.nodes.remove(name)
    }

    /// List the names of all nodes in the graph.
    pub fn node_names(&self) -> Vec<&str> {
        self.nodes.keys().map(|s| s.as_str()).collect()
    }

    /// Evaluate a Jetro expression against the virtual root built from all nodes.
    ///
    /// The virtual root is a JSON object `{ "node_name": value, ... }`, so you
    /// navigate into a node via `>/node_name/...`.
    pub fn query<S: Into<String>>(&self, expr: S) -> Result<PathResult, Error> {
        let root = self.virtual_root();
        Path::collect(root, expr)
    }

    /// Evaluate a Jetro expression against a single named node directly.
    pub fn query_node<S: Into<String>>(&self, node: S, expr: S) -> Result<PathResult, Error> {
        let name = node.into();
        match self.nodes.get(&name) {
            Some(value) => Path::collect(value.clone(), expr),
            None => Err(Error::Eval(format!("node '{}' not found in graph", name))),
        }
    }

    /// Build the virtual root object containing all nodes as top-level fields.
    pub fn virtual_root(&self) -> Value {
        let mut map = serde_json::Map::new();
        for (name, value) in &self.nodes {
            map.insert(name.clone(), value.clone());
        }
        Value::Object(map)
    }

    /// Evaluate a message-schema expression — same as `query` but named for
    /// clarity when you're constructing a new message object.
    ///
    /// ```text
    /// graph.message(r#">{
    ///   "total":  >/orders/..price/#sum,
    ///   "count":  >/orders/#len
    /// }"#)
    /// ```
    pub fn message<S: Into<String>>(&self, schema: S) -> Result<PathResult, Error> {
        self.query(schema)
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_graph() -> Graph {
        let mut g = Graph::new();
        g.add_node(
            "orders",
            json!([
                {"id": 1, "customer_id": 10, "price": 9.99, "is_gratis": false},
                {"id": 2, "customer_id": 20, "price": 4.50, "is_gratis": false},
                {"id": 3, "customer_id": 10, "price": 0.00, "is_gratis": true},
            ]),
        )
        .add_node(
            "customers",
            json!([
                {"id": 10, "name": "Alice"},
                {"id": 20, "name": "Bob"},
            ]),
        );
        g
    }

    #[test]
    fn test_node_count() {
        let g = make_graph();
        assert_eq!(g.node_names().len(), 2);
    }

    #[test]
    fn test_simple_query() {
        let g = make_graph();
        let mut r = g.query(">/orders/#len").expect("query failed");
        let count: i64 = r.from_index(0).unwrap();
        assert_eq!(count, 3);
    }

    #[test]
    fn test_sum_across_node() {
        let g = make_graph();
        let mut r = g
            .query(">/orders/#filter('is_gratis' == false)/..price/#sum")
            .expect("query failed");
        let total: f64 = r.from_index(0).unwrap();
        assert!((total - 14.49).abs() < 0.001);
    }

    #[test]
    fn test_message_schema() {
        let g = make_graph();
        let schema = r#">{"total": >/orders/#filter('is_gratis' == false)/..price/#sum, "count": >/orders/#len}"#;
        let r = g.message(schema).expect("message failed");
        assert_eq!(r.0.len(), 1);
        let obj = r.0[0].as_object().unwrap();
        assert!(obj.contains_key("total"));
        assert!(obj.contains_key("count"));
    }

    #[test]
    fn test_group_by_across_node() {
        let g = make_graph();
        let mut r = g
            .query(">/orders/#group_by('customer_id')")
            .expect("query failed");
        let grouped = r.from_index::<serde_json::Value>(0).unwrap();
        let obj = grouped.as_object().unwrap();
        // customer_id 10 has 2 orders; customer_id 20 has 1
        assert_eq!(obj.len(), 2);
    }

    #[test]
    fn test_query_node() {
        let g = make_graph();
        let mut r = g.query_node("customers", ">/[0]/name").expect("query_node failed");
        let name: String = r.from_index(0).unwrap();
        assert_eq!(name, "Alice");
    }

    #[test]
    fn test_virtual_root_shape() {
        let g = make_graph();
        let root = g.virtual_root();
        let obj = root.as_object().unwrap();
        assert!(obj.contains_key("orders"));
        assert!(obj.contains_key("customers"));
    }
}
