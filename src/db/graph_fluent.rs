//! Fluent builder over [`GraphBucket::query`].
//!
//! The underlying API takes `&[GraphNode<'_>]` — an array of enum
//! variants with borrowed string fields.  That works but reads poorly
//! at call sites:
//!
//! ```ignore
//! graph.query(&[
//!     GraphNode::Inline  { node: "orders",    value: event_json },
//!     GraphNode::ByField { node: "customers", field: "id", value: &cust_id },
//!     GraphNode::Hot     { node: "products" },
//! ], "$.orders.len()")?;
//! ```
//!
//! The builder replaces that with a chain:
//!
//! ```ignore
//! graph.build()
//!     .inline("orders", event_json)
//!     .lookup("customers", "id", &cust_id)
//!     .hot("products")
//!     .query("$.orders.len()")?;
//! ```
//!
//! The resulting call goes through the same `query` codepath.

use serde_json::Value;

use super::error::DbError;
use super::graph_bucket::{GraphBucket, GraphNode};

enum Step {
    Inline  { node: String, value: Value },
    ByKey   { node: String, doc_key: String },
    ByField { node: String, field: String, value: Value },
    All     { node: String },
    Hot     { node: String },
}

/// Builder produced by [`GraphBucket::build`].
pub struct GraphQueryBuilder<'g> {
    bucket: &'g GraphBucket,
    steps:  Vec<Step>,
}

impl<'g> GraphQueryBuilder<'g> {
    pub(super) fn new(bucket: &'g GraphBucket) -> Self {
        Self { bucket, steps: Vec::new() }
    }

    /// Attach `value` directly as the named node's content.  No disk IO.
    pub fn inline(mut self, node: impl Into<String>, value: Value) -> Self {
        self.steps.push(Step::Inline { node: node.into(), value });
        self
    }

    /// Load a single stored document by its primary key.
    pub fn by_key(mut self, node: impl Into<String>, doc_key: impl Into<String>) -> Self {
        self.steps.push(Step::ByKey { node: node.into(), doc_key: doc_key.into() });
        self
    }

    /// Index lookup: every document in `node` whose `field == value`.
    pub fn lookup(
        mut self,
        node: impl Into<String>,
        field: impl Into<String>,
        value: &Value,
    ) -> Self {
        self.steps.push(Step::ByField {
            node:  node.into(),
            field: field.into(),
            value: value.clone(),
        });
        self
    }

    /// Load every document in `node`.  O(n) scan — use only for
    /// small reference tables.
    pub fn all(mut self, node: impl Into<String>) -> Self {
        self.steps.push(Step::All { node: node.into() });
        self
    }

    /// Read `node` from its in-memory hot cache.  Requires
    /// [`GraphBucket::preload_hot`] to have been called first.
    pub fn hot(mut self, node: impl Into<String>) -> Self {
        self.steps.push(Step::Hot { node: node.into() });
        self
    }

    /// Evaluate `expr` against the assembled virtual graph.
    pub fn query(&self, expr: &str) -> Result<Value, DbError> {
        let nodes = self.borrow_steps();
        self.bucket.query(&nodes, expr)
    }

    /// Evaluate a named expression (looked up from the `ExprBucket`
    /// attached to this `GraphBucket`) against the assembled graph.
    pub fn query_named(&self, expr_key: &str) -> Result<Value, DbError> {
        let nodes = self.borrow_steps();
        self.bucket.query_named(&nodes, expr_key)
    }

    fn borrow_steps(&self) -> Vec<GraphNode<'_>> {
        self.steps.iter().map(|s| match s {
            Step::Inline  { node, value }        => GraphNode::Inline  { node, value: value.clone() },
            Step::ByKey   { node, doc_key }      => GraphNode::ByKey   { node, doc_key },
            Step::ByField { node, field, value } => GraphNode::ByField { node, field, value },
            Step::All     { node }               => GraphNode::All     { node },
            Step::Hot     { node }               => GraphNode::Hot     { node },
        }).collect()
    }
}

impl GraphBucket {
    /// Begin a fluent graph query.  See [`GraphQueryBuilder`].
    pub fn build(&self) -> GraphQueryBuilder<'_> {
        GraphQueryBuilder::new(self)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Database;
    use serde_json::json;

    fn setup() -> (Database, std::sync::Arc<crate::db::ExprBucket>, std::sync::Arc<GraphBucket>) {
        let db = Database::memory().unwrap();
        let exprs = db.expr_bucket("exprs").unwrap();
        let g = db.graph_bucket("analytics", &exprs).unwrap();
        g.add_node("orders").unwrap();
        g.add_node("customers").unwrap();
        g.add_index("customers", "id").unwrap();
        g.insert("customers", "c1", &json!({"id": 10, "name": "Alice"})).unwrap();
        g.insert("customers", "c2", &json!({"id": 20, "name": "Bob"})).unwrap();
        (db, exprs, g)
    }

    #[test]
    fn inline_and_lookup() {
        let (_db, _ex, g) = setup();
        let order = json!({"id": 1, "customer_id": 10, "price": 9.99});
        let result = g.build()
            .inline("orders", json!([order]))
            .lookup("customers", "id", &json!(10))
            .query("$.customers[0].name").unwrap();
        assert_eq!(result, json!("Alice"));
    }

    #[test]
    fn hot_cache() {
        let (_db, _ex, g) = setup();
        g.preload_hot("customers").unwrap();
        let r = g.build()
            .inline("orders", json!([]))
            .hot("customers")
            .query("$.customers.len()").unwrap();
        assert_eq!(r, json!(2));
    }

    #[test]
    fn all_scan() {
        let (_db, _ex, g) = setup();
        let r = g.build()
            .all("customers")
            .query("$.customers.len()").unwrap();
        assert_eq!(r, json!(2));
    }

    #[test]
    fn query_named() {
        let (_db, exprs, g) = setup();
        exprs.put("count_customers", "$.customers.len()").unwrap();
        let r = g.build()
            .all("customers")
            .query_named("count_customers").unwrap();
        assert_eq!(r, json!(2));
    }
}
