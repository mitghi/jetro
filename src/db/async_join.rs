//! Async façade over [`Join`].
//!
//! Feature-gated behind `async`. Writes and blocking waits are off-loaded
//! to [`tokio::task::spawn_blocking`] so they don't stall the reactor
//! while the B+ tree flushes or a condvar parks.
//!
//! ```ignore
//! use jetro::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), jetro::Error> {
//!     let db = Database::memory()?;
//!     let j = db.join("events")
//!         .id("order_id")
//!         .kinds(["order", "payment", "shipment"])
//!         .open()?
//!         .into_async();
//!
//!     j.emit("order",    &json!({"order_id": "A1"})).await?;
//!     j.emit("payment",  &json!({"order_id": "A1"})).await?;
//!     j.emit("shipment", &json!({"order_id": "A1"})).await?;
//!
//!     let _doc = j.wait("A1").await?;
//!     Ok(())
//! }
//! ```

#![cfg(feature = "async")]

use std::sync::Arc;
use std::time::Duration;

use serde::de::DeserializeOwned;
use serde_json::Value;

use super::error::DbError;
use super::join::{IntoJoinId, Join, JoinedDoc};

/// Async wrapper over [`Join`]. Cheap to clone — shares the inner handle.
#[derive(Clone)]
pub struct AsyncJoin {
    inner: Arc<Join>,
}

impl AsyncJoin {
    pub(crate) fn new(inner: Arc<Join>) -> Self {
        Self { inner }
    }

    /// Borrow the underlying sync handle.
    pub fn sync(&self) -> &Arc<Join> {
        &self.inner
    }

    pub async fn emit(&self, kind: &str, doc: &Value) -> Result<(), DbError> {
        let inner = Arc::clone(&self.inner);
        let kind = kind.to_string();
        let doc = doc.clone();
        tokio::task::spawn_blocking(move || inner.emit(&kind, &doc))
            .await
            .expect("join task panicked")
    }

    pub async fn remove(&self, id: impl IntoJoinId + Send + 'static) -> Result<(), DbError> {
        let inner = Arc::clone(&self.inner);
        tokio::task::spawn_blocking(move || inner.remove(id))
            .await
            .expect("join task panicked")
    }

    pub async fn unemit(
        &self,
        kind: &str,
        id: impl IntoJoinId + Send + 'static,
    ) -> Result<bool, DbError> {
        let inner = Arc::clone(&self.inner);
        let kind = kind.to_string();
        tokio::task::spawn_blocking(move || inner.unemit(&kind, id))
            .await
            .expect("join task panicked")
    }

    pub async fn wait(
        &self,
        id: impl IntoJoinId + Send + 'static,
    ) -> Result<JoinedDoc, DbError> {
        let inner = Arc::clone(&self.inner);
        tokio::task::spawn_blocking(move || inner.wait(id))
            .await
            .expect("join task panicked")
    }

    /// Async wait with an external timeout via `tokio::time::timeout`.
    /// Returns `Ok(None)` on timeout so callers can distinguish from
    /// a genuine DB error.
    pub async fn wait_for(
        &self,
        id: impl IntoJoinId + Send + 'static,
        dur: Duration,
    ) -> Result<Option<JoinedDoc>, DbError> {
        let inner = Arc::clone(&self.inner);
        match tokio::time::timeout(
            dur,
            tokio::task::spawn_blocking(move || inner.wait(id)),
        )
        .await
        {
            Ok(join_result) => {
                let doc = join_result.expect("join task panicked")?;
                Ok(Some(doc))
            }
            Err(_elapsed) => Ok(None),
        }
    }

    pub async fn peek(
        &self,
        id: impl IntoJoinId + Send + 'static,
    ) -> Result<Option<JoinedDoc>, DbError> {
        let inner = Arc::clone(&self.inner);
        tokio::task::spawn_blocking(move || inner.peek(id))
            .await
            .expect("join task panicked")
    }

    pub async fn arrived(&self, id: impl IntoJoinId + Send + 'static) -> Vec<String> {
        let inner = Arc::clone(&self.inner);
        tokio::task::spawn_blocking(move || inner.arrived(id))
            .await
            .expect("join task panicked")
    }

    /// Run a jetro expression over the joined document, awaiting arrival.
    pub async fn run(
        &self,
        id: impl IntoJoinId + Send + 'static,
        expr: &str,
    ) -> Result<Value, DbError> {
        let inner = Arc::clone(&self.inner);
        let expr = expr.to_string();
        tokio::task::spawn_blocking(move || inner.on(id).run(&expr))
            .await
            .expect("join task panicked")
    }

    /// Run + deserialise to `T`.
    pub async fn get<T: DeserializeOwned + Send + 'static>(
        &self,
        id: impl IntoJoinId + Send + 'static,
        expr: &str,
    ) -> Result<T, DbError> {
        let inner = Arc::clone(&self.inner);
        let expr = expr.to_string();
        tokio::task::spawn_blocking(move || inner.on(id).get::<T>(&expr))
            .await
            .expect("join task panicked")
    }
}

// ── Conversion helper ────────────────────────────────────────────────────────

impl Join {
    /// Wrap this `Join` in an async façade.
    pub fn into_async(self: Arc<Self>) -> AsyncJoin {
        AsyncJoin::new(self)
    }
}
