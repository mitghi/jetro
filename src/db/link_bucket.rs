//! Blocking join combiner for stream processing.
//!
//! A [`LinkBucket`] accepts N kinds of JSON messages.  Each kind has a
//! designated *id field*.  A **link** is complete when one document of every
//! kind has arrived carrying the **same id value**.
//!
//! `get` and `query` block (optionally with a timeout) until the link for
//! the requested id is complete.  Multiple threads can wait on the same id
//! concurrently — they all wake when the last missing kind arrives.
//!
//! # Example
//!
//! ```rust,no_run
//! use jetro::db::Database;
//! use serde_json::json;
//! use std::thread;
//!
//! let tmp = tempfile::tempdir().unwrap();
//! let db  = Database::open(tmp.path()).unwrap();
//!
//! // Three message kinds — all share "order_id" as the join key.
//! let lb = db.link_bucket("orders", &[
//!     ("order",    "order_id"),
//!     ("payment",  "order_id"),
//!     ("shipment", "order_id"),
//! ]).unwrap();
//!
//! let lb2 = lb.clone();
//! thread::spawn(move || {
//!     lb2.insert("payment",  &json!({"order_id": "X1", "amount": 99.0})).unwrap();
//!     lb2.insert("shipment", &json!({"order_id": "X1", "tracking": "TRK42"})).unwrap();
//! });
//!
//! lb.insert("order", &json!({"order_id": "X1", "item": "Gadget"})).unwrap();
//!
//! // Blocks until all three kinds for "X1" have arrived.
//! let linked = lb.get(&json!("X1")).unwrap();
//! println!("{}", linked.docs["order"]["item"]);   // "Gadget"
//! println!("{}", linked.docs["payment"]["amount"]); // 99.0
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::{Condvar, Mutex, RwLock};
use serde_json::Value;

use crate::VM;
use super::btree::BTree;
use super::error::DbError;

// ── Types ─────────────────────────────────────────────────────────────────────

/// A completed link: one document per kind, keyed by kind name.
pub struct LinkedDoc {
    /// Kind-name → JSON document.
    pub docs: HashMap<String, Value>,
}

impl LinkedDoc {
    /// Merge all kind documents into a single JSON object `{"kind": doc, ...}`.
    pub fn into_json(self) -> Value {
        Value::Object(self.docs.into_iter().collect())
    }
}

// ── Internal ──────────────────────────────────────────────────────────────────

struct KindDef {
    id_field: Arc<str>,
    docs: Arc<BTree>,
}

/// Per-id state tracking which kinds have arrived.
struct PendingLink {
    /// Bitmask: bit `i` is set when kind `i` has been inserted.
    arrived: u64,
    /// Condvar signalled (notify_all) when `arrived == complete_mask`.
    /// The `bool` inside the Mutex is `true` once complete, preventing missed
    /// wakeups regardless of ordering between insert and wait.
    gate: Arc<(Mutex<bool>, Condvar)>,
}

// ── LinkBucket ────────────────────────────────────────────────────────────────

/// Blocking join combiner: waits until all N kinds share the same id value.
pub struct LinkBucket {
    /// Ordered list of kinds (index = bit position in `arrived` mask).
    kinds: Vec<(Arc<str>, KindDef)>,
    /// Bitmask with all N low bits set; completion condition is `arrived == complete_mask`.
    complete_mask: u64,
    /// In-memory tracking state, keyed by canonical id string.
    pending: RwLock<HashMap<String, PendingLink>>,
    _dir: PathBuf,
    _name: Arc<str>,
}

impl LinkBucket {
    /// Open (or create) a `LinkBucket` with the given kind definitions.
    ///
    /// `kinds` is a slice of `(kind_name, id_field)` pairs.  Order is stable
    /// (bit 0 = kinds[0], bit 1 = kinds[1], …).  Maximum 64 kinds.
    pub(crate) fn open(
        dir: &std::path::Path,
        name: &str,
        kinds: &[(&str, &str)],
    ) -> Result<Arc<Self>, DbError> {
        assert!(!kinds.is_empty(), "LinkBucket: at least one kind required");
        assert!(kinds.len() <= 64, "LinkBucket: maximum 64 kinds");

        let mut kind_list: Vec<(Arc<str>, KindDef)> = Vec::with_capacity(kinds.len());
        for &(kind_name, id_field) in kinds {
            let path = dir.join(format!("{name}.{kind_name}.link"));
            let docs = BTree::open(&path).map_err(DbError::Io)?;
            kind_list.push((
                Arc::from(kind_name),
                KindDef { id_field: Arc::from(id_field), docs },
            ));
        }

        let complete_mask = (1u64 << kinds.len()) - 1;

        Ok(Arc::new(Self {
            kinds: kind_list,
            complete_mask,
            pending: RwLock::new(HashMap::new()),
            _dir: dir.to_path_buf(),
            _name: Arc::from(name),
        }))
    }

    // ── Write ─────────────────────────────────────────────────────────────────

    /// Upsert a document for `kind`.
    ///
    /// - **Update**: if this kind has already arrived for the given id, the stored
    ///   document is overwritten.  The bitmask and gate are untouched — a complete
    ///   link stays complete; an incomplete link stays incomplete.
    /// - **Insert**: if this kind has not yet arrived, the document is stored, its
    ///   bit is set, and all blocked `get`/`query` callers are unblocked if the
    ///   link is now complete.
    pub fn insert(&self, kind: &str, doc: &Value) -> Result<(), DbError> {
        let (kind_idx, kind_def) = self
            .kinds
            .iter()
            .enumerate()
            .find(|(_, (n, _))| n.as_ref() == kind)
            .map(|(i, (_, def))| (i, def))
            .ok_or_else(|| DbError::EvalError(format!("unknown kind '{kind}'")))?;

        let id_val = doc
            .get(kind_def.id_field.as_ref())
            .ok_or_else(|| DbError::EvalError(format!(
                "document for kind '{kind}' missing id field '{}'",
                kind_def.id_field
            )))?;

        let key = value_to_key(id_val);
        let bit = 1u64 << kind_idx;

        // Determine whether this is an update (bit already set) or a fresh insert.
        let already_present = {
            let guard = self.pending.read();
            guard.get(&key).map_or(false, |l| l.arrived & bit != 0)
        };

        // Persist to BTree first so the doc is readable before any condvar fires.
        let bytes = serde_json::to_vec(doc).map_err(|e| DbError::Serialize(e.to_string()))?;
        kind_def.docs.insert(key.as_bytes(), &bytes).map_err(DbError::Io)?;

        if already_present {
            // Update path: doc overwritten above; arrival state unchanged.
            return Ok(());
        }

        // Insert path: set the bit, signal if now complete.
        let gate_to_signal: Option<Arc<(Mutex<bool>, Condvar)>> = {
            let mut guard = self.pending.write();
            let link = guard
                .entry(key.clone())
                .or_insert_with(|| PendingLink {
                    arrived: 0,
                    gate: Arc::new((Mutex::new(false), Condvar::new())),
                });
            // Re-check after acquiring write lock: a concurrent insert may have
            // set this bit between our read check and now.
            if link.arrived & bit != 0 {
                return Ok(());
            }
            link.arrived |= bit;
            if link.arrived == self.complete_mask {
                Some(Arc::clone(&link.gate))
            } else {
                None
            }
        };

        // Signal outside the pending lock to minimise contention.
        if let Some(gate) = gate_to_signal {
            let (lock, cvar) = &*gate;
            *lock.lock() = true;
            cvar.notify_all();
        }

        Ok(())
    }

    /// Delete all stored documents for `id` and remove its pending state.
    ///
    /// Does **not** wake blocked waiters — callers waiting on a deleted id will
    /// block until timeout or until new documents arrive.
    pub fn remove(&self, id: &Value) -> Result<(), DbError> {
        let key = value_to_key(id);
        for (_, kind_def) in &self.kinds {
            kind_def.docs.delete(key.as_bytes()).map_err(DbError::Io)?;
        }
        self.pending.write().remove(key.as_str());
        Ok(())
    }

    /// Remove one kind's contribution from a link, breaking it if it was complete.
    ///
    /// If the link was complete before this call, the gate is replaced with a fresh
    /// one so subsequent `get`/`query` callers block again instead of seeing stale
    /// `done=true`.
    ///
    /// Returns `true` if the kind's document existed and was removed.
    pub fn delete_kind(&self, kind: &str, id: &Value) -> Result<bool, DbError> {
        let (kind_idx, kind_def) = self
            .kinds
            .iter()
            .enumerate()
            .find(|(_, (n, _))| n.as_ref() == kind)
            .map(|(i, (_, def))| (i, def))
            .ok_or_else(|| DbError::EvalError(format!("unknown kind '{kind}'")))?;

        let key = value_to_key(id);
        let bit = 1u64 << kind_idx;

        let existed = kind_def.docs.delete(key.as_bytes()).map_err(DbError::Io)?;
        if !existed {
            return Ok(false);
        }

        let mut guard = self.pending.write();
        if let Some(link) = guard.get_mut(&key) {
            let was_complete = link.arrived == self.complete_mask;
            link.arrived &= !bit;
            if was_complete {
                // Replace gate so new waiters block on the now-broken link.
                link.gate = Arc::new((Mutex::new(false), Condvar::new()));
            }
        }
        Ok(true)
    }

    /// Overwrite an existing kind's document.
    ///
    /// If the kind was already present (bit set), the link completion state is
    /// unchanged — a complete link stays complete with the updated document.
    /// If the kind was not yet present, behaves identically to `insert`.
    pub fn update_kind(&self, kind: &str, doc: &Value) -> Result<(), DbError> {
        self.insert(kind, doc)
    }

    /// Return all kinds that have arrived so far without blocking.
    ///
    /// Useful for event-sourcing read-side queries on in-progress links.
    pub fn get_partial(&self, id: &Value) -> Result<HashMap<String, Value>, DbError> {
        let key = value_to_key(id);
        let arrived = {
            let guard = self.pending.read();
            guard.get(&key).map(|l| l.arrived).unwrap_or(0)
        };
        let mut docs = HashMap::new();
        for (i, (kind_name, kind_def)) in self.kinds.iter().enumerate() {
            if arrived & (1u64 << i) != 0 {
                if let Some(bytes) = kind_def.docs.get(key.as_bytes()).map_err(DbError::Io)? {
                    let doc: Value = serde_json::from_slice(&bytes)
                        .map_err(|e| DbError::Serialize(e.to_string()))?;
                    docs.insert(kind_name.as_ref().to_string(), doc);
                }
            }
        }
        Ok(docs)
    }

    // ── Read (blocking) ───────────────────────────────────────────────────────

    /// Block until the link for `id` is complete, then return all documents.
    ///
    /// Waits indefinitely.  Use [`get_timeout`](Self::get_timeout) for bounded waits.
    pub fn get(&self, id: &Value) -> Result<LinkedDoc, DbError> {
        // unwrap: None only on timeout; we pass infinite timeout
        Ok(self.get_timeout(id, None)?.unwrap())
    }

    /// Block until the link for `id` is complete or `timeout` elapses.
    ///
    /// Returns `None` on timeout, `Some(LinkedDoc)` on completion.
    pub fn get_timeout(&self, id: &Value, timeout: Option<Duration>) -> Result<Option<LinkedDoc>, DbError> {
        let key = value_to_key(id);

        // ── Fast path: already complete ───────────────────────────────────────
        {
            let guard = self.pending.read();
            if let Some(link) = guard.get(&key) {
                if link.arrived == self.complete_mask {
                    return Ok(Some(self.collect_by_key(&key)?));
                }
            }
        }

        // ── Slow path: obtain or create gate, then wait ───────────────────────
        let gate: Arc<(Mutex<bool>, Condvar)> = {
            let mut guard = self.pending.write();
            let link = guard
                .entry(key.clone())
                .or_insert_with(|| PendingLink {
                    arrived: 0,
                    gate: Arc::new((Mutex::new(false), Condvar::new())),
                });
            // Re-check after acquiring the write lock (handles the race between
            // the read check above and a concurrent insert that completed the link).
            if link.arrived == self.complete_mask {
                return Ok(Some(self.collect_by_key(&key)?));
            }
            Arc::clone(&link.gate)
        }; // pending write lock released here

        let (lock, cvar) = &*gate;
        let mut done = lock.lock();

        // Classic condvar pattern: check predicate before waiting to handle
        // signals that arrived between the pending-lock release and this point.
        loop {
            if *done {
                return Ok(Some(self.collect_by_key(&key)?));
            }
            match timeout {
                None => cvar.wait(&mut done),
                Some(t) => {
                    if cvar.wait_for(&mut done, t).timed_out() {
                        return Ok(None);
                    }
                }
            }
        }
    }

    /// Non-blocking completion check.  Returns `Some` if the link is already
    /// complete, `None` if still waiting for kinds.
    pub fn try_get(&self, id: &Value) -> Result<Option<LinkedDoc>, DbError> {
        let key = value_to_key(id);
        let guard = self.pending.read();
        if let Some(link) = guard.get(&key) {
            if link.arrived == self.complete_mask {
                drop(guard);
                return Ok(Some(self.collect_by_key(&key)?));
            }
        }
        Ok(None)
    }

    /// Which kinds have arrived so far for `id` (returns names of arrived kinds).
    pub fn arrived_kinds(&self, id: &Value) -> Vec<String> {
        let key = value_to_key(id);
        let guard = self.pending.read();
        match guard.get(&key) {
            None => vec![],
            Some(link) => self
                .kinds
                .iter()
                .enumerate()
                .filter(|(i, _)| link.arrived & (1u64 << i) != 0)
                .map(|(_, (name, _))| name.as_ref().to_string())
                .collect(),
        }
    }

    // ── Query (blocking + jetro expression) ───────────────────────────────────

    /// Block until the link is complete, then run `expr` against the merged
    /// document `{"kind1": doc1, "kind2": doc2, …}`.
    ///
    /// # Example expressions
    ///
    /// ```text
    /// $.order.total                                    — field from one kind
    /// $.payment.status                                 — field from another kind
    /// {total: $.order.total, paid: $.payment.amount}   — cross-kind object
    /// $..total.sum()                                   — aggregate
    /// ```
    pub fn query(&self, id: &Value, expr: &str) -> Result<Value, DbError> {
        Ok(self.query_timeout(id, expr, None)?.unwrap())
    }

    /// Block (with optional timeout) then run `expr`.  Returns `None` on timeout.
    pub fn query_timeout(
        &self,
        id: &Value,
        expr: &str,
        timeout: Option<Duration>,
    ) -> Result<Option<Value>, DbError> {
        let linked = match self.get_timeout(id, timeout)? {
            Some(d) => d,
            None => return Ok(None),
        };
        let root = linked.into_json();
        let mut vm = VM::new();
        let result = vm.run_str(expr, &root)
            .map_err(|e| DbError::EvalError(e.to_string()))?;
        Ok(Some(result))
    }

    // ── Internal ──────────────────────────────────────────────────────────────

    fn collect_by_key(&self, key: &str) -> Result<LinkedDoc, DbError> {
        let mut docs = HashMap::with_capacity(self.kinds.len());
        for (kind_name, kind_def) in &self.kinds {
            let bytes = kind_def
                .docs
                .get(key.as_bytes())
                .map_err(DbError::Io)?
                .ok_or_else(|| DbError::EvalError(format!(
                    "link complete but kind '{}' missing from BTree (id='{key}')",
                    kind_name
                )))?;
            let doc: Value = serde_json::from_slice(&bytes)
                .map_err(|e| DbError::Serialize(e.to_string()))?;
            docs.insert(kind_name.as_ref().to_string(), doc);
        }
        Ok(LinkedDoc { docs })
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Canonical string representation of a JSON value used as HashMap key.
///
/// Strings are used as-is (no quotes), numbers/bools via Display, complex
/// types via JSON serialisation.  This keeps numeric ids like `42` and string
/// ids like `"42"` separate since they render differently.
fn value_to_key(v: &Value) -> String {
    match v {
        Value::String(s)  => s.clone(),
        Value::Number(n)  => n.to_string(),
        Value::Bool(b)    => (if *b { "true" } else { "false" }).to_string(),
        Value::Null       => "null".to_string(),
        other             => serde_json::to_string(other).unwrap(),
    }
}
