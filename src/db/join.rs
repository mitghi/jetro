//! Stream-join combiner: block until one document of every configured
//! kind has arrived with the same id value.
//!
//! A [`Join`] accepts N kinds of JSON messages.  Each kind has a
//! designated *id field*.  A join is complete when one document of
//! every kind has arrived carrying the **same id value**.
//!
//! Reader APIs make blocking vs non-blocking semantics explicit in
//! the method name:
//!
//! | Method            | Blocks? | On timeout |
//! |-------------------|---------|------------|
//! | [`Join::wait`]    | yes     | forever    |
//! | [`Join::wait_for`]| yes     | returns `None` |
//! | [`Join::peek`]    | no      | `None` if incomplete |
//! | [`Join::on`]      | yes, when `.run` or `.get` is called | — |
//!
//! Multiple threads can wait on the same id concurrently — all wake
//! when the last missing kind arrives.
//!
//! # Example
//!
//! ```rust,no_run
//! use jetro::prelude::*;
//! use std::thread;
//!
//! let db = Database::memory()?;
//!
//! // Three kinds; all share `order_id` as the join key.
//! let orders = db.join("orders")
//!     .id("order_id")
//!     .kinds(["order", "payment", "shipment"])
//!     .open()?;
//!
//! let o2 = orders.clone();
//! thread::spawn(move || {
//!     o2.emit("payment",  &json!({"order_id": "X1", "amount": 99.0})).unwrap();
//!     o2.emit("shipment", &json!({"order_id": "X1", "tracking": "TRK42"})).unwrap();
//! });
//!
//! orders.emit("order", &json!({"order_id": "X1", "item": "Gadget"}))?;
//!
//! // Blocks until all three kinds for "X1" have arrived.
//! let joined = orders.wait("X1")?;
//! println!("{}", joined.docs["order"]["item"]);
//!
//! // Or query directly:
//! let total: f64 = orders.on("X1").get("$.payment.amount")?;
//! # Ok::<(), jetro::Error>(())
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::{Condvar, Mutex, RwLock};
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::VM;
use super::btree::BTree;
use super::error::DbError;
use super::storage::{FileStorage, Storage, Tree};

// ── JoinId ────────────────────────────────────────────────────────────────────

/// Anything convertible to a canonical join-id string.
///
/// Implemented for the common cases so callers can pass plain values
/// without wrapping in `json!(...)`.
///
/// ```ignore
/// join.wait("X1")?;         // &str
/// join.wait(42)?;           // i64
/// join.wait("X1".to_string())?;
/// join.wait(&value)?;       // &Value
/// ```
pub trait IntoJoinId {
    fn into_join_id(self) -> String;
}

impl IntoJoinId for &str      { fn into_join_id(self) -> String { self.to_string() } }
impl IntoJoinId for String    { fn into_join_id(self) -> String { self } }
impl IntoJoinId for &String   { fn into_join_id(self) -> String { self.clone() } }
impl IntoJoinId for i64       { fn into_join_id(self) -> String { self.to_string() } }
impl IntoJoinId for i32       { fn into_join_id(self) -> String { self.to_string() } }
impl IntoJoinId for u64       { fn into_join_id(self) -> String { self.to_string() } }
impl IntoJoinId for u32       { fn into_join_id(self) -> String { self.to_string() } }
impl IntoJoinId for bool      { fn into_join_id(self) -> String { self.to_string() } }
impl IntoJoinId for &Value    { fn into_join_id(self) -> String { value_to_key(self) } }
impl IntoJoinId for Value     { fn into_join_id(self) -> String { value_to_key(&self) } }

// ── Types ─────────────────────────────────────────────────────────────────────

/// A completed join: one document per kind, keyed by kind name.
pub struct JoinedDoc {
    /// Kind-name → JSON document.
    pub docs: HashMap<String, Value>,
}

impl JoinedDoc {
    /// Merge all kind documents into a single JSON object
    /// `{"kind": doc, …}` suitable for evaluating Jetro expressions.
    pub fn into_json(self) -> Value {
        Value::Object(self.docs.into_iter().collect())
    }
}

// ── Internal ──────────────────────────────────────────────────────────────────

struct KindDef {
    id_field: Arc<str>,
    docs:     Arc<dyn Tree>,
}

/// Per-id state tracking which kinds have arrived.
struct PendingJoin {
    /// Bitmask: bit `i` is set when kind `i` has been emitted.
    arrived: u64,
    /// Condvar signalled (notify_all) when `arrived == complete_mask`.
    /// The `bool` inside the Mutex is `true` once complete, preventing missed
    /// wakeups regardless of ordering between emit and wait.
    gate: Arc<(Mutex<bool>, Condvar)>,
}

// ── JoinBuilder ───────────────────────────────────────────────────────────────

/// Fluent builder returned by [`crate::db::Database::join`] or
/// [`Join::builder`].
///
/// Configure the set of kinds and their id fields, then call
/// [`open`](Self::open).
pub struct JoinBuilder {
    storage:    Arc<dyn Storage>,
    name:       String,
    default_id: Option<String>,
    kinds:      Vec<(String, Option<String>)>,
}

impl JoinBuilder {
    /// Build against an existing [`Storage`] backend.
    pub fn new(storage: Arc<dyn Storage>, name: impl Into<String>) -> Self {
        Self { storage, name: name.into(), default_id: None, kinds: Vec::new() }
    }

    /// File-backed convenience — wraps the directory in a [`FileStorage`].
    pub(super) fn from_dir(dir: &Path, name: impl Into<String>) -> Result<Self, DbError> {
        let fs = FileStorage::new(dir).map_err(DbError::Io)?;
        Ok(Self::new(Arc::new(fs), name))
    }

    /// Set the default id field used for kinds declared via
    /// [`kinds`](Self::kinds).  Per-kind overrides via [`kind`](Self::kind)
    /// still apply.
    pub fn id(mut self, field: impl Into<String>) -> Self {
        self.default_id = Some(field.into());
        self
    }

    /// Declare one kind with an explicit id field.  Overrides the
    /// default id set by [`id`](Self::id) for this kind.
    pub fn kind(mut self, name: impl Into<String>, id_field: impl Into<String>) -> Self {
        self.kinds.push((name.into(), Some(id_field.into())));
        self
    }

    /// Declare several kinds that all share the default id field.
    pub fn kinds<I, S>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        for n in names { self.kinds.push((n.into(), None)); }
        self
    }

    /// Open (or create) the [`Join`].
    pub fn open(self) -> Result<Arc<Join>, DbError> {
        assert!(!self.kinds.is_empty(), "Join: at least one kind required");
        assert!(self.kinds.len() <= 64, "Join: maximum 64 kinds");

        let mut kind_list: Vec<(Arc<str>, KindDef)> = Vec::with_capacity(self.kinds.len());
        for (kind_name, id_override) in &self.kinds {
            let id_field = id_override
                .clone()
                .or_else(|| self.default_id.clone())
                .unwrap_or_else(|| panic!("Join: kind '{kind_name}' has no id field and no default was set via .id()"));
            let tree_name = format!("{name}.{kind_name}.join", name = self.name);
            let docs = self.storage.open_tree(&tree_name).map_err(DbError::Io)?;
            kind_list.push((
                Arc::from(kind_name.as_str()),
                KindDef { id_field: Arc::from(id_field.as_str()), docs },
            ));
        }

        let complete_mask = (1u64 << kind_list.len()) - 1;

        Ok(Arc::new(Join {
            kinds: kind_list,
            complete_mask,
            pending: RwLock::new(HashMap::new()),
            _dir: PathBuf::new(),
            _name: Arc::from(self.name.as_str()),
        }))
    }
}

impl Join {
    /// Build a [`Join`] against any [`Storage`] backend — in-memory or custom.
    pub fn builder(storage: Arc<dyn Storage>, name: impl Into<String>) -> JoinBuilder {
        JoinBuilder::new(storage, name)
    }
}

// ── Join ──────────────────────────────────────────────────────────────────────

/// Blocking stream-join: waits until all N kinds share the same id value.
pub struct Join {
    /// Ordered list of kinds (index = bit position in `arrived` mask).
    kinds: Vec<(Arc<str>, KindDef)>,
    /// Bitmask with all N low bits set; completion is `arrived == complete_mask`.
    complete_mask: u64,
    /// In-memory tracking state, keyed by canonical id string.
    pending: RwLock<HashMap<String, PendingJoin>>,
    _dir:   PathBuf,
    _name:  Arc<str>,
}

impl Join {
    // ── Write ─────────────────────────────────────────────────────────────────

    /// Emit a document for `kind`.
    ///
    /// - **Update**: if this kind has already arrived for the given id, the
    ///   stored document is overwritten.  Completion state is untouched.
    /// - **Insert**: if this kind has not yet arrived, the document is stored,
    ///   its bit is set, and all blocked [`wait`](Self::wait) / [`on`](Self::on)
    ///   callers are unblocked if the join is now complete.
    pub fn emit(&self, kind: &str, doc: &Value) -> Result<(), DbError> {
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

        // Determine whether this is an update (bit already set) or a fresh emit.
        let already_present = {
            let guard = self.pending.read();
            guard.get(&key).map_or(false, |l| l.arrived & bit != 0)
        };

        // Persist to BTree first so the doc is readable before any condvar fires.
        let bytes = serde_json::to_vec(doc).map_err(|e| DbError::Serialize(e.to_string()))?;
        kind_def.docs.insert(key.as_bytes(), &bytes).map_err(DbError::Io)?;

        if already_present {
            return Ok(());
        }

        // Insert path: set the bit, signal if now complete.
        let gate_to_signal: Option<Arc<(Mutex<bool>, Condvar)>> = {
            let mut guard = self.pending.write();
            let link = guard
                .entry(key.clone())
                .or_insert_with(|| PendingJoin {
                    arrived: 0,
                    gate: Arc::new((Mutex::new(false), Condvar::new())),
                });
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

        if let Some(gate) = gate_to_signal {
            let (lock, cvar) = &*gate;
            *lock.lock() = true;
            cvar.notify_all();
        }

        Ok(())
    }

    /// Remove all stored documents for `id` and drop its pending
    /// state.  Does **not** wake blocked waiters.
    pub fn remove(&self, id: impl IntoJoinId) -> Result<(), DbError> {
        let key = id.into_join_id();
        for (_, kind_def) in &self.kinds {
            kind_def.docs.delete(key.as_bytes()).map_err(DbError::Io)?;
        }
        self.pending.write().remove(&key);
        Ok(())
    }

    /// Remove one kind's contribution, breaking the join if it was complete.
    ///
    /// If the join was complete before this call, the gate is replaced with a
    /// fresh one so subsequent waiters block again instead of seeing stale
    /// `done=true`.
    pub fn unemit(&self, kind: &str, id: impl IntoJoinId) -> Result<bool, DbError> {
        let (kind_idx, kind_def) = self
            .kinds
            .iter()
            .enumerate()
            .find(|(_, (n, _))| n.as_ref() == kind)
            .map(|(i, (_, def))| (i, def))
            .ok_or_else(|| DbError::EvalError(format!("unknown kind '{kind}'")))?;

        let key = id.into_join_id();
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
                link.gate = Arc::new((Mutex::new(false), Condvar::new()));
            }
        }
        Ok(true)
    }

    // ── Read ──────────────────────────────────────────────────────────────────

    /// Block until the join is complete, then return every document.
    ///
    /// Waits indefinitely.  Use [`wait_for`](Self::wait_for) for bounded waits
    /// or [`peek`](Self::peek) for a non-blocking probe.
    pub fn wait(&self, id: impl IntoJoinId) -> Result<JoinedDoc, DbError> {
        Ok(self.wait_timeout_raw(id.into_join_id(), None)?.unwrap())
    }

    /// Block until the join is complete or `timeout` elapses.
    /// Returns `None` on timeout.
    pub fn wait_for(
        &self,
        id: impl IntoJoinId,
        timeout: Duration,
    ) -> Result<Option<JoinedDoc>, DbError> {
        self.wait_timeout_raw(id.into_join_id(), Some(timeout))
    }

    /// Non-blocking probe: `Some` if the join is already complete,
    /// `None` otherwise.
    pub fn peek(&self, id: impl IntoJoinId) -> Result<Option<JoinedDoc>, DbError> {
        let key = id.into_join_id();
        let guard = self.pending.read();
        if let Some(link) = guard.get(&key) {
            if link.arrived == self.complete_mask {
                drop(guard);
                return Ok(Some(self.collect_by_key(&key)?));
            }
        }
        Ok(None)
    }

    /// Kinds that have arrived so far for `id`.
    pub fn arrived(&self, id: impl IntoJoinId) -> Vec<String> {
        let key = id.into_join_id();
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

    /// All kinds that have been emitted for `id`, without blocking.
    /// Useful for event-sourced reads on in-progress joins.
    pub fn partial(&self, id: impl IntoJoinId) -> Result<HashMap<String, Value>, DbError> {
        let key = id.into_join_id();
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

    // ── Query ─────────────────────────────────────────────────────────────────

    /// Begin a query chain for `id`.  Calling `.run(expr)` or
    /// `.get::<T>(expr)` on the returned [`JoinQuery`] blocks until
    /// the join is complete, then evaluates the expression against
    /// `{"kind": doc, …}`.
    pub fn on(self: &Arc<Self>, id: impl IntoJoinId) -> JoinQuery {
        JoinQuery { join: Arc::clone(self), id: id.into_join_id(), timeout: None }
    }

    // ── Internal ──────────────────────────────────────────────────────────────

    fn wait_timeout_raw(
        &self,
        key: String,
        timeout: Option<Duration>,
    ) -> Result<Option<JoinedDoc>, DbError> {
        // Fast path: already complete.
        {
            let guard = self.pending.read();
            if let Some(link) = guard.get(&key) {
                if link.arrived == self.complete_mask {
                    return Ok(Some(self.collect_by_key(&key)?));
                }
            }
        }

        // Slow path: obtain or create gate, then wait.
        let gate: Arc<(Mutex<bool>, Condvar)> = {
            let mut guard = self.pending.write();
            let link = guard
                .entry(key.clone())
                .or_insert_with(|| PendingJoin {
                    arrived: 0,
                    gate: Arc::new((Mutex::new(false), Condvar::new())),
                });
            if link.arrived == self.complete_mask {
                return Ok(Some(self.collect_by_key(&key)?));
            }
            Arc::clone(&link.gate)
        };

        let (lock, cvar) = &*gate;
        let mut done = lock.lock();

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

    fn collect_by_key(&self, key: &str) -> Result<JoinedDoc, DbError> {
        let mut docs = HashMap::with_capacity(self.kinds.len());
        for (kind_name, kind_def) in &self.kinds {
            let bytes = kind_def
                .docs
                .get(key.as_bytes())
                .map_err(DbError::Io)?
                .ok_or_else(|| DbError::EvalError(format!(
                    "join complete but kind '{}' missing from BTree (id='{key}')",
                    kind_name
                )))?;
            let doc: Value = serde_json::from_slice(&bytes)
                .map_err(|e| DbError::Serialize(e.to_string()))?;
            docs.insert(kind_name.as_ref().to_string(), doc);
        }
        Ok(JoinedDoc { docs })
    }
}

// ── JoinQuery ─────────────────────────────────────────────────────────────────

/// Query handle returned by [`Join::on`].
///
/// Blocks until the join completes, then evaluates a Jetro expression
/// against the merged `{"kind": doc, …}` root.
pub struct JoinQuery {
    join:    Arc<Join>,
    id:      String,
    timeout: Option<Duration>,
}

impl JoinQuery {
    /// Apply a timeout to the blocking wait.  Without this, the
    /// query blocks indefinitely.
    pub fn timeout(mut self, t: Duration) -> Self {
        self.timeout = Some(t);
        self
    }

    /// Evaluate `expr` after the join completes, returning the raw [`Value`].
    pub fn run(&self, expr: &str) -> Result<Value, DbError> {
        Ok(self.run_opt(expr)?.unwrap())
    }

    /// Evaluate `expr` with the configured timeout.  Returns `None`
    /// if the timeout elapses before the join completes.
    pub fn run_opt(&self, expr: &str) -> Result<Option<Value>, DbError> {
        let joined = match self.join.wait_timeout_raw(self.id.clone(), self.timeout)? {
            Some(d) => d,
            None => return Ok(None),
        };
        let root = joined.into_json();
        let mut vm = VM::new();
        let result = vm.run_str(expr, &root)
            .map_err(|e| DbError::EvalError(e.to_string()))?;
        Ok(Some(result))
    }

    /// Evaluate `expr`, deserialising the result into `T`.
    pub fn get<T: DeserializeOwned>(&self, expr: &str) -> Result<T, DbError> {
        let v = self.run(expr)?;
        serde_json::from_value(v).map_err(|e| DbError::Serialize(e.to_string()))
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Canonical string representation of a JSON value used as HashMap key.
///
/// Strings are used as-is (no quotes), numbers/bools via Display, complex
/// types via JSON serialisation.  Keeps numeric ids like `42` and string
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
