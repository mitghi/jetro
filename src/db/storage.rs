//! Storage-engine abstraction.
//!
//! Two traits carve out a replaceable backend:
//!
//! - [`Tree`] — byte-oriented ordered KV surface shared by the on-disk
//!   [`BTree`] and any future in-memory replacement.
//! - [`Storage`] — opens named [`Tree`] handles rooted in some location
//!   (a directory, an in-process map, etc.).
//!
//! Keeping this narrow on purpose: the operations listed here are the
//! exact set every bucket and higher-level helper relies on, so any
//! conforming backend is a drop-in replacement.

use std::collections::{BTreeMap, HashMap};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::{Mutex, RwLock};

use super::btree::BTree;

// ── Tree ──────────────────────────────────────────────────────────────────────

/// Byte-oriented ordered KV store.
///
/// Implemented by [`BTree`] today; future in-memory backends plug in by
/// implementing this same surface.
pub trait Tree: Send + Sync {
    fn insert(&self, key: &[u8], val: &[u8]) -> io::Result<()>;
    fn get(&self, key: &[u8]) -> io::Result<Option<Vec<u8>>>;
    fn delete(&self, key: &[u8]) -> io::Result<bool>;
    fn all(&self) -> io::Result<Vec<(Vec<u8>, Vec<u8>)>>;
    fn range(&self, start: &[u8], end: &[u8]) -> io::Result<Vec<(Vec<u8>, Vec<u8>)>>;
    fn range_from(&self, start: &[u8]) -> io::Result<Vec<(Vec<u8>, Vec<u8>)>>;
}

impl Tree for BTree {
    fn insert(&self, key: &[u8], val: &[u8]) -> io::Result<()> {
        BTree::insert(self, key, val)
    }
    fn get(&self, key: &[u8]) -> io::Result<Option<Vec<u8>>> {
        BTree::get(self, key)
    }
    fn delete(&self, key: &[u8]) -> io::Result<bool> {
        BTree::delete(self, key)
    }
    fn all(&self) -> io::Result<Vec<(Vec<u8>, Vec<u8>)>> {
        BTree::all(self)
    }
    fn range(&self, start: &[u8], end: &[u8]) -> io::Result<Vec<(Vec<u8>, Vec<u8>)>> {
        BTree::range(self, start, end)
    }
    fn range_from(&self, start: &[u8]) -> io::Result<Vec<(Vec<u8>, Vec<u8>)>> {
        BTree::range_from(self, start)
    }
}

// ── Storage ───────────────────────────────────────────────────────────────────

/// Opens named [`Tree`] handles rooted in some backing location.
///
/// The returned handle is `Arc<dyn Tree>` so callers can share it across
/// threads without knowing the concrete backend.
pub trait Storage: Send + Sync {
    fn open_tree(&self, name: &str) -> io::Result<Arc<dyn Tree>>;
}

// ── FileStorage ───────────────────────────────────────────────────────────────

/// [`Storage`] backed by a directory on disk. Each `open_tree` call
/// returns a shared [`BTree`] handle for `<dir>/<name>`.
pub struct FileStorage {
    dir: PathBuf,
}

impl FileStorage {
    pub fn new(dir: impl AsRef<Path>) -> io::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;
        Ok(Self { dir })
    }

    pub fn path(&self) -> &Path {
        &self.dir
    }
}

impl Storage for FileStorage {
    fn open_tree(&self, name: &str) -> io::Result<Arc<dyn Tree>> {
        let path = self.dir.join(name);
        let tree = BTree::open(path)?;
        Ok(tree as Arc<dyn Tree>)
    }
}

// ── BTreeMem ──────────────────────────────────────────────────────────────────

/// In-memory [`Tree`] backed by a [`BTreeMap`].
///
/// Intended for tests and short-lived tools where durability isn't needed.
/// Persistence, mmap, and COW are absent — writes mutate the map directly
/// under a [`RwLock`]. Range semantics match the on-disk [`BTree`].
pub struct BTreeMem {
    inner: RwLock<BTreeMap<Vec<u8>, Vec<u8>>>,
}

impl BTreeMem {
    pub fn new() -> Arc<Self> {
        Arc::new(Self { inner: RwLock::new(BTreeMap::new()) })
    }
}

impl Default for BTreeMem {
    fn default() -> Self {
        Self { inner: RwLock::new(BTreeMap::new()) }
    }
}

impl Tree for BTreeMem {
    fn insert(&self, key: &[u8], val: &[u8]) -> io::Result<()> {
        assert!(key.len() <= 255, "key exceeds 255 bytes");
        self.inner.write().insert(key.to_vec(), val.to_vec());
        Ok(())
    }

    fn get(&self, key: &[u8]) -> io::Result<Option<Vec<u8>>> {
        Ok(self.inner.read().get(key).cloned())
    }

    fn delete(&self, key: &[u8]) -> io::Result<bool> {
        Ok(self.inner.write().remove(key).is_some())
    }

    fn all(&self) -> io::Result<Vec<(Vec<u8>, Vec<u8>)>> {
        Ok(self.inner.read().iter().map(|(k, v)| (k.clone(), v.clone())).collect())
    }

    fn range(&self, start: &[u8], end: &[u8]) -> io::Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let guard = self.inner.read();
        Ok(guard
            .range::<[u8], _>((
                std::ops::Bound::Included(start),
                std::ops::Bound::Excluded(end),
            ))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect())
    }

    fn range_from(&self, start: &[u8]) -> io::Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let guard = self.inner.read();
        Ok(guard
            .range::<[u8], _>((std::ops::Bound::Included(start), std::ops::Bound::Unbounded))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect())
    }
}

// ── MemStorage ────────────────────────────────────────────────────────────────

/// [`Storage`] backed by an in-process map of name → [`BTreeMem`].
///
/// Opening the same name twice returns the same underlying handle.
pub struct MemStorage {
    trees: Mutex<HashMap<String, Arc<BTreeMem>>>,
}

impl MemStorage {
    pub fn new() -> Self {
        Self { trees: Mutex::new(HashMap::new()) }
    }
}

impl Default for MemStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl Storage for MemStorage {
    fn open_tree(&self, name: &str) -> io::Result<Arc<dyn Tree>> {
        let mut g = self.trees.lock();
        let tree = g.entry(name.to_string()).or_insert_with(BTreeMem::new).clone();
        Ok(tree as Arc<dyn Tree>)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Database;

    #[test]
    fn btree_impls_tree() {
        let db = Database::memory().unwrap();
        let tree = BTree::open(db.path().join("raw")).unwrap();
        let t: Arc<dyn Tree> = tree;
        t.insert(b"a", b"1").unwrap();
        t.insert(b"b", b"2").unwrap();
        assert_eq!(t.get(b"a").unwrap().unwrap(), b"1");
        assert_eq!(t.all().unwrap().len(), 2);
        assert!(t.delete(b"a").unwrap());
        assert_eq!(t.all().unwrap().len(), 1);
    }

    #[test]
    fn file_storage_opens_named_trees() {
        let db = Database::memory().unwrap();
        let st = FileStorage::new(db.path()).unwrap();
        let t1 = st.open_tree("t1").unwrap();
        let t2 = st.open_tree("t2").unwrap();
        t1.insert(b"k", b"v1").unwrap();
        t2.insert(b"k", b"v2").unwrap();
        assert_eq!(t1.get(b"k").unwrap().unwrap(), b"v1");
        assert_eq!(t2.get(b"k").unwrap().unwrap(), b"v2");
    }

    #[test]
    fn mem_storage_opens_named_trees() {
        let st = MemStorage::new();
        let t1 = st.open_tree("t1").unwrap();
        let t2 = st.open_tree("t2").unwrap();
        t1.insert(b"k", b"v1").unwrap();
        t2.insert(b"k", b"v2").unwrap();
        assert_eq!(t1.get(b"k").unwrap().unwrap(), b"v1");
        assert_eq!(t2.get(b"k").unwrap().unwrap(), b"v2");
        // same name → same handle
        let t1b = st.open_tree("t1").unwrap();
        assert_eq!(t1b.get(b"k").unwrap().unwrap(), b"v1");
    }

    #[test]
    fn btreemem_range_and_delete() {
        let t = BTreeMem::new();
        t.insert(b"a", b"1").unwrap();
        t.insert(b"b", b"2").unwrap();
        t.insert(b"c", b"3").unwrap();
        assert_eq!(t.range(b"a", b"c").unwrap().len(), 2);
        assert_eq!(t.range_from(b"b").unwrap().len(), 2);
        assert!(t.delete(b"a").unwrap());
        assert!(!t.delete(b"a").unwrap());
        assert_eq!(t.all().unwrap().len(), 2);
    }

    #[test]
    fn file_storage_range_scans() {
        let db = Database::memory().unwrap();
        let st = FileStorage::new(db.path()).unwrap();
        let t = st.open_tree("scan").unwrap();
        t.insert(b"a", b"1").unwrap();
        t.insert(b"b", b"2").unwrap();
        t.insert(b"c", b"3").unwrap();
        let r = t.range(b"a", b"c").unwrap();
        assert_eq!(r.len(), 2);
        let rf = t.range_from(b"b").unwrap();
        assert_eq!(rf.len(), 2);
    }

    // ── MemStorage-backed buckets — proves Storage wiring ─────────────────────

    #[test]
    fn expr_bucket_on_memstorage() {
        use crate::db::ExprBucket;
        let st: Arc<dyn Storage> = Arc::new(MemStorage::new());
        let b = ExprBucket::from_storage(st, "exprs").unwrap();
        b.put("sum_x", "$.sum(x)").unwrap();
        assert_eq!(b.get("sum_x").unwrap().as_deref(), Some("$.sum(x)"));
        assert_eq!(b.all().unwrap().len(), 1);
        assert!(b.delete("sum_x").unwrap());
        assert_eq!(b.get("sum_x").unwrap(), None);
    }

    #[test]
    fn json_bucket_on_memstorage() {
        use crate::db::{ExprBucket, JsonBucket};
        use serde_json::json;
        let st: Arc<dyn Storage> = Arc::new(MemStorage::new());
        let exprs = ExprBucket::from_storage(st.clone(), "exprs").unwrap();
        exprs.put("total", "$.items.sum(n)").unwrap();
        let jb = JsonBucket::from_storage(st, "docs", vec!["total".into()], exprs).unwrap();
        jb.insert("d1", &json!({"items": [{"n": 1}, {"n": 2}, {"n": 3}]})).unwrap();
        assert_eq!(jb.get_result("d1", "total").unwrap(), Some(json!(6)));
    }

    #[test]
    fn graph_bucket_on_memstorage() {
        use crate::db::{ExprBucket, GraphBucket, GraphNode};
        use serde_json::json;
        let st: Arc<dyn Storage> = Arc::new(MemStorage::new());
        let exprs = ExprBucket::from_storage(st.clone(), "exprs").unwrap();
        let gb = GraphBucket::from_storage(st, "g", exprs).unwrap();
        gb.add_node("users").unwrap();
        gb.insert("users", "u1", &json!({"id": "u1", "name": "Alice"})).unwrap();
        gb.insert("users", "u2", &json!({"id": "u2", "name": "Bob"})).unwrap();
        let r = gb.query(&[GraphNode::All { node: "users" }], "$.users.len()").unwrap();
        assert_eq!(r, json!(2));
    }

    #[test]
    fn join_on_memstorage() {
        use crate::db::Join;
        use serde_json::json;
        let st: Arc<dyn Storage> = Arc::new(MemStorage::new());
        let join = Join::builder(st, "j")
            .id("id")
            .kinds(["orders", "customers"])
            .open()
            .unwrap();
        join.emit("orders",    &json!({"id": "x1", "total": 99})).unwrap();
        join.emit("customers", &json!({"id": "x1", "name": "Ada"})).unwrap();
        let jd = join.peek("x1").unwrap().unwrap();
        let merged = jd.into_json();
        assert_eq!(merged["orders"]["total"],   json!(99));
        assert_eq!(merged["customers"]["name"], json!("Ada"));
    }
}
