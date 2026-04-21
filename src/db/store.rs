//! Generic key/value abstractions and extension traits.
//!
//! Three ideas live here:
//!
//! 1. [`Store`] — one trait, four methods, common to every bucket and
//!    the raw [`BTree`].  Lets callers write generic helpers
//!    (replication, scanning, bulk import) that don't care about
//!    a specific storage flavour.
//! 2. [`BTreeBulk`] / [`BTreePrefix`] — extension traits that layer
//!    higher-level queries on top of [`BTree`].  The core type stays
//!    minimal; callers opt in to richer APIs via `use`.
//! 3. [`DocIter`] — chainable combinators over `Vec<(String, Value)>`
//!    so `bucket.iter_docs()? | .filter_expr(...) | .map_expr(...)`
//!    reads like stdlib iterators.

use std::io;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::{Error, VM};
use super::btree::BTree;
use super::bucket::{ExprBucket, JsonBucket};
use super::error::DbError;

// ── Store ─────────────────────────────────────────────────────────────────────

/// Unified key/value interface.
///
/// Implemented by every storage flavour — raw BTree, expression
/// bucket, document bucket.  Write cross-bucket helpers against this
/// trait instead of hard-coding concrete types.
pub trait Store {
    /// Borrowed key type.  `?Sized` so implementors can use
    /// `str`, `[u8]`, etc.
    type Key:      ?Sized;
    /// Owned counterpart used when returning keys from iterators.
    type KeyOwned;
    /// Stored value.
    type Item;
    /// Error type returned by every operation.
    type Error;

    fn insert(&self, key: &Self::Key, val: &Self::Item) -> Result<(), Self::Error>;
    fn get(&self,    key: &Self::Key)                    -> Result<Option<Self::Item>, Self::Error>;
    fn delete(&self, key: &Self::Key)                    -> Result<bool, Self::Error>;
    fn iter(&self)                                       -> Result<Vec<(Self::KeyOwned, Self::Item)>, Self::Error>;
}

// ── BTree impl ────────────────────────────────────────────────────────────────

impl Store for BTree {
    type Key      = [u8];
    type KeyOwned = Vec<u8>;
    type Item     = Vec<u8>;
    type Error    = io::Error;

    fn insert(&self, key: &[u8], val: &Vec<u8>) -> io::Result<()> {
        BTree::insert(self, key, val)
    }
    fn get(&self, key: &[u8]) -> io::Result<Option<Vec<u8>>> { BTree::get(self, key) }
    fn delete(&self, key: &[u8]) -> io::Result<bool> { BTree::delete(self, key) }
    fn iter(&self) -> io::Result<Vec<(Vec<u8>, Vec<u8>)>> { BTree::all(self) }
}

// ── ExprBucket impl ───────────────────────────────────────────────────────────

impl Store for ExprBucket {
    type Key      = str;
    type KeyOwned = String;
    type Item     = String;
    type Error    = DbError;

    fn insert(&self, key: &str, val: &String) -> Result<(), DbError> { self.put(key, val) }
    fn get(&self, key: &str) -> Result<Option<String>, DbError> { ExprBucket::get(self, key) }
    fn delete(&self, key: &str) -> Result<bool, DbError> { ExprBucket::delete(self, key) }
    fn iter(&self) -> Result<Vec<(String, String)>, DbError> { self.all() }
}

// ── JsonBucket impl (documents only; results go through get_result) ───────────

impl Store for JsonBucket {
    type Key      = str;
    type KeyOwned = String;
    type Item     = Value;
    type Error    = DbError;

    fn insert(&self, key: &str, val: &Value) -> Result<(), DbError> { JsonBucket::insert(self, key, val) }
    fn get(&self, key: &str) -> Result<Option<Value>, DbError> { self.get_doc(key) }
    fn delete(&self, key: &str) -> Result<bool, DbError> {
        let existed = self.get_doc(key)?.is_some();
        self.delete_doc(key)?;
        Ok(existed)
    }
    fn iter(&self) -> Result<Vec<(String, Value)>, DbError> { self.iter_docs() }
}

// ── BTree extension traits ────────────────────────────────────────────────────

/// Bulk write helpers on [`BTree`].
pub trait BTreeBulk {
    /// Insert every pair.  Returns the number of pairs written.
    /// Not atomic — a failure mid-batch leaves partial state.
    fn insert_many(&self, pairs: &[(Vec<u8>, Vec<u8>)]) -> io::Result<usize>;
}

impl BTreeBulk for BTree {
    fn insert_many(&self, pairs: &[(Vec<u8>, Vec<u8>)]) -> io::Result<usize> {
        for (k, v) in pairs {
            self.insert(k, v)?;
        }
        Ok(pairs.len())
    }
}

/// Prefix scan on [`BTree`].
pub trait BTreePrefix {
    /// Every `(key, val)` pair whose key starts with `prefix`.
    fn prefix_scan(&self, prefix: &[u8]) -> io::Result<Vec<(Vec<u8>, Vec<u8>)>>;
}

impl BTreePrefix for BTree {
    fn prefix_scan(&self, prefix: &[u8]) -> io::Result<Vec<(Vec<u8>, Vec<u8>)>> {
        // [prefix, prefix + 0x01) covers every key sharing this prefix.
        let mut end = prefix.to_vec();
        end.push(0x00);
        *end.last_mut().unwrap() = 0x01;
        let _ = end; // unused if prefix empty; fall through to range_from
        if prefix.is_empty() {
            return self.all();
        }
        // Build exclusive upper by incrementing the last byte; if that
        // overflows, fall back to range_from (unbounded).
        let mut upper = prefix.to_vec();
        let mut carry = true;
        for byte in upper.iter_mut().rev() {
            if *byte == 0xFF {
                *byte = 0;
            } else {
                *byte += 1;
                carry = false;
                break;
            }
        }
        if carry {
            self.range_from(prefix)
        } else {
            self.range(prefix, &upper)
        }
    }
}

// ── DocIter combinators ───────────────────────────────────────────────────────

/// Chainable combinators over `Vec<(String, Value)>` — the return
/// type of [`JsonBucket::iter_docs`] / [`Bucket::iter_docs`].
///
/// ```rust,no_run
/// use jetro::prelude::*;
/// use jetro::db::DocIter;
///
/// let db = Database::memory()?;
/// let b = db.bucket("x").open()?;
/// // ... insert ...
/// let titles: Vec<String> = b.iter_docs()?
///     .filter_expr("$.price > 10")?
///     .map_expr_as("$.title")?;
/// # Ok::<(), jetro::Error>(())
/// ```
pub trait DocIter: Sized {
    fn filter_expr(self, expr: &str) -> Result<Vec<(String, Value)>, Error>;
    fn map_expr(self, expr: &str) -> Result<Vec<Value>, Error>;
    fn map_expr_as<T: DeserializeOwned>(self, expr: &str) -> Result<Vec<T>, Error>;
}

impl DocIter for Vec<(String, Value)> {
    fn filter_expr(self, expr: &str) -> Result<Vec<(String, Value)>, Error> {
        let mut vm = VM::new();
        let mut out = Vec::new();
        for (k, v) in self {
            let r = vm.run_str(expr, &v)?;
            if truthy(&r) {
                out.push((k, v));
            }
        }
        Ok(out)
    }

    fn map_expr(self, expr: &str) -> Result<Vec<Value>, Error> {
        let mut vm = VM::new();
        self.into_iter()
            .map(|(_, v)| vm.run_str(expr, &v).map_err(Error::from))
            .collect()
    }

    fn map_expr_as<T: DeserializeOwned>(self, expr: &str) -> Result<Vec<T>, Error> {
        let mut vm = VM::new();
        self.into_iter()
            .map(|(_, v)| {
                let raw = vm.run_str(expr, &v)?;
                serde_json::from_value(raw)
                    .map_err(|e| Error::Eval(crate::EvalError(e.to_string())))
            })
            .collect()
    }
}

fn truthy(v: &Value) -> bool {
    match v {
        Value::Bool(b)    => *b,
        Value::Null       => false,
        Value::Number(n)  => n.as_f64().map(|f| f != 0.0).unwrap_or(false),
        Value::String(s)  => !s.is_empty(),
        Value::Array(a)   => !a.is_empty(),
        Value::Object(o)  => !o.is_empty(),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Database;
    use serde_json::json;

    #[test]
    fn btree_store_impl() {
        let db = Database::memory().unwrap();
        let tree = BTree::open(db.path().join("raw")).unwrap();
        Store::insert(&*tree, b"a", &b"1".to_vec()).unwrap();
        Store::insert(&*tree, b"b", &b"2".to_vec()).unwrap();
        let v = Store::get(&*tree, b"a").unwrap().unwrap();
        assert_eq!(v, b"1");
        let all = Store::iter(&*tree).unwrap();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn btree_bulk_insert() {
        let db = Database::memory().unwrap();
        let tree = BTree::open(db.path().join("bulk")).unwrap();
        let pairs = vec![
            (b"a".to_vec(), b"1".to_vec()),
            (b"b".to_vec(), b"2".to_vec()),
            (b"c".to_vec(), b"3".to_vec()),
        ];
        let n = tree.insert_many(&pairs).unwrap();
        assert_eq!(n, 3);
        assert_eq!(tree.all().unwrap().len(), 3);
    }

    #[test]
    fn btree_prefix_scan() {
        let db = Database::memory().unwrap();
        let tree = BTree::open(db.path().join("pfx")).unwrap();
        tree.insert(b"user:1", b"a").unwrap();
        tree.insert(b"user:2", b"b").unwrap();
        tree.insert(b"post:1", b"c").unwrap();
        let users = tree.prefix_scan(b"user:").unwrap();
        assert_eq!(users.len(), 2);
    }

    #[test]
    fn expr_bucket_store_impl() {
        let db = Database::memory().unwrap();
        let ex = db.expr_bucket("e").unwrap();
        Store::insert(&*ex, "k", &"$.x".to_string()).unwrap();
        let v = Store::get(&*ex, "k").unwrap().unwrap();
        assert_eq!(v, "$.x");
    }

    #[test]
    fn doc_iter_filter_map() {
        let db = Database::memory().unwrap();
        let b = db.bucket("books").with("k", "$").open().unwrap();
        b.insert("d1", &json!({"title": "Dune", "price": 12})).unwrap();
        b.insert("d2", &json!({"title": "Foundation", "price": 5})).unwrap();
        b.insert("d3", &json!({"title": "Neuromancer", "price": 20})).unwrap();

        let titles: Vec<String> = b.iter_docs().unwrap()
            .filter_expr("$.price > 10").unwrap()
            .map_expr_as("$.title").unwrap();
        let mut sorted = titles;
        sorted.sort();
        assert_eq!(sorted, vec!["Dune", "Neuromancer"]);
    }
}
