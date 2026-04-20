#[cfg(test)]
mod tests {
    use serde_json::json;
    use tempfile::TempDir;
    use crate::db::Database;
    use crate::db::btree::BTree;

    // ── B-Tree unit tests ─────────────────────────────────────────────────────

    fn tmp_tree() -> (TempDir, std::sync::Arc<BTree>) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.db");
        let tree = BTree::open(&path).unwrap();
        (dir, tree)
    }

    #[test]
    fn btree_insert_get() {
        let (_dir, tree) = tmp_tree();
        tree.insert(b"hello", b"world").unwrap();
        assert_eq!(tree.get(b"hello").unwrap().as_deref(), Some(b"world".as_ref()));
        assert_eq!(tree.get(b"missing").unwrap(), None);
    }

    #[test]
    fn btree_update() {
        let (_dir, tree) = tmp_tree();
        tree.insert(b"key", b"v1").unwrap();
        tree.insert(b"key", b"v2").unwrap();
        assert_eq!(tree.get(b"key").unwrap().as_deref(), Some(b"v2".as_ref()));
    }

    #[test]
    fn btree_delete() {
        let (_dir, tree) = tmp_tree();
        tree.insert(b"a", b"1").unwrap();
        assert!(tree.delete(b"a").unwrap());
        assert!(!tree.delete(b"a").unwrap());
        assert_eq!(tree.get(b"a").unwrap(), None);
    }

    #[test]
    fn btree_sorted_order() {
        let (_dir, tree) = tmp_tree();
        let keys = ["banana", "apple", "cherry", "date", "avocado"];
        for k in &keys {
            tree.insert(k.as_bytes(), b"v").unwrap();
        }
        let all = tree.all().unwrap();
        let retrieved: Vec<&str> = all.iter()
            .map(|(k, _)| std::str::from_utf8(k).unwrap())
            .collect();
        let mut sorted = keys.to_vec();
        sorted.sort_unstable();
        assert_eq!(retrieved, sorted);
    }

    #[test]
    fn btree_many_inserts_force_split() {
        let (_dir, tree) = tmp_tree();
        for i in 0u32..200 {
            let key = format!("key{:05}", i);
            let val = format!("val{}", i);
            tree.insert(key.as_bytes(), val.as_bytes()).unwrap();
        }
        for i in 0u32..200 {
            let key = format!("key{:05}", i);
            let expected = format!("val{}", i);
            let got = tree.get(key.as_bytes()).unwrap().unwrap();
            assert_eq!(String::from_utf8(got).unwrap(), expected, "key={key}");
        }
    }

    #[test]
    fn btree_large_value_overflow() {
        let (_dir, tree) = tmp_tree();
        let big = vec![0xABu8; 8000]; // > MAX_INLINE_VAL
        tree.insert(b"bigkey", &big).unwrap();
        assert_eq!(tree.get(b"bigkey").unwrap().unwrap(), big);
    }

    #[test]
    fn btree_range_scan() {
        let (_dir, tree) = tmp_tree();
        for i in 0u32..10 {
            let k = format!("{:03}", i);
            tree.insert(k.as_bytes(), k.as_bytes()).unwrap();
        }
        let range = tree.range(b"003", b"007").unwrap();
        let keys: Vec<_> = range.iter().map(|(k, _)| std::str::from_utf8(k).unwrap().to_string()).collect();
        assert_eq!(keys, vec!["003", "004", "005", "006"]);
    }

    #[test]
    fn btree_persists_across_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("persist.db");
        {
            let tree = BTree::open(&path).unwrap();
            tree.insert(b"persistent", b"data").unwrap();
        }
        {
            let tree = BTree::open(&path).unwrap();
            assert_eq!(
                tree.get(b"persistent").unwrap().as_deref(),
                Some(b"data".as_ref())
            );
        }
    }

    // ── Database / bucket tests ───────────────────────────────────────────────

    fn tmp_db() -> (TempDir, Database) {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(dir.path()).unwrap();
        (dir, db)
    }

    #[test]
    fn expr_bucket_put_get() {
        let (_dir, db) = tmp_db();
        let exprs = db.expr_bucket("main").unwrap();
        exprs.put("q1", "$.users.len()").unwrap();
        assert_eq!(exprs.get("q1").unwrap().as_deref(), Some("$.users.len()"));
        assert_eq!(exprs.get("missing").unwrap(), None);
    }

    #[test]
    fn expr_bucket_rejects_invalid_expr() {
        let (_dir, db) = tmp_db();
        let exprs = db.expr_bucket("main").unwrap();
        assert!(exprs.put("bad", "this is not valid >>>").is_err());
    }

    #[test]
    fn json_bucket_insert_and_retrieve() {
        let (_dir, db) = tmp_db();
        let exprs = db.expr_bucket("exprs").unwrap();
        exprs.put("names", "$.users.map(name)").unwrap();
        exprs.put("count", "$.users.len()").unwrap();

        let docs = db.json_bucket("docs", &["names", "count"], &exprs).unwrap();
        let doc = json!({
            "users": [
                {"name": "Alice", "age": 30},
                {"name": "Bob",   "age": 25}
            ]
        });
        docs.insert("doc1", &doc).unwrap();

        let names = docs.get_result("doc1", "names").unwrap().unwrap();
        assert_eq!(names, json!(["Alice", "Bob"]));

        let count = docs.get_result("doc1", "count").unwrap().unwrap();
        assert_eq!(count, json!(2));

        let original = docs.get_doc("doc1").unwrap().unwrap();
        assert_eq!(original, doc);
    }

    #[test]
    fn json_bucket_update_replaces_results() {
        let (_dir, db) = tmp_db();
        let exprs = db.expr_bucket("exprs").unwrap();
        exprs.put("count", "$.items.len()").unwrap();

        let docs = db.json_bucket("docs", &["count"], &exprs).unwrap();
        docs.insert("d", &json!({"items": [1, 2, 3]})).unwrap();
        docs.update("d", &json!({"items": [1, 2, 3, 4, 5]})).unwrap();

        let count = docs.get_result("d", "count").unwrap().unwrap();
        assert_eq!(count, json!(5));
    }

    #[test]
    fn json_bucket_concurrent_reads() {
        use std::sync::Arc;
        use std::thread;

        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(dir.path()).unwrap();
        let exprs = Arc::new(db.expr_bucket("exprs").unwrap());
        exprs.put("total", "$.v.sum()").unwrap();

        let docs = Arc::new(db.json_bucket("docs", &["total"], &exprs).unwrap());
        for i in 0u32..20 {
            docs.insert(&format!("doc{i}"), &json!({"v": [i, i+1, i+2]})).unwrap();
        }

        let handles: Vec<_> = (0u32..8)
            .map(|t| {
                let docs = Arc::clone(&docs);
                thread::spawn(move || {
                    for i in 0u32..20 {
                        let res = docs.get_result(&format!("doc{i}"), "total").unwrap().unwrap();
                        let sum: f64 = serde_json::from_value(res).unwrap();
                        assert_eq!(sum as u32, 3 * i + 3, "thread={t} i={i}");
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    }
}
