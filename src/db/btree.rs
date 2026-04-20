//! B+ Tree backed by a memory-mapped file with copy-on-write writes.
//!
//! # Concurrency model (bbolt-inspired)
//!
//! ```text
//! Read:  snapshot = clone(root, Arc<Mmap>)   ← holds RwLock::read() for ~5ns
//!        traverse snapshot                    ← zero locks, zero copies, zero syscalls
//!
//! Write: exclusive Mutex<WriteState>
//!        COW: write new pages to file (never touch existing mmap region)
//!        commit header → remap file → swap Arc<Mmap> + root atomically
//! ```
//!
//! N reader threads work simultaneously with no contention once they have a snapshot.
//! The only shared write point is the brief Arc clone at snapshot creation.

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;
use parking_lot::{Mutex, RwLock};

use super::node::{
    child_for_key, decode_internal, decode_leaf, decode_overflow, encode_inline_val,
    encode_internal, encode_leaf, encode_overflow, encode_overflow_ref, internal_is_full,
    leaf_fits, split_internal, split_leaf, InternalNode, LeafEntry, LeafNode,
};
use super::page::{
    MAX_INLINE_VAL, NODE_INTERNAL, NODE_LEAF, NULL_PAGE, OVERFLOW_DATA_SIZE, PAGE_SIZE,
    TAG_INLINE, TAG_OVERFLOW, PageId,
};

const MAGIC: &[u8; 4] = b"JTDB";
const VERSION: u32 = 1;

// ── Snapshot ──────────────────────────────────────────────────────────────────
//
// A snapshot is a consistent view of the tree: (root page id, mmap of the file).
// Cloning is O(1) — just an Arc refcount bump + u32 copy.
// All read operations work from a snapshot and hold no locks after clone.

#[derive(Clone)]
struct Snapshot {
    root: PageId,
    mmap: Arc<Mmap>,
}

impl Snapshot {
    #[inline]
    fn page(&self, id: PageId) -> &[u8; PAGE_SIZE] {
        let start = id as usize * PAGE_SIZE;
        (&self.mmap[start..start + PAGE_SIZE])
            .try_into()
            .expect("page id out of mmap bounds")
    }
}

// ── WriteState ────────────────────────────────────────────────────────────────
//
// All writes are COW: new pages are appended to the file. Existing pages
// (visible in any snapshot) are never modified.

struct WriteState {
    file: File,
    page_count: u32,
}

impl WriteState {
    fn alloc_id(&mut self) -> PageId {
        let id = self.page_count;
        self.page_count += 1;
        id
    }

    fn write_at(&mut self, id: PageId, data: &[u8; PAGE_SIZE]) -> io::Result<()> {
        self.file.seek(SeekFrom::Start(id as u64 * PAGE_SIZE as u64))?;
        self.file.write_all(data)
    }

    fn write_page(&mut self, data: &[u8; PAGE_SIZE]) -> io::Result<PageId> {
        let id = self.alloc_id();
        self.write_at(id, data)?;
        Ok(id)
    }

    /// Flush data pages first (crash-safe: old header = old valid root), then overwrite header.
    fn commit(&mut self, new_root: PageId) -> io::Result<()> {
        self.file.flush()?;
        let mut header = [0u8; PAGE_SIZE];
        header[0..4].copy_from_slice(MAGIC);
        header[4..8].copy_from_slice(&VERSION.to_le_bytes());
        header[8..12].copy_from_slice(&new_root.to_le_bytes());
        header[12..16].copy_from_slice(&self.page_count.to_le_bytes());
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(&header)?;
        self.file.flush()
    }

    fn remap(&self) -> io::Result<Mmap> {
        // SAFETY: we own the file exclusively via WriteState's Mutex.
        // The mmap covers the file at the moment of mapping; new pages written
        // above are visible in the new Mmap after this call.
        unsafe { Mmap::map(&self.file) }
    }
}

// ── BTree ─────────────────────────────────────────────────────────────────────

pub struct BTree {
    /// Current read snapshot. Held only for the duration of an Arc clone (~5 ns).
    snap: RwLock<Snapshot>,
    /// Exclusive write access. Serialises all mutations.
    write: Mutex<WriteState>,
}

impl BTree {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Arc<Self>> {
        let exists = path.as_ref().exists()
            && path.as_ref().metadata().map(|m| m.len() > 0).unwrap_or(false);

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path.as_ref())?;

        let (root, page_count) = if exists {
            let mut buf = [0u8; PAGE_SIZE];
            let mut f = &file;
            f.seek(SeekFrom::Start(0))?;
            f.read_exact(&mut buf)?;
            if &buf[0..4] != MAGIC {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic bytes"));
            }
            let root = u32::from_le_bytes(buf[8..12].try_into().unwrap());
            let page_count = u32::from_le_bytes(buf[12..16].try_into().unwrap());
            (root, page_count)
        } else {
            // New file: write initial header (page 0).
            let mut header = [0u8; PAGE_SIZE];
            header[0..4].copy_from_slice(MAGIC);
            header[4..8].copy_from_slice(&VERSION.to_le_bytes());
            header[8..12].copy_from_slice(&NULL_PAGE.to_le_bytes());
            header[12..16].copy_from_slice(&1u32.to_le_bytes());
            let mut f = &file;
            f.seek(SeekFrom::Start(0))?;
            f.write_all(&header)?;
            f.flush()?;
            (NULL_PAGE, 1)
        };

        let mmap = unsafe { Mmap::map(&file)? };
        let snap = Snapshot { root, mmap: Arc::new(mmap) };
        Ok(Arc::new(Self {
            snap: RwLock::new(snap),
            write: Mutex::new(WriteState { file, page_count }),
        }))
    }

    // ── Read API ──────────────────────────────────────────────────────────────

    pub fn get(&self, key: &[u8]) -> io::Result<Option<Vec<u8>>> {
        let snap = self.snapshot();
        if snap.root == NULL_PAGE {
            return Ok(None);
        }
        search(&snap, snap.root, key)
    }

    pub fn all(&self) -> io::Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let snap = self.snapshot();
        if snap.root == NULL_PAGE {
            return Ok(vec![]);
        }
        let mut out = vec![];
        range_scan(&snap, snap.root, &[], None, &mut out)?;
        Ok(out)
    }

    pub fn range(&self, start: &[u8], end: &[u8]) -> io::Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let snap = self.snapshot();
        if snap.root == NULL_PAGE {
            return Ok(vec![]);
        }
        let mut out = vec![];
        range_scan(&snap, snap.root, start, Some(end), &mut out)?;
        Ok(out)
    }

    pub fn range_from(&self, start: &[u8]) -> io::Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let snap = self.snapshot();
        if snap.root == NULL_PAGE {
            return Ok(vec![]);
        }
        let mut out = vec![];
        range_scan(&snap, snap.root, start, None, &mut out)?;
        Ok(out)
    }

    // ── Write API ─────────────────────────────────────────────────────────────

    pub fn insert(&self, key: &[u8], val: &[u8]) -> io::Result<()> {
        assert!(key.len() <= 255, "key exceeds 255 bytes");
        let snap = self.snapshot();
        let mut w = self.write.lock();

        let encoded_val = prepare_value(&mut w, val)?;

        let new_root = if snap.root == NULL_PAGE {
            let leaf = LeafNode {
                entries: vec![LeafEntry { key: key.to_vec(), encoded_val }],
                next: NULL_PAGE,
            };
            w.write_page(&encode_leaf(&leaf))?
        } else {
            match cow_insert(&snap, &mut w, snap.root, key, &encoded_val)? {
                (page, None) => page,
                (left, Some((med, right))) => {
                    let new_root = InternalNode {
                        keys: vec![med],
                        children: vec![left, right],
                    };
                    w.write_page(&encode_internal(&new_root))?
                }
            }
        };

        w.commit(new_root)?;
        *self.snap.write() = Snapshot { root: new_root, mmap: Arc::new(w.remap()?) };
        Ok(())
    }

    pub fn delete(&self, key: &[u8]) -> io::Result<bool> {
        let snap = self.snapshot();
        if snap.root == NULL_PAGE {
            return Ok(false);
        }
        let mut w = self.write.lock();
        let (new_root, deleted) = cow_delete(&snap, &mut w, snap.root, key)?;
        if deleted {
            w.commit(new_root)?;
            *self.snap.write() = Snapshot { root: new_root, mmap: Arc::new(w.remap()?) };
        }
        Ok(deleted)
    }

    // ── Internal ──────────────────────────────────────────────────────────────

    /// Clone the current snapshot. Holds RwLock::read() for the duration of
    /// an Arc refcount increment — roughly 5 ns.
    #[inline]
    fn snapshot(&self) -> Snapshot {
        self.snap.read().clone()
    }
}

// ── Value storage ─────────────────────────────────────────────────────────────

fn prepare_value(w: &mut WriteState, val: &[u8]) -> io::Result<Vec<u8>> {
    if val.len() <= MAX_INLINE_VAL {
        Ok(encode_inline_val(val))
    } else {
        write_overflow(w, val)
    }
}

fn write_overflow(w: &mut WriteState, val: &[u8]) -> io::Result<Vec<u8>> {
    let chunks: Vec<&[u8]> = val.chunks(OVERFLOW_DATA_SIZE).collect();
    let n = chunks.len();
    // Pre-allocate consecutive page IDs so we can set next pointers inline.
    let page_ids: Vec<PageId> = (0..n).map(|_| w.alloc_id()).collect();
    for i in 0..n {
        let next = if i + 1 < n { page_ids[i + 1] } else { NULL_PAGE };
        w.write_at(page_ids[i], &encode_overflow(next, chunks[i]))?;
    }
    Ok(encode_overflow_ref(page_ids[0], val.len() as u32))
}

fn read_val(snap: &Snapshot, encoded: &[u8]) -> io::Result<Vec<u8>> {
    match encoded[0] {
        TAG_INLINE => {
            let len = u32::from_le_bytes(encoded[1..5].try_into().unwrap()) as usize;
            Ok(encoded[5..5 + len].to_vec())
        }
        TAG_OVERFLOW => {
            let mut pid = u32::from_le_bytes(encoded[1..5].try_into().unwrap());
            let total = u32::from_le_bytes(encoded[5..9].try_into().unwrap()) as usize;
            let mut out = Vec::with_capacity(total);
            while pid != NULL_PAGE && out.len() < total {
                let page = snap.page(pid);
                let (next, data) = decode_overflow(page);
                let take = (total - out.len()).min(OVERFLOW_DATA_SIZE);
                out.extend_from_slice(&data[..take]);
                pid = next;
            }
            Ok(out)
        }
        _ => Err(io::Error::new(io::ErrorKind::InvalidData, "bad value tag")),
    }
}

// ── Read operations (work on a Snapshot, no locks) ───────────────────────────

fn search(snap: &Snapshot, page_id: PageId, key: &[u8]) -> io::Result<Option<Vec<u8>>> {
    let page = snap.page(page_id);
    match page[0] {
        NODE_INTERNAL => {
            let node = decode_internal(page);
            search(snap, child_for_key(&node, key), key)
        }
        NODE_LEAF => {
            let node = decode_leaf(page);
            for e in &node.entries {
                if e.key.as_slice() == key {
                    return Ok(Some(read_val(snap, &e.encoded_val)?));
                }
            }
            Ok(None)
        }
        _ => Err(io::Error::new(io::ErrorKind::InvalidData, "bad node type")),
    }
}

/// Tree-traversal range scan. Correct for COW trees (no stale linked-list pointers).
fn range_scan(
    snap: &Snapshot,
    page_id: PageId,
    start: &[u8],
    end: Option<&[u8]>,
    out: &mut Vec<(Vec<u8>, Vec<u8>)>,
) -> io::Result<()> {
    let page = snap.page(page_id);
    match page[0] {
        NODE_INTERNAL => {
            let node = decode_internal(page);
            let n = node.keys.len();
            for i in 0..=n {
                // children[i] holds keys in [ keys[i-1], keys[i] )
                // keys[i] is the minimum key of children[i+1].
                //
                // Skip if every key in children[i] is below `start`:
                //   children[i] max < keys[i], and if keys[i] <= start → skip.
                if i < n && !start.is_empty() && node.keys[i].as_slice() <= start {
                    continue;
                }
                // Break if every key in children[i] (and all after) is at or above `end`:
                //   children[i] min >= keys[i-1], and if keys[i-1] >= end → break.
                if i > 0 {
                    if let Some(end) = end {
                        if node.keys[i - 1].as_slice() >= end {
                            break;
                        }
                    }
                }
                range_scan(snap, node.children[i], start, end, out)?;
            }
        }
        NODE_LEAF => {
            let node = decode_leaf(page);
            for e in &node.entries {
                if !start.is_empty() && e.key.as_slice() < start {
                    continue;
                }
                if let Some(end) = end {
                    if e.key.as_slice() >= end {
                        return Ok(());
                    }
                }
                out.push((e.key.clone(), read_val(snap, &e.encoded_val)?));
            }
        }
        _ => return Err(io::Error::new(io::ErrorKind::InvalidData, "bad node type")),
    }
    Ok(())
}

// ── COW write operations ──────────────────────────────────────────────────────
//
// Every function reads from `snap` (immutable) and writes new pages via `w`.
// Existing pages are never modified.

/// Returns `(new_page_id, Option<(separator_key, new_sibling_id)>)`.
fn cow_insert(
    snap: &Snapshot,
    w: &mut WriteState,
    page_id: PageId,
    key: &[u8],
    encoded_val: &[u8],
) -> io::Result<(PageId, Option<(Vec<u8>, PageId)>)> {
    let page = snap.page(page_id);
    match page[0] {
        NODE_INTERNAL => {
            let mut node = decode_internal(page);
            let idx = node
                .keys
                .iter()
                .position(|k| key < k.as_slice())
                .unwrap_or(node.keys.len());
            let child = node.children[idx];

            let (new_child, split) = cow_insert(snap, w, child, key, encoded_val)?;
            node.children[idx] = new_child;

            if let Some((sep, new_sib)) = split {
                node.keys.insert(idx, sep);
                node.children.insert(idx + 1, new_sib);
                if internal_is_full(&node) {
                    let (left, right, median) = split_internal(node);
                    let left_id = w.write_page(&encode_internal(&left))?;
                    let right_id = w.write_page(&encode_internal(&right))?;
                    return Ok((left_id, Some((median, right_id))));
                }
            }
            Ok((w.write_page(&encode_internal(&node))?, None))
        }
        NODE_LEAF => {
            let mut node = decode_leaf(page);

            // Update in-place if key already exists.
            for e in &mut node.entries {
                if e.key.as_slice() == key {
                    e.encoded_val = encoded_val.to_vec();
                    return Ok((w.write_page(&encode_leaf(&node))?, None));
                }
            }

            let idx = node
                .entries
                .iter()
                .position(|e| key < e.key.as_slice())
                .unwrap_or(node.entries.len());
            node.entries.insert(
                idx,
                LeafEntry { key: key.to_vec(), encoded_val: encoded_val.to_vec() },
            );

            if !leaf_fits(&node.entries) {
                let (mut left, right, sep) = split_leaf(node);
                let right_id = w.write_page(&encode_leaf(&right))?;
                left.next = right_id;
                let left_id = w.write_page(&encode_leaf(&left))?;
                Ok((left_id, Some((sep, right_id))))
            } else {
                Ok((w.write_page(&encode_leaf(&node))?, None))
            }
        }
        _ => Err(io::Error::new(io::ErrorKind::InvalidData, "bad node type")),
    }
}

/// Returns `(new_page_id, deleted)`. If nothing was deleted, `new_page_id == page_id`.
fn cow_delete(
    snap: &Snapshot,
    w: &mut WriteState,
    page_id: PageId,
    key: &[u8],
) -> io::Result<(PageId, bool)> {
    let page = snap.page(page_id);
    match page[0] {
        NODE_INTERNAL => {
            let mut node = decode_internal(page);
            let idx = node
                .keys
                .iter()
                .position(|k| key < k.as_slice())
                .unwrap_or(node.keys.len());
            let child = node.children[idx];
            let (new_child, deleted) = cow_delete(snap, w, child, key)?;
            if !deleted {
                return Ok((page_id, false));
            }
            node.children[idx] = new_child;
            Ok((w.write_page(&encode_internal(&node))?, true))
        }
        NODE_LEAF => {
            let mut node = decode_leaf(page);
            let before = node.entries.len();
            node.entries.retain(|e| e.key.as_slice() != key);
            if node.entries.len() < before {
                Ok((w.write_page(&encode_leaf(&node))?, true))
            } else {
                Ok((page_id, false))
            }
        }
        _ => Err(io::Error::new(io::ErrorKind::InvalidData, "bad node type")),
    }
}
