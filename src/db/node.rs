use super::page::{
    NODE_INTERNAL, NODE_LEAF, OVERFLOW_DATA_SIZE, PAGE_SIZE,
    TAG_INLINE, TAG_OVERFLOW, MAX_INLINE_VAL, PageId,
};

// ── Internal node layout ─────────────────────────────────────────────────────
// [type:1][n_keys:2][child_0:4][(key_len:1)(key_bytes)(child:4)]*n_keys
//
// Max keys: solve (PAGE_SIZE - 7) / (1 + 255 + 4) → ≤ 15; use 14 to leave room.
pub const MAX_INTERNAL_KEYS: usize = 14;

// ── Leaf node layout ─────────────────────────────────────────────────────────
// [type:1][n_entries:2][next_leaf:4][(key_len:1)(key_bytes)(val_tag:1)(...)]*
//
// val_tag=0x00 (inline):   [val_len:4][val_bytes]
// val_tag=0x01 (overflow): [first_page:4][total_len:4]
const LEAF_HEADER: usize = 7;

// ── Overflow page layout ──────────────────────────────────────────────────────
// [next_page:4][data:4092]

// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct InternalNode {
    pub keys: Vec<Vec<u8>>,     // n keys
    pub children: Vec<PageId>,  // n+1 children
}

#[derive(Clone, Debug)]
pub struct LeafEntry {
    pub key: Vec<u8>,
    /// Encoded value: [TAG_INLINE len:4 bytes] or [TAG_OVERFLOW page:4 len:4]
    pub encoded_val: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct LeafNode {
    pub entries: Vec<LeafEntry>,
    pub next: PageId,
}

// ── Encode/decode helpers ─────────────────────────────────────────────────────

pub fn encode_inline_val(val: &[u8]) -> Vec<u8> {
    debug_assert!(val.len() <= MAX_INLINE_VAL);
    let mut v = Vec::with_capacity(1 + 4 + val.len());
    v.push(TAG_INLINE);
    v.extend_from_slice(&(val.len() as u32).to_le_bytes());
    v.extend_from_slice(val);
    v
}

pub fn encode_overflow_ref(first_page: PageId, total_len: u32) -> Vec<u8> {
    let mut v = vec![TAG_OVERFLOW];
    v.extend_from_slice(&first_page.to_le_bytes());
    v.extend_from_slice(&total_len.to_le_bytes());
    v
}

fn entry_encoded_size(e: &LeafEntry) -> usize {
    1 + e.key.len() + e.encoded_val.len()
}

pub fn leaf_fits(entries: &[LeafEntry]) -> bool {
    let total: usize = LEAF_HEADER + entries.iter().map(entry_encoded_size).sum::<usize>();
    total <= PAGE_SIZE
}

// ── Internal node encode/decode ───────────────────────────────────────────────

pub fn encode_internal(node: &InternalNode) -> [u8; PAGE_SIZE] {
    let mut page = [0u8; PAGE_SIZE];
    let n = node.keys.len();
    page[0] = NODE_INTERNAL;
    page[1..3].copy_from_slice(&(n as u16).to_le_bytes());
    let mut off = 3;
    // child_0
    page[off..off + 4].copy_from_slice(&node.children[0].to_le_bytes());
    off += 4;
    for i in 0..n {
        let k = &node.keys[i];
        assert!(k.len() <= 255);
        page[off] = k.len() as u8;
        off += 1;
        page[off..off + k.len()].copy_from_slice(k);
        off += k.len();
        page[off..off + 4].copy_from_slice(&node.children[i + 1].to_le_bytes());
        off += 4;
    }
    page
}

pub fn decode_internal(page: &[u8; PAGE_SIZE]) -> InternalNode {
    let n = u16::from_le_bytes(page[1..3].try_into().unwrap()) as usize;
    let mut off = 3;
    let child0 = u32::from_le_bytes(page[off..off + 4].try_into().unwrap());
    off += 4;
    let mut keys = Vec::with_capacity(n);
    let mut children = vec![child0];
    for _ in 0..n {
        let klen = page[off] as usize;
        off += 1;
        keys.push(page[off..off + klen].to_vec());
        off += klen;
        let child = u32::from_le_bytes(page[off..off + 4].try_into().unwrap());
        off += 4;
        children.push(child);
    }
    InternalNode { keys, children }
}

// ── Leaf node encode/decode ───────────────────────────────────────────────────

pub fn encode_leaf(node: &LeafNode) -> [u8; PAGE_SIZE] {
    let mut page = [0u8; PAGE_SIZE];
    page[0] = NODE_LEAF;
    let n = node.entries.len();
    page[1..3].copy_from_slice(&(n as u16).to_le_bytes());
    page[3..7].copy_from_slice(&node.next.to_le_bytes());
    let mut off = 7;
    for e in &node.entries {
        page[off] = e.key.len() as u8;
        off += 1;
        page[off..off + e.key.len()].copy_from_slice(&e.key);
        off += e.key.len();
        page[off..off + e.encoded_val.len()].copy_from_slice(&e.encoded_val);
        off += e.encoded_val.len();
    }
    page
}

pub fn decode_leaf(page: &[u8; PAGE_SIZE]) -> LeafNode {
    let n = u16::from_le_bytes(page[1..3].try_into().unwrap()) as usize;
    let next = u32::from_le_bytes(page[3..7].try_into().unwrap());
    let mut off = 7;
    let mut entries = Vec::with_capacity(n);
    for _ in 0..n {
        let klen = page[off] as usize;
        off += 1;
        let key = page[off..off + klen].to_vec();
        off += klen;
        let tag = page[off];
        let encoded_val = match tag {
            TAG_INLINE => {
                let vlen = u32::from_le_bytes(page[off + 1..off + 5].try_into().unwrap()) as usize;
                let ev = page[off..off + 1 + 4 + vlen].to_vec();
                off += 1 + 4 + vlen;
                ev
            }
            TAG_OVERFLOW => {
                let ev = page[off..off + 9].to_vec(); // tag + page(4) + len(4)
                off += 9;
                ev
            }
            _ => panic!("bad val tag"),
        };
        entries.push(LeafEntry { key, encoded_val });
    }
    LeafNode { entries, next }
}

// ── Overflow page encode/decode ───────────────────────────────────────────────

pub fn encode_overflow(next: PageId, data: &[u8]) -> [u8; PAGE_SIZE] {
    let mut page = [0u8; PAGE_SIZE];
    page[0..4].copy_from_slice(&next.to_le_bytes());
    let len = data.len().min(OVERFLOW_DATA_SIZE);
    page[4..4 + len].copy_from_slice(&data[..len]);
    page
}

pub fn decode_overflow(page: &[u8; PAGE_SIZE]) -> (PageId, &[u8]) {
    let next = u32::from_le_bytes(page[0..4].try_into().unwrap());
    (next, &page[4..])
}

// ── Navigating internal nodes ─────────────────────────────────────────────────

/// Returns the child page to follow for the given key.
pub fn child_for_key(node: &InternalNode, key: &[u8]) -> PageId {
    let idx = node.keys.iter().position(|k| key < k.as_slice()).unwrap_or(node.keys.len());
    node.children[idx]
}

// ── Splitting ─────────────────────────────────────────────────────────────────

/// Split a full internal node. Returns (left, right, median_key).
/// `left` keeps the first half (at page_id), `right` is new.
pub fn split_internal(mut node: InternalNode) -> (InternalNode, InternalNode, Vec<u8>) {
    let mid = node.keys.len() / 2;
    let median = node.keys.remove(mid);
    let right_keys = node.keys.split_off(mid);
    let right_children = node.children.split_off(mid + 1);
    let right = InternalNode { keys: right_keys, children: right_children };
    (node, right, median)
}

/// Split a full leaf node. Returns (left, right, separator_key).
/// `separator_key` is the first key of right (copied up to parent).
pub fn split_leaf(mut node: LeafNode) -> (LeafNode, LeafNode, Vec<u8>) {
    let mid = node.entries.len() / 2;
    let right_entries = node.entries.split_off(mid);
    let separator = right_entries[0].key.clone();
    let right = LeafNode { entries: right_entries, next: node.next };
    (node, right, separator)
}

pub fn internal_is_full(node: &InternalNode) -> bool {
    node.keys.len() >= MAX_INTERNAL_KEYS
}
