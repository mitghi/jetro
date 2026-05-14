use crate::{JetroEngineError, Val};
use std::borrow::Cow;
use std::collections::HashSet;

#[derive(Default)]
pub(super) struct AdaptiveDistinctKeys {
    exact: HashSet<Vec<u8>>,
    front: Option<DistinctFrontFilter>,
}

impl AdaptiveDistinctKeys {
    const FRONT_MIN_KEYS: usize = 64;
    const CUCKOO_MIN_KEYS: usize = 4096;
    const BLOOM_BITS_PER_KEY: usize = 16;

    pub(super) fn insert(&mut self, key: Vec<u8>) -> bool {
        if self.maybe_contains(&key) && self.exact.contains(&key) {
            return false;
        }
        let inserted = self.exact.insert(key.clone());
        if inserted {
            self.insert_front(&key);
        }
        inserted
    }

    pub(super) fn insert_slice(&mut self, key: &[u8]) -> bool {
        if self.maybe_contains(key) && self.exact.contains(key) {
            return false;
        }
        let inserted = self.exact.insert(key.to_vec());
        if inserted {
            self.insert_front(key);
        }
        inserted
    }

    pub(super) fn front_kind(&self) -> DistinctFrontFilterKind {
        self.front
            .as_ref()
            .map(DistinctFrontFilter::kind)
            .unwrap_or(DistinctFrontFilterKind::None)
    }

    fn maybe_contains(&mut self, key: &[u8]) -> bool {
        self.ensure_front_capacity();
        self.front
            .as_ref()
            .is_none_or(|front| front.might_contain(key))
    }

    fn insert_front(&mut self, key: &[u8]) {
        self.ensure_front_capacity();
        let Some(front) = self.front.as_mut() else {
            return;
        };
        if front.insert(key) {
            return;
        }
        self.rebuild_front(self.exact.len() * 2);
    }

    fn ensure_front_capacity(&mut self) {
        if self.exact.len() < Self::FRONT_MIN_KEYS {
            return;
        }

        let target = if self.exact.len() >= Self::CUCKOO_MIN_KEYS {
            DistinctFrontFilterKind::Cuckoo
        } else {
            DistinctFrontFilterKind::Bloom
        };
        if self.front.as_ref().is_some_and(|front| {
            front.kind() == target && front.capacity_satisfies(self.exact.len() + 1)
        }) {
            return;
        }

        self.rebuild_front(self.exact.len() + 1);
    }

    fn rebuild_front(&mut self, capacity_hint: usize) {
        if self.exact.len() < Self::FRONT_MIN_KEYS {
            self.front = None;
            return;
        }

        let mut front = if self.exact.len() >= Self::CUCKOO_MIN_KEYS {
            DistinctFrontFilter::Cuckoo(CuckooFilter::with_capacity(capacity_hint))
        } else {
            DistinctFrontFilter::Bloom(BloomFilter::with_min_bits(
                capacity_hint * Self::BLOOM_BITS_PER_KEY,
            ))
        };
        for key in &self.exact {
            if !front.insert(key) {
                front = DistinctFrontFilter::Bloom(BloomFilter::with_min_bits(
                    capacity_hint * Self::BLOOM_BITS_PER_KEY * 2,
                ));
                for key in &self.exact {
                    front.insert(key);
                }
                break;
            }
        }
        self.front = Some(front);
    }
}

pub(super) fn distinct_key_bytes(key: &Val) -> Result<Vec<u8>, JetroEngineError> {
    let mut out = Vec::new();
    super::ndjson::write_val_json(&mut out, key)?;
    Ok(out)
}

#[cfg(feature = "simd-json")]
pub(super) fn raw_distinct_key_bytes(value: &[u8]) -> Option<Cow<'_, [u8]>> {
    let first = value.iter().copied().find(|b| !b.is_ascii_whitespace())?;
    match first {
        b'n' | b't' | b'f' => Some(Cow::Borrowed(value)),
        b'-' | b'0'..=b'9' if raw_json_number_is_integer(value) => Some(Cow::Borrowed(value)),
        b'"' if !raw_json_string_has_escape(value) => Some(Cow::Borrowed(value)),
        b'"' => canonical_escaped_json_string_key(value).map(Cow::Owned),
        b'{' | b'[' => canonical_json_value_key(value).map(Cow::Owned),
        _ => None,
    }
}

#[cfg(feature = "simd-json")]
fn raw_json_number_is_integer(value: &[u8]) -> bool {
    value
        .iter()
        .copied()
        .take_while(|b| !b.is_ascii_whitespace())
        .all(|b| b != b'.' && b != b'e' && b != b'E')
}

#[cfg(feature = "simd-json")]
fn raw_json_string_has_escape(value: &[u8]) -> bool {
    for byte in value
        .iter()
        .copied()
        .skip_while(|b| b.is_ascii_whitespace())
        .skip(1)
    {
        match byte {
            b'\\' => return true,
            b'"' => return false,
            _ => {}
        }
    }
    true
}

#[cfg(feature = "simd-json")]
fn canonical_escaped_json_string_key(value: &[u8]) -> Option<Vec<u8>> {
    let decoded: String = serde_json::from_slice(value).ok()?;
    let mut out = Vec::with_capacity(value.len());
    super::ndjson::write_json_str(&mut out, &decoded).ok()?;
    Some(out)
}

#[cfg(feature = "simd-json")]
fn canonical_json_value_key(value: &[u8]) -> Option<Vec<u8>> {
    let decoded: serde_json::Value = serde_json::from_slice(value).ok()?;
    let mut out = Vec::with_capacity(value.len());
    serde_json::to_writer(&mut out, &decoded).ok()?;
    Some(out)
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum DistinctFrontFilterKind {
    #[default]
    None,
    Bloom,
    Cuckoo,
}

enum DistinctFrontFilter {
    Bloom(BloomFilter),
    Cuckoo(CuckooFilter),
}

impl DistinctFrontFilter {
    fn kind(&self) -> DistinctFrontFilterKind {
        match self {
            Self::Bloom(_) => DistinctFrontFilterKind::Bloom,
            Self::Cuckoo(_) => DistinctFrontFilterKind::Cuckoo,
        }
    }

    fn capacity_satisfies(&self, keys: usize) -> bool {
        match self {
            Self::Bloom(bloom) => bloom.bit_len() >= keys * AdaptiveDistinctKeys::BLOOM_BITS_PER_KEY,
            Self::Cuckoo(cuckoo) => cuckoo.capacity_satisfies(keys),
        }
    }

    fn insert(&mut self, key: &[u8]) -> bool {
        match self {
            Self::Bloom(bloom) => {
                bloom.insert(key);
                true
            }
            Self::Cuckoo(cuckoo) => cuckoo.insert(key),
        }
    }

    fn might_contain(&self, key: &[u8]) -> bool {
        match self {
            Self::Bloom(bloom) => bloom.might_contain(key),
            Self::Cuckoo(cuckoo) => cuckoo.might_contain(key),
        }
    }
}

struct BloomFilter {
    words: Vec<u64>,
    bit_mask: usize,
}

impl BloomFilter {
    fn with_min_bits(bits: usize) -> Self {
        let bit_len = bits.next_power_of_two().max(1024);
        Self {
            words: vec![0; bit_len / 64],
            bit_mask: bit_len - 1,
        }
    }

    fn bit_len(&self) -> usize {
        self.words.len() * 64
    }

    fn insert(&mut self, key: &[u8]) {
        let (a, b) = bloom_hashes(key);
        self.set(a);
        self.set(b);
        self.set(a.wrapping_add(b.rotate_left(17)));
    }

    fn might_contain(&self, key: &[u8]) -> bool {
        let (a, b) = bloom_hashes(key);
        self.get(a) && self.get(b) && self.get(a.wrapping_add(b.rotate_left(17)))
    }

    fn set(&mut self, hash: u64) {
        let bit = (hash as usize) & self.bit_mask;
        self.words[bit / 64] |= 1u64 << (bit % 64);
    }

    fn get(&self, hash: u64) -> bool {
        let bit = (hash as usize) & self.bit_mask;
        (self.words[bit / 64] & (1u64 << (bit % 64))) != 0
    }
}

fn bloom_hashes(key: &[u8]) -> (u64, u64) {
    (
        fast_key_hash(key, 0x9e37_79b9_7f4a_7c15),
        fast_key_hash(key, 0xbf58_476d_1ce4_e5b9),
    )
}

fn fast_key_hash(key: &[u8], seed: u64) -> u64 {
    let mut hash = seed ^ ((key.len() as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
    let mut chunks = key.chunks_exact(8);
    for chunk in &mut chunks {
        let lane = u64::from_le_bytes(chunk.try_into().unwrap());
        hash ^= mix_u64(lane.wrapping_add(0x9e37_79b9_7f4a_7c15));
        hash = hash.rotate_left(27).wrapping_mul(0x94d0_49bb_1331_11eb);
    }
    let rem = chunks.remainder();
    if !rem.is_empty() {
        let mut tail = 0u64;
        for (idx, byte) in rem.iter().enumerate() {
            tail |= (*byte as u64) << (idx * 8);
        }
        hash ^= mix_u64(tail ^ 0xd6e8_feb8_6659_fd93);
    }
    mix_u64(hash)
}

fn mix_u64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

struct CuckooFilter {
    buckets: Vec<[u16; 4]>,
    bucket_mask: usize,
}

impl CuckooFilter {
    const MAX_KICKS: usize = 64;

    fn with_capacity(keys: usize) -> Self {
        let bucket_count = ((keys.max(1) * 2).div_ceil(4))
            .next_power_of_two()
            .max(1024);
        Self {
            buckets: vec![[0; 4]; bucket_count],
            bucket_mask: bucket_count - 1,
        }
    }

    fn capacity_satisfies(&self, keys: usize) -> bool {
        self.buckets.len() * 2 >= keys
    }

    fn insert(&mut self, key: &[u8]) -> bool {
        let hash = bloom_hashes(key).0;
        let fp = cuckoo_fingerprint(hash);
        let i1 = (hash as usize) & self.bucket_mask;
        let i2 = self.alt_index(i1, fp);
        if self.bucket_contains(i1, fp) || self.bucket_contains(i2, fp) {
            return true;
        }
        if self.insert_bucket(i1, fp) || self.insert_bucket(i2, fp) {
            return true;
        }

        let mut index = if hash & 1 == 0 { i1 } else { i2 };
        let mut fp = fp;
        for kick in 0..Self::MAX_KICKS {
            let slot = ((hash >> ((kick % 8) * 8)) as usize) & 3;
            std::mem::swap(&mut self.buckets[index][slot], &mut fp);
            index = self.alt_index(index, fp);
            if self.insert_bucket(index, fp) {
                return true;
            }
        }
        false
    }

    fn might_contain(&self, key: &[u8]) -> bool {
        let hash = bloom_hashes(key).0;
        let fp = cuckoo_fingerprint(hash);
        let i1 = (hash as usize) & self.bucket_mask;
        let i2 = self.alt_index(i1, fp);
        self.bucket_contains(i1, fp) || self.bucket_contains(i2, fp)
    }

    fn alt_index(&self, index: usize, fp: u16) -> usize {
        (index ^ cuckoo_fp_hash(fp)) & self.bucket_mask
    }

    fn insert_bucket(&mut self, index: usize, fp: u16) -> bool {
        if let Some(slot) = self.buckets[index].iter_mut().find(|slot| **slot == 0) {
            *slot = fp;
            true
        } else {
            false
        }
    }

    fn bucket_contains(&self, index: usize, fp: u16) -> bool {
        self.buckets[index].contains(&fp)
    }
}

fn cuckoo_fingerprint(hash: u64) -> u16 {
    let fp = (hash as u16) ^ ((hash >> 16) as u16) ^ ((hash >> 32) as u16);
    if fp == 0 { 1 } else { fp }
}

fn cuckoo_fp_hash(fp: u16) -> usize {
    let mut x = fp as u64;
    x = x.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    x ^= x >> 32;
    x as usize
}

#[cfg(test)]
mod tests {
    #[test]
    fn adaptive_distinct_keys_remain_exact_after_front_filter_activation() {
        let mut keys = super::AdaptiveDistinctKeys::default();
        for n in 0..128 {
            assert!(keys.insert(format!("k{n}").into_bytes()));
        }
        assert!(keys.front.is_some());
        for n in 0..128 {
            assert!(!keys.insert(format!("k{n}").into_bytes()));
        }
        assert!(keys.insert(b"new".to_vec()));
    }

    #[test]
    fn adaptive_distinct_keys_promote_to_cuckoo_front_filter() {
        let mut keys = super::AdaptiveDistinctKeys::default();
        for n in 0..5000 {
            assert!(keys.insert(format!("k{n}").into_bytes()));
        }
        assert!(matches!(
            keys.front,
            Some(super::DistinctFrontFilter::Cuckoo(_))
        ));
        for n in 0..5000 {
            assert!(!keys.insert_slice(format!("k{n}").as_bytes()));
        }
        assert!(keys.insert_slice(b"fresh"));
    }
}
