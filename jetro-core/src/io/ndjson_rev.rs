use super::RowError;
use crate::util::is_truthy;
use crate::{JetroEngine, JetroEngineError};
use memchr::memrchr;
use serde_json::Value;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

#[cfg(feature = "simd-json")]
use super::ndjson_byte::{
    raw_json_byte_path_value, tape_plan_can_write_byte_row, write_ndjson_byte_tape_plan_row,
    BytePlanWrite, RawFieldValue,
};

/// Reverse NDJSON line reader over a seekable file.
///
/// The reader scans fixed-size chunks from EOF to BOF and returns owned line
/// bytes in reverse physical order. It keeps only the current chunk and one
/// cross-chunk carry buffer, so memory stays bounded by the longest row plus
/// the configured chunk size.
pub struct NdjsonReverseFileDriver {
    file: File,
    pos: u64,
    chunk_size: usize,
    max_line_len: usize,
    carry: Vec<u8>,
    pending: VecDeque<Vec<u8>>,
    finished_head: bool,
    reverse_line_no: u64,
}

impl NdjsonReverseFileDriver {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, RowError> {
        Self::with_options(path, super::ndjson::NdjsonOptions::default())
    }

    pub fn with_chunk_size<P: AsRef<Path>>(path: P, chunk_size: usize) -> Result<Self, RowError> {
        Self::with_options(
            path,
            super::ndjson::NdjsonOptions::default().with_reverse_chunk_size(chunk_size),
        )
    }

    pub fn with_options<P: AsRef<Path>>(
        path: P,
        options: super::ndjson::NdjsonOptions,
    ) -> Result<Self, RowError> {
        let mut file = File::open(path)?;
        let pos = file.seek(SeekFrom::End(0))?;
        Ok(Self {
            file,
            pos,
            chunk_size: options.reverse_chunk_size.max(1),
            max_line_len: options.max_line_len,
            carry: Vec::new(),
            pending: VecDeque::new(),
            finished_head: false,
            reverse_line_no: 0,
        })
    }

    pub fn next_line(&mut self) -> Result<Option<Vec<u8>>, RowError> {
        Ok(self.next_line_with_reverse_no()?.map(|(_, line)| line))
    }

    pub fn next_line_with_reverse_no(&mut self) -> Result<Option<(u64, Vec<u8>)>, RowError> {
        loop {
            if let Some(line) = self.pending.pop_front() {
                self.reverse_line_no += 1;
                return Ok(Some((self.reverse_line_no, line)));
            }

            if self.pos == 0 {
                if self.finished_head || self.carry.is_empty() {
                    return Ok(None);
                }
                self.finished_head = true;
                let mut line = std::mem::take(&mut self.carry);
                trim_line_ending(&mut line);
                self.check_line_len(line.len())?;
                if line.iter().any(|b| !b.is_ascii_whitespace()) {
                    self.reverse_line_no += 1;
                    return Ok(Some((self.reverse_line_no, line)));
                }
                return Ok(None);
            }

            let read_len = self.chunk_size.min(self.pos as usize);
            self.pos -= read_len as u64;
            let mut chunk = vec![0u8; read_len];
            self.file.seek(SeekFrom::Start(self.pos))?;
            self.file.read_exact(&mut chunk)?;

            let mut end = chunk.len();
            while let Some(nl) = memrchr(b'\n', &chunk[..end]) {
                let mut line = Vec::with_capacity(end - nl - 1 + self.carry.len());
                line.extend_from_slice(&chunk[nl + 1..end]);
                line.extend_from_slice(&self.carry);
                self.carry.clear();
                end = nl;
                trim_line_ending(&mut line);
                self.check_line_len(line.len())?;
                if line.iter().any(|b| !b.is_ascii_whitespace()) {
                    self.pending.push_back(line);
                }
            }

            if end > 0 {
                let mut next = Vec::with_capacity(end + self.carry.len());
                next.extend_from_slice(&chunk[..end]);
                next.extend_from_slice(&self.carry);
                self.check_line_len(next.len())?;
                self.carry = next;
            }
        }
    }

    fn check_line_len(&self, len: usize) -> Result<(), RowError> {
        if len > self.max_line_len {
            return Err(RowError::LineTooLarge {
                line_no: self.reverse_line_no + self.pending.len() as u64 + 1,
                len,
                max: self.max_line_len,
            });
        }
        Ok(())
    }
}

pub fn collect_ndjson_rev<P>(
    engine: &JetroEngine,
    path: P,
    query: &str,
) -> Result<Vec<Value>, JetroEngineError>
where
    P: AsRef<Path>,
{
    collect_ndjson_rev_with_options(engine, path, query, super::ndjson::NdjsonOptions::default())
}

pub fn collect_ndjson_rev_with_options<P>(
    engine: &JetroEngine,
    path: P,
    query: &str,
    options: super::ndjson::NdjsonOptions,
) -> Result<Vec<Value>, JetroEngineError>
where
    P: AsRef<Path>,
{
    let mut values = Vec::new();
    drive_rev(engine, path, query, options, |value| {
        values.push(Value::from(value));
        Ok(super::ndjson::NdjsonControl::Continue)
    })?;
    Ok(values)
}

pub fn for_each_ndjson_rev<P, F>(
    engine: &JetroEngine,
    path: P,
    query: &str,
    mut f: F,
) -> Result<usize, JetroEngineError>
where
    P: AsRef<Path>,
    F: FnMut(Value),
{
    for_each_ndjson_rev_with_options(
        engine,
        path,
        query,
        super::ndjson::NdjsonOptions::default(),
        |value| {
            f(value);
            Ok(super::ndjson::NdjsonControl::Continue)
        },
    )
}

pub fn for_each_ndjson_rev_with_options<P, F>(
    engine: &JetroEngine,
    path: P,
    query: &str,
    options: super::ndjson::NdjsonOptions,
    mut f: F,
) -> Result<usize, JetroEngineError>
where
    P: AsRef<Path>,
    F: FnMut(Value) -> Result<super::ndjson::NdjsonControl, JetroEngineError>,
{
    drive_rev(engine, path, query, options, |value| f(Value::from(value)))
}

pub fn collect_ndjson_rev_matches<P>(
    engine: &JetroEngine,
    path: P,
    predicate: &str,
    limit: usize,
) -> Result<Vec<Value>, JetroEngineError>
where
    P: AsRef<Path>,
{
    collect_ndjson_rev_matches_with_options(
        engine,
        path,
        predicate,
        limit,
        super::ndjson::NdjsonOptions::default(),
    )
}

pub fn collect_ndjson_rev_matches_with_options<P>(
    engine: &JetroEngine,
    path: P,
    predicate: &str,
    limit: usize,
    options: super::ndjson::NdjsonOptions,
) -> Result<Vec<Value>, JetroEngineError>
where
    P: AsRef<Path>,
{
    let mut values = Vec::with_capacity(limit);
    drive_rev_matches(engine, path, predicate, limit, options, |value| {
        values.push(Value::from(value));
        Ok(super::ndjson::NdjsonControl::Continue)
    })?;
    Ok(values)
}

pub fn run_ndjson_rev<P, W>(
    engine: &JetroEngine,
    path: P,
    query: &str,
    writer: W,
) -> Result<usize, JetroEngineError>
where
    P: AsRef<Path>,
    W: Write,
{
    run_ndjson_rev_with_options(
        engine,
        path,
        query,
        writer,
        super::ndjson::NdjsonOptions::default(),
    )
}

pub fn run_ndjson_rev_with_options<P, W>(
    engine: &JetroEngine,
    path: P,
    query: &str,
    writer: W,
    options: super::ndjson::NdjsonOptions,
) -> Result<usize, JetroEngineError>
where
    P: AsRef<Path>,
    W: Write,
{
    #[cfg(feature = "simd-json")]
    if let Some(plan) = super::ndjson::direct_tape_plan(engine, query) {
        return drive_rev_writer_tape(engine, path, &plan, None, options, writer);
    }

    let mut writer = super::ndjson::ndjson_writer_with_options(writer, options);
    let count = drive_rev(engine, path, query, options, |value| {
        super::ndjson::write_val_line(&mut writer, &value)?;
        Ok(super::ndjson::NdjsonControl::Continue)
    })?;
    writer.flush()?;
    Ok(count)
}

pub fn run_ndjson_rev_limit<P, W>(
    engine: &JetroEngine,
    path: P,
    query: &str,
    limit: usize,
    writer: W,
) -> Result<usize, JetroEngineError>
where
    P: AsRef<Path>,
    W: Write,
{
    run_ndjson_rev_limit_with_options(
        engine,
        path,
        query,
        limit,
        writer,
        super::ndjson::NdjsonOptions::default(),
    )
}

pub fn run_ndjson_rev_limit_with_options<P, W>(
    engine: &JetroEngine,
    path: P,
    query: &str,
    limit: usize,
    writer: W,
    options: super::ndjson::NdjsonOptions,
) -> Result<usize, JetroEngineError>
where
    P: AsRef<Path>,
    W: Write,
{
    if limit == 0 {
        return Ok(0);
    }

    #[cfg(feature = "simd-json")]
    if let Some(plan) = super::ndjson::direct_tape_plan(engine, query) {
        return drive_rev_writer_tape(engine, path, &plan, Some(limit), options, writer);
    }

    let mut writer = super::ndjson::ndjson_writer_with_options(writer, options);
    let mut emitted = 0usize;
    let count = drive_rev(engine, path, query, options, |value| {
        super::ndjson::write_val_line(&mut writer, &value)?;
        emitted += 1;
        Ok(if emitted >= limit {
            super::ndjson::NdjsonControl::Stop
        } else {
            super::ndjson::NdjsonControl::Continue
        })
    })?;
    writer.flush()?;
    Ok(count)
}

pub fn run_ndjson_rev_distinct_by<P, W>(
    engine: &JetroEngine,
    path: P,
    key_query: &str,
    query: &str,
    limit: usize,
    writer: W,
) -> Result<usize, JetroEngineError>
where
    P: AsRef<Path>,
    W: Write,
{
    run_ndjson_rev_distinct_by_with_options(
        engine,
        path,
        key_query,
        query,
        limit,
        writer,
        super::ndjson::NdjsonOptions::default(),
    )
}

pub fn run_ndjson_rev_distinct_by_with_options<P, W>(
    engine: &JetroEngine,
    path: P,
    key_query: &str,
    query: &str,
    limit: usize,
    writer: W,
    options: super::ndjson::NdjsonOptions,
) -> Result<usize, JetroEngineError>
where
    P: AsRef<Path>,
    W: Write,
{
    if limit == 0 {
        return Ok(0);
    }

    #[cfg(feature = "simd-json")]
    let direct_key_plan = super::ndjson::direct_tape_plan(engine, key_query);
    #[cfg(feature = "simd-json")]
    let direct_value_plan = super::ndjson::direct_tape_plan(engine, query)
        .filter(|plan| tape_plan_can_write_byte_row(plan));

    let mut key_plan = None;
    let mut value_plan = None;
    let mut vm = None;
    let mut driver = NdjsonReverseFileDriver::with_options(path, options)?;
    let mut writer = super::ndjson::ndjson_writer_with_options(writer, options);
    #[cfg(feature = "simd-json")]
    let mut byte_scratch = Vec::with_capacity(options.initial_buffer_capacity);
    let mut seen = AdaptiveDistinctKeys::default();
    let mut emitted = 0usize;

    while let Some((reverse_row_no, row)) = driver.next_line_with_reverse_no()? {
        let mut row = Some(row);
        let mut document = None;

        #[cfg(feature = "simd-json")]
        let direct_key = direct_key_plan
            .as_ref()
            .and_then(|plan| row.as_deref().and_then(|row| distinct_key_bytes_direct(row, plan)));
        #[cfg(not(feature = "simd-json"))]
        let direct_key = None;

        let inserted = if let Some(key) = direct_key {
            seen.insert_slice(key)
        } else {
            let parsed =
                super::ndjson::parse_row(engine, reverse_row_no, row.take().unwrap())?;
            let plan = key_plan.get_or_insert_with(|| {
                engine.cached_plan(key_query, crate::plan::physical::PlanningContext::bytes())
            });
            let vm = vm.get_or_insert_with(|| engine.lock_vm());
            let key = crate::exec::router::collect_plan_val_with_vm(&parsed, plan, vm)
                .map_err(|err| super::ndjson::row_eval_error(reverse_row_no, err))?;
            let key = distinct_key_bytes(&key)?;
            document = Some(parsed);
            seen.insert(key)
        };
        if !inserted {
            continue;
        }

        #[cfg(feature = "simd-json")]
        if let (Some(plan), Some(row)) = (direct_value_plan.as_ref(), row.as_deref()) {
            byte_scratch.clear();
            match write_ndjson_byte_tape_plan_row(&mut writer, row, plan, &mut byte_scratch)? {
                BytePlanWrite::Done => {
                    writer.write_all(b"\n")?;
                    emitted += 1;
                    if emitted >= limit {
                        break;
                    }
                    continue;
                }
                BytePlanWrite::Fallback => {}
            }
        }

        let parsed = match document {
            Some(document) => document,
            None => super::ndjson::parse_row(engine, reverse_row_no, row.take().unwrap())?,
        };
        let plan = value_plan.get_or_insert_with(|| {
            engine.cached_plan(query, crate::plan::physical::PlanningContext::bytes())
        });
        let vm = vm.get_or_insert_with(|| engine.lock_vm());
        let value = crate::exec::router::collect_plan_val_with_vm(&parsed, plan, vm)
            .map_err(|err| super::ndjson::row_eval_error(reverse_row_no, err))?;
        super::ndjson::write_val_line(&mut writer, &value)?;
        emitted += 1;
        if emitted >= limit {
            break;
        }
    }

    writer.flush()?;
    Ok(emitted)
}

fn distinct_key_bytes(key: &crate::data::value::Val) -> Result<Vec<u8>, JetroEngineError> {
    let mut out = Vec::new();
    super::ndjson::write_val_json(&mut out, key)?;
    Ok(out)
}

#[cfg(feature = "simd-json")]
fn distinct_key_bytes_direct<'a>(
    row: &'a [u8],
    plan: &super::ndjson::NdjsonDirectTapePlan,
) -> Option<&'a [u8]> {
    const NULL_KEY: &[u8] = b"null";

    let super::ndjson::NdjsonDirectTapePlan::RootPath(steps) = plan else {
        return None;
    };
    match raw_json_byte_path_value(row, steps) {
        RawFieldValue::Found(value) if raw_distinct_key_is_byte_stable(value) => Some(value),
        RawFieldValue::Found(_) => None,
        RawFieldValue::Missing => Some(NULL_KEY),
        RawFieldValue::Fallback => None,
    }
}

#[cfg(feature = "simd-json")]
fn raw_distinct_key_is_byte_stable(value: &[u8]) -> bool {
    let Some(first) = value.iter().copied().find(|b| !b.is_ascii_whitespace()) else {
        return false;
    };
    match first {
        b'n' | b't' | b'f' | b'-' | b'0'..=b'9' => true,
        b'"' => !raw_json_string_has_escape(value),
        _ => false,
    }
}

#[cfg(feature = "simd-json")]
fn raw_json_string_has_escape(value: &[u8]) -> bool {
    for byte in value.iter().copied().skip_while(|b| b.is_ascii_whitespace()).skip(1) {
        match byte {
            b'\\' => return true,
            b'"' => return false,
            _ => {}
        }
    }
    true
}

#[derive(Default)]
struct AdaptiveDistinctKeys {
    exact: HashSet<Vec<u8>>,
    front: Option<DistinctFrontFilter>,
}

impl AdaptiveDistinctKeys {
    const FRONT_MIN_KEYS: usize = 64;
    const CUCKOO_MIN_KEYS: usize = 4096;
    const BLOOM_BITS_PER_KEY: usize = 16;

    fn insert(&mut self, key: Vec<u8>) -> bool {
        if self.maybe_contains(&key) && self.exact.contains(&key) {
            return false;
        }
        let inserted = self.exact.insert(key.clone());
        if inserted {
            self.insert_front(&key);
        }
        inserted
    }

    fn insert_slice(&mut self, key: &[u8]) -> bool {
        if self.maybe_contains(key) && self.exact.contains(key) {
            return false;
        }
        let inserted = self.exact.insert(key.to_vec());
        if inserted {
            self.insert_front(key);
        }
        inserted
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
            DistinctFrontKind::Cuckoo
        } else {
            DistinctFrontKind::Bloom
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

#[derive(Clone, Copy, Eq, PartialEq)]
enum DistinctFrontKind {
    Bloom,
    Cuckoo,
}

enum DistinctFrontFilter {
    Bloom(BloomFilter),
    Cuckoo(CuckooFilter),
}

impl DistinctFrontFilter {
    fn kind(&self) -> DistinctFrontKind {
        match self {
            Self::Bloom(_) => DistinctFrontKind::Bloom,
            Self::Cuckoo(_) => DistinctFrontKind::Cuckoo,
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

pub fn run_ndjson_rev_matches<P, W>(
    engine: &JetroEngine,
    path: P,
    predicate: &str,
    limit: usize,
    writer: W,
) -> Result<usize, JetroEngineError>
where
    P: AsRef<Path>,
    W: Write,
{
    run_ndjson_rev_matches_with_options(
        engine,
        path,
        predicate,
        limit,
        writer,
        super::ndjson::NdjsonOptions::default(),
    )
}

pub fn run_ndjson_rev_matches_with_options<P, W>(
    engine: &JetroEngine,
    path: P,
    predicate: &str,
    limit: usize,
    writer: W,
    options: super::ndjson::NdjsonOptions,
) -> Result<usize, JetroEngineError>
where
    P: AsRef<Path>,
    W: Write,
{
    drive_rev_matches_writer(engine, path, predicate, limit, options, writer)
}

#[cfg(feature = "simd-json")]
fn drive_rev_writer_tape<P, W>(
    engine: &JetroEngine,
    path: P,
    plan: &super::ndjson::NdjsonDirectTapePlan,
    limit: Option<usize>,
    options: super::ndjson::NdjsonOptions,
    writer: W,
) -> Result<usize, JetroEngineError>
where
    P: AsRef<Path>,
    W: Write,
{
    let mut driver = NdjsonReverseFileDriver::with_options(path, options)?;
    let mut writer = super::ndjson::ndjson_writer_with_options(writer, options);
    let mut scratch =
        crate::data::tape::TapeScratch::with_capacity(options.initial_buffer_capacity);
    let mut runner = super::ndjson::NdjsonTapeWriterRunner::new(engine, plan);
    let mut count = 0usize;

    while let Some((reverse_row_no, row)) = driver.next_line_with_reverse_no()? {
        scratch.parse_slice(&row).map_err(|message| {
            super::ndjson::row_parse_error(
                reverse_row_no,
                JetroEngineError::Eval(crate::EvalError(format!("Invalid JSON: {message}"))),
            )
        })?;
        runner.write_row(&scratch, &mut writer)?;
        writer.write_all(b"\n")?;
        count += 1;
        if limit.is_some_and(|limit| count >= limit) {
            break;
        }
    }

    writer.flush()?;
    Ok(count)
}

fn drive_rev<P, F>(
    engine: &JetroEngine,
    path: P,
    query: &str,
    options: super::ndjson::NdjsonOptions,
    mut emit: F,
) -> Result<usize, JetroEngineError>
where
    P: AsRef<Path>,
    F: FnMut(crate::data::value::Val) -> Result<super::ndjson::NdjsonControl, JetroEngineError>,
{
    let mut driver = NdjsonReverseFileDriver::with_options(path, options)?;
    let mut executor = super::ndjson::NdjsonRowExecutor::new(engine, query);
    let mut count = 0usize;

    while let Some((reverse_row_no, row)) = driver.next_line_with_reverse_no()? {
        let out = executor.eval_owned_row(reverse_row_no, row)?;
        count += 1;
        if matches!(emit(out)?, super::ndjson::NdjsonControl::Stop) {
            break;
        }
    }

    Ok(count)
}

fn drive_rev_matches<P, F>(
    engine: &JetroEngine,
    path: P,
    predicate: &str,
    limit: usize,
    options: super::ndjson::NdjsonOptions,
    mut emit: F,
) -> Result<usize, JetroEngineError>
where
    P: AsRef<Path>,
    F: FnMut(crate::data::value::Val) -> Result<super::ndjson::NdjsonControl, JetroEngineError>,
{
    if limit == 0 {
        return Ok(0);
    }

    let mut driver = NdjsonReverseFileDriver::with_options(path, options)?;
    let mut executor = super::ndjson::NdjsonRowExecutor::new(engine, predicate);
    let mut emitted = 0usize;

    while let Some((reverse_row_no, row)) = driver.next_line_with_reverse_no()? {
        let document = executor.parse_owned_row(reverse_row_no, row)?;
        let matched = executor.eval_document(reverse_row_no, &document)?;
        if !is_truthy(&matched) {
            continue;
        }

        let root = document
            .root_val_with(executor.engine().keys())
            .map_err(|err| super::ndjson::row_eval_error(reverse_row_no, err))?;
        emitted += 1;
        if matches!(emit(root)?, super::ndjson::NdjsonControl::Stop) || emitted >= limit {
            break;
        }
    }

    Ok(emitted)
}

fn drive_rev_matches_writer<P, W>(
    engine: &JetroEngine,
    path: P,
    predicate: &str,
    limit: usize,
    options: super::ndjson::NdjsonOptions,
    writer: W,
) -> Result<usize, JetroEngineError>
where
    P: AsRef<Path>,
    W: Write,
{
    if limit == 0 {
        return Ok(0);
    }

    #[cfg(feature = "simd-json")]
    if let Some(predicate) = super::ndjson::direct_tape_predicate(engine, predicate) {
        return drive_rev_matches_writer_tape(engine, path, &predicate, limit, options, writer);
    }

    let mut driver = NdjsonReverseFileDriver::with_options(path, options)?;
    let mut executor = super::ndjson::NdjsonRowExecutor::new(engine, predicate);
    let mut writer = super::ndjson::ndjson_writer_with_options(writer, options);
    let mut emitted = 0usize;

    while let Some((reverse_row_no, row)) = driver.next_line_with_reverse_no()? {
        let document = executor.parse_owned_row(reverse_row_no, row)?;
        let matched = executor.eval_document(reverse_row_no, &document)?;
        if !is_truthy(&matched) {
            continue;
        }

        super::ndjson::write_document_line(
            &mut writer,
            &document,
            reverse_row_no,
            executor.engine(),
        )?;
        emitted += 1;
        if emitted >= limit {
            break;
        }
    }

    writer.flush()?;
    Ok(emitted)
}

#[cfg(feature = "simd-json")]
fn drive_rev_matches_writer_tape<P, W>(
    engine: &JetroEngine,
    path: P,
    predicate: &super::ndjson::NdjsonDirectPredicate,
    limit: usize,
    options: super::ndjson::NdjsonOptions,
    writer: W,
) -> Result<usize, JetroEngineError>
where
    P: AsRef<Path>,
    W: Write,
{
    let mut driver = NdjsonReverseFileDriver::with_options(path, options)?;
    let mut writer = super::ndjson::ndjson_writer_with_options(writer, options);
    let mut scratch =
        crate::data::tape::TapeScratch::with_capacity(options.initial_buffer_capacity);
    let mut emitted = 0usize;
    let needs_vm = super::ndjson::predicate_needs_vm(predicate);
    let mut vm = needs_vm.then(|| engine.lock_vm());
    let env = needs_vm.then(|| crate::data::context::Env::new(crate::Val::Null));
    let mut predicate_path = super::ndjson::NdjsonPathCache::default();

    while let Some((reverse_row_no, row)) = driver.next_line_with_reverse_no()? {
        scratch.parse_slice(&row).map_err(|message| {
            super::ndjson::row_parse_error(
                reverse_row_no,
                JetroEngineError::Eval(crate::EvalError(format!("Invalid JSON: {message}"))),
            )
        })?;
        if !super::ndjson::eval_tape_predicate(
            &scratch,
            predicate,
            env.as_ref(),
            &mut vm,
            &mut predicate_path,
        )
        .map_err(JetroEngineError::Eval)?
        {
            continue;
        }
        writer.write_all(&row)?;
        writer.write_all(b"\n")?;
        emitted += 1;
        if emitted >= limit {
            break;
        }
    }

    writer.flush()?;
    Ok(emitted)
}

fn trim_line_ending(buf: &mut Vec<u8>) {
    while matches!(buf.last(), Some(b'\n' | b'\r')) {
        buf.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::NdjsonReverseFileDriver;
    use crate::JetroEngine;
    use std::path::PathBuf;

    #[test]
    fn reverse_driver_reads_rows_from_tail() {
        let path = temp_path("jetro-ndjson-rev-basic");
        std::fs::write(&path, b"{\"n\":1}\n{\"n\":2}\n{\"n\":3}\n").unwrap();
        let mut driver = NdjsonReverseFileDriver::with_chunk_size(&path, 8).unwrap();

        assert_eq!(driver.next_line().unwrap().unwrap(), br#"{"n":3}"#);
        assert_eq!(driver.next_line().unwrap().unwrap(), br#"{"n":2}"#);
        assert_eq!(driver.next_line().unwrap().unwrap(), br#"{"n":1}"#);
        assert!(driver.next_line().unwrap().is_none());

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn reverse_driver_handles_missing_final_newline_and_blank_lines() {
        let path = temp_path("jetro-ndjson-rev-edge");
        std::fs::write(&path, b"\n{\"n\":1}\r\n\n{\"n\":2}").unwrap();
        let mut driver = NdjsonReverseFileDriver::with_chunk_size(&path, 5).unwrap();

        assert_eq!(driver.next_line().unwrap().unwrap(), br#"{"n":2}"#);
        assert_eq!(driver.next_line().unwrap().unwrap(), br#"{"n":1}"#);
        assert!(driver.next_line().unwrap().is_none());

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn reverse_driver_reports_reverse_row_numbers() {
        let path = temp_path("jetro-ndjson-rev-row-no");
        std::fs::write(&path, b"{\"n\":1}\n{\"n\":2}\n").unwrap();
        let mut driver = NdjsonReverseFileDriver::with_chunk_size(&path, 3).unwrap();

        assert_eq!(
            driver.next_line_with_reverse_no().unwrap().unwrap(),
            (1, br#"{"n":2}"#.to_vec())
        );
        assert_eq!(
            driver.next_line_with_reverse_no().unwrap().unwrap(),
            (2, br#"{"n":1}"#.to_vec())
        );
        assert!(driver.next_line_with_reverse_no().unwrap().is_none());

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn reverse_query_uses_direct_writer_shapes() {
        let path = temp_path("jetro-ndjson-rev-direct");
        std::fs::write(
            &path,
            b"{\"name\":\"ada\",\"attrs\":[{\"key\":\"a\",\"value\":1}]}\n{\"name\":\"bob\",\"attrs\":[{\"key\":\"b\",\"value\":2}]}\n",
        )
        .unwrap();
        let engine = JetroEngine::new();
        let mut out = Vec::new();

        super::run_ndjson_rev(&engine, &path, "attrs.map([@.key, @.value])", &mut out).unwrap();

        assert_eq!(
            String::from_utf8(out).unwrap(),
            "[[\"b\",2]]\n[[\"a\",1]]\n"
        );
        let _ = std::fs::remove_file(path);
    }

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

    #[cfg(feature = "simd-json")]
    #[test]
    fn direct_distinct_key_classifier_rejects_escaped_strings() {
        assert!(super::raw_distinct_key_is_byte_stable(br#""plain""#));
        assert!(!super::raw_distinct_key_is_byte_stable(br#""a\u0062""#));
        assert!(!super::raw_distinct_key_is_byte_stable(br#"{"k":"v"}"#));
        assert!(super::raw_distinct_key_is_byte_stable(b"123"));
    }

    fn temp_path(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("{}-{}.ndjson", name, std::process::id()));
        path
    }
}
