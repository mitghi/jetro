//! Engine-scoped JSON key interner.
//!
//! Object keys repeat across rows of a JSON document, so the parser
//! routes every key through this cache to share one `Arc<str>` per
//! distinct key. A `KeyCache` instance is engine-owned (held by
//! [`crate::JetroEngine`]) so per-engine isolation is possible without
//! threading the cache through every signature; trait-impl ingest
//! paths fall back to the per-thread [`default_cache`].

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

const KEY_CAP: usize = 4096;

/// Shared key-interning state. Cheap to clone (single `Arc` bump).
pub struct KeyCache {
    map: RwLock<HashMap<Box<str>, Arc<str>>>,
}

impl KeyCache {
    /// Allocate a fresh, empty cache wrapped in `Arc` for shared ownership.
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            map: RwLock::new(HashMap::with_capacity(64)),
        })
    }

    /// Return a shared `Arc<str>` for `k`, reusing a cached entry when
    /// possible. Falls back to a fresh allocation once the cache reaches
    /// `KEY_CAP` entries (preventing unbounded growth on docs with
    /// adversarial unique keys).
    #[inline]
    pub fn intern(&self, k: &str) -> Arc<str> {
        if let Ok(m) = self.map.read() {
            if let Some(a) = m.get(k) {
                return Arc::clone(a);
            }
        }
        let mut m = match self.map.write() {
            Ok(g) => g,
            Err(_) => return Arc::<str>::from(k),
        };
        if let Some(a) = m.get(k) {
            return Arc::clone(a);
        }
        if m.len() >= KEY_CAP {
            return Arc::<str>::from(k);
        }
        let a: Arc<str> = Arc::<str>::from(k);
        m.insert(k.into(), Arc::clone(&a));
        a
    }

    /// Drop every cached entry. Used by `JetroEngine::clear_cache`.
    pub fn clear(&self) {
        if let Ok(mut m) = self.map.write() {
            m.clear();
        }
    }
}

static DEFAULT_KEY_CACHE: OnceLock<Arc<KeyCache>> = OnceLock::new();

/// Process-wide key cache used by callers without an engine handle: the
/// trait impls `From<serde_json::Value> for Val` and the standalone
/// `Jetro::from_bytes` path. Engine-aware ingest goes through the
/// engine's own [`KeyCache`] instead.
#[inline]
pub fn default_cache() -> &'static Arc<KeyCache> {
    DEFAULT_KEY_CACHE.get_or_init(KeyCache::new)
}
