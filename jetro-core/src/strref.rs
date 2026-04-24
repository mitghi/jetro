//! Borrowed-slice string view that shares a parent `Arc<str>`.
//!
//! `StrRef` is a lightweight wrapper around an owning `Arc<str>` plus
//! byte-range offsets.  Cloning it is an atomic Arc bump plus two
//! word-sized copies — no heap allocation — so methods like `slice`,
//! `split.first`, `substring` can return a view into their input
//! without allocating a fresh `Arc<str>` per row.
//!
//! Invariants (enforced at construction):
//! - `start <= end <= parent.len()`.
//! - `parent[start..end]` is valid UTF-8 (the parent is, and we never
//!   split between code units — callers construct slices via
//!   `str::char_indices` or ASCII-only byte offsets).
//!
//! `as_str()` returns the view slice.  `Deref<Target=str>` /
//! `AsRef<str>` allow existing string APIs (len, chars, find, memchr,
//! etc.) to work against a `StrRef` transparently.

use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct StrRef {
    parent: Arc<str>,
    start: u32,
    end: u32,
}

impl StrRef {
    /// Construct a full view of `parent`.  Offsets = [0, parent.len()].
    #[inline]
    pub fn from_arc(parent: Arc<str>) -> Self {
        let end = parent.len() as u32;
        Self { parent, start: 0, end }
    }

    /// Byte-range view into `parent`.  Caller must ensure the range is
    /// a valid UTF-8 boundary (checked in debug via `is_char_boundary`).
    #[inline]
    pub fn slice(parent: Arc<str>, start: usize, end: usize) -> Self {
        debug_assert!(start <= end);
        debug_assert!(end <= parent.len());
        debug_assert!(parent.is_char_boundary(start));
        debug_assert!(parent.is_char_boundary(end));
        Self {
            parent,
            start: start as u32,
            end: end as u32,
        }
    }

    #[inline] pub fn as_str(&self) -> &str {
        &self.parent[self.start as usize .. self.end as usize]
    }

    #[inline] pub fn len(&self) -> usize { (self.end - self.start) as usize }
    #[inline] pub fn is_empty(&self) -> bool { self.end == self.start }

    /// Produce an owning `Arc<str>` — allocates a fresh buffer containing
    /// the view contents.  Use only when an owning Arc is required (e.g.
    /// to insert into an `IndexMap<Arc<str>, Val>` as a key).
    #[inline]
    pub fn to_arc(&self) -> Arc<str> {
        if self.start == 0 && self.end as usize == self.parent.len() {
            Arc::clone(&self.parent)
        } else {
            Arc::<str>::from(self.as_str())
        }
    }
}

impl AsRef<str> for StrRef {
    #[inline] fn as_ref(&self) -> &str { self.as_str() }
}

impl std::ops::Deref for StrRef {
    type Target = str;
    #[inline] fn deref(&self) -> &str { self.as_str() }
}

impl std::fmt::Display for StrRef {
    #[inline] fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl PartialEq for StrRef {
    #[inline] fn eq(&self, other: &Self) -> bool { self.as_str() == other.as_str() }
}
impl Eq for StrRef {}

impl PartialEq<str> for StrRef {
    #[inline] fn eq(&self, other: &str) -> bool { self.as_str() == other }
}
impl PartialEq<&str> for StrRef {
    #[inline] fn eq(&self, other: &&str) -> bool { self.as_str() == *other }
}
impl PartialEq<Arc<str>> for StrRef {
    #[inline] fn eq(&self, other: &Arc<str>) -> bool { self.as_str() == other.as_ref() }
}

impl std::hash::Hash for StrRef {
    #[inline] fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_str().hash(state)
    }
}

impl From<Arc<str>> for StrRef {
    #[inline] fn from(a: Arc<str>) -> Self { Self::from_arc(a) }
}
impl From<&str> for StrRef {
    #[inline] fn from(s: &str) -> Self { Self::from_arc(Arc::<str>::from(s)) }
}
impl From<String> for StrRef {
    #[inline] fn from(s: String) -> Self { Self::from_arc(Arc::<str>::from(s)) }
}
