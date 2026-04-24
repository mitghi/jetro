/// Internal value type for the v2 evaluator.
///
/// Every compound node (Arr, Obj) is `Arc`-wrapped so that `Val::clone()`
/// is a single atomic refcount bump — no deep copies during traversal.
///
/// The API boundary (`evaluate()`) converts `&serde_json::Value → Val`
/// once on entry and `Val → serde_json::Value` once on exit.
use std::borrow::Cow;
use std::sync::Arc;
use indexmap::IndexMap;
use serde::ser::{Serialize, SerializeMap, SerializeSeq, Serializer};
use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Number};

// ── Core type ─────────────────────────────────────────────────────────────────
//
// `IntVec` / `FloatVec` are columnar variants: mono-typed numeric arrays
// stored as `Vec<i64>` / `Vec<f64>` instead of `Vec<Val>`.  This cuts output
// bandwidth 3x (8B/elem vs 24B Val enum) on producers that emit homogeneous
// numeric arrays (`range()`, `accumulate` typed fast-path, `from_json` of
// all-int seq).  Consumers either handle them directly (typed aggregates,
// serde) or materialize on demand via `as_vals() -> Cow<[Val]>`.

#[derive(Clone, Debug)]
pub enum Val {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(Arc<str>),
    Arr(Arc<Vec<Val>>),
    IntVec(Arc<Vec<i64>>),
    FloatVec(Arc<Vec<f64>>),
    Obj(Arc<IndexMap<Arc<str>, Val>>),
}

// ── Constants (avoid heap allocation for common nulls) ────────────────────────

thread_local! {
    static NULL_VAL: Val = Val::Null;
}

// ── Cheap structural operations ───────────────────────────────────────────────

impl Val {
    /// O(1) field lookup — returns a clone of the child (cheap: Arc bump for Arr/Obj, copy for scalars).
    #[inline]
    pub fn get_field(&self, key: &str) -> Val {
        match self {
            Val::Obj(m) => m.get(key).cloned().unwrap_or(Val::Null),
            _           => Val::Null,
        }
    }

    /// O(1) index — returns a clone of the element (cheap: Arc bump for Arr/Obj).
    #[inline]
    pub fn get_index(&self, i: i64) -> Val {
        match self {
            Val::Arr(a) => {
                let idx = if i < 0 {
                    a.len().saturating_sub(i.unsigned_abs() as usize)
                } else { i as usize };
                a.get(idx).cloned().unwrap_or(Val::Null)
            }
            Val::IntVec(a) => {
                let idx = if i < 0 {
                    a.len().saturating_sub(i.unsigned_abs() as usize)
                } else { i as usize };
                a.get(idx).copied().map(Val::Int).unwrap_or(Val::Null)
            }
            Val::FloatVec(a) => {
                let idx = if i < 0 {
                    a.len().saturating_sub(i.unsigned_abs() as usize)
                } else { i as usize };
                a.get(idx).copied().map(Val::Float).unwrap_or(Val::Null)
            }
            _ => Val::Null,
        }
    }

    #[inline] pub fn is_null(&self)   -> bool { matches!(self, Val::Null) }
    #[inline] pub fn is_bool(&self)   -> bool { matches!(self, Val::Bool(_)) }
    #[inline] pub fn is_number(&self) -> bool { matches!(self, Val::Int(_) | Val::Float(_)) }
    #[inline] pub fn is_string(&self) -> bool { matches!(self, Val::Str(_)) }
    #[inline] pub fn is_array(&self)  -> bool { matches!(self, Val::Arr(_) | Val::IntVec(_) | Val::FloatVec(_)) }
    #[inline] pub fn is_object(&self) -> bool { matches!(self, Val::Obj(_)) }

    /// Array length — also works on columnar variants.
    #[inline]
    pub fn arr_len(&self) -> Option<usize> {
        match self {
            Val::Arr(a)      => Some(a.len()),
            Val::IntVec(a)   => Some(a.len()),
            Val::FloatVec(a) => Some(a.len()),
            _ => None,
        }
    }

    #[inline]
    pub fn as_bool(&self) -> Option<bool> {
        if let Val::Bool(b) = self { Some(*b) } else { None }
    }

    #[inline]
    pub fn as_i64(&self) -> Option<i64> {
        match self { Val::Int(n) => Some(*n), Val::Float(f) => Some(*f as i64), _ => None }
    }

    #[inline]
    pub fn as_f64(&self) -> Option<f64> {
        match self { Val::Float(f) => Some(*f), Val::Int(n) => Some(*n as f64), _ => None }
    }

    #[inline]
    pub fn as_str(&self) -> Option<&str> {
        if let Val::Str(s) = self { Some(s) } else { None }
    }

    #[inline]
    pub fn as_array(&self) -> Option<&[Val]> {
        if let Val::Arr(a) = self { Some(a) } else { None }
    }

    /// Materialize any array-like (including columnar) as a `Cow<[Val]>`.
    /// Borrowed for `Val::Arr`; owned allocation for `Val::IntVec`/`FloatVec`.
    pub fn as_vals(&self) -> Option<Cow<'_, [Val]>> {
        match self {
            Val::Arr(a)      => Some(Cow::Borrowed(a.as_slice())),
            Val::IntVec(a)   => Some(Cow::Owned(a.iter().map(|n| Val::Int(*n)).collect())),
            Val::FloatVec(a) => Some(Cow::Owned(a.iter().map(|f| Val::Float(*f)).collect())),
            _ => None,
        }
    }

    /// Force-materialize a columnar variant to `Val::Arr`.  No-op for `Arr`.
    pub fn into_arr(self) -> Val {
        match self {
            Val::IntVec(a) => {
                let v: Vec<Val> = a.iter().map(|n| Val::Int(*n)).collect();
                Val::arr(v)
            }
            Val::FloatVec(a) => {
                let v: Vec<Val> = a.iter().map(|f| Val::Float(*f)).collect();
                Val::arr(v)
            }
            other => other,
        }
    }

    pub fn as_array_mut(&mut self) -> Option<&mut Vec<Val>> {
        if let Val::Arr(a) = self { Some(Arc::make_mut(a)) } else { None }
    }

    #[inline]
    pub fn as_object(&self) -> Option<&IndexMap<Arc<str>, Val>> {
        if let Val::Obj(m) = self { Some(m) } else { None }
    }

    pub fn as_object_mut(&mut self) -> Option<&mut IndexMap<Arc<str>, Val>> {
        if let Val::Obj(m) = self { Some(Arc::make_mut(m)) } else { None }
    }

    /// Read-only get (serde_json compat shim).
    pub fn get(&self, key: &str) -> Option<&Val> {
        match self {
            Val::Obj(m) => m.get(key),
            _           => None,
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Val::Null    => "null",
            Val::Bool(_) => "bool",
            Val::Int(_) | Val::Float(_) => "number",
            Val::Str(_)  => "string",
            Val::Arr(_) | Val::IntVec(_) | Val::FloatVec(_) => "array",
            Val::Obj(_)  => "object",
        }
    }

    /// Consume self and produce a mutable Vec (clone only if shared).
    /// Columnar variants are materialized into `Vec<Val>`.
    pub fn into_vec(self) -> Option<Vec<Val>> {
        match self {
            Val::Arr(a) => Some(Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone())),
            Val::IntVec(a) => Some(a.iter().map(|n| Val::Int(*n)).collect()),
            Val::FloatVec(a) => Some(a.iter().map(|f| Val::Float(*f)).collect()),
            _ => None,
        }
    }

    /// Consume self and produce a mutable map (clone only if shared).
    pub fn into_map(self) -> Option<IndexMap<Arc<str>, Val>> {
        if let Val::Obj(m) = self {
            Some(Arc::try_unwrap(m).unwrap_or_else(|m| (*m).clone()))
        } else { None }
    }

    /// Build a Val::Arr from a Vec without an extra allocation.
    #[inline]
    pub fn arr(v: Vec<Val>) -> Self { Val::Arr(Arc::new(v)) }

    /// Build a columnar `Val::IntVec` from a `Vec<i64>`.
    #[inline]
    pub fn int_vec(v: Vec<i64>) -> Self { Val::IntVec(Arc::new(v)) }

    /// Build a columnar `Val::FloatVec` from a `Vec<f64>`.
    #[inline]
    pub fn float_vec(v: Vec<f64>) -> Self { Val::FloatVec(Arc::new(v)) }

    /// Build a Val::Obj from an IndexMap without an extra allocation.
    #[inline]
    pub fn obj(m: IndexMap<Arc<str>, Val>) -> Self { Val::Obj(Arc::new(m)) }

    /// Intern a string key.
    #[inline]
    pub fn key(s: &str) -> Arc<str> { Arc::from(s) }
}

// ── serde_json::Value ↔ Val conversions ───────────────────────────────────────

impl From<&serde_json::Value> for Val {
    fn from(v: &serde_json::Value) -> Self {
        match v {
            serde_json::Value::Null      => Val::Null,
            serde_json::Value::Bool(b)   => Val::Bool(*b),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() { Val::Int(i) }
                else { Val::Float(n.as_f64().unwrap_or(0.0)) }
            }
            serde_json::Value::String(s) => Val::Str(Arc::from(s.as_str())),
            serde_json::Value::Array(a)  => {
                // Columnar fast-path: homogeneous all-int arrays become
                // `Val::IntVec`.  Saves 3x storage for big numeric docs.
                let all_i64 = !a.is_empty() && a.iter().all(|v| matches!(v,
                    serde_json::Value::Number(n) if n.is_i64()));
                if all_i64 {
                    let out: Vec<i64> = a.iter().filter_map(|v|
                        if let serde_json::Value::Number(n) = v { n.as_i64() } else { None }
                    ).collect();
                    return Val::IntVec(Arc::new(out));
                }
                Val::Arr(Arc::new(a.iter().map(Val::from).collect()))
            }
            serde_json::Value::Object(m) => Val::Obj(Arc::new(
                m.iter().map(|(k, v)| (Arc::from(k.as_str()), Val::from(v))).collect()
            )),
        }
    }
}

impl From<Val> for serde_json::Value {
    fn from(v: Val) -> Self {
        match v {
            Val::Null    => serde_json::Value::Null,
            Val::Bool(b) => serde_json::Value::Bool(b),
            Val::Int(n)  => serde_json::Value::Number(n.into()),
            Val::Float(f) => serde_json::Value::Number(
                Number::from_f64(f).unwrap_or_else(|| 0.into())
            ),
            Val::Str(s)  => serde_json::Value::String(s.to_string()),
            Val::Arr(a)  => {
                let mut out: Vec<serde_json::Value> = Vec::with_capacity(a.len());
                match Arc::try_unwrap(a) {
                    Ok(vec) => for item in vec { out.push(item.into()); }
                    Err(a)  => for item in a.iter() { out.push(item.clone().into()); }
                }
                serde_json::Value::Array(out)
            }
            Val::IntVec(a) => {
                let out: Vec<serde_json::Value> =
                    a.iter().map(|n| serde_json::Value::Number((*n).into())).collect();
                serde_json::Value::Array(out)
            }
            Val::FloatVec(a) => {
                let out: Vec<serde_json::Value> = a.iter().map(|f|
                    serde_json::Value::Number(Number::from_f64(*f).unwrap_or_else(|| 0.into()))
                ).collect();
                serde_json::Value::Array(out)
            }
            Val::Obj(m)  => {
                let mut map: Map<String, serde_json::Value> = Map::with_capacity(m.len());
                match Arc::try_unwrap(m) {
                    Ok(im) => for (k, v) in im { map.insert(k.to_string(), v.into()); }
                    Err(m) => for (k, v) in m.iter() { map.insert(k.to_string(), v.clone().into()); }
                }
                serde_json::Value::Object(map)
            }
        }
    }
}

// ── Lazy serde adapter ────────────────────────────────────────────────────────
//
// `ValRef<'a>` serialises a borrowed `Val` directly to the output byte
// stream, skipping the intermediate `serde_json::Value` tree that
// `From<Val> for serde_json::Value` builds.  For Obj-heavy results this
// elides N key-string clones + the `Map` rebuild per call.
//
// Non-finite floats (NaN/±Inf) are coerced to `0` to match the existing
// `From<Val>` impl, which falls back to `0.into()` when `Number::from_f64`
// returns `None`.

pub struct ValRef<'a>(pub &'a Val);

impl<'a> Serialize for ValRef<'a> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self.0 {
            Val::Null    => s.serialize_unit(),
            Val::Bool(b) => s.serialize_bool(*b),
            Val::Int(n)  => s.serialize_i64(*n),
            Val::Float(f) => {
                if f.is_finite() { s.serialize_f64(*f) } else { s.serialize_i64(0) }
            }
            Val::Str(v)  => s.serialize_str(v),
            Val::Arr(a)  => {
                let mut seq = s.serialize_seq(Some(a.len()))?;
                for item in a.iter() { seq.serialize_element(&ValRef(item))?; }
                seq.end()
            }
            Val::IntVec(a) => {
                let mut seq = s.serialize_seq(Some(a.len()))?;
                for n in a.iter() { seq.serialize_element(n)?; }
                seq.end()
            }
            Val::FloatVec(a) => {
                let mut seq = s.serialize_seq(Some(a.len()))?;
                for f in a.iter() {
                    if f.is_finite() { seq.serialize_element(f)?; }
                    else { seq.serialize_element(&0i64)?; }
                }
                seq.end()
            }
            Val::Obj(m)  => {
                let mut map = s.serialize_map(Some(m.len()))?;
                for (k, v) in m.iter() {
                    map.serialize_entry(k.as_ref(), &ValRef(v))?;
                }
                map.end()
            }
        }
    }
}

impl Val {
    /// Serialise `self` as JSON bytes without building an intermediate
    /// `serde_json::Value` tree.  Preferred over `serde_json::to_vec(&Value::from(val))`
    /// for Obj-heavy results.
    pub fn to_json_vec(&self) -> Vec<u8> {
        serde_json::to_vec(&ValRef(self)).unwrap_or_default()
    }

    /// Stream `self` as JSON into an `io::Write` sink.
    pub fn write_json<W: std::io::Write>(&self, w: W) -> serde_json::Result<()> {
        serde_json::to_writer(w, &ValRef(self))
    }

    /// Parse JSON text into `Val` directly — one pass, no intermediate
    /// `serde_json::Value` tree.  Preferred over the
    /// `serde_json::from_str -> serde_json::Value -> Val::from` round-trip
    /// for hot `from_json` paths.
    pub fn from_json_str(s: &str) -> serde_json::Result<Val> {
        let mut de = serde_json::Deserializer::from_str(s);
        let v = Val::deserialize(&mut de)?;
        de.end()?;
        Ok(v)
    }

    pub fn from_json_slice(b: &[u8]) -> serde_json::Result<Val> {
        let mut de = serde_json::Deserializer::from_slice(b);
        let v = Val::deserialize(&mut de)?;
        de.end()?;
        Ok(v)
    }

    /// Parse JSON via simd-json (SIMD-accelerated structural scanner).
    /// Requires the `simd-json` feature.  Input bytes are mutated in-place
    /// by the simd-json parser — caller must own a writable buffer.
    /// Falls back to `from_json_slice` on parse error from the simd path.
    #[cfg(feature = "simd-json")]
    pub fn from_json_simd(bytes: &mut [u8]) -> Result<Val, String> {
        simd_json::serde::from_slice::<Val>(bytes).map_err(|e| e.to_string())
    }
}

// ── Direct Deserialize ────────────────────────────────────────────────────────
//
// Avoids the intermediate `serde_json::Value` tree that `b_from_json` used
// to build.  Preserves order by using IndexMap directly in the visitor.

struct ValVisitor;

impl<'de> Visitor<'de> for ValVisitor {
    type Value = Val;

    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("any JSON value")
    }

    fn visit_unit<E: de::Error>(self) -> Result<Val, E> { Ok(Val::Null) }
    fn visit_none<E: de::Error>(self) -> Result<Val, E> { Ok(Val::Null) }
    fn visit_some<D: Deserializer<'de>>(self, d: D) -> Result<Val, D::Error> {
        Val::deserialize(d)
    }
    fn visit_bool<E: de::Error>(self, b: bool) -> Result<Val, E> { Ok(Val::Bool(b)) }
    fn visit_i64<E: de::Error>(self, n: i64) -> Result<Val, E>  { Ok(Val::Int(n)) }
    fn visit_u64<E: de::Error>(self, n: u64) -> Result<Val, E>  {
        if n <= i64::MAX as u64 { Ok(Val::Int(n as i64)) } else { Ok(Val::Float(n as f64)) }
    }
    fn visit_f64<E: de::Error>(self, f: f64) -> Result<Val, E>  { Ok(Val::Float(f)) }
    fn visit_str<E: de::Error>(self, s: &str) -> Result<Val, E> { Ok(Val::Str(Arc::from(s))) }
    fn visit_string<E: de::Error>(self, s: String) -> Result<Val, E> { Ok(Val::Str(Arc::from(s.as_str()))) }
    fn visit_borrowed_str<E: de::Error>(self, s: &'de str) -> Result<Val, E> { Ok(Val::Str(Arc::from(s))) }

    fn visit_seq<A: SeqAccess<'de>>(self, mut a: A) -> Result<Val, A::Error> {
        // Speculative all-i64 fast path: collect into a `Vec<i64>` while
        // every element stays `Val::Int`; on first non-int, migrate to
        // `Vec<Val>` and keep going.  Emits columnar `Val::IntVec` for
        // homogeneous numeric arrays (saves 3x storage and write bandwidth
        // on big int payloads).
        let cap = a.size_hint().unwrap_or(0);
        let mut ints: Vec<i64> = Vec::with_capacity(cap);
        let mut fallback: Option<Vec<Val>> = None;
        while let Some(item) = a.next_element::<Val>()? {
            match (fallback.as_mut(), item) {
                (None, Val::Int(n)) => ints.push(n),
                (None, other) => {
                    let mut v: Vec<Val> = Vec::with_capacity(ints.len() + cap);
                    for n in &ints { v.push(Val::Int(*n)); }
                    v.push(other);
                    fallback = Some(v);
                }
                (Some(v), other) => v.push(other),
            }
        }
        Ok(match fallback {
            Some(v) => Val::arr(v),
            None    => Val::IntVec(Arc::new(ints)),
        })
    }
    fn visit_map<A: MapAccess<'de>>(self, mut m: A) -> Result<Val, A::Error> {
        let mut out: IndexMap<Arc<str>, Val> =
            IndexMap::with_capacity(m.size_hint().unwrap_or(0));
        while let Some((k, v)) = m.next_entry::<String, Val>()? {
            out.insert(Arc::from(k.as_str()), v);
        }
        Ok(Val::obj(out))
    }
}

impl<'de> serde::Deserialize<'de> for Val {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Val, D::Error> {
        d.deserialize_any(ValVisitor)
    }
}

#[cfg(test)]
mod valref_tests {
    use super::*;

    fn roundtrip(js: &str) {
        let v: serde_json::Value = serde_json::from_str(js).unwrap();
        let val = Val::from(&v);
        let via_tree = serde_json::to_vec(&serde_json::Value::from(val.clone())).unwrap();
        let via_ref  = val.to_json_vec();
        // Both paths must serialise to byte-identical JSON (IndexMap and
        // serde_json::Map both preserve insertion order).
        assert_eq!(via_tree, via_ref, "payload: {js}");
    }

    #[test]
    fn valref_parity_scalars()  { roundtrip(r#"null"#); roundtrip(r#"true"#); roundtrip(r#"42"#); roundtrip(r#"3.14"#); roundtrip(r#""hi""#); }

    #[test]
    fn valref_parity_array()    { roundtrip(r#"[1,2,3,"x",null,true,4.5]"#); }

    #[test]
    fn valref_parity_object()   { roundtrip(r#"{"a":1,"b":"x","c":[1,2],"d":{"nested":true}}"#); }

    #[test]
    fn valref_parity_deep() {
        roundtrip(r#"{"orders":[{"id":1,"items":[{"sku":"A","qty":2}]},{"id":2,"items":[]}]}"#);
    }

    #[test]
    fn valref_nan_inf_coerced_to_zero() {
        let val = Val::Float(f64::NAN);
        assert_eq!(val.to_json_vec(), b"0");
        let val = Val::Float(f64::INFINITY);
        assert_eq!(val.to_json_vec(), b"0");
    }
}

// ── PartialEq for comparison ──────────────────────────────────────────────────

impl PartialEq for Val {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Val::Null,    Val::Null)    => true,
            (Val::Bool(a), Val::Bool(b)) => a == b,
            (Val::Str(a),  Val::Str(b))  => a == b,
            (Val::Int(a),  Val::Int(b))  => a == b,
            (Val::Float(a), Val::Float(b)) => a == b,
            (Val::Int(a),  Val::Float(b)) => (*a as f64) == *b,
            (Val::Float(a), Val::Int(b))  => *a == (*b as f64),
            (Val::IntVec(a), Val::IntVec(b)) => a == b,
            (Val::FloatVec(a), Val::FloatVec(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for Val {}

impl std::hash::Hash for Val {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Val::Null       => 0u8.hash(state),
            Val::Bool(b)    => { 1u8.hash(state); b.hash(state); }
            Val::Int(n)     => { 2u8.hash(state); n.hash(state); }
            Val::Float(f)   => { 2u8.hash(state); f.to_bits().hash(state); }
            Val::Str(s)     => { 3u8.hash(state); s.hash(state); }
            Val::Arr(a)     => { 4u8.hash(state); (Arc::as_ptr(a) as usize).hash(state); }
            Val::IntVec(a)  => { 4u8.hash(state); (Arc::as_ptr(a) as usize).hash(state); }
            Val::FloatVec(a) => { 4u8.hash(state); (Arc::as_ptr(a) as usize).hash(state); }
            Val::Obj(m)     => { 5u8.hash(state); (Arc::as_ptr(m) as usize).hash(state); }
        }
    }
}

// ── Display ───────────────────────────────────────────────────────────────────

impl std::fmt::Display for Val {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Val::Null      => write!(f, "null"),
            Val::Bool(b)   => write!(f, "{}", b),
            Val::Int(n)    => write!(f, "{}", n),
            Val::Float(fl) => write!(f, "{}", fl),
            Val::Str(s)    => write!(f, "{}", s),
            other => {
                let bytes = serde_json::to_vec(&ValRef(other)).unwrap_or_default();
                f.write_str(std::str::from_utf8(&bytes).unwrap_or(""))
            }
        }
    }
}
