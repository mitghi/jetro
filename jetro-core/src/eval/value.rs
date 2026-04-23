/// Internal value type for the v2 evaluator.
///
/// Every compound node (Arr, Obj) is `Arc`-wrapped so that `Val::clone()`
/// is a single atomic refcount bump — no deep copies during traversal.
///
/// The API boundary (`evaluate()`) converts `&serde_json::Value → Val`
/// once on entry and `Val → serde_json::Value` once on exit.
use std::sync::Arc;
use indexmap::IndexMap;
use serde::ser::{Serialize, SerializeMap, SerializeSeq, Serializer};
use serde_json::{Map, Number};

// ── Core type ─────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub enum Val {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(Arc<str>),
    Arr(Arc<Vec<Val>>),
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
            _ => Val::Null,
        }
    }

    #[inline] pub fn is_null(&self)   -> bool { matches!(self, Val::Null) }
    #[inline] pub fn is_bool(&self)   -> bool { matches!(self, Val::Bool(_)) }
    #[inline] pub fn is_number(&self) -> bool { matches!(self, Val::Int(_) | Val::Float(_)) }
    #[inline] pub fn is_string(&self) -> bool { matches!(self, Val::Str(_)) }
    #[inline] pub fn is_array(&self)  -> bool { matches!(self, Val::Arr(_)) }
    #[inline] pub fn is_object(&self) -> bool { matches!(self, Val::Obj(_)) }

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
            Val::Arr(_)  => "array",
            Val::Obj(_)  => "object",
        }
    }

    /// Consume self and produce a mutable Vec (clone only if shared).
    pub fn into_vec(self) -> Option<Vec<Val>> {
        if let Val::Arr(a) = self {
            Some(Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone()))
        } else { None }
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
            serde_json::Value::Array(a)  => Val::Arr(Arc::new(a.iter().map(Val::from).collect())),
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
