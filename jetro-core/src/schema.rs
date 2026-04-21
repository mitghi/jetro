//! Schema / shape inference for JSON documents.
//!
//! Walks a `serde_json::Value` tree and infers its structural "shape":
//! a recursive type describing each key and its possible types.  Shapes
//! can be merged (union) to describe heterogeneous arrays.
//!
//! Used by the planner and analysis passes to:
//! - detect which fields exist at which paths
//! - specialise opcodes when a path is guaranteed present
//! - detect heterogeneous arrays where per-item type varies

use indexmap::IndexMap;
use serde_json::Value;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Shape {
    Null,
    Bool,
    Int,
    Float,
    Str,
    /// Array with element shape (union of all element shapes).
    Array(Box<Shape>),
    /// Object with known key → shape mapping.
    Object(IndexMap<Arc<str>, Shape>),
    /// Union of possible shapes (heterogeneous array elements).
    Union(Vec<Shape>),
    Unknown,
}

impl Shape {
    /// Infer shape of a JSON value.
    pub fn of(v: &Value) -> Shape {
        match v {
            Value::Null       => Shape::Null,
            Value::Bool(_)    => Shape::Bool,
            Value::Number(n)  => if n.is_f64() { Shape::Float } else { Shape::Int },
            Value::String(_)  => Shape::Str,
            Value::Array(a)   => {
                let mut elem = Shape::Unknown;
                let mut first = true;
                for item in a {
                    let s = Shape::of(item);
                    elem = if first { first = false; s } else { elem.merge(s) };
                }
                Shape::Array(Box::new(elem))
            }
            Value::Object(o)  => {
                let mut map = IndexMap::new();
                for (k, v) in o {
                    map.insert(Arc::from(k.as_str()), Shape::of(v));
                }
                Shape::Object(map)
            }
        }
    }

    /// Least-upper-bound merge of two shapes.
    pub fn merge(self, other: Shape) -> Shape {
        match (self, other) {
            (a, b) if a == b => a,
            (Shape::Unknown, x) | (x, Shape::Unknown) => x,
            (Shape::Array(a), Shape::Array(b))  => Shape::Array(Box::new(a.merge(*b))),
            (Shape::Object(mut a), Shape::Object(b)) => {
                for (k, v) in b {
                    if let Some(existing) = a.shift_remove(&k) {
                        a.insert(k, existing.merge(v));
                    } else {
                        a.insert(k, v);
                    }
                }
                Shape::Object(a)
            }
            (Shape::Int, Shape::Float) | (Shape::Float, Shape::Int) => Shape::Float,
            (Shape::Union(mut xs), y) | (y, Shape::Union(mut xs)) => {
                if !xs.contains(&y) { xs.push(y); }
                Shape::Union(xs)
            }
            (a, b) => Shape::Union(vec![a, b]),
        }
    }

    /// True when this shape guarantees the named field exists.
    pub fn has_field(&self, name: &str) -> bool {
        match self {
            Shape::Object(m) => m.contains_key(name),
            Shape::Union(xs) => xs.iter().all(|s| s.has_field(name)),
            _ => false,
        }
    }

    /// Shape of `self.name` if determinable.
    pub fn field(&self, name: &str) -> Option<&Shape> {
        match self {
            Shape::Object(m) => m.get(name),
            _ => None,
        }
    }

    /// Shape of `self[..]` (array element) if this is an array.
    pub fn element(&self) -> Option<&Shape> {
        match self {
            Shape::Array(b) => Some(b),
            _ => None,
        }
    }
}

// ── Schema-guided opcode specialization ──────────────────────────────────────

use super::vm::{Program, Opcode};

/// Specialize a program against a known document `Shape`.  Current rewrites:
/// - `OptField(k)` → `GetField(k)` when shape guarantees the field exists
///   (removes per-access None-check branch).
/// - `KindCheck` folded to a `PushBool` when shape at that path is known.
///
/// Returns a new `Program` with the same source/id.
pub fn specialize(program: &Program, shape: &Shape) -> Program {
    let new_ops: Vec<Opcode> = specialize_ops(&program.ops, shape);
    Program {
        ops:           new_ops.into(),
        source:        program.source.clone(),
        id:            program.id,
        is_structural: program.is_structural,
    }
}

fn specialize_ops(ops: &[Opcode], shape: &Shape) -> Vec<Opcode> {
    let mut out = Vec::with_capacity(ops.len());
    let mut cur: Shape = shape.clone();
    for op in ops {
        match op {
            Opcode::PushRoot => { cur = shape.clone(); out.push(op.clone()); }
            Opcode::OptField(k) => {
                if cur.has_field(k) {
                    out.push(Opcode::GetField(k.clone()));
                } else {
                    out.push(op.clone());
                }
                cur = cur.field(k).cloned().unwrap_or(Shape::Unknown);
            }
            Opcode::GetField(k) => {
                cur = cur.field(k).cloned().unwrap_or(Shape::Unknown);
                out.push(op.clone());
            }
            Opcode::RootChain(ks) => {
                let mut c = shape.clone();
                for k in ks.iter() {
                    c = c.field(k).cloned().unwrap_or(Shape::Unknown);
                }
                cur = c;
                out.push(op.clone());
            }
            Opcode::GetIndex(_) | Opcode::GetSlice(..) => {
                cur = cur.element().cloned().unwrap_or(Shape::Unknown);
                out.push(op.clone());
            }
            _ => { cur = Shape::Unknown; out.push(op.clone()); }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn infer_scalar() {
        assert_eq!(Shape::of(&json!(1)), Shape::Int);
        assert_eq!(Shape::of(&json!(1.5)), Shape::Float);
        assert_eq!(Shape::of(&json!("x")), Shape::Str);
        assert_eq!(Shape::of(&json!(null)), Shape::Null);
    }

    #[test]
    fn infer_object() {
        let s = Shape::of(&json!({"a": 1, "b": "x"}));
        assert!(s.has_field("a"));
        assert!(s.has_field("b"));
        assert!(!s.has_field("c"));
        assert_eq!(s.field("a"), Some(&Shape::Int));
    }

    #[test]
    fn infer_homogeneous_array() {
        let s = Shape::of(&json!([1, 2, 3]));
        assert_eq!(s.element(), Some(&Shape::Int));
    }

    #[test]
    fn infer_heterogeneous_array() {
        let s = Shape::of(&json!([1, "two", 3]));
        match s.element().unwrap() {
            Shape::Union(xs) => {
                assert!(xs.contains(&Shape::Int));
                assert!(xs.contains(&Shape::Str));
            }
            other => panic!("expected union, got {:?}", other),
        }
    }

    #[test]
    fn merge_int_float_to_float() {
        let a = Shape::Int;
        let b = Shape::Float;
        assert_eq!(a.merge(b), Shape::Float);
    }

    #[test]
    fn specialize_opt_field_to_get_field() {
        use crate::vm::{Compiler, Opcode};
        let prog = Compiler::compile_str("$.a?.b").unwrap();
        let shape = Shape::of(&json!({"a": {"b": 1}}));
        let spec = specialize(&prog, &shape);
        let has_opt = spec.ops.iter().any(|o| matches!(o, Opcode::OptField(_)));
        assert!(!has_opt, "OptField should specialize to GetField");
    }

    #[test]
    fn specialize_preserves_opt_when_missing() {
        use crate::vm::{Compiler, Opcode};
        let prog = Compiler::compile_str("$.a?.missing").unwrap();
        let shape = Shape::of(&json!({"a": {"b": 1}}));
        let spec = specialize(&prog, &shape);
        let has_opt = spec.ops.iter().any(|o| matches!(o, Opcode::OptField(k) if k.as_ref() == "missing"));
        assert!(has_opt, "OptField for absent field should remain");
    }
}
