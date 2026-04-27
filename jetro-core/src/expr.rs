//! Typed expression values.
//!
//! [`Expr<T>`] wraps a Jetro expression string together with a phantom
//! output type `T`.  The expression is parse-checked at construction
//! time so invalid syntax fails fast rather than at first evaluation.
//!
//! The phantom type carries the caller's intent about the result
//! shape.  It is not enforced at compile time (Jetro has no static
//! schema today), but it drives the typed accessors on [`Bucket`] and
//! similar helpers so that callers never have to hand-write
//! `.as_i64().unwrap()` chains.
//!
//! # Composition
//!
//! `Expr<T>` supports a pipeline operator that mirrors Jetro's own `|`
//! syntax.  `a | b` produces a new expression whose source is
//! `"(a) | (b)"`, letting callers build complex queries from reusable
//! fragments.

use std::marker::PhantomData;
use std::ops::BitOr;

use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::parser;
use crate::vm::VM;
use crate::Error;

/// A parse-checked Jetro expression with a phantom output type.
///
/// The phantom `T` captures the caller's intent about the result
/// shape and is used by typed accessors (e.g. `Bucket::get_as`) to
/// drive `serde` deserialisation.  It is *not* statically verified —
/// constructing an `Expr<u64>` from an expression that yields a
/// string is possible; the mismatch surfaces at evaluation time as a
/// deserialisation error.
#[derive(Debug, Clone)]
pub struct Expr<T> {
    src: String,
    _marker: PhantomData<fn() -> T>,
}

impl<T> Expr<T> {
    /// Parse and wrap `src`.  Returns [`Error::Parse`] if the
    /// expression is not syntactically valid.
    pub fn new<S: Into<String>>(src: S) -> Result<Self, Error> {
        let src = src.into();
        parser::parse(&src)?;
        Ok(Self {
            src,
            _marker: PhantomData,
        })
    }

    /// Raw source text — useful for storing in an [`ExprBucket`] or
    /// for debugging.
    pub fn as_str(&self) -> &str {
        &self.src
    }

    /// Discard the phantom output type.  Rarely needed; `cast::<U>`
    /// is usually what callers want.
    pub fn into_string(self) -> String {
        self.src
    }

    /// Re-tag the expression with a different phantom output type.
    /// No reparse; cheap.
    pub fn cast<U>(self) -> Expr<U> {
        Expr {
            src: self.src,
            _marker: PhantomData,
        }
    }
}

impl<T: DeserializeOwned> Expr<T> {
    /// Evaluate `self` against `doc`, returning a typed value.
    ///
    /// Goes through a fresh VM — for repeated evaluations prefer
    /// [`Expr::eval_with`] so caches accumulate.
    pub fn eval(&self, doc: &Value) -> Result<T, Error> {
        let raw = VM::new().run_str(&self.src, doc)?;
        serde_json::from_value(raw).map_err(|e| Error::Eval(crate::EvalError(e.to_string())))
    }

    /// Evaluate with a caller-supplied VM so its compile and
    /// resolution caches are shared across calls.
    pub fn eval_with(&self, vm: &mut VM, doc: &Value) -> Result<T, Error> {
        let raw = vm.run_str(&self.src, doc)?;
        serde_json::from_value(raw).map_err(|e| Error::Eval(crate::EvalError(e.to_string())))
    }
}

impl Expr<Value> {
    /// Evaluate and return the raw [`Value`] without deserialisation.
    pub fn eval_raw(&self, doc: &Value) -> Result<Value, Error> {
        Ok(VM::new().run_str(&self.src, doc)?)
    }
}

/// `a | b` → expression source `"(a) | (b)"`.  The resulting type is
/// `Expr<U>` since the right-hand side determines the output shape.
impl<T, U> BitOr<Expr<U>> for Expr<T> {
    type Output = Expr<U>;
    fn bitor(self, rhs: Expr<U>) -> Expr<U> {
        Expr {
            src: format!("({}) | ({})", self.src, rhs.src),
            _marker: PhantomData,
        }
    }
}

impl<T> AsRef<str> for Expr<T> {
    fn as_ref(&self) -> &str {
        &self.src
    }
}

impl<T> std::fmt::Display for Expr<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.src)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_ok() {
        let e: Expr<i64> = Expr::new("$.x.len()").unwrap();
        assert_eq!(e.as_str(), "$.x.len()");
    }

    #[test]
    fn parse_err() {
        let e: Result<Expr<i64>, _> = Expr::new("$$$ not valid");
        assert!(e.is_err());
    }

    #[test]
    fn eval_typed() {
        let e: Expr<i64> = Expr::new("$.xs.len()").unwrap();
        let n = e.eval(&json!({"xs": [1, 2, 3]})).unwrap();
        assert_eq!(n, 3);
    }

    #[test]
    fn eval_vec() {
        let e: Expr<Vec<String>> = Expr::new("$.users.map(name)").unwrap();
        let names = e
            .eval(&json!({
                "users": [{"name":"a"}, {"name":"b"}]
            }))
            .unwrap();
        assert_eq!(names, vec!["a", "b"]);
    }

    #[test]
    fn pipe_compose() {
        let a: Expr<Value> = Expr::new("$.books").unwrap();
        let b: Expr<Vec<Value>> = Expr::new("@.filter(price > 10)").unwrap();
        let piped = a | b;
        assert_eq!(piped.as_str(), "($.books) | (@.filter(price > 10))");
    }

    #[test]
    fn cast_keeps_src() {
        let e: Expr<i64> = Expr::new("$.n").unwrap();
        let s = e.clone().cast::<String>();
        assert_eq!(s.as_str(), e.as_str());
    }
}
