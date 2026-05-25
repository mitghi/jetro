//! Coverage for cross-product `+` over `Val::Str` and `Val::StrSlice`.
//!
//! Path-rooted accesses like `$.user.name` historically returned
//! `Val::StrSlice` (a borrowed view into the simd-json tape) for which the
//! `+` operator only matched `(Str, Str)` — meaning `$.user.first + "-"`
//! errored with "+ not supported between these types".
//!
//! The fix accepts any pairing of `Str` / `StrSlice`, allocating once into
//! a fresh `Val::Str`. Numeric and array concat hot paths are preserved.

use super::common::vm_query;
use serde_json::json;

fn user() -> serde_json::Value {
    json!({"user": {"first": "Ada", "last": "Lovelace", "email": "ada@x.com"}})
}

#[test]
fn path_str_plus_literal() {
    assert_eq!(
        vm_query(r#"$.user.first + "-""#, &user()).unwrap(),
        json!("Ada-")
    );
}

#[test]
fn literal_plus_path_str() {
    assert_eq!(
        vm_query(r#""prefix-" + $.user.first"#, &user()).unwrap(),
        json!("prefix-Ada")
    );
}

#[test]
fn path_str_plus_path_str() {
    assert_eq!(
        vm_query(r#"$.user.first + "-" + $.user.last"#, &user()).unwrap(),
        json!("Ada-Lovelace")
    );
}

#[test]
fn email_with_brackets() {
    assert_eq!(
        vm_query(r#"$.user.first + " <" + $.user.email + ">""#, &user()).unwrap(),
        json!("Ada <ada@x.com>")
    );
}

#[test]
fn fstring_interp_with_path_str() {
    // F-string interpolation goes through the same string concatenation
    // path; keep regression coverage here.
    assert_eq!(
        vm_query(r#"f"{$.user.first} {$.user.last}""#, &user()).unwrap(),
        json!("Ada Lovelace")
    );
}

#[test]
fn pure_literal_concat_unchanged() {
    // Hot path: pure literal-string add.
    assert_eq!(vm_query(r#""a" + "b""#, &json!(null)).unwrap(), json!("ab"));
}

#[test]
fn numeric_add_unchanged() {
    // Regression: numeric hot path must not regress.
    assert_eq!(vm_query("1 + 2", &json!(null)).unwrap(), json!(3));
    assert_eq!(vm_query("1.5 + 2.5", &json!(null)).unwrap(), json!(4.0));
    assert_eq!(vm_query("1 + 2.5", &json!(null)).unwrap(), json!(3.5));
}

#[test]
fn array_concat_unchanged() {
    assert_eq!(
        vm_query("[1, 2] + [3]", &json!(null)).unwrap(),
        json!([1, 2, 3])
    );
}

#[test]
fn incompatible_still_errors() {
    // Number + string still errors — no spurious coercion.
    let err = vm_query(r#"1 + "x""#, &json!(null));
    assert!(err.is_err());
}

#[test]
fn null_plus_int_errors() {
    let err = vm_query("null + 1", &json!(null));
    assert!(err.is_err());
}
