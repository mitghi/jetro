//! Shared test helpers used by every split module under `tests/`.

use crate::{compile::compiler, parse::parser, vm};
use serde_json::{json, Value};

/// Compile `expr` and run it through a fresh `VM` against `doc`.
/// Returns the collected `serde_json::Value` or a `crate::Error` for
/// either parse or evaluation failure.
pub(crate) fn vm_query(expr: &str, doc: &Value) -> Result<Value, crate::Error> {
    let ast = parser::parse(expr)?;
    let program = compiler::Compiler::compile(&ast, expr);
    let mut vm = vm::VM::new();
    Ok(vm.execute(&program, doc)?)
}

/// Canonical bookstore fixture used across many regression tests.
pub(crate) fn books() -> Value {
    json!({
        "store": {
            "books": [
                {"title": "Dune",        "price": 12.99, "rating": 4.8, "genre": "sci-fi",  "tags": ["sci-fi","classic"]},
                {"title": "Foundation",  "price":  9.99, "rating": 4.5, "genre": "sci-fi",  "tags": ["sci-fi","series"]},
                {"title": "Neuromancer", "price": 11.50, "rating": 4.2, "genre": "cyberpunk","tags": ["sci-fi","cyberpunk"]},
                {"title": "1984",        "price":  7.99, "rating": 4.6, "genre": "dystopia", "tags": ["classic","dystopia"]},
            ]
        },
        "user": {"name": "Alice", "age": 30, "score": 85}
    })
}
