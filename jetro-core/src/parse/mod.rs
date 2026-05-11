//! Frontend: parser and AST construction.
//!
//! Source text enters through `parser::parse` and leaves as an `ast::Expr`.

pub(crate) mod ast;
pub(crate) mod parser;
pub(crate) mod write_terminal;
