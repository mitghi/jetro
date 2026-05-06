//! Frontend: parser, AST, and chain-IR demand analysis.
//!
//! Source text enters through `parser::parse` and leaves as an `ast::Expr`.
//! `chain_ir` is a small post-AST helper used by downstream passes to reason
//! about per-step demand propagation in dotted method chains.

pub(crate) mod ast;
pub(crate) mod chain_ir;
pub(crate) mod parser;
