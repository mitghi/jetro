//! Bytecode compiler: lowers `parse::ast::Expr` to a `vm::Program`.
//!
//! `compiler` is the main entry; `passes` holds peephole and structural
//! rewrites applied to programs before they reach the VM.

pub(crate) mod compiler;
pub(crate) mod passes;
