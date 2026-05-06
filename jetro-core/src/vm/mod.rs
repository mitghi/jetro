//! Bytecode compiler and stack-machine VM for Jetro expressions.
//!
//! Split into two submodules:
//! - [`opcode`] — pure-data definitions (`Opcode`, `Program`, `Compiled*`,
//!   `FieldChainData`, comprehension specs, patch ops) plus the small
//!   helpers that operate only on those structures.
//! - [`exec`] — the `VM` struct, its caches, the execution loop, and all
//!   runtime helpers that consume opcodes.

pub(crate) mod exec;
pub(crate) mod opcode;

pub(crate) use exec::*;
pub(crate) use opcode::*;
