//! Top-level test module: groups feature-specific suites under one parent so
//! `cargo test tests::` continues to discover everything.
//!
//! Splits:
//! - `regression` — the original mixed-feature test corpus.
//! - `chain_write` — `patch $ { ... }` and chain-style write semantics.
//! - `deep_search` — `$..find` / `simd_scan` / route-C fallthrough.
//! - `common` — shared helpers (`vm_query`, fixture builders).

#[cfg(test)]
pub(crate) mod common;

#[cfg(test)]
mod chain_write;
#[cfg(test)]
mod deep_search;
#[cfg(test)]
mod patch_fusion_phase_c;
#[cfg(test)]
mod patch_fusion_phase_e;
#[cfg(test)]
mod patch_fusion_soundness;
#[cfg(test)]
mod examples;
#[cfg(test)]
mod pattern_match;
#[cfg(test)]
mod regression;
#[cfg(test)]
mod dyn_index_lambda;
#[cfg(test)]
mod lambda_forms;
#[cfg(test)]
mod neq_grammar;
#[cfg(test)]
mod comprehensions;
#[cfg(test)]
mod grammar_extensions;
#[cfg(test)]
mod strslice_arith;
#[cfg(test)]
mod entries_wrap;
#[cfg(test)]
mod builtin_migrations;
#[cfg(test)]
mod v0_5_5_quickfixes;
#[cfg(test)]
mod tape_parity;
#[cfg(test)]
mod has_probe;
