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
mod patch_fusion_soundness;
#[cfg(test)]
mod examples;
#[cfg(test)]
mod regression;
