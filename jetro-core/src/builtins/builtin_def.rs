//! `Builtin` trait — the per-method definition contract.
//!
//! Each builtin gets its own zero-sized struct implementing `Builtin`. The trait collects
//! all metadata (identity, spec, eventually runtime apply) in one place so a single struct
//! impl block fully describes the builtin. Static dispatch: `BuiltinMethod::spec()` matches
//! on the variant and calls `<TheStruct>::spec()` — monomorphised, no vtable.
//!
//! This replaces the prior split where:
//! - `BuiltinMethod::spec()` carried metadata in one giant match
//! - `builtin_registry!` macro carried names + aliases
//! - `executor_for_method` carried streaming classification
//! - `push_expr_stage` carried lowering
//!
//! After full migration, each builtin = one `impl Builtin for X` block.

use crate::value::Val;

use super::{BuiltinCancellation, BuiltinMethod, BuiltinSpec};

/// Per-method definition trait. Each `BuiltinMethod` variant has a corresponding zero-sized
/// struct in `builtins::defs::*` that implements this trait.
///
/// `METHOD`, `NAME`, and `ALIASES` are reserved for later migration phases (consumer code in
/// `builtin_registry` will derive name lookups from them); `#[allow(dead_code)]` keeps the
/// foundation intact while the spec piece is migrated first.
#[allow(dead_code)]
pub(crate) trait Builtin {
    /// The `BuiltinMethod` enum variant this struct corresponds to.
    const METHOD: BuiltinMethod;

    /// Canonical name used in source code (e.g. `"filter"`).
    const NAME: &'static str;

    /// Alternate names accepted by the parser (e.g. `&["where"]` for `Filter`).
    const ALIASES: &'static [&'static str] = &[];

    /// Returns the full `BuiltinSpec` for this method.
    fn spec() -> BuiltinSpec;

    /// Algebraic cancellation rule for this method, if any.
    /// Default `None`; override for inverse pairs (Base64/Url/Html encode/decode) and
    /// self-inverses (ReverseStr). Used by the optimizer to fuse-cancel adjacent stages.
    #[inline]
    fn cancellation() -> Option<BuiltinCancellation> {
        None
    }

    /// Element-wise pure runtime apply: takes a receiver value, returns the transformed
    /// value or `None` if the receiver type is not applicable.
    /// Default `None` (caller falls back to legacy dispatch).
    /// Override on element-wise scalar methods (Upper, Lower, Trim, encoders, ...).
    #[inline]
    fn apply_one(_recv: &Val) -> Option<Val> {
        None
    }
}
