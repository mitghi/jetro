//! Planner-facing chain operator representation.
//!
//! This IR is post-AST metadata for dotted pipeline chains. It carries stable
//! builtin identities and the small amount of argument shape needed by demand
//! propagation without tying the executor to parser syntax.

use crate::{
    builtins::registry::{BuiltinDemandArg, BuiltinId},
    builtins::BuiltinMethod,
};

/// A single operator node in the chain IR, carrying identity and demand
/// metadata for each step in a composed pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChainOp {
    /// A registered builtin with a stable `BuiltinId` and optional numeric demand argument.
    Builtin {
        /// Stable numeric ID identifying which builtin this operator represents.
        id: BuiltinId,
        /// Optional count argument used by demand-propagation for `take`/`skip`.
        demand_arg: BuiltinDemandArg,
    },
    /// A `match` expression participating in a streaming chain. Match
    /// behaves as one of two shapes depending on how the surrounding
    /// pipeline uses its result, captured here as `MatchRole`.
    Match {
        /// How the surrounding chain consumes match output.
        role: MatchRole,
    },
}

/// The role a `match` expression takes when embedded in a streaming chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchRole {
    /// Boolean predicate: arm bodies are `true` / `false`.
    Predicate,
    /// Single-value transform: every input row yields exactly one output.
    Transform,
}

impl ChainOp {
    /// Construct a `ChainOp::Builtin` from a `BuiltinMethod` with no demand argument.
    pub fn builtin(method: BuiltinMethod) -> Self {
        Self::Builtin {
            id: BuiltinId::from_method(method),
            demand_arg: BuiltinDemandArg::None,
        }
    }

    /// Construct a `ChainOp::Builtin` from a `BuiltinMethod` with a `Usize(n)` demand argument.
    pub fn builtin_usize(method: BuiltinMethod, n: usize) -> Self {
        Self::Builtin {
            id: BuiltinId::from_method(method),
            demand_arg: BuiltinDemandArg::Usize(n),
        }
    }

    /// Construct a `ChainOp::Match` for the given `role`.
    pub fn match_role(role: MatchRole) -> Self {
        Self::Match { role }
    }
}
