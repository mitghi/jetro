//! Top-level pipeline execution dispatcher.
//!
//! `Pipeline::run` selects among the columnar, indexed, composed, and legacy
//! execution paths based on the pipeline shape and source representation.
//! Each specialised path is in its own sub-module; this file contains only the
//! routing logic.

use crate::{
    context::{Env, EvalError},
    value::Val,
};

use super::columnar;
use super::composed_exec;
use super::indexed_exec;
use super::legacy_exec;
use super::{Pipeline, PipelineData};

impl Pipeline {
    /// Executes the pipeline against `root` using a freshly constructed environment.
    /// Convenience wrapper around [`run_with`] with no external cache.
    pub fn run(&self, root: &Val) -> Result<Val, EvalError> {
        self.run_with(root, None)
    }

    /// Executes the pipeline against `root`, optionally supplying a `PipelineData` cache
    /// that can promote plain `Val::Arr` sources to an `ObjVec` columnar layout.
    pub fn run_with(&self, root: &Val, cache: Option<&dyn PipelineData>) -> Result<Val, EvalError> {
        let env = Env::new(root.clone());
        self.run_with_env(root, &env, cache)
    }

    /// Core dispatch: tries each specialised execution path in priority order and
    /// falls through to the legacy path if none matches.
    pub fn run_with_env(
        &self,
        root: &Val,
        base_env: &Env,
        cache: Option<&dyn PipelineData>,
    ) -> Result<Val, EvalError> {
        // 1. Indexed fast-path for single-element lookups by position.
        if let Some(out) = indexed_exec::run(self, root, base_env) {
            return out;
        }

        // 2. Columnar path using an externally supplied ObjVec promotion cache.
        if let Some(out) = columnar::run_cached(self, root, cache) {
            return out;
        }

        // 3. Columnar path without a cache (promotes in-place if the source qualifies).
        if cache.is_none() {
            if let Some(out) = columnar::run_uncached(self, root) {
                return out;
            }
        }

        // 4. Composed stage substrate for pipelines that have been fully lowered.
        if let Some(out) = composed_exec::run(self, root, base_env) {
            return out;
        }

        // 5. Legacy per-shape fallback — always produces a result.
        legacy_exec::run(self, root, base_env)
    }
}
