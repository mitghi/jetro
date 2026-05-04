//! Top-level pipeline execution dispatcher.
//! `Pipeline::run` selects among the columnar, indexed, composed, and legacy execution paths
//! based on pipeline shape and source representation.
//! Each specialised path lives in its own sub-module; this file contains only routing logic.

use crate::{
    context::{Env, EvalError},
    value::Val,
};

use super::columnar;
use super::composed_exec;
use super::indexed_exec;
use super::legacy_exec;
use super::{Pipeline, PipelineData, Strategy};

impl Pipeline {
    /// Executes the pipeline against `root` using a freshly constructed environment.
    pub fn run(&self, root: &Val) -> Result<Val, EvalError> {
        self.run_with(root, None)
    }

    /// Executes the pipeline against `root`, optionally with a `PipelineData` cache for columnar promotion.
    pub fn run_with(&self, root: &Val, cache: Option<&dyn PipelineData>) -> Result<Val, EvalError> {
        let env = Env::new(root.clone());
        self.run_with_env(root, &env, cache)
    }

    /// Dispatches to the appropriate execution path using the pre-computed `strategy`.
    /// The indexed path is gated on `Strategy::IndexedDispatch`; columnar and composed
    /// paths remain document-dependent and are tried in priority order when applicable.
    pub fn run_with_env(
        &self,
        root: &Val,
        base_env: &Env,
        cache: Option<&dyn PipelineData>,
    ) -> Result<Val, EvalError> {
        // 1. Indexed fast-path: only valid when strategy was pre-classified as IndexedDispatch.
        if self.strategy == Strategy::IndexedDispatch {
            if let Some(out) = indexed_exec::run(self, root, base_env) {
                return out;
            }
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

        // 4. Composed stage substrate; document-dependent (source must resolve to an array).
        if let Some(out) = composed_exec::run(self, root, base_env) {
            return out;
        }

        // 5. Legacy per-shape fallback — always produces a result.
        legacy_exec::run(self, root, base_env)
    }
}
