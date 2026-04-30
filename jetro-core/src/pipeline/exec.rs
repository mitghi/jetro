use crate::{
    context::{Env, EvalError},
    value::Val,
};

use super::columnar;
use super::composed_exec;
use super::indexed_exec;
use super::legacy_exec;
use super::{composed_path_enabled, Pipeline, PipelineData};

impl Pipeline {
    /// Execute the pipeline against `root`, returning the sink.s
    /// produced [`Val`].
    pub fn run(&self, root: &Val) -> Result<Val, EvalError> {
        self.run_with(root, None)
    }

    /// Execute with an optional ObjVec promotion cache.  When `cache`
    /// is `Some`, the pipeline consults it before resolving sources;
    /// uniform-shape `Val::Arr<Val::Obj>` arrays are promoted to
    /// `Val::ObjVec` once and reused on every subsequent call.  Cost
    /// O(N×K) on first promotion, O(1) on hit.  Empty cache (`None`)
    /// matches legacy `run` semantics.
    pub fn run_with(&self, root: &Val, cache: Option<&dyn PipelineData>) -> Result<Val, EvalError> {
        let env = Env::new(root.clone());
        self.run_with_env(root, &env, cache)
    }

    /// Execute with caller-provided lexical environment. This is used
    /// by physical `let` / object plans so pipeline stage programs can
    /// see variables bound outside the pipeline.
    pub fn run_with_env(
        &self,
        root: &Val,
        base_env: &Env,
        cache: Option<&dyn PipelineData>,
    ) -> Result<Val, EvalError> {
        // O(1) route for positional sinks over indexable 1:1 chains.
        if let Some(out) = indexed_exec::run(self, root, base_env) {
            return out;
        }

        // Columnar route for typed lanes and promoted ObjVec slot kernels.
        if let Some(out) = columnar::run_cached(self, root, cache) {
            return out;
        }

        // Uncached columnar route for already-columnar sources.
        if cache.is_none() {
            if let Some(out) = columnar::run_uncached(self, root) {
                return out;
            }
        }

        // Generic composed route for supported streaming/barrier chains.
        if composed_path_enabled() {
            if let Some(out) = composed_exec::run(self, root, base_env) {
                return out;
            }
        }

        // Semantic fallback for every shape not handled above.
        legacy_exec::run(self, root, base_env)
    }
}
