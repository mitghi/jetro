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
        // Step 3d Phase 5 — IndexedDispatch.  When stages are all 1:1
        // (`Map`, `Identity`) and sink is positional (First/Last), pull
        // the target element from the source by index, run chain once,
        // return.  O(1) work for `$.books.map(@.x).first()` shape.
        if let Some(out) = indexed_exec::run(self, root, base_env) {
            return out;
        }

        // Phase 3 columnar fast path — runs before per-row loop.
        // Critical for Q12/Q15-class queries: ObjVec promotion +
        // typed-column slot kernels reach native parity. Composed
        // path runs AFTER, as fallback for the per-row generic case.
        if let Some(out) = columnar::run_cached(self, root, cache) {
            return out;
        }
        // Fall back to legacy try_columnar (no cache).
        if cache.is_none() {
            if let Some(out) = columnar::run_uncached(self, root) {
                return out;
            }
        }

        // Layer B — composed-Cow Stage chain. Opt-in under
        // `JETRO_COMPOSED=1`. Replaces the legacy per-row loop for
        // pipelines whose shape composed handles. Decomposes fused
        // Sinks (NumMap/NumFilterMap/CountIf/etc.) into base Stage +
        // base Sink at entry — composition handles the rest.
        if composed_path_enabled() {
            if let Some(out) = composed_exec::run(self, root, base_env) {
                return out;
            }
        }

        legacy_exec::run(self, root, base_env)
    }
}
