use std::sync::Arc;

use crate::{
    context::{Env, EvalError},
    value::Val,
};

use super::composed_exec;
use super::legacy_exec;
use super::{
    composed_path_enabled, select_strategy, walk_field_chain, BodyKernel, Pipeline, PipelineData,
    Position, Sink, Source, Stage, Strategy,
};

impl Pipeline {
    /// Execute the pipeline against `root`, returning the sink.s
    /// produced [`Val`].
    pub fn run(&self, root: &Val) -> Result<Val, EvalError> {
        self.run_with(root, None)
    }

    /// Decompose any fused Sink into a base `(stages, kernels, sink)`
    /// triple. Pure function — does not mutate `self`. Lets every
    /// Identity view — fused Sink variants deleted in Tier 3, so
    /// every Pipeline is already in base form. Kept as a stable name
    /// for downstream consumers (composed Val/tape runners and
    /// columnar fast paths) that built on the canonical-view abstraction.
    pub fn canonical(&self) -> (Vec<Stage>, Vec<BodyKernel>, Sink) {
        (
            self.stages.clone(),
            self.stage_kernels.clone(),
            self.sink.clone(),
        )
    }

    /// Layer B — composed-Cow Stage chain runner.
    ///
    /// Returns `Some(Ok(val))` when:
    ///   - source resolves to a `Val::Arr` (or other materialisable seq)
    ///   - sink is one of the base generic forms (Collect / Count /
    ///     Numeric(Sum/Min/Max/Avg) / First / Last)
    ///   - every stage is one of: Filter/Map/FlatMap/Take/Skip with a
    ///     borrow-form-recognised BodyKernel (FieldRead / FieldChain /
    ///     FieldCmpLit Eq / Generic-via-VM-fallback)
    ///   - no barrier stages (Sort/UniqueBy/Reverse/GroupBy) — Day 4-5
    ///
    /// Returns `None` for fused sinks (NumMap/NumFilterMap/CountIf/
    /// FilterFirst/etc.) — Tier 3 will lower those into base sink +
    /// stage chain so the composed path becomes the sole exec route.
    /// Step 3d Phase 5 — IndexedDispatch.  Generic O(1) optimisation
    /// for `(Map | identity)*` chains terminated by a positional sink
    /// (`First` / `Last`).  Pulls source[idx] only, runs the chain on
    /// that single element, returns.
    ///
    /// Falls through (`None`) when:
    /// - any stage is not 1:1 (Filter, FlatMap, Take, Skip, barriers),
    /// - sink isn't positional (Sum/Min/Max/Count/Avg/Collect),
    /// - source is not an indexable Val::Arr / typed-vec lane,
    /// - chain target index is out of bounds (returns Null via the
    ///   normal fallback so error semantics match).
    fn try_indexed_dispatch(&self, root: &Val, base_env: &Env) -> Option<Result<Val, EvalError>> {
        // Phase 5 strategy must be IndexedDispatch.
        let strategy = select_strategy(&self.stages, &self.sink);
        if strategy != Strategy::IndexedDispatch {
            return None;
        }

        // Resolve source — same rules as the generic loop.
        let recv = match &self.source {
            Source::Receiver(v) => v.clone(),
            Source::FieldChain { keys } => walk_field_chain(root, keys),
        };
        // Only indexable shapes; bail otherwise.
        let len = match &recv {
            Val::Arr(a) => a.len(),
            Val::IntVec(a) => a.len(),
            Val::FloatVec(a) => a.len(),
            Val::StrVec(a) => a.len(),
            Val::ObjVec(d) => d.nrows(),
            _ => return None,
        };

        // Compute target index from sink demand.
        let demand = self.sink.demand();
        let idx = match demand.positional? {
            Position::First => 0,
            Position::Last => len.checked_sub(1)?,
        };
        if idx >= len {
            // Out of bounds — sink::First/Last semantics return Null.
            return Some(Ok(Val::Null));
        }

        // Pull element[idx] from source.
        let elem = match &recv {
            Val::Arr(a) => a[idx].clone(),
            Val::IntVec(a) => Val::Int(a[idx]),
            Val::FloatVec(a) => Val::Float(a[idx]),
            Val::StrVec(a) => Val::Str(Arc::clone(&a[idx])),
            Val::ObjVec(d) => {
                let stride = d.stride();
                let mut m: indexmap::IndexMap<Arc<str>, Val> =
                    indexmap::IndexMap::with_capacity(stride);
                for (i, k) in d.keys.iter().enumerate() {
                    m.insert(Arc::clone(k), d.cells[idx * stride + i].clone());
                }
                Val::Obj(Arc::new(m))
            }
            _ => return None,
        };

        // Run chain on the single element.  All stages are 1:1 (Map),
        // so each apply produces exactly one Val.
        let mut vm = crate::vm::VM::new();
        let mut env = base_env.clone();
        let mut cur = elem;
        for stage in &self.stages {
            match stage {
                Stage::Map(prog) => {
                    let prev = env.swap_current(cur);
                    cur = match vm.exec_in_env(prog, &mut env) {
                        Ok(v) => v,
                        Err(e) => {
                            env.restore_current(prev);
                            return Some(Err(e));
                        }
                    };
                    env.restore_current(prev);
                }
                Stage::Builtin(call) => {
                    cur = call.apply(&cur).unwrap_or(cur);
                }
                _ => return None, // shape-check should have rejected; defensive.
            }
        }

        Some(Ok(cur))
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
        if let Some(out) = self.try_indexed_dispatch(root, base_env) {
            return out;
        }

        // Phase 3 columnar fast path — runs before per-row loop.
        // Critical for Q12/Q15-class queries: ObjVec promotion +
        // typed-column slot kernels reach native parity. Composed
        // path runs AFTER, as fallback for the per-row generic case.
        if let Some(out) = self.try_columnar_with(root, cache) {
            return out;
        }
        // Fall back to legacy try_columnar (no cache).
        if cache.is_none() {
            if let Some(out) = self.try_columnar(root) {
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
