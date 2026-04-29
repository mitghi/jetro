use std::sync::Arc;

use crate::{
    context::{Env, EvalError},
    value::Val,
};

use super::lower::run_compiled_map;
use super::{
    apply_item_in_env, cmp_val_total, composed_path_enabled, compute_strategies, eval_kernel,
    is_truthy, num_finalise, num_fold, select_strategy, walk_field_chain, BodyKernel, NumOp,
    Pipeline, PipelineData, Position, Sink, Source, Stage, StageStrategy, Strategy,
};

use crate::builtins::{
    chunk_apply, count_by_apply, drop_while_apply, filter_apply, filter_one, group_by_apply,
    index_by_apply, map_apply, map_one, replace_apply, slice_apply, split_apply, take_while_apply,
    take_while_one, window_apply,
};
use crate::chain_ir::PullDemand;

fn sort_buffer_by_key<F>(
    buf: Vec<Val>,
    descending: bool,
    strategy: StageStrategy,
    mut key_of: F,
) -> Result<Vec<Val>, EvalError>
where
    F: FnMut(&Val) -> Result<Val, EvalError>,
{
    let k = match strategy {
        StageStrategy::SortTopK(k) | StageStrategy::SortBottomK(k) => Some(k),
        StageStrategy::Default => None,
    };
    let keep_largest = match strategy {
        StageStrategy::SortTopK(_) => descending,
        StageStrategy::SortBottomK(_) => !descending,
        StageStrategy::Default => false,
    };

    let mut keyed: Vec<(Val, Val)> = match k {
        Some(0) => return Ok(Vec::new()),
        Some(limit) if buf.len() > limit => {
            let mut top: Vec<(Val, Val)> = Vec::with_capacity(limit + 1);
            for v in buf.into_iter() {
                let key = key_of(&v)?;
                if top.len() < limit {
                    top.push((key, v));
                    continue;
                }
                let mut worst_idx = 0usize;
                for (i, (candidate, _)) in top.iter().enumerate().skip(1) {
                    let ord = cmp_val_total(candidate, &top[worst_idx].0);
                    let displace = if keep_largest {
                        ord == std::cmp::Ordering::Less
                    } else {
                        ord == std::cmp::Ordering::Greater
                    };
                    if displace {
                        worst_idx = i;
                    }
                }
                let ord = cmp_val_total(&key, &top[worst_idx].0);
                let take = if keep_largest {
                    ord == std::cmp::Ordering::Greater
                } else {
                    ord == std::cmp::Ordering::Less
                };
                if take {
                    top[worst_idx] = (key, v);
                }
            }
            top
        }
        _ => {
            let mut keyed = Vec::with_capacity(buf.len());
            for v in buf.into_iter() {
                keyed.push((key_of(&v)?, v));
            }
            keyed
        }
    };

    keyed.sort_by(|a, b| cmp_val_total(&a.0, &b.0));
    if descending {
        keyed.reverse();
    }
    Ok(keyed.into_iter().map(|(_, v)| v).collect())
}

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

    fn try_run_composed(&self, root: &Val, base_env: &Env) -> Option<Result<Val, EvalError>> {
        use crate::composed as cmp;
        use crate::composed::Stage as ComposedStage;
        use std::cell::{Cell, RefCell};
        use std::rc::Rc;

        let (eff_stages, eff_kernels, eff_sink) = self.canonical();

        // Shared VM + Env for any Generic-kernel stage in the chain.
        // Constructed once per call so compile/path caches amortise.
        // Lazy: only built if a Generic stage actually appears.
        let vm_ctx: std::cell::OnceCell<Rc<RefCell<cmp::VmCtx>>> = std::cell::OnceCell::new();
        let make_ctx = || -> Rc<RefCell<cmp::VmCtx>> {
            let vm = crate::vm::VM::new();
            let env = base_env.clone();
            Rc::new(RefCell::new(cmp::VmCtx { vm, env }))
        };
        let get_ctx = || -> Rc<RefCell<cmp::VmCtx>> { Rc::clone(vm_ctx.get_or_init(make_ctx)) };

        // Sink mapping — operates on the post-decomposition base sink.
        enum SinkKind {
            Count,
            Sum,
            Min,
            Max,
            Avg,
            First,
            Last,
            Collect,
            GroupByOnly,
        }
        let sink_kind = match &eff_sink {
            Sink::Collect => SinkKind::Collect,
            Sink::Count => SinkKind::Count,
            Sink::Numeric(n) if n.project.is_some() => return None,
            Sink::Numeric(n) => match n.op {
                NumOp::Sum => SinkKind::Sum,
                NumOp::Min => SinkKind::Min,
                NumOp::Max => SinkKind::Max,
                NumOp::Avg => SinkKind::Avg,
            },
            Sink::First => SinkKind::First,
            Sink::Last => SinkKind::Last,
            Sink::ApproxCountDistinct => return None, // legacy Val path
        };
        let _ = SinkKind::GroupByOnly;

        // Build a `KeySource` from a barrier-style kernel. Returns None
        // when the kernel is computed or generic — caller bails to legacy.
        fn key_from_kernel(k: &BodyKernel) -> Option<cmp::KeySource> {
            match k {
                BodyKernel::FieldRead(f) => Some(cmp::KeySource::Field(Arc::clone(f))),
                BodyKernel::FieldChain(keys) => Some(cmp::KeySource::Chain(Arc::clone(keys))),
                _ => None,
            }
        }

        // Build a streaming Stage from (Stage, BodyKernel). Recognised
        // borrow kernels (FieldRead / FieldChain / FieldCmpLit Eq)
        // dispatch to zero-clone borrow stages; everything else falls
        // through to the Generic VM-fallback stage that re-enters
        // `vm.exec_in_env` per row using the shared `vm_ctx`. One arm
        // per kernel pattern; one mechanism for every body shape.
        let build_stream_stage = |s: &Stage, k: &BodyKernel| -> Option<Box<dyn ComposedStage>> {
            Some(match (s, k) {
                (Stage::Filter(_), BodyKernel::FieldCmpLit(field, op, lit))
                    if matches!(op, crate::ast::BinOp::Eq) =>
                {
                    Box::new(cmp::FilterFieldEqLit {
                        field: Arc::clone(field),
                        target: lit.clone(),
                    })
                }
                (Stage::Map(_), BodyKernel::FieldRead(field)) => Box::new(cmp::MapField {
                    field: Arc::clone(field),
                }),
                (Stage::Map(_), BodyKernel::FieldChain(keys)) => Box::new(cmp::MapFieldChain {
                    keys: Arc::clone(keys),
                }),
                (Stage::FlatMap(_), BodyKernel::FieldRead(field)) => Box::new(cmp::FlatMapField {
                    field: Arc::clone(field),
                }),
                (Stage::FlatMap(_), BodyKernel::FieldChain(keys)) => {
                    Box::new(cmp::FlatMapFieldChain {
                        keys: Arc::clone(keys),
                    })
                }
                (Stage::Take(n), _) => Box::new(cmp::Take {
                    remaining: Cell::new(*n),
                }),
                (Stage::Skip(n), _) => Box::new(cmp::Skip {
                    remaining: Cell::new(*n),
                }),
                // VM-fallback for any unrecognised body — Generic kernel,
                // FieldCmpLit non-Eq, custom lambdas.
                (Stage::Filter(p), _) => Box::new(cmp::GenericFilter {
                    prog: Arc::clone(p),
                    ctx: get_ctx(),
                }),
                (Stage::Map(p), _) => Box::new(cmp::GenericMap {
                    prog: Arc::clone(p),
                    ctx: get_ctx(),
                }),
                (Stage::FlatMap(p), _) => Box::new(cmp::GenericFlatMap {
                    prog: Arc::clone(p),
                    ctx: get_ctx(),
                }),
                _ => return None,
            })
        };

        // Resolve source to an owned Vec<Val>. Future: avoid clone on
        // pure Arr by holding Arc<Vec<Val>> for the first segment.
        let recv = match &self.source {
            Source::Receiver(v) => v.clone(),
            Source::FieldChain { keys } => walk_field_chain(root, keys),
        };
        let mut buf: Vec<Val> = match recv {
            Val::Arr(a) => a.as_ref().clone(),
            Val::IntVec(a) => a.iter().map(|n| Val::Int(*n)).collect(),
            Val::FloatVec(a) => a.iter().map(|f| Val::Float(*f)).collect(),
            Val::StrVec(a) => a.iter().map(|s| Val::Str(Arc::clone(s))).collect(),
            _ => return None,
        };

        // Walk stages, splitting at barriers. Each streaming run uses
        // a composed-Cow chain into a CollectSink to materialise the
        // intermediate Vec<Val>; each barrier consumes Vec, returns
        // Vec. Final segment uses the actual sink. GroupBy is treated
        // as a barrier whose output Val replaces the buffer (used
        // only when followed by no further stages + Sink::Collect).
        let kernels = &eff_kernels;
        let stages_ref = &eff_stages;

        // Step 3d Phase 1: demand propagation.  Each stage sees the
        // demand its downstream wants, picks an algorithm, and exposes
        // upstream demand.  Sort under bounded downstream demand picks
        // top-k instead of full sort — generic mechanism, no rewrite
        // rule.
        let strategies = compute_strategies(stages_ref, &eff_sink);

        // Find barrier positions. Each [last_split..barrier_idx] is a
        // streaming segment; [barrier_idx] is the barrier op.
        let mut last_split = 0usize;
        for (i, s) in stages_ref.iter().enumerate() {
            let is_barrier = matches!(
                s,
                Stage::Reverse | Stage::Sort(_) | Stage::UniqueBy(_) | Stage::GroupBy(_)
            );
            if !is_barrier {
                continue;
            }

            // Build streaming chain over [last_split..i].
            if i > last_split {
                let mut chain: Box<dyn ComposedStage> = Box::new(cmp::Identity);
                for j in last_split..i {
                    let stage = &stages_ref[j];
                    let kernel = kernels.get(j).unwrap_or(&BodyKernel::Generic);
                    let next = build_stream_stage(stage, kernel)?;
                    chain = Box::new(cmp::Composed { a: chain, b: next });
                }
                let out = cmp::run_pipeline::<cmp::CollectSink>(&buf, chain.as_ref());
                buf = match out {
                    Val::Arr(a) => a.as_ref().clone(),
                    _ => return None,
                };
            }

            // Apply barrier.
            let kernel = kernels.get(i).unwrap_or(&BodyKernel::Generic);
            let strategy = strategies.get(i).copied().unwrap_or(StageStrategy::Default);
            buf = match s {
                Stage::Reverse => cmp::barrier_reverse(buf),
                Stage::Sort(spec) => {
                    let key = match &spec.key {
                        None => cmp::KeySource::None,
                        Some(_) => key_from_kernel(kernel)?,
                    };
                    let mut out = match (strategy, spec.descending) {
                        (StageStrategy::SortTopK(k), false)
                        | (StageStrategy::SortBottomK(k), true) => cmp::barrier_top_k(buf, &key, k),
                        (StageStrategy::SortTopK(k), true)
                        | (StageStrategy::SortBottomK(k), false) => {
                            cmp::barrier_bottom_k(buf, &key, k)
                        }
                        (_, false) => cmp::barrier_sort(buf, &key),
                        (_, true) => cmp::barrier_sort(buf, &key),
                    };
                    if spec.descending {
                        out.reverse();
                    }
                    out
                }
                Stage::UniqueBy(None) => cmp::barrier_unique_by(buf, &cmp::KeySource::None),
                Stage::UniqueBy(Some(_)) => {
                    let key = key_from_kernel(kernel)?;
                    cmp::barrier_unique_by(buf, &key)
                }
                Stage::GroupBy(_) => {
                    if !matches!(eff_sink, Sink::Collect) {
                        return None;
                    }
                    if i + 1 != stages_ref.len() {
                        return None;
                    }
                    let key = key_from_kernel(kernel)?;
                    let val = cmp::barrier_group_by(buf, &key);
                    let _ = i;
                    return Some(Ok(val));
                }
                _ => unreachable!(),
            };

            last_split = i + 1;
        }

        // Final streaming segment + sink.
        let mut chain: Box<dyn ComposedStage> = Box::new(cmp::Identity);
        for j in last_split..stages_ref.len() {
            let stage = &stages_ref[j];
            let kernel = kernels.get(j).unwrap_or(&BodyKernel::Generic);
            let next = build_stream_stage(stage, kernel)?;
            chain = Box::new(cmp::Composed { a: chain, b: next });
        }
        let final_demand = Self::segment_source_demand(&stages_ref[last_split..], &eff_sink)
            .chain
            .pull;

        let out = match sink_kind {
            SinkKind::Count => {
                cmp::run_pipeline_with_demand::<cmp::CountSink>(&buf, chain.as_ref(), final_demand)
            }
            SinkKind::Sum => {
                cmp::run_pipeline_with_demand::<cmp::SumSink>(&buf, chain.as_ref(), final_demand)
            }
            SinkKind::Min => {
                cmp::run_pipeline_with_demand::<cmp::MinSink>(&buf, chain.as_ref(), final_demand)
            }
            SinkKind::Max => {
                cmp::run_pipeline_with_demand::<cmp::MaxSink>(&buf, chain.as_ref(), final_demand)
            }
            SinkKind::Avg => {
                cmp::run_pipeline_with_demand::<cmp::AvgSink>(&buf, chain.as_ref(), final_demand)
            }
            SinkKind::First => {
                cmp::run_pipeline_with_demand::<cmp::FirstSink>(&buf, chain.as_ref(), final_demand)
            }
            SinkKind::Last => {
                cmp::run_pipeline_with_demand::<cmp::LastSink>(&buf, chain.as_ref(), final_demand)
            }
            SinkKind::Collect => cmp::run_pipeline_with_demand::<cmp::CollectSink>(
                &buf,
                chain.as_ref(),
                final_demand,
            ),
            SinkKind::GroupByOnly => unreachable!(),
        };

        Some(Ok(out))
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
            if let Some(out) = self.try_run_composed(root, base_env) {
                return out;
            }
        }

        // One VM owned by the pull loop — shared across stage program
        // calls so VM compile / path caches amortise across the row
        // sweep.  Constructing a fresh VM per row regresses 250x.
        let mut vm = crate::vm::VM::new();
        // Phase A1: build one Env at loop entry; per-row apply uses
        // `swap_current` instead of full Env construction + doc-hash
        // recompute + cache clear (those add ~80 ns/row of pure
        // overhead in execute_val_raw).
        let mut loop_env = base_env.clone();

        // Resolve source to an iterable Val::Arr-like sequence.
        let recv = match &self.source {
            Source::Receiver(v) => v.clone(),
            Source::FieldChain { keys } => walk_field_chain(root, keys),
        };

        // Pull-based stage chain.  At Phase 1 the inner loop materialises
        // elements one at a time as `Val`; Phase 3 will switch this to a
        // per-batch pull over columnar lanes.
        let source_demand = self.source_demand().chain.pull;
        let mut pulled_inputs: usize = 0;
        let mut emitted_outputs: usize = 0;
        let mut stage_taken: Vec<usize> = vec![0; self.stages.len()];
        let mut stage_skipped: Vec<usize> = vec![0; self.stages.len()];

        let iter: Box<dyn Iterator<Item = Val>> = match &recv {
            Val::Arr(a) => Box::new(a.as_ref().clone().into_iter()),
            Val::IntVec(a) => Box::new(
                a.iter()
                    .map(|n| Val::Int(*n))
                    .collect::<Vec<_>>()
                    .into_iter(),
            ),
            Val::FloatVec(a) => Box::new(
                a.iter()
                    .map(|f| Val::Float(*f))
                    .collect::<Vec<_>>()
                    .into_iter(),
            ),
            Val::StrVec(a) => Box::new(
                a.iter()
                    .map(|s| Val::Str(Arc::clone(s)))
                    .collect::<Vec<_>>()
                    .into_iter(),
            ),
            // ObjVec: materialise rows into Val::Obj for the per-row pull
            // path.  Slot-indexed columnar fast paths in `try_columnar`
            // handle the common SumMap / CountIf / SumFilterMap shapes
            // before this point — landing here means the sink is
            // Collect / take / skip / etc., which truly need Val::Obj
            // rows for downstream stages.
            Val::ObjVec(d) => {
                let n = d.nrows();
                let mut out: Vec<Val> = Vec::with_capacity(n);
                let stride = d.stride();
                for row in 0..n {
                    let mut m: indexmap::IndexMap<Arc<str>, Val> =
                        indexmap::IndexMap::with_capacity(stride);
                    for (i, k) in d.keys.iter().enumerate() {
                        m.insert(Arc::clone(k), d.cells[row * stride + i].clone());
                    }
                    out.push(Val::Obj(Arc::new(m)));
                }
                Box::new(out.into_iter())
            }
            // Anything else (scalar, Obj, …): single-element "iterator".
            _ => Box::new(std::iter::once(recv.clone())),
        };

        // Sink accumulators.
        let mut acc_collect: Vec<Val> = Vec::new();
        let mut acc_count: i64 = 0;
        let mut acc_sum_i: i64 = 0;
        let mut acc_sum_f: f64 = 0.0;
        let mut sum_floated: bool = false;
        let mut acc_min_f: f64 = f64::INFINITY;
        let mut acc_max_f: f64 = f64::NEG_INFINITY;
        let mut acc_n_obs: usize = 0;
        let mut acc_first: Option<Val> = None;
        let mut acc_last: Option<Val> = None;
        // Category E: HLL-12 register array (4096 × u8) — only used when
        // sink is ApproxCountDistinct.  Allocates ~4KB even when unused;
        // optimiser could box this when not needed.  Cheap relative to
        // typical query memory.
        let mut acc_hll: [u8; HLL_M] = [0u8; HLL_M];

        // Stages that materialise force a buffer; stages preceding
        // them run as streaming filter/map over the buffer.  Process
        // every stage in order so the pipeline semantics match the
        // surface query.
        let needs_barrier = self.stages.iter().any(|s| {
            matches!(
                s,
                Stage::Reverse
                    | Stage::Sort(_)
                    | Stage::UniqueBy(_)
                    | Stage::FlatMap(_)
                    | Stage::GroupBy(_)
                    | Stage::Split(_)
                    | Stage::Chunk(_)
                    | Stage::Window(_)
                    | Stage::DropWhile(_)
                    | Stage::IndicesWhere(_)
                    | Stage::FindIndex(_)
                    | Stage::MaxBy(_)
                    | Stage::MinBy(_)
                    | Stage::CountBy(_)
                    | Stage::IndexBy(_)
                    | Stage::SortedDedup(_)
            )
        });
        let pre_iter: Box<dyn Iterator<Item = Val>> = if needs_barrier {
            let mut buf: Vec<Val> = iter.collect();
            let strategies = compute_strategies(&self.stages, &self.sink);
            // Phase 1.2 — barrier-stage path now reads stage_kernels[i]
            // and dispatches the inline kernel for Sort/UniqueBy keyed
            // variants too, not just streaming Filter/Map.  Extends
            // Layer A coverage to the keyed-barrier surface.
            for (stage_idx, stage) in self.stages.iter().enumerate() {
                let kernel = self
                    .stage_kernels
                    .get(stage_idx)
                    .unwrap_or(&BodyKernel::Generic);
                match stage {
                    Stage::Filter(prog) => {
                        buf = filter_apply(buf, |v| {
                            eval_kernel(kernel, v, |item| {
                                apply_item_in_env(&mut vm, &mut loop_env, item, prog)
                            })
                        })?;
                    }
                    Stage::Map(prog) => {
                        buf = map_apply(buf, |v| {
                            eval_kernel(kernel, v, |item| {
                                apply_item_in_env(&mut vm, &mut loop_env, item, prog)
                            })
                        })?;
                    }
                    Stage::Skip(n) => {
                        if buf.len() <= *n {
                            buf.clear();
                        } else {
                            buf.drain(..*n);
                        }
                    }
                    Stage::Take(n) => {
                        buf.truncate(*n);
                    }
                    Stage::Reverse => buf.reverse(),
                    Stage::Sort(spec) => match &spec.key {
                        None => {
                            let strategy = strategies
                                .get(stage_idx)
                                .copied()
                                .unwrap_or(StageStrategy::Default);
                            buf = sort_buffer_by_key(buf, spec.descending, strategy, |v| {
                                Ok(v.clone())
                            })?;
                        }
                        Some(prog) => {
                            let strategy = strategies
                                .get(stage_idx)
                                .copied()
                                .unwrap_or(StageStrategy::Default);
                            buf = sort_buffer_by_key(buf, spec.descending, strategy, |v| {
                                Ok(eval_kernel(kernel, v, |item| {
                                    apply_item_in_env(&mut vm, &mut loop_env, item, prog)
                                })
                                .unwrap_or(Val::Null))
                            })?;
                        }
                    },
                    Stage::UniqueBy(None) => {
                        let mut seen: std::collections::HashSet<String> = Default::default();
                        buf.retain(|v| seen.insert(format!("{:?}", v)));
                    }
                    Stage::UniqueBy(Some(prog)) => {
                        let mut seen: std::collections::HashSet<String> = Default::default();
                        let mut keep: Vec<bool> = Vec::with_capacity(buf.len());
                        for v in &buf {
                            let k = eval_kernel(kernel, v, |item| {
                                apply_item_in_env(&mut vm, &mut loop_env, item, prog)
                            })
                            .unwrap_or(Val::Null);
                            keep.push(seen.insert(format!("{:?}", k)));
                        }
                        let mut out: Vec<Val> = Vec::with_capacity(buf.len());
                        for (i, v) in buf.into_iter().enumerate() {
                            if keep[i] {
                                out.push(v);
                            }
                        }
                        buf = out;
                    }
                    Stage::FlatMap(prog) => {
                        let mut out: Vec<Val> = Vec::new();
                        for v in &buf {
                            let inner = eval_kernel(kernel, v, |item| {
                                apply_item_in_env(&mut vm, &mut loop_env, item, prog)
                            })?;
                            if let Some(arr) = inner.as_vals() {
                                out.extend(arr.iter().cloned());
                            } else {
                                out.push(inner);
                            }
                        }
                        buf = out;
                    }
                    Stage::GroupBy(prog) => {
                        // GroupBy is a barrier that yields one Val::Obj.
                        // Place it as a single-element buf so downstream
                        // stages see the grouped object.  Sink::Collect
                        // will return Val::arr([this_obj]); a separate
                        // shortcut below converts that to the bare obj.
                        let out_obj = group_by_apply(buf, |v| {
                            eval_kernel(kernel, v, |item| {
                                apply_item_in_env(&mut vm, &mut loop_env, item, prog)
                            })
                        })?;
                        buf = vec![Val::Obj(Arc::new(out_obj))];
                    }
                    Stage::Split(sep) => {
                        // Step 3d-extension (C): Expanding string Stage.
                        let mut out: Vec<Val> = Vec::with_capacity(buf.len());
                        for v in buf.into_iter() {
                            if let Some(Val::Arr(a)) = split_apply(&v, sep.as_ref()) {
                                out.extend(Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone()));
                            }
                        }
                        buf = out;
                    }
                    Stage::Slice(start, end) => {
                        let mut out: Vec<Val> = Vec::with_capacity(buf.len());
                        for v in buf.into_iter() {
                            out.push(slice_apply(v, *start, *end));
                        }
                        buf = out;
                    }
                    Stage::Replace {
                        needle,
                        replacement,
                        all,
                    } => {
                        let mut out: Vec<Val> = Vec::with_capacity(buf.len());
                        for v in buf.into_iter() {
                            match replace_apply(v.clone(), needle, replacement, *all) {
                                Some(r) => out.push(r),
                                None => out.push(v),
                            }
                        }
                        buf = out;
                    }
                    Stage::Chunk(n) => {
                        buf = chunk_apply(&buf, *n);
                    }
                    Stage::Window(n) => {
                        buf = window_apply(&buf, *n);
                    }
                    Stage::CompiledMap(plan) => {
                        let mut out: Vec<Val> = Vec::with_capacity(buf.len());
                        for v in buf.into_iter() {
                            out.push(run_compiled_map(plan, v)?);
                        }
                        buf = out;
                    }
                    Stage::Builtin(call) => {
                        buf = buf
                            .into_iter()
                            .map(|v| call.apply(&v).unwrap_or(v))
                            .collect();
                    }
                    // Lambda-bearing barrier-mode stages.
                    Stage::TakeWhile(prog) => {
                        buf = take_while_apply(buf, |v| {
                            eval_kernel(kernel, v, |item| {
                                apply_item_in_env(&mut vm, &mut loop_env, item, prog)
                            })
                        })?;
                    }
                    Stage::DropWhile(prog) => {
                        buf = drop_while_apply(buf, |v| {
                            eval_kernel(kernel, v, |item| {
                                apply_item_in_env(&mut vm, &mut loop_env, item, prog)
                            })
                        })?;
                    }
                    Stage::IndicesWhere(prog) => {
                        let mut out: Vec<i64> = Vec::new();
                        for (i, v) in buf.iter().enumerate() {
                            if filter_one(v, |item| {
                                eval_kernel(kernel, item, |item| {
                                    apply_item_in_env(&mut vm, &mut loop_env, item, prog)
                                })
                            })? {
                                out.push(i as i64);
                            }
                        }
                        buf = vec![Val::int_vec(out)];
                    }
                    Stage::FindIndex(prog) => {
                        let mut found: Val = Val::Null;
                        for (i, v) in buf.iter().enumerate() {
                            if filter_one(v, |item| {
                                eval_kernel(kernel, item, |item| {
                                    apply_item_in_env(&mut vm, &mut loop_env, item, prog)
                                })
                            })? {
                                found = Val::Int(i as i64);
                                break;
                            }
                        }
                        buf = vec![found];
                    }
                    Stage::MaxBy(prog) | Stage::MinBy(prog) => {
                        let want_max = matches!(stage, Stage::MaxBy(_));
                        if buf.is_empty() {
                            buf = vec![Val::Null];
                        } else {
                            let mut best_idx = 0usize;
                            let mut best_key = eval_kernel(kernel, &buf[0], |item| {
                                apply_item_in_env(&mut vm, &mut loop_env, item, prog)
                            })?;
                            for i in 1..buf.len() {
                                let k = eval_kernel(kernel, &buf[i], |item| {
                                    apply_item_in_env(&mut vm, &mut loop_env, item, prog)
                                })?;
                                let cmp = cmp_val_total(&k, &best_key);
                                let take = if want_max {
                                    cmp == std::cmp::Ordering::Greater
                                } else {
                                    cmp == std::cmp::Ordering::Less
                                };
                                if take {
                                    best_idx = i;
                                    best_key = k;
                                }
                            }
                            buf = vec![buf.into_iter().nth(best_idx).unwrap()];
                        }
                    }
                    Stage::CountBy(prog) => {
                        let map = count_by_apply(buf, |v| {
                            eval_kernel(kernel, v, |item| {
                                apply_item_in_env(&mut vm, &mut loop_env, item, prog)
                            })
                        })?;
                        buf = vec![Val::obj(map)];
                    }
                    Stage::SortedDedup(key_prog) => {
                        // Category F: combined sort + dedup-consecutive.
                        match key_prog {
                            None => {
                                buf.sort_by(|a, b| cmp_val_total(a, b));
                                buf.dedup_by(|a, b| crate::util::vals_eq(a, b));
                            }
                            Some(prog) => {
                                // Decorate-sort-undecorate via key prog.
                                let mut keyed: Vec<(Val, Val)> = Vec::with_capacity(buf.len());
                                for v in buf.iter() {
                                    let k = eval_kernel(kernel, v, |item| {
                                        apply_item_in_env(&mut vm, &mut loop_env, item, prog)
                                    })?;
                                    keyed.push((k, v.clone()));
                                }
                                keyed.sort_by(|a, b| cmp_val_total(&a.0, &b.0));
                                keyed.dedup_by(|a, b| crate::util::vals_eq(&a.0, &b.0));
                                buf = keyed.into_iter().map(|(_, v)| v).collect();
                            }
                        }
                    }
                    Stage::IndexBy(prog) => {
                        let map = index_by_apply(buf, |v| {
                            eval_kernel(kernel, v, |item| {
                                apply_item_in_env(&mut vm, &mut loop_env, item, prog)
                            })
                        })?;
                        buf = vec![Val::obj(map)];
                    }
                    // Per-Obj lambda-bearing (works on each row that
                    // happens to be an Obj).  Uses VM eval per (k, v).
                    Stage::TransformValues(prog)
                    | Stage::TransformKeys(prog)
                    | Stage::FilterValues(prog)
                    | Stage::FilterKeys(prog) => {
                        let mut out: Vec<Val> = Vec::with_capacity(buf.len());
                        for v in buf.into_iter() {
                            let mapped =
                                apply_lambda_obj(stage, &v, &mut vm, &mut loop_env, kernel, prog)?;
                            out.push(mapped);
                        }
                        buf = out;
                    }
                }
            }
            Box::new(buf.into_iter())
        } else {
            iter
        };

        'outer: for mut item in pre_iter {
            if matches!(source_demand, PullDemand::AtMost(n) if pulled_inputs >= n) {
                break 'outer;
            }
            pulled_inputs += 1;

            // When barriers ran above, stages have already been
            // applied — `pre_iter` yields the post-pipeline rows
            // directly.  When no barriers are present, run streaming
            // stages here.
            if !needs_barrier {
                for (stage_idx, stage) in self.stages.iter().enumerate() {
                    let kernel = self
                        .stage_kernels
                        .get(stage_idx)
                        .unwrap_or(&BodyKernel::Generic);
                    match stage {
                        Stage::Skip(n) => {
                            if stage_skipped[stage_idx] < *n {
                                stage_skipped[stage_idx] += 1;
                                continue 'outer;
                            }
                        }
                        Stage::Take(n) => {
                            if stage_taken[stage_idx] >= *n {
                                break 'outer;
                            }
                            stage_taken[stage_idx] += 1;
                        }
                        Stage::Filter(prog) => {
                            if !filter_one(&item, |v| {
                                eval_kernel(kernel, v, |item| {
                                    apply_item_in_env(&mut vm, &mut loop_env, item, prog)
                                })
                            })? {
                                continue 'outer;
                            }
                        }
                        Stage::Map(prog) => {
                            item = map_one(&item, |v| {
                                eval_kernel(kernel, v, |item| {
                                    apply_item_in_env(&mut vm, &mut loop_env, item, prog)
                                })
                            })?;
                        }
                        Stage::Reverse
                        | Stage::Sort(_)
                        | Stage::UniqueBy(_)
                        | Stage::FlatMap(_)
                        | Stage::GroupBy(_) => {}
                        Stage::Split(_) | Stage::Chunk(_) | Stage::Window(_) => {} // forced into barrier path above
                        Stage::Slice(start, end) => {
                            item = slice_apply(item, *start, *end);
                        }
                        Stage::Replace {
                            needle,
                            replacement,
                            all,
                        } => {
                            if let Some(r) = replace_apply(item.clone(), needle, replacement, *all)
                            {
                                item = r;
                            }
                        }
                        Stage::CompiledMap(plan) => {
                            item = run_compiled_map(plan, item)?;
                        }
                        Stage::Builtin(call) => {
                            item = call.apply(&item).unwrap_or(item);
                        }
                        // Lambda-bearing streaming arm: TakeWhile.
                        Stage::TakeWhile(prog) => {
                            if !take_while_one(&item, |v| {
                                eval_kernel(kernel, v, |item| {
                                    apply_item_in_env(&mut vm, &mut loop_env, item, prog)
                                })
                            })? {
                                break 'outer;
                            }
                        }
                        Stage::TransformValues(prog)
                        | Stage::TransformKeys(prog)
                        | Stage::FilterValues(prog)
                        | Stage::FilterKeys(prog) => {
                            item = apply_lambda_obj(
                                stage,
                                &item,
                                &mut vm,
                                &mut loop_env,
                                kernel,
                                prog,
                            )?;
                        }
                        // Forced into barrier path above (DropWhile needs
                        // cross-row state; reductions consume full stream).
                        Stage::DropWhile(_)
                        | Stage::IndicesWhere(_)
                        | Stage::FindIndex(_)
                        | Stage::MaxBy(_)
                        | Stage::MinBy(_)
                        | Stage::CountBy(_)
                        | Stage::IndexBy(_)
                        | Stage::SortedDedup(_) => {}
                    }
                }
            }

            // Sink.
            match &self.sink {
                Sink::Collect => acc_collect.push(item),
                Sink::Count => acc_count += 1,
                Sink::Numeric(n) => {
                    let numeric_item = if let Some(project) = &n.project {
                        let kernel = self.sink_kernels.first().unwrap_or(&BodyKernel::Generic);
                        eval_kernel(kernel, &item, |item| {
                            apply_item_in_env(&mut vm, &mut loop_env, item, project)
                        })?
                    } else {
                        item
                    };
                    num_fold(
                        &mut acc_sum_i,
                        &mut acc_sum_f,
                        &mut sum_floated,
                        &mut acc_min_f,
                        &mut acc_max_f,
                        &mut acc_n_obs,
                        n.op,
                        &numeric_item,
                    );
                }
                Sink::First => {
                    if acc_first.is_none() {
                        acc_first = Some(item.clone());
                        break 'outer;
                    }
                }
                Sink::Last => {
                    acc_last = Some(item.clone());
                }
                Sink::ApproxCountDistinct => {
                    hll_observe(&mut acc_hll, &item);
                }
            }
            emitted_outputs += 1;
            if matches!(source_demand, PullDemand::UntilOutput(n) if emitted_outputs >= n) {
                break 'outer;
            }
        }

        Ok(match &self.sink {
            Sink::Collect => {
                // GroupBy is a barrier that produces a single Val::Obj
                // which Sink::Collect would otherwise wrap as
                // [obj].  When the last stage is GroupBy, return the
                // bare object to match walker semantics.
                if matches!(self.stages.last(), Some(Stage::GroupBy(_)))
                    && acc_collect.len() == 1
                    && matches!(acc_collect[0], Val::Obj(_))
                {
                    acc_collect.into_iter().next().unwrap()
                } else {
                    Val::arr(acc_collect)
                }
            }
            Sink::Count => Val::Int(acc_count),
            Sink::Numeric(n) => num_finalise(
                n.op,
                acc_sum_i,
                acc_sum_f,
                sum_floated,
                acc_min_f,
                acc_max_f,
                acc_n_obs,
            ),
            Sink::First => acc_first.unwrap_or(Val::Null),
            Sink::Last => acc_last.unwrap_or(Val::Null),
            Sink::ApproxCountDistinct => Val::Int(hll_estimate(&acc_hll) as i64),
        })
    }
}
// ── Algorithmic Category E: HyperLogLog ──────────────────────────
//
// HLL-12: precision p=12, m=2^12=4096 registers, ±2% std error.
// 4 KB state.  Hash via FxHasher (jetro already uses); top 12 bits
// pick register, remaining 52 bits + 1 give leading-zero count + 1.
// Estimate: harmonic mean × m^2 × αm correction.
//
// Per `algorithmic_optimization_cold_only.md` Category E.

const HLL_P: u32 = 12;
const HLL_M: usize = 1 << HLL_P; // 4096

#[inline]
fn hll_hash(v: &Val) -> u64 {
    use crate::util::val_to_key;
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    // Re-use val_to_key for canonical string-form hashing — matches
    // UniqueBy semantics so `unique().count()` and
    // `approx_count_distinct()` agree on what counts as "distinct".
    static STATE: std::sync::OnceLock<RandomState> = std::sync::OnceLock::new();
    let bs = STATE.get_or_init(RandomState::new);
    let s = val_to_key(v);
    let mut h = bs.build_hasher();
    h.write(s.as_bytes());
    h.finish()
}

fn hll_observe(reg: &mut [u8; HLL_M], v: &Val) {
    let h = hll_hash(v);
    let idx = (h >> (64 - HLL_P)) as usize;
    let w = (h << HLL_P) | (1u64 << (HLL_P - 1));
    let lz = w.leading_zeros() as u8 + 1;
    if lz > reg[idx] {
        reg[idx] = lz;
    }
}

fn hll_estimate(reg: &[u8; HLL_M]) -> f64 {
    // Harmonic mean of 2^(-reg[i]).
    let mut z: f64 = 0.0;
    let mut zeros: usize = 0;
    for &r in reg.iter() {
        z += 1.0 / (1u64 << r) as f64;
        if r == 0 {
            zeros += 1;
        }
    }
    let m = HLL_M as f64;
    let alpha_m = 0.7213 / (1.0 + 1.079 / m); // p=12 form
    let raw = alpha_m * m * m / z;
    // Small-range correction.
    if raw <= 2.5 * m && zeros > 0 {
        return m * (m / zeros as f64).ln();
    }
    raw
}

/// Per-Obj lambda dispatch helper for `TransformKeys` /
/// `TransformValues` / `FilterKeys` / `FilterValues`.  Each visits
/// every (k, v) entry and runs the program with the arg as `@`.
/// Non-Obj receivers pass through unchanged.
pub(crate) fn apply_lambda_obj(
    stage: &Stage,
    recv: &Val,
    vm: &mut crate::vm::VM,
    loop_env: &mut crate::context::Env,
    kernel: &BodyKernel,
    prog: &std::sync::Arc<crate::vm::Program>,
) -> Result<Val, EvalError> {
    let m = match recv.as_object() {
        Some(m) => m,
        None => return Ok(recv.clone()),
    };
    let mut out: indexmap::IndexMap<std::sync::Arc<str>, Val> =
        indexmap::IndexMap::with_capacity(m.len());
    for (k, v) in m.iter() {
        match stage {
            Stage::TransformKeys(_) => {
                let k_val = Val::Str(k.clone());
                let new_k = eval_kernel(kernel, &k_val, |item| {
                    apply_item_in_env(vm, loop_env, item, prog)
                })?;
                let new_k_arc = match new_k {
                    Val::Str(s) => s,
                    other => std::sync::Arc::from(crate::util::val_to_string(&other).as_str()),
                };
                out.insert(new_k_arc, v.clone());
            }
            Stage::TransformValues(_) => {
                let new_v = eval_kernel(kernel, v, |item| {
                    apply_item_in_env(vm, loop_env, item, prog)
                })?;
                out.insert(k.clone(), new_v);
            }
            Stage::FilterKeys(_) => {
                let k_val = Val::Str(k.clone());
                if is_truthy(&eval_kernel(kernel, &k_val, |item| {
                    apply_item_in_env(vm, loop_env, item, prog)
                })?) {
                    out.insert(k.clone(), v.clone());
                }
            }
            Stage::FilterValues(_) => {
                if is_truthy(&eval_kernel(kernel, v, |item| {
                    apply_item_in_env(vm, loop_env, item, prog)
                })?) {
                    out.insert(k.clone(), v.clone());
                }
            }
            _ => unreachable!("apply_lambda_obj called with non-Obj-lambda Stage"),
        }
    }
    Ok(Val::obj(out))
}
