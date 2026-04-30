use std::sync::Arc;

use crate::{
    context::{Env, EvalError},
    value::Val,
};

use super::lower::run_compiled_map;
use super::row_source;
use super::sink_accumulator::SinkAccumulator;
use super::{
    apply_item_in_env, bounded_sort_by_key, cmp_val_total, compute_strategies, eval_kernel,
    is_truthy, BodyKernel, Pipeline, Sink, Stage, StageStrategy, TerminalMapCollector,
};

use crate::builtins::{
    chunk_apply, count_by_apply, drop_while_apply, filter_apply, filter_one, group_by_apply,
    index_by_apply, map_apply, map_one, replace_apply, slice_apply, split_apply, take_while_apply,
    take_while_one, window_apply,
};
use crate::chain_ir::PullDemand;

pub(super) fn run(pipeline: &Pipeline, root: &Val, base_env: &Env) -> Result<Val, EvalError> {
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
    let recv = row_source::resolve(&pipeline.source, root);

    // Pull-based stage chain.  At Phase 1 the inner loop materialises
    // elements one at a time as `Val`; Phase 3 will switch this to a
    // per-batch pull over columnar lanes.
    let source_demand = pipeline.source_demand().chain.pull;
    let mut pulled_inputs: usize = 0;
    let mut emitted_outputs: usize = 0;
    let mut stage_taken: Vec<usize> = vec![0; pipeline.stages.len()];
    let mut stage_skipped: Vec<usize> = vec![0; pipeline.stages.len()];

    let iter: Box<dyn Iterator<Item = Val>> = if let Some(rows) = row_source::array_like_rows(&recv)
    {
        Box::new(rows.into_iter())
    } else {
        match &recv {
            // ObjVec: materialise rows into Val::Obj for the per-row pull
            // path.  Slot-indexed columnar fast paths handle common aggregate
            // shapes before this point; landing here means downstream stages
            // need Val::Obj rows.
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
            // Anything else (scalar, Obj, ...): single-element iterator.
            _ => Box::new(std::iter::once(recv.clone())),
        }
    };

    let mut sink_acc = SinkAccumulator::new(&pipeline.sink);

    // Stages that materialise force a buffer; stages preceding
    // them run as streaming filter/map over the buffer.  Process
    // every stage in order so the pipeline semantics match the
    // surface query.
    let needs_barrier = pipeline
        .stages
        .iter()
        .any(Stage::requires_legacy_materialization);
    let terminal_map_idx = if !needs_barrier && matches!(pipeline.sink, Sink::Collect) {
        match pipeline.stages.last() {
            Some(Stage::Map(_)) => pipeline.stages.len().checked_sub(1),
            _ => None,
        }
    } else {
        None
    };
    let terminal_map_kernel = terminal_map_idx.map(|idx| {
        pipeline
            .stage_kernels
            .get(idx)
            .unwrap_or(&BodyKernel::Generic)
    });
    let mut terminal_map_collect = terminal_map_kernel.map(TerminalMapCollector::new);
    let pre_iter: Box<dyn Iterator<Item = Val>> = if needs_barrier {
        let mut buf: Vec<Val> = iter.collect();
        let strategies = compute_strategies(&pipeline.stages, &pipeline.sink);
        // Phase 1.2 — barrier-stage path now reads stage_kernels[i]
        // and dispatches the inline kernel for Sort/UniqueBy keyed
        // variants too, not just streaming Filter/Map.  Extends
        // Layer A coverage to the keyed-barrier surface.
        for (stage_idx, stage) in pipeline.stages.iter().enumerate() {
            let kernel = pipeline
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
                        buf =
                            bounded_sort_by_key(buf, spec.descending, strategy, |v| Ok(v.clone()))?;
                    }
                    Some(prog) => {
                        let strategy = strategies
                            .get(stage_idx)
                            .copied()
                            .unwrap_or(StageStrategy::Default);
                        buf = bounded_sort_by_key(buf, spec.descending, strategy, |v| {
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
            for (stage_idx, stage) in pipeline.stages.iter().enumerate() {
                let kernel = pipeline
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
                        if Some(stage_idx) == terminal_map_idx {
                            terminal_map_collect
                                .as_mut()
                                .expect("terminal map collector")
                                .push_val_row(&item, kernel, |item| {
                                    apply_item_in_env(&mut vm, &mut loop_env, item, prog)
                                })?;
                            emitted_outputs += 1;
                            if matches!(source_demand, PullDemand::UntilOutput(n) if emitted_outputs >= n)
                            {
                                break 'outer;
                            }
                            continue 'outer;
                        }
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
                        if let Some(r) = replace_apply(item.clone(), needle, replacement, *all) {
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
                        item =
                            apply_lambda_obj(stage, &item, &mut vm, &mut loop_env, kernel, prog)?;
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
        let sink_done = match &pipeline.sink {
            Sink::Numeric(n) => {
                if let Some(project) = &n.project {
                    let kernel = pipeline
                        .sink_kernels
                        .first()
                        .unwrap_or(&BodyKernel::Generic);
                    let numeric_item = eval_kernel(kernel, &item, |item| {
                        apply_item_in_env(&mut vm, &mut loop_env, item, project)
                    })?;
                    sink_acc.push_projected_numeric(&numeric_item);
                } else {
                    sink_acc.push(item);
                }
                false
            }
            _ => sink_acc.push(item),
        };
        if sink_done {
            break 'outer;
        }
        emitted_outputs += 1;
        if matches!(source_demand, PullDemand::UntilOutput(n) if emitted_outputs >= n) {
            break 'outer;
        }
    }

    if let Some(collector) = terminal_map_collect {
        return Ok(collector.finish());
    }

    // GroupBy is a barrier that produces a single Val::Obj which
    // Sink::Collect would otherwise wrap as [obj]. When the last
    // stage is GroupBy, return the bare object to match walker
    // semantics.
    let unwrap_single_collect_obj = matches!(pipeline.stages.last(), Some(Stage::GroupBy(_)));
    Ok(sink_acc.finish(unwrap_single_collect_obj))
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
