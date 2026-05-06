//! Pipeline planning functions: strategy selection, filter reordering, and optimisation passes.
//!
//! The IR type definitions (`Stage` impl, `StageDescriptor`, `Plan`, `Strategy`, etc.) live in
//! `ir.rs`; this module contains the algorithms that consume those types to produce an optimised
//! execution plan.

use std::sync::Arc;

use crate::parse::ast::{BinOp, Expr};
use crate::parse::chain_ir::PullDemand;
use crate::builtins::BuiltinViewStage;
use crate::vm::{Opcode, Program};

use super::{
    symbolic::normalize_symbolic, BodyKernel, Sink, Stage,
};

pub use super::ir::{PhysicalExecPath, Plan, Position, StageStrategy, Strategy};

/// Test-only wrapper around `compute_strategies_with_kernels` that passes an empty kernel slice.
#[cfg(test)]
pub fn compute_strategies(stages: &[Stage], sink: &Sink) -> Vec<StageStrategy> {
    compute_strategies_with_kernels(stages, &[], sink)
}

/// Assigns a `StageStrategy` to every stage in `stages` by walking backwards from the sink
/// demand and promoting bounded `Sort` stages to top-K or lazy-until-output strategies.
pub fn compute_strategies_with_kernels(
    stages: &[Stage],
    kernels: &[BodyKernel],
    sink: &Sink,
) -> Vec<StageStrategy> {
    let mut strategies: Vec<StageStrategy> = vec![StageStrategy::Default; stages.len()];
    let mut demand = sink.demand();
    for (i, stage) in stages.iter().enumerate().rev() {
        if let Stage::Sort(spec) = stage {
            match demand.chain.pull {
                PullDemand::FirstInput(k) => {
                    strategies[i] = match demand.positional {
                        Some(Position::Last) => StageStrategy::SortBottomK(k),
                        _ => StageStrategy::SortTopK(k),
                    };
                }
                PullDemand::UntilOutput(k) => {
                    let sort_kernel = kernels.get(i).unwrap_or(&BodyKernel::Generic);
                    let kernel_suffix = if kernels.len() == stages.len() {
                        &kernels[i + 1..]
                    } else {
                        &[]
                    };
                    if ordered_prefix_suffix_is_safe(
                        spec,
                        sort_kernel,
                        &stages[i + 1..],
                        kernel_suffix,
                    ) {
                        strategies[i] = match demand.positional {
                            Some(Position::Last) => StageStrategy::SortBottomK(k),
                            _ => StageStrategy::SortTopK(k),
                        };
                    } else {
                        strategies[i] = StageStrategy::SortUntilOutput(k);
                    }
                }
                PullDemand::LastInput(k) => {
                    strategies[i] = match demand.positional {
                        Some(Position::First) => StageStrategy::SortTopK(k),
                        _ => StageStrategy::SortBottomK(k),
                    };
                }
                PullDemand::NthInput(_) | PullDemand::All => {}
            }
        }
        demand = stage.upstream_demand(demand);
    }
    strategies
}

fn ordered_prefix_suffix_is_safe(
    sort: &super::SortSpec,
    sort_kernel: &BodyKernel,
    suffix: &[Stage],
    kernels: &[BodyKernel],
) -> bool {
    suffix.iter().enumerate().all(|(idx, stage)| {
        let kernel = kernels.get(idx).unwrap_or(&BodyKernel::Generic);
        stage.ordered_prefix_effect(sort, sort_kernel, kernel)
    })
}

// Returns `true` when `predicate` is a range comparison on the same key as `sort`, meaning
// all remaining elements beyond the predicate's cut-off can never satisfy the predicate.
pub(super) fn predicate_is_order_prefix(
    sort: &super::SortSpec,
    sort_kernel: &BodyKernel,
    predicate: &BodyKernel,
) -> bool {
    let Some((lhs, op)) = predicate_order_lhs(predicate) else {
        return false;
    };
    let Some(order_key) = sort_order_key(sort, sort_kernel) else {
        return false;
    };
    order_lhs_eq(lhs, order_key) && cmp_is_prefix_for_order(op, sort.descending)
}

fn predicate_order_lhs(predicate: &BodyKernel) -> Option<(OrderKey<'_>, BinOp)> {
    match predicate {
        BodyKernel::CurrentCmpLit(op, _) => Some((OrderKey::Current, *op)),
        BodyKernel::FieldCmpLit(field, op, _) => Some((OrderKey::Field(field.as_ref()), *op)),
        BodyKernel::FieldChainCmpLit(keys, op, _) => {
            Some((OrderKey::FieldChain(keys.as_ref()), *op))
        }
        BodyKernel::CmpLit { lhs, op, .. } => lhs_order_key(lhs).map(|lhs| (lhs, *op)),
        _ => None,
    }
}

fn lhs_order_key(lhs: &BodyKernel) -> Option<OrderKey<'_>> {
    match lhs {
        BodyKernel::Current => Some(OrderKey::Current),
        BodyKernel::FieldRead(field) => Some(OrderKey::Field(field.as_ref())),
        BodyKernel::FieldChain(keys) => Some(OrderKey::FieldChain(keys.as_ref())),
        _ => None,
    }
}

fn sort_order_key<'a>(sort: &super::SortSpec, sort_kernel: &'a BodyKernel) -> Option<OrderKey<'a>> {
    if sort.key.is_none() {
        return Some(OrderKey::Current);
    }
    match sort_kernel {
        BodyKernel::FieldRead(field) => Some(OrderKey::Field(field.as_ref())),
        BodyKernel::FieldChain(keys) => Some(OrderKey::FieldChain(keys)),
        BodyKernel::Current => Some(OrderKey::Current),
        _ => None,
    }
}

// Lightweight key descriptor used during sort-strategy analysis; avoids allocation.
enum OrderKey<'a> {
    // The key is the element value itself (no field lookup).
    Current,
    // The key is a single named field of the element object.
    Field(&'a str),
    // The key is obtained by traversing a sequence of field names.
    FieldChain(&'a [Arc<str>]),
}

fn order_lhs_eq(lhs: OrderKey<'_>, key: OrderKey<'_>) -> bool {
    match (lhs, key) {
        (OrderKey::Current, OrderKey::Current) => true,
        (OrderKey::Field(field), OrderKey::Field(key)) => field == key,
        (OrderKey::FieldChain(lhs), OrderKey::FieldChain(rhs)) => same_key_chain(lhs, rhs),
        _ => false,
    }
}

fn same_key_chain(lhs: &[Arc<str>], rhs: &[Arc<str>]) -> bool {
    lhs.len() == rhs.len()
        && lhs
            .iter()
            .zip(rhs.iter())
            .all(|(a, b)| a.as_ref() == b.as_ref())
}

// `>` / `>=` is a prefix for descending order; `<` / `<=` for ascending.
fn cmp_is_prefix_for_order(op: BinOp, descending: bool) -> bool {
    matches!(
        (descending, op),
        (true, BinOp::Gt | BinOp::Gte) | (false, BinOp::Lt | BinOp::Lte)
    )
}

/// Test-only convenience wrapper around `plan_with_kernels` with an empty kernel slice.
#[cfg(test)]
pub fn plan(stages: Vec<Stage>, sink: Sink) -> Plan {
    plan_with_kernels(stages, &[], sink)
}

/// Optimises `stages` and `sink` using the provided pre-classified `kernels`, running symbolic
/// normalisation, filter reordering, filter fusion, and merge-with passes.
pub fn plan_with_kernels(stages: Vec<Stage>, kernels: &[BodyKernel], sink: Sink) -> Plan {
    plan_with_exprs(stages, Vec::new(), kernels, sink)
}

/// Full planning entry point that also accepts `stage_exprs` for the demand optimiser; returns
/// an optimised `Plan` with updated stages, expressions, and sink.
pub fn plan_with_exprs(
    stages: Vec<Stage>,
    stage_exprs: Vec<Option<Arc<Expr>>>,
    kernels: &[BodyKernel],
    mut sink: Sink,
) -> Plan {
    let mut stages = stages;
    let mut e_buf: Vec<Option<Arc<Expr>>> = if stage_exprs.len() == stages.len() {
        stage_exprs
    } else {
        vec![None; stages.len()]
    };
    let mut k_buf: Vec<BodyKernel> = if kernels.len() == stages.len() {
        kernels.to_vec()
    } else {
        vec![BodyKernel::Generic; stages.len()]
    };

    normalize_symbolic(&mut stages, &mut e_buf, &mut k_buf, &mut sink);
    reorder_filter_runs(&mut stages, &mut e_buf, &mut k_buf);
    fuse_filter_runs(&mut stages, &mut e_buf, &mut k_buf);
    fold_merge_with_kernels(&mut stages, &mut e_buf, &mut k_buf);

    Plan {
        stages,
        stage_exprs: e_buf,
        sink,
    }
}

// Uses heuristic selectivity estimates based on the comparison operator type.
fn kernel_cost_selectivity(stage: &Stage, kernel: &BodyKernel) -> (f64, f64) {
    use crate::parse::ast::BinOp;
    match (stage, kernel) {
        (Stage::Filter(_, _), BodyKernel::FieldCmpLit(_, op, _)) => {
            let s = match op {
                BinOp::Eq => 0.10,
                BinOp::Neq => 0.90,
                BinOp::Lt | BinOp::Gt => 0.40,
                BinOp::Lte | BinOp::Gte => 0.50,
                _ => 0.50,
            };
            (1.5, s)
        }
        (Stage::Filter(_, _), BodyKernel::FieldChainCmpLit(keys, op, _)) => {
            let s = match op {
                BinOp::Eq => 0.10,
                BinOp::Neq => 0.90,
                BinOp::Lt | BinOp::Gt => 0.40,
                BinOp::Lte | BinOp::Gte => 0.50,
                _ => 0.50,
            };
            (1.0 + keys.len() as f64, s)
        }
        (Stage::Filter(_, _), BodyKernel::CurrentCmpLit(op, _)) => {
            let s = match op {
                BinOp::Eq => 0.10,
                BinOp::Neq => 0.90,
                BinOp::Lt | BinOp::Gt => 0.40,
                BinOp::Lte | BinOp::Gte => 0.50,
                _ => 0.50,
            };
            (0.8, s)
        }
        (Stage::Filter(_, _), BodyKernel::FieldRead(_)) => (1.0, 0.7),
        (Stage::Filter(_, _), BodyKernel::ConstBool(b)) => (0.1, if *b { 1.0 } else { 0.0 }),
        _ => {
            let sh = stage.shape();
            (sh.cost, sh.selectivity)
        }
    }
}

// Sorts consecutive runs of non-generic filter stages by ascending cost/selectivity ratio.
fn reorder_filter_runs(
    stages: &mut Vec<Stage>,
    exprs: &mut Vec<Option<Arc<Expr>>>,
    kernels: &mut Vec<BodyKernel>,
) {
    let mut i = 0;
    while i < stages.len() {
        let mut j = i;
        while j < stages.len()
            && matches!(stages[j], Stage::Filter(_, _))
            && !matches!(kernels.get(j), Some(BodyKernel::Generic) | None)
        {
            j += 1;
        }
        if j - i >= 2 {
            let mut run: Vec<(Stage, Option<Arc<Expr>>, BodyKernel)> = Vec::with_capacity(j - i);
            for idx in i..j {
                run.push((
                    stages[idx].clone(),
                    exprs[idx].clone(),
                    kernels[idx].clone(),
                ));
            }
            run.sort_by(|a, b| {
                let (ca, sa) = kernel_cost_selectivity(&a.0, &a.2);
                let (cb, sb) = kernel_cost_selectivity(&b.0, &b.2);
                let ra = ca / (1.0 - sa).max(1e-6);
                let rb = cb / (1.0 - sb).max(1e-6);
                ra.partial_cmp(&rb).unwrap_or(std::cmp::Ordering::Equal)
            });
            for (idx, (s, e, k)) in run.into_iter().enumerate() {
                stages[i + idx] = s;
                exprs[i + idx] = e;
                kernels[i + idx] = k;
            }
        }
        i = j.max(i + 1);
    }
}

// Merges consecutive Filter stages into a single filter with an AndOp program.
fn fuse_filter_runs(
    stages: &mut Vec<Stage>,
    exprs: &mut Vec<Option<Arc<Expr>>>,
    kernels: &mut Vec<BodyKernel>,
) {
    let mut i = 0;
    while i < stages.len() {
        let mut j = i;
        while j < stages.len() && matches!(stages[j], Stage::Filter(_, _)) {
            j += 1;
        }
        if j - i < 2 {
            i = j.max(i + 1);
            continue;
        }

        let merged = merge_filter_programs(&stages[i..j]);
        let merged_kernel = BodyKernel::classify(&merged);
        stages[i] = Stage::Filter(merged, BuiltinViewStage::Filter);
        exprs[i] = None;
        kernels[i] = merged_kernel;
        stages.drain(i + 1..j);
        exprs.drain(i + 1..j);
        kernels.drain(i + 1..j);
        i += 1;
    }
}

fn merge_filter_programs(filters: &[Stage]) -> Arc<Program> {
    let mut iter = filters.iter();
    let Some(Stage::Filter(first, _)) = iter.next() else {
        unreachable!("filter run contains only filter stages")
    };
    let mut ops: Vec<Opcode> = first.ops.as_ref().to_vec();
    for stage in iter {
        let Stage::Filter(prog, _) = stage else {
            unreachable!("filter run contains only filter stages")
        };
        ops.push(Opcode::AndOp(Arc::clone(prog)));
    }
    Arc::new(Program {
        ops: ops.into(),
        source: first.source.clone(),
        id: 0,
        is_structural: false,
        ics: first.ics.clone(),
    })
}

// Removes constant-true filters, then fuses or cancels adjacent stage pairs.
fn fold_merge_with_kernels(
    stages: &mut Vec<Stage>,
    exprs: &mut Vec<Option<Arc<Expr>>>,
    kernels: &mut Vec<BodyKernel>,
) {
    let mut i = 0;
    while i < stages.len() {
        if matches!(&stages[i], Stage::Filter(_, _))
            && matches!(kernels.get(i), Some(BodyKernel::ConstBool(true)))
        {
            stages.remove(i);
            exprs.remove(i);
            kernels.remove(i);
        } else {
            i += 1;
        }
    }

    let mut i = 0;
    while i + 1 < stages.len() {
        if stages[i].cancels_with(&stages[i + 1]) {
            stages.drain(i..=i + 1);
            exprs.drain(i..=i + 1);
            kernels.drain(i..=i + 1);
            i = i.saturating_sub(1);
            continue;
        }
        if let Some(merged) = stages[i].merge_with(&stages[i + 1]) {
            stages[i] = merged;
            stages.remove(i + 1);
            exprs[i] = None;
            exprs.remove(i + 1);
            kernels.remove(i + 1);
            continue;
        }
        i += 1;
    }
}

/// Chooses the top-level execution `Strategy` for a planned pipeline by inspecting cardinality,
/// indexed support, and pull demand of the stages and sink.
pub fn select_strategy(stages: &[Stage], sink: &Sink) -> Strategy {
    use crate::parse::chain_ir::Cardinality;

    let stages_can_indexed = stages.iter().all(|s| s.shape().can_indexed);
    let sink_positional = sink.demand().positional.is_some();
    let has_barrier = stages
        .iter()
        .any(|s| matches!(s.shape().cardinality, Cardinality::Barrier));
    let has_short_circuit = matches!(
        sink.demand().chain.pull,
        PullDemand::FirstInput(_) | PullDemand::LastInput(_) | PullDemand::NthInput(_)
    );

    if has_barrier {
        return Strategy::BarrierMaterialise;
    }
    if stages_can_indexed && sink_positional {
        return Strategy::IndexedDispatch;
    }
    if has_short_circuit {
        return Strategy::EarlyExit;
    }
    Strategy::PullLoop
}

/// Selects the physical execution path for a pipeline at lower time, eliminating runtime
/// fallthrough for paths that static analysis proves cannot fire.
///
/// Priority: Indexed > Columnar > Composed > Legacy.  Each variant is the first path that
/// *might* fire; paths ranked above it are guaranteed not to apply for this pipeline shape.
pub fn select_exec_path(stages: &[Stage], sink: &Sink) -> PhysicalExecPath {
    // Indexed: all stages support position access and sink is positional.
    if select_strategy(stages, sink) == Strategy::IndexedDispatch {
        return PhysicalExecPath::Indexed;
    }

    // Columnar: at least one stage has a BuiltinColumnarStage variant, meaning an ObjVec /
    // IntVec / StrVec / FloatVec fast path exists for it.
    let columnar_eligible = stages.iter().any(|s| {
        s.descriptor().is_some_and(|d| d.columnar_stage().is_some())
    });
    if columnar_eligible {
        return PhysicalExecPath::Columnar;
    }

    // Composed: any non-empty stage list can be driven through the composed segment loop,
    // provided the source resolves to an array at runtime.
    if !stages.is_empty() {
        return PhysicalExecPath::Composed;
    }

    PhysicalExecPath::Legacy
}
