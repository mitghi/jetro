use std::sync::Arc;

use crate::ast::Expr;
use crate::chain_ir::{ChainOp, Demand as ChainDemand, PullDemand, ValueNeed};

use super::{
    normalize::normalize_symbolic, BodyKernel, Pipeline, Sink, Stage, ViewSinkCapability,
    ViewStageCapability,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Position {
    First,
    Last,
}

#[derive(Debug, Clone, Copy)]
pub struct SinkDemand {
    pub chain: ChainDemand,
    pub positional: Option<Position>,
}

impl SinkDemand {
    pub const RESULT: SinkDemand = SinkDemand {
        chain: ChainDemand::RESULT,
        positional: None,
    };
}

impl Sink {
    pub fn demand(&self) -> SinkDemand {
        match self {
            Sink::First => SinkDemand {
                chain: ChainDemand::first(ValueNeed::Whole),
                positional: Some(Position::First),
            },
            Sink::Last => SinkDemand {
                chain: ChainDemand {
                    pull: PullDemand::All,
                    value: ValueNeed::Whole,
                    order: true,
                },
                positional: Some(Position::Last),
            },
            Sink::Count => SinkDemand {
                chain: ChainDemand {
                    pull: PullDemand::All,
                    value: ValueNeed::None,
                    order: false,
                },
                positional: None,
            },
            Sink::Numeric(_) => SinkDemand {
                chain: ChainDemand {
                    pull: PullDemand::All,
                    value: ValueNeed::Numeric,
                    order: false,
                },
                positional: None,
            },
            Sink::Collect | Sink::ApproxCountDistinct => SinkDemand::RESULT,
        }
    }

    pub(crate) fn can_run_with_receiver_only<F>(&self, mut program_ok: F) -> bool
    where
        F: FnMut(&crate::vm::Program) -> bool,
    {
        match self {
            Sink::Collect | Sink::Count | Sink::First | Sink::Last | Sink::ApproxCountDistinct => {
                true
            }
            Sink::Numeric(n) => n.project.as_ref().is_none_or(|prog| program_ok(prog)),
        }
    }

    pub(crate) fn view_capability(
        &self,
        sink_kernels: &[BodyKernel],
    ) -> Option<ViewSinkCapability> {
        match self {
            Sink::Collect => Some(ViewSinkCapability::Collect),
            Sink::Count => Some(ViewSinkCapability::Count),
            Sink::First => Some(ViewSinkCapability::First),
            Sink::Last => Some(ViewSinkCapability::Last),
            Sink::Numeric(n) => {
                let project_kernel = if n.project.is_some() {
                    Some(sink_kernels.first()?.is_view_native().then_some(0)?)
                } else {
                    None
                };
                Some(ViewSinkCapability::Numeric {
                    op: n.op,
                    project_kernel,
                })
            }
            Sink::ApproxCountDistinct => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum StageStrategy {
    Default,
    SortTopK(usize),
    SortBottomK(usize),
}

pub fn compute_strategies(stages: &[Stage], sink: &Sink) -> Vec<StageStrategy> {
    let mut strategies: Vec<StageStrategy> = vec![StageStrategy::Default; stages.len()];
    let mut demand = sink.demand();
    for (i, stage) in stages.iter().enumerate().rev() {
        if let Stage::Sort(_) = stage {
            match demand.chain.pull {
                PullDemand::AtMost(k) | PullDemand::UntilOutput(k) => {
                    strategies[i] = match demand.positional {
                        Some(Position::Last) => StageStrategy::SortBottomK(k),
                        _ => StageStrategy::SortTopK(k),
                    };
                }
                PullDemand::All => {}
            }
        }
        demand = stage.upstream_demand(demand);
    }
    strategies
}

impl Pipeline {
    pub fn segment_source_demand(stages: &[Stage], sink: &Sink) -> SinkDemand {
        stages
            .iter()
            .rev()
            .fold(sink.demand(), |demand, stage| stage.upstream_demand(demand))
    }

    pub fn source_demand(&self) -> SinkDemand {
        Self::segment_source_demand(&self.stages, &self.sink)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StageShape {
    pub cardinality: crate::chain_ir::Cardinality,
    pub can_indexed: bool,
    pub cost: f64,
    pub selectivity: f64,
}

impl Stage {
    pub(crate) fn view_capability(
        &self,
        idx: usize,
        kernel: Option<&BodyKernel>,
    ) -> Option<ViewStageCapability> {
        match self {
            Stage::Filter(_) if kernel?.is_view_native() => {
                Some(ViewStageCapability::Filter { kernel: idx })
            }
            Stage::Map(_) if kernel?.is_view_native() => {
                Some(ViewStageCapability::Map { kernel: idx })
            }
            Stage::Take(n) => Some(ViewStageCapability::Take(*n)),
            Stage::Skip(n) => Some(ViewStageCapability::Skip(*n)),
            _ => None,
        }
    }

    pub(crate) fn can_run_with_receiver_only<F>(&self, mut program_ok: F) -> bool
    where
        F: FnMut(&crate::vm::Program) -> bool,
    {
        match self {
            Stage::Filter(prog)
            | Stage::Map(prog)
            | Stage::FlatMap(prog)
            | Stage::Sort(Some(prog))
            | Stage::UniqueBy(Some(prog))
            | Stage::GroupBy(prog)
            | Stage::TakeWhile(prog)
            | Stage::DropWhile(prog)
            | Stage::IndicesWhere(prog)
            | Stage::FindIndex(prog)
            | Stage::MaxBy(prog)
            | Stage::MinBy(prog)
            | Stage::TransformValues(prog)
            | Stage::TransformKeys(prog)
            | Stage::FilterValues(prog)
            | Stage::FilterKeys(prog)
            | Stage::CountBy(prog)
            | Stage::IndexBy(prog)
            | Stage::SortedDedup(Some(prog)) => program_ok(prog),
            Stage::Take(_)
            | Stage::Skip(_)
            | Stage::Reverse
            | Stage::Sort(None)
            | Stage::UniqueBy(None)
            | Stage::Builtin(_)
            | Stage::Split(_)
            | Stage::Slice(_, _)
            | Stage::Replace { .. }
            | Stage::Chunk(_)
            | Stage::Window(_)
            | Stage::SortedDedup(None) => true,
            Stage::CompiledMap(_) => false,
        }
    }

    pub fn chain_op(&self) -> Option<ChainOp> {
        match self {
            Stage::Filter(_) => Some(ChainOp::Filter),
            Stage::Map(_) | Stage::CompiledMap(_) => Some(ChainOp::Map),
            Stage::FlatMap(_) | Stage::Split(_) => Some(ChainOp::FlatMap),
            Stage::Take(n) => Some(ChainOp::Take(*n)),
            Stage::Skip(n) => Some(ChainOp::Skip(*n)),
            Stage::Builtin(call) => Some(ChainOp::Builtin(call.method)),
            Stage::Slice(_, _) | Stage::Replace { .. } => Some(ChainOp::Map),
            _ => None,
        }
    }

    pub fn upstream_demand(&self, demand: SinkDemand) -> SinkDemand {
        let chain = match self.chain_op() {
            Some(op) => op.propagate_demand(demand.chain),
            None => ChainDemand::RESULT,
        };
        let positional = if matches!(
            self.shape().cardinality,
            crate::chain_ir::Cardinality::OneToOne
        ) {
            demand.positional
        } else {
            None
        };
        SinkDemand { chain, positional }
    }

    pub fn shape(&self) -> StageShape {
        use crate::chain_ir::Cardinality;
        match self {
            Stage::Map(_) => StageShape {
                cardinality: Cardinality::OneToOne,
                can_indexed: true,
                cost: 10.0,
                selectivity: 1.0,
            },
            Stage::Filter(_) => StageShape {
                cardinality: Cardinality::Filtering,
                can_indexed: false,
                cost: 10.0,
                selectivity: 0.5,
            },
            Stage::FlatMap(_) => StageShape {
                cardinality: Cardinality::Expanding,
                can_indexed: false,
                cost: 10.0,
                selectivity: 1.0,
            },
            Stage::Take(_) | Stage::Skip(_) => StageShape {
                cardinality: Cardinality::Bounded,
                can_indexed: false,
                cost: 0.5,
                selectivity: 0.5,
            },
            Stage::Reverse | Stage::Sort(_) | Stage::UniqueBy(_) | Stage::GroupBy(_) => {
                StageShape {
                    cardinality: Cardinality::Barrier,
                    can_indexed: false,
                    cost: 20.0,
                    selectivity: 1.0,
                }
            }
            Stage::Split(_) => StageShape {
                cardinality: Cardinality::Expanding,
                can_indexed: true,
                cost: 2.0,
                selectivity: 1.0,
            },
            Stage::Slice(_, _) => StageShape {
                cardinality: Cardinality::OneToOne,
                can_indexed: true,
                cost: 1.0,
                selectivity: 1.0,
            },
            Stage::Replace { .. } => StageShape {
                cardinality: Cardinality::OneToOne,
                can_indexed: true,
                cost: 2.0,
                selectivity: 1.0,
            },
            Stage::Chunk(_) | Stage::Window(_) => StageShape {
                cardinality: Cardinality::Barrier,
                can_indexed: true,
                cost: 2.0,
                selectivity: 1.0,
            },
            Stage::CompiledMap(_) => StageShape {
                cardinality: Cardinality::OneToOne,
                can_indexed: true,
                cost: 10.0,
                selectivity: 1.0,
            },
            Stage::Builtin(call) => {
                let spec = call.spec();
                StageShape {
                    cardinality: Cardinality::OneToOne,
                    can_indexed: spec.can_indexed,
                    cost: spec.cost,
                    selectivity: 1.0,
                }
            }
            Stage::TakeWhile(_)
            | Stage::DropWhile(_)
            | Stage::IndicesWhere(_)
            | Stage::FindIndex(_)
            | Stage::MaxBy(_)
            | Stage::MinBy(_)
            | Stage::TransformValues(_)
            | Stage::TransformKeys(_)
            | Stage::FilterValues(_)
            | Stage::FilterKeys(_)
            | Stage::CountBy(_)
            | Stage::IndexBy(_)
            | Stage::SortedDedup(_) => StageShape {
                cardinality: Cardinality::OneToOne,
                can_indexed: true,
                cost: 1.0,
                selectivity: 1.0,
            },
        }
    }

    pub fn merge_with(&self, other: &Self) -> Option<Self> {
        match (self, other) {
            (Stage::Take(a), Stage::Take(b)) => Some(Stage::Take((*a).min(*b))),
            (Stage::Skip(a), Stage::Skip(b)) => Some(Stage::Skip((*a).saturating_add(*b))),
            (Stage::Sort(_), Stage::Sort(b)) => Some(Stage::Sort(b.clone())),
            (Stage::UniqueBy(_), Stage::UniqueBy(b)) => Some(Stage::UniqueBy(b.clone())),
            (Stage::UniqueBy(None), Stage::Sort(None))
            | (Stage::Sort(None), Stage::UniqueBy(None)) => Some(Stage::SortedDedup(None)),
            (Stage::UniqueBy(Some(a)), Stage::Sort(Some(b)))
            | (Stage::Sort(Some(a)), Stage::UniqueBy(Some(b)))
                if Arc::ptr_eq(a, b) =>
            {
                Some(Stage::SortedDedup(Some(a.clone())))
            }
            (Stage::Builtin(a), Stage::Builtin(b)) if a.method == b.method && a.is_idempotent() => {
                Some(Stage::Builtin(a.clone()))
            }
            _ => None,
        }
    }

    pub fn cancels_with(&self, other: &Self) -> bool {
        if let (Stage::Builtin(a), Stage::Builtin(b)) = (self, other) {
            return a.cancels_with(b);
        }
        matches!((self, other), (Stage::Reverse, Stage::Reverse))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    IndexedDispatch,
    BarrierMaterialise,
    EarlyExit,
    PullLoop,
}

#[derive(Debug, Clone)]
pub struct Plan {
    pub stages: Vec<Stage>,
    pub stage_exprs: Vec<Option<Arc<Expr>>>,
    pub sink: Sink,
}

#[cfg(test)]
pub fn plan(stages: Vec<Stage>, sink: Sink) -> Plan {
    plan_with_kernels(stages, &[], sink)
}

pub fn plan_with_kernels(stages: Vec<Stage>, kernels: &[BodyKernel], sink: Sink) -> Plan {
    plan_with_exprs(stages, Vec::new(), kernels, sink)
}

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
    fold_merge_with_kernels(&mut stages, &mut e_buf, &mut k_buf);

    Plan {
        stages,
        stage_exprs: e_buf,
        sink,
    }
}

fn kernel_cost_selectivity(stage: &Stage, kernel: &BodyKernel) -> (f64, f64) {
    use crate::ast::BinOp;
    match (stage, kernel) {
        (Stage::Filter(_), BodyKernel::FieldCmpLit(_, op, _)) => {
            let s = match op {
                BinOp::Eq => 0.10,
                BinOp::Neq => 0.90,
                BinOp::Lt | BinOp::Gt => 0.40,
                BinOp::Lte | BinOp::Gte => 0.50,
                _ => 0.50,
            };
            (1.5, s)
        }
        (Stage::Filter(_), BodyKernel::FieldChainCmpLit(keys, op, _)) => {
            let s = match op {
                BinOp::Eq => 0.10,
                BinOp::Neq => 0.90,
                BinOp::Lt | BinOp::Gt => 0.40,
                BinOp::Lte | BinOp::Gte => 0.50,
                _ => 0.50,
            };
            (1.0 + keys.len() as f64, s)
        }
        (Stage::Filter(_), BodyKernel::CurrentCmpLit(op, _)) => {
            let s = match op {
                BinOp::Eq => 0.10,
                BinOp::Neq => 0.90,
                BinOp::Lt | BinOp::Gt => 0.40,
                BinOp::Lte | BinOp::Gte => 0.50,
                _ => 0.50,
            };
            (0.8, s)
        }
        (Stage::Filter(_), BodyKernel::FieldRead(_)) => (1.0, 0.7),
        (Stage::Filter(_), BodyKernel::ConstBool(b)) => (0.1, if *b { 1.0 } else { 0.0 }),
        _ => {
            let sh = stage.shape();
            (sh.cost, sh.selectivity)
        }
    }
}

fn reorder_filter_runs(
    stages: &mut Vec<Stage>,
    exprs: &mut Vec<Option<Arc<Expr>>>,
    kernels: &mut Vec<BodyKernel>,
) {
    let mut i = 0;
    while i < stages.len() {
        let mut j = i;
        while j < stages.len()
            && matches!(stages[j], Stage::Filter(_))
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

fn fold_merge_with_kernels(
    stages: &mut Vec<Stage>,
    exprs: &mut Vec<Option<Arc<Expr>>>,
    kernels: &mut Vec<BodyKernel>,
) {
    let mut i = 0;
    while i < stages.len() {
        if matches!(&stages[i], Stage::Filter(_))
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

pub fn select_strategy(stages: &[Stage], sink: &Sink) -> Strategy {
    use crate::chain_ir::Cardinality;

    let stages_can_indexed = stages.iter().all(|s| s.shape().can_indexed);
    let sink_positional = sink.demand().positional.is_some();
    let has_barrier = stages
        .iter()
        .any(|s| matches!(s.shape().cardinality, Cardinality::Barrier));
    let has_short_circuit = matches!(sink, Sink::First);

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
