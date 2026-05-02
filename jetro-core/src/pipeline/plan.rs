use std::sync::Arc;

use crate::ast::{BinOp, Expr};
use crate::builtin_registry::{
    participates_in_demand, pipeline_materialization, pipeline_order_effect, pipeline_shape,
    BuiltinId,
};
use crate::builtins::{
    BuiltinMethod, BuiltinPipelineExecutor, BuiltinPipelineMaterialization,
    BuiltinPipelineOrderEffect, BuiltinSelectionPosition, BuiltinSinkAccumulator,
    BuiltinSinkDemand, BuiltinSinkSpec, BuiltinSinkValueNeed, BuiltinViewStage,
};
use crate::chain_ir::{ChainOp, Demand as ChainDemand, PullDemand, ValueNeed};
use crate::vm::{CompiledObjEntry, Opcode, Program};

use super::{
    normalize::normalize_symbolic, BodyKernel, Pipeline, PipelineBody, Sink, Stage,
    ViewSinkCapability, ViewStageCapability,
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
        if let Some(spec) = self.builtin_sink_spec() {
            return sink_demand_from_builtin(spec);
        }
        match self {
            Sink::Reducer(spec) => reducer_demand(spec.op),
            Sink::Collect | Sink::ApproxCountDistinct | Sink::Terminal(_) => SinkDemand::RESULT,
        }
    }

    pub(crate) fn can_run_with_receiver_only<F>(&self, mut program_ok: F) -> bool
    where
        F: FnMut(&crate::vm::Program) -> bool,
    {
        match self {
            Sink::Collect | Sink::Terminal(_) | Sink::ApproxCountDistinct => true,
            Sink::Reducer(spec) => spec.sink_programs().all(|prog| program_ok(prog)),
        }
    }

    pub(crate) fn view_capability(
        &self,
        sink_kernels: &[BodyKernel],
    ) -> Option<ViewSinkCapability> {
        match self {
            Sink::Collect => Some(ViewSinkCapability::Collect),
            Sink::Reducer(spec) if spec.numeric_op().is_some() => {
                let spec = self.reducer_spec()?;
                let sink_spec = spec.method()?.spec().sink?;
                if sink_spec.accumulator != BuiltinSinkAccumulator::Numeric {
                    return None;
                }
                let predicate_kernel = if let Some(idx) = spec.predicate_kernel_index() {
                    Some(sink_kernels.get(idx)?.is_view_native().then_some(idx)?)
                } else {
                    None
                };
                let project_kernel = if let Some(idx) = spec.projection_kernel_index() {
                    Some(sink_kernels.get(idx)?.is_view_native().then_some(idx)?)
                } else {
                    None
                };
                Some(ViewSinkCapability::from_sink_spec(
                    sink_spec,
                    predicate_kernel,
                    project_kernel,
                    Some(spec.numeric_op()?),
                ))
            }
            Sink::Reducer(spec) if spec.op == super::ReducerOp::Count => {
                let sink_spec = spec.method()?.spec().sink?;
                if sink_spec.accumulator != BuiltinSinkAccumulator::Count {
                    return None;
                }
                let predicate_kernel = if let Some(idx) = spec.predicate_kernel_index() {
                    Some(sink_kernels.get(idx)?.is_view_native().then_some(idx)?)
                } else {
                    None
                };
                Some(ViewSinkCapability::from_sink_spec(
                    sink_spec,
                    predicate_kernel,
                    None,
                    None,
                ))
            }
            Sink::Terminal(method) => Some(ViewSinkCapability::from_sink_spec(
                method.spec().sink?,
                None,
                None,
                None,
            )),
            Sink::Reducer(_) => None,
            Sink::ApproxCountDistinct => None,
        }
    }

    pub(crate) fn builtin_sink_spec(&self) -> Option<BuiltinSinkSpec> {
        match self {
            Sink::Terminal(method) => method.spec().sink,
            Sink::Reducer(spec) => spec.method()?.spec().sink,
            Sink::Collect | Sink::ApproxCountDistinct => None,
        }
    }
}

fn sink_demand_from_builtin(spec: BuiltinSinkSpec) -> SinkDemand {
    match spec.demand {
        BuiltinSinkDemand::First { value } => SinkDemand {
            chain: ChainDemand::first(sink_value_need(value)),
            positional: match spec.accumulator {
                BuiltinSinkAccumulator::SelectOne(position) => Some(position.into()),
                _ => None,
            },
        },
        BuiltinSinkDemand::All { value, order } => SinkDemand {
            chain: ChainDemand {
                pull: PullDemand::All,
                value: sink_value_need(value),
                order,
            },
            positional: match spec.accumulator {
                BuiltinSinkAccumulator::SelectOne(position) => Some(position.into()),
                _ => None,
            },
        },
    }
}

fn sink_value_need(value: BuiltinSinkValueNeed) -> ValueNeed {
    match value {
        BuiltinSinkValueNeed::None => ValueNeed::None,
        BuiltinSinkValueNeed::Whole => ValueNeed::Whole,
        BuiltinSinkValueNeed::Numeric => ValueNeed::Numeric,
    }
}

impl From<BuiltinSelectionPosition> for Position {
    fn from(value: BuiltinSelectionPosition) -> Self {
        match value {
            BuiltinSelectionPosition::First => Position::First,
            BuiltinSelectionPosition::Last => Position::Last,
        }
    }
}

fn reducer_demand(op: super::ReducerOp) -> SinkDemand {
    let value = match op {
        super::ReducerOp::Count => ValueNeed::None,
        super::ReducerOp::Numeric(_) => ValueNeed::Numeric,
    };
    SinkDemand {
        chain: ChainDemand {
            pull: PullDemand::All,
            value,
            order: false,
        },
        positional: None,
    }
}

#[derive(Debug, Clone, Copy)]
pub enum StageStrategy {
    Default,
    SortTopK(usize),
    SortBottomK(usize),
    SortUntilOutput(usize),
}

#[cfg(test)]
pub fn compute_strategies(stages: &[Stage], sink: &Sink) -> Vec<StageStrategy> {
    compute_strategies_with_kernels(stages, &[], sink)
}

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
                PullDemand::All => {}
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

fn predicate_is_order_prefix(
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

enum OrderKey<'a> {
    Current,
    Field(&'a str),
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

fn cmp_is_prefix_for_order(op: BinOp, descending: bool) -> bool {
    matches!(
        (descending, op),
        (true, BinOp::Gt | BinOp::Gte) | (false, BinOp::Lt | BinOp::Lte)
    )
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

impl PipelineBody {
    pub(crate) fn can_run_with_materialized_receiver(&self) -> bool {
        stages_can_run_with_materialized_receiver(&self.stages)
            && self
                .sink
                .can_run_with_receiver_only(program_is_current_only)
    }

    pub(crate) fn suffix_can_run_with_materialized_receiver(&self, consumed_stages: usize) -> bool {
        consumed_stages <= self.stages.len()
            && stages_can_run_with_materialized_receiver(&self.stages[consumed_stages..])
            && self
                .sink
                .can_run_with_receiver_only(program_is_current_only)
    }
}

fn stages_can_run_with_materialized_receiver(stages: &[Stage]) -> bool {
    stages
        .iter()
        .all(|stage| stage.can_run_with_receiver_only(program_is_current_only))
}

fn program_is_current_only(program: &Program) -> bool {
    program.ops.iter().all(opcode_is_current_only)
}

fn opcode_is_current_only(opcode: &Opcode) -> bool {
    match opcode {
        Opcode::PushRoot | Opcode::RootChain(_) | Opcode::GetPointer(_) => false,
        Opcode::BindVar(_)
        | Opcode::StoreVar(_)
        | Opcode::BindObjDestructure(_)
        | Opcode::BindArrDestructure(_)
        | Opcode::PipelineRun { .. }
        | Opcode::LetExpr { .. }
        | Opcode::ListComp(_)
        | Opcode::DictComp(_)
        | Opcode::SetComp(_)
        | Opcode::PatchEval(_) => false,
        Opcode::DynIndex(prog)
        | Opcode::InlineFilter(prog)
        | Opcode::AndOp(prog)
        | Opcode::OrOp(prog)
        | Opcode::CoalesceOp(prog) => program_is_current_only(prog),
        Opcode::CallMethod(call) | Opcode::CallOptMethod(call) => call
            .sub_progs
            .iter()
            .all(|prog| program_is_current_only(prog)),
        Opcode::IfElse { then_, else_ } => {
            program_is_current_only(then_) && program_is_current_only(else_)
        }
        Opcode::TryExpr { body, default } => {
            program_is_current_only(body) && program_is_current_only(default)
        }
        Opcode::MakeArr(items) => items
            .iter()
            .all(|(prog, _spread)| program_is_current_only(prog)),
        Opcode::FString(parts) => parts.iter().all(|part| match part {
            crate::vm::CompiledFSPart::Lit(_) => true,
            crate::vm::CompiledFSPart::Interp { prog, .. } => program_is_current_only(prog),
        }),
        Opcode::MakeObj(entries) => entries.iter().all(obj_entry_is_current_only),
        Opcode::PushNull
        | Opcode::PushBool(_)
        | Opcode::PushInt(_)
        | Opcode::PushFloat(_)
        | Opcode::PushStr(_)
        | Opcode::PushCurrent
        | Opcode::LoadIdent(_)
        | Opcode::GetField(_)
        | Opcode::GetIndex(_)
        | Opcode::GetSlice(_, _)
        | Opcode::OptField(_)
        | Opcode::Descendant(_)
        | Opcode::DescendAll
        | Opcode::Quantifier(_)
        | Opcode::FieldChain(_)
        | Opcode::Add
        | Opcode::Sub
        | Opcode::Mul
        | Opcode::Div
        | Opcode::Mod
        | Opcode::Eq
        | Opcode::Neq
        | Opcode::Lt
        | Opcode::Lte
        | Opcode::Gt
        | Opcode::Gte
        | Opcode::Fuzzy
        | Opcode::Not
        | Opcode::Neg
        | Opcode::CastOp(_)
        | Opcode::KindCheck { .. }
        | Opcode::SetCurrent
        | Opcode::DeleteMarkErr => true,
    }
}

fn obj_entry_is_current_only(entry: &CompiledObjEntry) -> bool {
    match entry {
        CompiledObjEntry::Short { .. } | CompiledObjEntry::KvPath { .. } => true,
        CompiledObjEntry::Kv { prog, cond, .. } => {
            program_is_current_only(prog)
                && cond
                    .as_ref()
                    .is_none_or(|cond| program_is_current_only(cond))
        }
        CompiledObjEntry::Dynamic { key, val } => {
            program_is_current_only(key) && program_is_current_only(val)
        }
        CompiledObjEntry::Spread(prog) | CompiledObjEntry::SpreadDeep(prog) => {
            program_is_current_only(prog)
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StageShape {
    pub cardinality: crate::chain_ir::Cardinality,
    pub can_indexed: bool,
    pub cost: f64,
    pub selectivity: f64,
}

impl StageShape {
    fn from_view_stage(stage: BuiltinViewStage) -> Self {
        Self {
            cardinality: stage.cardinality().into(),
            can_indexed: stage.can_indexed(),
            cost: stage.cost(),
            selectivity: stage.selectivity(),
        }
    }

    fn from_builtin(method: BuiltinMethod) -> Self {
        use crate::builtins::BuiltinCategory;

        let spec = method.spec();
        if let Some(shape) = pipeline_shape(BuiltinId::from_method(method)) {
            return Self {
                cardinality: shape.cardinality.into(),
                can_indexed: shape.can_indexed,
                cost: shape.cost,
                selectivity: shape.selectivity,
            };
        }
        Self {
            cardinality: spec.cardinality.into(),
            can_indexed: spec.can_indexed,
            cost: spec.cost,
            selectivity: if matches!(spec.category, BuiltinCategory::StreamingFilter) {
                0.5
            } else {
                1.0
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct StageDescriptor<'a> {
    pub method: Option<BuiltinMethod>,
    pub body: Option<&'a Program>,
    pub usize_arg: Option<usize>,
    pub executor_override: Option<BuiltinPipelineExecutor>,
    view_stage_override: Option<BuiltinViewStage>,
    allow_one_to_one_order_fallback: bool,
}

impl<'a> StageDescriptor<'a> {
    #[inline]
    fn new(method: BuiltinMethod) -> Self {
        Self {
            method: Some(method),
            body: None,
            usize_arg: None,
            executor_override: None,
            view_stage_override: None,
            allow_one_to_one_order_fallback: false,
        }
    }

    #[inline]
    fn special(executor: BuiltinPipelineExecutor) -> Self {
        Self {
            method: None,
            body: None,
            usize_arg: None,
            executor_override: Some(executor),
            view_stage_override: None,
            allow_one_to_one_order_fallback: false,
        }
    }

    #[inline]
    fn body(mut self, body: &'a Program) -> Self {
        self.body = Some(body);
        self
    }

    #[inline]
    fn usize_arg(mut self, usize_arg: usize) -> Self {
        self.usize_arg = Some(usize_arg);
        self
    }

    #[inline]
    fn with_view_stage(mut self, stage: BuiltinViewStage) -> Self {
        self.view_stage_override = Some(stage);
        self
    }

    #[inline]
    fn allow_one_to_one_order_fallback(mut self) -> Self {
        self.allow_one_to_one_order_fallback = true;
        self
    }

    #[inline]
    pub(crate) fn view_stage(self) -> Option<BuiltinViewStage> {
        self.view_stage_override
            .or_else(|| self.method.and_then(|method| method.spec().view_stage))
    }

    #[inline]
    pub(crate) fn columnar_stage(self) -> Option<crate::builtins::BuiltinColumnarStage> {
        self.method.and_then(|method| method.spec().columnar_stage)
    }

    #[inline]
    pub(crate) fn pipeline_materialization(self) -> BuiltinPipelineMaterialization {
        self.method
            .map(|method| pipeline_materialization(BuiltinId::from_method(method)))
            .unwrap_or(BuiltinPipelineMaterialization::Streaming)
    }

    #[inline]
    pub(crate) fn pipeline_order_effect(self) -> BuiltinPipelineOrderEffect {
        let Some(method) = self.method else {
            return BuiltinPipelineOrderEffect::Blocks;
        };
        let spec = method.spec();
        if let Some(effect) = pipeline_order_effect(BuiltinId::from_method(method)) {
            return effect;
        }
        if self.allow_one_to_one_order_fallback
            && spec.cardinality == crate::builtins::BuiltinCardinality::OneToOne
        {
            return BuiltinPipelineOrderEffect::Preserves;
        }
        BuiltinPipelineOrderEffect::Blocks
    }
}

macro_rules! body_stage_descriptor {
    ($stage:expr, { $($variant:ident => $method:ident),+ $(,)? }) => {
        match $stage {
            $(
                Stage::$variant(prog) => Some(StageDescriptor::new(BuiltinMethod::$method).body(prog)),
            )+
            _ => None,
        }
    };
}

macro_rules! view_body_stage_descriptor {
    ($stage:expr, { $($variant:ident => $method:ident),+ $(,)? }) => {
        match $stage {
            $(
                Stage::$variant(prog, view_stage) => {
                    Some(StageDescriptor::new(BuiltinMethod::$method).body(prog).with_view_stage(*view_stage))
                },
            )+
            _ => None,
        }
    };
}

macro_rules! usize_stage_descriptor {
    ($stage:expr, { $($variant:ident => $method:ident),+ $(,)? }) => {
        match $stage {
            $(
                Stage::$variant(n) => Some(StageDescriptor::new(BuiltinMethod::$method).usize_arg(*n)),
            )+
            _ => None,
        }
    };
}

macro_rules! method_stage_descriptor {
    ($stage:expr, { $($pattern:pat => $method:ident),+ $(,)? }) => {
        match $stage {
            $(
                $pattern => Some(StageDescriptor::new(BuiltinMethod::$method)),
            )+
            _ => None,
        }
    };
}

impl Stage {
    pub(crate) fn is_composed_barrier(&self) -> bool {
        self.pipeline_materialization() == BuiltinPipelineMaterialization::ComposedBarrier
    }

    pub(crate) fn requires_legacy_materialization(&self) -> bool {
        !matches!(
            self.pipeline_materialization(),
            BuiltinPipelineMaterialization::Streaming
        )
    }

    pub(crate) fn view_capability(
        &self,
        idx: usize,
        kernel: Option<&BodyKernel>,
    ) -> Option<ViewStageCapability> {
        let desc = self.descriptor()?;
        let stage = desc.view_stage()?;
        ViewStageCapability::from_stage_metadata(
            stage,
            desc.usize_arg,
            idx,
            kernel.is_some_and(BodyKernel::is_view_native),
        )
    }

    pub(crate) fn descriptor(&self) -> Option<StageDescriptor<'_>> {
        if let Some(desc) = view_body_stage_descriptor!(self, {
            Filter => Filter,
            Map => Map,
            FlatMap => FlatMap,
        }) {
            return Some(desc);
        }
        if let Some(desc) = body_stage_descriptor!(self, {
            GroupBy => GroupBy,
            TakeWhile => TakeWhile,
            DropWhile => DropWhile,
            IndicesWhere => IndicesWhere,
            FindIndex => FindIndex,
            MaxBy => MaxBy,
            MinBy => MinBy,
            TransformValues => TransformValues,
            TransformKeys => TransformKeys,
            FilterValues => FilterValues,
            FilterKeys => FilterKeys,
            CountBy => CountBy,
            IndexBy => IndexBy,
        }) {
            return Some(desc);
        }
        if let Some(desc) = usize_stage_descriptor!(self, {
            Chunk => Chunk,
            Window => Window,
        }) {
            return Some(desc);
        }
        if let Some(desc) = method_stage_descriptor!(self, {
            Stage::Reverse(_) => Reverse,
            Stage::UniqueBy(None) => Unique,
            Stage::Split(_) => Split,
            Stage::Slice(_, _) => Slice,
        }) {
            return Some(desc);
        }

        match self {
            Stage::Take(n, view_stage, _) => Some(
                StageDescriptor::new(BuiltinMethod::Take)
                    .usize_arg(*n)
                    .with_view_stage(*view_stage),
            ),
            Stage::Skip(n, view_stage, _) => Some(
                StageDescriptor::new(BuiltinMethod::Skip)
                    .usize_arg(*n)
                    .with_view_stage(*view_stage),
            ),
            Stage::UniqueBy(Some(prog)) => {
                Some(StageDescriptor::new(BuiltinMethod::UniqueBy).body(prog))
            }
            Stage::Sort(super::SortSpec { key, .. }) => {
                let desc = StageDescriptor::new(BuiltinMethod::Sort);
                Some(if let Some(prog) = key {
                    desc.body(prog)
                } else {
                    desc
                })
            }
            Stage::Replace { all, .. } => Some(StageDescriptor::new(if *all {
                BuiltinMethod::ReplaceAll
            } else {
                BuiltinMethod::Replace
            })),
            Stage::Builtin(call) => {
                Some(StageDescriptor::new(call.method).allow_one_to_one_order_fallback())
            }
            Stage::SortedDedup(prog) => {
                let desc = StageDescriptor::special(BuiltinPipelineExecutor::SortedDedup);
                Some(if let Some(prog) = prog {
                    desc.body(prog)
                } else {
                    desc
                })
            }
            Stage::CompiledMap(_) => None,
            _ => None,
        }
    }

    fn pipeline_materialization(&self) -> BuiltinPipelineMaterialization {
        match self {
            Stage::CompiledMap(_) => BuiltinPipelineMaterialization::Streaming,
            Stage::SortedDedup(_) => BuiltinPipelineMaterialization::LegacyMaterialized,
            _ => self
                .descriptor()
                .map(StageDescriptor::pipeline_materialization)
                .unwrap_or(BuiltinPipelineMaterialization::Streaming),
        }
    }

    pub(crate) fn can_run_with_receiver_only<F>(&self, mut program_ok: F) -> bool
    where
        F: FnMut(&crate::vm::Program) -> bool,
    {
        if let Some(prog) = self.body_program() {
            return program_ok(prog);
        }
        match self {
            Stage::Take(_, _, _)
            | Stage::Skip(_, _, _)
            | Stage::Reverse(_)
            | Stage::Sort(super::SortSpec { key: None, .. })
            | Stage::UniqueBy(None)
            | Stage::Builtin(_)
            | Stage::Split(_)
            | Stage::Slice(_, _)
            | Stage::Replace { .. }
            | Stage::Chunk(_)
            | Stage::Window(_)
            | Stage::SortedDedup(None) => true,
            Stage::CompiledMap(_) => false,
            _ => false,
        }
    }

    pub(crate) fn body_program(&self) -> Option<&crate::vm::Program> {
        self.descriptor().and_then(|desc| desc.body)
    }

    pub fn chain_op(&self) -> Option<ChainOp> {
        match self {
            Stage::CompiledMap(_) => Some(ChainOp::builtin(BuiltinMethod::Map)),
            Stage::SortedDedup(_) => None,
            _ => self.chain_demand_op(),
        }
    }

    fn chain_demand_op(&self) -> Option<ChainOp> {
        let desc = self.descriptor()?;
        let method = desc.method?;
        match self {
            _ if desc.usize_arg.is_some() => Some(ChainOp::builtin_usize(method, desc.usize_arg?)),
            Stage::Builtin(_) => Some(ChainOp::builtin(method)),
            _ if participates_in_demand(BuiltinId::from_method(method)) => {
                Some(ChainOp::builtin(method))
            }
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

    fn ordered_prefix_effect(
        &self,
        sort: &super::SortSpec,
        sort_kernel: &BodyKernel,
        kernel: &BodyKernel,
    ) -> bool {
        match self.pipeline_order_effect() {
            BuiltinPipelineOrderEffect::Preserves => true,
            BuiltinPipelineOrderEffect::PredicatePrefix => {
                predicate_is_order_prefix(sort, sort_kernel, kernel)
            }
            BuiltinPipelineOrderEffect::Blocks => false,
        }
    }

    fn pipeline_order_effect(&self) -> BuiltinPipelineOrderEffect {
        match self {
            Stage::CompiledMap(_) => BuiltinPipelineOrderEffect::Preserves,
            Stage::SortedDedup(_) => BuiltinPipelineOrderEffect::Blocks,
            _ => self
                .descriptor()
                .map(StageDescriptor::pipeline_order_effect)
                .unwrap_or(BuiltinPipelineOrderEffect::Blocks),
        }
    }

    pub fn shape(&self) -> StageShape {
        use crate::chain_ir::Cardinality;
        match self {
            Stage::CompiledMap(_) => StageShape {
                cardinality: Cardinality::OneToOne,
                can_indexed: true,
                cost: 10.0,
                selectivity: 1.0,
            },
            Stage::SortedDedup(_) => StageShape {
                cardinality: Cardinality::OneToOne,
                can_indexed: true,
                cost: 1.0,
                selectivity: 1.0,
            },
            _ => self.descriptor().map_or(
                StageShape {
                    cardinality: Cardinality::OneToOne,
                    can_indexed: false,
                    cost: 1.0,
                    selectivity: 1.0,
                },
                |desc| {
                    desc.view_stage()
                        .map(StageShape::from_view_stage)
                        .or_else(|| desc.method.map(StageShape::from_builtin))
                        .unwrap_or(StageShape {
                            cardinality: Cardinality::OneToOne,
                            can_indexed: false,
                            cost: 1.0,
                            selectivity: 1.0,
                        })
                },
            ),
        }
    }

    pub fn merge_with(&self, other: &Self) -> Option<Self> {
        if let Some(merged) = self.merge_with_usize_stage(other) {
            return Some(merged);
        }
        match (self, other) {
            (Stage::Sort(_), Stage::Sort(b)) => Some(Stage::Sort(b.clone())),
            (Stage::UniqueBy(_), Stage::UniqueBy(b)) => Some(Stage::UniqueBy(b.clone())),
            (Stage::UniqueBy(None), Stage::Sort(super::SortSpec { key: None, .. }))
            | (Stage::Sort(super::SortSpec { key: None, .. }), Stage::UniqueBy(None)) => {
                Some(Stage::SortedDedup(None))
            }
            (Stage::UniqueBy(Some(a)), Stage::Sort(super::SortSpec { key: Some(b), .. }))
            | (Stage::Sort(super::SortSpec { key: Some(a), .. }), Stage::UniqueBy(Some(b)))
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

    fn merge_with_usize_stage(&self, other: &Self) -> Option<Self> {
        let lhs = self.usize_stage_merge_parts()?;
        let rhs = other.usize_stage_merge_parts()?;
        if lhs.stage != rhs.stage || lhs.merge != rhs.merge {
            return None;
        }
        self.with_usize_stage_value(lhs.merge.combine_usize(lhs.value, rhs.value))
    }

    fn usize_stage_merge_parts(&self) -> Option<UsizeStageMergeParts> {
        match self {
            Stage::Take(value, stage, merge) | Stage::Skip(value, stage, merge) => {
                Some(UsizeStageMergeParts {
                    value: *value,
                    stage: *stage,
                    merge: *merge,
                })
            }
            _ => None,
        }
    }

    fn with_usize_stage_value(&self, value: usize) -> Option<Self> {
        match self {
            Stage::Take(_, stage, merge) => Some(Stage::Take(value, *stage, *merge)),
            Stage::Skip(_, stage, merge) => Some(Stage::Skip(value, *stage, *merge)),
            _ => None,
        }
    }

    pub fn cancels_with(&self, other: &Self) -> bool {
        match (self.cancellation(), other.cancellation()) {
            (Some(a), Some(b)) => a.cancels_with(b),
            _ => false,
        }
    }

    fn cancellation(&self) -> Option<crate::builtins::BuiltinCancellation> {
        match self {
            Stage::Reverse(cancel) => Some(*cancel),
            Stage::Builtin(call) => call.spec().cancellation,
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct UsizeStageMergeParts {
    value: usize,
    stage: BuiltinViewStage,
    merge: crate::builtins::BuiltinStageMerge,
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

pub fn select_strategy(stages: &[Stage], sink: &Sink) -> Strategy {
    use crate::chain_ir::Cardinality;

    let stages_can_indexed = stages.iter().all(|s| s.shape().can_indexed);
    let sink_positional = sink.demand().positional.is_some();
    let has_barrier = stages
        .iter()
        .any(|s| matches!(s.shape().cardinality, Cardinality::Barrier));
    let has_short_circuit = matches!(sink.demand().chain.pull, PullDemand::FirstInput(_));

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
