use super::report::{PipelineInspection, PipelineStageInspection};
use crate::exec::pipeline::{
    FallbackBoundary, PhysicalExecPath, PipelineBody, Sink, Source, Stage,
};
use crate::data::value::Val;
use crate::ir::physical::{PipelinePlanSource, PlanNode, QueryPlan};

pub(crate) fn inspect_first_pipeline(plan: &QueryPlan) -> Option<PipelineInspection> {
    plan.node_ids().find_map(|id| {
        let PlanNode::Pipeline { source, body } = plan.node(id) else {
            return None;
        };
        Some(inspect_pipeline(source, body))
    })
}

pub(crate) fn inspect_pipeline(
    source: &PipelinePlanSource,
    body: &PipelineBody,
) -> PipelineInspection {
    let pipeline = body.clone().with_source(source_for_inspection(source));
    PipelineInspection {
        source: source_label(source),
        stages: pipeline
            .stages
            .iter()
            .enumerate()
            .map(|(index, stage)| inspect_stage(index, stage))
            .collect(),
        sink: sink_label(&pipeline.sink).to_string(),
        source_demand: format!("{:?}", pipeline.source_demand().chain.pull),
        payload_demand: format!("{:?}", pipeline.payload_demand()),
        source_access: format!("{:?}", pipeline.source_access()),
        source_capabilities: format!("{:?}", pipeline.source_capabilities()),
        payload_lanes_supported: pipeline.source_payload_lanes_supported(),
        selected_materialization_supported: pipeline.source_selected_materialization_supported(),
        fallback_boundary: fallback_boundary_label(pipeline.fallback_boundary()),
        execution_path: Some(exec_path_label(pipeline.exec_path).to_string()),
    }
}

fn source_for_inspection(source: &PipelinePlanSource) -> Source {
    match source {
        PipelinePlanSource::FieldChain { keys } => Source::FieldChain { keys: keys.clone() },
        PipelinePlanSource::Expr(_) => Source::Receiver(Val::Null),
    }
}

pub(crate) fn source_label(source: &PipelinePlanSource) -> String {
    match source {
        PipelinePlanSource::FieldChain { keys } => {
            let joined = keys
                .iter()
                .map(|key| key.as_ref())
                .collect::<Vec<_>>()
                .join(".");
            if joined.is_empty() {
                "field-chain:$".to_string()
            } else {
                format!("field-chain:$.{joined}")
            }
        }
        PipelinePlanSource::Expr(id) => format!("expr:{}", id.0),
    }
}

pub(crate) fn inspect_stage(index: usize, stage: &Stage) -> PipelineStageInspection {
    let (kind, detail) = stage_label(stage);
    PipelineStageInspection {
        index,
        kind: kind.to_string(),
        detail,
    }
}

pub(crate) fn sink_label(sink: &Sink) -> &'static str {
    match sink {
        Sink::Collect => "collect",
        Sink::Reducer(spec) => match spec.op {
            crate::exec::pipeline::ReducerOp::Count => "count",
            crate::exec::pipeline::ReducerOp::Numeric(crate::exec::pipeline::NumOp::Sum) => "sum",
            crate::exec::pipeline::ReducerOp::Numeric(crate::exec::pipeline::NumOp::Min) => "min",
            crate::exec::pipeline::ReducerOp::Numeric(crate::exec::pipeline::NumOp::Max) => "max",
            crate::exec::pipeline::ReducerOp::Numeric(crate::exec::pipeline::NumOp::Avg) => "avg",
        },
        Sink::Predicate(spec) => match spec.op {
            crate::builtins::BuiltinPredicateSink::Any => "any",
            crate::builtins::BuiltinPredicateSink::All => "all",
            crate::builtins::BuiltinPredicateSink::FindIndex => "find-index",
            crate::builtins::BuiltinPredicateSink::IndicesWhere => "indices-where",
            crate::builtins::BuiltinPredicateSink::FindOne => "find-one",
        },
        Sink::Membership(_) => "membership",
        Sink::ArgExtreme(spec) => match spec.op {
            crate::builtins::BuiltinArgExtremeSink::MaxBy => "max-by",
            crate::builtins::BuiltinArgExtremeSink::MinBy => "min-by",
        },
        Sink::Terminal(_) => "terminal",
        Sink::SelectMany { from_end: true, .. } => "take-last",
        Sink::SelectMany {
            from_end: false, ..
        } => "take",
        Sink::Nth(_) => "nth",
        Sink::ApproxCountDistinct => "approx-count-distinct",
    }
}

fn stage_label(stage: &Stage) -> (&'static str, Option<String>) {
    match stage {
        Stage::Filter(_, view) => ("filter", Some(format!("view={view:?}"))),
        Stage::Map(_, view) => ("map", Some(format!("view={view:?}"))),
        Stage::FlatMap(_, view) => ("flat-map", Some(format!("view={view:?}"))),
        Stage::Reverse(_) => ("reverse", None),
        Stage::UniqueBy(Some(_)) => ("unique-by", Some("keyed".to_string())),
        Stage::UniqueBy(None) => ("unique", None),
        Stage::Sort(spec) => (
            "sort",
            Some(format!(
                "descending={}, keyed={}",
                spec.descending,
                spec.key.is_some()
            )),
        ),
        Stage::Builtin(call) => ("builtin", Some(format!("{:?}", call.method))),
        Stage::UsizeBuiltin { method, value } => {
            ("builtin-usize", Some(format!("{method:?}({value})")))
        }
        Stage::StringBuiltin { method, value } => {
            ("builtin-string", Some(format!("{method:?}({value:?})")))
        }
        Stage::StringPairBuiltin {
            method,
            first,
            second,
        } => (
            "builtin-string-pair",
            Some(format!("{method:?}({first:?}, {second:?})")),
        ),
        Stage::IntRangeBuiltin { method, start, end } => (
            "builtin-range",
            Some(format!("{method:?}({start}, {end:?})")),
        ),
        Stage::CompiledMap(_) => ("compiled-map", None),
        Stage::ExprBuiltin { method, .. } => ("expr-builtin", Some(format!("{method:?}"))),
        Stage::SortedDedup(Some(_)) => ("sorted-dedup", Some("keyed".to_string())),
        Stage::SortedDedup(None) => ("sorted-dedup", None),
    }
}

fn fallback_boundary_label(boundary: FallbackBoundary) -> Option<String> {
    match boundary {
        FallbackBoundary::None => None,
        FallbackBoundary::LegacyStage { index } => Some(format!("legacy-stage:{index}")),
        FallbackBoundary::MaterializedExecution => Some("materialized-execution".to_string()),
    }
}

fn exec_path_label(path: PhysicalExecPath) -> &'static str {
    match path {
        PhysicalExecPath::Indexed => "indexed",
        PhysicalExecPath::Columnar => "columnar",
        PhysicalExecPath::Composed => "composed",
        PhysicalExecPath::Legacy => "legacy",
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn pipeline_inspection_exports_stage_sink_and_demand() {
        let plan = crate::plan::physical::plan_query_with_context(
            "$.books.filter(price > 10).take(2).map(title)",
            crate::plan::physical::PlanningContext::bytes(),
        );

        let pipeline = super::inspect_first_pipeline(&plan).expect("pipeline");

        assert_eq!(pipeline.source, "field-chain:$.books");
        assert!(!pipeline.stages.is_empty());
        assert!(!pipeline.sink.is_empty());
        assert!(!pipeline.source_demand.is_empty());
        assert!(!pipeline.payload_demand.is_empty());
        assert!(!pipeline.source_access.is_empty());
        assert!(!pipeline.source_capabilities.is_empty());
    }
}
