use super::report::{DirectPlanInspection, NdjsonInspection, RowStreamInspection};
use crate::io::{
    ndjson_route_plan, ndjson_writer_path_kind, NdjsonOptions, NdjsonRoutePlan, NdjsonSourceMode,
};
use crate::{JetroEngine, JetroEngineError};

pub(crate) fn inspect_ndjson_query(
    engine: &JetroEngine,
    query: &str,
    source: NdjsonSourceMode,
    options: NdjsonOptions,
) -> Result<NdjsonInspection, JetroEngineError> {
    let route = ndjson_route_plan(engine, source, query, options)?;
    let explain = route.explain().clone();
    let rows = match &route {
        NdjsonRoutePlan::Rows {
            plan: crate::io::NdjsonRowsFilePlan::Stream(plan),
            ..
        } => Some(inspect_row_stream(plan, explain.source.partitionable)),
        _ => None,
    };
    let mut direct_plans = Vec::new();
    if let Some(kind) = ndjson_writer_path_kind(engine, query) {
        direct_plans.push(DirectPlanInspection {
            kind: format!("writer:{kind}"),
            detail: None,
        });
    }

    Ok(NdjsonInspection {
        route_kind: explain.kind.to_string(),
        source: explain.source.to_string(),
        fallback_reason: explain.fallback_reason.map(|reason| reason.to_string()),
        writer_path: explain.writer_path.map(|kind| kind.to_string()),
        rows,
        direct_plans,
    })
}

fn inspect_row_stream(
    plan: &crate::io::RowStreamPlan,
    partition_available: bool,
) -> RowStreamInspection {
    RowStreamInspection {
        plan_kind: "stream".to_string(),
        source: row_source_label(plan.source).to_string(),
        direction: row_direction_label(plan.direction).to_string(),
        demand: row_demand_label(&plan.demand),
        file_strategy: Some(file_strategy_label(plan.file_strategy(partition_available))),
        stages: plan
            .stages
            .iter()
            .enumerate()
            .map(|(index, stage)| row_stage_inspection(index, stage))
            .collect(),
        sink: row_sink_label(&plan.stages).to_string(),
    }
}

fn row_stage_inspection(
    index: usize,
    stage: &crate::io::RowStreamStage,
) -> super::report::PipelineStageInspection {
    let (kind, detail) = match stage {
        crate::io::RowStreamStage::Filter(_) => ("filter", None),
        crate::io::RowStreamStage::DistinctBy(_) => ("distinct-by", None),
        crate::io::RowStreamStage::Take(n) => ("take", Some(n.to_string())),
        crate::io::RowStreamStage::Map(_) => ("map", None),
        crate::io::RowStreamStage::Last => ("last", None),
        crate::io::RowStreamStage::Count => ("count", None),
        crate::io::RowStreamStage::Numeric(reducer) => (numeric_reducer_label(*reducer), None),
        crate::io::RowStreamStage::PredicateSink { sink, .. } => {
            (predicate_sink_label(*sink), None)
        }
    };
    super::report::PipelineStageInspection {
        index,
        kind: kind.to_string(),
        detail,
    }
}

fn row_source_label(source: crate::io::RowStreamSourceKind) -> &'static str {
    match source {
        crate::io::RowStreamSourceKind::DocumentRows => "document-rows",
        crate::io::RowStreamSourceKind::NdjsonRows => "ndjson-rows",
    }
}

fn row_direction_label(direction: crate::io::RowStreamDirection) -> &'static str {
    match direction {
        crate::io::RowStreamDirection::Forward => "forward",
        crate::io::RowStreamDirection::Reverse => "reverse",
    }
}

fn row_demand_label(demand: &crate::io::RowStreamDemand) -> String {
    format!(
        "retained={:?}, scalar={}, predicates={}, keys={}, projectors={}, ordered_early_stop={}, parallel={}",
        demand.retained_limit,
        demand.scalar_output,
        demand.predicate_count,
        demand.key_count,
        demand.projector_count,
        demand.ordered_early_stop,
        row_parallel_label(demand.parallel),
    )
}

fn row_parallel_label(parallel: crate::io::RowStreamParallelism) -> String {
    match parallel {
        crate::io::RowStreamParallelism::Sequential => "sequential".to_string(),
        crate::io::RowStreamParallelism::PartitionFilter {
            retained_limit,
            direction,
        } => format!(
            "partition-filter(retained={retained_limit:?}, direction={})",
            row_direction_label(direction)
        ),
    }
}

fn file_strategy_label(strategy: crate::io::RowStreamFileStrategy) -> String {
    match strategy {
        crate::io::RowStreamFileStrategy::Sequential => "sequential".to_string(),
        crate::io::RowStreamFileStrategy::Partitioned { retained_limit } => {
            format!("partitioned(retained={retained_limit})")
        }
        crate::io::RowStreamFileStrategy::OrderedPartitionSearch {
            direction,
            retained_limit,
        } => format!(
            "ordered-partition-search(direction={}, retained={retained_limit})",
            row_direction_label(direction)
        ),
    }
}

fn row_sink_label(stages: &[crate::io::RowStreamStage]) -> &'static str {
    match stages.last() {
        Some(crate::io::RowStreamStage::Take(_)) => "take",
        Some(crate::io::RowStreamStage::Map(_)) => "collect",
        Some(crate::io::RowStreamStage::Last) => "last",
        Some(crate::io::RowStreamStage::Count) => "count",
        Some(crate::io::RowStreamStage::Numeric(reducer)) => numeric_reducer_label(*reducer),
        Some(crate::io::RowStreamStage::PredicateSink { sink, .. }) => predicate_sink_label(*sink),
        _ => "collect",
    }
}

fn predicate_sink_label(sink: crate::builtins::BuiltinPredicateSink) -> &'static str {
    match sink {
        crate::builtins::BuiltinPredicateSink::Any => "any",
        crate::builtins::BuiltinPredicateSink::All => "all",
        crate::builtins::BuiltinPredicateSink::FindOne => "find-one",
        crate::builtins::BuiltinPredicateSink::FindIndex => "find-index",
        crate::builtins::BuiltinPredicateSink::IndicesWhere => "indices-where",
    }
}

fn numeric_reducer_label(reducer: crate::builtins::BuiltinNumericReducer) -> &'static str {
    match reducer {
        crate::builtins::BuiltinNumericReducer::Sum => "sum",
        crate::builtins::BuiltinNumericReducer::Avg => "avg",
        crate::builtins::BuiltinNumericReducer::Min => "min",
        crate::builtins::BuiltinNumericReducer::Max => "max",
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn ndjson_inspection_exports_ordered_reverse_row_stream() {
        let engine = crate::JetroEngine::new();
        let inspection = super::inspect_ndjson_query(
            &engine,
            r#"$.rows().reverse().find(@.custom_attributes.find(@.value == "z"))"#,
            crate::io::NdjsonSourceMode::File,
            crate::io::NdjsonOptions::default(),
        )
        .expect("inspect");

        assert_eq!(inspection.route_kind, "rows-stream");
        let rows = inspection.rows.expect("row stream");
        assert_eq!(rows.direction, "reverse");
        assert_eq!(
            rows.file_strategy,
            Some("ordered-partition-search(direction=reverse, retained=1)".to_string())
        );
    }
}
