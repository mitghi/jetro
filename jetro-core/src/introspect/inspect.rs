use super::report::{
    InspectContext, InspectLevel, InspectOptions, InspectionSummary, LogicalInspection,
    QueryInspection,
};
use crate::io::{NdjsonOptions, NdjsonSourceMode};
use crate::parse::ast::Expr;
use crate::plan::physical::PlanningContext;
use crate::{JetroEngine, JetroEngineError};

pub(crate) fn inspect_query(
    engine: &JetroEngine,
    query: &str,
    options: InspectOptions,
    ndjson_options: NdjsonOptions,
) -> Result<QueryInspection, JetroEngineError> {
    let plan = engine.cached_plan(query, planning_context(options.context));
    let (summary, physical) = super::physical::inspect_physical_plan(&plan);
    let logical = (options.level != InspectLevel::Summary).then(|| inspect_logical(query));
    let physical = matches!(options.level, InspectLevel::Plan | InspectLevel::Detailed)
        .then_some(physical);
    let pipeline = (options.level == InspectLevel::Detailed)
        .then(|| super::pipeline::inspect_first_pipeline(&plan))
        .flatten();
    let ndjson = match options.context {
        InspectContext::NdjsonReader if options.level == InspectLevel::Detailed => Some(
            super::ndjson::inspect_ndjson_query(
                engine,
                query,
                NdjsonSourceMode::Reader,
                ndjson_options,
            )?,
        ),
        InspectContext::NdjsonFile if options.level == InspectLevel::Detailed => Some(
            super::ndjson::inspect_ndjson_query(
                engine,
                query,
                NdjsonSourceMode::File,
                ndjson_options,
            )?,
        ),
        _ => None,
    };

    Ok(QueryInspection {
        query: query.to_string(),
        context: options.context,
        level: options.level,
        summary: summary_with_context(summary, options.context),
        logical,
        physical,
        pipeline,
        ndjson,
        warnings: Vec::new(),
    })
}

fn planning_context(context: InspectContext) -> PlanningContext {
    match context {
        InspectContext::Value => PlanningContext::val(),
        InspectContext::Bytes | InspectContext::NdjsonReader | InspectContext::NdjsonFile => {
            PlanningContext::bytes()
        }
    }
}

fn summary_with_context(
    mut summary: InspectionSummary,
    context: InspectContext,
) -> InspectionSummary {
    if matches!(context, InspectContext::NdjsonReader | InspectContext::NdjsonFile) {
        summary.materializes_root = false;
    }
    summary
}

fn inspect_logical(query: &str) -> LogicalInspection {
    match crate::parse::parser::parse(query) {
        Ok(expr) => LogicalInspection {
            ast_root: expr_label(&expr).to_string(),
            root_shape: root_shape(&expr),
            notes: Vec::new(),
        },
        Err(err) => LogicalInspection {
            ast_root: "parse-error".to_string(),
            root_shape: "source-vm".to_string(),
            notes: vec![err.to_string()],
        },
    }
}

fn expr_label(expr: &Expr) -> &'static str {
    match expr {
        Expr::Null => "null",
        Expr::Bool(_) => "bool",
        Expr::Int(_) => "int",
        Expr::Float(_) => "float",
        Expr::Str(_) => "string",
        Expr::FString(_) => "f-string",
        Expr::Root => "root",
        Expr::Current => "current",
        Expr::Ident(_) => "ident",
        Expr::Chain(_, _) => "chain",
        Expr::BinOp(_, _, _) => "binary",
        Expr::UnaryNeg(_) => "unary-neg",
        Expr::Not(_) => "not",
        Expr::Kind { .. } => "kind",
        Expr::Coalesce(_, _) => "coalesce",
        Expr::Object(_) => "object",
        Expr::Array(_) => "array",
        Expr::Pipeline { .. } => "pipeline",
        Expr::ListComp { .. } => "list-comp",
        Expr::DictComp { .. } => "dict-comp",
        Expr::SetComp { .. } => "set-comp",
        Expr::GenComp { .. } => "gen-comp",
        Expr::Lambda { .. } => "lambda",
        Expr::Let { .. } => "let",
        Expr::IfElse { .. } => "if-else",
        Expr::Try { .. } => "try",
        Expr::GlobalCall { .. } => "global-call",
        Expr::Cast { .. } => "cast",
        Expr::Patch { .. } => "patch",
        Expr::UpdateBatch { .. } => "update-batch",
        Expr::DeleteMark => "delete-mark",
        Expr::Match { .. } => "match",
    }
}

fn root_shape(expr: &Expr) -> String {
    match expr {
        Expr::Chain(base, steps) => format!("{}+{} steps", expr_label(base), steps.len()),
        Expr::Object(fields) => format!("object({} fields)", fields.len()),
        Expr::Array(items) => format!("array({} items)", items.len()),
        other => expr_label(other).to_string(),
    }
}

#[cfg(test)]
mod tests {
    use crate::introspect::{InspectContext, InspectLevel, InspectOptions};

    #[test]
    fn inspect_query_assembles_static_plan_report() {
        let engine = crate::JetroEngine::new();
        let report = engine
            .inspect_query(
                "$.books.filter(price > 10).map(title)",
                InspectOptions::detailed(InspectContext::Bytes),
            )
            .expect("inspection");

        assert_eq!(report.level, InspectLevel::Detailed);
        assert!(report.physical.is_some());
        assert!(report.pipeline.is_some());
        assert!(report.ndjson.is_none());
    }

    #[test]
    fn public_ndjson_inspection_reports_rows_route() {
        let engine = crate::JetroEngine::new();
        let report = engine
            .inspect_ndjson_query_with_options(
                r#"$.rows().reverse().find(@.custom_attributes.find(@.value == "z"))"#,
                crate::io::NdjsonSourceMode::File,
                crate::io::NdjsonOptions::default(),
                InspectLevel::Detailed,
            )
            .expect("inspection");

        assert!(report.physical.is_some());
        assert_eq!(
            report.ndjson.as_ref().map(|ndjson| ndjson.route_kind.as_str()),
            Some("rows-stream")
        );
        assert!(report.format_tree().contains("ordered-partition-search"));
    }
}
