use super::stream_fanout::{lower_rows_fanout_expr, RowStreamFanoutPlan};
use super::stream_plan::{lower_root_rows_query, RowStreamPlan, RowStreamSourceKind};
use super::stream_subquery::{lower_single_rows_subquery, RowStreamSubqueryPlan};
use crate::{EvalError, JetroEngineError};

pub(super) enum NdjsonRowsFilePlan {
    Stream(RowStreamPlan),
    Fanout(RowStreamFanoutPlan),
    Subquery(RowStreamSubqueryPlan),
}

pub(super) fn ndjson_rows_stream_plan(
    query: &str,
) -> Result<Option<RowStreamPlan>, JetroEngineError> {
    lower_root_rows_query(query, RowStreamSourceKind::NdjsonRows)
        .map_err(|err| JetroEngineError::Eval(EvalError(err.to_string())))
}

pub(super) fn ndjson_rows_file_plan(
    query: &str,
) -> Result<Option<NdjsonRowsFilePlan>, JetroEngineError> {
    if let Some(plan) = ndjson_rows_stream_plan(query)? {
        return Ok(Some(NdjsonRowsFilePlan::Stream(plan)));
    }
    if let Some(plan) = ndjson_rows_fanout_plan(query)? {
        return Ok(Some(NdjsonRowsFilePlan::Fanout(plan)));
    }
    if let Some(plan) = ndjson_rows_subquery_plan(query)? {
        return Ok(Some(NdjsonRowsFilePlan::Subquery(plan)));
    }
    Ok(None)
}

pub(super) fn ndjson_rows_subquery_plan(
    query: &str,
) -> Result<Option<RowStreamSubqueryPlan>, JetroEngineError> {
    let Some(expr) = parse_ndjson_rows_expr(query)? else {
        return Ok(None);
    };
    lower_single_rows_subquery(&expr, RowStreamSourceKind::NdjsonRows)
        .map_err(|err| JetroEngineError::Eval(EvalError(err.to_string())))
}

fn ndjson_rows_fanout_plan(
    query: &str,
) -> Result<Option<RowStreamFanoutPlan>, JetroEngineError> {
    let Some(expr) = parse_ndjson_rows_expr(query)? else {
        return Ok(None);
    };
    lower_rows_fanout_expr(&expr, RowStreamSourceKind::NdjsonRows)
        .map_err(|err| JetroEngineError::Eval(EvalError(err.to_string())))
}

fn parse_ndjson_rows_expr(query: &str) -> Result<Option<crate::parse::ast::Expr>, JetroEngineError> {
    if !query.contains("$.rows") {
        return Ok(None);
    }
    crate::parse::parser::parse(query)
        .map(Some)
        .map_err(|err| JetroEngineError::Eval(EvalError(err.to_string())))
}
