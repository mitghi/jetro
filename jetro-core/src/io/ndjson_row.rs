use super::RowError;
use crate::data::value::Val;
use crate::{Jetro, JetroEngine, JetroEngineError};

pub(super) fn collect_row_val(
    engine: &JetroEngine,
    document: &Jetro,
    plan: &crate::ir::physical::QueryPlan,
    line_no: u64,
) -> Result<Val, JetroEngineError> {
    engine
        .collect_prepared_val(document, plan)
        .map_err(|err| row_eval_error(line_no, err))
}

pub(super) fn parse_row(
    engine: &JetroEngine,
    line_no: u64,
    row: Vec<u8>,
) -> Result<Jetro, JetroEngineError> {
    engine
        .parse_bytes_lazy(row)
        .map_err(|err| row_parse_error(line_no, err))
}

pub(super) fn row_parse_error(line_no: u64, err: JetroEngineError) -> JetroEngineError {
    match err {
        JetroEngineError::Json(source) => RowError::InvalidJson { line_no, source }.into(),
        JetroEngineError::Eval(eval) => RowError::InvalidJsonMessage {
            line_no,
            message: eval.to_string(),
        }
        .into(),
        other => other,
    }
}

pub(super) fn row_eval_error(line_no: u64, err: crate::EvalError) -> JetroEngineError {
    let message = err.0;
    if message.starts_with("Invalid JSON:") {
        RowError::InvalidJsonMessage { line_no, message }.into()
    } else {
        crate::EvalError(message).into()
    }
}
