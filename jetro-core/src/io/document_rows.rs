use super::stream_exec::{CompiledRowStream, RowStreamRowResult};
use super::stream_plan::{
    lower_root_rows_query, RowStreamDirection, RowStreamPlan, RowStreamSourceKind,
};
use crate::data::value::Val;
use crate::{EvalError, Jetro, JetroEngine};
use std::borrow::Cow;
use std::sync::Arc;

pub(crate) fn collect_document_rows(
    engine: &JetroEngine,
    document: &Jetro,
    query: &str,
) -> Result<Option<Val>, EvalError> {
    let Some(plan) = document_rows_stream_plan(query)? else {
        return Ok(None);
    };

    let root = document.root_val_with(engine.keys())?;
    let mut rows = document_rows(root);
    if plan.direction == RowStreamDirection::Reverse {
        rows.reverse();
    }

    let mut stream = CompiledRowStream::new(&plan);
    let mut vm = engine.lock_vm();
    let mut out = Vec::new();
    for row in rows {
        if stream.is_exhausted() {
            break;
        }
        match stream.apply_val_row(&mut vm, row)? {
            RowStreamRowResult::Emit(value) => out.push(value),
            RowStreamRowResult::EmitBytes(bytes) => {
                return Err(EvalError(format!(
                    "internal rows() stream error: byte output in document mode ({} bytes)",
                    bytes.len()
                )));
            }
            RowStreamRowResult::Skip => {}
            RowStreamRowResult::Stop => break,
        }
    }

    Ok(Some(Val::Arr(Arc::new(out))))
}

fn document_rows(root: Val) -> Vec<Val> {
    match root.as_vals() {
        Some(Cow::Borrowed(rows)) => rows.to_vec(),
        Some(Cow::Owned(rows)) => rows,
        None => vec![root],
    }
}

fn document_rows_stream_plan(query: &str) -> Result<Option<RowStreamPlan>, EvalError> {
    lower_root_rows_query(query, RowStreamSourceKind::DocumentRows)
        .map_err(|err| EvalError(err.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn document_rows_array_maps_elements() {
        let engine = JetroEngine::new();
        let document = engine.parse_value(json!([
            {"id": 1, "name": "Ada"},
            {"id": 2, "name": "Bob"},
            {"id": 3, "name": "Cid"}
        ]));

        let out = collect_document_rows(&engine, &document, "$.rows().take(2).map($.name)")
            .unwrap()
            .unwrap();

        assert_eq!(serde_json::Value::from(out), json!(["Ada", "Bob"]));
    }

    #[test]
    fn document_rows_object_is_single_row() {
        let engine = JetroEngine::new();
        let document = engine.parse_value(json!({"id": 1}));

        let out = collect_document_rows(&engine, &document, "$.rows().map($.id)")
            .unwrap()
            .unwrap();

        assert_eq!(serde_json::Value::from(out), json!([1]));
    }

    #[test]
    fn document_rows_reverse_distinct_keeps_stream_order() {
        let engine = JetroEngine::new();
        let document = engine.parse_value(json!([
            {"id": "a", "v": 1},
            {"id": "b", "v": 2},
            {"id": "a", "v": 3},
            {"id": "c", "v": 4}
        ]));

        let out = collect_document_rows(
            &engine,
            &document,
            "$.rows().reverse().distinct_by($.id).take(2).map($.v)",
        )
        .unwrap()
        .unwrap();

        assert_eq!(serde_json::Value::from(out), json!([4, 3]));
    }

    #[test]
    fn document_rows_scalar_is_single_row() {
        let engine = JetroEngine::new();
        let document = engine.parse_value(json!(7));

        let out = collect_document_rows(&engine, &document, "$.rows().map(@ + 1)")
            .unwrap()
            .unwrap();

        assert_eq!(serde_json::Value::from(out), json!([8]));
    }
}
