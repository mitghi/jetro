use super::stream_exec::CompiledRowStream;
use super::stream_plan::{
    lower_root_rows_query, RowStreamDirection, RowStreamPlan, RowStreamSourceKind,
};
use super::stream_types::RowStreamRowResult;
use crate::data::value::Val;
use crate::{EvalError, Jetro, JetroEngine};
use std::sync::Arc;

pub(crate) fn collect_document_rows(
    engine: &JetroEngine,
    document: &Jetro,
    query: &str,
) -> Result<Option<Val>, EvalError> {
    let Some(plan) = document_rows_stream_plan(query)? else {
        return Ok(None);
    };

    let (out, _) = run_document_rows(engine, document, &plan)?;
    Ok(Some(out))
}

fn run_document_rows(
    engine: &JetroEngine,
    document: &Jetro,
    plan: &RowStreamPlan,
) -> Result<(Val, super::stream_types::RowStreamStats), EvalError> {
    let mut stream = CompiledRowStream::new(&plan);
    let mut vm = engine.lock_vm();
    let mut out = Vec::new();

    let root = document.root_val_with(engine.keys())?;
    match root {
        Val::Arr(rows) => match plan.direction {
            RowStreamDirection::Forward => {
                for row in rows.iter().cloned() {
                    if apply_document_row(&mut stream, &mut vm, row, &mut out)? {
                        break;
                    }
                }
            }
            RowStreamDirection::Reverse => {
                for row in rows.iter().rev().cloned() {
                    if apply_document_row(&mut stream, &mut vm, row, &mut out)? {
                        break;
                    }
                }
            }
        },
        root => {
            let mut rows = document_rows(root);
            if plan.direction == RowStreamDirection::Reverse {
                rows.reverse();
            }
            for row in rows {
                if apply_document_row(&mut stream, &mut vm, row, &mut out)? {
                    break;
                }
            }
        }
    }

    let stats = stream.stats().clone();
    Ok((Val::Arr(Arc::new(out)), stats))
}

fn apply_document_row(
    stream: &mut CompiledRowStream,
    vm: &mut crate::VM,
    row: Val,
    out: &mut Vec<Val>,
) -> Result<bool, EvalError> {
    if stream.is_exhausted() {
        return Ok(true);
    }
    match stream.apply_val_row(vm, row)? {
        RowStreamRowResult::Emit(value) => out.push(value),
        RowStreamRowResult::EmitBytes(bytes) => {
            return Err(EvalError(format!(
                "internal rows() stream error: byte output in document mode ({} bytes)",
                bytes.len()
            )));
        }
        RowStreamRowResult::Skip => {}
        RowStreamRowResult::Stop => return Ok(true),
    }
    Ok(false)
}

fn document_rows(root: Val) -> Vec<Val> {
    match root.into_vals() {
        Ok(rows) => rows,
        Err(root) => vec![root],
    }
}

#[cfg(test)]
fn collect_document_rows_with_stats(
    engine: &JetroEngine,
    document: &Jetro,
    query: &str,
) -> Result<Option<(Val, super::stream_types::RowStreamStats)>, EvalError> {
    let Some(plan) = document_rows_stream_plan(query)? else {
        return Ok(None);
    };

    let (out, stats) = run_document_rows(engine, document, &plan)?;
    Ok(Some((out, stats)))
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

    #[test]
    fn document_rows_take_stops_without_scanning_full_array() {
        let engine = JetroEngine::new();
        let document = engine.parse_value(json!([
            {"id": 1},
            {"id": 2},
            {"id": 3},
            {"id": 4}
        ]));

        let (out, stats) = collect_document_rows_with_stats(
            &engine,
            &document,
            "$.rows().take(2).map($.id)",
        )
        .unwrap()
        .unwrap();

        assert_eq!(serde_json::Value::from(out), json!([1, 2]));
        assert_eq!(stats.rows_scanned, 2);
        assert_eq!(stats.rows_emitted, 2);
    }
}
