use super::stream_exec::{finish_collected_row_stream, CompiledRowStream};
use super::stream_plan::{lower_root_rows_query, RowStreamPlan, RowStreamSourceKind};
use super::stream_source::{DocumentRowSource, ValueRowSource};
use super::stream_types::RowStreamRowResult;
use crate::data::value::Val;
use crate::{EvalError, Jetro, JetroEngine};

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
    let mut source = DocumentRowSource::new(root, plan.direction);
    while let Some(row) = source.next_row() {
        if apply_document_row(&mut stream, &mut vm, row, &mut out)? {
            break;
        }
    }

    Ok(finish_collected_row_stream(plan, &stream, out))
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

        let (out, stats) =
            collect_document_rows_with_stats(&engine, &document, "$.rows().take(2).map($.id)")
                .unwrap()
                .unwrap();

        assert_eq!(serde_json::Value::from(out), json!([1, 2]));
        assert_eq!(stats.source, RowStreamSourceKind::DocumentRows);
        assert_eq!(stats.rows_scanned, 2);
        assert_eq!(stats.rows_emitted, 2);
    }

    #[test]
    fn document_rows_scalar_sinks_return_finished_value() {
        let engine = JetroEngine::new();
        let document = engine.parse_value(json!([
            {"active": true, "price": 10},
            {"active": false, "price": 30},
            {"active": true, "price": 5}
        ]));

        let count = collect_document_rows(
            &engine,
            &document,
            "$.rows().filter($.active == true).count()",
        )
        .unwrap()
        .unwrap();
        assert_eq!(serde_json::Value::from(count), json!(2));

        let sum = collect_document_rows(
            &engine,
            &document,
            "$.rows().filter($.active == true).map($.price).sum()",
        )
        .unwrap()
        .unwrap();
        assert_eq!(serde_json::Value::from(sum), json!(15));

        let any = collect_document_rows(&engine, &document, "$.rows().any($.price > 20)")
            .unwrap()
            .unwrap();
        assert_eq!(serde_json::Value::from(any), json!(true));
    }

    #[test]
    fn document_rows_last_returns_final_matching_row() {
        let engine = JetroEngine::new();
        let document = engine.parse_value(json!([
            {"active": true, "id": 1},
            {"active": false, "id": 2},
            {"active": true, "id": 3}
        ]));

        let out = collect_document_rows(
            &engine,
            &document,
            "$.rows().filter($.active == true).last()",
        )
        .unwrap()
        .unwrap();

        assert_eq!(
            serde_json::Value::from(out),
            json!({"active": true, "id": 3})
        );
    }

    #[test]
    fn document_rows_scalar_sink_empty_edges_are_finished() {
        let engine = JetroEngine::new();
        let document = engine.parse_value(json!([]));

        let any = collect_document_rows(&engine, &document, "$.rows().any($.active)")
            .unwrap()
            .unwrap();
        assert_eq!(serde_json::Value::from(any), json!(false));

        let all = collect_document_rows(&engine, &document, "$.rows().all($.active)")
            .unwrap()
            .unwrap();
        assert_eq!(serde_json::Value::from(all), json!(true));

        let last = collect_document_rows(&engine, &document, "$.rows().last()")
            .unwrap()
            .unwrap();
        assert_eq!(serde_json::Value::from(last), json!(null));
    }
}
