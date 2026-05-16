use super::ndjson::{
    collect_row_stream_result, ndjson_writer_with_options, write_val_line_with_options,
    NdjsonOptions,
};
use super::stream_exec::CompiledRowStream;
use super::stream_plan::{
    lower_root_rows_expr, RowStreamDirection, RowStreamPlan, RowStreamPlanError, RowStreamSourceKind,
};
use crate::data::value::Val;
use crate::parse::ast::{Arg, Expr, ObjField, Step};
use crate::{EvalError, JetroEngine, JetroEngineError};
use indexmap::IndexMap;
use std::io::Write;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub(super) struct RowStreamFanoutPlan {
    pub source: RowStreamPlan,
    consumers: Vec<RowStreamFanoutConsumer>,
    outputs: Vec<RowStreamFanoutOutput>,
}

#[derive(Clone, Debug)]
struct RowStreamFanoutConsumer {
    binding: String,
    stream: RowStreamPlan,
    scalar: bool,
}

#[derive(Clone, Debug)]
struct RowStreamFanoutOutput {
    key: String,
    binding: String,
}

pub(super) fn lower_rows_fanout_query(
    query: &str,
    source: RowStreamSourceKind,
) -> Result<Option<RowStreamFanoutPlan>, JetroEngineError> {
    if !query.contains("$.rows") {
        return Ok(None);
    }
    let expr = crate::parse::parser::parse(query)
        .map_err(|err| JetroEngineError::Eval(EvalError(err.to_string())))?;
    lower_rows_fanout_expr(&expr, source)
        .map_err(|err| JetroEngineError::Eval(EvalError(err.to_string())))
}

fn lower_rows_fanout_expr(
    expr: &Expr,
    source_kind: RowStreamSourceKind,
) -> Result<Option<RowStreamFanoutPlan>, RowStreamPlanError> {
    let mut bindings = Vec::new();
    let body = collect_let_chain(expr, &mut bindings);
    if bindings.len() < 2 {
        return Ok(None);
    }

    let mut stream_binding = None;
    for (idx, (name, init)) in bindings.iter().enumerate() {
        if let Some(stream) = lower_root_rows_expr(init, source_kind)? {
            stream_binding = Some((idx, name.clone(), stream));
            break;
        }
    }
    let Some((stream_idx, stream_name, source)) = stream_binding else {
        return Ok(None);
    };
    if stream_idx != 0 {
        return Ok(None);
    }

    let mut consumers = Vec::new();
    for (name, init) in bindings.iter().skip(1) {
        let Some(stream) = lower_consumer_stream(init, &stream_name, &source)? else {
            return Ok(None);
        };
        let scalar = stream.demand.retained_limit == Some(1);
        consumers.push(RowStreamFanoutConsumer {
            binding: name.clone(),
            stream,
            scalar,
        });
    }
    if consumers.is_empty() {
        return Ok(None);
    }

    let outputs = lower_output_object(body, &consumers)?;
    if outputs.is_empty() {
        return Ok(None);
    }

    Ok(Some(RowStreamFanoutPlan {
        source,
        consumers,
        outputs,
    }))
}

fn collect_let_chain<'a>(expr: &'a Expr, bindings: &mut Vec<(String, &'a Expr)>) -> &'a Expr {
    let mut cur = expr;
    while let Expr::Let { name, init, body } = cur {
        bindings.push((name.clone(), init.as_ref()));
        cur = body.as_ref();
    }
    cur
}

fn lower_consumer_stream(
    expr: &Expr,
    stream_name: &str,
    source: &RowStreamPlan,
) -> Result<Option<RowStreamPlan>, RowStreamPlanError> {
    let Expr::Chain(base, steps) = expr else {
        return Ok(None);
    };
    if !matches!(base.as_ref(), Expr::Ident(name) if name == stream_name) {
        return Ok(None);
    }

    let mut fake_steps = Vec::with_capacity(steps.len() + 1);
    fake_steps.push(Step::Method("rows".to_string(), Vec::new()));
    fake_steps.extend(steps.iter().cloned().map(normalize_step));
    let fake = Expr::Chain(Box::new(Expr::Root), fake_steps);
    let Some(suffix) = lower_root_rows_expr(&fake, source.source)? else {
        return Ok(None);
    };

    let mut merged = source.clone();
    if suffix.direction == RowStreamDirection::Reverse {
        merged.direction = match merged.direction {
            RowStreamDirection::Forward => RowStreamDirection::Reverse,
            RowStreamDirection::Reverse => RowStreamDirection::Forward,
        };
    }
    merged.stages.extend(suffix.stages);
    merged.refresh_demand();
    Ok(Some(merged))
}

fn normalize_step(step: Step) -> Step {
    match step {
        Step::Method(name, args) => Step::Method(
            name,
            args.into_iter()
                .map(|arg| match arg {
                    Arg::Pos(expr) => Arg::Pos(normalize_bare_ident_predicate(expr)),
                    Arg::Named(name, expr) => Arg::Named(name, normalize_bare_ident_predicate(expr)),
                })
                .collect(),
        ),
        other => other,
    }
}

fn normalize_bare_ident_predicate(expr: Expr) -> Expr {
    match expr {
        Expr::Ident(name) => Expr::Chain(Box::new(Expr::Current), vec![Step::Field(name)]),
        Expr::BinOp(left, op, right) => Expr::BinOp(
            Box::new(normalize_bare_ident_predicate(*left)),
            op,
            Box::new(normalize_bare_ident_predicate(*right)),
        ),
        Expr::Not(inner) => Expr::Not(Box::new(normalize_bare_ident_predicate(*inner))),
        Expr::UnaryNeg(inner) => Expr::UnaryNeg(Box::new(normalize_bare_ident_predicate(*inner))),
        Expr::Coalesce(left, right) => Expr::Coalesce(
            Box::new(normalize_bare_ident_predicate(*left)),
            Box::new(normalize_bare_ident_predicate(*right)),
        ),
        other => other,
    }
}

fn lower_output_object(
    body: &Expr,
    consumers: &[RowStreamFanoutConsumer],
) -> Result<Vec<RowStreamFanoutOutput>, RowStreamPlanError> {
    let Expr::Object(fields) = body else {
        return Err(RowStreamPlanError::new(
            "$.rows() fanout currently requires an object result",
        ));
    };
    let mut out = Vec::new();
    for field in fields {
        match field {
            ObjField::Short(name) if consumers.iter().any(|consumer| consumer.binding == *name) => {
                out.push(RowStreamFanoutOutput {
                    key: name.clone(),
                    binding: name.clone(),
                });
            }
            ObjField::Kv {
                key,
                val: Expr::Ident(name),
                optional: false,
                cond: None,
            } if consumers.iter().any(|consumer| consumer.binding == *name) => {
                out.push(RowStreamFanoutOutput {
                    key: key.clone(),
                    binding: name.clone(),
                });
            }
            _ => {
                return Err(RowStreamPlanError::new(
                    "$.rows() fanout object fields must reference stream consumer bindings",
                ));
            }
        }
    }
    Ok(out)
}

pub(super) fn drive_ndjson_rows_fanout_file<P, W>(
    engine: &JetroEngine,
    path: P,
    plan: &RowStreamFanoutPlan,
    options: NdjsonOptions,
    writer: W,
) -> Result<usize, JetroEngineError>
where
    P: AsRef<std::path::Path>,
    W: Write,
{
    let value = collect_ndjson_rows_fanout_file(engine, path, plan, options)?;
    let mut writer = ndjson_writer_with_options(writer, options);
    let emitted = write_val_line_with_options(&mut writer, &value, options)? as usize;
    writer.flush()?;
    Ok(emitted)
}

fn collect_ndjson_rows_fanout_file<P>(
    engine: &JetroEngine,
    path: P,
    plan: &RowStreamFanoutPlan,
    options: NdjsonOptions,
) -> Result<Val, JetroEngineError>
where
    P: AsRef<std::path::Path>,
{
    let mut consumers: Vec<_> = plan
        .consumers
        .iter()
        .map(|consumer| RunningConsumer {
            binding: consumer.binding.clone(),
            scalar: consumer.scalar,
            stream: CompiledRowStream::new(&consumer.stream),
            values: Vec::new(),
        })
        .collect();

    if plan.source.direction == RowStreamDirection::Forward {
        let file = std::fs::File::open(path)?;
        let mut driver = super::ndjson::NdjsonPerRowDriver::new(std::io::BufReader::with_capacity(
            options.reader_buffer_capacity,
            file,
        ))
        .with_options(options);
        let mut buf = Vec::with_capacity(options.initial_buffer_capacity);
        while !all_consumers_done(&consumers) {
            let Some((line_no, row)) = driver.read_next_owned(&mut buf)? else {
                break;
            };
            apply_fanout_row(engine, line_no, row, &mut consumers)?;
        }
    } else {
        let mut driver = super::ndjson_rev::NdjsonReverseFileDriver::with_options(path, options)?;
        while !all_consumers_done(&consumers) {
            let Some((line_no, row)) = driver.next_line_with_reverse_no()? else {
                break;
            };
            apply_fanout_row(engine, line_no, row, &mut consumers)?;
        }
    }

    let mut by_name = IndexMap::new();
    for consumer in consumers {
        let value = if consumer.scalar {
            consumer.values.into_iter().next().unwrap_or(Val::Null)
        } else {
            Val::Arr(Arc::new(consumer.values))
        };
        by_name.insert(consumer.binding, value);
    }

    let mut object = IndexMap::with_capacity(plan.outputs.len());
    for output in &plan.outputs {
        object.insert(
            Val::key(&output.key),
            by_name.get(&output.binding).cloned().unwrap_or(Val::Null),
        );
    }
    Ok(Val::obj(object))
}

struct RunningConsumer {
    binding: String,
    scalar: bool,
    stream: CompiledRowStream,
    values: Vec<Val>,
}

fn all_consumers_done(consumers: &[RunningConsumer]) -> bool {
    consumers.iter().all(|consumer| consumer.stream.is_exhausted())
}

fn apply_fanout_row(
    engine: &JetroEngine,
    line_no: u64,
    row: Vec<u8>,
    consumers: &mut [RunningConsumer],
) -> Result<(), JetroEngineError> {
    for consumer in consumers {
        if consumer.stream.is_exhausted() {
            continue;
        }
        let mut values = Vec::new();
        let stop = collect_row_stream_result(
            engine,
            line_no,
            consumer
                .stream
                .apply_owned_row(engine, line_no, row.clone())?,
            &mut values,
        )?;
        consumer.values.extend(values);
        if stop {
            // `CompiledRowStream` marks exhaustion internally for bounded sinks.
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_ndjson(name: &str, rows: &[&str]) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "jetro-stream-fanout-{}-{}.ndjson",
            name,
            std::process::id()
        ));
        let mut file = std::fs::File::create(&path).unwrap();
        for row in rows {
            writeln!(file, "{row}").unwrap();
        }
        path
    }

    #[test]
    fn lowers_let_bound_rows_fanout() {
        let query = r#"let stream = $.rows().reverse(), user_a = stream.find(name == "Ada").first(), user_b = stream.find(name == "Bob").first() in {user_a, user_b}"#;
        let plan = lower_rows_fanout_query(query, RowStreamSourceKind::NdjsonRows)
            .unwrap()
            .expect("fanout plan");
        assert_eq!(plan.source.direction, RowStreamDirection::Reverse);
        assert_eq!(plan.consumers.len(), 2);
        assert_eq!(plan.outputs.len(), 2);
    }

    #[test]
    fn executes_reverse_first_match_fanout_in_one_result() {
        let path = temp_ndjson(
            "reverse",
            &[
                r#"{"name":"Ada","version":1}"#,
                r#"{"name":"Bob","version":1}"#,
                r#"{"name":"Ada","version":2}"#,
            ],
        );
        let query = r#"let stream = $.rows().reverse(), user_a = stream.find(name == "Ada").first(), user_b = stream.find(name == "Bob").first() in {user_a, user_b}"#;
        let engine = JetroEngine::new();
        let mut out = Vec::new();
        super::super::ndjson::run_ndjson_file_with_options(
            &engine,
            &path,
            query,
            &mut out,
            NdjsonOptions::default(),
        )
        .unwrap();
        std::fs::remove_file(path).ok();
        let got = String::from_utf8(out).unwrap();
        assert_eq!(
            got.trim(),
            r#"{"user_a":{"name":"Ada","version":2},"user_b":{"name":"Bob","version":1}}"#
        );
    }
}
