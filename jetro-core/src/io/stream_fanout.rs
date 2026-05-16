use super::ndjson::{
    collect_row_stream_result, ndjson_writer_with_options, parse_row, row_eval_error,
    write_val_line_with_options, NdjsonOptions,
};
#[cfg(feature = "simd-json")]
use super::ndjson_byte::{eval_ndjson_byte_predicate_row, raw_json_path_view};
#[cfg(feature = "simd-json")]
use super::ndjson_direct::{
    direct_tape_predicate_for_expr, NdjsonDirectPredicate, NdjsonPhysicalPath,
};
use super::stream_exec::CompiledRowStream;
use super::stream_plan::{
    lower_root_rows_expr, RowStreamDirection, RowStreamPlan, RowStreamPlanError,
    RowStreamSourceKind, RowStreamStage,
};
use crate::compile::compiler::Compiler;
use crate::data::context::Env;
use crate::data::value::Val;
use crate::ir::physical::PhysicalPathStep;
use crate::parse::ast::{Arg, ArrayElem, BinOp, Expr, ObjField, Step};
use crate::util::{json_cmp_binop, JsonView};
use crate::{EvalError, JetroEngine, JetroEngineError};
use std::io::Write;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub(super) struct RowStreamFanoutPlan {
    pub source: RowStreamPlan,
    consumers: Vec<RowStreamFanoutConsumer>,
    body: Expr,
}

#[derive(Clone, Debug)]
struct RowStreamFanoutConsumer {
    binding: String,
    stream: RowStreamPlan,
    scalar: bool,
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
    if let Some(plan) = lower_object_rows_fanout_expr(expr, source_kind)? {
        return Ok(Some(plan));
    }
    if let Some(plan) = lower_array_rows_fanout_expr(expr, source_kind)? {
        return Ok(Some(plan));
    }

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

    Ok(Some(RowStreamFanoutPlan {
        source,
        consumers,
        body: body.clone(),
    }))
}

fn lower_object_rows_fanout_expr(
    expr: &Expr,
    source_kind: RowStreamSourceKind,
) -> Result<Option<RowStreamFanoutPlan>, RowStreamPlanError> {
    let Expr::Object(fields) = expr else {
        return Ok(None);
    };
    let mut consumers = Vec::new();
    let mut rewritten = Vec::with_capacity(fields.len());
    let mut source = None;
    let mut saw_stream = false;

    for (idx, field) in fields.iter().enumerate() {
        match field {
            ObjField::Kv {
                key,
                val,
                optional: false,
                cond: None,
            } => {
                if let Some(stream) = lower_root_rows_expr(val, source_kind)? {
                    saw_stream = true;
                    let base = source.get_or_insert_with(|| RowStreamPlan {
                        source: source_kind,
                        direction: stream.direction,
                        stages: Vec::new(),
                        demand: Default::default(),
                    });
                    if base.direction != stream.direction {
                        return Ok(None);
                    }
                    let binding = format!("__jetro_rows_fanout_{idx}");
                    consumers.push(RowStreamFanoutConsumer {
                        binding: binding.clone(),
                        scalar: stream.demand.retained_limit == Some(1),
                        stream,
                    });
                    rewritten.push(ObjField::Kv {
                        key: key.clone(),
                        val: Expr::Ident(binding),
                        optional: false,
                        cond: None,
                    });
                } else {
                    rewritten.push(field.clone());
                }
            }
            _ => rewritten.push(field.clone()),
        }
    }

    if !saw_stream || consumers.len() < 2 {
        return Ok(None);
    }
    Ok(Some(RowStreamFanoutPlan {
        source: source.expect("fanout source"),
        consumers,
        body: Expr::Object(rewritten),
    }))
}

fn lower_array_rows_fanout_expr(
    expr: &Expr,
    source_kind: RowStreamSourceKind,
) -> Result<Option<RowStreamFanoutPlan>, RowStreamPlanError> {
    let Expr::Array(items) = expr else {
        return Ok(None);
    };
    let mut consumers = Vec::new();
    let mut rewritten = Vec::with_capacity(items.len());
    let mut source = None;
    let mut saw_stream = false;

    for (idx, item) in items.iter().enumerate() {
        match item {
            ArrayElem::Expr(val) => {
                if let Some(stream) = lower_root_rows_expr(val, source_kind)? {
                    saw_stream = true;
                    let base = source.get_or_insert_with(|| RowStreamPlan {
                        source: source_kind,
                        direction: stream.direction,
                        stages: Vec::new(),
                        demand: Default::default(),
                    });
                    if base.direction != stream.direction {
                        return Ok(None);
                    }
                    let binding = format!("__jetro_rows_fanout_{idx}");
                    consumers.push(RowStreamFanoutConsumer {
                        binding: binding.clone(),
                        scalar: stream.demand.retained_limit == Some(1),
                        stream,
                    });
                    rewritten.push(ArrayElem::Expr(Expr::Ident(binding)));
                } else {
                    rewritten.push(item.clone());
                }
            }
            _ => rewritten.push(item.clone()),
        }
    }

    if !saw_stream || consumers.len() < 2 {
        return Ok(None);
    }
    Ok(Some(RowStreamFanoutPlan {
        source: source.expect("fanout source"),
        consumers,
        body: Expr::Array(rewritten),
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
        Expr::Ident(name) => Expr::Chain(Box::new(Expr::Root), vec![Step::Field(name)]),
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
            #[cfg(feature = "simd-json")]
            direct_first_predicate: direct_first_match_predicate(&consumer.stream),
            #[cfg(feature = "simd-json")]
            direct_cmp: direct_first_match_cmp(&consumer.stream),
            done: false,
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

    let mut env = Env::new(Val::Null);
    for consumer in consumers {
        let value = if consumer.scalar {
            consumer.values.into_iter().next().unwrap_or(Val::Null)
        } else {
            Val::Arr(Arc::new(consumer.values))
        };
        env = env.with_var(&consumer.binding, value);
    }
    let body = Compiler::compile(&plan.body, "<ndjson-rows-fanout-body>");
    engine
        .lock_vm()
        .exec_in_env(&body, &env)
        .map_err(JetroEngineError::Eval)
}

struct RunningConsumer {
    binding: String,
    scalar: bool,
    #[cfg(feature = "simd-json")]
    direct_first_predicate: Option<NdjsonDirectPredicate>,
    #[cfg(feature = "simd-json")]
    direct_cmp: Option<DirectCmp>,
    done: bool,
    stream: CompiledRowStream,
    values: Vec<Val>,
}

#[cfg(feature = "simd-json")]
#[derive(Clone)]
struct DirectCmp {
    steps: NdjsonPhysicalPath,
    op: BinOp,
    lit: Val,
}

fn all_consumers_done(consumers: &[RunningConsumer]) -> bool {
    consumers
        .iter()
        .all(|consumer| consumer.done || consumer.stream.is_exhausted())
}

#[cfg(feature = "simd-json")]
fn direct_first_match_predicate(plan: &RowStreamPlan) -> Option<NdjsonDirectPredicate> {
    let [RowStreamStage::Filter(expr), rest @ ..] = plan.stages.as_slice() else {
        return None;
    };
    if rest
        .iter()
        .all(|stage| matches!(stage, RowStreamStage::Take(1)))
    {
        direct_tape_predicate_for_expr(expr)
    } else {
        None
    }
}

#[cfg(feature = "simd-json")]
fn direct_first_match_cmp(plan: &RowStreamPlan) -> Option<DirectCmp> {
    let predicate = direct_first_match_predicate(plan)?;
    direct_cmp_from_predicate(&predicate)
}

#[cfg(feature = "simd-json")]
fn direct_cmp_from_predicate(predicate: &NdjsonDirectPredicate) -> Option<DirectCmp> {
    match predicate {
        NdjsonDirectPredicate::Binary { lhs, op, rhs } if *op == BinOp::Eq => {
            match (lhs.as_ref(), rhs.as_ref()) {
                (NdjsonDirectPredicate::Path(steps), NdjsonDirectPredicate::Literal(lit)) => {
                    Some(DirectCmp {
                        steps: steps.clone(),
                        op: *op,
                        lit: lit.clone(),
                    })
                }
                (NdjsonDirectPredicate::Literal(lit), NdjsonDirectPredicate::Path(steps)) => {
                    Some(DirectCmp {
                        steps: steps.clone(),
                        op: *op,
                        lit: lit.clone(),
                    })
                }
                _ => None,
            }
        }
        _ => None,
    }
}

fn apply_fanout_row(
    engine: &JetroEngine,
    line_no: u64,
    row: Vec<u8>,
    consumers: &mut [RunningConsumer],
) -> Result<(), JetroEngineError> {
    let mut matched_value = None;
    #[cfg(feature = "simd-json")]
    let shared_view = shared_cmp_path(consumers).and_then(|steps| {
        raw_json_path_view(&row, steps).map(|view| (steps.to_vec(), view))
    });
    for consumer in consumers {
        if consumer.done || consumer.stream.is_exhausted() {
            continue;
        }
        #[cfg(feature = "simd-json")]
        if let (Some((steps, view)), Some(cmp)) = (shared_view.as_ref(), consumer.direct_cmp.as_ref())
        {
            if same_path(steps, &cmp.steps) {
                if !json_cmp_binop(*view, cmp.op, JsonView::from_val(&cmp.lit)) {
                    continue;
                }
                if matched_value.is_none() {
                    matched_value = Some(row_to_val(engine, line_no, row.clone())?);
                }
                consumer
                    .values
                    .push(matched_value.as_ref().expect("matched value").clone());
                consumer.done = true;
                continue;
            }
        }
        #[cfg(feature = "simd-json")]
        if let Some(predicate) = consumer.direct_first_predicate.as_ref() {
            match eval_ndjson_byte_predicate_row(&row, predicate)? {
                Some(false) => continue,
                Some(true) => {
                    if matched_value.is_none() {
                        matched_value = Some(row_to_val(engine, line_no, row.clone())?);
                    }
                    consumer
                        .values
                        .push(matched_value.as_ref().expect("matched value").clone());
                    consumer.done = true;
                    continue;
                }
                None => {}
            }
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

#[cfg(feature = "simd-json")]
fn shared_cmp_path(consumers: &[RunningConsumer]) -> Option<&[PhysicalPathStep]> {
    let mut shared = None;
    let mut count = 0usize;
    for consumer in consumers {
        if consumer.done || consumer.stream.is_exhausted() {
            continue;
        }
        let Some(cmp) = consumer.direct_cmp.as_ref() else {
            continue;
        };
        count += 1;
        match shared {
            None => shared = Some(cmp.steps.as_slice()),
            Some(path) if same_path(path, &cmp.steps) => {}
            Some(_) => return None,
        }
    }
    (count > 1).then_some(shared?) 
}

#[cfg(feature = "simd-json")]
fn same_path(a: &[PhysicalPathStep], b: &[PhysicalPathStep]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(a, b)| match (a, b) {
            (PhysicalPathStep::Field(a), PhysicalPathStep::Field(b)) => a == b,
            (PhysicalPathStep::Index(a), PhysicalPathStep::Index(b)) => a == b,
            _ => false,
        })
}

fn row_to_val(engine: &JetroEngine, line_no: u64, row: Vec<u8>) -> Result<Val, JetroEngineError> {
    let document = parse_row(engine, line_no, row)?;
    document
        .root_val_with(engine.keys())
        .map_err(|err| row_eval_error(line_no, err))
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
        #[cfg(feature = "simd-json")]
        assert!(direct_first_match_cmp(&plan.consumers[0].stream).is_some());
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

    #[test]
    fn executes_fanout_with_shaped_body() {
        let path = temp_ndjson(
            "body",
            &[
                r#"{"name":"Ada","version":1}"#,
                r#"{"name":"Bob","version":1}"#,
                r#"{"name":"Ada","version":2}"#,
            ],
        );
        let query = r#"let stream = $.rows().reverse(), user_a = stream.find(name == "Ada").first(), user_b = stream.find(name == "Bob").first() in {latest: user_a.version, pair: [user_a.name, user_b.name]}"#;
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
        assert_eq!(got.trim(), r#"{"latest":2,"pair":["Ada","Bob"]}"#);
    }

    #[test]
    fn executes_object_rows_subqueries_as_fanout() {
        let path = temp_ndjson(
            "object",
            &[
                r#"{"name":"Ada","version":1}"#,
                r#"{"name":"Bob","version":1}"#,
                r#"{"name":"Ada","version":2}"#,
            ],
        );
        let query = r#"{user_a: $.rows().reverse().find($.name == "Ada").first(), user_b: $.rows().reverse().find($.name == "Bob").first()}"#;
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

    #[test]
    fn executes_array_rows_subqueries_as_fanout() {
        let path = temp_ndjson(
            "array",
            &[
                r#"{"name":"Ada","version":1}"#,
                r#"{"name":"Bob","version":1}"#,
                r#"{"name":"Ada","version":2}"#,
            ],
        );
        let query = r#"[$.rows().reverse().find($.name == "Ada").first(), $.rows().reverse().find($.name == "Bob").first()]"#;
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
            r#"[{"name":"Ada","version":2},{"name":"Bob","version":1}]"#
        );
    }
}
