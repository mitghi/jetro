use super::mapped_bytes::{split_line_aligned_ranges, MappedBytes};
use super::ndjson::{
    collect_row_stream_result, ndjson_writer_with_options, parse_row, row_eval_error,
    write_val_line_with_options, NdjsonOptions, NdjsonParallelism,
};
use super::ndjson_byte::{
    eval_ndjson_byte_predicate_row, eval_ndjson_byte_predicates_all, raw_json_path_view,
};
use super::ndjson_direct::{
    direct_cmp_literal_predicate, direct_tape_predicate_for_expr, direct_tape_predicates_for_exprs,
    direct_tape_root_path_for_expr, physical_paths_equal, NdjsonDirectPredicate,
    NdjsonPhysicalPath,
};
use super::ndjson_scan::for_each_framed_payload_in_range;
use super::stream_exec::CompiledRowStream;
use super::stream_numeric::NumericAccumulator;
#[cfg(test)]
use super::stream_plan::parse_rows_stream_candidate_query;
use super::stream_plan::{
    lower_root_rows_expr, RowStreamDirection, RowStreamPlan, RowStreamPlanError,
    RowStreamSourceKind, RowStreamStage,
};
use super::stream_types::RowStreamStats;
use crate::builtins::registry::predicate_sink_result_demand;
use crate::builtins::BuiltinPredicateSink;
use crate::compile::compiler::Compiler;
use crate::data::context::Env;
use crate::data::value::Val;
use crate::ir::physical::PhysicalPathStep;
use crate::parse::ast::{Arg, ArrayElem, BinOp, Expr, ObjField, Step};
use crate::plan::demand::SinkResultDemand;
use crate::util::{json_cmp_binop, JsonView};
use crate::{JetroEngine, JetroEngineError};
use rayon::prelude::*;
use std::io::Write;
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub(crate) struct RowStreamFanoutPlan {
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

#[cfg(test)]
pub(super) fn lower_rows_fanout_query(
    query: &str,
    source: RowStreamSourceKind,
) -> Result<Option<RowStreamFanoutPlan>, JetroEngineError> {
    let candidate = parse_rows_stream_candidate_query(query)
        .map_err(|err| JetroEngineError::Eval(crate::EvalError(err.to_string())))?;
    let Some(expr) = candidate else {
        return Ok(None);
    };
    lower_rows_fanout_expr(&expr, source)
        .map_err(|err| JetroEngineError::Eval(crate::EvalError(err.to_string())))
}

pub(super) fn lower_rows_fanout_expr(
    expr: &Expr,
    source_kind: RowStreamSourceKind,
) -> Result<Option<RowStreamFanoutPlan>, RowStreamPlanError> {
    if let Some(plan) = lower_shape_rows_fanout_expr(expr, source_kind)? {
        return Ok(Some(plan));
    }

    let mut bindings = Vec::new();
    let body = collect_let_chain(expr, &mut bindings);
    if bindings.is_empty() {
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
        let scalar = stream.returns_scalar_value();
        consumers.push(RowStreamFanoutConsumer {
            binding: name.clone(),
            stream,
            scalar,
        });
    }

    let mut body_builder = LetFanoutBodyBuilder {
        stream_name: &stream_name,
        source: &source,
        consumers,
        next_id: 0,
    };
    let body = rewrite_let_stream_fanout_body(body, &mut body_builder)?;
    let consumers = body_builder.consumers;

    if consumers.is_empty() {
        return Ok(None);
    }

    Ok(Some(RowStreamFanoutPlan {
        source,
        consumers,
        body,
    }))
}

struct LetFanoutBodyBuilder<'a> {
    stream_name: &'a str,
    source: &'a RowStreamPlan,
    consumers: Vec<RowStreamFanoutConsumer>,
    next_id: usize,
}

fn rewrite_let_stream_fanout_body(
    expr: &Expr,
    builder: &mut LetFanoutBodyBuilder<'_>,
) -> Result<Expr, RowStreamPlanError> {
    if let Some(stream) = lower_consumer_stream_lenient(expr, builder.stream_name, builder.source)?
    {
        let binding = format!("__jetro_rows_fanout_body_{}", builder.next_id);
        builder.next_id += 1;
        builder.consumers.push(RowStreamFanoutConsumer {
            binding: binding.clone(),
            scalar: stream.returns_scalar_value(),
            stream,
        });
        return Ok(Expr::Ident(binding));
    }

    match expr {
        Expr::Object(fields) => {
            let mut out = Vec::with_capacity(fields.len());
            for field in fields {
                out.push(match field {
                    ObjField::Kv {
                        key,
                        val,
                        optional,
                        cond,
                    } => ObjField::Kv {
                        key: key.clone(),
                        val: rewrite_let_stream_fanout_body(val, builder)?,
                        optional: *optional,
                        cond: cond
                            .as_ref()
                            .map(|cond| rewrite_let_stream_fanout_body(cond, builder))
                            .transpose()?,
                    },
                    ObjField::Dynamic { key, val } => ObjField::Dynamic {
                        key: rewrite_let_stream_fanout_body(key, builder)?,
                        val: rewrite_let_stream_fanout_body(val, builder)?,
                    },
                    ObjField::Spread(value) => {
                        ObjField::Spread(rewrite_let_stream_fanout_body(value, builder)?)
                    }
                    ObjField::SpreadDeep(value) => {
                        ObjField::SpreadDeep(rewrite_let_stream_fanout_body(value, builder)?)
                    }
                    ObjField::Short(name) => ObjField::Short(name.clone()),
                });
            }
            Ok(Expr::Object(out))
        }
        Expr::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                out.push(match item {
                    ArrayElem::Expr(value) => {
                        ArrayElem::Expr(rewrite_let_stream_fanout_body(value, builder)?)
                    }
                    ArrayElem::Spread(value) => {
                        ArrayElem::Spread(rewrite_let_stream_fanout_body(value, builder)?)
                    }
                });
            }
            Ok(Expr::Array(out))
        }
        _ => Ok(expr.clone()),
    }
}

fn lower_shape_rows_fanout_expr(
    expr: &Expr,
    source_kind: RowStreamSourceKind,
) -> Result<Option<RowStreamFanoutPlan>, RowStreamPlanError> {
    let mut builder = FanoutBuilder::new(source_kind);
    let mut next_id = 0usize;
    let Some(body) = rewrite_rows_fanout_body(expr, &mut builder, &mut next_id)? else {
        return Ok(None);
    };
    Ok(builder.finish(body))
}

fn rewrite_rows_fanout_body(
    expr: &Expr,
    builder: &mut FanoutBuilder,
    next_id: &mut usize,
) -> Result<Option<Expr>, RowStreamPlanError> {
    if let Some(stream) = lower_fanout_rows_expr(expr, builder.source_kind)? {
        let binding = format!("__jetro_rows_fanout_{}", *next_id);
        *next_id += 1;
        if !builder.push_stream(binding.clone(), stream) {
            return Ok(None);
        }
        return Ok(Some(Expr::Ident(binding)));
    }

    match expr {
        Expr::Object(fields) => {
            let mut out = Vec::with_capacity(fields.len());
            for field in fields {
                out.push(match field {
                    ObjField::Kv {
                        key,
                        val,
                        optional,
                        cond,
                    } => ObjField::Kv {
                        key: key.clone(),
                        val: rewrite_rows_fanout_body(val, builder, next_id)?
                            .unwrap_or_else(|| val.clone()),
                        optional: *optional,
                        cond: match cond {
                            Some(cond) => Some(
                                rewrite_rows_fanout_body(cond, builder, next_id)?
                                    .unwrap_or_else(|| cond.clone()),
                            ),
                            None => None,
                        },
                    },
                    ObjField::Dynamic { key, val } => ObjField::Dynamic {
                        key: rewrite_rows_fanout_body(key, builder, next_id)?
                            .unwrap_or_else(|| key.clone()),
                        val: rewrite_rows_fanout_body(val, builder, next_id)?
                            .unwrap_or_else(|| val.clone()),
                    },
                    ObjField::Spread(value) => ObjField::Spread(
                        rewrite_rows_fanout_body(value, builder, next_id)?
                            .unwrap_or_else(|| value.clone()),
                    ),
                    ObjField::SpreadDeep(value) => ObjField::SpreadDeep(
                        rewrite_rows_fanout_body(value, builder, next_id)?
                            .unwrap_or_else(|| value.clone()),
                    ),
                    ObjField::Short(name) => ObjField::Short(name.clone()),
                });
            }
            Ok(Some(Expr::Object(out)))
        }
        Expr::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                out.push(match item {
                    ArrayElem::Expr(value) => ArrayElem::Expr(
                        rewrite_rows_fanout_body(value, builder, next_id)?
                            .unwrap_or_else(|| value.clone()),
                    ),
                    ArrayElem::Spread(value) => ArrayElem::Spread(
                        rewrite_rows_fanout_body(value, builder, next_id)?
                            .unwrap_or_else(|| value.clone()),
                    ),
                });
            }
            Ok(Some(Expr::Array(out)))
        }
        _ => Ok(Some(expr.clone())),
    }
}

fn lower_fanout_rows_expr(
    expr: &Expr,
    source_kind: RowStreamSourceKind,
) -> Result<Option<RowStreamPlan>, RowStreamPlanError> {
    let normalized = normalize_rows_stream_expr(expr);
    match lower_root_rows_expr(&normalized, source_kind) {
        Ok(plan) => Ok(plan),
        Err(_) => Ok(None),
    }
}

fn normalize_rows_stream_expr(expr: &Expr) -> Expr {
    match expr {
        Expr::Chain(base, steps) => Expr::Chain(
            base.clone(),
            steps.iter().cloned().map(normalize_step).collect(),
        ),
        _ => expr.clone(),
    }
}

struct FanoutBuilder {
    source_kind: RowStreamSourceKind,
    source: Option<RowStreamPlan>,
    consumers: Vec<RowStreamFanoutConsumer>,
    abandoned: bool,
}

impl FanoutBuilder {
    fn new(source_kind: RowStreamSourceKind) -> Self {
        Self {
            source_kind,
            source: None,
            consumers: Vec::new(),
            abandoned: false,
        }
    }

    fn push_stream(&mut self, binding: String, stream: RowStreamPlan) -> bool {
        let base = self.source.get_or_insert_with(|| RowStreamPlan {
            source: self.source_kind,
            direction: stream.direction,
            stages: Vec::new(),
            demand: Default::default(),
        });
        if base.direction != stream.direction {
            self.abandoned = true;
            return false;
        }
        self.consumers.push(RowStreamFanoutConsumer {
            binding,
            scalar: stream.returns_scalar_value(),
            stream,
        });
        true
    }

    fn finish(self, body: Expr) -> Option<RowStreamFanoutPlan> {
        if self.abandoned || self.consumers.len() < 2 {
            return None;
        }
        Some(RowStreamFanoutPlan {
            source: self.source.expect("fanout source"),
            consumers: self.consumers,
            body,
        })
    }
}

fn collect_let_chain<'a>(expr: &'a Expr, bindings: &mut Vec<(String, &'a Expr)>) -> &'a Expr {
    let mut cur = expr;
    while let Expr::Let { name, init, body } = cur {
        bindings.push((name.clone(), init.as_ref()));
        cur = body.as_ref();
    }
    cur
}

fn lower_consumer_stream_lenient(
    expr: &Expr,
    stream_name: &str,
    source: &RowStreamPlan,
) -> Result<Option<RowStreamPlan>, RowStreamPlanError> {
    match lower_consumer_stream(expr, stream_name, source) {
        Ok(plan) => Ok(plan),
        Err(_) => Ok(None),
    }
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
                    Arg::Named(name, expr) => {
                        Arg::Named(name, normalize_bare_ident_predicate(expr))
                    }
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
    let (value, _) = collect_ndjson_rows_fanout_file_with_stats(engine, path, plan, options)?;
    let mut writer = ndjson_writer_with_options(writer, options);
    let emitted = write_val_line_with_options(&mut writer, &value, options)? as usize;
    writer.flush()?;
    Ok(emitted)
}

pub(super) fn drive_ndjson_rows_fanout_file_with_stats<P, W>(
    engine: &JetroEngine,
    path: P,
    plan: &RowStreamFanoutPlan,
    options: NdjsonOptions,
    writer: W,
) -> Result<(usize, RowStreamStats), JetroEngineError>
where
    P: AsRef<std::path::Path>,
    W: Write,
{
    let (value, mut stats) =
        collect_ndjson_rows_fanout_file_with_stats(engine, path, plan, options)?;
    let mut writer = ndjson_writer_with_options(writer, options);
    let emitted = write_val_line_with_options(&mut writer, &value, options)? as usize;
    stats.rows_emitted = emitted;
    writer.flush()?;
    Ok((emitted, stats))
}

fn collect_ndjson_rows_fanout_file_with_stats<P>(
    engine: &JetroEngine,
    path: P,
    plan: &RowStreamFanoutPlan,
    options: NdjsonOptions,
) -> Result<(Val, RowStreamStats), JetroEngineError>
where
    P: AsRef<std::path::Path>,
{
    let mut stats = RowStreamStats {
        source: plan.source.source,
        direction: plan.source.direction,
        ..RowStreamStats::default()
    };
    if let Some(value) =
        collect_parallel_direct_reducer_fanout(engine, path.as_ref(), plan, options)?
    {
        return Ok((value, stats));
    }

    let mut consumers: Vec<_> = plan
        .consumers
        .iter()
        .map(|consumer| RunningConsumer {
            binding: consumer.binding.clone(),
            scalar: consumer.scalar,
            direct_first_predicate: direct_first_match_predicate(&consumer.stream),
            direct_cmp: direct_first_match_cmp(&consumer.stream),
            direct_count: direct_count_consumer(&consumer.stream),
            direct_numeric: direct_numeric_consumer(&consumer.stream),
            direct_predicate_sink: direct_predicate_sink_consumer(&consumer.stream),
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
            stats.rows_scanned += 1;
            apply_fanout_row(engine, line_no, row, &mut consumers)?;
        }
    } else {
        let mut driver = super::ndjson_rev::NdjsonReverseFileDriver::with_options(path, options)?;
        while !all_consumers_done(&consumers) {
            let Some((line_no, row)) = driver.next_line_with_reverse_no()? else {
                break;
            };
            stats.rows_scanned += 1;
            apply_fanout_row(engine, line_no, row, &mut consumers)?;
        }
    }

    let mut env = Env::new(Val::Null);
    for consumer in consumers {
        merge_fanout_consumer_stats(&mut stats, consumer.stream.stats());
        let value = if let Some(value) = consumer.finish_value().map_err(JetroEngineError::Eval)? {
            value
        } else if consumer.scalar {
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
        .map(|value| (value, stats))
        .map_err(JetroEngineError::Eval)
}

fn merge_fanout_consumer_stats(stats: &mut RowStreamStats, consumer: &RowStreamStats) {
    stats.rows_emitted += consumer.rows_emitted;
    stats.rows_filtered += consumer.rows_filtered;
    stats.duplicate_rows += consumer.duplicate_rows;
    stats.direct_filter_rows += consumer.direct_filter_rows;
    stats.fallback_filter_rows += consumer.fallback_filter_rows;
    stats.direct_key_rows += consumer.direct_key_rows;
    stats.fallback_key_rows += consumer.fallback_key_rows;
    stats.direct_project_rows += consumer.direct_project_rows;
    stats.fallback_project_rows += consumer.fallback_project_rows;
}

struct RunningConsumer {
    binding: String,
    scalar: bool,
    direct_first_predicate: Option<NdjsonDirectPredicate>,
    direct_cmp: Option<DirectCmp>,
    direct_count: Option<DirectCount>,
    direct_numeric: Option<DirectNumericReducer>,
    direct_predicate_sink: Option<DirectPredicateSink>,
    done: bool,
    stream: CompiledRowStream,
    values: Vec<Val>,
}

impl RunningConsumer {
    fn finish_value(&self) -> Result<Option<Val>, crate::EvalError> {
        if let Some(count) = self.direct_count.as_ref() {
            return Ok(Some(Val::Int(count.count as i64)));
        }
        if let Some(numeric) = self.direct_numeric.as_ref() {
            return Ok(Some(numeric.acc.value()));
        }
        if let Some(sink) = self.direct_predicate_sink.as_ref() {
            return Ok(Some(sink.value()));
        }
        self.stream.finish_result()
    }
}
#[derive(Clone)]
struct DirectCmp {
    steps: NdjsonPhysicalPath,
    op: BinOp,
    lit: Val,
}
#[derive(Clone)]
struct DirectCount {
    predicates: Vec<NdjsonDirectPredicate>,
    limit: Option<usize>,
    count: usize,
}
#[derive(Clone)]
struct DirectNumericReducer {
    predicates: Vec<NdjsonDirectPredicate>,
    value_path: NdjsonPhysicalPath,
    acc: NumericAccumulator,
}
#[derive(Clone)]
struct DirectPredicateSink {
    predicates: Vec<NdjsonDirectPredicate>,
    test: NdjsonDirectPredicate,
    sink: BuiltinPredicateSink,
    matched: bool,
    failed: bool,
    decided: bool,
}
impl DirectPredicateSink {
    fn apply_row(&mut self, row: &[u8]) -> Result<(), JetroEngineError> {
        if !eval_ndjson_byte_predicates_all(row, &self.predicates)? {
            return Ok(());
        }
        let keep = eval_ndjson_byte_predicate_row(row, &self.test)?.unwrap_or(false);
        match predicate_sink_result_demand(self.sink) {
            SinkResultDemand::UntilMatch if keep => {
                self.matched = true;
                self.decided = true;
            }
            SinkResultDemand::UntilFailure if !keep => {
                self.failed = true;
                self.decided = true;
            }
            SinkResultDemand::UntilMatch
            | SinkResultDemand::UntilFailure
            | SinkResultDemand::None => {}
        }
        Ok(())
    }

    fn value(&self) -> Val {
        match self.sink {
            BuiltinPredicateSink::Any => Val::Bool(self.matched),
            BuiltinPredicateSink::All => Val::Bool(!self.failed),
            BuiltinPredicateSink::FindIndex
            | BuiltinPredicateSink::IndicesWhere
            | BuiltinPredicateSink::FindOne => Val::Null,
        }
    }

    fn merge(&mut self, other: &Self) {
        debug_assert_eq!(self.sink, other.sink);
        self.matched |= other.matched;
        self.failed |= other.failed;
        match predicate_sink_result_demand(self.sink) {
            SinkResultDemand::UntilMatch => self.decided |= self.matched,
            SinkResultDemand::UntilFailure => self.decided |= self.failed,
            SinkResultDemand::None => {}
        }
    }
}

fn all_consumers_done(consumers: &[RunningConsumer]) -> bool {
    consumers
        .iter()
        .all(|consumer| consumer.done || consumer.stream.is_exhausted())
}
fn direct_first_match_predicate(plan: &RowStreamPlan) -> Option<NdjsonDirectPredicate> {
    let expr = RowStreamStage::first_filter_with_take_one_suffix(&plan.stages)?;
    direct_tape_predicate_for_expr(expr)
}
fn direct_first_match_cmp(plan: &RowStreamPlan) -> Option<DirectCmp> {
    let predicate = direct_first_match_predicate(plan)?;
    direct_cmp_from_predicate(&predicate)
}
fn direct_count_consumer(plan: &RowStreamPlan) -> Option<DirectCount> {
    let [prefix @ .., RowStreamStage::Count] = plan.stages.as_slice() else {
        return None;
    };
    let (predicates, limit) = direct_filter_take_prefix(prefix)?;
    Some(DirectCount {
        predicates,
        limit,
        count: 0,
    })
}
fn direct_numeric_consumer(plan: &RowStreamPlan) -> Option<DirectNumericReducer> {
    let [prefix @ .., RowStreamStage::Map(map), terminal] = plan.stages.as_slice() else {
        return None;
    };
    let reducer = terminal.numeric_reducer()?;
    let value_path = direct_tape_root_path_for_expr(map)?;
    let predicates = direct_filter_prefix(prefix)?;
    Some(DirectNumericReducer {
        predicates,
        value_path,
        acc: NumericAccumulator::from_reducer(reducer),
    })
}
fn direct_predicate_sink_consumer(plan: &RowStreamPlan) -> Option<DirectPredicateSink> {
    let [prefix @ .., terminal] = plan.stages.as_slice() else {
        return None;
    };
    let (sink, expr) = terminal.predicate_sink()?;
    if !matches!(
        predicate_sink_result_demand(sink),
        SinkResultDemand::UntilMatch | SinkResultDemand::UntilFailure
    ) {
        return None;
    }
    let test = direct_tape_predicate_for_expr(expr)?;
    let predicates = direct_filter_prefix(prefix)?;
    Some(DirectPredicateSink {
        predicates,
        test,
        sink,
        matched: false,
        failed: false,
        decided: false,
    })
}
fn direct_filter_prefix(stages: &[RowStreamStage]) -> Option<Vec<NdjsonDirectPredicate>> {
    direct_tape_predicates_for_exprs(RowStreamStage::filter_prefix(stages)?)
}
fn direct_filter_take_prefix(
    stages: &[RowStreamStage],
) -> Option<(Vec<NdjsonDirectPredicate>, Option<usize>)> {
    let prefix = RowStreamStage::filter_take_prefix(stages)?;
    let predicates = direct_tape_predicates_for_exprs(prefix.filters)?;
    Some((predicates, prefix.limit))
}
fn direct_cmp_from_predicate(predicate: &NdjsonDirectPredicate) -> Option<DirectCmp> {
    let (steps, op, lit) = direct_cmp_literal_predicate(predicate)?;
    Some(DirectCmp { steps, op, lit })
}

fn apply_fanout_row(
    engine: &JetroEngine,
    line_no: u64,
    row: Vec<u8>,
    consumers: &mut [RunningConsumer],
) -> Result<(), JetroEngineError> {
    let mut matched_value = None;
    let shared_view = shared_cmp_path(consumers)
        .and_then(|steps| raw_json_path_view(&row, steps).map(|view| (steps.to_vec(), view)));
    for consumer in consumers {
        if consumer.done || consumer.stream.is_exhausted() {
            continue;
        }
        if let Some(count) = consumer.direct_count.as_mut() {
            if eval_ndjson_byte_predicates_all(&row, &count.predicates)? {
                count.count += 1;
                if count.limit.is_some_and(|limit| count.count >= limit) {
                    consumer.done = true;
                }
            }
            continue;
        }
        if let Some(numeric) = consumer.direct_numeric.as_mut() {
            if eval_ndjson_byte_predicates_all(&row, &numeric.predicates)? {
                if let Some(value) = raw_json_path_view(&row, &numeric.value_path) {
                    numeric.acc.add_view(value);
                }
            }
            continue;
        }
        if let Some(sink) = consumer.direct_predicate_sink.as_mut() {
            sink.apply_row(&row)?;
            if sink.decided {
                consumer.done = true;
            }
            continue;
        }
        if let (Some((steps, view)), Some(cmp)) =
            (shared_view.as_ref(), consumer.direct_cmp.as_ref())
        {
            if physical_paths_equal(steps, &cmp.steps) {
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
            Some(path) if physical_paths_equal(path, &cmp.steps) => {}
            Some(_) => return None,
        }
    }
    (count > 1).then_some(shared?)
}

fn row_to_val(engine: &JetroEngine, line_no: u64, row: Vec<u8>) -> Result<Val, JetroEngineError> {
    let document = parse_row(engine, line_no, row)?;
    document
        .root_val_with(engine.keys())
        .map_err(|err| row_eval_error(line_no, err))
}
fn collect_parallel_direct_reducer_fanout(
    engine: &JetroEngine,
    path: &Path,
    plan: &RowStreamFanoutPlan,
    options: NdjsonOptions,
) -> Result<Option<Val>, JetroEngineError> {
    if options.parallelism == NdjsonParallelism::Off
        || plan.source.direction != RowStreamDirection::Forward
        || rayon::current_num_threads() <= 1
    {
        return Ok(None);
    }
    let metadata = std::fs::metadata(path)?;
    if metadata.len() < options.parallel_min_bytes {
        return Ok(None);
    }
    let reducers = plan
        .consumers
        .iter()
        .map(|consumer| DirectFanoutReducer::from_consumer(consumer))
        .collect::<Option<Vec<_>>>();
    let Some(reducers) = reducers.filter(|reducers| reducers.len() >= 2) else {
        return Ok(None);
    };

    let bytes = Arc::new(MappedBytes::open(path)?);
    let ranges = split_line_aligned_ranges(bytes.as_slice(), 4);
    if ranges.len() <= 1 {
        return Ok(None);
    }
    let parts = ranges
        .into_par_iter()
        .map(|range| scan_direct_reducer_partition(bytes.clone(), range, &reducers, options))
        .collect::<Result<Vec<_>, _>>()?;

    let mut merged = reducers;
    for part in parts {
        for (dst, src) in merged.iter_mut().zip(part) {
            dst.merge(&src);
        }
    }
    let value = finish_direct_reducer_body(engine, plan, &merged)?;
    Ok(Some(value))
}
#[derive(Clone)]
struct DirectFanoutReducer {
    binding: String,
    kind: DirectFanoutReducerKind,
}
#[derive(Clone)]
enum DirectFanoutReducerKind {
    Count(DirectCount),
    Numeric(DirectNumericReducer),
    PredicateSink(DirectPredicateSink),
}
impl DirectFanoutReducer {
    fn from_consumer(consumer: &RowStreamFanoutConsumer) -> Option<Self> {
        if let Some(mut count) =
            direct_count_consumer(&consumer.stream).filter(|count| count.limit.is_none())
        {
            count.count = 0;
            return Some(Self {
                binding: consumer.binding.clone(),
                kind: DirectFanoutReducerKind::Count(count),
            });
        }
        if let Some(numeric) = direct_numeric_consumer(&consumer.stream) {
            return Some(Self {
                binding: consumer.binding.clone(),
                kind: DirectFanoutReducerKind::Numeric(numeric),
            });
        }
        if let Some(sink) = direct_predicate_sink_consumer(&consumer.stream) {
            return Some(Self {
                binding: consumer.binding.clone(),
                kind: DirectFanoutReducerKind::PredicateSink(sink),
            });
        }
        None
    }

    fn apply_row(&mut self, row: &[u8]) -> Result<(), JetroEngineError> {
        match &mut self.kind {
            DirectFanoutReducerKind::Count(count) => {
                if eval_ndjson_byte_predicates_all(row, &count.predicates)? {
                    count.count += 1;
                }
            }
            DirectFanoutReducerKind::Numeric(numeric) => {
                if eval_ndjson_byte_predicates_all(row, &numeric.predicates)? {
                    if let Some(value) = raw_json_path_view(row, &numeric.value_path) {
                        numeric.acc.add_view(value);
                    }
                }
            }
            DirectFanoutReducerKind::PredicateSink(sink) => {
                if !sink.decided {
                    sink.apply_row(row)?;
                }
            }
        }
        Ok(())
    }

    fn merge(&mut self, other: &Self) {
        match (&mut self.kind, &other.kind) {
            (DirectFanoutReducerKind::Count(count), DirectFanoutReducerKind::Count(other)) => {
                count.count += other.count;
            }
            (
                DirectFanoutReducerKind::Numeric(numeric),
                DirectFanoutReducerKind::Numeric(other),
            ) => numeric.acc.merge(&other.acc),
            (
                DirectFanoutReducerKind::PredicateSink(sink),
                DirectFanoutReducerKind::PredicateSink(other),
            ) => sink.merge(other),
            _ => {}
        }
    }

    fn value(&self) -> Val {
        match &self.kind {
            DirectFanoutReducerKind::Count(count) => Val::Int(count.count as i64),
            DirectFanoutReducerKind::Numeric(numeric) => numeric.acc.value(),
            DirectFanoutReducerKind::PredicateSink(sink) => sink.value(),
        }
    }
}
fn scan_direct_reducer_partition(
    bytes: Arc<MappedBytes>,
    range: Range<usize>,
    reducers: &[DirectFanoutReducer],
    options: NdjsonOptions,
) -> Result<Vec<DirectFanoutReducer>, JetroEngineError> {
    let mut reducers = reducers.to_vec();
    let bytes = bytes.as_slice();
    for_each_framed_payload_in_range(bytes, range, options, |_row_range, row| {
        for reducer in &mut reducers {
            reducer.apply_row(row)?;
        }
        Ok(false)
    })?;
    Ok(reducers)
}
fn finish_direct_reducer_body(
    engine: &JetroEngine,
    plan: &RowStreamFanoutPlan,
    reducers: &[DirectFanoutReducer],
) -> Result<Val, JetroEngineError> {
    let mut env = Env::new(Val::Null);
    for reducer in reducers {
        env = env.with_var(&reducer.binding, reducer.value());
    }
    let body = Compiler::compile(&plan.body, "<ndjson-rows-fanout-body>");
    engine
        .lock_vm()
        .exec_in_env(&body, &env)
        .map_err(JetroEngineError::Eval)
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
        assert!(direct_first_match_cmp(&plan.consumers[0].stream).is_some());
    }

    #[test]
    fn declines_mixed_direction_shape_fanout() {
        let query = r#"{newest: $.rows().reverse().find(name == "Ada").first(), oldest: $.rows().find(name == "Ada").first()}"#;
        let plan = lower_rows_fanout_query(query, RowStreamSourceKind::NdjsonRows).unwrap();
        assert!(plan.is_none());
    }

    #[test]
    fn declines_single_shape_stream_for_existing_subquery_path() {
        let query = r#"{active_count: $.rows().filter(active == true).count()}"#;
        let plan = lower_rows_fanout_query(query, RowStreamSourceKind::NdjsonRows).unwrap();
        assert!(plan.is_none());
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
        let query = r#"{user_a: $.rows().reverse().find(name == "Ada").first(), user_b: $.rows().reverse().find(name == "Bob").first()}"#;
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
    fn executes_let_stream_body_shape_fanout() {
        let path = temp_ndjson(
            "let-body",
            &[
                r#"{"name":"Ada","version":1}"#,
                r#"{"name":"Bob","version":1}"#,
                r#"{"name":"Ada","version":2}"#,
            ],
        );
        let query = r#"let stream = $.rows().reverse() in {user_a: stream.find(name == "Ada").first(), user_b: stream.find(name == "Bob").first()}"#;
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

    #[test]
    fn executes_nested_shape_rows_subqueries_as_fanout() {
        let path = temp_ndjson(
            "nested",
            &[
                r#"{"name":"Ada","version":1}"#,
                r#"{"name":"Bob","version":1}"#,
                r#"{"name":"Ada","version":2}"#,
            ],
        );
        let query = r#"{users: {a: $.rows().reverse().find($.name == "Ada").first(), b: $.rows().reverse().find($.name == "Bob").first()}}"#;
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
            r#"{"users":{"a":{"name":"Ada","version":2},"b":{"name":"Bob","version":1}}}"#
        );
    }

    #[test]
    fn executes_mixed_find_and_count_fanout() {
        let path = temp_ndjson(
            "count",
            &[
                r#"{"name":"Ada","active":true}"#,
                r#"{"name":"Bob","active":false}"#,
                r#"{"name":"Cara","active":true}"#,
            ],
        );
        let query = r#"{first_active: $.rows().find(active == true).first(), active_count: $.rows().filter(active == true).count()}"#;
        let plan = lower_rows_fanout_query(query, RowStreamSourceKind::NdjsonRows)
            .unwrap()
            .expect("fanout plan");
        assert!(direct_count_consumer(&plan.consumers[1].stream).is_some());
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
            r#"{"first_active":{"name":"Ada","active":true},"active_count":2}"#
        );
    }

    #[test]
    fn executes_len_alias_count_fanout() {
        let path = temp_ndjson(
            "len",
            &[
                r#"{"active":true}"#,
                r#"{"active":false}"#,
                r#"{"active":true}"#,
            ],
        );
        let query =
            r#"{active_len: $.rows().filter($.active == true).len(), all_len: $.rows().len()}"#;
        let plan = lower_rows_fanout_query(query, RowStreamSourceKind::NdjsonRows)
            .unwrap()
            .expect("fanout plan");
        assert!(plan
            .consumers
            .iter()
            .all(|consumer| direct_count_consumer(&consumer.stream).is_some()));
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
        assert_eq!(
            String::from_utf8(out).unwrap().trim(),
            r#"{"active_len":2,"all_len":3}"#
        );
    }

    #[test]
    fn executes_shared_path_comparison_fanout() {
        let path = temp_ndjson(
            "cmp",
            &[
                r#"{"name":"low","score":3}"#,
                r#"{"name":"mid","score":7}"#,
                r#"{"name":"high","score":11}"#,
            ],
        );
        let query =
            r#"{gte: $.rows().find(score >= 10).first(), lt: $.rows().find(5 > score).first()}"#;
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
            r#"{"gte":{"name":"high","score":11},"lt":{"name":"low","score":3}}"#
        );
    }

    #[test]
    fn executes_mixed_find_count_and_sum_fanout() {
        let path = temp_ndjson(
            "sum",
            &[
                r#"{"name":"Ada","active":true,"price":10}"#,
                r#"{"name":"Bob","active":false,"price":30}"#,
                r#"{"name":"Cara","active":true,"price":5}"#,
            ],
        );
        let query = r#"{first_active: $.rows().find(active == true).first(), active_count: $.rows().filter(active == true).count(), active_total: $.rows().filter(active == true).map($.price).sum()}"#;
        let plan = lower_rows_fanout_query(query, RowStreamSourceKind::NdjsonRows)
            .unwrap()
            .expect("fanout plan");
        assert!(direct_numeric_consumer(&plan.consumers[2].stream).is_some());
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
            r#"{"first_active":{"name":"Ada","active":true,"price":10},"active_count":2,"active_total":15}"#
        );
    }

    #[test]
    fn executes_numeric_reducer_fanout() {
        let path = temp_ndjson(
            "numeric",
            &[
                r#"{"active":true,"price":10}"#,
                r#"{"active":false,"price":30}"#,
                r#"{"active":true,"price":5}"#,
            ],
        );
        let query = r#"{avg_price: $.rows().filter(active == true).map($.price).avg(), min_price: $.rows().filter(active == true).map($.price).min(), max_price: $.rows().filter(active == true).map($.price).max()}"#;
        let plan = lower_rows_fanout_query(query, RowStreamSourceKind::NdjsonRows)
            .unwrap()
            .expect("fanout plan");
        assert!(plan
            .consumers
            .iter()
            .all(|consumer| direct_numeric_consumer(&consumer.stream).is_some()));
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
            r#"{"avg_price":7.5,"min_price":5,"max_price":10}"#
        );
    }

    #[test]
    fn direct_numeric_fanout_preserves_mixed_number_semantics() {
        let path = temp_ndjson(
            "numeric-mixed",
            &[
                r#"{"active":true,"price":10}"#,
                r#"{"active":true,"price":2.5}"#,
                r#"{"active":true,"price":3}"#,
            ],
        );
        let query = r#"{min_price: $.rows().filter(active == true).map($.price).min(), max_price: $.rows().filter(active == true).map($.price).max()}"#;
        let plan = lower_rows_fanout_query(query, RowStreamSourceKind::NdjsonRows)
            .unwrap()
            .expect("fanout plan");
        assert!(plan
            .consumers
            .iter()
            .all(|consumer| direct_numeric_consumer(&consumer.stream).is_some()));

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
        assert_eq!(got.trim(), r#"{"min_price":2.5,"max_price":10.0}"#);
    }

    #[test]
    fn executes_predicate_sink_fanout() {
        let path = temp_ndjson(
            "predicate-sinks",
            &[
                r#"{"active":true,"price":10}"#,
                r#"{"active":false,"price":30}"#,
                r#"{"active":true,"price":5}"#,
            ],
        );
        let query = r#"{has_inactive: $.rows().any(active == false), all_active: $.rows().all(active == true), active_count: $.rows().filter(active == true).count()}"#;
        let plan = lower_rows_fanout_query(query, RowStreamSourceKind::NdjsonRows)
            .unwrap()
            .expect("fanout plan");
        assert_eq!(plan.consumers.len(), 3);
        assert!(plan
            .consumers
            .iter()
            .take(2)
            .all(|consumer| direct_predicate_sink_consumer(&consumer.stream).is_some()));
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
            r#"{"has_inactive":true,"all_active":false,"active_count":2}"#
        );
    }

    #[test]
    fn executes_filtered_predicate_sink_fanout() {
        let path = temp_ndjson(
            "filtered-predicate-sinks",
            &[
                r#"{"kind":"prod","ok":true}"#,
                r#"{"kind":"test","ok":false}"#,
                r#"{"kind":"prod","ok":true}"#,
            ],
        );
        let query = r#"{all_prod_ok: $.rows().filter($.kind == "prod").all($.ok == true), any_test_ok: $.rows().filter($.kind == "test").any($.ok == true)}"#;
        let plan = lower_rows_fanout_query(query, RowStreamSourceKind::NdjsonRows)
            .unwrap()
            .expect("fanout plan");
        assert!(plan
            .consumers
            .iter()
            .all(|consumer| direct_predicate_sink_consumer(&consumer.stream).is_some()));
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
        assert_eq!(
            String::from_utf8(out).unwrap().trim(),
            r#"{"all_prod_ok":true,"any_test_ok":false}"#
        );
    }

    #[test]
    fn parallel_direct_reducer_fanout_matches_sequential_result() {
        let path = temp_ndjson(
            "parallel-reducers",
            &[
                r#"{"active":true,"price":10}"#,
                r#"{"active":false,"price":30}"#,
                r#"{"active":true,"price":5}"#,
                r#"{"active":true,"price":7}"#,
            ],
        );
        let query = r#"{active_count: $.rows().filter(active == true).count(), active_total: $.rows().filter(active == true).map($.price).sum()}"#;
        let plan = lower_rows_fanout_query(query, RowStreamSourceKind::NdjsonRows)
            .unwrap()
            .expect("fanout plan");
        let engine = JetroEngine::new();
        let value = collect_parallel_direct_reducer_fanout(
            &engine,
            &path,
            &plan,
            NdjsonOptions::default().with_parallel_min_bytes(0),
        )
        .unwrap()
        .expect("parallel direct reducer fanout");
        std::fs::remove_file(path).ok();
        assert_eq!(
            serde_json::Value::from(value),
            serde_json::json!({"active_count": 3, "active_total": 22})
        );
    }

    #[test]
    fn parallel_direct_predicate_fanout_matches_sequential_result() {
        let path = temp_ndjson(
            "parallel-predicate-sinks",
            &[
                r#"{"active":true,"price":10}"#,
                r#"{"active":true,"price":20}"#,
                r#"{"active":false,"price":30}"#,
                r#"{"active":true,"price":40}"#,
            ],
        );
        let query = r#"{has_expensive: $.rows().any($.price > 35), all_active: $.rows().all($.active == true), active_count: $.rows().filter($.active == true).count()}"#;
        let plan = lower_rows_fanout_query(query, RowStreamSourceKind::NdjsonRows)
            .unwrap()
            .expect("fanout plan");
        let engine = JetroEngine::new();
        let value = collect_parallel_direct_reducer_fanout(
            &engine,
            &path,
            &plan,
            NdjsonOptions::default().with_parallel_min_bytes(0),
        )
        .unwrap()
        .expect("parallel direct predicate fanout");
        std::fs::remove_file(path).ok();
        assert_eq!(
            serde_json::Value::from(value),
            serde_json::json!({"has_expensive": true, "all_active": false, "active_count": 3})
        );
    }
}
