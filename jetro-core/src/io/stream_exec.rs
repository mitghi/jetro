use super::ndjson::{parse_row, row_eval_error};
use super::ndjson_byte::eval_ndjson_byte_predicate_row;
use super::ndjson_direct::{
    direct_tape_plan_for_expr, direct_tape_predicate_for_expr, NdjsonDirectPredicate,
    NdjsonDirectTapePlan,
};
use super::ndjson_distinct::{distinct_key_bytes, AdaptiveDistinctKeys};
use super::stream_direct::{direct_map_can_write, insert_direct_distinct_key, write_direct_map};
use super::stream_numeric::NumericAccumulator;
use super::stream_plan::{RowStreamPlan, RowStreamStage};
use super::stream_types::{RowStreamRowResult, RowStreamStats};
use crate::builtins::registry::predicate_sink_result_demand;
use crate::builtins::BuiltinPredicateSink;
use crate::compile::compiler::Compiler;
use crate::data::value::Val;
use crate::plan::demand::SinkResultDemand;
use crate::util::is_truthy;
use crate::vm::opcode::Program;
use crate::{EvalError, Jetro, JetroEngine, JetroEngineError, VM};
use std::sync::Arc;

pub(super) struct CompiledRowStream {
    stages: Vec<CompiledRowStreamStage>,
    exhausted: bool,
    stats: RowStreamStats,
}

impl CompiledRowStream {
    pub(super) fn new(plan: &RowStreamPlan) -> Self {
        let stages: Vec<_> = plan
            .stages
            .iter()
            .enumerate()
            .map(|(idx, stage)| CompiledRowStreamStage::new(stage, idx + 1 == plan.stages.len()))
            .collect();
        let exhausted = plan.demand.retained_limit == Some(0);
        Self {
            stages,
            exhausted,
            stats: RowStreamStats {
                source: plan.source,
                direction: plan.direction,
                ..RowStreamStats::default()
            },
        }
    }

    pub(super) fn is_exhausted(&self) -> bool {
        self.exhausted
    }

    pub(super) fn stats(&self) -> &RowStreamStats {
        &self.stats
    }

    pub(super) fn apply_owned_row(
        &mut self,
        engine: &JetroEngine,
        line_no: u64,
        row: Vec<u8>,
    ) -> Result<RowStreamRowResult, JetroEngineError> {
        self.stats.rows_scanned += 1;
        let mut row = Some(row);
        let mut document = None;
        let mut value = None;
        let mut vm = None;
        for stage in &mut self.stages {
            match stage {
                CompiledRowStreamStage::Filter { program, direct } => {
                    let keep = eval_owned_row_predicate(
                        engine,
                        line_no,
                        program,
                        direct.as_ref(),
                        &mut row,
                        &mut document,
                        &mut value,
                        &mut vm,
                        &mut self.stats,
                    )?;
                    if !is_truthy(&keep) {
                        self.stats.rows_filtered += 1;
                        return Ok(RowStreamRowResult::Skip);
                    }
                }
                CompiledRowStreamStage::DistinctBy {
                    program,
                    seen,
                    direct,
                } => {
                    if let (Some(plan), Some(raw_row)) = (direct.as_ref(), row.as_deref()) {
                        if let Some(inserted) = insert_direct_distinct_key(seen, raw_row, plan) {
                            self.stats.direct_key_rows += 1;
                            if !inserted {
                                self.stats.duplicate_rows += 1;
                                return Ok(RowStreamRowResult::Skip);
                            }
                            continue;
                        }
                    }

                    let value = ensure_row_stream_value(
                        engine,
                        line_no,
                        &mut row,
                        &mut document,
                        &mut value,
                    )?;
                    let vm = vm.get_or_insert_with(|| engine.lock_vm());
                    let key = vm
                        .execute_val_raw_fresh_root(program, value.clone())
                        .map_err(|err| row_eval_error(line_no, err))?;
                    let key = distinct_key_bytes(&key)?;
                    self.stats.fallback_key_rows += 1;
                    if !seen.insert(key) {
                        self.stats.duplicate_rows += 1;
                        return Ok(RowStreamRowResult::Skip);
                    }
                }
                CompiledRowStreamStage::Take { limit, seen } => {
                    if *seen >= *limit {
                        self.exhausted = true;
                        return Ok(RowStreamRowResult::Stop);
                    }
                    *seen += 1;
                    if *seen >= *limit {
                        self.exhausted = true;
                    }
                }
                CompiledRowStreamStage::Last { value: last } => {
                    *last = Some(ensure_row_stream_value(
                        engine,
                        line_no,
                        &mut row,
                        &mut document,
                        &mut value,
                    )?);
                    return Ok(RowStreamRowResult::Skip);
                }
                CompiledRowStreamStage::Count { count } => {
                    *count += 1;
                    return Ok(RowStreamRowResult::Skip);
                }
                CompiledRowStreamStage::Numeric { acc } => {
                    let value = ensure_row_stream_value(
                        engine,
                        line_no,
                        &mut row,
                        &mut document,
                        &mut value,
                    )?;
                    acc.add_val(&value);
                    return Ok(RowStreamRowResult::Skip);
                }
                CompiledRowStreamStage::PredicateSink {
                    sink,
                    program,
                    state,
                    direct,
                } => {
                    let keep = eval_owned_row_predicate(
                        engine,
                        line_no,
                        program,
                        direct.as_ref(),
                        &mut row,
                        &mut document,
                        &mut value,
                        &mut vm,
                        &mut self.stats,
                    )?;
                    let sink = *sink;
                    let keep = is_truthy(&keep);
                    if sink.returns_matching_row() && keep {
                        if state.value.is_some() {
                            return Err(row_eval_error(
                                line_no,
                                EvalError(
                                    "find_one: expected exactly one element, got multiple".into(),
                                ),
                            ));
                        }
                        state.value = Some(ensure_row_stream_value(
                            engine,
                            line_no,
                            &mut row,
                            &mut document,
                            &mut value,
                        )?);
                    } else if state.observe_short_circuit(sink, keep) {
                        self.exhausted = true;
                        return Ok(RowStreamRowResult::Stop);
                    }
                    return Ok(RowStreamRowResult::Skip);
                }
                CompiledRowStreamStage::Map { program, direct } => {
                    if let Some(raw_row) = row.as_deref() {
                        if let Some(bytes) = write_direct_map(raw_row, direct.as_ref())? {
                            self.stats.direct_project_rows += 1;
                            self.stats.rows_emitted += 1;
                            return Ok(RowStreamRowResult::EmitBytes(bytes));
                        }
                    }

                    let current = ensure_row_stream_value(
                        engine,
                        line_no,
                        &mut row,
                        &mut document,
                        &mut value,
                    )?;
                    let vm = vm.get_or_insert_with(|| engine.lock_vm());
                    value = Some(
                        vm.execute_val_raw_fresh_root(program, current)
                            .map_err(|err| row_eval_error(line_no, err))?,
                    );
                    self.stats.fallback_project_rows += 1;
                }
            }
        }

        if value.is_none() {
            if let Some(row) = row {
                self.stats.rows_emitted += 1;
                return Ok(RowStreamRowResult::EmitBytes(row));
            }
        }
        let value = ensure_row_stream_value(engine, line_no, &mut row, &mut document, &mut value)?;
        self.stats.rows_emitted += 1;
        Ok(RowStreamRowResult::Emit(value))
    }

    pub(super) fn apply_val_row(
        &mut self,
        vm: &mut VM,
        row: Val,
    ) -> Result<RowStreamRowResult, EvalError> {
        self.stats.rows_scanned += 1;
        let mut value = row;
        for stage in &mut self.stages {
            match stage {
                CompiledRowStreamStage::Filter { program, .. } => {
                    let keep = vm.execute_val_raw_fresh_root(program, value.clone())?;
                    self.stats.fallback_filter_rows += 1;
                    if !is_truthy(&keep) {
                        self.stats.rows_filtered += 1;
                        return Ok(RowStreamRowResult::Skip);
                    }
                }
                CompiledRowStreamStage::DistinctBy { program, seen, .. } => {
                    let key = vm.execute_val_raw_fresh_root(program, value.clone())?;
                    let key = distinct_key_bytes(&key).map_err(|err| EvalError(err.to_string()))?;
                    self.stats.fallback_key_rows += 1;
                    if !seen.insert(key) {
                        self.stats.duplicate_rows += 1;
                        return Ok(RowStreamRowResult::Skip);
                    }
                }
                CompiledRowStreamStage::Take { limit, seen } => {
                    if *seen >= *limit {
                        self.exhausted = true;
                        return Ok(RowStreamRowResult::Stop);
                    }
                    *seen += 1;
                    if *seen >= *limit {
                        self.exhausted = true;
                    }
                }
                CompiledRowStreamStage::Last { value: last } => {
                    *last = Some(value);
                    return Ok(RowStreamRowResult::Skip);
                }
                CompiledRowStreamStage::Count { count } => {
                    *count += 1;
                    return Ok(RowStreamRowResult::Skip);
                }
                CompiledRowStreamStage::Numeric { acc } => {
                    acc.add_val(&value);
                    return Ok(RowStreamRowResult::Skip);
                }
                CompiledRowStreamStage::PredicateSink {
                    sink,
                    program,
                    state,
                    ..
                } => {
                    let sink = *sink;
                    if sink.returns_matching_row() {
                        let keep = vm.execute_val_raw_fresh_root(program, value.clone())?;
                        self.stats.fallback_filter_rows += 1;
                        if is_truthy(&keep) {
                            if state.value.is_some() {
                                return Err(EvalError(
                                    "find_one: expected exactly one element, got multiple".into(),
                                ));
                            }
                            state.value = Some(value);
                        }
                    } else {
                        let keep = vm.execute_val_raw_fresh_root(program, value)?;
                        self.stats.fallback_filter_rows += 1;
                        let keep = is_truthy(&keep);
                        if state.observe_short_circuit(sink, keep) {
                            self.exhausted = true;
                            return Ok(RowStreamRowResult::Stop);
                        }
                    }
                    return Ok(RowStreamRowResult::Skip);
                }
                CompiledRowStreamStage::Map { program, .. } => {
                    value = vm.execute_val_raw_fresh_root(program, value)?;
                    self.stats.fallback_project_rows += 1;
                }
            }
        }
        self.stats.rows_emitted += 1;
        Ok(RowStreamRowResult::Emit(value))
    }

    pub(super) fn finish_result(&self) -> Result<Option<Val>, EvalError> {
        for stage in &self.stages {
            let value = match stage {
                CompiledRowStreamStage::Last { value } => value.clone().unwrap_or(Val::Null),
                CompiledRowStreamStage::Count { count } => Val::Int(*count as i64),
                CompiledRowStreamStage::Numeric { acc } => acc.value(),
                CompiledRowStreamStage::PredicateSink { sink, state, .. } => match sink {
                    BuiltinPredicateSink::Any => Val::Bool(state.matched),
                    BuiltinPredicateSink::All => Val::Bool(!state.failed),
                    BuiltinPredicateSink::FindOne => {
                        return state.value.clone().map(Some).ok_or_else(|| {
                            EvalError("find_one: expected exactly one element, got 0".into())
                        });
                    }
                    BuiltinPredicateSink::FindIndex | BuiltinPredicateSink::IndicesWhere => {
                        continue;
                    }
                },
                _ => continue,
            };
            return Ok(Some(value));
        }
        Ok(None)
    }
}

fn eval_owned_row_predicate<'a>(
    engine: &'a JetroEngine,
    line_no: u64,
    program: &Program,
    direct: Option<&NdjsonDirectPredicate>,
    row: &mut Option<Vec<u8>>,
    document: &mut Option<Jetro>,
    value: &mut Option<Val>,
    vm: &mut Option<std::sync::MutexGuard<'a, VM>>,
    stats: &mut RowStreamStats,
) -> Result<Val, JetroEngineError> {
    if let (Some(predicate), Some(raw_row)) = (direct, row.as_deref()) {
        if let Some(keep) = eval_ndjson_byte_predicate_row(raw_row, predicate)? {
            stats.direct_filter_rows += 1;
            return Ok(Val::Bool(keep));
        }
    }

    let value = ensure_row_stream_value(engine, line_no, row, document, value)?;
    let vm = vm.get_or_insert_with(|| engine.lock_vm());
    let keep = vm
        .execute_val_raw_fresh_root(program, value)
        .map_err(|err| row_eval_error(line_no, err))?;
    stats.fallback_filter_rows += 1;
    Ok(keep)
}

pub(super) fn finish_collected_row_stream(
    plan: &RowStreamPlan,
    stream: &CompiledRowStream,
    out: Vec<Val>,
) -> Result<(Val, RowStreamStats), EvalError> {
    let stats = stream.stats().clone();
    let value = if let Some(value) = stream.finish_result()? {
        value
    } else if plan.demand.retained_limit == Some(1) {
        out.into_iter().next().unwrap_or(Val::Null)
    } else {
        Val::Arr(Arc::new(out))
    };
    Ok((value, stats))
}

fn ensure_row_stream_value(
    engine: &JetroEngine,
    line_no: u64,
    row: &mut Option<Vec<u8>>,
    document: &mut Option<Jetro>,
    value: &mut Option<Val>,
) -> Result<Val, JetroEngineError> {
    if let Some(value) = value.as_ref() {
        return Ok(value.clone());
    }
    if document.is_none() {
        let row = row.take().ok_or_else(|| {
            JetroEngineError::Eval(EvalError("rows() stream row was already consumed".into()))
        })?;
        *document = Some(parse_row(engine, line_no, row)?);
    }
    let root = document
        .as_ref()
        .expect("row document initialized")
        .root_val_with(engine.keys())
        .map_err(|err| row_eval_error(line_no, err))?;
    *value = Some(root.clone());
    Ok(root)
}

#[derive(Default)]
struct PredicateSinkState {
    matched: bool,
    failed: bool,
    value: Option<Val>,
}

impl PredicateSinkState {
    fn observe_short_circuit(&mut self, sink: BuiltinPredicateSink, matched: bool) -> bool {
        match predicate_sink_result_demand(sink) {
            SinkResultDemand::UntilMatch if matched => {
                self.matched = true;
                true
            }
            SinkResultDemand::UntilFailure if !matched => {
                self.failed = true;
                true
            }
            SinkResultDemand::UntilMatch
            | SinkResultDemand::UntilFailure
            | SinkResultDemand::None => false,
        }
    }
}

enum CompiledRowStreamStage {
    Filter {
        program: Program,
        direct: Option<NdjsonDirectPredicate>,
    },
    DistinctBy {
        program: Program,
        seen: AdaptiveDistinctKeys,
        direct: Option<NdjsonDirectTapePlan>,
    },
    Take {
        limit: usize,
        seen: usize,
    },
    Last {
        value: Option<Val>,
    },
    Count {
        count: usize,
    },
    Numeric {
        acc: NumericAccumulator,
    },
    PredicateSink {
        sink: BuiltinPredicateSink,
        program: Program,
        state: PredicateSinkState,
        direct: Option<NdjsonDirectPredicate>,
    },
    Map {
        program: Program,
        direct: Option<NdjsonDirectTapePlan>,
    },
}

impl CompiledRowStreamStage {
    fn new(stage: &RowStreamStage, is_last: bool) -> Self {
        if let Some((sink, expr)) = stage.predicate_sink() {
            return Self::PredicateSink {
                sink,
                program: Compiler::compile(expr, "<ndjson-rows-predicate-sink>"),
                state: PredicateSinkState::default(),
                direct: direct_tape_predicate_for_expr(expr),
            };
        }

        match stage {
            RowStreamStage::Filter(expr) => Self::Filter {
                program: Compiler::compile(expr, "<ndjson-rows-filter>"),
                direct: direct_tape_predicate_for_expr(expr),
            },
            RowStreamStage::DistinctBy(expr) => Self::DistinctBy {
                program: Compiler::compile(expr, "<ndjson-rows-distinct-by>"),
                seen: AdaptiveDistinctKeys::default(),
                direct: direct_tape_plan_for_expr(expr),
            },
            RowStreamStage::Take(limit) => Self::Take {
                limit: *limit,
                seen: 0,
            },
            RowStreamStage::Last => Self::Last { value: None },
            RowStreamStage::Count => Self::Count { count: 0 },
            RowStreamStage::Numeric(reducer) => Self::Numeric {
                acc: NumericAccumulator::from_reducer(*reducer),
            },
            RowStreamStage::PredicateSink { .. } => {
                unreachable!("predicate sink stages are handled through builtin metadata")
            }
            RowStreamStage::Map(expr) => Self::Map {
                program: Compiler::compile(expr, "<ndjson-rows-map>"),
                direct: is_last
                    .then(|| direct_tape_plan_for_expr(expr).filter(direct_map_can_write))
                    .flatten(),
            },
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::stream_plan::{lower_root_rows_expr, RowStreamSourceKind};
    use crate::parse::parser::parse;

    #[test]
    fn stats_track_direct_and_fallback_stage_paths() {
        let expr = parse("$.rows().filter($.active).distinct_by($.id).map($.id)").unwrap();
        let plan = lower_root_rows_expr(&expr, RowStreamSourceKind::NdjsonRows)
            .unwrap()
            .unwrap();
        let engine = JetroEngine::new();
        let mut stream = CompiledRowStream::new(&plan);

        assert!(matches!(
            stream
                .apply_owned_row(&engine, 1, br#"{"id":"a","active":false}"#.to_vec())
                .unwrap(),
            RowStreamRowResult::Skip
        ));
        assert!(matches!(
            stream
                .apply_owned_row(&engine, 2, br#"{"id":"a","active":true}"#.to_vec())
                .unwrap(),
            RowStreamRowResult::Emit(_) | RowStreamRowResult::EmitBytes(_)
        ));
        assert!(matches!(
            stream
                .apply_owned_row(&engine, 3, br#"{"id":"a","active":true}"#.to_vec())
                .unwrap(),
            RowStreamRowResult::Skip
        ));

        let stats = stream.stats();
        assert_eq!(stats.rows_scanned, 3);
        assert_eq!(stats.rows_filtered, 1);
        assert_eq!(stats.duplicate_rows, 1);
        assert_eq!(stats.rows_emitted, 1);
        assert_eq!(stats.direct_filter_rows + stats.fallback_filter_rows, 3);
        assert_eq!(stats.direct_key_rows + stats.fallback_key_rows, 2);
        assert_eq!(stats.direct_project_rows + stats.fallback_project_rows, 1);
    }
}
