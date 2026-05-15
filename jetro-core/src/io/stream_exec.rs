use super::ndjson::{parse_row, row_eval_error};
use super::ndjson_distinct::{distinct_key_bytes, AdaptiveDistinctKeys};
use super::stream_plan::{RowStreamPlan, RowStreamStage};
use crate::compile::compiler::Compiler;
use crate::data::value::Val;
use crate::util::is_truthy;
use crate::vm::opcode::Program;
use crate::{EvalError, Jetro, JetroEngine, JetroEngineError, VM};

#[cfg(feature = "simd-json")]
use super::ndjson_byte::eval_ndjson_byte_predicate_row;
#[cfg(feature = "simd-json")]
use super::ndjson_direct::{direct_tape_predicate_for_expr, NdjsonDirectPredicate};

pub(super) enum RowStreamRowResult {
    Emit(Val),
    EmitBytes(Vec<u8>),
    Skip,
    Stop,
}

pub(super) struct CompiledRowStream {
    stages: Vec<CompiledRowStreamStage>,
    exhausted: bool,
}

impl CompiledRowStream {
    pub(super) fn new(plan: &RowStreamPlan) -> Self {
        let stages: Vec<_> = plan.stages.iter().map(CompiledRowStreamStage::new).collect();
        let exhausted = stages
            .iter()
            .any(|stage| matches!(stage, CompiledRowStreamStage::Take { limit: 0, .. }));
        Self { stages, exhausted }
    }

    pub(super) fn is_exhausted(&self) -> bool {
        self.exhausted
    }

    pub(super) fn apply_owned_row(
        &mut self,
        engine: &JetroEngine,
        line_no: u64,
        row: Vec<u8>,
    ) -> Result<RowStreamRowResult, JetroEngineError> {
        let mut row = Some(row);
        let mut document = None;
        let mut value = None;
        let mut vm = engine.lock_vm();
        for stage in &mut self.stages {
            match stage {
                CompiledRowStreamStage::Filter {
                    program,
                    #[cfg(feature = "simd-json")]
                    direct,
                } => {
                    #[cfg(feature = "simd-json")]
                    if let (Some(predicate), Some(raw_row)) = (direct.as_ref(), row.as_deref()) {
                        if let Some(keep) = eval_ndjson_byte_predicate_row(raw_row, predicate)? {
                            if !keep {
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
                    let keep = vm
                        .execute_val_raw_fresh_root(program, value.clone())
                        .map_err(|err| row_eval_error(line_no, err))?;
                    if !is_truthy(&keep) {
                        return Ok(RowStreamRowResult::Skip);
                    }
                }
                CompiledRowStreamStage::DistinctBy {
                    program,
                    seen,
                } => {
                    let value = ensure_row_stream_value(
                        engine,
                        line_no,
                        &mut row,
                        &mut document,
                        &mut value,
                    )?;
                    let key = vm
                        .execute_val_raw_fresh_root(program, value.clone())
                        .map_err(|err| row_eval_error(line_no, err))?;
                    let key = distinct_key_bytes(&key)?;
                    if !seen.insert(key) {
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
                CompiledRowStreamStage::Map { program } => {
                    let current = ensure_row_stream_value(
                        engine,
                        line_no,
                        &mut row,
                        &mut document,
                        &mut value,
                    )?;
                    value = Some(
                        vm.execute_val_raw_fresh_root(program, current)
                            .map_err(|err| row_eval_error(line_no, err))?,
                    );
                }
            }
        }

        if value.is_none() {
            if let Some(row) = row {
                return Ok(RowStreamRowResult::EmitBytes(row));
            }
        }
        let value = ensure_row_stream_value(engine, line_no, &mut row, &mut document, &mut value)?;
        Ok(RowStreamRowResult::Emit(value))
    }

    #[allow(dead_code)]
    pub(super) fn apply_val_row(
        &mut self,
        vm: &mut VM,
        row: Val,
    ) -> Result<RowStreamRowResult, EvalError> {
        let mut value = row;
        for stage in &mut self.stages {
            match stage {
                CompiledRowStreamStage::Filter { program, .. } => {
                    let keep = vm.execute_val_raw_fresh_root(program, value.clone())?;
                    if !is_truthy(&keep) {
                        return Ok(RowStreamRowResult::Skip);
                    }
                }
                CompiledRowStreamStage::DistinctBy { program, seen } => {
                    let key = vm.execute_val_raw_fresh_root(program, value.clone())?;
                    let key = distinct_key_bytes(&key)
                        .map_err(|err| EvalError(err.to_string()))?;
                    if !seen.insert(key) {
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
                CompiledRowStreamStage::Map { program } => {
                    value = vm.execute_val_raw_fresh_root(program, value)?;
                }
            }
        }
        Ok(RowStreamRowResult::Emit(value))
    }
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

enum CompiledRowStreamStage {
    Filter {
        program: Program,
        #[cfg(feature = "simd-json")]
        direct: Option<NdjsonDirectPredicate>,
    },
    DistinctBy {
        program: Program,
        seen: AdaptiveDistinctKeys,
    },
    Take {
        limit: usize,
        seen: usize,
    },
    Map { program: Program },
}

impl CompiledRowStreamStage {
    fn new(stage: &RowStreamStage) -> Self {
        match stage {
            RowStreamStage::Filter(expr) => Self::Filter {
                program: Compiler::compile(expr, "<ndjson-rows-filter>"),
                #[cfg(feature = "simd-json")]
                direct: direct_tape_predicate_for_expr(expr),
            },
            RowStreamStage::DistinctBy(expr) => Self::DistinctBy {
                program: Compiler::compile(expr, "<ndjson-rows-distinct-by>"),
                seen: AdaptiveDistinctKeys::default(),
            },
            RowStreamStage::Take(limit) => Self::Take {
                limit: *limit,
                seen: 0,
            },
            RowStreamStage::Map(expr) => Self::Map {
                program: Compiler::compile(expr, "<ndjson-rows-map>"),
            },
        }
    }
}
