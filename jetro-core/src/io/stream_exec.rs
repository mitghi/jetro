use super::ndjson::{parse_row, row_eval_error};
use super::ndjson_distinct::{distinct_key_bytes, AdaptiveDistinctKeys};
use super::stream_plan::{RowStreamPlan, RowStreamStage};
use crate::compile::compiler::Compiler;
use crate::data::value::Val;
use crate::util::is_truthy;
use crate::vm::opcode::Program;
use crate::{EvalError, Jetro, JetroEngine, JetroEngineError};

#[cfg(feature = "simd-json")]
use super::ndjson_byte::{
    eval_ndjson_byte_predicate_row, raw_json_byte_path_value, tape_plan_can_write_byte_row,
    write_ndjson_byte_tape_plan_row, BytePlanWrite, RawFieldValue,
};
#[cfg(feature = "simd-json")]
use super::ndjson_direct::{
    direct_tape_plan_for_expr, direct_tape_predicate_for_expr, NdjsonDirectPredicate,
    NdjsonDirectTapePlan,
};
#[cfg(feature = "simd-json")]
use super::ndjson_distinct::raw_distinct_key_bytes;

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
        let stage_count = self.stages.len();

        for (idx, stage) in self.stages.iter_mut().enumerate() {
            let is_last_stage = idx + 1 == stage_count;
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
                        .execute_val_raw(program, value.clone())
                        .map_err(|err| row_eval_error(line_no, err))?;
                    if !is_truthy(&keep) {
                        return Ok(RowStreamRowResult::Skip);
                    }
                }
                CompiledRowStreamStage::DistinctBy {
                    program,
                    seen,
                    #[cfg(feature = "simd-json")]
                    direct,
                } => {
                    #[cfg(feature = "simd-json")]
                    if let (Some(plan), Some(raw_row)) = (direct.as_ref(), row.as_deref()) {
                        if let Some(inserted) = insert_direct_distinct_key(seen, raw_row, plan) {
                            if !inserted {
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
                    let key = vm
                        .execute_val_raw(program, value.clone())
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
                CompiledRowStreamStage::Map {
                    program,
                    #[cfg(feature = "simd-json")]
                    direct,
                } => {
                    #[cfg(feature = "simd-json")]
                    if is_last_stage {
                        if let (Some(plan), Some(raw_row)) = (direct.as_ref(), row.as_deref()) {
                            let mut out = Vec::new();
                            let mut scratch = Vec::new();
                            match write_ndjson_byte_tape_plan_row(
                                &mut out,
                                raw_row,
                                plan,
                                &mut scratch,
                            )? {
                                BytePlanWrite::Done => {
                                    return Ok(RowStreamRowResult::EmitBytes(out));
                                }
                                BytePlanWrite::Fallback => {}
                            }
                        }
                    }

                    let current = ensure_row_stream_value(
                        engine,
                        line_no,
                        &mut row,
                        &mut document,
                        &mut value,
                    )?;
                    value = Some(
                        vm.execute_val_raw(program, current)
                            .map_err(|err| row_eval_error(line_no, err))?,
                    );
                }
            }
        }

        let value = ensure_row_stream_value(engine, line_no, &mut row, &mut document, &mut value)?;
        Ok(RowStreamRowResult::Emit(value))
    }
}

#[cfg(feature = "simd-json")]
fn insert_direct_distinct_key(
    seen: &mut AdaptiveDistinctKeys,
    row: &[u8],
    plan: &NdjsonDirectTapePlan,
) -> Option<bool> {
    const NULL_KEY: &[u8] = b"null";

    let NdjsonDirectTapePlan::RootPath(steps) = plan else {
        return None;
    };
    let key = match raw_json_byte_path_value(row, steps) {
        RawFieldValue::Found(value) => raw_distinct_key_bytes(value)?,
        RawFieldValue::Missing => std::borrow::Cow::Borrowed(NULL_KEY),
        RawFieldValue::Fallback => return None,
    };
    Some(match key {
        std::borrow::Cow::Borrowed(key) => seen.insert_slice(key),
        std::borrow::Cow::Owned(key) => seen.insert(key),
    })
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
        #[cfg(feature = "simd-json")]
        direct: Option<NdjsonDirectTapePlan>,
    },
    Take {
        limit: usize,
        seen: usize,
    },
    Map {
        program: Program,
        #[cfg(feature = "simd-json")]
        direct: Option<NdjsonDirectTapePlan>,
    },
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
                #[cfg(feature = "simd-json")]
                direct: direct_tape_plan_for_expr(expr),
            },
            RowStreamStage::Take(limit) => Self::Take {
                limit: *limit,
                seen: 0,
            },
            RowStreamStage::Map(expr) => Self::Map {
                program: Compiler::compile(expr, "<ndjson-rows-map>"),
                #[cfg(feature = "simd-json")]
                direct: direct_tape_plan_for_expr(expr).filter(tape_plan_can_write_byte_row),
            },
        }
    }
}
