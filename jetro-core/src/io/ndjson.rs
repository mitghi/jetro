use super::ndjson_byte::{
    eval_ndjson_byte_predicate_row, tape_plan_can_write_byte_row, write_ndjson_byte_plan_row,
    write_ndjson_byte_tape_plan_row, write_ndjson_hinted_tape_plan_row, BytePlanWrite,
};
#[cfg(test)]
pub(super) use super::ndjson_direct::{
    direct_byte_plan, direct_writer_plan_kind, NdjsonDirectPlanKind,
};
pub(super) use super::ndjson_direct::{
    direct_tape_plan, direct_tape_predicate, direct_writer_plans, NdjsonDirectBytePlan,
    NdjsonDirectElement, NdjsonDirectItemPredicate, NdjsonDirectPredicate,
    NdjsonDirectProjectionValue, NdjsonDirectStreamMap, NdjsonDirectStreamPlan,
    NdjsonDirectStreamSink, NdjsonDirectTapePlan,
};
use super::ndjson_frame::{frame_payload, FramePayload, NdjsonRowFrame};
use super::ndjson_hint::{
    NdjsonHintAccessPlan, NdjsonHintConfig, NdjsonHintDecision, NdjsonHintState,
};
pub(super) use super::ndjson_row::{collect_row_val, parse_row, row_eval_error, row_parse_error};
use super::ndjson_rows::NdjsonRowsFilePlan;
use super::ndjson_route::{
    ndjson_route_plan, NdjsonExecutionReport, NdjsonExecutionStats, NdjsonRouteExplain,
    NdjsonRoutePlan, NdjsonSourceCaps, NdjsonSourceMode,
};
use super::ndjson_stream_cache::NdjsonConstantStreamCache;
pub(super) use super::ndjson_write::{
    ndjson_writer_with_options, write_json_bytes_line_with_options, write_val_line,
    write_val_line_with_options,
};
use super::stream_exec::{finish_collected_row_stream, CompiledRowStream};
use super::stream_fanout::{
    drive_ndjson_rows_fanout_file, drive_ndjson_rows_fanout_file_with_stats,
};
#[cfg(test)]
use super::stream_plan::RowStreamSourceKind;
use super::stream_plan::{RowStreamDirection, RowStreamPlan};
use super::stream_subquery::{RowStreamSubqueryPlan, STREAM_BINDING};
use super::stream_types::{RowStreamRowResult, RowStreamStats};
use super::{NdjsonSource, RowError};
pub use super::ndjson_driver::NdjsonPerRowDriver;
use crate::compile::compiler::Compiler;
use crate::builtins::registry::{view_object_projection, BuiltinId};
use crate::builtins::BuiltinViewObjectProjection;
use crate::data::context::Env;
use crate::data::value::Val;
use crate::plan::physical::PlanningContext;
use crate::util::is_truthy;
use crate::{EvalError, Jetro, JetroEngine, JetroEngineError, VM};
use memchr::memchr;
use serde_json::Value;
use std::fs::File;
use std::io::{BufRead, Write};
use std::path::Path;
use std::sync::MutexGuard;

pub(super) const DEFAULT_MAX_LINE_LEN: usize = 64 * 1024 * 1024;
const DEFAULT_LINE_BUFFER_CAPACITY: usize = 8192;
pub(super) const DEFAULT_READER_BUFFER_CAPACITY: usize = 1024 * 1024;
pub(super) const DEFAULT_REVERSE_CHUNK_SIZE: usize = 64 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NdjsonWriterPathKind {
    ByteExpr,
    ByteWritableTape,
    Tape,
}

impl std::fmt::Display for NdjsonWriterPathKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::ByteExpr => "byte-expr",
            Self::ByteWritableTape => "byte-writable-tape",
            Self::Tape => "tape",
        })
    }
}

#[cfg(test)]
pub(super) fn direct_writer_path_kind(
    engine: &JetroEngine,
    query: &str,
) -> Option<NdjsonWriterPathKind> {
    ndjson_writer_path_kind(engine, query)
}

pub fn ndjson_writer_path_kind(
    engine: &JetroEngine,
    query: &str,
) -> Option<NdjsonWriterPathKind> {
    let (byte, tape) = direct_writer_plans(engine, query)?;
    if byte.is_some() {
        return Some(NdjsonWriterPathKind::ByteExpr);
    }
    if tape_plan_can_write_byte_row(&tape) {
        return Some(NdjsonWriterPathKind::ByteWritableTape);
    }
    Some(NdjsonWriterPathKind::Tape)
}

/// Configuration for per-row NDJSON execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NdjsonOptions {
    pub max_line_len: usize,
    pub initial_buffer_capacity: usize,
    pub reader_buffer_capacity: usize,
    pub reverse_chunk_size: usize,
    pub parallel_min_bytes: u64,
    pub parallelism: NdjsonParallelism,
    pub row_frame: NdjsonRowFrame,
    pub null_output: NdjsonNullOutput,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NdjsonParallelism {
    Auto,
    Off,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NdjsonNullOutput {
    Skip,
    Emit,
}

impl Default for NdjsonOptions {
    fn default() -> Self {
        Self {
            max_line_len: DEFAULT_MAX_LINE_LEN,
            initial_buffer_capacity: DEFAULT_LINE_BUFFER_CAPACITY,
            reader_buffer_capacity: DEFAULT_READER_BUFFER_CAPACITY,
            reverse_chunk_size: DEFAULT_REVERSE_CHUNK_SIZE,
            parallel_min_bytes: 64 * 1024 * 1024,
            parallelism: NdjsonParallelism::Auto,
            row_frame: NdjsonRowFrame::JsonLine,
            null_output: NdjsonNullOutput::Skip,
        }
    }
}

impl NdjsonOptions {
    pub fn with_max_line_len(mut self, max_line_len: usize) -> Self {
        self.max_line_len = max_line_len;
        self
    }

    pub fn with_initial_buffer_capacity(mut self, capacity: usize) -> Self {
        self.initial_buffer_capacity = capacity;
        self
    }

    pub fn with_reader_buffer_capacity(mut self, capacity: usize) -> Self {
        self.reader_buffer_capacity = capacity;
        self
    }

    pub fn with_reverse_chunk_size(mut self, capacity: usize) -> Self {
        self.reverse_chunk_size = capacity;
        self
    }

    pub fn with_parallel_min_bytes(mut self, bytes: u64) -> Self {
        self.parallel_min_bytes = bytes;
        self
    }

    pub fn with_parallelism(mut self, parallelism: NdjsonParallelism) -> Self {
        self.parallelism = parallelism;
        self
    }

    pub fn with_row_frame(mut self, row_frame: NdjsonRowFrame) -> Self {
        self.row_frame = row_frame;
        self
    }

    pub fn with_null_output(mut self, null_output: NdjsonNullOutput) -> Self {
        self.null_output = null_output;
        self
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NdjsonControl {
    Continue,
    Stop,
}

pub fn for_each_ndjson<R, F>(
    engine: &JetroEngine,
    reader: R,
    query: &str,
    f: F,
) -> Result<usize, JetroEngineError>
where
    R: BufRead,
    F: FnMut(Value),
{
    for_each_ndjson_with_options(engine, reader, query, NdjsonOptions::default(), f)
}

pub fn for_each_ndjson_with_options<R, F>(
    engine: &JetroEngine,
    reader: R,
    query: &str,
    options: NdjsonOptions,
    mut f: F,
) -> Result<usize, JetroEngineError>
where
    R: BufRead,
    F: FnMut(Value),
{
    drive_ndjson(engine, reader, query, options, |value| {
        f(value);
        Ok(NdjsonControl::Continue)
    })
}

pub fn for_each_ndjson_until<R, F>(
    engine: &JetroEngine,
    reader: R,
    query: &str,
    f: F,
) -> Result<usize, JetroEngineError>
where
    R: BufRead,
    F: FnMut(Value) -> Result<NdjsonControl, JetroEngineError>,
{
    for_each_ndjson_until_with_options(engine, reader, query, NdjsonOptions::default(), f)
}

pub fn for_each_ndjson_until_with_options<R, F>(
    engine: &JetroEngine,
    reader: R,
    query: &str,
    options: NdjsonOptions,
    f: F,
) -> Result<usize, JetroEngineError>
where
    R: BufRead,
    F: FnMut(Value) -> Result<NdjsonControl, JetroEngineError>,
{
    drive_ndjson(engine, reader, query, options, f)
}

pub fn for_each_ndjson_source<F>(
    engine: &JetroEngine,
    source: NdjsonSource,
    query: &str,
    f: F,
) -> Result<usize, JetroEngineError>
where
    F: FnMut(Value),
{
    for_each_ndjson_source_with_options(engine, source, query, NdjsonOptions::default(), f)
}

pub fn for_each_ndjson_source_with_options<F>(
    engine: &JetroEngine,
    source: NdjsonSource,
    query: &str,
    options: NdjsonOptions,
    f: F,
) -> Result<usize, JetroEngineError>
where
    F: FnMut(Value),
{
    match source {
        NdjsonSource::File(path) => {
            let file = File::open(path)?;
            for_each_ndjson_with_options(
                engine,
                std::io::BufReader::with_capacity(options.reader_buffer_capacity, file),
                query,
                options,
                f,
            )
        }
        NdjsonSource::Reader(reader) => {
            for_each_ndjson_with_options(engine, reader, query, options, f)
        }
    }
}

pub fn for_each_ndjson_source_until<F>(
    engine: &JetroEngine,
    source: NdjsonSource,
    query: &str,
    f: F,
) -> Result<usize, JetroEngineError>
where
    F: FnMut(Value) -> Result<NdjsonControl, JetroEngineError>,
{
    for_each_ndjson_source_until_with_options(engine, source, query, NdjsonOptions::default(), f)
}

pub fn for_each_ndjson_source_until_with_options<F>(
    engine: &JetroEngine,
    source: NdjsonSource,
    query: &str,
    options: NdjsonOptions,
    f: F,
) -> Result<usize, JetroEngineError>
where
    F: FnMut(Value) -> Result<NdjsonControl, JetroEngineError>,
{
    match source {
        NdjsonSource::File(path) => {
            let file = File::open(path)?;
            for_each_ndjson_until_with_options(
                engine,
                std::io::BufReader::with_capacity(options.reader_buffer_capacity, file),
                query,
                options,
                f,
            )
        }
        NdjsonSource::Reader(reader) => {
            for_each_ndjson_until_with_options(engine, reader, query, options, f)
        }
    }
}

pub fn collect_ndjson<R>(
    engine: &JetroEngine,
    reader: R,
    query: &str,
) -> Result<Vec<Value>, JetroEngineError>
where
    R: BufRead,
{
    collect_ndjson_with_options(engine, reader, query, NdjsonOptions::default())
}

pub fn collect_ndjson_with_options<R>(
    engine: &JetroEngine,
    reader: R,
    query: &str,
    options: NdjsonOptions,
) -> Result<Vec<Value>, JetroEngineError>
where
    R: BufRead,
{
    let mut values = Vec::new();
    for_each_ndjson_with_options(engine, reader, query, options, |value| values.push(value))?;
    Ok(values)
}

pub fn collect_ndjson_file<P>(
    engine: &JetroEngine,
    path: P,
    query: &str,
) -> Result<Vec<Value>, JetroEngineError>
where
    P: AsRef<Path>,
{
    let file = File::open(path)?;
    let options = NdjsonOptions::default();
    collect_ndjson_with_options(
        engine,
        std::io::BufReader::with_capacity(options.reader_buffer_capacity, file),
        query,
        options,
    )
}

pub fn collect_ndjson_file_with_options<P>(
    engine: &JetroEngine,
    path: P,
    query: &str,
    options: NdjsonOptions,
) -> Result<Vec<Value>, JetroEngineError>
where
    P: AsRef<Path>,
{
    let file = File::open(path)?;
    collect_ndjson_with_options(
        engine,
        std::io::BufReader::with_capacity(options.reader_buffer_capacity, file),
        query,
        options,
    )
}

pub fn collect_ndjson_source(
    engine: &JetroEngine,
    source: NdjsonSource,
    query: &str,
) -> Result<Vec<Value>, JetroEngineError> {
    collect_ndjson_source_with_options(engine, source, query, NdjsonOptions::default())
}

pub fn collect_ndjson_source_with_options(
    engine: &JetroEngine,
    source: NdjsonSource,
    query: &str,
    options: NdjsonOptions,
) -> Result<Vec<Value>, JetroEngineError> {
    match source {
        NdjsonSource::File(path) => collect_ndjson_file_with_options(engine, path, query, options),
        NdjsonSource::Reader(reader) => collect_ndjson_with_options(engine, reader, query, options),
    }
}

pub fn collect_ndjson_matches<R>(
    engine: &JetroEngine,
    reader: R,
    predicate: &str,
    limit: usize,
) -> Result<Vec<Value>, JetroEngineError>
where
    R: BufRead,
{
    collect_ndjson_matches_with_options(engine, reader, predicate, limit, NdjsonOptions::default())
}

pub fn collect_ndjson_matches_with_options<R>(
    engine: &JetroEngine,
    reader: R,
    predicate: &str,
    limit: usize,
    options: NdjsonOptions,
) -> Result<Vec<Value>, JetroEngineError>
where
    R: BufRead,
{
    let mut values = Vec::with_capacity(limit);
    drive_ndjson_matches(engine, reader, predicate, limit, options, |value| {
        values.push(Value::from(value));
        Ok(NdjsonControl::Continue)
    })?;
    Ok(values)
}

pub fn collect_ndjson_matches_file<P>(
    engine: &JetroEngine,
    path: P,
    predicate: &str,
    limit: usize,
) -> Result<Vec<Value>, JetroEngineError>
where
    P: AsRef<Path>,
{
    let file = File::open(path)?;
    let options = NdjsonOptions::default();
    collect_ndjson_matches_with_options(
        engine,
        std::io::BufReader::with_capacity(options.reader_buffer_capacity, file),
        predicate,
        limit,
        options,
    )
}

pub fn collect_ndjson_matches_file_with_options<P>(
    engine: &JetroEngine,
    path: P,
    predicate: &str,
    limit: usize,
    options: NdjsonOptions,
) -> Result<Vec<Value>, JetroEngineError>
where
    P: AsRef<Path>,
{
    let file = File::open(path)?;
    collect_ndjson_matches_with_options(
        engine,
        std::io::BufReader::with_capacity(options.reader_buffer_capacity, file),
        predicate,
        limit,
        options,
    )
}

pub fn collect_ndjson_matches_source(
    engine: &JetroEngine,
    source: NdjsonSource,
    predicate: &str,
    limit: usize,
) -> Result<Vec<Value>, JetroEngineError> {
    collect_ndjson_matches_source_with_options(
        engine,
        source,
        predicate,
        limit,
        NdjsonOptions::default(),
    )
}

pub fn collect_ndjson_matches_source_with_options(
    engine: &JetroEngine,
    source: NdjsonSource,
    predicate: &str,
    limit: usize,
    options: NdjsonOptions,
) -> Result<Vec<Value>, JetroEngineError> {
    match source {
        NdjsonSource::File(path) => {
            collect_ndjson_matches_file_with_options(engine, path, predicate, limit, options)
        }
        NdjsonSource::Reader(reader) => {
            collect_ndjson_matches_with_options(engine, reader, predicate, limit, options)
        }
    }
}

pub fn run_ndjson<R, W>(
    engine: &JetroEngine,
    reader: R,
    query: &str,
    writer: W,
) -> Result<usize, JetroEngineError>
where
    R: BufRead,
    W: Write,
{
    run_ndjson_with_options(engine, reader, query, writer, NdjsonOptions::default())
}

pub fn run_ndjson_file<P, W>(
    engine: &JetroEngine,
    path: P,
    query: &str,
    writer: W,
) -> Result<usize, JetroEngineError>
where
    P: AsRef<Path>,
    W: Write,
{
    let options = NdjsonOptions::default();
    run_ndjson_file_with_options(engine, path, query, writer, options)
}

pub fn run_ndjson_file_with_options<P, W>(
    engine: &JetroEngine,
    path: P,
    query: &str,
    writer: W,
    options: NdjsonOptions,
) -> Result<usize, JetroEngineError>
where
    P: AsRef<Path>,
    W: Write,
{
    let path = path.as_ref();
    match ndjson_route_plan(engine, NdjsonSourceMode::File, query, options)? {
        NdjsonRoutePlan::Rows { plan, .. } => {
            return drive_ndjson_rows_file_plan(engine, path, &plan, None, options, writer);
        }
        NdjsonRoutePlan::Unsupported { explain } => {
            return Err(unsupported_ndjson_route_error(&explain));
        }
        NdjsonRoutePlan::RowLocal { .. } => {}
    }

    let file = File::open(path)?;
    run_ndjson_with_options(
        engine,
        std::io::BufReader::with_capacity(options.reader_buffer_capacity, file),
        query,
        writer,
        options,
    )
}

pub fn run_ndjson_file_with_report<P, W>(
    engine: &JetroEngine,
    path: P,
    query: &str,
    writer: W,
) -> Result<NdjsonExecutionReport, JetroEngineError>
where
    P: AsRef<Path>,
    W: Write,
{
    run_ndjson_file_with_report_and_options(engine, path, query, writer, NdjsonOptions::default())
}

pub fn run_ndjson_file_with_report_and_options<P, W>(
    engine: &JetroEngine,
    path: P,
    query: &str,
    writer: W,
    options: NdjsonOptions,
) -> Result<NdjsonExecutionReport, JetroEngineError>
where
    P: AsRef<Path>,
    W: Write,
{
    let path = path.as_ref();
    match ndjson_route_plan(engine, NdjsonSourceMode::File, query, options)? {
        NdjsonRoutePlan::Rows {
            explain,
            plan: NdjsonRowsFilePlan::Stream(plan),
        } => {
            let (_, stats) =
                drive_ndjson_rows_stream_file_with_stats(engine, path, &plan, None, options, writer)?;
            Ok(row_stream_report(explain, stats))
        }
        NdjsonRoutePlan::Rows {
            explain,
            plan: NdjsonRowsFilePlan::Fanout(plan),
        } => {
            let (_, stats) =
                drive_ndjson_rows_fanout_file_with_stats(engine, path, &plan, options, writer)?;
            Ok(row_stream_report(explain, stats))
        }
        NdjsonRoutePlan::Rows {
            explain,
            plan: NdjsonRowsFilePlan::Subquery(plan),
        } => {
            let (_, stats) =
                drive_ndjson_rows_subquery_file_with_stats(engine, path, &plan, options, writer)?;
            Ok(row_stream_report(explain, stats))
        }
        NdjsonRoutePlan::Unsupported { explain } => Err(unsupported_ndjson_route_error(&explain)),
        NdjsonRoutePlan::RowLocal { explain } => {
            let file = File::open(path)?;
            let (_, stats) = drive_ndjson_writer_with_stats(
                engine,
                std::io::BufReader::with_capacity(options.reader_buffer_capacity, file),
                query,
                None,
                options,
                writer,
            )?;
            Ok(NdjsonExecutionReport::new(explain, stats))
        }
    }
}

pub fn run_ndjson_with_options<R, W>(
    engine: &JetroEngine,
    reader: R,
    query: &str,
    writer: W,
    options: NdjsonOptions,
) -> Result<usize, JetroEngineError>
where
    R: BufRead,
    W: Write,
{
    match ndjson_route_plan(engine, NdjsonSourceMode::Reader, query, options)? {
        NdjsonRoutePlan::Rows {
            plan: NdjsonRowsFilePlan::Stream(plan),
            ..
        } => drive_ndjson_rows_stream_reader(engine, reader, &plan, None, options, writer),
        NdjsonRoutePlan::Rows { explain, .. } | NdjsonRoutePlan::Unsupported { explain } => {
            Err(unsupported_ndjson_route_error(&explain))
        }
        NdjsonRoutePlan::RowLocal { .. } => {
            drive_ndjson_writer(engine, reader, query, None, options, writer)
        }
    }
}

pub fn run_ndjson_with_report<R, W>(
    engine: &JetroEngine,
    reader: R,
    query: &str,
    writer: W,
) -> Result<NdjsonExecutionReport, JetroEngineError>
where
    R: BufRead,
    W: Write,
{
    run_ndjson_with_report_and_options(engine, reader, query, writer, NdjsonOptions::default())
}

pub fn run_ndjson_with_report_and_options<R, W>(
    engine: &JetroEngine,
    reader: R,
    query: &str,
    writer: W,
    options: NdjsonOptions,
) -> Result<NdjsonExecutionReport, JetroEngineError>
where
    R: BufRead,
    W: Write,
{
    match ndjson_route_plan(engine, NdjsonSourceMode::Reader, query, options)? {
        NdjsonRoutePlan::Rows {
            explain,
            plan: NdjsonRowsFilePlan::Stream(plan),
        } => {
            let (_, stats) =
                drive_ndjson_rows_stream_reader_with_stats(engine, reader, &plan, None, options, writer)?;
            Ok(row_stream_report(explain, stats))
        }
        NdjsonRoutePlan::Rows { explain, .. } | NdjsonRoutePlan::Unsupported { explain } => {
            Err(unsupported_ndjson_route_error(&explain))
        }
        NdjsonRoutePlan::RowLocal { explain } => {
            let (_, stats) =
                drive_ndjson_writer_with_stats(engine, reader, query, None, options, writer)?;
            Ok(NdjsonExecutionReport::new(explain, stats))
        }
    }
}

pub fn run_ndjson_limit<R, W>(
    engine: &JetroEngine,
    reader: R,
    query: &str,
    limit: usize,
    writer: W,
) -> Result<usize, JetroEngineError>
where
    R: BufRead,
    W: Write,
{
    run_ndjson_limit_with_options(
        engine,
        reader,
        query,
        limit,
        writer,
        NdjsonOptions::default(),
    )
}

pub fn run_ndjson_limit_with_options<R, W>(
    engine: &JetroEngine,
    reader: R,
    query: &str,
    limit: usize,
    writer: W,
    options: NdjsonOptions,
) -> Result<usize, JetroEngineError>
where
    R: BufRead,
    W: Write,
{
    if limit == 0 {
        return Ok(0);
    }

    match ndjson_route_plan(engine, NdjsonSourceMode::Reader, query, options)? {
        NdjsonRoutePlan::Rows {
            plan: NdjsonRowsFilePlan::Stream(plan),
            ..
        } => drive_ndjson_rows_stream_reader(engine, reader, &plan, Some(limit), options, writer),
        NdjsonRoutePlan::Rows { explain, .. } | NdjsonRoutePlan::Unsupported { explain } => {
            Err(unsupported_ndjson_route_error(&explain))
        }
        NdjsonRoutePlan::RowLocal { .. } => {
            drive_ndjson_writer(engine, reader, query, Some(limit), options, writer)
        }
    }
}

pub fn run_ndjson_limit_with_report<R, W>(
    engine: &JetroEngine,
    reader: R,
    query: &str,
    limit: usize,
    writer: W,
) -> Result<NdjsonExecutionReport, JetroEngineError>
where
    R: BufRead,
    W: Write,
{
    run_ndjson_limit_with_report_and_options(
        engine,
        reader,
        query,
        limit,
        writer,
        NdjsonOptions::default(),
    )
}

pub fn run_ndjson_limit_with_report_and_options<R, W>(
    engine: &JetroEngine,
    reader: R,
    query: &str,
    limit: usize,
    writer: W,
    options: NdjsonOptions,
) -> Result<NdjsonExecutionReport, JetroEngineError>
where
    R: BufRead,
    W: Write,
{
    if limit == 0 {
        let route = ndjson_route_plan(engine, NdjsonSourceMode::Reader, query, options)?
            .explain()
            .clone();
        return Ok(NdjsonExecutionReport::emitted_only(route, 0));
    }

    match ndjson_route_plan(engine, NdjsonSourceMode::Reader, query, options)? {
        NdjsonRoutePlan::Rows {
            explain,
            plan: NdjsonRowsFilePlan::Stream(plan),
        } => {
            let (_, stats) = drive_ndjson_rows_stream_reader_with_stats(
                engine,
                reader,
                &plan,
                Some(limit),
                options,
                writer,
            )?;
            Ok(row_stream_report(explain, stats))
        }
        NdjsonRoutePlan::Rows { explain, .. } | NdjsonRoutePlan::Unsupported { explain } => {
            Err(unsupported_ndjson_route_error(&explain))
        }
        NdjsonRoutePlan::RowLocal { explain } => {
            let (_, stats) = drive_ndjson_writer_with_stats(
                engine,
                reader,
                query,
                Some(limit),
                options,
                writer,
            )?;
            Ok(NdjsonExecutionReport::new(explain, stats))
        }
    }
}

pub fn run_ndjson_file_limit<P, W>(
    engine: &JetroEngine,
    path: P,
    query: &str,
    limit: usize,
    writer: W,
) -> Result<usize, JetroEngineError>
where
    P: AsRef<Path>,
    W: Write,
{
    let options = NdjsonOptions::default();
    run_ndjson_file_limit_with_options(engine, path, query, limit, writer, options)
}

pub fn run_ndjson_file_limit_with_options<P, W>(
    engine: &JetroEngine,
    path: P,
    query: &str,
    limit: usize,
    writer: W,
    options: NdjsonOptions,
) -> Result<usize, JetroEngineError>
where
    P: AsRef<Path>,
    W: Write,
{
    if limit == 0 {
        return Ok(0);
    }
    let path = path.as_ref();
    match ndjson_route_plan(engine, NdjsonSourceMode::File, query, options)? {
        NdjsonRoutePlan::Rows { plan, .. } => {
            return drive_ndjson_rows_file_plan(engine, path, &plan, Some(limit), options, writer);
        }
        NdjsonRoutePlan::Unsupported { explain } => {
            return Err(unsupported_ndjson_route_error(&explain));
        }
        NdjsonRoutePlan::RowLocal { .. } => {}
    }

    let file = File::open(path)?;
    run_ndjson_limit_with_options(
        engine,
        std::io::BufReader::with_capacity(options.reader_buffer_capacity, file),
        query,
        limit,
        writer,
        options,
    )
}

pub fn run_ndjson_file_limit_with_report<P, W>(
    engine: &JetroEngine,
    path: P,
    query: &str,
    limit: usize,
    writer: W,
) -> Result<NdjsonExecutionReport, JetroEngineError>
where
    P: AsRef<Path>,
    W: Write,
{
    run_ndjson_file_limit_with_report_and_options(
        engine,
        path,
        query,
        limit,
        writer,
        NdjsonOptions::default(),
    )
}

pub fn run_ndjson_file_limit_with_report_and_options<P, W>(
    engine: &JetroEngine,
    path: P,
    query: &str,
    limit: usize,
    writer: W,
    options: NdjsonOptions,
) -> Result<NdjsonExecutionReport, JetroEngineError>
where
    P: AsRef<Path>,
    W: Write,
{
    if limit == 0 {
        let route = ndjson_route_plan(engine, NdjsonSourceMode::File, query, options)?
            .explain()
            .clone();
        return Ok(NdjsonExecutionReport::emitted_only(route, 0));
    }

    let path = path.as_ref();
    match ndjson_route_plan(engine, NdjsonSourceMode::File, query, options)? {
        NdjsonRoutePlan::Rows {
            explain,
            plan: NdjsonRowsFilePlan::Stream(plan),
        } => {
            let (_, stats) = drive_ndjson_rows_stream_file_with_stats(
                engine,
                path,
                &plan,
                Some(limit),
                options,
                writer,
            )?;
            Ok(row_stream_report(explain, stats))
        }
        NdjsonRoutePlan::Rows {
            explain,
            plan: NdjsonRowsFilePlan::Fanout(plan),
        } => {
            let (_, stats) =
                drive_ndjson_rows_fanout_file_with_stats(engine, path, &plan, options, writer)?;
            Ok(row_stream_report(explain, stats))
        }
        NdjsonRoutePlan::Rows {
            explain,
            plan: NdjsonRowsFilePlan::Subquery(plan),
        } => {
            let (_, stats) =
                drive_ndjson_rows_subquery_file_with_stats(engine, path, &plan, options, writer)?;
            Ok(row_stream_report(explain, stats))
        }
        NdjsonRoutePlan::Unsupported { explain } => Err(unsupported_ndjson_route_error(&explain)),
        NdjsonRoutePlan::RowLocal { explain } => {
            let file = File::open(path)?;
            let (_, stats) = drive_ndjson_writer_with_stats(
                engine,
                std::io::BufReader::with_capacity(options.reader_buffer_capacity, file),
                query,
                Some(limit),
                options,
                writer,
            )?;
            Ok(NdjsonExecutionReport::new(explain, stats))
        }
    }
}

fn unsupported_ndjson_route_error(explain: &NdjsonRouteExplain) -> JetroEngineError {
    let message = explain
        .unsupported_message()
        .unwrap_or_else(|| "unsupported NDJSON route".to_string());
    JetroEngineError::Eval(EvalError(message))
}

fn row_stream_report(explain: NdjsonRouteExplain, stats: RowStreamStats) -> NdjsonExecutionReport {
    NdjsonExecutionReport::new(explain, NdjsonExecutionStats::from(&stats))
}

pub fn run_ndjson_source<W>(
    engine: &JetroEngine,
    source: NdjsonSource,
    query: &str,
    writer: W,
) -> Result<usize, JetroEngineError>
where
    W: Write,
{
    run_ndjson_source_with_options(engine, source, query, writer, NdjsonOptions::default())
}

pub fn run_ndjson_source_with_options<W>(
    engine: &JetroEngine,
    source: NdjsonSource,
    query: &str,
    writer: W,
    options: NdjsonOptions,
) -> Result<usize, JetroEngineError>
where
    W: Write,
{
    match source {
        NdjsonSource::File(path) => {
            run_ndjson_file_with_options(engine, path, query, writer, options)
        }
        NdjsonSource::Reader(reader) => {
            run_ndjson_with_options(engine, reader, query, writer, options)
        }
    }
}

pub fn run_ndjson_source_with_report<W>(
    engine: &JetroEngine,
    source: NdjsonSource,
    query: &str,
    writer: W,
) -> Result<NdjsonExecutionReport, JetroEngineError>
where
    W: Write,
{
    run_ndjson_source_with_report_and_options(engine, source, query, writer, NdjsonOptions::default())
}

pub fn run_ndjson_source_with_report_and_options<W>(
    engine: &JetroEngine,
    source: NdjsonSource,
    query: &str,
    writer: W,
    options: NdjsonOptions,
) -> Result<NdjsonExecutionReport, JetroEngineError>
where
    W: Write,
{
    match source {
        NdjsonSource::File(path) => {
            run_ndjson_file_with_report_and_options(engine, path, query, writer, options)
        }
        NdjsonSource::Reader(reader) => {
            run_ndjson_with_report_and_options(engine, reader, query, writer, options)
        }
    }
}

pub fn run_ndjson_source_limit<W>(
    engine: &JetroEngine,
    source: NdjsonSource,
    query: &str,
    limit: usize,
    writer: W,
) -> Result<usize, JetroEngineError>
where
    W: Write,
{
    run_ndjson_source_limit_with_options(
        engine,
        source,
        query,
        limit,
        writer,
        NdjsonOptions::default(),
    )
}

pub fn run_ndjson_source_limit_with_options<W>(
    engine: &JetroEngine,
    source: NdjsonSource,
    query: &str,
    limit: usize,
    writer: W,
    options: NdjsonOptions,
) -> Result<usize, JetroEngineError>
where
    W: Write,
{
    match source {
        NdjsonSource::File(path) => {
            run_ndjson_file_limit_with_options(engine, path, query, limit, writer, options)
        }
        NdjsonSource::Reader(reader) => {
            run_ndjson_limit_with_options(engine, reader, query, limit, writer, options)
        }
    }
}

pub fn run_ndjson_source_limit_with_report<W>(
    engine: &JetroEngine,
    source: NdjsonSource,
    query: &str,
    limit: usize,
    writer: W,
) -> Result<NdjsonExecutionReport, JetroEngineError>
where
    W: Write,
{
    run_ndjson_source_limit_with_report_and_options(
        engine,
        source,
        query,
        limit,
        writer,
        NdjsonOptions::default(),
    )
}

pub fn run_ndjson_source_limit_with_report_and_options<W>(
    engine: &JetroEngine,
    source: NdjsonSource,
    query: &str,
    limit: usize,
    writer: W,
    options: NdjsonOptions,
) -> Result<NdjsonExecutionReport, JetroEngineError>
where
    W: Write,
{
    match source {
        NdjsonSource::File(path) => {
            run_ndjson_file_limit_with_report_and_options(engine, path, query, limit, writer, options)
        }
        NdjsonSource::Reader(reader) => {
            run_ndjson_limit_with_report_and_options(engine, reader, query, limit, writer, options)
        }
    }
}

pub fn run_ndjson_matches<R, W>(
    engine: &JetroEngine,
    reader: R,
    predicate: &str,
    limit: usize,
    writer: W,
) -> Result<usize, JetroEngineError>
where
    R: BufRead,
    W: Write,
{
    run_ndjson_matches_with_options(
        engine,
        reader,
        predicate,
        limit,
        writer,
        NdjsonOptions::default(),
    )
}

pub fn run_ndjson_matches_with_options<R, W>(
    engine: &JetroEngine,
    reader: R,
    predicate: &str,
    limit: usize,
    writer: W,
    options: NdjsonOptions,
) -> Result<usize, JetroEngineError>
where
    R: BufRead,
    W: Write,
{
    drive_ndjson_matches_writer(engine, reader, predicate, limit, options, writer)
}

pub fn run_ndjson_matches_with_report<R, W>(
    engine: &JetroEngine,
    reader: R,
    predicate: &str,
    limit: usize,
    writer: W,
) -> Result<NdjsonExecutionReport, JetroEngineError>
where
    R: BufRead,
    W: Write,
{
    run_ndjson_matches_with_report_and_options(
        engine,
        reader,
        predicate,
        limit,
        writer,
        NdjsonOptions::default(),
    )
}

pub fn run_ndjson_matches_with_report_and_options<R, W>(
    engine: &JetroEngine,
    reader: R,
    predicate: &str,
    limit: usize,
    writer: W,
    options: NdjsonOptions,
) -> Result<NdjsonExecutionReport, JetroEngineError>
where
    R: BufRead,
    W: Write,
{
    let (_, stats) =
        drive_ndjson_matches_writer_with_stats(engine, reader, predicate, limit, options, writer)?;
    Ok(NdjsonExecutionReport::new(
        NdjsonRouteExplain::matches(NdjsonSourceCaps::reader(options)),
        stats,
    ))
}

pub fn run_ndjson_matches_file<P, W>(
    engine: &JetroEngine,
    path: P,
    predicate: &str,
    limit: usize,
    writer: W,
) -> Result<usize, JetroEngineError>
where
    P: AsRef<Path>,
    W: Write,
{
    let file = File::open(path)?;
    let options = NdjsonOptions::default();
    run_ndjson_matches_with_options(
        engine,
        std::io::BufReader::with_capacity(options.reader_buffer_capacity, file),
        predicate,
        limit,
        writer,
        options,
    )
}

pub fn run_ndjson_matches_file_with_options<P, W>(
    engine: &JetroEngine,
    path: P,
    predicate: &str,
    limit: usize,
    writer: W,
    options: NdjsonOptions,
) -> Result<usize, JetroEngineError>
where
    P: AsRef<Path>,
    W: Write,
{
    let file = File::open(path)?;
    run_ndjson_matches_with_options(
        engine,
        std::io::BufReader::with_capacity(options.reader_buffer_capacity, file),
        predicate,
        limit,
        writer,
        options,
    )
}

pub fn run_ndjson_matches_file_with_report<P, W>(
    engine: &JetroEngine,
    path: P,
    predicate: &str,
    limit: usize,
    writer: W,
) -> Result<NdjsonExecutionReport, JetroEngineError>
where
    P: AsRef<Path>,
    W: Write,
{
    run_ndjson_matches_file_with_report_and_options(
        engine,
        path,
        predicate,
        limit,
        writer,
        NdjsonOptions::default(),
    )
}

pub fn run_ndjson_matches_file_with_report_and_options<P, W>(
    engine: &JetroEngine,
    path: P,
    predicate: &str,
    limit: usize,
    writer: W,
    options: NdjsonOptions,
) -> Result<NdjsonExecutionReport, JetroEngineError>
where
    P: AsRef<Path>,
    W: Write,
{
    let file = File::open(path)?;
    let (_, stats) = drive_ndjson_matches_writer_with_stats(
        engine,
        std::io::BufReader::with_capacity(options.reader_buffer_capacity, file),
        predicate,
        limit,
        options,
        writer,
    )?;
    Ok(NdjsonExecutionReport::new(
        NdjsonRouteExplain::matches(NdjsonSourceCaps::file(options)),
        stats,
    ))
}

pub fn run_ndjson_matches_source<W>(
    engine: &JetroEngine,
    source: NdjsonSource,
    predicate: &str,
    limit: usize,
    writer: W,
) -> Result<usize, JetroEngineError>
where
    W: Write,
{
    run_ndjson_matches_source_with_options(
        engine,
        source,
        predicate,
        limit,
        writer,
        NdjsonOptions::default(),
    )
}

pub fn run_ndjson_matches_source_with_report<W>(
    engine: &JetroEngine,
    source: NdjsonSource,
    predicate: &str,
    limit: usize,
    writer: W,
) -> Result<NdjsonExecutionReport, JetroEngineError>
where
    W: Write,
{
    run_ndjson_matches_source_with_report_and_options(
        engine,
        source,
        predicate,
        limit,
        writer,
        NdjsonOptions::default(),
    )
}

pub fn run_ndjson_matches_source_with_report_and_options<W>(
    engine: &JetroEngine,
    source: NdjsonSource,
    predicate: &str,
    limit: usize,
    writer: W,
    options: NdjsonOptions,
) -> Result<NdjsonExecutionReport, JetroEngineError>
where
    W: Write,
{
    match source {
        NdjsonSource::File(path) => run_ndjson_matches_file_with_report_and_options(
            engine, path, predicate, limit, writer, options,
        ),
        NdjsonSource::Reader(reader) => run_ndjson_matches_with_report_and_options(
            engine, reader, predicate, limit, writer, options,
        ),
    }
}

pub fn run_ndjson_matches_source_with_options<W>(
    engine: &JetroEngine,
    source: NdjsonSource,
    predicate: &str,
    limit: usize,
    writer: W,
    options: NdjsonOptions,
) -> Result<usize, JetroEngineError>
where
    W: Write,
{
    match source {
        NdjsonSource::File(path) => {
            run_ndjson_matches_file_with_options(engine, path, predicate, limit, writer, options)
        }
        NdjsonSource::Reader(reader) => {
            run_ndjson_matches_with_options(engine, reader, predicate, limit, writer, options)
        }
    }
}

fn drive_ndjson<R, F>(
    engine: &JetroEngine,
    reader: R,
    query: &str,
    options: NdjsonOptions,
    mut emit: F,
) -> Result<usize, JetroEngineError>
where
    R: BufRead,
    F: FnMut(Value) -> Result<NdjsonControl, JetroEngineError>,
{
    let mut driver = NdjsonPerRowDriver::new(reader).with_options(options);
    let plan = engine.cached_plan(query, PlanningContext::bytes());
    let mut buf = Vec::with_capacity(options.initial_buffer_capacity);
    let mut count = 0;

    while let Some((line_no, row)) = driver.read_next_owned(&mut buf)? {
        let document = parse_row(engine, line_no, row)?;
        let out = collect_row_val(engine, &document, &plan, line_no)?;
        count += 1;
        if matches!(emit(Value::from(out))?, NdjsonControl::Stop) {
            break;
        }
    }

    Ok(count)
}

fn drive_ndjson_rows_file_plan<W>(
    engine: &JetroEngine,
    path: &Path,
    plan: &NdjsonRowsFilePlan,
    limit: Option<usize>,
    options: NdjsonOptions,
    writer: W,
) -> Result<usize, JetroEngineError>
where
    W: Write,
{
    match plan {
        NdjsonRowsFilePlan::Stream(plan) => {
            drive_ndjson_rows_stream_file(engine, path, plan, limit, options, writer)
        }
        NdjsonRowsFilePlan::Fanout(plan) => {
            drive_ndjson_rows_fanout_file(engine, path, plan, options, writer)
        }
        NdjsonRowsFilePlan::Subquery(plan) => {
            drive_ndjson_rows_subquery_file(engine, path, plan, options, writer)
        }
    }
}

fn drive_ndjson_rows_subquery_file<P, W>(
    engine: &JetroEngine,
    path: P,
    plan: &RowStreamSubqueryPlan,
    options: NdjsonOptions,
    writer: W,
) -> Result<usize, JetroEngineError>
where
    P: AsRef<Path>,
    W: Write,
{
    let (stream_value, _) =
        collect_ndjson_rows_stream_file_with_stats(engine, path, &plan.stream, options)?;
    let wrapper = Compiler::compile(&plan.wrapper, "<ndjson-rows-wrapper>");
    let env = Env::new(Val::Null).with_var(STREAM_BINDING, stream_value);
    let value = engine
        .lock_vm()
        .exec_in_env(&wrapper, &env)
        .map_err(JetroEngineError::Eval)?;
    let mut writer = ndjson_writer_with_options(writer, options);
    let emitted = write_val_line_with_options(&mut writer, &value, options)? as usize;
    writer.flush()?;
    Ok(emitted)
}

fn drive_ndjson_rows_subquery_file_with_stats<P, W>(
    engine: &JetroEngine,
    path: P,
    plan: &RowStreamSubqueryPlan,
    options: NdjsonOptions,
    writer: W,
) -> Result<(usize, RowStreamStats), JetroEngineError>
where
    P: AsRef<Path>,
    W: Write,
{
    let (stream_value, mut stats) =
        collect_ndjson_rows_stream_file_with_stats(engine, path, &plan.stream, options)?;
    let wrapper = Compiler::compile(&plan.wrapper, "<ndjson-rows-wrapper>");
    let env = Env::new(Val::Null).with_var(STREAM_BINDING, stream_value);
    let value = engine
        .lock_vm()
        .exec_in_env(&wrapper, &env)
        .map_err(JetroEngineError::Eval)?;
    let mut writer = ndjson_writer_with_options(writer, options);
    let emitted = write_val_line_with_options(&mut writer, &value, options)? as usize;
    stats.rows_emitted = emitted;
    writer.flush()?;
    Ok((emitted, stats))
}

fn collect_ndjson_rows_stream_file_with_stats<P>(
    engine: &JetroEngine,
    path: P,
    plan: &RowStreamPlan,
    options: NdjsonOptions,
) -> Result<(Val, RowStreamStats), JetroEngineError>
where
    P: AsRef<Path>,
{
    if let Some(result) =
        super::ndjson_parallel::collect_rows_stream_file_with_stats(engine, path.as_ref(), plan, options)?
    {
        return Ok((result.value, result.stats));
    }

    let mut executor = CompiledRowStream::new(plan);
    let mut out = Vec::new();

    if plan.direction == RowStreamDirection::Forward {
        let file = File::open(path)?;
        let mut driver = NdjsonPerRowDriver::new(std::io::BufReader::with_capacity(
            options.reader_buffer_capacity,
            file,
        ))
        .with_options(options);
        let mut buf = Vec::with_capacity(options.initial_buffer_capacity);
        while !executor.is_exhausted() {
            let Some((line_no, row)) = driver.read_next_owned(&mut buf)? else {
                break;
            };
            if collect_row_stream_result(
                engine,
                line_no,
                executor.apply_owned_row(engine, line_no, row)?,
                &mut out,
            )? || executor.is_exhausted()
            {
                break;
            }
        }
    } else {
        let mut driver = super::ndjson_rev::NdjsonReverseFileDriver::with_options(path, options)?;
        while !executor.is_exhausted() {
            let Some((line_no, row)) = driver.next_line_with_reverse_no()? else {
                break;
            };
            if collect_row_stream_result(
                engine,
                line_no,
                executor.apply_owned_row(engine, line_no, row)?,
                &mut out,
            )? || executor.is_exhausted()
            {
                break;
            }
        }
    }

    finish_collected_row_stream(plan, &executor, out).map_err(JetroEngineError::Eval)
}

pub(super) fn collect_row_stream_result(
    engine: &JetroEngine,
    line_no: u64,
    result: RowStreamRowResult,
    out: &mut Vec<Val>,
) -> Result<bool, JetroEngineError> {
    match result {
        RowStreamRowResult::Emit(value) => out.push(value),
        RowStreamRowResult::EmitBytes(bytes) => {
            let document = parse_row(engine, line_no, bytes)?;
            let value = document
                .root_val_with(engine.keys())
                .map_err(|err| row_eval_error(line_no, err))?;
            out.push(value);
        }
        RowStreamRowResult::Skip => {}
        RowStreamRowResult::Stop => return Ok(true),
    }
    Ok(false)
}

fn drive_ndjson_rows_stream_reader<R, W>(
    engine: &JetroEngine,
    reader: R,
    plan: &RowStreamPlan,
    external_limit: Option<usize>,
    options: NdjsonOptions,
    writer: W,
) -> Result<usize, JetroEngineError>
where
    R: BufRead,
    W: Write,
{
    let (emitted, _) = drive_ndjson_rows_stream_reader_with_stats(
        engine,
        reader,
        plan,
        external_limit,
        options,
        writer,
    )?;
    Ok(emitted)
}

fn drive_ndjson_rows_stream_reader_with_stats<R, W>(
    engine: &JetroEngine,
    reader: R,
    plan: &RowStreamPlan,
    external_limit: Option<usize>,
    options: NdjsonOptions,
    writer: W,
) -> Result<(usize, RowStreamStats), JetroEngineError>
where
    R: BufRead,
    W: Write,
{
    if plan.direction == RowStreamDirection::Reverse {
        return Err(JetroEngineError::Eval(EvalError(
            "$.rows().reverse() requires a file-backed NDJSON source".into(),
        )));
    }

    let mut driver = NdjsonPerRowDriver::new(reader).with_options(options);
    let mut executor = CompiledRowStream::new(plan);
    let mut writer = ndjson_writer_with_options(writer, options);
    let mut buf = Vec::with_capacity(options.initial_buffer_capacity);
    let mut emitted = 0usize;

    while !executor.is_exhausted() {
        let Some((line_no, row)) = driver.read_next_owned(&mut buf)? else {
            break;
        };
        if emit_row_stream_result(
            executor.apply_owned_row(engine, line_no, row)?,
            &mut writer,
            &mut emitted,
            external_limit,
            options,
        )? {
            break;
        }
        if executor.is_exhausted() {
            break;
        }
    }

    emit_row_stream_finish(
        &executor,
        &mut writer,
        &mut emitted,
        external_limit,
        options,
    )?;
    writer.flush()?;
    Ok((emitted, executor.stats().clone()))
}

fn drive_ndjson_rows_stream_file<P, W>(
    engine: &JetroEngine,
    path: P,
    plan: &RowStreamPlan,
    external_limit: Option<usize>,
    options: NdjsonOptions,
    writer: W,
) -> Result<usize, JetroEngineError>
where
    P: AsRef<Path>,
    W: Write,
{
    if let Some(value) =
        super::ndjson_parallel::collect_rows_stream_file(engine, path.as_ref(), plan, options)?
    {
        return write_collected_rows_stream(value, external_limit, options, writer);
    }

    let (emitted, _) = drive_ndjson_rows_stream_file_with_stats(
        engine,
        path,
        plan,
        external_limit,
        options,
        writer,
    )?;
    Ok(emitted)
}

fn write_collected_rows_stream<W: Write>(
    value: Val,
    external_limit: Option<usize>,
    options: NdjsonOptions,
    writer: W,
) -> Result<usize, JetroEngineError> {
    let mut writer = ndjson_writer_with_options(writer, options);
    let mut emitted = 0usize;
    match value {
        Val::Arr(values) => {
            for value in values.iter() {
                if external_limit.is_some_and(|limit| emitted >= limit) {
                    break;
                }
                if write_val_line_with_options(&mut writer, value, options)? {
                    emitted += 1;
                }
            }
        }
        value => {
            if write_val_line_with_options(&mut writer, &value, options)? {
                emitted += 1;
            }
        }
    }
    writer.flush()?;
    Ok(emitted)
}

fn drive_ndjson_rows_stream_file_with_stats<P, W>(
    engine: &JetroEngine,
    path: P,
    plan: &RowStreamPlan,
    external_limit: Option<usize>,
    options: NdjsonOptions,
    writer: W,
) -> Result<(usize, RowStreamStats), JetroEngineError>
where
    P: AsRef<Path>,
    W: Write,
{
    if let Some(result) = super::ndjson_parallel::collect_rows_stream_file_with_stats(
        engine,
        path.as_ref(),
        plan,
        options,
    )? {
        let mut stats = result.stats;
        let emitted = write_collected_rows_stream(result.value, external_limit, options, writer)?;
        stats.rows_emitted = emitted;
        return Ok((emitted, stats));
    }

    if plan.direction == RowStreamDirection::Forward {
        let file = File::open(path)?;
        return drive_ndjson_rows_stream_reader_with_stats(
            engine,
            std::io::BufReader::with_capacity(options.reader_buffer_capacity, file),
            plan,
            external_limit,
            options,
            writer,
        );
    }

    let mut driver = super::ndjson_rev::NdjsonReverseFileDriver::with_options(path, options)?;
    let mut executor = CompiledRowStream::new(plan);
    let mut writer = ndjson_writer_with_options(writer, options);
    let mut emitted = 0usize;

    while !executor.is_exhausted() {
        let Some((line_no, row)) = driver.next_line_with_reverse_no()? else {
            break;
        };
        if emit_row_stream_result(
            executor.apply_owned_row(engine, line_no, row)?,
            &mut writer,
            &mut emitted,
            external_limit,
            options,
        )? {
            break;
        }
        if executor.is_exhausted() {
            break;
        }
    }

    emit_row_stream_finish(
        &executor,
        &mut writer,
        &mut emitted,
        external_limit,
        options,
    )?;
    writer.flush()?;
    Ok((emitted, executor.stats().clone()))
}

fn emit_row_stream_finish<W: Write>(
    executor: &CompiledRowStream,
    writer: &mut W,
    emitted: &mut usize,
    external_limit: Option<usize>,
    options: NdjsonOptions,
) -> Result<(), JetroEngineError> {
    if external_limit.is_some_and(|limit| *emitted >= limit) {
        return Ok(());
    }
    if let Some(value) = executor.finish_result().map_err(JetroEngineError::Eval)? {
        if write_val_line_with_options(writer, &value, options)? {
            *emitted += 1;
        }
    }
    Ok(())
}

fn emit_row_stream_result<W: Write>(
    result: RowStreamRowResult,
    writer: &mut W,
    emitted: &mut usize,
    external_limit: Option<usize>,
    options: NdjsonOptions,
) -> Result<bool, JetroEngineError> {
    let wrote = match result {
        RowStreamRowResult::Emit(value) => write_val_line_with_options(writer, &value, options)?,
        RowStreamRowResult::EmitBytes(bytes) => {
            write_json_bytes_line_with_options(writer, &bytes, options)?
        }
        RowStreamRowResult::Skip => return Ok(false),
        RowStreamRowResult::Stop => return Ok(true),
    };
    if wrote {
        *emitted += 1;
    }
    Ok(external_limit.is_some_and(|limit| *emitted >= limit))
}

fn drive_ndjson_writer<R, W>(
    engine: &JetroEngine,
    reader: R,
    query: &str,
    limit: Option<usize>,
    options: NdjsonOptions,
    writer: W,
) -> Result<usize, JetroEngineError>
where
    R: BufRead,
    W: Write,
{
    Ok(drive_ndjson_writer_with_stats(engine, reader, query, limit, options, writer)?.0)
}

fn drive_ndjson_writer_with_stats<R, W>(
    engine: &JetroEngine,
    reader: R,
    query: &str,
    limit: Option<usize>,
    options: NdjsonOptions,
    writer: W,
) -> Result<(usize, NdjsonExecutionStats), JetroEngineError>
where
    R: BufRead,
    W: Write,
{
    if let Some((byte_plan, tape_plan)) = direct_writer_plans(engine, query) {
        if let Some(byte_plan) = byte_plan {
            return drive_ndjson_byte_writer(
                engine, reader, &byte_plan, &tape_plan, limit, options, writer,
            );
        }
        if tape_plan_can_write_byte_row(&tape_plan) {
            return drive_ndjson_tape_byte_writer(
                engine, reader, &tape_plan, limit, options, writer,
            );
        }
        return drive_ndjson_tape_writer(engine, reader, &tape_plan, limit, options, writer);
    }

    let mut driver = NdjsonPerRowDriver::new(reader).with_options(options);
    let mut executor = NdjsonRowExecutor::new(engine, query);
    let mut writer = ndjson_writer_with_options(writer, options);
    let mut buf = Vec::with_capacity(options.initial_buffer_capacity);
    let mut count = 0usize;
    let mut stats = NdjsonExecutionStats::default();

    while let Some((line_no, row)) = driver.read_next_owned(&mut buf)? {
        stats.rows_scanned += 1;
        let value = executor.eval_owned_row(line_no, row)?;
        stats.fallback_project_rows += 1;
        if write_val_line_with_options(&mut writer, &value, options)? {
            count += 1;
            stats.rows_emitted += 1;
        }
        if limit.is_some_and(|limit| count >= limit) {
            break;
        }
    }

    writer.flush()?;
    Ok((count, stats))
}
fn drive_ndjson_byte_writer<R, W>(
    engine: &JetroEngine,
    reader: R,
    byte_plan: &NdjsonDirectBytePlan,
    tape_plan: &NdjsonDirectTapePlan,
    limit: Option<usize>,
    options: NdjsonOptions,
    writer: W,
) -> Result<(usize, NdjsonExecutionStats), JetroEngineError>
where
    R: BufRead,
    W: Write,
{
    let mut driver = NdjsonPerRowDriver::new(reader).with_options(options);
    let mut writer = ndjson_writer_with_options(writer, options);
    let mut line = Vec::with_capacity(options.initial_buffer_capacity);
    let mut out = Vec::with_capacity(options.initial_buffer_capacity);
    let mut scratch =
        crate::data::tape::TapeScratch::with_capacity(options.initial_buffer_capacity);
    let mut tape_runner = NdjsonTapeWriterRunner::new(engine, tape_plan);
    let mut count = 0usize;
    let mut stats = NdjsonExecutionStats::default();

    visit_ndjson_borrowed_rows(&mut driver, &mut line, |line_no, row| {
        stats.rows_scanned += 1;
        out.clear();
        match write_ndjson_byte_plan_row(&mut out, row, byte_plan)? {
            BytePlanWrite::Done => {
                stats.direct_project_rows += 1;
            }
            BytePlanWrite::Fallback => {
                stats.fallback_project_rows += 1;
                scratch.parse_slice(row).map_err(|message| {
                    row_parse_error(
                        line_no,
                        JetroEngineError::Eval(crate::EvalError(format!(
                            "Invalid JSON: {message}"
                        ))),
                    )
                })?;
                tape_runner.write_row(&scratch, &mut out)?;
            }
        }
        if write_json_bytes_line_with_options(&mut writer, &out, options)? {
            count += 1;
            stats.rows_emitted += 1;
        }
        Ok(!limit.is_some_and(|limit| count >= limit))
    })?;

    writer.flush()?;
    Ok((count, stats))
}
fn drive_ndjson_tape_byte_writer<R, W>(
    engine: &JetroEngine,
    reader: R,
    tape_plan: &NdjsonDirectTapePlan,
    limit: Option<usize>,
    options: NdjsonOptions,
    writer: W,
) -> Result<(usize, NdjsonExecutionStats), JetroEngineError>
where
    R: BufRead,
    W: Write,
{
    let mut driver = NdjsonPerRowDriver::new(reader).with_options(options);
    let mut writer = ndjson_writer_with_options(writer, options);
    let mut line = Vec::with_capacity(options.initial_buffer_capacity);
    let mut out = Vec::with_capacity(options.initial_buffer_capacity);
    let mut scratch =
        crate::data::tape::TapeScratch::with_capacity(options.initial_buffer_capacity);
    let mut byte_scratch = Vec::with_capacity(options.initial_buffer_capacity);
    let mut tape_runner = NdjsonTapeWriterRunner::new(engine, tape_plan);
    let mut constant_stream_cache = NdjsonConstantStreamCache::default();
    let mut hint_state = matches!(
        tape_plan,
        NdjsonDirectTapePlan::Object(_)
            | NdjsonDirectTapePlan::Array(_)
            | NdjsonDirectTapePlan::Stream(NdjsonDirectStreamPlan {
                sink: NdjsonDirectStreamSink::Collect(_),
                ..
            })
            | NdjsonDirectTapePlan::Stream(NdjsonDirectStreamPlan {
                sink: NdjsonDirectStreamSink::First(_),
                ..
            })
            | NdjsonDirectTapePlan::Stream(NdjsonDirectStreamPlan {
                sink: NdjsonDirectStreamSink::Last(_),
                ..
            })
            | NdjsonDirectTapePlan::Stream(NdjsonDirectStreamPlan {
                sink: NdjsonDirectStreamSink::Count,
                ..
            })
            | NdjsonDirectTapePlan::Stream(NdjsonDirectStreamPlan {
                sink: NdjsonDirectStreamSink::Numeric { .. },
                ..
            })
            | NdjsonDirectTapePlan::Stream(NdjsonDirectStreamPlan {
                sink: NdjsonDirectStreamSink::Extreme { .. },
                ..
            })
    )
    .then(|| {
        NdjsonHintState::new(
            NdjsonHintConfig::default(),
            NdjsonHintAccessPlan::from_direct_plans(None, tape_plan),
        )
    });
    let mut count = 0usize;
    let mut stats = NdjsonExecutionStats::default();

    visit_ndjson_borrowed_rows(&mut driver, &mut line, |line_no, row| {
        stats.rows_scanned += 1;
        out.clear();
        if let Some(write) = constant_stream_cache.write_row(&mut out, row, tape_plan)? {
            if matches!(write, BytePlanWrite::Done) {
                stats.direct_project_rows += 1;
                if write_json_bytes_line_with_options(&mut writer, &out, options)? {
                    count += 1;
                    stats.rows_emitted += 1;
                }
                return Ok(!limit.is_some_and(|limit| count >= limit));
            }
        }
        let hinted = if let Some(state) = hint_state.as_mut() {
            if state.observe_row(row) == NdjsonHintDecision::UseHints {
                byte_scratch.clear();
                let write = state
                    .with_root_layout_match(row, |root, matched| {
                        write_ndjson_hinted_tape_plan_row(
                            &mut byte_scratch,
                            tape_plan,
                            root,
                            matched,
                        )
                    })
                    .transpose()?
                    .unwrap_or(BytePlanWrite::Fallback);
                if matches!(write, BytePlanWrite::Done) {
                    out.extend_from_slice(&byte_scratch);
                }
                Some(write)
            } else {
                None
            }
        } else {
            None
        };
        let write = match hinted {
            Some(write) => Ok(write),
            None => write_ndjson_byte_tape_plan_row(&mut out, row, tape_plan, &mut byte_scratch),
        };
        match write? {
            BytePlanWrite::Done => {
                stats.direct_project_rows += 1;
            }
            BytePlanWrite::Fallback => {
                stats.fallback_project_rows += 1;
                scratch.parse_slice(row).map_err(|message| {
                    row_parse_error(
                        line_no,
                        JetroEngineError::Eval(crate::EvalError(format!(
                            "Invalid JSON: {message}"
                        ))),
                    )
                })?;
                tape_runner.write_row(&scratch, &mut out)?;
            }
        }
        if write_json_bytes_line_with_options(&mut writer, &out, options)? {
            count += 1;
            stats.rows_emitted += 1;
        }
        Ok(!limit.is_some_and(|limit| count >= limit))
    })?;

    if let Some(state) = hint_state.as_ref() {
        let hint_stats = state.stats();
        stats.hint_learned_rows = hint_stats.learned_rows;
        stats.hint_rejected_rows = hint_stats.rejected_rows;
        stats.hint_rows = hint_stats.hinted_rows;
        stats.hint_layout_misses = hint_stats.layout_misses;
        stats.hint_disabled = hint_stats.disabled;
    }

    writer.flush()?;
    Ok((count, stats))
}
fn visit_ndjson_borrowed_rows<R, F>(
    driver: &mut NdjsonPerRowDriver<R>,
    spill: &mut Vec<u8>,
    mut visit: F,
) -> Result<(), JetroEngineError>
where
    R: BufRead,
    F: FnMut(u64, &[u8]) -> Result<bool, JetroEngineError>,
{
    loop {
        spill.clear();
        let available = driver.reader.fill_buf()?;
        if available.is_empty() {
            return Ok(());
        }
        if let Some(pos) = memchr(b'\n', available) {
            driver.line_no += 1;
            let line_no = driver.line_no;
            let mut row = &available[..pos];
            if row.last() == Some(&b'\r') {
                row = &row[..row.len() - 1];
            }
            if line_no == 1 && row.starts_with(&[0xef, 0xbb, 0xbf]) {
                row = &row[3..];
            }
            let (start, end) = non_ws_range(row);
            let keep_going = if start == end {
                true
            } else {
                let trimmed = &row[start..end];
                if trimmed.len() > driver.max_line_len {
                    return Err(RowError::LineTooLarge {
                        line_no,
                        len: trimmed.len(),
                        max: driver.max_line_len,
                    }
                    .into());
                }
                match frame_payload(driver.row_frame, line_no, trimmed)? {
                    FramePayload::Data(range) => visit(line_no, &trimmed[range])?,
                    FramePayload::Skip => true,
                }
            };
            driver.reader.consume(pos + 1);
            if !keep_going {
                return Ok(());
            }
        } else {
            let read = driver.read_physical_line(spill)?;
            if read == 0 {
                return Ok(());
            }
            driver.line_no += 1;
            strip_initial_bom(driver.line_no, spill);
            trim_line_ending(spill);
            let (start, end) = non_ws_range(spill);
            if start == end {
                continue;
            }
            let len = end - start;
            if len > driver.max_line_len {
                return Err(RowError::LineTooLarge {
                    line_no: driver.line_no,
                    len,
                    max: driver.max_line_len,
                }
                .into());
            }
            match frame_payload(driver.row_frame, driver.line_no, &spill[start..end])? {
                FramePayload::Data(range) => {
                    if !visit(
                        driver.line_no,
                        &spill[start + range.start..start + range.end],
                    )? {
                        return Ok(());
                    }
                }
                FramePayload::Skip => {}
            }
        }
    }
}
fn drive_ndjson_tape_writer<R, W>(
    engine: &JetroEngine,
    reader: R,
    plan: &NdjsonDirectTapePlan,
    limit: Option<usize>,
    options: NdjsonOptions,
    writer: W,
) -> Result<(usize, NdjsonExecutionStats), JetroEngineError>
where
    R: BufRead,
    W: Write,
{
    let mut driver = NdjsonPerRowDriver::new(reader).with_options(options);
    let mut writer = ndjson_writer_with_options(writer, options);
    let mut line = Vec::with_capacity(options.initial_buffer_capacity);
    let mut scratch =
        crate::data::tape::TapeScratch::with_capacity(options.initial_buffer_capacity);
    let mut out = Vec::with_capacity(options.initial_buffer_capacity);
    let mut count = 0usize;
    let mut runner = NdjsonTapeWriterRunner::new(engine, plan);
    let mut stats = NdjsonExecutionStats::default();

    while let Some((line_no, row)) = driver.read_next_nonempty(&mut line)? {
        stats.rows_scanned += 1;
        out.clear();
        scratch.parse_slice(row).map_err(|message| {
            row_parse_error(
                line_no,
                JetroEngineError::Eval(crate::EvalError(format!("Invalid JSON: {message}"))),
            )
        })?;
        runner.write_row(&scratch, &mut out)?;
        stats.fallback_project_rows += 1;
        if write_json_bytes_line_with_options(&mut writer, &out, options)? {
            count += 1;
            stats.rows_emitted += 1;
        }
        if limit.is_some_and(|limit| count >= limit) {
            break;
        }
    }

    writer.flush()?;
    Ok((count, stats))
}
pub(super) struct NdjsonTapeWriterRunner<'a, 'p> {
    plan: &'p NdjsonDirectTapePlan,
    vm: Option<MutexGuard<'a, VM>>,
    env: Option<crate::data::context::Env>,
    root_path: NdjsonPathCache,
    source_path: NdjsonPathCache,
    suffix_path: NdjsonPathCache,
    predicate_path: NdjsonPathCache,
    object_paths: Vec<NdjsonPathCache>,
}
impl<'a, 'p> NdjsonTapeWriterRunner<'a, 'p> {
    pub(super) fn new(engine: &'a JetroEngine, plan: &'p NdjsonDirectTapePlan) -> Self {
        let needs_vm = plan.needs_vm();
        Self {
            plan,
            vm: needs_vm.then(|| engine.lock_vm()),
            env: needs_vm.then(|| crate::data::context::Env::new(Val::Null)),
            root_path: NdjsonPathCache::default(),
            source_path: NdjsonPathCache::default(),
            suffix_path: NdjsonPathCache::default(),
            predicate_path: NdjsonPathCache::default(),
            object_paths: Vec::new(),
        }
    }

    pub(super) fn write_row<W: Write>(
        &mut self,
        scratch: &crate::data::tape::TapeScratch,
        writer: &mut W,
    ) -> Result<(), JetroEngineError> {
        match self.plan {
            NdjsonDirectTapePlan::RootPath(steps) => {
                if let Some(idx) = self.root_path.index(scratch, 0, steps) {
                    write_json_tape_at(writer, scratch, idx)?;
                } else {
                    writer.write_all(b"null")?;
                }
            }
            NdjsonDirectTapePlan::ViewScalarCall {
                steps,
                call,
                optional,
            } => {
                let idx = self.root_path.index(scratch, 0, steps);
                let value = idx
                    .map(|idx| json_tape_scalar(scratch, idx))
                    .unwrap_or(crate::util::JsonView::Null);
                if *optional && matches!(value, crate::util::JsonView::Null) {
                    writer.write_all(b"null")?;
                } else if let Some(value) = call.try_apply_json_view(value) {
                    write_val_json(writer, &value)?;
                } else if let Some(idx) = idx {
                    write_json_tape_at(writer, scratch, idx)?;
                } else {
                    writer.write_all(b"null")?;
                }
            }
            NdjsonDirectTapePlan::ArrayElementViewScalarCall {
                source_steps,
                element,
                suffix_steps,
                call,
            } => {
                let idx = self
                    .source_path
                    .index(scratch, 0, source_steps)
                    .and_then(|idx| json_tape_array_element(scratch, idx, *element))
                    .and_then(|idx| self.suffix_path.index(scratch, idx, suffix_steps));
                if let Some(value) = idx
                    .map(|idx| json_tape_scalar(scratch, idx))
                    .and_then(|value| call.try_apply_json_view(value))
                {
                    write_val_json(writer, &value)?;
                } else if let Some(idx) = idx {
                    write_json_tape_at(writer, scratch, idx)?;
                } else {
                    writer.write_all(b"null")?;
                }
            }
            NdjsonDirectTapePlan::ObjectItems { steps, method } => {
                let idx = self.root_path.index(scratch, 0, steps);
                write_json_tape_object_items(writer, scratch, idx, *method)?;
            }
            NdjsonDirectTapePlan::ArrayElementPath {
                source_steps,
                element,
                suffix_steps,
            } => {
                let idx = self
                    .source_path
                    .index(scratch, 0, source_steps)
                    .and_then(|idx| json_tape_array_element(scratch, idx, *element))
                    .and_then(|idx| self.suffix_path.index(scratch, idx, suffix_steps));
                if let Some(idx) = idx {
                    write_json_tape_at(writer, scratch, idx)?;
                } else {
                    writer.write_all(b"null")?;
                }
            }
            NdjsonDirectTapePlan::Stream(plan) => {
                write_json_tape_stream(
                    writer,
                    scratch,
                    plan,
                    &mut self.source_path,
                    &mut self.suffix_path,
                    &mut self.predicate_path,
                    &mut self.object_paths,
                )?;
            }
            NdjsonDirectTapePlan::Object(fields) => {
                write_json_tape_object_projection(writer, scratch, fields, &mut self.object_paths)?;
            }
            NdjsonDirectTapePlan::Array(items) => {
                write_json_tape_array_projection(writer, scratch, items, &mut self.object_paths)?;
            }
            NdjsonDirectTapePlan::ViewPipeline { source_steps, body } => {
                let (Some(vm), Some(env)) = (self.vm.as_deref_mut(), self.env.as_ref()) else {
                    return Err(JetroEngineError::Eval(crate::EvalError(
                        "NDJSON view pipeline requires VM state".to_string(),
                    )));
                };
                let source = json_tape_path_index(scratch, source_steps)
                    .map(|idx| crate::data::view::TapeScratchView::Node { tape: scratch, idx })
                    .unwrap_or(crate::data::view::TapeScratchView::Missing);
                let Some(result) =
                    crate::exec::view::run_with_env_and_vm(source, body, None, &env, vm)
                else {
                    writer.write_all(b"null")?;
                    return Ok(());
                };
                write_val_json(writer, &result.map_err(JetroEngineError::Eval)?)?;
            }
        }
        Ok(())
    }
}
#[derive(Default)]
pub(super) struct NdjsonPathCache {
    // Per-depth verified field deltas for stable object layouts. Every use
    // checks the cached key before returning the value node, so mixed-schema
    // rows safely fall back to a normal object scan.
    fields: Vec<Option<NdjsonFieldCache>>,
}
#[derive(Clone, Copy)]
struct NdjsonFieldCache {
    key_delta: usize,
    value_delta: usize,
}
struct NdjsonPathCaches<'a> {
    source: &'a mut NdjsonPathCache,
    suffix: &'a mut NdjsonPathCache,
    predicate: &'a mut NdjsonPathCache,
}
impl NdjsonPathCache {
    fn index<T: JsonTape>(
        &mut self,
        tape: &T,
        start: usize,
        steps: &[crate::ir::physical::PhysicalPathStep],
    ) -> Option<usize> {
        if let Some(idx) = self.index_cached(tape, start, steps) {
            return Some(idx);
        }
        self.index_uncached(tape, start, steps)
    }

    fn index_cached<T: JsonTape>(
        &self,
        tape: &T,
        start: usize,
        steps: &[crate::ir::physical::PhysicalPathStep],
    ) -> Option<usize> {
        use crate::ir::physical::PhysicalPathStep;

        let [PhysicalPathStep::Field(key), rest @ ..] = steps else {
            return None;
        };
        if rest
            .iter()
            .any(|step| matches!(step, PhysicalPathStep::Field(_)))
        {
            return None;
        }
        let Some(field) = self
            .fields
            .first()
            .copied()
            .flatten()
            .filter(|field| field.key_delta > 1)
        else {
            return None;
        };
        let idx = json_tape_object_cached_field(tape, start, field, key.as_ref())?;
        let mut cur = idx;
        for step in rest {
            cur = json_tape_step_index(tape, cur, step)?;
        }
        Some(cur)
    }

    fn index_uncached<T: JsonTape>(
        &mut self,
        tape: &T,
        start: usize,
        steps: &[crate::ir::physical::PhysicalPathStep],
    ) -> Option<usize> {
        self.index_from_depth(tape, start, steps, 0)
    }

    fn index_from_depth<T: JsonTape>(
        &mut self,
        tape: &T,
        start: usize,
        steps: &[crate::ir::physical::PhysicalPathStep],
        depth: usize,
    ) -> Option<usize> {
        use crate::ir::physical::PhysicalPathStep;

        match steps {
            [] => Some(start),
            [PhysicalPathStep::Field(key), rest @ ..] => {
                if self.fields.len() <= depth {
                    self.fields.resize(depth + 1, None);
                }

                if let Some(field) = self.fields[depth].filter(|field| field.key_delta > 1) {
                    if let Some(idx) =
                        json_tape_object_cached_field(tape, start, field, key.as_ref())
                    {
                        return self.index_from_depth(tape, idx, rest, depth + 1);
                    }
                }

                let (idx, field) =
                    json_tape_object_field_index_and_cache(tape, start, key.as_ref())?;
                self.fields[depth] = Some(field);
                self.index_from_depth(tape, idx, rest, depth + 1)
            }
            [step, rest @ ..] => {
                let idx = json_tape_step_index(tape, start, step)?;
                self.index_from_depth(tape, idx, rest, depth + 1)
            }
        }
    }
}
fn drive_ndjson_tape_matches_writer_with_stats<R, W>(
    engine: &JetroEngine,
    reader: R,
    predicate: &NdjsonDirectPredicate,
    limit: usize,
    options: NdjsonOptions,
    writer: W,
) -> Result<(usize, NdjsonExecutionStats), JetroEngineError>
where
    R: BufRead,
    W: Write,
{
    let mut driver = NdjsonPerRowDriver::new(reader).with_options(options);
    let mut writer = ndjson_writer_with_options(writer, options);
    let mut line = Vec::with_capacity(options.initial_buffer_capacity);
    let mut scratch =
        crate::data::tape::TapeScratch::with_capacity(options.initial_buffer_capacity);
    let mut emitted = 0usize;
    let needs_vm = predicate_needs_vm(predicate);
    let mut vm = needs_vm.then(|| engine.lock_vm());
    let env = needs_vm.then(|| crate::data::context::Env::new(Val::Null));
    let mut predicate_path = NdjsonPathCache::default();
    let mut stats = NdjsonExecutionStats::default();

    while let Some((line_no, row)) = driver.read_next_owned(&mut line)? {
        stats.rows_scanned += 1;
        if let Some(matched) = eval_ndjson_byte_predicate_row(&row, predicate)? {
            stats.direct_filter_rows += 1;
            if !matched {
                stats.rows_filtered += 1;
                continue;
            }
            writer.write_all(&row)?;
            writer.write_all(b"\n")?;
            emitted += 1;
            stats.rows_emitted += 1;
            if emitted >= limit {
                break;
            }
            continue;
        }

        scratch.parse_slice(&row).map_err(|message| {
            row_parse_error(
                line_no,
                JetroEngineError::Eval(crate::EvalError(format!("Invalid JSON: {message}"))),
            )
        })?;
        if !eval_tape_predicate(
            &scratch,
            predicate,
            env.as_ref(),
            &mut vm,
            &mut predicate_path,
        )
        .map_err(JetroEngineError::Eval)?
        {
            stats.fallback_filter_rows += 1;
            stats.rows_filtered += 1;
            continue;
        }
        stats.fallback_filter_rows += 1;
        writer.write_all(&row)?;
        writer.write_all(b"\n")?;
        emitted += 1;
        stats.rows_emitted += 1;
        if emitted >= limit {
            break;
        }
    }

    writer.flush()?;
    Ok((emitted, stats))
}

fn drive_ndjson_matches<R, F>(
    engine: &JetroEngine,
    reader: R,
    predicate: &str,
    limit: usize,
    options: NdjsonOptions,
    mut emit: F,
) -> Result<usize, JetroEngineError>
where
    R: BufRead,
    F: FnMut(Val) -> Result<NdjsonControl, JetroEngineError>,
{
    if limit == 0 {
        return Ok(0);
    }

    let mut driver = NdjsonPerRowDriver::new(reader).with_options(options);
    let direct_predicate = direct_tape_predicate(engine, predicate);
    let mut executor = NdjsonRowExecutor::new(engine, predicate);
    let mut buf = Vec::with_capacity(options.initial_buffer_capacity);
    let mut emitted = 0usize;

    while let Some((line_no, row)) = driver.read_next_owned(&mut buf)? {
        if let Some(predicate) = direct_predicate.as_ref() {
            if let Some(false) = eval_ndjson_byte_predicate_row(&row, predicate)? {
                continue;
            }
        }

        let document = executor.parse_owned_row(line_no, row)?;
        let matched = executor.eval_document(line_no, &document)?;
        if !is_truthy(&matched) {
            continue;
        }

        let root = document
            .root_val_with(engine.keys())
            .map_err(|err| row_eval_error(line_no, err))?;
        emitted += 1;
        if matches!(emit(root)?, NdjsonControl::Stop) || emitted >= limit {
            break;
        }
    }

    Ok(emitted)
}

fn drive_ndjson_matches_writer<R, W>(
    engine: &JetroEngine,
    reader: R,
    predicate: &str,
    limit: usize,
    options: NdjsonOptions,
    writer: W,
) -> Result<usize, JetroEngineError>
where
    R: BufRead,
    W: Write,
{
    Ok(drive_ndjson_matches_writer_with_stats(
        engine, reader, predicate, limit, options, writer,
    )?
    .0)
}

fn drive_ndjson_matches_writer_with_stats<R, W>(
    engine: &JetroEngine,
    reader: R,
    predicate: &str,
    limit: usize,
    options: NdjsonOptions,
    writer: W,
) -> Result<(usize, NdjsonExecutionStats), JetroEngineError>
where
    R: BufRead,
    W: Write,
{
    if limit == 0 {
        return Ok((0, NdjsonExecutionStats::default()));
    }
    if let Some(predicate) = direct_tape_predicate(engine, predicate) {
        return drive_ndjson_tape_matches_writer_with_stats(
            engine, reader, &predicate, limit, options, writer,
        );
    }

    let mut driver = NdjsonPerRowDriver::new(reader).with_options(options);
    let mut executor = NdjsonRowExecutor::new(engine, predicate);
    let mut writer = ndjson_writer_with_options(writer, options);
    let mut buf = Vec::with_capacity(options.initial_buffer_capacity);
    let mut emitted = 0usize;
    let mut stats = NdjsonExecutionStats::default();

    while let Some((line_no, row)) = driver.read_next_owned(&mut buf)? {
        stats.rows_scanned += 1;
        let document = executor.parse_owned_row(line_no, row)?;
        let matched = executor.eval_document(line_no, &document)?;
        stats.fallback_filter_rows += 1;
        if !is_truthy(&matched) {
            stats.rows_filtered += 1;
            continue;
        }

        write_document_line(&mut writer, &document, line_no, executor.engine())?;
        emitted += 1;
        stats.rows_emitted += 1;
        if emitted >= limit {
            break;
        }
    }

    writer.flush()?;
    Ok((emitted, stats))
}

pub(super) struct NdjsonRowExecutor<'a> {
    engine: &'a JetroEngine,
    plan: crate::ir::physical::QueryPlan,
    vm: MutexGuard<'a, VM>,
}

impl<'a> NdjsonRowExecutor<'a> {
    pub(super) fn new(engine: &'a JetroEngine, query: &str) -> Self {
        Self {
            engine,
            plan: engine.cached_plan(query, PlanningContext::bytes()),
            vm: engine.lock_vm(),
        }
    }

    pub(super) fn eval_owned_row(
        &mut self,
        line_no: u64,
        row: Vec<u8>,
    ) -> Result<Val, JetroEngineError> {
        let document = self.parse_owned_row(line_no, row)?;
        self.eval_document(line_no, &document)
    }

    pub(super) fn parse_owned_row(
        &self,
        line_no: u64,
        row: Vec<u8>,
    ) -> Result<Jetro, JetroEngineError> {
        parse_row(self.engine, line_no, row)
    }

    pub(super) fn eval_document(
        &mut self,
        line_no: u64,
        document: &Jetro,
    ) -> Result<Val, JetroEngineError> {
        crate::exec::router::collect_plan_val_with_vm(document, &self.plan, &mut self.vm)
            .map_err(|err| row_eval_error(line_no, err))
    }

    pub(super) fn engine(&self) -> &'a JetroEngine {
        self.engine
    }
}
trait JsonTape {
    fn nodes(&self) -> &[crate::data::tape::TapeNode];
    fn str_at(&self, idx: usize) -> &str;
    fn span(&self, idx: usize) -> usize;
}
impl JsonTape for crate::data::tape::TapeData {
    #[inline]
    fn nodes(&self) -> &[crate::data::tape::TapeNode] {
        &self.nodes
    }

    #[inline]
    fn str_at(&self, idx: usize) -> &str {
        self.str_at(idx)
    }

    #[inline]
    fn span(&self, idx: usize) -> usize {
        self.span(idx)
    }
}
impl JsonTape for crate::data::tape::TapeScratch {
    #[inline]
    fn nodes(&self) -> &[crate::data::tape::TapeNode] {
        &self.nodes
    }

    #[inline]
    fn str_at(&self, idx: usize) -> &str {
        self.str_at(idx)
    }

    #[inline]
    fn span(&self, idx: usize) -> usize {
        self.span(idx)
    }
}
fn json_tape_path_index<T: JsonTape>(
    tape: &T,
    steps: &[crate::ir::physical::PhysicalPathStep],
) -> Option<usize> {
    json_tape_path_index_from(tape, 0, steps)
}
fn json_tape_path_index_from<T: JsonTape>(
    tape: &T,
    start: usize,
    steps: &[crate::ir::physical::PhysicalPathStep],
) -> Option<usize> {
    if tape.nodes().is_empty() {
        return None;
    }

    return match steps {
        [] => Some(start),
        [step] => json_tape_step_index(tape, start, step),
        [first, second] => json_tape_step_index(tape, start, first)
            .and_then(|idx| json_tape_step_index(tape, idx, second)),
        _ => json_tape_path_index_slow(tape, start, steps),
    };
}
fn json_tape_path_index_slow<T: JsonTape>(
    tape: &T,
    start: usize,
    steps: &[crate::ir::physical::PhysicalPathStep],
) -> Option<usize> {
    let mut idx = start;
    for step in steps {
        idx = json_tape_step_index(tape, idx, step)?;
    }
    Some(idx)
}
fn json_tape_step_index<T: JsonTape>(
    tape: &T,
    start: usize,
    step: &crate::ir::physical::PhysicalPathStep,
) -> Option<usize> {
    use crate::data::tape::TapeNode;
    use crate::ir::physical::PhysicalPathStep;

    match step {
        PhysicalPathStep::Field(key) => {
            let TapeNode::Object { len, .. } = tape.nodes()[start] else {
                return None;
            };
            let mut cur = start + 1;
            for _ in 0..len {
                if tape.str_at(cur) == key.as_ref() {
                    return Some(cur + 1);
                }
                cur += 1;
                cur += tape.span(cur);
            }
            None
        }
        PhysicalPathStep::Index(wanted) => {
            let TapeNode::Array { len, .. } = tape.nodes()[start] else {
                return None;
            };
            let wanted = if *wanted < 0 {
                len.checked_sub(wanted.unsigned_abs() as usize)?
            } else {
                *wanted as usize
            };
            if wanted >= len {
                return None;
            }
            let mut cur = start + 1;
            for _ in 0..wanted {
                cur += tape.span(cur);
            }
            Some(cur)
        }
    }
}
fn json_tape_object_cached_field<T: JsonTape>(
    tape: &T,
    obj_idx: usize,
    cache: NdjsonFieldCache,
    key: &str,
) -> Option<usize> {
    let crate::data::tape::TapeNode::Object { .. } = tape.nodes().get(obj_idx).copied()? else {
        return None;
    };
    let key_idx = obj_idx.checked_add(cache.key_delta)?;
    let value_idx = obj_idx.checked_add(cache.value_delta)?;
    if value_idx >= tape.nodes().len() {
        return None;
    }
    if !matches!(
        tape.nodes().get(key_idx),
        Some(crate::data::tape::TapeNode::String(_))
    ) {
        return None;
    }
    (tape.str_at(key_idx) == key).then_some(value_idx)
}
fn json_tape_object_field_index_and_cache<T: JsonTape>(
    tape: &T,
    obj_idx: usize,
    key: &str,
) -> Option<(usize, NdjsonFieldCache)> {
    let crate::data::tape::TapeNode::Object { len, .. } = tape.nodes()[obj_idx] else {
        return None;
    };
    let mut cur = obj_idx + 1;
    for _ in 0..len {
        if tape.str_at(cur) == key {
            return Some((
                cur + 1,
                NdjsonFieldCache {
                    key_delta: cur - obj_idx,
                    value_delta: cur + 1 - obj_idx,
                },
            ));
        }
        cur += 1;
        cur += tape.span(cur);
    }
    None
}
fn json_tape_array_element<T: JsonTape>(
    tape: &T,
    idx: usize,
    element: NdjsonDirectElement,
) -> Option<usize> {
    let crate::data::tape::TapeNode::Array { len, .. } = tape.nodes().get(idx).copied()? else {
        return None;
    };
    let wanted = match element {
        NdjsonDirectElement::First => 0,
        NdjsonDirectElement::Last => len.checked_sub(1)?,
        NdjsonDirectElement::Nth(n) => n,
    };
    if wanted >= len {
        return None;
    }
    let mut cur = idx + 1;
    for _ in 0..wanted {
        cur += tape.span(cur);
    }
    Some(cur)
}
pub(super) fn eval_tape_predicate(
    tape: &crate::data::tape::TapeScratch,
    predicate: &NdjsonDirectPredicate,
    env: Option<&crate::data::context::Env>,
    vm: &mut Option<std::sync::MutexGuard<'_, crate::vm::exec::VM>>,
    cache: &mut NdjsonPathCache,
) -> Result<bool, crate::EvalError> {
    use crate::parse::ast::BinOp;

    Ok(match predicate {
        NdjsonDirectPredicate::Path(steps) => cache
            .index(tape, 0, steps)
            .map(|idx| json_view_truthy(json_tape_scalar(tape, idx)))
            .unwrap_or(false),
        NdjsonDirectPredicate::Literal(value) => crate::util::is_truthy(value),
        NdjsonDirectPredicate::Not(inner) => !eval_tape_predicate(tape, inner, env, vm, cache)?,
        NdjsonDirectPredicate::Binary { lhs, op, rhs } if *op == BinOp::And => {
            eval_tape_predicate(tape, lhs, env, vm, cache)?
                && eval_tape_predicate(tape, rhs, env, vm, cache)?
        }
        NdjsonDirectPredicate::Binary { lhs, op, rhs } if *op == BinOp::Or => {
            eval_tape_predicate(tape, lhs, env, vm, cache)?
                || eval_tape_predicate(tape, rhs, env, vm, cache)?
        }
        NdjsonDirectPredicate::Binary { lhs, op, rhs } => {
            let Some(lhs) = eval_tape_scalar(tape, lhs, cache) else {
                return Ok(false);
            };
            let Some(rhs) = eval_tape_scalar(tape, rhs, cache) else {
                return Ok(false);
            };
            crate::util::json_cmp_binop(lhs, *op, rhs)
        }
        NdjsonDirectPredicate::ViewScalarCall { steps, call } => cache
            .index(tape, 0, steps)
            .map(|idx| json_tape_scalar(tape, idx))
            .and_then(|value| call.try_apply_json_view(value))
            .is_some_and(|value| crate::util::is_truthy(&value)),
        NdjsonDirectPredicate::ArrayElementViewScalarCall {
            source_steps,
            element,
            suffix_steps,
            call,
        } => json_tape_path_index(tape, source_steps)
            .and_then(|idx| json_tape_array_element(tape, idx, *element))
            .and_then(|idx| json_tape_path_index_from(tape, idx, suffix_steps))
            .map(|idx| json_tape_scalar(tape, idx))
            .and_then(|value| call.try_apply_json_view(value))
            .is_some_and(|value| crate::util::is_truthy(&value)),
        NdjsonDirectPredicate::ArrayAny { .. } => {
            return Err(crate::EvalError(
                "array-any predicate requires VM state".to_string(),
            ));
        }
        NdjsonDirectPredicate::ViewPipeline { source_steps, body } => {
            let (Some(vm), Some(env)) = (vm.as_deref_mut(), env) else {
                return Err(crate::EvalError(
                    "view pipeline predicate requires VM state".to_string(),
                ));
            };
            let source = json_tape_path_index(tape, source_steps)
                .map(|idx| crate::data::view::TapeScratchView::Node { tape, idx })
                .unwrap_or(crate::data::view::TapeScratchView::Missing);
            crate::exec::view::run_with_env_and_vm(source, body, None, env, vm)
                .transpose()?
                .is_some_and(|value| crate::util::is_truthy(&value))
        }
    })
}
pub(super) fn predicate_needs_vm(predicate: &NdjsonDirectPredicate) -> bool {
    match predicate {
        NdjsonDirectPredicate::Not(inner) => predicate_needs_vm(inner),
        NdjsonDirectPredicate::Binary { lhs, rhs, .. } => {
            predicate_needs_vm(lhs) || predicate_needs_vm(rhs)
        }
        NdjsonDirectPredicate::ArrayAny { .. } | NdjsonDirectPredicate::ViewPipeline { .. } => true,
        NdjsonDirectPredicate::Path(_)
        | NdjsonDirectPredicate::Literal(_)
        | NdjsonDirectPredicate::ViewScalarCall { .. }
        | NdjsonDirectPredicate::ArrayElementViewScalarCall { .. } => false,
    }
}
fn eval_tape_scalar<'a>(
    tape: &'a crate::data::tape::TapeScratch,
    predicate: &'a NdjsonDirectPredicate,
    cache: &mut NdjsonPathCache,
) -> Option<crate::util::JsonView<'a>> {
    match predicate {
        NdjsonDirectPredicate::Path(steps) => cache
            .index(tape, 0, steps)
            .map(|idx| json_tape_scalar(tape, idx)),
        NdjsonDirectPredicate::Literal(value) => Some(crate::util::JsonView::from_val(value)),
        _ => None,
    }
}
fn json_view_truthy(value: crate::util::JsonView<'_>) -> bool {
    match value {
        crate::util::JsonView::Null => false,
        crate::util::JsonView::Bool(value) => value,
        crate::util::JsonView::Int(value) => value != 0,
        crate::util::JsonView::UInt(value) => value != 0,
        crate::util::JsonView::Float(value) => value != 0.0,
        crate::util::JsonView::Str(value) => !value.is_empty(),
        crate::util::JsonView::ArrayLen(len) | crate::util::JsonView::ObjectLen(len) => len > 0,
    }
}
fn json_tape_scalar<T: JsonTape>(tape: &T, idx: usize) -> crate::util::JsonView<'_> {
    use crate::data::tape::TapeNode;
    use simd_json::StaticNode as SN;

    let Some(node) = tape.nodes().get(idx).copied() else {
        return crate::util::JsonView::Null;
    };
    match node {
        TapeNode::Static(SN::Null) => crate::util::JsonView::Null,
        TapeNode::Static(SN::Bool(value)) => crate::util::JsonView::Bool(value),
        TapeNode::Static(SN::I64(value)) => crate::util::JsonView::Int(value),
        TapeNode::Static(SN::U64(value)) => crate::util::JsonView::UInt(value),
        TapeNode::Static(SN::F64(value)) => crate::util::JsonView::Float(value),
        TapeNode::String(_) => crate::util::JsonView::Str(tape.str_at(idx)),
        TapeNode::Array { len, .. } => crate::util::JsonView::ArrayLen(len),
        TapeNode::Object { len, .. } => crate::util::JsonView::ObjectLen(len),
    }
}

pub(super) fn write_document_line<W: Write>(
    writer: &mut W,
    document: &Jetro,
    line_no: u64,
    engine: &JetroEngine,
) -> Result<(), JetroEngineError> {
    if let Some(bytes) = document.raw_bytes() {
        writer.write_all(bytes)?;
        writer.write_all(b"\n")?;
        return Ok(());
    }

    let root = document
        .root_val_with(engine.keys())
        .map_err(|err| row_eval_error(line_no, err))?;
    write_val_line(writer, &root)
}

pub(super) fn write_val_json<W: Write>(
    writer: &mut W,
    value: &Val,
) -> Result<(), JetroEngineError> {
    match value {
        Val::Null => writer.write_all(b"null")?,
        Val::Bool(true) => writer.write_all(b"true")?,
        Val::Bool(false) => writer.write_all(b"false")?,
        Val::Int(n) => write_i64(writer, *n)?,
        Val::Float(n) => write_f64(writer, *n)?,
        Val::Str(s) => write_json_str(writer, s.as_ref())?,
        Val::StrSlice(s) => write_json_str(writer, s.as_str())?,
        Val::Arr(items) => write_json_array(writer, items.iter())?,
        Val::IntVec(items) => write_json_int_array(writer, items.iter().copied())?,
        Val::FloatVec(items) => write_json_float_array(writer, items.iter().copied())?,
        Val::StrVec(items) => write_json_str_array(writer, items.iter().map(|s| s.as_ref()))?,
        Val::StrSliceVec(items) => write_json_str_array(writer, items.iter().map(|s| s.as_str()))?,
        Val::Obj(entries) => write_json_object(
            writer,
            entries.iter().map(|(key, value)| (key.as_ref(), value)),
        )?,
        Val::ObjSmall(entries) => write_json_object(
            writer,
            entries.iter().map(|(key, value)| (key.as_ref(), value)),
        )?,
        Val::ObjVec(data) => write_json_objvec(writer, data)?,
    }
    Ok(())
}
fn write_json_tape_at<W: Write, T: JsonTape>(
    writer: &mut W,
    tape: &T,
    idx: usize,
) -> Result<usize, JetroEngineError> {
    use crate::data::tape::TapeNode;
    use simd_json::StaticNode as SN;

    let Some(node) = tape.nodes().get(idx).copied() else {
        writer.write_all(b"null")?;
        return Ok(idx);
    };

    match node {
        TapeNode::Static(SN::Null) => {
            writer.write_all(b"null")?;
            Ok(idx + 1)
        }
        TapeNode::Static(SN::Bool(true)) => {
            writer.write_all(b"true")?;
            Ok(idx + 1)
        }
        TapeNode::Static(SN::Bool(false)) => {
            writer.write_all(b"false")?;
            Ok(idx + 1)
        }
        TapeNode::Static(SN::I64(value)) => {
            write_i64(writer, value)?;
            Ok(idx + 1)
        }
        TapeNode::Static(SN::U64(value)) => {
            write_u64(writer, value)?;
            Ok(idx + 1)
        }
        TapeNode::Static(SN::F64(value)) => {
            write_f64(writer, value)?;
            Ok(idx + 1)
        }
        TapeNode::String(_) => {
            write_json_str(writer, tape.str_at(idx))?;
            Ok(idx + 1)
        }
        TapeNode::Array { len, .. } => {
            writer.write_all(b"[")?;
            let mut cur = idx + 1;
            for item_idx in 0..len {
                if item_idx > 0 {
                    writer.write_all(b",")?;
                }
                cur = write_json_tape_at(writer, tape, cur)?;
            }
            writer.write_all(b"]")?;
            Ok(cur)
        }
        TapeNode::Object { len, .. } => {
            writer.write_all(b"{")?;
            let mut cur = idx + 1;
            for field_idx in 0..len {
                if field_idx > 0 {
                    writer.write_all(b",")?;
                }
                write_json_str(writer, tape.str_at(cur))?;
                writer.write_all(b":")?;
                cur = write_json_tape_at(writer, tape, cur + 1)?;
            }
            writer.write_all(b"}")?;
            Ok(cur)
        }
    }
}
fn visit_json_tape_source_items<T, E, F>(tape: &T, source_idx: usize, mut visit: F) -> Result<(), E>
where
    T: JsonTape,
    F: FnMut(usize) -> Result<(), E>,
{
    use crate::data::tape::TapeNode;

    match tape.nodes().get(source_idx).copied() {
        Some(TapeNode::Array { len, .. }) => {
            let mut cur = source_idx + 1;
            for _ in 0..len {
                visit(cur)?;
                cur += tape.span(cur);
            }
        }
        Some(_) => visit(source_idx)?,
        None => {}
    }
    Ok(())
}
fn find_json_tape_source_item<T, F>(tape: &T, source_idx: usize, mut matches: F) -> Option<usize>
where
    T: JsonTape,
    F: FnMut(usize) -> bool,
{
    use crate::data::tape::TapeNode;

    match tape.nodes().get(source_idx).copied()? {
        TapeNode::Array { len, .. } => {
            let mut cur = source_idx + 1;
            for _ in 0..len {
                if matches(cur) {
                    return Some(cur);
                }
                cur += tape.span(cur);
            }
            None
        }
        _ => matches(source_idx).then_some(source_idx),
    }
}
fn write_json_tape_stream<W: Write, T: JsonTape>(
    writer: &mut W,
    tape: &T,
    plan: &NdjsonDirectStreamPlan,
    source_cache: &mut NdjsonPathCache,
    suffix_cache: &mut NdjsonPathCache,
    predicate_cache: &mut NdjsonPathCache,
    projection_caches: &mut Vec<NdjsonPathCache>,
) -> Result<(), JetroEngineError> {
    let Some(source_idx) = source_cache.index(tape, 0, &plan.source_steps) else {
        write_json_tape_empty_stream_result(writer, &plan.sink)?;
        return Ok(());
    };

    match &plan.sink {
        NdjsonDirectStreamSink::Collect(map) => {
            writer.write_all(b"[")?;
            let mut wrote_row = false;
            visit_json_tape_source_items(tape, source_idx, |item_idx| {
                if !plan.predicate.as_ref().is_none_or(|predicate| {
                    eval_json_tape_item_predicate_cached(tape, item_idx, predicate, predicate_cache)
                }) {
                    return Ok::<(), JetroEngineError>(());
                }
                if wrote_row {
                    writer.write_all(b",")?;
                }
                write_json_tape_stream_map(
                    writer,
                    tape,
                    item_idx,
                    map,
                    suffix_cache,
                    projection_caches,
                )?;
                wrote_row = true;
                Ok(())
            })?;
            writer.write_all(b"]")?;
        }
        NdjsonDirectStreamSink::Count => {
            let mut count = 0usize;
            let _: Result<(), ()> = visit_json_tape_source_items(tape, source_idx, |item_idx| {
                if plan.predicate.as_ref().is_none_or(|predicate| {
                    eval_json_tape_item_predicate_cached(tape, item_idx, predicate, predicate_cache)
                }) {
                    count += 1;
                }
                Ok(())
            });
            write_i64(writer, count as i64)?;
        }
        NdjsonDirectStreamSink::First(map) => {
            let selected = find_json_tape_source_item(tape, source_idx, |item_idx| {
                plan.predicate.as_ref().is_none_or(|predicate| {
                    eval_json_tape_item_predicate_cached(tape, item_idx, predicate, predicate_cache)
                })
            });
            if let Some(item_idx) = selected {
                write_json_tape_stream_map(
                    writer,
                    tape,
                    item_idx,
                    map,
                    suffix_cache,
                    projection_caches,
                )?;
            } else {
                writer.write_all(b"null")?;
            }
        }
        NdjsonDirectStreamSink::Last(map) => {
            let mut selected = None;
            visit_json_tape_source_items(tape, source_idx, |item_idx| {
                if plan.predicate.as_ref().is_none_or(|predicate| {
                    eval_json_tape_item_predicate_cached(tape, item_idx, predicate, predicate_cache)
                }) {
                    selected = Some(item_idx);
                }
                Ok::<(), JetroEngineError>(())
            })?;
            if let Some(item_idx) = selected {
                write_json_tape_stream_map(
                    writer,
                    tape,
                    item_idx,
                    map,
                    suffix_cache,
                    projection_caches,
                )?;
            } else {
                writer.write_all(b"null")?;
            }
        }
        NdjsonDirectStreamSink::Numeric { suffix_steps, op } => {
            let caches = NdjsonPathCaches {
                source: source_cache,
                suffix: suffix_cache,
                predicate: predicate_cache,
            };
            let value = reduce_json_tape_numeric_path(
                tape,
                &plan.source_steps,
                plan.predicate.as_ref(),
                suffix_steps,
                *op,
                caches,
            );
            write_val_json(writer, &value)?;
        }
        NdjsonDirectStreamSink::Extreme {
            key_steps,
            op,
            value,
        } => {
            let mut best_idx = None;
            visit_json_tape_source_items(tape, source_idx, |item_idx| {
                let Some(key_idx) = suffix_cache.index(tape, item_idx, key_steps) else {
                    return Ok::<(), JetroEngineError>(());
                };
                let key = json_tape_scalar(tape, key_idx);
                let replace = best_idx
                    .and_then(|idx| suffix_cache.index(tape, idx, key_steps))
                    .map(|idx| {
                        let order = crate::util::json_cmp_vals(key, json_tape_scalar(tape, idx));
                        (op.wants_max() && order.is_gt()) || (!op.wants_max() && order.is_lt())
                    })
                    .unwrap_or(true);
                if replace {
                    best_idx = Some(item_idx);
                }
                Ok(())
            })?;
            if let Some(item_idx) = best_idx {
                let path_idx = value
                    .path_steps()
                    .and_then(|steps| suffix_cache.index(tape, item_idx, steps));
                write_json_tape_direct_value(writer, tape, value, path_idx)?;
            } else {
                writer.write_all(b"null")?;
            }
        }
    }

    Ok(())
}
fn write_json_tape_empty_stream_result<W: Write>(
    writer: &mut W,
    sink: &NdjsonDirectStreamSink,
) -> Result<(), JetroEngineError> {
    match sink {
        NdjsonDirectStreamSink::Collect(_) => writer.write_all(b"[]")?,
        NdjsonDirectStreamSink::First(_) | NdjsonDirectStreamSink::Last(_) => {
            writer.write_all(b"null")?
        }
        NdjsonDirectStreamSink::Count => writer.write_all(b"0")?,
        NdjsonDirectStreamSink::Numeric { op, .. } => {
            let value = crate::exec::pipeline::num_finalise(
                *op,
                0,
                0.0,
                false,
                f64::INFINITY,
                f64::NEG_INFINITY,
                0,
            );
            write_val_json(writer, &value)?;
        }
        NdjsonDirectStreamSink::Extreme { .. } => writer.write_all(b"null")?,
    }
    Ok(())
}
fn write_json_tape_stream_map<W: Write, T: JsonTape>(
    writer: &mut W,
    tape: &T,
    item_idx: usize,
    map: &NdjsonDirectStreamMap,
    suffix_cache: &mut NdjsonPathCache,
    projection_caches: &mut Vec<NdjsonPathCache>,
) -> Result<(), JetroEngineError> {
    match map {
        NdjsonDirectStreamMap::Value(value) => {
            let path_idx = value
                .path_steps()
                .and_then(|steps| suffix_cache.index(tape, item_idx, steps));
            write_json_tape_direct_value(writer, tape, value, path_idx)?;
        }
        NdjsonDirectStreamMap::Array(items) => {
            write_json_tape_array_projection_from(
                writer,
                tape,
                item_idx,
                items,
                projection_caches,
            )?;
        }
        NdjsonDirectStreamMap::Object(fields) => {
            write_json_tape_object_projection_from(
                writer,
                tape,
                item_idx,
                fields,
                projection_caches,
            )?;
        }
    }
    Ok(())
}
fn write_json_tape_object_projection<W: Write, T: JsonTape>(
    writer: &mut W,
    tape: &T,
    fields: &[super::ndjson_direct::NdjsonDirectObjectField],
    path_caches: &mut Vec<NdjsonPathCache>,
) -> Result<(), JetroEngineError> {
    write_json_tape_object_projection_from(writer, tape, 0, fields, path_caches)
}
fn write_json_tape_object_projection_from<W: Write, T: JsonTape>(
    writer: &mut W,
    tape: &T,
    start: usize,
    fields: &[super::ndjson_direct::NdjsonDirectObjectField],
    path_caches: &mut Vec<NdjsonPathCache>,
) -> Result<(), JetroEngineError> {
    if path_caches.len() < fields.len() {
        path_caches.resize_with(fields.len(), NdjsonPathCache::default);
    }
    writer.write_all(b"{")?;
    let mut wrote = false;
    for (field_idx, field) in fields.iter().enumerate() {
        let path_cache = &mut path_caches[field_idx];
        let mut path_idx = None;
        match &field.value {
            NdjsonDirectProjectionValue::Path(steps) => {
                let idx = path_cache.index(tape, start, steps);
                path_idx = idx;
                if field.optional
                    && idx
                        .map(|idx| {
                            matches!(json_tape_scalar(tape, idx), crate::util::JsonView::Null)
                        })
                        .unwrap_or(true)
                {
                    continue;
                }
            }
            NdjsonDirectProjectionValue::ViewScalarCall {
                steps,
                call,
                optional,
            } => {
                let idx = path_cache.index(tape, start, steps);
                path_idx = idx;
                if (*optional || field.optional)
                    && idx
                        .map(|idx| {
                            matches!(json_tape_scalar(tape, idx), crate::util::JsonView::Null)
                        })
                        .unwrap_or(true)
                {
                    continue;
                }
                if field.optional
                    && idx
                        .map(|idx| json_tape_scalar(tape, idx))
                        .and_then(|value| call.try_apply_json_view(value))
                        .is_some_and(|value| matches!(value, Val::Null))
                {
                    continue;
                }
            }
            NdjsonDirectProjectionValue::Literal(Val::Null) if field.optional => {
                continue;
            }
            NdjsonDirectProjectionValue::Nested(_) => {}
            NdjsonDirectProjectionValue::Literal(_) => {}
        }
        if wrote {
            writer.write_all(b",")?;
        }
        write_json_str(writer, field.key.as_ref())?;
        writer.write_all(b":")?;
        write_json_tape_direct_value(writer, tape, &field.value, path_idx)?;
        wrote = true;
    }
    writer.write_all(b"}")?;
    Ok(())
}
fn write_json_tape_array_projection<W: Write, T: JsonTape>(
    writer: &mut W,
    tape: &T,
    items: &[NdjsonDirectProjectionValue],
    path_caches: &mut Vec<NdjsonPathCache>,
) -> Result<(), JetroEngineError> {
    write_json_tape_array_projection_from(writer, tape, 0, items, path_caches)
}
fn write_json_tape_array_projection_from<W: Write, T: JsonTape>(
    writer: &mut W,
    tape: &T,
    start: usize,
    items: &[NdjsonDirectProjectionValue],
    path_caches: &mut Vec<NdjsonPathCache>,
) -> Result<(), JetroEngineError> {
    if path_caches.len() < items.len() {
        path_caches.resize_with(items.len(), NdjsonPathCache::default);
    }
    writer.write_all(b"[")?;
    for (idx, item) in items.iter().enumerate() {
        if idx > 0 {
            writer.write_all(b",")?;
        }
        let path_idx = item
            .path_steps()
            .and_then(|steps| path_caches[idx].index(tape, start, steps));
        write_json_tape_direct_value(writer, tape, item, path_idx)?;
    }
    writer.write_all(b"]")?;
    Ok(())
}
fn write_json_tape_direct_value<W: Write, T: JsonTape>(
    writer: &mut W,
    tape: &T,
    value: &NdjsonDirectProjectionValue,
    path_idx: Option<usize>,
) -> Result<(), JetroEngineError> {
    match value {
        NdjsonDirectProjectionValue::Path(_) => {
            if let Some(idx) = path_idx {
                write_json_tape_at(writer, tape, idx)?;
            } else {
                writer.write_all(b"null")?;
            }
        }
        NdjsonDirectProjectionValue::ViewScalarCall { call, .. } => {
            if let Some(idx) = path_idx {
                let value = json_tape_scalar(tape, idx);
                if let Some(value) = call.try_apply_json_view(value) {
                    write_val_json(writer, &value)?;
                } else {
                    write_json_tape_at(writer, tape, idx)?;
                }
            } else {
                writer.write_all(b"null")?;
            }
        }
        NdjsonDirectProjectionValue::Literal(value) => write_val_json(writer, value)?,
        NdjsonDirectProjectionValue::Nested(plan) => {
            write_json_tape_nested_plan(writer, tape, plan)?;
        }
    }
    Ok(())
}
fn write_json_tape_nested_plan<W: Write, T: JsonTape>(
    writer: &mut W,
    tape: &T,
    plan: &NdjsonDirectTapePlan,
) -> Result<(), JetroEngineError> {
    let mut root_cache = NdjsonPathCache::default();
    let mut source_cache = NdjsonPathCache::default();
    let mut suffix_cache = NdjsonPathCache::default();
    let mut predicate_cache = NdjsonPathCache::default();
    let mut projection_caches = Vec::new();
    match plan {
        NdjsonDirectTapePlan::RootPath(steps) => {
            if let Some(idx) = root_cache.index(tape, 0, steps) {
                write_json_tape_at(writer, tape, idx)?;
            } else {
                writer.write_all(b"null")?;
            }
        }
        NdjsonDirectTapePlan::ViewScalarCall {
            steps,
            call,
            optional,
        } => {
            let idx = root_cache.index(tape, 0, steps);
            let value = idx
                .map(|idx| json_tape_scalar(tape, idx))
                .unwrap_or(crate::util::JsonView::Null);
            if *optional && matches!(value, crate::util::JsonView::Null) {
                writer.write_all(b"null")?;
            } else if let Some(value) = call.try_apply_json_view(value) {
                write_val_json(writer, &value)?;
            } else if let Some(idx) = idx {
                write_json_tape_at(writer, tape, idx)?;
            } else {
                writer.write_all(b"null")?;
            }
        }
        NdjsonDirectTapePlan::ArrayElementPath {
            source_steps,
            element,
            suffix_steps,
        } => {
            write_json_tape_array_element_path(
                writer,
                tape,
                source_steps,
                *element,
                suffix_steps,
                &mut source_cache,
                &mut suffix_cache,
            )?;
        }
        NdjsonDirectTapePlan::ArrayElementViewScalarCall {
            source_steps,
            element,
            suffix_steps,
            call,
        } => {
            write_json_tape_array_element_scalar(
                writer,
                tape,
                source_steps,
                *element,
                suffix_steps,
                call,
                &mut source_cache,
                &mut suffix_cache,
            )?;
        }
        NdjsonDirectTapePlan::Stream(stream) => {
            write_json_tape_stream(
                writer,
                tape,
                stream,
                &mut source_cache,
                &mut suffix_cache,
                &mut predicate_cache,
                &mut projection_caches,
            )?;
        }
        NdjsonDirectTapePlan::Object(fields) => {
            write_json_tape_object_projection(writer, tape, fields, &mut projection_caches)?;
        }
        NdjsonDirectTapePlan::Array(items) => {
            write_json_tape_array_projection(writer, tape, items, &mut projection_caches)?;
        }
        NdjsonDirectTapePlan::ObjectItems { steps, method } => {
            let idx = root_cache.index(tape, 0, steps);
            write_json_tape_object_items(writer, tape, idx, *method)?;
        }
        NdjsonDirectTapePlan::ViewPipeline { .. } => {
            writer.write_all(b"null")?;
        }
    }
    Ok(())
}
fn write_json_tape_array_element_path<W: Write, T: JsonTape>(
    writer: &mut W,
    tape: &T,
    source_steps: &[crate::ir::physical::PhysicalPathStep],
    element: super::ndjson_direct::NdjsonDirectElement,
    suffix_steps: &[crate::ir::physical::PhysicalPathStep],
    source_cache: &mut NdjsonPathCache,
    suffix_cache: &mut NdjsonPathCache,
) -> Result<(), JetroEngineError> {
    let idx = source_cache
        .index(tape, 0, source_steps)
        .and_then(|idx| json_tape_array_element(tape, idx, element))
        .and_then(|idx| suffix_cache.index(tape, idx, suffix_steps));
    if let Some(idx) = idx {
        write_json_tape_at(writer, tape, idx)?;
    } else {
        writer.write_all(b"null")?;
    }
    Ok(())
}
fn write_json_tape_array_element_scalar<W: Write, T: JsonTape>(
    writer: &mut W,
    tape: &T,
    source_steps: &[crate::ir::physical::PhysicalPathStep],
    element: super::ndjson_direct::NdjsonDirectElement,
    suffix_steps: &[crate::ir::physical::PhysicalPathStep],
    call: &crate::builtins::BuiltinCall,
    source_cache: &mut NdjsonPathCache,
    suffix_cache: &mut NdjsonPathCache,
) -> Result<(), JetroEngineError> {
    let idx = source_cache
        .index(tape, 0, source_steps)
        .and_then(|idx| json_tape_array_element(tape, idx, element))
        .and_then(|idx| suffix_cache.index(tape, idx, suffix_steps));
    if let Some(value) = idx
        .map(|idx| json_tape_scalar(tape, idx))
        .and_then(|value| call.try_apply_json_view(value))
    {
        write_val_json(writer, &value)?;
    } else if let Some(idx) = idx {
        write_json_tape_at(writer, tape, idx)?;
    } else {
        writer.write_all(b"null")?;
    }
    Ok(())
}
fn write_json_tape_object_items<W: Write, T: JsonTape>(
    writer: &mut W,
    tape: &T,
    obj_idx: Option<usize>,
    method: crate::builtins::BuiltinMethod,
) -> Result<(), JetroEngineError> {
    let Some(projection) = view_object_projection(BuiltinId::from_method(method)) else {
        writer.write_all(b"[]")?;
        return Ok(());
    };
    let Some(obj_idx) = obj_idx else {
        writer.write_all(b"[]")?;
        return Ok(());
    };
    let Some(crate::data::tape::TapeNode::Object { len, .. }) = tape.nodes().get(obj_idx).copied()
    else {
        writer.write_all(b"[]")?;
        return Ok(());
    };

    writer.write_all(b"[")?;
    let mut cur = obj_idx + 1;
    for field_idx in 0..len {
        if field_idx > 0 {
            writer.write_all(b",")?;
        }
        match projection {
            BuiltinViewObjectProjection::Keys => {
                write_json_str(writer, tape.str_at(cur))?;
                cur += 1;
                cur += tape.span(cur);
            }
            BuiltinViewObjectProjection::Values => {
                cur = write_json_tape_at(writer, tape, cur + 1)?;
            }
            BuiltinViewObjectProjection::Entries => {
                writer.write_all(b"[")?;
                write_json_str(writer, tape.str_at(cur))?;
                writer.write_all(b",")?;
                cur = write_json_tape_at(writer, tape, cur + 1)?;
                writer.write_all(b"]")?;
            }
            BuiltinViewObjectProjection::ToPairs => {
                writer.write_all(br#"{"key":"#)?;
                write_json_str(writer, tape.str_at(cur))?;
                writer.write_all(br#","val":"#)?;
                cur = write_json_tape_at(writer, tape, cur + 1)?;
                writer.write_all(b"}")?;
            }
            _ => unreachable!("non-object-items builtin"),
        }
    }
    writer.write_all(b"]")?;
    Ok(())
}
fn reduce_json_tape_numeric_path<T: JsonTape>(
    tape: &T,
    source_steps: &[crate::ir::physical::PhysicalPathStep],
    predicate: Option<&NdjsonDirectItemPredicate>,
    suffix_steps: &[crate::ir::physical::PhysicalPathStep],
    op: crate::exec::pipeline::NumOp,
    caches: NdjsonPathCaches<'_>,
) -> Val {
    let mut acc_i = 0i64;
    let mut acc_f = 0.0f64;
    let mut floated = false;
    let mut min_f = f64::INFINITY;
    let mut max_f = f64::NEG_INFINITY;
    let mut n_obs = 0usize;

    let Some(source_idx) = caches.source.index(tape, 0, source_steps) else {
        return crate::exec::pipeline::num_finalise(op, acc_i, acc_f, floated, min_f, max_f, n_obs);
    };

    let suffix_cache = caches.suffix;
    let predicate_cache = caches.predicate;
    let _: Result<(), ()> = visit_json_tape_source_items(tape, source_idx, |item_idx| {
        if !predicate.is_none_or(|predicate| {
            eval_json_tape_item_predicate_cached(tape, item_idx, predicate, predicate_cache)
        }) {
            return Ok(());
        }
        if let Some(idx) = suffix_cache.index(tape, item_idx, suffix_steps) {
            fold_json_tape_numeric(
                json_tape_scalar(tape, idx),
                op,
                &mut acc_i,
                &mut acc_f,
                &mut floated,
                &mut min_f,
                &mut max_f,
                &mut n_obs,
            );
        }
        Ok(())
    });

    crate::exec::pipeline::num_finalise(op, acc_i, acc_f, floated, min_f, max_f, n_obs)
}
#[allow(clippy::too_many_arguments)]
fn fold_json_tape_numeric(
    value: crate::util::JsonView<'_>,
    op: crate::exec::pipeline::NumOp,
    acc_i: &mut i64,
    acc_f: &mut f64,
    floated: &mut bool,
    min_f: &mut f64,
    max_f: &mut f64,
    n_obs: &mut usize,
) {
    match value {
        crate::util::JsonView::Int(value) => crate::exec::pipeline::num_fold_i64(
            acc_i, acc_f, floated, min_f, max_f, n_obs, op, value,
        ),
        crate::util::JsonView::UInt(value) if value <= i64::MAX as u64 => {
            crate::exec::pipeline::num_fold_i64(
                acc_i,
                acc_f,
                floated,
                min_f,
                max_f,
                n_obs,
                op,
                value as i64,
            )
        }
        crate::util::JsonView::UInt(value) => crate::exec::pipeline::num_fold_f64(
            acc_i,
            acc_f,
            floated,
            min_f,
            max_f,
            n_obs,
            op,
            value as f64,
        ),
        crate::util::JsonView::Float(value) => crate::exec::pipeline::num_fold_f64(
            acc_i, acc_f, floated, min_f, max_f, n_obs, op, value,
        ),
        _ => {}
    }
}
fn eval_json_tape_item_predicate_cached<T: JsonTape>(
    tape: &T,
    item_idx: usize,
    predicate: &NdjsonDirectItemPredicate,
    cache: &mut NdjsonPathCache,
) -> bool {
    use crate::parse::ast::BinOp;

    match predicate {
        NdjsonDirectItemPredicate::Path(steps) => cache
            .index(tape, item_idx, steps)
            .map(|idx| json_view_truthy(json_tape_scalar(tape, idx)))
            .unwrap_or(false),
        NdjsonDirectItemPredicate::Literal(value) => crate::util::is_truthy(value),
        NdjsonDirectItemPredicate::Binary { lhs, op, rhs } if *op == BinOp::And => {
            eval_json_tape_item_predicate_cached(tape, item_idx, lhs, cache)
                && eval_json_tape_item_predicate_cached(tape, item_idx, rhs, cache)
        }
        NdjsonDirectItemPredicate::Binary { lhs, op, rhs } if *op == BinOp::Or => {
            eval_json_tape_item_predicate_cached(tape, item_idx, lhs, cache)
                || eval_json_tape_item_predicate_cached(tape, item_idx, rhs, cache)
        }
        NdjsonDirectItemPredicate::Binary { lhs, op, rhs } => {
            let Some(lhs) = eval_json_tape_item_scalar_cached(tape, item_idx, lhs, cache) else {
                return false;
            };
            let Some(rhs) = eval_json_tape_item_scalar_cached(tape, item_idx, rhs, cache) else {
                return false;
            };
            crate::util::json_cmp_binop(lhs, *op, rhs)
        }
        NdjsonDirectItemPredicate::CmpLit { lhs, op, lit } => cache
            .index(tape, item_idx, lhs)
            .map(|idx| json_tape_scalar(tape, idx))
            .is_some_and(|value| {
                crate::util::json_cmp_binop(value, *op, crate::util::JsonView::from_val(lit))
            }),
        NdjsonDirectItemPredicate::ViewScalarCall { suffix_steps, call } => cache
            .index(tape, item_idx, suffix_steps)
            .map(|idx| json_tape_scalar(tape, idx))
            .and_then(|value| call.try_apply_json_view(value))
            .is_some_and(|value| crate::util::is_truthy(&value)),
    }
}
fn eval_json_tape_item_scalar_cached<'a, T: JsonTape>(
    tape: &'a T,
    item_idx: usize,
    predicate: &'a NdjsonDirectItemPredicate,
    cache: &mut NdjsonPathCache,
) -> Option<crate::util::JsonView<'a>> {
    match predicate {
        NdjsonDirectItemPredicate::Path(steps) => cache
            .index(tape, item_idx, steps)
            .map(|idx| json_tape_scalar(tape, idx)),
        NdjsonDirectItemPredicate::Literal(value) => Some(crate::util::JsonView::from_val(value)),
        _ => None,
    }
}

fn write_json_array<'a, W, I>(writer: &mut W, items: I) -> Result<(), JetroEngineError>
where
    W: Write,
    I: IntoIterator<Item = &'a Val>,
{
    writer.write_all(b"[")?;
    let mut first = true;
    for item in items {
        if first {
            first = false;
        } else {
            writer.write_all(b",")?;
        }
        write_val_json(writer, item)?;
    }
    writer.write_all(b"]")?;
    Ok(())
}

fn write_json_int_array<W, I>(writer: &mut W, items: I) -> Result<(), JetroEngineError>
where
    W: Write,
    I: IntoIterator<Item = i64>,
{
    writer.write_all(b"[")?;
    let mut first = true;
    let mut buf = itoa::Buffer::new();
    for item in items {
        if first {
            first = false;
        } else {
            writer.write_all(b",")?;
        }
        writer.write_all(buf.format(item).as_bytes())?;
    }
    writer.write_all(b"]")?;
    Ok(())
}

fn write_json_float_array<W, I>(writer: &mut W, items: I) -> Result<(), JetroEngineError>
where
    W: Write,
    I: IntoIterator<Item = f64>,
{
    writer.write_all(b"[")?;
    let mut first = true;
    let mut buf = ryu::Buffer::new();
    for item in items {
        if first {
            first = false;
        } else {
            writer.write_all(b",")?;
        }
        if item.is_finite() {
            writer.write_all(buf.format(item).as_bytes())?;
        } else {
            writer.write_all(b"0")?;
        }
    }
    writer.write_all(b"]")?;
    Ok(())
}

fn write_json_str_array<'a, W, I>(writer: &mut W, items: I) -> Result<(), JetroEngineError>
where
    W: Write,
    I: IntoIterator<Item = &'a str>,
{
    writer.write_all(b"[")?;
    let mut first = true;
    for item in items {
        if first {
            first = false;
        } else {
            writer.write_all(b",")?;
        }
        write_json_str(writer, item)?;
    }
    writer.write_all(b"]")?;
    Ok(())
}

fn write_json_object<'a, W, I>(writer: &mut W, entries: I) -> Result<(), JetroEngineError>
where
    W: Write,
    I: IntoIterator<Item = (&'a str, &'a Val)>,
{
    writer.write_all(b"{")?;
    let mut first = true;
    for (key, value) in entries {
        if first {
            first = false;
        } else {
            writer.write_all(b",")?;
        }
        write_json_str(writer, key)?;
        writer.write_all(b":")?;
        write_val_json(writer, value)?;
    }
    writer.write_all(b"}")?;
    Ok(())
}

fn write_json_objvec<W: Write>(
    writer: &mut W,
    data: &crate::data::value::ObjVecData,
) -> Result<(), JetroEngineError> {
    writer.write_all(b"[")?;
    for row in 0..data.nrows() {
        if row > 0 {
            writer.write_all(b",")?;
        }
        writer.write_all(b"{")?;
        for slot in 0..data.stride() {
            if slot > 0 {
                writer.write_all(b",")?;
            }
            write_json_str(writer, data.keys[slot].as_ref())?;
            writer.write_all(b":")?;
            write_val_json(writer, data.cell(row, slot))?;
        }
        writer.write_all(b"}")?;
    }
    writer.write_all(b"]")?;
    Ok(())
}

pub(super) fn write_json_str<W: Write>(
    writer: &mut W,
    value: &str,
) -> Result<(), JetroEngineError> {
    writer.write_all(b"\"")?;
    let bytes = value.as_bytes();
    if !needs_json_escape(bytes) {
        writer.write_all(bytes)?;
        writer.write_all(b"\"")?;
        return Ok(());
    }

    let mut start = 0usize;

    for (idx, &byte) in bytes.iter().enumerate() {
        let escaped = match byte {
            b'"' => Some(br#"\""#.as_slice()),
            b'\\' => Some(br#"\\"#.as_slice()),
            b'\n' => Some(br#"\n"#.as_slice()),
            b'\r' => Some(br#"\r"#.as_slice()),
            b'\t' => Some(br#"\t"#.as_slice()),
            0x08 => Some(br#"\b"#.as_slice()),
            0x0c => Some(br#"\f"#.as_slice()),
            0x00..=0x1f => None,
            _ => continue,
        };

        if start < idx {
            writer.write_all(&bytes[start..idx])?;
        }
        match escaped {
            Some(seq) => writer.write_all(seq)?,
            None => write_control_escape(writer, byte)?,
        }
        start = idx + 1;
    }

    if start < bytes.len() {
        writer.write_all(&bytes[start..])?;
    }
    writer.write_all(b"\"")?;
    Ok(())
}

#[inline]
pub(super) fn write_i64<W: Write>(writer: &mut W, value: i64) -> Result<(), JetroEngineError> {
    let mut buf = itoa::Buffer::new();
    writer.write_all(buf.format(value).as_bytes())?;
    Ok(())
}

#[inline]
fn write_u64<W: Write>(writer: &mut W, value: u64) -> Result<(), JetroEngineError> {
    let mut buf = itoa::Buffer::new();
    writer.write_all(buf.format(value).as_bytes())?;
    Ok(())
}

#[inline]
fn write_f64<W: Write>(writer: &mut W, value: f64) -> Result<(), JetroEngineError> {
    if value.is_finite() {
        let mut buf = ryu::Buffer::new();
        writer.write_all(buf.format(value).as_bytes())?;
    } else {
        writer.write_all(b"0")?;
    }
    Ok(())
}

#[inline]
fn needs_json_escape(bytes: &[u8]) -> bool {
    bytes
        .iter()
        .any(|byte| matches!(byte, b'"' | b'\\' | 0x00..=0x1f))
}

fn write_control_escape<W: Write>(writer: &mut W, byte: u8) -> Result<(), JetroEngineError> {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    writer.write_all(&[
        b'\\',
        b'u',
        b'0',
        b'0',
        HEX[(byte >> 4) as usize],
        HEX[(byte & 0x0f) as usize],
    ])?;
    Ok(())
}

pub(super) fn trim_line_ending(buf: &mut Vec<u8>) {
    while matches!(buf.last(), Some(b'\n' | b'\r')) {
        buf.pop();
    }
}

pub(super) fn strip_initial_bom(line_no: u64, buf: &mut Vec<u8>) {
    if line_no == 1 && buf.starts_with(&[0xEF, 0xBB, 0xBF]) {
        buf.drain(..3);
    }
}

pub(super) fn non_ws_range(buf: &[u8]) -> (usize, usize) {
    let start = buf
        .iter()
        .position(|b| !b.is_ascii_whitespace())
        .unwrap_or(buf.len());
    let end = buf
        .iter()
        .rposition(|b| !b.is_ascii_whitespace())
        .map(|idx| idx + 1)
        .unwrap_or(start);
    (start, end)
}

#[cfg(test)]
mod tests {
    #[test]
    fn rows_stream_driver_reports_direct_stage_stats() {
        let engine = crate::JetroEngine::new();
        let plan = super::super::ndjson_rows::ndjson_rows_stream_plan(
            "$.rows().filter($.active == true).distinct_by($.id).take(2).map($.id)",
        )
        .unwrap()
        .unwrap();
        let input = std::io::Cursor::new(
            br#"{"id":"a","active":false}
{"id":"a","active":true}
{"id":"b","active":true}
not-json
"#
            .to_vec(),
        );
        let mut out = Vec::new();

        let (emitted, stats) = super::drive_ndjson_rows_stream_reader_with_stats(
            &engine,
            input,
            &plan,
            None,
            super::NdjsonOptions::default(),
            &mut out,
        )
        .unwrap();

        assert_eq!(emitted, 2);
        assert_eq!(String::from_utf8(out).unwrap(), "\"a\"\n\"b\"\n");
        assert_eq!(stats.source, super::RowStreamSourceKind::NdjsonRows);
        assert_eq!(stats.direction, super::RowStreamDirection::Forward);
        assert_eq!(stats.rows_scanned, 3);
        assert_eq!(stats.rows_filtered, 1);
        assert_eq!(stats.rows_emitted, 2);
        assert_eq!(stats.direct_filter_rows, 3);
        assert_eq!(stats.direct_key_rows, 2);
        assert_eq!(stats.direct_project_rows, 2);
    }

    #[test]
    fn rows_stream_driver_filters_array_find_on_byte_predicate() {
        let engine = crate::JetroEngine::new();
        let plan = super::super::ndjson_rows::ndjson_rows_stream_plan(
            r#"$.rows().filter(@.custom_attributes.find(@.value == "z")).map($.id)"#,
        )
        .unwrap()
        .unwrap();
        let input = std::io::Cursor::new(
            br#"{"id":"a","custom_attributes":[{"value":"x"}]}
{"id":"b","custom_attributes":[{"value":"z"}]}
{"id":"c","custom_attributes":[{"value":null}]}
"#
            .to_vec(),
        );
        let mut out = Vec::new();

        let (emitted, stats) = super::drive_ndjson_rows_stream_reader_with_stats(
            &engine,
            input,
            &plan,
            None,
            super::NdjsonOptions::default(),
            &mut out,
        )
        .unwrap();

        assert_eq!(emitted, 1);
        assert_eq!(String::from_utf8(out).unwrap(), "\"b\"\n");
        assert_eq!(stats.rows_scanned, 3);
        assert_eq!(stats.rows_filtered, 2);
        assert_eq!(stats.direct_filter_rows, 3);
        assert_eq!(stats.fallback_filter_rows, 0);
    }

    #[test]
    fn parse_row_keeps_simd_document_lazy() {
        let engine = crate::JetroEngine::new();
        let row = br#"{"name":"Ada","age":30}"#.to_vec();

        let document = super::parse_row(&engine, 1, row).expect("row parses lazily");

        assert!(!document.root_val_is_materialized());
        assert!(!document.tape_is_built());
    }

    #[test]
    fn owned_row_read_preserves_reusable_buffer_capacity() {
        let input = std::io::Cursor::new(b"{\"n\":1}\n{\"n\":2}\n");
        let mut driver = super::NdjsonPerRowDriver::new(input);
        let mut buf = Vec::with_capacity(128);

        let first = driver
            .read_next_owned(&mut buf)
            .expect("row read succeeds")
            .expect("first row exists");
        assert_eq!(first.1, br#"{"n":1}"#);
        assert_eq!(buf.capacity(), 128);

        let second = driver
            .read_next_owned(&mut buf)
            .expect("row read succeeds")
            .expect("second row exists");
        assert_eq!(second.1, br#"{"n":2}"#);
        assert_eq!(buf.capacity(), 128);
    }

    #[test]
    fn direct_tape_plan_accepts_first_suffix() {
        let engine = crate::JetroEngine::new();
        for query in [
            "attributes.first().value",
            "attributes.last().value",
            "attributes.nth(1).value",
        ] {
            let plan =
                super::direct_tape_plan(&engine, query).expect("array suffix should be direct");
            assert!(matches!(
                plan,
                super::NdjsonDirectTapePlan::ArrayElementPath { .. }
            ));
        }
    }

    #[test]
    fn direct_tape_plan_accepts_rooted_bench_shapes() {
        let engine = crate::JetroEngine::new();
        for query in [
            "$.id",
            "$.a.b.c",
            "$.meta.id",
            "$.name",
            "$.attributes.len()",
            "$.store.attributes.len()",
            "$.attributes.map(@.key)",
            "$.attributes.first().value",
            "$.store.attributes.first().value",
            "$.attributes.last().value",
            "$.name.upper()",
            "$.store.name.upper()",
            "$.attributes.map([@.key, @.value])",
            r#"$.attributes.filter(@.value.contains("_3")).len()"#,
            "$.keys()",
        ] {
            super::direct_tape_plan(&engine, query)
                .unwrap_or_else(|| panic!("{query} should have a direct NDJSON tape plan"));
        }
    }

    #[test]
    fn direct_writer_plan_kind_exposes_hot_path_selection() {
        let engine = crate::JetroEngine::new();
        use super::NdjsonDirectPlanKind::{
            ByteExpr, TapeArrayProjection, TapeObjectProjection, TapeRootPath, TapeStreamCollect,
            TapeStreamCount, TapeStreamExtreme, TapeStreamFirst, TapeStreamLast, TapeStreamNumeric,
        };

        for (query, expected) in [
            ("$.name", (Some(ByteExpr), TapeRootPath)),
            ("$.a.b.c", (Some(ByteExpr), TapeRootPath)),
            (r#"{test: $.a.b.c, b: $.a.b}"#, (None, TapeObjectProjection)),
            (r#"[$.id, $.name]"#, (None, TapeArrayProjection)),
            ("$.attributes.map(@.key)", (None, TapeStreamCollect)),
            (
                "$.attributes.map({k: @.key, code: @.meta.code.upper()})",
                (None, TapeStreamCollect),
            ),
            (
                r#"$.attributes.filter(@.value.contains("_3")).map({key: @.key, value: @.value}).first()"#,
                (None, TapeStreamFirst),
            ),
            ("$.attributes.map(@.value).last()", (None, TapeStreamLast)),
            (
                r#"$.attributes.filter(@.value.contains("_3")).len()"#,
                (None, TapeStreamCount),
            ),
            (
                "$.attributes.map(@.weight).sum()",
                (None, TapeStreamNumeric),
            ),
            (
                r#"$.attributes.filter(@.value.contains("_3")).map(@.weight).sum()"#,
                (None, TapeStreamNumeric),
            ),
            (
                "$.attributes.sort_by(@.value).last().key",
                (None, TapeStreamExtreme),
            ),
        ] {
            let actual = super::direct_writer_plan_kind(&engine, query)
                .unwrap_or_else(|| panic!("{query} should have an observable direct plan"));
            assert_eq!(actual, expected, "{query}");
        }
    }

    #[test]
    fn direct_writer_path_kind_matches_runtime_writer_family() {
        let engine = crate::JetroEngine::new();
        use super::NdjsonWriterPathKind::{ByteExpr, ByteWritableTape};

        for (query, expected) in [
            ("$.name", ByteExpr),
            ("$.a.b.c", ByteExpr),
            (r#"{test: $.a.b.c, b: $.a.b}"#, ByteWritableTape),
            (r#"[$.id, $.name]"#, ByteWritableTape),
            ("$.attributes.map(@.key)", ByteWritableTape),
            (
                "$.attributes.map({k: @.key, code: @.meta.code.upper()})",
                ByteWritableTape,
            ),
            (
                r#"$.attributes.filter(@.value.contains("_3")).len()"#,
                ByteWritableTape,
            ),
            ("$.attributes.map(@.weight).sum()", ByteWritableTape),
            (
                r#"$.attributes.filter(@.value.contains("_3")).map(@.weight).sum()"#,
                ByteWritableTape,
            ),
            (
                r#"{id: $.id, name: $.name, count: $.attributes.len()}"#,
                ByteWritableTape,
            ),
            (
                r#"[$.id, $.name, $.attributes.first().value, $.attributes.last().value]"#,
                ByteWritableTape,
            ),
            (
                r#"{name_upper: $.name.upper(), values: $.attributes.map(@.value), last: $.attributes.last().value}"#,
                ByteWritableTape,
            ),
            ("$.attributes.sort_by(@.value).last().key", ByteWritableTape),
        ] {
            assert_eq!(
                super::direct_writer_path_kind(&engine, query),
                Some(expected),
                "{query}"
            );
        }
    }

    #[test]
    fn public_writer_path_kind_reports_direct_family() {
        let engine = crate::JetroEngine::new();
        let kind = crate::io::ndjson_writer_path_kind(&engine, "$.name").unwrap();
        assert_eq!(kind, super::NdjsonWriterPathKind::ByteExpr);
        assert_eq!(kind.to_string(), "byte-expr");
    }

    #[test]
    fn direct_tape_plan_lowers_stream_shapes_generically() {
        let engine = crate::JetroEngine::new();
        for query in [
            "$.attributes.map(@.key)",
            "$.attributes.map(@.key.upper())",
            "$.attributes.map(@.value).first()",
            "$.attributes.map(@.value).last()",
            r#"$.attributes.filter(@.value.contains("_3")).map(@.key)"#,
            r#"$.attributes.filter(@.value.contains("_3")).filter(@.weight > 10).map(@.key)"#,
            r#"$.attributes.filter(@.value.contains("_3")).map(@.key.upper())"#,
            r#"$.attributes.filter(@.value.contains("_3")).map({key: @.key, value: @.value}).first()"#,
            r#"$.attributes.filter(@.value.contains("_3")).len()"#,
            r#"$.attributes.filter(@.value.contains("_3")).filter(@.weight > 10).len()"#,
            "$.attributes.map(@.weight).sum()",
            r#"$.attributes.filter(@.value.contains("_3")).map(@.weight).sum()"#,
            r#"$.attributes.filter(@.value.contains("_3")).filter(@.weight > 10).map(@.weight).sum()"#,
            "$.attributes.sort_by(@.value).last().key",
        ] {
            let plan =
                super::direct_tape_plan(&engine, query).expect("query should be direct NDJSON");
            assert!(
                matches!(plan, super::NdjsonDirectTapePlan::Stream(_)),
                "{query} should lower to a generic NDJSON stream plan"
            );
        }
    }

    #[test]
    fn direct_byte_plan_accepts_fast_root_shapes() {
        let engine = crate::JetroEngine::new();
        for query in [
            "$.id",
            "$.name",
            "$.name.upper()",
            "$.name.lower()",
            "$.keys()",
            "$.meta.keys()",
            "$.values()",
            "$.entries()",
            "$.attributes.first().value",
            "$.store.attributes.first().value",
            "$.attributes.first().key.upper()",
            "$.attributes.last().value",
            "$.attributes.nth(1).value",
        ] {
            super::direct_byte_plan(&engine, query)
                .unwrap_or_else(|| panic!("{query} should have a direct NDJSON byte plan"));
        }
    }

    #[test]
    fn direct_byte_predicates_cover_match_shapes() {
        let engine = crate::JetroEngine::new();
        let row = br#"{"active":true,"score":9910,"attributes":[{"key":"k1","value":"v_1"}]}"#;
        for predicate in [
            ("active", true),
            ("score > 9900", true),
            ("score < 100", false),
            (r#"attributes.first().value.contains("_1")"#, true),
        ] {
            let plan = super::direct_tape_predicate(&engine, predicate.0)
                .unwrap_or_else(|| panic!("{} should have a direct predicate", predicate.0));
            let matched = super::eval_ndjson_byte_predicate_row(row, &plan)
                .expect("byte predicate should evaluate")
                .unwrap_or_else(|| panic!("{} should not need tape fallback", predicate.0));
            assert_eq!(matched, predicate.1, "{}", predicate.0);
        }
    }

    #[test]
    fn direct_byte_predicate_covers_array_find_field_comparison() {
        let row = br#"{"custom_attributes":[{"attribute_name":"a","value":"x"},{"attribute_name":"b","value":"z"},{"attribute_name":"c","value":null},{"attribute_name":"d","value":""}]}"#;
        for (predicate, expected) in [
            (r#"@.custom_attributes.find(@.value == "z")"#, true),
            (r#"@.custom_attributes.find(value == "missing")"#, false),
            (r#"@.custom_attributes.find(@.value == null)"#, true),
            (r#"@.custom_attributes.find(@.value == "")"#, true),
        ] {
            let expr = crate::parse::parser::parse(predicate).expect("parse");
            let plan = super::super::ndjson_direct::direct_tape_predicate_for_expr(&expr)
                .unwrap_or_else(|| panic!("{predicate} should have a direct predicate"));
            let matched = super::eval_ndjson_byte_predicate_row(row, &plan)
                .expect("byte predicate should evaluate")
                .unwrap_or_else(|| panic!("{predicate} should not need tape fallback"));
            assert_eq!(matched, expected, "{predicate}");
        }
    }

    #[test]
    fn direct_byte_tape_plan_counts_filtered_rows() {
        let engine = crate::JetroEngine::new();
        let row =
            br#"{"attributes":[{"value":"a_3","weight":2},{"value":"b","weight":7},{"value":"c_3","weight":9}]}"#;
        for (query, expected) in [
            (r#"attributes.filter(@.value.contains("_3")).len()"#, "2"),
            (
                r#"attributes.filter(@.value.contains("_3")).filter(@.weight > 5).len()"#,
                "1",
            ),
            ("attributes.len()", "3"),
        ] {
            let plan =
                super::direct_tape_plan(&engine, query).expect("filter count should be direct");
            assert!(super::tape_plan_can_write_byte_row(&plan), "{query}");

            let mut out = Vec::new();
            let mut scratch = Vec::new();
            let wrote = super::write_ndjson_byte_tape_plan_row(&mut out, row, &plan, &mut scratch)
                .expect("byte count should write");
            assert!(matches!(wrote, super::BytePlanWrite::Done), "{query}");
            assert_eq!(std::str::from_utf8(&out).unwrap(), expected, "{query}");
        }
    }

    #[test]
    fn direct_byte_tape_plan_reduces_numeric_streams() {
        let engine = crate::JetroEngine::new();
        let row = br#"{"attributes":[{"weight":1},{"weight":2.5},{"weight":3},{"weight":"skip"}]}"#;
        for (query, expected) in [
            ("$.attributes.map(@.weight).sum()", "6.5"),
            ("$.attributes.map(@.weight).avg()", "2.1666666666666665"),
            ("$.attributes.map(@.weight).min()", "1.0"),
            ("$.attributes.map(@.weight).max()", "3.0"),
        ] {
            let plan = super::direct_tape_plan(&engine, query)
                .unwrap_or_else(|| panic!("{query} should be direct"));
            assert!(
                super::tape_plan_can_write_byte_row(&plan),
                "{query} should be byte-writable"
            );
            let mut out = Vec::new();
            let mut scratch = Vec::new();
            let wrote = super::write_ndjson_byte_tape_plan_row(&mut out, row, &plan, &mut scratch)
                .expect("byte numeric stream should write");
            assert!(matches!(wrote, super::BytePlanWrite::Done), "{query}");
            assert_eq!(std::str::from_utf8(&out).unwrap(), expected, "{query}");
        }
    }

    #[test]
    fn direct_byte_tape_plan_collects_stream_maps() {
        let engine = crate::JetroEngine::new();
        let row = br#"{"attributes":[{"key":"k1","value":"v1"},{"key":"k2","value":"v2"}]}"#;
        for (query, expected) in [
            ("attributes.map(@.key)", r#"["k1","k2"]"#),
            (
                "attributes.map([@.key, @.value])",
                r#"[["k1","v1"],["k2","v2"]]"#,
            ),
            (
                "attributes.map({key: @.key, value: @.value})",
                r#"[{"key":"k1","value":"v1"},{"key":"k2","value":"v2"}]"#,
            ),
            ("attributes.map(@.key.upper())", r#"["K1","K2"]"#),
            (
                r#"attributes.filter(@.value.contains("2")).map(@.key)"#,
                r#"["k2"]"#,
            ),
            (
                r#"attributes.filter(@.value.contains("2")).filter(@.key == "k2").map(@.key)"#,
                r#"["k2"]"#,
            ),
            (
                r#"attributes.filter(@.value.contains("2")).map({key: @.key, value: @.value})"#,
                r#"[{"key":"k2","value":"v2"}]"#,
            ),
            (
                r#"attributes.filter(@.key != "k1").map([@.key, @.value])"#,
                r#"[["k2","v2"]]"#,
            ),
        ] {
            let plan = super::direct_tape_plan(&engine, query)
                .unwrap_or_else(|| panic!("{query} should be direct"));
            assert!(
                super::tape_plan_can_write_byte_row(&plan),
                "{query} should be byte-writable"
            );
            let mut out = Vec::new();
            let mut scratch = Vec::new();
            let wrote = super::write_ndjson_byte_tape_plan_row(&mut out, row, &plan, &mut scratch)
                .expect("byte stream should write");
            assert!(matches!(wrote, super::BytePlanWrite::Done), "{query}");
            assert_eq!(std::str::from_utf8(&out).unwrap(), expected, "{query}");
        }
    }

    #[test]
    fn direct_byte_tape_plan_matches_engine_for_multi_filter_streams() {
        let engine = crate::JetroEngine::new();
        let row = br#"{"attributes":[{"key":"k1","value":"v1","weight":2},{"key":"k2","value":"v2","weight":7},{"key":"k3","value":"v2","weight":11}]}"#;
        for query in [
            r#"$.attributes.filter(@.value.contains("2")).filter(@.weight > 5).map(@.key)"#,
            r#"$.attributes.filter(@.value.contains("2")).filter(@.weight > 5).len()"#,
            r#"$.attributes.filter(@.value.contains("2")).filter(@.weight > 5).map(@.weight).sum()"#,
            r#"$.attributes.filter(@.value.contains("2")).filter(@.weight > 5).map({key: @.key, weight: @.weight}).last()"#,
        ] {
            let plan = super::direct_tape_plan(&engine, query)
                .unwrap_or_else(|| panic!("{query} should be direct"));
            assert!(
                super::tape_plan_can_write_byte_row(&plan),
                "{query} should be byte-writable"
            );
            let mut out = Vec::new();
            let mut scratch = Vec::new();
            let wrote = super::write_ndjson_byte_tape_plan_row(&mut out, row, &plan, &mut scratch)
                .expect("byte stream should write");
            assert!(matches!(wrote, super::BytePlanWrite::Done), "{query}");

            let direct: serde_json::Value =
                serde_json::from_slice(&out).unwrap_or_else(|err| panic!("{query}: {err}"));
            let engine_value = crate::Jetro::from_bytes(row.to_vec())
                .unwrap()
                .collect(query.to_string())
                .unwrap_or_else(|err| panic!("{query}: {}", err.0));
            let expected: serde_json::Value =
                serde_json::from_str(&engine_value.to_string()).unwrap();
            assert_eq!(direct, expected, "{query}");
        }
    }

    #[test]
    fn direct_byte_tape_plan_writes_static_projections() {
        let engine = crate::JetroEngine::new();
        let row = br#"{"id":7,"a":{"b":{"c":1}}}"#;
        for (query, expected) in [
            ("$.a.b.c", "1"),
            (r#"{test: $.a.b.c, b: $.a.b}"#, r#"{"test":1,"b":{"c":1}}"#),
            (r#"[$.a.b.c, $.id]"#, r#"[1,7]"#),
        ] {
            let plan = super::direct_tape_plan(&engine, query)
                .unwrap_or_else(|| panic!("{query} should be direct"));
            assert!(
                super::tape_plan_can_write_byte_row(&plan),
                "{query} should be byte-writable"
            );
            let mut out = Vec::new();
            let mut scratch = Vec::new();
            let wrote = super::write_ndjson_byte_tape_plan_row(&mut out, row, &plan, &mut scratch)
                .expect("byte projection should write");
            assert!(matches!(wrote, super::BytePlanWrite::Done), "{query}");
            assert_eq!(std::str::from_utf8(&out).unwrap(), expected, "{query}");
        }
    }

    #[test]
    fn run_ndjson_uses_byte_paths_for_nested_object_items() {
        let engine = crate::JetroEngine::new();
        let rows = std::io::Cursor::new(
            br#"{"id":1}
{"id":2}
"#,
        );
        let mut out = Vec::new();
        engine
            .run_ndjson(rows, "$.id", &mut out)
            .expect("rooted byte path should run");
        assert_eq!(std::str::from_utf8(&out).unwrap(), "1\n2\n");

        let rows = std::io::Cursor::new(
            br#"{"meta":{"id":1,"kind":"a"}}
{"meta":{"id":2,"kind":"b"}}
"#,
        );

        let mut out = Vec::new();
        engine
            .run_ndjson(rows, "$.meta.id", &mut out)
            .expect("nested byte path should run");
        assert_eq!(std::str::from_utf8(&out).unwrap(), "1\n2\n");

        let rows = std::io::Cursor::new(br#"{"meta":{"id":1,"kind":"a"}}"#);
        let mut out = Vec::new();
        engine
            .run_ndjson(rows, "$.meta.keys()", &mut out)
            .expect("nested byte object items should run");
        assert_eq!(std::str::from_utf8(&out).unwrap(), "[\"id\",\"kind\"]\n");
    }

    #[test]
    fn direct_byte_projection_calls_match_engine_results() {
        let engine = crate::JetroEngine::new();
        let row = br#"{"id":7,"name":"Ada","tags":["sf","hugo"],"meta":{"id":1,"kind":"a"}}"#;
        for query in ["$.name.len()", "$.name.upper()", "$.name.lower()"] {
            let plan = super::direct_tape_plan(&engine, query)
                .unwrap_or_else(|| panic!("{query} should have a direct tape plan"));
            assert!(
                super::tape_plan_can_write_byte_row(&plan),
                "{query} should be byte-writable"
            );

            let mut direct = Vec::new();
            let mut scratch = Vec::new();
            let wrote =
                super::write_ndjson_byte_tape_plan_row(&mut direct, row, &plan, &mut scratch)
                    .expect("direct byte writer should run");
            assert!(matches!(wrote, super::BytePlanWrite::Done), "{query}");

            let direct: serde_json::Value =
                serde_json::from_slice(&direct).unwrap_or_else(|err| panic!("{query}: {err}"));
            let engine_value = crate::Jetro::from_bytes(row.to_vec())
                .unwrap()
                .collect(query.to_string())
                .unwrap_or_else(|err| panic!("{query}: {}", err.0));
            let expected: serde_json::Value =
                serde_json::from_str(&engine_value.to_string()).unwrap();
            assert_eq!(direct, expected, "{query}");
        }

        for query in ["$.meta.keys()", "$.meta.values()", "$.meta.entries()"] {
            let mut direct = Vec::new();
            engine
                .run_ndjson(std::io::Cursor::new(row), query, &mut direct)
                .unwrap_or_else(|err| panic!("{query}: {err}"));
            let direct: serde_json::Value =
                serde_json::from_slice(direct.trim_ascii_end())
                    .unwrap_or_else(|err| panic!("{query}: {err}"));
            let engine_value = crate::Jetro::from_bytes(row.to_vec())
                .unwrap()
                .collect(query.to_string())
                .unwrap_or_else(|err| panic!("{query}: {}", err.0));
            let expected: serde_json::Value =
                serde_json::from_str(&engine_value.to_string()).unwrap();
            assert_eq!(direct, expected, "{query}");
        }
    }

    #[test]
    fn run_ndjson_uses_byte_paths_for_nested_array_demands() {
        let engine = crate::JetroEngine::new();
        let rows = std::io::Cursor::new(
            br#"{"store":{"attributes":[{"value":"a"},{"value":"b"}],"after":1}}
{"store":{"attributes":[{"value":"c"},{"value":"d"}],"after":2}}
"#,
        );

        let mut out = Vec::new();
        engine
            .run_ndjson(rows, "$.store.attributes.first().value", &mut out)
            .expect("nested byte array demand should run");
        assert_eq!(std::str::from_utf8(&out).unwrap(), "\"a\"\n\"c\"\n");

        out.clear();
        let rows = std::io::Cursor::new(
            br#"{"store":{"attributes":[{"value":"a"},{"value":"b"}],"after":1}}
{"store":{"attributes":[{"value":"c"},{"value":"d"}],"after":2}}
"#,
        );
        engine
            .run_ndjson(rows, "$.store.attributes.last().value", &mut out)
            .expect("nested byte last demand should run from field prefix");
        assert_eq!(std::str::from_utf8(&out).unwrap(), "\"b\"\n\"d\"\n");
    }

    #[test]
    fn run_ndjson_static_projection_survives_hint_activation() {
        let engine = crate::JetroEngine::new();
        let rows = std::io::Cursor::new(
            br#"{"id":1,"name":"a","active":true}
{"id":2,"name":"b","active":true}
{"id":3,"name":"c","active":true}
{"id":4,"name":"d","active":true}
{"id":5,"name":"e","active":true}
{"id":6,"name":"f","active":true}
{"id":7,"name":"g","active":true}
{"id":8,"name":"h","active":true}
{"id":9,"name":"i","active":true}
"#,
        );
        let mut out = Vec::new();
        engine
            .run_ndjson(rows, r#"{id: $.id, name: $.name}"#, &mut out)
            .expect("hinted static projection should run");
        assert_eq!(
            std::str::from_utf8(&out).unwrap(),
            "{\"id\":1,\"name\":\"a\"}\n\
{\"id\":2,\"name\":\"b\"}\n\
{\"id\":3,\"name\":\"c\"}\n\
{\"id\":4,\"name\":\"d\"}\n\
{\"id\":5,\"name\":\"e\"}\n\
{\"id\":6,\"name\":\"f\"}\n\
{\"id\":7,\"name\":\"g\"}\n\
{\"id\":8,\"name\":\"h\"}\n\
{\"id\":9,\"name\":\"i\"}\n"
        );
    }

    #[test]
    fn run_ndjson_nested_projection_survives_hint_activation() {
        let engine = crate::JetroEngine::new();
        let rows = std::io::Cursor::new(
            br#"{"id":1,"profile":{"name":"a","score":10},"active":true}
{"id":2,"profile":{"name":"b","score":20},"active":true}
{"id":3,"profile":{"name":"c","score":30},"active":true}
{"id":4,"profile":{"name":"d","score":40},"active":true}
{"id":5,"profile":{"name":"e","score":50},"active":true}
{"id":6,"profile":{"name":"f","score":60},"active":true}
{"id":7,"profile":{"name":"g","score":70},"active":true}
{"id":8,"profile":{"name":"h","score":80},"active":true}
{"id":9,"profile":{"name":"i","score":90},"active":true}
"#,
        );
        let mut out = Vec::new();
        engine
            .run_ndjson(
                rows,
                r#"{id: $.id, name: $.profile.name, profile: $.profile}"#,
                &mut out,
            )
            .expect("hinted nested projection should run");
        assert_eq!(
            std::str::from_utf8(&out).unwrap(),
            "{\"id\":1,\"name\":\"a\",\"profile\":{\"name\":\"a\",\"score\":10}}\n\
{\"id\":2,\"name\":\"b\",\"profile\":{\"name\":\"b\",\"score\":20}}\n\
{\"id\":3,\"name\":\"c\",\"profile\":{\"name\":\"c\",\"score\":30}}\n\
{\"id\":4,\"name\":\"d\",\"profile\":{\"name\":\"d\",\"score\":40}}\n\
{\"id\":5,\"name\":\"e\",\"profile\":{\"name\":\"e\",\"score\":50}}\n\
{\"id\":6,\"name\":\"f\",\"profile\":{\"name\":\"f\",\"score\":60}}\n\
{\"id\":7,\"name\":\"g\",\"profile\":{\"name\":\"g\",\"score\":70}}\n\
{\"id\":8,\"name\":\"h\",\"profile\":{\"name\":\"h\",\"score\":80}}\n\
{\"id\":9,\"name\":\"i\",\"profile\":{\"name\":\"i\",\"score\":90}}\n"
        );
    }

    #[test]
    fn run_ndjson_scalar_projection_survives_hint_activation() {
        let engine = crate::JetroEngine::new();
        let rows = std::io::Cursor::new(
            br#"{"id":1,"profile":{"name":"a"}}
{"id":2,"profile":{"name":"b"}}
{"id":3,"profile":{"name":"c"}}
{"id":4,"profile":{"name":"d"}}
{"id":5,"profile":{"name":"e"}}
{"id":6,"profile":{"name":"f"}}
{"id":7,"profile":{"name":"g"}}
{"id":8,"profile":{"name":"h"}}
{"id":9,"profile":{"name":"i"}}
"#,
        );
        let mut out = Vec::new();
        engine
            .run_ndjson(
                rows,
                r#"{id: $.id, name: $.profile.name.upper()}"#,
                &mut out,
            )
            .expect("hinted scalar projection should run");
        assert_eq!(
            std::str::from_utf8(&out).unwrap(),
            "{\"id\":1,\"name\":\"A\"}\n\
{\"id\":2,\"name\":\"B\"}\n\
{\"id\":3,\"name\":\"C\"}\n\
{\"id\":4,\"name\":\"D\"}\n\
{\"id\":5,\"name\":\"E\"}\n\
{\"id\":6,\"name\":\"F\"}\n\
{\"id\":7,\"name\":\"G\"}\n\
{\"id\":8,\"name\":\"H\"}\n\
{\"id\":9,\"name\":\"I\"}\n"
        );
    }

    #[test]
    fn run_ndjson_stream_collect_survives_hint_activation() {
        let engine = crate::JetroEngine::new();
        let rows = std::io::Cursor::new(
            br#"{"id":1,"attributes":[{"key":"a","value":"x"},{"key":"b","value":"y"}]}
{"id":2,"attributes":[{"key":"c","value":"z"}]}
{"id":3,"attributes":[{"key":"d","value":"w"}]}
"#,
        );
        let mut out = Vec::new();
        engine
            .run_ndjson(rows, "$.attributes.map([@.key, @.value])", &mut out)
            .expect("hinted stream collect should run");
        assert_eq!(
            std::str::from_utf8(&out).unwrap(),
            "[[\"a\",\"x\"],[\"b\",\"y\"]]\n[[\"c\",\"z\"]]\n[[\"d\",\"w\"]]\n"
        );
    }

    #[test]
    fn run_ndjson_stream_cache_rejects_reordered_item_prefixes() {
        let engine = crate::JetroEngine::new();
        let rows = std::io::Cursor::new(
            br#"{"attributes":[{"key":"k1","value":"a"}]}
{"attributes":[{"value":"k1","key":"actual"}]}
"#,
        );
        let mut out = Vec::new();
        engine
            .run_ndjson(rows, "$.attributes.map(@.key)", &mut out)
            .expect("stream cache should fall back on reordered item fields");
        assert_eq!(
            std::str::from_utf8(&out).unwrap(),
            "[\"k1\"]\n[\"actual\"]\n"
        );
    }

    #[test]
    fn run_ndjson_stream_map_preserves_missing_field_nulls() {
        let engine = crate::JetroEngine::new();
        let rows = std::io::Cursor::new(
            br#"{"attributes":[{"key":"a","value":"x"},{"key":"b"}]}
{"attributes":[{"value":"z"}]}
"#,
        );
        let mut out = Vec::new();
        engine
            .run_ndjson(rows, "$.attributes.map([@.key, @.value])", &mut out)
            .expect("stream map should preserve nulls for missing fields");
        assert_eq!(
            std::str::from_utf8(&out).unwrap(),
            "[[\"a\",\"x\"],[\"b\",null]]\n[[null,\"z\"]]\n"
        );
    }

    #[test]
    fn run_ndjson_stream_object_map_preserves_scalar_calls() {
        let engine = crate::JetroEngine::new();
        let rows = std::io::Cursor::new(
            br#"{"attributes":[{"key":"a","value":"x"},{"key":"b","value":"y"}]}
"#,
        );
        let mut out = Vec::new();
        engine
            .run_ndjson(
                rows,
                "$.attributes.map({k: @.key, v: @.value.upper()})",
                &mut out,
            )
            .expect("stream object map should preserve scalar calls");
        assert_eq!(
            std::str::from_utf8(&out).unwrap(),
            "[{\"k\":\"a\",\"v\":\"X\"},{\"k\":\"b\",\"v\":\"Y\"}]\n"
        );
    }

    #[test]
    fn run_ndjson_stream_map_projects_nested_item_paths() {
        let engine = crate::JetroEngine::new();
        let rows = std::io::Cursor::new(
            br#"{"attributes":[{"key":"a","meta":{"code":"x"}},{"key":"b","meta":{"code":"y"}}]}
"#,
        );
        let mut out = Vec::new();
        engine
            .run_ndjson(
                rows,
                "$.attributes.map({k: @.key, code: @.meta.code.upper()})",
                &mut out,
            )
            .expect("stream map should project nested item paths");
        assert_eq!(
            std::str::from_utf8(&out).unwrap(),
            "[{\"k\":\"a\",\"code\":\"X\"},{\"k\":\"b\",\"code\":\"Y\"}]\n"
        );
    }

    #[test]
    fn run_ndjson_stream_count_survives_hint_activation() {
        let engine = crate::JetroEngine::new();
        let rows = std::io::Cursor::new(
            br#"{"id":1,"attributes":[{"value":"x_3"},{"value":"y"}]}
{"id":2,"attributes":[{"value":"z_3"},{"value":"w_3"}]}
{"id":3,"attributes":[{"value":"n"}]}
"#,
        );
        let mut out = Vec::new();
        engine
            .run_ndjson(
                rows,
                r#"$.attributes.filter(@.value.contains("_3")).len()"#,
                &mut out,
            )
            .expect("hinted stream count should run");
        assert_eq!(std::str::from_utf8(&out).unwrap(), "1\n2\n0\n");
    }

    #[test]
    fn run_ndjson_filtered_count_ignores_missing_predicate_fields() {
        let engine = crate::JetroEngine::new();
        let rows = std::io::Cursor::new(
            br#"{"attributes":[{"value":"x_3"},{"key":"missing"},{"value":"y"}]}
{"attributes":[{"key":"missing"}]}
"#,
        );
        let mut out = Vec::new();
        engine
            .run_ndjson(
                rows,
                r#"$.attributes.filter(@.value.contains("_3")).len()"#,
                &mut out,
            )
            .expect("filtered count should ignore missing predicate fields");
        assert_eq!(std::str::from_utf8(&out).unwrap(), "1\n0\n");
    }

    #[test]
    fn run_ndjson_filter_last_returns_last_matching_output() {
        let engine = crate::JetroEngine::new();
        let rows = std::io::Cursor::new(
            br#"{"attributes":[{"key":"keep","value":"first"},{"key":"drop","value":"physical-last"}]}
{"attributes":[{"key":"drop","value":"first"},{"key":"keep","value":"semantic-last"},{"key":"drop","value":"physical-last"}]}
"#,
        );
        let mut out = Vec::new();

        engine
            .run_ndjson(
                rows,
                r#"$.attributes.filter(@.key == "keep").last().value"#,
                &mut out,
            )
            .expect("filtered last should preserve semantic output order");

        assert_eq!(
            std::str::from_utf8(&out).unwrap(),
            "\"first\"\n\"semantic-last\"\n"
        );
    }

    #[test]
    fn run_ndjson_filter_map_first_stops_at_first_matching_output() {
        let engine = crate::JetroEngine::new();
        let rows = std::io::Cursor::new(
            br#"{"attributes":[{"key":"a","value":"x_3"},{"key":"b","value":"later_3"}]}
{"attributes":[{"key":"a","value":"skip"},{"key":"b","value":"y_3"}]}
{"attributes":[{"key":"a","value":"skip"}]}
"#,
        );
        let mut out = Vec::new();

        engine
            .run_ndjson(
                rows,
                r#"$.attributes.filter(@.value.contains("_3")).map({key: @.key, value: @.value}).first()"#,
                &mut out,
            )
            .expect("filtered first should use direct stream first");

        assert_eq!(
            std::str::from_utf8(&out).unwrap(),
            "{\"key\":\"a\",\"value\":\"x_3\"}\n{\"key\":\"b\",\"value\":\"y_3\"}\n"
        );
    }

    #[test]
    fn run_ndjson_map_first_projects_first_item_without_filter() {
        let engine = crate::JetroEngine::new();
        let rows = std::io::Cursor::new(
            br#"{"attributes":[{"key":"a","value":"first"},{"key":"b","value":"later"}]}
{"attributes":[]}
{"attributes":[{"key":"c","value":"only"}]}
"#,
        );
        let mut out = Vec::new();

        engine
            .run_ndjson(rows, "$.attributes.map(@.value).first()", &mut out)
            .expect("unfiltered first should use direct stream first");

        assert_eq!(std::str::from_utf8(&out).unwrap(), "\"first\"\n\"only\"\n");
    }

    #[test]
    fn run_ndjson_map_last_projects_last_item_without_filter() {
        let engine = crate::JetroEngine::new();
        let rows = std::io::Cursor::new(
            br#"{"attributes":[{"key":"a","value":"first"},{"key":"b","value":"last"}]}
{"attributes":[]}
{"attributes":[{"key":"c","value":"only"}]}
"#,
        );
        let mut out = Vec::new();

        engine
            .run_ndjson(rows, "$.attributes.map(@.value).last()", &mut out)
            .expect("unfiltered last should use direct stream last");

        assert_eq!(std::str::from_utf8(&out).unwrap(), "\"last\"\n\"only\"\n");
    }

    #[test]
    fn run_ndjson_filter_map_last_keeps_latest_matching_output() {
        let engine = crate::JetroEngine::new();
        let rows = std::io::Cursor::new(
            br#"{"attributes":[{"key":"a","value":"x_3"},{"key":"b","value":"later_3"}]}
{"attributes":[{"key":"a","value":"skip"},{"key":"b","value":"y_3"}]}
{"attributes":[{"key":"a","value":"skip"}]}
"#,
        );
        let mut out = Vec::new();

        engine
            .run_ndjson(
                rows,
                r#"$.attributes.filter(@.value.contains("_3")).map({key: @.key, value: @.value}).last()"#,
                &mut out,
            )
            .expect("filtered last should use direct stream last");

        assert_eq!(
            std::str::from_utf8(&out).unwrap(),
            "{\"key\":\"b\",\"value\":\"later_3\"}\n{\"key\":\"b\",\"value\":\"y_3\"}\n"
        );
    }

    #[test]
    fn run_ndjson_stream_numeric_survives_hint_activation() {
        let engine = crate::JetroEngine::new();
        let rows = std::io::Cursor::new(
            br#"{"id":1,"attributes":[{"weight":1},{"weight":2}]}
{"id":2,"attributes":[{"weight":3.5},{"weight":4}]}
{"id":3,"attributes":[{"weight":"skip"}]}
"#,
        );
        let mut out = Vec::new();
        engine
            .run_ndjson(rows, "$.attributes.map(@.weight).sum()", &mut out)
            .expect("hinted numeric stream should run");
        assert_eq!(std::str::from_utf8(&out).unwrap(), "3\n7.5\n0\n");
    }

    #[test]
    fn run_ndjson_filtered_stream_numeric_uses_shared_fields() {
        let engine = crate::JetroEngine::new();
        let rows = std::io::Cursor::new(
            br#"{"attributes":[{"value":"x_3","weight":1},{"value":"skip","weight":10},{"value":"y_3","weight":2.5}]}
{"attributes":[{"value":"skip","weight":4}]}
"#,
        );
        let mut out = Vec::new();
        engine
            .run_ndjson(
                rows,
                r#"$.attributes.filter(@.value.contains("_3")).map(@.weight).sum()"#,
                &mut out,
            )
            .expect("filtered numeric stream should use byte path");
        assert_eq!(std::str::from_utf8(&out).unwrap(), "3.5\n0\n");
    }

    #[test]
    fn run_ndjson_stream_extreme_projects_selected_item_field() {
        let engine = crate::JetroEngine::new();
        let rows = std::io::Cursor::new(
            br#"{"attributes":[{"key":"a","value":"m"},{"key":"b","value":"z"},{"key":"c","value":"n"}]}
{"attributes":[{"key":"x","value":"b"},{"key":"y","value":"a"}]}
"#,
        );
        let mut out = Vec::new();
        engine
            .run_ndjson(rows, "$.attributes.sort_by(@.value).last().key", &mut out)
            .expect("stream extreme should project selected item field");
        assert_eq!(std::str::from_utf8(&out).unwrap(), "\"b\"\n\"x\"\n");
    }

    #[test]
    fn run_ndjson_stream_extreme_handles_escaped_string_keys() {
        let engine = crate::JetroEngine::new();
        let rows = std::io::Cursor::new(
            br#"{"attributes":[{"key":"a","value":"v\"1"},{"key":"b","value":"v_9"}]}
"#,
        );
        let mut out = Vec::new();
        engine
            .run_ndjson(rows, "$.attributes.sort_by(@.value).last().key", &mut out)
            .expect("escaped extrema keys should fall back safely");
        assert_eq!(std::str::from_utf8(&out).unwrap(), "\"b\"\n");
    }

    #[test]
    fn run_ndjson_stream_extreme_handles_numeric_keys() {
        let engine = crate::JetroEngine::new();
        let rows = std::io::Cursor::new(
            br#"{"attributes":[{"key":"a","score":1},{"key":"b","score":10},{"key":"c","score":2}]}
{"attributes":[{"key":"x","score":-2},{"key":"y","score":-1.5}]}
"#,
        );
        let mut out = Vec::new();
        engine
            .run_ndjson(rows, "$.attributes.sort_by(@.score).last().key", &mut out)
            .expect("numeric extrema keys should use direct stream extrema");
        assert_eq!(std::str::from_utf8(&out).unwrap(), "\"b\"\n\"y\"\n");
    }

    #[test]
    fn run_ndjson_nested_direct_projection_writes_without_fallback() {
        let engine = crate::JetroEngine::new();
        let rows = std::io::Cursor::new(
            br#"{"id":1,"name":"ada","attributes":[{"key":"a","value":"x"},{"key":"b","value":"y"}]}
{"id":2,"name":"bob","attributes":[{"key":"c","value":"z"}]}
"#,
        );
        let mut out = Vec::new();
        engine
            .run_ndjson(
                rows,
                r#"{id: $.id, name: $.name, count: $.attributes.len(), first: $.attributes.first().value, values: $.attributes.map(@.value)}"#,
                &mut out,
            )
            .expect("nested direct projection should run");
        assert_eq!(
            std::str::from_utf8(&out).unwrap(),
            "{\"id\":1,\"name\":\"ada\",\"count\":2,\"first\":\"x\",\"values\":[\"x\",\"y\"]}\n\
{\"id\":2,\"name\":\"bob\",\"count\":1,\"first\":\"z\",\"values\":[\"z\"]}\n"
        );
    }
}
