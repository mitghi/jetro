use super::ndjson_byte::{
    raw_json_byte_path_value, tape_plan_can_write_byte_row, write_ndjson_byte_tape_plan_row,
    BytePlanWrite, RawFieldValue,
};
use super::ndjson_distinct::{
    distinct_key_bytes, raw_distinct_key_bytes, AdaptiveDistinctKeys, DistinctFrontFilterKind,
};
use super::ndjson_frame::{frame_payload, FramePayload, NdjsonRowFrame};
use super::ndjson_route::{
    NdjsonExecutionReport, NdjsonExecutionStats, NdjsonRouteExplain, NdjsonRouteKind,
    NdjsonSourceCaps,
};
use super::RowError;
use crate::util::is_truthy;
use crate::{JetroEngine, JetroEngineError};
use memchr::memrchr;
use serde_json::Value;
use std::borrow::Cow;
use std::collections::VecDeque;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

/// Reverse NDJSON line reader over a seekable file.
///
/// The reader scans fixed-size chunks from EOF to BOF and returns owned line
/// bytes in reverse physical order. It keeps only the current chunk and one
/// cross-chunk carry buffer, so memory stays bounded by the longest row plus
/// the configured chunk size.
pub struct NdjsonReverseFileDriver {
    file: File,
    pos: u64,
    chunk_size: usize,
    max_line_len: usize,
    row_frame: NdjsonRowFrame,
    carry: Vec<u8>,
    pending: VecDeque<Vec<u8>>,
    finished_head: bool,
    reverse_line_no: u64,
}

impl NdjsonReverseFileDriver {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, RowError> {
        Self::with_options(path, super::ndjson::NdjsonOptions::default())
    }

    pub fn with_chunk_size<P: AsRef<Path>>(path: P, chunk_size: usize) -> Result<Self, RowError> {
        Self::with_options(
            path,
            super::ndjson::NdjsonOptions::default().with_reverse_chunk_size(chunk_size),
        )
    }

    pub fn with_options<P: AsRef<Path>>(
        path: P,
        options: super::ndjson::NdjsonOptions,
    ) -> Result<Self, RowError> {
        let mut file = File::open(path)?;
        let pos = file.seek(SeekFrom::End(0))?;
        Ok(Self {
            file,
            pos,
            chunk_size: options.reverse_chunk_size.max(1),
            max_line_len: options.max_line_len,
            row_frame: options.row_frame,
            carry: Vec::new(),
            pending: VecDeque::new(),
            finished_head: false,
            reverse_line_no: 0,
        })
    }

    pub fn next_line(&mut self) -> Result<Option<Vec<u8>>, RowError> {
        Ok(self.next_line_with_reverse_no()?.map(|(_, line)| line))
    }

    pub fn next_line_with_reverse_no(&mut self) -> Result<Option<(u64, Vec<u8>)>, RowError> {
        loop {
            if let Some(mut line) = self.pending.pop_front() {
                self.reverse_line_no += 1;
                if let Some(line) = self.frame_line(self.reverse_line_no, &mut line)? {
                    return Ok(Some((self.reverse_line_no, line)));
                }
                continue;
            }

            if self.pos == 0 {
                if self.finished_head || self.carry.is_empty() {
                    return Ok(None);
                }
                self.finished_head = true;
                let mut line = std::mem::take(&mut self.carry);
                trim_line_ending(&mut line);
                self.check_line_len(line.len())?;
                if line.iter().any(|b| !b.is_ascii_whitespace()) {
                    self.reverse_line_no += 1;
                    if let Some(line) = self.frame_line(self.reverse_line_no, &mut line)? {
                        return Ok(Some((self.reverse_line_no, line)));
                    }
                }
                return Ok(None);
            }

            let read_len = self.chunk_size.min(self.pos as usize);
            self.pos -= read_len as u64;
            let mut chunk = vec![0u8; read_len];
            self.file.seek(SeekFrom::Start(self.pos))?;
            self.file.read_exact(&mut chunk)?;

            let mut end = chunk.len();
            while let Some(nl) = memrchr(b'\n', &chunk[..end]) {
                let mut line = Vec::with_capacity(end - nl - 1 + self.carry.len());
                line.extend_from_slice(&chunk[nl + 1..end]);
                line.extend_from_slice(&self.carry);
                self.carry.clear();
                end = nl;
                trim_line_ending(&mut line);
                self.check_line_len(line.len())?;
                if line.iter().any(|b| !b.is_ascii_whitespace()) {
                    self.pending.push_back(line);
                }
            }

            if end > 0 {
                let mut next = Vec::with_capacity(end + self.carry.len());
                next.extend_from_slice(&chunk[..end]);
                next.extend_from_slice(&self.carry);
                self.check_line_len(next.len())?;
                self.carry = next;
            }
        }
    }

    fn frame_line(&self, line_no: u64, line: &mut Vec<u8>) -> Result<Option<Vec<u8>>, RowError> {
        match frame_payload(self.row_frame, line_no, line)? {
            FramePayload::Data(range) => {
                if range.start > 0 || range.end < line.len() {
                    line.copy_within(range.clone(), 0);
                    line.truncate(range.end - range.start);
                }
                Ok(Some(std::mem::take(line)))
            }
            FramePayload::Skip => Ok(None),
        }
    }

    fn check_line_len(&self, len: usize) -> Result<(), RowError> {
        if len > self.max_line_len {
            return Err(RowError::LineTooLarge {
                line_no: self.reverse_line_no + self.pending.len() as u64 + 1,
                len,
                max: self.max_line_len,
            });
        }
        Ok(())
    }
}

pub fn collect_ndjson_rev<P>(
    engine: &JetroEngine,
    path: P,
    query: &str,
) -> Result<Vec<Value>, JetroEngineError>
where
    P: AsRef<Path>,
{
    collect_ndjson_rev_with_options(engine, path, query, super::ndjson::NdjsonOptions::default())
}

pub fn collect_ndjson_rev_with_options<P>(
    engine: &JetroEngine,
    path: P,
    query: &str,
    options: super::ndjson::NdjsonOptions,
) -> Result<Vec<Value>, JetroEngineError>
where
    P: AsRef<Path>,
{
    let mut values = Vec::new();
    drive_rev(engine, path, query, options, |value| {
        values.push(Value::from(value));
        Ok(super::ndjson::NdjsonControl::Continue)
    })?;
    Ok(values)
}

pub fn for_each_ndjson_rev<P, F>(
    engine: &JetroEngine,
    path: P,
    query: &str,
    mut f: F,
) -> Result<usize, JetroEngineError>
where
    P: AsRef<Path>,
    F: FnMut(Value),
{
    for_each_ndjson_rev_with_options(
        engine,
        path,
        query,
        super::ndjson::NdjsonOptions::default(),
        |value| {
            f(value);
            Ok(super::ndjson::NdjsonControl::Continue)
        },
    )
}

pub fn for_each_ndjson_rev_with_options<P, F>(
    engine: &JetroEngine,
    path: P,
    query: &str,
    options: super::ndjson::NdjsonOptions,
    mut f: F,
) -> Result<usize, JetroEngineError>
where
    P: AsRef<Path>,
    F: FnMut(Value) -> Result<super::ndjson::NdjsonControl, JetroEngineError>,
{
    drive_rev(engine, path, query, options, |value| f(Value::from(value)))
}

pub fn collect_ndjson_rev_matches<P>(
    engine: &JetroEngine,
    path: P,
    predicate: &str,
    limit: usize,
) -> Result<Vec<Value>, JetroEngineError>
where
    P: AsRef<Path>,
{
    collect_ndjson_rev_matches_with_options(
        engine,
        path,
        predicate,
        limit,
        super::ndjson::NdjsonOptions::default(),
    )
}

pub fn collect_ndjson_rev_matches_with_options<P>(
    engine: &JetroEngine,
    path: P,
    predicate: &str,
    limit: usize,
    options: super::ndjson::NdjsonOptions,
) -> Result<Vec<Value>, JetroEngineError>
where
    P: AsRef<Path>,
{
    let mut values = Vec::with_capacity(limit);
    drive_rev_matches(engine, path, predicate, limit, options, |value| {
        values.push(Value::from(value));
        Ok(super::ndjson::NdjsonControl::Continue)
    })?;
    Ok(values)
}

pub fn run_ndjson_rev<P, W>(
    engine: &JetroEngine,
    path: P,
    query: &str,
    writer: W,
) -> Result<usize, JetroEngineError>
where
    P: AsRef<Path>,
    W: Write,
{
    run_ndjson_rev_with_options(
        engine,
        path,
        query,
        writer,
        super::ndjson::NdjsonOptions::default(),
    )
}

pub fn run_ndjson_rev_with_options<P, W>(
    engine: &JetroEngine,
    path: P,
    query: &str,
    writer: W,
    options: super::ndjson::NdjsonOptions,
) -> Result<usize, JetroEngineError>
where
    P: AsRef<Path>,
    W: Write,
{
    if let Some(plan) = super::ndjson::direct_tape_plan(engine, query) {
        return drive_rev_writer_tape(engine, path, &plan, None, options, writer);
    }

    let mut writer = super::ndjson::ndjson_writer_with_options(writer, options);
    let mut emitted = 0usize;
    drive_rev(engine, path, query, options, |value| {
        if super::ndjson::write_val_line_with_options(&mut writer, &value, options)? {
            emitted += 1;
        }
        Ok(super::ndjson::NdjsonControl::Continue)
    })?;
    writer.flush()?;
    Ok(emitted)
}

pub fn run_ndjson_rev_limit<P, W>(
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
    run_ndjson_rev_limit_with_options(
        engine,
        path,
        query,
        limit,
        writer,
        super::ndjson::NdjsonOptions::default(),
    )
}

pub fn run_ndjson_rev_limit_with_options<P, W>(
    engine: &JetroEngine,
    path: P,
    query: &str,
    limit: usize,
    writer: W,
    options: super::ndjson::NdjsonOptions,
) -> Result<usize, JetroEngineError>
where
    P: AsRef<Path>,
    W: Write,
{
    if limit == 0 {
        return Ok(0);
    }
    if let Some(plan) = super::ndjson::direct_tape_plan(engine, query) {
        return drive_rev_writer_tape(engine, path, &plan, Some(limit), options, writer);
    }

    let mut writer = super::ndjson::ndjson_writer_with_options(writer, options);
    let mut emitted = 0usize;
    drive_rev(engine, path, query, options, |value| {
        let wrote = super::ndjson::write_val_line_with_options(&mut writer, &value, options)?;
        if wrote {
            emitted += 1;
        }
        Ok(if wrote && emitted >= limit {
            super::ndjson::NdjsonControl::Stop
        } else {
            super::ndjson::NdjsonControl::Continue
        })
    })?;
    writer.flush()?;
    Ok(emitted)
}

pub fn run_ndjson_rev_distinct_by<P, W>(
    engine: &JetroEngine,
    path: P,
    key_query: &str,
    query: &str,
    limit: usize,
    writer: W,
) -> Result<usize, JetroEngineError>
where
    P: AsRef<Path>,
    W: Write,
{
    run_ndjson_rev_distinct_by_with_options(
        engine,
        path,
        key_query,
        query,
        limit,
        writer,
        super::ndjson::NdjsonOptions::default(),
    )
}

pub fn run_ndjson_rev_distinct_by_with_options<P, W>(
    engine: &JetroEngine,
    path: P,
    key_query: &str,
    query: &str,
    limit: usize,
    writer: W,
    options: super::ndjson::NdjsonOptions,
) -> Result<usize, JetroEngineError>
where
    P: AsRef<Path>,
    W: Write,
{
    run_ndjson_rev_distinct_by_with_stats_and_options(
        engine, path, key_query, query, limit, writer, options,
    )
    .map(|stats| stats.emitted)
}

pub fn run_ndjson_rev_distinct_by_with_stats<P, W>(
    engine: &JetroEngine,
    path: P,
    key_query: &str,
    query: &str,
    limit: usize,
    writer: W,
) -> Result<NdjsonRevDistinctStats, JetroEngineError>
where
    P: AsRef<Path>,
    W: Write,
{
    run_ndjson_rev_distinct_by_with_stats_and_options(
        engine,
        path,
        key_query,
        query,
        limit,
        writer,
        super::ndjson::NdjsonOptions::default(),
    )
}

pub fn run_ndjson_rev_distinct_by_with_stats_and_options<P, W>(
    engine: &JetroEngine,
    path: P,
    key_query: &str,
    query: &str,
    limit: usize,
    writer: W,
    options: super::ndjson::NdjsonOptions,
) -> Result<NdjsonRevDistinctStats, JetroEngineError>
where
    P: AsRef<Path>,
    W: Write,
{
    if limit == 0 {
        return Ok(NdjsonRevDistinctStats::default());
    }
    let direct_key_plan = super::ndjson::direct_tape_plan(engine, key_query);
    let direct_value_plan = super::ndjson::direct_tape_plan(engine, query)
        .filter(|plan| tape_plan_can_write_byte_row(plan));

    let mut key_plan = None;
    let mut value_plan = None;
    let mut vm = None;
    let mut driver = NdjsonReverseFileDriver::with_options(path, options)?;
    let mut writer = super::ndjson::ndjson_writer_with_options(writer, options);
    let mut byte_scratch = Vec::with_capacity(options.initial_buffer_capacity);
    let mut out = Vec::with_capacity(options.initial_buffer_capacity);
    let mut seen = AdaptiveDistinctKeys::default();
    let mut stats = NdjsonRevDistinctStats::default();

    while let Some((reverse_row_no, row)) = driver.next_line_with_reverse_no()? {
        stats.rows_scanned += 1;
        let mut row = Some(row);
        let mut document = None;
        let direct_key = direct_key_plan.as_ref().and_then(|plan| {
            row.as_deref()
                .and_then(|row| distinct_key_direct(row, plan))
        });

        let inserted = if let Some(key) = direct_key {
            stats.direct_key_rows += 1;
            match key {
                Cow::Borrowed(key) => seen.insert_slice(key),
                Cow::Owned(key) => seen.insert(key),
            }
        } else {
            stats.fallback_key_rows += 1;
            let parsed = super::ndjson::parse_row(engine, reverse_row_no, row.take().unwrap())?;
            let plan = key_plan.get_or_insert_with(|| {
                engine.cached_plan(key_query, crate::plan::physical::PlanningContext::bytes())
            });
            let vm = vm.get_or_insert_with(|| engine.lock_vm());
            let key = crate::exec::router::collect_plan_val_with_vm(&parsed, plan, vm)
                .map_err(|err| super::ndjson::row_eval_error(reverse_row_no, err))?;
            let key = distinct_key_bytes(&key)?;
            document = Some(parsed);
            seen.insert(key)
        };
        if !inserted {
            stats.duplicate_rows += 1;
            continue;
        }
        if let (Some(plan), Some(row)) = (direct_value_plan.as_ref(), row.as_deref()) {
            byte_scratch.clear();
            out.clear();
            match write_ndjson_byte_tape_plan_row(&mut out, row, plan, &mut byte_scratch)? {
                BytePlanWrite::Done => {
                    if super::ndjson::write_json_bytes_line_with_options(
                        &mut writer,
                        &out,
                        options,
                    )? {
                        stats.direct_value_rows += 1;
                        stats.emitted += 1;
                    }
                    if stats.emitted >= limit {
                        break;
                    }
                    continue;
                }
                BytePlanWrite::Fallback => {}
            }
        }

        let parsed = match document {
            Some(document) => document,
            None => super::ndjson::parse_row(engine, reverse_row_no, row.take().unwrap())?,
        };
        let plan = value_plan.get_or_insert_with(|| {
            engine.cached_plan(query, crate::plan::physical::PlanningContext::bytes())
        });
        let vm = vm.get_or_insert_with(|| engine.lock_vm());
        let value = crate::exec::router::collect_plan_val_with_vm(&parsed, plan, vm)
            .map_err(|err| super::ndjson::row_eval_error(reverse_row_no, err))?;
        if super::ndjson::write_val_line_with_options(&mut writer, &value, options)? {
            stats.fallback_value_rows += 1;
            stats.emitted += 1;
        }
        if stats.emitted >= limit {
            break;
        }
    }

    writer.flush()?;
    stats.front_filter = seen.front_kind();
    Ok(stats)
}

pub fn run_ndjson_rev_distinct_by_with_report<P, W>(
    engine: &JetroEngine,
    path: P,
    key_query: &str,
    query: &str,
    limit: usize,
    writer: W,
) -> Result<NdjsonExecutionReport, JetroEngineError>
where
    P: AsRef<Path>,
    W: Write,
{
    run_ndjson_rev_distinct_by_with_report_and_options(
        engine,
        path,
        key_query,
        query,
        limit,
        writer,
        super::ndjson::NdjsonOptions::default(),
    )
}

pub fn run_ndjson_rev_distinct_by_with_report_and_options<P, W>(
    engine: &JetroEngine,
    path: P,
    key_query: &str,
    query: &str,
    limit: usize,
    writer: W,
    options: super::ndjson::NdjsonOptions,
) -> Result<NdjsonExecutionReport, JetroEngineError>
where
    P: AsRef<Path>,
    W: Write,
{
    let stats = run_ndjson_rev_distinct_by_with_stats_and_options(
        engine, path, key_query, query, limit, writer, options,
    )?;
    Ok(NdjsonExecutionReport::new(
        NdjsonRouteExplain {
            kind: NdjsonRouteKind::RowLocal,
            source: NdjsonSourceCaps::file(options),
            writer_path: super::ndjson::ndjson_writer_path_kind(engine, query),
            rows_plan: None,
            fallback_reason: None,
        },
        NdjsonExecutionStats {
            rows_scanned: stats.rows_scanned,
            rows_emitted: stats.emitted,
            duplicate_rows: stats.duplicate_rows,
            direct_key_rows: stats.direct_key_rows,
            fallback_key_rows: stats.fallback_key_rows,
            direct_project_rows: stats.direct_value_rows,
            fallback_project_rows: stats.fallback_value_rows,
            ..NdjsonExecutionStats::default()
        },
    ))
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NdjsonRevDistinctStats {
    pub rows_scanned: usize,
    pub emitted: usize,
    pub duplicate_rows: usize,
    pub direct_key_rows: usize,
    pub fallback_key_rows: usize,
    pub direct_value_rows: usize,
    pub fallback_value_rows: usize,
    pub front_filter: DistinctFrontFilterKind,
}
fn distinct_key_direct<'a>(
    row: &'a [u8],
    plan: &super::ndjson::NdjsonDirectTapePlan,
) -> Option<Cow<'a, [u8]>> {
    const NULL_KEY: &[u8] = b"null";

    let super::ndjson::NdjsonDirectTapePlan::RootPath(steps) = plan else {
        return None;
    };
    match raw_json_byte_path_value(row, steps) {
        RawFieldValue::Found(value) => raw_distinct_key_bytes(value),
        RawFieldValue::Missing => Some(Cow::Borrowed(NULL_KEY)),
        RawFieldValue::Fallback => None,
    }
}

pub fn run_ndjson_rev_matches<P, W>(
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
    run_ndjson_rev_matches_with_options(
        engine,
        path,
        predicate,
        limit,
        writer,
        super::ndjson::NdjsonOptions::default(),
    )
}

pub fn run_ndjson_rev_matches_with_options<P, W>(
    engine: &JetroEngine,
    path: P,
    predicate: &str,
    limit: usize,
    writer: W,
    options: super::ndjson::NdjsonOptions,
) -> Result<usize, JetroEngineError>
where
    P: AsRef<Path>,
    W: Write,
{
    drive_rev_matches_writer(engine, path, predicate, limit, options, writer)
}

pub fn run_ndjson_rev_matches_with_report<P, W>(
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
    run_ndjson_rev_matches_with_report_and_options(
        engine,
        path,
        predicate,
        limit,
        writer,
        super::ndjson::NdjsonOptions::default(),
    )
}

pub fn run_ndjson_rev_matches_with_report_and_options<P, W>(
    engine: &JetroEngine,
    path: P,
    predicate: &str,
    limit: usize,
    writer: W,
    options: super::ndjson::NdjsonOptions,
) -> Result<NdjsonExecutionReport, JetroEngineError>
where
    P: AsRef<Path>,
    W: Write,
{
    let (_, stats) =
        drive_rev_matches_writer_with_stats(engine, path, predicate, limit, options, writer)?;
    Ok(NdjsonExecutionReport::new(
        NdjsonRouteExplain::matches(NdjsonSourceCaps::file(options)),
        stats,
    ))
}
fn drive_rev_writer_tape<P, W>(
    engine: &JetroEngine,
    path: P,
    plan: &super::ndjson::NdjsonDirectTapePlan,
    limit: Option<usize>,
    options: super::ndjson::NdjsonOptions,
    writer: W,
) -> Result<usize, JetroEngineError>
where
    P: AsRef<Path>,
    W: Write,
{
    let mut driver = NdjsonReverseFileDriver::with_options(path, options)?;
    let mut writer = super::ndjson::ndjson_writer_with_options(writer, options);
    let mut scratch =
        crate::data::tape::TapeScratch::with_capacity(options.initial_buffer_capacity);
    let mut out = Vec::with_capacity(options.initial_buffer_capacity);
    let mut runner = super::ndjson::NdjsonTapeWriterRunner::new(engine, plan);
    let mut count = 0usize;

    while let Some((reverse_row_no, row)) = driver.next_line_with_reverse_no()? {
        out.clear();
        scratch.parse_slice(&row).map_err(|message| {
            super::ndjson::row_parse_error(
                reverse_row_no,
                JetroEngineError::Eval(crate::EvalError(format!("Invalid JSON: {message}"))),
            )
        })?;
        runner.write_row(&scratch, &mut out)?;
        if super::ndjson::write_json_bytes_line_with_options(&mut writer, &out, options)? {
            count += 1;
        }
        if limit.is_some_and(|limit| count >= limit) {
            break;
        }
    }

    writer.flush()?;
    Ok(count)
}

fn drive_rev<P, F>(
    engine: &JetroEngine,
    path: P,
    query: &str,
    options: super::ndjson::NdjsonOptions,
    mut emit: F,
) -> Result<usize, JetroEngineError>
where
    P: AsRef<Path>,
    F: FnMut(crate::data::value::Val) -> Result<super::ndjson::NdjsonControl, JetroEngineError>,
{
    let mut driver = NdjsonReverseFileDriver::with_options(path, options)?;
    let mut executor = super::ndjson::NdjsonRowExecutor::new(engine, query);
    let mut count = 0usize;

    while let Some((reverse_row_no, row)) = driver.next_line_with_reverse_no()? {
        let out = executor.eval_owned_row(reverse_row_no, row)?;
        count += 1;
        if matches!(emit(out)?, super::ndjson::NdjsonControl::Stop) {
            break;
        }
    }

    Ok(count)
}

fn drive_rev_matches<P, F>(
    engine: &JetroEngine,
    path: P,
    predicate: &str,
    limit: usize,
    options: super::ndjson::NdjsonOptions,
    mut emit: F,
) -> Result<usize, JetroEngineError>
where
    P: AsRef<Path>,
    F: FnMut(crate::data::value::Val) -> Result<super::ndjson::NdjsonControl, JetroEngineError>,
{
    if limit == 0 {
        return Ok(0);
    }

    let mut driver = NdjsonReverseFileDriver::with_options(path, options)?;
    let mut executor = super::ndjson::NdjsonRowExecutor::new(engine, predicate);
    let mut emitted = 0usize;

    while let Some((reverse_row_no, row)) = driver.next_line_with_reverse_no()? {
        let document = executor.parse_owned_row(reverse_row_no, row)?;
        let matched = executor.eval_document(reverse_row_no, &document)?;
        if !is_truthy(&matched) {
            continue;
        }

        let root = document
            .root_val_with(executor.engine().keys())
            .map_err(|err| super::ndjson::row_eval_error(reverse_row_no, err))?;
        emitted += 1;
        if matches!(emit(root)?, super::ndjson::NdjsonControl::Stop) || emitted >= limit {
            break;
        }
    }

    Ok(emitted)
}

fn drive_rev_matches_writer<P, W>(
    engine: &JetroEngine,
    path: P,
    predicate: &str,
    limit: usize,
    options: super::ndjson::NdjsonOptions,
    writer: W,
) -> Result<usize, JetroEngineError>
where
    P: AsRef<Path>,
    W: Write,
{
    Ok(drive_rev_matches_writer_with_stats(engine, path, predicate, limit, options, writer)?.0)
}

fn drive_rev_matches_writer_with_stats<P, W>(
    engine: &JetroEngine,
    path: P,
    predicate: &str,
    limit: usize,
    options: super::ndjson::NdjsonOptions,
    writer: W,
) -> Result<(usize, NdjsonExecutionStats), JetroEngineError>
where
    P: AsRef<Path>,
    W: Write,
{
    if limit == 0 {
        return Ok((0, NdjsonExecutionStats::default()));
    }
    if let Some(predicate) = super::ndjson::direct_tape_predicate(engine, predicate) {
        return drive_rev_matches_writer_tape_with_stats(
            engine, path, &predicate, limit, options, writer,
        );
    }

    let mut driver = NdjsonReverseFileDriver::with_options(path, options)?;
    let mut executor = super::ndjson::NdjsonRowExecutor::new(engine, predicate);
    let mut writer = super::ndjson::ndjson_writer_with_options(writer, options);
    let mut emitted = 0usize;
    let mut stats = NdjsonExecutionStats::default();

    while let Some((reverse_row_no, row)) = driver.next_line_with_reverse_no()? {
        stats.rows_scanned += 1;
        let document = executor.parse_owned_row(reverse_row_no, row)?;
        let matched = executor.eval_document(reverse_row_no, &document)?;
        stats.fallback_filter_rows += 1;
        if !is_truthy(&matched) {
            stats.rows_filtered += 1;
            continue;
        }

        super::ndjson::write_document_line(
            &mut writer,
            &document,
            reverse_row_no,
            executor.engine(),
        )?;
        emitted += 1;
        stats.rows_emitted += 1;
        if emitted >= limit {
            break;
        }
    }

    writer.flush()?;
    Ok((emitted, stats))
}
fn drive_rev_matches_writer_tape_with_stats<P, W>(
    engine: &JetroEngine,
    path: P,
    predicate: &super::ndjson::NdjsonDirectPredicate,
    limit: usize,
    options: super::ndjson::NdjsonOptions,
    writer: W,
) -> Result<(usize, NdjsonExecutionStats), JetroEngineError>
where
    P: AsRef<Path>,
    W: Write,
{
    let mut driver = NdjsonReverseFileDriver::with_options(path, options)?;
    let mut writer = super::ndjson::ndjson_writer_with_options(writer, options);
    let mut scratch =
        crate::data::tape::TapeScratch::with_capacity(options.initial_buffer_capacity);
    let mut emitted = 0usize;
    let needs_vm = super::ndjson::predicate_needs_vm(predicate);
    let mut vm = needs_vm.then(|| engine.lock_vm());
    let env = needs_vm.then(|| crate::data::context::Env::new(crate::Val::Null));
    let mut predicate_path = super::ndjson::NdjsonPathCache::default();
    let mut stats = NdjsonExecutionStats::default();

    while let Some((reverse_row_no, row)) = driver.next_line_with_reverse_no()? {
        stats.rows_scanned += 1;
        scratch.parse_slice(&row).map_err(|message| {
            super::ndjson::row_parse_error(
                reverse_row_no,
                JetroEngineError::Eval(crate::EvalError(format!("Invalid JSON: {message}"))),
            )
        })?;
        if !super::ndjson::eval_tape_predicate(
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

fn trim_line_ending(buf: &mut Vec<u8>) {
    while matches!(buf.last(), Some(b'\n' | b'\r')) {
        buf.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::NdjsonReverseFileDriver;
    use crate::JetroEngine;
    use std::path::PathBuf;

    #[test]
    fn reverse_driver_reads_rows_from_tail() {
        let path = temp_path("jetro-ndjson-rev-basic");
        std::fs::write(&path, b"{\"n\":1}\n{\"n\":2}\n{\"n\":3}\n").unwrap();
        let mut driver = NdjsonReverseFileDriver::with_chunk_size(&path, 8).unwrap();

        assert_eq!(driver.next_line().unwrap().unwrap(), br#"{"n":3}"#);
        assert_eq!(driver.next_line().unwrap().unwrap(), br#"{"n":2}"#);
        assert_eq!(driver.next_line().unwrap().unwrap(), br#"{"n":1}"#);
        assert!(driver.next_line().unwrap().is_none());

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn reverse_driver_handles_missing_final_newline_and_blank_lines() {
        let path = temp_path("jetro-ndjson-rev-edge");
        std::fs::write(&path, b"\n{\"n\":1}\r\n\n{\"n\":2}").unwrap();
        let mut driver = NdjsonReverseFileDriver::with_chunk_size(&path, 5).unwrap();

        assert_eq!(driver.next_line().unwrap().unwrap(), br#"{"n":2}"#);
        assert_eq!(driver.next_line().unwrap().unwrap(), br#"{"n":1}"#);
        assert!(driver.next_line().unwrap().is_none());

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn reverse_driver_reports_reverse_row_numbers() {
        let path = temp_path("jetro-ndjson-rev-row-no");
        std::fs::write(&path, b"{\"n\":1}\n{\"n\":2}\n").unwrap();
        let mut driver = NdjsonReverseFileDriver::with_chunk_size(&path, 3).unwrap();

        assert_eq!(
            driver.next_line_with_reverse_no().unwrap().unwrap(),
            (1, br#"{"n":2}"#.to_vec())
        );
        assert_eq!(
            driver.next_line_with_reverse_no().unwrap().unwrap(),
            (2, br#"{"n":1}"#.to_vec())
        );
        assert!(driver.next_line_with_reverse_no().unwrap().is_none());

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn reverse_query_uses_direct_writer_shapes() {
        let path = temp_path("jetro-ndjson-rev-direct");
        std::fs::write(
            &path,
            b"{\"name\":\"ada\",\"attrs\":[{\"key\":\"a\",\"value\":1}]}\n{\"name\":\"bob\",\"attrs\":[{\"key\":\"b\",\"value\":2}]}\n",
        )
        .unwrap();
        let engine = JetroEngine::new();
        let mut out = Vec::new();

        super::run_ndjson_rev(&engine, &path, "attrs.map([@.key, @.value])", &mut out).unwrap();

        assert_eq!(
            String::from_utf8(out).unwrap(),
            "[[\"b\",2]]\n[[\"a\",1]]\n"
        );
        let _ = std::fs::remove_file(path);
    }
    #[test]
    fn direct_distinct_key_classifier_rejects_escaped_strings() {
        assert_eq!(
            super::raw_distinct_key_bytes(br#""plain""#).as_deref(),
            Some(br#""plain""#.as_slice())
        );
        assert_eq!(
            super::raw_distinct_key_bytes(br#""a\u0062""#).as_deref(),
            Some(br#""ab""#.as_slice())
        );
        assert_eq!(
            super::raw_distinct_key_bytes(br#"{"k":"v"}"#).as_deref(),
            Some(br#"{"k":"v"}"#.as_slice())
        );
        assert_eq!(
            super::raw_distinct_key_bytes(b"123").as_deref(),
            Some(b"123".as_slice())
        );
        assert_eq!(
            super::raw_distinct_key_bytes(br#"{"a" : 1,"b":"x\u0079"}"#).as_deref(),
            Some(br#"{"a":1,"b":"xy"}"#.as_slice())
        );
        assert_eq!(super::raw_distinct_key_bytes(b"1.0"), None);
    }

    fn temp_path(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("{}-{}.ndjson", name, std::process::id()));
        path
    }
}
