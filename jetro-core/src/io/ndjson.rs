use super::RowError;
use crate::plan::physical::PlanningContext;
use crate::{Jetro, JetroEngine, JetroEngineError};
use serde_json::Value;
use std::io::{BufRead, BufWriter, Write};

const DEFAULT_MAX_LINE_LEN: usize = 64 * 1024 * 1024;

/// Configuration for per-row NDJSON execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NdjsonOptions {
    pub max_line_len: usize,
}

impl Default for NdjsonOptions {
    fn default() -> Self {
        Self {
            max_line_len: DEFAULT_MAX_LINE_LEN,
        }
    }
}

impl NdjsonOptions {
    pub fn with_max_line_len(mut self, max_line_len: usize) -> Self {
        self.max_line_len = max_line_len;
        self
    }
}

/// Forward-only per-row NDJSON reader.
pub struct NdjsonPerRowDriver<R> {
    reader: R,
    line_no: u64,
    max_line_len: usize,
}

impl<R: BufRead> NdjsonPerRowDriver<R> {
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            line_no: 0,
            max_line_len: DEFAULT_MAX_LINE_LEN,
        }
    }

    pub fn with_max_line_len(mut self, max_line_len: usize) -> Self {
        self.max_line_len = max_line_len;
        self
    }

    pub fn line_no(&self) -> u64 {
        self.line_no
    }

    /// Read the next non-empty NDJSON row into `buf`, returning its 1-based line
    /// number. Empty and whitespace-only rows are skipped.
    pub fn read_next_nonempty<'a>(
        &mut self,
        buf: &'a mut Vec<u8>,
    ) -> Result<Option<(u64, &'a [u8])>, RowError> {
        loop {
            buf.clear();
            let read = self.reader.read_until(b'\n', buf)?;
            if read == 0 {
                return Ok(None);
            }
            self.line_no += 1;

            strip_initial_bom(self.line_no, buf);
            trim_line_ending(buf);

            let (start, end) = non_ws_range(buf);
            if start == end {
                continue;
            }

            let len = end - start;
            if len > self.max_line_len {
                return Err(RowError::LineTooLarge {
                    line_no: self.line_no,
                    len,
                    max: self.max_line_len,
                });
            }

            return Ok(Some((self.line_no, &buf[start..end])));
        }
    }

    /// Read the next non-empty row and transfer ownership of `buf` to the
    /// caller. This is the hot path used by `JetroEngine` NDJSON execution so
    /// the row can be parsed without an extra bytes copy.
    pub fn read_next_owned(
        &mut self,
        buf: &mut Vec<u8>,
    ) -> Result<Option<(u64, Vec<u8>)>, RowError> {
        loop {
            buf.clear();
            let read = self.reader.read_until(b'\n', buf)?;
            if read == 0 {
                return Ok(None);
            }
            self.line_no += 1;

            strip_initial_bom(self.line_no, buf);
            trim_line_ending(buf);

            let (start, end) = non_ws_range(buf);
            if start == end {
                continue;
            }

            let len = end - start;
            if len > self.max_line_len {
                return Err(RowError::LineTooLarge {
                    line_no: self.line_no,
                    len,
                    max: self.max_line_len,
                });
            }

            return Ok(Some((self.line_no, std::mem::take(buf))));
        }
    }
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
        Ok(())
    })
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
    let mut writer = BufWriter::new(writer);
    let count = drive_ndjson(engine, reader, query, options, |value| {
        serde_json::to_writer(&mut writer, &value)?;
        writer.write_all(b"\n")?;
        Ok(())
    })?;
    writer.flush()?;
    Ok(count)
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
    F: FnMut(Value) -> Result<(), JetroEngineError>,
{
    let mut driver = NdjsonPerRowDriver::new(reader).with_max_line_len(options.max_line_len);
    let plan = engine.cached_plan(query, PlanningContext::bytes());
    let mut buf = Vec::with_capacity(8192);
    let mut count = 0;

    while let Some((line_no, row)) = driver.read_next_owned(&mut buf)? {
        let document = parse_row(engine, line_no, row)?;
        let out = engine.collect_prepared(&document, &plan)?;
        emit(out)?;
        count += 1;
    }

    Ok(count)
}

fn parse_row(engine: &JetroEngine, line_no: u64, row: Vec<u8>) -> Result<Jetro, JetroEngineError> {
    engine
        .parse_bytes(row)
        .map_err(|err| row_parse_error(line_no, err))
}

fn row_parse_error(line_no: u64, err: JetroEngineError) -> JetroEngineError {
    match err {
        JetroEngineError::Json(source) => RowError::InvalidJson { line_no, source }.into(),
        JetroEngineError::Eval(eval) => RowError::InvalidJsonMessage {
            line_no,
            message: eval.to_string(),
        }
        .into(),
        other => other,
    }
}

fn trim_line_ending(buf: &mut Vec<u8>) {
    while matches!(buf.last(), Some(b'\n' | b'\r')) {
        buf.pop();
    }
}

fn strip_initial_bom(line_no: u64, buf: &mut Vec<u8>) {
    if line_no == 1 && buf.starts_with(&[0xEF, 0xBB, 0xBF]) {
        buf.drain(..3);
    }
}

fn non_ws_range(buf: &[u8]) -> (usize, usize) {
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
