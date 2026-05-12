use super::{NdjsonSource, RowError};
use crate::data::value::{Val, ValRef};
use crate::plan::physical::PlanningContext;
use crate::{Jetro, JetroEngine, JetroEngineError};
use memchr::memchr;
use serde_json::Value;
use std::fs::File;
use std::io::{BufRead, BufWriter, Write};
use std::path::Path;

const DEFAULT_MAX_LINE_LEN: usize = 64 * 1024 * 1024;
const DEFAULT_LINE_BUFFER_CAPACITY: usize = 8192;
const DEFAULT_READER_BUFFER_CAPACITY: usize = 64 * 1024;
pub(super) const DEFAULT_REVERSE_CHUNK_SIZE: usize = 64 * 1024;

/// Configuration for per-row NDJSON execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NdjsonOptions {
    pub max_line_len: usize,
    pub initial_buffer_capacity: usize,
    pub reader_buffer_capacity: usize,
    pub reverse_chunk_size: usize,
}

impl Default for NdjsonOptions {
    fn default() -> Self {
        Self {
            max_line_len: DEFAULT_MAX_LINE_LEN,
            initial_buffer_capacity: DEFAULT_LINE_BUFFER_CAPACITY,
            reader_buffer_capacity: DEFAULT_READER_BUFFER_CAPACITY,
            reverse_chunk_size: DEFAULT_REVERSE_CHUNK_SIZE,
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
            let read = self.read_physical_line(buf)?;
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
            let read = self.read_physical_line(buf)?;
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

            let capacity = buf.capacity();
            return Ok(Some((
                self.line_no,
                std::mem::replace(buf, Vec::with_capacity(capacity)),
            )));
        }
    }

    fn read_physical_line(&mut self, buf: &mut Vec<u8>) -> Result<usize, RowError> {
        loop {
            let available = self.reader.fill_buf()?;
            if available.is_empty() {
                return Ok(buf.len());
            }

            if let Some(pos) = memchr(b'\n', available) {
                buf.extend_from_slice(&available[..=pos]);
                self.reader.consume(pos + 1);
                self.check_physical_line_len(buf.len())?;
                return Ok(buf.len());
            }

            let len = available.len();
            buf.extend_from_slice(available);
            self.reader.consume(len);
            self.check_physical_line_len(buf.len())?;
        }
    }

    fn check_physical_line_len(&self, len: usize) -> Result<(), RowError> {
        let hard_max = self.max_line_len.saturating_add(2);
        if len > hard_max {
            return Err(RowError::LineTooLarge {
                line_no: self.line_no + 1,
                len,
                max: self.max_line_len,
            });
        }
        Ok(())
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
    let file = File::open(path)?;
    let options = NdjsonOptions::default();
    run_ndjson_with_options(
        engine,
        std::io::BufReader::with_capacity(options.reader_buffer_capacity, file),
        query,
        writer,
        options,
    )
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
    let file = File::open(path)?;
    run_ndjson_with_options(
        engine,
        std::io::BufReader::with_capacity(options.reader_buffer_capacity, file),
        query,
        writer,
        options,
    )
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
    let count = drive_ndjson_val(engine, reader, query, options, |value| {
        serde_json::to_writer(&mut writer, &ValRef(&value))?;
        writer.write_all(b"\n")?;
        Ok(NdjsonControl::Continue)
    })?;
    writer.flush()?;
    Ok(count)
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
    let mut driver = NdjsonPerRowDriver::new(reader).with_max_line_len(options.max_line_len);
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

fn drive_ndjson_val<R, F>(
    engine: &JetroEngine,
    reader: R,
    query: &str,
    options: NdjsonOptions,
    mut emit: F,
) -> Result<usize, JetroEngineError>
where
    R: BufRead,
    F: FnMut(Val) -> Result<NdjsonControl, JetroEngineError>,
{
    let mut driver = NdjsonPerRowDriver::new(reader).with_max_line_len(options.max_line_len);
    let plan = engine.cached_plan(query, PlanningContext::bytes());
    let mut buf = Vec::with_capacity(options.initial_buffer_capacity);
    let mut count = 0;

    while let Some((line_no, row)) = driver.read_next_owned(&mut buf)? {
        let document = parse_row(engine, line_no, row)?;
        count += 1;
        if matches!(
            emit(collect_row_val(engine, &document, &plan, line_no)?)?,
            NdjsonControl::Stop
        ) {
            break;
        }
    }

    Ok(count)
}

pub(super) fn collect_row_val(
    engine: &JetroEngine,
    document: &Jetro,
    plan: &crate::ir::physical::QueryPlan,
    line_no: u64,
) -> Result<Val, JetroEngineError> {
    engine
        .collect_prepared_val(document, plan)
        .map_err(|err| row_eval_error(line_no, err))
}

pub(super) fn parse_row(
    engine: &JetroEngine,
    line_no: u64,
    row: Vec<u8>,
) -> Result<Jetro, JetroEngineError> {
    engine
        .parse_bytes_lazy(row)
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

fn row_eval_error(line_no: u64, err: crate::EvalError) -> JetroEngineError {
    let message = err.0;
    if message.starts_with("Invalid JSON:") {
        RowError::InvalidJsonMessage { line_no, message }.into()
    } else {
        crate::EvalError(message).into()
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

#[cfg(test)]
mod tests {
    #[test]
    #[cfg(feature = "simd-json")]
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
}
