use super::{NdjsonSource, RowError};
use crate::data::value::Val;
use crate::plan::physical::PlanningContext;
use crate::util::is_truthy;
use crate::{Jetro, JetroEngine, JetroEngineError, VM};
use memchr::memchr;
use serde_json::Value;
use std::fs::File;
use std::io::{BufRead, BufWriter, Write};
use std::path::Path;
use std::sync::MutexGuard;

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
    drive_ndjson_writer(engine, reader, query, None, options, writer)
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

    drive_ndjson_writer(engine, reader, query, Some(limit), options, writer)
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
    let file = File::open(path)?;
    let options = NdjsonOptions::default();
    run_ndjson_limit_with_options(
        engine,
        std::io::BufReader::with_capacity(options.reader_buffer_capacity, file),
        query,
        limit,
        writer,
        options,
    )
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
    #[cfg(feature = "simd-json")]
    if let Some(plan) = NdjsonRowExecutor::direct_tape_plan(engine, query) {
        return drive_ndjson_tape_writer(engine, reader, &plan, limit, options, writer);
    }

    let mut driver = NdjsonPerRowDriver::new(reader).with_max_line_len(options.max_line_len);
    let mut executor = NdjsonRowExecutor::new(engine, query);
    let mut writer = ndjson_writer_with_options(writer, options);
    let mut buf = Vec::with_capacity(options.initial_buffer_capacity);
    let mut count = 0usize;

    while let Some((line_no, row)) = driver.read_next_owned(&mut buf)? {
        count += 1;
        executor.write_owned_row(line_no, row, &mut writer)?;
        if limit.is_some_and(|limit| count >= limit) {
            break;
        }
    }

    writer.flush()?;
    Ok(count)
}

#[cfg(feature = "simd-json")]
#[derive(Clone)]
enum NdjsonDirectTapePlan {
    RootPath(Vec<crate::ir::physical::PhysicalPathStep>),
    ViewScalarCall {
        steps: Vec<crate::ir::physical::PhysicalPathStep>,
        call: crate::builtins::BuiltinCall,
        optional: bool,
    },
    ArrayElementPath {
        source_steps: Vec<crate::ir::physical::PhysicalPathStep>,
        element: NdjsonDirectElement,
        suffix_steps: Vec<crate::ir::physical::PhysicalPathStep>,
    },
    ViewPipeline {
        source_steps: Vec<crate::ir::physical::PhysicalPathStep>,
        body: crate::exec::pipeline::PipelineBody,
    },
}

#[cfg(feature = "simd-json")]
#[derive(Clone, Copy)]
pub(super) enum NdjsonDirectElement {
    First,
    Last,
    Nth(usize),
}

#[cfg(feature = "simd-json")]
#[derive(Clone)]
pub(super) enum NdjsonDirectPredicate {
    Path(Vec<crate::ir::physical::PhysicalPathStep>),
    Literal(Val),
    Not(Box<NdjsonDirectPredicate>),
    Binary {
        lhs: Box<NdjsonDirectPredicate>,
        op: crate::parse::ast::BinOp,
        rhs: Box<NdjsonDirectPredicate>,
    },
    ViewScalarCall {
        steps: Vec<crate::ir::physical::PhysicalPathStep>,
        call: crate::builtins::BuiltinCall,
    },
    ArrayElementViewScalarCall {
        source_steps: Vec<crate::ir::physical::PhysicalPathStep>,
        element: NdjsonDirectElement,
        suffix_steps: Vec<crate::ir::physical::PhysicalPathStep>,
        call: crate::builtins::BuiltinCall,
    },
    ViewPipeline {
        source_steps: Vec<crate::ir::physical::PhysicalPathStep>,
        body: crate::exec::pipeline::PipelineBody,
    },
}

#[cfg(feature = "simd-json")]
fn drive_ndjson_tape_writer<R, W>(
    engine: &JetroEngine,
    reader: R,
    plan: &NdjsonDirectTapePlan,
    limit: Option<usize>,
    options: NdjsonOptions,
    writer: W,
) -> Result<usize, JetroEngineError>
where
    R: BufRead,
    W: Write,
{
    let mut driver = NdjsonPerRowDriver::new(reader).with_max_line_len(options.max_line_len);
    let mut writer = ndjson_writer_with_options(writer, options);
    let mut line = Vec::with_capacity(options.initial_buffer_capacity);
    let mut scratch =
        crate::data::tape::TapeScratch::with_capacity(options.initial_buffer_capacity);
    let mut count = 0usize;
    let mut vm = engine.lock_vm();
    let env = crate::data::context::Env::new(Val::Null);

    while let Some((line_no, row)) = driver.read_next_nonempty(&mut line)? {
        scratch.parse_slice(row).map_err(|message| {
            row_parse_error(
                line_no,
                JetroEngineError::Eval(crate::EvalError(format!("Invalid JSON: {message}"))),
            )
        })?;
        match plan {
            NdjsonDirectTapePlan::RootPath(steps) => {
                if let Some(idx) = json_tape_path_index(&scratch, steps) {
                    write_json_tape_at(&mut writer, &scratch, idx)?;
                } else {
                    writer.write_all(b"null")?;
                }
            }
            NdjsonDirectTapePlan::ViewScalarCall {
                steps,
                call,
                optional,
            } => {
                let value = json_tape_path_index(&scratch, steps)
                    .map(|idx| json_tape_scalar(&scratch, idx))
                    .unwrap_or(crate::util::JsonView::Null);
                if *optional && matches!(value, crate::util::JsonView::Null) {
                    writer.write_all(b"null")?;
                } else if let Some(value) = call.try_apply_json_view(value) {
                    write_val_json(&mut writer, &value)?;
                } else {
                    write_json_tape_at(
                        &mut writer,
                        &scratch,
                        json_tape_path_index(&scratch, steps).unwrap_or(usize::MAX),
                    )?;
                }
            }
            NdjsonDirectTapePlan::ArrayElementPath {
                source_steps,
                element,
                suffix_steps,
            } => {
                let idx = json_tape_path_index(&scratch, source_steps)
                    .and_then(|idx| json_tape_array_element(&scratch, idx, *element))
                    .and_then(|idx| json_tape_path_index_from(&scratch, idx, suffix_steps));
                if let Some(idx) = idx {
                    write_json_tape_at(&mut writer, &scratch, idx)?;
                } else {
                    writer.write_all(b"null")?;
                }
            }
            NdjsonDirectTapePlan::ViewPipeline { source_steps, body } => {
                let source = json_tape_path_index(&scratch, source_steps)
                    .map(|idx| crate::data::view::TapeScratchView::Node {
                        tape: &scratch,
                        idx,
                    })
                    .unwrap_or(crate::data::view::TapeScratchView::Missing);
                let Some(result) =
                    crate::exec::view::run_with_env_and_vm(source, body, None, &env, &mut vm)
                else {
                    writer.write_all(b"null")?;
                    writer.write_all(b"\n")?;
                    count += 1;
                    if limit.is_some_and(|limit| count >= limit) {
                        break;
                    }
                    continue;
                };
                write_val_json(
                    &mut writer,
                    &result.map_err(|err| JetroEngineError::Eval(err))?,
                )?;
            }
        }
        writer.write_all(b"\n")?;
        count += 1;
        if limit.is_some_and(|limit| count >= limit) {
            break;
        }
    }

    writer.flush()?;
    Ok(count)
}

#[cfg(feature = "simd-json")]
fn drive_ndjson_tape_matches_writer<R, W>(
    engine: &JetroEngine,
    reader: R,
    predicate: &NdjsonDirectPredicate,
    limit: usize,
    options: NdjsonOptions,
    writer: W,
) -> Result<usize, JetroEngineError>
where
    R: BufRead,
    W: Write,
{
    let mut driver = NdjsonPerRowDriver::new(reader).with_max_line_len(options.max_line_len);
    let mut writer = ndjson_writer_with_options(writer, options);
    let mut line = Vec::with_capacity(options.initial_buffer_capacity);
    let mut scratch =
        crate::data::tape::TapeScratch::with_capacity(options.initial_buffer_capacity);
    let mut emitted = 0usize;
    let mut vm = predicate_needs_vm(predicate).then(|| engine.lock_vm());
    let env = crate::data::context::Env::new(Val::Null);

    while let Some((line_no, row)) = driver.read_next_owned(&mut line)? {
        scratch.parse_slice(&row).map_err(|message| {
            row_parse_error(
                line_no,
                JetroEngineError::Eval(crate::EvalError(format!("Invalid JSON: {message}"))),
            )
        })?;
        if !eval_tape_predicate(&scratch, predicate, &env, &mut vm)
            .map_err(JetroEngineError::Eval)?
        {
            continue;
        }
        writer.write_all(&row)?;
        writer.write_all(b"\n")?;
        emitted += 1;
        if emitted >= limit {
            break;
        }
    }

    writer.flush()?;
    Ok(emitted)
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

    let mut driver = NdjsonPerRowDriver::new(reader).with_max_line_len(options.max_line_len);
    let mut executor = NdjsonRowExecutor::new(engine, predicate);
    let mut buf = Vec::with_capacity(options.initial_buffer_capacity);
    let mut emitted = 0usize;

    while let Some((line_no, row)) = driver.read_next_owned(&mut buf)? {
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
    if limit == 0 {
        return Ok(0);
    }

    #[cfg(feature = "simd-json")]
    if let Some(predicate) = direct_tape_predicate(engine, predicate) {
        return drive_ndjson_tape_matches_writer(
            engine, reader, &predicate, limit, options, writer,
        );
    }

    let mut driver = NdjsonPerRowDriver::new(reader).with_max_line_len(options.max_line_len);
    let mut executor = NdjsonRowExecutor::new(engine, predicate);
    let mut writer = ndjson_writer_with_options(writer, options);
    let mut buf = Vec::with_capacity(options.initial_buffer_capacity);
    let mut emitted = 0usize;

    while let Some((line_no, row)) = driver.read_next_owned(&mut buf)? {
        let document = executor.parse_owned_row(line_no, row)?;
        let matched = executor.eval_document(line_no, &document)?;
        if !is_truthy(&matched) {
            continue;
        }

        write_document_line(&mut writer, &document, line_no, executor.engine())?;
        emitted += 1;
        if emitted >= limit {
            break;
        }
    }

    writer.flush()?;
    Ok(emitted)
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

    #[cfg(feature = "simd-json")]
    fn direct_tape_plan(engine: &'a JetroEngine, query: &str) -> Option<NdjsonDirectTapePlan> {
        use crate::builtins::{BuiltinArgs, BuiltinMethod};
        use crate::ir::physical::{PlanNode, QueryRoot};

        let plan = engine.cached_plan(query, PlanningContext::bytes());
        let QueryRoot::Node(root) = plan.root() else {
            return None;
        };
        if let PlanNode::Chain { base, steps } = plan.node(*root) {
            let (source_steps, element) = direct_array_element_source(&plan, *base)?;
            return Some(NdjsonDirectTapePlan::ArrayElementPath {
                source_steps,
                element,
                suffix_steps: physical_chain_to_path(steps)?,
            });
        }
        if let Some((source_steps, element)) = direct_array_element_source(&plan, *root) {
            return Some(NdjsonDirectTapePlan::ArrayElementPath {
                source_steps,
                element,
                suffix_steps: Vec::new(),
            });
        }
        match plan.node(*root) {
            PlanNode::RootPath(steps) => Some(NdjsonDirectTapePlan::RootPath(steps.clone())),
            PlanNode::Pipeline {
                source: crate::ir::physical::PipelinePlanSource::FieldChain { keys },
                body,
            } if body.stages.is_empty()
                && matches!(
                    body.sink,
                    crate::exec::pipeline::Sink::Reducer(ref spec)
                        if spec.op == crate::exec::pipeline::ReducerOp::Count
                            && spec.predicate.is_none()
                ) =>
            {
                Some(NdjsonDirectTapePlan::ViewScalarCall {
                    steps: keys
                        .iter()
                        .map(|key| crate::ir::physical::PhysicalPathStep::Field(key.clone()))
                        .collect(),
                    call: crate::builtins::BuiltinCall::new(BuiltinMethod::Len, BuiltinArgs::None),
                    optional: false,
                })
            }
            PlanNode::Pipeline {
                source: crate::ir::physical::PipelinePlanSource::Expr(source),
                body,
            } if body.stages.is_empty()
                && matches!(
                    body.sink,
                    crate::exec::pipeline::Sink::Reducer(ref spec)
                        if spec.op == crate::exec::pipeline::ReducerOp::Count
                            && spec.predicate.is_none()
                ) =>
            {
                let PlanNode::RootPath(steps) = plan.node(*source) else {
                    return None;
                };
                Some(NdjsonDirectTapePlan::ViewScalarCall {
                    steps: steps.clone(),
                    call: crate::builtins::BuiltinCall::new(BuiltinMethod::Len, BuiltinArgs::None),
                    optional: false,
                })
            }
            PlanNode::Call {
                receiver,
                call,
                optional,
            } if call.method == BuiltinMethod::Len && matches!(call.args, BuiltinArgs::None) => {
                let PlanNode::RootPath(steps) = plan.node(*receiver) else {
                    return None;
                };
                Some(NdjsonDirectTapePlan::ViewScalarCall {
                    steps: steps.clone(),
                    call: call.clone(),
                    optional: *optional,
                })
            }
            PlanNode::Pipeline { source, body } if body.can_run_with_view() => {
                Some(NdjsonDirectTapePlan::ViewPipeline {
                    source_steps: pipeline_source_to_steps(&plan, source)?,
                    body: body.clone(),
                })
            }
            _ => None,
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

    pub(super) fn write_owned_row<W: Write>(
        &mut self,
        line_no: u64,
        row: Vec<u8>,
        writer: &mut W,
    ) -> Result<(), JetroEngineError> {
        let document = self.parse_owned_result_row(line_no, row)?;
        self.write_document_result(line_no, &document, writer)
    }

    fn parse_owned_result_row(
        &self,
        line_no: u64,
        row: Vec<u8>,
    ) -> Result<Jetro, JetroEngineError> {
        #[cfg(feature = "simd-json")]
        {
            crate::data::tape::TapeData::parse(row)
                .map(Jetro::from_tape_data)
                .map_err(|message| {
                    row_parse_error(
                        line_no,
                        JetroEngineError::Eval(crate::EvalError(format!(
                            "Invalid JSON: {message}"
                        ))),
                    )
                })
        }
        #[cfg(not(feature = "simd-json"))]
        {
            self.parse_owned_row(line_no, row)
        }
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

    pub(super) fn write_document_result<W: Write>(
        &mut self,
        line_no: u64,
        document: &Jetro,
        writer: &mut W,
    ) -> Result<(), JetroEngineError> {
        if self.try_write_tape_result(line_no, document, writer)? {
            return Ok(());
        }
        let value = self.eval_document(line_no, document)?;
        write_val_line(writer, &value)
    }

    fn try_write_tape_result<W: Write>(
        &self,
        line_no: u64,
        document: &Jetro,
        writer: &mut W,
    ) -> Result<bool, JetroEngineError> {
        #[cfg(feature = "simd-json")]
        {
            use crate::ir::physical::{PlanNode, QueryRoot};

            let QueryRoot::Node(root) = self.plan.root() else {
                return Ok(false);
            };
            let PlanNode::RootPath(steps) = self.plan.node(*root) else {
                return Ok(false);
            };
            let Some(tape) = document
                .lazy_tape()
                .map_err(|err| row_eval_error(line_no, err))?
            else {
                return Ok(false);
            };
            if let Some(idx) = json_tape_path_index(tape.as_ref(), steps) {
                write_json_tape_at(writer, tape.as_ref(), idx)?;
            } else {
                writer.write_all(b"null")?;
            }
            writer.write_all(b"\n")?;
            Ok(true)
        }
        #[cfg(not(feature = "simd-json"))]
        {
            let _ = (line_no, document, writer);
            Ok(false)
        }
    }

    pub(super) fn engine(&self) -> &'a JetroEngine {
        self.engine
    }
}

#[cfg(feature = "simd-json")]
fn pipeline_source_to_steps(
    plan: &crate::ir::physical::QueryPlan,
    source: &crate::ir::physical::PipelinePlanSource,
) -> Option<Vec<crate::ir::physical::PhysicalPathStep>> {
    match source {
        crate::ir::physical::PipelinePlanSource::FieldChain { keys } => Some(
            keys.iter()
                .map(|key| crate::ir::physical::PhysicalPathStep::Field(key.clone()))
                .collect(),
        ),
        crate::ir::physical::PipelinePlanSource::Expr(source) => {
            let crate::ir::physical::PlanNode::RootPath(steps) = plan.node(*source) else {
                return None;
            };
            Some(steps.clone())
        }
    }
}

#[cfg(feature = "simd-json")]
pub(super) fn direct_tape_predicate(
    engine: &JetroEngine,
    predicate: &str,
) -> Option<NdjsonDirectPredicate> {
    let plan = engine.cached_plan(predicate, PlanningContext::bytes());
    let crate::ir::physical::QueryRoot::Node(root) = plan.root() else {
        return None;
    };
    direct_tape_predicate_node(&plan, *root)
}

#[cfg(feature = "simd-json")]
fn direct_tape_predicate_node(
    plan: &crate::ir::physical::QueryPlan,
    id: crate::ir::physical::NodeId,
) -> Option<NdjsonDirectPredicate> {
    use crate::ir::physical::PlanNode;

    match plan.node(id) {
        PlanNode::Literal(value) => Some(NdjsonDirectPredicate::Literal(value.clone())),
        PlanNode::RootPath(steps) => Some(NdjsonDirectPredicate::Path(steps.clone())),
        PlanNode::Not(inner) => Some(NdjsonDirectPredicate::Not(Box::new(
            direct_tape_predicate_node(plan, *inner)?,
        ))),
        PlanNode::Binary { lhs, op, rhs } => Some(NdjsonDirectPredicate::Binary {
            lhs: Box::new(direct_tape_predicate_node(plan, *lhs)?),
            op: *op,
            rhs: Box::new(direct_tape_predicate_node(plan, *rhs)?),
        }),
        PlanNode::Call {
            receiver,
            call,
            optional,
        } if !*optional && call.spec().view_scalar => {
            direct_tape_predicate_scalar_call(plan, *receiver, call.clone())
        }
        PlanNode::Pipeline { source, body } => {
            if let Some(predicate) = direct_tape_predicate_membership_sink(plan, source, body) {
                return Some(predicate);
            }
            if !body.can_run_with_view() {
                return None;
            }
            Some(NdjsonDirectPredicate::ViewPipeline {
                source_steps: pipeline_source_to_steps(plan, source)?,
                body: body.clone(),
            })
        }
        _ => None,
    }
}

#[cfg(feature = "simd-json")]
fn direct_tape_predicate_membership_sink(
    plan: &crate::ir::physical::QueryPlan,
    source: &crate::ir::physical::PipelinePlanSource,
    body: &crate::exec::pipeline::PipelineBody,
) -> Option<NdjsonDirectPredicate> {
    use crate::builtins::{BuiltinArgs, BuiltinCall};
    use crate::exec::pipeline::{MembershipSinkOp, MembershipSinkTarget, Sink};

    if !body.stages.is_empty() {
        return None;
    }
    let Sink::Membership(spec) = &body.sink else {
        return None;
    };
    if spec.op != MembershipSinkOp::Includes {
        return None;
    }
    let MembershipSinkTarget::Literal(target) = &spec.target else {
        return None;
    };
    let call = BuiltinCall::new(spec.method, BuiltinArgs::Val(target.clone()));
    direct_tape_predicate_source_scalar_call(plan, source, call)
}

#[cfg(feature = "simd-json")]
fn direct_tape_predicate_source_scalar_call(
    plan: &crate::ir::physical::QueryPlan,
    source: &crate::ir::physical::PipelinePlanSource,
    call: crate::builtins::BuiltinCall,
) -> Option<NdjsonDirectPredicate> {
    match source {
        crate::ir::physical::PipelinePlanSource::FieldChain { keys } => {
            Some(NdjsonDirectPredicate::ViewScalarCall {
                steps: keys
                    .iter()
                    .map(|key| crate::ir::physical::PhysicalPathStep::Field(key.clone()))
                    .collect(),
                call,
            })
        }
        crate::ir::physical::PipelinePlanSource::Expr(receiver) => {
            direct_tape_predicate_scalar_call(plan, *receiver, call)
        }
    }
}

#[cfg(feature = "simd-json")]
fn direct_tape_predicate_scalar_call(
    plan: &crate::ir::physical::QueryPlan,
    receiver: crate::ir::physical::NodeId,
    call: crate::builtins::BuiltinCall,
) -> Option<NdjsonDirectPredicate> {
    use crate::ir::physical::PlanNode;

    if let PlanNode::RootPath(steps) = plan.node(receiver) {
        return Some(NdjsonDirectPredicate::ViewScalarCall {
            steps: steps.clone(),
            call,
        });
    }

    let PlanNode::Chain { base, steps } = plan.node(receiver) else {
        return None;
    };
    let (source_steps, element) = direct_array_element_source(plan, *base)?;
    Some(NdjsonDirectPredicate::ArrayElementViewScalarCall {
        source_steps,
        element,
        suffix_steps: physical_chain_to_path(steps)?,
        call,
    })
}

#[cfg(feature = "simd-json")]
fn direct_array_element_source(
    plan: &crate::ir::physical::QueryPlan,
    id: crate::ir::physical::NodeId,
) -> Option<(
    Vec<crate::ir::physical::PhysicalPathStep>,
    NdjsonDirectElement,
)> {
    use crate::builtins::BuiltinMethod;
    use crate::exec::pipeline::Sink;
    use crate::ir::physical::{PipelinePlanSource, PlanNode};

    if let PlanNode::Call {
        receiver,
        call,
        optional,
    } = plan.node(id)
    {
        if *optional {
            return None;
        }
        let element = match call.method {
            BuiltinMethod::First => NdjsonDirectElement::First,
            BuiltinMethod::Last => NdjsonDirectElement::Last,
            _ => return None,
        };
        let PlanNode::RootPath(steps) = plan.node(*receiver) else {
            return None;
        };
        return Some((steps.clone(), element));
    }

    let PlanNode::Pipeline { source, body } = plan.node(id) else {
        return None;
    };
    if !body.stages.is_empty() {
        return None;
    }
    let element = match body.sink {
        Sink::Terminal(BuiltinMethod::First) => NdjsonDirectElement::First,
        Sink::Terminal(BuiltinMethod::Last) => NdjsonDirectElement::Last,
        Sink::SelectMany {
            n: 1,
            from_end: false,
        } => NdjsonDirectElement::First,
        Sink::SelectMany {
            n: 1,
            from_end: true,
        } => NdjsonDirectElement::Last,
        Sink::Nth(n) => NdjsonDirectElement::Nth(n),
        _ => return None,
    };
    let source_steps = match source {
        PipelinePlanSource::FieldChain { keys } => keys
            .iter()
            .map(|key| crate::ir::physical::PhysicalPathStep::Field(key.clone()))
            .collect(),
        PipelinePlanSource::Expr(source) => {
            let PlanNode::RootPath(steps) = plan.node(*source) else {
                return None;
            };
            steps.clone()
        }
    };
    Some((source_steps, element))
}

#[cfg(feature = "simd-json")]
fn physical_chain_to_path(
    steps: &[crate::ir::physical::PhysicalChainStep],
) -> Option<Vec<crate::ir::physical::PhysicalPathStep>> {
    steps
        .iter()
        .map(|step| match step {
            crate::ir::physical::PhysicalChainStep::Field(key) => {
                Some(crate::ir::physical::PhysicalPathStep::Field(key.clone()))
            }
            crate::ir::physical::PhysicalChainStep::Index(idx) => {
                Some(crate::ir::physical::PhysicalPathStep::Index(*idx))
            }
            crate::ir::physical::PhysicalChainStep::DynIndex(_) => None,
        })
        .collect()
}

#[cfg(feature = "simd-json")]
trait JsonTape {
    fn nodes(&self) -> &[crate::data::tape::TapeNode];
    fn str_at(&self, idx: usize) -> &str;
    fn span(&self, idx: usize) -> usize;
}

#[cfg(feature = "simd-json")]
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

#[cfg(feature = "simd-json")]
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

#[cfg(feature = "simd-json")]
fn json_tape_path_index<T: JsonTape>(
    tape: &T,
    steps: &[crate::ir::physical::PhysicalPathStep],
) -> Option<usize> {
    json_tape_path_index_from(tape, 0, steps)
}

#[cfg(feature = "simd-json")]
fn json_tape_path_index_from<T: JsonTape>(
    tape: &T,
    start: usize,
    steps: &[crate::ir::physical::PhysicalPathStep],
) -> Option<usize> {
    use crate::data::tape::TapeNode;
    use crate::ir::physical::PhysicalPathStep;

    if tape.nodes().is_empty() {
        return None;
    }

    let mut idx = start;
    for step in steps {
        match step {
            PhysicalPathStep::Field(key) => {
                let TapeNode::Object { len, .. } = tape.nodes()[idx] else {
                    return None;
                };
                let mut cur = idx + 1;
                let mut found = None;
                for _ in 0..len {
                    if tape.str_at(cur) == key.as_ref() {
                        found = Some(cur + 1);
                        break;
                    }
                    cur += 1;
                    cur += tape.span(cur);
                }
                idx = found?;
            }
            PhysicalPathStep::Index(wanted) => {
                let TapeNode::Array { len, .. } = tape.nodes()[idx] else {
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
                let mut cur = idx + 1;
                for _ in 0..wanted {
                    cur += tape.span(cur);
                }
                idx = cur;
            }
        }
    }
    Some(idx)
}

#[cfg(feature = "simd-json")]
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

#[cfg(feature = "simd-json")]
pub(super) fn eval_tape_predicate(
    tape: &crate::data::tape::TapeScratch,
    predicate: &NdjsonDirectPredicate,
    env: &crate::data::context::Env,
    vm: &mut Option<std::sync::MutexGuard<'_, crate::vm::exec::VM>>,
) -> Result<bool, crate::EvalError> {
    use crate::parse::ast::BinOp;

    Ok(match predicate {
        NdjsonDirectPredicate::Path(steps) => json_tape_path_index(tape, steps)
            .map(|idx| json_view_truthy(json_tape_scalar(tape, idx)))
            .unwrap_or(false),
        NdjsonDirectPredicate::Literal(value) => crate::util::is_truthy(value),
        NdjsonDirectPredicate::Not(inner) => !eval_tape_predicate(tape, inner, env, vm)?,
        NdjsonDirectPredicate::Binary { lhs, op, rhs } if *op == BinOp::And => {
            eval_tape_predicate(tape, lhs, env, vm)? && eval_tape_predicate(tape, rhs, env, vm)?
        }
        NdjsonDirectPredicate::Binary { lhs, op, rhs } if *op == BinOp::Or => {
            eval_tape_predicate(tape, lhs, env, vm)? || eval_tape_predicate(tape, rhs, env, vm)?
        }
        NdjsonDirectPredicate::Binary { lhs, op, rhs } => {
            let Some(lhs) = eval_tape_scalar(tape, lhs) else {
                return Ok(false);
            };
            let Some(rhs) = eval_tape_scalar(tape, rhs) else {
                return Ok(false);
            };
            crate::util::json_cmp_binop(lhs, *op, rhs)
        }
        NdjsonDirectPredicate::ViewScalarCall { steps, call } => json_tape_path_index(tape, steps)
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
        NdjsonDirectPredicate::ViewPipeline { source_steps, body } => {
            let Some(vm) = vm.as_deref_mut() else {
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

#[cfg(feature = "simd-json")]
pub(super) fn predicate_needs_vm(predicate: &NdjsonDirectPredicate) -> bool {
    match predicate {
        NdjsonDirectPredicate::Not(inner) => predicate_needs_vm(inner),
        NdjsonDirectPredicate::Binary { lhs, rhs, .. } => {
            predicate_needs_vm(lhs) || predicate_needs_vm(rhs)
        }
        NdjsonDirectPredicate::ViewPipeline { .. } => true,
        NdjsonDirectPredicate::Path(_)
        | NdjsonDirectPredicate::Literal(_)
        | NdjsonDirectPredicate::ViewScalarCall { .. }
        | NdjsonDirectPredicate::ArrayElementViewScalarCall { .. } => false,
    }
}

#[cfg(feature = "simd-json")]
fn eval_tape_scalar<'a>(
    tape: &'a crate::data::tape::TapeScratch,
    predicate: &'a NdjsonDirectPredicate,
) -> Option<crate::util::JsonView<'a>> {
    match predicate {
        NdjsonDirectPredicate::Path(steps) => {
            json_tape_path_index(tape, steps).map(|idx| json_tape_scalar(tape, idx))
        }
        NdjsonDirectPredicate::Literal(value) => Some(crate::util::JsonView::from_val(value)),
        _ => None,
    }
}

#[cfg(feature = "simd-json")]
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

#[cfg(feature = "simd-json")]
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

pub(super) fn write_val_line<W: Write>(
    writer: &mut W,
    value: &Val,
) -> Result<(), JetroEngineError> {
    write_val_json(writer, value)?;
    writer.write_all(b"\n")?;
    Ok(())
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

pub(super) fn ndjson_writer_with_options<W: Write>(
    writer: W,
    options: NdjsonOptions,
) -> BufWriter<W> {
    let capacity = options
        .reader_buffer_capacity
        .max(DEFAULT_READER_BUFFER_CAPACITY);
    BufWriter::with_capacity(capacity, writer)
}

fn write_val_json<W: Write>(writer: &mut W, value: &Val) -> Result<(), JetroEngineError> {
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

#[cfg(feature = "simd-json")]
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

fn write_json_str<W: Write>(writer: &mut W, value: &str) -> Result<(), JetroEngineError> {
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
fn write_i64<W: Write>(writer: &mut W, value: i64) -> Result<(), JetroEngineError> {
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

pub(super) fn row_parse_error(line_no: u64, err: JetroEngineError) -> JetroEngineError {
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

pub(super) fn row_eval_error(line_no: u64, err: crate::EvalError) -> JetroEngineError {
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

    #[test]
    #[cfg(feature = "simd-json")]
    fn direct_tape_plan_accepts_first_suffix() {
        let engine = crate::JetroEngine::new();
        let query = "attributes.first().value";
        let plan = super::NdjsonRowExecutor::direct_tape_plan(&engine, query)
            .expect("first suffix should be direct");
        assert!(matches!(
            plan,
            super::NdjsonDirectTapePlan::ArrayElementPath { .. }
        ));
    }
}
