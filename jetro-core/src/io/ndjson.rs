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

#[cfg(feature = "simd-json")]
pub(super) use super::ndjson_direct::{
    direct_tape_plan, direct_tape_predicate, NdjsonDirectElement, NdjsonDirectItemPredicate,
    NdjsonDirectPredicate, NdjsonDirectTapePlan,
};

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
    if let Some(plan) = direct_tape_plan(engine, query) {
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
    let mut runner = NdjsonTapeWriterRunner::new(engine, plan);

    while let Some((line_no, row)) = driver.read_next_nonempty(&mut line)? {
        scratch.parse_slice(row).map_err(|message| {
            row_parse_error(
                line_no,
                JetroEngineError::Eval(crate::EvalError(format!("Invalid JSON: {message}"))),
            )
        })?;
        runner.write_row(&scratch, &mut writer)?;
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
struct NdjsonTapeWriterRunner<'a, 'p> {
    plan: &'p NdjsonDirectTapePlan,
    vm: Option<MutexGuard<'a, VM>>,
    env: Option<crate::data::context::Env>,
}

#[cfg(feature = "simd-json")]
impl<'a, 'p> NdjsonTapeWriterRunner<'a, 'p> {
    fn new(engine: &'a JetroEngine, plan: &'p NdjsonDirectTapePlan) -> Self {
        let needs_vm = plan.needs_vm();
        Self {
            plan,
            vm: needs_vm.then(|| engine.lock_vm()),
            env: needs_vm.then(|| crate::data::context::Env::new(Val::Null)),
        }
    }

    fn write_row<W: Write>(
        &mut self,
        scratch: &crate::data::tape::TapeScratch,
        writer: &mut W,
    ) -> Result<(), JetroEngineError> {
        match self.plan {
            NdjsonDirectTapePlan::RootPath(steps) => {
                if let Some(idx) = json_tape_path_index(scratch, steps) {
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
                let idx = json_tape_path_index(scratch, steps);
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
            NdjsonDirectTapePlan::ArrayElementPath {
                source_steps,
                element,
                suffix_steps,
            } => {
                let idx = json_tape_path_index(scratch, source_steps)
                    .and_then(|idx| json_tape_array_element(scratch, idx, *element))
                    .and_then(|idx| json_tape_path_index_from(scratch, idx, suffix_steps));
                if let Some(idx) = idx {
                    write_json_tape_at(writer, scratch, idx)?;
                } else {
                    writer.write_all(b"null")?;
                }
            }
            NdjsonDirectTapePlan::MapPath {
                source_steps,
                suffix_steps,
            } => {
                write_json_tape_map_path(writer, scratch, source_steps, suffix_steps)?;
            }
            NdjsonDirectTapePlan::FilterMapPath {
                source_steps,
                predicate,
                suffix_steps,
            } => {
                write_json_tape_filter_map_path(
                    writer,
                    scratch,
                    source_steps,
                    predicate,
                    suffix_steps,
                )?;
            }
            NdjsonDirectTapePlan::CountFiltered {
                source_steps,
                predicate,
            } => {
                let count = count_json_tape_filtered(scratch, source_steps, predicate);
                write_i64(writer, count as i64)?;
            }
            NdjsonDirectTapePlan::NumericReducePath {
                source_steps,
                suffix_steps,
                op,
            } => {
                let value =
                    reduce_json_tape_numeric_path(scratch, source_steps, None, suffix_steps, *op);
                write_val_json(writer, &value)?;
            }
            NdjsonDirectTapePlan::FilterNumericReducePath {
                source_steps,
                predicate,
                suffix_steps,
                op,
            } => {
                let value = reduce_json_tape_numeric_path(
                    scratch,
                    source_steps,
                    Some(predicate),
                    suffix_steps,
                    *op,
                );
                write_val_json(writer, &value)?;
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
    let needs_vm = predicate_needs_vm(predicate);
    let mut vm = needs_vm.then(|| engine.lock_vm());
    let env = needs_vm.then(|| crate::data::context::Env::new(Val::Null));

    while let Some((line_no, row)) = driver.read_next_owned(&mut line)? {
        scratch.parse_slice(&row).map_err(|message| {
            row_parse_error(
                line_no,
                JetroEngineError::Eval(crate::EvalError(format!("Invalid JSON: {message}"))),
            )
        })?;
        if !eval_tape_predicate(&scratch, predicate, env.as_ref(), &mut vm)
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

#[cfg(feature = "simd-json")]
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

#[cfg(feature = "simd-json")]
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
    env: Option<&crate::data::context::Env>,
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

#[cfg(feature = "simd-json")]
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

#[cfg(feature = "simd-json")]
fn write_json_tape_map_path<W: Write, T: JsonTape>(
    writer: &mut W,
    tape: &T,
    source_steps: &[crate::ir::physical::PhysicalPathStep],
    suffix_steps: &[crate::ir::physical::PhysicalPathStep],
) -> Result<(), JetroEngineError> {
    writer.write_all(b"[")?;
    let Some(source_idx) = json_tape_path_index(tape, source_steps) else {
        writer.write_all(b"]")?;
        return Ok(());
    };

    let mut wrote = false;
    visit_json_tape_source_items(tape, source_idx, |item_idx| {
        if wrote {
            writer.write_all(b",")?;
        }
        write_json_tape_path_or_null(writer, tape, item_idx, suffix_steps)?;
        wrote = true;
        Ok::<(), JetroEngineError>(())
    })?;

    writer.write_all(b"]")?;
    Ok(())
}

#[cfg(feature = "simd-json")]
fn write_json_tape_filter_map_path<W: Write, T: JsonTape>(
    writer: &mut W,
    tape: &T,
    source_steps: &[crate::ir::physical::PhysicalPathStep],
    predicate: &NdjsonDirectItemPredicate,
    suffix_steps: &[crate::ir::physical::PhysicalPathStep],
) -> Result<(), JetroEngineError> {
    writer.write_all(b"[")?;
    let Some(source_idx) = json_tape_path_index(tape, source_steps) else {
        writer.write_all(b"]")?;
        return Ok(());
    };

    let mut wrote = false;
    visit_json_tape_source_items(tape, source_idx, |item_idx| {
        if eval_json_tape_item_predicate(tape, item_idx, predicate) {
            if wrote {
                writer.write_all(b",")?;
            }
            write_json_tape_path_or_null(writer, tape, item_idx, suffix_steps)?;
            wrote = true;
        }
        Ok::<(), JetroEngineError>(())
    })?;

    writer.write_all(b"]")?;
    Ok(())
}

#[cfg(feature = "simd-json")]
fn write_json_tape_path_or_null<W: Write, T: JsonTape>(
    writer: &mut W,
    tape: &T,
    start: usize,
    suffix_steps: &[crate::ir::physical::PhysicalPathStep],
) -> Result<(), JetroEngineError> {
    if let Some(idx) = json_tape_path_index_from(tape, start, suffix_steps) {
        write_json_tape_at(writer, tape, idx)?;
    } else {
        writer.write_all(b"null")?;
    }
    Ok(())
}

#[cfg(feature = "simd-json")]
fn count_json_tape_filtered<T: JsonTape>(
    tape: &T,
    source_steps: &[crate::ir::physical::PhysicalPathStep],
    predicate: &NdjsonDirectItemPredicate,
) -> usize {
    let Some(source_idx) = json_tape_path_index(tape, source_steps) else {
        return 0;
    };

    let mut count = 0usize;
    let _: Result<(), ()> = visit_json_tape_source_items(tape, source_idx, |item_idx| {
        if eval_json_tape_item_predicate(tape, item_idx, predicate) {
            count += 1;
        }
        Ok(())
    });
    count
}

#[cfg(feature = "simd-json")]
fn reduce_json_tape_numeric_path<T: JsonTape>(
    tape: &T,
    source_steps: &[crate::ir::physical::PhysicalPathStep],
    predicate: Option<&NdjsonDirectItemPredicate>,
    suffix_steps: &[crate::ir::physical::PhysicalPathStep],
    op: crate::exec::pipeline::NumOp,
) -> Val {
    let mut acc_i = 0i64;
    let mut acc_f = 0.0f64;
    let mut floated = false;
    let mut min_f = f64::INFINITY;
    let mut max_f = f64::NEG_INFINITY;
    let mut n_obs = 0usize;

    let Some(source_idx) = json_tape_path_index(tape, source_steps) else {
        return crate::exec::pipeline::num_finalise(op, acc_i, acc_f, floated, min_f, max_f, n_obs);
    };

    let _: Result<(), ()> = visit_json_tape_source_items(tape, source_idx, |item_idx| {
        if !predicate
            .is_none_or(|predicate| eval_json_tape_item_predicate(tape, item_idx, predicate))
        {
            return Ok(());
        }
        if let Some(idx) = json_tape_path_index_from(tape, item_idx, suffix_steps) {
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

#[cfg(feature = "simd-json")]
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

#[cfg(feature = "simd-json")]
fn eval_json_tape_item_predicate<T: JsonTape>(
    tape: &T,
    item_idx: usize,
    predicate: &NdjsonDirectItemPredicate,
) -> bool {
    use crate::parse::ast::BinOp;

    match predicate {
        NdjsonDirectItemPredicate::Path(steps) => json_tape_path_index_from(tape, item_idx, steps)
            .map(|idx| json_view_truthy(json_tape_scalar(tape, idx)))
            .unwrap_or(false),
        NdjsonDirectItemPredicate::Literal(value) => crate::util::is_truthy(value),
        NdjsonDirectItemPredicate::Binary { lhs, op, rhs } if *op == BinOp::And => {
            eval_json_tape_item_predicate(tape, item_idx, lhs)
                && eval_json_tape_item_predicate(tape, item_idx, rhs)
        }
        NdjsonDirectItemPredicate::Binary { lhs, op, rhs } if *op == BinOp::Or => {
            eval_json_tape_item_predicate(tape, item_idx, lhs)
                || eval_json_tape_item_predicate(tape, item_idx, rhs)
        }
        NdjsonDirectItemPredicate::Binary { lhs, op, rhs } => {
            let Some(lhs) = eval_json_tape_item_scalar(tape, item_idx, lhs) else {
                return false;
            };
            let Some(rhs) = eval_json_tape_item_scalar(tape, item_idx, rhs) else {
                return false;
            };
            crate::util::json_cmp_binop(lhs, *op, rhs)
        }
        NdjsonDirectItemPredicate::CmpLit { lhs, op, lit } => {
            json_tape_path_index_from(tape, item_idx, lhs)
                .map(|idx| json_tape_scalar(tape, idx))
                .is_some_and(|value| {
                    crate::util::json_cmp_binop(value, *op, crate::util::JsonView::from_val(lit))
                })
        }
        NdjsonDirectItemPredicate::ViewScalarCall { suffix_steps, call } => {
            json_tape_path_index_from(tape, item_idx, suffix_steps)
                .map(|idx| json_tape_scalar(tape, idx))
                .and_then(|value| call.try_apply_json_view(value))
                .is_some_and(|value| crate::util::is_truthy(&value))
        }
    }
}

#[cfg(feature = "simd-json")]
fn eval_json_tape_item_scalar<'a, T: JsonTape>(
    tape: &'a T,
    item_idx: usize,
    predicate: &'a NdjsonDirectItemPredicate,
) -> Option<crate::util::JsonView<'a>> {
    match predicate {
        NdjsonDirectItemPredicate::Path(steps) => {
            json_tape_path_index_from(tape, item_idx, steps).map(|idx| json_tape_scalar(tape, idx))
        }
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
}
