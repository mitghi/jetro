use super::RowError;
use crate::util::is_truthy;
use crate::{JetroEngine, JetroEngineError};
use memchr::memrchr;
use serde_json::Value;
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
            if let Some(line) = self.pending.pop_front() {
                self.reverse_line_no += 1;
                return Ok(Some((self.reverse_line_no, line)));
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
                    return Ok(Some((self.reverse_line_no, line)));
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
    #[cfg(feature = "simd-json")]
    if let Some(plan) = super::ndjson::direct_tape_plan(engine, query) {
        return drive_rev_writer_tape(engine, path, &plan, None, options, writer);
    }

    let mut writer = super::ndjson::ndjson_writer_with_options(writer, options);
    let count = drive_rev(engine, path, query, options, |value| {
        super::ndjson::write_val_line(&mut writer, &value)?;
        Ok(super::ndjson::NdjsonControl::Continue)
    })?;
    writer.flush()?;
    Ok(count)
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

    #[cfg(feature = "simd-json")]
    if let Some(plan) = super::ndjson::direct_tape_plan(engine, query) {
        return drive_rev_writer_tape(engine, path, &plan, Some(limit), options, writer);
    }

    let mut writer = super::ndjson::ndjson_writer_with_options(writer, options);
    let mut emitted = 0usize;
    let count = drive_rev(engine, path, query, options, |value| {
        super::ndjson::write_val_line(&mut writer, &value)?;
        emitted += 1;
        Ok(if emitted >= limit {
            super::ndjson::NdjsonControl::Stop
        } else {
            super::ndjson::NdjsonControl::Continue
        })
    })?;
    writer.flush()?;
    Ok(count)
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

#[cfg(feature = "simd-json")]
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
    let mut runner = super::ndjson::NdjsonTapeWriterRunner::new(engine, plan);
    let mut count = 0usize;

    while let Some((reverse_row_no, row)) = driver.next_line_with_reverse_no()? {
        scratch.parse_slice(&row).map_err(|message| {
            super::ndjson::row_parse_error(
                reverse_row_no,
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
    if limit == 0 {
        return Ok(0);
    }

    #[cfg(feature = "simd-json")]
    if let Some(predicate) = super::ndjson::direct_tape_predicate(engine, predicate) {
        return drive_rev_matches_writer_tape(engine, path, &predicate, limit, options, writer);
    }

    let mut driver = NdjsonReverseFileDriver::with_options(path, options)?;
    let mut executor = super::ndjson::NdjsonRowExecutor::new(engine, predicate);
    let mut writer = super::ndjson::ndjson_writer_with_options(writer, options);
    let mut emitted = 0usize;

    while let Some((reverse_row_no, row)) = driver.next_line_with_reverse_no()? {
        let document = executor.parse_owned_row(reverse_row_no, row)?;
        let matched = executor.eval_document(reverse_row_no, &document)?;
        if !is_truthy(&matched) {
            continue;
        }

        super::ndjson::write_document_line(
            &mut writer,
            &document,
            reverse_row_no,
            executor.engine(),
        )?;
        emitted += 1;
        if emitted >= limit {
            break;
        }
    }

    writer.flush()?;
    Ok(emitted)
}

#[cfg(feature = "simd-json")]
fn drive_rev_matches_writer_tape<P, W>(
    engine: &JetroEngine,
    path: P,
    predicate: &super::ndjson::NdjsonDirectPredicate,
    limit: usize,
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
    let mut emitted = 0usize;
    let needs_vm = super::ndjson::predicate_needs_vm(predicate);
    let mut vm = needs_vm.then(|| engine.lock_vm());
    let env = needs_vm.then(|| crate::data::context::Env::new(crate::Val::Null));
    let mut predicate_path = super::ndjson::NdjsonPathCache::default();

    while let Some((reverse_row_no, row)) = driver.next_line_with_reverse_no()? {
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

    fn temp_path(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("{}-{}.ndjson", name, std::process::id()));
        path
    }
}
