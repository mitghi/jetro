use super::RowError;
use crate::data::value::ValRef;
use crate::plan::physical::PlanningContext;
use crate::{JetroEngine, JetroEngineError};
use memchr::memrchr;
use serde_json::Value;
use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
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
        loop {
            if let Some(line) = self.pending.pop_front() {
                self.reverse_line_no += 1;
                return Ok(Some(line));
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
                    return Ok(Some(line));
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
        Ok(())
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
    let mut writer = BufWriter::new(writer);
    let count = drive_rev(engine, path, query, options, |value| {
        serde_json::to_writer(&mut writer, &ValRef(&value))?;
        writer.write_all(b"\n")?;
        Ok(())
    })?;
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
    F: FnMut(crate::data::value::Val) -> Result<(), JetroEngineError>,
{
    let mut driver = NdjsonReverseFileDriver::with_options(path, options)?;
    let plan = engine.cached_plan(query, PlanningContext::bytes());
    let mut reverse_row_no = 0u64;

    while let Some(row) = driver.next_line()? {
        reverse_row_no += 1;
        let document = super::ndjson::parse_row(engine, reverse_row_no, row)?;
        emit(super::ndjson::collect_row_val(
            engine,
            &document,
            &plan,
            reverse_row_no,
        )?)?;
    }

    Ok(reverse_row_no as usize)
}

fn trim_line_ending(buf: &mut Vec<u8>) {
    while matches!(buf.last(), Some(b'\n' | b'\r')) {
        buf.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::NdjsonReverseFileDriver;
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

    fn temp_path(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("{}-{}.ndjson", name, std::process::id()));
        path
    }
}
