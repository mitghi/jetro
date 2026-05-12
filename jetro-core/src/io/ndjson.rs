use super::RowError;
use crate::{Jetro, JetroEngine, JetroEngineError};
use serde_json::Value;
use std::io::{BufRead, BufWriter, Write};

const DEFAULT_MAX_LINE_LEN: usize = 64 * 1024 * 1024;

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

            while matches!(buf.last(), Some(b'\n' | b'\r')) {
                buf.pop();
            }

            let start = buf
                .iter()
                .position(|b| !b.is_ascii_whitespace())
                .unwrap_or(buf.len());
            let end = buf
                .iter()
                .rposition(|b| !b.is_ascii_whitespace())
                .map(|idx| idx + 1)
                .unwrap_or(start);
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
}

pub fn for_each_ndjson<R, F>(
    engine: &JetroEngine,
    reader: R,
    query: &str,
    mut f: F,
) -> Result<usize, JetroEngineError>
where
    R: BufRead,
    F: FnMut(Value),
{
    let mut driver = NdjsonPerRowDriver::new(reader);
    let mut buf = Vec::with_capacity(8192);
    let mut count = 0;

    while let Some((line_no, row)) = driver.read_next_nonempty(&mut buf)? {
        let document = parse_row(engine, line_no, row)?;
        let out = engine.collect(&document, query)?;
        f(out);
        count += 1;
    }

    Ok(count)
}

pub fn collect_ndjson<R>(
    engine: &JetroEngine,
    reader: R,
    query: &str,
) -> Result<Vec<Value>, JetroEngineError>
where
    R: BufRead,
{
    let mut values = Vec::new();
    for_each_ndjson(engine, reader, query, |value| values.push(value))?;
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
    let mut writer = BufWriter::new(writer);
    let mut driver = NdjsonPerRowDriver::new(reader);
    let mut buf = Vec::with_capacity(8192);
    let mut count = 0;

    while let Some((line_no, row)) = driver.read_next_nonempty(&mut buf)? {
        let document = parse_row(engine, line_no, row)?;
        let out = engine.collect(&document, query)?;
        serde_json::to_writer(&mut writer, &out)?;
        writer.write_all(b"\n")?;
        count += 1;
    }

    writer.flush()?;
    Ok(count)
}

fn parse_row(engine: &JetroEngine, line_no: u64, row: &[u8]) -> Result<Jetro, JetroEngineError> {
    engine
        .parse_bytes(row.to_vec())
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
