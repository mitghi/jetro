use super::RowError;
use memchr::memrchr;
use std::collections::VecDeque;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

const DEFAULT_REVERSE_CHUNK_SIZE: usize = 64 * 1024;

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
    carry: Vec<u8>,
    pending: VecDeque<Vec<u8>>,
    finished_head: bool,
}

impl NdjsonReverseFileDriver {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, RowError> {
        Self::with_chunk_size(path, DEFAULT_REVERSE_CHUNK_SIZE)
    }

    pub fn with_chunk_size<P: AsRef<Path>>(path: P, chunk_size: usize) -> Result<Self, RowError> {
        let mut file = File::open(path)?;
        let pos = file.seek(SeekFrom::End(0))?;
        Ok(Self {
            file,
            pos,
            chunk_size: chunk_size.max(1),
            carry: Vec::new(),
            pending: VecDeque::new(),
            finished_head: false,
        })
    }

    pub fn next_line(&mut self) -> Result<Option<Vec<u8>>, RowError> {
        loop {
            if let Some(line) = self.pending.pop_front() {
                return Ok(Some(line));
            }

            if self.pos == 0 {
                if self.finished_head || self.carry.is_empty() {
                    return Ok(None);
                }
                self.finished_head = true;
                let mut line = std::mem::take(&mut self.carry);
                trim_line_ending(&mut line);
                if line.iter().any(|b| !b.is_ascii_whitespace()) {
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
                if line.iter().any(|b| !b.is_ascii_whitespace()) {
                    self.pending.push_back(line);
                }
            }

            if end > 0 {
                let mut next = Vec::with_capacity(end + self.carry.len());
                next.extend_from_slice(&chunk[..end]);
                next.extend_from_slice(&self.carry);
                self.carry = next;
            }
        }
    }
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
