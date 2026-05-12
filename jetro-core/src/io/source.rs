use std::fmt;
use std::io::BufRead;
use std::path::{Path, PathBuf};

/// Input source for NDJSON APIs that can operate on either files or callers'
/// existing buffered readers.
pub enum NdjsonSource {
    File(PathBuf),
    Reader(Box<dyn BufRead + Send>),
}

impl NdjsonSource {
    pub fn file<P: Into<PathBuf>>(path: P) -> Self {
        Self::File(path.into())
    }

    pub fn reader<R>(reader: R) -> Self
    where
        R: BufRead + Send + 'static,
    {
        Self::Reader(Box::new(reader))
    }

    pub fn as_file_path(&self) -> Option<&Path> {
        match self {
            Self::File(path) => Some(path.as_path()),
            Self::Reader(_) => None,
        }
    }
}

impl fmt::Debug for NdjsonSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::File(path) => f.debug_tuple("File").field(path).finish(),
            Self::Reader(_) => f.write_str("Reader(<bufread>)"),
        }
    }
}
