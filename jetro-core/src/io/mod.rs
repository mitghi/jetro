//! Streaming input/output adapters for query execution.
//!
//! The initial NDJSON path evaluates each non-empty row independently while
//! reusing the caller's [`crate::JetroEngine`] plan and VM caches.

mod ndjson;
mod ndjson_rev;

pub use ndjson::{
    collect_ndjson, collect_ndjson_file, collect_ndjson_file_with_options,
    collect_ndjson_with_options, for_each_ndjson, for_each_ndjson_with_options, run_ndjson,
    run_ndjson_file, run_ndjson_file_with_options, run_ndjson_with_options, NdjsonOptions,
    NdjsonPerRowDriver,
};
pub use ndjson_rev::NdjsonReverseFileDriver;
use std::fmt;

/// Error with enough row context for users to find malformed input quickly.
#[derive(Debug)]
pub enum RowError {
    Io(std::io::Error),
    InvalidJson {
        line_no: u64,
        source: serde_json::Error,
    },
    InvalidJsonMessage {
        line_no: u64,
        message: String,
    },
    LineTooLarge {
        line_no: u64,
        len: usize,
        max: usize,
    },
}

impl RowError {
    pub fn invalid_json(line_no: u64, source: serde_json::Error) -> Self {
        Self::InvalidJson { line_no, source }
    }
}

impl fmt::Display for RowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "{}", err),
            Self::InvalidJson { line_no, source } => {
                write!(f, "invalid JSON on NDJSON line {line_no}: {source}")
            }
            Self::InvalidJsonMessage { line_no, message } => {
                write!(f, "invalid JSON on NDJSON line {line_no}: {message}")
            }
            Self::LineTooLarge { line_no, len, max } => write!(
                f,
                "NDJSON line {line_no} is too large: {len} bytes exceeds {max} byte limit"
            ),
        }
    }
}

impl std::error::Error for RowError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::InvalidJson { source, .. } => Some(source),
            Self::InvalidJsonMessage { .. } => None,
            Self::LineTooLarge { .. } => None,
        }
    }
}

impl From<std::io::Error> for RowError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}
