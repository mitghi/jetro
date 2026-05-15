//! Streaming input/output adapters for query execution.
//!
//! NDJSON adapters evaluate each non-empty row independently while reusing the
//! caller's [`crate::JetroEngine`] plan, VM, and tape execution caches.

mod document_rows;
mod ndjson;
#[cfg(feature = "simd-json")]
mod ndjson_byte;
#[cfg(feature = "simd-json")]
mod ndjson_direct;
mod ndjson_distinct;
mod ndjson_frame;
#[cfg(feature = "simd-json")]
#[cfg_attr(not(test), allow(dead_code))]
mod ndjson_hint;
mod ndjson_parallel;
mod ndjson_rev;
#[cfg(feature = "simd-json")]
mod ndjson_stream_cache;
mod source;
mod stream_exec;
#[allow(dead_code)]
mod stream_plan;
mod stream_source;
mod stream_subquery;
mod stream_types;

pub(crate) use document_rows::collect_document_rows;
pub use ndjson::{
    collect_ndjson, collect_ndjson_file, collect_ndjson_file_with_options, collect_ndjson_matches,
    collect_ndjson_matches_file, collect_ndjson_matches_file_with_options,
    collect_ndjson_matches_source, collect_ndjson_matches_source_with_options,
    collect_ndjson_matches_with_options, collect_ndjson_source, collect_ndjson_source_with_options,
    collect_ndjson_with_options, for_each_ndjson, for_each_ndjson_source,
    for_each_ndjson_source_until, for_each_ndjson_source_until_with_options,
    for_each_ndjson_source_with_options, for_each_ndjson_until, for_each_ndjson_until_with_options,
    for_each_ndjson_with_options, run_ndjson, run_ndjson_file, run_ndjson_file_limit,
    run_ndjson_file_limit_with_options, run_ndjson_file_with_options, run_ndjson_limit,
    run_ndjson_limit_with_options, run_ndjson_matches, run_ndjson_matches_file,
    run_ndjson_matches_file_with_options, run_ndjson_matches_source,
    run_ndjson_matches_source_with_options, run_ndjson_matches_with_options, run_ndjson_source,
    run_ndjson_source_limit, run_ndjson_source_limit_with_options, run_ndjson_source_with_options,
    run_ndjson_with_options, NdjsonControl, NdjsonNullOutput, NdjsonOptions, NdjsonPerRowDriver,
};
pub use ndjson_distinct::DistinctFrontFilterKind;
pub use ndjson_frame::{NdjsonRowFrame, NullPayload};
pub use ndjson_rev::{
    collect_ndjson_rev, collect_ndjson_rev_matches, collect_ndjson_rev_matches_with_options,
    collect_ndjson_rev_with_options, for_each_ndjson_rev, for_each_ndjson_rev_with_options,
    run_ndjson_rev, run_ndjson_rev_distinct_by, run_ndjson_rev_distinct_by_with_options,
    run_ndjson_rev_distinct_by_with_stats, run_ndjson_rev_distinct_by_with_stats_and_options,
    run_ndjson_rev_limit, run_ndjson_rev_limit_with_options, run_ndjson_rev_matches,
    run_ndjson_rev_matches_with_options, run_ndjson_rev_with_options, NdjsonRevDistinctStats,
    NdjsonReverseFileDriver,
};
pub use source::NdjsonSource;
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
    EmptyPayload {
        line_no: u64,
    },
    NullPayload {
        line_no: u64,
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
            Self::EmptyPayload { line_no } => {
                write!(f, "NDJSON line {line_no} has an empty framed payload")
            }
            Self::NullPayload { line_no } => {
                write!(f, "NDJSON line {line_no} has a null framed payload")
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
            Self::EmptyPayload { .. } => None,
            Self::NullPayload { .. } => None,
            Self::LineTooLarge { .. } => None,
        }
    }
}

impl From<std::io::Error> for RowError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}
