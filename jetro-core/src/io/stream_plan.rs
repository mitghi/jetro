//! Source-level row stream plan IR.
//!
//! This IR models expressions rooted at `$.rows()` before they are bound to a
//! concrete source implementation such as NDJSON rows or document-array rows.
//! It deliberately contains stream semantics only; byte/tape and materialized
//! execution details live behind source/projector implementations.

use crate::parse::ast::Expr;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum RowStreamSourceKind {
    DocumentRows,
    NdjsonRows,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum RowStreamDirection {
    Forward,
    Reverse,
}

impl Default for RowStreamDirection {
    fn default() -> Self {
        Self::Forward
    }
}

#[derive(Clone, Debug)]
pub(super) struct RowStreamPlan {
    pub source: RowStreamSourceKind,
    pub direction: RowStreamDirection,
    pub stages: Vec<RowStreamStage>,
}

impl RowStreamPlan {
    pub fn new(source: RowStreamSourceKind) -> Self {
        Self {
            source,
            direction: RowStreamDirection::Forward,
            stages: Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub(super) enum RowStreamStage {
    Filter(Expr),
    DistinctBy(Expr),
    Take(usize),
    Map(Expr),
}

