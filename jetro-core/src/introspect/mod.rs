//! Opt-in query introspection.
//!
//! This module is deliberately read-only. Normal query execution does not
//! allocate or populate these structures; callers receive them only through
//! explicit inspection APIs.

mod inspect;
mod report;
pub(crate) mod physical;
pub(crate) mod pipeline;
pub(crate) mod ndjson;
mod render;

pub(crate) use inspect::inspect_query;
pub use report::{
    BackendInspection, BackendKind, BackendStatus, DirectPlanInspection, ExecutionFactsInspection,
    InspectContext, InspectLevel, InspectOptions, InspectionSummary, LogicalInspection,
    NdjsonInspection, PhysicalInspection, PhysicalNodeInspection, PipelineInspection,
    PipelineStageInspection, QueryInspection, RowStreamInspection,
};
