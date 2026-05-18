//! Opt-in query introspection.
//!
//! This module is deliberately read-only. Normal query execution does not
//! allocate or populate these structures; callers receive them only through
//! explicit inspection APIs.

mod report;
#[allow(dead_code)]
pub(crate) mod physical;
#[allow(dead_code)]
pub(crate) mod pipeline;
#[allow(dead_code)]
pub(crate) mod ndjson;

pub use report::{
    BackendInspection, BackendKind, BackendStatus, DirectPlanInspection, ExecutionFactsInspection,
    InspectContext, InspectLevel, InspectOptions, InspectionSummary, LogicalInspection,
    NdjsonInspection, PhysicalInspection, PhysicalNodeInspection, PipelineInspection,
    PipelineStageInspection, QueryInspection, RowStreamInspection,
};
