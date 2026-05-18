use serde::{Deserialize, Serialize};

/// How much data an explicit inspection call should collect.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum InspectLevel {
    /// Cheap high-level labels.
    Summary,
    /// Adds the physical plan DAG and execution facts.
    Plan,
    /// Adds lowering, pipeline, NDJSON, and direct-plan details.
    Detailed,
}

/// Planning context used for static inspection.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum InspectContext {
    /// Normal value-domain planning.
    Value,
    /// Byte/tape-domain planning for byte-backed documents.
    Bytes,
    /// NDJSON reader capabilities.
    NdjsonReader,
    /// NDJSON file capabilities.
    NdjsonFile,
}

/// Options for explicit query inspection.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct InspectOptions {
    pub level: InspectLevel,
    pub context: InspectContext,
}

impl InspectOptions {
    pub fn summary(context: InspectContext) -> Self {
        Self {
            level: InspectLevel::Summary,
            context,
        }
    }

    pub fn plan(context: InspectContext) -> Self {
        Self {
            level: InspectLevel::Plan,
            context,
        }
    }

    pub fn detailed(context: InspectContext) -> Self {
        Self {
            level: InspectLevel::Detailed,
            context,
        }
    }
}

impl Default for InspectOptions {
    fn default() -> Self {
        Self::plan(InspectContext::Value)
    }
}

/// Complete static inspection report for a query.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct QueryInspection {
    pub query: String,
    pub context: InspectContext,
    pub level: InspectLevel,
    pub summary: InspectionSummary,
    pub logical: Option<LogicalInspection>,
    pub physical: Option<PhysicalInspection>,
    pub pipeline: Option<PipelineInspection>,
    pub ndjson: Option<NdjsonInspection>,
    pub warnings: Vec<String>,
}

/// One-screen summary for quick developer checks.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct InspectionSummary {
    pub root: String,
    pub selected_executor: Option<BackendKind>,
    pub materializes_root: bool,
    pub contains_vm_fallback: bool,
    pub can_run_byte_native: bool,
}

/// Logical/lowering details that are useful before physical execution.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct LogicalInspection {
    pub ast_root: String,
    pub root_shape: String,
    pub notes: Vec<String>,
}

/// Physical DAG exported from the real `QueryPlan`.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct PhysicalInspection {
    pub root: String,
    pub nodes: Vec<PhysicalNodeInspection>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PhysicalNodeInspection {
    pub id: usize,
    pub kind: String,
    pub children: Vec<usize>,
    pub facts: ExecutionFactsInspection,
    pub backends: Vec<BackendInspection>,
    pub detail: Option<String>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExecutionFactsInspection {
    pub can_avoid_root_materialization: bool,
    pub contains_vm_fallback: bool,
    pub can_run_byte_native: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BackendInspection {
    pub backend: BackendKind,
    pub status: BackendStatus,
    pub reason: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum BackendKind {
    ByteNative,
    StructuralIndex,
    TapePath,
    ViewPipeline,
    TapeRows,
    ValView,
    MaterializedSource,
    FastChildren,
    Pipeline,
    Interpreted,
    Vm,
    NdjsonRows,
    NdjsonRowLocal,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum BackendStatus {
    Planned,
    Selected,
    Fallback,
    Rejected,
}

/// Pipeline lowering summary.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct PipelineInspection {
    pub source: String,
    pub stages: Vec<PipelineStageInspection>,
    pub sink: String,
    pub source_demand: String,
    pub fallback_boundary: Option<String>,
    pub execution_path: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PipelineStageInspection {
    pub index: usize,
    pub kind: String,
    pub detail: Option<String>,
}

/// NDJSON-specific static route and row-stream details.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NdjsonInspection {
    pub route_kind: String,
    pub source: String,
    pub fallback_reason: Option<String>,
    pub writer_path: Option<String>,
    pub rows: Option<RowStreamInspection>,
    pub direct_plans: Vec<DirectPlanInspection>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RowStreamInspection {
    pub plan_kind: String,
    pub source: String,
    pub direction: String,
    pub demand: String,
    pub file_strategy: Option<String>,
    pub stages: Vec<PipelineStageInspection>,
    pub sink: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DirectPlanInspection {
    pub kind: String,
    pub detail: Option<String>,
}
