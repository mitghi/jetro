use super::report::QueryInspection;
use std::fmt::Write;

impl QueryInspection {
    /// Renders the inspection as a compact tree for humans.
    pub fn format_tree(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "query:");
        let _ = writeln!(out, "  {}", self.query);
        let _ = writeln!(out, "context:");
        let _ = writeln!(out, "  {:?} / {:?}", self.context, self.level);
        let _ = writeln!(out, "summary:");
        let _ = writeln!(out, "  root: {}", self.summary.root);
        let _ = writeln!(
            out,
            "  selected executor: {}",
            self.summary
                .selected_executor
                .map(|backend| format!("{backend:?}"))
                .unwrap_or_else(|| "unknown".to_string())
        );
        let _ = writeln!(
            out,
            "  materializes root: {}",
            self.summary.materializes_root
        );
        let _ = writeln!(
            out,
            "  contains vm fallback: {}",
            self.summary.contains_vm_fallback
        );
        let _ = writeln!(out, "  byte native: {}", self.summary.can_run_byte_native);

        if let Some(logical) = &self.logical {
            let _ = writeln!(out, "logical:");
            let _ = writeln!(out, "  ast root: {}", logical.ast_root);
            let _ = writeln!(out, "  root shape: {}", logical.root_shape);
            for note in &logical.notes {
                let _ = writeln!(out, "  note: {note}");
            }
        }

        if let Some(physical) = &self.physical {
            let _ = writeln!(out, "physical:");
            let _ = writeln!(out, "  root: {}", physical.root);
            for node in &physical.nodes {
                let _ = writeln!(out, "  node {}: {}", node.id, node.kind);
                if !node.children.is_empty() {
                    let _ = writeln!(out, "    children: {:?}", node.children);
                }
                if let Some(detail) = &node.detail {
                    let _ = writeln!(out, "    detail: {detail}");
                }
                let _ = writeln!(
                    out,
                    "    facts: byte_native={}, avoid_root_materialization={}, vm_fallback={}",
                    node.facts.can_run_byte_native,
                    node.facts.can_avoid_root_materialization,
                    node.facts.contains_vm_fallback
                );
                if !node.backends.is_empty() {
                    let backends = node
                        .backends
                        .iter()
                        .map(|backend| format!("{:?}:{:?}", backend.backend, backend.status))
                        .collect::<Vec<_>>()
                        .join(", ");
                    let _ = writeln!(out, "    backends: {backends}");
                }
            }
        }

        if let Some(pipeline) = &self.pipeline {
            let _ = writeln!(out, "pipeline:");
            let _ = writeln!(out, "  source: {}", pipeline.source);
            let _ = writeln!(out, "  demand: {}", pipeline.source_demand);
            let _ = writeln!(out, "  sink: {}", pipeline.sink);
            if let Some(path) = &pipeline.execution_path {
                let _ = writeln!(out, "  execution path: {path}");
            }
            if let Some(boundary) = &pipeline.fallback_boundary {
                let _ = writeln!(out, "  fallback boundary: {boundary}");
            }
            for stage in &pipeline.stages {
                let _ = writeln!(out, "  stage {}: {}", stage.index, stage.kind);
                if let Some(detail) = &stage.detail {
                    let _ = writeln!(out, "    {detail}");
                }
            }
        }

        if let Some(ndjson) = &self.ndjson {
            let _ = writeln!(out, "ndjson:");
            let _ = writeln!(out, "  route: {}", ndjson.route_kind);
            let _ = writeln!(out, "  source: {}", ndjson.source);
            if let Some(reason) = &ndjson.fallback_reason {
                let _ = writeln!(out, "  fallback: {reason}");
            }
            if let Some(path) = &ndjson.writer_path {
                let _ = writeln!(out, "  writer path: {path}");
            }
            if let Some(rows) = &ndjson.rows {
                let _ = writeln!(out, "  rows:");
                let _ = writeln!(out, "    plan: {}", rows.plan_kind);
                let _ = writeln!(out, "    source: {}", rows.source);
                let _ = writeln!(out, "    direction: {}", rows.direction);
                let _ = writeln!(out, "    demand: {}", rows.demand);
                if let Some(strategy) = &rows.file_strategy {
                    let _ = writeln!(out, "    file strategy: {strategy}");
                }
                let _ = writeln!(out, "    sink: {}", rows.sink);
                for stage in &rows.stages {
                    let _ = writeln!(out, "    stage {}: {}", stage.index, stage.kind);
                    if let Some(detail) = &stage.detail {
                        let _ = writeln!(out, "      {detail}");
                    }
                }
            }
            for plan in &ndjson.direct_plans {
                let _ = writeln!(out, "  direct: {}", plan.kind);
                if let Some(detail) = &plan.detail {
                    let _ = writeln!(out, "    {detail}");
                }
            }
        }

        if !self.warnings.is_empty() {
            let _ = writeln!(out, "warnings:");
            for warning in &self.warnings {
                let _ = writeln!(out, "  {warning}");
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use crate::introspect::{
        BackendKind, InspectContext, InspectLevel, InspectionSummary, QueryInspection,
    };

    #[test]
    fn tree_renderer_includes_key_sections() {
        let report = QueryInspection {
            query: "$.a".to_string(),
            context: InspectContext::Bytes,
            level: InspectLevel::Plan,
            summary: InspectionSummary {
                root: "node:0".to_string(),
                selected_executor: Some(BackendKind::TapePath),
                materializes_root: false,
                contains_vm_fallback: false,
                can_run_byte_native: true,
            },
            logical: None,
            physical: None,
            pipeline: None,
            ndjson: None,
            warnings: Vec::new(),
        };

        let text = report.format_tree();

        assert!(text.contains("query:"));
        assert!(text.contains("summary:"));
        assert!(text.contains("TapePath"));
    }
}
