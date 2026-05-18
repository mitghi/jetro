use super::ndjson::{ndjson_writer_path_kind, NdjsonOptions, NdjsonWriterPathKind};
use super::ndjson_frame::NdjsonRowFrame;
use super::ndjson_rows::{ndjson_rows_plan_kind, NdjsonRowsPlanKind};
use crate::{JetroEngine, JetroEngineError};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NdjsonSourceMode {
    Reader,
    File,
}

impl std::fmt::Display for NdjsonSourceMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Reader => "reader",
            Self::File => "file",
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NdjsonSourceCaps {
    pub mode: NdjsonSourceMode,
    pub forward: bool,
    pub reverse: bool,
    pub mmap: bool,
    pub partitionable: bool,
    pub framed_payload: bool,
}

impl NdjsonSourceCaps {
    pub fn for_mode(mode: NdjsonSourceMode, options: NdjsonOptions) -> Self {
        match mode {
            NdjsonSourceMode::Reader => Self::reader(options),
            NdjsonSourceMode::File => Self::file(options),
        }
    }

    pub fn reader(options: NdjsonOptions) -> Self {
        Self {
            mode: NdjsonSourceMode::Reader,
            forward: true,
            reverse: false,
            mmap: false,
            partitionable: false,
            framed_payload: options.row_frame != NdjsonRowFrame::JsonLine,
        }
    }

    pub fn file(options: NdjsonOptions) -> Self {
        Self {
            mode: NdjsonSourceMode::File,
            forward: true,
            reverse: true,
            mmap: true,
            partitionable: true,
            framed_payload: options.row_frame != NdjsonRowFrame::JsonLine,
        }
    }
}

impl std::fmt::Display for NdjsonSourceCaps {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.mode)?;
        if self.reverse {
            f.write_str("+reverse")?;
        }
        if self.mmap {
            f.write_str("+mmap")?;
        }
        if self.partitionable {
            f.write_str("+partitionable")?;
        }
        if self.framed_payload {
            f.write_str("+framed-payload")?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NdjsonRouteKind {
    RowLocal,
    RowsStream,
    RowsFanout,
    RowsSubquery,
    UnsupportedRows,
}

impl std::fmt::Display for NdjsonRouteKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::RowLocal => "row-local",
            Self::RowsStream => "rows-stream",
            Self::RowsFanout => "rows-fanout",
            Self::RowsSubquery => "rows-subquery",
            Self::UnsupportedRows => "unsupported-rows",
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NdjsonFallbackReason {
    FileBackedRowsRequired,
}

impl std::fmt::Display for NdjsonFallbackReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::FileBackedRowsRequired => "rows plan requires a file-backed NDJSON source",
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NdjsonRouteExplain {
    pub kind: NdjsonRouteKind,
    pub source: NdjsonSourceCaps,
    pub writer_path: Option<NdjsonWriterPathKind>,
    pub rows_plan: Option<NdjsonRowsPlanKind>,
    pub fallback_reason: Option<NdjsonFallbackReason>,
}

impl NdjsonRouteExplain {
    pub fn is_rows_route(&self) -> bool {
        self.rows_plan.is_some()
    }

    pub fn is_supported(&self) -> bool {
        self.kind != NdjsonRouteKind::UnsupportedRows
    }

    pub fn unsupported_message(&self) -> Option<String> {
        if self.is_supported() {
            return None;
        }
        Some(
            self.fallback_reason
                .map(|reason| reason.to_string())
                .unwrap_or_else(|| "unsupported $.rows() NDJSON route".to_string()),
        )
    }
}

pub fn ndjson_explain(
    engine: &JetroEngine,
    source: NdjsonSourceMode,
    query: &str,
    options: NdjsonOptions,
) -> Result<NdjsonRouteExplain, JetroEngineError> {
    let source = NdjsonSourceCaps::for_mode(source, options);
    let rows_plan = ndjson_rows_plan_kind(query)?;
    if let Some(rows_plan) = rows_plan {
        let file_required = matches!(
            rows_plan,
            NdjsonRowsPlanKind::Fanout | NdjsonRowsPlanKind::Subquery
        );
        if file_required && source.mode == NdjsonSourceMode::Reader {
            return Ok(NdjsonRouteExplain {
                kind: NdjsonRouteKind::UnsupportedRows,
                source,
                writer_path: None,
                rows_plan: Some(rows_plan),
                fallback_reason: Some(NdjsonFallbackReason::FileBackedRowsRequired),
            });
        }
        let kind = match rows_plan {
            NdjsonRowsPlanKind::Stream => NdjsonRouteKind::RowsStream,
            NdjsonRowsPlanKind::Fanout => NdjsonRouteKind::RowsFanout,
            NdjsonRowsPlanKind::Subquery => NdjsonRouteKind::RowsSubquery,
        };
        return Ok(NdjsonRouteExplain {
            kind,
            source,
            writer_path: None,
            rows_plan: Some(rows_plan),
            fallback_reason: None,
        });
    }

    Ok(NdjsonRouteExplain {
        kind: NdjsonRouteKind::RowLocal,
        source,
        writer_path: ndjson_writer_path_kind(engine, query),
        rows_plan: None,
        fallback_reason: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_explain_reports_row_local_and_rows_modes() {
        let engine = JetroEngine::new();
        let row = ndjson_explain(
            &engine,
            NdjsonSourceMode::Reader,
            "$.name",
            NdjsonOptions::default(),
        )
        .unwrap();
        assert_eq!(row.kind, NdjsonRouteKind::RowLocal);
        assert_eq!(row.writer_path, Some(NdjsonWriterPathKind::ByteExpr));
        assert_eq!(row.source.to_string(), "reader");

        let rows = ndjson_explain(
            &engine,
            NdjsonSourceMode::File,
            "$.rows().take(1)",
            NdjsonOptions::default(),
        )
        .unwrap();
        assert_eq!(rows.kind, NdjsonRouteKind::RowsStream);
        assert_eq!(rows.rows_plan, Some(NdjsonRowsPlanKind::Stream));
        assert_eq!(rows.source.to_string(), "file+reverse+mmap+partitionable");
    }

    #[test]
    fn route_explain_marks_reader_rows_subquery_unsupported() {
        let engine = JetroEngine::new();
        let route = ndjson_explain(
            &engine,
            NdjsonSourceMode::Reader,
            r#"{head: $.rows().take(1)}"#,
            NdjsonOptions::default(),
        )
        .unwrap();
        assert_eq!(route.kind, NdjsonRouteKind::UnsupportedRows);
        assert_eq!(
            route.fallback_reason,
            Some(NdjsonFallbackReason::FileBackedRowsRequired)
        );
        assert_eq!(route.kind.to_string(), "unsupported-rows");
        assert_eq!(
            route.fallback_reason.unwrap().to_string(),
            "rows plan requires a file-backed NDJSON source"
        );
        assert!(!route.is_supported());
        assert_eq!(
            route.unsupported_message().unwrap(),
            "rows plan requires a file-backed NDJSON source"
        );
    }
}
