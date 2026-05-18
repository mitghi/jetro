use super::ndjson::{ndjson_writer_path_kind, NdjsonOptions, NdjsonWriterPathKind};
use super::ndjson_frame::NdjsonRowFrame;
use super::ndjson_rows::{ndjson_rows_plan_kind, NdjsonRowsPlanKind};
use crate::{JetroEngine, JetroEngineError};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NdjsonSourceMode {
    Reader,
    File,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NdjsonRouteKind {
    RowLocal,
    RowsStream,
    RowsFanout,
    RowsSubquery,
    UnsupportedRows,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NdjsonRouteExplain {
    pub kind: NdjsonRouteKind,
    pub source: NdjsonSourceCaps,
    pub writer_path: Option<NdjsonWriterPathKind>,
    pub rows_plan: Option<NdjsonRowsPlanKind>,
    pub fallback_reason: Option<&'static str>,
}

pub fn ndjson_explain(
    engine: &JetroEngine,
    source: NdjsonSourceMode,
    query: &str,
    options: NdjsonOptions,
) -> Result<NdjsonRouteExplain, JetroEngineError> {
    let source = match source {
        NdjsonSourceMode::Reader => NdjsonSourceCaps::reader(options),
        NdjsonSourceMode::File => NdjsonSourceCaps::file(options),
    };
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
                fallback_reason: Some("rows plan requires a file-backed NDJSON source"),
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

        let rows = ndjson_explain(
            &engine,
            NdjsonSourceMode::File,
            "$.rows().take(1)",
            NdjsonOptions::default(),
        )
        .unwrap();
        assert_eq!(rows.kind, NdjsonRouteKind::RowsStream);
        assert_eq!(rows.rows_plan, Some(NdjsonRowsPlanKind::Stream));
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
            Some("rows plan requires a file-backed NDJSON source")
        );
    }
}
