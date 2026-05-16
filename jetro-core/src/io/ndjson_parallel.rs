use super::mapped_bytes::MappedBytes;
use super::ndjson::{parse_row, row_eval_error, NdjsonParallelism};
use super::ndjson_frame::{frame_payload, FramePayload};
use super::stream_exec::CompiledRowStream;
use super::stream_plan::{RowStreamDirection, RowStreamParallelism, RowStreamPlan};
use super::stream_types::{RowStreamRowResult, RowStreamStats};
use crate::data::value::Val;
use crate::{JetroEngine, JetroEngineError};
use rayon::prelude::*;
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

const TARGET_RANGES_PER_THREAD: usize = 4;

pub(super) fn collect_rows_stream_file<P>(
    engine: &JetroEngine,
    path: P,
    plan: &RowStreamPlan,
    options: super::ndjson::NdjsonOptions,
) -> Result<Option<Val>, JetroEngineError>
where
    P: AsRef<Path>,
{
    Ok(collect_rows_stream_file_with_stats(engine, path, plan, options)?.map(|result| result.value))
}

#[cfg_attr(not(test), allow(dead_code))]
pub(super) struct ParallelRowsResult {
    pub(super) value: Val,
    pub(super) stats: RowStreamStats,
}

pub(super) fn collect_rows_stream_file_with_stats<P>(
    engine: &JetroEngine,
    path: P,
    plan: &RowStreamPlan,
    options: super::ndjson::NdjsonOptions,
) -> Result<Option<ParallelRowsResult>, JetroEngineError>
where
    P: AsRef<Path>,
{
    let Some(limit) = parallel_limit(plan) else {
        return Ok(None);
    };
    if options.parallelism == NdjsonParallelism::Off {
        return Ok(None);
    }
    let metadata = std::fs::metadata(path.as_ref())?;
    if metadata.len() < options.parallel_min_bytes || rayon::current_num_threads() <= 1 {
        return Ok(None);
    }

    let bytes = Arc::new(MappedBytes::open(path.as_ref())?);
    let ranges = split_line_aligned_ranges(bytes.as_slice());
    if ranges.len() <= 1 {
        return Ok(None);
    }

    let mut parts = ranges
        .into_par_iter()
        .map(|range| scan_partition(engine, bytes.clone(), range, plan, options))
        .collect::<Result<Vec<_>, _>>()?;
    parts.sort_by_key(|part| part.ordinal);
    if plan.direction == RowStreamDirection::Reverse {
        parts.reverse();
    }

    let mut out = Vec::new();
    let mut stats = RowStreamStats {
        source: plan.source,
        direction: plan.direction,
        parallel_partitions: parts.len(),
        ..RowStreamStats::default()
    };
    for mut part in parts {
        stats.merge_partition(&part.stats);
        out.append(&mut part.values);
        if out.len() >= limit {
            out.truncate(limit);
            break;
        }
    }

    let value = if limit == 1 {
        out.into_iter().next().unwrap_or(Val::Null)
    } else {
        Val::Arr(Arc::new(out))
    };
    Ok(Some(ParallelRowsResult { value, stats }))
}

fn parallel_limit(plan: &RowStreamPlan) -> Option<usize> {
    match plan.demand.parallel {
        RowStreamParallelism::PartitionFilter {
            retained_limit: Some(limit),
            ..
        }
        | RowStreamParallelism::PartitionMap {
            retained_limit: Some(limit),
            ..
        } if limit > 0 => Some(limit),
        _ => None,
    }
}

struct PartitionOutput {
    ordinal: usize,
    values: Vec<Val>,
    stats: RowStreamStats,
}

fn scan_partition(
    engine: &JetroEngine,
    bytes: Arc<MappedBytes>,
    range: Range<usize>,
    plan: &RowStreamPlan,
    options: super::ndjson::NdjsonOptions,
) -> Result<PartitionOutput, JetroEngineError> {
    let ordinal = range.start;
    let mut stream = CompiledRowStream::new(plan);
    let mut values = Vec::new();
    let bytes = bytes.as_slice();
    if plan.direction == RowStreamDirection::Forward {
        scan_forward_partition_rows(engine, bytes, range, &mut stream, &mut values, options)?;
    } else {
        let lines = collect_line_ranges(bytes, range, plan.direction, options)?;
        for row_range in lines {
            if stream.is_exhausted() {
                break;
            }
            let row = bytes[row_range].to_vec();
            let result = stream.apply_owned_row(engine, 1, row)?;
            if collect_result(engine, result, &mut values)? {
                break;
            }
        }
    }

    Ok(PartitionOutput {
        ordinal,
        values,
        stats: stream.stats().clone(),
    })
}

fn scan_forward_partition_rows(
    engine: &JetroEngine,
    bytes: &[u8],
    range: Range<usize>,
    stream: &mut CompiledRowStream,
    values: &mut Vec<Val>,
    options: super::ndjson::NdjsonOptions,
) -> Result<(), JetroEngineError> {
    let mut cursor = range.start;
    while cursor < range.end && !stream.is_exhausted() {
        let start = cursor;
        let rel_end = memchr::memchr(b'\n', &bytes[cursor..range.end]);
        let mut end = rel_end.map_or(range.end, |pos| cursor + pos);
        cursor = rel_end.map_or(range.end, |pos| cursor + pos + 1);
        if end > start && bytes[end - 1] == b'\r' {
            end -= 1;
        }
        if bytes[start..end].iter().all(|b| b.is_ascii_whitespace()) {
            continue;
        }
        let row_range = match frame_payload(options.row_frame, 1, &bytes[start..end])? {
            FramePayload::Data(payload) => start + payload.start..start + payload.end,
            FramePayload::Skip => continue,
        };
        let row = bytes[row_range].to_vec();
        let result = stream.apply_owned_row(engine, 1, row)?;
        if collect_result(engine, result, values)? {
            break;
        }
    }
    Ok(())
}

fn collect_result(
    engine: &JetroEngine,
    result: RowStreamRowResult,
    values: &mut Vec<Val>,
) -> Result<bool, JetroEngineError> {
    match result {
        RowStreamRowResult::Emit(value) => values.push(value),
        RowStreamRowResult::EmitBytes(row) => {
            let document = parse_row(engine, 1, row)?;
            let value = document
                .root_val_with(engine.keys())
                .map_err(|err| row_eval_error(1, err))?;
            values.push(value);
        }
        RowStreamRowResult::Skip => {}
        RowStreamRowResult::Stop => return Ok(true),
    }
    Ok(false)
}

fn collect_line_ranges(
    bytes: &[u8],
    range: Range<usize>,
    direction: RowStreamDirection,
    options: super::ndjson::NdjsonOptions,
) -> Result<Vec<Range<usize>>, JetroEngineError> {
    let mut rows = Vec::new();
    let mut cursor = range.start;
    while cursor < range.end {
        let start = cursor;
        let rel_end = memchr::memchr(b'\n', &bytes[cursor..range.end]);
        let mut end = rel_end.map_or(range.end, |pos| cursor + pos);
        cursor = rel_end.map_or(range.end, |pos| cursor + pos + 1);
        if end > start && bytes[end - 1] == b'\r' {
            end -= 1;
        }
        if bytes[start..end].iter().all(|b| b.is_ascii_whitespace()) {
            continue;
        }
        match frame_payload(options.row_frame, 1, &bytes[start..end])? {
            FramePayload::Data(payload) => rows.push(start + payload.start..start + payload.end),
            FramePayload::Skip => {}
        }
    }
    if direction == RowStreamDirection::Reverse {
        rows.reverse();
    }
    Ok(rows)
}

fn split_line_aligned_ranges(bytes: &[u8]) -> Vec<Range<usize>> {
    let target = rayon::current_num_threads().max(1) * TARGET_RANGES_PER_THREAD;
    let approx = (bytes.len() / target.max(1)).max(1);
    let mut ranges = Vec::new();
    let mut start = 0usize;

    while start < bytes.len() {
        let mut end = (start + approx).min(bytes.len());
        if end < bytes.len() {
            while end < bytes.len() && bytes[end - 1] != b'\n' {
                end += 1;
            }
        }
        if end > start {
            ranges.push(start..end);
        }
        start = end;
    }

    ranges
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::stream_plan::{lower_root_rows_query, RowStreamSourceKind};
    use crate::JetroEngine;
    use serde_json::json;

    #[test]
    fn parallel_limit_uses_generic_demand_metadata() {
        let plan = lower_root_rows_query(
            r#"$.rows().reverse().find($.name == "Ada").first()"#,
            RowStreamSourceKind::NdjsonRows,
        )
        .unwrap()
        .unwrap();

        assert_eq!(parallel_limit(&plan), Some(1));
    }

    #[test]
    fn line_ranges_preserve_reverse_partition_order() {
        let rows = collect_line_ranges(
            b"{\"a\":1}\n{\"a\":2}\n",
            0..16,
            RowStreamDirection::Reverse,
            super::super::ndjson::NdjsonOptions::default(),
        )
        .unwrap();
        assert_eq!(&rows, &[8..15, 0..7]);
    }

    #[test]
    fn forced_parallel_collects_reverse_first_match() {
        let engine = JetroEngine::new();
        let path = std::env::temp_dir().join(format!(
            "jetro-parallel-{}-{}.ndjson",
            std::process::id(),
            "reverse-first"
        ));
        std::fs::write(
            &path,
            b"{\"id\":1,\"name\":\"old\"}\n{\"id\":2,\"name\":\"target\"}\n{\"id\":3,\"name\":\"new\"}\n",
        )
        .unwrap();
        let plan = lower_root_rows_query(
            r#"$.rows().reverse().find($.name == "target").first()"#,
            RowStreamSourceKind::NdjsonRows,
        )
        .unwrap()
        .unwrap();

        let value = collect_rows_stream_file(
            &engine,
            &path,
            &plan,
            super::super::ndjson::NdjsonOptions::default().with_parallel_min_bytes(0),
        )
        .unwrap()
        .expect("forced parallel path should run");

        let _ = std::fs::remove_file(&path);
        assert_eq!(serde_json::Value::from(value), json!({"id": 2, "name": "target"}));
    }

    #[test]
    fn forced_parallel_merges_forward_filter_take() {
        let engine = JetroEngine::new();
        let path = std::env::temp_dir().join(format!(
            "jetro-parallel-{}-{}.ndjson",
            std::process::id(),
            "filter-take"
        ));
        std::fs::write(
            &path,
            b"{\"id\":1,\"active\":false}\n{\"id\":2,\"active\":true}\n{\"id\":3,\"active\":true}\n{\"id\":4,\"active\":true}\n",
        )
        .unwrap();
        let plan = lower_root_rows_query(
            "$.rows().filter($.active).take(2)",
            RowStreamSourceKind::NdjsonRows,
        )
        .unwrap()
        .unwrap();

        let value = collect_rows_stream_file(
            &engine,
            &path,
            &plan,
            super::super::ndjson::NdjsonOptions::default().with_parallel_min_bytes(0),
        )
        .unwrap()
        .expect("forced parallel path should run");

        let _ = std::fs::remove_file(&path);
        assert_eq!(
            serde_json::Value::from(value),
            json!([
                {"id": 2, "active": true},
                {"id": 3, "active": true}
            ])
        );
    }

    #[test]
    fn retained_map_take_stays_sequential() {
        let engine = JetroEngine::new();
        let path = std::env::temp_dir().join(format!(
            "jetro-parallel-{}-{}.ndjson",
            std::process::id(),
            "map-take"
        ));
        std::fs::write(
            &path,
            b"{\"id\":1}\n{\"id\":2}\n{\"id\":3}\n{\"id\":4}\n",
        )
        .unwrap();
        let plan = lower_root_rows_query(
            "$.rows().reverse().map($.id).take(3)",
            RowStreamSourceKind::NdjsonRows,
        )
        .unwrap()
        .unwrap();

        let value = collect_rows_stream_file(
            &engine,
            &path,
            &plan,
            super::super::ndjson::NdjsonOptions::default().with_parallel_min_bytes(0),
        )
        .unwrap();

        let _ = std::fs::remove_file(&path);
        assert!(value.is_none());
    }

    #[test]
    fn forced_parallel_reports_partition_stats() {
        let engine = JetroEngine::new();
        let path = std::env::temp_dir().join(format!(
            "jetro-parallel-{}-{}.ndjson",
            std::process::id(),
            "stats"
        ));
        std::fs::write(
            &path,
            b"{\"id\":1,\"active\":false}\n{\"id\":2,\"active\":true}\n{\"id\":3,\"active\":true}\n{\"id\":4,\"active\":true}\n",
        )
        .unwrap();
        let plan = lower_root_rows_query(
            "$.rows().filter($.active).map($.id).take(2)",
            RowStreamSourceKind::NdjsonRows,
        )
        .unwrap()
        .unwrap();

        let result = collect_rows_stream_file_with_stats(
            &engine,
            &path,
            &plan,
            super::super::ndjson::NdjsonOptions::default().with_parallel_min_bytes(0),
        )
        .unwrap()
        .expect("forced parallel path should run");

        let _ = std::fs::remove_file(&path);
        assert_eq!(serde_json::Value::from(result.value), json!([2, 3]));
        assert!(result.stats.parallel_partitions > 1);
        assert!(result.stats.rows_scanned >= 3);
        assert_eq!(
            result.stats.direct_filter_rows + result.stats.fallback_filter_rows,
            result.stats.rows_scanned
        );
        assert_eq!(result.stats.rows_filtered, 1);
        assert!(result.stats.direct_project_rows + result.stats.fallback_project_rows >= 2);
    }

    #[test]
    fn parallel_policy_off_disables_partition_collection() {
        let engine = JetroEngine::new();
        let path = std::env::temp_dir().join(format!(
            "jetro-parallel-{}-{}.ndjson",
            std::process::id(),
            "off"
        ));
        std::fs::write(
            &path,
            b"{\"id\":1,\"active\":false}\n{\"id\":2,\"active\":true}\n{\"id\":3,\"active\":true}\n",
        )
        .unwrap();
        let plan = lower_root_rows_query(
            "$.rows().filter($.active).take(1)",
            RowStreamSourceKind::NdjsonRows,
        )
        .unwrap()
        .unwrap();

        let value = collect_rows_stream_file(
            &engine,
            &path,
            &plan,
            super::super::ndjson::NdjsonOptions::default()
                .with_parallel_min_bytes(0)
                .with_parallelism(super::super::ndjson::NdjsonParallelism::Off),
        )
        .unwrap();

        let _ = std::fs::remove_file(&path);
        assert!(value.is_none());
    }
}
