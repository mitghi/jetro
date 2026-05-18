use super::mapped_bytes::{split_line_aligned_ranges, MappedBytes};
use super::ndjson::{collect_row_stream_result, NdjsonOptions, NdjsonParallelism};
use super::ndjson_scan::{for_each_framed_payload_in_range, framed_payload_ranges_in_range};
use super::stream_exec::CompiledRowStream;
use super::stream_plan::{RowStreamDirection, RowStreamParallelism, RowStreamPlan};
use super::stream_types::RowStreamStats;
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
    options: NdjsonOptions,
) -> Result<Option<Val>, JetroEngineError>
where
    P: AsRef<Path>,
{
    Ok(
        collect_rows_stream_file_with_stats(engine, path, plan, options)?
            .map(ParallelRowsResult::into_value),
    )
}

pub(super) struct ParallelRowsResult {
    pub(super) value: Val,
    pub(super) stats: RowStreamStats,
}

impl ParallelRowsResult {
    fn into_value(self) -> Val {
        let _stats = self.stats;
        self.value
    }
}

pub(super) fn collect_rows_stream_file_with_stats<P>(
    engine: &JetroEngine,
    path: P,
    plan: &RowStreamPlan,
    options: NdjsonOptions,
) -> Result<Option<ParallelRowsResult>, JetroEngineError>
where
    P: AsRef<Path>,
{
    let metadata = std::fs::metadata(path.as_ref())?;
    let Some(limit) = parallel_collection_limit(plan, options, metadata.len()) else {
        return Ok(None);
    };

    let bytes = Arc::new(MappedBytes::open(path.as_ref())?);
    let ranges = split_line_aligned_ranges(bytes.as_slice(), TARGET_RANGES_PER_THREAD);
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
        } if limit > 0 => Some(limit),
        _ => None,
    }
}

fn parallel_collection_limit(
    plan: &RowStreamPlan,
    options: NdjsonOptions,
    file_len: u64,
) -> Option<usize> {
    if options.parallelism == NdjsonParallelism::Off {
        return None;
    }
    if file_len < options.parallel_min_bytes || rayon::current_num_threads() <= 1 {
        return None;
    }
    if plan.direction == RowStreamDirection::Reverse && plan.demand.ordered_early_stop {
        return None;
    }
    parallel_limit(plan)
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
    options: NdjsonOptions,
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
            if collect_row_stream_result(engine, 1, result, &mut values)? {
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
    options: NdjsonOptions,
) -> Result<(), JetroEngineError> {
    for_each_framed_payload_in_range(bytes, range, options, |row_range, _| {
        if stream.is_exhausted() {
            return Ok(true);
        }
        let row = bytes[row_range].to_vec();
        let result = stream.apply_owned_row(engine, 1, row)?;
        collect_row_stream_result(engine, 1, result, values)
    })
}

fn collect_line_ranges(
    bytes: &[u8],
    range: Range<usize>,
    direction: RowStreamDirection,
    options: NdjsonOptions,
) -> Result<Vec<Range<usize>>, JetroEngineError> {
    let mut rows = framed_payload_ranges_in_range(bytes, range, options)?;
    if direction == RowStreamDirection::Reverse {
        rows.reverse();
    }
    Ok(rows)
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
    fn reverse_ordered_early_stop_declines_partition_collection() {
        let plan = lower_root_rows_query(
            r#"$.rows().reverse().find($.name == "Ada").first()"#,
            RowStreamSourceKind::NdjsonRows,
        )
        .unwrap()
        .unwrap();

        assert_eq!(
            parallel_collection_limit(
                &plan,
                super::super::ndjson::NdjsonOptions::default().with_parallel_min_bytes(0),
                1024,
            ),
            None
        );
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
    fn reverse_first_match_uses_sequential_file_driver() {
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
        .unwrap();

        let _ = std::fs::remove_file(&path);
        assert!(value.is_none());
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
        std::fs::write(&path, b"{\"id\":1}\n{\"id\":2}\n{\"id\":3}\n{\"id\":4}\n").unwrap();
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
