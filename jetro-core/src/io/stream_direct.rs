use super::ndjson_byte::{
    raw_json_byte_path_value, tape_plan_can_write_byte_row, write_ndjson_byte_tape_plan_row,
    BytePlanWrite, RawFieldValue,
};
use super::ndjson_direct::NdjsonDirectTapePlan;
use super::ndjson_distinct::{raw_distinct_key_bytes, AdaptiveDistinctKeys};
use crate::JetroEngineError;
pub(super) fn direct_map_can_write(plan: &NdjsonDirectTapePlan) -> bool {
    tape_plan_can_write_byte_row(plan)
}
pub(super) fn insert_direct_distinct_key(
    seen: &mut AdaptiveDistinctKeys,
    row: &[u8],
    plan: &NdjsonDirectTapePlan,
) -> Option<bool> {
    const NULL_KEY: &[u8] = b"null";

    let NdjsonDirectTapePlan::RootPath(steps) = plan else {
        return None;
    };
    let key = match raw_json_byte_path_value(row, steps) {
        RawFieldValue::Found(value) => raw_distinct_key_bytes(value)?,
        RawFieldValue::Missing => std::borrow::Cow::Borrowed(NULL_KEY),
        RawFieldValue::Fallback => return None,
    };
    Some(match key {
        std::borrow::Cow::Borrowed(key) => seen.insert_slice(key),
        std::borrow::Cow::Owned(key) => seen.insert(key),
    })
}
pub(super) fn write_direct_map(
    row: &[u8],
    direct: Option<&NdjsonDirectTapePlan>,
) -> Result<Option<Vec<u8>>, JetroEngineError> {
    let Some(plan) = direct else {
        return Ok(None);
    };
    let mut out = Vec::new();
    let mut scratch = Vec::new();
    match write_ndjson_byte_tape_plan_row(&mut out, row, plan, &mut scratch)? {
        BytePlanWrite::Done => Ok(Some(out)),
        BytePlanWrite::Fallback => Ok(None),
    }
}
