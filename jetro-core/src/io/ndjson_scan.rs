use super::ndjson::NdjsonOptions;
use super::ndjson_frame::{frame_payload, FramePayload};
use crate::JetroEngineError;
use std::ops::Range;

pub(super) fn for_each_framed_payload_in_range<F>(
    bytes: &[u8],
    range: Range<usize>,
    options: NdjsonOptions,
    mut visit: F,
) -> Result<(), JetroEngineError>
where
    F: FnMut(Range<usize>, &[u8]) -> Result<bool, JetroEngineError>,
{
    let mut cursor = range.start;
    while cursor < range.end {
        let start = cursor;
        let rel_end = memchr::memchr(b'\n', &bytes[cursor..range.end]);
        let mut end = rel_end.map_or(range.end, |pos| cursor + pos);
        cursor = rel_end.map_or(range.end, |pos| cursor + pos + 1);

        if end > start && bytes[end - 1] == b'\r' {
            end -= 1;
        }
        if bytes[start..end]
            .iter()
            .all(|byte| byte.is_ascii_whitespace())
        {
            continue;
        }

        let payload = match frame_payload(options.row_frame, 1, &bytes[start..end])? {
            FramePayload::Data(payload) => start + payload.start..start + payload.end,
            FramePayload::Skip => continue,
        };
        if visit(payload.clone(), &bytes[payload])? {
            break;
        }
    }
    Ok(())
}

pub(super) fn framed_payload_ranges_in_range(
    bytes: &[u8],
    range: Range<usize>,
    options: NdjsonOptions,
) -> Result<Vec<Range<usize>>, JetroEngineError> {
    let mut rows = Vec::new();
    for_each_framed_payload_in_range(bytes, range, options, |row_range, _| {
        rows.push(row_range);
        Ok(false)
    })?;
    Ok(rows)
}
