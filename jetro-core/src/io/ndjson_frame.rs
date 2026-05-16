use super::RowError;
use memchr::memchr;
use std::ops::Range;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NdjsonRowFrame {
    JsonLine,
    DelimitedPayload {
        separator: u8,
        null_payload: NullPayload,
    },
}

impl Default for NdjsonRowFrame {
    fn default() -> Self {
        Self::JsonLine
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NullPayload {
    Skip,
    Keep,
    Error,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum FramePayload {
    Data(Range<usize>),
    Skip,
}

#[inline]
pub(super) fn frame_payload(
    frame: NdjsonRowFrame,
    line_no: u64,
    row: &[u8],
) -> Result<FramePayload, RowError> {
    let range = match frame {
        NdjsonRowFrame::JsonLine => 0..row.len(),
        NdjsonRowFrame::DelimitedPayload { separator, .. } => {
            let Some(sep) = memchr(separator, row) else {
                return Ok(FramePayload::Skip);
            };
            sep + 1..row.len()
        }
    };
    let range = trim_range(row, range);
    if range.is_empty() {
        return match frame {
            NdjsonRowFrame::JsonLine => Err(RowError::EmptyPayload { line_no }),
            NdjsonRowFrame::DelimitedPayload { .. } => Ok(FramePayload::Skip),
        };
    }

    if let NdjsonRowFrame::DelimitedPayload { null_payload, .. } = frame {
        if &row[range.clone()] == b"null" {
            return match null_payload {
                NullPayload::Skip => Ok(FramePayload::Skip),
                NullPayload::Keep => Ok(FramePayload::Data(range)),
                NullPayload::Error => Err(RowError::NullPayload { line_no }),
            };
        }
        if !payload_starts_like_json(&row[range.clone()]) {
            return Ok(FramePayload::Skip);
        }
    }

    Ok(FramePayload::Data(range))
}

#[inline]
fn payload_starts_like_json(payload: &[u8]) -> bool {
    matches!(
        payload[0],
        b'{' | b'[' | b'"' | b't' | b'f' | b'-' | b'0'..=b'9'
    )
}

#[inline]
fn trim_range(row: &[u8], range: Range<usize>) -> Range<usize> {
    let mut start = range.start;
    let mut end = range.end;
    while start < end && row[start].is_ascii_whitespace() {
        start += 1;
    }
    while end > start && row[end - 1].is_ascii_whitespace() {
        end -= 1;
    }
    start..end
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delimited_payload_skips_null() {
        let frame = NdjsonRowFrame::DelimitedPayload {
            separator: b'|',
            null_payload: NullPayload::Skip,
        };

        assert_eq!(
            frame_payload(frame, 1, b"k|null").unwrap(),
            FramePayload::Skip
        );
        assert_eq!(
            frame_payload(frame, 2, br#"k| {"id":1} "#).unwrap(),
            FramePayload::Data(3..11)
        );
        assert_eq!(frame_payload(frame, 3, b"k|").unwrap(), FramePayload::Skip);
        assert_eq!(
            frame_payload(frame, 4, b"no-separator").unwrap(),
            FramePayload::Skip
        );
        assert_eq!(
            frame_payload(frame, 5, b"k|not-json").unwrap(),
            FramePayload::Skip
        );
    }
}
