use super::RowError;
use memchr::memchr;
use std::ops::Range;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NdjsonRowFrame {
    JsonLine,
    DelimitedPayload {
        separator: u8,
        side: PayloadSide,
        null_payload: NullPayload,
    },
}

impl Default for NdjsonRowFrame {
    fn default() -> Self {
        Self::JsonLine
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PayloadSide {
    BeforeSeparator,
    AfterSeparator,
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
        NdjsonRowFrame::DelimitedPayload {
            separator, side, ..
        } => {
            let Some(sep) = memchr(separator, row) else {
                return Err(RowError::MissingPayloadSeparator { line_no, separator });
            };
            match side {
                PayloadSide::BeforeSeparator => 0..sep,
                PayloadSide::AfterSeparator => sep + 1..row.len(),
            }
        }
    };
    let range = trim_range(row, range);
    if range.is_empty() {
        return Err(RowError::EmptyPayload { line_no });
    }

    if let NdjsonRowFrame::DelimitedPayload { null_payload, .. } = frame {
        if &row[range.clone()] == b"null" {
            return match null_payload {
                NullPayload::Skip => Ok(FramePayload::Skip),
                NullPayload::Keep => Ok(FramePayload::Data(range)),
                NullPayload::Error => Err(RowError::NullPayload { line_no }),
            };
        }
    }

    Ok(FramePayload::Data(range))
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
            side: PayloadSide::AfterSeparator,
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
    }
}
