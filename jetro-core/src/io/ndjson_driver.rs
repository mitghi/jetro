use super::ndjson::{non_ws_range, strip_initial_bom, trim_line_ending, NdjsonOptions};
use super::ndjson_frame::{frame_payload, FramePayload, NdjsonRowFrame};
use super::RowError;
use memchr::memchr;
use std::io::BufRead;

/// Forward-only per-row NDJSON reader.
pub struct NdjsonPerRowDriver<R> {
    pub(super) reader: R,
    pub(super) line_no: u64,
    pub(super) max_line_len: usize,
    pub(super) row_frame: NdjsonRowFrame,
}

impl<R: BufRead> NdjsonPerRowDriver<R> {
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            line_no: 0,
            max_line_len: super::ndjson::DEFAULT_MAX_LINE_LEN,
            row_frame: NdjsonRowFrame::JsonLine,
        }
    }

    pub fn with_options(mut self, options: NdjsonOptions) -> Self {
        self.max_line_len = options.max_line_len;
        self.row_frame = options.row_frame;
        self
    }

    pub fn with_max_line_len(mut self, max_line_len: usize) -> Self {
        self.max_line_len = max_line_len;
        self
    }

    pub fn with_row_frame(mut self, row_frame: NdjsonRowFrame) -> Self {
        self.row_frame = row_frame;
        self
    }

    pub fn line_no(&self) -> u64 {
        self.line_no
    }

    pub fn read_next_nonempty<'a>(
        &mut self,
        buf: &'a mut Vec<u8>,
    ) -> Result<Option<(u64, &'a [u8])>, RowError> {
        loop {
            buf.clear();
            let read = self.read_physical_line(buf)?;
            if read == 0 {
                return Ok(None);
            }
            self.line_no += 1;

            strip_initial_bom(self.line_no, buf);
            trim_line_ending(buf);

            let (start, end) = non_ws_range(buf);
            if start == end {
                continue;
            }

            let len = end - start;
            if len > self.max_line_len {
                return Err(RowError::LineTooLarge {
                    line_no: self.line_no,
                    len,
                    max: self.max_line_len,
                });
            }

            match frame_payload(self.row_frame, self.line_no, &buf[start..end])? {
                FramePayload::Data(range) => {
                    return Ok(Some((
                        self.line_no,
                        &buf[start + range.start..start + range.end],
                    )));
                }
                FramePayload::Skip => continue,
            }
        }
    }

    pub fn read_next_owned(
        &mut self,
        buf: &mut Vec<u8>,
    ) -> Result<Option<(u64, Vec<u8>)>, RowError> {
        loop {
            buf.clear();
            let read = self.read_physical_line(buf)?;
            if read == 0 {
                return Ok(None);
            }
            self.line_no += 1;

            strip_initial_bom(self.line_no, buf);
            trim_line_ending(buf);

            let (start, end) = non_ws_range(buf);
            if start == end {
                continue;
            }

            let len = end - start;
            if len > self.max_line_len {
                return Err(RowError::LineTooLarge {
                    line_no: self.line_no,
                    len,
                    max: self.max_line_len,
                });
            }

            let payload = match frame_payload(self.row_frame, self.line_no, &buf[start..end])? {
                FramePayload::Data(range) => start + range.start..start + range.end,
                FramePayload::Skip => continue,
            };
            if payload.start > 0 || payload.end < buf.len() {
                buf.copy_within(payload.clone(), 0);
                buf.truncate(payload.end - payload.start);
            }

            let capacity = buf.capacity();
            return Ok(Some((
                self.line_no,
                std::mem::replace(buf, Vec::with_capacity(capacity)),
            )));
        }
    }

    pub(super) fn read_physical_line(&mut self, buf: &mut Vec<u8>) -> Result<usize, RowError> {
        loop {
            let available = self.reader.fill_buf()?;
            if available.is_empty() {
                return Ok(buf.len());
            }

            if let Some(pos) = memchr(b'\n', available) {
                buf.extend_from_slice(&available[..=pos]);
                self.reader.consume(pos + 1);
                self.check_physical_line_len(buf.len())?;
                return Ok(buf.len());
            }

            let len = available.len();
            buf.extend_from_slice(available);
            self.reader.consume(len);
            self.check_physical_line_len(buf.len())?;
        }
    }

    fn check_physical_line_len(&self, len: usize) -> Result<(), RowError> {
        let hard_max = self.max_line_len.saturating_add(2);
        if len > hard_max {
            return Err(RowError::LineTooLarge {
                line_no: self.line_no + 1,
                len,
                max: self.max_line_len,
            });
        }
        Ok(())
    }
}
