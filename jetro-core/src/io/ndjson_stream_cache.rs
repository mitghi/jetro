use super::ndjson_byte::{
    collect_constant_stream_single_field, constant_stream_single_field, raw_json_byte_path_value,
    validate_constant_stream_single_field, validate_constant_stream_single_field_fast,
    BytePlanWrite, RawFieldValue,
};
use super::ndjson_direct::NdjsonDirectTapePlan;
use crate::JetroEngineError;
use std::io::Write;

#[derive(Default)]
pub(super) struct NdjsonConstantStreamCache {
    output: Vec<u8>,
    values: Vec<Vec<u8>>,
    ranges: Vec<std::ops::Range<usize>>,
    prefixes: Vec<Vec<u8>>,
    disabled: bool,
    learned: bool,
}

impl NdjsonConstantStreamCache {
    pub(super) fn write_row<W: Write>(
        &mut self,
        writer: &mut W,
        row: &[u8],
        plan: &NdjsonDirectTapePlan,
    ) -> Result<Option<BytePlanWrite>, JetroEngineError> {
        if self.disabled {
            return Ok(None);
        }
        let Some((source_steps, field)) = constant_stream_single_field(plan) else {
            self.disabled = true;
            return Ok(None);
        };
        let source = match raw_json_byte_path_value(row, source_steps) {
            RawFieldValue::Found(source) => source,
            RawFieldValue::Missing => {
                if self.learned && self.values.is_empty() {
                    writer.write_all(&self.output)?;
                    return Ok(Some(BytePlanWrite::Done));
                }
                self.disabled = true;
                return Ok(None);
            }
            RawFieldValue::Fallback => {
                self.disabled = true;
                return Ok(None);
            }
        };
        if !self.learned {
            self.values.clear();
            self.ranges.clear();
            self.prefixes.clear();
            self.output.clear();
            if !collect_constant_stream_single_field(
                &mut self.output,
                source,
                field,
                Some(&mut self.values),
                Some(&mut self.ranges),
                Some(&mut self.prefixes),
            )? {
                self.disabled = true;
                return Ok(None);
            }
            self.learned = true;
            writer.write_all(&self.output)?;
            return Ok(Some(BytePlanWrite::Done));
        }
        if validate_constant_stream_single_field_fast(
            source,
            &self.values,
            &self.ranges,
            &self.prefixes,
        ) || validate_constant_stream_single_field(source, field, &self.values)
        {
            writer.write_all(&self.output)?;
            Ok(Some(BytePlanWrite::Done))
        } else {
            self.disabled = true;
            Ok(None)
        }
    }
}
