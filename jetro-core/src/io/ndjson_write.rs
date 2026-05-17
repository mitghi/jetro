use super::ndjson::{
    write_val_json, NdjsonNullOutput, NdjsonOptions, DEFAULT_READER_BUFFER_CAPACITY,
};
use crate::data::value::Val;
use crate::JetroEngineError;
use std::io::{BufWriter, Write};

pub(super) fn write_val_line<W: Write>(
    writer: &mut W,
    value: &Val,
) -> Result<(), JetroEngineError> {
    write_val_json(writer, value)?;
    writer.write_all(b"\n")?;
    Ok(())
}

pub(super) fn write_val_line_with_options<W: Write>(
    writer: &mut W,
    value: &Val,
    options: NdjsonOptions,
) -> Result<bool, JetroEngineError> {
    if value == &Val::Null && options.null_output == NdjsonNullOutput::Skip {
        return Ok(false);
    }
    write_val_line(writer, value)?;
    Ok(true)
}

pub(super) fn write_json_bytes_line_with_options<W: Write>(
    writer: &mut W,
    bytes: &[u8],
    options: NdjsonOptions,
) -> Result<bool, JetroEngineError> {
    if is_json_null_bytes(bytes) && options.null_output == NdjsonNullOutput::Skip {
        return Ok(false);
    }
    writer.write_all(bytes)?;
    writer.write_all(b"\n")?;
    Ok(true)
}

pub(super) fn ndjson_writer_with_options<W: Write>(
    writer: W,
    options: NdjsonOptions,
) -> BufWriter<W> {
    let capacity = options
        .reader_buffer_capacity
        .max(DEFAULT_READER_BUFFER_CAPACITY);
    BufWriter::with_capacity(capacity, writer)
}

fn is_json_null_bytes(bytes: &[u8]) -> bool {
    bytes == b"null"
}
