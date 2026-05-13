use super::ndjson::{write_i64, write_val_json};
use super::ndjson_direct::{NdjsonDirectBytePlan, NdjsonDirectElement};
use crate::builtins::BuiltinMethod;
use crate::ir::physical::PhysicalPathStep;
use crate::util::JsonView;
use crate::JetroEngineError;
use std::io::Write;

#[derive(Clone, Copy)]
pub(super) enum BytePlanWrite {
    Done,
    Fallback,
}

pub(super) fn write_ndjson_byte_plan_row<W: Write>(
    writer: &mut W,
    row: &[u8],
    plan: &NdjsonDirectBytePlan,
) -> Result<BytePlanWrite, JetroEngineError> {
    match plan {
        NdjsonDirectBytePlan::RootField(key) => match root_field_raw_value(row, key.as_ref()) {
            RawFieldValue::Found(value) => {
                writer.write_all(value)?;
                Ok(BytePlanWrite::Done)
            }
            RawFieldValue::Missing => {
                writer.write_all(b"null")?;
                Ok(BytePlanWrite::Done)
            }
            RawFieldValue::Fallback => Ok(BytePlanWrite::Fallback),
        },
        NdjsonDirectBytePlan::RootFieldScalarCall { key, call } => {
            match root_field_raw_value(row, key.as_ref()) {
                RawFieldValue::Found(value) => {
                    if write_raw_string_case_call(writer, value, call.method)? {
                        return Ok(BytePlanWrite::Done);
                    }
                    let Some(view) = raw_json_view(value) else {
                        return Ok(BytePlanWrite::Fallback);
                    };
                    if call.method == BuiltinMethod::Len {
                        let Some(len) = raw_json_view_len(view) else {
                            return Ok(BytePlanWrite::Fallback);
                        };
                        write_i64(writer, len)?;
                    } else if let Some(value) = call.try_apply_json_view(view) {
                        write_val_json(writer, &value)?;
                    } else {
                        writer.write_all(value)?;
                    }
                    Ok(BytePlanWrite::Done)
                }
                RawFieldValue::Missing => {
                    writer.write_all(b"null")?;
                    Ok(BytePlanWrite::Done)
                }
                RawFieldValue::Fallback => Ok(BytePlanWrite::Fallback),
            }
        }
        NdjsonDirectBytePlan::RootObjectItems { method } => {
            match write_root_object_items_raw(writer, row, *method)? {
                BytePlanWrite::Done => Ok(BytePlanWrite::Done),
                BytePlanWrite::Fallback => Ok(BytePlanWrite::Fallback),
            }
        }
        NdjsonDirectBytePlan::RootArrayElementPath {
            key,
            element,
            suffix_steps,
        } => match root_field_raw_value(row, key.as_ref()) {
            RawFieldValue::Found(value) => {
                let Some(element) = raw_json_array_element(value, *element) else {
                    writer.write_all(b"null")?;
                    return Ok(BytePlanWrite::Done);
                };
                if suffix_steps.is_empty() {
                    writer.write_all(element)?;
                    return Ok(BytePlanWrite::Done);
                }
                let Some(value) = raw_json_path_value(element, suffix_steps) else {
                    writer.write_all(b"null")?;
                    return Ok(BytePlanWrite::Done);
                };
                writer.write_all(value)?;
                Ok(BytePlanWrite::Done)
            }
            RawFieldValue::Missing => {
                writer.write_all(b"null")?;
                Ok(BytePlanWrite::Done)
            }
            RawFieldValue::Fallback => Ok(BytePlanWrite::Fallback),
        },
    }
}

enum RawFieldValue<'a> {
    Found(&'a [u8]),
    Missing,
    Fallback,
}

fn root_field_raw_value<'a>(row: &'a [u8], key: &str) -> RawFieldValue<'a> {
    let mut pos = skip_json_ws(row, 0);
    if row.get(pos) != Some(&b'{') {
        return RawFieldValue::Fallback;
    }
    pos += 1;
    loop {
        pos = skip_json_ws(row, pos);
        match row.get(pos).copied() {
            Some(b'}') => return RawFieldValue::Missing,
            Some(b'"') => {}
            _ => return RawFieldValue::Fallback,
        }
        let Some((field_key, next)) = parse_simple_json_string(row, pos) else {
            return RawFieldValue::Fallback;
        };
        pos = skip_json_ws(row, next);
        if row.get(pos) != Some(&b':') {
            return RawFieldValue::Fallback;
        }
        pos += 1;
        let value_start = skip_json_ws(row, pos);
        let Some(value_end) = skip_json_value(row, value_start) else {
            return RawFieldValue::Fallback;
        };
        if field_key == key.as_bytes() {
            return RawFieldValue::Found(&row[value_start..value_end]);
        }
        pos = skip_json_ws(row, value_end);
        match row.get(pos).copied() {
            Some(b',') => pos += 1,
            Some(b'}') => return RawFieldValue::Missing,
            _ => return RawFieldValue::Fallback,
        }
    }
}

fn write_root_object_items_raw<W: Write>(
    writer: &mut W,
    row: &[u8],
    method: BuiltinMethod,
) -> Result<BytePlanWrite, JetroEngineError> {
    let mut pos = skip_json_ws(row, 0);
    if row.get(pos) != Some(&b'{') {
        return Ok(BytePlanWrite::Fallback);
    }
    pos += 1;
    writer.write_all(b"[")?;
    let mut wrote = false;
    loop {
        pos = skip_json_ws(row, pos);
        match row.get(pos).copied() {
            Some(b'}') => {
                writer.write_all(b"]")?;
                return Ok(BytePlanWrite::Done);
            }
            Some(b'"') => {}
            _ => return Ok(BytePlanWrite::Fallback),
        }
        let Some((key, next)) = parse_simple_json_string(row, pos) else {
            return Ok(BytePlanWrite::Fallback);
        };
        pos = skip_json_ws(row, next);
        if row.get(pos) != Some(&b':') {
            return Ok(BytePlanWrite::Fallback);
        }
        let value_start = skip_json_ws(row, pos + 1);
        let Some(value_end) = skip_json_value(row, value_start) else {
            return Ok(BytePlanWrite::Fallback);
        };
        if wrote {
            writer.write_all(b",")?;
        }
        match method {
            BuiltinMethod::Keys => write_json_escaped_ascii_slice(writer, key)?,
            BuiltinMethod::Values => writer.write_all(&row[value_start..value_end])?,
            BuiltinMethod::Entries => {
                writer.write_all(b"[")?;
                write_json_escaped_ascii_slice(writer, key)?;
                writer.write_all(b",")?;
                writer.write_all(&row[value_start..value_end])?;
                writer.write_all(b"]")?;
            }
            _ => return Ok(BytePlanWrite::Fallback),
        }
        wrote = true;
        pos = skip_json_ws(row, value_end);
        match row.get(pos).copied() {
            Some(b',') => pos += 1,
            Some(b'}') => {
                writer.write_all(b"]")?;
                return Ok(BytePlanWrite::Done);
            }
            _ => return Ok(BytePlanWrite::Fallback),
        }
    }
}

fn write_json_escaped_ascii_slice<W: Write>(
    writer: &mut W,
    value: &[u8],
) -> Result<(), JetroEngineError> {
    writer.write_all(b"\"")?;
    writer.write_all(value)?;
    writer.write_all(b"\"")?;
    Ok(())
}

fn skip_json_ws(row: &[u8], mut pos: usize) -> usize {
    while matches!(row.get(pos), Some(b' ' | b'\n' | b'\r' | b'\t')) {
        pos += 1;
    }
    pos
}

fn parse_simple_json_string(row: &[u8], start: usize) -> Option<(&[u8], usize)> {
    if row.get(start) != Some(&b'"') {
        return None;
    }
    let mut pos = start + 1;
    while let Some(byte) = row.get(pos).copied() {
        match byte {
            b'"' => return Some((&row[start + 1..pos], pos + 1)),
            b'\\' | 0x00..=0x1f => return None,
            _ => pos += 1,
        }
    }
    None
}

fn skip_json_string(row: &[u8], start: usize) -> Option<usize> {
    if row.get(start) != Some(&b'"') {
        return None;
    }
    let mut pos = start + 1;
    while let Some(byte) = row.get(pos).copied() {
        match byte {
            b'"' => return Some(pos + 1),
            b'\\' => {
                pos += 2;
            }
            0x00..=0x1f => return None,
            _ => pos += 1,
        }
    }
    None
}

fn skip_json_value(row: &[u8], start: usize) -> Option<usize> {
    match row.get(start).copied()? {
        b'"' => skip_json_string(row, start),
        b'{' => skip_json_compound(row, start, b'{', b'}'),
        b'[' => skip_json_compound(row, start, b'[', b']'),
        b'-' | b'0'..=b'9' | b't' | b'f' | b'n' => {
            let mut pos = start + 1;
            while let Some(byte) = row.get(pos).copied() {
                if matches!(byte, b',' | b'}' | b']' | b' ' | b'\n' | b'\r' | b'\t') {
                    break;
                }
                pos += 1;
            }
            Some(pos)
        }
        _ => None,
    }
}

fn raw_json_view(value: &[u8]) -> Option<JsonView<'_>> {
    let start = skip_json_ws(value, 0);
    let end = trim_json_ws_end(value);
    if start >= end {
        return None;
    }
    match value[start] {
        b'n' if &value[start..end] == b"null" => Some(JsonView::Null),
        b't' if &value[start..end] == b"true" => Some(JsonView::Bool(true)),
        b'f' if &value[start..end] == b"false" => Some(JsonView::Bool(false)),
        b'"' => {
            let (s, next) = parse_simple_json_string(value, start)?;
            (skip_json_ws(value, next) == end)
                .then(|| std::str::from_utf8(s).ok())
                .flatten()
                .map(JsonView::Str)
        }
        b'[' => raw_json_array_len(value, start, end).map(JsonView::ArrayLen),
        b'{' => raw_json_object_len(value, start, end).map(JsonView::ObjectLen),
        b'-' | b'0'..=b'9' => raw_json_number_view(&value[start..end]),
        _ => None,
    }
}

fn write_raw_string_case_call<W: Write>(
    writer: &mut W,
    value: &[u8],
    method: BuiltinMethod,
) -> Result<bool, JetroEngineError> {
    if !matches!(method, BuiltinMethod::Upper | BuiltinMethod::Lower) {
        return Ok(false);
    }
    let start = skip_json_ws(value, 0);
    let Some((s, next)) = parse_simple_json_string(value, start) else {
        return Ok(false);
    };
    if skip_json_ws(value, next) != trim_json_ws_end(value) || !s.is_ascii() {
        return Ok(false);
    }
    writer.write_all(b"\"")?;
    match method {
        BuiltinMethod::Upper => {
            for &byte in s {
                writer.write_all(&[byte.to_ascii_uppercase()])?;
            }
        }
        BuiltinMethod::Lower => {
            for &byte in s {
                writer.write_all(&[byte.to_ascii_lowercase()])?;
            }
        }
        _ => unreachable!("case method checked"),
    }
    writer.write_all(b"\"")?;
    Ok(true)
}

fn raw_json_view_len(value: JsonView<'_>) -> Option<i64> {
    match value {
        JsonView::Str(value) => Some(value.chars().count() as i64),
        JsonView::ArrayLen(value) | JsonView::ObjectLen(value) => Some(value as i64),
        _ => None,
    }
}

fn trim_json_ws_end(value: &[u8]) -> usize {
    let mut end = value.len();
    while end > 0 && matches!(value[end - 1], b' ' | b'\n' | b'\r' | b'\t') {
        end -= 1;
    }
    end
}

fn raw_json_number_view(value: &[u8]) -> Option<JsonView<'_>> {
    let s = std::str::from_utf8(value).ok()?;
    if s.as_bytes()
        .iter()
        .any(|byte| matches!(byte, b'.' | b'e' | b'E'))
    {
        return s.parse::<f64>().ok().map(JsonView::Float);
    }
    if let Ok(value) = s.parse::<i64>() {
        return Some(JsonView::Int(value));
    }
    s.parse::<u64>().ok().map(JsonView::UInt)
}

fn raw_json_array_len(value: &[u8], start: usize, end: usize) -> Option<usize> {
    let mut pos = skip_json_ws(value, start + 1);
    if pos < end && value[pos] == b']' {
        return (skip_json_ws(value, pos + 1) == end).then_some(0);
    }
    let mut len = 0usize;
    loop {
        pos = skip_json_ws(value, pos);
        pos = skip_json_value(value, pos)?;
        len += 1;
        pos = skip_json_ws(value, pos);
        match value.get(pos).copied() {
            Some(b',') => pos += 1,
            Some(b']') => return (skip_json_ws(value, pos + 1) == end).then_some(len),
            _ => return None,
        }
    }
}

fn raw_json_object_len(value: &[u8], start: usize, end: usize) -> Option<usize> {
    let mut pos = skip_json_ws(value, start + 1);
    if pos < end && value[pos] == b'}' {
        return (skip_json_ws(value, pos + 1) == end).then_some(0);
    }
    let mut len = 0usize;
    loop {
        let (_, next) = parse_simple_json_string(value, pos)?;
        pos = skip_json_ws(value, next);
        if value.get(pos) != Some(&b':') {
            return None;
        }
        pos = skip_json_value(value, skip_json_ws(value, pos + 1))?;
        len += 1;
        pos = skip_json_ws(value, pos);
        match value.get(pos).copied() {
            Some(b',') => pos = skip_json_ws(value, pos + 1),
            Some(b'}') => return (skip_json_ws(value, pos + 1) == end).then_some(len),
            _ => return None,
        }
    }
}

fn raw_json_array_element(value: &[u8], element: NdjsonDirectElement) -> Option<&[u8]> {
    let start = skip_json_ws(value, 0);
    let end = trim_json_ws_end(value);
    if value.get(start) != Some(&b'[') {
        return None;
    }
    let mut pos = skip_json_ws(value, start + 1);
    if pos < end && value[pos] == b']' {
        return None;
    }
    let wanted = match element {
        NdjsonDirectElement::First => 0usize,
        NdjsonDirectElement::Nth(n) => n,
        NdjsonDirectElement::Last => return raw_json_last_array_element(value, pos),
    };
    let mut idx = 0usize;
    loop {
        let value_start = skip_json_ws(value, pos);
        let value_end = skip_json_value(value, value_start)?;
        if wanted == idx {
            return Some(&value[value_start..value_end]);
        }
        idx += 1;
        pos = skip_json_ws(value, value_end);
        match value.get(pos).copied() {
            Some(b',') => pos += 1,
            Some(b']') => return None,
            _ => return None,
        }
    }
}

fn raw_json_last_array_element(value: &[u8], mut pos: usize) -> Option<&[u8]> {
    let first_start = skip_json_ws(value, pos);
    let first_end = skip_json_value(value, first_start)?;
    let mut last_start = first_start;
    let mut last_end = first_end;
    pos = skip_json_ws(value, first_end);
    if value.get(pos) == Some(&b']') {
        return Some(&value[last_start..last_end]);
    }
    loop {
        match value.get(pos).copied() {
            Some(b',') => pos += 1,
            _ => return None,
        }
        let value_start = skip_json_ws(value, pos);
        let value_end = skip_json_value(value, value_start)?;
        last_start = value_start;
        last_end = value_end;
        pos = skip_json_ws(value, value_end);
        match value.get(pos).copied() {
            Some(b']') => return Some(&value[last_start..last_end]),
            Some(b',') => {}
            _ => return None,
        }
    }
}

fn raw_json_path_value<'a>(mut value: &'a [u8], steps: &[PhysicalPathStep]) -> Option<&'a [u8]> {
    for step in steps {
        match step {
            PhysicalPathStep::Field(key) => {
                value = match root_field_raw_value(value, key.as_ref()) {
                    RawFieldValue::Found(value) => value,
                    RawFieldValue::Missing => return None,
                    RawFieldValue::Fallback => return None,
                };
            }
            PhysicalPathStep::Index(index) => {
                let Ok(index) = usize::try_from(*index) else {
                    return None;
                };
                value = raw_json_array_element(value, NdjsonDirectElement::Nth(index))?;
            }
        }
    }
    Some(value)
}

fn skip_json_compound(row: &[u8], start: usize, open: u8, close: u8) -> Option<usize> {
    if row.get(start) != Some(&open) {
        return None;
    }
    let mut pos = start + 1;
    let mut depth = 1usize;
    while let Some(byte) = row.get(pos).copied() {
        match byte {
            b'"' => pos = skip_json_string(row, pos)?,
            b if b == open => {
                depth += 1;
                pos += 1;
            }
            b if b == close => {
                depth -= 1;
                pos += 1;
                if depth == 0 {
                    return Some(pos);
                }
            }
            _ => pos += 1,
        }
    }
    None
}
