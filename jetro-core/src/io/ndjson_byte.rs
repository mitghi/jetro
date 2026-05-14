use super::ndjson::{write_i64, write_val_json};
use super::ndjson_direct::{
    NdjsonDirectByteExpr, NdjsonDirectBytePlan, NdjsonDirectElement, NdjsonDirectItemPredicate,
    NdjsonDirectPredicate, NdjsonDirectProjectionValue, NdjsonDirectStreamMap,
    NdjsonDirectStreamPlan, NdjsonDirectStreamSink, NdjsonDirectTapePlan,
};
use super::ndjson_hint::NdjsonObjectLayoutHint;
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
        NdjsonDirectBytePlan::Expr(expr) => write_ndjson_byte_expr(writer, row, expr),
    }
}

#[inline(always)]
fn write_ndjson_byte_expr<W: Write>(
    writer: &mut W,
    row: &[u8],
    expr: &NdjsonDirectByteExpr,
) -> Result<BytePlanWrite, JetroEngineError> {
    match expr {
        NdjsonDirectByteExpr::Path(steps) => match raw_json_byte_path_value(row, steps) {
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
        NdjsonDirectByteExpr::ScalarCall { value, call } => {
            match raw_json_byte_expr_value(row, value) {
                RawFieldValue::Found(value) => {
                    if write_raw_scalar_call(writer, value, call.method).is_err() {
                        return Ok(BytePlanWrite::Fallback);
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
        NdjsonDirectByteExpr::ObjectItems { path, method } => {
            match raw_json_byte_path_value(row, path) {
                RawFieldValue::Found(value) => write_json_object_items_raw(writer, value, *method),
                RawFieldValue::Missing => {
                    writer.write_all(b"[]")?;
                    Ok(BytePlanWrite::Done)
                }
                RawFieldValue::Fallback => Ok(BytePlanWrite::Fallback),
            }
        }
        NdjsonDirectByteExpr::ArrayElementPath {
            source_steps,
            element,
            suffix_steps,
        } => {
            let demand = Some(BytePathDemand::ArrayElement(*element));
            match raw_json_path_value_demand(row, source_steps, demand) {
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
            }
        }
    }
}

#[inline(always)]
fn raw_json_byte_expr_value<'a>(row: &'a [u8], expr: &NdjsonDirectByteExpr) -> RawFieldValue<'a> {
    match expr {
        NdjsonDirectByteExpr::Path(steps) => raw_json_byte_path_value(row, steps),
        NdjsonDirectByteExpr::ArrayElementPath {
            source_steps,
            element,
            suffix_steps,
        } => {
            let demand = Some(BytePathDemand::ArrayElement(*element));
            let source = match raw_json_path_value_demand(row, source_steps, demand) {
                RawFieldValue::Found(value) => value,
                RawFieldValue::Missing => return RawFieldValue::Missing,
                RawFieldValue::Fallback => return RawFieldValue::Fallback,
            };
            let Some(element) = raw_json_array_element(source, *element) else {
                return RawFieldValue::Missing;
            };
            if suffix_steps.is_empty() {
                RawFieldValue::Found(element)
            } else {
                raw_json_path_value(element, suffix_steps)
                    .map(RawFieldValue::Found)
                    .unwrap_or(RawFieldValue::Missing)
            }
        }
        NdjsonDirectByteExpr::ScalarCall { .. } | NdjsonDirectByteExpr::ObjectItems { .. } => {
            RawFieldValue::Fallback
        }
    }
}

#[inline(always)]
fn raw_json_byte_path_value<'a>(row: &'a [u8], steps: &[PhysicalPathStep]) -> RawFieldValue<'a> {
    if let [PhysicalPathStep::Field(key)] = steps {
        return root_field_raw_value(row, key.as_ref());
    }
    raw_json_path_value_demand(row, steps, None)
}

pub(super) fn eval_ndjson_byte_predicate_row(
    row: &[u8],
    predicate: &NdjsonDirectPredicate,
) -> Result<Option<bool>, JetroEngineError> {
    Ok(eval_raw_predicate(row, predicate))
}

pub(super) fn tape_plan_can_write_byte_row(plan: &NdjsonDirectTapePlan) -> bool {
    let NdjsonDirectTapePlan::Stream(stream) = plan else {
        return byte_projection_plan_supported(plan);
    };
    match &stream.sink {
        NdjsonDirectStreamSink::Count => stream.predicate.is_some(),
        NdjsonDirectStreamSink::Collect(map) => byte_stream_map_supported(map),
        NdjsonDirectStreamSink::Numeric { .. } => false,
    }
}

fn byte_projection_plan_supported(plan: &NdjsonDirectTapePlan) -> bool {
    match plan {
        NdjsonDirectTapePlan::RootPath(steps) => byte_path_supported(steps),
        NdjsonDirectTapePlan::Object(fields) => fields
            .iter()
            .all(|field| byte_projection_value_supported(&field.value)),
        NdjsonDirectTapePlan::Array(items) => items.iter().all(byte_projection_value_supported),
        _ => false,
    }
}

pub(super) fn write_ndjson_byte_tape_plan_row<W: Write>(
    writer: &mut W,
    row: &[u8],
    plan: &NdjsonDirectTapePlan,
    scratch: &mut Vec<u8>,
) -> Result<BytePlanWrite, JetroEngineError> {
    match plan {
        NdjsonDirectTapePlan::Stream(stream)
            if matches!(stream.sink, NdjsonDirectStreamSink::Count) =>
        {
            let Some(predicate) = stream.predicate.as_ref() else {
                return Ok(BytePlanWrite::Fallback);
            };
            let Some(count) = raw_json_count_filtered(row, &stream.source_steps, predicate) else {
                return Ok(BytePlanWrite::Fallback);
            };
            write_i64(writer, count as i64)?;
            Ok(BytePlanWrite::Done)
        }
        NdjsonDirectTapePlan::Stream(stream) => {
            let NdjsonDirectStreamSink::Collect(map) = &stream.sink else {
                return Ok(BytePlanWrite::Fallback);
            };
            if !byte_stream_map_supported(map) {
                return Ok(BytePlanWrite::Fallback);
            }
            scratch.clear();
            match write_raw_json_stream_collect(scratch, row, stream, map)? {
                BytePlanWrite::Done => {
                    writer.write_all(scratch)?;
                    Ok(BytePlanWrite::Done)
                }
                BytePlanWrite::Fallback => Ok(BytePlanWrite::Fallback),
            }
        }
        NdjsonDirectTapePlan::RootPath(steps) if byte_path_supported(steps) => {
            write_raw_json_path(writer, row, steps)?;
            Ok(BytePlanWrite::Done)
        }
        NdjsonDirectTapePlan::Object(fields)
            if fields
                .iter()
                .all(|field| byte_projection_value_supported(&field.value)) =>
        {
            write_raw_json_object_projection(writer, row, fields)?;
            Ok(BytePlanWrite::Done)
        }
        NdjsonDirectTapePlan::Array(items) if items.iter().all(byte_projection_value_supported) => {
            write_raw_json_array_projection(writer, row, items)?;
            Ok(BytePlanWrite::Done)
        }
        _ => Ok(BytePlanWrite::Fallback),
    }
}

pub(super) fn write_ndjson_hinted_tape_plan_row<W: Write>(
    writer: &mut W,
    plan: &NdjsonDirectTapePlan,
    root: &NdjsonObjectLayoutHint,
    matched: &super::ndjson_hint::NdjsonRootLayoutMatch<'_, '_>,
) -> Result<BytePlanWrite, JetroEngineError> {
    match plan {
        NdjsonDirectTapePlan::Object(fields) => {
            writer.write_all(b"{")?;
            let mut wrote = false;
            for field in fields {
                if field.optional {
                    let Some(is_null) =
                        hinted_projection_value_is_null_or_missing(root, matched, &field.value)
                    else {
                        return Ok(BytePlanWrite::Fallback);
                    };
                    if is_null {
                        continue;
                    }
                }
                if wrote {
                    writer.write_all(b",")?;
                }
                write_json_escaped_ascii_slice(writer, field.key.as_bytes())?;
                writer.write_all(b":")?;
                if !write_hinted_projection_value(writer, root, matched, &field.value)? {
                    return Ok(BytePlanWrite::Fallback);
                }
                wrote = true;
            }
            writer.write_all(b"}")?;
            Ok(BytePlanWrite::Done)
        }
        NdjsonDirectTapePlan::Array(items) => {
            writer.write_all(b"[")?;
            for (idx, item) in items.iter().enumerate() {
                if idx > 0 {
                    writer.write_all(b",")?;
                }
                if !write_hinted_projection_value(writer, root, matched, item)? {
                    return Ok(BytePlanWrite::Fallback);
                }
            }
            writer.write_all(b"]")?;
            Ok(BytePlanWrite::Done)
        }
        _ => Ok(BytePlanWrite::Fallback),
    }
}

fn byte_path_supported(steps: &[PhysicalPathStep]) -> bool {
    steps.iter().all(|step| {
        matches!(
            step,
            PhysicalPathStep::Field(_) | PhysicalPathStep::Index(_)
        )
    })
}

fn byte_stream_map_supported(map: &NdjsonDirectStreamMap) -> bool {
    match map {
        NdjsonDirectStreamMap::Value(value) => byte_projection_value_supported(value),
        NdjsonDirectStreamMap::Array(items) => items.iter().all(byte_projection_value_supported),
        NdjsonDirectStreamMap::Object(fields) => fields
            .iter()
            .all(|field| byte_projection_value_supported(&field.value)),
    }
}

fn byte_projection_value_supported(value: &NdjsonDirectProjectionValue) -> bool {
    match value {
        NdjsonDirectProjectionValue::Path(_) | NdjsonDirectProjectionValue::Literal(_) => true,
        NdjsonDirectProjectionValue::ViewScalarCall { call, .. } => {
            call.method == BuiltinMethod::Len
                || call.method == BuiltinMethod::Upper
                || call.method == BuiltinMethod::Lower
        }
    }
}

fn write_hinted_projection_value<W: Write>(
    writer: &mut W,
    root: &NdjsonObjectLayoutHint,
    matched: &super::ndjson_hint::NdjsonRootLayoutMatch<'_, '_>,
    value: &NdjsonDirectProjectionValue,
) -> Result<bool, JetroEngineError> {
    match value {
        NdjsonDirectProjectionValue::Path(steps) => match hinted_path_value(root, matched, steps) {
            RawFieldValue::Found(value) => writer.write_all(value)?,
            RawFieldValue::Missing => writer.write_all(b"null")?,
            RawFieldValue::Fallback => return Ok(false),
        },
        NdjsonDirectProjectionValue::ViewScalarCall {
            steps,
            call,
            optional,
        } => {
            let value = match hinted_path_value(root, matched, steps) {
                RawFieldValue::Found(value) => value,
                RawFieldValue::Missing => {
                    writer.write_all(b"null")?;
                    return Ok(true);
                }
                RawFieldValue::Fallback => return Ok(false),
            };
            if *optional && is_json_null(value) {
                writer.write_all(b"null")?;
            } else {
                write_raw_scalar_call(writer, value, call.method)?;
            }
        }
        NdjsonDirectProjectionValue::Literal(lit) => match lit {
            crate::data::value::Val::Null => writer.write_all(b"null")?,
            crate::data::value::Val::Bool(true) => writer.write_all(b"true")?,
            crate::data::value::Val::Bool(false) => writer.write_all(b"false")?,
            _ => write_val_json(writer, lit)?,
        },
    }
    Ok(true)
}

fn hinted_projection_value_is_null_or_missing(
    root: &NdjsonObjectLayoutHint,
    matched: &super::ndjson_hint::NdjsonRootLayoutMatch<'_, '_>,
    value: &NdjsonDirectProjectionValue,
) -> Option<bool> {
    match value {
        NdjsonDirectProjectionValue::Path(steps) => {
            match hinted_path_value(root, matched, steps) {
                RawFieldValue::Found(value) => Some(is_json_null(value)),
                RawFieldValue::Missing => Some(true),
                RawFieldValue::Fallback => None,
            }
        }
        NdjsonDirectProjectionValue::ViewScalarCall {
            steps, optional, ..
        } => {
            if !optional {
                return Some(false);
            }
            match hinted_path_value(root, matched, steps) {
                RawFieldValue::Found(value) => Some(is_json_null(value)),
                RawFieldValue::Missing => Some(true),
                RawFieldValue::Fallback => None,
            }
        }
        NdjsonDirectProjectionValue::Literal(value) => {
            Some(matches!(value, crate::data::value::Val::Null))
        }
    }
}

fn hinted_path_value<'a>(
    root: &NdjsonObjectLayoutHint,
    matched: &super::ndjson_hint::NdjsonRootLayoutMatch<'a, '_>,
    steps: &[PhysicalPathStep],
) -> RawFieldValue<'a> {
    let Some((PhysicalPathStep::Field(key), rest)) = steps.split_first() else {
        return RawFieldValue::Fallback;
    };
    let Some(slot) = root.slot_for(key.as_ref()) else {
        return RawFieldValue::Missing;
    };
    let Some(value) = matched.value_at(slot) else {
        return RawFieldValue::Missing;
    };
    if rest.is_empty() {
        RawFieldValue::Found(value)
    } else {
        raw_json_path_value_precise(value, rest)
    }
}

fn is_json_null(value: &[u8]) -> bool {
    let start = skip_json_ws(value, 0);
    let end = trim_json_ws_end(value);
    start < end && &value[start..end] == b"null"
}

enum RawFieldValue<'a> {
    Found(&'a [u8]),
    Missing,
    Fallback,
}

#[derive(Clone, Copy)]
enum BytePathDemand {
    ArrayElement(NdjsonDirectElement),
}

#[inline(always)]
fn raw_json_path_value_demand<'a>(
    row: &'a [u8],
    steps: &[PhysicalPathStep],
    demand: Option<BytePathDemand>,
) -> RawFieldValue<'a> {
    let Some((PhysicalPathStep::Field(key), rest)) = steps.split_first() else {
        return if steps.is_empty() {
            RawFieldValue::Found(row)
        } else {
            RawFieldValue::Fallback
        };
    };
    let value = match demand {
        Some(BytePathDemand::ArrayElement(element)) if rest.is_empty() => {
            root_field_raw_value_for_element(row, key.as_ref(), element)
        }
        _ => root_field_raw_value(row, key.as_ref()),
    };
    let RawFieldValue::Found(mut value) = value else {
        return value;
    };
    if rest.is_empty() {
        return RawFieldValue::Found(value);
    }
    for (idx, step) in rest.iter().enumerate() {
        let is_last = idx + 1 == rest.len();
        value = match step {
            PhysicalPathStep::Field(key) => {
                let found = match demand {
                    Some(BytePathDemand::ArrayElement(element)) if is_last => {
                        root_field_raw_value_for_element(value, key.as_ref(), element)
                    }
                    _ => root_field_raw_value(value, key.as_ref()),
                };
                match found {
                    RawFieldValue::Found(value) => value,
                    RawFieldValue::Missing => return RawFieldValue::Missing,
                    RawFieldValue::Fallback => return RawFieldValue::Fallback,
                }
            }
            PhysicalPathStep::Index(index) => {
                let Ok(index) = usize::try_from(*index) else {
                    return RawFieldValue::Missing;
                };
                let Some(value) = raw_json_array_element(value, NdjsonDirectElement::Nth(index))
                else {
                    return RawFieldValue::Missing;
                };
                value
            }
        };
    }
    RawFieldValue::Found(value)
}

#[inline(always)]
fn root_field_raw_value<'a>(row: &'a [u8], key: &str) -> RawFieldValue<'a> {
    let mut found = None;
    let visited = visit_root_object_fields(row, |field_key, value_start, value_end| {
        if field_key == key.as_bytes() {
            found = Some(&row[value_start..value_end]);
            return false;
        }
        true
    });
    if let Some(value) = found {
        return RawFieldValue::Found(value);
    }
    if visited {
        RawFieldValue::Missing
    } else {
        RawFieldValue::Fallback
    }
}

fn root_field_raw_value_for_element<'a>(
    row: &'a [u8],
    key: &str,
    element: NdjsonDirectElement,
) -> RawFieldValue<'a> {
    if matches!(element, NdjsonDirectElement::Last) {
        return root_field_raw_value(row, key);
    }
    root_field_raw_value_prefix(row, key)
}

fn root_field_raw_value_prefix<'a>(row: &'a [u8], key: &str) -> RawFieldValue<'a> {
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
        let value_start = skip_json_ws(row, pos + 1);
        if field_key == key.as_bytes() {
            return RawFieldValue::Found(&row[value_start..]);
        }
        let Some(value_end) = skip_json_value(row, value_start) else {
            return RawFieldValue::Fallback;
        };
        pos = skip_json_ws(row, value_end);
        match row.get(pos).copied() {
            Some(b',') => pos += 1,
            Some(b'}') => return RawFieldValue::Missing,
            _ => return RawFieldValue::Fallback,
        }
    }
}

fn write_json_object_items_raw<W: Write>(
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

pub(super) fn visit_root_object_fields<F>(row: &[u8], mut visit: F) -> bool
where
    F: FnMut(&[u8], usize, usize) -> bool,
{
    let mut pos = skip_json_ws(row, 0);
    if row.get(pos) != Some(&b'{') {
        return false;
    }
    pos += 1;
    loop {
        pos = skip_json_ws(row, pos);
        match row.get(pos).copied() {
            Some(b'}') => return true,
            Some(b'"') => {}
            _ => return false,
        }
        let Some((field_key, next)) = parse_simple_json_string(row, pos) else {
            return false;
        };
        pos = skip_json_ws(row, next);
        if row.get(pos) != Some(&b':') {
            return false;
        }
        let value_start = skip_json_ws(row, pos + 1);
        let Some(value_end) = skip_json_value(row, value_start) else {
            return false;
        };
        if !visit(field_key, value_start, value_end) {
            return true;
        }
        pos = skip_json_ws(row, value_end);
        match row.get(pos).copied() {
            Some(b',') => pos += 1,
            Some(b'}') => return true,
            _ => return false,
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

fn write_raw_scalar_call<W: Write>(
    writer: &mut W,
    value: &[u8],
    method: BuiltinMethod,
) -> Result<(), JetroEngineError> {
    if write_raw_string_case_call(writer, value, method)? {
        return Ok(());
    }
    let Some(view) = raw_json_view(value) else {
        return Err(JetroEngineError::Eval(crate::EvalError(
            "unsupported raw scalar call".to_string(),
        )));
    };
    if method == BuiltinMethod::Len {
        let Some(len) = raw_json_view_len(view) else {
            return Err(JetroEngineError::Eval(crate::EvalError(
                "unsupported raw len call".to_string(),
            )));
        };
        write_i64(writer, len)?;
    } else {
        let Some(value) =
            crate::builtins::BuiltinCall::new(method, crate::builtins::BuiltinArgs::None)
                .try_apply_json_view(view)
        else {
            return Err(JetroEngineError::Eval(crate::EvalError(
                "unsupported raw scalar call".to_string(),
            )));
        };
        write_val_json(writer, &value)?;
    }
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

fn raw_json_path_value_precise<'a>(
    mut value: &'a [u8],
    steps: &[PhysicalPathStep],
) -> RawFieldValue<'a> {
    for step in steps {
        match step {
            PhysicalPathStep::Field(key) => {
                value = match root_field_raw_value(value, key.as_ref()) {
                    RawFieldValue::Found(value) => value,
                    RawFieldValue::Missing => return RawFieldValue::Missing,
                    RawFieldValue::Fallback => return RawFieldValue::Fallback,
                };
            }
            PhysicalPathStep::Index(index) => {
                let Ok(index) = usize::try_from(*index) else {
                    return RawFieldValue::Missing;
                };
                let Some(next) = raw_json_array_element(value, NdjsonDirectElement::Nth(index))
                else {
                    return RawFieldValue::Missing;
                };
                value = next;
            }
        }
    }
    RawFieldValue::Found(value)
}

fn eval_raw_predicate(row: &[u8], predicate: &NdjsonDirectPredicate) -> Option<bool> {
    use crate::parse::ast::BinOp;

    match predicate {
        NdjsonDirectPredicate::Path(steps) => raw_json_path_view(row, steps).map(json_view_truthy),
        NdjsonDirectPredicate::Literal(value) => Some(crate::util::is_truthy(value)),
        NdjsonDirectPredicate::Not(inner) => eval_raw_predicate(row, inner).map(|value| !value),
        NdjsonDirectPredicate::Binary { lhs, op, rhs } if *op == BinOp::And => {
            let lhs = eval_raw_predicate(row, lhs)?;
            if !lhs {
                return Some(false);
            }
            eval_raw_predicate(row, rhs)
        }
        NdjsonDirectPredicate::Binary { lhs, op, rhs } if *op == BinOp::Or => {
            let lhs = eval_raw_predicate(row, lhs)?;
            if lhs {
                return Some(true);
            }
            eval_raw_predicate(row, rhs)
        }
        NdjsonDirectPredicate::Binary { lhs, op, rhs } => {
            let lhs = eval_raw_predicate_scalar(row, lhs)?;
            let rhs = eval_raw_predicate_scalar(row, rhs)?;
            Some(crate::util::json_cmp_binop(lhs, *op, rhs))
        }
        NdjsonDirectPredicate::ViewScalarCall { steps, call } => {
            let value = raw_json_path_view(row, steps)?;
            call.try_apply_json_view(value)
                .map(|value| crate::util::is_truthy(&value))
        }
        NdjsonDirectPredicate::ArrayElementViewScalarCall {
            source_steps,
            element,
            suffix_steps,
            call,
        } => {
            let source = raw_json_path_value(row, source_steps)?;
            let element = raw_json_array_element(source, *element)?;
            let value = raw_json_path_view(element, suffix_steps)?;
            call.try_apply_json_view(value)
                .map(|value| crate::util::is_truthy(&value))
        }
        NdjsonDirectPredicate::ViewPipeline { .. } => None,
    }
}

fn raw_json_count_filtered(
    row: &[u8],
    source_steps: &[PhysicalPathStep],
    predicate: &NdjsonDirectItemPredicate,
) -> Option<usize> {
    let source = raw_json_path_value(row, source_steps)?;
    let mut count = 0usize;
    raw_json_source_items(source, |item| {
        if eval_raw_item_predicate(item, predicate)? {
            count += 1;
        }
        Some(())
    })?;
    Some(count)
}

fn write_raw_json_stream_collect<W: Write>(
    writer: &mut W,
    row: &[u8],
    stream: &NdjsonDirectStreamPlan,
    map: &NdjsonDirectStreamMap,
) -> Result<BytePlanWrite, JetroEngineError> {
    let source = match raw_json_byte_path_value(row, &stream.source_steps) {
        RawFieldValue::Found(source) => source,
        RawFieldValue::Missing => {
            writer.write_all(b"[]")?;
            return Ok(BytePlanWrite::Done);
        }
        RawFieldValue::Fallback => return Ok(BytePlanWrite::Fallback),
    };

    writer.write_all(b"[")?;
    let mut wrote = false;
    let mut failed = false;
    let visited = raw_json_source_items(source, |item| {
        let Some(matches) = stream.predicate.as_ref().map_or(Some(true), |predicate| {
            eval_raw_item_predicate(item, predicate)
        }) else {
            failed = true;
            return None;
        };
        if !matches {
            return Some(());
        }
        if wrote && writer.write_all(b",").is_err() {
            failed = true;
            return None;
        }
        if write_raw_json_stream_map(writer, item, map).is_err() {
            failed = true;
            return None;
        }
        wrote = true;
        Some(())
    });
    if failed || visited.is_none() {
        return Ok(BytePlanWrite::Fallback);
    }
    writer.write_all(b"]")?;
    Ok(BytePlanWrite::Done)
}

fn write_raw_json_stream_map<W: Write>(
    writer: &mut W,
    item: &[u8],
    map: &NdjsonDirectStreamMap,
) -> Result<(), JetroEngineError> {
    match map {
        NdjsonDirectStreamMap::Value(value) => write_raw_json_projection_value(writer, item, value),
        NdjsonDirectStreamMap::Array(items) => {
            writer.write_all(b"[")?;
            for (idx, value) in items.iter().enumerate() {
                if idx > 0 {
                    writer.write_all(b",")?;
                }
                write_raw_json_projection_value(writer, item, value)?;
            }
            writer.write_all(b"]")?;
            Ok(())
        }
        NdjsonDirectStreamMap::Object(fields) => {
            writer.write_all(b"{")?;
            let mut wrote = false;
            for field in fields {
                if field.optional
                    && raw_json_projection_value_is_null_or_missing(item, &field.value)?
                {
                    continue;
                }
                if wrote {
                    writer.write_all(b",")?;
                }
                write_val_json(writer, &crate::data::value::Val::Str(field.key.clone()))?;
                writer.write_all(b":")?;
                write_raw_json_projection_value(writer, item, &field.value)?;
                wrote = true;
            }
            writer.write_all(b"}")?;
            Ok(())
        }
    }
}

fn write_raw_json_object_projection<W: Write>(
    writer: &mut W,
    row: &[u8],
    fields: &[super::ndjson_direct::NdjsonDirectObjectField],
) -> Result<(), JetroEngineError> {
    writer.write_all(b"{")?;
    let mut wrote = false;
    for field in fields {
        if field.optional && raw_json_projection_value_is_null_or_missing(row, &field.value)? {
            continue;
        }
        if wrote {
            writer.write_all(b",")?;
        }
        write_val_json(writer, &crate::data::value::Val::Str(field.key.clone()))?;
        writer.write_all(b":")?;
        write_raw_json_projection_value(writer, row, &field.value)?;
        wrote = true;
    }
    writer.write_all(b"}")?;
    Ok(())
}

fn write_raw_json_array_projection<W: Write>(
    writer: &mut W,
    row: &[u8],
    items: &[NdjsonDirectProjectionValue],
) -> Result<(), JetroEngineError> {
    writer.write_all(b"[")?;
    for (idx, value) in items.iter().enumerate() {
        if idx > 0 {
            writer.write_all(b",")?;
        }
        write_raw_json_projection_value(writer, row, value)?;
    }
    writer.write_all(b"]")?;
    Ok(())
}

fn write_raw_json_path<W: Write>(
    writer: &mut W,
    row: &[u8],
    steps: &[PhysicalPathStep],
) -> Result<(), JetroEngineError> {
    match raw_json_path_value_demand(row, steps, None) {
        RawFieldValue::Found(value) => writer.write_all(value)?,
        RawFieldValue::Missing => writer.write_all(b"null")?,
        RawFieldValue::Fallback => {
            return Err(JetroEngineError::Eval(crate::EvalError(
                "unsupported raw path".to_string(),
            )));
        }
    }
    Ok(())
}

fn raw_json_projection_value_is_null_or_missing(
    item: &[u8],
    value: &NdjsonDirectProjectionValue,
) -> Result<bool, JetroEngineError> {
    match value {
        NdjsonDirectProjectionValue::Path(steps) => {
            match raw_json_path_value_demand(item, steps, None) {
                RawFieldValue::Found(value) => Ok(raw_json_is_null(value)),
                RawFieldValue::Missing => Ok(true),
                RawFieldValue::Fallback => Err(JetroEngineError::Eval(crate::EvalError(
                    "unsupported byte stream optional path".to_string(),
                ))),
            }
        }
        NdjsonDirectProjectionValue::ViewScalarCall {
            steps, optional, ..
        } => match raw_json_path_value_demand(item, steps, None) {
            RawFieldValue::Found(value) if *optional && raw_json_is_null(value) => Ok(true),
            RawFieldValue::Found(_) => Ok(false),
            RawFieldValue::Missing => Ok(true),
            RawFieldValue::Fallback => Err(JetroEngineError::Eval(crate::EvalError(
                "unsupported byte stream optional scalar".to_string(),
            ))),
        },
        NdjsonDirectProjectionValue::Literal(value) => {
            Ok(matches!(value, crate::data::value::Val::Null))
        }
    }
}

fn write_raw_json_projection_value<W: Write>(
    writer: &mut W,
    item: &[u8],
    value: &NdjsonDirectProjectionValue,
) -> Result<(), JetroEngineError> {
    match value {
        NdjsonDirectProjectionValue::Path(steps) => {
            match raw_json_path_value_demand(item, steps, None) {
                RawFieldValue::Found(value) => writer.write_all(value)?,
                RawFieldValue::Missing => writer.write_all(b"null")?,
                RawFieldValue::Fallback => {
                    return Err(JetroEngineError::Eval(crate::EvalError(
                        "unsupported byte stream path".to_string(),
                    )));
                }
            }
        }
        NdjsonDirectProjectionValue::ViewScalarCall { steps, call, .. } => {
            match raw_json_path_value_demand(item, steps, None) {
                RawFieldValue::Found(value) => write_raw_scalar_call(writer, value, call.method)?,
                RawFieldValue::Missing => writer.write_all(b"null")?,
                RawFieldValue::Fallback => {
                    return Err(JetroEngineError::Eval(crate::EvalError(
                        "unsupported byte stream scalar".to_string(),
                    )));
                }
            }
        }
        NdjsonDirectProjectionValue::Literal(value) => write_val_json(writer, value)?,
    }
    Ok(())
}

fn raw_json_is_null(value: &[u8]) -> bool {
    let start = skip_json_ws(value, 0);
    let end = trim_json_ws_end(value);
    value.get(start..end) == Some(b"null")
}

fn raw_json_source_items<F>(value: &[u8], mut visit: F) -> Option<()>
where
    F: FnMut(&[u8]) -> Option<()>,
{
    let start = skip_json_ws(value, 0);
    let end = trim_json_ws_end(value);
    if value.get(start) != Some(&b'[') {
        return visit(&value[start..end]);
    }
    let mut pos = skip_json_ws(value, start + 1);
    if pos < end && value[pos] == b']' {
        return Some(());
    }
    loop {
        let value_start = skip_json_ws(value, pos);
        let value_end = skip_json_value(value, value_start)?;
        visit(&value[value_start..value_end])?;
        pos = skip_json_ws(value, value_end);
        match value.get(pos).copied() {
            Some(b',') => pos += 1,
            Some(b']') => return Some(()),
            _ => return None,
        }
    }
}

fn eval_raw_item_predicate(row: &[u8], predicate: &NdjsonDirectItemPredicate) -> Option<bool> {
    use crate::parse::ast::BinOp;

    match predicate {
        NdjsonDirectItemPredicate::Path(steps) => {
            raw_json_path_view(row, steps).map(json_view_truthy)
        }
        NdjsonDirectItemPredicate::Literal(value) => Some(crate::util::is_truthy(value)),
        NdjsonDirectItemPredicate::Binary { lhs, op, rhs } if *op == BinOp::And => {
            let lhs = eval_raw_item_predicate(row, lhs)?;
            if !lhs {
                return Some(false);
            }
            eval_raw_item_predicate(row, rhs)
        }
        NdjsonDirectItemPredicate::Binary { lhs, op, rhs } if *op == BinOp::Or => {
            let lhs = eval_raw_item_predicate(row, lhs)?;
            if lhs {
                return Some(true);
            }
            eval_raw_item_predicate(row, rhs)
        }
        NdjsonDirectItemPredicate::Binary { lhs, op, rhs } => {
            let lhs = eval_raw_item_predicate_scalar(row, lhs)?;
            let rhs = eval_raw_item_predicate_scalar(row, rhs)?;
            Some(crate::util::json_cmp_binop(lhs, *op, rhs))
        }
        NdjsonDirectItemPredicate::CmpLit { lhs, op, lit } => raw_json_path_view(row, lhs)
            .map(|value| crate::util::json_cmp_binop(value, *op, JsonView::from_val(lit))),
        NdjsonDirectItemPredicate::ViewScalarCall { suffix_steps, call } => {
            let value = raw_json_path_view(row, suffix_steps)?;
            call.try_apply_json_view(value)
                .map(|value| crate::util::is_truthy(&value))
        }
    }
}

fn eval_raw_item_predicate_scalar<'a>(
    row: &'a [u8],
    predicate: &'a NdjsonDirectItemPredicate,
) -> Option<JsonView<'a>> {
    match predicate {
        NdjsonDirectItemPredicate::Path(steps) => raw_json_path_view(row, steps),
        NdjsonDirectItemPredicate::Literal(value) => Some(JsonView::from_val(value)),
        _ => None,
    }
}

fn eval_raw_predicate_scalar<'a>(
    row: &'a [u8],
    predicate: &'a NdjsonDirectPredicate,
) -> Option<JsonView<'a>> {
    match predicate {
        NdjsonDirectPredicate::Path(steps) => raw_json_path_view(row, steps),
        NdjsonDirectPredicate::Literal(value) => Some(JsonView::from_val(value)),
        _ => None,
    }
}

fn raw_json_path_view<'a>(row: &'a [u8], steps: &[PhysicalPathStep]) -> Option<JsonView<'a>> {
    raw_json_path_value(row, steps).and_then(raw_json_view)
}

fn json_view_truthy(value: JsonView<'_>) -> bool {
    match value {
        JsonView::Null => false,
        JsonView::Bool(value) => value,
        JsonView::Int(value) => value != 0,
        JsonView::UInt(value) => value != 0,
        JsonView::Float(value) => value != 0.0 && !value.is_nan(),
        JsonView::Str(value) => !value.is_empty(),
        JsonView::ArrayLen(value) | JsonView::ObjectLen(value) => value > 0,
    }
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
