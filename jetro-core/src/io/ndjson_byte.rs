use super::ndjson::{write_i64, write_json_str, write_val_json};
use super::ndjson_direct::{
    NdjsonDirectByteExpr, NdjsonDirectBytePlan, NdjsonDirectElement, NdjsonDirectItemPredicate,
    NdjsonDirectPredicate, NdjsonDirectProjectionValue, NdjsonDirectStreamMap,
    NdjsonDirectStreamPlan, NdjsonDirectStreamSink, NdjsonDirectTapePlan,
};
use super::ndjson_hint::NdjsonObjectLayoutHint;
use crate::builtins::registry::{
    raw_json_scalar, view_object_items_projection_call, view_scalar_value_projection_call,
    BuiltinId,
};
use crate::builtins::{
    BuiltinArgs, BuiltinCall, BuiltinMethod, BuiltinRawJsonScalar,
    BuiltinViewObjectProjection,
};
use crate::data::value::Val;
use crate::ir::physical::PhysicalPathStep;
use crate::util::JsonView;
use crate::JetroEngineError;
use memchr::memchr2;
use smallvec::SmallVec;
use std::io::Write;

type RootFieldSet<'a> = SmallVec<[&'a str; 4]>;
type RootFieldSpans = SmallVec<[Option<std::ops::Range<usize>>; 4]>;
type RootFieldOrdinals = SmallVec<[usize; 4]>;
type DirectRootArraySlots<'a> = SmallVec<[DirectRootProjection<'a>; 4]>;
type DirectRootObjectSlots<'a> = SmallVec<[(&'a str, DirectRootProjection<'a>); 4]>;

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
                    if write_raw_scalar_call(writer, value, call).is_err() {
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
pub(super) fn raw_json_byte_path_value<'a>(
    row: &'a [u8],
    steps: &[PhysicalPathStep],
) -> RawFieldValue<'a> {
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

pub(super) fn eval_ndjson_byte_predicates_all(
    row: &[u8],
    predicates: &[NdjsonDirectPredicate],
) -> Result<bool, JetroEngineError> {
    for predicate in predicates {
        match eval_ndjson_byte_predicate_row(row, predicate)? {
            Some(true) => {}
            Some(false) | None => return Ok(false),
        }
    }
    Ok(true)
}

pub(super) fn tape_plan_can_write_byte_row(plan: &NdjsonDirectTapePlan) -> bool {
    let NdjsonDirectTapePlan::Stream(stream) = plan else {
        return byte_projection_plan_supported(plan);
    };
    match &stream.sink {
        NdjsonDirectStreamSink::Count => stream.predicate.is_some(),
        NdjsonDirectStreamSink::Collect(map)
        | NdjsonDirectStreamSink::First(map)
        | NdjsonDirectStreamSink::Last(map) => byte_stream_map_supported(map),
        NdjsonDirectStreamSink::Numeric { suffix_steps, .. } => byte_path_supported(suffix_steps),
        NdjsonDirectStreamSink::Extreme {
            key_steps, value, ..
        } => byte_path_supported(key_steps) && byte_projection_value_supported(value),
    }
}

fn byte_projection_plan_supported(plan: &NdjsonDirectTapePlan) -> bool {
    match plan {
        NdjsonDirectTapePlan::RootPath(steps) => byte_path_supported(steps),
        NdjsonDirectTapePlan::ViewScalarCall { steps, call, .. } => {
            byte_path_supported(steps) && byte_scalar_call_supported(call)
        }
        NdjsonDirectTapePlan::ArrayElementPath {
            source_steps,
            suffix_steps,
            ..
        } => byte_path_supported(source_steps) && byte_path_supported(suffix_steps),
        NdjsonDirectTapePlan::ArrayElementViewScalarCall {
            source_steps,
            suffix_steps,
            call,
            ..
        } => {
            byte_path_supported(source_steps)
                && byte_path_supported(suffix_steps)
                && byte_scalar_call_supported(call)
        }
        NdjsonDirectTapePlan::Object(fields) => fields
            .iter()
            .all(|field| byte_projection_value_supported(&field.value)),
        NdjsonDirectTapePlan::Array(items) => items.iter().all(byte_projection_value_supported),
        _ => false,
    }
}

fn byte_scalar_call_supported(call: &BuiltinCall) -> bool {
    view_scalar_value_projection_call(BuiltinId::from_method(call.method), &call.args)
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
        NdjsonDirectTapePlan::Stream(stream)
            if matches!(stream.sink, NdjsonDirectStreamSink::Numeric { .. }) =>
        {
            let NdjsonDirectStreamSink::Numeric { suffix_steps, op } = &stream.sink else {
                unreachable!();
            };
            let Some(value) = reduce_raw_json_numeric_path(
                row,
                &stream.source_steps,
                stream.predicate.as_ref(),
                suffix_steps,
                *op,
            ) else {
                return Ok(BytePlanWrite::Fallback);
            };
            write_val_json(writer, &value)?;
            Ok(BytePlanWrite::Done)
        }
        NdjsonDirectTapePlan::Stream(stream)
            if matches!(stream.sink, NdjsonDirectStreamSink::Extreme { .. }) =>
        {
            let NdjsonDirectStreamSink::Extreme {
                key_steps,
                want_max,
                value,
            } = &stream.sink
            else {
                unreachable!();
            };
            let source = match raw_json_byte_path_value(row, &stream.source_steps) {
                RawFieldValue::Found(source) => source,
                RawFieldValue::Missing => {
                    writer.write_all(b"null")?;
                    return Ok(BytePlanWrite::Done);
                }
                RawFieldValue::Fallback => return Ok(BytePlanWrite::Fallback),
            };
            write_raw_json_stream_extreme_source(writer, source, key_steps, *want_max, value)
        }
        NdjsonDirectTapePlan::Stream(stream) => {
            let map = match &stream.sink {
                NdjsonDirectStreamSink::Collect(map)
                | NdjsonDirectStreamSink::First(map)
                | NdjsonDirectStreamSink::Last(map) => map,
                _ => return Ok(BytePlanWrite::Fallback),
            };
            if !byte_stream_map_supported(map) {
                return Ok(BytePlanWrite::Fallback);
            }
            scratch.clear();
            let written = match &stream.sink {
                NdjsonDirectStreamSink::Collect(_) => {
                    write_raw_json_stream_collect(scratch, row, stream, map)?
                }
                NdjsonDirectStreamSink::First(_) => {
                    write_raw_json_stream_first(scratch, row, stream, map)?
                }
                NdjsonDirectStreamSink::Last(_) => {
                    write_raw_json_stream_last(scratch, row, stream, map)?
                }
                _ => unreachable!(),
            };
            match written {
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
        NdjsonDirectTapePlan::ViewScalarCall { .. }
        | NdjsonDirectTapePlan::ArrayElementPath { .. }
        | NdjsonDirectTapePlan::ArrayElementViewScalarCall { .. }
            if byte_projection_plan_supported(plan) =>
        {
            write_raw_json_tape_plan_value(writer, row, plan)
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
        NdjsonDirectTapePlan::RootPath(_)
        | NdjsonDirectTapePlan::ViewScalarCall { .. }
        | NdjsonDirectTapePlan::ArrayElementPath { .. }
        | NdjsonDirectTapePlan::ArrayElementViewScalarCall { .. }
            if byte_projection_plan_supported(plan) =>
        {
            write_raw_json_tape_plan_value(writer, matched.row(), plan)
        }
        NdjsonDirectTapePlan::Stream(stream) => {
            match hinted_path_value(root, matched, &stream.source_steps) {
                RawFieldValue::Found(source) => match &stream.sink {
                    NdjsonDirectStreamSink::Collect(map) => {
                        if !byte_stream_map_supported(map) {
                            return Ok(BytePlanWrite::Fallback);
                        }
                        write_raw_json_stream_collect_from_source(writer, source, stream, map)
                    }
                    NdjsonDirectStreamSink::First(map) => {
                        if !byte_stream_map_supported(map) {
                            return Ok(BytePlanWrite::Fallback);
                        }
                        write_raw_json_stream_first_from_source(writer, source, stream, map)
                    }
                    NdjsonDirectStreamSink::Last(map) => {
                        if !byte_stream_map_supported(map) {
                            return Ok(BytePlanWrite::Fallback);
                        }
                        write_raw_json_stream_last_from_source(writer, source, stream, map)
                    }
                    NdjsonDirectStreamSink::Count => {
                        let Some(predicate) = stream.predicate.as_ref() else {
                            return Ok(BytePlanWrite::Fallback);
                        };
                        let Some(count) = raw_json_count_filtered_source(source, predicate) else {
                            return Ok(BytePlanWrite::Fallback);
                        };
                        write_i64(writer, count as i64)?;
                        Ok(BytePlanWrite::Done)
                    }
                    NdjsonDirectStreamSink::Numeric { suffix_steps, op } => {
                        let Some(value) = reduce_raw_json_numeric_source(
                            source,
                            stream.predicate.as_ref(),
                            suffix_steps,
                            *op,
                        ) else {
                            return Ok(BytePlanWrite::Fallback);
                        };
                        write_val_json(writer, &value)?;
                        Ok(BytePlanWrite::Done)
                    }
                    NdjsonDirectStreamSink::Extreme {
                        key_steps,
                        want_max,
                        value,
                    } => write_raw_json_stream_extreme_source(
                        writer, source, key_steps, *want_max, value,
                    ),
                },
                RawFieldValue::Missing => {
                    match &stream.sink {
                        NdjsonDirectStreamSink::Collect(_) => writer.write_all(b"[]")?,
                        NdjsonDirectStreamSink::First(_) | NdjsonDirectStreamSink::Last(_) => {
                            writer.write_all(b"null")?
                        }
                        NdjsonDirectStreamSink::Count => writer.write_all(b"0")?,
                        NdjsonDirectStreamSink::Numeric { op, .. } => {
                            let value = crate::exec::pipeline::num_finalise(
                                *op,
                                0,
                                0.0,
                                false,
                                f64::INFINITY,
                                f64::NEG_INFINITY,
                                0,
                            );
                            write_val_json(writer, &value)?;
                        }
                        NdjsonDirectStreamSink::Extreme { .. } => writer.write_all(b"null")?,
                    }
                    Ok(BytePlanWrite::Done)
                }
                RawFieldValue::Fallback => Ok(BytePlanWrite::Fallback),
            }
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
        NdjsonDirectProjectionValue::ViewScalarCall { call, .. } => byte_scalar_call_supported(call),
        NdjsonDirectProjectionValue::Nested(plan) => tape_plan_can_write_byte_row(plan),
    }
}

#[inline(always)]
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
                write_raw_scalar_call(writer, value, call)?;
            }
        }
        NdjsonDirectProjectionValue::Literal(lit) => match lit {
            crate::data::value::Val::Null => writer.write_all(b"null")?,
            crate::data::value::Val::Bool(true) => writer.write_all(b"true")?,
            crate::data::value::Val::Bool(false) => writer.write_all(b"false")?,
            _ => write_val_json(writer, lit)?,
        },
        NdjsonDirectProjectionValue::Nested(plan) => {
            return write_ndjson_hinted_tape_plan_row(writer, plan, root, matched)
                .map(|write| matches!(write, BytePlanWrite::Done));
        }
    }
    Ok(true)
}

#[inline(always)]
fn hinted_projection_value_is_null_or_missing(
    root: &NdjsonObjectLayoutHint,
    matched: &super::ndjson_hint::NdjsonRootLayoutMatch<'_, '_>,
    value: &NdjsonDirectProjectionValue,
) -> Option<bool> {
    match value {
        NdjsonDirectProjectionValue::Path(steps) => match hinted_path_value(root, matched, steps) {
            RawFieldValue::Found(value) => Some(is_json_null(value)),
            RawFieldValue::Missing => Some(true),
            RawFieldValue::Fallback => None,
        },
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
        NdjsonDirectProjectionValue::Nested(_) => Some(false),
    }
}

#[inline(always)]
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

#[inline(always)]
fn is_json_null(value: &[u8]) -> bool {
    let start = skip_json_ws(value, 0);
    let end = trim_json_ws_end(value);
    start < end && &value[start..end] == b"null"
}

pub(super) enum RawFieldValue<'a> {
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
    _element: NdjsonDirectElement,
) -> RawFieldValue<'a> {
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
    let Some(projection) = view_object_items_projection_call(
        BuiltinId::from_method(method),
        &BuiltinArgs::None,
    ) else {
        return Ok(BytePlanWrite::Fallback);
    };
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
        match projection {
            BuiltinViewObjectProjection::Keys => write_json_escaped_ascii_slice(writer, key)?,
            BuiltinViewObjectProjection::Values => writer.write_all(&row[value_start..value_end])?,
            BuiltinViewObjectProjection::Entries => {
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
    call: &BuiltinCall,
) -> Result<(), JetroEngineError> {
    if write_raw_string_case_call(writer, value, call)? {
        return Ok(());
    }
    let Some(view) = raw_json_view(value) else {
        return Err(JetroEngineError::Eval(crate::EvalError(
            "unsupported raw scalar call".to_string(),
        )));
    };
    if raw_json_scalar(BuiltinId::from_method(call.method), &call.args)
        == Some(BuiltinRawJsonScalar::Len)
    {
        let Some(len) = raw_json_view_len(view) else {
            return Err(JetroEngineError::Eval(crate::EvalError(
                "unsupported raw len call".to_string(),
            )));
        };
        write_i64(writer, len)?;
    } else {
        let Some(value) = call.try_apply_json_view(view) else {
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

#[inline(always)]
fn parse_simple_json_string(row: &[u8], start: usize) -> Option<(&[u8], usize)> {
    if row.get(start) != Some(&b'"') {
        return None;
    }
    let body = row.get(start + 1..)?;
    let end = memchr2(b'"', b'\\', body)?;
    if body[end] == b'\\' || has_json_control_byte(&body[..end]) {
        return None;
    }
    Some((&body[..end], start + end + 2))
}

#[inline(always)]
fn skip_json_string(row: &[u8], start: usize) -> Option<usize> {
    if row.get(start) != Some(&b'"') {
        return None;
    }
    let mut pos = start + 1;
    loop {
        let tail = row.get(pos..)?;
        let found = memchr2(b'"', b'\\', tail)?;
        if has_json_control_byte(&tail[..found]) {
            return None;
        }
        match tail[found] {
            b'"' => return Some(pos + found + 1),
            b'\\' => {
                pos += found + 2;
                if pos > row.len() {
                    return None;
                }
            }
            _ => unreachable!(),
        }
    }
}

#[inline(always)]
fn has_json_control_byte(bytes: &[u8]) -> bool {
    bytes.iter().any(|byte| *byte < 0x20)
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

fn raw_json_cmp_values_fast(a: &[u8], b: &[u8]) -> Option<std::cmp::Ordering> {
    let a = trim_json_ws(a);
    let b = trim_json_ws(b);
    if let Some(order) = raw_json_simple_string_bytes(a)
        .zip(raw_json_simple_string_bytes(b))
        .map(|(a, b)| a.cmp(b))
    {
        return Some(order);
    }
    raw_json_number_view(a)
        .zip(raw_json_number_view(b))
        .map(|(a, b)| crate::util::json_cmp_vals(a, b))
}

fn raw_json_value_has_fast_comparison(value: &[u8]) -> bool {
    let value = trim_json_ws(value);
    raw_json_simple_string_bytes(value).is_some() || raw_json_number_view(value).is_some()
}

fn trim_json_ws(value: &[u8]) -> &[u8] {
    let start = skip_json_ws(value, 0);
    let end = trim_json_ws_end(value);
    &value[start..end]
}

fn raw_json_simple_string_bytes(value: &[u8]) -> Option<&[u8]> {
    if value.len() < 2 || value[0] != b'"' || *value.last()? != b'"' {
        return None;
    }
    let body = &value[1..value.len() - 1];
    (!body.iter().any(|byte| *byte == b'\\' || *byte < 0x20)).then_some(body)
}

fn write_raw_string_case_call<W: Write>(
    writer: &mut W,
    value: &[u8],
    call: &BuiltinCall,
) -> Result<bool, JetroEngineError> {
    let method = call.method;
    let Some(op) = raw_json_scalar(BuiltinId::from_method(method), &call.args) else {
        return Ok(false);
    };
    let start = skip_json_ws(value, 0);
    let Some((s, next)) = parse_simple_json_string(value, start) else {
        return Ok(false);
    };
    if skip_json_ws(value, next) != trim_json_ws_end(value) || !s.is_ascii() {
        return Ok(false);
    }
    writer.write_all(b"\"")?;
    match op {
        BuiltinRawJsonScalar::AsciiUpper => {
            for &byte in s {
                writer.write_all(&[byte.to_ascii_uppercase()])?;
            }
        }
        BuiltinRawJsonScalar::AsciiLower => {
            for &byte in s {
                writer.write_all(&[byte.to_ascii_lowercase()])?;
            }
        }
        BuiltinRawJsonScalar::Len => return Ok(false),
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

fn raw_json_is_empty_array(value: &[u8]) -> bool {
    let start = skip_json_ws(value, 0);
    value.get(start) == Some(&b'[') && value.get(skip_json_ws(value, start + 1)) == Some(&b']')
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
        NdjsonDirectPredicate::ArrayAny {
            source_steps,
            predicate,
        } => {
            let source = raw_json_path_value(row, source_steps)?;
            raw_json_any_filtered_source(source, predicate)
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
    raw_json_count_filtered_source(source, predicate)
}

fn raw_json_count_filtered_source(
    source: &[u8],
    predicate: &NdjsonDirectItemPredicate,
) -> Option<usize> {
    let mut count = 0usize;
    raw_json_visit_matching_source(source, predicate, || {
        count += 1;
        true
    })?;
    Some(count)
}

fn raw_json_any_filtered_source(
    source: &[u8],
    predicate: &NdjsonDirectItemPredicate,
) -> Option<bool> {
    let mut matched = false;
    raw_json_visit_matching_source(source, predicate, || {
        matched = true;
        false
    })?;
    Some(matched)
}

fn raw_json_visit_matching_source<F>(
    source: &[u8],
    predicate: &NdjsonDirectItemPredicate,
    on_match: F,
) -> Option<()>
where
    F: FnMut() -> bool,
{
    let mut root_fields = RootFieldSet::new();
    if collect_stream_predicate_root_fields(predicate, &mut root_fields) {
        return raw_json_visit_matching_source_from_root_fields(
            source,
            predicate,
            &root_fields,
            on_match,
        );
    }
    let start = skip_json_ws(source, 0);
    let end = trim_json_ws_end(source);
    if source.get(start) != Some(&b'[') {
        return None;
    }
    let mut pos = skip_json_ws(source, start + 1);
    if pos < end && source[pos] == b']' {
        return Some(());
    }
    let mut on_match = on_match;
    loop {
        let value_start = skip_json_ws(source, pos);
        let value_end = skip_json_value(source, value_start)?;
        if eval_raw_item_predicate(&source[value_start..value_end], predicate)? && !on_match() {
            return Some(());
        }
        pos = skip_json_ws(source, value_end);
        match source.get(pos).copied() {
            Some(b',') => pos += 1,
            Some(b']') => return Some(()),
            _ => return None,
        }
    }
}

fn raw_json_visit_matching_source_from_root_fields<F>(
    source: &[u8],
    predicate: &NdjsonDirectItemPredicate,
    root_fields: &[&str],
    mut on_match: F,
) -> Option<()>
where
    F: FnMut() -> bool,
{
    let start = skip_json_ws(source, 0);
    let end = trim_json_ws_end(source);
    if source.get(start) != Some(&b'[') {
        return None;
    }
    let mut pos = skip_json_ws(source, start + 1);
    if pos < end && source[pos] == b']' {
        return Some(());
    }
    let ordinals = infer_raw_json_object_field_ordinals_at(source, pos, root_fields);
    let mut spans = RootFieldSpans::new();
    loop {
        pos = skip_json_ws(source, pos);
        spans.clear();
        let next = if let Some(ordinals) = ordinals.as_ref() {
            scan_raw_json_object_field_spans_by_ordinals_at(
                source,
                pos,
                root_fields,
                ordinals,
                &mut spans,
            )
        } else {
            scan_raw_json_object_field_spans_at(source, pos, root_fields, &mut spans)
        }?;
        if eval_raw_item_predicate_from_root_fields(source, root_fields, &spans, predicate)?
            && !on_match()
        {
            return Some(());
        }
        pos = skip_json_ws(source, next);
        match source.get(pos).copied() {
            Some(b',') => pos += 1,
            Some(b']') => return Some(()),
            _ => return None,
        }
    }
}

fn reduce_raw_json_numeric_path(
    row: &[u8],
    source_steps: &[PhysicalPathStep],
    predicate: Option<&NdjsonDirectItemPredicate>,
    suffix_steps: &[PhysicalPathStep],
    op: crate::exec::pipeline::NumOp,
) -> Option<Val> {
    let source = raw_json_path_value(row, source_steps)?;
    reduce_raw_json_numeric_source(source, predicate, suffix_steps, op)
}

fn reduce_raw_json_numeric_source(
    source: &[u8],
    predicate: Option<&NdjsonDirectItemPredicate>,
    suffix_steps: &[PhysicalPathStep],
    op: crate::exec::pipeline::NumOp,
) -> Option<Val> {
    let mut root_fields = RootFieldSet::new();
    let root_projectable = collect_path_root_field(suffix_steps, &mut root_fields)
        && predicate.map_or(true, |predicate| {
            collect_stream_predicate_root_fields(predicate, &mut root_fields)
        });
    if root_projectable {
        return reduce_raw_json_numeric_source_from_root_fields(
            source,
            predicate,
            suffix_steps,
            op,
            &root_fields,
        );
    }
    let mut acc_i = 0i64;
    let mut acc_f = 0.0f64;
    let mut floated = false;
    let mut min_f = f64::INFINITY;
    let mut max_f = f64::NEG_INFINITY;
    let mut n_obs = 0usize;
    raw_json_source_items(source, |item| {
        if !predicate.map_or(Some(true), |predicate| {
            eval_raw_item_predicate(item, predicate)
        })? {
            return Some(());
        }
        if let Some(value) = raw_json_path_view(item, suffix_steps) {
            fold_raw_json_numeric(
                value,
                op,
                &mut acc_i,
                &mut acc_f,
                &mut floated,
                &mut min_f,
                &mut max_f,
                &mut n_obs,
            );
        }
        Some(())
    })?;
    Some(crate::exec::pipeline::num_finalise(
        op, acc_i, acc_f, floated, min_f, max_f, n_obs,
    ))
}

fn reduce_raw_json_numeric_source_from_root_fields(
    source: &[u8],
    predicate: Option<&NdjsonDirectItemPredicate>,
    suffix_steps: &[PhysicalPathStep],
    op: crate::exec::pipeline::NumOp,
    root_fields: &[&str],
) -> Option<Val> {
    let start = skip_json_ws(source, 0);
    let end = trim_json_ws_end(source);
    if source.get(start) != Some(&b'[') {
        return None;
    }
    let mut pos = skip_json_ws(source, start + 1);
    if pos < end && source[pos] == b']' {
        return Some(crate::exec::pipeline::num_finalise(
            op,
            0,
            0.0,
            false,
            f64::INFINITY,
            f64::NEG_INFINITY,
            0,
        ));
    }
    let ordinals = infer_raw_json_object_field_ordinals_at(source, pos, root_fields);
    let mut spans = RootFieldSpans::new();
    let mut acc_i = 0i64;
    let mut acc_f = 0.0f64;
    let mut floated = false;
    let mut min_f = f64::INFINITY;
    let mut max_f = f64::NEG_INFINITY;
    let mut n_obs = 0usize;
    loop {
        pos = skip_json_ws(source, pos);
        spans.clear();
        let next = if let Some(ordinals) = ordinals.as_ref() {
            scan_raw_json_object_field_spans_by_ordinals_at(
                source,
                pos,
                root_fields,
                ordinals,
                &mut spans,
            )
        } else {
            scan_raw_json_object_field_spans_at(source, pos, root_fields, &mut spans)
        }?;
        if predicate.map_or(Some(true), |predicate| {
            eval_raw_item_predicate_from_root_fields(source, root_fields, &spans, predicate)
        })? {
            if let Some(value) =
                raw_json_projection_view_from_root(source, root_fields, &spans, suffix_steps)
            {
                fold_raw_json_numeric(
                    value,
                    op,
                    &mut acc_i,
                    &mut acc_f,
                    &mut floated,
                    &mut min_f,
                    &mut max_f,
                    &mut n_obs,
                );
            }
        }
        pos = skip_json_ws(source, next);
        match source.get(pos).copied() {
            Some(b',') => pos += 1,
            Some(b']') => {
                return Some(crate::exec::pipeline::num_finalise(
                    op, acc_i, acc_f, floated, min_f, max_f, n_obs,
                ));
            }
            _ => return None,
        }
    }
}

fn write_raw_json_stream_extreme_source<W: Write>(
    writer: &mut W,
    source: &[u8],
    key_steps: &[PhysicalPathStep],
    want_max: bool,
    value: &NdjsonDirectProjectionValue,
) -> Result<BytePlanWrite, JetroEngineError> {
    let mut root_fields = RootFieldSet::new();
    if collect_path_root_field(key_steps, &mut root_fields)
        && collect_projection_root_field(value, &mut root_fields)
    {
        if let Some(()) = write_raw_json_stream_extreme_from_root_fields(
            writer,
            source,
            key_steps,
            want_max,
            value,
            &root_fields,
        )? {
            return Ok(BytePlanWrite::Done);
        }
    }
    let mut best_item = Vec::new();
    let mut best_key = Vec::new();
    let mut failed = false;
    let visited = raw_json_source_items(source, |item| {
        let Some(key_value) = raw_json_path_value(item, key_steps) else {
            failed = true;
            return None;
        };
        let replace = if best_item.is_empty() {
            if !raw_json_value_has_fast_comparison(key_value) && raw_json_view(key_value).is_none()
            {
                failed = true;
                return None;
            }
            true
        } else {
            let order = if let Some(order) = raw_json_cmp_values_fast(key_value, &best_key) {
                order
            } else {
                let Some(key_view) = raw_json_view(key_value) else {
                    failed = true;
                    return None;
                };
                let Some(best_view) = raw_json_view(&best_key) else {
                    failed = true;
                    return None;
                };
                crate::util::json_cmp_vals(key_view, best_view)
            };
            (want_max && order.is_gt()) || (!want_max && order.is_lt())
        };
        if replace {
            best_item.clear();
            best_item.extend_from_slice(item);
            best_key.clear();
            best_key.extend_from_slice(key_value);
        }
        Some(())
    });
    if failed || visited.is_none() {
        return Ok(BytePlanWrite::Fallback);
    }
    if best_item.is_empty() {
        writer.write_all(b"null")?;
        return Ok(BytePlanWrite::Done);
    }
    write_raw_json_projection_value(writer, &best_item, value)?;
    Ok(BytePlanWrite::Done)
}

fn write_raw_json_stream_extreme_from_root_fields<W: Write>(
    writer: &mut W,
    source: &[u8],
    key_steps: &[PhysicalPathStep],
    want_max: bool,
    value: &NdjsonDirectProjectionValue,
    root_fields: &[&str],
) -> Result<Option<()>, JetroEngineError> {
    let start = skip_json_ws(source, 0);
    let end = trim_json_ws_end(source);
    if source.get(start) != Some(&b'[') {
        return Ok(None);
    }
    let mut pos = skip_json_ws(source, start + 1);
    if pos < end && source[pos] == b']' {
        writer.write_all(b"null")?;
        return Ok(Some(()));
    }
    let ordinals = infer_raw_json_object_field_ordinals_at(source, pos, root_fields);
    let mut spans = RootFieldSpans::new();
    let mut best_key = Vec::new();
    let mut best_output = Vec::new();
    loop {
        pos = skip_json_ws(source, pos);
        spans.clear();
        let next = if let Some(ordinals) = ordinals.as_ref() {
            scan_raw_json_object_field_spans_by_ordinals_at(
                source,
                pos,
                root_fields,
                ordinals,
                &mut spans,
            )
        } else {
            scan_raw_json_object_field_spans_at(source, pos, root_fields, &mut spans)
        };
        let Some(next) = next else {
            return Ok(None);
        };
        let key_value =
            match raw_json_projection_value_from_root(source, root_fields, &spans, key_steps) {
                RawFieldValue::Found(value) => value,
                RawFieldValue::Missing | RawFieldValue::Fallback => return Ok(None),
            };
        let replace = if best_output.is_empty() {
            if !raw_json_value_has_fast_comparison(key_value) && raw_json_view(key_value).is_none()
            {
                return Ok(None);
            }
            true
        } else {
            let order = if let Some(order) = raw_json_cmp_values_fast(key_value, &best_key) {
                order
            } else {
                let Some(key_view) = raw_json_view(key_value) else {
                    return Ok(None);
                };
                let Some(best_view) = raw_json_view(&best_key) else {
                    return Ok(None);
                };
                crate::util::json_cmp_vals(key_view, best_view)
            };
            (want_max && order.is_gt()) || (!want_max && order.is_lt())
        };
        if replace {
            best_key.clear();
            best_key.extend_from_slice(key_value);
            best_output.clear();
            if !write_raw_json_projection_value_from_root_fields(
                &mut best_output,
                source,
                root_fields,
                &spans,
                value,
            )? {
                return Ok(None);
            }
        }
        pos = skip_json_ws(source, next);
        match source.get(pos).copied() {
            Some(b',') => pos += 1,
            Some(b']') => {
                if best_output.is_empty() {
                    writer.write_all(b"null")?;
                } else {
                    writer.write_all(&best_output)?;
                }
                return Ok(Some(()));
            }
            _ => return Ok(None),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn fold_raw_json_numeric(
    value: JsonView<'_>,
    op: crate::exec::pipeline::NumOp,
    acc_i: &mut i64,
    acc_f: &mut f64,
    floated: &mut bool,
    min_f: &mut f64,
    max_f: &mut f64,
    n_obs: &mut usize,
) {
    let (as_f, as_i) = match value {
        JsonView::Int(value) => (value as f64, Some(value)),
        JsonView::UInt(value) => (value as f64, i64::try_from(value).ok()),
        JsonView::Float(value) => (value, None),
        _ => return,
    };
    *n_obs += 1;
    match op {
        crate::exec::pipeline::NumOp::Sum | crate::exec::pipeline::NumOp::Avg => {
            if let Some(value) = as_i.filter(|_| !*floated) {
                *acc_i += value;
            } else {
                if !*floated {
                    *acc_f = *acc_i as f64;
                    *floated = true;
                }
                *acc_f += as_f;
            }
        }
        crate::exec::pipeline::NumOp::Min => {
            if as_f < *min_f {
                *min_f = as_f;
            }
        }
        crate::exec::pipeline::NumOp::Max => {
            if as_f > *max_f {
                *max_f = as_f;
            }
        }
    }
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

    write_raw_json_stream_collect_from_source(writer, source, stream, map)
}

fn write_raw_json_stream_first<W: Write>(
    writer: &mut W,
    row: &[u8],
    stream: &NdjsonDirectStreamPlan,
    map: &NdjsonDirectStreamMap,
) -> Result<BytePlanWrite, JetroEngineError> {
    let source_demand = stream
        .predicate
        .is_none()
        .then_some(BytePathDemand::ArrayElement(NdjsonDirectElement::First));
    let source = match raw_json_path_value_demand(row, &stream.source_steps, source_demand) {
        RawFieldValue::Found(source) => source,
        RawFieldValue::Missing => {
            writer.write_all(b"null")?;
            return Ok(BytePlanWrite::Done);
        }
        RawFieldValue::Fallback => return Ok(BytePlanWrite::Fallback),
    };

    write_raw_json_stream_first_from_source(writer, source, stream, map)
}

fn write_raw_json_stream_last<W: Write>(
    writer: &mut W,
    row: &[u8],
    stream: &NdjsonDirectStreamPlan,
    map: &NdjsonDirectStreamMap,
) -> Result<BytePlanWrite, JetroEngineError> {
    let source_demand = stream
        .predicate
        .is_none()
        .then_some(BytePathDemand::ArrayElement(NdjsonDirectElement::Last));
    let source = match raw_json_path_value_demand(row, &stream.source_steps, source_demand) {
        RawFieldValue::Found(source) => source,
        RawFieldValue::Missing => {
            writer.write_all(b"null")?;
            return Ok(BytePlanWrite::Done);
        }
        RawFieldValue::Fallback => return Ok(BytePlanWrite::Fallback),
    };

    write_raw_json_stream_last_from_source(writer, source, stream, map)
}

fn write_raw_json_stream_first_from_source<W: Write>(
    writer: &mut W,
    source: &[u8],
    stream: &NdjsonDirectStreamPlan,
    map: &NdjsonDirectStreamMap,
) -> Result<BytePlanWrite, JetroEngineError> {
    let mut root_fields = RootFieldSet::new();
    let root_projectable = collect_stream_map_root_fields(map, &mut root_fields);
    if root_projectable {
        if let Some(predicate) = stream.predicate.as_ref() {
            if collect_stream_predicate_root_fields(predicate, &mut root_fields) {
                if let Some(()) = write_raw_json_stream_first_projected(
                    writer,
                    source,
                    map,
                    Some(predicate),
                    &root_fields,
                )? {
                    return Ok(BytePlanWrite::Done);
                }
            }
        } else if let Some(()) =
            write_raw_json_stream_first_projected(writer, source, map, None, &root_fields)?
        {
            return Ok(BytePlanWrite::Done);
        }
    }

    let mut failed = false;
    let mut matched = false;
    let visited = raw_json_source_items(source, |item| {
        let Some(matches) = stream.predicate.as_ref().map_or(Some(true), |predicate| {
            eval_raw_item_predicate(item, predicate)
        }) else {
            failed = true;
            return None;
        };
        if matches {
            if write_raw_json_stream_map(writer, item, map).is_err() {
                failed = true;
                return None;
            }
            matched = true;
            return None;
        }
        Some(())
    });
    if failed {
        return Ok(BytePlanWrite::Fallback);
    }
    if matched {
        return Ok(BytePlanWrite::Done);
    }
    if visited.is_none() {
        return Ok(BytePlanWrite::Fallback);
    }
    writer.write_all(b"null")?;
    Ok(BytePlanWrite::Done)
}

fn write_raw_json_stream_last_from_source<W: Write>(
    writer: &mut W,
    source: &[u8],
    stream: &NdjsonDirectStreamPlan,
    map: &NdjsonDirectStreamMap,
) -> Result<BytePlanWrite, JetroEngineError> {
    let mut root_fields = RootFieldSet::new();
    let root_projectable = collect_stream_map_root_fields(map, &mut root_fields);
    if stream.predicate.is_none() {
        if let Some(item) = raw_json_array_element(source, NdjsonDirectElement::Last) {
            write_raw_json_stream_map(writer, item, map)?;
            return Ok(BytePlanWrite::Done);
        }
        if raw_json_is_empty_array(source) {
            writer.write_all(b"null")?;
            return Ok(BytePlanWrite::Done);
        }
    } else if root_projectable {
        if let Some(predicate) = stream.predicate.as_ref() {
            if collect_stream_predicate_root_fields(predicate, &mut root_fields) {
                if let Some(()) = write_raw_json_stream_last_projected(
                    writer,
                    source,
                    map,
                    predicate,
                    &root_fields,
                )? {
                    return Ok(BytePlanWrite::Done);
                }
            }
        }
    }

    let mut failed = false;
    let mut selected = Vec::new();
    let visited = raw_json_source_items(source, |item| {
        let Some(matches) = stream.predicate.as_ref().map_or(Some(true), |predicate| {
            eval_raw_item_predicate(item, predicate)
        }) else {
            failed = true;
            return None;
        };
        if matches {
            selected.clear();
            if write_raw_json_stream_map(&mut selected, item, map).is_err() {
                failed = true;
                return None;
            }
        }
        Some(())
    });
    if failed || visited.is_none() {
        return Ok(BytePlanWrite::Fallback);
    }
    if selected.is_empty() {
        writer.write_all(b"null")?;
    } else {
        writer.write_all(&selected)?;
    }
    Ok(BytePlanWrite::Done)
}

fn write_raw_json_stream_last_projected<W: Write>(
    writer: &mut W,
    source: &[u8],
    map: &NdjsonDirectStreamMap,
    predicate: &NdjsonDirectItemPredicate,
    root_fields: &[&str],
) -> Result<Option<()>, JetroEngineError> {
    let start = skip_json_ws(source, 0);
    let end = trim_json_ws_end(source);
    if source.get(start) != Some(&b'[') {
        return Ok(None);
    }
    let mut pos = skip_json_ws(source, start + 1);
    if pos < end && source[pos] == b']' {
        writer.write_all(b"null")?;
        return Ok(Some(()));
    }
    let ordinals = infer_raw_json_object_field_ordinals_at(source, pos, root_fields);
    let direct_map = direct_root_stream_map(map, root_fields);
    let mut spans = RootFieldSpans::new();
    let mut selected = Vec::new();
    loop {
        pos = skip_json_ws(source, pos);
        spans.clear();
        let next = if let Some(ordinals) = ordinals.as_ref() {
            scan_raw_json_object_field_spans_by_ordinals_at(
                source,
                pos,
                root_fields,
                ordinals,
                &mut spans,
            )
        } else {
            scan_raw_json_object_field_spans_at(source, pos, root_fields, &mut spans)
        };
        let Some(next) = next else {
            return Ok(None);
        };
        let Some(matches) =
            eval_raw_item_predicate_from_root_fields(source, root_fields, &spans, predicate)
        else {
            return Ok(None);
        };
        if matches {
            selected.clear();
            if let Some(direct_map) = direct_map.as_ref() {
                write_direct_root_stream_map(&mut selected, source, &spans, direct_map)?;
            } else if !write_raw_json_stream_map_with_root_spans(
                &mut selected,
                source,
                map,
                root_fields,
                &spans,
            )? {
                return Ok(None);
            }
        }
        pos = skip_json_ws(source, next);
        match source.get(pos).copied() {
            Some(b',') => pos += 1,
            Some(b']') => {
                if selected.is_empty() {
                    writer.write_all(b"null")?;
                } else {
                    writer.write_all(&selected)?;
                }
                return Ok(Some(()));
            }
            _ => return Ok(None),
        }
    }
}

fn write_raw_json_stream_collect_from_source<W: Write>(
    writer: &mut W,
    source: &[u8],
    stream: &NdjsonDirectStreamPlan,
    map: &NdjsonDirectStreamMap,
) -> Result<BytePlanWrite, JetroEngineError> {
    let mut root_fields = RootFieldSet::new();
    let root_projectable = collect_stream_map_root_fields(map, &mut root_fields);
    if root_projectable && stream.predicate.is_none() {
        if let Some(()) = write_raw_json_stream_collect_single_field(writer, source, map)? {
            return Ok(BytePlanWrite::Done);
        }
        if let Some(()) =
            write_raw_json_stream_collect_root_projected(writer, source, map, &root_fields)?
        {
            return Ok(BytePlanWrite::Done);
        }
    }
    if root_projectable {
        if let Some(predicate) = stream.predicate.as_ref() {
            if collect_stream_predicate_root_fields(predicate, &mut root_fields) {
                if let Some(()) = write_raw_json_stream_collect_projected_filtered(
                    writer,
                    source,
                    map,
                    predicate,
                    &root_fields,
                )? {
                    return Ok(BytePlanWrite::Done);
                }
            }
        }
    }
    let mut root_spans = RootFieldSpans::new();
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
        let wrote_item = if root_projectable {
            write_raw_json_stream_map_from_root_fields(
                writer,
                item,
                map,
                &root_fields,
                &mut root_spans,
            )
        } else {
            write_raw_json_stream_map(writer, item, map).map(|_| true)
        };
        if !matches!(wrote_item, Ok(true)) {
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

fn write_raw_json_stream_collect_projected_filtered<W: Write>(
    writer: &mut W,
    source: &[u8],
    map: &NdjsonDirectStreamMap,
    predicate: &NdjsonDirectItemPredicate,
    root_fields: &[&str],
) -> Result<Option<()>, JetroEngineError> {
    let start = skip_json_ws(source, 0);
    let end = trim_json_ws_end(source);
    if source.get(start) != Some(&b'[') {
        return Ok(None);
    }
    let mut pos = skip_json_ws(source, start + 1);
    writer.write_all(b"[")?;
    if pos < end && source[pos] == b']' {
        writer.write_all(b"]")?;
        return Ok(Some(()));
    }
    let ordinals = infer_raw_json_object_field_ordinals_at(source, pos, root_fields);
    let direct_map = direct_root_stream_map(map, root_fields);
    let mut spans = RootFieldSpans::new();
    let mut wrote = false;
    loop {
        pos = skip_json_ws(source, pos);
        spans.clear();
        let next = if let Some(ordinals) = ordinals.as_ref() {
            scan_raw_json_object_field_spans_by_ordinals_at(
                source,
                pos,
                root_fields,
                ordinals,
                &mut spans,
            )
        } else {
            scan_raw_json_object_field_spans_at(source, pos, root_fields, &mut spans)
        };
        let Some(next) = next else {
            return Ok(None);
        };
        let Some(matches) =
            eval_raw_item_predicate_from_root_fields(source, root_fields, &spans, predicate)
        else {
            return Ok(None);
        };
        if matches {
            if wrote {
                writer.write_all(b",")?;
            }
            if let Some(direct_map) = direct_map.as_ref() {
                write_direct_root_stream_map(writer, source, &spans, direct_map)?;
            } else if !write_raw_json_stream_map_with_root_spans(
                writer,
                source,
                map,
                root_fields,
                &spans,
            )? {
                return Ok(None);
            }
            wrote = true;
        }
        pos = skip_json_ws(source, next);
        match source.get(pos).copied() {
            Some(b',') => pos += 1,
            Some(b']') => {
                writer.write_all(b"]")?;
                return Ok(Some(()));
            }
            _ => return Ok(None),
        }
    }
}

fn write_raw_json_stream_first_projected<W: Write>(
    writer: &mut W,
    source: &[u8],
    map: &NdjsonDirectStreamMap,
    predicate: Option<&NdjsonDirectItemPredicate>,
    root_fields: &[&str],
) -> Result<Option<()>, JetroEngineError> {
    let start = skip_json_ws(source, 0);
    let end = trim_json_ws_end(source);
    if source.get(start) != Some(&b'[') {
        return Ok(None);
    }
    let mut pos = skip_json_ws(source, start + 1);
    if pos < end && source[pos] == b']' {
        writer.write_all(b"null")?;
        return Ok(Some(()));
    }
    let ordinals = infer_raw_json_object_field_ordinals_at(source, pos, root_fields);
    let direct_map = direct_root_stream_map(map, root_fields);
    let mut spans = RootFieldSpans::new();
    loop {
        pos = skip_json_ws(source, pos);
        spans.clear();
        let next = if let Some(ordinals) = ordinals.as_ref() {
            scan_raw_json_object_field_spans_by_ordinals_at(
                source,
                pos,
                root_fields,
                ordinals,
                &mut spans,
            )
        } else {
            scan_raw_json_object_field_spans_at(source, pos, root_fields, &mut spans)
        };
        let Some(next) = next else {
            return Ok(None);
        };
        let matches = if let Some(predicate) = predicate {
            let Some(matches) =
                eval_raw_item_predicate_from_root_fields(source, root_fields, &spans, predicate)
            else {
                return Ok(None);
            };
            matches
        } else {
            true
        };
        if matches {
            if let Some(direct_map) = direct_map.as_ref() {
                write_direct_root_stream_map(writer, source, &spans, direct_map)?;
            } else if !write_raw_json_stream_map_with_root_spans(
                writer,
                source,
                map,
                root_fields,
                &spans,
            )? {
                return Ok(None);
            }
            return Ok(Some(()));
        }
        pos = skip_json_ws(source, next);
        match source.get(pos).copied() {
            Some(b',') => pos += 1,
            Some(b']') => {
                writer.write_all(b"null")?;
                return Ok(Some(()));
            }
            _ => return Ok(None),
        }
    }
}

fn write_raw_json_stream_collect_single_field<W: Write>(
    writer: &mut W,
    source: &[u8],
    map: &NdjsonDirectStreamMap,
) -> Result<Option<()>, JetroEngineError> {
    let (steps, call) = match map {
        NdjsonDirectStreamMap::Value(NdjsonDirectProjectionValue::Path(steps)) => {
            (steps.as_slice(), None)
        }
        NdjsonDirectStreamMap::Value(NdjsonDirectProjectionValue::ViewScalarCall {
            steps,
            call,
            ..
        }) if byte_scalar_call_supported(call) => (steps.as_slice(), Some(call)),
        _ => return Ok(None),
    };
    let [PhysicalPathStep::Field(field)] = steps else {
        return Ok(None);
    };
    let start = skip_json_ws(source, 0);
    let end = trim_json_ws_end(source);
    if source.get(start) != Some(&b'[') {
        return Ok(None);
    }
    let mut pos = skip_json_ws(source, start + 1);
    writer.write_all(b"[")?;
    if pos < end && source[pos] == b']' {
        writer.write_all(b"]")?;
        return Ok(Some(()));
    }
    let mut wrote = false;
    loop {
        pos = skip_json_ws(source, pos);
        let Some((value, next)) = raw_json_object_field_value_and_end(source, pos, field.as_ref())
        else {
            return Ok(None);
        };
        if wrote {
            writer.write_all(b",")?;
        }
        if let Some(call) = call {
            write_raw_scalar_call(writer, value, call)?;
        } else {
            writer.write_all(value)?;
        }
        wrote = true;
        pos = skip_json_ws(source, next);
        match source.get(pos).copied() {
            Some(b',') => pos += 1,
            Some(b']') => {
                writer.write_all(b"]")?;
                return Ok(Some(()));
            }
            _ => return Ok(None),
        }
    }
}

pub(super) fn constant_stream_single_field(
    plan: &NdjsonDirectTapePlan,
) -> Option<(&[PhysicalPathStep], &str)> {
    let NdjsonDirectTapePlan::Stream(stream) = plan else {
        return None;
    };
    if stream.predicate.is_some() {
        return None;
    }
    let NdjsonDirectStreamSink::Collect(NdjsonDirectStreamMap::Value(
        NdjsonDirectProjectionValue::Path(steps),
    )) = &stream.sink
    else {
        return None;
    };
    let [PhysicalPathStep::Field(field)] = steps.as_slice() else {
        return None;
    };
    Some((&stream.source_steps, field.as_ref()))
}

pub(super) fn collect_constant_stream_single_field<W: Write>(
    writer: &mut W,
    source: &[u8],
    field: &str,
    mut values: Option<&mut Vec<Vec<u8>>>,
    mut ranges: Option<&mut Vec<std::ops::Range<usize>>>,
    mut prefixes: Option<&mut Vec<Vec<u8>>>,
) -> Result<bool, JetroEngineError> {
    let start = skip_json_ws(source, 0);
    let end = trim_json_ws_end(source);
    if source.get(start) != Some(&b'[') {
        return Ok(false);
    }
    let mut pos = skip_json_ws(source, start + 1);
    writer.write_all(b"[")?;
    if pos < end && source[pos] == b']' {
        writer.write_all(b"]")?;
        return Ok(true);
    }
    let mut wrote = false;
    loop {
        pos = skip_json_ws(source, pos);
        let Some((value, range, next)) =
            raw_json_object_field_value_range_and_end(source, pos, field)
        else {
            return Ok(false);
        };
        if wrote {
            writer.write_all(b",")?;
        }
        writer.write_all(value)?;
        if let Some(values) = values.as_deref_mut() {
            values.push(value.to_vec());
        }
        let prefix_end = pos + range.start;
        if let Some(ranges) = ranges.as_deref_mut() {
            ranges.push(range);
        }
        if let Some(prefixes) = prefixes.as_deref_mut() {
            prefixes.push(source[pos..prefix_end].to_vec());
        }
        wrote = true;
        pos = skip_json_ws(source, next);
        match source.get(pos).copied() {
            Some(b',') => pos += 1,
            Some(b']') => {
                writer.write_all(b"]")?;
                return Ok(true);
            }
            _ => return Ok(false),
        }
    }
}

pub(super) fn validate_constant_stream_single_field_fast(
    source: &[u8],
    values: &[Vec<u8>],
    ranges: &[std::ops::Range<usize>],
    prefixes: &[Vec<u8>],
) -> bool {
    if values.len() != ranges.len() || values.len() != prefixes.len() {
        return false;
    }
    let start = skip_json_ws(source, 0);
    let end = trim_json_ws_end(source);
    if source.get(start) != Some(&b'[') {
        return false;
    }
    let mut pos = skip_json_ws(source, start + 1);
    if pos < end && source[pos] == b']' {
        return values.is_empty();
    }
    for (idx, expected) in values.iter().enumerate() {
        pos = skip_json_ws(source, pos);
        if source.get(pos) != Some(&b'{') {
            return false;
        }
        let range = &ranges[idx];
        let Some(value_start) = pos.checked_add(range.start) else {
            return false;
        };
        let Some(value_end) = pos.checked_add(range.end) else {
            return false;
        };
        if source.get(pos..value_start) != Some(prefixes[idx].as_slice()) {
            return false;
        }
        if source.get(value_start..value_end) != Some(expected.as_slice()) {
            return false;
        }
        let Some(close_rel) = memchr::memchr(b'}', &source[pos..]) else {
            return false;
        };
        pos = skip_json_ws(source, pos + close_rel + 1);
        match source.get(pos).copied() {
            Some(b',') if idx + 1 < values.len() => pos += 1,
            Some(b']') if idx + 1 == values.len() => return true,
            _ => return false,
        }
    }
    false
}

pub(super) fn validate_constant_stream_single_field(
    source: &[u8],
    field: &str,
    values: &[Vec<u8>],
) -> bool {
    let start = skip_json_ws(source, 0);
    let end = trim_json_ws_end(source);
    if source.get(start) != Some(&b'[') {
        return false;
    }
    let mut pos = skip_json_ws(source, start + 1);
    if pos < end && source[pos] == b']' {
        return values.is_empty();
    }
    let mut idx = 0usize;
    loop {
        pos = skip_json_ws(source, pos);
        let Some((value, next)) = raw_json_object_field_value_and_end(source, pos, field) else {
            return false;
        };
        if values.get(idx).map(Vec::as_slice) != Some(value) {
            return false;
        }
        idx += 1;
        pos = skip_json_ws(source, next);
        match source.get(pos).copied() {
            Some(b',') => pos += 1,
            Some(b']') => return idx == values.len(),
            _ => return false,
        }
    }
}

fn write_raw_json_stream_collect_root_projected<W: Write>(
    writer: &mut W,
    source: &[u8],
    map: &NdjsonDirectStreamMap,
    root_fields: &[&str],
) -> Result<Option<()>, JetroEngineError> {
    let start = skip_json_ws(source, 0);
    let end = trim_json_ws_end(source);
    if source.get(start) != Some(&b'[') {
        return Ok(None);
    }
    let mut pos = skip_json_ws(source, start + 1);
    writer.write_all(b"[")?;
    if pos < end && source[pos] == b']' {
        writer.write_all(b"]")?;
        return Ok(Some(()));
    }
    let ordinals = infer_raw_json_object_field_ordinals_at(source, pos, root_fields);
    let direct_map = direct_root_stream_map(map, root_fields);
    let mut spans = RootFieldSpans::new();
    let mut wrote = false;
    loop {
        pos = skip_json_ws(source, pos);
        spans.clear();
        let next = if let Some(ordinals) = ordinals.as_ref() {
            scan_raw_json_object_field_spans_by_ordinals_at(
                source,
                pos,
                root_fields,
                ordinals,
                &mut spans,
            )
        } else {
            scan_raw_json_object_field_spans_at(source, pos, root_fields, &mut spans)
        };
        let Some(next) = next else {
            return Ok(None);
        };
        if wrote {
            writer.write_all(b",")?;
        }
        if let Some(direct_map) = direct_map.as_ref() {
            write_direct_root_stream_map(writer, source, &spans, direct_map)?;
        } else if !write_raw_json_stream_map_with_root_spans(
            writer,
            source,
            map,
            root_fields,
            &spans,
        )? {
            return Ok(None);
        }
        wrote = true;
        pos = skip_json_ws(source, next);
        match source.get(pos).copied() {
            Some(b',') => pos += 1,
            Some(b']') => {
                writer.write_all(b"]")?;
                return Ok(Some(()));
            }
            _ => return Ok(None),
        }
    }
}

fn raw_json_object_field_value_and_end<'a>(
    row: &'a [u8],
    pos: usize,
    field: &str,
) -> Option<(&'a [u8], usize)> {
    raw_json_object_field_value_range_and_end(row, pos, field).map(|(value, _, next)| (value, next))
}

fn raw_json_object_field_value_range_and_end<'a>(
    row: &'a [u8],
    mut pos: usize,
    field: &str,
) -> Option<(&'a [u8], std::ops::Range<usize>, usize)> {
    let item_start = pos;
    if row.get(pos) != Some(&b'{') {
        return None;
    }
    pos += 1;
    let mut found = None;
    loop {
        pos = skip_json_ws(row, pos);
        match row.get(pos).copied()? {
            b'}' => return found.map(|(value, range)| (value, range, pos + 1)),
            b'"' => {}
            _ => return None,
        }
        let (field_key, next) = parse_simple_json_string(row, pos)?;
        pos = skip_json_ws(row, next);
        if row.get(pos) != Some(&b':') {
            return None;
        }
        let value_start = skip_json_ws(row, pos + 1);
        let value_end = skip_json_value(row, value_start)?;
        if field_key == field.as_bytes() {
            found = Some((
                &row[value_start..value_end],
                value_start - item_start..value_end - item_start,
            ));
        }
        pos = skip_json_ws(row, value_end);
        match row.get(pos).copied()? {
            b',' => pos += 1,
            b'}' => return found.map(|(value, range)| (value, range, pos + 1)),
            _ => return None,
        }
    }
}

fn infer_raw_json_object_field_ordinals_at(
    row: &[u8],
    mut pos: usize,
    root_fields: &[&str],
) -> Option<RootFieldOrdinals> {
    let mut ordinals = RootFieldOrdinals::new();
    ordinals.resize(root_fields.len(), usize::MAX);
    let mut remaining = root_fields.len();
    if row.get(pos) != Some(&b'{') {
        return None;
    }
    pos += 1;
    let mut ordinal = 0usize;
    loop {
        pos = skip_json_ws(row, pos);
        match row.get(pos).copied()? {
            b'}' => {
                return (remaining == 0).then_some(ordinals);
            }
            b'"' => {}
            _ => return None,
        }
        let (field_key, next) = parse_simple_json_string(row, pos)?;
        for (idx, field) in root_fields.iter().enumerate() {
            if ordinals[idx] == usize::MAX && field_key == field.as_bytes() {
                ordinals[idx] = ordinal;
                remaining -= 1;
                break;
            }
        }
        pos = skip_json_ws(row, next);
        if row.get(pos) != Some(&b':') {
            return None;
        }
        let value_start = skip_json_ws(row, pos + 1);
        pos = skip_json_ws(row, skip_json_value(row, value_start)?);
        match row.get(pos).copied()? {
            b',' => {
                pos += 1;
                ordinal += 1;
            }
            b'}' => return (remaining == 0).then_some(ordinals),
            _ => return None,
        }
    }
}

fn collect_stream_map_root_fields<'a>(
    map: &'a NdjsonDirectStreamMap,
    out: &mut RootFieldSet<'a>,
) -> bool {
    out.clear();
    match map {
        NdjsonDirectStreamMap::Value(value) => collect_projection_root_field(value, out),
        NdjsonDirectStreamMap::Array(items) => items
            .iter()
            .all(|value| collect_projection_root_field(value, out)),
        NdjsonDirectStreamMap::Object(fields) => fields
            .iter()
            .all(|field| collect_projection_root_field(&field.value, out)),
    }
}

fn collect_projection_root_field<'a>(
    value: &'a NdjsonDirectProjectionValue,
    out: &mut RootFieldSet<'a>,
) -> bool {
    match value {
        NdjsonDirectProjectionValue::Path(steps)
        | NdjsonDirectProjectionValue::ViewScalarCall { steps, .. } => {
            collect_path_root_field(steps, out)
        }
        NdjsonDirectProjectionValue::Nested(_) => false,
        NdjsonDirectProjectionValue::Literal(_) => true,
    }
}

fn collect_stream_predicate_root_fields<'a>(
    predicate: &'a NdjsonDirectItemPredicate,
    out: &mut RootFieldSet<'a>,
) -> bool {
    match predicate {
        NdjsonDirectItemPredicate::Path(steps)
        | NdjsonDirectItemPredicate::CmpLit { lhs: steps, .. }
        | NdjsonDirectItemPredicate::ViewScalarCall {
            suffix_steps: steps,
            ..
        } => collect_path_root_field(steps, out),
        NdjsonDirectItemPredicate::Literal(_) => true,
        NdjsonDirectItemPredicate::Binary { lhs, rhs, .. } => {
            collect_stream_predicate_root_fields(lhs, out)
                && collect_stream_predicate_root_fields(rhs, out)
        }
    }
}

fn collect_path_root_field<'a>(steps: &'a [PhysicalPathStep], out: &mut RootFieldSet<'a>) -> bool {
    let Some(PhysicalPathStep::Field(key)) = steps.first() else {
        return false;
    };
    if !out.contains(&key.as_ref()) {
        out.push(key.as_ref());
    }
    true
}

fn write_raw_json_stream_map_from_root_fields<W: Write>(
    writer: &mut W,
    item: &[u8],
    map: &NdjsonDirectStreamMap,
    root_fields: &[&str],
    spans: &mut RootFieldSpans,
) -> Result<bool, JetroEngineError> {
    if !scan_raw_json_root_field_spans(item, root_fields, spans) {
        return Ok(false);
    }
    write_raw_json_stream_map_with_root_spans(writer, item, map, root_fields, spans)
}

fn write_raw_json_stream_map_with_root_spans<W: Write>(
    writer: &mut W,
    item: &[u8],
    map: &NdjsonDirectStreamMap,
    root_fields: &[&str],
    spans: &[Option<std::ops::Range<usize>>],
) -> Result<bool, JetroEngineError> {
    match map {
        NdjsonDirectStreamMap::Value(value) => write_raw_json_projection_value_from_root_fields(
            writer,
            item,
            root_fields,
            spans,
            value,
        ),
        NdjsonDirectStreamMap::Array(items) => {
            writer.write_all(b"[")?;
            for (idx, value) in items.iter().enumerate() {
                if idx > 0 {
                    writer.write_all(b",")?;
                }
                if !write_raw_json_projection_value_from_root_fields(
                    writer,
                    item,
                    root_fields,
                    spans,
                    value,
                )? {
                    return Ok(false);
                }
            }
            writer.write_all(b"]")?;
            Ok(true)
        }
        NdjsonDirectStreamMap::Object(fields) => {
            writer.write_all(b"{")?;
            let mut wrote = false;
            for field in fields {
                if field.optional {
                    let Some(is_null) = raw_json_projection_value_from_root_is_null_or_missing(
                        item,
                        root_fields,
                        spans,
                        &field.value,
                    ) else {
                        return Ok(false);
                    };
                    if is_null {
                        continue;
                    }
                }
                if wrote {
                    writer.write_all(b",")?;
                }
                write_json_str(writer, field.key.as_ref())?;
                writer.write_all(b":")?;
                if !write_raw_json_projection_value_from_root_fields(
                    writer,
                    item,
                    root_fields,
                    spans,
                    &field.value,
                )? {
                    return Ok(false);
                }
                wrote = true;
            }
            writer.write_all(b"}")?;
            Ok(true)
        }
    }
}

enum DirectRootStreamMap<'a> {
    Array(DirectRootArraySlots<'a>),
    Object(DirectRootObjectSlots<'a>),
}

#[derive(Clone, Copy)]
enum DirectRootProjection<'a> {
    Raw {
        slot: usize,
        suffix: &'a [PhysicalPathStep],
    },
    Scalar {
        slot: usize,
        suffix: &'a [PhysicalPathStep],
        call: &'a BuiltinCall,
        optional: bool,
    },
}

fn direct_root_stream_map<'a>(
    map: &'a NdjsonDirectStreamMap,
    root_fields: &[&str],
) -> Option<DirectRootStreamMap<'a>> {
    match map {
        NdjsonDirectStreamMap::Array(items) => {
            let mut slots = DirectRootArraySlots::new();
            for value in items {
                slots.push(direct_root_projection(value, root_fields)?);
            }
            Some(DirectRootStreamMap::Array(slots))
        }
        NdjsonDirectStreamMap::Object(fields) => {
            let mut slots = DirectRootObjectSlots::new();
            for field in fields {
                if field.optional {
                    return None;
                }
                slots.push((
                    field.key.as_ref(),
                    direct_root_projection(&field.value, root_fields)?,
                ));
            }
            Some(DirectRootStreamMap::Object(slots))
        }
        _ => None,
    }
}

fn direct_root_projection<'a>(
    value: &'a NdjsonDirectProjectionValue,
    root_fields: &[&str],
) -> Option<DirectRootProjection<'a>> {
    match value {
        NdjsonDirectProjectionValue::Path(steps) => {
            let (slot, suffix) = direct_root_path_slot(steps, root_fields)?;
            Some(DirectRootProjection::Raw { slot, suffix })
        }
        NdjsonDirectProjectionValue::ViewScalarCall {
            steps,
            call,
            optional,
        } if byte_scalar_call_supported(call) => {
            let (slot, suffix) = direct_root_path_slot(steps, root_fields)?;
            Some(DirectRootProjection::Scalar {
                slot,
                suffix,
                call,
                optional: *optional,
            })
        }
        _ => None,
    }
}

fn direct_root_path_slot<'a>(
    steps: &'a [PhysicalPathStep],
    root_fields: &[&str],
) -> Option<(usize, &'a [PhysicalPathStep])> {
    let Some((PhysicalPathStep::Field(key), suffix)) = steps.split_first() else {
        return None;
    };
    root_fields
        .iter()
        .position(|field| *field == key.as_ref())
        .map(|slot| (slot, suffix))
}

fn write_direct_root_stream_map<W: Write>(
    writer: &mut W,
    item: &[u8],
    spans: &[Option<std::ops::Range<usize>>],
    map: &DirectRootStreamMap<'_>,
) -> Result<(), JetroEngineError> {
    match map {
        DirectRootStreamMap::Array(slots) => {
            writer.write_all(b"[")?;
            for (idx, projection) in slots.iter().enumerate() {
                if idx > 0 {
                    writer.write_all(b",")?;
                }
                write_direct_root_projection_value(writer, item, spans, *projection)?;
            }
            writer.write_all(b"]")?;
        }
        DirectRootStreamMap::Object(slots) => {
            writer.write_all(b"{")?;
            for (idx, (key, projection)) in slots.iter().enumerate() {
                if idx > 0 {
                    writer.write_all(b",")?;
                }
                write_json_str(writer, key)?;
                writer.write_all(b":")?;
                write_direct_root_projection_value(writer, item, spans, *projection)?;
            }
            writer.write_all(b"}")?;
        }
    }
    Ok(())
}

fn write_direct_root_projection_value<W: Write>(
    writer: &mut W,
    item: &[u8],
    spans: &[Option<std::ops::Range<usize>>],
    projection: DirectRootProjection<'_>,
) -> Result<(), JetroEngineError> {
    match projection {
        DirectRootProjection::Raw { slot, suffix } => {
            write_direct_root_slot_value(writer, item, spans, slot, suffix)
        }
        DirectRootProjection::Scalar {
            slot,
            suffix,
            call,
            optional,
        } => {
            let Some(value) = direct_root_slot_value(item, spans, slot, suffix) else {
                writer.write_all(b"null")?;
                return Ok(());
            };
            if optional && is_json_null(value) {
                writer.write_all(b"null")?;
            } else {
                write_raw_scalar_call(writer, value, call)?;
            }
            Ok(())
        }
    }
}

fn write_direct_root_slot_value<W: Write>(
    writer: &mut W,
    item: &[u8],
    spans: &[Option<std::ops::Range<usize>>],
    slot: usize,
    suffix: &[PhysicalPathStep],
) -> Result<(), JetroEngineError> {
    match direct_root_slot_value(item, spans, slot, suffix) {
        Some(value) => writer.write_all(value)?,
        None => writer.write_all(b"null")?,
    }
    Ok(())
}

fn direct_root_slot_value<'a>(
    item: &'a [u8],
    spans: &[Option<std::ops::Range<usize>>],
    slot: usize,
    suffix: &[PhysicalPathStep],
) -> Option<&'a [u8]> {
    let span = spans.get(slot).and_then(Option::as_ref)?;
    let value = &item[span.clone()];
    if suffix.is_empty() {
        Some(value)
    } else {
        match raw_json_path_value_precise(value, suffix) {
            RawFieldValue::Found(value) => Some(value),
            RawFieldValue::Missing | RawFieldValue::Fallback => None,
        }
    }
}

fn scan_raw_json_root_field_spans(
    item: &[u8],
    root_fields: &[&str],
    spans: &mut RootFieldSpans,
) -> bool {
    spans.clear();
    spans.resize(root_fields.len(), None);
    let mut remaining = root_fields.len();
    let visited = visit_root_object_fields(item, |key, value_start, value_end| {
        for (idx, field) in root_fields.iter().enumerate() {
            if spans[idx].is_none() && key == field.as_bytes() {
                spans[idx] = Some(value_start..value_end);
                remaining -= 1;
                break;
            }
        }
        remaining > 0
    });
    visited || remaining == 0
}

fn scan_raw_json_object_field_spans_at(
    row: &[u8],
    mut pos: usize,
    root_fields: &[&str],
    spans: &mut RootFieldSpans,
) -> Option<usize> {
    spans.resize(root_fields.len(), None);
    let mut remaining = root_fields.len();
    if row.get(pos) != Some(&b'{') {
        return None;
    }
    pos += 1;
    loop {
        pos = skip_json_ws(row, pos);
        match row.get(pos).copied()? {
            b'}' => return Some(pos + 1),
            b'"' => {}
            _ => return None,
        }
        let (field_key, next) = parse_simple_json_string(row, pos)?;
        pos = skip_json_ws(row, next);
        if row.get(pos) != Some(&b':') {
            return None;
        }
        let value_start = skip_json_ws(row, pos + 1);
        let value_end = skip_json_value(row, value_start)?;
        if remaining > 0 {
            for (idx, field) in root_fields.iter().enumerate() {
                if spans[idx].is_none() && field_key == field.as_bytes() {
                    spans[idx] = Some(value_start..value_end);
                    remaining -= 1;
                    break;
                }
            }
        }
        pos = skip_json_ws(row, value_end);
        match row.get(pos).copied()? {
            b',' => pos += 1,
            b'}' => return Some(pos + 1),
            _ => return None,
        }
    }
}

fn scan_raw_json_object_field_spans_by_ordinals_at(
    row: &[u8],
    mut pos: usize,
    root_fields: &[&str],
    ordinals: &[usize],
    spans: &mut RootFieldSpans,
) -> Option<usize> {
    spans.resize(root_fields.len(), None);
    if row.get(pos) != Some(&b'{') {
        return None;
    }
    pos += 1;
    let mut ordinal = 0usize;
    loop {
        pos = skip_json_ws(row, pos);
        match row.get(pos).copied()? {
            b'}' => return Some(pos + 1),
            b'"' => {}
            _ => return None,
        }
        let requested = ordinals
            .iter()
            .position(|field_ordinal| *field_ordinal == ordinal);
        let next = if let Some(field_idx) = requested {
            let (field_key, next) = parse_simple_json_string(row, pos)?;
            if field_key != root_fields[field_idx].as_bytes() {
                return None;
            }
            next
        } else {
            skip_json_string(row, pos)?
        };
        pos = skip_json_ws(row, next);
        if row.get(pos) != Some(&b':') {
            return None;
        }
        let value_start = skip_json_ws(row, pos + 1);
        let value_end = skip_json_value(row, value_start)?;
        if let Some(field_idx) = requested {
            spans[field_idx] = Some(value_start..value_end);
        }
        pos = skip_json_ws(row, value_end);
        match row.get(pos).copied()? {
            b',' => {
                pos += 1;
                ordinal += 1;
            }
            b'}' => return Some(pos + 1),
            _ => return None,
        }
    }
}

fn write_raw_json_projection_value_from_root_fields<W: Write>(
    writer: &mut W,
    item: &[u8],
    root_fields: &[&str],
    spans: &[Option<std::ops::Range<usize>>],
    value: &NdjsonDirectProjectionValue,
) -> Result<bool, JetroEngineError> {
    match value {
        NdjsonDirectProjectionValue::Path(steps) => {
            match raw_json_projection_value_from_root(item, root_fields, spans, steps) {
                RawFieldValue::Found(value) => writer.write_all(value)?,
                RawFieldValue::Missing => writer.write_all(b"null")?,
                RawFieldValue::Fallback => return Ok(false),
            }
        }
        NdjsonDirectProjectionValue::ViewScalarCall {
            steps,
            call,
            optional,
        } => {
            let value = match raw_json_projection_value_from_root(item, root_fields, spans, steps) {
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
                write_raw_scalar_call(writer, value, call)?;
            }
        }
        NdjsonDirectProjectionValue::Literal(value) => write_val_json(writer, value)?,
        NdjsonDirectProjectionValue::Nested(_) => return Ok(false),
    }
    Ok(true)
}

fn raw_json_projection_value_from_root_is_null_or_missing(
    item: &[u8],
    root_fields: &[&str],
    spans: &[Option<std::ops::Range<usize>>],
    value: &NdjsonDirectProjectionValue,
) -> Option<bool> {
    match value {
        NdjsonDirectProjectionValue::Path(steps)
        | NdjsonDirectProjectionValue::ViewScalarCall { steps, .. } => {
            match raw_json_projection_value_from_root(item, root_fields, spans, steps) {
                RawFieldValue::Found(value) => Some(is_json_null(value)),
                RawFieldValue::Missing => Some(true),
                RawFieldValue::Fallback => None,
            }
        }
        NdjsonDirectProjectionValue::Literal(value) => {
            Some(matches!(value, crate::data::value::Val::Null))
        }
        NdjsonDirectProjectionValue::Nested(_) => Some(false),
    }
}

fn raw_json_projection_value_from_root<'a>(
    item: &'a [u8],
    root_fields: &[&str],
    spans: &[Option<std::ops::Range<usize>>],
    steps: &[PhysicalPathStep],
) -> RawFieldValue<'a> {
    let Some((PhysicalPathStep::Field(key), rest)) = steps.split_first() else {
        return RawFieldValue::Fallback;
    };
    let Some(idx) = root_fields.iter().position(|field| *field == key.as_ref()) else {
        return RawFieldValue::Fallback;
    };
    let Some(span) = spans.get(idx).and_then(Option::as_ref) else {
        return RawFieldValue::Missing;
    };
    let value = &item[span.clone()];
    if rest.is_empty() {
        RawFieldValue::Found(value)
    } else {
        raw_json_path_value_precise(value, rest)
    }
}

fn eval_raw_item_predicate_from_root_fields<'a>(
    item: &'a [u8],
    root_fields: &[&str],
    spans: &[Option<std::ops::Range<usize>>],
    predicate: &'a NdjsonDirectItemPredicate,
) -> Option<bool> {
    use crate::parse::ast::BinOp;

    match predicate {
        NdjsonDirectItemPredicate::Path(steps) => {
            raw_json_projection_view_from_root(item, root_fields, spans, steps)
                .map(json_view_truthy)
        }
        NdjsonDirectItemPredicate::Literal(value) => Some(crate::util::is_truthy(value)),
        NdjsonDirectItemPredicate::Binary { lhs, op, rhs } if *op == BinOp::And => {
            let lhs = eval_raw_item_predicate_from_root_fields(item, root_fields, spans, lhs)?;
            if !lhs {
                return Some(false);
            }
            eval_raw_item_predicate_from_root_fields(item, root_fields, spans, rhs)
        }
        NdjsonDirectItemPredicate::Binary { lhs, op, rhs } if *op == BinOp::Or => {
            let lhs = eval_raw_item_predicate_from_root_fields(item, root_fields, spans, lhs)?;
            if lhs {
                return Some(true);
            }
            eval_raw_item_predicate_from_root_fields(item, root_fields, spans, rhs)
        }
        NdjsonDirectItemPredicate::Binary { lhs, op, rhs } => {
            let lhs =
                eval_raw_item_predicate_scalar_from_root_fields(item, root_fields, spans, lhs)?;
            let rhs =
                eval_raw_item_predicate_scalar_from_root_fields(item, root_fields, spans, rhs)?;
            Some(crate::util::json_cmp_binop(lhs, *op, rhs))
        }
        NdjsonDirectItemPredicate::CmpLit { lhs, op, lit } => {
            raw_json_projection_view_from_root(item, root_fields, spans, lhs)
                .map(|value| crate::util::json_cmp_binop(value, *op, JsonView::from_val(lit)))
        }
        NdjsonDirectItemPredicate::ViewScalarCall { suffix_steps, call } => {
            let value = raw_json_projection_view_from_root(item, root_fields, spans, suffix_steps)?;
            call.try_apply_json_view(value)
                .map(|value| crate::util::is_truthy(&value))
        }
    }
}

fn eval_raw_item_predicate_scalar_from_root_fields<'a>(
    item: &'a [u8],
    root_fields: &[&str],
    spans: &[Option<std::ops::Range<usize>>],
    predicate: &'a NdjsonDirectItemPredicate,
) -> Option<JsonView<'a>> {
    match predicate {
        NdjsonDirectItemPredicate::Path(steps) => {
            raw_json_projection_view_from_root(item, root_fields, spans, steps)
        }
        NdjsonDirectItemPredicate::Literal(value) => Some(JsonView::from_val(value)),
        _ => None,
    }
}

fn raw_json_projection_view_from_root<'a>(
    item: &'a [u8],
    root_fields: &[&str],
    spans: &[Option<std::ops::Range<usize>>],
    steps: &[PhysicalPathStep],
) -> Option<JsonView<'a>> {
    match raw_json_projection_value_from_root(item, root_fields, spans, steps) {
        RawFieldValue::Found(value) => raw_json_view(value),
        RawFieldValue::Missing | RawFieldValue::Fallback => None,
    }
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
                write_json_str(writer, field.key.as_ref())?;
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
        write_json_str(writer, field.key.as_ref())?;
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

fn write_raw_json_tape_plan_value<W: Write>(
    writer: &mut W,
    row: &[u8],
    plan: &NdjsonDirectTapePlan,
) -> Result<BytePlanWrite, JetroEngineError> {
    match plan {
        NdjsonDirectTapePlan::RootPath(steps) => {
            write_raw_json_path(writer, row, steps)?;
            Ok(BytePlanWrite::Done)
        }
        NdjsonDirectTapePlan::ViewScalarCall { steps, call, .. } => {
            match raw_json_path_value_demand(row, steps, None) {
                RawFieldValue::Found(value) => write_raw_scalar_call(writer, value, call)?,
                RawFieldValue::Missing => writer.write_all(b"null")?,
                RawFieldValue::Fallback => return Ok(BytePlanWrite::Fallback),
            }
            Ok(BytePlanWrite::Done)
        }
        NdjsonDirectTapePlan::ArrayElementPath {
            source_steps,
            element,
            suffix_steps,
        } => {
            write_raw_json_array_element_path(writer, row, source_steps, *element, suffix_steps)?;
            Ok(BytePlanWrite::Done)
        }
        NdjsonDirectTapePlan::ArrayElementViewScalarCall {
            source_steps,
            element,
            suffix_steps,
            call,
        } => {
            match raw_json_array_element_path_value(row, source_steps, *element, suffix_steps) {
                RawFieldValue::Found(value) => write_raw_scalar_call(writer, value, call)?,
                RawFieldValue::Missing => writer.write_all(b"null")?,
                RawFieldValue::Fallback => return Ok(BytePlanWrite::Fallback),
            }
            Ok(BytePlanWrite::Done)
        }
        NdjsonDirectTapePlan::Stream(stream) => match &stream.sink {
            NdjsonDirectStreamSink::Collect(map) if byte_stream_map_supported(map) => {
                write_raw_json_stream_collect(writer, row, stream, map)
            }
            NdjsonDirectStreamSink::First(map) if byte_stream_map_supported(map) => {
                write_raw_json_stream_first(writer, row, stream, map)
            }
            NdjsonDirectStreamSink::Last(map) if byte_stream_map_supported(map) => {
                write_raw_json_stream_last(writer, row, stream, map)
            }
            NdjsonDirectStreamSink::Count => {
                let Some(predicate) = stream.predicate.as_ref() else {
                    return Ok(BytePlanWrite::Fallback);
                };
                let Some(count) = raw_json_count_filtered(row, &stream.source_steps, predicate)
                else {
                    return Ok(BytePlanWrite::Fallback);
                };
                write_i64(writer, count as i64)?;
                Ok(BytePlanWrite::Done)
            }
            NdjsonDirectStreamSink::Numeric { suffix_steps, op } => {
                let Some(value) = reduce_raw_json_numeric_path(
                    row,
                    &stream.source_steps,
                    stream.predicate.as_ref(),
                    suffix_steps,
                    *op,
                ) else {
                    return Ok(BytePlanWrite::Fallback);
                };
                write_val_json(writer, &value)?;
                Ok(BytePlanWrite::Done)
            }
            NdjsonDirectStreamSink::Extreme {
                key_steps,
                want_max,
                value,
            } => {
                let source = match raw_json_byte_path_value(row, &stream.source_steps) {
                    RawFieldValue::Found(source) => source,
                    RawFieldValue::Missing => {
                        writer.write_all(b"null")?;
                        return Ok(BytePlanWrite::Done);
                    }
                    RawFieldValue::Fallback => return Ok(BytePlanWrite::Fallback),
                };
                write_raw_json_stream_extreme_source(writer, source, key_steps, *want_max, value)
            }
            _ => Ok(BytePlanWrite::Fallback),
        },
        NdjsonDirectTapePlan::Object(fields) => {
            write_raw_json_object_projection(writer, row, fields)?;
            Ok(BytePlanWrite::Done)
        }
        NdjsonDirectTapePlan::Array(items) => {
            write_raw_json_array_projection(writer, row, items)?;
            Ok(BytePlanWrite::Done)
        }
        NdjsonDirectTapePlan::ObjectItems { steps, method } => {
            match raw_json_path_value_demand(row, steps, None) {
                RawFieldValue::Found(value) => {
                    if !matches!(
                        write_json_object_items_raw(writer, value, *method)?,
                        BytePlanWrite::Done
                    ) {
                        return Ok(BytePlanWrite::Fallback);
                    }
                }
                RawFieldValue::Missing => writer.write_all(b"[]")?,
                RawFieldValue::Fallback => return Ok(BytePlanWrite::Fallback),
            }
            Ok(BytePlanWrite::Done)
        }
        NdjsonDirectTapePlan::ViewPipeline { .. } => Ok(BytePlanWrite::Fallback),
    }
}

fn write_raw_json_array_element_path<W: Write>(
    writer: &mut W,
    row: &[u8],
    source_steps: &[PhysicalPathStep],
    element: NdjsonDirectElement,
    suffix_steps: &[PhysicalPathStep],
) -> Result<(), JetroEngineError> {
    match raw_json_array_element_path_value(row, source_steps, element, suffix_steps) {
        RawFieldValue::Found(value) => writer.write_all(value)?,
        RawFieldValue::Missing => writer.write_all(b"null")?,
        RawFieldValue::Fallback => {
            return Err(JetroEngineError::Eval(crate::EvalError(
                "unsupported raw array element path".to_string(),
            )));
        }
    }
    Ok(())
}

fn raw_json_array_element_path_value<'a>(
    row: &'a [u8],
    source_steps: &[PhysicalPathStep],
    element: NdjsonDirectElement,
    suffix_steps: &[PhysicalPathStep],
) -> RawFieldValue<'a> {
    let demand = Some(BytePathDemand::ArrayElement(element));
    let source = match raw_json_path_value_demand(row, source_steps, demand) {
        RawFieldValue::Found(value) => value,
        RawFieldValue::Missing => return RawFieldValue::Missing,
        RawFieldValue::Fallback => return RawFieldValue::Fallback,
    };
    let Some(element) = raw_json_array_element(source, element) else {
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
        NdjsonDirectProjectionValue::Nested(plan) => match plan.as_ref() {
            NdjsonDirectTapePlan::RootPath(steps) => {
                match raw_json_path_value_demand(item, steps, None) {
                    RawFieldValue::Found(value) => Ok(raw_json_is_null(value)),
                    RawFieldValue::Missing => Ok(true),
                    RawFieldValue::Fallback => Err(JetroEngineError::Eval(crate::EvalError(
                        "unsupported byte stream optional nested path".to_string(),
                    ))),
                }
            }
            NdjsonDirectTapePlan::ArrayElementPath {
                source_steps,
                element,
                suffix_steps,
            } => {
                match raw_json_array_element_path_value(item, source_steps, *element, suffix_steps)
                {
                    RawFieldValue::Found(value) => Ok(raw_json_is_null(value)),
                    RawFieldValue::Missing => Ok(true),
                    RawFieldValue::Fallback => Err(JetroEngineError::Eval(crate::EvalError(
                        "unsupported byte stream optional nested array element".to_string(),
                    ))),
                }
            }
            _ => Ok(false),
        },
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
                RawFieldValue::Found(value) => write_raw_scalar_call(writer, value, call)?,
                RawFieldValue::Missing => writer.write_all(b"null")?,
                RawFieldValue::Fallback => {
                    return Err(JetroEngineError::Eval(crate::EvalError(
                        "unsupported byte stream scalar".to_string(),
                    )));
                }
            }
        }
        NdjsonDirectProjectionValue::Literal(value) => write_val_json(writer, value)?,
        NdjsonDirectProjectionValue::Nested(plan) => {
            if !matches!(
                write_raw_json_tape_plan_value(writer, item, plan)?,
                BytePlanWrite::Done
            ) {
                return Err(JetroEngineError::Eval(crate::EvalError(
                    "unsupported nested byte stream projection".to_string(),
                )));
            }
        }
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

pub(super) fn raw_json_path_view<'a>(
    row: &'a [u8],
    steps: &[PhysicalPathStep],
) -> Option<JsonView<'a>> {
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
