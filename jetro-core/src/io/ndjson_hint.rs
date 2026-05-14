use super::ndjson_byte::visit_root_object_fields;
use super::ndjson_direct::{
    NdjsonDirectByteExpr, NdjsonDirectBytePlan, NdjsonDirectProjectionValue, NdjsonDirectStreamMap,
    NdjsonDirectStreamSink, NdjsonDirectTapePlan,
};
use crate::ir::physical::PhysicalPathStep;
use std::sync::Arc;

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(super) enum NdjsonHintPathStep {
    Field(Arc<str>),
    Index(i64),
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(super) struct NdjsonHintPath {
    pub(super) steps: Vec<NdjsonHintPathStep>,
}

impl NdjsonHintPath {
    fn from_physical(steps: &[PhysicalPathStep]) -> Option<Self> {
        let steps = steps
            .iter()
            .map(|step| match step {
                PhysicalPathStep::Field(key) => Some(NdjsonHintPathStep::Field(key.clone())),
                PhysicalPathStep::Index(index) => Some(NdjsonHintPathStep::Index(*index)),
            })
            .collect::<Option<Vec<_>>>()?;
        Some(Self { steps })
    }

    pub(super) fn root_field(&self) -> Option<&str> {
        match self.steps.first()? {
            NdjsonHintPathStep::Field(key) => Some(key.as_ref()),
            NdjsonHintPathStep::Index(_) => None,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(super) struct NdjsonHintAccessPlan {
    pub(super) paths: Vec<NdjsonHintPath>,
}

impl NdjsonHintAccessPlan {
    pub(super) fn from_direct_plans(
        byte: Option<&NdjsonDirectBytePlan>,
        tape: &NdjsonDirectTapePlan,
    ) -> Self {
        let mut out = Self::default();
        if let Some(byte) = byte {
            out.collect_byte_plan(byte);
        }
        out.collect_tape_plan(tape);
        out.dedup();
        out
    }

    fn push_physical(&mut self, steps: &[PhysicalPathStep]) {
        if let Some(path) = NdjsonHintPath::from_physical(steps) {
            if path.root_field().is_some() {
                self.paths.push(path);
            }
        }
    }

    fn collect_byte_plan(&mut self, plan: &NdjsonDirectBytePlan) {
        match plan {
            NdjsonDirectBytePlan::Expr(expr) => self.collect_byte_expr(expr),
        }
    }

    fn collect_byte_expr(&mut self, expr: &NdjsonDirectByteExpr) {
        match expr {
            NdjsonDirectByteExpr::Path(steps) => self.push_physical(steps),
            NdjsonDirectByteExpr::ScalarCall { value, .. } => self.collect_byte_expr(value),
            NdjsonDirectByteExpr::ObjectItems { path, .. } => self.push_physical(path),
            NdjsonDirectByteExpr::ArrayElementPath {
                source_steps,
                suffix_steps,
                ..
            } => {
                self.push_physical(source_steps);
                let mut combined = source_steps.clone();
                combined.extend_from_slice(suffix_steps);
                self.push_physical(&combined);
            }
        }
    }

    fn collect_tape_plan(&mut self, plan: &NdjsonDirectTapePlan) {
        match plan {
            NdjsonDirectTapePlan::RootPath(steps)
            | NdjsonDirectTapePlan::ViewScalarCall { steps, .. }
            | NdjsonDirectTapePlan::ObjectItems { steps, .. } => self.push_physical(steps),
            NdjsonDirectTapePlan::ArrayElementViewScalarCall {
                source_steps,
                suffix_steps,
                ..
            }
            | NdjsonDirectTapePlan::ArrayElementPath {
                source_steps,
                suffix_steps,
                ..
            } => {
                self.push_physical(source_steps);
                let mut combined = source_steps.clone();
                combined.extend_from_slice(suffix_steps);
                self.push_physical(&combined);
            }
            NdjsonDirectTapePlan::Stream(stream) => {
                self.push_physical(&stream.source_steps);
                if let NdjsonDirectStreamSink::Collect(map) = &stream.sink {
                    self.collect_stream_map(map);
                }
            }
            NdjsonDirectTapePlan::Object(fields) => {
                for field in fields {
                    self.collect_projection_value(&field.value);
                }
            }
            NdjsonDirectTapePlan::Array(items) => {
                for item in items {
                    self.collect_projection_value(item);
                }
            }
            NdjsonDirectTapePlan::ViewPipeline { source_steps, .. } => {
                self.push_physical(source_steps)
            }
        }
    }

    fn collect_stream_map(&mut self, map: &NdjsonDirectStreamMap) {
        match map {
            NdjsonDirectStreamMap::Value(value) => self.collect_projection_value(value),
            NdjsonDirectStreamMap::Array(items) => {
                for item in items {
                    self.collect_projection_value(item);
                }
            }
            NdjsonDirectStreamMap::Object(fields) => {
                for field in fields {
                    self.collect_projection_value(&field.value);
                }
            }
        }
    }

    fn collect_projection_value(&mut self, value: &NdjsonDirectProjectionValue) {
        match value {
            NdjsonDirectProjectionValue::Path(steps)
            | NdjsonDirectProjectionValue::ViewScalarCall { steps, .. } => {
                self.push_physical(steps)
            }
            NdjsonDirectProjectionValue::Literal(_) => {}
        }
    }

    fn dedup(&mut self) {
        self.paths.sort();
        self.paths.dedup();
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct NdjsonFieldHint {
    pub(super) key: Arc<str>,
    pub(super) slot: usize,
    pub(super) seen: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct NdjsonObjectLayoutHint {
    pub(super) fields: Vec<NdjsonFieldHint>,
    pub(super) stable_order: bool,
}

impl NdjsonObjectLayoutHint {
    fn from_keys(keys: Vec<Arc<str>>) -> Self {
        Self {
            fields: keys
                .into_iter()
                .enumerate()
                .map(|(slot, key)| NdjsonFieldHint { key, slot, seen: 1 })
                .collect(),
            stable_order: true,
        }
    }

    fn observe_keys(&mut self, keys: &[Arc<str>]) {
        if self.fields.len() != keys.len()
            || self
                .fields
                .iter()
                .zip(keys)
                .any(|(field, key)| field.key.as_ref() != key.as_ref())
        {
            self.stable_order = false;
        }

        for (slot, key) in keys.iter().enumerate() {
            if let Some(field) = self.fields.iter_mut().find(|field| field.key == *key) {
                field.seen += 1;
            } else {
                self.fields.push(NdjsonFieldHint {
                    key: key.clone(),
                    slot,
                    seen: 1,
                });
            }
        }
    }

    pub(super) fn slot_for(&self, key: &str) -> Option<usize> {
        self.fields
            .iter()
            .find(|field| field.key.as_ref() == key)
            .map(|field| field.slot)
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(super) struct NdjsonSchemaHints {
    pub(super) rows_observed: usize,
    pub(super) rows_rejected: usize,
    pub(super) root_object: Option<NdjsonObjectLayoutHint>,
}

impl NdjsonSchemaHints {
    pub(super) fn observe_row(&mut self, row: &[u8]) {
        let Some(keys) = root_simple_keys(row) else {
            self.rows_rejected += 1;
            return;
        };
        self.rows_observed += 1;
        match &mut self.root_object {
            Some(root) => root.observe_keys(&keys),
            None => self.root_object = Some(NdjsonObjectLayoutHint::from_keys(keys)),
        }
    }

    pub(super) fn root_slot_for(&self, key: &str) -> Option<usize> {
        self.root_object.as_ref()?.slot_for(key)
    }
}

fn root_simple_keys(row: &[u8]) -> Option<Vec<Arc<str>>> {
    let mut keys = Vec::new();
    let ok = visit_root_object_fields(row, |key, _, _| {
        let Some(key) = std::str::from_utf8(key).ok() else {
            return false;
        };
        keys.push(Arc::from(key));
        true
    });
    ok.then_some(keys)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::ndjson_direct::direct_writer_plans;

    #[test]
    fn schema_hints_capture_stable_root_field_slots() {
        let mut hints = NdjsonSchemaHints::default();
        hints.observe_row(br#"{"id":1,"name":"a","profile":{"city":"x"}}"#);
        hints.observe_row(br#"{"id":2,"name":"b","profile":{"city":"y"}}"#);

        assert_eq!(hints.rows_observed, 2);
        assert_eq!(hints.rows_rejected, 0);
        assert_eq!(hints.root_slot_for("id"), Some(0));
        assert_eq!(hints.root_slot_for("name"), Some(1));
        assert!(hints.root_object.as_ref().unwrap().stable_order);
    }

    #[test]
    fn schema_hints_mark_unstable_root_order_without_losing_slots() {
        let mut hints = NdjsonSchemaHints::default();
        hints.observe_row(br#"{"id":1,"name":"a"}"#);
        hints.observe_row(br#"{"name":"b","id":2}"#);

        assert_eq!(hints.rows_observed, 2);
        assert_eq!(hints.root_slot_for("id"), Some(0));
        assert_eq!(hints.root_slot_for("name"), Some(1));
        assert!(!hints.root_object.as_ref().unwrap().stable_order);
    }

    #[test]
    fn schema_hints_reject_rows_that_byte_scanner_cannot_validate() {
        let mut hints = NdjsonSchemaHints::default();
        hints.observe_row(br#"{"escaped\nkey":1}"#);

        assert_eq!(hints.rows_observed, 0);
        assert_eq!(hints.rows_rejected, 1);
        assert!(hints.root_object.is_none());
    }

    #[test]
    fn access_plan_collects_static_projection_paths_algorithmically() {
        let engine = crate::JetroEngine::new();
        let (byte, tape) =
            direct_writer_plans(&engine, r#"{id: $.id, city: $.profile.address.city}"#).unwrap();
        let access = NdjsonHintAccessPlan::from_direct_plans(byte.as_ref(), &tape);

        let roots = access
            .paths
            .iter()
            .filter_map(NdjsonHintPath::root_field)
            .collect::<Vec<_>>();
        assert_eq!(roots, vec!["id", "profile"]);
    }

    #[test]
    fn access_plan_collects_stream_source_and_projection_paths() {
        let engine = crate::JetroEngine::new();
        let (byte, tape) = direct_writer_plans(
            &engine,
            r#"$.attributes.filter(@.value.contains("_3")).map({k: @.key, v: @.value})"#,
        )
        .unwrap();
        let access = NdjsonHintAccessPlan::from_direct_plans(byte.as_ref(), &tape);

        assert!(access
            .paths
            .iter()
            .any(|path| path.root_field() == Some("attributes")));
        assert!(access.paths.iter().any(|path| path.steps
            == vec![NdjsonHintPathStep::Field(Arc::from("key"))]));
        assert!(access.paths.iter().any(|path| path.steps
            == vec![NdjsonHintPathStep::Field(Arc::from("value"))]));
    }
}
