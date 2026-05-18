use super::ndjson_byte::visit_root_object_fields;
use super::ndjson_direct::{
    NdjsonDirectByteExpr, NdjsonDirectBytePlan, NdjsonDirectProjectionValue,
    NdjsonDirectTapePlan,
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

    fn collect_projection_value(&mut self, value: &NdjsonDirectProjectionValue) {
        match value {
            NdjsonDirectProjectionValue::Path(steps)
            | NdjsonDirectProjectionValue::ViewScalarCall { steps, .. } => {
                self.push_physical(steps)
            }
            NdjsonDirectProjectionValue::Nested(plan) => self.collect_tape_plan(plan),
            NdjsonDirectProjectionValue::Literal(_) => {}
        }
    }

    fn dedup(&mut self) {
        self.paths.sort();
        self.paths.dedup();
    }

    fn required_root_fields(&self) -> Vec<&str> {
        let mut fields = self
            .paths
            .iter()
            .filter_map(NdjsonHintPath::root_field)
            .collect::<Vec<_>>();
        fields.sort_unstable();
        fields.dedup();
        fields
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
        if self.fields.iter().any(|field| match keys.get(field.slot) {
            Some(key) => field.key.as_ref() != key.as_ref(),
            None => true,
        }) {
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

    #[cfg(test)]
    pub(super) fn match_row<'a, 's>(
        &self,
        row: &'a [u8],
        spans: &'s mut Vec<std::ops::Range<usize>>,
    ) -> Option<NdjsonRootLayoutMatch<'a, 's>> {
        if !self.stable_order {
            return None;
        }
        spans.clear();
        spans.reserve(self.fields.len());
        let mut slot = 0usize;
        let ok = visit_root_object_fields(row, |key, value_start, value_end| {
            let Some(expected) = self.fields.get(slot) else {
                return false;
            };
            if expected.key.as_bytes() != key {
                return false;
            }
            spans.push(value_start..value_end);
            slot += 1;
            true
        });
        (ok && slot == self.fields.len()).then_some(NdjsonRootLayoutMatch {
            row,
            spans: spans.as_slice(),
        })
    }

    pub(super) fn match_required_slots<'a, 's>(
        &self,
        row: &'a [u8],
        required_slots: &[usize],
        spans: &'s mut Vec<std::ops::Range<usize>>,
    ) -> Option<NdjsonRootLayoutMatch<'a, 's>> {
        if !self.stable_order {
            return None;
        }
        spans.clear();
        let Some(max_slot) = required_slots.last().copied() else {
            return Some(NdjsonRootLayoutMatch {
                row,
                spans: spans.as_slice(),
            });
        };
        spans.resize(max_slot + 1, 0..0);
        let mut slot = 0usize;
        let mut required_idx = 0usize;
        let mut captured_all = false;
        let visited_all = visit_root_object_fields(row, |key, value_start, value_end| {
            let Some(expected) = self.fields.get(slot) else {
                return false;
            };
            if expected.key.as_bytes() != key {
                return false;
            }
            if required_slots.get(required_idx) == Some(&slot) {
                spans[slot] = value_start..value_end;
                required_idx += 1;
                if required_idx == required_slots.len() {
                    captured_all = true;
                    return false;
                }
            }
            slot += 1;
            true
        });
        (captured_all || (visited_all && required_idx == required_slots.len())).then_some(
            NdjsonRootLayoutMatch {
                row,
                spans: spans.as_slice(),
            },
        )
    }
}

pub(super) struct NdjsonRootLayoutMatch<'a, 's> {
    row: &'a [u8],
    spans: &'s [std::ops::Range<usize>],
}

impl<'a, 's> NdjsonRootLayoutMatch<'a, 's> {
    pub(super) fn row(&self) -> &'a [u8] {
        self.row
    }

    pub(super) fn value_at(&self, slot: usize) -> Option<&'a [u8]> {
        let span = self.spans.get(slot)?;
        Some(&self.row[span.clone()])
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

    #[cfg(test)]
    pub(super) fn root_slot_for(&self, key: &str) -> Option<usize> {
        self.root_object.as_ref()?.slot_for(key)
    }

    fn root_has_stable_fields<'a>(&self, fields: impl IntoIterator<Item = &'a str>) -> bool {
        let Some(root) = &self.root_object else {
            return false;
        };
        root.stable_order && fields.into_iter().all(|field| root.slot_for(field).is_some())
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct NdjsonHintConfig {
    pub(super) min_rows: usize,
    pub(super) max_rejects: usize,
    pub(super) max_layout_misses: usize,
}

impl Default for NdjsonHintConfig {
    fn default() -> Self {
        Self {
            min_rows: 2,
            max_rejects: 2,
            max_layout_misses: 8,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum NdjsonHintDecision {
    Learning,
    UseHints,
    Disabled,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(super) struct NdjsonHintStats {
    pub(super) learned_rows: usize,
    pub(super) rejected_rows: usize,
    pub(super) hinted_rows: usize,
    pub(super) layout_misses: usize,
    pub(super) disabled: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct NdjsonHintState {
    config: NdjsonHintConfig,
    access: NdjsonHintAccessPlan,
    schema: NdjsonSchemaHints,
    stats: NdjsonHintStats,
    required_root_slots: Vec<usize>,
    span_scratch: Vec<std::ops::Range<usize>>,
    active: bool,
    disabled: bool,
}

impl NdjsonHintState {
    pub(super) fn new(config: NdjsonHintConfig, access: NdjsonHintAccessPlan) -> Self {
        Self {
            config,
            access,
            schema: NdjsonSchemaHints::default(),
            stats: NdjsonHintStats::default(),
            required_root_slots: Vec::new(),
            span_scratch: Vec::new(),
            active: false,
            disabled: false,
        }
    }

    pub(super) fn observe_row(&mut self, row: &[u8]) -> NdjsonHintDecision {
        if self.disabled {
            return NdjsonHintDecision::Disabled;
        }
        if self.active {
            self.stats.hinted_rows += 1;
            return NdjsonHintDecision::UseHints;
        }
        self.schema.observe_row(row);
        self.stats.learned_rows = self.schema.rows_observed;
        self.stats.rejected_rows = self.schema.rows_rejected;
        if self.schema.rows_rejected > self.config.max_rejects {
            self.disabled = true;
            self.active = false;
            self.stats.disabled = true;
            return NdjsonHintDecision::Disabled;
        }
        if self.schema.rows_observed < self.config.min_rows {
            return NdjsonHintDecision::Learning;
        }
        if self
            .schema
            .root_has_stable_fields(self.access.required_root_fields())
        {
            self.refresh_required_root_slots();
            self.active = true;
            self.stats.hinted_rows += 1;
            NdjsonHintDecision::UseHints
        } else {
            NdjsonHintDecision::Learning
        }
    }

    #[cfg(test)]
    pub(super) fn schema(&self) -> &NdjsonSchemaHints {
        &self.schema
    }

    pub(super) fn with_root_layout_match<'a, R>(
        &mut self,
        row: &'a [u8],
        f: impl FnOnce(&NdjsonObjectLayoutHint, &NdjsonRootLayoutMatch<'a, '_>) -> R,
    ) -> Option<R> {
        let root = self.schema.root_object.as_ref()?;
        let matched = match root.match_required_slots(
            row,
            &self.required_root_slots,
            &mut self.span_scratch,
        ) {
            Some(matched) => matched,
            None => {
                self.stats.layout_misses += 1;
                if self.stats.layout_misses > self.config.max_layout_misses {
                    self.disabled = true;
                    self.active = false;
                    self.stats.disabled = true;
                }
                return None;
            }
        };
        Some(f(root, &matched))
    }

    pub(super) fn stats(&self) -> &NdjsonHintStats {
        &self.stats
    }

    fn refresh_required_root_slots(&mut self) {
        let Some(root) = self.schema.root_object.as_ref() else {
            self.required_root_slots.clear();
            return;
        };
        self.required_root_slots = self
            .access
            .required_root_fields()
            .into_iter()
            .filter_map(|field| root.slot_for(field))
            .collect();
        self.required_root_slots.sort_unstable();
        self.required_root_slots.dedup();
    }
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
    fn hint_config_defaults_for_cold_path_activation() {
        let config = NdjsonHintConfig::default();
        assert_eq!(config.min_rows, 2);
        assert_eq!(config.max_rejects, 2);
        assert_eq!(config.max_layout_misses, 8);
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
    fn access_plan_collects_stream_source_without_item_projection_paths() {
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
        assert!(!access
            .paths
            .iter()
            .any(|path| path.root_field() == Some("key")));
        assert!(!access
            .paths
            .iter()
            .any(|path| path.root_field() == Some("value")));
    }

    #[test]
    fn hint_state_enables_only_after_stable_required_fields_are_seen() {
        let access = NdjsonHintAccessPlan {
            paths: vec![
                NdjsonHintPath {
                    steps: vec![NdjsonHintPathStep::Field(Arc::from("id"))],
                },
                NdjsonHintPath {
                    steps: vec![NdjsonHintPathStep::Field(Arc::from("name"))],
                },
            ],
        };
        let mut state = NdjsonHintState::new(
            NdjsonHintConfig {
                min_rows: 2,
                max_rejects: 0,
                max_layout_misses: 8,
            },
            access,
        );

        assert_eq!(
            state.observe_row(br#"{"id":1,"name":"a","extra":true}"#),
            NdjsonHintDecision::Learning
        );
        assert_eq!(
            state.observe_row(br#"{"id":2,"name":"b","extra":false}"#),
            NdjsonHintDecision::UseHints
        );
        assert_eq!(state.schema().root_slot_for("name"), Some(1));
        assert_eq!(state.stats().hinted_rows, 1);
    }

    #[test]
    fn hint_state_keeps_learning_when_required_fields_are_unstable() {
        let access = NdjsonHintAccessPlan {
            paths: vec![NdjsonHintPath {
                steps: vec![NdjsonHintPathStep::Field(Arc::from("name"))],
            }],
        };
        let mut state = NdjsonHintState::new(
            NdjsonHintConfig {
                min_rows: 2,
                max_rejects: 0,
                max_layout_misses: 8,
            },
            access,
        );

        assert_eq!(state.observe_row(br#"{"id":1,"name":"a"}"#), NdjsonHintDecision::Learning);
        assert_eq!(state.observe_row(br#"{"name":"b","id":2}"#), NdjsonHintDecision::Learning);
    }

    #[test]
    fn hint_state_disables_after_too_many_rejected_rows() {
        let access = NdjsonHintAccessPlan::default();
        let mut state = NdjsonHintState::new(
            NdjsonHintConfig {
                min_rows: 1,
                max_rejects: 1,
                max_layout_misses: 8,
            },
            access,
        );

        assert_eq!(state.observe_row(br#"[]"#), NdjsonHintDecision::Learning);
        assert_eq!(state.observe_row(br#"{"bad\nkey":1}"#), NdjsonHintDecision::Disabled);
        assert_eq!(state.observe_row(br#"{"id":1}"#), NdjsonHintDecision::Disabled);
        assert!(state.stats().disabled);
    }

    #[test]
    fn hint_state_disables_after_too_many_layout_misses() {
        let access = NdjsonHintAccessPlan {
            paths: vec![NdjsonHintPath {
                steps: vec![NdjsonHintPathStep::Field(Arc::from("name"))],
            }],
        };
        let mut state = NdjsonHintState::new(
            NdjsonHintConfig {
                min_rows: 1,
                max_rejects: 0,
                max_layout_misses: 1,
            },
            access,
        );

        assert_eq!(state.observe_row(br#"{"id":1,"name":"a"}"#), NdjsonHintDecision::UseHints);
        assert!(state
            .with_root_layout_match(br#"{"name":"b","id":2}"#, |_, _| ())
            .is_none());
        assert!(!state.stats().disabled);
        assert!(state
            .with_root_layout_match(br#"{"name":"c","id":3}"#, |_, _| ())
            .is_none());
        assert!(state.stats().disabled);
        assert_eq!(state.stats().layout_misses, 2);
        assert_eq!(state.observe_row(br#"{"id":4,"name":"d"}"#), NdjsonHintDecision::Disabled);
    }

    #[test]
    fn hint_state_matches_only_required_root_slots() {
        let access = NdjsonHintAccessPlan {
            paths: vec![
                NdjsonHintPath {
                    steps: vec![NdjsonHintPathStep::Field(Arc::from("id"))],
                },
                NdjsonHintPath {
                    steps: vec![NdjsonHintPathStep::Field(Arc::from("name"))],
                },
            ],
        };
        let mut state = NdjsonHintState::new(
            NdjsonHintConfig {
                min_rows: 1,
                max_rejects: 0,
                max_layout_misses: 0,
            },
            access,
        );

        assert_eq!(state.observe_row(br#"{"id":1,"name":"a"}"#), NdjsonHintDecision::UseHints);
        assert_eq!(
            state.observe_row(br#"{"id":2,"name":"b","extra":{"large":true}}"#),
            NdjsonHintDecision::UseHints
        );
        let mut saw_name = false;
        assert!(state
            .with_root_layout_match(br#"{"id":3,"name":"c","extra":{"large":true}}"#, |root, matched| {
                assert_eq!(
                    matched.value_at(root.slot_for("name").unwrap()),
                    Some(&b"\"c\""[..])
                );
                saw_name = true;
            })
            .is_some());
        assert!(saw_name);
        assert_eq!(state.stats().layout_misses, 0);
    }

    #[test]
    fn hint_state_stops_learning_after_activation() {
        let access = NdjsonHintAccessPlan {
            paths: vec![NdjsonHintPath {
                steps: vec![NdjsonHintPathStep::Field(Arc::from("id"))],
            }],
        };
        let mut state = NdjsonHintState::new(
            NdjsonHintConfig {
                min_rows: 2,
                max_rejects: 0,
                max_layout_misses: 0,
            },
            access,
        );

        assert_eq!(state.observe_row(br#"{"id":1,"tail":{"x":1}}"#), NdjsonHintDecision::Learning);
        assert_eq!(state.observe_row(br#"{"id":2,"tail":{"x":2}}"#), NdjsonHintDecision::UseHints);
        assert_eq!(state.observe_row(br#"{"id":3,"tail":{"different":true}}"#), NdjsonHintDecision::UseHints);
        assert_eq!(state.stats().learned_rows, 2);
        assert_eq!(state.stats().hinted_rows, 2);
    }

    #[test]
    fn root_layout_match_validates_order_and_exposes_value_spans() {
        let mut hints = NdjsonSchemaHints::default();
        hints.observe_row(br#"{"id":1,"name":"a"}"#);
        hints.observe_row(br#"{"id":2,"name":"b"}"#);
        let root = hints.root_object.as_ref().unwrap();

        let mut spans = Vec::new();
        let matched = root
            .match_row(br#"{"id":3,"name":"c"}"#, &mut spans)
            .unwrap();
        assert_eq!(matched.value_at(root.slot_for("id").unwrap()), Some(&b"3"[..]));
        assert_eq!(
            matched.value_at(root.slot_for("name").unwrap()),
            Some(&b"\"c\""[..])
        );
        assert!(root
            .match_row(br#"{"name":"c","id":3}"#, &mut spans)
            .is_none());
    }
}
