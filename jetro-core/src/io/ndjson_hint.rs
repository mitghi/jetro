use super::ndjson_byte::visit_root_object_fields;
use std::sync::Arc;

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
}
