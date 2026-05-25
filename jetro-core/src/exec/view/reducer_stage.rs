//! View-pipeline reducer stage for keyed aggregations (`group_by`, `count_by`,
//! `index_by`). Processes rows through a `ValueView` and accumulates results
//! into an `IndexMap` keyed by the `ViewKey` extracted from each row.

use indexmap::IndexMap;

use crate::{
    builtins::{
        registry::{view_projection_needs_only_object_keys, BuiltinId},
        BuiltinArgs, BuiltinKeyedReducer, BuiltinMethod,
    },
    data::value::Val,
    data::view::ValueView,
    exec::pipeline,
};

use super::key::ViewKey;

/// Execution plan for a keyed-reduce barrier stage detected in the view pipeline.
/// Carries the view-domain prefix, the reducer accumulator, and the stage count
/// consumed so the caller knows where to resume materialised execution.
pub(super) struct ReducingStagePlan<Row> {
    /// View-domain stages that precede the keyed-reduce barrier.
    pub(super) prefix: Vec<pipeline::ViewStageCapability>,
    /// Accumulator that aggregates keyed observations from the view frontier.
    pub(super) reducer: ViewStageReducer<Row>,
    /// Number of pipeline stages consumed by this plan (prefix + barrier).
    pub(super) consumed_stages: usize,
}

/// State machine for view-level keyed aggregation. Currently the only variant
/// is `Keyed`, which handles `group_by`, `count_by`, and `index_by`.
pub(super) enum ViewStageReducer<Row> {
    /// A keyed reducer accumulating entries into an `IndexMap` keyed by `ViewKey`.
    Keyed {
        /// The specific keyed aggregation kind to perform.
        kind: BuiltinKeyedReducer,
        /// Index into `stage_kernels` for the key-extraction kernel.
        kernel: usize,
        /// Whether downstream suffixes need full values or only the object key set.
        value_need: KeyedValueNeed,
        /// Accumulated entries, one per distinct key observed so far.
        entries: IndexMap<ViewKey, KeyedEntry<Row>>,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum KeyedValueNeed {
    Full,
    KeysOnly,
}

/// The per-key accumulated value for a `ViewStageReducer::Keyed` operation.
pub(super) enum KeyedEntry<Row> {
    /// Key presence only; enough for suffixes such as `.keys()`.
    KeyOnly,
    /// Running count for `count_by`.
    Count(i64),
    /// Last-seen borrowed value for `index_by`.
    Value(Row),
    /// Accumulating list of borrowed values for `group_by`.
    Group(Vec<Row>),
}

impl<Row> ViewStageReducer<Row>
where
    Row: Clone,
{
    // Constructs a `ViewStageReducer` from a `KeyedReduce` capability; returns `None` for all other variants.
    fn from_capability(
        capability: pipeline::ViewStageCapability,
        value_need: KeyedValueNeed,
    ) -> Option<Self> {
        match capability {
            pipeline::ViewStageCapability::KeyedReduce { kind, kernel } => Some(Self::Keyed {
                kind,
                kernel,
                value_need,
                entries: IndexMap::new(),
            }),
            _ => None,
        }
    }

    /// Processes one view row: extracts the group key via the configured kernel
    /// and updates the appropriate `KeyedEntry` for that key. Returns `None`
    /// when the kernel index is out of bounds or the key cannot be extracted.
    pub(super) fn observe<'a, V, F>(
        &mut self,
        item: &V,
        stage_kernels: &[pipeline::BodyKernel],
        vm: &mut crate::vm::VM,
        mut eval_key: F,
    ) -> Option<()>
    where
        V: ValueView<'a> + Clone + Into<Row> + 'a,
        F: FnMut(&V, Option<&pipeline::BodyKernel>, &mut crate::vm::VM) -> Option<ViewKey>,
    {
        match self {
            Self::Keyed {
                kind,
                kernel,
                value_need,
                entries,
            } => {
                let key = eval_key(item, Some(stage_kernels.get(*kernel)?), vm)?;
                match kind {
                    BuiltinKeyedReducer::Count => match entries.entry(key) {
                        indexmap::map::Entry::Occupied(mut entry) => {
                            if let KeyedEntry::Count(count) = entry.get_mut() {
                                *count += 1;
                            }
                        }
                        indexmap::map::Entry::Vacant(entry) => {
                            entry.insert(KeyedEntry::Count(1));
                        }
                    },
                    BuiltinKeyedReducer::Index => {
                        if matches!(value_need, KeyedValueNeed::KeysOnly) {
                            entries.entry(key).or_insert(KeyedEntry::KeyOnly);
                        } else {
                            entries.insert(key, KeyedEntry::Value(item.clone().into()));
                        }
                    }
                    BuiltinKeyedReducer::Group => {
                        if matches!(value_need, KeyedValueNeed::KeysOnly) {
                            entries.entry(key).or_insert(KeyedEntry::KeyOnly);
                        } else {
                            match entries.entry(key) {
                                indexmap::map::Entry::Occupied(mut entry) => {
                                    if let KeyedEntry::Group(items) = entry.get_mut() {
                                        items.push(item.clone().into());
                                    }
                                }
                                indexmap::map::Entry::Vacant(entry) => {
                                    entry.insert(KeyedEntry::Group(vec![item.clone().into()]));
                                }
                            }
                        }
                    }
                }
                Some(())
            }
        }
    }

    /// Converts the accumulated `KeyedEntry` map into a final `Val::Obj`,
    /// where each key maps to its count, indexed value, or grouped array.
    pub(super) fn finish<'a>(self) -> Val
    where
        Row: ValueView<'a> + 'a,
    {
        match self {
            Self::Keyed { entries, .. } => Val::obj(
                entries
                    .into_iter()
                    .map(|(key, entry)| {
                        let value = match entry {
                            KeyedEntry::KeyOnly => Val::Null,
                            KeyedEntry::Count(count) => Val::Int(count),
                            KeyedEntry::Value(value) => pipeline::view_kernel_view_to_owned(value),
                            KeyedEntry::Group(items) => Val::arr(
                                items
                                    .into_iter()
                                    .map(pipeline::view_kernel_view_to_owned)
                                    .collect(),
                            ),
                        };
                        (key.object_key(), value)
                    })
                    .collect(),
            ),
        }
    }
}

/// Scans `body.stages` for the first keyed-reduce barrier stage
/// (`KeyedReduce` with `StageFinalValue` materialisation) preceded only by
/// view-native `Never`-materialisation stages. Returns a `ReducingStagePlan`
/// on success, or `None` when no qualifying barrier is found.
pub(super) fn plan<Row>(body: &pipeline::PipelineBody) -> Option<ReducingStagePlan<Row>>
where
    Row: Clone,
{
    let mut prefix = Vec::new();
    for (idx, stage) in body.stages.iter().enumerate() {
        let capability = stage.view_capability(idx, body.stage_kernels.get(idx))?;
        match capability.materialization() {
            pipeline::ViewMaterialization::Never => prefix.push(capability),
            pipeline::ViewMaterialization::StageFinalValue => {
                let value_need = keyed_reducer_value_need(body, idx);
                return Some(ReducingStagePlan {
                    prefix,
                    reducer: ViewStageReducer::from_capability(capability, value_need)?,
                    consumed_stages: idx + 1,
                });
            }
            _ => return None,
        }
    }
    None
}

fn keyed_reducer_value_need(body: &pipeline::PipelineBody, reducer_stage: usize) -> KeyedValueNeed {
    match body.stages.get(reducer_stage + 1) {
        Some(pipeline::Stage::Builtin(call))
            if key_only_object_projection(call.method, &call.args) =>
        {
            KeyedValueNeed::KeysOnly
        }
        _ => KeyedValueNeed::Full,
    }
}

fn key_only_object_projection(method: BuiltinMethod, args: &BuiltinArgs) -> bool {
    view_projection_needs_only_object_keys(BuiltinId::from_method(method), args)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::builtins::{BuiltinArgs, BuiltinMethod};

    use super::key_only_object_projection;

    #[test]
    fn keyed_reducer_key_only_policy_comes_from_registry_metadata() {
        assert!(key_only_object_projection(
            BuiltinMethod::Len,
            &BuiltinArgs::None
        ));
        assert!(key_only_object_projection(
            BuiltinMethod::HasKey,
            &BuiltinArgs::Str(Arc::from("open"))
        ));
        assert!(key_only_object_projection(
            BuiltinMethod::HasAll,
            &BuiltinArgs::StrVec(vec![Arc::from("open"), Arc::from("closed")])
        ));
        assert!(!key_only_object_projection(
            BuiltinMethod::Missing,
            &BuiltinArgs::Str(Arc::from("open"))
        ));
        assert!(!key_only_object_projection(
            BuiltinMethod::Values,
            &BuiltinArgs::None
        ));
    }
}
