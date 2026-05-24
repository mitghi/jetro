//! View-pipeline reducer stage for keyed aggregations (`group_by`, `count_by`,
//! `index_by`). Processes rows through a `ValueView` and accumulates results
//! into an `IndexMap` keyed by the `ViewKey` extracted from each row.

use indexmap::IndexMap;

use crate::{
    builtins::{BuiltinArgs, BuiltinKeyedReducer, BuiltinMethod},
    data::value::Val,
    data::view::ValueView,
    exec::pipeline,
};

use super::key::ViewKey;

/// Execution plan for a keyed-reduce barrier stage detected in the view pipeline.
/// Carries the view-domain prefix, the reducer accumulator, and the stage count
/// consumed so the caller knows where to resume materialised execution.
pub(super) struct ReducingStagePlan {
    /// View-domain stages that precede the keyed-reduce barrier.
    pub(super) prefix: Vec<pipeline::ViewStageCapability>,
    /// Accumulator that aggregates keyed observations from the view frontier.
    pub(super) reducer: ViewStageReducer,
    /// Number of pipeline stages consumed by this plan (prefix + barrier).
    pub(super) consumed_stages: usize,
}

/// State machine for view-level keyed aggregation. Currently the only variant
/// is `Keyed`, which handles `group_by`, `count_by`, and `index_by`.
pub(super) enum ViewStageReducer {
    /// A keyed reducer accumulating entries into an `IndexMap` keyed by `ViewKey`.
    Keyed {
        /// The specific keyed aggregation kind to perform.
        kind: BuiltinKeyedReducer,
        /// Index into `stage_kernels` for the key-extraction kernel.
        kernel: usize,
        /// Whether downstream suffixes need full values or only the object key set.
        value_need: KeyedValueNeed,
        /// Accumulated entries, one per distinct key observed so far.
        entries: IndexMap<ViewKey, KeyedEntry>,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum KeyedValueNeed {
    Full,
    KeysOnly,
}

/// The per-key accumulated value for a `ViewStageReducer::Keyed` operation.
pub(super) enum KeyedEntry {
    /// Key presence only; enough for suffixes such as `.keys()`.
    KeyOnly,
    /// Running count for `count_by`.
    Count(i64),
    /// Last-seen materialised value for `index_by`.
    Value(Val),
    /// Accumulating list of materialised values for `group_by`.
    Group(Vec<Val>),
}

impl ViewStageReducer {
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
    pub(super) fn observe<'a, V>(
        &mut self,
        item: &V,
        stage_kernels: &[pipeline::BodyKernel],
        vm: &mut crate::vm::VM,
    ) -> Option<()>
    where
        V: ValueView<'a> + 'a,
    {
        match self {
            Self::Keyed {
                kind,
                kernel,
                value_need,
                entries,
            } => {
                let key =
                    super::eval_view_key_with_vm(item, Some(stage_kernels.get(*kernel)?), vm)?;
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
                            entries.insert(key, KeyedEntry::Value(item.materialize()));
                        }
                    }
                    BuiltinKeyedReducer::Group => {
                        if matches!(value_need, KeyedValueNeed::KeysOnly) {
                            entries.entry(key).or_insert(KeyedEntry::KeyOnly);
                        } else {
                            match entries.entry(key) {
                                indexmap::map::Entry::Occupied(mut entry) => {
                                    if let KeyedEntry::Group(items) = entry.get_mut() {
                                        items.push(item.materialize());
                                    }
                                }
                                indexmap::map::Entry::Vacant(entry) => {
                                    entry.insert(KeyedEntry::Group(vec![item.materialize()]));
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
    pub(super) fn finish(self) -> Val {
        match self {
            Self::Keyed { entries, .. } => Val::obj(
                entries
                    .into_iter()
                    .map(|(key, entry)| {
                        let value = match entry {
                            KeyedEntry::KeyOnly => Val::Null,
                            KeyedEntry::Count(count) => Val::Int(count),
                            KeyedEntry::Value(value) => value,
                            KeyedEntry::Group(items) => Val::arr(items),
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
pub(super) fn plan(body: &pipeline::PipelineBody) -> Option<ReducingStagePlan> {
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
            if call.method == BuiltinMethod::Keys && matches!(call.args, BuiltinArgs::None) =>
        {
            KeyedValueNeed::KeysOnly
        }
        _ => KeyedValueNeed::Full,
    }
}
