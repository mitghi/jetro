use indexmap::IndexMap;

use crate::{pipeline, value::Val, value_view::ValueView};

use super::key::ViewKey;

pub(super) struct ReducingStagePlan {
    pub(super) prefix: Vec<pipeline::ViewStageCapability>,
    pub(super) reducer: ViewStageReducer,
    pub(super) consumed_stages: usize,
}

pub(super) enum ViewStageReducer {
    Keyed {
        kind: pipeline::ViewKeyedReducer,
        kernel: usize,
        entries: IndexMap<ViewKey, KeyedEntry>,
    },
}

pub(super) enum KeyedEntry {
    Count(i64),
    Value(Val),
    Group(Vec<Val>),
}

impl ViewStageReducer {
    fn from_capability(capability: pipeline::ViewStageCapability) -> Option<Self> {
        match capability {
            pipeline::ViewStageCapability::KeyedReduce { kind, kernel } => Some(Self::Keyed {
                kind,
                kernel,
                entries: IndexMap::new(),
            }),
            _ => None,
        }
    }

    pub(super) fn observe<'a, V>(
        &mut self,
        item: &V,
        stage_kernels: &[pipeline::BodyKernel],
    ) -> Option<()>
    where
        V: ValueView<'a>,
    {
        match self {
            Self::Keyed {
                kind,
                kernel,
                entries,
            } => {
                let key = super::eval_view_key(item, Some(stage_kernels.get(*kernel)?))?;
                match kind {
                    pipeline::ViewKeyedReducer::Count => match entries.entry(key) {
                        indexmap::map::Entry::Occupied(mut entry) => {
                            if let KeyedEntry::Count(count) = entry.get_mut() {
                                *count += 1;
                            }
                        }
                        indexmap::map::Entry::Vacant(entry) => {
                            entry.insert(KeyedEntry::Count(1));
                        }
                    },
                    pipeline::ViewKeyedReducer::Index => {
                        entries.insert(key, KeyedEntry::Value(item.materialize()));
                    }
                    pipeline::ViewKeyedReducer::Group => match entries.entry(key) {
                        indexmap::map::Entry::Occupied(mut entry) => {
                            if let KeyedEntry::Group(items) = entry.get_mut() {
                                items.push(item.materialize());
                            }
                        }
                        indexmap::map::Entry::Vacant(entry) => {
                            entry.insert(KeyedEntry::Group(vec![item.materialize()]));
                        }
                    },
                }
                Some(())
            }
        }
    }

    pub(super) fn finish(self) -> Val {
        match self {
            Self::Keyed { entries, .. } => Val::obj(
                entries
                    .into_iter()
                    .map(|(key, entry)| {
                        let value = match entry {
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

pub(super) fn plan(body: &pipeline::PipelineBody) -> Option<ReducingStagePlan> {
    let mut prefix = Vec::new();
    for (idx, stage) in body.stages.iter().enumerate() {
        let capability = stage.view_capability(idx, body.stage_kernels.get(idx))?;
        match capability.materialization() {
            pipeline::ViewMaterialization::Never => prefix.push(capability),
            pipeline::ViewMaterialization::StageFinalValue => {
                return Some(ReducingStagePlan {
                    prefix,
                    reducer: ViewStageReducer::from_capability(capability)?,
                    consumed_stages: idx + 1,
                });
            }
            _ => return None,
        }
    }
    None
}
