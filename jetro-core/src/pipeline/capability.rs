use crate::builtins::{BuiltinViewMaterialization, BuiltinViewSink};

use super::{NumOp, PipelineBody, Stage};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ViewInputMode {
    ReadsView,
    SkipsViewRead,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ViewOutputMode {
    PreservesInputView,
    BorrowedSubview,
    BorrowedSubviews,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ViewMaterialization {
    Never,
    SinkOutputRows,
    SinkFinalRow,
    SinkNumericInput,
}

#[derive(Debug, Clone)]
pub(crate) struct ViewPipelineCapabilities {
    pub stages: Vec<ViewStageCapability>,
    pub sink: ViewSinkCapability,
}

#[derive(Debug, Clone)]
pub(crate) struct ViewPrefixCapabilities {
    pub stages: Vec<ViewStageCapability>,
    pub consumed_stages: usize,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum ViewStageCapability {
    Filter { kernel: usize },
    Map { kernel: usize },
    FlatMap { kernel: usize },
    Take(usize),
    Skip(usize),
}

impl ViewStageCapability {
    pub(crate) fn input_mode(self) -> ViewInputMode {
        match self {
            Self::Filter { .. } | Self::Map { .. } | Self::FlatMap { .. } => {
                ViewInputMode::ReadsView
            }
            Self::Take(_) | Self::Skip(_) => ViewInputMode::SkipsViewRead,
        }
    }

    pub(crate) fn output_mode(self) -> ViewOutputMode {
        match self {
            Self::Map { .. } => ViewOutputMode::BorrowedSubview,
            Self::FlatMap { .. } => ViewOutputMode::BorrowedSubviews,
            Self::Filter { .. } | Self::Take(_) | Self::Skip(_) => {
                ViewOutputMode::PreservesInputView
            }
        }
    }

    pub(crate) fn materialization(self) -> ViewMaterialization {
        ViewMaterialization::Never
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum ViewSinkCapability {
    Collect,
    Count {
        predicate_kernel: Option<usize>,
    },
    Numeric {
        op: NumOp,
        predicate_kernel: Option<usize>,
        project_kernel: Option<usize>,
    },
    First,
    Last,
}

impl ViewSinkCapability {
    pub(crate) fn from_builtin_sink(sink: BuiltinViewSink) -> Option<Self> {
        match sink {
            BuiltinViewSink::Count => Some(Self::Count {
                predicate_kernel: None,
            }),
            BuiltinViewSink::First => Some(Self::First),
            BuiltinViewSink::Last => Some(Self::Last),
            BuiltinViewSink::Numeric => None,
        }
    }

    pub(crate) fn materialization(self) -> ViewMaterialization {
        match self {
            Self::Collect => ViewMaterialization::SinkOutputRows,
            Self::Count { .. } => view_materialization(BuiltinViewSink::Count.materialization()),
            Self::Numeric { .. } => {
                view_materialization(BuiltinViewSink::Numeric.materialization())
            }
            Self::First => view_materialization(BuiltinViewSink::First.materialization()),
            Self::Last => view_materialization(BuiltinViewSink::Last.materialization()),
        }
    }
}

fn view_materialization(materialization: BuiltinViewMaterialization) -> ViewMaterialization {
    match materialization {
        BuiltinViewMaterialization::Never => ViewMaterialization::Never,
        BuiltinViewMaterialization::SinkFinalRow => ViewMaterialization::SinkFinalRow,
        BuiltinViewMaterialization::SinkNumericInput => ViewMaterialization::SinkNumericInput,
    }
}

pub(crate) fn view_capabilities(body: &PipelineBody) -> Option<ViewPipelineCapabilities> {
    Some(ViewPipelineCapabilities {
        stages: view_stage_capabilities(body)?,
        sink: view_sink_capability(body)?,
    })
}

pub(crate) fn view_prefix_capabilities(body: &PipelineBody) -> Option<ViewPrefixCapabilities> {
    let mut stages = Vec::new();
    for (idx, stage) in body.stages.iter().enumerate() {
        let Some(capability) = view_stage_capability(body, idx, stage) else {
            break;
        };
        if !matches!(capability.materialization(), ViewMaterialization::Never) {
            break;
        }
        stages.push(capability);
    }
    if stages.is_empty() {
        return None;
    }
    Some(ViewPrefixCapabilities {
        consumed_stages: stages.len(),
        stages,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::ast::BinOp;
    use crate::builtins::{BuiltinStageMerge, BuiltinViewSink, BuiltinViewStage};
    use crate::pipeline::{
        BodyKernel, NumOp, PipelineBody, ReducerOp, ReducerSpec, Sink, Stage, ViewInputMode,
        ViewMaterialization, ViewOutputMode, ViewSinkCapability, ViewStageCapability,
    };
    use crate::value::Val;

    use super::{view_capabilities, view_prefix_capabilities};

    #[test]
    fn view_stage_metadata_describes_borrowing_and_materialization() {
        let filter = ViewStageCapability::Filter { kernel: 0 };
        assert_eq!(filter.input_mode(), ViewInputMode::ReadsView);
        assert_eq!(filter.output_mode(), ViewOutputMode::PreservesInputView);
        assert_eq!(filter.materialization(), ViewMaterialization::Never);

        let map = ViewStageCapability::Map { kernel: 0 };
        assert_eq!(map.input_mode(), ViewInputMode::ReadsView);
        assert_eq!(map.output_mode(), ViewOutputMode::BorrowedSubview);
        assert_eq!(map.materialization(), ViewMaterialization::Never);

        let flat_map = ViewStageCapability::FlatMap { kernel: 0 };
        assert_eq!(flat_map.input_mode(), ViewInputMode::ReadsView);
        assert_eq!(flat_map.output_mode(), ViewOutputMode::BorrowedSubviews);
        assert_eq!(flat_map.materialization(), ViewMaterialization::Never);

        let take = ViewStageCapability::Take(2);
        assert_eq!(take.input_mode(), ViewInputMode::SkipsViewRead);
        assert_eq!(take.output_mode(), ViewOutputMode::PreservesInputView);
        assert_eq!(take.materialization(), ViewMaterialization::Never);
    }

    #[test]
    fn stage_view_capability_comes_from_stage_metadata() {
        let prog = Arc::new(crate::vm::Program::new(Vec::new(), ""));
        let filter = Stage::Filter(prog.clone(), BuiltinViewStage::Filter)
            .view_capability(4, Some(&BodyKernel::FieldRead(Arc::<str>::from("score"))))
            .unwrap();
        let map = Stage::Map(prog, BuiltinViewStage::Map)
            .view_capability(5, Some(&BodyKernel::FieldRead(Arc::<str>::from("name"))))
            .unwrap();
        let flat_map = Stage::FlatMap(
            Arc::new(crate::vm::Program::new(Vec::new(), "")),
            BuiltinViewStage::FlatMap,
        )
        .view_capability(6, Some(&BodyKernel::FieldRead(Arc::<str>::from("items"))))
        .unwrap();
        let take = Stage::Take(2, BuiltinViewStage::Take, BuiltinStageMerge::UsizeMin)
            .view_capability(7, None)
            .unwrap();
        let skip = Stage::Skip(
            1,
            BuiltinViewStage::Skip,
            BuiltinStageMerge::UsizeSaturatingAdd,
        )
        .view_capability(8, None)
        .unwrap();

        assert!(matches!(filter, ViewStageCapability::Filter { kernel: 4 }));
        assert_eq!(map.output_mode(), ViewOutputMode::BorrowedSubview);
        assert_eq!(flat_map.output_mode(), ViewOutputMode::BorrowedSubviews);
        assert!(matches!(take, ViewStageCapability::Take(2)));
        assert!(matches!(skip, ViewStageCapability::Skip(1)));
        let cancel = crate::builtins::BuiltinMethod::Reverse
            .spec()
            .cancellation
            .unwrap();
        assert!(Stage::Reverse(cancel).view_capability(9, None).is_none());
    }

    #[test]
    fn view_sink_metadata_describes_materialization_policy() {
        assert_eq!(
            ViewSinkCapability::Collect.materialization(),
            ViewMaterialization::SinkOutputRows
        );
        assert_eq!(
            ViewSinkCapability::Count {
                predicate_kernel: None
            }
            .materialization(),
            ViewMaterialization::Never
        );
        assert_eq!(
            ViewSinkCapability::Numeric {
                op: NumOp::Sum,
                predicate_kernel: None,
                project_kernel: Some(0)
            }
            .materialization(),
            ViewMaterialization::SinkNumericInput
        );
        assert_eq!(
            ViewSinkCapability::First.materialization(),
            ViewMaterialization::SinkFinalRow
        );
    }

    #[test]
    fn sink_view_capability_uses_carried_metadata() {
        assert!(matches!(
            Sink::Reducer(ReducerSpec::count()).view_capability(&[]),
            Some(ViewSinkCapability::Count {
                predicate_kernel: None
            })
        ));
        assert!(matches!(
            Sink::First(BuiltinViewSink::First).view_capability(&[]),
            Some(ViewSinkCapability::First)
        ));
        assert!(matches!(
            Sink::Last(BuiltinViewSink::Last).view_capability(&[]),
            Some(ViewSinkCapability::Last)
        ));
    }

    #[test]
    fn view_capabilities_preserve_expected_metadata() {
        let body = PipelineBody {
            stages: vec![
                Stage::Filter(
                    Arc::new(crate::vm::Program::new(Vec::new(), "")),
                    BuiltinViewStage::Filter,
                ),
                Stage::Map(
                    Arc::new(crate::vm::Program::new(Vec::new(), "")),
                    BuiltinViewStage::Map,
                ),
                Stage::Take(
                    2,
                    crate::builtins::BuiltinViewStage::Take,
                    crate::builtins::BuiltinStageMerge::UsizeMin,
                ),
            ],
            stage_exprs: Vec::new(),
            sink: Sink::Reducer(ReducerSpec {
                op: ReducerOp::Numeric(NumOp::Sum),
                predicate: None,
                projection: Some(Arc::new(crate::vm::Program::new(Vec::new(), ""))),
            }),
            stage_kernels: vec![
                BodyKernel::FieldCmpLit(Arc::from("score"), BinOp::Gt, Val::Int(10)),
                BodyKernel::FieldRead(Arc::from("score")),
                BodyKernel::Generic,
            ],
            sink_kernels: vec![BodyKernel::FieldRead(Arc::from("score"))],
        };

        let capabilities = view_capabilities(&body).unwrap();
        assert_eq!(capabilities.stages.len(), 3);
        assert_eq!(
            capabilities.stages[0].output_mode(),
            ViewOutputMode::PreservesInputView
        );
        assert_eq!(
            capabilities.stages[1].output_mode(),
            ViewOutputMode::BorrowedSubview
        );
        assert_eq!(
            capabilities.sink.materialization(),
            ViewMaterialization::SinkNumericInput
        );
    }

    #[test]
    fn view_prefix_stops_at_first_non_view_stage() {
        let body = PipelineBody {
            stages: vec![
                Stage::Filter(
                    Arc::new(crate::vm::Program::new(Vec::new(), "")),
                    BuiltinViewStage::Filter,
                ),
                Stage::Map(
                    Arc::new(crate::vm::Program::new(Vec::new(), "")),
                    BuiltinViewStage::Map,
                ),
                Stage::Builtin(crate::pipeline::PipelineBuiltinCall {
                    method: crate::builtins::BuiltinMethod::Upper,
                    args: crate::builtins::BuiltinArgs::None,
                }),
            ],
            stage_exprs: Vec::new(),
            sink: Sink::Collect,
            stage_kernels: vec![
                BodyKernel::FieldCmpLit(Arc::from("score"), BinOp::Gt, Val::Int(10)),
                BodyKernel::FieldRead(Arc::from("name")),
                BodyKernel::Generic,
            ],
            sink_kernels: Vec::new(),
        };

        assert!(view_capabilities(&body).is_none());
        let prefix = view_prefix_capabilities(&body).unwrap();
        assert_eq!(prefix.consumed_stages, 2);
        assert_eq!(prefix.stages.len(), 2);
    }
}

fn view_stage_capabilities(body: &PipelineBody) -> Option<Vec<ViewStageCapability>> {
    let mut out = Vec::with_capacity(body.stages.len());
    for (idx, stage) in body.stages.iter().enumerate() {
        out.push(view_stage_capability(body, idx, stage)?);
    }
    Some(out)
}

fn view_stage_capability(
    body: &PipelineBody,
    idx: usize,
    stage: &Stage,
) -> Option<ViewStageCapability> {
    stage.view_capability(idx, body.stage_kernels.get(idx))
}

fn view_sink_capability(body: &PipelineBody) -> Option<ViewSinkCapability> {
    body.sink.view_capability(&body.sink_kernels)
}
