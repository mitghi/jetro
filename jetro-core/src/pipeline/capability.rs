use crate::builtins::{
    BuiltinKeyedReducer, BuiltinSinkAccumulator, BuiltinSinkSpec, BuiltinViewInputMode,
    BuiltinViewMaterialization, BuiltinViewOutputMode, BuiltinViewStage,
};

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
    EmitsOwnedValue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ViewMaterialization {
    Never,
    StageFinalValue,
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
    Filter {
        kernel: usize,
    },
    Map {
        kernel: usize,
    },
    FlatMap {
        kernel: usize,
    },
    TakeWhile {
        kernel: usize,
    },
    DropWhile {
        kernel: usize,
    },
    Distinct {
        kernel: Option<usize>,
    },
    KeyedReduce {
        kind: BuiltinKeyedReducer,
        kernel: usize,
    },
    Take(usize),
    Skip(usize),
}

impl ViewStageCapability {
    pub(crate) fn from_stage_metadata(
        stage: BuiltinViewStage,
        usize_arg: Option<usize>,
        kernel_index: usize,
        kernel_is_view_native: bool,
    ) -> Option<Self> {
        match stage {
            BuiltinViewStage::Filter if kernel_is_view_native => Some(Self::Filter {
                kernel: kernel_index,
            }),
            BuiltinViewStage::Map if kernel_is_view_native => Some(Self::Map {
                kernel: kernel_index,
            }),
            BuiltinViewStage::FlatMap if kernel_is_view_native => Some(Self::FlatMap {
                kernel: kernel_index,
            }),
            BuiltinViewStage::TakeWhile if kernel_is_view_native => Some(Self::TakeWhile {
                kernel: kernel_index,
            }),
            BuiltinViewStage::DropWhile if kernel_is_view_native => Some(Self::DropWhile {
                kernel: kernel_index,
            }),
            BuiltinViewStage::Take => Some(Self::Take(usize_arg?)),
            BuiltinViewStage::Skip => Some(Self::Skip(usize_arg?)),
            _ => None,
        }
    }

    pub(crate) fn view_stage(self) -> BuiltinViewStage {
        match self {
            Self::Filter { .. } => BuiltinViewStage::Filter,
            Self::Map { .. } => BuiltinViewStage::Map,
            Self::FlatMap { .. } => BuiltinViewStage::FlatMap,
            Self::TakeWhile { .. } => BuiltinViewStage::TakeWhile,
            Self::DropWhile { .. } => BuiltinViewStage::DropWhile,
            Self::Distinct { .. } => BuiltinViewStage::Distinct,
            Self::KeyedReduce { .. } => BuiltinViewStage::KeyedReduce,
            Self::Take(_) => BuiltinViewStage::Take,
            Self::Skip(_) => BuiltinViewStage::Skip,
        }
    }

    pub(crate) fn input_mode(self) -> ViewInputMode {
        view_input_mode(self.view_stage().input_mode())
    }

    pub(crate) fn output_mode(self) -> ViewOutputMode {
        view_output_mode(self.view_stage().output_mode())
    }

    pub(crate) fn materialization(self) -> ViewMaterialization {
        if matches!(self, Self::KeyedReduce { .. }) {
            return ViewMaterialization::StageFinalValue;
        }
        view_materialization(self.view_stage().materialization())
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum ViewSinkCapability {
    Collect,
    Builtin {
        accumulator: BuiltinSinkAccumulator,
        predicate_kernel: Option<usize>,
        project_kernel: Option<usize>,
        numeric_op: Option<NumOp>,
        materialization: ViewMaterialization,
    },
}

impl ViewSinkCapability {
    pub(crate) fn from_sink_spec(
        spec: BuiltinSinkSpec,
        predicate_kernel: Option<usize>,
        project_kernel: Option<usize>,
        numeric_op: Option<NumOp>,
    ) -> Self {
        Self::Builtin {
            accumulator: spec.accumulator,
            predicate_kernel,
            project_kernel,
            numeric_op,
            materialization: sink_materialization(spec),
        }
    }

    pub(crate) fn materialization(self) -> ViewMaterialization {
        match self {
            Self::Collect => ViewMaterialization::SinkOutputRows,
            Self::Builtin {
                materialization, ..
            } => materialization,
        }
    }
}

fn sink_materialization(spec: BuiltinSinkSpec) -> ViewMaterialization {
    match spec.accumulator {
        BuiltinSinkAccumulator::Count | BuiltinSinkAccumulator::ApproxDistinct => {
            ViewMaterialization::Never
        }
        BuiltinSinkAccumulator::Numeric => ViewMaterialization::SinkNumericInput,
        BuiltinSinkAccumulator::SelectOne(_) => ViewMaterialization::SinkFinalRow,
    }
}

fn view_materialization(materialization: BuiltinViewMaterialization) -> ViewMaterialization {
    match materialization {
        BuiltinViewMaterialization::Never => ViewMaterialization::Never,
    }
}

fn view_input_mode(mode: BuiltinViewInputMode) -> ViewInputMode {
    match mode {
        BuiltinViewInputMode::ReadsView => ViewInputMode::ReadsView,
        BuiltinViewInputMode::SkipsViewRead => ViewInputMode::SkipsViewRead,
    }
}

fn view_output_mode(mode: BuiltinViewOutputMode) -> ViewOutputMode {
    match mode {
        BuiltinViewOutputMode::PreservesInputView => ViewOutputMode::PreservesInputView,
        BuiltinViewOutputMode::BorrowedSubview => ViewOutputMode::BorrowedSubview,
        BuiltinViewOutputMode::BorrowedSubviews => ViewOutputMode::BorrowedSubviews,
        BuiltinViewOutputMode::EmitsOwnedValue => ViewOutputMode::EmitsOwnedValue,
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
    use crate::builtins::{
        BuiltinMethod, BuiltinSelectionPosition, BuiltinSinkAccumulator, BuiltinStageMerge,
        BuiltinViewStage,
    };
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
            ViewSinkCapability::Builtin {
                accumulator: BuiltinSinkAccumulator::Count,
                predicate_kernel: None,
                project_kernel: None,
                numeric_op: None,
                materialization: ViewMaterialization::Never,
            }
            .materialization(),
            ViewMaterialization::Never
        );
        assert_eq!(
            ViewSinkCapability::Builtin {
                accumulator: BuiltinSinkAccumulator::Numeric,
                predicate_kernel: None,
                project_kernel: Some(0),
                numeric_op: Some(NumOp::Sum),
                materialization: ViewMaterialization::SinkNumericInput,
            }
            .materialization(),
            ViewMaterialization::SinkNumericInput
        );
        assert_eq!(
            ViewSinkCapability::Builtin {
                accumulator: BuiltinSinkAccumulator::SelectOne(BuiltinSelectionPosition::First),
                predicate_kernel: None,
                project_kernel: None,
                numeric_op: None,
                materialization: ViewMaterialization::SinkFinalRow,
            }
            .materialization(),
            ViewMaterialization::SinkFinalRow
        );
    }

    #[test]
    fn sink_view_capability_uses_carried_metadata() {
        assert!(matches!(
            Sink::Reducer(ReducerSpec::count()).view_capability(&[]),
            Some(ViewSinkCapability::Builtin {
                accumulator: BuiltinSinkAccumulator::Count,
                predicate_kernel: None,
                project_kernel: None,
                numeric_op: None,
                materialization: ViewMaterialization::Never,
            })
        ));
        assert!(matches!(
            Sink::Terminal(BuiltinMethod::First).view_capability(&[]),
            Some(ViewSinkCapability::Builtin {
                accumulator: BuiltinSinkAccumulator::SelectOne(BuiltinSelectionPosition::First),
                predicate_kernel: None,
                project_kernel: None,
                numeric_op: None,
                materialization: ViewMaterialization::SinkFinalRow,
            })
        ));
        assert!(matches!(
            Sink::Terminal(BuiltinMethod::Last).view_capability(&[]),
            Some(ViewSinkCapability::Builtin {
                accumulator: BuiltinSinkAccumulator::SelectOne(BuiltinSelectionPosition::Last),
                predicate_kernel: None,
                project_kernel: None,
                numeric_op: None,
                materialization: ViewMaterialization::SinkFinalRow,
            })
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
                predicate_expr: None,
                projection_expr: None,
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
