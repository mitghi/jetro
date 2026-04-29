use super::{NumOp, PipelineBody, Sink, Stage};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ViewInputMode {
    ReadsView,
    SkipsViewRead,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ViewOutputMode {
    PreservesInputView,
    BorrowedSubview,
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
    Take(usize),
    Skip(usize),
}

impl ViewStageCapability {
    pub(crate) fn input_mode(self) -> ViewInputMode {
        match self {
            Self::Filter { .. } | Self::Map { .. } => ViewInputMode::ReadsView,
            Self::Take(_) | Self::Skip(_) => ViewInputMode::SkipsViewRead,
        }
    }

    pub(crate) fn output_mode(self) -> ViewOutputMode {
        match self {
            Self::Map { .. } => ViewOutputMode::BorrowedSubview,
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
    Count,
    Numeric {
        op: NumOp,
        project_kernel: Option<usize>,
    },
    First,
    Last,
}

impl ViewSinkCapability {
    pub(crate) fn materialization(self) -> ViewMaterialization {
        match self {
            Self::Collect => ViewMaterialization::SinkOutputRows,
            Self::Count => ViewMaterialization::Never,
            Self::Numeric { .. } => ViewMaterialization::SinkNumericInput,
            Self::First | Self::Last => ViewMaterialization::SinkFinalRow,
        }
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
    use crate::pipeline::{
        BodyKernel, NumOp, NumericSink, PipelineBody, Sink, Stage, ViewInputMode,
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

        let take = ViewStageCapability::Take(2);
        assert_eq!(take.input_mode(), ViewInputMode::SkipsViewRead);
        assert_eq!(take.output_mode(), ViewOutputMode::PreservesInputView);
        assert_eq!(take.materialization(), ViewMaterialization::Never);
    }

    #[test]
    fn view_sink_metadata_describes_materialization_policy() {
        assert_eq!(
            ViewSinkCapability::Collect.materialization(),
            ViewMaterialization::SinkOutputRows
        );
        assert_eq!(
            ViewSinkCapability::Count.materialization(),
            ViewMaterialization::Never
        );
        assert_eq!(
            ViewSinkCapability::Numeric {
                op: NumOp::Sum,
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
    fn view_capabilities_preserve_expected_metadata() {
        let body = PipelineBody {
            stages: vec![
                Stage::Filter(Arc::new(crate::vm::Program::new(Vec::new(), ""))),
                Stage::Map(Arc::new(crate::vm::Program::new(Vec::new(), ""))),
                Stage::Take(2),
            ],
            stage_exprs: Vec::new(),
            sink: Sink::Numeric(NumericSink::projected(
                NumOp::Sum,
                Arc::new(crate::vm::Program::new(Vec::new(), "")),
            )),
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
                Stage::Filter(Arc::new(crate::vm::Program::new(Vec::new(), ""))),
                Stage::Map(Arc::new(crate::vm::Program::new(Vec::new(), ""))),
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
    match stage {
        Stage::Filter(_) if body.stage_kernels.get(idx)?.is_view_native() => {
            Some(ViewStageCapability::Filter { kernel: idx })
        }
        Stage::Map(_) if body.stage_kernels.get(idx)?.is_view_native() => {
            Some(ViewStageCapability::Map { kernel: idx })
        }
        Stage::Take(n) => Some(ViewStageCapability::Take(*n)),
        Stage::Skip(n) => Some(ViewStageCapability::Skip(*n)),
        _ => None,
    }
}

fn view_sink_capability(body: &PipelineBody) -> Option<ViewSinkCapability> {
    match &body.sink {
        Sink::Collect => Some(ViewSinkCapability::Collect),
        Sink::Count => Some(ViewSinkCapability::Count),
        Sink::First => Some(ViewSinkCapability::First),
        Sink::Last => Some(ViewSinkCapability::Last),
        Sink::Numeric(n) => {
            let project_kernel = if n.project.is_some() {
                if body.sink_kernels.first()?.is_view_native() {
                    Some(0)
                } else {
                    return None;
                }
            } else {
                None
            };
            Some(ViewSinkCapability::Numeric {
                op: n.op,
                project_kernel,
            })
        }
        Sink::ApproxCountDistinct => None,
    }
}
