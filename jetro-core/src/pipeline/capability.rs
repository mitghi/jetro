use super::{NumOp, PipelineBody, Sink, Stage};

#[derive(Debug, Clone)]
pub(crate) struct ViewPipelineCapabilities {
    pub stages: Vec<ViewStageCapability>,
    pub sink: ViewSinkCapability,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum ViewStageCapability {
    Filter { kernel: usize },
    Map { kernel: usize },
    Take(usize),
    Skip(usize),
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

pub(crate) fn view_capabilities(body: &PipelineBody) -> Option<ViewPipelineCapabilities> {
    Some(ViewPipelineCapabilities {
        stages: view_stage_capabilities(body)?,
        sink: view_sink_capability(body)?,
    })
}

fn view_stage_capabilities(body: &PipelineBody) -> Option<Vec<ViewStageCapability>> {
    let mut out = Vec::with_capacity(body.stages.len());
    for (idx, stage) in body.stages.iter().enumerate() {
        out.push(match stage {
            Stage::Filter(_) if body.stage_kernels.get(idx)?.is_view_native() => {
                ViewStageCapability::Filter { kernel: idx }
            }
            Stage::Map(_) if body.stage_kernels.get(idx)?.is_view_native() => {
                ViewStageCapability::Map { kernel: idx }
            }
            Stage::Take(n) => ViewStageCapability::Take(*n),
            Stage::Skip(n) => ViewStageCapability::Skip(*n),
            _ => return None,
        });
    }
    Some(out)
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
