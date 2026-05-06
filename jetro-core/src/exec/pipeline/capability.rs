//! View-pipeline capability descriptors for stages and sinks.
//!
//! Defines the borrowing, materialisation, and input/output mode traits that let
//! the view execution path decide, per stage, whether it can operate on borrowed
//! `ValueView` slices or must materialise rows into owned `Val`s.

use crate::builtins::{
    BuiltinKeyedReducer, BuiltinSinkAccumulator, BuiltinSinkSpec, BuiltinViewInputMode,
    BuiltinViewOutputMode, BuiltinViewStage,
};
use crate::parse::chain_ir::PullDemand;

use super::{PipelineBody, Stage};

/// Describes how a source can be traversed without materialising the full row set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SourceCapabilities {
    /// Source can be streamed from the beginning.
    pub forward_stream: bool,
    /// Source can be streamed from the end.
    pub reverse_stream: bool,
    /// Source can seek directly to a zero-based array child.
    pub indexed_array_child: bool,
    /// Source rows can remain in the borrowed tape/view domain.
    pub tape_view: bool,
    /// Source can fall back to materialising owned values.
    pub materialized_fallback: bool,
}

impl SourceCapabilities {
    /// Capabilities for a `ValueView` array source.
    pub(crate) const VIEW_ARRAY: Self = Self {
        forward_stream: true,
        reverse_stream: true,
        indexed_array_child: true,
        tape_view: true,
        materialized_fallback: true,
    };

    /// Chooses the most direct access mode that satisfies `demand`.
    pub(crate) fn choose_access(self, demand: PullDemand) -> SourceAccessMode {
        match demand {
            PullDemand::NthInput(idx) if self.indexed_array_child => SourceAccessMode::Indexed(idx),
            PullDemand::LastInput(n) if self.reverse_stream => SourceAccessMode::Reverse { outputs: n },
            PullDemand::FirstInput(n) if self.forward_stream => SourceAccessMode::ForwardBounded(n),
            _ if self.forward_stream => SourceAccessMode::Forward,
            _ => SourceAccessMode::MaterializedFallback,
        }
    }
}

/// Physical traversal selected from source capabilities plus propagated demand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SourceAccessMode {
    /// Stream rows from the beginning with no demand cap.
    Forward,
    /// Stream at most this many input rows from the beginning.
    ForwardBounded(usize),
    /// Stream rows from the end until enough outputs have been accepted.
    Reverse {
        /// Number of demanded outputs.
        outputs: usize,
    },
    /// Seek directly to this array child.
    Indexed(usize),
    /// Conservative materialised fallback.
    MaterializedFallback,
}

/// Describes whether a view-pipeline stage reads the input `ValueView` or only acts on position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ViewInputMode {
    /// The stage examines the view's fields or scalar value.
    ReadsView,
    /// The stage ignores view content and acts on position alone.
    SkipsViewRead,
}

/// Describes whether a view-pipeline stage's output is the same view, a sub-view, or an owned value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ViewOutputMode {
    /// The stage passes the same input view through unchanged (e.g. `Filter`).
    PreservesInputView,
    /// The stage yields a single borrowed sub-view of the input (e.g. `Map` on a field).
    BorrowedSubview,
    /// The stage yields multiple borrowed sub-views (e.g. `FlatMap`).
    BorrowedSubviews,
    /// The stage produces a new owned `Val` that cannot be represented as a borrowed view.
    EmitsOwnedValue,
}

/// When, if ever, a view-pipeline stage or sink must materialise elements into owned `Val`s.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ViewMaterialization {
    /// No materialisation is needed; the stage/sink can operate entirely on borrowed views.
    Never,
    /// The stage must materialise the final value it emits (e.g. keyed reduce output).
    StageFinalValue,
    /// The sink materialises each output row into the result array (e.g. `Collect`).
    SinkOutputRows,
    /// The sink materialises only the single selected row (e.g. `first` / `last`).
    SinkFinalRow,
    /// The sink materialises each element's numeric input for folding (e.g. `sum`).
    SinkNumericInput,
}

/// Full capability descriptor for a `PipelineBody`: per-stage entries plus the sink capability.
#[derive(Debug, Clone)]
pub(crate) struct ViewPipelineCapabilities {
    /// Per-stage capabilities, parallel to `PipelineBody::stages`.
    pub stages: Vec<ViewStageCapability>,
    /// Sink capability describing how and when elements are materialised.
    pub sink: ViewSinkCapability,
}

/// Capability descriptor for the view-native prefix of a `PipelineBody` up to the first incompatible stage.
#[derive(Debug, Clone)]
pub(crate) struct ViewPrefixCapabilities {
    /// View-native stage capabilities for the prefix portion.
    pub stages: Vec<ViewStageCapability>,
    /// The number of stages from the body that are consumed by this prefix.
    pub consumed_stages: usize,
}

/// Per-stage capability for the view execution path; each variant carries a kernel index into `stage_kernels`.
#[derive(Debug, Clone, Copy)]
pub(crate) enum ViewStageCapability {
    /// Filter stage: evaluates the view-native predicate at `kernel`, keeping matching views.
    Filter {
        /// Index into `stage_kernels` for the predicate kernel.
        kernel: usize,
    },
    /// Map stage: evaluates the view-native projection at `kernel`, yielding a sub-view.
    Map {
        /// Index into `stage_kernels` for the projection kernel.
        kernel: usize,
    },
    /// FlatMap stage: evaluates the view-native body at `kernel`, yielding multiple sub-views.
    FlatMap {
        /// Index into `stage_kernels` for the body kernel.
        kernel: usize,
    },
    /// TakeWhile stage: passes views while the predicate at `kernel` is truthy.
    TakeWhile {
        /// Index into `stage_kernels` for the predicate kernel.
        kernel: usize,
    },
    /// DropWhile stage: skips views while the predicate at `kernel` is truthy.
    DropWhile {
        /// Index into `stage_kernels` for the predicate kernel.
        kernel: usize,
    },
    /// Deduplicate stage; `kernel` is `Some` when deduplication uses a view-native key program.
    Distinct {
        /// Optional index into `stage_kernels` for the key kernel.
        kernel: Option<usize>,
    },
    /// Keyed-reduce stage (e.g. `group_by`, `count_by`); uses the view-native key kernel.
    KeyedReduce {
        /// The kind of keyed reduction to perform.
        kind: BuiltinKeyedReducer,
        /// Index into `stage_kernels` for the key kernel.
        kernel: usize,
    },
    /// Take the first `n` elements without reading their content.
    Take(usize),
    /// Skip the first `n` elements without reading their content.
    Skip(usize),
}

impl ViewStageCapability {
    /// Constructs a `ViewStageCapability` from `BuiltinViewStage` metadata; returns `None` when incompatible.
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

    /// Returns the `BuiltinViewStage` tag that corresponds to this capability variant.
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

    /// Returns whether this stage reads the input view or only acts on position.
    pub(crate) fn input_mode(self) -> ViewInputMode {
        view_input_mode(self.view_stage().input_mode())
    }

    /// Returns how this stage's output relates to the input view (same view, sub-view, or owned).
    pub(crate) fn output_mode(self) -> ViewOutputMode {
        view_output_mode(self.view_stage().output_mode())
    }

    /// Returns when (if ever) this stage must materialise an element into an owned `Val`.
    pub(crate) fn materialization(self) -> ViewMaterialization {
        if matches!(self, Self::KeyedReduce { .. }) {
            return ViewMaterialization::StageFinalValue;
        }
        ViewMaterialization::Never
    }
}

/// Describes how a pipeline sink interacts with the view domain.
#[derive(Debug, Clone, Copy)]
pub(crate) enum ViewSinkCapability {
    /// The sink collects all views, materialising each row into the output array.
    Collect,
    /// A built-in accumulator sink (count, numeric reducer, first/last selector).
    Builtin {
        /// The kind of accumulation performed by this sink.
        accumulator: BuiltinSinkAccumulator,
        /// Index of the view-native predicate kernel in `sink_kernels`, if any.
        predicate_kernel: Option<usize>,
        /// Index of the view-native projection kernel in `sink_kernels`, if any.
        project_kernel: Option<usize>,
        /// When the sink must materialise element values.
        materialization: ViewMaterialization,
    },
    /// Positional nth selector with a runtime index.
    Nth {
        /// Zero-based output index selected by the sink.
        index: usize,
    },
}

impl ViewSinkCapability {
    /// Constructs a `Builtin` view sink capability from a `BuiltinSinkSpec` and optional kernel indices.
    pub(crate) fn from_sink_spec(
        spec: BuiltinSinkSpec,
        predicate_kernel: Option<usize>,
        project_kernel: Option<usize>,
    ) -> Self {
        Self::Builtin {
            accumulator: spec.accumulator,
            predicate_kernel,
            project_kernel,
            materialization: sink_materialization(spec),
        }
    }

    /// Returns when this sink must materialise element values from the view domain.
    pub(crate) fn materialization(self) -> ViewMaterialization {
        match self {
            Self::Collect => ViewMaterialization::SinkOutputRows,
            Self::Builtin {
                materialization, ..
            } => materialization,
            Self::Nth { .. } => ViewMaterialization::SinkFinalRow,
        }
    }
}

// maps the builtin sink accumulator kind to the materialisation policy it requires
fn sink_materialization(spec: BuiltinSinkSpec) -> ViewMaterialization {
    match spec.accumulator {
        BuiltinSinkAccumulator::Count | BuiltinSinkAccumulator::ApproxDistinct => {
            ViewMaterialization::Never
        }
        BuiltinSinkAccumulator::Numeric => ViewMaterialization::SinkNumericInput,
        BuiltinSinkAccumulator::SelectOne(_) => ViewMaterialization::SinkFinalRow,
    }
}

// bridges the registry's BuiltinViewInputMode tag to the pipeline's enum
fn view_input_mode(mode: BuiltinViewInputMode) -> ViewInputMode {
    match mode {
        BuiltinViewInputMode::ReadsView => ViewInputMode::ReadsView,
        BuiltinViewInputMode::SkipsViewRead => ViewInputMode::SkipsViewRead,
    }
}

// bridges the registry's BuiltinViewOutputMode tag to the pipeline's enum
fn view_output_mode(mode: BuiltinViewOutputMode) -> ViewOutputMode {
    match mode {
        BuiltinViewOutputMode::PreservesInputView => ViewOutputMode::PreservesInputView,
        BuiltinViewOutputMode::BorrowedSubview => ViewOutputMode::BorrowedSubview,
        BuiltinViewOutputMode::BorrowedSubviews => ViewOutputMode::BorrowedSubviews,
        BuiltinViewOutputMode::EmitsOwnedValue => ViewOutputMode::EmitsOwnedValue,
    }
}

/// Computes `ViewPipelineCapabilities` for `body`; returns `None` when any stage or the sink is incompatible.
pub(crate) fn view_capabilities(body: &PipelineBody) -> Option<ViewPipelineCapabilities> {
    Some(ViewPipelineCapabilities {
        stages: view_stage_capabilities(body)?,
        sink: view_sink_capability(body)?,
    })
}

/// Computes the longest view-native stage prefix of `body`; returns `None` when even the first stage is incompatible.
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

    use crate::parse::ast::BinOp;
    use crate::builtins::{
        BuiltinMethod, BuiltinSelectionPosition, BuiltinSinkAccumulator, BuiltinViewStage,
    };
    use crate::exec::pipeline::{
        BodyKernel, NumOp, PipelineBody, ReducerOp, ReducerSpec, Sink, Stage, ViewInputMode,
        ViewMaterialization, ViewOutputMode, ViewSinkCapability, ViewStageCapability,
    };
    use crate::data::value::Val;

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
        let take = Stage::UsizeBuiltin {
            method: BuiltinMethod::Take,
            value: 2,
        }
        .view_capability(7, None)
        .unwrap();
        let skip = Stage::UsizeBuiltin {
            method: BuiltinMethod::Skip,
            value: 1,
        }
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
                materialization: ViewMaterialization::Never,
            })
        ));
        assert!(matches!(
            Sink::Terminal(BuiltinMethod::First).view_capability(&[]),
            Some(ViewSinkCapability::Builtin {
                accumulator: BuiltinSinkAccumulator::SelectOne(BuiltinSelectionPosition::First),
                predicate_kernel: None,
                project_kernel: None,
                materialization: ViewMaterialization::SinkFinalRow,
            })
        ));
        assert!(matches!(
            Sink::Terminal(BuiltinMethod::Last).view_capability(&[]),
            Some(ViewSinkCapability::Builtin {
                accumulator: BuiltinSinkAccumulator::SelectOne(BuiltinSelectionPosition::Last),
                predicate_kernel: None,
                project_kernel: None,
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
                Stage::UsizeBuiltin {
                    method: BuiltinMethod::Take,
                    value: 2,
                },
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
                Stage::Builtin(crate::exec::pipeline::PipelineBuiltinCall {
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

// short-circuits on the first incompatible stage, returning None rather than a partial list
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
