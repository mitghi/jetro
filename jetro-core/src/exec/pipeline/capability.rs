//! View-pipeline capability descriptors for stages and sinks.
//!
//! Defines the borrowing, materialisation, and input/output mode traits that let
//! the view execution path decide, per stage, whether it can operate on borrowed
//! `ValueView` slices or must materialise rows into owned `Val`s.

use crate::builtins::{
    BuiltinCardinality, BuiltinKeyedReducer, BuiltinSinkAccumulator, BuiltinSinkSpec,
    BuiltinViewInputMode, BuiltinViewOutputMode, BuiltinViewStage,
};
use crate::data::value::Val;
use crate::plan::demand::{FieldDemand, PullDemand};
use crate::vm::Program;

use super::{MembershipSinkOp, MembershipSinkTarget, PipelineBody, PredicateSinkOp, Stage};

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
    /// Source can read object fields by key without materialising the whole object row.
    pub field_key_read: bool,
    /// Source can skip unneeded nested subtrees while scanning.
    pub subtree_skip: bool,
    /// Source can materialise only rows selected by the sink/predicate path.
    pub selected_row_materialization: bool,
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
        field_key_read: true,
        subtree_skip: true,
        selected_row_materialization: true,
        materialized_fallback: true,
    };

    /// Capabilities for an already materialised `Val` array source.
    pub(crate) const MATERIALIZED_ARRAY: Self = Self {
        forward_stream: true,
        reverse_stream: true,
        indexed_array_child: true,
        tape_view: false,
        field_key_read: true,
        subtree_skip: false,
        selected_row_materialization: true,
        materialized_fallback: true,
    };

    /// Chooses the most direct access mode that satisfies `demand`.
    pub(crate) fn choose_access(self, demand: PullDemand) -> SourceAccessMode {
        match demand {
            PullDemand::NthInput(idx) if self.indexed_array_child => SourceAccessMode::Indexed(idx),
            PullDemand::LastInput(1) if self.indexed_array_child => {
                SourceAccessMode::IndexedFromEnd(0)
            }
            PullDemand::LastInput(n) if self.indexed_array_child => {
                SourceAccessMode::IndexedSuffix(n)
            }
            PullDemand::FirstInput(1) if self.indexed_array_child => SourceAccessMode::Indexed(0),
            PullDemand::LastInput(n) if self.reverse_stream => {
                SourceAccessMode::Reverse { outputs: n }
            }
            PullDemand::FirstInput(n) if self.forward_stream => SourceAccessMode::ForwardBounded(n),
            _ if self.forward_stream => SourceAccessMode::Forward,
            _ => SourceAccessMode::MaterializedFallback,
        }
    }

    /// Chooses source access for a full pipeline, demoting direct positional
    /// seeks when the stage prefix changes cardinality and physical source
    /// positions no longer equal semantic output positions.
    pub(crate) fn choose_stage_access(
        self,
        demand: PullDemand,
        stages: &[Stage],
    ) -> SourceAccessMode {
        let access = self.choose_access(demand);
        if direct_seek_requires_cardinality_preservation(access)
            && !stages.iter().all(Stage::preserves_cardinality)
        {
            return demote_direct_seek(self, access);
        }
        access
    }

    /// Chooses source access for a view prefix, demoting direct seeks when the
    /// prefix can change cardinality and physical positions no longer match
    /// semantic output positions.
    pub(crate) fn choose_view_access(
        self,
        demand: PullDemand,
        stages: &[ViewStageCapability],
    ) -> SourceAccessMode {
        let access = self.choose_access(demand);
        if direct_seek_requires_cardinality_preservation(access)
            && !ViewStageCapability::all_preserve_cardinality(stages)
        {
            return demote_direct_seek(self, access);
        }
        access
    }

    /// Returns true when this source can satisfy split payload lanes without
    /// materialising every row as a full owned value.
    pub(crate) fn supports_payload_lanes(
        self,
        scan_need: &FieldDemand,
        result_need: &FieldDemand,
    ) -> bool {
        payload_lane_supported(scan_need, self.field_key_read, self.subtree_skip)
            && payload_lane_supported(
                result_need,
                self.field_key_read,
                self.selected_row_materialization,
            )
    }

    /// Returns true when this source can defer owned materialization to only
    /// rows selected by bounded or positional demand.
    pub(crate) fn supports_selected_materialization(self, demand: PullDemand) -> bool {
        self.selected_row_materialization && demand.permits_selected_materialization()
    }
}

fn direct_seek_requires_cardinality_preservation(access: SourceAccessMode) -> bool {
    matches!(
        access,
        SourceAccessMode::Indexed(_)
            | SourceAccessMode::IndexedFromEnd(_)
            | SourceAccessMode::IndexedSuffix(_)
    )
}

fn demote_direct_seek(caps: SourceCapabilities, access: SourceAccessMode) -> SourceAccessMode {
    if caps.reverse_stream {
        match access {
            SourceAccessMode::IndexedFromEnd(_) => return SourceAccessMode::Reverse { outputs: 1 },
            SourceAccessMode::IndexedSuffix(outputs) => {
                return SourceAccessMode::Reverse { outputs };
            }
            _ => {}
        }
    }
    if caps.forward_stream {
        return SourceAccessMode::Forward;
    }
    SourceAccessMode::MaterializedFallback
}

fn payload_lane_supported(need: &FieldDemand, field_key_read: bool, whole_value_ok: bool) -> bool {
    match need {
        FieldDemand::None => true,
        FieldDemand::Fields(_) => field_key_read,
        FieldDemand::Whole => whole_value_ok,
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
    /// Seek directly to this array child counted from the end.
    IndexedFromEnd(usize),
    /// Seek directly to a suffix of `n` array children, preserving input order.
    IndexedSuffix(usize),
    /// Conservative materialised fallback.
    MaterializedFallback,
}

impl SourceAccessMode {
    /// Demand that should be handed to a row iterator after this access mode has been selected.
    pub(crate) fn iterator_demand(self, requested: PullDemand) -> PullDemand {
        match self {
            Self::Reverse { outputs } => PullDemand::LastInput(outputs),
            Self::ForwardBounded(inputs) => PullDemand::FirstInput(inputs),
            Self::Indexed(_) | Self::IndexedFromEnd(_) | Self::IndexedSuffix(_) => PullDemand::All,
            Self::Forward | Self::MaterializedFallback if requested.is_suffix() => PullDemand::All,
            Self::Forward | Self::MaterializedFallback => requested,
        }
    }
}

/// Return the absolute index for `offset` counted from the end of a sequence.
#[inline]
pub(crate) fn index_from_end(len: usize, offset: usize) -> Option<usize> {
    len.checked_sub(offset.checked_add(1)?)
}

impl Stage {
    fn preserves_cardinality(&self) -> bool {
        self.shape().cardinality == BuiltinCardinality::OneToOne
    }
}

#[cfg(test)]
mod source_capability_tests {
    use super::{SourceAccessMode, SourceCapabilities, ViewStageCapability};
    use crate::builtins::BuiltinViewStage;
    use crate::data::value::Val;
    use crate::exec::pipeline::Stage;
    use crate::plan::demand::{FieldDemand, FieldSet, PullDemand};
    use std::sync::Arc;

    #[test]
    fn indexed_sources_choose_direct_positional_access() {
        assert_eq!(
            SourceCapabilities::MATERIALIZED_ARRAY.choose_access(PullDemand::NthInput(3)),
            SourceAccessMode::Indexed(3)
        );
        assert_eq!(
            SourceCapabilities::MATERIALIZED_ARRAY.choose_access(PullDemand::LastInput(2)),
            SourceAccessMode::IndexedSuffix(2)
        );
        assert_eq!(
            SourceCapabilities::MATERIALIZED_ARRAY.choose_access(PullDemand::LastInput(1)),
            SourceAccessMode::IndexedFromEnd(0)
        );
        assert_eq!(
            SourceCapabilities::MATERIALIZED_ARRAY.choose_access(PullDemand::FirstInput(4)),
            SourceAccessMode::ForwardBounded(4)
        );
        assert_eq!(
            SourceCapabilities::MATERIALIZED_ARRAY.choose_access(PullDemand::FirstInput(1)),
            SourceAccessMode::Indexed(0)
        );
    }

    #[test]
    fn view_array_sources_advertise_tape_backed_access() {
        let caps = SourceCapabilities::VIEW_ARRAY;
        assert!(caps.tape_view);
        assert!(caps.forward_stream);
        assert!(caps.reverse_stream);
        assert!(caps.indexed_array_child);
        assert!(caps.materialized_fallback);
        assert!(caps.field_key_read);
        assert!(caps.subtree_skip);
        assert!(caps.selected_row_materialization);
        assert_eq!(
            caps.choose_access(PullDemand::NthInput(2)),
            SourceAccessMode::Indexed(2)
        );
        assert_eq!(
            caps.choose_access(PullDemand::LastInput(1)),
            SourceAccessMode::IndexedFromEnd(0)
        );
    }

    #[test]
    fn selective_view_prefix_demotes_indexed_last_to_reverse_scan() {
        let access = SourceCapabilities::VIEW_ARRAY.choose_view_access(
            PullDemand::LastInput(1),
            &[ViewStageCapability::Filter { kernel: 0 }],
        );

        assert_eq!(access, SourceAccessMode::Reverse { outputs: 1 });
    }

    #[test]
    fn selective_view_prefix_demotes_indexed_last_to_forward_without_reverse() {
        let caps = SourceCapabilities {
            reverse_stream: false,
            ..SourceCapabilities::VIEW_ARRAY
        };

        let access = caps.choose_view_access(
            PullDemand::LastInput(1),
            &[ViewStageCapability::RemoveValue(Val::Int(2))],
        );

        assert_eq!(access, SourceAccessMode::Forward);
    }

    #[test]
    fn selective_pipeline_prefix_demotes_direct_seek() {
        let filter = Stage::Filter(
            Arc::new(crate::vm::Program::new(Vec::new(), "")),
            BuiltinViewStage::Filter,
        );
        let map = Stage::Map(
            Arc::new(crate::vm::Program::new(Vec::new(), "")),
            BuiltinViewStage::Map,
        );

        assert_eq!(
            SourceCapabilities::VIEW_ARRAY
                .choose_stage_access(PullDemand::LastInput(1), &[filter.clone()]),
            SourceAccessMode::Reverse { outputs: 1 }
        );
        assert_eq!(
            SourceCapabilities::VIEW_ARRAY
                .choose_stage_access(PullDemand::LastInput(1), &[map.clone()]),
            SourceAccessMode::IndexedFromEnd(0)
        );
        assert_eq!(
            SourceCapabilities::VIEW_ARRAY.choose_stage_access(PullDemand::LastInput(2), &[filter]),
            SourceAccessMode::Reverse { outputs: 2 }
        );
        assert_eq!(
            SourceCapabilities::VIEW_ARRAY.choose_stage_access(PullDemand::LastInput(2), &[map]),
            SourceAccessMode::IndexedSuffix(2)
        );
    }

    #[test]
    fn cardinality_preserving_view_prefix_keeps_indexed_last_seek() {
        let access = SourceCapabilities::VIEW_ARRAY.choose_view_access(
            PullDemand::LastInput(1),
            &[ViewStageCapability::Map { kernel: 0 }],
        );

        assert_eq!(access, SourceAccessMode::IndexedFromEnd(0));
    }

    #[test]
    fn non_seekable_sources_fall_back_to_forward_streaming() {
        let forward_only = SourceCapabilities {
            forward_stream: true,
            reverse_stream: false,
            indexed_array_child: false,
            tape_view: false,
            field_key_read: false,
            subtree_skip: false,
            selected_row_materialization: false,
            materialized_fallback: true,
        };

        assert_eq!(
            forward_only.choose_access(PullDemand::NthInput(3)),
            SourceAccessMode::Forward
        );
        assert_eq!(
            forward_only.choose_access(PullDemand::LastInput(1)),
            SourceAccessMode::Forward
        );
        assert_eq!(
            forward_only.choose_access(PullDemand::FirstInput(2)),
            SourceAccessMode::ForwardBounded(2)
        );
    }

    #[test]
    fn indexed_without_reverse_still_scans_forward_for_last_demand() {
        let indexed_forward = SourceCapabilities {
            forward_stream: true,
            reverse_stream: false,
            indexed_array_child: true,
            tape_view: false,
            field_key_read: true,
            subtree_skip: false,
            selected_row_materialization: true,
            materialized_fallback: true,
        };

        assert_eq!(
            indexed_forward.choose_access(PullDemand::NthInput(5)),
            SourceAccessMode::Indexed(5)
        );
        assert_eq!(
            indexed_forward.choose_access(PullDemand::LastInput(1)),
            SourceAccessMode::IndexedFromEnd(0)
        );
    }

    #[test]
    fn indexed_only_sources_seek_single_positional_demands_and_materialize_ranges() {
        let indexed_only = SourceCapabilities {
            forward_stream: false,
            reverse_stream: false,
            indexed_array_child: true,
            tape_view: true,
            field_key_read: true,
            subtree_skip: true,
            selected_row_materialization: true,
            materialized_fallback: true,
        };

        assert_eq!(
            indexed_only.choose_access(PullDemand::NthInput(7)),
            SourceAccessMode::Indexed(7)
        );
        assert_eq!(
            indexed_only.choose_access(PullDemand::FirstInput(1)),
            SourceAccessMode::Indexed(0)
        );
        assert_eq!(
            indexed_only.choose_access(PullDemand::FirstInput(2)),
            SourceAccessMode::MaterializedFallback
        );
        assert_eq!(
            indexed_only.choose_access(PullDemand::LastInput(1)),
            SourceAccessMode::IndexedFromEnd(0)
        );
    }

    #[test]
    fn non_streaming_sources_request_materialized_fallback() {
        let fallback_only = SourceCapabilities {
            forward_stream: false,
            reverse_stream: false,
            indexed_array_child: false,
            tape_view: false,
            field_key_read: false,
            subtree_skip: false,
            selected_row_materialization: false,
            materialized_fallback: true,
        };

        assert_eq!(
            fallback_only.choose_access(PullDemand::All),
            SourceAccessMode::MaterializedFallback
        );
    }

    #[test]
    fn access_mode_rewrites_iterator_demand_after_fallback_choice() {
        assert_eq!(
            SourceAccessMode::Reverse { outputs: 2 }.iterator_demand(PullDemand::LastInput(10)),
            PullDemand::LastInput(2)
        );
        assert_eq!(
            SourceAccessMode::ForwardBounded(3).iterator_demand(PullDemand::All),
            PullDemand::FirstInput(3)
        );
        assert_eq!(
            SourceAccessMode::Indexed(4).iterator_demand(PullDemand::NthInput(4)),
            PullDemand::All
        );
        assert_eq!(
            SourceAccessMode::Forward.iterator_demand(PullDemand::LastInput(1)),
            PullDemand::All
        );
    }

    #[test]
    fn payload_lanes_require_matching_source_capabilities() {
        let fields = FieldDemand::Fields(FieldSet::single(Arc::from("price")));
        assert!(SourceCapabilities::VIEW_ARRAY.supports_payload_lanes(&fields, &fields));
        assert!(SourceCapabilities::MATERIALIZED_ARRAY
            .supports_payload_lanes(&fields, &FieldDemand::Whole));
        assert!(!SourceCapabilities::MATERIALIZED_ARRAY
            .supports_payload_lanes(&FieldDemand::Whole, &fields));
    }

    #[test]
    fn selected_materialization_requires_bounded_demand() {
        assert!(SourceCapabilities::VIEW_ARRAY
            .supports_selected_materialization(PullDemand::LastInput(1)));
        assert!(SourceCapabilities::MATERIALIZED_ARRAY
            .supports_selected_materialization(PullDemand::UntilOutput(3)));
        assert!(!SourceCapabilities::VIEW_ARRAY.supports_selected_materialization(PullDemand::All));

        let no_selected = SourceCapabilities {
            selected_row_materialization: false,
            ..SourceCapabilities::MATERIALIZED_ARRAY
        };
        assert!(!no_selected.supports_selected_materialization(PullDemand::FirstInput(1)));
    }
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
    /// The sink materialises input rows for its own comparison/state, not for output.
    SinkInputRows,
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
#[derive(Debug, Clone)]
pub(crate) enum ViewStageCapability {
    /// Filter stage: evaluates the view-native predicate at `kernel`, keeping matching views.
    Filter {
        /// Index into `stage_kernels` for the predicate kernel.
        kernel: usize,
    },
    /// Compact stage: keeps non-null views.
    Compact,
    /// Remove stage: drops views equal to a literal target.
    RemoveValue(Val),
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
            BuiltinViewStage::Compact => Some(Self::Compact),
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
    pub(crate) fn view_stage(&self) -> BuiltinViewStage {
        match self {
            Self::Filter { .. } => BuiltinViewStage::Filter,
            Self::Compact => BuiltinViewStage::Compact,
            Self::RemoveValue(_) => BuiltinViewStage::RemoveValue,
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
    pub(crate) fn input_mode(&self) -> ViewInputMode {
        view_input_mode(self.view_stage().input_mode())
    }

    /// Returns how this stage's output relates to the input view (same view, sub-view, or owned).
    pub(crate) fn output_mode(&self) -> ViewOutputMode {
        view_output_mode(self.view_stage().output_mode())
    }

    /// Returns when (if ever) this stage must materialise an element into an owned `Val`.
    pub(crate) fn materialization(&self) -> ViewMaterialization {
        if matches!(self, Self::KeyedReduce { .. }) {
            return ViewMaterialization::StageFinalValue;
        }
        ViewMaterialization::Never
    }

    /// Returns true when this view stage emits exactly one row for every input row.
    pub(crate) fn preserves_cardinality(&self) -> bool {
        self.view_stage().cardinality() == BuiltinCardinality::OneToOne
    }

    /// Returns true when every stage in a prefix preserves input/output cardinality.
    pub(crate) fn all_preserve_cardinality(stages: &[Self]) -> bool {
        stages.iter().all(Self::preserves_cardinality)
    }
}

/// Describes how a pipeline sink interacts with the view domain.
#[derive(Debug, Clone)]
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
    /// Predicate terminal sink (`any`, `all`, `find_index`, `indices_where`, `find_one`).
    Predicate {
        /// Predicate terminal operation to perform.
        op: PredicateSinkOp,
        /// Index of the view-native predicate kernel in `sink_kernels`.
        predicate_kernel: usize,
    },
    /// Literal value-membership terminal sink (`includes`, `index`, `indices_of`).
    Membership {
        /// Membership terminal operation to perform.
        op: MembershipSinkOp,
        /// Target compared against each row.
        target: ViewMembershipTarget,
    },
    /// Arg-extreme terminal sink (`max_by`, `min_by`).
    ArgExtreme {
        /// When true, keeps the row with the largest key; otherwise the smallest key.
        want_max: bool,
        /// Index of the view-native key kernel in `sink_kernels`.
        key_kernel: usize,
    },
    /// Bounded positional selector for terminal `first(n)` / `last(n)`.
    SelectMany {
        /// Number of rows requested by the terminal sink.
        n: usize,
        /// Whether the semantic selector wants rows from the end.
        from_end: bool,
        /// Whether the source iterator is running in reverse physical order.
        source_reversed: bool,
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
    pub(crate) fn materialization(&self) -> ViewMaterialization {
        match self {
            Self::Collect => ViewMaterialization::SinkOutputRows,
            Self::Builtin {
                materialization, ..
            } => *materialization,
            Self::Nth { .. } => ViewMaterialization::SinkFinalRow,
            Self::Predicate { op, .. } => {
                if op.is_find_one() {
                    ViewMaterialization::SinkFinalRow
                } else {
                    ViewMaterialization::Never
                }
            }
            Self::Membership { target, .. } => {
                if target.is_scalar_literal() {
                    ViewMaterialization::Never
                } else {
                    ViewMaterialization::SinkInputRows
                }
            }
            Self::ArgExtreme { .. } => ViewMaterialization::SinkFinalRow,
            Self::SelectMany { .. } => ViewMaterialization::SinkOutputRows,
        }
    }
}

/// Target for a view-native membership terminal.
#[derive(Debug, Clone)]
pub(crate) enum ViewMembershipTarget {
    /// Literal known during lowering.
    Literal(Val),
    /// Expression evaluated once against the outer environment before row streaming.
    Program(std::sync::Arc<Program>),
}

impl ViewMembershipTarget {
    fn is_scalar_literal(&self) -> bool {
        match self {
            Self::Literal(value) => target_is_scalar(value),
            Self::Program(_) => false,
        }
    }
}

impl From<&MembershipSinkTarget> for ViewMembershipTarget {
    fn from(target: &MembershipSinkTarget) -> Self {
        match target {
            MembershipSinkTarget::Literal(value) => Self::Literal(value.clone()),
            MembershipSinkTarget::Program(program) => Self::Program(std::sync::Arc::clone(program)),
        }
    }
}

fn target_is_scalar(value: &Val) -> bool {
    matches!(
        value,
        Val::Null | Val::Bool(_) | Val::Int(_) | Val::Float(_) | Val::Str(_) | Val::StrSlice(_)
    )
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

    use crate::builtins::{
        BuiltinMethod, BuiltinSelectionPosition, BuiltinSinkAccumulator, BuiltinViewStage,
    };
    use crate::data::value::Val;
    use crate::exec::pipeline::{
        ArgExtremeSinkSpec, BodyKernel, MembershipSinkOp, MembershipSinkSpec, MembershipSinkTarget,
        NumOp, PipelineBody, PredicateSinkOp, PredicateSinkSpec, ReducerOp, ReducerSpec, Sink,
        Stage, ViewInputMode, ViewMaterialization, ViewMembershipTarget, ViewOutputMode,
        ViewSinkCapability, ViewStageCapability,
    };
    use crate::parse::ast::BinOp;

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
        assert!(!flat_map.preserves_cardinality());

        let remove = ViewStageCapability::RemoveValue(Val::Int(2));
        assert_eq!(remove.input_mode(), ViewInputMode::ReadsView);
        assert_eq!(remove.output_mode(), ViewOutputMode::PreservesInputView);
        assert_eq!(remove.materialization(), ViewMaterialization::Never);
        assert!(!remove.preserves_cardinality());

        let take = ViewStageCapability::Take(2);
        assert_eq!(take.input_mode(), ViewInputMode::SkipsViewRead);
        assert_eq!(take.output_mode(), ViewOutputMode::PreservesInputView);
        assert_eq!(take.materialization(), ViewMaterialization::Never);
        assert!(!take.preserves_cardinality());

        assert!(map.preserves_cardinality());
        assert!(!filter.preserves_cardinality());
        assert!(!ViewStageCapability::Compact.preserves_cardinality());
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
        let compact = Stage::Builtin(crate::builtins::BuiltinCall::new(
            BuiltinMethod::Compact,
            crate::builtins::BuiltinArgs::None,
        ))
        .view_capability(9, None)
        .unwrap();
        let remove = Stage::Builtin(crate::builtins::BuiltinCall::new(
            BuiltinMethod::Remove,
            crate::builtins::BuiltinArgs::Val(Val::Int(2)),
        ))
        .view_capability(10, None)
        .unwrap();

        assert!(matches!(filter, ViewStageCapability::Filter { kernel: 4 }));
        assert_eq!(map.output_mode(), ViewOutputMode::BorrowedSubview);
        assert_eq!(flat_map.output_mode(), ViewOutputMode::BorrowedSubviews);
        assert!(matches!(take, ViewStageCapability::Take(2)));
        assert!(matches!(skip, ViewStageCapability::Skip(1)));
        assert!(matches!(compact, ViewStageCapability::Compact));
        assert!(matches!(
            remove,
            ViewStageCapability::RemoveValue(Val::Int(2))
        ));
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
        assert_eq!(
            ViewSinkCapability::Predicate {
                op: PredicateSinkOp::Any,
                predicate_kernel: 0,
            }
            .materialization(),
            ViewMaterialization::Never
        );
        assert_eq!(
            ViewSinkCapability::Predicate {
                op: PredicateSinkOp::FindOne,
                predicate_kernel: 0,
            }
            .materialization(),
            ViewMaterialization::SinkFinalRow
        );
        assert_eq!(
            ViewSinkCapability::Membership {
                op: MembershipSinkOp::Includes,
                target: ViewMembershipTarget::Literal(Val::Int(3)),
            }
            .materialization(),
            ViewMaterialization::Never
        );
        assert_eq!(
            ViewSinkCapability::Membership {
                op: MembershipSinkOp::Includes,
                target: ViewMembershipTarget::Literal(Val::arr(vec![Val::Int(3)])),
            }
            .materialization(),
            ViewMaterialization::SinkInputRows
        );
        assert_eq!(
            ViewSinkCapability::ArgExtreme {
                want_max: true,
                key_kernel: 0,
            }
            .materialization(),
            ViewMaterialization::SinkFinalRow
        );
        assert_eq!(
            ViewSinkCapability::SelectMany {
                n: 2,
                from_end: true,
                source_reversed: true,
            }
            .materialization(),
            ViewMaterialization::SinkOutputRows
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
        assert!(matches!(
            Sink::Predicate(PredicateSinkSpec {
                op: PredicateSinkOp::Any,
                predicate: Arc::new(crate::vm::Program::new(Vec::new(), "")),
                predicate_expr: None,
            })
            .view_capability(&[BodyKernel::FieldCmpLit(
                Arc::from("score"),
                BinOp::Gt,
                Val::Int(10),
            )]),
            Some(ViewSinkCapability::Predicate {
                op: PredicateSinkOp::Any,
                predicate_kernel: 0,
            })
        ));
        assert!(matches!(
            Sink::SelectMany {
                n: 2,
                from_end: true,
            }
            .view_capability(&[]),
            Some(ViewSinkCapability::SelectMany {
                n: 2,
                from_end: true,
                source_reversed: false,
            })
        ));
        assert!(matches!(
            Sink::Membership(MembershipSinkSpec {
                op: MembershipSinkOp::Includes,
                target: MembershipSinkTarget::Literal(Val::Int(3)),
            })
            .view_capability(&[]),
            Some(ViewSinkCapability::Membership {
                op: MembershipSinkOp::Includes,
                target: ViewMembershipTarget::Literal(Val::Int(3)),
            })
        ));
        assert!(matches!(
            Sink::Membership(MembershipSinkSpec {
                op: MembershipSinkOp::Includes,
                target: MembershipSinkTarget::Program(Arc::new(crate::vm::Program::new(
                    Vec::new(),
                    ""
                ))),
            })
            .view_capability(&[]),
            Some(ViewSinkCapability::Membership {
                op: MembershipSinkOp::Includes,
                target: ViewMembershipTarget::Program(_),
            })
        ));
        assert!(matches!(
            Sink::ArgExtreme(ArgExtremeSinkSpec {
                want_max: true,
                key: Arc::new(crate::vm::Program::new(Vec::new(), "")),
                key_expr: None,
            })
            .view_capability(&[BodyKernel::FieldRead(Arc::from("score"))]),
            Some(ViewSinkCapability::ArgExtreme {
                want_max: true,
                key_kernel: 0,
            })
        ));
        assert!(Sink::ArgExtreme(ArgExtremeSinkSpec {
            want_max: false,
            key: Arc::new(crate::vm::Program::new(Vec::new(), "")),
            key_expr: None,
        })
        .view_capability(&[BodyKernel::Generic])
        .is_none());
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

    #[test]
    fn view_prefix_keeps_remove_value_before_materializing_stage() {
        let body = PipelineBody {
            stages: vec![
                Stage::Map(
                    Arc::new(crate::vm::Program::new(Vec::new(), "")),
                    BuiltinViewStage::Map,
                ),
                Stage::Builtin(crate::exec::pipeline::PipelineBuiltinCall {
                    method: crate::builtins::BuiltinMethod::Remove,
                    args: crate::builtins::BuiltinArgs::Val(Val::Int(2)),
                }),
                Stage::Builtin(crate::exec::pipeline::PipelineBuiltinCall {
                    method: crate::builtins::BuiltinMethod::Upper,
                    args: crate::builtins::BuiltinArgs::None,
                }),
            ],
            stage_exprs: Vec::new(),
            sink: Sink::Collect,
            stage_kernels: vec![BodyKernel::FieldRead(Arc::from("id"))],
            sink_kernels: Vec::new(),
        };

        let prefix = view_prefix_capabilities(&body).unwrap();
        assert_eq!(prefix.consumed_stages, 2);
        assert!(matches!(
            prefix.stages[0],
            ViewStageCapability::Map { kernel: 0 }
        ));
        assert!(matches!(
            prefix.stages[1],
            ViewStageCapability::RemoveValue(Val::Int(2))
        ));
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
