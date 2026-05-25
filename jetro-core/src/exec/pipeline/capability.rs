//! View-pipeline capability descriptors for stages and sinks.
//!
//! Defines the borrowing, materialisation, and input/output mode traits that let
//! the view execution path decide, per stage, whether it can operate on borrowed
//! `ValueView` slices or must materialise rows into owned `Val`s.

use std::borrow::Cow;

use crate::builtins::registry::BuiltinId;
pub(crate) use crate::builtins::BuiltinViewInputMode as ViewInputMode;
pub(crate) use crate::builtins::BuiltinViewMaterialization as ViewMaterialization;
pub(crate) use crate::builtins::BuiltinViewOutputMode as ViewOutputMode;
use crate::builtins::{
    BuiltinArgExtremeSink, BuiltinArgs, BuiltinKeyedReducer, BuiltinMembershipSink,
    BuiltinPredicateSink, BuiltinSinkAccumulator, BuiltinSinkSpec, BuiltinViewStage,
    BuiltinViewStringExpand,
};
use crate::data::value::Val;
use crate::plan::demand::{FieldDemand, PullDemand};
use crate::vm::Program;

use super::{BodyKernel, MembershipSinkTarget, PipelineBody, Stage};

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

/// Concrete indexed source read derived from a selected source access mode and
/// a known source length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SourceIndexedAccess {
    /// The selected indexed access is in range and targets a single row.
    Single(usize),
    /// The selected indexed access is in range and targets a half-open row range.
    Range {
        /// First row index to read.
        start: usize,
        /// One-past-the-last row index to read.
        end: usize,
    },
    /// The selected indexed access is valid but cannot yield any row for this length.
    Empty,
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

    /// Converts direct indexed access modes into concrete source indexes for a
    /// source with `len` rows. Non-indexed traversal modes return `None`.
    pub(crate) fn indexed_access(self, len: usize) -> Option<SourceIndexedAccess> {
        match self {
            Self::Indexed(idx) => Some(if idx < len {
                SourceIndexedAccess::Single(idx)
            } else {
                SourceIndexedAccess::Empty
            }),
            Self::IndexedFromEnd(offset) => Some(
                index_from_end(len, offset)
                    .map(SourceIndexedAccess::Single)
                    .unwrap_or(SourceIndexedAccess::Empty),
            ),
            Self::IndexedSuffix(count) => {
                let take = count.min(len);
                let start = len - take;
                Some(SourceIndexedAccess::Range { start, end: len })
            }
            Self::Forward
            | Self::ForwardBounded(_)
            | Self::Reverse { .. }
            | Self::MaterializedFallback => None,
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
        self.shape().is_one_to_one()
    }
}

#[cfg(test)]
mod source_capability_tests {
    use super::{SourceAccessMode, SourceCapabilities, SourceIndexedAccess, ViewStageCapability};
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
    fn access_mode_resolves_direct_indexed_reads() {
        assert_eq!(
            SourceAccessMode::Indexed(2).indexed_access(5),
            Some(SourceIndexedAccess::Single(2))
        );
        assert_eq!(
            SourceAccessMode::Indexed(5).indexed_access(5),
            Some(SourceIndexedAccess::Empty)
        );
        assert_eq!(
            SourceAccessMode::IndexedFromEnd(0).indexed_access(5),
            Some(SourceIndexedAccess::Single(4))
        );
        assert_eq!(
            SourceAccessMode::IndexedFromEnd(5).indexed_access(5),
            Some(SourceIndexedAccess::Empty)
        );
        assert_eq!(
            SourceAccessMode::IndexedSuffix(3).indexed_access(5),
            Some(SourceIndexedAccess::Range { start: 2, end: 5 })
        );
        assert_eq!(
            SourceAccessMode::IndexedSuffix(8).indexed_access(5),
            Some(SourceIndexedAccess::Range { start: 0, end: 5 })
        );
        assert_eq!(SourceAccessMode::Forward.indexed_access(5), None);
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
    /// Direct registry-declared borrowed-view builtin projection.
    BuiltinProjection {
        /// Builtin registry id.
        id: BuiltinId,
        /// Decoded static builtin arguments.
        args: BuiltinArgs,
    },
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
    /// Expand a string row into owned rows without materialising the receiver.
    StringExpand {
        /// Expansion operation declared by builtin metadata.
        op: BuiltinViewStringExpand,
        /// Optional static string argument, used by `split`.
        arg: Option<std::sync::Arc<str>>,
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

/// Deterministic source-access effect of a view stage when its predicate kernel
/// is compile-time constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ViewStageConstantEffect {
    /// Keep the stage in the source-access prefix.
    Keep,
    /// Drop the stage because it accepts every row.
    NoOp,
    /// Replace the suffix with an empty stream.
    Empty,
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
            Self::BuiltinProjection { .. } => BuiltinViewStage::Map,
            Self::Filter { .. } => BuiltinViewStage::Filter,
            Self::Compact => BuiltinViewStage::Compact,
            Self::RemoveValue(_) => BuiltinViewStage::RemoveValue,
            Self::Map { .. } => BuiltinViewStage::Map,
            Self::FlatMap { .. } => BuiltinViewStage::FlatMap,
            Self::StringExpand { .. } => BuiltinViewStage::FlatMap,
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
        self.view_stage().input_mode()
    }

    /// Returns how this stage's output relates to the input view (same view, sub-view, or owned).
    pub(crate) fn output_mode(&self) -> ViewOutputMode {
        match self {
            Self::StringExpand { .. } => ViewOutputMode::EmitsOwnedValue,
            _ => self.view_stage().output_mode(),
        }
    }

    /// Returns when (if ever) this stage must materialise an element into an owned `Val`.
    pub(crate) fn materialization(&self) -> ViewMaterialization {
        self.view_stage().materialization()
    }

    /// Returns true when this view stage emits exactly one row for every input row.
    pub(crate) fn preserves_cardinality(&self) -> bool {
        !matches!(self, Self::StringExpand { .. }) && self.view_stage().preserves_cardinality()
    }

    /// Returns true when every stage in a prefix preserves input/output cardinality.
    pub(crate) fn all_preserve_cardinality(stages: &[Self]) -> bool {
        stages.iter().all(Self::preserves_cardinality)
    }

    /// Returns the deterministic source-access effect for stages with
    /// compile-time constant predicate kernels.
    pub(crate) fn constant_access_effect(
        &self,
        stage_kernels: &[BodyKernel],
    ) -> ViewStageConstantEffect {
        match self {
            Self::Filter { kernel } | Self::TakeWhile { kernel } => {
                match stage_kernels
                    .get(*kernel)
                    .and_then(BodyKernel::constant_truthy)
                {
                    Some(true) => ViewStageConstantEffect::NoOp,
                    Some(false) => ViewStageConstantEffect::Empty,
                    None => ViewStageConstantEffect::Keep,
                }
            }
            Self::DropWhile { kernel } => {
                match stage_kernels
                    .get(*kernel)
                    .and_then(BodyKernel::constant_truthy)
                {
                    Some(true) => ViewStageConstantEffect::Empty,
                    Some(false) => ViewStageConstantEffect::NoOp,
                    None => ViewStageConstantEffect::Keep,
                }
            }
            _ => ViewStageConstantEffect::Keep,
        }
    }

    /// Applies this stage's deterministic cardinality effect to `count`.
    /// Returns `None` when the stage can change cardinality in a data-dependent way.
    pub(crate) fn deterministic_cardinality_after(
        &self,
        stage_kernels: &[BodyKernel],
        count: usize,
    ) -> Option<usize> {
        match self {
            Self::BuiltinProjection { .. } | Self::Map { .. } => Some(count),
            Self::Take(n) => Some(count.min(*n)),
            Self::Skip(n) => Some(count.saturating_sub(*n)),
            Self::Filter { kernel } | Self::TakeWhile { kernel } => {
                let keep = stage_kernels.get(*kernel)?.constant_truthy()?;
                Some(if keep { count } else { 0 })
            }
            Self::DropWhile { kernel } => {
                let drop_all = stage_kernels.get(*kernel)?.constant_truthy()?;
                Some(if drop_all { 0 } else { count })
            }
            Self::Compact
            | Self::RemoveValue(_)
            | Self::FlatMap { .. }
            | Self::StringExpand { .. }
            | Self::Distinct { .. }
            | Self::KeyedReduce { .. } => None,
        }
    }

    /// Applies deterministic cardinality effects for a stage prefix.
    pub(crate) fn deterministic_prefix_cardinality_after(
        stages: &[Self],
        stage_kernels: &[BodyKernel],
        mut count: usize,
    ) -> Option<usize> {
        for stage in stages {
            count = stage.deterministic_cardinality_after(stage_kernels, count)?;
        }
        Some(count)
    }

    /// Returns true when deterministic prefix analysis proves no row can survive.
    pub(crate) fn prefix_forces_empty(stages: &[Self], stage_kernels: &[BodyKernel]) -> bool {
        let mut upper_bound = None::<usize>;
        for stage in stages {
            match stage {
                Self::Take(n) => {
                    if *n == 0 {
                        return true;
                    }
                    upper_bound = Some(upper_bound.map_or(*n, |bound| bound.min(*n)));
                }
                Self::Skip(n) => {
                    if upper_bound.is_some_and(|bound| *n >= bound) {
                        return true;
                    }
                    upper_bound = upper_bound.map(|bound| bound.saturating_sub(*n));
                }
                Self::Filter { .. } | Self::TakeWhile { .. } | Self::DropWhile { .. } => {
                    if matches!(
                        stage.constant_access_effect(stage_kernels),
                        ViewStageConstantEffect::Empty
                    ) {
                        return true;
                    }
                }
                Self::BuiltinProjection { .. } | Self::Map { .. } => {}
                Self::Compact
                | Self::RemoveValue(_)
                | Self::FlatMap { .. }
                | Self::StringExpand { .. }
                | Self::Distinct { .. }
                | Self::KeyedReduce { .. } => return false,
            }
        }
        false
    }

    /// Rewrites a view-stage prefix for source-access selection by removing
    /// constant no-op predicate stages and truncating at a provably-empty stage.
    pub(crate) fn source_access_stages<'a>(
        stages: &'a [Self],
        stage_kernels: &[BodyKernel],
    ) -> Cow<'a, [Self]> {
        let mut rewritten: Option<Vec<Self>> = None;
        for (idx, stage) in stages.iter().enumerate() {
            match stage.constant_access_effect(stage_kernels) {
                ViewStageConstantEffect::Keep => {
                    if let Some(out) = rewritten.as_mut() {
                        out.push(stage.clone());
                    }
                }
                ViewStageConstantEffect::NoOp => {
                    if rewritten.is_none() {
                        let mut out = Vec::with_capacity(stages.len());
                        out.extend_from_slice(&stages[..idx]);
                        rewritten = Some(out);
                    }
                }
                ViewStageConstantEffect::Empty => {
                    let mut out = rewritten.unwrap_or_else(|| {
                        let mut out = Vec::with_capacity(idx + 1);
                        out.extend_from_slice(&stages[..idx]);
                        out
                    });
                    out.push(Self::Take(0));
                    return Cow::Owned(out);
                }
            }
        }
        rewritten.map_or(Cow::Borrowed(stages), Cow::Owned)
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
        op: BuiltinPredicateSink,
        /// Index of the view-native predicate kernel in `sink_kernels`.
        predicate_kernel: usize,
    },
    /// Literal value-membership terminal sink (`includes`, `index`, `indices_of`).
    Membership {
        /// Membership terminal operation to perform.
        op: BuiltinMembershipSink,
        /// Target compared against each row.
        target: ViewMembershipTarget,
    },
    /// Arg-extreme terminal sink (`max_by`, `min_by`).
    ArgExtreme {
        /// Terminal arg-extreme operation.
        op: BuiltinArgExtremeSink,
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
            materialization: spec.view_materialization(),
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
            Self::Predicate { op, .. } => op.view_materialization(),
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

    /// Result for an empty input stream when the capability fully describes it.
    pub(crate) fn empty_stream_result(&self) -> Option<Val> {
        match self {
            Self::Collect => Some(Val::arr(Vec::new())),
            Self::Builtin { accumulator, .. } => accumulator.empty_stream_result(),
            Self::Nth { .. } | Self::ArgExtreme { .. } => Some(Val::Null),
            Self::Predicate { op, .. } => op.empty_stream_result(),
            Self::Membership { op, .. } => Some(op.empty_stream_result()),
            Self::SelectMany { n, .. } => {
                if *n <= 1 {
                    Some(Val::Null)
                } else {
                    Some(Val::arr(Vec::new()))
                }
            }
        }
    }

    /// Returns a sink adjusted for the selected source traversal demand.
    pub(crate) fn for_source_demand(self, demand: PullDemand, source_reversed: bool) -> Self {
        match (demand, self) {
            (PullDemand::NthInput(_), Self::Nth { .. }) => Self::Nth { index: 0 },
            (PullDemand::LastInput(_), Self::SelectMany { n, from_end, .. }) => Self::SelectMany {
                n,
                from_end,
                source_reversed,
            },
            (_, sink) => sink,
        }
    }

    /// Whether this sink semantically wants the last retained row(s).
    pub(crate) fn selects_from_end(&self) -> bool {
        match self {
            Self::SelectMany { from_end, .. } => *from_end,
            Self::Builtin { accumulator, .. } => accumulator
                .selection_position()
                .is_some_and(crate::builtins::BuiltinSelectionPosition::wants_last),
            _ => false,
        }
    }

    /// Returns true when a reversed source cannot use a bounded last-input
    /// pull because a selective suffix can change which semantic row is last.
    pub(crate) fn requires_full_reverse_scan_for_selective_last(
        &self,
        source_demand: PullDemand,
        source_reversed: bool,
        stages: &[ViewStageCapability],
    ) -> bool {
        source_reversed
            && matches!(source_demand, PullDemand::LastInput(_))
            && self.selects_from_end()
            && !ViewStageCapability::all_preserve_cardinality(stages)
    }

    /// Returns the optional predicate kernel for count sinks whose result can
    /// be derived from known source cardinality without row materialization.
    pub(crate) fn count_from_cardinality_predicate(&self) -> Option<Option<usize>> {
        match self {
            Self::Builtin {
                accumulator: BuiltinSinkAccumulator::Count,
                predicate_kernel,
                project_kernel: None,
                ..
            } => Some(*predicate_kernel),
            _ => None,
        }
    }

    /// Returns the predicate terminal contract for sinks whose result can be
    /// derived from known source cardinality and a compile-time constant predicate.
    pub(crate) fn constant_predicate_cardinality_contract(
        &self,
    ) -> Option<(BuiltinPredicateSink, usize)> {
        match self {
            Self::Predicate {
                op,
                predicate_kernel,
            } if !op.returns_matching_row() => Some((*op, *predicate_kernel)),
            _ => None,
        }
    }

    /// Returns a complete sink result when the executor has proven the stream
    /// cardinality without reading rows.
    pub(crate) fn result_from_known_cardinality(
        &self,
        count: usize,
        stream_forced_empty: bool,
        sink_kernels: &[BodyKernel],
    ) -> Option<Val> {
        if stream_forced_empty {
            return self.empty_stream_result();
        }

        if let Some(predicate_kernel) = self.count_from_cardinality_predicate() {
            let count = match predicate_kernel {
                Some(predicate_kernel) if count > 0 => {
                    if sink_kernels.get(predicate_kernel)?.constant_truthy()? {
                        count
                    } else {
                        0
                    }
                }
                Some(_) | None => count,
            };
            return Some(Val::Int(count as i64));
        }

        let (op, predicate_kernel) = self.constant_predicate_cardinality_contract()?;
        let matched = sink_kernels.get(predicate_kernel)?.constant_truthy()?;
        op.constant_predicate_stream_result(matched, count)
    }

    /// Returns true when it is worth asking the source for exact cardinality
    /// because this sink can potentially finish from that fact alone.
    pub(crate) fn can_finish_from_known_cardinality(&self, sink_kernels: &[BodyKernel]) -> bool {
        match self.count_from_cardinality_predicate() {
            Some(None) => return true,
            Some(Some(_)) => return true,
            None => {}
        }

        self.constant_predicate_cardinality_contract()
            .and_then(|(_, predicate_kernel)| sink_kernels.get(predicate_kernel))
            .and_then(BodyKernel::constant_truthy)
            .is_some()
    }

    /// Returns the arg-extreme sink contract when this sink selects a row by a
    /// view-native key program.
    pub(crate) fn arg_extreme_contract(&self) -> Option<(BuiltinArgExtremeSink, usize)> {
        match self {
            Self::ArgExtreme { op, key_kernel } => Some((*op, *key_kernel)),
            _ => None,
        }
    }

    /// Returns the runtime target program for membership sinks that still need
    /// one environment-level target evaluation before row streaming.
    pub(crate) fn membership_target_program(&self) -> Option<std::sync::Arc<Program>> {
        match self {
            Self::Membership {
                target: ViewMembershipTarget::Program(program),
                ..
            } => Some(std::sync::Arc::clone(program)),
            _ => None,
        }
    }

    /// Replaces a runtime membership target with the resolved literal target.
    pub(crate) fn with_resolved_membership_target(self, resolved: Val) -> Self {
        match self {
            Self::Membership {
                op,
                target: ViewMembershipTarget::Program(_),
            } => Self::Membership {
                op,
                target: ViewMembershipTarget::Literal(resolved),
            },
            sink => sink,
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
    pub(crate) fn is_scalar_literal(&self) -> bool {
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
    for idx in 0..body.stages.len() {
        let Some(capability) = view_never_materializing_stage_capability(body, idx) else {
            break;
        };
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

/// Returns a view-stage capability only when the stage can run without materializing rows.
pub(crate) fn view_never_materializing_stage_capability(
    body: &PipelineBody,
    idx: usize,
) -> Option<ViewStageCapability> {
    let capability = view_stage_capability(body, idx, body.stages.get(idx)?)?;
    matches!(capability.materialization(), ViewMaterialization::Never).then_some(capability)
}

/// Computes capabilities for the exact stage range `[start, end)` when every
/// stage is view-native and never materializes rows.
pub(crate) fn view_never_materializing_stage_range(
    body: &PipelineBody,
    start: usize,
    end: usize,
) -> Option<Vec<ViewStageCapability>> {
    if start > end || end > body.stages.len() {
        return None;
    }
    let mut stages = Vec::with_capacity(end.saturating_sub(start));
    for idx in start..end {
        stages.push(view_never_materializing_stage_capability(body, idx)?);
    }
    Some(stages)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::builtins::{
        registry::{cancellation, BuiltinId},
        BuiltinArgExtremeSink, BuiltinMembershipSink, BuiltinMethod, BuiltinPredicateSink,
        BuiltinSelectionPosition, BuiltinSinkAccumulator, BuiltinViewStage,
    };
    use crate::data::value::Val;
    use crate::exec::pipeline::{
        ArgExtremeSinkSpec, BodyKernel, MembershipSinkSpec, MembershipSinkTarget, NumOp,
        PipelineBody, PredicateSinkSpec, ReducerOp, ReducerSpec, Sink, Stage, ViewInputMode,
        ViewMaterialization, ViewMembershipTarget, ViewOutputMode, ViewSinkCapability,
        ViewStageCapability,
    };
    use crate::parse::ast::BinOp;
    use crate::plan::demand::PullDemand;

    use super::{
        view_capabilities, view_never_materializing_stage_range, view_prefix_capabilities,
    };

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
        assert!(Stage::Map(
            Arc::new(crate::vm::Program::new(Vec::new(), "")),
            BuiltinViewStage::Map
        )
        .view_capability(11, Some(&BodyKernel::Const(Val::Int(1))))
        .is_some());
        assert!(Stage::FlatMap(
            Arc::new(crate::vm::Program::new(Vec::new(), "")),
            BuiltinViewStage::FlatMap
        )
        .view_capability(
            12,
            Some(&BodyKernel::Array(Arc::from([BodyKernel::Current]))),
        )
        .is_some());
        let cancel = cancellation(BuiltinId::from_method(BuiltinMethod::Reverse)).unwrap();
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
                op: BuiltinPredicateSink::Any,
                predicate_kernel: 0,
            }
            .materialization(),
            ViewMaterialization::Never
        );
        assert_eq!(
            ViewSinkCapability::Predicate {
                op: BuiltinPredicateSink::FindOne,
                predicate_kernel: 0,
            }
            .materialization(),
            ViewMaterialization::SinkFinalRow
        );
        assert_eq!(
            ViewSinkCapability::Membership {
                op: BuiltinMembershipSink::Includes,
                target: ViewMembershipTarget::Literal(Val::Int(3)),
            }
            .materialization(),
            ViewMaterialization::Never
        );
        assert_eq!(
            ViewSinkCapability::Membership {
                op: BuiltinMembershipSink::Includes,
                target: ViewMembershipTarget::Literal(Val::arr(vec![Val::Int(3)])),
            }
            .materialization(),
            ViewMaterialization::SinkInputRows
        );
        assert_eq!(
            ViewSinkCapability::ArgExtreme {
                op: BuiltinArgExtremeSink::MaxBy,
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
    fn view_sink_capability_describes_empty_stream_results() {
        assert_eq!(
            serde_json::Value::from(ViewSinkCapability::Collect.empty_stream_result().unwrap()),
            serde_json::json!([])
        );
        assert_eq!(
            ViewSinkCapability::Builtin {
                accumulator: BuiltinSinkAccumulator::Count,
                predicate_kernel: Some(0),
                project_kernel: None,
                materialization: ViewMaterialization::Never,
            }
            .empty_stream_result(),
            Some(Val::Int(0))
        );
        assert_eq!(
            ViewSinkCapability::Builtin {
                accumulator: BuiltinSinkAccumulator::Numeric,
                predicate_kernel: None,
                project_kernel: Some(0),
                materialization: ViewMaterialization::SinkNumericInput,
            }
            .empty_stream_result(),
            None
        );
        assert_eq!(
            ViewSinkCapability::Predicate {
                op: BuiltinPredicateSink::All,
                predicate_kernel: 0,
            }
            .empty_stream_result(),
            Some(Val::Bool(true))
        );
        assert_eq!(
            ViewSinkCapability::Membership {
                op: BuiltinMembershipSink::Index,
                target: ViewMembershipTarget::Literal(Val::Int(3)),
            }
            .empty_stream_result(),
            Some(Val::Null)
        );
        assert_eq!(
            ViewSinkCapability::SelectMany {
                n: 1,
                from_end: false,
                source_reversed: false,
            }
            .empty_stream_result(),
            Some(Val::Null)
        );
        assert_eq!(
            serde_json::Value::from(
                ViewSinkCapability::SelectMany {
                    n: 2,
                    from_end: false,
                    source_reversed: false,
                }
                .empty_stream_result()
                .unwrap()
            ),
            serde_json::json!([])
        );
    }

    #[test]
    fn view_sink_capability_describes_count_cardinality_contract() {
        assert_eq!(
            ViewSinkCapability::Builtin {
                accumulator: BuiltinSinkAccumulator::Count,
                predicate_kernel: None,
                project_kernel: None,
                materialization: ViewMaterialization::Never,
            }
            .count_from_cardinality_predicate(),
            Some(None)
        );
        assert_eq!(
            ViewSinkCapability::Builtin {
                accumulator: BuiltinSinkAccumulator::Count,
                predicate_kernel: Some(2),
                project_kernel: None,
                materialization: ViewMaterialization::Never,
            }
            .count_from_cardinality_predicate(),
            Some(Some(2))
        );
        assert_eq!(
            ViewSinkCapability::Builtin {
                accumulator: BuiltinSinkAccumulator::Count,
                predicate_kernel: None,
                project_kernel: Some(1),
                materialization: ViewMaterialization::SinkInputRows,
            }
            .count_from_cardinality_predicate(),
            None
        );
        assert_eq!(
            ViewSinkCapability::Builtin {
                accumulator: BuiltinSinkAccumulator::Numeric,
                predicate_kernel: None,
                project_kernel: None,
                materialization: ViewMaterialization::SinkNumericInput,
            }
            .count_from_cardinality_predicate(),
            None
        );
        assert_eq!(
            ViewSinkCapability::Collect.count_from_cardinality_predicate(),
            None
        );
    }

    #[test]
    fn view_sink_capability_describes_constant_predicate_cardinality_contract() {
        assert_eq!(
            ViewSinkCapability::Predicate {
                op: BuiltinPredicateSink::Any,
                predicate_kernel: 3,
            }
            .constant_predicate_cardinality_contract(),
            Some((BuiltinPredicateSink::Any, 3))
        );
        assert_eq!(
            ViewSinkCapability::Predicate {
                op: BuiltinPredicateSink::IndicesWhere,
                predicate_kernel: 4,
            }
            .constant_predicate_cardinality_contract(),
            Some((BuiltinPredicateSink::IndicesWhere, 4))
        );
        assert_eq!(
            ViewSinkCapability::Predicate {
                op: BuiltinPredicateSink::FindOne,
                predicate_kernel: 5,
            }
            .constant_predicate_cardinality_contract(),
            None
        );
        assert_eq!(
            ViewSinkCapability::Collect.constant_predicate_cardinality_contract(),
            None
        );
    }

    #[test]
    fn view_sink_capability_finishes_from_known_cardinality() {
        let count = ViewSinkCapability::Builtin {
            accumulator: BuiltinSinkAccumulator::Count,
            predicate_kernel: None,
            project_kernel: None,
            materialization: ViewMaterialization::Never,
        };
        assert_eq!(
            count.result_from_known_cardinality(7, false, &[]),
            Some(Val::Int(7))
        );

        let pred_count = ViewSinkCapability::Builtin {
            accumulator: BuiltinSinkAccumulator::Count,
            predicate_kernel: Some(0),
            project_kernel: None,
            materialization: ViewMaterialization::Never,
        };
        assert_eq!(
            pred_count.result_from_known_cardinality(7, false, &[BodyKernel::ConstBool(true)]),
            Some(Val::Int(7))
        );
        assert_eq!(
            pred_count.result_from_known_cardinality(7, false, &[BodyKernel::ConstBool(false)]),
            Some(Val::Int(0))
        );
        assert_eq!(
            pred_count.result_from_known_cardinality(0, false, &[BodyKernel::Current]),
            Some(Val::Int(0))
        );

        let any = ViewSinkCapability::Predicate {
            op: BuiltinPredicateSink::Any,
            predicate_kernel: 0,
        };
        assert!(any.can_finish_from_known_cardinality(&[BodyKernel::ConstBool(true)]));
        assert!(!any.can_finish_from_known_cardinality(&[BodyKernel::Current]));
        assert_eq!(
            any.result_from_known_cardinality(7, false, &[BodyKernel::ConstBool(true)]),
            Some(Val::Bool(true))
        );
        assert_eq!(
            any.result_from_known_cardinality(7, false, &[BodyKernel::ConstBool(false)]),
            Some(Val::Bool(false))
        );
        assert_eq!(
            any.result_from_known_cardinality(0, true, &[BodyKernel::Current]),
            Some(Val::Bool(false))
        );

        let first = ViewSinkCapability::Builtin {
            accumulator: BuiltinSinkAccumulator::SelectOne(BuiltinSelectionPosition::First),
            predicate_kernel: None,
            project_kernel: None,
            materialization: ViewMaterialization::SinkFinalRow,
        };
        assert!(!first.can_finish_from_known_cardinality(&[]));
    }

    #[test]
    fn view_sink_capability_describes_arg_extreme_contract() {
        assert_eq!(
            ViewSinkCapability::ArgExtreme {
                op: BuiltinArgExtremeSink::MaxBy,
                key_kernel: 7,
            }
            .arg_extreme_contract(),
            Some((BuiltinArgExtremeSink::MaxBy, 7))
        );
        assert_eq!(
            ViewSinkCapability::ArgExtreme {
                op: BuiltinArgExtremeSink::MinBy,
                key_kernel: 8,
            }
            .arg_extreme_contract(),
            Some((BuiltinArgExtremeSink::MinBy, 8))
        );
        assert_eq!(ViewSinkCapability::Collect.arg_extreme_contract(), None);
    }

    #[test]
    fn view_sink_capability_describes_membership_target_resolution() {
        let program = Arc::new(crate::vm::Program::new(Vec::new(), ""));
        let sink = ViewSinkCapability::Membership {
            op: BuiltinMembershipSink::Includes,
            target: ViewMembershipTarget::Program(Arc::clone(&program)),
        };

        let resolved_program = sink.membership_target_program().unwrap();
        assert!(Arc::ptr_eq(&resolved_program, &program));

        assert!(matches!(
            sink.with_resolved_membership_target(Val::Int(9)),
            ViewSinkCapability::Membership {
                op: BuiltinMembershipSink::Includes,
                target: ViewMembershipTarget::Literal(Val::Int(9)),
            }
        ));
        assert!(ViewSinkCapability::Membership {
            op: BuiltinMembershipSink::Index,
            target: ViewMembershipTarget::Literal(Val::Int(1)),
        }
        .membership_target_program()
        .is_none());
    }

    #[test]
    fn view_sink_capability_carries_source_demand_adjustments() {
        let nth =
            ViewSinkCapability::Nth { index: 7 }.for_source_demand(PullDemand::NthInput(7), false);
        assert!(matches!(nth, ViewSinkCapability::Nth { index: 0 }));

        let last_many = ViewSinkCapability::SelectMany {
            n: 3,
            from_end: true,
            source_reversed: false,
        }
        .for_source_demand(PullDemand::LastInput(3), true);
        assert!(matches!(
            last_many,
            ViewSinkCapability::SelectMany {
                n: 3,
                from_end: true,
                source_reversed: true,
            }
        ));

        assert!(last_many.selects_from_end());
        assert!(ViewSinkCapability::Builtin {
            accumulator: BuiltinSinkAccumulator::SelectOne(BuiltinSelectionPosition::Last),
            predicate_kernel: None,
            project_kernel: None,
            materialization: ViewMaterialization::SinkFinalRow,
        }
        .selects_from_end());
        assert!(!ViewSinkCapability::Builtin {
            accumulator: BuiltinSinkAccumulator::SelectOne(BuiltinSelectionPosition::First),
            predicate_kernel: None,
            project_kernel: None,
            materialization: ViewMaterialization::SinkFinalRow,
        }
        .selects_from_end());
    }

    #[test]
    fn view_sink_capability_describes_selective_reverse_last_scan() {
        let last = ViewSinkCapability::Builtin {
            accumulator: BuiltinSinkAccumulator::SelectOne(BuiltinSelectionPosition::Last),
            predicate_kernel: None,
            project_kernel: None,
            materialization: ViewMaterialization::SinkFinalRow,
        };
        let selective = [ViewStageCapability::Filter { kernel: 0 }];
        let preserving = [ViewStageCapability::Map { kernel: 0 }];

        assert!(last.requires_full_reverse_scan_for_selective_last(
            PullDemand::LastInput(1),
            true,
            &selective
        ));
        assert!(!last.requires_full_reverse_scan_for_selective_last(
            PullDemand::LastInput(1),
            true,
            &preserving
        ));
        assert!(!last.requires_full_reverse_scan_for_selective_last(
            PullDemand::FirstInput(1),
            true,
            &selective
        ));
        assert!(!last.requires_full_reverse_scan_for_selective_last(
            PullDemand::LastInput(1),
            false,
            &selective
        ));
        assert!(
            !ViewSinkCapability::Collect.requires_full_reverse_scan_for_selective_last(
                PullDemand::LastInput(1),
                true,
                &selective
            )
        );
    }

    #[test]
    fn view_stage_capability_describes_constant_access_effects() {
        assert_eq!(
            ViewStageCapability::Filter { kernel: 0 }
                .constant_access_effect(&[BodyKernel::ConstBool(true)]),
            super::ViewStageConstantEffect::NoOp
        );
        assert_eq!(
            ViewStageCapability::Filter { kernel: 0 }
                .constant_access_effect(&[BodyKernel::ConstBool(false)]),
            super::ViewStageConstantEffect::Empty
        );
        assert_eq!(
            ViewStageCapability::TakeWhile { kernel: 0 }
                .constant_access_effect(&[BodyKernel::Const(Val::Int(1))]),
            super::ViewStageConstantEffect::NoOp
        );
        assert_eq!(
            ViewStageCapability::DropWhile { kernel: 0 }
                .constant_access_effect(&[BodyKernel::ConstBool(true)]),
            super::ViewStageConstantEffect::Empty
        );
        assert_eq!(
            ViewStageCapability::DropWhile { kernel: 0 }
                .constant_access_effect(&[BodyKernel::ConstBool(false)]),
            super::ViewStageConstantEffect::NoOp
        );
        assert_eq!(
            ViewStageCapability::Map { kernel: 0 }
                .constant_access_effect(&[BodyKernel::ConstBool(false)]),
            super::ViewStageConstantEffect::Keep
        );
    }

    #[test]
    fn view_stage_capability_tracks_deterministic_cardinality() {
        let stages = [
            ViewStageCapability::Map { kernel: 0 },
            ViewStageCapability::Take(5),
            ViewStageCapability::Skip(2),
            ViewStageCapability::Filter { kernel: 1 },
        ];
        let kernels = [BodyKernel::Current, BodyKernel::ConstBool(true)];
        assert_eq!(
            ViewStageCapability::deterministic_prefix_cardinality_after(&stages, &kernels, 10),
            Some(3)
        );

        let false_filter = [ViewStageCapability::Filter { kernel: 0 }];
        assert_eq!(
            ViewStageCapability::deterministic_prefix_cardinality_after(
                &false_filter,
                &[BodyKernel::ConstBool(false)],
                10,
            ),
            Some(0)
        );
        assert!(ViewStageCapability::prefix_forces_empty(
            &false_filter,
            &[BodyKernel::ConstBool(false)]
        ));

        let unsupported = [ViewStageCapability::Compact];
        assert_eq!(
            ViewStageCapability::deterministic_prefix_cardinality_after(&unsupported, &[], 10),
            None
        );
        assert!(!ViewStageCapability::prefix_forces_empty(&unsupported, &[]));
    }

    #[test]
    fn view_stage_capability_rewrites_source_access_prefix() {
        let stages = [
            ViewStageCapability::Map { kernel: 0 },
            ViewStageCapability::Filter { kernel: 1 },
            ViewStageCapability::Take(3),
        ];
        let kernels = [BodyKernel::Current, BodyKernel::ConstBool(true)];
        let rewritten = ViewStageCapability::source_access_stages(&stages, &kernels);
        assert!(matches!(
            rewritten.as_ref(),
            [
                ViewStageCapability::Map { kernel: 0 },
                ViewStageCapability::Take(3)
            ]
        ));

        let empty = [
            ViewStageCapability::Map { kernel: 0 },
            ViewStageCapability::TakeWhile { kernel: 1 },
            ViewStageCapability::Take(3),
        ];
        let kernels = [BodyKernel::Current, BodyKernel::ConstBool(false)];
        let rewritten = ViewStageCapability::source_access_stages(&empty, &kernels);
        assert!(matches!(
            rewritten.as_ref(),
            [
                ViewStageCapability::Map { kernel: 0 },
                ViewStageCapability::Take(0)
            ]
        ));

        let unchanged = [ViewStageCapability::Filter { kernel: 0 }];
        let rewritten = ViewStageCapability::source_access_stages(&unchanged, &[]);
        assert!(matches!(
            rewritten.as_ref(),
            [ViewStageCapability::Filter { kernel: 0 }]
        ));
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
                op: BuiltinPredicateSink::Any,
                predicate: Arc::new(crate::vm::Program::new(Vec::new(), "")),
                predicate_expr: None,
            })
            .view_capability(&[BodyKernel::FieldCmpLit(
                Arc::from("score"),
                BinOp::Gt,
                Val::Int(10),
            )]),
            Some(ViewSinkCapability::Predicate {
                op: BuiltinPredicateSink::Any,
                predicate_kernel: 0,
            })
        ));
        assert!(matches!(
            Sink::Predicate(PredicateSinkSpec {
                op: BuiltinPredicateSink::FindOne,
                predicate: Arc::new(crate::vm::Program::new(Vec::new(), "")),
                predicate_expr: None,
            })
            .view_capability(&[BodyKernel::FieldCmpLit(
                Arc::from("score"),
                BinOp::Eq,
                Val::Int(10),
            )]),
            Some(ViewSinkCapability::Predicate {
                op: BuiltinPredicateSink::FindOne,
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
                op: BuiltinMembershipSink::Includes,
                target: MembershipSinkTarget::Literal(Val::Int(3)),
            })
            .view_capability(&[]),
            Some(ViewSinkCapability::Membership {
                op: BuiltinMembershipSink::Includes,
                target: ViewMembershipTarget::Literal(Val::Int(3)),
            })
        ));
        assert!(matches!(
            Sink::Membership(MembershipSinkSpec {
                op: BuiltinMembershipSink::Includes,
                target: MembershipSinkTarget::Program(Arc::new(crate::vm::Program::new(
                    Vec::new(),
                    ""
                ))),
            })
            .view_capability(&[]),
            Some(ViewSinkCapability::Membership {
                op: BuiltinMembershipSink::Includes,
                target: ViewMembershipTarget::Program(_),
            })
        ));
        assert!(matches!(
            Sink::ArgExtreme(ArgExtremeSinkSpec {
                op: crate::builtins::BuiltinArgExtremeSink::MaxBy,
                key: Arc::new(crate::vm::Program::new(Vec::new(), "")),
                key_expr: None,
            })
            .view_capability(&[BodyKernel::FieldRead(Arc::from("score"))]),
            Some(ViewSinkCapability::ArgExtreme {
                op: crate::builtins::BuiltinArgExtremeSink::MaxBy,
                key_kernel: 0,
            })
        ));
        assert!(Sink::ArgExtreme(ArgExtremeSinkSpec {
            op: crate::builtins::BuiltinArgExtremeSink::MinBy,
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
    fn view_prefix_includes_registry_scalar_stage() {
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

        assert!(view_capabilities(&body).is_some());
        let prefix = view_prefix_capabilities(&body).unwrap();
        assert_eq!(prefix.consumed_stages, 3);
        assert_eq!(prefix.stages.len(), 3);
        assert_eq!(
            view_never_materializing_stage_range(&body, 0, 3)
                .expect("all stages are view-native")
                .len(),
            3
        );
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
        assert_eq!(prefix.consumed_stages, 3);
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
