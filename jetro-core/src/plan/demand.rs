//! Shared demand model and backward propagation helpers.
//!
//! Demand is a planning concern, not parser syntax: sinks describe how much
//! input and value payload they need, and stage/operator adapters translate
//! that demand backward toward the source.

use std::sync::Arc;

/// Describes how much of each element's content a pipeline stage actually
/// needs to read, used to skip deserialisation or evaluation work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueNeed {
    /// The stage only counts matching elements; payload can be skipped unless predicates need it.
    CountOnly,
    /// The stage only needs to know whether at least one element exists.
    ExistsOnly,
    /// The stage evaluates a predicate and needs enough of the value to test it.
    Predicate,
    /// The stage only needs fields used by a projection.
    Projection,
    /// The full element value is required.
    Whole,
    /// Only the numeric interpretation of the element is needed (e.g. for `sum`).
    Numeric,
}

impl ValueNeed {
    /// Returns `true` when satisfying this need requires reading row payload.
    pub(crate) fn requires_payload(self) -> bool {
        !matches!(self, ValueNeed::CountOnly | ValueNeed::ExistsOnly)
    }

    /// Return the stricter of two `ValueNeed` values; `Whole` dominates all others.
    pub(crate) fn merge(self, other: Self) -> Self {
        use ValueNeed::*;
        match (self, other) {
            (Whole, _) | (_, Whole) => Whole,
            (Numeric, _) | (_, Numeric) => Numeric,
            (Projection, _) | (_, Projection) => Projection,
            (Predicate, _) | (_, Predicate) => Predicate,
            (ExistsOnly, _) | (_, ExistsOnly) => ExistsOnly,
            (CountOnly, CountOnly) => CountOnly,
        }
    }
}

/// A rooted field path needed from an input row, e.g. `price` or `user.name`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FieldPath {
    keys: Arc<[Arc<str>]>,
}

impl FieldPath {
    /// Construct a single-key field path.
    pub fn single(key: Arc<str>) -> Self {
        Self {
            keys: Arc::from([key]),
        }
    }

    /// Construct a field path from an existing key chain.
    pub fn chain(keys: Arc<[Arc<str>]>) -> Self {
        Self { keys }
    }

    /// Borrow the key chain.
    #[cfg(test)]
    pub fn keys(&self) -> &[Arc<str>] {
        &self.keys
    }

    /// Return this path with `prefix` inserted before its first key.
    pub fn prefixed(&self, prefix: &[Arc<str>]) -> Self {
        let mut keys = Vec::with_capacity(prefix.len() + self.keys.len());
        keys.extend(prefix.iter().cloned());
        keys.extend(self.keys.iter().cloned());
        Self { keys: keys.into() }
    }
}

/// Small ordered set of field paths. Insertion preserves first-seen order so diagnostics
/// and tests remain deterministic.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FieldSet {
    paths: Vec<FieldPath>,
}

impl FieldSet {
    /// Construct an empty field set.
    pub fn new() -> Self {
        Self { paths: Vec::new() }
    }

    /// Construct a field set containing one single-key path.
    pub fn single(key: Arc<str>) -> Self {
        let mut out = Self::new();
        out.insert(FieldPath::single(key));
        out
    }

    /// Construct a field set containing one field chain.
    pub fn chain(keys: Arc<[Arc<str>]>) -> Self {
        let mut out = Self::new();
        out.insert(FieldPath::chain(keys));
        out
    }

    /// Insert a path if it is not already present.
    pub fn insert(&mut self, path: FieldPath) {
        if !self.paths.iter().any(|existing| existing == &path) {
            self.paths.push(path);
        }
    }

    /// Merge all paths from `other` into `self`.
    pub fn extend(&mut self, other: &FieldSet) {
        for path in other.paths.iter() {
            self.insert(path.clone());
        }
    }

    /// Borrow all paths in deterministic order.
    #[cfg(test)]
    pub fn paths(&self) -> &[FieldPath] {
        &self.paths
    }

    /// Return all paths with `prefix` inserted before each path.
    pub fn prefixed(&self, prefix: &[Arc<str>]) -> Self {
        let mut out = Self::new();
        for path in self.paths.iter() {
            out.insert(path.prefixed(prefix));
        }
        out
    }

}

/// Precise value payload need for high-performance planning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldDemand {
    /// The row payload is not inspected.
    None,
    /// Only the listed field paths are inspected.
    Fields(FieldSet),
    /// The whole row may be inspected or materialised.
    Whole,
}

impl FieldDemand {
    /// Merge two field demands. `Whole` dominates; field sets are unioned.
    pub fn merge(self, other: Self) -> Self {
        match (self, other) {
            (Self::Whole, _) | (_, Self::Whole) => Self::Whole,
            (Self::None, need) | (need, Self::None) => need,
            (Self::Fields(mut lhs), Self::Fields(rhs)) => {
                lhs.extend(&rhs);
                Self::Fields(lhs)
            }
        }
    }

    /// Returns `true` if no row payload is required.
    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }
}

/// Two-lane payload demand: fields needed while scanning versus fields needed only
/// for rows that survive selection and become output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DemandLanes {
    /// Payload needed before a row is known to be selected, e.g. filter/sort keys.
    pub scan_need: FieldDemand,
    /// Payload needed only for selected/emitted rows, e.g. a delayed projection.
    pub result_need: FieldDemand,
}

impl DemandLanes {
    /// No row payload is needed in either lane.
    pub const NONE: Self = Self {
        scan_need: FieldDemand::None,
        result_need: FieldDemand::None,
    };

    /// Full output-row materialisation, with no scan-time payload requirement.
    pub const RESULT: Self = Self {
        scan_need: FieldDemand::None,
        result_need: FieldDemand::Whole,
    };

    /// Merge `need` into the scan lane.
    pub fn merge_scan(&mut self, need: FieldDemand) {
        self.scan_need = self.scan_need.clone().merge(need);
    }

    /// Merge `need` into the result lane.
    pub fn merge_result(&mut self, need: FieldDemand) {
        self.result_need = self.result_need.clone().merge(need);
    }
}

/// Specifies how many input elements a stage must pull from its source to
/// satisfy a downstream consumer's limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PullDemand {
    /// Pull all available input elements without any limit.
    All,
    /// Pull at most the first `n` input elements regardless of how many outputs they produce.
    FirstInput(usize),
    /// Pull from the end of the input until `n` outputs have been produced.
    LastInput(usize),
    /// Pull the input element at zero-based index `i` when the source can seek to it.
    NthInput(usize),
    /// Pull input until exactly `n` output elements have been produced.
    UntilOutput(usize),
}

/// Demand for scalar terminal sinks whose result can be decided before all rows
/// are consumed. This is intentionally separate from `PullDemand::UntilOutput`,
/// which counts rows emitted by the stage chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SinkResultDemand {
    /// The sink result cannot be decided early by a single row event.
    None,
    /// Stop once a predicate or membership row matches.
    UntilMatch,
    /// Stop once a predicate row fails.
    UntilFailure,
}

impl SinkResultDemand {
    /// Returns true when the sink result may allow executor-level short-circuiting.
    #[cfg(test)]
    pub(crate) fn can_short_circuit(self) -> bool {
        !matches!(self, Self::None)
    }
}

impl PullDemand {
    /// Returns true when this demand can be satisfied without reading any input rows.
    pub(crate) fn is_zero(self) -> bool {
        matches!(
            self,
            PullDemand::FirstInput(0) | PullDemand::LastInput(0) | PullDemand::UntilOutput(0)
        )
    }

    /// Return a `PullDemand` capped to at most `n` input elements,
    /// converting `All` or `UntilOutput` variants to `FirstInput(n)`.
    pub(crate) fn cap_inputs(self, n: usize) -> Self {
        match self {
            PullDemand::All | PullDemand::UntilOutput(_) | PullDemand::LastInput(_) => {
                PullDemand::FirstInput(n)
            }
            PullDemand::FirstInput(m) => PullDemand::FirstInput(m.min(n)),
            PullDemand::NthInput(i) => {
                if i < n {
                    PullDemand::NthInput(i)
                } else {
                    PullDemand::FirstInput(n)
                }
            }
        }
    }
}

/// Combined downstream demand annotation: how much to pull, what payload is
/// needed, and whether the consumer requires stable input ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Demand {
    /// How many upstream elements must be consumed.
    pub pull: PullDemand,
    /// How much of each element's payload is required.
    pub value: ValueNeed,
    /// Whether the consumer depends on elements arriving in their original order.
    pub order: bool,
}

impl Demand {
    /// The terminal demand used at the sink of a pipeline: pull everything,
    /// need whole values, and require ordering.
    pub const RESULT: Demand = Demand {
        pull: PullDemand::All,
        value: ValueNeed::Whole,
        order: true,
    };

    /// Construct a demand that pulls all input with the given value need and
    /// order requirement.
    pub fn all(value: ValueNeed) -> Self {
        Self {
            pull: PullDemand::All,
            value,
            order: true,
        }
    }

    /// Construct a demand that pulls only the first input element with the
    /// given value need, and no ordering requirement.
    pub fn first(value: ValueNeed) -> Self {
        Self {
            pull: PullDemand::FirstInput(1),
            value,
            order: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{FieldDemand, FieldSet, PullDemand, SinkResultDemand};

    fn paths(need: FieldDemand) -> Vec<String> {
        match need {
            FieldDemand::Fields(fields) => fields
                .paths()
                .iter()
                .map(|path| {
                    path.keys()
                        .iter()
                        .map(|key| key.as_ref())
                        .collect::<Vec<_>>()
                        .join(".")
                })
                .collect(),
            FieldDemand::None => Vec::new(),
            FieldDemand::Whole => vec!["*".to_string()],
        }
    }

    #[test]
    fn field_sets_prefix_nested_paths() {
        let mut fields = FieldSet::chain(Arc::from([Arc::<str>::from("name")]));
        fields.insert(super::FieldPath::chain(Arc::from([
            Arc::<str>::from("address"),
            Arc::<str>::from("city"),
        ])));
        let prefixed = fields.prefixed(&[Arc::from("user")]);
        assert_eq!(
            paths(FieldDemand::Fields(prefixed)),
            vec!["user.name", "user.address.city"]
        );
    }

    #[test]
    fn pull_demand_caps_inputs_without_crossing_prefix_bounds() {
        assert_eq!(PullDemand::All.cap_inputs(3), PullDemand::FirstInput(3));
        assert_eq!(
            PullDemand::UntilOutput(2).cap_inputs(3),
            PullDemand::FirstInput(3)
        );
        assert_eq!(
            PullDemand::LastInput(2).cap_inputs(3),
            PullDemand::FirstInput(3)
        );
        assert_eq!(
            PullDemand::FirstInput(5).cap_inputs(3),
            PullDemand::FirstInput(3)
        );
        assert_eq!(PullDemand::NthInput(2).cap_inputs(3), PullDemand::NthInput(2));
        assert_eq!(
            PullDemand::NthInput(3).cap_inputs(3),
            PullDemand::FirstInput(3)
        );
    }

    #[test]
    fn pull_demand_zero_only_matches_no_read_variants() {
        assert!(PullDemand::FirstInput(0).is_zero());
        assert!(PullDemand::LastInput(0).is_zero());
        assert!(PullDemand::UntilOutput(0).is_zero());
        assert!(!PullDemand::NthInput(0).is_zero());
        assert!(!PullDemand::All.is_zero());
    }

    #[test]
    fn sink_result_demand_is_separate_from_row_output_demand() {
        assert!(SinkResultDemand::UntilMatch.can_short_circuit());
        assert!(SinkResultDemand::UntilFailure.can_short_circuit());
        assert!(!SinkResultDemand::None.can_short_circuit());
    }
}

/// Adapter trait implemented by whichever operator representation a planner
/// uses for demand propagation.
pub trait DemandOperator {
    /// Propagate `downstream` demand through this operator, returning the
    /// upstream demand that its source must satisfy.
    fn propagate_demand(&self, downstream: Demand) -> Demand;
}

/// A single annotated step produced by `propagate_demands`, recording an
/// operator alongside the demand it receives and the demand it places upstream.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg(test)]
pub struct DemandStep<Op> {
    /// The operator at this position in the chain.
    pub op: Op,
    /// Demand flowing into this operator from the downstream consumer.
    pub downstream: Demand,
    /// Demand this operator forwards to its upstream source.
    pub upstream: Demand,
}

/// Walk `ops` in reverse and compute each operator's upstream demand given
/// `final_demand` at the sink, returning annotated `DemandStep`s in forward order.
#[cfg(test)]
pub fn propagate_demands<Op>(ops: &[Op], final_demand: Demand) -> Vec<DemandStep<Op>>
where
    Op: DemandOperator + Clone,
{
    let mut demand = final_demand;
    let mut out = Vec::with_capacity(ops.len());
    for op in ops.iter().rev() {
        let upstream = op.propagate_demand(demand);
        out.push(DemandStep {
            op: op.clone(),
            downstream: demand,
            upstream,
        });
        demand = upstream;
    }
    out.reverse();
    out
}

/// Fold demand propagation over `ops` from sink to source and return only
/// the final upstream demand without allocating intermediate `DemandStep`s.
#[cfg(test)]
pub fn source_demand<Op>(ops: &[Op], final_demand: Demand) -> Demand
where
    Op: DemandOperator,
{
    ops.iter()
        .rev()
        .fold(final_demand, |demand, op| op.propagate_demand(demand))
}
