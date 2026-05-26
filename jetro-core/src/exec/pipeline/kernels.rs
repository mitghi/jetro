//! Classified expression kernels for pipeline stage evaluation.
//!
//! `BodyKernel` is a pre-classified form of a stage body expression that lets
//! the pipeline executor skip the VM for common patterns (field reads,
//! literal comparisons, `FieldCmpLit` fusions). Generic/unknown expressions
//! fall back to `BodyKernel::Generic`, which re-enters the VM.

use std::sync::Arc;

use crate::builtins::registry::{
    apply_view_projection, by_name as builtin_by_name, count_sink_accepts_predicate, expr_stage,
    numeric_reducer, view_projection, view_projection_receiver_field_demand,
    view_projection_result_field_path, ViewProjectionResult,
};
use crate::builtins::{BuiltinArgs, BuiltinArraySelector, BuiltinCall, BuiltinExprStage};
use crate::data::context::EvalError;
use crate::data::value::Val;
use crate::data::view::{scalar_view_to_owned_val, write_json_view, ValView, ValueView};
use crate::parse::ast::{Expr, ObjField, Step};
use crate::plan::demand::{FieldDemand, FieldSet};
use crate::util::JsonView;

/// Prepared nested collection pipeline carried by a classified body kernel.
/// It keeps the original plan for demand analysis and a reusable prepared
/// runner for row-by-row execution.
#[derive(Debug, Clone)]
pub struct NestedPlanKernel {
    plan: Arc<super::Plan>,
    prepared: super::nested::PreparedPlan,
    view_plan: Option<NestedViewPlan>,
}

/// View-native source metadata for a nested plan relative to the current row.
#[derive(Debug, Clone)]
pub(crate) enum NestedViewSource {
    /// The nested plan runs directly against the current row.
    Receiver,
    /// The nested plan first reads a field path from the current row.
    FieldChain(Arc<[Arc<str>]>),
}

/// Cached view-native body/source pair for nested plan execution.
#[derive(Debug, Clone)]
pub(crate) struct NestedViewPlan {
    source: NestedViewSource,
    body: super::PipelineBody,
}

impl NestedViewPlan {
    #[inline]
    pub(crate) fn source(&self) -> &NestedViewSource {
        &self.source
    }

    #[inline]
    pub(crate) fn body(&self) -> &super::PipelineBody {
        &self.body
    }
}

impl NestedPlanKernel {
    pub(crate) fn new(plan: Arc<super::Plan>) -> Self {
        let prepared = super::nested::PreparedPlan::new(&plan);
        let view_plan = nested_view_plan(&plan);
        Self {
            plan,
            prepared,
            view_plan,
        }
    }

    pub(crate) fn parent_field_demand(&self) -> FieldDemand {
        self.plan.parent_field_demand()
    }

    pub(crate) fn run(&self, seed: Val) -> Result<Val, EvalError> {
        self.prepared.run(seed)
    }

    pub(crate) fn view_plan(&self) -> Option<&NestedViewPlan> {
        self.view_plan.as_ref()
    }
}

fn nested_view_plan(plan: &super::Plan) -> Option<NestedViewPlan> {
    let source = match &plan.source {
        super::Source::Receiver(_) => NestedViewSource::Receiver,
        super::Source::FieldChain { keys } => NestedViewSource::FieldChain(Arc::clone(keys)),
    };
    Some(NestedViewPlan {
        source,
        body: super::PipelineBody {
            stages: plan.stages.clone(),
            stage_exprs: plan.stage_exprs.clone(),
            sink: plan.sink.clone(),
            stage_kernels: plan.stage_kernels.clone(),
            sink_kernels: plan.sink_kernels.clone(),
        },
    })
}

/// Pre-classified stage body expression; variants are ordered least-to-most expensive, `Generic` re-enters the VM.
#[derive(Debug, Clone)]
pub enum BodyKernel {
    /// Expression not classifiable into a faster form; falls back to full VM evaluation.
    Generic,
    /// Returns the current element unchanged (`@`).
    Current,
    /// Reads a single named field from the current element object.
    FieldRead(Arc<str>),
    /// Traverses a chain of field names left-to-right, returning the final value.
    FieldChain(Arc<[Arc<str>]>),
    /// Applies a view-scalar builtin to the result of the receiver kernel.
    BuiltinCall {
        /// Sub-kernel that computes the value the builtin is called on.
        receiver: Box<BodyKernel>,
        /// The resolved builtin method and its static arguments.
        call: BuiltinCall,
    },
    /// Chains two kernels: applies `first`, then feeds the result into `then`.
    Compose {
        /// The first kernel in the composition chain.
        first: Box<BodyKernel>,
        /// The kernel applied to the output of `first`.
        then: Box<BodyKernel>,
    },
    /// Compares the result of `lhs` to a literal using `op`, returning a boolean.
    CmpLit {
        /// The sub-kernel whose result is the left-hand side of the comparison.
        lhs: Box<BodyKernel>,
        /// The comparison operator.
        op: crate::parse::ast::BinOp,
        /// The literal right-hand side value.
        lit: Val,
    },
    /// Applies a binary arithmetic/string/array operation to two sub-kernels.
    Binary {
        /// The left-hand side kernel.
        lhs: Box<BodyKernel>,
        /// The binary operator.
        op: crate::parse::ast::BinOp,
        /// The right-hand side kernel.
        rhs: Box<BodyKernel>,
    },
    /// Selects one child from an array-like sub-kernel.
    ArraySelect {
        /// Kernel that yields the child array.
        array: Box<BodyKernel>,
        /// Position to select from the child array.
        selector: ArraySelector,
    },
    /// Materialises a slice of an array-like view without materialising the receiver row.
    Slice {
        array: Box<BodyKernel>,
        from: Option<i64>,
        to: Option<i64>,
        step: Option<i64>,
    },
    /// Selects an object field or array index using a view-native key expression.
    DynIndex {
        receiver: Box<BodyKernel>,
        key: Box<BodyKernel>,
    },
    /// Runs a compiled pattern match against a view-native scrutinee.
    Match {
        /// Kernel that yields the match scrutinee.
        scrutinee: Box<BodyKernel>,
        /// Compiled match program.
        compiled: Arc<crate::vm::CompiledMatch>,
        /// Whether arm bodies can read `@` and therefore need the scrutinee as `Env::current`.
        body_needs_current: bool,
    },
    /// Short-circuits through a list of predicates, returning `false` on the first failure.
    And(Arc<[BodyKernel]>),
    /// Short-circuits through a list of predicates, returning `true` on the first success.
    Or(Arc<[BodyKernel]>),
    /// Arithmetic negation of a view-native numeric expression.
    Neg(Box<BodyKernel>),
    /// Boolean negation of a view-native expression.
    Not(Box<BodyKernel>),
    /// Runtime type check over a view-native expression.
    KindCheck {
        expr: Box<BodyKernel>,
        ty: crate::parse::ast::KindType,
        negate: bool,
    },
    /// Error-free explicit cast over a view-native expression.
    Cast {
        expr: Box<BodyKernel>,
        ty: crate::parse::ast::CastType,
    },
    /// Null coalescing: returns the first non-null expression.
    Coalesce {
        lhs: Box<BodyKernel>,
        rhs: Box<BodyKernel>,
    },
    /// Conditional expression with view-native condition and branches.
    IfElse {
        cond: Box<BodyKernel>,
        then_: Box<BodyKernel>,
        else_: Box<BodyKernel>,
    },
    /// Reads a single field and compares it to a literal in one fused step.
    FieldCmpLit(Arc<str>, crate::parse::ast::BinOp, Val),
    /// Traverses a field chain and compares the result to a literal in one fused step.
    FieldChainCmpLit(Arc<[Arc<str>]>, crate::parse::ast::BinOp, Val),
    /// Compares the current element directly to a literal.
    CurrentCmpLit(crate::parse::ast::BinOp, Val),
    /// Always produces the given boolean constant, regardless of the current element.
    ConstBool(bool),
    /// Always produces the given `Val` constant.
    Const(Val),
    /// Evaluates an interpolated format string by evaluating each part kernel.
    FString(FStringKernel),
    /// Evaluates an object literal by evaluating each field-value kernel.
    Object(ObjectKernel),
    /// Evaluates an array literal by evaluating each item kernel.
    Array(Arc<[BodyKernel]>),
    /// Evaluates an array literal that contains one or more spread elements.
    ArraySpread(Arc<[ArrayKernelElem]>),
    /// Runs a nested array reducer such as `items.map(qty * price).sum()` without
    /// materialising the outer row.
    NestedArrayReducer {
        /// Kernel that resolves the nested array from the current row.
        source: Box<BodyKernel>,
        /// Optional per-child predicate applied before reducing.
        predicate: Option<Box<BodyKernel>>,
        /// Optional per-child projection applied before reducing.
        map: Option<Box<BodyKernel>>,
        /// Numeric reducer operation.
        op: super::NumOp,
    },
    /// Counts a nested array, optionally after a view-native per-child predicate.
    NestedArrayCount {
        /// Kernel that resolves the nested array from the current row.
        source: Box<BodyKernel>,
        /// Optional per-child predicate applied before counting.
        predicate: Option<Box<BodyKernel>>,
    },
    /// Runs a prepared nested collection pipeline against the current row.
    NestedPlan(Arc<NestedPlanKernel>),
}

fn compose_field_demand(first: &BodyKernel, then: &BodyKernel) -> FieldDemand {
    match then.field_demand() {
        FieldDemand::None => FieldDemand::None,
        FieldDemand::Whole => first.field_demand(),
        FieldDemand::Fields(fields) => match first {
            BodyKernel::Current => FieldDemand::Fields(fields),
            BodyKernel::FieldRead(key) => FieldDemand::Fields(fields.prefixed(&[Arc::clone(key)])),
            BodyKernel::FieldChain(keys) => FieldDemand::Fields(fields.prefixed(keys)),
            _ => first.field_demand(),
        },
    }
}

fn object_key_call_field_demand(receiver: &BodyKernel, call: &BuiltinCall) -> Option<FieldDemand> {
    let receiver_path = receiver.field_path_keys()?;
    view_projection_receiver_field_demand(call.id(), &call.args, &receiver_path)
}

fn view_result_field_path(receiver: &BodyKernel, call: &BuiltinCall) -> Option<Vec<Arc<str>>> {
    view_projection_result_field_path(call.id(), &call.args, &receiver.field_path_keys()?)
}

fn nested_array_child_field_demand(source: &BodyKernel, child: &BodyKernel) -> FieldDemand {
    match child.field_demand() {
        FieldDemand::None => FieldDemand::None,
        FieldDemand::Whole => source.field_demand(),
        FieldDemand::Fields(fields) => match source.field_path_keys() {
            Some(keys) => FieldDemand::Fields(fields.prefixed(&keys)),
            None => source.field_demand(),
        },
    }
}

fn nested_array_field_demand(
    source: &BodyKernel,
    predicate: Option<&BodyKernel>,
    map: Option<&BodyKernel>,
) -> FieldDemand {
    let need = source.field_demand();
    let need = match predicate {
        Some(predicate) => need.merge(nested_array_child_field_demand(source, predicate)),
        None => need,
    };
    match map {
        Some(map) => need.merge(nested_array_child_field_demand(source, map)),
        None => need,
    }
}

/// Pre-classified kernel for a format-string expression, avoiding VM re-entry for each part.
#[derive(Debug, Clone)]
pub struct FStringKernel {
    // ordered parts (literals and interpolated sub-kernels) that make up the format string
    parts: Arc<[FStringKernelPart]>,
    // pre-computed lower-bound capacity hint for the output string buffer
    base_capacity: usize,
}

impl FStringKernel {
    #[inline]
    #[cfg(test)]
    pub(crate) fn new(parts: Arc<[FStringKernelPart]>, base_capacity: usize) -> Self {
        Self {
            parts,
            base_capacity,
        }
    }

    #[inline]
    pub(crate) fn parts(&self) -> &[FStringKernelPart] {
        &self.parts
    }

    #[inline]
    pub(crate) fn base_capacity(&self) -> usize {
        self.base_capacity
    }
}

/// A single part of an `FStringKernel`: either a fixed literal or a dynamic interpolation.
#[derive(Debug, Clone)]
pub enum FStringKernelPart {
    /// A constant string segment that is copied verbatim into the output.
    Lit(Arc<str>),
    /// A sub-kernel whose result is formatted and appended to the output string.
    Interp(BodyKernel),
}

/// One element in an array literal kernel.
#[derive(Debug, Clone)]
pub enum ArrayKernelElem {
    /// Push the evaluated value as one array element.
    Value(BodyKernel),
    /// Spread the evaluated value using VM-compatible array spread semantics.
    Spread(BodyKernel),
}

impl ArrayKernelElem {
    #[inline]
    fn kernel(&self) -> &BodyKernel {
        match self {
            Self::Value(kernel) | Self::Spread(kernel) => kernel,
        }
    }
}

/// Pre-classified kernel for an object-literal expression; bypasses the VM's object-construction opcodes.
#[derive(Debug, Clone)]
pub struct ObjectKernel {
    // ordered key/value entries that constitute the produced object
    entries: Arc<[ObjectKernelEntry]>,
}

/// A single key/value entry in an `ObjectKernel`.
#[derive(Debug, Clone)]
pub struct ObjectKernelEntry {
    // key expression in the produced object
    key: ObjectKernelKey,
    // kernel used to compute the value for this key
    value: BodyKernel,
    // optional guard; falsy skips this entry
    cond: Option<BodyKernel>,
    // when true, a null result causes this entry to be silently omitted
    optional: bool,
    // when true, null values are omitted regardless of the optional flag
    omit_null: bool,
}

/// Static or computed object-key expression for `ObjectKernel`.
#[derive(Debug, Clone)]
pub enum ObjectKernelKey {
    /// A statically known key, eligible for uniform columnar output.
    Static(Arc<str>),
    /// A computed key evaluated against the current row and coerced via `val_to_key`.
    Dynamic(BodyKernel),
    /// An object spread; `value` is evaluated and merged if it is an object.
    Spread(ObjectSpreadMode),
}

/// Object spread merge mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectSpreadMode {
    /// One-level key overwrite semantics.
    Shallow,
    /// Recursive object merge and array concatenation semantics.
    Deep,
}

/// Positional selector used by `BodyKernel::ArraySelect`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArraySelector {
    /// First child.
    First,
    /// Last child.
    Last,
    /// Zero-based child index.
    Nth(usize),
}

/// Metadata for kernels shaped as `array-selector.suffix.scalar_call()`.
///
/// This is intentionally path/key based rather than backend-specific; byte,
/// tape, and view executors can lower it to their own path representation.
#[derive(Debug, Clone)]
pub(crate) struct ArrayElementScalarCall {
    /// Path to the source array, relative to the current row.
    pub source_keys: Vec<Arc<str>>,
    /// Array element selected from `source_keys`.
    pub selector: ArraySelector,
    /// Field suffix read from the selected element before applying `call`.
    pub suffix_keys: Vec<Arc<str>>,
    /// View-scalar builtin applied to the selected value.
    pub call: BuiltinCall,
}

/// Metadata for kernels shaped as `path.scalar_call()`.
///
/// The receiver is a direct current-row field path; callers can lower the keys
/// to byte, tape, or owned-value access without re-matching the kernel shape.
#[derive(Debug, Clone)]
pub(crate) struct PathScalarCall {
    /// Receiver path relative to the current row.
    pub receiver_keys: Vec<Arc<str>>,
    /// View-scalar builtin applied to the receiver value.
    pub call: BuiltinCall,
}

impl ArraySelector {
    /// Build a pipeline selector from registry selector metadata plus an optional integer arg.
    pub(crate) fn from_builtin_selector(
        selector: BuiltinArraySelector,
        index: Option<i64>,
    ) -> Option<Self> {
        match selector {
            BuiltinArraySelector::First => Some(Self::First),
            BuiltinArraySelector::Last => Some(Self::Last),
            BuiltinArraySelector::Nth => match index {
                Some(index) if index >= 0 => Some(Self::Nth(index as usize)),
                _ => None,
            },
        }
    }

    /// Return the selected array index for an array of `len`, if the selector can be satisfied.
    #[inline]
    pub(crate) fn index_for_len(self, len: usize) -> Option<usize> {
        match self {
            Self::First => Some(0),
            Self::Last => len.checked_sub(1),
            Self::Nth(idx) => Some(idx),
        }
    }
}

impl ObjectKernel {
    /// Returns the number of key/value entries in this object kernel.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    /// Collects and returns the ordered list of key names from this object kernel.
    pub(crate) fn keys(&self) -> Arc<[Arc<str>]> {
        self.entries
            .iter()
            .filter_map(|entry| entry.static_key().cloned())
            .collect::<Vec<_>>()
            .into()
    }

    pub(crate) fn entries(&self) -> &[ObjectKernelEntry] {
        &self.entries
    }

    #[inline]
    pub(crate) fn has_static_layout(&self) -> bool {
        !self.entries.is_empty()
            && self
                .entries
                .iter()
                .all(|entry| matches!(entry.key, ObjectKernelKey::Static(_)))
    }

    /// Evaluates each entry against `item` using caller-owned VM state, appending to `cells`;
    /// returns `false` on null-optional skip.
    pub(crate) fn eval_val_row_cells_with_vm(
        &self,
        item: &Val,
        cells: &mut Vec<Val>,
        vm: &mut crate::vm::VM,
    ) -> bool {
        let start = cells.len();
        for entry in self.entries.iter() {
            if !matches!(entry.key, ObjectKernelKey::Static(_)) {
                cells.truncate(start);
                return false;
            }
            if let Some(cond) = &entry.cond {
                let keep = eval_native_kernel_with_vm(cond, item, vm)
                    .map(|value| crate::util::is_truthy(&value))
                    .unwrap_or(false);
                if !keep {
                    cells.truncate(start);
                    return false;
                }
            }
            let value = eval_native_kernel_with_vm(&entry.value, item, vm).unwrap_or(Val::Null);
            if (entry.optional || entry.omit_null) && value.is_null() {
                cells.truncate(start);
                return false;
            }
            cells.push(value);
        }
        true
    }

    /// Evaluates all entries against `item` using caller-owned VM state into a `Val::ObjSmall`,
    /// returning `Val::Null` on sub-kernel failure.
    pub(crate) fn eval_val_with_vm(&self, item: &Val, vm: &mut crate::vm::VM) -> Val {
        eval_object_kernel(self, |kernel| eval_native_kernel_with_vm(kernel, item, vm))
            .unwrap_or(Val::Null)
    }

    /// Returns the source-row field payload needed to evaluate every value entry.
    pub(crate) fn field_demand(&self) -> FieldDemand {
        self.entries.iter().fold(FieldDemand::None, |need, entry| {
            need.merge(entry.key.field_demand())
                .merge(entry.value.field_demand())
                .merge(
                    entry
                        .cond
                        .as_ref()
                        .map(BodyKernel::field_demand)
                        .unwrap_or(FieldDemand::None),
                )
        })
    }
}

impl ObjectKernelEntry {
    pub(crate) fn static_key(&self) -> Option<&Arc<str>> {
        match &self.key {
            ObjectKernelKey::Static(key) => Some(key),
            ObjectKernelKey::Dynamic(_) | ObjectKernelKey::Spread(_) => None,
        }
    }

    pub(crate) fn key(&self) -> &Arc<str> {
        self.static_key()
            .expect("dynamic object keys do not have a static key")
    }

    pub(crate) fn key_kernel(&self) -> &ObjectKernelKey {
        &self.key
    }

    pub(crate) fn value(&self) -> &BodyKernel {
        &self.value
    }

    pub(crate) fn cond(&self) -> Option<&BodyKernel> {
        self.cond.as_ref()
    }

    pub(crate) fn optional(&self) -> bool {
        self.optional
    }

    pub(crate) fn omit_null(&self) -> bool {
        self.omit_null
    }

    pub(crate) fn omits_null(&self) -> bool {
        self.optional || self.omit_null
    }
}

impl ObjectKernelKey {
    fn field_demand(&self) -> FieldDemand {
        match self {
            Self::Static(_) => FieldDemand::None,
            Self::Dynamic(kernel) => kernel.field_demand(),
            Self::Spread(_) => FieldDemand::None,
        }
    }

    fn is_view_native(&self) -> bool {
        match self {
            Self::Static(_) => true,
            Self::Dynamic(kernel) => kernel.is_view_native(),
            Self::Spread(_) => true,
        }
    }

    fn mentions_any_field_like_ident(&self, names: &[Arc<str>]) -> bool {
        match self {
            Self::Static(_) => false,
            Self::Dynamic(kernel) => kernel.mentions_any_field_like_ident(names),
            Self::Spread(_) => false,
        }
    }
}

fn classify_object_expr(fields: &[ObjField]) -> BodyKernel {
    let mut entries = Vec::with_capacity(fields.len());
    for field in fields {
        let (key, value, cond, optional, omit_null) = match field {
            ObjField::Short(name) => (
                ObjectKernelKey::Static(Arc::from(name.as_str())),
                BodyKernel::FieldRead(Arc::from(name.as_str())),
                None,
                false,
                true,
            ),
            ObjField::Kv {
                key,
                val,
                optional,
                cond,
            } => {
                let value = BodyKernel::classify_expr(val);
                if matches!(value, BodyKernel::Generic) {
                    return BodyKernel::Generic;
                }
                let cond = match cond {
                    Some(cond) => {
                        let cond = BodyKernel::classify_expr(cond);
                        if matches!(cond, BodyKernel::Generic) {
                            return BodyKernel::Generic;
                        }
                        Some(cond)
                    }
                    None => None,
                };
                (
                    ObjectKernelKey::Static(Arc::from(key.as_str())),
                    value,
                    cond,
                    *optional,
                    false,
                )
            }
            ObjField::Dynamic { key, val } => {
                let key = BodyKernel::classify_expr(key);
                let value = BodyKernel::classify_expr(val);
                if matches!(key, BodyKernel::Generic) || matches!(value, BodyKernel::Generic) {
                    return BodyKernel::Generic;
                }
                (ObjectKernelKey::Dynamic(key), value, None, false, false)
            }
            ObjField::Spread(value) => {
                let value = BodyKernel::classify_expr(value);
                if matches!(value, BodyKernel::Generic) {
                    return BodyKernel::Generic;
                }
                (
                    ObjectKernelKey::Spread(ObjectSpreadMode::Shallow),
                    value,
                    None,
                    false,
                    false,
                )
            }
            ObjField::SpreadDeep(value) => {
                let value = BodyKernel::classify_expr(value);
                if matches!(value, BodyKernel::Generic) {
                    return BodyKernel::Generic;
                }
                (
                    ObjectKernelKey::Spread(ObjectSpreadMode::Deep),
                    value,
                    None,
                    false,
                    false,
                )
            }
        };
        entries.push(ObjectKernelEntry {
            key,
            value,
            cond,
            optional,
            omit_null,
        });
    }
    BodyKernel::Object(ObjectKernel {
        entries: entries.into(),
    })
}

fn classify_array_expr(elems: &[crate::parse::ast::ArrayElem]) -> BodyKernel {
    let mut out = Vec::with_capacity(elems.len());
    let mut spread = false;
    for elem in elems {
        let (expr, is_spread) = match elem {
            crate::parse::ast::ArrayElem::Expr(expr) => (expr, false),
            crate::parse::ast::ArrayElem::Spread(expr) => {
                spread = true;
                (expr, true)
            }
        };
        let item = BodyKernel::classify_expr(expr);
        if matches!(item, BodyKernel::Generic) {
            return BodyKernel::Generic;
        }
        if is_spread {
            out.push(ArrayKernelElem::Spread(item));
        } else {
            out.push(ArrayKernelElem::Value(item));
        }
    }
    if spread {
        BodyKernel::ArraySpread(out.into())
    } else {
        BodyKernel::Array(
            out.into_iter()
                .map(|elem| match elem {
                    ArrayKernelElem::Value(kernel) => kernel,
                    ArrayKernelElem::Spread(_) => unreachable!("spread tracked above"),
                })
                .collect::<Vec<_>>()
                .into(),
        )
    }
}

fn classify_fstring_expr(parts: &[crate::parse::ast::FStringPart]) -> BodyKernel {
    let mut out = Vec::with_capacity(parts.len());
    let mut base_capacity = 0usize;
    for part in parts {
        match part {
            crate::parse::ast::FStringPart::Lit(value) => {
                base_capacity += value.len();
                out.push(FStringKernelPart::Lit(Arc::from(value.as_str())));
            }
            crate::parse::ast::FStringPart::Interp { expr, fmt: None } => {
                let kernel = BodyKernel::classify_expr(expr);
                if matches!(kernel, BodyKernel::Generic | BodyKernel::FString(_)) {
                    return BodyKernel::Generic;
                }
                base_capacity += 8;
                out.push(FStringKernelPart::Interp(kernel));
            }
            crate::parse::ast::FStringPart::Interp { .. } => return BodyKernel::Generic,
        }
    }
    BodyKernel::FString(FStringKernel {
        parts: out.into(),
        base_capacity,
    })
}

fn try_classify_nested_array_reducer(base: &Expr, steps: &[Step]) -> Option<BodyKernel> {
    let (last, prefix) = steps.split_last()?;
    let Step::Method(name, args) = last else {
        return None;
    };
    if !args.is_empty() {
        return None;
    }
    let id = builtin_by_name(name.as_str())?;
    let op = numeric_reducer(id).map(super::NumOp::from_builtin_reducer);

    let (source_steps, map) = match prefix.split_last() {
        Some((Step::Method(map_name, map_args), source_steps))
            if builtin_by_name(map_name.as_str()).and_then(expr_stage)
                == Some(BuiltinExprStage::Map) =>
        {
            let [crate::parse::ast::Arg::Pos(map_expr)] = map_args.as_slice() else {
                return None;
            };
            let map = BodyKernel::classify_expr(map_expr);
            if matches!(map, BodyKernel::Generic) {
                return None;
            }
            (source_steps, Some(Box::new(map)))
        }
        _ => (prefix, None),
    };
    let (source_steps, predicate) = match source_steps.split_last() {
        Some((Step::Method(filter_name, filter_args), source_steps))
            if builtin_by_name(filter_name.as_str()).and_then(expr_stage)
                == Some(BuiltinExprStage::Filter) =>
        {
            let [crate::parse::ast::Arg::Pos(filter_expr)] = filter_args.as_slice() else {
                return None;
            };
            let predicate = BodyKernel::classify_expr(filter_expr);
            if matches!(predicate, BodyKernel::Generic) {
                return None;
            }
            (source_steps, Some(Box::new(predicate)))
        }
        _ => (source_steps, None),
    };

    let source = if source_steps.is_empty() {
        BodyKernel::classify_expr(base)
    } else {
        classify_chain_expr(base, source_steps)
    };
    if matches!(source, BodyKernel::Generic) {
        return None;
    }
    match (id, op, map) {
        (_, Some(op), map) => Some(BodyKernel::NestedArrayReducer {
            source: Box::new(source),
            predicate,
            map,
            op,
        }),
        (id, None, None) if count_sink_accepts_predicate(id) => {
            Some(BodyKernel::NestedArrayCount {
                source: Box::new(source),
                predicate,
            })
        }
        _ => None,
    }
}

fn classify_chain_expr(base: &Expr, steps: &[Step]) -> BodyKernel {
    if let Some(kernel) = try_classify_nested_array_reducer(base, steps) {
        return kernel;
    }

    let nested_arg =
        crate::parse::ast::Arg::Pos(Expr::Chain(Box::new(base.clone()), steps.to_vec()));
    let mut receiver = match base {
        Expr::Current => BodyKernel::Current,
        Expr::Ident(name) => BodyKernel::FieldRead(Arc::from(name.as_str())),
        _ => return BodyKernel::Generic,
    };

    for step in steps {
        match step {
            Step::Field(key) => {
                receiver = match receiver {
                    BodyKernel::Current => BodyKernel::FieldRead(Arc::from(key.as_str())),
                    BodyKernel::FieldRead(first) => {
                        BodyKernel::FieldChain(vec![first, Arc::from(key.as_str())].into())
                    }
                    BodyKernel::FieldChain(keys) => {
                        let mut next = keys.to_vec();
                        next.push(Arc::from(key.as_str()));
                        BodyKernel::FieldChain(next.into())
                    }
                    other => BodyKernel::Compose {
                        first: Box::new(other),
                        then: Box::new(BodyKernel::FieldRead(Arc::from(key.as_str()))),
                    },
                };
            }
            Step::Method(name, args) => {
                let Some(call) = BuiltinCall::from_literal_ast_args(name.as_str(), args) else {
                    return super::lower::try_decode_map_body(&nested_arg)
                        .map(|plan| {
                            BodyKernel::NestedPlan(Arc::new(NestedPlanKernel::new(Arc::new(plan))))
                        })
                        .unwrap_or(BodyKernel::Generic);
                };
                let id = call.id();
                if let Some(selector) = array_selector_builtin_call(&call) {
                    if !receiver.is_view_native() {
                        return BodyKernel::Generic;
                    }
                    receiver = BodyKernel::ArraySelect {
                        array: Box::new(receiver),
                        selector,
                    };
                    continue;
                }
                if !view_projection(id) && !receiver.view_result_owned() {
                    return super::lower::try_decode_map_body(&nested_arg)
                        .map(|plan| {
                            BodyKernel::NestedPlan(Arc::new(NestedPlanKernel::new(Arc::new(plan))))
                        })
                        .unwrap_or(BodyKernel::Generic);
                }
                receiver = BodyKernel::BuiltinCall {
                    receiver: Box::new(receiver),
                    call,
                };
            }
            Step::Index(index) => {
                let selector = match *index {
                    -1 => ArraySelector::Last,
                    idx if idx >= 0 => ArraySelector::Nth(idx as usize),
                    _ => return BodyKernel::Generic,
                };
                if !receiver.is_view_native() {
                    return BodyKernel::Generic;
                }
                receiver = BodyKernel::ArraySelect {
                    array: Box::new(receiver),
                    selector,
                };
            }
            Step::DynIndex(index) => {
                let key = BodyKernel::classify_expr(index);
                if !receiver.is_view_native() || matches!(key, BodyKernel::Generic) {
                    return BodyKernel::Generic;
                }
                receiver = BodyKernel::DynIndex {
                    receiver: Box::new(receiver),
                    key: Box::new(key),
                };
            }
            Step::Slice(from, to, step) => {
                if !receiver.is_view_native() {
                    return BodyKernel::Generic;
                }
                receiver = BodyKernel::Slice {
                    array: Box::new(receiver),
                    from: *from,
                    to: *to,
                    step: *step,
                };
            }
            _ => return BodyKernel::Generic,
        }
    }

    receiver
}

fn array_selector_builtin_call(call: &BuiltinCall) -> Option<ArraySelector> {
    let index = match call.args {
        BuiltinArgs::I64(index) => Some(index),
        _ => None,
    };
    ArraySelector::from_builtin_selector(call.array_selector()?, index)
}

impl BodyKernel {
    /// Classifies an AST expression into the generic row-kernel IR when it is row-local and
    /// view-native. This complements opcode classification for object projections where the AST
    /// still carries useful structure such as shorthand fields and method-chain literals.
    pub(crate) fn classify_expr(expr: &Expr) -> Self {
        match expr {
            Expr::Current => Self::Current,
            Expr::Ident(name) => Self::FieldRead(Arc::from(name.as_str())),
            Expr::Null => Self::Const(Val::Null),
            Expr::Bool(value) => Self::ConstBool(*value),
            Expr::Int(value) => Self::Const(Val::Int(*value)),
            Expr::Float(value) => Self::Const(Val::Float(*value)),
            Expr::Str(value) => Self::Const(Val::Str(Arc::from(value.as_str()))),
            Expr::BinOp(lhs, op, rhs) => {
                let lhs = Self::classify_expr(lhs);
                let rhs = Self::classify_expr(rhs);
                if matches!(lhs, Self::Generic) || matches!(rhs, Self::Generic) {
                    Self::Generic
                } else if matches!(op, crate::parse::ast::BinOp::And) {
                    let mut predicates = Vec::new();
                    flatten_and_kernel(lhs, &mut predicates);
                    flatten_and_kernel(rhs, &mut predicates);
                    Self::And(predicates.into())
                } else if matches!(op, crate::parse::ast::BinOp::Or) {
                    let mut predicates = Vec::new();
                    flatten_or_kernel(lhs, &mut predicates);
                    flatten_or_kernel(rhs, &mut predicates);
                    Self::Or(predicates.into())
                } else if op.is_predicate_comparison() {
                    match literal_kernel_value(&rhs) {
                        Some(lit) => Self::CmpLit {
                            lhs: Box::new(lhs),
                            op: *op,
                            lit,
                        },
                        None => Self::Binary {
                            lhs: Box::new(lhs),
                            op: *op,
                            rhs: Box::new(rhs),
                        },
                    }
                } else {
                    Self::Binary {
                        lhs: Box::new(lhs),
                        op: *op,
                        rhs: Box::new(rhs),
                    }
                }
            }
            Expr::FString(parts) => classify_fstring_expr(parts),
            Expr::Object(fields) => classify_object_expr(fields),
            Expr::Array(elems) => classify_array_expr(elems),
            Expr::UnaryNeg(expr) => {
                let kernel = Self::classify_expr(expr);
                if matches!(kernel, Self::Generic) {
                    Self::Generic
                } else {
                    Self::Neg(Box::new(kernel))
                }
            }
            Expr::Not(expr) => {
                let kernel = Self::classify_expr(expr);
                if matches!(kernel, Self::Generic) {
                    Self::Generic
                } else {
                    Self::Not(Box::new(kernel))
                }
            }
            Expr::Kind { expr, ty, negate } => {
                let kernel = Self::classify_expr(expr);
                if matches!(kernel, Self::Generic) {
                    Self::Generic
                } else {
                    Self::KindCheck {
                        expr: Box::new(kernel),
                        ty: *ty,
                        negate: *negate,
                    }
                }
            }
            Expr::Cast { expr, ty } if safe_view_cast_type(*ty) => {
                let kernel = Self::classify_expr(expr);
                if matches!(kernel, Self::Generic) {
                    Self::Generic
                } else {
                    Self::Cast {
                        expr: Box::new(kernel),
                        ty: *ty,
                    }
                }
            }
            Expr::Coalesce(lhs, rhs) => {
                let lhs = Self::classify_expr(lhs);
                let rhs = Self::classify_expr(rhs);
                if matches!(lhs, Self::Generic) || matches!(rhs, Self::Generic) {
                    Self::Generic
                } else {
                    Self::Coalesce {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    }
                }
            }
            Expr::IfElse { cond, then_, else_ } => {
                let cond = Self::classify_expr(cond);
                let then_ = Self::classify_expr(then_);
                let else_ = Self::classify_expr(else_);
                if matches!(cond, Self::Generic)
                    || matches!(then_, Self::Generic)
                    || matches!(else_, Self::Generic)
                {
                    Self::Generic
                } else {
                    Self::IfElse {
                        cond: Box::new(cond),
                        then_: Box::new(then_),
                        else_: Box::new(else_),
                    }
                }
            }
            Expr::Chain(base, steps) => classify_chain_expr(base, steps),
            Expr::Match { .. } => {
                let program = crate::compile::compiler::Compiler::compile(expr, "<match-kernel>");
                Self::classify(&program)
            }
            _ => Self::Generic,
        }
    }

    /// Returns the field-only path read by this kernel, relative to the current
    /// row. `Current` is represented as an empty path.
    pub(crate) fn field_path_keys(&self) -> Option<Vec<Arc<str>>> {
        match self {
            Self::Current => Some(Vec::new()),
            Self::FieldRead(key) => Some(vec![Arc::clone(key)]),
            Self::FieldChain(keys) => Some(keys.iter().cloned().collect()),
            Self::BuiltinCall { receiver, call } => view_result_field_path(receiver, call),
            Self::Compose { first, then } => {
                let mut keys = first.field_path_keys()?;
                keys.extend(then.field_path_keys()?);
                Some(keys)
            }
            _ => None,
        }
    }

    /// Returns a constant value produced by this kernel, if the kernel is a
    /// literal-only expression.
    pub(crate) fn literal_value(&self) -> Option<Val> {
        match self {
            Self::Const(value) => Some(value.clone()),
            Self::ConstBool(value) => Some(Val::Bool(*value)),
            _ => None,
        }
    }

    /// Returns the truthiness of a literal-only kernel.
    pub(crate) fn constant_truthy(&self) -> Option<bool> {
        match self {
            Self::ConstBool(value) => Some(*value),
            Self::Const(value) => Some(crate::util::is_truthy(value)),
            _ => None,
        }
    }

    /// Returns a direct field-path-to-literal comparison when this kernel is
    /// exactly that shape.
    pub(crate) fn field_path_literal_cmp(
        &self,
    ) -> Option<(Vec<Arc<str>>, crate::parse::ast::BinOp, Val)> {
        match self {
            Self::FieldCmpLit(field, op, lit) => Some((vec![Arc::clone(field)], *op, lit.clone())),
            Self::FieldChainCmpLit(keys, op, lit) => {
                Some((keys.iter().cloned().collect(), *op, lit.clone()))
            }
            Self::CurrentCmpLit(op, lit) => Some((Vec::new(), *op, lit.clone())),
            Self::CmpLit { lhs, op, lit } => Some((lhs.field_path_keys()?, *op, lit.clone())),
            _ => None,
        }
    }

    /// Returns metadata for `field_path.view_scalar_call()` kernels.
    pub(crate) fn path_scalar_call(&self) -> Option<PathScalarCall> {
        let Self::BuiltinCall { receiver, call } = self else {
            return None;
        };
        if !call.is_direct_view_scalar_call() {
            return None;
        }
        Some(PathScalarCall {
            receiver_keys: receiver.field_path_keys()?,
            call: call.clone(),
        })
    }

    /// Returns metadata for `array_selector[.suffix].view_scalar_call()` kernels.
    pub(crate) fn array_element_scalar_call(&self) -> Option<ArrayElementScalarCall> {
        let Self::BuiltinCall { receiver, call } = self else {
            return None;
        };
        if !call.is_direct_view_scalar_call() {
            return None;
        }
        let (source_keys, selector, suffix_keys) = receiver.array_element_path()?;
        Some(ArrayElementScalarCall {
            source_keys,
            selector,
            suffix_keys,
            call: call.clone(),
        })
    }

    /// Returns metadata for direct array element path kernels.
    pub(crate) fn array_element_path(
        &self,
    ) -> Option<(Vec<Arc<str>>, ArraySelector, Vec<Arc<str>>)> {
        match self {
            Self::ArraySelect { array, selector } => {
                Some((array.field_path_keys()?, *selector, Vec::new()))
            }
            Self::Compose { first, then } => {
                let (source_keys, selector, mut suffix_keys) = first.array_element_path()?;
                suffix_keys.extend(then.field_path_keys()?);
                Some((source_keys, selector, suffix_keys))
            }
            _ => None,
        }
    }

    /// Returns the field payload needed from the current row to evaluate this kernel.
    pub(crate) fn field_demand(&self) -> FieldDemand {
        match self {
            Self::Generic | Self::Current | Self::CurrentCmpLit(_, _) => FieldDemand::Whole,
            Self::FieldRead(key) | Self::FieldCmpLit(key, _, _) => {
                FieldDemand::Fields(FieldSet::single(Arc::clone(key)))
            }
            Self::FieldChain(keys) | Self::FieldChainCmpLit(keys, _, _) => {
                FieldDemand::Fields(FieldSet::chain(Arc::clone(keys)))
            }
            Self::BuiltinCall { receiver, call } => object_key_call_field_demand(receiver, call)
                .unwrap_or_else(|| receiver.field_demand()),
            Self::Compose { first, then } => compose_field_demand(first, then),
            Self::CmpLit { lhs, .. } => lhs.field_demand(),
            Self::Binary { lhs, rhs, .. } => lhs.field_demand().merge(rhs.field_demand()),
            Self::ArraySelect { array, .. } => array.field_demand(),
            Self::Slice { array, .. } => array.field_demand(),
            Self::DynIndex { receiver, key } => receiver.field_demand().merge(key.field_demand()),
            Self::Match { .. } => FieldDemand::Whole,
            Self::And(predicates) | Self::Or(predicates) => predicates
                .iter()
                .fold(FieldDemand::None, |need, predicate| {
                    need.merge(predicate.field_demand())
                }),
            Self::Neg(kernel) => kernel.field_demand(),
            Self::Not(kernel) => kernel.field_demand(),
            Self::KindCheck { expr, .. } => expr.field_demand(),
            Self::Cast { expr, .. } => expr.field_demand(),
            Self::Coalesce { lhs, rhs } => lhs.field_demand().merge(rhs.field_demand()),
            Self::IfElse { cond, then_, else_ } => cond
                .field_demand()
                .merge(then_.field_demand())
                .merge(else_.field_demand()),
            Self::FString(fstring) => {
                fstring
                    .parts
                    .iter()
                    .fold(FieldDemand::None, |need, part| match part {
                        FStringKernelPart::Lit(_) => need,
                        FStringKernelPart::Interp(kernel) => need.merge(kernel.field_demand()),
                    })
            }
            Self::Object(object) => object.field_demand(),
            Self::Array(items) => items.iter().fold(FieldDemand::None, |need, item| {
                need.merge(item.field_demand())
            }),
            Self::ArraySpread(items) => items.iter().fold(FieldDemand::None, |need, item| {
                need.merge(item.kernel().field_demand())
            }),
            Self::NestedArrayReducer {
                source,
                predicate,
                map,
                ..
            } => nested_array_field_demand(source, predicate.as_deref(), map.as_deref()),
            Self::NestedArrayCount { source, predicate } => {
                nested_array_field_demand(source, predicate.as_deref(), None)
            }
            Self::NestedPlan(plan) => plan.parent_field_demand(),
            Self::ConstBool(_) | Self::Const(_) => FieldDemand::None,
        }
    }

    /// Returns `true` when this kernel references any name in `names` as a field-like access.
    pub(crate) fn mentions_any_field_like_ident(&self, names: &[Arc<str>]) -> bool {
        fn matches_name(name: &str, names: &[Arc<str>]) -> bool {
            names.iter().any(|candidate| candidate.as_ref() == name)
        }

        match self {
            Self::FieldRead(name) | Self::FieldCmpLit(name, _, _) => {
                matches_name(name.as_ref(), names)
            }
            Self::FieldChain(keys) | Self::FieldChainCmpLit(keys, _, _) => keys
                .first()
                .is_some_and(|name| matches_name(name.as_ref(), names)),
            Self::BuiltinCall { receiver, .. } => receiver.mentions_any_field_like_ident(names),
            Self::Compose { first, then } => {
                first.mentions_any_field_like_ident(names)
                    || then.mentions_any_field_like_ident(names)
            }
            Self::CmpLit { lhs, .. } => lhs.mentions_any_field_like_ident(names),
            Self::Binary { lhs, rhs, .. } => {
                lhs.mentions_any_field_like_ident(names) || rhs.mentions_any_field_like_ident(names)
            }
            Self::ArraySelect { array, .. } => array.mentions_any_field_like_ident(names),
            Self::Slice { array, .. } => array.mentions_any_field_like_ident(names),
            Self::DynIndex { receiver, key } => {
                receiver.mentions_any_field_like_ident(names)
                    || key.mentions_any_field_like_ident(names)
            }
            Self::Match { scrutinee, .. } => scrutinee.mentions_any_field_like_ident(names),
            Self::And(predicates) | Self::Or(predicates) => predicates
                .iter()
                .any(|predicate| predicate.mentions_any_field_like_ident(names)),
            Self::Neg(kernel) => kernel.mentions_any_field_like_ident(names),
            Self::Not(kernel) => kernel.mentions_any_field_like_ident(names),
            Self::KindCheck { expr, .. } => expr.mentions_any_field_like_ident(names),
            Self::Cast { expr, .. } => expr.mentions_any_field_like_ident(names),
            Self::Coalesce { lhs, rhs } => {
                lhs.mentions_any_field_like_ident(names) || rhs.mentions_any_field_like_ident(names)
            }
            Self::IfElse { cond, then_, else_ } => {
                cond.mentions_any_field_like_ident(names)
                    || then_.mentions_any_field_like_ident(names)
                    || else_.mentions_any_field_like_ident(names)
            }
            Self::FString(fstring) => fstring.parts.iter().any(|part| match part {
                FStringKernelPart::Lit(_) => false,
                FStringKernelPart::Interp(kernel) => kernel.mentions_any_field_like_ident(names),
            }),
            Self::Object(object) => object.entries.iter().any(|entry| {
                entry.key.mentions_any_field_like_ident(names)
                    || entry.value.mentions_any_field_like_ident(names)
                    || entry
                        .cond
                        .as_ref()
                        .is_some_and(|cond| cond.mentions_any_field_like_ident(names))
            }),
            Self::Array(items) => items
                .iter()
                .any(|item| item.mentions_any_field_like_ident(names)),
            Self::ArraySpread(items) => items
                .iter()
                .any(|item| item.kernel().mentions_any_field_like_ident(names)),
            Self::NestedArrayReducer {
                source,
                predicate,
                map,
                ..
            } => {
                source.mentions_any_field_like_ident(names)
                    || predicate
                        .as_ref()
                        .is_some_and(|predicate| predicate.mentions_any_field_like_ident(names))
                    || map
                        .as_ref()
                        .is_some_and(|map| map.mentions_any_field_like_ident(names))
            }
            Self::NestedArrayCount { source, predicate } => {
                source.mentions_any_field_like_ident(names)
                    || predicate
                        .as_ref()
                        .is_some_and(|predicate| predicate.mentions_any_field_like_ident(names))
            }
            Self::NestedPlan(_) => true,
            Self::Generic
            | Self::Current
            | Self::CurrentCmpLit(_, _)
            | Self::ConstBool(_)
            | Self::Const(_) => false,
        }
    }

    /// Returns `true` when the kernel can operate entirely on borrowed `ValueView` without materialising.
    pub(crate) fn is_view_native(&self) -> bool {
        match self {
            Self::Generic => false,
            Self::BuiltinCall { receiver, call } => {
                receiver.is_view_native()
                    && (call.is_view_projection() || receiver.view_result_owned())
            }
            Self::Compose { first, then } => first.is_view_native() && then.is_view_native(),
            Self::CmpLit { lhs, .. } => lhs.is_view_native(),
            Self::Binary { lhs, rhs, .. } => lhs.is_view_native() && rhs.is_view_native(),
            Self::ArraySelect { array, .. } => array.is_view_native(),
            Self::Slice { array, .. } => array.is_view_native(),
            Self::DynIndex { receiver, key } => receiver.is_view_native() && key.is_view_native(),
            Self::Match { scrutinee, .. } => scrutinee.is_view_native(),
            Self::And(predicates) | Self::Or(predicates) => {
                predicates.iter().all(Self::is_view_native)
            }
            Self::Neg(kernel) => kernel.is_view_native(),
            Self::Not(kernel) => kernel.is_view_native(),
            Self::KindCheck { expr, .. } => expr.is_view_native(),
            Self::Cast { expr, .. } => expr.is_view_native(),
            Self::Coalesce { lhs, rhs } => lhs.is_view_native() && rhs.is_view_native(),
            Self::IfElse { cond, then_, else_ } => {
                cond.is_view_native() && then_.is_view_native() && else_.is_view_native()
            }
            Self::Object(object) => object.entries.iter().all(|entry| {
                entry.key.is_view_native()
                    && entry.value.is_view_native()
                    && entry.cond.as_ref().is_none_or(BodyKernel::is_view_native)
            }),
            Self::Array(items) => items.iter().all(Self::is_view_native),
            Self::ArraySpread(items) => items.iter().all(|item| item.kernel().is_view_native()),
            Self::NestedArrayReducer {
                source,
                predicate,
                map,
                ..
            } => {
                source.is_view_native()
                    && predicate
                        .as_ref()
                        .is_none_or(|predicate| predicate.is_view_native())
                    && map.as_ref().is_none_or(|map| map.is_view_native())
            }
            Self::NestedArrayCount { source, predicate } => {
                source.is_view_native()
                    && predicate
                        .as_ref()
                        .is_none_or(|predicate| predicate.is_view_native())
            }
            Self::NestedPlan(plan) => plan
                .view_plan()
                .is_some_and(|plan| plan.body().can_run_with_view()),
            _ => true,
        }
    }

    pub(crate) fn view_result_owned(&self) -> bool {
        match self {
            Self::BuiltinCall { receiver, call } => {
                receiver.is_view_native() && call.view_projection_returns_owned()
            }
            Self::Compose { first, then } => first.is_view_native() && then.view_result_owned(),
            Self::ConstBool(_)
            | Self::Const(_)
            | Self::FString(_)
            | Self::Object(_)
            | Self::Array(_)
            | Self::ArraySpread(_)
            | Self::NestedArrayReducer { .. }
            | Self::NestedArrayCount { .. }
            | Self::NestedPlan(_)
            | Self::CmpLit { .. }
            | Self::Binary { .. }
            | Self::ArraySelect { .. }
            | Self::Slice { .. }
            | Self::Match { .. }
            | Self::And(_)
            | Self::Or(_)
            | Self::Neg(_)
            | Self::Not(_)
            | Self::KindCheck { .. }
            | Self::Cast { .. }
            | Self::IfElse { .. }
            | Self::CurrentCmpLit(_, _)
            | Self::FieldCmpLit(_, _, _)
            | Self::FieldChainCmpLit(_, _, _) => true,
            Self::Coalesce { lhs, rhs } => lhs.view_result_owned() && rhs.view_result_owned(),
            Self::Current
            | Self::FieldRead(_)
            | Self::FieldChain(_)
            | Self::DynIndex { .. }
            | Self::Generic => false,
        }
    }

    /// Returns the `CollectLayout` hint indicating whether outputs form a uniform-object or generic collection.
    pub(crate) fn collect_layout(&self) -> CollectLayout<'_> {
        match self {
            Self::Object(object) if object.has_static_layout() => {
                CollectLayout::UniformObject(object)
            }
            _ => CollectLayout::Values,
        }
    }

    /// Classifies a compiled `Program` into the most specific `BodyKernel` variant, falling back to `Generic`.
    pub fn classify(prog: &crate::vm::Program) -> Self {
        use crate::vm::Opcode;
        let ops = prog.ops.as_ref();
        if ops
            .iter()
            .any(|op| matches!(op, Opcode::BindLamCurrent { .. }))
        {
            return Self::Generic;
        }
        if ops.len() == 1 {
            if let Some(lit) = trivial_lit(&ops[0]) {
                return match &ops[0] {
                    Opcode::PushBool(b) => Self::ConstBool(*b),
                    _ => Self::Const(lit),
                };
            }
        }
        match ops {
            [Opcode::Match(cm)] if match_is_receiver_local(cm) => {
                let scrutinee = match &cm.scrutinee {
                    crate::vm::MatchScrutinee::Current => BodyKernel::Current,
                    crate::vm::MatchScrutinee::Program(program) => BodyKernel::classify(program),
                    crate::vm::MatchScrutinee::Root => BodyKernel::Generic,
                };
                if !matches!(scrutinee, BodyKernel::Generic) && scrutinee.is_view_native() {
                    return Self::Match {
                        scrutinee: Box::new(scrutinee),
                        compiled: Arc::clone(cm),
                        body_needs_current: match_bodies_need_current(cm),
                    };
                }
            }
            [receiver @ .., Opcode::CallMethod(call)] if array_selector_call(call).is_some() => {
                let array = if receiver.is_empty() {
                    BodyKernel::Current
                } else {
                    BodyKernel::classify(&crate::vm::Program::new(
                        receiver.to_vec(),
                        "<array-select-receiver>",
                    ))
                };
                if !matches!(array, BodyKernel::Generic) && array.is_view_native() {
                    return Self::ArraySelect {
                        array: Box::new(array),
                        selector: array_selector_call(call).expect("selector checked"),
                    };
                }
            }
            [receiver @ .., Opcode::DynIndex(key)] => {
                let receiver = if receiver.is_empty() {
                    BodyKernel::Current
                } else {
                    BodyKernel::classify(&crate::vm::Program::new(
                        receiver.to_vec(),
                        "<dynamic-index-receiver>",
                    ))
                };
                let key = BodyKernel::classify(key);
                if !matches!(receiver, BodyKernel::Generic)
                    && !matches!(key, BodyKernel::Generic)
                    && receiver.is_view_native()
                    && key.is_view_native()
                {
                    return Self::DynIndex {
                        receiver: Box::new(receiver),
                        key: Box::new(key),
                    };
                }
            }
            [receiver @ .., Opcode::GetSlice(from, to, step)] => {
                let array = if receiver.is_empty() {
                    BodyKernel::Current
                } else {
                    BodyKernel::classify(&crate::vm::Program::new(
                        receiver.to_vec(),
                        "<slice-receiver>",
                    ))
                };
                if !matches!(array, BodyKernel::Generic) && array.is_view_native() {
                    return Self::Slice {
                        array: Box::new(array),
                        from: *from,
                        to: *to,
                        step: *step,
                    };
                }
            }
            [Opcode::MakeObj(entries)] => {
                let mut out = Vec::with_capacity(entries.len());
                for entry in entries.iter() {
                    let (key, value, cond, optional, omit_null) = match entry {
                        crate::vm::CompiledObjEntry::Short { name, .. } => (
                            ObjectKernelKey::Static(Arc::clone(name)),
                            BodyKernel::FieldRead(Arc::clone(name)),
                            None,
                            false,
                            true,
                        ),
                        crate::vm::CompiledObjEntry::Kv {
                            key,
                            prog,
                            optional,
                            cond,
                        } => {
                            let value = BodyKernel::classify(prog);
                            if matches!(value, BodyKernel::Generic) {
                                return Self::Generic;
                            }
                            let cond = match cond {
                                Some(cond) => {
                                    let cond = BodyKernel::classify(cond);
                                    if matches!(cond, BodyKernel::Generic) {
                                        return Self::Generic;
                                    }
                                    Some(cond)
                                }
                                None => None,
                            };
                            (
                                ObjectKernelKey::Static(Arc::clone(key)),
                                value,
                                cond,
                                *optional,
                                false,
                            )
                        }
                        crate::vm::CompiledObjEntry::KvPath {
                            key,
                            steps,
                            optional,
                            ..
                        } => {
                            let Some(value) = classify_kv_path(steps) else {
                                return Self::Generic;
                            };
                            (
                                ObjectKernelKey::Static(Arc::clone(key)),
                                value,
                                None,
                                *optional,
                                false,
                            )
                        }
                        crate::vm::CompiledObjEntry::Dynamic { key, val } => {
                            let key = BodyKernel::classify(key);
                            let value = BodyKernel::classify(val);
                            if matches!(key, BodyKernel::Generic)
                                || matches!(value, BodyKernel::Generic)
                            {
                                return Self::Generic;
                            }
                            (ObjectKernelKey::Dynamic(key), value, None, false, false)
                        }
                        crate::vm::CompiledObjEntry::Spread(prog) => {
                            let value = BodyKernel::classify(prog);
                            if matches!(value, BodyKernel::Generic) {
                                return Self::Generic;
                            }
                            (
                                ObjectKernelKey::Spread(ObjectSpreadMode::Shallow),
                                value,
                                None,
                                false,
                                false,
                            )
                        }
                        crate::vm::CompiledObjEntry::SpreadDeep(prog) => {
                            let value = BodyKernel::classify(prog);
                            if matches!(value, BodyKernel::Generic) {
                                return Self::Generic;
                            }
                            (
                                ObjectKernelKey::Spread(ObjectSpreadMode::Deep),
                                value,
                                None,
                                false,
                                false,
                            )
                        }
                    };
                    out.push(ObjectKernelEntry {
                        key,
                        value,
                        cond,
                        optional,
                        omit_null,
                    });
                }
                return Self::Object(ObjectKernel {
                    entries: out.into(),
                });
            }
            [Opcode::MakeArr(items)] => {
                let mut out = Vec::with_capacity(items.len());
                let mut has_spread = false;
                for (program, spread) in items.iter() {
                    let item = BodyKernel::classify(program);
                    if matches!(item, BodyKernel::Generic) {
                        return Self::Generic;
                    }
                    if *spread {
                        has_spread = true;
                        out.push(ArrayKernelElem::Spread(item));
                    } else {
                        out.push(ArrayKernelElem::Value(item));
                    }
                }
                return if has_spread {
                    Self::ArraySpread(out.into())
                } else {
                    Self::Array(
                        out.into_iter()
                            .map(|elem| match elem {
                                ArrayKernelElem::Value(kernel) => kernel,
                                ArrayKernelElem::Spread(_) => unreachable!("spread tracked above"),
                            })
                            .collect::<Vec<_>>()
                            .into(),
                    )
                };
            }
            [Opcode::FString(parts)] => {
                let mut out = Vec::with_capacity(parts.len());
                let mut base_capacity = 0usize;
                for part in parts.iter() {
                    match part {
                        crate::vm::CompiledFSPart::Lit(value) => {
                            base_capacity += value.len();
                            out.push(FStringKernelPart::Lit(Arc::clone(value)));
                        }
                        crate::vm::CompiledFSPart::Interp { prog, fmt } if fmt.is_none() => {
                            let kernel = BodyKernel::classify(prog);
                            if matches!(kernel, BodyKernel::Generic | BodyKernel::FString(_)) {
                                return Self::Generic;
                            }
                            base_capacity += 8;
                            out.push(FStringKernelPart::Interp(kernel));
                        }
                        crate::vm::CompiledFSPart::Interp { .. } => return Self::Generic,
                    }
                }
                return Self::FString(FStringKernel {
                    parts: out.into(),
                    base_capacity,
                });
            }
            [body @ .., Opcode::Not] => {
                let body =
                    BodyKernel::classify(&crate::vm::Program::new(body.to_vec(), "<not-kernel>"));
                if !matches!(body, BodyKernel::Generic) && body.is_view_native() {
                    return Self::Not(Box::new(body));
                }
            }
            [body @ .., Opcode::Neg] => {
                let body =
                    BodyKernel::classify(&crate::vm::Program::new(body.to_vec(), "<neg-kernel>"));
                if !matches!(body, BodyKernel::Generic) && body.is_view_native() {
                    return Self::Neg(Box::new(body));
                }
            }
            [body @ .., Opcode::KindCheck { ty, negate }] => {
                let body = BodyKernel::classify(&crate::vm::Program::new(
                    body.to_vec(),
                    "<kind-check-kernel>",
                ));
                if !matches!(body, BodyKernel::Generic) && body.is_view_native() {
                    return Self::KindCheck {
                        expr: Box::new(body),
                        ty: *ty,
                        negate: *negate,
                    };
                }
            }
            [body @ .., Opcode::CastOp(ty)] if safe_view_cast_type(*ty) => {
                let body =
                    BodyKernel::classify(&crate::vm::Program::new(body.to_vec(), "<cast-kernel>"));
                if !matches!(body, BodyKernel::Generic) && body.is_view_native() {
                    return Self::Cast {
                        expr: Box::new(body),
                        ty: *ty,
                    };
                }
            }
            [lhs @ .., Opcode::CoalesceOp(rhs)] => {
                let lhs = BodyKernel::classify(&crate::vm::Program::new(
                    lhs.to_vec(),
                    "<coalesce-lhs>",
                ));
                let rhs = BodyKernel::classify(rhs);
                if !matches!(lhs, BodyKernel::Generic)
                    && !matches!(rhs, BodyKernel::Generic)
                    && lhs.is_view_native()
                    && rhs.is_view_native()
                {
                    return Self::Coalesce {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    };
                }
            }
            [cond @ .., Opcode::IfElse { then_, else_ }] => {
                let cond =
                    BodyKernel::classify(&crate::vm::Program::new(cond.to_vec(), "<if-cond>"));
                let then_ = BodyKernel::classify(then_);
                let else_ = BodyKernel::classify(else_);
                if !matches!(cond, BodyKernel::Generic)
                    && !matches!(then_, BodyKernel::Generic)
                    && !matches!(else_, BodyKernel::Generic)
                    && cond.is_view_native()
                    && then_.is_view_native()
                    && else_.is_view_native()
                {
                    return Self::IfElse {
                        cond: Box::new(cond),
                        then_: Box::new(then_),
                        else_: Box::new(else_),
                    };
                }
            }
            [Opcode::PushCurrent, Opcode::GetField(k)]
            | [Opcode::GetField(k)]
            | [Opcode::LoadIdent(k)] => return Self::FieldRead(k.clone()),
            [Opcode::PushCurrent, Opcode::FieldChain(fc)] | [Opcode::FieldChain(fc)] => {
                return Self::FieldChain(fc.keys.clone())
            }
            [Opcode::LoadIdent(k1), rest @ ..]
                if rest.iter().all(|o| matches!(o, Opcode::GetField(_))) =>
            {
                let mut keys = vec![k1.clone()];
                for o in rest {
                    if let Opcode::GetField(k) = o {
                        keys.push(k.clone());
                    }
                }
                return Self::FieldChain(keys.into());
            }
            [Opcode::LoadIdent(k1), Opcode::FieldChain(fc)] => {
                let mut keys = vec![k1.clone()];
                for k in fc.keys.iter() {
                    keys.push(k.clone());
                }
                return Self::FieldChain(keys.into());
            }
            _ => {}
        }
        let rest: &[Opcode] = if matches!(ops.first(), Some(Opcode::PushCurrent)) {
            &ops[1..]
        } else {
            ops
        };
        if rest.len() == 3 {
            if matches!(&rest[0], Opcode::PushCurrent) {
                if let Some(lit) = trivial_lit(&rest[1]) {
                    if let Some(bo) = cmp_to_binop(&rest[2]) {
                        return Self::CurrentCmpLit(bo, lit);
                    }
                }
            }
            let single_key = match &rest[0] {
                Opcode::LoadIdent(k) | Opcode::GetField(k) => Some(k.clone()),
                _ => None,
            };
            if let Some(k) = single_key {
                if let Some(lit) = trivial_lit(&rest[1]) {
                    if let Some(bo) = cmp_to_binop(&rest[2]) {
                        return Self::FieldCmpLit(k, bo, lit);
                    }
                }
            }
            if let Opcode::FieldChain(fc) = &rest[0] {
                if let Some(lit) = trivial_lit(&rest[1]) {
                    if let Some(bo) = cmp_to_binop(&rest[2]) {
                        return Self::FieldChainCmpLit(fc.keys.clone(), bo, lit);
                    }
                }
            }
            if let Some(op) = arithmetic_binop(&rest[2]) {
                if let (Some(lhs), Some(rhs)) = (
                    classify_structural_view_kernel(&rest[..1]),
                    classify_structural_view_kernel(&rest[1..2]),
                ) {
                    return Self::Binary {
                        lhs: Box::new(lhs),
                        op,
                        rhs: Box::new(rhs),
                    };
                }
            }
        }
        if let Some(kernel) = classify_structural_view_kernel(rest) {
            return kernel;
        }
        if let Some(kernel) = classify_rpn_structural_kernel(rest) {
            return kernel;
        }
        if let Some(kernel) = classify_and_kernel(ops) {
            return kernel;
        }
        if let Some(kernel) = classify_or_kernel(ops) {
            return kernel;
        }
        Self::Generic
    }
}

fn literal_kernel_value(kernel: &BodyKernel) -> Option<Val> {
    kernel.literal_value()
}

// returns None rather than wrapping a Generic sub-kernel, which would defeat specialisation
fn classify_and_kernel(ops: &[crate::vm::Opcode]) -> Option<BodyKernel> {
    let (lhs_ops, rhs) = match ops {
        [lhs @ .., crate::vm::Opcode::AndOp(rhs)] if !lhs.is_empty() => (lhs, rhs),
        _ => return None,
    };
    let lhs_prog = crate::vm::Program::new(lhs_ops.to_vec(), "<pipeline-and-lhs>");
    let lhs = BodyKernel::classify(&lhs_prog);
    let rhs = BodyKernel::classify(rhs);
    if matches!(lhs, BodyKernel::Generic) || matches!(rhs, BodyKernel::Generic) {
        return None;
    }
    let mut predicates = Vec::new();
    flatten_and_kernel(lhs, &mut predicates);
    flatten_and_kernel(rhs, &mut predicates);
    Some(BodyKernel::And(predicates.into()))
}

fn flatten_and_kernel(kernel: BodyKernel, out: &mut Vec<BodyKernel>) {
    match kernel {
        BodyKernel::And(predicates) => out.extend(predicates.iter().cloned()),
        other => out.push(other),
    }
}

// returns None rather than wrapping a Generic sub-kernel, which would defeat specialisation
fn classify_or_kernel(ops: &[crate::vm::Opcode]) -> Option<BodyKernel> {
    let (lhs_ops, rhs) = match ops {
        [lhs @ .., crate::vm::Opcode::OrOp(rhs)] if !lhs.is_empty() => (lhs, rhs),
        _ => return None,
    };
    let lhs_prog = crate::vm::Program::new(lhs_ops.to_vec(), "<pipeline-or-lhs>");
    let lhs = BodyKernel::classify(&lhs_prog);
    let rhs = BodyKernel::classify(rhs);
    if matches!(lhs, BodyKernel::Generic) || matches!(rhs, BodyKernel::Generic) {
        return None;
    }
    let mut predicates = Vec::new();
    flatten_or_kernel(lhs, &mut predicates);
    flatten_or_kernel(rhs, &mut predicates);
    Some(BodyKernel::Or(predicates.into()))
}

fn flatten_or_kernel(kernel: BodyKernel, out: &mut Vec<BodyKernel>) {
    match kernel {
        BodyKernel::Or(predicates) => out.extend(predicates.iter().cloned()),
        other => out.push(other),
    }
}

fn match_is_receiver_local(cm: &crate::vm::CompiledMatch) -> bool {
    match_is_materialized_env_local(cm, false)
}

fn match_is_materialized_env_local(cm: &crate::vm::CompiledMatch, allow_root: bool) -> bool {
    let scrutinee_ok = match cm.scrutinee {
        crate::vm::MatchScrutinee::Current => true,
        crate::vm::MatchScrutinee::Root => allow_root,
        crate::vm::MatchScrutinee::Program(_) => true,
    };
    scrutinee_ok
        && cm
            .guards
            .iter()
            .all(|program| program_ops_are_materialized_env_local(program, allow_root))
        && cm
            .bodies
            .iter()
            .all(|program| program_ops_are_materialized_env_local(program, allow_root))
}

fn match_bodies_need_current(cm: &crate::vm::CompiledMatch) -> bool {
    cm.bodies
        .iter()
        .any(|program| program_uses_current(program))
}

fn program_uses_current(program: &crate::vm::Program) -> bool {
    program.ops.iter().any(opcode_uses_current)
}

fn opcode_uses_current(opcode: &crate::vm::Opcode) -> bool {
    use crate::vm::Opcode;
    match opcode {
        Opcode::PushCurrent | Opcode::SetCurrent | Opcode::BindLamCurrent { .. } => true,
        Opcode::DynIndex(prog)
        | Opcode::InlineFilter(prog)
        | Opcode::AndOp(prog)
        | Opcode::OrOp(prog)
        | Opcode::CoalesceOp(prog) => program_uses_current(prog),
        Opcode::CallMethod(call) | Opcode::CallOptMethod(call) => {
            call.sub_progs.iter().any(|prog| program_uses_current(prog))
        }
        Opcode::MakeObj(entries) => entries.iter().any(|entry| match entry {
            crate::vm::CompiledObjEntry::Short { .. }
            | crate::vm::CompiledObjEntry::KvPath { .. } => true,
            crate::vm::CompiledObjEntry::Kv { prog, cond, .. } => {
                program_uses_current(prog)
                    || cond.as_ref().is_some_and(|cond| program_uses_current(cond))
            }
            crate::vm::CompiledObjEntry::Dynamic { key, val } => {
                program_uses_current(key) || program_uses_current(val)
            }
            crate::vm::CompiledObjEntry::Spread(prog)
            | crate::vm::CompiledObjEntry::SpreadDeep(prog) => program_uses_current(prog),
        }),
        Opcode::MakeArr(items) => items
            .iter()
            .any(|(program, _)| program_uses_current(program)),
        Opcode::FString(parts) => parts.iter().any(|part| match part {
            crate::vm::CompiledFSPart::Lit(_) => false,
            crate::vm::CompiledFSPart::Interp { prog, .. } => program_uses_current(prog),
        }),
        Opcode::LetExpr { body, .. } => program_uses_current(body),
        Opcode::IfElse { then_, else_ } => {
            program_uses_current(then_) || program_uses_current(else_)
        }
        Opcode::TryExpr { body, default } => {
            program_uses_current(body) || program_uses_current(default)
        }
        Opcode::Match(cm) | Opcode::DeepMatchAll(cm) | Opcode::DeepMatchFirst(cm) => {
            matches!(cm.scrutinee, crate::vm::MatchScrutinee::Current)
                || cm
                    .guards
                    .iter()
                    .any(|program| program_uses_current(program))
                || cm
                    .bodies
                    .iter()
                    .any(|program| program_uses_current(program))
        }
        _ => false,
    }
}

pub(super) fn program_is_receiver_local(program: &crate::vm::Program) -> bool {
    program
        .ops
        .iter()
        .all(|opcode| opcode_is_materialized_env_local(opcode, false))
}

pub(super) fn program_is_materialized_env_local(program: &crate::vm::Program) -> bool {
    program
        .ops
        .iter()
        .all(|opcode| opcode_is_materialized_env_local(opcode, true))
}

fn opcode_is_materialized_env_local(opcode: &crate::vm::Opcode, allow_root: bool) -> bool {
    use crate::vm::Opcode;
    match opcode {
        Opcode::PushRoot | Opcode::RootChain(_) => allow_root,
        Opcode::PipelineRun { .. }
        | Opcode::ListComp(_)
        | Opcode::DictComp(_)
        | Opcode::SetComp(_)
        | Opcode::PatchEval(_)
        | Opcode::UpdateBatchEval(_)
        | Opcode::DeepMatchAll(_)
        | Opcode::DeepMatchFirst(_) => false,
        Opcode::Match(cm) => match_is_materialized_env_local(cm, allow_root),
        Opcode::DynIndex(prog)
        | Opcode::InlineFilter(prog)
        | Opcode::AndOp(prog)
        | Opcode::OrOp(prog)
        | Opcode::CoalesceOp(prog) => program_ops_are_materialized_env_local(prog, allow_root),
        Opcode::BindLamCurrent { body, .. } => {
            program_ops_are_materialized_env_local(body, allow_root)
        }
        Opcode::CallMethod(call) | Opcode::CallOptMethod(call) => call
            .sub_progs
            .iter()
            .all(|prog| program_ops_are_materialized_env_local(prog, allow_root)),
        Opcode::MakeObj(entries) => entries.iter().all(|entry| match entry {
            crate::vm::CompiledObjEntry::Short { .. }
            | crate::vm::CompiledObjEntry::KvPath { .. } => true,
            crate::vm::CompiledObjEntry::Kv { prog, cond, .. } => {
                program_ops_are_materialized_env_local(prog, allow_root)
                    && cond
                        .as_ref()
                        .is_none_or(|cond| program_ops_are_materialized_env_local(cond, allow_root))
            }
            crate::vm::CompiledObjEntry::Dynamic { key, val } => {
                program_ops_are_materialized_env_local(key, allow_root)
                    && program_ops_are_materialized_env_local(val, allow_root)
            }
            crate::vm::CompiledObjEntry::Spread(prog)
            | crate::vm::CompiledObjEntry::SpreadDeep(prog) => {
                program_ops_are_materialized_env_local(prog, allow_root)
            }
        }),
        Opcode::MakeArr(items) => items
            .iter()
            .all(|(program, _)| program_ops_are_materialized_env_local(program, allow_root)),
        Opcode::FString(parts) => parts.iter().all(|part| match part {
            crate::vm::CompiledFSPart::Lit(_) => true,
            crate::vm::CompiledFSPart::Interp { prog, .. } => {
                program_ops_are_materialized_env_local(prog, allow_root)
            }
        }),
        Opcode::LetExpr { body, .. } => program_ops_are_materialized_env_local(body, allow_root),
        Opcode::IfElse { then_, else_ } => {
            program_ops_are_materialized_env_local(then_, allow_root)
                && program_ops_are_materialized_env_local(else_, allow_root)
        }
        Opcode::TryExpr { body, default } => {
            program_ops_are_materialized_env_local(body, allow_root)
                && program_ops_are_materialized_env_local(default, allow_root)
        }
        _ => true,
    }
}

fn program_ops_are_materialized_env_local(program: &crate::vm::Program, allow_root: bool) -> bool {
    program
        .ops
        .iter()
        .all(|opcode| opcode_is_materialized_env_local(opcode, allow_root))
}

fn classify_rpn_structural_kernel(ops: &[crate::vm::Opcode]) -> Option<BodyKernel> {
    use crate::vm::Opcode;

    let mut stack: Vec<BodyKernel> = Vec::new();
    for op in ops {
        match op {
            Opcode::PushCurrent => stack.push(BodyKernel::Current),
            Opcode::LoadIdent(key) => stack.push(BodyKernel::FieldRead(Arc::clone(key))),
            Opcode::GetField(key) => {
                let receiver = stack.pop().unwrap_or(BodyKernel::Current);
                stack.push(compose_field_kernel(receiver, Arc::clone(key)));
            }
            Opcode::FieldChain(chain) => {
                let receiver = stack.pop().unwrap_or(BodyKernel::Current);
                stack.push(compose_field_chain_kernel(
                    receiver,
                    Arc::clone(&chain.keys),
                ));
            }
            Opcode::CallMethod(call) if array_selector_call(call).is_some() => {
                let receiver = stack.pop().unwrap_or(BodyKernel::Current);
                if !receiver.is_view_native() {
                    return None;
                }
                stack.push(compose_array_select_kernel(
                    receiver,
                    array_selector_call(call)?,
                ));
            }
            Opcode::GetIndex(index) => {
                let receiver = stack.pop().unwrap_or(BodyKernel::Current);
                if !receiver.is_view_native() {
                    return None;
                }
                stack.push(compose_array_select_kernel(
                    receiver,
                    array_selector_from_index(*index)?,
                ));
            }
            op if trivial_lit(op).is_some() => stack.push(BodyKernel::Const(trivial_lit(op)?)),
            op if arithmetic_binop(op).is_some() => {
                let rhs = stack.pop()?;
                let lhs = stack.pop()?;
                stack.push(BodyKernel::Binary {
                    lhs: Box::new(lhs),
                    op: arithmetic_binop(op)?,
                    rhs: Box::new(rhs),
                });
            }
            _ => return None,
        }
    }
    match stack.as_slice() {
        [kernel] => Some(kernel.clone()),
        _ => None,
    }
}

fn compose_field_kernel(receiver: BodyKernel, key: Arc<str>) -> BodyKernel {
    match receiver {
        BodyKernel::Current => BodyKernel::FieldRead(key),
        BodyKernel::FieldRead(first) => BodyKernel::FieldChain(Arc::from([first, key])),
        BodyKernel::FieldChain(keys) => {
            let mut out = Vec::with_capacity(keys.len() + 1);
            out.extend(keys.iter().cloned());
            out.push(key);
            BodyKernel::FieldChain(out.into())
        }
        receiver => BodyKernel::Compose {
            first: Box::new(receiver),
            then: Box::new(BodyKernel::FieldRead(key)),
        },
    }
}

fn compose_field_chain_kernel(receiver: BodyKernel, keys: Arc<[Arc<str>]>) -> BodyKernel {
    match receiver {
        BodyKernel::Current => BodyKernel::FieldChain(keys),
        BodyKernel::FieldRead(first) => {
            let mut out = Vec::with_capacity(keys.len() + 1);
            out.push(first);
            out.extend(keys.iter().cloned());
            BodyKernel::FieldChain(out.into())
        }
        BodyKernel::FieldChain(prefix) => {
            let mut out = Vec::with_capacity(prefix.len() + keys.len());
            out.extend(prefix.iter().cloned());
            out.extend(keys.iter().cloned());
            BodyKernel::FieldChain(out.into())
        }
        receiver => BodyKernel::Compose {
            first: Box::new(receiver),
            then: Box::new(BodyKernel::FieldChain(keys)),
        },
    }
}

fn compose_array_select_kernel(receiver: BodyKernel, selector: ArraySelector) -> BodyKernel {
    BodyKernel::ArraySelect {
        array: Box::new(receiver),
        selector,
    }
}

fn array_selector_from_index(index: i64) -> Option<ArraySelector> {
    match index {
        -1 => Some(ArraySelector::Last),
        idx if idx >= 0 => Some(ArraySelector::Nth(idx as usize)),
        _ => None,
    }
}

fn classify_kv_path(steps: &[crate::vm::KvStep]) -> Option<BodyKernel> {
    if steps.is_empty() {
        return None;
    }
    let mut kernel = BodyKernel::Current;
    for step in steps {
        match step {
            crate::vm::KvStep::Field(key) => {
                kernel = compose_field_kernel(kernel, Arc::clone(key));
            }
            crate::vm::KvStep::Index(index) => {
                kernel = compose_array_select_kernel(kernel, array_selector_from_index(*index)?);
            }
        }
    }
    Some(kernel)
}

/// Describes how a Map stage's output elements should be collected by the sink.
pub(crate) enum CollectLayout<'a> {
    /// Output elements are heterogeneous `Val`s; collect into a plain array.
    Values,
    /// Every output element is a uniform object with the same key schema; collect into a columnar layout.
    UniformObject(&'a ObjectKernel),
}

#[inline]
fn trivial_lit(op: &crate::vm::Opcode) -> Option<Val> {
    use crate::vm::Opcode;
    match op {
        Opcode::PushInt(n) => Some(Val::Int(*n)),
        Opcode::PushFloat(f) => Some(Val::Float(*f)),
        Opcode::PushStr(s) => Some(Val::Str(s.clone())),
        Opcode::PushBool(b) => Some(Val::Bool(*b)),
        Opcode::PushNull => Some(Val::Null),
        _ => None,
    }
}

// called after the simple fused patterns are exhausted; handles structural view patterns
fn classify_structural_view_kernel(ops: &[crate::vm::Opcode]) -> Option<BodyKernel> {
    use crate::vm::Opcode;

    let ops = match ops {
        [Opcode::PushCurrent] => return Some(BodyKernel::Current),
        [Opcode::PushCurrent, rest @ ..] => rest,
        other => other,
    };

    if let [lhs @ .., lit_op, cmp_op] = ops {
        if let Some(lit) = trivial_lit(lit_op) {
            if let Some(op) = cmp_to_binop(cmp_op) {
                let lhs = classify_structural_view_kernel(lhs)?;
                return Some(BodyKernel::CmpLit {
                    lhs: Box::new(lhs),
                    op,
                    lit,
                });
            }
        }
    }

    match ops {
        [Opcode::LoadIdent(k) | Opcode::GetField(k)] => Some(BodyKernel::FieldRead(k.clone())),
        [Opcode::FieldChain(fc)] => Some(BodyKernel::FieldChain(fc.keys.clone())),
        [receiver @ .., Opcode::GetIndex(index)] => {
            let receiver = if receiver.is_empty() {
                BodyKernel::Current
            } else {
                classify_structural_view_kernel(receiver)?
            };
            if !receiver.is_view_native() {
                return None;
            }
            Some(compose_array_select_kernel(
                receiver,
                array_selector_from_index(*index)?,
            ))
        }
        [Opcode::LoadIdent(k1), rest @ ..]
            if rest.iter().all(|op| matches!(op, Opcode::GetField(_))) =>
        {
            let mut keys = Vec::with_capacity(rest.len() + 1);
            keys.push(k1.clone());
            for op in rest {
                if let Opcode::GetField(k) = op {
                    keys.push(k.clone());
                }
            }
            Some(BodyKernel::FieldChain(keys.into()))
        }
        [receiver @ .., Opcode::CallMethod(call)] if call.is_view_projection() => {
            let receiver = if receiver.is_empty() {
                BodyKernel::Current
            } else {
                classify_structural_view_kernel(receiver)?
            };
            let builtin_call = BuiltinCall::from_static_ast_args(
                call.method,
                call.name.as_ref(),
                &call.orig_args,
                |idx| {
                    Ok(call
                        .sub_progs
                        .get(idx)
                        .and_then(|prog| static_prog_val(prog)))
                },
            )
            .ok()
            .flatten()?;
            if !builtin_call.is_view_projection() {
                return None;
            }
            Some(BodyKernel::BuiltinCall {
                receiver: Box::new(receiver),
                call: builtin_call,
            })
        }
        _ => None,
    }
}

fn static_prog_val(prog: &crate::vm::Program) -> Option<Val> {
    match prog.ops.as_ref() {
        [op] => trivial_lit(op),
        _ => None,
    }
}

fn array_selector_call(call: &crate::vm::CompiledCall) -> Option<ArraySelector> {
    let index = match call.sub_progs.as_ref() {
        [prog] => match static_prog_val(prog)? {
            Val::Int(index) => Some(index),
            _ => None,
        },
        _ => None,
    };
    ArraySelector::from_builtin_selector(call.array_selector()?, index)
}

fn arithmetic_binop(op: &crate::vm::Opcode) -> Option<crate::parse::ast::BinOp> {
    use crate::parse::ast::BinOp as B;
    use crate::vm::Opcode as O;
    match op {
        O::Add => Some(B::Add),
        O::Sub => Some(B::Sub),
        O::Mul => Some(B::Mul),
        O::Div => Some(B::Div),
        O::Mod => Some(B::Mod),
        _ => None,
    }
}

#[inline]
fn cmp_to_binop(op: &crate::vm::Opcode) -> Option<crate::parse::ast::BinOp> {
    use crate::parse::ast::BinOp as B;
    use crate::vm::Opcode as O;
    match op {
        O::Eq => Some(B::Eq),
        O::Neq => Some(B::Neq),
        O::Lt => Some(B::Lt),
        O::Lte => Some(B::Lte),
        O::Gt => Some(B::Gt),
        O::Gte => Some(B::Gte),
        _ => None,
    }
}

/// Evaluates `kernel` against `item`, invoking `fallback` for VM re-entry only on `Generic`.
#[cfg(test)]
#[inline]
pub fn eval_kernel<F>(kernel: &BodyKernel, item: &Val, fallback: F) -> Result<Val, EvalError>
where
    F: FnOnce(&Val) -> Result<Val, EvalError>,
{
    let mut vm = crate::vm::VM::new();
    eval_kernel_with_vm(kernel, item, &mut vm, |item, _| fallback(item))
}

/// Evaluates `kernel` with caller-owned VM state for nested compiled match execution.
#[inline]
pub(crate) fn eval_kernel_with_vm<F>(
    kernel: &BodyKernel,
    item: &Val,
    vm: &mut crate::vm::VM,
    fallback: F,
) -> Result<Val, EvalError>
where
    F: FnOnce(&Val, &mut crate::vm::VM) -> Result<Val, EvalError>,
{
    if matches!(kernel, BodyKernel::Generic) {
        return fallback(item, vm);
    }
    eval_native_kernel_with_vm(kernel, item, vm)
}

pub(crate) fn eval_kernel_view_first_with_vm<F>(
    kernel: &BodyKernel,
    item: &Val,
    vm: &mut crate::vm::VM,
    fallback: F,
) -> Result<Val, EvalError>
where
    F: FnOnce(&Val, &mut crate::vm::VM) -> Result<Val, EvalError>,
{
    if let Some(value) = eval_view_kernel_with_vm(kernel, &ValView::new(item), vm) {
        return Ok(view_kernel_value_to_owned(value));
    }
    eval_kernel_with_vm(kernel, item, vm, fallback)
}

// panics on Generic — callers must route Generic through eval_kernel's fallback instead
#[cfg(test)]
fn eval_native_kernel(kernel: &BodyKernel, item: &Val) -> Result<Val, EvalError> {
    let mut vm = crate::vm::VM::new();
    eval_native_kernel_with_vm(kernel, item, &mut vm)
}

// panics on Generic — callers must route Generic through eval_kernel's fallback instead
fn eval_native_kernel_with_vm(
    kernel: &BodyKernel,
    item: &Val,
    vm: &mut crate::vm::VM,
) -> Result<Val, EvalError> {
    match kernel {
        BodyKernel::Current => Ok(item.clone()),
        BodyKernel::FieldRead(k) => Ok(item.get_field(k.as_ref())),
        BodyKernel::FieldChain(ks) => {
            let mut v = item.clone();
            for k in ks.iter() {
                v = v.get_field(k.as_ref());
                if matches!(v, Val::Null) {
                    break;
                }
            }
            Ok(v)
        }
        BodyKernel::ConstBool(b) => Ok(Val::Bool(*b)),
        BodyKernel::Const(v) => Ok(v.clone()),
        BodyKernel::FString(fstring) => eval_fstring_kernel(fstring, |kernel| {
            eval_native_kernel_with_vm(kernel, item, vm)
        }),
        BodyKernel::Object(object) => eval_object_kernel(object, |kernel| {
            eval_native_kernel_with_vm(kernel, item, vm)
        }),
        BodyKernel::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for item_kernel in items.iter() {
                out.push(eval_native_kernel_with_vm(item_kernel, item, vm)?);
            }
            Ok(Val::arr(out))
        }
        BodyKernel::ArraySpread(items) => {
            let mut out = Vec::with_capacity(items.len());
            for item_kernel in items.iter() {
                match item_kernel {
                    ArrayKernelElem::Value(kernel) => {
                        out.push(eval_native_kernel_with_vm(kernel, item, vm)?)
                    }
                    ArrayKernelElem::Spread(kernel) => {
                        append_array_spread_val_items(
                            &mut out,
                            eval_native_kernel_with_vm(kernel, item, vm)?,
                        );
                    }
                }
            }
            Ok(Val::arr(out))
        }
        BodyKernel::NestedArrayReducer {
            source,
            predicate,
            map,
            op,
        } => eval_nested_array_reducer_native(
            source,
            predicate.as_deref(),
            map.as_deref(),
            *op,
            item,
            vm,
        ),
        BodyKernel::NestedArrayCount { source, predicate } => {
            eval_nested_array_count_native(source, predicate.as_deref(), item, vm)
        }
        BodyKernel::NestedPlan(plan) => plan.run(item.clone()),
        BodyKernel::BuiltinCall { receiver, call } => {
            let recv = eval_native_kernel_with_vm(receiver, item, vm)?;
            call.try_apply(&recv)?
                .ok_or_else(|| EvalError(format!("{:?}: unsupported receiver", call.method)))
        }
        BodyKernel::Compose { first, then } => {
            let recv = eval_native_kernel_with_vm(first, item, vm)?;
            eval_native_kernel_with_vm(then, &recv, vm)
        }
        BodyKernel::CmpLit { lhs, op, lit } => {
            let lhs = eval_native_kernel_with_vm(lhs, item, vm)?;
            Ok(Val::Bool(eval_cmp_op(&lhs, *op, lit)))
        }
        BodyKernel::Binary { lhs, op, rhs } => {
            let lhs = eval_native_kernel_with_vm(lhs, item, vm)?;
            let rhs = eval_native_kernel_with_vm(rhs, item, vm)?;
            eval_binary_op(lhs, *op, rhs)
        }
        BodyKernel::ArraySelect { array, selector } => {
            let array = eval_native_kernel_with_vm(array, item, vm)?;
            Ok(eval_array_select_native(&array, *selector))
        }
        BodyKernel::Slice {
            array,
            from,
            to,
            step,
        } => {
            let array = eval_native_kernel_with_vm(array, item, vm)?;
            Ok(eval_slice_native(&array, *from, *to, *step))
        }
        BodyKernel::DynIndex { receiver, key } => {
            let receiver = eval_native_kernel_with_vm(receiver, item, vm)?;
            let key = eval_native_kernel_with_vm(key, item, vm)?;
            Ok(eval_dyn_index_native(&receiver, &key))
        }
        BodyKernel::Match {
            scrutinee,
            compiled,
            ..
        } => {
            let scrutinee = eval_native_kernel_with_vm(scrutinee, item, vm)?;
            let env = crate::data::context::Env::new(scrutinee.clone());
            vm.exec_match(compiled, &scrutinee, &env)
        }
        BodyKernel::And(predicates) => {
            for predicate in predicates.iter() {
                if !crate::util::is_truthy(&eval_native_kernel_with_vm(predicate, item, vm)?) {
                    return Ok(Val::Bool(false));
                }
            }
            Ok(Val::Bool(true))
        }
        BodyKernel::Or(predicates) => {
            for predicate in predicates.iter() {
                if crate::util::is_truthy(&eval_native_kernel_with_vm(predicate, item, vm)?) {
                    return Ok(Val::Bool(true));
                }
            }
            Ok(Val::Bool(false))
        }
        BodyKernel::Neg(kernel) => match eval_native_numeric_kernel(kernel, item) {
            Some(value) => Ok(numeric_kernel_value_to_val(neg_numeric(value))),
            None => Err(EvalError("unary minus requires a number".into())),
        },
        BodyKernel::Not(kernel) => Ok(Val::Bool(!crate::util::is_truthy(
            &eval_native_kernel_with_vm(kernel, item, vm)?,
        ))),
        BodyKernel::KindCheck { expr, ty, negate } => {
            let value = eval_native_kernel_with_vm(expr, item, vm)?;
            let matches = crate::util::kind_matches(&value, *ty);
            Ok(Val::Bool(if *negate { !matches } else { matches }))
        }
        BodyKernel::Cast { expr, ty } => {
            let value = eval_native_kernel_with_vm(expr, item, vm)?;
            crate::util::cast_val(&value, *ty)
        }
        BodyKernel::Coalesce { lhs, rhs } => {
            let value = eval_native_kernel_with_vm(lhs, item, vm)?;
            if value.is_null() {
                eval_native_kernel_with_vm(rhs, item, vm)
            } else {
                Ok(value)
            }
        }
        BodyKernel::IfElse { cond, then_, else_ } => {
            let cond = eval_native_kernel_with_vm(cond, item, vm)?;
            if crate::util::is_truthy(&cond) {
                eval_native_kernel_with_vm(then_, item, vm)
            } else {
                eval_native_kernel_with_vm(else_, item, vm)
            }
        }
        BodyKernel::FieldCmpLit(k, op, lit) => {
            let lhs = item.get_field(k.as_ref());
            Ok(Val::Bool(eval_cmp_op(&lhs, *op, lit)))
        }
        BodyKernel::FieldChainCmpLit(ks, op, lit) => {
            let mut v = item.clone();
            for k in ks.iter() {
                v = v.get_field(k.as_ref());
                if matches!(v, Val::Null) {
                    break;
                }
            }
            Ok(Val::Bool(eval_cmp_op(&v, *op, lit)))
        }
        BodyKernel::CurrentCmpLit(op, lit) => Ok(Val::Bool(eval_cmp_op(item, *op, lit))),
        BodyKernel::Generic => unreachable!("generic body kernels are handled by eval_kernel"),
    }
}

fn eval_nested_array_reducer_native(
    source: &BodyKernel,
    predicate: Option<&BodyKernel>,
    map: Option<&BodyKernel>,
    op: super::NumOp,
    item: &Val,
    vm: &mut crate::vm::VM,
) -> Result<Val, EvalError> {
    let source = eval_native_kernel_with_vm(source, item, vm)?;
    let Some(items) = source.as_vals() else {
        return Ok(op.empty());
    };
    let mut acc_i = 0i64;
    let mut acc_f = 0.0f64;
    let mut floated = false;
    let mut min_f = f64::INFINITY;
    let mut max_f = f64::NEG_INFINITY;
    let mut n_obs = 0usize;

    for child in items.iter() {
        if !native_predicate_matches(predicate, child, vm)? {
            continue;
        }
        let value;
        let observed = match map {
            Some(map) => {
                value = eval_native_kernel_with_vm(map, child, vm)?;
                &value
            }
            None => child,
        };
        super::num_fold(
            &mut acc_i,
            &mut acc_f,
            &mut floated,
            &mut min_f,
            &mut max_f,
            &mut n_obs,
            op,
            observed,
        );
    }

    Ok(super::num_finalise(
        op, acc_i, acc_f, floated, min_f, max_f, n_obs,
    ))
}

fn eval_nested_array_count_native(
    source: &BodyKernel,
    predicate: Option<&BodyKernel>,
    item: &Val,
    vm: &mut crate::vm::VM,
) -> Result<Val, EvalError> {
    let source = eval_native_kernel_with_vm(source, item, vm)?;
    let Some(items) = source.as_vals() else {
        return Ok(Val::Int(0));
    };
    let Some(predicate) = predicate else {
        return Ok(Val::Int(items.len() as i64));
    };
    let mut count = 0i64;
    for child in items.iter() {
        if native_predicate_matches(Some(predicate), child, vm)? {
            count += 1;
        }
    }
    Ok(Val::Int(count))
}

#[inline]
fn native_predicate_matches(
    predicate: Option<&BodyKernel>,
    item: &Val,
    vm: &mut crate::vm::VM,
) -> Result<bool, EvalError> {
    match predicate {
        Some(predicate) => Ok(crate::util::is_truthy(&eval_native_kernel_with_vm(
            predicate, item, vm,
        )?)),
        None => Ok(true),
    }
}

fn eval_object_kernel<F>(object: &ObjectKernel, mut eval: F) -> Result<Val, EvalError>
where
    F: FnMut(&BodyKernel) -> Result<Val, EvalError>,
{
    let mut pairs = Vec::with_capacity(object.entries.len());
    for entry in object.entries.iter() {
        if let ObjectKernelKey::Spread(mode) = entry.key {
            append_spread_val_pairs_for_mode(&mut pairs, eval(&entry.value)?, mode);
            continue;
        }
        if let Some(cond) = &entry.cond {
            let keep = crate::util::is_truthy(&eval(cond)?);
            if !keep {
                continue;
            }
        }
        let key = match &entry.key {
            ObjectKernelKey::Static(key) => Arc::clone(key),
            ObjectKernelKey::Dynamic(kernel) => {
                Arc::from(crate::util::val_to_key(&eval(kernel)?).as_str())
            }
            ObjectKernelKey::Spread(_) => unreachable!("spread entries are handled above"),
        };
        let value = eval(&entry.value)?;
        if (entry.optional || entry.omit_null) && value.is_null() {
            continue;
        }
        pairs.push((key, value));
    }
    Ok(Val::ObjSmall(pairs.into()))
}

pub(crate) fn append_spread_val_pairs_for_mode(
    pairs: &mut Vec<(Arc<str>, Val)>,
    value: Val,
    mode: ObjectSpreadMode,
) {
    match mode {
        ObjectSpreadMode::Shallow => append_spread_val_pairs(pairs, value),
        ObjectSpreadMode::Deep => {
            let base = Val::obj(pairs_to_index_map(std::mem::take(pairs)));
            let merged = crate::util::deep_merge_concat(base, normalize_spread_object(value));
            append_spread_val_pairs(pairs, merged);
        }
    }
}

pub(crate) fn append_spread_val_pairs(pairs: &mut Vec<(Arc<str>, Val)>, value: Val) {
    match value {
        Val::Obj(map) => {
            for (key, value) in map.iter() {
                pairs.push((Arc::clone(key), value.clone()));
            }
        }
        Val::ObjSmall(spread_pairs) => {
            for (key, value) in spread_pairs.iter() {
                pairs.push((Arc::clone(key), value.clone()));
            }
        }
        _ => {}
    }
}

fn pairs_to_index_map(pairs: Vec<(Arc<str>, Val)>) -> indexmap::IndexMap<Arc<str>, Val> {
    let mut map = indexmap::IndexMap::with_capacity(pairs.len());
    for (key, value) in pairs {
        map.insert(key, value);
    }
    map
}

fn normalize_spread_object(value: Val) -> Val {
    match value {
        Val::ObjSmall(pairs) => Val::obj(pairs_to_index_map(pairs.iter().cloned().collect())),
        other => other,
    }
}

pub(crate) fn append_array_spread_val_items(out: &mut Vec<Val>, value: Val) {
    match value {
        Val::Arr(items) => out.extend(items.iter().cloned()),
        Val::IntVec(items) => out.extend(items.iter().copied().map(Val::Int)),
        Val::FloatVec(items) => out.extend(items.iter().copied().map(Val::Float)),
        Val::StrVec(items) => out.extend(items.iter().cloned().map(Val::Str)),
        Val::StrSliceVec(items) => out.extend(items.iter().cloned().map(Val::StrSlice)),
        other => out.push(other),
    }
}

fn eval_fstring_kernel<F>(fstring: &FStringKernel, mut eval: F) -> Result<Val, EvalError>
where
    F: FnMut(&BodyKernel) -> Result<Val, EvalError>,
{
    let mut out = String::with_capacity(fstring.base_capacity);
    for part in fstring.parts.iter() {
        match part {
            FStringKernelPart::Lit(value) => out.push_str(value),
            FStringKernelPart::Interp(kernel) => append_val_to_string(&mut out, &eval(kernel)?)?,
        }
    }
    Ok(Val::Str(Arc::from(out)))
}

// uses itoa/ryu for numeric fast paths; val_to_string only for compound types
pub(crate) fn append_val_to_string(out: &mut String, value: &Val) -> Result<(), EvalError> {
    match value {
        Val::Str(value) => out.push_str(value),
        Val::StrSlice(value) => out.push_str(value.as_str()),
        Val::Int(value) => out.push_str(itoa::Buffer::new().format(*value)),
        Val::Float(value) => out.push_str(ryu::Buffer::new().format(*value)),
        Val::Bool(true) => out.push_str("true"),
        Val::Bool(false) => out.push_str("false"),
        Val::Null => out.push_str("null"),
        other => out.push_str(&crate::util::val_to_string(other)),
    }
    Ok(())
}

pub(crate) fn append_json_view_to_string<'a, V>(
    out: &mut String,
    view: &V,
    scalar: JsonView<'_>,
) -> Result<(), EvalError>
where
    V: ValueView<'a> + 'a,
{
    match scalar {
        JsonView::Null => out.push_str("null"),
        JsonView::Bool(true) => out.push_str("true"),
        JsonView::Bool(false) => out.push_str("false"),
        JsonView::Int(value) => out.push_str(itoa::Buffer::new().format(value)),
        JsonView::UInt(value) => out.push_str(itoa::Buffer::new().format(value)),
        JsonView::Float(value) => out.push_str(ryu::Buffer::new().format(value)),
        JsonView::Str(value) => out.push_str(value),
        JsonView::ArrayLen(_) | JsonView::ObjectLen(_) => {
            write_json_view(view, out).ok_or_else(|| {
                EvalError("view format error: missing child iterator".to_string())
            })?;
        }
    }
    Ok(())
}

pub(crate) fn view_kernel_value_to_owned<'a, V>(value: ViewKernelValue<V>) -> Val
where
    V: ValueView<'a> + 'a,
{
    match value {
        ViewKernelValue::View(view) => view_kernel_view_to_owned(view),
        ViewKernelValue::Owned(value) => value,
    }
}

pub(crate) fn view_kernel_view_to_owned<'a, V>(view: V) -> Val
where
    V: ValueView<'a> + 'a,
{
    scalar_view_to_owned_val(view.scalar()).unwrap_or_else(|| view.materialize())
}

#[derive(Debug, Clone, Copy)]
enum NumericKernelValue {
    Int(i64),
    Float(f64),
}

#[inline]
fn numeric_kernel_value_to_val(value: NumericKernelValue) -> Val {
    match value {
        NumericKernelValue::Int(value) => Val::Int(value),
        NumericKernelValue::Float(value) => Val::Float(value),
    }
}

pub(crate) fn eval_binary_op(
    lhs: Val,
    op: crate::parse::ast::BinOp,
    rhs: Val,
) -> Result<Val, EvalError> {
    use crate::parse::ast::BinOp;
    match op {
        BinOp::Add => crate::util::add_vals(lhs, rhs),
        BinOp::Sub => crate::util::num_op(lhs, rhs, |a, b| a - b, |a, b| a - b),
        BinOp::Mul => crate::util::num_op(lhs, rhs, |a, b| a * b, |a, b| a * b),
        BinOp::Div => {
            let b = rhs.as_f64().unwrap_or(0.0);
            if b == 0.0 {
                return Err(EvalError("division by zero".into()));
            }
            Ok(Val::Float(lhs.as_f64().unwrap_or(0.0) / b))
        }
        BinOp::Mod => crate::util::num_op(lhs, rhs, |a, b| a % b, |a, b| a % b),
        BinOp::Eq | BinOp::Neq | BinOp::Lt | BinOp::Lte | BinOp::Gt | BinOp::Gte | BinOp::Fuzzy => {
            Ok(Val::Bool(eval_cmp_op(&lhs, op, &rhs)))
        }
        BinOp::And => Ok(Val::Bool(
            crate::util::is_truthy(&lhs) && crate::util::is_truthy(&rhs),
        )),
        BinOp::Or => Ok(Val::Bool(
            crate::util::is_truthy(&lhs) || crate::util::is_truthy(&rhs),
        )),
    }
}

fn eval_array_select_native(array: &Val, selector: ArraySelector) -> Val {
    let Some(items) = array.as_vals() else {
        return Val::Null;
    };
    let Some(idx) = selector.index_for_len(items.len()) else {
        return Val::Null;
    };
    items.get(idx).cloned().unwrap_or(Val::Null)
}

fn eval_array_select_view<'a, V>(array: V, selector: ArraySelector) -> V
where
    V: ValueView<'a> + 'a,
{
    let idx = match selector {
        ArraySelector::First => Some(0),
        ArraySelector::Last => array.array_len().and_then(|len| len.checked_sub(1)),
        ArraySelector::Nth(idx) => Some(idx),
    };
    idx.map(|idx| array.array_child(idx))
        .unwrap_or_else(|| array.index(-1))
}

fn eval_dyn_index_native(receiver: &Val, key: &Val) -> Val {
    match key {
        Val::Int(index) => receiver.get_index(*index),
        Val::Str(key) => receiver.get_field(key.as_ref()),
        Val::StrSlice(key) => receiver.get_field(key.as_str()),
        _ => Val::Null,
    }
}

fn eval_dyn_index_view<'a, V>(receiver: V, key: &Val) -> Option<ViewKernelValue<V>>
where
    V: ValueView<'a> + 'a,
{
    match key {
        Val::Int(index) => Some(ViewKernelValue::View(receiver.index(*index))),
        Val::Str(key) => Some(ViewKernelValue::View(receiver.field(key.as_ref()))),
        Val::StrSlice(key) => Some(ViewKernelValue::View(receiver.field(key.as_str()))),
        _ => Some(ViewKernelValue::Owned(Val::Null)),
    }
}

fn eval_slice_native(array: &Val, from: Option<i64>, to: Option<i64>, step: Option<i64>) -> Val {
    let Some(items) = array.as_vals() else {
        return Val::Null;
    };
    Val::arr(slice_indices(items.len(), from, to, step).map(|idx| items[idx].clone()).collect())
}

fn eval_slice_view<'a, V>(
    array: V,
    from: Option<i64>,
    to: Option<i64>,
    step: Option<i64>,
) -> Option<Val>
where
    V: ValueView<'a> + 'a,
{
    let len = array.array_len()?;
    let mut out = Vec::new();
    for idx in slice_indices(len, from, to, step) {
        out.push(view_kernel_view_to_owned(array.array_child(idx)));
    }
    Some(Val::arr(out))
}

fn slice_indices(
    len: usize,
    from: Option<i64>,
    to: Option<i64>,
    step: Option<i64>,
) -> impl Iterator<Item = usize> {
    let len_i = len as i64;
    let step = step.unwrap_or(1);
    let mut out = Vec::new();
    if step == 0 || len == 0 {
        return out.into_iter();
    }
    if step > 0 {
        let end_default = len_i;
        let mut i = resolve_slice_idx(from.unwrap_or(0), len_i).min(len_i);
        let end = resolve_slice_idx(to.unwrap_or(end_default), len_i).min(len_i);
        while i < end {
            out.push(i as usize);
            i += step;
        }
    } else {
        let s_raw = from.unwrap_or(len_i - 1);
        let mut i = if s_raw < 0 {
            (len_i + s_raw).max(-1)
        } else {
            s_raw.min(len_i - 1)
        };
        let e_raw = to.unwrap_or(-1);
        let end = if e_raw < 0 && e_raw != -1 {
            (len_i + e_raw).max(-1)
        } else if e_raw == -1 && to.is_none() {
            -1
        } else {
            e_raw
        };
        while i > end && i >= 0 {
            out.push(i as usize);
            i += step;
        }
    }
    out.into_iter()
}

#[inline]
fn resolve_slice_idx(idx: i64, len: i64) -> i64 {
    if idx < 0 {
        (len + idx).max(0)
    } else {
        idx
    }
}

fn eval_nested_array_reducer_view<'a, V>(
    source: &BodyKernel,
    predicate: Option<&BodyKernel>,
    map: Option<&BodyKernel>,
    op: super::NumOp,
    item: &V,
    vm: &mut crate::vm::VM,
) -> Option<Val>
where
    V: ValueView<'a> + 'a,
{
    let source = match eval_view_kernel_inner(source, item, vm)? {
        ViewKernelValue::View(view) => view,
        ViewKernelValue::Owned(value) => {
            return eval_nested_array_reducer_native_owned_with_vm(value, predicate, map, op, vm)
        }
    };
    let mut iter = source.array_iter()?;
    let mut acc_i = 0i64;
    let mut acc_f = 0.0f64;
    let mut floated = false;
    let mut min_f = f64::INFINITY;
    let mut max_f = f64::NEG_INFINITY;
    let mut n_obs = 0usize;

    iter.try_for_each(|child| {
        if !view_predicate_matches(predicate, &child, vm)? {
            return Some(());
        }
        match map {
            Some(map) => {
                if let Some(value) = eval_view_numeric_kernel(map, &child, vm) {
                    fold_numeric_kernel_value(
                        value,
                        &mut acc_i,
                        &mut acc_f,
                        &mut floated,
                        &mut min_f,
                        &mut max_f,
                        &mut n_obs,
                        op,
                    );
                    return Some(());
                }

                match eval_view_kernel_inner(map, &child, vm)? {
                    ViewKernelValue::View(view) => {
                        fold_json_view_scalar(
                            view.scalar(),
                            &mut acc_i,
                            &mut acc_f,
                            &mut floated,
                            &mut min_f,
                            &mut max_f,
                            &mut n_obs,
                            op,
                        );
                    }
                    ViewKernelValue::Owned(value) => {
                        super::num_fold(
                            &mut acc_i,
                            &mut acc_f,
                            &mut floated,
                            &mut min_f,
                            &mut max_f,
                            &mut n_obs,
                            op,
                            &value,
                        );
                    }
                }
            }
            None => {
                fold_json_view_scalar(
                    child.scalar(),
                    &mut acc_i,
                    &mut acc_f,
                    &mut floated,
                    &mut min_f,
                    &mut max_f,
                    &mut n_obs,
                    op,
                );
            }
        }
        Some(())
    })?;

    Some(super::num_finalise(
        op, acc_i, acc_f, floated, min_f, max_f, n_obs,
    ))
}

fn eval_nested_array_count_view<'a, V>(
    source: &BodyKernel,
    predicate: Option<&BodyKernel>,
    item: &V,
    vm: &mut crate::vm::VM,
) -> Option<Val>
where
    V: ValueView<'a> + 'a,
{
    let source = match eval_view_kernel_inner(source, item, vm)? {
        ViewKernelValue::View(view) => view,
        ViewKernelValue::Owned(value) => {
            return eval_nested_array_count_native_owned_with_vm(value, predicate, vm)
        }
    };
    let Some(predicate) = predicate else {
        return Some(Val::Int(source.array_len().unwrap_or(0) as i64));
    };
    let mut count = 0i64;
    let mut iter = source.array_iter()?;
    iter.try_for_each(|child| {
        if view_predicate_matches(Some(predicate), &child, vm)? {
            count += 1;
        }
        Some(())
    })?;
    Some(Val::Int(count))
}

#[inline]
fn view_predicate_matches<'a, V>(
    predicate: Option<&BodyKernel>,
    item: &V,
    vm: &mut crate::vm::VM,
) -> Option<bool>
where
    V: ValueView<'a> + 'a,
{
    match predicate {
        Some(predicate) => match eval_view_kernel_inner(predicate, item, vm)? {
            ViewKernelValue::View(view) => Some(view.scalar().truthy()),
            ViewKernelValue::Owned(value) => Some(crate::util::is_truthy(&value)),
        },
        None => Some(true),
    }
}

fn eval_view_numeric_kernel<'a, V>(
    kernel: &BodyKernel,
    item: &V,
    vm: &mut crate::vm::VM,
) -> Option<NumericKernelValue>
where
    V: ValueView<'a> + 'a,
{
    match kernel {
        BodyKernel::Current => numeric_from_json_view(item.scalar()),
        BodyKernel::FieldRead(key) => numeric_from_json_view(item.field(key).scalar()),
        BodyKernel::FieldChain(keys) => numeric_from_json_view(item.field_chain(keys).scalar()),
        BodyKernel::Const(Val::Int(value)) => Some(NumericKernelValue::Int(*value)),
        BodyKernel::Const(Val::Float(value)) => Some(NumericKernelValue::Float(*value)),
        BodyKernel::Binary { lhs, op, rhs } => {
            let lhs = eval_view_numeric_kernel(lhs, item, vm)?;
            let rhs = eval_view_numeric_kernel(rhs, item, vm)?;
            eval_numeric_binary(lhs, *op, rhs)
        }
        BodyKernel::Neg(kernel) => eval_view_numeric_kernel(kernel, item, vm).map(neg_numeric),
        BodyKernel::Compose { first, then } => match eval_view_kernel_inner(first, item, vm)? {
            ViewKernelValue::View(view) => eval_view_numeric_kernel(then, &view, vm),
            ViewKernelValue::Owned(value) => eval_native_numeric_kernel(then, &value),
        },
        BodyKernel::ArraySelect { array, selector } => {
            match eval_view_kernel_inner(array, item, vm)? {
                ViewKernelValue::View(view) => {
                    numeric_from_json_view(eval_array_select_view(view, *selector).scalar())
                }
                ViewKernelValue::Owned(value) => {
                    numeric_from_val(&eval_array_select_native(&value, *selector))
                }
            }
        }
        BodyKernel::Slice { .. } => None,
        BodyKernel::DynIndex { receiver, key } => {
            let key = view_kernel_value_to_owned(eval_view_kernel_inner(key, item, vm)?);
            match eval_view_kernel_inner(receiver, item, vm)? {
                ViewKernelValue::View(view) => match eval_dyn_index_view(view, &key)? {
                    ViewKernelValue::View(view) => numeric_from_json_view(view.scalar()),
                    ViewKernelValue::Owned(value) => numeric_from_val(&value),
                },
                ViewKernelValue::Owned(value) => numeric_from_val(&eval_dyn_index_native(&value, &key)),
            }
        }
        _ => None,
    }
}

pub(crate) fn eval_view_numeric_kernel_value<'a, V>(
    kernel: &BodyKernel,
    item: &V,
    vm: &mut crate::vm::VM,
) -> Option<Val>
where
    V: ValueView<'a> + 'a,
{
    eval_view_numeric_kernel(kernel, item, vm).map(numeric_kernel_value_to_val)
}

fn eval_native_numeric_kernel(kernel: &BodyKernel, item: &Val) -> Option<NumericKernelValue> {
    match kernel {
        BodyKernel::Current => numeric_from_val(item),
        BodyKernel::FieldRead(key) => numeric_from_val(&item.get_field(key)),
        BodyKernel::FieldChain(keys) => {
            let mut cur = item.clone();
            for key in keys.iter() {
                cur = cur.get_field(key);
            }
            numeric_from_val(&cur)
        }
        BodyKernel::Const(Val::Int(value)) => Some(NumericKernelValue::Int(*value)),
        BodyKernel::Const(Val::Float(value)) => Some(NumericKernelValue::Float(*value)),
        BodyKernel::Binary { lhs, op, rhs } => {
            let lhs = eval_native_numeric_kernel(lhs, item)?;
            let rhs = eval_native_numeric_kernel(rhs, item)?;
            eval_numeric_binary(lhs, *op, rhs)
        }
        BodyKernel::Neg(kernel) => eval_native_numeric_kernel(kernel, item).map(neg_numeric),
        _ => None,
    }
}

#[inline]
fn numeric_from_json_view(scalar: JsonView<'_>) -> Option<NumericKernelValue> {
    match scalar {
        JsonView::Int(value) => Some(NumericKernelValue::Int(value)),
        JsonView::UInt(value) if value <= i64::MAX as u64 => {
            Some(NumericKernelValue::Int(value as i64))
        }
        JsonView::UInt(value) => Some(NumericKernelValue::Float(value as f64)),
        JsonView::Float(value) => Some(NumericKernelValue::Float(value)),
        _ => None,
    }
}

#[inline]
fn numeric_from_val(value: &Val) -> Option<NumericKernelValue> {
    match value {
        Val::Int(value) => Some(NumericKernelValue::Int(*value)),
        Val::Float(value) => Some(NumericKernelValue::Float(*value)),
        _ => None,
    }
}

#[inline]
fn neg_numeric(value: NumericKernelValue) -> NumericKernelValue {
    match value {
        NumericKernelValue::Int(value) => NumericKernelValue::Int(-value),
        NumericKernelValue::Float(value) => NumericKernelValue::Float(-value),
    }
}

#[inline]
fn json_view_matches_kind(view: JsonView<'_>, ty: crate::parse::ast::KindType) -> bool {
    use crate::parse::ast::KindType;
    matches!(
        (view, ty),
        (JsonView::Null, KindType::Null)
            | (JsonView::Bool(_), KindType::Bool)
            | (JsonView::Int(_) | JsonView::UInt(_) | JsonView::Float(_), KindType::Number)
            | (JsonView::Str(_), KindType::Str)
            | (JsonView::ArrayLen(_), KindType::Array)
            | (JsonView::ObjectLen(_), KindType::Object)
    )
}

#[inline]
fn safe_view_cast_type(ty: crate::parse::ast::CastType) -> bool {
    use crate::parse::ast::CastType;
    matches!(
        ty,
        CastType::Str | CastType::Bool | CastType::Array | CastType::Null
    )
}

fn eval_numeric_binary(
    lhs: NumericKernelValue,
    op: crate::parse::ast::BinOp,
    rhs: NumericKernelValue,
) -> Option<NumericKernelValue> {
    use crate::parse::ast::BinOp;
    match (lhs, rhs, op) {
        (NumericKernelValue::Int(a), NumericKernelValue::Int(b), BinOp::Add) => {
            Some(NumericKernelValue::Int(a + b))
        }
        (NumericKernelValue::Int(a), NumericKernelValue::Int(b), BinOp::Sub) => {
            Some(NumericKernelValue::Int(a - b))
        }
        (NumericKernelValue::Int(a), NumericKernelValue::Int(b), BinOp::Mul) => {
            Some(NumericKernelValue::Int(a * b))
        }
        (NumericKernelValue::Int(a), NumericKernelValue::Int(b), BinOp::Mod) if b != 0 => {
            Some(NumericKernelValue::Int(a % b))
        }
        (a, b, BinOp::Add) => Some(NumericKernelValue::Float(
            numeric_to_f64(a) + numeric_to_f64(b),
        )),
        (a, b, BinOp::Sub) => Some(NumericKernelValue::Float(
            numeric_to_f64(a) - numeric_to_f64(b),
        )),
        (a, b, BinOp::Mul) => Some(NumericKernelValue::Float(
            numeric_to_f64(a) * numeric_to_f64(b),
        )),
        (a, b, BinOp::Div) => {
            let denom = numeric_to_f64(b);
            (denom != 0.0).then(|| NumericKernelValue::Float(numeric_to_f64(a) / denom))
        }
        (a, b, BinOp::Mod) => {
            let denom = numeric_to_f64(b);
            (denom != 0.0).then(|| NumericKernelValue::Float(numeric_to_f64(a) % denom))
        }
        _ => None,
    }
}

#[inline]
fn numeric_to_f64(value: NumericKernelValue) -> f64 {
    match value {
        NumericKernelValue::Int(value) => value as f64,
        NumericKernelValue::Float(value) => value,
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn fold_numeric_kernel_value(
    value: NumericKernelValue,
    acc_i: &mut i64,
    acc_f: &mut f64,
    floated: &mut bool,
    min_f: &mut f64,
    max_f: &mut f64,
    n_obs: &mut usize,
    op: super::NumOp,
) {
    match value {
        NumericKernelValue::Int(value) => {
            super::num_fold_i64(acc_i, acc_f, floated, min_f, max_f, n_obs, op, value)
        }
        NumericKernelValue::Float(value) => {
            super::num_fold_f64(acc_i, acc_f, floated, min_f, max_f, n_obs, op, value)
        }
    }
}

fn eval_nested_array_reducer_native_owned_with_vm(
    source: Val,
    predicate: Option<&BodyKernel>,
    map: Option<&BodyKernel>,
    op: super::NumOp,
    vm: &mut crate::vm::VM,
) -> Option<Val> {
    let Some(items) = source.as_vals() else {
        return Some(op.empty());
    };
    let mut acc_i = 0i64;
    let mut acc_f = 0.0f64;
    let mut floated = false;
    let mut min_f = f64::INFINITY;
    let mut max_f = f64::NEG_INFINITY;
    let mut n_obs = 0usize;
    for child in items.iter() {
        if !native_predicate_matches_opt_with_vm(predicate, child, vm)? {
            continue;
        }
        let value;
        let observed = match map {
            Some(map) => {
                value = eval_native_kernel_with_vm(map, child, vm).ok()?;
                &value
            }
            None => child,
        };
        super::num_fold(
            &mut acc_i,
            &mut acc_f,
            &mut floated,
            &mut min_f,
            &mut max_f,
            &mut n_obs,
            op,
            observed,
        );
    }
    Some(super::num_finalise(
        op, acc_i, acc_f, floated, min_f, max_f, n_obs,
    ))
}

fn eval_nested_array_count_native_owned_with_vm(
    source: Val,
    predicate: Option<&BodyKernel>,
    vm: &mut crate::vm::VM,
) -> Option<Val> {
    let Some(items) = source.as_vals() else {
        return Some(Val::Int(0));
    };
    let Some(predicate) = predicate else {
        return Some(Val::Int(items.len() as i64));
    };
    let mut count = 0i64;
    for child in items.iter() {
        if native_predicate_matches_opt_with_vm(Some(predicate), child, vm)? {
            count += 1;
        }
    }
    Some(Val::Int(count))
}

#[inline]
fn native_predicate_matches_opt_with_vm(
    predicate: Option<&BodyKernel>,
    item: &Val,
    vm: &mut crate::vm::VM,
) -> Option<bool> {
    match predicate {
        Some(predicate) => eval_native_kernel_with_vm(predicate, item, vm)
            .ok()
            .map(|value| crate::util::is_truthy(&value)),
        None => Some(true),
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn fold_json_view_scalar(
    scalar: JsonView<'_>,
    acc_i: &mut i64,
    acc_f: &mut f64,
    floated: &mut bool,
    min_f: &mut f64,
    max_f: &mut f64,
    n_obs: &mut usize,
    op: super::NumOp,
) {
    match scalar {
        JsonView::Int(value) => super::num_fold(
            acc_i,
            acc_f,
            floated,
            min_f,
            max_f,
            n_obs,
            op,
            &Val::Int(value),
        ),
        JsonView::UInt(value) if value <= i64::MAX as u64 => super::num_fold(
            acc_i,
            acc_f,
            floated,
            min_f,
            max_f,
            n_obs,
            op,
            &Val::Int(value as i64),
        ),
        JsonView::UInt(value) => super::num_fold(
            acc_i,
            acc_f,
            floated,
            min_f,
            max_f,
            n_obs,
            op,
            &Val::Float(value as f64),
        ),
        JsonView::Float(value) => super::num_fold(
            acc_i,
            acc_f,
            floated,
            min_f,
            max_f,
            n_obs,
            op,
            &Val::Float(value),
        ),
        _ => {}
    }
}

/// Result of a view-native kernel evaluation: a borrowed sub-view or a newly-owned `Val`.
pub(crate) enum ViewKernelValue<V> {
    /// The kernel produced a borrowed sub-view of the input without materialising.
    View(V),
    /// The kernel produced an owned `Val` (e.g. a literal, comparison result, or builtin output).
    Owned(Val),
}

/// Evaluates `kernel` on the borrowed `item` view, returning a sub-view or owned `Val`; `None` for `Generic`.
#[cfg(test)]
#[inline]
pub(crate) fn eval_view_kernel<'a, V>(kernel: &BodyKernel, item: &V) -> Option<ViewKernelValue<V>>
where
    V: ValueView<'a> + 'a,
{
    let mut vm = crate::vm::VM::new();
    eval_view_kernel_inner(kernel, item, &mut vm)
}

/// Evaluates `kernel` on a borrowed view using caller-owned VM state for nested
/// fallback paths.
#[inline]
pub(crate) fn eval_view_kernel_with_vm<'a, V>(
    kernel: &BodyKernel,
    item: &V,
    vm: &mut crate::vm::VM,
) -> Option<ViewKernelValue<V>>
where
    V: ValueView<'a> + 'a,
{
    eval_view_kernel_inner(kernel, item, vm)
}

fn eval_view_kernel_inner<'a, V>(
    kernel: &BodyKernel,
    item: &V,
    vm: &mut crate::vm::VM,
) -> Option<ViewKernelValue<V>>
where
    V: ValueView<'a> + 'a,
{
    match kernel {
        BodyKernel::Current => Some(ViewKernelValue::View(item.clone())),
        BodyKernel::FieldRead(key) => Some(ViewKernelValue::View(item.field(key))),
        BodyKernel::FieldChain(keys) => Some(ViewKernelValue::View(item.field_chain(keys))),
        BodyKernel::ConstBool(value) => Some(ViewKernelValue::Owned(Val::Bool(*value))),
        BodyKernel::Const(value) => Some(ViewKernelValue::Owned(value.clone())),
        BodyKernel::FString(fstring) => {
            let mut out = String::with_capacity(fstring.base_capacity);
            for part in fstring.parts.iter() {
                match part {
                    FStringKernelPart::Lit(value) => out.push_str(value),
                    FStringKernelPart::Interp(kernel) => {
                        match eval_view_kernel_inner(kernel, item, vm)? {
                            ViewKernelValue::View(view) => {
                                append_json_view_to_string(&mut out, &view, view.scalar()).ok()?;
                            }
                            ViewKernelValue::Owned(value) => {
                                append_val_to_string(&mut out, &value).ok()?;
                            }
                        }
                    }
                }
            }
            Some(ViewKernelValue::Owned(Val::Str(Arc::from(out))))
        }
        BodyKernel::Object(object) => {
            let mut pairs = Vec::with_capacity(object.entries.len());
            for entry in object.entries.iter() {
                if let ObjectKernelKey::Spread(mode) = entry.key {
                    match eval_view_kernel_inner(&entry.value, item, vm)? {
                        ViewKernelValue::View(view) => {
                            append_spread_view_pairs_for_mode(&mut pairs, view, mode)?
                        }
                        ViewKernelValue::Owned(value) => {
                            append_spread_val_pairs_for_mode(&mut pairs, value, mode)
                        }
                    }
                    continue;
                }
                if let Some(cond) = &entry.cond {
                    let keep = match eval_view_kernel_inner(cond, item, vm)? {
                        ViewKernelValue::View(view) => view.scalar().truthy(),
                        ViewKernelValue::Owned(value) => crate::util::is_truthy(&value),
                    };
                    if !keep {
                        continue;
                    }
                }
                let key = match &entry.key {
                    ObjectKernelKey::Static(key) => Arc::clone(key),
                    ObjectKernelKey::Dynamic(kernel) => {
                        let value =
                            view_kernel_value_to_owned(eval_view_kernel_inner(kernel, item, vm)?);
                        Arc::from(crate::util::val_to_key(&value).as_str())
                    }
                    ObjectKernelKey::Spread(_) => unreachable!("spread entries are handled above"),
                };
                let value = match eval_view_kernel_inner(&entry.value, item, vm)? {
                    ViewKernelValue::View(view) => view_kernel_view_to_owned(view),
                    ViewKernelValue::Owned(value) => value,
                };
                if (entry.optional || entry.omit_null) && value.is_null() {
                    continue;
                }
                pairs.push((key, value));
            }
            Some(ViewKernelValue::Owned(Val::ObjSmall(pairs.into())))
        }
        BodyKernel::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for item_kernel in items.iter() {
                out.push(view_kernel_value_to_owned(eval_view_kernel_inner(
                    item_kernel,
                    item,
                    vm,
                )?));
            }
            Some(ViewKernelValue::Owned(Val::arr(out)))
        }
        BodyKernel::ArraySpread(items) => {
            let mut out = Vec::with_capacity(items.len());
            for item_kernel in items.iter() {
                match item_kernel {
                    ArrayKernelElem::Value(kernel) => {
                        out.push(view_kernel_value_to_owned(eval_view_kernel_inner(
                            kernel, item, vm,
                        )?));
                    }
                    ArrayKernelElem::Spread(kernel) => {
                        match eval_view_kernel_inner(kernel, item, vm)? {
                            ViewKernelValue::View(view) => {
                                append_array_spread_view_items(&mut out, view)?
                            }
                            ViewKernelValue::Owned(value) => {
                                append_array_spread_val_items(&mut out, value)
                            }
                        }
                    }
                }
            }
            Some(ViewKernelValue::Owned(Val::arr(out)))
        }
        BodyKernel::NestedArrayReducer {
            source,
            predicate,
            map,
            op,
        } => eval_nested_array_reducer_view(
            source,
            predicate.as_deref(),
            map.as_deref(),
            *op,
            item,
            vm,
        )
        .map(ViewKernelValue::Owned),
        BodyKernel::NestedArrayCount { source, predicate } => {
            eval_nested_array_count_view(source, predicate.as_deref(), item, vm)
                .map(ViewKernelValue::Owned)
        }
        BodyKernel::NestedPlan(plan) => {
            let result = plan.run(item.materialize());
            result.ok().map(ViewKernelValue::Owned)
        }
        BodyKernel::BuiltinCall { receiver, call } => {
            match eval_view_kernel_inner(receiver, item, vm)? {
                ViewKernelValue::View(view) => {
                    match apply_view_projection(call.id(), &call.args, view)? {
                        ViewProjectionResult::View(view) => Some(ViewKernelValue::View(view)),
                        ViewProjectionResult::Owned(value) => Some(ViewKernelValue::Owned(value)),
                    }
                }
                ViewKernelValue::Owned(value) => call
                    .try_apply(&value)
                    .ok()
                    .flatten()
                    .map(ViewKernelValue::Owned),
            }
        }
        BodyKernel::Compose { first, then } => match eval_view_kernel_inner(first, item, vm)? {
            ViewKernelValue::View(view) => eval_view_kernel_inner(then, &view, vm),
            ViewKernelValue::Owned(value) => eval_native_kernel_with_vm(then, &value, vm)
                .ok()
                .map(ViewKernelValue::Owned),
        },
        BodyKernel::CmpLit { lhs, op, lit } => {
            let passes = match eval_view_kernel_inner(lhs, item, vm)? {
                ViewKernelValue::View(view) => crate::util::json_cmp_binop(
                    view.scalar(),
                    *op,
                    crate::util::JsonView::from_val(lit),
                ),
                ViewKernelValue::Owned(value) => crate::util::json_cmp_binop(
                    JsonView::from_val(&value),
                    *op,
                    crate::util::JsonView::from_val(lit),
                ),
            };
            Some(ViewKernelValue::Owned(Val::Bool(passes)))
        }
        BodyKernel::Binary { lhs, op, rhs } => {
            if let Some(value) = eval_view_numeric_kernel(kernel, item, vm) {
                return Some(ViewKernelValue::Owned(numeric_kernel_value_to_val(value)));
            }
            let lhs = view_kernel_value_to_owned(eval_view_kernel_inner(lhs, item, vm)?);
            let rhs = view_kernel_value_to_owned(eval_view_kernel_inner(rhs, item, vm)?);
            eval_binary_op(lhs, *op, rhs)
                .ok()
                .map(ViewKernelValue::Owned)
        }
        BodyKernel::ArraySelect { array, selector } => {
            match eval_view_kernel_inner(array, item, vm)? {
                ViewKernelValue::View(view) => Some(ViewKernelValue::View(eval_array_select_view(
                    view, *selector,
                ))),
                ViewKernelValue::Owned(value) => Some(ViewKernelValue::Owned(
                    eval_array_select_native(&value, *selector),
                )),
            }
        }
        BodyKernel::Slice {
            array,
            from,
            to,
            step,
        } => match eval_view_kernel_inner(array, item, vm)? {
            ViewKernelValue::View(view) => eval_slice_view(view, *from, *to, *step)
                .map(ViewKernelValue::Owned),
            ViewKernelValue::Owned(value) => Some(ViewKernelValue::Owned(eval_slice_native(
                &value, *from, *to, *step,
            ))),
        },
        BodyKernel::DynIndex { receiver, key } => {
            let key = view_kernel_value_to_owned(eval_view_kernel_inner(key, item, vm)?);
            match eval_view_kernel_inner(receiver, item, vm)? {
                ViewKernelValue::View(view) => eval_dyn_index_view(view, &key),
                ViewKernelValue::Owned(value) => {
                    Some(ViewKernelValue::Owned(eval_dyn_index_native(&value, &key)))
                }
            }
        }
        BodyKernel::Match {
            scrutinee,
            compiled,
            body_needs_current,
        } => match eval_view_kernel_inner(scrutinee, item, vm)? {
            ViewKernelValue::View(view) => {
                let current = if *body_needs_current {
                    view_kernel_view_to_owned(view.clone())
                } else {
                    Val::Null
                };
                let env = crate::data::context::Env::new(current);
                crate::vm::exec_match_view(vm, compiled, view, &env)
                    .ok()
                    .map(ViewKernelValue::Owned)
            }
            ViewKernelValue::Owned(value) => {
                let env = crate::data::context::Env::new(value.clone());
                vm.exec_match(compiled, &value, &env)
                    .ok()
                    .map(ViewKernelValue::Owned)
            }
        },
        BodyKernel::And(predicates) => {
            for predicate in predicates.iter() {
                let passes = match eval_view_kernel_inner(predicate, item, vm)? {
                    ViewKernelValue::View(view) => view.scalar().truthy(),
                    ViewKernelValue::Owned(value) => crate::util::is_truthy(&value),
                };
                if !passes {
                    return Some(ViewKernelValue::Owned(Val::Bool(false)));
                }
            }
            Some(ViewKernelValue::Owned(Val::Bool(true)))
        }
        BodyKernel::Or(predicates) => {
            for predicate in predicates.iter() {
                let passes = match eval_view_kernel_inner(predicate, item, vm)? {
                    ViewKernelValue::View(view) => view.scalar().truthy(),
                    ViewKernelValue::Owned(value) => crate::util::is_truthy(&value),
                };
                if passes {
                    return Some(ViewKernelValue::Owned(Val::Bool(true)));
                }
            }
            Some(ViewKernelValue::Owned(Val::Bool(false)))
        }
        BodyKernel::Neg(kernel) => eval_view_numeric_kernel(kernel, item, vm)
            .map(neg_numeric)
            .map(numeric_kernel_value_to_val)
            .map(ViewKernelValue::Owned),
        BodyKernel::Not(kernel) => {
            let passes = match eval_view_kernel_inner(kernel, item, vm)? {
                ViewKernelValue::View(view) => view.scalar().truthy(),
                ViewKernelValue::Owned(value) => crate::util::is_truthy(&value),
            };
            Some(ViewKernelValue::Owned(Val::Bool(!passes)))
        }
        BodyKernel::KindCheck { expr, ty, negate } => {
            let matches = match eval_view_kernel_inner(expr, item, vm)? {
                ViewKernelValue::View(view) => json_view_matches_kind(view.scalar(), *ty),
                ViewKernelValue::Owned(value) => crate::util::kind_matches(&value, *ty),
            };
            Some(ViewKernelValue::Owned(Val::Bool(if *negate {
                !matches
            } else {
                matches
            })))
        }
        BodyKernel::Cast { expr, ty } => {
            let value = match eval_view_kernel_inner(expr, item, vm)? {
                ViewKernelValue::View(view) => match crate::util::cast_json_view(view.scalar(), *ty)
                {
                    Some(result) => result.ok()?,
                    None => crate::util::cast_val(&view_kernel_view_to_owned(view), *ty).ok()?,
                },
                ViewKernelValue::Owned(value) => crate::util::cast_val(&value, *ty).ok()?,
            };
            Some(ViewKernelValue::Owned(value))
        }
        BodyKernel::Coalesce { lhs, rhs } => match eval_view_kernel_inner(lhs, item, vm)? {
            ViewKernelValue::View(view) if !matches!(view.scalar(), JsonView::Null) => {
                Some(ViewKernelValue::View(view))
            }
            ViewKernelValue::Owned(value) if !value.is_null() => {
                Some(ViewKernelValue::Owned(value))
            }
            _ => eval_view_kernel_inner(rhs, item, vm),
        },
        BodyKernel::IfElse { cond, then_, else_ } => {
            let passes = match eval_view_kernel_inner(cond, item, vm)? {
                ViewKernelValue::View(view) => view.scalar().truthy(),
                ViewKernelValue::Owned(value) => crate::util::is_truthy(&value),
            };
            if passes {
                eval_view_kernel_inner(then_, item, vm)
            } else {
                eval_view_kernel_inner(else_, item, vm)
            }
        }
        BodyKernel::FieldCmpLit(key, op, lit) => {
            let lhs = item.field(key);
            Some(ViewKernelValue::Owned(Val::Bool(
                crate::util::json_cmp_binop(
                    lhs.scalar(),
                    *op,
                    crate::util::JsonView::from_val(lit),
                ),
            )))
        }
        BodyKernel::FieldChainCmpLit(keys, op, lit) => {
            let lhs = item.field_chain(keys);
            Some(ViewKernelValue::Owned(Val::Bool(
                crate::util::json_cmp_binop(
                    lhs.scalar(),
                    *op,
                    crate::util::JsonView::from_val(lit),
                ),
            )))
        }
        BodyKernel::CurrentCmpLit(op, lit) => Some(ViewKernelValue::Owned(Val::Bool(
            crate::util::json_cmp_binop(item.scalar(), *op, crate::util::JsonView::from_val(lit)),
        ))),
        BodyKernel::Generic => None,
    }
}

fn append_spread_view_pairs<'a, V>(pairs: &mut Vec<(Arc<str>, Val)>, view: V) -> Option<()>
where
    V: ValueView<'a> + 'a,
{
    for (key, child) in view.object_iter()? {
        pairs.push((key, view_kernel_view_to_owned(child)));
    }
    Some(())
}

fn append_spread_view_pairs_for_mode<'a, V>(
    pairs: &mut Vec<(Arc<str>, Val)>,
    view: V,
    mode: ObjectSpreadMode,
) -> Option<()>
where
    V: ValueView<'a> + 'a,
{
    match mode {
        ObjectSpreadMode::Shallow => append_spread_view_pairs(pairs, view),
        ObjectSpreadMode::Deep => {
            append_spread_val_pairs_for_mode(pairs, view_kernel_view_to_owned(view), mode);
            Some(())
        }
    }
}

fn append_array_spread_view_items<'a, V>(out: &mut Vec<Val>, view: V) -> Option<()>
where
    V: ValueView<'a> + 'a,
{
    let Some(children) = view.array_iter() else {
        out.push(view_kernel_view_to_owned(view));
        return Some(());
    };
    for child in children {
        out.push(view_kernel_view_to_owned(child));
    }
    Some(())
}

/// Evaluates `lhs op rhs` using JSON-view comparison semantics, returning the boolean result.
#[inline]
pub fn eval_cmp_op(lhs: &Val, op: crate::parse::ast::BinOp, rhs: &Val) -> bool {
    crate::util::json_cmp_binop(
        crate::util::JsonView::from_val(lhs),
        op,
        crate::util::JsonView::from_val(rhs),
    )
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::builtins::{BuiltinArgs, BuiltinCall, BuiltinMethod};
    use crate::compile::compiler::Compiler;
    use crate::data::value::Val;
    use crate::data::view::{TapeView, ValView, ValueView};
    use crate::parse::ast::BinOp;
    use crate::parse::parser::parse;

    use super::{
        eval_view_kernel, ArrayKernelElem, BodyKernel, CollectLayout, FStringKernel,
        FStringKernelPart, ObjectKernelKey, ObjectSpreadMode, ViewKernelValue,
    };

    fn key_call(method: BuiltinMethod, key: &str) -> BodyKernel {
        BodyKernel::BuiltinCall {
            receiver: Box::new(BodyKernel::Current),
            call: BuiltinCall::new(method, BuiltinArgs::Str(Arc::from(key))),
        }
    }

    fn key_vec_call(method: BuiltinMethod, keys: &[&str]) -> BodyKernel {
        BodyKernel::BuiltinCall {
            receiver: Box::new(BodyKernel::Current),
            call: BuiltinCall::new(
                method,
                BuiltinArgs::StrVec(keys.iter().map(|key| Arc::from(*key)).collect()),
            ),
        }
    }

    fn path_call(method: BuiltinMethod, path: &str) -> BodyKernel {
        BodyKernel::BuiltinCall {
            receiver: Box::new(BodyKernel::Current),
            call: BuiltinCall::new(method, BuiltinArgs::Str(Arc::from(path))),
        }
    }

    fn owned_bool(value: Option<ViewKernelValue<ValView<'_>>>) -> Option<bool> {
        match value? {
            ViewKernelValue::Owned(Val::Bool(value)) => Some(value),
            _ => None,
        }
    }

    fn owned_tape_value(value: Option<ViewKernelValue<TapeView<'_>>>) -> Option<Val> {
        match value? {
            ViewKernelValue::Owned(value) => Some(value),
            ViewKernelValue::View(view) => Some(view.materialize()),
        }
    }

    #[test]
    fn fstring_formats_compound_tape_views_without_materializing() {
        let tape = crate::data::tape::TapeData::parse(
            br#"{"tags":["sf","hugo"],"meta":{"ok":true}}"#.to_vec(),
        )
        .unwrap();
        let view = TapeView::root(&tape);
        tape.reset_materialized_subtrees();
        let kernel = BodyKernel::FString(FStringKernel {
            parts: vec![
                FStringKernelPart::Lit(Arc::from("tags=")),
                FStringKernelPart::Interp(BodyKernel::FieldRead(Arc::from("tags"))),
                FStringKernelPart::Lit(Arc::from(" meta=")),
                FStringKernelPart::Interp(BodyKernel::FieldRead(Arc::from("meta"))),
            ]
            .into(),
            base_capacity: 11,
        });

        let out = owned_tape_value(eval_view_kernel(&kernel, &view)).expect("fstring output");

        assert_eq!(
            out,
            Val::Str(Arc::from(r#"tags=["sf","hugo"] meta={"ok":true}"#))
        );
        assert_eq!(tape.materialized_subtrees(), 0);
    }

    fn owned_value(value: Option<ViewKernelValue<ValView<'_>>>) -> Option<Val> {
        match value? {
            ViewKernelValue::Owned(value) => Some(value),
            ViewKernelValue::View(view) => Some(view.materialize()),
        }
    }

    #[test]
    fn binop_reports_shared_comparison_metadata() {
        assert!(BinOp::Eq.is_scalar_comparison());
        assert!(BinOp::Gte.is_scalar_comparison());
        assert!(!BinOp::Fuzzy.is_scalar_comparison());
        assert!(BinOp::Fuzzy.is_predicate_comparison());
        assert_eq!(BinOp::Lt.flipped_comparison(), Some(BinOp::Gt));
        assert_eq!(BinOp::Lte.flipped_comparison(), Some(BinOp::Gte));
        assert_eq!(BinOp::Add.flipped_comparison(), None);
    }

    fn field_paths(kernel: &BodyKernel) -> Vec<String> {
        match kernel.field_demand() {
            crate::plan::demand::FieldDemand::None => Vec::new(),
            crate::plan::demand::FieldDemand::Whole => vec!["*".to_string()],
            crate::plan::demand::FieldDemand::Fields(fields) => fields
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
        }
    }

    fn path_keys(kernel: &BodyKernel) -> Option<Vec<String>> {
        kernel.field_path_keys().map(|keys| {
            keys.iter()
                .map(|key| key.as_ref().to_string())
                .collect::<Vec<_>>()
        })
    }

    #[test]
    fn field_path_keys_reports_only_direct_row_paths() {
        assert_eq!(path_keys(&BodyKernel::Current), Some(Vec::new()));
        assert_eq!(
            path_keys(&BodyKernel::FieldRead(Arc::from("isbn"))),
            Some(vec!["isbn".to_string()])
        );
        assert_eq!(
            path_keys(&BodyKernel::FieldChain(
                vec![Arc::from("user"), Arc::from("name")].into()
            )),
            Some(vec!["user".to_string(), "name".to_string()])
        );
        assert_eq!(
            path_keys(&BodyKernel::classify_expr(
                &parse(r#"profile.get_path("author.name")"#).expect("parse get_path")
            )),
            Some(vec![
                "profile".to_string(),
                "author".to_string(),
                "name".to_string()
            ])
        );
        assert_eq!(
            path_keys(&BodyKernel::classify_expr(
                &parse(r#"profile.get_path("items[0].sku")"#).expect("parse indexed get_path")
            )),
            None
        );
        assert_eq!(
            path_keys(&BodyKernel::FieldCmpLit(
                Arc::from("score"),
                crate::parse::ast::BinOp::Gt,
                Val::Int(10)
            )),
            None
        );
    }

    #[test]
    fn literal_value_reports_only_constant_kernels() {
        assert_eq!(
            BodyKernel::Const(Val::Int(7)).literal_value(),
            Some(Val::Int(7))
        );
        assert_eq!(
            BodyKernel::ConstBool(true).literal_value(),
            Some(Val::Bool(true))
        );
        assert_eq!(BodyKernel::Current.literal_value(), None);
        assert_eq!(
            BodyKernel::FieldRead(Arc::from("isbn")).literal_value(),
            None
        );
    }

    #[test]
    fn constant_truthy_reports_literal_kernel_truthiness() {
        assert_eq!(BodyKernel::ConstBool(true).constant_truthy(), Some(true));
        assert_eq!(BodyKernel::ConstBool(false).constant_truthy(), Some(false));
        assert_eq!(BodyKernel::Const(Val::Int(1)).constant_truthy(), Some(true));
        assert_eq!(BodyKernel::Const(Val::Null).constant_truthy(), Some(false));
        assert_eq!(BodyKernel::Current.constant_truthy(), None);
    }

    #[test]
    fn field_path_literal_cmp_reports_direct_comparisons() {
        let cmp = BodyKernel::CmpLit {
            lhs: Box::new(BodyKernel::FieldChain(
                vec![Arc::from("user"), Arc::from("score")].into(),
            )),
            op: crate::parse::ast::BinOp::Gte,
            lit: Val::Int(90),
        };
        let (keys, op, lit) = cmp.field_path_literal_cmp().unwrap();
        assert_eq!(
            keys.iter().map(|key| key.as_ref()).collect::<Vec<_>>(),
            vec!["user", "score"]
        );
        assert_eq!(op, crate::parse::ast::BinOp::Gte);
        assert_eq!(lit, Val::Int(90));

        let get_path_cmp = BodyKernel::classify_expr(
            &parse(r#"profile.get_path("author.name") == "ada""#)
                .expect("parse get_path comparison"),
        );
        let (keys, op, lit) = get_path_cmp.field_path_literal_cmp().unwrap();
        assert_eq!(
            keys.iter().map(|key| key.as_ref()).collect::<Vec<_>>(),
            vec!["profile", "author", "name"]
        );
        assert_eq!(op, crate::parse::ast::BinOp::Eq);
        assert_eq!(lit, Val::Str(Arc::from("ada")));

        let computed = BodyKernel::CmpLit {
            lhs: Box::new(BodyKernel::Binary {
                lhs: Box::new(BodyKernel::FieldRead(Arc::from("a"))),
                op: crate::parse::ast::BinOp::Add,
                rhs: Box::new(BodyKernel::FieldRead(Arc::from("b"))),
            }),
            op: crate::parse::ast::BinOp::Eq,
            lit: Val::Int(10),
        };
        assert!(computed.field_path_literal_cmp().is_none());
    }

    #[test]
    fn object_key_calls_on_current_have_field_demand() {
        assert_eq!(
            field_paths(&key_call(BuiltinMethod::HasKey, "isbn")),
            vec!["isbn"]
        );
        assert_eq!(
            field_paths(&key_call(BuiltinMethod::Missing, "title")),
            vec!["title"]
        );
        assert_eq!(
            field_paths(&key_vec_call(BuiltinMethod::Missing, &["title", "isbn"])),
            vec!["title", "isbn"]
        );
        assert_eq!(
            field_paths(&key_call(BuiltinMethod::Missing, "meta.author.name")),
            vec!["meta.author.name"]
        );
        assert_eq!(
            field_paths(&key_vec_call(
                BuiltinMethod::Missing,
                &["meta.author.name", "items[0].sku"]
            )),
            vec!["meta.author.name", "items"]
        );
        assert_eq!(
            field_paths(&key_vec_call(BuiltinMethod::Pick, &["title", "isbn"])),
            vec!["title", "isbn"]
        );
        assert_eq!(
            field_paths(&path_call(BuiltinMethod::HasPath, "user.name")),
            vec!["user.name"]
        );
        assert_eq!(
            field_paths(&path_call(BuiltinMethod::GetPath, "items[0].price")),
            vec!["items"]
        );
        let chained = BodyKernel::classify_expr(
            &parse(r#"profile.get_path("author").has_key("name")"#)
                .expect("parse chained get_path key check"),
        );
        assert_eq!(field_paths(&chained), vec!["profile.author.name"]);

        let indexed = BodyKernel::classify_expr(
            &parse(r#"profile.get_path("items[0]").has_key("sku")"#)
                .expect("parse indexed get_path key check"),
        );
        assert_eq!(field_paths(&indexed), vec!["profile.items"]);
        assert_eq!(
            field_paths(&key_call(BuiltinMethod::Has, "isbn")),
            vec!["*"]
        );
    }

    #[test]
    fn arithmetic_kernels_run_on_value_views() {
        let expr = parse("qty * price").expect("parse arithmetic");
        let program = Compiler::compile(&expr, "qty * price");
        let kernel = BodyKernel::classify(&program);
        assert!(matches!(kernel, BodyKernel::Binary { .. }));
        assert!(kernel.is_view_native());

        let value = Val::obj(
            [
                (Arc::from("qty"), Val::Int(3)),
                (Arc::from("price"), Val::Float(12.5)),
            ]
            .into(),
        );
        let view = ValView::new(&value);

        assert_eq!(
            owned_value(eval_view_kernel(&kernel, &view)),
            Some(Val::Float(37.5))
        );
    }

    #[test]
    fn string_len_method_chain_stays_scalar_builtin_kernel() {
        let expr = parse("name.len() == 3").expect("parse string len predicate");
        let program = Compiler::compile(&expr, "name.len() == 3");
        let kernel = BodyKernel::classify(&program);

        assert!(
            matches!(
                &kernel,
                BodyKernel::CmpLit { lhs, .. }
                    if matches!(
                        lhs.as_ref(),
                        BodyKernel::BuiltinCall { receiver, call }
                            if call.method == BuiltinMethod::Len
                                && matches!(
                                    receiver.as_ref(),
                                    BodyKernel::FieldRead(field) if field.as_ref() == "name"
                                )
                    )
            ),
            "{kernel:#?}"
        );

        let value = Val::obj([(Arc::from("name"), Val::Str(Arc::from("ada")))].into());
        assert_eq!(
            owned_bool(eval_view_kernel(&kernel, &ValView::new(&value))),
            Some(true)
        );
    }

    #[test]
    fn rpn_arithmetic_kernels_compose_nested_expressions() {
        let expr = parse("qty * price + fee").expect("parse nested arithmetic");
        let program = Compiler::compile(&expr, "qty * price + fee");
        let kernel = BodyKernel::classify(&program);
        assert!(matches!(kernel, BodyKernel::Binary { .. }));
        assert!(kernel.is_view_native());

        let value = Val::obj(
            [
                (Arc::from("qty"), Val::Int(3)),
                (Arc::from("price"), Val::Float(12.5)),
                (Arc::from("fee"), Val::Float(2.25)),
            ]
            .into(),
        );
        let view = ValView::new(&value);

        assert_eq!(
            owned_value(eval_view_kernel(&kernel, &view)),
            Some(Val::Float(39.75))
        );
    }

    #[test]
    fn expression_combinator_kernels_run_on_value_views() {
        let expr = parse(
            r#"{ok: not archived, label: nickname ?? name, tier: "high" if score > 90 else "normal"}"#,
        )
        .expect("parse expression combinator projection");
        let program = Compiler::compile(&expr, "expression-combinators");
        let kernel = BodyKernel::classify(&program);

        assert!(matches!(kernel, BodyKernel::Object(_)), "{kernel:#?}");
        assert!(kernel.is_view_native(), "{kernel:#?}");

        let value = Val::from(&serde_json::json!({
            "archived": false,
            "name": "Ada",
            "nickname": null,
            "score": 95
        }));
        let out = owned_value(eval_view_kernel(&kernel, &ValView::new(&value)))
            .expect("expression combinator output");

        assert_eq!(
            serde_json::Value::from(out),
            serde_json::json!({"ok": true, "label": "Ada", "tier": "high"})
        );
    }

    #[test]
    fn neg_and_kind_kernels_run_on_value_views() {
        let expr =
            parse(r#"{sort_key: -score, numeric: score kind number, named: name kind string}"#)
                .expect("parse neg and kind projection");
        let program = Compiler::compile(&expr, "neg-kind");
        let kernel = BodyKernel::classify(&program);

        assert!(matches!(kernel, BodyKernel::Object(_)), "{kernel:#?}");
        assert!(kernel.is_view_native(), "{kernel:#?}");

        let value = Val::from(&serde_json::json!({"score": 42, "name": "Ada"}));
        let out = owned_value(eval_view_kernel(&kernel, &ValView::new(&value)))
            .expect("neg and kind projection output");

        assert_eq!(
            serde_json::Value::from(out),
            serde_json::json!({"sort_key": -42, "numeric": true, "named": true})
        );
    }

    #[test]
    fn safe_cast_kernels_run_on_value_views() {
        let expr =
            parse(r#"{id: id as string, ok: score as bool, tags: tag as array, gone: tag as null}"#)
            .expect("parse safe cast projection");
        let program = Compiler::compile(&expr, "safe-cast");
        let kernel = BodyKernel::classify(&program);

        assert!(matches!(kernel, BodyKernel::Object(_)), "{kernel:#?}");
        assert!(kernel.is_view_native(), "{kernel:#?}");

        let value = Val::from(&serde_json::json!({"id": 42, "score": 1, "tag": "sf"}));
        let out = owned_value(eval_view_kernel(&kernel, &ValView::new(&value)))
            .expect("safe cast projection output");

        assert_eq!(
            serde_json::Value::from(out),
            serde_json::json!({"id": "42", "ok": true, "tags": ["sf"], "gone": null})
        );
    }

    #[test]
    fn ast_expression_combinators_stay_view_native() {
        let expr = parse(r#"nickname ?? name"#).expect("parse coalesce projection");
        let kernel = BodyKernel::classify_expr(&expr);

        assert!(matches!(kernel, BodyKernel::Coalesce { .. }), "{kernel:#?}");
        assert!(kernel.is_view_native(), "{kernel:#?}");

        let value = Val::from(&serde_json::json!({"name": "Ada", "nickname": null}));
        assert_eq!(
            owned_value(eval_view_kernel(&kernel, &ValView::new(&value))),
            Some(Val::Str(Arc::from("Ada")))
        );
    }

    #[test]
    fn array_select_kernels_run_on_value_views() {
        let expr = parse("events.last().kind").expect("parse array select");
        let program = Compiler::compile(&expr, "events.last().kind");
        let kernel = BodyKernel::classify(&program);
        assert!(matches!(kernel, BodyKernel::Compose { .. }));
        assert!(kernel.is_view_native());

        let value = Val::obj(
            [(
                Arc::from("events"),
                Val::arr(vec![
                    Val::obj([(Arc::from("kind"), Val::Str(Arc::from("placed")))].into()),
                    Val::obj([(Arc::from("kind"), Val::Str(Arc::from("delivered")))].into()),
                ]),
            )]
            .into(),
        );
        let view = ValView::new(&value);

        assert_eq!(
            owned_value(eval_view_kernel(&kernel, &view)),
            Some(Val::Str(Arc::from("delivered")))
        );
    }

    #[test]
    fn dynamic_index_kernels_run_on_value_views() {
        let expr = parse(r#"{picked: obj[key], numbered: items[i]}"#)
            .expect("parse dynamic index projection");
        let program = Compiler::compile(&expr, "dynamic-index");
        let kernel = BodyKernel::classify(&program);

        assert!(matches!(kernel, BodyKernel::Object(_)), "{kernel:#?}");
        assert!(kernel.is_view_native(), "{kernel:#?}");

        let value = Val::from(&serde_json::json!({
            "obj": {"a": "alpha", "b": "beta"},
            "key": "b",
            "items": [10, 20, 30],
            "i": 1
        }));
        let out = owned_value(eval_view_kernel(&kernel, &ValView::new(&value)))
            .expect("dynamic index output");

        assert_eq!(
            serde_json::Value::from(out),
            serde_json::json!({"picked": "beta", "numbered": 20})
        );
    }

    #[test]
    fn slice_kernels_run_on_value_views() {
        let expr = parse(r#"{prefix: items[:2], middle: items[1:4:2], reverse: items[::-1]}"#)
            .expect("parse slice projection");
        let program = Compiler::compile(&expr, "slice");
        let kernel = BodyKernel::classify(&program);

        assert!(matches!(kernel, BodyKernel::Object(_)), "{kernel:#?}");
        assert!(kernel.is_view_native(), "{kernel:#?}");

        let value = Val::from(&serde_json::json!({"items": [10, 20, 30, 40]}));
        let out =
            owned_value(eval_view_kernel(&kernel, &ValView::new(&value))).expect("slice output");

        assert_eq!(
            serde_json::Value::from(out),
            serde_json::json!({
                "prefix": [10, 20],
                "middle": [20, 40],
                "reverse": [40, 30, 20, 10]
            })
        );
    }

    #[test]
    fn ast_array_selector_methods_use_shared_kernel_metadata() {
        let expr = parse("events.last().kind").expect("parse array selector method");
        let kernel = BodyKernel::classify_expr(&expr);
        assert!(matches!(kernel, BodyKernel::Compose { .. }), "{kernel:#?}");
        assert!(kernel.is_view_native());
        assert_eq!(field_paths(&kernel), vec!["events"]);

        let value = Val::obj(
            [(
                Arc::from("events"),
                Val::arr(vec![
                    Val::obj([(Arc::from("kind"), Val::Str(Arc::from("placed")))].into()),
                    Val::obj([(Arc::from("kind"), Val::Str(Arc::from("delivered")))].into()),
                ]),
            )]
            .into(),
        );

        assert_eq!(
            owned_value(eval_view_kernel(&kernel, &ValView::new(&value))),
            Some(Val::Str(Arc::from("delivered")))
        );
    }

    #[test]
    fn array_selector_conversion_is_shared_and_bounds_safe() {
        assert_eq!(
            super::ArraySelector::from_builtin_selector(
                crate::builtins::BuiltinArraySelector::First,
                None
            ),
            Some(super::ArraySelector::First)
        );
        assert_eq!(
            super::ArraySelector::from_builtin_selector(
                crate::builtins::BuiltinArraySelector::Last,
                None
            ),
            Some(super::ArraySelector::Last)
        );
        assert_eq!(
            super::ArraySelector::from_builtin_selector(
                crate::builtins::BuiltinArraySelector::Nth,
                Some(2)
            ),
            Some(super::ArraySelector::Nth(2))
        );
        assert_eq!(
            super::ArraySelector::from_builtin_selector(
                crate::builtins::BuiltinArraySelector::Nth,
                Some(-1)
            ),
            None
        );
        assert_eq!(super::ArraySelector::First.index_for_len(0), Some(0));
        assert_eq!(super::ArraySelector::Last.index_for_len(0), None);
        assert_eq!(super::ArraySelector::Last.index_for_len(3), Some(2));
        assert_eq!(super::ArraySelector::Nth(5).index_for_len(3), Some(5));
    }

    #[test]
    fn compiled_index_path_projection_stays_view_native() {
        let expr = parse("tags[0].name").expect("parse indexed path projection");
        let program = Compiler::compile(&expr, "indexed-path");
        let kernel = BodyKernel::classify(&program);

        assert!(kernel.is_view_native(), "{kernel:#?}");
        let Some((source_keys, selector, suffix_keys)) = kernel.array_element_path() else {
            panic!("expected array element path kernel, got {kernel:#?}");
        };
        assert_eq!(
            source_keys
                .iter()
                .map(|key| key.as_ref())
                .collect::<Vec<_>>(),
            ["tags"]
        );
        assert_eq!(selector, super::ArraySelector::Nth(0));
        assert_eq!(
            suffix_keys
                .iter()
                .map(|key| key.as_ref())
                .collect::<Vec<_>>(),
            ["name"]
        );

        let json = serde_json::json!({"tags":[{"name":"sf"},{"name":"classic"}]});
        let val = Val::from(&json);
        assert_eq!(
            owned_value(eval_view_kernel(&kernel, &ValView::new(&val))),
            Some(Val::Str(Arc::from("sf")))
        );
    }

    #[test]
    fn compiled_object_index_projection_stays_view_native() {
        let expr = parse(r#"{tag: tags[0].name}"#).expect("parse indexed object projection");
        let program = Compiler::compile(&expr, "indexed-object");
        let kernel = BodyKernel::classify(&program);

        assert!(matches!(kernel, BodyKernel::Object(_)), "{kernel:#?}");
        assert!(kernel.is_view_native(), "{kernel:#?}");

        let json = serde_json::json!({"tags":[{"name":"sf"},{"name":"classic"}]});
        let val = Val::from(&json);
        let out = owned_value(eval_view_kernel(&kernel, &ValView::new(&val)))
            .expect("object projection output");
        assert_eq!(
            serde_json::Value::from(out),
            serde_json::json!({"tag":"sf"})
        );
    }

    #[test]
    fn dynamic_object_key_projection_stays_view_native() {
        let expr = parse(r#"{[@.kind]: value}"#).expect("parse dynamic object key projection");
        let program = Compiler::compile(&expr, "dynamic-object-key");
        let kernel = BodyKernel::classify(&program);

        let BodyKernel::Object(object) = &kernel else {
            panic!("expected object kernel, got {kernel:#?}");
        };
        assert!(!object.has_static_layout());
        assert!(matches!(
            object.entries()[0].key_kernel(),
            ObjectKernelKey::Dynamic(_)
        ));
        assert!(kernel.is_view_native(), "{kernel:#?}");
        assert!(matches!(kernel.collect_layout(), CollectLayout::Values));

        let json = serde_json::json!({"kind":"isbn","value":"978"});
        let val = Val::from(&json);
        let out = owned_value(eval_view_kernel(&kernel, &ValView::new(&val)))
            .expect("dynamic object projection output");
        assert_eq!(
            serde_json::Value::from(out),
            serde_json::json!({"isbn":"978"})
        );
    }

    #[test]
    fn shallow_object_spread_projection_stays_view_native() {
        let expr = parse(r#"{...base, extra: value}"#).expect("parse object spread projection");
        let program = Compiler::compile(&expr, "object-spread");
        let kernel = BodyKernel::classify(&program);

        let BodyKernel::Object(object) = &kernel else {
            panic!("expected object kernel, got {kernel:#?}");
        };
        assert!(!object.has_static_layout());
        assert!(matches!(
            object.entries()[0].key_kernel(),
            ObjectKernelKey::Spread(ObjectSpreadMode::Shallow)
        ));
        assert!(kernel.is_view_native(), "{kernel:#?}");
        assert!(matches!(kernel.collect_layout(), CollectLayout::Values));

        let json = serde_json::json!({"base":{"isbn":"978","price":20},"value":"ok"});
        let val = Val::from(&json);
        let out = owned_value(eval_view_kernel(&kernel, &ValView::new(&val)))
            .expect("object spread projection output");
        assert_eq!(
            serde_json::Value::from(out),
            serde_json::json!({"isbn":"978","price":20,"extra":"ok"})
        );
    }

    #[test]
    fn deep_object_spread_projection_stays_view_native() {
        let expr = parse(r#"{...base, ...**delta}"#).expect("parse deep object spread projection");
        let program = Compiler::compile(&expr, "deep-object-spread");
        let kernel = BodyKernel::classify(&program);

        let BodyKernel::Object(object) = &kernel else {
            panic!("expected object kernel, got {kernel:#?}");
        };
        assert!(!object.has_static_layout());
        assert!(matches!(
            object.entries()[1].key_kernel(),
            ObjectKernelKey::Spread(ObjectSpreadMode::Deep)
        ));
        assert!(kernel.is_view_native(), "{kernel:#?}");

        let json = serde_json::json!({
            "base": {"meta": {"tags": ["sf"], "score": 1}, "name": "Dune"},
            "delta": {"meta": {"tags": ["classic"], "year": 1965}}
        });
        let val = Val::from(&json);
        let out = owned_value(eval_view_kernel(&kernel, &ValView::new(&val)))
            .expect("deep object spread projection output");
        assert_eq!(
            serde_json::Value::from(out),
            serde_json::json!({
                "meta": {"tags": ["sf", "classic"], "score": 1, "year": 1965},
                "name": "Dune"
            })
        );
    }

    #[test]
    fn array_spread_projection_stays_view_native() {
        let expr = parse(r#"[head, ...tags, tail]"#).expect("parse array spread projection");
        let program = Compiler::compile(&expr, "array-spread");
        let kernel = BodyKernel::classify(&program);

        let BodyKernel::ArraySpread(items) = &kernel else {
            panic!("expected spread array kernel, got {kernel:#?}");
        };
        assert!(matches!(items[0], ArrayKernelElem::Value(_)));
        assert!(matches!(items[1], ArrayKernelElem::Spread(_)));
        assert!(matches!(items[2], ArrayKernelElem::Value(_)));
        assert!(kernel.is_view_native(), "{kernel:#?}");

        let json = serde_json::json!({"head":"a","tags":["sf","hugo"],"tail":"z"});
        let val = Val::from(&json);
        let out = owned_value(eval_view_kernel(&kernel, &ValView::new(&val)))
            .expect("array spread projection output");
        assert_eq!(
            serde_json::Value::from(out),
            serde_json::json!(["a", "sf", "hugo", "z"])
        );
    }

    #[test]
    fn compiled_scalar_call_comparison_uses_kernel_metadata() {
        let expr = parse(r#"name.upper() == "ADA""#).expect("parse scalar comparison");
        let program = Compiler::compile(&expr, "scalar-call-cmp");
        let kernel = BodyKernel::classify(&program);
        assert!(matches!(kernel, BodyKernel::CmpLit { .. }), "{kernel:#?}");
        assert!(kernel.is_view_native());
    }

    #[test]
    fn scalar_call_path_metadata_is_shared() {
        let expr = parse("user.name.len()").expect("parse scalar call");
        let kernel = BodyKernel::classify_expr(&expr);
        let call = kernel
            .path_scalar_call()
            .expect("path scalar call metadata");

        assert_eq!(
            call.receiver_keys
                .iter()
                .map(|key| key.as_ref())
                .collect::<Vec<_>>(),
            vec!["user", "name"]
        );
        assert_eq!(call.call.method, crate::builtins::BuiltinMethod::Len);

        let expr =
            parse(r#"profile.get_path("author.name").len()"#).expect("parse get_path scalar call");
        let kernel = BodyKernel::classify_expr(&expr);
        let call = kernel
            .path_scalar_call()
            .expect("get_path scalar call metadata");
        assert_eq!(
            call.receiver_keys
                .iter()
                .map(|key| key.as_ref())
                .collect::<Vec<_>>(),
            vec!["profile", "author", "name"]
        );
        assert_eq!(call.call.method, crate::builtins::BuiltinMethod::Len);
    }

    #[test]
    fn array_element_path_metadata_accepts_get_path_sources() {
        let expr = parse(r#"profile.get_path("events").last().kind"#)
            .expect("parse get_path array selector");
        let kernel = BodyKernel::classify_expr(&expr);
        let (source_keys, selector, suffix_keys) = kernel
            .array_element_path()
            .expect("array element path metadata");

        assert_eq!(
            source_keys
                .iter()
                .map(|key| key.as_ref())
                .collect::<Vec<_>>(),
            vec!["profile", "events"]
        );
        assert_eq!(selector, super::ArraySelector::Last);
        assert_eq!(
            suffix_keys
                .iter()
                .map(|key| key.as_ref())
                .collect::<Vec<_>>(),
            vec!["kind"]
        );
    }

    #[test]
    fn composed_field_demand_prefixes_downstream_paths() {
        let kernel = BodyKernel::Compose {
            first: Box::new(BodyKernel::FieldRead(Arc::from("user"))),
            then: Box::new(BodyKernel::FieldRead(Arc::from("name"))),
        };

        assert_eq!(field_paths(&kernel), vec!["user.name"]);

        let computed_receiver = BodyKernel::Compose {
            first: Box::new(BodyKernel::ArraySelect {
                array: Box::new(BodyKernel::FieldRead(Arc::from("events"))),
                selector: super::ArraySelector::Last,
            }),
            then: Box::new(BodyKernel::FieldRead(Arc::from("kind"))),
        };

        assert_eq!(field_paths(&computed_receiver), vec!["events"]);
    }

    #[test]
    fn nested_plan_field_demand_prefixes_inner_payload() {
        let expr =
            parse("items.filter(price > 20).map(isbn).last()").expect("parse nested pipeline");
        let kernel = BodyKernel::classify_expr(&expr);

        assert!(matches!(kernel, BodyKernel::NestedPlan(_)), "{kernel:#?}");
        assert_eq!(field_paths(&kernel), vec!["items.price", "items.isbn"]);
    }

    #[test]
    fn nested_plan_reuses_cached_field_chain_view_body() {
        let expr =
            parse("items.filter(price > 20).map(isbn).last()").expect("parse nested pipeline");
        let BodyKernel::NestedPlan(plan) = BodyKernel::classify_expr(&expr) else {
            panic!("expected nested plan");
        };
        let first = plan
            .view_plan()
            .expect("field-chain nested plan should expose view plan");
        let second = plan
            .view_plan()
            .expect("field-chain nested plan should expose cached view plan");

        assert!(std::ptr::eq(first, second));
        assert!(matches!(
            first.source(),
            super::NestedViewSource::FieldChain(_)
        ));
        assert!(first.body().can_run_with_view());
    }

    #[test]
    fn guarded_object_field_classifies_as_view_native() {
        let expr = parse("{id, isbn: isbn when active}").expect("parse guarded object");
        let kernel = BodyKernel::classify_expr(&expr);

        let BodyKernel::Object(object) = kernel else {
            panic!("expected object kernel");
        };
        assert!(object.entries()[1].cond().is_some());
        assert!(BodyKernel::Object(object).is_view_native());
    }

    #[test]
    fn nested_array_reducer_field_demand_prefixes_child_payload() {
        let kernel = BodyKernel::classify_expr(
            &parse("items.filter(price > 6).map(qty * price).sum()").expect("parse nested reducer"),
        );

        assert!(
            matches!(kernel, BodyKernel::NestedArrayReducer { .. }),
            "{kernel:#?}"
        );
        assert_eq!(
            field_paths(&kernel),
            vec!["items", "items.price", "items.qty"]
        );

        let count = BodyKernel::classify_expr(
            &parse("items.filter(price > 6).count()").expect("parse nested count"),
        );
        assert!(
            matches!(count, BodyKernel::NestedArrayCount { .. }),
            "{count:#?}"
        );
        assert_eq!(field_paths(&count), vec!["items", "items.price"]);
    }

    #[test]
    fn or_predicate_kernels_run_on_value_views() {
        let src = r#"user.tier == "gold" or user.tier == "platinum""#;
        let expr = parse(src).expect("parse or predicate");
        let program = Compiler::compile(&expr, src);
        let kernel = BodyKernel::classify(&program);
        assert!(matches!(kernel, BodyKernel::Or(_)));
        assert!(kernel.is_view_native());

        let gold = Val::obj(
            [(
                Arc::from("user"),
                Val::obj([(Arc::from("tier"), Val::Str(Arc::from("gold")))].into()),
            )]
            .into(),
        );
        let bronze = Val::obj(
            [(
                Arc::from("user"),
                Val::obj([(Arc::from("tier"), Val::Str(Arc::from("bronze")))].into()),
            )]
            .into(),
        );

        assert_eq!(
            owned_bool(eval_view_kernel(&kernel, &ValView::new(&gold))),
            Some(true)
        );
        assert_eq!(
            owned_bool(eval_view_kernel(&kernel, &ValView::new(&bronze))),
            Some(false)
        );
    }

    #[test]
    fn match_kernels_run_on_selected_value_views() {
        let src = r#"match events.last() with {
            {kind: "delivered", at: t} -> {state: "ok", at: t},
            {kind: "refund", reason: r} -> {state: "refund", reason: r},
            _ -> {state: "unknown"}
        }"#;
        let expr = parse(src).expect("parse match");
        let program = Compiler::compile(&expr, src);
        let kernel = BodyKernel::classify(&program);
        assert!(matches!(kernel, BodyKernel::Match { .. }));
        assert!(kernel.is_view_native());

        let value = Val::obj(
            [(
                Arc::from("events"),
                Val::arr(vec![
                    Val::obj([(Arc::from("kind"), Val::Str(Arc::from("placed")))].into()),
                    Val::obj(
                        [
                            (Arc::from("kind"), Val::Str(Arc::from("delivered"))),
                            (Arc::from("at"), Val::Str(Arc::from("2025-04-14"))),
                        ]
                        .into(),
                    ),
                ]),
            )]
            .into(),
        );
        let view = ValView::new(&value);
        let out = owned_value(eval_view_kernel(&kernel, &view)).expect("match output");
        let json: serde_json::Value = out.into();

        assert_eq!(json, serde_json::json!({"state": "ok", "at": "2025-04-14"}));
    }

    #[test]
    fn match_kernel_tracks_whether_bodies_need_current() {
        let binding_only = r#"match events.last() with {
            {kind: "delivered", at: t} -> {state: "ok", at: t},
            _ -> {state: "unknown"}
        }"#;
        let expr = parse(binding_only).expect("parse binding-only match");
        let program = Compiler::compile(&expr, binding_only);
        let kernel = BodyKernel::classify(&program);
        let BodyKernel::Match {
            body_needs_current, ..
        } = kernel
        else {
            panic!("expected match kernel");
        };
        assert!(!body_needs_current);

        let current_body = r#"match events.last() with {
            _ -> @.kind
        }"#;
        let expr = parse(current_body).expect("parse current-body match");
        let program = Compiler::compile(&expr, current_body);
        let kernel = BodyKernel::classify(&program);
        let BodyKernel::Match {
            body_needs_current, ..
        } = kernel
        else {
            panic!("expected match kernel");
        };
        assert!(body_needs_current);
    }

    #[test]
    fn object_key_builtin_kernels_run_on_value_views() {
        let value = Val::obj(
            [
                (Arc::from("isbn"), Val::Str(Arc::from("x"))),
                (Arc::from("score"), Val::Int(10)),
            ]
            .into(),
        );
        let view = ValView::new(&value);

        assert_eq!(
            owned_bool(eval_view_kernel(
                &key_call(BuiltinMethod::Has, "isbn"),
                &view
            )),
            Some(true)
        );
        assert_eq!(
            owned_bool(eval_view_kernel(
                &key_call(BuiltinMethod::HasKey, "isbn"),
                &view
            )),
            Some(true)
        );
        assert_eq!(
            owned_bool(eval_view_kernel(
                &key_call(BuiltinMethod::Missing, "title"),
                &view
            )),
            Some(true)
        );
        assert_eq!(
            owned_bool(eval_view_kernel(
                &key_call(BuiltinMethod::Missing, "isbn"),
                &view
            )),
            Some(false)
        );
        assert_eq!(
            owned_bool(eval_view_kernel(
                &key_vec_call(BuiltinMethod::HasAll, &["isbn", "score"]),
                &view
            )),
            Some(true)
        );
        assert_eq!(
            owned_bool(eval_view_kernel(
                &key_vec_call(BuiltinMethod::HasAll, &["title", "score"]),
                &view
            )),
            Some(false)
        );
    }

    #[test]
    fn ast_classifier_builds_deep_object_projection_kernel() {
        let expr = crate::parse::parser::parse(
            "{id, city: user.addr.city, item_count: items.len(), total: items.map(qty * price).sum(), label: f\"#{id}-{user.name}\"}",
        )
        .unwrap();
        let kernel = BodyKernel::classify_expr(&expr);
        assert!(matches!(kernel, BodyKernel::Object(_)), "{kernel:#?}");

        let row: Val = (&serde_json::json!({
            "id": 42,
            "user": {"name": "ada", "addr": {"city": "London"}},
            "items": [{"sku": "a", "qty": 2, "price": 10}, {"sku": "b", "qty": 3, "price": 5}]
        }))
            .into();
        let out = super::eval_native_kernel(&kernel, &row).unwrap();
        assert_eq!(out.get_field("id"), Val::Int(42));
        assert_eq!(out.get_field("city").as_str_ref(), Some("London"));
        assert_eq!(out.get_field("item_count"), Val::Int(2));
        assert_eq!(out.get_field("total"), Val::Int(35));
        assert_eq!(out.get_field("label").as_str_ref(), Some("#42-ada"));
    }

    #[test]
    fn nested_array_reducer_kernels_run_on_value_views() {
        let expr =
            parse("items.filter(price > 6).map(qty * price).sum()").expect("parse nested reducer");
        let kernel = BodyKernel::classify_expr(&expr);
        assert!(
            matches!(kernel, BodyKernel::NestedArrayReducer { .. }),
            "{kernel:#?}"
        );
        assert!(kernel.is_view_native());

        let row: Val = (&serde_json::json!({
            "items": [
                {"qty": 2, "price": 10},
                {"qty": 3, "price": 5},
                {"qty": 1, "price": 8}
            ]
        }))
            .into();
        assert_eq!(
            owned_value(eval_view_kernel(&kernel, &ValView::new(&row))),
            Some(Val::Int(28))
        );

        let count = BodyKernel::classify_expr(
            &parse("items.filter(price > 6).count()").expect("parse nested count"),
        );
        assert!(
            matches!(count, BodyKernel::NestedArrayCount { .. }),
            "{count:#?}"
        );
        assert!(count.is_view_native());
        assert_eq!(
            owned_value(eval_view_kernel(&count, &ValView::new(&row))),
            Some(Val::Int(2))
        );
    }

    #[test]
    fn ast_object_classifier_preserves_match_kernels() {
        let expr = parse(
            r#"{last_event: match events.last() with {
                {kind: "delivered", at: t} -> {state: "ok", at: t},
                _ -> {state: "unknown"}
            }}"#,
        )
        .expect("parse object match");
        let kernel = BodyKernel::classify_expr(&expr);
        assert!(matches!(kernel, BodyKernel::Object(_)), "{kernel:#?}");
        assert!(kernel.is_view_native());

        let row: Val = (&serde_json::json!({
            "events": [
                {"kind": "placed"},
                {"kind": "delivered", "at": "2025-04-14"}
            ]
        }))
            .into();
        let out = owned_value(eval_view_kernel(&kernel, &ValView::new(&row))).expect("output");
        let json: serde_json::Value = out.into();
        assert_eq!(
            json,
            serde_json::json!({"last_event": {"state": "ok", "at": "2025-04-14"}})
        );
    }

    #[test]
    fn missing_key_kernel_treats_null_as_missing_on_value_views() {
        let value = Val::obj([(Arc::from("isbn"), Val::Null)].into());
        let view = ValView::new(&value);

        assert_eq!(
            owned_bool(eval_view_kernel(
                &key_call(BuiltinMethod::Missing, "isbn"),
                &view
            )),
            Some(true)
        );
    }

    #[test]
    fn has_kernel_preserves_array_and_string_membership_on_value_views() {
        let tags = Val::arr(vec![Val::Str(Arc::from("sf")), Val::Str(Arc::from("hugo"))]);
        let tags_view = ValView::new(&tags);
        assert_eq!(
            owned_bool(eval_view_kernel(
                &key_call(BuiltinMethod::Has, "sf"),
                &tags_view
            )),
            Some(true)
        );
        assert_eq!(
            owned_bool(eval_view_kernel(
                &key_call(BuiltinMethod::HasKey, "sf"),
                &tags_view
            )),
            Some(false)
        );
        assert_eq!(
            owned_bool(eval_view_kernel(
                &key_vec_call(BuiltinMethod::HasAll, &["missing", "hugo"]),
                &tags_view
            )),
            Some(false)
        );

        let text = Val::Str(Arc::from("science fiction"));
        let text_view = ValView::new(&text);
        assert_eq!(
            owned_bool(eval_view_kernel(
                &key_call(BuiltinMethod::Has, "fiction"),
                &text_view
            )),
            Some(true)
        );
    }

    #[test]
    fn path_builtin_kernels_run_on_value_views() {
        let value = Val::obj(
            [(
                Arc::from("user"),
                Val::obj([(Arc::from("name"), Val::Str(Arc::from("ada")))].into()),
            )]
            .into(),
        );
        let view = ValView::new(&value);

        let name = eval_view_kernel(&key_call(BuiltinMethod::GetPath, "user.name"), &view);
        match name {
            Some(ViewKernelValue::View(view)) => {
                assert_eq!(view.materialize(), Val::Str(Arc::from("ada")))
            }
            _ => panic!("expected borrowed path view"),
        }
        assert_eq!(
            owned_bool(eval_view_kernel(
                &key_call(BuiltinMethod::HasPath, "user.name"),
                &view
            )),
            Some(true)
        );
        assert_eq!(
            owned_bool(eval_view_kernel(
                &key_call(BuiltinMethod::HasPath, "user.missing"),
                &view
            )),
            Some(false)
        );
    }

    #[test]
    fn object_helper_builtin_kernels_run_on_value_views() {
        let value = Val::obj(
            [
                (Arc::from("title"), Val::Str(Arc::from("b"))),
                (Arc::from("score"), Val::Int(2)),
                (Arc::from("debug"), Val::Bool(false)),
            ]
            .into(),
        );
        let view = ValView::new(&value);

        let pick = BodyKernel::BuiltinCall {
            receiver: Box::new(BodyKernel::Current),
            call: BuiltinCall::new(
                BuiltinMethod::Pick,
                BuiltinArgs::StrVec(vec![Arc::from("title"), Arc::from("score")]),
            ),
        };
        let picked = eval_view_kernel(&pick, &view).and_then(|value| match value {
            ViewKernelValue::Owned(value) => Some(value),
            _ => None,
        });
        let picked_json: serde_json::Value = picked.expect("pick output").into();
        assert_eq!(picked_json, serde_json::json!({"title": "b", "score": 2}));

        let omit = BodyKernel::BuiltinCall {
            receiver: Box::new(BodyKernel::Current),
            call: BuiltinCall::new(
                BuiltinMethod::Omit,
                BuiltinArgs::StrVec(vec![Arc::from("debug")]),
            ),
        };
        let omitted = eval_view_kernel(&omit, &view).and_then(|value| match value {
            ViewKernelValue::Owned(value) => Some(value),
            _ => None,
        });
        let omitted_json: serde_json::Value = omitted.expect("omit output").into();
        assert_eq!(omitted_json, serde_json::json!({"title": "b", "score": 2}));

        for method in [
            BuiltinMethod::Values,
            BuiltinMethod::Entries,
            BuiltinMethod::ToPairs,
        ] {
            let kernel = BodyKernel::BuiltinCall {
                receiver: Box::new(BodyKernel::Current),
                call: BuiltinCall::new(method, BuiltinArgs::None),
            };
            assert!(matches!(
                eval_view_kernel(&kernel, &view),
                Some(ViewKernelValue::Owned(_))
            ));
        }
    }

    #[test]
    fn owned_view_projection_receivers_keep_followup_calls_native() {
        let expr = parse("profile.entries().first()").unwrap();
        let kernel = BodyKernel::classify_expr(&expr);
        assert!(kernel.is_view_native());

        let value = Val::obj(
            [(
                Arc::from("profile"),
                Val::obj(
                    [
                        (Arc::from("role"), Val::Str(Arc::from("admin"))),
                        (Arc::from("tier"), Val::Int(2)),
                    ]
                    .into(),
                ),
            )]
            .into(),
        );
        let out = eval_view_kernel(&kernel, &ValView::new(&value)).and_then(|value| match value {
            ViewKernelValue::Owned(value) => Some(value),
            _ => None,
        });
        let out: serde_json::Value = out.expect("owned first entry").into();
        assert_eq!(out, serde_json::json!(["role", "admin"]));
    }

    #[test]
    fn owned_view_projection_receivers_apply_followup_chain_to_owned_value() {
        let expr = parse(r#"profile.pick("role", "contact").keys().last()"#).unwrap();
        let kernel = BodyKernel::classify_expr(&expr);
        assert!(kernel.is_view_native());

        let value = Val::obj(
            [(
                Arc::from("profile"),
                Val::obj(
                    [
                        (Arc::from("role"), Val::Str(Arc::from("admin"))),
                        (
                            Arc::from("contact"),
                            Val::obj(
                                [(Arc::from("email"), Val::Str(Arc::from("a@example.test")))]
                                    .into(),
                            ),
                        ),
                        (Arc::from("flags"), Val::Bool(true)),
                    ]
                    .into(),
                ),
            )]
            .into(),
        );
        let out = eval_view_kernel(&kernel, &ValView::new(&value)).and_then(|value| match value {
            ViewKernelValue::Owned(value) => Some(value),
            _ => None,
        });
        assert_eq!(out, Some(Val::Str(Arc::from("contact"))));

        let program =
            crate::compile::compiler::Compiler::compile(&expr, "<owned-view-projection-test>");
        let compiled_kernel = BodyKernel::classify(&program);
        assert!(compiled_kernel.is_view_native());
        let out =
            eval_view_kernel(&compiled_kernel, &ValView::new(&value)).and_then(
                |value| match value {
                    ViewKernelValue::Owned(value) => Some(value),
                    _ => None,
                },
            );
        assert_eq!(out, Some(Val::Str(Arc::from("contact"))));

        let method_arg_program =
            crate::exec::pipeline::lower::compile_subexpr(&crate::parse::ast::Arg::Pos(expr))
                .expect("method arg program");
        let method_arg_kernel = BodyKernel::classify(&method_arg_program);
        assert!(method_arg_kernel.is_view_native());
        let mut expected_fields = crate::plan::demand::FieldSet::new();
        expected_fields.insert(crate::plan::demand::FieldPath::chain(Arc::from([
            Arc::<str>::from("profile"),
            Arc::<str>::from("role"),
        ])));
        expected_fields.insert(crate::plan::demand::FieldPath::chain(Arc::from([
            Arc::<str>::from("profile"),
            Arc::<str>::from("contact"),
        ])));
        assert_eq!(
            method_arg_kernel.field_demand(),
            crate::plan::demand::FieldDemand::Fields(expected_fields)
        );
        let out =
            eval_view_kernel(&method_arg_kernel, &ValView::new(&value)).and_then(
                |value| match value {
                    ViewKernelValue::Owned(value) => Some(value),
                    _ => None,
                },
            );
        assert_eq!(out, Some(Val::Str(Arc::from("contact"))));
    }
}
