//! Classified expression kernels for pipeline stage evaluation.
//!
//! `BodyKernel` is a pre-classified form of a stage body expression that lets
//! the pipeline executor skip the VM for common patterns (field reads,
//! literal comparisons, `FieldCmpLit` fusions). Generic/unknown expressions
//! fall back to `BodyKernel::Generic`, which re-enters the VM.

use std::sync::Arc;

use crate::builtins::registry::{
    array_selector as builtin_array_selector, by_name as builtin_by_name,
    count_sink_accepts_predicate, expr_stage, numeric_reducer, view_object_projection,
    view_projection, BuiltinExprStage, BuiltinId,
};
use crate::builtins::{BuiltinArraySelector, BuiltinCall, BuiltinViewObjectProjection};
use crate::data::context::EvalError;
use crate::data::value::Val;
use crate::data::view::{scalar_view_to_owned_val, ValueView};
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
}

impl NestedPlanKernel {
    pub(crate) fn new(plan: Arc<super::Plan>) -> Self {
        let prepared = super::nested::PreparedPlan::new(&plan);
        Self { plan, prepared }
    }

    pub(crate) fn parent_field_demand(&self) -> FieldDemand {
        self.plan.parent_field_demand()
    }

    pub(crate) fn run(&self, seed: Val) -> Result<Val, EvalError> {
        self.prepared.run(seed)
    }

    pub(crate) fn run_view<'a, V>(
        &self,
        seed: &V,
        vm: &mut crate::vm::VM,
    ) -> Option<Result<Val, EvalError>>
    where
        V: ValueView<'a>,
    {
        super::nested::run_plan_view(&self.plan, seed, vm)
    }
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
    if !matches!(receiver, BodyKernel::Current) {
        return None;
    }
    match (
        view_object_projection(BuiltinId::from_method(call.method))?,
        &call.args,
    ) {
        (
            BuiltinViewObjectProjection::HasKey | BuiltinViewObjectProjection::Missing,
            crate::builtins::BuiltinArgs::Str(key),
        ) => {
            Some(FieldDemand::Fields(FieldSet::single(Arc::clone(key))))
        }
        (BuiltinViewObjectProjection::Missing, crate::builtins::BuiltinArgs::StrVec(keys)) => {
            let mut fields = FieldSet::new();
            for key in keys {
                fields.insert(crate::plan::demand::FieldPath::single(Arc::clone(key)));
            }
            Some(FieldDemand::Fields(fields))
        }
        (
            BuiltinViewObjectProjection::GetPath | BuiltinViewObjectProjection::HasPath,
            crate::builtins::BuiltinArgs::Str(path),
        ) => {
            path_field_demand(&crate::builtins::parse_path_segs(path.as_ref()))
        }
        (
            BuiltinViewObjectProjection::GetPath | BuiltinViewObjectProjection::HasPath,
            crate::builtins::BuiltinArgs::Path(path),
        ) => path_field_demand(path),
        _ => None,
    }
}

fn path_field_demand(path: &[crate::builtins::PathSeg]) -> Option<FieldDemand> {
    let mut keys: Vec<Arc<str>> = Vec::new();
    for segment in path {
        match segment {
            crate::builtins::PathSeg::Field(key) => keys.push(Arc::from(key.as_str())),
            crate::builtins::PathSeg::Index(_) => break,
        }
    }
    match keys.len() {
        0 => None,
        1 => Some(FieldDemand::Fields(FieldSet::single(keys.remove(0)))),
        _ => Some(FieldDemand::Fields(FieldSet::chain(keys.into()))),
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

/// A single part of an `FStringKernel`: either a fixed literal or a dynamic interpolation.
#[derive(Debug, Clone)]
pub enum FStringKernelPart {
    /// A constant string segment that is copied verbatim into the output.
    Lit(Arc<str>),
    /// A sub-kernel whose result is formatted and appended to the output string.
    Interp(BodyKernel),
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
    // key name in the produced object
    key: Arc<str>,
    // kernel used to compute the value for this key
    value: BodyKernel,
    // when true, a null result causes this entry to be silently omitted
    optional: bool,
    // when true, null values are omitted regardless of the optional flag
    omit_null: bool,
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
            .map(|entry| Arc::clone(&entry.key))
            .collect::<Vec<_>>()
            .into()
    }

    pub(crate) fn entries(&self) -> &[ObjectKernelEntry] {
        &self.entries
    }

    /// Evaluates each entry against `item`, appending to `cells`; returns `Some(false)` on null-optional, `None` on view failure.
    pub(crate) fn eval_view_row_cells<'a, V>(&self, item: &V, cells: &mut Vec<Val>) -> Option<bool>
    where
        V: ValueView<'a>,
    {
        let start = cells.len();
        for entry in self.entries.iter() {
            let value = match eval_view_kernel(&entry.value, item)? {
                ViewKernelValue::View(view) => {
                    scalar_view_to_owned_val(view.scalar()).unwrap_or_else(|| view.materialize())
                }
                ViewKernelValue::Owned(value) => value,
            };
            if (entry.optional || entry.omit_null) && value.is_null() {
                cells.truncate(start);
                return Some(false);
            }
            cells.push(value);
        }
        Some(true)
    }

    /// Evaluates each entry against `item`, appending to `cells`; returns `false` on null-optional skip.
    pub(crate) fn eval_val_row_cells(&self, item: &Val, cells: &mut Vec<Val>) -> bool {
        let start = cells.len();
        for entry in self.entries.iter() {
            let value = eval_native_kernel(&entry.value, item).unwrap_or(Val::Null);
            if (entry.optional || entry.omit_null) && value.is_null() {
                cells.truncate(start);
                return false;
            }
            cells.push(value);
        }
        true
    }

    /// Evaluates all entries against `item` into a `Val::ObjSmall`, returning `Val::Null` on sub-kernel failure.
    pub(crate) fn eval_val(&self, item: &Val) -> Val {
        eval_object_kernel(self, |kernel| eval_native_kernel(kernel, item)).unwrap_or(Val::Null)
    }

    /// Returns the source-row field payload needed to evaluate every value entry.
    pub(crate) fn field_demand(&self) -> FieldDemand {
        self.entries.iter().fold(FieldDemand::None, |need, entry| {
            need.merge(entry.value.field_demand())
        })
    }
}

impl ObjectKernelEntry {
    pub(crate) fn key(&self) -> &Arc<str> {
        &self.key
    }

    pub(crate) fn value(&self) -> &BodyKernel {
        &self.value
    }

    pub(crate) fn optional(&self) -> bool {
        self.optional
    }

    pub(crate) fn omit_null(&self) -> bool {
        self.omit_null
    }
}

fn classify_object_expr(fields: &[ObjField]) -> BodyKernel {
    let mut entries = Vec::with_capacity(fields.len());
    for field in fields {
        let (key, value, optional, omit_null) = match field {
            ObjField::Short(name) => (
                Arc::from(name.as_str()),
                BodyKernel::FieldRead(Arc::from(name.as_str())),
                false,
                true,
            ),
            ObjField::Kv {
                key,
                val,
                optional,
                cond: None,
            } => {
                let value = BodyKernel::classify_expr(val);
                if matches!(value, BodyKernel::Generic) {
                    return BodyKernel::Generic;
                }
                (Arc::from(key.as_str()), value, *optional, false)
            }
            _ => return BodyKernel::Generic,
        };
        entries.push(ObjectKernelEntry {
            key,
            value,
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
    for elem in elems {
        let crate::parse::ast::ArrayElem::Expr(expr) = elem else {
            return BodyKernel::Generic;
        };
        let item = BodyKernel::classify_expr(expr);
        if matches!(item, BodyKernel::Generic) {
            return BodyKernel::Generic;
        }
        out.push(item);
    }
    BodyKernel::Array(out.into())
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
        (id, None, None) if count_sink_accepts_predicate(id) => Some(BodyKernel::NestedArrayCount {
            source: Box::new(source),
            predicate,
        }),
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
                if !view_projection(BuiltinId::from_method(call.method)) {
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
            _ => return BodyKernel::Generic,
        }
    }

    receiver
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
                    Self::And(vec![lhs, rhs].into())
                } else if matches!(op, crate::parse::ast::BinOp::Or) {
                    Self::Or(vec![lhs, rhs].into())
                } else if is_comparison_op(*op) {
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
            Expr::Chain(base, steps) => classify_chain_expr(base, steps),
            Expr::Match { .. } => {
                let program = crate::compile::compiler::Compiler::compile(expr, "<match-kernel>");
                Self::classify(&program)
            }
            _ => Self::Generic,
        }
    }

    /// Returns the field payload needed from the current row to evaluate this kernel.
    pub(crate) fn field_demand(&self) -> FieldDemand {
        match self {
            Self::Generic | Self::Current | Self::CurrentCmpLit(_, _) => {
                FieldDemand::Whole
            }
            Self::FieldRead(key) | Self::FieldCmpLit(key, _, _) => {
                FieldDemand::Fields(FieldSet::single(Arc::clone(key)))
            }
            Self::FieldChain(keys) | Self::FieldChainCmpLit(keys, _, _) => {
                FieldDemand::Fields(FieldSet::chain(Arc::clone(keys)))
            }
            Self::BuiltinCall { receiver, call } => {
                object_key_call_field_demand(receiver, call)
                    .unwrap_or_else(|| receiver.field_demand())
            }
            Self::Compose { first, then } => compose_field_demand(first, then),
            Self::CmpLit { lhs, .. } => lhs.field_demand(),
            Self::Binary { lhs, rhs, .. } => lhs.field_demand().merge(rhs.field_demand()),
            Self::ArraySelect { array, .. } => array.field_demand(),
            Self::Match { .. } => FieldDemand::Whole,
            Self::And(predicates) | Self::Or(predicates) => predicates
                .iter()
                .fold(FieldDemand::None, |need, predicate| {
                    need.merge(predicate.field_demand())
                }),
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
            Self::NestedArrayReducer {
                source,
                predicate,
                map,
                ..
            } => {
                let need = source.field_demand();
                let need = match predicate {
                    Some(predicate) => need.merge(predicate.field_demand()),
                    None => need,
                };
                match map {
                    Some(map) => need.merge(map.field_demand()),
                    None => need,
                }
            }
            Self::NestedArrayCount { source, predicate } => {
                let need = source.field_demand();
                match predicate {
                    Some(predicate) => need.merge(predicate.field_demand()),
                    None => need,
                }
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
            Self::Match { scrutinee, .. } => scrutinee.mentions_any_field_like_ident(names),
            Self::And(predicates) | Self::Or(predicates) => predicates
                .iter()
                .any(|predicate| predicate.mentions_any_field_like_ident(names)),
            Self::FString(fstring) => fstring.parts.iter().any(|part| match part {
                FStringKernelPart::Lit(_) => false,
                FStringKernelPart::Interp(kernel) => kernel.mentions_any_field_like_ident(names),
            }),
            Self::Object(object) => object
                .entries
                .iter()
                .any(|entry| entry.value.mentions_any_field_like_ident(names)),
            Self::Array(items) => items
                .iter()
                .any(|item| item.mentions_any_field_like_ident(names)),
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
                receiver.is_view_native() && view_projection(BuiltinId::from_method(call.method))
            }
            Self::Compose { first, then } => first.is_view_native() && then.is_view_native(),
            Self::CmpLit { lhs, .. } => lhs.is_view_native(),
            Self::Binary { lhs, rhs, .. } => lhs.is_view_native() && rhs.is_view_native(),
            Self::ArraySelect { array, .. } => array.is_view_native(),
            Self::Match { scrutinee, .. } => scrutinee.is_view_native(),
            Self::And(predicates) | Self::Or(predicates) => {
                predicates.iter().all(Self::is_view_native)
            }
            Self::Object(object) => object
                .entries
                .iter()
                .all(|entry| entry.value.is_view_native()),
            Self::Array(items) => items.iter().all(Self::is_view_native),
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
            Self::NestedPlan(_) => false,
            _ => true,
        }
    }

    /// Returns the `CollectLayout` hint indicating whether outputs form a uniform-object or generic collection.
    pub(crate) fn collect_layout(&self) -> CollectLayout<'_> {
        match self {
            Self::Object(object) if object.len() > 0 => CollectLayout::UniformObject(object),
            _ => CollectLayout::Values,
        }
    }

    /// Classifies a compiled `Program` into the most specific `BodyKernel` variant, falling back to `Generic`.
    pub fn classify(prog: &crate::vm::Program) -> Self {
        use crate::vm::Opcode;
        let ops = prog.ops.as_ref();
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
            [Opcode::MakeObj(entries)] => {
                let mut out = Vec::with_capacity(entries.len());
                for entry in entries.iter() {
                    let (key, value, optional, omit_null) = match entry {
                        crate::vm::CompiledObjEntry::Short { name, .. } => (
                            Arc::clone(name),
                            BodyKernel::FieldRead(Arc::clone(name)),
                            false,
                            true,
                        ),
                        crate::vm::CompiledObjEntry::Kv {
                            key,
                            prog,
                            optional,
                            cond: None,
                        } => {
                            let value = BodyKernel::classify(prog);
                            if matches!(value, BodyKernel::Generic) {
                                return Self::Generic;
                            }
                            (Arc::clone(key), value, *optional, false)
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
                            (Arc::clone(key), value, *optional, false)
                        }
                        _ => return Self::Generic,
                    };
                    out.push(ObjectKernelEntry {
                        key,
                        value,
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
                for (program, spread) in items.iter() {
                    if *spread {
                        return Self::Generic;
                    }
                    let item = BodyKernel::classify(program);
                    if matches!(item, BodyKernel::Generic) {
                        return Self::Generic;
                    }
                    out.push(item);
                }
                return Self::Array(out.into());
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

#[inline]
fn is_comparison_op(op: crate::parse::ast::BinOp) -> bool {
    matches!(
        op,
        crate::parse::ast::BinOp::Eq
            | crate::parse::ast::BinOp::Neq
            | crate::parse::ast::BinOp::Lt
            | crate::parse::ast::BinOp::Lte
            | crate::parse::ast::BinOp::Gt
            | crate::parse::ast::BinOp::Gte
            | crate::parse::ast::BinOp::Fuzzy
    )
}

fn literal_kernel_value(kernel: &BodyKernel) -> Option<Val> {
    match kernel {
        BodyKernel::Const(value) => Some(value.clone()),
        BodyKernel::ConstBool(value) => Some(Val::Bool(*value)),
        _ => None,
    }
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
    matches!(
        cm.scrutinee,
        crate::vm::MatchScrutinee::Current | crate::vm::MatchScrutinee::Program(_)
    ) && cm.guards.is_empty()
        && cm
            .bodies
            .iter()
            .all(|program| program_is_receiver_local(program))
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

fn program_is_receiver_local(program: &crate::vm::Program) -> bool {
    program.ops.iter().all(opcode_is_receiver_local)
}

fn opcode_is_receiver_local(opcode: &crate::vm::Opcode) -> bool {
    use crate::vm::Opcode;
    match opcode {
        Opcode::PushRoot | Opcode::RootChain(_) => false,
        Opcode::PipelineRun { .. }
        | Opcode::ListComp(_)
        | Opcode::DictComp(_)
        | Opcode::SetComp(_)
        | Opcode::PatchEval(_)
        | Opcode::UpdateBatchEval(_)
        | Opcode::DeepMatchAll(_)
        | Opcode::DeepMatchFirst(_) => false,
        Opcode::Match(cm) => match_is_receiver_local(cm),
        Opcode::DynIndex(prog)
        | Opcode::InlineFilter(prog)
        | Opcode::AndOp(prog)
        | Opcode::OrOp(prog)
        | Opcode::CoalesceOp(prog) => program_is_receiver_local(prog),
        Opcode::BindLamCurrent { body, .. } => program_is_receiver_local(body),
        Opcode::CallMethod(call) | Opcode::CallOptMethod(call) => call
            .sub_progs
            .iter()
            .all(|prog| program_is_receiver_local(prog)),
        Opcode::MakeObj(entries) => entries.iter().all(|entry| match entry {
            crate::vm::CompiledObjEntry::Short { .. }
            | crate::vm::CompiledObjEntry::KvPath { .. } => true,
            crate::vm::CompiledObjEntry::Kv { prog, cond, .. } => {
                program_is_receiver_local(prog)
                    && cond
                        .as_ref()
                        .is_none_or(|cond| program_is_receiver_local(cond))
            }
            crate::vm::CompiledObjEntry::Dynamic { key, val } => {
                program_is_receiver_local(key) && program_is_receiver_local(val)
            }
            crate::vm::CompiledObjEntry::Spread(prog)
            | crate::vm::CompiledObjEntry::SpreadDeep(prog) => program_is_receiver_local(prog),
        }),
        Opcode::MakeArr(items) => items
            .iter()
            .all(|(program, _)| program_is_receiver_local(program)),
        Opcode::FString(parts) => parts.iter().all(|part| match part {
            crate::vm::CompiledFSPart::Lit(_) => true,
            crate::vm::CompiledFSPart::Interp { prog, .. } => program_is_receiver_local(prog),
        }),
        Opcode::LetExpr { body, .. } => program_is_receiver_local(body),
        Opcode::IfElse { then_, else_ } => {
            program_is_receiver_local(then_) && program_is_receiver_local(else_)
        }
        Opcode::TryExpr { body, default } => {
            program_is_receiver_local(body) && program_is_receiver_local(default)
        }
        _ => true,
    }
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
                stack.push(BodyKernel::ArraySelect {
                    array: Box::new(receiver),
                    selector: array_selector_call(call)?,
                });
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

/// Describes how a Map stage's output elements should be collected by the sink.
pub(crate) enum CollectLayout<'a> {
    /// Output elements are heterogeneous `Val`s; collect into a plain array.
    Values,
    /// Every output element is a uniform object with the same key schema; collect into a columnar layout.
    UniformObject(&'a ObjectKernel),
}

// numeric index steps cannot be represented as a FieldChain, so return None
fn classify_kv_path(steps: &[crate::vm::KvStep]) -> Option<BodyKernel> {
    if steps.is_empty() {
        return None;
    }
    let mut keys = Vec::with_capacity(steps.len());
    for step in steps {
        match step {
            crate::vm::KvStep::Field(key) => keys.push(Arc::clone(key)),
            crate::vm::KvStep::Index(_) => return None,
        }
    }
    match keys.len() {
        0 => None,
        1 => Some(BodyKernel::FieldRead(keys.pop().unwrap())),
        _ => Some(BodyKernel::FieldChain(keys.into())),
    }
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
        [receiver @ .., Opcode::CallMethod(call)]
            if view_projection(BuiltinId::from_method(call.method)) =>
        {
            let receiver = if receiver.is_empty() {
                BodyKernel::Current
            } else {
                classify_structural_view_kernel(receiver)?
            };
            let builtin_call = if view_object_projection(BuiltinId::from_method(call.method))
                == Some(BuiltinViewObjectProjection::Pick)
            {
                static_positional_pick_call(call)?
            } else {
                BuiltinCall::from_static_args(
                    call.method,
                    call.name.as_ref(),
                    call.orig_args.len(),
                    |idx| {
                        Ok(call
                            .sub_progs
                            .get(idx)
                            .and_then(|prog| static_prog_val(prog)))
                    },
                    |idx| match call.orig_args.get(idx) {
                        Some(crate::parse::ast::Arg::Pos(crate::parse::ast::Expr::Ident(
                            value,
                        ))) => Some(Arc::from(value.as_str())),
                        _ => None,
                    },
                )
                .ok()
                .flatten()?
            };
            if !view_projection(BuiltinId::from_method(builtin_call.method)) {
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

fn walk_view_path<'a, V>(mut cur: V, segs: &[crate::builtins::PathSeg]) -> V
where
    V: ValueView<'a>,
{
    for seg in segs {
        cur = match seg {
            crate::builtins::PathSeg::Field(field) => cur.field(field.as_str()),
            crate::builtins::PathSeg::Index(index) => cur.index(*index),
        };
    }
    cur
}

fn static_prog_val(prog: &crate::vm::Program) -> Option<Val> {
    match prog.ops.as_ref() {
        [op] => trivial_lit(op),
        _ => None,
    }
}

fn static_positional_pick_call(call: &crate::vm::CompiledCall) -> Option<BuiltinCall> {
    let mut keys = Vec::with_capacity(call.orig_args.len());
    for (idx, arg) in call.orig_args.iter().enumerate() {
        let key = match arg {
            crate::parse::ast::Arg::Pos(crate::parse::ast::Expr::Ident(value)) => {
                Arc::from(value.as_str())
            }
            crate::parse::ast::Arg::Pos(_) => match call
                .sub_progs
                .get(idx)
                .and_then(|prog| static_prog_val(prog))
            {
                Some(Val::Str(value)) => value,
                _ => return None,
            },
            crate::parse::ast::Arg::Named(_, _) => return None,
        };
        keys.push(key);
    }
    Some(BuiltinCall::new(
        crate::builtins::BuiltinMethod::Pick,
        crate::builtins::BuiltinArgs::StrVec(keys),
    ))
}

fn array_selector_call(call: &crate::vm::CompiledCall) -> Option<ArraySelector> {
    match builtin_array_selector(BuiltinId::from_method(call.method))? {
        BuiltinArraySelector::First => Some(ArraySelector::First),
        BuiltinArraySelector::Last => Some(ArraySelector::Last),
        BuiltinArraySelector::Nth => match call.sub_progs.as_ref() {
            [prog] => match static_prog_val(prog)? {
                Val::Int(index) if index >= 0 => Some(ArraySelector::Nth(index as usize)),
                _ => None,
            },
            _ => None,
        },
    }
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

// panics on Generic — callers must route Generic through eval_kernel's fallback instead
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
        BodyKernel::FString(fstring) => {
            eval_fstring_kernel(fstring, |kernel| eval_native_kernel_with_vm(kernel, item, vm))
        }
        BodyKernel::Object(object) => {
            eval_object_kernel(object, |kernel| eval_native_kernel_with_vm(kernel, item, vm))
        }
        BodyKernel::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for item_kernel in items.iter() {
                out.push(eval_native_kernel_with_vm(item_kernel, item, vm)?);
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
        let value = eval(&entry.value)?;
        if (entry.optional || entry.omit_null) && value.is_null() {
            continue;
        }
        pairs.push((Arc::clone(&entry.key), value));
    }
    Ok(Val::ObjSmall(pairs.into()))
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
fn append_val_to_string(out: &mut String, value: &Val) -> Result<(), EvalError> {
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

// compound scalars (array/object len variants) fall through to view.materialize()
fn append_json_view_to_string<'a, V>(
    out: &mut String,
    view: &V,
    scalar: JsonView<'_>,
) -> Result<(), EvalError>
where
    V: ValueView<'a>,
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
            out.push_str(&crate::util::val_to_string(&view.materialize()));
        }
    }
    Ok(())
}

fn view_kernel_value_to_owned<'a, V>(value: ViewKernelValue<V>) -> Val
where
    V: ValueView<'a>,
{
    match value {
        ViewKernelValue::View(view) => view_kernel_view_to_owned(view),
        ViewKernelValue::Owned(value) => value,
    }
}

fn view_kernel_view_to_owned<'a, V>(view: V) -> Val
where
    V: ValueView<'a>,
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

fn eval_binary_op(lhs: Val, op: crate::parse::ast::BinOp, rhs: Val) -> Result<Val, EvalError> {
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
    let idx = match selector {
        ArraySelector::First => 0,
        ArraySelector::Last => match items.len().checked_sub(1) {
            Some(idx) => idx,
            None => return Val::Null,
        },
        ArraySelector::Nth(idx) => idx,
    };
    items.get(idx).cloned().unwrap_or(Val::Null)
}

fn eval_array_select_view<'a, V>(array: V, selector: ArraySelector) -> V
where
    V: ValueView<'a>,
{
    let idx = match selector {
        ArraySelector::First => Some(0),
        ArraySelector::Last => match array.scalar() {
            JsonView::ArrayLen(len) => len.checked_sub(1),
            _ => None,
        },
        ArraySelector::Nth(idx) => Some(idx),
    };
    idx.map(|idx| array.index(idx as i64))
        .unwrap_or_else(|| array.index(-1))
}

fn eval_nested_array_reducer_view<'a, V>(
    source: &BodyKernel,
    predicate: Option<&BodyKernel>,
    map: Option<&BodyKernel>,
    op: super::NumOp,
    item: &V,
    mut vm: Option<&mut crate::vm::VM>,
) -> Option<Val>
where
    V: ValueView<'a>,
{
    let source = match eval_view_kernel_inner(source, item, vm.as_deref_mut())? {
        ViewKernelValue::View(view) => view,
        ViewKernelValue::Owned(value) => {
            return eval_nested_array_reducer_native_owned(value, predicate, map, op)
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
        if !view_predicate_matches(predicate, &child, vm.as_deref_mut())? {
            return Some(());
        }
        match map {
            Some(map) => {
                if let Some(value) = eval_view_numeric_kernel(map, &child) {
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

                match eval_view_kernel_inner(map, &child, vm.as_deref_mut())? {
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
    mut vm: Option<&mut crate::vm::VM>,
) -> Option<Val>
where
    V: ValueView<'a>,
{
    let source = match eval_view_kernel_inner(source, item, vm.as_deref_mut())? {
        ViewKernelValue::View(view) => view,
        ViewKernelValue::Owned(value) => {
            return eval_nested_array_count_native_owned(value, predicate)
        }
    };
    let Some(predicate) = predicate else {
        return match source.scalar() {
            JsonView::ArrayLen(len) => Some(Val::Int(len as i64)),
            _ => Some(Val::Int(0)),
        };
    };
    let mut count = 0i64;
    let mut iter = source.array_iter()?;
    iter.try_for_each(|child| {
        if view_predicate_matches(Some(predicate), &child, vm.as_deref_mut())? {
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
    vm: Option<&mut crate::vm::VM>,
) -> Option<bool>
where
    V: ValueView<'a>,
{
    match predicate {
        Some(predicate) => match eval_view_kernel_inner(predicate, item, vm)? {
            ViewKernelValue::View(view) => Some(view.scalar().truthy()),
            ViewKernelValue::Owned(value) => Some(crate::util::is_truthy(&value)),
        },
        None => Some(true),
    }
}

fn eval_view_numeric_kernel<'a, V>(kernel: &BodyKernel, item: &V) -> Option<NumericKernelValue>
where
    V: ValueView<'a>,
{
    match kernel {
        BodyKernel::Current => numeric_from_json_view(item.scalar()),
        BodyKernel::FieldRead(key) => numeric_from_json_view(item.field(key).scalar()),
        BodyKernel::FieldChain(keys) => {
            numeric_from_json_view(walk_view_fields(item.clone(), keys.as_ref()).scalar())
        }
        BodyKernel::Const(Val::Int(value)) => Some(NumericKernelValue::Int(*value)),
        BodyKernel::Const(Val::Float(value)) => Some(NumericKernelValue::Float(*value)),
        BodyKernel::Binary { lhs, op, rhs } => {
            let lhs = eval_view_numeric_kernel(lhs, item)?;
            let rhs = eval_view_numeric_kernel(rhs, item)?;
            eval_numeric_binary(lhs, *op, rhs)
        }
        BodyKernel::Compose { first, then } => match eval_view_kernel(first, item)? {
            ViewKernelValue::View(view) => eval_view_numeric_kernel(then, &view),
            ViewKernelValue::Owned(value) => eval_native_numeric_kernel(then, &value),
        },
        BodyKernel::ArraySelect { array, selector } => match eval_view_kernel(array, item)? {
            ViewKernelValue::View(view) => {
                numeric_from_json_view(eval_array_select_view(view, *selector).scalar())
            }
            ViewKernelValue::Owned(value) => {
                numeric_from_val(&eval_array_select_native(&value, *selector))
            }
        },
        _ => None,
    }
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

fn eval_nested_array_reducer_native_owned(
    source: Val,
    predicate: Option<&BodyKernel>,
    map: Option<&BodyKernel>,
    op: super::NumOp,
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
        if !native_predicate_matches_opt(predicate, child)? {
            continue;
        }
        let value;
        let observed = match map {
            Some(map) => {
                value = eval_native_kernel(map, child).ok()?;
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

fn eval_nested_array_count_native_owned(
    source: Val,
    predicate: Option<&BodyKernel>,
) -> Option<Val> {
    let Some(items) = source.as_vals() else {
        return Some(Val::Int(0));
    };
    let Some(predicate) = predicate else {
        return Some(Val::Int(items.len() as i64));
    };
    let mut count = 0i64;
    for child in items.iter() {
        if native_predicate_matches_opt(Some(predicate), child)? {
            count += 1;
        }
    }
    Some(Val::Int(count))
}

#[inline]
fn native_predicate_matches_opt(predicate: Option<&BodyKernel>, item: &Val) -> Option<bool> {
    match predicate {
        Some(predicate) => eval_native_kernel(predicate, item)
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
#[inline]
pub(crate) fn eval_view_kernel<'a, V>(kernel: &BodyKernel, item: &V) -> Option<ViewKernelValue<V>>
where
    V: ValueView<'a>,
{
    eval_view_kernel_inner(kernel, item, None)
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
    V: ValueView<'a>,
{
    eval_view_kernel_inner(kernel, item, Some(vm))
}

fn eval_view_kernel_inner<'a, V>(
    kernel: &BodyKernel,
    item: &V,
    mut vm: Option<&mut crate::vm::VM>,
) -> Option<ViewKernelValue<V>>
where
    V: ValueView<'a>,
{
    match kernel {
        BodyKernel::Current => Some(ViewKernelValue::View(item.clone())),
        BodyKernel::FieldRead(key) => Some(ViewKernelValue::View(item.field(key))),
        BodyKernel::FieldChain(keys) => Some(ViewKernelValue::View(walk_view_fields(
            item.clone(),
            keys.as_ref(),
        ))),
        BodyKernel::ConstBool(value) => Some(ViewKernelValue::Owned(Val::Bool(*value))),
        BodyKernel::Const(value) => Some(ViewKernelValue::Owned(value.clone())),
        BodyKernel::FString(fstring) => {
            let mut out = String::with_capacity(fstring.base_capacity);
            for part in fstring.parts.iter() {
                match part {
                    FStringKernelPart::Lit(value) => out.push_str(value),
                    FStringKernelPart::Interp(kernel) => match eval_view_kernel_inner(kernel, item, vm.as_deref_mut())? {
                        ViewKernelValue::View(view) => {
                            append_json_view_to_string(&mut out, &view, view.scalar()).ok()?;
                        }
                        ViewKernelValue::Owned(value) => {
                            append_val_to_string(&mut out, &value).ok()?;
                        }
                    },
                }
            }
            Some(ViewKernelValue::Owned(Val::Str(Arc::from(out))))
        }
        BodyKernel::Object(object) => {
            let mut pairs = Vec::with_capacity(object.entries.len());
            for entry in object.entries.iter() {
                let value = match eval_view_kernel_inner(&entry.value, item, vm.as_deref_mut())? {
                    ViewKernelValue::View(view) => view_kernel_view_to_owned(view),
                    ViewKernelValue::Owned(value) => value,
                };
                if (entry.optional || entry.omit_null) && value.is_null() {
                    continue;
                }
                pairs.push((Arc::clone(&entry.key), value));
            }
            Some(ViewKernelValue::Owned(Val::ObjSmall(pairs.into())))
        }
        BodyKernel::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for item_kernel in items.iter() {
                out.push(view_kernel_value_to_owned(eval_view_kernel_inner(
                    item_kernel,
                    item,
                    vm.as_deref_mut(),
                )?));
            }
            Some(ViewKernelValue::Owned(Val::arr(out)))
        }
        BodyKernel::NestedArrayReducer {
            source,
            predicate,
            map,
            op,
        } => {
            eval_nested_array_reducer_view(
                source,
                predicate.as_deref(),
                map.as_deref(),
                *op,
                item,
                vm.as_deref_mut(),
            )
            .map(ViewKernelValue::Owned)
        }
        BodyKernel::NestedArrayCount { source, predicate } => {
            eval_nested_array_count_view(source, predicate.as_deref(), item, vm.as_deref_mut())
                .map(ViewKernelValue::Owned)
        }
        BodyKernel::NestedPlan(plan) => {
            let result = match vm.as_deref_mut() {
                Some(vm) => plan.run_view(item, vm),
                None => {
                    let mut local_vm = crate::vm::VM::new();
                    plan.run_view(item, &mut local_vm)
                }
            }
            .unwrap_or_else(|| plan.run(item.materialize()));
            result.ok().map(ViewKernelValue::Owned)
        }
        BodyKernel::BuiltinCall { receiver, call } => match eval_view_kernel_inner(receiver, item, vm.as_deref_mut())? {
            ViewKernelValue::View(view) => match (
                view_object_projection(BuiltinId::from_method(call.method)),
                &call.args,
            ) {
                (Some(BuiltinViewObjectProjection::Has), crate::builtins::BuiltinArgs::Str(key)) => {
                    view_has(&view, key.as_ref())
                        .map(|found| ViewKernelValue::Owned(Val::Bool(found)))
                }
                (
                    Some(BuiltinViewObjectProjection::HasAll),
                    crate::builtins::BuiltinArgs::StrVec(keys),
                ) => view_has_all(&view, keys)
                    .map(|found| ViewKernelValue::Owned(Val::Bool(found))),
                (
                    Some(BuiltinViewObjectProjection::HasKey),
                    crate::builtins::BuiltinArgs::Str(key),
                ) => Some(ViewKernelValue::Owned(Val::Bool(
                    view.has_key(key.as_ref()).unwrap_or(false),
                ))),
                (
                    Some(BuiltinViewObjectProjection::Missing),
                    crate::builtins::BuiltinArgs::Str(key),
                ) => view.has_key(key.as_ref()).map(|present| {
                    let missing =
                        !present || matches!(view.field(key.as_ref()).scalar(), JsonView::Null);
                    ViewKernelValue::Owned(Val::Bool(missing))
                }),
                (
                    Some(BuiltinViewObjectProjection::GetPath),
                    crate::builtins::BuiltinArgs::Str(path),
                ) => Some(ViewKernelValue::View(walk_view_path(
                    view,
                    &crate::builtins::parse_path_segs(path.as_ref()),
                ))),
                (
                    Some(BuiltinViewObjectProjection::GetPath),
                    crate::builtins::BuiltinArgs::Path(path),
                ) => Some(ViewKernelValue::View(walk_view_path(view, path))),
                (
                    Some(BuiltinViewObjectProjection::HasPath),
                    crate::builtins::BuiltinArgs::Str(path),
                ) => {
                    let found = !matches!(
                        walk_view_path(view, &crate::builtins::parse_path_segs(path.as_ref()))
                            .scalar(),
                        JsonView::Null
                    );
                    Some(ViewKernelValue::Owned(Val::Bool(found)))
                }
                (
                    Some(BuiltinViewObjectProjection::HasPath),
                    crate::builtins::BuiltinArgs::Path(path),
                ) => {
                    let found = !matches!(walk_view_path(view, path).scalar(), JsonView::Null);
                    Some(ViewKernelValue::Owned(Val::Bool(found)))
                }
                (Some(BuiltinViewObjectProjection::Keys), crate::builtins::BuiltinArgs::None) => {
                    view.object_keys().map(ViewKernelValue::Owned)
                }
                (Some(BuiltinViewObjectProjection::Values), crate::builtins::BuiltinArgs::None) => {
                    view.object_values().map(ViewKernelValue::Owned)
                }
                (Some(BuiltinViewObjectProjection::Entries), crate::builtins::BuiltinArgs::None) => {
                    view.object_entries().map(ViewKernelValue::Owned)
                }
                (
                    Some(BuiltinViewObjectProjection::Pick),
                    crate::builtins::BuiltinArgs::StrVec(keys),
                ) => view.pick_keys(keys).map(ViewKernelValue::Owned),
                (
                    Some(BuiltinViewObjectProjection::Omit),
                    crate::builtins::BuiltinArgs::StrVec(keys),
                ) => view.omit_keys(keys).map(ViewKernelValue::Owned),
                _ => call
                    .try_apply_json_view(view.scalar())
                    .map(ViewKernelValue::Owned),
            },
            ViewKernelValue::Owned(value) => call
                .try_apply(&value)
                .ok()
                .flatten()
                .map(ViewKernelValue::Owned),
        },
        BodyKernel::Compose { first, then } => match eval_view_kernel_inner(first, item, vm.as_deref_mut())? {
            ViewKernelValue::View(view) => eval_view_kernel_inner(then, &view, vm),
            ViewKernelValue::Owned(value) => match vm {
                Some(vm) => eval_native_kernel_with_vm(then, &value, vm)
                    .ok()
                    .map(ViewKernelValue::Owned),
                None => eval_native_kernel(then, &value)
                    .ok()
                    .map(ViewKernelValue::Owned),
            },
        },
        BodyKernel::CmpLit { lhs, op, lit } => {
            let passes = match eval_view_kernel_inner(lhs, item, vm.as_deref_mut())? {
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
            if let Some(value) = eval_view_numeric_kernel(kernel, item) {
                return Some(ViewKernelValue::Owned(numeric_kernel_value_to_val(value)));
            }
            let lhs = view_kernel_value_to_owned(eval_view_kernel_inner(lhs, item, vm.as_deref_mut())?);
            let rhs = view_kernel_value_to_owned(eval_view_kernel_inner(rhs, item, vm.as_deref_mut())?);
            eval_binary_op(lhs, *op, rhs)
                .ok()
                .map(ViewKernelValue::Owned)
        }
        BodyKernel::ArraySelect { array, selector } => match eval_view_kernel_inner(array, item, vm.as_deref_mut())? {
            ViewKernelValue::View(view) => Some(ViewKernelValue::View(eval_array_select_view(
                view, *selector,
            ))),
            ViewKernelValue::Owned(value) => Some(ViewKernelValue::Owned(
                eval_array_select_native(&value, *selector),
            )),
        },
        BodyKernel::Match {
            scrutinee,
            compiled,
            body_needs_current,
        } => match eval_view_kernel_inner(scrutinee, item, vm.as_deref_mut())? {
            ViewKernelValue::View(view) => {
                let current = if *body_needs_current {
                    view.materialize()
                } else {
                    Val::Null
                };
                let env = crate::data::context::Env::new(current);
                match vm {
                    Some(vm) => crate::vm::exec_match_view(vm, compiled, view, &env),
                    None => {
                        let mut local_vm = crate::vm::VM::new();
                        crate::vm::exec_match_view(&mut local_vm, compiled, view, &env)
                    }
                }
                .ok()
                .map(ViewKernelValue::Owned)
            }
            ViewKernelValue::Owned(value) => {
                let env = crate::data::context::Env::new(value.clone());
                match vm {
                    Some(vm) => vm.exec_match(compiled, &value, &env),
                    None => {
                        let mut local_vm = crate::vm::VM::new();
                        local_vm.exec_match(compiled, &value, &env)
                    }
                }
                .ok()
                .map(ViewKernelValue::Owned)
            }
        },
        BodyKernel::And(predicates) => {
            for predicate in predicates.iter() {
                let passes = match eval_view_kernel_inner(predicate, item, vm.as_deref_mut())? {
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
                let passes = match eval_view_kernel_inner(predicate, item, vm.as_deref_mut())? {
                    ViewKernelValue::View(view) => view.scalar().truthy(),
                    ViewKernelValue::Owned(value) => crate::util::is_truthy(&value),
                };
                if passes {
                    return Some(ViewKernelValue::Owned(Val::Bool(true)));
                }
            }
            Some(ViewKernelValue::Owned(Val::Bool(false)))
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
            let lhs = walk_view_fields(item.clone(), keys.as_ref());
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

fn view_has<'a, V>(view: &V, key: &str) -> Option<bool>
where
    V: ValueView<'a>,
{
    if let Some(found) = view.has_key(key) {
        return Some(found);
    }
    if let JsonView::Str(value) = view.scalar() {
        return Some(value.contains(key));
    }
    if let Some(mut iter) = view.array_iter() {
        return Some(iter.any(|item| scalar_matches_key(item.scalar(), key)));
    }
    None
}

fn view_has_all<'a, V>(view: &V, keys: &[Arc<str>]) -> Option<bool>
where
    V: ValueView<'a>,
{
    for key in keys {
        if !view_has(view, key.as_ref())? {
            return Some(false);
        }
    }
    Some(true)
}

#[inline]
fn scalar_matches_key(value: JsonView<'_>, key: &str) -> bool {
    match value {
        JsonView::Str(value) => value == key,
        JsonView::Int(value) => value.to_string() == key,
        JsonView::UInt(value) => value.to_string() == key,
        JsonView::Float(value) => value.to_string() == key,
        JsonView::Bool(true) => key == "true",
        JsonView::Bool(false) => key == "false",
        JsonView::Null => key == "null",
        JsonView::ArrayLen(_) | JsonView::ObjectLen(_) => false,
    }
}

#[inline]
fn walk_view_fields<'a, V>(mut cur: V, keys: &[Arc<str>]) -> V
where
    V: ValueView<'a>,
{
    for key in keys {
        cur = cur.field(key.as_ref());
    }
    cur
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
    use crate::data::view::{ValView, ValueView};
    use crate::parse::parser::parse;

    use super::{eval_view_kernel, BodyKernel, ViewKernelValue};

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

    fn owned_value(value: Option<ViewKernelValue<ValView<'_>>>) -> Option<Val> {
        match value? {
            ViewKernelValue::Owned(value) => Some(value),
            ViewKernelValue::View(view) => Some(view.materialize()),
        }
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

    #[test]
    fn object_key_calls_on_current_have_field_demand() {
        assert_eq!(field_paths(&key_call(BuiltinMethod::HasKey, "isbn")), vec!["isbn"]);
        assert_eq!(
            field_paths(&key_call(BuiltinMethod::Missing, "title")),
            vec!["title"]
        );
        assert_eq!(
            field_paths(&key_vec_call(BuiltinMethod::Missing, &["title", "isbn"])),
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
        assert_eq!(field_paths(&key_call(BuiltinMethod::Has, "isbn")), vec!["*"]);
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

        for method in [BuiltinMethod::Values, BuiltinMethod::Entries] {
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
}
