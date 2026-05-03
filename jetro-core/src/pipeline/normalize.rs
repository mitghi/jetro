//! Symbolic expression normalization for pipeline lowering.
//!
//! Classifies `Expr` sub-trees into `BodyKernel`, `Stage`, and `Sink`
//! representations the pipeline can evaluate without re-entering the VM.
//! Expressions that cannot be classified return `None`; the lowering layer
//! then either wraps them in a VM-backed `Generic` kernel or aborts lowering.

use std::sync::Arc;

use crate::ast::{Arg, ArrayElem, Expr, FStringPart, ObjField, PatchOp, PathStep, PipeStep, Step};

use super::{BodyKernel, ReducerOp, Sink, Stage};

/// Describes how much of each element's value the sink actually requires.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ValueDemand {
    /// The sink does not inspect element values at all (e.g. pure count).
    None,
    /// The sink needs the complete materialised value.
    Whole,
    /// The sink only needs a numeric projection of each element (e.g. sum, min).
    Numeric,
}

/// Combined runtime demand of a sink: value-level and order-level.
#[derive(Debug, Clone, Copy)]
struct RuntimeDemand {
    /// How much of each element value the sink consumes.
    value: ValueDemand,
    /// `true` when the sink is sensitive to element order (Collect, Terminal).
    order: bool,
}

/// Converts a `Sink` into the `RuntimeDemand` that describes what its input elements must provide.
fn sink_runtime_demand(sink: &Sink) -> RuntimeDemand {
    match sink {
        Sink::Reducer(spec) if spec.op == ReducerOp::Count => RuntimeDemand {
            value: ValueDemand::None,
            order: false,
        },
        Sink::ApproxCountDistinct => RuntimeDemand {
            value: ValueDemand::Whole,
            order: false,
        },
        Sink::Reducer(spec) if matches!(spec.op, ReducerOp::Numeric(_)) => RuntimeDemand {
            value: ValueDemand::Numeric,
            order: false,
        },
        Sink::Reducer(_) => RuntimeDemand {
            value: ValueDemand::Whole,
            order: false,
        },
        Sink::Collect | Sink::Terminal(_) => RuntimeDemand {
            value: ValueDemand::Whole,
            order: true,
        },
    }
}

/// Demand-driven symbolic optimiser: tracks the current-element expression symbolically
/// through map stages, substitutes `@` into downstream predicates, and drops stages whose
/// output is unused by the sink (e.g. order-only stages before a numeric reducer).
pub(super) fn normalize_symbolic(
    stages: &mut Vec<Stage>,
    exprs: &mut Vec<Option<Arc<Expr>>>,
    kernels: &mut Vec<BodyKernel>,
    sink: &mut Sink,
) {
    let demand = sink_runtime_demand(sink);
    let in_stages = std::mem::take(stages);
    let in_exprs = std::mem::take(exprs);
    std::mem::take(kernels);

    let mut out = SymbolicEmitter::new(demand);
    for idx in 0..in_stages.len() {
        let stage = in_stages[idx].clone();
        let expr = in_exprs.get(idx).cloned().unwrap_or(None);
        match stage {
            _ if stage.is_symbolic_map_stage() => {
                if let Some(expr) = expr.as_ref().filter(|e| is_pure_expr(e)) {
                    out.item = simplify_expr(substitute_current(expr, &out.item));
                } else {
                    out.flush_all();
                    out.push_stage(stage, expr);
                }
            }
            _ if stage.is_symbolic_filter_stage() => {
                if let Some(expr) = expr.as_ref().filter(|e| is_pure_expr(e)) {
                    let pred = simplify_expr(substitute_current(expr, &out.item));
                    out.predicate = Some(match out.predicate.take() {
                        Some(prev) => and_expr(prev, pred),
                        None => pred,
                    });
                } else {
                    out.flush_all();
                    out.push_stage(stage, expr);
                }
            }
            _ if stage.is_positional_stage() => {
                out.flush_predicate();
                out.push_stage(stage, expr);
            }
            _ if stage.is_order_only_stage() => {
                if out.demand.order || suffix_needs_order(&in_stages[idx + 1..]) {
                    out.flush_all();
                    out.push_stage(stage, expr);
                }
            }
            _ if stage.can_drop_when_value_unused()
                && out.demand.value == ValueDemand::None
                && !out.demand.order
                && !suffix_consumes_value(&in_stages[idx + 1..]) =>
            {
                
                
            }
            other => {
                out.flush_all();
                out.push_stage(other, expr);
            }
        }
    }
    out.finish(sink);
    *stages = out.out_stages;
    *exprs = out.out_exprs;
    *kernels = out.out_kernels;
}

/// Accumulates the output of symbolic normalisation: tracks a pending item expression,
/// a pending predicate expression, and the partially-built output stage list.
struct SymbolicEmitter {
    /// The symbolic expression representing the current element after all pending maps.
    item: Expr,
    /// An accumulated predicate to be emitted as a filter when a non-symbolic stage is encountered.
    predicate: Option<Expr>,
    /// The sink demand that drives which stages can be dropped or deferred.
    demand: RuntimeDemand,
    /// Stages emitted so far.
    out_stages: Vec<Stage>,
    /// Stage expressions emitted so far, parallel to `out_stages`.
    out_exprs: Vec<Option<Arc<Expr>>>,
    /// Classified kernels emitted so far, parallel to `out_stages`.
    out_kernels: Vec<BodyKernel>,
}

impl SymbolicEmitter {
    /// Creates a new emitter with `@` as the initial item expression and no pending predicate.
    fn new(demand: RuntimeDemand) -> Self {
        Self {
            item: Expr::Current,
            predicate: None,
            demand,
            out_stages: Vec::new(),
            out_exprs: Vec::new(),
            out_kernels: Vec::new(),
        }
    }

    /// Appends a pre-classified stage and its optional expression directly to the output without
    /// symbolic processing.
    fn push_stage(&mut self, stage: Stage, expr: Option<Arc<Expr>>) {
        let kernel = stage_kernel(&stage);
        self.out_stages.push(stage);
        self.out_exprs.push(expr);
        self.out_kernels.push(kernel);
    }

    /// Simplifies `expr`, compiles it into a program, classifies it as a kernel, and appends the
    /// resulting `Filter` or `Map` stage to the output.
    fn push_expr_stage(&mut self, expr: Expr, kind: ExprStageKind) {
        let expr = simplify_expr(expr);
        let prog = compile_stage_expr(&expr);
        let kernel = BodyKernel::classify(&prog);
        let stage = match kind {
            ExprStageKind::Filter => Stage::Filter(prog, crate::builtins::BuiltinViewStage::Filter),
            ExprStageKind::Map => Stage::Map(prog, crate::builtins::BuiltinViewStage::Map),
        };
        self.out_stages.push(stage);
        self.out_exprs.push(Some(Arc::new(expr)));
        self.out_kernels.push(kernel);
    }

    /// Emits the pending predicate as a `Filter` stage (unless it is trivially `true`), then
    /// clears the pending predicate field.
    fn flush_predicate(&mut self) {
        if let Some(pred) = self.predicate.take() {
            if !matches!(pred, Expr::Bool(true)) {
                self.push_expr_stage(pred, ExprStageKind::Filter);
            }
        }
    }

    /// Emits the pending item expression as a `Map` stage (unless it is identity `@`), then
    /// resets the item to `Expr::Current`.
    fn flush_item(&mut self) {
        if !matches!(self.item, Expr::Current) {
            let item = std::mem::replace(&mut self.item, Expr::Current);
            self.push_expr_stage(item, ExprStageKind::Map);
        }
    }

    /// Flushes both the pending predicate and the pending item expression in predicate-first order.
    fn flush_all(&mut self) {
        self.flush_predicate();
        self.flush_item();
    }

    /// Finalises the emitter by flushing the predicate and, when possible, folding the pending
    /// item expression into the sink's projection rather than emitting a standalone `Map` stage.
    fn finish(&mut self, sink: &mut Sink) {
        self.flush_predicate();
        match sink {
            Sink::Reducer(spec) if spec.op == ReducerOp::Count => {}
            Sink::Reducer(spec)
                if matches!(spec.op, ReducerOp::Numeric(_))
                    && spec.projection.is_none()
                    && !matches!(self.item, Expr::Current) =>
            {
                let item = simplify_expr(std::mem::replace(&mut self.item, Expr::Current));
                spec.projection = Some(compile_stage_expr(&item));
            }
            _ if self.demand.value == ValueDemand::Whole && !matches!(self.item, Expr::Current) => {
                self.flush_item();
            }
            _ => {}
        }
    }
}

/// Specifies whether a symbolic expression should be emitted as a `Filter` or a `Map` stage.
#[derive(Debug, Clone, Copy)]
enum ExprStageKind {
    /// Emit a `Stage::Filter` that keeps elements for which the expression is truthy.
    Filter,
    /// Emit a `Stage::Map` that replaces each element with the expression's result.
    Map,
}

/// Classifies the body program of `stage` into a `BodyKernel`, returning `Generic` when
/// the stage carries no body.
fn stage_kernel(stage: &Stage) -> BodyKernel {
    stage
        .body_program()
        .map(BodyKernel::classify)
        .unwrap_or(BodyKernel::Generic)
}

/// Constructs `lhs && rhs`, applying short-circuit constant folding when either operand is
/// a `Bool` literal.
fn and_expr(lhs: Expr, rhs: Expr) -> Expr {
    match (lhs, rhs) {
        (Expr::Bool(true), r) => r,
        (l, Expr::Bool(true)) => l,
        (l, r) => Expr::BinOp(Box::new(l), crate::ast::BinOp::And, Box::new(r)),
    }
}

/// Returns `true` when any stage in `stages` is positional, meaning upstream order must be
/// preserved.
fn suffix_needs_order(stages: &[Stage]) -> bool {
    stages.iter().any(Stage::is_positional_stage)
}

/// Returns `true` when any stage in `stages` reads element values, preventing the demand
/// optimiser from dropping a preceding value-producing stage.
fn suffix_consumes_value(stages: &[Stage]) -> bool {
    stages.iter().any(Stage::consumes_input_value)
}

/// Compiles `expr` into a VM `Program` tagged with the `<pipeline-rewrite>` source label.
fn compile_stage_expr(expr: &Expr) -> Arc<crate::vm::Program> {
    Arc::new(crate::vm::Compiler::compile(expr, "<pipeline-rewrite>"))
}

/// Recursively simplifies `expr` by constant-folding boolean operators, evaluating static
/// object/array projections, and propagating `null` through coalescing.
fn simplify_expr(expr: Expr) -> Expr {
    match expr {
        Expr::Chain(base, steps) => simplify_chain(simplify_expr(*base), steps),
        Expr::FString(parts) => Expr::FString(
            parts
                .into_iter()
                .map(|part| match part {
                    FStringPart::Lit(s) => FStringPart::Lit(s),
                    FStringPart::Interp { expr, fmt } => FStringPart::Interp {
                        expr: simplify_expr(expr),
                        fmt,
                    },
                })
                .collect(),
        ),
        Expr::BinOp(lhs, op, rhs) => simplify_binop(simplify_expr(*lhs), op, simplify_expr(*rhs)),
        Expr::UnaryNeg(inner) => Expr::UnaryNeg(Box::new(simplify_expr(*inner))),
        Expr::Not(inner) => match simplify_expr(*inner) {
            Expr::Bool(b) => Expr::Bool(!b),
            e => Expr::Not(Box::new(e)),
        },
        Expr::Kind { expr, ty, negate } => Expr::Kind {
            expr: Box::new(simplify_expr(*expr)),
            ty,
            negate,
        },
        Expr::Coalesce(lhs, rhs) => match simplify_expr(*lhs) {
            Expr::Null => simplify_expr(*rhs),
            lhs => Expr::Coalesce(Box::new(lhs), Box::new(simplify_expr(*rhs))),
        },
        Expr::Object(fields) => Expr::Object(
            fields
                .into_iter()
                .map(|field| match field {
                    ObjField::Kv {
                        key,
                        val,
                        optional,
                        cond,
                    } => ObjField::Kv {
                        key,
                        val: simplify_expr(val),
                        optional,
                        cond: cond.map(simplify_expr),
                    },
                    ObjField::Dynamic { key, val } => ObjField::Dynamic {
                        key: simplify_expr(key),
                        val: simplify_expr(val),
                    },
                    ObjField::Spread(e) => ObjField::Spread(simplify_expr(e)),
                    ObjField::SpreadDeep(e) => ObjField::SpreadDeep(simplify_expr(e)),
                    ObjField::Short(s) => ObjField::Short(s),
                })
                .collect(),
        ),
        Expr::Array(elems) => Expr::Array(
            elems
                .into_iter()
                .map(|elem| match elem {
                    ArrayElem::Expr(e) => ArrayElem::Expr(simplify_expr(e)),
                    ArrayElem::Spread(e) => ArrayElem::Spread(simplify_expr(e)),
                })
                .collect(),
        ),
        Expr::Pipeline { base, steps } => Expr::Pipeline {
            base: Box::new(simplify_expr(*base)),
            steps: steps
                .into_iter()
                .map(|step| match step {
                    PipeStep::Forward(e) => PipeStep::Forward(simplify_expr(e)),
                    PipeStep::Bind(b) => PipeStep::Bind(b),
                })
                .collect(),
        },
        Expr::ListComp {
            expr,
            vars,
            iter,
            cond,
        } => Expr::ListComp {
            expr: Box::new(simplify_expr(*expr)),
            vars,
            iter: Box::new(simplify_expr(*iter)),
            cond: cond.map(|c| Box::new(simplify_expr(*c))),
        },
        Expr::DictComp {
            key,
            val,
            vars,
            iter,
            cond,
        } => Expr::DictComp {
            key: Box::new(simplify_expr(*key)),
            val: Box::new(simplify_expr(*val)),
            vars,
            iter: Box::new(simplify_expr(*iter)),
            cond: cond.map(|c| Box::new(simplify_expr(*c))),
        },
        Expr::SetComp {
            expr,
            vars,
            iter,
            cond,
        } => Expr::SetComp {
            expr: Box::new(simplify_expr(*expr)),
            vars,
            iter: Box::new(simplify_expr(*iter)),
            cond: cond.map(|c| Box::new(simplify_expr(*c))),
        },
        Expr::GenComp {
            expr,
            vars,
            iter,
            cond,
        } => Expr::GenComp {
            expr: Box::new(simplify_expr(*expr)),
            vars,
            iter: Box::new(simplify_expr(*iter)),
            cond: cond.map(|c| Box::new(simplify_expr(*c))),
        },
        Expr::Lambda { params, body } => Expr::Lambda {
            params,
            body: Box::new(simplify_expr(*body)),
        },
        Expr::Let { name, init, body } => Expr::Let {
            name,
            init: Box::new(simplify_expr(*init)),
            body: Box::new(simplify_expr(*body)),
        },
        Expr::IfElse { cond, then_, else_ } => match simplify_expr(*cond) {
            Expr::Bool(true) => simplify_expr(*then_),
            Expr::Bool(false) => simplify_expr(*else_),
            cond => Expr::IfElse {
                cond: Box::new(cond),
                then_: Box::new(simplify_expr(*then_)),
                else_: Box::new(simplify_expr(*else_)),
            },
        },
        Expr::Try { body, default } => Expr::Try {
            body: Box::new(simplify_expr(*body)),
            default: Box::new(simplify_expr(*default)),
        },
        Expr::GlobalCall { name, args } => Expr::GlobalCall {
            name,
            args: args
                .into_iter()
                .map(|arg| match arg {
                    Arg::Pos(e) => Arg::Pos(simplify_expr(e)),
                    Arg::Named(name, e) => Arg::Named(name, simplify_expr(e)),
                })
                .collect(),
        },
        Expr::Cast { expr, ty } => Expr::Cast {
            expr: Box::new(simplify_expr(*expr)),
            ty,
        },
        Expr::Patch { root, ops } => Expr::Patch {
            root: Box::new(simplify_expr(*root)),
            ops: ops
                .into_iter()
                .map(|op| PatchOp {
                    path: op
                        .path
                        .into_iter()
                        .map(|step| match step {
                            PathStep::DynIndex(e) => PathStep::DynIndex(simplify_expr(e)),
                            PathStep::WildcardFilter(e) => {
                                PathStep::WildcardFilter(Box::new(simplify_expr(*e)))
                            }
                            step => step,
                        })
                        .collect(),
                    val: simplify_expr(op.val),
                    cond: op.cond.map(simplify_expr),
                })
                .collect(),
        },
        e => e,
    }
}

/// Simplifies a `Chain` expression by statically evaluating field/index steps against literal
/// object/array bases, folding away steps that can be resolved at compile time.
fn simplify_chain(mut base: Expr, steps: Vec<Step>) -> Expr {
    let mut remaining: Vec<Step> = Vec::new();
    for step in steps {
        if remaining.is_empty() {
            if let Some(next) = project_static_step(&base, &step) {
                base = simplify_expr(next);
                continue;
            }
        }
        remaining.push(simplify_step(step));
    }
    if remaining.is_empty() {
        base
    } else {
        Expr::Chain(Box::new(base), remaining)
    }
}

/// Attempts to evaluate `step` applied to a literal `base` at simplification time, returning
/// `None` when dynamic dispatch is required.
fn project_static_step(base: &Expr, step: &Step) -> Option<Expr> {
    match (base, step) {
        (Expr::Object(fields), Step::Field(key)) => project_object_field(fields, key),
        (Expr::Array(elems), Step::Index(idx)) => project_array_index(elems, *idx),
        (Expr::Array(elems), Step::DynIndex(idx)) => match idx.as_ref() {
            Expr::Int(i) => project_array_index(elems, *i),
            _ => None,
        },
        (Expr::Object(fields), Step::DynIndex(idx)) => match idx.as_ref() {
            Expr::Str(k) => project_object_field(fields, k),
            _ => None,
        },
        _ => None,
    }
}

/// Looks up `key` in a literal object field list, returning the value expression or `None` if
/// the object contains dynamic/optional/conditional entries that prevent static resolution.
fn project_object_field(fields: &[ObjField], key: &str) -> Option<Expr> {
    let mut found = None;
    for field in fields {
        match field {
            ObjField::Kv {
                key: k,
                val,
                optional: false,
                cond: None,
            } if k == key => found = Some(val.clone()),
            ObjField::Short(k) if k == key => found = Some(Expr::Ident(k.clone())),
            ObjField::Dynamic { .. }
            | ObjField::Spread(_)
            | ObjField::SpreadDeep(_)
            | ObjField::Kv { optional: true, .. }
            | ObjField::Kv { cond: Some(_), .. } => return None,
            _ => {}
        }
    }
    found
}

/// Returns the element at `idx` from a literal array (supporting negative indices), or `None`
/// when the array contains spread elements or the index is out of range.
fn project_array_index(elems: &[ArrayElem], idx: i64) -> Option<Expr> {
    if elems
        .iter()
        .any(|elem| matches!(elem, ArrayElem::Spread(_)))
    {
        return None;
    }
    let len = elems.len() as i64;
    let idx = if idx < 0 { len + idx } else { idx };
    if !(0..len).contains(&idx) {
        return None;
    }
    match &elems[idx as usize] {
        ArrayElem::Expr(e) => Some(e.clone()),
        ArrayElem::Spread(_) => None,
    }
}

/// Recursively simplifies the expression arguments within a `Step` variant.
fn simplify_step(step: Step) -> Step {
    match step {
        Step::DynIndex(e) => Step::DynIndex(Box::new(simplify_expr(*e))),
        Step::Method(name, args) => Step::Method(name, simplify_args(args)),
        Step::OptMethod(name, args) => Step::OptMethod(name, simplify_args(args)),
        Step::InlineFilter(e) => Step::InlineFilter(Box::new(simplify_expr(*e))),
        step => step,
    }
}

/// Simplifies all expressions inside a list of positional or named arguments.
fn simplify_args(args: Vec<Arg>) -> Vec<Arg> {
    args.into_iter()
        .map(|arg| match arg {
            Arg::Pos(e) => Arg::Pos(simplify_expr(e)),
            Arg::Named(name, e) => Arg::Named(name, simplify_expr(e)),
        })
        .collect()
}

/// Applies constant-folding rules to a binary expression: short-circuits `&&` / `||` when
/// either operand is a boolean literal.
fn simplify_binop(lhs: Expr, op: crate::ast::BinOp, rhs: Expr) -> Expr {
    use crate::ast::BinOp;
    match (lhs, op, rhs) {
        (Expr::Bool(true), BinOp::And, rhs) | (rhs, BinOp::And, Expr::Bool(true)) => rhs,
        (Expr::Bool(false), BinOp::And, _) | (_, BinOp::And, Expr::Bool(false)) => {
            Expr::Bool(false)
        }
        (Expr::Bool(false), BinOp::Or, rhs) | (rhs, BinOp::Or, Expr::Bool(false)) => rhs,
        (Expr::Bool(true), BinOp::Or, _) | (_, BinOp::Or, Expr::Bool(true)) => Expr::Bool(true),
        (lhs, op, rhs) => Expr::BinOp(Box::new(lhs), op, Box::new(rhs)),
    }
}

/// Performs capture-avoiding substitution of `Expr::Current` (`@`) with `replacement` throughout
/// `expr`; lambdas are not descended into since they rebind `@`.
fn substitute_current(expr: &Expr, replacement: &Expr) -> Expr {
    match expr {
        Expr::Current => replacement.clone(),
        Expr::Null
        | Expr::Bool(_)
        | Expr::Int(_)
        | Expr::Float(_)
        | Expr::Str(_)
        | Expr::Root
        | Expr::Ident(_)
        | Expr::DeleteMark => expr.clone(),
        Expr::FString(parts) => Expr::FString(
            parts
                .iter()
                .map(|part| match part {
                    FStringPart::Lit(s) => FStringPart::Lit(s.clone()),
                    FStringPart::Interp { expr, fmt } => FStringPart::Interp {
                        expr: substitute_current(expr, replacement),
                        fmt: fmt.clone(),
                    },
                })
                .collect(),
        ),
        Expr::Chain(base, steps) => Expr::Chain(
            Box::new(substitute_current(base, replacement)),
            steps
                .iter()
                .map(|step| substitute_current_step(step, replacement))
                .collect(),
        ),
        Expr::BinOp(lhs, op, rhs) => Expr::BinOp(
            Box::new(substitute_current(lhs, replacement)),
            *op,
            Box::new(substitute_current(rhs, replacement)),
        ),
        Expr::UnaryNeg(inner) => Expr::UnaryNeg(Box::new(substitute_current(inner, replacement))),
        Expr::Not(inner) => Expr::Not(Box::new(substitute_current(inner, replacement))),
        Expr::Kind { expr, ty, negate } => Expr::Kind {
            expr: Box::new(substitute_current(expr, replacement)),
            ty: *ty,
            negate: *negate,
        },
        Expr::Coalesce(lhs, rhs) => Expr::Coalesce(
            Box::new(substitute_current(lhs, replacement)),
            Box::new(substitute_current(rhs, replacement)),
        ),
        Expr::Object(fields) => Expr::Object(
            fields
                .iter()
                .map(|field| substitute_current_obj_field(field, replacement))
                .collect(),
        ),
        Expr::Array(elems) => Expr::Array(
            elems
                .iter()
                .map(|elem| match elem {
                    ArrayElem::Expr(e) => ArrayElem::Expr(substitute_current(e, replacement)),
                    ArrayElem::Spread(e) => ArrayElem::Spread(substitute_current(e, replacement)),
                })
                .collect(),
        ),
        Expr::Pipeline { base, steps } => Expr::Pipeline {
            base: Box::new(substitute_current(base, replacement)),
            steps: steps
                .iter()
                .map(|step| match step {
                    PipeStep::Forward(e) => PipeStep::Forward(substitute_current(e, replacement)),
                    PipeStep::Bind(b) => PipeStep::Bind(b.clone()),
                })
                .collect(),
        },
        Expr::ListComp {
            expr,
            vars,
            iter,
            cond,
        } => Expr::ListComp {
            expr: Box::new(substitute_current(expr, replacement)),
            vars: vars.clone(),
            iter: Box::new(substitute_current(iter, replacement)),
            cond: cond
                .as_ref()
                .map(|c| Box::new(substitute_current(c, replacement))),
        },
        Expr::DictComp {
            key,
            val,
            vars,
            iter,
            cond,
        } => Expr::DictComp {
            key: Box::new(substitute_current(key, replacement)),
            val: Box::new(substitute_current(val, replacement)),
            vars: vars.clone(),
            iter: Box::new(substitute_current(iter, replacement)),
            cond: cond
                .as_ref()
                .map(|c| Box::new(substitute_current(c, replacement))),
        },
        Expr::SetComp {
            expr,
            vars,
            iter,
            cond,
        } => Expr::SetComp {
            expr: Box::new(substitute_current(expr, replacement)),
            vars: vars.clone(),
            iter: Box::new(substitute_current(iter, replacement)),
            cond: cond
                .as_ref()
                .map(|c| Box::new(substitute_current(c, replacement))),
        },
        Expr::GenComp {
            expr,
            vars,
            iter,
            cond,
        } => Expr::GenComp {
            expr: Box::new(substitute_current(expr, replacement)),
            vars: vars.clone(),
            iter: Box::new(substitute_current(iter, replacement)),
            cond: cond
                .as_ref()
                .map(|c| Box::new(substitute_current(c, replacement))),
        },
        Expr::Lambda { .. } => expr.clone(),
        Expr::Let { name, init, body } => Expr::Let {
            name: name.clone(),
            init: Box::new(substitute_current(init, replacement)),
            body: Box::new(substitute_current(body, replacement)),
        },
        Expr::IfElse { cond, then_, else_ } => Expr::IfElse {
            cond: Box::new(substitute_current(cond, replacement)),
            then_: Box::new(substitute_current(then_, replacement)),
            else_: Box::new(substitute_current(else_, replacement)),
        },
        Expr::Try { body, default } => Expr::Try {
            body: Box::new(substitute_current(body, replacement)),
            default: Box::new(substitute_current(default, replacement)),
        },
        Expr::GlobalCall { name, args } => Expr::GlobalCall {
            name: name.clone(),
            args: args
                .iter()
                .map(|arg| substitute_current_arg(arg, replacement))
                .collect(),
        },
        Expr::Cast { expr, ty } => Expr::Cast {
            expr: Box::new(substitute_current(expr, replacement)),
            ty: *ty,
        },
        Expr::Patch { root, ops } => Expr::Patch {
            root: Box::new(substitute_current(root, replacement)),
            ops: ops
                .iter()
                .map(|op| PatchOp {
                    path: op
                        .path
                        .iter()
                        .map(|step| substitute_current_path_step(step, replacement))
                        .collect(),
                    val: substitute_current(&op.val, replacement),
                    cond: op.cond.as_ref().map(|c| substitute_current(c, replacement)),
                })
                .collect(),
        },
    }
}

/// Substitutes `@` with `replacement` inside the expressions carried by a `Step`.
fn substitute_current_step(step: &Step, replacement: &Expr) -> Step {
    match step {
        Step::DynIndex(e) => Step::DynIndex(Box::new(substitute_current(e, replacement))),
        Step::Method(name, args) => Step::Method(
            name.clone(),
            args.iter()
                .map(|arg| substitute_current_arg(arg, replacement))
                .collect(),
        ),
        Step::OptMethod(name, args) => Step::OptMethod(
            name.clone(),
            args.iter()
                .map(|arg| substitute_current_arg(arg, replacement))
                .collect(),
        ),
        Step::InlineFilter(e) => Step::InlineFilter(Box::new(substitute_current(e, replacement))),
        _ => step.clone(),
    }
}

/// Substitutes `@` with `replacement` inside a positional or named argument expression.
fn substitute_current_arg(arg: &Arg, replacement: &Expr) -> Arg {
    match arg {
        Arg::Pos(e) => Arg::Pos(substitute_current(e, replacement)),
        Arg::Named(name, e) => Arg::Named(name.clone(), substitute_current(e, replacement)),
    }
}

/// Substitutes `@` with `replacement` in all expression slots of an object field.
fn substitute_current_obj_field(field: &ObjField, replacement: &Expr) -> ObjField {
    match field {
        ObjField::Kv {
            key,
            val,
            optional,
            cond,
        } => ObjField::Kv {
            key: key.clone(),
            val: substitute_current(val, replacement),
            optional: *optional,
            cond: cond.as_ref().map(|c| substitute_current(c, replacement)),
        },
        ObjField::Short(s) => ObjField::Short(s.clone()),
        ObjField::Dynamic { key, val } => ObjField::Dynamic {
            key: substitute_current(key, replacement),
            val: substitute_current(val, replacement),
        },
        ObjField::Spread(e) => ObjField::Spread(substitute_current(e, replacement)),
        ObjField::SpreadDeep(e) => ObjField::SpreadDeep(substitute_current(e, replacement)),
    }
}

/// Substitutes `@` with `replacement` in the dynamic-index and wildcard-filter slots of a
/// `PathStep`, leaving static field and index steps unchanged.
fn substitute_current_path_step(step: &PathStep, replacement: &Expr) -> PathStep {
    match step {
        PathStep::DynIndex(e) => PathStep::DynIndex(substitute_current(e, replacement)),
        PathStep::WildcardFilter(e) => {
            PathStep::WildcardFilter(Box::new(substitute_current(e, replacement)))
        }
        _ => step.clone(),
    }
}

/// Returns `true` when `expr` is side-effect-free and can safely be substituted or reordered
/// by the demand optimiser; `Patch`, `DeleteMark`, and `GlobalCall` are considered impure.
fn is_pure_expr(expr: &Expr) -> bool {
    match expr {
        Expr::Patch { .. } | Expr::DeleteMark => false,
        Expr::Lambda { body, .. } => is_pure_expr(body),
        Expr::GlobalCall { .. } => false,
        Expr::Chain(base, steps) => {
            is_pure_expr(base)
                && steps.iter().all(|step| match step {
                    Step::DynIndex(e) | Step::InlineFilter(e) => is_pure_expr(e),
                    Step::Method(_, args) | Step::OptMethod(_, args) => {
                        args.iter().all(is_pure_arg)
                    }
                    _ => true,
                })
        }
        Expr::FString(parts) => parts.iter().all(|part| match part {
            FStringPart::Lit(_) => true,
            FStringPart::Interp { expr, .. } => is_pure_expr(expr),
        }),
        Expr::BinOp(lhs, _, rhs) | Expr::Coalesce(lhs, rhs) => {
            is_pure_expr(lhs) && is_pure_expr(rhs)
        }
        Expr::UnaryNeg(e)
        | Expr::Not(e)
        | Expr::Kind { expr: e, .. }
        | Expr::Cast { expr: e, .. } => is_pure_expr(e),
        Expr::Object(fields) => fields.iter().all(|field| match field {
            ObjField::Kv { val, cond, .. } => {
                is_pure_expr(val) && cond.as_ref().map(is_pure_expr).unwrap_or(true)
            }
            ObjField::Short(_) => true,
            ObjField::Dynamic { key, val } => is_pure_expr(key) && is_pure_expr(val),
            ObjField::Spread(e) | ObjField::SpreadDeep(e) => is_pure_expr(e),
        }),
        Expr::Array(elems) => elems.iter().all(|elem| match elem {
            ArrayElem::Expr(e) | ArrayElem::Spread(e) => is_pure_expr(e),
        }),
        Expr::Pipeline { base, steps } => {
            is_pure_expr(base)
                && steps.iter().all(|step| match step {
                    PipeStep::Forward(e) => is_pure_expr(e),
                    PipeStep::Bind(_) => true,
                })
        }
        Expr::ListComp {
            expr, iter, cond, ..
        }
        | Expr::SetComp {
            expr, iter, cond, ..
        }
        | Expr::GenComp {
            expr, iter, cond, ..
        } => {
            is_pure_expr(expr)
                && is_pure_expr(iter)
                && cond.as_ref().map(|c| is_pure_expr(c)).unwrap_or(true)
        }
        Expr::DictComp {
            key,
            val,
            iter,
            cond,
            ..
        } => {
            is_pure_expr(key)
                && is_pure_expr(val)
                && is_pure_expr(iter)
                && cond.as_ref().map(|c| is_pure_expr(c)).unwrap_or(true)
        }
        Expr::Let { init, body, .. } => is_pure_expr(init) && is_pure_expr(body),
        Expr::IfElse { cond, then_, else_ } => {
            is_pure_expr(cond) && is_pure_expr(then_) && is_pure_expr(else_)
        }
        Expr::Try { body, default } => is_pure_expr(body) && is_pure_expr(default),
        Expr::Null
        | Expr::Bool(_)
        | Expr::Int(_)
        | Expr::Float(_)
        | Expr::Str(_)
        | Expr::Root
        | Expr::Current
        | Expr::Ident(_) => true,
    }
}

/// Returns `true` when the expression carried by `arg` is pure.
fn is_pure_arg(arg: &Arg) -> bool {
    match arg {
        Arg::Pos(e) | Arg::Named(_, e) => is_pure_expr(e),
    }
}
