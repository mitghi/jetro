use std::sync::Arc;

use crate::ast::{Arg, ArrayElem, Expr, FStringPart, ObjField, PatchOp, PathStep, PipeStep, Step};

use super::{BodyKernel, NumericSink, Sink, Stage};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ValueDemand {
    None,
    Whole,
    Numeric,
}

#[derive(Debug, Clone, Copy)]
struct RuntimeDemand {
    value: ValueDemand,
    order: bool,
}

fn sink_runtime_demand(sink: &Sink) -> RuntimeDemand {
    match sink {
        Sink::Count => RuntimeDemand {
            value: ValueDemand::None,
            order: false,
        },
        Sink::ApproxCountDistinct => RuntimeDemand {
            value: ValueDemand::Whole,
            order: false,
        },
        Sink::Numeric(_) => RuntimeDemand {
            value: ValueDemand::Numeric,
            order: false,
        },
        Sink::Collect | Sink::First | Sink::Last => RuntimeDemand {
            value: ValueDemand::Whole,
            order: true,
        },
    }
}

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
            Stage::Map(_) | Stage::CompiledMap(_) => {
                if let Some(expr) = expr.as_ref().filter(|e| is_pure_expr(e)) {
                    out.item = simplify_expr(substitute_current(expr, &out.item));
                } else {
                    out.flush_all();
                    out.push_stage(stage, expr);
                }
            }
            Stage::Filter(_) => {
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
            Stage::Take(_) | Stage::Skip(_) => {
                out.flush_predicate();
                out.push_stage(stage, expr);
            }
            Stage::Reverse | Stage::Sort(_) => {
                if out.demand.order || suffix_needs_order(&in_stages[idx + 1..]) {
                    out.flush_all();
                    out.push_stage(stage, expr);
                }
            }
            Stage::Builtin(call)
                if call.spec().pure
                    && out.demand.value == ValueDemand::None
                    && out.demand.order == false
                    && !suffix_consumes_value(&in_stages[idx + 1..]) =>
            {
                // Pure one-to-one value work is dead for cardinality-only sinks
                // unless a later symbolic filter consumes it.
            }
            Stage::Slice(_, _)
            | Stage::Replace { .. }
            | Stage::TransformValues(_)
            | Stage::TransformKeys(_)
            | Stage::FilterValues(_)
            | Stage::FilterKeys(_)
                if out.demand.value == ValueDemand::None
                    && out.demand.order == false
                    && !suffix_consumes_value(&in_stages[idx + 1..]) =>
            {
                // Pure one-to-one value work is dead for cardinality-only sinks
                // unless a later symbolic filter consumes it. At this point a
                // non-symbolic stage cannot be represented in `item`, so only
                // drop it when it is safe to ignore values entirely.
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

struct SymbolicEmitter {
    item: Expr,
    predicate: Option<Expr>,
    demand: RuntimeDemand,
    out_stages: Vec<Stage>,
    out_exprs: Vec<Option<Arc<Expr>>>,
    out_kernels: Vec<BodyKernel>,
}

impl SymbolicEmitter {
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

    fn push_stage(&mut self, stage: Stage, expr: Option<Arc<Expr>>) {
        let kernel = stage_kernel(&stage);
        self.out_stages.push(stage);
        self.out_exprs.push(expr);
        self.out_kernels.push(kernel);
    }

    fn push_expr_stage(&mut self, expr: Expr, kind: ExprStageKind) {
        let expr = simplify_expr(expr);
        let prog = compile_stage_expr(&expr);
        let kernel = BodyKernel::classify(&prog);
        let stage = match kind {
            ExprStageKind::Filter => Stage::Filter(prog),
            ExprStageKind::Map => Stage::Map(prog),
        };
        self.out_stages.push(stage);
        self.out_exprs.push(Some(Arc::new(expr)));
        self.out_kernels.push(kernel);
    }

    fn flush_predicate(&mut self) {
        if let Some(pred) = self.predicate.take() {
            if !matches!(pred, Expr::Bool(true)) {
                self.push_expr_stage(pred, ExprStageKind::Filter);
            }
        }
    }

    fn flush_item(&mut self) {
        if !matches!(self.item, Expr::Current) {
            let item = std::mem::replace(&mut self.item, Expr::Current);
            self.push_expr_stage(item, ExprStageKind::Map);
        }
    }

    fn flush_all(&mut self) {
        self.flush_predicate();
        self.flush_item();
    }

    fn finish(&mut self, sink: &mut Sink) {
        self.flush_predicate();
        match sink {
            Sink::Count => {}
            Sink::Numeric(n) if n.is_identity() && !matches!(self.item, Expr::Current) => {
                let item = simplify_expr(std::mem::replace(&mut self.item, Expr::Current));
                *sink = Sink::Numeric(NumericSink::projected(n.op, compile_stage_expr(&item)));
            }
            _ if self.demand.value == ValueDemand::Whole && !matches!(self.item, Expr::Current) => {
                self.flush_item();
            }
            _ => {}
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum ExprStageKind {
    Filter,
    Map,
}

fn stage_kernel(stage: &Stage) -> BodyKernel {
    match stage {
        Stage::Filter(p)
        | Stage::Map(p)
        | Stage::FlatMap(p)
        | Stage::TakeWhile(p)
        | Stage::DropWhile(p)
        | Stage::IndicesWhere(p)
        | Stage::FindIndex(p)
        | Stage::MaxBy(p)
        | Stage::MinBy(p)
        | Stage::TransformValues(p)
        | Stage::TransformKeys(p)
        | Stage::FilterValues(p)
        | Stage::FilterKeys(p)
        | Stage::CountBy(p)
        | Stage::IndexBy(p)
        | Stage::GroupBy(p)
        | Stage::Sort(Some(p))
        | Stage::UniqueBy(Some(p)) => BodyKernel::classify(p),
        _ => BodyKernel::Generic,
    }
}

fn and_expr(lhs: Expr, rhs: Expr) -> Expr {
    match (lhs, rhs) {
        (Expr::Bool(true), r) => r,
        (l, Expr::Bool(true)) => l,
        (l, r) => Expr::BinOp(Box::new(l), crate::ast::BinOp::And, Box::new(r)),
    }
}

fn suffix_needs_order(stages: &[Stage]) -> bool {
    stages
        .iter()
        .any(|s| matches!(s, Stage::Take(_) | Stage::Skip(_)))
}

fn suffix_consumes_value(stages: &[Stage]) -> bool {
    stages.iter().any(|s| {
        matches!(
            s,
            Stage::Filter(_)
                | Stage::UniqueBy(_)
                | Stage::GroupBy(_)
                | Stage::FlatMap(_)
                | Stage::Sort(Some(_))
                | Stage::TakeWhile(_)
                | Stage::DropWhile(_)
                | Stage::IndicesWhere(_)
                | Stage::FindIndex(_)
                | Stage::MaxBy(_)
                | Stage::MinBy(_)
                | Stage::CountBy(_)
                | Stage::IndexBy(_)
                | Stage::TransformValues(_)
                | Stage::TransformKeys(_)
                | Stage::FilterValues(_)
                | Stage::FilterKeys(_)
        )
    })
}

fn compile_stage_expr(expr: &Expr) -> Arc<crate::vm::Program> {
    Arc::new(crate::vm::Compiler::compile(expr, "<pipeline-rewrite>"))
}

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

fn simplify_step(step: Step) -> Step {
    match step {
        Step::DynIndex(e) => Step::DynIndex(Box::new(simplify_expr(*e))),
        Step::Method(name, args) => Step::Method(name, simplify_args(args)),
        Step::OptMethod(name, args) => Step::OptMethod(name, simplify_args(args)),
        Step::InlineFilter(e) => Step::InlineFilter(Box::new(simplify_expr(*e))),
        step => step,
    }
}

fn simplify_args(args: Vec<Arg>) -> Vec<Arg> {
    args.into_iter()
        .map(|arg| match arg {
            Arg::Pos(e) => Arg::Pos(simplify_expr(e)),
            Arg::Named(name, e) => Arg::Named(name, simplify_expr(e)),
        })
        .collect()
}

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

fn substitute_current_arg(arg: &Arg, replacement: &Expr) -> Arg {
    match arg {
        Arg::Pos(e) => Arg::Pos(substitute_current(e, replacement)),
        Arg::Named(name, e) => Arg::Named(name.clone(), substitute_current(e, replacement)),
    }
}

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

fn substitute_current_path_step(step: &PathStep, replacement: &Expr) -> PathStep {
    match step {
        PathStep::DynIndex(e) => PathStep::DynIndex(substitute_current(e, replacement)),
        PathStep::WildcardFilter(e) => {
            PathStep::WildcardFilter(Box::new(substitute_current(e, replacement)))
        }
        _ => step.clone(),
    }
}

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

fn is_pure_arg(arg: &Arg) -> bool {
    match arg {
        Arg::Pos(e) | Arg::Named(_, e) => is_pure_expr(e),
    }
}
