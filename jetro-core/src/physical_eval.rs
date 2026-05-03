//! Interpreter for `physical.rs` expression plans.

use std::sync::Arc;

use crate::ast::BinOp;
use crate::context::{Env, EvalError};
use crate::physical::{
    NodeId, PhysicalArrayElem, PhysicalChainStep, PhysicalObjField, PhysicalPathStep,
    PipelinePlanSource, PlanNode, QueryPlan,
};
use crate::pipeline;
use crate::runtime::{PipelineSourceResolver, ResolvedPipelineSource};
use crate::value::Val;
use crate::value_view::{ValView, ValueView};
use crate::view_pipeline;
use crate::{Jetro, VM};

#[cfg(feature = "simd-json")]
type TapePipelineFallback<'a> = Option<&'a crate::strref::TapeData>;

#[cfg(not(feature = "simd-json"))]
type TapePipelineFallback<'a> = Option<&'a ()>;

#[derive(Clone, Copy)]
struct ViewEvalCtx<'p, 'a> {
    plan: &'p QueryPlan,
    cache: Option<&'p dyn pipeline::PipelineData>,
    allow_materialized_pipeline_fallback: bool,
    tape_pipeline_fallback: TapePipelineFallback<'a>,
}

pub(crate) fn run(j: &Jetro, plan: &QueryPlan, root_id: NodeId) -> Result<Val, EvalError> {
    if let Some(result) = try_run_structural_derived_plan(j, plan, root_id) {
        return result;
    }

    if let Some(result) = try_run_structural_plan(j, plan, root_id)? {
        return Ok(result);
    }

    #[cfg(feature = "simd-json")]
    if let Some(tape) = j.lazy_tape()? {
        if let Some(result) = try_run_view_plan(
            plan,
            root_id,
            crate::value_view::TapeView::root(tape),
            Some(j),
            false,
            Some(tape),
        ) {
            return result;
        }
        if let Some(result) = try_run_tape_row_plan(plan, root_id, tape) {
            return result;
        }
    }

    let root = j.root_val()?;
    if let Some(result) = try_run_view_plan(plan, root_id, ValView::new(&root), Some(j), true, None)
    {
        return result;
    }

    let mut ctx = ExecCtx {
        j,
        plan,
        root,
        env: Env::new(j.root_val()?),
        vm: VM::new(),
    };
    ctx.eval(root_id)
}

fn try_run_structural_derived_plan(
    j: &Jetro,
    plan: &QueryPlan,
    root_id: NodeId,
) -> Option<Result<Val, EvalError>> {
    eval_structural_derived(j, plan, root_id)
}

fn eval_structural_derived(
    j: &Jetro,
    plan: &QueryPlan,
    id: NodeId,
) -> Option<Result<Val, EvalError>> {
    match plan.node(id) {
        PlanNode::Structural {
            plan: structural, ..
        } => {
            let bytes = match j.raw_bytes() {
                Some(bytes) => bytes,
                None => return None,
            };
            let index = match j.lazy_structural_index() {
                Ok(Some(index)) => index,
                Ok(None) => return None,
                Err(err) => return Some(Err(err)),
            };
            Some(structural.run(index, bytes))
        }
        PlanNode::Pipeline {
            source: PipelinePlanSource::Expr(source),
            body,
        } if pipeline_body_is_current_free(body) => {
            let source = match eval_structural_derived(j, plan, *source)? {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            let pipeline = body.clone().with_source(pipeline::Source::Receiver(source));
            let root = Val::Null;
            let env = Env::new(Val::Null);
            Some(pipeline.run_with_env(&root, &env, Some(j as &dyn pipeline::PipelineData)))
        }
        PlanNode::Call {
            receiver,
            call,
            optional,
        } => {
            let receiver = match eval_structural_derived(j, plan, *receiver)? {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            if *optional && receiver.is_null() {
                return Some(Ok(Val::Null));
            }
            Some(call.try_apply(&receiver).and_then(|result| {
                result.ok_or_else(|| EvalError(format!("{:?}: builtin unsupported", call.method)))
            }))
        }
        PlanNode::Chain { base, steps } => {
            let base = match eval_structural_derived(j, plan, *base)? {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            Some(eval_materialized_chain_suffix(base, steps))
        }
        _ => None,
    }
}

fn pipeline_body_is_current_free(body: &pipeline::PipelineBody) -> bool {
    body.stage_exprs.iter().all(|expr| expr.is_none())
        && body
            .sink
            .reducer_spec()
            .map(|spec| spec.sink_programs().next().is_none())
            .unwrap_or(true)
}

fn eval_materialized_chain_suffix(
    mut cur: Val,
    steps: &[PhysicalChainStep],
) -> Result<Val, EvalError> {
    for step in steps {
        cur = match step {
            PhysicalChainStep::Field(key) => cur.get_field(key.as_ref()),
            PhysicalChainStep::Index(idx) => cur.get_index(*idx),
            PhysicalChainStep::DynIndex(_) => {
                return Err(EvalError(
                    "structural suffix dynamic index requires generic evaluation".into(),
                ))
            }
        };
    }
    Ok(cur)
}

fn try_run_structural_plan(
    j: &Jetro,
    plan: &QueryPlan,
    root_id: NodeId,
) -> Result<Option<Val>, EvalError> {
    let PlanNode::Structural {
        plan: structural, ..
    } = plan.node(root_id)
    else {
        return Ok(None);
    };
    let Some(bytes) = j.raw_bytes() else {
        return Ok(None);
    };
    let Some(index) = j.lazy_structural_index()? else {
        return Ok(None);
    };
    structural.run(index, bytes).map(Some)
}

#[cfg(feature = "simd-json")]
fn try_run_tape_row_plan(
    plan: &QueryPlan,
    root_id: NodeId,
    tape: &crate::strref::TapeData,
) -> Option<Result<Val, EvalError>> {
    match plan.node(root_id) {
        PlanNode::Pipeline {
            source: PipelinePlanSource::FieldChain { keys },
            body,
        } => pipeline::run_tape_field_chain(body, tape, keys),
        _ => None,
    }
}

fn try_run_view_plan<'a, V>(
    plan: &QueryPlan,
    root_id: NodeId,
    root: V,
    cache: Option<&dyn pipeline::PipelineData>,
    allow_materialized_pipeline_fallback: bool,
    tape_pipeline_fallback: TapePipelineFallback<'a>,
) -> Option<Result<Val, EvalError>>
where
    V: ValueView<'a>,
{
    let ctx = ViewEvalCtx {
        plan,
        cache,
        allow_materialized_pipeline_fallback,
        tape_pipeline_fallback,
    };
    eval_view_val(&ctx, root_id, &root, &root)
}

fn eval_view_val<'a, V>(
    ctx: &ViewEvalCtx<'_, 'a>,
    id: NodeId,
    root: &V,
    current: &V,
) -> Option<Result<Val, EvalError>>
where
    V: ValueView<'a>,
{
    if let Some(view) = eval_view_ref(ctx.plan, id, root, current) {
        return Some(view.map(|view| view.materialize()));
    }

    match ctx.plan.node(id) {
        PlanNode::Literal(value) => Some(Ok(value.clone())),
        PlanNode::Pipeline {
            source: PipelinePlanSource::FieldChain { keys },
            body,
        } => eval_view_pipeline(ctx, root, keys, body),
        PlanNode::Structural { .. } => None,
        PlanNode::Object(fields) => eval_view_object(ctx, fields, root, current),
        PlanNode::Array(elems) => eval_view_array(ctx, elems, root, current),
        _ => None,
    }
}

fn eval_view_pipeline<'a, V>(
    ctx: &ViewEvalCtx<'_, 'a>,
    root: &V,
    keys: &[Arc<str>],
    body: &pipeline::PipelineBody,
) -> Option<Result<Val, EvalError>>
where
    V: ValueView<'a>,
{
    let source = view_pipeline::walk_fields(root.clone(), keys);
    if let Some(result) = view_pipeline::run(source.clone(), body, ctx.cache) {
        return Some(result);
    }
    #[cfg(feature = "simd-json")]
    if let Some(tape) = ctx.tape_pipeline_fallback {
        if let Some(result) = pipeline::run_tape_field_chain(body, tape, keys) {
            return Some(result);
        }
    }
    if !ctx.allow_materialized_pipeline_fallback {
        return None;
    }
    if !view_pipeline::can_run_materialized_receiver(body) {
        return None;
    }

    let pipeline = body
        .clone()
        .with_source(pipeline::Source::Receiver(source.materialize()));
    let root = Val::Null;
    let env = Env::new(Val::Null);
    Some(pipeline.run_with_env(&root, &env, ctx.cache))
}

fn eval_view_ref<'a, V>(
    plan: &QueryPlan,
    id: NodeId,
    root: &V,
    current: &V,
) -> Option<Result<V, EvalError>>
where
    V: ValueView<'a>,
{
    match plan.node(id) {
        PlanNode::Root => Some(Ok(root.clone())),
        PlanNode::Current => Some(Ok(current.clone())),
        PlanNode::RootPath(steps) => Some(Ok(walk_path_view(root.clone(), steps))),
        PlanNode::Chain { base, steps } => {
            let base = match eval_view_ref(plan, *base, root, current)? {
                Ok(base) => base,
                Err(err) => return Some(Err(err)),
            };
            Some(Ok(walk_chain_view(base, steps)?))
        }
        _ => None,
    }
}

fn eval_view_object<'a, V>(
    ctx: &ViewEvalCtx<'_, 'a>,
    fields: &[PhysicalObjField],
    root: &V,
    current: &V,
) -> Option<Result<Val, EvalError>>
where
    V: ValueView<'a>,
{
    let mut map: indexmap::IndexMap<Arc<str>, Val> =
        indexmap::IndexMap::with_capacity(fields.len());
    for field in fields {
        match field {
            PhysicalObjField::Kv {
                key,
                val,
                optional,
                cond,
            } => {
                if cond.is_some() {
                    return None;
                }
                let value = match eval_view_val(ctx, *val, root, current)? {
                    Ok(value) => value,
                    Err(err) => return Some(Err(err)),
                };
                if *optional && value.is_null() {
                    continue;
                }
                map.insert(Arc::clone(key), value);
            }
            PhysicalObjField::Short(_)
            | PhysicalObjField::Dynamic { .. }
            | PhysicalObjField::Spread(_)
            | PhysicalObjField::SpreadDeep(_) => return None,
        }
    }
    Some(Ok(Val::obj(map)))
}

fn eval_view_array<'a, V>(
    ctx: &ViewEvalCtx<'_, 'a>,
    elems: &[PhysicalArrayElem],
    root: &V,
    current: &V,
) -> Option<Result<Val, EvalError>>
where
    V: ValueView<'a>,
{
    let mut out = Vec::with_capacity(elems.len());
    for elem in elems {
        match elem {
            PhysicalArrayElem::Expr(expr) => match eval_view_val(ctx, *expr, root, current)? {
                Ok(value) => out.push(value),
                Err(err) => return Some(Err(err)),
            },
            PhysicalArrayElem::Spread(_) => return None,
        }
    }
    Some(Ok(Val::arr(out)))
}

fn walk_path_view<'a, V>(mut cur: V, steps: &[PhysicalPathStep]) -> V
where
    V: ValueView<'a>,
{
    for step in steps {
        cur = match step {
            PhysicalPathStep::Field(key) => cur.field(key.as_ref()),
            PhysicalPathStep::Index(idx) => cur.index(*idx),
        };
    }
    cur
}

fn walk_chain_view<'a, V>(mut cur: V, steps: &[PhysicalChainStep]) -> Option<V>
where
    V: ValueView<'a>,
{
    for step in steps {
        cur = match step {
            PhysicalChainStep::Field(key) => cur.field(key.as_ref()),
            PhysicalChainStep::Index(idx) => cur.index(*idx),
            PhysicalChainStep::DynIndex(_) => return None,
        };
    }
    Some(cur)
}

struct ExecCtx<'a> {
    j: &'a Jetro,
    plan: &'a QueryPlan,
    root: Val,
    env: Env,
    vm: VM,
}

impl ExecCtx<'_> {
    fn eval(&mut self, id: NodeId) -> Result<Val, EvalError> {
        match self.plan.node(id) {
            PlanNode::Literal(value) => Ok(value.clone()),
            PlanNode::Root => Ok(self.root.clone()),
            PlanNode::Current => Ok(self.env.current.clone()),
            PlanNode::Ident(name) => Ok(self
                .env
                .get_var(name.as_ref())
                .cloned()
                .unwrap_or_else(|| self.env.current.get_field(name.as_ref()))),
            PlanNode::Pipeline { source, body } => {
                let source = self.resolve_pipeline_source(source, body)?;
                let pipeline = body.clone().with_source(source.into_pipeline_source());
                pipeline.run_with_env(
                    &self.root,
                    &self.env,
                    Some(self.j as &dyn pipeline::PipelineData),
                )
            }
            PlanNode::RootPath(steps) => Ok(run_root_path(&self.root, steps)),
            PlanNode::Chain { base, steps } => self.eval_chain(*base, steps),
            PlanNode::Call {
                receiver,
                call,
                optional,
            } => {
                let receiver = self.eval(*receiver)?;
                if *optional && receiver.is_null() {
                    Ok(Val::Null)
                } else {
                    call.try_apply(&receiver)?
                        .ok_or_else(|| EvalError(format!("{:?}: builtin unsupported", call.method)))
                }
            }
            PlanNode::UnaryNeg(inner) => match self.eval(*inner)? {
                Val::Int(n) => Ok(Val::Int(-n)),
                Val::Float(f) => Ok(Val::Float(-f)),
                _ => Err(EvalError("unary minus requires a number".into())),
            },
            PlanNode::Not(inner) => {
                let value = self.eval(*inner)?;
                Ok(Val::Bool(!crate::util::is_truthy(&value)))
            }
            PlanNode::Binary { lhs, op, rhs } => self.eval_binary(*lhs, *op, *rhs),
            PlanNode::Kind { expr, ty, negate } => {
                let value = self.eval(*expr)?;
                let matched = crate::util::kind_matches(&value, *ty);
                Ok(Val::Bool(if *negate { !matched } else { matched }))
            }
            PlanNode::Coalesce { lhs, rhs } => {
                let lhs = self.eval(*lhs)?;
                if lhs.is_null() {
                    self.eval(*rhs)
                } else {
                    Ok(lhs)
                }
            }
            PlanNode::IfElse { cond, then_, else_ } => {
                let cond = self.eval(*cond)?;
                if crate::util::is_truthy(&cond) {
                    self.eval(*then_)
                } else {
                    self.eval(*else_)
                }
            }
            PlanNode::Try { body, default } => match self.eval(*body) {
                Ok(value) if !value.is_null() => Ok(value),
                Ok(_) | Err(_) => self.eval(*default),
            },
            PlanNode::Object(fields) => self.eval_object(fields),
            PlanNode::Array(elems) => self.eval_array(elems),
            PlanNode::Structural { fallback, .. } => self.vm.exec_in_env(fallback, &mut self.env),
            PlanNode::Let { name, init, body } => {
                let init_val = self.eval(*init)?;
                let body_env = self.env.with_var(name.as_ref(), init_val);
                let outer_env = std::mem::replace(&mut self.env, body_env);
                let result = self.eval(*body);
                self.env = outer_env;
                result
            }
            PlanNode::Vm(program) => self.vm.exec_in_env(program, &mut self.env),
        }
    }

    fn eval_chain(&mut self, base: NodeId, steps: &[PhysicalChainStep]) -> Result<Val, EvalError> {
        let mut cur = self.eval(base)?;
        for step in steps {
            cur = match step {
                PhysicalChainStep::Field(key) => cur.get_field(key.as_ref()),
                PhysicalChainStep::Index(idx) => cur.get_index(*idx),
                PhysicalChainStep::DynIndex(expr) => {
                    let key = self.eval(*expr)?;
                    match key {
                        Val::Int(idx) => cur.get_index(idx),
                        Val::Str(key) => cur.get_field(key.as_ref()),
                        Val::StrSlice(key) => cur.get_field(key.as_str()),
                        _ => Val::Null,
                    }
                }
            };
        }
        Ok(cur)
    }

    fn eval_binary(&mut self, lhs: NodeId, op: BinOp, rhs: NodeId) -> Result<Val, EvalError> {
        if op == BinOp::And {
            let lhs = self.eval(lhs)?;
            if !crate::util::is_truthy(&lhs) {
                return Ok(Val::Bool(false));
            }
            let rhs = self.eval(rhs)?;
            return Ok(Val::Bool(crate::util::is_truthy(&rhs)));
        }
        if op == BinOp::Or {
            let lhs = self.eval(lhs)?;
            if crate::util::is_truthy(&lhs) {
                return Ok(lhs);
            }
            return self.eval(rhs);
        }

        let lhs = self.eval(lhs)?;
        let rhs = self.eval(rhs)?;
        match op {
            BinOp::Add => crate::util::add_vals(lhs, rhs),
            BinOp::Sub => crate::util::num_op(lhs, rhs, |a, b| a - b, |a, b| a - b),
            BinOp::Mul => crate::util::num_op(lhs, rhs, |a, b| a * b, |a, b| a * b),
            BinOp::Div => {
                let denom = rhs.as_f64().unwrap_or(0.0);
                if denom == 0.0 {
                    Err(EvalError("division by zero".into()))
                } else {
                    Ok(Val::Float(lhs.as_f64().unwrap_or(0.0) / denom))
                }
            }
            BinOp::Mod => crate::util::num_op(lhs, rhs, |a, b| a % b, |a, b| a % b),
            BinOp::Eq => Ok(Val::Bool(crate::util::vals_eq(&lhs, &rhs))),
            BinOp::Neq => Ok(Val::Bool(!crate::util::vals_eq(&lhs, &rhs))),
            BinOp::Lt | BinOp::Lte | BinOp::Gt | BinOp::Gte => {
                Ok(Val::Bool(crate::util::cmp_vals_binop(&lhs, op, &rhs)))
            }
            BinOp::Fuzzy => {
                let lhs = match &lhs {
                    Val::Str(s) => s.to_lowercase(),
                    _ => crate::util::val_to_string(&lhs).to_lowercase(),
                };
                let rhs = match &rhs {
                    Val::Str(s) => s.to_lowercase(),
                    _ => crate::util::val_to_string(&rhs).to_lowercase(),
                };
                Ok(Val::Bool(lhs.contains(&rhs) || rhs.contains(&lhs)))
            }
            BinOp::And | BinOp::Or => unreachable!(),
        }
    }

    fn eval_object(&mut self, fields: &[PhysicalObjField]) -> Result<Val, EvalError> {
        let mut map: indexmap::IndexMap<Arc<str>, Val> =
            indexmap::IndexMap::with_capacity(fields.len());
        for field in fields {
            match field {
                PhysicalObjField::Kv {
                    key,
                    val,
                    optional,
                    cond,
                } => {
                    if let Some(cond) = cond {
                        if !crate::util::is_truthy(&self.eval(*cond)?) {
                            continue;
                        }
                    }
                    let value = self.eval(*val)?;
                    if *optional && value.is_null() {
                        continue;
                    }
                    map.insert(Arc::clone(key), value);
                }
                PhysicalObjField::Short(name) => {
                    let value = if let Some(value) = self.env.get_var(name.as_ref()) {
                        value.clone()
                    } else {
                        self.env.current.get_field(name.as_ref())
                    };
                    if !value.is_null() {
                        map.insert(Arc::clone(name), value);
                    }
                }
                PhysicalObjField::Dynamic { key, val } => {
                    let key = Arc::from(crate::util::val_to_key(&self.eval(*key)?).as_str());
                    let value = self.eval(*val)?;
                    map.insert(key, value);
                }
                PhysicalObjField::Spread(expr) => {
                    if let Val::Obj(other) = self.eval(*expr)? {
                        let entries = Arc::try_unwrap(other).unwrap_or_else(|m| (*m).clone());
                        for (key, value) in entries {
                            map.insert(key, value);
                        }
                    }
                }
                PhysicalObjField::SpreadDeep(expr) => {
                    if let Val::Obj(other) = self.eval(*expr)? {
                        let base = std::mem::take(&mut map);
                        let merged =
                            crate::util::deep_merge_concat(Val::obj(base), Val::Obj(other));
                        if let Val::Obj(merged) = merged {
                            map = Arc::try_unwrap(merged).unwrap_or_else(|m| (*m).clone());
                        }
                    }
                }
            }
        }
        Ok(Val::obj(map))
    }

    fn eval_array(&mut self, elems: &[PhysicalArrayElem]) -> Result<Val, EvalError> {
        let mut out = Vec::with_capacity(elems.len());
        for elem in elems {
            match elem {
                PhysicalArrayElem::Expr(expr) => out.push(self.eval(*expr)?),
                PhysicalArrayElem::Spread(expr) => {
                    let value = self.eval(*expr)?;
                    match value.into_vals() {
                        Ok(items) => out.extend(items),
                        Err(value) => out.push(value),
                    }
                }
            }
        }
        Ok(Val::arr(out))
    }
}

impl PipelineSourceResolver for ExecCtx<'_> {
    fn resolve_pipeline_source(
        &mut self,
        source: &PipelinePlanSource,
        _body: &pipeline::PipelineBody,
    ) -> Result<ResolvedPipelineSource, EvalError> {
        match source {
            PipelinePlanSource::FieldChain { keys } => {
                Ok(ResolvedPipelineSource::ValFieldChain { keys: keys.clone() })
            }
            PipelinePlanSource::Expr(source) => {
                Ok(ResolvedPipelineSource::ValReceiver(self.eval(*source)?))
            }
        }
    }
}

fn run_root_path(root: &Val, steps: &[PhysicalPathStep]) -> Val {
    let mut cur = root.clone();
    for step in steps {
        cur = match step {
            PhysicalPathStep::Field(key) => cur.get_field(key.as_ref()),
            PhysicalPathStep::Index(idx) => cur.get_index(*idx),
        };
    }
    cur
}
