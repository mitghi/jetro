//! Interpreter for `physical.rs` expression plans.

use std::sync::Arc;

use crate::ast::BinOp;
use crate::context::{Env, EvalError};
use crate::physical::{
    BackendPreference, NodeId, PhysicalArrayElem, PhysicalChainStep, PhysicalObjField,
    PhysicalPathStep, PipelinePlanSource, PlanNode, QueryPlan,
};
use crate::pipeline;
use crate::runtime::{PipelineSourceResolver, ResolvedPipelineSource};
use crate::value::Val;
use crate::value_view::{ValView, ValueView};
use crate::view_pipeline;
use crate::{Jetro, VM};

pub(crate) fn run(j: &Jetro, plan: &QueryPlan, root_id: NodeId) -> Result<Val, EvalError> {
    let mut ctx = ExecCtx {
        j,
        plan,
        root_id,
        root: None,
        env: None,
        vm: VM::new(),
    };
    ctx.eval(root_id)
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

struct ExecCtx<'a> {
    j: &'a Jetro,
    plan: &'a QueryPlan,
    root_id: NodeId,
    root: Option<Val>,
    env: Option<Env>,
    vm: VM,
}

impl ExecCtx<'_> {
    fn eval(&mut self, id: NodeId) -> Result<Val, EvalError> {
        self.eval_fast(id).unwrap_or_else(|| {
            Err(EvalError(format!(
                "no planned backend could execute physical node {}",
                id.0
            )))
        })
    }

    fn eval_interpreted(&mut self, id: NodeId) -> Result<Val, EvalError> {
        match self.plan.node(id) {
            PlanNode::Literal(value) => Ok(value.clone()),
            PlanNode::Root => self.root(),
            PlanNode::Current => Ok(self.env()?.current.clone()),
            PlanNode::Ident(name) => {
                let env = self.env()?;
                Ok(env
                    .get_var(name.as_ref())
                    .cloned()
                    .unwrap_or_else(|| env.current.get_field(name.as_ref())))
            }
            PlanNode::Pipeline { source, body } => {
                let source = self.resolve_pipeline_source(source, body)?;
                let pipeline = body.clone().with_source(source.into_pipeline_source());
                let root = self.root()?;
                let env = self.env()?.clone();
                pipeline.run_with_env(&root, &env, Some(self.j as &dyn pipeline::PipelineData))
            }
            PlanNode::RootPath(steps) => Ok(run_root_path(&self.root()?, steps)),
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
            PlanNode::Structural { fallback, .. } => {
                let mut env = self.take_env()?;
                let result = self.vm.exec_in_env(fallback, &mut env);
                self.env = Some(env);
                result
            }
            PlanNode::Let { name, init, body } => {
                let init_val = self.eval(*init)?;
                let body_env = self.env()?.with_var(name.as_ref(), init_val);
                let outer_env = self.env.replace(body_env);
                let result = self.eval(*body);
                self.env = outer_env;
                result
            }
            PlanNode::Vm(program) => {
                let mut env = self.take_env()?;
                let result = self.vm.exec_in_env(program, &mut env);
                self.env = Some(env);
                result
            }
        }
    }

    fn root(&mut self) -> Result<Val, EvalError> {
        if let Some(root) = &self.root {
            return Ok(root.clone());
        }
        let root = self.j.root_val()?;
        self.root = Some(root.clone());
        Ok(root)
    }

    fn env(&mut self) -> Result<&mut Env, EvalError> {
        if self.env.is_none() {
            let root = self.root()?;
            self.env = Some(Env::new(root));
        }
        Ok(self.env.as_mut().expect("env initialized"))
    }

    fn take_env(&mut self) -> Result<Env, EvalError> {
        if self.env.is_none() {
            let root = self.root()?;
            self.env = Some(Env::new(root));
        }
        Ok(self.env.take().expect("env initialized"))
    }

    fn eval_fast(&mut self, id: NodeId) -> Option<Result<Val, EvalError>> {
        for backend in self.plan.backend_preferences(id) {
            if let Some(result) = self.eval_backend(id, *backend) {
                return Some(result);
            }
        }
        None
    }

    fn eval_backend(
        &mut self,
        id: NodeId,
        backend: BackendPreference,
    ) -> Option<Result<Val, EvalError>> {
        match (backend, self.plan.node(id)) {
            (
                BackendPreference::Structural,
                PlanNode::Structural {
                    plan: structural, ..
                },
            ) => {
                let bytes = match self.j.raw_bytes() {
                    Some(bytes) => bytes,
                    None => return None,
                };
                let index = match self.j.lazy_structural_index() {
                    Ok(Some(index)) => index,
                    Ok(None) => return None,
                    Err(err) => return Some(Err(err)),
                };
                Some(structural.run(index, bytes))
            }
            (
                BackendPreference::TapeView
                | BackendPreference::TapeRows
                | BackendPreference::ValView
                | BackendPreference::MaterializedSource,
                PlanNode::Pipeline { source, body },
            ) => self.eval_pipeline_backend(backend, source, body),
            (BackendPreference::FastChildren, PlanNode::Pipeline { source, body }) => {
                self.eval_pipeline_backend(backend, source, body)
            }
            (BackendPreference::TapePath, PlanNode::RootPath(steps)) => {
                self.eval_root_path_fast(steps)
            }
            (BackendPreference::FastChildren, PlanNode::Object(fields)) => {
                self.eval_object_fast(fields)
            }
            (BackendPreference::FastChildren, PlanNode::Literal(value)) => Some(Ok(value.clone())),
            (BackendPreference::FastChildren, PlanNode::Array(elems)) => {
                self.eval_array_fast(elems)
            }
            (
                BackendPreference::FastChildren,
                PlanNode::Call {
                    receiver,
                    call,
                    optional,
                },
            ) => {
                let receiver = match self.eval_fast(*receiver)? {
                    Ok(value) => value,
                    Err(err) => return Some(Err(err)),
                };
                if *optional && receiver.is_null() {
                    return Some(Ok(Val::Null));
                }
                Some(call.try_apply(&receiver).and_then(|result| {
                    result
                        .ok_or_else(|| EvalError(format!("{:?}: builtin unsupported", call.method)))
                }))
            }
            (BackendPreference::FastChildren, PlanNode::UnaryNeg(inner)) => {
                let value = match self.eval_fast(*inner)? {
                    Ok(value) => value,
                    Err(err) => return Some(Err(err)),
                };
                Some(match value {
                    Val::Int(n) => Ok(Val::Int(-n)),
                    Val::Float(f) => Ok(Val::Float(-f)),
                    _ => Err(EvalError("unary minus requires a number".into())),
                })
            }
            (BackendPreference::FastChildren, PlanNode::Not(inner)) => {
                let value = match self.eval_fast(*inner)? {
                    Ok(value) => value,
                    Err(err) => return Some(Err(err)),
                };
                Some(Ok(Val::Bool(!crate::util::is_truthy(&value))))
            }
            (BackendPreference::FastChildren, PlanNode::Binary { lhs, op, rhs }) => {
                self.eval_binary_fast(*lhs, *op, *rhs)
            }
            (BackendPreference::FastChildren, PlanNode::Kind { expr, ty, negate }) => {
                let value = match self.eval_fast(*expr)? {
                    Ok(value) => value,
                    Err(err) => return Some(Err(err)),
                };
                let matched = crate::util::kind_matches(&value, *ty);
                Some(Ok(Val::Bool(if *negate { !matched } else { matched })))
            }
            (BackendPreference::FastChildren, PlanNode::Coalesce { lhs, rhs }) => {
                let lhs = match self.eval_fast(*lhs)? {
                    Ok(value) => value,
                    Err(err) => return Some(Err(err)),
                };
                if lhs.is_null() {
                    self.eval_fast(*rhs)
                } else {
                    Some(Ok(lhs))
                }
            }
            (BackendPreference::FastChildren, PlanNode::IfElse { cond, then_, else_ }) => {
                let cond = match self.eval_fast(*cond)? {
                    Ok(value) => value,
                    Err(err) => return Some(Err(err)),
                };
                if crate::util::is_truthy(&cond) {
                    self.eval_fast(*then_)
                } else {
                    self.eval_fast(*else_)
                }
            }
            (BackendPreference::FastChildren, PlanNode::Try { body, default }) => {
                match self.eval_fast(*body)? {
                    Ok(value) if !value.is_null() => Some(Ok(value)),
                    Ok(_) | Err(_) => self.eval_fast(*default),
                }
            }
            (BackendPreference::FastChildren, PlanNode::Chain { base, steps }) => {
                self.eval_chain_fast(*base, steps)
            }
            (BackendPreference::Interpreted, _) => {
                if id == self.root_id
                    && self.j.raw_bytes().is_some()
                    && self.plan.execution_facts(id).is_byte_native()
                {
                    return None;
                }
                Some(self.eval_interpreted(id))
            }
            _ => None,
        }
    }

    fn eval_pipeline_backend(
        &mut self,
        backend: BackendPreference,
        source: &PipelinePlanSource,
        body: &pipeline::PipelineBody,
    ) -> Option<Result<Val, EvalError>> {
        match (backend, source) {
            (
                BackendPreference::TapeView
                | BackendPreference::TapeRows
                | BackendPreference::ValView
                | BackendPreference::MaterializedSource,
                PipelinePlanSource::FieldChain { keys },
            ) => self.eval_field_chain_pipeline_backend(backend, keys, body),
            (BackendPreference::FastChildren, PipelinePlanSource::Expr(source))
                if body.can_run_with_materialized_receiver() =>
            {
                let source = match self.eval_fast(*source)? {
                    Ok(value) => value,
                    Err(err) => return Some(Err(err)),
                };
                let pipeline = body.clone().with_source(pipeline::Source::Receiver(source));
                let root = Val::Null;
                let env = Env::new(Val::Null);
                Some(pipeline.run_with_env(
                    &root,
                    &env,
                    Some(self.j as &dyn pipeline::PipelineData),
                ))
            }
            _ => None,
        }
    }

    fn eval_field_chain_pipeline_backend(
        &mut self,
        backend: BackendPreference,
        keys: &[Arc<str>],
        body: &pipeline::PipelineBody,
    ) -> Option<Result<Val, EvalError>> {
        match backend {
            BackendPreference::TapeView => self.eval_tape_view_pipeline(keys, body),
            BackendPreference::TapeRows => self.eval_tape_rows_pipeline(keys, body),
            BackendPreference::MaterializedSource => {
                self.eval_materialized_source_pipeline(keys, body)
            }
            BackendPreference::ValView => self.eval_val_view_pipeline(keys, body),
            _ => None,
        }
    }

    fn eval_tape_view_pipeline(
        &mut self,
        keys: &[Arc<str>],
        body: &pipeline::PipelineBody,
    ) -> Option<Result<Val, EvalError>> {
        #[cfg(feature = "simd-json")]
        {
            if let Some(tape) = match self.j.lazy_tape() {
                Ok(tape) => tape,
                Err(err) => return Some(Err(err)),
            } {
                let root = crate::value_view::TapeView::root(tape);
                let source = view_pipeline::walk_fields(root, keys);
                return view_pipeline::run(source, body, Some(self.j));
            }
        }
        None
    }

    fn eval_tape_rows_pipeline(
        &mut self,
        keys: &[Arc<str>],
        body: &pipeline::PipelineBody,
    ) -> Option<Result<Val, EvalError>> {
        #[cfg(feature = "simd-json")]
        {
            if let Some(tape) = match self.j.lazy_tape() {
                Ok(tape) => tape,
                Err(err) => return Some(Err(err)),
            } {
                return pipeline::run_tape_field_chain(body, tape, keys);
            }
        }
        None
    }

    fn eval_materialized_source_pipeline(
        &mut self,
        keys: &[Arc<str>],
        body: &pipeline::PipelineBody,
    ) -> Option<Result<Val, EvalError>> {
        if !body.can_run_with_materialized_receiver() {
            return None;
        }
        #[cfg(feature = "simd-json")]
        {
            if let Some(tape) = match self.j.lazy_tape() {
                Ok(tape) => tape,
                Err(err) => return Some(Err(err)),
            } {
                let root = crate::value_view::TapeView::root(tape);
                let source = view_pipeline::walk_fields(root, keys);
                let pipeline = body
                    .clone()
                    .with_source(pipeline::Source::Receiver(source.materialize()));
                let root = Val::Null;
                let env = Env::new(Val::Null);
                return Some(pipeline.run_with_env(
                    &root,
                    &env,
                    Some(self.j as &dyn pipeline::PipelineData),
                ));
            }
        }
        None
    }

    fn eval_val_view_pipeline(
        &mut self,
        keys: &[Arc<str>],
        body: &pipeline::PipelineBody,
    ) -> Option<Result<Val, EvalError>> {
        if self.j.raw_bytes().is_none() || self.root.is_some() {
            let root = match self.root() {
                Ok(root) => root,
                Err(err) => return Some(Err(err)),
            };
            let source = view_pipeline::walk_fields(ValView::new(&root), keys);
            if let Some(result) = view_pipeline::run(source, body, Some(self.j)) {
                return Some(result);
            }
        }
        None
    }

    fn eval_root_path_fast(
        &mut self,
        steps: &[PhysicalPathStep],
    ) -> Option<Result<Val, EvalError>> {
        #[cfg(feature = "simd-json")]
        if let Some(tape) = match self.j.lazy_tape() {
            Ok(tape) => tape,
            Err(err) => return Some(Err(err)),
        } {
            let root = crate::value_view::TapeView::root(tape);
            return Some(Ok(walk_path_view(root, steps).materialize()));
        }
        None
    }

    fn eval_object_fast(&mut self, fields: &[PhysicalObjField]) -> Option<Result<Val, EvalError>> {
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
                        let cond = match self.eval_fast(*cond)? {
                            Ok(value) => value,
                            Err(err) => return Some(Err(err)),
                        };
                        if !crate::util::is_truthy(&cond) {
                            continue;
                        }
                    }
                    let value = match self.eval_fast(*val)? {
                        Ok(value) => value,
                        Err(err) => return Some(Err(err)),
                    };
                    if *optional && value.is_null() {
                        continue;
                    }
                    map.insert(Arc::clone(key), value);
                }
                PhysicalObjField::Dynamic { key, val } => {
                    let key = match self.eval_fast(*key)? {
                        Ok(value) => Arc::from(crate::util::val_to_key(&value).as_str()),
                        Err(err) => return Some(Err(err)),
                    };
                    let value = match self.eval_fast(*val)? {
                        Ok(value) => value,
                        Err(err) => return Some(Err(err)),
                    };
                    map.insert(key, value);
                }
                PhysicalObjField::Spread(expr) => {
                    if let Val::Obj(other) = match self.eval_fast(*expr)? {
                        Ok(value) => value,
                        Err(err) => return Some(Err(err)),
                    } {
                        let entries = Arc::try_unwrap(other).unwrap_or_else(|m| (*m).clone());
                        for (key, value) in entries {
                            map.insert(key, value);
                        }
                    }
                }
                PhysicalObjField::SpreadDeep(expr) => {
                    if let Val::Obj(other) = match self.eval_fast(*expr)? {
                        Ok(value) => value,
                        Err(err) => return Some(Err(err)),
                    } {
                        let base = std::mem::take(&mut map);
                        let merged =
                            crate::util::deep_merge_concat(Val::obj(base), Val::Obj(other));
                        if let Val::Obj(merged) = merged {
                            map = Arc::try_unwrap(merged).unwrap_or_else(|m| (*m).clone());
                        }
                    }
                }
                PhysicalObjField::Short(_) => return None,
            }
        }
        Some(Ok(Val::obj(map)))
    }

    fn eval_array_fast(&mut self, elems: &[PhysicalArrayElem]) -> Option<Result<Val, EvalError>> {
        let mut out = Vec::with_capacity(elems.len());
        for elem in elems {
            match elem {
                PhysicalArrayElem::Expr(expr) => match self.eval_fast(*expr)? {
                    Ok(value) => out.push(value),
                    Err(err) => return Some(Err(err)),
                },
                PhysicalArrayElem::Spread(expr) => {
                    let value = match self.eval_fast(*expr)? {
                        Ok(value) => value,
                        Err(err) => return Some(Err(err)),
                    };
                    match value.into_vals() {
                        Ok(items) => out.extend(items),
                        Err(value) => out.push(value),
                    }
                }
            }
        }
        Some(Ok(Val::arr(out)))
    }

    fn eval_chain_fast(
        &mut self,
        base: NodeId,
        steps: &[PhysicalChainStep],
    ) -> Option<Result<Val, EvalError>> {
        let mut cur = match self.eval_fast(base)? {
            Ok(value) => value,
            Err(err) => return Some(Err(err)),
        };
        for step in steps {
            cur = match step {
                PhysicalChainStep::Field(key) => cur.get_field(key.as_ref()),
                PhysicalChainStep::Index(idx) => cur.get_index(*idx),
                PhysicalChainStep::DynIndex(expr) => {
                    let key = match self.eval_fast(*expr)? {
                        Ok(value) => value,
                        Err(err) => return Some(Err(err)),
                    };
                    match key {
                        Val::Int(idx) => cur.get_index(idx),
                        Val::Str(key) => cur.get_field(key.as_ref()),
                        Val::StrSlice(key) => cur.get_field(key.as_str()),
                        _ => Val::Null,
                    }
                }
            };
        }
        Some(Ok(cur))
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

    fn eval_binary_fast(
        &mut self,
        lhs: NodeId,
        op: BinOp,
        rhs: NodeId,
    ) -> Option<Result<Val, EvalError>> {
        if op == BinOp::And {
            let lhs = match self.eval_fast(lhs)? {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            if !crate::util::is_truthy(&lhs) {
                return Some(Ok(Val::Bool(false)));
            }
            let rhs = match self.eval_fast(rhs)? {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            return Some(Ok(Val::Bool(crate::util::is_truthy(&rhs))));
        }
        if op == BinOp::Or {
            let lhs = match self.eval_fast(lhs)? {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            if crate::util::is_truthy(&lhs) {
                return Some(Ok(lhs));
            }
            return self.eval_fast(rhs);
        }

        let lhs = match self.eval_fast(lhs)? {
            Ok(value) => value,
            Err(err) => return Some(Err(err)),
        };
        let rhs = match self.eval_fast(rhs)? {
            Ok(value) => value,
            Err(err) => return Some(Err(err)),
        };
        Some(self.apply_binary(lhs, op, rhs))
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
        self.apply_binary(lhs, op, rhs)
    }

    fn apply_binary(&self, lhs: Val, op: BinOp, rhs: Val) -> Result<Val, EvalError> {
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
                    let env = self.env()?;
                    let value = if let Some(value) = env.get_var(name.as_ref()) {
                        value.clone()
                    } else {
                        env.current.get_field(name.as_ref())
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
