//! Interpreter for physical `QueryPlan` IR nodes.
//!
//! `run` walks the plan tree produced by `planner`, evaluating each
//! `PlanNode` variant. Backend selection per node follows the preference list
//! attached by the planner: structural (jetro-experimental bitmap index),
//! view (tape/borrowed), pipeline (pull IR), then VM as the final fallback.

use std::sync::Arc;

use crate::ast::BinOp;
use crate::context::{Env, EvalError};
use crate::physical::{
    BackendPreference, NodeId, PhysicalArrayElem, PhysicalChainStep, PhysicalObjField,
    PhysicalPathStep, PipelinePlanSource, PlanNode, QueryPlan,
};
use crate::pipeline;
use crate::data::runtime::{PipelineSourceResolver, ResolvedPipelineSource};
use crate::data::value::Val;
use crate::data::view::{ValView, ValueView};
use crate::view_pipeline;
use crate::{Jetro, VM};

/// Entry point: constructs an `ExecCtx` and evaluates the plan DAG starting from `root_id`.
pub(crate) fn run(j: &Jetro, plan: &QueryPlan, root_id: NodeId) -> Result<Val, EvalError> {
    let mut ctx = ExecCtx {
        j,
        plan,
        root_id,
        root: None,
        env: None,
        locals: Vec::new(),
        vm: VM::new(),
    };
    ctx.eval(root_id)
}

/// Applies a sequence of field/index navigation steps to a borrowed `ValueView` without
/// materialising intermediate values.
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

/// Stateful execution context that drives tree-walking evaluation of a `QueryPlan`.
struct ExecCtx<'a> {
    /// The document handle providing raw bytes, tape, structural index, and `Val` root.
    j: &'a Jetro,
    /// The plan DAG being evaluated.
    plan: &'a QueryPlan,
    /// The `NodeId` of the plan's root, used to suppress interpreted fallback on byte-native roots.
    root_id: NodeId,
    /// Lazily-materialised document root value; `None` until first needed.
    root: Option<Val>,
    /// Lazily-constructed evaluation environment; `None` until a node requires `Env` access.
    env: Option<Env>,
    /// Stack of let-bound variable values visible to `FastChildren` evaluation paths.
    locals: Vec<(Arc<str>, Val)>,
    /// Private VM instance used for `Vm` and `Structural` fallback nodes.
    vm: VM,
}

impl ExecCtx<'_> {
    /// Evaluates node `id`, returning an error if no backend in its preference list could run.
    fn eval(&mut self, id: NodeId) -> Result<Val, EvalError> {
        self.eval_fast(id).unwrap_or_else(|| {
            Err(EvalError(format!(
                "no planned backend could execute physical node {}",
                id.0
            )))
        })
    }

    /// Evaluates node `id` through the full interpreted tree-walker, constructing `Env` as needed.
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
            PlanNode::Local(name) => self
                .env()?
                .get_var(name.as_ref())
                .cloned()
                .ok_or_else(|| EvalError(format!("unbound local {}", name))),
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

    /// Returns the materialised document root, lazily computing and caching it on first call.
    fn root(&mut self) -> Result<Val, EvalError> {
        if let Some(root) = &self.root {
            return Ok(root.clone());
        }
        let root = self.j.root_val()?;
        self.root = Some(root.clone());
        Ok(root)
    }

    /// Returns a mutable reference to the evaluation environment, creating it if not yet built.
    fn env(&mut self) -> Result<&mut Env, EvalError> {
        if self.env.is_none() {
            let root = self.root()?;
            self.env = Some(self.env_with_fast_locals(root));
        }
        Ok(self.env.as_mut().expect("env initialized"))
    }

    /// Removes and returns the `Env`, rebuilding it from the root if it was previously absent.
    fn take_env(&mut self) -> Result<Env, EvalError> {
        if self.env.is_none() {
            let root = self.root()?;
            self.env = Some(self.env_with_fast_locals(root));
        }
        Ok(self.env.take().expect("env initialized"))
    }

    /// Looks up `name` in the fast-local stack without constructing a full `Env`.
    fn fast_local(&self, name: &str) -> Option<Val> {
        self.locals
            .iter()
            .rev()
            .find(|(local, _)| local.as_ref() == name)
            .map(|(_, value)| value.clone())
    }

    /// Constructs an `Env` with `current = Null` and all fast-locals pre-populated.
    fn null_env_with_fast_locals(&self) -> Env {
        self.env_with_fast_locals(Val::Null)
    }

    /// Constructs an `Env` with the given `current` value and all fast-locals pre-populated.
    fn env_with_fast_locals(&self, current: Val) -> Env {
        let mut env = Env::new(current);
        for (name, value) in &self.locals {
            env = env.with_var(name.as_ref(), value.clone());
        }
        env
    }

    /// Iterates the backend preference list for `id` and returns the first backend that succeeds.
    ///
    /// Returns `None` when every backend is ineligible or declines to handle the node.
    fn eval_fast(&mut self, id: NodeId) -> Option<Result<Val, EvalError>> {
        let capabilities = self.plan.backend_capabilities(id);
        for backend in self.plan.backend_preferences(id) {
            if !capabilities.contains(backend.backend_set()) {
                continue;
            }
            if let Some(result) = self.eval_backend(id, *backend) {
                return Some(result);
            }
        }
        None
    }

    /// Attempts to run node `id` under the specific `backend`; returns `None` when the backend
    /// cannot handle the node (wrong node kind, missing tape/index, or preconditions not met).
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
            (BackendPreference::FastChildren, PlanNode::Local(name)) => {
                self.fast_local(name.as_ref()).map(Ok)
            }
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
            (BackendPreference::FastChildren, PlanNode::Let { name, init, body }) => {
                let init = match self.eval_fast(*init)? {
                    Ok(value) => value,
                    Err(err) => return Some(Err(err)),
                };
                self.locals.push((Arc::clone(name), init));
                let result = self.eval_fast(*body);
                self.locals.pop();
                result
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

    /// Routes a `Pipeline` node to the appropriate sub-backend based on its source type and
    /// the requested backend preference.
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
                let env = self.null_env_with_fast_locals();
                Some(pipeline.run_with_env(
                    &root,
                    &env,
                    Some(self.j as &dyn pipeline::PipelineData),
                ))
            }
            _ => None,
        }
    }

    /// Dispatches a field-chain pipeline to one of the four concrete sub-backends based on
    /// the preference value (TapeView, TapeRows, MaterializedSource, or ValView).
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

    /// Runs the pipeline entirely on the simd-json tape via a zero-copy view; requires
    /// `simd-json` feature and a live tape, otherwise returns `None`.
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
                let root = crate::data::view::TapeView::root(tape);
                let source = view_pipeline::walk_fields(root, keys);
                let env = self.null_env_with_fast_locals();
                return view_pipeline::run_with_env(source, body, Some(self.j), &env);
            }
        }
        None
    }

    /// Runs the pipeline using the tape row-bridge, materialising individual rows on demand;
    /// suitable when the first stage is not natively tape-evaluable.
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
                let env = self.null_env_with_fast_locals();
                return pipeline::run_tape_field_chain(body, tape, keys, &env);
            }
        }
        None
    }

    /// Materialises the field-chain source array from the tape and runs the pipeline on a `Val`
    /// receiver; used when the body requires a fully-materialised array but not the full document.
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
                let root = crate::data::view::TapeView::root(tape);
                let source = view_pipeline::walk_fields(root, keys);
                let pipeline = body
                    .clone()
                    .with_source(pipeline::Source::Receiver(source.materialize()));
                let root = Val::Null;
                let env = self.null_env_with_fast_locals();
                return Some(pipeline.run_with_env(
                    &root,
                    &env,
                    Some(self.j as &dyn pipeline::PipelineData),
                ));
            }
        }
        None
    }

    /// Runs the pipeline against a borrowed `ValView` of the already-materialised root value;
    /// used when raw bytes are unavailable or the root `Val` has already been constructed.
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
            let env = self.null_env_with_fast_locals();
            if let Some(result) = view_pipeline::run_with_env(source, body, Some(self.j), &env) {
                return Some(result);
            }
        }
        None
    }

    /// Navigates a `RootPath` directly on the simd-json tape, materialising only the leaf value;
    /// returns `None` when the tape is unavailable.
    fn eval_root_path_fast(
        &mut self,
        steps: &[PhysicalPathStep],
    ) -> Option<Result<Val, EvalError>> {
        #[cfg(feature = "simd-json")]
        if let Some(tape) = match self.j.lazy_tape() {
            Ok(tape) => tape,
            Err(err) => return Some(Err(err)),
        } {
            let root = crate::data::view::TapeView::root(tape);
            return Some(Ok(walk_path_view(root, steps).materialize()));
        }
        None
    }

    /// Evaluates an `Object` node via `FastChildren`, using `eval_fast` for every field value;
    /// returns `None` if any field requires `Env` access (e.g. `Short` fields).
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

    /// Evaluates an `Array` node via `FastChildren`, using `eval_fast` for every element and
    /// flattening spreads inline.
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

    /// Evaluates a `Chain` node via `FastChildren`, navigating field/index/dynamic steps on `Val`
    /// without constructing an `Env`; returns `None` if the base cannot be fast-evaluated.
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

    /// Evaluates a `Chain` node through the full interpreted path, always constructing values.
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

    /// Evaluates a binary operation via `FastChildren`; uses short-circuit logic for `And`/`Or`;
    /// returns `None` if either operand cannot be fast-evaluated.
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

    /// Evaluates a binary operation through the full interpreter, using short-circuit logic for
    /// `And`/`Or`.
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

    /// Applies a non-short-circuit binary operator to two already-evaluated operands.
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

    /// Evaluates an `Object` node through the full interpreter, handling shorthand fields via `Env`.
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

    /// Evaluates an `Array` node through the full interpreter, flattening spread elements.
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
    /// Converts a `PipelinePlanSource` into a `ResolvedPipelineSource` for use by the interpreter.
    ///
    /// Field-chain sources are returned as `ValFieldChain`; expression sources are evaluated eagerly.
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

/// Applies a `RootPath` step sequence to an already-materialised root `Val`.
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
