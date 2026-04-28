//! Interpreter for `physical.rs` expression plans.

use std::sync::Arc;

use crate::ast::BinOp;
use crate::context::{Env, EvalError};
use crate::physical::{
    NodeId, PhysicalArrayElem, PhysicalChainStep, PhysicalObjField, PhysicalPathStep, PlanNode,
    QueryPlan,
};
use crate::pipeline;
use crate::value::Val;
use crate::{Jetro, VM};

pub(crate) fn run(j: &Jetro, plan: &QueryPlan, root_id: NodeId) -> Result<Val, EvalError> {
    let root = j.root_val();
    let mut ctx = ExecCtx {
        j,
        plan,
        root,
        env: Env::new(j.root_val()),
        vm: VM::new(),
    };
    ctx.eval(root_id)
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
            PlanNode::Pipeline(pipeline) => {
                pipeline.run_with(&self.root, Some(self.j as &dyn pipeline::PipelineData))
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
                PhysicalArrayElem::Spread(expr) => match self.eval(*expr)? {
                    Val::Arr(items) => {
                        let items = Arc::try_unwrap(items).unwrap_or_else(|items| (*items).clone());
                        out.extend(items);
                    }
                    Val::IntVec(items) => out.extend(items.iter().map(|n| Val::Int(*n))),
                    Val::FloatVec(items) => out.extend(items.iter().map(|f| Val::Float(*f))),
                    Val::StrVec(items) => out.extend(items.iter().cloned().map(Val::Str)),
                    Val::StrSliceVec(items) => out.extend(items.iter().cloned().map(Val::StrSlice)),
                    other => out.push(other),
                },
            }
        }
        Ok(Val::arr(out))
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
