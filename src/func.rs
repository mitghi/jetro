//! Module func provides abstraction for jetro functions.

use crate::context::{Context, Error, Filter, Func, FuncArg, MapBody, Path};
use serde_json::Value;
use std::{cell::RefCell, collections::BTreeMap, rc::Rc};

use super::context::MapAST;

pub(crate) trait Callable {
    fn call(
        &mut self,
        func: &Func,
        value: &Value,
        ctx: Option<&mut Context<'_>>,
    ) -> Result<Value, Error>;
}

pub(crate) trait Registry: Callable {}

pub(crate) struct FuncRegistry {
    map: BTreeMap<String, Box<dyn Callable>>,
}

impl FuncRegistry {
    pub fn new() -> Self {
        Self {
            map: BTreeMap::new(),
        }
    }

    #[inline(always)]
    pub fn register<S: Into<String>>(&mut self, name: S, func: Box<dyn Callable>) {
        self.map.insert(name.into(), func);
    }

    #[inline(always)]
    pub fn get<S: Into<String>>(&mut self, name: S) -> Option<&mut Box<dyn Callable>> {
        self.map.get_mut(&name.into())
    }
}

impl Callable for FuncRegistry {
    fn call(
        &mut self,
        func: &Func,
        value: &Value,
        ctx: Option<&mut Context<'_>>,
    ) -> Result<Value, Error> {
        let target = if let Some(result) = self.get(&func.name) {
            result
        } else {
            return Err(Error::FuncEval("target function not found".into()));
        };

        target.call(func, value, ctx)
    }
}

impl Registry for FuncRegistry {}

struct Reverse;
impl Callable for Reverse {
    fn call(
        &mut self,
        _: &Func,
        value: &Value,
        _ctx: Option<&mut Context<'_>>,
    ) -> Result<Value, Error> {
        match &value {
            Value::Array(ref array) => {
                let mut v = array.clone();
                v.reverse();
                Ok(Value::Array(v))
            }
            _ => Err(Error::FuncEval(
                "expected json value of type array".to_string(),
            )),
        }
    }
}

struct Formats;
impl Callable for Formats {
    fn call(
        &mut self,
        func: &Func,
        value: &Value,
        _ctx: Option<&mut Context<'_>>,
    ) -> Result<Value, Error> {
        if func.args.len() < 2 {
            return Err(Error::FuncEval("deficit number of arguments".to_owned()));
        }
        let format = match func.args.get(0).unwrap() {
            FuncArg::Key(some_key) => some_key,
            _ => {
                return Err(Error::FuncEval("invalid type, expected string".to_owned()));
            }
        };
        if func.alias.is_none() {
            return Err(Error::FuncEval(
                "expected alias to have some value".to_owned(),
            ));
        }
        let mut args: Vec<String> = vec![];
        for v in func.args[1..].iter() {
            match &v {
                FuncArg::Key(some_key) => {
                    args.push(some_key.clone());
                }
                _ => {
                    return Err(Error::FuncEval("invalid type, expected string".to_owned()));
                }
            };
        }

        let formater: Box<dyn crate::fmt::KeyFormater> = crate::fmt::default_formater();
        match formater.eval(
            &crate::context::FormatOp::FormatString {
                format: format.to_string(),
                arguments: args,
                alias: func.alias.clone().unwrap(),
            },
            &value,
        ) {
            Some(output) => Ok(output),
            _ => Err(Error::FuncEval("format failed".to_owned())),
        }
    }
}

struct SumFn;
impl Callable for SumFn {
    fn call(
        &mut self,
        _func: &Func,
        value: &Value,
        ctx: Option<&mut Context<'_>>,
    ) -> Result<Value, Error> {
        match &value {
            Value::Object(ref _obj) => {
                return Err(Error::FuncEval(
                    "sum on objects is not implemented".to_owned(),
                ));
            }
            Value::Array(ref array) => {
                let mut sum = ctx.unwrap().reduce_stack_to_sum();
                for value in array {
                    if value.is_number() {
                        sum.add(&value);
                    }
                }

                return Ok(Value::Number(serde_json::Number::from(sum)));
            }
            _ => {
                let mut sum = ctx.unwrap().reduce_stack_to_sum();
                if value.is_number() {
                    sum.add(&value);
                }

                return Ok(Value::Number(serde_json::Number::from(sum)));
            }
        };
    }
}

struct LenFn;
impl Callable for LenFn {
    fn call(
        &mut self,
        _func: &Func,
        value: &Value,
        ctx: Option<&mut Context<'_>>,
    ) -> Result<Value, Error> {
        match &value {
            Value::Object(ref obj) => {
                return Ok(Value::Number(serde_json::Number::from(obj.len())));
            }
            Value::Array(ref array) => {
                return Ok(Value::Number(serde_json::Number::from(array.len())));
            }
            _ => {
                let count: i64 = ctx.unwrap().reduce_stack_to_num_count() + 1;
                return Ok(Value::Number(serde_json::Number::from(count)));
            }
        }
    }
}

struct Head;
impl Callable for Head {
    fn call(
        &mut self,
        _func: &Func,
        value: &Value,
        _ctx: Option<&mut Context<'_>>,
    ) -> Result<Value, Error> {
        match &value {
            Value::Array(ref array) => {
                if array.len() == 0 {
                    return Ok(value.clone());
                } else {
                    let head = array[0].clone();
                    return Ok(head);
                }
            }
            _ => {
                return Err(Error::FuncEval("expected array".to_owned()));
            }
        }
    }
}

struct Tail;
impl Callable for Tail {
    fn call(
        &mut self,
        _func: &Func,
        value: &Value,
        _ctx: Option<&mut Context<'_>>,
    ) -> Result<Value, Error> {
        match &value {
            Value::Array(ref array) => match array.len() {
                0 | 1 => {
                    return Ok(Value::Array(vec![]));
                }
                _ => {
                    let tail = array[1..].to_vec();
                    return Ok(Value::Array(tail));
                }
            },
            _ => {
                return Err(Error::FuncEval("expected array".to_owned()));
            }
        }
    }
}

struct AllOnBoolean;
impl Callable for AllOnBoolean {
    fn call(
        &mut self,
        _func: &Func,
        value: &Value,
        ctx: Option<&mut Context<'_>>,
    ) -> Result<Value, Error> {
        match &value {
            Value::Bool(ref value) => {
                let all = ctx.unwrap().reduce_stack_to_all_truth();
                return Ok(Value::Bool(all & (*value == true)));
            }
            Value::Array(ref array) => {
                let mut all = ctx.unwrap().reduce_stack_to_all_truth();
                for value in array {
                    if value.is_boolean() {
                        all = all & (value.as_bool().unwrap() == true);
                    }
                }
                return Ok(Value::Bool(all));
            }
            _ => {
                return Err(Error::FuncEval("input is not reducable".to_owned()));
            }
        }
    }
}

struct MapFn;
impl MapFn {
    fn eval(&mut self, value: &Value, subpath: &[Filter]) -> Result<Vec<Value>, Error> {
        let mut output: Vec<Value> = vec![];
        match &value {
            Value::Array(ref array) => {
                for item in array {
                    let result = Path::collect_with_filter(item.clone(), &subpath);
                    if result.0.borrow().len() == 0 {
                        return Err(Error::FuncEval(
                            "map statement do not evaluates to anything".to_owned(),
                        ));
                    }
                    let head = result.0.borrow()[0].clone();
                    output.push((*head).clone());
                }
            }
            _ => {}
        };
        return Ok(output);
    }
}

impl Callable for MapFn {
    fn call(
        &mut self,
        func: &Func,
        value: &Value,
        _ctx: Option<&mut Context<'_>>,
    ) -> Result<Value, Error> {
        match func.args.get(0) {
            Some(&FuncArg::MapStmt(MapAST { arg: _, ref body })) => match &body {
                MapBody::Method {
                    name: _,
                    subpath: _,
                } => {
                    todo!("not implemented");
                    return Err(Error::FuncEval("WIP: not implemented".to_owned()));
                }
                MapBody::Subpath(ref subpath) => {
                    let output = self.eval(&value, &subpath.as_slice())?;
                    return Ok(Value::Array(output));
                }
                _ => {
                    return Err(Error::FuncEval("expetcted method call on path".to_owned()));
                }
            },
            _ => {
                return Err(Error::FuncEval(
                    "expected first argument to be map statement".to_owned(),
                ));
            }
        };
    }
}

struct Keys;
impl Callable for Keys {
    fn call(
        &mut self,
        func: &Func,
        value: &Value,
        _ctx: Option<&mut Context<'_>>,
    ) -> Result<Value, Error> {
        if func.args.len() > 0 {
            return Err(Error::FuncEval("keys takes no argument".to_owned()));
        }

        match &value {
            Value::Object(ref obj) => {
                let keys: Vec<Value> = obj.keys().map(|v| Value::String(v.to_string())).collect();
                return Ok(Value::Array(keys));
            }
            _ => {
                return Err(Error::FuncEval(
                    "keys can be exec only on objects".to_owned(),
                ));
            }
        }
    }
}

struct Values;
impl Callable for Values {
    fn call(
        &mut self,
        func: &Func,
        value: &Value,
        _ctx: Option<&mut Context<'_>>,
    ) -> Result<Value, Error> {
        if func.args.len() > 0 {
            return Err(Error::FuncEval("values takes no argument".to_owned()));
        }

        match &value {
            Value::Object(ref obj) => {
                let values: Vec<Value> = obj.values().map(|v| v.clone()).collect();
                return Ok(Value::Array(values));
            }
            _ => {
                return Err(Error::FuncEval(
                    "values can be exec only on objects".to_owned(),
                ));
            }
        }
    }
}

struct Min;
impl Callable for Min {
    fn call(
        &mut self,
        _func: &Func,
        value: &Value,
        _ctx: Option<&mut Context<'_>>,
    ) -> Result<Value, Error> {
        match &value {
            Value::Array(ref array) => {
                let mut result = array
                    .iter()
                    .filter(|v| v.is_number())
                    .map(|v| v.as_f64().unwrap())
                    .collect::<Vec<f64>>();
                result.sort_by(|a, b| a.partial_cmp(b).unwrap());
                match result.get(0) {
                    Some(output) => {
                        return Ok(Value::Number(
                            serde_json::Number::from_f64(*output).unwrap(),
                        ));
                    }
                    _ => {
                        return Ok(Value::Array(Vec::<Value>::new()));
                    }
                }
            }
            _ => {
                return Err(Error::FuncEval("expected value to be array".to_owned()));
            }
        }
    }
}

struct Max;
impl Callable for Max {
    fn call(
        &mut self,
        _func: &Func,
        value: &Value,
        _ctx: Option<&mut Context<'_>>,
    ) -> Result<Value, Error> {
        match &value {
            Value::Array(ref array) => {
                let mut result = array
                    .iter()
                    .filter(|v| v.is_number())
                    .map(|v| v.as_f64().unwrap())
                    .collect::<Vec<f64>>();
                result.sort_by(|a, b| b.partial_cmp(a).unwrap());
                match result.get(0) {
                    Some(output) => {
                        return Ok(Value::Number(
                            serde_json::Number::from_f64(*output).unwrap(),
                        ));
                    }
                    _ => {
                        return Ok(Value::Array(Vec::<Value>::new()));
                    }
                }
            }
            _ => {
                return Err(Error::FuncEval("expected value to be array".to_owned()));
            }
        }
    }
}

impl Default for FuncRegistry {
    fn default() -> Self {
        let mut output = FuncRegistry::new();
        output.register("reverse", Box::new(Reverse));
        output.register("formats", Box::new(Formats));
        output.register("sum", Box::new(SumFn));
        output.register("len", Box::new(LenFn));
        output.register("head", Box::new(Head));
        output.register("tail", Box::new(Tail));
        output.register("all", Box::new(AllOnBoolean));
        output.register("map", Box::new(MapFn));
        output.register("keys", Box::new(Keys));
        output.register("values", Box::new(Values));
        output.register("min", Box::new(Min));
        output.register("max", Box::new(Max));
        output
    }
}

pub(crate) fn default_registry() -> Rc<RefCell<impl Registry>> {
    let output: FuncRegistry = Default::default();
    Rc::new(RefCell::new(output))
}

#[cfg(test)]
mod test {
    use crate::context::FuncArg;

    use super::*;

    #[test]
    fn insert_and_fetch() {
        const NAME: &'static str = "foo";
        let mut m = FuncRegistry::new();
        m.register(NAME, Box::new(Reverse));

        assert_eq!(m.map.len(), 1);

        let v = m.get(NAME);
        assert_eq!(v.is_none(), false);

        let d = v.unwrap().as_mut();
        let serde_value = Value::Array(vec![Value::Bool(false), Value::Bool(true)]);

        let mut func = Func::new();
        func.args.push(FuncArg::Key("test".to_owned()));

        match d.call(&func, &serde_value, None) {
            Ok(result) => match result {
                Value::Array(ref array) => {
                    assert_eq!(*array, vec![Value::Bool(true), Value::Bool(false)]);
                }
                _ => {
                    panic!("invalid types");
                }
            },
            Err(err) => panic!("{}", err),
        };
    }

    #[test]
    fn get_head() {
        const NAME: &'static str = "head";
        let mut m = FuncRegistry::new();
        m.register("head", Box::new(Head));

        assert_eq!(m.map.len(), 1);

        let v = m.get(NAME);
        assert_eq!(v.is_none(), false);

        let d = v.unwrap().as_mut();
        let serde_value = Value::Array(vec![
            Value::String("foo".to_string()),
            Value::String("bar".to_string()),
        ]);

        let func = Func::new();
        match d.call(&func, &serde_value, None) {
            Ok(result) => match result {
                Value::String(ref output) => {
                    assert_eq!(*output, Value::String("foo".to_string()));
                }
                _ => {
                    panic!("invalid types {:?}", result);
                }
            },
            Err(err) => {
                panic!("{}", err);
            }
        }
    }
}
