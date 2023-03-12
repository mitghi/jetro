//! Module func provides abstraction for jetro functions.

use crate::context::{Context, Error, Func, FuncArg};
use serde_json::Value;
use std::{cell::RefCell, collections::BTreeMap, rc::Rc};

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

impl Default for FuncRegistry {
    fn default() -> Self {
        let mut output = FuncRegistry::new();
        output.register("reverse", Box::new(Reverse));
        output.register("formats", Box::new(Formats));
        output.register("sum", Box::new(SumFn));
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
}
