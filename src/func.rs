//! Module func provides abstraction for jetro functions.

use crate::context::{Error, Func, FuncArg};
use serde_json::Value;
use std::collections::BTreeMap;

pub(crate) trait Callable {
    fn call(&mut self, func: &Func, value: &Value) -> Result<Value, Error>;
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
    fn call(&mut self, func: &Func, value: &Value) -> Result<Value, Error> {
        let target = if let Some(result) = self.get(&func.name) {
            result
        } else {
            return Err(Error::FuncEval("target function not found".into()));
        };

        target.call(func, value)
    }
}

impl Registry for FuncRegistry {}

struct Reverse;
impl Callable for Reverse {
    fn call(&mut self, _: &Func, value: &Value) -> Result<Value, Error> {
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

impl Formats {
    fn ensure_str(&mut self, value: &Value) -> Result<(), Error> {
        match &value {
            Value::String(_) => Ok(()),
            _ => Err(Error::FuncEval("invalid type, expected string".to_string())),
        }
    }
}

impl Callable for Formats {
    fn call(&mut self, func: &Func, value: &Value) -> Result<Value, Error> {
        if func.args.len() < 2 {
            return Err(Error::FuncEval("deficit number of arguments".to_string()));
        }
        let format = match func.args.get(0).unwrap() {
            FuncArg::Key(some_key) => some_key,
            _ => {
                return Err(Error::FuncEval("invalid type, expected string".to_string()));
            }
        };
        let mut args: Vec<String> = vec![];
        for v in func.args[1..].iter() {
            match &v {
                FuncArg::Key(some_key) => {
                    args.push(some_key.clone());
                }
                _ => {
                    return Err(Error::FuncEval("invalid type, expected string".to_string()));
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
            _ => Err(Error::FuncEval("format failed".to_string())),
        }
    }
}

impl Default for FuncRegistry {
    fn default() -> Self {
        let mut output = FuncRegistry::new();
        output.register("reverse", Box::new(Reverse));
        output.register("formats", Box::new(Formats));
        output
    }
}

pub(crate) fn default_registry() -> Box<impl Registry> {
    let output: FuncRegistry = Default::default();
    Box::new(output)
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
        func.args.push(FuncArg::Key("test".to_string()));

        match d.call(&func, &serde_value) {
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
