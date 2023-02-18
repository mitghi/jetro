use crate::parser;
use serde::{Deserialize, Serialize};
use serde_json::map::Map;
use serde_json::Value;
use std::cell::{Ref, RefCell};
use std::rc::Rc;

#[derive(Debug, PartialEq)]
pub enum Filter {
    Root,
    AnyChild,
    Child(String),
    Descendant(String),
    Pick(Vec<String>),
}

impl Filter {
    fn pick(&self, obj: &Value) -> Option<Value> {
        match &self {
            Filter::Pick(ref values) => {
                let mut new_map: Value = Value::Object(Map::new());
                for key in values {
                    match obj.get(&key) {
                        Some(v) => {
                            new_map
                                .as_object_mut()
                                .unwrap()
                                .insert(key.to_string(), v.clone());
                        }
                        _ => {}
                    };
                }
                Some(new_map)
            }
            _ => None,
        }
    }
}

struct StackItem<'a> {
    value: Rc<Value>,
    filters: &'a [Filter],
    stack: Rc<RefCell<Vec<StackItem<'a>>>>,
}

pub struct Context<'a> {
    stack: Rc<RefCell<Vec<StackItem<'a>>>>,
    filters: &'a [Filter],
    pub results: Rc<RefCell<Vec<Rc<Value>>>>,
}

type PathOutput = Rc<RefCell<Vec<Rc<Value>>>>;
pub struct Path;
pub struct PathResult(PathOutput);

impl Path {
    pub fn collect<S: Into<String>>(v: Value, expr: S) -> PathResult {
        let expr: String = expr.into();
        let filters = parser::parse(&expr);
        let mut ctx = Context::new(v, filters.as_slice());
        ctx.collect();

        PathResult(ctx.results)
    }
}

impl PathResult {
    pub fn shove<T: serde::de::DeserializeOwned>(&mut self, index: usize) -> T {
        let final_value: &Value = &*self.0.borrow_mut().remove(index);

        serde_json::from_value(final_value.clone().take()).unwrap()
    }
}

impl<'a> StackItem<'a> {
    pub fn new(
        value: Rc<Value>,
        filters: &'a [Filter],
        stack: Rc<RefCell<Vec<StackItem<'a>>>>,
    ) -> Self {
        Self {
            value,
            filters,
            stack,
        }
    }

    #[inline(always)]
    pub fn step(&self) -> Option<(&'a Filter, Option<&'a [Filter]>)> {
        match self.filters.split_first() {
            Some((ref head, tail)) => Some((head, Some(tail))),
            _ => None,
        }
    }
}

impl<'a> Context<'a> {
    pub fn new(value: Value, filters: &'a [Filter]) -> Self {
        let results: Rc<RefCell<Vec<Rc<Value>>>> = Rc::new(RefCell::new(Vec::new()));
        let stack: Rc<RefCell<Vec<StackItem<'a>>>> = Rc::new(RefCell::new(Vec::new()));
        let rv: Rc<Value> = Rc::new(value);
        stack
            .borrow_mut()
            .push(StackItem::new(rv.clone(), filters, Rc::clone(&stack)));

        Self {
            stack,
            filters,
            results,
        }
    }

    pub fn collect(&mut self) {
        while self.stack.borrow().len() > 0 {
            let current = if let Some(value) = self.stack.borrow_mut().pop() {
                value
            } else {
                return;
            };

            match current.step() {
                Some(step) => match step {
                    (Filter::Root, Some(tail)) => self.stack.borrow_mut().push(StackItem::new(
                        current.value,
                        tail,
                        self.stack.clone(),
                    )),

                    (Filter::Pick(ref values), Some(tail)) => match *current.value {
                        Value::Object(_) => {
                            let new_map = current.filters[0].pick(&current.value).unwrap();
                            let tlen = tail.len();
                            if tlen == 0 {
                                self.results.borrow_mut().push(Rc::new(new_map));
                            } else {
                                self.stack.borrow_mut().push(StackItem::new(
                                    Rc::new(new_map),
                                    tail,
                                    self.stack.clone(),
                                ));
                            }
                        }
                        Value::Array(_) => {}
                        _ => {}
                    },

                    (Filter::AnyChild, Some(tail)) => match *current.value {
                        Value::Object(ref obj) => {
                            let tlen = tail.len();
                            for (_, v) in obj {
                                if tlen == 0 {
                                    self.results.borrow_mut().push(Rc::new(v.clone()));
                                } else {
                                    self.stack.borrow_mut().push(StackItem::new(
                                        Rc::new(v.clone()),
                                        tail,
                                        self.stack.clone(),
                                    ));
                                }
                            }
                        }
                        Value::Array(ref array) => {
                            let tlen = tail.len();
                            for v in array.iter() {
                                if tlen == 0 {
                                    self.results.borrow_mut().push(Rc::new(v.clone()));
                                } else {
                                    self.stack.borrow_mut().push(StackItem::new(
                                        Rc::new(v.clone()),
                                        tail,
                                        self.stack.clone(),
                                    ));
                                }
                            }
                        }
                        _ => {}
                    },

                    (Filter::Child(ref name), Some(tail)) => match *current.value {
                        Value::Object(ref obj) => match obj.into_iter().find(|(k, _)| *k == name) {
                            Some((_, v)) => {
                                if tail.len() == 0 {
                                    self.results.borrow_mut().push(Rc::new(v.clone()));
                                } else {
                                    self.stack.borrow_mut().push(StackItem::new(
                                        Rc::new(v.clone()),
                                        tail,
                                        self.stack.clone(),
                                    ));
                                }
                            }
                            _ => {}
                        },
                        _ => {}
                    },

                    (Filter::Descendant(ref descendant), Some(tail)) => match *current.value {
                        Value::Object(ref obj) => {
                            for (k, v) in obj {
                                let filters: &[Filter] = if k == descendant {
                                    tail
                                } else {
                                    current.filters
                                };

                                if filters.len() == 0 {
                                    self.results.borrow_mut().push(Rc::new(v.clone()));
                                } else {
                                    self.stack.borrow_mut().push(StackItem::new(
                                        Rc::new(v.clone()),
                                        filters,
                                        self.stack.clone(),
                                    ));
                                }
                            }
                        }
                        Value::Array(ref array) => {
                            for v in array {
                                self.stack.borrow_mut().push(StackItem::new(
                                    Rc::new(v.clone()),
                                    current.filters,
                                    self.stack.clone(),
                                ));
                            }
                        }
                        _ => {}
                    },

                    _ => {}
                },
                _ => {}
            }
        }
    }
}

#[cfg(test)]
mod test {
    use serde_json::Value;

    use super::*;
    use serde_json;

    #[test]
    fn test_filter_pick() {
        let data = r#"
	 {
	   "name": "mr snuggle",
	   "obj": {
	     "a": "object_a",
	     "b": "object_b",
	     "c": "object_c",
	     "d": "object_d"
	   }
        }
        "#;
        let v: Value = serde_json::from_str(data).unwrap();
        let f: Filter = Filter::Pick(vec!["a".to_string(), "b".to_string()]);
        let v = v.get("obj".to_string()).unwrap();
        let result = f.pick(&v);

        assert_eq!(result.is_some(), true);
        assert_eq!(
            result.unwrap(),
            serde_json::json!({"a": "object_a", "b": "object_b" })
        );
    }

    #[test]
    fn test_new_jpath_context() {
        let data = r#"
	 {
	   "name": "mr snuggle",
	   "obj": {
	     "some": "object"
	   },
	   "arr": [
	     1,
	     2,
	     3,
	     4
	   ],
	   "foo": {
	     "deep": {
	       "of": {
		 "nested": {
		   "deeply": {
		     "within": "value"
		   }
		 }
	       }
	     }
	   }
        }
        "#;

        let v: Value = serde_json::from_str(&data).unwrap();

        let filters = vec![
            Filter::Root,
            Filter::Child("foo".to_string()),
            Filter::Descendant("within".to_string()),
        ];

        let mut ctx = Context::new(v, &filters);
        ctx.collect();

        for v in ctx.results.borrow().iter() {
            println!("got v {:?}", v);
        }

        assert_eq!(
            *ctx.results.borrow().clone(),
            vec![Rc::new(Value::String("value".to_string()))]
        );
    }

    #[test]
    fn test_pick() {
        let data = r#"
	 {
	   "name": "mr snuggle",
           "some_entry": {
           "some_obj": {
    	       "obj": {
	         "a": "object_a",
	         "b": "object_b",
	         "c": "object_c",
	         "d": "object_d"
	      }
            }
          }
        }
        "#;

        let mut values = Path::collect(
            serde_json::from_str(&data).unwrap(),
            ">/..obj/pick('a','b')",
        );

        #[derive(Serialize, Deserialize)]
        pub struct Output {
            pub a: String,
            pub b: String,
        }

        let output: Output = values.shove(0);
        println!("output: {:?}", output.a);
    }
}
