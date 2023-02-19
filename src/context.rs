use crate::parser;
#[allow(unused_imports)]
use serde::{Deserialize, Serialize};
use serde_json::map::Map;
use serde_json::Value;
use std::cell::RefCell;
use std::rc::Rc;

type PathOutput = Rc<RefCell<Vec<Rc<Value>>>>;
pub struct Path;
#[derive(Debug)]
pub struct PathResult(pub PathOutput);

#[derive(Debug, PartialEq)]
pub enum PickFilterInner {
    None,
    Str(String),
    KeyedStr { key: String, alias: String },
    Subpath(Vec<Filter>),
}

#[derive(Debug, PartialEq)]
pub enum Filter {
    Root,
    AnyChild,
    Child(String),
    Descendant(String),
    Pick(Vec<PickFilterInner>),
    ArrayIndex(usize),
    ArrayFrom(usize),
    ArrayTo(usize),
    Slice(usize, usize),
}

#[allow(dead_code)]
struct StackItem<'a> {
    value: Rc<Value>,
    filters: &'a [Filter],
    stack: Rc<RefCell<Vec<StackItem<'a>>>>,
}

pub struct Context<'a> {
    stack: Rc<RefCell<Vec<StackItem<'a>>>>,
    pub results: Rc<RefCell<Vec<Rc<Value>>>>,
}

impl Filter {
    fn pick(&self, obj: &Value) -> Option<Value> {
        let target_key = self.key();
        let descendant_key = target_key.unwrap_or("descendant".to_string());
        match &self {
            Filter::Pick(ref values) => {
                let mut new_map: Value = Value::Object(Map::new());
                for value in values {
                    match value {
                        PickFilterInner::Str(ref some_key) => {
                            match obj.get(&some_key) {
                                Some(v) => {
                                    new_map
                                        .as_object_mut()
                                        .unwrap()
                                        .insert(some_key.to_string(), v.clone());
                                }
                                _ => {}
                            };
                        }
                        PickFilterInner::KeyedStr { ref key, ref alias } => {
                            match obj.get(&key) {
                                Some(v) => {
                                    new_map
                                        .as_object_mut()
                                        .unwrap()
                                        .insert(alias.clone(), v.clone());
                                }
                                _ => {}
                            };
                        }
                        PickFilterInner::Subpath(ref some_subpath) => {
                            for item in
                                Path::collect_with_filter(obj.clone(), some_subpath.as_slice())
                                    .0
                                    .borrow()
                                    .iter()
                            {
                                match &*item.clone() {
                                    Value::Object(ref target_object) => {
                                        for (k, v) in target_object {
                                            new_map
                                                .as_object_mut()
                                                .unwrap()
                                                .insert(k.to_string(), v.clone());
                                        }
                                    }
                                    Value::String(ref some_str) => {
                                        new_map.as_object_mut().unwrap().insert(
                                            descendant_key.clone(),
                                            Value::String(some_str.to_string()),
                                        );
                                    }
                                    Value::Bool(ref some_bool) => {
                                        new_map.as_object_mut().unwrap().insert(
                                            descendant_key.clone(),
                                            Value::Bool(some_bool.clone()),
                                        );
                                    }
                                    Value::Number(ref some_number) => {
                                        new_map.as_object_mut().unwrap().insert(
                                            descendant_key.clone(),
                                            Value::Number(some_number.clone()),
                                        );
                                    }
                                    Value::Array(ref some_array) => {
                                        new_map.as_object_mut().unwrap().insert(
                                            descendant_key.clone(),
                                            Value::Array(some_array.clone()),
                                        );
                                    }
                                    Value::Null => {
                                        new_map
                                            .as_object_mut()
                                            .unwrap()
                                            .insert(descendant_key.clone(), Value::Null);
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
                Some(new_map)
            }
            _ => None,
        }
    }

    pub fn key(&self) -> Option<String> {
        match &self {
            Filter::Descendant(ref descendant) => {
                return Some(descendant[2..].to_string());
            }
            _ => None,
        }
    }
}

impl Path {
    pub fn collect<S: Into<String>>(v: Value, expr: S) -> PathResult {
        let expr: String = expr.into();
        let filters = parser::parse(&expr);
        let mut ctx = Context::new(v, filters.as_slice());
        ctx.collect();

        PathResult(ctx.results)
    }

    pub fn collect_with_filter(v: Value, filters: &[Filter]) -> PathResult {
        let mut ctx = Context::new(v, filters);
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

        Self { stack, results }
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

                    (Filter::ArrayIndex(ref index), Some(tail)) => match *current.value {
                        Value::Array(ref array) => {
                            if *index < array.len() {
                                let new_value = array[*index].clone();
                                let tlen = tail.len();
                                if tlen == 0 {
                                    self.results.borrow_mut().push(Rc::new(new_value));
                                } else {
                                    self.stack.borrow_mut().push(StackItem::new(
                                        Rc::new(new_value),
                                        tail,
                                        self.stack.clone(),
                                    ));
                                }
                            }
                        }
                        _ => {}
                    },

                    (Filter::Slice(ref from, ref to), Some(tail)) => match *current.value {
                        Value::Array(ref array) => {
                            if array.len() >= *to && *from < *to {
                                let new_slice = Value::Array(array[*from..*to].to_vec());
                                let tlen = tail.len();
                                if tlen == 0 {
                                    self.results.borrow_mut().push(Rc::new(new_slice));
                                } else {
                                    self.stack.borrow_mut().push(StackItem::new(
                                        Rc::new(new_slice),
                                        tail,
                                        self.stack.clone(),
                                    ));
                                }
                            }
                        }
                        _ => {}
                    },

                    (Filter::ArrayTo(ref index), Some(tail)) => match *current.value {
                        Value::Array(ref array) => {
                            if array.len() >= *index {
                                let new_array = Value::Array(array[..*index].to_vec());
                                let tlen = tail.len();
                                if tlen == 0 {
                                    self.results.borrow_mut().push(Rc::new(new_array));
                                } else {
                                    self.stack.borrow_mut().push(StackItem::new(
                                        Rc::new(new_array),
                                        tail,
                                        self.stack.clone(),
                                    ));
                                }
                            }
                        }
                        _ => {}
                    },

                    (Filter::ArrayFrom(ref index), Some(tail)) => match *current.value {
                        Value::Array(ref array) => {
                            if array.len() >= *index {
                                let new_array = Value::Array(array[*index..].to_vec());
                                let tlen = tail.len();
                                if tlen == 0 {
                                    self.results.borrow_mut().push(Rc::new(new_array));
                                } else {
                                    self.stack.borrow_mut().push(StackItem::new(
                                        Rc::new(new_array),
                                        tail,
                                        self.stack.clone(),
                                    ));
                                }
                            }
                        }
                        _ => {}
                    },

                    (Filter::Pick(_), Some(tail)) => match *current.value {
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
        let f: Filter = Filter::Pick(vec![
            PickFilterInner::Str("a".to_string()),
            PickFilterInner::Str("b".to_string()),
        ]);
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
        assert_eq!(*&output.a, "object_a".to_string());
        assert_eq!(*&output.b, "object_b".to_string());
    }

    #[test]
    fn test_pick_with_subpath() {
        let data = r#"
	 {
	   "name": "mr snuggle",
           "some_entry": {
           "some_obj": {
	       "obj": {
	         "a": "object_a",
	         "b": "object_b",
	         "c": "object_c",
	         "d": {"object_d": "some_value", "with_nested": {"object": "final_value"}}
	      }
            }
          }
        }
        "#;

        let values = Path::collect(
            serde_json::from_str(&data).unwrap(),
            ">/..obj/pick('a', >/..with_nested/pick('object'))",
        );

        assert_eq!(
            *values.0.borrow().clone(),
            vec![Rc::new(
                serde_json::json! {{"a": "object_a", "object": "final_value"}}
            )]
        );
    }

    #[test]
    fn test_pick_with_mixed_path() {
        let data = r#"
	 {
	   "name": "mr snuggle",
           "some_entry": {
           "some_obj": {
    	       "obj": {
	         "a": "object_a",
	         "b": "object_b",
	         "c": "object_c",
	         "d": {"object_d": "some_value", "with_nested": {"object": "final_value"}}
	      }
            }
          }
        }
        "#;

        let values = Path::collect(
            serde_json::from_str(&data).unwrap(),
            ">/..obj/pick('a' as 'foo', >/..object)",
        );

        assert_eq!(
            *values.0.borrow().clone(),
            vec![Rc::new(
                serde_json::json! {{"descendant": "final_value", "foo": "object_a"}}
            )]
        );
    }

    #[test]
    fn test_slice() {
        let data = r#"
	 {
	   "name": "mr snuggle",
           "values": [1,2,3,4,5,6]
        }
        "#;

        let values = Path::collect(serde_json::from_str(&data).unwrap(), ">/values/[1]");

        assert_eq!(
            *values.0.borrow().clone(),
            vec![Rc::new(serde_json::json!(2))],
        );
    }
}
