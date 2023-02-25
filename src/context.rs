use crate::parser;
use dynfmt::{Format, SimpleCurlyFormat};
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
pub enum FilterOp {
    Less,
    Gt,
    Lq,
    Gq,
    Eq,
}

impl FilterOp {
    pub fn get(input: &str) -> Option<Self> {
        match input {
            "==" => Some(Self::Eq),
            "<=" => Some(Self::Lq),
            ">=" => Some(Self::Gq),
            "<" => Some(Self::Less),
            ">" => Some(Self::Gt),
            _ => None,
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum FormatOp {
    FormatString {
        format: String,
        arguments: Vec<String>,
        alias: String,
    },
}

#[derive(Debug, PartialEq)]
pub enum PickFilterInner {
    None,
    Str(String),
    KeyedStr {
        key: String,
        alias: String,
    },
    Subpath(Vec<Filter>, bool),
    KeyedSubpath {
        subpath: Vec<Filter>,
        alias: String,
        reverse: bool,
    },
}

#[derive(Debug, PartialEq)]
pub enum FilterInner {
    Cond {
        left: String,
        op: FilterOp,
        right: String,
    },
}

#[derive(Debug, PartialEq)]
pub enum Filter {
    Root,
    AnyChild,
    Child(String),
    Descendant(String),
    Pick(Vec<PickFilterInner>),
    Format(FormatOp),
    ArrayIndex(usize),
    ArrayFrom(usize),
    ArrayTo(usize),
    Slice(usize, usize),
    Filter(FilterInner),
    GroupedChild(Vec<String>),
    All,
    Len,
    Sum,
}

#[allow(dead_code)]
struct StackItem<'a> {
    value: Rc<Value>,
    filters: &'a [Filter],
    stack: Rc<RefCell<Vec<StackItem<'a>>>>,
}

pub struct Context<'a> {
    root: Rc<Value>,
    stack: Rc<RefCell<Vec<StackItem<'a>>>>,
    pub results: Rc<RefCell<Vec<Rc<Value>>>>,
}

trait KeyFormater {
    fn format(&self) -> Option<String>;
}

struct FormatImpl<'a> {
    format: &'a str,
    value: &'a Value,
    keys: &'a Vec<String>,
}

impl<'a> FormatImpl<'a> {
    fn format(&self) -> Option<String> {
        let mut values: Vec<&'a Value> = vec![];
        for key in self.keys.iter() {
            match self.value.get(&key) {
                Some(result) => values.push(result),
                _ => {}
            };
        }
        match SimpleCurlyFormat.format(self.format, values) {
            Ok(result) => Some(result.into()),
            _ => None,
        }
    }
}

impl<'a> KeyFormater for FormatImpl<'a> {
    fn format(&self) -> Option<String> {
        self.format()
    }
}

macro_rules! push_to_stack_or_produce {
    ($results:expr, $stack:expr, $tail:expr, $value:expr) => {
        let tlen = $tail.len();
        if tlen == 0 {
            $results.borrow_mut().push(Rc::new($value));
        } else {
            $stack
                .borrow_mut()
                .push(StackItem::new(Rc::new($value), $tail, $stack.clone()));
        }
    };
}

macro_rules! push_to_stack_iter_or_produce {
    ($results:expr, $stack:expr, $tail:expr, $obj:expr) => {
        let tlen = $tail.len();
        for (_k, v) in $obj {
            if tlen == 0 {
                $results.borrow_mut().push(Rc::new(v.clone()));
            } else {
                $stack
                    .borrow_mut()
                    .push(StackItem::new(Rc::new(v.clone()), $tail, $stack.clone()));
            }
        }
    };
}

macro_rules! match_value {
    ($target_object:expr, $target_map:expr, $target_key:expr, $some_path:expr) => {{
        for item in Path::collect_with_filter($target_object.clone(), $some_path.as_slice())
            .0
            .borrow()
            .iter()
        {
            match &*item.clone() {
                Value::Object(ref target_object) => {
                    for (k, v) in target_object {
                        $target_map
                            .as_object_mut()
                            .unwrap()
                            .insert(k.clone(), v.clone());
                    }
                }
                Value::String(ref some_str) => {
                    $target_map
                        .as_object_mut()
                        .unwrap()
                        .insert($target_key.clone(), Value::String(some_str.to_string()));
                }
                Value::Bool(ref some_bool) => {
                    $target_map
                        .as_object_mut()
                        .unwrap()
                        .insert($target_key.clone(), Value::Bool(some_bool.clone()));
                }
                Value::Number(ref some_number) => {
                    $target_map
                        .as_object_mut()
                        .unwrap()
                        .insert($target_key.clone(), Value::Number(some_number.clone()));
                }
                Value::Array(ref some_array) => {
                    $target_map
                        .as_object_mut()
                        .unwrap()
                        .insert($target_key.clone(), Value::Array(some_array.clone()));
                }
                Value::Null => {
                    $target_map
                        .as_object_mut()
                        .unwrap()
                        .insert($target_key.clone(), Value::Null);
                }
            }
        }
    }};
}

impl Filter {
    fn pick(&self, obj: &Value, ctx: Option<&Context>) -> Option<Value> {
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

                        PickFilterInner::KeyedSubpath {
                            subpath: ref some_subpath,
                            alias: ref some_alias,
                            reverse,
                        } => {
                            match_value!(
                                if *reverse && ctx.is_some() {
                                    ctx.unwrap().root.as_ref()
                                } else {
                                    obj
                                },
                                new_map,
                                some_alias,
                                some_subpath
                            );
                        }

                        PickFilterInner::Subpath(ref some_subpath, reverse) => {
                            match_value!(
                                if *reverse && ctx.is_some() {
                                    ctx.unwrap().root.as_ref()
                                } else {
                                    obj
                                },
                                new_map,
                                descendant_key,
                                some_subpath
                            )
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
    pub fn from_index<T: serde::de::DeserializeOwned>(&mut self, index: usize) -> Option<T> {
        let final_value: &Value = &*self.0.borrow_mut().remove(index);

        match serde_json::from_value(final_value.clone().take()) {
            Ok(result) => Some(result),
            _ => None,
        }
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
            root: rv.clone(),
            stack,
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

                    (Filter::GroupedChild(ref vec), Some(tail)) => match *current.value {
                        Value::Object(ref obj) => {
                            let mut target: Option<&Value> = None;
                            'ML: for target_key in vec.iter() {
                                match obj.get(target_key) {
                                    Some(result) => {
                                        if result.is_null() {
                                            continue 'ML;
                                        }
                                        target = Some(result);
                                        break 'ML;
                                    }
                                    _ => {}
                                };
                            }
                            match target {
                                Some(result) => {
                                    push_to_stack_or_produce!(
                                        self.results,
                                        self.stack,
                                        tail,
                                        result.clone()
                                    );
                                }
                                _ => {}
                            }
                        }
                        _ => {}
                    },

                    (Filter::Filter(ref _cond), Some(_tail)) => match *current.value {
                        Value::Object(ref _obj) => {
                            todo!();
                        }
                        Value::Array(ref _array) => {
                            todo!();
                        }
                        _ => {
                            todo!();
                        }
                    },

                    (Filter::Len, Some(tail)) => match *current.value {
                        Value::Object(ref obj) => {
                            push_to_stack_or_produce!(
                                self.results,
                                self.stack,
                                tail,
                                Value::Number(serde_json::Number::from(obj.len()))
                            );
                        }
                        Value::Array(ref array) => {
                            push_to_stack_or_produce!(
                                self.results,
                                self.stack,
                                tail,
                                Value::Number(serde_json::Number::from(array.len()))
                            );
                        }
                        _ => {}
                    },

                    (Filter::Sum, Some(tail)) => match *current.value {
                        Value::Object(ref _obj) => {}
                        Value::Array(ref array) => {
                            let mut sum: i64 = 0;
                            let values = self.results.to_owned();
                            self.results = Rc::new(RefCell::new(Vec::new()));
                            for value in values.borrow().clone() {
                                match *value.as_ref() {
                                    Value::Array(ref inner_array) => {
                                        for v in inner_array {
                                            if v.is_number() {
                                                sum += v.as_i64().unwrap();
                                            }
                                        }
                                    }
                                    Value::Number(ref num) => {
                                        sum += num.as_i64().unwrap();
                                    }
                                    _ => {}
                                }
                            }
                            for value in array {
                                if value.is_number() {
                                    sum += value.as_i64().unwrap();
                                }
                            }
                            push_to_stack_or_produce!(
                                self.results,
                                self.stack,
                                tail,
                                Value::Number(serde_json::Number::from(sum))
                            );
                        }
                        _ => {}
                    },

                    (Filter::All, Some(tail)) => match *current.value {
                        Value::Object(ref _obj) => {}
                        Value::Bool(ref value) => {
                            let mut all = true;
                            let values = self.results.to_owned();
                            self.results = Rc::new(RefCell::new(Vec::new()));
                            for value in values.borrow().clone() {
                                match *value.as_ref() {
                                    Value::Array(ref inner_array) => {
                                        for v in inner_array {
                                            if v.is_boolean() {
                                                all = all & (v.as_bool().unwrap() == true);
                                            }
                                        }
                                    }
                                    Value::Bool(ref value) => {
                                        all = all & (*value == true);
                                    }
                                    _ => {}
                                }
                            }

                            all = all & (*value == true);

                            push_to_stack_or_produce!(
                                self.results,
                                self.stack,
                                tail,
                                Value::Bool(all)
                            );
                        }
                        Value::Array(ref array) => {
                            let mut all = true;
                            let values = self.results.to_owned();
                            self.results = Rc::new(RefCell::new(Vec::new()));
                            for value in values.borrow().clone() {
                                match *value.as_ref() {
                                    Value::Array(ref inner_array) => {
                                        for v in inner_array {
                                            if v.is_boolean() {
                                                all = all & (v.as_bool().unwrap() == true);
                                            }
                                        }
                                    }
                                    Value::Bool(ref value) => {
                                        all = all & (*value == true);
                                    }
                                    _ => {}
                                }
                            }
                            for value in array {
                                if value.is_boolean() {
                                    all = all & (value.as_bool().unwrap() == true);
                                }
                            }
                            push_to_stack_or_produce!(
                                self.results,
                                self.stack,
                                tail,
                                Value::Bool(all)
                            );
                        }
                        _ => {}
                    },

                    (Filter::Format(ref target_format), Some(tail)) => match *current.value {
                        Value::Object(ref obj) => {
                            let FormatOp::FormatString {
                                format: ref fmt,
                                arguments: ref args,
                                ref alias,
                            } = target_format;

                            let output: Option<String> = FormatImpl {
                                format: fmt,
                                value: &current.value,
                                keys: args,
                            }
                            .format();

                            match output {
                                Some(output) => {
                                    let mut result = obj.clone();
                                    result.insert(alias.to_string(), Value::String(output));

                                    push_to_stack_or_produce!(
                                        self.results,
                                        self.stack,
                                        tail,
                                        Value::Object(result)
                                    );
                                }
                                _ => {}
                            }
                        }
                        Value::Array(ref array) => {
                            for e in array.iter() {
                                let FormatOp::FormatString {
                                    format: ref fmt,
                                    arguments: ref args,
                                    ref alias,
                                } = target_format;

                                let output: Option<String> = FormatImpl {
                                    format: fmt,
                                    value: &current.value,
                                    keys: args,
                                }
                                .format();

                                match output {
                                    Some(output) => {
                                        let mut result = e.clone();
                                        match result.as_object_mut() {
                                            Some(ref mut handle) => {
                                                handle.insert(
                                                    alias.to_string(),
                                                    Value::String(output),
                                                );
                                            }
                                            _ => {}
                                        };

                                        push_to_stack_or_produce!(
                                            self.results,
                                            self.stack,
                                            tail,
                                            result
                                        );
                                    }
                                    _ => {}
                                }
                            }
                        }
                        _ => {}
                    },

                    (Filter::ArrayIndex(ref index), Some(tail)) => match *current.value {
                        Value::Array(ref array) => {
                            if *index < array.len() {
                                let new_value = array[*index].clone();
                                push_to_stack_or_produce!(
                                    self.results,
                                    self.stack,
                                    tail,
                                    new_value
                                );
                            }
                        }
                        _ => {}
                    },

                    (Filter::Slice(ref from, ref to), Some(tail)) => match *current.value {
                        Value::Array(ref array) => {
                            if array.len() >= *to && *from < *to {
                                let new_slice = Value::Array(array[*from..*to].to_vec());
                                push_to_stack_or_produce!(
                                    self.results,
                                    self.stack,
                                    tail,
                                    new_slice
                                );
                            }
                        }
                        _ => {}
                    },

                    (Filter::ArrayTo(ref index), Some(tail)) => match *current.value {
                        Value::Array(ref array) => {
                            if array.len() >= *index {
                                let new_array = Value::Array(array[..*index].to_vec());
                                push_to_stack_or_produce!(
                                    self.results,
                                    self.stack,
                                    tail,
                                    new_array
                                );
                            }
                        }
                        _ => {}
                    },

                    (Filter::ArrayFrom(ref index), Some(tail)) => match *current.value {
                        Value::Array(ref array) => {
                            if array.len() >= *index {
                                let new_array = Value::Array(array[*index..].to_vec());
                                push_to_stack_or_produce!(
                                    self.results,
                                    self.stack,
                                    tail,
                                    new_array
                                );
                            }
                        }
                        _ => {}
                    },

                    (Filter::Pick(_), Some(tail)) => match *current.value {
                        Value::Object(_) => {
                            let new_map = current.filters[0]
                                .pick(&current.value, Some(&self))
                                .unwrap();
                            push_to_stack_or_produce!(self.results, self.stack, tail, new_map);
                        }
                        Value::Array(_) => {
                            let new_array = current.filters[0]
                                .pick(&current.value, Some(&self))
                                .unwrap();
                            push_to_stack_or_produce!(self.results, self.stack, tail, new_array);
                        }
                        _ => {}
                    },

                    (Filter::AnyChild, Some(tail)) => match *current.value {
                        Value::Object(ref obj) => {
                            push_to_stack_iter_or_produce!(self.results, self.stack, tail, obj);
                        }
                        Value::Array(ref array) => {
                            push_to_stack_iter_or_produce!(
                                self.results,
                                self.stack,
                                tail,
                                array.iter().enumerate()
                            );
                        }
                        _ => {}
                    },

                    (Filter::Child(ref name), Some(tail)) => match *current.value {
                        Value::Object(ref obj) => match obj.into_iter().find(|(k, _)| *k == name) {
                            Some((_, v)) => {
                                push_to_stack_or_produce!(
                                    self.results,
                                    self.stack,
                                    tail,
                                    v.clone()
                                );
                            }
                            _ => {}
                        },
                        Value::Number(ref num) => {
                            push_to_stack_or_produce!(
                                self.results,
                                self.stack,
                                tail,
                                Value::Number(num.clone())
                            );
                        }
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
        let result = f.pick(&v, None);

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
        let data = serde_json::json!(
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
          );

        let mut values = Path::collect(data, ">/..obj/#pick('a','b')");

        #[derive(Serialize, Deserialize)]
        pub struct Output {
            pub a: String,
            pub b: String,
        }

        let result: Option<Output> = values.from_index(0);
        assert_eq!(result.is_some(), true);

        let output = result.unwrap();
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
            ">/..obj/#pick('a', >/..with_nested/#pick('object'))",
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
            ">/..obj/#pick('a' as 'foo', >/..object)",
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
           "values": [1,2,3,4,5,6, {"isEnabled": true}],
           "some_value": {"isEnabled": true}
        }
        "#;

        let values = Path::collect(serde_json::from_str(&data).unwrap(), ">/values/[1]");

        assert_eq!(
            *values.0.borrow().clone(),
            vec![Rc::new(serde_json::json!(2))],
        );
    }

    #[test]
    fn test_format_impl() {
        let data = serde_json::json!({
            "name": "mr snuggle",
            "alias": "jetro",
        });

        let keys = vec!["name".to_string(), "alias".to_string()];

        let format_impl: Box<dyn KeyFormater> = Box::new(FormatImpl {
            format: "{}_{}",
            value: &data,
            keys: &keys,
        });

        assert_eq!(format_impl.format(), Some("mr snuggle_jetro".to_string()));
    }

    #[test]
    fn test_grouped_child() {
        let data = serde_json::json!({
            "entry": {
        "some": "value",
        "foo": null,
        "another": "word",
        "till": "deal"
            }
        });

        let values = Path::collect(data, ">/entry/('foo' | 'bar' | 'another')");

        assert_eq!(values.0.borrow().len(), 1);
        assert_eq!(
            values.0.borrow()[0],
            Rc::new(Value::String(String::from("word")))
        );
    }
}
