//! Module containing types and functionalities for
//! evaluating jetro paths.

use crate::fmt as jetrofmt;
use crate::parser;
#[allow(unused_imports)]
use serde::{Deserialize, Serialize};
use serde_json::map::Map;
use serde_json::Value;
use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;
type PathOutput = Rc<RefCell<Vec<Rc<Value>>>>;

/// Path represents the entry for parsing and
/// evaluating a jetro path.
pub struct Path;

#[derive(Debug)]
pub struct PathResult(pub PathOutput);

enum Either<A, B> {
    Left(A),
    Right(B),
}

struct Sum(Either<i64, f64>);

impl Sum {
    fn new() -> Self {
        Self(Either::Left(0))
    }

    fn add_i64(&mut self, value: i64) {
        match self.0 {
            Either::Left(left) => {
                self.0 = Either::Left(left + value);
            }
            Either::Right(right) => {
                self.0 = Either::Right(right + value as f64);
            }
        }
    }

    fn add_f64(&mut self, value: f64) {
        match self.0 {
            Either::Left(left) => {
                self.0 = Either::Right(left as f64 + value);
            }
            Either::Right(right) => {
                self.0 = Either::Right(right + value);
            }
        }
    }

    fn add(&mut self, input: &Value) {
        if input.is_i64() {
            self.add_i64(input.as_i64().unwrap());
            return;
        }
        if input.is_f64() {
            self.add_f64(input.as_f64().unwrap());
            return;
        }
    }
}

impl From<Sum> for serde_json::Number {
    fn from(value: Sum) -> Self {
        match value.0 {
            Either::Left(left) => serde_json::Number::from(left),
            Either::Right(right) => serde_json::Number::from_f64(right).unwrap(),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum FilterOp {
    None,
    Less,
    Gt,
    Lq,
    Gq,
    Eq,
}

#[derive(Debug, PartialEq, Clone)]
pub enum FilterLogicalOp {
    None,
    And,
    Or,
}

impl FilterLogicalOp {
    pub fn get(input: &str) -> Option<Self> {
        match input {
            "and" => Some(FilterLogicalOp::And),
            "or" => Some(FilterLogicalOp::Or),
            _ => None,
        }
    }
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

#[derive(Debug, PartialEq, Clone)]
pub enum FormatOp {
    FormatString {
        format: String,
        arguments: Vec<String>,
        alias: String,
    },
}

#[derive(Debug, PartialEq, Clone)]
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

#[derive(Debug, PartialEq, Clone)]
pub enum FilterInnerRighthand {
    String(String),
    Bool(bool),
    Number(i64),
    Float(f64),
}

#[derive(Debug, PartialEq, Clone)]
pub enum FilterInner {
    Cond {
        left: String,
        op: FilterOp,
        right: FilterInnerRighthand,
    },
}

#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct FilterAST {
    pub operand: FilterLogicalOp,
    pub left: Option<Rc<RefCell<FilterInner>>>,
    pub right: Option<Rc<RefCell<FilterAST>>>,
}

#[derive(Debug, PartialEq, Clone)]
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

struct Context<'a> {
    root: Rc<Value>,
    stack: Rc<RefCell<Vec<StackItem<'a>>>>,
    formater: Box<dyn jetrofmt::KeyFormater>,
    pub results: Rc<RefCell<Vec<Rc<Value>>>>,
}

#[derive(Debug)]
pub enum Error {
    EmptyQuery,
    Parse(String),
    Eval(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Error::EmptyQuery => write!(f, "query is empty"),
            Error::Parse(ref msg) => write!(f, "error while parsing query: {}", &msg),
            Error::Eval(ref msg) => write!(f, "error while evaluating query: {}", &msg),
        }
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

macro_rules! do_comparision {
    ($lhs:expr, $op:expr, $rhs:expr) => {
        match $op {
            FilterOp::Less => $lhs < $rhs,
            FilterOp::Eq => $lhs == $rhs,
            FilterOp::Gt => $lhs > $rhs,
            FilterOp::Lq => $lhs <= $rhs,
            FilterOp::Gq => $lhs >= $rhs,
            _ => false,
        }
    };
}

macro_rules! search_filter_in_array {
    ($array:expr, $key:expr, $value:expr, $op:expr, $results:expr, $stack:expr, $tail:expr) => {
        let mut results: Vec<Value> = vec![];
        for value in $array {
            if value.is_object() {
                let obj = value.as_object().unwrap();
                match obj.get(&$key.clone()) {
                    Some(result) => match result {
                        Value::String(ref target_string) => match $value {
                            FilterInnerRighthand::String(ref str_value) => {
                                if do_comparision!(target_string, $op, str_value) {
                                    results.push(value.clone());
                                }
                            }
                            _ => {}
                        },
                        Value::Bool(v) => match $value {
                            FilterInnerRighthand::Bool(bool_value) => {
                                if do_comparision!(v, $op, bool_value) {
                                    results.push(value.clone());
                                }
                            }
                            _ => {}
                        },
                        Value::Number(n) => match $value {
                            FilterInnerRighthand::Number(number_value) => {
                                if do_comparision!(n.as_i64().unwrap(), $op, *number_value) {
                                    results.push(value.clone());
                                }
                            }
                            FilterInnerRighthand::Float(float_value) => {
                                if do_comparision!(n.as_f64().unwrap(), $op, *float_value) {
                                    results.push(value.clone());
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
        if results.len() > 0 {
            push_to_stack_or_produce!($results, $stack, $tail, Value::Array(results));
        }
    };
}

impl FilterAST {
    pub fn new(left: FilterInner) -> Self {
        Self {
            operand: FilterLogicalOp::None,
            left: Some(Rc::new(RefCell::new(left))),
            right: None,
        }
    }

    pub fn set_operand(&mut self, operand: FilterLogicalOp) {
        self.operand = operand;
    }

    pub fn link_right(
        &mut self,
        statement: FilterInner,
        operand: FilterLogicalOp,
    ) -> Rc<RefCell<Self>> {
        let mut rhs: Self = Self::new(statement);
        rhs.set_operand(operand);

        let inner = Rc::new(RefCell::new(rhs));
        let output = inner.clone();
        self.right = Some(inner);

        output
    }

    #[allow(dead_code)]
    pub fn repr(&self) -> String {
        if self.operand == FilterLogicalOp::None && self.right.is_none() {
            return format!("Filter({:?})", &self.left);
        }
        let rhs = &self.right.clone().unwrap().clone();
        return format!(
            "Filter({:?}) {:?} {:?}",
            &self.left,
            &self.operand,
            rhs.borrow().repr(),
        );
    }
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
    /// Returns a result given:
    ///
    /// # Arguments
    ///
    /// * `v` - A serde value
    /// * `expr` - A jetro path expression
    ///
    /// # Examples
    ///
    /// ```
    /// use jetro;
    /// use serde_json;
    ///
    /// let data = serde_json::json!({"name": "furryfurr", "purs": [{"sound": "prrrr"}, {"sound": "purrrrrr"}]});
    /// let mut results = jetro::context::Path::collect(data, ">/purs/#filter('sound' == 'prrrr')").expect("evaluation failed");
    ///
    /// #[derive(serde::Deserialize)]
    /// struct Item {
    ///   sound: String,
    /// }
    ///
    /// let items: Vec<Item> = results.from_index(0).unwrap();
    ///
    /// ```
    pub fn collect<S: Into<String>>(v: Value, expr: S) -> Result<PathResult, Error> {
        let expr: String = expr.into();
        let filters = match parser::parse(&expr) {
            Ok(result) => result,
            Err(err) => return Err(Error::Parse(err.to_string())),
        };

        let mut ctx = Context::new(v, filters.as_slice());
        ctx.collect();

        Ok(PathResult(ctx.results))
    }

    fn collect_with_filter(v: Value, filters: &[Filter]) -> PathResult {
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
        let formater: Box<dyn jetrofmt::KeyFormater> = jetrofmt::default();
        stack
            .borrow_mut()
            .push(StackItem::new(rv.clone(), filters, Rc::clone(&stack)));

        Self {
            root: rv.clone(),
            stack,
            formater,
            results,
        }
    }

    #[inline]
    pub fn reduce_stack_to_num_count(&mut self) -> i64 {
        let mut count: i64 = 0;
        let values = self.results.to_owned();
        self.results = Rc::new(RefCell::new(Vec::new()));
        for value in values.borrow().clone() {
            if value.is_number() {
                count += 1
            }
        }

        return count;
    }

    #[inline]
    pub fn reduce_stack_to_all_truth(&mut self) -> bool {
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

        return all;
    }

    #[inline]
    pub fn reduce_stack_to_sum(&mut self) -> Sum {
        let mut sum = Sum::new();
        let values = self.results.to_owned();
        self.results = Rc::new(RefCell::new(Vec::new()));
        for value in values.borrow().clone() {
            match *value.as_ref() {
                Value::Array(ref inner_array) => {
                    for v in inner_array {
                        if v.is_number() {
                            sum.add(&v);
                        }
                    }
                }
                Value::Number(_) => {
                    sum.add(&*value.as_ref());
                }
                _ => {}
            }
        }

        return sum;
    }

    pub fn collect(&mut self) {
        // TODO(mitghi): implement context errors

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
                            todo!("implement filter support for object");
                        }
                        Value::Array(ref _array) => match _cond {
                            FilterInner::Cond {
                                ref left,
                                ref op,
                                ref right,
                            } => {
                                search_filter_in_array!(
                                    _array,
                                    left,
                                    right,
                                    op,
                                    self.results,
                                    self.stack,
                                    _tail
                                );
                            }
                        },
                        _ => {
                            todo!("implement handling of unmatched arm for filter");
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
                        _ => {
                            let count: i64 = self.reduce_stack_to_num_count() + 1;
                            push_to_stack_or_produce!(
                                self.results,
                                self.stack,
                                tail,
                                Value::Number(serde_json::Number::from(count))
                            );
                        }
                    },

                    (Filter::Sum, Some(tail)) => match *current.value {
                        Value::Object(ref _obj) => {}
                        Value::Array(ref array) => {
                            let mut sum = self.reduce_stack_to_sum();
                            for value in array {
                                if value.is_number() {
                                    sum.add(&value);
                                }
                            }
                            push_to_stack_or_produce!(
                                self.results,
                                self.stack,
                                tail,
                                Value::Number(serde_json::Number::from(sum))
                            );
                        }
                        _ => {
                            let mut sum = self.reduce_stack_to_sum();
                            if current.value.is_number() {
                                sum.add(&current.value);
                            }
                            push_to_stack_or_produce!(
                                self.results,
                                self.stack,
                                tail,
                                Value::Number(serde_json::Number::from(sum))
                            );
                        }
                    },

                    (Filter::All, Some(tail)) => match *current.value {
                        Value::Object(ref _obj) => {}
                        Value::Bool(ref value) => {
                            let all = self.reduce_stack_to_all_truth() & (*value == true);
                            push_to_stack_or_produce!(
                                self.results,
                                self.stack,
                                tail,
                                Value::Bool(all)
                            );
                        }
                        Value::Array(ref array) => {
                            let mut all = self.reduce_stack_to_all_truth();
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

                            let output: Option<String> =
                                self.formater.format(&fmt, &current.value, args);

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

                                let output: Option<String> =
                                    self.formater.format(fmt, &current.value, args);

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

        let mut values = Path::collect(data, ">/..obj/#pick('a','b')").expect("unable to parse");

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
        )
        .expect("unable to parse");

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
        )
        .expect("unable to parse");

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

        let values = Path::collect(serde_json::from_str(&data).unwrap(), ">/values/[1]")
            .expect("unable to parse");

        assert_eq!(
            *values.0.borrow().clone(),
            vec![Rc::new(serde_json::json!(2))],
        );
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

        let values =
            Path::collect(data, ">/entry/('foo' | 'bar' | 'another')").expect("unable to parse");

        assert_eq!(values.0.borrow().len(), 1);
        assert_eq!(
            values.0.borrow()[0],
            Rc::new(Value::String(String::from("word")))
        );
    }

    #[test]
    fn test_filter() {
        let data = serde_json::json!(
            {
            "values": [
                {"name": "foo", "is_eligable": true},
                {"name": "bar", "is_eligable": false},
                {"name": "abc"},
                {"name": "xyz"}]
            }
        );

        let values = Path::collect(data, ">/values/#filter('is_eligable' == true)")
            .expect("unable to parse");

        assert_eq!(values.0.borrow().len(), 1);
        assert_eq!(
            values.0.borrow()[0],
            Rc::new(Value::Array(vec![
                serde_json::json!({"name": "foo", "is_eligable": true})
            ]))
        );
    }

    #[test]
    fn test_invalid() {
        let data = serde_json::json!({"k": "v"});
        let values = Path::collect(data, ">/asdfasdfw9u-q23r- q23r 2323r ");
        assert_eq!(values.is_err(), true);
    }

    #[test]
    fn test_prio() {
        let data = serde_json::json!({"entry": {"values": [{"name": "gearbox", "priority": 10}, {"name": "steam", "priority": 2}]}});
        let values = Path::collect(data, ">/..priority/#len");
        assert_eq!(values.is_ok(), true);
        assert_eq!(
            *values.unwrap().0.borrow(),
            vec![Rc::new(Value::Number(serde_json::Number::from(2)))]
        );
    }

    #[test]
    fn test_filter_number() {
        let data = serde_json::json!({"entry": {"values": [{"name": "gearbox", "priority": 10}, {"name": "steam", "priority": 2}]}});
        let values = Path::collect(data, ">/entry/values/#filter('priority' == 2)");
        assert_eq!(values.is_ok(), true);
        assert_eq!(
            *values.unwrap().0.borrow(),
            vec![Rc::new(Value::Array(vec![
                serde_json::json!({"name": "steam", "priority": 2})
            ]))]
        );
    }

    #[test]
    fn test_truth() {
        let data = serde_json::json!({"entry": {"values": [{"name": "gearbox", "priority": 10, "truth_a": true, "truth_b": false, "truth_c": false, "truth_d": true}, {"name": "steam", "priority": 2, "truth_a": false, "truth_b": true, "truth_c": false, "truth_d": true}]}});
        let tests: Vec<(String, Vec<Rc<Value>>)> = vec![
            (
                ">/..truth_a/#all".to_string(),
                vec![Rc::new(Value::Bool(false))],
            ),
            (
                ">/..truth_b/#all".to_string(),
                vec![Rc::new(Value::Bool(false))],
            ),
            (
                ">/..truth_c/#all".to_string(),
                vec![Rc::new(Value::Bool(false))],
            ),
            (
                ">/..truth_d/#all".to_string(),
                vec![Rc::new(Value::Bool(true))],
            ),
        ];

        for (path, expect) in tests.iter() {
            let values = Path::collect(data.clone(), path).unwrap();
            assert_eq!(*values.0.borrow(), *expect);
        }
    }

    #[test]
    fn test_float_filter_evaluation() {
        let data = serde_json::json!({"entry": {"values": [{"name": "gearbox", "some_float": 10.1112}, {"name": "steam", "some_float": 2.48}]}});
        let tests: Vec<(String, Vec<Rc<Value>>)> = vec![
            (
                ">/entry/values/#filter('some_float' >= 10.1112)".to_string(),
                vec![Rc::new(Value::Array(vec![
                    serde_json::json!({"name": "gearbox", "some_float": 10.1112}),
                ]))],
            ),
            (
                ">/entry/values/#filter('some_float' == 2.48)".to_string(),
                vec![Rc::new(Value::Array(vec![
                    serde_json::json!({"name": "steam", "some_float": 2.48}),
                ]))],
            ),
        ];

        for (path, expect) in tests.iter() {
            let values = Path::collect(data.clone(), path).unwrap();
            assert_eq!(*values.0.borrow(), *expect);
        }
    }
}
