//! Module containing types and functionalities for
//! evaluating jetro paths.

use crate::parser;
#[allow(unused_imports)]
use serde::{Deserialize, Serialize};
use serde_json::map::Map;
use serde_json::Value;
use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

/// PathOutput is type of fully evaluated jetro expression.
type PathOutput = Vec<Value>;

/// Path represents the entry for parsing and
/// evaluating a jetro path.
pub struct Path;

#[derive(Debug)]
pub struct PathResult(pub PathOutput);

/// Either contains either left or right value.
enum Either<A, B> {
    Left(A),
    Right(B),
}

/// Sum either contains left parameter or
/// upgrades permanentely to the type of
/// right parameter when addition of both
/// types take place.
pub(crate) struct Sum(Either<i64, f64>);

impl Sum {
    fn new() -> Self {
        // initially, use zero value of left parameter
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

    pub(crate) fn add(&mut self, input: &Value) {
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

/// FilterOp represents comparision
/// operators inside filter function.
#[derive(Debug, PartialEq, Clone)]
pub enum FilterOp {
    None,
    Less,
    Gt,
    Lq,
    Gq,
    Eq,
    NotEq,
    Almost,
}

/// FilterLogicalOp is logical operation
/// between several filters.
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
            "~=" => Some(Self::Almost),
            "!=" => Some(Self::NotEq),
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

/// PickFilterInner represents arguments
/// of pick function.
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

/// FilterInnerRighthand represents the
/// right handside of filter operator.
#[derive(Debug, PartialEq, Clone)]
pub enum FilterInnerRighthand {
    String(String),
    Bool(bool),
    Number(i64),
    Float(f64),
}

/// FilterInner represents a filter
/// statement.
#[derive(Debug, PartialEq, Clone)]
pub enum FilterInner {
    Cond {
        left: String,
        op: FilterOp,
        right: FilterInnerRighthand,
    },
}

#[allow(dead_code)]
#[derive(Debug, PartialEq, Clone)]
// FilterAST represents multi filter in form of
// abstract syntax tree. It represents the following
// structure:
//
//                     Filter
//         .______________|_____________.
//         |              |             |
// Left(FilterInner)  operator  Right(FilterAST)
//                                      |
//                              Filter(Operator)
//                                      |
//                                     ...
//
// The operator operates at least two filters.
// The left inner filter, evaluates single filter
//  expression.
// The right filter is recursive definition of
//  the same structure.
//
// In case of odd arrity of filters, the inner
// right most FilterAST contains a No-Op operator
// with its left filter set, and its right filter
// set to None.
//
// Filter with arrity one, represents the same structure
// with No-Op opeator, therefore evaluates to left expression.
pub struct FilterAST {
    pub operator: FilterLogicalOp,
    pub left: Option<FilterInner>,
    pub right: Option<Rc<FilterAST>>,
}

#[derive(Debug, PartialEq, Clone)]
pub enum FilterDescendant {
    Single(String),
    Pair(String, String),
}

/// Filter contains operations that transform, match
/// or search the input based on structure generated
/// the from parser.
#[derive(Debug, PartialEq, Clone)]
pub enum Filter {
    Root,
    AnyChild,
    Child(String),
    DescendantChild(FilterDescendant),
    Pick(Vec<PickFilterInner>),
    ArrayIndex(usize),
    ArrayFrom(usize),
    ArrayTo(usize),
    Slice(usize, usize),
    Filter(FilterInner),
    MultiFilter(FilterAST),
    GroupedChild(Vec<String>),
    Function(Func),
}

/// StackItem represents an abstract step
/// in the control stack. It evaluates the
/// head of filters and either produces a
/// result ( when terminal ) or produces
/// more stack items.
#[allow(dead_code)]
struct StackItem<'a> {
    value: Value,
    filters: &'a [Filter],
    stack: Rc<RefCell<Vec<StackItem<'a>>>>,
}

/// Context binds components required for traversing
/// and evaluating a jetro expression.
pub(crate) struct Context<'a> {
    root: Value,
    stack: Rc<RefCell<Vec<StackItem<'a>>>>,
    registry: Rc<RefCell<dyn crate::func::Registry>>,
    pub results: Rc<RefCell<Vec<Value>>>,
    step_results: Rc<RefCell<Vec<Value>>>,
}

/// MapBody represents the body of map function.
#[derive(Debug, PartialEq, Clone)]
pub enum MapBody {
    None,
    Method { name: String, subpath: Vec<Filter> },
    Subpath(Vec<Filter>),
}

/// MapAST represents the abstract structure
/// of map function.
#[derive(Debug, PartialEq, Clone)]
pub struct MapAST {
    pub arg: String,
    pub body: MapBody,
}

impl Default for MapAST {
    fn default() -> Self {
        Self {
            arg: String::from(""),
            body: MapBody::None,
        }
    }
}

/// Error represents jetro errors.
#[derive(Debug)]
pub enum Error {
    EmptyQuery,
    Parse(String),
    Eval(String),
    FuncEval(String),
}

/// FuncArg represents an individual argument
/// produced by the parser which gets passed
/// to module reponsible for dynamic functions
/// and gets evaluated during runtime.
#[allow(dead_code)]
#[derive(Debug, PartialEq, Clone)]
pub enum FuncArg {
    None,
    Key(String),
    Ord(Filter),
    SubExpr(Vec<Filter>),
    MapStmt(MapAST),
}

/// Func represents abstract structure of
/// a jetro function.
#[derive(Debug, PartialEq, Clone)]
pub struct Func {
    pub name: String,
    pub args: Vec<FuncArg>,
    pub alias: Option<String>,
    pub should_deref: bool,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Error::EmptyQuery => write!(f, "query is empty"),
            Error::Parse(ref msg) => write!(f, "parse: {}", &msg),
            Error::Eval(ref msg) => write!(f, "eval: {}", &msg),
            Error::FuncEval(ref msg) => write!(f, "func_eval: {}", &msg),
        }
    }
}

impl std::error::Error for Error {}

macro_rules! push_to_stack_or_produce {
    ($results:expr, $stack:expr, $tail:expr, $value:expr) => {
        let tlen = $tail.len();
        if tlen == 0 {
            $results.borrow_mut().push($value);
        } else {
            $stack
                .borrow_mut()
                .push(StackItem::new($value, $tail, $stack.clone()));
        }
    };
}

macro_rules! push_to_stack_iter_or_produce {
    ($results:expr, $stack:expr, $tail:expr, $obj:expr) => {
        let tlen = $tail.len();
        for (_k, v) in $obj {
            if tlen == 0 {
                $results.borrow_mut().push(v.clone());
            } else {
                $stack
                    .borrow_mut()
                    .push(StackItem::new(v.clone(), $tail, $stack.clone()));
            }
        }
    };
}

macro_rules! match_value {
    ($target_object:expr, $target_map:expr, $target_key:expr, $some_path:expr) => {{
        for item in Path::collect_with_filter($target_object.clone(), $some_path.as_slice())
            .0
            .iter()
        {
            match item.clone() {
                Value::Object(ref target_object) => {
                    for (k, v) in target_object {
                        $target_map
                            .as_object_mut()
                            .unwrap()
                            .insert(k.clone(), v.clone());
                    }
                }
                Value::String(ref some_str) => {
                    let map = $target_map.as_object_mut().unwrap();
                    let v = map.remove(&$target_key.clone());
                    if v.is_none() {
                        map.insert($target_key.clone(), Value::String(some_str.clone()));
                        continue;
                    }
                    match &v {
                        Some(Value::String(ref s)) => {
                            map.insert(
                                $target_key.clone(),
                                Value::Array(vec![
                                    Value::String(some_str.clone()),
                                    Value::String(s.clone()),
                                ]),
                            );
                        }
                        Some(Value::Array(ref array)) => {
                            let mut array = array.clone();
                            array.push(Value::String(some_str.clone()));
                            $target_map
                                .as_object_mut()
                                .unwrap()
                                .insert($target_key.clone(), Value::Array(array));
                        }
                        _ => {
                            map.insert($target_key.clone(), Value::String(some_str.clone()));
                        }
                    };
                }
                Value::Bool(ref some_bool) => {
                    $target_map
                        .as_object_mut()
                        .unwrap()
                        .insert($target_key.clone(), Value::Bool(some_bool.clone()));
                }
                Value::Number(ref some_number) => {
                    let map = $target_map.as_object_mut().unwrap();
                    let v = map.remove(&$target_key.clone());
                    if v.is_none() {
                        map.insert($target_key.clone(), Value::Number(some_number.clone()));
                        continue;
                    }
                    match &v {
                        Some(Value::Number(ref n)) => {
                            map.insert(
                                $target_key.clone(),
                                Value::Array(vec![
                                    Value::Number(serde_json::Number::from(some_number.clone())),
                                    Value::Number(serde_json::Number::from(n.clone())),
                                ]),
                            );
                        }
                        Some(Value::Array(ref array)) => {
                            let mut array = array.clone();
                            array
                                .push(Value::Number(serde_json::Number::from(some_number.clone())));
                            $target_map
                                .as_object_mut()
                                .unwrap()
                                .insert($target_key.clone(), Value::Array(array));
                        }
                        _ => {
                            map.insert($target_key.clone(), Value::Number(some_number.clone()));
                        }
                    };
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
    ($lhs:expr, $op:expr, $rhs:expr, String) => {
        match $op {
            FilterOp::Less => $lhs < $rhs,
            FilterOp::Eq => $lhs == $rhs,
            FilterOp::Gt => $lhs > $rhs,
            FilterOp::Lq => $lhs <= $rhs,
            FilterOp::Gq => $lhs >= $rhs,
            FilterOp::NotEq => $lhs != $rhs,
            FilterOp::Almost => $lhs.to_lowercase() == $rhs.to_lowercase(),
            _ => false,
        }
    };

    ($lhs:expr, $op:expr, $rhs:expr, ()) => {
        match $op {
            FilterOp::Less => $lhs < $rhs,
            FilterOp::Eq => $lhs == $rhs,
            FilterOp::Gt => $lhs > $rhs,
            FilterOp::Lq => $lhs <= $rhs,
            FilterOp::Gq => $lhs >= $rhs,
            FilterOp::NotEq => $lhs != $rhs,
            FilterOp::Almost => {
                todo!("implement 'almost' operator");
            }
            _ => false,
        }
    };
}

impl Func {
    pub fn new() -> Self {
        Self {
            name: String::new(),
            args: Vec::new(),
            alias: None,
            should_deref: false,
        }
    }
}

impl FilterInner {
    fn eval(&self, entry: &Value) -> bool {
        let obj = if let Some(output) = entry.as_object() {
            output
        } else {
            return false;
        };
        match self {
            FilterInner::Cond {
                ref left,
                ref op,
                ref right,
            } => match obj.get(left) {
                Some(result) => match result {
                    Value::String(ref target_string) => match right {
                        FilterInnerRighthand::String(ref str_value) => {
                            return do_comparision!(target_string, op, str_value, String);
                        }
                        _ => {}
                    },
                    Value::Bool(v) => match right {
                        FilterInnerRighthand::Bool(bool_value) => {
                            return do_comparision!(v, op, bool_value, ());
                        }
                        _ => {}
                    },
                    Value::Number(n) => match right {
                        FilterInnerRighthand::Number(number_value) => {
                            return do_comparision!(n.as_i64().unwrap(), op, *number_value, ());
                        }
                        FilterInnerRighthand::Float(float_value) => {
                            return do_comparision!(n.as_f64().unwrap(), op, *float_value, ());
                        }
                        _ => {}
                    },
                    _ => {}
                },
                _ => {}
            },
        };

        return false;
    }
}

impl FilterAST {
    pub fn new(left: FilterInner) -> Self {
        Self {
            operator: FilterLogicalOp::None,
            left: Some(left),
            right: None,
        }
    }

    pub fn set_operator(&mut self, operator: FilterLogicalOp) {
        self.operator = operator;
    }

    pub fn link_right(&mut self, statement: FilterInner, operator: FilterLogicalOp) -> Self {
        let mut rhs: Self = Self::new(statement);
        rhs.set_operator(operator);

        let inner = rhs;
        let output = inner.clone();
        self.right = Some(Rc::new(inner));

        output
    }

    pub fn eval(&self, value: &Value) -> bool {
        if self.operator == FilterLogicalOp::None && self.right.is_none() {
            return self.left.clone().unwrap().eval(&value);
        }

        let lhs = self.left.clone().unwrap().eval(&value);
        let rhs = self.right.clone().unwrap().eval(&value);

        match self.operator {
            FilterLogicalOp::And => return lhs && rhs,
            FilterLogicalOp::Or => return lhs || rhs,
            _ => todo!("inconsistent state in filter comp"),
        };
    }

    #[allow(dead_code)]
    pub fn repr(&self) -> String {
        if self.operator == FilterLogicalOp::None && self.right.is_none() {
            return format!("Filter({:?})", &self.left);
        }
        let rhs = &self.right.clone().unwrap().clone();
        return format!(
            "Filter({:?}) {:?} {:?}",
            &self.left,
            &self.operator,
            rhs.repr(),
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
                                    &ctx.unwrap().root
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
                                    &ctx.unwrap().root
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
            Filter::DescendantChild(entry) => match entry {
                FilterDescendant::Single(ref descendant) => {
                    return Some(descendant[2..].to_string());
                }
                _ => {
                    todo!("pair descendant not implemented");
                }
            },
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
        match ctx.collect() {
            Ok(_) => {
                let x = ctx.results.take().to_owned();
                Ok(PathResult(x))
            }
            Err(err) => Err(err),
        }
    }

    pub(crate) fn collect_with_filter(v: Value, filters: &[Filter]) -> PathResult {
        // TODO(): handle errors similar to collect method
        let mut ctx = Context::new(v, filters);
        let _ = ctx.collect();

        let x = ctx.results.take().to_owned();

        PathResult(x)
    }
}

impl PathResult {
    pub fn from_index<T: serde::de::DeserializeOwned>(&mut self, index: usize) -> Option<T> {
        let final_value: Value = self.0.remove(index);

        match serde_json::from_value(final_value.clone().take()) {
            Ok(result) => Some(result),
            _ => None,
        }
    }
}

impl<'a> StackItem<'a> {
    pub fn new(
        value: Value,
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
        let results: Rc<RefCell<Vec<Value>>> = Rc::new(RefCell::new(Vec::new()));
        let stack: Rc<RefCell<Vec<StackItem<'a>>>> = Rc::new(RefCell::new(Vec::new()));
        let step_results: Rc<RefCell<Vec<Value>>> = Rc::new(RefCell::new(Vec::new()));
        let registry: Rc<RefCell<dyn crate::func::Registry>> = crate::func::default_registry();
        stack
            .borrow_mut()
            .push(StackItem::new(value.clone(), filters, Rc::clone(&stack)));

        Self {
            root: value.clone(),
            stack,
            registry,
            results,
            step_results,
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
            match value {
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
            match value {
                Value::Array(ref inner_array) => {
                    for v in inner_array {
                        if v.is_number() {
                            sum.add(&v);
                        }
                    }
                }
                Value::Number(_) => {
                    sum.add(&value);
                }
                _ => {}
            }
        }

        return sum;
    }

    pub fn collect(&mut self) -> Result<(), Error> {
        // TODO(mitghi): implement context errors

        while self.stack.borrow().len() > 0 {
            let current = if let Some(value) = self.stack.borrow_mut().pop() {
                value
            } else {
                return Ok(());
            };
            match current.step() {
                Some(step) => match step {
                    (Filter::Root, Some(tail)) => self.stack.borrow_mut().push(StackItem::new(
                        current.value,
                        tail,
                        self.stack.clone(),
                    )),

                    (Filter::Function(ref func), Some(tail)) => {
                        let registry = self.registry.clone();
                        match registry
                            .borrow_mut()
                            .call(&func, &current.value, Some(self))
                        {
                            Ok(result) => {
                                push_to_stack_or_produce!(
                                    self.results,
                                    self.stack,
                                    tail,
                                    result.clone()
                                );
                            }
                            Err(err) => {
                                return Err(Error::FuncEval(err.to_string()));
                            }
                        };
                    }

                    (Filter::GroupedChild(ref vec), Some(tail)) => match &current.value {
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

                    (Filter::MultiFilter(ref ast), Some(tail)) => match &current.value {
                        Value::Object(ref _obj) => {
                            todo!("implement filter support for object");
                        }
                        Value::Array(ref array) => {
                            let mut results: Vec<Value> = vec![];
                            for value in array {
                                if ast.eval(&value) {
                                    results.push(value.clone());
                                };
                            }
                            if results.len() > 0 {
                                push_to_stack_or_produce!(
                                    self.results,
                                    self.stack,
                                    tail,
                                    Value::Array(results)
                                );
                            }
                        }
                        _ => {
                            todo!("handle unmatched arm of experimental filter");
                        }
                    },
                    (Filter::Filter(ref cond), Some(tail)) => match &current.value {
                        Value::Object(ref _obj) => {
                            todo!("implement filter support for object");
                        }
                        Value::Array(ref array) => {
                            let mut results: Vec<Value> = vec![];
                            for value in array {
                                if cond.eval(&value) {
                                    results.push(value.clone());
                                }
                            }
                            if results.len() > 0 {
                                push_to_stack_or_produce!(
                                    self.results,
                                    self.stack,
                                    tail,
                                    Value::Array(results)
                                );
                            }
                        }
                        _ => {
                            todo!("implement handling of unmatched arm for filter");
                        }
                    },

                    (Filter::ArrayIndex(ref index), Some(tail)) => match &current.value {
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

                    (Filter::Slice(ref from, ref to), Some(tail)) => match &current.value {
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

                    (Filter::ArrayTo(ref index), Some(tail)) => match &current.value {
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

                    (Filter::ArrayFrom(ref index), Some(tail)) => match &current.value {
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

                    (Filter::Pick(_), Some(tail)) => match &current.value {
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

                    (Filter::AnyChild, Some(tail)) => match &current.value {
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

                    (Filter::Child(ref name), Some(tail)) => match &current.value {
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

                    (Filter::DescendantChild(ref entry), Some(tail)) => match entry {
                        FilterDescendant::Single(ref descendant) => match &current.value {
                            Value::Object(ref obj) => {
                                for (k, v) in obj {
                                    let filters: &[Filter] = if k == descendant {
                                        tail
                                    } else {
                                        current.filters
                                    };

                                    if filters.len() == 0 {
                                        self.results.borrow_mut().push(v.clone());
                                    } else {
                                        self.stack.borrow_mut().push(StackItem::new(
                                            v.clone(),
                                            filters,
                                            self.stack.clone(),
                                        ));
                                    }
                                }
                            }
                            Value::Array(ref array) => {
                                for v in array {
                                    self.stack.borrow_mut().push(StackItem::new(
                                        v.clone(),
                                        current.filters,
                                        self.stack.clone(),
                                    ));
                                }
                            }
                            _ => {}
                        },
                        FilterDescendant::Pair(ref descendant, ref target_value) => {
                            match &current.value {
                                Value::Object(ref obj) => {
                                    let mut found_match = false;
                                    for (k, v) in obj {
                                        let filters: &[Filter] = if k == descendant {
                                            match &v {
                                                Value::String(ref current_value) => {
                                                    if *current_value == *target_value {
                                                        found_match = true;
                                                        tail
                                                    } else {
                                                        current.filters
                                                    }
                                                }
                                                _ => current.filters,
                                            }
                                        } else {
                                            current.filters
                                        };

                                        if filters.len() == 0 && found_match {
                                            self.step_results
                                                .borrow_mut()
                                                .push(serde_json::Value::Object(obj.clone()));
                                        }
                                        self.stack.borrow_mut().push(StackItem::new(
                                            v.clone(),
                                            filters,
                                            self.stack.clone(),
                                        ));
                                    }
                                }
                                Value::Array(ref array) => {
                                    for v in array {
                                        self.stack.borrow_mut().push(StackItem::new(
                                            v.clone(),
                                            current.filters,
                                            self.stack.clone(),
                                        ));
                                    }
                                }
                                _ => {}
                            }
                        }
                    },

                    _ => {}
                },
                _ => {}
            }
        }

        // no more filter to process
        // push intermediate results
        // such as from recursive search
        // with pair (key, value) into
        // final results
        {
            let sr = self.step_results.borrow();
            if sr.len() > 0 {
                let mut output: Vec<Value> = Vec::new();
                for v in self.step_results.borrow().iter() {
                    output.insert(0, v.clone());
                }
                self.results.borrow_mut().push(Value::Array(output).into());
            }
        }

        Ok(())
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
            Filter::DescendantChild(FilterDescendant::Single("within".to_string())),
        ];

        let mut ctx = Context::new(v, &filters);
        let _ = ctx.collect();

        assert_eq!(ctx.results.take(), vec![Value::String("value".to_string())]);
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
            values.0,
            vec![serde_json::json! {{"a": "object_a", "object": "final_value"}}]
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
            values.0,
            vec![serde_json::json! {{"descendant": "final_value", "foo": "object_a"}}]
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

        assert_eq!(values.0, vec![serde_json::json!(2)],);
    }

    #[test]
    fn test_grouped_child() {
        let data = serde_json::json!({"entry": {"some": "value","foo": null, "another": "word", "till": "deal"}});

        let values =
            Path::collect(data, ">/entry/('foo' | 'bar' | 'another')").expect("unable to parse");

        assert_eq!(values.0.len(), 1);
        assert_eq!(values.0[0], Value::String(String::from("word")));
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

        assert_eq!(values.0.len(), 1);
        assert_eq!(
            values.0[0],
            Value::Array(vec![
                serde_json::json!({"name": "foo", "is_eligable": true})
            ])
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
            values.unwrap().0,
            vec![Value::Number(serde_json::Number::from(2))]
        );
    }

    #[test]
    fn test_filter_number() {
        let data = serde_json::json!({"entry": {"values": [{"name": "gearbox", "priority": 10}, {"name": "steam", "priority": 2}]}});
        let values = Path::collect(data, ">/entry/values/#filter('priority' == 2)");
        assert_eq!(values.is_ok(), true);
        assert_eq!(
            values.unwrap().0,
            vec![Value::Array(vec![
                serde_json::json!({"name": "steam", "priority": 2})
            ])]
        );
    }

    #[test]
    fn test_truth() {
        let data = serde_json::json!({"entry": {"values": [{"name": "gearbox", "priority": 10, "truth_a": true, "truth_b": false, "truth_c": false, "truth_d": true}, {"name": "steam", "priority": 2, "truth_a": false, "truth_b": true, "truth_c": false, "truth_d": true}]}});
        let tests: Vec<(String, Vec<Value>)> = vec![
            (">/..truth_a/#all".to_string(), vec![Value::Bool(false)]),
            (">/..truth_b/#all".to_string(), vec![Value::Bool(false)]),
            (">/..truth_c/#all".to_string(), vec![Value::Bool(false)]),
            (">/..truth_d/#all".to_string(), vec![Value::Bool(true)]),
        ];

        for (path, expect) in tests.iter() {
            let values = Path::collect(data.clone(), path).unwrap();
            assert_eq!(values.0, *expect);
        }
    }

    #[test]
    fn test_float_filter_evaluation() {
        let data = serde_json::json!({"entry": {"values": [{"name": "gearbox", "some_float": 10.1112}, {"name": "steam", "some_float": 2.48}]}});
        let tests: Vec<(String, Vec<Value>)> = vec![
            (
                ">/entry/values/#filter('some_float' >= 10.1112)".to_string(),
                vec![Value::Array(vec![
                    serde_json::json!({"name": "gearbox", "some_float": 10.1112}),
                ])],
            ),
            (
                ">/entry/values/#filter('some_float' == 2.48)".to_string(),
                vec![Value::Array(vec![
                    serde_json::json!({"name": "steam", "some_float": 2.48}),
                ])],
            ),
        ];

        for (path, expect) in tests.iter() {
            let values = Path::collect(data.clone(), path).unwrap();
            assert_eq!(values.0, *expect);
        }
    }

    #[test]
    fn test_func() {
        let data =
            serde_json::json!({"entry": {"values": [{"name": "gearbox"}, {"name": "steam"}]}});
        let result = Path::collect(data, ">/entry/values/#reverse()").unwrap();
        assert_eq!(
            result.0,
            vec![Value::Array(vec![
                serde_json::json!({"name": "steam"}),
                serde_json::json!({"name": "gearbox"})
            ])]
        );
    }

    #[test]
    fn test_pick_and_zip() {
        let data = serde_json::json!({"a": [{"name": "tool", "value": {"nested": "field"}}, {"name": "pneuma", "value": {"nested": "seal"}}], "b": [2000, 2100]});
        let result = Path::collect(
            data,
            ">/#pick(>/..name as 'name', >/..nested as 'field', >/..b as 'release')/#zip",
        )
        .unwrap();
        assert_eq!(
            result.0,
            vec![Value::Array(vec![
                serde_json::json!({"field": "field", "name": "tool", "release": 2000}),
                serde_json::json!({"field": "seal", "name": "pneuma", "release": 2100})
            ])]
        );
    }

    #[test]
    fn test_map_on_path() {
        let data =
            serde_json::json!({"entry": {"values": [{"name": "gearbox"}, {"name": "steam"}]}});
        let result = Path::collect(data, ">/..values/#map(x: x.name)").unwrap();
        assert_eq!(
            result.0,
            vec![Value::Array(vec![
                Value::String("gearbox".to_string()),
                Value::String("steam".to_string()),
            ])]
        );
    }

    #[test]
    fn test_descendant_keyed() {
        let data = serde_json::json!({"entry": {"values": [{"name": "gearbox"}, {"name": "gearbox", "test": "2000"}]}});
        let result = Path::collect(data, ">/..('name'='gearbox')").unwrap();
        assert_eq!(
            result.0,
            vec![Value::Array(vec![
                serde_json::json!({"name": "gearbox"}),
                serde_json::json!({"name": "gearbox", "test": "2000"})
            ])]
        );
    }
}
