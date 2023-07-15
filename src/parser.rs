//! Module containing parser for jetro.

use std::cell::RefCell;
use std::rc::Rc;

use pest::iterators::Pair;

use crate::context::{
    Filter, FilterAST, FilterDescendant, FilterInner, FilterInnerRighthand, FilterLogicalOp,
    FilterOp, Func, FuncArg, MapAST, MapBody, PickFilterInner,
};

use crate::*;

#[derive(Parse)]
#[grammar = "./grammar.pest"]
pub struct ExpressionParser;

/// parse parses the jetro expression, builds the filter along with
/// abstract syntax tree. It returns an error containing details about
/// cause of parser failure.
pub(crate) fn parse<'a>(input: &'a str) -> Result<Vec<Filter>, pest::error::Error<Rule>> {
    let pairs = ExpressionParser::parse(Rule::expression, input);
    if pairs.is_err() {
        return Err(pairs.err().unwrap());
    }
    let root = pairs.unwrap().next().unwrap();
    let mut actions: Vec<Filter> = vec![];
    for token in root.into_inner() {
        match token.as_rule() {
            Rule::r#fn => {
                let inner: Pair<Rule> = token.clone().into_inner().nth(1).unwrap();
                let mut func = Func::new();
                let mut should_deref = false;

                func.name
                    .insert_str(0, inner.clone().into_inner().next().unwrap().as_str());

                for value in inner.into_inner() {
                    match &value.as_rule() {
                        Rule::ident => {}
                        Rule::fnLit => {
                            let literal = &value
                                .clone()
                                .into_inner()
                                .next()
                                .unwrap()
                                .into_inner()
                                .as_str();
                            func.args.push(FuncArg::Key(literal.to_string()));
                        }
                        Rule::fnExpr => {
                            let expr = parse(
                                &value
                                    .clone()
                                    .into_inner()
                                    .next()
                                    .unwrap()
                                    .into_inner()
                                    .as_str(),
                            )?;
                            func.args.push(FuncArg::SubExpr(expr));
                        }
                        Rule::filterStmtCollection => {
                            todo!("handle filter statements");
                        }
                        Rule::arrow => {
                            func.alias = Some(
                                value
                                    .clone()
                                    .into_inner()
                                    .next()
                                    .unwrap()
                                    .into_inner()
                                    .as_str()
                                    .to_string(),
                            );
                        }
                        Rule::arrowDeref => {
                            func.alias = Some(
                                value
                                    .clone()
                                    .into_inner()
                                    .next()
                                    .unwrap()
                                    .into_inner()
                                    .as_str()
                                    .to_string(),
                            );
                            func.should_deref = true;
                            should_deref = true;
                        }
                        Rule::mapStmt => {
                            let mut ast = MapAST::default();
                            ast.arg = value
                                .clone()
                                .into_inner()
                                .nth(0)
                                .unwrap()
                                .as_str()
                                .to_string();
                            let stmt = value.clone().into_inner().nth(1).unwrap().into_inner();
                            for value in stmt {
                                match &value.as_rule() {
                                    Rule::ident => match &mut ast.body {
                                        MapBody::None => {
                                            let v = value.as_str();
                                            if v != ast.arg {
                                                todo!("handle wrong map arg");
                                            }
                                            ast.body = MapBody::Subpath(vec![Filter::Root]);
                                        }
                                        MapBody::Subpath(ref mut subpath) => {
                                            let v = value.as_str();
                                            subpath.push(Filter::Child(v.to_string()));
                                        }
                                        _ => {
                                            todo!("handle unmatched arm");
                                        }
                                    },
                                    Rule::methodCall => {
                                        let v = &value.clone().into_inner().as_str();
                                        match &ast.body {
                                            MapBody::Subpath(subpath) => {
                                                ast.body = MapBody::Method {
                                                    name: v.to_string(),
                                                    subpath: subpath.to_vec(),
                                                };
                                            }
                                            _ => {
                                                todo!("handle invalid type");
                                            }
                                        }
                                    }
                                    _ => {
                                        todo!("in unmatched arm");
                                    }
                                }
                            }
                            func.args.push(FuncArg::MapStmt(ast));
                        }
                        _ => {
                            todo!("handle unmatched arm of function generalization",);
                        }
                    }
                }

                let alias = if func.alias.is_some() {
                    func.alias.clone()
                } else {
                    None
                };
                actions.push(Filter::Function(func));
                if should_deref {
                    alias.is_some().then(|| {
                        let alias = alias.unwrap();
                        actions.push(Filter::Pick(vec![PickFilterInner::KeyedStr {
                            key: alias.clone(),
                            alias: alias.clone(),
                        }]));
                    });
                }
            }
            Rule::path | Rule::reverse_path => actions.push(Filter::Root),
            Rule::grouped_any => {
                let elem = token.into_inner().nth(1).unwrap().into_inner();
                let mut values: Vec<String> = vec![];
                for e in elem {
                    values.push(e.into_inner().as_str().to_string());
                }

                actions.push(Filter::GroupedChild(values));
            }
            Rule::filterFn => {
                let rule = token.into_inner().nth(1).unwrap().into_inner().into_iter();

                let mut fast_root: Option<Rc<RefCell<FilterAST>>> = None;
                let mut current: Option<Rc<RefCell<FilterAST>>> = None;

                for item in rule.into_iter() {
                    let rule = item.as_rule();
                    let elem = item.clone();
                    match rule {
                        Rule::filterStmt => {
                            let mut inner = elem.into_inner();
                            let left = inner
                                .nth(0)
                                .unwrap()
                                .into_inner()
                                .next()
                                .unwrap()
                                .into_inner()
                                .next()
                                .unwrap()
                                .as_str()
                                .to_string();
                            let op =
                                FilterOp::get(inner.nth(0).unwrap().into_inner().as_str()).unwrap();
                            let right: Option<FilterInnerRighthand>;

                            match inner.clone().into_iter().next().unwrap().as_rule() {
                                Rule::literal => {
                                    right = Some(FilterInnerRighthand::String(
                                        inner.next().unwrap().into_inner().as_str().to_string(),
                                    ));
                                }
                                Rule::truthy => {
                                    match inner
                                        .next()
                                        .unwrap()
                                        .into_inner()
                                        .next()
                                        .unwrap()
                                        .as_rule()
                                    {
                                        Rule::true_bool => {
                                            right = Some(FilterInnerRighthand::Bool(true));
                                        }
                                        Rule::false_bool => {
                                            right = Some(FilterInnerRighthand::Bool(false));
                                        }
                                        _ => {
                                            todo!("handle error");
                                        }
                                    }
                                }
                                Rule::number => {
                                    right = Some(FilterInnerRighthand::Number(
                                        inner.next().unwrap().as_str().parse::<i64>().unwrap(),
                                    ));
                                }
                                Rule::float => {
                                    right = Some(FilterInnerRighthand::Float(
                                        inner.next().unwrap().as_str().parse::<f64>().unwrap(),
                                    ));
                                }
                                _ => {
                                    right = None;
                                }
                            }

                            let filter_inner = FilterInner::Cond {
                                left,
                                op,
                                right: right.unwrap(),
                            };

                            if fast_root.is_none() {
                                fast_root = Some(Rc::new(RefCell::new(FilterAST::new(
                                    filter_inner.clone(),
                                ))));
                                current = fast_root.clone();
                            } else {
                                let node = current.clone().unwrap();
                                let rhs = node
                                    .borrow_mut()
                                    .link_right(filter_inner, FilterLogicalOp::None);
                                current = Some(rhs);
                            }
                        }
                        Rule::logical_cmp => {
                            let op = FilterLogicalOp::get(elem.as_str()).unwrap();
                            let node = current.clone().unwrap();
                            node.borrow_mut().set_operator(op.clone());
                        }
                        _ => {
                            todo!("implement unmatched arm");
                        }
                    }
                }
                actions.push(Filter::MultiFilter(fast_root.unwrap()));
            }
            Rule::any_child => actions.push(Filter::AnyChild),
            Rule::array_index => {
                let elem = token.into_inner().nth(1).unwrap().as_span();
                let index: usize = elem.as_str().parse::<usize>().unwrap();
                actions.push(Filter::ArrayIndex(index));
            }
            Rule::array_from => {
                let elem = token.into_inner().nth(1).unwrap().as_span();
                let index: usize = elem.as_str().parse::<usize>().unwrap();
                actions.push(Filter::ArrayFrom(index));
            }
            Rule::array_to => {
                let elem = token.into_inner().nth(1).unwrap().as_span();
                let index: usize = elem.as_str().parse::<usize>().unwrap();
                actions.push(Filter::ArrayTo(index));
            }
            Rule::slice => {
                let mut it = token.into_inner();
                it.next();
                let from: usize = {
                    let value = it.next().unwrap().as_span();
                    value.as_str().parse::<usize>().unwrap()
                };
                let to: usize = {
                    let value = it.next().unwrap().as_span();
                    value.as_str().parse::<usize>().unwrap()
                };
                actions.push(Filter::Slice(from, to));
            }
            Rule::pickFn => {
                let mut elems: Vec<PickFilterInner> = vec![];
                for v in token.into_inner().nth(1).unwrap().into_inner() {
                    match &v.as_rule() {
                        Rule::sub_expression_keyed | Rule::sub_expression_keyed_reversed => {
                            let reverse = *&v.as_rule() == Rule::sub_expression_keyed_reversed;
                            let mut l = v.into_inner();
                            let subexpr = l.next().unwrap().as_str();
                            let alias: Option<String> = match l.next() {
                                Some(ref result) => {
                                    let result = result
                                        .clone()
                                        .into_inner()
                                        .next()
                                        .unwrap()
                                        .into_inner()
                                        .as_str()
                                        .to_string();
                                    Some(result)
                                }
                                _ => None,
                            };

                            match alias {
                                Some(alias) => {
                                    let subpath = parse(subexpr);
                                    if subpath.is_err() {
                                        return Err(subpath.err().unwrap());
                                    }
                                    elems.push(PickFilterInner::KeyedSubpath {
                                        subpath: subpath.unwrap(),
                                        alias,
                                        reverse,
                                    });
                                }
                                None => {
                                    let subpath = parse(subexpr);
                                    if subpath.is_err() {
                                        return Err(subpath.err().unwrap());
                                    }
                                    elems.push(PickFilterInner::Subpath(subpath.unwrap(), reverse));
                                }
                            }
                        }
                        Rule::sub_expression => {
                            let subpath = parse(v.as_str());
                            if subpath.is_err() {
                                return Err(subpath.err().unwrap());
                            }
                            elems.push(PickFilterInner::Subpath(subpath.unwrap(), false));
                        }
                        Rule::sub_expression_reversed => {
                            let subpath = parse(v.as_str());
                            if subpath.is_err() {
                                return Err(subpath.err().unwrap());
                            }
                            elems.push(PickFilterInner::Subpath(subpath.unwrap(), true));
                        }
                        Rule::literal_keyed => {
                            let mut l = v.into_inner();
                            let span = l.next().unwrap().as_span().as_str().to_string();

                            let alias: Option<String> = match l.next() {
                                Some(ref result) => {
                                    let mut result = result.as_span().as_str()[4..].to_string();
                                    if result.len() > 2 {
                                        result = result[1..result.len() - 1].to_string();
                                    }
                                    Some(result)
                                }
                                _ => None,
                            };

                            match (span.len(), alias) {
                                (2, _) => elems.push(PickFilterInner::Str("".to_string())),
                                (_, None) => {
                                    elems.push(PickFilterInner::Str(
                                        span[1..span.len() - 1].to_string(),
                                    ));
                                }
                                (_, Some(alias)) => {
                                    elems.push(PickFilterInner::KeyedStr {
                                        key: span[1..span.len() - 1].to_string(),
                                        alias: alias.to_string(),
                                    });
                                }
                            }
                        }
                        _ => {
                            elems.push(PickFilterInner::None);
                        }
                    }
                }
                actions.push(Filter::Pick(elems));
            }
            Rule::descendant_child => {
                let entry = token.into_inner().nth(1).unwrap();
                let node = entry.into_inner().next().unwrap();
                let rule = node.as_rule();
                match &rule {
                    Rule::ident => {
                        let ident = node.as_str().to_owned();
                        actions.push(Filter::DescendantChild(FilterDescendant::Single(ident)));
                    }
                    Rule::keyed_ident => {
                        let mut node = node.into_inner();
                        let mut lnode = node.next().unwrap().into_inner();
                        let mut rnode = node.next().unwrap().into_inner();
                        let lstr = lnode.next().unwrap().as_str().to_owned();
                        let rstr = rnode.next().unwrap().as_str().to_owned();
                        actions.push(Filter::DescendantChild(FilterDescendant::Pair(lstr, rstr)));
                    }
                    _ => {
                        panic!("unreachable")
                    }
                }
            }
            Rule::child => {
                let ident = token.into_inner().nth(1).unwrap().as_str().to_owned();
                actions.push(Filter::Child(ident));
            }
            _ => {}
        }
    }
    Ok(actions)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_simple_path() {
        let actions = parse(">/obj/some/*/name").unwrap();
        assert_eq!(
            actions,
            vec![
                Filter::Root,
                Filter::Child("obj".to_string()),
                Filter::Child("some".to_string()),
                Filter::AnyChild,
                Filter::Child("name".to_string()),
            ]
        );
    }

    #[test]
    fn test_simple_path_with_recursive_descendant() {
        let actions = parse(">/obj/some/..descendant/name").unwrap();
        assert_eq!(
            actions,
            vec![
                Filter::Root,
                Filter::Child("obj".to_string()),
                Filter::Child("some".to_string()),
                Filter::DescendantChild(FilterDescendant::Single("descendant".to_string())),
                Filter::Child("name".to_string()),
            ]
        );
    }

    #[test]
    fn test_recursive_descendant_keyed() {
        let actions = parse(">/..('key' = 'value')").unwrap();
        assert_eq!(
            actions,
            vec![
                Filter::Root,
                Filter::DescendantChild(FilterDescendant::Pair(
                    "key".to_string(),
                    "value".to_string()
                )),
            ]
        );
    }

    #[test]
    fn test_simple_path_with_recursive_descendant_and_pick() {
        let actions = parse(">/obj/some/..descendant/#pick('a', 'b', 'c', 'd')").unwrap();
        assert_eq!(
            actions,
            vec![
                Filter::Root,
                Filter::Child("obj".to_string()),
                Filter::Child("some".to_string()),
                Filter::DescendantChild(FilterDescendant::Single("descendant".to_string())),
                Filter::Pick(vec![
                    PickFilterInner::Str("a".to_string()),
                    PickFilterInner::Str("b".to_string()),
                    PickFilterInner::Str("c".to_string()),
                    PickFilterInner::Str("d".to_string())
                ]),
            ]
        );
    }

    #[test]
    fn test_with_keyed_literal() {
        let actions = parse(">/obj/some/..descendant/#pick('f' as 'foo', 'b' as 'bar')").unwrap();
        assert_eq!(
            actions,
            vec![
                Filter::Root,
                Filter::Child("obj".to_string()),
                Filter::Child("some".to_string()),
                Filter::DescendantChild(FilterDescendant::Single("descendant".to_string())),
                Filter::Pick(vec![
                    PickFilterInner::KeyedStr {
                        key: "f".to_string(),
                        alias: "foo".to_string()
                    },
                    PickFilterInner::KeyedStr {
                        key: "b".to_string(),
                        alias: "bar".to_string()
                    },
                ]),
            ]
        );
    }

    #[test]
    fn test_with_sub_expression_keyed() {
        let actions =
            parse(">/obj/some/..descendant/#pick('f' as 'foo', >/some/branch as 'path')").unwrap();
        assert_eq!(
            actions,
            vec![
                Filter::Root,
                Filter::Child("obj".to_string()),
                Filter::Child("some".to_string()),
                Filter::DescendantChild(FilterDescendant::Single("descendant".to_string())),
                Filter::Pick(vec![
                    PickFilterInner::KeyedStr {
                        key: "f".to_string(),
                        alias: "foo".to_string()
                    },
                    PickFilterInner::KeyedSubpath {
                        subpath: vec![
                            Filter::Root,
                            Filter::Child("some".to_string()),
                            Filter::Child("branch".to_string()),
                        ],
                        alias: "path".to_string(),
                        reverse: false,
                    },
                ]),
            ]
        );
    }

    #[test]
    fn test_with_sub_expression_keyed_reverse() {
        let actions =
            parse(">/obj/some/..descendant/#pick('f' as 'foo', </some/branch as 'path')").unwrap();
        assert_eq!(
            actions,
            vec![
                Filter::Root,
                Filter::Child("obj".to_string()),
                Filter::Child("some".to_string()),
                Filter::DescendantChild(FilterDescendant::Single("descendant".to_string())),
                Filter::Pick(vec![
                    PickFilterInner::KeyedStr {
                        key: "f".to_string(),
                        alias: "foo".to_string()
                    },
                    PickFilterInner::KeyedSubpath {
                        subpath: vec![
                            Filter::Root,
                            Filter::Child("some".to_string()),
                            Filter::Child("branch".to_string()),
                        ],
                        alias: "path".to_string(),
                        reverse: true,
                    },
                ]),
            ]
        );
    }

    #[test]
    fn test_slice() {
        let actions = parse(">/[1:4]").unwrap();
        assert_eq!(actions, vec![Filter::Root, Filter::Slice(1, 4),]);
    }

    #[test]
    fn test_format() {
        let actions = parse(">/#formats('{}{}', 'name', 'alias') -> 'some_key'").unwrap();
        assert_eq!(
            actions,
            vec![
                Filter::Root,
                Filter::Function(Func {
                    name: "formats".to_string(),
                    args: vec![
                        FuncArg::Key("{}{}".to_string()),
                        FuncArg::Key("name".to_string()),
                        FuncArg::Key("alias".to_string()),
                    ],
                    alias: Some("some_key".to_string()),
                    should_deref: false,
                }),
            ]
        );
    }

    #[test]
    fn test_filter() {
        let actions = parse(">/..meows/#filter('some' == 'value')").unwrap();
        assert_eq!(
            actions,
            vec![
                Filter::Root,
                Filter::DescendantChild(FilterDescendant::Single("meows".to_string())),
                Filter::MultiFilter(Rc::new(RefCell::new(FilterAST {
                    operator: FilterLogicalOp::None,
                    left: Some(Rc::new(RefCell::new(FilterInner::Cond {
                        left: "some".to_string(),
                        op: FilterOp::Eq,
                        right: FilterInnerRighthand::String("value".to_string()),
                    }))),
                    right: None,
                })))
            ]
        );
    }

    #[test]
    fn test_groupped_literal() {
        let actions = parse(">/foo/('bar' | 'baz' | 'cuz')").unwrap();
        assert_eq!(
            actions,
            vec![
                Filter::Root,
                Filter::Child("foo".to_string()),
                Filter::GroupedChild(vec![
                    "bar".to_string(),
                    "baz".to_string(),
                    "cuz".to_string()
                ]),
            ],
        );
    }

    #[test]
    fn test_truthy_filter() {
        let actions = parse(">/foo/#filter('is_furry' == true)").unwrap();
        assert_eq!(
            actions,
            vec![
                Filter::Root,
                Filter::Child("foo".to_string()),
                Filter::MultiFilter(Rc::new(RefCell::new(FilterAST {
                    operator: FilterLogicalOp::None,
                    left: Some(Rc::new(RefCell::new(FilterInner::Cond {
                        left: "is_furry".to_string(),
                        op: FilterOp::Eq,
                        right: FilterInnerRighthand::Bool(true)
                    }))),
                    right: None,
                })))
            ]
        );
    }
    #[test]
    fn test_number_filter() {
        let actions = parse(">/foo/#filter('some_value' == 1234)").unwrap();
        assert_eq!(
            actions,
            vec![
                Filter::Root,
                Filter::Child("foo".to_string()),
                Filter::MultiFilter(Rc::new(RefCell::new(FilterAST {
                    operator: FilterLogicalOp::None,
                    left: Some(Rc::new(RefCell::new(FilterInner::Cond {
                        left: "some_value".to_string(),
                        op: FilterOp::Eq,
                        right: FilterInnerRighthand::Number(1234),
                    }))),
                    right: None,
                })))
            ],
        );
    }

    #[test]
    fn test_flaot_filter() {
        let actions = parse(">/foo/#filter('some_value' == 2.48)").unwrap();
        assert_eq!(
            actions,
            vec![
                Filter::Root,
                Filter::Child("foo".to_string()),
                Filter::MultiFilter(Rc::new(RefCell::new(FilterAST {
                    operator: FilterLogicalOp::None,
                    left: Some(Rc::new(RefCell::new(FilterInner::Cond {
                        left: "some_value".to_string(),
                        op: FilterOp::Eq,
                        right: FilterInnerRighthand::Float(2.48),
                    }))),
                    right: None,
                })))
            ],
        );
    }

    #[test]
    fn test_filter_ast() {
        type FilterNode = Rc<RefCell<FilterAST>>;
        let root: FilterNode;
        let mut current: FilterNode;

        let lhs = FilterInner::Cond {
            left: "some_key".to_string(),
            op: FilterOp::Eq,
            right: FilterInnerRighthand::String("some_value".to_string()),
        };

        let op = FilterLogicalOp::And;

        let rhs = FilterInner::Cond {
            left: "whiskers".to_string(),
            op: FilterOp::Gt,
            right: FilterInnerRighthand::Float(2.48),
        };

        let standalone_lhs = FilterInner::Cond {
            left: "standalone".to_string(),
            op: FilterOp::Eq,
            right: FilterInnerRighthand::Bool(true),
        };

        root = Rc::new(RefCell::new(FilterAST::new(lhs)));
        root.borrow_mut().set_operator(op.clone());

        current = root.clone();

        let new_node = current.borrow_mut().link_right(rhs, FilterLogicalOp::None);
        current = new_node;

        current.borrow_mut().set_operator(FilterLogicalOp::Or);

        let new_node = current
            .borrow_mut()
            .link_right(standalone_lhs, FilterLogicalOp::None);
        current = new_node;
        _ = current;

        assert_eq!(root.borrow().operator, FilterLogicalOp::And);

        let lhs = &root.borrow().left.clone().unwrap().clone();
        match *lhs.borrow() {
            FilterInner::Cond {
                left: _,
                ref op,
                right: _,
            } => {
                assert_eq!(*op, FilterOp::Eq);
            }
        }

        let rhs_lhs = &root
            .borrow()
            .right
            .clone()
            .unwrap()
            .clone()
            .borrow()
            .left
            .clone()
            .unwrap()
            .clone();
        match *rhs_lhs.borrow() {
            FilterInner::Cond {
                left: _,
                ref op,
                right: _,
            } => {
                assert_eq!(*op, FilterOp::Gt);
            }
        };
    }

    #[test]
    fn formats_without_deref() {
        let actions = parse(">/foo/#formats('Msg {}', 'msg') -> 'message'").unwrap();
        assert_eq!(
            actions,
            vec![
                Filter::Root,
                Filter::Child("foo".to_string()),
                Filter::Function(Func {
                    name: "formats".to_string(),
                    args: vec![
                        FuncArg::Key("Msg {}".to_string()),
                        FuncArg::Key("msg".to_string()),
                    ],
                    alias: Some("message".to_string()),
                    should_deref: false,
                }),
            ]
        );
    }

    #[test]
    fn pick_formats_deref() {
        let actions =
            parse(">/#pick('foo', >/bar/#formats('Msg {}', 'msg') ->* 'message')").unwrap();
        assert_eq!(
            actions,
            vec![
                Filter::Root,
                Filter::Pick(vec![
                    PickFilterInner::Str("foo".to_string()),
                    PickFilterInner::Subpath(
                        vec![
                            Filter::Root,
                            Filter::Child("bar".to_string()),
                            Filter::Function(Func {
                                name: "formats".to_string(),
                                args: vec![
                                    FuncArg::Key("Msg {}".to_string()),
                                    FuncArg::Key("msg".to_string()),
                                ],
                                alias: Some("message".to_string()),
                                should_deref: true,
                            }),
                            Filter::Pick(vec![PickFilterInner::KeyedStr {
                                key: "message".to_string(),
                                alias: "message".to_string(),
                            }]),
                        ],
                        false /* reverse path? */
                    ),
                ]),
            ]
        );
    }

    #[test]
    fn filter_almost_op() {
        let actions = parse(">/foo/#filter('some_key' ~= 'test')").unwrap();
        assert_eq!(
            actions,
            vec![
                Filter::Root,
                Filter::Child("foo".to_string()),
                Filter::MultiFilter(Rc::new(RefCell::new(FilterAST {
                    operator: FilterLogicalOp::None,
                    left: Some(Rc::new(RefCell::new(FilterInner::Cond {
                        left: "some_key".to_string(),
                        op: FilterOp::Almost,
                        right: FilterInnerRighthand::String("test".to_string()),
                    },))),
                    right: None
                })))
            ]
        );
    }

    #[test]
    fn filter_not_equal() {
        let actions = parse(">/foo/#filter('some_key' != 10)").unwrap();
        assert_eq!(
            actions,
            vec![
                Filter::Root,
                Filter::Child("foo".to_string()),
                Filter::MultiFilter(Rc::new(RefCell::new(FilterAST {
                    operator: FilterLogicalOp::None,
                    left: Some(Rc::new(RefCell::new(FilterInner::Cond {
                        left: "some_key".to_string(),
                        op: FilterOp::NotEq,
                        right: FilterInnerRighthand::Number(10),
                    }))),
                    right: None,
                })))
            ]
        );
    }

    #[test]
    fn parse_fncall() {
        let actions = parse(">/#someFn('some', 'argument', >/and/path)").unwrap();
        assert_eq!(
            actions,
            vec![
                Filter::Root,
                Filter::Function(Func {
                    name: "someFn".to_string(),
                    args: vec![
                        FuncArg::Key("some".to_string()),
                        FuncArg::Key("argument".to_string()),
                        FuncArg::SubExpr(vec![
                            Filter::Root,
                            Filter::Child("and".to_string()),
                            Filter::Child("path".to_string()),
                        ])
                    ],
                    alias: None,
                    should_deref: false,
                })
            ]
        );
    }

    #[test]
    fn parse_call_ending_with_deref() {
        let actions = parse(">/#call('format {}', 'a', 'a') ->* 'test'").unwrap();
        assert_eq!(
            actions,
            vec![
                Filter::Root,
                Filter::Function(Func {
                    name: "call".to_string(),
                    args: vec![
                        FuncArg::Key("format {}".to_string()),
                        FuncArg::Key("a".to_string()),
                        FuncArg::Key("a".to_string()),
                    ],
                    alias: Some("test".to_string()),
                    should_deref: true,
                }),
                Filter::Pick(vec![PickFilterInner::KeyedStr {
                    key: "test".to_string(),
                    alias: "test".to_string(),
                }]),
            ]
        );
    }

    #[test]
    fn parse_len_inside_pick() {
        let actions = parse(">/#pick(>/..priority/#len as 'test')").unwrap();
        assert_eq!(
            actions,
            vec![
                Filter::Root,
                Filter::Pick(vec![PickFilterInner::KeyedSubpath {
                    subpath: vec![
                        Filter::Root,
                        Filter::DescendantChild(FilterDescendant::Single("priority".to_string())),
                        Filter::Function(Func {
                            name: "len".to_string(),
                            args: vec![],
                            alias: None,
                            should_deref: false,
                        }),
                    ],
                    alias: "test".to_string(),
                    reverse: false,
                }]),
            ]
        );
    }

    #[test]
    fn parse_map_statement() {
        let actions = parse(">/..values/#map(x: x.a.b.c.to_string())").unwrap();
        assert_eq!(
            actions,
            vec![
                Filter::Root,
                Filter::DescendantChild(FilterDescendant::Single("values".to_string())),
                Filter::Function(Func {
                    name: "map".to_string(),
                    alias: None,
                    should_deref: false,
                    args: vec![FuncArg::MapStmt(MapAST {
                        arg: "x".to_string(),
                        body: MapBody::Method {
                            name: "to_string".to_string(),
                            subpath: vec![
                                Filter::Root,
                                Filter::Child("a".to_string()),
                                Filter::Child("b".to_string()),
                                Filter::Child("c".to_string()),
                            ],
                        },
                    })],
                })
            ]
        );
    }

    #[test]
    fn parse_with_line_break() {
        let actions = parse(
            r#">/#pick(
  >/get
   /..recursive
   /value
   /#filter('some_key' > 10) as 'values',

  >/..preferences
   /#map(x: x.preference) as 'preferences'
)
"#,
        )
        .unwrap();

        assert_eq!(
            actions,
            vec![
                Filter::Root,
                Filter::Pick(vec![
                    PickFilterInner::KeyedSubpath {
                        alias: "values".to_string(),
                        reverse: false,
                        subpath: vec![
                            Filter::Root,
                            Filter::Child("get".to_string()),
                            Filter::DescendantChild(FilterDescendant::Single(
                                "recursive".to_string()
                            )),
                            Filter::Child("value".to_string()),
                            Filter::MultiFilter(Rc::new(RefCell::new(FilterAST {
                                operator: FilterLogicalOp::None,
                                left: Some(Rc::new(RefCell::new(FilterInner::Cond {
                                    left: "some_key".to_string(),
                                    op: FilterOp::Gt,
                                    right: FilterInnerRighthand::Number(10),
                                }))),
                                right: None,
                            }))),
                        ],
                    },
                    PickFilterInner::KeyedSubpath {
                        alias: "preferences".to_string(),
                        reverse: false,
                        subpath: vec![
                            Filter::Root,
                            Filter::DescendantChild(FilterDescendant::Single(
                                "preferences".to_string()
                            )),
                            Filter::Function(Func {
                                name: "map".to_string(),
                                alias: None,
                                should_deref: false,
                                args: vec![FuncArg::MapStmt(MapAST {
                                    arg: "x".to_string(),
                                    body: MapBody::Subpath(vec![
                                        Filter::Root,
                                        Filter::Child("preference".to_string()),
                                    ],),
                                })],
                            }),
                        ],
                    }
                ]),
            ]
        );
    }
}
