//! Module containing parser for jetro.

use crate::context::{
    Filter, FilterInner, FilterInnerRighthand, FilterOp, FormatOp, PickFilterInner,
};

use crate::*;

#[derive(Parse)]
#[grammar = "./grammar.pest"]
pub struct ExpressionParser;

pub(crate) fn parse<'a>(input: &'a str) -> Result<Vec<Filter>, pest::error::Error<Rule>> {
    let pairs = ExpressionParser::parse(Rule::expression, input);
    if pairs.is_err() {
        return Err(pairs.err().unwrap());
    }
    let root = pairs.unwrap().next().unwrap();
    let mut actions: Vec<Filter> = vec![];
    for token in root.into_inner() {
        match token.as_rule() {
            Rule::path | Rule::reverse_path => actions.push(Filter::Root),
            Rule::allFn => actions.push(Filter::All),
            Rule::lenFn => actions.push(Filter::Len),
            Rule::sumFn => actions.push(Filter::Sum),
            Rule::grouped_any => {
                let elem = token.into_inner().nth(1).unwrap().into_inner();
                let mut values: Vec<String> = vec![];
                for e in elem {
                    values.push(e.into_inner().as_str().to_string());
                }

                actions.push(Filter::GroupedChild(values));
            }
            Rule::filterFn => {
                let mut elem = token.into_inner().nth(1).unwrap().into_inner();
                let left = elem
                    .next()
                    .unwrap()
                    .into_inner()
                    .next()
                    .unwrap()
                    .into_inner()
                    .as_str()
                    .to_string();
                let op = FilterOp::get(elem.next().unwrap().into_inner().as_str()).unwrap();
                let right: Option<FilterInnerRighthand>;

                match elem.clone().into_iter().next().unwrap().as_rule() {
                    Rule::literal => {
                        right = Some(FilterInnerRighthand::String(
                            elem.next().unwrap().into_inner().as_str().to_string(),
                        ));
                    }
                    Rule::truthy => {
                        match elem.next().unwrap().into_inner().next().unwrap().as_rule() {
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
                    _ => {
                        todo!();
                    }
                }
                if right.is_none() {
                    todo!("handle error unknown case in right handside");
                }

                actions.push(Filter::Filter(FilterInner::Cond {
                    left,
                    op,
                    right: right.unwrap(),
                }));
            }
            Rule::formatsFn => {
                let mut arguments: Vec<String> = vec![];
                let mut elem = token.into_inner().nth(1).unwrap().into_inner();
                let head = elem.next().unwrap().as_str();
                let format: String = {
                    if head.len() > 2 {
                        head[1..head.len() - 1].to_string()
                    } else {
                        "".to_string()
                    }
                };
                let mut alias: String = "unknown".to_string();
                for e in elem {
                    match e.as_rule() {
                        Rule::literal => {
                            let name: String = {
                                let elem = e.as_str();
                                if elem.len() > 2 {
                                    elem[1..elem.len() - 1].to_string()
                                } else {
                                    "".to_string()
                                }
                            };
                            arguments.push(name);
                        }
                        _ => {
                            let e = e.as_str();
                            if e.len() > 2 {
                                alias = e[4..e.len() - 1].to_string();
                            }
                        }
                    }
                }
                actions.push(Filter::Format(FormatOp::FormatString {
                    format,
                    arguments,
                    alias,
                }));
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
                                    let mut result = result.as_span().as_str()[4..].to_string();
                                    if result.len() > 2 {
                                        result = result[1..result.len() - 1].to_string();
                                    }
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
                let ident = token.into_inner().nth(1).unwrap().as_str().to_owned();
                actions.push(Filter::Descendant(ident));
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
    fn test_one() {
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
    fn test_two() {
        let actions = parse(">/obj/some/..descendant/name").unwrap();
        assert_eq!(
            actions,
            vec![
                Filter::Root,
                Filter::Child("obj".to_string()),
                Filter::Child("some".to_string()),
                Filter::Descendant("descendant".to_string()),
                Filter::Child("name".to_string()),
            ]
        );
    }

    #[test]
    fn test_three() {
        let actions = parse(">/obj/some/..descendant/#pick('a', 'b', 'c', 'd')").unwrap();
        assert_eq!(
            actions,
            vec![
                Filter::Root,
                Filter::Child("obj".to_string()),
                Filter::Child("some".to_string()),
                Filter::Descendant("descendant".to_string()),
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
                Filter::Descendant("descendant".to_string()),
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
                Filter::Descendant("descendant".to_string()),
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
                Filter::Descendant("descendant".to_string()),
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
        let actions = parse(">/#formats('{}{}', 'name', 'alias') as 'some_key'").unwrap();
        assert_eq!(
            actions,
            vec![
                Filter::Root,
                Filter::Format(FormatOp::FormatString {
                    format: "{}{}".to_string(),
                    arguments: vec!["name".to_string(), "alias".to_string()],
                    alias: "some_key".to_string(),
                })
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
                Filter::Descendant("meows".to_string()),
                Filter::Filter(FilterInner::Cond {
                    left: "some".to_string(),
                    op: FilterOp::Eq,
                    right: FilterInnerRighthand::String("value".to_string()),
                })
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
                Filter::Filter(FilterInner::Cond {
                    left: "is_furry".to_string(),
                    op: FilterOp::Eq,
                    right: FilterInnerRighthand::Bool(true)
                })
            ]
        );
    }
}
