use pest_derive::Parser as Parse;

use crate::context::{Filter, FormatOp, PickFilterInner};
use pest::Parser;

#[derive(Parse)]
#[grammar = "./grammar.pest"]
pub struct ExpressionParser;

pub fn parse<'a>(input: &'a str) -> Vec<Filter> {
    let pairs = ExpressionParser::parse(Rule::expression, input);
    let root = pairs.unwrap().next().unwrap();
    let mut actions: Vec<Filter> = vec![];
    for token in root.into_inner() {
        match token.as_rule() {
            Rule::path => actions.push(Filter::Root),
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
                let elems: Vec<PickFilterInner> = token
                    .into_inner()
                    .nth(1)
                    .unwrap()
                    .into_inner()
                    .map(|v| match &v.as_rule() {
                        Rule::sub_expression_keyed => {
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
                                    return PickFilterInner::KeyedSubpath {
                                        subpath: parse(subexpr),
                                        alias,
                                    };
                                }
                                None => {
                                    return PickFilterInner::Subpath(parse(subexpr));
                                }
                            }
                        }
                        Rule::sub_expression => {
                            return PickFilterInner::Subpath(parse(v.as_str()));
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
                                (2, _) => return PickFilterInner::Str("".to_string()),
                                (_, None) => {
                                    return PickFilterInner::Str(
                                        span[1..span.len() - 1].to_string(),
                                    );
                                }
                                (_, Some(alias)) => {
                                    return PickFilterInner::KeyedStr {
                                        key: span[1..span.len() - 1].to_string(),
                                        alias: alias.to_string(),
                                    };
                                }
                            }
                        }
                        _ => {
                            return PickFilterInner::None;
                        }
                    })
                    .collect();
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
    actions
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_one() {
        let actions = parse(">/obj/some/*/name");
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
        let actions = parse(">/obj/some/..descendant/name");
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
        let actions = parse(">/obj/some/..descendant/pick('a', 'b', 'c', 'd')");
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
        let actions = parse(">/obj/some/..descendant/pick('f' as 'foo', 'b' as 'bar')");
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
        let actions = parse(">/obj/some/..descendant/pick('f' as 'foo', >/some/branch as 'path')");
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
                        alias: "path".to_string()
                    },
                ]),
            ]
        );
    }

    #[test]
    fn test_slice() {
        let actions = parse(">/[1:4]");
        assert_eq!(actions, vec![Filter::Root, Filter::Slice(1, 4),]);
    }

    #[test]
    fn test_format() {
        let actions = parse(">/formats('{}{}', 'name', 'alias') as 'some_key'");
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
}
