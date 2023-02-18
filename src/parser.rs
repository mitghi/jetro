use pest_derive::Parser as Parse;

use crate::context::{Filter, PickFilterInner};
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
            Rule::any_child => actions.push(Filter::AnyChild),
            Rule::pickFn => {
                let elems: Vec<PickFilterInner> = token
                    .into_inner()
                    .nth(1)
                    .unwrap()
                    .into_inner()
                    .map(|v| match &v.as_rule() {
                        Rule::sub_expression => {
                            return PickFilterInner::Subpath(parse(v.as_str()));
                        }
                        Rule::literal => {
                            println!("rule is : {:?}", &v);
                            let span = v.as_span().as_str().to_string();
                            if span.len() == 2 {
                                return PickFilterInner::Str("".to_string());
                            }
                            return PickFilterInner::Str(span[1..span.len() - 1].to_string());
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
}
