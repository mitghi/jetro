//! PEG parser for Jetro v2 source text.
//!
//! The grammar lives in [`grammar.pest`]; this module walks the pest
//! parse tree and builds an [`Expr`] AST.  Operator precedence and
//! associativity are encoded in the grammar — the walker here is a
//! near-mechanical tree fold with no precedence decisions of its own.
//!
//! # Error handling
//!
//! Pest reports errors with line/column spans; we wrap them into a
//! [`ParseError`] that implements `Display` and `Error`.  The wrapper
//! exists because callers shouldn't need to depend on pest types.
//!
//! # Parser state
//!
//! The original v1 parser threaded a `RefCell`-shared counter for
//! gensym (unique ident generation).  v2 does that differently — we
//! emit fresh temp names in the compiler, not the parser, so the
//! parser is a pure function of its input.

use pest::iterators::Pair;
use pest::Parser as PestParser;
use pest_derive::Parser;
use std::fmt;

use super::ast::*;

#[derive(Parser)]
#[grammar = "v2/grammar.pest"]
pub struct V2Parser;

// ── Error ─────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct ParseError(pub String);

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "parse error: {}", self.0)
    }
}

impl std::error::Error for ParseError {}

impl From<pest::error::Error<Rule>> for ParseError {
    fn from(e: pest::error::Error<Rule>) -> Self {
        ParseError(e.to_string())
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

pub fn parse(input: &str) -> Result<Expr, ParseError> {
    let mut pairs = V2Parser::parse(Rule::program, input)?;
    let program = pairs.next().unwrap();
    let expr_pair = program.into_inner().next().unwrap();
    Ok(parse_expr(expr_pair))
}

// ── Keyword helpers ───────────────────────────────────────────────────────────

fn is_kw(rule: Rule) -> bool {
    matches!(
        rule,
        Rule::kw_and | Rule::kw_or | Rule::kw_not | Rule::kw_for
            | Rule::kw_in | Rule::kw_if | Rule::kw_let | Rule::kw_lambda | Rule::kw_kind
            | Rule::kw_is | Rule::kw_as
    )
}

// ── Expr dispatch ─────────────────────────────────────────────────────────────

fn parse_expr(pair: Pair<Rule>) -> Expr {
    match pair.as_rule() {
        Rule::expr          => parse_expr(pair.into_inner().next().unwrap()),
        Rule::pipe_expr     => parse_pipeline(pair),
        Rule::coalesce_expr => parse_coalesce(pair),
        Rule::or_expr       => parse_or(pair),
        Rule::and_expr      => parse_and(pair),
        Rule::not_expr      => parse_not(pair),
        Rule::kind_expr     => parse_kind(pair),
        Rule::cmp_expr      => parse_cmp(pair),
        Rule::add_expr      => parse_add(pair),
        Rule::mul_expr      => parse_mul(pair),
        Rule::cast_expr     => parse_cast(pair),
        Rule::unary_expr    => parse_unary(pair),
        Rule::postfix_expr  => parse_postfix_expr(pair),
        Rule::primary       => parse_primary(pair),
        r => panic!("unexpected rule in parse_expr: {:?}", r),
    }
}

// ── Pipeline (pipe + bind) ────────────────────────────────────────────────────

fn parse_pipeline(pair: Pair<Rule>) -> Expr {
    let mut inner = pair.into_inner();
    let base = parse_expr(inner.next().unwrap()); // coalesce_expr
    let mut steps: Vec<PipeStep> = Vec::new();
    for step_pair in inner {
        // each is a pipe_step
        let inner_step = step_pair.into_inner().next().unwrap();
        match inner_step.as_rule() {
            Rule::pipe_forward => {
                let inner_pair = inner_step.into_inner().next().unwrap();
                let expr = if inner_pair.as_rule() == Rule::pipe_method_call {
                    let mut mi = inner_pair.into_inner();
                    let name = mi.next().unwrap().as_str().to_string();
                    let args = mi.next().map(parse_arg_list).unwrap_or_default();
                    Expr::Chain(Box::new(Expr::Current), vec![Step::Method(name, args)])
                } else {
                    parse_expr(inner_pair)
                };
                steps.push(PipeStep::Forward(expr));
            }
            Rule::pipe_bind => {
                let target = parse_bind_target(inner_step.into_inner().next().unwrap());
                steps.push(PipeStep::Bind(target));
            }
            r => panic!("unexpected pipe_step inner: {:?}", r),
        }
    }
    if steps.is_empty() { base } else { Expr::Pipeline { base: Box::new(base), steps } }
}

fn parse_bind_target(pair: Pair<Rule>) -> BindTarget {
    // pair.as_rule() == Rule::bind_target
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::ident => BindTarget::Name(inner.as_str().to_string()),
        Rule::bind_obj => {
            let mut fields = Vec::new();
            let mut rest = None;
            for p in inner.into_inner() {
                match p.as_rule() {
                    Rule::ident => fields.push(p.as_str().to_string()),
                    Rule::bind_rest => {
                        rest = Some(
                            p.into_inner()
                             .find(|x| x.as_rule() == Rule::ident)
                             .unwrap()
                             .as_str()
                             .to_string()
                        );
                    }
                    _ => {}
                }
            }
            BindTarget::Obj { fields, rest }
        }
        Rule::bind_arr => {
            let fields: Vec<String> = inner.into_inner()
                .filter(|p| p.as_rule() == Rule::ident)
                .map(|p| p.as_str().to_string())
                .collect();
            BindTarget::Arr(fields)
        }
        r => panic!("unexpected bind_target inner: {:?}", r),
    }
}

// ── Coalesce ──────────────────────────────────────────────────────────────────

fn parse_coalesce(pair: Pair<Rule>) -> Expr {
    let mut inner = pair.into_inner();
    let first = parse_expr(inner.next().unwrap());
    inner.fold(first, |acc, rhs| {
        Expr::Coalesce(Box::new(acc), Box::new(parse_expr(rhs)))
    })
}

// ── Logical ──────────────────────────────────────────────────────────────────

fn parse_or(pair: Pair<Rule>) -> Expr {
    let mut inner = pair.into_inner().filter(|p| !is_kw(p.as_rule()));
    let first = parse_expr(inner.next().unwrap());
    inner.fold(first, |acc, rhs| {
        Expr::BinOp(Box::new(acc), BinOp::Or, Box::new(parse_expr(rhs)))
    })
}

fn parse_and(pair: Pair<Rule>) -> Expr {
    let mut inner = pair.into_inner().filter(|p| !is_kw(p.as_rule()));
    let first = parse_expr(inner.next().unwrap());
    inner.fold(first, |acc, rhs| {
        Expr::BinOp(Box::new(acc), BinOp::And, Box::new(parse_expr(rhs)))
    })
}

fn parse_not(pair: Pair<Rule>) -> Expr {
    let mut inner = pair.into_inner();
    let first = inner.next().unwrap();
    if first.as_rule() == Rule::kw_not {
        let operand = inner.next().unwrap();
        Expr::Not(Box::new(parse_expr(operand)))
    } else {
        parse_expr(first)
    }
}

// ── Kind check ────────────────────────────────────────────────────────────────

fn parse_kind(pair: Pair<Rule>) -> Expr {
    let mut inner = pair.into_inner();
    let cmp = parse_expr(inner.next().unwrap());
    match inner.next() {
        None => cmp,
        Some(p) if matches!(p.as_rule(), Rule::kw_kind | Rule::kw_is) => {
            let next = inner.next().unwrap();
            let (negate, kind_type_str) = if next.as_rule() == Rule::kw_not {
                (true, inner.next().unwrap().as_str())
            } else {
                (false, next.as_str())
            };
            let ty = match kind_type_str {
                "null"   => KindType::Null,
                "bool"   => KindType::Bool,
                "number" => KindType::Number,
                "string" => KindType::Str,
                "array"  => KindType::Array,
                "object" => KindType::Object,
                other    => panic!("unknown kind type: {}", other),
            };
            Expr::Kind { expr: Box::new(cmp), ty, negate }
        }
        _ => cmp,
    }
}

// ── Comparison ───────────────────────────────────────────────────────────────

fn parse_cmp(pair: Pair<Rule>) -> Expr {
    let mut inner = pair.into_inner();
    let lhs = parse_expr(inner.next().unwrap());
    if let Some(op_pair) = inner.next() {
        let op = match op_pair.as_str() {
            "==" => BinOp::Eq,  "!=" => BinOp::Neq,
            "<"  => BinOp::Lt,  "<=" => BinOp::Lte,
            ">"  => BinOp::Gt,  ">=" => BinOp::Gte,
            "~=" => BinOp::Fuzzy,
            o    => panic!("unknown cmp op: {}", o),
        };
        let rhs = parse_expr(inner.next().unwrap());
        Expr::BinOp(Box::new(lhs), op, Box::new(rhs))
    } else {
        lhs
    }
}

// ── Additive / multiplicative ─────────────────────────────────────────────────

fn parse_add(pair: Pair<Rule>) -> Expr {
    parse_left_assoc(pair, |s| match s {
        "+" => Some(BinOp::Add), "-" => Some(BinOp::Sub), _ => None,
    })
}

fn parse_mul(pair: Pair<Rule>) -> Expr {
    parse_left_assoc(pair, |s| match s {
        "*" => Some(BinOp::Mul), "/" => Some(BinOp::Div), "%" => Some(BinOp::Mod), _ => None,
    })
}

fn parse_left_assoc<F>(pair: Pair<Rule>, op_fn: F) -> Expr
where
    F: Fn(&str) -> Option<BinOp>,
{
    let mut inner = pair.into_inner().peekable();
    let first = parse_expr(inner.next().unwrap());
    let mut acc = first;
    while inner.peek().is_some() {
        let op_pair = inner.next().unwrap();
        let op = op_fn(op_pair.as_str()).unwrap();
        let rhs = parse_expr(inner.next().unwrap());
        acc = Expr::BinOp(Box::new(acc), op, Box::new(rhs));
    }
    acc
}

// ── Cast ──────────────────────────────────────────────────────────────────────

fn parse_cast(pair: Pair<Rule>) -> Expr {
    // cast_expr = { unary_expr ~ (kw_as ~ kind_type)* }
    let mut inner = pair.into_inner().peekable();
    let mut acc = parse_expr(inner.next().unwrap());
    while inner.peek().is_some() {
        // Consume kw_as then kind_type
        let kw = inner.next().unwrap();
        debug_assert_eq!(kw.as_rule(), Rule::kw_as);
        let ty_pair = inner.next().unwrap();
        let ty = match ty_pair.as_str() {
            "int"    => CastType::Int,
            "float"  => CastType::Float,
            "number" => CastType::Number,
            "string" => CastType::Str,
            "bool"   => CastType::Bool,
            "array"  => CastType::Array,
            "object" => CastType::Object,
            "null"   => CastType::Null,
            o        => panic!("unknown cast type: {}", o),
        };
        acc = Expr::Cast { expr: Box::new(acc), ty };
    }
    acc
}

// ── Unary ─────────────────────────────────────────────────────────────────────

fn parse_unary(pair: Pair<Rule>) -> Expr {
    let mut inner = pair.into_inner();
    let first = inner.next().unwrap();
    match first.as_rule() {
        Rule::unary_neg => {
            let operand = parse_expr(inner.next().unwrap());
            Expr::UnaryNeg(Box::new(operand))
        }
        _ => parse_expr(first),
    }
}

// ── Postfix chain ─────────────────────────────────────────────────────────────

fn parse_postfix_expr(pair: Pair<Rule>) -> Expr {
    let mut inner = pair.into_inner();
    let base = parse_primary(inner.next().unwrap());
    let steps: Vec<Step> = inner.flat_map(parse_postfix_step).collect();
    base.maybe_chain(steps)
}

fn parse_postfix_step(pair: Pair<Rule>) -> Vec<Step> {
    let inner_pair = pair.into_inner().next().unwrap();
    match inner_pair.as_rule() {
        Rule::field_access => {
            let name = inner_pair.into_inner().next().unwrap().as_str().to_string();
            vec![Step::Field(name)]
        }
        Rule::opt_field => {
            let name = inner_pair.into_inner().next().unwrap().as_str().to_string();
            vec![Step::OptField(name)]
        }
        Rule::descendant => {
            let mut di = inner_pair.into_inner();
            match di.next() {
                Some(p) => vec![Step::Descendant(p.as_str().to_string())],
                None    => vec![Step::DescendAll],
            }
        }
        Rule::inline_filter => {
            let expr = parse_expr(inner_pair.into_inner().next().unwrap());
            vec![Step::InlineFilter(Box::new(expr))]
        }
        Rule::quantifier => {
            let s = inner_pair.as_str();
            if s.starts_with('!') { vec![Step::Quantifier(QuantifierKind::One)] }
            else                  { vec![Step::Quantifier(QuantifierKind::First)] }
        }
        Rule::method_call => {
            let mut mi = inner_pair.into_inner();
            let name = mi.next().unwrap().as_str().to_string();
            let args = mi.next().map(parse_arg_list).unwrap_or_default();
            vec![Step::Method(name, args)]
        }
        Rule::opt_method => {
            let mut mi = inner_pair.into_inner();
            let name = mi.next().unwrap().as_str().to_string();
            let args = mi.next().map(parse_arg_list).unwrap_or_default();
            vec![Step::OptMethod(name, args)]
        }
        Rule::index_access => {
            let bi = inner_pair.into_inner().next().unwrap();
            vec![parse_bracket(bi)]
        }
        Rule::dyn_field => {
            let expr = parse_expr(inner_pair.into_inner().next().unwrap());
            vec![Step::DynIndex(Box::new(expr))]
        }
        Rule::map_into_shape => {
            // `[*] => body` or `[* if g] => body`
            // Desugar: with guard  → .filter(g).map(body)
            //          no guard    → .map(body)
            let mut guard: Option<Expr> = None;
            let mut body:  Option<Expr> = None;
            let mut saw_if = false;
            for p in inner_pair.into_inner() {
                match p.as_rule() {
                    Rule::kw_if => saw_if = true,
                    Rule::expr  => {
                        if saw_if && guard.is_none() { guard = Some(parse_expr(p)); }
                        else                         { body  = Some(parse_expr(p)); }
                    }
                    _ => {}
                }
            }
            let body = body.expect("map_into_shape requires body");
            let mut steps = Vec::new();
            if let Some(g) = guard {
                steps.push(Step::Method("filter".into(), vec![Arg::Pos(g)]));
            }
            steps.push(Step::Method("map".into(), vec![Arg::Pos(body)]));
            steps
        }
        r => panic!("unexpected postfix rule: {:?}", r),
    }
}

fn parse_bracket(pair: Pair<Rule>) -> Step {
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::idx_only   => Step::Index(inner.as_str().parse().unwrap()),
        Rule::slice_full => {
            let mut i = inner.into_inner();
            let a = i.next().unwrap().as_str().parse().ok();
            let b = i.next().unwrap().as_str().parse().ok();
            Step::Slice(a, b)
        }
        Rule::slice_from => {
            let a = inner.into_inner().next().unwrap().as_str().parse().ok();
            Step::Slice(a, None)
        }
        Rule::slice_to => {
            let b = inner.into_inner().next().unwrap().as_str().parse().ok();
            Step::Slice(None, b)
        }
        Rule::expr => Step::DynIndex(Box::new(parse_expr(inner))),
        r => panic!("unexpected bracket rule: {:?}", r),
    }
}

// ── Primary ───────────────────────────────────────────────────────────────────

fn parse_primary(pair: Pair<Rule>) -> Expr {
    let inner = if pair.as_rule() == Rule::primary {
        pair.into_inner().next().unwrap()
    } else {
        pair
    };
    match inner.as_rule() {
        Rule::literal       => parse_literal(inner),
        Rule::root          => Expr::Root,
        Rule::current       => Expr::Current,
        Rule::ident         => Expr::Ident(inner.as_str().to_string()),
        Rule::let_expr      => parse_let(inner),
        Rule::lambda_expr   => parse_lambda(inner),
        Rule::arrow_lambda  => parse_arrow_lambda(inner),
        Rule::list_comp     => parse_list_comp(inner),
        Rule::dict_comp     => parse_dict_comp(inner),
        Rule::set_comp      => parse_set_comp(inner),
        Rule::gen_comp      => parse_gen_comp(inner),
        Rule::obj_construct => parse_obj(inner),
        Rule::arr_construct => parse_arr(inner),
        Rule::global_call   => parse_global_call(inner),
        Rule::expr          => parse_expr(inner),
        Rule::patch_block   => parse_patch(inner),
        Rule::kw_delete     => Expr::DeleteMark,
        r => panic!("unexpected primary rule: {:?}", r),
    }
}

// ── Literals ──────────────────────────────────────────────────────────────────

fn parse_literal(pair: Pair<Rule>) -> Expr {
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::lit_null    => Expr::Null,
        Rule::lit_true    => Expr::Bool(true),
        Rule::lit_false   => Expr::Bool(false),
        Rule::lit_int     => Expr::Int(inner.as_str().parse().unwrap()),
        Rule::lit_float   => Expr::Float(inner.as_str().parse().unwrap()),
        Rule::lit_fstring => {
            let raw = inner.as_str();
            let content = &raw[2..raw.len() - 1]; // strip f" and "
            let parts = parse_fstring_content(content);
            Expr::FString(parts)
        }
        Rule::lit_str => {
            let s = inner.into_inner().next().unwrap();
            let raw = s.as_str();
            // Strip surrounding quotes (now atomic rules, so as_str includes them)
            Expr::Str(raw[1..raw.len()-1].to_string())
        }
        r => panic!("unexpected literal rule: {:?}", r),
    }
}

// ── F-string parser ───────────────────────────────────────────────────────────

fn parse_fstring_content(raw: &str) -> Vec<FStringPart> {
    let mut parts = Vec::new();
    let mut lit = String::new();
    let mut chars = raw.chars().peekable();

    while let Some(c) = chars.next() {
        match c {
            '{' => {
                if chars.peek() == Some(&'{') {
                    chars.next();
                    lit.push('{');
                } else {
                    if !lit.is_empty() {
                        parts.push(FStringPart::Lit(std::mem::take(&mut lit)));
                    }
                    let mut inner = String::new();
                    let mut depth = 1usize;
                    for c2 in chars.by_ref() {
                        match c2 {
                            '{' => { depth += 1; inner.push(c2); }
                            '}' => {
                                depth -= 1;
                                if depth == 0 { break; }
                                inner.push(c2);
                            }
                            _ => inner.push(c2),
                        }
                    }
                    let (expr_str, fmt) = split_fstring_interp(&inner);
                    let expr = parse(expr_str.trim())
                        .unwrap_or_else(|e| panic!("f-string parse error in {{{}}}: {}", inner, e));
                    parts.push(FStringPart::Interp { expr, fmt });
                }
            }
            '}' if chars.peek() == Some(&'}') => {
                chars.next();
                lit.push('}');
            }
            _ => lit.push(c),
        }
    }
    if !lit.is_empty() {
        parts.push(FStringPart::Lit(lit));
    }
    parts
}

fn split_fstring_interp(inner: &str) -> (&str, Option<FmtSpec>) {
    // Find top-level `|` or `:` (not inside parens/brackets/braces)
    let mut depth = 0usize;
    let mut pipe_pos: Option<usize> = None;
    let mut colon_pos: Option<usize> = None;
    for (i, c) in inner.char_indices() {
        match c {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => { if depth > 0 { depth -= 1; } }
            '|' if depth == 0 && pipe_pos.is_none() => pipe_pos = Some(i),
            ':' if depth == 0 && colon_pos.is_none() => colon_pos = Some(i),
            _ => {}
        }
    }
    if let Some(p) = pipe_pos {
        return (&inner[..p], Some(FmtSpec::Pipe(inner[p + 1..].trim().to_string())));
    }
    if let Some(c) = colon_pos {
        return (&inner[..c], Some(FmtSpec::Spec(inner[c + 1..].to_string())));
    }
    (inner, None)
}

// ── Let ───────────────────────────────────────────────────────────────────────

fn parse_let(pair: Pair<Rule>) -> Expr {
    // let_expr = { kw_let ~ let_binding ~ ("," ~ let_binding)* ~ kw_in ~ expr }
    // Desugar multi-binding into nested Let: `let a=x, b=y in body` → Let a x (Let b y body)
    let inner: Vec<Pair<Rule>> = pair.into_inner()
        .filter(|p| !matches!(p.as_rule(), Rule::kw_let | Rule::kw_in))
        .collect();
    let (bindings, body_pair) = inner.split_at(inner.len() - 1);
    let body = parse_expr(body_pair[0].clone());
    bindings.iter().rev().fold(body, |acc, b| {
        let mut bi = b.clone().into_inner();
        let name = bi.next().unwrap().as_str().to_string();
        let init = parse_expr(bi.next().unwrap());
        Expr::Let { name, init: Box::new(init), body: Box::new(acc) }
    })
}

// ── Lambda ────────────────────────────────────────────────────────────────────

fn parse_lambda(pair: Pair<Rule>) -> Expr {
    let mut inner = pair.into_inner()
        .filter(|p| p.as_rule() != Rule::kw_lambda);
    let params_pair = inner.next().unwrap();
    let params: Vec<String> = params_pair
        .into_inner()
        .filter(|p| p.as_rule() == Rule::ident)
        .map(|p| p.as_str().to_string())
        .collect();
    let body = parse_expr(inner.next().unwrap());
    Expr::Lambda { params, body: Box::new(body) }
}

/// Arrow lambda: `x => body` or `(x, y) => body`.  Lowered to same
/// `Expr::Lambda` node — the `=>` form is pure surface sugar.
fn parse_arrow_lambda(pair: Pair<Rule>) -> Expr {
    let mut inner = pair.into_inner();
    let params_pair = inner.next().unwrap();
    let params: Vec<String> = params_pair
        .into_inner()
        .filter(|p| p.as_rule() == Rule::ident)
        .map(|p| p.as_str().to_string())
        .collect();
    let body = parse_expr(inner.next().unwrap());
    Expr::Lambda { params, body: Box::new(body) }
}

// ── Comprehensions ────────────────────────────────────────────────────────────

fn comp_inner_filter(pair: Pair<Rule>) -> impl Iterator<Item = Pair<Rule>> {
    pair.into_inner()
        .filter(|p| !matches!(p.as_rule(), Rule::kw_for | Rule::kw_in | Rule::kw_if))
}

fn parse_comp_vars(pair: Pair<Rule>) -> Vec<String> {
    pair.into_inner()
        .filter(|p| p.as_rule() == Rule::ident)
        .map(|p| p.as_str().to_string())
        .collect()
}

fn parse_list_comp(pair: Pair<Rule>) -> Expr {
    let mut inner = comp_inner_filter(pair);
    let expr = parse_expr(inner.next().unwrap());
    let vars = parse_comp_vars(inner.next().unwrap());
    let iter = parse_expr(inner.next().unwrap());
    let cond = inner.next().map(|p| Box::new(parse_expr(p)));
    Expr::ListComp { expr: Box::new(expr), vars, iter: Box::new(iter), cond }
}

fn parse_dict_comp(pair: Pair<Rule>) -> Expr {
    let mut inner = comp_inner_filter(pair);
    let key  = parse_expr(inner.next().unwrap());
    let val  = parse_expr(inner.next().unwrap());
    let vars = parse_comp_vars(inner.next().unwrap());
    let iter = parse_expr(inner.next().unwrap());
    let cond = inner.next().map(|p| Box::new(parse_expr(p)));
    Expr::DictComp { key: Box::new(key), val: Box::new(val), vars, iter: Box::new(iter), cond }
}

fn parse_set_comp(pair: Pair<Rule>) -> Expr {
    let mut inner = comp_inner_filter(pair);
    let expr = parse_expr(inner.next().unwrap());
    let vars = parse_comp_vars(inner.next().unwrap());
    let iter = parse_expr(inner.next().unwrap());
    let cond = inner.next().map(|p| Box::new(parse_expr(p)));
    Expr::SetComp { expr: Box::new(expr), vars, iter: Box::new(iter), cond }
}

fn parse_gen_comp(pair: Pair<Rule>) -> Expr {
    let mut inner = comp_inner_filter(pair);
    let expr = parse_expr(inner.next().unwrap());
    let vars = parse_comp_vars(inner.next().unwrap());
    let iter = parse_expr(inner.next().unwrap());
    let cond = inner.next().map(|p| Box::new(parse_expr(p)));
    Expr::GenComp { expr: Box::new(expr), vars, iter: Box::new(iter), cond }
}

// ── Object / array construction ────────────────────────────────────────────────

fn parse_obj(pair: Pair<Rule>) -> Expr {
    let fields = pair.into_inner()
        .filter(|p| p.as_rule() == Rule::obj_field)
        .map(parse_obj_field)
        .collect();
    Expr::Object(fields)
}

fn parse_obj_field(pair: Pair<Rule>) -> ObjField {
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::obj_field_dyn => {
            let mut i = inner.into_inner();
            let key = parse_expr(i.next().unwrap());
            let val = parse_expr(i.next().unwrap());
            ObjField::Dynamic { key, val }
        }
        Rule::obj_field_opt_v => {
            let mut i = inner.into_inner();
            let key = obj_key_str(i.next().unwrap());
            let val = parse_expr(i.next().unwrap());
            ObjField::Kv { key, val, optional: true, cond: None }
        }
        Rule::obj_field_opt => {
            let key = obj_key_str(inner.into_inner().next().unwrap());
            ObjField::Kv { key: key.clone(), val: Expr::Ident(key), optional: true, cond: None }
        }
        Rule::obj_field_spread => {
            let expr = parse_expr(inner.into_inner().next().unwrap());
            ObjField::Spread(expr)
        }
        Rule::obj_field_spread_deep => {
            let expr = parse_expr(inner.into_inner().next().unwrap());
            ObjField::SpreadDeep(expr)
        }
        Rule::obj_field_kv => {
            let mut cond: Option<Expr> = None;
            let mut key: Option<String> = None;
            let mut val: Option<Expr> = None;
            let mut saw_when = false;
            for p in inner.into_inner() {
                match p.as_rule() {
                    Rule::kw_when => saw_when = true,
                    Rule::obj_key_expr => key = Some(obj_key_str(p)),
                    Rule::expr => {
                        if saw_when { cond = Some(parse_expr(p)); }
                        else        { val  = Some(parse_expr(p)); }
                    }
                    _ => {}
                }
            }
            ObjField::Kv {
                key:      key.expect("obj_field_kv missing key"),
                val:      val.expect("obj_field_kv missing val"),
                optional: false,
                cond,
            }
        }
        Rule::obj_field_short => {
            let name = inner.into_inner().next().unwrap().as_str().to_string();
            ObjField::Short(name)
        }
        r => panic!("unexpected obj_field rule: {:?}", r),
    }
}

fn obj_key_str(pair: Pair<Rule>) -> String {
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::ident => inner.as_str().to_string(),
        Rule::lit_str => {
            let s = inner.into_inner().next().unwrap();
            let raw = s.as_str();
            raw[1..raw.len()-1].to_string()
        }
        r => panic!("unexpected obj_key_expr rule: {:?}", r),
    }
}

fn parse_arr(pair: Pair<Rule>) -> Expr {
    let elems = pair.into_inner()
        .filter(|p| p.as_rule() == Rule::arr_elem)
        .map(|elem| {
            let inner = elem.into_inner().next().unwrap();
            match inner.as_rule() {
                Rule::arr_spread => {
                    let expr = parse_expr(inner.into_inner().next().unwrap());
                    ArrayElem::Spread(expr)
                }
                _ => ArrayElem::Expr(parse_expr(inner)),
            }
        })
        .collect();
    Expr::Array(elems)
}

// ── Global call ───────────────────────────────────────────────────────────────

fn parse_global_call(pair: Pair<Rule>) -> Expr {
    let mut inner = pair.into_inner();
    let name = inner.next().unwrap().as_str().to_string();
    let args = inner.next().map(parse_arg_list).unwrap_or_default();
    Expr::GlobalCall { name, args }
}

// ── Patch block ───────────────────────────────────────────────────────────────

fn parse_patch(pair: Pair<Rule>) -> Expr {
    let mut root: Option<Expr> = None;
    let mut ops: Vec<PatchOp> = Vec::new();
    for p in pair.into_inner() {
        match p.as_rule() {
            Rule::kw_patch   => {}
            Rule::patch_field => ops.push(parse_patch_field(p)),
            _ => {
                // This branch handles the root expression (coalesce_expr and
                // any of its descendants that parse_expr accepts).
                if root.is_none() { root = Some(parse_expr(p)); }
            }
        }
    }
    Expr::Patch {
        root: Box::new(root.expect("patch requires root expression")),
        ops,
    }
}

fn parse_patch_field(pair: Pair<Rule>) -> PatchOp {
    let mut path: Vec<PathStep> = Vec::new();
    let mut val:  Option<Expr> = None;
    let mut cond: Option<Expr> = None;
    let mut saw_when = false;
    for p in pair.into_inner() {
        match p.as_rule() {
            Rule::patch_key => path = parse_patch_key(p),
            Rule::kw_when   => saw_when = true,
            Rule::expr => {
                if saw_when { cond = Some(parse_expr(p)); }
                else        { val  = Some(parse_expr(p)); }
            }
            _ => {}
        }
    }
    PatchOp {
        path,
        val:  val.expect("patch_field missing val"),
        cond,
    }
}

fn parse_patch_key(pair: Pair<Rule>) -> Vec<PathStep> {
    let mut steps: Vec<PathStep> = Vec::new();
    let mut first = true;
    for p in pair.into_inner() {
        match p.as_rule() {
            Rule::ident if first => {
                steps.push(PathStep::Field(p.as_str().to_string()));
                first = false;
            }
            Rule::patch_step => steps.push(parse_patch_step(p)),
            _ => {}
        }
    }
    steps
}

fn parse_patch_step(pair: Pair<Rule>) -> PathStep {
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::pp_dot_field => {
            let name = inner.into_inner().next().unwrap().as_str().to_string();
            PathStep::Field(name)
        }
        Rule::pp_index => {
            let idx: i64 = inner.into_inner().next().unwrap().as_str().parse().unwrap();
            PathStep::Index(idx)
        }
        Rule::pp_wild => PathStep::Wildcard,
        Rule::pp_wild_filter => {
            // `[* if expr]`
            let mut e: Option<Expr> = None;
            for p in inner.into_inner() {
                if p.as_rule() == Rule::expr { e = Some(parse_expr(p)); }
            }
            PathStep::WildcardFilter(Box::new(e.expect("pp_wild_filter missing expr")))
        }
        Rule::pp_descendant => {
            let name = inner.into_inner().next().unwrap().as_str().to_string();
            PathStep::Descendant(name)
        }
        r => panic!("unexpected patch_step rule: {:?}", r),
    }
}

// ── Arguments ─────────────────────────────────────────────────────────────────

fn parse_arg_list(pair: Pair<Rule>) -> Vec<Arg> {
    pair.into_inner()
        .filter(|p| p.as_rule() == Rule::arg)
        .map(parse_arg)
        .collect()
}

fn parse_arg(pair: Pair<Rule>) -> Arg {
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::named_arg => {
            let mut i = inner.into_inner();
            let name = i.next().unwrap().as_str().to_string();
            let val  = parse_expr(i.next().unwrap());
            Arg::Named(name, val)
        }
        Rule::pos_arg => {
            Arg::Pos(parse_expr(inner.into_inner().next().unwrap()))
        }
        r => panic!("unexpected arg rule: {:?}", r),
    }
}
