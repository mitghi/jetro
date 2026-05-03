//! PEG parser for the Jetro query language.
//!
//! `grammar.pest` defines the grammar; `pest_derive` generates `V2Parser`.
//! `parse()` drives the parser and walks the parse tree into an `Expr` AST.
//! `classify_chain_write` post-processes rooted chain expressions that end in
//! `.set` / `.modify` / `.delete` / `.unset` into `Expr::Patch` nodes so the
//! evaluator never needs to special-case the write surface at runtime.

use pest::iterators::Pair;
use pest::Parser as PestParser;
use pest_derive::Parser;
use std::fmt;

use super::ast::*;

/// Pest-derived parser for the v2 grammar. The grammar file is embedded at
/// compile time via the `#[grammar]` attribute and is not loaded at runtime.
#[derive(Parser)]
#[grammar = "grammar.pest"]
pub struct V2Parser;


/// Returned by `parse` when the input does not conform to the grammar or when
/// a semantic constraint (e.g. unknown cast type) is violated.
#[derive(Debug)]
pub struct ParseError(pub String);

impl fmt::Display for ParseError {
    /// Format the error as a human-readable message including the source snippet.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "parse error: {}", self.0)
    }
}

impl std::error::Error for ParseError {}

impl From<pest::error::Error<Rule>> for ParseError {
    /// Convert a pest parse error into a `ParseError`, preserving the full
    /// location and message pest provides.
    fn from(e: pest::error::Error<Rule>) -> Self {
        ParseError(e.to_string())
    }
}


/// Parse a Jetro query string into an `Expr` AST. This is the primary public
/// entry point; all other `parse_*` functions are internal helpers.
pub fn parse(input: &str) -> Result<Expr, ParseError> {
    let mut pairs = V2Parser::parse(Rule::program, input)?;
    let program = pairs.next().unwrap();
    let expr_pair = program.into_inner().next().unwrap();
    Ok(parse_expr(expr_pair))
}


/// Return `true` when `rule` is a keyword terminal (`and`, `or`, `not`, …).
/// Used to skip keyword tokens that appear as decoration in binary/unary rules.
fn is_kw(rule: Rule) -> bool {
    matches!(
        rule,
        Rule::kw_and
            | Rule::kw_or
            | Rule::kw_not
            | Rule::kw_for
            | Rule::kw_in
            | Rule::kw_if
            | Rule::kw_else
            | Rule::kw_let
            | Rule::kw_lambda
            | Rule::kw_kind
            | Rule::kw_is
            | Rule::kw_as
            | Rule::kw_try
    )
}


/// Dispatch on `pair.as_rule()` and delegate to the appropriate specialised
/// `parse_*` function, covering the full expression precedence hierarchy.
fn parse_expr(pair: Pair<Rule>) -> Expr {
    match pair.as_rule() {
        Rule::expr => parse_expr(pair.into_inner().next().unwrap()),
        Rule::cond_expr => parse_cond(pair),
        Rule::pipe_expr => parse_pipeline(pair),
        Rule::coalesce_expr => parse_coalesce(pair),
        Rule::or_expr => parse_or(pair),
        Rule::and_expr => parse_and(pair),
        Rule::not_expr => parse_not(pair),
        Rule::kind_expr => parse_kind(pair),
        Rule::contains_expr => parse_contains(pair),
        Rule::cmp_expr => parse_cmp(pair),
        Rule::add_expr => parse_add(pair),
        Rule::mul_expr => parse_mul(pair),
        Rule::cast_expr => parse_cast(pair),
        Rule::unary_expr => parse_unary(pair),
        Rule::postfix_expr => parse_postfix_expr(pair),
        Rule::primary => parse_primary(pair),
        r => panic!("unexpected rule in parse_expr: {:?}", r),
    }
}


/// Parse a conditional expression (`if … then … else …`) or a `try … else …`
/// expression. When the pair contains only one sub-expression, it is returned
/// directly without wrapping.
fn parse_cond(pair: Pair<Rule>) -> Expr {
    // Grammar: cond_expr = { try_expr | (pipe_expr ("if" pipe_expr "else" pipe_expr)?) }
    // After filtering keywords the order is: then, cond, else.
    let mut inner = pair.into_inner().filter(|p| !is_kw(p.as_rule()));
    let head = inner.next().unwrap();
    if head.as_rule() == Rule::try_expr {
        return parse_try(head);
    }
    let then_ = parse_expr(head);
    let cond = match inner.next() {
        Some(p) => parse_expr(p),
        None => return then_,
    };
    let else_ = parse_expr(inner.next().unwrap());
    Expr::IfElse {
        cond: Box::new(cond),
        then_: Box::new(then_),
        else_: Box::new(else_),
    }
}


/// Parse a `try <expr> else <default>` expression into `Expr::Try`,
/// evaluating `body` and falling back to `default` on any evaluation error.
fn parse_try(pair: Pair<Rule>) -> Expr {
    let mut inner = pair.into_inner().filter(|p| !is_kw(p.as_rule()));
    // The body is wrapped in a try_body rule; unwrap it.
    let body_pair = inner.next().unwrap();
    let body = {
        // try_body contains a single expr child
        let mut bi = body_pair.into_inner();
        parse_expr(bi.next().unwrap())
    };
    let default = parse_expr(inner.next().unwrap());
    Expr::Try {
        body: Box::new(body),
        default: Box::new(default),
    }
}


/// Parse a pipeline expression `base | step1 | step2 …` into `Expr::Pipeline`.
/// Each step is either a forward expression or a `-> pattern` bind target.
/// Returns `base` directly when there are no pipeline steps.
fn parse_pipeline(pair: Pair<Rule>) -> Expr {
    let mut inner = pair.into_inner();
    let base = parse_expr(inner.next().unwrap()); // first child is the base expr
    let mut steps: Vec<PipeStep> = Vec::new();
    for step_pair in inner {
        // Each pipe_step has exactly one inner rule: pipe_forward or pipe_bind.
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
    if steps.is_empty() {
        base
    } else {
        Expr::Pipeline {
            base: Box::new(base),
            steps,
        }
    }
}

/// Parse a bind target for a pipe bind step (`-> name`, `-> {a, b, ..rest}`,
/// or `-> [a, b]`), returning the corresponding `BindTarget` variant.
fn parse_bind_target(pair: Pair<Rule>) -> BindTarget {
    // pair is bind_target; its single inner child determines the variant.
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
                                .to_string(),
                        );
                    }
                    _ => {}
                }
            }
            BindTarget::Obj { fields, rest }
        }
        Rule::bind_arr => {
            let fields: Vec<String> = inner
                .into_inner()
                .filter(|p| p.as_rule() == Rule::ident)
                .map(|p| p.as_str().to_string())
                .collect();
            BindTarget::Arr(fields)
        }
        r => panic!("unexpected bind_target inner: {:?}", r),
    }
}


/// Parse a coalesce expression `a ?? b ?? c` left-associatively into nested
/// `Expr::Coalesce` nodes, returning the first non-null result at runtime.
fn parse_coalesce(pair: Pair<Rule>) -> Expr {
    let mut inner = pair.into_inner();
    let first = parse_expr(inner.next().unwrap());
    inner.fold(first, |acc, rhs| {
        Expr::Coalesce(Box::new(acc), Box::new(parse_expr(rhs)))
    })
}


/// Parse an `or` expression `a or b or c` left-associatively, filtering keyword
/// tokens, into nested `Expr::BinOp(_, BinOp::Or, _)` nodes.
fn parse_or(pair: Pair<Rule>) -> Expr {
    let mut inner = pair.into_inner().filter(|p| !is_kw(p.as_rule()));
    let first = parse_expr(inner.next().unwrap());
    inner.fold(first, |acc, rhs| {
        Expr::BinOp(Box::new(acc), BinOp::Or, Box::new(parse_expr(rhs)))
    })
}

/// Parse an `and` expression left-associatively, filtering keyword tokens,
/// into nested `Expr::BinOp(_, BinOp::And, _)` nodes.
fn parse_and(pair: Pair<Rule>) -> Expr {
    let mut inner = pair.into_inner().filter(|p| !is_kw(p.as_rule()));
    let first = parse_expr(inner.next().unwrap());
    inner.fold(first, |acc, rhs| {
        Expr::BinOp(Box::new(acc), BinOp::And, Box::new(parse_expr(rhs)))
    })
}

/// Parse a `not` expression; wraps the operand in `Expr::Not` when the first
/// token is the `not` keyword, otherwise delegates to the operand directly.
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


/// Parse a `kind is <type>` or `kind is not <type>` type-check expression,
/// returning the bare operand when no kind clause is present.
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
                "null" => KindType::Null,
                "bool" => KindType::Bool,
                "number" => KindType::Number,
                "string" => KindType::Str,
                "array" => KindType::Array,
                "object" => KindType::Object,
                other => panic!("unknown kind type: {}", other),
            };
            Expr::Kind {
                expr: Box::new(cmp),
                ty,
                negate,
            }
        }
        _ => cmp,
    }
}


/// Parse a `contains` / `in` membership test, desugaring it into a call to the
/// `.includes(rhs)` method on the left-hand side. Returns `lhs` when no
/// operator is present.
fn parse_contains(pair: Pair<Rule>) -> Expr {
    let mut inner = pair.into_inner();
    let lhs = parse_expr(inner.next().unwrap());
    match inner.next() {
        None => lhs,
        Some(_op_pair) => {
            let rhs = parse_expr(inner.next().unwrap());
            Expr::Chain(
                Box::new(lhs),
                vec![Step::Method("includes".to_string(), vec![Arg::Pos(rhs)])],
            )
        }
    }
}


/// Parse a comparison expression `lhs op rhs` (`==`, `!=`, `<`, `<=`, `>`,
/// `>=`, `~=`) into `Expr::BinOp`. Returns the bare `lhs` when no operator
/// is present.
fn parse_cmp(pair: Pair<Rule>) -> Expr {
    let mut inner = pair.into_inner();
    let lhs = parse_expr(inner.next().unwrap());
    if let Some(op_pair) = inner.next() {
        let op = match op_pair.as_str() {
            "==" => BinOp::Eq,
            "!=" => BinOp::Neq,
            "<" => BinOp::Lt,
            "<=" => BinOp::Lte,
            ">" => BinOp::Gt,
            ">=" => BinOp::Gte,
            "~=" => BinOp::Fuzzy,
            o => panic!("unknown cmp op: {}", o),
        };
        let rhs = parse_expr(inner.next().unwrap());
        Expr::BinOp(Box::new(lhs), op, Box::new(rhs))
    } else {
        lhs
    }
}


/// Parse additive expressions (`+`, `-`) left-associatively using
/// `parse_left_assoc`.
fn parse_add(pair: Pair<Rule>) -> Expr {
    parse_left_assoc(pair, |s| match s {
        "+" => Some(BinOp::Add),
        "-" => Some(BinOp::Sub),
        _ => None,
    })
}

/// Parse multiplicative expressions (`*`, `/`, `%`) left-associatively using
/// `parse_left_assoc`.
fn parse_mul(pair: Pair<Rule>) -> Expr {
    parse_left_assoc(pair, |s| match s {
        "*" => Some(BinOp::Mul),
        "/" => Some(BinOp::Div),
        "%" => Some(BinOp::Mod),
        _ => None,
    })
}

/// Generic left-associative binary expression builder. Iterates alternating
/// `expr op expr` children; `op_fn` maps operator text to `BinOp`.
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


/// Parse a chain of `as <type>` cast suffixes, building nested `Expr::Cast`
/// nodes left-to-right. Returns the bare operand when no cast is present.
fn parse_cast(pair: Pair<Rule>) -> Expr {
    // cast_expr = { unary_expr ~ (kw_as ~ cast_type)* }
    let mut inner = pair.into_inner().peekable();
    let mut acc = parse_expr(inner.next().unwrap());
    while inner.peek().is_some() {
        // consume the `as` keyword token
        let kw = inner.next().unwrap();
        debug_assert_eq!(kw.as_rule(), Rule::kw_as);
        let ty_pair = inner.next().unwrap();
        let ty = match ty_pair.as_str() {
            "int" => CastType::Int,
            "float" => CastType::Float,
            "number" => CastType::Number,
            "string" => CastType::Str,
            "bool" => CastType::Bool,
            "array" => CastType::Array,
            "object" => CastType::Object,
            "null" => CastType::Null,
            o => panic!("unknown cast type: {}", o),
        };
        acc = Expr::Cast {
            expr: Box::new(acc),
            ty,
        };
    }
    acc
}


/// Parse a unary negation expression (`-expr`); delegates to `parse_expr` for
/// any rule that is not a `unary_neg` marker.
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


/// Parse a postfix expression: a primary value followed by zero or more
/// postfix steps (field access, method calls, index, slice, `?` optional).
/// Coalesces adjacent `?` quantifiers into `OptField`/`OptMethod` steps and
/// delegates to `classify_chain_write` for write rewrites.
fn parse_postfix_expr(pair: Pair<Rule>) -> Expr {
    let mut inner = pair.into_inner();
    let base = parse_primary(inner.next().unwrap());
    let raw_steps: Vec<Step> = inner.flat_map(parse_postfix_step).collect();

    // Merge `?` quantifiers into the preceding step by converting Field → OptField
    // and Method → OptMethod; bare quantifiers with no suitable predecessor are kept.
    let mut steps: Vec<Step> = Vec::with_capacity(raw_steps.len());
    for s in raw_steps {
        match s {
            Step::Quantifier(QuantifierKind::First) => {
                match steps.last() {
                    Some(Step::Field(_)) => {
                        if let Some(Step::Field(k)) = steps.pop() {
                            steps.push(Step::OptField(k));
                        }
                    }
                    Some(Step::Method(_, _)) => {
                        if let Some(Step::Method(n, a)) = steps.pop() {
                            steps.push(Step::OptMethod(n, a));
                        }
                    }
                    _ => {
                        // No suitable predecessor; discard the quantifier to
                        // avoid silently changing semantics.
                    }
                }
            }
            other => steps.push(other),
        }
    }
    if let Some(rewritten) = classify_chain_write(&base, &steps) {
        return rewritten;
    }
    base.maybe_chain(steps)
}


/// Detect rooted write terminals (`$.path.set(v)` etc.) and rewrite them into
/// `Expr::Patch` nodes. Only fires when `base` is `Expr::Root` and the last
/// step is a recognised terminal write method. Returns `None` for all other
/// expressions, leaving them unchanged.
fn classify_chain_write(base: &Expr, steps: &[Step]) -> Option<Expr> {
    if !matches!(base, Expr::Root) {
        return None;
    }
    let last = steps.last()?;
    let (name, args) = match last {
        Step::Method(n, a) => (n.as_str(), a),
        _ => return None,
    };
    if !is_terminal_write(name) {
        return None;
    }

    let prefix = &steps[..steps.len() - 1];
    let path = match steps_to_path(prefix) {
        Ok(p) => p,
        Err(_) => return None, // complex step in path — fall back to method call
    };

    let op = build_write_op(name, args, path)?;
    Some(Expr::Patch {
        root: Box::new(Expr::Root),
        ops: vec![op],
    })
}

/// Return `true` when `name` is one of the chain-write terminal method names
/// that `classify_chain_write` should rewrite to `Expr::Patch`.
fn is_terminal_write(name: &str) -> bool {
    // `.replace` is intentionally absent — it is the two-arg string builtin.
    matches!(
        name,
        "set" | "modify" | "delete" | "unset" | "merge" | "deep_merge" | "deepMerge"
    )
}

/// Convert a slice of `Step` values (the path prefix before the write
/// terminal) into `PathStep` values, returning an error when an unsupported
/// step type is encountered.
fn steps_to_path(steps: &[Step]) -> Result<Vec<PathStep>, String> {
    let mut out = Vec::with_capacity(steps.len());
    for s in steps {
        match s {
            Step::Field(f) => out.push(PathStep::Field(f.clone())),
            Step::Index(i) => out.push(PathStep::Index(*i)),
            Step::OptField(f) => out.push(PathStep::Field(f.clone())),
            Step::Descendant(f) => out.push(PathStep::Descendant(f.clone())),
            Step::DynIndex(e) => {
                // Dynamic index expressions are supported in patch paths for
                // computed keys, e.g. `$.items[$i].set(v)`.
                out.push(PathStep::DynIndex((**e).clone()));
            }
            _ => return Err("chain-write: unsupported step in path".into()),
        }
    }
    Ok(out)
}

/// Build the `PatchOp` for a terminal write method, encoding the write
/// semantics: `set` → value write, `modify` → lambda rewrite, `delete` →
/// `DeleteMark`, `unset` → child `DeleteMark`, `merge`/`deep_merge` →
/// method call on current.
fn build_write_op(name: &str, args: &[Arg], path: Vec<PathStep>) -> Option<PatchOp> {
    match name {
        "set" => {
            let v = arg_expr(args.first()?).clone();
            Some(PatchOp {
                path,
                val: v,
                cond: None,
            })
        }
        // `.modify(|x| expr)` desugars the lambda into a `let` binding so
        // the current value is accessible as the named parameter inside `expr`.
        "modify" => {
            let v = match arg_expr(args.first()?).clone() {
                Expr::Lambda { params, body } => {
                    if let Some(p) = params.into_iter().next() {
                        Expr::Let {
                            name: p,
                            init: Box::new(Expr::Current),
                            body,
                        }
                    } else {
                        *body
                    }
                }
                other => other,
            };
            Some(PatchOp {
                path,
                val: v,
                cond: None,
            })
        }
        "delete" => {
            if !args.is_empty() {
                return None;
            }
            Some(PatchOp {
                path,
                val: Expr::DeleteMark,
                cond: None,
            })
        }
        // `.merge(obj)` and `.deep_merge(obj)` wrap the arg in a method call
        // on the current value so the patch engine can apply the merge in place.
        "merge" | "deep_merge" | "deepMerge" => {
            let arg = arg_expr(args.first()?).clone();
            let method = if name == "merge" {
                "merge".to_string()
            } else {
                "deep_merge".to_string()
            };
            let v = Expr::Chain(
                Box::new(Expr::Current),
                vec![Step::Method(method, vec![Arg::Pos(arg)])],
            );
            Some(PatchOp {
                path,
                val: v,
                cond: None,
            })
        }
        // `.unset(key)` appends the key as a `PathStep::Field` and marks it
        // for deletion, equivalent to `patch $ { path.key: DELETE }`.
        "unset" => {
            let key = match arg_expr(args.first()?) {
                Expr::Str(s) => s.clone(),
                Expr::Ident(s) => s.clone(),
                _ => return None,
            };
            let mut p = path;
            p.push(PathStep::Field(key));
            Some(PatchOp {
                path: p,
                val: Expr::DeleteMark,
                cond: None,
            })
        }
        _ => None,
    }
}

/// Extract the expression from a positional or named `Arg`, unwrapping the
/// outer `Arg` wrapper.
fn arg_expr(a: &Arg) -> &Expr {
    match a {
        Arg::Pos(e) | Arg::Named(_, e) => e,
    }
}

/// Parse a single postfix step from a `postfix_step` rule pair, returning a
/// `Vec<Step>` because `map_into_shape` can expand to two steps.
fn parse_postfix_step(pair: Pair<Rule>) -> Vec<Step> {
    let inner_pair = pair.into_inner().next().unwrap();
    match inner_pair.as_rule() {
        Rule::field_access => {
            let name = inner_pair.into_inner().next().unwrap().as_str().to_string();
            vec![Step::Field(name)]
        }
        Rule::descendant => {
            let mut di = inner_pair.into_inner();
            match di.next() {
                Some(p) => vec![Step::Descendant(p.as_str().to_string())],
                None => vec![Step::DescendAll],
            }
        }
        Rule::deep_method => {
            // `$..find(pred)` etc. are parsed here and mapped to `deep_*` method names.
            let mut mi = inner_pair.into_inner();
            let name = mi.next().unwrap().as_str().to_string();
            let args = mi.next().map(parse_arg_list).unwrap_or_default();
            let mapped = match name.as_str() {
                "find" | "find_all" | "findAll" => "deep_find".to_string(),
                "shape" => "deep_shape".to_string(),
                "like" => "deep_like".to_string(),
                other => format!("deep_{}", other),
            };
            vec![Step::Method(mapped, args)]
        }
        Rule::inline_filter => {
            let expr = parse_expr(inner_pair.into_inner().next().unwrap());
            vec![Step::InlineFilter(Box::new(expr))]
        }
        Rule::quantifier => {
            let s = inner_pair.as_str();
            if s.starts_with('!') {
                vec![Step::Quantifier(QuantifierKind::One)]
            } else {
                vec![Step::Quantifier(QuantifierKind::First)]
            }
        }
        Rule::method_call => {
            let mut mi = inner_pair.into_inner();
            let name = mi.next().unwrap().as_str().to_string();
            let args = mi.next().map(parse_arg_list).unwrap_or_default();
            vec![Step::Method(name, args)]
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
            // `[if pred] { body }` desugars to an optional `.filter(pred)` step
            // followed by a `.map(body)` step.
            let mut guard: Option<Expr> = None;
            let mut body: Option<Expr> = None;
            let mut saw_if = false;
            for p in inner_pair.into_inner() {
                match p.as_rule() {
                    Rule::kw_if => saw_if = true,
                    Rule::expr => {
                        if saw_if && guard.is_none() {
                            guard = Some(parse_expr(p));
                        } else {
                            body = Some(parse_expr(p));
                        }
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

/// Parse a bracket access expression (`[n]`, `[a:b]`, `[a:]`, `[:b]`, or a
/// dynamic expression `[expr]`) into the appropriate `Step` variant.
fn parse_bracket(pair: Pair<Rule>) -> Step {
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::idx_only => Step::Index(inner.as_str().parse().unwrap()),
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


/// Parse a primary expression: a literal, `$`, `@`, identifier, `let`, lambda,
/// comprehension, object/array constructor, global call, or patch block.
fn parse_primary(pair: Pair<Rule>) -> Expr {
    let inner = if pair.as_rule() == Rule::primary {
        pair.into_inner().next().unwrap()
    } else {
        pair
    };
    match inner.as_rule() {
        Rule::literal => parse_literal(inner),
        Rule::root => Expr::Root,
        Rule::current => Expr::Current,
        Rule::ident => Expr::Ident(inner.as_str().to_string()),
        Rule::let_expr => parse_let(inner),
        Rule::lambda_expr => parse_lambda(inner),
        Rule::arrow_lambda => parse_arrow_lambda(inner),
        Rule::list_comp => parse_list_comp(inner),
        Rule::dict_comp => parse_dict_comp(inner),
        Rule::set_comp => parse_set_comp(inner),
        Rule::gen_comp => parse_gen_comp(inner),
        Rule::obj_construct => parse_obj(inner),
        Rule::arr_construct => parse_arr(inner),
        Rule::global_call => parse_global_call(inner),
        Rule::expr => parse_expr(inner),
        Rule::patch_block => parse_patch(inner),
        Rule::kw_delete => Expr::DeleteMark,
        r => panic!("unexpected primary rule: {:?}", r),
    }
}


/// Parse a literal token (null, true, false, integer, float, f-string, or
/// quoted string) into the corresponding `Expr` variant.
fn parse_literal(pair: Pair<Rule>) -> Expr {
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::lit_null => Expr::Null,
        Rule::lit_true => Expr::Bool(true),
        Rule::lit_false => Expr::Bool(false),
        Rule::lit_int => Expr::Int(inner.as_str().parse().unwrap()),
        Rule::lit_float => Expr::Float(inner.as_str().parse().unwrap()),
        Rule::lit_fstring => {
            let raw = inner.as_str();
            let content = &raw[2..raw.len() - 1]; // strip `f"` prefix and `"` suffix
            let parts = parse_fstring_content(content);
            Expr::FString(parts)
        }
        Rule::lit_str => {
            let s = inner.into_inner().next().unwrap();
            let raw = s.as_str();
            // Strip surrounding quote characters (always single or double quote).
            Expr::Str(raw[1..raw.len() - 1].to_string())
        }
        r => panic!("unexpected literal rule: {:?}", r),
    }
}


/// Parse the interior of an f-string (`f"…{expr}…"`) into a list of `FStringPart`
/// values. `{{` and `}}` are escape sequences for literal braces.
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
                            '{' => {
                                depth += 1;
                                inner.push(c2);
                            }
                            '}' => {
                                depth -= 1;
                                if depth == 0 {
                                    break;
                                }
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

/// Split an f-string interpolation `{…}` interior at the first top-level `|`
/// (pipe format) or `:` (spec format), returning the expression substring and
/// an optional `FmtSpec`. Depth tracking avoids splitting inside nested braces.
fn split_fstring_interp(inner: &str) -> (&str, Option<FmtSpec>) {
    // Scan at depth 0 only; `(`, `[`, `{` increase depth.
    let mut depth = 0usize;
    let mut pipe_pos: Option<usize> = None;
    let mut colon_pos: Option<usize> = None;
    for (i, c) in inner.char_indices() {
        match c {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => {
                if depth > 0 {
                    depth -= 1;
                }
            }
            '|' if depth == 0 && pipe_pos.is_none() => pipe_pos = Some(i),
            ':' if depth == 0 && colon_pos.is_none() => colon_pos = Some(i),
            _ => {}
        }
    }
    if let Some(p) = pipe_pos {
        return (
            &inner[..p],
            Some(FmtSpec::Pipe(inner[p + 1..].trim().to_string())),
        );
    }
    if let Some(c) = colon_pos {
        return (&inner[..c], Some(FmtSpec::Spec(inner[c + 1..].to_string())));
    }
    (inner, None)
}


/// Parse a `let name = init in body` expression, supporting multiple bindings
/// (`let a = 1, b = 2 in …`) by folding them right-to-left into nested
/// `Expr::Let` nodes.
fn parse_let(pair: Pair<Rule>) -> Expr {
    // Filter out `let` and `in` keywords, then split: all but the last
    // pair are bindings; the last is the body expression.
    let inner: Vec<Pair<Rule>> = pair
        .into_inner()
        .filter(|p| !matches!(p.as_rule(), Rule::kw_let | Rule::kw_in))
        .collect();
    let (bindings, body_pair) = inner.split_at(inner.len() - 1);
    let body = parse_expr(body_pair[0].clone());
    bindings.iter().rev().fold(body, |acc, b| {
        let mut bi = b.clone().into_inner();
        let name = bi.next().unwrap().as_str().to_string();
        let init = parse_expr(bi.next().unwrap());
        Expr::Let {
            name,
            init: Box::new(init),
            body: Box::new(acc),
        }
    })
}


/// Parse a `lambda params body` expression (keyword-form lambda) into
/// `Expr::Lambda`, collecting parameter identifiers before the body.
fn parse_lambda(pair: Pair<Rule>) -> Expr {
    let mut inner = pair.into_inner().filter(|p| p.as_rule() != Rule::kw_lambda);
    let params_pair = inner.next().unwrap();
    let params: Vec<String> = params_pair
        .into_inner()
        .filter(|p| p.as_rule() == Rule::ident)
        .map(|p| p.as_str().to_string())
        .collect();
    let body = parse_expr(inner.next().unwrap());
    Expr::Lambda {
        params,
        body: Box::new(body),
    }
}


/// Parse an arrow-lambda expression (`(params) => body`) into `Expr::Lambda`,
/// using the same representation as keyword-form lambdas.
fn parse_arrow_lambda(pair: Pair<Rule>) -> Expr {
    let mut inner = pair.into_inner();
    let params_pair = inner.next().unwrap();
    let params: Vec<String> = params_pair
        .into_inner()
        .filter(|p| p.as_rule() == Rule::ident)
        .map(|p| p.as_str().to_string())
        .collect();
    let body = parse_expr(inner.next().unwrap());
    Expr::Lambda {
        params,
        body: Box::new(body),
    }
}


/// Filter keyword tokens (`for`, `in`, `if`) out of a comprehension pair's
/// children, returning only the meaningful sub-expressions and variable lists.
fn comp_inner_filter(pair: Pair<Rule>) -> impl Iterator<Item = Pair<Rule>> {
    pair.into_inner()
        .filter(|p| !matches!(p.as_rule(), Rule::kw_for | Rule::kw_in | Rule::kw_if))
}

/// Collect all `ident` children of a `comp_vars` pair as `Vec<String>`.
fn parse_comp_vars(pair: Pair<Rule>) -> Vec<String> {
    pair.into_inner()
        .filter(|p| p.as_rule() == Rule::ident)
        .map(|p| p.as_str().to_string())
        .collect()
}

/// Parse a list comprehension `[expr for vars in iter if cond]` into
/// `Expr::ListComp`.
fn parse_list_comp(pair: Pair<Rule>) -> Expr {
    let mut inner = comp_inner_filter(pair);
    let expr = parse_expr(inner.next().unwrap());
    let vars = parse_comp_vars(inner.next().unwrap());
    let iter = parse_expr(inner.next().unwrap());
    let cond = inner.next().map(|p| Box::new(parse_expr(p)));
    Expr::ListComp {
        expr: Box::new(expr),
        vars,
        iter: Box::new(iter),
        cond,
    }
}

/// Parse a dict comprehension `{key: val for vars in iter if cond}` into
/// `Expr::DictComp`.
fn parse_dict_comp(pair: Pair<Rule>) -> Expr {
    let mut inner = comp_inner_filter(pair);
    let key = parse_expr(inner.next().unwrap());
    let val = parse_expr(inner.next().unwrap());
    let vars = parse_comp_vars(inner.next().unwrap());
    let iter = parse_expr(inner.next().unwrap());
    let cond = inner.next().map(|p| Box::new(parse_expr(p)));
    Expr::DictComp {
        key: Box::new(key),
        val: Box::new(val),
        vars,
        iter: Box::new(iter),
        cond,
    }
}

/// Parse a set comprehension `{expr for vars in iter if cond}` into
/// `Expr::SetComp`.
fn parse_set_comp(pair: Pair<Rule>) -> Expr {
    let mut inner = comp_inner_filter(pair);
    let expr = parse_expr(inner.next().unwrap());
    let vars = parse_comp_vars(inner.next().unwrap());
    let iter = parse_expr(inner.next().unwrap());
    let cond = inner.next().map(|p| Box::new(parse_expr(p)));
    Expr::SetComp {
        expr: Box::new(expr),
        vars,
        iter: Box::new(iter),
        cond,
    }
}

/// Parse a generator comprehension `(expr for vars in iter if cond)` into
/// `Expr::GenComp`. Semantically identical to `ListComp` but distinct in AST.
fn parse_gen_comp(pair: Pair<Rule>) -> Expr {
    let mut inner = comp_inner_filter(pair);
    let expr = parse_expr(inner.next().unwrap());
    let vars = parse_comp_vars(inner.next().unwrap());
    let iter = parse_expr(inner.next().unwrap());
    let cond = inner.next().map(|p| Box::new(parse_expr(p)));
    Expr::GenComp {
        expr: Box::new(expr),
        vars,
        iter: Box::new(iter),
        cond,
    }
}


/// Parse an object constructor `{ field, … }` into `Expr::Object`, collecting
/// all `obj_field` children via `parse_obj_field`.
fn parse_obj(pair: Pair<Rule>) -> Expr {
    let fields = pair
        .into_inner()
        .filter(|p| p.as_rule() == Rule::obj_field)
        .map(parse_obj_field)
        .collect();
    Expr::Object(fields)
}

/// Parse a single object field entry, dispatching on the field variant:
/// dynamic key-value, optional value, optional shorthand, spread, deep-spread,
/// conditional kv, or plain shorthand name.
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
            ObjField::Kv {
                key,
                val,
                optional: true,
                cond: None,
            }
        }
        Rule::obj_field_opt => {
            let key = obj_key_str(inner.into_inner().next().unwrap());
            ObjField::Kv {
                key: key.clone(),
                val: Expr::Ident(key),
                optional: true,
                cond: None,
            }
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
                        if saw_when {
                            cond = Some(parse_expr(p));
                        } else {
                            val = Some(parse_expr(p));
                        }
                    }
                    _ => {}
                }
            }
            ObjField::Kv {
                key: key.expect("obj_field_kv missing key"),
                val: val.expect("obj_field_kv missing val"),
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

/// Extract the string key from an `obj_key_expr` pair, unwrapping either a
/// bare identifier or a quoted string literal.
fn obj_key_str(pair: Pair<Rule>) -> String {
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::ident => inner.as_str().to_string(),
        Rule::lit_str => {
            let s = inner.into_inner().next().unwrap();
            let raw = s.as_str();
            raw[1..raw.len() - 1].to_string()
        }
        r => panic!("unexpected obj_key_expr rule: {:?}", r),
    }
}

/// Parse an array constructor `[elem, …]` into `Expr::Array`, handling both
/// plain expressions and spread elements (`...expr`).
fn parse_arr(pair: Pair<Rule>) -> Expr {
    let elems = pair
        .into_inner()
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


/// Parse a top-level global function call `name(args)` into
/// `Expr::GlobalCall`, used for functions that are not dot-method syntax
/// (e.g. `coalesce(…)`, `range(…)`).
fn parse_global_call(pair: Pair<Rule>) -> Expr {
    let mut inner = pair.into_inner();
    let name = inner.next().unwrap().as_str().to_string();
    let args = inner.next().map(parse_arg_list).unwrap_or_default();
    Expr::GlobalCall { name, args }
}


/// Parse a `patch root { field: val … }` block into `Expr::Patch`, collecting
/// all `patch_field` operations and the mandatory root expression.
fn parse_patch(pair: Pair<Rule>) -> Expr {
    let mut root: Option<Expr> = None;
    let mut ops: Vec<PatchOp> = Vec::new();
    for p in pair.into_inner() {
        match p.as_rule() {
            Rule::kw_patch => {}
            Rule::patch_field => ops.push(parse_patch_field(p)),
            _ => {
                // The root expression comes first (before any patch_field rules);
                // ignore the kw_patch token by matching it above.
                if root.is_none() {
                    root = Some(parse_expr(p));
                }
            }
        }
    }
    Expr::Patch {
        root: Box::new(root.expect("patch requires root expression")),
        ops,
    }
}

/// Parse a single `field: value [when cond]` entry inside a patch block into
/// a `PatchOp`, extracting the path, value, and optional condition.
fn parse_patch_field(pair: Pair<Rule>) -> PatchOp {
    let mut path: Vec<PathStep> = Vec::new();
    let mut val: Option<Expr> = None;
    let mut cond: Option<Expr> = None;
    let mut saw_when = false;
    for p in pair.into_inner() {
        match p.as_rule() {
            Rule::patch_key => path = parse_patch_key(p),
            Rule::kw_when => saw_when = true,
            Rule::expr => {
                if saw_when {
                    cond = Some(parse_expr(p));
                } else {
                    val = Some(parse_expr(p));
                }
            }
            _ => {}
        }
    }
    PatchOp {
        path,
        val: val.expect("patch_field missing val"),
        cond,
    }
}

/// Parse a patch key (`field.sub[0].*` etc.) into a `Vec<PathStep>`, starting
/// with the mandatory leading identifier and followed by zero or more
/// `patch_step` refinements.
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

/// Parse a single patch path step (`pp_dot_field`, `pp_index`, `pp_wild`,
/// `pp_wild_filter`, or `pp_descendant`) into a `PathStep`.
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
            // Extract the inner filter expression from `[* if expr]`.
            let mut e: Option<Expr> = None;
            for p in inner.into_inner() {
                if p.as_rule() == Rule::expr {
                    e = Some(parse_expr(p));
                }
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


/// Parse an argument list into a `Vec<Arg>`, filtering out separator tokens
/// and mapping each `arg` rule to a positional or named `Arg`.
fn parse_arg_list(pair: Pair<Rule>) -> Vec<Arg> {
    pair.into_inner()
        .filter(|p| p.as_rule() == Rule::arg)
        .map(parse_arg)
        .collect()
}

/// Parse a single argument, returning `Arg::Named(name, expr)` for named args
/// (`key: expr`) and `Arg::Pos(expr)` for positional args.
fn parse_arg(pair: Pair<Rule>) -> Arg {
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::named_arg => {
            let mut i = inner.into_inner();
            let name = i.next().unwrap().as_str().to_string();
            let val = parse_expr(i.next().unwrap());
            Arg::Named(name, val)
        }
        Rule::pos_arg => Arg::Pos(parse_expr(inner.into_inner().next().unwrap())),
        r => panic!("unexpected arg rule: {:?}", r),
    }
}
