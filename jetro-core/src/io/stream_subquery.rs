use super::stream_plan::{
    lower_root_rows_expr, root_rows_stream_prefix_len, RowStreamPlan, RowStreamPlanError,
    RowStreamSourceKind,
};
use crate::parse::ast::{Arg, ArrayElem, Expr, FStringPart, MatchArm, ObjField};

pub(super) const STREAM_BINDING: &str = "__jetro_rows_stream_0";

#[derive(Clone, Debug)]
pub(crate) struct RowStreamSubqueryPlan {
    pub stream: RowStreamPlan,
    pub wrapper: Expr,
}

pub(super) fn lower_single_rows_subquery(
    expr: &Expr,
    source: RowStreamSourceKind,
) -> Result<Option<RowStreamSubqueryPlan>, RowStreamPlanError> {
    let mut lifted = None;
    let wrapper = replace_rows_expr(expr, source, &mut lifted)?;
    Ok(lifted.map(|stream| RowStreamSubqueryPlan { stream, wrapper }))
}

fn replace_rows_expr(
    expr: &Expr,
    source: RowStreamSourceKind,
    lifted: &mut Option<RowStreamPlan>,
) -> Result<Expr, RowStreamPlanError> {
    if let Some((stream, replacement)) = lift_root_rows_prefix(expr, source)? {
        if lifted.is_some() {
            return Err(RowStreamPlanError::new(
                "multiple $.rows() stream subqueries are not supported",
            ));
        }
        *lifted = Some(stream);
        return Ok(replacement);
    }

    if let Some(stream) = lower_root_rows_expr(expr, source)? {
        if lifted.is_some() {
            return Err(RowStreamPlanError::new(
                "multiple $.rows() stream subqueries are not supported",
            ));
        }
        *lifted = Some(stream);
        return Ok(Expr::Ident(STREAM_BINDING.to_string()));
    }

    Ok(match expr {
        Expr::Chain(base, steps) => Expr::Chain(
            Box::new(replace_rows_expr(base, source, lifted)?),
            replace_steps(steps, source, lifted)?,
        ),
        Expr::BinOp(left, op, right) => Expr::BinOp(
            Box::new(replace_rows_expr(left, source, lifted)?),
            *op,
            Box::new(replace_rows_expr(right, source, lifted)?),
        ),
        Expr::UnaryNeg(inner) => {
            Expr::UnaryNeg(Box::new(replace_rows_expr(inner, source, lifted)?))
        }
        Expr::Not(inner) => Expr::Not(Box::new(replace_rows_expr(inner, source, lifted)?)),
        Expr::Kind { expr, ty, negate } => Expr::Kind {
            expr: Box::new(replace_rows_expr(expr, source, lifted)?),
            ty: *ty,
            negate: *negate,
        },
        Expr::Coalesce(left, right) => Expr::Coalesce(
            Box::new(replace_rows_expr(left, source, lifted)?),
            Box::new(replace_rows_expr(right, source, lifted)?),
        ),
        Expr::Object(fields) => Expr::Object(
            fields
                .iter()
                .map(|field| replace_obj_field(field, source, lifted))
                .collect::<Result<_, _>>()?,
        ),
        Expr::Array(elems) => Expr::Array(
            elems
                .iter()
                .map(|elem| replace_array_elem(elem, source, lifted))
                .collect::<Result<_, _>>()?,
        ),
        Expr::Let { name, init, body } => Expr::Let {
            name: name.clone(),
            init: Box::new(replace_rows_expr(init, source, lifted)?),
            body: Box::new(replace_rows_expr(body, source, lifted)?),
        },
        Expr::IfElse { cond, then_, else_ } => Expr::IfElse {
            cond: Box::new(replace_rows_expr(cond, source, lifted)?),
            then_: Box::new(replace_rows_expr(then_, source, lifted)?),
            else_: Box::new(replace_rows_expr(else_, source, lifted)?),
        },
        Expr::Try { body, default } => Expr::Try {
            body: Box::new(replace_rows_expr(body, source, lifted)?),
            default: Box::new(replace_rows_expr(default, source, lifted)?),
        },
        Expr::GlobalCall { name, args } => Expr::GlobalCall {
            name: name.clone(),
            args: replace_args(args, source, lifted)?,
        },
        Expr::Cast { expr, ty } => Expr::Cast {
            expr: Box::new(replace_rows_expr(expr, source, lifted)?),
            ty: *ty,
        },
        Expr::Match { scrutinee, arms } => Expr::Match {
            scrutinee: Box::new(replace_rows_expr(scrutinee, source, lifted)?),
            arms: arms
                .iter()
                .map(|arm| replace_match_arm(arm, source, lifted))
                .collect::<Result<_, _>>()?,
        },
        Expr::FString(parts) => Expr::FString(
            parts
                .iter()
                .map(|part| replace_fstring_part(part, source, lifted))
                .collect::<Result<_, _>>()?,
        ),
        _ => expr.clone(),
    })
}

fn lift_root_rows_prefix(
    expr: &Expr,
    source: RowStreamSourceKind,
) -> Result<Option<(RowStreamPlan, Expr)>, RowStreamPlanError> {
    let Expr::Chain(base, steps) = expr else {
        return Ok(None);
    };
    if !matches!(base.as_ref(), Expr::Root) {
        return Ok(None);
    }
    let Some(split) = root_rows_stream_prefix_len(expr) else {
        return Ok(None);
    };
    if split == steps.len() {
        return Ok(None);
    }

    let prefix = Expr::Chain(Box::new(Expr::Root), steps[..split].to_vec());
    let Some(stream) = lower_root_rows_expr(&prefix, source)? else {
        return Ok(None);
    };
    let suffix = steps[split..].to_vec();
    let replacement = if suffix.is_empty() {
        Expr::Ident(STREAM_BINDING.to_string())
    } else {
        Expr::Chain(Box::new(Expr::Ident(STREAM_BINDING.to_string())), suffix)
    };
    Ok(Some((stream, replacement)))
}

fn replace_steps(
    steps: &[crate::parse::ast::Step],
    source: RowStreamSourceKind,
    lifted: &mut Option<RowStreamPlan>,
) -> Result<Vec<crate::parse::ast::Step>, RowStreamPlanError> {
    steps
        .iter()
        .map(|step| {
            Ok(match step {
                crate::parse::ast::Step::DynIndex(expr) => crate::parse::ast::Step::DynIndex(
                    Box::new(replace_rows_expr(expr, source, lifted)?),
                ),
                crate::parse::ast::Step::InlineFilter(expr) => {
                    crate::parse::ast::Step::InlineFilter(Box::new(replace_rows_expr(
                        expr, source, lifted,
                    )?))
                }
                crate::parse::ast::Step::Method(name, args) => crate::parse::ast::Step::Method(
                    name.clone(),
                    replace_args(args, source, lifted)?,
                ),
                crate::parse::ast::Step::OptMethod(name, args) => {
                    crate::parse::ast::Step::OptMethod(
                        name.clone(),
                        replace_args(args, source, lifted)?,
                    )
                }
                _ => step.clone(),
            })
        })
        .collect()
}

fn replace_args(
    args: &[Arg],
    source: RowStreamSourceKind,
    lifted: &mut Option<RowStreamPlan>,
) -> Result<Vec<Arg>, RowStreamPlanError> {
    args.iter()
        .map(|arg| match arg {
            Arg::Pos(expr) => Ok(Arg::Pos(replace_rows_expr(expr, source, lifted)?)),
            Arg::Named(name, expr) => Ok(Arg::Named(
                name.clone(),
                replace_rows_expr(expr, source, lifted)?,
            )),
        })
        .collect()
}

fn replace_obj_field(
    field: &ObjField,
    source: RowStreamSourceKind,
    lifted: &mut Option<RowStreamPlan>,
) -> Result<ObjField, RowStreamPlanError> {
    Ok(match field {
        ObjField::Kv {
            key,
            val,
            optional,
            cond,
        } => ObjField::Kv {
            key: key.clone(),
            val: replace_rows_expr(val, source, lifted)?,
            optional: *optional,
            cond: cond
                .as_ref()
                .map(|cond| replace_rows_expr(cond, source, lifted))
                .transpose()?,
        },
        ObjField::Dynamic { key, val } => ObjField::Dynamic {
            key: replace_rows_expr(key, source, lifted)?,
            val: replace_rows_expr(val, source, lifted)?,
        },
        _ => field.clone(),
    })
}

fn replace_array_elem(
    elem: &ArrayElem,
    source: RowStreamSourceKind,
    lifted: &mut Option<RowStreamPlan>,
) -> Result<ArrayElem, RowStreamPlanError> {
    Ok(match elem {
        ArrayElem::Expr(expr) => ArrayElem::Expr(replace_rows_expr(expr, source, lifted)?),
        ArrayElem::Spread(expr) => ArrayElem::Spread(replace_rows_expr(expr, source, lifted)?),
    })
}

fn replace_match_arm(
    arm: &MatchArm,
    source: RowStreamSourceKind,
    lifted: &mut Option<RowStreamPlan>,
) -> Result<MatchArm, RowStreamPlanError> {
    Ok(MatchArm {
        pat: arm.pat.clone(),
        guard: arm
            .guard
            .as_ref()
            .map(|guard| replace_rows_expr(guard, source, lifted))
            .transpose()?,
        body: replace_rows_expr(&arm.body, source, lifted)?,
    })
}

fn replace_fstring_part(
    part: &FStringPart,
    source: RowStreamSourceKind,
    lifted: &mut Option<RowStreamPlan>,
) -> Result<FStringPart, RowStreamPlanError> {
    Ok(match part {
        FStringPart::Interp { expr, fmt } => FStringPart::Interp {
            expr: replace_rows_expr(expr, source, lifted)?,
            fmt: fmt.clone(),
        },
        FStringPart::Lit(_) => part.clone(),
    })
}
