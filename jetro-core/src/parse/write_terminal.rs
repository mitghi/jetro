use super::ast::{Arg, Expr, PatchOp, PathStep, Step};

/// Chain-write terminals that lower to patch/update AST nodes.
///
/// `.replace` is intentionally absent because it is the two-arg string
/// builtin, not a chain-write terminal.
pub(crate) fn is_chain_write_terminal(name: &str) -> bool {
    is_patch_write_terminal(name) || name == "update"
}

/// Chain-write terminals that can be represented as a single [`PatchOp`].
pub(crate) fn is_patch_write_terminal(name: &str) -> bool {
    matches!(
        name,
        "set" | "modify" | "delete" | "unset" | "merge" | "deep_merge" | "deepMerge"
    )
}

/// Chain-write terminals that pipeline fusion may speculatively lift.
///
/// Merge-style terminals remain parser-lowered for rooted chains only until
/// pipe-stage semantics are covered by focused tests.
pub(crate) fn is_pipeline_fusion_terminal(name: &str) -> bool {
    matches!(name, "set" | "modify" | "delete" | "unset")
}

/// Convert chain steps into patch path steps.
///
/// Wildcards and inline filters are valid for rooted parser rewrites. Planner
/// fusion uses the same lowering conservatively with `allow_selective = false`
/// so it does not speculate across per-element selectors it cannot yet batch.
pub(crate) fn steps_to_path(
    steps: &[Step],
    allow_selective: bool,
) -> Option<Vec<PathStep>> {
    let mut out = Vec::with_capacity(steps.len());
    for step in steps {
        match step {
            Step::Field(field) | Step::OptField(field) => {
                out.push(PathStep::Field(field.clone()))
            }
            Step::Index(index) => out.push(PathStep::Index(*index)),
            Step::Descendant(field) => out.push(PathStep::Descendant(field.clone())),
            Step::DynIndex(expr) => out.push(PathStep::DynIndex((**expr).clone())),
            Step::Wildcard if allow_selective => out.push(PathStep::Wildcard),
            Step::InlineFilter(expr) if allow_selective => {
                out.push(PathStep::WildcardFilter(expr.clone()))
            }
            _ => return None,
        }
    }
    Some(out)
}

/// Build the patch operation for a chain-write terminal.
pub(crate) fn build_patch_op(name: &str, args: &[Arg], path: Vec<PathStep>) -> Option<PatchOp> {
    match name {
        "set" => Some(PatchOp {
            path,
            val: arg_expr(args.first()?).clone(),
            cond: None,
        }),
        "modify" => {
            let val = match arg_expr(args.first()?).clone() {
                Expr::Lambda { params, body } => {
                    if let Some(param) = params.into_iter().next() {
                        Expr::Let {
                            name: param,
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
                val,
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
        "merge" | "deep_merge" | "deepMerge" => {
            let arg = arg_expr(args.first()?).clone();
            let method = if name == "merge" {
                "merge"
            } else {
                "deep_merge"
            };
            Some(PatchOp {
                path,
                val: Expr::Chain(
                    Box::new(Expr::Current),
                    vec![Step::Method(method.to_string(), vec![Arg::Pos(arg)])],
                ),
                cond: None,
            })
        }
        "unset" => {
            let key = match arg_expr(args.first()?) {
                Expr::Str(key) | Expr::Ident(key) => key.clone(),
                _ => return None,
            };
            let mut path = path;
            path.push(PathStep::Field(key));
            Some(PatchOp {
                path,
                val: Expr::DeleteMark,
                cond: None,
            })
        }
        _ => None,
    }
}

fn arg_expr(arg: &Arg) -> &Expr {
    match arg {
        Arg::Pos(expr) | Arg::Named(_, expr) => expr,
    }
}
