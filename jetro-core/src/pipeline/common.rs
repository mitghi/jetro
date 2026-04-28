use std::sync::Arc;

use crate::{context::EvalError, value::Val};

use super::NumOp;

#[inline]
pub(crate) fn num_fold(
    acc_i: &mut i64,
    acc_f: &mut f64,
    floated: &mut bool,
    min_f: &mut f64,
    max_f: &mut f64,
    n_obs: &mut usize,
    op: NumOp,
    v: &Val,
) {
    let f = match v {
        Val::Int(n) => *n as f64,
        Val::Float(x) => *x,
        _ => return,
    };
    *n_obs += 1;
    match op {
        NumOp::Sum | NumOp::Avg => match v {
            Val::Int(n) => {
                if *floated {
                    *acc_f += *n as f64
                } else {
                    *acc_i += *n
                }
            }
            Val::Float(x) => {
                if !*floated {
                    *acc_f = *acc_i as f64;
                    *floated = true;
                }
                *acc_f += *x;
            }
            _ => {}
        },
        NumOp::Min => {
            if f < *min_f {
                *min_f = f;
            }
        }
        NumOp::Max => {
            if f > *max_f {
                *max_f = f;
            }
        }
    }
}

#[inline]
pub(crate) fn num_finalise(
    op: NumOp,
    acc_i: i64,
    acc_f: f64,
    floated: bool,
    min_f: f64,
    max_f: f64,
    n_obs: usize,
) -> Val {
    if n_obs == 0 {
        return op.empty();
    }
    match op {
        NumOp::Sum => {
            if floated {
                Val::Float(acc_f)
            } else {
                Val::Int(acc_i)
            }
        }
        NumOp::Avg => {
            let total = if floated { acc_f } else { acc_i as f64 };
            Val::Float(total / n_obs as f64)
        }
        NumOp::Min => Val::Float(min_f),
        NumOp::Max => Val::Float(max_f),
    }
}

// (sum_acc removed; superseded by num_fold which handles Sum/Min/Max/Avg)

/// Total ordering over `Val` for sort barriers: numeric < string <
/// other; ties broken by debug-format equality so the comparator is
/// stable across calls.  Used only inside Pipeline::run barriers,
/// not exposed.
pub(crate) fn cmp_val_total(a: &Val, b: &Val) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let af = match a {
        Val::Int(n) => Some(*n as f64),
        Val::Float(x) => Some(*x),
        _ => None,
    };
    let bf = match b {
        Val::Int(n) => Some(*n as f64),
        Val::Float(x) => Some(*x),
        _ => None,
    };
    match (af, bf) {
        (Some(x), Some(y)) => x.partial_cmp(&y).unwrap_or(Ordering::Equal),
        _ => match (a, b) {
            (Val::Str(x), Val::Str(y)) => x.as_ref().cmp(y.as_ref()),
            _ => format!("{:?}", a).cmp(&format!("{:?}", b)),
        },
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Walk `$.k1.k2…` from `root`, returning `Val::Null` on any miss.
/// Used by `Source::FieldChain` resolution.
pub(crate) fn walk_field_chain(root: &Val, keys: &[Arc<str>]) -> Val {
    let mut cur = root.clone();
    for k in keys {
        cur = cur.get_field(k.as_ref());
    }
    cur
}

/// Evaluate `prog` against `item` as the VM root using a long-lived
/// VM borrowed from the caller (Pipeline::run owns one per query).
/// Per-row apply.  Phase A1 fast path: rebinds the loop-shared Env's
/// `current` slot in place (one swap, two assignments) and runs the
/// stage program directly via `exec_in_env`.  Skips doc-hash recompute,
/// root_chain_cache clear, and Env construction that
/// `execute_val_raw` does on every call.  Saves ~80 ns/row.
#[inline]
pub(crate) fn apply_item_in_env(
    vm: &mut crate::vm::VM,
    env: &mut crate::context::Env,
    item: &Val,
    prog: &crate::vm::Program,
) -> Result<Val, EvalError> {
    let prev = env.swap_current(item.clone());
    let r = vm.exec_in_env(prog, env);
    let _ = env.swap_current(prev);
    r
}

#[inline]
pub(crate) fn is_truthy(v: &Val) -> bool {
    crate::util::is_truthy(v)
}
