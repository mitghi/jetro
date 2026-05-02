use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::Arc;

use crate::{context::EvalError, value::Val};

use super::{NumOp, StageStrategy};

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

pub(crate) fn bounded_sort_by_key<T, I, F>(
    items: I,
    descending: bool,
    strategy: StageStrategy,
    key_of: F,
) -> Result<Vec<T>, EvalError>
where
    I: IntoIterator<Item = T>,
    F: FnMut(&T) -> Result<Val, EvalError>,
{
    bounded_sort_by_key_cmp(items, descending, strategy, key_of, cmp_val_total)
}

pub(crate) fn bounded_sort_by_key_cmp<T, I, F>(
    items: I,
    descending: bool,
    strategy: StageStrategy,
    mut key_of: F,
    cmp: fn(&Val, &Val) -> std::cmp::Ordering,
) -> Result<Vec<T>, EvalError>
where
    I: IntoIterator<Item = T>,
    F: FnMut(&T) -> Result<Val, EvalError>,
{
    let mut sorter = BoundedKeySorter::new(descending, strategy, cmp);
    for item in items {
        let key = key_of(&item)?;
        sorter.push_keyed(key, item);
    }
    Ok(sorter.finish())
}

pub(crate) struct BoundedKeySorter<T> {
    descending: bool,
    limit: Option<usize>,
    keep_largest: bool,
    cmp: fn(&Val, &Val) -> Ordering,
    keyed: Vec<(Val, T)>,
}

impl<T> BoundedKeySorter<T> {
    pub(crate) fn new(
        descending: bool,
        strategy: StageStrategy,
        cmp: fn(&Val, &Val) -> Ordering,
    ) -> Self {
        let k = match strategy {
            StageStrategy::SortTopK(k) | StageStrategy::SortBottomK(k) => Some(k),
            StageStrategy::Default | StageStrategy::SortUntilOutput(_) => None,
        };
        let keep_largest = match strategy {
            StageStrategy::SortTopK(_) => descending,
            StageStrategy::SortBottomK(_) => !descending,
            StageStrategy::Default | StageStrategy::SortUntilOutput(_) => false,
        };

        let capacity = k.unwrap_or(0).saturating_add(1);
        Self {
            descending,
            limit: k,
            keep_largest,
            cmp,
            keyed: Vec::with_capacity(capacity),
        }
    }

    pub(crate) fn push_keyed(&mut self, key: Val, item: T) {
        match self.limit {
            Some(0) => {}
            Some(limit) if self.keyed.len() >= limit => {
                let mut worst_idx = 0usize;
                for (idx, (candidate, _)) in self.keyed.iter().enumerate().skip(1) {
                    let ord = (self.cmp)(candidate, &self.keyed[worst_idx].0);
                    let displace = if self.keep_largest {
                        ord == Ordering::Less
                    } else {
                        ord == Ordering::Greater
                    };
                    if displace {
                        worst_idx = idx;
                    }
                }
                let ord = (self.cmp)(&key, &self.keyed[worst_idx].0);
                let take = if self.keep_largest {
                    ord == Ordering::Greater
                } else {
                    ord == Ordering::Less
                };
                if take {
                    self.keyed[worst_idx] = (key, item);
                }
            }
            Some(_) | None => self.keyed.push((key, item)),
        }
    }

    pub(crate) fn finish(mut self) -> Vec<T> {
        self.keyed.sort_by(|a, b| (self.cmp)(&a.0, &b.0));
        if self.descending {
            self.keyed.reverse();
        }
        self.keyed.into_iter().map(|(_, item)| item).collect()
    }
}

pub(crate) fn ordered_by_key_cmp<T, I, F>(
    items: I,
    descending: bool,
    mut key_of: F,
    cmp: fn(&Val, &Val) -> Ordering,
) -> Result<OrderedByKey<T>, EvalError>
where
    I: IntoIterator<Item = T>,
    F: FnMut(&T) -> Result<Val, EvalError>,
{
    let mut sorter = OrderedKeySorter::new(descending, cmp);
    for item in items {
        let key = key_of(&item)?;
        sorter.push_keyed(key, item);
    }
    Ok(sorter.finish())
}

pub(crate) struct OrderedKeySorter<T> {
    heap: BinaryHeap<OrderedEntry<T>>,
    next_seq: usize,
    descending: bool,
    cmp: fn(&Val, &Val) -> Ordering,
}

impl<T> OrderedKeySorter<T> {
    pub(crate) fn new(descending: bool, cmp: fn(&Val, &Val) -> Ordering) -> Self {
        Self {
            heap: BinaryHeap::new(),
            next_seq: 0,
            descending,
            cmp,
        }
    }

    pub(crate) fn push_keyed(&mut self, key: Val, item: T) {
        let seq = self.next_seq;
        self.next_seq += 1;
        self.heap.push(OrderedEntry {
            key,
            item,
            seq,
            descending: self.descending,
            cmp: self.cmp,
        });
    }

    pub(crate) fn finish(self) -> OrderedByKey<T> {
        OrderedByKey { heap: self.heap }
    }
}

pub(crate) struct OrderedByKey<T> {
    heap: BinaryHeap<OrderedEntry<T>>,
}

impl<T> Iterator for OrderedByKey<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.heap.pop().map(|entry| entry.item)
    }
}

struct OrderedEntry<T> {
    key: Val,
    item: T,
    seq: usize,
    descending: bool,
    cmp: fn(&Val, &Val) -> Ordering,
}

impl<T> PartialEq for OrderedEntry<T> {
    fn eq(&self, other: &Self) -> bool {
        self.seq == other.seq && (self.cmp)(&self.key, &other.key) == Ordering::Equal
    }
}

impl<T> Eq for OrderedEntry<T> {}

impl<T> PartialOrd for OrderedEntry<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for OrderedEntry<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        let key_order = (self.cmp)(&self.key, &other.key);
        let priority = if self.descending {
            key_order
        } else {
            key_order.reverse()
        };
        priority.then_with(|| other.seq.cmp(&self.seq))
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
