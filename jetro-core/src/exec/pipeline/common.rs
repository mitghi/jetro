//! Shared primitives reused across all pipeline execution paths:
//! numeric fold helpers, bounded sort, total-order comparators, and the
//! `BoundedKeySorter` / `OrderedKeySorter` for `sort_by` and `group_by`.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::Arc;

use crate::{data::context::EvalError, data::value::Val};

use super::{NumOp, StageStrategy};

/// Accumulates one numeric `Val` into the running aggregate state; promotes integer accumulators to `f64` on the first float.
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

/// Converts the running aggregate state from `num_fold` into a final `Val`, returning `op.empty()` when no observations were made.
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


/// Total-order comparator for `Val`, promoting mixed numeric types to `f64` and falling back to debug-string comparison.
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

/// Sorts `items` by key using `cmp_val_total`, delegating to `bounded_sort_by_key_cmp` for top-K heap optimisation.
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

/// Sorts `items` using caller-supplied `key_of` and `cmp`, restricting memory to top-K elements when the strategy demands it.
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

/// Key-based sorter that caps memory to top-K or bottom-K entries via a `BinaryHeap`; degrades to a plain vec sort when unbounded.
pub(crate) struct BoundedKeySorter<T> {
    // Direction of the final sort; reversed relative to heap priority order.
    descending: bool,
    // Maximum number of entries to retain; `None` means keep all.
    limit: Option<usize>,
    // When `true`, the heap evicts the smallest entry when over capacity; otherwise the largest.
    keep_largest: bool,
    // Caller-supplied comparator for `Val` keys.
    cmp: fn(&Val, &Val) -> Ordering,
    // Accumulator used when no limit is active.
    keyed: Vec<(Val, usize, T)>,
    // Bounded heap used when a limit is active.
    heap: BinaryHeap<BoundedEntry<T>>,
    // Monotonically increasing sequence counter for stable ordering.
    next_seq: usize,
}

impl<T> BoundedKeySorter<T> {
    /// Constructs a `BoundedKeySorter`; `SortTopK`/`SortBottomK` activate heap mode, `Default` uses a plain vec.
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
            keyed: Vec::with_capacity(if k.is_none() { capacity } else { 0 }),
            heap: BinaryHeap::with_capacity(k.unwrap_or(0)),
            next_seq: 0,
        }
    }

    /// Inserts `item` with `key`; in heap mode evicts the current worst entry when at capacity and the new key is better.
    pub(crate) fn push_keyed(&mut self, key: Val, item: T) {
        let seq = self.next_seq;
        self.next_seq += 1;
        match self.limit {
            Some(0) => {}
            Some(limit) if self.heap.len() >= limit => {
                let Some(worst) = self.heap.peek() else {
                    return;
                };
                let ord = (self.cmp)(&key, &worst.key);
                let should_replace = if self.keep_largest {
                    ord == Ordering::Greater
                } else {
                    ord == Ordering::Less
                };
                if should_replace {
                    let _ = self.heap.pop();
                    self.heap.push(BoundedEntry {
                        key,
                        item,
                        seq,
                        keep_largest: self.keep_largest,
                        cmp: self.cmp,
                    });
                }
            }
            Some(_) => {
                self.heap.push(BoundedEntry {
                    key,
                    item,
                    seq,
                    keep_largest: self.keep_largest,
                    cmp: self.cmp,
                });
            }
            None => self.keyed.push((key, seq, item)),
        }
    }

    /// Drains the heap or vec, sorts retained entries stably by sequence, and returns items in final order.
    pub(crate) fn finish(mut self) -> Vec<T> {
        if self.limit.is_some() {
            self.keyed = self
                .heap
                .into_vec()
                .into_iter()
                .map(|entry| (entry.key, entry.seq, entry.item))
                .collect();
        }
        let cmp = self.cmp;
        self.keyed.sort_by(|a, b| {
            let order = cmp(&a.0, &b.0);
            let order = if self.descending {
                order.reverse()
            } else {
                order
            };
            order.then_with(|| a.1.cmp(&b.1))
        });
        self.keyed.into_iter().map(|(_, _, item)| item).collect()
    }
}

// Heap entry for `BoundedKeySorter`; ordering is inverted so `BinaryHeap::pop` removes the least-desirable entry.
struct BoundedEntry<T> {
    key: Val,
    item: T,
    // Insertion sequence for stable ordering among equal keys.
    seq: usize,
    // When `true`, the heap is a max-heap (evicts smallest); otherwise min-heap.
    keep_largest: bool,
    cmp: fn(&Val, &Val) -> Ordering,
}

impl<T> PartialEq for BoundedEntry<T> {
    fn eq(&self, other: &Self) -> bool {
        self.seq == other.seq && (self.cmp)(&self.key, &other.key) == Ordering::Equal
    }
}

impl<T> Eq for BoundedEntry<T> {}

impl<T> PartialOrd for BoundedEntry<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for BoundedEntry<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        let key_order = (self.cmp)(&self.key, &other.key);
        let priority = if self.keep_largest {
            key_order.reverse()
        } else {
            key_order
        };
        priority.then_with(|| self.seq.cmp(&other.seq))
    }
}

/// Sorts `items` via `key_of` and `cmp`, returning a lazy `OrderedByKey` iterator backed by a `BinaryHeap`.
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

/// Collects `(key, item)` pairs into a `BinaryHeap` so `finish` can serve them via `OrderedByKey` for lazy sorted pulls.
pub(crate) struct OrderedKeySorter<T> {
    heap: BinaryHeap<OrderedEntry<T>>,
    // Monotonically increasing insertion sequence for stable ordering.
    next_seq: usize,
    // Output order (`true` = largest first).
    descending: bool,
    cmp: fn(&Val, &Val) -> Ordering,
}

impl<T> OrderedKeySorter<T> {
    /// Creates a new `OrderedKeySorter` with the given sort direction and key comparator.
    pub(crate) fn new(descending: bool, cmp: fn(&Val, &Val) -> Ordering) -> Self {
        Self {
            heap: BinaryHeap::new(),
            next_seq: 0,
            descending,
            cmp,
        }
    }

    /// Inserts `item` associated with `key` into the heap.
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

    /// Converts the sorter into a lazy `OrderedByKey` iterator over the heap.
    pub(crate) fn finish(self) -> OrderedByKey<T> {
        OrderedByKey { heap: self.heap }
    }
}

/// Lazy iterator that extracts items from a `BinaryHeap` in sorted order, created by `OrderedKeySorter::finish`.
pub(crate) struct OrderedByKey<T> {
    heap: BinaryHeap<OrderedEntry<T>>,
}

impl<T> Iterator for OrderedByKey<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.heap.pop().map(|entry| entry.item)
    }
}

// Internal heap entry for `OrderedKeySorter`; ordering ensures `BinaryHeap::pop` always yields the next sorted item.
struct OrderedEntry<T> {
    key: Val,
    item: T,
    // Insertion sequence number for stable ordering.
    seq: usize,
    // `true` when larger keys should appear first in the output.
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


/// Traverses `keys` on `root`, returning the nested value or `Val::Null` when any step yields a missing field.
pub(crate) fn walk_field_chain(root: &Val, keys: &[Arc<str>]) -> Val {
    let mut cur = root.clone();
    for k in keys {
        cur = cur.get_field(k.as_ref());
    }
    cur
}


/// Sets `item` as the current value in `env`, executes `prog`, then restores the previous current value.
#[inline]
pub(crate) fn apply_item_in_env(
    vm: &mut crate::vm::VM,
    env: &mut crate::data::context::Env,
    item: &Val,
    prog: &crate::vm::Program,
) -> Result<Val, EvalError> {
    let prev = env.swap_current(item.clone());
    let r = vm.exec_in_env(prog, env);
    let _ = env.swap_current(prev);
    r
}

/// Returns `true` when `v` is truthy (non-null, non-false, non-zero, non-empty).
#[inline]
pub(crate) fn is_truthy(v: &Val) -> bool {
    crate::util::is_truthy(v)
}
