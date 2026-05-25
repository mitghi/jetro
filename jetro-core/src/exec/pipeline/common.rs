//! Shared primitives reused across all pipeline execution paths:
//! numeric fold helpers, bounded sort, total-order comparators, and the
//! `BoundedKeySorter` / `OrderedKeySorter` for `sort_by` and `group_by`.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::Arc;

use crate::{data::context::EvalError, data::value::Val, util::JsonView};

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
    match v {
        Val::Int(n) => num_fold_i64(acc_i, acc_f, floated, min_f, max_f, n_obs, op, *n),
        Val::Float(x) => num_fold_f64(acc_i, acc_f, floated, min_f, max_f, n_obs, op, *x),
        _ => return,
    }
}

/// Accumulates one borrowed JSON scalar into numeric aggregate state without
/// allocating a temporary `Val`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn num_fold_json_view(
    acc_i: &mut i64,
    acc_f: &mut f64,
    floated: &mut bool,
    min_f: &mut f64,
    max_f: &mut f64,
    n_obs: &mut usize,
    op: NumOp,
    scalar: JsonView<'_>,
) {
    match scalar {
        JsonView::Int(value) => num_fold_i64(acc_i, acc_f, floated, min_f, max_f, n_obs, op, value),
        JsonView::UInt(value) if value <= i64::MAX as u64 => num_fold_i64(
            acc_i,
            acc_f,
            floated,
            min_f,
            max_f,
            n_obs,
            op,
            value as i64,
        ),
        JsonView::UInt(value) => num_fold_f64(
            acc_i,
            acc_f,
            floated,
            min_f,
            max_f,
            n_obs,
            op,
            value as f64,
        ),
        JsonView::Float(value) => {
            num_fold_f64(acc_i, acc_f, floated, min_f, max_f, n_obs, op, value)
        }
        _ => {}
    }
}

/// Accumulates one integer value into numeric aggregate state.
#[inline]
pub(crate) fn num_fold_i64(
    acc_i: &mut i64,
    acc_f: &mut f64,
    floated: &mut bool,
    min_f: &mut f64,
    max_f: &mut f64,
    n_obs: &mut usize,
    op: NumOp,
    n: i64,
) {
    *n_obs += 1;
    match op {
        NumOp::Sum | NumOp::Avg => {
            if *floated {
                *acc_f += n as f64
            } else {
                *acc_i += n
            }
        }
        NumOp::Min => {
            if *n_obs == 1 || n < *acc_i {
                *acc_i = n;
            }
            let f = n as f64;
            if f < *min_f {
                *min_f = f;
            }
        }
        NumOp::Max => {
            if *n_obs == 1 || n > *acc_i {
                *acc_i = n;
            }
            let f = n as f64;
            if f > *max_f {
                *max_f = f;
            }
        }
    }
}

/// Accumulates one floating-point value into numeric aggregate state.
#[inline]
pub(crate) fn num_fold_f64(
    acc_i: &mut i64,
    acc_f: &mut f64,
    floated: &mut bool,
    min_f: &mut f64,
    max_f: &mut f64,
    n_obs: &mut usize,
    op: NumOp,
    x: f64,
) {
    *n_obs += 1;
    match op {
        NumOp::Sum | NumOp::Avg => {
            if !*floated {
                *acc_f = *acc_i as f64;
                *floated = true;
            }
            *acc_f += x;
        }
        NumOp::Min => {
            *floated = true;
            if x < *min_f {
                *min_f = x;
            }
        }
        NumOp::Max => {
            *floated = true;
            if x > *max_f {
                *max_f = x;
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
        NumOp::Min => {
            if floated {
                Val::Float(min_f)
            } else {
                Val::Int(acc_i)
            }
        }
        NumOp::Max => {
            if floated {
                Val::Float(max_f)
            } else {
                Val::Int(acc_i)
            }
        }
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
pub(crate) struct BoundedKeySorter<T, K = Val> {
    // Direction of the final sort; reversed relative to heap priority order.
    descending: bool,
    // Maximum number of entries to retain; `None` means keep all.
    limit: Option<usize>,
    // When `true`, the heap evicts the smallest entry when over capacity; otherwise the largest.
    keep_largest: bool,
    // Caller-supplied comparator for sort keys.
    cmp: fn(&K, &K) -> Ordering,
    // Accumulator used when no limit is active.
    keyed: Vec<(K, usize, T)>,
    // Bounded heap used when a limit is active.
    heap: BinaryHeap<BoundedEntry<T, K>>,
    // Monotonically increasing sequence counter for stable ordering.
    next_seq: usize,
}

impl<T, K> BoundedKeySorter<T, K> {
    /// Constructs a `BoundedKeySorter`; `SortTopK`/`SortBottomK` activate heap mode, `Default` uses a plain vec.
    pub(crate) fn new(
        descending: bool,
        strategy: StageStrategy,
        cmp: fn(&K, &K) -> Ordering,
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
    pub(crate) fn push_keyed(&mut self, key: K, item: T) {
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
struct BoundedEntry<T, K> {
    key: K,
    item: T,
    // Insertion sequence for stable ordering among equal keys.
    seq: usize,
    // When `true`, the heap is a max-heap (evicts smallest); otherwise min-heap.
    keep_largest: bool,
    cmp: fn(&K, &K) -> Ordering,
}

impl<T, K> PartialEq for BoundedEntry<T, K> {
    fn eq(&self, other: &Self) -> bool {
        self.seq == other.seq && (self.cmp)(&self.key, &other.key) == Ordering::Equal
    }
}

impl<T, K> Eq for BoundedEntry<T, K> {}

impl<T, K> PartialOrd for BoundedEntry<T, K> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T, K> Ord for BoundedEntry<T, K> {
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
pub(crate) struct OrderedKeySorter<T, K = Val> {
    heap: BinaryHeap<OrderedEntry<T, K>>,
    // Monotonically increasing insertion sequence for stable ordering.
    next_seq: usize,
    // Output order (`true` = largest first).
    descending: bool,
    cmp: fn(&K, &K) -> Ordering,
}

impl<T, K> OrderedKeySorter<T, K> {
    /// Creates a new `OrderedKeySorter` with the given sort direction and key comparator.
    pub(crate) fn new(descending: bool, cmp: fn(&K, &K) -> Ordering) -> Self {
        Self {
            heap: BinaryHeap::new(),
            next_seq: 0,
            descending,
            cmp,
        }
    }

    /// Inserts `item` associated with `key` into the heap.
    pub(crate) fn push_keyed(&mut self, key: K, item: T) {
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
    pub(crate) fn finish(self) -> OrderedByKey<T, K> {
        OrderedByKey { heap: self.heap }
    }
}

/// Lazy iterator that extracts items from a `BinaryHeap` in sorted order, created by `OrderedKeySorter::finish`.
pub(crate) struct OrderedByKey<T, K = Val> {
    heap: BinaryHeap<OrderedEntry<T, K>>,
}

impl<T, K> Iterator for OrderedByKey<T, K> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.heap.pop().map(|entry| entry.item)
    }
}

// Internal heap entry for `OrderedKeySorter`; ordering ensures `BinaryHeap::pop` always yields the next sorted item.
struct OrderedEntry<T, K> {
    key: K,
    item: T,
    // Insertion sequence number for stable ordering.
    seq: usize,
    // `true` when larger keys should appear first in the output.
    descending: bool,
    cmp: fn(&K, &K) -> Ordering,
}

impl<T, K> PartialEq for OrderedEntry<T, K> {
    fn eq(&self, other: &Self) -> bool {
        self.seq == other.seq && (self.cmp)(&self.key, &other.key) == Ordering::Equal
    }
}

impl<T, K> Eq for OrderedEntry<T, K> {}

impl<T, K> PartialOrd for OrderedEntry<T, K> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T, K> Ord for OrderedEntry<T, K> {
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
