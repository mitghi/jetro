use crate::value::Val;

use super::{num_finalise, num_fold, Sink};

pub(crate) struct SinkAccumulator<'a> {
    sink: &'a Sink,
    collect: Vec<Val>,
    count: i64,
    sum_i: i64,
    sum_f: f64,
    sum_floated: bool,
    min_f: f64,
    max_f: f64,
    n_obs: usize,
    first: Option<Val>,
    last: Option<Val>,
    hll: [u8; HLL_M],
}

impl<'a> SinkAccumulator<'a> {
    pub(crate) fn new(sink: &'a Sink) -> Self {
        Self {
            sink,
            collect: Vec::new(),
            count: 0,
            sum_i: 0,
            sum_f: 0.0,
            sum_floated: false,
            min_f: f64::INFINITY,
            max_f: f64::NEG_INFINITY,
            n_obs: 0,
            first: None,
            last: None,
            hll: [0; HLL_M],
        }
    }

    pub(crate) fn push(&mut self, item: Val) -> bool {
        match self.sink {
            Sink::Collect => self.observe_collect(item),
            Sink::Count => self.observe_count(),
            Sink::Numeric(_) => self.observe_numeric(&item),
            Sink::First => return self.observe_first(item),
            Sink::Last => self.observe_last(item),
            Sink::ApproxCountDistinct => self.observe_approx_distinct(&item),
        }
        false
    }

    pub(crate) fn observe_count(&mut self) {
        self.count += 1;
    }

    pub(crate) fn observe_collect(&mut self, item: Val) {
        self.collect.push(item);
    }

    pub(crate) fn observe_numeric(&mut self, item: &Val) {
        if let Sink::Numeric(numeric) = self.sink {
            self.push_numeric(numeric.op, item);
        }
    }

    pub(crate) fn observe_first(&mut self, item: Val) -> bool {
        if self.first.is_none() {
            self.first = Some(item);
            true
        } else {
            false
        }
    }

    pub(crate) fn observe_last(&mut self, item: Val) {
        self.last = Some(item);
    }

    pub(crate) fn observe_approx_distinct(&mut self, item: &Val) {
        hll_observe(&mut self.hll, item);
    }

    pub(crate) fn push_projected_numeric(&mut self, numeric_item: &Val) {
        self.observe_numeric(numeric_item);
    }

    pub(crate) fn finish(self, unwrap_single_collect_obj: bool) -> Val {
        match self.sink {
            Sink::Collect => {
                if unwrap_single_collect_obj
                    && self.collect.len() == 1
                    && matches!(self.collect[0], Val::Obj(_))
                {
                    self.collect.into_iter().next().unwrap()
                } else {
                    Val::arr(self.collect)
                }
            }
            Sink::Count => Val::Int(self.count),
            Sink::Numeric(numeric) => num_finalise(
                numeric.op,
                self.sum_i,
                self.sum_f,
                self.sum_floated,
                self.min_f,
                self.max_f,
                self.n_obs,
            ),
            Sink::First => self.first.unwrap_or(Val::Null),
            Sink::Last => self.last.unwrap_or(Val::Null),
            Sink::ApproxCountDistinct => Val::Int(hll_estimate(&self.hll) as i64),
        }
    }

    fn push_numeric(&mut self, op: super::NumOp, item: &Val) {
        num_fold(
            &mut self.sum_i,
            &mut self.sum_f,
            &mut self.sum_floated,
            &mut self.min_f,
            &mut self.max_f,
            &mut self.n_obs,
            op,
            item,
        );
    }
}

const HLL_P: u32 = 12;
const HLL_M: usize = 1 << HLL_P;

#[inline]
fn hll_hash(v: &Val) -> u64 {
    use crate::util::val_to_key;
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};

    static STATE: std::sync::OnceLock<RandomState> = std::sync::OnceLock::new();
    let bs = STATE.get_or_init(RandomState::new);
    let s = val_to_key(v);
    let mut h = bs.build_hasher();
    h.write(s.as_bytes());
    h.finish()
}

fn hll_observe(reg: &mut [u8; HLL_M], v: &Val) {
    let h = hll_hash(v);
    let idx = (h >> (64 - HLL_P)) as usize;
    let w = (h << HLL_P) | (1u64 << (HLL_P - 1));
    let lz = w.leading_zeros() as u8 + 1;
    if lz > reg[idx] {
        reg[idx] = lz;
    }
}

fn hll_estimate(reg: &[u8; HLL_M]) -> f64 {
    let mut z: f64 = 0.0;
    let mut zeros: usize = 0;
    for &r in reg.iter() {
        z += 1.0 / (1u64 << r) as f64;
        if r == 0 {
            zeros += 1;
        }
    }
    let m = HLL_M as f64;
    let alpha_m = 0.7213 / (1.0 + 1.079 / m);
    let raw = alpha_m * m * m / z;
    if raw <= 2.5 * m && zeros > 0 {
        return m * (m / zeros as f64).ln();
    }
    raw
}
