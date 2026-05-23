//! Shared HyperLogLog implementation for `approx_count_distinct`.

use crate::{data::value::Val, util::val_to_key};

/// HyperLogLog precision: 2^14 = 16384 one-byte registers.
///
/// The small-range linear-counting correction keeps tiny inputs exact while
/// preserving the low-memory approximate behavior for large streams.
const HLL_P: u32 = 14;
const HLL_M: usize = 1 << HLL_P;
const HLL_W: u32 = 64 - HLL_P;
const HLL_ALPHA: f64 = 0.7213 / (1.0 + 1.079 / (HLL_M as f64));

#[derive(Clone)]
pub(crate) struct Hll {
    registers: [u8; HLL_M],
}

impl Hll {
    #[inline]
    pub(crate) fn new() -> Self {
        Self {
            registers: [0; HLL_M],
        }
    }

    #[inline]
    pub(crate) fn observe_val(&mut self, value: &Val) {
        self.observe_key(&val_to_key(value));
    }

    #[inline]
    pub(crate) fn observe_key(&mut self, key: &str) {
        let h = stable_hash_key(key);
        let bucket = (h >> HLL_W) as usize;
        let w = (h << HLL_P) | (1u64 << (HLL_P - 1));
        let rho = (w.leading_zeros() + 1) as u8;
        if rho > self.registers[bucket] {
            self.registers[bucket] = rho;
        }
    }

    #[inline]
    pub(crate) fn estimate(&self) -> u64 {
        estimate_registers(&self.registers)
    }
}

#[inline]
pub(crate) fn count_distinct(items: &[Val]) -> u64 {
    let mut hll = Hll::new();
    for value in items {
        hll.observe_val(value);
    }
    hll.estimate()
}

#[inline]
fn stable_hash_key(key: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    key.as_bytes().hash(&mut hasher);
    hasher.finish()
}

fn estimate_registers(registers: &[u8; HLL_M]) -> u64 {
    let mut sum = 0.0f64;
    let mut zeros = 0usize;
    for &register in registers {
        sum += (-(register as f64)).exp2();
        if register == 0 {
            zeros += 1;
        }
    }
    let raw = HLL_ALPHA * (HLL_M as f64) * (HLL_M as f64) / sum;
    let estimate = if raw <= 2.5 * (HLL_M as f64) && zeros > 0 {
        (HLL_M as f64) * ((HLL_M as f64) / zeros as f64).ln()
    } else {
        raw
    };
    estimate.round().max(0.0) as u64
}
