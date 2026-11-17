//! What the search tests share: the context they run in and a NaN-propagating minimum.
#![allow(dead_code)]

use mandacaru_core::Context;

/// The smaller of two values with a NaN in either kept (`f64::min` returns the other operand, and
/// a depth that is not a number would then rank as no ghost at all).
pub fn nan_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.min(b)
    }
}

pub fn context() -> Context {
    Context::new(Some(1), 1 << 30).unwrap()
}
