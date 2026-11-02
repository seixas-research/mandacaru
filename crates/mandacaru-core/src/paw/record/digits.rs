//! The storage rule of the library tables: ten significant digits, every
//! `stride`-th point.
//!
//! Python wrote `float("%.10g" % v)` of `values[stride - 1::stride]`
//! (`pseudopotentials/io.py::_table`). The kept points are `stride - 1,
//! 2 stride - 1, ...`: the generation grid starts one step `h` from the origin,
//! so these are the radii `stride h, 2 stride h, ...`, a grid that again
//! starts one (coarser) step from the origin. `%g` and `%e` differ only in
//! how the text is laid out; the double read back from ten significant digits
//! is the same, so the digits are produced with Rust's exact `{:.9e}`.
//! Both round the exact binary value half to even.

use std::io::Write;

use crate::paw::invalid;
use crate::{Buffer, Context, Result};

/// Significant digits kept per number (`io.py::DIGITS`).
pub const DIGITS: usize = 10;
/// The stride a library is generated at (`io.py::STRIDE`).
pub const LIBRARY_STRIDE: usize = 4;

/// `float("%.10g" % value)`. A value that rounds past `f64::MAX` becomes
/// infinity, as in Python; zero, infinities and NaN pass through unchanged.
pub fn round_digits(value: f64) -> f64 {
    if !value.is_finite() || value == 0.0 {
        return value;
    }
    let mut text = [0u8; 40];
    let mut cursor = &mut text[..];
    write!(cursor, "{:.*e}", DIGITS - 1, value).expect("a ten-digit float fits in 40 bytes");
    let used = 40 - cursor.len();
    std::str::from_utf8(&text[..used])
        .expect("float formatting is ASCII")
        .parse()
        .expect("float formatting is a valid float")
}

/// `round_digits` of `values[stride - 1::stride]`.
pub fn decimate(values: &[f64], stride: usize) -> Result<Vec<f64>> {
    if stride == 0 {
        return Err(invalid("paw record: the stride must be at least 1"));
    }
    Ok(values
        .iter()
        .skip(stride - 1)
        .step_by(stride)
        .map(|v| round_digits(*v))
        .collect())
}

/// [`decimate`] into a budgeted buffer, reserved before it is allocated.
pub fn table(ctx: &Context, values: &[f64], stride: usize) -> Result<Buffer<f64>> {
    if stride == 0 {
        return Err(invalid("paw record: the stride must be at least 1"));
    }
    let count = values.len().saturating_sub(stride - 1).div_ceil(stride);
    let mut out = ctx.allocate::<f64>(count)?;
    for (slot, value) in out
        .as_mut_slice()
        .iter_mut()
        .zip(values.iter().skip(stride - 1).step_by(stride))
    {
        *slot = round_digits(*value);
    }
    Ok(out)
}
