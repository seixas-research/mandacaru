//! A tabulated radial function `R(r)` of a pseudo-atomic orbital or projector:
//! the not-a-knot cubic spline of its table, the continuation below the first
//! node, and the piecewise-polynomial coefficients the Bloch-sum kernels take.
use super::invalid;
use crate::{math::spline::Cubic, product, Context, Result};

fn spline(ctx: &Context, r: &[f64], values: &[f64]) -> Result<Cubic> {
    Cubic::new(ctx, ctx.copy_slice(r)?, ctx.copy_slice(values)?)
}

/// `R(r)` (or `dR/dr` with `slope`) at `nodes`.
///
/// Zero beyond the last table node and at a NaN node. Below the first node
/// `r_0 > 0` the function continues as `R(r_0) (r / r_0)^l` (a constant for
/// `l = 0`; its derivative is zero there); with `r_0 <= 0` the first value is
/// held. This is the continuation the Python implementation used, so a grid node
/// on the nucleus samples `0` for `l > 0` instead of `R(r_0)`.
pub fn evaluate(
    ctx: &Context,
    r: &[f64],
    values: &[f64],
    l: usize,
    nodes: &[f64],
    slope: bool,
) -> Result<Vec<f64>> {
    let cubic = spline(ctx, r, values)?;
    let _out = ctx.reserve(product(&[nodes.len(), 8])?)?;
    let (r0, r_max) = (r[0], r[r.len() - 1]);
    let first = cubic.value_slope(r0).0;
    let l_i = i32::try_from(l).map_err(|_| invalid("angular momentum too large"))?;
    Ok(nodes
        .iter()
        .map(|&x| {
            if x.is_nan() || x > r_max {
                return 0.;
            }
            let (value, derivative) = cubic.value_slope(x.clamp(r0, r_max));
            let result = if x < r0 && r0 > 0. {
                match (slope, l) {
                    (false, 0) => value,
                    (false, _) => value * (x.max(0.) / r0).powi(l_i),
                    (true, 0) => 0.,
                    (true, _) => first * l as f64 * x.max(0.).powi(l_i - 1) / r0.powi(l_i),
                }
            } else if slope {
                derivative
            } else {
                value
            };
            if result.is_nan() {
                0.
            } else {
                result
            }
        })
        .collect())
}

/// Piecewise-cubic coefficients of the spline, `(4, n - 1)` row-major, highest
/// power first: `R(r) = c[0] t^3 + c[1] t^2 + c[2] t + c[3]` with `t = r - r_i`
/// on `[r_i, r_{i+1})`.
pub fn coefficients(ctx: &Context, r: &[f64], values: &[f64]) -> Result<Vec<f64>> {
    let cubic = spline(ctx, r, values)?;
    let s = cubic.curvature();
    let intervals = r.len() - 1;
    let _out = ctx.reserve(product(&[intervals, 4, 8])?)?;
    let mut out = vec![0.; 4 * intervals];
    for i in 0..intervals {
        let h = r[i + 1] - r[i];
        out[i] = (s[i + 1] - s[i]) / (6. * h);
        out[intervals + i] = s[i] / 2.;
        out[2 * intervals + i] = (values[i + 1] - values[i]) / h - h * (2. * s[i] + s[i + 1]) / 6.;
        out[3 * intervals + i] = values[i];
    }
    Ok(out)
}
