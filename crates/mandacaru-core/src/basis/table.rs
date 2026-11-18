//! A tabulated radial function `R(r)` of a pseudo-atomic orbital or projector:
//! the not-a-knot cubic spline of its table, the continuation below the first
//! node, and the piecewise-polynomial coefficients the Bloch-sum kernels take.
use crate::{math::spline::Cubic, product, Context, Result};

fn spline(ctx: &Context, r: &[f64], values: &[f64]) -> Result<Cubic> {
    Cubic::new(ctx, ctx.copy_slice(r)?, ctx.copy_slice(values)?)
}

/// `R(r)` (or `dR/dr` with `slope`) at `nodes`.
///
/// Zero beyond the last table node and at a NaN node. Below the first node
/// `r_0 > 0` the function continues regularly as $r^l (a + b r^2)$ matching both
/// value and slope at `r_0`; with `r_0 <= 0` the first value is held.
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
    let r_max = r[r.len() - 1];
    Ok(nodes
        .iter()
        .map(|&x| {
            if x.is_nan() || x > r_max {
                return 0.;
            }
            let (value, derivative) = cubic.value_slope_regular(x, l);
            let result = if slope { derivative } else { value };
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
