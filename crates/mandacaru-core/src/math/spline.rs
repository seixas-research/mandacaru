//! Not-a-knot cubic interpolation on increasing, nonuniform nodes.
use super::invalid;
use crate::{finite, Buffer, Context, Result};

#[derive(Debug)]
pub struct Cubic {
    pub knots: Buffer<f64>,
    pub values: Buffer<f64>,
    second: Buffer<f64>,
}
impl Cubic {
    pub fn new(ctx: &Context, knots: Buffer<f64>, values: Buffer<f64>) -> Result<Self> {
        let n = knots.as_slice().len();
        let x = knots.as_slice();
        let y = values.as_slice();
        if n < 4 || y.len() != n || x.windows(2).any(|w| w[1] <= w[0]) {
            return Err(invalid(
                "cubic spline needs four increasing knots and matching values",
            ));
        }
        finite(x, "spline knots")?;
        finite(y, "spline values")?;
        let mut second = ctx.allocate::<f64>(n)?;
        let mut upper = ctx.allocate::<f64>(n)?;
        for i in 1..n - 1 {
            let h0 = x[i] - x[i - 1];
            let h1 = x[i + 1] - x[i];
            let (lower, diagonal, up) = if i == 1 {
                (0., h0 * (1. + h0 / h1) + 2. * (h0 + h1), h1 - h0 * h0 / h1)
            } else if i == n - 2 {
                (h0 - h1 * h1 / h0, 2. * (h0 + h1) + h1 * (1. + h1 / h0), 0.)
            } else {
                (h0, 2. * (h0 + h1), h1)
            };
            let rhs = 6. * ((y[i + 1] - y[i]) / h1 - (y[i] - y[i - 1]) / h0);
            let pivot = diagonal - lower * upper.as_slice()[i - 1];
            upper.as_mut_slice()[i] = up / pivot;
            second.as_mut_slice()[i] = (rhs - lower * second.as_slice()[i - 1]) / pivot;
        }
        for i in (1..n - 2).rev() {
            second.as_mut_slice()[i] -= upper.as_slice()[i] * second.as_slice()[i + 1];
        }
        let first = (x[1] - x[0]) / (x[2] - x[1]);
        let last = (x[n - 1] - x[n - 2]) / (x[n - 2] - x[n - 3]);
        second.as_mut_slice()[0] =
            (1. + first) * second.as_slice()[1] - first * second.as_slice()[2];
        second.as_mut_slice()[n - 1] =
            (1. + last) * second.as_slice()[n - 2] - last * second.as_slice()[n - 3];
        finite(second.as_slice(), "spline curvature")?;
        Ok(Self {
            knots,
            values,
            second,
        })
    }
    pub fn value_slope(&self, at: f64) -> (f64, f64) {
        let x = self.knots.as_slice();
        let y = self.values.as_slice();
        let s = self.second.as_slice();
        let i = x
            .partition_point(|v| *v <= at)
            .saturating_sub(1)
            .min(x.len() - 2);
        let h = x[i + 1] - x[i];
        let b = (at - x[i]) / h;
        let a = 1. - b;
        (
            a * y[i]
                + b * y[i + 1]
                + ((a * a * a - a) * s[i] + (b * b * b - b) * s[i + 1]) * h * h / 6.,
            (y[i + 1] - y[i]) / h
                + ((1. - 3. * a * a) * s[i] + (3. * b * b - 1.) * s[i + 1]) * h / 6.,
        )
    }
}

/// Composite quadratic interpolation, including the last-interval correction
/// on an even number of nodes. Supports unequal radial steps.
pub fn simpson(x: &[f64], y: &[f64]) -> Result<f64> {
    if x.len() < 2 || x.len() != y.len() || x.windows(2).any(|w| w[1] <= w[0]) {
        return Err(invalid("invalid Simpson nodes"));
    }
    finite(x, "quadrature nodes")?;
    finite(y, "quadrature samples")?;
    let n = x.len();
    if n == 2 {
        return Ok(0.5 * (x[1] - x[0]) * (y[0] + y[1]));
    }
    let mut total = 0.;
    for i in (0..n - 2).step_by(2) {
        let a = x[i + 1] - x[i];
        let b = x[i + 2] - x[i + 1];
        total += (a + b) / 6.
            * ((2. - b / a) * y[i]
                + (a + b).powi(2) / (a * b) * y[i + 1]
                + (2. - a / b) * y[i + 2]);
    }
    if n % 2 == 0 {
        let a = x[n - 2] - x[n - 3];
        let b = x[n - 1] - x[n - 2];
        total += (2. * b * b + 3. * a * b) / (6. * (a + b)) * y[n - 1]
            + (b * b + 3. * a * b) / (6. * a) * y[n - 2]
            - b.powi(3) / (6. * a * (a + b)) * y[n - 3];
    }
    Ok(total)
}
