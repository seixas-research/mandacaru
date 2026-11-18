//! Not-a-knot cubic interpolation on increasing, nonuniform nodes.
use super::invalid;
use crate::{finite, Buffer, Context, Result};

pub(crate) fn compute_not_a_knot_second_derivatives(
    x: &[f64],
    y: &[f64],
    second: &mut [f64],
    upper: &mut [f64],
) {
    let n = x.len();
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
        let pivot = diagonal - lower * upper[i - 1];
        upper[i] = up / pivot;
        second[i] = (rhs - lower * second[i - 1]) / pivot;
    }
    for i in (1..n - 2).rev() {
        second[i] -= upper[i] * second[i + 1];
    }
    let first = (x[1] - x[0]) / (x[2] - x[1]);
    let last = (x[n - 1] - x[n - 2]) / (x[n - 2] - x[n - 3]);
    second[0] = (1. + first) * second[1] - first * second[2];
    second[n - 1] = (1. + last) * second[n - 2] - last * second[n - 3];
}

pub(crate) fn eval_cubic(x: &[f64], y: &[f64], s: &[f64], at: f64) -> (f64, f64) {
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
        (y[i + 1] - y[i]) / h + ((1. - 3. * a * a) * s[i] + (3. * b * b - 1.) * s[i + 1]) * h / 6.,
    )
}

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
        compute_not_a_knot_second_derivatives(x, y, second.as_mut_slice(), upper.as_mut_slice());
        finite(second.as_slice(), "spline curvature")?;
        Ok(Self {
            knots,
            values,
            second,
        })
    }
    pub fn clone_with(&self, ctx: &Context) -> Result<Self> {
        Ok(Self {
            knots: ctx.copy_slice(self.knots.as_slice())?,
            values: ctx.copy_slice(self.values.as_slice())?,
            second: ctx.copy_slice(self.second.as_slice())?,
        })
    }
    /// Second derivatives at the knots (the spline's curvature).
    pub fn curvature(&self) -> &[f64] {
        self.second.as_slice()
    }
    pub fn value_slope(&self, at: f64) -> (f64, f64) {
        eval_cubic(
            self.knots.as_slice(),
            self.values.as_slice(),
            self.second.as_slice(),
            at,
        )
    }

    /// Regular small-$r$ $C^1$ continuation $R(r) = (r/r_0)^l (c_0 + c_2 (r/r_0)^2)$ for $0 \le r < r_0$
    /// matching $(y_0, y'_0)$ at $r_0 > 0$; returns `(R(r), dR/dr)`.
    pub fn small_r_continue(y0: f64, y0_prime: f64, l: usize, r0: f64, r: f64) -> (f64, f64) {
        if r0 <= 0. || !r.is_finite() {
            return (y0, 0.);
        }
        let r = r.max(0.);
        let t = r / r0;
        let lf = l as f64;
        let c0 = 0.5 * ((lf + 2.) * y0 - r0 * y0_prime);
        let c2 = y0 - c0;
        if l == 0 {
            return (c0 + c2 * t * t, y0_prime * t);
        }
        let tl_1 = t.powi((l - 1) as i32);
        let tl = tl_1 * t;
        (
            (c0 + c2 * t * t) * tl,
            (lf * c0 * tl_1 + (lf + 2.) * c2 * tl * t) / r0,
        )
    }

    /// Reduced regular continuation factor $(R(r) / r^l, \frac{d}{dr}(R(r) / r^l))$ for $0 \le r < r_0$.
    pub fn small_r_factor(y0: f64, y0_prime: f64, l: usize, r0: f64, r: f64) -> (f64, f64) {
        if r0 <= 0. || !r.is_finite() {
            return (y0, 0.);
        }
        let r = r.max(0.);
        let t = r / r0;
        let lf = l as f64;
        let c0 = 0.5 * ((lf + 2.) * y0 - r0 * y0_prime);
        let c2 = y0 - c0;
        let rl = r0.powi(l as i32);
        ((c0 + c2 * t * t) / rl, 2. * c2 * r / (rl * r0 * r0))
    }

    /// Evaluate `(R(r), dR/dr)` with regular $r^l(a + b r^2)$ $C^1$ continuation below `knots[0]`.
    pub fn value_slope_regular(&self, r: f64, l: usize) -> (f64, f64) {
        let r0 = self.knots.as_slice()[0];
        if r < r0 {
            if r0 > 0. {
                let (y0, y0p) = self.value_slope(r0);
                return Self::small_r_continue(y0, y0p, l, r0, r);
            }
            return (self.values.as_slice()[0], 0.);
        }
        self.value_slope(r)
    }
}

/// Quadrature weights for [`simpson`] on strictly increasing nodes `x`, so
/// `simpson_weights(x).dot(y)` matches `simpson(x, y)` on uniform and non-uniform meshes.
pub fn simpson_weights(x: &[f64]) -> Result<Vec<f64>> {
    if x.len() < 2 || x.windows(2).any(|w| w[1] <= w[0]) {
        return Err(invalid("invalid Simpson nodes"));
    }
    finite(x, "quadrature nodes")?;
    let n = x.len();
    let mut w = vec![0.; n];
    if n == 2 {
        let half = 0.5 * (x[1] - x[0]);
        w[0] = half;
        w[1] = half;
        return Ok(w);
    }
    for i in (0..n - 2).step_by(2) {
        let a = x[i + 1] - x[i];
        let b = x[i + 2] - x[i + 1];
        let scale = (a + b) / 6.;
        w[i] += scale * (2. - b / a);
        w[i + 1] += scale * (a + b).powi(2) / (a * b);
        w[i + 2] += scale * (2. - a / b);
    }
    if n % 2 == 0 {
        let a = x[n - 2] - x[n - 3];
        let b = x[n - 1] - x[n - 2];
        w[n - 1] += (2. * b * b + 3. * a * b) / (6. * (a + b));
        w[n - 2] += (b * b + 3. * a * b) / (6. * a);
        w[n - 3] -= b.powi(3) / (6. * a * (a + b));
    }
    Ok(w)
}

/// Composite quadratic interpolation, including the last-interval correction
/// on an even number of nodes. Supports unequal radial steps.
pub fn simpson(x: &[f64], y: &[f64]) -> Result<f64> {
    if x.len() != y.len() {
        return Err(invalid("invalid Simpson nodes"));
    }
    let weights = simpson_weights(x)?;
    finite(y, "quadrature samples")?;
    Ok(weights.iter().zip(y).map(|(w, y)| w * y).sum())
}

#[cfg(test)]
mod tests {
    use super::{simpson, simpson_weights};

    #[test]
    fn simpson_weights_reproduce_basis_and_polynomial_integrals() {
        for n in [2usize, 5, 6, 17, 18] {
            let x: Vec<f64> = (0..n)
                .map(|i| 0.1 * (i as f64) + 0.01 * ((i * i) as f64))
                .collect();
            let weights = simpson_weights(&x).unwrap();
            for i in 0..n {
                let mut basis = vec![0.0; n];
                basis[i] = 1.0;
                let direct = simpson(&x, &basis).unwrap();
                assert!((weights[i] - direct).abs() < 1e-15);
            }
            let poly: Vec<f64> = x.iter().map(|&t| 1.0 - 2.0 * t + 3.0 * t * t).collect();
            let quad = weights.iter().zip(&poly).map(|(w, y)| w * y).sum::<f64>();
            let a = x[0];
            let b = x[n - 1];
            let exact = (b - a) - (b * b - a * a) + (b * b * b - a * a * a);
            if n >= 3 {
                assert!((quad - exact).abs() < 1e-13, "n={n}: {quad} vs {exact}");
            }
        }
    }
}
