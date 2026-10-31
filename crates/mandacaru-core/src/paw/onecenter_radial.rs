//! Radial kernels of the PAW one-center electron-electron terms: the Slater
//! integral, the multipole components of the screened Coulomb kernel, the
//! potential of a compensation shape under it, and the small helpers the
//! one-center assembly shares (the `np.gradient` of the first edge order, the
//! squared 3j symbol).
use super::{invalid, multipole::shape_function};
use crate::{
    math::{
        radial::gauss_legendre,
        spline::{simpson, Cubic},
    },
    Context, Result,
};
use std::f64::consts::PI;

/// Gauss-Legendre nodes of the long-range kernel's angular projection, at least;
/// the rule grows with omega times the largest radius.
pub const KERNEL_POINTS: usize = 24;
/// Extra nodes per unit of omega r: near mu = 1 the integrand varies on a scale
/// about `1 / (omega^2 r r')`, which a fixed rule misses once omega r is large.
pub const KERNEL_POINTS_PER_OMEGA_R: f64 = 8.;
/// Gauss-Legendre nodes over a compensation shape (its long-range potential).
pub const SHAPE_POINTS: usize = 48;
/// The most nodes the Gauss-Legendre rule provides.
pub const MAX_KERNEL_POINTS: usize = 4096;
/// Radial step (Bohr) of the long-range compensation potential table.
pub const SHAPE_TABLE_STEP: f64 = 0.02;

/// The largest multipole `L` of [`radial_coulomb`]. The integrand carries `r^(L+2)`
/// and `r^(-L)`, which stay finite in `f64` for radii of tens of Bohr up to this
/// order (the PAW one-center sums use `L` below 12); a larger one is refused, not
/// wrapped into a smaller one by an integer conversion.
pub const MAX_RADIAL_L: usize = 64;

/// `int int rho_1(r) rho_2(r') r_<^L / r_>^(L+1) r^2 r'^2 dr dr'`: the inner
/// integral is accumulated once (cumulative trapezoid), the outer one is Simpson.
/// `L` is at most [`MAX_RADIAL_L`].
pub fn radial_coulomb(r: &[f64], rho_1: &[f64], rho_2: &[f64], l: usize) -> Result<f64> {
    if l > MAX_RADIAL_L {
        return Err(invalid(format!(
            "the radial Coulomb integral is computed for L up to {MAX_RADIAL_L}, got {l}"
        )));
    }
    let n = r.len();
    if rho_1.len() != n || rho_2.len() != n {
        return Err(invalid("radial Coulomb arrays differ in length"));
    }
    if n < 2 {
        return Err(invalid(format!(
            "the radial Coulomb integral needs at least two grid points, got {n}"
        )));
    }
    let l = l as i32;
    let (mut below, mut above) = (vec![0.; n], vec![0.; n]);
    let weight_in = |i: usize| rho_2[i] * r[i].powi(l + 2);
    let weight_out = |i: usize| {
        if r[i] > 0. {
            rho_2[i] * r[i].powi(1 - l)
        } else {
            0.
        }
    };
    for i in 1..n {
        let h = 0.5 * (r[i] - r[i - 1]);
        below[i] = below[i - 1] + h * (weight_in(i - 1) + weight_in(i));
        above[i] = above[i - 1] + h * (weight_out(i - 1) + weight_out(i));
    }
    let total = above[n - 1];
    let integrand = (0..n)
        .map(|i| {
            let tail = total - above[i];
            let potential = if r[i] > 0. {
                below[i] / r[i].powi(l + 1)
            } else {
                0.
            } + r[i].powi(l) * tail;
            rho_1[i] * potential * r[i] * r[i]
        })
        .collect::<Vec<_>>();
    simpson(r, &integrand)
}

/// `P_L(x)` by the three-term recurrence.
pub fn legendre(l: usize, x: f64) -> f64 {
    let (mut prior, mut current) = (1., x);
    if l == 0 {
        return prior;
    }
    for k in 2..=l {
        let next = ((2 * k - 1) as f64 * x * current - (k - 1) as f64 * prior) / k as f64;
        prior = current;
        current = next;
    }
    current
}

/// `f_L(r1_i, r2_j)` for every requested level (`out[level][i * r2.len() + j]`):
/// the multipole components of `erf(omega R) / R`,
/// `(2L+1)/2 int_{-1}^{1} erf(omega R) / R P_L(mu) dmu`, `R^2 = r^2 + r'^2 - 2 r r' mu`.
/// `points` overrides the node count; `omega = 0` gives zeros.
pub fn long_range_radial_kernel(
    ctx: &Context,
    r1: &[f64],
    r2: &[f64],
    levels: &[usize],
    omega: f64,
    points: Option<usize>,
) -> Result<Vec<Vec<f64>>> {
    if !omega.is_finite() || omega < 0. {
        return Err(invalid("omega must be finite and nonnegative"));
    }
    let cells = r1
        .len()
        .checked_mul(r2.len())
        .ok_or(crate::Error::Overflow)?;
    let _memory = ctx.reserve(crate::product(&[cells, levels.len(), 8])?)?;
    let mut out = vec![vec![0.; cells]; levels.len()];
    if omega == 0. || levels.is_empty() || cells == 0 {
        return Ok(out);
    }
    let points = match points {
        Some(p) => p,
        None => {
            let reach = omega * r1.iter().chain(r2).fold(0_f64, |m, v| m.max(*v));
            KERNEL_POINTS + (KERNEL_POINTS_PER_OMEGA_R * reach).ceil() as usize
        }
    };
    if points > MAX_KERNEL_POINTS {
        return Err(invalid(format!(
            "the screened kernel needs {points} Gauss-Legendre nodes (omega times the largest \
             radius is too large); the rule supports {MAX_KERNEL_POINTS}"
        )));
    }
    let rule = gauss_legendre(points)?;
    let projections = rule
        .iter()
        .map(|(mu, w)| {
            levels
                .iter()
                .map(|&l| 0.5 * (2 * l + 1) as f64 * w * legendre(l, *mu))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let limit = 2. * omega / PI.sqrt();
    for (i, &a) in r1.iter().enumerate() {
        for (j, &b) in r2.iter().enumerate() {
            for ((mu, _), projection) in rule.iter().zip(&projections) {
                let square = (a * a + b * b - 2. * a * b * mu).max(0.);
                let r = square.sqrt();
                let x = omega * r;
                let f = if x < 1e-4 {
                    limit * (1. - x * x / 3.)
                } else {
                    libm::erf(x) / r
                };
                for (level, value) in out.iter_mut().zip(projection) {
                    level[i * r2.len() + j] += f * value;
                }
            }
        }
    }
    Ok(out)
}

/// Potential of the compensation multipole `g_L Y_LM` under `erf(omega r)/r`,
/// without the `Y_LM` factor: `4 pi / (2L+1) int_0^{r_g} f_L(r, s) g_L(s) s^2 ds`,
/// tabulated every [`SHAPE_TABLE_STEP`] Bohr out to the largest radius asked
/// for and interpolated by a not-a-knot cubic spline.
pub fn long_range_shape_potential(
    ctx: &Context,
    radius: &[f64],
    r_g: f64,
    l: usize,
    omega: f64,
) -> Result<Vec<f64>> {
    if omega == 0. {
        return Ok(vec![0.; radius.len()]);
    }
    if r_g.is_nan() || r_g <= 0. || !r_g.is_finite() {
        return Err(invalid("the compensation radius must be positive"));
    }
    let reach = radius.iter().fold(0_f64, |m, v| m.max(*v)) + 1.;
    let r_max = reach.max(2. * r_g);
    // np.arange(0, r_max + step, step)
    let count = ((r_max + SHAPE_TABLE_STEP) / SHAPE_TABLE_STEP).ceil() as usize;
    let table = (0..count)
        .map(|i| i as f64 * SHAPE_TABLE_STEP)
        .collect::<Vec<_>>();
    let rule = gauss_legendre(SHAPE_POINTS)?;
    let s = rule
        .iter()
        .map(|(x, _)| 0.5 * r_g * (x + 1.))
        .collect::<Vec<_>>();
    let shape = shape_function(&s, r_g, l)?;
    let weight = rule
        .iter()
        .zip(&s)
        .zip(&shape)
        .map(|(((_, w), s), g)| 0.5 * r_g * w * s * s * g)
        .collect::<Vec<_>>();
    let kernel = long_range_radial_kernel(ctx, &table, &s, &[l], omega, None)?;
    let scale = 4. * PI / (2 * l + 1) as f64;
    let values = (0..table.len())
        .map(|i| {
            scale
                * (0..s.len())
                    .map(|j| kernel[0][i * s.len() + j] * weight[j])
                    .sum::<f64>()
        })
        .collect::<Vec<_>>();
    let spline = Cubic::new(ctx, ctx.copy_slice(&table)?, ctx.copy_slice(&values)?)?;
    Ok(radius.iter().map(|&r| spline.value_slope(r).0).collect())
}

/// `np.gradient(f, x)` with the default first-order edges.
pub fn gradient_first_edge(x: &[f64], f: &[f64]) -> Result<Vec<f64>> {
    let n = x.len();
    if n < 2 || f.len() != n {
        return Err(invalid("a gradient needs matching arrays of two points"));
    }
    let steps = x.windows(2).map(|w| w[1] - w[0]).collect::<Vec<_>>();
    let uniform = steps.iter().all(|s| *s == steps[0]);
    let mut out = vec![0.; n];
    out[0] = (f[1] - f[0]) / steps[0];
    out[n - 1] = (f[n - 1] - f[n - 2]) / steps[n - 2];
    for i in 1..n.saturating_sub(1) {
        out[i] = if uniform {
            (f[i + 1] - f[i - 1]) / (2. * steps[0])
        } else {
            let (hs, hd) = (steps[i - 1], steps[i]);
            (hs * hs * f[i + 1] - hd * hd * f[i - 1] + (hd * hd - hs * hs) * f[i])
                / (hs * hd * (hd + hs))
        };
    }
    Ok(out)
}

/// Composite trapezoid (`np.trapezoid`).
pub fn trapezoid(x: &[f64], y: &[f64]) -> f64 {
    x.windows(2)
        .zip(y.windows(2))
        .map(|(x, y)| 0.5 * (x[1] - x[0]) * (y[0] + y[1]))
        .sum()
}

/// The largest `a`, `b` or `c` [`three_j_squared`] accepts. The formula below is
/// computed in `f64` from binomial coefficients, whose largest intermediate is
/// `3^(a + b + c)` (about 1e183 at this bound), and its relative error grows with
/// the number of factors, about `a + b + c` units of round-off; the bound keeps both
/// far from their limits (the PAW one-center sums use `l` below 6).
pub const MAX_THREE_J: usize = 128;

/// `C(n, k)` by the multiplicative formula (`c <- c (n - k + i) / i`): every
/// intermediate is an integer, exact in `f64` below 2^53.
fn binomial(n: usize, k: usize) -> f64 {
    let k = k.min(n - k);
    (1..=k).fold(1_f64, |c, i| c * (n - k + i) as f64 / i as f64)
}

/// `n! / (x! y! z!)` for `x + y + z = n`, as two binomial coefficients.
fn multinomial(x: usize, y: usize, z: usize) -> f64 {
    binomial(x + y + z, x) * binomial(y + z, y)
}

/// `(a b c; 0 0 0)^2`, zero when `a + b + c` is odd or the triangle rule fails.
///
/// With `j = a + b + c` and `g = j / 2`,
/// `(a b c; 0 0 0)^2 = [g! / ((g-a)! (g-b)! (g-c)!)]^2 (j-2a)! (j-2b)! (j-2c)! / (j+1)!
/// = multinomial(g; g-a, g-b, g-c)^2 / ((j + 1) multinomial(j; j-2a, j-2b, j-2c))`,
/// which never forms a factorial (the factorial form returns 0 at (60, 60, 60) and NaN
/// at (100, 100, 100)). Each of `a`, `b`, `c` must be at most [`MAX_THREE_J`]; a larger
/// one is refused rather than computed with an unstated error.
pub fn three_j_squared(a: usize, b: usize, c: usize) -> Result<f64> {
    if a.max(b).max(c) > MAX_THREE_J {
        return Err(invalid(format!(
            "the squared 3j symbol is computed for angular momenta up to {MAX_THREE_J}, got \
             ({a}, {b}, {c})"
        )));
    }
    let j = a + b + c;
    if j % 2 == 1 || c < a.abs_diff(b) || c > a + b {
        return Ok(0.);
    }
    let g = j / 2;
    let half = multinomial(g - a, g - b, g - c);
    let full = multinomial(j - 2 * a, j - 2 * b, j - 2 * c);
    Ok(half * half / ((j + 1) as f64 * full))
}
