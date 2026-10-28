//! rVV10 correlation, atomic units. The same reciprocal kernel defines the
//! energy and its variational derivatives. No external functional runtime.
//!
//! Vydrov & Van Voorhis, JCP 133, 244103 (2010),
//! Sabatini, Gorni & de Gironcoli, PRB 87, 041108 (2013),
//! Román-Pérez & Soler, PRL 103, 096102 (2009).
use super::{
    fft::{fast_len, wave_squared, Fft3},
    invalid, SpatialGrid,
};
use crate::{finite, product, quantum::C64, Buffer, Context, Result};
use std::f64::consts::PI;

#[derive(Clone, Copy)]
pub struct Options {
    pub b: f64,
    pub c: f64,
    pub q_min: f64,
    pub q_cut: f64,
    pub nodes: usize,
    pub padding: usize,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            b: 11.95,
            c: 0.0093,
            q_min: 1e-4,
            q_cut: 0.5,
            nodes: 16,
            padding: 2,
        }
    }
}
pub struct Terms {
    pub energy: f64,
    pub density: Buffer<f64>,
    pub sigma: Buffer<f64>,
}
pub fn beta(b: f64) -> f64 {
    (3. / (b * b)).powf(0.75) / 32.
}

/// Fourier transform of 1/[(a R²+1)(b R²+1)((a+b)R²+2)].
pub fn kernel_transform(a: f64, b: f64, k: f64) -> Result<f64> {
    if !a.is_finite() || !b.is_finite() || !k.is_finite() || a <= 0. || b <= 0. || k < 0. {
        return Err(invalid(
            "nonlocal kernel needs positive q and nonnegative wave number",
        ));
    }
    if (a - b).abs() <= 1e-12 * a.max(b) {
        let mu = a.sqrt().recip();
        return Ok(PI * PI * (-k * mu).exp() * (1. + k * mu) * mu.powi(3) / 8.);
    }
    let d2 = (a - b).powi(2);
    let mut result = 0.;
    for (coefficient, alpha) in [
        (a * a / d2, a),
        (b * b / d2, b),
        (-0.5 * (a + b).powi(2) / d2, 0.5 * (a + b)),
    ] {
        let mu = alpha.sqrt().recip();
        result += if k < 1e-12 {
            -coefficient * mu.powi(3)
        } else {
            coefficient * (-k * mu).exp_m1() / (alpha * k)
        };
    }
    Ok(2. * PI * PI * result)
}

/// Natural cardinal splines on a uniform log(q) mesh; storage is bounded by
/// the context. Each column is the curvature of one cardinal function.
struct Cardinals {
    nodes: usize,
    lo: f64,
    hi: f64,
    step: f64,
    second: Buffer<f64>,
}
impl Cardinals {
    fn new(ctx: &Context, options: &Options) -> Result<Self> {
        let n = options.nodes;
        let step = (options.q_cut.ln() - options.q_min.ln()) / (n - 1) as f64;
        let mut second = ctx.allocate::<f64>(product(&[n, n])?)?;
        let mut diagonal = ctx.allocate::<f64>(n)?;
        let mut rhs = ctx.allocate::<f64>(n)?;
        for column in 0..n {
            diagonal.as_mut_slice().fill(4.);
            rhs.as_mut_slice().fill(0.);
            for i in 1..n - 1 {
                rhs.as_mut_slice()[i] = 6. / step.powi(2)
                    * (f64::from(i - 1 == column) - 2. * f64::from(i == column)
                        + f64::from(i + 1 == column));
                if i > 1 {
                    diagonal.as_mut_slice()[i] -= 1. / diagonal.as_slice()[i - 1];
                    rhs.as_mut_slice()[i] -= rhs.as_slice()[i - 1] / diagonal.as_slice()[i - 1];
                }
            }
            for i in (1..n - 1).rev() {
                second.as_mut_slice()[i * n + column] = (rhs.as_slice()[i]
                    - second.as_slice()[(i + 1) * n + column])
                    / diagonal.as_slice()[i];
            }
        }
        Ok(Self {
            nodes: n,
            lo: options.q_min,
            hi: options.q_cut,
            step,
            second,
        })
    }
    fn value(&self, q: f64, column: usize) -> (f64, f64) {
        let clipped = q.clamp(self.lo, self.hi);
        let x = (clipped.ln() - self.lo.ln()) / self.step;
        let i = (x.floor() as usize).min(self.nodes - 2);
        let t = x - i as f64;
        let a = 1. - t;
        let yi = f64::from(i == column);
        let yj = f64::from(i + 1 == column);
        let mi = self.second.as_slice()[i * self.nodes + column];
        let mj = self.second.as_slice()[(i + 1) * self.nodes + column];
        let value = a * yi
            + t * yj
            + ((a.powi(3) - a) * mi + (t.powi(3) - t) * mj) * self.step.powi(2) / 6.;
        let derivative = if q <= self.lo || q > self.hi {
            0.
        } else {
            ((yj - yi) / self.step
                + ((-3. * a * a + 1.) * mi + (3. * t * t - 1.) * mj) * self.step / 6.)
                / clipped
        };
        (value, derivative)
    }
}
fn saturation(q: f64, cut: f64) -> (f64, f64) {
    // Above 2 q_c exp(-sum x^m/m) underflows; skip powers which could overflow.
    if q >= 2. * cut {
        return (cut, 0.);
    }
    let x = q / cut;
    let (mut sum, mut slope, mut power) = (0., 0., 1.);
    for m in 1..=12 {
        slope += power;
        power *= x;
        sum += power / m as f64;
    }
    let damping = (-sum).exp();
    (-cut * (-sum).exp_m1(), damping * slope)
}
fn local(
    n: f64,
    sigma: f64,
    options: &Options,
    splines: &Cardinals,
    column: usize,
) -> (f64, f64, f64) {
    if n <= 1e-12 {
        return (0., 0., 0.);
    }
    let kappa = options.b * 1.5 * PI * (9. * PI).powf(-1. / 6.) * n.powf(1. / 6.);
    let gap2 = options.c * (sigma / (n * n)).powi(2);
    let omega = (gap2 + 4. * PI * n / 3.).sqrt();
    let q = omega / kappa;
    let dqdn = (-4. * gap2 / n + 4. * PI / 3.) / (2. * omega * kappa) - q / (6. * n);
    let dqds = options.c * sigma / (n.powi(4) * omega * kappa);
    let (h, dh) = saturation(q, options.q_cut);
    let (p, dp) = splines.value(h, column);
    let weight = n * kappa.powf(-1.5);
    (
        weight * p,
        0.75 * weight / n * p + weight * dp * dh * dqdn,
        weight * dp * dh * dqds,
    )
}

pub fn calculate(
    ctx: &Context,
    grid: &SpatialGrid,
    density: &[f64],
    sigma: &[f64],
    options: Options,
) -> Result<Terms> {
    ctx.run(|| evaluate(ctx, grid, density, sigma, options))
}
pub fn evaluate(
    ctx: &Context,
    grid: &SpatialGrid,
    density: &[f64],
    sigma: &[f64],
    options: Options,
) -> Result<Terms> {
    let shape = grid.index.shape();
    let n = grid.index.points();
    finite(density, "nonlocal density")?;
    finite(sigma, "nonlocal gradient contraction")?;
    if density.len() != n || sigma.len() != n || density.iter().chain(sigma).any(|v| *v < 0.) {
        return Err(invalid(
            "nonlocal fields must be nonnegative and match the grid",
        ));
    }
    if !options.b.is_finite()
        || options.b <= 0.
        || !options.c.is_finite()
        || options.c < 0.
        || !options.q_min.is_finite()
        || options.q_min <= 0.
        || !options.q_cut.is_finite()
        || options.q_cut <= options.q_min
        || !(2..=64).contains(&options.nodes)
        || !(2..=8).contains(&options.padding)
    {
        return Err(invalid("invalid nonlocal parameters or interpolation mesh"));
    }
    if (options.q_cut / options.q_min).ln() / ((options.nodes - 1) as f64) < 1e-2 {
        return Err(invalid(
            "nonlocal log-mesh nodes must be separated by at least 0.01",
        ));
    }
    let mut padded = shape;
    if !grid.periodic {
        for a in 0..3 {
            padded[a] = fast_len(
                shape[a]
                    .checked_mul(options.padding)
                    .ok_or(crate::Error::Overflow)?,
            )?;
        }
    }
    let size = product(&padded)?;
    let half = padded[2] / 2 + 1;
    let spectrum = product(&[padded[0], padded[1], half])?;
    let nodes = options.nodes;
    let splines = Cardinals::new(ctx, &options)?;
    let mut mesh = ctx.allocate::<f64>(nodes)?;
    for a in 0..nodes {
        mesh.as_mut_slice()[a] = (options.q_min.ln() + a as f64 * splines.step).exp();
    }
    let mut transforms = ctx.allocate::<C64>(product(&[nodes, spectrum])?)?;
    let mut fft = Fft3::new(ctx, padded)?;
    let mut work = ctx.allocate::<C64>(size)?;
    let mut wave = ctx.allocate::<f64>(spectrum)?;
    {
        let periodic = if grid.periodic {
            Some(super::reciprocal::norms(ctx, grid)?)
        } else {
            None
        };
        for i in 0..padded[0] {
            for j in 0..padded[1] {
                for k in 0..half {
                    wave.as_mut_slice()[(i * padded[1] + j) * half + k] =
                        if let Some(ref norms) = periodic {
                            norms.as_slice()[(i * padded[1] + j) * padded[2] + k]
                        } else {
                            wave_squared(padded, &grid.step, [i, j, k], false).sqrt()
                        };
                }
            }
        }
    }
    for a in 0..nodes {
        work.as_mut_slice().fill(C64::default());
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    let g = (i * shape[1] + j) * shape[2] + k;
                    work.as_mut_slice()[(i * padded[1] + j) * padded[2] + k] =
                        C64::new(local(density[g], sigma[g], &options, &splines, a).0, 0.);
                }
            }
        }
        fft.transform(work.as_mut_slice(), false)?;
        for row in 0..padded[0] * padded[1] {
            transforms.as_mut_slice()[a * spectrum + row * half..a * spectrum + (row + 1) * half]
                .copy_from_slice(&work.as_slice()[row * padded[2]..row * padded[2] + half]);
        }
    }
    // Tabulating the smooth analytic radial transform avoids exponentials on
    // every FFT frequency. The same table is used in energy and response.
    let table_scale = 800.;
    let maximum = wave.as_slice().iter().copied().fold(0f64, f64::max);
    let count_float = (maximum * table_scale).ceil() + 2.;
    if count_float > usize::MAX as f64 {
        return Err(crate::Error::Overflow);
    }
    let count = count_float as usize;
    let mut table = ctx.allocate::<f64>(product(&[nodes, nodes, count])?)?;
    for a in 0..nodes {
        for b in a..nodes {
            for k in 0..count {
                let value = kernel_transform(
                    mesh.as_slice()[a],
                    mesh.as_slice()[b],
                    k as f64 / table_scale,
                )?;
                table.as_mut_slice()[(a * nodes + b) * count + k] = value;
                table.as_mut_slice()[(b * nodes + a) * count + k] = value;
            }
        }
    }
    let correction = beta(options.b);
    let mut result = Terms {
        energy: correction * density.iter().sum::<f64>() * grid.index.dv(),
        density: ctx.allocate::<f64>(n)?,
        sigma: ctx.allocate::<f64>(n)?,
    };
    result.density.as_mut_slice().fill(correction);
    for a in 0..nodes {
        work.as_mut_slice().fill(C64::default());
        for b in 0..nodes {
            for g in 0..spectrum {
                let position = wave.as_slice()[g] * table_scale;
                let index = position as usize;
                let fraction = position - index as f64;
                let row = (a * nodes + b) * count;
                let kernel = table.as_slice()[row + index] * (1. - fraction)
                    + table.as_slice()[row + index + 1] * fraction;
                let full_index = (g / half) * padded[2] + g % half;
                work.as_mut_slice()[full_index] += kernel * transforms.as_slice()[b * spectrum + g];
            }
        }
        // Reconstruct negative z frequencies from Hermitian symmetry. Real
        // inputs need only half-spectrum storage across the interpolation mesh.
        for i in 0..padded[0] {
            for j in 0..padded[1] {
                for k in half..padded[2] {
                    let mirror = (((padded[0] - i) % padded[0]) * padded[1]
                        + (padded[1] - j) % padded[1])
                        * padded[2]
                        + padded[2]
                        - k;
                    work.as_mut_slice()[(i * padded[1] + j) * padded[2] + k] =
                        work.as_slice()[mirror].conj();
                }
            }
        }
        fft.transform(work.as_mut_slice(), true)?;
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    let g = (i * shape[1] + j) * shape[2] + k;
                    let u = work.as_slice()[(i * padded[1] + j) * padded[2] + k].re;
                    let (theta, dn, ds) = local(density[g], sigma[g], &options, &splines, a);
                    result.energy -= 0.75 * theta * u * grid.index.dv();
                    result.density.as_mut_slice()[g] -= 1.5 * u * dn;
                    result.sigma.as_mut_slice()[g] -= 1.5 * u * ds;
                }
            }
        }
    }
    if !result.energy.is_finite()
        || result
            .density
            .as_slice()
            .iter()
            .chain(result.sigma.as_slice())
            .any(|v| !v.is_finite())
    {
        return Err(invalid("nonfinite nonlocal correlation result"));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn analytic_transform_matches_independent_radial_quadrature() {
        let steps = 200_000;
        let maximum = 400.;
        let h = maximum / steps as f64;
        for (a, b) in [(0.05, 0.05), (0.01, 0.3), (0.2, 0.07)] {
            for k in [0., 0.4, 2.] {
                let mut value = 0.;
                for i in 0..=steps {
                    let r = i as f64 * h;
                    let kr: f64 = k * r;
                    let sinc = if kr == 0. { 1. } else { kr.sin() / kr };
                    let r2 = r * r;
                    let phi = 1. / ((a * r2 + 1.) * (b * r2 + 1.) * ((a + b) * r2 + 2.));
                    let weight = if i == 0 || i == steps {
                        1.
                    } else if i % 2 == 0 {
                        2.
                    } else {
                        4.
                    };
                    value += weight * 4. * PI * r2 * phi * sinc;
                }
                value *= h / 3.;
                if k == 0. {
                    value += 4. * PI / (3. * a * b * (a + b) * maximum.powi(3));
                }
                let analytic = kernel_transform(a, b, k).unwrap();
                assert!((analytic - value).abs() < 1e-6 * value.abs());
            }
        }
    }
    #[test]
    fn nonorthogonal_reciprocal_alias_norms_match_reference() {
        let ctx = Context::new(Some(1), 1 << 20).unwrap();
        let grid = SpatialGrid::new(
            [6, 6, 4],
            [0.; 3],
            [1., 0.5, 0., 0., 3f64.sqrt() / 2., 0., 0., 0., 1.2],
            true,
        )
        .unwrap();
        let norms = super::super::reciprocal::norms(&ctx, &grid).unwrap();
        for (index, expected) in [
            ([0, 0, 0], 0.),
            ([3, 0, 0], 3.627598728468436),
            ([3, 3, 0], 3.627598728468436),
            ([0, 3, 2], 4.473629899755539),
            ([5, 3, 1], 3.968283205650369),
        ] {
            let k = (index[0] * 6 + index[1]) * 4 + index[2];
            assert!((norms.as_slice()[k] - expected).abs() < 2e-14);
        }
        drop(norms);
        assert_eq!(ctx.memory_used(), 0);
    }
}
