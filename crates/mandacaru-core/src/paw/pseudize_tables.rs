//! Quadrature tables of the Bessel pseudization: cubic resampling, the inner
//! fine grid, Bessel tables and the Fourier-Bessel transform of the all-electron
//! tail that the residual kinetic energy needs.
use super::invalid;
use crate::math::spline::{simpson, Cubic};
use crate::{product, Context, Result};
use mandacaru_kernels::two_center::spherical_bessel as bessel;
use std::f64::consts::PI;

/// Points of the fine quadrature grid inside `r_cut`.
pub const INNER_POINTS: usize = 801;
/// Finest spacing (Bohr) of the tail quadrature of the residual energy.
pub const TAIL_SPACING: f64 = 0.00125;
/// Upper wave vector and spacing of the Fourier grid of the residual energy.
pub const Q_MAX: f64 = 60.;
pub const Q_STEP: f64 = 0.1;

/// `n` evenly spaced points from `a` to `b` inclusive.
pub fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    let step = if n > 1 { (b - a) / (n - 1) as f64 } else { 0. };
    let mut out = (0..n).map(|i| a + i as f64 * step).collect::<Vec<_>>();
    if n > 1 {
        out[n - 1] = b;
    }
    out
}

pub fn inner_grid(r_cut: f64) -> Vec<f64> {
    linspace(0., r_cut, INNER_POINTS)
}

/// The Fourier grid `0, 0.1, ..., 60`.
pub fn q_grid() -> Vec<f64> {
    let n = ((Q_MAX + 0.5 * Q_STEP) / Q_STEP).ceil() as usize;
    (0..n).map(|i| i as f64 * Q_STEP).collect()
}

/// Not-a-knot cubic-spline resampling of `values(r)` onto `r_new`.
pub fn resample(ctx: &Context, r: &[f64], values: &[f64], r_new: &[f64]) -> Result<Vec<f64>> {
    let mut knots = ctx.allocate::<f64>(r.len())?;
    knots.as_mut_slice().copy_from_slice(r);
    let mut samples = ctx.allocate::<f64>(values.len())?;
    samples.as_mut_slice().copy_from_slice(values);
    let spline = Cubic::new(ctx, knots, samples)?;
    Ok(r_new.iter().map(|&x| spline.value_slope(x).0).collect())
}

/// `table[n][k] = j_l(q_n r_k)`, row-major `(len(q), len(r))`.
pub fn bessel_table(l: usize, q: &[f64], r: &[f64]) -> Vec<f64> {
    q.iter()
        .flat_map(|q| r.iter().map(move |r| bessel(l as i32, q * r)))
        .collect()
}

/// `sum_k f_k g_k r_k^2` by Simpson's rule on `r`.
pub fn weighted_overlap(r: &[f64], f: &[f64], g: &[f64]) -> Result<f64> {
    let integrand = (0..r.len())
        .map(|k| f[k] * g[k] * r[k] * r[k])
        .collect::<Vec<_>>();
    simpson(r, &integrand)
}

/// `G[n][m] = int_0^rc j_l(q_n r) j_l(q_m r) r^2 dr` on the inner grid.
pub fn gram(l: usize, qs: &[f64], r_in: &[f64]) -> Result<Vec<f64>> {
    let table = bessel_table(l, qs, r_in);
    let n = r_in.len();
    let mut out = vec![0.; qs.len() * qs.len()];
    for a in 0..qs.len() {
        for b in 0..qs.len() {
            out[a * qs.len() + b] =
                weighted_overlap(r_in, &table[a * n..(a + 1) * n], &table[b * n..(b + 1) * n])?;
        }
    }
    Ok(out)
}

/// `B[n][k] = sqrt(2/pi) int_0^rc j_l(q_n r) j_l(Q_k r) r^2 dr`, `(len(qs), len(Q))`.
pub fn transform_table(l: usize, qs: &[f64], r_in: &[f64], q_grid: &[f64]) -> Result<Vec<f64>> {
    let (basis, kernel) = (bessel_table(l, qs, r_in), bessel_table(l, q_grid, r_in));
    let n = r_in.len();
    let mut out = vec![0.; qs.len() * q_grid.len()];
    for a in 0..qs.len() {
        for k in 0..q_grid.len() {
            out[a * q_grid.len() + k] = (2. / PI).sqrt()
                * weighted_overlap(
                    r_in,
                    &basis[a * n..(a + 1) * n],
                    &kernel[k * n..(k + 1) * n],
                )?;
        }
    }
    Ok(out)
}

/// Bessel transform of the all-electron tail from exactly `r_cut`. The tail is
/// resampled onto a grid four times finer than the atomic one (at least as
/// fine as [`TAIL_SPACING`]); a scattering tail is tapered between 3 and 5 `r_cut`.
pub fn tail_transform(
    ctx: &Context,
    l: usize,
    r: &[f64],
    wave: &[f64],
    r_cut: f64,
    bound: bool,
    q_grid: &[f64],
) -> Result<Vec<f64>> {
    let n = r.len();
    if wave.len() != n || n < 4 {
        return Err(invalid("tail transform needs matching radii and wave"));
    }
    let r_end = if bound {
        let last = (0..n)
            .rposition(|k| (wave[k] * r[k]).abs() > 1e-10)
            .ok_or_else(|| invalid("the bound wave vanishes everywhere"))?;
        r[(last + 1).min(n - 1)]
    } else {
        (5. * r_cut).min(r[n - 1])
    };
    let step = (0.25 * (r[1] - r[0])).min(TAIL_SPACING);
    let intervals = ((r_end - r_cut) / step).ceil() as usize | 1;
    let r_tail = linspace(r_cut, r_end, intervals + 1);
    let _workspace = ctx.reserve(product(&[r_tail.len(), 4, 8])?)?;
    let mut values = resample(ctx, r, wave, &r_tail)?;
    if !bound {
        for (v, &x) in values.iter_mut().zip(&r_tail) {
            let t = ((x - 3. * r_cut) / (2. * r_cut)).clamp(0., 1.);
            *v *= (0.5 * PI * t).cos().powi(2);
        }
    }
    let weighted = values
        .iter()
        .zip(&r_tail)
        .map(|(v, x)| v * x * x)
        .collect::<Vec<_>>();
    q_grid
        .iter()
        .map(|&q| {
            let kernel = r_tail
                .iter()
                .zip(&weighted)
                .map(|(x, w)| bessel(l as i32, q * x) * w)
                .collect::<Vec<_>>();
            Ok((2. / PI).sqrt() * simpson(&r_tail, &kernel)?)
        })
        .collect()
}
