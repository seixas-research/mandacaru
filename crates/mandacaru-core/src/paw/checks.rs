//! Numerics behind the checks of a generated dataset: logarithmic derivatives
//! and phase errors against the all-electron atom, bound levels of a radial
//! Hamiltonian, node counts and the extent of a radial wave.
use super::{invalid, pseudize::snap, smooth::bessel as bessel_derivatives};
use crate::models::numerov::{derivative4, scattering_wave};
use crate::models::relativity::Treatment;
use crate::Result;
use crate::{finite, Context};
use std::f64::consts::PI;

/// `R'/R = u'/u - 1/r` at the node nearest `r_cut`, from a fourth-order stencil.
pub fn log_derivative_of_u(r: &[f64], u: &[f64], r_cut: f64) -> Result<f64> {
    if r.len() < 5 || u.len() != r.len() {
        return Err(invalid("log derivative needs matching radial arrays"));
    }
    let r_snapped = snap(r, r_cut);
    let index = r.iter().position(|x| *x == r_snapped).unwrap_or(0);
    if index < 2 || index + 3 > u.len() {
        return Err(invalid(format!(
            "the fourth-order derivative stencil needs a grid index in [2, {}] and was asked for {index}; the matching radius reached the edge of the radial grid",
            u.len() - 3
        )));
    }
    Ok(derivative4(u, r[1] - r[0], index) / u[index] - 1. / r[index])
}

/// All-electron logarithmic derivative `R'/R` at `r_cut` for each energy.
#[allow(clippy::too_many_arguments)]
pub fn log_derivatives_ae(
    ctx: &Context,
    r: &[f64],
    potential: &[f64],
    l: usize,
    energies: &[f64],
    r_cut: f64,
    z: f64,
    treatment: Treatment,
    kappa: Option<i32>,
) -> Result<Vec<f64>> {
    energies
        .iter()
        .map(|&e| {
            log_derivative_of_u(
                r,
                &scattering_wave(ctx, r, potential, l, e, z, treatment, kappa)?,
                r_cut,
            )
        })
        .collect()
}

/// `(nodes, sum)` of the regular all-electron solution at `energy` once values
/// below 1e-300 are dropped: the oscillation theorem counts bound states below it.
#[allow(clippy::too_many_arguments)]
pub fn node_count(
    ctx: &Context,
    r: &[f64],
    potential: &[f64],
    l: usize,
    energy: f64,
    z: f64,
    treatment: Treatment,
) -> Result<(usize, f64)> {
    let u = scattering_wave(ctx, r, potential, l, energy, z, treatment, None)?;
    let kept = u
        .into_iter()
        .filter(|v| v.abs() > 1e-300)
        .collect::<Vec<_>>();
    let sum = kept.iter().sum::<f64>();
    let nodes = kept
        .windows(2)
        .filter(|w| w[0].signum() != w[1].signum())
        .count();
    Ok((nodes, sum))
}

/// Eigenvalues in `(lower, upper]` of the uniform-grid radial Hamiltonian
/// `-1/2 d^2/dr^2 + v + l(l+1)/2r^2`, by Sturm bisection on the tridiagonal matrix.
pub fn bound_levels(
    r: &[f64],
    v: &[f64],
    l: usize,
    lower: Option<f64>,
    upper: f64,
) -> Result<Vec<f64>> {
    if r.len() < 4 || v.len() != r.len() {
        return Err(invalid("bound levels need matching radial arrays"));
    }
    finite(r, "radial grid")?;
    finite(v, "potential")?;
    let lower = lower.unwrap_or_else(|| v.iter().copied().fold(f64::INFINITY, f64::min) - 1.);
    let h = r[1] - r[0];
    let off2 = (0.5 / (h * h)).powi(2);
    let diag = (0..r.len())
        .map(|i| 1. / (h * h) + v[i] + (l * (l + 1)) as f64 / (2. * r[i] * r[i]))
        .collect::<Vec<_>>();
    // Number of eigenvalues strictly below `x`.
    let below = |x: f64| {
        let (mut count, mut q) = (0, 1.);
        for (i, d) in diag.iter().enumerate() {
            q = d - x - if i == 0 { 0. } else { off2 / q };
            if q == 0. {
                q = f64::MIN_POSITIVE;
            }
            if q < 0. {
                count += 1;
            }
        }
        count
    };
    let (first, last) = (below(lower), below(upper));
    let mut out = vec![];
    for index in first..last {
        let (mut lo, mut hi) = (lower, upper);
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if below(mid) > index {
                hi = mid;
            } else {
                lo = mid;
            }
            if hi - lo <= 4. * f64::EPSILON * hi.abs().max(lo.abs()).max(1e-300) {
                break;
            }
        }
        out.push(0.5 * (lo + hi));
    }
    Ok(out)
}
/// The offsets `-w, -w + s, ...` of a symmetric energy window (`arange(-w, w + s/2, s)`).
pub fn window_offsets(width: f64, step: f64) -> Vec<f64> {
    let count = ((2. * width + 0.5 * step) / step).ceil() as usize;
    (0..count).map(|i| -width + i as f64 * step).collect()
}

/// Largest phase error `|atan L_ps - atan L_ae|` (wrapped modulo pi) over the
/// window, as `(near, far, first_nonfinite)`: `near` within `|offset| <= window`,
/// `far` over every offset; `first_nonfinite` is the index of the first
/// phase that is not finite (`None` if all are).
pub fn phase_summary(
    pseudo: &[f64],
    all_electron: &[f64],
    offsets: &[f64],
    window: f64,
) -> Result<(f64, f64, Option<usize>)> {
    if pseudo.len() != all_electron.len() || pseudo.len() != offsets.len() {
        return Err(invalid("phase arrays must have the same length"));
    }
    let (mut near, mut far) = (0f64, 0f64);
    for i in 0..pseudo.len() {
        let d = pseudo[i].atan() - all_electron[i].atan();
        let error = ((d + 0.5 * PI).rem_euclid(PI) - 0.5 * PI).abs();
        if !error.is_finite() {
            return Ok((near, far, Some(i)));
        }
        far = far.max(error);
        if offsets[i].abs() <= window + 1e-9 {
            near = near.max(error);
        }
    }
    Ok((near, far, None))
}

fn sign_changes(values: &[f64]) -> usize {
    let top = values.iter().fold(0f64, |m, v| m.max(v.abs()));
    let kept = values
        .iter()
        .filter(|v| v.abs() > 1e-6 * top)
        .collect::<Vec<_>>();
    kept.windows(2)
        .filter(|w| w[0].signum() != w[1].signum())
        .count()
}

/// Linear interpolation (`np.interp`): values outside the nodes take the end values.
pub fn interpolate_linear(x: &[f64], nodes: &[f64], values: &[f64]) -> Result<Vec<f64>> {
    if nodes.len() < 2 || values.len() != nodes.len() || nodes.windows(2).any(|w| w[1] <= w[0]) {
        return Err(invalid(
            "interpolation needs increasing nodes and matching values",
        ));
    }
    Ok(x.iter()
        .map(|&p| {
            if p <= nodes[0] {
                return values[0];
            }
            if p >= nodes[nodes.len() - 1] {
                return values[nodes.len() - 1];
            }
            let i = nodes.partition_point(|n| *n <= p) - 1;
            let slope = (values[i + 1] - values[i]) / (nodes[i + 1] - nodes[i]);
            values[i] + slope * (p - nodes[i])
        })
        .collect())
}

/// Indices of the smooth waves (Bessel expansions `coefficients`/`wavevectors`,
/// `waves x terms`) with more nodes inside `r_cut` than their all-electron waves
/// `ae` (`waves x len(r)`): pseudization removes nodes, it never has to add one.
pub fn spurious_nodes(
    l: usize,
    r_cut: f64,
    r: &[f64],
    ae: &[f64],
    wavevectors: &[f64],
    coefficients: &[f64],
    waves: usize,
) -> Result<Vec<usize>> {
    let n = r.len();
    if waves == 0
        || ae.len() != waves * n
        || coefficients.len() != wavevectors.len()
        || coefficients.len() % waves != 0
    {
        return Err(invalid("inconsistent partial-wave arrays"));
    }
    let terms = coefficients.len() / waves;
    let radius = super::pseudize_tables::linspace(0.02 * r_cut, r_cut, 400);
    let mut out = vec![];
    for i in 0..waves {
        let smooth = radius
            .iter()
            .map(|&x| {
                (0..terms)
                    .map(|t| {
                        coefficients[i * terms + t]
                            * bessel_derivatives(l, wavevectors[i * terms + t] * x)[0]
                    })
                    .sum::<f64>()
            })
            .collect::<Vec<_>>();
        let exact = interpolate_linear(&radius, r, &ae[i * n..(i + 1) * n])?;
        if sign_changes(&smooth) > sign_changes(&exact) {
            out.push(i);
        }
    }
    Ok(out)
}

/// Box radius where a wave has decayed to `floor` of its maximum, clipped to `[lo, hi]`.
pub fn spectrum_extent(r: &[f64], wave: &[f64], floor: f64, lo: f64, hi: f64) -> Result<f64> {
    if r.is_empty() || wave.len() != r.len() {
        return Err(invalid("extent needs matching radial arrays"));
    }
    let u = r
        .iter()
        .zip(wave)
        .map(|(r, w)| (w * r).abs())
        .collect::<Vec<_>>();
    let peak = (0..u.len()).fold(0, |best, i| if u[i] > u[best] { i } else { best });
    let extent = (peak..u.len())
        .rev()
        .find(|&i| u[i] > floor * u[peak])
        .map_or(hi, |i| r[i]);
    Ok(extent.clamp(lo, hi))
}

/// The radius where `|r R(r)|` peaks, the first one on ties.
pub fn peak_radius(r: &[f64], radial: &[f64]) -> Result<f64> {
    if r.is_empty() || radial.len() != r.len() {
        return Err(invalid("peak radius needs matching radial arrays"));
    }
    let u = r
        .iter()
        .zip(radial)
        .map(|(r, v)| (v * r).abs())
        .collect::<Vec<_>>();
    Ok(r[(0..u.len()).fold(0, |best, i| if u[i] > u[best] { i } else { best })])
}
