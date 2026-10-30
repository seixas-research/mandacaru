//! Exchange-correlation potential of a spherical atom on a uniform radial grid.
//!
//! LDA is pointwise. PBE needs `d rho/dr` and a radial divergence, and differentiating a
//! density that is accurate only to the eigensolver's round-off twice multiplies its noise
//! by about `1/h^2`. On the fine grid of a heavy atom that noise then enters the
//! scalar-relativistic Darwin term, which differentiates the potential twice more. So the
//! derivatives are taken on a subset of nodes of spacing `max(h, DELTA r)` (every point near
//! the nucleus, logarithmic farther out) and splined back; the potential is
//! `df/drho - (1/r^2) d/dr (r^2 df/d(drho/dr))`.
use super::{invalid, xc};
use crate::{
    math::{radial::gradient, spline::Cubic},
    Context, Result,
};

/// Relative spacing of the nodes a GGA's radial derivatives are taken on.
pub const GGA_DERIVATIVE_STEP: f64 = 0.01;
/// Densities at or below this carry no gradient correction.
pub const DENSITY_FLOOR: f64 = 1e-12;
/// The grid kernels drop exchange below a spin density of 1e-12 (a total of 2e-12 when
/// unpolarized) and correlation below a total of 1e-12; the atom supplies the uniform-gas
/// tail itself from there down, so its potential decays smoothly instead of stepping to zero.
pub const EXCHANGE_FLOOR: f64 = 2e-12;

/// Indices of the grid points the radial derivatives are taken on, or `None` for all of them
/// (grids that are non-uniform or already coarser than `GGA_DERIVATIVE_STEP * r` everywhere).
pub fn derivative_nodes(r: &[f64]) -> Option<Vec<usize>> {
    let n = r.len();
    if n < 8 {
        return None;
    }
    let h = r[1] - r[0];
    let uniform = r.windows(2).all(|w| {
        let step = w[1] - w[0];
        (step - h).abs() <= 1e-6 * h.abs()
    });
    if !uniform {
        return None;
    }
    let stride = r
        .iter()
        .map(|r| ((GGA_DERIVATIVE_STEP * r / h).floor() as usize).max(1))
        .collect::<Vec<_>>();
    if stride.iter().all(|s| *s <= 1) {
        return None;
    }
    let mut nodes = vec![0usize];
    loop {
        let next = nodes[nodes.len() - 1] + stride[nodes[nodes.len() - 1]];
        if next >= n - 1 {
            break;
        }
        nodes.push(next);
    }
    nodes.push(n - 1);
    Some(nodes)
}

/// `dy/dr` on the `nodes` subset, splined back onto `r` (all points when `nodes` is `None`).
fn radial_derivative(
    ctx: &Context,
    r: &[f64],
    y: &[f64],
    nodes: Option<&[usize]>,
) -> Result<Vec<f64>> {
    let Some(nodes) = nodes else {
        return gradient(r, y);
    };
    let x = nodes.iter().map(|&i| r[i]).collect::<Vec<_>>();
    let values = nodes.iter().map(|&i| y[i]).collect::<Vec<_>>();
    let derivative = gradient(&x, &values)?;
    let spline = Cubic::new(ctx, ctx.copy_slice(&x)?, ctx.copy_slice(&derivative)?)?;
    Ok(r.iter().map(|&at| spline.value_slope(at).0).collect())
}

/// `(energy per electron, potential)` of `functional` ("lda" or "pbe") for the spherical
/// density `rho` on the uniform grid `r`, with the relativistic exchange factor when asked.
pub fn xc_potential(
    ctx: &Context,
    r: &[f64],
    rho: &[f64],
    functional: &str,
    relativistic: bool,
) -> Result<(Vec<f64>, Vec<f64>)> {
    xc::functional(functional)?;
    let n = r.len();
    if rho.len() != n || n < 4 || r[0] <= 0. {
        return Err(invalid(
            "radial XC needs matching positive radii and density",
        ));
    }
    let gga = functional == "pbe";
    let nodes = if gga { derivative_nodes(r) } else { None };
    let slope = if gga {
        radial_derivative(ctx, r, rho, nodes.as_deref())?
    } else {
        vec![0.; n]
    };
    let mut energy = vec![0.; n];
    let mut df_drho = vec![0.; n];
    let mut flux = vec![0.; n];
    for i in 0..n {
        let density = rho[i].max(0.);
        let g = slope[i];
        let spin = 0.5 * density;
        let sigma = 0.25 * g * g;
        if !gga && density <= EXCHANGE_FLOOR {
            // LDA keeps decaying as rho^(1/3) (density floored at 1e-30).
            let (ex, vx, ec, vc) = xc::lda_terms(density);
            energy[i] = ex + ec;
            df_drho[i] = vx + vc;
            continue;
        }
        if gga && density <= DENSITY_FLOOR {
            // PBE holds the uniform-gas value at the floor density, with no gradient term.
            let probe = DENSITY_FLOOR * (1. + 1e-9);
            let (f, d) = xc::point_partials(
                functional,
                [0.5 * probe, 0.5 * probe, 0., 0., 0., 0., 0.],
                false,
            )?;
            let (ex, vx, _, _) = xc::lda_terms(probe);
            energy[i] = f / probe + ex;
            df_drho[i] = 0.5 * (d[0] + d[1]) + vx;
            continue;
        }
        let (f, d) = xc::point_partials(
            functional,
            [spin, spin, sigma, sigma, sigma, 0., 0.],
            relativistic,
        )?;
        energy[i] = f / density;
        df_drho[i] = 0.5 * (d[0] + d[1]);
        if density <= EXCHANGE_FLOOR {
            // PBE between the two floors: the kernel has dropped exchange but kept correlation;
            // add the uniform-gas exchange (the enhancement factor is not applied here).
            let (ex, vx, _, _) = xc::lda_terms(density);
            energy[i] += ex;
            df_drho[i] += vx;
        }
        flux[i] = r[i] * r[i] * 0.5 * g * (d[2] + d[3] + d[4]);
    }
    if !gga {
        return Ok((energy, df_drho));
    }
    let divergence = radial_derivative(ctx, r, &flux, nodes.as_deref())?;
    let potential = (0..n)
        .map(|i| df_drho[i] - divergence[i] / (r[i] * r[i]))
        .collect();
    Ok((energy, potential))
}
