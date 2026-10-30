//! Nonlinear core correction: the Louie-Froyen-Cohen partial core density,
//! `A sin(Br)/r` inside `r_nlcc` and the true core outside it.
use super::invalid;
use crate::math::{radial::gradient, roots::brent};
use crate::{finite, Result};
use std::f64::consts::PI;

/// Where the core density falls to `ratio` times the valence density: the
/// outermost crossing, interpolated linearly between its bracketing nodes.
/// `None` when the core never dominates the valence density.
pub fn crossover_radius(
    r: &[f64],
    core: &[f64],
    valence: &[f64],
    ratio: f64,
) -> Result<Option<f64>> {
    if r.len() < 2 || core.len() != r.len() || valence.len() != r.len() {
        return Err(invalid(
            "radius, core and valence densities must have the same length",
        ));
    }
    finite(core, "core density")?;
    finite(valence, "valence density")?;
    let excess = |i: usize| core[i] - ratio * valence[i];
    let Some(i) = (0..r.len() - 1)
        .rev()
        .find(|&i| excess(i) > 0. && excess(i + 1) <= 0.)
    else {
        return Ok(None);
    };
    let span = excess(i) - excess(i + 1);
    Ok(Some(if span.abs() < 1e-300 {
        r[i]
    } else {
        r[i] + (r[i + 1] - r[i]) * excess(i) / span
    }))
}

/// The smooth core density and its `(A, B)`, matched to `core` in value and
/// first derivative at the grid node nearest `r_nlcc`.
pub fn smooth_core_density(r: &[f64], core: &[f64], r_nlcc: f64) -> Result<(Vec<f64>, f64, f64)> {
    if r.len() < 3 || core.len() != r.len() || !r_nlcc.is_finite() {
        return Err(invalid(
            "invalid radius or core density for the core correction",
        ));
    }
    finite(core, "core density")?;
    let index = (0..r.len())
        .min_by(|&a, &b| (r[a] - r_nlcc).abs().total_cmp(&(r[b] - r_nlcc).abs()))
        .unwrap_or(0);
    let (r_match, value) = (r[index], core[index]);
    if value <= 0. {
        return Err(invalid(format!(
            "the core density vanishes at r_nlcc = {r_match:.4} Bohr, so there is nothing to match a smooth continuation to"
        )));
    }
    let slope = gradient(r, core)?[index];
    let target = 1. + r_match * slope / value;
    if target >= 1. {
        return Err(invalid(format!(
            "the core density is not decreasing at r_nlcc = {r_match:.4} Bohr, so the sin(Br)/r form has no nodeless match there"
        )));
    }
    let x = brent(
        |x| Ok(x / x.tan() - target),
        1e-9,
        PI - 1e-9,
        1e-14,
        1e-15,
        100,
    )?;
    let (b, a) = (x / r_match, value * r_match / x.sin());
    let smooth = r
        .iter()
        .zip(core)
        .map(|(&r, &c)| {
            if r < r_match {
                a * (b * r).sin() / r
            } else {
                c
            }
        })
        .collect();
    Ok((smooth, a, b))
}

/// Details of the partial core density that was built.
pub struct PartialCore {
    pub density: Vec<f64>,
    pub r_nlcc: Option<f64>,
    pub amplitude: f64,
    pub wavevector: f64,
    pub core_electrons: f64,
    pub partial_core_electrons: f64,
}

/// The density to unscreen with. `r_nlcc` defaults to [`crossover_radius`];
/// the density is zero (and `r_nlcc` `None`) when the core never dominates.
pub fn partial_core_density(
    r: &[f64],
    core: &[f64],
    valence: &[f64],
    r_nlcc: Option<f64>,
    ratio: f64,
) -> Result<PartialCore> {
    let radius = match r_nlcc {
        Some(radius) => Some(radius),
        None => crossover_radius(r, core, valence, ratio)?,
    };
    let Some(radius) = radius else {
        return Ok(PartialCore {
            density: vec![0.; r.len()],
            r_nlcc: None,
            amplitude: 0.,
            wavevector: 0.,
            core_electrons: 0.,
            partial_core_electrons: 0.,
        });
    };
    let (smooth, amplitude, wavevector) = smooth_core_density(r, core, radius)?;
    let electrons = |density: &[f64]| {
        r.windows(2)
            .zip(density.windows(2))
            .map(|(r, d)| {
                0.5 * (r[1] - r[0]) * (4. * PI) * (d[0] * r[0] * r[0] + d[1] * r[1] * r[1])
            })
            .sum::<f64>()
    };
    Ok(PartialCore {
        core_electrons: electrons(core),
        partial_core_electrons: electrons(&smooth),
        density: smooth,
        r_nlcc: Some(radius),
        amplitude,
        wavevector,
    })
}
