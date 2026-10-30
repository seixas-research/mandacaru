//! The smooth confining potential of a confined pseudo-atomic orbital and the
//! uniform grids its radial problem is solved on (the problem itself, which
//! needs a PAW dataset, is `paw::confinement`).
use super::invalid;
use crate::Result;

/// Amplitude `A` (Hartree) of the confining potential.
pub const AMPLITUDE: f64 = 12.;
/// Inner radius of the confining potential as a fraction of `r_c`.
pub const INNER_FRACTION: f64 = 0.6;
/// Target spacing (Bohr) of the finer of the two radial grids.
pub const SPACING: f64 = 0.01;
/// Growth factor of the outward bracketing scan of the radius search.
pub const BRACKET_GROWTH: f64 = 1.25;
/// Tolerance (Bohr) of the root search on `r_c`.
pub const RADIUS_TOLERANCE: f64 = 1e-4;
/// Fraction of the radial table the cutoff may reach.
pub const TABLE_FRACTION: f64 = 0.95;

/// Amplitude and inner radius (a fraction of `r_c`) of the confining potential.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Potential {
    pub amplitude: f64,
    pub inner_fraction: f64,
}
impl Default for Potential {
    fn default() -> Self {
        Self {
            amplitude: AMPLITUDE,
            inner_fraction: INNER_FRACTION,
        }
    }
}
impl Potential {
    pub fn check(&self) -> Result<()> {
        if self.amplitude.is_finite()
            && self.amplitude > 0.
            && self.inner_fraction > 0.
            && self.inner_fraction < 1.
        {
            Ok(())
        } else {
            Err(invalid(
                "the confining potential needs a positive amplitude and an inner fraction in (0, 1)",
            ))
        }
    }
}

/// Junquera's smooth confining potential (Hartree) on `r` (Bohr): zero up to
/// `r_i = inner_fraction r_c`, `+inf` from `r_c` on.
pub fn potential(r: &[f64], r_c: f64, amplitude: f64, inner_fraction: f64) -> Vec<f64> {
    let r_i = inner_fraction * r_c;
    r.iter()
        .map(|&x| {
            if x >= r_c {
                f64::INFINITY
            } else if x > r_i {
                amplitude / (r_c - x) * (-(r_c - r_i) / (x - r_i)).exp()
            } else {
                0.
            }
        })
        .collect()
}

/// Interior nodes of the uniform grid whose node `n` is the wall: the fine
/// grid (`doubling == 0`) and the coarse one.
pub fn grid(r_c: f64, doubling: usize) -> Vec<f64> {
    let half = ((r_c / (2. * SPACING)).round_ties_even() as usize).max(8);
    let n = if doubling == 0 { 2 * half } else { half };
    (1..n).map(|i| (r_c / n as f64) * i as f64).collect()
}
