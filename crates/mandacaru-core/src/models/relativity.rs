//! Scalar-relativistic (Koelling-Harmon) and Dirac radial operator on a uniform grid.
//!
//! Eliminating the small component leaves one equation for `P = r g`,
//! `W'' = [l(l+1)/r^2 + kappa M'/(M r) - M''/(2M) + 3 M'^2/(4 M^2) + 2 M (V - eps)] W` with
//! `P = sqrt(M) W` and `M = 1 + (eps - V)/(2 c^2)`. The three theories differ only in
//! `kappa`: any value with `M = 1` is nonrelativistic, `-1` is scalar-relativistic, and
//! `-(l+1)` / `+l` are the two Dirac branches. The bound-state problem stays a symmetric
//! tridiagonal eigenproblem (`M^{-1/2} A M^{-1/2}`, eigenvalue `2 eps`), re-solved with `M`
//! rebuilt from the previous eigenvalue.
use super::invalid;
use crate::{kernels, math::radial::gradient, Context, Result};

/// Speed of light in atomic units (CODATA 2022 inverse fine-structure constant).
pub const SPEED_OF_LIGHT: f64 = 137.035999177;
/// `kappa` of the scalar-relativistic (j-averaged) equation, for every `l`.
pub const SCALAR_KAPPA: i32 = -1;
/// Convergence of the `M(eps)` iteration, relative to the level (floor of the same size).
const ENERGY_TOLERANCE: f64 = 1e-12;
/// Smallest `M` the square root is taken of.
const MASS_FLOOR: f64 = 1e-3;
const ITERATIONS: usize = 16;

/// The level of theory of a radial equation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Treatment {
    None,
    Scalar,
    Dirac,
}

impl Treatment {
    /// Canonical name, or any of the accepted aliases, case and spaces ignored.
    pub fn parse(name: &str) -> Result<Self> {
        match name.trim().to_lowercase().as_str() {
            "none" | "nonrelativistic" | "non-relativistic" | "nr" | "schrodinger" => {
                Ok(Self::None)
            }
            "scalar" | "scalar-relativistic" | "sr" | "koelling-harmon" | "kh" => Ok(Self::Scalar),
            "dirac" | "full" | "relativistic" => Ok(Self::Dirac),
            _ => Err(invalid(format!(
                "unknown relativistic treatment {name:?}; available: none, scalar, dirac"
            ))),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Scalar => "scalar",
            Self::Dirac => "dirac",
        }
    }
    /// Every relativistic treatment also uses the relativistic LDA/PBE exchange factor.
    pub fn relativistic_exchange(self) -> bool {
        self != Self::None
    }
}

/// The `kappa` values of a subshell, ordered by increasing `j`: `[l, -(l+1)]` or `[-1]` for s.
pub fn kappa_values(l: usize) -> Vec<i32> {
    if l == 0 {
        vec![-1]
    } else {
        vec![l as i32, -(l as i32 + 1)]
    }
}
/// `2j + 1`: how many electrons the `(l, j)` subshell holds.
pub fn degeneracy(kappa: i32) -> usize {
    2 * kappa.unsigned_abs() as usize
}
/// `(2j+1)`-weighted average over the two `j` of one `l`.
pub fn j_average(l: usize, values: impl Fn(i32) -> f64) -> f64 {
    let ks = kappa_values(l);
    let total = ks.iter().map(|&k| degeneracy(k) as f64).sum::<f64>();
    ks.iter()
        .map(|&k| degeneracy(k) as f64 * values(k))
        .sum::<f64>()
        / total
}

/// Energy-independent derivatives of one potential, built once and shared by every state.
///
/// `dv` is `V'`; `d_smooth` and `d2_smooth` are the first and second derivatives of the
/// potential with its `-Z/r` part removed. The Coulomb singularity is differentiated in
/// closed form: taking `V''` through it numerically costs three digits in the 1s level.
pub struct PotentialTerms {
    pub z: f64,
    pub dv: Vec<f64>,
    pub d_smooth: Vec<f64>,
    pub d2_smooth: Vec<f64>,
}

impl PotentialTerms {
    pub fn new(r: &[f64], potential: &[f64], z: f64) -> Result<Self> {
        let smooth = r
            .iter()
            .zip(potential)
            .map(|(r, v)| v + z / r)
            .collect::<Vec<_>>();
        let d_smooth = gradient(r, &smooth)?;
        let d2_smooth = gradient(r, &d_smooth)?;
        let dv = r
            .iter()
            .zip(&d_smooth)
            .map(|(r, d)| z / (r * r) + d)
            .collect();
        Ok(Self {
            z,
            dv,
            d_smooth,
            d2_smooth,
        })
    }
    /// `kappa M'/(M r) - M''/(2M)` with its `1/r^3` part cancelled exactly:
    /// `[-(1+kappa) Z/r^3 + V_s''/2 - kappa V_s'/r] / (2 c^2 M)`.
    pub fn darwin_spin_orbit(&self, r: &[f64], kappa: i32, mass: &[f64]) -> Vec<f64> {
        let k = kappa as f64;
        let c = SPEED_OF_LIGHT;
        (0..r.len())
            .map(|i| {
                let bracket = -(1.0 + k) * self.z / r[i].powi(3) + 0.5 * self.d2_smooth[i]
                    - k * self.d_smooth[i] / r[i];
                bracket / (2.0 * c * c * mass[i])
            })
            .collect()
    }
}

/// `(M, M')` of `M = 1 + (eps - V)/(2 c^2)`, with `M` floored.
pub fn mass_factor(potential: &[f64], energy: f64, terms: &PotentialTerms) -> (Vec<f64>, Vec<f64>) {
    let two_c2 = 2.0 * SPEED_OF_LIGHT * SPEED_OF_LIGHT;
    (
        potential
            .iter()
            .map(|v| (1.0 + (energy - v) / two_c2).max(MASS_FLOOR))
            .collect(),
        terms.dv.iter().map(|d| -d / two_c2).collect(),
    )
}

/// The Numerov coefficient of `W'' = f W` (finite wherever `r > 0`). It is
/// `2[V + l(l+1)/2r^2 - eps]` exactly for `Treatment::None`.
pub fn relativistic_f(
    r: &[f64],
    potential: &[f64],
    l: usize,
    kappa: i32,
    energy: f64,
    terms: &PotentialTerms,
    treatment: Treatment,
) -> Vec<f64> {
    let ll = (l * (l + 1)) as f64;
    if treatment == Treatment::None {
        return (0..r.len())
            .map(|i| ll / (r[i] * r[i]) + 2.0 * (potential[i] - energy))
            .collect();
    }
    let k = if treatment == Treatment::Scalar {
        SCALAR_KAPPA
    } else {
        kappa
    };
    let (mass, dm) = mass_factor(potential, energy, terms);
    let darwin = terms.darwin_spin_orbit(r, k, &mass);
    (0..r.len())
        .map(|i| {
            ll / (r[i] * r[i])
                + darwin[i]
                + 3.0 * dm[i] * dm[i] / (4.0 * mass[i] * mass[i])
                + 2.0 * mass[i] * (potential[i] - energy)
        })
        .collect()
}

/// `u / sqrt(int u^2 dr)` with the trapezoid rule, sign fixed so `u[0] >= 0`.
pub fn normalize(u: &mut [f64], r: &[f64]) {
    let mut sum = 0.;
    for i in 1..u.len() {
        sum += 0.5 * (r[i] - r[i - 1]) * (u[i - 1] * u[i - 1] + u[i] * u[i]);
    }
    let scale = sum.sqrt();
    let flip = u[0] / scale < 0.;
    for v in u.iter_mut() {
        *v /= scale;
        if flip {
            *v = -*v;
        }
    }
}

#[allow(clippy::too_many_arguments)]
/// Bound state of the (relativistic) radial equation: `(P, eps)` with `P = r g` normalized to
/// `int P^2 dr = 1` and positive near the origin.
///
/// `energy_guess` seeds `M` (a self-consistent atom passes the previous eigenvalue); `terms`
/// are the shared potential derivatives (built here when absent). `kappa` is required for
/// `Treatment::Dirac`.
pub fn solve_radial(
    ctx: &Context,
    r: &[f64],
    potential: &[f64],
    l: usize,
    n_nodes: usize,
    treatment: Treatment,
    kappa: Option<i32>,
    z: f64,
    energy_guess: Option<f64>,
    terms: Option<&PotentialTerms>,
) -> Result<(Vec<f64>, f64)> {
    let n = r.len();
    if n < 4 || potential.len() != n {
        return Err(invalid(
            "radial solve needs matching grids of at least 4 points",
        ));
    }
    if treatment == Treatment::Dirac && kappa.is_none() {
        return Err(invalid("treatment 'dirac' needs kappa"));
    }
    let step = r[1] - r[0];
    let ll = (l * (l + 1)) as f64;
    let nonrelativistic = || -> Result<(Vec<f64>, f64)> {
        let diagonal = (0..n)
            .map(|i| 1.0 / (step * step) + potential[i] + ll / (2.0 * r[i] * r[i]))
            .collect::<Vec<_>>();
        let off = vec![-0.5 / (step * step); n - 1];
        let (value, vector) = kernels::eigenpair(ctx, &diagonal, &off, n_nodes)?;
        Ok((vector.as_slice().to_vec(), value))
    };
    if treatment == Treatment::None {
        let (mut u, eps) = nonrelativistic()?;
        normalize(&mut u, r);
        return Ok((u, eps));
    }
    let mut eps = match energy_guess {
        Some(e) => e,
        None => nonrelativistic()?.1,
    };
    let owned;
    let terms = match terms {
        Some(t) => t,
        None => {
            owned = PotentialTerms::new(r, potential, z)?;
            &owned
        }
    };
    let k = if treatment == Treatment::Scalar {
        SCALAR_KAPPA
    } else {
        kappa.unwrap()
    };
    let mut vector = Vec::new();
    let mut last_change = f64::INFINITY;
    for _ in 0..ITERATIONS {
        let (mass, dm) = mass_factor(potential, eps, terms);
        let darwin = terms.darwin_spin_orbit(r, k, &mass);
        let inverse = mass.iter().map(|m| 1.0 / m.sqrt()).collect::<Vec<_>>();
        let diagonal = (0..n)
            .map(|i| {
                let bracket = 2.0 * ll / (2.0 * r[i] * r[i])
                    + darwin[i]
                    + 3.0 * dm[i] * dm[i] / (4.0 * mass[i] * mass[i])
                    + 2.0 * mass[i] * potential[i];
                (2.0 / (step * step) + bracket) * inverse[i] * inverse[i]
            })
            .collect::<Vec<_>>();
        let off = (0..n - 1)
            .map(|i| -1.0 / (step * step) * inverse[i] * inverse[i + 1])
            .collect::<Vec<_>>();
        let (value, v) = kernels::eigenpair(ctx, &diagonal, &off, n_nodes)?;
        vector = v.as_slice().to_vec();
        let new = 0.5 * value;
        let change = (new - eps).abs();
        eps = new;
        if change <= ENERGY_TOLERANCE * new.abs().max(1.0) || change >= last_change {
            break;
        }
        last_change = change;
    }
    normalize(&mut vector, r);
    Ok((vector, eps))
}
