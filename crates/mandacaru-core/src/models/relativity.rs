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
/// `kappa` of the `(l, j)` subshell: `-(l+1)` for `j = l + 1/2`, `l` for `j = l - 1/2` (`l > 0`).
pub fn kappa_of(l: usize, j: f64) -> Result<i32> {
    if (j - (l as f64 + 0.5)).abs() < 1e-9 {
        return Ok(-(l as i32 + 1));
    }
    if (j - (l as f64 - 0.5)).abs() < 1e-9 && l > 0 {
        return Ok(l as i32);
    }
    Err(invalid(format!("j must be l +/- 1/2 (and positive); got l={l}, j={j}")))
}
/// The orbital angular momentum of a Dirac quantum number `kappa` (never 0).
pub fn l_of(kappa: i32) -> Result<usize> {
    match kappa {
        0 => Err(invalid("kappa = 0 is not a Dirac quantum number")),
        k if k < 0 => Ok((-k - 1) as usize),
        k => Ok(k as usize),
    }
}
/// `j = |kappa| - 1/2`.
pub fn j_of(kappa: i32) -> f64 {
    kappa.unsigned_abs() as f64 - 0.5
}
/// `2 / (2l + 1) (eps_{l+1/2} - eps_{l-1/2})`: zero for an s shell.
pub fn spin_orbit_difference(l: usize, up: f64, down: f64) -> f64 {
    if l == 0 {
        0.
    } else {
        2. / (2 * l + 1) as f64 * (up - down)
    }
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

/// The exact Dirac energy of a one-electron ion with a point nucleus (Hartree, from the rest
/// mass): `c^2 [(1 + (Z/c)^2 / (n - |kappa| + gamma)^2)^(-1/2) - 1]`, `gamma = sqrt(kappa^2 -
/// (Z/c)^2)`, written with `ln_1p` and `exp_m1` so the nonrelativistic limit keeps its digits.
pub fn hydrogenic_dirac_energy(atomic_number: f64, n: usize, kappa: i32, c: f64) -> Result<f64> {
    let za = atomic_number / c;
    let k = kappa;
    if k.unsigned_abs() as usize > n {
        return Err(invalid(format!("|kappa| = {} exceeds n = {}", k.abs(), n)));
    }
    if za >= k.abs() as f64 {
        return Err(invalid(format!(
            "Z/c = {:.4} reaches |kappa| = {}: the point-nucleus Dirac equation has no bound state there",
            za,
            k.abs()
        )));
    }
    let gamma = ((k * k) as f64 - za * za).sqrt();
    let denom = n as f64 - k.abs() as f64 + gamma;
    let x = za * za / (denom * denom);
    let exp_term = (-0.5 * x.ln_1p()).exp_m1();
    Ok(c * c * exp_term)
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
        self.darwin_spin_orbit_with(r, kappa, mass, SPEED_OF_LIGHT)
    }
    /// [`PotentialTerms::darwin_spin_orbit`] at the speed of light `c`.
    pub fn darwin_spin_orbit_with(&self, r: &[f64], kappa: i32, mass: &[f64], c: f64) -> Vec<f64> {
        let k = kappa as f64;
        (0..r.len())
            .map(|i| {
                let bracket = -(1.0 + k) * self.z / r[i].powi(3) + 0.5 * self.d2_smooth[i]
                    - k * self.d_smooth[i] / r[i];
                bracket / (2.0 * c * c * mass[i])
            })
            .collect()
    }
}

impl PotentialTerms {
    /// `V''` with the `-Z/r` part in closed form: `-2 Z / r^3 + V_s''`.
    pub fn second_derivative(&self, r: &[f64]) -> Vec<f64> {
        (0..r.len())
            .map(|i| -2.0 * self.z / (r[i].powi(3)) + self.d2_smooth[i])
            .collect()
    }
}

/// `(M, M', M'')` of `M = 1 + (eps - V)/(2 c^2)` (`M` floored at its floor; the derivatives are
/// those of the unfloored form).
pub fn mass_terms(
    r: &[f64],
    potential: &[f64],
    energy: f64,
    terms: &PotentialTerms,
    c: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let two_c2 = 2.0 * c * c;
    let d2v = terms.second_derivative(r);
    (
        potential
            .iter()
            .map(|v| (1.0 + (energy - v) / two_c2).max(MASS_FLOOR))
            .collect(),
        terms.dv.iter().map(|d| -d / two_c2).collect(),
        d2v.iter().map(|d| -d / two_c2).collect(),
    )
}

/// The spin-orbit radial factor `xi(r) = V' / (2 c^2 r)`, divided by `M^2` when
/// `mass_corrected` (`M` at `energy`, floored).
pub fn spin_orbit_radial(
    r: &[f64],
    potential: &[f64],
    terms: &PotentialTerms,
    c: f64,
    mass_corrected: bool,
    energy: f64,
) -> Vec<f64> {
    let two_c2 = 2.0 * c * c;
    (0..r.len())
        .map(|i| {
            let mut x = terms.dv[i] / (two_c2 * r[i]);
            if mass_corrected {
                let m = (1.0 + (energy - potential[i]) / two_c2).max(MASS_FLOOR);
                x /= m * m;
            }
            x
        })
        .collect()
}

/// The large component `P = W sqrt(M)` of a scalar-relativistic solution `W`.
pub fn large_component(w: &[f64], mass: &[f64]) -> Result<Vec<f64>> {
    if w.len() != mass.len() {
        return Err(invalid("w and M must have the same length"));
    }
    Ok(w.iter().zip(mass).map(|(w, m)| w * m.sqrt()).collect())
}

/// The small component `Q = (P' + kappa P / r) / (2 c M)`, `P'` by three-point differences on
/// the nonuniform grid (one-sided at the two ends).
pub fn small_component(r: &[f64], p: &[f64], kappa: i32, mass: &[f64], c: f64) -> Result<Vec<f64>> {
    let n = r.len();
    if p.len() != n || mass.len() != n || n < 3 {
        return Err(invalid("r, P, and M must have matching length >= 3"));
    }
    let mut dp = vec![0.0; n];
    let h1_0 = r[1] - r[0];
    let h2_0 = r[2] - r[1];
    dp[0] = -(2.0 * h1_0 + h2_0) / (h1_0 * (h1_0 + h2_0)) * p[0]
        + (h1_0 + h2_0) / (h1_0 * h2_0) * p[1]
        - h1_0 / (h2_0 * (h1_0 + h2_0)) * p[2];
    for i in 1..(n - 1) {
        let h1 = r[i] - r[i - 1];
        let h2 = r[i + 1] - r[i];
        dp[i] = (h1 * h1 * p[i + 1] + (h2 * h2 - h1 * h1) * p[i] - h2 * h2 * p[i - 1])
            / (h1 * h2 * (h1 + h2));
    }
    let h1_n = r[n - 1] - r[n - 2];
    let h2_n = r[n - 2] - r[n - 3];
    dp[n - 1] = (2.0 * h1_n + h2_n) / (h1_n * (h1_n + h2_n)) * p[n - 1]
        - (h1_n + h2_n) / (h1_n * h2_n) * p[n - 2]
        + h1_n / (h2_n * (h1_n + h2_n)) * p[n - 3];
    let two_c = 2.0 * c;
    let k = kappa as f64;
    Ok((0..n)
        .map(|i| (dp[i] + k * p[i] / r[i]) / (two_c * mass[i]))
        .collect())
}

/// `(M, M')` of `M = 1 + (eps - V)/(2 c^2)`, with `M` floored.
pub fn mass_factor(potential: &[f64], energy: f64, terms: &PotentialTerms) -> (Vec<f64>, Vec<f64>) {
    mass_factor_with(potential, energy, terms, SPEED_OF_LIGHT)
}

/// [`mass_factor`] at the speed of light `c`.
pub fn mass_factor_with(
    potential: &[f64],
    energy: f64,
    terms: &PotentialTerms,
    c: f64,
) -> (Vec<f64>, Vec<f64>) {
    let two_c2 = 2.0 * c * c;
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
    relativistic_f_with(r, potential, l, kappa, energy, terms, treatment, SPEED_OF_LIGHT)
}

/// [`relativistic_f`] at the speed of light `c`.
#[allow(clippy::too_many_arguments)]
pub fn relativistic_f_with(
    r: &[f64],
    potential: &[f64],
    l: usize,
    kappa: i32,
    energy: f64,
    terms: &PotentialTerms,
    treatment: Treatment,
    c: f64,
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
    let (mass, dm) = mass_factor_with(potential, energy, terms, c);
    let darwin = terms.darwin_spin_orbit_with(r, k, &mass, c);
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
