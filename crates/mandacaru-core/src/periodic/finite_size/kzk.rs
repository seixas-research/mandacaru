//! Finite-size correction through a size-dependent LDA (Kwee, Zhang and Krakauer, PRL 100,
//! 126404 (2008)): `dE = int n [eps_xc^inf(n) - eps_xc^FS(n, L)] dr`, both on the SAME density.
//!
//! **Table I below (a1..a3, g1..g4) is ported as frozen from the pre-port Python and is
//! TO-VERIFY against the paper; no physical validation is claimed.** The correlation
//! branch (g1..g4 and the cubic) has no independent reference. The exchange branch is
//! discontinuous at `gamma = r_s(N = 2)`: also TO-VERIFY against Eq. (3).
//!
//! The parametrization is in Rydberg; every public energy is Hartree except the functions
//! suffixed `_rydberg`. The infinite-size correlation is the continuity-fixed Perdew-Zunger
//! form of `models::xc::lda_terms` (not the literature constants below `r_s = 1`).
//! `kzk` streams over the density: it allocates nothing, so there is nothing to reserve.
use super::Scope;
use crate::models::xc::lda_terms;
use crate::periodic::invalid;
use crate::Result;
use std::f64::consts::PI;

/// Hartree per Rydberg.
pub const HARTREE_PER_RYDBERG: f64 = 0.5;
/// Densities at or below this are vacuum and contribute nothing (as in the old code).
pub const DENSITY_FLOOR: f64 = 1e-12;
/// `a_1..a_3` of Eq. (3), Table I, Rydberg. TO-VERIFY against PRL 100, 126404.
pub const A1: f64 = -2.2037;
pub const A2: f64 = 0.4710;
pub const A3: f64 = -0.0150;
/// `g_1..g_4` of Eq. (6), Table I, Rydberg. TO-VERIFY against PRL 100, 126404.
pub const G1: f64 = 0.1182;
pub const G2: f64 = 1.1656;
pub const G3: f64 = -5.2884;
pub const G4: f64 = -1.1233;

/// `a_0` of Eq. (3), Rydberg: the Slater constant, computed (-0.9163305866), not rounded.
pub fn a0() -> f64 {
    -0.75 * (9.0 / (4.0 * PI * PI)).powf(1.0 / 3.0) / HARTREE_PER_RYDBERG
}

/// Side of the cube with the volume of the supercell (the paper's prescription), Bohr.
pub fn effective_length(volume: f64) -> f64 {
    volume.cbrt()
}

fn rs_at(electrons: f64, length: f64) -> f64 {
    length * (3.0 / (4.0 * PI * electrons)).powf(1.0 / 3.0)
}
fn rs_of_density(density: f64) -> f64 {
    (3.0 / (4.0 * PI * density)).powf(1.0 / 3.0)
}

/// `(e_c, de_c/dr_s)` of the native Perdew-Zunger correlation, Rydberg. The derivative is the
/// exact relation `v_c = e_c - (r_s/3) de_c/dr_s`, not a finite difference.
pub fn pz_correlation_rydberg(rs: f64) -> (f64, f64) {
    let (_, _, ec, vc) = lda_terms(3.0 / (4.0 * PI * rs.powi(3)));
    (
        ec / HARTREE_PER_RYDBERG,
        3.0 * (ec - vc) / rs / HARTREE_PER_RYDBERG,
    )
}

/// `g(r_s)` of Eq. (6), Rydberg.
pub fn g_function(rs: f64) -> f64 {
    G1 * rs * rs.ln() + G2 * rs + G3 * rs.powf(1.5) + G4 * rs * rs
}
/// `dg/dr_s`, Rydberg.
pub fn g_derivative(rs: f64) -> f64 {
    G1 * (rs.ln() + 1.0) + G2 + 1.5 * G3 * rs.sqrt() + 2.0 * G4 * rs
}

/// Infinite-size `e_xc` (Slater exchange plus Perdew-Zunger correlation), Hartree.
pub fn infinite_exchange_correlation(rs: f64) -> f64 {
    (a0() / rs + pz_correlation_rydberg(rs).0) * HARTREE_PER_RYDBERG
}

/// Residual of the `a_1/L^2` cancellation in `e_xc`, as a fraction of the term itself, maximum
/// over `rs`. At round-off wherever `r_s <= min(gamma, gamma_h)` (the caller picks the points).
pub fn leading_term_cancellation(rs: &[f64], length: f64) -> Result<f64> {
    if rs.is_empty() {
        return Err(invalid("leading_term_cancellation needs at least one r_s"));
    }
    let fit = Fit::new(length)?;
    let l3 = length.powi(3);
    Ok(rs.iter().fold(0.0, |worst, &r| {
        let term = (A1 / length.powi(2)).abs() * r * HARTREE_PER_RYDBERG;
        let without =
            (a0() / r + pz_correlation_rydberg(r).0 + (A2 / l3) * r * r + g_function(r) / l3)
                * HARTREE_PER_RYDBERG;
        let residual = (fit.exchange_correlation(r) - without).abs();
        f64::max(worst, residual / term.max(1e-300))
    }))
}

/// The size-dependent functional of one cubic supercell of side `length`: its three density
/// boundaries and the parameter-free intermediate cubic of the correlation.
#[derive(Clone, Copy, Debug)]
pub struct Fit {
    pub length: f64,
    /// `r_s(N = 2)`, `r_s(N = 12)`, `r_s(N = 1/2)`.
    pub gamma: f64,
    pub gamma_h: f64,
    pub gamma_l: f64,
    /// `f(r) = (r - gamma_l)^2 [alpha + beta (r - gamma_l)]` on `[gamma_h, gamma_l]`.
    pub alpha: f64,
    pub beta: f64,
}

impl Fit {
    pub fn new(length: f64) -> Result<Self> {
        if !length.is_finite() || length <= 0.0 {
            return Err(invalid(format!(
                "the supercell length must be finite and positive, got {length}"
            )));
        }
        let (gamma, gamma_h, gamma_l) =
            (rs_at(2.0, length), rs_at(12.0, length), rs_at(0.5, length));
        let mut fit = Self {
            length,
            gamma,
            gamma_h,
            gamma_l,
            alpha: 0.0,
            beta: 0.0,
        };
        let (value, slope) = fit.high_correlation_rydberg(gamma_h);
        let u = gamma_h - gamma_l; // negative
        fit.beta = (slope - 2.0 * value / u) / (u * u);
        fit.alpha = value / (u * u) - fit.beta * u;
        Ok(fit)
    }

    /// The `r_s <= gamma_h` correlation branch `(value, slope)`, Rydberg.
    pub fn high_correlation_rydberg(&self, rs: f64) -> (f64, f64) {
        let (l2, l3) = (self.length.powi(2), self.length.powi(3));
        let (ec, dec) = pz_correlation_rydberg(rs);
        (
            ec - (A1 / l2) * rs + g_function(rs) / l3,
            dec - A1 / l2 + g_derivative(rs) / l3,
        )
    }

    /// The intermediate cubic `(value, slope)`, Rydberg.
    pub fn cubic_rydberg(&self, rs: f64) -> (f64, f64) {
        let s = rs - self.gamma_l;
        (
            s * s * (self.alpha + self.beta * s),
            s * (2.0 * self.alpha + 3.0 * self.beta * s),
        )
    }

    /// Finite-size exchange per electron, Eq. (3), Hartree.
    pub fn exchange(&self, rs: f64) -> f64 {
        let l = self.length;
        let value = if rs <= self.gamma {
            a0() / rs + (A1 / l.powi(2)) * rs + (A2 / l.powi(3)) * rs.powi(2)
        } else {
            A3 * l.powi(5) / rs.powi(6)
        };
        value * HARTREE_PER_RYDBERG
    }

    /// Finite-size correlation per electron, Eq. (6), Hartree. `ec_rydberg` is the infinite-size
    /// correlation at `rs`; it is read only on the `r_s <= gamma_h` branch.
    fn correlation_with(&self, rs: f64, ec_rydberg: f64) -> f64 {
        let value = if rs <= self.gamma_h {
            let (l2, l3) = (self.length.powi(2), self.length.powi(3));
            ec_rydberg - (A1 / l2) * rs + g_function(rs) / l3
        } else if rs <= self.gamma_l {
            self.cubic_rydberg(rs).0
        } else {
            0.0
        };
        value * HARTREE_PER_RYDBERG
    }
    pub fn correlation(&self, rs: f64) -> f64 {
        let ec = if rs <= self.gamma_h {
            pz_correlation_rydberg(rs).0
        } else {
            0.0
        };
        self.correlation_with(rs, ec)
    }
    pub fn exchange_correlation(&self, rs: f64) -> f64 {
        self.exchange(rs) + self.correlation(rs)
    }
}

/// How the density is screened: nodes at or below `floor` are vacuum and skipped, and the run
/// refuses when the electron fraction in vacuum (`r_s > gamma_l`, or at or below `floor`)
/// exceeds `max_fraction` (FS-D4: the bound is the caller's, with no default).
#[derive(Clone, Copy, Debug)]
pub struct Vacuum {
    pub floor: f64,
    pub max_fraction: f64,
}

/// The correction and the bookkeeping that qualifies it. Energies in Hartree per supercell.
#[derive(Clone, Debug)]
pub struct Kzk {
    /// `xc_infinite - xc_finite`: add to the supercell energy.
    pub correction: f64,
    pub xc_infinite: f64,
    pub xc_finite: f64,
    /// The effective cubic `L`, Bohr.
    pub length: f64,
    pub scope: Scope,
    /// `int n` over the whole grid; equals the integrated part plus `electrons_below_floor`.
    pub n_electrons: f64,
    pub electrons_below_floor: f64,
    pub electrons_beyond_gamma_l: f64,
    /// `max |l_i - L| / L` over the lattice-vector lengths.
    pub cubic_deviation: f64,
    /// Smallest and largest `r_s` that contributed, Bohr.
    pub rs_range: (f64, f64),
}

impl Kzk {
    pub fn correction_per_cell(&self) -> f64 {
        self.scope.per_cell(self.correction)
    }
    pub fn corrected_energy_per_cell(&self, energy_per_cell: f64) -> f64 {
        energy_per_cell + self.correction_per_cell()
    }
    pub fn vacuum_fraction(&self) -> f64 {
        self.electrons_beyond_gamma_l / self.n_electrons
    }
}

/// Neumaier compensated sum.
#[derive(Default)]
struct Sum {
    s: f64,
    c: f64,
}
impl Sum {
    fn add(&mut self, v: f64) {
        let t = self.s + v;
        self.c += if self.s.abs() >= v.abs() {
            (self.s - t) + v
        } else {
            (v - t) + self.s
        };
        self.s = t;
    }
    fn value(&self) -> f64 {
        self.s + self.c
    }
}

fn determinant(c: &[[f64; 3]; 3]) -> f64 {
    c[0][0] * (c[1][1] * c[2][2] - c[1][2] * c[2][1])
        - c[0][1] * (c[1][0] * c[2][2] - c[1][2] * c[2][0])
        + c[0][2] * (c[1][0] * c[2][1] - c[1][1] * c[2][0])
}

/// Integrate `n (eps_xc^inf - eps_xc^FS)` over a supercell grid.
///
/// `density` is the electron density at the grid nodes (Bohr^-3) with node volume `dv`
/// (Bohr^3); the nodes must tile `cell` (rows are lattice vectors, Bohr). A negative, NaN or
/// infinite density, an all-vacuum density, a grid that does not tile the cell, or a vacuum
/// fraction above `vacuum.max_fraction` is an error: nothing is dropped silently.
pub fn kzk(
    density: &[f64],
    dv: f64,
    cell: &[[f64; 3]; 3],
    n_cells: usize,
    vacuum: Vacuum,
) -> Result<Kzk> {
    // Below 1e-30 `lda_terms` floors the density, so a smaller floor would change the number.
    if !vacuum.floor.is_finite() || vacuum.floor < 1e-30 {
        return Err(invalid(format!(
            "the vacuum floor must be finite and at least 1e-30, got {}",
            vacuum.floor
        )));
    }
    if !(0.0..=1.0).contains(&vacuum.max_fraction) {
        return Err(invalid(format!(
            "the vacuum fraction bound must lie in [0, 1], got {}",
            vacuum.max_fraction
        )));
    }
    if !dv.is_finite() || dv <= 0.0 || density.is_empty() {
        return Err(invalid(format!("the node volume must be finite and positive and the density non-empty (dv {dv}, {} nodes)", density.len())));
    }
    let norms = cell.map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt());
    let volume = determinant(cell).abs();
    if !volume.is_finite() || volume <= 1e-12 * norms[0] * norms[1] * norms[2] {
        return Err(invalid(
            "the supercell lattice vectors must be finite and linearly independent",
        ));
    }
    let scope = Scope::new(n_cells, volume)?;
    let covered = density.len() as f64 * dv;
    if (covered - volume).abs() > 1e-8 * volume {
        return Err(invalid(format!(
            "the {} grid nodes of volume {dv:e} cover {covered:e} Bohr^3, not the supercell volume {volume:e}",
            density.len()
        )));
    }
    let length = effective_length(volume);
    let cubic_deviation = norms
        .iter()
        .fold(0.0_f64, |m, l| m.max((l - length).abs() / length));
    let fit = Fit::new(length)?;

    let (mut total, mut below, mut beyond) = (Sum::default(), Sum::default(), Sum::default());
    let (mut infinite, mut finite) = (Sum::default(), Sum::default());
    let (mut rs_min, mut rs_max, mut kept) = (f64::INFINITY, 0.0_f64, 0usize);
    for (node, &n) in density.iter().enumerate() {
        if !n.is_finite() {
            return Err(invalid(format!(
                "density node {node} is not finite ({n}); no correction is computed from it"
            )));
        }
        if n < 0.0 {
            return Err(invalid(format!(
                "density node {node} is negative ({n:e}); no correction is computed from it"
            )));
        }
        total.add(n);
        if n <= vacuum.floor {
            below.add(n);
            beyond.add(n);
            continue;
        }
        let rs = rs_of_density(n);
        let (ec, _) = pz_correlation_rydberg(rs);
        infinite.add(n * (a0() / rs + ec) * HARTREE_PER_RYDBERG);
        finite.add(n * (fit.exchange(rs) + fit.correlation_with(rs, ec)));
        if rs > fit.gamma_l {
            beyond.add(n);
        }
        (rs_min, rs_max, kept) = (rs_min.min(rs), rs_max.max(rs), kept + 1);
    }
    if kept == 0 {
        return Err(invalid(
            "the density is everywhere at or below the vacuum floor, so there is nothing to integrate; \
             check that the state and grid are the ones the run produced",
        ));
    }
    let n_electrons = total.value() * dv;
    let electrons_beyond_gamma_l = beyond.value() * dv;
    if electrons_beyond_gamma_l > vacuum.max_fraction * n_electrons {
        return Err(invalid(format!(
            "{:.4} of the {n_electrons:.4} electrons sit in vacuum (r_s above gamma_l = {:.3} Bohr); \
             the KZK fit leaves its range there and the bound is {}",
            electrons_beyond_gamma_l / n_electrons, fit.gamma_l, vacuum.max_fraction
        )));
    }
    let (xc_infinite, xc_finite) = (infinite.value() * dv, finite.value() * dv);
    Ok(Kzk {
        correction: xc_infinite - xc_finite,
        xc_infinite,
        xc_finite,
        length,
        scope,
        n_electrons,
        electrons_below_floor: below.value() * dv,
        electrons_beyond_gamma_l,
        cubic_deviation,
        rs_range: (rs_min, rs_max),
    })
}
