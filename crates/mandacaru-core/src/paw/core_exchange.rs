//! The exchange of the valence with a frozen core, one-center, for the PAW Hartree-Fock
//! Hamiltonian (stage 4-0c of `dev_plan/BLOCH_VQE_PLAN.md`).
//!
//! A frozen core subshell `(n_c, l_c)` holding `q_c` electrons, spherically averaged (a closed
//! shell is spherical already), exchanges with a valence function of angular momentum `l`
//! through every multipole `L` the triangle `(l, L, l_c)` allows. Summed over the core's `m_c`,
//! the matrix of the core-valence exchange between two partial waves `phi_i`, `phi_j` of the
//! channel is diagonal in `m` and the same for every `m`:
//!
//! ```text
//! X_ij = - sum_c (q_c / 2) sum_L (l L l_c; 0 0 0)^2 R^L[phi_i R_c, phi_j R_c]
//! ```
//!
//! with `R_c = u_c / r` the core's radial function (the record's `core_wave` is `u_c`), `q_c / 2`
//! the electrons of one spin (the valence electron exchanges with its own spin), and `R^L` the
//! radial Slater integral of `pair_tables::coulomb`, symmetrized over its two arguments because
//! the discrete kernel is not exactly symmetric. For a closed `s` core (`q_c = 2`, `l_c = 0`) only
//! `L = l` survives, with `(l l 0; 0 0 0)^2 = 1 / (2l + 1)`: the term the PAW Hartree-Fock
//! Hamiltonian has used since it existed for a single closed 1s shell.
//!
//! [`FrozenCore`] is the core of a record's stored states, one owner for the molecule and the
//! PAW-LCAO supercell (`paw::molecular::data::Species`): it refuses, by reason, a core with no
//! operator (a subshell that is not closed, states that do not hold the core's electrons or do
//! not reproduce the stored core density). The operator enters the one-center coupling in the raw
//! projector basis as `V^-1 X V^-T`.
use super::{
    generate::core_states::{CHARGE_TOLERANCE, DENSITY_TOLERANCE},
    invalid,
    onecenter_radial::three_j_squared,
    pair_tables::coulomb,
};
use crate::{math::radial::integrate, Result};
use nalgebra::DMatrix;
use std::f64::consts::PI;

/// One frozen core subshell on the radial grid.
pub struct CoreShell<'a> {
    pub l: usize,
    /// Electrons in the subshell, `0 < occupation <= 2 (2l + 1)`.
    pub occupation: f64,
    /// `R(r) = u(r) / r` on the grid.
    pub radial: &'a [f64],
}

/// `X_ij` of the partial waves `waves` (on `r`, in the layout the species stores them) of the
/// channel `l`, over the first `extent` points of the grid (the core's support).
pub fn core_valence_exchange(
    r: &[f64],
    l: usize,
    waves: &[Vec<f64>],
    core: &[CoreShell<'_>],
    extent: usize,
) -> Result<DMatrix<f64>> {
    if extent < 4 || extent > r.len() {
        return Err(invalid("invalid core radial extent"));
    }
    if waves.iter().any(|w| w.len() < extent) || core.iter().any(|c| c.radial.len() < extent) {
        return Err(invalid(
            "a wave or a core orbital is shorter than the core's extent",
        ));
    }
    let refs = waves.len();
    let mut out = DMatrix::zeros(refs, refs);
    let r = &r[..extent];
    for shell in core {
        let capacity = (2 * (2 * shell.l + 1)) as f64;
        if !(shell.occupation > 0. && shell.occupation <= capacity + 1e-12) {
            return Err(invalid(format!(
                "a core subshell with l = {} holds {} electrons",
                shell.l, shell.occupation
            )));
        }
        let pairs: Vec<Vec<f64>> = waves
            .iter()
            .map(|w| (0..extent).map(|k| w[k] * shell.radial[k]).collect())
            .collect();
        for big_l in l.abs_diff(shell.l)..=l + shell.l {
            let angular = three_j_squared(l, big_l, shell.l)?;
            if angular == 0. {
                continue;
            }
            let factor = -0.5 * shell.occupation * angular;
            for i in 0..refs {
                for j in 0..refs {
                    out[(i, j)] += factor
                        * 0.5
                        * (coulomb(r, &pairs[i], &pairs[j], big_l)
                            + coulomb(r, &pairs[j], &pairs[i], big_l));
                }
            }
        }
    }
    Ok(out)
}

/// One core subshell as a record stores it: `orbital` is `u = r R(r)` on the record's radial grid
/// (the function its core density sums, `PawDataset::core_states`).
pub struct StoredShell<'a> {
    pub n: u32,
    pub l: u32,
    pub occupation: f64,
    pub orbital: &'a [f64],
}

/// The frozen core of a species as the core-valence exchange needs it: its closed subshells as
/// `R = u / r` on the record's grid and the first `extent` points of the grid that hold them (the
/// trapezoid charge of the stored core density reaches `1 - 1e-14` of its total, plus two).
pub struct FrozenCore {
    shells: Vec<(usize, f64, Vec<f64>)>,
    extent: usize,
}

impl FrozenCore {
    /// The core of `symbol` from its stored subshells, or the reason the Hartree-Fock exchange of
    /// this core has no operator: a subshell that is not closed (a spherical average of an open
    /// shell is not the exchange of any state), states that do not hold the `electrons` of the
    /// core, or states that do not reproduce the stored core `density` to
    /// [`DENSITY_TOLERANCE`] of its peak at every point (a corrupted or mismatched file).
    pub fn from_states(
        symbol: &str,
        r: &[f64],
        density: &[f64],
        electrons: f64,
        states: &[StoredShell<'_>],
    ) -> std::result::Result<Self, String> {
        if states.is_empty() {
            return Err(format!("{symbol}: the dataset lists no core states"));
        }
        let mut held = 0.;
        let mut rebuilt = vec![0.; r.len()];
        let mut shells = Vec::with_capacity(states.len());
        for s in states {
            let capacity = (2 * (2 * s.l + 1)) as f64;
            if (s.occupation - capacity).abs() > 1e-9 {
                return Err(format!(
                    "{symbol}: the core subshell ({}, {}) holds {} of {capacity} electrons; the \
                     core-valence exchange of a partially filled core has no operator",
                    s.n, s.l, s.occupation
                ));
            }
            if s.orbital.len() != r.len() {
                return Err(format!(
                    "{symbol}: the core state ({}, {}) has {} points on a grid of {}",
                    s.n,
                    s.l,
                    s.orbital.len(),
                    r.len()
                ));
            }
            held += s.occupation;
            let radial: Vec<f64> = s.orbital.iter().zip(r).map(|(u, x)| u / x).collect();
            for (rho, f) in rebuilt.iter_mut().zip(&radial) {
                *rho += s.occupation * f * f / (4. * PI);
            }
            shells.push((s.l as usize, s.occupation, radial));
        }
        if (held - electrons).abs() > CHARGE_TOLERANCE {
            return Err(format!(
                "{symbol}: the core states hold {held} electrons, the dataset's core {electrons}"
            ));
        }
        let peak = density.iter().fold(0f64, |m, v| m.max(v.abs()));
        let deviation = rebuilt
            .iter()
            .zip(density)
            .fold(0f64, |m, (a, b)| m.max((a - b).abs()))
            / peak;
        if !(deviation <= DENSITY_TOLERANCE) {
            return Err(format!(
                "{symbol}: the core states differ from the stored core density by {deviation:.2e} of \
                 its peak (tolerance {DENSITY_TOLERANCE:e})"
            ));
        }
        let total = integrate(
            r,
            &r.iter().zip(density).map(|(r, n)| 4. * PI * r * r * n).collect::<Vec<_>>(),
        )
        .map_err(|e| format!("{symbol}: {e}"))?;
        let mut integrated = 0.;
        let mut extent = r.len();
        for i in 1..r.len() {
            integrated += 0.5
                * (r[i] - r[i - 1])
                * 4.
                * PI
                * (r[i - 1] * r[i - 1] * density[i - 1] + r[i] * r[i] * density[i]);
            if integrated >= (1. - 1e-14) * total {
                extent = (i + 2).min(r.len());
                break;
            }
        }
        Ok(Self { shells, extent })
    }

    /// The core-valence exchange `X` of the partial waves `waves` (on the record's grid) of the
    /// channel `l` ([`core_valence_exchange`]).
    pub fn exchange(&self, r: &[f64], l: usize, waves: &[Vec<f64>]) -> Result<DMatrix<f64>> {
        let shells: Vec<CoreShell<'_>> = self
            .shells
            .iter()
            .map(|(l, occupation, radial)| CoreShell { l: *l, occupation: *occupation, radial })
            .collect();
        core_valence_exchange(r, l, waves, &shells, self.extent)
    }
}
