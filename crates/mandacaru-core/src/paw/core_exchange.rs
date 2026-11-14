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
use super::{invalid, onecenter_radial::three_j_squared, pair_tables::coulomb};
use crate::Result;
use nalgebra::DMatrix;

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
