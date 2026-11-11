//! The DFT+U term of the gradient: the dependence of the Hubbard energy on the geometry
//! through the shell projections.
//!
//! The shell occupation is `n^s = q^T P^s q` with `P^s = sum_{i in s} a_i a_i^T` the density of the
//! atomic-orbital coefficients and `q[mu, m] = sum_p C[mu, p] w_p` the projection of the basis on
//! the shell's projectors (`w_p` the weight of the partial wave of `p`). At fixed `P` only `C`
//! moves, so `dE_U / dC` is a matrix `Gamma` and the force is its contraction with the projection
//! derivatives like every other term of this module. The Loewdin factor and the response of the
//! orbitals are in the overlap term `-tr[W dS]` (`W` carries the +U Fock matrix) and in
//! stationarity, as for every other energy term.
use super::{super::hubbard::shell_projectors, Occupied};
use crate::{molecular::MolecularIntegrals, Result};
use nalgebra::{DMatrix, DVector};

/// `Pi[atom][orbital] = dE_U / d(C^T a_i)` of the occupied orbitals, or `None` for a molecule
/// without +U. `unrestricted` tells whether the state has two spin channels: an unrestricted
/// state with no beta electron still has a beta channel the energy sees.
pub(super) fn derivatives(
    setup: &super::Setup,
    integrals: &MolecularIntegrals,
    occupied: &[Occupied],
    projected: &[Vec<DVector<f64>>],
    unrestricted: bool,
) -> Result<Option<Vec<Vec<DVector<f64>>>>> {
    let Some(hubbard) = integrals
        .corrections
        .as_ref()
        .and_then(|c| c.hubbard.as_ref())
    else {
        return Ok(None);
    };
    let spins = if unrestricted { 2 } else { 1 };
    let mut pi: Vec<Vec<DVector<f64>>> = setup
        .atoms
        .iter()
        .map(|&s| vec![DVector::zeros(setup.species_terms[s].projectors.len()); occupied.len()])
        .collect();
    for shell in &hubbard.shells {
        let terms = &setup.species_terms[setup.atoms[shell.atom]];
        let map = shell_projectors(terms, shell.l, &shell.weights);
        let size = shell.size();
        // `x_i = q^T a_i`, the orbital's amplitude on the shell.
        let amplitudes: Vec<DVector<f64>> = projected[shell.atom]
            .iter()
            .map(|c| {
                let mut x = DVector::zeros(size);
                for &(p, m, w) in &map {
                    x[m] += w * c[p];
                }
                x
            })
            .collect();
        // Only an orbital's density role (`weight > 0`) builds `n`.
        let mut n = vec![DMatrix::<f64>::zeros(size, size); spins];
        for (o, x) in occupied.iter().zip(&amplitudes) {
            if o.weight > 0. {
                n[o.spin] += x * x.transpose();
            }
        }
        let (_, potentials) = hubbard.shell_response(&integrals.context, shell, &n)?;
        for (i, (o, x)) in occupied.iter().zip(&amplitudes).enumerate() {
            // `dE = sum_s tr[V^s dn^s]` and `dn = dq^T a a^T q + h.c.` give `2 weight w_p (V x)_m`.
            let field = &potentials[o.spin] * x;
            for &(p, m, w) in &map {
                pi[shell.atom][i][p] += 2. * o.weight * w * field[m];
            }
        }
    }
    Ok(Some(pi))
}
