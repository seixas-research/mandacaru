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
use super::{super::hubbard::shell_projectors, projections, Occupied};
use crate::{molecular::MolecularIntegrals, quantum::C64, Result};
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
        // An orbital's density role (`weight != 0`) builds the per-spin `n` with its weight per
        // spin channel: the state's own orbitals carry the degeneracy (one, so `n = sum x x^T`),
        // the signed pseudo-orbitals of a general density (an excited state's `D +- P`) their
        // eigenvalue.
        let degeneracy = if unrestricted { 1. } else { 2. };
        let mut n = vec![DMatrix::<f64>::zeros(size, size); spins];
        for (o, x) in occupied.iter().zip(&amplitudes) {
            if o.weight != 0. {
                n[o.spin] += x * x.transpose() * (o.weight / degeneracy);
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

/// The geometry derivative of `d^2 E_U / dt^2` along the imaginary Hermitian direction
/// `D + t i A` of the antisymmetric raw-AO spin matrices `a` (alpha and beta), at fixed `a`, three
/// per atom; `None` for a molecule without +U. With `q` the shell's raw-AO projection
/// (`MolecularHubbardShell::q_ao`), the occupation change is `i q^T A_s q` and the potential change
/// `i v_s` (`MolecularHubbard::shell_linear`, exact: the energy is quadratic in `n`), so
/// `d^2 E_U / dt^2 = -sum_s tr[v_s q^T A_s q]`, whose derivative at fixed `A` is
/// `-4 sum_s tr[v_s q^T A_s dq]` (the antisymmetry of `A_s` and `v_s` makes the two factors of
/// `q` equal). `dq` is the derivative of the projections, contracted like every other term of
/// this module, with the basis functions themselves as the orbitals.
pub fn imaginary_force(
    setup: &super::Setup,
    integrals: &MolecularIntegrals,
    a: [&DMatrix<f64>; 2],
) -> Result<Option<Vec<f64>>> {
    let Some(hubbard) = integrals
        .corrections
        .as_ref()
        .and_then(|c| c.hubbard.as_ref())
    else {
        return Ok(None);
    };
    let m = integrals.m();
    // `gamma[atom][mu][p] = dE / d<p|chi_mu>`.
    let mut gamma: Vec<Vec<DVector<f64>>> = setup
        .atoms
        .iter()
        .map(|&s| vec![DVector::zeros(setup.species_terms[s].projectors.len()); m])
        .collect();
    for shell in &hubbard.shells {
        let terms = &setup.species_terms[setup.atoms[shell.atom]];
        let map = shell_projectors(terms, shell.l, &shell.weights);
        let q = &shell.q_ao;
        let changes: Vec<DMatrix<C64>> = a
            .iter()
            .map(|a| (q.transpose() * *a * q).map(|v| C64::new(0., v)))
            .collect();
        let potentials = hubbard.shell_linear(&integrals.context, shell, &changes)?;
        for (a, v) in a.iter().zip(&potentials) {
            let field = *a * q * v.map(|z| z.im) * -4.;
            for mu in 0..m {
                for &(p, k, w) in &map {
                    gamma[shell.atom][mu][p] += w * field[(mu, k)];
                }
            }
        }
    }
    let basis: Vec<Occupied> = (0..m)
        .map(|mu| Occupied {
            spin: 0,
            weight: 0.,
            energy: 0.,
            w_weight: 0.,
            ao: (0..m).map(|nu| if nu == mu { 1. } else { 0. }).collect(),
        })
        .collect();
    let samples = projections::derivative_samples(setup, integrals)?;
    Ok(Some(projections::force(
        integrals, &basis, &samples, &gamma,
    )))
}
