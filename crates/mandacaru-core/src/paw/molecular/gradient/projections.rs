//! The projections `C = <chi | p>` of the basis on the projectors of every atom
//! and the force they carry.
//!
//! `C[a', q]` integrates the basis function `a'` with the projector `q` of atom
//! `a` on a sphere around `a`. The sphere moves with `a` and the function with
//! its own atom, so the translation-invariance (ownership) identity gives
//! `dC[a', q] / dR_X = (delta_{X,a} - delta_{X,atom(a')}) G[a', q]` with
//! `G = <grad chi_a' | p_q>` on the same rule. Every energy term that depends on
//! the geometry through `C` is a matrix `Gamma = dE / dC` contracted with `G`.
use super::{
    super::{projection, projection_points, Setup},
    compensation::{Channel, Moments},
    Occupied,
};
use crate::{molecular::MolecularIntegrals, Result};
use nalgebra::{DMatrix, DVector};

fn sym(m: &DMatrix<f64>) -> DMatrix<f64> {
    (m + m.transpose()) * 0.5
}
/// `C` of every atom and `C^T a_i` of every occupied orbital.
pub(super) fn project(
    setup: &Setup,
    integrals: &MolecularIntegrals,
    occupied: &[Occupied],
) -> Result<(Vec<DMatrix<f64>>, Vec<Vec<DVector<f64>>>)> {
    let positions = integrals.geometry.positions();
    let mut matrices = Vec::new();
    let mut projected = Vec::new();
    for (a, &s) in setup.atoms.iter().enumerate() {
        let c = projection(
            &integrals.context,
            &integrals.basis,
            &setup.species[s],
            &setup.species_terms[s],
            positions[a],
        )?;
        projected.push(
            occupied
                .iter()
                .map(|o| c.transpose() * DVector::from_column_slice(&o.ao))
                .collect(),
        );
        matrices.push(c);
    }
    Ok((matrices, projected))
}

/// `G^axis[atom][a', q]`, the derivative samples of the projections.
pub(super) fn derivative_samples(
    setup: &Setup,
    integrals: &MolecularIntegrals,
) -> Result<Vec<[DMatrix<f64>; 3]>> {
    let m = integrals.m();
    let positions = integrals.geometry.positions();
    let mut out = Vec::new();
    for (a, &s) in setup.atoms.iter().enumerate() {
        let p = setup.species_terms[s].projectors.len();
        let mut samples = [(); 3].map(|_| DMatrix::<f64>::zeros(m, p));
        projection_points(
            &setup.species[s],
            &setup.species_terms[s],
            |offset, target| {
                let point = std::array::from_fn(|i| positions[a][i] + offset[i]);
                for (i, orbital) in integrals.basis.orbitals.iter().enumerate() {
                    let gradient = orbital.gradient(point);
                    for (q, &t) in target.iter().enumerate().filter(|(_, t)| **t != 0.) {
                        for axis in 0..3 {
                            samples[axis][(i, q)] += gradient[axis] * t;
                        }
                    }
                }
                Ok(())
            },
        )?;
        out.push(samples);
    }
    Ok(out)
}

/// `Pi[atom][orbital]`, the derivative of the energy with respect to the
/// projection of one orbital, for each of the energy terms that depend on `C`.
pub(super) struct Pi {
    pub potential: Vec<Vec<DVector<f64>>>,
    pub overlap: Vec<Vec<DVector<f64>>>,
    pub coulomb: Vec<Vec<DVector<f64>>>,
    pub exchange: Vec<Vec<DVector<f64>>>,
}
#[allow(clippy::too_many_arguments)]
pub(super) fn derivatives(
    setup: &Setup,
    channels: &[Channel<'_>],
    moments: &Moments,
    occupied: &[Occupied],
    projected: &[Vec<DVector<f64>>],
    hf: bool,
) -> Pi {
    let nocc = occupied.len();
    let nch = channels.len();
    let mut pi = Pi {
        potential: vec![],
        overlap: vec![],
        coulomb: vec![],
        exchange: vec![],
    };
    // Coulomb coefficient of a multipole's moment, and its exchange partner
    // for the orbital pair (i, j).
    let coulomb_coefficient = |a: usize| {
        moments.w[a]
            + (0..nch)
                .map(|b| moments.usym[(a, b)] * moments.q[b])
                .sum::<f64>()
    };
    for (atom, &s) in setup.atoms.iter().enumerate() {
        let terms = &setup.species_terms[s];
        let p = terms.projectors.len();
        let own: Vec<usize> = (0..nch).filter(|&a| channels[a].atom == atom).collect();
        let hp = sym(&if hf {
            terms.coupling.clone()
        } else {
            &terms.coupling + &terms.ks_difference
        });
        let overlap = sym(&terms.overlap);
        let blocks: Vec<DMatrix<f64>> = own.iter().map(|&a| sym(channels[a].block)).collect();
        // One-center Coulomb and exchange factors exist for the Hartree-Fock
        // model only; the Kohn-Sham model has them frozen in its one-body part.
        let factors: Vec<DMatrix<f64>> = if hf { terms.factors.clone() } else { vec![] };
        let moment: Vec<f64> = factors
            .iter()
            .map(|f| {
                (0..nocc)
                    .map(|o| {
                        occupied[o].weight * projected[atom][o].dot(&(f * &projected[atom][o]))
                    })
                    .sum()
            })
            .collect();
        let (mut potential, mut over, mut coulomb) = (vec![], vec![], vec![]);
        for o in 0..nocc {
            let f = occupied[o].weight;
            let x = &projected[atom][o];
            let mut vp = &hp * x;
            let mut vj = DVector::<f64>::zeros(p);
            for (slot, &a) in own.iter().enumerate() {
                vp += &blocks[slot] * x * moments.ionic[a];
                vj += &blocks[slot] * x * coulomb_coefficient(a);
            }
            for (k, factor) in factors.iter().enumerate() {
                vj += factor * x * (terms.weights[k] * moment[k]);
            }
            potential.push(vp * (2. * f));
            over.push((&overlap * x) * (-2. * f * occupied[o].energy));
            coulomb.push(vj * (2. * f));
        }
        pi.potential.push(potential);
        pi.overlap.push(over);
        pi.coulomb.push(coulomb);
    }
    if hf {
        pi.exchange = exchange_derivatives(
            setup,
            channels,
            moments,
            occupied,
            projected,
            &|s| {
                (
                    setup.species_terms[s].factors.clone(),
                    setup.species_terms[s].weights.clone(),
                )
            },
        );
    }
    pi
}
/// `Pi` of the exchange form `-1/2 sum_ij [sum_AB K_AB X_A,ij X_B,ij + sum_k w_k X_k,ij^2]`
/// over the occupied pairs of one spin, with the Coulomb coupling `usym` and the grid
/// moments `xw` of `moments` and the one-center factors and weights `one_center(species)`.
pub(super) fn exchange_derivatives(
    setup: &Setup,
    channels: &[Channel<'_>],
    moments: &Moments,
    occupied: &[Occupied],
    projected: &[Vec<DVector<f64>>],
    one_center: &dyn Fn(usize) -> (Vec<DMatrix<f64>>, Vec<f64>),
) -> Vec<Vec<DVector<f64>>> {
    let nocc = occupied.len();
    let nch = channels.len();
    let coefficient = |a: usize, i: usize, j: usize| {
        (0..nch)
            .map(|b| moments.usym[(a, b)] * moments.xq[(b * nocc + i) * nocc + j])
            .sum::<f64>()
            + moments.xw[(a * nocc + i) * nocc + j]
    };
    let mut out = Vec::new();
    for (atom, &s) in setup.atoms.iter().enumerate() {
        let p = projected[atom].first().map_or(0, |x| x.len());
        let own: Vec<usize> = (0..nch).filter(|&a| channels[a].atom == atom).collect();
        let blocks: Vec<DMatrix<f64>> = own.iter().map(|&a| sym(channels[a].block)).collect();
        let (factors, weights) = one_center(s);
        let mut rows = Vec::new();
        for o in 0..nocc {
            let mut vk = DVector::<f64>::zeros(p);
            for j in (0..nocc).filter(|&j| occupied[j].spin == occupied[o].spin) {
                for (slot, &a) in own.iter().enumerate() {
                    vk += &blocks[slot] * &projected[atom][j] * coefficient(a, o, j);
                }
                for (k, factor) in factors.iter().enumerate() {
                    let element = projected[atom][o].dot(&(factor * &projected[atom][j]));
                    vk += factor * &projected[atom][j] * (weights[k] * element);
                }
            }
            rows.push(vk * (-2. * occupied[o].weight));
        }
        out.push(rows);
    }
    out
}

/// The force on every atom from the energy's dependence on `C`, given
/// `Pi` of one energy term.
pub(super) fn force(
    integrals: &MolecularIntegrals,
    occupied: &[Occupied],
    samples: &[[DMatrix<f64>; 3]],
    pi: &[Vec<DVector<f64>>],
) -> Vec<f64> {
    let m = integrals.m();
    let atoms = integrals.geometry.numbers().len();
    let mut out = vec![0.; 3 * atoms];
    for (a, sample) in samples.iter().enumerate() {
        let p = sample[0].ncols();
        let gamma = DMatrix::from_fn(m, p, |i, q| {
            occupied
                .iter()
                .zip(&pi[a])
                .map(|(o, pi)| o.ao[i] * pi[q])
                .sum::<f64>()
        });
        for axis in 0..3 {
            for (i, orbital) in integrals.basis.orbitals.iter().enumerate() {
                let value: f64 = (0..p).map(|q| gamma[(i, q)] * sample[axis][(i, q)]).sum();
                out[3 * a + axis] += value;
                out[3 * orbital.atom + axis] -= value;
            }
        }
    }
    out
}
