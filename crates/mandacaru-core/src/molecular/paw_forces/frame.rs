//! The occupied orbitals of a converged state in the atomic-orbital basis.
//!
//! The state is a pair of orthonormal-frame densities `D` and the Fock matrix
//! `F` rebuilt from them. A converged state has `[F, D] = 0`; the occupied
//! orbitals are the eigenvectors of `F` within each occupation group of `D` (so a
//! degenerate partially filled shell is rotated, never re-chosen), scaled by
//! `sqrt(f_i)`, and their energy-weighted density `sum_i f_i eps_i |i><i| = F D`
//! is the Pulay weight. An aufbau state has one group (`f = 1`);
//! Fermi-Dirac occupations add one group per fractional value. At fixed
//! occupations the fixed-density formula is the derivative of the Mermin free
//! energy `E - width S`, which the SCF minimizes.
use crate::molecular::{invalid, MeanField, MolecularIntegrals};
use crate::paw::molecular::gradient::Occupied;
use crate::{math::linalg::real_eigen, quantum::C64, Result};
use nalgebra::DMatrix;

/// Largest element of `(1 - D) F D` the state may have: the envelope theorem
/// is exact at zero and linear in it otherwise.
pub(super) fn stationarity_limit(float32: bool) -> f64 {
    // Measured: 5e-11 (water, default tolerance) to 1e-7 (a synthetic H2 at a 1e-6
    // tolerance) in float32; 1e-13 and below in float64.
    if float32 {
        1e-5
    } else {
        1e-6
    }
}

pub(super) struct Frame {
    pub orbitals: Vec<Occupied>,
    pub stationarity: f64,
}
fn real(matrix: &DMatrix<C64>) -> DMatrix<f64> {
    matrix.map(|v| v.re)
}
pub(super) fn occupied(
    integrals: &MolecularIntegrals,
    state: &MeanField,
    fock: &[DMatrix<C64>],
) -> Result<Frame> {
    let m = integrals.m();
    let restricted = state.method == "rhf";
    let limit =
        stationarity_limit(integrals.precision() == crate::math::storage::Precision::Float32);
    let mut orbitals = Vec::new();
    let mut stationarity = 0f64;
    for (spin, f) in fock.iter().enumerate() {
        let count = if restricted {
            state.particles[0]
        } else {
            state.particles[spin]
        };
        if count == 0 {
            continue;
        }
        let density = real(&state.densities[spin]);
        let (values, vectors) = real_eigen((&density + density.transpose()) * 0.5)?;
        // Occupation groups of the density (ascending eigenvalues): an aufbau state has one,
        // its eigenvalues 1; an ensemble adds the degenerate shell's shared value. Each group
        // must be one of the state's own occupations, and no eigenvalue may lie between them.
        let occupations = &state.occupations[if restricted { 0 } else { spin }];
        let mut levels: Vec<f64> = occupations.iter().copied().filter(|&o| o > 0.).collect();
        levels.sort_by(f64::total_cmp);
        levels.dedup_by(|a, b| (*a - *b).abs() <= limit);
        let mut groups: Vec<(f64, Vec<usize>)> = levels.iter().map(|&o| (o, vec![])).collect();
        for (j, &value) in values.iter().enumerate() {
            match groups.iter_mut().find(|(o, _)| (value - o).abs() <= limit) {
                Some((_, members)) => members.push(j),
                None if value.abs() <= limit => {}
                None => {
                    return Err(invalid(
                        "the density of the state is not built from its occupations",
                    ))
                }
            }
        }
        let expected: f64 = occupations.iter().sum();
        let held: f64 = groups
            .iter()
            .map(|(o, members)| o * members.len() as f64)
            .sum();
        if (held - expected).abs() > limit * m as f64 || (expected - count as f64).abs() > limit {
            return Err(invalid(
                "the density of the state is not built from its occupations",
            ));
        }
        let fock = real(f);
        let fock = (&fock + fock.transpose()) * 0.5;
        // [F, D] = 0 makes F block-diagonal across occupation groups: the residual of F on each
        // group's space is the stationarity, and F is diagonalized inside each group.
        for (occupation, members) in groups {
            if members.is_empty() {
                continue;
            }
            let space = DMatrix::from_fn(m, members.len(), |r, c| vectors[(r, members[c])]);
            let block = space.transpose() * &fock * &space;
            let residual = &fock * &space - &space * &block;
            stationarity = stationarity.max(residual.iter().fold(0f64, |a, v| a.max(v.abs())));
            if stationarity > limit {
                return Err(invalid(format!(
                    "the SCF state is not stationary: (1 - D) F D reaches {stationarity:.2e}, above the {limit:.0e} that its precision allows"
                )));
            }
            let (energies, rotation) = real_eigen((&block + block.transpose()) * 0.5)?;
            let rotated = &integrals.lowdin * (&space * rotation);
            // A fractional orbital enters as `sqrt(f) a` with the channel's whole weight: every
            // term quadratic in the orbitals (density, kinetic, projections, `W`) then carries
            // `f`, and the exchange pairs of a hybrid `f_i f_j`, as the energy has them. An
            // integer orbital is unchanged.
            let degeneracy = if restricted { 2. } else { 1. };
            let scale = occupation.sqrt();
            for (i, energy) in energies.into_iter().enumerate() {
                orbitals.push(Occupied {
                    spin,
                    weight: degeneracy,
                    energy,
                    w_weight: degeneracy * energy,
                    ao: rotated.column(i).iter().map(|v| scale * v).collect(),
                });
            }
        }
    }
    Ok(Frame {
        orbitals,
        stationarity,
    })
}
