//! The occupied orbitals of a converged state in the atomic-orbital basis.
//!
//! The state is a pair of orthonormal-frame densities `D` and the Fock matrix
//! `F` rebuilt from them. A converged state has `[F, D] = 0`; the occupied
//! orbitals are the eigenvectors of `F` within the occupied space of `D` (so a
//! degenerate partially filled shell is rotated, never re-chosen), and their
//! energy-weighted density `sum_i eps_i |i><i| = D F D` is the Pulay weight.
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
        if values[m - count..].iter().any(|v| (v - 1.).abs() > limit)
            || (count < m && values[m - count - 1].abs() > limit)
        {
            return Err(invalid(
                "the density of the state is not a projector on its occupied orbitals",
            ));
        }
        let space = vectors.columns(m - count, count).into_owned();
        let fock = real(f);
        let fock = (&fock + fock.transpose()) * 0.5;
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
        for (i, energy) in energies.into_iter().enumerate() {
            orbitals.push(Occupied {
                spin,
                weight: if restricted { 2. } else { 1. },
                energy,
                ao: rotated.column(i).iter().copied().collect(),
            });
        }
    }
    Ok(Frame {
        orbitals,
        stationarity,
    })
}
