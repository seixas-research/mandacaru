//! Basis-function motion on the grid: contraction of an action on the occupied
//! orbitals with the derivative of the basis functions.
//!
//! For an energy `E[psi_i]` of the occupied orbitals, `dE / d psi_i = 2 f_i A_i`
//! with `A_i` the action of the part of the Fock operator that acts on the
//! grid. Moving atom `X` moves its basis functions rigidly, `d chi_a / d R_X =
//! -grad chi_a`, so the force from that part is `sum_i 2 f_i sum_{a in X}
//! c_ia <-grad chi_a | A_i>`. One function at a time is differentiated, so the
//! derivative stack of the whole basis never exists.
use crate::basis::Radial;
use crate::molecular::MolecularIntegrals;
use crate::paw::molecular::gradient::Occupied;
use crate::{product, Buffer, Result};

/// The grid points as `[x, y, z]` triples.
pub(super) fn points(integrals: &MolecularIntegrals) -> Result<Buffer<f64>> {
    let n = integrals.grid.index.points();
    let mut out = integrals.context.allocate::<f64>(product(&[3, n])?)?;
    for k in 0..n {
        out.as_mut_slice()[3 * k..3 * k + 3].copy_from_slice(&integrals.grid.point(k));
    }
    Ok(out)
}
/// The force on every atom from one action (`nocc * n` values).
pub(super) fn contract(
    integrals: &MolecularIntegrals,
    occupied: &[Occupied],
    points: &[f64],
    action: &[f64],
) -> Vec<f64> {
    let n = integrals.grid.index.points();
    let dv = integrals.grid.index.dv();
    let mut out = vec![0.; 3 * integrals.geometry.numbers().len()];
    let mut weight = vec![0.; occupied.len()];
    for (a, orbital) in integrals.basis.orbitals.iter().enumerate() {
        for (w, o) in weight.iter_mut().zip(occupied) {
            *w = 2. * o.weight * o.ao[a];
        }
        if weight.iter().all(|w| *w == 0.) {
            continue;
        }
        // A tabulated function is zero past its last node.
        let support2 = match &orbital.radial {
            Radial::Tabulated(table) => table
                .knots
                .as_slice()
                .last()
                .map_or(f64::INFINITY, |r| r * r),
            _ => f64::INFINITY,
        };
        let mut sum = [0.; 3];
        for k in 0..n {
            let r2: f64 = (0..3)
                .map(|i| (points[3 * k + i] - orbital.center[i]).powi(2))
                .sum();
            if r2 > support2 {
                continue;
            }
            let t: f64 = weight
                .iter()
                .enumerate()
                .map(|(o, w)| w * action[o * n + k])
                .sum();
            if t == 0. {
                continue;
            }
            let gradient = orbital.gradient([points[3 * k], points[3 * k + 1], points[3 * k + 2]]);
            for axis in 0..3 {
                sum[axis] += gradient[axis] * t;
            }
        }
        for axis in 0..3 {
            out[3 * orbital.atom + axis] -= dv * sum[axis];
        }
    }
    out
}
