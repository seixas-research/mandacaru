//! Stationary PAW-LCAO gradient of a converged restricted or unrestricted
//! Hartree-Fock or semilocal Kohn-Sham state.
//!
//! Every matrix of the engine (core operator, Coulomb pair factors, PAW
//! augmentation factors, orbital samples) lives in the Lowdin-orthonormal
//! frame of its own geometry, so the orthonormality constraint on the
//! occupied orbitals does not depend on the atomic positions. At a converged
//! SCF the envelope theorem then gives the total derivative as the partial
//! derivative at **fixed orthonormal-frame densities**,
//!
//! ```text
//! dE/dR = d/dR E[P_orth; R],   E[P; R] = E_elec[P; R] + E_nuclear(R),
//! ```
//!
//! in which the overlap (Pulay) term, `-Tr[W dS/dR]` of an atomic-orbital
//! formulation, is carried by the position dependence of the Lowdin matrix
//! inside the rebuilt integrals. The partial derivative is a symmetric finite
//! difference of the geometry-dependent energy expression, evaluated on the
//! same fixed grid with the retained PAW setup (`at_geometry`), without any
//! displaced SCF. The finite difference differentiates, at once, the basis
//! and projector centers, overlap and compensation augmentation, the
//! compensation charges and core densities on the fixed grid, the one-center
//! Coulomb and exchange-correlation terms, frozen-core exchange, the local
//! potential, the kinetic energy and the ion-ion repulsion.
use super::{forces::Gradient, invalid, MeanField, MolecularIntegrals};
use crate::math::storage::Precision;
use crate::{product, quantum::C64, Geometry, LengthUnit, Result};
use nalgebra::DMatrix;

/// Displacement, in Bohr, of the coarse central difference. Richardson
/// extrapolation combines it with the half step, so the retained truncation
/// error is O(`STEP`^4) and the reported estimate is |fine - coarse| / 3.
pub const STEP: f64 = 1e-3;
type Matrix = DMatrix<C64>;

impl MolecularIntegrals {
    /// Electronic energy at these geometry-specific integrals for the given
    /// orthonormal-frame densities, without nuclear repulsion.
    fn state_energy(&self, state: &MeanField, d: &[Matrix]) -> Result<f64> {
        match state.functional.as_deref() {
            Some(name) => self.kohn_sham_energy(name, d),
            None => self.hartree_fock_energy(&state.method, d),
        }
    }
    /// Total energy of the densities of `state` carried to another geometry.
    /// The Coulomb factorization keeps the reference pivots so that its
    /// truncation is a smooth function of the positions.
    fn carried_energy(
        &self,
        state: &MeanField,
        atom: usize,
        axis: usize,
        delta: f64,
    ) -> Result<f64> {
        let mut positions = self.geometry.positions().to_vec();
        positions[atom][axis] += delta;
        let geometry = Geometry::new(
            self.geometry.numbers().to_vec(),
            positions,
            *self.geometry.cell(),
            self.geometry.pbc(),
            LengthUnit::Bohr,
        )?;
        let rebuilt = if state.functional.is_some() {
            self.at_geometry(geometry)?
        } else {
            self.at_geometry_with_fixed_pivots(geometry)?
        };
        let e = rebuilt.state_energy(state, &state.densities)?;
        Ok(e + rebuilt.nuclear)
    }
    pub(super) fn paw_gradient(&self, state: &MeanField) -> Result<Gradient> {
        let m = self.m();
        if self.paw_setup.is_none() {
            return Err(invalid(
                "forces need a PAW setup that rebuilds its geometry; custom molecular corrections have none",
            ));
        }
        if !["rhf", "uhf"].contains(&state.method.as_str()) {
            return Err(invalid(
                "PAW forces support restricted and unrestricted collinear states only",
            ));
        }
        if self.samples.precision() != Precision::Float64 {
            return Err(invalid(
                "PAW forces differentiate the energy and require float64 orbital samples",
            ));
        }
        if let Some(name) = state.functional.as_deref() {
            let definition = crate::models::functional(name)?;
            if definition.screened_exchange.is_some() {
                return Err(invalid(format!(
                    "PAW forces are not implemented for the screened exact exchange of {name}"
                )));
            }
            if definition.nonlocal_b.is_some() {
                return Err(invalid(format!(
                    "PAW forces are not implemented for the nonlocal correlation of {name}"
                )));
            }
        }
        let channels = if state.method == "rhf" { 1 } else { 2 };
        if state.densities.len() != channels
            || state.densities.iter().any(|d| d.shape() != (m, m))
            || state.particles.iter().any(|&n| n > m)
        {
            return Err(invalid(
                "SCF state dimensions do not match molecular integrals",
            ));
        }
        if state
            .densities
            .iter()
            .any(|d| d.iter().any(|v| v.im.abs() > 1e-12))
        {
            return Err(invalid("PAW forces require real collinear densities"));
        }
        self.context.run(|| {
            let ctx = &self.context;
            let atoms = self.geometry.numbers().len();
            let _workspace = ctx.reserve(product(&[3 * atoms, 64])?)?;
            let reference = self.state_energy(state, &state.densities)?;
            if (reference - state.electronic_energy).abs() > 1e-7 {
                return Err(invalid(
                    "the SCF state does not reproduce its energy on these integrals; it was not computed with them",
                ));
            }
            let mut total = ctx.allocate::<f64>(3 * atoms)?;
            let mut estimate = 0f64;
            for atom in 0..atoms {
                for axis in 0..3 {
                    let mut values = [0.; 4];
                    for (slot, sign) in [-1., 1., -0.5, 0.5].into_iter().enumerate() {
                        values[slot] = self.carried_energy(state, atom, axis, sign * STEP)?;
                    }
                    let coarse = (values[1] - values[0]) / (2. * STEP);
                    let fine = (values[3] - values[2]) / STEP;
                    estimate = estimate.max((fine - coarse).abs() / 3.);
                    total.as_mut_slice()[3 * atom + axis] = (4. * fine - coarse) / 3.;
                }
            }
            if total.as_slice().iter().any(|v| !v.is_finite()) {
                return Err(invalid("nonfinite stationary gradient"));
            }
            Ok(Gradient {
                hellmann_feynman: None,
                pulay: None,
                total,
                scheme: "Richardson-fixed-density-PAW",
                step: Some(STEP),
                estimate: Some(estimate),
            })
        })
    }
}
