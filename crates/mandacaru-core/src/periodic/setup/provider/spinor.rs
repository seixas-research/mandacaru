//! The spinor half of the geometry provider: the spin-orbit union projectors and couplings of the
//! dataset, the operator `B` of [`SpinOrbit`] built from them, and the tie of the force's
//! `h^SO(k)` to the one the SCF crystal diagonalized with, and the doubled frame (`I2 x (.)`, the
//! Pauli sum of the local term) the engine contracts a spinor in.
//!
//! The projections of the Bloch sums on the spin-orbit union list stay in the scalar frame
//! (`M x P_so`); [`SpinOrbit::add_k`] contracts them with the `2M x 2M` density. The coupling is
//! applied by [`pair`](crate::models::spin_orbit::pair) alone, the single source of the angular
//! algebra.
use super::{Fields, SetupDerivatives};
use crate::periodic::{
    gradient::{
        spin_orbit::{operator, SpinOrbit},
        spinor::{doubled, pauli_sum},
    },
    invalid,
    setup::{
        pulay::{grid_cross, projection, tau_cross, Matrix},
        spheres::Projector,
        spin_orbit::labels,
    },
};
use crate::{Error, Result};
use std::collections::BTreeMap;

/// Largest relative gap, over the entries of `h^SO`, between the force's block and the crystal's:
/// the two agree to 3.5e-13 (`setup_spin_orbit.rs`, the Python-built matrix against the Rust
/// setup), so this is about 30x that.
const BLOCK_TOLERANCE: f64 = 1e-11;
/// Below this a spin-orbit matrix element (Hartree) is numerical zero, not a signal to tie (the
/// Li2 basis has no `p` function: its block is 2e-19 and ours is exactly zero).
const BLOCK_ZERO: f64 = 1e-15;

/// The spin-orbit union projectors of every atom and the operator over them.
pub(super) struct SpinOrbitData<'a> {
    pub projectors: &'a [Projector],
    pub operator: SpinOrbit,
}

impl<'a> SetupDerivatives<'a> {
    /// The frame the engine contracts in: `I2 x a` for a spinor, `a` itself otherwise.
    pub(super) fn frame(&self, a: Matrix) -> Matrix {
        if self.layout.request.spinor {
            doubled(&a)
        } else {
            a
        }
    }

    /// The local term of a spinor: `sum_a sigma_a x G^a` over the Pauli channels of the fields.
    pub(super) fn spinor_local(&self, k: usize, fields: &Fields) -> Result<[Matrix; 3]> {
        let mut channels = Vec::with_capacity(fields.local.len());
        for v in &fields.local {
            channels.push(self.with_samples(k, |s| grid_cross(self.ctx, &self.mesh, s, Some(v)))?);
        }
        let mut out = [
            Matrix::zeros(0, 0),
            Matrix::zeros(0, 0),
            Matrix::zeros(0, 0),
        ];
        for (axis, to) in out.iter_mut().enumerate() {
            let parts: Vec<Matrix> = channels.iter().map(|g| g[axis].clone()).collect();
            *to = pauli_sum(&parts)?;
        }
        Ok(out)
    }

    /// P6 for a spinor: the meta-GGA operator of each Pauli channel of `v_tau`
    /// (`tau_cross`), combined like the local potential's (`pauli_sum`).
    pub(super) fn spinor_kinetic(&self, k: usize, fields: &Fields) -> Result<[Matrix; 3]> {
        let (blocks, _) = fields.kinetic.as_ref().ok_or_else(|| {
            invalid("crystal gradient: the meta-GGA term needs the fields' v_tau")
        })?;
        let mut channels = Vec::with_capacity(blocks.len());
        for v in blocks {
            channels.push(self.with_samples(k, |s| tau_cross(self.ctx, self.inputs.grid, s, v))?);
        }
        let mut out = [
            Matrix::zeros(0, 0),
            Matrix::zeros(0, 0),
            Matrix::zeros(0, 0),
        ];
        for (axis, to) in out.iter_mut().enumerate() {
            let parts: Vec<Matrix> = channels.iter().map(|g| g[axis].clone()).collect();
            *to = pauli_sum(&parts)?;
        }
        Ok(out)
    }

    /// Gives a spinor geometry its spin-orbit union projectors (every atom's, with `Projector::atom`
    /// the owner) and the radial couplings `D^SO` per `(atom, l)`. Empty couplings are a crystal
    /// whose spin-orbit matrix is zero (`B = 0`); the projections are still formed. Refused for a
    /// scalar request.
    pub fn with_spin_orbit(
        mut self,
        projectors: &'a [Projector],
        couplings: &BTreeMap<(usize, usize), crate::models::spin_orbit::Coupling>,
    ) -> Result<Self> {
        if !self.layout.request.spinor {
            return Err(invalid(
                "crystal gradient: spin-orbit projectors belong to a spinor request",
            ));
        }
        let found = crate::models::spin_orbit::channels(self.ctx, &labels(projectors), couplings, false)?;
        let b = operator(self.ctx, projectors.len(), &found)?;
        let owner = self.layout.row_owner[..self.functions].to_vec();
        let operator = SpinOrbit::new(
            b,
            owner,
            projectors.iter().map(|p| p.atom).collect(),
            self.layout.atoms,
        )?;
        self.spin_orbit = Some(SpinOrbitData {
            projectors,
            operator,
        });
        self.same_spin_orbit()?;
        Ok(self)
    }

    pub(super) fn spin_orbit_data(&self) -> Result<&SpinOrbitData<'a>> {
        self.spin_orbit.as_ref().ok_or_else(|| {
            Error::Unsupported("this geometry provider carries no spin-orbit projectors".into())
        })
    }

    /// With both the SCF crystal and the spin-orbit data present: `h^SO(k)` of the force's
    /// projections and operator is the matrix the crystal diagonalizes with, at every k-point.
    /// The force then differentiates the crystal's own spin-orbit term (a wrong coupling table,
    /// projector list or ordering would otherwise differentiate another matrix).
    pub(super) fn same_spin_orbit(&self) -> Result<()> {
        let (Some((crystal, _)), Some(so)) = (&self.scf, &self.spin_orbit) else {
            return Ok(());
        };
        for (k, point) in crystal.points.iter().enumerate() {
            let c = projection(
                self.ctx,
                self.inputs.shells,
                &self.inputs.sites.lattice,
                self.kpoint(k)?,
                so.projectors,
            )?;
            let ours = so.operator.matrix(&c);
            let Some(theirs) = &point.spin_orbit else {
                return Err(invalid(
                    "crystal gradient: the SCF spinor crystal carries no spin-orbit matrix",
                ));
            };
            if theirs.shape() != ours.shape() {
                return Err(invalid(
                    "crystal gradient: spin-orbit matrix shapes disagree",
                ));
            }
            let peak = theirs.iter().map(|x| x.norm()).fold(0f64, f64::max);
            let gap = (&ours - theirs)
                .iter()
                .map(|x| x.norm())
                .fold(0f64, f64::max);
            if gap > BLOCK_TOLERANCE * peak && gap > BLOCK_ZERO {
                return Err(invalid(format!(
                    "crystal gradient: the spin-orbit matrix of k-point {k} differs from the \
                     crystal's by {gap:.3e} (peak {peak:.3e})"
                )));
            }
        }
        Ok(())
    }
}
