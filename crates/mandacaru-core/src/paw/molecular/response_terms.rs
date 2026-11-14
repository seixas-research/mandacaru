//! The one-center pieces of a PAW-LCAO basis that linear response reads: the position
//! operator, the gradient (through `paw::momentum`) and the count of the one-center Coulomb
//! factors.
//!
//! Inside a sphere the all-electron pair density differs from the smooth one by
//! `sum_ij <chi|p_i> (phi_i phi_j - phi~_i phi~_j) <p_j|chi>`. Its integral against `r - O` is
//! `(R_A - O) q_ij + d_ij`, with `q_ij` the overlap correction the orthonormal frame is built
//! with and `d_ij` the L = 1 multipole blocks of the compensation charges (real harmonics
//! `r Y_1m = sqrt(3 / 4 pi) {y, z, x}` for `m = -1, 0, 1`), so `d = sqrt(4 pi / 3) Delta^(1)`.
//! These are exactly the moments the compensation charges give the smooth density, so the
//! position integrals of the basis carry the all-electron dipole of every pair density.
use super::Setup;
use crate::{
    paw::momentum::{one_center_gradient, SphereWaves},
    Result,
};
use nalgebra::DMatrix;
use std::f64::consts::PI;

/// One atom's blocks in its projector basis: `q` and the Cartesian `d_x, d_y, d_z`.
pub(crate) struct PositionBlocks {
    pub overlap: DMatrix<f64>,
    pub dipole: [DMatrix<f64>; 3],
}

impl Setup {
    /// How many of the Hartree-Fock pair factors (`MolecularCorrections::pairs`, in order) are the
    /// one-center `DeltaW` tensors of the atoms; the compensation-charge factors follow them.
    pub(crate) fn one_center_factor_count(&self) -> usize {
        self.atoms
            .iter()
            .map(|&s| self.species_terms[s].factors.len())
            .sum()
    }
    /// The one-center position blocks of every atom, in the order of the geometry.
    pub(crate) fn position_blocks(&self) -> Vec<PositionBlocks> {
        let scale = (4. * PI / 3.).sqrt();
        self.atoms
            .iter()
            .map(|&s| {
                let terms = &self.species_terms[s];
                let p = terms.overlap.nrows();
                let mut dipole = [
                    DMatrix::zeros(p, p),
                    DMatrix::zeros(p, p),
                    DMatrix::zeros(p, p),
                ];
                for multipole in terms.multipoles.iter().filter(|m| m.l == 1) {
                    let axis = match multipole.m {
                        -1 => 1,
                        0 => 2,
                        _ => 0,
                    };
                    dipole[axis] += &multipole.block * scale;
                }
                PositionBlocks {
                    overlap: terms.overlap.clone(),
                    dipole,
                }
            })
            .collect()
    }

    /// The one-center part of the gradient operator of every atom, in its projector basis
    /// (`paw::momentum::one_center_gradient`, the routine the crystal's momentum shares).
    pub(crate) fn gradient_blocks(&self, ctx: &crate::Context) -> Result<Vec<[DMatrix<f64>; 3]>> {
        let mut out = Vec::with_capacity(self.atoms.len());
        for &s in &self.atoms {
            let species = &self.species[s];
            let channels: Vec<_> = species
                .channels
                .iter()
                .map(|c| SphereWaves {
                    radius: c.radius,
                    ae: &c.ae,
                    ps: &c.ps,
                })
                .collect();
            out.push(one_center_gradient(
                ctx,
                species.ionic.knots.as_slice(),
                &channels,
                &self.species_terms[s].projectors,
            )?);
        }
        Ok(out)
    }
}
