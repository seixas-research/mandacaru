//! The spinor half of the strain provider (AS-8): the doubled frame the contraction runs in, the
//! Pauli sum of the local term's channel tangents, and the spin-orbit union projectors with the
//! operator `B` they are contracted with.
//!
//! The frame is the force engine's ([`crate::periodic::gradient::spinor`]); this file only builds
//! it for the strain. Every spin-diagonal tangent is `I2 x dM` (the strain acts on space only,
//! `T (x) I2`), the local term is `sum_a sigma_a x dV^a` over the one (time-reversal) or four
//! Pauli channels of the potential, the spin-orbit projections stay in the scalar frame.
use super::{ProjectionTable, StrainSetup};
use crate::periodic::{
    gradient::{
        spin_orbit::operator,
        spinor::{doubled, doubled_layout},
        Layout, SpinOrbit,
    },
    invalid,
    setup::{provider::Operators, pulay::Matrix, spheres::Projector, spin_orbit::labels},
};
use crate::{Error, Result};
use std::collections::BTreeMap;

/// The spin-orbit union projectors of every atom and the operator over them.
pub(super) struct SpinOrbitData<'a> {
    pub projectors: &'a [Projector],
    pub operator: SpinOrbit,
}

/// The layout and operators the contraction reads: the doubled frame for a spinor request, the
/// scalar description itself otherwise.
pub(super) fn frame_of(layout: Layout, operators: Operators) -> Result<(Layout, Operators)> {
    if !layout.request.spinor {
        return Ok((layout, operators));
    }
    let operators = Operators {
        blocks: operators.blocks.iter().map(doubled).collect(),
        augmentation: doubled(&operators.augmentation),
        ion_operator: doubled(&operators.ion_operator),
    };
    Ok((doubled_layout(&layout)?, operators))
}

impl<'a> StrainSetup<'a> {
    /// `I2 x a` for a spinor, `a` itself otherwise.
    pub(super) fn frame(&self, a: Matrix) -> Matrix {
        if self.layout.request.spinor {
            doubled(&a)
        } else {
            a
        }
    }

    pub(super) fn spin_orbit_table(&self) -> Result<&ProjectionTable> {
        self.projections.spin_orbit(
            self.ctx,
            self.inputs.shells,
            self.lattice(),
            self.spin_orbit_data()?.projectors,
        )
    }

    /// Gives a spinor geometry its spin-orbit union projectors (every atom's, with
    /// `Projector::atom` the owner) and the radial couplings `D^SO` per `(atom, l)`. Empty
    /// couplings are a crystal whose spin-orbit matrix is zero (`B = 0`); the projections are still
    /// formed. Refused for a scalar request.
    pub fn with_spin_orbit(
        mut self,
        projectors: &'a [Projector],
        couplings: &BTreeMap<(usize, usize), crate::models::spin_orbit::Coupling>,
    ) -> Result<Self> {
        if !self.layout.request.spinor {
            return Err(invalid(
                "crystal stress: spin-orbit projectors belong to a spinor request",
            ));
        }
        let found =
            crate::models::spin_orbit::channels(self.ctx, &labels(projectors), couplings, false)?;
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
        Ok(self)
    }

    pub(super) fn spin_orbit_data(&self) -> Result<&SpinOrbitData<'a>> {
        self.spin_orbit.as_ref().ok_or_else(|| {
            Error::Unsupported("this strain provider carries no spin-orbit projectors".into())
        })
    }
}
