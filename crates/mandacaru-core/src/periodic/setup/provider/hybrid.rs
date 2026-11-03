//! The hybrid half of the geometry provider: the explicit derivative of the screened exact
//! exchange at the occupied states of a state, with the basis functions of the grid pair moving
//! with their atoms ([`crate::periodic::hybrid_gradient`]).
//!
//! It needs the SCF crystal ([`SetupDerivatives::with_scf`]): the density matrices, the PAW
//! spheres the compensation charges are built from, the divergence treatment of the SCF. The
//! setup's atoms and moment channels must be those of the crystal's spheres, else the phase of
//! the compensation shapes would move another atom's charge.
use super::SetupDerivatives;
use crate::{
    periodic::{
        gradient::{scf, State},
        hybrid_gradient::{explicit, Gradient, Motion},
        invalid,
        screened_exchange::Screening,
        setup::pulay::KSamples,
        Crystal,
    },
    quantum::C64,
    Buffer, Error, Result,
};

/// Tolerance on the atom centers of the setup against the crystal's spheres (Bohr).
const CENTER_TOLERANCE: f64 = 1e-12;

/// The basis functions of the setup moving with their atoms: the center derivative of a Bloch
/// sum is minus its Cartesian derivative.
struct Moving<'p, 'a> {
    provider: &'p SetupDerivatives<'a>,
}

impl Motion for Moving<'_, '_> {
    /// The owners of the `M` functions of one spin block: the first `M` entries of a spinor's
    /// doubled list (the center derivatives are scalar, a bra row `(sigma, mu)` is `owner[mu]`).
    fn owner(&self) -> &[usize] {
        &self.provider.layout.row_owner[..self.provider.functions]
    }

    fn center_derivative(&self, c: &Crystal, k: usize) -> Result<[Buffer<C64>; 3]> {
        let p = self.provider;
        let samples = KSamples::new(
            p.ctx,
            p.inputs.shells,
            &p.inputs.sites.lattice,
            &p.mesh,
            p.kpoint(k)?,
        )?;
        let len = samples.slope(0).len();
        let mut out = [
            c.context.allocate::<C64>(len)?,
            c.context.allocate::<C64>(len)?,
            c.context.allocate::<C64>(len)?,
        ];
        for (axis, to) in out.iter_mut().enumerate() {
            for (x, y) in to.as_mut_slice().iter_mut().zip(samples.slope(axis)) {
                *x = -y;
            }
        }
        Ok(out)
    }
}

impl SetupDerivatives<'_> {
    /// The atoms and moment channels of the setup are the crystal's spheres' (hybrid crystals
    /// only; a semilocal crystal has none).
    pub(super) fn same_spheres(&self, crystal: &Crystal) -> Result<()> {
        if crystal.screened_exchange()?.is_none() {
            return Ok(());
        }
        let augmentation = crystal.hybrid_augmentation()?;
        let sites = augmentation.sites();
        let mine = &self.inputs.sites;
        let same =
            sites.atoms.len() == mine.atoms.len()
                && sites.channels == mine.channels
                && sites.atoms.iter().zip(mine.atoms).all(|(a, b)| {
                    (0..3).all(|x| (a.center[x] - b.center[x]).abs() < CENTER_TOLERANCE)
                });
        if !same {
            return Err(invalid(
                "crystal gradient: the setup's atoms and moment channels are not the PAW spheres \
                 of the hybrid crystal",
            ));
        }
        Ok(())
    }

    /// The explicit exact-exchange derivative at the density of `state`.
    pub(super) fn exchange_terms(&self, state: &State) -> Result<Gradient> {
        let Some((crystal, options)) = &self.scf else {
            return Err(Error::Unsupported(
                "the exchange derivative needs the SCF crystal: tie the geometry to it with \
                 with_scf"
                    .into(),
            ));
        };
        let densities = scf::densities(crystal, &state.bands)?;
        let augmentation = crystal.hybrid_augmentation()?;
        explicit(
            crystal,
            &densities,
            Screening::of(crystal)?,
            options.exxdiv_treatment,
            &augmentation,
            Some(&Moving { provider: self }),
        )
    }
}
