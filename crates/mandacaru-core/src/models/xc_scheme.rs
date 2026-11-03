//! The radial exchange-correlation scheme of a spherical atom and of the PAW one-center terms.
//!
//! Two radial schemes exist (D-2, `PAW_COMPLETION_PLAN.md`):
//!
//! * [`Variational`], the scheme of the molecular PAW path: the potential is the exact derivative
//!   of the discrete energy quadrature (`spherical::exchange_correlation`). It is the production
//!   scheme everywhere: the reference-atom SCF, the unscreening and the one-center terms of a
//!   dataset all read the same functional derivative (D-2b), so the all-electron partial waves
//!   are eigenfunctions of the potential the unscreening subtracts.
//! * [`AtomSolver`], the scheme the 431 shipped datasets were produced with (derivatives on subset
//!   nodes, splined back; `radial_xc`). It exists only to prove the port against the frozen
//!   Python numbers and to diagnose a shipped file against the atom it was made on; nothing
//!   selects it at run time and the production constructors never take it.
use super::{radial_xc, spherical};
use crate::{Context, Result};

/// Energy per electron and potential of a spherical density on a radial mesh.
pub struct XcField {
    pub per_particle: Vec<f64>,
    pub potential: Vec<f64>,
}

/// A radial exchange-correlation scheme.
pub trait XcSource {
    fn evaluate(
        &self,
        ctx: &Context,
        r: &[f64],
        rho: &[f64],
        functional: &str,
        relativistic: bool,
    ) -> Result<XcField>;
}

/// The variational radial scheme: the potential differentiates the discrete energy quadrature.
pub struct Variational;

impl XcSource for Variational {
    fn evaluate(
        &self,
        ctx: &Context,
        r: &[f64],
        rho: &[f64],
        functional: &str,
        relativistic: bool,
    ) -> Result<XcField> {
        let out = spherical::exchange_correlation(ctx, r, rho, functional, relativistic)?;
        Ok(XcField {
            per_particle: out.per_particle.as_slice().to_vec(),
            potential: out.potential.as_slice().to_vec(),
        })
    }
}

/// The scheme of the shipped datasets and of the frozen Python numbers (the legacy seam).
#[doc(hidden)]
pub struct AtomSolver;

impl XcSource for AtomSolver {
    fn evaluate(
        &self,
        ctx: &Context,
        r: &[f64],
        rho: &[f64],
        functional: &str,
        relativistic: bool,
    ) -> Result<XcField> {
        let (per_particle, potential) =
            radial_xc::xc_potential(ctx, r, rho, functional, relativistic)?;
        Ok(XcField {
            per_particle,
            potential,
        })
    }
}
