//! Hartree and exchange-correlation energies of the spherical densities of a reference atom, and
//! the frozen one-center constant built from them. Atomic units.
//!
//! The one-center constant is the all-electron valence energy minus the smooth one, both
//! written as the band energy less the double-counting terms, so it needs the energy and the
//! `int rho v` of each functional on each density: [`xc_energies`] takes the functional as an
//! [`XcSource`] (`models::xc_scheme`): the production path passes `Variational`, the scheme the
//! reference atom is solved with; the legacy `AtomSolver` is the parity seam.
use crate::{
    math::radial::integrate,
    models::{
        spherical,
        xc_scheme::{XcField, XcSource},
    },
    Context, Result,
};
use std::f64::consts::PI;

/// `E_H[rho] = 1/2 int v_H rho d^3r` by the trapezoid rule on `r`.
pub fn hartree_energy(ctx: &Context, r: &[f64], rho: &[f64]) -> Result<f64> {
    let potential = spherical::hartree(ctx, r, rho)?;
    let integrand = (0..r.len())
        .map(|i| potential.as_slice()[i] * rho[i] * 4. * PI * r[i] * r[i])
        .collect::<Vec<_>>();
    Ok(0.5 * integrate(r, &integrand)?)
}

/// `(E_xc[rho], int rho v_xc d^3r)` and the fields they come from.
pub struct XcEnergies {
    pub energy: f64,
    pub rho_potential: f64,
    pub field: XcField,
}

/// The exchange-correlation energy and `int rho v_xc` of `rho`, with the relativistic exchange
/// factor when the reference atom was solved with it: the one-center constant subtracts these
/// double-counting terms from that atom's band energy, so they must be its own functional.
pub fn xc_energies(
    ctx: &Context,
    source: &dyn XcSource,
    r: &[f64],
    rho: &[f64],
    functional: &str,
    relativistic: bool,
) -> Result<XcEnergies> {
    let field = source.evaluate(ctx, r, rho, functional, relativistic)?;
    let shell = |i: usize| 4. * PI * r[i] * r[i];
    let energy = (0..r.len())
        .map(|i| field.per_particle[i] * rho[i] * shell(i))
        .collect::<Vec<_>>();
    let virial = (0..r.len())
        .map(|i| field.potential[i] * rho[i] * shell(i))
        .collect::<Vec<_>>();
    Ok(XcEnergies {
        energy: integrate(r, &energy)?,
        rho_potential: integrate(r, &virial)?,
        field,
    })
}

/// The densities of a reference atom the energies are taken on (each on the generation mesh).
pub struct EnergyDensities<'a> {
    /// Band energy of the occupied references, `sum occupation * energy`.
    pub band: f64,
    pub ae_valence: &'a [f64],
    pub smooth_valence: &'a [f64],
    /// The smooth valence density plus its compensation charge.
    pub augmented: &'a [f64],
    pub core: &'a [f64],
    pub smooth_core: &'a [f64],
}

/// Energies of the frozen one-center constant (Hartree).
#[derive(Debug, Clone, Copy)]
pub struct OneCenterEnergies {
    pub band: f64,
    pub reference_valence: f64,
    pub pseudo_atom: f64,
    /// `reference_valence - pseudo_atom`: the constant the dataset stores.
    pub one_center: f64,
    pub hartree_ae: f64,
    pub hartree_ps: f64,
    pub xc_ae: f64,
    pub xc_ps: f64,
    /// The core-valence exchange-correlation difference the constant leaves out.
    pub core_valence_xc_omitted: f64,
}

/// The reference atom evaluated with the molecular machinery must give the all-electron valence
/// energy in the norm-conserving convention; this is the constant that makes it so.
pub fn one_center_energies(
    ctx: &Context,
    source: &dyn XcSource,
    r: &[f64],
    densities: &EnergyDensities<'_>,
    functional: &str,
    relativistic: bool,
) -> Result<OneCenterEnergies> {
    let hartree_ae = hartree_energy(ctx, r, densities.ae_valence)?;
    let hartree_ps = hartree_energy(ctx, r, densities.augmented)?;
    let xc = |rho: &[f64]| xc_energies(ctx, source, r, rho, functional, relativistic);
    let ae = xc(densities.ae_valence)?;
    let ps = xc(densities.smooth_valence)?;
    let reference_valence = densities.band - hartree_ae - ae.rho_potential + ae.energy;
    let pseudo_atom = densities.band - hartree_ps - ps.rho_potential + ps.energy;
    let total = |valence: &[f64], core: &[f64]| {
        let sum = valence
            .iter()
            .zip(core)
            .map(|(a, b)| a + b)
            .collect::<Vec<_>>();
        xc(&sum)
    };
    let full_ae = total(densities.ae_valence, densities.core)?;
    let full_ps = total(densities.smooth_valence, densities.smooth_core)?;
    Ok(OneCenterEnergies {
        band: densities.band,
        reference_valence,
        pseudo_atom,
        one_center: reference_valence - pseudo_atom,
        hartree_ae,
        hartree_ps,
        xc_ae: ae.energy,
        xc_ps: ps.energy,
        core_valence_xc_omitted: (full_ae.energy - ae.energy) - (full_ps.energy - ps.energy),
    })
}
