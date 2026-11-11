//! Analytic geometry derivative of the isolated PAW-LCAO energy at a converged
//! state: the terms that belong to the PAW setup (projections, compensation
//! multipoles, short-range and long-range ionic potentials, partial core).
//!
//! The derivative is the partial derivative of the energy at fixed
//! atomic-orbital density, `tr[D dh] - tr[W dS]` and the like, in which `W` is
//! the energy-weighted density of the occupied orbitals. Every piece that
//! depends on the geometry through the projections `C` is `tr[Gamma dC]`; the
//! pieces that move on the grid (basis functions, potentials centered on an
//! atom) are contracted by the caller or here, term by term.
mod compensation;
mod densities;
mod hubbard;
mod projections;
mod radial;
mod screened;
mod spheres;
mod spinor_screened;

pub use densities::{DensityTerms, Forms};
pub use screened::HybridResponse;
pub use spinor_screened::{pseudo_index, SpinorExchange};
pub use radial::{core_slope, local_slope, long_range_slope, multipole_gradient, potential_slope};

use super::Setup;
use crate::{molecular::MolecularIntegrals, Buffer, Result};
use compensation::{Channel, Moments};
use nalgebra::DVector;

/// An occupied spin orbital of the converged state in the atomic-orbital
/// (non-orthogonal) basis: `ao` are its coefficients over the basis functions.
///
/// It enters the force in two roles: the density `D = sum weight a a^T` and the
/// energy-weighted density `W = sum w_weight a a^T` of the overlap derivative
/// `-tr[W dS]`. A state's own orbitals have `w_weight = weight * energy`; a
/// spinor state's pseudo-orbitals carry the two roles in separate entries
/// (`w_weight = 0` for a density orbital, `weight = 0` for a `W` orbital),
/// because its `W` is not diagonal in the eigenbasis of its density.
pub struct Occupied {
    pub spin: usize,
    pub weight: f64,
    pub energy: f64,
    pub w_weight: f64,
    pub ao: Vec<f64>,
}
/// The occupied orbitals and the valence density on the grid, as the energy
/// builds them.
pub struct GridState<'a> {
    /// `psi[i * n + k]`: orbital `i` at grid point `k`.
    pub psi: &'a [f64],
    /// The valence density of both spins at each grid point.
    pub rho: &'a [f64],
}
/// One named part of the nuclear gradient, `3 * atoms` values in Ha/Bohr.
/// The name is `term/part`; the term is the energy term the part belongs to.
pub struct Contribution {
    pub name: &'static str,
    pub gradient: Vec<f64>,
}

/// The state-dependent quantities of the PAW terms, built once per gradient.
pub struct Response<'a> {
    setup: &'a Setup,
    integrals: &'a MolecularIntegrals,
    occupied: &'a [Occupied],
    grid: GridState<'a>,
    hf: bool,
    unrestricted: bool,
    channels: Vec<Channel<'a>>,
    projected: Vec<Vec<DVector<f64>>>,
    moments: Moments,
}
impl Setup {
    /// `hf` selects the Hartree-Fock model (frozen-core exchange and one-center
    /// tensor in the model, exact exchange in the pairs); otherwise the
    /// Kohn-Sham model. `unrestricted` is whether the state carries two spin channels.
    pub fn response<'a>(
        &'a self,
        integrals: &'a MolecularIntegrals,
        occupied: &'a [Occupied],
        grid: GridState<'a>,
        hf: bool,
        unrestricted: bool,
    ) -> Result<Response<'a>> {
        if hf && integrals.corrections.as_ref().is_some_and(|c| c.hubbard.is_some()) {
            // The Hartree-Fock energy has no +U term: a +U force would not be its derivative.
            return Err(crate::Error::Unsupported(
                "DFT+U is a Kohn-Sham correction: the Hartree-Fock gradient of a molecule with hubbard_u is not defined"
                    .into(),
            ));
        }
        let n = integrals.grid.index.points();
        let m = integrals.m();
        if grid.rho.len() != n
            || grid.psi.len() != occupied.len() * n
            || occupied.iter().any(|o| o.ao.len() != m)
        {
            return Err(super::invalid(
                "the occupied orbitals and the density must match the molecular grid",
            ));
        }
        let channels = compensation::channels(self);
        let (_, projected) = projections::project(self, integrals, occupied)?;
        // The bare channels match the calculation's cached electrostatic matrices.
        let cached = integrals.corrections.as_ref().and_then(|c| {
            Some((c.electrostatic.as_ref()?, c.ionic.as_deref()?))
        });
        let moments = compensation::moments(
            self, integrals, &channels, cached, occupied, &projected, grid.psi, grid.rho, hf,
        )?;
        Ok(Response {
            setup: self,
            integrals,
            occupied,
            grid,
            hf,
            unrestricted,
            channels,
            projected,
            moments,
        })
    }
}
impl Response<'_> {
    fn atoms(&self) -> usize {
        self.integrals.geometry.numbers().len()
    }
    /// The potential `sum_A q[A] phi_A(r)` of the compensation charges that the
    /// valence density feels, on the grid (the Coulomb action's local part).
    pub fn coulomb_potential(&self) -> Result<Buffer<f64>> {
        let n = self.integrals.grid.index.points();
        let mut out = self.integrals.context.allocate::<f64>(n)?;
        compensation::grid_actions(
            self.integrals,
            &self.channels,
            &self.moments,
            self.occupied,
            self.grid.psi,
            None,
            out.as_mut_slice(),
        )?;
        Ok(out)
    }
    /// The compensation part of the exact-exchange action on every occupied
    /// orbital, `nocc * n` values (Hartree-Fock only).
    pub fn exchange_action(&self) -> Result<Buffer<f64>> {
        let n = self.integrals.grid.index.points();
        let mut out = self
            .integrals
            .context
            .allocate::<f64>(self.occupied.len() * n)?;
        let mut scratch = self.integrals.context.allocate::<f64>(n)?;
        compensation::grid_actions(
            self.integrals,
            &self.channels,
            &self.moments,
            self.occupied,
            self.grid.psi,
            Some(out.as_mut_slice()),
            scratch.as_mut_slice(),
        )?;
        Ok(out)
    }
    /// Every part of the gradient that the caller does not contract on the grid
    /// orbitals: the projections, the short-range sphere, the compensation
    /// charges and the ionic potentials. `core` are what the partial core
    /// density feels (see the caller), for its motion.
    pub fn contributions(
        &self,
        core: Option<(&[f64], Option<&[f64]>)>,
    ) -> Result<Vec<Contribution>> {
        let (setup, integrals) = (self.setup, self.integrals);
        let nocc = self.occupied.len();
        let n = integrals.grid.index.points();
        // The derivative samples of the projections (three per matrix element of
        // `C`), the projection derivatives of the four terms and the matrices
        // `Gamma` contracted with them.
        let pmax = setup
            .atoms
            .iter()
            .map(|&s| setup.species_terms[s].projectors.len())
            .max()
            .unwrap_or(0);
        let atoms = self.atoms();
        let _memory = integrals.context.reserve(
            crate::product(&[
                integrals
                    .m()
                    .checked_mul(pmax)
                    .and_then(|v| v.checked_mul(4))
                    .ok_or(crate::Error::Overflow)?,
                atoms.max(1),
                8,
            ])?
            .checked_add(crate::product(&[
                nocc.max(1),
                pmax.max(1),
                atoms.max(1),
                4,
                8,
            ])?)
            .ok_or(crate::Error::Overflow)?,
        )?;
        let mut out = Vec::new();
        let samples = projections::derivative_samples(setup, integrals)?;
        let pi = projections::derivatives(
            setup,
            &self.channels,
            &self.moments,
            self.occupied,
            &self.projected,
            self.hf,
        );
        let contract = |pi: &Vec<Vec<DVector<f64>>>| {
            projections::force(integrals, self.occupied, &samples, pi)
        };
        out.push(Contribution {
            name: "projector_potential/projection",
            gradient: contract(&pi.potential),
        });
        out.push(Contribution {
            name: "pulay_projector/projection",
            gradient: contract(&pi.overlap),
        });
        out.push(Contribution {
            name: "pairs_coulomb/projection",
            gradient: contract(&pi.coulomb),
        });
        if let Some(pi) =
            hubbard::derivatives(setup, integrals, self.occupied, &self.projected, self.unrestricted)?
        {
            out.push(Contribution {
                name: "hubbard/projection",
                gradient: contract(&pi),
            });
        }
        out.push(Contribution {
            name: "projector_potential/local_sphere",
            gradient: spheres::local_force(setup, integrals, self.occupied)?,
        });
        // The ionic potentials in the compensation spheres are weighted by the
        // moments of the projections.
        out.push(Contribution {
            name: "projector_potential/ionic",
            gradient: compensation::ionic_force(setup, integrals, &self.channels, &self.moments.q)?,
        });
        let weights = compensation::pair_weights(&self.moments, self.occupied, self.hf);
        out.push(Contribution {
            name: "pairs_coulomb/electrostatic",
            gradient: compensation::electrostatic_force(
                setup,
                integrals,
                &self.channels,
                &weights.coulomb,
            )?,
        });
        let q = &self.moments.q;
        let rho = self.grid.rho;
        out.push(Contribution {
            name: "pairs_coulomb/compensation_center",
            gradient: compensation::field_force(integrals, &self.channels, |a, k| q[a] * rho[k])?,
        });
        out.push(Contribution {
            name: "external/center",
            gradient: spheres::external_force(setup, integrals, rho),
        });
        if let Some((potential, tau)) = core {
            out.push(Contribution {
                name: "xc/core",
                gradient: spheres::core_force(setup, integrals, potential, tau),
            });
        }
        if self.hf {
            out.push(Contribution {
                name: "pairs_exchange/projection",
                gradient: contract(&pi.exchange),
            });
            out.push(Contribution {
                name: "pairs_exchange/electrostatic",
                gradient: compensation::electrostatic_force(
                    setup,
                    integrals,
                    &self.channels,
                    &weights.exchange,
                )?,
            });
            let (xq, psi) = (&self.moments.xq, self.grid.psi);
            let occupied = self.occupied;
            out.push(Contribution {
                name: "pairs_exchange/compensation_center",
                gradient: compensation::field_force(integrals, &self.channels, |a, k| {
                    let mut sum = 0.;
                    for i in 0..nocc {
                        for j in 0..nocc {
                            if occupied[i].spin == occupied[j].spin {
                                sum += occupied[i].weight
                                    * xq[(a * nocc + i) * nocc + j]
                                    * psi[i * n + k]
                                    * psi[j * n + k];
                            }
                        }
                    }
                    -sum
                })?,
            });
        }
        debug_assert!(out.iter().all(|c| c.gradient.len() == 3 * self.atoms()));
        Ok(out)
    }
}
