//! The screened exact exchange of a hybrid (HSE06) in the gradient.
//!
//! A screened hybrid adds `-fraction` times the exchange form of the occupied
//! pairs under `erfc(omega r) / r`: the grid pairs (the caller's screened
//! Poisson solver), the compensation charges with their shape potentials
//! under `erfc`, the screened one-center tensor `Delta W^SR`, the frozen
//! short-range semilocal exchange of the spheres (a one-body matrix) and, for
//! two spin channels, its polarization (a nonlinear function of the spheres'
//! spin densities). The exchange form is the one of Hartree-Fock with the
//! screened kernels, so the Hartree-Fock pieces (`compensation`,
//! `projections::exchange_derivatives`) are reused with screened channels.
use super::{
    compensation::{self, Channel, Moments},
    projections, Contribution, Response,
};
use crate::{molecular::MolecularIntegrals, paw::screened_atom::ScreenedAtom, Buffer, Result};
use nalgebra::{DMatrix, DVector};
use std::sync::Arc;

pub struct HybridResponse<'a> {
    parent: &'a Response<'a>,
    fraction: f64,
    spin_resolved: bool,
    atoms: Arc<Vec<ScreenedAtom>>,
    channels: Vec<Channel<'a>>,
    moments: Moments,
}
impl<'a> Response<'a> {
    /// The terms of the screened exchange at `omega` with exchange `fraction`;
    /// `spin_resolved` is a state of two spin channels (it has the polarization).
    pub fn hybrid(
        &'a self,
        omega: f64,
        fraction: f64,
        spin_resolved: bool,
    ) -> Result<HybridResponse<'a>> {
        let integrals: &MolecularIntegrals = self.integrals;
        let atoms = self.setup.screened_atoms(&integrals.context, omega)?;
        let channels = compensation::screened_channels(self.setup, integrals, omega)?;
        let moments = compensation::moments(
            self.setup,
            integrals,
            &channels,
            self.occupied,
            &self.projected,
            self.grid.psi,
            self.grid.rho,
            true,
        )?;
        Ok(HybridResponse {
            parent: self,
            fraction,
            spin_resolved,
            atoms,
            channels,
            moments,
        })
    }
}
impl HybridResponse<'_> {
    /// `fraction` times the compensation part of the screened exchange action
    /// on every occupied orbital (`nocc * n` values).
    pub fn exchange_action(&self) -> Result<Buffer<f64>> {
        let (integrals, occupied) = (self.parent.integrals, self.parent.occupied);
        let n = integrals.grid.index.points();
        let mut out = integrals.context.allocate::<f64>(occupied.len() * n)?;
        let mut scratch = integrals.context.allocate::<f64>(n)?;
        compensation::grid_actions(
            integrals,
            &self.channels,
            &self.moments,
            occupied,
            self.parent.grid.psi,
            Some(out.as_mut_slice()),
            scratch.as_mut_slice(),
        )?;
        for v in out.as_mut_slice() {
            *v *= self.fraction;
        }
        Ok(out)
    }
    /// Every part of the screened exchange that the caller does not contract on
    /// the grid orbitals. All are `fraction` times the exchange form.
    pub fn contributions(&self) -> Result<Vec<Contribution>> {
        let parent = self.parent;
        let (setup, integrals, occupied) = (parent.setup, parent.integrals, parent.occupied);
        let (nocc, n, fraction) = (
            occupied.len(),
            integrals.grid.index.points(),
            self.fraction,
        );
        let samples = projections::derivative_samples(setup, integrals)?;
        let scaled = |mut values: Vec<f64>| {
            for v in &mut values {
                *v *= fraction;
            }
            values
        };
        let mut out = Vec::new();
        // The exchange form of the pair integrals: projections, the Coulomb
        // coupling of the compensation charges and their potential on the grid.
        let atoms = &self.atoms;
        let pi = projections::exchange_derivatives(
            setup,
            &self.channels,
            &self.moments,
            occupied,
            &parent.projected,
            &|s| (atoms[s].factors.clone(), atoms[s].weights.clone()),
        );
        out.push(Contribution {
            name: "hybrid_pairs/projection",
            gradient: scaled(projections::force(integrals, occupied, &samples, &pi)),
        });
        let weights = compensation::pair_weights(&self.moments, occupied, true);
        out.push(Contribution {
            name: "hybrid_pairs/electrostatic",
            gradient: scaled(compensation::electrostatic_force(
                setup,
                integrals,
                &self.channels,
                &weights.exchange,
            )?),
        });
        let (xq, psi) = (&self.moments.xq, parent.grid.psi);
        out.push(Contribution {
            name: "hybrid_pairs/compensation_center",
            gradient: scaled(compensation::field_force(
                integrals,
                &self.channels,
                |a, k| {
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
                },
            )?),
        });
        // The frozen short-range semilocal exchange of the spheres, linear in
        // the spheres' density: `-fraction f P^T V P`.
        let one_body: Vec<Vec<DVector<f64>>> = setup
            .atoms
            .iter()
            .enumerate()
            .map(|(a, &s)| {
                let v = (&self.atoms[s].v + self.atoms[s].v.transpose()) * 0.5;
                occupied
                    .iter()
                    .enumerate()
                    .map(|(o, orbital)| &v * &parent.projected[a][o] * (-2. * orbital.weight))
                    .collect()
            })
            .collect();
        out.push(Contribution {
            name: "hybrid_one_body/projection",
            gradient: scaled(projections::force(integrals, occupied, &samples, &one_body)),
        });
        // Its polarization, a nonlinear function of the two spin densities of
        // each sphere, only for a spin-resolved state.
        if self.spin_resolved {
            out.push(Contribution {
                name: "hybrid_polarization/projection",
                gradient: scaled(self.polarization(&samples)?),
            });
        }
        Ok(out)
    }
    /// The force from `-fraction` times the spheres' polarization energy: its
    /// derivative with respect to a sphere's spin density, `up` and `down` in
    /// the projector layout, times the density's dependence on the projections.
    fn polarization(&self, samples: &[[DMatrix<f64>; 3]]) -> Result<Vec<f64>> {
        let parent = self.parent;
        let (setup, integrals, occupied) = (parent.setup, parent.integrals, parent.occupied);
        let mut pi: Vec<Vec<DVector<f64>>> = Vec::new();
        for (a, &s) in setup.atoms.iter().enumerate() {
            let p = setup.species_terms[s].projectors.len();
            let mut density = [DMatrix::<f64>::zeros(p, p), DMatrix::<f64>::zeros(p, p)];
            for (o, orbital) in occupied.iter().enumerate() {
                let x = &parent.projected[a][o];
                density[orbital.spin] += x * x.transpose() * orbital.weight;
            }
            let terms = self.atoms[s].polarization.evaluate(
                &integrals.context,
                &density[0],
                &density[1],
            )?;
            let fock = [terms.up, terms.down];
            pi.push(
                occupied
                    .iter()
                    .enumerate()
                    .map(|(o, orbital)| {
                        let f = (&fock[orbital.spin] + fock[orbital.spin].transpose()) * 0.5;
                        &f * &parent.projected[a][o] * (-2. * orbital.weight)
                    })
                    .collect(),
            );
        }
        Ok(projections::force(integrals, occupied, samples, &pi))
    }
}
