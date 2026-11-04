//! Analytic nuclear gradient of a converged restricted or unrestricted
//! Hartree-Fock or semilocal Kohn-Sham PAW-LCAO state.
//!
//! At a converged state the derivative of the energy is its partial derivative
//! at fixed atomic-orbital density minus the Pulay term (`-tr[W dS]`, `W` the
//! energy-weighted density of the occupied orbitals). It splits into named
//! terms, each the derivative of one energy term of the model:
//!
//! * the parts that act on the grid orbitals (kinetic, ionic long-range
//!   potential, Hartree, exchange-correlation, compensation potential, exact
//!   exchange, overlap) are contracted with the derivative of the basis
//!   functions ([`grid`]), one function at a time;
//! * the parts that live on the atoms' spheres (projections, short-range local
//!   potential, compensation charges, ionic potentials) and the potentials
//!   centered on an atom (long-range ionic, partial core) are differentiated
//!   in `paw::molecular::gradient`, which owns the setup's quadrature rules;
//! * the ion-ion repulsion is added here.
use super::{
    dft, forces::Gradient, integrals::laplacian_of, invalid, scf, MeanField, MolecularIntegrals,
};
use crate::math::fft::Poisson;
use crate::models::xc;
use crate::paw::molecular::gradient::{GridState, Occupied};
use crate::{product, quantum::C64, Buffer, Result};
use nalgebra::DMatrix;

mod frame;
mod grid;

impl MolecularIntegrals {
    /// The energy-weighted density `W = sum_i f_i eps_i a_i a_i^T` of the
    /// occupied orbitals of a converged state in the atomic-orbital basis: the
    /// weight of the overlap derivative, `-tr[W dS]`, in the gradient.
    pub fn energy_weighted_density(&self, state: &MeanField) -> Result<DMatrix<f64>> {
        let fock = match state.functional.as_deref() {
            Some(name) => dft::response(self, &state.densities, name)?.0,
            None => {
                self.hartree_fock_operators(&state.method, &state.densities)?
                    .1
            }
        };
        let frame = frame::occupied(self, state, &fock)?;
        let m = self.m();
        let mut w = DMatrix::<f64>::zeros(m, m);
        for o in &frame.orbitals {
            let a = nalgebra::DVector::from_column_slice(&o.ao);
            w += &a * a.transpose() * (o.weight * o.energy);
        }
        Ok(w)
    }
    pub(super) fn paw_gradient(&self, state: &MeanField) -> Result<Gradient> {
        let m = self.m();
        let setup = self.paw_setup.as_ref().ok_or_else(|| {
            invalid("forces need a PAW setup that rebuilds its geometry; custom molecular corrections have none")
        })?;
        if !["rhf", "uhf"].contains(&state.method.as_str()) {
            return Err(invalid(
                "PAW forces support restricted and unrestricted collinear states only",
            ));
        }
        if let Some(name) = state.functional.as_deref() {
            let definition = crate::models::functional(name)?;
            if definition.nonlocal_b.is_some() {
                return Err(crate::Error::Unsupported(format!(
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
        self.context.run(|| self.analytic_gradient(state, setup))
    }
    fn analytic_gradient(
        &self,
        state: &MeanField,
        setup: &crate::paw::molecular::Setup,
    ) -> Result<Gradient> {
        let ctx = &self.context;
        let n = self.grid.index.points();
        let m = self.m();
        let atoms = self.geometry.numbers().len();
        let hf = state.functional.is_none();
        let _workspace = ctx.reserve(product(&[3 * atoms, 64])?)?;
        // The Fock matrices and grid potentials of the converged densities.
        let mut poisson = Poisson::isolated(ctx, &self.grid)?;
        let (fock, valence, hartree, xc, reference) = if let Some(name) =
            state.functional.as_deref()
        {
            let (fock, fields, energy) = dft::response_with_energy(self, &state.densities, name)?;
            (
                fock,
                fields.valence,
                fields.hartree,
                Some(fields.terms),
                energy,
            )
        } else {
            let (core, fock) = self.hartree_fock_operators(&state.method, &state.densities)?;
            let energy = scf::energy(&state.densities, &fock, &core, &state.method);
            let valence = dft::grid_densities(self, &state.densities)?;
            let mut source = ctx.allocate::<C64>(n)?;
            let mut solved = ctx.allocate::<C64>(n)?;
            for k in 0..n {
                source.as_mut_slice()[k] =
                    C64::new(valence.as_slice()[k] + valence.as_slice()[n + k], 0.);
            }
            poisson.solve_into(source.as_slice(), solved.as_mut_slice())?;
            let mut hartree = ctx.allocate::<f64>(n)?;
            for (to, from) in hartree.as_mut_slice().iter_mut().zip(solved.as_slice()) {
                *to = from.re;
            }
            (fock, valence, hartree, None, energy)
        };
        if (reference - state.electronic_energy).abs() > 1e-7 {
            return Err(invalid(
                "the SCF state does not reproduce its energy on these integrals; it was not computed with them",
            ));
        }
        let frame = frame::occupied(self, state, &fock)?;
        let nocc = frame.orbitals.len();
        let occupied: &[Occupied] = &frame.orbitals;
        let _orbital_memory = ctx.reserve(product(&[nocc.max(1), n, 8, 2])?)?;
        // The occupied orbitals on the grid, from the stored samples.
        let mut psi = ctx.allocate::<f64>(product(&[nocc.max(1), n])?)?;
        {
            let mut row = ctx.allocate::<f64>(n)?;
            for a in 0..m {
                self.samples.widen_range(a * n, row.as_mut_slice())?;
                for (i, o) in occupied.iter().enumerate() {
                    let c = o.ao[a];
                    if c != 0. {
                        for (to, from) in psi.as_mut_slice()[i * n..(i + 1) * n]
                            .iter_mut()
                            .zip(row.as_slice())
                        {
                            *to += c * from;
                        }
                    }
                }
            }
        }
        let mut rho = ctx.allocate::<f64>(n)?;
        for k in 0..n {
            rho.as_mut_slice()[k] = valence.as_slice()[k] + valence.as_slice()[n + k];
        }
        drop(valence);
        let response = setup.response(
            self,
            occupied,
            GridState {
                psi: psi.as_slice(),
                rho: rho.as_slice(),
            },
            hf,
        )?;
        let corrections = self
            .corrections
            .as_ref()
            .ok_or_else(|| invalid("PAW forces need the PAW corrections"))?;
        let points = grid::points(self)?;
        let mut terms: Vec<(&'static str, Vec<f64>)> = Vec::new();
        let mut action = ctx.allocate::<f64>(product(&[nocc.max(1), n])?)?;
        let contract = |name: &'static str, action: &Buffer<f64>| {
            (
                name,
                grid::contract(self, occupied, points.as_slice(), action.as_slice()),
            )
        };
        for i in 0..nocc {
            let lap = laplacian_of(
                ctx,
                &self.grid,
                &self.kinetic_method,
                &psi.as_slice()[i * n..(i + 1) * n],
            )?;
            for k in 0..n {
                action.as_mut_slice()[i * n + k] = -0.5 * lap.as_slice()[k];
            }
        }
        terms.push(contract("kinetic/basis", &action));
        let coulomb = response.coulomb_potential()?;
        let [external_term, hartree_term, xc_term, coulomb_term, pulay_term] = grid::contract_local(
            self,
            occupied,
            points.as_slice(),
            psi.as_slice(),
            corrections.external.as_slice(),
            hartree.as_slice(),
            xc.as_ref().map(|fields| fields.potentials.as_slice()),
            coulomb.as_slice(),
        );
        drop(hartree);
        drop(coulomb);
        terms.push(("external/basis", external_term));
        terms.push(("hartree/basis", hartree_term));
        if let Some(fields) = xc.as_ref() {
            terms.push(("xc/basis", xc_term));
            if let Some(tau) = fields.tau_potentials.as_ref() {
                kinetic_density_action(
                    self,
                    occupied,
                    psi.as_slice(),
                    tau.as_slice(),
                    &mut action,
                )?;
                terms.push(contract("xc/tau_basis", &action));
            }
            if let Some(flux) = fields.gradient_potentials.as_ref() {
                density_gradient_action(
                    self,
                    occupied,
                    psi.as_slice(),
                    flux.as_slice(),
                    &mut action,
                )?;
                terms.push(contract("xc/gradient_basis", &action));
            }
        }
        terms.push(("pairs_coulomb/compensation_basis", coulomb_term));
        terms.push(("pulay_grid/basis", pulay_term));
        if hf {
            contract_exchange(self, occupied, psi.as_slice(), &mut poisson, &mut action)?;
            terms.push(contract("exchange/basis", &action));
            let compensation = response.exchange_action()?;
            terms.push(contract("pairs_exchange/compensation_basis", &compensation));
        }
        // A screened hybrid adds `fraction` times the exchange form under
        // `erfc(omega r) / r`: its grid pairs here, the rest in the PAW setup.
        let hybrid = match state.functional.as_deref() {
            Some(name) => crate::models::functional(name)?.screened_exchange,
            None => None,
        };
        let screened = hybrid
            .map(|(omega, fraction)| response.hybrid(omega, fraction, state.method == "uhf"))
            .transpose()?;
        if let (Some((omega, fraction)), Some(screened)) = (hybrid, screened.as_ref()) {
            let mut solver = Poisson::screened(ctx, &self.grid, omega)?;
            contract_exchange(self, occupied, psi.as_slice(), &mut solver, &mut action)?;
            for v in action.as_mut_slice() {
                *v *= fraction;
            }
            terms.push(contract("hybrid_grid/basis", &action));
            let compensation = screened.exchange_action()?;
            terms.push(contract("hybrid_pairs/compensation_basis", &compensation));
        }
        drop(action);
        drop(points);
        let core = xc
            .as_ref()
            .map(|fields| core_fields(self, fields))
            .transpose()?;
        let core_view = core
            .as_ref()
            .map(|(p, t)| (p.as_slice(), t.as_ref().map(|t| t.as_slice())));
        for part in response.contributions(core_view)? {
            terms.push((part.name, part.gradient));
        }
        if let Some(screened) = screened.as_ref() {
            for part in screened.contributions()? {
                terms.push((part.name, part.gradient));
            }
        }
        terms.push(("nuclear/ion_ion", nuclear_gradient(self, corrections)));
        let mut total = ctx.allocate::<f64>(3 * atoms)?;
        let mut named = Vec::new();
        for (name, values) in terms {
            for (t, v) in total.as_mut_slice().iter_mut().zip(&values) {
                *t += v;
            }
            let mut buffer = ctx.allocate::<f64>(3 * atoms)?;
            buffer.as_mut_slice().copy_from_slice(&values);
            named.push((name, buffer));
        }
        if total.as_slice().iter().any(|v| !v.is_finite()) {
            return Err(invalid("nonfinite stationary gradient"));
        }
        Ok(Gradient {
            hellmann_feynman: None,
            pulay: None,
            total,
            scheme: "analytic",
            stationarity: Some(frame.stationarity),
            terms: named,
        })
    }
}

/// `-sum_j psi_j v_ij` for every occupied orbital: the exact-exchange action,
/// from one Poisson solve per orbital pair of the same spin.
fn contract_exchange(
    integrals: &MolecularIntegrals,
    occupied: &[Occupied],
    psi: &[f64],
    poisson: &mut Poisson,
    action: &mut Buffer<f64>,
) -> Result<()> {
    let ctx = &integrals.context;
    let n = integrals.grid.index.points();
    let mut source = ctx.allocate::<C64>(n)?;
    let mut solved = ctx.allocate::<C64>(n)?;
    action.as_mut_slice().fill(0.);
    for i in 0..occupied.len() {
        for j in 0..=i {
            if occupied[i].spin != occupied[j].spin {
                continue;
            }
            for k in 0..n {
                source.as_mut_slice()[k] = C64::new(psi[i * n + k] * psi[j * n + k], 0.);
            }
            poisson.solve_into(source.as_slice(), solved.as_mut_slice())?;
            for k in 0..n {
                let v = solved.as_slice()[k].re;
                action.as_mut_slice()[i * n + k] -= psi[j * n + k] * v;
                if i != j {
                    action.as_mut_slice()[j * n + k] -= psi[i * n + k] * v;
                }
            }
        }
    }
    Ok(())
}

/// Gradient of the ion-ion repulsion of the valence charges.
fn nuclear_gradient(
    integrals: &MolecularIntegrals,
    corrections: &super::integrals::MolecularCorrections,
) -> Vec<f64> {
    let positions = integrals.geometry.positions();
    let mut out = vec![0.; 3 * positions.len()];
    for a in 0..positions.len() {
        for b in 0..a {
            let delta: [f64; 3] = std::array::from_fn(|i| positions[a][i] - positions[b][i]);
            let r = delta.iter().map(|v| v * v).sum::<f64>().sqrt();
            let factor = corrections.charges[a] * corrections.charges[b] / (r * r * r);
            for axis in 0..3 {
                out[3 * a + axis] -= factor * delta[axis];
                out[3 * b + axis] += factor * delta[axis];
            }
        }
    }
    out
}

/// `-1/2 D (tau_pot D psi)` for every occupied orbital: the action of the
/// kinetic-energy-density potential of a meta-GGA, with `D` the (antisymmetric)
/// grid derivative.
fn kinetic_density_action(
    integrals: &MolecularIntegrals,
    occupied: &[Occupied],
    psi: &[f64],
    tau: &[f64],
    action: &mut Buffer<f64>,
) -> Result<()> {
    let ctx = &integrals.context;
    let n = integrals.grid.index.points();
    let mut slope = ctx.allocate::<f64>(n)?;
    let mut weighted = ctx.allocate::<f64>(n)?;
    let mut derivative = ctx.allocate::<f64>(n)?;
    action.as_mut_slice().fill(0.);
    for (i, o) in occupied.iter().enumerate() {
        for axis in 0..3 {
            xc::derivative(
                &integrals.grid,
                &psi[i * n..(i + 1) * n],
                axis,
                slope.as_mut_slice(),
            )?;
            for k in 0..n {
                weighted.as_mut_slice()[k] = slope.as_slice()[k] * tau[o.spin * n + k];
            }
            xc::derivative(
                &integrals.grid,
                weighted.as_slice(),
                axis,
                derivative.as_mut_slice(),
            )?;
            for k in 0..n {
                action.as_mut_slice()[i * n + k] -= 0.5 * derivative.as_slice()[k];
            }
        }
    }
    Ok(())
}
/// `w . D psi - D (w psi)` for every occupied orbital: the action of the flux
/// `dE / d(grad rho)` of the density gradient `2 psi grad psi` that a meta-GGA
/// takes from the orbitals' own product rule.
fn density_gradient_action(
    integrals: &MolecularIntegrals,
    occupied: &[Occupied],
    psi: &[f64],
    flux: &[f64],
    action: &mut Buffer<f64>,
) -> Result<()> {
    let ctx = &integrals.context;
    let n = integrals.grid.index.points();
    let mut slope = ctx.allocate::<f64>(n)?;
    let mut weighted = ctx.allocate::<f64>(n)?;
    let mut divergence = ctx.allocate::<f64>(n)?;
    action.as_mut_slice().fill(0.);
    for (i, o) in occupied.iter().enumerate() {
        for axis in 0..3 {
            let w = &flux[(o.spin * 3 + axis) * n..(o.spin * 3 + axis + 1) * n];
            xc::derivative(
                &integrals.grid,
                &psi[i * n..(i + 1) * n],
                axis,
                slope.as_mut_slice(),
            )?;
            for k in 0..n {
                weighted.as_mut_slice()[k] = w[k] * psi[i * n + k];
            }
            xc::derivative(
                &integrals.grid,
                weighted.as_slice(),
                axis,
                divergence.as_mut_slice(),
            )?;
            for k in 0..n {
                action.as_mut_slice()[i * n + k] +=
                    w[k] * slope.as_slice()[k] - divergence.as_slice()[k];
            }
        }
    }
    Ok(())
}
/// What the partial core density feels: its potential (the exchange-correlation
/// potentials of both spins, half the core each, and the divergence of the
/// density-gradient flux through its grid derivative) and, for a meta-GGA, the
/// kinetic-energy-density potential of both spins.
fn core_fields(
    integrals: &MolecularIntegrals,
    fields: &xc::Xc,
) -> Result<(Buffer<f64>, Option<Buffer<f64>>)> {
    let ctx = &integrals.context;
    let n = integrals.grid.index.points();
    let mut potential = ctx.allocate::<f64>(n)?;
    for k in 0..n {
        potential.as_mut_slice()[k] =
            0.5 * (fields.potentials.as_slice()[k] + fields.potentials.as_slice()[n + k]);
    }
    if let Some(flux) = fields.gradient_potentials.as_ref() {
        // sum_k flux . D (dcore) = - sum_k (D flux) dcore, the grid derivative being antisymmetric.
        let mut average = ctx.allocate::<f64>(n)?;
        let mut derivative = ctx.allocate::<f64>(n)?;
        for axis in 0..3 {
            for k in 0..n {
                average.as_mut_slice()[k] =
                    0.5 * (flux.as_slice()[axis * n + k] + flux.as_slice()[(3 + axis) * n + k]);
            }
            xc::derivative(
                &integrals.grid,
                average.as_slice(),
                axis,
                derivative.as_mut_slice(),
            )?;
            for k in 0..n {
                potential.as_mut_slice()[k] -= derivative.as_slice()[k];
            }
        }
    }
    let tau = match fields.tau_potentials.as_ref() {
        Some(tau) => {
            let mut out = ctx.allocate::<f64>(n)?;
            for k in 0..n {
                out.as_mut_slice()[k] = 0.5 * (tau.as_slice()[k] + tau.as_slice()[n + k]);
            }
            Some(out)
        }
        None => None,
    };
    Ok((potential, tau))
}
