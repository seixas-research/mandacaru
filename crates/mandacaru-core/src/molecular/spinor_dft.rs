//! Spinor (generalized, noncollinear) Kohn-Sham for molecules: MG-F2a.
//!
//! A spinor state is one `2m x 2m` density matrix `D` in the orthonormal frame, alpha block first.
//! Its Pauli components `D_a = noncollinear::component(D, m, a)` give the charge `n` (`a = 0`) and the
//! magnetization `m_a` on the grid. The functional is the collinear one in the local spin frame
//! ([`crate::models::noncollinear`], shared with the crystal engine).
//!
//! Everything else the molecular PAW Kohn-Sham energy holds (the core Hamiltonian, Hartree, the
//! PAW pair, one-body and core corrections) depends on the total density alone. So
//! [`spinor_response`] evaluates the restricted state `D_0 / 2` with [`super::dft`] and replaces
//! its exchange-correlation part:
//! `E = E[D_0 / 2] - E_xc[n/2, n/2] + E_xc[n_up, n_dn]`, `F_0 = F[D_0 / 2] - V[v_xc] + V[v_0]`,
//! `F_a = V[v_a]`, with `V[v]_ab = dv sum_x v phi_a phi_b`. LDA, PBE and their rVV10 forms reach
//! the Fock through the multiplicative potential alone. A meta-GGA (r2SCAN) also has, per channel,
//! the kinetic density `tau_a = 1/2 sum Re D_a grad phi . grad phi` and the product-rule density
//! gradient `grad n_a = 2 sum Re D_a phi grad phi` (the molecular scalar meta-GGA's own choice),
//! rotated into the frame with their frame terms, and its channel matrices carry
//! `1/2 v_tau grad phi . grad phi + F . (phi grad phi + grad phi phi)` as the scalar Fock does.
//! A screened hybrid (HSE06, F2c) keeps the restricted evaluation's frozen sphere terms (linear in
//! the charge) and replaces its exact exchange by that of the four spin blocks,
//! `-(a/2) sum_st tr(D_ts K[D_st])` (spin-flip blocks included), and adds the spheres' vector
//! polarization (`SpinTerms::vector_polarization`). Its force contracts the occupied spinors
//! themselves (`paw::molecular::gradient::spinor_screened`), so it needs unit occupations. DFT+U
//! raises by name until its package.
use super::integrals::MolecularIntegrals;
use crate::{
    math::storage::{RealSamples, SampleBlocks},
    models::{noncollinear, xc},
    molecular::invalid,
    quantum::C64,
    Buffer, Result,
};
use nalgebra::DMatrix;

type Matrix = DMatrix<C64>;

/// The energy and `2m x 2m` Fock of a spinor density.
pub struct SpinorEvaluation {
    pub fock: Matrix,
    pub energy: f64,
    /// The exchange-correlation energy in the local frame.
    pub xc: f64,
    /// The Hartree energy (of the charge alone).
    pub hartree: f64,
}

/// What a spinor Kohn-Sham state cannot carry yet, refused by name.
pub fn check(integrals: &MolecularIntegrals, name: &str) -> Result<()> {
    crate::models::functional(name)?;
    let refuse = |what: &str| {
        Err(crate::Error::Unsupported(format!(
            "spinor molecular Kohn-Sham is not available for {what}"
        )))
    };
    if integrals.corrections.as_ref().is_some_and(|c| c.hubbard.is_some()) {
        return refuse("DFT+U");
    }
    Ok(())
}

/// `n_c(x) = sum_ab M_c[a, b] phi_a(x) phi_b(x)` for real symmetric `matrices` (unclamped: a
/// magnetization channel takes either sign), one block of the grid each.
fn channel_densities(integrals: &MolecularIntegrals, matrices: &[DMatrix<f64>]) -> Result<Buffer<f64>> {
    let (m, n) = (integrals.m(), integrals.grid.index.points());
    let mut out = integrals.context.allocate::<f64>(crate::product(&[matrices.len(), n])?)?;
    let mut blocks = SampleBlocks::new(&integrals.context, &integrals.orthonormal_samples, m, n)?;
    for start in (0..n).step_by(blocks.block()) {
        let count = blocks.load(start)?;
        let (phi, w) = (blocks.values(), blocks.stride());
        for (c, matrix) in matrices.iter().enumerate() {
            for i in 0..count {
                let mut value = 0.;
                for a in 0..m {
                    let row: f64 = (0..m).map(|b| matrix[(a, b)] * phi[b * w + i]).sum();
                    value += phi[a * w + i] * row;
                }
                out.as_mut_slice()[c * n + start + i] = value;
            }
        }
    }
    Ok(out)
}

/// `V[v]_ab = dv sum_x v(x) phi_a(x) phi_b(x)`, the matrix of a multiplicative potential.
fn local_matrix(integrals: &MolecularIntegrals, v: &[f64]) -> Result<DMatrix<f64>> {
    let (m, n) = (integrals.m(), integrals.grid.index.points());
    if v.len() != n {
        return Err(invalid("a local potential has one value per grid point"));
    }
    let dv = integrals.grid.index.dv();
    let mut out = DMatrix::<f64>::zeros(m, m);
    let mut blocks = SampleBlocks::new(&integrals.context, &integrals.orthonormal_samples, m, n)?;
    for start in (0..n).step_by(blocks.block()) {
        let count = blocks.load(start)?;
        let (phi, w) = (blocks.values(), blocks.stride());
        for b in 0..m {
            for a in 0..=b {
                let dot: f64 = (0..count)
                    .map(|i| phi[a * w + i] * phi[b * w + i] * v[start + i])
                    .sum();
                out[(a, b)] += dot * dv;
                if a != b {
                    out[(b, a)] += dot * dv;
                }
            }
        }
    }
    Ok(out)
}

/// [`local_matrix`] of a meta-GGA channel: `dv sum_x [v phi_a phi_b + 1/2 v_tau grad phi_a . grad
/// phi_b + F . (phi_a grad phi_b + grad phi_a phi_b)]`, the derivative of the energy by the
/// channel's density matrix through `n`, `tau` and the product-rule gradient (`flux` is
/// `[axis][point]`).
fn meta_matrix(
    integrals: &MolecularIntegrals,
    gradients: &crate::math::storage::RealSamples,
    v: &[f64],
    tau: &[f64],
    flux: &[f64],
) -> Result<DMatrix<f64>> {
    let (m, n) = (integrals.m(), integrals.grid.index.points());
    if v.len() != n || tau.len() != n || flux.len() != 3 * n {
        return Err(invalid("a meta-GGA channel has a potential, a kinetic potential and three fluxes per point"));
    }
    let ctx = &integrals.context;
    let dv = integrals.grid.index.dv();
    let mut out = DMatrix::<f64>::zeros(m, m);
    let mut orbitals = SampleBlocks::new(ctx, &integrals.orthonormal_samples, m, n)?;
    let mut slopes = SampleBlocks::new(ctx, gradients, 3 * m, n)?;
    for start in (0..n).step_by(orbitals.block()) {
        let count = orbitals.load(start)?;
        slopes.load(start)?;
        let (phi, g, w) = (orbitals.values(), slopes.values(), orbitals.stride());
        for b in 0..m {
            for a in 0..=b {
                let mut dot = 0.;
                for i in 0..count {
                    let k = start + i;
                    let mut value = v[k] * phi[a * w + i] * phi[b * w + i];
                    for axis in 0..3 {
                        let (ga, gb) = (g[(axis * m + a) * w + i], g[(axis * m + b) * w + i]);
                        value += 0.5 * tau[k] * ga * gb + flux[axis * n + k] * (phi[a * w + i] * gb + ga * phi[b * w + i]);
                    }
                    dot += value;
                }
                out[(a, b)] += dot * dv;
                if a != b {
                    out[(b, a)] += dot * dv;
                }
            }
        }
    }
    Ok(out)
}

/// The PAW spin-orbit term in the orthonormal frame, `(I_2 x L)^T h^SO (I_2 x L)` with `L` the
/// integrals' Loewdin transform, or `None` for a molecule without one.
pub fn spin_orbit_term(integrals: &MolecularIntegrals) -> Result<Option<Matrix>> {
    let Some(setup) = integrals.paw_setup.as_ref() else {
        return Ok(None);
    };
    let Some(h) = setup.spin_orbit(&integrals.context, integrals.geometry.positions())? else {
        return Ok(None);
    };
    let lowdin = &integrals.lowdin;
    let (ao, m) = (lowdin.nrows(), lowdin.ncols());
    if h.shape() != (2 * ao, 2 * ao) {
        return Err(invalid("the spin-orbit term does not match the basis of the integrals"));
    }
    let l = lowdin.map(|v| C64::new(v, 0.));
    let mut out = Matrix::zeros(2 * m, 2 * m);
    for s in 0..2 {
        for t in 0..2 {
            let block = h.view((s * ao, t * ao), (ao, ao));
            out.view_mut((s * m, t * m), (m, m)).copy_from(&(l.transpose() * block * &l));
        }
    }
    Ok(Some(out))
}

/// The energy and Fock of the spinor density `d` (`2m x 2m`, orthonormal frame) for `name`, with
/// the spin-orbit term `spin_orbit` ([`spin_orbit_term`]) as a fixed one-body operator when given.
pub fn spinor_response(
    integrals: &MolecularIntegrals,
    d: &Matrix,
    name: &str,
    spin_orbit: Option<&Matrix>,
) -> Result<SpinorEvaluation> {
    let shared = Shared::new(integrals, name)?;
    Ok(spinor_parts(integrals, d, name, spin_orbit, &shared)?.0)
}

/// What every evaluation of one functional on one geometry shares, built once per SCF rather
/// than per iteration: a meta-GGA's orbital gradients and a screened hybrid's pair integrals.
pub(super) struct Shared {
    gradients: Option<RealSamples>,
    screening: Option<super::dft::Screening>,
}
impl Shared {
    pub(super) fn new(integrals: &MolecularIntegrals, name: &str) -> Result<Self> {
        check(integrals, name)?;
        let gradients = crate::models::functional(name)?
            .requires_tau
            .then(|| super::dft::orbital_gradients(integrals))
            .transpose()?;
        Ok(Self { gradients, screening: integrals.screening(name)? })
    }
}

/// The Pauli-channel exchange-correlation potentials of a spinor density, frame terms included:
/// `v` (`4 n`) and, for a meta-GGA, `(v_tau, flux)` (`4 n`, `4 x 3 n`).
pub(super) struct ChannelPotentials {
    pub v: Buffer<f64>,
    pub meta: Option<(Buffer<f64>, Buffer<f64>)>,
}

/// [`spinor_response`] and what its force needs: the restricted evaluation's grid fields (the
/// charge, its Hartree potential) and the Pauli-channel exchange-correlation potentials.
pub(super) fn spinor_parts(
    integrals: &MolecularIntegrals,
    d: &Matrix,
    name: &str,
    spin_orbit: Option<&Matrix>,
    shared: &Shared,
) -> Result<(SpinorEvaluation, super::dft::Fields, ChannelPotentials)> {
    let (m, n) = (integrals.m(), integrals.grid.index.points());
    if d.shape() != (2 * m, 2 * m) {
        return Err(invalid("a spinor density is 2m x 2m"));
    }
    let ctx = &integrals.context;
    // The restricted state of the same charge: every term but the exchange-correlation.
    let charge = noncollinear::component(d, m, 0);
    let half = &charge * C64::new(0.5, 0.);
    let (mut fock, fields, mut energy, hartree) = super::dft::response_with_shared(
        integrals,
        &[half.clone()],
        name,
        shared.gradients.as_ref(),
        shared.screening.as_ref(),
    )?;
    // A screened hybrid: the restricted exact exchange of `D_0 / 2` (both spins) is replaced by the
    // exchange of the four spin blocks, `-(a/2) sum_st tr(D_ts K[D_st])`; the frozen sphere terms
    // are linear in the charge and stay.
    let mut exchange = None;
    if let Some(screening) = shared.screening.as_ref() {
        let a = screening.fraction;
        let (_, k) = super::scf::jk(&screening.eri, m, &half);
        energy += a * (&half * &k).trace().re;
        fock[0] += k * C64::new(a, 0.);
        let mut operator = Matrix::zeros(2 * m, 2 * m);
        for s in 0..2 {
            for t in 0..2 {
                let block = d.view((s * m, t * m), (m, m)).into_owned();
                let (_, k) = super::scf::jk(&screening.eri, m, &block);
                energy -= 0.5 * a * (d.view((t * m, s * m), (m, m)) * &k).trace().re;
                let mut view = operator.view_mut((s * m, t * m), (m, m));
                view -= k * C64::new(a, 0.);
            }
        }
        exchange = Some(operator);
    }
    let restricted_xc = fields.terms.energy;
    let unpolarized = &fields.terms.potentials.as_slice()[..n];
    // The Pauli channels on the grid, the partial core in the charge (half per spin, as the
    // collinear evaluation adds it).
    let channels: Vec<DMatrix<f64>> =
        (0..4).map(|a| noncollinear::component(d, m, a).map(|v| v.re)).collect();
    let mut grid = channel_densities(integrals, &channels)?;
    let paw = integrals.corrections.as_ref().and_then(|c| c.dft.as_ref());
    if let Some(paw) = paw {
        for k in 0..n {
            grid.as_mut_slice()[k] += paw.core.as_slice()[k];
        }
    }
    // A meta-GGA's kinetic densities and product-rule gradients of the four channels, the
    // partial core in the charge channel (its tau, and its gradient by the grid, as the scalar).
    let meta = shared.gradients.as_ref().map(|gradients| -> Result<_> {
        let complex: Vec<Matrix> = channels.iter().map(|c| c.map(|v| C64::new(v, 0.))).collect();
        let mut tau = super::dft::kinetic_channels(integrals, &complex, gradients)?;
        let mut slopes = ctx.allocate::<f64>(12 * n)?;
        for pair in 0..2 {
            let half = super::dft::grid_density_gradients(integrals, &complex[2 * pair..2 * pair + 2], gradients)?;
            slopes.as_mut_slice()[6 * pair * n..6 * (pair + 1) * n].copy_from_slice(half.as_slice());
        }
        if let Some(paw) = paw {
            for k in 0..n {
                tau.as_mut_slice()[k] += paw.core_tau.as_slice()[k];
            }
            let mut core = ctx.allocate::<f64>(n)?;
            for axis in 0..3 {
                xc::derivative(&integrals.grid, paw.core.as_slice(), axis, core.as_mut_slice())?;
                for k in 0..n {
                    slopes.as_mut_slice()[axis * n + k] += core.as_slice()[k];
                }
            }
        }
        Ok((gradients, tau, slopes))
    }).transpose()?;
    let rotated = noncollinear::rotate(ctx, n, grid.as_slice(), meta.as_ref().map(|(_, t, _)| t.as_slice()))?;
    let relativistic = paw.is_some_and(|p| p.relativistic_exchange)
        && matches!(crate::models::functional(name)?.semilocal, "lda" | "pbe");
    // The frame densities (and kinetic densities) are nonnegative up to roundoff, as the
    // collinear grid fields are; the Pauli channels are not clamped.
    let clamp = |values: &[f64]| -> Result<Buffer<f64>> {
        let mut out = ctx.copy_slice(values)?;
        for value in out.as_mut_slice() {
            *value = value.max(0.);
        }
        Ok(out)
    };
    let frame = clamp(rotated.densities.as_slice())?;
    let frame_tau = rotated.tau.as_ref().map(|t| clamp(t.as_slice())).transpose()?;
    let frame_gradients = meta
        .as_ref()
        .map(|(_, _, slopes)| noncollinear::rotate_gradients(ctx, n, grid.as_slice(), &rotated, slopes.as_slice()))
        .transpose()?;
    let terms = xc::evaluate_fields_relativistic(
        ctx,
        &integrals.grid,
        frame.as_slice(),
        frame_tau.as_ref().map(|t| t.as_slice()),
        frame_gradients.as_ref().map(|g| g.as_slice()),
        name,
        relativistic,
    )?;
    let (mut potentials, kinetic) = noncollinear::unrotate(
        ctx,
        n,
        grid.as_slice(),
        &rotated,
        terms.potentials.as_slice(),
        terms.tau_potentials.as_ref().map(|t| t.as_slice()),
        meta.as_ref().map(|(_, t, _)| t.as_slice()),
    )?;
    let mut channel_fock: Vec<Matrix> = Vec::with_capacity(4);
    let mut meta_potentials = None;
    if let Some((gradients, _, slopes)) = &meta {
        let (frame_flux, kinetic) = match (terms.gradient_potentials.as_ref(), kinetic.as_ref()) {
            (Some(f), Some(k)) => (f, k),
            _ => return Err(invalid("a meta-GGA evaluation returned no kinetic or gradient potentials")),
        };
        let flux = noncollinear::unrotate_gradients(ctx, n, grid.as_slice(), &rotated, slopes.as_slice(), frame_flux.as_slice(), potentials.as_mut_slice())?;
        let restricted = (fields.terms.tau_potentials.as_ref(), fields.terms.gradient_potentials.as_ref());
        let (Some(restricted_tau), Some(restricted_flux)) = restricted else {
            return Err(invalid("the restricted meta-GGA evaluation returned no kinetic or gradient potentials"));
        };
        let difference = |a: &[f64], b: &[f64]| -> Vec<f64> { a.iter().zip(b).map(|(x, y)| x - y).collect() };
        let (p, k, f) = (potentials.as_slice(), kinetic.as_slice(), flux.as_slice());
        let charge = meta_matrix(
            integrals,
            gradients,
            &difference(&p[..n], unpolarized),
            &difference(&k[..n], &restricted_tau.as_slice()[..n]),
            &difference(&f[..3 * n], &restricted_flux.as_slice()[..3 * n]),
        )?;
        channel_fock.push(&fock[0] + charge.map(|v| C64::new(v, 0.)));
        for a in 1..4 {
            let matrix = meta_matrix(integrals, gradients, &p[a * n..(a + 1) * n], &k[a * n..(a + 1) * n], &f[3 * a * n..3 * (a + 1) * n])?;
            channel_fock.push(matrix.map(|v| C64::new(v, 0.)));
        }
        meta_potentials = Some((ctx.copy_slice(k)?, flux));
    } else {
        let mut charge_potential = ctx.allocate::<f64>(n)?;
        for k in 0..n {
            charge_potential.as_mut_slice()[k] = potentials.as_slice()[k] - unpolarized[k];
        }
        channel_fock.push(&fock[0] + local_matrix(integrals, charge_potential.as_slice())?.map(|v| C64::new(v, 0.)));
        for a in 1..4 {
            channel_fock.push(
                local_matrix(integrals, &potentials.as_slice()[a * n..(a + 1) * n])?.map(|v| C64::new(v, 0.)),
            );
        }
    }
    // The spheres' non-collinear polarization of the short-range semilocal exchange the hybrid
    // replaces (the restricted evaluation has none).
    if let Some(screening) = shared.screening.as_ref() {
        if let Some(spin) = screening.spin.as_ref() {
            let pauli: [DMatrix<f64>; 4] = std::array::from_fn(|a| channels[a].clone());
            let (polarization, operators) = spin.vector_polarization(ctx, &pauli)?;
            energy -= screening.fraction * polarization;
            for (f, o) in channel_fock.iter_mut().zip(&operators) {
                *f -= o.map(|v| C64::new(screening.fraction * v, 0.));
            }
        }
    }
    let mut out = Matrix::zeros(2 * m, 2 * m);
    noncollinear::assemble_into(&mut out, &channel_fock, m);
    if let Some(exchange) = exchange {
        out += exchange;
    }
    let mut fixed = 0.;
    if let Some(h) = spin_orbit {
        if h.shape() != (2 * m, 2 * m) {
            return Err(invalid("the spin-orbit term is 2m x 2m"));
        }
        out += h;
        fixed = (d * h).trace().re;
    }
    Ok((
        SpinorEvaluation {
            fock: out,
            energy: energy - restricted_xc + terms.energy + fixed,
            xc: terms.energy,
            hartree,
        },
        fields,
        ChannelPotentials { v: potentials, meta: meta_potentials },
    ))
}

impl MolecularIntegrals {
    /// Spinor Kohn-Sham with `electrons` in one `2m x 2m` set of orbitals (the `ghf` state shape
    /// with a functional). Two starts, the core guess and one with the frontier spinors rotated in
    /// spin space, run through the SCF iteration [`MolecularIntegrals::iterate_start`] shared with
    /// restricted and unrestricted Kohn-Sham; the converged lower one is kept.
    pub fn spinor_kohn_sham(
        &self,
        functional: &str,
        electrons: usize,
        options: &super::scf::ScfOptions,
    ) -> Result<super::scf::MeanField> {
        self.spinor_kohn_sham_from(functional, electrons, options, None)
    }
    /// Spinor Kohn-Sham from a noncollinear start: one moment vector per atom (Bohr magneton).
    /// Collinear unrestricted Kohn-Sham runs first with the moments' lengths (the scalar part of
    /// a spin-orbit dataset), then each Loewdin function's share of its magnetization is turned
    /// to its atom's direction and the spinor SCF starts from that density (its first
    /// diagonalization makes it a projector). Without a spin-orbit term the collinear solution is
    /// a stable fixed point of the spinor SCF, so a noncollinear state is reached only from such
    /// a start.
    pub fn spinor_kohn_sham_noncollinear(
        &self,
        functional: &str,
        charge: i32,
        moments: &[[f64; 3]],
        electrons: Option<usize>,
        options: &super::scf::ScfOptions,
    ) -> Result<super::scf::MeanField> {
        let atoms = self.geometry.numbers().len();
        if moments.len() != atoms || moments.iter().flatten().any(|v| !v.is_finite()) {
            return Err(invalid("noncollinear initial moments need one finite vector per atom"));
        }
        let lengths: Vec<f64> = moments
            .iter()
            .map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
            .collect();
        let particles = self.particles_from_moments(charge, &lengths, electrons)?;
        let collinear = self.collinear_kohn_sham(
            functional,
            particles,
            true,
            options,
            "fermi-dirac",
            super::occupation::DEFAULT_WIDTH,
        )?;
        let m = self.m();
        // The atom of each Loewdin function: the one holding most of its weight.
        let lowdin = &self.lowdin;
        let directions: Vec<[f64; 3]> = (0..m)
            .map(|j| {
                let mut weights = vec![0f64; atoms];
                for (mu, orbital) in self.basis.orbitals.iter().enumerate() {
                    weights[orbital.atom] += lowdin[(mu, j)].powi(2);
                }
                let atom = (0..atoms)
                    .max_by(|&a, &b| weights[a].total_cmp(&weights[b]))
                    .expect("a molecule has atoms");
                let length = lengths[atom];
                if length > 0. {
                    moments[atom].map(|v| v / length)
                } else {
                    [0.; 3]
                }
            })
            .collect();
        let (p, pm) = (
            &collinear.densities[0] + &collinear.densities[1],
            &collinear.densities[0] - &collinear.densities[1],
        );
        // m_k = (W_k P_m + P_m W_k) / 2, W_k = diag of the k-th direction component.
        let channel = |k: usize| {
            Matrix::from_fn(m, m, |a, b| pm[(a, b)] * C64::new(0.5 * (directions[a][k] + directions[b][k]), 0.))
        };
        let (mx, my, mz) = (channel(0), channel(1), channel(2));
        let half = C64::new(0.5, 0.);
        let mut start = Matrix::zeros(2 * m, 2 * m);
        start.view_mut((0, 0), (m, m)).copy_from(&((&p + &mz) * half));
        start.view_mut((m, m), (m, m)).copy_from(&((&p - &mz) * half));
        let off = (&mx - &my * C64::new(0., 1.)) * half;
        start.view_mut((0, m), (m, m)).copy_from(&off);
        start.view_mut((m, 0), (m, m)).copy_from(&off.adjoint());
        self.spinor_kohn_sham_from(functional, particles[0] + particles[1], options, Some(start))
    }
    /// Spinor Kohn-Sham from the default starts, or from the orbitals of the Fock matrix of
    /// `density` (a `2m x 2m` start density) when one is given.
    fn spinor_kohn_sham_from(
        &self,
        functional: &str,
        electrons: usize,
        options: &super::scf::ScfOptions,
        density: Option<Matrix>,
    ) -> Result<super::scf::MeanField> {
        if self.corrections.as_ref().is_some_and(|c| c.dft.is_none()) {
            return Err(invalid("native molecular PAW Kohn-Sham augmentation is not yet implemented"));
        }
        check(self, functional)?;
        let m = self.m();
        if electrons == 0
            || electrons > 2 * m
            || options.max_iterations == 0
            || !options.tolerance.is_finite()
            || options.tolerance <= 0.
        {
            return Err(invalid("invalid spinor Kohn-Sham occupation, iteration budget or tolerance"));
        }
        self.context.run(|| {
            let ctx = &self.context;
            let workspace = ctx.reserve(
                crate::product(&[2 * m, 2 * m, 16, 160])?
                    .checked_add(crate::product(&[options.max_iterations, 128, 4])?)
                    .ok_or(crate::Error::Overflow)?,
            )?;
            let core_h = self.kohn_sham_core();
            let mut h = Matrix::zeros(2 * m, 2 * m);
            h.view_mut((0, 0), (m, m)).copy_from(&core_h);
            h.view_mut((m, m), (m, m)).copy_from(&core_h);
            // The spin-orbit term, built once: a fixed one-body operator of every iteration.
            let spin_orbit = spin_orbit_term(self)?;
            if let Some(so) = &spin_orbit {
                h += so;
            }
            let (_, core) = crate::math::linalg::eigen(h)?;
            // A start with the highest occupied and lowest empty spinors turned in spin space, so the
            // iteration does not begin collinear.
            let mut turned = core.clone();
            if electrons < 2 * m {
                let (i, a) = (electrons - 1, electrons);
                let (cos, sin) = (0.4f64.cos(), 0.4f64.sin());
                for k in 0..2 * m {
                    let (ci, ca) = (core[(k, i)], core[(k, a)]);
                    turned[(k, i)] = ci * cos + ca * C64::new(0., sin);
                    turned[(k, a)] = ci * C64::new(0., sin) + ca * cos;
                }
            }
            let shared = Shared::new(self, functional)?;
            let evaluate = |d: &Matrix| spinor_parts(self, d, functional, spin_orbit.as_ref(), &shared).map(|p| p.0);
            let starts = match &density {
                Some(d) => vec![crate::math::linalg::eigen(evaluate(d)?.fock)?.1],
                None => vec![core, turned],
            };
            let mut best: Option<(super::scf::Candidate, f64, f64)> = None;
            for start in starts {
                let run = self.iterate_start(&[start], &[electrons], None, options, |d| {
                    let e = evaluate(&d[0])?;
                    Ok((vec![e.fock], e.energy))
                })?;
                let (c, d, history, converged) = (run.coefficients, run.densities, run.history, run.converged);
                let value = evaluate(&d[0])?;
                let (energies, _) = crate::math::linalg::eigen(value.fock.clone())?;
                let candidate = super::scf::Candidate {
                    energy: value.energy,
                    converged,
                    history,
                    coefficients: c,
                    densities: d,
                    energies: vec![energies],
                    unstable_pair: None,
                    occupations: None,
                    aufbau_inversion: None,
                    entropy: 0.,
                };
                if best.as_ref().is_none_or(|(b, _, _)| {
                    (candidate.converged && !b.converged)
                        || (candidate.converged == b.converged && candidate.energy < b.energy - 1e-12)
                }) {
                    best = Some((candidate, value.hartree, value.xc));
                }
            }
            let (candidate, eh, exc) = best.expect("a start ran");
            self.finish(
                "ghf",
                [electrons.div_ceil(2), electrons / 2],
                options,
                candidate,
                workspace,
                Some((functional, eh, exc)),
                None,
            )
        })
    }
}

/// The magnetization channels' explicit exchange-correlation gradient at fixed AO-frame densities
/// `D_a,ao = L Re(D_a) L^T` (`a = 1, 2, 3`): `dE/dR_X = -2 sum_a sum_{mu in X, nu} D_a,ao[mu, nu]
/// dv sum_k grad chi_mu(k) (O_a chi_nu)(k)` (`3 x atoms`, atom-major), with `O_a` the channel's
/// operator on the grid: `v_a`, and for a meta-GGA `- 1/2 D (v_tau,a D) + F_a . D - D (F_a .)`
/// with `D` the grid derivative (the scalar contraction's `kinetic_density_action` and
/// `density_gradient_action`). The charge channel is the shared contraction's, with channel 0.
fn magnetization_gradient(integrals: &MolecularIntegrals, d: &Matrix, potentials: &ChannelPotentials) -> Result<Vec<f64>> {
    let (n, m) = (integrals.grid.index.points(), integrals.m());
    let ao = integrals.basis.orbitals.len();
    let atoms = integrals.geometry.numbers().len();
    let positions = integrals.geometry.positions();
    let dv = integrals.grid.index.dv();
    let l = &integrals.lowdin;
    let ctx = &integrals.context;
    let mut out = vec![0.; 3 * atoms];
    let owner: Vec<usize> = (0..ao)
        .map(|mu| {
            let center = integrals.basis.orbitals[mu].center;
            let gap = |a: usize| (0..3).map(|k| (positions[a][k] - center[k]).powi(2)).sum::<f64>();
            (0..atoms).min_by(|&a, &b| gap(a).total_cmp(&gap(b))).expect("one atom")
        })
        .collect();
    // chi_nu(k) for every AO function, and T_a = D_a,ao chi (ao x n) per channel.
    let _memory = ctx.reserve(crate::product(&[4 * ao, n, 8])?)?;
    let mut samples = vec![0.; ao * n];
    let mut row = ctx.allocate::<f64>(n)?;
    for nu in 0..ao {
        integrals.samples.widen_range(nu * n, row.as_mut_slice())?;
        samples[nu * n..(nu + 1) * n].copy_from_slice(row.as_slice());
    }
    let mut weighted = vec![vec![0.; ao * n]; 3];
    let mut field = vec![0.; n];
    let (mut slope, mut product, mut derivative) = (vec![0.; n], vec![0.; n], vec![0.; n]);
    for a in 1..4 {
        let channel = noncollinear::component(d, m, a).map(|v| v.re);
        let dao = l * ((&channel + channel.transpose()) * 0.5) * l.transpose();
        let v = &potentials.v.as_slice()[a * n..(a + 1) * n];
        for mu in 0..ao {
            // u_mu = sum_nu D_a,ao[mu, nu] chi_nu, then O_a u_mu.
            field.fill(0.);
            for nu in 0..ao {
                let c = dao[(mu, nu)];
                if c != 0. {
                    for (k, t) in field.iter_mut().enumerate() {
                        *t += c * samples[nu * n + k];
                    }
                }
            }
            if field.iter().all(|&u| u == 0.) {
                continue;
            }
            let target = &mut weighted[a - 1][mu * n..(mu + 1) * n];
            for k in 0..n {
                target[k] = v[k] * field[k];
            }
            if let Some((kinetic, flux)) = &potentials.meta {
                let tau = &kinetic.as_slice()[a * n..(a + 1) * n];
                for axis in 0..3 {
                    let w = &flux.as_slice()[(3 * a + axis) * n..(3 * a + axis + 1) * n];
                    xc::derivative(&integrals.grid, &field, axis, &mut slope)?;
                    for k in 0..n {
                        product[k] = tau[k] * slope[k];
                    }
                    xc::derivative(&integrals.grid, &product, axis, &mut derivative)?;
                    for k in 0..n {
                        target[k] += w[k] * slope[k] - 0.5 * derivative[k];
                        product[k] = w[k] * field[k];
                    }
                    xc::derivative(&integrals.grid, &product, axis, &mut derivative)?;
                    for k in 0..n {
                        target[k] -= derivative[k];
                    }
                }
            }
        }
    }
    for mu in 0..ao {
        let orbital = &integrals.basis.orbitals[mu];
        let x = owner[mu];
        for k in 0..n {
            let s: f64 = (0..3).map(|a| weighted[a][mu * n + k]).sum();
            if s == 0. {
                continue;
            }
            let g = orbital.gradient(integrals.grid.point(k));
            for axis in 0..3 {
                out[3 * x + axis] -= 2. * dv * g[axis] * s;
            }
        }
    }
    Ok(out)
}

impl MolecularIntegrals {
    /// The analytic gradient `dE/dR` (`3 x atoms`) of a converged spinor Kohn-Sham state
    /// ([`Self::spinor_kohn_sham`]): the shared PAW contraction of the charge (pseudo-orbitals of
    /// `Re(D_0)`, the W of the full spinor Fock, the exchange-correlation potential `v_0`), plus
    /// the magnetization channels' explicit term and the spin-orbit term's
    /// (`Setup::spin_orbit_gradient`). LDA, PBE, PBE+rVV10 and r2SCAN.
    pub fn spinor_gradient(&self, state: &super::scf::MeanField) -> Result<super::forces::Gradient> {
        let name = state
            .functional
            .as_deref()
            .filter(|_| state.method == "ghf")
            .ok_or_else(|| invalid("a spinor gradient needs a spinor Kohn-Sham state"))?;
        let d = state.densities.first().ok_or_else(|| invalid("a spinor state holds one density"))?;
        self.spinor_gradient_of(d, name, state.electronic_energy)
    }

    /// [`Self::spinor_gradient`] of a stationary spinor density `d` (orthonormal frame, `2m x 2m`)
    /// with electronic energy `energy`: the same checks (it must reproduce `energy` and be
    /// stationary), for a state that is not a `MeanField` of this crate (an embedded collinear
    /// one in the tests).
    #[doc(hidden)]
    pub fn spinor_gradient_of(&self, d: &Matrix, name: &str, energy: f64) -> Result<super::forces::Gradient> {
        use super::paw_forces::Frozen;
        use crate::paw::molecular::gradient::Occupied;
        check(self, name)?;
        let hybrid = crate::models::functional(name)?.screened_exchange;
        let setup = self.paw_setup.as_ref().ok_or_else(|| invalid("forces need a PAW setup"))?;
        let (m, n) = (self.m(), self.grid.index.points());
        if d.shape() != (2 * m, 2 * m) {
            return Err(invalid("a spinor state holds one 2m x 2m density"));
        }
        let electronic_energy = energy;
        self.context.run(|| {
            let spin_orbit = spin_orbit_term(self)?;
            let (evaluation, mut fields, potentials) = spinor_parts(self, d, name, spin_orbit.as_ref(), &Shared::new(self, name)?)?;
            if (evaluation.energy - electronic_energy).abs() > 1e-7 {
                return Err(invalid("the spinor state does not reproduce its energy on these integrals"));
            }
            let f = &evaluation.fock;
            let stationarity = ((Matrix::identity(2 * m, 2 * m) - d) * f * d)
                .iter()
                .fold(0f64, |a, v| a.max(v.norm()));
            if stationarity > 1e-6 {
                return Err(invalid(format!("the spinor state is not stationary: (1 - D) F D reaches {stationarity:.2e}")));
            }
            // The charge channel's xc potentials in both spin blocks of the shared contraction
            // (a meta-GGA's kinetic potential and flux likewise).
            for k in 0..n {
                let v0 = potentials.v.as_slice()[k];
                fields.terms.potentials.as_mut_slice()[k] = v0;
                fields.terms.potentials.as_mut_slice()[n + k] = v0;
            }
            if let Some((kinetic, flux)) = &potentials.meta {
                let (Some(tau), Some(gradient)) = (fields.terms.tau_potentials.as_mut(), fields.terms.gradient_potentials.as_mut()) else {
                    return Err(invalid("the restricted meta-GGA evaluation returned no kinetic or gradient potentials"));
                };
                for spin in 0..2 {
                    tau.as_mut_slice()[spin * n..(spin + 1) * n].copy_from_slice(&kinetic.as_slice()[..n]);
                    gradient.as_mut_slice()[3 * spin * n..3 * (spin + 1) * n].copy_from_slice(&flux.as_slice()[..3 * n]);
                }
            }
            let l = &self.lowdin;
            let mut orbitals = Vec::new();
            let charge = noncollinear::component(d, m, 0).map(|v| v.re);
            let (values, vectors) = crate::math::linalg::real_eigen((&charge + charge.transpose()) * 0.5)?;
            for (i, &weight) in values.iter().enumerate() {
                if weight.abs() > 1e-13 {
                    let ao = l * vectors.column(i);
                    orbitals.push(Occupied { spin: 0, weight, energy: 0., w_weight: 0., ao: ao.iter().copied().collect() });
                }
            }
            // W = D F D (F with h^SO and the magnetization potentials), spin-traced.
            let dfd = d * f * d;
            let w = noncollinear::component(&dfd, m, 0).map(|v| v.re);
            let (values, vectors) = crate::math::linalg::real_eigen((&w + w.transpose()) * 0.5)?;
            for (i, &w_weight) in values.iter().enumerate() {
                if w_weight.abs() > 1e-15 {
                    let ao = l * vectors.column(i);
                    orbitals.push(Occupied { spin: 0, weight: 0., energy: 0., w_weight, ao: ao.iter().copied().collect() });
                }
            }
            let frozen = Frozen {
                functional: Some(name.to_string()),
                unrestricted: false,
                hf: false,
                valence: fields.valence,
                hartree: fields.hartree,
                xc: Some(fields.terms),
                orbitals,
                stationarity,
                exact_exchange: false,
            };
            let mut gradient = self.contract_frozen(frozen, setup)?;
            let atoms = self.geometry.numbers().len();
            let mut extra = vec![("xc/magnetization_basis", magnetization_gradient(self, d, &potentials)?)];
            if let Some((omega, fraction)) = hybrid {
                // The exchange pairs need the occupied spinors themselves: D must be a projector.
                let (occupations, vectors) = crate::math::linalg::eigen(d.clone())?;
                let mut spinors = Vec::new();
                for (k, &occupation) in occupations.iter().enumerate() {
                    if (occupation * (1. - occupation)).abs() > 1e-8 {
                        return Err(invalid(format!(
                            "a hybrid spinor force needs unit occupations; the density has one of {occupation:.3e}"
                        )));
                    }
                    if occupation > 0.5 {
                        let li = l.map(|v| C64::new(v, 0.));
                        let column = vectors.column(k);
                        let up = &li * column.rows(0, m);
                        let down = &li * column.rows(m, m);
                        spinors.push([up.iter().copied().collect(), down.iter().copied().collect()]);
                    }
                }
                for (term, values) in self.spinor_exchange_terms(setup, &spinors, omega, fraction)? {
                    extra.push((term, values));
                }
            }
            if spin_orbit.is_some() {
                let (ao, li) = (l.nrows(), l.map(|v| C64::new(v, 0.)));
                let mut lift = Matrix::zeros(2 * ao, 2 * m);
                lift.view_mut((0, 0), (ao, m)).copy_from(&li);
                lift.view_mut((ao, m), (ao, m)).copy_from(&li);
                let d_ao = &lift * d * lift.transpose();
                let so = setup.spin_orbit_gradient(&self.context, self.geometry.positions(), &d_ao)?;
                extra.push(("spin_orbit/projections", so.into_iter().flatten().collect()));
            }
            for (name, values) in extra {
                for (t, v) in gradient.total.as_mut_slice().iter_mut().zip(&values) {
                    *t += v;
                }
                let mut buffer = self.context.allocate::<f64>(3 * atoms)?;
                buffer.as_mut_slice().copy_from_slice(&values);
                gradient.terms.push((name, buffer));
            }
            Ok(gradient)
        })
    }
}
