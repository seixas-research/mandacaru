//! The two-component linear response of a spinor Kohn-Sham PAW-LCAO molecule
//! (`dev_plan/TDDFT_PLAN.md`, T8): Tamm-Dancoff and Casida excitations of a reference whose
//! orbitals are complex spinors (a spin-orbit dataset, or noncollinear moments), with the
//! noncollinear semilocal kernel ([`NoncollinearKernel`]).
//!
//! The response is over the complex amplitudes `k` (occupied x empty) of the Hermitian change
//! `dD(k) = sum_ia k_ia |a><i| + conj(k_ia) |i><a|`. With `G(k)_ia = <a| F1[dD(k)] |i>` (the
//! first-order Fock of that change: Hartree of the charge, the compensation pairs and the
//! kernel of the four Pauli channels), the operators of Casida's paired problem are real-linear
//! maps of `k` (the coupling `B` acts on `conj(k)`):
//!
//! ```text
//! (A + B) k = Delta k + G(k),      (A - B) k = Delta k - i G(i k),
//! ```
//!
//! and the Tamm-Dancoff `A` is their half sum. For real orbitals and a density-only kernel the
//! second is `Delta k` (an imaginary antisymmetric change has no density); with spinors it is not:
//! the magnetization is odd under time reversal. Each acts on `k` as a real vector `(Re k, Im k)`,
//! so the solvers are the real ones; every root then comes twice (`(P, Q)` and `(i Q, i P)`,
//! the same state), and the complex amplitudes are a basis over C of each group of equal roots.
//!
//! Atomic units: Hartree.
use super::{
    invalid, properties, response::view, Approximation, Channel, Coupling, Excitations, Operator,
    Request, Spin,
    Transition,
};
use crate::math::{fft::Poisson, linalg::eigen};
use crate::models::{noncollinear, xc_kernel::NoncollinearKernel};
use crate::molecular::{
    multipoles::{gradient_integrals, position_integrals},
    spinor_dft, MeanField, MolecularIntegrals,
};
use crate::{product, quantum::C64, Buffer, Result};
use nalgebra::DMatrix;

pub mod gradient;

const BLOCK: usize = 2048;

/// The complex part of the excitations of a spinor reference ([`Excitations::spinor`]).
///
/// A state is `s = (X, Y)` with `c s = (c X, conj(c) Y)` (the partner `(i Q, i P)` of a real root
/// `(P, Q)` is `i s`); the states of each group of equal roots are orthonormal in
/// `h(s, s') = sum conj(X) X' - sum Y conj(Y')`, so every sum over a group is basis-free. The
/// phase of each state is a convention.
pub struct Spinors {
    /// `mu_n = <0| r |n> = sum <i|r|a> X_ia + conj(<i|r|a> Y_ia)`, e Bohr.
    pub transition_dipoles: Vec<[C64; 3]>,
    /// `p_n = <0| grad |n> = sum <i|grad|a> X_ia - conj(<i|grad|a> Y_ia)` (Bohr^-1).
    pub transition_velocities: Vec<[C64; 3]>,
    /// `X + Y` and `X - Y` of each state (occupied x empty spinors; Tamm-Dancoff: `X`).
    pub plus: Vec<DMatrix<C64>>,
    pub minus: Vec<DMatrix<C64>>,
    /// The spinors (columns, ascending) over the nonorthogonal atomic basis, spin-up block first.
    pub atomic_orbitals: DMatrix<C64>,
}

/// The paired operators of a spinor reference, on `(Re k, Im k)`.
pub struct Spinor<'a> {
    integrals: &'a MolecularIntegrals,
    m: usize,
    o: usize,
    v: usize,
    energies: Vec<f64>,
    orbitals: DMatrix<C64>,
    delta: Vec<f64>,
    kernel: NoncollinearKernel,
    /// The orthonormal atomic functions on the grid, `[function][point]`.
    samples: Buffer<f64>,
    /// Whether the molecule carries a spin-orbit term.
    spin_orbit: bool,
    /// The spin-rotation zero modes ([`spin_rotations`]), empty with spin-orbit coupling.
    rotations: Vec<Vec<f64>>,
    /// Whether `i` maps the zero modes onto themselves (one spin flips, as in a one-electron
    /// reference): then they are the zero modes of `A - B` and of `A` too.
    closed: bool,
}

/// Where [`Spinor`] puts its spin-rotation zero modes (Hartree): far above every excitation, so the
/// lowest roots and the Z-vector see only the complement.
const ROTATION_SHIFT: f64 = 1e3;

/// The real column `(Re k, Im k)` (row-major) of the amplitudes `k` (occupied x empty).
fn real_column(k: &DMatrix<C64>) -> Vec<f64> {
    let (o, v) = (k.nrows(), k.ncols());
    let size = o * v;
    let mut out = vec![0.; 2 * size];
    for i in 0..o {
        for a in 0..v {
            out[i * v + a] = k[(i, a)].re;
            out[size + i * v + a] = k[(i, a)].im;
        }
    }
    out
}

/// The global spin rotations `i sigma_a` of the reference (`sigma_a` on the spin blocks of the
/// orthonormal frame) as orthonormal occupied-empty columns of the real embedding: the change of
/// the spinors that turns every spin together. A rotation that leaves the reference unchanged (each
/// rotation of a Kramers-unpolarized one, the one about a collinear axis) has no occupied-empty
/// part and is dropped. Without spin-orbit coupling these are zero modes of `A + B`, and `i` turns
/// them into those of `A - B` (the Goldstone modes of the spin axis, at `omega = 0`): they change
/// the reference, not its state, so [`Spinor`] solves on their complement and lifts them to
/// [`ROTATION_SHIFT`].
fn spin_rotations(m: usize, o: usize, orbitals: &DMatrix<C64>) -> Vec<Vec<f64>> {
    let v = 2 * m - o;
    let (co, cv) = (orbitals.columns(0, o), orbitals.columns(o, v));
    let (one, i) = (C64::new(1., 0.), C64::i());
    let mut modes: Vec<Vec<f64>> = vec![];
    let zero = C64::new(0., 0.);
    for axis in 0..3 {
        let mut k = DMatrix::<C64>::zeros(2 * m, 2 * m);
        for p in 0..m {
            let (up, down, flip) = match axis {
                0 => (zero, zero, [i, i]),
                1 => (zero, zero, [one, -one]),
                _ => (i, -i, [zero; 2]),
            };
            k[(p, p)] = up;
            k[(m + p, m + p)] = down;
            k[(p, m + p)] = flip[0];
            k[(m + p, p)] = flip[1];
        }
        let mut column = real_column(&(cv.adjoint() * k * co).transpose());
        let size = (column.iter().map(|x| x * x).sum::<f64>()).sqrt();
        project(&modes, &mut column);
        let norm = column.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-6 * size.max(1.) {
            column.iter_mut().for_each(|x| *x /= norm);
            modes.push(column);
        }
    }
    modes
}

/// Removes the share of `x` along the orthonormal `modes`.
pub(super) fn project(modes: &[Vec<f64>], x: &mut [f64]) {
    for n in modes {
        let c: f64 = n.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        x.iter_mut().zip(n).for_each(|(y, a)| *y -= c * a);
    }
}

impl<'a> Spinor<'a> {
    /// The operator of `reference`, a converged spinor Kohn-Sham state of `integrals`.
    pub fn new(integrals: &'a MolecularIntegrals, reference: &MeanField) -> Result<Self> {
        let unsupported = |what: &str| Err(crate::Error::Unsupported(what.into()));
        super::response::check_common(integrals, reference)?;
        let name = reference.functional.as_deref().ok_or_else(|| {
            invalid("two-component Hartree-Fock (method='ghf') has no response yet")
        })?;
        let definition = crate::models::functional(name)?;
        if definition.requires_tau
            || definition.screened_exchange.is_some()
            || definition.nonlocal_b.is_some()
        {
            return unsupported(
                "the spinor response is implemented for xc='lda' and xc='pbe' (r2SCAN's local frame, a hybrid's spinor exchange and rVV10 are not)",
            );
        }
        let m = integrals.m();
        let d = reference
            .densities
            .first()
            .filter(|d| reference.densities.len() == 1 && d.shape() == (2 * m, 2 * m))
            .ok_or_else(|| invalid("a spinor reference has one 2m x 2m density matrix"))?;
        let o = reference.particles[0] + reference.particles[1];
        if o == 0 || o >= 2 * m {
            return Err(invalid(
                "a spinor reference needs occupied and empty spinors",
            ));
        }
        let spin_orbit = spinor_dft::spin_orbit_term(integrals)?;
        let evaluation = spinor_dft::spinor_response(integrals, d, name, spin_orbit.as_ref())?;
        let fock = (&evaluation.fock + evaluation.fock.adjoint()) * C64::new(0.5, 0.);
        let (energies, orbitals) = eigen(fock)?;
        let projector = orbitals.columns(0, o) * orbitals.columns(0, o).adjoint();
        let stationarity = (projector - d).iter().fold(0f64, |a, z| a.max(z.norm()));
        if stationarity > 1e-6 {
            return Err(invalid(format!(
                "the spinor reference is not a converged projector of its Fock matrix ({stationarity:.1e}): converge the SCF or fill the lowest spinors"
            )));
        }
        let v = 2 * m - o;
        let delta: Vec<f64> = (0..2)
            .flat_map(|_| {
                (0..o)
                    .flat_map(|i| (0..v).map(move |a| (i, a)))
                    .map(|(i, a)| energies[o + a] - energies[i])
                    .collect::<Vec<_>>()
            })
            .collect();
        let samples = super::response::mo_samples(integrals, &DMatrix::identity(m, m))?;
        let ctx = &integrals.context;
        let n = integrals.grid.index.points();
        let channels: Vec<DMatrix<f64>> = (0..4)
            .map(|a| noncollinear::component(d, m, a).map(|z| z.re))
            .collect();
        let mut fields = ctx.allocate::<f64>(product(&[4, n])?)?;
        for (a, matrix) in channels.iter().enumerate() {
            density(
                samples.as_slice(),
                n,
                m,
                matrix,
                &mut fields.as_mut_slice()[a * n..(a + 1) * n],
            )?;
        }
        let paw = integrals.corrections.as_ref().and_then(|c| c.dft.as_ref());
        if let Some(paw) = paw {
            for k in 0..n {
                fields.as_mut_slice()[k] += paw.core.as_slice()[k];
            }
        }
        let relativistic = paw.is_some_and(|p| p.relativistic_exchange)
            && matches!(definition.semilocal, "lda" | "pbe");
        let kernel = NoncollinearKernel::new(
            ctx,
            &integrals.grid,
            fields.as_slice(),
            definition.semilocal,
            relativistic,
        )?;
        let rotations = if spin_orbit.is_some() { vec![] } else { spin_rotations(m, o, &orbitals) };
        // `i` maps the zero modes of `A + B` onto those of `A - B`. They are the same space where
        // the turns about two perpendicular axes are each other's `i` (one spin flips), not where
        // both spins flip or the magnetization is noncollinear.
        let turned = |x: &[f64]| -> Vec<f64> {
            let half = x.len() / 2;
            x[half..].iter().map(|y| -y).chain(x[..half].iter().copied()).collect()
        };
        let closed = rotations.iter().all(|n| {
            let mut rest = turned(n);
            project(&rotations, &mut rest);
            rest.iter().map(|x| x * x).sum::<f64>().sqrt() < 1e-6
        });
        Ok(Self {
            integrals,
            m,
            o,
            v,
            energies,
            orbitals,
            delta,
            kernel,
            samples,
            spin_orbit: spin_orbit.is_some(),
            rotations,
            closed,
        })
    }
    /// `k` (occupied x empty) of a real column `(Re k, Im k)`, row-major.
    fn amplitudes(&self, column: &[f64]) -> DMatrix<C64> {
        let size = self.o * self.v;
        DMatrix::from_fn(self.o, self.v, |i, a| {
            C64::new(column[i * self.v + a], column[size + i * self.v + a])
        })
    }
    fn column(&self, k: &DMatrix<C64>) -> Vec<f64> {
        real_column(k)
    }
    /// The occupied spinor count.
    pub fn occupied(&self) -> usize {
        self.o
    }
    /// The spinors of the reference's Fock matrix in the orthonormal frame (columns, ascending).
    pub fn orbitals(&self) -> &DMatrix<C64> {
        &self.orbitals
    }
    /// `dD(k) = sum_ia k_ia |a><i| + conj(k_ia) |i><a|` (`2m x 2m`, orthonormal frame).
    pub fn change(&self, k: &DMatrix<C64>) -> DMatrix<C64> {
        let (co, cv) = (
            self.orbitals.columns(0, self.o),
            self.orbitals.columns(self.o, self.v),
        );
        let half = cv * k.transpose() * co.adjoint();
        &half + half.adjoint()
    }
    /// `G(k)_ia = <a| F1[dD(k)] |i>`: the first-order Fock of the Hermitian change of `k`
    /// (occupied x empty).
    pub fn first_order(&self, k: &DMatrix<C64>) -> Result<DMatrix<C64>> {
        let (o, v) = (self.o, self.v);
        let (co, cv) = (self.orbitals.columns(0, o), self.orbitals.columns(o, v));
        let (fock, _) = self.first_order_fock(&self.change(k))?;
        Ok((cv.adjoint() * fock * co).transpose())
    }
    /// The four Pauli-channel densities (`4 x points`) of a Hermitian `2m x 2m` matrix.
    pub(super) fn channel_densities(&self, x: &DMatrix<C64>) -> Result<Buffer<f64>> {
        let (integrals, m) = (self.integrals, self.m);
        let n = integrals.grid.index.points();
        let mut out = integrals.context.allocate::<f64>(product(&[4, n])?)?;
        for a in 0..4 {
            let channel = noncollinear::component(x, m, a).map(|z| z.re);
            density(
                self.samples.as_slice(),
                n,
                m,
                &channel,
                &mut out.as_mut_slice()[a * n..(a + 1) * n],
            )?;
        }
        Ok(out)
    }
    /// `sum_a sigma_a x V[w_a]` (`2m x 2m`) of four Pauli-channel local potentials `w`.
    pub(super) fn local_fock(&self, w: &[f64]) -> Result<DMatrix<C64>> {
        let (integrals, m) = (self.integrals, self.m);
        let n = integrals.grid.index.points();
        let dv = integrals.grid.index.dv();
        let matrices: Vec<DMatrix<C64>> = (0..4)
            .map(|a| {
                elements(self.samples.as_slice(), n, m, &w[a * n..(a + 1) * n], dv)
                    .map(|x| x.map(|y| C64::new(y, 0.)))
            })
            .collect::<Result<_>>()?;
        let mut fock = DMatrix::zeros(2 * m, 2 * m);
        noncollinear::assemble_into(&mut fock, &matrices, m);
        Ok(fock)
    }
    /// The first-order Fock `F1[x]` (`2m x 2m`) of a Hermitian change `x` (Hartree of the
    /// charge, the compensation pairs and the kernel of the four Pauli channels), and the
    /// kernel's exchange-correlation change `dv` (`4 x points`).
    pub(super) fn first_order_fock(&self, x: &DMatrix<C64>) -> Result<(DMatrix<C64>, Buffer<f64>)> {
        let (integrals, m) = (self.integrals, self.m);
        let ctx = &integrals.context;
        let n = integrals.grid.index.points();
        let delta = self.channel_densities(x)?;
        let mut xc = ctx.allocate::<f64>(product(&[4, n])?)?;
        let mut scratch = self.kernel.scratch(ctx)?;
        self.kernel.apply(
            ctx,
            &integrals.grid,
            delta.as_slice(),
            xc.as_mut_slice(),
            &mut scratch,
        )?;
        let mut poisson = Poisson::isolated(ctx, &integrals.grid)?;
        let mut source = ctx.allocate::<C64>(n)?;
        let mut hartree = ctx.allocate::<C64>(n)?;
        for k in 0..n {
            source.as_mut_slice()[k] = C64::new(delta.as_slice()[k], 0.);
        }
        poisson.solve_into(source.as_slice(), hartree.as_mut_slice())?;
        let mut potentials = ctx.copy_slice(xc.as_slice())?;
        for k in 0..n {
            potentials.as_mut_slice()[k] += hartree.as_slice()[k].re;
        }
        let mut fock = self.local_fock(potentials.as_slice())?;
        // The compensation pairs see the charge: `w_k tr[B_k D_0]`.
        if let Some(paw) = integrals.corrections.as_ref().and_then(|c| c.dft.as_ref()) {
            let charge = noncollinear::component(x, m, 0).map(|z| z.re);
            let mut pairs = DMatrix::<C64>::zeros(m, m);
            for (b, &w) in paw.pairs.factors.iter().zip(&paw.pairs.weights) {
                let weight = w * b.dot(&charge);
                pairs += b.map(|y| C64::new(y * weight, 0.));
            }
            noncollinear::assemble_into(&mut fock, &[pairs], m);
        }
        Ok((fock, xc))
    }
    /// `Delta k + G(k)` (`plus`) or `Delta k - i G(i k)` on the columns of `block`.
    /// `A + B`, `A - B` or `A` (`plus`, `minus`, both) on the columns of `block`, on the
    /// complement of the spin-rotation zero modes `N`, which it lifts to [`ROTATION_SHIFT`]. `N`
    /// is the null space of `A + B`, and `i N` that of `A - B`; a Casida root `(p, q)` has
    /// `q` orthogonal to `N`, so `A - B` deflated by `N` too returns its `q` and `omega` with `p`
    /// less its share along `N`, which [`spinor_excitations`] restores as `(A - B) q / omega`.
    /// `A` (Tamm-Dancoff) is deflated only where `i N = N` (`closed`), its null space then.
    fn paired(&self, block: &DMatrix<f64>, plus: bool, minus: bool) -> Result<DMatrix<f64>> {
        if self.rotations.is_empty() || (plus && minus && !self.closed) {
            return self.bare(block, plus, minus);
        }
        let mut inside = block.clone();
        for mut c in inside.column_iter_mut() {
            project(&self.rotations, c.as_mut_slice());
        }
        let mut out = self.bare(&inside, plus, minus)?;
        for (mut c, x) in out.column_iter_mut().zip(block.column_iter()) {
            project(&self.rotations, c.as_mut_slice());
            for n in &self.rotations {
                let share: f64 = n.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
                c.iter_mut().zip(n).for_each(|(y, a)| *y += ROTATION_SHIFT * share * a);
            }
        }
        Ok(out)
    }
    /// The operator of [`Self::paired`] without the deflation.
    fn bare(&self, block: &DMatrix<f64>, plus: bool, minus: bool) -> Result<DMatrix<f64>> {
        let mut out = DMatrix::zeros(block.nrows(), block.ncols());
        for c in 0..block.ncols() {
            let column: Vec<f64> = block.column(c).iter().copied().collect();
            let k = self.amplitudes(&column);
            let mut total = k.zip_map(&self.amplitudes(&self.delta), |x, d| x * d.re);
            let weight = if plus && minus { 0.5 } else { 1. };
            if plus {
                total += self.first_order(&k)? * C64::new(weight, 0.);
            }
            if minus {
                let i = C64::i();
                total -= self.first_order(&(&k * i))? * (i * weight);
            }
            out.set_column(c, &nalgebra::DVector::from_vec(self.column(&total)));
        }
        Ok(out)
    }
}

impl Operator for Spinor<'_> {
    fn dimension(&self) -> usize {
        2 * self.o * self.v
    }
    fn delta(&self) -> &[f64] {
        &self.delta
    }
    fn sum(&self, block: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        self.paired(block, true, false)
    }
    fn difference(&self, block: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        self.paired(block, false, true)
    }
    fn resonant(&self, block: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        self.paired(block, true, true)
    }
    fn spin(&self) -> Spin {
        Spin::Spinor
    }
}

/// `n(r) = sum_pq x_pq phi_p(r) phi_q(r)` of a real symmetric matrix of the orthonormal functions.
fn density(samples: &[f64], n: usize, m: usize, x: &DMatrix<f64>, out: &mut [f64]) -> Result<()> {
    if x.shape() != (m, m) || out.len() != n {
        return Err(invalid("a spinor channel does not match the grid"));
    }
    for start in (0..n).step_by(BLOCK) {
        let count = BLOCK.min(n - start);
        let phi = view(samples, start, count, m, n);
        let px = phi * x;
        for i in 0..count {
            out[start + i] = (0..m).map(|q| px[(i, q)] * phi[(i, q)]).sum::<f64>();
        }
    }
    Ok(())
}

/// `sum_r phi_p(r) w(r) phi_q(r) dv` of a local potential.
fn elements(samples: &[f64], n: usize, m: usize, w: &[f64], dv: f64) -> Result<DMatrix<f64>> {
    let mut out = DMatrix::zeros(m, m);
    for start in (0..n).step_by(BLOCK) {
        let count = BLOCK.min(n - start);
        let phi = view(samples, start, count, m, n);
        out += phi.transpose() * DMatrix::from_fn(count, m, |i, q| phi[(i, q)] * w[start + i]);
    }
    Ok(out * dv)
}

/// The lowest `request.num_states` excitations of the spinor reference `reference` (Tamm-Dancoff or
/// Casida; adiabatic LDA or PBE), completed to whole groups of equal roots.
pub(super) fn spinor_excitations(
    integrals: &MolecularIntegrals,
    reference: &MeanField,
    request: &Request,
) -> Result<Excitations> {
    if request.coupling != Coupling::Adiabatic {
        return Err(invalid(
            "the spinor response follows the adiabatic kernel (kernel='adiabatic')",
        ));
    }
    if request.spin.is_some() {
        return Err(invalid(
            "a spinor reference mixes the spins: leave spin unset",
        ));
    }
    let operator = Spinor::new(integrals, reference)?;
    // The lifted spin-rotation zero modes are no states: each pair is one complex amplitude.
    let transitions = operator.o * operator.v - if operator.closed { operator.rotations.len() / 2 } else { 0 };
    if request.num_states > transitions {
        return Err(invalid(format!(
            "num_states={} exceeds the {transitions} occupied-to-empty spinor transitions of this \
             basis",
            request.num_states
        )));
    }
    // Each state is a pair of real roots, `k` and `i k`.
    let mut doubled = request.clone();
    doubled.num_states = 2 * request.num_states;
    let (solved, count) = super::roots(&operator, &doubled)?;
    let (m, o, v) = (operator.m, operator.o, operator.v);
    let (co, cv) = (
        operator.orbitals.columns(0, o),
        operator.orbitals.columns(o, v),
    );
    let spinor = |matrices: [DMatrix<f64>; 3]| -> [DMatrix<C64>; 3] {
        matrices.map(|r| {
            let mut out = DMatrix::zeros(2 * m, 2 * m);
            noncollinear::assemble_into(&mut out, &[r.map(|x| C64::new(x, 0.))], m);
            co.adjoint() * out * cv
        })
    };
    let dipoles = spinor(position_integrals(integrals, [0.; 3])?);
    let gradients = spinor(gradient_integrals(integrals)?);
    // `sum_ia w_ia X_ia +- conj(sum_ia w_ia Y_ia)`.
    let moment =
        |w: &[DMatrix<C64>; 3], x: &DMatrix<C64>, y: &DMatrix<C64>, sign: f64| -> [C64; 3] {
            std::array::from_fn(|axis| {
                let (wx, wy) = (
                    w[axis].zip_fold(x, C64::new(0., 0.), |acc, a, b| acc + a * b),
                    w[axis].zip_fold(y, C64::new(0., 0.), |acc, a, b| acc + a * b),
                );
                wx + wy.conj() * sign
            })
        };
    let mut lowdin = DMatrix::zeros(2 * m, 2 * m);
    noncollinear::assemble_into(&mut lowdin, &[integrals.lowdin.map(|x| C64::new(x, 0.))], m);
    let mut spinors = Spinors {
        transition_dipoles: vec![],
        transition_velocities: vec![],
        plus: vec![],
        minus: vec![],
        atomic_orbitals: lowdin * &operator.orbitals,
    };
    let mut out = Excitations {
        approximation: request.approximation,
        spin: Spin::Spinor,
        coupling: request.coupling,
        functional: reference.functional.clone(),
        energies: vec![],
        oscillator_strengths: vec![],
        transition_dipoles: vec![],
        transition_velocities: vec![],
        velocity_oscillator_strengths: vec![],
        plus: vec![],
        minus: vec![],
        transitions: vec![],
        nto_weights: vec![],
        one_center_hartree: None,
        channels: vec![Channel {
            occupied: o,
            virtual_orbitals: v,
            orbital_energies: operator.energies.clone(),
            atomic_orbitals: DMatrix::zeros(0, 0),
        }],
        spin_squared: None,
        reference_spin_squared: None,
        requested: request.num_states,
        added: 0,
        residuals: vec![],
        iterations: solved.iterations,
        applications: solved.applications,
        spinor: None,
    };
    // The states of the group of equal roots being read, `h`-orthonormal.
    let mut group: Vec<(DMatrix<C64>, DMatrix<C64>)> = vec![];
    let half = C64::new(0.5, 0.);
    for j in 0..count {
        let omega = solved.values[j];
        let column =
            |matrix: &DMatrix<f64>| -> Vec<f64> { matrix.column(j).iter().copied().collect() };
        let minus = column(&solved.minus);
        // Where `i N` is not `N` (both spins flip), `p = (A - B) q / omega` takes back its share
        // along the zero modes that the deflated solve drops.
        let plus = if request.approximation == Approximation::Casida && !operator.closed {
            let restored = operator.bare(&DMatrix::from_column_slice(minus.len(), 1, &minus), false, true)?;
            restored.iter().map(|x| x / omega).collect()
        } else {
            column(&solved.plus)
        };
        let (p, q) = (operator.amplitudes(&plus), operator.amplitudes(&minus));
        // A group of equal roots, chained as `roots` completes them: a root of another value is
        // already `h`-orthogonal to the group, so a wide group only costs projections.
        if j == 0 || omega - solved.values[j - 1] >= super::DEGENERATE_ROOTS {
            group.clear();
        }
        let (mut x, mut y) = ((&p + &q) * half, (&p - &q) * half);
        for (gx, gy) in &group {
            let c = gx.dotc(&x) - gy.dotc(&y).conj();
            x -= gx * c;
            y -= gy * c.conj();
        }
        let norm = x.norm_squared() - y.norm_squared();
        // `h(s, s) = P . Q = 1` for a root; a partner (or a combination of the group's states)
        // leaves nothing.
        if norm < 1e-6 {
            continue;
        }
        let scale = C64::new(norm.sqrt().recip(), 0.);
        let (x, y) = (x * scale, y * scale);
        let mu = moment(&dipoles, &x, &y, 1.);
        let velocity = moment(&gradients, &x, &y, -1.);
        let (plus, minus) = (&x + &y, &x - &y);
        out.energies.push(omega);
        out.oscillator_strengths
            .push(2. / 3. * omega * mu.iter().map(|z| z.norm_sqr()).sum::<f64>());
        out.velocity_oscillator_strengths
            .push(2. / 3. * velocity.iter().map(|z| z.norm_sqr()).sum::<f64>() / omega);
        let mut transitions: Vec<Transition> = (0..o)
            .flat_map(|i| (0..v).map(move |a| (i, a)))
            .map(|(i, a)| Transition {
                channel: 0,
                occupied: i,
                virtual_orbital: o + a,
                weight: (plus[(i, a)].conj() * minus[(i, a)]).re,
            })
            .collect();
        properties::keep_dominant(&mut transitions);
        out.transitions.push(transitions);
        let (values, _) = eigen(&plus * plus.adjoint())?;
        let total: f64 = values.iter().map(|w| w.max(0.)).sum();
        out.nto_weights
            .push(values.iter().rev().map(|w| w.max(0.) / total).collect());
        out.residuals.push(solved.residuals[j]);
        spinors.transition_dipoles.push(mu);
        spinors.transition_velocities.push(velocity);
        group.push((x, y));
        spinors.plus.push(plus);
        spinors.minus.push(minus);
    }
    if 2 * out.energies.len() != count {
        return Err(crate::Error::Kernel(format!(
            "{count} real roots of the spinor response gave {} states, not {}: a root and its \
             partner were not both found; tighten the tolerance",
            out.energies.len(),
            count / 2
        )));
    }
    out.added = out.energies.len().saturating_sub(request.num_states);
    out.spinor = Some(spinors);
    Ok(out)
}
