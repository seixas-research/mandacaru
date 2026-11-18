//! The analytic gradient of one excited state of a spinor Kohn-Sham reference
//! (`dev_plan/TDDFT_PLAN.md`, T8): `E = E_0 + omega` for a Kramers-unpolarized reference (a
//! closed shell, with or without spin-orbit coupling; adiabatic LDA or PBE) or a magnetized one
//! without spin-orbit coupling (an odd electron count or initial moments; LDA), Tamm-Dancoff or
//! Casida. A magnetized reference with spin-orbit coupling refuses by name (a Kramers doublet's
//! moment direction is nearly free, and the ones measured are saddles of it), as does one with
//! PBE (measured, its gradient does not follow its energy).
//!
//! In the real embedding of [`Spinor`] a state `(X, Y)` has `p = X + Y` and `q = X - Y`
//! (`Re <p, q> = 1`), and
//!
//! ```text
//! omega = [<p, (A + B) p> + <q, (A - B) q>] / 2 = tr(F Delta) + [E''[T_p] + E''[T_q]] / 4,
//! ```
//!
//! with the Hermitian directions `T_p = dD(p)` and `T_q = dD(i q)` (each an ordinary change of
//! the density, its imaginary part included) and the unrelaxed difference density
//! `Delta = [C_v (p^T conj p + q^T conj q) C_v^+ - C_o (conj p p^T + conj q q^T) C_o^+] / 2`.
//! Every matrix is a `2m x 2m` one of the orthonormal frame. A unitary rotation `e^K` of the
//! spinors moves every matrix `X` built on them by `[K, X]`, so the Lagrangian
//! `L = E[D] + omega + tr(F dD(z))` is stationary in the spinors when `A - A^+ = 0` for
//!
//! ```text
//! A = D (F + F1[P] + sum_j F2[T_j, T_j] / 4) + P F + sum_j T_j F1[T_j] / 2,   P = Delta + dD(z),
//! ```
//!
//! (`F1` the first-order Fock of [`Spinor`], `F2` the second order one of the noncollinear
//! kernel's [`NoncollinearKernel::second_change`]). Its occupied-empty block is the Z-vector
//! equation `(A + B) z = r` (`r` the block of the part without `z`), solved by conjugate
//! gradients on the real embedding; its occupied and empty diagonal blocks vanish because the
//! amplitudes are a root. The Hermitian `A` is the energy-weighted density of the orthonormality
//! constraint. Without spin-orbit coupling the energy does not depend on the spin axis, so the
//! global spin rotations `i sigma_a` of a magnetized reference are zero modes of `A + B`; the
//! right-hand side has no share along them (the invariance of `omega`, up to the root's
//! convergence), and the Z-vector is solved on their complement (`Spinor`'s deflated `A + B`).
//!
//! The explicit part is the spinor force contraction at a fixed density (the charge through the
//! shared PAW contraction, the magnetization channels and the spin-orbit term) along `P` once and
//! along each `T_j` twice, taken exactly as the collinear gradient takes its directions
//! (`super::super::gradient`): `[G(D + P) - G(D - P)] / 2` with the potentials `v_0 +- dv[P]`,
//! and `G(D + T) + G(D - T) - 2 G(D)` with `v_0 +- dv[T] + d^2 v[T, T] / 2`.
//!
//! [`NoncollinearKernel::second_change`]: crate::models::xc_kernel::NoncollinearKernel::second_change
//!
//! Atomic units: Hartree and Hartree per Bohr.
use super::{density, invalid, project, Spinor, Spinors};
use crate::models::noncollinear;
use crate::molecular::excitations::{
    gradient::{frozen_contraction, z_vector, Hessian},
    Excitations,
};
use crate::molecular::{forces::Gradient, spinor_dft, MeanField, MolecularIntegrals};
use crate::{quantum::C64, Result};
use nalgebra::DMatrix;

type Matrix = DMatrix<C64>;

impl Hessian for Spinor<'_> {
    fn gaps(&self) -> &[f64] {
        &self.delta
    }
    fn apply(&self, block: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        self.paired(block, true, false)
    }
}

/// The gradient of one spinor state and the pieces the excited state reports.
pub(in crate::molecular::excitations) struct SpinorGradient {
    pub gradient: Gradient,
    pub parts: Vec<(&'static str, Vec<f64>)>,
    /// The charge (spin-traced, real) parts of `P`, `T_p`, `T_q` and the excitation's `A`
    /// (orthonormal frame).
    pub relaxed: DMatrix<f64>,
    pub transition: [DMatrix<f64>; 2],
    pub energy_weighted: DMatrix<f64>,
    pub z_residual: f64,
    pub z_iterations: usize,
}

/// The identities of the engine's own numbers a gate reads: `omega` rebuilt from the
/// Lagrangian's pieces, the relative anti-Hermitian part of `A` (all blocks), the largest
/// magnetization `P` puts on the grid relative to its charge (the real parts of the channel
/// matrices; zero for a state of a Kramers-unpolarized reference that is not a member of a
/// degenerate pair), and the largest miss of the fixed-density
/// contraction at the reference against the ground-state gradient (Hartree per Bohr); and, without
/// spin-orbit coupling, the count of spin-rotation zero modes projected out of the Z-vector, the
/// largest `|(A + B) n|` of those unit modes and the share `|N^T r| / |r|` of the right-hand side
/// along them (both zero up to the reference's and the root's convergence).
#[derive(Clone, Copy, Debug)]
pub struct Diagnostics {
    pub omega: f64,
    pub asymmetry: f64,
    pub odd: f64,
    pub ground: f64,
    pub modes: usize,
    pub mode_action: f64,
    pub mode_share: f64,
}

/// `[Re tau, Im tau]` of the half transition density `tau = C_v p^T C_o^+` of state `j` over the
/// raw atomic basis (`2m x 2m`): what a scan follows ([`transition_overlap`]).
pub(in crate::molecular::excitations) fn followed_transition(
    spinors: &Spinors,
    o: usize,
    j: usize,
) -> Vec<DMatrix<f64>> {
    let a = &spinors.atomic_orbitals;
    let v = a.ncols() - o;
    let tau = a.columns(o, v) * spinors.plus[j].transpose() * a.columns(0, o).adjoint();
    vec![tau.map(|z| z.re), tau.map(|z| z.im)]
}

/// `|<tau_a, tau_b>| / (|tau_a| |tau_b|)` with `<x, y> = tr(x^+ S y S)` (`S` the overlap of each
/// spin block): invariant under the phase of either state.
pub(in crate::molecular::excitations) fn transition_overlap(
    s: &DMatrix<f64>,
    a: &[DMatrix<f64>],
    b: &[DMatrix<f64>],
) -> f64 {
    let m = s.nrows();
    let mut metric = Matrix::zeros(2 * m, 2 * m);
    noncollinear::assemble_into(&mut metric, &[s.map(|x| C64::new(x, 0.))], m);
    let complex = |x: &[DMatrix<f64>]| x[0].zip_map(&x[1], C64::new);
    let (a, b) = (complex(a), complex(b));
    let inner = |x: &Matrix, y: &Matrix| (x.adjoint() * &metric * y * &metric).trace();
    let norm = (inner(&a, &a).re * inner(&b, &b).re).sqrt();
    if norm > 0. {
        inner(&a, &b).norm() / norm
    } else {
        0.
    }
}

/// The real symmetric half charge `Re(x_0) / 2` of a Hermitian `2m x 2m` matrix: each spin's
/// share in the collinear contraction.
fn half_charge(x: &Matrix, m: usize) -> DMatrix<f64> {
    let c = noncollinear::component(x, m, 0).map(|z| z.re);
    (&c + c.transpose()) * 0.25
}

/// The spinor force contraction at the fixed Hermitian density `x` (`2m x 2m`, orthonormal
/// frame, unfloored), with the energy-weighted density `w` in the overlap term and the
/// Pauli-channel XC potentials `v` (`4 x points`): the charge through the shared contraction
/// (half to each spin, `v_0` in both), the magnetization channels and the spin-orbit term.
fn fixed(op: &Spinor, x: &Matrix, w: Option<&Matrix>, v: &[f64]) -> Result<Vec<f64>> {
    let integrals = op.integrals;
    let ctx = &integrals.context;
    let (m, n) = (op.m, integrals.grid.index.points());
    let setup = integrals
        .paw_setup
        .as_ref()
        .ok_or_else(|| invalid("excited-state forces need a PAW setup"))?;
    let charge = half_charge(x, m);
    let mut valence = ctx.allocate::<f64>(2 * n)?;
    density(
        op.samples.as_slice(),
        n,
        m,
        &charge,
        &mut valence.as_mut_slice()[..n],
    )?;
    let copy = valence.as_slice()[..n].to_vec();
    valence.as_mut_slice()[n..].copy_from_slice(&copy);
    let l = &integrals.lowdin;
    let weighted = w.map(|w| half_charge(w, m));
    let mut matrices = vec![(&charge, l, 0, false), (&charge, l, 1, false)];
    if let Some(w) = &weighted {
        matrices.push((w, l, 0, true));
        matrices.push((w, l, 1, true));
    }
    let both = [&v[..n], &v[..n]].concat();
    let mut out = frozen_contraction(integrals, setup, valence, &matrices, &both)?
        .total
        .as_slice()
        .to_vec();
    let potentials = spinor_dft::ChannelPotentials {
        v: ctx.copy_slice(v)?,
        meta: None,
    };
    let magnetization = spinor_dft::magnetization_gradient(integrals, x, &potentials)?;
    for (t, g) in out.iter_mut().zip(&magnetization) {
        *t += g;
    }
    if op.spin_orbit {
        let ao = l.nrows();
        let li = l.map(|y| C64::new(y, 0.));
        let mut lift = Matrix::zeros(2 * ao, 2 * m);
        lift.view_mut((0, 0), (ao, m)).copy_from(&li);
        lift.view_mut((ao, m), (ao, m)).copy_from(&li);
        let x_ao = &lift * x * lift.transpose();
        let so = setup.spin_orbit_gradient(ctx, integrals.geometry.positions(), &x_ao)?;
        for (t, g) in out.iter_mut().zip(so.iter().flatten()) {
            *t += g;
        }
    }
    Ok(out)
}

/// The gradient of state `state` of `found` (the spinor excitations of `reference`), with the
/// identities a gate reads. Degeneracy is not checked here (the caller's [`super::super::pick`]
/// averages a degenerate group over its members); a degenerate state's gradient is that of the
/// member given.
pub(in crate::molecular::excitations) fn state_gradient(
    integrals: &MolecularIntegrals,
    reference: &MeanField,
    found: &Excitations,
    state: usize,
) -> Result<(SpinorGradient, Diagnostics)> {
    let spinors = found
        .spinor
        .as_ref()
        .ok_or_else(|| invalid("a spinor excited-state gradient needs the spinor excitations"))?;
    let name = reference
        .functional
        .as_deref()
        .ok_or_else(|| invalid("a spinor excited state needs a Kohn-Sham reference"))?;
    if state >= spinors.plus.len() {
        return Err(invalid(format!(
            "state {state} is beyond the {} spinor states solved",
            spinors.plus.len()
        )));
    }
    if integrals
        .corrections
        .as_ref()
        .is_some_and(|c| c.hubbard.is_some())
    {
        return Err(crate::Error::Unsupported(
            "analytic excited-state forces of a spinor DFT+U reference are not yet implemented"
                .into(),
        ));
    }
    integrals.context.run(|| {
        let op = Spinor::new(integrals, reference)?;
        if op.spin_orbit && !op.kernel.unpolarized() {
            return Err(crate::Error::Unsupported(
                "excited-state forces of a magnetized spinor reference with spin-orbit coupling are \
                 not implemented: a Kramers doublet's moment direction is barely fixed by the \
                 spin-orbit term, and the references measured are saddles of it (A + B not \
                 positive); without spin-orbit coupling they are available"
                    .into(),
            ));
        }
        if !op.kernel.unpolarized() && name != "lda" {
            return Err(crate::Error::Unsupported(
                "excited-state forces of a magnetized spinor reference with a gradient functional \
                 are not implemented: measured on H2+ and H3 with PBE, the gradient does not follow \
                 its energy (the noncollinear GGA's third derivative); use xc='lda'"
                    .into(),
            ));
        }
        let ctx = &integrals.context;
        let grid = &integrals.grid;
        let (m, o, v) = (op.m, op.o, op.v);
        let mut lowdin = Matrix::zeros(2 * m, 2 * m);
        noncollinear::assemble_into(&mut lowdin, &[integrals.lowdin.map(|x| C64::new(x, 0.))], m);
        let gap = (lowdin * &op.orbitals - &spinors.atomic_orbitals)
            .iter()
            .fold(0f64, |a, z| a.max(z.norm()));
        if gap > 1e-10 {
            return Err(invalid(
                "the excitation and the gradient see different spinors",
            ));
        }
        let (p, q) = (&spinors.plus[state], &spinors.minus[state]);
        let (co, cv) = (op.orbitals.columns(0, o), op.orbitals.columns(o, v));
        let d = co * co.adjoint();
        let energies = Matrix::from_diagonal(&nalgebra::DVector::from_iterator(
            2 * m,
            op.energies.iter().map(|&e| C64::new(e, 0.)),
        ));
        let fock = &op.orbitals * energies * op.orbitals.adjoint();
        let half = C64::new(0.5, 0.);
        let directions = [op.change(p), op.change(&(q * C64::i()))];
        let delta = (cv
            * (p.transpose() * p.conjugate() + q.transpose() * q.conjugate())
            * cv.adjoint()
            - co * (p.conjugate() * p.transpose() + q.conjugate() * q.transpose()) * co.adjoint())
            * half;
        let (f_delta, dv_delta) = op.first_order_fock(&delta)?;
        let mut first = vec![];
        let mut second = vec![];
        let mut g_d = f_delta;
        let mut a0 = &delta * &fock;
        let mut omega = (&fock * &delta).trace().re;
        for t in &directions {
            let (f, dv) = op.first_order_fock(t)?;
            let sec = op
                .kernel
                .second_change(ctx, grid, op.channel_densities(t)?.as_slice())?;
            // `F2 = d^2 F / dt^2`, twice the second change's matrices.
            g_d += op.local_fock(sec.as_slice())? * C64::new(0.5, 0.);
            omega += 0.25 * (&f * t).trace().re;
            a0 += t * &f * half;
            first.push(dv);
            second.push(sec);
        }
        a0 += &d * &g_d;
        let r = (cv.adjoint() * (&a0 - a0.adjoint()) * co).transpose();
        let mut rhs = op.column(&r);
        let size = rhs.iter().map(|x| x * x).sum::<f64>().sqrt();
        let (mut mode_share, mut mode_action) = (0f64, 0f64);
        for n in &op.rotations {
            let c: f64 = n.iter().zip(&rhs).map(|(a, b)| a * b).sum();
            mode_share = mode_share.max(c.abs() / size.max(f64::MIN_POSITIVE));
            let action = op.bare(&DMatrix::from_column_slice(n.len(), 1, n), true, false)?;
            mode_action = mode_action.max(action.norm());
        }
        project(&op.rotations, &mut rhs);
        let (column, z_residual, z_iterations) = z_vector(&op, &rhs, 1e-11, 500)?;
        let z = op.change(&op.amplitudes(&column));
        let (f_z, dv_z) = op.first_order_fock(&z)?;
        let a = a0 + &d * f_z + &z * &fock;
        let asymmetry = (&a - a.adjoint()).norm() / a.norm();
        let w = (&a + a.adjoint()) * half;
        let relaxed = &delta + &z;
        // The magnetization `P` puts on the grid (the real parts of its spin channels; their
        // imaginary parts are time-even spin currents, which spin-orbit coupling makes).
        let real = |c: usize| noncollinear::component(&relaxed, m, c).map(|z| z.re).norm();
        let odd = (1..4).map(real).fold(0f64, f64::max) / real(0);
        // The reference's Pauli-channel XC potentials.
        let spin_orbit = spinor_dft::spin_orbit_term(integrals)?;
        let shared = spinor_dft::Shared::new(integrals, name)?;
        let (evaluation, _, potentials) =
            spinor_dft::spinor_parts(integrals, &d, name, spin_orbit.as_ref(), &shared)?;
        let v0 = potentials.v.as_slice();
        let combine = |terms: &[(f64, &[f64])]| -> Vec<f64> {
            (0..v0.len())
                .map(|k| v0[k] + terms.iter().map(|(c, x)| c * x[k]).sum::<f64>())
                .collect()
        };
        let shifted = |x: &Matrix, sign: f64| &d + x * C64::new(sign, 0.);
        let (dp, dz) = (dv_delta.as_slice(), dv_z.as_slice());
        let plus = fixed(
            &op,
            &shifted(&relaxed, 1.),
            None,
            &combine(&[(1., dp), (1., dz)]),
        )?;
        let minus = fixed(
            &op,
            &shifted(&relaxed, -1.),
            None,
            &combine(&[(-1., dp), (-1., dz)]),
        )?;
        let middle = fixed(&op, &d, None, v0)?;
        let size = middle.len();
        let mut transition = vec![0.; size];
        for (j, t) in directions.iter().enumerate() {
            let (dt, sec) = (first[j].as_slice(), second[j].as_slice());
            let up = fixed(&op, &shifted(t, 1.), None, &combine(&[(1., dt), (1., sec)]))?;
            let down = fixed(
                &op,
                &shifted(t, -1.),
                None,
                &combine(&[(-1., dt), (1., sec)]),
            )?;
            for k in 0..size {
                transition[k] += 0.25 * (up[k] + down[k] - 2. * middle[k]);
            }
        }
        let weighted = fixed(&op, &d, Some(&w), v0)?;
        let mut gradient = integrals.gradient(reference)?;
        // The contraction at the reference with its own W is the ground-state gradient.
        let ground_w = &d * (&evaluation.fock + evaluation.fock.adjoint()) * &d * half;
        let seam = fixed(&op, &d, Some(&ground_w), v0)?;
        let ground = gradient.total.as_slice().to_vec();
        let seam = seam
            .iter()
            .zip(&ground)
            .map(|(a, b)| (a - b).abs())
            .fold(0f64, f64::max);
        let parts: Vec<(&'static str, Vec<f64>)> = vec![
            ("ground", ground),
            (
                "relaxed",
                (0..size).map(|k| 0.5 * (plus[k] - minus[k])).collect(),
            ),
            ("transition", transition),
            (
                "overlap",
                (0..size).map(|k| weighted[k] - middle[k]).collect(),
            ),
        ];
        let total: Vec<f64> = (0..size)
            .map(|k| parts.iter().map(|(_, x)| x[k]).sum())
            .collect();
        if total.iter().any(|x| !x.is_finite()) {
            return Err(invalid("nonfinite excited-state gradient"));
        }
        gradient.total.as_mut_slice().copy_from_slice(&total);
        gradient.terms.clear();
        let charge = |x: &Matrix| noncollinear::component(x, m, 0).map(|z| z.re);
        Ok((
            SpinorGradient {
                gradient,
                parts,
                relaxed: charge(&relaxed),
                transition: [charge(&directions[0]), charge(&directions[1])],
                energy_weighted: charge(&w),
                z_residual,
                z_iterations,
            },
            Diagnostics {
                omega,
                asymmetry,
                odd,
                ground: seam,
                modes: op.rotations.len(),
                mode_action,
                mode_share,
            },
        ))
    })
}
