//! Real-time propagation of molecular references (`molecular::realtime`), checked against the
//! linear response of the same builder (`TDDFT_PLAN.md`, sections 13.4 and 13.8):
//!
//! - V9: for a weak kick, the odd part `[mu(kappa) - mu(-kappa)] / 2 kappa` of the induced
//!   dipole equals `alpha(t) = sum_n (f_n / omega_n) sin(omega_n t)` over every Casida root,
//!   here `4 d^T S sin(sqrt(M) t) M^{-1/2} S d` with `S = (A - B)^{1/2}`, `M = S (A + B) S` from
//!   the dense T1 matrices, to `O(dt^2)`; the spin kick against the triplet matrices;
//! - R2: the energy after a kick is conserved in both gauges;
//! - R5: the initial slope of the dipole is the finite-basis f-sum `4 d^T (A - B) d`;
//! - R16(a): the r2SCAN current-density term of `A - B` is the derivative of the propagated Fock
//!   matrix (with `tau~`) along an imaginary density change, and V9 holds for r2SCAN;
//! - R16(b), measured: after a length kick in the finite basis plain `tau` and `tau~` shift at the
//!   kick's order, the basis's defect far above the partial core's share
//!   `(1 - n_v / n) n_v kappa^2 / 2` that `tau~` alone should keep (not resolved here);
//! - the momentum observable is `dE / dA` at fixed `D`, with the exchange-correlation current of
//!   r2SCAN's `tau~` with a partial core;
//! - refusals of the references the propagation does not take.
mod paw_molecular_support;

use mandacaru_core::{
    math::{linalg::real_eigen, propagator::Scheme},
    molecular::{
        excitations::{Coupling, Response, Spin, Unrestricted},
        multipoles::{gradient_integrals, position_integrals},
        realtime::{fock_matrices, propagate, Gauge, Kick, RealTime, Request},
        MeanField, MolecularIntegrals, ScfOptions,
    },
    Context,
};
use nalgebra::{DMatrix, DVector};
use paw_molecular_support::{analytic_shells, dataset, molecule, species, H2};

fn options() -> ScfOptions {
    ScfOptions {
        tolerance: 1e-12,
        max_iterations: 400,
        ..Default::default()
    }
}
/// H2 on the synthetic PAW species with an s and a p shell.
fn paw_hydrogen(ctx: &Context) -> MolecularIntegrals {
    let mut arrays = mandacaru_core::paw::molecular::Arrays::new();
    let species = species(ctx, &mut arrays);
    molecule(
        ctx,
        &[1, 1],
        &H2,
        paw_molecular_support::CELL,
        0.5,
        (vec![species], vec![0, 0], arrays),
        "raw",
        &analytic_shells(&[0, 1]),
    )
}
fn reference(integrals: &MolecularIntegrals, method: &str) -> MeanField {
    let state = if method == "rhf" {
        integrals.mean_field("rhf", [1, 1], &options())
    } else {
        integrals.kohn_sham(method, [1, 1], false, &options())
    }
    .unwrap();
    assert!(state.converged);
    state
}

/// The Casida lines of dense `A + B`, `A - B` (`S = (A - B)^{1/2}`, `M = S (A + B) S =
/// sum_n omega_n^2 u_n u_n^T`) along the occupied-empty blocks `d` of the position and `g` of the
/// gradient: `(omega_n, factor (d^T S u_n)^2, factor (g^T S^-1 u_n)^2)`, `factor` 4 for a
/// spin-adapted (restricted) operator and 2 for an unrestricted one. The length response is
/// `alpha(t) = sum_n w_n sin(omega_n t) / omega_n`, the velocity one (the kinetic momentum after
/// a vector-potential step `A_0`) `pi(t) / A_0 = N - sum_n v_n (1 - cos(omega_n t))`.
struct Lines(Vec<(f64, f64, f64)>);
impl Lines {
    fn new(
        sum: &DMatrix<f64>,
        difference: &DMatrix<f64>,
        d: &DVector<f64>,
        g: &DVector<f64>,
        factor: f64,
    ) -> Self {
        let n = sum.nrows();
        let (values, vectors) = real_eigen((difference + difference.transpose()) * 0.5).unwrap();
        let power = |p: f64| {
            &vectors
                * DMatrix::from_diagonal(&DVector::from_iterator(
                    n,
                    values.iter().map(|v| v.powf(p)),
                ))
                * vectors.transpose()
        };
        let (root, inverse_root) = (power(0.5), power(-0.5));
        let m = &root * sum * &root;
        let (squares, modes) = real_eigen((&m + m.transpose()) * 0.5).unwrap();
        let length = modes.transpose() * (&root * d);
        let velocity = modes.transpose() * (&inverse_root * g);
        Self(
            (0..n)
                .map(|k| {
                    (
                        squares[k].sqrt(),
                        factor * length[k] * length[k],
                        factor * velocity[k] * velocity[k],
                    )
                })
                .collect(),
        )
    }
    fn alpha(&self, t: f64) -> f64 {
        self.0.iter().map(|(w, a, _)| a * (w * t).sin() / w).sum()
    }
    fn momentum(&self, electrons: f64, t: f64) -> f64 {
        electrons
            - self
                .0
                .iter()
                .map(|(w, _, v)| v * (1. - (w * t).cos()))
                .sum::<f64>()
    }
}

/// The occupied-empty block of `operator` in the orbitals `c` (`o` occupied, `v` empty), i-major.
fn block(c: &DMatrix<f64>, o: usize, v: usize, operator: &DMatrix<f64>) -> Vec<f64> {
    let b = c.columns(0, o).transpose() * operator * c.columns(o, v);
    (0..o)
        .flat_map(|i| (0..v).map(move |a| (i, a)))
        .map(|(i, a)| b[(i, a)])
        .collect()
}

fn dense(apply: impl Fn(&DMatrix<f64>) -> DMatrix<f64>, n: usize) -> DMatrix<f64> {
    apply(&DMatrix::identity(n, n))
}

/// The lines of the spin-adapted `response` along `axis`.
fn restricted_lines(integrals: &MolecularIntegrals, response: &Response<'_>, axis: usize) -> Lines {
    let n = response.dimension();
    let (o, v, c) = (
        response.occupied(),
        response.virtual_orbitals(),
        response.orbitals(),
    );
    let r = &position_integrals(integrals, [0.; 3]).unwrap()[axis];
    let g = &gradient_integrals(integrals).unwrap()[axis];
    Lines::new(
        &dense(|b| response.sum(b).unwrap(), n),
        &dense(|b| response.difference(b).unwrap(), n),
        &DVector::from_vec(block(c, o, v, r)),
        &DVector::from_vec(block(c, o, v, g)),
        4.,
    )
}

/// The lines of the unrestricted `operator` along `axis` (alpha then beta transitions).
fn unrestricted_lines(
    integrals: &MolecularIntegrals,
    operator: &Unrestricted<'_>,
    axis: usize,
) -> Lines {
    let n = operator.dimension();
    let r = &position_integrals(integrals, [0.; 3]).unwrap()[axis];
    let g = &gradient_integrals(integrals).unwrap()[axis];
    let along = |x: &DMatrix<f64>| {
        DVector::from_vec(
            (0..2)
                .flat_map(|s| {
                    let (o, v) = operator.channel(s);
                    block(operator.orbitals(s), o, v, x)
                })
                .collect(),
        )
    };
    Lines::new(
        &dense(|b| operator.sum(b).unwrap(), n),
        &dense(|b| operator.difference(b).unwrap(), n),
        &along(r),
        &along(g),
        2.,
    )
}

fn run(
    integrals: &MolecularIntegrals,
    state: &MeanField,
    axis: usize,
    strength: f64,
    kick: Kick,
    scheme: Scheme,
    dt: f64,
    duration: f64,
) -> RealTime {
    let mut direction = [0.; 3];
    direction[axis] = 1.;
    let mut request = Request::new(direction, strength, dt, (duration / dt).round() as usize);
    request.scheme = scheme;
    request.kick = kick;
    request.corrector_tolerance = 1e-12;
    request.max_corrector_iterations = 30;
    propagate(integrals, state, &request).unwrap()
}

/// The worst difference over the trajectory between the odd kick response and `alpha(t)`,
/// relative to the largest `alpha`.
fn v9_error(
    integrals: &MolecularIntegrals,
    state: &MeanField,
    axis: usize,
    kick: Kick,
    scheme: Scheme,
    dt: f64,
    duration: f64,
    alpha: &dyn Fn(f64) -> f64,
) -> f64 {
    let kappa = 1e-3;
    let plus = run(integrals, state, axis, kappa, kick, scheme, dt, duration);
    let minus = run(integrals, state, axis, -kappa, kick, scheme, dt, duration);
    let (mut worst, mut scale) = (0f64, 0f64);
    for ((t, a), b) in plus.times.iter().zip(&plus.dipole).zip(&minus.dipole) {
        let odd = (a[axis] - b[axis]) / (2. * kappa);
        let exact = alpha(*t);
        worst = worst.max((odd - exact).abs());
        scale = scale.max(exact.abs());
    }
    worst / scale
}

/// V9 for `methods` along `axes` (density kick) and the spin kick along z for `spin_methods`.
fn check_v9(
    methods: &[&str],
    axes: &[usize],
    spin_methods: &[&str],
    scheme: Scheme,
    duration: f64,
) {
    let ctx = Context::new(Some(1), 512 << 20).unwrap();
    let integrals = paw_hydrogen(&ctx);
    let cases = methods
        .iter()
        .flat_map(|m| {
            axes.iter()
                .map(move |a| (*m, *a, Kick::Density, Spin::Singlet))
        })
        .chain(
            spin_methods
                .iter()
                .map(|m| (*m, 2, Kick::Spin, Spin::Triplet)),
        );
    for (method, axis, kick, spin) in cases {
        let state = reference(&integrals, method);
        let response = Response::new(&integrals, &state, spin, Coupling::Adiabatic).unwrap();
        let lines = restricted_lines(&integrals, &response, axis);
        let alpha = |t: f64| lines.alpha(t);
        let coarse = v9_error(
            &integrals, &state, axis, kick, scheme, 0.1, duration, &alpha,
        );
        let fine = v9_error(
            &integrals, &state, axis, kick, scheme, 0.05, duration, &alpha,
        );
        eprintln!("{method} {kick:?} {scheme:?} axis {axis}: V9 errors {coarse:.3e} (dt 0.1), {fine:.3e} (dt 0.05)");
        assert!(fine < 2e-3, "{method} {kick:?} axis {axis}: {fine:.3e}");
        assert!(
            coarse / fine > 3.,
            "{method} {kick:?} axis {axis}: not second order"
        );
    }
}

#[test]
fn the_weak_kick_response_is_the_casida_response_function() {
    // AETRS (one build per step) keeps the fast tier short; the slow tier runs ETRS.
    check_v9(&["lda"], &[2], &["lda"], Scheme::Aetrs, 4.);
}

#[test]
#[ignore = "slow: 157 s; V9 for lda, pbe and rhf on two axes and the rhf spin kick; run: cargo test --profile release-test -p mandacaru-core --test molecular_realtime -- --ignored"]
fn the_weak_kick_response_is_the_casida_response_function_for_every_builder() {
    check_v9(
        &["lda", "pbe", "rhf"],
        &[0, 2],
        &["lda", "rhf"],
        Scheme::Etrs,
        10.,
    );
}

#[test]
fn the_r2scan_weak_kick_response_is_the_amended_casida_response_function() {
    // V9 for r2SCAN against T3's kernel with the current-density term of tau~ (owner decision
    // Q19a): the propagation sees tau~, so the two agree only with the amendment.
    check_v9(&["r2scan"], &[2], &["r2scan"], Scheme::Aetrs, 4.);
}

/// R16(a), V1b for r2SCAN: along an imaginary antisymmetric change i A of the density the
/// imaginary part of the propagated Fock difference, projected on the occupied-empty block, is the
/// antisymmetric kernel -2 int (v_tau / n) j^ia . j^jb (plain tau gives zero there). Returns the
/// largest kernel entry.
fn check_current_term(integrals: &MolecularIntegrals, state: &MeanField, label: &str) -> f64 {
    use mandacaru_core::quantum::C64;
    let mut largest = 0f64;
    for spin in [Spin::Singlet, Spin::Triplet] {
        let response = Response::new(integrals, state, spin, Coupling::Adiabatic).unwrap();
        let (o, v) = (response.occupied(), response.virtual_orbitals());
        let c = response.orbitals();
        let y: Vec<f64> = (0..o * v)
            .map(|i| ((i as f64 + 1.) * 0.731).sin() / (1. + i as f64).sqrt())
            .collect();
        let ym = response.matrix(&y);
        let a = c.columns(0, o) * &ym * c.columns(o, v).transpose();
        let change = (&a - a.transpose()).map(|x| C64::new(0., x));
        let d = &state.densities[0];
        let sign = if spin == Spin::Singlet { 1. } else { -1. };
        let fock = |h: f64| {
            let up = d + &change * C64::new(h, 0.);
            let down = d + &change * C64::new(sign * h, 0.);
            fock_matrices(integrals, state, &[up, down], None)
                .unwrap()
                .0[0]
                .clone()
        };
        let imaginary = ((fock(1e-4) - fock(-1e-4)) / C64::new(2e-4, 0.)).map(|z| z.im);
        let projected = c.columns(0, o).transpose() * imaginary * c.columns(o, v);
        let kernel = response
            .antisymmetric_kernel(&DMatrix::from_column_slice(o * v, 1, &y))
            .unwrap();
        let kernel = response.matrix(kernel.column(0).as_slice());
        let error = (&projected - &kernel).amax();
        eprintln!(
            "{label} {spin:?}: current kernel {:.4e}, error {error:.2e}",
            kernel.amax()
        );
        assert!(
            kernel.amax() > 1e-6,
            "{label} {spin:?}: the current kernel vanishes"
        );
        assert!(error < 1e-6 * kernel.amax(), "{label} {spin:?}: {error:e}");
        largest = largest.max(kernel.amax());
    }
    largest
}

#[test]
fn the_r2scan_current_term_is_the_derivative_of_the_propagated_fock() {
    let ctx = Context::new(Some(1), 512 << 20).unwrap();
    let integrals = paw_hydrogen(&ctx);
    check_current_term(&integrals, &reference(&integrals, "r2scan"), "H2");
}

#[test]
#[ignore = "slow: 14 s; V1b for r2SCAN on library water, whose datasets carry a partial core (needs MANDACARU_PAW_PATH)"]
fn the_r2scan_current_term_with_a_partial_core_is_the_derivative_of_the_propagated_fock() {
    // The weights v_tau / n see the total density (the smooth core included), which synthetic
    // hydrogen has none of.
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    let integrals = library_water(&ctx);
    let state = integrals
        .kohn_sham("r2scan", [5, 5], false, &options())
        .unwrap();
    assert!(state.converged);
    check_current_term(&integrals, &state, "water");
}

#[test]
fn the_initial_slope_of_the_dipole_is_the_finite_basis_f_sum() {
    // R5: d(delta mu / kappa)/dt at 0+ is sum_n f_n = 4 d^T (A - B) d (V4's finite-basis form,
    // not N). The odd response over the first steps at dt = 0.005 is alpha(t) ~ s t - c t^3.
    let ctx = Context::new(Some(1), 512 << 20).unwrap();
    let integrals = paw_hydrogen(&ctx);
    for method in ["lda", "rhf", "r2scan"] {
        let state = reference(&integrals, method);
        let response =
            Response::new(&integrals, &state, Spin::Singlet, Coupling::Adiabatic).unwrap();
        let n = response.dimension();
        let difference = response.difference(&DMatrix::identity(n, n)).unwrap();
        let (o, v) = (response.occupied(), response.virtual_orbitals());
        let c = response.orbitals();
        let r = &position_integrals(&integrals, [0.; 3]).unwrap()[2];
        let block = c.columns(0, o).transpose() * r * c.columns(o, v);
        let d = DVector::from_iterator(
            o * v,
            (0..o)
                .flat_map(|i| (0..v).map(move |a| (i, a)))
                .map(|(i, a)| block[(i, a)]),
        );
        let sum_rule = 4. * d.dot(&(&difference * &d));
        let kappa = 1e-3;
        let dt = 0.005;
        let plus = run(
            &integrals,
            &state,
            2,
            kappa,
            Kick::Density,
            Scheme::Aetrs,
            dt,
            2. * dt,
        );
        let minus = run(
            &integrals,
            &state,
            2,
            -kappa,
            Kick::Density,
            Scheme::Aetrs,
            dt,
            2. * dt,
        );
        let odd = |k: usize| (plus.dipole[k][2] - minus.dipole[k][2]) / (2. * kappa);
        // Remove the cubic term from the samples at dt and 2 dt.
        let slope = (8. * odd(1) - odd(2)) / (6. * dt);
        eprintln!("{method}: slope {slope:.8} against 4 d^T (A - B) d = {sum_rule:.8}");
        assert!(
            (slope / sum_rule - 1.).abs() < 1e-4,
            "{method}: {slope} vs {sum_rule}"
        );
    }
}

#[test]
fn the_energy_is_conserved_after_a_kick_in_both_gauges() {
    let ctx = Context::new(Some(1), 512 << 20).unwrap();
    let integrals = paw_hydrogen(&ctx);
    for (method, gauge) in [
        ("lda", Gauge::Length),
        ("lda", Gauge::Velocity),
        // tau~ with its density term (vera fix 3): without it the finite kick drifts.
        ("r2scan", Gauge::Length),
    ] {
        let state = reference(&integrals, method);
        let mut request = Request::new([0., 0., 1.], 1e-2, 0.05, 80);
        request.gauge = gauge;
        request.corrector_tolerance = 1e-12;
        request.max_corrector_iterations = 30;
        let rt = propagate(&integrals, &state, &request).unwrap();
        eprintln!(
            "{method} {gauge:?}: kick energy {:.3e}, drift {:.3e} kick energies, W {:.3}, builds {}",
            rt.kick_energy, rt.energy_drift, rt.width, rt.builds
        );
        assert!(rt.kick_energy > 0.);
        assert!(rt.energy_drift < 1e-3, "{gauge:?}: {:.3e}", rt.energy_drift);
        assert!(rt.norm_error.iter().all(|e| *e < 1e-12));
    }
}

#[test]
fn the_corrector_tolerance_scales_with_the_kick() {
    // Q18c: the requested tolerance holds at the default kick (0.002 / Angstrom) and scales with
    // the strength, because the induced density does.
    use mandacaru_core::molecular::realtime::{default_strength, effective_corrector_tolerance};
    let at_default = Request::new([1., 0., 0.], default_strength(), 0.1, 1);
    assert_eq!(effective_corrector_tolerance(&at_default), 1e-6);
    let weak = Request::new([1., 0., 0.], -1e-4, 0.1, 1);
    let scaled = effective_corrector_tolerance(&weak);
    assert!((scaled / (1e-6 * 1e-4 / default_strength()) - 1.).abs() < 1e-15);
    assert!(scaled > 9e-8 && scaled < 1e-7, "{scaled:e}");
    let from_user = Request::from_user_units([1., 0., 0.], 0.002, 0.005, 0.01, false).unwrap();
    assert_eq!(effective_corrector_tolerance(&from_user), 1e-6);
}

#[test]
fn references_the_propagation_does_not_take_raise() {
    let ctx = Context::new(Some(1), 512 << 20).unwrap();
    let integrals = paw_hydrogen(&ctx);
    let state = reference(&integrals, "lda");
    let request = Request::new([0., 0., 1.], 1e-3, 0.1, 2);
    let mut bad = request.clone();
    bad.direction = [0.; 3];
    assert!(propagate(&integrals, &state, &bad).is_err());
    // dt W = 2.08 x 0.6 is above the AETRS bound of 1 and below the ETRS bound of 4.
    let mut bad = request.clone();
    bad.scheme = Scheme::Aetrs;
    bad.time_step = 0.6;
    let message = propagate(&integrals, &state, &bad).unwrap_err().to_string();
    assert!(
        message.contains("time_step") && message.contains("aetrs"),
        "{message}"
    );
    bad.scheme = Scheme::Etrs;
    assert!(propagate(&integrals, &state, &bad).is_ok());
    let mut bad = request.clone();
    bad.kick = Kick::Spin;
    bad.gauge = Gauge::Velocity;
    assert!(propagate(&integrals, &state, &bad).is_err());
}

/// The odd and even parts `[x(kappa) -+ x(-kappa)] / 2 kappa` of an observable over a run pair.
fn parts(
    integrals: &MolecularIntegrals,
    state: &MeanField,
    request: &Request,
    observable: impl Fn(&RealTime, usize) -> f64,
) -> (RealTime, Vec<f64>, Vec<f64>) {
    let plus = propagate(integrals, state, request).unwrap();
    let mut flipped = request.clone();
    flipped.strength = -request.strength;
    let minus = propagate(integrals, state, &flipped).unwrap();
    let kappa = request.strength;
    let odd = (0..plus.times.len())
        .map(|k| (observable(&plus, k) - observable(&minus, k)) / (2. * kappa))
        .collect();
    let even = (0..plus.times.len())
        .map(|k| (observable(&plus, k) + observable(&minus, k)) / (2. * kappa))
        .collect();
    (plus, odd, even)
}

fn tight(direction: [f64; 3], strength: f64, scheme: Scheme, dt: f64, duration: f64) -> Request {
    let mut request = Request::new(direction, strength, dt, (duration / dt).round() as usize);
    request.scheme = scheme;
    request.corrector_tolerance = 1e-12;
    request.max_corrector_iterations = 30;
    request
}

/// The worst difference between `odd` and `exact(t)` relative to the largest `exact`.
fn relative(times: &[f64], odd: &[f64], exact: impl Fn(f64) -> f64) -> f64 {
    let (mut worst, mut scale) = (0f64, 0f64);
    for (t, value) in times.iter().zip(odd) {
        worst = worst.max((value - exact(*t)).abs());
        scale = scale.max(exact(*t).abs());
    }
    worst / scale
}

#[test]
fn the_unrestricted_weak_kick_response_is_the_unrestricted_casida_response_function() {
    // V9 on an unrestricted reference: H2+ (one alpha electron, an empty beta channel).
    let ctx = Context::new(Some(1), 512 << 20).unwrap();
    let integrals = paw_hydrogen(&ctx);
    for method in ["lda", "uhf", "r2scan"] {
        let state = if method == "uhf" {
            integrals.mean_field("uhf", [1, 0], &options())
        } else {
            integrals.kohn_sham(method, [1, 0], true, &options())
        }
        .unwrap();
        assert!(state.converged && state.method == "uhf");
        let operator = Unrestricted::new(&integrals, &state, Coupling::Adiabatic).unwrap();
        let lines = unrestricted_lines(&integrals, &operator, 2);
        let errors: Vec<f64> = [0.1, 0.05]
            .iter()
            .map(|&dt| {
                let request = tight([0., 0., 1.], 1e-3, Scheme::Aetrs, dt, 4.);
                let (plus, odd, _) = parts(&integrals, &state, &request, |r, k| r.dipole[k][2]);
                relative(&plus.times, &odd, |t| lines.alpha(t))
            })
            .collect();
        eprintln!(
            "{method} H2+: V9 errors {:.3e} (dt 0.1), {:.3e} (dt 0.05)",
            errors[0], errors[1]
        );
        assert!(
            errors[1] < 2e-3 && errors[0] / errors[1] > 3.,
            "{method}: {errors:?}"
        );
    }
}

#[test]
fn the_velocity_kick_momentum_follows_the_casida_velocity_lines() {
    // R6: after a vector-potential step A_0 the kinetic momentum is
    // A_0 [N - sum_n v_n (1 - cos omega_n t)], v_n = 4 (g^T (A - B)^{-1/2} u_n)^2 (the T2 velocity
    // strengths times omega_n / 2 summed into the diamagnetic balance); its static limit
    // N - 4 g^T (A - B)^{-1} g is the finite-basis diamagnetic deficit.
    let ctx = Context::new(Some(1), 512 << 20).unwrap();
    let integrals = paw_hydrogen(&ctx);
    let state = reference(&integrals, "lda");
    let response = Response::new(&integrals, &state, Spin::Singlet, Coupling::Adiabatic).unwrap();
    let lines = restricted_lines(&integrals, &response, 2);
    let errors: Vec<f64> = [0.1, 0.05]
        .iter()
        .map(|&dt| {
            let mut request = tight([0., 0., 1.], 1e-3, Scheme::Aetrs, dt, 4.);
            request.gauge = Gauge::Velocity;
            let (plus, odd, _) = parts(&integrals, &state, &request, |r, k| r.momentum[k][2]);
            relative(&plus.times, &odd, |t| lines.momentum(2., t))
        })
        .collect();
    let deficit = 2. - lines.0.iter().map(|l| l.2).sum::<f64>();
    eprintln!(
        "velocity gauge: R6 errors {:.3e} (dt 0.1), {:.3e} (dt 0.05); N - N_dia = {deficit:.4}",
        errors[0], errors[1]
    );
    assert!(errors[1] < 2e-3 && errors[0] / errors[1] > 3., "{errors:?}");
}

/// Slater shells `r^l exp(-zeta r)` for every `l <= lmax` and every `zeta`, cut smoothly at 7 Bohr.
fn slater_shells(lmax: usize, zetas: &[f64]) -> Vec<paw_molecular_support::Shell> {
    let radii: Vec<f64> = (0..=350).map(|k| 0.02 * k as f64).collect();
    let mut out = Vec::new();
    for l in 0..=lmax {
        for &zeta in zetas {
            let values = radii
                .iter()
                .map(|&r| {
                    let cut = if r >= 7. {
                        0.
                    } else {
                        (1. - (r / 7.).powi(2)).powi(3)
                    };
                    r.powi(l as i32) * (-zeta * r).exp() * cut
                })
                .collect();
            out.push((l, radii.clone(), values));
        }
    }
    out
}

#[test]
fn both_gauges_approach_the_electron_count_as_the_basis_grows() {
    // Vera's completeness check (2026-11-12): on library lda-sr H2 with two Slater exponents
    // (0.6, 1.2) and l <= 1 (16 functions), the length sum N_len = 4 d^T (A - B) d and the
    // velocity sum N_dia = 4 g^T (A - B)^-1 g both lie within 3 % of N = 2 (measured 2.024 and
    // 1.964), and so does the velocity-to-length amplitude ratio of the brightest root. Neither
    // sum is bounded by N in a finite basis (the Hylleraas bound needs an exact reference; the
    // synthetic PAW H2 of R6 gives N_dia = 2.549): only their convergence is asserted.
    let ctx = Context::new(Some(1), 1 << 30).unwrap();
    let mut arrays = mandacaru_core::paw::molecular::Arrays::new();
    let hydrogen = dataset::species(&ctx, &mut arrays, &dataset::read(&ctx, "lda-sr", "H"));
    let integrals = molecule(
        &ctx,
        &[1, 1],
        &H2,
        paw_molecular_support::CELL,
        0.5,
        (vec![hydrogen], vec![0, 0], arrays),
        "raw",
        &slater_shells(1, &[0.6, 1.2]),
    );
    assert_eq!(integrals.m(), 16);
    let state = reference(&integrals, "lda");
    let response = Response::new(&integrals, &state, Spin::Singlet, Coupling::Adiabatic).unwrap();
    let lines = restricted_lines(&integrals, &response, 2);
    let length: f64 = lines.0.iter().map(|l| l.1).sum();
    let velocity: f64 = lines.0.iter().map(|l| l.2).sum();
    let bright = lines.0.iter().max_by(|a, b| a.1.total_cmp(&b.1)).unwrap();
    let ratio = (bright.2 / bright.1).sqrt();
    eprintln!(
        "library H2, 16 functions: N_len {length:.4}, N_vel {velocity:.4}, bright line at \
         {:.4} Ha with velocity/length amplitude {ratio:.4}",
        bright.0
    );
    assert!((length / 2. - 1.).abs() < 0.03, "N_len {length}");
    assert!((velocity / 2. - 1.).abs() < 0.03, "N_vel {velocity}");
    assert!((ratio - 1.).abs() < 0.03, "bright-line ratio {ratio}");
}

#[test]
fn the_kick_response_is_linear_with_a_quadratic_remainder() {
    // R4: the even part of the dipole vanishes (H2 is centrosymmetric) and the odd part departs
    // from linearity at O(kappa^2): the departure grows four times when kappa doubles.
    let ctx = Context::new(Some(1), 512 << 20).unwrap();
    let integrals = paw_hydrogen(&ctx);
    let state = reference(&integrals, "lda");
    let odd = |kappa: f64| {
        let request = tight([0., 0., 1.], kappa, Scheme::Aetrs, 0.1, 4.);
        parts(&integrals, &state, &request, |r, k| r.dipole[k][2])
    };
    let (_, small, even) = odd(1e-3);
    let (_, medium, _) = odd(2e-3);
    let (_, large, _) = odd(4e-3);
    let scale = small.iter().fold(0f64, |t, v| t.max(v.abs()));
    let even = even.iter().fold(0f64, |t, v| t.max(v.abs()));
    let departure = |a: &[f64], b: &[f64]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f64, f64::max)
    };
    let (first, second) = (departure(&medium, &small), departure(&large, &medium));
    eprintln!(
        "R4: even part {:.3e}, departures {first:.3e} and {second:.3e}",
        even / scale
    );
    assert!(even / scale < 1e-8);
    assert!(first / scale > 1e-9 && (second / first / 4. - 1.).abs() < 0.05);
}

#[test]
fn trajectories_are_bit_identical_across_thread_counts() {
    // R7.
    let mut reference_trace: Option<Vec<[f64; 3]>> = None;
    for threads in [1, 2, 4] {
        let ctx = Context::new(Some(threads), 512 << 20).unwrap();
        let integrals = paw_hydrogen(&ctx);
        let state = reference(&integrals, "lda");
        let request = Request::new([0.3, 0.2, 1.], 1e-3, 0.1, 20);
        let rt = propagate(&integrals, &state, &request).unwrap();
        match &reference_trace {
            None => reference_trace = Some(rt.dipole.clone()),
            Some(trace) => assert!(
                trace
                    .iter()
                    .zip(&rt.dipole)
                    .all(|(a, b)| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())),
                "{threads} threads differ"
            ),
        }
    }
}

#[test]
#[ignore = "slow: 121 s; V9 for HSE06; run: cargo test --profile release-test -p mandacaru-core --test molecular_realtime -- --ignored the_screened_hybrid_weak_kick_response_is_the_casida_response_function"]
fn the_screened_hybrid_weak_kick_response_is_the_casida_response_function() {
    // V9 for HSE06 at kappa = 1e-4 (Q17: the hole model's kink at t = 1 makes larger kicks
    // miss erratically), on PAW H2.
    let ctx = Context::new(Some(1), 512 << 20).unwrap();
    let integrals = paw_hydrogen(&ctx);
    let state = reference(&integrals, "hse06");
    let response = Response::new(&integrals, &state, Spin::Singlet, Coupling::Adiabatic).unwrap();
    let lines = restricted_lines(&integrals, &response, 2);
    let errors: Vec<f64> = [0.1, 0.05]
        .iter()
        .map(|&dt| {
            let request = tight([0., 0., 1.], 1e-4, Scheme::Aetrs, dt, 4.);
            let (plus, odd, _) = parts(&integrals, &state, &request, |r, k| r.dipole[k][2]);
            relative(&plus.times, &odd, |t| lines.alpha(t))
        })
        .collect();
    eprintln!(
        "hse06: V9 errors {:.3e} (dt 0.1), {:.3e} (dt 0.05)",
        errors[0], errors[1]
    );
    assert!(errors[1] < 2e-3 && errors[0] / errors[1] > 3., "{errors:?}");
}

/// Copper hydride from the `lda-sr` library with a +U shell on the copper d channel when
/// `hubbard` is `Some((form, U, J))` (eV), as `molecular_excitations.rs` builds it, and its
/// electron count.
fn copper_hydride(ctx: &Context, hubbard: Option<(&str, f64, f64)>) -> (MolecularIntegrals, usize) {
    let mut arrays = mandacaru_core::paw::molecular::Arrays::new();
    let copper = dataset::read(ctx, "lda-sr", "Cu");
    let electrons = copper.valence_charge.round() as usize + 1;
    let copper = dataset::species(ctx, &mut arrays, &copper);
    let hydrogen = dataset::species(ctx, &mut arrays, &dataset::read(ctx, "lda-sr", "H"));
    let mut extra = Vec::new();
    if let Some((form, u, j)) = hubbard {
        let request = format!(r#"{{"Cu": {{"l": "d", "U": {u}, "J": {j}, "form": "{form}"}}}}"#);
        extra.push(("hubbard", serde_json::Value::String(request)));
    }
    let integrals = paw_molecular_support::molecule_with(
        ctx,
        &[29, 1],
        &[[8.5, 8.5, 7.12], [8.5, 8.5, 9.88]],
        17.,
        0.5,
        (vec![copper, hydrogen], vec![0, 1], arrays),
        "raw",
        &analytic_shells(&[0, 1, 2]),
        &extra,
    );
    (integrals, electrons)
}

#[test]
#[ignore = "slow: 19 s; V9 and energy conservation with +U on library CuH (needs MANDACARU_PAW_PATH); run: cargo test --profile release-test -p mandacaru-core --test molecular_realtime -- --ignored the_hubbard_weak_kick_response_is_the_casida_response_function"]
fn the_hubbard_weak_kick_response_is_the_casida_response_function() {
    // V9 with +U on the complex shell occupations, both forms (U = 20 eV, J = 3 eV for
    // Liechtenstein), on library CuH; the +U part of alpha(t) is checked to be far above the
    // V9 error, and the energy after a finite kick is conserved (the complex potential is the
    // derivative of the energy).
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    let (plain, electrons) = copper_hydride(&ctx, None);
    let half = electrons / 2;
    let plain_state = plain
        .kohn_sham("lda", [half, half], false, &options())
        .unwrap();
    let plain_lines = restricted_lines(
        &plain,
        &Response::new(&plain, &plain_state, Spin::Singlet, Coupling::Adiabatic).unwrap(),
        0,
    );
    for (form, u, j) in [("dudarev", 20., 0.), ("liechtenstein", 20., 3.)] {
        let (integrals, _) = copper_hydride(&ctx, Some((form, u, j)));
        let state = integrals
            .kohn_sham("lda", [half, half], false, &options())
            .unwrap();
        assert!(state.converged);
        let response =
            Response::new(&integrals, &state, Spin::Singlet, Coupling::Adiabatic).unwrap();
        let lines = restricted_lines(&integrals, &response, 0);
        let dt = 0.05;
        let request = tight([1., 0., 0.], 1e-3, Scheme::Aetrs, dt, 2.);
        let (plus, odd, _) = parts(&integrals, &state, &request, |r, k| r.dipole[k][0]);
        let error = relative(&plus.times, &odd, |t| lines.alpha(t));
        let share = relative(
            &plus.times,
            &plus
                .times
                .iter()
                .map(|t| plain_lines.alpha(*t))
                .collect::<Vec<_>>(),
            |t| lines.alpha(t),
        );
        eprintln!(
            "{form}: V9 error {error:.3e} at dt {dt}; +U share of alpha(t) {share:.3e}; W {:.3}",
            plus.width
        );
        assert!(error < 2e-3, "{form}: {error:.3e}");
        assert!(
            share > 10. * error,
            "{form}: +U share {share:.3e} against V9 {error:.3e}"
        );
        let mut finite = tight([1., 0., 0.], 5e-2, Scheme::Etrs, dt, 2.);
        finite.max_energy_drift = None;
        let rt = propagate(&integrals, &state, &finite).unwrap();
        eprintln!(
            "{form}: finite kick, drift {:.3e} kick energies",
            rt.energy_drift
        );
        assert!(rt.energy_drift < 1e-3, "{form}: {:.3e}", rt.energy_drift);
    }
}

/// Water on the shipped `lda-sr` set, as `molecular_excitations.rs` builds it.
fn library_water(ctx: &Context) -> MolecularIntegrals {
    let mut arrays = mandacaru_core::paw::molecular::Arrays::new();
    let (o, h) = (
        dataset::read(ctx, "lda-sr", "O"),
        dataset::read(ctx, "lda-sr", "H"),
    );
    let oxygen = dataset::species(ctx, &mut arrays, &o);
    let hydrogen = dataset::species(ctx, &mut arrays, &h);
    molecule(
        ctx,
        &[8, 1, 1],
        &[[6.11, 6.07, 6.12], [7.55, 6.04, 5.07], [4.57, 5.95, 5.03]],
        12.,
        0.5,
        (vec![oxygen, hydrogen], vec![0, 1, 1], arrays),
        "raw",
        &analytic_shells(&[0, 1, 2]),
    )
}

/// The stability map on water (R1b): for each scheme against `dt W` over 20 a.u., the energy drift
/// (kick energies), the worst error of the odd kick response against the dense-Casida `alpha(t)`
/// (relative to its largest value; the `kappa^2` floor is about 1e-6) and the Fock builds, with
/// ETRS at three corrector tolerances so that schemes can be compared at equal builds. Printed.
#[test]
#[ignore = "slow: 187 s; the R1b stability map on library water (needs MANDACARU_PAW_PATH), printed; run: cargo test --profile release-test -p mandacaru-core --test molecular_realtime -- --ignored --nocapture stability_map_on_water"]
fn stability_map_on_water() {
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    let integrals = library_water(&ctx);
    let state = integrals
        .kohn_sham("lda", [4, 4], false, &options())
        .unwrap();
    assert!(state.converged);
    let response = Response::new(&integrals, &state, Spin::Singlet, Coupling::Adiabatic).unwrap();
    let lines = restricted_lines(&integrals, &response, 0);
    let alpha = |t: f64| lines.alpha(t);
    let probe = run(
        &integrals,
        &state,
        0,
        1e-3,
        Kick::Density,
        Scheme::Etrs,
        0.01,
        0.01,
    );
    let width = probe.width;
    let duration = 20.;
    eprintln!(
        "water lda: W = {width:.3} Ha, m = {}, aliased weight at dt 0.2: see runs",
        integrals.m()
    );
    eprintln!("dt W | scheme (corrector) | drift (kick energies) | odd-response error | builds (+ run) | s");
    let kappa = 1e-3;
    for dt_w in [0.5, 1., 2., 4.] {
        for (scheme, tolerance) in [
            (Scheme::Aetrs, 1e-10),
            (Scheme::Midpoint, 1e-10),
            (Scheme::Etrs, 1e-10),
            (Scheme::Etrs, 1e-8),
            (Scheme::Etrs, 1e-6),
        ] {
            let started = std::time::Instant::now();
            let dt = dt_w / width;
            let steps = (duration / dt).round() as usize;
            let go = |strength: f64| {
                let mut request = Request::new([1., 0., 0.], strength, dt, steps);
                request.scheme = scheme;
                request.max_energy_drift = None;
                request.corrector_tolerance = tolerance;
                request.max_corrector_iterations = 30;
                propagate(&integrals, &state, &request)
            };
            match (go(kappa), go(-kappa)) {
                (Ok(plus), Ok(minus)) => {
                    let (mut worst, mut scale) = (0f64, 0f64);
                    for ((t, a), b) in plus.times.iter().zip(&plus.dipole).zip(&minus.dipole) {
                        let exact = alpha(*t);
                        worst = worst.max(((a[0] - b[0]) / (2. * kappa) - exact).abs());
                        scale = scale.max(exact.abs());
                    }
                    eprintln!(
                        "{dt_w:4.2} | {:8} ({tolerance:.0e}) | {:9.2e} | {:9.2e} | {:5} | {:.1}  aliased {:.2e}",
                        scheme.name(),
                        plus.energy_drift,
                        worst / scale,
                        plus.builds,
                        started.elapsed().as_secs_f64(),
                        plus.aliased_weight
                    );
                }
                (Err(error), _) | (_, Err(error)) => {
                    eprintln!(
                        "{dt_w:4.2} | {:8} ({tolerance:.0e}) | failed: {error}",
                        scheme.name()
                    )
                }
            }
        }
    }
}

/// R16(b) on library water (its datasets carry a partial core): after the propagation's length
/// kick `exp(i kappa X)` of the occupied orbitals, plain `tau` shifts by `kappa^2 n_v / 2` and the
/// gauge-invariant `tau~ = tau - |j|^2 / (2 n)` only by the partial core's share
/// `(1 - n_v / n) n_v kappa^2 / 2` (13.6.1, item 3), each up to the finite basis's defect of the
/// kick (`molecular::dft::kinetic_fields`). Measured at two kicks, as integrals over the points
/// where the valence density exceeds 1e-4. In this basis the kick's defect (`j` against
/// `kappa n_v`, about 30 %) is far above the core term (7.3e-6 of `int kappa^2 n_v / 2`): `tau~`
/// still shifts by about 0.6 of the boost against plain `tau`'s 0.72. Asserted: every defect is
/// of the kick's order (`kappa^2`, `kappa` for `j`: the same relative miss at both kicks), and
/// the core's part of `tau~`, `|j|^2 / 2 (1 / n_v - 1 / n)`, is the core term to within the
/// current's defect, `(1 + d_j)^2 - 1`. The core term itself is resolved by subtracting the
/// defect measured without the core: on the same kicked state, `tau~` less the core-free
/// `tau - |j|^2 / (2 n_v)` is the core's share `(1 - n_v / n) |j|^2 / (2 n_v)` at the measured
/// current, with `n = n_v + n_c / 2` from the integrals' own partial core (not from the
/// accessor), so the kick's defect cancels. The accessor forms `n_s` with the builder's code
/// (`add_partial_core`, `subtract_current`).
#[test]
fn after_a_length_kick_the_kinetic_densities_shift_at_the_order_of_the_kick() {
    use mandacaru_core::molecular::dft::kinetic_fields;
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    let integrals = library_water(&ctx);
    let state = integrals
        .kohn_sham("lda", [5, 5], false, &options())
        .unwrap();
    assert!(state.converged);
    let m = integrals.m();
    let n = integrals.grid.index.points();
    let c64 = |x: f64| nalgebra::Complex::new(x, 0.);
    let (occupation, vectors) = real_eigen(state.densities[0].map(|z| z.re)).unwrap();
    let occupied: Vec<usize> = (0..m).filter(|&i| occupation[i] > 0.5).collect();
    assert_eq!(occupied.len(), 5);
    let orbitals = DMatrix::from_fn(m, 5, |a, i| c64(vectors[(a, occupied[i])]));
    let x = position_integrals(&integrals, [0.; 3]).unwrap()[0].clone();
    let (positions, frame) = real_eigen(x).unwrap();
    let ground = kinetic_fields(&integrals, &[&orbitals * orbitals.adjoint()], None).unwrap();
    let partial = integrals
        .corrections
        .as_ref()
        .and_then(|c| c.dft.as_ref())
        .map(|paw| paw.core.as_slice().to_vec())
        .expect("library water carries a partial core");
    let mut measured = vec![];
    for kappa in [0.05, 0.1] {
        let phases = DMatrix::from_fn(m, m, |a, b| {
            (0..m)
                .map(|l| {
                    nalgebra::Complex::from_polar(
                        frame[(a, l)] * frame[(b, l)],
                        kappa * positions[l],
                    )
                })
                .sum::<nalgebra::Complex<f64>>()
        });
        let kicked = &phases * &orbitals;
        let fields = kinetic_fields(&integrals, &[&kicked * kicked.adjoint()], None).unwrap();
        // Sums over the points, each relative to `int kappa^2 n_v / 2` (`scale`) or, for the
        // current, to `int kappa n_v`.
        let [mut scale, mut plain, mut plain_miss, mut tilde, mut tilde_miss, mut core] = [0f64; 6];
        let [mut flow, mut flow_miss, mut nlcc, mut nlcc_miss] = [0f64; 4];
        let [mut held, mut held_miss, mut held_boost] = [0f64; 3];
        let mut clamped = 0usize;
        for s in 0..2 {
            for k in 0..n {
                let valence = ground.valence.as_slice()[s * n + k];
                if valence <= 1e-4 {
                    continue;
                }
                let total = ground.density.as_slice()[s * n + k];
                let boost = 0.5 * kappa * kappa * valence;
                let share = (1. - valence / total) * boost;
                let d_plain =
                    fields.plain.as_slice()[s * n + k] - ground.plain.as_slice()[s * n + k];
                let d_tilde = fields.gauge_invariant.as_slice()[s * n + k]
                    - ground.gauge_invariant.as_slice()[s * n + k];
                let j: Vec<f64> = (0..3)
                    .map(|axis| fields.current.as_slice()[(s * 3 + axis) * n + k])
                    .collect();
                let j2: f64 = j.iter().map(|v| v * v).sum();
                scale += boost;
                plain += d_plain.abs();
                plain_miss += (d_plain - boost).abs();
                tilde += d_tilde.abs();
                tilde_miss += (d_tilde - share).abs();
                core += share;
                flow += kappa * valence;
                flow_miss += ((j[0] - kappa * valence).powi(2) + j[1] * j[1] + j[2] * j[2]).sqrt();
                // The core's part of tau~ alone: `|j|^2 / 2 (1 / n_v - 1 / n)` against the kick's
                // `(1 - n_v / n) kappa^2 n_v / 2`.
                let part = 0.5 * j2 * (1. / valence - 1. / total);
                nlcc += share;
                nlcc_miss += (part - share).abs();
                // The subtraction: tau~ with the core less the core-free tau~ of the same
                // (kicked) state, whose own valence density the basis kick has changed.
                let moved = fields.valence.as_slice()[s * n + k];
                let reference = moved + 0.5 * partial[k];
                let with_core = fields.gauge_invariant.as_slice()[s * n + k];
                let without = fields.plain.as_slice()[s * n + k] - 0.5 * j2 / moved;
                if with_core <= 0. || without <= 0. {
                    clamped += 1;
                } else {
                    let expected = (1. - moved / reference) * 0.5 * j2 / moved;
                    held += expected;
                    held_miss += ((with_core - without) - expected).abs();
                    held_boost += share;
                }
            }
        }
        eprintln!(
            "R16(b) water, kappa {kappa}: relative to int kappa^2 n_v / 2, plain tau shifts by \
             {:.3e} (misses kappa^2 n_v / 2 by {:.3e}), tau~ by {:.3e} (misses the core term by \
             {:.3e}); the core term {:.3e}; j misses kappa n_v by {:.3e} of int kappa n_v; the \
             core's part of tau~ misses the core term by {:.3e} of it",
            plain / scale,
            plain_miss / scale,
            tilde / scale,
            tilde_miss / scale,
            core / scale,
            flow_miss / flow,
            nlcc_miss / nlcc
        );
        eprintln!(
            "R16(b) water, kappa {kappa}, the subtraction: the core term at the measured current \
             {:.4e} of the boost ({:.4e} at kappa n_v), missed by {:.2e} of itself; {clamped} \
             clamped points skipped",
            held / scale,
            held_boost / scale,
            held_miss / held
        );
        // Measured 2026-11-16: 9.0e-9 of the core term (the cancellation floor of the two
        // tau~, each of the size of tau); without the core in n_s (a red check) it is 1.
        assert!(held_miss < 1e-7 * held, "{held_miss:e} of {held:e}");
        assert!(held > 1e-6 * scale, "{held:e}");
        measured.push([
            plain_miss / scale,
            tilde_miss / scale,
            flow_miss / flow,
            nlcc_miss / nlcc,
        ]);
    }
    for (a, b) in measured[0].iter().zip(&measured[1]) {
        assert!((a - b).abs() < 2e-2 * a.max(1e-12), "{measured:?}");
    }
    let [_, _, current, core] = measured[1];
    assert!(core <= (1. + current).powi(2) - 1., "{measured:?}");
}

/// The velocity gauge and the partial core (R16(b), vera's note of the sixteenth session): in a
/// uniform vector potential `A = kappa e_x` the ground state's kinetic fields are exactly the
/// length kick's in a complete basis, `j = kappa n_v` and `tau = tau_0 + kappa^2 n_v / 2`, so
/// `tau~` shifts by the partial core's share `(1 - n_v / n) n_v kappa^2 / 2`, the length kick's
/// core term at the kick's current (`after_a_length_kick_...`, whose basis kick misses that
/// current by 0.30). No basis defect enters here: the state is the basis's own. Without the
/// field in the fields (the code before) the share is lost entirely, which the red line shows.
#[test]
fn in_the_velocity_gauge_tau_tilde_keeps_the_partial_cores_share() {
    use mandacaru_core::molecular::dft::kinetic_fields;
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    let integrals = library_water(&ctx);
    let state = integrals
        .kohn_sham("lda", [5, 5], false, &options())
        .unwrap();
    assert!(state.converged);
    let n = integrals.grid.index.points();
    let d = [state.densities[0].clone()];
    let ground = kinetic_fields(&integrals, &d, None).unwrap();
    for kappa in [0.05, 0.1] {
        let moved = kinetic_fields(&integrals, &d, Some([kappa, 0., 0.])).unwrap();
        let [mut scale, mut core, mut tilde_miss, mut plain_miss, mut flow, mut flow_miss] =
            [0f64; 6];
        let mut clamped = 0usize;
        for s in 0..2 {
            for k in 0..n {
                let valence = ground.valence.as_slice()[s * n + k];
                if valence <= 1e-4 {
                    continue;
                }
                let total = ground.density.as_slice()[s * n + k];
                let boost = 0.5 * kappa * kappa * valence;
                let share = (1. - valence / total) * boost;
                let (before, after) = (
                    ground.gauge_invariant.as_slice()[s * n + k],
                    moved.gauge_invariant.as_slice()[s * n + k],
                );
                if before <= 0. {
                    clamped += 1;
                    continue;
                }
                scale += boost;
                core += share;
                tilde_miss += ((after - before) - share).abs();
                plain_miss += ((moved.plain.as_slice()[s * n + k]
                    - ground.plain.as_slice()[s * n + k])
                    - boost)
                    .abs();
                flow += kappa * valence;
                flow_miss += (0..3)
                    .map(|axis| {
                        let j = moved.current.as_slice()[(s * 3 + axis) * n + k];
                        let expected = if axis == 0 { kappa * valence } else { 0. };
                        (j - expected).abs()
                    })
                    .sum::<f64>();
            }
        }
        eprintln!(
            "velocity gauge, water, kappa {kappa}: the core term {:.4e} of int kappa^2 n_v / 2 \
             (the length kick's at kappa n_v), tau~ misses it by {:.2e} of itself, plain tau \
             misses kappa^2 n_v / 2 by {:.2e}, j misses kappa n_v by {:.2e}; {clamped} clamped \
             points skipped; red (no field in tau~): 1.00",
            core / scale,
            tilde_miss / core,
            plain_miss / scale,
            flow_miss / flow
        );
        assert!(
            tilde_miss < 1e-8 * core && core > 1e-6 * scale,
            "{tilde_miss:e} of {core:e}"
        );
        assert!(plain_miss < 1e-12 * scale && flow_miss < 1e-12 * flow);
    }
}

/// The velocity gauge's meta-GGA Fock matrix with a partial core is the derivative of its energy
/// at a nonzero vector potential, along a complex Hermitian density change (real and imaginary
/// parts, so the current moves): `dE / de = 2 Re tr(F dD)` of one restricted channel, central
/// differences at 1e-4 and 5e-5 (Richardson), on library water (r2SCAN, oxygen's partial core).
/// The field enters `tau~` through the kinetic fields, and both current terms of the Fock matrix
/// through `u = j - n_s A`.
#[test]
#[ignore = "slow: 24 s (measured 2026-11-16 at load 15; r2SCAN on library water, needs MANDACARU_PAW_PATH); run: cargo test --profile release-test -p mandacaru-core --test molecular_realtime -- --ignored the_velocity_gauge_meta_gga"]
fn the_velocity_gauge_meta_gga_fock_is_the_derivative_of_its_energy_with_a_partial_core() {
    use mandacaru_core::quantum::C64;
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    let integrals = library_water(&ctx);
    let state = integrals
        .kohn_sham("r2scan", [5, 5], false, &options())
        .unwrap();
    assert!(state.converged);
    let m = integrals.m();
    let d = state.densities[0].clone();
    let change = DMatrix::from_fn(m, m, |a, b| {
        let (i, j) = (a.min(b) as f64, a.max(b) as f64);
        let real = ((i + 1.) * 0.37 + (j + 1.) * 0.91).sin() / (1. + i + j);
        let imaginary = ((i + 2.) * 0.53 - (j + 1.) * 0.29).cos() / (1. + i + j);
        C64::new(
            real,
            if a < b {
                imaginary
            } else if a > b {
                -imaginary
            } else {
                0.
            },
        )
    });
    for field in [None, Some([0.1, 0., 0.]), Some([0.03, -0.05, 0.08])] {
        let energy = |h: f64| {
            fock_matrices(&integrals, &state, &[&d + &change * C64::new(h, 0.)], field)
                .unwrap()
                .1
        };
        let slope = |h: f64| (energy(h) - energy(-h)) / (2. * h);
        let numeric = (4. * slope(5e-5) - slope(1e-4)) / 3.;
        let (focks, _) = fock_matrices(&integrals, &state, &[d.clone()], field).unwrap();
        let analytic = 2. * (&focks[0] * &change).trace().re;
        eprintln!(
            "velocity-gauge r2SCAN water, A {field:?}: dE/de {numeric:.12e}, 2 Re tr(F dD) \
             {analytic:.12e}, relative {:.3e}",
            (numeric - analytic) / analytic.abs()
        );
        assert!(
            (numeric - analytic).abs() < 1e-7 * analytic.abs(),
            "{field:?}"
        );
    }
}

/// The momentum observable is the energy's slope in the vector potential (item 4, molecules):
/// with a partial core `tau~` depends on `A`, so `dE / dA` at fixed `D` is
/// `g tr[D (P + A)] + dE_xc / dA`, `dE_xc / dA = sum_s int v_tau,s (1 - n_v,s / n_s) j_s` with the
/// kinetic `j_s`. Library water (r2SCAN, oxygen's partial core) after a length kick
/// `exp(i kappa X)` of its density matrix (a state that carries a current), in the length gauge
/// (`A = 0`) and at `A = (0.03, -0.05, 0.08)`: `realtime::momentum` against central differences of
/// `E(A) = E_bare(D, A) + 2 (A . tr[D P] + A^2 / 2 tr D)` at 1e-3 and 5e-4 (Richardson). Red: the
/// observable without its exchange-correlation term (the code before) misses by that term, read on
/// the kick's axis.
#[test]
#[ignore = "slow: about 20 s (36 s with the velocity-gauge Fock test, measured 2026-11-16 at load 10; r2SCAN on library water, needs MANDACARU_PAW_PATH); run: cargo test --profile release-test -p mandacaru-core --test molecular_realtime -- --ignored the_momentum_is_the_energys_slope"]
fn the_momentum_is_the_energys_slope_in_the_vector_potential_with_a_partial_core() {
    use mandacaru_core::{molecular::realtime::momentum, quantum::C64};
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    let integrals = library_water(&ctx);
    let state = integrals
        .kohn_sham("r2scan", [5, 5], false, &options())
        .unwrap();
    assert!(state.converged);
    let m = integrals.m();
    let x = position_integrals(&integrals, [0.; 3]).unwrap()[0].clone();
    let (positions, frame) = real_eigen(x).unwrap();
    let kappa = 0.1;
    let phases = DMatrix::from_fn(m, m, |a, b| {
        (0..m)
            .map(|l| C64::from_polar(frame[(a, l)] * frame[(b, l)], kappa * positions[l]))
            .sum::<C64>()
    });
    let d = &phases * &state.densities[0] * phases.adjoint();
    let gradients = gradient_integrals(&integrals).unwrap();
    let orbital = |a: [f64; 3]| -> [f64; 3] {
        std::array::from_fn(|k| {
            let trace: f64 = (0..m)
                .flat_map(|i| (0..m).map(move |j| (i, j)))
                .map(|(i, j)| d[(i, j)].im * gradients[k][(j, i)])
                .sum();
            2. * (trace + a[k] * d.trace().re)
        })
    };
    let energy = |a: [f64; 3]| {
        let bare = fock_matrices(&integrals, &state, &[d.clone()], Some(a))
            .unwrap()
            .1;
        let t = orbital([0.; 3]);
        bare + (0..3).map(|k| a[k] * t[k]).sum::<f64>()
            + a.iter().map(|v| v * v).sum::<f64>() * d.trace().re
    };
    for field in [None, Some([0.03, -0.05, 0.08])] {
        let a = field.unwrap_or([0.; 3]);
        let observable = momentum(&integrals, &state, &[d.clone()], field).unwrap();
        let without = orbital(a);
        for axis in 0..3 {
            let slope = |h: f64| {
                let (mut up, mut down) = (a, a);
                up[axis] += h;
                down[axis] -= h;
                (energy(up) - energy(down)) / (2. * h)
            };
            let numeric = (4. * slope(5e-4) - slope(1e-3)) / 3.;
            let miss = (numeric - observable[axis]).abs();
            let exchange = observable[axis] - without[axis];
            eprintln!(
                "r2SCAN water, kappa {kappa}, A {field:?}, axis {axis}: dE/dA {numeric:.12e}, the \
                 momentum {:.12e} (miss {miss:.2e}), its exchange-correlation part {exchange:.6e} \
                 (red: without it the miss is {:.2e})",
                observable[axis],
                (numeric - without[axis]).abs()
            );
            assert!(
                miss < 1e-8 * observable[axis].abs().max(1.),
                "{field:?} {axis}"
            );
            // The kick's axis carries the signal (the others' currents are 1e-4 of it).
            if axis == 0 {
                assert!(exchange.abs() > 1e3 * miss, "{field:?} {axis}: no signal");
            }
        }
    }
}
