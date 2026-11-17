//! Linear-response excitations of closed-shell molecules (`molecular::excitations`), checked by
//! identities of the engine's own numbers (`dev_plan/TDDFT_PLAN.md`, section 6):
//!
//! - V1/V1b: the kernel is the derivative of the reference's Fock builder, in the symmetric
//!   (`A + B`) and the antisymmetric (`A - B`, exchange) channels;
//! - V2: the kernel is symmetric;
//! - V3: Davidson roots equal dense eigensolves of the assembled matrices;
//! - V4: with every root, `sum_n omega_n mu_n^2 = 2 d^T (A - B) d` (for a pure functional the
//!   Casida strengths sum to the bare Kohn-Sham ones; TDA does not);
//! - V5: `alpha = 2 sum mu^2 / omega = 4 d^T (A + B)^-1 d = -d^2 E / dF^2`;
//! - V6: the independent and Hartree limits, and a two-level system;
//! - V11: a request that cuts a degenerate multiplet is completed;
//! - V12: results are bit-identical across thread counts;
//! - V16(i): the Hartree kernel of the grid equals the Coulomb pair factors.
mod paw_molecular_support;

use mandacaru_core::{
    math::{linalg::real_eigen, SpatialGrid},
    molecular::{
        excitations::{
            excitations, reference_fock, Approximation, Coupling, Request, Response, Spin,
            Unrestricted,
        },
        multipoles::position_integrals,
        MeanField, MolecularIntegrals, ScfOptions,
    },
    quantum::C64,
    Context, Geometry, LengthUnit,
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
/// H2 in a minimal hydrogenic all-electron basis (5 Angstrom cell, 0.35 Angstrom grid).
fn hydrogen(ctx: &Context) -> MolecularIntegrals {
    let geometry = Geometry::new(
        vec![1, 1],
        vec![[2.5, 2.5, 2.13], [2.5, 2.5, 2.87]],
        [[5., 0., 0.], [0., 5., 0.], [0., 0., 5.]],
        [false; 3],
        LengthUnit::Angstrom,
    )
    .unwrap();
    let grid =
        SpatialGrid::from_geometry(&geometry, [0.35 / 0.529177210903; 3], false, false).unwrap();
    MolecularIntegrals::new(
        ctx,
        geometry,
        grid,
        r#"{"name":"HAO"}"#,
        "spectral",
        None,
        Default::default(),
    )
    .unwrap()
}
/// H2 on the synthetic PAW species with the radial shells `shells` (`[0]` is two-level).
fn paw_hydrogen(ctx: &Context, shells: &[usize]) -> MolecularIntegrals {
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
        &analytic_shells(shells),
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
fn request(num_states: usize, approximation: Approximation, spin: Spin) -> Request {
    let mut r = Request::new(num_states);
    r.approximation = approximation;
    r.spin = Some(spin);
    r.tolerance = 1e-9;
    r
}
/// A deterministic trial vector.
fn trial(n: usize, seed: f64) -> DVector<f64> {
    DVector::from_fn(n, |i, _| {
        ((i as f64 + 1.) * seed).sin() / (1. + i as f64).sqrt()
    })
}
fn dense(response: &Response<'_>) -> (DMatrix<f64>, DMatrix<f64>) {
    let n = response.dimension();
    let identity = DMatrix::identity(n, n);
    let sum = response.sum(&identity).unwrap();
    let difference = response.difference(&identity).unwrap();
    (sum, difference)
}
fn symmetric_root(m: &DMatrix<f64>) -> DMatrix<f64> {
    let (values, vectors) = real_eigen(m.clone()).unwrap();
    let d = DMatrix::from_diagonal(&DVector::from_iterator(
        values.len(),
        values.iter().map(|v| v.sqrt()),
    ));
    &vectors * d * vectors.transpose()
}

/// V1: `<i| F1 |a>` of a symmetric density change, by a Richardson central difference of
/// `reference_fock`, against `Response::symmetric_kernel`.
fn check_symmetric_channel(
    integrals: &MolecularIntegrals,
    state: &MeanField,
    spin: Spin,
    label: &str,
) {
    let response = Response::new(integrals, state, spin, Coupling::Adiabatic).unwrap();
    let (o, v) = (response.occupied(), response.virtual_orbitals());
    let c = response.orbitals();
    let y = trial(o * v, 0.731);
    let ym = response.matrix(y.as_slice());
    let change = c.columns(0, o) * &ym * c.columns(o, v).transpose();
    let change = (&change + change.transpose()).map(|x| C64::new(x, 0.));
    let d = &state.densities[0];
    let fock = |h: f64| -> DMatrix<f64> {
        let step = &change * C64::new(h, 0.);
        let f = match spin {
            Spin::Singlet => reference_fock(integrals, state, &[d + &step]).unwrap(),
            _ => reference_fock(integrals, state, &[d + &step, d - &step]).unwrap(),
        };
        f[0].map(|z| z.re)
    };
    let difference = |h: f64| (fock(h) - fock(-h)) / (2. * h);
    let estimate = (difference(1e-4) * 4. - difference(2e-4)) / 3.;
    let projected = c.columns(0, o).transpose() * estimate * c.columns(o, v);
    let kernel = response
        .symmetric_kernel(&DMatrix::from_column_slice(o * v, 1, y.as_slice()))
        .unwrap();
    let kernel = response.matrix(kernel.column(0).as_slice());
    let scale = kernel.amax();
    let error = (&projected - &kernel).amax();
    assert!(scale > 1e-6, "{label}: the kernel vanishes");
    assert!(
        error < 1e-6 * scale,
        "{label}: kernel misses the Fock difference by {error:e} of {scale:e}"
    );
}

#[test]
fn the_kernel_is_the_derivative_of_the_reference_fock() {
    let ctx = Context::new(Some(2), 256 << 20).unwrap();
    let integrals = hydrogen(&ctx);
    for method in ["lda", "pbe", "rhf"] {
        let state = reference(&integrals, method);
        for spin in [Spin::Singlet, Spin::Triplet] {
            check_symmetric_channel(
                &integrals,
                &state,
                spin,
                &format!("HAO H2 {method} {spin:?}"),
            );
        }
    }
    let paw = paw_hydrogen(&ctx, &[0, 1]);
    for method in ["lda", "pbe", "rhf"] {
        let state = reference(&paw, method);
        for spin in [Spin::Singlet, Spin::Triplet] {
            check_symmetric_channel(&paw, &state, spin, &format!("PAW H2 {method} {spin:?}"));
        }
    }
}

#[test]
fn the_exchange_kernel_is_the_derivative_in_the_antisymmetric_channel() {
    // V1b: an imaginary antisymmetric change i A of the density moves only the exchange term;
    // the imaginary part of the Fock difference is -K[A], the (A - B) kernel.
    let ctx = Context::new(Some(2), 256 << 20).unwrap();
    for (integrals, label) in [
        (hydrogen(&ctx), "HAO"),
        (paw_hydrogen(&ctx, &[0, 1]), "PAW"),
    ] {
        for method in ["rhf", "lda"] {
            let state = reference(&integrals, method);
            let response =
                Response::new(&integrals, &state, Spin::Singlet, Coupling::Adiabatic).unwrap();
            let (o, v) = (response.occupied(), response.virtual_orbitals());
            let c = response.orbitals();
            let y = trial(o * v, 1.37);
            let ym = response.matrix(y.as_slice());
            let a = c.columns(0, o) * &ym * c.columns(o, v).transpose();
            let change = (&a - a.transpose()).map(|x| C64::new(0., x));
            let d = &state.densities[0];
            let fock = |h: f64| {
                reference_fock(&integrals, &state, &[d + &change * C64::new(h, 0.)]).unwrap()[0]
                    .clone()
            };
            let imaginary = ((fock(1e-4) - fock(-1e-4)) / C64::new(2e-4, 0.)).map(|z| z.im);
            let projected = c.columns(0, o).transpose() * imaginary * c.columns(o, v);
            let kernel = response
                .antisymmetric_kernel(&DMatrix::from_column_slice(o * v, 1, y.as_slice()))
                .unwrap();
            let kernel = response.matrix(kernel.column(0).as_slice());
            let error = (&projected - &kernel).amax();
            if method == "rhf" {
                assert!(
                    kernel.amax() > 1e-6,
                    "{label}: the exchange kernel vanishes"
                );
                assert!(error < 1e-7 * kernel.amax(), "{label} rhf: {error:e}");
            } else {
                assert_eq!(
                    kernel.amax(),
                    0.,
                    "{label}: a pure functional has no A - B kernel"
                );
                assert!(
                    projected.amax() < 1e-10,
                    "{label} lda: {}",
                    projected.amax()
                );
            }
        }
    }
}

#[test]
fn the_kernel_is_symmetric() {
    let ctx = Context::new(Some(1), 256 << 20).unwrap();
    let integrals = paw_hydrogen(&ctx, &[0, 1]);
    for method in ["lda", "pbe", "rhf"] {
        let state = reference(&integrals, method);
        for spin in [Spin::Singlet, Spin::Triplet] {
            let response = Response::new(&integrals, &state, spin, Coupling::Adiabatic).unwrap();
            let n = response.dimension();
            let block = DMatrix::from_columns(&[trial(n, 0.41), trial(n, 2.9)]);
            let k = response.symmetric_kernel(&block).unwrap();
            let (ab, ba) = (
                block.column(0).dot(&k.column(1)),
                block.column(1).dot(&k.column(0)),
            );
            assert!(
                (ab - ba).abs() < 1e-10 * ab.abs().max(ba.abs()).max(1e-12),
                "{method} {spin:?}: {ab} {ba}"
            );
        }
    }
}

#[test]
fn davidson_roots_equal_the_dense_matrices() {
    let ctx = Context::new(Some(1), 256 << 20).unwrap();
    let integrals = paw_hydrogen(&ctx, &[0, 1]);
    for method in ["lda", "pbe", "rhf"] {
        let state = reference(&integrals, method);
        for spin in [Spin::Singlet, Spin::Triplet] {
            let response = Response::new(&integrals, &state, spin, Coupling::Adiabatic).unwrap();
            let (sum, difference) = dense(&response);
            let (tda, _) = real_eigen((&sum + &difference) * 0.5).unwrap();
            let root = symmetric_root(&difference);
            let (squares, _) = real_eigen(&root * &sum * &root).unwrap();
            let k = 3.min(response.dimension());
            let found = excitations(
                &integrals,
                &state,
                &request(k, Approximation::TammDancoff, spin),
            )
            .unwrap();
            for (r, e) in found.energies.iter().enumerate() {
                assert!((e - tda[r]).abs() < 1e-10, "{method} {spin:?} TDA root {r}");
            }
            let found =
                excitations(&integrals, &state, &request(k, Approximation::Casida, spin)).unwrap();
            for (r, e) in found.energies.iter().enumerate() {
                assert!(
                    (e - squares[r].sqrt()).abs() < 1e-10,
                    "{method} {spin:?} Casida root {r}"
                );
            }
        }
    }
}

/// The occupied-virtual dipole blocks of the reference's canonical orbitals.
fn dipole_blocks(integrals: &MolecularIntegrals, response: &Response<'_>) -> [DVector<f64>; 3] {
    let (o, v) = (response.occupied(), response.virtual_orbitals());
    let c = response.orbitals();
    position_integrals(integrals, [0.; 3]).unwrap().map(|r| {
        let block = c.columns(0, o).transpose() * r * c.columns(o, v);
        DVector::from_iterator(
            o * v,
            (0..o)
                .flat_map(|i| (0..v).map(move |a| (i, a)))
                .map(|(i, a)| block[(i, a)]),
        )
    })
}

#[test]
fn the_sum_rule_and_the_static_polarizability_hold_in_the_basis() {
    // V4 and the first two equalities of V5, with every root.
    let ctx = Context::new(Some(1), 256 << 20).unwrap();
    let integrals = paw_hydrogen(&ctx, &[0, 1]);
    for method in ["lda", "pbe", "rhf"] {
        let state = reference(&integrals, method);
        let response =
            Response::new(&integrals, &state, Spin::Singlet, Coupling::Adiabatic).unwrap();
        let n = response.dimension();
        let (sum, difference) = dense(&response);
        let d = dipole_blocks(&integrals, &response);
        let casida = excitations(
            &integrals,
            &state,
            &request(n, Approximation::Casida, Spin::Singlet),
        )
        .unwrap();
        let tda = excitations(
            &integrals,
            &state,
            &request(n, Approximation::TammDancoff, Spin::Singlet),
        )
        .unwrap();
        let inverse = sum.clone().try_inverse().unwrap();
        for axis in 0..3 {
            let weighted = |e: &mandacaru_core::molecular::excitations::Excitations| -> f64 {
                e.energies
                    .iter()
                    .zip(&e.transition_dipoles)
                    .map(|(w, mu)| w * mu[axis] * mu[axis])
                    .sum()
            };
            let exact = 2. * d[axis].dot(&(&difference * &d[axis]));
            let scale = exact.abs().max(1e-8);
            assert!(
                (weighted(&casida) - exact).abs() < 1e-9 * scale,
                "{method} axis {axis}: Casida sum rule"
            );
            if method != "rhf" {
                let bare: f64 = 2.
                    * d[axis]
                        .iter()
                        .zip(response.delta())
                        .map(|(x, w)| w * x * x)
                        .sum::<f64>();
                assert!(
                    (exact - bare).abs() < 1e-12 * scale,
                    "{method}: A - B is diagonal"
                );
            }
            if exact.abs() > 1e-6 {
                assert!(
                    (weighted(&tda) - exact).abs() > 1e-8 * scale,
                    "{method}: TDA should miss the sum rule"
                );
            }
            let alpha: f64 = casida
                .energies
                .iter()
                .zip(&casida.transition_dipoles)
                .map(|(w, mu)| 2. * mu[axis] * mu[axis] / w)
                .sum();
            let solved = 4. * d[axis].dot(&(&inverse * &d[axis]));
            assert!(
                (alpha - solved).abs() < 1e-9 * solved.abs().max(1e-8),
                "{method} axis {axis}: polarizability"
            );
        }
    }
}

#[test]
fn the_static_polarizability_is_the_finite_field_curvature() {
    // V5, third equality: -d^2 E / dF^2 of the reference's own SCF under a field term F z in
    // the one-body operator (a test-only perturbation; there is no user option).
    let ctx = Context::new(Some(2), 256 << 20).unwrap();
    for method in ["lda", "pbe", "rhf"] {
        let mut integrals = paw_hydrogen(&ctx, &[0, 1]);
        let state = reference(&integrals, method);
        let n = Response::new(&integrals, &state, Spin::Singlet, Coupling::Adiabatic)
            .unwrap()
            .dimension();
        let casida = excitations(
            &integrals,
            &state,
            &request(n, Approximation::Casida, Spin::Singlet),
        )
        .unwrap();
        let axis = 2;
        let alpha: f64 = casida
            .energies
            .iter()
            .zip(&casida.transition_dipoles)
            .map(|(w, mu)| 2. * mu[axis] * mu[axis] / w)
            .sum();
        let z = position_integrals(&integrals, [0.; 3]).unwrap()[axis].clone();
        let base: Vec<f64> = integrals.one.as_slice().to_vec();
        let m = integrals.m();
        let mut energy = |field: f64| {
            for a in 0..m {
                for b in 0..m {
                    integrals.one.as_mut_slice()[a * m + b] = base[a * m + b] + field * z[(a, b)];
                }
            }
            reference(&integrals, method).electronic_energy
        };
        let e0 = energy(0.);
        let curvature = |f: f64, energy: &mut dyn FnMut(f64) -> f64| {
            -(energy(f) + energy(-f) - 2. * e0) / (f * f)
        };
        let (coarse, fine) = (curvature(4e-3, &mut energy), curvature(2e-3, &mut energy));
        let finite_field = (4. * fine - coarse) / 3.;
        assert!(
            (finite_field - alpha).abs() < 1e-6 * alpha.abs(),
            "{method}: response {alpha} against finite field {finite_field}"
        );
    }
}

#[test]
fn the_independent_and_hartree_limits() {
    let ctx = Context::new(Some(1), 256 << 20).unwrap();
    let integrals = paw_hydrogen(&ctx, &[0, 1]);
    let state = reference(&integrals, "lda");
    // No kernel: the bare orbital-energy differences, A = A + B = A - B = diag.
    let independent =
        Response::new(&integrals, &state, Spin::Singlet, Coupling::Independent).unwrap();
    let mut gaps = independent.delta().to_vec();
    gaps.sort_by(f64::total_cmp);
    let mut r = request(3, Approximation::Casida, Spin::Singlet);
    r.coupling = Coupling::Independent;
    let found = excitations(&integrals, &state, &r).unwrap();
    for (k, e) in found.energies.iter().enumerate() {
        assert!((e - gaps[k]).abs() < 1e-12);
    }
    // The Hartree kernel: no triplet coupling at all, and the singlet's equals four times the
    // Coulomb pair factors of the same grid (V16(i)).
    let hartree = Response::new(&integrals, &state, Spin::Triplet, Coupling::Hartree).unwrap();
    let n = hartree.dimension();
    let block = DMatrix::from_columns(&[trial(n, 0.9)]);
    assert_eq!(hartree.symmetric_kernel(&block).unwrap().amax(), 0.);
    // On the all-electron basis the Cholesky factors are the grid Poisson of the same pair
    // densities, so the two Coulomb routes agree to the factorization's tolerance.
    let ae = hydrogen(&ctx);
    let state = reference(&ae, "lda");
    let singlet = Response::new(&ae, &state, Spin::Singlet, Coupling::Hartree).unwrap();
    let n = singlet.dimension();
    let block = DMatrix::from_columns(&[trial(n, 0.9)]);
    let grid = singlet.matrix(
        singlet
            .symmetric_kernel(&block)
            .unwrap()
            .column(0)
            .as_slice(),
    );
    let eri = ae.eri().unwrap();
    let (o, v) = (singlet.occupied(), singlet.virtual_orbitals());
    let c = singlet.orbitals();
    let y = singlet.matrix(block.column(0).as_slice());
    let m = c.nrows();
    let mut exact = DMatrix::<f64>::zeros(o, v);
    for i in 0..o {
        for a in 0..v {
            let mut value = 0.;
            for j in 0..o {
                for b in 0..v {
                    let mut ij = 0.;
                    for p in 0..m {
                        for q in 0..m {
                            for rr in 0..m {
                                for s in 0..m {
                                    ij += c[(p, i)]
                                        * c[(q, o + a)]
                                        * c[(rr, j)]
                                        * c[(s, o + b)]
                                        * eri.value(p, rr, q, s);
                                }
                            }
                        }
                    }
                    value += ij * y[(j, b)];
                }
            }
            exact[(i, a)] = 4. * value;
        }
    }
    let error = (&grid - &exact).amax();
    assert!(
        error < 1e-6 * exact.amax(),
        "grid Hartree kernel against the Cholesky factors: {error:e}"
    );
}

#[test]
fn a_two_level_system_has_the_closed_form_roots() {
    // V6: one s function per atom, so one occupied and one empty orbital.
    let ctx = Context::new(Some(1), 256 << 20).unwrap();
    let integrals = paw_hydrogen(&ctx, &[0]);
    for method in ["lda", "pbe", "rhf"] {
        let state = reference(&integrals, method);
        for spin in [Spin::Singlet, Spin::Triplet] {
            let response = Response::new(&integrals, &state, spin, Coupling::Adiabatic).unwrap();
            assert_eq!(response.dimension(), 1);
            let delta = response.delta()[0];
            let unit = DMatrix::from_element(1, 1, 1.);
            let s = response.sum(&unit).unwrap()[(0, 0)];
            let d = response.difference(&unit).unwrap()[(0, 0)];
            let tda = excitations(
                &integrals,
                &state,
                &request(1, Approximation::TammDancoff, spin),
            )
            .unwrap();
            let casida =
                excitations(&integrals, &state, &request(1, Approximation::Casida, spin)).unwrap();
            assert!((tda.energies[0] - 0.5 * (s + d)).abs() < 1e-13);
            assert!((casida.energies[0] - (s * d).sqrt()).abs() < 1e-13);
            if method != "rhf" {
                assert!((d - delta).abs() < 1e-15, "{method}: A - B is the gap");
            }
            let plus = casida.plus[0][0][(0, 0)];
            let minus = casida.minus[0][0][(0, 0)];
            assert!((plus * minus - 1.).abs() < 1e-13, "(X + Y)(X - Y) = 1");
            assert!(
                (plus * plus - d / casida.energies[0]).abs() < 1e-12,
                "(X + Y)^2 = (A - B) / omega"
            );
        }
    }
}

#[test]
fn a_request_that_cuts_a_multiplet_is_completed() {
    // V11: H2 with p functions has degenerate pi transitions.
    let ctx = Context::new(Some(1), 256 << 20).unwrap();
    let integrals = paw_hydrogen(&ctx, &[0, 1]);
    let state = reference(&integrals, "lda");
    let all = excitations(
        &integrals,
        &state,
        &request(6, Approximation::TammDancoff, Spin::Singlet),
    )
    .unwrap();
    let cut = (0..all.energies.len() - 1)
        .find(|&k| all.energies[k + 1] - all.energies[k] < 1e-6)
        .expect("H2 with p functions has a degenerate pair");
    let found = excitations(
        &integrals,
        &state,
        &request(cut + 1, Approximation::TammDancoff, Spin::Singlet),
    )
    .unwrap();
    assert!(found.added >= 1, "the multiplet is completed");
    assert_eq!(found.energies.len(), found.requested + found.added);
    let pair: f64 = found.oscillator_strengths[cut..cut + 2].iter().sum();
    let reference_pair: f64 = all.oscillator_strengths[cut..cut + 2].iter().sum();
    assert!(
        (pair - reference_pair).abs() < 1e-9 * reference_pair.max(1e-9),
        "summed strengths are frame-invariant"
    );
}

#[test]
fn thread_counts_give_identical_excitations() {
    // V12: the same reference, the response run on one and on four threads.
    let one = Context::new(Some(1), 256 << 20).unwrap();
    let four = Context::new(Some(4), 256 << 20).unwrap();
    let mut integrals = paw_hydrogen(&one, &[0, 1]);
    for method in ["lda", "pbe"] {
        let state = reference(&integrals, method);
        for approximation in [Approximation::TammDancoff, Approximation::Casida] {
            integrals.context = one.clone();
            let a = excitations(
                &integrals,
                &state,
                &request(4, approximation, Spin::Singlet),
            )
            .unwrap();
            integrals.context = four.clone();
            let b = excitations(
                &integrals,
                &state,
                &request(4, approximation, Spin::Singlet),
            )
            .unwrap();
            assert_eq!(a.energies, b.energies, "{method} {approximation:?}");
            assert_eq!(a.oscillator_strengths, b.oscillator_strengths);
        }
        integrals.context = one.clone();
    }
}

#[test]
fn the_one_center_diagnostic_is_reported_for_paw_kohn_sham_only() {
    let ctx = Context::new(Some(1), 256 << 20).unwrap();
    let paw = paw_hydrogen(&ctx, &[0, 1]);
    let state = reference(&paw, "lda");
    let singlet = excitations(
        &paw,
        &state,
        &request(3, Approximation::TammDancoff, Spin::Singlet),
    )
    .unwrap();
    let shifts = singlet
        .one_center_hartree
        .as_ref()
        .expect("PAW Kohn-Sham reports it");
    assert!(shifts.iter().all(|s| s.is_finite()) && shifts.iter().any(|s| *s != 0.));
    let triplet = excitations(
        &paw,
        &state,
        &request(3, Approximation::TammDancoff, Spin::Triplet),
    )
    .unwrap();
    assert!(triplet.one_center_hartree.unwrap().iter().all(|s| *s == 0.));
    let hf = reference(&paw, "rhf");
    assert!(excitations(
        &paw,
        &hf,
        &request(2, Approximation::TammDancoff, Spin::Singlet)
    )
    .unwrap()
    .one_center_hartree
    .is_none());
    // H2 in HAO has one occupied and one empty orbital: one transition.
    let ae = hydrogen(&ctx);
    let state = reference(&ae, "lda");
    assert!(excitations(
        &ae,
        &state,
        &request(1, Approximation::TammDancoff, Spin::Singlet)
    )
    .unwrap()
    .one_center_hartree
    .is_none());
}

#[test]
fn triplets_have_no_dipole_and_singlet_strengths_follow_the_dipoles() {
    let ctx = Context::new(Some(1), 256 << 20).unwrap();
    let integrals = paw_hydrogen(&ctx, &[0, 1]);
    let state = reference(&integrals, "pbe");
    let triplet = excitations(
        &integrals,
        &state,
        &request(3, Approximation::Casida, Spin::Triplet),
    )
    .unwrap();
    assert!(triplet.oscillator_strengths.iter().all(|f| *f == 0.));
    let singlet = excitations(
        &integrals,
        &state,
        &request(3, Approximation::Casida, Spin::Singlet),
    )
    .unwrap();
    for (k, (f, mu)) in singlet
        .oscillator_strengths
        .iter()
        .zip(&singlet.transition_dipoles)
        .enumerate()
    {
        let expected = 2. / 3. * singlet.energies[k] * mu.iter().map(|x| x * x).sum::<f64>();
        assert!((f - expected).abs() <= 1e-15 * expected.max(1.));
        let weights: f64 = singlet.nto_weights[k].iter().sum();
        assert!((weights - 1.).abs() < 1e-12);
    }
    // The lowest singlet of H2 is the bright sigma_g -> sigma_u transition along the bond.
    assert!(singlet.transition_dipoles[0][2].abs() > 0.1);
}

#[test]
fn unsupported_references_raise_by_name() {
    let ctx = Context::new(Some(1), 256 << 20).unwrap();
    let integrals = hydrogen(&ctx);
    let open = integrals
        .kohn_sham("lda", [1, 1], true, &options())
        .unwrap();
    // An unrestricted reference takes no spin symmetry.
    let mut singlet = Request::new(1);
    singlet.spin = Some(Spin::Singlet);
    let error = excitations(&integrals, &open, &singlet).err().unwrap();
    assert!(error.to_string().contains("spin-conserving"), "{error}");
    for (name, word) in [("pbe+rvv10", "rVV10")] {
        let state = reference(&integrals, name);
        let error = excitations(&integrals, &state, &Request::new(1))
            .err()
            .unwrap();
        assert!(error.to_string().contains(word), "{name}: {error}");
    }
    let state = reference(&integrals, "lda");
    let error = excitations(&integrals, &state, &Request::new(10_000))
        .err()
        .unwrap();
    assert!(error.to_string().contains("exceeds"), "{error}");
}

/// Water on the shipped library (`$MANDACARU_PAW_PATH`; a missing library FAILS): LDA on
/// `lda-sr` and PBE on `pbe-sr`, singlets and triplets, TDA and Casida, against dense solves,
/// with the energies and strengths printed (5 s in release-test, measured 2026-11-12).
#[test]
fn water_roots_on_the_library_datasets_match_dense_solves() {
    use paw_molecular_support::dataset;
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    let water = [
        [0.11, 0.07, 0.12],
        [1.55, 0.04, -0.93],
        [-1.43, -0.05, -0.97],
    ];
    for (set, name) in [("lda-sr", "lda"), ("pbe-sr", "pbe")] {
        let mut arrays = mandacaru_core::paw::molecular::Arrays::new();
        let oxygen = dataset::species(&ctx, &mut arrays, &dataset::read(&ctx, set, "O"));
        let hydrogen = dataset::species(&ctx, &mut arrays, &dataset::read(&ctx, set, "H"));
        let integrals = molecule(
            &ctx,
            &[8, 1, 1],
            &water,
            12.,
            0.5,
            (vec![oxygen, hydrogen], vec![0, 1, 1], arrays),
            "raw",
            &analytic_shells(&[0, 1, 2]),
        );
        let state = integrals
            .kohn_sham(name, [4, 4], false, &options())
            .unwrap();
        assert!(state.converged);
        for spin in [Spin::Singlet, Spin::Triplet] {
            let response = Response::new(&integrals, &state, spin, Coupling::Adiabatic).unwrap();
            let (sum, difference) = dense(&response);
            let (tda, _) = real_eigen((&sum + &difference) * 0.5).unwrap();
            let root = symmetric_root(&difference);
            let (squares, _) = real_eigen(&root * &sum * &root).unwrap();
            for approximation in [Approximation::TammDancoff, Approximation::Casida] {
                let found =
                    excitations(&integrals, &state, &request(5, approximation, spin)).unwrap();
                for (k, e) in found.energies.iter().enumerate() {
                    let exact = match approximation {
                        Approximation::TammDancoff => tda[k],
                        Approximation::Casida => squares[k].sqrt(),
                    };
                    assert!(
                        (e - exact).abs() < 1e-9,
                        "{set} {spin:?} {approximation:?} root {k}"
                    );
                }
                println!(
                    "{set} {spin:?} {approximation:?}: energies {:?} strengths {:?} one-center {:?}",
                    found.energies, found.oscillator_strengths, found.one_center_hartree
                );
            }
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

/// Water centered in a 12 Bohr cell on the shipped library set `set` (oxygen has a partial core
/// and, like hydrogen, the relativistic exchange factor), with analytic s, p and d shells.
fn library_water(ctx: &Context, set: &str) -> MolecularIntegrals {
    let mut arrays = mandacaru_core::paw::molecular::Arrays::new();
    let (o, h) = (dataset::read(ctx, set, "O"), dataset::read(ctx, set, "H"));
    assert!(
        o.relativistic_exchange && h.relativistic_exchange,
        "{set}: the case needs the relativistic factor"
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

#[test]
fn the_kernel_is_the_derivative_of_the_fock_on_library_water() {
    // V1 with a partial core in the XC density and the relativistic exchange factor.
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    for (set, name) in [("lda-sr", "lda"), ("pbe-sr", "pbe")] {
        let integrals = library_water(&ctx, set);
        let state = integrals
            .kohn_sham(name, [4, 4], false, &options())
            .unwrap();
        assert!(state.converged);
        for spin in [Spin::Singlet, Spin::Triplet] {
            check_symmetric_channel(&integrals, &state, spin, &format!("{set} water {spin:?}"));
        }
    }
}

#[test]
fn an_unstable_restricted_reference_raises_in_both_approximations() {
    // H2 stretched to 5 Bohr: the restricted LDA state lies above a triplet of its own response
    // (the symmetry-broken unrestricted state), so the triplet roots are not excitations.
    let ctx = Context::new(Some(2), 512 << 20).unwrap();
    let stretched = |d: f64| {
        let mut arrays = mandacaru_core::paw::molecular::Arrays::new();
        let h = dataset::species(&ctx, &mut arrays, &dataset::read(&ctx, "lda-sr", "H"));
        molecule(
            &ctx,
            &[1, 1],
            &[[6., 6., 6. - d / 2.], [6., 6., 6. + d / 2.]],
            14.,
            0.5,
            (vec![h], vec![0, 0], arrays),
            "raw",
            &analytic_shells(&[0, 1]),
        )
    };
    let bound = stretched(1.4);
    let state = reference(&bound, "lda");
    for approximation in [Approximation::TammDancoff, Approximation::Casida] {
        let found = excitations(&bound, &state, &request(1, approximation, Spin::Triplet)).unwrap();
        assert!(found.energies[0] > 0.);
    }
    let apart = stretched(5.0);
    let state = reference(&apart, "lda");
    let error = excitations(
        &apart,
        &state,
        &request(1, Approximation::TammDancoff, Spin::Triplet),
    )
    .err()
    .expect("a negative TDA triplet is refused");
    assert!(
        error.to_string().contains("non-positive root")
            && error.to_string().contains("unrestricted"),
        "{error}"
    );
    let error = excitations(
        &apart,
        &state,
        &request(1, Approximation::Casida, Spin::Triplet),
    )
    .err()
    .expect("an imaginary Casida triplet is refused");
    assert!(
        error.to_string().contains("no real root") && !error.to_string().contains("tda"),
        "{error}"
    );
}

#[test]
fn the_one_center_dipole_has_the_far_field_coulomb_scale_and_sign() {
    // O and H 15 Bohr apart along x (the analytic functions end at 7 Bohr, so the supports are
    // disjoint). The pair density of the orthonormal s and p_x functions of
    // oxygen has no charge and is a pure dipole, so its Coulomb integral with the unit charge of
    // the hydrogen s function is <s|x|p_x> / R^2. The one-center L = 1 block carries a few
    // percent of the moment, so a wrong scale or sign of it would show here.
    let ctx = Context::new(Some(2), 2 << 30).unwrap();
    let mut arrays = mandacaru_core::paw::molecular::Arrays::new();
    let oxygen = dataset::species(&ctx, &mut arrays, &dataset::read(&ctx, "lda-sr", "O"));
    let hydrogen = dataset::species(&ctx, &mut arrays, &dataset::read(&ctx, "lda-sr", "H"));
    let r = 15.;
    let integrals = molecule(
        &ctx,
        &[8, 1],
        &[[7.5, 15., 15.], [7.5 + r, 15., 15.]],
        30.,
        0.5,
        (vec![oxygen, hydrogen], vec![0, 1], arrays),
        "raw",
        &analytic_shells(&[0, 1, 2]),
    );
    // Functions in order: oxygen s (0), p with m = -1, 0, 1 (y, z, x: 1, 2, 3), d (4-8); hydrogen s (9).
    let (s, px, w) = (0, 3, 9);
    let moment = position_integrals(&integrals, [0.; 3]).unwrap()[0][(s, px)];
    let coulomb = integrals.eri().unwrap().value(s, w, px, w);
    let ratio = coulomb / (moment / (r * r));
    println!("far-field ratio {ratio:.7} (moment {moment:.6})");
    assert!(moment.abs() > 0.1, "the s-p_x moment is large: {moment}");
    assert!(
        (ratio - 1.).abs() < 1e-4,
        "far-field ratio {ratio} (moment {moment}, Coulomb {coulomb:e})"
    );
}

#[test]
fn the_two_level_roots_follow_independently_computed_integrals() {
    // V6 with integrals built outside the response code: (ia|ia) from the Cholesky factors and the
    // XC kernel integrals from central differences of the engine's first derivatives.
    use mandacaru_core::{math::storage::SampleBlocks, models::xc};
    let ctx = Context::new(Some(1), 512 << 20).unwrap();
    let integrals = hydrogen(&ctx);
    let state = reference(&integrals, "lda");
    let (m, n, dv) = (
        integrals.m(),
        integrals.grid.index.points(),
        integrals.grid.index.dv(),
    );
    let response = Response::new(&integrals, &state, Spin::Singlet, Coupling::Adiabatic).unwrap();
    assert_eq!(
        (response.occupied(), response.virtual_orbitals(), m),
        (1, 1, 2)
    );
    let c = response.orbitals().clone();
    let mut phi = [vec![0.; n], vec![0.; n]];
    let mut blocks = SampleBlocks::new(&ctx, &integrals.orthonormal_samples, m, n).unwrap();
    let stride = blocks.stride();
    for start in (0..n).step_by(stride) {
        let count = blocks.load(start).unwrap();
        let samples = nalgebra::DMatrixView::from_slice_with_strides_generic(
            blocks.values(),
            nalgebra::Dyn(count),
            nalgebra::Dyn(m),
            nalgebra::Const::<1>,
            nalgebra::Dyn(stride),
        );
        let mo = samples * &c;
        for i in 0..count {
            phi[0][start + i] = mo[(i, 0)];
            phi[1][start + i] = mo[(i, 1)];
        }
    }
    let (mut singlet_xc, mut triplet_xc) = (0., 0.);
    for k in 0..n {
        let rho = phi[0][k] * phi[0][k];
        if rho < 1e-9 {
            continue;
        }
        let pair = (phi[0][k] * phi[1][k]).powi(2);
        let step = 1e-4 * rho;
        let potential = |a: f64, b: f64| xc::partials("lda", [a, b, 0., 0., 0.]).unwrap().1[0];
        let same = (potential(rho + step, rho) - potential(rho - step, rho)) / (2. * step);
        let other = (potential(rho, rho + step) - potential(rho, rho - step)) / (2. * step);
        singlet_xc += pair * (same + other) * dv;
        triplet_xc += pair * (same - other) * dv;
    }
    let eri = integrals.eri().unwrap();
    let mut coulomb = 0.;
    for p in 0..m {
        for q in 0..m {
            for r in 0..m {
                for s in 0..m {
                    coulomb +=
                        c[(p, 0)] * c[(q, 1)] * c[(r, 0)] * c[(s, 1)] * eri.value(p, r, q, s);
                }
            }
        }
    }
    let delta = response.delta()[0];
    for spin in [Spin::Singlet, Spin::Triplet] {
        // A + B and A of the closed-shell two-level system; A - B is the gap.
        let (sum, resonant) = match spin {
            Spin::Singlet => (
                delta + 4. * coulomb + 2. * singlet_xc,
                delta + 2. * coulomb + singlet_xc,
            ),
            _ => (delta + 2. * triplet_xc, delta + triplet_xc),
        };
        let tda = excitations(
            &integrals,
            &state,
            &request(1, Approximation::TammDancoff, spin),
        )
        .unwrap();
        let casida =
            excitations(&integrals, &state, &request(1, Approximation::Casida, spin)).unwrap();
        assert!(
            (tda.energies[0] - resonant).abs() < 1e-8,
            "{spin:?} TDA {} vs {resonant}",
            tda.energies[0]
        );
        let exact = (delta * sum).sqrt();
        assert!(
            (casida.energies[0] - exact).abs() < 1e-8,
            "{spin:?} Casida {} vs {exact}",
            casida.energies[0]
        );
    }
}

#[test]
#[ignore = "slow: 70 s (measured 2026-11-12); run: cargo test --profile release-test -p mandacaru-core --test molecular_excitations -- --ignored the_screened_hybrid_kernel_is_the_derivative_of_the_fock"]
fn the_screened_hybrid_kernel_is_the_derivative_of_the_fock() {
    // V1 for HSE06: the semilocal kernel, the short-range exact exchange and, for triplets on
    // PAW-LCAO, the second derivative of the spheres' spin polarization (the triplet difference
    // displaces the two spins oppositely, so the polarization the restricted SCF never sees is in
    // the oracle).
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    let all_electron = hydrogen(&ctx);
    let state = reference(&all_electron, "hse06");
    for spin in [Spin::Singlet, Spin::Triplet] {
        check_symmetric_channel(
            &all_electron,
            &state,
            spin,
            &format!("HAO H2 hse06 {spin:?}"),
        );
    }
    let paw = paw_hydrogen(&ctx, &[0, 1]);
    let state = reference(&paw, "hse06");
    for spin in [Spin::Singlet, Spin::Triplet] {
        check_symmetric_channel(&paw, &state, spin, &format!("PAW H2 hse06 {spin:?}"));
    }
    let water = library_water(&ctx, "pbe-sr");
    let state = water.kohn_sham("hse06", [4, 4], false, &options()).unwrap();
    assert!(state.converged);
    for spin in [Spin::Singlet, Spin::Triplet] {
        check_symmetric_channel(
            &water,
            &state,
            spin,
            &format!("pbe-sr water hse06 {spin:?}"),
        );
    }
}

#[test]
#[ignore = "slow: 34 s (measured 2026-11-12); run: cargo test --profile release-test -p mandacaru-core --test molecular_excitations -- --ignored the_screened_exchange_is_the_derivative_in_the_antisymmetric_channel"]
fn the_screened_exchange_is_the_derivative_in_the_antisymmetric_channel() {
    // V1b for HSE06: an imaginary antisymmetric change moves only the screened exchange.
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    for (integrals, label) in [
        (hydrogen(&ctx), "HAO H2"),
        (paw_hydrogen(&ctx, &[0, 1]), "PAW H2"),
        (library_water(&ctx, "pbe-sr"), "pbe-sr water"),
    ] {
        let particles = if label.contains("water") {
            [4, 4]
        } else {
            [1, 1]
        };
        let state = integrals
            .kohn_sham("hse06", particles, false, &options())
            .unwrap();
        let response =
            Response::new(&integrals, &state, Spin::Singlet, Coupling::Adiabatic).unwrap();
        let (o, v) = (response.occupied(), response.virtual_orbitals());
        let c = response.orbitals();
        let y = trial(o * v, 1.37);
        let ym = response.matrix(y.as_slice());
        let a = c.columns(0, o) * &ym * c.columns(o, v).transpose();
        let change = (&a - a.transpose()).map(|x| C64::new(0., x));
        let d = &state.densities[0];
        let fock = |h: f64| {
            reference_fock(&integrals, &state, &[d + &change * C64::new(h, 0.)]).unwrap()[0].clone()
        };
        let imaginary = ((fock(1e-4) - fock(-1e-4)) / C64::new(2e-4, 0.)).map(|z| z.im);
        let projected = c.columns(0, o).transpose() * imaginary * c.columns(o, v);
        let kernel = response
            .antisymmetric_kernel(&DMatrix::from_column_slice(o * v, 1, y.as_slice()))
            .unwrap();
        let kernel = response.matrix(kernel.column(0).as_slice());
        assert!(
            kernel.amax() > 1e-8,
            "{label}: the screened exchange kernel vanishes"
        );
        let error = (&projected - &kernel).amax();
        assert!(
            error < 1e-7 * kernel.amax(),
            "{label}: {error:e} of {:e}",
            kernel.amax()
        );
    }
}

#[test]
#[ignore = "slow: 69 s (measured 2026-11-12); run: cargo test --profile release-test -p mandacaru-core --test molecular_excitations -- --ignored screened_hybrid_roots_satisfy_the_dense_and_sum_rule_identities"]
fn screened_hybrid_roots_satisfy_the_dense_and_sum_rule_identities() {
    // V3, V4 and V5 with a non-diagonal A - B (the short-range exchange).
    let ctx = Context::new(Some(1), 512 << 20).unwrap();
    let mut integrals = paw_hydrogen(&ctx, &[0, 1]);
    let state = reference(&integrals, "hse06");
    for spin in [Spin::Singlet, Spin::Triplet] {
        let response = Response::new(&integrals, &state, spin, Coupling::Adiabatic).unwrap();
        let n = response.dimension();
        let (sum, difference) = dense(&response);
        let delta = DMatrix::from_diagonal(&DVector::from_column_slice(response.delta()));
        assert!(
            (&difference - &delta).amax() > 1e-6,
            "A - B carries the screened exchange"
        );
        let (tda, _) = real_eigen((&sum + &difference) * 0.5).unwrap();
        let root = symmetric_root(&difference);
        let (squares, _) = real_eigen(&root * &sum * &root).unwrap();
        let found = excitations(
            &integrals,
            &state,
            &request(n, Approximation::TammDancoff, spin),
        )
        .unwrap();
        for (k, e) in found.energies.iter().enumerate() {
            assert!((e - tda[k]).abs() < 1e-10, "{spin:?} TDA root {k}");
        }
        let casida =
            excitations(&integrals, &state, &request(n, Approximation::Casida, spin)).unwrap();
        for (k, e) in casida.energies.iter().enumerate() {
            assert!(
                (e - squares[k].sqrt()).abs() < 1e-10,
                "{spin:?} Casida root {k}"
            );
        }
        if spin == Spin::Singlet {
            let d = dipole_blocks(&integrals, &response);
            let inverse = sum.clone().try_inverse().unwrap();
            for axis in 0..3 {
                let weighted: f64 = casida
                    .energies
                    .iter()
                    .zip(&casida.transition_dipoles)
                    .map(|(w, mu)| w * mu[axis] * mu[axis])
                    .sum();
                let exact = 2. * d[axis].dot(&(&difference * &d[axis]));
                assert!(
                    (weighted - exact).abs() < 1e-9 * exact.abs().max(1e-8),
                    "axis {axis}: sum rule"
                );
                let alpha: f64 = casida
                    .energies
                    .iter()
                    .zip(&casida.transition_dipoles)
                    .map(|(w, mu)| 2. * mu[axis] * mu[axis] / w)
                    .sum();
                let solved = 4. * d[axis].dot(&(&inverse * &d[axis]));
                assert!(
                    (alpha - solved).abs() < 1e-9 * solved.abs().max(1e-8),
                    "axis {axis}: alpha"
                );
            }
        }
    }
    // The finite-field curvature of the HSE06 SCF along z.
    let response = Response::new(&integrals, &state, Spin::Singlet, Coupling::Adiabatic).unwrap();
    let n = response.dimension();
    let casida = excitations(
        &integrals,
        &state,
        &request(n, Approximation::Casida, Spin::Singlet),
    )
    .unwrap();
    let alpha: f64 = casida
        .energies
        .iter()
        .zip(&casida.transition_dipoles)
        .map(|(w, mu)| 2. * mu[2] * mu[2] / w)
        .sum();
    let z = position_integrals(&integrals, [0.; 3]).unwrap()[2].clone();
    let base: Vec<f64> = integrals.one.as_slice().to_vec();
    let m = integrals.m();
    let mut energy = |field: f64| {
        for a in 0..m {
            for b in 0..m {
                integrals.one.as_mut_slice()[a * m + b] = base[a * m + b] + field * z[(a, b)];
            }
        }
        reference(&integrals, "hse06").electronic_energy
    };
    let e0 = energy(0.);
    let curvature =
        |f: f64, energy: &mut dyn FnMut(f64) -> f64| -(energy(f) + energy(-f) - 2. * e0) / (f * f);
    let (coarse, fine) = (curvature(4e-3, &mut energy), curvature(2e-3, &mut energy));
    let finite_field = (4. * fine - coarse) / 3.;
    assert!(
        (finite_field - alpha).abs() < 1e-4 * alpha.abs(),
        "hse06: response {alpha} against finite field {finite_field}"
    );
}

#[test]
fn the_all_electron_screened_hybrid_kernel_is_the_derivative_in_both_channels() {
    // The fast HSE06 gate (no PAW spheres): V1 for singlets and triplets, V1b for the exchange.
    let ctx = Context::new(Some(2), 512 << 20).unwrap();
    let integrals = hydrogen(&ctx);
    let state = reference(&integrals, "hse06");
    for spin in [Spin::Singlet, Spin::Triplet] {
        check_symmetric_channel(&integrals, &state, spin, &format!("HAO H2 hse06 {spin:?}"));
    }
    let response = Response::new(&integrals, &state, Spin::Singlet, Coupling::Adiabatic).unwrap();
    let unit = DMatrix::from_element(1, 1, 1.);
    let k = response.antisymmetric_kernel(&unit).unwrap()[(0, 0)];
    let c = response.orbitals();
    let a = c.column(0) * c.column(1).transpose();
    let change = (&a - a.transpose()).map(|x| C64::new(0., x));
    let d = &state.densities[0];
    let fock = |h: f64| {
        reference_fock(&integrals, &state, &[d + &change * C64::new(h, 0.)]).unwrap()[0].clone()
    };
    let imaginary = ((fock(1e-4) - fock(-1e-4)) / C64::new(2e-4, 0.)).map(|z| z.im);
    let projected = (c.column(0).transpose() * imaginary * c.column(1))[(0, 0)];
    assert!(
        k.abs() > 1e-6 && (projected - k).abs() < 1e-7 * k.abs(),
        "{projected} against {k}"
    );
}

#[test]
fn the_velocity_form_follows_the_length_form_on_a_resolved_basis() {
    // A diagnostic of the basis, not an identity: for exact states of a local Hamiltonian
    // <0|grad|n> = omega <0|r|n>. Triplets have neither moment.
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    for (integrals, label, particles) in [
        (hydrogen(&ctx), "HAO H2", [1, 1]),
        (paw_hydrogen(&ctx, &[0, 1]), "PAW H2", [1, 1]),
        (library_water(&ctx, "lda-sr"), "lda-sr water", [4, 4]),
    ] {
        let state = integrals
            .kohn_sham("lda", particles, false, &options())
            .unwrap();
        let n = Response::new(&integrals, &state, Spin::Singlet, Coupling::Adiabatic)
            .unwrap()
            .dimension();
        let casida = excitations(
            &integrals,
            &state,
            &request(n.min(6), Approximation::Casida, Spin::Singlet),
        )
        .unwrap();
        for k in 0..casida.energies.len() {
            println!(
                "{label} root {k}: omega {:.6} Ha f_length {:.6} f_velocity {:.6} mu {:?} p/omega {:?}",
                casida.energies[k],
                casida.oscillator_strengths[k],
                casida.velocity_oscillator_strengths[k],
                casida.transition_dipoles[k],
                casida.transition_velocities[k].map(|p| p / casida.energies[k])
            );
        }
        let triplet = excitations(
            &integrals,
            &state,
            &request(n.min(2), Approximation::Casida, Spin::Triplet),
        )
        .unwrap();
        assert!(triplet
            .velocity_oscillator_strengths
            .iter()
            .all(|f| *f == 0.));
    }
}

/// NH2 (a doublet with a nondegenerate singly occupied orbital) on the shipped library set `set`,
/// with analytic s, p and d shells: 4 alpha and 3 beta valence electrons.
fn library_amidogen(ctx: &Context, set: &str) -> MolecularIntegrals {
    let mut arrays = mandacaru_core::paw::molecular::Arrays::new();
    let nitrogen = dataset::species(ctx, &mut arrays, &dataset::read(ctx, set, "N"));
    let hydrogen = dataset::species(ctx, &mut arrays, &dataset::read(ctx, set, "H"));
    molecule(
        ctx,
        &[7, 1, 1],
        &[[6.03, 6.05, 6.31], [7.53, 6.02, 5.08], [4.49, 5.97, 5.12]],
        12.,
        0.5,
        (vec![nitrogen, hydrogen], vec![0, 1, 1], arrays),
        "raw",
        &analytic_shells(&[0, 1, 2]),
    )
}
fn unrestricted_reference(
    integrals: &MolecularIntegrals,
    method: &str,
    particles: [usize; 2],
) -> MeanField {
    let state = if method == "uhf" {
        integrals.mean_field("uhf", particles, &options())
    } else {
        integrals.kohn_sham(method, particles, true, &options())
    }
    .unwrap();
    assert!(state.converged && state.method == "uhf");
    state
}
fn unrestricted_request(num_states: usize, approximation: Approximation) -> Request {
    let mut r = Request::new(num_states);
    r.approximation = approximation;
    r.tolerance = 1e-9;
    r
}

#[test]
#[ignore = "slow: 18 s (measured 2026-11-12); run: cargo test --profile release-test -p mandacaru-core --test molecular_excitations -- --ignored closed_shells_through_the_unrestricted_path_give_singlets_and_triplets"]
fn closed_shells_through_the_unrestricted_path_give_singlets_and_triplets() {
    // V8: the unrestricted spectrum of a closed shell is the union of its singlets and triplets,
    // and <S^2> is exactly 0 and 2 on them.
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    for (integrals, label) in [
        (hydrogen(&ctx), "HAO H2"),
        (paw_hydrogen(&ctx, &[0, 1]), "PAW H2"),
    ] {
        for (method, restricted) in [
            ("lda", "lda"),
            ("pbe", "pbe"),
            ("uhf", "rhf"),
            ("hse06", "hse06"),
        ] {
            if method == "hse06" && label == "PAW H2" {
                continue; // the PAW screened case is in the slow tier
            }
            let open = unrestricted_reference(&integrals, method, [1, 1]);
            let closed = reference(&integrals, restricted);
            let n = Response::new(&integrals, &closed, Spin::Singlet, Coupling::Adiabatic)
                .unwrap()
                .dimension();
            for approximation in [Approximation::TammDancoff, Approximation::Casida] {
                let mut expected: Vec<(f64, f64)> = Vec::new();
                for (spin, s2) in [(Spin::Singlet, 0.), (Spin::Triplet, 2.)] {
                    let found =
                        excitations(&integrals, &closed, &request(n, approximation, spin)).unwrap();
                    expected.extend(found.energies.iter().map(|&e| (e, s2)));
                }
                expected.sort_by(|a, b| a.0.total_cmp(&b.0));
                let found = excitations(
                    &integrals,
                    &open,
                    &unrestricted_request(2 * n, approximation),
                )
                .unwrap();
                assert_eq!(found.energies.len(), expected.len());
                let squared = found.spin_squared.as_ref().unwrap();
                for (k, ((e, s2), (&f, &g))) in expected
                    .iter()
                    .zip(found.energies.iter().zip(squared))
                    .enumerate()
                {
                    assert!(
                        (e - f).abs() < 1e-8,
                        "{label} {method} {approximation:?} root {k}: {e} vs {f}"
                    );
                    // Exactly degenerate singlet-triplet pairs mix; their <S^2> sums are 2.
                    let degenerate = expected.iter().filter(|x| (x.0 - e).abs() < 1e-8).count() > 1;
                    if !degenerate {
                        assert!(
                            (s2 - g).abs() < 1e-8,
                            "{label} {method} root {k}: <S^2> {g} vs {s2}"
                        );
                    }
                }
                assert!(found.reference_spin_squared.unwrap().abs() < 1e-10);
            }
        }
    }
}

/// V1 on an unrestricted reference: the kernel of both spin blocks against a central difference
/// of the reference's spin-resolved Fock matrices along the two spins' density changes.
fn check_unrestricted_channel(integrals: &MolecularIntegrals, state: &MeanField, label: &str) {
    check_unrestricted_channel_with(integrals, state, label, 1e-4)
}
/// [`check_unrestricted_channel`] with the difference steps `step` and `2 step`. HSE06 needs small
/// steps: its hole model bends the reduced gradient above `t = 1` with a kink, and a larger step
/// carries tail points across it (the difference then misses by up to 5e-3 at 1e-4 and agrees to
/// 5e-10 at 1e-5 on the synthetic PAW H2+, 2026-11-12).
fn check_unrestricted_channel_with(
    integrals: &MolecularIntegrals,
    state: &MeanField,
    label: &str,
    step: f64,
) {
    let response = Unrestricted::new(integrals, state, Coupling::Adiabatic).unwrap();
    let n = response.dimension();
    let y = trial(n, 0.731);
    let blocks = response.matrices(y.as_slice());
    let changes: Vec<DMatrix<C64>> = (0..2)
        .map(|s| {
            let (o, v) = response.channel(s);
            let c = response.orbitals(s);
            let d = c.columns(0, o) * &blocks[s] * c.columns(o, v).transpose();
            (&d + d.transpose()).map(|x| C64::new(x, 0.))
        })
        .collect();
    let fock = |h: f64| -> Vec<DMatrix<f64>> {
        let displaced: Vec<DMatrix<C64>> = (0..2)
            .map(|s| &state.densities[s] + &changes[s] * C64::new(h, 0.))
            .collect();
        reference_fock(integrals, state, &displaced)
            .unwrap()
            .into_iter()
            .map(|f| f.map(|z| z.re))
            .collect()
    };
    let difference = |h: f64| -> Vec<DMatrix<f64>> {
        let (p, m) = (fock(h), fock(-h));
        (0..2).map(|s| (&p[s] - &m[s]) / (2. * h)).collect()
    };
    let (coarse, fine) = (difference(2. * step), difference(step));
    let kernel = response
        .symmetric_kernel(&DMatrix::from_column_slice(n, 1, y.as_slice()))
        .unwrap();
    let kernel = response.matrices(kernel.column(0).as_slice());
    let mut scale = 0f64;
    let mut error = 0f64;
    for s in 0..2 {
        let (o, v) = response.channel(s);
        let c = response.orbitals(s);
        let estimate = (&fine[s] * 4. - &coarse[s]) / 3.;
        let projected = c.columns(0, o).transpose() * estimate * c.columns(o, v);
        scale = scale.max(kernel[s].amax());
        error = error.max((&projected - &kernel[s]).amax());
    }
    assert!(scale > 1e-6, "{label}: the kernel vanishes");
    assert!(
        error < 1e-6 * scale,
        "{label}: kernel misses the Fock difference by {error:e} of {scale:e}"
    );
}

#[test]
fn the_unrestricted_kernel_is_the_derivative_of_the_fock_on_an_open_shell() {
    // V1 and V1b on NH2: UKS LDA and PBE (partial core, relativistic exchange factor) and UHF.
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    // Each functional on its own library set (PBE on `pbe-sr`, as a calculation selects it).
    let pbe = library_amidogen(&ctx, "pbe-sr");
    let state = unrestricted_reference(&pbe, "pbe", [4, 3]);
    check_unrestricted_channel(&pbe, &state, "NH2 pbe on pbe-sr");
    let integrals = library_amidogen(&ctx, "lda-sr");
    for method in ["lda", "uhf"] {
        let state = unrestricted_reference(&integrals, method, [4, 3]);
        check_unrestricted_channel(&integrals, &state, &format!("NH2 {method}"));
    }
    let state = unrestricted_reference(&integrals, "uhf", [4, 3]);
    let response = Unrestricted::new(&integrals, &state, Coupling::Adiabatic).unwrap();
    let n = response.dimension();
    let y = trial(n, 1.37);
    let blocks = response.matrices(y.as_slice());
    let changes: Vec<DMatrix<C64>> = (0..2)
        .map(|s| {
            let (o, v) = response.channel(s);
            let c = response.orbitals(s);
            let a = c.columns(0, o) * &blocks[s] * c.columns(o, v).transpose();
            (&a - a.transpose()).map(|x| C64::new(0., x))
        })
        .collect();
    let fock = |h: f64| {
        let displaced: Vec<DMatrix<C64>> = (0..2)
            .map(|s| &state.densities[s] + &changes[s] * C64::new(h, 0.))
            .collect();
        reference_fock(&integrals, &state, &displaced).unwrap()
    };
    let (p, m) = (fock(1e-4), fock(-1e-4));
    let kernel = response
        .antisymmetric_kernel(&DMatrix::from_column_slice(n, 1, y.as_slice()))
        .unwrap();
    let kernel = response.matrices(kernel.column(0).as_slice());
    for s in 0..2 {
        let (o, v) = response.channel(s);
        let c = response.orbitals(s);
        let imaginary = ((&p[s] - &m[s]) / C64::new(2e-4, 0.)).map(|z| z.im);
        let projected = c.columns(0, o).transpose() * imaginary * c.columns(o, v);
        let error = (&projected - &kernel[s]).amax();
        assert!(
            kernel[s].amax() > 1e-6 && error < 1e-7 * kernel[s].amax(),
            "UHF spin {s}: {error:e}"
        );
    }
}

#[test]
fn the_sphere_polarization_enters_the_kernel_of_a_polarized_reference() {
    // HSE06 on the synthetic PAW H2+ (one alpha electron): the spheres' spin polarization at a
    // polarized reference, in both spin blocks of V1.
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    let integrals = paw_hydrogen(&ctx, &[0, 1]);
    let state = unrestricted_reference(&integrals, "hse06", [1, 0]);
    check_unrestricted_channel_with(&integrals, &state, "PAW H2+ hse06", 1e-5);
}

#[test]
#[ignore = "slow: 12 s (measured 2026-11-12); run: cargo test --profile release-test -p mandacaru-core --test molecular_excitations -- --ignored unrestricted_roots_satisfy_the_dense_sum_rule_and_polarizability_identities"]
fn unrestricted_roots_satisfy_the_dense_sum_rule_and_polarizability_identities() {
    // V3, V4 (sum_n omega mu^2 = d^T (A - B) d in spin orbitals) and V5 (alpha = 2 sum mu^2 / omega
    // = 2 d^T (A + B)^-1 d = the finite-field curvature) on NH2, LDA and UHF; and V12.
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    let mut integrals = library_amidogen(&ctx, "lda-sr");
    for method in ["lda", "uhf"] {
        let state = unrestricted_reference(&integrals, method, [4, 3]);
        let response = Unrestricted::new(&integrals, &state, Coupling::Adiabatic).unwrap();
        let n = response.dimension();
        let identity = DMatrix::identity(n, n);
        let sum = response.sum(&identity).unwrap();
        let difference = response.difference(&identity).unwrap();
        let (tda, _) = real_eigen((&sum + &difference) * 0.5).unwrap();
        let root = symmetric_root(&difference);
        let (squares, _) = real_eigen(&root * &sum * &root).unwrap();
        let few = excitations(
            &integrals,
            &state,
            &unrestricted_request(4, Approximation::TammDancoff),
        )
        .unwrap();
        for (k, e) in few.energies.iter().enumerate() {
            assert!((e - tda[k]).abs() < 1e-10, "{method} TDA root {k}");
        }
        let all = excitations(
            &integrals,
            &state,
            &unrestricted_request(n, Approximation::Casida),
        )
        .unwrap();
        for (k, e) in all.energies.iter().enumerate() {
            assert!(
                (e - squares[k].sqrt()).abs() < 1e-10,
                "{method} Casida root {k}"
            );
        }
        let position = position_integrals(&integrals, [0.; 3]).unwrap();
        let inverse = sum.clone().try_inverse().unwrap();
        for axis in 0..3 {
            let mut d = Vec::with_capacity(n);
            for s in 0..2 {
                let (o, v) = response.channel(s);
                let c = response.orbitals(s);
                let block = c.columns(0, o).transpose() * &position[axis] * c.columns(o, v);
                for i in 0..o {
                    for a in 0..v {
                        d.push(block[(i, a)]);
                    }
                }
            }
            let d = DVector::from_vec(d);
            let weighted: f64 = all
                .energies
                .iter()
                .zip(&all.transition_dipoles)
                .map(|(w, mu)| w * mu[axis] * mu[axis])
                .sum();
            let exact = d.dot(&(&difference * &d));
            assert!(
                (weighted - exact).abs() < 1e-9 * exact.abs().max(1e-8),
                "{method} axis {axis}: sum rule"
            );
            let alpha: f64 = all
                .energies
                .iter()
                .zip(&all.transition_dipoles)
                .map(|(w, mu)| 2. * mu[axis] * mu[axis] / w)
                .sum();
            let solved = 2. * d.dot(&(&inverse * &d));
            assert!(
                (alpha - solved).abs() < 1e-9 * solved.abs(),
                "{method} axis {axis}: polarizability"
            );
        }
        // The finite-field curvature along z of the unrestricted SCF.
        let alpha: f64 = all
            .energies
            .iter()
            .zip(&all.transition_dipoles)
            .map(|(w, mu)| 2. * mu[2] * mu[2] / w)
            .sum();
        let z = position[2].clone();
        let base: Vec<f64> = integrals.one.as_slice().to_vec();
        let m = integrals.m();
        let mut energy = |field: f64| {
            for a in 0..m {
                for b in 0..m {
                    integrals.one.as_mut_slice()[a * m + b] = base[a * m + b] + field * z[(a, b)];
                }
            }
            unrestricted_reference(&integrals, method, [4, 3]).electronic_energy
        };
        let e0 = energy(0.);
        let curvature = |f: f64, energy: &mut dyn FnMut(f64) -> f64| {
            -(energy(f) + energy(-f) - 2. * e0) / (f * f)
        };
        let (coarse, fine) = (curvature(4e-3, &mut energy), curvature(2e-3, &mut energy));
        let finite_field = (4. * fine - coarse) / 3.;
        energy(0.);
        assert!(
            (finite_field - alpha).abs() < 1e-6 * alpha.abs(),
            "{method}: response {alpha} against finite field {finite_field}"
        );
        // The doublet reference and its roots: <S^2> near 3/4, never below it by more than round-off.
        let s2 = all.spin_squared.as_ref().unwrap();
        assert!((all.reference_spin_squared.unwrap() - 0.75).abs() < 0.05);
        assert!(
            s2.iter().all(|v| *v > 0.75 - 1e-8),
            "{method}: <S^2> {s2:?}"
        );
    }
    // V12: the same state, one and four threads (PBE on its own set, `pbe-sr`).
    let mut integrals = library_amidogen(&ctx, "pbe-sr");
    let state = unrestricted_reference(&integrals, "pbe", [4, 3]);
    let four = Context::new(Some(4), 1 << 30).unwrap();
    let one = integrals.context.clone();
    let a = excitations(
        &integrals,
        &state,
        &unrestricted_request(4, Approximation::Casida),
    )
    .unwrap();
    integrals.context = four;
    let b = excitations(
        &integrals,
        &state,
        &unrestricted_request(4, Approximation::Casida),
    )
    .unwrap();
    integrals.context = one;
    assert_eq!(a.energies, b.energies);
    assert_eq!(a.oscillator_strengths, b.oscillator_strengths);
    assert_eq!(a.spin_squared, b.spin_squared);
}

#[test]
fn an_all_electron_closed_shell_through_the_unrestricted_path_is_a_singlet_and_a_triplet() {
    // The fast V8 gate: H2 in HAO has one transition per spin, so the unrestricted path gives
    // exactly the singlet and the triplet, with <S^2> 0 and 2.
    let ctx = Context::new(Some(2), 512 << 20).unwrap();
    let integrals = hydrogen(&ctx);
    for (method, restricted) in [("lda", "lda"), ("uhf", "rhf")] {
        let open = unrestricted_reference(&integrals, method, [1, 1]);
        let closed = reference(&integrals, restricted);
        for approximation in [Approximation::TammDancoff, Approximation::Casida] {
            let singlet = excitations(
                &integrals,
                &closed,
                &request(1, approximation, Spin::Singlet),
            )
            .unwrap();
            let triplet = excitations(
                &integrals,
                &closed,
                &request(1, approximation, Spin::Triplet),
            )
            .unwrap();
            let found =
                excitations(&integrals, &open, &unrestricted_request(2, approximation)).unwrap();
            let squared = found.spin_squared.unwrap();
            assert!(
                (found.energies[0] - triplet.energies[0]).abs() < 1e-8,
                "{method} {approximation:?}"
            );
            assert!(
                (found.energies[1] - singlet.energies[0]).abs() < 1e-8,
                "{method} {approximation:?}"
            );
            assert!(
                (squared[0] - 2.).abs() < 1e-8 && squared[1].abs() < 1e-8,
                "{method}: {squared:?}"
            );
            assert!((found.oscillator_strengths[1] - singlet.oscillator_strengths[0]).abs() < 1e-8);
            assert!(found.oscillator_strengths[0].abs() < 1e-12);
        }
    }
}

#[test]
#[ignore = "slow: 34 s (measured 2026-11-12); run: cargo test --profile release-test -p mandacaru-core --test molecular_excitations -- --ignored the_unrestricted_screened_hybrid_kernel_on_an_open_shell"]
fn the_unrestricted_screened_hybrid_kernel_on_an_open_shell_is_the_derivative_of_the_fock() {
    // V1 for HSE06 on NH2 (`pbe-sr`): both spin blocks, the short-range exchange of each spin and
    // the spheres' polarization at the polarized reference. Step 1e-5: larger steps carry tail
    // points across the hole model's kink at t = 1 (Q17, kept by the owner).
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    let integrals = library_amidogen(&ctx, "pbe-sr");
    let state = unrestricted_reference(&integrals, "hse06", [4, 3]);
    check_unrestricted_channel_with(&integrals, &state, "NH2 hse06 on pbe-sr", 1e-5);
}

#[test]
fn the_meta_gga_kernel_is_the_derivative_of_the_fock() {
    // V1 for r2SCAN: the kinetic-density and product-rule gradient responses, restricted singlet
    // and triplet on H2 (HAO and synthetic PAW). The open shell is the next test.
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    for (integrals, label) in [
        (hydrogen(&ctx), "HAO H2"),
        (paw_hydrogen(&ctx, &[0, 1]), "PAW H2"),
    ] {
        let state = reference(&integrals, "r2scan");
        for spin in [Spin::Singlet, Spin::Triplet] {
            check_symmetric_channel(
                &integrals,
                &state,
                spin,
                &format!("{label} r2scan {spin:?}"),
            );
        }
    }
}

#[test]
#[ignore = "slow: 16 s (measured 2026-11-12); run: cargo test --profile release-test -p mandacaru-core --test molecular_excitations -- --ignored the_unrestricted_meta_gga_kernel"]
fn the_unrestricted_meta_gga_kernel_on_an_open_shell_is_the_derivative_of_the_fock() {
    // V1 for r2SCAN on NH2 (`pbe-sr`, partial core with its kinetic density), both spin blocks.
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    let amidogen = library_amidogen(&ctx, "pbe-sr");
    let state = unrestricted_reference(&amidogen, "r2scan", [4, 3]);
    check_unrestricted_channel(&amidogen, &state, "NH2 r2scan on pbe-sr");
}

#[test]
fn a_rounding_sized_negative_total_gradient_contraction_is_zero_and_a_real_one_raises() {
    // `suu + 2 sud + sdd` cancels to rounding where the spin gradients are opposite. The point is
    // the H2 midpoint of the triplet trial density that raised "must be nonnegative"
    // (sigma = -1.65e-24 against suu + sdd = 2.6e-8, half an ulp of the sum).
    use mandacaru_core::models::xc;
    let (suu, sud, sdd) = (
        1.3044386476184982e-8,
        -1.3044386475288054e-8,
        1.3044386474391124e-8,
    );
    assert!(suu + 2. * sud + sdd < 0.);
    let point = |sud: f64| {
        [
            0.11695117185962953,
            0.11696117484947544,
            suu,
            sud,
            sdd,
            0.05,
            0.05,
        ]
    };
    let (energy, partials) = xc::point_partials("r2scan", point(sud), false).unwrap();
    let (zero, _) = xc::point_partials("r2scan", point(-0.5 * (suu + sdd)), false).unwrap();
    assert!(energy.is_finite() && partials.iter().all(|v| v.is_finite()));
    assert!(
        (energy - zero).abs() < 1e-12 * zero.abs(),
        "{energy} against {zero}"
    );
    // The second derivatives take the same path.
    let (_, _, hessian) = xc::point_hessian_meta(point(sud)).unwrap();
    assert!(hessian.iter().flatten().all(|v| v.is_finite()));
    // A contraction that is negative by more than rounding is not a gradient field.
    let error = xc::point_partials("r2scan", point(-0.6 * (suu + sdd)), false)
        .err()
        .expect("a genuinely negative contraction must raise")
        .to_string();
    assert!(error.contains("must be nonnegative"), "{error}");
}

#[test]
fn meta_gga_roots_satisfy_the_dense_and_sum_rule_identities() {
    // V3 and V4 for r2SCAN with the current-density term of tau~ (owner decision Q19a): A - B is
    // the gap plus -2 int (v_tau / n) j^ia . j^jb, symmetric and not diagonal, so the Casida
    // roots are those of the paired problem and V4 takes its general form
    // sum_n omega_n mu_n^2 = 2 d^T (A - B) d.
    let ctx = Context::new(Some(1), 512 << 20).unwrap();
    let integrals = paw_hydrogen(&ctx, &[0, 1]);
    let state = reference(&integrals, "r2scan");
    let response = Response::new(&integrals, &state, Spin::Singlet, Coupling::Adiabatic).unwrap();
    let n = response.dimension();
    let (sum, difference) = dense(&response);
    let delta = DMatrix::from_diagonal(&DVector::from_column_slice(response.delta()));
    let current = &difference - &delta;
    eprintln!(
        "r2SCAN: current-density part of A - B {:.3e} (largest entry), asymmetry {:.1e}",
        current.amax(),
        (&current - current.transpose()).amax()
    );
    assert!(current.amax() > 1e-6, "the current-density term vanishes");
    assert!((&current - current.transpose()).amax() < 1e-12 * current.amax().max(1.));
    let root = symmetric_root(&difference);
    let (squares, _) = real_eigen(&root * &sum * &root).unwrap();
    let casida = excitations(
        &integrals,
        &state,
        &request(n, Approximation::Casida, Spin::Singlet),
    )
    .unwrap();
    for (k, e) in casida.energies.iter().enumerate() {
        assert!((e - squares[k].sqrt()).abs() < 1e-10, "Casida root {k}");
    }
    let d = dipole_blocks(&integrals, &response);
    for axis in 0..3 {
        let weighted: f64 = casida
            .energies
            .iter()
            .zip(&casida.transition_dipoles)
            .map(|(w, mu)| w * mu[axis] * mu[axis])
            .sum();
        let x = d[axis].clone();
        let general = 2. * x.dot(&(&difference * &x));
        assert!(
            (weighted - general).abs() < 1e-9 * general.abs().max(1e-8),
            "axis {axis}: {weighted} vs {general}"
        );
    }
}

/// Copper hydride from the `lda-sr` library (Cu 3d shell, a real occupation near 10) with a +U
/// shell on the copper d channel when `hubbard` is `Some((form, U, J))` (eV), and its electron
/// count. The synthetic hydrogen d shell this replaces holds about 1e-8 electrons, so its +U
/// kernel (about U n_d) sat below the 1e-8 guards and, in the finite-difference test, under the
/// Hartree and xc kernels by ten orders of magnitude (2026-11-12).
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
/// The share of the +U term in the oracle's response to the density change the V1 test applies
/// (`check_symmetric_channel`): the largest occupied-empty entry of the central difference of
/// `reference_fock` with the +U shell minus the same without it, over the largest entry with it.
/// The oracle is independent of the kernel under test. A finite-difference test in which +U is
/// a rounding-sized part of the whole cannot fail for a wrong +U term, so every +U test asserts
/// this share is large against its tolerance.
fn hubbard_share(
    with: &MolecularIntegrals,
    plain: &MolecularIntegrals,
    state: &MeanField,
    spin: Spin,
) -> f64 {
    let response = Response::new(with, state, spin, Coupling::Adiabatic).unwrap();
    let (o, v) = (response.occupied(), response.virtual_orbitals());
    let c = response.orbitals();
    let y = response.matrix(trial(o * v, 0.731).as_slice());
    let change = c.columns(0, o) * &y * c.columns(o, v).transpose();
    let change = (&change + change.transpose()).map(|x| C64::new(x, 0.));
    let d = &state.densities[0];
    let h = 1e-4;
    let derivative = |integrals: &MolecularIntegrals| {
        let fock = |sign: f64| {
            let step = &change * C64::new(sign * h, 0.);
            let f = match spin {
                Spin::Singlet => reference_fock(integrals, state, &[d + &step]).unwrap(),
                _ => reference_fock(integrals, state, &[d + &step, d - &step]).unwrap(),
            };
            f[0].map(|z| z.re)
        };
        c.columns(0, o).transpose() * ((fock(1.) - fock(-1.)) / (2. * h)) * c.columns(o, v)
    };
    let (u, p) = (derivative(with), derivative(plain));
    (&u - &p).amax() / u.amax()
}
/// [`hubbard_share`] of the unrestricted response (the change `check_unrestricted_channel` uses).
fn unrestricted_hubbard_share(
    with: &MolecularIntegrals,
    plain: &MolecularIntegrals,
    state: &MeanField,
) -> f64 {
    let response = Unrestricted::new(with, state, Coupling::Adiabatic).unwrap();
    let blocks = response.matrices(trial(response.dimension(), 0.731).as_slice());
    let changes: Vec<DMatrix<C64>> = (0..2)
        .map(|s| {
            let (o, v) = response.channel(s);
            let c = response.orbitals(s);
            let d = c.columns(0, o) * &blocks[s] * c.columns(o, v).transpose();
            (&d + d.transpose()).map(|x| C64::new(x, 0.))
        })
        .collect();
    let h = 1e-4;
    let derivative = |integrals: &MolecularIntegrals| -> Vec<DMatrix<f64>> {
        let fock = |sign: f64| {
            let displaced: Vec<DMatrix<C64>> = (0..2)
                .map(|s| &state.densities[s] + &changes[s] * C64::new(sign * h, 0.))
                .collect();
            reference_fock(integrals, state, &displaced).unwrap()
        };
        let (p, m) = (fock(1.), fock(-1.));
        (0..2)
            .map(|s| {
                let (o, v) = response.channel(s);
                let c = response.orbitals(s);
                let f = ((&p[s] - &m[s]) / C64::new(2. * h, 0.)).map(|z| z.re);
                c.columns(0, o).transpose() * f * c.columns(o, v)
            })
            .collect()
    };
    let (u, p) = (derivative(with), derivative(plain));
    let top = |m: &[DMatrix<f64>]| m.iter().fold(0f64, |t, x| t.max(x.amax()));
    let moved: Vec<DMatrix<f64>> = u.iter().zip(&p).map(|(a, b)| a - b).collect();
    top(&moved) / top(&u)
}
/// The smallest share of the +U part of a kernel a +U test accepts: a missing or wrong +U term
/// then moves the finite difference by at least 1e-3 of the kernel, against its 1e-6 gate.
const MIN_HUBBARD_SHARE: f64 = 1e-3;
/// The smallest +U kernel entry (Hartree) the closed-form tests accept: measurably nonzero,
/// where the synthetic hydrogen d shell this replaces gave 3e-12.
const MIN_HUBBARD_KERNEL: f64 = 1e-5;

#[test]
fn the_hubbard_kernel_is_the_derivative_of_the_fock() {
    // V1 for +U, both forms: restricted singlet and triplet, and the unrestricted CuH+, each with
    // a guard that the +U part is a measurable share of the kernel.
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    let (plain, electrons) = copper_hydride(&ctx, None);
    for (form, u, j) in [("dudarev-fll", 20., 0.), ("liechtenstein-fll", 20., 3.)] {
        let (integrals, _) = copper_hydride(&ctx, Some((form, u, j)));
        assert!(integrals.corrections.as_ref().unwrap().hubbard.is_some());
        let state = integrals
            .kohn_sham("lda", [electrons / 2, electrons / 2], false, &options())
            .unwrap();
        assert!(state.converged);
        for spin in [Spin::Singlet, Spin::Triplet] {
            let share = hubbard_share(&integrals, &plain, &state, spin);
            println!("{form} {spin:?}: +U share of the kernel {share:.3e}");
            assert!(
                share > MIN_HUBBARD_SHARE,
                "{form} {spin:?}: +U share {share:e}"
            );
            check_symmetric_channel(&integrals, &state, spin, &format!("{form} {spin:?}"));
        }
        let cation = unrestricted_reference(&integrals, "lda", [electrons / 2, electrons / 2 - 1]);
        let share = unrestricted_hubbard_share(&integrals, &plain, &cation);
        println!("{form} CuH+: +U share of the kernel {share:.3e}");
        assert!(share > MIN_HUBBARD_SHARE, "{form} CuH+: +U share {share:e}");
        check_unrestricted_channel(&integrals, &cation, &format!("{form} CuH+"));
    }
}

#[test]
fn the_hubbard_kernel_has_the_dudarev_antisymmetric_channel() {
    // V1b for +U, against the closed form: the Dudarev potential is u_eff (1/2 - n), so along an
    // imaginary Hermitian change iA of the occupations its change is -u_eff i A, and the A - B
    // kernel is -u_eff P_o A P_v^T with A = P_o^T y P_v - (P_o^T y P_v)^T, P = C^T Q.
    let ctx = Context::new(Some(1), 1 << 30).unwrap();
    let (integrals, electrons) = copper_hydride(&ctx, Some(("dudarev-fll", 20., 0.)));
    let state = integrals
        .kohn_sham("lda", [electrons / 2, electrons / 2], false, &options())
        .unwrap();
    assert!(state.converged);
    let model = integrals
        .corrections
        .as_ref()
        .unwrap()
        .hubbard
        .as_ref()
        .unwrap()
        .clone();
    for spin in [Spin::Singlet, Spin::Triplet] {
        let response = Response::new(&integrals, &state, spin, Coupling::Adiabatic).unwrap();
        let (o, v) = (response.occupied(), response.virtual_orbitals());
        let c = response.orbitals();
        let y = trial(o * v, 0.53);
        let ym = response.matrix(y.as_slice());
        let kernel = response
            .antisymmetric_kernel(&DMatrix::from_column_slice(o * v, 1, y.as_slice()))
            .unwrap();
        let kernel = response.matrix(kernel.column(0).as_slice());
        let mut expected = DMatrix::<f64>::zeros(o, v);
        for shell in &model.shells {
            let (po, pv) = (
                c.columns(0, o).transpose() * &shell.q,
                c.columns(o, v).transpose() * &shell.q,
            );
            let x = po.transpose() * &ym * &pv;
            expected -= &po * (&x - x.transpose()) * pv.transpose() * shell.u_eff();
        }
        println!("{spin:?}: +U antisymmetric kernel {:.3e}", expected.amax());
        assert!(
            expected.amax() > MIN_HUBBARD_KERNEL,
            "{spin:?}: the +U antisymmetric kernel {:e} is too small to test",
            expected.amax()
        );
        assert!(
            (&kernel - &expected).amax() < 1e-12 * expected.amax(),
            "{spin:?}"
        );
    }
}

#[test]
fn hubbard_roots_satisfy_the_dense_and_sum_rule_identities() {
    // V3 and V4 with +U: A - B carries the +U antisymmetric channel, and the sum rule holds with it.
    let ctx = Context::new(Some(1), 1 << 30).unwrap();
    let (integrals, electrons) = copper_hydride(&ctx, Some(("liechtenstein-fll", 20., 3.)));
    let state = integrals
        .kohn_sham("lda", [electrons / 2, electrons / 2], false, &options())
        .unwrap();
    assert!(state.converged);
    let response = Response::new(&integrals, &state, Spin::Singlet, Coupling::Adiabatic).unwrap();
    let n = response.dimension();
    let (sum, difference) = dense(&response);
    let delta = DMatrix::from_diagonal(&DVector::from_column_slice(response.delta()));
    println!(
        "A - B differs from the gap by {:.3e}",
        (&difference - &delta).amax()
    );
    assert!(
        (&difference - &delta).amax() > MIN_HUBBARD_KERNEL,
        "A - B carries the +U term"
    );
    let (tda, _) = real_eigen((&sum + &difference) * 0.5).unwrap();
    let root = symmetric_root(&difference);
    let (squares, _) = real_eigen(&root * &sum * &root).unwrap();
    let found = excitations(
        &integrals,
        &state,
        &request(n, Approximation::TammDancoff, Spin::Singlet),
    )
    .unwrap();
    for (k, e) in found.energies.iter().enumerate() {
        assert!((e - tda[k]).abs() < 1e-10, "TDA root {k}");
    }
    let casida = excitations(
        &integrals,
        &state,
        &request(n, Approximation::Casida, Spin::Singlet),
    )
    .unwrap();
    for (k, e) in casida.energies.iter().enumerate() {
        assert!((e - squares[k].sqrt()).abs() < 1e-10, "Casida root {k}");
    }
    let d = dipole_blocks(&integrals, &response);
    for axis in 0..3 {
        let weighted: f64 = casida
            .energies
            .iter()
            .zip(&casida.transition_dipoles)
            .map(|(w, mu)| w * mu[axis] * mu[axis])
            .sum();
        let exact = 2. * d[axis].dot(&(&difference * &d[axis]));
        assert!(
            (weighted - exact).abs() < 1e-9 * exact.abs().max(1e-8),
            "axis {axis}"
        );
    }
}

/// V14: the roots are invariant under a rigid motion of the molecule, to the grid's egg-box
/// level, measured first and then bounded. Library water (`lda-sr`, LDA, the 12 Bohr cell of
/// [`library_water`], spacing 0.5 Bohr): the lowest five Tamm-Dancoff singlets and their
/// transition moments at the base geometry against the molecule translated (by one grid step
/// along x, and by half a step along x and y), turned by 90 degrees about z through the cell's
/// center (the moments turn with it: `|mu_x|` and `|mu_y|` exchange), and turned by 30 degrees
/// about z through its centroid (the roots only). The molecular grid is centered on the atoms'
/// centroid, so a translation moves it with the molecule and the 90-degree turn maps it onto
/// itself: those three are exact up to rounding, through every angular part (the d shells, the
/// projectors, the one-center terms). The 30-degree turn is the egg-box level, at spacing 0.25
/// Bohr (measured 2026-11-16, the lowest root's change: 2.45e-2, 2.97e-3 and 1.20e-4 Ha at 0.5,
/// 0.35 and 0.25 Bohr; the ground state's 5.2e-3, 1.9e-3 and 9.3e-5 Ha).
#[test]
#[ignore = "slow: 11 s, five water SCFs and their roots, two at spacing 0.25 Bohr; run: cargo test --profile release-test -p mandacaru-core --test molecular_excitations -- --ignored the_roots_are_invariant"]
fn the_roots_are_invariant_under_a_rigid_motion_to_the_grid_level() {
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    let base = [[6.11, 6.07, 6.12], [7.55, 6.04, 5.07], [4.57, 5.95, 5.03]];
    let water = |positions: &[[f64; 3]; 3], spacing: f64| {
        let mut arrays = mandacaru_core::paw::molecular::Arrays::new();
        let oxygen = dataset::species(&ctx, &mut arrays, &dataset::read(&ctx, "lda-sr", "O"));
        let hydrogen = dataset::species(&ctx, &mut arrays, &dataset::read(&ctx, "lda-sr", "H"));
        let integrals = molecule(
            &ctx,
            &[8, 1, 1],
            positions,
            12.,
            spacing,
            (vec![oxygen, hydrogen], vec![0, 1, 1], arrays),
            "raw",
            &analytic_shells(&[0, 1, 2]),
        );
        let state = integrals
            .kohn_sham(
                "lda",
                [4, 4],
                false,
                &ScfOptions {
                    tolerance: 1e-11,
                    ..options()
                },
            )
            .unwrap();
        assert!(state.converged);
        excitations(
            &integrals,
            &state,
            &request(5, Approximation::TammDancoff, Spin::Singlet),
        )
        .unwrap()
    };
    let moved = |shift: [f64; 3]| base.map(|r| [r[0] + shift[0], r[1] + shift[1], r[2] + shift[2]]);
    let turned = base.map(|r| [12. - r[1], r[0], r[2]]);
    let centroid: [f64; 3] = std::array::from_fn(|k| base.iter().map(|r| r[k]).sum::<f64>() / 3.);
    let (c, s) = (30f64.to_radians().cos(), 30f64.to_radians().sin());
    let oblique = base.map(|r| {
        let (x, y) = (r[0] - centroid[0], r[1] - centroid[1]);
        [
            centroid[0] + c * x - s * y,
            centroid[1] + s * x + c * y,
            r[2],
        ]
    });
    let reference = water(&base, 0.5);
    let fine = water(&base, 0.25);
    let mut worst = vec![];
    for (label, positions, swap) in [
        ("one grid step along x", moved([0.5, 0., 0.]), false),
        ("half a step along x and y", moved([0.25, 0.25, 0.]), false),
        ("90 degrees about z", turned, true),
        ("30 degrees about z", oblique, false),
    ] {
        let oblique_case = label.starts_with("30");
        let other = water(&positions, if oblique_case { 0.25 } else { 0.5 });
        let reference = if oblique_case { &fine } else { &reference };
        let roots = reference
            .energies
            .iter()
            .zip(&other.energies)
            .map(|(a, b)| (a - b).abs())
            .fold(0f64, f64::max);
        let moments = reference
            .transition_dipoles
            .iter()
            .zip(&other.transition_dipoles)
            .map(|(a, b)| {
                if label.starts_with("30") {
                    let norm = |v: &[f64; 3]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
                    return (norm(a) - norm(b)).abs();
                }
                let b = if swap { [b[1], b[0], b[2]] } else { *b };
                (0..3)
                    .map(|k| (a[k].abs() - b[k].abs()).abs())
                    .fold(0f64, f64::max)
            })
            .fold(0f64, f64::max);
        println!(
            "V14 water, {label}: roots {:.10?} Ha, largest change {roots:.2e} Ha, of |mu| {moments:.2e} Bohr",
            other.energies
        );
        worst.push((label, roots, moments));
    }
    println!(
        "V14 water: base roots {:.10?} Ha (at 0.25 Bohr {:.10?})",
        reference.energies, fine.energies
    );
    for (k, (label, roots, moments)) in worst.iter().enumerate() {
        if k < 3 {
            assert!(
                *roots < 1e-11 && *moments < 1e-10,
                "{label}: {roots:e} Ha, {moments:e} Bohr"
            );
        } else {
            assert!(
                *roots < 5e-4 && *moments < 1e-2,
                "{label}: {roots:e} Ha, {moments:e} Bohr"
            );
        }
    }
}

/// The transition density as a field (`Excitations::transition_field`, T1): sampled on the
/// integration grid, its moment plus the PAW one-center dipole is the root's transition dipole,
/// its integral plus the one-center charge vanishes (the orbitals are orthogonal in the PAW
/// metric), and `2/3 omega |mu|^2` of that dipole is the root's oscillator strength. Library water
/// (`lda-sr`, a spin-adapted singlet: `sqrt(2) T`), NH2 unrestricted (`T_alpha + T_beta`) and H2 in
/// HAO (all-electron: the field alone, no one-center part).
#[test]
fn the_transition_density_field_carries_the_transition_dipole_and_the_strength() {
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    let check =
        |label: &str, integrals: &MolecularIntegrals, state: &MeanField, request: &Request| {
            let found = excitations(integrals, state, request).unwrap();
            let mut worst = [0f64; 3];
            for root in 0..found.energies.len() {
                let field = found
                    .transition_field(integrals, root, None, &integrals.grid)
                    .unwrap();
                let mu = found.transition_dipoles[root];
                let dipole: [f64; 3] =
                    std::array::from_fn(|i| field.moment[i] + field.augmentation_dipole[i]);
                let size = mu.iter().map(|x| x * x).sum::<f64>().sqrt();
                let miss = (0..3).map(|i| (dipole[i] - mu[i]).abs()).fold(0., f64::max);
                let charge = (field.integral + field.augmentation_charge).abs();
                let strength =
                    2. / 3. * found.energies[root] * dipole.iter().map(|x| x * x).sum::<f64>();
                let f = found.oscillator_strengths[root];
                println!(
                "{label} root {root}: omega {:.6} Ha, |mu| {size:.6e}, dipole miss {miss:.3e}, \
                 field charge {:.3e} + one-center {:.3e} = {charge:.3e}, one-center dipole \
                 {:.3e}, f {f:.6e} against {strength:.6e}, norm {:.4e}",
                found.energies[root],
                field.integral,
                field.augmentation_charge,
                field.augmentation_dipole.iter().map(|x| x.abs()).fold(0., f64::max),
                field.norm
            );
                worst[0] = worst[0].max(miss);
                worst[1] = worst[1].max(charge);
                worst[2] = worst[2].max((strength - f).abs() / f.max(1e-6));
            }
            println!(
                "{label}: worst dipole miss {:.3e} e Bohr, charge {:.3e}, strength {:.3e}",
                worst[0], worst[1], worst[2]
            );
            assert!(
                worst[0] < 1e-10 && worst[1] < 1e-10 && worst[2] < 1e-9,
                "{label}: {worst:?}"
            );
            found
        };
    let water = library_water(&ctx, "lda-sr");
    let state = water.kohn_sham("lda", [4, 4], false, &options()).unwrap();
    assert!(state.converged);
    let singlet = check(
        "water singlet",
        &water,
        &state,
        &request(4, Approximation::Casida, Spin::Singlet),
    );
    assert!(singlet.oscillator_strengths.iter().any(|f| *f > 1e-3));
    let triplet = excitations(
        &water,
        &state,
        &request(2, Approximation::TammDancoff, Spin::Triplet),
    )
    .unwrap();
    assert!(triplet
        .transition_field(&water, 0, None, &water.grid)
        .is_err());
    let spatial = triplet
        .transition_field(&water, 0, Some(0), &water.grid)
        .unwrap();
    assert!((spatial.integral + spatial.augmentation_charge).abs() < 1e-10);
    let amidogen = library_amidogen(&ctx, "lda-sr");
    let open = unrestricted_reference(&amidogen, "lda", [4, 3]);
    check(
        "NH2 unrestricted",
        &amidogen,
        &open,
        &unrestricted_request(3, Approximation::TammDancoff),
    );
    let hydrogen = hydrogen(&ctx);
    let state = reference(&hydrogen, "lda");
    let found = check(
        "H2 HAO",
        &hydrogen,
        &state,
        &request(1, Approximation::TammDancoff, Spin::Singlet),
    );
    let field = found
        .transition_field(&hydrogen, 0, None, &hydrogen.grid)
        .unwrap();
    assert!(field.augmentation_charge == 0. && field.augmentation_dipole == [0.; 3]);
}

/// The one-center exchange-correlation sphere along `(up, down)` by brute force on the radial
/// knots times a Gauss-Legendre angular rule, with the moved spin densities and their gradients
/// from the waves' slopes and the harmonics' Cartesian gradients (no multipole reduction):
/// `(pointwise, difference)`, the second variation `dv^T H dv + sum e_st d2 sigma_st` of the
/// point kernel at every node, and the second difference of the energy `E[n^1] - E[n~^1]` in the
/// step (Richardson of `h` and `h / 2`).
fn brute_force_second_change(
    sphere: &mandacaru_core::paw::molecular::XcSphere,
    functional: &str,
    up: &DMatrix<f64>,
    down: &DMatrix<f64>,
) -> (f64, f64) {
    use mandacaru_core::math::{angular, radial};
    let rule = mandacaru_core::paw::pair_tables::angular_rule(16, 32).unwrap();
    let p = sphere.harmonics.len();
    // At each node: the reference spin densities and gradients, and the changes per unit step.
    let node = |side: usize,
                k: usize,
                u: &[f64; 3]|
     -> ([f64; 2], [[f64; 3]; 2], [f64; 2], [[f64; 3]; 2]) {
        let (waves, slopes) = (&sphere.waves[side], &sphere.wave_slopes[side]);
        let r = sphere.r[k];
        let x = u.map(|c| c * r);
        let y: Vec<f64> = sphere
            .harmonics
            .iter()
            .map(|&(l, m)| angular::cartesian(l, m, *u))
            .collect();
        let dy: Vec<[f64; 3]> = sphere
            .harmonics
            .iter()
            .map(|&(l, m)| angular::cartesian_gradient(l, m, x))
            .collect();
        let rho = sphere.density[side][k];
        let n0 = [0.5 * rho; 2];
        let g0 = [u.map(|c| 0.5 * sphere.slope[side][k] * c); 2];
        let mut dn = [0.; 2];
        let mut dg = [[0.; 3]; 2];
        for (s, d) in [up, down].into_iter().enumerate() {
            for a in 0..p {
                for b in 0..p {
                    let c = d[(a, b)];
                    if c == 0. {
                        continue;
                    }
                    let pair = waves[a][k] * waves[b][k];
                    let radial = slopes[a][k] * waves[b][k] + waves[a][k] * slopes[b][k];
                    dn[s] += c * pair * y[a] * y[b];
                    for i in 0..3 {
                        dg[s][i] += c
                            * (radial * y[a] * y[b] * u[i]
                                + pair * (y[a] * dy[b][i] + y[b] * dy[a][i]));
                    }
                }
            }
        }
        (n0, g0, dn, dg)
    };
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let mut pointwise = 0.;
    for (side, sign) in [(0, 1.), (1, -1.)] {
        let shell: Vec<f64> = (0..sphere.r.len())
            .map(|k| {
                if sphere.density[side][k] < 1e-10 {
                    return 0.;
                }
                let r = sphere.r[k];
                r * r
                    * rule
                        .iter()
                        .map(|(u, w)| {
                            let (n0, g0, dn, dg) = node(side, k, u);
                            let (_, first, second) = mandacaru_core::models::xc::point_hessian(
                                functional,
                                [
                                    n0[0],
                                    n0[1],
                                    dot(g0[0], g0[0]),
                                    dot(g0[0], g0[1]),
                                    dot(g0[1], g0[1]),
                                ],
                                sphere.relativistic,
                            )
                            .unwrap();
                            let dv = [
                                dn[0],
                                dn[1],
                                2. * dot(g0[0], dg[0]),
                                dot(g0[0], dg[1]) + dot(dg[0], g0[1]),
                                2. * dot(g0[1], dg[1]),
                            ];
                            let quadratic: f64 = (0..5)
                                .map(|i| (0..5).map(|j| dv[i] * second[i][j] * dv[j]).sum::<f64>())
                                .sum();
                            let curvature = 2.
                                * (first[2] * dot(dg[0], dg[0])
                                    + first[3] * dot(dg[0], dg[1])
                                    + first[4] * dot(dg[1], dg[1]));
                            w * (quadratic + curvature)
                        })
                        .sum::<f64>()
            })
            .collect();
        pointwise += sign * radial::integrate(&sphere.r, &shell).unwrap();
    }
    let energy = |step: f64| -> f64 {
        let mut total = 0.;
        for (side, sign) in [(0, 1.), (1, -1.)] {
            let (waves, slopes) = (&sphere.waves[side], &sphere.wave_slopes[side]);
            let shell: Vec<f64> = (0..sphere.r.len())
                .map(|k| {
                    let (r, rho) = (sphere.r[k], sphere.density[side][k]);
                    if rho < 1e-10 {
                        return 0.;
                    }
                    let shell: f64 = rule
                        .iter()
                        .map(|(u, w)| {
                            let x = u.map(|c| c * r);
                            let y: Vec<f64> = sphere
                                .harmonics
                                .iter()
                                .map(|&(l, m)| angular::cartesian(l, m, *u))
                                .collect();
                            let dy: Vec<[f64; 3]> = sphere
                                .harmonics
                                .iter()
                                .map(|&(l, m)| angular::cartesian_gradient(l, m, x))
                                .collect();
                            let mut n = [0.5 * rho; 2];
                            let mut g = [u.map(|c| 0.5 * sphere.slope[side][k] * c); 2];
                            for (s, d) in [up, down].into_iter().enumerate() {
                                for a in 0..p {
                                    for b in 0..p {
                                        let c = step * d[(a, b)];
                                        if c == 0. {
                                            continue;
                                        }
                                        let pair = waves[a][k] * waves[b][k];
                                        let radial =
                                            slopes[a][k] * waves[b][k] + waves[a][k] * slopes[b][k];
                                        n[s] += c * pair * y[a] * y[b];
                                        for i in 0..3 {
                                            g[s][i] += c
                                                * (radial * y[a] * y[b] * u[i]
                                                    + pair * (y[a] * dy[b][i] + y[b] * dy[a][i]));
                                        }
                                    }
                                }
                            }
                            let dot =
                                |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
                            let (e, _) = mandacaru_core::models::xc::point_partials(
                                functional,
                                [
                                    n[0],
                                    n[1],
                                    dot(g[0], g[0]),
                                    dot(g[0], g[1]),
                                    dot(g[1], g[1]),
                                    0.,
                                    0.,
                                ],
                                sphere.relativistic,
                            )
                            .unwrap();
                            w * e
                        })
                        .sum();
                    r * r * shell
                })
                .collect();
            total += sign * radial::integrate(&sphere.r, &shell).unwrap();
        }
        total
    };
    let center = energy(0.);
    let second = |h: f64| (energy(h) - 2. * center + energy(-h)) / (h * h);
    let h = 8e-3;
    (pointwise, (4. * second(h / 2.) - second(h)) / 3.)
}

/// The one-center exchange-correlation part of the diagnostic (`Excitations::one_center_xc`,
/// T1, plan section 3): on library water the multipole-reduced second change of each sphere
/// (`XcSpheres::second_change`) equals the brute-force angular quadrature of the energy's second
/// difference, for a singlet's, a triplet's and an unequal pair of spin changes, with LDA and
/// PBE; and for LDA the first change at the reference is the dataset's linearized one-center
/// potential (`Channel::xc`): the reference densities carry the core and the relativistic
/// factor. The shifts are printed beside the one-center Hartree part.
#[test]
#[ignore = "slow: 28 s (measured 2026-11-16, the brute-force angular quadrature); run: cargo test --profile release-test -p mandacaru-core --test molecular_excitations -- --ignored the_one_center_exchange_correlation_shift"]
fn the_one_center_exchange_correlation_shift_is_the_second_change_of_the_sphere_energy() {
    use mandacaru_core::math::radial::integrate;
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    for (set, functional) in [("lda-sr", "lda"), ("pbe-sr", "pbe")] {
        let water = library_water(&ctx, set);
        let state = water
            .kohn_sham(functional, [4, 4], false, &options())
            .unwrap();
        assert!(state.converged);
        let setup = water.paw_setup.as_ref().unwrap();
        let spheres = setup.xc_spheres(functional).unwrap();
        let projections = water
            .corrections
            .as_ref()
            .unwrap()
            .projections
            .as_ref()
            .unwrap();
        if functional == "lda" {
            for atom in 0..2 {
                let sphere = spheres.sphere(atom).unwrap();
                let p = sphere.harmonics.len();
                let mut worst = 0f64;
                for a in 0..p {
                    for b in 0..p {
                        if sphere.harmonics[a] != sphere.harmonics[b] {
                            continue;
                        }
                        let mut value = 0.;
                        for (side, sign) in [(0, 1.), (1, -1.)] {
                            let integrand: Vec<f64> = (0..sphere.r.len())
                                .map(|k| {
                                    let rho = sphere.density[side][k];
                                    if rho < 1e-10 {
                                        return 0.;
                                    }
                                    let (_, g) = mandacaru_core::models::xc::point_partials(
                                        "lda",
                                        [0.5 * rho, 0.5 * rho, 0., 0., 0., 0., 0.],
                                        sphere.relativistic,
                                    )
                                    .unwrap();
                                    g[0] * sphere.waves[side][a][k]
                                        * sphere.waves[side][b][k]
                                        * sphere.r[k]
                                        * sphere.r[k]
                                })
                                .collect();
                            value += sign * integrate(&sphere.r, &integrand).unwrap();
                        }
                        worst = worst.max((value - sphere.linearized[(a, b)]).abs());
                    }
                }
                println!(
                    "{set} atom {atom}: first change at the reference against Channel::xc, worst {worst:.3e} \
                     (largest {:.3e})",
                    sphere.linearized.amax()
                );
                assert!(worst < 1e-6 * sphere.linearized.amax(), "{worst:e}");
            }
        }
        let singlet = excitations(
            &water,
            &state,
            &request(3, Approximation::TammDancoff, Spin::Singlet),
        )
        .unwrap();
        let t = singlet.transition_density(0, 0).unwrap();
        let half = std::f64::consts::FRAC_1_SQRT_2;
        let other = singlet.transition_density(1, 0).unwrap();
        for atom in 0..2 {
            let sphere = spheres.sphere(atom).unwrap();
            let c = &projections[atom];
            let d = c.transpose() * &t * c * half;
            let e = c.transpose() * &other * c;
            for (label, up, down) in [
                ("singlet", d.clone(), d.clone()),
                ("triplet", d.clone(), -&d),
                ("unequal", d.clone(), e.clone()),
            ] {
                let reduced = spheres.second_change(atom, &up, &down).unwrap();
                let (brute, difference) = brute_force_second_change(sphere, functional, &up, &down);
                println!(
                    "{set} atom {atom} {label}: reduced {reduced:.10e}, pointwise {brute:.10e} \
                     (relative {:.3e}), energy difference {difference:.10e} (relative {:.3e})",
                    (reduced - brute) / brute.abs(),
                    (reduced - difference) / brute.abs()
                );
                assert!(
                    (reduced - brute).abs() < 1e-10 * brute.abs(),
                    "{set} {atom} {label}"
                );
                assert!(
                    (reduced - difference).abs() < 1e-4 * brute.abs(),
                    "{set} {atom} {label}"
                );
            }
        }
        let triplet = excitations(
            &water,
            &state,
            &request(3, Approximation::TammDancoff, Spin::Triplet),
        )
        .unwrap();
        for (spin, found) in [("singlet", &singlet), ("triplet", &triplet)] {
            println!(
                "{set} {spin}: omega {:?} Ha, one-center Hartree {:?}, exchange-correlation {:?} Ha",
                found.energies,
                found.one_center_hartree,
                found.one_center_xc
            );
            assert!(found
                .one_center_xc
                .as_ref()
                .unwrap()
                .iter()
                .all(|s| s.is_finite() && *s != 0.));
        }
    }
    // Off PAW-LCAO Kohn-Sham there is none.
    let hydrogen = hydrogen(&ctx);
    let state = reference(&hydrogen, "lda");
    assert!(excitations(
        &hydrogen,
        &state,
        &request(1, Approximation::TammDancoff, Spin::Singlet)
    )
    .unwrap()
    .one_center_xc
    .is_none());
}

/// V13, the one-electron limit (T4): with one electron CIS is full configuration interaction and
/// TDHF adds nothing (`J - K` of the electron with itself cancels), so on H2+ run unrestricted
/// Hartree-Fock both give every root as an eigenvalue of the bare one-electron operator the SCF
/// diagonalizes (`MolecularIntegrals::one`, orthonormal frame) less its lowest. Synthetic PAW H2+
/// and H2+ in HAO.
#[test]
fn one_electron_cis_and_tdhf_roots_are_the_one_electron_spectrum() {
    let ctx = Context::new(Some(1), 256 << 20).unwrap();
    for (label, integrals) in [
        ("PAW", paw_hydrogen(&ctx, &[0, 1])),
        ("HAO", hydrogen(&ctx)),
    ] {
        let m = integrals.m();
        let one = DMatrix::from_row_slice(m, m, integrals.one.as_slice());
        let (levels, _) = real_eigen((&one + one.transpose()) * 0.5).unwrap();
        let state = unrestricted_reference(&integrals, "uhf", [1, 0]);
        for approximation in [Approximation::TammDancoff, Approximation::Casida] {
            let found = excitations(
                &integrals,
                &state,
                &unrestricted_request(m - 1, approximation),
            )
            .unwrap();
            let worst = found
                .energies
                .iter()
                .zip(&levels[1..])
                .map(|(w, e)| (w - (e - levels[0])).abs())
                .fold(0f64, f64::max);
            println!(
                "V13 {label} H2+ {}: {} roots, lowest {:.10} Ha against {:.10}, worst miss {worst:.2e}",
                approximation.name(),
                found.energies.len(),
                found.energies[0],
                levels[1] - levels[0]
            );
            assert!(worst < 1e-8, "{label} {approximation:?}: {worst:e}");
        }
    }
}
