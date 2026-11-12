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
            (finite_field - alpha).abs() < 1e-4 * alpha.abs(),
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
    for (name, word) in [("r2scan", "meta-GGA")] {
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
    let integrals = library_amidogen(&ctx, "lda-sr");
    for method in ["lda", "pbe", "uhf"] {
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
            (finite_field - alpha).abs() < 1e-4 * alpha.abs(),
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
    // V12: the same state, one and four threads.
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
