//! Excited-state energies and analytic gradients of closed-shell Kohn-Sham PAW-LCAO molecules
//! (`molecular::excitations::excited_state`, `dev_plan/TDDFT_PLAN.md` T7), checked by the
//! engine's own numbers:
//!
//! - the third-order change of the XC potential is the derivative of the kernel's action;
//! - the first-order Fock change of a general symmetric density (occupied-occupied, virtual-virtual
//!   and occupied-virtual blocks) is the derivative of the reference's Fock builder;
//! - the fixed-density contraction at the reference's own densities is the ground-state force, at a
//!   general density the derivative of the energy at fixed raw-AO matrices, and exactly quadratic
//!   in the density term by term;
//! - the relaxed density `Delta + Z` is the field derivative of `omega`;
//! - each explicit part (relaxed, transition, overlap) against nested central differences;
//! - the analytic gradient of `E_0 + omega` against a Richardson central difference of
//!   independently converged excited-state energies on the same fixed grid, on the synthetic
//!   species and on the library's lithium (a partial core);
//! - tracking by transition-density overlap follows a state through a crossing that tracking by
//!   index jumps;
//! - DFT+U (Dudarev and Liechtenstein on the library's copper 3d): the first-order and imaginary
//!   Fock of the shells are the builder's, the explicit derivative along `i T'` is the derivative
//!   of the shells' second change, and the gradient follows reconverged energies;
//! - HSE06: the coupled screened exchange's gradient is its derivative at fixed orbitals, and the
//!   gradient follows reconverged energies on the library's LiH (`pbe-sr`), closed shell and
//!   unrestricted;
//! - r2SCAN: the third-order change of its three potential parts (with the kinetic curvature of
//!   an imaginary change) is the derivative of its kernel, the current-density imaginary Fock is
//!   the complex builder's, and the gradient follows reconverged energies on the synthetic H4 and
//!   the library's LiH (`pbe-sr`), closed shell and unrestricted.
mod paw_forces_support;
mod paw_molecular_support;

use mandacaru_core::{
    math::SpatialGrid,
    models::xc_kernel::{
        second_change, second_change_meta, Kernel, MetaChange, MetaKernel, NoncollinearKernel,
    },
    molecular::{
        excitations::{
            excitations, excited_state, followed_excited_state, gradient::seams, pick,
            reference_fock, Approximation, Request, Spin,
        },
        hubbard::{MolecularHubbard, MolecularHubbardShell},
        multipoles::position_integrals,
        MeanField, MolecularIntegrals,
    },
    paw::molecular::gradient::hubbard_imaginary_force,
    quantum::C64,
    Context, Geometry, LengthUnit,
};
use nalgebra::DMatrix;
use paw_forces_support::synthetic::{displaced, integrals, options, solve, Method, H2};

fn request(spin: Spin, approximation: Approximation) -> Request {
    let mut r = Request::new(3);
    r.approximation = approximation;
    r.spin = Some(spin);
    r.tolerance = 1e-10;
    r.max_iterations = 400;
    r
}
/// A bent H3+ (Bohr): no symmetry, every component of the gradient nonzero.
const H3: [[f64; 3]; 3] = [[0., 0., 0.], [1.7, 0., 0.3], [0.4, 1.6, -0.5]];
/// An irregular H4 (Bohr), two occupied orbitals: the occupied-occupied blocks are not 1 x 1.
const H4: [[f64; 3]; 4] = [
    [0., 0., 0.],
    [1.5, 0., 0.2],
    [3.1, 0.3, 0.],
    [4.4, 1.2, 0.4],
];

#[test]
fn the_third_order_change_is_the_derivative_of_the_kernel() {
    third_order_check(&[
        ("lda", 1., false),
        ("lda", -1., false),
        ("pbe", 1., false),
        ("pbe", -1., false),
        ("lda", 1., true),
        ("pbe", -1., true),
    ]);
}

#[test]
#[ignore = "slow (about 75 s with 3 threads: the hole model's quadrature at every point of four kernels per case): run: cargo test --profile release-test -p mandacaru-core --test molecular_excited_forces -- --ignored the_hse06_third_order_change"]
fn the_hse06_third_order_change_is_the_derivative_of_its_kernel() {
    third_order_check(&[("hse06", 1., false), ("hse06", -1., false)]);
}

fn third_order_check(cases: &[(&str, f64, bool)]) {
    let ctx = Context::new(Some(2), 256 << 20).unwrap();
    {
        let geometry = Geometry::new(
            vec![1],
            vec![[3., 3., 3.]],
            [[6., 0., 0.], [0., 6., 0.], [0., 0., 6.]],
            [false; 3],
            LengthUnit::Bohr,
        )
        .unwrap();
        let grid = SpatialGrid::from_geometry(&geometry, [0.3; 3], false, false).unwrap();
        let n = grid.index.points();
        let shape = |k: usize, center: [f64; 3], width: f64| {
            let r = grid.point(k);
            let d2: f64 = (0..3).map(|a| (r[a] - center[a]).powi(2)).sum();
            (-d2 / width).exp()
        };
        let mut rho = vec![0.; 2 * n];
        for k in 0..n {
            rho[k] = 0.3 * shape(k, [3., 3., 3.], 1.2) + 1e-3;
            rho[n + k] = 0.2 * shape(k, [3.2, 2.9, 3.1], 1.0) + 1e-3;
        }
        for &(name, sign, relativistic) in cases {
            let mut delta = vec![0.; 2 * n];
            for k in 0..n {
                delta[k] = 0.01 * shape(k, [2.6, 3.3, 3.], 0.8);
                delta[n + k] = sign * 0.007 * shape(k, [3.4, 3., 2.8], 0.9);
            }
            let c2 = second_change(&ctx, &grid, &rho, name, relativistic, &delta).unwrap();
            let apply = |h: f64| {
                let shifted: Vec<f64> = (0..2 * n).map(|k| rho[k] + h * delta[k]).collect();
                let kernel = Kernel::new(&ctx, &grid, &shifted, name, relativistic).unwrap();
                let mut out = vec![0.; 2 * n];
                let mut scratch = kernel.scratch(&ctx).unwrap();
                kernel.apply(&grid, &delta, &mut out, &mut scratch).unwrap();
                out
            };
            // dv/dt at t = +-h is v1 +- 2 h c2.
            let fd = |h: f64| -> Vec<f64> {
                let (a, b) = (apply(h), apply(-h));
                (0..2 * n).map(|k| (a[k] - b[k]) / (4. * h)).collect()
            };
            let (coarse, fine) = (fd(2e-3), fd(1e-3));
            let scale = c2.as_slice().iter().fold(0f64, |a, v| a.max(v.abs()));
            let worst = (0..2 * n)
                .map(|k| ((4. * fine[k] - coarse[k]) / 3. - c2.as_slice()[k]).abs())
                .fold(0f64, f64::max);
            println!("{name} (sign {sign}, relativistic {relativistic}): max |c2 - FD| = {worst:.3e} of max |c2| = {scale:.3e}");
            assert!(
                scale > 0. && worst < 1e-6 * scale,
                "{name}: {worst:.3e} of {scale:.3e}"
            );
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

/// `C^T F C` of the reference's Fock builder at the frame densities `d +- h C V_s C^T`.
fn fock_derivative(
    integrals: &MolecularIntegrals,
    state: &MeanField,
    frame: &DMatrix<f64>,
    v: [&DMatrix<f64>; 2],
) -> [DMatrix<f64>; 2] {
    let d = &state.densities[0];
    let change = |s: usize| (frame * v[s] * frame.transpose()).map(|x| C64::new(x, 0.));
    let fock = |h: f64| {
        let f = reference_fock(
            integrals,
            state,
            &[
                d + change(0) * C64::new(h, 0.),
                d + change(1) * C64::new(h, 0.),
            ],
        )
        .unwrap();
        [0, 1].map(|s| frame.transpose() * f[s].map(|z| z.re) * frame)
    };
    let central = |h: f64| {
        let (a, b) = (fock(h), fock(-h));
        [0, 1].map(|s| (&a[s] - &b[s]) / (2. * h))
    };
    let (coarse, fine) = (central(2e-4), central(1e-4));
    [0, 1].map(|s| (&fine[s] * 4. - &coarse[s]) / 3.)
}

#[test]
#[ignore = "slow (11 s with 3 threads, timed alone 2026-11-15): run: cargo test --profile release-test -p mandacaru-core --test molecular_excited_forces -- --ignored the_first_order_fock_and_the_ground_contraction_are_the_builders_own"]
fn the_first_order_fock_and_the_ground_contraction_are_the_builders_own() {
    let ctx = Context::new(Some(2), 512 << 20).unwrap();
    {
        for functional in ["lda", "pbe"] {
            let base = integrals(&ctx, &H3, functional);
            let state = solve(&base, Method::Ks(functional), [1, 1]);
            // The contraction at the reference's densities is its force.
            let fixed = seams::ground_fixed(&base, &state).unwrap();
            let force = base.gradient(&state).unwrap();
            let worst = fixed
                .iter()
                .zip(force.total.as_slice())
                .map(|(a, b)| (a - b).abs())
                .fold(0f64, f64::max);
            println!("{functional}: |G_fixed - G_ground| = {worst:.3e} Ha/Bohr");
            assert!(worst < 1e-12, "{functional}: {worst:.3e}");
            // A change with occupied-virtual and virtual-virtual blocks. The Fock builder floors the
            // density at zero, so the finite difference is exact only where `D + h V` stays positive:
            // the virtual-virtual block is positive semidefinite and enters with the same sign in both
            // spins; the spin-flip (opposite) change keeps the occupied-virtual block only.
            let m = base.m();
            let ov = DMatrix::from_fn(m, m, |p, q| {
                if (p == 0) != (q == 0) {
                    ((p * 7 + q * 3) as f64 * 0.37).sin() * 0.05
                } else {
                    0.
                }
            });
            let u = DMatrix::from_fn(m, 1, |p, _| {
                if p == 0 {
                    0.
                } else {
                    ((p * 5) as f64 * 0.23).cos() * 0.1
                }
            });
            let full = &ov + &u * u.transpose();
            for (label, up, down) in [
                ("same", full.clone(), full.clone()),
                ("opposite", ov.clone(), -&ov),
            ] {
                let (analytic, _, frame) =
                    seams::first_and_second(&base, &state, [&up, &down]).unwrap();
                let numeric = fock_derivative(&base, &state, &frame, [&up, &down]);
                let scale = analytic[0].amax();
                let worst = (0..2)
                    .map(|s| (&analytic[s] - &numeric[s]).amax())
                    .fold(0f64, f64::max);
                println!("{functional} ({label}): |F1 - dF| = {worst:.3e} of {scale:.3e}");
                // The builder's density floors (2e-12 in the exchange, 1e-10 for the gradient terms)
                // switch points on and off under +-h: the difference is good to about 1e-6 relative
                // (measured 1.2e-6), while the field gate checks the same kernel to 1e-8.
                assert!(worst < 5e-6 * scale, "{functional} {label}: {worst:.3e}");
            }
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
#[ignore = "slow (67 s with 3 threads): run: cargo test --profile release-test -p mandacaru-core --test molecular_excited_forces -- --ignored the_fixed_density_contraction_is_the_derivative_at_fixed_atomic_densities"]
fn the_fixed_density_contraction_is_the_derivative_at_fixed_atomic_densities() {
    // At a general (non-stationary) density D and weight W, the contraction is
    // d/dR [E_KS(D_raw) - tr(W_raw S)] + d E_nn / dR at fixed raw-AO matrices; the displaced
    // frame density is L^-1 D_raw L^-T (L^T S L = 1, so S = (L L^T)^-1).
    let ctx = Context::new(Some(2), 512 << 20).unwrap();
    {
        for functional in ["lda", "pbe"] {
            let base = integrals(&ctx, &H3, functional);
            let state = solve(&base, Method::Ks(functional), [1, 1]);
            let m = base.m();
            let mut d0 = DMatrix::zeros(m, m);
            d0[(0, 0)] = 1.;
            let bend = DMatrix::from_fn(m, m, |p, q| {
                ((p * 5 + q * 2) as f64 * 0.41).cos() * if p == 0 || q == 0 { 0.003 } else { 0. }
            });
            let d = &d0 + (&bend + bend.transpose()) * 0.5;
            let w = DMatrix::from_fn(m, m, |p, q| ((p + q * 3) as f64 * 0.29).sin() * 0.2);
            let w = (&w + w.transpose()) * 0.5;
            let (_, raw) = seams::fixed_at(&base, &state, [&d0, &d0], None).unwrap();
            let inverse = raw.clone().try_inverse().unwrap();
            let w0 = &inverse * base.energy_weighted_density(&state).unwrap() * inverse.transpose();
            println!(
                "lowdin {:?}, ground |G| {:.3e}",
                base.lowdin.shape(),
                base.gradient(&state).unwrap().total.as_slice()[0]
            );
            for (label, d, weight) in [
                ("ground", &d0, Some(&w0)),
                ("density", &d, None),
                ("density and W", &d, Some(&w)),
            ] {
                let (analytic, raw) = seams::fixed_at(&base, &state, [d, d], weight).unwrap();
                let d_raw = &raw * d * raw.transpose();
                let w_raw = weight.map(|w| &raw * w * raw.transpose());
                for (atom, axis) in [(0, 0), (1, 2), (2, 1)] {
                    let value = |delta: f64| {
                        let moved = displaced(&base, atom, axis, delta);
                        let l = moved.lowdin.clone().try_inverse().unwrap();
                        let frame = (&l * &d_raw * l.transpose()).map(|x| C64::new(x, 0.));
                        let mut e = moved
                            .kohn_sham_energy(functional, &[frame.clone(), frame])
                            .unwrap()
                            + moved.nuclear;
                        if let Some(w_raw) = &w_raw {
                            let s = (&moved.lowdin * moved.lowdin.transpose())
                                .try_inverse()
                                .unwrap();
                            e -= w_raw.dot(&s);
                        }
                        e
                    };
                    let h = 1e-3;
                    let coarse = (value(h) - value(-h)) / (2. * h);
                    let fine = (value(0.5 * h) - value(-0.5 * h)) / h;
                    let numeric = (4. * fine - coarse) / 3.;
                    let a = analytic[3 * atom + axis];
                    println!("{functional} {label} ({atom}, {axis}): contraction {a:.10} FD {numeric:.10} diff {:.2e}", a - numeric);
                    assert!(
                        (a - numeric).abs() < 1e-7,
                        "{functional} {label}: {a} against {numeric}"
                    );
                }
            }
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
#[ignore = "slow (10 s with 3 threads, timed alone 2026-11-15): run: cargo test --profile release-test -p mandacaru-core --test molecular_excited_forces -- --ignored each_explicit_part_is_its_derivative_at_fixed_atomic_densities"]
fn each_explicit_part_is_its_derivative_at_fixed_atomic_densities() {
    // The relaxed, transition and overlap parts against nested central differences of the
    // energy at fixed raw-AO matrices (the frame density at a displaced geometry is L^-1 D L^-T).
    let ctx = Context::new(Some(2), 512 << 20).unwrap();
    {
        let functional = "lda";
        let base = integrals(&ctx, &H2, functional);
        let state = solve(&base, Method::Ks(functional), [1, 1]);
        let excited = excited_state(
            &base,
            &state,
            &request(Spin::Singlet, Approximation::TammDancoff),
            0,
        )
        .unwrap();
        let l = &base.lowdin;
        let raw = |frame: &DMatrix<f64>| l * frame * l.transpose();
        let d0 = raw(&state.densities[0].map(|z| z.re));
        let p = raw(&(&excited.relaxed * 0.5));
        let t = raw(&excited.transition[0]);
        let w = raw(&excited.energy_weighted);
        let energy = |moved: &MolecularIntegrals, d: &DMatrix<f64>| {
            let inverse = moved.lowdin.clone().try_inverse().unwrap();
            let frame = (&inverse * d * inverse.transpose()).map(|x| C64::new(x, 0.));
            moved
                .kohn_sham_energy(functional, &[frame.clone(), frame])
                .unwrap()
        };
        let (atom, axis) = (1, 2);
        let part =
            |name: &str| excited.parts.iter().find(|(n, _)| *n == name).unwrap().1[3 * atom + axis];
        let derivative = |value: &dyn Fn(f64) -> f64| {
            let h = 1e-3;
            let coarse = (value(h) - value(-h)) / (2. * h);
            let fine = (value(0.5 * h) - value(-0.5 * h)) / h;
            (4. * fine - coarse) / 3.
        };
        let s = 1e-3;
        let relaxed = derivative(&|delta| {
            let moved = displaced(&base, atom, axis, delta);
            (energy(&moved, &(&d0 + &p * s)) - energy(&moved, &(&d0 - &p * s))) / (2. * s)
        });
        let transition = derivative(&|delta| {
            let moved = displaced(&base, atom, axis, delta);
            (energy(&moved, &(&d0 + &t * s)) + energy(&moved, &(&d0 - &t * s))
                - 2. * energy(&moved, &d0))
                / (s * s)
        });
        let overlap = derivative(&|delta| {
            let moved = displaced(&base, atom, axis, delta);
            -w.dot(
                &(&moved.lowdin * moved.lowdin.transpose())
                    .try_inverse()
                    .unwrap(),
            )
        });
        println!("relaxed: analytic {:.10} FD {relaxed:.10}", part("relaxed"));
        println!(
            "transition: analytic {:.10} FD {transition:.10}",
            part("transition")
        );
        println!("overlap: analytic {:.10} FD {overlap:.10}", part("overlap"));
        assert!((part("relaxed") - relaxed).abs() < 1e-7);
        assert!((part("overlap") - overlap).abs() < 1e-7);
        // The nested second difference of the floored energy is the weakest oracle (2e-5 measured);
        // the whole gradient's gate above is exact.
        assert!((part("transition") - transition).abs() < 1e-4);
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn the_contraction_is_quadratic_in_the_density() {
    let ctx = Context::new(Some(2), 512 << 20).unwrap();
    {
        let base = integrals(&ctx, &H2, "lda");
        let state = solve(&base, Method::Ks("lda"), [1, 1]);
        let m = base.m();
        let mut d0 = DMatrix::zeros(m, m);
        d0[(0, 0)] = 1.;
        let t = DMatrix::from_fn(m, m, |p, q| {
            if (p == 0) != (q == 0) {
                0.1 * ((p + q) as f64).sin()
            } else {
                0.
            }
        });
        let g = |k: f64, v: Option<&[f64]>| {
            seams::fixed_with(&base, &state, [&(&d0 + &t * k), &(&d0 + &t * k)], v).unwrap()
        };
        let (_, v0) = g(0., None);
        let at = |k: f64| g(k, Some(&v0)).0;
        let (a, b, c, d) = (at(2.), at(1.), at(0.), at(-1.));
        for (k, (name, _)) in a.iter().enumerate() {
            let third: Vec<f64> = (0..a[k].1.len())
                .map(|i| a[k].1[i] - 3. * b[k].1[i] + 3. * c[k].1[i] - d[k].1[i])
                .collect();
            let worst = third.iter().fold(0f64, |x, v| x.max(v.abs()));
            println!("third difference of {name}: {worst:.3e}");
            assert!(worst < 1e-11, "{name}: {worst:.3e}");
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

/// `E_0 + omega_root` of independently converged states at a displaced atom.
fn excited_energy(
    base: &MolecularIntegrals,
    functional: &'static str,
    spin: Spin,
    approximation: Approximation,
    root: usize,
    atom: usize,
    axis: usize,
    delta: f64,
) -> f64 {
    let moved = displaced(base, atom, axis, delta);
    let state = solve(&moved, Method::Ks(functional), occupied(base));
    let found = excitations(&moved, &state, &request(spin, approximation)).unwrap();
    state.free_energy() + moved.nuclear + found.energies[root]
}

/// A closed shell of one electron per hydrogen atom, minus the H3+ charge.
fn occupied(base: &MolecularIntegrals) -> [usize; 2] {
    let pairs = base.geometry.numbers().len() / 2;
    [pairs, pairs]
}
fn check(
    label: &str,
    base: &MolecularIntegrals,
    functional: &'static str,
    spin: Spin,
    approximation: Approximation,
    components: &[(usize, usize)],
) {
    let state = solve(base, Method::Ks(functional), occupied(base));
    let excited = excited_state(base, &state, &request(spin, approximation), 0).unwrap();
    let mut worst = 0f64;
    for &(atom, axis) in components {
        let e = |d: f64| excited_energy(base, functional, spin, approximation, 0, atom, axis, d);
        let h = 1e-3;
        let coarse = (e(h) - e(-h)) / (2. * h);
        let fine = (e(0.5 * h) - e(-0.5 * h)) / h;
        let reference = (4. * fine - coarse) / 3.;
        let analytic = excited.gradient.total.as_slice()[3 * atom + axis];
        println!(
            "{label} ({atom}, {axis}): analytic {analytic:.10} FD {reference:.10} diff {:.2e}",
            analytic - reference
        );
        worst = worst.max((analytic - reference).abs());
    }
    for (name, values) in &excited.parts {
        println!(
            "  {name}: {:?}",
            components
                .iter()
                .map(|&(a, x)| values[3 * a + x])
                .collect::<Vec<_>>()
        );
    }
    println!(
        "{label}: omega {:.8} Ha, separation {:.2e}, Z residual {:.1e} in {} iterations, max |dE| {worst:.3e} Ha/Bohr",
        excited.excitation, excited.separation, excited.z_residual, excited.z_iterations
    );
    // 1e-5 eV/A, the ground-state force gates' tolerance.
    assert!(worst < 2e-7, "{label}: {worst:.3e} Ha/Bohr");
}

#[test]
#[ignore = "slow (80 s with 3 threads): run: cargo test --profile release-test -p mandacaru-core --test molecular_excited_forces -- --ignored the_relaxed_density_is_the_field_derivative_of_the_excitation"]
fn the_relaxed_density_is_the_field_derivative_of_the_excitation() {
    // d omega / dF of a field term F z in the one-body operator (test-only) is tr[(Delta + Z) z].
    let ctx = Context::new(Some(2), 512 << 20).unwrap();
    {
        for (functional, spin, approximation, geometry, particles) in [
            (
                "lda",
                Spin::Singlet,
                Approximation::TammDancoff,
                &H3[..],
                [1, 1],
            ),
            (
                "pbe",
                Spin::Triplet,
                Approximation::TammDancoff,
                &H3[..],
                [1, 1],
            ),
            ("lda", Spin::Singlet, Approximation::Casida, &H3[..], [1, 1]),
            (
                "lda",
                Spin::Singlet,
                Approximation::TammDancoff,
                &H4[..],
                [2, 2],
            ),
            ("pbe", Spin::Triplet, Approximation::Casida, &H4[..], [2, 2]),
        ] {
            let mut base = integrals(&ctx, geometry, functional);
            let state = solve(&base, Method::Ks(functional), particles);
            let excited = excited_state(&base, &state, &request(spin, approximation), 0).unwrap();
            let m = base.m();
            for axis in [0, 2] {
                let z = position_integrals(&base, [0.; 3]).unwrap()[axis].clone();
                let relaxed = excited.relaxed.dot(&z);
                let original: Vec<f64> = base.one.as_slice().to_vec();
                let mut omega = |field: f64| {
                    for a in 0..m * m {
                        base.one.as_mut_slice()[a] = original[a] + field * z.as_slice()[a];
                    }
                    let state = solve(&base, Method::Ks(functional), particles);
                    excitations(&base, &state, &request(spin, approximation))
                        .unwrap()
                        .energies[0]
                };
                let slope =
                    |f: f64, omega: &mut dyn FnMut(f64) -> f64| (omega(f) - omega(-f)) / (2. * f);
                let (coarse, fine) = (slope(2e-3, &mut omega), slope(1e-3, &mut omega));
                let numeric = (4. * fine - coarse) / 3.;
                base.one.as_mut_slice().copy_from_slice(&original);
                println!("{} atoms {functional} {} {} axis {axis}: tr[P z] {relaxed:.10} d omega/dF {numeric:.10} diff {:.2e}", geometry.len(), spin.name(), approximation.name(), relaxed - numeric);
                assert!(
                    (relaxed - numeric).abs() < 1e-7,
                    "{functional} axis {axis}: {relaxed} against {numeric}"
                );
            }
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
#[ignore = "slow (63 s with 3 threads): run: cargo test --profile release-test -p mandacaru-core --test molecular_excited_forces -- --ignored the_excited_state_gradient_follows_the_excited_state_energy"]
fn the_excited_state_gradient_follows_the_excited_state_energy() {
    let ctx = Context::new(Some(2), 512 << 20).unwrap();
    {
        let base = integrals(&ctx, &H2, "lda");
        check(
            "H2 LDA singlet TDA",
            &base,
            "lda",
            Spin::Singlet,
            Approximation::TammDancoff,
            &[(1, 2)],
        );
        check(
            "H2 LDA singlet Casida",
            &base,
            "lda",
            Spin::Singlet,
            Approximation::Casida,
            &[(1, 2)],
        );
        let base = integrals(&ctx, &H4, "lda");
        check(
            "H4 LDA singlet TDA",
            &base,
            "lda",
            Spin::Singlet,
            Approximation::TammDancoff,
            &[(1, 0), (2, 2)],
        );
        check(
            "H4 LDA triplet Casida",
            &base,
            "lda",
            Spin::Triplet,
            Approximation::Casida,
            &[(3, 1)],
        );
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
#[ignore = "slow: run: cargo test --profile release-test -p mandacaru-core --test molecular_excited_forces -- --ignored the_excited_state_gradients_of_h3_and_h2_follow_their_energies"]
fn the_excited_state_gradients_of_h3_and_h2_follow_their_energies() {
    let ctx = Context::new(Some(2), 512 << 20).unwrap();
    {
        // PBE on H3+: the SCF at a displaced geometry does not reach the fixture's 1e-12 in 400
        // iterations (measured), so PBE runs on H2's bond component.
        for (functional, geometry, components) in [
            ("lda", &H3[..], &[(0, 0), (1, 2), (2, 1)][..]),
            ("pbe", &H2[..], &[(1, 2)][..]),
        ] {
            let base = integrals(&ctx, geometry, functional);
            for (spin, approximation) in [
                (Spin::Singlet, Approximation::TammDancoff),
                (Spin::Triplet, Approximation::TammDancoff),
                (Spin::Singlet, Approximation::Casida),
                (Spin::Triplet, Approximation::Casida),
            ] {
                check(
                    &format!(
                        "{} atoms {functional} {} {}",
                        geometry.len(),
                        spin.name(),
                        approximation.name()
                    ),
                    &base,
                    functional,
                    spin,
                    approximation,
                    components,
                );
            }
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

/// Two perpendicular H2 (Bohr): A along z at y = -2.5 with bond `a`, B along x at y = +2.5 with
/// bond 1.45. Their local sigma -> sigma* singlets are odd under z -> -z (A) and under x -> -x
/// (B), so they do not mix, and A's crosses B's where `a` passes 1.45.
fn crossing_pair(a: f64) -> [[f64; 3]; 4] {
    [
        [0., -2.5, -0.5 * a],
        [0., -2.5, 0.5 * a],
        [-0.725, 2.5, 0.],
        [0.725, 2.5, 0.],
    ]
}

#[test]
#[ignore = "slow (91 s with 3 threads): run: cargo test --profile release-test -p mandacaru-core --test molecular_excited_forces -- --ignored overlap_tracking_follows_an_excited_state_through_a_crossing"]
fn overlap_tracking_follows_an_excited_state_through_a_crossing() {
    let ctx = Context::new(Some(2), 512 << 20).unwrap();
    {
        let singlet = request(Spin::Singlet, Approximation::TammDancoff);
        let mut followed = None;
        let (mut start, mut index_kept_a) = (None, true);
        for a in [1.30, 1.38, 1.52, 1.60] {
            let base = integrals(&ctx, &crossing_pair(a), "lda");
            let state = solve(&base, Method::Ks("lda"), [2, 2]);
            let found = excitations(&base, &state, &Request::new(6)).unwrap();
            // A's local excitation is the root polarized along z (B's is along x).
            let along_z = |j: usize| {
                let mu = found.transition_dipoles[j];
                mu[2].abs() > 10. * mu[0].abs().max(mu[1].abs()) && mu[2].abs() > 0.1
            };
            let a_root = (0..found.energies.len()).find(|&j| along_z(j)).unwrap();
            let start = *start.get_or_insert(a_root);
            let tracked = match &followed {
                None => excited_state(&base, &state, &singlet, start),
                Some(previous) => followed_excited_state(&base, &state, &singlet, previous),
            }
            .unwrap();
            let indexed = excited_state(&base, &state, &singlet, start).unwrap();
            index_kept_a &= along_z(indexed.root);
            println!(
                "a = {a:.2} Bohr: A's root {a_root} (omega {:.6} Ha), roots {:?}; overlap tracking picks {} (overlap {:?}), index tracking {} along z: {}",
                found.energies[a_root],
                &found.energies[..4],
                tracked.root,
                tracked.overlap,
                indexed.root,
                along_z(indexed.root)
            );
            assert_eq!(tracked.root, a_root, "a = {a}");
            assert!(
                tracked.overlap.is_none_or(|o| o > 0.9),
                "a = {a}: {:?}",
                tracked.overlap
            );
            followed = Some(tracked.followed);
        }
        // The crossing happened: A's index moved, and the fixed index lost A's state.
        assert_ne!(followed.map(|f| f.root), start);
        assert!(!index_kept_a);
    }
    assert_eq!(ctx.memory_used(), 0);
}

/// LiH on the shipped library (`$MANDACARU_PAW_PATH`; a missing library FAILS): lithium's
/// datasets carry a partial core (`nlcc` applied), so the kernel and its third-order change see
/// a real smooth core, which the synthetic hydrogen species has not. The analytic gradient of
/// `E_0 + omega` against reconverged energies on the same grid, LDA singlet TDA and PBE triplet
/// Casida, both atoms along the bond.
#[test]
#[ignore = "slow (24 s with 3 threads, timed alone 2026-11-15): run: cargo test --profile release-test -p mandacaru-core --test molecular_excited_forces -- --ignored the_excited_state_gradient_with_a_partial_core_follows_its_energy"]
fn the_excited_state_gradient_with_a_partial_core_follows_its_energy() {
    use paw_molecular_support::{analytic_shells, dataset, molecule};
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    {
        for (set, functional, spin, approximation) in [
            ("lda-sr", "lda", Spin::Singlet, Approximation::TammDancoff),
            ("pbe-sr", "pbe", Spin::Triplet, Approximation::Casida),
        ] {
            let lithium = dataset::read(&ctx, set, "Li");
            assert!(lithium.nlcc.applied, "{set}: lithium has no partial core");
            let mut arrays = mandacaru_core::paw::molecular::Arrays::new();
            let li = dataset::species(&ctx, &mut arrays, &lithium);
            let h = dataset::species(&ctx, &mut arrays, &dataset::read(&ctx, set, "H"));
            let base = molecule(
                &ctx,
                &[3, 1],
                &[[0.1, -0.05, -1.5], [-0.05, 0.08, 1.52]],
                12.,
                0.5,
                (vec![li, h], vec![0, 1], arrays),
                "raw",
                &analytic_shells(&[0, 1]),
            );
            check(
                &format!("LiH {set} {} {}", spin.name(), approximation.name()),
                &base,
                functional,
                spin,
                approximation,
                &[(0, 2), (1, 0), (1, 2)],
            );
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

/// `E_0 + omega_root` of independently converged unrestricted states at a displaced atom.
fn unrestricted_energy(
    base: &MolecularIntegrals,
    functional: &'static str,
    particles: [usize; 2],
    approximation: Approximation,
    root: usize,
    (atom, axis, delta): (usize, usize, f64),
) -> f64 {
    let moved = displaced(base, atom, axis, delta);
    let state = moved
        .kohn_sham(functional, particles, true, &options())
        .unwrap();
    assert!(state.converged);
    let mut open = request(Spin::Singlet, approximation);
    open.spin = None;
    let found = excitations(&moved, &state, &open).unwrap();
    state.free_energy() + moved.nuclear + found.energies[root]
}

#[test]
fn the_unrestricted_excited_state_gradient_follows_its_energy() {
    let ctx = Context::new(Some(2), 512 << 20).unwrap();
    {
        // A closed shell solved unrestricted: its lowest spin-conserving root is the Ms = 0
        // component of the lowest triplet, with the restricted triplet's energy and gradient.
        let base = integrals(&ctx, &H3, "lda");
        let closed = solve(&base, Method::Ks("lda"), [1, 1]);
        let restricted = excited_state(
            &base,
            &closed,
            &request(Spin::Triplet, Approximation::TammDancoff),
            0,
        )
        .unwrap();
        let open_state = base.kohn_sham("lda", [1, 1], true, &options()).unwrap();
        let mut open = request(Spin::Singlet, Approximation::TammDancoff);
        open.spin = None;
        let unrestricted = excited_state(&base, &open_state, &open, 0).unwrap();
        let worst = (0..9)
            .map(|k| {
                (restricted.gradient.total.as_slice()[k]
                    - unrestricted.gradient.total.as_slice()[k])
                    .abs()
            })
            .fold(0f64, f64::max);
        println!(
            "H3+ as UKS: omega {:.10} against the triplet's {:.10}; gradients differ by {worst:.2e} Ha/Bohr",
            unrestricted.excitation, restricted.excitation
        );
        assert!((unrestricted.excitation - restricted.excitation).abs() < 1e-8);
        assert!(worst < 1e-8, "{worst:e}");
        // A fully polarized reference (H2+, one alpha electron, no beta one): the beta channel
        // holds no transition; the gradient follows the energy all the same.
        let ion = integrals(&ctx, &H2, "lda");
        let polarized = ion.kohn_sham("lda", [1, 0], true, &options()).unwrap();
        let mut open = request(Spin::Singlet, Approximation::TammDancoff);
        open.spin = None;
        let excited = excited_state(&ion, &polarized, &open, 0).unwrap();
        let e = |d: f64| {
            unrestricted_energy(
                &ion,
                "lda",
                [1, 0],
                Approximation::TammDancoff,
                0,
                (1, 2, d),
            )
        };
        let (coarse, fine) = ((e(1e-3) - e(-1e-3)) / 2e-3, (e(5e-4) - e(-5e-4)) / 1e-3);
        let reference = (4. * fine - coarse) / 3.;
        let analytic = excited.gradient.total.as_slice()[5];
        println!(
            "H2+ UKS tda (1, 2): analytic {analytic:.10} FD {reference:.10} diff {:.2e}",
            analytic - reference
        );
        assert!((analytic - reference).abs() < 2e-7);
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
#[ignore = "slow (about 75 s with 3 threads): run: cargo test --profile release-test -p mandacaru-core --test molecular_excited_forces -- --ignored the_unrestricted_open_shell"]
fn the_unrestricted_open_shell_excited_state_gradient_follows_its_energy() {
    let ctx = Context::new(Some(2), 512 << 20).unwrap();
    {
        let base = integrals(&ctx, &H3, "lda");
        // An open shell: neutral bent H3 (two alpha, one beta), TDA and Casida, against
        // Richardson differences of reconverged energies.
        for approximation in [Approximation::TammDancoff, Approximation::Casida] {
            let state = base.kohn_sham("lda", [2, 1], true, &options()).unwrap();
            let mut open = request(Spin::Singlet, approximation);
            open.spin = None;
            let excited = excited_state(&base, &state, &open, 0).unwrap();
            // Tracking by overlap over both spin channels: at a moved geometry it keeps the root.
            let moved = displaced(&base, 2, 1, 0.02);
            let moved_state = moved.kohn_sham("lda", [2, 1], true, &options()).unwrap();
            let followed = pick(&moved, &moved_state, &open, 0, Some(&excited.followed)).unwrap();
            println!(
                "H3 UKS {}: followed root {} with overlap {:?} at the moved geometry",
                approximation.name(),
                followed.root,
                followed.overlap
            );
            assert_eq!(followed.root, 0);
            assert!(followed.overlap.is_some_and(|o| o > 0.99));
            let mut worst = 0f64;
            for (atom, axis) in [(1, 0), (2, 1), (2, 2)] {
                let e = |d: f64| {
                    unrestricted_energy(&base, "lda", [2, 1], approximation, 0, (atom, axis, d))
                };
                let h = 1e-3;
                let coarse = (e(h) - e(-h)) / (2. * h);
                let fine = (e(0.5 * h) - e(-0.5 * h)) / h;
                let reference = (4. * fine - coarse) / 3.;
                let analytic = excited.gradient.total.as_slice()[3 * atom + axis];
                println!(
                    "H3 UKS {} ({atom}, {axis}): analytic {analytic:.10} FD {reference:.10} diff {:.2e}",
                    approximation.name(),
                    analytic - reference
                );
                worst = worst.max((analytic - reference).abs());
            }
            println!(
                "H3 UKS {}: omega {:.8} Ha, separation {:.2e}, Z residual {:.1e} in {} iterations, max |dE| {worst:.3e} Ha/Bohr",
                approximation.name(),
                excited.excitation,
                excited.separation,
                excited.z_residual,
                excited.z_iterations
            );
            assert!(worst < 2e-7, "{worst:.3e} Ha/Bohr");
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

/// NH2 on the shipped library (`$MANDACARU_PAW_PATH`; a missing library FAILS): an open shell
/// (four alpha, three beta valence electrons) whose nitrogen carries a partial core, unrestricted
/// LDA, TDA and Casida, against reconverged energies on the same grid.
#[test]
#[ignore = "slow, recorded: NH2 on the library, TDA and Casida (about 4 minutes); run: cargo test --profile release-test -p mandacaru-core --test molecular_excited_forces -- --ignored the_unrestricted_excited_state_gradient_on_the_library"]
fn the_unrestricted_excited_state_gradient_on_the_library_follows_its_energy() {
    use paw_molecular_support::{analytic_shells, dataset, molecule};
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    {
        let nitrogen = dataset::read(&ctx, "lda-sr", "N");
        assert!(nitrogen.nlcc.applied, "nitrogen has no partial core");
        let mut arrays = mandacaru_core::paw::molecular::Arrays::new();
        let n = dataset::species(&ctx, &mut arrays, &nitrogen);
        let h = dataset::species(&ctx, &mut arrays, &dataset::read(&ctx, "lda-sr", "H"));
        let base = molecule(
            &ctx,
            &[7, 1, 1],
            &[[0.02, 0.05, 0.1], [1.55, 0.03, -1.0], [-1.45, -0.08, -1.1]],
            12.,
            0.5,
            (vec![n, h], vec![0, 1, 1], arrays),
            "raw",
            &analytic_shells(&[0, 1]),
        );
        for approximation in [Approximation::TammDancoff, Approximation::Casida] {
            let state = base.kohn_sham("lda", [4, 3], true, &options()).unwrap();
            assert!(state.converged);
            let mut open = request(Spin::Singlet, approximation);
            open.spin = None;
            let excited = excited_state(&base, &state, &open, 0).unwrap();
            let mut worst = 0f64;
            for (atom, axis) in [(0, 2), (1, 0), (2, 1)] {
                let e = |d: f64| {
                    unrestricted_energy(&base, "lda", [4, 3], approximation, 0, (atom, axis, d))
                };
                let h = 1e-3;
                let coarse = (e(h) - e(-h)) / (2. * h);
                let fine = (e(0.5 * h) - e(-0.5 * h)) / h;
                let reference = (4. * fine - coarse) / 3.;
                let analytic = excited.gradient.total.as_slice()[3 * atom + axis];
                println!(
                    "NH2 UKS {} ({atom}, {axis}): analytic {analytic:.10} FD {reference:.10} diff {:.2e}",
                    approximation.name(),
                    analytic - reference
                );
                worst = worst.max((analytic - reference).abs());
            }
            println!(
                "NH2 UKS {}: omega {:.8} Ha, separation {:.2e}, Z residual {:.1e} in {} iterations, max |dE| {worst:.3e} Ha/Bohr",
                approximation.name(),
                excited.excitation,
                excited.separation,
                excited.z_residual,
                excited.z_iterations
            );
            assert!(worst < 2e-7, "{worst:.3e} Ha/Bohr");
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

// Hartree-Fock (CIS and TDHF): exact exchange is the part of the kernel that sees the imaginary
// (antisymmetric) transition density, the `A - B` channel the semilocal kernels lack.

/// The canonical orbitals' `C^T F C` of the Hartree-Fock builder at `d + h C (v_s + i w_s) C^T`:
/// its central difference in `h`, real and imaginary parts (exact: the Fock matrix is linear).
fn hartree_fock_change(
    integrals: &MolecularIntegrals,
    state: &MeanField,
    frame: &DMatrix<f64>,
    v: [&DMatrix<f64>; 2],
    w: [&DMatrix<f64>; 2],
) -> ([DMatrix<f64>; 2], [DMatrix<f64>; 2]) {
    let d = &state.densities[0];
    let change = |s: usize, h: f64| {
        let (a, b) = (
            frame * v[s] * frame.transpose(),
            frame * w[s] * frame.transpose(),
        );
        DMatrix::from_fn(a.nrows(), a.ncols(), |p, q| {
            C64::new(h * a[(p, q)], h * b[(p, q)])
        })
    };
    let fock =
        |h: f64| reference_fock(integrals, state, &[d + change(0, h), d + change(1, h)]).unwrap();
    let (a, b) = (fock(1e-3), fock(-1e-3));
    let part = |re: bool| {
        [0, 1].map(|s| {
            let f = (&a[s] - &b[s]).map(|z| if re { z.re } else { z.im }) / 2e-3;
            frame.transpose() * f * frame
        })
    };
    (part(true), part(false))
}

#[test]
fn the_hartree_fock_first_order_fock_of_a_real_and_an_imaginary_change_is_the_builders() {
    let ctx = Context::new(Some(2), 512 << 20).unwrap();
    {
        let base = integrals(&ctx, &H4, "lda");
        let state = solve(&base, Method::Hf("rhf"), [2, 2]);
        let m = base.m();
        let symmetric =
            DMatrix::from_fn(m, m, |p, q| (((p + 1) * (q + 1)) as f64 * 0.31).sin() * 0.1);
        let symmetric = (&symmetric + symmetric.transpose()) * 0.5;
        let antisymmetric =
            DMatrix::from_fn(m, m, |p, q| ((p * 3 + q * 5) as f64 * 0.17).cos() * 0.1);
        let antisymmetric = (&antisymmetric - antisymmetric.transpose()) * 0.5;
        let zero = DMatrix::zeros(m, m);
        for (label, up, down) in [
            ("same", symmetric.clone(), symmetric.clone()),
            ("opposite", symmetric.clone(), -&symmetric),
        ] {
            let (analytic, second, frame) =
                seams::first_and_second(&base, &state, [&up, &down]).unwrap();
            assert!(second.is_empty());
            let (numeric, _) =
                hartree_fock_change(&base, &state, &frame, [&up, &down], [&zero, &zero]);
            let worst = (0..2)
                .map(|s| (&analytic[s] - &numeric[s]).amax())
                .fold(0f64, f64::max);
            println!(
                "HF real ({label}): |F1 - dF| = {worst:.3e} of {:.3e}",
                analytic[0].amax()
            );
            assert!(
                worst < 1e-10 * analytic[0].amax().max(1.),
                "{label}: {worst:.3e}"
            );
        }
        for (label, up, down) in [
            ("same", antisymmetric.clone(), antisymmetric.clone()),
            ("opposite", antisymmetric.clone(), -&antisymmetric),
        ] {
            let analytic = seams::imaginary_fock(&base, &state, [&up, &down])
                .unwrap()
                .unwrap();
            let frame = seams::first_and_second(&base, &state, [&zero, &zero])
                .unwrap()
                .2;
            let (real, numeric) =
                hartree_fock_change(&base, &state, &frame, [&zero, &zero], [&up, &down]);
            let worst = (0..2)
                .map(|s| (&analytic[s] - &numeric[s]).amax())
                .fold(0f64, f64::max);
            println!(
                "HF imaginary ({label}): |Im F1 - Im dF| = {worst:.3e} of {:.3e}, real part {:.1e}",
                analytic[0].amax(),
                real[0].amax()
            );
            assert!(
                worst < 1e-10 * analytic[0].amax().max(1.),
                "{label}: {worst:.3e}"
            );
            assert!(real[0].amax() < 1e-10);
        }
        // A density-only kernel has no imaginary channel.
        let ks = solve(&base, Method::Ks("lda"), [2, 2]);
        assert!(
            seams::imaginary_fock(&base, &ks, [&antisymmetric, &antisymmetric])
                .unwrap()
                .is_none()
        );
    }
    assert_eq!(ctx.memory_used(), 0);
}

/// `E_0 + omega_root` of an independently converged Hartree-Fock state at a displaced atom.
fn hartree_fock_energy(
    base: &MolecularIntegrals,
    kind: &'static str,
    particles: [usize; 2],
    request: &Request,
    root: usize,
    (atom, axis, delta): (usize, usize, f64),
) -> f64 {
    let moved = displaced(base, atom, axis, delta);
    let state = solve(&moved, Method::Hf(kind), particles);
    let found = excitations(&moved, &state, request).unwrap();
    state.free_energy() + moved.nuclear + found.energies[root]
}

/// The analytic gradient of a Hartree-Fock excited state against Richardson differences of
/// reconverged energies (2e-7 Ha/Bohr).
fn check_hartree_fock(
    label: &str,
    base: &MolecularIntegrals,
    kind: &'static str,
    particles: [usize; 2],
    request: &Request,
    components: &[(usize, usize)],
) {
    let state = solve(base, Method::Hf(kind), particles);
    let excited = excited_state(base, &state, request, 0).unwrap();
    let mut worst = 0f64;
    for &(atom, axis) in components {
        let e = |d: f64| hartree_fock_energy(base, kind, particles, request, 0, (atom, axis, d));
        let h = 1e-3;
        let coarse = (e(h) - e(-h)) / (2. * h);
        let fine = (e(0.5 * h) - e(-0.5 * h)) / h;
        let reference = (4. * fine - coarse) / 3.;
        let analytic = excited.gradient.total.as_slice()[3 * atom + axis];
        println!(
            "{label} ({atom}, {axis}): analytic {analytic:.10} FD {reference:.10} diff {:.2e}",
            analytic - reference
        );
        worst = worst.max((analytic - reference).abs());
    }
    println!(
        "{label}: omega {:.8} Ha, separation {:.2e}, Z residual {:.1e} in {} iterations, max |dE| {worst:.3e} Ha/Bohr",
        excited.excitation, excited.separation, excited.z_residual, excited.z_iterations
    );
    assert!(worst < 2e-7, "{label}: {worst:.3e} Ha/Bohr");
}

#[test]
#[ignore = "slow (15 s with 3 threads, timed alone 2026-11-15): run: cargo test --profile release-test -p mandacaru-core --test molecular_excited_forces -- --ignored the_hartree_fock_relaxed_density_is_the_field_derivative_of_the_excitation"]
fn the_hartree_fock_relaxed_density_is_the_field_derivative_of_the_excitation() {
    let ctx = Context::new(Some(2), 512 << 20).unwrap();
    {
        for (spin, approximation, geometry, particles) in [
            (Spin::Singlet, Approximation::Casida, &H4[..], [2, 2]),
            (Spin::Triplet, Approximation::TammDancoff, &H4[..], [2, 2]),
            (Spin::Triplet, Approximation::Casida, &H3[..], [1, 1]),
        ] {
            let mut base = integrals(&ctx, geometry, "lda");
            let state = solve(&base, Method::Hf("rhf"), particles);
            let excited = excited_state(&base, &state, &request(spin, approximation), 0).unwrap();
            let m = base.m();
            let z = position_integrals(&base, [0.; 3]).unwrap()[2].clone();
            let relaxed = excited.relaxed.dot(&z);
            let original: Vec<f64> = base.one.as_slice().to_vec();
            let mut omega = |field: f64| {
                for a in 0..m * m {
                    base.one.as_mut_slice()[a] = original[a] + field * z.as_slice()[a];
                }
                let state = solve(&base, Method::Hf("rhf"), particles);
                excitations(&base, &state, &request(spin, approximation))
                    .unwrap()
                    .energies[0]
            };
            let slope =
                |f: f64, omega: &mut dyn FnMut(f64) -> f64| (omega(f) - omega(-f)) / (2. * f);
            let (coarse, fine) = (slope(2e-3, &mut omega), slope(1e-3, &mut omega));
            let numeric = (4. * fine - coarse) / 3.;
            base.one.as_mut_slice().copy_from_slice(&original);
            println!(
                "{} atoms HF {} {}: tr[P z] {relaxed:.10} d omega/dF {numeric:.10} diff {:.2e}",
                geometry.len(),
                spin.name(),
                approximation.name(),
                relaxed - numeric
            );
            assert!(
                (relaxed - numeric).abs() < 1e-7,
                "{relaxed} against {numeric}"
            );
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
#[ignore = "slow (32 s with 3 threads, timed alone 2026-11-15): run: cargo test --profile release-test -p mandacaru-core --test molecular_excited_forces -- --ignored the_hartree_fock_excited_state_gradient_follows_its_energy"]
fn the_hartree_fock_excited_state_gradient_follows_its_energy() {
    let ctx = Context::new(Some(2), 512 << 20).unwrap();
    {
        let base = integrals(&ctx, &H4, "lda");
        check_hartree_fock(
            "H4 TDHF singlet",
            &base,
            "rhf",
            [2, 2],
            &request(Spin::Singlet, Approximation::Casida),
            &[(1, 0), (2, 2)],
        );
        check_hartree_fock(
            "H4 CIS triplet",
            &base,
            "rhf",
            [2, 2],
            &request(Spin::Triplet, Approximation::TammDancoff),
            &[(3, 1)],
        );
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
#[ignore = "slow: run: cargo test --profile release-test -p mandacaru-core --test molecular_excited_forces -- --ignored the_hartree_fock_excited_state_gradients_of_h3"]
fn the_hartree_fock_excited_state_gradients_of_h3_follow_their_energies() {
    let ctx = Context::new(Some(2), 512 << 20).unwrap();
    {
        let base = integrals(&ctx, &H3, "lda");
        let components = [(0, 0), (1, 2), (2, 1)];
        for (spin, approximation) in [
            (Spin::Singlet, Approximation::TammDancoff),
            (Spin::Triplet, Approximation::TammDancoff),
            (Spin::Singlet, Approximation::Casida),
            (Spin::Triplet, Approximation::Casida),
        ] {
            check_hartree_fock(
                &format!("H3+ HF {} {}", spin.name(), approximation.name()),
                &base,
                "rhf",
                [1, 1],
                &request(spin, approximation),
                &components,
            );
        }
        // An open shell: neutral bent H3 (two alpha, one beta) on an unrestricted reference.
        for approximation in [Approximation::TammDancoff, Approximation::Casida] {
            let mut open = request(Spin::Singlet, approximation);
            open.spin = None;
            check_hartree_fock(
                &format!("H3 UHF {}", approximation.name()),
                &base,
                "uhf",
                [2, 1],
                &open,
                &components,
            );
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

/// The two +U forms on copper's 3d shell (eV).
const DUDAREV: (&str, f64, f64) = ("dudarev", 20., 0.);
const LIECHTENSTEIN: (&str, f64, f64) = ("liechtenstein", 20., 3.);

/// A bent, unequal H-Cu-H anion from the `lda-sr` library (`$MANDACARU_PAW_PATH`; a missing
/// library FAILS) with a +U shell on copper's 3d (a real occupation near 10), and its occupied
/// count per spin. The synthetic hydrogen `d` shell holds about 1e-6 electrons: there the +U
/// part of an excited-state force is 1e-13 Ha/Bohr (measured 2026-11-15), unresolvable. A
/// diatomic's nondegenerate roots carry no antisymmetric shell occupation (sigma to sigma: one
/// `m`), so the molecule is bent (Cs: three `d` components in one irrep).
fn copper_dihydride(
    ctx: &Context,
    (form, u, j): (&str, f64, f64),
) -> (MolecularIntegrals, [usize; 2]) {
    use paw_molecular_support::{analytic_shells, dataset, molecule_with};
    let mut arrays = mandacaru_core::paw::molecular::Arrays::new();
    let copper = dataset::read(ctx, "lda-sr", "Cu");
    let electrons = copper.valence_charge.round() as usize + 3;
    let copper = dataset::species(ctx, &mut arrays, &copper);
    let hydrogen = dataset::species(ctx, &mut arrays, &dataset::read(ctx, "lda-sr", "H"));
    let request = format!(r#"{{"Cu": {{"l": "d", "U": {u}, "J": {j}, "form": "{form}"}}}}"#);
    let integrals = molecule_with(
        ctx,
        &[29, 1, 1],
        &[[8.5, 8.5, 7.6], [8.5, 8.5, 10.4], [10.9, 8.6, 6.3]],
        17.,
        0.5,
        (vec![copper, hydrogen], vec![0, 1, 1], arrays),
        "raw",
        &analytic_shells(&[0, 1, 2]),
        &[("hubbard", serde_json::Value::String(request))],
    );
    (integrals, [electrons / 2, electrons / 2])
}

fn hubbard_of(integrals: &MolecularIntegrals) -> &MolecularHubbard {
    integrals
        .corrections
        .as_ref()
        .and_then(|c| c.hubbard.as_ref())
        .expect("a +U molecule")
}

/// The builder's +U Fock additions (`MolecularHubbard::evaluate_complex`) at the frame spin
/// densities `d + C x_s C^T`, in the canonical orbitals `C` (`frame`).
fn hubbard_builder(
    integrals: &MolecularIntegrals,
    state: &MeanField,
    frame: &DMatrix<f64>,
    x: [&DMatrix<C64>; 2],
) -> [DMatrix<C64>; 2] {
    let c = frame.map(|v| C64::new(v, 0.));
    let densities: Vec<DMatrix<C64>> = (0..2)
        .map(|s| &state.densities[0] + &c * x[s] * c.transpose())
        .collect();
    let (_, focks) = hubbard_of(integrals)
        .evaluate_complex(&integrals.context, &densities)
        .unwrap();
    [0, 1].map(|s| c.transpose() * &focks[s] * &c)
}

#[test]
#[ignore = "slow (22 s with 3 threads, timed alone 2026-11-15): run: cargo test --profile release-test -p mandacaru-core --test molecular_excited_forces -- --ignored the_hubbard_first_order_and_imaginary_focks_are_the_builders"]
fn the_hubbard_first_order_and_imaginary_focks_are_the_builders() {
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    {
        for request in [DUDAREV, LIECHTENSTEIN] {
            let form = request.0;
            let (base, particles) = copper_dihydride(&ctx, request);
            let state = solve(&base, Method::Ks("lda"), particles);
            // Both spin channels pushed (the frozen flag) reproduce the restricted ground force.
            let fixed = seams::ground_fixed(&base, &state).unwrap();
            let force = base.gradient(&state).unwrap();
            let ground = fixed
                .iter()
                .zip(force.total.as_slice())
                .map(|(a, b)| (a - b).abs())
                .fold(0f64, f64::max);
            let scale = force
                .total
                .as_slice()
                .iter()
                .fold(1f64, |t, v| t.max(v.abs()));
            println!("+U: |G_fixed - G_ground| = {ground:.3e} of {scale:.2} Ha/Bohr");
            assert!(ground < 1e-12 * scale, "{ground:.3e}");
            let m = base.m();
            let symmetric =
                DMatrix::from_fn(m, m, |p, q| (((p + 1) * (q + 1)) as f64 * 0.31).sin() * 0.1);
            let symmetric = (&symmetric + symmetric.transpose()) * 0.5;
            let antisymmetric =
                DMatrix::from_fn(m, m, |p, q| ((p * 3 + q * 5) as f64 * 0.17).cos() * 0.1);
            let antisymmetric = (&antisymmetric - antisymmetric.transpose()) * 0.5;
            let lift = |x: &DMatrix<f64>, imaginary: bool| {
                x.map(|v| {
                    if imaginary {
                        C64::new(0., v)
                    } else {
                        C64::new(v, 0.)
                    }
                })
            };
            for imaginary in [false, true] {
                let x = if imaginary {
                    &antisymmetric
                } else {
                    &symmetric
                };
                for (label, up, down) in
                    [("same", x.clone(), x.clone()), ("opposite", x.clone(), -x)]
                {
                    let (analytic, frame) =
                        seams::hubbard_fock(&base, &state, [&up, &down], imaginary).unwrap();
                    let analytic = analytic.expect("a +U molecule has a +U Fock");
                    let (plus, minus) = (
                        hubbard_builder(
                            &base,
                            &state,
                            &frame,
                            [&lift(&up, imaginary), &lift(&down, imaginary)],
                        ),
                        hubbard_builder(
                            &base,
                            &state,
                            &frame,
                            [&lift(&-&up, imaginary), &lift(&-&down, imaginary)],
                        ),
                    );
                    let worst = (0..2)
                        .map(|s| {
                            let change = (&plus[s] - &minus[s]) * C64::new(0.5, 0.);
                            let part = change.map(|z| if imaginary { z.im } else { z.re });
                            (&analytic[s] - part).amax()
                        })
                        .fold(0f64, f64::max);
                    let scale = analytic[0].amax();
                    println!(
                        "+U {} {} ({label}): |F1 - dF| = {worst:.3e} of {scale:.3e}",
                        form,
                        if imaginary { "imaginary" } else { "real" }
                    );
                    assert!(scale > 1e-6, "{label}: the +U Fock change is {scale:.1e}");
                    assert!(worst < 1e-10 * scale.max(1.), "{label}: {worst:.3e}");
                }
            }
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

/// `d^2 E_U / dt^2` along `D + t i A` of the raw-AO spin matrices `a`: the shells evaluated on the
/// raw projections `q_ao` (exact by three points: the energy is quadratic).
fn hubbard_second_change(integrals: &MolecularIntegrals, a: [&DMatrix<f64>; 2]) -> f64 {
    let hubbard = hubbard_of(integrals);
    let raw = MolecularHubbard {
        form: hubbard.form,
        shells: hubbard
            .shells
            .iter()
            .map(|s| MolecularHubbardShell {
                q: s.q_ao.clone(),
                ..s.clone()
            })
            .collect(),
    };
    let energy = |t: f64| {
        let d: Vec<DMatrix<C64>> = a.iter().map(|a| a.map(|v| C64::new(0., t * v))).collect();
        raw.evaluate_complex(&integrals.context, &d).unwrap().0
    };
    energy(1.) + energy(-1.) - 2. * energy(0.)
}

#[test]
#[ignore = "slow (50 s with 3 threads, timed alone 2026-11-15): run: cargo test --profile release-test -p mandacaru-core --test molecular_excited_forces -- --ignored the_hubbard_imaginary_force_is_the_derivative_of_the_second_change"]
fn the_hubbard_imaginary_force_is_the_derivative_of_the_second_change() {
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    {
        for request in [DUDAREV, LIECHTENSTEIN] {
            let form = request.0;
            let (base, _) = copper_dihydride(&ctx, request);
            let m = base.m();
            let a = |phase: f64| {
                let x = DMatrix::from_fn(m, m, |p, q| {
                    ((p * 3 + q * 5) as f64 * 0.17 + phase).cos() * 0.1
                });
                (&x - x.transpose()) * 0.5
            };
            let (up, down) = (a(0.), a(0.4));
            let setup = base.paw_setup.as_ref().unwrap();
            let analytic = hubbard_imaginary_force(setup, &base, [&up, &down])
                .unwrap()
                .expect("a +U molecule has the term");
            let mut worst = 0f64;
            let scale = analytic.iter().fold(0f64, |t, v| t.max(v.abs()));
            for (atom, axis) in [(0, 0), (1, 2), (2, 1), (2, 2)] {
                let f =
                    |d: f64| hubbard_second_change(&displaced(&base, atom, axis, d), [&up, &down]);
                let h = 1e-3;
                let coarse = (f(h) - f(-h)) / (2. * h);
                let fine = (f(0.5 * h) - f(-0.5 * h)) / h;
                let reference = (4. * fine - coarse) / 3.;
                let value = analytic[3 * atom + axis];
                println!(
                    "+U {} ({atom}, {axis}): analytic {value:.10e} FD {reference:.10e}",
                    form
                );
                worst = worst.max((value - reference).abs());
            }
            // The translation sum vanishes.
            let net = (0..3)
                .map(|x| (0..3).map(|a| analytic[3 * a + x]).sum::<f64>().abs())
                .fold(0f64, f64::max);
            println!(
                "+U {}: worst {worst:.3e} of {scale:.3e}, net {net:.1e}",
                form
            );
            assert!(scale > 1e-6, "the term is {scale:.1e}");
            assert!(worst < 1e-8 * scale, "{worst:.3e}");
            assert!(net < 1e-10 * scale, "{net:.3e}");
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

/// `E_0 + omega_root` of independently converged +U states at a displaced atom.
fn hubbard_check(
    label: &str,
    functional: &'static str,
    (base, particles): (&MolecularIntegrals, [usize; 2]),
    spin: Option<Spin>,
    approximation: Approximation,
    components: &[(usize, usize)],
) {
    // Unset spin: an unrestricted reference's spin-conserving roots.
    let request = || {
        let mut r = request(spin.unwrap_or(Spin::Singlet), approximation);
        r.spin = spin;
        r
    };
    let state = solve(base, Method::Ks(functional), particles);
    let excited = excited_state(base, &state, &request(), 0).unwrap();
    let imaginary = &excited
        .parts
        .iter()
        .find(|(name, _)| *name == "hubbard_imaginary")
        .expect("a +U excited state has its imaginary-direction term")
        .1;
    let (mut worst, mut without, mut size) = (0f64, 0f64, 0f64);
    for &(atom, axis) in components {
        let e = |d: f64| {
            let moved = displaced(base, atom, axis, d);
            let state = solve(&moved, Method::Ks(functional), particles);
            let found = excitations(&moved, &state, &request()).unwrap();
            state.free_energy() + moved.nuclear + found.energies[0]
        };
        let h = 1e-3;
        let coarse = (e(h) - e(-h)) / (2. * h);
        let fine = (e(0.5 * h) - e(-0.5 * h)) / h;
        let reference = (4. * fine - coarse) / 3.;
        let k = 3 * atom + axis;
        let analytic = excited.gradient.total.as_slice()[k];
        println!(
            "{label} ({atom}, {axis}): analytic {analytic:.10} FD {reference:.10} diff {:.2e}, i T' term {:.2e}",
            analytic - reference,
            imaginary[k]
        );
        worst = worst.max((analytic - reference).abs());
        without = without.max((analytic - imaginary[k] - reference).abs());
        size = size.max(imaginary[k].abs());
    }
    println!(
        "{label}: omega {:.8} Ha, separation {:.2e}, max |dE| {worst:.3e} Ha/Bohr, without the i T' term {without:.3e}",
        excited.excitation, excited.separation
    );
    assert!(worst < 2e-7, "{label}: {worst:.3e} Ha/Bohr");
    // The term is measured, not passed by being zero: dropping it misses by far more than the gate.
    assert!(
        without > 10. * worst.max(1e-9),
        "{label}: the i T' term {size:.2e} is not resolved"
    );
}

#[test]
#[ignore = "slow (27.0 s alone, 57.1 s user, at load 6 to 14, 2026-11-16: five CuH2- +U SCFs in a 17 Bohr box), moved out of the fast tier so that tier stays under 120 s on a loaded machine; its case is the first of the_hubbard_excited_state_gradients_of_both_forms_follow_their_energies; run: cargo test --profile release-test -p mandacaru-core --test molecular_excited_forces -- --ignored the_hubbard_excited_state_gradient_follows_its_energy"]
fn the_hubbard_excited_state_gradient_follows_its_energy() {
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    {
        let (base, particles) = copper_dihydride(&ctx, DUDAREV);
        hubbard_check(
            "CuH2- +U Dudarev singlet TDA",
            "lda",
            (&base, particles),
            Some(Spin::Singlet),
            Approximation::TammDancoff,
            &[(0, 0)],
        );
    }
    assert_eq!(ctx.memory_used(), 0);
}

/// r2SCAN with `hubbard_u` (Dudarev on the library's copper 3d, CuH2-, singlet TDA): the meta-GGA's
/// kernel, current and +U shells together against reconverged energies.
#[test]
#[ignore = "slow: 210 s, 442 MB (measured 2026-11-16); run: cargo test --profile release-test -p mandacaru-core --test molecular_excited_forces -- --ignored the_r2scan_hubbard_excited_state_gradient --nocapture"]
fn the_r2scan_hubbard_excited_state_gradient_follows_its_energy() {
    let ctx = Context::new(Some(3), 1 << 30).unwrap();
    {
        let (base, particles) = copper_dihydride(&ctx, DUDAREV);
        hubbard_check(
            "CuH2- r2SCAN +U Dudarev singlet TDA",
            "r2scan",
            (&base, particles),
            Some(Spin::Singlet),
            Approximation::TammDancoff,
            &[(0, 0), (1, 2)],
        );
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
#[ignore = "slow (about 290 s with 3 threads: the four +U tests ran 387 s together): run: cargo test --profile release-test -p mandacaru-core --test molecular_excited_forces -- --ignored the_hubbard_excited_state_gradients_of_both_forms"]
/// Both forms, a closed shell (CuH2-) and an unrestricted reference (neutral CuH2, seven alpha
/// and six beta electrons, spin-conserving roots).
fn the_hubbard_excited_state_gradients_of_both_forms_follow_their_energies() {
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    {
        for (label, request, spin, approximation, neutral) in [
            (
                "CuH2- +U Dudarev singlet TDA",
                DUDAREV,
                Some(Spin::Singlet),
                Approximation::TammDancoff,
                false,
            ),
            (
                "CuH2- +U Liechtenstein triplet Casida",
                LIECHTENSTEIN,
                Some(Spin::Triplet),
                Approximation::Casida,
                false,
            ),
            (
                "CuH2 +U Dudarev unrestricted TDA",
                DUDAREV,
                None,
                Approximation::TammDancoff,
                true,
            ),
            (
                "CuH2 +U Liechtenstein unrestricted Casida",
                LIECHTENSTEIN,
                None,
                Approximation::Casida,
                true,
            ),
        ] {
            let (base, [alpha, beta]) = copper_dihydride(&ctx, request);
            let particles = if neutral {
                [alpha, beta - 1]
            } else {
                [alpha, beta]
            };
            hubbard_check(
                label,
                "lda",
                (&base, particles),
                spin,
                approximation,
                &[(0, 0), (2, 2)],
            );
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

/// LiH on the shipped library (`$MANDACARU_PAW_PATH`; a missing library FAILS): lithium's
/// partial core (`nlcc` applied) and the frozen-core exchange of its core states with the valence
/// enter the Hartree-Fock reference and its one-center terms, which the synthetic hydrogen has
/// not. CIS singlet and TDHF triplet against reconverged energies on the same grid.
#[test]
fn the_hartree_fock_excited_state_gradient_on_the_library_follows_its_energy() {
    use paw_molecular_support::{analytic_shells, dataset, molecule};
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    {
        let lithium = dataset::read(&ctx, "lda-sr", "Li");
        assert!(lithium.nlcc.applied, "lithium has no partial core");
        let core_exchange =
            mandacaru_core::paw::molecular::hartree_fock_onsite_channels(&ctx, &lithium)
                .unwrap()
                .iter()
                .fold(0f64, |t, c| t.max(c.core_exchange.amax()));
        println!("LiH: largest core exchange entry {core_exchange:.3e} Ha");
        assert!(
            core_exchange > 1e-3,
            "lithium's core exchange is {core_exchange:.1e}"
        );
        let mut arrays = mandacaru_core::paw::molecular::Arrays::new();
        let li = dataset::species(&ctx, &mut arrays, &lithium);
        let h = dataset::species(&ctx, &mut arrays, &dataset::read(&ctx, "lda-sr", "H"));
        let base = molecule(
            &ctx,
            &[3, 1],
            &[[0.1, -0.05, -1.5], [-0.05, 0.08, 1.52]],
            12.,
            0.5,
            (vec![li, h], vec![0, 1], arrays),
            "raw",
            &analytic_shells(&[0, 1]),
        );
        for (label, spin, approximation) in [
            ("LiH CIS singlet", Spin::Singlet, Approximation::TammDancoff),
            ("LiH TDHF triplet", Spin::Triplet, Approximation::Casida),
        ] {
            check_hartree_fock(
                label,
                &base,
                "rhf",
                [1, 1],
                &request(spin, approximation),
                &[(0, 2), (1, 0), (1, 2)],
            );
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

/// The coupled screened exchange `-(a/2) sum c_kl (rho_kl | rho_lk)` of fixed complex orbitals
/// (raw atomic basis) and signed couplings, against a Richardson central difference of its energy
/// from the screened pair factors at displaced geometries: the carrier of HSE06's exchange along
/// the relaxed, transition and imaginary directions (`gradient::seams::coupled_exchange`).
#[test]
#[ignore = "slow (53 s with 3 threads: a screened pair factorization per displaced geometry): run: cargo test --profile release-test -p mandacaru-core --test molecular_excited_forces -- --ignored the_coupled_screened_exchange"]
fn the_coupled_screened_exchange_gradient_is_its_derivative_at_fixed_orbitals() {
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    {
        let base = integrals(&ctx, &H3, "pbe");
        let m = base.m();
        let orbitals: Vec<Vec<C64>> = (0..4)
            .map(|k| {
                (0..m)
                    .map(|a| {
                        let x = (a * 5 + k * 3) as f64;
                        C64::new(
                            (0.37 * x).sin(),
                            if k % 2 == 0 {
                                0.2 * (0.71 * x).cos()
                            } else {
                                0.
                            },
                        )
                    })
                    .collect()
            })
            .collect();
        // Signed couplings, a cross block and zeros (pairs that are not solved).
        let couplings = DMatrix::from_row_slice(
            4,
            4,
            &[
                0.8, -0.3, 0., 0.5, -0.3, 0., 0.7, 0., 0., 0.7, -1.1, 0.2, 0.5, 0., 0.2, 0.,
            ],
        );
        let (_, analytic) = seams::coupled_exchange(&base, "hse06", &orbitals, &couplings).unwrap();
        let mut worst = 0f64;
        let mut scale = 0f64;
        for (atom, axis) in [(0, 0), (1, 2), (2, 1)] {
            let e = |d: f64| {
                seams::coupled_exchange(
                    &displaced(&base, atom, axis, d),
                    "hse06",
                    &orbitals,
                    &couplings,
                )
                .unwrap()
                .0
            };
            let h = 1e-3;
            let coarse = (e(h) - e(-h)) / (2. * h);
            let fine = (e(0.5 * h) - e(-0.5 * h)) / h;
            let reference = (4. * fine - coarse) / 3.;
            let value = analytic[3 * atom + axis];
            println!("coupled exchange ({atom}, {axis}): analytic {value:.10} FD {reference:.10} diff {:.2e}", value - reference);
            worst = worst.max((value - reference).abs());
            scale = scale.max(value.abs());
        }
        println!("coupled exchange: max |dE| {worst:.3e} of max |g| {scale:.3e} Ha/Bohr");
        assert!(
            scale > 1e-4,
            "the coupled exchange gradient vanishes: {scale:.2e}"
        );
        assert!(worst < 1e-8, "{worst:.3e} Ha/Bohr");
    }
    assert_eq!(ctx.memory_used(), 0);
}

/// LiH on the shipped `pbe-sr` library (`$MANDACARU_PAW_PATH`; a missing library FAILS): the
/// screened hybrid's compensation charges, one-center tensor and spheres of a real dataset
/// (lithium's single closed 1s core, the hybrid core rule: one valence electron per atom).
fn hse06_lithium_hydride(ctx: &Context) -> MolecularIntegrals {
    use paw_molecular_support::{analytic_shells, dataset, molecule};
    let mut arrays = mandacaru_core::paw::molecular::Arrays::new();
    let li = dataset::species(ctx, &mut arrays, &dataset::read(ctx, "pbe-sr", "Li"));
    let h = dataset::species(ctx, &mut arrays, &dataset::read(ctx, "pbe-sr", "H"));
    molecule(
        ctx,
        &[3, 1],
        &[[0.04, -0.06, -0.08], [0.09, 0.03, 3.07]],
        12.,
        0.5,
        (vec![li, h], vec![0, 1], arrays),
        "raw",
        &analytic_shells(&[0, 1]),
    )
}

/// A converged HSE06 state at 1e-11 in at most `iterations`: the nested steps of 2e-5 need
/// energies below 1e-11 Ha, and a displaced LiH2- did not converge in 400 iterations at 1e-12
/// (measured 2026-11-15). Neutral LiH (one valence electron on lithium) converges in 9 to 11
/// iterations at 1e-10 to 1e-12, all 24 displaced runs of the gate (gap 0.268 Ha; remeasured
/// 2026-11-15 on the current tree: an earlier report of no convergence came from a stale build).
fn hybrid_solve(
    integrals: &MolecularIntegrals,
    particles: [usize; 2],
    iterations: usize,
) -> MeanField {
    let options = mandacaru_core::molecular::ScfOptions {
        tolerance: 1e-11,
        max_iterations: iterations,
        ..Default::default()
    };
    let state = integrals
        .kohn_sham("hse06", particles, particles[0] != particles[1], &options)
        .unwrap();
    assert!(state.converged, "the HSE06 SCF did not converge");
    state
}

/// The HSE06 excited-state gradient against central differences of reconverged excited energies.
/// The hole model's bend at `t = 1` (TDDFT_PLAN Q17) kinks the kernel, so `omega` has a slope
/// jump wherever a grid point crosses it: the analytic gradient is the slope between crossings,
/// and a difference whose interval holds a crossing is off by the jump (central differences of
/// LiH2- at 5e-4, 1e-3 and 2e-3 scatter by 8e-6 Ha/Bohr; one at 2e-5 missed by 9e-4). The
/// reference is the central difference at the first of the steps `h, h/2, h/4` that agrees
/// with the next one to 1e-7 Ha/Bohr (no crossing in either interval); the gate stays 2e-7.
/// The hybrid's own parts are printed and must be resolved.
fn hybrid_check(
    label: &str,
    (base, particles, iterations): (&MolecularIntegrals, [usize; 2], usize),
    spin: Option<Spin>,
    approximation: Approximation,
    components: &[(usize, usize)],
    h: f64,
) {
    let request = || {
        let mut r = request(spin.unwrap_or(Spin::Singlet), approximation);
        r.spin = spin;
        r
    };
    let state = hybrid_solve(base, particles, iterations);
    let excited = excited_state(base, &state, &request(), 0).unwrap();
    let part = |name: &str| {
        excited
            .parts
            .iter()
            .find(|(n, _)| *n == name)
            .unwrap_or_else(|| panic!("an HSE06 excited state has its {name} part"))
            .1
            .clone()
    };
    let (exchange, spheres) = (part("hybrid_exchange"), part("hybrid_spheres"));
    let (mut worst, mut without) = (0f64, 0f64);
    for &(atom, axis) in components {
        let e = |d: f64| {
            let moved = displaced(base, atom, axis, d);
            let state = hybrid_solve(&moved, particles, iterations);
            let found = excitations(&moved, &state, &request()).unwrap();
            state.free_energy() + moved.nuclear + found.energies[0]
        };
        let central: Vec<f64> = [h, 0.5 * h, 0.25 * h]
            .iter()
            .map(|&step| (e(step) - e(-step)) / (2. * step))
            .collect();
        println!("{label} ({atom}, {axis}): central differences {central:?}");
        let reference = (0..2)
            .find(|&j| (central[j] - central[j + 1]).abs() < 1e-7)
            .map(|j| central[j])
            .unwrap_or_else(|| panic!("{label}: every step interval holds a kink: {central:?}"));
        let k = 3 * atom + axis;
        let analytic = excited.gradient.total.as_slice()[k];
        println!(
            "{label} ({atom}, {axis}): analytic {analytic:.10} FD {reference:.10} diff {:.2e}, exchange {:.2e}, spheres {:.2e}",
            analytic - reference,
            exchange[k],
            spheres[k]
        );
        worst = worst.max((analytic - reference).abs());
        without = without.max((analytic - exchange[k] - reference).abs());
    }
    println!(
        "{label}: omega {:.8} Ha, separation {:.2e}, Z residual {:.1e}, max |dE| {worst:.3e} Ha/Bohr, without the exchange part {without:.3e}",
        excited.excitation, excited.separation, excited.z_residual
    );
    assert!(worst < 2e-7, "{label}: {worst:.3e} Ha/Bohr");
    assert!(
        without > 10. * worst.max(1e-9),
        "{label}: the exchange part is not resolved"
    );
}

#[test]
#[ignore = "slow (755 s with 3 threads), the HSE06 gate on the library: run: cargo test --profile release-test -p mandacaru-core --test molecular_excited_forces -- --ignored the_hse06_excited_state_gradients_of_lih_follow_their_energies"]
fn the_hse06_excited_state_gradients_of_lih_follow_their_energies() {
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    {
        let base = hse06_lithium_hydride(&ctx);
        for (label, particles, spin, approximation, components) in [
            // Along the bond: a displaced LiH2- across it (hydrogen along x) did not converge in
            // 400 SCF iterations at 1e-11 (measured 2026-11-15).
            (
                "LiH2- HSE06 singlet TDA",
                [2, 2],
                Some(Spin::Singlet),
                Approximation::TammDancoff,
                &[(0, 2), (1, 2)][..],
            ),
            // LiH2- is triplet-unstable (the triplet TDA root is -4.3e-2 Ha): the spin-resolved
            // pieces (the spheres' polarization and its changes) are gated on the doublet.
            (
                "LiH- HSE06 unrestricted Casida",
                [2, 1],
                None,
                Approximation::Casida,
                &[(1, 2)][..],
            ),
        ] {
            // Steps of 2e-5: the hole model's bend at t = 1 kinks the kernel, so `omega` has a
            // derivative jump wherever a grid point crosses it (TDDFT_PLAN Q17): its central
            // differences at 5e-4, 1e-3 and 2e-3 scatter by 8e-6 Ha/Bohr (measured on LiH2-).
            hybrid_check(
                label,
                (&base, particles, 400),
                spin,
                approximation,
                components,
                2e-5,
            );
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

/// The closed-shell HSE06 gate on neutral LiH, which converges in 9 to 11 SCF iterations: capped
/// at 60, so a run that stops converging fails in minutes.
#[test]
#[ignore = "slow, the neutral HSE06 gate on the library: run: cargo test --profile release-test -p mandacaru-core --test molecular_excited_forces -- --ignored the_hse06_excited_state_gradients_of_neutral_lih"]
fn the_hse06_excited_state_gradients_of_neutral_lih_follow_their_energies() {
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    {
        let base = hse06_lithium_hydride(&ctx);
        hybrid_check(
            "LiH HSE06 singlet TDA",
            (&base, [1, 1], 60),
            Some(Spin::Singlet),
            Approximation::TammDancoff,
            &[(0, 2), (1, 2)],
            2e-5,
        );
    }
    assert_eq!(ctx.memory_used(), 0);
}

/// r2SCAN's third-order change: the `t^2` coefficients of its three potential parts along a
/// curve of densities, gradients and kinetic densities `tau + t dtau + t^2 tau2` (the curvature of
/// the gauge-invariant kinetic density along an imaginary change) against a Richardson central
/// difference of the kernel's action along the curve's tangent, on smooth fields whose
/// iso-orbital indicator stays inside the interpolation's smooth branch.
#[test]
fn the_meta_third_order_change_is_the_derivative_of_its_kernel() {
    let ctx = Context::new(Some(2), 256 << 20).unwrap();
    {
        let geometry = Geometry::new(
            vec![1],
            vec![[3., 3., 3.]],
            [[6., 0., 0.], [0., 6., 0.], [0., 0., 6.]],
            [false; 3],
            LengthUnit::Bohr,
        )
        .unwrap();
        let grid = SpatialGrid::from_geometry(&geometry, [0.4; 3], false, false).unwrap();
        let n = grid.index.points();
        let shape = |k: usize, center: [f64; 3], width: f64| {
            let r = grid.point(k);
            let d2: f64 = (0..3).map(|a| (r[a] - center[a]).powi(2)).sum();
            (-d2 / width).exp()
        };
        let mut rho = vec![0.; 2 * n];
        let mut gradient = vec![0.; 6 * n];
        let mut tau = vec![0.; 2 * n];
        for k in 0..n {
            let r = grid.point(k);
            for (s, (scale, center, width)) in
                [(0.3, [3., 3., 3.], 1.2), (0.2, [3.2, 2.9, 3.1], 1.0)]
                    .into_iter()
                    .enumerate()
            {
                let e = scale * shape(k, center, width);
                rho[s * n + k] = e + 1e-3;
                let mut g2 = 0.;
                for a in 0..3 {
                    let g = -2. * (r[a] - center[a]) / width * e;
                    gradient[(3 * s + a) * n + k] = g;
                    g2 += g * g;
                }
                // Von Weizsacker plus a uniform-gas share: alpha near 0.7.
                tau[s * n + k] = g2 / (8. * rho[s * n + k]) + 2.0 * rho[s * n + k].powf(5. / 3.);
            }
        }
        for sign in [1., -1.] {
            let (mut dn, mut dg, mut dtau, mut tau2) = (
                vec![0.; 2 * n],
                vec![0.; 6 * n],
                vec![0.; 2 * n],
                vec![0.; 2 * n],
            );
            for k in 0..n {
                for s in 0..2 {
                    let f = if s == 0 { 1. } else { sign * 0.7 };
                    let e = shape(k, [2.6 + 0.8 * s as f64, 3.3, 3.], 0.8);
                    dn[s * n + k] = 0.01 * f * e;
                    dtau[s * n + k] = 0.004 * f * e;
                    tau2[s * n + k] = -0.002 * shape(k, [3.1, 3.0, 2.8 + 0.3 * s as f64], 0.9);
                    for a in 0..3 {
                        dg[(3 * s + a) * n + k] = 0.003 * f * e * (a as f64 - 1.);
                    }
                }
            }
            let c2 =
                second_change_meta(&ctx, &rho, &tau, &gradient, &dn, &dg, &dtau, &tau2).unwrap();
            // The kernel at x(t), applied to the tangent x'(t) = (dn, dg, dtau + 2 t tau2).
            let apply = |h: f64| {
                let at = |base: &[f64], change: &[f64]| -> Vec<f64> {
                    base.iter().zip(change).map(|(b, c)| b + h * c).collect()
                };
                let kinetic: Vec<f64> = (0..2 * n)
                    .map(|k| tau[k] + h * dtau[k] + h * h * tau2[k])
                    .collect();
                let kernel =
                    MetaKernel::new(&ctx, &at(&rho, &dn), &kinetic, &at(&gradient, &dg)).unwrap();
                let tangent: Vec<f64> = (0..2 * n).map(|k| dtau[k] + 2. * h * tau2[k]).collect();
                let mut out = vec![0.; 10 * n];
                let (local, rest) = out.split_at_mut(2 * n);
                let (kin, flux) = rest.split_at_mut(2 * n);
                kernel
                    .apply(
                        &dn,
                        &dg,
                        &tangent,
                        MetaChange {
                            local,
                            tau: kin,
                            flux,
                        },
                    )
                    .unwrap();
                out
            };
            let fd = |h: f64| -> Vec<f64> {
                let (a, b) = (apply(h), apply(-h));
                (0..10 * n).map(|k| (a[k] - b[k]) / (4. * h)).collect()
            };
            let (coarse, fine) = (fd(2e-3), fd(1e-3));
            for (part, range) in [
                ("local", 0..2 * n),
                ("tau", 2 * n..4 * n),
                ("flux", 4 * n..10 * n),
            ] {
                let scale = c2.as_slice()[range.clone()]
                    .iter()
                    .fold(0f64, |a, v| a.max(v.abs()));
                let worst = range
                    .map(|k| ((4. * fine[k] - coarse[k]) / 3. - c2.as_slice()[k]).abs())
                    .fold(0f64, f64::max);
                println!("r2scan {part} (sign {sign}): max |c2 - FD| = {worst:.3e} of max |c2| = {scale:.3e}");
                assert!(
                    scale > 0. && worst < 1e-6 * scale,
                    "{part}: {worst:.3e} of {scale:.3e}"
                );
            }
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

/// r2SCAN's imaginary first-order Fock (the current-density term of `tau~ = tau - |j|^2 / 2n`)
/// of a general antisymmetric change against a Richardson central difference of the complex
/// Fock builder of the real-time path at `D +- i h V`.
#[test]
#[ignore = "slow: run: cargo test --profile release-test -p mandacaru-core --test molecular_excited_forces -- --ignored the_meta_imaginary_fock_is_the_complex_builders"]
fn the_meta_imaginary_fock_is_the_complex_builders() {
    let ctx = Context::new(Some(2), 512 << 20).unwrap();
    {
        let base = integrals(&ctx, &H4, "pbe");
        let state = solve(&base, Method::Ks("r2scan"), [2, 2]);
        let m = base.m();
        let zero = DMatrix::zeros(m, m);
        let (_, _, frame) = seams::first_and_second(&base, &state, [&zero, &zero]).unwrap();
        let v = DMatrix::from_fn(m, m, |p, q| {
            let x = ((p * 7 + q * 3) as f64 * 0.37).sin() * 0.05;
            match p.cmp(&q) {
                std::cmp::Ordering::Less => x,
                std::cmp::Ordering::Greater => -((q * 7 + p * 3) as f64 * 0.37).sin() * 0.05,
                std::cmp::Ordering::Equal => 0.,
            }
        });
        for (label, up, down) in [("same", v.clone(), v.clone()), ("opposite", v.clone(), -&v)] {
            let analytic = seams::imaginary_fock(&base, &state, [&up, &down])
                .unwrap()
                .unwrap();
            let d = &state.densities[0];
            let change =
                |x: &DMatrix<f64>| (&frame * x * frame.transpose()).map(|z| C64::new(0., z));
            let fock = |h: f64| {
                let f = seams::complex_fock(
                    &base,
                    "r2scan",
                    &[
                        d + change(&up) * C64::new(h, 0.),
                        d + change(&down) * C64::new(h, 0.),
                    ],
                )
                .unwrap();
                [0, 1].map(|s| frame.transpose() * f[s].map(|z| z.im) * &frame)
            };
            let central = |h: f64| {
                let (a, b) = (fock(h), fock(-h));
                [0, 1].map(|s| (&a[s] - &b[s]) / (2. * h))
            };
            let (coarse, fine) = (central(2e-3), central(1e-3));
            let scale = analytic[0].amax();
            let worst = (0..2)
                .map(|s| (&analytic[s] - (&fine[s] * 4. - &coarse[s]) / 3.).amax())
                .fold(0f64, f64::max);
            println!("r2scan ({label}): |Im F1 - dIm F| = {worst:.3e} of {scale:.3e}");
            assert!(
                scale > 0. && worst < 1e-6 * scale,
                "{label}: {worst:.3e} of {scale:.3e}"
            );
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

/// r2SCAN excited-state gradients on the synthetic hydrogens against reconverged energies: a
/// closed-shell singlet (Tamm-Dancoff) and triplet (Casida) of H4, and the unrestricted H2+.
#[test]
#[ignore = "slow: run: cargo test --profile release-test -p mandacaru-core --test molecular_excited_forces -- --ignored the_r2scan_excited_state_gradients_follow_their_energies"]
fn the_r2scan_excited_state_gradients_follow_their_energies() {
    let ctx = Context::new(Some(2), 512 << 20).unwrap();
    {
        let base = integrals(&ctx, &H4, "pbe");
        check(
            "H4 r2scan singlet TDA",
            &base,
            "r2scan",
            Spin::Singlet,
            Approximation::TammDancoff,
            &[(1, 0), (2, 2)],
        );
        check(
            "H4 r2scan triplet Casida",
            &base,
            "r2scan",
            Spin::Triplet,
            Approximation::Casida,
            &[(3, 1)],
        );
    }
    assert_eq!(ctx.memory_used(), 0);
}

/// LiH on the library (`pbe-sr`: lithium's partial core and its kinetic density) for r2SCAN.
fn r2scan_lithium_hydride(ctx: &Context) -> MolecularIntegrals {
    use paw_molecular_support::{analytic_shells, dataset, molecule};
    let mut arrays = mandacaru_core::paw::molecular::Arrays::new();
    let li = dataset::species(ctx, &mut arrays, &dataset::read(ctx, "pbe-sr", "Li"));
    let h = dataset::species(ctx, &mut arrays, &dataset::read(ctx, "pbe-sr", "H"));
    molecule(
        ctx,
        &[3, 1],
        &[[0.1, -0.05, -1.5], [-0.05, 0.08, 1.52]],
        12.,
        0.5,
        (vec![li, h], vec![0, 1], arrays),
        "raw",
        &analytic_shells(&[0, 1]),
    )
}

/// A converged r2SCAN state at 1e-11 in at most 300 iterations, restricted or unrestricted.
fn meta_solve(
    integrals: &MolecularIntegrals,
    particles: [usize; 2],
    unrestricted: bool,
) -> MeanField {
    let options = mandacaru_core::molecular::ScfOptions {
        tolerance: 1e-11,
        max_iterations: 300,
        ..Default::default()
    };
    let state = integrals
        .kohn_sham("r2scan", particles, unrestricted, &options)
        .unwrap();
    assert!(state.converged, "the r2SCAN SCF did not converge");
    state
}

/// The r2SCAN excited-state gradient against Richardson differences of reconverged excited
/// energies. On the library's LiH at a 0.5 Bohr grid the energy has a large third derivative
/// along a displacement (cause not isolated): central differences at 1e-3, 5e-4 and 2.5e-4 Bohr
/// fall off as `h^2` with the same coefficient (1.4e3 Ha/Bohr^3, measured 2026-11-15), so the
/// reference is the Richardson value of the steps `h, h/2`, taken where it agrees with that of
/// `h/2, h/4` to 1e-7 Ha/Bohr. The current-density term's explicit part is printed and must be
/// resolved.
fn meta_check(
    label: &str,
    (base, particles): (&MolecularIntegrals, [usize; 2]),
    spin: Option<Spin>,
    approximation: Approximation,
    components: &[(usize, usize)],
    h: f64,
) {
    let request = || {
        let mut r = request(spin.unwrap_or(Spin::Singlet), approximation);
        r.spin = spin;
        r
    };
    let state = meta_solve(base, particles, spin.is_none());
    let excited = excited_state(base, &state, &request(), 0).unwrap();
    let current = excited
        .parts
        .iter()
        .find(|(n, _)| *n == "current")
        .expect("an r2SCAN excited state has its current part")
        .1
        .clone();
    let (mut worst, mut without) = (0f64, 0f64);
    for &(atom, axis) in components {
        let e = |d: f64| {
            let moved = displaced(base, atom, axis, d);
            let state = meta_solve(&moved, particles, spin.is_none());
            let found = excitations(&moved, &state, &request()).unwrap();
            state.free_energy() + moved.nuclear + found.energies[0]
        };
        let central: Vec<f64> = [h, 0.5 * h, 0.25 * h]
            .iter()
            .map(|&step| (e(step) - e(-step)) / (2. * step))
            .collect();
        let richardson = [0, 1].map(|j| (4. * central[j + 1] - central[j]) / 3.);
        let k = 3 * atom + axis;
        let analytic = excited.gradient.total.as_slice()[k];
        println!("{label} ({atom}, {axis}): analytic {analytic:.10}, central {central:?}, Richardson {richardson:?}");
        assert!(
            (richardson[0] - richardson[1]).abs() < 1e-7,
            "{label}: the Richardson differences disagree: {richardson:?}"
        );
        let reference = richardson[1];
        println!(
            "{label} ({atom}, {axis}): analytic {analytic:.10} FD {reference:.10} diff {:.2e}, current {:.2e}",
            analytic - reference,
            current[k]
        );
        worst = worst.max((analytic - reference).abs());
        without = without.max((analytic - current[k] - reference).abs());
    }
    println!(
        "{label}: omega {:.8} Ha, separation {:.2e}, Z residual {:.1e}, max |dE| {worst:.3e} Ha/Bohr, without the current part {without:.3e}",
        excited.excitation, excited.separation, excited.z_residual
    );
    assert!(worst < 2e-7, "{label}: {worst:.3e} Ha/Bohr");
    assert!(
        without > 10. * worst.max(1e-9),
        "{label}: the current part is not resolved"
    );
}

#[test]
#[ignore = "slow, the r2SCAN gate on the library: run: cargo test --profile release-test -p mandacaru-core --test molecular_excited_forces -- --ignored the_r2scan_excited_state_gradients_of_lih_follow_their_energies"]
fn the_r2scan_excited_state_gradients_of_lih_follow_their_energies() {
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    {
        let base = r2scan_lithium_hydride(&ctx);
        for (label, particles, spin, approximation, components) in [
            (
                "LiH r2SCAN singlet TDA",
                [1, 1],
                Some(Spin::Singlet),
                Approximation::TammDancoff,
                &[(0, 2), (1, 0), (1, 2)][..],
            ),
            (
                "LiH r2SCAN triplet Casida",
                [1, 1],
                Some(Spin::Triplet),
                Approximation::Casida,
                &[(1, 2)][..],
            ),
            // LiH- is unstable in r2SCAN (its unrestricted singlet orbital Hessian is not positive,
            // its Casida omega^2 -2.5e-4 Ha^2, measured 2026-11-15): the cation is the open shell.
            (
                "LiH+ r2SCAN unrestricted Casida",
                [1, 0],
                None,
                Approximation::Casida,
                &[(0, 2), (1, 2)][..],
            ),
        ] {
            meta_check(
                label,
                (&base, particles),
                spin,
                approximation,
                components,
                2e-4,
            );
        }
        // Both spin channels occupied: the closed shell solved unrestricted, whose lowest
        // spin-conserving root is the Ms = 0 triplet. (The synthetic neutral H3 doublet's
        // r2SCAN differences scatter by 1e-2 across steps of 5e-5 to 2e-4 Bohr, measured
        // 2026-11-15, cause not measured, so it is no gate; NH2 is the two-channel gate.)
        meta_check(
            "LiH r2SCAN unrestricted TDA",
            (&base, [1, 1]),
            None,
            Approximation::TammDancoff,
            &[(0, 2), (1, 2)],
            2e-4,
        );
    }
    assert_eq!(ctx.memory_used(), 0);
}

/// The noncollinear kernel of LDA and PBE (T8) against a Richardson central difference of the
/// spinor Pauli-channel potentials themselves (`noncollinear::rotate`, the collinear potentials
/// of the frame densities, `noncollinear::unrotate`): on a magnetization that turns in space,
/// and on an unpolarized state, where the energy depends on `|dm|` and the kernel is the
/// collinear spin kernel along each axis: exact for LDA along any change and for PBE along a
/// change whose magnetization keeps one direction and sign. Along a turning change the PBE
/// difference is not linear in the change (the miss is printed, not gated).
#[test]
fn the_noncollinear_kernel_is_the_derivative_of_the_spinor_potential() {
    use mandacaru_core::models::{noncollinear, xc};
    let ctx = Context::new(Some(2), 256 << 20).unwrap();
    {
        let geometry = Geometry::new(
            vec![1],
            vec![[3., 3., 3.]],
            [[6., 0., 0.], [0., 6., 0.], [0., 0., 6.]],
            [false; 3],
            LengthUnit::Bohr,
        )
        .unwrap();
        let grid = SpatialGrid::from_geometry(&geometry, [0.4; 3], false, false).unwrap();
        let n = grid.index.points();
        let shape = |k: usize, center: [f64; 3], width: f64| {
            let r = grid.point(k);
            let d2: f64 = (0..3).map(|a| (r[a] - center[a]).powi(2)).sum();
            (-d2 / width).exp()
        };
        let potential = |name: &str, fields: &[f64]| -> Vec<f64> {
            let rotated = noncollinear::rotate(&ctx, n, fields, None).unwrap();
            let frame: Vec<f64> = rotated
                .densities
                .as_slice()
                .iter()
                .map(|v| v.max(0.))
                .collect();
            let terms =
                xc::evaluate_fields_relativistic(&ctx, &grid, &frame, None, None, name, false)
                    .unwrap();
            let (v, _) = noncollinear::unrotate(
                &ctx,
                n,
                fields,
                &rotated,
                terms.potentials.as_slice(),
                None,
                None,
            )
            .unwrap();
            v.as_slice().to_vec()
        };
        let mut charge = vec![0.; n];
        let mut turning = vec![0.; 4 * n];
        for k in 0..n {
            let r = grid.point(k);
            let e = 0.3 * shape(k, [3., 3., 3.], 1.2) + 0.2 * shape(k, [3.3, 2.8, 3.1], 1.0);
            charge[k] = e + 1e-3;
            let (theta, phi) = (0.6 + 0.3 * (r[0] - 3.), 0.4 * (r[1] - 3.) + 0.2 * r[2]);
            turning[k] = charge[k];
            turning[n + k] = 0.5 * e * theta.sin() * phi.cos();
            turning[2 * n + k] = 0.5 * e * theta.sin() * phi.sin();
            turning[3 * n + k] = 0.5 * e * theta.cos();
        }
        let mut unpolarized = vec![0.; 4 * n];
        unpolarized[..n].copy_from_slice(&charge);
        // A general change, and one whose magnetization keeps the direction (1, 2, 2) / 3 and sign.
        let (mut general, mut fixed) = (vec![0.; 4 * n], vec![0.; 4 * n]);
        for k in 0..n {
            general[k] = 0.01 * shape(k, [2.6, 3.3, 3.], 0.8);
            fixed[k] = general[k];
            let amplitude = 0.006 * shape(k, [3.2, 3.1, 2.7], 0.9);
            for a in 0..3 {
                general[(a + 1) * n + k] = 0.005
                    * shape(k, [2.7 + 0.3 * a as f64, 3.2, 2.9 + 0.2 * a as f64], 0.7)
                    * (a as f64 - 0.8);
                fixed[(a + 1) * n + k] = amplitude * [1., 2., 2.][a] / 3.;
            }
        }
        for (label, name, state, change, gated) in [
            ("LDA turning", "lda", &turning, &general, true),
            ("PBE turning", "pbe", &turning, &general, true),
            ("LDA unpolarized", "lda", &unpolarized, &general, true),
            (
                "PBE unpolarized, one direction",
                "pbe",
                &unpolarized,
                &fixed,
                true,
            ),
            (
                "PBE unpolarized, turning change",
                "pbe",
                &unpolarized,
                &general,
                false,
            ),
        ] {
            let kernel = NoncollinearKernel::new(&ctx, &grid, state, name, false).unwrap();
            let mut scratch = kernel.scratch(&ctx).unwrap();
            let mut analytic = vec![0.; 4 * n];
            kernel
                .apply(&ctx, &grid, change, &mut analytic, &mut scratch)
                .unwrap();
            let central = |h: f64| -> Vec<f64> {
                let at = |t: f64| -> Vec<f64> {
                    potential(
                        name,
                        &(0..4 * n)
                            .map(|k| state[k] + t * change[k])
                            .collect::<Vec<_>>(),
                    )
                };
                let (a, b) = (at(h), at(-h));
                (0..4 * n).map(|k| (a[k] - b[k]) / (2. * h)).collect()
            };
            let (coarse, fine) = (central(2e-3), central(1e-3));
            // The potential has no transverse part where `|m|` is at or below the floor
            // (`noncollinear::NORM_FLOOR`), so a magnetization channel is compared only where the
            // smaller step's `|m|` clears it by 1e3 at both ends.
            let clear = |k: usize| {
                let point = k % n;
                k < n
                    || [-1e-3, 1e-3].iter().all(|t| {
                        (1..4)
                            .map(|a| (state[a * n + point] + t * change[a * n + point]).powi(2))
                            .sum::<f64>()
                            .sqrt()
                            > 1e3 * noncollinear::NORM_FLOOR
                    })
            };
            let scale = analytic.iter().fold(0f64, |a, v| a.max(v.abs()));
            let worst = (0..4 * n)
                .filter(|&k| clear(k))
                .map(|k| ((4. * fine[k] - coarse[k]) / 3. - analytic[k]).abs())
                .fold(0f64, f64::max);
            let skipped = (n..4 * n).filter(|&k| !clear(k)).count();
            println!(
                "{label}: max |dv - FD| = {worst:.3e} of max |dv| = {scale:.3e} ({skipped} of {} magnetization values below the floor)",
                3 * n
            );
            if gated {
                assert!(
                    scale > 0. && worst < 1e-6 * scale,
                    "{label}: {worst:.3e} of {scale:.3e}"
                );
            }
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

/// T8: the second change of the noncollinear kernel at a Kramers-unpolarized reference
/// (`NoncollinearKernel::second_change`) is the derivative of its action along the reference's
/// charge: `d/ds K(f + s e) delta = [c(e + delta) - c(e - delta)] / 2` for a charge change `e`
/// (the reference stays unpolarized) and a change `delta` whose magnetization turns, against
/// Richardson differences of `apply` (LDA, PBE); and at magnetized references whose axis turns in
/// space, one of them through a node of `m`, along a change `e` with magnetization (the turn of
/// the local spin axis).
#[test]
fn the_noncollinear_second_change_is_the_derivative_of_its_kernel() {
    let ctx = Context::new(Some(2), 256 << 20).unwrap();
    {
        let geometry = Geometry::new(
            vec![1],
            vec![[3., 3., 3.]],
            [[6., 0., 0.], [0., 6., 0.], [0., 0., 6.]],
            [false; 3],
            LengthUnit::Bohr,
        )
        .unwrap();
        let grid = SpatialGrid::from_geometry(&geometry, [0.4; 3], false, false).unwrap();
        let n = grid.index.points();
        let shape = |k: usize, center: [f64; 3], width: f64| {
            let r = grid.point(k);
            let d2: f64 = (0..3).map(|a| (r[a] - center[a]).powi(2)).sum();
            (-d2 / width).exp()
        };
        let mut state = vec![0.; 4 * n];
        let (mut e, mut delta) = (vec![0.; 4 * n], vec![0.; 4 * n]);
        for k in 0..n {
            state[k] =
                0.3 * shape(k, [3., 3., 3.], 1.2) + 0.2 * shape(k, [3.3, 2.8, 3.1], 1.0) + 1e-3;
            e[k] = 0.02 * shape(k, [3.2, 2.9, 3.1], 0.9) - 0.01 * shape(k, [2.8, 3.1, 2.9], 0.6);
            delta[k] = 0.01 * shape(k, [2.6, 3.3, 3.], 0.8);
            for a in 0..3 {
                delta[(a + 1) * n + k] = 0.005
                    * shape(k, [2.7 + 0.3 * a as f64, 3.2, 2.9 + 0.2 * a as f64], 0.7)
                    * (a as f64 - 0.8);
            }
        }
        // A magnetized reference whose axis turns in space, moved along a change `e_m` that
        // carries magnetization too.
        // and one whose magnetization passes through zero on the plane x = 3.05 Bohr, between grid
        // planes (the three 1/|m| pieces of the turn cancel as |m| falls).
        let mut magnetized = vec![0.; 4 * n];
        let mut node = vec![0.; 4 * n];
        let mut e_m = e.clone();
        for k in 0..n {
            let r = grid.point(k);
            let bulk = 0.3 * shape(k, [3., 3., 3.], 1.2) + 0.2 * shape(k, [3.3, 2.8, 3.1], 1.0);
            let (theta, phi) = (0.6 + 0.3 * (r[0] - 3.), 0.4 * (r[1] - 3.) + 0.2 * r[2]);
            let axis = [
                theta.sin() * phi.cos(),
                theta.sin() * phi.sin(),
                theta.cos(),
            ];
            let signed = ((r[0] - 3.05) / 0.3).tanh();
            magnetized[k] = bulk + 1e-3;
            node[k] = bulk + 1e-3;
            for a in 0..3 {
                magnetized[(a + 1) * n + k] = 0.5 * bulk * axis[a];
                node[(a + 1) * n + k] = 0.5 * bulk * signed * axis[a];
                e_m[(a + 1) * n + k] = 0.004
                    * shape(k, [3.1, 2.8 + 0.2 * a as f64, 3.2], 0.8)
                    * (0.7 - a as f64 * 0.6);
            }
        }
        for (label, reference, e, unpolarized) in [
            ("unpolarized", &state, &e, true),
            ("magnetized", &magnetized, &e_m, false),
            ("magnetized through a node", &node, &e_m, false),
        ] {
            for name in ["lda", "pbe"] {
                let kernel = NoncollinearKernel::new(&ctx, &grid, reference, name, false).unwrap();
                assert_eq!(kernel.unpolarized(), unpolarized);
                let change = |x: &[f64], sign: f64| -> Vec<f64> {
                    (0..4 * n).map(|k| e[k] + sign * x[k]).collect()
                };
                let (plus, minus) = (
                    kernel
                        .second_change(&ctx, &grid, &change(&delta, 1.))
                        .unwrap(),
                    kernel
                        .second_change(&ctx, &grid, &change(&delta, -1.))
                        .unwrap(),
                );
                let analytic: Vec<f64> = (0..4 * n)
                    .map(|k| 0.5 * (plus.as_slice()[k] - minus.as_slice()[k]))
                    .collect();
                let action = |s: f64| -> Vec<f64> {
                    let at: Vec<f64> = (0..4 * n).map(|k| reference[k] + s * e[k]).collect();
                    let moved = NoncollinearKernel::new(&ctx, &grid, &at, name, false).unwrap();
                    let mut scratch = moved.scratch(&ctx).unwrap();
                    let mut out = vec![0.; 4 * n];
                    moved
                        .apply(&ctx, &grid, &delta, &mut out, &mut scratch)
                        .unwrap();
                    out
                };
                let central = |h: f64| -> Vec<f64> {
                    let (a, b) = (action(h), action(-h));
                    (0..4 * n).map(|k| (a[k] - b[k]) / (2. * h)).collect()
                };
                let (coarse, fine) = (central(2e-3), central(1e-3));
                let scale = analytic.iter().fold(0f64, |a, v| a.max(v.abs()));
                let channel = |c: usize| {
                    (c * n..(c + 1) * n)
                        .map(|k| ((4. * fine[k] - coarse[k]) / 3. - analytic[k]).abs())
                        .fold(0f64, f64::max)
                };
                let misses: Vec<f64> = (0..4).map(channel).collect();
                println!(
                    "{label} {name}: second change of {scale:.3e}, misses per channel {misses:?}"
                );
                assert!(
                    scale > 0. && misses.iter().all(|&w| w < 1e-6 * scale),
                    "{label} {name}: {misses:?} of {scale:.3e}"
                );
            }
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

/// T8: the excited-state gradient of a spinor reference without spin-orbit coupling is the
/// collinear one (LiH on the library's `lda-sr` and `pbe-sr`, a partial core): its singlet state
/// against `excited_state` of the restricted singlet, and one member of its triplet (the seam
/// skips the multiplet check; spin rotation keeps the three degenerate at every geometry)
/// against the restricted triplet, Tamm-Dancoff and Casida; with the Lagrangian's identities
/// (`omega` rebuilt from its pieces, `A` Hermitian, the fixed contraction at the reference is the
/// ground-state gradient).
#[test]
#[ignore = "slow: 163 s (measured 2026-11-16); run: cargo test --profile release-test -p mandacaru-core --test molecular_excited_forces -- --ignored the_spinor_excited_state_gradient_of_a_closed_shell --nocapture"]
fn the_spinor_excited_state_gradient_of_a_closed_shell_is_the_collinear_one() {
    use paw_molecular_support::{analytic_shells, dataset, molecule};
    let ctx = Context::new(Some(3), 1 << 30).unwrap();
    let mut failures: Vec<String> = vec![];
    {
        for (set, functional) in [("lda-sr", "lda"), ("pbe-sr", "pbe")] {
            let mut arrays = mandacaru_core::paw::molecular::Arrays::new();
            let li = dataset::species(&ctx, &mut arrays, &dataset::read(&ctx, set, "Li"));
            let h = dataset::species(&ctx, &mut arrays, &dataset::read(&ctx, set, "H"));
            let base = molecule(
                &ctx,
                &[3, 1],
                &[[0.1, -0.05, -1.5], [-0.05, 0.08, 1.52]],
                12.,
                0.5,
                (vec![li, h], vec![0, 1], arrays),
                "raw",
                &analytic_shells(&[0, 1]),
            );
            // Lithium's partial core leaves one valence electron: LiH is a closed shell of two.
            let restricted = solve(&base, Method::Ks(functional), [1, 1]);
            let spinor = base.spinor_kohn_sham(functional, 2, &options()).unwrap();
            assert!(spinor.converged);
            for approximation in [Approximation::TammDancoff, Approximation::Casida] {
                let mut r = request(Spin::Singlet, approximation);
                r.spin = None;
                r.num_states = 8;
                let found = excitations(&base, &spinor, &r).unwrap();
                for spin in [Spin::Singlet, Spin::Triplet] {
                    let collinear =
                        excited_state(&base, &restricted, &request(spin, approximation), 0)
                            .unwrap();
                    let Some(state) = found
                        .energies
                        .iter()
                        .position(|w| (w - collinear.excitation).abs() < 1e-7)
                    else {
                        failures.push(format!(
                            "{set} {}: no spinor state at the {} root {}",
                            approximation.name(),
                            spin.name(),
                            collinear.excitation
                        ));
                        continue;
                    };
                    let (total, parts, diagnostics) =
                        seams::spinor_state(&base, &spinor, &found, state).unwrap();
                    let expected = collinear.gradient.total.as_slice();
                    let miss = total
                        .iter()
                        .zip(expected)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f64, f64::max);
                    let scale = expected
                        .iter()
                        .zip(&parts[0].1)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f64, f64::max);
                    let label = format!(
                        "LiH {set} {} {} (spinor state {state})",
                        approximation.name(),
                        spin.name()
                    );
                    println!(
                        "{label}: omega {:.10} rebuilt {:.10}, asymmetry {:.1e}, odd {:.1e}, ground seam {:.1e}; excitation's gradient {scale:.3e}, max |spinor - collinear| {miss:.2e} Ha/Bohr",
                        found.energies[state], diagnostics.omega, diagnostics.asymmetry, diagnostics.odd, diagnostics.ground
                    );
                    for (name, values) in &parts {
                        println!(
                            "  {name}: {:?}",
                            values
                                .iter()
                                .map(|v| format!("{v:+.6e}"))
                                .collect::<Vec<_>>()
                        );
                    }
                    println!(
                        "  collinear: {:?}",
                        expected
                            .iter()
                            .map(|v| format!("{v:+.6e}"))
                            .collect::<Vec<_>>()
                    );
                    if (diagnostics.omega - found.energies[state]).abs() > 1e-9
                        || diagnostics.asymmetry > 1e-7
                        || diagnostics.ground > 1e-9
                        || miss > 1e-8
                        || scale < 1e-4
                    {
                        failures.push(label.clone());
                    }
                    // Tracking by overlap: the singlet follows itself, at the same geometry
                    // (overlap 1 whatever the phases) and after a small step of the hydrogen.
                    if spin == Spin::Singlet {
                        let picked = pick(&base, &spinor, &r, state, None).unwrap();
                        let again = pick(&base, &spinor, &r, 0, Some(&picked.followed)).unwrap();
                        let moved = displaced(&base, 1, 2, 2e-2);
                        let moved_state =
                            moved.spinor_kohn_sham(functional, 2, &options()).unwrap();
                        let next =
                            pick(&moved, &moved_state, &r, 0, Some(&picked.followed)).unwrap();
                        let (same, step) =
                            (again.overlap.unwrap_or(0.), next.overlap.unwrap_or(0.));
                        println!("  followed: root {} overlap {same:.12} at the same geometry, root {} overlap {step:.6} after 2e-2 Bohr", again.root, next.root);
                        if again.root != state
                            || (same - 1.).abs() > 1e-10
                            || next.root != state
                            || step < 0.99
                        {
                            failures.push(format!("{label}: tracking"));
                        }
                    }
                }
            }
        }
    }
    assert_eq!(ctx.memory_used(), 0);
    assert!(failures.is_empty(), "{failures:#?}");
}

/// T8: the excited-state gradient of a magnetized spinor reference without spin-orbit coupling
/// (the synthetic H2+, one electron, and the bent neutral H3, two up and one down; LDA, PBE
/// refusing by name; Tamm-Dancoff and Casida, H3's Casida, where both spins flip, from the dense
/// solve of `spinor::dense_casida`). Its spin-conserving
/// state is the unrestricted one (the longitudinal kernel), so its gradient is the unrestricted
/// `excited_state`'s; a spin-flip state (the transverse kernel and the turn of the spin axis) has
/// no collinear partner and is gated against Richardson differences of reconverged `E_0 + omega`.
/// The two global spin rotations that turn the moment are zero modes of `A + B`: the right-hand
/// side has no share along them, and the Z-vector is solved on their complement.
#[test]
#[ignore = "slow: 740 s (measured 2026-11-16 at load 15 to 21); run: cargo test --profile release-test -p mandacaru-core --test molecular_excited_forces -- --ignored the_spinor_excited_state_gradient_of_a_magnetized_reference --nocapture"]
fn the_spinor_excited_state_gradient_of_a_magnetized_reference_follows_its_energy() {
    let ctx = Context::new(Some(3), 512 << 20).unwrap();
    let mut failures: Vec<String> = vec![];
    {
        // (label, geometry, particles, functional, spin-flip components). PBE refuses by name:
        // measured 2026-11-16, its spin-flip gradients missed reconverged energies by 3e-2 to
        // 1.6e-1 on H2+ (whose down frame density vanishes) and by up to 2.6 on H3, and did not
        // sum to zero.
        let h2: &[(usize, usize)] = &[(0, 2), (1, 2)];
        let h3: &[(usize, usize)] = &[(0, 0), (1, 2), (2, 1)];
        for (molecule, geometry, particles, functional, components) in [
            ("H2+", &H2[..], [1usize, 0usize], "lda", h2),
            ("H2+", &H2[..], [1, 0], "pbe", h2),
            ("H3", &H3[..], [2, 1], "lda", h3),
            ("H3", &H3[..], [2, 1], "pbe", h3),
        ] {
            let ion = integrals(&ctx, geometry, functional);
            let electrons = particles[0] + particles[1];
            let polarized = ion
                .kohn_sham(functional, particles, true, &options())
                .unwrap();
            let spinor = ion
                .spinor_kohn_sham(functional, electrons, &options())
                .unwrap();
            assert!(spinor.converged);
            for approximation in [Approximation::TammDancoff, Approximation::Casida] {
                let mut open = request(Spin::Singlet, approximation);
                open.spin = None;
                if molecule == "H3" && approximation == Approximation::Casida && functional != "lda"
                {
                    // Both spins flip: `A + B` and `A - B` have different zero modes. The roots are
                    // solved densely (gated in `casida_where_both_spins_flip_...`); with PBE the
                    // forces refuse by name, as every magnetized gradient functional does.
                    let mut r = open.clone();
                    r.num_states = 6;
                    if excitations(&ion, &spinor, &r).is_err() {
                        failures.push(format!("{molecule} {functional}: Casida roots must solve"));
                    }
                    let refusal = excited_state(&ion, &spinor, &r, 0);
                    if !refusal
                        .as_ref()
                        .err()
                        .is_some_and(|e| e.to_string().contains("gradient functional"))
                    {
                        failures.push(format!(
                            "{molecule} {functional}: Casida forces must refuse by name"
                        ));
                    }
                    continue;
                }
                let collinear = excited_state(&ion, &polarized, &open, 0).unwrap();
                let unrestricted = excitations(&ion, &polarized, &open).unwrap();
                let mut r = open.clone();
                r.num_states = 6;
                if functional == "pbe" {
                    let refusal = excited_state(&ion, &spinor, &r, 0);
                    if !refusal
                        .as_ref()
                        .err()
                        .is_some_and(|e| e.to_string().contains("gradient functional"))
                    {
                        failures.push(format!(
                            "{molecule} pbe {}: must refuse by name",
                            approximation.name()
                        ));
                    }
                    continue;
                }
                let found = excitations(&ion, &spinor, &r).unwrap();
                let label = format!("{molecule} {functional} {}", approximation.name());
                println!(
                    "{label}: spinor states {:?}, unrestricted {:?}",
                    found
                        .energies
                        .iter()
                        .map(|w| format!("{w:.8}"))
                        .collect::<Vec<_>>(),
                    unrestricted
                        .energies
                        .iter()
                        .map(|w| format!("{w:.8}"))
                        .collect::<Vec<_>>()
                );
                // (omega rebuilt, asymmetry, ground seam, zero modes, |(A + B) n|, share of r).
                let identities = |name: &str,
                                  d: (f64, f64, f64, usize, f64, f64),
                                  omega: f64,
                                  failures: &mut Vec<String>| {
                    println!(
                        "  {name}: omega {omega:.10} rebuilt {:.10}, asymmetry {:.1e}, ground seam {:.1e}, {} zero modes (|(A + B) n| {:.1e}, share of r {:.1e})",
                        d.0, d.1, d.2, d.3, d.4, d.5
                    );
                    if (d.0 - omega).abs() > 1e-9
                        || d.1 > 1e-7
                        || d.2 > 1e-9
                        || d.3 != 2
                        || d.4 > 1e-7
                        || d.5 > 1e-6
                    {
                        failures.push(format!("{label} {name}: identities {d:?}"));
                    }
                };
                // The spin-conserving state against the unrestricted gradient.
                match found
                    .energies
                    .iter()
                    .position(|w| (w - collinear.excitation).abs() < 1e-7)
                {
                    None => failures.push(format!(
                        "{label}: no spinor state at the unrestricted root {}",
                        collinear.excitation
                    )),
                    Some(state) => {
                        let (total, _, d) =
                            seams::spinor_state(&ion, &spinor, &found, state).unwrap();
                        identities(
                            &format!("spin-conserving state {state}"),
                            (
                                d.omega,
                                d.asymmetry,
                                d.ground,
                                d.modes,
                                d.mode_action,
                                d.mode_share,
                            ),
                            found.energies[state],
                            &mut failures,
                        );
                        let expected = collinear.gradient.total.as_slice();
                        let miss = total
                            .iter()
                            .zip(expected)
                            .map(|(a, b)| (a - b).abs())
                            .fold(0f64, f64::max);
                        println!(
                            "  spin-conserving: max |spinor - unrestricted| {miss:.2e} Ha/Bohr"
                        );
                        if miss > 1e-8 {
                            failures.push(format!("{label}: spin-conserving gradient {miss:.2e}"));
                        }
                    }
                }
                // The lowest spin-flip state clear of its neighbors and of the zero modes.
                let separation = |energies: &[f64], j: usize| {
                    (0..energies.len())
                        .filter(|&k| k != j)
                        .map(|k| (energies[k] - energies[j]).abs())
                        .fold(f64::INFINITY, f64::min)
                };
                let Some(root) = (0..found.energies.len() - 1).find(|&j| {
                    let w = found.energies[j];
                    w > 1e-3
                        && separation(&found.energies, j) > 1e-3
                        && unrestricted.energies.iter().all(|u| (u - w).abs() > 1e-6)
                }) else {
                    failures.push(format!("{label}: no separated spin-flip state"));
                    continue;
                };
                let excited = excited_state(&ion, &spinor, &r, root).unwrap();
                let (_, _, d) = seams::spinor_state(&ion, &spinor, &found, root).unwrap();
                identities(
                    &format!("spin-flip state {root}"),
                    (
                        d.omega,
                        d.asymmetry,
                        d.ground,
                        d.modes,
                        d.mode_action,
                        d.mode_share,
                    ),
                    excited.excitation,
                    &mut failures,
                );
                let energy = |atom: usize, axis: usize, step: f64| {
                    let moved = displaced(&ion, atom, axis, step);
                    let state = moved
                        .spinor_kohn_sham(functional, electrons, &options())
                        .unwrap();
                    assert!(state.converged);
                    let found = excitations(&moved, &state, &r).unwrap();
                    state.free_energy() + moved.nuclear + found.energies[root]
                };
                for &(atom, axis) in components {
                    let central =
                        |h: f64| (energy(atom, axis, h) - energy(atom, axis, -h)) / (2. * h);
                    let (coarse, fine) = (central(1e-3), central(5e-4));
                    let reference = (4. * fine - coarse) / 3.;
                    let analytic = excited.gradient.total.as_slice()[3 * atom + axis];
                    println!("  spin-flip ({atom}, {axis}): analytic {analytic:+.10} FD {reference:+.10} diff {:.2e}", analytic - reference);
                    if (analytic - reference).abs() > 2e-7 {
                        failures.push(format!(
                            "{label} spin-flip ({atom}, {axis}): {:.2e}",
                            analytic - reference
                        ));
                    }
                }
            }
        }
    }
    assert_eq!(ctx.memory_used(), 0);
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The spinor response (T8) of a closed shell without spin-orbit coupling is the collinear one:
/// its Tamm-Dancoff and Casida roots are the restricted singlets once and the triplets three
/// times (the noncollinear kernel's isotropic spin part at an unpolarized reference), on the
/// synthetic H4 with LDA and PBE.
#[test]
#[ignore = "slow: 164 s (measured 2026-11-16); run: cargo test --profile release-test -p mandacaru-core --test molecular_excited_forces -- --ignored the_spinor_response_of_a_closed_shell_is_the_collinear_one"]
fn the_spinor_response_of_a_closed_shell_is_the_collinear_one() {
    let ctx = Context::new(Some(2), 512 << 20).unwrap();
    {
        let base = integrals(&ctx, &H4, "pbe");
        for functional in ["lda", "pbe"] {
            let restricted = solve(&base, Method::Ks(functional), [2, 2]);
            let spinor = base.spinor_kohn_sham(functional, 4, &options()).unwrap();
            assert!(spinor.converged);
            for approximation in [Approximation::TammDancoff, Approximation::Casida] {
                let mut collinear = vec![];
                for (spin, copies) in [(Spin::Singlet, 1), (Spin::Triplet, 3)] {
                    let mut r = request(spin, approximation);
                    r.num_states = 6;
                    for omega in excitations(&base, &restricted, &r)
                        .unwrap()
                        .energies
                        .iter()
                        .take(6)
                    {
                        collinear.extend(std::iter::repeat_n(*omega, copies));
                    }
                }
                collinear.sort_by(f64::total_cmp);
                let mut r = request(Spin::Singlet, approximation);
                r.spin = None;
                r.num_states = 8;
                let found = excitations(&base, &spinor, &r).unwrap();
                let worst = (0..8)
                    .map(|j| (found.energies[j] - collinear[j]).abs())
                    .fold(0f64, f64::max);
                println!(
                    "H4 {functional} {}: spinor {:?}\n  collinear {:?}\n  max |d omega| = {worst:.3e} Ha",
                    approximation.name(),
                    &found.energies[..8],
                    &collinear[..8]
                );
                assert!(
                    worst < 1e-8,
                    "{functional} {}: {worst:.3e}",
                    approximation.name()
                );
            }
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

/// The r2SCAN gate on a spin-polarized open shell whose two channels are both occupied and
/// unequal: the NH2 doublet (`pbe-sr`, lithium's gate's protocol), the spin-resolved weights,
/// kinetic curvatures and currents of the second spin included.
#[test]
#[ignore = "slow, the r2SCAN open-shell gate on the library: run: cargo test --profile release-test -p mandacaru-core --test molecular_excited_forces -- --ignored the_r2scan_excited_state_gradient_of_nh2_follows_its_energy"]
fn the_r2scan_excited_state_gradient_of_nh2_follows_its_energy() {
    use paw_molecular_support::{analytic_shells, dataset, molecule};
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    {
        let mut arrays = mandacaru_core::paw::molecular::Arrays::new();
        let n = dataset::species(&ctx, &mut arrays, &dataset::read(&ctx, "pbe-sr", "N"));
        let h = dataset::species(&ctx, &mut arrays, &dataset::read(&ctx, "pbe-sr", "H"));
        let base = molecule(
            &ctx,
            &[7, 1, 1],
            &[[0.02, 0.05, 0.1], [1.55, 0.03, -1.0], [-1.45, -0.08, -1.1]],
            12.,
            0.5,
            (vec![n, h], vec![0, 1, 1], arrays),
            "raw",
            &analytic_shells(&[0, 1]),
        );
        meta_check(
            "NH2 r2SCAN unrestricted TDA",
            (&base, [4, 3]),
            None,
            Approximation::TammDancoff,
            &[(0, 2), (1, 0)],
            2e-4,
        );
    }
    assert_eq!(ctx.memory_used(), 0);
}

/// T8, Casida where both spins flip (`spinor::dense_casida`, the seventeenth session): the
/// magnetized spinor H3 (two up, one down, LDA, no spin-orbit coupling), whose spin turns `N`
/// (of `A + B`) differ from `i N` (of `A - B`), is solved densely on the complement of `N`. Its
/// spectrum holds every spin-conserving root of the unrestricted Casida problem (the collinear
/// reference's own path, an independent operator), each as a pair of equal real roots, at the base
/// geometry and at two displaced ones (where Davidson did not converge); every root's residual in
/// the undeflated `(A - B)(A + B)` is at round-off.
#[test]
#[ignore = "slow: 48 s (measured 2026-11-16, three geometries); run: cargo test --profile release-test -p mandacaru-core --test molecular_excited_forces -- --ignored casida_where_both_spins_flip"]
fn casida_where_both_spins_flip_holds_the_spin_conserving_roots_at_displaced_geometries() {
    let ctx = Context::new(Some(2), 512 << 20).unwrap();
    let mut geometries = vec![H3.to_vec()];
    for (atom, axis, step) in [(1usize, 2usize, 0.05), (2, 0, -0.08)] {
        let mut g = H3.to_vec();
        g[atom][axis] += step;
        geometries.push(g);
    }
    for (index, geometry) in geometries.iter().enumerate() {
        let ion = integrals(&ctx, geometry, "lda");
        let polarized = ion.kohn_sham("lda", [2, 1], true, &options()).unwrap();
        let spinor = ion.spinor_kohn_sham("lda", 3, &options()).unwrap();
        assert!(polarized.converged && spinor.converged);
        let mut open = request(Spin::Singlet, Approximation::Casida);
        open.spin = None;
        open.num_states = 4;
        let unrestricted = excitations(&ion, &polarized, &open).unwrap();
        let mut r = open.clone();
        r.num_states = 12;
        let found = excitations(&ion, &spinor, &r).unwrap();
        let worst_residual = found.residuals.iter().copied().fold(0f64, f64::max);
        let mut worst = 0f64;
        for omega in &unrestricted.energies {
            let nearest = found
                .energies
                .iter()
                .map(|w| (w - omega).abs())
                .fold(f64::INFINITY, f64::min);
            worst = worst.max(nearest);
        }
        println!(
            "H3 spinor LDA Casida (both spins flip), geometry {index}: spinor states {:?}; \
             unrestricted {:?}; worst spin-conserving miss {worst:.2e} Ha; largest residual \
             {worst_residual:.2e}",
            found
                .energies
                .iter()
                .map(|w| format!("{w:.9}"))
                .collect::<Vec<_>>(),
            unrestricted
                .energies
                .iter()
                .map(|w| format!("{w:.9}"))
                .collect::<Vec<_>>()
        );
        assert!(worst < 1e-8, "geometry {index}: {worst:e}");
        assert!(
            worst_residual < 1e-8,
            "geometry {index}: {worst_residual:e}"
        );
    }
}
