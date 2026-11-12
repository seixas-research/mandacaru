//! The screened-hybrid terms of a PAW-LCAO molecule.
//!
//! The erfc-screened Coulomb energy of two concentric normalized Gaussians has
//! a closed form, which fixes the radial kernel the one-center tensor is built
//! with. The augmentation of the pair densities (compensation charges and the
//! one-center tensor) then has to be the bare one minus its long-range part, so
//! it tends to the bare augmentation as the screening vanishes, and a screened
//! hybrid on a synthetic hydrogen-like PAW setup converges and lowers the
//! energy of the semilocal functional it replaces by the exchange it adds.
mod paw_molecular_support;

use mandacaru_core::{
    math::spline::simpson,
    molecular::ScfOptions,
    paw::onecenter_radial::{long_range_radial_kernel, radial_coulomb},
    Context, Geometry, LengthUnit,
};
use paw_molecular_support::{grid_pairs, integrals, H2};
use std::f64::consts::PI;

/// `(4 pi)^2 int int rho_1 rho_2 K_0` for the bare kernel `1/r_>` and the
/// `erf(omega R)/R` one, of two spherical densities tabulated on `r`.
fn concentric(r: &[f64], rho_1: &[f64], rho_2: &[f64], omega: f64, ctx: &Context) -> (f64, f64) {
    let bare = radial_coulomb(r, rho_1, rho_2, 0).unwrap();
    let kernel = &long_range_radial_kernel(ctx, r, r, &[0], omega, None).unwrap()[0];
    let weights: Vec<f64> = (0..r.len())
        .map(|i| {
            let mut unit = vec![0.; r.len()];
            unit[i] = 1.;
            simpson(r, &unit).unwrap() * r[i] * r[i]
        })
        .collect();
    let long: f64 = (0..r.len())
        .map(|i| {
            rho_1[i]
                * weights[i]
                * (0..r.len())
                    .map(|j| kernel[i * r.len() + j] * rho_2[j] * weights[j])
                    .sum::<f64>()
        })
        .sum();
    let scale = (4. * PI).powi(2);
    (scale * bare, scale * long)
}

#[test]
fn the_screened_coulomb_energy_of_two_gaussians_is_analytic() {
    // rho_a = (a/pi)^(3/2) exp(-a r^2) integrates to one. In reciprocal space the
    // Coulomb kernel carries exp(-k^2 / 4 omega^2) under erf, and each Gaussian
    // exp(-k^2 / 4a): the energy is (2 / sqrt(pi)) sqrt(q) with
    // 1 / q = 1 / a + 1 / b (+ 1 / omega^2 under erf). Bare: (2/sqrt(pi)) sqrt(ab/(a+b)).
    let ctx = Context::new(Some(1), 256 << 20).unwrap();
    let r: Vec<f64> = (0..1600).map(|i| 9. * i as f64 / 1599.).collect();
    for (a, b, omega) in [(0.7, 1.9, 0.11), (1.3, 1.3, 0.11), (0.9, 2.5, 0.4)] {
        let density = |e: f64| -> Vec<f64> {
            r.iter()
                .map(|r| (e / PI).powf(1.5) * (-e * r * r).exp())
                .collect()
        };
        let (bare, long) = concentric(&r, &density(a), &density(b), omega, &ctx);
        let p = a * b / (a + b);
        let q = 1. / (1. / a + 1. / b + 1. / (omega * omega));
        let two_over_root_pi = 2. / PI.sqrt();
        let bare_exact = two_over_root_pi * p.sqrt();
        let long_exact = two_over_root_pi * q.sqrt();
        assert!(
            (bare - bare_exact).abs() < 2e-5 * bare_exact,
            "bare {bare} vs {bare_exact}"
        );
        assert!(
            (long - long_exact).abs() < 2e-5 * long_exact,
            "erf {long} vs {long_exact} (a {a}, b {b}, omega {omega})"
        );
        let short = bare - long;
        let short_exact = bare_exact - long_exact;
        assert!(short > 0. && (short - short_exact).abs() < 2e-5 * bare_exact);
    }
}

#[test]
fn the_screened_augmentation_tends_to_the_bare_one_as_the_screening_vanishes() {
    // erfc(omega R)/R -> 1/R: the compensation charges and the one-center tensor
    // under the screened kernel must be the bare augmentation of the same
    // Hartree-Fock model, entry by entry, in the orthonormal frame.
    let ctx = Context::new(Some(2), 768 << 20).unwrap();
    for kind in ["dual", "raw"] {
        let base = integrals(&ctx, &H2, kind);
        let bare = &base.corrections.as_ref().unwrap().pairs;
        let setup = base.paw_setup.as_ref().unwrap();
        let screened = setup.screened_exchange(&base, 1e-6).unwrap();
        let m = base.m();
        let mut worst = 0f64;
        let mut size = 0f64;
        for a in 0..m {
            for b in 0..m {
                for c in 0..m {
                    for d in 0..m {
                        let reference = bare.value(a, b, c, d);
                        size = size.max(reference.abs());
                        worst = worst.max((screened.pairs.value(a, b, c, d) - reference).abs());
                    }
                }
            }
        }
        println!("{kind}: max |screened - bare| = {worst:.3e}, max |bare| = {size:.3e}");
        assert!(size > 1e-3, "the augmentation is not trivially zero");
        assert!(worst < 2e-5 * size, "{kind}: {worst:e} against {size:e}");
    }
    drop(ctx);
}

#[test]
fn the_screened_pair_integrals_lose_the_unit_charge_long_range_part() {
    // Every augmented pair density of the orthonormal basis carries the charge
    // delta_ab, so its long-range interaction, as omega -> 0, is the constant
    // 2 omega / sqrt(pi) of the kernel times delta_ab delta_cd: the grid pairs and
    // the compensation charges together, with the one-center tensor adding none.
    let ctx = Context::new(Some(2), 768 << 20).unwrap();
    for kind in ["dual", "raw"] {
        let base = integrals(&ctx, &H2, kind);
        let setup = base.paw_setup.as_ref().unwrap();
        let m = base.m();
        let (w1, w2) = (0.005, 0.01);
        let total = |omega: f64| -> Vec<f64> {
            let mut grid = grid_pairs(&ctx, &base, omega);
            let screened = setup.screened_exchange(&base, omega).unwrap();
            for a in 0..m {
                for b in 0..m {
                    for c in 0..m {
                        for d in 0..m {
                            grid[((a * m + b) * m + c) * m + d] += screened.pairs.value(a, b, c, d);
                        }
                    }
                }
            }
            grid
        };
        let (first, second) = (total(w1), total(w2));
        let unit = 2. * (w2 - w1) / PI.sqrt();
        let mut worst = 0f64;
        for a in 0..m {
            for b in 0..m {
                for c in 0..m {
                    for d in 0..m {
                        let i = ((a * m + b) * m + c) * m + d;
                        let expected = if a == b && c == d { unit } else { 0. };
                        worst = worst.max((first[i] - second[i] - expected).abs());
                    }
                }
            }
        }
        println!("{kind}: max |LR - 2 dw/sqrt(pi) delta delta| = {worst:.3e} (unit {unit:.3e})");
        assert!(worst < 2e-5, "{kind}: {worst:e}");
    }
    drop(ctx);
}

#[test]
fn the_screened_augmentation_is_smaller_than_the_bare_one_and_symmetric() {
    let ctx = Context::new(Some(2), 768 << 20).unwrap();
    let base = integrals(&ctx, &H2, "dual");
    let bare = &base.corrections.as_ref().unwrap().pairs;
    let setup = base.paw_setup.as_ref().unwrap();
    let screened = setup.screened_exchange(&base, 0.11).unwrap();
    let m = base.m();
    let (mut sum_bare, mut sum_short) = (0., 0.);
    for a in 0..m {
        for b in 0..m {
            // A pair density of an orbital with itself: its self-repulsion.
            sum_bare += bare.value(a, b, a, b);
            sum_short += screened.pairs.value(a, b, a, b);
            for c in 0..m {
                for d in 0..m {
                    let (left, right) = (
                        screened.pairs.value(a, b, c, d),
                        screened.pairs.value(c, d, a, b),
                    );
                    assert!((left - right).abs() < 1e-12 * (1. + left.abs()));
                    let swapped = screened.pairs.value(b, a, c, d);
                    assert!((left - swapped).abs() < 1e-12 * (1. + left.abs()));
                }
            }
        }
    }
    println!("sum (ab|ab): bare {sum_bare:.6}, erfc-screened {sum_short:.6}");
    assert!(sum_short.abs() < sum_bare.abs());
    // The frozen short-range semilocal exchange of hydrogen's 1s-like reference:
    // one atom's constant is negative (a hole energy difference) and finite.
    assert!(screened.constant.is_finite());
    assert!(screened.one_body.iter().all(|v| v.is_finite()));
    drop(ctx);
}

#[test]
#[ignore = "slow: 20 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test paw_molecular_screened -- --ignored a_screened_hybrid_on_a_paw_molecule_converges_and_is_stationary_in_the_forces"]
fn a_screened_hybrid_on_a_paw_molecule_converges_and_is_stationary_in_the_forces() {
    let ctx = Context::new(Some(2), 768 << 20).unwrap();
    let base = integrals(&ctx, &H2, "dual");
    let options = ScfOptions {
        tolerance: 1e-11,
        max_iterations: 300,
        ..Default::default()
    };
    let pbe = base.kohn_sham("pbe", [1, 1], false, &options).unwrap();
    let hse = base.kohn_sham("hse06", [1, 1], false, &options).unwrap();
    assert!(pbe.converged && hse.converged);
    println!(
        "H2 synthetic: E(pbe) = {:.8}, E(hse06) = {:.8} Ha",
        pbe.electronic_energy, hse.electronic_energy
    );
    // The stationary gradient is the finite difference of the fixed-density energy,
    // which now includes the screened terms, against an SCF finite difference.
    let gradient = base.gradient(&hse).unwrap();
    let energy = |delta: f64| {
        let mut positions = base.geometry.positions().to_vec();
        positions[1][2] += delta;
        let geometry = Geometry::new(
            base.geometry.numbers().to_vec(),
            positions,
            *base.geometry.cell(),
            base.geometry.pbc(),
            LengthUnit::Bohr,
        )
        .unwrap();
        let moved = base.at_geometry(geometry).unwrap();
        let state = moved.kohn_sham("hse06", [1, 1], false, &options).unwrap();
        assert!(state.converged);
        state.electronic_energy + moved.nuclear
    };
    let h = 1e-3;
    let coarse = (energy(h) - energy(-h)) / (2. * h);
    let fine = (energy(0.5 * h) - energy(-0.5 * h)) / h;
    let reference = (4. * fine - coarse) / 3.;
    let analytic = gradient.total.as_slice()[5];
    let to_ev = 27.211386245988 / 0.529177210903;
    println!(
        "H2 HSE06 PAW force: analytic {:.8e}, finite difference {reference:.8e} Ha/Bohr",
        analytic
    );
    assert!(
        (analytic - reference).abs() * to_ev < 1e-5,
        "{analytic:e} vs {reference:e}"
    );
    drop((gradient, pbe, hse, base));
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn the_sphere_polarization_kernel_is_the_derivative_of_its_fock() {
    // The spheres' spin polarization P (the open-shell part of the frozen short-range semilocal
    // exchange) through its exact second derivative (`SpinTerms::polarization_kernel`), against a
    // central difference of its Fock matrices, at a spin-polarized point and at a closed shell.
    // This is the fast gate of the term the triplet and unrestricted kernels of HSE06 carry.
    use mandacaru_core::{models::HSE06_OMEGA, quantum::C64};
    use nalgebra::DMatrix;
    let ctx = Context::new(Some(1), 512 << 20).unwrap();
    let base = integrals(&ctx, &H2, "raw");
    let state = base
        .kohn_sham(
            "lda",
            [1, 1],
            false,
            &ScfOptions {
                tolerance: 1e-10,
                ..Default::default()
            },
        )
        .unwrap();
    let screened = base
        .paw_setup
        .as_ref()
        .unwrap()
        .screened_exchange(&base, HSE06_OMEGA)
        .unwrap();
    let d = state.densities[0].map(|z: C64| z.re);
    let m = d.nrows();
    // A symmetric orbital product to polarize with, and two generic directions.
    let c = DMatrix::from_fn(m, m, |i, j| ((i * 7 + j * 3) as f64 * 0.37).sin());
    let product = |a: usize, b: usize| {
        let x = c.column(a) * c.column(b).transpose();
        (&x + x.transpose()) * 0.5
    };
    let polarize = product(0, 1) * 0.05;
    let (d_up, d_down) = (
        product(1, 2) * 0.3 + product(0, 0) * 0.1,
        product(2, 3) * -0.2,
    );
    let check = |up: &DMatrix<f64>, down: &DMatrix<f64>, du: &DMatrix<f64>, dd: &DMatrix<f64>| {
        let kernel = screened.spin.polarization_kernel(&ctx, up, down).unwrap();
        let [ku, kd] = kernel.apply(du, dd).unwrap();
        let fock = |h: f64| {
            screened
                .spin
                .polarization(&ctx, &(up + du * h), &(down + dd * h))
                .unwrap()
                .1
        };
        let difference = |h: f64| {
            let (p, q) = (fock(h), fock(-h));
            [(&p[0] - &q[0]) / (2. * h), (&p[1] - &q[1]) / (2. * h)]
        };
        let (coarse, fine) = (difference(2e-4), difference(1e-4));
        let estimate = [
            (&fine[0] * 4. - &coarse[0]) / 3.,
            (&fine[1] * 4. - &coarse[1]) / 3.,
        ];
        let scale = ku.amax().max(kd.amax());
        let error = (&ku - &estimate[0]).amax().max((&kd - &estimate[1]).amax());
        (scale, error)
    };
    let (up, down) = (&d + &polarize, &d - &polarize);
    let (scale, error) = check(&up, &down, &d_up, &d_down);
    assert!(scale > 1e-6, "the polarization kernel vanishes: {scale:e}");
    assert!(
        error < 1e-7 * scale,
        "polarized point: {error:e} of {scale:e}"
    );
    // At a closed shell: a magnetizing change responds, an equal change of both spins does not.
    let (scale, error) = check(&d, &d, &d_up, &(-&d_up));
    assert!(
        scale > 1e-6 && error < 1e-7 * scale,
        "closed shell, opposite spins: {error:e} of {scale:e}"
    );
    let kernel = screened.spin.polarization_kernel(&ctx, &d, &d).unwrap();
    let [ku, kd] = kernel.apply(&d_up, &d_up).unwrap();
    assert!(
        ku.amax().max(kd.amax()) < 1e-12 * scale,
        "an equal change of both spins keeps P at zero"
    );
}
