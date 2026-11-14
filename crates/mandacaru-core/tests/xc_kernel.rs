//! The adiabatic XC kernel (`models::xc::point_hessian`, `models::xc_kernel::Kernel`): the
//! exact second derivative of the energy whose first derivative is the potential.
use mandacaru_core::{
    math::SpatialGrid,
    models::{xc, xc_kernel::Kernel},
    Context, Geometry, LengthUnit,
};

/// Sample points `(n_up, n_dn, s_uu, s_ud, s_dd)`: unpolarized, polarized, a weak and a
/// strong gradient, a low and a high density (both Perdew-Zunger branches).
const POINTS: [[f64; 5]; 5] = [
    [0.12, 0.12, 0.010, 0.010, 0.010],
    [0.30, 0.07, 0.050, 0.012, 0.004],
    [0.002, 0.0015, 1e-5, 8e-6, 6e-6],
    [1.7, 1.6, 2.0, 1.8, 1.7],
    [0.04, 0.01, 0.003, -0.001, 0.0008],
];

#[test]
fn the_point_hessian_is_the_derivative_of_the_partials() {
    for name in ["lda", "pbe"] {
        for point in POINTS {
            let (energy, first, second) = xc::point_hessian(name, point, false).unwrap();
            let (reference, partials) = xc::partials(name, point).unwrap();
            assert!((energy - reference).abs() <= 1e-14 * reference.abs().max(1e-300));
            for i in 0..5 {
                assert!(
                    (first[i] - partials[i]).abs() <= 1e-13 * partials[i].abs().max(1e-12),
                    "{name} {point:?}: first partial {i} {} against {}",
                    first[i],
                    partials[i]
                );
            }
            for j in 0..5 {
                // A central difference of the analytic partials, with Richardson.
                let step = 1e-4 * point[j].abs().max(1e-6);
                let at = |h: f64| {
                    let mut p = point;
                    p[j] += h;
                    xc::partials(name, p).unwrap().1
                };
                let difference = |h: f64| {
                    let (plus, minus) = (at(h), at(-h));
                    std::array::from_fn::<f64, 5, _>(|i| (plus[i] - minus[i]) / (2. * h))
                };
                let (coarse, fine) = (difference(2. * step), difference(step));
                for i in 0..5 {
                    let estimate = (4. * fine[i] - coarse[i]) / 3.;
                    let scale = second[i][j].abs().max(1e-6 * second[i][i].abs()).max(1e-10);
                    assert!(
                        (second[i][j] - estimate).abs() <= 1e-6 * scale,
                        "{name} {point:?}: d2f/dx{i}dx{j} = {} against {estimate}",
                        second[i][j]
                    );
                    assert_eq!(
                        second[i][j], second[j][i],
                        "{name}: the kernel is symmetric"
                    );
                }
            }
        }
    }
}

#[test]
fn the_relativistic_kernel_follows_the_relativistic_partials() {
    for name in ["lda", "pbe"] {
        let point = [4.0, 4.0, 3.0, 3.0, 3.0];
        let (_, first, second) = xc::point_hessian(name, point, true).unwrap();
        let step = 1e-5;
        let mut p = point;
        p[0] += step;
        let plus = xc::point_partials(name, [p[0], p[1], p[2], p[3], p[4], 0., 0.], true)
            .unwrap()
            .1;
        p[0] -= 2. * step;
        let minus = xc::point_partials(name, [p[0], p[1], p[2], p[3], p[4], 0., 0.], true)
            .unwrap()
            .1;
        let fd = (plus[0] - minus[0]) / (2. * step);
        assert!(
            (second[0][0] - fd).abs() < 1e-6 * fd.abs(),
            "{name}: {} vs {fd}",
            second[0][0]
        );
        assert!(first[0].is_finite());
    }
}

fn gaussian_densities(spacing: f64) -> (SpatialGrid, Vec<f64>, Vec<f64>) {
    let geometry = Geometry::new(
        vec![1],
        vec![[3., 3., 3.]],
        [[6., 0., 0.], [0., 6., 0.], [0., 0., 6.]],
        [false; 3],
        LengthUnit::Bohr,
    )
    .unwrap();
    let grid = SpatialGrid::from_geometry(&geometry, [spacing; 3], false, false).unwrap();
    let n = grid.index.points();
    let (mut rho, mut delta) = (vec![0.; 2 * n], vec![0.; 2 * n]);
    for k in 0..n {
        let p = grid.point(k);
        let u = [p[0] - 3., p[1] - 3.1, p[2] - 2.9];
        let r2 = u.iter().map(|x| x * x).sum::<f64>();
        rho[k] = 0.4 * (-r2 / 1.1).exp() + 1e-6;
        rho[n + k] = 0.3 * (-r2 / 0.9).exp() + 1e-6;
        // A transition-like change: signed, with a node, decaying faster than the density.
        delta[k] = u[0] * (-r2 / 0.8).exp();
        delta[n + k] = (u[2] - 0.2 * u[1]) * (-r2 / 0.7).exp();
    }
    (grid, rho, delta)
}

#[test]
fn the_grid_kernel_is_the_derivative_of_the_grid_potential() {
    let ctx = Context::new(Some(1), 256 << 20).unwrap();
    let (grid, rho, delta) = gaussian_densities(0.3);
    let n = grid.index.points();
    for name in ["lda", "pbe"] {
        let kernel = Kernel::new(&ctx, &grid, &rho, name, false).unwrap();
        let mut scratch = kernel.scratch(&ctx).unwrap();
        let mut response = vec![0.; 2 * n];
        kernel
            .apply(&grid, &delta, &mut response, &mut scratch)
            .unwrap();
        let potential = |h: f64| {
            let shifted: Vec<f64> = rho.iter().zip(&delta).map(|(r, d)| r + h * d).collect();
            xc::evaluate_fields(&ctx, &grid, &shifted, None, name)
                .unwrap()
                .potentials
                .as_slice()
                .to_vec()
        };
        let difference = |h: f64| -> Vec<f64> {
            let (plus, minus) = (potential(h), potential(-h));
            plus.iter()
                .zip(&minus)
                .map(|(p, m)| (p - m) / (2. * h))
                .collect()
        };
        let (coarse, fine) = (difference(2e-5), difference(1e-5));
        let scale = response.iter().fold(0f64, |m, v| m.max(v.abs()));
        let error = (0..2 * n)
            .map(|k| (response[k] - (4. * fine[k] - coarse[k]) / 3.).abs())
            .fold(0f64, f64::max);
        assert!(
            error < 1e-6 * scale,
            "{name}: kernel against the potential's difference {error:e} of {scale:e}"
        );
    }
}

#[test]
fn the_grid_kernel_is_symmetric() {
    let ctx = Context::new(Some(1), 256 << 20).unwrap();
    let (grid, rho, delta) = gaussian_densities(0.35);
    let n = grid.index.points();
    let other: Vec<f64> = (0..2 * n)
        .map(|k| {
            let p = grid.point(k % n);
            (p[1] - 3.).cos()
                * (-(p[0] - 3.).powi(2) - (p[2] - 3.).powi(2)).exp()
                * if k < n { 1. } else { -0.5 }
        })
        .collect();
    for name in ["lda", "pbe"] {
        let kernel = Kernel::new(&ctx, &grid, &rho, name, false).unwrap();
        let mut scratch = kernel.scratch(&ctx).unwrap();
        let (mut ka, mut kb) = (vec![0.; 2 * n], vec![0.; 2 * n]);
        kernel.apply(&grid, &delta, &mut ka, &mut scratch).unwrap();
        kernel.apply(&grid, &other, &mut kb, &mut scratch).unwrap();
        let ab: f64 = other.iter().zip(&ka).map(|(x, y)| x * y).sum();
        let ba: f64 = delta.iter().zip(&kb).map(|(x, y)| x * y).sum();
        assert!(
            (ab - ba).abs() <= 1e-11 * ab.abs().max(ba.abs()),
            "{name}: {ab} against {ba}"
        );
    }
}

#[test]
fn the_screened_hybrid_point_kernel_is_the_derivative_of_its_partials() {
    // The semilocal part of HSE06: PBE correlation, the hole-model exchange and minus a quarter
    // of its short-range part, through the adaptive quadrature of the screened hole.
    for point in POINTS {
        let (_, first, second) = xc::point_hessian("hse06", point, false).unwrap();
        let partials = |p: [f64; 5]| {
            xc::point_partials("hse06", [p[0], p[1], p[2], p[3], p[4], 0., 0.], false)
                .unwrap()
                .1
        };
        let reference = partials(point);
        for i in 0..5 {
            assert!(
                (first[i] - reference[i]).abs() <= 1e-12 * reference[i].abs().max(1e-12),
                "hse06 {point:?}: first partial {i}"
            );
        }
        for j in 0..5 {
            let step = 1e-4 * point[j].abs().max(1e-6);
            let difference = |h: f64| {
                let (mut plus, mut minus) = (point, point);
                plus[j] += h;
                minus[j] -= h;
                let (a, b) = (partials(plus), partials(minus));
                std::array::from_fn::<f64, 5, _>(|i| (a[i] - b[i]) / (2. * h))
            };
            let (coarse, fine) = (difference(2. * step), difference(step));
            for i in 0..5 {
                let estimate = (4. * fine[i] - coarse[i]) / 3.;
                let scale = second[i][j].abs().max(1e-6 * second[i][i].abs()).max(1e-10);
                assert!(
                    (second[i][j] - estimate).abs() <= 1e-6 * scale,
                    "hse06 {point:?}: d2f/dx{i}dx{j} = {} against {estimate}",
                    second[i][j]
                );
            }
        }
    }
}

#[test]
fn the_screened_hybrid_grid_kernel_is_the_derivative_of_its_grid_potential() {
    let ctx = Context::new(Some(1), 256 << 20).unwrap();
    let (grid, rho, delta) = gaussian_densities(0.35);
    let n = grid.index.points();
    let kernel = Kernel::new(&ctx, &grid, &rho, "hse06", false).unwrap();
    let mut scratch = kernel.scratch(&ctx).unwrap();
    let mut response = vec![0.; 2 * n];
    kernel
        .apply(&grid, &delta, &mut response, &mut scratch)
        .unwrap();
    let potential = |h: f64| {
        let shifted: Vec<f64> = rho.iter().zip(&delta).map(|(r, d)| r + h * d).collect();
        xc::evaluate_fields(&ctx, &grid, &shifted, None, "hse06")
            .unwrap()
            .potentials
            .as_slice()
            .to_vec()
    };
    let difference = |h: f64| -> Vec<f64> {
        let (plus, minus) = (potential(h), potential(-h));
        plus.iter()
            .zip(&minus)
            .map(|(p, m)| (p - m) / (2. * h))
            .collect()
    };
    let (coarse, fine) = (difference(2e-5), difference(1e-5));
    let scale = response.iter().fold(0f64, |m, v| m.max(v.abs()));
    let error = (0..2 * n)
        .map(|k| (response[k] - (4. * fine[k] - coarse[k]) / 3.).abs())
        .fold(0f64, f64::max);
    assert!(error < 1e-6 * scale, "hse06: {error:e} of {scale:e}");
}

/// r2SCAN points `(n_up, n_dn, s_uu, s_ud, s_dd, tau_up, tau_dn)` with the iso-orbital indicator of
/// each spin above zero, polarized and not, a low and a high density.
const META_POINTS: [[f64; 7]; 4] = [
    [0.12, 0.10, 0.010, 0.008, 0.009, 0.05, 0.04],
    [0.30, 0.07, 0.050, 0.012, 0.004, 0.12, 0.03],
    [0.004, 0.003, 2e-5, 1.5e-5, 1.2e-5, 0.004, 0.003],
    [1.5, 1.5, 1.2, 1.2, 1.2, 1.9, 1.9],
];

#[test]
fn the_meta_gga_point_hessian_is_the_derivative_of_its_partials() {
    for point in META_POINTS {
        let (energy, first, second) = xc::point_hessian_meta(point).unwrap();
        let partials = |p: [f64; 7]| xc::point_partials("r2scan", p, false).unwrap();
        let (reference, slopes) = partials(point);
        assert!((energy - reference).abs() <= 1e-14 * reference.abs());
        for i in 0..7 {
            assert!(
                (first[i] - slopes[i]).abs() <= 1e-12 * slopes[i].abs().max(1e-12),
                "{point:?}: first partial {i}"
            );
        }
        for j in 0..7 {
            let step = 1e-4 * point[j].abs();
            let difference = |h: f64| {
                let (mut plus, mut minus) = (point, point);
                plus[j] += h;
                minus[j] -= h;
                let (a, b) = (partials(plus).1, partials(minus).1);
                std::array::from_fn::<f64, 7, _>(|i| (a[i] - b[i]) / (2. * h))
            };
            let (coarse, fine) = (difference(2. * step), difference(step));
            for i in 0..7 {
                let estimate = (4. * fine[i] - coarse[i]) / 3.;
                let scale = second[i][j].abs().max(1e-6 * second[i][i].abs()).max(1e-10);
                assert!(
                    (second[i][j] - estimate).abs() <= 1e-6 * scale,
                    "{point:?}: d2f/dx{i}dx{j} = {} against {estimate}",
                    second[i][j]
                );
                assert_eq!(second[i][j], second[j][i]);
            }
        }
    }
}

#[test]
fn the_meta_gga_grid_kernel_is_the_derivative_of_its_grid_potential() {
    // The density gradient and the kinetic density are inputs (as the orbitals give them); the
    // kernel's three outputs are the changes of dE/dn, dE/dtau and the flux dE/d(grad n).
    use mandacaru_core::models::xc_kernel::{MetaChange, MetaKernel};
    let ctx = Context::new(Some(1), 256 << 20).unwrap();
    let (grid, rho, delta) = gaussian_densities(0.35);
    let n = grid.index.points();
    let mut gradient = vec![0.; 6 * n];
    let mut d_gradient = vec![0.; 6 * n];
    for s in 0..2 {
        for a in 0..3 {
            xc::derivative(
                &grid,
                &rho[s * n..(s + 1) * n],
                a,
                &mut gradient[(s * 3 + a) * n..(s * 3 + a + 1) * n],
            )
            .unwrap();
            xc::derivative(
                &grid,
                &delta[s * n..(s + 1) * n],
                a,
                &mut d_gradient[(s * 3 + a) * n..(s * 3 + a + 1) * n],
            )
            .unwrap();
        }
    }
    // A kinetic density above the von Weizsaecker one, and a smooth change of it.
    let tau: Vec<f64> = (0..2 * n)
        .map(|k| {
            let (s, p) = (k / n, k % n);
            let g2: f64 = (0..3).map(|a| gradient[(s * 3 + a) * n + p].powi(2)).sum();
            g2 / (8. * rho[k]) + 0.3 * rho[k]
        })
        .collect();
    let d_tau: Vec<f64> = (0..2 * n).map(|k| 0.2 * delta[k] + 0.05 * rho[k]).collect();
    let kernel = MetaKernel::new(&ctx, &rho, &tau, &gradient).unwrap();
    let (mut local, mut tau_out, mut flux) = (vec![0.; 2 * n], vec![0.; 2 * n], vec![0.; 6 * n]);
    kernel
        .apply(
            &delta,
            &d_gradient,
            &d_tau,
            MetaChange {
                local: &mut local,
                tau: &mut tau_out,
                flux: &mut flux,
            },
        )
        .unwrap();
    let parts = |h: f64| {
        let shift = |a: &[f64], b: &[f64]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| x + h * y)
                .collect::<Vec<f64>>()
        };
        let xc = xc::evaluate_fields_relativistic(
            &ctx,
            &grid,
            &shift(&rho, &delta),
            Some(&shift(&tau, &d_tau)),
            Some(&shift(&gradient, &d_gradient)),
            "r2scan",
            false,
        )
        .unwrap();
        (
            xc.potentials.as_slice().to_vec(),
            xc.tau_potentials.unwrap().as_slice().to_vec(),
            xc.gradient_potentials.unwrap().as_slice().to_vec(),
        )
    };
    let difference = |h: f64| {
        let (p, m) = (parts(h), parts(-h));
        let d = |a: &[f64], b: &[f64]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y) / (2. * h))
                .collect::<Vec<f64>>()
        };
        (d(&p.0, &m.0), d(&p.1, &m.1), d(&p.2, &m.2))
    };
    let (coarse, fine) = (difference(2e-5), difference(1e-5));
    for (label, exact, c, f) in [
        ("dE/dn", &local, &coarse.0, &fine.0),
        ("dE/dtau", &tau_out, &coarse.1, &fine.1),
        ("flux", &flux, &coarse.2, &fine.2),
    ] {
        let scale = exact.iter().fold(0f64, |m, v| m.max(v.abs()));
        let error = (0..exact.len())
            .map(|k| (exact[k] - (4. * f[k] - c[k]) / 3.).abs())
            .fold(0f64, f64::max);
        assert!(scale > 0., "{label}: the kernel vanishes");
        assert!(error < 1e-6 * scale, "{label}: {error:e} of {scale:e}");
    }
}
